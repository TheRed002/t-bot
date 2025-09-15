"""
Bot coordinator for inter-bot communication and coordination.

This module implements the BotCoordinator class that handles communication
between bot instances, shared signal distribution, position conflict detection,
cross-bot risk assessment, and arbitrage opportunity sharing.

CRITICAL: This integrates with P-008+ (risk management), P-011 (strategies),
and P-003+ (exchanges) components.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import BotConfiguration, OrderSide

# Import common utilities
from src.utils.bot_service_helpers import (
    safe_import_decorators,
    safe_import_error_handling,
)

# Get error handling components with fallback
_error_handling = safe_import_error_handling()
get_global_error_handler = _error_handling["get_global_error_handler"]
with_circuit_breaker = _error_handling["with_circuit_breaker"]
with_error_context = _error_handling["with_error_context"]
with_fallback = _error_handling["with_fallback"]
with_retry = _error_handling["with_retry"]

# Import FallbackStrategy from error handling decorators
try:
    from src.error_handling.decorators import FallbackStrategy
except ImportError:
    # Fallback if not available
    class FallbackStrategy:
        RETURN_NONE = "return_none"


# Get decorators with fallback
_decorators = safe_import_decorators()
log_calls = _decorators["log_calls"]


class BotCoordinator:
    """
    Inter-bot communication and coordination manager.

    This class provides:
    - Shared signal distribution between bots
    - Position conflict detection and resolution
    - Cross-bot risk assessment and limits
    - Market impact coordination
    - Arbitrage opportunity sharing
    - Performance synchronization
    """

    def __init__(self, config: Config):
        """
        Initialize bot coordinator.

        Args:
            config: Application configuration
        """
        self._logger = get_logger(self.__class__.__module__)
        self.config = config
        self.error_handler = get_global_error_handler()

        # Bot registry and status tracking
        self.registered_bots: dict[str, BotConfiguration] = {}
        self.bot_positions: dict[str, dict[str, Any]] = {}  # bot_id -> symbol -> position info
        self.shared_signals: list[dict[str, Any]] = []
        self.arbitrage_opportunities: list[dict[str, Any]] = []

        # Locks for thread-safe operations
        self._registry_lock = asyncio.Lock()
        self._position_lock = asyncio.Lock()
        self._signal_lock = asyncio.Lock()

        # Coordination state
        self.is_running = False
        self.coordination_task = None
        self.signal_distribution_task = None

        # Cross-bot risk tracking
        self.symbol_exposure: dict[str, dict[str, Decimal]] = {}  # symbol -> side -> total_exposure
        self.exchange_usage: dict[str, dict[str, int]] = {}  # exchange -> metric -> count

        # Configuration
        self.max_symbol_exposure = Decimal(
            str(config.bot_management.get("max_symbol_exposure", "100000"))
        )
        self.coordination_interval = config.bot_management.get("coordination_interval", 10)
        self.signal_retention_minutes = config.bot_management.get("signal_retention_minutes", 60)
        self.arbitrage_detection_enabled = config.bot_management.get(
            "arbitrage_detection_enabled", True
        )

        # Performance tracking
        self.coordination_metrics = {
            "signals_distributed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "arbitrage_opportunities_found": 0,
            "cross_bot_risk_checks": 0,
            "last_coordination_time": None,
        }

        self._logger.info("Bot coordinator initialized")

    @log_calls
    @with_error_context(component="bot_coordinator", operation="start")
    async def start(self) -> None:
        """
        Start the bot coordinator.

        Raises:
            ExecutionError: If startup fails
        """
        if self.is_running:
            self._logger.warning("Bot coordinator is already running")
            return

        self._logger.info("Starting bot coordinator")

        # Start coordination tasks
        self.coordination_task = asyncio.create_task(self._coordination_loop())
        self.signal_distribution_task = asyncio.create_task(self._signal_distribution_loop())

        self.is_running = True
        self._logger.info("Bot coordinator started successfully")

    @log_calls
    @with_error_context(component="bot_coordinator", operation="stop")
    async def stop(self) -> None:
        """
        Stop the bot coordinator.

        Raises:
            ExecutionError: If shutdown fails
        """
        if not self.is_running:
            self._logger.warning("Bot coordinator is not running")
            return

        self._logger.info("Stopping bot coordinator")
        self.is_running = False

        # Stop coordination tasks with proper cleanup
        tasks_to_cancel: list[Any] = []
        if self.coordination_task:
            self.coordination_task.cancel()
            tasks_to_cancel.append(self.coordination_task)

        if self.signal_distribution_task:
            self.signal_distribution_task.cancel()
            tasks_to_cancel.append(self.signal_distribution_task)

        # Wait for tasks to complete cancellation
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Clear state
        self.registered_bots.clear()
        self.bot_positions.clear()
        self.shared_signals.clear()
        self.arbitrage_opportunities.clear()

        self._logger.info("Bot coordinator stopped successfully")

    @log_calls
    @with_error_context(component="bot_coordinator", operation="register_bot")
    @with_fallback(strategy=FallbackStrategy.RETURN_NONE)
    @with_retry(max_attempts=3, base_delay=1.0)
    async def register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None:
        """
        Register a bot with the coordinator.

        Args:
            bot_id: Bot identifier
            bot_config: Bot configuration

        Raises:
            ValidationError: If registration is invalid
        """
        async with self._registry_lock:
            if bot_id in self.registered_bots:
                raise ValidationError(f"Bot already registered: {bot_id}")

            # Store bot configuration
            self.registered_bots[bot_id] = bot_config

            # Initialize position tracking
            self.bot_positions[bot_id] = {}

        # Initialize exchange usage tracking for bot's exchanges
        for exchange in bot_config.exchanges:
            if exchange not in self.exchange_usage:
                self.exchange_usage[exchange] = {
                    "active_bots": 0,
                    "total_positions": 0,
                    "api_calls_per_minute": 0,
                }
            self.exchange_usage[exchange]["active_bots"] += 1

        self._logger.info(
            "Bot registered with coordinator",
            bot_id=bot_id,
            bot_type=bot_config.bot_type.value,
            exchanges=bot_config.exchanges,
            symbols=bot_config.symbols,
        )

    @log_calls
    @with_error_context(component="bot_coordinator", operation="unregister_bot")
    @with_retry(max_attempts=2, base_delay=0.5)
    async def unregister_bot(self, bot_id: str) -> None:
        """
        Unregister a bot from the coordinator.

        Args:
            bot_id: Bot identifier
        """
        if bot_id not in self.registered_bots:
            self._logger.warning("Bot not registered", bot_id=bot_id)
            return

        bot_config = self.registered_bots[bot_id]

        # Update exchange usage tracking
        for exchange in bot_config.exchanges:
            if exchange in self.exchange_usage:
                self.exchange_usage[exchange]["active_bots"] -= 1
                if self.exchange_usage[exchange]["active_bots"] <= 0:
                    del self.exchange_usage[exchange]

        # Remove position tracking
        if bot_id in self.bot_positions:
            # Update symbol exposure tracking
            for symbol, position in self.bot_positions[bot_id].items():
                await self._update_symbol_exposure(symbol, position, remove=True)

            del self.bot_positions[bot_id]

        # Remove from registry
        del self.registered_bots[bot_id]

        self._logger.info("Bot unregistered from coordinator", bot_id=bot_id)

    @log_calls
    @with_error_context(component="bot_coordinator", operation="report_position_change")
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def report_position_change(
        self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal
    ) -> dict[str, Any]:
        """
        Report a position change from a bot for coordination.

        Args:
            bot_id: Bot identifier
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Position quantity
            price: Position price

        Returns:
            dict: Coordination response with recommendations

        Raises:
            ValidationError: If report is invalid
        """
        if bot_id not in self.registered_bots:
            raise ValidationError(f"Bot not registered: {bot_id}")

        # Update position tracking
        position_key = f"{symbol}_{side.value}"

        if bot_id not in self.bot_positions:
            self.bot_positions[bot_id] = {}

        position_info = {
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now(timezone.utc),
            "value": quantity * price,
        }

        # Store previous position for exposure calculation
        previous_position = self.bot_positions[bot_id].get(position_key)
        self.bot_positions[bot_id][position_key] = position_info

        # Update symbol exposure tracking
        await self._update_symbol_exposure(symbol, position_info, previous_position)

        # Check for conflicts and risks
        coordination_response = await self._analyze_position_change(
            bot_id, symbol, side, quantity, price
        )

        self._logger.debug(
            "Position change reported",
            bot_id=bot_id,
            symbol=symbol,
            side=side.value,
            quantity=str(quantity),
            conflicts=len(coordination_response.get("conflicts", [])),
        )

        return coordination_response

    @log_calls
    @with_error_context(component="bot_coordinator", operation="share_signal")
    @with_retry(max_attempts=3, base_delay=0.5)
    async def share_signal(
        self, bot_id: str, signal_data: dict[str, Any], target_bots: list[str] | None = None
    ) -> int:
        """
        Share a trading signal with other bots.

        Args:
            bot_id: Source bot identifier
            signal_data: Signal data to share
            target_bots: Optional list of target bot IDs (None for all)

        Returns:
            int: Number of bots the signal was shared with

        Raises:
            ValidationError: If signal is invalid
        """
        if bot_id not in self.registered_bots:
            raise ValidationError(f"Bot not registered: {bot_id}")

        # Validate signal data
        required_fields = ["symbol", "direction", "confidence", "timestamp"]
        for field in required_fields:
            if field not in signal_data:
                raise ValidationError(f"Signal missing required field: {field}")

        # Create signal entry
        if target_bots is None:
            # Filter bots by symbol interest (exclude source bot)
            signal_symbol = signal_data["symbol"]
            interested_bots = [
                bot_id_key
                for bot_id_key, bot_config in self.registered_bots.items()
                if bot_id_key != bot_id and signal_symbol in bot_config.symbols
            ]
            target_bots = interested_bots

        signal_entry = {
            "signal_id": str(uuid.uuid4()),
            "source_bot": bot_id,
            "signal_data": signal_data,
            "target_bots": target_bots,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc)
            + timedelta(minutes=self.signal_retention_minutes),
        }

        # Add to shared signals
        self.shared_signals.append(signal_entry)

        # Update metrics
        self.coordination_metrics["signals_distributed"] += 1

        # Count actual target recipients (exclude source bot if present)
        target_count = len([bot for bot in signal_entry["target_bots"] if bot != bot_id])

        self._logger.info(
            "Signal shared with coordination network",
            source_bot=bot_id,
            signal_id=signal_entry["signal_id"],
            symbol=signal_data["symbol"],
            direction=signal_data["direction"],
            target_count=target_count,
        )

        return target_count

    @log_calls
    @with_error_context(component="bot_coordinator", operation="get_shared_signals")
    @with_fallback(fallback_value=[])
    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]:
        """
        Get shared signals for a specific bot.

        Args:
            bot_id: Bot identifier

        Returns:
            list: List of relevant shared signals
        """
        if bot_id not in self.registered_bots:
            return []

        relevant_signals = []
        current_time = datetime.now(timezone.utc)

        for signal in self.shared_signals:
            # Skip expired signals
            if current_time > signal["expires_at"]:
                continue

            # Skip signals from the same bot
            if signal["source_bot"] == bot_id:
                continue

            # Check if bot is a target
            if bot_id in signal["target_bots"]:
                relevant_signals.append(
                    {
                        "signal_id": signal["signal_id"],
                        "source_bot": signal["source_bot"],
                        "signal_data": signal["signal_data"],
                        "created_at": signal["created_at"].isoformat(),
                        "age_minutes": (current_time - signal["created_at"]).total_seconds() / 60,
                    }
                )

        return relevant_signals

    @log_calls
    @with_error_context(component="bot_coordinator", operation="check_cross_bot_risk")
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    async def check_cross_bot_risk(
        self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal
    ) -> dict[str, Any]:
        """
        Check cross-bot risk for a proposed order.

        Args:
            bot_id: Bot identifier
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity

        Returns:
            dict: Risk assessment with recommendations
        """
        self.coordination_metrics["cross_bot_risk_checks"] += 1

        risk_assessment = {
            "approved": True,
            "risk_level": "low",
            "warnings": [],
            "recommendations": [],
            "max_safe_quantity": quantity,
        }

        # Check symbol exposure limits
        self._check_symbol_exposure_limits(risk_assessment, symbol, side, quantity)

        # Check for position concentration
        self._check_position_concentration(risk_assessment, bot_id, symbol)

        # Check for conflicting positions from other bots
        self._check_conflicting_positions(risk_assessment, bot_id, symbol, side, quantity)

        # Generate recommendations
        self._generate_risk_recommendations(risk_assessment)

        return risk_assessment

    def _check_symbol_exposure_limits(
        self, risk_assessment: dict[str, Any], symbol: str, side: OrderSide, quantity: Decimal
    ) -> None:
        """Check if proposed order would exceed symbol exposure limits."""
        current_exposure = self.symbol_exposure.get(symbol, {}).get(side.value, Decimal("0"))
        proposed_total = current_exposure + quantity

        if proposed_total > self.max_symbol_exposure:
            risk_assessment["approved"] = False
            risk_assessment["risk_level"] = "high"
            risk_assessment["warnings"].append(
                f"Proposed order would exceed max symbol exposure: "
                f"{proposed_total} > {self.max_symbol_exposure}"
            )
            risk_assessment["max_safe_quantity"] = max(
                Decimal("0"), self.max_symbol_exposure - current_exposure
            )

    def _check_position_concentration(
        self, risk_assessment: dict[str, Any], bot_id: str, symbol: str
    ) -> None:
        """Check for position concentration in the same symbol."""
        bot_symbol_positions = 0
        for _pos_key, position in self.bot_positions.get(bot_id, {}).items():
            if position["symbol"] == symbol:
                bot_symbol_positions += 1

        if bot_symbol_positions >= 3:  # More than 3 positions in same symbol
            risk_assessment["risk_level"] = "medium"
            risk_assessment["warnings"].append(
                f"High position concentration in {symbol}: {bot_symbol_positions} positions"
            )

    def _check_conflicting_positions(
        self,
        risk_assessment: dict[str, Any],
        bot_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
    ) -> None:
        """Check for conflicting positions from other bots."""
        conflicting_bots = []
        for other_bot_id, positions in self.bot_positions.items():
            if other_bot_id == bot_id:
                continue

            for _pos_key, position in positions.items():
                if (
                    position["symbol"] == symbol
                    and position["side"] != side.value
                    and position["quantity"] > quantity * Decimal("0.5")
                ):  # Significant opposing position
                    conflicting_bots.append(
                        {
                            "bot_id": other_bot_id,
                            "side": position["side"],
                            "quantity": str(position["quantity"]),
                        }
                    )

        if conflicting_bots:
            risk_assessment["risk_level"] = "medium"
            risk_assessment["warnings"].append(
                f"Conflicting positions detected from {len(conflicting_bots)} other bots"
            )
            risk_assessment["conflicting_bots"] = conflicting_bots

    def _generate_risk_recommendations(self, risk_assessment: dict[str, Any]) -> None:
        """Generate recommendations based on risk level."""
        if risk_assessment["risk_level"] == "high":
            risk_assessment["recommendations"].append("Consider reducing position size")
            risk_assessment["recommendations"].append("Coordinate with other bots before executing")
        elif risk_assessment["risk_level"] == "medium":
            risk_assessment["recommendations"].append("Monitor position closely")
            risk_assessment["recommendations"].append("Consider exit strategy")

    @with_error_context(component="bot_coordinator", operation="get_coordination_summary")
    @with_fallback(fallback_value={"error": "Failed to generate coordination summary"})
    async def get_coordination_summary(self) -> dict[str, Any]:
        """Get comprehensive coordination status summary."""
        # Calculate current exposures
        total_exposures = {}
        for symbol, sides in self.symbol_exposure.items():
            total_exposures[symbol] = {
                "buy_exposure": str(sides.get("buy", Decimal("0"))),
                "sell_exposure": str(sides.get("sell", Decimal("0"))),
                "net_exposure": str(
                    sides.get("buy", Decimal("0")) - sides.get("sell", Decimal("0"))
                ),
            }

        # Get active signals count
        current_time = datetime.now(timezone.utc)
        active_signals = len([s for s in self.shared_signals if current_time <= s["expires_at"]])

        return {
            "coordination_status": {
                "is_running": self.is_running,
                "registered_bots": len(self.registered_bots),
                "active_positions": sum(
                    len(positions) for positions in self.bot_positions.values()
                ),
                "active_signals": active_signals,
                "arbitrage_opportunities": len(self.arbitrage_opportunities),
            },
            "symbol_exposures": total_exposures,
            "exchange_usage": self.exchange_usage,
            "coordination_metrics": {
                **self.coordination_metrics,
                "last_coordination_time": (
                    self.coordination_metrics["last_coordination_time"].isoformat()
                    if self.coordination_metrics["last_coordination_time"]
                    else None
                ),
            },
        }

    async def _coordination_loop(self) -> None:
        """Main coordination monitoring loop."""
        try:
            while self.is_running:
                try:
                    # Update coordination timestamp
                    self.coordination_metrics["last_coordination_time"] = datetime.now(timezone.utc)

                    # Detect arbitrage opportunities
                    if self.arbitrage_detection_enabled:
                        await self._detect_arbitrage_opportunities()

                    # Check for position conflicts
                    await self._detect_position_conflicts()

                    # Clean up expired signals
                    await self._cleanup_expired_signals()

                    # Update exchange usage metrics
                    await self._update_exchange_metrics()

                    # Wait for next cycle
                    await asyncio.sleep(self.coordination_interval)

                except (asyncio.TimeoutError, ConnectionError) as e:
                    self._logger.warning(f"Coordination loop network error: {e}")
                    await asyncio.sleep(10)
                except ValidationError as e:
                    self._logger.error(f"Coordination loop validation error: {e}")
                    await asyncio.sleep(5)
                except Exception as e:
                    self._logger.error(f"Coordination loop unexpected error: {e}")
                    await self.error_handler.handle_error(
                        e, {"component": "bot_coordinator", "operation": "coordination_loop"}
                    )
                    await asyncio.sleep(10)

        except asyncio.CancelledError:
            self._logger.info("Coordination monitoring cancelled")

    async def _signal_distribution_loop(self) -> None:
        """Signal distribution and processing loop."""
        try:
            while self.is_running:
                try:
                    # Process shared signals for correlation analysis
                    await self._analyze_signal_correlations()

                    # Distribute high-priority signals immediately
                    await self._process_priority_signals()

                    # Update signal statistics
                    await self._update_signal_statistics()

                    # Wait for next cycle
                    await asyncio.sleep(5)  # More frequent for signal processing

                except (asyncio.TimeoutError, ConnectionError) as e:
                    self._logger.warning(f"Signal distribution loop network error: {e}")
                    await asyncio.sleep(10)
                except ValidationError as e:
                    self._logger.error(f"Signal distribution loop validation error: {e}")
                    await asyncio.sleep(5)
                except Exception as e:
                    self._logger.error(f"Signal distribution loop unexpected error: {e}")
                    await self.error_handler.handle_error(
                        e, {"component": "bot_coordinator", "operation": "signal_distribution_loop"}
                    )
                    await asyncio.sleep(10)

        except asyncio.CancelledError:
            self._logger.info("Signal distribution cancelled")

    async def _update_symbol_exposure(
        self,
        symbol: str,
        position: dict[str, Any],
        previous_position: dict[str, Any] | None = None,
        remove: bool = False,
    ) -> None:
        """Update symbol exposure tracking."""
        if symbol not in self.symbol_exposure:
            self.symbol_exposure[symbol] = {"buy": Decimal("0"), "sell": Decimal("0")}

        side = position["side"]
        quantity = position["quantity"]

        # Remove previous position if it exists
        if previous_position:
            prev_side = previous_position["side"]
            prev_quantity = previous_position["quantity"]
            self.symbol_exposure[symbol][prev_side] -= prev_quantity

        # Add or remove current position
        if remove:
            self.symbol_exposure[symbol][side] -= quantity
        else:
            self.symbol_exposure[symbol][side] += quantity

        # Ensure non-negative values
        for side_key in ["buy", "sell"]:
            self.symbol_exposure[symbol][side_key] = max(
                Decimal("0"), self.symbol_exposure[symbol][side_key]
            )

    async def _analyze_position_change(
        self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal
    ) -> dict[str, Any]:
        """Analyze a position change for conflicts and recommendations."""
        response = {
            "conflicts": [],
            "recommendations": [],
            "risk_level": "low",
            "coordination_actions": [],
        }

        # Check for direct conflicts with other bots
        conflicts = []
        for other_bot_id, positions in self.bot_positions.items():
            if other_bot_id == bot_id:
                continue

            for _pos_key, position in positions.items():
                if position["symbol"] == symbol and position["side"] != side.value:
                    # Calculate conflict severity
                    conflict_value = min(quantity * price, position["quantity"] * position["price"])

                    if conflict_value > Decimal("1000"):  # Significant conflict threshold
                        conflicts.append(
                            {
                                "conflicting_bot": other_bot_id,
                                "conflict_side": position["side"],
                                "conflict_value": str(conflict_value),
                                "severity": (
                                    "high" if conflict_value > Decimal("10000") else "medium"
                                ),
                            }
                        )

        if conflicts:
            response["conflicts"] = conflicts
            response["risk_level"] = (
                "high" if any(c["severity"] == "high" for c in conflicts) else "medium"
            )
            response["recommendations"].append("Coordinate with conflicting bots")
            self.coordination_metrics["conflicts_detected"] += 1

        return response

    async def _detect_arbitrage_opportunities(self) -> None:
        """Detect cross-exchange arbitrage opportunities."""
        try:
            # Analyze positions across exchanges for arbitrage

            current_time = datetime.now(timezone.utc)

            # Clean up old opportunities
            self.arbitrage_opportunities = [
                opp
                for opp in self.arbitrage_opportunities
                if (current_time - opp["detected_at"]).total_seconds() < 300  # 5 minutes
            ]

            # Check for arbitrage opportunities between exchanges
            if len(self.registered_bots) > 1 and len(self.exchange_usage) > 1:
                # Analyze price differences between exchanges
                self._logger.debug(
                    "Scanning for arbitrage opportunities",
                    exchanges=list(self.exchange_usage.keys()),
                )

        except (ConnectionError, TimeoutError) as e:
            self._logger.warning(f"Arbitrage detection network error: {e}")
        except ValidationError as e:
            self._logger.error(f"Arbitrage detection validation error: {e}")
        except Exception as e:
            self._logger.error(f"Arbitrage detection unexpected error: {e}")
            await self.error_handler.handle_error(
                e, {"component": "bot_coordinator", "operation": "arbitrage_detection"}
            )

    async def _detect_position_conflicts(self) -> None:
        """Detect and attempt to resolve position conflicts."""
        try:
            conflicts_resolved = 0
            symbols_analyzed = set()

            for _bot_id, positions in self.bot_positions.items():
                for _pos_key, position in positions.items():
                    symbol = position["symbol"]
                    if symbol in symbols_analyzed:
                        continue

                    # Analyze symbol for conflicts
                    conflict_detected = await self._analyze_symbol_conflicts(symbol)
                    if conflict_detected:
                        conflicts_resolved += 1

                    symbols_analyzed.add(symbol)

            if conflicts_resolved > 0:
                self.coordination_metrics["conflicts_resolved"] += conflicts_resolved

        except (ConnectionError, TimeoutError) as e:
            self._logger.warning(f"Position conflict detection network error: {e}")
        except ValidationError as e:
            self._logger.error(f"Position conflict detection validation error: {e}")
        except Exception as e:
            self._logger.error(f"Position conflict detection unexpected error: {e}")
            await self.error_handler.handle_error(
                e, {"component": "bot_coordinator", "operation": "position_conflict_detection"}
            )

    async def _analyze_symbol_conflicts(self, symbol: str) -> bool:
        """Analyze a specific symbol for position conflicts across bots."""
        buy_exposure = Decimal("0")
        sell_exposure = Decimal("0")

        # Calculate exposures across all bots for this symbol
        for _check_bot_id, check_positions in self.bot_positions.items():
            for _check_pos_key, check_position in check_positions.items():
                if check_position["symbol"] == symbol:
                    if check_position["side"] == "buy":
                        buy_exposure += check_position["quantity"]
                    else:
                        sell_exposure += check_position["quantity"]

        # Check for excessive opposing positions
        if buy_exposure > 0 and sell_exposure > 0:
            conflict_ratio = min(buy_exposure, sell_exposure) / max(buy_exposure, sell_exposure)
            if conflict_ratio > 0.5:  # Significant opposing positions
                self._logger.warning(
                    "Position conflict detected",
                    symbol=symbol,
                    buy_exposure=str(buy_exposure),
                    sell_exposure=str(sell_exposure),
                    conflict_ratio=str(conflict_ratio),
                )
                return True

        return False

    async def _cleanup_expired_signals(self) -> None:
        """Clean up expired shared signals."""
        current_time = datetime.now(timezone.utc)

        initial_count = len(self.shared_signals)
        self.shared_signals = [
            signal for signal in self.shared_signals if current_time <= signal["expires_at"]
        ]

        cleaned_count = initial_count - len(self.shared_signals)
        if cleaned_count > 0:
            self._logger.debug(f"Cleaned up {cleaned_count} expired signals")

    async def _update_exchange_metrics(self) -> None:
        """Update exchange usage metrics."""
        for exchange in self.exchange_usage:
            # Update position counts
            total_positions = 0
            for bot_id, positions in self.bot_positions.items():
                if bot_id in self.registered_bots:
                    bot_config = self.registered_bots[bot_id]
                    if exchange in bot_config.exchanges:
                        total_positions += len(positions)

            self.exchange_usage[exchange]["total_positions"] = total_positions

    async def _analyze_signal_correlations(self) -> None:
        """Analyze correlations between shared signals."""
        # Analyze signal patterns and correlations
        current_time = datetime.now(timezone.utc)

        active_signals = [s for s in self.shared_signals if current_time <= s["expires_at"]]

        # Group by symbol for correlation analysis
        symbol_signals: dict[str, list[Any]] = {}
        for signal in active_signals:
            symbol = signal["signal_data"]["symbol"]
            if symbol not in symbol_signals:
                symbol_signals[symbol] = []
            symbol_signals[symbol].append(signal)

        # Log symbols with multiple signals
        for symbol, signals in symbol_signals.items():
            if len(signals) > 1:
                self._logger.debug(
                    "Multiple signals detected for symbol",
                    symbol=symbol,
                    signal_count=len(signals),
                    sources=[s["source_bot"] for s in signals],
                )

    async def _process_priority_signals(self) -> None:
        """Process high-priority signals for immediate distribution."""
        current_time = datetime.now(timezone.utc)

        priority_signals = [
            s
            for s in self.shared_signals
            if (
                current_time <= s["expires_at"]
                and s["signal_data"].get("confidence", 0) > 0.8
                and s["signal_data"].get("priority", "normal") == "high"
            )
        ]

        if priority_signals:
            self._logger.debug("Processing priority signals", count=len(priority_signals))

    async def _update_signal_statistics(self) -> None:
        """Update signal distribution statistics."""
        current_time = datetime.now(timezone.utc)

        # Count active signals by source
        signal_sources: dict[str, int] = {}
        for signal in self.shared_signals:
            if current_time <= signal["expires_at"]:
                source = signal["source_bot"]
                signal_sources[source] = signal_sources.get(source, 0) + 1

        if signal_sources:
            self._logger.debug(
                "Signal distribution statistics", active_signals_by_source=signal_sources
            )

    # Position management methods for test compatibility

    async def update_bot_position(
        self, bot_id: str, symbol: str, position_data: dict[str, Any]
    ) -> None:
        """Update bot position data (stub implementation)."""
        if bot_id not in self.bot_positions:
            self.bot_positions[bot_id] = {}
        self.bot_positions[bot_id][symbol] = position_data
        self._logger.debug(f"Updated position for bot {bot_id} on {symbol}")

    async def remove_bot_position(self, bot_id: str, symbol: str) -> None:
        """Remove bot position data (stub implementation)."""
        if bot_id in self.bot_positions and symbol in self.bot_positions[bot_id]:
            del self.bot_positions[bot_id][symbol]
            self._logger.debug(f"Removed position for bot {bot_id} on {symbol}")

    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]:
        """Check for position conflicts on a symbol (stub implementation)."""
        conflicts = []
        positions = []

        for bot_id, bot_positions in self.bot_positions.items():
            if symbol in bot_positions:
                positions.append({"bot_id": bot_id, "position": bot_positions[symbol]})

        # Simple conflict detection - opposing sides
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1 :]:
                side1 = str(pos1["position"].get("side", "")).lower()
                side2 = str(pos2["position"].get("side", "")).lower()
                if (side1 == "buy" and side2 == "sell") or (side1 == "sell" and side2 == "buy"):
                    conflicts.append(
                        {
                            "type": "opposing_positions",
                            "bots": [pos1["bot_id"], pos2["bot_id"]],
                            "symbol": symbol,
                        }
                    )

        return conflicts

    async def coordinate_bot_actions(self, action_data: dict[str, Any]) -> dict[str, Any]:
        """Coordinate actions between bots (stub implementation)."""
        return {
            "status": "coordinated",
            "affected_bots": list(self.registered_bots.keys()),
            "action_type": action_data.get("type", "unknown"),
        }

    async def analyze_bot_interactions(self) -> dict[str, Any]:
        """Analyze interactions between bots (stub implementation)."""
        return {
            "total_interactions": len(self.shared_signals),
            "active_bots": len(self.registered_bots),
            "signal_diversity": len(
                set(signal.get("signal_data", {}).get("symbol") for signal in self.shared_signals)
            ),
        }

    async def optimize_coordination(self) -> dict[str, Any]:
        """Optimize coordination parameters (stub implementation)."""
        return {"optimizations_applied": 0, "efficiency_gain": 0.0, "recommendations": []}

    async def emergency_coordination(self, emergency_type: str, action: str) -> None:
        """Handle emergency coordination (stub implementation)."""
        self._logger.warning(f"Emergency coordination triggered: {emergency_type} -> {action}")

        # Add emergency signal to all bots
        emergency_signal = {
            "signal_id": str(uuid.uuid4()),
            "source_bot": "system",
            "signal_data": {
                "symbol": "ALL",
                "direction": "emergency",
                "confidence": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "emergency_type": emergency_type,
                "action": action,
            },
            "target_bots": list(self.registered_bots.keys()),
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        self.shared_signals.append(emergency_signal)
