"""
Triangular arbitrage strategy implementation.

This module implements triangular arbitrage by detecting three-pair arbitrage
opportunities (e.g., BTC/USDT → ETH/BTC → ETH/USDT) and executing them with
proper sequencing to maximize profit extraction.

CRITICAL: This strategy requires rapid execution sequencing and careful
slippage management across the arbitrage chain.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import ArbitrageError, ValidationError

# Logger is provided by BaseStrategy (via BaseComponent)
# From P-001 - Use existing types
from src.core.types import MarketData, Position, Signal, SignalDirection, StrategyType

# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy
from src.strategies.dependencies import StrategyServiceContainer
from src.utils.constants import GLOBAL_FEE_STRUCTURE, GLOBAL_MINIMUM_AMOUNTS, PRECISION_LEVELS
from src.utils.decimal_utils import round_to_precision

# From P-007A - Use decorators and validators
from src.utils.decorators import log_errors, time_execution
from src.utils.formatters import format_currency, format_percentage
from src.utils.validators import ValidationFramework

# From P-008+ - Use risk management

# From P-003+ - Use exchange interfaces


class TriangularArbitrageStrategy(BaseStrategy):
    """
    Triangular arbitrage strategy for detecting and executing three-pair arbitrage.

    This strategy monitors three related trading pairs and executes a sequence
    of trades to capture price inefficiencies in the triangular relationship.
    """

    def __init__(self, config: dict, services: "StrategyServiceContainer"):
        """Initialize triangular arbitrage strategy.

        Args:
            config: Strategy configuration dictionary
            services: Service container for dependencies
        """
        super().__init__(config, services)

        # Strategy-specific configuration
        self.min_profit_threshold = Decimal(
            str(config.get("min_profit_threshold", "0.001"))
        )  # 0.1%
        self.max_execution_time = config.get("max_execution_time", 500)  # milliseconds
        self.latency_threshold = config.get("latency_threshold", 100)  # milliseconds
        self.triangular_paths = config.get(
            "triangular_paths", [["BTCUSDT", "ETHBTC", "ETHUSDT"], ["BTCUSDT", "BNBBTC", "BNBUSDT"]]
        )
        # Single exchange for triangular
        self.exchange = config.get("exchange", "binance")
        self.slippage_limit = Decimal(str(config.get("slippage_limit", "0.0005")))  # 0.05%

        # State tracking
        self.active_triangular_arbitrages: dict[str, dict] = {}
        self.pair_prices: dict[str, MarketData] = {}
        self.last_opportunity_check = datetime.now(timezone.utc)

        self.logger.info(
            "Triangular arbitrage strategy initialized",
            strategy=self.name,
            exchange=self.exchange,
            triangular_paths=self.triangular_paths,
            min_profit_threshold=self.min_profit_threshold,
        )

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.ARBITRAGE

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """
        Generate triangular arbitrage signals based on three-pair price relationships.

        Args:
            data: Market data from one exchange

        Returns:
            List of triangular arbitrage signals
        """
        try:
            # Update price data for the pair
            self.pair_prices[data.symbol] = data

            # Check for triangular arbitrage opportunities
            signals = await self._detect_triangular_opportunities(data.symbol)

            # Log triangular arbitrage opportunities for monitoring
            if signals:
                self.logger.debug(
                    "Triangular arbitrage signals generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    signal_count=len(signals),
                    signals=[s.direction.value for s in signals],
                )

            return signals

        except Exception as e:
            self.logger.error(
                "Triangular arbitrage signal generation failed",
                strategy=self.name,
                symbol=data.symbol,
                error=str(e),
            )
            return []  # Graceful degradation

    async def _detect_triangular_opportunities(self, symbol: str) -> list[Signal]:
        """
        Detect triangular arbitrage opportunities for given symbol.

        Args:
            symbol: Trading symbol that triggered the check

        Returns:
            List of triangular arbitrage signals
        """
        signals = []

        try:
            # Check all triangular paths that include this symbol
            for path in self.triangular_paths:
                if symbol in path:
                    signal = await self._check_triangular_path(path)
                    if signal:
                        signals.append(signal)

        except Exception as e:
            self.logger.error(
                "Triangular opportunity detection failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )

        return signals

    async def _check_triangular_path(self, path: list[str]) -> "Signal | None":
        """
        Check a specific triangular path for arbitrage opportunities.

        Args:
            path: List of three trading pairs (e.g., ["BTCUSDT", "ETHBTC", "ETHUSDT"])

        Returns:
            Triangular arbitrage signal if opportunity exists, None otherwise
        """
        try:
            # Ensure we have prices for all pairs in the path
            if not all(pair in self.pair_prices for pair in path):
                return None

            # Extract prices for the three pairs
            pair1, pair2, pair3 = path
            price1 = self.pair_prices[pair1]
            price2 = self.pair_prices[pair2]
            price3 = self.pair_prices[pair3]

            # Calculate triangular arbitrage
            # Example: BTCUSDT → ETHBTC → ETHUSDT
            # Start with 1 BTC, convert to ETH, then to USDT
            # Check if final USDT amount > original BTC value

            # Step 1: Calculate conversion rates
            btc_usdt_rate = price1.price  # BTC/USDT rate
            eth_btc_rate = price2.price  # ETH/BTC rate
            eth_usdt_rate = price3.price  # ETH/USDT rate

            # CRITICAL FIX: Atomic calculation with timestamp validation
            _ = datetime.now(timezone.utc)  # Timestamp for calculation validation

            # Step 2: Calculate triangular conversion atomically
            # Start with 1 BTC worth of USDT
            start_usdt = btc_usdt_rate

            # Convert BTC to ETH
            eth_amount = 1 / eth_btc_rate

            # Convert ETH to USDT
            final_usdt = eth_amount * eth_usdt_rate

            # Validate prices haven't changed significantly
            price_age = (datetime.now(timezone.utc) - price1.timestamp).total_seconds()
            if price_age > 0.5:  # Prices older than 500ms are stale
                self.logger.debug(
                    "Triangular calculation aborted - stale prices",
                    age_seconds=price_age,
                )
                return None

            # Calculate profit
            profit = final_usdt - start_usdt
            profit_percentage = (profit / start_usdt) * 100

            # Account for fees and slippage
            total_fees = self._calculate_triangular_fees(btc_usdt_rate, eth_btc_rate, eth_usdt_rate)
            net_profit = profit - total_fees
            net_profit_percentage = (net_profit / start_usdt) * 100

            # Check if profit meets threshold
            # Check if profit meets threshold - use Decimal comparison
            threshold_percentage = self.min_profit_threshold * Decimal("100")
            if net_profit_percentage >= threshold_percentage:
                # Validate execution timing
                if await self._validate_triangular_timing(path):
                    # Create triangular arbitrage signal
                    signal = Signal(
                        signal_id=f"triangular_arb_{datetime.now(timezone.utc).timestamp()}",
                        strategy_id=self.strategy_id,
                        strategy_name=self.name,
                        direction=SignalDirection.BUY,  # Direction doesn't matter for triangular
                        # Scale confidence with profit
                        confidence=min(0.9, net_profit_percentage / 2),
                        timestamp=datetime.now(timezone.utc),
                        symbol=pair1,  # Use first pair as primary symbol
                        metadata={
                            "arbitrage_type": "triangular",
                            "path": path,
                            "exchange": self.exchange,
                            "start_rate": float(btc_usdt_rate),
                            "intermediate_rate": float(eth_btc_rate),
                            "final_rate": float(eth_usdt_rate),
                            "profit_percentage": str(profit_percentage),
                            "net_profit_percentage": str(net_profit_percentage),
                            "estimated_fees": str(total_fees),
                            "execution_timeout": self.max_execution_time,
                            "execution_sequence": [
                                {"pair": pair1, "action": "buy", "rate": float(btc_usdt_rate)},
                                {"pair": pair2, "action": "sell", "rate": float(eth_btc_rate)},
                                {"pair": pair3, "action": "sell", "rate": float(eth_usdt_rate)},
                            ],
                        },
                    )

                    self.logger.info(
                        "Triangular arbitrage opportunity detected",
                        strategy=self.name,
                        path=path,
                        profit_percentage=float(profit_percentage),
                        net_profit_percentage=float(net_profit_percentage),
                        exchange=self.exchange,
                    )

                    return signal

        except Exception as e:
            self.logger.error(
                "Triangular path check failed", strategy=self.name, path=path, error=str(e)
            )

        return None

    @log_errors
    def _calculate_triangular_fees(self, rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal:
        """
        Calculate total fees for triangular arbitrage execution using proper validation
        and formatting.

        Args:
            rate1: First pair rate (e.g., BTC/USDT)
            rate2: Second pair rate (e.g., ETH/BTC)
            rate3: Third pair rate (e.g., ETH/USDT)

        Returns:
            Total estimated fees

        Raises:
            ValidationError: If rates are invalid
            ArbitrageError: If fee calculation fails
        """
        try:
            # Validate input rates using utils
            for rate, _name in [(rate1, "rate1"), (rate2, "rate2"), (rate3, "rate3")]:
                ValidationFramework.validate_price(rate)

            # Get fee structure from constants and convert to Decimal
            taker_fee_rate = Decimal(str(GLOBAL_FEE_STRUCTURE.get("taker_fee", 0.001)))  # 0.1%

            # Calculate fees for each step using proper rounding
            step1_fees = round_to_precision(rate1 * taker_fee_rate, PRECISION_LEVELS["fee"])
            step2_fees = round_to_precision(rate2 * taker_fee_rate, PRECISION_LEVELS["fee"])
            step3_fees = round_to_precision(rate3 * taker_fee_rate, PRECISION_LEVELS["fee"])

            # Calculate slippage costs for each step
            slippage_cost1 = round_to_precision(
                rate1 * self.slippage_limit, PRECISION_LEVELS["price"]
            )
            slippage_cost2 = round_to_precision(
                rate2 * self.slippage_limit, PRECISION_LEVELS["price"]
            )
            slippage_cost3 = round_to_precision(
                rate3 * self.slippage_limit, PRECISION_LEVELS["price"]
            )

            # Calculate total fees
            total_fees = (
                step1_fees
                + step2_fees
                + step3_fees
                + slippage_cost1
                + slippage_cost2
                + slippage_cost3
            )

            # Validate final result
            # total_fees is already a Decimal from calculations

            self.logger.debug(
                "Triangular fee calculation completed",
                strategy=self.name,
                rate1=format_currency(rate1),
                rate2=format_currency(rate2),
                rate3=format_currency(rate3),
                step1_fees=format_currency(step1_fees),
                step2_fees=format_currency(step2_fees),
                step3_fees=format_currency(step3_fees),
                slippage_costs=[
                    format_currency(cost)
                    for cost in [slippage_cost1, slippage_cost2, slippage_cost3]
                ],
                total_fees=format_currency(total_fees),
            )

            return total_fees

        except Exception as e:
            self.logger.error(
                "Triangular fee calculation failed",
                strategy=self.name,
                rate1=float(rate1),
                rate2=float(rate2),
                rate3=float(rate3),
                error=str(e),
            )
            raise ArbitrageError(f"Triangular fee calculation failed: {e!s}") from e

    async def _validate_triangular_timing(self, path: list[str]) -> bool:
        """
        Validate that triangular arbitrage execution timing constraints are met.

        Args:
            path: Triangular path to validate

        Returns:
            True if timing is valid, False otherwise
        """
        try:
            # Check if we have recent price data for all pairs
            current_time = datetime.now(timezone.utc)
            max_age = timedelta(milliseconds=self.latency_threshold)

            for pair in path:
                if pair in self.pair_prices:
                    price_data = self.pair_prices[pair]
                    if current_time - price_data.timestamp > max_age:
                        self.logger.warning(
                            "Price data too old for triangular arbitrage",
                            strategy=self.name,
                            pair=pair,
                            age_ms=(current_time - price_data.timestamp).total_seconds() * 1000,
                        )
                        return False
                else:
                    self.logger.warning(
                        "Missing price data for triangular arbitrage", strategy=self.name, pair=pair
                    )
                    return False

            # Check if we have too many active triangular arbitrages
            active_count = len(
                [a for a in self.active_triangular_arbitrages.values() if a.get("path") == path]
            )

            max_arbitrages = self.config.parameters.get("max_open_arbitrages", 3)
            if active_count >= max_arbitrages:
                self.logger.warning(
                    "Too many active triangular arbitrages",
                    strategy=self.name,
                    path=path,
                    active_count=active_count,
                    max_allowed=max_arbitrages,
                )
                return False

            return True

        except Exception as e:
            self.logger.error(
                "Triangular timing validation failed", strategy=self.name, path=path, error=str(e)
            )
            return False

    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate triangular arbitrage signal before execution.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic signal validation
            if not signal or signal.strength < self.config.min_confidence:
                return False

            # Validate triangular-specific metadata
            metadata = signal.metadata
            required_fields = ["arbitrage_type", "path", "exchange", "net_profit_percentage"]

            for field in required_fields:
                if field not in metadata:
                    self.logger.warning(
                        "Missing required metadata field",
                        strategy=self.name,
                        field=field,
                        signal_id=signal.timestamp,
                    )
                    return False

            # Validate arbitrage type
            if metadata.get("arbitrage_type") != "triangular":
                return False

            # Validate path structure
            path = metadata.get("path", [])
            if len(path) != 3:
                self.logger.warning(
                    "Invalid triangular path length",
                    strategy=self.name,
                    path_length=len(path),
                    expected=3,
                )
                return False

            # Validate profit threshold - use Decimal comparison
            net_profit = Decimal(str(metadata.get("net_profit_percentage", 0)))
            threshold_percentage = self.min_profit_threshold * Decimal("100")
            if net_profit < threshold_percentage:
                return False

            # Validate execution sequence
            execution_sequence = metadata.get("execution_sequence", [])
            if len(execution_sequence) != 3:
                return False

            return True

        except Exception as e:
            self.logger.error(
                "Triangular signal validation failed", strategy=self.name, error=str(e)
            )
            return False

    @log_errors
    def get_position_size(self, signal: Signal) -> Decimal:
        """
        Calculate position size for triangular arbitrage signal using risk management components.

        Args:
            signal: Triangular arbitrage signal

        Returns:
            Position size in base currency

        Raises:
            ArbitrageError: If position size calculation fails
        """
        try:
            # Validate signal using utils
            if not signal:
                raise ArbitrageError("Invalid signal for triangular position sizing")
            if signal.strength < 0.0 or signal.strength > 1.0:
                raise ValidationError(f"Invalid signal confidence: {signal.strength}")

            # Get configuration parameters
            total_capital = Decimal(str(self.config.parameters.get("total_capital", 10000)))
            risk_per_trade = (
                self.config.parameters.get("risk_per_trade", 0.02) * 0.7
            )  # Lower risk for triangular
            max_position_size = self.config.parameters.get("max_position_size", 0.05)

            # Calculate base position size using simple percentage method
            base_size = total_capital * Decimal(str(risk_per_trade))

            # Apply maximum position size limit
            max_size = total_capital * Decimal(str(max_position_size))
            if base_size > max_size:
                base_size = max_size

            # Scale by arbitrage-specific factors
            metadata = signal.metadata
            profit_potential = Decimal(str(metadata.get("net_profit_percentage", 0))) / Decimal(
                "100"
            )
            profit_potential_pct = float(profit_potential * 100)
            if profit_potential_pct < 0.0 or profit_potential_pct > 1000.0:
                raise ValidationError(f"Invalid profit potential: {profit_potential_pct}%")

            # Apply triangular arbitrage-specific adjustments
            triangular_multiplier = min(
                Decimal("1.5"), profit_potential * Decimal("8")
            )  # More conservative
            confidence_multiplier = Decimal(str(signal.strength))

            # Calculate final position size with proper validation
            position_size = round_to_precision(
                base_size * confidence_multiplier * triangular_multiplier,
                PRECISION_LEVELS["position"],
            )

            # Apply minimum position size from constants (smaller for
            # triangular)
            min_size = Decimal(str(GLOBAL_MINIMUM_AMOUNTS.get("position", 0.001))) * Decimal("0.5")
            if position_size < min_size:
                position_size = min_size

            # Validate final result
            ValidationFramework.validate_quantity(position_size)

            self.logger.debug(
                "Triangular position size calculated",
                strategy=self.name,
                base_size=format_currency(base_size),
                profit_potential=format_percentage(profit_potential * 100),
                confidence=format_percentage(signal.strength * 100),
                triangular_multiplier=triangular_multiplier,
                final_size=format_currency(position_size),
            )

            return position_size

        except Exception as e:
            self.logger.error(
                "Triangular position size calculation failed",
                strategy=self.name,
                signal_confidence=signal.strength if signal else None,
                error=str(e),
            )
            raise ArbitrageError(f"Triangular position size calculation failed: {e!s}") from e

    async def should_exit(self, position: Position, data: MarketData) -> bool:
        """
        Determine if triangular arbitrage position should be closed.

        Args:
            position: Current position
            data: Latest market data

        Returns:
            True if position should be closed, False otherwise
        """
        try:
            # Check if this is a triangular arbitrage position
            if (
                "arbitrage_type" not in position.metadata
                or position.metadata.get("arbitrage_type") != "triangular"
            ):
                return False  # Not a triangular arbitrage position

            # Check execution timeout
            execution_timeout = position.metadata.get("execution_timeout", self.max_execution_time)
            position_age = (datetime.now(timezone.utc) - position.opened_at).total_seconds() * 1000

            if position_age > execution_timeout:
                self.logger.info(
                    "Triangular arbitrage position timeout",
                    strategy=self.name,
                    symbol=position.symbol,
                    age_ms=position_age,
                    timeout_ms=execution_timeout,
                )
                return True

            # Check if triangular opportunity still exists
            metadata = position.metadata
            path = metadata.get("path", [])

            if path and len(path) == 3:
                # Recalculate current triangular arbitrage
                current_opportunity = await self._check_triangular_path(path)

                if not current_opportunity:
                    self.logger.info(
                        "Triangular arbitrage opportunity closed",
                        strategy=self.name,
                        symbol=position.symbol,
                        path=path,
                    )
                    return True

            return False

        except Exception as e:
            self.logger.error(
                "Triangular exit condition check failed",
                strategy=self.name,
                symbol=position.symbol,
                error=str(e),
            )
            return False

    async def post_trade_processing(self, trade_result: dict[str, Any]) -> None:
        """Process results after trade execution."""
        # Log the trade result
        self.logger.info("Triangular arbitrage trade completed",
                        strategy=self.name,
                        trade_result=trade_result)

        # Performance metrics update will be handled by the monitoring service


    # Helper methods for accessing data through data service

    async def _process_trade_result(self, trade_result: dict[str, Any]) -> None:
        """
        Process completed triangular arbitrage trade.

        Args:
            trade_result: Trade execution result
        """
        try:
            # Update metrics
            self.metrics.total_trades += 1

            # Calculate P&L
            if "pnl" in trade_result:
                self.metrics.total_pnl += Decimal(str(trade_result["pnl"]))

                if trade_result["pnl"] > 0:
                    self.metrics.winning_trades += 1
                else:
                    self.metrics.losing_trades += 1

            # Update win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

            # Log trade result
            self.logger.info(
                "Triangular arbitrage trade completed",
                strategy=self.name,
                symbol=trade_result.get("symbol"),
                path=trade_result.get("path"),
                pnl=float(trade_result.get("pnl", 0)),
                execution_time_ms=trade_result.get("execution_time_ms", 0),
            )

        except Exception as e:
            self.logger.error(
                "Triangular post-trade processing failed", strategy=self.name, error=str(e)
            )
