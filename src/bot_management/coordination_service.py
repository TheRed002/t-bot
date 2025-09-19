"""
Bot Coordination Service Implementation.

This service handles coordination between multiple bot instances,
including signal sharing, position conflict detection, and risk coordination.
"""

from decimal import Decimal
from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotConfiguration, OrderSide

from .interfaces import IBotCoordinationService


class BotCoordinationService(BaseService, IBotCoordinationService):
    """
    Service for coordinating bot operations and interactions.

    This service manages inter-bot communication, signal sharing,
    and coordination of trading activities to prevent conflicts.
    """

    def __init__(
        self,
        name: str = "BotCoordinationService",
        config: dict[str, Any] = None,
    ):
        """Initialize bot coordination service."""
        super().__init__(name=name, config=config)
        self._logger = get_logger(__name__)
        self._registered_bots: dict[str, BotConfiguration] = {}
        self._shared_signals: dict[str, list[dict[str, Any]]] = {}
        self._position_data: dict[str, dict[str, Any]] = {}
        self._risk_assessments: dict[str, dict[str, Any]] = {}

    @property
    def registered_bots(self) -> dict[str, BotConfiguration]:
        """Get registered bots."""
        return self._registered_bots.copy()

    async def register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None:
        """
        Register a bot for coordination.

        Args:
            bot_id: Bot ID
            bot_config: Bot configuration
        """
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            if not bot_config:
                raise ValidationError("Bot configuration is required")

            self._registered_bots[bot_id] = bot_config
            self._shared_signals[bot_id] = []
            self._position_data[bot_id] = {}

            self._logger.info(f"Registered bot for coordination: {bot_id}")

        except Exception as e:
            self._logger.error(f"Failed to register bot {bot_id}: {e}")
            raise ServiceError(f"Failed to register bot: {e}") from e

    async def unregister_bot(self, bot_id: str) -> None:
        """
        Unregister a bot from coordination.

        Args:
            bot_id: Bot ID to unregister
        """
        try:
            if bot_id in self._registered_bots:
                del self._registered_bots[bot_id]

            if bot_id in self._shared_signals:
                del self._shared_signals[bot_id]

            if bot_id in self._position_data:
                del self._position_data[bot_id]

            if bot_id in self._risk_assessments:
                del self._risk_assessments[bot_id]

            self._logger.info(f"Unregistered bot from coordination: {bot_id}")

        except Exception as e:
            self._logger.error(f"Failed to unregister bot {bot_id}: {e}")
            raise ServiceError(f"Failed to unregister bot: {e}") from e

    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]:
        """
        Check for position conflicts across bots for a symbol.

        Args:
            symbol: Trading symbol to check

        Returns:
            List of detected conflicts
        """
        try:
            conflicts = []

            # Get all bots with positions in this symbol
            bots_with_positions = []
            for bot_id, positions in self._position_data.items():
                if symbol in positions:
                    bots_with_positions.append({
                        "bot_id": bot_id,
                        "position": positions[symbol],
                        "config": self._registered_bots.get(bot_id)
                    })

            # Check for conflicting positions
            for i, bot1 in enumerate(bots_with_positions):
                for bot2 in bots_with_positions[i + 1:]:
                    conflict = self._detect_position_conflict(bot1, bot2, symbol)
                    if conflict:
                        conflicts.append(conflict)

            return conflicts

        except Exception as e:
            self._logger.error(f"Failed to check position conflicts for {symbol}: {e}")
            raise ServiceError(f"Failed to check position conflicts: {e}") from e

    async def share_signal(
        self,
        bot_id: str,
        signal_type: str,
        symbol: str,
        direction: str,
        strength: float,
        metadata: dict[str, Any] = None,
    ) -> int:
        """
        Share a trading signal with other bots.

        Args:
            bot_id: Bot ID sharing the signal
            signal_type: Type of signal
            symbol: Trading symbol
            direction: Signal direction
            strength: Signal strength
            metadata: Additional metadata

        Returns:
            Number of bots the signal was shared with
        """
        try:
            if bot_id not in self._registered_bots:
                raise ValidationError(f"Bot {bot_id} not registered")

            signal = {
                "source_bot_id": bot_id,
                "signal_type": signal_type,
                "symbol": symbol,
                "direction": direction,
                "strength": strength,
                "metadata": metadata or {},
                "timestamp": self._get_current_timestamp(),
            }

            # Share with all other registered bots
            shared_count = 0
            for other_bot_id in self._registered_bots:
                if other_bot_id != bot_id:
                    self._shared_signals[other_bot_id].append(signal)
                    shared_count += 1

            # Clean up old signals
            await self.cleanup_expired_signals()

            self._logger.debug(f"Shared signal from {bot_id} with {shared_count} bots")
            return shared_count

        except Exception as e:
            self._logger.error(f"Failed to share signal from {bot_id}: {e}")
            raise ServiceError(f"Failed to share signal: {e}") from e

    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]:
        """
        Get shared signals for a bot.

        Args:
            bot_id: Bot ID

        Returns:
            List of shared signals
        """
        try:
            if bot_id not in self._registered_bots:
                raise ValidationError(f"Bot {bot_id} not registered")

            signals = self._shared_signals.get(bot_id, [])

            # Filter out expired signals
            from datetime import datetime, timezone, timedelta
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

            valid_signals = [
                signal for signal in signals
                if signal.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)) > cutoff_time
            ]

            self._shared_signals[bot_id] = valid_signals
            return valid_signals

        except Exception as e:
            self._logger.error(f"Failed to get shared signals for {bot_id}: {e}")
            raise ServiceError(f"Failed to get shared signals: {e}") from e

    async def check_cross_bot_risk(
        self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal
    ) -> dict[str, Any]:
        """
        Check cross-bot risk exposure.

        Args:
            bot_id: Bot ID
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity

        Returns:
            Risk assessment
        """
        try:
            if bot_id not in self._registered_bots:
                raise ValidationError(f"Bot {bot_id} not registered")

            risk_assessment = {
                "bot_id": bot_id,
                "symbol": symbol,
                "side": side.value if hasattr(side, 'value') else str(side),
                "quantity": str(quantity),
                "risk_level": "low",
                "warnings": [],
                "recommendations": [],
                "total_exposure": Decimal("0"),
                "correlation_risk": "low",
            }

            # Calculate total exposure across all bots
            total_exposure = self._calculate_total_exposure(symbol)
            risk_assessment["total_exposure"] = str(total_exposure)

            # Check exposure limits
            self._check_symbol_exposure_limits(risk_assessment, symbol, quantity)

            # Check position concentration
            self._check_position_concentration(risk_assessment, bot_id, symbol)

            # Check conflicting positions
            self._check_conflicting_positions(risk_assessment, bot_id, symbol, side, quantity)

            # Generate recommendations
            self._generate_risk_recommendations(risk_assessment)

            # Store assessment
            self._risk_assessments[bot_id] = risk_assessment

            return risk_assessment

        except Exception as e:
            self._logger.error(f"Failed to check cross-bot risk for {bot_id}: {e}")
            raise ServiceError(f"Failed to check cross-bot risk: {e}") from e

    async def cleanup_expired_signals(self) -> int:
        """
        Clean up expired signals.

        Returns:
            Number of signals cleaned up
        """
        try:
            from datetime import datetime, timezone, timedelta

            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            cleaned_count = 0

            for bot_id in self._shared_signals:
                original_count = len(self._shared_signals[bot_id])
                self._shared_signals[bot_id] = [
                    signal for signal in self._shared_signals[bot_id]
                    if signal.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)) > cutoff_time
                ]
                cleaned_count += original_count - len(self._shared_signals[bot_id])

            if cleaned_count > 0:
                self._logger.debug(f"Cleaned up {cleaned_count} expired signals")

            return cleaned_count

        except Exception as e:
            self._logger.error(f"Failed to cleanup expired signals: {e}")
            return 0

    def _detect_position_conflict(
        self, bot1: dict[str, Any], bot2: dict[str, Any], symbol: str
    ) -> dict[str, Any] | None:
        """Detect position conflicts between two bots."""
        try:
            pos1 = bot1["position"]
            pos2 = bot2["position"]

            # Check for opposing positions
            if pos1.get("side") != pos2.get("side"):
                return {
                    "type": "opposing_positions",
                    "symbol": symbol,
                    "bot1": {"id": bot1["bot_id"], "side": pos1.get("side")},
                    "bot2": {"id": bot2["bot_id"], "side": pos2.get("side")},
                    "severity": "high",
                }

            return None

        except Exception as e:
            self._logger.error(f"Error detecting position conflict: {e}")
            return None

    def _calculate_total_exposure(self, symbol: str) -> Decimal:
        """Calculate total exposure across all bots for a symbol."""
        total = Decimal("0")
        try:
            for positions in self._position_data.values():
                if symbol in positions:
                    quantity = positions[symbol].get("quantity", "0")
                    total += Decimal(str(quantity))
        except Exception as e:
            self._logger.error(f"Error calculating total exposure: {e}")
        return total

    def _check_symbol_exposure_limits(
        self, risk_assessment: dict[str, Any], symbol: str, quantity: Decimal
    ) -> None:
        """Check symbol exposure limits."""
        # Placeholder implementation
        pass

    def _check_position_concentration(
        self, risk_assessment: dict[str, Any], bot_id: str, symbol: str
    ) -> None:
        """Check position concentration risk."""
        # Placeholder implementation
        pass

    def _check_conflicting_positions(
        self,
        risk_assessment: dict[str, Any],
        bot_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
    ) -> None:
        """Check for conflicting positions."""
        # Placeholder implementation
        pass

    def _generate_risk_recommendations(self, risk_assessment: dict[str, Any]) -> None:
        """Generate risk recommendations."""
        # Placeholder implementation
        pass

    def _get_current_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)