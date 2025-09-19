"""
Bot Instance Entity.

This module implements a lightweight BotInstance entity that holds bot configuration
and state. Business logic has been moved to appropriate service classes following
the service layer pattern.
"""

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import (
    BotConfiguration,
    BotMetrics,
    BotState,
    OrderRequest,
)

if TYPE_CHECKING:
    from src.capital_management.service import CapitalService
    from src.execution.interfaces import IExecutionService
    from src.risk_management.interfaces import IRiskService


class BotInstance:
    """
    Bot Instance Entity.

    This is a lightweight entity class that holds bot configuration and state.
    Business logic has been moved to appropriate service classes.
    This class now only contains essential data and minimal behavior.
    """

    def __init__(
        self,
        bot_config: BotConfiguration,
        capital_service: "CapitalService" = None,
        execution_service: "IExecutionService" = None,
        risk_service: "IRiskService" = None,
    ):
        """
        Initialize bot instance with minimal dependencies.

        Args:
            bot_config: Bot configuration
            capital_service: Capital service (optional)
            execution_service: Execution service (optional)
            risk_service: Risk service (optional)
        """
        self._logger = get_logger(__name__)
        self.bot_config = bot_config
        self._capital_service = capital_service
        self._execution_service = execution_service
        self._risk_service = risk_service

        # Internal state
        self._state = BotState.STOPPED
        self._metrics = self._create_default_metrics()
        self._last_heartbeat = datetime.now(timezone.utc)
        self._positions = {}
        self._created_at = datetime.now(timezone.utc)

        # Validate configuration
        self._validate_configuration()

        self._logger.info(f"Created bot instance: {bot_config.id}")

    def _validate_configuration(self) -> None:
        """
        Validate bot configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if not self.bot_config:
            raise ValidationError("Bot configuration is required")

        if not self.bot_config.id:
            raise ValidationError("Bot ID is required")

        if not self.bot_config.name:
            raise ValidationError("Bot name is required")

        if not self.bot_config.exchange:
            raise ValidationError("Exchange is required")

        if not self.bot_config.strategy:
            raise ValidationError("Strategy is required")

    def _create_default_metrics(self) -> BotMetrics:
        """
        Create default metrics for the bot.

        Returns:
            Default bot metrics
        """
        from decimal import Decimal

        return BotMetrics(
            bot_id=self.bot_config.id,
            timestamp=datetime.now(timezone.utc),
            total_pnl=Decimal("0"),
            total_trades=0,
            win_rate=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            error_count=0,
        )

    async def start(self) -> None:
        """
        Start the bot instance.

        Note: Actual business logic should be in BotInstanceService.
        This method only updates the state.
        """
        try:
            if self._state == BotState.RUNNING:
                self._logger.warning(f"Bot {self.bot_config.id} is already running")
                return

            self._state = BotState.RUNNING
            self._last_heartbeat = datetime.now(timezone.utc)

            self._logger.info(f"Started bot: {self.bot_config.id}")

        except Exception as e:
            self._logger.error(f"Failed to start bot {self.bot_config.id}: {e}")
            self._state = BotState.ERROR
            raise

    async def stop(self) -> None:
        """
        Stop the bot instance.

        Note: Actual business logic should be in BotInstanceService.
        This method only updates the state.
        """
        try:
            if self._state == BotState.STOPPED:
                self._logger.warning(f"Bot {self.bot_config.id} is already stopped")
                return

            self._state = BotState.STOPPED

            self._logger.info(f"Stopped bot: {self.bot_config.id}")

        except Exception as e:
            self._logger.error(f"Failed to stop bot {self.bot_config.id}: {e}")
            self._state = BotState.ERROR
            raise

    async def pause(self) -> None:
        """
        Pause the bot instance.

        Note: Actual business logic should be in BotInstanceService.
        This method only updates the state.
        """
        try:
            if self._state != BotState.RUNNING:
                self._logger.warning(f"Bot {self.bot_config.id} is not running, cannot pause")
                return

            self._state = BotState.PAUSED

            self._logger.info(f"Paused bot: {self.bot_config.id}")

        except Exception as e:
            self._logger.error(f"Failed to pause bot {self.bot_config.id}: {e}")
            self._state = BotState.ERROR
            raise

    async def resume(self) -> None:
        """
        Resume the bot instance.

        Note: Actual business logic should be in BotInstanceService.
        This method only updates the state.
        """
        try:
            if self._state != BotState.PAUSED:
                self._logger.warning(f"Bot {self.bot_config.id} is not paused, cannot resume")
                return

            self._state = BotState.RUNNING
            self._last_heartbeat = datetime.now(timezone.utc)

            self._logger.info(f"Resumed bot: {self.bot_config.id}")

        except Exception as e:
            self._logger.error(f"Failed to resume bot {self.bot_config.id}: {e}")
            self._state = BotState.ERROR
            raise

    def get_bot_state(self) -> BotState:
        """
        Get current bot state.

        Returns:
            Current bot state
        """
        return self._state

    def get_bot_metrics(self) -> BotMetrics:
        """
        Get bot metrics.

        Returns:
            Current bot metrics
        """
        # Update timestamp
        self._metrics.timestamp = datetime.now(timezone.utc)
        return self._metrics

    def get_bot_config(self) -> BotConfiguration:
        """
        Get bot configuration.

        Returns:
            Bot configuration
        """
        return self.bot_config

    async def get_bot_summary(self) -> dict[str, Any]:
        """
        Get bot summary information.

        Returns:
            Bot summary data
        """
        return {
            "id": self.bot_config.id,
            "name": self.bot_config.name,
            "state": self._state.value if hasattr(self._state, 'value') else str(self._state),
            "exchange": self.bot_config.exchange,
            "strategy": self.bot_config.strategy,
            "created_at": self._created_at.isoformat(),
            "last_heartbeat": self._last_heartbeat.isoformat(),
            "metrics": self._metrics.model_dump() if hasattr(self._metrics, 'model_dump') else dict(self._metrics),
        }

    async def execute_trade(self, order_request: OrderRequest, execution_params: dict) -> Any:
        """
        Execute a trade.

        Note: This method delegates to the execution service.
        Complex business logic should be in BotInstanceService.

        Args:
            order_request: Order request
            execution_params: Execution parameters

        Returns:
            Execution result
        """
        try:
            if not self._execution_service:
                raise ValidationError("Execution service not available")

            if self._state != BotState.RUNNING:
                raise ValidationError(f"Bot {self.bot_config.id} is not running")

            # Delegate to execution service
            result = await self._execution_service.execute_order(order_request)

            # Update metrics
            self._metrics.total_trades += 1

            self._logger.info(f"Executed trade for bot {self.bot_config.id}")
            return result

        except Exception as e:
            self._logger.error(f"Failed to execute trade for bot {self.bot_config.id}: {e}")
            self._metrics.error_count += 1
            raise

    async def update_position(self, symbol: str, position_data: dict) -> None:
        """
        Update position data.

        Args:
            symbol: Trading symbol
            position_data: Position data
        """
        try:
            self._positions[symbol] = position_data
            self._logger.debug(f"Updated position for bot {self.bot_config.id}, symbol {symbol}")

        except Exception as e:
            self._logger.error(f"Failed to update position for bot {self.bot_config.id}: {e}")
            raise

    async def close_position(self, symbol: str, reason: str) -> bool:
        """
        Close a position.

        Args:
            symbol: Trading symbol
            reason: Reason for closing

        Returns:
            True if position closed successfully
        """
        try:
            if symbol in self._positions:
                del self._positions[symbol]
                self._logger.info(f"Closed position for bot {self.bot_config.id}, symbol {symbol}, reason: {reason}")
                return True
            else:
                self._logger.warning(f"No position found for bot {self.bot_config.id}, symbol {symbol}")
                return False

        except Exception as e:
            self._logger.error(f"Failed to close position for bot {self.bot_config.id}: {e}")
            return False

    async def get_heartbeat(self) -> dict[str, Any]:
        """
        Get bot heartbeat information.

        Returns:
            Heartbeat data
        """
        self._last_heartbeat = datetime.now(timezone.utc)

        return {
            "bot_id": self.bot_config.id,
            "state": self._state.value if hasattr(self._state, 'value') else str(self._state),
            "timestamp": self._last_heartbeat.isoformat(),
            "uptime_seconds": (self._last_heartbeat - self._created_at).total_seconds(),
        }

    async def restart(self, reason: str) -> None:
        """
        Restart the bot instance.

        Args:
            reason: Reason for restart
        """
        try:
            self._logger.info(f"Restarting bot {self.bot_config.id}, reason: {reason}")

            # Stop then start
            await self.stop()
            await self.start()

            self._logger.info(f"Successfully restarted bot {self.bot_config.id}")

        except Exception as e:
            self._logger.error(f"Failed to restart bot {self.bot_config.id}: {e}")
            self._state = BotState.ERROR
            raise

    async def queue_websocket_message(self, message: dict) -> bool:
        """
        Queue a websocket message for processing.

        Note: This is a placeholder. Real implementation would
        delegate to appropriate service.

        Args:
            message: Websocket message

        Returns:
            True if queued successfully
        """
        try:
            # Placeholder implementation
            self._logger.debug(f"Queued websocket message for bot {self.bot_config.id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to queue websocket message for bot {self.bot_config.id}: {e}")
            return False

    def set_metrics_collector(self, metrics_collector) -> None:
        """
        Set metrics collector.

        Args:
            metrics_collector: Metrics collector instance
        """
        # Placeholder implementation
        pass