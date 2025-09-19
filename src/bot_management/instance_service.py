"""
Bot Instance Service Implementation.

This service handles bot instance management operations following the service layer pattern.
It coordinates with repositories and other services but contains the business logic for
bot instance lifecycle operations.
"""

from typing import Any, TYPE_CHECKING

from src.core.base.service import BaseService
from src.core.config import Config
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotConfiguration, BotState, OrderRequest

from .interfaces import IBotInstanceService

if TYPE_CHECKING:
    from src.capital_management.service import CapitalService
    from src.execution.interfaces import IExecutionService
    from src.risk_management.interfaces import IRiskService
    from .bot_instance import BotInstance


class BotInstanceService(BaseService, IBotInstanceService):
    """
    Service for managing bot instances.

    This service handles the business logic for bot instance operations,
    delegating to repositories for data persistence and coordinating
    with other services as needed.
    """

    def __init__(
        self,
        config: Config,
        capital_service: "CapitalService",
        execution_service: "IExecutionService" = None,
        risk_service: "IRiskService" = None,
    ):
        """Initialize bot instance service."""
        super().__init__()
        self._config = config
        self._capital_service = capital_service
        self._execution_service = execution_service
        self._risk_service = risk_service
        self._logger = get_logger(__name__)
        self._bot_instances: dict[str, "BotInstance"] = {}

    async def create_bot_instance(self, bot_config: BotConfiguration) -> str:
        """
        Create a new bot instance.

        Args:
            bot_config: Bot configuration

        Returns:
            Bot ID of the created instance
        """
        try:
            # Validate configuration
            if not bot_config:
                raise ValidationError("Bot configuration is required")

            if not bot_config.name:
                raise ValidationError("Bot name is required")

            # Check if bot with same name already exists
            if any(
                instance.get_bot_config().name == bot_config.name
                for instance in self._bot_instances.values()
            ):
                raise ValidationError(f"Bot with name '{bot_config.name}' already exists")

            # Validate exchange connectivity if execution service available
            if self._execution_service:
                await self._validate_exchange_connectivity(bot_config)

            # Create bot instance
            from .bot_entity import BotInstance

            bot_instance = BotInstance(
                bot_config=bot_config,
                capital_service=self._capital_service,
                execution_service=self._execution_service,
                risk_service=self._risk_service,
            )

            # Store instance
            bot_id = bot_config.id
            self._bot_instances[bot_id] = bot_instance

            self._logger.info(f"Created bot instance: {bot_id}")
            return bot_id

        except Exception as e:
            self._logger.error(f"Failed to create bot instance: {e}")
            raise ServiceError(f"Failed to create bot instance: {e}") from e

    async def start_bot(self, bot_id: str) -> bool:
        """
        Start a bot instance.

        Args:
            bot_id: Bot ID to start

        Returns:
            True if started successfully
        """
        try:
            if bot_id not in self._bot_instances:
                raise ValidationError(f"Bot {bot_id} not found")

            bot_instance = self._bot_instances[bot_id]
            await bot_instance.start()

            self._logger.info(f"Started bot: {bot_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to start bot {bot_id}: {e}")
            return False

    async def stop_bot(self, bot_id: str) -> bool:
        """
        Stop a bot instance.

        Args:
            bot_id: Bot ID to stop

        Returns:
            True if stopped successfully
        """
        try:
            if bot_id not in self._bot_instances:
                raise ValidationError(f"Bot {bot_id} not found")

            bot_instance = self._bot_instances[bot_id]
            await bot_instance.stop()

            self._logger.info(f"Stopped bot: {bot_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to stop bot {bot_id}: {e}")
            return False

    async def pause_bot(self, bot_id: str) -> bool:
        """
        Pause a bot instance.

        Args:
            bot_id: Bot ID to pause

        Returns:
            True if paused successfully
        """
        try:
            if bot_id not in self._bot_instances:
                raise ValidationError(f"Bot {bot_id} not found")

            bot_instance = self._bot_instances[bot_id]
            await bot_instance.pause()

            self._logger.info(f"Paused bot: {bot_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to pause bot {bot_id}: {e}")
            return False

    async def resume_bot(self, bot_id: str) -> bool:
        """
        Resume a bot instance.

        Args:
            bot_id: Bot ID to resume

        Returns:
            True if resumed successfully
        """
        try:
            if bot_id not in self._bot_instances:
                raise ValidationError(f"Bot {bot_id} not found")

            bot_instance = self._bot_instances[bot_id]
            await bot_instance.resume()

            self._logger.info(f"Resumed bot: {bot_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to resume bot {bot_id}: {e}")
            return False

    async def get_bot_state(self, bot_id: str) -> BotState:
        """
        Get bot state.

        Args:
            bot_id: Bot ID

        Returns:
            Bot state
        """
        try:
            if bot_id not in self._bot_instances:
                raise ValidationError(f"Bot {bot_id} not found")

            bot_instance = self._bot_instances[bot_id]
            return bot_instance.get_bot_state()

        except Exception as e:
            self._logger.error(f"Failed to get bot state {bot_id}: {e}")
            raise ServiceError(f"Failed to get bot state: {e}") from e

    async def execute_trade(
        self,
        bot_id: str,
        order_request: OrderRequest,
        execution_params: dict[str, Any],
    ) -> Any:
        """
        Execute a trade for a bot.

        Args:
            bot_id: Bot ID
            order_request: Order request
            execution_params: Execution parameters

        Returns:
            Execution result
        """
        try:
            if bot_id not in self._bot_instances:
                raise ValidationError(f"Bot {bot_id} not found")

            bot_instance = self._bot_instances[bot_id]
            result = await bot_instance.execute_trade(order_request, execution_params)

            self._logger.info(f"Executed trade for bot {bot_id}")
            return result

        except Exception as e:
            self._logger.error(f"Failed to execute trade for bot {bot_id}: {e}")
            raise ServiceError(f"Failed to execute trade: {e}") from e

    async def update_position(
        self, bot_id: str, symbol: str, position_data: dict[str, Any]
    ) -> None:
        """
        Update position for a bot.

        Args:
            bot_id: Bot ID
            symbol: Trading symbol
            position_data: Position data
        """
        try:
            if bot_id not in self._bot_instances:
                raise ValidationError(f"Bot {bot_id} not found")

            bot_instance = self._bot_instances[bot_id]
            await bot_instance.update_position(symbol, position_data)

            self._logger.debug(f"Updated position for bot {bot_id}, symbol {symbol}")

        except Exception as e:
            self._logger.error(f"Failed to update position for bot {bot_id}: {e}")
            raise ServiceError(f"Failed to update position: {e}") from e

    async def close_position(self, bot_id: str, symbol: str, reason: str) -> bool:
        """
        Close position for a bot.

        Args:
            bot_id: Bot ID
            symbol: Trading symbol
            reason: Reason for closing

        Returns:
            True if closed successfully
        """
        try:
            if bot_id not in self._bot_instances:
                raise ValidationError(f"Bot {bot_id} not found")

            bot_instance = self._bot_instances[bot_id]
            result = await bot_instance.close_position(symbol, reason)

            self._logger.info(f"Closed position for bot {bot_id}, symbol {symbol}, reason: {reason}")
            return result

        except Exception as e:
            self._logger.error(f"Failed to close position for bot {bot_id}: {e}")
            return False

    async def remove_bot_instance(self, bot_id: str) -> bool:
        """
        Remove a bot instance.

        Args:
            bot_id: Bot ID to remove

        Returns:
            True if removed successfully
        """
        try:
            if bot_id not in self._bot_instances:
                return True  # Already removed

            # Stop bot first if running
            bot_instance = self._bot_instances[bot_id]
            bot_state = bot_instance.get_bot_state()

            if bot_state in [BotState.RUNNING, BotState.PAUSED]:
                await bot_instance.stop()

            # Remove from instances
            del self._bot_instances[bot_id]

            self._logger.info(f"Removed bot instance: {bot_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to remove bot instance {bot_id}: {e}")
            return False

    def get_active_bot_ids(self) -> list[str]:
        """Get list of active bot IDs."""
        return [
            bot_id for bot_id, instance in self._bot_instances.items()
            if instance.get_bot_state() in [BotState.RUNNING, BotState.PAUSED]
        ]

    def get_bot_count(self) -> int:
        """Get total number of bot instances."""
        return len(self._bot_instances)

    async def _validate_exchange_connectivity(self, bot_config: BotConfiguration) -> None:
        """
        Validate exchange connectivity.

        Args:
            bot_config: Bot configuration
        """
        try:
            if not self._execution_service:
                return  # Skip validation if no execution service

            # Basic validation - more detailed validation would be in execution service
            if not bot_config.exchange:
                raise ValidationError("Exchange is required")

            # Additional validation logic would go here
            self._logger.debug(f"Validated exchange connectivity for {bot_config.exchange}")

        except Exception as e:
            self._logger.error(f"Exchange validation failed: {e}")
            raise ValidationError(f"Exchange validation failed: {e}") from e