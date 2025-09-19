"""
Bot Management Controller.

This module implements the controller layer for bot management operations,
providing a clean API interface that delegates to appropriate service classes.
Controllers should NOT contain business logic - only coordinate service calls.
"""

from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import EntityNotFoundError, ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotConfiguration, OrderRequest

from .interfaces import (
    IBotCoordinationService,
    IBotInstanceService,
    IBotLifecycleService,
    IBotMonitoringService,
    IResourceManagementService,
)


class BotManagementController(BaseService):
    """
    Controller for bot management operations.

    This controller provides the API layer for bot management, delegating
    all business logic to appropriate service classes. It should NOT contain
    any business logic, only coordinate service calls and handle basic
    request/response formatting.
    """

    def __init__(
        self,
        bot_instance_service: IBotInstanceService,
        bot_coordination_service: IBotCoordinationService,
        bot_lifecycle_service: IBotLifecycleService,
        bot_monitoring_service: IBotMonitoringService,
        resource_management_service: IResourceManagementService,
    ):
        """Initialize controller with required services."""
        super().__init__()
        self._bot_instance_service = bot_instance_service
        self._bot_coordination_service = bot_coordination_service
        self._bot_lifecycle_service = bot_lifecycle_service
        self._bot_monitoring_service = bot_monitoring_service
        self._resource_management_service = resource_management_service
        self._logger = get_logger(__name__)

    @property
    def bot_instance_service(self) -> IBotInstanceService:
        """Get bot instance service."""
        return self._bot_instance_service

    @property
    def bot_coordination_service(self) -> IBotCoordinationService:
        """Get bot coordination service."""
        return self._bot_coordination_service

    @property
    def bot_lifecycle_service(self) -> IBotLifecycleService:
        """Get bot lifecycle service."""
        return self._bot_lifecycle_service

    @property
    def bot_monitoring_service(self) -> IBotMonitoringService:
        """Get bot monitoring service."""
        return self._bot_monitoring_service

    @property
    def resource_management_service(self) -> IResourceManagementService:
        """Get resource management service."""
        return self._resource_management_service

    async def create_bot(
        self,
        template_name: str,
        bot_name: str,
        exchange: str,
        strategy: str,
        capital_amount: str,
        deployment_strategy: str = "immediate",
        priority: str = "medium",
        custom_config: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Create a new bot - delegates to lifecycle service."""
        try:
            # Basic validation - no business logic
            if not all([template_name, bot_name, exchange, strategy, capital_amount]):
                raise ValidationError("Missing required parameters for bot creation")

            # Delegate to service
            bot_config = await self._bot_lifecycle_service.create_bot_from_template(
                template_name=template_name,
                bot_name=bot_name,
                exchange=exchange,
                strategy=strategy,
                capital_amount=capital_amount,
                deployment_strategy=deployment_strategy,
                priority=priority,
                custom_config=custom_config or {},
            )

            # Create bot instance
            bot_id = await self._bot_instance_service.create_bot_instance(bot_config)

            return {
                "success": True,
                "bot_id": bot_id,
                "message": f"Bot '{bot_name}' created successfully",
                "config": bot_config.model_dump() if hasattr(bot_config, 'model_dump') else dict(bot_config),
            }

        except Exception as e:
            self._logger.error(f"Failed to create bot: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Bot creation failed",
            }

    async def start_bot(self, bot_id: str) -> dict[str, Any]:
        """Start a bot - delegates to instance service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Check if bot exists by attempting to get its state
            try:
                await self._bot_instance_service.get_bot_state(bot_id)
            except EntityNotFoundError:
                raise EntityNotFoundError(f"Bot not found: {bot_id}")

            success = await self._bot_instance_service.start_bot(bot_id)

            return {
                "success": success,
                "bot_id": bot_id,
                "message": f"Bot {bot_id} started successfully" if success else f"Failed to start bot {bot_id}",
            }

        except (ValidationError, EntityNotFoundError):
            raise  # Re-raise these specific exceptions
        except Exception as e:
            self._logger.error(f"Failed to start bot {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to start bot {bot_id}",
            }

    async def stop_bot(self, bot_id: str) -> dict[str, Any]:
        """Stop a bot - delegates to instance service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Check if bot exists by attempting to get its state
            try:
                await self._bot_instance_service.get_bot_state(bot_id)
            except EntityNotFoundError:
                raise EntityNotFoundError(f"Bot not found: {bot_id}")

            success = await self._bot_instance_service.stop_bot(bot_id)

            return {
                "success": success,
                "bot_id": bot_id,
                "message": f"Bot {bot_id} stopped successfully" if success else f"Failed to stop bot {bot_id}",
            }

        except (ValidationError, EntityNotFoundError):
            raise  # Re-raise these specific exceptions
        except Exception as e:
            self._logger.error(f"Failed to stop bot {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to stop bot {bot_id}",
            }

    async def terminate_bot(self, bot_id: str, reason: str = "user_request") -> dict[str, Any]:
        """Terminate a bot - delegates to lifecycle service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Check if bot exists by attempting to get its state
            try:
                await self._bot_instance_service.get_bot_state(bot_id)
            except EntityNotFoundError:
                raise EntityNotFoundError(f"Bot not found: {bot_id}")

            success = await self._bot_lifecycle_service.terminate_bot(bot_id, reason)

            return {
                "success": success,
                "bot_id": bot_id,
                "reason": reason,
                "message": f"Bot {bot_id} terminated successfully" if success else f"Failed to terminate bot {bot_id}",
            }

        except (ValidationError, EntityNotFoundError):
            raise  # Re-raise these specific exceptions
        except Exception as e:
            self._logger.error(f"Failed to terminate bot {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to terminate bot {bot_id}",
            }

    async def get_bot_status(self, bot_id: str) -> dict[str, Any]:
        """Get bot status - delegates to monitoring service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Get bot state first, this will raise EntityNotFoundError if bot doesn't exist
            bot_state = await self._bot_instance_service.get_bot_state(bot_id)

            # Get status from monitoring service
            health_data = await self._bot_monitoring_service.get_bot_health(bot_id)

            return {
                "success": True,
                "bot_id": bot_id,
                "state": bot_state.value if hasattr(bot_state, 'value') else str(bot_state),
                "health": health_data,
                "timestamp": health_data.get("timestamp"),
            }

        except (ValidationError, EntityNotFoundError):
            raise  # Re-raise these specific exceptions
        except Exception as e:
            self._logger.error(f"Failed to get bot status {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get status for bot {bot_id}",
            }

    async def execute_bot_trade(
        self,
        bot_id: str,
        order_request: OrderRequest,
        execution_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute trade for a bot - delegates to instance service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")
            if not order_request:
                raise ValidationError("Order request is required")

            result = await self._bot_instance_service.execute_trade(
                bot_id, order_request, execution_params
            )

            return {
                "success": True,
                "bot_id": bot_id,
                "trade_result": result,
                "message": "Trade executed successfully",
            }

        except Exception as e:
            self._logger.error(f"Failed to execute trade for bot {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to execute trade for bot {bot_id}",
            }

    async def get_system_overview(self) -> dict[str, Any]:
        """Get system overview - delegates to monitoring service."""
        try:
            system_health = await self._bot_monitoring_service.get_system_health()
            resource_summary = await self._resource_management_service.get_resource_summary()

            return {
                "success": True,
                "system_health": system_health,
                "resource_summary": resource_summary,
                "timestamp": system_health.get("timestamp"),
            }

        except Exception as e:
            self._logger.error(f"Failed to get system overview: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get system overview",
            }

    async def pause_bot(self, bot_id: str) -> dict[str, Any]:
        """Pause a bot - delegates to instance service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            success = await self._bot_instance_service.pause_bot(bot_id)

            return {
                "success": success,
                "bot_id": bot_id,
                "message": f"Bot {bot_id} paused successfully" if success else f"Failed to pause bot {bot_id}",
            }

        except Exception as e:
            self._logger.error(f"Failed to pause bot {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to pause bot {bot_id}",
            }

    async def resume_bot(self, bot_id: str) -> dict[str, Any]:
        """Resume a bot - delegates to instance service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            success = await self._bot_instance_service.resume_bot(bot_id)

            return {
                "success": success,
                "bot_id": bot_id,
                "message": f"Bot {bot_id} resumed successfully" if success else f"Failed to resume bot {bot_id}",
            }

        except Exception as e:
            self._logger.error(f"Failed to resume bot {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to resume bot {bot_id}",
            }

    async def get_bot_state(self, bot_id: str) -> dict[str, Any]:
        """Get bot state - delegates to instance service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            bot_state = await self._bot_instance_service.get_bot_state(bot_id)

            return {
                "success": True,
                "bot_id": bot_id,
                "state": bot_state.value if hasattr(bot_state, 'value') else str(bot_state),
            }

        except Exception as e:
            self._logger.error(f"Failed to get bot state {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get state for bot {bot_id}",
            }

    async def get_bot_metrics(self, bot_id: str) -> dict[str, Any]:
        """Get bot metrics - delegates to monitoring service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            metrics = await self._bot_monitoring_service.get_bot_metrics(bot_id)

            return {
                "success": True,
                "bot_id": bot_id,
                "metrics": metrics.model_dump() if hasattr(metrics, 'model_dump') else dict(metrics),
            }

        except Exception as e:
            self._logger.error(f"Failed to get bot metrics {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get metrics for bot {bot_id}",
            }

    async def allocate_resources(self, bot_id: str, resources: dict[str, Any]) -> dict[str, Any]:
        """Allocate resources - delegates to resource management service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Extract required parameters from resources dict
            capital_amount = resources.get("capital_amount")
            priority = resources.get("priority", "medium")

            if capital_amount is None:
                raise ValidationError("Capital amount is required")

            success = await self._resource_management_service.request_resources(
                bot_id, capital_amount, priority
            )

            return {
                "success": success,
                "bot_id": bot_id,
                "allocated_resources": resources,
                "message": f"Resources allocated to bot {bot_id}" if success else f"Failed to allocate resources to bot {bot_id}",
            }

        except Exception as e:
            self._logger.error(f"Failed to allocate resources for bot {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to allocate resources for bot {bot_id}",
            }

    async def deallocate_resources(self, bot_id: str) -> dict[str, Any]:
        """Deallocate resources - delegates to resource management service."""
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            success = await self._resource_management_service.release_resources(bot_id)

            return {
                "success": success,
                "bot_id": bot_id,
                "message": f"Resources deallocated from bot {bot_id}" if success else f"Failed to deallocate resources from bot {bot_id}",
            }

        except Exception as e:
            self._logger.error(f"Failed to deallocate resources for bot {bot_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to deallocate resources for bot {bot_id}",
            }

    async def list_bots(self) -> dict[str, Any]:
        """List all bots - delegates to instance service."""
        try:
            # Get active bot IDs from instance service
            active_bot_ids = self._bot_instance_service.get_active_bot_ids()

            # Get detailed information for each bot
            bots = []
            for bot_id in active_bot_ids:
                try:
                    bot_status = await self.get_bot_status(bot_id)
                    if bot_status.get("success", False):
                        bots.append(bot_status.get("bot", {}))
                except Exception as e:
                    self._logger.warning(f"Failed to get status for bot {bot_id}: {e}")
                    # Continue with other bots

            return {
                "success": True,
                "bots": bots,
                "total": len(bots),
                "message": f"Found {len(bots)} bots",
            }

        except Exception as e:
            self._logger.error(f"Failed to list bots: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to list bots",
                "bots": [],
                "total": 0,
            }

    async def delete_bot(self, bot_id: str) -> dict[str, Any]:
        """Delete a bot - aliases to terminate_bot for API compatibility."""
        return await self.terminate_bot(bot_id, reason="user_delete")