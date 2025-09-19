"""
Bot Resource Management Service Implementation.

This service handles resource allocation and management for bot instances,
following the service layer pattern with proper dependency injection.
"""

from decimal import Decimal
from typing import Any, TYPE_CHECKING

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotPriority

from .interfaces import IResourceManagementService

if TYPE_CHECKING:
    from .resource_manager import ResourceManager


class BotResourceService(BaseService, IResourceManagementService):
    """
    Service for managing bot resources with dependency injection.

    This service provides the business logic layer for resource management,
    coordinating with the ResourceManager component and other services.
    """

    def __init__(
        self,
        resource_manager: "ResourceManager",
        name: str = "BotResourceService",
        config: dict[str, Any] = None,
    ):
        """Initialize bot resource service."""
        super().__init__(name=name, config=config)
        self._resource_manager = resource_manager
        self._logger = get_logger(__name__)
        self._resource_limits = self._load_resource_limits()

    def _load_resource_limits(self) -> dict[str, Any]:
        """Load resource limits configuration."""
        return {
            "max_capital_per_bot": Decimal("10000.00"),
            "max_total_capital": Decimal("100000.00"),
            "max_cpu_percentage": 80.0,
            "max_memory_mb": 1024,
            "max_api_requests_per_minute": 100,
            "max_database_connections": 10,
        }

    async def request_resources(
        self,
        bot_id: str,
        capital_amount: Decimal,
        priority: BotPriority = BotPriority.NORMAL,
    ) -> bool:
        """
        Request resources for a bot.

        Args:
            bot_id: Bot ID requesting resources
            capital_amount: Amount of capital to allocate
            priority: Bot priority level

        Returns:
            True if resources allocated successfully
        """
        try:
            # Validate inputs
            if not bot_id:
                raise ValidationError("Bot ID is required")

            if capital_amount <= 0:
                raise ValidationError("Capital amount must be positive")

            if capital_amount > self._resource_limits["max_capital_per_bot"]:
                raise ValidationError(f"Capital amount exceeds maximum per bot: {self._resource_limits['max_capital_per_bot']}")

            # Check resource availability
            available = await self._resource_manager.check_resource_availability("capital", capital_amount)
            if not available:
                self._logger.warning(f"Insufficient capital resources for bot {bot_id}")
                return False

            # Request resources from resource manager
            success = await self._resource_manager.request_resources(
                bot_id=bot_id,
                capital_amount=capital_amount,
                priority=priority,
            )

            if success:
                self._logger.info(f"Successfully allocated resources to bot {bot_id}: {capital_amount}")
            else:
                self._logger.warning(f"Failed to allocate resources to bot {bot_id}")

            return success

        except Exception as e:
            self._logger.error(f"Failed to request resources for bot {bot_id}: {e}")
            raise ServiceError(f"Failed to request resources: {e}") from e

    async def release_resources(self, bot_id: str) -> bool:
        """
        Release resources for a bot.

        Args:
            bot_id: Bot ID to release resources for

        Returns:
            True if resources released successfully
        """
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Release resources through resource manager
            success = await self._resource_manager.release_resources(bot_id)

            if success:
                self._logger.info(f"Successfully released resources for bot {bot_id}")
            else:
                self._logger.warning(f"Failed to release resources for bot {bot_id}")

            return success

        except Exception as e:
            self._logger.error(f"Failed to release resources for bot {bot_id}: {e}")
            raise ServiceError(f"Failed to release resources: {e}") from e

    async def verify_resources(self, bot_id: str) -> bool:
        """
        Verify resource allocation for a bot.

        Args:
            bot_id: Bot ID to verify

        Returns:
            True if resources are properly allocated
        """
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            # Verify through resource manager
            verified = await self._resource_manager.verify_resources(bot_id)

            if verified:
                self._logger.debug(f"Resources verified for bot {bot_id}")
            else:
                self._logger.warning(f"Resource verification failed for bot {bot_id}")

            return verified

        except Exception as e:
            self._logger.error(f"Failed to verify resources for bot {bot_id}: {e}")
            raise ServiceError(f"Failed to verify resources: {e}") from e

    async def get_resource_summary(self) -> dict[str, Any]:
        """
        Get resource usage summary.

        Returns:
            Resource usage summary
        """
        try:
            # Get summary from resource manager
            summary = await self._resource_manager.get_resource_summary()

            # Add service-level information
            enhanced_summary = {
                **summary,
                "limits": self._resource_limits,
                "utilization": self._calculate_utilization(summary),
                "timestamp": self._get_current_timestamp(),
            }

            return enhanced_summary

        except Exception as e:
            self._logger.error(f"Failed to get resource summary: {e}")
            raise ServiceError(f"Failed to get resource summary: {e}") from e

    async def check_resource_availability(
        self, resource_type: str, amount: Decimal
    ) -> bool:
        """
        Check resource availability.

        Args:
            resource_type: Type of resource to check
            amount: Amount to check availability for

        Returns:
            True if resources are available
        """
        try:
            if not resource_type:
                raise ValidationError("Resource type is required")

            if amount <= 0:
                raise ValidationError("Amount must be positive")

            # Check through resource manager
            available = await self._resource_manager.check_resource_availability(resource_type, amount)

            self._logger.debug(f"Resource availability check for {resource_type}: {amount} = {available}")
            return available

        except Exception as e:
            self._logger.error(f"Failed to check resource availability: {e}")
            raise ServiceError(f"Failed to check resource availability: {e}") from e

    async def update_capital_allocation(
        self, bot_id: str, new_amount: Decimal
    ) -> bool:
        """
        Update capital allocation for a bot.

        Args:
            bot_id: Bot ID
            new_amount: New capital amount

        Returns:
            True if updated successfully
        """
        try:
            if not bot_id:
                raise ValidationError("Bot ID is required")

            if new_amount < 0:
                raise ValidationError("Capital amount cannot be negative")

            if new_amount > self._resource_limits["max_capital_per_bot"]:
                raise ValidationError(f"Capital amount exceeds maximum per bot: {self._resource_limits['max_capital_per_bot']}")

            # Update through resource manager
            success = await self._resource_manager.update_capital_allocation(bot_id, new_amount)

            if success:
                self._logger.info(f"Updated capital allocation for bot {bot_id}: {new_amount}")
            else:
                self._logger.warning(f"Failed to update capital allocation for bot {bot_id}")

            return success

        except Exception as e:
            self._logger.error(f"Failed to update capital allocation for bot {bot_id}: {e}")
            raise ServiceError(f"Failed to update capital allocation: {e}") from e

    def _calculate_utilization(self, summary: dict[str, Any]) -> dict[str, float]:
        """Calculate resource utilization percentages."""
        utilization = {}

        try:
            # Calculate capital utilization
            total_allocated = summary.get("total_allocated_capital", 0)
            max_capital = float(self._resource_limits["max_total_capital"])
            if max_capital > 0:
                utilization["capital"] = (float(total_allocated) / max_capital) * 100

            # Calculate other utilizations based on summary data
            # This would be expanded based on actual resource types tracked

        except Exception as e:
            self._logger.error(f"Failed to calculate utilization: {e}")

        return utilization

    def _get_current_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)

    def _get_current_timestamp_iso(self) -> str:
        """Get current timestamp in ISO format."""
        return self._get_current_timestamp().isoformat()