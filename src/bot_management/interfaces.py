"""
Bot Management Service Interfaces.

This module defines the service interfaces for bot management operations,
following proper service layer architecture patterns.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from src.core.types import (
    BotConfiguration,
    BotMetrics,
    BotPriority,
    BotState,
    OrderRequest,
    OrderSide,
)


class IBotCoordinationService(ABC):
    """Interface for bot coordination services."""

    @abstractmethod
    async def register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None:
        """Register a bot for coordination."""

    @abstractmethod
    async def unregister_bot(self, bot_id: str) -> None:
        """Unregister a bot from coordination."""

    @abstractmethod
    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]:
        """Check for position conflicts across bots."""

    @abstractmethod
    async def share_signal(
        self,
        bot_id: str,
        signal_type: str,
        symbol: str,
        direction: str,
        strength: float,
        metadata: dict[str, Any] = None,
    ) -> int:
        """Share trading signal with other bots."""

    @abstractmethod
    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]:
        """Get shared signals for a bot."""

    @abstractmethod
    async def check_cross_bot_risk(
        self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal
    ) -> dict[str, Any]:
        """Check cross-bot risk exposure."""


class IBotLifecycleService(ABC):
    """Interface for bot lifecycle management services."""

    @abstractmethod
    async def create_bot_from_template(
        self,
        template_name: str,
        bot_name: str,
        exchange: str,
        strategy: str,
        capital_amount: Decimal,
        deployment_strategy: str = "immediate",
        priority: BotPriority = BotPriority.NORMAL,
        custom_config: dict[str, Any] = None,
    ) -> BotConfiguration:
        """Create bot from template."""

    @abstractmethod
    async def deploy_bot(
        self, bot_config: BotConfiguration, strategy: str = "immediate"
    ) -> bool:
        """Deploy a bot."""

    @abstractmethod
    async def terminate_bot(self, bot_id: str, reason: str = "user_request") -> bool:
        """Terminate a bot."""

    @abstractmethod
    async def restart_bot(self, bot_id: str, reason: str = "restart_request") -> bool:
        """Restart a bot."""

    @abstractmethod
    async def get_lifecycle_status(self, bot_id: str) -> dict[str, Any]:
        """Get bot lifecycle status."""

    @abstractmethod
    async def rollback_deployment(self, bot_id: str, target_version: str) -> bool:
        """Rollback bot deployment."""


class IBotMonitoringService(ABC):
    """Interface for bot monitoring services."""

    @abstractmethod
    async def get_bot_health(self, bot_id: str) -> dict[str, Any]:
        """Get bot health status."""

    @abstractmethod
    async def get_bot_metrics(self, bot_id: str) -> BotMetrics:
        """Get bot metrics."""

    @abstractmethod
    async def get_system_health(self) -> dict[str, Any]:
        """Get overall system health."""

    @abstractmethod
    async def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""

    @abstractmethod
    async def check_alert_conditions(self) -> list[dict[str, Any]]:
        """Check for alert conditions."""


class IResourceManagementService(ABC):
    """Interface for resource management services."""

    @abstractmethod
    async def request_resources(
        self,
        bot_id: str,
        capital_amount: Decimal,
        priority: BotPriority = BotPriority.NORMAL,
    ) -> bool:
        """Request resources for a bot."""

    @abstractmethod
    async def release_resources(self, bot_id: str) -> bool:
        """Release resources for a bot."""

    @abstractmethod
    async def verify_resources(self, bot_id: str) -> bool:
        """Verify resource allocation for a bot."""

    @abstractmethod
    async def get_resource_summary(self) -> dict[str, Any]:
        """Get resource usage summary."""

    @abstractmethod
    async def check_resource_availability(
        self, resource_type: str, amount: Decimal
    ) -> bool:
        """Check resource availability."""

    @abstractmethod
    async def update_capital_allocation(
        self, bot_id: str, new_amount: Decimal
    ) -> bool:
        """Update capital allocation for a bot."""


class IBotInstanceService(ABC):
    """Interface for bot instance management services."""

    @abstractmethod
    async def create_bot_instance(self, bot_config: BotConfiguration) -> str:
        """Create a new bot instance."""

    @abstractmethod
    async def start_bot(self, bot_id: str) -> bool:
        """Start a bot instance."""

    @abstractmethod
    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot instance."""

    @abstractmethod
    async def pause_bot(self, bot_id: str) -> bool:
        """Pause a bot instance."""

    @abstractmethod
    async def resume_bot(self, bot_id: str) -> bool:
        """Resume a bot instance."""

    @abstractmethod
    async def get_bot_state(self, bot_id: str) -> BotState:
        """Get bot state."""

    @abstractmethod
    async def execute_trade(
        self,
        bot_id: str,
        order_request: OrderRequest,
        execution_params: dict[str, Any],
    ) -> Any:
        """Execute a trade for a bot."""

    @abstractmethod
    async def update_position(
        self, bot_id: str, symbol: str, position_data: dict[str, Any]
    ) -> None:
        """Update position for a bot."""

    @abstractmethod
    async def close_position(self, bot_id: str, symbol: str, reason: str) -> bool:
        """Close position for a bot."""