"""
Resource manager for bot instances and system resources.

This module implements the ResourceManager class that handles allocation and
management of shared resources across all bot instances, including capital
allocation, API rate limits, database connections, and system resources.

CRITICAL: This integrates with P-010A (capital management), P-007 (rate limiting),
and P-002 (database) components.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

# MANDATORY: Import from P-010A (capital management)
from src.capital_management.capital_allocator import CapitalAllocator
from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotPriority, ResourceAllocation, ResourceType

# MANDATORY: Import from P-002 (database)
from src.database.connection import DatabaseConnectionManager

# MANDATORY: Import from P-002A (error handling)
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import log_calls, time_execution


class ResourceManager:
    """
    Central resource manager for bot instances.

    This class manages:
    - Capital allocation per bot
    - API rate limit distribution
    - Database connection pooling
    - Memory and CPU monitoring
    - Resource conflict resolution
    - Resource usage tracking and optimization
    """

    def __init__(self, config: Config):
        """
        Initialize resource manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.ResourceManager")
        self.error_handler = ErrorHandler(config.error_handling)

        # Core components
        self.capital_allocator = CapitalAllocator(config)
        self.db_connection = DatabaseConnectionManager(config)

        # Resource tracking
        self.resource_allocations: dict[str, dict[ResourceType, ResourceAllocation]] = {}
        self.global_resource_limits: dict[ResourceType, Decimal] = {}
        self.resource_usage_history: dict[ResourceType, list[dict[str, Any]]] = {}

        # Manager state
        self.is_running = False
        self.monitoring_task = None

        # Initialize global resource limits from config
        self._initialize_resource_limits()

        # Resource monitoring
        self.monitoring_interval = config.bot_management.get("resource_monitoring_interval", 30)
        self.resource_cleanup_interval = config.bot_management.get("resource_cleanup_interval", 300)

        self.logger.info("Resource manager initialized")

    def _initialize_resource_limits(self) -> None:
        """Initialize global resource limits from configuration."""
        resource_config = self.config.bot_management.get("resource_limits", {})

        self.global_resource_limits = {
            ResourceType.CAPITAL: Decimal(str(resource_config.get("total_capital", "1000000"))),
            ResourceType.API_RATE_LIMIT: Decimal(
                str(resource_config.get("total_api_calls_per_minute", "6000"))
            ),
            ResourceType.WEBSOCKET_CONNECTIONS: Decimal(
                str(resource_config.get("max_websocket_connections", "100"))
            ),
            ResourceType.DATABASE_CONNECTIONS: Decimal(
                str(resource_config.get("max_database_connections", "50"))
            ),
            ResourceType.CPU: Decimal(str(resource_config.get("max_cpu_percentage", "80"))),
            ResourceType.MEMORY: Decimal(str(resource_config.get("max_memory_mb", "8192"))),
        }

        # Initialize usage history
        for resource_type in ResourceType:
            self.resource_usage_history[resource_type] = []

    @log_calls
    async def start(self) -> None:
        """
        Start the resource manager.

        Raises:
            ExecutionError: If startup fails
        """
        try:
            if self.is_running:
                self.logger.warning("Resource manager is already running")
                return

            self.logger.info("Starting resource manager")

            # Initialize capital allocator
            await self.capital_allocator.initialize()

            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.is_running = True
            self.logger.info("Resource manager started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start resource manager: {e}")
            raise ExecutionError(f"Resource manager startup failed: {e}")

    @log_calls
    async def stop(self) -> None:
        """
        Stop the resource manager.

        Raises:
            ExecutionError: If shutdown fails
        """
        try:
            if not self.is_running:
                self.logger.warning("Resource manager is not running")
                return

            self.logger.info("Stopping resource manager")
            self.is_running = False

            # Stop monitoring task
            if self.monitoring_task:
                self.monitoring_task.cancel()

            # Release all allocated resources
            await self._release_all_resources()

            self.logger.info("Resource manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Failed to stop resource manager: {e}")
            raise ExecutionError(f"Resource manager shutdown failed: {e}")

    @time_execution
    @log_calls
    async def request_resources(
        self, bot_id: str, capital_amount: Decimal, priority: BotPriority = BotPriority.NORMAL
    ) -> bool:
        """
        Request resource allocation for a bot.

        Args:
            bot_id: Bot identifier
            capital_amount: Required capital amount
            priority: Bot priority for allocation

        Returns:
            bool: True if all resources allocated successfully

        Raises:
            ValidationError: If request is invalid
            ExecutionError: If allocation fails
        """
        try:
            if bot_id in self.resource_allocations:
                raise ValidationError(f"Resources already allocated for bot: {bot_id}")

            self.logger.info(
                "Processing resource request",
                bot_id=bot_id,
                capital_amount=float(capital_amount),
                priority=priority.value,
            )

            # Calculate resource requirements
            resource_requirements = await self._calculate_resource_requirements(
                bot_id, capital_amount, priority
            )

            # Check resource availability
            availability_check = await self._check_resource_availability(resource_requirements)
            if not availability_check["all_available"]:
                self.logger.warning(
                    "Insufficient resources available",
                    bot_id=bot_id,
                    unavailable=availability_check["unavailable_resources"],
                )
                return False

            # Allocate resources
            allocations = await self._allocate_resources(bot_id, resource_requirements)

            # Store allocations
            self.resource_allocations[bot_id] = allocations

            # Update usage tracking
            await self._update_resource_usage_tracking()

            self.logger.info(
                "Resources allocated successfully",
                bot_id=bot_id,
                allocated_capital=float(capital_amount),
                total_allocations=len(allocations),
            )

            return True

        except Exception as e:
            # Cleanup on failure
            await self._cleanup_failed_allocation(bot_id)
            self.logger.error(f"Failed to allocate resources: {e}", bot_id=bot_id)
            raise

    @log_calls
    async def release_resources(self, bot_id: str) -> bool:
        """
        Release all resources allocated to a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if release successful

        Raises:
            ExecutionError: If release fails
        """
        try:
            if bot_id not in self.resource_allocations:
                self.logger.warning("No resources allocated for bot", bot_id=bot_id)
                return True

            allocations = self.resource_allocations[bot_id]

            self.logger.info(
                "Releasing bot resources", bot_id=bot_id, allocation_count=len(allocations)
            )

            # Release each resource type
            for _resource_type, allocation in allocations.items():
                await self._release_resource_allocation(allocation)

            # Remove from tracking
            del self.resource_allocations[bot_id]

            # Update usage tracking
            await self._update_resource_usage_tracking()

            self.logger.info("Resources released successfully", bot_id=bot_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to release resources: {e}", bot_id=bot_id)
            raise ExecutionError(f"Resource release failed: {e}")

    @log_calls
    async def verify_resources(self, bot_id: str) -> bool:
        """
        Verify that allocated resources are still available and valid.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if resources are available
        """
        try:
            if bot_id not in self.resource_allocations:
                return False

            allocations = self.resource_allocations[bot_id]

            # Check each resource allocation
            for resource_type, allocation in allocations.items():
                if not await self._verify_resource_allocation(allocation):
                    self.logger.warning(
                        "Resource verification failed",
                        bot_id=bot_id,
                        resource_type=resource_type.value,
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Resource verification failed: {e}", bot_id=bot_id)
            return False

    @log_calls
    async def update_resource_usage(
        self, bot_id: str, resource_type: ResourceType, used_amount: Decimal
    ) -> None:
        """
        Update resource usage for a bot.

        Args:
            bot_id: Bot identifier
            resource_type: Type of resource
            used_amount: Current usage amount
        """
        try:
            if bot_id not in self.resource_allocations:
                return

            if resource_type not in self.resource_allocations[bot_id]:
                return

            allocation = self.resource_allocations[bot_id][resource_type]

            # Update usage
            allocation.used_amount = used_amount
            allocation.last_usage_time = datetime.now(timezone.utc)

            # Update peak usage
            if used_amount > allocation.peak_usage:
                allocation.peak_usage = used_amount

            # Calculate average usage (simplified rolling average)
            allocation.average_usage = (allocation.average_usage + used_amount) / 2

        except Exception as e:
            self.logger.warning(
                f"Failed to update resource usage: {e}",
                bot_id=bot_id,
                resource_type=resource_type.value,
            )

    async def get_resource_summary(self) -> dict[str, Any]:
        """Get comprehensive resource usage summary."""
        try:
            total_allocations = {}
            total_usage = {}

            # Calculate totals across all bots
            for resource_type in ResourceType:
                total_allocations[resource_type.value] = Decimal("0")
                total_usage[resource_type.value] = Decimal("0")

            for bot_allocations in self.resource_allocations.values():
                for resource_type, allocation in bot_allocations.items():
                    total_allocations[resource_type.value] += allocation.allocated_amount
                    total_usage[resource_type.value] += allocation.used_amount

            # Calculate usage percentages
            usage_percentages = {}
            for resource_type in ResourceType:
                limit = self.global_resource_limits[resource_type]
                allocated = total_allocations[resource_type.value]
                used = total_usage[resource_type.value]

                usage_percentages[resource_type.value] = {
                    "limit": float(limit),
                    "allocated": float(allocated),
                    "used": float(used),
                    "allocation_percentage": float(allocated / limit) if limit > 0 else 0.0,
                    "usage_percentage": float(used / limit) if limit > 0 else 0.0,
                }

            return {
                "total_bots": len(self.resource_allocations),
                "resource_limits": {
                    rt.value: float(limit) for rt, limit in self.global_resource_limits.items()
                },
                "resource_usage": usage_percentages,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to generate resource summary: {e}")
            return {"error": str(e)}

    async def get_bot_resource_usage(self, bot_id: str) -> dict[str, Any] | None:
        """
        Get resource usage for a specific bot.

        Args:
            bot_id: Bot identifier

        Returns:
            dict: Bot resource usage or None if not found
        """
        if bot_id not in self.resource_allocations:
            return None

        allocations = self.resource_allocations[bot_id]

        bot_usage = {}
        for resource_type, allocation in allocations.items():
            bot_usage[resource_type.value] = {
                "allocated": float(allocation.allocated_amount),
                "used": float(allocation.used_amount),
                "reserved": float(allocation.reserved_amount),
                "peak_usage": float(allocation.peak_usage),
                "average_usage": float(allocation.average_usage),
                "usage_percentage": (
                    float(allocation.used_amount / allocation.allocated_amount)
                    if allocation.allocated_amount > 0
                    else 0.0
                ),
                "last_usage": (
                    allocation.last_usage_time.isoformat() if allocation.last_usage_time else None
                ),
            }

        return bot_usage

    async def _calculate_resource_requirements(
        self, bot_id: str, capital_amount: Decimal, priority: BotPriority
    ) -> dict[ResourceType, Decimal]:
        """Calculate resource requirements for a bot."""
        # Base requirements calculation
        requirements = {}

        # Capital requirement (direct)
        requirements[ResourceType.CAPITAL] = capital_amount

        # API rate limit allocation based on capital and priority
        base_api_allocation = Decimal("100")  # Base API calls per minute
        priority_multiplier = {
            BotPriority.CRITICAL: Decimal("2.0"),
            BotPriority.HIGH: Decimal("1.5"),
            BotPriority.NORMAL: Decimal("1.0"),
            BotPriority.LOW: Decimal("0.5"),
        }

        requirements[ResourceType.API_RATE_LIMIT] = (
            base_api_allocation * priority_multiplier[priority]
        )

        # WebSocket connections (typically 1-3 per bot)
        requirements[ResourceType.WEBSOCKET_CONNECTIONS] = Decimal("2")

        # Database connections (1-2 per bot)
        requirements[ResourceType.DATABASE_CONNECTIONS] = Decimal("1")

        # CPU allocation (percentage, based on priority)
        cpu_allocation = {
            BotPriority.CRITICAL: Decimal("15"),
            BotPriority.HIGH: Decimal("10"),
            BotPriority.NORMAL: Decimal("5"),
            BotPriority.LOW: Decimal("2"),
        }
        requirements[ResourceType.CPU] = cpu_allocation[priority]

        # Memory allocation (MB, based on priority and complexity)
        memory_allocation = {
            BotPriority.CRITICAL: Decimal("512"),
            BotPriority.HIGH: Decimal("256"),
            BotPriority.NORMAL: Decimal("128"),
            BotPriority.LOW: Decimal("64"),
        }
        requirements[ResourceType.MEMORY] = memory_allocation[priority]

        return requirements

    async def _check_resource_availability(
        self, requirements: dict[ResourceType, Decimal]
    ) -> dict[str, Any]:
        """Check if requested resources are available."""
        unavailable_resources = []

        # Calculate current usage
        current_usage = {}
        for resource_type in ResourceType:
            current_usage[resource_type] = Decimal("0")

        for bot_allocations in self.resource_allocations.values():
            for resource_type, allocation in bot_allocations.items():
                current_usage[resource_type] += allocation.allocated_amount

        # Check each requirement
        for resource_type, required_amount in requirements.items():
            limit = self.global_resource_limits[resource_type]
            used = current_usage[resource_type]
            available = limit - used

            if required_amount > available:
                unavailable_resources.append(
                    {
                        "resource_type": resource_type.value,
                        "required": float(required_amount),
                        "available": float(available),
                        "limit": float(limit),
                        "current_usage": float(used),
                    }
                )

        return {
            "all_available": len(unavailable_resources) == 0,
            "unavailable_resources": unavailable_resources,
        }

    async def _allocate_resources(
        self, bot_id: str, requirements: dict[ResourceType, Decimal]
    ) -> dict[ResourceType, ResourceAllocation]:
        """Allocate resources for a bot."""
        allocations = {}

        for resource_type, amount in requirements.items():
            allocation = ResourceAllocation(
                bot_id=bot_id,
                resource_type=resource_type,
                allocated_amount=amount,
                max_amount=amount * Decimal("1.2"),  # 20% buffer
                created_at=datetime.now(timezone.utc),
            )

            # Special handling for capital allocation
            if resource_type == ResourceType.CAPITAL:
                success = await self.capital_allocator.allocate_capital(
                    bot_id, amount, "bot_resource_allocation"
                )
                if not success:
                    raise ExecutionError(f"Failed to allocate capital: {amount}")

            allocations[resource_type] = allocation

        return allocations

    async def _release_resource_allocation(self, allocation: ResourceAllocation) -> None:
        """Release a specific resource allocation."""
        try:
            # Special handling for capital release
            if allocation.resource_type == ResourceType.CAPITAL:
                await self.capital_allocator.release_capital(
                    allocation.bot_id, allocation.allocated_amount
                )

            self.logger.debug(
                "Resource allocation released",
                bot_id=allocation.bot_id,
                resource_type=allocation.resource_type.value,
                amount=float(allocation.allocated_amount),
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to release resource allocation: {e}",
                bot_id=allocation.bot_id,
                resource_type=allocation.resource_type.value,
            )

    async def _verify_resource_allocation(self, allocation: ResourceAllocation) -> bool:
        """Verify that a resource allocation is still valid."""
        try:
            # Check if allocation has expired
            if allocation.expires_at and datetime.now(timezone.utc) > allocation.expires_at:
                return False

            # Check if usage is within limits
            if allocation.max_amount and allocation.used_amount > allocation.max_amount:
                return False

            # Special verification for capital
            if allocation.resource_type == ResourceType.CAPITAL:
                # Verify capital is still allocated
                available_capital = await self.capital_allocator.get_available_capital(
                    allocation.bot_id
                )
                if available_capital < allocation.allocated_amount:
                    return False

            return True

        except Exception as e:
            self.logger.warning(
                f"Resource verification error: {e}",
                bot_id=allocation.bot_id,
                resource_type=allocation.resource_type.value,
            )
            return False

    async def _monitoring_loop(self) -> None:
        """Resource monitoring and optimization loop."""
        try:
            cleanup_counter = 0

            while self.is_running:
                try:
                    # Update resource usage tracking
                    await self._update_resource_usage_tracking()

                    # Check for resource violations
                    await self._check_resource_violations()

                    # Optimize resource allocations
                    await self._optimize_resource_allocations()

                    # Periodic cleanup
                    cleanup_counter += 1
                    if cleanup_counter >= (
                        self.resource_cleanup_interval // self.monitoring_interval
                    ):
                        await self._cleanup_expired_allocations()
                        cleanup_counter = 0

                    # Wait for next cycle
                    await asyncio.sleep(self.monitoring_interval)

                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    await asyncio.sleep(10)

        except asyncio.CancelledError:
            self.logger.info("Resource monitoring cancelled")

    async def _update_resource_usage_tracking(self) -> None:
        """Update resource usage tracking and history."""
        current_time = datetime.now(timezone.utc)

        for resource_type in ResourceType:
            total_allocated = Decimal("0")
            total_used = Decimal("0")

            for bot_allocations in self.resource_allocations.values():
                if resource_type in bot_allocations:
                    allocation = bot_allocations[resource_type]
                    total_allocated += allocation.allocated_amount
                    total_used += allocation.used_amount

            # Add to history
            usage_entry = {
                "timestamp": current_time,
                "total_allocated": total_allocated,
                "total_used": total_used,
                "usage_percentage": (
                    float(total_used / self.global_resource_limits[resource_type])
                    if self.global_resource_limits[resource_type] > 0
                    else 0.0
                ),
            }

            # Keep only recent history (last 24 hours)
            history = self.resource_usage_history[resource_type]
            history.append(usage_entry)

            # Cleanup old entries
            cutoff_time = current_time - timedelta(hours=24)
            self.resource_usage_history[resource_type] = [
                entry for entry in history if entry["timestamp"] > cutoff_time
            ]

    async def _check_resource_violations(self) -> None:
        """Check for resource usage violations."""
        violations = []

        for resource_type in ResourceType:
            limit = self.global_resource_limits[resource_type]
            total_used = Decimal("0")

            for bot_allocations in self.resource_allocations.values():
                if resource_type in bot_allocations:
                    total_used += bot_allocations[resource_type].used_amount

            usage_percentage = float(total_used / limit) if limit > 0 else 0.0

            if usage_percentage > 0.9:  # 90% threshold
                violations.append(
                    {
                        "resource_type": resource_type.value,
                        "usage_percentage": usage_percentage,
                        "total_used": float(total_used),
                        "limit": float(limit),
                    }
                )

        if violations:
            self.logger.warning("Resource usage violations detected", violations=violations)

    async def _optimize_resource_allocations(self) -> None:
        """Optimize resource allocations based on usage patterns."""
        # Implementation would analyze usage patterns and optimize allocations
        # For now, just log optimization opportunities

        optimization_suggestions = []

        for bot_id, bot_allocations in self.resource_allocations.items():
            for resource_type, allocation in bot_allocations.items():
                if allocation.allocated_amount > 0:
                    utilization = allocation.average_usage / allocation.allocated_amount

                    if utilization < 0.2:  # Under-utilized
                        optimization_suggestions.append(
                            {
                                "bot_id": bot_id,
                                "resource_type": resource_type.value,
                                "suggestion": "reduce_allocation",
                                "current_utilization": float(utilization),
                            }
                        )
                    elif utilization > 0.8:  # Over-utilized
                        optimization_suggestions.append(
                            {
                                "bot_id": bot_id,
                                "resource_type": resource_type.value,
                                "suggestion": "increase_allocation",
                                "current_utilization": float(utilization),
                            }
                        )

        if optimization_suggestions:
            self.logger.debug(
                "Resource optimization suggestions available", suggestions=optimization_suggestions
            )

    async def _cleanup_expired_allocations(self) -> None:
        """Cleanup expired resource allocations."""
        current_time = datetime.now(timezone.utc)
        expired_bots = []

        for bot_id, bot_allocations in self.resource_allocations.items():
            for allocation in bot_allocations.values():
                if allocation.expires_at and current_time > allocation.expires_at:
                    expired_bots.append(bot_id)
                    break

        for bot_id in expired_bots:
            self.logger.info("Cleaning up expired resource allocation", bot_id=bot_id)
            await self.release_resources(bot_id)

    async def _release_all_resources(self) -> None:
        """Release all allocated resources during shutdown."""
        bot_ids = list(self.resource_allocations.keys())

        for bot_id in bot_ids:
            try:
                await self.release_resources(bot_id)
            except Exception as e:
                self.logger.warning(
                    f"Failed to release resources during shutdown: {e}", bot_id=bot_id
                )

    async def _cleanup_failed_allocation(self, bot_id: str) -> None:
        """Cleanup after failed resource allocation."""
        if bot_id in self.resource_allocations:
            try:
                await self.release_resources(bot_id)
            except Exception as e:
                self.logger.warning(
                    f"Failed to cleanup after allocation failure: {e}", bot_id=bot_id
                )
