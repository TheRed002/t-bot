"""
Resource Management Service for bot resource allocation and monitoring.

This service replaces the direct database and capital management access in the
original resource_manager.py with proper service layer patterns.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError
from src.core.types import BotPriority


class ResourceManagementService(BaseService):
    """
    Comprehensive resource management service using service layer pattern.

    This service provides:
    - Resource allocation and tracking for bot instances
    - Capital allocation management
    - API rate limit distribution
    - Resource usage monitoring and optimization
    - Resource conflict detection and resolution
    """

    def __init__(self):
        """
        Initialize resource management service.

        Note: Dependencies are resolved during startup via dependency injection.
        """
        super().__init__(name="ResourceManagementService")

        # Declare required service dependencies for DI resolution
        self.add_dependency("ConfigService")
        self.add_dependency("DatabaseService")
        self.add_dependency("CapitalService")

        # Service instances (resolved during startup via DI)
        self._config_service = None
        self._database_service = None
        self._capital_service = None

        # Resource tracking
        self._resource_allocations: dict[str, dict[str, Any]] = {}
        self._resource_reservations: dict[str, dict[str, Any]] = {}
        self._resource_usage_tracking: dict[str, dict[str, Any]] = {}

        # Global resource limits
        self._global_resource_limits: dict[str, Decimal] = {}

        # Monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 30  # seconds
        self._cleanup_interval = 300  # seconds

        self._logger.info("ResourceManagementService initialized")

    async def _do_start(self) -> None:
        """Start the resource management service and resolve dependencies."""
        # Resolve service dependencies through DI container
        self._config_service = self.resolve_dependency("ConfigService")
        self._database_service = self.resolve_dependency("DatabaseService")
        self._capital_service = self.resolve_dependency("CapitalService")

        # Verify critical dependencies are available
        if not all([self._config_service, self._database_service, self._capital_service]):
            raise ServiceError("Failed to resolve required service dependencies")

        # Load configuration
        await self._load_configuration()

        # Initialize resource limits
        await self._initialize_resource_limits()

        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        self._logger.info("ResourceManagementService started successfully")

    async def _do_stop(self) -> None:
        """Stop the resource management service."""
        # Cancel monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Release all allocated resources
        await self._release_all_resources()

        # Clear tracking
        self._resource_allocations.clear()
        self._resource_reservations.clear()
        self._resource_usage_tracking.clear()

        self._logger.info("ResourceManagementService stopped successfully")

    async def _load_configuration(self) -> None:
        """Load resource management configuration."""
        try:
            config = self._config_service.get_config().get("resource_management", {})

            self._monitoring_interval = config.get("monitoring_interval", 30)
            self._cleanup_interval = config.get("cleanup_interval", 300)

            self._logger.debug("Resource management configuration loaded")

        except Exception as e:
            self._logger.warning(f"Failed to load configuration, using defaults: {e}")

    async def _initialize_resource_limits(self) -> None:
        """Initialize global resource limits."""
        try:
            config = self._config_service.get_config().get("resource_limits", {})

            self._global_resource_limits = {
                "capital": Decimal(str(config.get("total_capital", "1000000"))),
                "api_rate_limit": Decimal(str(config.get("total_api_calls_per_minute", "6000"))),
                "websocket_connections": Decimal(
                    str(config.get("max_websocket_connections", "100"))
                ),
                "database_connections": Decimal(str(config.get("max_database_connections", "50"))),
                "cpu_percentage": Decimal(str(config.get("max_cpu_percentage", "80"))),
                "memory_mb": Decimal(str(config.get("max_memory_mb", "8192"))),
            }

            self._logger.debug("Resource limits initialized", limits=self._global_resource_limits)

        except Exception as e:
            self._logger.warning(f"Failed to load resource limits, using defaults: {e}")
            # Set default limits
            self._global_resource_limits = {
                "capital": Decimal("1000000"),
                "api_rate_limit": Decimal("6000"),
                "websocket_connections": Decimal("100"),
                "database_connections": Decimal("50"),
                "cpu_percentage": Decimal("80"),
                "memory_mb": Decimal("8192"),
            }

    # Resource Allocation

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
        """
        return await self.execute_with_monitoring(
            "request_resources", self._request_resources_impl, bot_id, capital_amount, priority
        )

    async def _request_resources_impl(
        self, bot_id: str, capital_amount: Decimal, priority: BotPriority
    ) -> bool:
        """Implementation of resource allocation request."""
        if bot_id in self._resource_allocations:
            raise ServiceError(f"Resources already allocated for bot: {bot_id}")

        self._logger.info(
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
            # Handle high priority override
            if priority in [BotPriority.HIGH, BotPriority.CRITICAL]:
                reallocation_success = await self._reallocate_for_high_priority(
                    bot_id, resource_requirements, priority
                )
                if not reallocation_success:
                    self._logger.warning(
                        "High priority reallocation failed",
                        bot_id=bot_id,
                        unavailable=availability_check["unavailable_resources"],
                    )
                    return False
            else:
                self._logger.warning(
                    "Insufficient resources available",
                    bot_id=bot_id,
                    unavailable=availability_check["unavailable_resources"],
                )
                return False

        # Allocate resources
        allocations = await self._allocate_resources(bot_id, resource_requirements, priority)

        # Store allocations
        self._resource_allocations[bot_id] = allocations

        # Store allocation record in database
        allocation_record = {
            "bot_id": bot_id,
            "allocations": allocations,
            "priority": priority.value,
            "allocated_at": datetime.now(timezone.utc),
            "status": "active",
        }

        await self._database_service.store_resource_allocation(allocation_record)

        self._logger.info(
            "Resources allocated successfully",
            bot_id=bot_id,
            allocated_capital=float(capital_amount),
            total_allocations=len(allocations),
        )

        return True

    async def release_resources(self, bot_id: str) -> bool:
        """
        Release all resources allocated to a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if release successful
        """
        return await self.execute_with_monitoring(
            "release_resources", self._release_resources_impl, bot_id
        )

    async def _release_resources_impl(self, bot_id: str) -> bool:
        """Implementation of resource release."""
        if bot_id not in self._resource_allocations:
            self._logger.warning("No resources allocated for bot", bot_id=bot_id)
            return False

        allocations = self._resource_allocations[bot_id]

        self._logger.info(
            "Releasing bot resources", bot_id=bot_id, allocation_count=len(allocations)
        )

        # Release capital through capital service
        if "capital" in allocations:
            capital_amount = allocations["capital"]["allocated_amount"]
            capital_released = await self._capital_service.release_capital(bot_id, capital_amount)
            if not capital_released:
                self._logger.warning("Failed to release capital", bot_id=bot_id)

        # Release other resources (API limits, connections, etc.)
        for resource_type, allocation in allocations.items():
            if resource_type != "capital":
                await self._release_resource_allocation(bot_id, resource_type, allocation)

        # Remove from tracking
        del self._resource_allocations[bot_id]

        # Update database
        await self._database_service.update_resource_allocation_status(bot_id, "released")

        # Remove from usage tracking
        if bot_id in self._resource_usage_tracking:
            del self._resource_usage_tracking[bot_id]

        self._logger.info("Resources released successfully", bot_id=bot_id)
        return True

    async def verify_resources(self, bot_id: str) -> bool:
        """
        Verify that allocated resources are still available and valid.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if resources are available
        """
        return await self.execute_with_monitoring(
            "verify_resources", self._verify_resources_impl, bot_id
        )

    async def _verify_resources_impl(self, bot_id: str) -> bool:
        """Implementation of resource verification."""
        if bot_id not in self._resource_allocations:
            return False

        allocations = self._resource_allocations[bot_id]

        # Check each resource allocation
        for resource_type, allocation in allocations.items():
            if not await self._verify_resource_allocation(bot_id, resource_type, allocation):
                self._logger.warning(
                    "Resource verification failed",
                    bot_id=bot_id,
                    resource_type=resource_type,
                )
                return False

        return True

    # Resource Usage Tracking

    async def update_resource_usage(self, bot_id: str, usage_data: dict[str, float]) -> bool:
        """
        Update resource usage for a bot.

        Args:
            bot_id: Bot identifier
            usage_data: Dictionary of resource usage data

        Returns:
            bool: True if updated successfully
        """
        return await self.execute_with_monitoring(
            "update_resource_usage", self._update_resource_usage_impl, bot_id, usage_data
        )

    async def _update_resource_usage_impl(self, bot_id: str, usage_data: dict[str, float]) -> bool:
        """Implementation of resource usage update."""
        if bot_id not in self._resource_allocations:
            self._logger.warning("Cannot update usage for non-allocated bot", bot_id=bot_id)
            return False

        # Update local tracking
        if bot_id not in self._resource_usage_tracking:
            self._resource_usage_tracking[bot_id] = {}

        self._resource_usage_tracking[bot_id].update(usage_data)
        self._resource_usage_tracking[bot_id]["last_updated"] = datetime.now(timezone.utc)

        # Store usage data in database
        usage_record = {"bot_id": bot_id, "timestamp": datetime.now(timezone.utc), **usage_data}

        await self._database_service.store_resource_usage(usage_record)

        # Check for resource violations
        violations = await self._check_resource_violations(bot_id, usage_data)
        if violations:
            self._logger.warning(
                "Resource violations detected", bot_id=bot_id, violations=violations
            )

        return True

    # Resource Reservations

    async def reserve_resources(
        self, bot_id: str, amount: Decimal, priority: BotPriority, duration_minutes: int = 60
    ) -> str | None:
        """
        Reserve resources for future use.

        Args:
            bot_id: Bot identifier
            amount: Amount to reserve
            priority: Priority level
            duration_minutes: Reservation duration in minutes

        Returns:
            str: Reservation ID if successful, None otherwise
        """
        return await self.execute_with_monitoring(
            "reserve_resources",
            self._reserve_resources_impl,
            bot_id,
            amount,
            priority,
            duration_minutes,
        )

    async def _reserve_resources_impl(
        self, bot_id: str, amount: Decimal, priority: BotPriority, duration_minutes: int
    ) -> str | None:
        """Implementation of resource reservation."""
        # Check if enough capital is available
        available_capital = await self._capital_service.get_available_capital()
        if available_capital < amount:
            self._logger.warning(
                "Insufficient capital for reservation",
                bot_id=bot_id,
                requested=float(amount),
                available=float(available_capital),
            )
            return None

        # Generate reservation ID
        reservation_id = f"res_{bot_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Create reservation
        expiry_time = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)

        reservation = {
            "reservation_id": reservation_id,
            "bot_id": bot_id,
            "resource_type": "capital",
            "amount": amount,
            "priority": priority.value,
            "created_at": datetime.now(timezone.utc),
            "expires_at": expiry_time,
            "status": "active",
        }

        # Store reservation
        self._resource_reservations[reservation_id] = reservation

        # Reserve capital through capital service
        reservation_success = await self._capital_service.reserve_capital(
            bot_id, amount, duration_minutes
        )

        if not reservation_success:
            # Remove failed reservation
            del self._resource_reservations[reservation_id]
            return None

        # Store reservation in database
        await self._database_service.store_resource_reservation(reservation)

        self._logger.info(
            "Resource reservation created",
            reservation_id=reservation_id,
            bot_id=bot_id,
            amount=float(amount),
            duration_minutes=duration_minutes,
        )

        return reservation_id

    # Resource Analysis and Optimization

    async def get_resource_summary(self) -> dict[str, Any]:
        """Get comprehensive resource usage summary."""
        return await self.execute_with_monitoring(
            "get_resource_summary", self._get_resource_summary_impl
        )

    async def _get_resource_summary_impl(self) -> dict[str, Any]:
        """Implementation of get resource summary."""
        # Get capital summary from capital service
        capital_summary = await self._capital_service.get_capital_summary()

        # Calculate resource utilization
        resource_utilization = {}

        for resource_type, limit in self._global_resource_limits.items():
            total_allocated = Decimal("0")

            for _bot_id, allocations in self._resource_allocations.items():
                if resource_type in allocations:
                    allocation = allocations[resource_type]
                    total_allocated += Decimal(str(allocation.get("allocated_amount", 0)))

            utilization_percentage = float(total_allocated / limit * 100) if limit > 0 else 0.0

            resource_utilization[resource_type] = {
                "total_limit": float(limit),
                "total_allocated": float(total_allocated),
                "available": float(limit - total_allocated),
                "utilization_percentage": utilization_percentage,
            }

        # Bot allocations summary
        bot_allocations_summary = {}
        for bot_id, allocations in self._resource_allocations.items():
            bot_allocations_summary[bot_id] = {}
            for resource_type, allocation in allocations.items():
                bot_allocations_summary[bot_id][resource_type] = {
                    "allocated": allocation.get("allocated_amount", 0),
                    "used": allocation.get("used_amount", 0),
                    "utilization": (
                        allocation.get("used_amount", 0)
                        / allocation.get("allocated_amount", 1)
                        * 100
                        if allocation.get("allocated_amount", 0) > 0
                        else 0.0
                    ),
                }

        # System health indicators
        total_bots = len(self._resource_allocations)
        capital_utilization = resource_utilization.get("capital", {}).get(
            "utilization_percentage", 0
        )
        system_health = {
            "active_bots": total_bots,
            "healthy": capital_utilization < 90.0,
            "status": "healthy" if capital_utilization < 90.0 else "warning",
        }

        return {
            "capital_management": capital_summary,
            "resource_utilization": resource_utilization,
            "bot_allocations": bot_allocations_summary,
            "system_health": system_health,
            "active_reservations": len(self._resource_reservations),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    async def get_optimization_suggestions(self) -> list[dict[str, Any]]:
        """Get resource optimization suggestions."""
        return await self.execute_with_monitoring(
            "get_optimization_suggestions", self._get_optimization_suggestions_impl
        )

    async def _get_optimization_suggestions_impl(self) -> list[dict[str, Any]]:
        """Implementation of get optimization suggestions."""
        suggestions = []

        for bot_id, allocations in self._resource_allocations.items():
            for resource_type, allocation in allocations.items():
                allocated_amount = allocation.get("allocated_amount", 0)
                used_amount = allocation.get("used_amount", 0)

                if allocated_amount > 0:
                    utilization = used_amount / allocated_amount

                    if utilization < 0.2:  # Under-utilized
                        suggestions.append(
                            {
                                "bot_id": bot_id,
                                "resource_type": resource_type,
                                "suggestion": "reduce_allocation",
                                "current_utilization": float(utilization),
                                "allocated_amount": float(allocated_amount),
                                "used_amount": float(used_amount),
                            }
                        )
                    elif utilization > 0.8:  # Over-utilized
                        suggestions.append(
                            {
                                "bot_id": bot_id,
                                "resource_type": resource_type,
                                "suggestion": "increase_allocation",
                                "current_utilization": float(utilization),
                                "allocated_amount": float(allocated_amount),
                                "used_amount": float(used_amount),
                            }
                        )

        return suggestions

    async def detect_resource_conflicts(self) -> list[dict[str, Any]]:
        """Detect resource conflicts between bot allocations."""
        return await self.execute_with_monitoring(
            "detect_resource_conflicts", self._detect_resource_conflicts_impl
        )

    async def _detect_resource_conflicts_impl(self) -> list[dict[str, Any]]:
        """Implementation of resource conflict detection."""
        conflicts = []

        # Check for over-allocation across all resources
        for resource_type, limit in self._global_resource_limits.items():
            total_allocated = Decimal("0")

            for allocations in self._resource_allocations.values():
                if resource_type in allocations:
                    total_allocated += Decimal(
                        str(allocations[resource_type].get("allocated_amount", 0))
                    )

            if total_allocated > limit:
                conflicts.append(
                    {
                        "resource_type": resource_type,
                        "total_allocated": float(total_allocated),
                        "limit": float(limit),
                        "over_allocation": float(total_allocated - limit),
                    }
                )

        return conflicts

    # Utility Methods

    async def _calculate_resource_requirements(
        self, bot_id: str, capital_amount: Decimal, priority: BotPriority
    ) -> dict[str, Decimal]:
        """Calculate resource requirements for a bot."""
        requirements = {}

        # Capital requirement (direct)
        requirements["capital"] = capital_amount

        # API rate limit allocation based on priority
        base_api_allocation = Decimal("100")  # Base API calls per minute
        priority_multipliers = {
            BotPriority.CRITICAL: Decimal("2.0"),
            BotPriority.HIGH: Decimal("1.5"),
            BotPriority.NORMAL: Decimal("1.0"),
            BotPriority.LOW: Decimal("0.5"),
        }

        requirements["api_rate_limit"] = base_api_allocation * priority_multipliers[priority]

        # WebSocket connections (typically 1-3 per bot)
        requirements["websocket_connections"] = Decimal("2")

        # Database connections (1-2 per bot)
        requirements["database_connections"] = Decimal("1")

        # CPU allocation (percentage, based on priority)
        cpu_allocations = {
            BotPriority.CRITICAL: Decimal("15"),
            BotPriority.HIGH: Decimal("10"),
            BotPriority.NORMAL: Decimal("5"),
            BotPriority.LOW: Decimal("2"),
        }
        requirements["cpu_percentage"] = cpu_allocations[priority]

        # Memory allocation (MB, based on priority)
        memory_allocations = {
            BotPriority.CRITICAL: Decimal("512"),
            BotPriority.HIGH: Decimal("256"),
            BotPriority.NORMAL: Decimal("128"),
            BotPriority.LOW: Decimal("64"),
        }
        requirements["memory_mb"] = memory_allocations[priority]

        return requirements

    async def _check_resource_availability(
        self, requirements: dict[str, Decimal]
    ) -> dict[str, Any]:
        """Check if requested resources are available."""
        unavailable_resources = []

        # Calculate current usage
        current_usage = {}
        for resource_type in self._global_resource_limits.keys():
            current_usage[resource_type] = Decimal("0")

        for allocations in self._resource_allocations.values():
            for resource_type, allocation in allocations.items():
                if resource_type in current_usage:
                    current_usage[resource_type] += Decimal(
                        str(allocation.get("allocated_amount", 0))
                    )

        # Check each requirement
        for resource_type, required_amount in requirements.items():
            if resource_type in self._global_resource_limits:
                limit = self._global_resource_limits[resource_type]
                used = current_usage.get(resource_type, Decimal("0"))
                available = limit - used

                if required_amount > available:
                    unavailable_resources.append(
                        {
                            "resource_type": resource_type,
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
        self, bot_id: str, requirements: dict[str, Decimal], priority: BotPriority
    ) -> dict[str, Any]:
        """Allocate resources for a bot."""
        allocations = {}

        for resource_type, amount in requirements.items():
            allocation = {
                "bot_id": bot_id,
                "resource_type": resource_type,
                "allocated_amount": amount,
                "max_amount": amount * Decimal("1.2"),  # 20% buffer
                "used_amount": Decimal("0"),
                "created_at": datetime.now(timezone.utc),
                "priority": priority.value,
            }

            # Special handling for capital allocation
            if resource_type == "capital":
                capital_allocated = await self._capital_service.allocate_capital(
                    bot_id, amount, priority
                )
                if not capital_allocated:
                    raise ServiceError(f"Failed to allocate capital: {amount}")

            allocations[resource_type] = allocation

        return allocations

    async def _reallocate_for_high_priority(
        self, bot_id: str, requirements: dict[str, Decimal], priority: BotPriority
    ) -> bool:
        """Attempt to reallocate resources for high priority requests."""
        # For capital, allow using emergency reserve for high priority
        if "capital" in requirements:
            capital_needed = requirements["capital"]

            # Try high priority allocation through capital service
            # Use regular allocation with priority flag
            emergency_success = await self._capital_service.allocate_capital(
                bot_id, capital_needed, priority.value
            )

            if emergency_success:
                self._logger.info(
                    "High priority capital allocation approved using emergency reserve",
                    bot_id=bot_id,
                    capital_needed=float(capital_needed),
                    priority=priority.value,
                )
                return True

        return False

    async def _release_resource_allocation(
        self, bot_id: str, resource_type: str, allocation: dict[str, Any]
    ) -> None:
        """Release a specific resource allocation."""
        try:
            self._logger.debug(
                "Resource allocation released",
                bot_id=bot_id,
                resource_type=resource_type,
                amount=allocation.get("allocated_amount", 0),
            )
        except Exception as e:
            self._logger.warning(
                f"Failed to release resource allocation: {e}",
                bot_id=bot_id,
                resource_type=resource_type,
            )

    async def _verify_resource_allocation(
        self, bot_id: str, resource_type: str, allocation: dict[str, Any]
    ) -> bool:
        """Verify that a resource allocation is still valid."""
        try:
            # Check if allocation has expired
            expires_at = allocation.get("expires_at")
            if expires_at and datetime.now(timezone.utc) > expires_at:
                return False

            # Check if usage is within limits
            max_amount = allocation.get("max_amount")
            used_amount = allocation.get("used_amount", 0)
            if max_amount and used_amount > max_amount:
                return False

            # Special verification for capital
            if resource_type == "capital":
                # Verify capital is still allocated through capital service
                allocated_amount = allocation.get("allocated_amount", 0)
                available_capital = await self._capital_service.get_bot_available_capital(bot_id)
                if available_capital < allocated_amount:
                    return False

            return True

        except Exception as e:
            self._logger.warning(
                f"Resource verification error: {e}",
                bot_id=bot_id,
                resource_type=resource_type,
            )
            return False

    async def _check_resource_violations(
        self, bot_id: str, usage_data: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Check for resource usage violations."""
        violations = []

        if bot_id not in self._resource_allocations:
            return violations

        allocations = self._resource_allocations[bot_id]

        for resource_type, usage_value in usage_data.items():
            if resource_type in allocations:
                allocation = allocations[resource_type]
                allocated_amount = allocation.get("allocated_amount", 0)

                if allocated_amount > 0:
                    usage_percentage = usage_value / allocated_amount

                    if usage_percentage > 1.0:  # Over 100% usage
                        violations.append(
                            {
                                "resource_type": resource_type,
                                "usage_percentage": usage_percentage * 100,
                                "usage_value": usage_value,
                                "allocated_amount": allocated_amount,
                                "severity": "critical" if usage_percentage > 1.5 else "warning",
                            }
                        )

        return violations

    async def _release_all_resources(self) -> None:
        """Release all allocated resources during shutdown."""
        bot_ids = list(self._resource_allocations.keys())

        for bot_id in bot_ids:
            try:
                await self._release_resources_impl(bot_id)
            except Exception as e:
                self._logger.warning(
                    f"Failed to release resources during shutdown: {e}", bot_id=bot_id
                )

    async def _cleanup_expired_reservations(self) -> int:
        """Clean up expired resource reservations."""
        cleaned_count = 0
        current_time = datetime.now(timezone.utc)

        expired_reservations = []

        for reservation_id, reservation in self._resource_reservations.items():
            expires_at = reservation.get("expires_at")
            if expires_at and current_time > expires_at:
                expired_reservations.append(reservation_id)

        for reservation_id in expired_reservations:
            reservation = self._resource_reservations[reservation_id]

            # Release capital reservation
            if reservation["resource_type"] == "capital":
                await self._capital_service.release_reserved_capital(
                    reservation["bot_id"], reservation["amount"]
                )

            # Update database
            await self._database_service.update_resource_reservation_status(
                reservation_id, "expired"
            )

            del self._resource_reservations[reservation_id]
            cleaned_count += 1

            self._logger.info(
                "Expired reservation cleaned up",
                reservation_id=reservation_id,
                bot_id=reservation["bot_id"],
            )

        return cleaned_count

    # Monitoring Loop

    async def _monitoring_loop(self) -> None:
        """Resource monitoring and optimization loop."""
        try:
            cleanup_counter = 0

            while self.is_running:
                try:
                    # Update resource usage tracking
                    await self._update_resource_usage_tracking()

                    # Check for resource violations
                    await self._check_all_resource_violations()

                    # Optimize resource allocations
                    await self._optimize_resource_allocations()

                    # Periodic cleanup
                    cleanup_counter += 1
                    if cleanup_counter >= (self._cleanup_interval // self._monitoring_interval):
                        await self._cleanup_expired_reservations()
                        cleanup_counter = 0

                    await asyncio.sleep(self._monitoring_interval)

                except Exception as e:
                    self._logger.error(f"Resource monitoring error: {e}")
                    await asyncio.sleep(10)

        except asyncio.CancelledError:
            self._logger.info("Resource monitoring cancelled")

    async def _update_resource_usage_tracking(self) -> None:
        """Update resource usage tracking and history."""
        current_time = datetime.now(timezone.utc)

        # Update usage tracking for all resources
        for resource_type, limit in self._global_resource_limits.items():
            total_allocated = Decimal("0")
            total_used = Decimal("0")

            for allocations in self._resource_allocations.values():
                if resource_type in allocations:
                    allocation = allocations[resource_type]
                    total_allocated += Decimal(str(allocation.get("allocated_amount", 0)))
                    total_used += Decimal(str(allocation.get("used_amount", 0)))

            # Store usage history in database
            usage_entry = {
                "resource_type": resource_type,
                "timestamp": current_time,
                "total_allocated": total_allocated,
                "total_used": total_used,
                "usage_percentage": (float(total_used / limit) * 100 if limit > 0 else 0.0),
            }

            await self._database_service.store_resource_usage_history(usage_entry)

    async def _check_all_resource_violations(self) -> None:
        """Check for resource usage violations across all bots."""
        violations = []

        for resource_type, limit in self._global_resource_limits.items():
            total_used = Decimal("0")

            for allocations in self._resource_allocations.values():
                if resource_type in allocations:
                    total_used += Decimal(str(allocations[resource_type].get("used_amount", 0)))

            usage_percentage = float(total_used / limit) * 100 if limit > 0 else 0.0

            if usage_percentage > 90:  # 90% threshold
                violations.append(
                    {
                        "resource_type": resource_type,
                        "usage_percentage": usage_percentage,
                        "total_used": float(total_used),
                        "limit": float(limit),
                    }
                )

        if violations:
            self._logger.warning("Global resource usage violations detected", violations=violations)

    async def _optimize_resource_allocations(self) -> None:
        """Optimize resource allocations based on usage patterns."""
        # Get optimization suggestions
        suggestions = await self._get_optimization_suggestions_impl()

        if suggestions:
            self._logger.debug(
                "Resource optimization suggestions available", suggestions_count=len(suggestions)
            )

            # Store suggestions in database for potential automatic optimization
            for suggestion in suggestions:
                await self._database_service.store_optimization_suggestion(suggestion)

    # Service-specific health check
    async def _service_health_check(self) -> Any:
        """Service-specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check if required services are healthy
            services_to_check = [self._database_service, self._capital_service]

            for service in services_to_check:
                if not service:
                    return HealthStatus.UNHEALTHY

                if hasattr(service, "health_check"):
                    health_result = await service.health_check()
                    if health_result.get("status") != "healthy":
                        return HealthStatus.DEGRADED

            # Check resource utilization
            resource_summary = await self._get_resource_summary_impl()
            capital_utilization = (
                resource_summary.get("resource_utilization", {})
                .get("capital", {})
                .get("utilization_percentage", 0)
            )

            if capital_utilization > 95:
                return HealthStatus.DEGRADED
            elif capital_utilization > 85:
                return HealthStatus.DEGRADED

            # Check for active resource conflicts
            conflicts = await self._detect_resource_conflicts_impl()
            if conflicts:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(f"Service health check failed: {e}")
            return HealthStatus.UNHEALTHY
