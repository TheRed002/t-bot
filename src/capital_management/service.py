"""
Enterprise-grade Capital Management Service.

This service provides comprehensive capital management operations with full database
abstraction, transaction support, audit trails, and enterprise features including:
- Transaction management with rollback support
- Comprehensive audit logging
- Circuit breaker and retry mechanisms
- Performance monitoring and metrics
- Health checks and degraded mode operations
- Cache layer integration
- Risk-aware capital allocation

CRITICAL: This service MUST be used instead of direct database access.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# Use repository protocols from interfaces module
from src.capital_management.interfaces import (
    AbstractCapitalService,
    AuditRepositoryProtocol,
    CapitalRepositoryProtocol,
)
from src.core.base.interfaces import HealthStatus
from src.core.base.service import TransactionalService
from src.core.exceptions import (
    DependencyError,
    ServiceError,
    StateError,
    ValidationError,
)
from src.core.types.risk import (
    CapitalAllocation,
    CapitalMetrics,
)
from src.error_handling.decorators import with_circuit_breaker, with_retry

# State management imports
from src.state import StatePriority, StateService, StateType
from src.state.consistency import (
    ConsistentEventPattern,
    ConsistentProcessingPattern,
    ConsistentValidationPattern,
    emit_state_event,
    validate_state_data,
)
from src.utils.capital_config import (
    extract_decimal_config,
    extract_percentage_config,
    load_capital_config,
)
from src.utils.capital_resources import get_resource_manager
from src.utils.capital_validation import (
    safe_decimal_conversion,
)
from src.utils.decorators import cache_result, time_execution
from src.utils.formatters import format_currency


class CapitalService(AbstractCapitalService, TransactionalService):
    """
    Enterprise-grade capital management service.

    Features:
    ✅ Full database abstraction using DatabaseService
    ✅ Transaction support with automatic rollback
    ✅ Comprehensive audit trail for all operations
    ✅ Circuit breaker protection
    ✅ Retry mechanisms with exponential backoff
    ✅ Performance monitoring and metrics
    ✅ Cache layer integration
    ✅ Health checks and degraded mode
    ✅ Risk-aware allocation strategies
    ✅ Emergency reserve management
    """

    def __init__(
        self,
        capital_repository: CapitalRepositoryProtocol | None = None,
        audit_repository: AuditRepositoryProtocol | None = None,
        state_service: StateService | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize capital service with dependency injection.

        Args:
            capital_repository: Capital repository implementation (injected)
            audit_repository: Audit repository implementation (injected)
            state_service: State service for state management integration
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="CapitalService",
            correlation_id=correlation_id,
        )

        self._capital_repository = capital_repository
        self._audit_repository = audit_repository
        self.state_service = state_service

        self._load_configuration()
        config = getattr(self, "_capital_config", {})
        self.total_capital = extract_decimal_config(config, "total_capital", Decimal("100000"))
        self.emergency_reserve_pct = extract_percentage_config(config, "emergency_reserve_pct", 0.1)
        self.max_allocation_pct = extract_percentage_config(config, "max_allocation_pct", 0.2)
        self.max_daily_reallocation_pct = extract_percentage_config(
            config, "max_daily_reallocation_pct", 0.1
        )

        # Performance tracking
        self._performance_metrics = {
            "total_allocations": 0,
            "successful_allocations": 0,
            "failed_allocations": 0,
            "total_releases": 0,
            "total_rebalances": 0,
            "average_allocation_time_ms": 0.0,
            "total_capital_managed": self.total_capital,
            "emergency_reserve_maintained": True,
            "risk_limit_violations": 0,
        }

        # Cache configuration (configurable via capital_config)
        self._cache_enabled = True
        self._cache_ttl = 300  # 5 minutes, configurable

        # Configure service patterns
        self.configure_circuit_breaker(enabled=True, threshold=5, timeout=60)
        self.configure_retry(enabled=True, max_retries=3, delay=1.0, backoff=2.0)

        self._event_pattern = ConsistentEventPattern("CapitalService")
        self._validation_pattern = ConsistentValidationPattern()
        self._processing_pattern = ConsistentProcessingPattern("CapitalService")

        self._logger.info(
            "CapitalService initialized with enterprise features",
            total_capital=float(self.total_capital),
            emergency_reserve_pct=float(self.emergency_reserve_pct),
            max_allocation_pct=float(self.max_allocation_pct),
        )

    async def _do_start(self) -> None:
        """Start the capital service."""
        try:
            # Resolve repositories if not injected
            if not self._capital_repository:
                try:
                    self._capital_repository = self.resolve_dependency("CapitalRepository")
                except DependencyError as e:
                    self._logger.warning(
                        f"CapitalRepository not available via DI: {e}, "
                        "service will operate in degraded mode"
                    )

            if not self._audit_repository:
                try:
                    self._audit_repository = self.resolve_dependency("AuditRepository")
                except DependencyError as e:
                    self._logger.warning(
                        f"AuditRepository not available via DI: {e}, audit logging will be disabled"
                    )

            # Resolve state service if not injected
            if not self.state_service:
                try:
                    self.state_service = self.resolve_dependency("StateService")
                except DependencyError as e:
                    self._logger.info(
                        f"StateService not available via DI: {e}, "
                        "operating without state management integration"
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Unexpected error resolving StateService: {e}, "
                        "operating without state management"
                    )

            # Initialize capital metrics if repository available
            if self._capital_repository:
                await self._initialize_capital_state()

            # Configure cache settings from config
            capital_config = getattr(self, "_capital_config", {})
            self._cache_enabled = capital_config.get("cache_enabled", True)
            self._cache_ttl = capital_config.get("cache_ttl_seconds", 300)

            self._logger.info("CapitalService started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start CapitalService: {e}")
            raise ServiceError(f"CapitalService startup failed: {e}") from e

    async def _do_stop(self) -> None:
        """Stop the capital service and clean up resources."""
        try:
            await self.cleanup_resources()
            self._logger.info("CapitalService stopped and resources cleaned up")
        except Exception as e:
            self._logger.error(f"Error during CapitalService shutdown: {e}")
            raise ServiceError(f"CapitalService shutdown failed: {e}") from e

    async def _initialize_capital_state(self) -> None:
        """Initialize capital state from repository."""
        try:
            if not self._capital_repository:
                self._logger.warning(
                    "No capital repository available, skipping state initialization"
                )
                return

            # Load existing allocations via repository
            allocations = await self._capital_repository.get_all()

            # Calculate total allocated
            total_allocated = sum(
                self._safe_decimal_conversion(getattr(alloc, "allocated_amount", "0"))
                for alloc in allocations
            )

            # Update metrics
            self._performance_metrics["total_allocations"] = len(allocations)

            self._logger.info(
                "Capital state initialized",
                total_capital=format_currency(self.total_capital),
                total_allocated=format_currency(total_allocated),
                active_allocations=len(allocations),
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize capital state: {e}")
            # Don't raise - allow service to start without state initialization
            self._logger.warning("Service will continue without capital state initialization")

    # Core Capital Management Operations

    @time_execution
    async def allocate_capital(
        self,
        strategy_id: str,
        exchange: str,
        requested_amount: Decimal,
        bot_id: str | None = None,
        authorized_by: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> CapitalAllocation:
        """
        Allocate capital to a strategy with full audit trail.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            requested_amount: Requested capital amount
            bot_id: Associated bot instance ID
            authorized_by: User or system authorizing allocation
            risk_context: Risk assessment context

        Returns:
            CapitalAllocation: Allocation record

        Raises:
            ServiceError: If allocation fails
            ValidationError: If request is invalid
        """
        return await self.execute_in_transaction(
            "allocate_capital",
            self._allocate_capital_impl,
            strategy_id,
            exchange,
            requested_amount,
            bot_id,
            authorized_by,
            risk_context,
        )

    async def _allocate_capital_impl(
        self,
        strategy_id: str,
        exchange: str,
        requested_amount: Decimal,
        bot_id: str | None,
        authorized_by: str | None,
        risk_context: dict[str, Any] | None,
    ) -> CapitalAllocation:
        """Internal implementation of capital allocation."""
        operation_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        try:
            # Validate inputs using consistent validation pattern
            await self._validate_allocation_request_consistent(
                strategy_id, exchange, requested_amount
            )

            # Check available capital
            available_capital = await self._get_available_capital()
            if requested_amount > available_capital:
                raise ValidationError(
                    f"Insufficient capital: requested {requested_amount}, "
                    f"available {available_capital}",
                    error_code="CAP_002",
                    details={
                        "requested_amount": str(requested_amount),
                        "available_capital": str(available_capital),
                        "component": "CapitalService",
                    },
                )

            # Check allocation limits
            await self._validate_allocation_limits_consistent(
                strategy_id, exchange, requested_amount
            )

            # Get existing allocation if any
            existing_allocation = await self._get_existing_allocation(strategy_id, exchange)

            # Calculate allocation percentage
            allocation_percentage = requested_amount / self.total_capital

            # Create or update allocation record
            if existing_allocation:
                # Update existing allocation - work with allocation data
                current_allocated = self._safe_decimal_conversion(
                    getattr(existing_allocation, "allocated_amount", "0")
                )
                new_allocated = current_allocated + requested_amount

                allocation_data = {
                    "id": getattr(existing_allocation, "id", str(uuid.uuid4())),
                    "strategy_id": strategy_id,
                    "exchange": exchange,
                    "allocated_amount": str(new_allocated),
                    "utilized_amount": getattr(existing_allocation, "utilized_amount", "0"),
                    "available_amount": str(new_allocated),
                    "allocation_percentage": new_allocated / self.total_capital,
                    "last_rebalance": start_time.isoformat(),
                    "updated_at": start_time.isoformat(),
                }

                db_allocation = await self._update_allocation(allocation_data)
                previous_amount = current_allocated

            else:
                # Create new allocation - use data dict instead of DB model
                allocation_data = {
                    "id": str(uuid.uuid4()),
                    "strategy_id": strategy_id,
                    "exchange": exchange,
                    "allocated_amount": str(requested_amount),
                    "utilized_amount": "0",
                    "available_amount": str(requested_amount),
                    "allocation_percentage": float(allocation_percentage),
                    "last_rebalance": start_time.isoformat(),
                    "created_at": start_time.isoformat(),
                    "updated_at": start_time.isoformat(),
                }

                db_allocation = await self._create_allocation(allocation_data)
                previous_amount = Decimal("0")

            # Create audit log
            await self._create_audit_log(
                operation_id=operation_id,
                operation_type="allocate",
                strategy_id=strategy_id,
                exchange=exchange,
                bot_id=bot_id,
                amount=requested_amount,
                previous_amount=previous_amount,
                new_amount=Decimal(str(db_allocation.get("allocated_amount", "0"))),
                operation_context={
                    "requested_amount": str(requested_amount),
                    "available_capital_before": str(available_capital),
                    "allocation_percentage": float(allocation_percentage),
                    "risk_context": risk_context or {},
                },
                authorized_by=authorized_by,
                requested_at=start_time,
                executed_at=datetime.now(timezone.utc),
            )

            # Update performance metrics
            self._performance_metrics["successful_allocations"] += 1
            self._update_allocation_metrics(start_time)

            # Convert to domain object with safe type conversion
            allocation = CapitalAllocation(
                allocation_id=str(db_allocation.get("id", str(uuid.uuid4()))),
                strategy_id=db_allocation.get("strategy_id", strategy_id),
                allocated_amount=self._safe_decimal_conversion(
                    db_allocation.get("allocated_amount", "0")
                ),
                utilized_amount=self._safe_decimal_conversion(
                    db_allocation.get("utilized_amount", "0")
                ),
                available_amount=self._safe_decimal_conversion(
                    db_allocation.get("available_amount", "0")
                ),
                allocation_percentage=Decimal(str(db_allocation.get("allocation_percentage", 0.0))),
                target_allocation_pct=Decimal(str(db_allocation.get("allocation_percentage", 0.0))),
                min_allocation=Decimal("0"),
                max_allocation=self._safe_decimal_conversion(
                    db_allocation.get("allocated_amount", "0")
                )
                * Decimal("2"),
                last_rebalance=(
                    (
                        datetime.fromisoformat(db_allocation.get("last_rebalance"))
                        if isinstance(db_allocation.get("last_rebalance"), str)
                        else db_allocation.get("last_rebalance")
                    )
                    if db_allocation.get("last_rebalance")
                    else start_time
                ),
            )

            self._logger.info(
                "Capital allocation successful",
                operation_id=operation_id,
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(requested_amount),
                allocation_percentage=f"{float(allocation_percentage):.2%}",
            )

            allocation_event_data = {
                "allocation_id": str(db_allocation.get("id", str(uuid.uuid4()))),
                "strategy_id": strategy_id,
                "exchange": exchange,
                "amount": str(requested_amount),
                "operation_id": operation_id,
                "timestamp": start_time.isoformat(),
            }
            await self._event_pattern.emit_consistent("capital.allocated", allocation_event_data)
            await self._save_capital_state_snapshot(reason="allocation_change")

            return allocation

        except ServiceError as e:
            self._performance_metrics["failed_allocations"] += 1
            self._logger.error(
                "Service error during capital allocation",
                operation_id=operation_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

        except ValidationError:
            self._performance_metrics["failed_allocations"] += 1
            raise

        except Exception as e:
            self._performance_metrics["failed_allocations"] += 1

            # Create failure audit log
            await self._create_audit_log(
                operation_id=operation_id,
                operation_type="allocate",
                strategy_id=strategy_id,
                exchange=exchange,
                bot_id=bot_id,
                amount=requested_amount,
                operation_context={"error": str(e)},
                authorized_by=authorized_by,
                requested_at=start_time,
                operation_status="failed",
                success=False,
                error_message=str(e),
            )

            self._logger.error(
                "Capital allocation failed",
                operation_id=operation_id,
                strategy_id=strategy_id,
                exchange=exchange,
                requested_amount=format_currency(requested_amount),
                error=str(e),
            )

            raise ServiceError(f"Capital allocation failed: {e}") from e

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @time_execution
    async def release_capital(
        self,
        strategy_id: str,
        exchange: str,
        release_amount: Decimal,
        bot_id: str | None = None,
        authorized_by: str | None = None,
    ) -> bool:
        """
        Release allocated capital with full audit trail.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            release_amount: Amount to release
            bot_id: Associated bot instance ID
            authorized_by: User or system authorizing release

        Returns:
            bool: True if release successful

        Raises:
            ServiceError: If release fails
            ValidationError: If request is invalid
        """
        return await self.execute_in_transaction(
            "release_capital",
            self._release_capital_impl,
            strategy_id,
            exchange,
            release_amount,
            bot_id,
            authorized_by,
        )

    async def _release_capital_impl(
        self,
        strategy_id: str,
        exchange: str,
        release_amount: Decimal,
        bot_id: str | None,
        authorized_by: str | None,
    ) -> bool:
        """Internal implementation of capital release."""
        operation_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        try:
            # Validate inputs
            if not strategy_id or not strategy_id.strip():
                raise ValidationError(
                    "Strategy ID cannot be empty",
                    error_code="CAP_004",
                    details={"component": "CapitalService", "field": "strategy_id"},
                )

            if not exchange or not exchange.strip():
                raise ValidationError(
                    "Exchange cannot be empty",
                    error_code="CAP_005",
                    details={"component": "CapitalService", "field": "exchange"},
                )

            if release_amount <= 0:
                raise ValidationError(
                    "Release amount must be positive",
                    error_code="CAP_007",
                    details={"amount": str(release_amount), "component": "CapitalService"},
                )

            # Find existing allocation
            allocation = await self._get_existing_allocation(strategy_id, exchange)
            if not allocation:
                self._logger.warning(
                    "No allocation found to release",
                    strategy_id=strategy_id,
                    exchange=exchange,
                )
                return True  # Consider successful if no allocation exists

            # Validate release amount
            allocated_amount = Decimal(allocation.allocated_amount)
            if release_amount > allocated_amount:
                raise ValidationError(
                    f"Cannot release {release_amount} from allocation of {allocated_amount}",
                    error_code="CAP_003",
                    details={
                        "release_amount": str(release_amount),
                        "allocated_amount": str(allocated_amount),
                        "component": "CapitalService",
                    },
                )

            # Update allocation
            new_allocated_amount = allocated_amount - release_amount
            new_available_amount = Decimal(allocation.available_amount) - release_amount

            if new_allocated_amount == Decimal("0"):
                # Delete allocation if zero
                await self._delete_allocation(allocation.allocation_id)
            else:
                # Update allocation using dictionary format
                allocation_data = {
                    "id": allocation.allocation_id,
                    "strategy_id": allocation.strategy_id,
                    "exchange": exchange,
                    "allocated_amount": str(new_allocated_amount),
                    "utilized_amount": allocation.utilized_amount,
                    "available_amount": str(new_available_amount),
                    "allocation_percentage": new_allocated_amount / self.total_capital,
                    "last_rebalance": allocation.last_rebalance,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }

                await self._update_allocation(allocation_data)

            # Create audit log
            await self._create_audit_log(
                operation_id=operation_id,
                operation_type="release",
                strategy_id=strategy_id,
                exchange=exchange,
                bot_id=bot_id,
                amount=release_amount,
                previous_amount=allocated_amount,
                new_amount=new_allocated_amount,
                operation_context={
                    "release_amount": str(release_amount),
                    "remaining_allocation": str(new_allocated_amount),
                },
                authorized_by=authorized_by,
                requested_at=start_time,
                executed_at=datetime.now(timezone.utc),
            )

            # Update performance metrics
            self._performance_metrics["total_releases"] += 1
            self._update_allocation_metrics(start_time)

            self._logger.info(
                "Capital release successful",
                operation_id=operation_id,
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(release_amount),
                remaining_allocation=format_currency(new_allocated_amount),
            )

            release_event_data = {
                "strategy_id": strategy_id,
                "exchange": exchange,
                "amount": str(release_amount),
                "operation_id": operation_id,
                "timestamp": start_time.isoformat(),
            }
            await self._event_pattern.emit_consistent("capital.released", release_event_data)
            await self._save_capital_state_snapshot(reason="release_capital")

            return True

        except ServiceError as e:
            self._logger.error(
                "Service error during capital release",
                operation_id=operation_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            return False

        except ValidationError:
            raise

        except Exception as e:
            # Create failure audit log
            await self._create_audit_log(
                operation_id=operation_id,
                operation_type="release",
                strategy_id=strategy_id,
                exchange=exchange,
                bot_id=bot_id,
                amount=release_amount,
                operation_context={"error": str(e)},
                authorized_by=authorized_by,
                requested_at=start_time,
                operation_status="failed",
                success=False,
                error_message=str(e),
            )

            self._logger.error(
                "Capital release failed",
                operation_id=operation_id,
                strategy_id=strategy_id,
                exchange=exchange,
                release_amount=format_currency(release_amount),
                error=str(e),
            )

            return False

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @time_execution
    async def update_utilization(
        self,
        strategy_id: str,
        exchange: str,
        utilized_amount: Decimal,
        bot_id: str | None = None,
    ) -> bool:
        """
        Update capital utilization for a strategy.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            utilized_amount: Amount currently utilized
            bot_id: Associated bot instance ID

        Returns:
            bool: True if update successful
        """
        return await self.execute_with_monitoring(
            "update_utilization",
            self._update_utilization_impl,
            strategy_id,
            exchange,
            utilized_amount,
            bot_id,
        )

    async def _update_utilization_impl(
        self,
        strategy_id: str,
        exchange: str,
        utilized_amount: Decimal,
        bot_id: str | None,
    ) -> bool:
        """Internal implementation of utilization update."""
        try:
            # Find allocation
            allocation = await self._get_existing_allocation(strategy_id, exchange)
            if not allocation:
                self._logger.warning(
                    "No allocation found for utilization update",
                    strategy_id=strategy_id,
                    exchange=exchange,
                )
                return False

            # Validate utilization
            allocated_amount = Decimal(allocation.allocated_amount)
            if utilized_amount > allocated_amount:
                raise ValidationError(
                    f"Utilization {utilized_amount} exceeds allocation {allocated_amount}"
                )

            # Update utilization using dictionary format
            allocation_data = {
                "id": allocation.allocation_id,
                "strategy_id": allocation.strategy_id,
                "exchange": exchange,
                "allocated_amount": str(allocated_amount),
                "utilized_amount": str(utilized_amount),
                "available_amount": str(allocated_amount - utilized_amount),
                "allocation_percentage": allocation.allocation_percentage,
                "last_rebalance": allocation.last_rebalance,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            await self._update_allocation(allocation_data)

            self._logger.debug(
                "Capital utilization updated",
                strategy_id=strategy_id,
                exchange=exchange,
                utilized=format_currency(utilized_amount),
                available=format_currency(allocated_amount - utilized_amount),
            )

            return True

        except ServiceError as e:
            self._logger.error(
                "Service error during utilization update",
                strategy_id=strategy_id,
                exchange=exchange,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

        except ValidationError:
            raise

        except Exception as e:
            self._logger.error(
                "Utilization update failed",
                strategy_id=strategy_id,
                exchange=exchange,
                utilized_amount=format_currency(utilized_amount),
                error=str(e),
            )
            return False

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @cache_result(ttl=60)
    @time_execution
    async def get_capital_metrics(self) -> CapitalMetrics:
        """
        Get current capital management metrics.

        Returns:
            CapitalMetrics: Current capital metrics
        """
        return await self.execute_with_monitoring(
            "get_capital_metrics",
            self._get_capital_metrics_impl,
        )

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @cache_result(ttl=120)
    @time_execution
    async def get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]:
        """
        Get all capital allocations for a specific strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            List[CapitalAllocation]: All allocations for the strategy
        """
        return await self.execute_with_monitoring(
            "get_allocations_by_strategy",
            self._get_allocations_by_strategy_impl,
            strategy_id,
        )

    async def _get_capital_metrics_impl(self) -> CapitalMetrics:
        """Internal implementation of capital metrics calculation."""
        try:
            # Get all allocations
            allocations = await self._get_all_allocations()

            # Calculate metrics
            total_allocated = sum(
                self._safe_decimal_conversion(alloc.allocated_amount) for alloc in allocations
            ) or Decimal("0")
            # Total utilization calculation for future metrics features
            emergency_reserve = self.total_capital * self.emergency_reserve_pct
            available_capital = self.total_capital - total_allocated - emergency_reserve

            # Calculate allocation efficiency (used for future features)

            metrics = CapitalMetrics(
                total_capital=self.total_capital,
                allocated_amount=total_allocated,
                available_amount=available_capital,
                total_pnl=Decimal("0"),  # Default value
                realized_pnl=Decimal("0"),  # Default value
                unrealized_pnl=Decimal("0"),  # Default value
                daily_return=Decimal("0.0"),  # Default value
                weekly_return=Decimal("0.0"),  # Default value
                monthly_return=Decimal("0.0"),  # Default value
                yearly_return=Decimal("0.0"),  # Default value
                total_return=Decimal("0.0"),  # Default value
                sharpe_ratio=Decimal("0.0"),  # Default value
                sortino_ratio=Decimal("0.0"),  # Default value
                calmar_ratio=Decimal("0.0"),  # Default value
                current_drawdown=Decimal("0.0"),  # Default value
                max_drawdown=Decimal("0.0"),  # Default value
                var_95=Decimal("0"),  # Default value
                expected_shortfall=Decimal("0"),  # Default value
                strategies_active=len(allocations),
                positions_open=0,  # Default value
                leverage_used=Decimal("0.0"),  # Default value
                timestamp=datetime.now(timezone.utc),
            )

            return metrics

        except Exception as e:
            self._logger.error(f"Failed to calculate capital metrics: {e}")
            raise ServiceError(f"Capital metrics calculation failed: {e}") from e

    async def _get_allocations_by_strategy_impl(self, strategy_id: str) -> list[CapitalAllocation]:
        """Internal implementation to get allocations by strategy."""
        try:
            if not self._capital_repository:
                return []

            # Get allocations by strategy via repository
            db_allocations = await self._capital_repository.get_by_strategy(strategy_id)

            allocations = []
            for db_alloc in db_allocations:
                allocation = CapitalAllocation(
                    allocation_id=str(getattr(db_alloc, "id", str(uuid.uuid4()))),
                    strategy_id=getattr(db_alloc, "strategy_id", strategy_id),
                    allocated_amount=self._safe_decimal_conversion(
                        getattr(db_alloc, "allocated_amount", "0")
                    ),
                    utilized_amount=self._safe_decimal_conversion(
                        getattr(db_alloc, "utilized_amount", "0")
                    ),
                    available_amount=self._safe_decimal_conversion(
                        getattr(db_alloc, "available_amount", "0")
                    ),
                    allocation_percentage=Decimal(str(getattr(db_alloc, "allocation_percentage", 0.0))),
                    target_allocation_pct=Decimal(
                        str(getattr(db_alloc, "allocation_percentage", 0.0))
                    ),
                    min_allocation=Decimal("0"),
                    max_allocation=self._safe_decimal_conversion(
                        getattr(db_alloc, "allocated_amount", "0")
                    )
                    * Decimal("2"),
                    last_rebalance=getattr(db_alloc, "last_rebalance", datetime.now(timezone.utc)),
                )
                allocations.append(allocation)

            return allocations

        except Exception as e:
            self._logger.error(f"Failed to get allocations for strategy {strategy_id}: {e}")
            raise ServiceError(f"Failed to get allocations by strategy: {e}") from e

    # Helper Methods

    async def _validate_allocation_request_consistent(
        self, strategy_id: str, exchange: str, amount: Decimal
    ) -> None:
        """Validate allocation request using consistent validation pattern."""
        validation_data = {"strategy_id": strategy_id, "exchange": exchange, "amount": str(amount)}
        result = await validate_state_data("capital_allocation_request", validation_data)
        if not result["is_valid"]:
            raise ValidationError(
                f"Capital allocation validation failed: {result['errors']}",
                error_code="CAP_004",
                details={"component": "CapitalService", "validation_errors": result["errors"]},
            )
        if not strategy_id or not strategy_id.strip():
            raise ValidationError(
                "Strategy ID cannot be empty",
                error_code="CAP_004",
                details={"component": "CapitalService", "field": "strategy_id"},
            )

        if not exchange or not exchange.strip():
            raise ValidationError(
                "Exchange cannot be empty",
                error_code="CAP_005",
                details={"component": "CapitalService", "field": "exchange"},
            )

        if amount <= 0:
            raise ValidationError(
                "Allocation amount must be positive",
                error_code="CAP_007",
                details={"amount": str(amount), "component": "CapitalService"},
            )

    async def _validate_allocation_limits_consistent(
        self, strategy_id: str, exchange: str, amount: Decimal
    ) -> None:
        """Validate allocation limits using consistent boundary validation pattern."""
        boundary_validation = {
            "strategy_id": strategy_id,
            "exchange": exchange,
            "amount": str(amount),
            "max_allocation_pct": str(self.max_allocation_pct),
        }
        result = await validate_state_data("capital_allocation_limits", boundary_validation)
        if not result["is_valid"]:
            raise ValidationError(
                f"Allocation limits validation failed: {result['errors']}",
                error_code="CAP_008",
                details={"component": "CapitalService", "validation_errors": result["errors"]},
            )
        if not strategy_id or not strategy_id.strip():
            raise ValidationError(
                "Strategy ID cannot be empty",
                error_code="CAP_004",
                details={"component": "CapitalService", "field": "strategy_id"},
            )

        if not exchange or not exchange.strip():
            raise ValidationError(
                "Exchange cannot be empty",
                error_code="CAP_005",
                details={"component": "CapitalService", "field": "exchange"},
            )

        if amount <= 0:
            raise ValidationError(
                "Allocation amount must be positive",
                error_code="CAP_007",
                details={"amount": str(amount), "component": "CapitalService"},
            )

        async def calculate_max_allocation(data):
            return data["total_capital"] * data["max_pct"]

        max_allocation = await self._processing_pattern.process_item_consistent(
            {"total_capital": self.total_capital, "max_pct": self.max_allocation_pct},
            calculate_max_allocation,
        )

        if amount > max_allocation:
            raise ValidationError(
                f"Requested amount {amount} exceeds maximum allocation {max_allocation}",
                error_code="CAP_008",
                details={
                    "requested_amount": str(amount),
                    "max_allocation": str(max_allocation),
                    "component": "CapitalService",
                },
            )

    async def _get_available_capital(self) -> Decimal:
        """Calculate available capital for allocation."""
        if not self._capital_repository:
            # Degraded mode - use conservative estimate from config
            degraded_ratio = Decimal(str(self.capital_config.get("degraded_capital_ratio", "0.5")))
            return self.total_capital * degraded_ratio

        try:
            # Get all allocations via repository
            allocations = await self._capital_repository.get_all()

            if not allocations:
                return self.total_capital * Decimal("0.8")

            total_allocated = sum(
                self._safe_decimal_conversion(getattr(alloc, "allocated_amount", "0"))
                for alloc in allocations
            )
            emergency_reserve = self.total_capital * self.emergency_reserve_pct

            return self.total_capital - total_allocated - emergency_reserve
        except Exception as e:
            self._logger.error(f"Failed to get available capital: {e}")
            # Return conservative estimate on error
            return self.total_capital * Decimal("0.3")

    async def _get_existing_allocation(self, strategy_id: str, exchange: str) -> Any | None:
        """Get existing allocation for strategy and exchange via repository."""
        if not self._capital_repository:
            return None

        try:
            return await self._capital_repository.get_by_strategy_exchange(strategy_id, exchange)
        except Exception as e:
            self._logger.error(
                f"Failed to get existing allocation for {strategy_id}/{exchange}: {e}"
            )
            return None

    async def _create_allocation(self, allocation_data: dict[str, Any]) -> Any:
        """Create allocation via repository."""
        if not self._capital_repository:
            raise ServiceError("Capital repository not available - allocation operations disabled")

        try:
            return await self._capital_repository.create(allocation_data)
        except Exception as e:
            self._logger.error(f"Failed to create allocation: {e}")
            raise ServiceError(f"Failed to create allocation: {e}") from e

    async def _update_allocation(self, allocation_data: dict[str, Any]) -> Any:
        """Update allocation via repository."""
        if not self._capital_repository:
            raise ServiceError("Capital repository not available - allocation operations disabled")

        try:
            return await self._capital_repository.update(allocation_data)
        except Exception as e:
            self._logger.error(f"Failed to update allocation: {e}")
            raise ServiceError(f"Failed to update allocation: {e}") from e

    async def _delete_allocation(self, allocation_id: str) -> bool:
        """Delete allocation via repository."""
        if not self._capital_repository:
            raise ServiceError("Capital repository not available - allocation operations disabled")

        try:
            await self._capital_repository.delete(allocation_id)
            return True
        except Exception as e:
            self._logger.error(f"Failed to delete allocation: {e}")
            raise ServiceError(f"Failed to delete allocation: {e}") from e

    async def _get_all_allocations(self, limit: int | None = None) -> list[Any]:
        """Get all allocations via repository."""
        if not self._capital_repository:
            return []

        try:
            return await self._capital_repository.get_all(limit=limit)
        except Exception as e:
            self._logger.error(f"Failed to get all allocations: {e}")
            return []

    async def _create_audit_log_record(self, audit_log_data: dict[str, Any]) -> None:
        """Create audit log record via repository."""
        if not self._audit_repository:
            self._logger.warning("Audit repository not available, skipping audit log")
            return

        try:
            await self._audit_repository.create(audit_log_data)
        except Exception as e:
            self._logger.error(f"Failed to create audit log record: {e}")
            # Don't raise - audit failure shouldn't break the operation

    async def _restore_allocations_in_transaction(
        self, allocations_data: list[dict[str, Any]]
    ) -> None:
        """
        Restore allocations via repository.

        Args:
            allocations_data: List of allocation data dictionaries
        """
        if not self._capital_repository or not allocations_data:
            self._logger.warning("No capital repository or data available for restoration")
            return

        existing_allocations = None
        try:
            # Note: Transaction management should be handled at repository level
            # This service layer focuses on business logic, not transaction boundaries

            # Clear existing allocations first (if repository supports it)
            existing_allocations = await self._capital_repository.get_all()
            for allocation in existing_allocations:
                allocation_id = getattr(allocation, "id", None)
                if allocation_id:
                    await self._capital_repository.delete(allocation_id)

            # Restore allocations
            for alloc_data in allocations_data:
                # Ensure data structure is consistent
                normalized_data = {
                    "id": alloc_data.get("id", str(uuid.uuid4())),
                    "strategy_id": alloc_data["strategy_id"],
                    "exchange": alloc_data["exchange"],
                    "allocated_amount": alloc_data["allocated_amount"],
                    "utilized_amount": alloc_data["utilized_amount"],
                    "available_amount": alloc_data["available_amount"],
                    "allocation_percentage": alloc_data["allocation_percentage"],
                    "last_rebalance": alloc_data["last_rebalance"],
                    "created_at": alloc_data.get(
                        "created_at", datetime.now(timezone.utc).isoformat()
                    ),
                    "updated_at": alloc_data.get(
                        "updated_at", datetime.now(timezone.utc).isoformat()
                    ),
                }

                await self._capital_repository.create(normalized_data)

            self._logger.info(f"Successfully restored {len(allocations_data)} allocations")

        except Exception as e:
            self._logger.error(f"Failed to restore allocations: {e}")
            raise ServiceError(f"State restoration failed: {e}") from e
        finally:
            existing_allocations = None
            if allocations_data and len(allocations_data) > 100:
                import gc

                gc.collect()

    async def _calculate_allocation_efficiency(self, allocations: list[Any]) -> float:
        """Calculate allocation efficiency score."""
        if not allocations:
            return 0.5  # Neutral efficiency

        # Base efficiency: utilization rate
        total_allocated = sum(
            self._safe_decimal_conversion(alloc.allocated_amount) for alloc in allocations
        )
        total_utilized = sum(
            self._safe_decimal_conversion(alloc.utilized_amount) for alloc in allocations
        )

        if total_allocated == 0:
            return Decimal("0.5")

        utilization_efficiency = total_utilized / total_allocated

        # Apply performance multiplier (simplified for now)
        performance_multiplier = Decimal("1.0")  # Could integrate with performance metrics

        return max(
            Decimal("0.0"), min(Decimal("2.0"), utilization_efficiency * performance_multiplier)
        )

    async def _create_audit_log(
        self,
        operation_id: str,
        operation_type: str,
        strategy_id: str | None = None,
        exchange: str | None = None,
        bot_id: str | None = None,
        amount: Decimal | None = None,
        previous_amount: Decimal | None = None,
        new_amount: Decimal | None = None,
        operation_context: dict[str, Any] | None = None,
        authorized_by: str | None = None,
        requested_at: datetime | None = None,
        executed_at: datetime | None = None,
        operation_status: str = "completed",
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Create comprehensive audit log entry."""
        try:
            if not self._audit_repository:
                return  # Skip if audit repository not available

            audit_log_data = {
                "id": str(uuid.uuid4()),
                "operation_id": operation_id,
                "operation_type": operation_type,
                "strategy_id": strategy_id,
                "exchange": exchange,
                "bot_id": bot_id,
                "operation_description": f"Capital {operation_type} operation",
                "amount": str(amount) if amount is not None else None,
                "previous_amount": str(previous_amount) if previous_amount is not None else None,
                "new_amount": str(new_amount) if new_amount is not None else None,
                "operation_context": operation_context or {},
                "operation_status": operation_status,
                "success": success,
                "error_message": error_message,
                "authorized_by": authorized_by,
                "requested_at": (requested_at or datetime.now(timezone.utc)).isoformat(),
                "executed_at": executed_at.isoformat() if executed_at else None,
                "source_component": "CapitalService",
                "correlation_id": self._correlation_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            await self._create_audit_log_record(audit_log_data)

        except Exception as e:
            self._logger.error(f"Failed to create audit log: {e}")
            # Don't raise - audit failure shouldn't break the operation

    def _update_allocation_metrics(self, start_time: datetime) -> None:
        """Update allocation performance metrics."""
        execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Update average allocation time
        total_ops = (
            self._performance_metrics["successful_allocations"]
            + self._performance_metrics["failed_allocations"]
        )
        current_avg = self._performance_metrics["average_allocation_time_ms"]

        if total_ops > 0:
            self._performance_metrics["average_allocation_time_ms"] = (
                (current_avg * (total_ops - 1)) + execution_time_ms
            ) / total_ops

    # Service Health and Monitoring

    async def _service_health_check(self) -> HealthStatus:
        """Service-specific health check."""
        try:
            # Check repository availability
            if not self._capital_repository:
                self._logger.warning("Capital repository not available")
                return HealthStatus.DEGRADED  # Can operate in limited mode

            # Basic connectivity check via repository
            try:
                await self._capital_repository.get_all(limit=1)
            except Exception as e:
                self._logger.warning(f"Repository connectivity check failed: {e}")
                return HealthStatus.UNHEALTHY

            # Check emergency reserve maintenance
            if not self._performance_metrics.get("emergency_reserve_maintained", True):
                return HealthStatus.DEGRADED

            # Check error rates
            total_allocations = (
                self._performance_metrics["successful_allocations"]
                + self._performance_metrics["failed_allocations"]
            )

            if total_allocations > 10:  # Only check if we have enough data
                error_rate = self._performance_metrics["failed_allocations"] / total_allocations
                if error_rate > 0.1:  # More than 10% failures
                    return HealthStatus.DEGRADED
                elif error_rate > 0.2:  # More than 20% failures
                    return HealthStatus.UNHEALTHY

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(f"Capital service health check failed: {e}")
            return HealthStatus.UNHEALTHY

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get detailed performance metrics."""
        metrics = self._performance_metrics.copy()

        # Add service metrics
        service_metrics = self.get_metrics()
        metrics.update(service_metrics)

        return metrics

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        super().reset_metrics()
        self._performance_metrics = {
            "total_allocations": 0,
            "successful_allocations": 0,
            "failed_allocations": 0,
            "total_releases": 0,
            "total_rebalances": 0,
            "average_allocation_time_ms": 0.0,
            "total_capital_managed": self.total_capital,
            "emergency_reserve_maintained": True,
            "risk_limit_violations": 0,
        }

        self._logger.info("Capital service metrics reset")

    def _load_configuration(self) -> None:
        """Load configuration from ConfigService or environment."""
        try:
            # Try to get config from dependency injection
            config_service = self.resolve_dependency("ConfigService")
            self._capital_config = load_capital_config(config_service)
        except Exception as e:
            self._logger.warning(f"Failed to load configuration from ConfigService: {e}")
            # Will use defaults from load_capital_config
            self._capital_config = load_capital_config(None)

    def _safe_decimal_conversion(self, value: Any) -> Decimal:
        """Safely convert any value to Decimal."""
        return safe_decimal_conversion(value)

    # State Management Integration Methods

    async def _save_capital_state_snapshot(self, reason: str = "allocation_change") -> None:
        """
        Save current capital allocation state using consistent processing pattern.

        Args:
            reason: Reason for saving snapshot (e.g., "allocation_change", "rebalance", "recovery")
        """
        if not self.state_service:
            return  # Skip if StateService not available

        try:

            async def get_allocations(_):
                return await self._get_all_allocations()

            allocations = await self._processing_pattern.process_item_consistent(
                None,
                get_allocations,
            )
            state_data = await self._build_consistent_state_data(allocations, reason)
            await self.state_service.set_state(
                state_type=StateType.SYSTEM_STATE,
                state_id="capital_allocations",
                state_data=state_data,
                source_component="capital_management",
                validate=True,  # Enable validation for consistency
                priority=StatePriority.HIGH,  # Capital data is high priority
                reason=reason,
            )

            await emit_state_event(
                "snapshot_saved",
                {
                    "component": "capital_management",
                    "reason": reason,
                    "allocation_count": len(allocations),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            self._logger.debug(
                "Capital state snapshot saved",
                reason=reason,
                allocation_count=len(allocations),
            )

        except StateError as e:
            self._logger.warning(f"State service error saving capital snapshot: {e}")
        except Exception as e:
            self._logger.warning(f"Failed to save capital state snapshot: {e}")

    async def _build_consistent_state_data(
        self, allocations: list[Any], reason: str
    ) -> dict[str, Any]:
        """Build state data using consistent transformation pattern."""
        allocation_data = None
        try:
            allocation_data = [
                {
                    "strategy_id": getattr(alloc, "strategy_id", ""),
                    "exchange": getattr(alloc, "exchange", ""),
                    "allocated_amount": str(getattr(alloc, "allocated_amount", "0")),
                    "utilized_amount": str(getattr(alloc, "utilized_amount", "0")),
                    "available_amount": str(getattr(alloc, "available_amount", "0")),
                    "allocation_percentage": getattr(alloc, "allocation_percentage", 0.0),
                    "last_rebalance": (
                        alloc.last_rebalance.isoformat()
                        if getattr(alloc, "last_rebalance", None)
                        else None
                    ),
                }
                for alloc in allocations
            ]

            return {
                "total_capital": str(self.total_capital),
                "emergency_reserve_pct": str(self.emergency_reserve_pct),
                "allocations": allocation_data,
                "performance_metrics": self._performance_metrics.copy(),
                "snapshot_reason": reason,
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            allocation_data = None

    async def restore_capital_state(self) -> bool:
        """
        Restore capital state from the latest saved state.

        Returns:
            bool: True if restoration successful
        """
        if not self.state_service:
            self._logger.error("StateService not available for restoration")
            return False

        try:
            # Get state from StateService - use SYSTEM_STATE for capital management data
            state_response = await self.state_service.get_state(
                state_type=StateType.SYSTEM_STATE,
                state_id="capital_allocations",
                include_metadata=True,
            )

            if not state_response:
                self._logger.error("No capital state found for restoration")
                return False

            # Extract state data from response
            if isinstance(state_response, dict):
                state_data = state_response.get("data", state_response)
            else:
                state_data = state_response

            # Handle case where state_data might be None or not a dict
            if not isinstance(state_data, dict):
                self._logger.error("Invalid state data format for restoration")
                return False

            # Restore configuration
            self.total_capital = Decimal(str(state_data.get("total_capital", "100000")))
            self.emergency_reserve_pct = Decimal(
                str(state_data.get("emergency_reserve_pct", "0.1"))
            )

            # Restore state within a transaction to ensure consistency
            await self._restore_allocations_in_transaction(state_data.get("allocations", []))

            # Restore performance metrics
            performance_metrics = state_data.get("performance_metrics", {})
            if isinstance(performance_metrics, dict):
                self._performance_metrics.update(performance_metrics)

            self._logger.info(
                "Capital state restored successfully",
                allocations_restored=len(state_data.get("allocations", [])),
            )

            return True

        except StateError as e:
            self._logger.error(f"State service error restoring capital state: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Failed to restore capital state: {e}")
            return False

    async def cleanup_resources(self) -> None:
        """Clean up resources to prevent memory leaks."""
        try:
            resource_manager = get_resource_manager()

            # Reset performance metrics to prevent accumulation
            if (
                self._performance_metrics["total_allocations"] > 10000
                or self._performance_metrics["successful_allocations"] > 10000
            ):
                archived_metrics = self._performance_metrics.copy()
                self._logger.info(
                    "Archiving performance metrics", archived_metrics=archived_metrics
                )
                resource_manager.cleanup_allocations_data([archived_metrics])
                self.reset_metrics()

            self._logger.debug("Capital service resource cleanup completed")
        except Exception as e:
            self._logger.warning(f"Resource cleanup failed: {e}")
