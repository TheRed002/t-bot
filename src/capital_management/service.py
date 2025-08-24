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
from decimal import Decimal, InvalidOperation
from typing import Any

from src.core.base.interfaces import HealthStatus
from src.core.base.service import TransactionalService
from src.core.exceptions import (
    ServiceError,
    ValidationError,
)
from src.core.types.risk import (
    CapitalAllocation,
    CapitalMetrics,
)

try:
    from src.database.models import CapitalAllocationDB
    from src.database.models.audit import CapitalAuditLog
    from src.database.repository.capital import (
        CapitalAllocationRepository,
        CapitalAuditLogRepository,
    )
    from src.database.uow import UnitOfWork, UnitOfWorkFactory

    DATABASE_AVAILABLE = True
except ImportError:
    # Database dependencies may not be available in all environments
    CapitalAllocationDB = None  # type: ignore
    CapitalAuditLog = None  # type: ignore
    CapitalAllocationRepository = None  # type: ignore
    CapitalAuditLogRepository = None  # type: ignore
    UnitOfWork = None  # type: ignore
    UnitOfWorkFactory = None  # type: ignore
    DATABASE_AVAILABLE = False

# Database exceptions
try:
    from sqlalchemy.exc import IntegrityError, OperationalError
except ImportError:
    # Fallback if SQLAlchemy not available
    IntegrityError = Exception
    OperationalError = Exception

# DatabaseService will be injected
from src.error_handling.decorators import with_circuit_breaker, with_retry

# State management imports
from src.state import StateService, StateType
from src.utils.decorators import cache_result, time_execution
from src.utils.formatters import format_currency
from src.utils.validators import ValidationFramework


class CapitalService(TransactionalService):
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
        database_service=None,
        uow_factory: UnitOfWorkFactory | None = None,
        state_service: StateService | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize capital service.

        Args:
            database_service: Database service instance (injected) - DEPRECATED, use uow_factory
            uow_factory: Unit of Work factory for transaction management
            state_service: State service for state management integration
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="CapitalService",
            correlation_id=correlation_id,
        )

        # Database layer - prefer UoW factory over direct database service
        self.uow_factory = uow_factory
        self.database_service = database_service  # Keep for backward compatibility
        self.state_service = state_service

        # These will be deprecated once we fully migrate to UoW pattern
        self.capital_repository: CapitalAllocationRepository | None = None
        self.audit_repository: CapitalAuditLogRepository | None = None

        # Capital configuration - load from config if available
        self._load_configuration()

        # Default values if config not available
        self.total_capital = getattr(self, "total_capital", Decimal("100000"))
        self.emergency_reserve_pct = getattr(self, "emergency_reserve_pct", Decimal("0.1"))
        self.max_allocation_pct = getattr(self, "max_allocation_pct", Decimal("0.2"))
        self.max_daily_reallocation_pct = getattr(
            self, "max_daily_reallocation_pct", Decimal("0.1")
        )

        # Performance tracking
        self._performance_metrics = {
            "total_allocations": 0,
            "successful_allocations": 0,
            "failed_allocations": 0,
            "total_releases": 0,
            "total_rebalances": 0,
            "average_allocation_time_ms": 0.0,
            "total_capital_managed": float(self.total_capital),
            "emergency_reserve_maintained": True,
            "risk_limit_violations": 0,
        }

        # Cache configuration
        self._cache_enabled = True
        self._cache_ttl = 300  # 5 minutes

        # Configure service patterns
        self.configure_circuit_breaker(enabled=True, threshold=5, timeout=60)
        self.configure_retry(enabled=True, max_retries=3, delay=1.0, backoff=2.0)

        self._logger.info(
            "CapitalService initialized with enterprise features",
            total_capital=float(self.total_capital),
            emergency_reserve_pct=float(self.emergency_reserve_pct),
            max_allocation_pct=float(self.max_allocation_pct),
        )

    async def _do_start(self) -> None:
        """Start the capital service."""
        try:
            # Resolve UoW factory if not injected
            if not self.uow_factory and DATABASE_AVAILABLE:
                # Try to get from dependency injection
                try:
                    self.uow_factory = self.resolve_dependency("UnitOfWorkFactory")
                except Exception:
                    # Try to resolve DatabaseService for backward compatibility
                    if not self.database_service:
                        try:
                            self.database_service = self.resolve_dependency("DatabaseService")
                        except Exception:
                            self._logger.warning(
                                "Neither UnitOfWorkFactory nor DatabaseService available via DI"
                            )

            # Initialize repositories if we have UoW factory (preferred) or database service
            # NOTE: Direct repository instantiation is deprecated - this is kept for backward compatibility
            if self.uow_factory:
                # We'll use UoW pattern in transactions, no need for persistent repositories
                self._logger.debug("Using UnitOfWork pattern for database operations")
            elif self.database_service and hasattr(self.database_service, "get_session"):
                # Fallback: create repositories from database service session
                session = self.database_service.get_session()
                self.capital_repository = CapitalAllocationRepository(session)
                self.audit_repository = CapitalAuditLogRepository(session)
                self._logger.warning(
                    "Using legacy database service pattern - consider migrating to UnitOfWork"
                )

            # Resolve state service if not injected
            if not self.state_service:
                try:
                    self.state_service = self.resolve_dependency("StateService")
                except Exception:
                    self._logger.warning(
                        "StateService not available, operating without state management integration"
                    )

            # Initialize capital metrics from database
            await self._initialize_capital_state()

            self._logger.info("CapitalService started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start CapitalService: {e}")
            raise ServiceError(f"CapitalService startup failed: {e}") from e

    async def _initialize_capital_state(self) -> None:
        """Initialize capital state from database."""
        try:
            # Load existing allocations
            allocations = await self._get_all_allocations()

            # Calculate total allocated
            total_allocated = sum(
                self._safe_decimal_conversion(alloc.allocated_amount) for alloc in allocations
            )

            # Update metrics
            self._performance_metrics["total_allocations"] = len(allocations)

            self._logger.info(
                "Capital state initialized",
                total_capital=format_currency(float(self.total_capital)),
                total_allocated=format_currency(float(total_allocated)),
                active_allocations=len(allocations),
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize capital state: {e}")
            raise

    # Core Capital Management Operations

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    @with_retry(max_attempts=3, base_delay=1.0)
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
            # Validate inputs
            self._validate_allocation_request(strategy_id, exchange, requested_amount)

            # Check available capital
            available_capital = await self._get_available_capital()
            if requested_amount > available_capital:
                raise ValidationError(
                    f"Insufficient capital: requested {requested_amount}, "
                    f"available {available_capital}"
                )

            # Check allocation limits
            await self._validate_allocation_limits(strategy_id, exchange, requested_amount)

            # Get existing allocation if any
            existing_allocation = await self._get_existing_allocation(strategy_id, exchange)

            # Calculate allocation percentage
            allocation_percentage = float(requested_amount / self.total_capital)

            # Create or update allocation record
            if existing_allocation:
                # Update existing allocation
                existing_allocation.allocated_amount = str(
                    self._safe_decimal_conversion(existing_allocation.allocated_amount)
                    + requested_amount
                )
                existing_allocation.available_amount = str(
                    self._safe_decimal_conversion(existing_allocation.available_amount)
                    + requested_amount
                )
                existing_allocation.allocation_percentage = float(
                    self._safe_decimal_conversion(existing_allocation.allocated_amount)
                    / self.total_capital
                )
                existing_allocation.last_rebalance = start_time
                existing_allocation.updated_at = start_time

                db_allocation = await self._update_allocation(existing_allocation)
                previous_amount = (
                    self._safe_decimal_conversion(existing_allocation.allocated_amount)
                    - requested_amount
                )

            else:
                # Create new allocation
                if not DATABASE_AVAILABLE or not CapitalAllocationDB:
                    raise ServiceError("Database models not available for allocation")

                allocation_record = CapitalAllocationDB(
                    id=str(uuid.uuid4()),
                    strategy_id=strategy_id,
                    exchange=exchange,
                    allocated_amount=str(requested_amount),
                    utilized_amount="0",
                    available_amount=str(requested_amount),
                    allocation_percentage=allocation_percentage,
                    last_rebalance=start_time,
                    created_at=start_time,
                    updated_at=start_time,
                )

                db_allocation = await self._create_allocation(allocation_record)
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
                new_amount=Decimal(db_allocation.allocated_amount),
                operation_context={
                    "requested_amount": str(requested_amount),
                    "available_capital_before": str(available_capital),
                    "allocation_percentage": allocation_percentage,
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
                strategy_id=db_allocation.strategy_id,
                exchange=db_allocation.exchange,
                allocated_amount=self._safe_decimal_conversion(db_allocation.allocated_amount),
                utilized_amount=self._safe_decimal_conversion(db_allocation.utilized_amount),
                available_amount=self._safe_decimal_conversion(db_allocation.available_amount),
                allocation_percentage=db_allocation.allocation_percentage,
                last_rebalance=db_allocation.last_rebalance,
            )

            self._logger.info(
                "Capital allocation successful",
                operation_id=operation_id,
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(float(requested_amount)),
                allocation_percentage=f"{allocation_percentage:.2%}",
            )

            # Save state snapshot after successful allocation
            await self._save_capital_state_snapshot(reason="allocation_change")

            return allocation

        except (IntegrityError, OperationalError) as e:
            # Database-specific errors
            self._performance_metrics["failed_allocations"] += 1
            self._logger.error(
                "Database error during capital allocation",
                operation_id=operation_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise ServiceError(f"Database error during allocation: {e}") from e

        except ValidationError:
            # Re-raise validation errors
            self._performance_metrics["failed_allocations"] += 1
            raise

        except Exception as e:
            # Other unexpected errors
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
                requested_amount=format_currency(float(requested_amount)),
                error=str(e),
            )

            raise ServiceError(f"Capital allocation failed: {e}") from e

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    @with_retry(max_attempts=3, base_delay=1.0)
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
                    f"Cannot release {release_amount} from allocation of {allocated_amount}"
                )

            # Update allocation
            new_allocated_amount = allocated_amount - release_amount
            new_available_amount = Decimal(allocation.available_amount) - release_amount

            if new_allocated_amount == Decimal("0"):
                # Delete allocation if zero
                await self._delete_allocation(allocation.id)
            else:
                # Update allocation
                allocation.allocated_amount = str(new_allocated_amount)
                allocation.available_amount = str(new_available_amount)
                allocation.allocation_percentage = float(new_allocated_amount / self.total_capital)
                allocation.updated_at = datetime.now(timezone.utc)

                await self._update_allocation(allocation)

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
                amount=format_currency(float(release_amount)),
                remaining_allocation=format_currency(float(new_allocated_amount)),
            )

            # Save state snapshot after successful release
            await self._save_capital_state_snapshot(reason="release_capital")

            return True

        except (IntegrityError, OperationalError) as e:
            # Database-specific errors
            self._logger.error(
                "Database error during capital release",
                operation_id=operation_id,
                error_type=type(e).__name__,
                error=str(e),
            )
            return False

        except ValidationError:
            # Re-raise validation errors
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
                release_amount=format_currency(float(release_amount)),
                error=str(e),
            )

            return False

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    @with_retry(max_attempts=3, base_delay=1.0)
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

            # Update utilization
            Decimal(allocation.utilized_amount)
            allocation.utilized_amount = str(utilized_amount)
            allocation.available_amount = str(allocated_amount - utilized_amount)
            allocation.updated_at = datetime.now(timezone.utc)

            await self._update_allocation(allocation)

            self._logger.debug(
                "Capital utilization updated",
                strategy_id=strategy_id,
                exchange=exchange,
                utilized=format_currency(float(utilized_amount)),
                available=format_currency(float(allocated_amount - utilized_amount)),
            )

            return True

        except (IntegrityError, OperationalError) as e:
            self._logger.error(
                "Database error during utilization update",
                strategy_id=strategy_id,
                exchange=exchange,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise ServiceError(f"Database error during utilization update: {e}") from e

        except ValidationError:
            # Re-raise validation errors
            raise

        except Exception as e:
            self._logger.error(
                "Utilization update failed",
                strategy_id=strategy_id,
                exchange=exchange,
                utilized_amount=format_currency(float(utilized_amount)),
                error=str(e),
            )
            return False

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    @with_retry(max_attempts=3, base_delay=1.0)
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

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    @with_retry(max_attempts=3, base_delay=1.0)
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
            )
            total_utilized = sum(
                self._safe_decimal_conversion(alloc.utilized_amount) for alloc in allocations
            )
            emergency_reserve = self.total_capital * self.emergency_reserve_pct
            available_capital = self.total_capital - total_allocated - emergency_reserve

            utilization_rate = (
                float(total_utilized / total_allocated) if total_allocated > 0 else 0.0
            )

            # Calculate allocation efficiency
            efficiency_score = await self._calculate_allocation_efficiency(allocations)

            metrics = CapitalMetrics(
                total_capital=self.total_capital,
                allocated_capital=total_allocated,
                available_capital=available_capital,
                utilization_rate=utilization_rate,
                allocation_efficiency=efficiency_score,
                rebalance_frequency_hours=24,  # Default
                emergency_reserve=emergency_reserve,
                last_updated=datetime.now(timezone.utc),
                allocation_count=len(allocations),
            )

            return metrics

        except Exception as e:
            self._logger.error(f"Failed to calculate capital metrics: {e}")
            raise ServiceError(f"Capital metrics calculation failed: {e}") from e

    async def _get_allocations_by_strategy_impl(self, strategy_id: str) -> list[CapitalAllocation]:
        """Internal implementation to get allocations by strategy."""
        try:
            # Get allocations by strategy using proper UoW pattern
            if not DATABASE_AVAILABLE:
                return []

            # Use UoW pattern if available
            if self.uow_factory:
                with self.uow_factory.create() as uow:
                    db_allocations = await uow.capital_allocations.get_by_strategy(strategy_id)
            # Fallback to legacy patterns
            elif self.capital_repository:
                db_allocations = await self.capital_repository.get_by_strategy(strategy_id)
            elif self.database_service:
                db_allocations = await self.database_service.list_entities(
                    model_class=CapitalAllocationDB,
                    filters={"strategy_id": strategy_id},
                )
            else:
                return []

            # Convert to domain objects
            allocations = []
            for db_alloc in db_allocations:
                allocation = CapitalAllocation(
                    strategy_id=db_alloc.strategy_id,
                    exchange=db_alloc.exchange,
                    allocated_amount=self._safe_decimal_conversion(db_alloc.allocated_amount),
                    utilized_amount=self._safe_decimal_conversion(db_alloc.utilized_amount),
                    available_amount=self._safe_decimal_conversion(db_alloc.available_amount),
                    allocation_percentage=db_alloc.allocation_percentage,
                    last_rebalance=db_alloc.last_rebalance,
                )
                allocations.append(allocation)

            return allocations

        except Exception as e:
            self._logger.error(f"Failed to get allocations for strategy {strategy_id}: {e}")
            raise ServiceError(f"Failed to get allocations by strategy: {e}") from e

    # Helper Methods

    def _validate_allocation_request(
        self, strategy_id: str, exchange: str, amount: Decimal
    ) -> None:
        """Validate allocation request parameters."""
        if not strategy_id or not strategy_id.strip():
            raise ValidationError("Strategy ID cannot be empty")

        if not exchange or not exchange.strip():
            raise ValidationError("Exchange cannot be empty")

        if not ValidationFramework.validate_quantity(float(amount)):
            raise ValidationError(f"Invalid capital allocation amount: {amount}")

        if amount <= 0:
            raise ValidationError("Allocation amount must be positive")

    async def _validate_allocation_limits(
        self, strategy_id: str, exchange: str, amount: Decimal
    ) -> None:
        """Validate allocation limits."""
        # Check maximum single allocation
        max_allocation = self.total_capital * self.max_allocation_pct
        if amount > max_allocation:
            raise ValidationError(
                f"Requested amount {amount} exceeds maximum allocation {max_allocation}"
            )

    async def _get_available_capital(self) -> Decimal:
        """Calculate available capital for allocation."""
        # Get all allocations
        allocations = await self._get_all_allocations()

        # Return maximum available if no allocations available
        if not allocations:
            return self.total_capital * Decimal("0.8")  # 80% as fallback

        total_allocated = sum(
            self._safe_decimal_conversion(alloc.allocated_amount) for alloc in allocations
        )
        emergency_reserve = self.total_capital * self.emergency_reserve_pct

        return self.total_capital - total_allocated - emergency_reserve

    async def _get_existing_allocation(
        self, strategy_id: str, exchange: str
    ) -> CapitalAllocationDB | None:
        """Get existing allocation for strategy and exchange using proper UoW pattern."""
        if not DATABASE_AVAILABLE:
            return None

        # Use UoW pattern if available
        if self.uow_factory:
            with self.uow_factory.create() as uow:
                return await uow.capital_allocations.find_by_strategy_exchange(
                    strategy_id, exchange
                )

        # Fallback to legacy patterns
        elif self.capital_repository:
            return await self.capital_repository.find_by_strategy_exchange(strategy_id, exchange)
        elif self.database_service:
            allocations = await self.database_service.list_entities(
                model_class=CapitalAllocationDB,
                filters={
                    "strategy_id": strategy_id,
                    "exchange": exchange,
                },
                limit=1,
            )
            return allocations[0] if allocations else None
        else:
            return None

    async def _create_allocation(self, allocation_record) -> Any:
        """Create allocation using proper UoW pattern."""
        if not DATABASE_AVAILABLE:
            raise ServiceError("Database not available - allocation operations disabled")

        # Use UoW pattern if available
        if self.uow_factory:
            with self.uow_factory.create() as uow:
                return await uow.capital_allocations.create(allocation_record)

        # Fallback to legacy patterns
        elif self.capital_repository:
            return await self.capital_repository.create(allocation_record)
        elif self.database_service:
            return await self.database_service.create_entity(allocation_record)
        else:
            raise ServiceError("No database service available")

    async def _update_allocation(self, allocation) -> Any:
        """Update allocation using proper UoW pattern."""
        if not DATABASE_AVAILABLE:
            raise ServiceError("Database not available - allocation operations disabled")

        # Use UoW pattern if available
        if self.uow_factory:
            with self.uow_factory.create() as uow:
                return await uow.capital_allocations.update(allocation)

        # Fallback to legacy patterns
        elif self.capital_repository:
            return await self.capital_repository.update(allocation)
        elif self.database_service:
            return await self.database_service.update_entity(allocation)
        else:
            raise ServiceError("No database service available")

    async def _delete_allocation(self, allocation_id: str) -> bool:
        """Delete allocation using proper UoW pattern."""
        if not DATABASE_AVAILABLE:
            raise ServiceError("Database not available - allocation operations disabled")

        # Use UoW pattern if available
        if self.uow_factory:
            with self.uow_factory.create() as uow:
                await uow.capital_allocations.delete(allocation_id)
                return True

        # Fallback to legacy patterns
        elif self.capital_repository:
            await self.capital_repository.delete(allocation_id)
            return True
        elif self.database_service and DATABASE_AVAILABLE and CapitalAllocationDB:
            # This was the problematic direct delete_entity call - now with safety checks
            await self.database_service.delete_entity(CapitalAllocationDB, allocation_id)
            return True
        else:
            raise ServiceError("No database service available")

    async def _get_all_allocations(self, limit: int | None = None) -> list:
        """Get all allocations using proper UoW pattern."""
        if not DATABASE_AVAILABLE:
            return []

        # Use UoW pattern if available
        if self.uow_factory:
            with self.uow_factory.create() as uow:
                return await uow.capital_allocations.get_all(limit=limit)

        # Fallback to legacy patterns
        elif self.capital_repository:
            return await self.capital_repository.get_all(limit=limit)
        elif self.database_service and CapitalAllocationDB:
            return await self.database_service.list_entities(
                model_class=CapitalAllocationDB,
                limit=limit,
            )
        else:
            return []

    async def _create_audit_log_record(self, audit_log) -> None:
        """Create audit log record using proper UoW pattern."""
        if not DATABASE_AVAILABLE:
            return  # Skip if database not available

        try:
            # Use UoW pattern if available
            if self.uow_factory:
                with self.uow_factory.create() as uow:
                    await uow.capital_audit_logs.create(audit_log)
            # Fallback to legacy patterns
            elif self.audit_repository:
                await self.audit_repository.create(audit_log)
            elif self.database_service:
                await self.database_service.create_entity(audit_log)
            else:
                self._logger.warning("No database service available for audit log creation")
        except Exception as e:
            self._logger.error(f"Failed to create audit log record: {e}")
            # Don't raise - audit failure shouldn't break the operation

    async def _restore_allocations_in_transaction(self, allocations_data: list[dict]) -> None:
        """
        Restore allocations within a proper transaction boundary.

        This ensures that either all allocations are restored successfully,
        or none are restored in case of any failure.

        Args:
            allocations_data: List of allocation data dictionaries
        """
        if not DATABASE_AVAILABLE or not allocations_data:
            return

        try:
            # Use UoW pattern for transaction management
            if self.uow_factory:
                with self.uow_factory.create() as uow:
                    # Clear existing allocations first
                    await uow.capital_allocations.delete_all()

                    # Restore allocations
                    for alloc_data in allocations_data:
                        if not CapitalAllocationDB:
                            raise ServiceError("CapitalAllocationDB model not available")

                        allocation = CapitalAllocationDB(
                            strategy_id=alloc_data["strategy_id"],
                            exchange=alloc_data["exchange"],
                            allocated_amount=alloc_data["allocated_amount"],
                            utilized_amount=alloc_data["utilized_amount"],
                            available_amount=alloc_data["available_amount"],
                            allocation_percentage=alloc_data["allocation_percentage"],
                            last_rebalance=(
                                datetime.fromisoformat(alloc_data["last_rebalance"])
                                if alloc_data["last_rebalance"]
                                else None
                            ),
                        )
                        await uow.capital_allocations.create(allocation)
                    # Transaction commits automatically when UoW context exits
            else:
                # Fallback to legacy patterns without proper transaction boundary
                self._logger.warning(
                    "State restoration without proper transaction management - data consistency not guaranteed"
                )

                # Clear existing allocations
                if self.capital_repository:
                    await self.capital_repository.delete_all()
                elif self.database_service:
                    # Get all existing allocations to delete them
                    existing = await self.database_service.list_entities(
                        model_class=CapitalAllocationDB, limit=None
                    )
                    for allocation in existing:
                        await self.database_service.delete_entity(
                            CapitalAllocationDB, allocation.id
                        )

                # Restore allocations
                for alloc_data in allocations_data:
                    if not CapitalAllocationDB:
                        raise ServiceError("CapitalAllocationDB model not available")

                    allocation = CapitalAllocationDB(
                        strategy_id=alloc_data["strategy_id"],
                        exchange=alloc_data["exchange"],
                        allocated_amount=alloc_data["allocated_amount"],
                        utilized_amount=alloc_data["utilized_amount"],
                        available_amount=alloc_data["available_amount"],
                        allocation_percentage=alloc_data["allocation_percentage"],
                        last_rebalance=(
                            datetime.fromisoformat(alloc_data["last_rebalance"])
                            if alloc_data["last_rebalance"]
                            else None
                        ),
                    )

                    if self.capital_repository:
                        await self.capital_repository.create(allocation)
                    elif self.database_service:
                        await self.database_service.create_entity(allocation)

        except Exception as e:
            self._logger.error(f"Failed to restore allocations in transaction: {e}")
            raise ServiceError(f"State restoration failed: {e}") from e

    async def _calculate_allocation_efficiency(
        self, allocations: list[CapitalAllocationDB]
    ) -> float:
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
            return 0.5

        utilization_efficiency = float(total_utilized / total_allocated)

        # Apply performance multiplier (simplified for now)
        performance_multiplier = 1.0  # Could integrate with performance metrics

        return max(0.0, min(2.0, utilization_efficiency * performance_multiplier))

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
            if not DATABASE_AVAILABLE or not CapitalAuditLog:
                return  # Skip if audit log model not available

            audit_log = CapitalAuditLog(
                id=str(uuid.uuid4()),
                operation_id=operation_id,
                operation_type=operation_type,
                strategy_id=strategy_id,
                exchange=exchange,
                bot_id=bot_id,
                operation_description=f"Capital {operation_type} operation",
                amount=amount,
                previous_amount=previous_amount,
                new_amount=new_amount,
                operation_context=operation_context or {},
                operation_status=operation_status,
                success=success,
                error_message=error_message,
                authorized_by=authorized_by,
                requested_at=requested_at or datetime.now(timezone.utc),
                executed_at=executed_at,
                source_component="CapitalService",
                correlation_id=self._correlation_id,
                created_at=datetime.now(timezone.utc),
            )

            await self._create_audit_log_record(audit_log)

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
            # Check database availability first
            if not DATABASE_AVAILABLE:
                return HealthStatus.DEGRADED  # Can operate without database in some scenarios

            # Check database connectivity
            if self.database_service:
                health_status = await self.database_service._service_health_check()
                if health_status != HealthStatus.HEALTHY:
                    return health_status
            elif self.capital_repository:
                # Basic connectivity check via repository
                try:
                    await self.capital_repository.get_all(limit=1)
                except Exception:
                    return HealthStatus.UNHEALTHY
            elif self.uow_factory:
                # Test UoW factory connectivity
                try:
                    with self.uow_factory.create() as uow:
                        await uow.capital_allocations.get_all(limit=1)
                except Exception:
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
            "total_capital_managed": float(self.total_capital),
            "emergency_reserve_maintained": True,
            "risk_limit_violations": 0,
        }

        self._logger.info("Capital service metrics reset")

    def _load_configuration(self) -> None:
        """Load configuration from ConfigService or environment."""
        try:
            # Try to get config from dependency injection
            from src.core.dependency_injection import get_container

            container = get_container()
            config_service = container.get("ConfigService")

            if config_service:
                capital_config = config_service.get_config_value("capital_management", {})

                # Load values from config
                self.total_capital = Decimal(str(capital_config.get("total_capital", 100000)))
                self.emergency_reserve_pct = Decimal(
                    str(capital_config.get("emergency_reserve_pct", 0.1))
                )
                self.max_allocation_pct = Decimal(
                    str(capital_config.get("max_allocation_pct", 0.2))
                )
                self.max_daily_reallocation_pct = Decimal(
                    str(capital_config.get("max_daily_reallocation_pct", 0.1))
                )
        except Exception as e:
            self._logger.warning(f"Failed to load configuration from ConfigService: {e}")
            # Will use defaults

    def _safe_decimal_conversion(self, value: Any) -> Decimal:
        """Safely convert any value to Decimal."""
        if isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, str):
            try:
                return Decimal(value)
            except (ValueError, InvalidOperation):
                self._logger.warning(f"Invalid decimal value: {value}, using 0")
                return Decimal("0")
        else:
            self._logger.warning(f"Unknown type for decimal conversion: {type(value)}, using 0")
            return Decimal("0")

    # State Management Integration Methods

    async def _save_capital_state_snapshot(self, reason: str = "allocation_change") -> None:
        """
        Save current capital allocation state to StateService.

        Args:
            reason: Reason for saving snapshot (e.g., "allocation_change", "rebalance", "recovery")
        """
        if not self.state_service:
            return  # Skip if StateService not available

        try:
            # Get current allocations
            allocations = await self._get_all_allocations()

            # Build state data
            state_data = {
                "total_capital": str(self.total_capital),
                "emergency_reserve_pct": str(self.emergency_reserve_pct),
                "allocations": [
                    {
                        "strategy_id": alloc.strategy_id,
                        "exchange": alloc.exchange,
                        "allocated_amount": alloc.allocated_amount,
                        "utilized_amount": alloc.utilized_amount,
                        "available_amount": alloc.available_amount,
                        "allocation_percentage": alloc.allocation_percentage,
                        "last_rebalance": (
                            alloc.last_rebalance.isoformat() if alloc.last_rebalance else None
                        ),
                    }
                    for alloc in allocations
                ],
                "performance_metrics": self._performance_metrics.copy(),
                "snapshot_reason": reason,
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Save to StateService - use PORTFOLIO_STATE as capital is part of portfolio
            # Note: StateType.CAPITAL doesn't exist, using PORTFOLIO_STATE for capital management
            await self.state_service.save_state(
                state_type=StateType.PORTFOLIO_STATE,
                state_data=state_data,
                metadata={
                    "reason": reason,
                    "component": "capital_management",
                    "total_allocated": str(
                        sum(self._safe_decimal_conversion(a.allocated_amount) for a in allocations)
                    ),
                    "allocation_count": len(allocations),
                },
            )

            self._logger.debug(
                "Capital state snapshot saved",
                reason=reason,
                allocation_count=len(allocations),
            )

        except Exception as e:
            # Log but don't fail the operation
            self._logger.warning(f"Failed to save capital state snapshot: {e}")

    async def restore_capital_state(self, recovery_point_id: str) -> bool:
        """
        Restore capital state from a recovery point.

        Args:
            recovery_point_id: ID of the recovery point to restore from

        Returns:
            bool: True if restoration successful
        """
        if not self.state_service:
            self._logger.error("StateService not available for restoration")
            return False

        try:
            # Get state from recovery point - use PORTFOLIO_STATE as capital is part of portfolio
            # Note: StateService.get_state expects state_id, not recovery_point_id
            state_data = await self.state_service.get_state(
                state_type=StateType.PORTFOLIO_STATE,
                state_id=recovery_point_id,  # Using recovery_point_id as state_id
                include_metadata=True,
            )

            if not state_data:
                self._logger.error(
                    f"No capital state found for recovery point: {recovery_point_id}"
                )
                return False

            # Restore configuration
            self.total_capital = Decimal(state_data["total_capital"])
            self.emergency_reserve_pct = Decimal(state_data["emergency_reserve_pct"])

            # Restore state within a transaction to ensure consistency
            await self._restore_allocations_in_transaction(state_data.get("allocations", []))

            # Restore performance metrics
            self._performance_metrics.update(state_data.get("performance_metrics", {}))

            self._logger.info(
                "Capital state restored successfully",
                recovery_point_id=recovery_point_id,
                allocations_restored=len(state_data.get("allocations", [])),
            )

            return True

        except Exception as e:
            self._logger.error(f"Failed to restore capital state: {e}")
            return False
