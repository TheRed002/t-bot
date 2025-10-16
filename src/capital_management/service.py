"""
Capital Management Service.

Simple capital management operations with database access and basic validation.
"""

import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Any

import structlog

from src.capital_management.constants import (
    DEFAULT_TOTAL_CAPITAL,
    EMERGENCY_RESERVE_PCT,
    FINANCIAL_DECIMAL_PRECISION,
    MAX_ALLOCATION_PCT,
    PERCENTAGE_MULTIPLIER,
)
from src.capital_management.interfaces import (
    AbstractCapitalService,
)
from src.core.base.service import TransactionalService
from src.core.exceptions import (
    CapitalAllocationError,
    ServiceError,
    ValidationError,
)
from src.capital_management.data_transformer import CapitalDataTransformer
from src.core.logging import get_logger
from src.core.types import CapitalAllocation, CapitalMetrics
from src.error_handling.decorators import with_circuit_breaker, with_retry
from src.utils.messaging_patterns import ErrorPropagationMixin

# Set decimal context for financial precision
getcontext().prec = FINANCIAL_DECIMAL_PRECISION
getcontext().rounding = ROUND_HALF_UP


class CapitalService(AbstractCapitalService, TransactionalService, ErrorPropagationMixin):
    """
    Simple capital management service.

    Provides basic capital allocation, release, and metrics operations.
    """

    def __init__(
        self,
        capital_repository: Any = None,
        audit_repository: Any = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize capital service.

        Args:
            capital_repository: Repository for capital allocation operations
            audit_repository: Repository for audit logging
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(name="CapitalService", correlation_id=correlation_id)
        self._logger: structlog.BoundLogger = get_logger(self.__class__.__name__)

        # Repository dependencies
        self._capital_repository = capital_repository
        self._audit_repository = audit_repository

        # Basic configuration
        self.total_capital = DEFAULT_TOTAL_CAPITAL
        self.emergency_reserve_pct = EMERGENCY_RESERVE_PCT
        self.max_allocation_pct = MAX_ALLOCATION_PCT

        # Simple metrics tracking
        self._allocations_count = 0
        self._releases_count = 0

        self._logger.info("CapitalService initialized")

    async def _do_start(self) -> None:
        """Start the capital service."""
        self._logger.info("CapitalService started")

    async def _do_stop(self) -> None:
        """Stop the capital service."""
        self._logger.info("CapitalService stopped")

    @with_circuit_breaker()
    @with_retry(max_attempts=3)
    async def allocate_capital(
        self,
        strategy_id: str,
        exchange: str,
        requested_amount: Decimal,
        min_allocation: Decimal | None = None,
        max_allocation: Decimal | None = None,
        target_allocation_pct: Decimal | None = None,
        authorized_by: str | None = None,
        bot_id: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> CapitalAllocation:
        """
        Allocate capital to a strategy on an exchange.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            requested_amount: Amount to allocate
            min_allocation: Minimum allocation (optional)
            max_allocation: Maximum allocation (optional)
            target_allocation_pct: Target allocation percentage (optional)
            authorized_by: Authorization identifier (optional)
            bot_id: Bot instance ID (optional)
            risk_context: Risk context information (optional)

        Returns:
            CapitalAllocation object

        Raises:
            ValidationError: If allocation parameters are invalid
            CapitalAllocationError: If allocation fails
        """
        # Basic validation
        if requested_amount <= 0:
            raise ValidationError("Allocation amount must be positive")

        if not strategy_id or not exchange:
            raise ValidationError("Strategy ID and exchange are required")

        # Check allocation limits
        available_capital = await self._get_available_capital()
        if requested_amount > available_capital:
            raise CapitalAllocationError(
                f"Insufficient capital: requested {requested_amount}, available {available_capital}"
            )

        max_allowed = self.total_capital * self.max_allocation_pct
        if requested_amount > max_allowed:
            raise CapitalAllocationError(f"Amount exceeds maximum allocation limit: {max_allowed}")

        try:
            # Create capital allocation object
            allocation = CapitalAllocation(
                allocation_id=str(uuid.uuid4()),
                strategy_id=strategy_id,
                exchange=exchange,
                symbol="",
                allocated_amount=requested_amount,
                utilized_amount=Decimal("0"),
                available_amount=requested_amount,
                allocation_percentage=(requested_amount / self.total_capital) * PERCENTAGE_MULTIPLIER,
                target_allocation_pct=target_allocation_pct or Decimal("0"),
                min_allocation=min_allocation or Decimal("0"),
                max_allocation=max_allocation or requested_amount,
                last_rebalance=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            # Persist allocation if repository is available
            if self._capital_repository:
                allocation_data = {
                    "id": allocation.allocation_id,
                    "strategy_id": allocation.strategy_id,
                    "exchange": allocation.exchange,
                    "symbol": allocation.symbol,
                    "allocated_amount": allocation.allocated_amount,
                    "utilized_amount": allocation.utilized_amount,
                    "available_amount": allocation.available_amount,
                    "allocation_percentage": allocation.allocation_percentage,
                    "target_allocation_pct": allocation.target_allocation_pct,
                    "min_allocation": allocation.min_allocation,
                    "max_allocation": allocation.max_allocation,
                    "last_rebalance": allocation.last_rebalance,
                    "created_at": allocation.created_at,
                    "updated_at": allocation.updated_at,
                }
                await self._capital_repository.create(allocation_data)

            # Create audit log if repository is available
            if self._audit_repository:
                audit_data = {
                    "allocation_id": allocation.allocation_id,
                    "operation": "allocate_capital",
                    "strategy_id": strategy_id,
                    "exchange": exchange,
                    "amount": requested_amount,
                    "authorized_by": authorized_by,
                    "bot_id": bot_id,
                    "timestamp": datetime.now(timezone.utc),
                }
                await self._audit_repository.create(audit_data)

            self._allocations_count += 1
            self._logger.info(
                f"Capital allocated: {requested_amount} to {strategy_id} on {exchange}"
            )

            return allocation

        except Exception as e:
            # Use consistent error propagation aligned with risk_management module

            # Check if it's a validation error and propagate accordingly like risk_management
            if hasattr(e, "__class__") and (
                "ValidationError" in e.__class__.__name__
                or "DataValidationError" in e.__class__.__name__
            ):
                self.propagate_validation_error(e, "capital_allocation_validation")
            else:
                self.propagate_service_error(e, "capital_allocation")

            # Still transform for local logging but with aligned format
            error_data = CapitalDataTransformer.transform_error_to_event_data(
                error=e,
                operation="allocate_capital",
                strategy_id=strategy_id,
                metadata={"exchange": exchange, "requested_amount": str(requested_amount)}
            )
            self._logger.error(f"Capital allocation failed: {error_data}")
            raise CapitalAllocationError(f"Failed to allocate capital: {e}") from e

    @with_circuit_breaker()
    @with_retry(max_attempts=3)
    async def release_capital(
        self,
        strategy_id: str,
        exchange: str,
        release_amount: Decimal,
        authorized_by: str | None = None,
        bot_id: str | None = None,
    ) -> bool:
        """
        Release capital from a strategy allocation.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            release_amount: Amount to release
            authorized_by: Authorization identifier (optional)
            bot_id: Bot instance ID (optional)

        Returns:
            True if successful

        Raises:
            ValidationError: If release parameters are invalid
            CapitalAllocationError: If release fails
        """
        if release_amount <= 0:
            raise ValidationError("Release amount must be positive")

        try:
            # Service logic - validate release amount
            if release_amount > self.total_capital:
                raise CapitalAllocationError(
                    f"Cannot release {release_amount}, exceeds total capital {self.total_capital}"
                )

            # Find and release allocation if repository is available
            if self._capital_repository:
                existing_allocation = await self._capital_repository.get_by_strategy_exchange(strategy_id, exchange)
                if existing_allocation:
                    allocation_id = existing_allocation.get("id") if isinstance(existing_allocation, dict) else getattr(existing_allocation, "allocation_id", None)
                    if allocation_id:
                        await self._capital_repository.delete(allocation_id)

            # Create audit log if repository is available
            if self._audit_repository:
                audit_data = {
                    "operation": "release_capital",
                    "strategy_id": strategy_id,
                    "exchange": exchange,
                    "amount": release_amount,
                    "authorized_by": authorized_by,
                    "bot_id": bot_id,
                    "timestamp": datetime.now(timezone.utc),
                }
                await self._audit_repository.create(audit_data)

            self._releases_count += 1
            self._logger.info(
                f"Capital released: {release_amount} from {strategy_id} on {exchange}"
            )

            return True

        except Exception as e:
            # Use consistent error propagation aligned with risk_management module

            # Check if it's a validation error and propagate accordingly like risk_management
            if hasattr(e, "__class__") and (
                "ValidationError" in e.__class__.__name__
                or "DataValidationError" in e.__class__.__name__
            ):
                self.propagate_validation_error(e, "capital_release_validation")
            else:
                self.propagate_service_error(e, "capital_release")

            # Still transform for local logging but with aligned format
            error_data = CapitalDataTransformer.transform_error_to_event_data(
                error=e,
                operation="release_capital",
                strategy_id=strategy_id,
                metadata={"exchange": exchange, "requested_amount": str(release_amount)}
            )
            self._logger.error(f"Capital release failed: {error_data}")
            raise CapitalAllocationError(f"Failed to release capital: {e}") from e

    async def update_utilization(
        self,
        strategy_id: str,
        exchange: str,
        utilized_amount: Decimal,
        authorized_by: str | None = None,
    ) -> bool:
        """
        Update capital utilization for a strategy allocation.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            utilized_amount: New utilized amount
            authorized_by: Authorization identifier (optional)

        Returns:
            True if successful
        """
        try:
            if utilized_amount > self.total_capital:
                raise ValidationError("Utilized amount cannot exceed total capital")

            # Update allocation if repository is available
            if self._capital_repository:
                existing_allocation = await self._capital_repository.get_by_strategy_exchange(strategy_id, exchange)
                if existing_allocation:
                    allocation_data = existing_allocation.copy() if isinstance(existing_allocation, dict) else {
                        "id": getattr(existing_allocation, "allocation_id", None),
                        "strategy_id": strategy_id,
                        "exchange": exchange,
                        "utilized_amount": utilized_amount,
                        "updated_at": datetime.now(timezone.utc),
                    }
                    allocation_data["utilized_amount"] = utilized_amount
                    allocation_data["updated_at"] = datetime.now(timezone.utc)
                    await self._capital_repository.update(allocation_data)

            # Create audit log if repository is available
            if self._audit_repository:
                audit_data = {
                    "operation": "update_utilization",
                    "strategy_id": strategy_id,
                    "exchange": exchange,
                    "amount": utilized_amount,
                    "authorized_by": authorized_by,
                    "timestamp": datetime.now(timezone.utc),
                }
                await self._audit_repository.create(audit_data)

            self._logger.info(f"Utilization updated for {strategy_id} on {exchange}: {utilized_amount}")
            return True

        except Exception as e:
            # Use consistent error propagation aligned with risk_management module

            # Check if it's a validation error and propagate accordingly like risk_management
            if hasattr(e, "__class__") and (
                "ValidationError" in e.__class__.__name__
                or "DataValidationError" in e.__class__.__name__
            ):
                self.propagate_validation_error(e, "capital_utilization_validation")
            else:
                self.propagate_service_error(e, "capital_utilization")

            # Still transform for local logging but with aligned format
            error_data = CapitalDataTransformer.transform_error_to_event_data(
                error=e,
                operation="update_utilization",
                strategy_id=strategy_id,
                metadata={"exchange": exchange, "utilized_amount": str(utilized_amount)}
            )
            self._logger.error(f"Utilization update failed: {error_data}")
            raise CapitalAllocationError(f"Failed to update utilization: {e}") from e

    async def get_capital_metrics(self) -> CapitalMetrics:
        """
        Get current capital metrics.

        Returns:
            CapitalMetrics object
        """
        try:
            # Calculate metrics from actual allocations if repository is available
            allocated_amount = Decimal("0")
            strategies_active = 0

            if self._capital_repository:
                allocations = await self._capital_repository.get_all()
                if allocations:
                    for allocation in allocations:
                        if isinstance(allocation, dict):
                            allocated_amount += allocation.get("allocated_amount", Decimal("0"))
                        else:
                            allocated_amount += getattr(allocation, "allocated_amount", Decimal("0"))
                    strategies_active = len(set(
                        allocation.get("strategy_id") if isinstance(allocation, dict)
                        else getattr(allocation, "strategy_id", "")
                        for allocation in allocations
                    ))

            available_amount = self.total_capital - allocated_amount

            return CapitalMetrics(
                total_capital=self.total_capital,
                allocated_amount=allocated_amount,
                available_amount=available_amount,
                total_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                daily_return=Decimal("0"),
                weekly_return=Decimal("0"),
                monthly_return=Decimal("0"),
                yearly_return=Decimal("0"),
                total_return=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                sortino_ratio=Decimal("0"),
                calmar_ratio=Decimal("0"),
                current_drawdown=Decimal("0"),
                max_drawdown=Decimal("0"),
                var_95=Decimal("0"),
                expected_shortfall=Decimal("0"),
                strategies_active=strategies_active,
                positions_open=0,
                leverage_used=Decimal("0"),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            # Use consistent error propagation aligned with risk_management module

            # Check if it's a validation error and propagate accordingly like risk_management
            if hasattr(e, "__class__") and (
                "ValidationError" in e.__class__.__name__
                or "DataValidationError" in e.__class__.__name__
            ):
                self.propagate_validation_error(e, "capital_metrics_validation")
            else:
                self.propagate_service_error(e, "capital_metrics")

            # Still transform for local logging but with aligned format
            error_data = CapitalDataTransformer.transform_error_to_event_data(
                error=e,
                operation="get_capital_metrics",
                metadata={"operation_context": "metrics_retrieval"}
            )
            self._logger.error(f"Failed to get capital metrics: {error_data}")
            raise ServiceError(f"Failed to get capital metrics: {e}") from e

    async def get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]:
        """Get all allocations for a strategy."""
        try:
            if not self._capital_repository:
                self._logger.warning("No capital repository available, returning empty list")
                return []

            allocations_data = await self._capital_repository.get_by_strategy(strategy_id)
            return [CapitalAllocation(**data) if isinstance(data, dict) else data for data in allocations_data]
        except Exception as e:
            self._logger.error(f"Failed to get allocations for strategy {strategy_id}: {e}")
            raise ServiceError(f"Failed to get allocations for strategy: {e}") from e

    async def get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]:
        """Get all allocations."""
        try:
            if not self._capital_repository:
                self._logger.warning("No capital repository available, returning empty list")
                return []

            allocations_data = await self._capital_repository.get_all(limit)
            return [CapitalAllocation(**data) if isinstance(data, dict) else data for data in allocations_data]
        except Exception as e:
            self._logger.error(f"Failed to get all allocations: {e}")
            raise ServiceError(f"Failed to get all allocations: {e}") from e

    # Private helper methods
    async def _get_available_capital(self) -> Decimal:
        """Get available capital for allocation."""
        emergency_reserve = self.total_capital * self.emergency_reserve_pct
        allocated_amount = Decimal("0")

        # Calculate currently allocated amount from repository
        if self._capital_repository:
            try:
                allocations = await self._capital_repository.get_all()
                for allocation in allocations:
                    if isinstance(allocation, dict):
                        allocated_amount += allocation.get("allocated_amount", Decimal("0"))
                    else:
                        allocated_amount += getattr(allocation, "allocated_amount", Decimal("0"))
            except Exception as e:
                self._logger.warning(f"Failed to get allocations for available capital calculation: {e}")

        return self.total_capital - emergency_reserve - allocated_amount
