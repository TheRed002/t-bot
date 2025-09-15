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
    AuditRepositoryProtocol,
    CapitalRepositoryProtocol,
)
from src.core.base.service import TransactionalService
from src.core.exceptions import (
    AllocationError,
    ServiceError,
    ValidationError,
)
from src.core.logging import get_logger
from src.core.types import CapitalAllocation, CapitalMetrics
from src.error_handling.decorators import with_circuit_breaker, with_retry

# Set decimal context for financial precision
getcontext().prec = FINANCIAL_DECIMAL_PRECISION
getcontext().rounding = ROUND_HALF_UP


class CapitalService(AbstractCapitalService, TransactionalService):
    """
    Simple capital management service.

    Provides basic capital allocation, release, and metrics operations.
    """

    def __init__(
        self,
        capital_repository: CapitalRepositoryProtocol | None = None,
        audit_repository: AuditRepositoryProtocol | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize capital service.

        Args:
            capital_repository: Capital repository implementation
            audit_repository: Audit repository implementation
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(name="CapitalService", correlation_id=correlation_id)

        self._capital_repository = capital_repository
        self._audit_repository = audit_repository
        self._logger: structlog.BoundLogger = get_logger(self.__class__.__name__)

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
        if self._capital_repository:
            try:
                allocations = await self._capital_repository.get_all()
                self._allocations_count = len(allocations)
                self._logger.info(
                    f"CapitalService started with {self._allocations_count} existing allocations"
                )
            except Exception as e:
                self._logger.error(f"Failed to load existing allocations: {e}")
                raise ServiceError(f"Capital service startup failed: {e}") from e
        else:
            self._logger.warning("No capital repository available")

    async def _do_stop(self) -> None:
        """Stop the capital service."""
        self._logger.info("CapitalService stopped")

    @with_circuit_breaker
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
            AllocationError: If allocation fails
        """
        # Basic validation
        if requested_amount <= 0:
            raise ValidationError("Allocation amount must be positive")

        if not strategy_id or not exchange:
            raise ValidationError("Strategy ID and exchange are required")

        # Check allocation limits
        available_capital = await self._get_available_capital()
        if requested_amount > available_capital:
            raise AllocationError(
                f"Insufficient capital: requested {requested_amount}, available {available_capital}"
            )

        max_allowed = self.total_capital * self.max_allocation_pct
        if requested_amount > max_allowed:
            raise AllocationError(f"Amount exceeds maximum allocation limit: {max_allowed}")

        try:
            # Check for existing allocation
            existing = await self._get_existing_allocation(strategy_id, exchange)

            allocation_data = {
                "allocation_id": str(uuid.uuid4()),
                "strategy_id": strategy_id,
                "exchange": exchange,
                "allocated_amount": requested_amount,
                "utilized_amount": Decimal("0"),
                "available_amount": requested_amount,
                "allocation_percentage": (requested_amount / self.total_capital)
                * PERCENTAGE_MULTIPLIER,
                "target_allocation_pct": target_allocation_pct or Decimal("0"),
                "min_allocation": min_allocation or Decimal("0"),
                "max_allocation": max_allocation or requested_amount,
                "last_rebalance": datetime.now(timezone.utc),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }

            if not self._capital_repository:
                raise ServiceError("Capital repository not available")

            if existing:
                # Update existing allocation
                allocation_data["allocation_id"] = existing.allocation_id
                allocation_data["allocated_amount"] = existing.allocated_amount + requested_amount
                allocation_data["available_amount"] = existing.available_amount + requested_amount
                result = await self._capital_repository.update(allocation_data)
            else:
                # Create new allocation
                result = await self._capital_repository.create(allocation_data)

            # Create audit log
            if self._audit_repository:
                await self._create_audit_log("allocation_created", allocation_data, authorized_by)

            self._allocations_count += 1
            self._logger.info(
                f"Capital allocated: {requested_amount} to {strategy_id} on {exchange}"
            )

            return result

        except Exception as e:
            self._logger.error(f"Capital allocation failed: {e}")
            raise AllocationError(f"Failed to allocate capital: {e}") from e

    @with_circuit_breaker
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
            AllocationError: If release fails
        """
        if release_amount <= 0:
            raise ValidationError("Release amount must be positive")

        try:
            existing = await self._get_existing_allocation(strategy_id, exchange)
            if not existing:
                raise AllocationError(f"No allocation found for {strategy_id} on {exchange}")

            if release_amount > existing.available_amount:
                raise AllocationError(
                    f"Cannot release {release_amount}, only {existing.available_amount} available"
                )

            # Update allocation
            allocation_data = {
                "allocation_id": existing.allocation_id,
                "strategy_id": strategy_id,
                "exchange": exchange,
                "allocated_amount": existing.allocated_amount - release_amount,
                "utilized_amount": existing.utilized_amount,
                "available_amount": existing.available_amount - release_amount,
                "allocation_percentage": (
                    (existing.allocated_amount - release_amount) / self.total_capital
                )
                * PERCENTAGE_MULTIPLIER,
                "target_allocation_pct": existing.target_allocation_pct,
                "min_allocation": existing.min_allocation,
                "max_allocation": existing.max_allocation,
                "last_rebalance": existing.last_rebalance,
                "updated_at": datetime.now(timezone.utc),
            }

            if not self._capital_repository:
                raise ServiceError("Capital repository not available")
            await self._capital_repository.update(allocation_data)

            # Create audit log
            if self._audit_repository:
                await self._create_audit_log("capital_released", allocation_data, authorized_by)

            self._releases_count += 1
            self._logger.info(
                f"Capital released: {release_amount} from {strategy_id} on {exchange}"
            )

            return True

        except Exception as e:
            self._logger.error(f"Capital release failed: {e}")
            raise AllocationError(f"Failed to release capital: {e}") from e

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
            existing = await self._get_existing_allocation(strategy_id, exchange)
            if not existing:
                raise AllocationError(f"No allocation found for {strategy_id} on {exchange}")

            if utilized_amount > existing.allocated_amount:
                raise ValidationError("Utilized amount cannot exceed allocated amount")

            allocation_data = {
                "allocation_id": existing.allocation_id,
                "strategy_id": strategy_id,
                "exchange": exchange,
                "allocated_amount": existing.allocated_amount,
                "utilized_amount": utilized_amount,
                "available_amount": existing.allocated_amount - utilized_amount,
                "allocation_percentage": existing.allocation_percentage,
                "target_allocation_pct": existing.target_allocation_pct,
                "min_allocation": existing.min_allocation,
                "max_allocation": existing.max_allocation,
                "last_rebalance": existing.last_rebalance,
                "updated_at": datetime.now(timezone.utc),
            }

            if not self._capital_repository:
                raise ServiceError("Capital repository not available")
            await self._capital_repository.update(allocation_data)

            # Create audit log
            if self._audit_repository:
                await self._create_audit_log("utilization_updated", allocation_data, authorized_by)

            return True

        except Exception as e:
            self._logger.error(f"Utilization update failed: {e}")
            raise AllocationError(f"Failed to update utilization: {e}") from e

    async def get_capital_metrics(self) -> CapitalMetrics:
        """
        Get current capital metrics.

        Returns:
            CapitalMetrics object
        """
        try:
            allocations = (
                await self._capital_repository.get_all() if self._capital_repository else []
            )

            total_allocated = sum(alloc.allocated_amount for alloc in allocations)
            available_capital = self.total_capital - total_allocated

            return CapitalMetrics(
                total_capital=self.total_capital,
                allocated_amount=total_allocated,
                available_amount=available_capital,
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
                strategies_active=len(set(alloc.strategy_id for alloc in allocations)),
                positions_open=len([alloc for alloc in allocations if alloc.utilized_amount > 0]),
                leverage_used=Decimal("0"),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            self._logger.error(f"Failed to get capital metrics: {e}")
            raise ServiceError(f"Failed to get capital metrics: {e}") from e

    async def get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]:
        """Get all allocations for a strategy."""
        if not self._capital_repository:
            return []
        return await self._capital_repository.get_by_strategy(strategy_id)

    async def get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]:
        """Get all allocations."""
        if not self._capital_repository:
            return []
        return await self._capital_repository.get_all(limit)

    # Private helper methods
    async def _get_available_capital(self) -> Decimal:
        """Get available capital for allocation."""
        if not self._capital_repository:
            return self.total_capital * (Decimal("1") - self.emergency_reserve_pct)

        allocations = await self._capital_repository.get_all()
        total_allocated = sum(alloc.allocated_amount for alloc in allocations)
        emergency_reserve = self.total_capital * self.emergency_reserve_pct
        return self.total_capital - total_allocated - emergency_reserve

    async def _get_existing_allocation(
        self, strategy_id: str, exchange: str
    ) -> CapitalAllocation | None:
        """Get existing allocation for strategy and exchange."""
        if not self._capital_repository:
            return None
        return await self._capital_repository.get_by_strategy_exchange(strategy_id, exchange)

    async def _create_audit_log(
        self, operation: str, data: dict[str, Any], authorized_by: str | None = None
    ) -> None:
        """Create audit log entry."""
        if not self._audit_repository:
            return

        audit_data = {
            "audit_id": str(uuid.uuid4()),
            "operation": operation,
            "data": data,
            "authorized_by": authorized_by or "system",
            "timestamp": datetime.now(timezone.utc),
        }

        await self._audit_repository.create(audit_data)
