"""Capital management repositories implementation."""

from decimal import Decimal

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import DatabaseError
from src.database.models.audit import CapitalAuditLog
from src.database.models.capital import (
    CapitalAllocationDB,
    CurrencyExposureDB,
    ExchangeAllocationDB,
    FundFlowDB,
)
from src.database.repository.base import DatabaseRepository
from src.database.repository.utils import RepositoryUtils


class CapitalAllocationRepository(DatabaseRepository):
    """Repository for CapitalAllocationDB entities."""

    def __init__(self, session: AsyncSession):
        """Initialize capital allocation repository."""
        super().__init__(
            session=session,
            model=CapitalAllocationDB,
            entity_type=CapitalAllocationDB,
            key_type=str,
            name="CapitalAllocationRepository",
        )

    async def get_by_strategy(self, strategy_id: str) -> list[CapitalAllocationDB]:
        """Get allocations by strategy."""
        return await RepositoryUtils.get_entities_by_field(self, "strategy_id", strategy_id)

    async def get_by_exchange(self, exchange: str) -> list[CapitalAllocationDB]:
        """Get allocations by exchange."""
        return await RepositoryUtils.get_entities_by_field(self, "exchange", exchange)

    async def find_by_strategy_exchange(
        self, strategy_id: str, exchange: str
    ) -> CapitalAllocationDB | None:
        """Find allocation by strategy and exchange using proper query."""
        try:
            filters = {"strategy_id": strategy_id, "exchange": exchange}
            result = await RepositoryUtils.get_entities_by_multiple_fields(self, filters)
            return result[0] if result else None
        except (IntegrityError, OperationalError) as e:
            raise DatabaseError(f"Failed to find allocation: {e}")

    async def get_total_allocated_by_strategy(self, strategy_id: str) -> Decimal:
        """Get total allocated amount for a strategy."""
        allocations = await self.get_by_strategy(strategy_id)
        return sum(Decimal(str(alloc.allocated_amount)) for alloc in allocations)

    async def get_available_capital_by_exchange(self, exchange: str) -> Decimal:
        """Get total available capital for an exchange."""
        allocations = await self.get_by_exchange(exchange)
        return sum(Decimal(str(alloc.available_amount)) for alloc in allocations)


class FundFlowRepository(DatabaseRepository):
    """Repository for FundFlowDB entities."""

    def __init__(self, session: AsyncSession):
        """Initialize fund flow repository."""
        super().__init__(
            session=session,
            model=FundFlowDB,
            entity_type=FundFlowDB,
            key_type=str,
            name="FundFlowRepository",
        )

    async def get_by_from_strategy(self, strategy_id: str) -> list[FundFlowDB]:
        """Get flows from a strategy."""
        return await RepositoryUtils.get_entities_by_field(
            self, "from_strategy", strategy_id, "-timestamp"
        )

    async def get_by_to_strategy(self, strategy_id: str) -> list[FundFlowDB]:
        """Get flows to a strategy."""
        return await RepositoryUtils.get_entities_by_field(
            self, "to_strategy", strategy_id, "-timestamp"
        )

    async def get_by_exchange_flow(self, from_exchange: str, to_exchange: str) -> list[FundFlowDB]:
        """Get flows between exchanges."""
        filters = {"from_exchange": from_exchange, "to_exchange": to_exchange}
        return await RepositoryUtils.get_entities_by_multiple_fields(self, filters, "-timestamp")

    async def get_by_reason(self, reason: str) -> list[FundFlowDB]:
        """Get flows by reason."""
        return await RepositoryUtils.get_entities_by_field(self, "reason", reason, "-timestamp")

    async def get_by_currency(self, currency: str) -> list[FundFlowDB]:
        """Get flows by currency."""
        return await RepositoryUtils.get_entities_by_field(self, "currency", currency, "-timestamp")


class CurrencyExposureRepository(DatabaseRepository):
    """Repository for CurrencyExposureDB entities."""

    def __init__(self, session: AsyncSession):
        """Initialize currency exposure repository."""
        super().__init__(
            session=session,
            model=CurrencyExposureDB,
            entity_type=CurrencyExposureDB,
            key_type=str,
            name="CurrencyExposureRepository",
        )

    async def get_by_currency(self, currency: str) -> CurrencyExposureDB | None:
        """Get exposure by currency."""
        return await self.get_by(currency=currency)

    async def get_hedging_required(self) -> list[CurrencyExposureDB]:
        """Get exposures that require hedging."""
        return await self.get_all(filters={"hedging_required": True})

    async def get_total_exposure(self) -> Decimal:
        """Get total currency exposure."""
        exposures = await self.get_all()
        return sum(exp.total_exposure for exp in exposures)


class ExchangeAllocationRepository(DatabaseRepository):
    """Repository for ExchangeAllocationDB entities."""

    def __init__(self, session: AsyncSession):
        """Initialize exchange allocation repository."""
        super().__init__(
            session=session,
            model=ExchangeAllocationDB,
            entity_type=ExchangeAllocationDB,
            key_type=str,
            name="ExchangeAllocationRepository",
        )

    async def get_by_exchange(self, exchange: str) -> ExchangeAllocationDB | None:
        """Get allocation by exchange."""
        return await self.get_by(exchange=exchange)

    async def get_total_allocated(self) -> Decimal:
        """Get total allocated amount across all exchanges."""
        allocations = await self.get_all()
        return sum(alloc.total_allocation for alloc in allocations)

    async def get_total_available(self) -> Decimal:
        """Get total available amount across all exchanges."""
        allocations = await self.get_all()
        # Note: ExchangeAllocationDB doesn't have available_amount field, using unutilized
        return sum(alloc.total_allocation - alloc.utilized_allocation for alloc in allocations)

    async def get_underutilized_exchanges(
        self, threshold: Decimal = Decimal("0.5")
    ) -> list[ExchangeAllocationDB]:
        """Get exchanges with low utilization."""
        allocations = await self.get_all()
        return [
            alloc
            for alloc in allocations
            if alloc.allocated_amount > 0
            and (alloc.utilized_amount / alloc.allocated_amount) < threshold
        ]


class CapitalAuditLogRepository(DatabaseRepository):
    """Repository for capital audit log entities."""

    def __init__(self, session: AsyncSession):
        """Initialize capital audit repository."""
        super().__init__(
            session=session,
            model=CapitalAuditLog,
            entity_type=CapitalAuditLog,
            key_type=str,
            name="CapitalAuditLogRepository",
        )

    async def get_by_operation_id(self, operation_id: str) -> CapitalAuditLog | None:
        """Get audit log by operation ID."""
        return await self.get_by(operation_id=operation_id)

    async def get_by_strategy(self, strategy_id: str) -> list[CapitalAuditLog]:
        """Get audit logs by strategy."""
        return await self.get_all(filters={"strategy_id": strategy_id}, order_by="-created_at")

    async def get_by_exchange(self, exchange: str) -> list[CapitalAuditLog]:
        """Get audit logs by exchange."""
        return await self.get_all(filters={"exchange": exchange}, order_by="-created_at")

    async def get_failed_operations(self, limit: int = 100) -> list[CapitalAuditLog]:
        """Get failed audit operations."""
        return await self.get_all(filters={"success": False}, order_by="-created_at", limit=limit)
