"""Market data repository implementation."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.market_data import MarketDataRecord
from src.database.repository.base import DatabaseRepository
from src.database.repository.utils import RepositoryUtils


class MarketDataRepository(DatabaseRepository):
    """Repository for MarketDataRecord entities."""

    def __init__(self, session: AsyncSession):
        """Initialize market data repository."""

        super().__init__(
            session=session,
            model=MarketDataRecord,
            entity_type=MarketDataRecord,
            key_type=str,
            name="MarketDataRepository",
        )

    async def get_by_symbol(self, symbol: str) -> list[MarketDataRecord]:
        """Get market data by symbol."""
        return await RepositoryUtils.get_entities_by_field(self, "symbol", symbol, "-data_timestamp")

    async def get_by_exchange(self, exchange: str) -> list[MarketDataRecord]:
        """Get market data by exchange."""
        return await RepositoryUtils.get_entities_by_field(self, "exchange", exchange, "-data_timestamp")

    async def get_by_symbol_and_exchange(self, symbol: str, exchange: str) -> list[MarketDataRecord]:
        """Get market data by symbol and exchange."""
        return await self.get_all(filters={"symbol": symbol, "exchange": exchange}, order_by="-data_timestamp")

    async def get_latest_price(self, symbol: str, exchange: str) -> MarketDataRecord | None:
        """Get latest price data."""
        records = await self.get_all(
            filters={"symbol": symbol, "exchange": exchange}, order_by="-data_timestamp", limit=1
        )
        return records[0] if records else None

    async def get_ohlc_data(
        self, symbol: str, exchange: str, start_time: datetime, end_time: datetime
    ) -> list[MarketDataRecord]:
        """Get OHLC data for time range."""
        return await self.get_all(
            filters={
                "symbol": symbol,
                "exchange": exchange,
                "data_timestamp": {"gte": start_time, "lte": end_time},
            },
            order_by="data_timestamp",
        )

    async def get_recent_data(self, symbol: str, exchange: str, hours: int = 24) -> list[MarketDataRecord]:
        """Get recent market data."""
        return await self._execute_recent_query(
            timestamp_field="data_timestamp",
            hours=hours,
            additional_filters={"symbol": symbol, "exchange": exchange},
        )

    async def get_by_data_source(self, data_source: str) -> list[MarketDataRecord]:
        """Get data by source."""
        return await RepositoryUtils.get_entities_by_field(self, "source", data_source, "-data_timestamp")

    async def get_poor_quality_data(self, threshold: Decimal = Decimal("0.8")) -> list[MarketDataRecord]:
        """Get data with poor quality scores - not applicable for this model."""
        # MarketDataRecord doesn't have quality_score field
        return []

    async def get_invalid_data(self) -> list[MarketDataRecord]:
        """Get invalid data records - not applicable for this model."""
        # MarketDataRecord doesn't have validation_status field
        return []

    async def cleanup_old_data(self, days: int = 90) -> int:
        """Clean up old market data."""
        return await RepositoryUtils.cleanup_old_entities(self.session, self.model, days, "data_timestamp")

    async def get_volume_leaders(self, exchange: str | None = None, limit: int = 10) -> list[MarketDataRecord]:
        """Get symbols with highest volume."""
        filters = {}
        if exchange:
            filters["exchange"] = exchange

        # This would ideally use a more complex query to get latest volume leaders
        # For now, return recent high-volume records
        return await self.get_all(filters=filters, order_by="-volume", limit=limit)

    async def get_price_changes(
        self, symbol: str, exchange: str, hours: int = 24
    ) -> tuple[Decimal | None, Decimal | None]:
        """Get price change and percentage change."""
        records = await self._execute_recent_query(
            timestamp_field="data_timestamp",  # Fixed: was "timestamp"
            hours=hours,
            additional_filters={"symbol": symbol, "exchange": exchange},
            order_by="data_timestamp",  # Override default desc ordering
        )

        if len(records) < 2:
            return None, None

        first_price = records[0].price or records[0].open_price
        last_price = records[-1].price or records[-1].close_price

        if not first_price or not last_price:
            return None, None

        price_change = last_price - first_price
        percentage_change = (price_change / first_price) * Decimal("100")

        return price_change, percentage_change

    async def _execute_recent_query(
        self,
        timestamp_field: str,
        hours: int,
        additional_filters: dict[str, Any] | None = None,
        order_by: str | None = None,
    ) -> list[MarketDataRecord]:
        """Execute query for recent entities within time range."""
        from datetime import timedelta

        from sqlalchemy import asc, desc, select

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        stmt = select(self.model).where(getattr(self.model, timestamp_field) >= cutoff_time)

        if additional_filters:
            for key, value in additional_filters.items():
                if hasattr(self.model, key):
                    column = getattr(self.model, key)
                    if isinstance(value, list):
                        stmt = stmt.where(column.in_(value))
                    else:
                        stmt = stmt.where(column == value)

        # Apply ordering
        if order_by:
            if order_by.startswith("-"):
                # Descending order
                field_name = order_by[1:]
                if hasattr(self.model, field_name):
                    stmt = stmt.order_by(desc(getattr(self.model, field_name)))
            else:
                # Ascending order
                if hasattr(self.model, order_by):
                    stmt = stmt.order_by(asc(getattr(self.model, order_by)))

        result = await self.session.execute(stmt)
        return list(result.scalars().all())
