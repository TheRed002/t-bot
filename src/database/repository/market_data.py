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
        return await RepositoryUtils.get_entities_by_field(
            self, "symbol", symbol, "-data_timestamp"
        )

    async def get_by_exchange(self, exchange: str) -> list[MarketDataRecord]:
        """Get market data by exchange."""
        return await RepositoryUtils.get_entities_by_field(
            self, "exchange", exchange, "-data_timestamp"
        )

    async def get_by_symbol_and_exchange(
        self, symbol: str, exchange: str
    ) -> list[MarketDataRecord]:
        """Get market data by symbol and exchange."""
        return await self.get_all(
            filters={"symbol": symbol, "exchange": exchange}, order_by="-data_timestamp"
        )

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

    async def get_recent_data(
        self, symbol: str, exchange: str, hours: int = 24
    ) -> list[MarketDataRecord]:
        """Get recent market data."""
        return await RepositoryUtils.execute_time_based_query(
            self.session,
            self.model,
            timestamp_field="data_timestamp",
            hours=hours,
            additional_filters={"symbol": symbol, "exchange": exchange},
        )

    async def get_by_data_source(self, data_source: str) -> list[MarketDataRecord]:
        """Get data by source."""
        return await RepositoryUtils.get_entities_by_field(
            self, "source", data_source, "-data_timestamp"
        )

    async def get_poor_quality_data(
        self, threshold: Decimal = Decimal("0.8")
    ) -> list[MarketDataRecord]:
        """Get data with poor quality scores - not applicable for this model."""
        # MarketDataRecord doesn't have quality_score field
        return []

    async def get_invalid_data(self) -> list[MarketDataRecord]:
        """Get invalid data records - not applicable for this model."""
        # MarketDataRecord doesn't have validation_status field
        return []

    async def cleanup_old_data(self, days: int = 90) -> int:
        """Clean up old market data."""
        return await RepositoryUtils.cleanup_old_entities(
            self.session, self.model, days, "data_timestamp"
        )

    async def save_ticker(self, exchange: str, symbol: str, data: dict[str, Any]) -> None:
        """Save ticker data from exchange."""
        record = MarketDataRecord(
            exchange=exchange,
            symbol=symbol,
            interval="ticker",
            open_price=Decimal(str(data.get("open_price", 0))),
            high_price=Decimal(str(data.get("high_24h", 0))),
            low_price=Decimal(str(data.get("low_24h", 0))),
            close_price=Decimal(str(data.get("last_price", 0))),
            volume=Decimal(str(data.get("volume_24h", 0))),
            data_timestamp=data.get("timestamp", datetime.now(timezone.utc)),
            source="exchange_api",
        )
        await self.create(record)

    async def get_volume_leaders(
        self, exchange: str | None = None, limit: int = 10
    ) -> list[MarketDataRecord]:
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
        records = await RepositoryUtils.execute_time_based_query(
            self.session,
            self.model,
            timestamp_field="data_timestamp",
            hours=hours,
            additional_filters={"symbol": symbol, "exchange": exchange},
            order_by="data_timestamp",
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
