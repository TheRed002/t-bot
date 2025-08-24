"""Market data repository implementation."""

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from src.database.models.market_data import MarketDataRecord
from src.database.repository.base import BaseRepository


class MarketDataRepository(BaseRepository[MarketDataRecord]):
    """Repository for MarketDataRecord entities."""

    def __init__(self, session: Session):
        """Initialize market data repository."""
        super().__init__(session, MarketDataRecord)

    async def get_by_symbol(self, symbol: str) -> list[MarketDataRecord]:
        """Get market data by symbol."""
        return await self.get_all(filters={"symbol": symbol}, order_by="-timestamp")

    async def get_by_exchange(self, exchange: str) -> list[MarketDataRecord]:
        """Get market data by exchange."""
        return await self.get_all(filters={"exchange": exchange}, order_by="-timestamp")

    async def get_by_symbol_and_exchange(
        self, symbol: str, exchange: str
    ) -> list[MarketDataRecord]:
        """Get market data by symbol and exchange."""
        return await self.get_all(
            filters={"symbol": symbol, "exchange": exchange}, order_by="-timestamp"
        )

    async def get_latest_price(self, symbol: str, exchange: str) -> MarketDataRecord | None:
        """Get latest price data."""
        records = await self.get_all(
            filters={"symbol": symbol, "exchange": exchange}, order_by="-timestamp", limit=1
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
                "timestamp": {"gte": start_time, "lte": end_time},
            },
            order_by="timestamp",
        )

    async def get_recent_data(
        self, symbol: str, exchange: str, hours: int = 24
    ) -> list[MarketDataRecord]:
        """Get recent market data."""
        since = datetime.utcnow() - timedelta(hours=hours)
        return await self.get_all(
            filters={"symbol": symbol, "exchange": exchange, "timestamp": {"gte": since}},
            order_by="-timestamp",
        )

    async def get_by_data_source(self, data_source: str) -> list[MarketDataRecord]:
        """Get data by source."""
        return await self.get_all(filters={"data_source": data_source}, order_by="-timestamp")

    async def get_poor_quality_data(self, threshold: float = 0.8) -> list[MarketDataRecord]:
        """Get data with poor quality scores."""
        records = await self.get_all()
        return [
            record
            for record in records
            if record.quality_score is not None and record.quality_score < threshold
        ]

    async def get_invalid_data(self) -> list[MarketDataRecord]:
        """Get invalid data records."""
        return await self.get_all(filters={"validation_status": "invalid"}, order_by="-timestamp")

    async def cleanup_old_data(self, days: int = 90) -> int:
        """Clean up old market data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        old_records = await self.get_all(filters={"timestamp": {"lt": cutoff_date}})

        count = 0
        for record in old_records:
            await self.delete(record.id)
            count += 1
        return count

    async def get_volume_leaders(
        self, exchange: str = None, limit: int = 10
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
    ) -> tuple[float | None, float | None]:
        """Get price change and percentage change."""
        since = datetime.utcnow() - timedelta(hours=hours)
        records = await self.get_all(
            filters={"symbol": symbol, "exchange": exchange, "timestamp": {"gte": since}},
            order_by="timestamp",
        )

        if len(records) < 2:
            return None, None

        first_price = records[0].price or records[0].open_price
        last_price = records[-1].price or records[-1].close_price

        if not first_price or not last_price:
            return None, None

        price_change = float(last_price - first_price)
        percentage_change = (price_change / float(first_price)) * 100

        return price_change, percentage_change
