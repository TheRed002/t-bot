"""Market data database models."""

import uuid
from decimal import Decimal

from sqlalchemy import DECIMAL, Column, DateTime, Index, String
from sqlalchemy.dialects.postgresql import UUID

from src.database.models.base import Base, TimestampMixin


class MarketDataRecord(Base, TimestampMixin):
    """Market data record model."""

    __tablename__ = "market_data_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(50), nullable=False)

    # Price data
    open_price = Column(DECIMAL(20, 8))
    high_price = Column(DECIMAL(20, 8))
    low_price = Column(DECIMAL(20, 8))
    close_price = Column(DECIMAL(20, 8))
    volume = Column(DECIMAL(20, 8))

    # Timestamp for the market data
    data_timestamp = Column(DateTime(timezone=True), nullable=False)

    # Data type (1m, 5m, 1h, 1d, etc.)
    interval = Column(String(10), nullable=False)

    # Data source
    source = Column(String(50), nullable=False, default="exchange")

    # Indexes - High-Performance Market Data Access
    __table_args__ = (
        Index("idx_market_data_symbol_exchange", "symbol", "exchange"),
        Index("idx_market_data_timestamp", "data_timestamp"),
        Index("idx_market_data_interval", "interval"),
        # Critical composite index for real-time data access
        Index("idx_market_data_composite", "symbol", "exchange", "interval", "data_timestamp"),
        # High-frequency trading optimized indexes
        Index(
            "idx_market_data_symbol_timestamp", "symbol", "data_timestamp"
        ),  # Fast symbol lookups
        Index(
            "idx_market_data_exchange_timestamp", "exchange", "data_timestamp"
        ),  # Exchange-specific data
        Index(
            "idx_market_data_recent", "data_timestamp", "symbol", "exchange"
        ),  # BRIN index for time-series optimization
        # Partial index for recent data (last 7 days) - most frequently accessed
        Index("idx_market_data_hot", "symbol", "exchange", "data_timestamp"),
    )

    def __repr__(self):
        return f"<MarketDataRecord {self.symbol} {self.exchange} {self.interval} @ {self.data_timestamp}>"

    @property
    def price_change(self) -> Decimal:
        """Calculate price change from open to close."""
        if self.open_price and self.close_price:
            return self.close_price - self.open_price
        return Decimal("0")

    @property
    def price_change_percent(self) -> Decimal:
        """Calculate percentage price change."""
        if self.open_price and self.close_price and self.open_price > 0:
            return (self.close_price - self.open_price) / self.open_price * Decimal("100")
        return Decimal("0")
