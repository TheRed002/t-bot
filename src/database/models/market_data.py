"""Market data database models."""

import uuid
from decimal import Decimal

from sqlalchemy import DECIMAL, CheckConstraint, Column, DateTime, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import Base, TimestampMixin


class MarketDataRecord(Base, TimestampMixin):
    """Market data record model."""

    __tablename__ = "market_data_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(50), nullable=False)

    # Price data
    open_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8))
    high_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8))
    low_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8))
    close_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8))
    volume: Mapped[Decimal] = mapped_column(DECIMAL(20, 8))

    # Timestamp for the market data
    data_timestamp = Column(DateTime(timezone=True), nullable=False)

    # Data type (1m, 5m, 1h, 1d, etc.)
    interval = Column(String(10), nullable=False)

    # Data source
    source = Column(String(50), nullable=False, default="exchange")

    # Relationships
    feature_records = relationship("FeatureRecord", back_populates="market_data")
    data_quality_records = relationship("DataQualityRecord", back_populates="market_data", cascade="all, delete-orphan")

    # Indexes - High-Performance Market Data Access
    __table_args__ = (
        Index("idx_market_data_symbol_exchange", "symbol", "exchange"),
        Index("idx_market_data_timestamp", "data_timestamp"),
        Index("idx_market_data_interval", "interval"),
        # Critical composite index for real-time data access
        Index("idx_market_data_composite", "symbol", "exchange", "interval", "data_timestamp"),
        # High-frequency trading optimized indexes
        Index("idx_market_data_symbol_timestamp", "symbol", "data_timestamp"),  # Fast symbol lookups
        Index("idx_market_data_exchange_timestamp", "exchange", "data_timestamp"),  # Exchange-specific data
        Index(
            "idx_market_data_recent", "data_timestamp", "symbol", "exchange"
        ),  # BRIN index for time-series optimization
        # Partial index for recent data (last 7 days) - most frequently accessed
        Index("idx_market_data_hot", "symbol", "exchange", "data_timestamp"),
        # Unique constraint to prevent duplicate market data records
        UniqueConstraint("symbol", "exchange", "interval", "data_timestamp", name="uq_market_data_record"),
        CheckConstraint("open_price > 0", name="check_open_price_positive"),
        CheckConstraint("high_price > 0", name="check_high_price_positive"),
        CheckConstraint("low_price > 0", name="check_low_price_positive"),
        CheckConstraint("close_price > 0", name="check_close_price_positive"),
        CheckConstraint("volume >= 0", name="check_volume_non_negative"),
        CheckConstraint("high_price >= low_price", name="check_high_low_relationship"),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')",
            name="check_market_data_supported_exchange",
        ),
    )

    def __repr__(self):
        timestamp_str = str(self.data_timestamp)[:19] if self.data_timestamp else "None"
        return f"<MarketDataRecord {self.symbol} {self.exchange} {self.interval} @ {timestamp_str}>"

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
