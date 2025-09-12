"""Exchange-specific database models."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class ExchangeConfiguration(Base, TimestampMixin):
    """Exchange configuration model."""

    __tablename__ = "exchange_configurations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange_name = Column(String(50), nullable=False, unique=True)
    display_name = Column(String(100), nullable=False)

    # Connection configuration
    base_url = Column(String(255), nullable=False)
    websocket_url = Column(String(255))
    api_version = Column(String(20))

    # Status and capabilities
    status = Column(String(20), nullable=False, default="offline")
    is_enabled = Column(Boolean, default=True)
    capabilities = Column(JSONB, default=list)

    # Rate limiting
    requests_per_minute: Mapped[int] = mapped_column(Integer, default=1200)
    orders_per_second: Mapped[int] = mapped_column(Integer, default=10)
    websocket_connections: Mapped[int] = mapped_column(Integer, default=10)
    weight_per_request: Mapped[int] = mapped_column(Integer, default=1)

    # Fee structure
    default_maker_fee: Mapped[Decimal] = mapped_column(DECIMAL(6, 4), default=Decimal("0.001"))
    default_taker_fee: Mapped[Decimal] = mapped_column(DECIMAL(6, 4), default=Decimal("0.001"))

    # Configuration metadata
    configuration = Column(JSONB, default=dict)
    metadata_json = Column(JSONB, default=dict)

    # Relationships
    trading_pairs = relationship(
        "ExchangeTradingPair",
        back_populates="exchange_config",
        cascade="all, delete-orphan"
    )
    connection_statuses = relationship(
        "ExchangeConnectionStatus",
        back_populates="exchange_config",
        cascade="all, delete-orphan"
    )

    # Indexes and constraints
    __table_args__ = (
        Index("idx_exchange_config_name", "exchange_name"),
        Index("idx_exchange_config_status", "status"),
        Index("idx_exchange_config_enabled", "is_enabled"),
        Index("idx_exchange_config_status_enabled", "status", "is_enabled"),
        CheckConstraint(
            "status IN ('online', 'offline', 'maintenance', 'degraded')",
            name="check_exchange_status"
        ),
        CheckConstraint(
            "exchange_name IN ('binance', 'coinbase', 'okx', 'mock')",
            name="check_supported_exchange_name"
        ),
        CheckConstraint("requests_per_minute > 0", name="check_requests_per_minute_positive"),
        CheckConstraint("orders_per_second > 0", name="check_orders_per_second_positive"),
        CheckConstraint("websocket_connections > 0", name="check_websocket_connections_positive"),
        CheckConstraint("weight_per_request > 0", name="check_weight_per_request_positive"),
        CheckConstraint(
            "default_maker_fee >= 0 AND default_maker_fee <= 1",
            name="check_maker_fee_range"
        ),
        CheckConstraint(
            "default_taker_fee >= 0 AND default_taker_fee <= 1",
            name="check_taker_fee_range"
        ),
        CheckConstraint("LENGTH(exchange_name) >= 1", name="check_exchange_name_not_empty"),
        CheckConstraint("LENGTH(display_name) >= 1", name="check_display_name_not_empty"),
        CheckConstraint("LENGTH(base_url) >= 1", name="check_base_url_not_empty"),
    )

    def __repr__(self):
        return f"<ExchangeConfiguration {self.exchange_name}: {self.status}>"


class ExchangeTradingPair(Base, TimestampMixin):
    """Exchange-specific trading pair information."""

    __tablename__ = "exchange_trading_pairs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange_config_id = Column(
        UUID(as_uuid=True),
        ForeignKey("exchange_configurations.id", ondelete="CASCADE"),
        nullable=False
    )

    # Symbol information
    symbol = Column(String(50), nullable=False)
    base_asset = Column(String(20), nullable=False)
    quote_asset = Column(String(20), nullable=False)

    # Trading limits
    min_quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    max_quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    step_size: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    min_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    max_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    tick_size: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    min_notional: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))

    # Status and metadata
    is_active = Column(Boolean, default=True)
    is_trading = Column(Boolean, default=True)
    status = Column(String(20), default="active")

    # Fee structure (pair-specific overrides)
    maker_fee: Mapped[Decimal | None] = mapped_column(DECIMAL(6, 4))
    taker_fee: Mapped[Decimal | None] = mapped_column(DECIMAL(6, 4))

    # Additional metadata
    metadata_json = Column(JSONB, default=dict)

    # Relationships
    exchange_config = relationship("ExchangeConfiguration", back_populates="trading_pairs")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_trading_pair_symbol", "symbol"),
        Index("idx_trading_pair_exchange", "exchange_config_id"),
        Index("idx_trading_pair_base_asset", "base_asset"),
        Index("idx_trading_pair_quote_asset", "quote_asset"),
        Index("idx_trading_pair_active", "is_active"),
        Index("idx_trading_pair_trading", "is_trading"),
        Index("idx_trading_pair_status", "status"),
        Index("idx_trading_pair_composite", "exchange_config_id", "symbol", "is_active"),
        UniqueConstraint(
            "exchange_config_id", "symbol",
            name="uq_exchange_trading_pair"
        ),
        CheckConstraint("min_quantity > 0", name="check_min_quantity_positive"),
        CheckConstraint("max_quantity > 0", name="check_max_quantity_positive"),
        CheckConstraint("max_quantity >= min_quantity", name="check_quantity_range"),
        CheckConstraint("step_size > 0", name="check_step_size_positive"),
        CheckConstraint("min_price > 0", name="check_min_price_positive"),
        CheckConstraint("max_price > 0", name="check_max_price_positive"),
        CheckConstraint("max_price >= min_price", name="check_price_range"),
        CheckConstraint("tick_size > 0", name="check_tick_size_positive"),
        CheckConstraint(
            "min_notional IS NULL OR min_notional > 0",
            name="check_min_notional_positive"
        ),
        CheckConstraint(
            "maker_fee IS NULL OR (maker_fee >= 0 AND maker_fee <= 1)",
            name="check_pair_maker_fee_range"
        ),
        CheckConstraint(
            "taker_fee IS NULL OR (taker_fee >= 0 AND taker_fee <= 1)",
            name="check_pair_taker_fee_range"
        ),
        CheckConstraint(
            "status IN ('active', 'inactive', 'delisted')",
            name="check_trading_pair_status"
        ),
        CheckConstraint("LENGTH(symbol) >= 1", name="check_symbol_not_empty"),
        CheckConstraint("LENGTH(base_asset) >= 1", name="check_base_asset_not_empty"),
        CheckConstraint("LENGTH(quote_asset) >= 1", name="check_quote_asset_not_empty"),
    )

    def __repr__(self):
        return f"<ExchangeTradingPair {self.symbol} on {self.exchange_config.exchange_name if self.exchange_config else 'Unknown'}>"

    def round_price(self, price: Decimal) -> Decimal:
        """Round price to valid tick size."""
        return (price // self.tick_size) * self.tick_size

    def round_quantity(self, quantity: Decimal) -> Decimal:
        """Round quantity to valid step size."""
        return (quantity // self.step_size) * self.step_size

    def validate_order(self, price: Decimal, quantity: Decimal) -> bool:
        """Validate if order parameters are within exchange limits."""
        if price < self.min_price or price > self.max_price:
            return False
        if quantity < self.min_quantity or quantity > self.max_quantity:
            return False
        if self.min_notional and (price * quantity) < self.min_notional:
            return False
        return True


class ExchangeConnectionStatus(Base, TimestampMixin):
    """Exchange connection status tracking."""

    __tablename__ = "exchange_connection_statuses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange_config_id = Column(
        UUID(as_uuid=True),
        ForeignKey("exchange_configurations.id", ondelete="CASCADE"),
        nullable=False
    )

    # Connection details
    connection_type = Column(String(20), nullable=False)  # 'rest', 'websocket'
    status = Column(String(20), nullable=False, default="disconnected")

    # Performance metrics
    latency_ms: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 2))
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), default=Decimal("0"))

    # Timestamps
    last_heartbeat = Column(DateTime(timezone=True))
    last_error = Column(DateTime(timezone=True))
    last_success = Column(DateTime(timezone=True))

    # Error details
    last_error_message = Column(Text)
    last_error_code = Column(String(50))

    # Additional metadata
    metadata_json = Column(JSONB, default=dict)

    # Relationships
    exchange_config = relationship("ExchangeConfiguration", back_populates="connection_statuses")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_connection_status_exchange", "exchange_config_id"),
        Index("idx_connection_status_type", "connection_type"),
        Index("idx_connection_status_status", "status"),
        Index("idx_connection_status_composite", "exchange_config_id", "connection_type", "status"),
        Index("idx_connection_status_heartbeat", "last_heartbeat"),
        Index("idx_connection_status_success_rate", "success_rate"),
        Index("idx_connection_status_latency", "latency_ms"),
        UniqueConstraint(
            "exchange_config_id", "connection_type",
            name="uq_exchange_connection_type"
        ),
        CheckConstraint(
            "connection_type IN ('rest', 'websocket')",
            name="check_connection_type"
        ),
        CheckConstraint(
            "status IN ('connected', 'disconnected', 'connecting', 'error')",
            name="check_connection_status"
        ),
        CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="check_latency_non_negative"
        ),
        CheckConstraint("error_count >= 0", name="check_error_count_non_negative"),
        CheckConstraint("success_count >= 0", name="check_success_count_non_negative"),
        CheckConstraint(
            "success_rate >= 0 AND success_rate <= 1",
            name="check_success_rate_range"
        ),
    )

    def __repr__(self):
        return f"<ExchangeConnectionStatus {self.exchange_config.exchange_name if self.exchange_config else 'Unknown'} {self.connection_type}: {self.status}>"

    def calculate_success_rate(self) -> Decimal:
        """Calculate and update success rate."""
        total = self.success_count + self.error_count
        if total == 0:
            self.success_rate = Decimal("0")
        else:
            self.success_rate = Decimal(str(self.success_count / total)).quantize(Decimal("0.0001"))
        return self.success_rate


class ExchangeRateLimit(Base, TimestampMixin):
    """Exchange rate limit tracking and enforcement."""

    __tablename__ = "exchange_rate_limits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange_config_id = Column(
        UUID(as_uuid=True),
        ForeignKey("exchange_configurations.id", ondelete="CASCADE"),
        nullable=False
    )

    # Rate limit configuration
    limit_type = Column(String(50), nullable=False)  # 'request', 'order', 'weight'
    limit_name = Column(String(100), nullable=False)
    max_requests: Mapped[int] = mapped_column(Integer, nullable=False)
    time_window_seconds: Mapped[int] = mapped_column(Integer, nullable=False)

    # Current usage
    current_usage: Mapped[int] = mapped_column(Integer, default=0)
    window_start = Column(DateTime(timezone=True), nullable=False, default=datetime.now)
    window_end = Column(DateTime(timezone=True), nullable=False)

    # Status
    is_active = Column(Boolean, default=True)
    last_reset = Column(DateTime(timezone=True))

    # Violation tracking
    violation_count: Mapped[int] = mapped_column(Integer, default=0)
    last_violation = Column(DateTime(timezone=True))

    # Relationships
    exchange_config = relationship("ExchangeConfiguration")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_rate_limit_exchange", "exchange_config_id"),
        Index("idx_rate_limit_type", "limit_type"),
        Index("idx_rate_limit_active", "is_active"),
        Index("idx_rate_limit_window", "window_start", "window_end"),
        Index("idx_rate_limit_usage", "current_usage"),
        UniqueConstraint(
            "exchange_config_id", "limit_type", "limit_name",
            name="uq_exchange_rate_limit"
        ),
        CheckConstraint("max_requests > 0", name="check_max_requests_positive"),
        CheckConstraint("time_window_seconds > 0", name="check_time_window_positive"),
        CheckConstraint("current_usage >= 0", name="check_current_usage_non_negative"),
        CheckConstraint("current_usage <= max_requests", name="check_usage_within_limit"),
        CheckConstraint("window_start < window_end", name="check_window_order"),
        CheckConstraint("violation_count >= 0", name="check_violation_count_non_negative"),
        CheckConstraint(
            "limit_type IN ('request', 'order', 'weight', 'websocket')",
            name="check_limit_type"
        ),
        CheckConstraint("LENGTH(limit_name) >= 1", name="check_limit_name_not_empty"),
    )

    def __repr__(self):
        return f"<ExchangeRateLimit {self.limit_type}: {self.current_usage}/{self.max_requests}>"

    @property
    def is_exceeded(self) -> bool:
        """Check if rate limit is currently exceeded."""
        return self.current_usage >= self.max_requests

    @property
    def remaining_requests(self) -> int:
        """Get remaining requests in current window."""
        return max(0, self.max_requests - self.current_usage)

    @property
    def usage_percentage(self) -> Decimal:
        """Get current usage as percentage."""
        if self.max_requests == 0:
            return Decimal("0")
        return Decimal(str(self.current_usage / self.max_requests * 100)).quantize(Decimal("0.01"))
