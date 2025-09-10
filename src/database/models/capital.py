"""Capital management database models."""

import uuid
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class CapitalAllocationDB(Base, TimestampMixin):
    """Capital allocation tracking model."""

    __tablename__ = "capital_allocations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False
    )
    exchange = Column(String(50), nullable=False)

    # Allocation amounts
    allocated_amount: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    utilized_amount: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    available_amount: Mapped[Decimal] = mapped_column(
        DECIMAL(20, 8), default=0
    )  # Added field expected by service
    reserved_amount: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    # Allocation metadata
    allocation_percentage: Mapped[Decimal] = mapped_column(
        DECIMAL(5, 2), default=0.0
    )  # Added field expected by service
    allocation_type = Column(
        String(50), nullable=False, default="dynamic"
    )  # fixed, percentage, dynamic
    priority = Column(Integer, default=5)  # 1-10 priority scale

    # Performance tracking
    total_return: Mapped[Decimal] = mapped_column(DECIMAL(10, 4), default=0)
    utilization_ratio: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), default=0)

    # Rebalancing
    last_rebalance = Column(
        DateTime(timezone=True), nullable=True
    )  # Added field expected by service

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Relationships
    strategy = relationship("Strategy", back_populates="capital_allocations")
    fund_flows = relationship(
        "FundFlowDB", back_populates="capital_allocation", cascade="all, delete-orphan"
    )

    # Indexes and constraints
    __table_args__ = (
        Index("idx_capital_strategy", "strategy_id"),
        Index("idx_capital_exchange", "exchange"),
        Index("idx_capital_type", "allocation_type"),
        Index("idx_capital_created", "created_at"),
        Index(
            "idx_capital_strategy_exchange", "strategy_id", "exchange"
        ),  # Composite index for lookups
        Index(
            "idx_capital_allocation_performance",
            "allocated_amount",
            "utilized_amount",
            "utilization_ratio",
        ),  # Performance analysis
        Index(
            "idx_capital_priority_type", "priority", "allocation_type"
        ),  # Priority-based allocation
        UniqueConstraint("strategy_id", "exchange", name="uq_strategy_exchange_allocation"),
        CheckConstraint("allocated_amount >= 0", name="check_allocated_amount_non_negative"),
        CheckConstraint("utilized_amount >= 0", name="check_utilized_amount_non_negative"),
        CheckConstraint("available_amount >= 0", name="check_available_amount_non_negative"),
        CheckConstraint("reserved_amount >= 0", name="check_reserved_amount_non_negative"),
        CheckConstraint(
            "allocation_percentage >= 0 AND allocation_percentage <= 100",
            name="check_allocation_percentage_range",
        ),
        CheckConstraint("priority >= 1 AND priority <= 10", name="check_priority_range"),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')",
            name="check_capital_supported_exchange",
        ),
        CheckConstraint(
            "utilized_amount <= allocated_amount",
            name="check_capital_utilization_within_allocation",
        ),
        CheckConstraint(
            "available_amount = allocated_amount - utilized_amount - reserved_amount",
            name="check_capital_accounting_balance",
        ),
    )

    def __repr__(self):
        return (
            f"<CapitalAllocationDB {self.strategy_id} on {self.exchange}: {self.allocated_amount}>"
        )


class FundFlowDB(Base, TimestampMixin):
    """Fund flow tracking model."""

    __tablename__ = "fund_flows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    flow_type = Column(String(50), nullable=False)  # deposit, withdrawal, allocation, rebalance
    from_account = Column(String(100))
    to_account = Column(String(100))

    # Foreign key relationships
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"), nullable=True)
    strategy_id = Column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE"), nullable=True
    )
    capital_allocation_id = Column(
        UUID(as_uuid=True), ForeignKey("capital_allocations.id", ondelete="SET NULL"), nullable=True
    )

    # Flow details
    currency = Column(String(10), nullable=False)
    amount: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    # Status tracking
    status = Column(String(20), default="PENDING")  # PENDING, PROCESSING, COMPLETED, FAILED
    transaction_id = Column(String(100))

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Relationships
    bot = relationship("Bot", back_populates="fund_flows")
    strategy = relationship("Strategy", back_populates="fund_flows")
    capital_allocation = relationship("CapitalAllocationDB", back_populates="fund_flows")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_fund_flow_type", "flow_type"),
        Index("idx_fund_flow_currency", "currency"),
        Index("idx_fund_flow_status", "status"),
        Index("idx_fund_flow_created", "created_at"),
        Index("idx_fund_flow_bot_id", "bot_id"),
        Index("idx_fund_flow_strategy_id", "strategy_id"),
        Index("idx_fund_flow_capital_allocation_id", "capital_allocation_id"),
        Index(
            "idx_fund_flow_accounts", "from_account", "to_account"
        ),  # Account tracking optimization
        Index(
            "idx_fund_flow_transaction", "transaction_id", "status"
        ),  # Transaction status tracking
        Index("idx_fund_flow_bot_status", "bot_id", "status"),  # Bot flow tracking
        Index("idx_fund_flow_strategy_status", "strategy_id", "status"),  # Strategy flow tracking
        UniqueConstraint("transaction_id", name="uq_fund_flow_transaction"),
        CheckConstraint("amount > 0", name="check_fund_flow_amount_positive"),
        CheckConstraint("fee >= 0", name="check_fund_flow_fee_non_negative"),
        CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')",
            name="check_fund_flow_status",
        ),
        CheckConstraint(
            "flow_type IN ('deposit', 'withdrawal', 'allocation', 'rebalance')",
            name="check_flow_type",
        ),
    )

    def __repr__(self):
        return f"<FundFlowDB {self.flow_type}: {self.amount} {self.currency}>"


class CurrencyExposureDB(Base, TimestampMixin):
    """Currency exposure tracking model."""

    __tablename__ = "currency_exposures"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    currency = Column(String(10), nullable=False)

    # Exposure amounts
    total_exposure: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    spot_exposure: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    futures_exposure: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    # Risk metrics
    var_1d: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))  # Value at Risk 1 day
    var_7d: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))  # Value at Risk 7 days
    volatility: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 6))

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Indexes and constraints
    __table_args__ = (
        Index("idx_currency_exposure", "currency"),
        Index("idx_currency_created", "created_at"),
        Index(
            "idx_currency_risk_metrics", "var_1d", "var_7d", "volatility"
        ),  # Risk analysis optimization
        Index("idx_currency_exposure_total", "total_exposure", "currency"),  # Exposure ranking
        Index(
            "idx_currency_exposure_breakdown", "spot_exposure", "futures_exposure"
        ),  # Exposure breakdown analysis
        UniqueConstraint("currency", "created_at", name="uq_currency_exposure_timestamp"),
        CheckConstraint("total_exposure >= 0", name="check_total_exposure_non_negative"),
        CheckConstraint("spot_exposure >= 0", name="check_spot_exposure_non_negative"),
        CheckConstraint("futures_exposure >= 0", name="check_futures_exposure_non_negative"),
        CheckConstraint(
            "spot_exposure + futures_exposure = total_exposure", name="check_exposure_balance"
        ),
    )

    def __repr__(self):
        return f"<CurrencyExposureDB {self.currency}: {self.total_exposure}>"


class ExchangeAllocationDB(Base, TimestampMixin):
    """Exchange allocation tracking model."""

    __tablename__ = "exchange_allocations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)

    # Allocation amounts
    total_allocation: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    utilized_allocation: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)
    reserved_allocation: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)

    # Performance metrics
    efficiency_score: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=0)
    utilization_ratio: Mapped[Decimal] = mapped_column(DECIMAL(5, 4), default=0)

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Indexes and constraints
    __table_args__ = (
        Index("idx_exchange_allocation", "exchange"),
        Index("idx_exchange_created", "created_at"),
        Index(
            "idx_exchange_utilization", "utilization_ratio", "efficiency_score"
        ),  # Performance optimization
        UniqueConstraint("exchange", name="uq_exchange_allocation"),
        CheckConstraint("total_allocation >= 0", name="check_total_allocation_non_negative"),
        CheckConstraint("utilized_allocation >= 0", name="check_utilized_allocation_non_negative"),
        CheckConstraint("reserved_allocation >= 0", name="check_reserved_allocation_non_negative"),
        CheckConstraint(
            "utilized_allocation + reserved_allocation <= total_allocation",
            name="check_allocation_balance",
        ),
        CheckConstraint(
            "efficiency_score >= 0 AND efficiency_score <= 100", name="check_efficiency_score_range"
        ),
        CheckConstraint(
            "utilization_ratio >= 0 AND utilization_ratio <= 1",
            name="check_utilization_ratio_range",
        ),
        CheckConstraint(
            "exchange IN ('binance', 'coinbase', 'okx', 'mock')",
            name="check_exchange_allocation_supported_exchange",
        ),
    )

    def __repr__(self):
        return f"<ExchangeAllocationDB {self.exchange}: {self.total_allocation}>"
