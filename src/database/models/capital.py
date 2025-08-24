"""Capital management database models."""

import uuid

from sqlalchemy import DECIMAL, Column, DateTime, Float, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .base import Base, TimestampMixin


class CapitalAllocationDB(Base, TimestampMixin):
    """Capital allocation tracking model."""

    __tablename__ = "capital_allocations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(String(100), nullable=False)
    exchange = Column(String(50), nullable=False)

    # Allocation amounts
    allocated_amount = Column(DECIMAL(20, 8), nullable=False)
    utilized_amount = Column(DECIMAL(20, 8), default=0)
    available_amount = Column(DECIMAL(20, 8), default=0)  # Added field expected by service
    reserved_amount = Column(DECIMAL(20, 8), default=0)

    # Allocation metadata
    allocation_percentage = Column(Float, default=0.0)  # Added field expected by service
    allocation_type = Column(String(50), nullable=False, default="dynamic")  # fixed, percentage, dynamic
    priority = Column(Integer, default=5)  # 1-10 priority scale

    # Performance tracking
    total_return = Column(Float, default=0)
    utilization_ratio = Column(Float, default=0)

    # Rebalancing
    last_rebalance = Column(DateTime(timezone=True), nullable=True)  # Added field expected by service

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Indexes
    __table_args__ = (
        Index("idx_capital_strategy", "strategy_id"),
        Index("idx_capital_exchange", "exchange"),
        Index("idx_capital_type", "allocation_type"),
        Index("idx_capital_created", "created_at"),
        Index("idx_capital_strategy_exchange", "strategy_id", "exchange"),  # Composite index for lookups
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

    # Flow details
    currency = Column(String(10), nullable=False)
    amount = Column(DECIMAL(20, 8), nullable=False)
    fee = Column(DECIMAL(20, 8), default=0)

    # Status tracking
    status = Column(String(20), default="PENDING")  # PENDING, PROCESSING, COMPLETED, FAILED
    transaction_id = Column(String(100))

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Indexes
    __table_args__ = (
        Index("idx_fund_flow_type", "flow_type"),
        Index("idx_fund_flow_currency", "currency"),
        Index("idx_fund_flow_status", "status"),
        Index("idx_fund_flow_created", "created_at"),
    )

    def __repr__(self):
        return f"<FundFlowDB {self.flow_type}: {self.amount} {self.currency}>"


class CurrencyExposureDB(Base, TimestampMixin):
    """Currency exposure tracking model."""

    __tablename__ = "currency_exposures"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    currency = Column(String(10), nullable=False)

    # Exposure amounts
    total_exposure = Column(DECIMAL(20, 8), nullable=False)
    spot_exposure = Column(DECIMAL(20, 8), default=0)
    futures_exposure = Column(DECIMAL(20, 8), default=0)

    # Risk metrics
    var_1d = Column(Float)  # Value at Risk 1 day
    var_7d = Column(Float)  # Value at Risk 7 days
    volatility = Column(Float)

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Indexes
    __table_args__ = (
        Index("idx_currency_exposure", "currency"),
        Index("idx_currency_created", "created_at"),
    )

    def __repr__(self):
        return f"<CurrencyExposureDB {self.currency}: {self.total_exposure}>"


class ExchangeAllocationDB(Base, TimestampMixin):
    """Exchange allocation tracking model."""

    __tablename__ = "exchange_allocations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(50), nullable=False)

    # Allocation amounts
    total_allocation = Column(DECIMAL(20, 8), nullable=False)
    utilized_allocation = Column(DECIMAL(20, 8), default=0)
    reserved_allocation = Column(DECIMAL(20, 8), default=0)

    # Performance metrics
    efficiency_score = Column(Float, default=0)
    utilization_ratio = Column(Float, default=0)

    # Additional data
    metadata_json = Column(JSONB, default={})

    # Indexes
    __table_args__ = (
        Index("idx_exchange_allocation", "exchange"),
        Index("idx_exchange_created", "created_at"),
    )

    def __repr__(self):
        return f"<ExchangeAllocationDB {self.exchange}: {self.total_allocation}>"
