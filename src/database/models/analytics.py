"""Analytics database models."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import NUMERIC as DECIMAL, UUID

from src.database.models.base import Base, TimestampMixin


class AnalyticsPortfolioMetrics(Base, TimestampMixin):
    """Portfolio metrics storage."""
    
    __tablename__ = "analytics_portfolio_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    total_value = Column(DECIMAL(20, 8), nullable=False)
    unrealized_pnl = Column(DECIMAL(20, 8), nullable=False, default=0)
    realized_pnl = Column(DECIMAL(20, 8), nullable=False, default=0)
    daily_pnl = Column(DECIMAL(20, 8), nullable=False, default=0)
    number_of_positions = Column(Integer, nullable=False, default=0)
    leverage_ratio = Column(DECIMAL(10, 4), nullable=True)
    margin_usage = Column(DECIMAL(10, 4), nullable=True)
    cash_balance = Column(DECIMAL(20, 8), nullable=True)


class AnalyticsPositionMetrics(Base, TimestampMixin):
    """Position metrics storage."""
    
    __tablename__ = "analytics_position_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    quantity = Column(DECIMAL(20, 8), nullable=False)
    market_value = Column(DECIMAL(20, 8), nullable=False)
    unrealized_pnl = Column(DECIMAL(20, 8), nullable=False, default=0)
    realized_pnl = Column(DECIMAL(20, 8), nullable=False, default=0)
    average_price = Column(DECIMAL(20, 8), nullable=False)
    current_price = Column(DECIMAL(20, 8), nullable=False)
    position_side = Column(String(10), nullable=False)  # 'long' or 'short'


class AnalyticsRiskMetrics(Base, TimestampMixin):
    """Risk metrics storage."""
    
    __tablename__ = "analytics_risk_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    portfolio_var_95 = Column(DECIMAL(20, 8), nullable=True)
    portfolio_var_99 = Column(DECIMAL(20, 8), nullable=True)
    expected_shortfall_95 = Column(DECIMAL(20, 8), nullable=True)
    maximum_drawdown = Column(DECIMAL(10, 6), nullable=True)
    volatility = Column(DECIMAL(10, 6), nullable=True)
    sharpe_ratio = Column(DECIMAL(10, 4), nullable=True)
    sortino_ratio = Column(DECIMAL(10, 4), nullable=True)
    correlation_risk = Column(DECIMAL(10, 6), nullable=True)
    concentration_risk = Column(DECIMAL(10, 6), nullable=True)


class AnalyticsStrategyMetrics(Base, TimestampMixin):
    """Strategy performance metrics storage."""
    
    __tablename__ = "analytics_strategy_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    total_pnl = Column(DECIMAL(20, 8), nullable=False, default=0)
    average_win = Column(DECIMAL(20, 8), nullable=True)
    average_loss = Column(DECIMAL(20, 8), nullable=True)
    win_rate = Column(DECIMAL(5, 4), nullable=True)  # Percentage as decimal
    profit_factor = Column(DECIMAL(10, 4), nullable=True)
    sharpe_ratio = Column(DECIMAL(10, 4), nullable=True)
    maximum_drawdown = Column(DECIMAL(10, 6), nullable=True)


class AnalyticsOperationalMetrics(Base, TimestampMixin):
    """Operational metrics storage."""
    
    __tablename__ = "analytics_operational_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    orders_per_minute = Column(DECIMAL(10, 2), nullable=False, default=0)
    trades_per_minute = Column(DECIMAL(10, 2), nullable=False, default=0)
    api_latency_avg = Column(DECIMAL(10, 3), nullable=True)  # milliseconds
    api_latency_p95 = Column(DECIMAL(10, 3), nullable=True)
    websocket_latency_avg = Column(DECIMAL(10, 3), nullable=True)
    error_rate = Column(DECIMAL(5, 4), nullable=False, default=0)
    success_rate = Column(DECIMAL(5, 4), nullable=False, default=1)
    active_connections = Column(Integer, nullable=False, default=0)
    memory_usage_mb = Column(DECIMAL(10, 2), nullable=True)
    cpu_usage_percent = Column(DECIMAL(5, 2), nullable=True)
    database_connections_active = Column(Integer, nullable=False, default=0)
    database_query_avg_time = Column(DECIMAL(10, 3), nullable=True)  # milliseconds