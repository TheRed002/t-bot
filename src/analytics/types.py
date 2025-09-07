"""
Analytics Types for T-Bot Trading System.

This module defines comprehensive types for analytics data structures,
including performance metrics, risk measures, and reporting formats.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AnalyticsFrequency(Enum):
    """Analytics calculation and reporting frequency."""

    REAL_TIME = "real_time"
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    FOUR_HOURS = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"
    QUARTERLY = "3M"
    YEARLY = "1Y"


class RiskMetricType(Enum):
    """Types of risk metrics."""

    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional Value at Risk (Expected Shortfall)
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    TRACKING_ERROR = "tracking_error"
    CONCENTRATION = "concentration"


class PerformanceMetricType(Enum):
    """Types of performance metrics."""

    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    TREYNOR_RATIO = "treynor_ratio"
    ALPHA = "alpha"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    PAYOFF_RATIO = "payoff_ratio"


from src.core.types import AlertSeverity


class ReportType(Enum):
    """Types of analytics reports."""

    DAILY_PERFORMANCE = "daily_performance"
    WEEKLY_PERFORMANCE = "weekly_performance"
    MONTHLY_PERFORMANCE = "monthly_performance"
    RISK_REPORT = "risk_report"
    ATTRIBUTION_REPORT = "attribution_report"
    OPERATIONAL_REPORT = "operational_report"
    COMPLIANCE_REPORT = "compliance_report"
    STRATEGY_REPORT = "strategy_report"


class AnalyticsDataPoint(BaseModel):
    """Single analytics data point with timestamp and metadata."""

    timestamp: datetime
    value: Decimal
    metric_type: str
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Decimal) -> Decimal:
        """Validate that value is finite."""
        if not v.is_finite():
            raise ValueError("Value must be finite")
        return v


class TimeSeries(BaseModel):
    """Time series data structure for analytics."""

    name: str
    description: str
    frequency: AnalyticsFrequency
    data_points: list[AnalyticsDataPoint] = Field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_point(self, timestamp: datetime, value: Decimal, **kwargs) -> None:
        """Add a data point to the time series."""
        point = AnalyticsDataPoint(
            timestamp=timestamp,
            value=value,
            metric_type=self.name,
            labels=kwargs.get("labels", {}),
            metadata=kwargs.get("metadata", {}),
        )
        self.data_points.append(point)

        if self.start_time is None or timestamp < self.start_time:
            self.start_time = timestamp
        if self.end_time is None or timestamp > self.end_time:
            self.end_time = timestamp

    def get_latest_value(self) -> Decimal | None:
        """Get the most recent value in the time series."""
        if not self.data_points:
            return None
        return max(self.data_points, key=lambda x: x.timestamp).value


class PortfolioMetrics(BaseModel):
    """Portfolio-level metrics and analytics."""

    timestamp: datetime
    total_value: Decimal
    cash: Decimal
    invested_capital: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_pnl: Decimal
    daily_return: Decimal | None = None
    mtd_return: Decimal | None = None
    ytd_return: Decimal | None = None
    total_return: Decimal | None = None
    volatility: Decimal | None = None
    sharpe_ratio: Decimal | None = None
    max_drawdown: Decimal | None = None
    var_95: Decimal | None = None
    beta: Decimal | None = None
    positions_count: int = 0
    active_strategies: int = 0
    leverage: Decimal | None = None
    margin_used: Decimal | None = None
    currency_exposures: dict[str, Decimal] = Field(default_factory=dict)
    sector_exposures: dict[str, Decimal] = Field(default_factory=dict)
    geography_exposures: dict[str, Decimal] = Field(default_factory=dict)


class PositionMetrics(BaseModel):
    """Individual position metrics and analytics."""

    timestamp: datetime
    symbol: str
    exchange: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: Decimal
    realized_pnl: Decimal
    total_pnl: Decimal
    weight: Decimal  # Position weight in portfolio
    duration_hours: Decimal | None = None
    max_profit: Decimal | None = None
    max_loss: Decimal | None = None
    volatility: Decimal | None = None
    beta: Decimal | None = None
    correlation_to_portfolio: Decimal | None = None
    var_contribution: Decimal | None = None
    fees_paid: Decimal = Decimal("0")
    slippage: Decimal | None = None
    strategy: str | None = None
    entry_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyMetrics(BaseModel):
    """Strategy-level performance metrics."""

    timestamp: datetime
    strategy_name: str
    total_pnl: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_return: Decimal
    daily_return: Decimal | None = None
    volatility: Decimal | None = None
    sharpe_ratio: Decimal | None = None
    sortino_ratio: Decimal | None = None
    max_drawdown: Decimal | None = None
    win_rate: Decimal | None = None
    profit_factor: Decimal | None = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: Decimal | None = None
    avg_loss: Decimal | None = None
    largest_win: Decimal | None = None
    largest_loss: Decimal | None = None
    avg_holding_period: Decimal | None = None
    trades_per_day: Decimal | None = None
    capital_allocated: Decimal
    capital_utilized: Decimal
    utilization_rate: Decimal
    active_positions: int = 0
    exposure: Decimal = Decimal("0")
    leverage: Decimal | None = None
    correlation_to_market: Decimal | None = None
    alpha: Decimal | None = None
    beta: Decimal | None = None
    tracking_error: Decimal | None = None
    information_ratio: Decimal | None = None
    fees_paid: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")
    execution_quality: Decimal | None = None


class RiskMetrics(BaseModel):
    """Comprehensive risk metrics."""

    timestamp: datetime
    portfolio_var_95: Decimal | None = None
    portfolio_var_99: Decimal | None = None
    portfolio_cvar_95: Decimal | None = None
    portfolio_cvar_99: Decimal | None = None
    max_drawdown: Decimal | None = None
    current_drawdown: Decimal | None = None
    volatility: Decimal | None = None
    downside_deviation: Decimal | None = None
    value_at_risk_1d: Decimal | None = None
    value_at_risk_10d: Decimal | None = None
    expected_shortfall: Decimal | None = None
    tail_expectation: Decimal | None = None
    concentration_risk: Decimal | None = None
    correlation_risk: Decimal | None = None
    liquidity_risk: Decimal | None = None
    currency_risk: Decimal | None = None
    leverage_ratio: Decimal | None = None
    margin_to_equity: Decimal | None = None
    largest_position_weight: Decimal | None = None
    top5_concentration: Decimal | None = None
    sector_concentration: dict[str, Decimal] = Field(default_factory=dict)
    currency_concentration: dict[str, Decimal] = Field(default_factory=dict)
    exchange_concentration: dict[str, Decimal] = Field(default_factory=dict)
    correlation_matrix: dict[str, dict[str, Decimal]] | None = None
    risk_budget_utilization: Decimal | None = None
    stress_test_results: dict[str, Decimal] = Field(default_factory=dict)


class TradeAnalytics(BaseModel):
    """Individual trade analytics and metrics."""

    trade_id: str
    timestamp: datetime
    symbol: str
    exchange: str
    strategy: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal | None = None
    realized_pnl: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    fees: Decimal = Decimal("0")
    slippage: Decimal | None = None
    duration_minutes: int | None = None
    entry_spread: Decimal | None = None
    exit_spread: Decimal | None = None
    market_impact: Decimal | None = None
    timing_cost: Decimal | None = None
    opportunity_cost: Decimal | None = None
    execution_quality_score: Decimal | None = None
    risk_adjusted_return: Decimal | None = None
    sharpe_contribution: Decimal | None = None
    var_impact: Decimal | None = None
    portfolio_weight: Decimal | None = None
    benchmark_return: Decimal | None = None
    alpha_contribution: Decimal | None = None
    attribution_factors: dict[str, Decimal] = Field(default_factory=dict)


class PerformanceAttribution(BaseModel):
    """Performance attribution analysis results."""

    timestamp: datetime
    period_start: datetime
    period_end: datetime
    total_return: Decimal
    benchmark_return: Decimal | None = None
    active_return: Decimal | None = None
    asset_selection: Decimal | None = None
    timing_effect: Decimal | None = None
    interaction_effect: Decimal | None = None
    currency_effect: Decimal | None = None
    sector_allocation: dict[str, Decimal] = Field(default_factory=dict)
    stock_selection: dict[str, Decimal] = Field(default_factory=dict)
    strategy_attribution: dict[str, Decimal] = Field(default_factory=dict)
    factor_attribution: dict[str, Decimal] = Field(default_factory=dict)
    residual_return: Decimal | None = None
    tracking_error: Decimal | None = None
    information_ratio: Decimal | None = None


class AnalyticsAlert(BaseModel):
    """Analytics alert for threshold breaches and anomalies."""

    id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    current_value: Decimal | None = None
    threshold_value: Decimal | None = None
    breach_percentage: Decimal | None = None
    affected_entities: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolved_timestamp: datetime | None = None


class AnalyticsReport(BaseModel):
    """Comprehensive analytics report."""

    report_id: str
    report_type: ReportType
    generated_timestamp: datetime
    period_start: datetime
    period_end: datetime
    title: str
    executive_summary: str
    portfolio_metrics: PortfolioMetrics | None = None
    position_metrics: list[PositionMetrics] = Field(default_factory=list)
    strategy_metrics: list[StrategyMetrics] = Field(default_factory=list)
    risk_metrics: RiskMetrics | None = None
    performance_attribution: PerformanceAttribution | None = None
    trade_analytics: list[TradeAnalytics] = Field(default_factory=list)
    alerts: list[AnalyticsAlert] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    methodology_notes: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationalMetrics(BaseModel):
    """Operational analytics and system performance metrics."""

    timestamp: datetime
    system_uptime: Decimal  # Hours
    strategies_active: int
    strategies_total: int
    exchanges_connected: int
    exchanges_total: int
    orders_placed_today: int
    orders_filled_today: int
    order_fill_rate: Decimal
    avg_order_execution_time: Decimal | None = None  # Milliseconds
    avg_order_slippage: Decimal | None = None  # Basis points
    api_call_success_rate: Decimal
    websocket_uptime_percent: Decimal
    data_latency_p50: Decimal | None = None  # Milliseconds
    data_latency_p95: Decimal | None = None  # Milliseconds
    error_rate: Decimal
    critical_errors_today: int
    memory_usage_percent: Decimal
    cpu_usage_percent: Decimal
    disk_usage_percent: Decimal
    network_throughput_mbps: Decimal | None = None
    database_connections_active: int
    database_query_avg_time: Decimal | None = None  # Milliseconds
    cache_hit_rate: Decimal
    backup_status: str
    last_backup_timestamp: datetime | None = None
    compliance_checks_passed: int
    compliance_checks_failed: int
    risk_limit_breaches: int
    circuit_breaker_triggers: int
    performance_degradation_events: int
    data_quality_issues: int
    exchange_outages: int
    recovery_time_minutes: Decimal | None = None


class BenchmarkData(BaseModel):
    """Benchmark data for performance comparison."""

    benchmark_name: str
    timestamp: datetime
    price: Decimal
    return_1d: Decimal | None = None
    return_1w: Decimal | None = None
    return_1m: Decimal | None = None
    return_3m: Decimal | None = None
    return_6m: Decimal | None = None
    return_1y: Decimal | None = None
    return_ytd: Decimal | None = None
    volatility: Decimal | None = None
    sharpe_ratio: Decimal | None = None
    max_drawdown: Decimal | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyticsConfiguration(BaseModel):
    """Configuration for analytics calculations and reporting."""

    risk_free_rate: Decimal = Decimal("0.02")  # 2% annual
    confidence_levels: list[int] = Field(default_factory=lambda: [95, 99])
    lookback_periods: dict[str, int] = Field(
        default_factory=lambda: {
            "var": 252,  # 1 year
            "volatility": 30,  # 1 month
            "correlation": 60,  # 2 months
            "beta": 252,  # 1 year
        }
    )
    benchmark_symbols: list[str] = Field(default_factory=lambda: ["SPY", "BTC-USD"])
    alert_thresholds: dict[str, Decimal] = Field(
        default_factory=lambda: {
            "max_drawdown": Decimal("0.05"),  # 5%
            "daily_var_breach": Decimal("0.02"),  # 2%
            "concentration_risk": Decimal("0.20"),  # 20%
            "leverage_ratio": Decimal("2.0"),  # 2x
        }
    )
    reporting_frequency: AnalyticsFrequency = AnalyticsFrequency.DAILY
    calculation_frequency: AnalyticsFrequency = AnalyticsFrequency.MINUTE
    cache_ttl_seconds: int = 60
    enable_real_time_alerts: bool = True
    enable_stress_testing: bool = True
    stress_test_scenarios: list[str] = Field(
        default_factory=lambda: ["market_crash", "volatility_spike", "liquidity_crisis"]
    )
    factor_models: list[str] = Field(default_factory=lambda: ["fama_french_3", "capm"])
    attribution_method: str = "brinson"  # brinson, brinson_hood_beebower
    currency: str = "USD"
    timezone: str = "UTC"
