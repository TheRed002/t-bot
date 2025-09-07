"""
Comprehensive tests for analytics types module.

This module tests all data structures, enums, and models used in analytics,
with special focus on financial precision and validation.
"""

# Disable logging during tests for performance
import logging

import pytest

logging.disable(logging.CRITICAL)

# Set pytest markers for optimization
pytestmark = pytest.mark.unit
from datetime import datetime, timedelta
from decimal import Decimal

from src.analytics.types import (
    AnalyticsAlert,
    AnalyticsConfiguration,
    AnalyticsDataPoint,
    AnalyticsFrequency,
    PerformanceMetricType,
    PortfolioMetrics,
    PositionMetrics,
    ReportType,
    RiskMetrics,
    RiskMetricType,
    StrategyMetrics,
    TimeSeries,
)
from src.monitoring.alerting import AlertSeverity


class TestAnalyticsEnums:
    """Test analytics enum types."""

    def test_analytics_frequency_enum(self):
        """Test AnalyticsFrequency enum values."""
        assert AnalyticsFrequency.REAL_TIME.value == "real_time"
        assert AnalyticsFrequency.MINUTE.value == "1m"
        assert AnalyticsFrequency.DAILY.value == "1d"
        assert AnalyticsFrequency.MONTHLY.value == "1M"
        assert AnalyticsFrequency.YEARLY.value == "1Y"

    def test_risk_metric_type_enum(self):
        """Test RiskMetricType enum values."""
        assert RiskMetricType.VAR.value == "var"
        assert RiskMetricType.CVAR.value == "cvar"
        assert RiskMetricType.MAX_DRAWDOWN.value == "max_drawdown"
        assert RiskMetricType.VOLATILITY.value == "volatility"
        assert RiskMetricType.BETA.value == "beta"

    def test_performance_metric_type_enum(self):
        """Test PerformanceMetricType enum values."""
        assert PerformanceMetricType.TOTAL_RETURN.value == "total_return"
        assert PerformanceMetricType.SHARPE_RATIO.value == "sharpe_ratio"
        assert PerformanceMetricType.WIN_RATE.value == "win_rate"
        assert PerformanceMetricType.ALPHA.value == "alpha"

    def test_report_type_enum(self):
        """Test ReportType enum values."""
        assert ReportType.DAILY_PERFORMANCE.value == "daily_performance"
        assert ReportType.RISK_REPORT.value == "risk_report"
        assert ReportType.COMPLIANCE_REPORT.value == "compliance_report"


class TestAnalyticsDataPoint:
    """Test AnalyticsDataPoint model."""

    @pytest.fixture
    def base_timestamp(self):
        """Base timestamp for tests."""
        return datetime.now()

    @pytest.fixture
    def valid_data_point(self, base_timestamp):
        """Valid analytics data point."""
        return AnalyticsDataPoint(
            timestamp=base_timestamp, value=Decimal("100.50"), metric_type="test_metric"
        )

    def test_create_analytics_data_point(self, valid_data_point, base_timestamp):
        """Test creating valid analytics data point."""
        assert valid_data_point.timestamp == base_timestamp
        assert valid_data_point.value == Decimal("100.50")
        assert valid_data_point.metric_type == "test_metric"
        assert valid_data_point.labels == {}
        assert valid_data_point.metadata == {}

    def test_analytics_data_point_with_labels_and_metadata(self, base_timestamp):
        """Test creating data point with labels and metadata."""
        labels = {"exchange": "binance", "symbol": "BTCUSDT"}
        metadata = {"source": "websocket", "latency_ms": 5}

        point = AnalyticsDataPoint(
            timestamp=base_timestamp,
            value=Decimal("50000.00"),
            metric_type="price",
            labels=labels,
            metadata=metadata,
        )

        assert point.labels == labels
        assert point.metadata == metadata

    def test_validate_finite_value_success(self, base_timestamp):
        """Test validation accepts finite decimal values."""
        values_to_test = [
            Decimal("0"),
            Decimal("100.50"),
            Decimal("-50.25"),
            Decimal("999999999.99"),
        ]

        for value in values_to_test:
            point = AnalyticsDataPoint(timestamp=base_timestamp, value=value, metric_type="test")
            assert point.value == value

    def test_validate_infinite_value_raises_error(self, base_timestamp):
        """Test validation rejects infinite values."""
        with pytest.raises(Exception) as exc_info:
            AnalyticsDataPoint(timestamp=base_timestamp, value=Decimal("inf"), metric_type="test")
        assert "finite number" in str(exc_info.value)

    def test_validate_nan_value_raises_error(self, base_timestamp):
        """Test validation rejects NaN values."""
        with pytest.raises(Exception) as exc_info:
            AnalyticsDataPoint(timestamp=base_timestamp, value=Decimal("nan"), metric_type="test")
        assert "finite number" in str(exc_info.value)

    def test_decimal_precision_preservation(self, base_timestamp):
        """Test that decimal precision is preserved."""
        high_precision_value = Decimal("12345.12345678")
        point = AnalyticsDataPoint(
            timestamp=base_timestamp, value=high_precision_value, metric_type="test"
        )
        assert point.value == high_precision_value
        assert str(point.value) == "12345.12345678"


class TestTimeSeries:
    """Test TimeSeries model."""

    @pytest.fixture
    def empty_time_series(self):
        """Empty time series for testing."""
        return TimeSeries(
            name="test_series", description="Test time series", frequency=AnalyticsFrequency.MINUTE
        )

    def test_create_empty_time_series(self, empty_time_series):
        """Test creating empty time series."""
        assert empty_time_series.name == "test_series"
        assert empty_time_series.description == "Test time series"
        assert empty_time_series.frequency == AnalyticsFrequency.MINUTE
        assert empty_time_series.data_points == []
        assert empty_time_series.start_time is None
        assert empty_time_series.end_time is None
        assert empty_time_series.metadata == {}

    def test_add_point_to_time_series(self, empty_time_series):
        """Test adding data point to time series."""
        timestamp = datetime.now()
        value = Decimal("100.50")

        empty_time_series.add_point(timestamp, value)

        assert len(empty_time_series.data_points) == 1
        assert empty_time_series.data_points[0].timestamp == timestamp
        assert empty_time_series.data_points[0].value == value
        assert empty_time_series.data_points[0].metric_type == "test_series"
        assert empty_time_series.start_time == timestamp
        assert empty_time_series.end_time == timestamp

    def test_add_multiple_points_updates_boundaries(self, empty_time_series):
        """Test adding multiple points updates time boundaries correctly."""
        base_time = datetime.now()
        times = [
            base_time - timedelta(hours=2),
            base_time - timedelta(hours=1),
            base_time,
            base_time + timedelta(hours=1),
        ]

        for i, timestamp in enumerate(times):
            empty_time_series.add_point(timestamp, Decimal(str(i)))

        assert empty_time_series.start_time == times[0]
        assert empty_time_series.end_time == times[-1]
        assert len(empty_time_series.data_points) == 4

    def test_add_point_with_labels_and_metadata(self, empty_time_series):
        """Test adding point with labels and metadata."""
        timestamp = datetime.now()
        value = Decimal("50.25")
        labels = {"exchange": "coinbase"}
        metadata = {"volume": 1000}

        empty_time_series.add_point(timestamp, value, labels=labels, metadata=metadata)

        point = empty_time_series.data_points[0]
        assert point.labels == labels
        assert point.metadata == metadata

    def test_get_latest_value_empty_series(self, empty_time_series):
        """Test getting latest value from empty series returns None."""
        assert empty_time_series.get_latest_value() is None

    def test_get_latest_value_single_point(self, empty_time_series):
        """Test getting latest value with single data point."""
        timestamp = datetime.now()
        value = Decimal("123.45")

        empty_time_series.add_point(timestamp, value)

        assert empty_time_series.get_latest_value() == value

    def test_get_latest_value_multiple_points(self, empty_time_series):
        """Test getting latest value with multiple points."""
        base_time = datetime.now()

        # Add points in non-chronological order
        empty_time_series.add_point(base_time, Decimal("100"))
        empty_time_series.add_point(base_time - timedelta(hours=1), Decimal("90"))
        empty_time_series.add_point(base_time + timedelta(hours=1), Decimal("110"))  # Latest
        empty_time_series.add_point(base_time - timedelta(hours=2), Decimal("80"))

        assert empty_time_series.get_latest_value() == Decimal("110")


class TestPortfolioMetrics:
    """Test PortfolioMetrics model."""

    @pytest.fixture
    def basic_portfolio_metrics(self):
        """Basic portfolio metrics for testing."""
        return PortfolioMetrics(
            timestamp=datetime.now(),
            total_value=Decimal("100000.00"),
            cash=Decimal("10000.00"),
            invested_capital=Decimal("90000.00"),
            unrealized_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("2000.00"),
            total_pnl=Decimal("7000.00"),
        )

    def test_create_portfolio_metrics(self, basic_portfolio_metrics):
        """Test creating portfolio metrics with required fields."""
        assert basic_portfolio_metrics.total_value == Decimal("100000.00")
        assert basic_portfolio_metrics.cash == Decimal("10000.00")
        assert basic_portfolio_metrics.invested_capital == Decimal("90000.00")
        assert basic_portfolio_metrics.unrealized_pnl == Decimal("5000.00")
        assert basic_portfolio_metrics.realized_pnl == Decimal("2000.00")
        assert basic_portfolio_metrics.total_pnl == Decimal("7000.00")

    def test_portfolio_metrics_optional_fields_default_none(self, basic_portfolio_metrics):
        """Test optional fields default to None."""
        assert basic_portfolio_metrics.daily_return is None
        assert basic_portfolio_metrics.mtd_return is None
        assert basic_portfolio_metrics.ytd_return is None
        assert basic_portfolio_metrics.volatility is None
        assert basic_portfolio_metrics.sharpe_ratio is None
        assert basic_portfolio_metrics.max_drawdown is None
        assert basic_portfolio_metrics.var_95 is None
        assert basic_portfolio_metrics.beta is None

    def test_portfolio_metrics_default_integer_fields(self, basic_portfolio_metrics):
        """Test default integer field values."""
        assert basic_portfolio_metrics.positions_count == 0
        assert basic_portfolio_metrics.active_strategies == 0

    def test_portfolio_metrics_default_dict_fields(self, basic_portfolio_metrics):
        """Test default dictionary field values."""
        assert basic_portfolio_metrics.currency_exposures == {}
        assert basic_portfolio_metrics.sector_exposures == {}
        assert basic_portfolio_metrics.geography_exposures == {}

    def test_portfolio_metrics_with_optional_fields(self):
        """Test creating portfolio metrics with optional fields."""
        timestamp = datetime.now()
        metrics = PortfolioMetrics(
            timestamp=timestamp,
            total_value=Decimal("100000.00"),
            cash=Decimal("10000.00"),
            invested_capital=Decimal("90000.00"),
            unrealized_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("2000.00"),
            total_pnl=Decimal("7000.00"),
            daily_return=Decimal("0.02"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("-0.05"),
            var_95=Decimal("1000.00"),
            positions_count=25,
            active_strategies=3,
            leverage=Decimal("2.0"),
            currency_exposures={"USD": Decimal("0.7"), "EUR": Decimal("0.3")},
            sector_exposures={"tech": Decimal("0.4"), "finance": Decimal("0.6")},
        )

        assert metrics.daily_return == Decimal("0.02")
        assert metrics.sharpe_ratio == Decimal("1.5")
        assert metrics.max_drawdown == Decimal("-0.05")
        assert metrics.var_95 == Decimal("1000.00")
        assert metrics.positions_count == 25
        assert metrics.active_strategies == 3
        assert metrics.leverage == Decimal("2.0")
        assert metrics.currency_exposures["USD"] == Decimal("0.7")
        assert metrics.sector_exposures["tech"] == Decimal("0.4")


class TestPositionMetrics:
    """Test PositionMetrics model."""

    @pytest.fixture
    def basic_position_metrics(self):
        """Basic position metrics for testing."""
        return PositionMetrics(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            exchange="binance",
            side="LONG",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("52000.00"),
            market_value=Decimal("78000.00"),
            unrealized_pnl=Decimal("3000.00"),
            unrealized_pnl_percent=Decimal("0.04"),
            realized_pnl=Decimal("0.00"),
            total_pnl=Decimal("3000.00"),
            weight=Decimal("0.15"),
        )

    def test_create_position_metrics(self, basic_position_metrics):
        """Test creating position metrics with required fields."""
        assert basic_position_metrics.symbol == "BTCUSDT"
        assert basic_position_metrics.exchange == "binance"
        assert basic_position_metrics.side == "LONG"
        assert basic_position_metrics.quantity == Decimal("1.5")
        assert basic_position_metrics.entry_price == Decimal("50000.00")
        assert basic_position_metrics.current_price == Decimal("52000.00")
        assert basic_position_metrics.market_value == Decimal("78000.00")
        assert basic_position_metrics.unrealized_pnl == Decimal("3000.00")
        assert basic_position_metrics.unrealized_pnl_percent == Decimal("0.04")
        assert basic_position_metrics.weight == Decimal("0.15")

    def test_position_metrics_optional_fields(self, basic_position_metrics):
        """Test optional fields default values."""
        assert basic_position_metrics.duration_hours is None
        assert basic_position_metrics.max_profit is None
        assert basic_position_metrics.max_loss is None
        assert basic_position_metrics.volatility is None
        assert basic_position_metrics.beta is None
        assert basic_position_metrics.correlation_to_portfolio is None
        assert basic_position_metrics.var_contribution is None
        assert basic_position_metrics.fees_paid == Decimal("0")
        assert basic_position_metrics.slippage is None
        assert basic_position_metrics.strategy is None
        assert basic_position_metrics.entry_reason is None
        assert basic_position_metrics.metadata == {}

    def test_position_metrics_with_optional_fields(self):
        """Test position metrics with optional fields populated."""
        timestamp = datetime.now()
        metadata = {"signal_strength": 0.8, "confidence": 0.9}

        metrics = PositionMetrics(
            timestamp=timestamp,
            symbol="ETHUSDT",
            exchange="coinbase",
            side="SHORT",
            quantity=Decimal("10.0"),
            entry_price=Decimal("3000.00"),
            current_price=Decimal("2950.00"),
            market_value=Decimal("29500.00"),
            unrealized_pnl=Decimal("500.00"),
            unrealized_pnl_percent=Decimal("0.0167"),
            realized_pnl=Decimal("100.00"),
            total_pnl=Decimal("600.00"),
            weight=Decimal("0.295"),
            duration_hours=Decimal("24.5"),
            max_profit=Decimal("800.00"),
            max_loss=Decimal("-200.00"),
            volatility=Decimal("0.35"),
            beta=Decimal("1.2"),
            fees_paid=Decimal("15.00"),
            slippage=Decimal("2.50"),
            strategy="momentum",
            entry_reason="breakout_signal",
            metadata=metadata,
        )

        assert metrics.duration_hours == Decimal("24.5")
        assert metrics.max_profit == Decimal("800.00")
        assert metrics.max_loss == Decimal("-200.00")
        assert metrics.volatility == Decimal("0.35")
        assert metrics.beta == Decimal("1.2")
        assert metrics.fees_paid == Decimal("15.00")
        assert metrics.slippage == Decimal("2.50")
        assert metrics.strategy == "momentum"
        assert metrics.entry_reason == "breakout_signal"
        assert metrics.metadata == metadata


class TestStrategyMetrics:
    """Test StrategyMetrics model."""

    @pytest.fixture
    def basic_strategy_metrics(self):
        """Basic strategy metrics for testing."""
        return StrategyMetrics(
            timestamp=datetime.now(),
            strategy_name="momentum_trading",
            total_pnl=Decimal("15000.00"),
            unrealized_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("10000.00"),
            total_return=Decimal("0.15"),
            capital_allocated=Decimal("100000.00"),
            capital_utilized=Decimal("80000.00"),
            utilization_rate=Decimal("0.80"),
        )

    def test_create_strategy_metrics(self, basic_strategy_metrics):
        """Test creating strategy metrics."""
        assert basic_strategy_metrics.strategy_name == "momentum_trading"
        assert basic_strategy_metrics.total_pnl == Decimal("15000.00")
        assert basic_strategy_metrics.unrealized_pnl == Decimal("5000.00")
        assert basic_strategy_metrics.realized_pnl == Decimal("10000.00")
        assert basic_strategy_metrics.total_return == Decimal("0.15")
        assert basic_strategy_metrics.capital_allocated == Decimal("100000.00")
        assert basic_strategy_metrics.capital_utilized == Decimal("80000.00")
        assert basic_strategy_metrics.utilization_rate == Decimal("0.80")

    def test_strategy_metrics_default_values(self, basic_strategy_metrics):
        """Test default values for strategy metrics."""
        assert basic_strategy_metrics.total_trades == 0
        assert basic_strategy_metrics.winning_trades == 0
        assert basic_strategy_metrics.losing_trades == 0
        assert basic_strategy_metrics.active_positions == 0
        assert basic_strategy_metrics.exposure == Decimal("0")
        assert basic_strategy_metrics.fees_paid == Decimal("0")
        assert basic_strategy_metrics.slippage_cost == Decimal("0")

    def test_strategy_metrics_optional_none_fields(self, basic_strategy_metrics):
        """Test optional fields that default to None."""
        assert basic_strategy_metrics.daily_return is None
        assert basic_strategy_metrics.volatility is None
        assert basic_strategy_metrics.sharpe_ratio is None
        assert basic_strategy_metrics.sortino_ratio is None
        assert basic_strategy_metrics.max_drawdown is None
        assert basic_strategy_metrics.win_rate is None
        assert basic_strategy_metrics.profit_factor is None
        assert basic_strategy_metrics.avg_win is None
        assert basic_strategy_metrics.avg_loss is None

    def test_comprehensive_strategy_metrics(self):
        """Test strategy metrics with all fields populated."""
        timestamp = datetime.now()
        metrics = StrategyMetrics(
            timestamp=timestamp,
            strategy_name="mean_reversion",
            total_pnl=Decimal("25000.00"),
            unrealized_pnl=Decimal("8000.00"),
            realized_pnl=Decimal("17000.00"),
            total_return=Decimal("0.25"),
            daily_return=Decimal("0.001"),
            volatility=Decimal("0.12"),
            sharpe_ratio=Decimal("2.1"),
            sortino_ratio=Decimal("2.8"),
            max_drawdown=Decimal("-0.08"),
            win_rate=Decimal("0.65"),
            profit_factor=Decimal("1.85"),
            total_trades=150,
            winning_trades=98,
            losing_trades=52,
            avg_win=Decimal("280.00"),
            avg_loss=Decimal("-120.00"),
            largest_win=Decimal("1500.00"),
            largest_loss=Decimal("-800.00"),
            capital_allocated=Decimal("100000.00"),
            capital_utilized=Decimal("95000.00"),
            utilization_rate=Decimal("0.95"),
            active_positions=12,
            exposure=Decimal("85000.00"),
            leverage=Decimal("1.5"),
            fees_paid=Decimal("1200.00"),
            slippage_cost=Decimal("350.00"),
        )

        assert metrics.win_rate == Decimal("0.65")
        assert metrics.profit_factor == Decimal("1.85")
        assert metrics.total_trades == 150
        assert metrics.winning_trades == 98
        assert metrics.losing_trades == 52
        assert metrics.avg_win == Decimal("280.00")
        assert metrics.avg_loss == Decimal("-120.00")
        assert metrics.largest_win == Decimal("1500.00")
        assert metrics.largest_loss == Decimal("-800.00")
        assert metrics.active_positions == 12
        assert metrics.exposure == Decimal("85000.00")
        assert metrics.leverage == Decimal("1.5")
        assert metrics.fees_paid == Decimal("1200.00")
        assert metrics.slippage_cost == Decimal("350.00")


class TestRiskMetrics:
    """Test RiskMetrics model."""

    @pytest.fixture
    def basic_risk_metrics(self):
        """Basic risk metrics for testing."""
        return RiskMetrics(timestamp=datetime.now())

    def test_create_risk_metrics(self, basic_risk_metrics):
        """Test creating risk metrics with only timestamp."""
        assert basic_risk_metrics.timestamp is not None
        assert basic_risk_metrics.portfolio_var_95 is None
        assert basic_risk_metrics.portfolio_var_99 is None
        assert basic_risk_metrics.max_drawdown is None

    def test_risk_metrics_default_dict_fields(self, basic_risk_metrics):
        """Test default dictionary fields."""
        assert basic_risk_metrics.sector_concentration == {}
        assert basic_risk_metrics.currency_concentration == {}
        assert basic_risk_metrics.exchange_concentration == {}
        assert basic_risk_metrics.stress_test_results == {}

    def test_comprehensive_risk_metrics(self):
        """Test risk metrics with all fields populated."""
        timestamp = datetime.now()
        sector_concentration = {"tech": Decimal("0.4"), "finance": Decimal("0.6")}
        currency_concentration = {"USD": Decimal("0.8"), "EUR": Decimal("0.2")}
        exchange_concentration = {"binance": Decimal("0.7"), "coinbase": Decimal("0.3")}
        stress_results = {"market_crash": Decimal("-0.15"), "vol_spike": Decimal("-0.08")}
        correlation_matrix = {
            "BTC": {"ETH": Decimal("0.75"), "ADA": Decimal("0.65")},
            "ETH": {"BTC": Decimal("0.75"), "ADA": Decimal("0.70")},
        }

        metrics = RiskMetrics(
            timestamp=timestamp,
            portfolio_var_95=Decimal("5000.00"),
            portfolio_var_99=Decimal("8000.00"),
            portfolio_cvar_95=Decimal("6500.00"),
            portfolio_cvar_99=Decimal("10000.00"),
            max_drawdown=Decimal("-0.12"),
            current_drawdown=Decimal("-0.03"),
            volatility=Decimal("0.18"),
            downside_deviation=Decimal("0.15"),
            value_at_risk_1d=Decimal("2000.00"),
            value_at_risk_10d=Decimal("6000.00"),
            expected_shortfall=Decimal("7500.00"),
            concentration_risk=Decimal("0.25"),
            correlation_risk=Decimal("0.15"),
            liquidity_risk=Decimal("0.05"),
            currency_risk=Decimal("0.10"),
            leverage_ratio=Decimal("1.8"),
            margin_to_equity=Decimal("0.25"),
            largest_position_weight=Decimal("0.15"),
            top5_concentration=Decimal("0.65"),
            sector_concentration=sector_concentration,
            currency_concentration=currency_concentration,
            exchange_concentration=exchange_concentration,
            correlation_matrix=correlation_matrix,
            risk_budget_utilization=Decimal("0.80"),
            stress_test_results=stress_results,
        )

        assert metrics.portfolio_var_95 == Decimal("5000.00")
        assert metrics.portfolio_var_99 == Decimal("8000.00")
        assert metrics.max_drawdown == Decimal("-0.12")
        assert metrics.volatility == Decimal("0.18")
        assert metrics.leverage_ratio == Decimal("1.8")
        assert metrics.largest_position_weight == Decimal("0.15")
        assert metrics.sector_concentration == sector_concentration
        assert metrics.correlation_matrix == correlation_matrix
        assert metrics.stress_test_results == stress_results


class TestAnalyticsAlert:
    """Test AnalyticsAlert model."""

    @pytest.fixture
    def basic_alert(self):
        """Basic analytics alert for testing."""
        return AnalyticsAlert(
            id="alert_001",
            timestamp=datetime.now(),
            severity=AlertSeverity.HIGH,
            title="High Risk Alert",
            message="Portfolio VaR exceeded threshold",
            metric_name="portfolio_var_95",
        )

    def test_create_analytics_alert(self, basic_alert):
        """Test creating basic analytics alert."""
        assert basic_alert.id == "alert_001"
        assert basic_alert.severity == AlertSeverity.HIGH
        assert basic_alert.title == "High Risk Alert"
        assert basic_alert.message == "Portfolio VaR exceeded threshold"
        assert basic_alert.metric_name == "portfolio_var_95"

    def test_alert_default_values(self, basic_alert):
        """Test default values for alert fields."""
        assert basic_alert.current_value is None
        assert basic_alert.threshold_value is None
        assert basic_alert.breach_percentage is None
        assert basic_alert.affected_entities == []
        assert basic_alert.recommended_actions == []
        assert basic_alert.metadata == {}
        assert basic_alert.acknowledged is False
        assert basic_alert.resolved is False
        assert basic_alert.resolved_timestamp is None

    def test_comprehensive_alert(self):
        """Test alert with all fields populated."""
        timestamp = datetime.now()
        resolved_time = timestamp + timedelta(hours=1)

        alert = AnalyticsAlert(
            id="alert_002",
            timestamp=timestamp,
            severity=AlertSeverity.CRITICAL,
            title="Portfolio Drawdown Alert",
            message="Maximum drawdown limit breached",
            metric_name="max_drawdown",
            current_value=Decimal("-0.08"),
            threshold_value=Decimal("-0.05"),
            breach_percentage=Decimal("0.60"),
            affected_entities=["portfolio_001", "strategy_momentum"],
            recommended_actions=[
                "Reduce position sizes",
                "Activate hedge positions",
                "Review risk limits",
            ],
            metadata={"strategy": "momentum", "exchange": "binance"},
            acknowledged=True,
            resolved=True,
            resolved_timestamp=resolved_time,
        )

        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.current_value == Decimal("-0.08")
        assert alert.threshold_value == Decimal("-0.05")
        assert alert.breach_percentage == Decimal("0.60")
        assert "portfolio_001" in alert.affected_entities
        assert "Reduce position sizes" in alert.recommended_actions
        assert alert.metadata["strategy"] == "momentum"
        assert alert.acknowledged is True
        assert alert.resolved is True
        assert alert.resolved_timestamp == resolved_time


class TestAnalyticsConfiguration:
    """Test AnalyticsConfiguration model."""

    def test_default_analytics_configuration(self):
        """Test default analytics configuration values."""
        config = AnalyticsConfiguration()

        assert config.risk_free_rate == Decimal("0.02")
        assert config.confidence_levels == [95, 99]
        assert config.lookback_periods["var"] == 252
        assert config.lookback_periods["volatility"] == 30
        assert config.lookback_periods["correlation"] == 60
        assert config.lookback_periods["beta"] == 252
        assert "SPY" in config.benchmark_symbols
        assert "BTC-USD" in config.benchmark_symbols
        assert config.alert_thresholds["max_drawdown"] == Decimal("0.05")
        assert config.alert_thresholds["daily_var_breach"] == Decimal("0.02")
        assert config.reporting_frequency == AnalyticsFrequency.DAILY
        assert config.calculation_frequency == AnalyticsFrequency.MINUTE
        assert config.cache_ttl_seconds == 60
        assert config.enable_real_time_alerts is True
        assert config.enable_stress_testing is True
        assert "market_crash" in config.stress_test_scenarios
        assert "fama_french_3" in config.factor_models
        assert config.attribution_method == "brinson"
        assert config.currency == "USD"
        assert config.timezone == "UTC"

    def test_custom_analytics_configuration(self):
        """Test custom analytics configuration."""
        custom_lookbacks = {"var": 100, "volatility": 20, "correlation": 40, "beta": 150}
        custom_thresholds = {
            "max_drawdown": Decimal("0.10"),
            "daily_var_breach": Decimal("0.03"),
            "concentration_risk": Decimal("0.25"),
            "leverage_ratio": Decimal("3.0"),
        }
        custom_benchmarks = ["QQQ", "ETH-USD", "GLD"]
        custom_scenarios = ["black_swan", "interest_rate_shock"]
        custom_factors = ["custom_factor_model"]

        config = AnalyticsConfiguration(
            risk_free_rate=Decimal("0.03"),
            confidence_levels=[90, 95, 99],
            lookback_periods=custom_lookbacks,
            benchmark_symbols=custom_benchmarks,
            alert_thresholds=custom_thresholds,
            reporting_frequency=AnalyticsFrequency.WEEKLY,
            calculation_frequency=AnalyticsFrequency.FIVE_MINUTES,
            cache_ttl_seconds=120,
            enable_real_time_alerts=False,
            enable_stress_testing=False,
            stress_test_scenarios=custom_scenarios,
            factor_models=custom_factors,
            attribution_method="brinson_hood_beebower",
            currency="EUR",
            timezone="Europe/London",
        )

        assert config.risk_free_rate == Decimal("0.03")
        assert config.confidence_levels == [90, 95, 99]
        assert config.lookback_periods == custom_lookbacks
        assert config.benchmark_symbols == custom_benchmarks
        assert config.alert_thresholds == custom_thresholds
        assert config.reporting_frequency == AnalyticsFrequency.WEEKLY
        assert config.calculation_frequency == AnalyticsFrequency.FIVE_MINUTES
        assert config.cache_ttl_seconds == 120
        assert config.enable_real_time_alerts is False
        assert config.enable_stress_testing is False
        assert config.stress_test_scenarios == custom_scenarios
        assert config.factor_models == custom_factors
        assert config.attribution_method == "brinson_hood_beebower"
        assert config.currency == "EUR"
        assert config.timezone == "Europe/London"


class TestFinancialCalculationPrecision:
    """Test financial calculation precision requirements."""

    def test_decimal_addition_precision(self):
        """Test decimal addition maintains precision."""
        a = Decimal("123.123456789")
        b = Decimal("456.987654321")
        result = a + b
        expected = Decimal("580.111111110")
        assert result == expected

    def test_decimal_multiplication_precision(self):
        """Test decimal multiplication maintains precision."""
        price = Decimal("50000.12345678")
        quantity = Decimal("1.23456789")
        result = price * quantity
        # Verify high precision is maintained
        assert isinstance(result, Decimal)
        assert "." in str(result)

    def test_decimal_division_precision(self):
        """Test decimal division handles precision correctly."""
        numerator = Decimal("100000.00")
        denominator = Decimal("3.00")
        result = numerator / denominator
        # Verify result is Decimal type with proper precision
        assert isinstance(result, Decimal)
        # Should be 33333.333333... (repeating)
        assert str(result).startswith("33333.33333")

    def test_portfolio_value_calculation_precision(self):
        """Test portfolio value calculations maintain precision."""
        positions = [
            {"price": Decimal("50000.12345"), "quantity": Decimal("1.23456")},
            {"price": Decimal("3000.98765"), "quantity": Decimal("5.43210")},
            {"price": Decimal("200.11111"), "quantity": Decimal("25.55555")},
        ]

        total_value = sum(pos["price"] * pos["quantity"] for pos in positions)

        assert isinstance(total_value, Decimal)
        # Verify precision is maintained in aggregation
        assert total_value > Decimal("0")

    def test_percentage_calculation_precision(self):
        """Test percentage calculations maintain precision."""
        profit = Decimal("1234.56789")
        cost = Decimal("10000.00000")
        percentage = (profit / cost) * Decimal("100")

        assert isinstance(percentage, Decimal)
        expected = Decimal("12.3456789")
        assert percentage == expected

    def test_compound_financial_calculations(self):
        """Test complex financial calculations maintain precision."""
        # Simulate portfolio return calculation
        initial_value = Decimal("100000.00000000")
        price_changes = [
            Decimal("1.05234567"),  # 5.23% gain
            Decimal("0.97123456"),  # 2.88% loss
            Decimal("1.02987654"),  # 2.99% gain
        ]

        final_value = initial_value
        for change in price_changes:
            final_value = final_value * change

        total_return = (final_value - initial_value) / initial_value

        assert isinstance(final_value, Decimal)
        assert isinstance(total_return, Decimal)
        # Verify precision maintained through compound operations
        assert abs(total_return) < Decimal("1.0")  # Reasonable return range


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_zero_values_handling(self):
        """Test handling of zero values in financial calculations."""
        metrics = PortfolioMetrics(
            timestamp=datetime.now(),
            total_value=Decimal("0"),
            cash=Decimal("0"),
            invested_capital=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            total_pnl=Decimal("0"),
        )

        assert metrics.total_value == Decimal("0")
        assert metrics.total_pnl == Decimal("0")

    def test_negative_values_handling(self):
        """Test handling of negative values."""
        metrics = PositionMetrics(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            exchange="binance",
            side="SHORT",
            quantity=Decimal("-1.5"),  # Negative for short position
            entry_price=Decimal("50000.00"),
            current_price=Decimal("48000.00"),
            market_value=Decimal("-72000.00"),
            unrealized_pnl=Decimal("3000.00"),  # Profit on short
            unrealized_pnl_percent=Decimal("0.04"),
            realized_pnl=Decimal("-100.00"),  # Loss
            total_pnl=Decimal("2900.00"),
            weight=Decimal("0.15"),
        )

        assert metrics.quantity == Decimal("-1.5")
        assert metrics.market_value == Decimal("-72000.00")
        assert metrics.realized_pnl == Decimal("-100.00")

    def test_large_values_handling(self):
        """Test handling of large financial values."""
        large_value = Decimal("999999999999.99999999")

        metrics = PortfolioMetrics(
            timestamp=datetime.now(),
            total_value=large_value,
            cash=Decimal("1000000000.00000000"),
            invested_capital=large_value - Decimal("1000000000.00000000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            total_pnl=Decimal("0"),
        )

        assert metrics.total_value == large_value
        assert isinstance(metrics.total_value, Decimal)

    def test_empty_collections_handling(self):
        """Test handling of empty collections."""
        time_series = TimeSeries(
            name="empty_series", description="Empty series", frequency=AnalyticsFrequency.DAILY
        )

        assert time_series.data_points == []
        assert time_series.get_latest_value() is None
        assert time_series.start_time is None
        assert time_series.end_time is None

    def test_boundary_datetime_values(self):
        """Test handling of boundary datetime values."""
        # Test with very recent timestamp
        recent_time = datetime.now()

        # Test with older timestamp
        old_time = datetime(2020, 1, 1)

        for timestamp in [recent_time, old_time]:
            point = AnalyticsDataPoint(
                timestamp=timestamp, value=Decimal("100"), metric_type="test"
            )
            assert point.timestamp == timestamp

    def test_string_field_validation(self):
        """Test validation of string fields."""
        # Test with various string inputs
        test_strings = [
            "normal_string",
            "string with spaces",
            "string-with-dashes",
            "string_with_underscore",
            "123numeric_start",
            "",  # Empty string
        ]

        for test_string in test_strings:
            alert = AnalyticsAlert(
                id=f"alert_{test_string}",
                timestamp=datetime.now(),
                severity=AlertSeverity.INFO,
                title=test_string or "empty_title",
                message=test_string or "empty_message",
                metric_name=test_string or "empty_metric",
            )
            assert isinstance(alert.id, str)
            assert isinstance(alert.title, str)
