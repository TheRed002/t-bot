"""
Tests for strategy metrics module.
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

# Disable logging during tests for performance
logging.disable(logging.CRITICAL)

# Fast mock time for deterministic tests
FIXED_TIME = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

from src.core.types import Signal, SignalDirection
from src.strategies.metrics import (
    MetricsCalculator,
    PerformanceMetrics,
    RealTimeMetricsTracker,
    StrategyComparator,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics model."""

    def test_performance_metrics_initialization(self):
        """Test PerformanceMetrics initialization with defaults."""
        metrics = PerformanceMetrics()

        assert metrics.total_return == Decimal("0")
        assert metrics.annual_return == Decimal("0")
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == Decimal("0")
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_signals == 0

    def test_performance_metrics_with_values(self):
        """Test PerformanceMetrics with specific values."""
        metrics = PerformanceMetrics(
            total_return=Decimal("15.5"),
            sharpe_ratio=1.2,
            max_drawdown=Decimal("5.3"),
            total_trades=150,
            win_rate=0.65,
        )

        assert metrics.total_return == Decimal("15.5")
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == Decimal("5.3")
        assert metrics.total_trades == 150
        assert metrics.win_rate == 0.65

    def test_performance_metrics_decimal_precision(self):
        """Test that financial values use Decimal for precision."""
        metrics = PerformanceMetrics(
            total_return=Decimal("10.123456789"), var_95=Decimal("-2.987654321")
        )

        assert isinstance(metrics.total_return, Decimal)
        assert isinstance(metrics.var_95, Decimal)
        assert metrics.total_return == Decimal("10.123456789")
        assert metrics.var_95 == Decimal("-2.987654321")

    def test_performance_metrics_serialization(self):
        """Test PerformanceMetrics can be serialized."""
        metrics = PerformanceMetrics(
            total_return=Decimal("10.5"), sharpe_ratio=1.3, total_trades=100
        )

        # Should be able to convert to dict
        metrics_dict = metrics.model_dump()

        assert isinstance(metrics_dict, dict)
        assert "total_return" in metrics_dict
        assert "sharpe_ratio" in metrics_dict
        assert "total_trades" in metrics_dict


class TestMetricsCalculator:
    """Test MetricsCalculator functionality."""

    @pytest.fixture(scope="class")
    def calculator(self):
        """Create a MetricsCalculator instance - cached for class scope."""
        config = {"risk_free_rate": 0.02, "trading_days_per_year": 252}
        return MetricsCalculator(config)

    @pytest.fixture(scope="class")
    def sample_equity_curve(self):
        """Create sample equity curve data - cached for class scope with fixed times."""
        base_time = FIXED_TIME  # Use fixed time for performance
        return [
            {"timestamp": base_time, "equity": 10000},
            {"timestamp": base_time + timedelta(days=1), "equity": 10100},
            {"timestamp": base_time + timedelta(days=2), "equity": 10050},
            {"timestamp": base_time + timedelta(days=3), "equity": 10200},
            {"timestamp": base_time + timedelta(days=4), "equity": 10150},
        ]

    @pytest.fixture(scope="class")
    def sample_trades(self):
        """Create sample trade data - cached for class scope with fixed times."""
        return [
            {"pnl": 100, "entry_time": FIXED_TIME, "exit_time": FIXED_TIME + timedelta(hours=2)},
            {"pnl": -50, "entry_time": FIXED_TIME, "exit_time": FIXED_TIME + timedelta(hours=1)},
            {"pnl": 75, "entry_time": FIXED_TIME, "exit_time": FIXED_TIME + timedelta(hours=3)},
            {"pnl": -25, "entry_time": FIXED_TIME, "exit_time": FIXED_TIME + timedelta(hours=1.5)},
        ]

    @pytest.fixture(scope="class")
    def sample_signals(self):
        """Create sample signal data - cached for class scope with fixed times."""
        return [
            Signal(
                signal_id="test_signal_1",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol="BTC/USD",
                direction=SignalDirection.BUY,
                strength=Decimal("0.8"),
                timestamp=FIXED_TIME,  # Use fixed time for performance
                source="test_strategy",
            ),
            Signal(
                signal_id="test_signal_2",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol="BTC/USD",
                direction=SignalDirection.SELL,
                strength=Decimal("0.6"),
                timestamp=FIXED_TIME,  # Use fixed time for performance
                source="test_strategy",
            ),
            Signal(
                signal_id="test_signal_3",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol="BTC/USD",
                direction=SignalDirection.BUY,
                strength=Decimal("0.3"),  # Below typical threshold
                timestamp=FIXED_TIME,  # Use fixed time for performance
                source="test_strategy",
            ),
        ]

    def test_calculator_initialization(self):
        """Test MetricsCalculator initialization."""
        calculator = MetricsCalculator()

        assert calculator.risk_free_rate == 0.02  # Default
        assert calculator.trading_days_per_year == 252  # Default
        assert calculator._benchmark_returns == []

    def test_calculator_custom_config(self):
        """Test MetricsCalculator with custom configuration."""
        config = {"risk_free_rate": 0.03, "trading_days_per_year": 365}
        calculator = MetricsCalculator(config)

        assert calculator.risk_free_rate == 0.03
        assert calculator.trading_days_per_year == 365

    @pytest.mark.asyncio
    async def test_calculate_comprehensive_metrics_success(
        self, calculator, sample_equity_curve, sample_trades, sample_signals
    ):
        """Test successful comprehensive metrics calculation."""
        metrics = await calculator.calculate_comprehensive_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            signals=sample_signals,
            initial_capital=10000,
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > 0  # Should have positive return
        assert metrics.total_trades == 4  # Number of trades
        assert metrics.total_signals == 3  # Number of signals

    @pytest.mark.asyncio
    async def test_calculate_basic_metrics(self, calculator, sample_equity_curve):
        """Test basic metrics calculation."""
        metrics = await calculator._calculate_basic_metrics(sample_equity_curve, 10000)

        assert "total_return" in metrics
        assert "annual_return" in metrics
        assert isinstance(metrics["total_return"], Decimal)
        assert isinstance(metrics["annual_return"], Decimal)

    @pytest.mark.asyncio
    async def test_calculate_basic_metrics_empty_curve(self, calculator):
        """Test basic metrics calculation with empty equity curve."""
        metrics = await calculator._calculate_basic_metrics([], 10000)

        assert metrics == {}

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, calculator, sample_equity_curve, sample_trades):
        """Test risk metrics calculation."""
        metrics = await calculator._calculate_risk_metrics(sample_equity_curve, sample_trades)

        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        assert "beta" in metrics

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics_insufficient_data(self, calculator):
        """Test risk metrics calculation with insufficient data."""
        short_curve = [{"equity": 10000, "timestamp": FIXED_TIME}]
        metrics = await calculator._calculate_risk_metrics(short_curve, [])

        assert metrics == {}

    @pytest.mark.asyncio
    async def test_calculate_drawdown_metrics(self, calculator, sample_equity_curve):
        """Test drawdown metrics calculation."""
        metrics = await calculator._calculate_drawdown_metrics(sample_equity_curve)

        assert "max_drawdown" in metrics
        assert "current_drawdown" in metrics
        assert "drawdown_duration" in metrics
        assert "avg_drawdown_duration" in metrics
        assert isinstance(metrics["max_drawdown"], Decimal)

    @pytest.mark.asyncio
    async def test_calculate_drawdown_metrics_empty(self, calculator):
        """Test drawdown metrics calculation with empty data."""
        metrics = await calculator._calculate_drawdown_metrics([])

        assert metrics == {}

    @pytest.mark.asyncio
    async def test_calculate_trade_metrics(self, calculator, sample_trades):
        """Test trade metrics calculation."""
        metrics = await calculator._calculate_trade_metrics(sample_trades)

        assert metrics["total_trades"] == 4
        assert metrics["winning_trades"] == 2  # Positive PnL trades
        assert metrics["losing_trades"] == 2  # Negative PnL trades
        assert 0 <= metrics["win_rate"] <= 1
        assert metrics["profit_factor"] > 0

    @pytest.mark.asyncio
    async def test_calculate_trade_metrics_empty(self, calculator):
        """Test trade metrics calculation with no trades."""
        metrics = await calculator._calculate_trade_metrics([])

        assert metrics["total_trades"] == 0
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_signal_metrics(self, calculator, sample_signals, sample_trades):
        """Test signal metrics calculation."""
        metrics = await calculator._calculate_signal_metrics(sample_signals, sample_trades)

        assert metrics["total_signals"] == 3
        assert metrics["valid_signals"] == 2  # Signals with strength >= 0.5
        assert metrics["executed_signals"] == 4  # Number of trades
        assert 0 <= metrics["signal_quality_score"] <= 1

    @pytest.mark.asyncio
    async def test_calculate_signal_metrics_empty(self, calculator):
        """Test signal metrics calculation with no signals."""
        metrics = await calculator._calculate_signal_metrics([], [])

        assert metrics["total_signals"] == 0
        assert metrics["valid_signals"] == 0
        assert metrics["executed_signals"] == 0
        assert metrics["signal_quality_score"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_timing_metrics(self, calculator, sample_trades, sample_signals):
        """Test timing metrics calculation."""
        metrics = await calculator._calculate_timing_metrics(sample_trades, sample_signals)

        assert "avg_trade_duration" in metrics
        assert "avg_signal_latency" in metrics
        assert metrics["avg_trade_duration"] >= 0

    @pytest.mark.asyncio
    async def test_comprehensive_metrics_with_benchmark(
        self, calculator, sample_equity_curve, sample_trades, sample_signals
    ):
        """Test comprehensive metrics calculation with benchmark."""
        benchmark_returns = [0.001, -0.0005, 0.002, -0.001]  # Sample benchmark returns

        metrics = await calculator.calculate_comprehensive_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            signals=sample_signals,
            initial_capital=10000,
            benchmark_returns=benchmark_returns,
        )

        assert isinstance(metrics, PerformanceMetrics)
        # Beta should be calculated when benchmark is provided
        # Note: Actual beta value depends on correlation

    @pytest.mark.asyncio
    async def test_comprehensive_metrics_exception_handling(self, calculator):
        """Test comprehensive metrics handles exceptions gracefully."""
        # Pass invalid data that should cause an exception
        invalid_equity_curve = [{"invalid": "data"}]

        metrics = await calculator.calculate_comprehensive_metrics(
            equity_curve=invalid_equity_curve, trades=[], signals=[], initial_capital=10000
        )

        # Should return empty metrics on error
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == Decimal("0")


class TestRealTimeMetricsTracker:
    """Test RealTimeMetricsTracker functionality."""

    @pytest.fixture(scope="function")
    def tracker(self):
        """Create a RealTimeMetricsTracker instance - fresh for each test."""
        config = {
            "update_interval_seconds": 5,
            "max_equity_points": 20,  # Reduced for performance
            "max_trade_history": 10,  # Reduced for performance
            "max_signal_history": 10,  # Reduced for performance
        }
        return RealTimeMetricsTracker("test_strategy", config)

    @pytest.fixture(scope="class")
    def sample_signal(self):
        """Create a sample signal - cached for class scope."""
        return Signal(
            signal_id="test_signal_4",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=FIXED_TIME,
            source="test_strategy",
        )

    def test_tracker_initialization(self):
        """Test RealTimeMetricsTracker initialization."""
        tracker = RealTimeMetricsTracker("test_strategy")

        assert tracker.strategy_id == "test_strategy"
        assert isinstance(tracker._current_metrics, PerformanceMetrics)
        assert len(tracker._equity_points) == 0
        assert len(tracker._trade_history) == 0
        assert len(tracker._signal_history) == 0

    @pytest.mark.asyncio
    async def test_update_equity(self, tracker):
        """Test updating equity values."""
        await tracker.update_equity(10100)
        await tracker.update_equity(10050)

        assert len(tracker._equity_points) == 2
        assert tracker._equity_points[0]["equity"] == 10100
        assert tracker._equity_points[1]["equity"] == 10050

    @pytest.mark.asyncio
    async def test_update_equity_with_timestamp(self, tracker):
        """Test updating equity with custom timestamp."""
        custom_time = FIXED_TIME - timedelta(hours=1)
        await tracker.update_equity(10100, custom_time)

        assert len(tracker._equity_points) == 1
        assert tracker._equity_points[0]["timestamp"] == custom_time

    @pytest.mark.asyncio
    async def test_update_equity_size_limit(self, tracker):
        """Test equity points size limit enforcement."""
        # Add more points than the limit
        for i in range(25):  # Reduced for performance  # More than max_equity_points (100)
            await tracker.update_equity(10000 + i)

        assert len(tracker._equity_points) == 20  # Should be limited to max_equity_points

    @pytest.mark.asyncio
    async def test_add_trade(self, tracker):
        """Test adding trade data."""
        trade_data = {
            "pnl": 100,
            "entry_time": FIXED_TIME,
            "exit_time": FIXED_TIME + timedelta(hours=1),
        }

        await tracker.add_trade(trade_data)

        assert len(tracker._trade_history) == 1
        assert tracker._trade_history[0]["pnl"] == 100

    @pytest.mark.asyncio
    async def test_add_trade_size_limit(self, tracker):
        """Test trade history size limit enforcement."""
        # Add more trades than the limit
        for i in range(15):  # Reduced for performance  # More than max_trade_history (50)
            trade_data = {"pnl": i, "trade_id": i}
            await tracker.add_trade(trade_data)

        assert len(tracker._trade_history) == 10  # Should be limited to max_trade_history

    @pytest.mark.asyncio
    async def test_add_signal(self, tracker, sample_signal):
        """Test adding signal data."""
        await tracker.add_signal(sample_signal)

        assert len(tracker._signal_history) == 1
        assert tracker._signal_history[0] == sample_signal

    @pytest.mark.asyncio
    async def test_add_signal_size_limit(self, tracker):
        """Test signal history size limit enforcement."""
        # Add more signals than the limit
        for i in range(15):  # Reduced for performance  # More than max_signal_history (50)
            signal = Signal(
                signal_id="test_signal_5",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol="BTC/USD",
                direction=SignalDirection.BUY,
                strength=Decimal("0.8"),
                timestamp=FIXED_TIME,
                source=f"strategy_{i}",
            )
            await tracker.add_signal(signal)

        assert len(tracker._signal_history) == 10  # Should be limited to max_signal_history

    def test_get_current_metrics(self, tracker):
        """Test getting current metrics."""
        metrics = tracker.get_current_metrics()

        assert isinstance(metrics, PerformanceMetrics)

    def test_get_metrics_summary(self, tracker):
        """Test getting metrics summary."""
        summary = tracker.get_metrics_summary()

        assert isinstance(summary, dict)
        assert "strategy_id" in summary
        assert "total_return" in summary
        assert "sharpe_ratio" in summary
        assert "last_update" in summary
        assert summary["strategy_id"] == "test_strategy"

    @pytest.mark.asyncio
    async def test_reset_metrics(self, tracker, sample_signal):
        """Test resetting all metrics."""
        # Add some data first
        await tracker.update_equity(10100)
        await tracker.add_signal(sample_signal)
        await tracker.add_trade({"pnl": 50})

        assert len(tracker._equity_points) > 0
        assert len(tracker._signal_history) > 0
        assert len(tracker._trade_history) > 0

        # Reset
        await tracker.reset_metrics()

        assert len(tracker._equity_points) == 0
        assert len(tracker._signal_history) == 0
        assert len(tracker._trade_history) == 0

    @pytest.mark.asyncio
    async def test_metrics_update_timing(self, tracker):
        """Test that metrics update based on timing interval."""
        # Mock the update method to verify it's called
        tracker._update_metrics = AsyncMock()

        # Set last update to past to trigger update
        tracker._last_update = FIXED_TIME - timedelta(seconds=10)

        await tracker.update_equity(10100)

        # Should trigger metrics update due to interval
        tracker._update_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_metrics_with_data(self, tracker):
        """Test metrics update with real data."""
        # Add some equity points
        base_time = FIXED_TIME
        equity_points = [
            {"timestamp": base_time, "equity": 10000},
            {"timestamp": base_time + timedelta(days=1), "equity": 10100},
        ]

        for point in equity_points:
            tracker._equity_points.append(point)

        # Mock calculator to avoid complex calculations in unit test
        mock_metrics = PerformanceMetrics(total_return=Decimal("1.0"))
        tracker._calculator.calculate_comprehensive_metrics = AsyncMock(return_value=mock_metrics)

        await tracker._update_metrics()

        assert tracker._current_metrics.total_return == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_update_metrics_exception_handling(self, tracker):
        """Test metrics update handles exceptions gracefully."""
        # Add some data
        await tracker.update_equity(10100)

        # Mock calculator to raise exception
        tracker._calculator.calculate_comprehensive_metrics = AsyncMock(
            side_effect=Exception("Test error")
        )

        # Should not raise exception
        await tracker._update_metrics()

        # Metrics should remain at defaults
        assert tracker._current_metrics.total_return == Decimal("0")


class TestStrategyComparator:
    """Test StrategyComparator functionality."""

    @pytest.fixture(scope="class")
    def comparator(self):
        """Create a StrategyComparator instance - cached for class scope."""
        return StrategyComparator()

    @pytest.fixture
    def sample_strategy_metrics(self):
        """Create sample metrics for multiple strategies."""
        return {
            "strategy_1": PerformanceMetrics(
                total_return=Decimal("15.0"),
                sharpe_ratio=1.2,
                max_drawdown=Decimal("5.0"),
                win_rate=0.65,
            ),
            "strategy_2": PerformanceMetrics(
                total_return=Decimal("10.0"),
                sharpe_ratio=1.5,
                max_drawdown=Decimal("3.0"),
                win_rate=0.70,
            ),
            "strategy_3": PerformanceMetrics(
                total_return=Decimal("8.0"),
                sharpe_ratio=0.8,
                max_drawdown=Decimal("7.0"),
                win_rate=0.55,
            ),
        }

    @pytest.mark.asyncio
    async def test_compare_strategies_success(self, comparator, sample_strategy_metrics):
        """Test successful strategy comparison."""
        result = await comparator.compare_strategies(sample_strategy_metrics)

        assert "comparison_timestamp" in result
        assert "strategies_compared" in result
        assert "rankings" in result
        assert "best_overall" in result
        assert result["strategies_compared"] == 3

    @pytest.mark.asyncio
    async def test_compare_strategies_rankings(self, comparator, sample_strategy_metrics):
        """Test strategy rankings are correct."""
        result = await comparator.compare_strategies(sample_strategy_metrics)

        rankings = result["rankings"]

        # Check sharpe ratio ranking
        assert "sharpe_ratio" in rankings
        sharpe_ranking = rankings["sharpe_ratio"]
        assert sharpe_ranking[0]["strategy_id"] == "strategy_2"  # Highest Sharpe
        assert sharpe_ranking[0]["value"] == 1.5

        # Check total return ranking
        assert "total_return" in rankings
        return_ranking = rankings["total_return"]
        assert return_ranking[0]["strategy_id"] == "strategy_1"  # Highest return
        assert return_ranking[0]["value"] == 15.0

        # Check drawdown ranking (lower is better)
        assert "max_drawdown" in rankings
        drawdown_ranking = rankings["max_drawdown"]
        assert drawdown_ranking[0]["strategy_id"] == "strategy_2"  # Lowest drawdown
        assert drawdown_ranking[0]["value"] == 3.0

    @pytest.mark.asyncio
    async def test_compare_strategies_overall_score(self, comparator, sample_strategy_metrics):
        """Test overall score calculation."""
        result = await comparator.compare_strategies(sample_strategy_metrics)

        rankings = result["rankings"]

        # Check overall score ranking exists
        assert "overall_score" in rankings
        overall_ranking = rankings["overall_score"]
        assert len(overall_ranking) == 3

        # All strategies should have scores
        for ranking in overall_ranking:
            assert "strategy_id" in ranking
            assert "value" in ranking
            assert isinstance(ranking["value"], (int, float))

    @pytest.mark.asyncio
    async def test_compare_strategies_best_overall(self, comparator, sample_strategy_metrics):
        """Test best overall strategy identification."""
        result = await comparator.compare_strategies(sample_strategy_metrics)

        best_overall = result["best_overall"]
        assert best_overall in ["strategy_1", "strategy_2", "strategy_3"]

    @pytest.mark.asyncio
    async def test_compare_strategies_insufficient_data(self, comparator):
        """Test comparison with insufficient strategies."""
        single_strategy = {"strategy_1": PerformanceMetrics(total_return=Decimal("10.0"))}

        result = await comparator.compare_strategies(single_strategy)

        assert "error" in result
        assert "Need at least 2 strategies" in result["error"]

    @pytest.mark.asyncio
    async def test_compare_strategies_empty_input(self, comparator):
        """Test comparison with empty input."""
        result = await comparator.compare_strategies({})

        assert "error" in result
        assert "Need at least 2 strategies" in result["error"]

    @pytest.mark.asyncio
    async def test_compare_strategies_edge_case_values(self, comparator):
        """Test comparison with edge case metric values."""
        edge_case_metrics = {
            "strategy_zero": PerformanceMetrics(
                total_return=Decimal("0.0"),
                sharpe_ratio=0.0,
                max_drawdown=Decimal("0.0"),
                win_rate=0.0,
            ),
            "strategy_negative": PerformanceMetrics(
                total_return=Decimal("-5.0"),
                sharpe_ratio=-0.5,
                max_drawdown=Decimal("15.0"),
                win_rate=0.3,
            ),
        }

        result = await comparator.compare_strategies(edge_case_metrics)

        # Should still work with edge case values
        assert "rankings" in result
        assert result["strategies_compared"] == 2


class TestMetricsEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_metrics_calculator_zero_division_handling(self):
        """Test metrics calculator handles zero division gracefully."""
        calculator = MetricsCalculator()

        # Create equity curve with no variance (all same values)
        flat_curve = [
            {"timestamp": FIXED_TIME, "equity": 10000},
            {"timestamp": FIXED_TIME + timedelta(days=1), "equity": 10000},
            {"timestamp": FIXED_TIME + timedelta(days=2), "equity": 10000},
        ]

        metrics = await calculator._calculate_risk_metrics(flat_curve, [])

        # Should handle zero variance without crashing
        assert metrics["volatility"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0

    @pytest.mark.asyncio
    async def test_drawdown_calculation_with_negative_equity(self):
        """Test drawdown calculation with negative equity values."""
        calculator = MetricsCalculator()

        # Create curve that goes negative
        negative_curve = [
            {"timestamp": FIXED_TIME, "equity": 10000},
            {"timestamp": FIXED_TIME + timedelta(days=1), "equity": 5000},
            {"timestamp": FIXED_TIME + timedelta(days=2), "equity": -1000},
        ]

        metrics = await calculator._calculate_drawdown_metrics(negative_curve)

        # Should still calculate drawdown
        assert isinstance(metrics["max_drawdown"], Decimal)
        assert metrics["max_drawdown"] >= 0

    @pytest.mark.asyncio
    async def test_signal_metrics_with_invalid_strengths(self):
        """Test signal metrics with invalid strength values."""
        calculator = MetricsCalculator()

        # Create signals with edge case strengths
        edge_signals = [
            Signal(
                signal_id="test_signal_6",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol="BTC/USD",
                direction=SignalDirection.BUY,
                strength=Decimal("0"),  # Zero strength
                timestamp=FIXED_TIME,
                source="test",
            ),
            Signal(
                signal_id="test_signal_7",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol="BTC/USD",
                direction=SignalDirection.BUY,
                strength=Decimal("1"),  # Maximum strength
                timestamp=FIXED_TIME,
                source="test",
            ),
        ]

        metrics = await calculator._calculate_signal_metrics(edge_signals, [])

        # Should handle edge case strengths
        assert metrics["total_signals"] == 2
        assert 0 <= metrics["signal_quality_score"] <= 1

    def test_performance_metrics_with_extreme_decimals(self):
        """Test PerformanceMetrics with extreme Decimal values."""
        extreme_metrics = PerformanceMetrics(
            total_return=Decimal("999999.999999"),
            max_drawdown=Decimal("0.000001"),
            var_95=Decimal("-999.999"),
        )

        # Should handle extreme values without precision loss
        assert extreme_metrics.total_return == Decimal("999999.999999")
        assert extreme_metrics.max_drawdown == Decimal("0.000001")
        assert extreme_metrics.var_95 == Decimal("-999.999")

    @pytest.mark.asyncio
    async def test_real_time_tracker_concurrent_updates(self):
        """Test real-time tracker handles concurrent updates."""
        tracker = RealTimeMetricsTracker("test", {"update_interval_seconds": 0.1})

        # Simulate concurrent updates
        import asyncio

        tasks = []
        for i in range(5):  # Reduced for performance
            tasks.append(tracker.update_equity(10000 + i))

        await asyncio.gather(*tasks)

        # Should have all equity points
        assert len(tracker._equity_points) == 5

    @pytest.mark.asyncio
    async def test_metrics_with_numpy_array_conversion_errors(self):
        """Test metrics calculation handles numpy conversion errors."""
        calculator = MetricsCalculator()

        # Mock numpy to raise error - performance optimized mock
        with patch("numpy.array", side_effect=ValueError("Mock numpy error")):
            metrics = await calculator._calculate_risk_metrics(
                [{"equity": 10000, "timestamp": FIXED_TIME}], []
            )

            # Should return empty dict on numpy errors
            assert metrics == {}
