"""
Unit tests for P-013C Backtesting Metrics Calculator.

Tests cover:
- Return metrics calculation (total, annual returns)
- Risk-adjusted metrics (Sharpe, Sortino ratios)
- Drawdown analysis (max drawdown, recovery)
- Trade statistics (win rate, profit factor)
- Risk metrics (VaR, CVaR, skewness, kurtosis)
- Rolling metrics calculation
- Edge cases and numerical accuracy
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

from src.backtesting.metrics import MetricsCalculator, BacktestMetrics


class TestBacktestMetrics:
    """Test BacktestMetrics container functionality."""
    
    def test_metrics_container_creation(self):
        """Test creating and using metrics container."""
        metrics = BacktestMetrics()
        
        assert len(metrics.metrics) == 0
        
        # Add metrics
        metrics.add("sharpe_ratio", 1.5)
        metrics.add("total_return", Decimal("15.5"))
        
        assert metrics.get("sharpe_ratio") == 1.5
        assert metrics.get("total_return") == Decimal("15.5")
        assert metrics.get("nonexistent", "default") == "default"
        
        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert metrics_dict["sharpe_ratio"] == 1.5
        assert metrics_dict["total_return"] == Decimal("15.5")


class TestMetricsCalculator:
    """Test MetricsCalculator functionality."""
    
    @pytest.fixture
    def calculator(self):
        """Create metrics calculator with standard risk-free rate."""
        return MetricsCalculator(risk_free_rate=0.02)
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve data."""
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(365)]
        
        # Generate realistic equity curve with some volatility
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, 365)  # Daily returns
        equity_values = [10000]  # Starting equity
        
        for ret in returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        equity_curve = [
            {"timestamp": date, "equity": equity}
            for date, equity in zip(dates, equity_values[1:])
        ]
        
        return equity_curve
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        trades = []
        base_time = datetime(2023, 1, 1)
        
        # Mix of winning and losing trades
        pnls = [100, -50, 200, -30, 150, -75, 80, -40, 250, -60]
        
        for i, pnl in enumerate(pnls):
            entry_time = base_time + timedelta(days=i*3)
            exit_time = entry_time + timedelta(hours=6)
            
            trades.append({
                "symbol": "BTC/USD",
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl/10,  # Approximate
                "size": 1000.0,
                "pnl": pnl,
                "side": "buy"
            })
        
        return trades
    
    @pytest.fixture
    def sample_daily_returns(self):
        """Create sample daily returns."""
        np.random.seed(42)
        # 252 trading days with realistic return distribution
        returns = np.random.normal(0.0008, 0.02, 252)
        return returns.tolist()
    
    def test_calculator_initialization(self):
        """Test calculator initialization with custom parameters."""
        calc = MetricsCalculator(risk_free_rate=0.03)
        assert calc.risk_free_rate == 0.03
    
    def test_calculate_all_metrics(self, calculator, sample_equity_curve, 
                                  sample_trades, sample_daily_returns):
        """Test calculation of all metrics together."""
        metrics = calculator.calculate_all(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            daily_returns=sample_daily_returns,
            initial_capital=10000.0
        )
        
        # Verify all expected metrics are present
        expected_metrics = [
            "total_return", "annual_return", "final_equity",
            "volatility", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "recovery_factor",
            "win_rate", "profit_factor", "avg_win", "avg_loss",
            "var_95", "cvar_95", "skewness", "kurtosis"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Verify metric types and ranges
        assert isinstance(metrics["total_return"], Decimal)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert 0 <= metrics["win_rate"] <= 100
        assert metrics["profit_factor"] >= 0
    
    def test_return_metrics_calculation(self, calculator, sample_equity_curve):
        """Test return-based metrics calculation."""
        metrics = calculator._calculate_return_metrics(sample_equity_curve, 10000.0)
        
        assert "total_return" in metrics
        assert "annual_return" in metrics
        assert "final_equity" in metrics
        
        final_equity = sample_equity_curve[-1]["equity"]
        expected_total_return = (final_equity - 10000.0) / 10000.0 * 100
        
        assert abs(float(metrics["total_return"]) - expected_total_return) < 0.01
        assert isinstance(metrics["annual_return"], Decimal)
        assert float(metrics["final_equity"]) == final_equity
    
    def test_return_metrics_empty_data(self, calculator):
        """Test return metrics with empty equity curve."""
        metrics = calculator._calculate_return_metrics([], 10000.0)
        assert metrics == {}
    
    def test_risk_adjusted_metrics_calculation(self, calculator, sample_daily_returns):
        """Test risk-adjusted metrics calculation."""
        metrics = calculator._calculate_risk_adjusted_metrics(sample_daily_returns)
        
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "mean_daily_return" in metrics
        assert "std_daily_return" in metrics
        
        # Verify Sharpe ratio calculation
        returns_array = np.array(sample_daily_returns)
        expected_annual_return = np.mean(returns_array) * 252
        expected_volatility = np.std(returns_array) * np.sqrt(252)
        expected_sharpe = (expected_annual_return - calculator.risk_free_rate) / expected_volatility
        
        assert abs(metrics["sharpe_ratio"] - expected_sharpe) < 0.01
        assert metrics["volatility"] > 0
    
    def test_risk_adjusted_metrics_empty_data(self, calculator):
        """Test risk-adjusted metrics with empty returns."""
        metrics = calculator._calculate_risk_adjusted_metrics([])
        assert metrics == {}
    
    def test_risk_adjusted_metrics_with_nan_values(self, calculator):
        """Test risk-adjusted metrics with NaN values in returns."""
        returns_with_nan = [0.01, np.nan, -0.005, 0.02, np.nan, 0.008]
        metrics = calculator._calculate_risk_adjusted_metrics(returns_with_nan)
        
        # Should handle NaN values gracefully
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert not np.isnan(metrics["volatility"])
        assert not np.isnan(metrics["sharpe_ratio"])
    
    def test_drawdown_metrics_calculation(self, calculator, sample_equity_curve):
        """Test drawdown-related metrics calculation."""
        metrics = calculator._calculate_drawdown_metrics(sample_equity_curve)
        
        assert "max_drawdown" in metrics
        assert "max_drawdown_duration_days" in metrics
        assert "recovery_factor" in metrics
        assert "current_drawdown" in metrics
        
        # Max drawdown should be positive percentage
        assert float(metrics["max_drawdown"]) >= 0
        assert isinstance(metrics["max_drawdown_duration_days"], int)
        assert metrics["max_drawdown_duration_days"] >= 0
    
    def test_drawdown_metrics_empty_data(self, calculator):
        """Test drawdown metrics with empty equity curve."""
        metrics = calculator._calculate_drawdown_metrics([])
        assert metrics == {}
    
    def test_drawdown_calculation_accuracy(self, calculator):
        """Test drawdown calculation with known data."""
        # Create equity curve with known drawdown
        equity_curve = [
            {"timestamp": datetime(2023, 1, 1), "equity": 10000},
            {"timestamp": datetime(2023, 1, 2), "equity": 12000},  # Peak
            {"timestamp": datetime(2023, 1, 3), "equity": 10800},  # -10% drawdown
            {"timestamp": datetime(2023, 1, 4), "equity": 9600},   # -20% drawdown (max)
            {"timestamp": datetime(2023, 1, 5), "equity": 11000},  # Recovery
        ]
        
        metrics = calculator._calculate_drawdown_metrics(equity_curve)
        
        # Max drawdown should be 20%
        expected_max_dd = 20.0
        assert abs(float(metrics["max_drawdown"]) - expected_max_dd) < 0.1
    
    def test_trade_statistics_calculation(self, calculator, sample_trades):
        """Test trade statistics calculation."""
        metrics = calculator._calculate_trade_statistics(sample_trades)
        
        assert "win_rate" in metrics
        assert "avg_win" in metrics
        assert "avg_loss" in metrics
        assert "profit_factor" in metrics
        assert "payoff_ratio" in metrics
        assert "max_consecutive_wins" in metrics
        assert "max_consecutive_losses" in metrics
        assert "largest_win" in metrics
        assert "largest_loss" in metrics
        
        # Verify win rate calculation
        winning_trades = [t for t in sample_trades if t["pnl"] > 0]
        expected_win_rate = len(winning_trades) / len(sample_trades) * 100
        assert abs(metrics["win_rate"] - expected_win_rate) < 0.1
        
        # Verify profit factor
        total_wins = sum(t["pnl"] for t in sample_trades if t["pnl"] > 0)
        total_losses = abs(sum(t["pnl"] for t in sample_trades if t["pnl"] <= 0))
        expected_profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        
        if expected_profit_factor != float("inf"):
            assert abs(metrics["profit_factor"] - expected_profit_factor) < 0.01
        else:
            assert metrics["profit_factor"] == 999.99  # Capped infinity
    
    def test_trade_statistics_empty_data(self, calculator):
        """Test trade statistics with no trades."""
        metrics = calculator._calculate_trade_statistics([])
        
        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0
    
    def test_trade_statistics_all_winners(self, calculator):
        """Test trade statistics with all winning trades."""
        winning_trades = [
            {"pnl": 100, "entry_time": datetime.now(), "exit_time": datetime.now()},
            {"pnl": 150, "entry_time": datetime.now(), "exit_time": datetime.now()},
            {"pnl": 200, "entry_time": datetime.now(), "exit_time": datetime.now()},
        ]
        
        metrics = calculator._calculate_trade_statistics(winning_trades)
        
        assert metrics["win_rate"] == 100.0
        assert metrics["profit_factor"] == 999.99  # Capped infinity (no losses)
        assert float(metrics["avg_win"]) > 0
        assert float(metrics["avg_loss"]) == 0
    
    def test_risk_metrics_calculation(self, calculator, sample_daily_returns):
        """Test risk metrics calculation (VaR, CVaR, etc.)."""
        metrics = calculator._calculate_risk_metrics(sample_daily_returns, 10000.0)
        
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        assert "max_daily_loss" in metrics
        assert "skewness" in metrics
        assert "kurtosis" in metrics
        assert "omega_ratio" in metrics
        
        # VaR should be positive (loss amount)
        assert float(metrics["var_95"]) >= 0
        assert float(metrics["cvar_95"]) >= 0
        assert float(metrics["max_daily_loss"]) >= 0
        
        # Verify VaR calculation
        returns_array = np.array(sample_daily_returns)
        expected_var = abs(np.percentile(returns_array, 5)) * 10000.0
        assert abs(float(metrics["var_95"]) - expected_var) < 1.0
    
    def test_risk_metrics_empty_data(self, calculator):
        """Test risk metrics with empty returns."""
        metrics = calculator._calculate_risk_metrics([], 10000.0)
        assert metrics == {}
    
    def test_risk_metrics_with_extreme_values(self, calculator):
        """Test risk metrics with extreme return values."""
        extreme_returns = [-0.1, -0.08, 0.12, 0.15, -0.05, 0.03, -0.02, 0.08]
        metrics = calculator._calculate_risk_metrics(extreme_returns, 10000.0)
        
        assert "var_95" in metrics
        assert "skewness" in metrics
        assert "kurtosis" in metrics
        
        # Should handle extreme values without errors
        assert not np.isnan(metrics["skewness"])
        assert not np.isnan(metrics["kurtosis"])
    
    def test_rolling_metrics_calculation(self, calculator, sample_equity_curve):
        """Test rolling metrics calculation."""
        rolling_df = calculator.calculate_rolling_metrics(sample_equity_curve, window=30)
        
        assert isinstance(rolling_df, pd.DataFrame)
        assert "rolling_return" in rolling_df.columns
        assert "rolling_volatility" in rolling_df.columns
        assert "rolling_sharpe" in rolling_df.columns
        assert "rolling_drawdown" in rolling_df.columns
        
        # Should have same length as equity curve (minus some initial NaN values)
        assert len(rolling_df) == len(sample_equity_curve)
        
        # First few values might be NaN due to rolling window
        assert not rolling_df["rolling_return"].iloc[-1:].isna().any()
        assert not rolling_df["rolling_volatility"].iloc[-1:].isna().any()
    
    def test_rolling_metrics_empty_data(self, calculator):
        """Test rolling metrics with empty equity curve."""
        rolling_df = calculator.calculate_rolling_metrics([])
        
        assert isinstance(rolling_df, pd.DataFrame)
        assert len(rolling_df) == 0
    
    def test_rolling_metrics_insufficient_data(self, calculator):
        """Test rolling metrics with insufficient data."""
        short_equity_curve = [
            {"timestamp": datetime(2023, 1, 1), "equity": 10000},
            {"timestamp": datetime(2023, 1, 2), "equity": 10100},
        ]
        
        rolling_df = calculator.calculate_rolling_metrics(short_equity_curve, window=30)
        
        # Should still create DataFrame but with NaN values
        assert isinstance(rolling_df, pd.DataFrame)
        assert len(rolling_df) == 2
    
    @patch('src.backtesting.metrics.stats')
    def test_scipy_import_error_handling(self, mock_stats, calculator, sample_daily_returns):
        """Test handling of scipy import errors."""
        mock_stats.skew.side_effect = ImportError("scipy not available")
        mock_stats.kurtosis.side_effect = ImportError("scipy not available")
        
        with pytest.raises(ImportError):
            calculator._calculate_risk_metrics(sample_daily_returns, 10000.0)
    
    def test_numerical_precision_with_decimals(self, calculator):
        """Test numerical precision with Decimal types."""
        # Test with high-precision values
        equity_curve = [
            {"timestamp": datetime(2023, 1, 1), "equity": 10000.123456789},
            {"timestamp": datetime(2023, 1, 2), "equity": 10001.987654321},
        ]
        
        metrics = calculator._calculate_return_metrics(equity_curve, 10000.0)
        
        # Should maintain precision in Decimal fields
        assert isinstance(metrics["total_return"], Decimal)
        assert isinstance(metrics["annual_return"], Decimal)
    
    def test_edge_case_single_data_point(self, calculator):
        """Test handling of single data point scenarios."""
        single_equity = [{"timestamp": datetime(2023, 1, 1), "equity": 10000}]
        single_return = [0.01]
        single_trade = [{
            "pnl": 100,
            "entry_time": datetime.now(),
            "exit_time": datetime.now()
        }]
        
        # Should handle single data points gracefully
        return_metrics = calculator._calculate_return_metrics(single_equity, 10000.0)
        risk_metrics = calculator._calculate_risk_adjusted_metrics(single_return)
        trade_metrics = calculator._calculate_trade_statistics(single_trade)
        
        assert isinstance(return_metrics, dict)
        assert isinstance(risk_metrics, dict)
        assert isinstance(trade_metrics, dict)
    
    def test_zero_division_protection(self, calculator):
        """Test protection against zero division errors."""
        # Create scenario that could cause zero division
        zero_volatility_returns = [0.0] * 100  # No volatility
        
        metrics = calculator._calculate_risk_adjusted_metrics(zero_volatility_returns)
        
        # Should handle zero volatility case
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        # Sharpe ratio should be 0 when volatility is 0
        assert metrics["sharpe_ratio"] == 0
    
    def test_performance_with_large_dataset(self, calculator):
        """Test performance with large dataset."""
        # Create large dataset (5 years of daily data)
        large_equity_curve = []
        large_returns = []
        large_trades = []
        
        start_date = datetime(2019, 1, 1)
        np.random.seed(42)
        
        for i in range(1825):  # 5 years * 365 days
            date = start_date + timedelta(days=i)
            equity = 10000 * (1.1 ** (i / 365))  # Growing equity
            daily_return = np.random.normal(0.0005, 0.01)
            
            large_equity_curve.append({"timestamp": date, "equity": equity})
            large_returns.append(daily_return)
            
            # Add some trades
            if i % 10 == 0:
                large_trades.append({
                    "pnl": np.random.normal(50, 100),
                    "entry_time": date,
                    "exit_time": date + timedelta(hours=6)
                })
        
        import time
        start_time = time.time()
        
        metrics = calculator.calculate_all(
            equity_curve=large_equity_curve,
            trades=large_trades,
            daily_returns=large_returns,
            initial_capital=10000.0
        )
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (5 seconds for 5 years of data)
        assert execution_time < 5.0
        assert len(metrics) > 10  # Should calculate multiple metrics


# Integration test
def test_metrics_integration_with_realistic_data():
    """Integration test with realistic trading data."""
    calculator = MetricsCalculator(risk_free_rate=0.025)
    
    # Create realistic equity curve with drawdowns and recoveries
    start_date = datetime(2023, 1, 1)
    np.random.seed(123)  # For reproducible results
    
    equity_curve = []
    equity = 100000.0
    
    # Simulate 252 trading days
    for i in range(252):
        date = start_date + timedelta(days=i)
        
        # Add market-like volatility and trend
        daily_return = np.random.normal(0.0008, 0.015)  # Slightly positive expected return
        
        # Add some correlation/momentum
        if i > 0 and np.random.random() < 0.3:
            # 30% chance of following previous day's direction
            prev_return = (equity_curve[-1]["equity"] - (equity_curve[-2]["equity"] if i > 1 else 100000)) / (equity_curve[-2]["equity"] if i > 1 else 100000)
            daily_return *= (1 + 0.5 * np.sign(prev_return))
        
        equity *= (1 + daily_return)
        equity_curve.append({"timestamp": date, "equity": equity})
    
    # Create realistic trades
    trades = []
    for i in range(50):  # 50 trades over the year
        entry_time = start_date + timedelta(days=i*5)
        exit_time = entry_time + timedelta(days=2)
        
        # Mix of winners and losers with realistic distribution
        if np.random.random() < 0.6:  # 60% win rate
            pnl = np.random.lognormal(4, 0.5)  # Log-normal distribution for wins
        else:
            pnl = -np.random.lognormal(3.5, 0.4)  # Smaller losses on average
        
        trades.append({
            "symbol": "BTC/USD",
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": pnl,
        })
    
    # Generate daily returns from equity curve
    daily_returns = []
    for i in range(1, len(equity_curve)):
        prev_equity = equity_curve[i-1]["equity"]
        curr_equity = equity_curve[i]["equity"]
        daily_return = (curr_equity - prev_equity) / prev_equity
        daily_returns.append(daily_return)
    
    # Calculate all metrics
    metrics = calculator.calculate_all(
        equity_curve=equity_curve,
        trades=trades,
        daily_returns=daily_returns,
        initial_capital=100000.0
    )
    
    # Verify results are reasonable
    assert 0 <= metrics["win_rate"] <= 100
    assert abs(metrics["sharpe_ratio"]) < 5  # Reasonable Sharpe ratio
    assert float(metrics["max_drawdown"]) >= 0
    assert metrics["profit_factor"] >= 0
    assert len(metrics) >= 15  # Should have calculated many metrics
    
    # Test rolling metrics
    rolling_metrics = calculator.calculate_rolling_metrics(equity_curve, window=30)
    assert len(rolling_metrics) == len(equity_curve)
    assert not rolling_metrics["rolling_sharpe"].iloc[-30:].isna().all()