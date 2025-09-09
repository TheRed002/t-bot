"""
Tests for ML Metrics utilities.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.utils.ml_metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
    calculate_volatility_metrics,
    calculate_volatility_accuracy,
    calculate_directional_volatility_accuracy,
    calculate_volatility_regime_accuracy,
    calculate_trading_metrics,
    calculate_max_drawdown
)


class TestRegressionMetrics:
    """Test regression metrics calculation."""

    def test_calculate_regression_metrics_basic(self):
        """Test basic regression metrics calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "directional_accuracy" in metrics
        assert "mape" in metrics
        
        assert metrics["mae"] > 0
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert metrics["r2"] > 0.9  # Should be high for good predictions
        assert 0 <= metrics["directional_accuracy"] <= 1
        assert metrics["mape"] >= 0

    def test_calculate_regression_metrics_perfect_prediction(self):
        """Test regression metrics with perfect prediction."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["directional_accuracy"] == 1.0
        assert metrics["mape"] == 0.0

    def test_calculate_regression_metrics_single_value(self):
        """Test regression metrics with single value."""
        y_true = np.array([1.0])
        y_pred = np.array([1.1])
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert metrics["directional_accuracy"] == 0.0  # No direction with single value

    def test_calculate_regression_metrics_with_zeros(self):
        """Test regression metrics with zero values."""
        y_true = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
        y_pred = np.array([0.1, 1.1, 0.1, 1.9, 0.1])
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        # Should handle zero values in MAPE calculation
        assert "mape" in metrics
        assert not np.isinf(metrics["mape"])

    def test_calculate_regression_metrics_all_zeros(self):
        """Test regression metrics with all zero true values."""
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.1, 0.3])
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert metrics["mape"] == 0.0  # Should be 0 when no non-zero true values

    def test_calculate_regression_metrics_exception_handling(self):
        """Test regression metrics exception handling."""
        with patch('src.utils.ml_metrics.mean_absolute_error', side_effect=Exception("Test error")):
            y_true = np.array([1.0, 2.0, 3.0])
            y_pred = np.array([1.1, 1.9, 3.1])
            
            metrics = calculate_regression_metrics(y_true, y_pred)
            
            assert metrics["mae"] == float("inf")
            assert metrics["mse"] == float("inf")
            assert metrics["rmse"] == float("inf")
            assert metrics["r2"] == -float("inf")
            assert metrics["directional_accuracy"] == 0.0
            assert metrics["mape"] == float("inf")


class TestClassificationMetrics:
    """Test classification metrics calculation."""

    def test_calculate_classification_metrics_basic(self):
        """Test basic classification metrics calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_calculate_classification_metrics_perfect(self):
        """Test classification metrics with perfect predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = y_true.copy()
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_calculate_classification_metrics_multiclass(self):
        """Test classification metrics with multiple classes."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 2])
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_calculate_classification_metrics_exception_handling(self):
        """Test classification metrics exception handling."""
        with patch('src.utils.ml_metrics.accuracy_score', side_effect=Exception("Test error")):
            y_true = np.array([0, 1, 1])
            y_pred = np.array([1, 1, 0])
            
            metrics = calculate_classification_metrics(y_true, y_pred)
            
            assert metrics["accuracy"] == 0.0
            assert metrics["precision"] == 0.0
            assert metrics["recall"] == 0.0
            assert metrics["f1_score"] == 0.0


class TestVolatilityMetrics:
    """Test volatility-specific metrics."""

    def test_calculate_volatility_accuracy_basic(self):
        """Test volatility accuracy calculation."""
        y_true = np.array([0.1, 0.2, 0.15, 0.25, 0.18])
        y_pred = np.array([0.11, 0.19, 0.14, 0.26, 0.17])
        
        accuracy = calculate_volatility_accuracy(y_true, y_pred, tolerance=0.2)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)

    def test_calculate_volatility_accuracy_perfect(self):
        """Test volatility accuracy with perfect predictions."""
        y_true = np.array([0.1, 0.2, 0.15, 0.25])
        y_pred = y_true.copy()
        
        accuracy = calculate_volatility_accuracy(y_true, y_pred)
        
        assert accuracy == 1.0

    def test_calculate_volatility_accuracy_exception_handling(self):
        """Test volatility accuracy exception handling."""
        with patch('numpy.abs', side_effect=Exception("Test error")):
            y_true = np.array([0.1, 0.2])
            y_pred = np.array([0.11, 0.19])
            
            accuracy = calculate_volatility_accuracy(y_true, y_pred)
            
            assert accuracy == 0.0

    def test_calculate_directional_volatility_accuracy_basic(self):
        """Test directional volatility accuracy."""
        y_true = np.array([0.1, 0.2, 0.15, 0.25, 0.18])
        y_pred = np.array([0.11, 0.19, 0.14, 0.26, 0.17])
        
        accuracy = calculate_directional_volatility_accuracy(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)

    def test_calculate_directional_volatility_accuracy_single_value(self):
        """Test directional volatility accuracy with insufficient data."""
        y_true = np.array([0.1])
        y_pred = np.array([0.11])
        
        accuracy = calculate_directional_volatility_accuracy(y_true, y_pred)
        
        assert accuracy == 0.0

    def test_calculate_directional_volatility_accuracy_exception_handling(self):
        """Test directional volatility accuracy exception handling."""
        with patch('numpy.diff', side_effect=Exception("Test error")):
            y_true = np.array([0.1, 0.2, 0.15])
            y_pred = np.array([0.11, 0.19, 0.14])
            
            accuracy = calculate_directional_volatility_accuracy(y_true, y_pred)
            
            assert accuracy == 0.0

    def test_calculate_volatility_regime_accuracy_basic(self):
        """Test volatility regime accuracy."""
        y_true = np.array([0.05, 0.1, 0.2, 0.3, 0.15, 0.25, 0.08, 0.35, 0.12, 0.22])
        y_pred = np.array([0.06, 0.09, 0.18, 0.28, 0.14, 0.24, 0.09, 0.33, 0.11, 0.21])
        
        accuracy = calculate_volatility_regime_accuracy(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)

    def test_calculate_volatility_regime_accuracy_exception_handling(self):
        """Test volatility regime accuracy exception handling."""
        with patch('numpy.percentile', side_effect=Exception("Test error")):
            y_true = np.array([0.1, 0.2, 0.15])
            y_pred = np.array([0.11, 0.19, 0.14])
            
            accuracy = calculate_volatility_regime_accuracy(y_true, y_pred)
            
            assert accuracy == 0.0

    def test_calculate_volatility_metrics_basic(self):
        """Test comprehensive volatility metrics."""
        y_true = np.array([0.1, 0.2, 0.15, 0.25, 0.18])
        y_pred = np.array([0.11, 0.19, 0.14, 0.26, 0.17])
        
        metrics = calculate_volatility_metrics(y_true, y_pred)
        
        # Should include all regression metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "directional_accuracy" in metrics
        assert "mape" in metrics
        
        # Should include volatility-specific metrics
        assert "volatility_accuracy" in metrics
        assert "directional_volatility_accuracy" in metrics
        assert "volatility_regime_accuracy" in metrics

    def test_calculate_volatility_metrics_exception_handling(self):
        """Test volatility metrics exception handling."""
        with patch('src.utils.ml_metrics.calculate_volatility_accuracy', side_effect=Exception("Test error")):
            y_true = np.array([0.1, 0.2, 0.15])
            y_pred = np.array([0.11, 0.19, 0.14])
            
            metrics = calculate_volatility_metrics(y_true, y_pred)
            
            # Should fallback to regression metrics
            assert "mae" in metrics
            assert "mse" in metrics


class TestTradingMetrics:
    """Test trading-specific metrics."""

    def test_calculate_trading_metrics_basic(self):
        """Test basic trading metrics calculation."""
        y_true = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
        y_pred = np.array([100.5, 101.5, 101.5, 102.5, 104.5])
        
        metrics = calculate_trading_metrics(y_true, y_pred)
        
        assert "directional_accuracy" in metrics
        assert "strategy_return" in metrics
        assert "strategy_sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "hit_rate" in metrics
        assert "avg_win" in metrics
        assert "avg_loss" in metrics
        
        assert 0 <= metrics["directional_accuracy"] <= 1
        assert 0 <= metrics["hit_rate"] <= 1

    def test_calculate_trading_metrics_with_transaction_costs(self):
        """Test trading metrics with transaction costs."""
        y_true = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
        y_pred = np.array([100.5, 101.5, 101.5, 102.5, 104.5])
        
        metrics = calculate_trading_metrics(y_true, y_pred, transaction_cost=0.002)
        
        # Strategy return should be lower with higher transaction costs
        assert "strategy_return" in metrics

    def test_calculate_trading_metrics_with_zero_prices(self):
        """Test trading metrics with zero prices."""
        y_true = np.array([100.0, 0.0, 101.0, 103.0])  # Contains zero
        y_pred = np.array([100.5, 0.5, 101.5, 102.5])
        
        metrics = calculate_trading_metrics(y_true, y_pred)
        
        # Should handle zero prices gracefully
        assert "strategy_return" in metrics
        assert not np.isnan(metrics["strategy_return"])

    def test_calculate_trading_metrics_constant_volatility(self):
        """Test trading metrics with constant volatility."""
        y_true = np.array([100.0, 100.0, 100.0, 100.0])
        y_pred = np.array([100.5, 100.5, 100.5, 100.5])
        
        metrics = calculate_trading_metrics(y_true, y_pred)
        
        # Sharpe ratio should be 0 with zero volatility
        assert metrics["strategy_sharpe"] == 0.0

    def test_calculate_trading_metrics_exception_handling(self):
        """Test trading metrics exception handling."""
        with patch('numpy.sign', side_effect=Exception("Test error")):
            y_true = np.array([100.0, 102.0, 101.0])
            y_pred = np.array([100.5, 101.5, 101.5])
            
            metrics = calculate_trading_metrics(y_true, y_pred)
            
            assert metrics["directional_accuracy"] == 0.0
            assert metrics["strategy_return"] == 0.0
            assert metrics["strategy_sharpe"] == 0.0
            assert metrics["max_drawdown"] == 0.0
            assert metrics["hit_rate"] == 0.0
            assert metrics["avg_win"] == 0.0
            assert metrics["avg_loss"] == 0.0

    def test_calculate_trading_metrics_edge_cases(self):
        """Test trading metrics edge cases."""
        # Test with very small arrays
        y_true = np.array([100.0, 101.0])
        y_pred = np.array([100.0, 101.0])
        
        metrics = calculate_trading_metrics(y_true, y_pred)
        assert "strategy_return" in metrics
        
        # Test with negative returns
        y_true = np.array([100.0, 95.0, 90.0, 85.0])
        y_pred = np.array([100.0, 96.0, 91.0, 86.0])
        
        metrics = calculate_trading_metrics(y_true, y_pred)
        assert "strategy_return" in metrics


class TestMaxDrawdown:
    """Test maximum drawdown calculation."""

    def test_calculate_max_drawdown_basic(self):
        """Test basic max drawdown calculation."""
        cumulative_returns = np.array([0.0, 0.1, 0.05, 0.15, 0.08, 0.20])
        
        max_dd = calculate_max_drawdown(cumulative_returns)
        
        assert max_dd <= 0.0  # Drawdown should be negative or zero
        assert isinstance(max_dd, float)

    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with no drawdown."""
        cumulative_returns = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # Only increases
        
        max_dd = calculate_max_drawdown(cumulative_returns)
        
        assert max_dd == 0.0

    def test_calculate_max_drawdown_empty_array(self):
        """Test max drawdown with empty array."""
        cumulative_returns = np.array([])
        
        max_dd = calculate_max_drawdown(cumulative_returns)
        
        assert max_dd == 0.0

    def test_calculate_max_drawdown_single_value(self):
        """Test max drawdown with single value."""
        cumulative_returns = np.array([0.1])
        
        max_dd = calculate_max_drawdown(cumulative_returns)
        
        assert max_dd == 0.0

    def test_calculate_max_drawdown_large_drawdown(self):
        """Test max drawdown with significant drawdown."""
        cumulative_returns = np.array([0.0, 0.2, 0.1, -0.1, -0.2, 0.1])
        
        max_dd = calculate_max_drawdown(cumulative_returns)
        
        assert max_dd < -0.5  # Should capture the large drawdown

    def test_calculate_max_drawdown_exception_handling(self):
        """Test max drawdown exception handling."""
        # Use an input that will cause numpy operations to fail
        with patch('src.utils.ml_metrics.logger') as mock_logger:
            # Pass an object that will fail when np.maximum.accumulate is called
            bad_input = object()  # This will fail in numpy operations
            
            max_dd = calculate_max_drawdown(bad_input)
            
            assert max_dd == 0.0
            mock_logger.error.assert_called_once()

    def test_calculate_max_drawdown_zero_peak(self):
        """Test max drawdown with zero peak values."""
        cumulative_returns = np.array([0.0, 0.0, -0.1, 0.0])
        
        max_dd = calculate_max_drawdown(cumulative_returns)
        
        # Should handle zero peak without division by zero error
        assert isinstance(max_dd, float)
        assert not np.isnan(max_dd)
        assert not np.isinf(max_dd)


class TestMetricsIntegration:
    """Test integration between different metric functions."""

    def test_all_metrics_consistent_inputs(self):
        """Test all metrics functions work with consistent inputs."""
        y_true = np.array([1.0, 2.0, 1.5, 2.5, 1.8])
        y_pred = np.array([1.1, 1.9, 1.4, 2.6, 1.7])
        
        # Regression metrics
        reg_metrics = calculate_regression_metrics(y_true, y_pred)
        assert len(reg_metrics) > 0
        
        # Classification metrics (convert to binary)
        y_true_binary = (y_true > np.median(y_true)).astype(int)
        y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
        class_metrics = calculate_classification_metrics(y_true_binary, y_pred_binary)
        assert len(class_metrics) > 0
        
        # Volatility metrics
        vol_metrics = calculate_volatility_metrics(y_true, y_pred)
        assert len(vol_metrics) > 0
        
        # Trading metrics (scale up to price-like values)
        y_true_price = y_true * 100
        y_pred_price = y_pred * 100
        trading_metrics = calculate_trading_metrics(y_true_price, y_pred_price)
        assert len(trading_metrics) > 0

    def test_numpy_array_types(self):
        """Test different numpy array types."""
        # Test with different dtypes
        y_true_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_pred_float32 = np.array([1.1, 1.9, 3.1], dtype=np.float32)
        
        metrics = calculate_regression_metrics(y_true_float32, y_pred_float32)
        assert "mae" in metrics
        
        # Test with int arrays
        y_true_int = np.array([1, 2, 3])
        y_pred_int = np.array([1, 2, 3])
        
        metrics = calculate_regression_metrics(y_true_int, y_pred_int)
        assert "mae" in metrics