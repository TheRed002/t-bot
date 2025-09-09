"""
ML Metrics Calculation Utilities.

This module provides common metric calculation functions to eliminate
duplicate metric calculation code across ML models.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal

logger = get_logger(__name__)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate standard regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with regression metrics
    """
    try:
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }

        # Calculate directional accuracy (important for trading)
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
            metrics["directional_accuracy"] = float(directional_accuracy)
        else:
            metrics["directional_accuracy"] = 0.0

        # Calculate mean absolute percentage error
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = (
                np.mean(
                    np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
                )
                * 100
            )
            metrics["mape"] = float(mape)
        else:
            metrics["mape"] = 0.0

        return metrics

    except Exception as e:
        logger.error(f"Failed to calculate regression metrics: {e}")
        return {
            "mae": float("inf"),
            "mse": float("inf"),
            "rmse": float("inf"),
            "r2": -float("inf"),
            "directional_accuracy": 0.0,
            "mape": float("inf"),
        }


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate standard classification metrics.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels

    Returns:
        Dictionary with classification metrics
    """
    try:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    except Exception as e:
        logger.error(f"Failed to calculate classification metrics: {e}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }


def calculate_volatility_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate volatility-specific metrics.

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values

    Returns:
        Dictionary with volatility metrics
    """
    try:
        # Standard regression metrics
        base_metrics = calculate_regression_metrics(y_true, y_pred)

        # Volatility-specific metrics
        volatility_accuracy = calculate_volatility_accuracy(y_true, y_pred)
        directional_volatility_accuracy = calculate_directional_volatility_accuracy(y_true, y_pred)
        regime_accuracy = calculate_volatility_regime_accuracy(y_true, y_pred)

        return {
            **base_metrics,
            "volatility_accuracy": volatility_accuracy,
            "directional_volatility_accuracy": directional_volatility_accuracy,
            "volatility_regime_accuracy": regime_accuracy,
        }

    except Exception as e:
        logger.error(f"Failed to calculate volatility metrics: {e}")
        return calculate_regression_metrics(y_true, y_pred)  # Fallback to base metrics


def calculate_volatility_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.2
) -> float:
    """
    Calculate volatility-specific accuracy metric.

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values
        tolerance: Tolerance for accuracy calculation (default 20%)

    Returns:
        Volatility accuracy score
    """
    try:
        # Use relative accuracy within a tolerance band
        relative_error = np.abs((y_true - y_pred) / (y_true + 1e-8))
        accurate_predictions = (relative_error <= tolerance).sum()
        return float(accurate_predictions / len(y_true))
    except Exception as e:
        logger.error(f"Volatility accuracy calculation failed: {e}")
        return 0.0


def calculate_directional_volatility_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional volatility accuracy (increasing vs decreasing).

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values

    Returns:
        Directional volatility accuracy
    """
    try:
        if len(y_true) < 2:
            return 0.0

        # Calculate volatility changes
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)

        # Check directional agreement
        directional_agreement = (
            (true_changes > 0) & (pred_changes > 0) | (true_changes < 0) & (pred_changes < 0)
        ).sum()

        return float(directional_agreement / len(true_changes))

    except Exception as e:
        logger.error(f"Directional volatility accuracy calculation failed: {e}")
        return 0.0


def calculate_volatility_regime_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy for volatility regime classification.

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values

    Returns:
        Volatility regime accuracy
    """
    try:
        # Define volatility regimes based on percentiles
        low_threshold = np.percentile(y_true, 33)
        high_threshold = np.percentile(y_true, 67)

        # Classify true volatility regimes
        true_regimes = np.zeros(len(y_true))
        true_regimes[y_true > high_threshold] = 2  # High
        true_regimes[y_true <= low_threshold] = 0  # Low
        true_regimes[(y_true > low_threshold) & (y_true <= high_threshold)] = 1  # Medium

        # Classify predicted volatility regimes
        pred_regimes = np.zeros(len(y_pred))
        pred_regimes[y_pred > high_threshold] = 2  # High
        pred_regimes[y_pred <= low_threshold] = 0  # Low
        pred_regimes[(y_pred > low_threshold) & (y_pred <= high_threshold)] = 1  # Medium

        # Calculate regime accuracy
        regime_accuracy = (true_regimes == pred_regimes).sum() / len(y_true)
        return float(regime_accuracy)

    except Exception as e:
        logger.error(f"Volatility regime accuracy calculation failed: {e}")
        return 0.0


def calculate_trading_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, transaction_cost: float = 0.001
) -> dict[str, Any]:
    """
    Calculate trading-specific performance metrics with financial precision.

    Args:
        y_true: True prices/returns
        y_pred: Predicted prices/returns
        transaction_cost: Transaction cost as decimal (0.001 = 0.1%)

    Returns:
        Trading performance metrics
    """
    try:
        # Calculate directional signals
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        # Directional accuracy
        directional_accuracy = np.mean(true_direction == pred_direction)

        # Calculate hypothetical returns from trading signals using Decimal for precision
        returns_decimal = []
        for i in range(1, len(y_true)):
            if y_true[i - 1] != 0:
                current_return = to_decimal(y_true[i]) / to_decimal(y_true[i - 1]) - to_decimal("1")
                returns_decimal.append(current_return)
            else:
                returns_decimal.append(to_decimal("0"))

        # Simple trading strategy: buy if prediction > current, sell if < current
        positions = pred_direction

        # Calculate strategy returns with Decimal precision
        strategy_returns_decimal = []
        for i, ret in enumerate(
            returns_decimal[1:] if len(returns_decimal) > 1 else returns_decimal
        ):
            if i < len(positions):
                strategy_returns_decimal.append(to_decimal(positions[i]) * ret)

        # Apply transaction costs with Decimal precision
        position_changes = np.diff(np.concatenate([[0], positions]))
        strategy_returns_net_decimal = []
        txn_cost_decimal = to_decimal(transaction_cost)

        for i, strategy_ret in enumerate(strategy_returns_decimal):
            if i + 1 < len(position_changes):
                txn_cost = abs(to_decimal(position_changes[i + 1])) * txn_cost_decimal
                strategy_returns_net_decimal.append(strategy_ret - txn_cost)
            else:
                strategy_returns_net_decimal.append(strategy_ret)

        # Convert back to float for numpy operations in metrics
        strategy_returns_net = np.array([float(ret) for ret in strategy_returns_net_decimal])

        # Calculate metrics with financial precision
        total_return = sum(strategy_returns_net_decimal)

        metrics = {
            "directional_accuracy": float(directional_accuracy),
            "strategy_return": float(total_return),
            "strategy_sharpe": (
                float(np.mean(strategy_returns_net) / np.std(strategy_returns_net))
                if np.std(strategy_returns_net) > 0
                else 0.0
            ),
            "max_drawdown": calculate_max_drawdown(np.cumsum(strategy_returns_net)),
            "hit_rate": float(np.mean(strategy_returns_net > 0)),
            "avg_win": (
                float(np.mean(strategy_returns_net[strategy_returns_net > 0]))
                if np.any(strategy_returns_net > 0)
                else 0.0
            ),
            "avg_loss": (
                float(np.mean(strategy_returns_net[strategy_returns_net < 0]))
                if np.any(strategy_returns_net < 0)
                else 0.0
            ),
        }

        return metrics

    except Exception as e:
        logger.error(f"Trading metrics calculation failed: {e}")
        return {
            "directional_accuracy": 0.0,
            "strategy_return": 0.0,
            "strategy_sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative returns.

    Args:
        cumulative_returns: Cumulative returns array

    Returns:
        Maximum drawdown
    """
    try:
        if len(cumulative_returns) == 0:
            return 0.0

        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-8)  # Avoid division by zero
        return float(np.min(drawdown))
    except Exception as e:
        logger.error(f"Max drawdown calculation failed: {e}")
        return 0.0
