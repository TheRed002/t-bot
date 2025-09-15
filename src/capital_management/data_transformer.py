"""
Capital Management Data Transformation Utilities.

Simple data transformation for capital management types.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.types import CapitalAllocation, CapitalMetrics


class CapitalDataTransformer:
    """Simple data transformation for capital management module."""

    @staticmethod
    def transform_allocation_to_dict(allocation: CapitalAllocation) -> dict[str, Any]:
        """
        Transform CapitalAllocation to dictionary format.

        Args:
            allocation: CapitalAllocation to transform

        Returns:
            Dict with allocation data
        """
        return {
            "allocation_id": allocation.allocation_id,
            "strategy_id": allocation.strategy_id,
            "symbol": allocation.symbol,
            "allocated_amount": str(allocation.allocated_amount),
            "utilized_amount": str(allocation.utilized_amount),
            "available_amount": str(allocation.available_amount),
            "allocation_percentage": str(allocation.allocation_percentage),
            "target_allocation_pct": str(allocation.target_allocation_pct),
            "min_allocation": str(allocation.min_allocation),
            "max_allocation": str(allocation.max_allocation),
            "last_rebalance": allocation.last_rebalance.isoformat()
            if allocation.last_rebalance
            else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def transform_metrics_to_dict(metrics: CapitalMetrics) -> dict[str, Any]:
        """
        Transform CapitalMetrics to dictionary format.

        Args:
            metrics: Capital metrics to transform

        Returns:
            Dict with metrics data
        """
        return {
            "total_capital": str(metrics.total_capital),
            "allocated_amount": str(metrics.allocated_amount),
            "available_amount": str(metrics.available_amount),
            "total_pnl": str(metrics.total_pnl),
            "realized_pnl": str(metrics.realized_pnl),
            "unrealized_pnl": str(metrics.unrealized_pnl),
            "daily_return": str(metrics.daily_return),
            "weekly_return": str(metrics.weekly_return),
            "monthly_return": str(metrics.monthly_return),
            "yearly_return": str(metrics.yearly_return),
            "total_return": str(metrics.total_return),
            "sharpe_ratio": str(metrics.sharpe_ratio),
            "sortino_ratio": str(metrics.sortino_ratio),
            "calmar_ratio": str(metrics.calmar_ratio),
            "current_drawdown": str(metrics.current_drawdown),
            "max_drawdown": str(metrics.max_drawdown),
            "var_95": str(metrics.var_95),
            "expected_shortfall": str(metrics.expected_shortfall),
            "strategies_active": metrics.strategies_active,
            "positions_open": metrics.positions_open,
            "leverage_used": str(metrics.leverage_used),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate financial precision for decimal values.

        Args:
            data: Data to validate

        Returns:
            Validated data
        """
        # Simple validation - ensure numeric strings are valid
        validated_data = data.copy()

        for key, value in validated_data.items():
            if isinstance(value, str) and key.endswith(
                ("_amount", "_pnl", "_return", "_ratio", "_drawdown")
            ):
                try:
                    float(value)  # Simple validation
                except (ValueError, TypeError):
                    validated_data[key] = "0.0"

        return validated_data
