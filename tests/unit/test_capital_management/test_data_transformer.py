"""
Tests for Capital Management Data Transformation Utilities.

Simple tests for the simplified CapitalDataTransformer class.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from src.capital_management.data_transformer import CapitalDataTransformer
from src.core.types import CapitalAllocation, CapitalMetrics


class TestCapitalDataTransformer:
    """Test CapitalDataTransformer static methods."""

    @pytest.fixture
    def sample_allocation(self):
        """Create sample CapitalAllocation for testing."""
        return CapitalAllocation(
            allocation_id="alloc-123",
            strategy_id="momentum-strategy",
            symbol="BTC/USDT",
            allocated_amount=Decimal("1000.00"),
            utilized_amount=Decimal("500.00"),
            available_amount=Decimal("500.00"),
            allocation_percentage=Decimal("0.1"),
            target_allocation_pct=Decimal("0.15"),
            min_allocation=Decimal("100.00"),
            max_allocation=Decimal("2000.00"),
            last_rebalance=datetime(2023, 12, 1, 10, 0, 0, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def sample_metrics(self):
        """Create sample CapitalMetrics for testing."""
        return CapitalMetrics(
            total_capital=Decimal("100000.00"),
            allocated_amount=Decimal("75000.00"),
            available_amount=Decimal("25000.00"),
            total_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("3000.00"),
            unrealized_pnl=Decimal("2000.00"),
            daily_return=Decimal("0.01"),
            weekly_return=Decimal("0.05"),
            monthly_return=Decimal("0.20"),
            yearly_return=Decimal("1.50"),
            total_return=Decimal("2.00"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("1.8"),
            calmar_ratio=Decimal("2.0"),
            current_drawdown=Decimal("0.05"),
            max_drawdown=Decimal("0.15"),
            var_95=Decimal("0.02"),
            expected_shortfall=Decimal("0.03"),
            strategies_active=5,
            positions_open=12,
            leverage_used=Decimal("1.5"),
            timestamp=datetime.now(timezone.utc),
        )

    def test_transform_allocation_to_dict_basic(self, sample_allocation):
        """Test basic allocation transformation."""
        result = CapitalDataTransformer.transform_allocation_to_event_data(sample_allocation)

        assert isinstance(result, dict)
        assert result["allocation_id"] == "alloc-123"
        assert result["strategy_id"] == "momentum-strategy"
        assert result["symbol"] == "BTC/USDT"
        assert result["allocated_amount"] == "1000.00"
        assert result["utilized_amount"] == "500.00"
        assert result["available_amount"] == "500.00"
        assert "timestamp" in result

    def test_transform_metrics_to_dict_basic(self, sample_metrics):
        """Test basic metrics transformation."""
        result = CapitalDataTransformer.transform_metrics_to_event_data(sample_metrics)

        assert isinstance(result, dict)
        assert result["total_capital"] == "100000.00"
        assert result["allocated_amount"] == "75000.00"
        assert result["available_amount"] == "25000.00"
        assert result["strategies_active"] == 5
        assert result["positions_open"] == 12
        assert "timestamp" in result

    def test_validate_financial_precision_valid_data(self):
        """Test financial precision validation with valid data."""
        data = {
            "allocated_amount": "1000.50",
            "total_pnl": "150.25",
            "daily_return": "0.015",
            "non_financial": "some_string"
        }

        result = CapitalDataTransformer.validate_financial_precision(data)

        assert isinstance(result, dict)
        assert result["allocated_amount"] == "1000.50"
        assert result["total_pnl"] == "150.25"
        assert result["daily_return"] == "0.015"
        assert result["non_financial"] == "some_string"

    def test_validate_financial_precision_invalid_data(self):
        """Test financial precision validation with invalid data."""
        data = {
            "allocated_amount": "invalid_number",
            "total_pnl": "150.25",
            "non_financial": "some_string"
        }

        result = CapitalDataTransformer.validate_financial_precision(data)

        assert isinstance(result, dict)
        assert result["allocated_amount"] == "invalid_number"  # Keeps original value when conversion fails
        assert result["total_pnl"] == "150.25"
        assert result["non_financial"] == "some_string"

    def test_allocation_transform_basic_fields(self):
        """Test allocation transformation basic fields."""
        allocation = CapitalAllocation(
            allocation_id="alloc-456",
            strategy_id="test-strategy",
            allocated_amount=Decimal("500.00"),
            utilized_amount=Decimal("200.00"),
            available_amount=Decimal("300.00"),
            allocation_percentage=Decimal("0.05"),
            target_allocation_pct=Decimal("0.10"),
            min_allocation=Decimal("50.00"),
            max_allocation=Decimal("1000.00"),
            last_rebalance=datetime.now(timezone.utc),
        )

        result = CapitalDataTransformer.transform_allocation_to_event_data(allocation)

        assert isinstance(result, dict)
        assert result["allocation_id"] == "alloc-456"
        assert result["strategy_id"] == "test-strategy"
        assert result["allocated_amount"] == "500.00"
        assert "timestamp" in result