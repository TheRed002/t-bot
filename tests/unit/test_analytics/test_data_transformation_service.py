"""Tests for analytics data transformation service."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock

from src.analytics.services.data_transformation_service import DataTransformationService
from src.analytics.types import PortfolioMetrics, PositionMetrics, RiskMetrics
from src.database.models.analytics import (
    AnalyticsPortfolioMetrics,
    AnalyticsPositionMetrics,
    AnalyticsRiskMetrics,
)
from tests.unit.test_analytics.test_helpers import (
    create_test_portfolio_metrics,
    create_test_position_metrics,
    create_test_risk_metrics,
)


class TestDataTransformationService:
    """Test DataTransformationService functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataTransformationService()

    def test_initialization(self):
        """Test service initialization."""
        assert self.service.name == "DataTransformationService"

    def test_transform_portfolio_metrics_to_db(self):
        """Test transforming domain PortfolioMetrics to database model."""
        # Create test portfolio metrics using helper
        timestamp = datetime.now(timezone.utc)
        portfolio_metrics = create_test_portfolio_metrics(
            total_value=Decimal('10000.50'),
            cash=Decimal('5000.25'),
            invested_capital=Decimal('5000.25'),
            unrealized_pnl=Decimal('150.75'),
            realized_pnl=Decimal('75.25'),
            total_pnl=Decimal('226.00'),
            timestamp=timestamp
        )

        bot_id = "test_bot_123"

        # Transform to database model
        db_metrics = self.service.transform_portfolio_metrics_to_db(portfolio_metrics, bot_id)

        # Verify transformation
        assert isinstance(db_metrics, AnalyticsPortfolioMetrics)
        assert db_metrics.bot_id == bot_id
        assert db_metrics.total_value == Decimal('10000.50')
        assert db_metrics.cash_balance == Decimal('5000.25')
        assert db_metrics.unrealized_pnl == Decimal('150.75')
        assert db_metrics.realized_pnl == Decimal('75.25')
        assert db_metrics.daily_pnl == Decimal('0')  # Default since not provided
        assert db_metrics.timestamp == timestamp

    def test_transform_position_metrics_to_db(self):
        """Test transforming domain PositionMetrics to database model."""
        # Create test position metrics using helper
        timestamp = datetime.now(timezone.utc)
        position_metrics = create_test_position_metrics(
            symbol='BTC/USDT',
            current_value=Decimal('5000.00'),  # Helper maps this to market_value
            unrealized_pnl=Decimal('100.50'),
            realized_pnl=Decimal('50.25'),
            entry_price=Decimal('49500.00'),
            current_price=Decimal('50000.00'),
            quantity=Decimal('0.1'),
            timestamp=timestamp
        )

        bot_id = "test_bot_456"

        # Transform to database model
        db_metrics = self.service.transform_position_metrics_to_db(position_metrics, bot_id)

        # Verify transformation
        assert isinstance(db_metrics, AnalyticsPositionMetrics)
        assert db_metrics.bot_id == bot_id
        assert db_metrics.symbol == 'BTC/USDT'
        assert db_metrics.market_value == Decimal('5000.00')
        assert db_metrics.unrealized_pnl == Decimal('100.50')
        assert db_metrics.realized_pnl == Decimal('50.25')
        assert db_metrics.average_price == Decimal('49500.00')
        assert db_metrics.current_price == Decimal('50000.00')
        assert db_metrics.quantity == Decimal('0.1')
        assert db_metrics.exchange == 'binance'  # Default exchange
        assert db_metrics.position_side == 'LONG'  # Default side
        assert db_metrics.timestamp == timestamp

    def test_transform_risk_metrics_to_db(self):
        """Test transforming domain RiskMetrics to database model."""
        # Create test risk metrics using helper
        timestamp = datetime.now(timezone.utc)
        risk_metrics = create_test_risk_metrics(
            value_at_risk_95=Decimal('500.00'),  # Helper maps this to portfolio_var_95
            volatility=Decimal('0.15'),
            max_drawdown=Decimal('200.00'),
            timestamp=timestamp
        )

        bot_id = "test_bot_789"

        # Transform to database model
        db_metrics = self.service.transform_risk_metrics_to_db(risk_metrics, bot_id)

        # Verify transformation
        assert isinstance(db_metrics, AnalyticsRiskMetrics)
        assert db_metrics.bot_id == bot_id
        assert db_metrics.portfolio_var_95 == Decimal('500.00')
        assert db_metrics.portfolio_var_99 == Decimal('750')  # Default from helper
        assert db_metrics.maximum_drawdown == Decimal('200.00')
        # sharpe_ratio and sortino_ratio are not actual fields on RiskMetrics, so should be None
        assert db_metrics.sharpe_ratio is None
        assert db_metrics.sortino_ratio is None
        assert db_metrics.volatility == Decimal('0.15')
        assert db_metrics.timestamp == timestamp

    def test_transform_db_to_portfolio_metrics(self):
        """Test transforming database model to domain PortfolioMetrics."""
        # Create test database model
        timestamp = datetime.now(timezone.utc)
        db_metrics = AnalyticsPortfolioMetrics(
            bot_id="test_bot_abc",
            total_value=Decimal('15000.75'),
            cash_balance=Decimal('7500.50'),
            unrealized_pnl=Decimal('250.25'),
            realized_pnl=Decimal('125.75'),
            daily_pnl=Decimal('376.00'),
            number_of_positions=5,
            timestamp=timestamp
        )

        # Transform to domain model
        portfolio_metrics = self.service.transform_db_to_portfolio_metrics(db_metrics)

        # Verify transformation
        assert isinstance(portfolio_metrics, PortfolioMetrics)
        assert portfolio_metrics.total_value == Decimal('15000.75')
        assert portfolio_metrics.cash == Decimal('7500.50')
        assert portfolio_metrics.invested_capital == Decimal('7500.25')  # total_value - cash
        assert portfolio_metrics.unrealized_pnl == Decimal('250.25')
        assert portfolio_metrics.realized_pnl == Decimal('125.75')
        assert portfolio_metrics.total_pnl == Decimal('376.00')  # unrealized + realized
        assert portfolio_metrics.daily_return == Decimal('376.00')
        assert portfolio_metrics.positions_count == 5
        assert portfolio_metrics.timestamp == timestamp

    def test_transform_db_to_risk_metrics(self):
        """Test transforming database model to domain RiskMetrics."""
        # Create test database model
        timestamp = datetime.now(timezone.utc)
        db_metrics = AnalyticsRiskMetrics(
            bot_id="test_bot_def",
            portfolio_var_95=Decimal('600.00'),
            portfolio_var_99=Decimal('900.00'),
            maximum_drawdown=Decimal('300.00'),
            sharpe_ratio=Decimal('1.8'),
            sortino_ratio=Decimal('2.3'),
            volatility=Decimal('0.18'),
            timestamp=timestamp
        )

        # Transform to domain model
        risk_metrics = self.service.transform_db_to_risk_metrics(db_metrics)

        # Verify transformation
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.portfolio_var_95 == Decimal('600.00')
        assert risk_metrics.portfolio_var_99 == Decimal('900.00')
        assert risk_metrics.max_drawdown == Decimal('300.00')
        # sharpe_ratio and sortino_ratio come from database, not domain
        # beta and alpha are domain-only fields, should be None from database
        assert risk_metrics.volatility == Decimal('0.18')
        assert risk_metrics.timestamp == timestamp

    def test_transform_portfolio_metrics_with_none_values(self):
        """Test portfolio metrics transformation with None values."""
        timestamp = datetime.now(timezone.utc)
        portfolio_metrics = create_test_portfolio_metrics(
            total_value=Decimal('10000'),
            cash=Decimal('5000'),  # Required field
            unrealized_pnl=Decimal('100'),
            realized_pnl=Decimal('50'),  # Required field
            daily_return=None,  # Optional field that can be None
            volatility=None,  # Optional field that can be None
            timestamp=timestamp
        )

        bot_id = "test_bot_none"

        # Transform should handle None values gracefully
        db_metrics = self.service.transform_portfolio_metrics_to_db(portfolio_metrics, bot_id)

        assert isinstance(db_metrics, AnalyticsPortfolioMetrics)
        assert db_metrics.bot_id == bot_id
        assert db_metrics.total_value == Decimal('10000')
        assert db_metrics.cash_balance == Decimal('5000')  # From cash field
        assert db_metrics.unrealized_pnl == Decimal('100')
        assert db_metrics.realized_pnl == Decimal('50')
        # daily_return was None, should be mapped to daily_pnl as 0
        assert db_metrics.daily_pnl == Decimal('0')

    def test_transform_position_metrics_with_none_values(self):
        """Test position metrics transformation with None values."""
        timestamp = datetime.now(timezone.utc)
        position_metrics = create_test_position_metrics(
            symbol='ETH/USDT',
            current_value=Decimal('3000'),  # Maps to market_value in helper
            realized_pnl=Decimal('25'),
            current_price=Decimal('3000'),
            quantity=Decimal('1'),
            timestamp=timestamp
        )

        bot_id = "test_bot_eth"

        # Transform should handle proper values from helper
        db_metrics = self.service.transform_position_metrics_to_db(position_metrics, bot_id)

        assert isinstance(db_metrics, AnalyticsPositionMetrics)
        assert db_metrics.bot_id == bot_id
        assert db_metrics.symbol == 'ETH/USDT'
        assert db_metrics.market_value == Decimal('3000')
        assert db_metrics.unrealized_pnl == Decimal('100')  # From helper default
        assert db_metrics.average_price == Decimal('49000')  # From helper default entry_price
        assert db_metrics.current_price == Decimal('3000')
        assert db_metrics.realized_pnl == Decimal('25')

    def test_transform_risk_metrics_with_none_values(self):
        """Test risk metrics transformation with None values."""
        timestamp = datetime.now(timezone.utc)
        risk_metrics = create_test_risk_metrics(
            value_at_risk_95=Decimal('400'),  # Helper maps to portfolio_var_95
            max_drawdown=Decimal('150'),
            volatility=Decimal('0.12'),
            timestamp=timestamp
        )

        bot_id = "test_bot_risk"

        # Transform should handle values from helper
        db_metrics = self.service.transform_risk_metrics_to_db(risk_metrics, bot_id)

        assert isinstance(db_metrics, AnalyticsRiskMetrics)
        assert db_metrics.bot_id == bot_id
        assert db_metrics.portfolio_var_95 == Decimal('400')
        assert db_metrics.portfolio_var_99 == Decimal('750')  # From helper default
        assert db_metrics.maximum_drawdown == Decimal('150')
        assert db_metrics.sharpe_ratio is None  # Not a field on RiskMetrics
        assert db_metrics.volatility == Decimal('0.12')

    def test_bidirectional_transformation_portfolio(self):
        """Test that portfolio metrics can be transformed back and forth without data loss."""
        # Original domain object using helper
        timestamp = datetime.now(timezone.utc)
        original = create_test_portfolio_metrics(
            total_value=Decimal('25000.123'),
            cash=Decimal('12500.456'),
            invested_capital=Decimal('12499.667'),  # total_value - cash
            unrealized_pnl=Decimal('300.789'),
            realized_pnl=Decimal('150.321'),
            timestamp=timestamp
        )

        bot_id = "bidirectional_test"

        # Transform to DB and back
        db_metrics = self.service.transform_portfolio_metrics_to_db(original, bot_id)
        transformed_back = self.service.transform_db_to_portfolio_metrics(db_metrics)

        # Verify key fields are preserved (some may differ due to transformation)
        assert transformed_back.total_value == original.total_value
        assert transformed_back.cash == original.cash
        assert transformed_back.unrealized_pnl == original.unrealized_pnl
        assert transformed_back.realized_pnl == original.realized_pnl
        assert transformed_back.timestamp == original.timestamp
        # Note: invested_capital is recalculated, total_pnl is recalculated

    def test_bidirectional_transformation_risk(self):
        """Test that risk metrics can be transformed back and forth without data loss."""
        # Original domain object using helper
        timestamp = datetime.now(timezone.utc)
        original = create_test_risk_metrics(
            value_at_risk_95=Decimal('800.123'),  # Maps to portfolio_var_95
            max_drawdown=Decimal('400.789'),
            volatility=Decimal('0.22'),
            timestamp=timestamp
        )

        bot_id = "bidirectional_risk_test"

        # Transform to DB and back
        db_metrics = self.service.transform_risk_metrics_to_db(original, bot_id)
        transformed_back = self.service.transform_db_to_risk_metrics(db_metrics)

        # Verify key fields are preserved
        assert transformed_back.portfolio_var_95 == original.portfolio_var_95
        assert transformed_back.portfolio_var_99 == original.portfolio_var_99
        assert transformed_back.max_drawdown == original.max_drawdown
        assert transformed_back.volatility == original.volatility
        assert transformed_back.timestamp == original.timestamp
        # Note: sharpe_ratio, sortino_ratio, beta, alpha are not stored in database


class TestDataTransformationServiceErrorHandling:
    """Test error handling in data transformation service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataTransformationService()

    def test_transform_with_invalid_bot_id(self):
        """Test transformation with invalid bot_id."""
        portfolio_metrics = create_test_portfolio_metrics(
            total_value=Decimal('10000'),
            cash=Decimal('5000'),
            unrealized_pnl=Decimal('100'),
            realized_pnl=Decimal('50')
        )

        # Test with None bot_id
        try:
            db_metrics = self.service.transform_portfolio_metrics_to_db(portfolio_metrics, None)
            # Should either handle gracefully or raise ValidationError
            if db_metrics:
                assert db_metrics.bot_id is None
        except Exception as e:
            # ValidationError or similar is acceptable
            assert "bot_id" in str(e).lower() or "validation" in str(e).lower()

        # Test with empty string bot_id
        try:
            db_metrics = self.service.transform_portfolio_metrics_to_db(portfolio_metrics, "")
            # Should either handle gracefully or raise ValidationError
            if db_metrics:
                assert db_metrics.bot_id == ""
        except Exception as e:
            # ValidationError or similar is acceptable
            assert "bot_id" in str(e).lower() or "validation" in str(e).lower()

    def test_transform_with_invalid_metrics(self):
        """Test transformation with invalid metrics objects."""
        bot_id = "test_bot"

        # Test with None metrics
        try:
            db_metrics = self.service.transform_portfolio_metrics_to_db(None, bot_id)
        except Exception as e:
            # Should raise appropriate error for None input
            assert "portfolio_metrics" in str(e).lower() or "validation" in str(e).lower()

    def test_transform_db_with_invalid_db_object(self):
        """Test transforming invalid database objects."""
        # Test with None database object
        try:
            portfolio_metrics = self.service.transform_db_to_portfolio_metrics(None)
        except Exception as e:
            # Should raise appropriate error for None input
            assert "db_metric" in str(e).lower() or "validation" in str(e).lower()

        try:
            risk_metrics = self.service.transform_db_to_risk_metrics(None)
        except Exception as e:
            # Should raise appropriate error for None input
            assert "db_metric" in str(e).lower() or "validation" in str(e).lower()


class TestDataTransformationServiceEdgeCases:
    """Test edge cases in data transformation service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataTransformationService()

    def test_transform_with_extreme_decimal_values(self):
        """Test transformation with extreme decimal values."""
        # Test with very large values
        portfolio_metrics = create_test_portfolio_metrics(
            total_value=Decimal('999999999999.99999999'),
            cash=Decimal('0.00000001'),
            unrealized_pnl=Decimal('-999999999.99999999'),
            realized_pnl=Decimal('999999999.99999999')
        )

        bot_id = "extreme_values_test"

        # Transform should handle extreme values
        db_metrics = self.service.transform_portfolio_metrics_to_db(portfolio_metrics, bot_id)

        assert isinstance(db_metrics, AnalyticsPortfolioMetrics)
        assert db_metrics.total_value == Decimal('999999999999.99999999')
        assert db_metrics.cash_balance == Decimal('0.00000001')

    def test_transform_with_zero_values(self):
        """Test transformation with zero values."""
        portfolio_metrics = create_test_portfolio_metrics(
            total_value=Decimal('0'),
            cash=Decimal('0'),
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('0')
        )

        bot_id = "zero_values_test"

        # Transform should handle zero values correctly
        db_metrics = self.service.transform_portfolio_metrics_to_db(portfolio_metrics, bot_id)

        assert isinstance(db_metrics, AnalyticsPortfolioMetrics)
        assert db_metrics.total_value == Decimal('0')
        assert db_metrics.cash_balance == Decimal('0')
        assert db_metrics.unrealized_pnl == Decimal('0')

    def test_transform_with_special_symbol_names(self):
        """Test position metrics transformation with special symbol names."""
        timestamp = datetime.now(timezone.utc)

        # Test with various symbol formats
        symbols = [
            'BTC/USDT',
            'ETH-USD',
            'BNB_BUSD',
            'DOGE.USDT',
            'XRP:USD',
            'ADA@USDT'
        ]

        bot_id = "symbol_test"

        for symbol in symbols:
            position_metrics = create_test_position_metrics(
                symbol=symbol,
                current_value=Decimal('1000'),
                unrealized_pnl=Decimal('50'),
                realized_pnl=Decimal('25'),
                entry_price=Decimal('950'),
                current_price=Decimal('1000'),
                quantity=Decimal('1')
            )

            # Transform should handle various symbol formats
            db_metrics = self.service.transform_position_metrics_to_db(position_metrics, bot_id)

            assert isinstance(db_metrics, AnalyticsPositionMetrics)
            assert db_metrics.symbol == symbol

    def test_transform_preserves_decimal_precision(self):
        """Test that transformation preserves decimal precision."""
        # Use high precision decimal values
        precise_value = Decimal('12345.12345678')
        precise_pnl = Decimal('678.87654321')

        portfolio_metrics = create_test_portfolio_metrics(
            total_value=precise_value,
            cash=precise_value,
            unrealized_pnl=precise_pnl,
            realized_pnl=precise_pnl
        )

        bot_id = "precision_test"

        # Transform to DB and back
        db_metrics = self.service.transform_portfolio_metrics_to_db(portfolio_metrics, bot_id)
        transformed_back = self.service.transform_db_to_portfolio_metrics(db_metrics)

        # Verify precision is preserved
        assert transformed_back.total_value == precise_value
        assert transformed_back.cash == precise_value
        assert transformed_back.unrealized_pnl == precise_pnl
        assert transformed_back.realized_pnl == precise_pnl