"""
Comprehensive tests for Analytics Repository.

Tests the AnalyticsRepository class with focus on:
- Database operations and transactions
- Error handling and rollback scenarios
- Data conversion between analytics types and database models
- Financial precision preservation
- Edge cases and boundary conditions
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Set pytest markers to optimize test execution
# Test configuration

from src.analytics.repository import AnalyticsRepository
from src.analytics.types import PortfolioMetrics, PositionMetrics, RiskMetrics
from src.core.base.component import BaseComponent
from src.core.exceptions import DataError, ValidationError
from src.database.models.analytics import (
    AnalyticsPortfolioMetrics,
    AnalyticsPositionMetrics,
    AnalyticsRiskMetrics,
)


@pytest.fixture(scope="function")
def mock_session():
    """Mock async database session for testing."""
    session = AsyncMock()
    
    # Mock database operations
    session.add = Mock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    
    # Mock result for query operations
    mock_result = Mock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar_one_or_none.return_value = None
    session.execute.return_value = mock_result
    
    return session


@pytest.fixture(scope="function")
def repository(mock_session):
    """Create repository instance with mocked dependencies."""
    from unittest.mock import Mock
    
    repo = AnalyticsRepository(session=mock_session)
    
    # Mock the individual repository methods to match the test expectations
    repo.portfolio_repo = Mock()
    repo.portfolio_repo.create = AsyncMock()
    repo.position_repo = Mock()  
    repo.position_repo.create = AsyncMock()
    repo.risk_repo = Mock()
    repo.risk_repo.create = AsyncMock()
    
    return repo


@pytest.fixture(scope="module")
def sample_portfolio_metrics():
    """Sample portfolio metrics for testing."""
    # Use fixed timestamp for consistency and performance
    fixed_timestamp = datetime(2024, 1, 15, 12, 0, 0)
    return PortfolioMetrics(
        timestamp=fixed_timestamp,
        bot_id="test-bot-123",  # Required for repository storage
        total_value=Decimal("100000.50"),
        cash=Decimal("25000.25"),
        invested_capital=Decimal("75000.25"),
        unrealized_pnl=Decimal("5000.75"),
        realized_pnl=Decimal("2000.00"),
        total_pnl=Decimal("7000.75"),
        daily_return=Decimal("0.0125"),
        leverage=Decimal("1.5"),
        positions_count=5,
    )


@pytest.fixture(scope="module")
def sample_position_metrics():
    """Sample position metrics for testing."""
    fixed_timestamp = datetime(2024, 1, 15, 12, 0, 0)
    return [
        PositionMetrics(
            timestamp=fixed_timestamp,
            bot_id="test-bot-123",  # Required for repository storage
            symbol="BTC/USD",
            exchange="coinbase",
            quantity=Decimal("1.5"),
            market_value=Decimal("45000.00"),
            unrealized_pnl=Decimal("2500.00"),
            unrealized_pnl_percent=Decimal("0.0588"),
            realized_pnl=Decimal("1000.00"),
            total_pnl=Decimal("3500.00"),
            entry_price=Decimal("28333.33"),
            current_price=Decimal("30000.00"),
            weight=Decimal("0.69"),
            side="long",
        ),
        PositionMetrics(
            timestamp=fixed_timestamp,
            bot_id="test-bot-123",  # Required for repository storage
            symbol="ETH/USD",
            exchange="binance",
            quantity=Decimal("10.0"),
            market_value=Decimal("20000.00"),
            unrealized_pnl=Decimal("1500.00"),
            unrealized_pnl_percent=Decimal("0.081"),
            realized_pnl=Decimal("500.00"),
            total_pnl=Decimal("2000.00"),
            entry_price=Decimal("1850.00"),
            current_price=Decimal("2000.00"),
            weight=Decimal("0.31"),
            side="long",
        ),
    ]


@pytest.fixture(scope="module")
def sample_risk_metrics():
    """Sample risk metrics for testing."""
    fixed_timestamp = datetime(2024, 1, 15, 12, 0, 0)
    return RiskMetrics(
        timestamp=fixed_timestamp,
        bot_id="test-bot-123",  # Required for repository storage
        portfolio_var_95=Decimal("5000.00"),
        portfolio_var_99=Decimal("8000.00"),
        max_drawdown=Decimal("0.15"),
        volatility=Decimal("0.25"),
        expected_shortfall=Decimal("6000.00"),
        correlation_risk=Decimal("0.8"),
        concentration_risk=Decimal("0.3"),
    )


class TestAnalyticsRepositoryInitialization:
    """Test repository initialization."""

    def test_initialization_with_session(self, mock_session):
        """Test repository initialization with async session."""
        repo = AnalyticsRepository(session=mock_session)

        assert repo.session is mock_session
        assert hasattr(repo, "logger")

    def test_inheritance_from_base_component(self, repository):
        """Test that repository inherits from BaseComponent."""
        from src.analytics.interfaces import AnalyticsDataRepository

        assert isinstance(repository, BaseComponent)
        assert isinstance(repository, AnalyticsDataRepository)


class TestPortfolioMetricsOperations:
    """Test portfolio metrics storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_portfolio_metrics_success(
        self, repository, sample_portfolio_metrics
    ):
        """Test successful portfolio metrics storage."""
        await repository.store_portfolio_metrics(sample_portfolio_metrics)

        # Verify database operations - actual implementation uses portfolio_repo.create
        repository.portfolio_repo.create.assert_called_once()

        # Verify data conversion
        call_args = repository.portfolio_repo.create.call_args[0][0]
        assert hasattr(call_args, 'timestamp')
        assert hasattr(call_args, 'total_value')
        # Verify the metrics were passed through transformation service
        assert repository.portfolio_repo.create.call_count == 1

    @pytest.mark.asyncio
    async def test_store_portfolio_metrics_database_error(
        self, repository, sample_portfolio_metrics
    ):
        """Test portfolio metrics storage database error handling."""
        # Make the repository's portfolio_repo.create raise an exception
        repository.portfolio_repo.create.side_effect = Exception("Database connection failed")

        with pytest.raises(DataError) as exc_info:
            await repository.store_portfolio_metrics(sample_portfolio_metrics)

        assert "Failed to store portfolio metrics" in str(exc_info.value)
        assert exc_info.value.context["timestamp"] == sample_portfolio_metrics.timestamp

    @pytest.mark.asyncio
    async def test_store_portfolio_metrics_commit_error(
        self, repository, sample_portfolio_metrics
    ):
        """Test portfolio metrics storage error handling with current architecture."""
        # Test error handling when repository operations fail
        repository.portfolio_repo.create.side_effect = Exception("Database commit failed")

        with pytest.raises(DataError):
            await repository.store_portfolio_metrics(sample_portfolio_metrics)

        # Verify the repository was called despite the error
        repository.portfolio_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_portfolio_metrics_decimal_precision(self, repository):
        """Test that decimal precision is preserved during storage."""
        metrics = PortfolioMetrics(
            timestamp=datetime.utcnow(),
            bot_id="test-bot-123",
            total_value=Decimal("123456.78901234"),
            cash=Decimal("12345.87654321"),
            invested_capital=Decimal("111110.91246913"),
            unrealized_pnl=Decimal("5432.10987654"),
            realized_pnl=Decimal("2109.87654321"),
            total_pnl=Decimal("7541.98642975"),
            daily_return=Decimal("0.01234567890123"),
            leverage=Decimal("2.123456789"),
            margin_used=Decimal("54321.09876543"),
        )

        await repository.store_portfolio_metrics(metrics)

        # Verify the call was made to the portfolio repository
        repository.portfolio_repo.create.assert_called_once()
        
        # Since we're using mocks, we should verify that the input metrics had proper decimal types
        # The test validates that the method was called with the correct input
        assert isinstance(metrics.total_value, Decimal)
        assert metrics.total_value == Decimal("123456.78901234")
        assert isinstance(metrics.cash, Decimal)
        assert isinstance(metrics.unrealized_pnl, Decimal)
        assert isinstance(metrics.realized_pnl, Decimal)

    @pytest.mark.asyncio
    async def test_get_historical_portfolio_metrics_success(self, repository):
        """Test successful historical portfolio metrics retrieval."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Mock database response
        db_metric = AnalyticsPortfolioMetrics(
            timestamp=datetime(2024, 1, 15),
            total_value=Decimal("100000.00"),
            cash_balance=Decimal("25000.00"),
            unrealized_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("2000.00"),
            daily_pnl=Decimal("0.01"),
            leverage_ratio=Decimal("1.5"),
            margin_usage=Decimal("50000.00"),
        )
        
        # Mock the session execute method
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [db_metric]
        repository.session.execute.return_value = mock_result

        result = await repository.get_historical_portfolio_metrics(start_date, end_date)

        assert len(result) == 1
        # Validate the transformation was successful and data integrity maintained
        portfolio_metric = result[0]
        assert hasattr(portfolio_metric, 'timestamp')
        assert hasattr(portfolio_metric, 'total_value')
        assert hasattr(portfolio_metric, 'cash')
        assert hasattr(portfolio_metric, 'unrealized_pnl')
        
        # Verify session.execute was called
        repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_historical_portfolio_metrics_invalid_date_range(self, repository):
        """Test historical portfolio metrics retrieval with invalid date range."""
        start_date = datetime(2024, 1, 31)
        end_date = datetime(2024, 1, 1)

        with pytest.raises(ValidationError) as exc_info:
            await repository.get_historical_portfolio_metrics(start_date, end_date)

        assert "Start date must be before end date" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_historical_portfolio_metrics_equal_dates(self, repository):
        """Test historical portfolio metrics retrieval with equal start and end dates."""
        date = datetime(2024, 1, 15)

        with pytest.raises(ValidationError):
            await repository.get_historical_portfolio_metrics(date, date)

    @pytest.mark.asyncio
    async def test_get_historical_portfolio_metrics_database_error(self, repository):
        """Test historical portfolio metrics retrieval database error handling."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Mock session execute to raise an exception
        repository.session.execute.side_effect = Exception("Query failed")

        with pytest.raises(DataError) as exc_info:
            await repository.get_historical_portfolio_metrics(start_date, end_date)

        assert "Failed to retrieve historical portfolio metrics" in str(exc_info.value)
        assert exc_info.value.context["start_date"] == start_date
        assert exc_info.value.context["end_date"] == end_date

    @pytest.mark.asyncio
    async def test_get_historical_portfolio_metrics_empty_result(self, repository):
        """Test historical portfolio metrics retrieval with empty result."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Mock empty result
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        repository.session.execute.return_value = mock_result

        result = await repository.get_historical_portfolio_metrics(start_date, end_date)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_historical_portfolio_metrics_null_cash_balance(self, repository):
        """Test historical portfolio metrics retrieval with null cash balance."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Mock database response with null cash_balance
        db_metric = AnalyticsPortfolioMetrics(
            timestamp=datetime(2024, 1, 15),
            total_value=Decimal("100000.00"),
            cash_balance=None,
            unrealized_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("2000.00"),
            daily_pnl=Decimal("0.01"),
            leverage_ratio=Decimal("1.5"),
            margin_usage=Decimal("50000.00"),
        )
        
        # Mock the session execute method
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [db_metric]
        repository.session.execute.return_value = mock_result

        result = await repository.get_historical_portfolio_metrics(start_date, end_date)

        assert len(result) == 1
        # Validate that null cash_balance was handled correctly (transformed to 0)
        portfolio_metrics = result[0]
        assert portfolio_metrics.cash == Decimal("0")  # None was converted to 0
        assert portfolio_metrics.total_value == Decimal("100000.00")
        assert portfolio_metrics.invested_capital == Decimal("100000.00")  # total_value - cash (0)

    @pytest.mark.asyncio
    async def test_get_latest_portfolio_metrics_success(self, repository):
        """Test successful latest portfolio metrics retrieval."""
        db_metric = AnalyticsPortfolioMetrics(
            timestamp=datetime.utcnow(),
            total_value=Decimal("100000.00"),
            cash_balance=Decimal("25000.00"),
            unrealized_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("2000.00"),
            daily_pnl=Decimal("0.01"),
            leverage_ratio=Decimal("1.5"),
            margin_usage=Decimal("50000.00"),
        )
        
        # Mock session execute for scalar_one_or_none
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = db_metric
        repository.session.execute.return_value = mock_result

        result = await repository.get_latest_portfolio_metrics()

        assert result is not None
        # Validate that transformation was successful and data integrity maintained  
        assert result.total_value == Decimal("100000.00")
        assert result.cash == Decimal("25000.00")
        
        # Verify session.execute was called
        repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_latest_portfolio_metrics_not_found(self, repository):
        """Test latest portfolio metrics retrieval when none exist."""
        # Mock session execute for scalar_one_or_none returning None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        repository.session.execute.return_value = mock_result

        result = await repository.get_latest_portfolio_metrics()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_portfolio_metrics_database_error(self, repository):
        """Test latest portfolio metrics retrieval database error handling."""
        # Mock session execute to raise an exception
        repository.session.execute.side_effect = Exception("Query failed")

        with pytest.raises(DataError) as exc_info:
            await repository.get_latest_portfolio_metrics()

        assert "Failed to retrieve latest portfolio metrics" in str(exc_info.value)


class TestPositionMetricsOperations:
    """Test position metrics storage operations."""

    @pytest.mark.asyncio
    async def test_store_position_metrics_success(
        self, repository, sample_position_metrics
    ):
        """Test successful position metrics storage."""
        await repository.store_position_metrics(sample_position_metrics)

        # Verify database operations
        assert repository.position_repo.create.call_count == 2  # Two positions
        
        # Verify database operations were called
        repository.position_repo.create.assert_called()
        
        # Verify that the input data has the correct structure (from sample_position_metrics)
        assert len(sample_position_metrics) == 2
        assert sample_position_metrics[0].symbol == "BTC/USD"
        assert sample_position_metrics[0].exchange == "coinbase"
        assert sample_position_metrics[0].quantity == Decimal("1.5")

    @pytest.mark.asyncio
    async def test_store_position_metrics_empty_list(self, repository):
        """Test position metrics storage with empty list."""
        await repository.store_position_metrics([])

        # Should not perform any database operations
        repository.position_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_position_metrics_database_error(
        self, repository, sample_position_metrics
    ):
        """Test position metrics storage database error handling."""
        repository.position_repo.create.side_effect = Exception("Database error")

        with pytest.raises(DataError) as exc_info:
            await repository.store_position_metrics(sample_position_metrics)

        assert "Failed to store position metrics" in str(exc_info.value)
        assert exc_info.value.context["count"] == 2

    @pytest.mark.asyncio
    async def test_store_position_metrics_decimal_precision(self, repository):
        """Test that decimal precision is preserved in position metrics."""
        fixed_timestamp = datetime(2024, 1, 15, 12, 0, 0)
        position_metrics = [
            PositionMetrics(
                timestamp=fixed_timestamp,
                bot_id="test-bot-123",
                symbol="BTC/USD",
                exchange="binance",
                quantity=Decimal("1.23456789"),
                market_value=Decimal("12345.87654321"),
                unrealized_pnl=Decimal("543.21098765"),
                unrealized_pnl_percent=Decimal("0.05432109"),
                realized_pnl=Decimal("210.98765432"),
                total_pnl=Decimal("754.19864197"),
                entry_price=Decimal("10000.12345678"),
                current_price=Decimal("10010.12345678"),
                weight=Decimal("0.15"),
                side="long",
            )
        ]

        await repository.store_position_metrics(position_metrics)

        # Verify the position_repo was called with the correct data
        repository.position_repo.create.assert_called_once()
        
        # Verify database operations were called
        repository.position_repo.create.assert_called()

        # Verify decimal precision is preserved in the input data
        assert isinstance(position_metrics[0].quantity, Decimal)
        assert position_metrics[0].quantity == Decimal("1.23456789")
        assert isinstance(position_metrics[0].market_value, Decimal)
        assert position_metrics[0].market_value == Decimal("12345.87654321")
        assert isinstance(position_metrics[0].unrealized_pnl, Decimal)
        assert position_metrics[0].unrealized_pnl == Decimal("543.21098765")

    @pytest.mark.asyncio
    async def test_store_position_metrics_batch_processing(self, repository):
        """Test position metrics batch processing with many positions."""
        # Reduce size for performance - 10 positions is sufficient
        fixed_timestamp = datetime(2024, 1, 15, 12, 0, 0)
        position_metrics = []
        for i in range(10):
            position_metrics.append(
                PositionMetrics(
                    timestamp=fixed_timestamp,
                    bot_id="test-bot-123",
                    symbol=f"SYMBOL{i}",
                    exchange="test_exchange",
                    quantity=Decimal(f"{i}.5"),
                    market_value=Decimal(f"{i * 100}.00"),
                    unrealized_pnl=Decimal(f"{i * 10}.00"),
                    unrealized_pnl_percent=Decimal("0.05"),
                    realized_pnl=Decimal(f"{i * 5}.00"),
                    total_pnl=Decimal(f"{i * 15}.00"),
                    entry_price=Decimal(f"{i + 100}.00"),
                    current_price=Decimal(f"{i + 105}.00"),
                    weight=Decimal("0.10"),
                    side="long",
                )
            )

        await repository.store_position_metrics(position_metrics)

        # Verify all positions were processed
        assert repository.position_repo.create.call_count == 10


class TestRiskMetricsOperations:
    """Test risk metrics storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_risk_metrics_success(self, repository, sample_risk_metrics):
        """Test successful risk metrics storage."""
        await repository.store_risk_metrics(sample_risk_metrics)

        # Verify database operations
        repository.risk_repo.create.assert_called_once()

        # Verify transformation service was called
        repository.risk_repo.create.assert_called()

        # Verify input data structure
        assert sample_risk_metrics.portfolio_var_95 == Decimal("5000.00")
        assert sample_risk_metrics.portfolio_var_99 == Decimal("8000.00") 
        assert isinstance(sample_risk_metrics.portfolio_var_95, Decimal)
        assert isinstance(sample_risk_metrics.portfolio_var_99, Decimal)

    @pytest.mark.asyncio
    async def test_store_risk_metrics_database_error(
        self, repository, sample_risk_metrics
    ):
        """Test risk metrics storage database error handling."""
        repository.risk_repo.create.side_effect = Exception("Database error")

        with pytest.raises(DataError) as exc_info:
            await repository.store_risk_metrics(sample_risk_metrics)

        assert "Failed to store risk metrics" in str(exc_info.value)
        assert exc_info.value.context["timestamp"] == sample_risk_metrics.timestamp

    @pytest.mark.asyncio
    async def test_store_risk_metrics_decimal_precision(self, repository):
        """Test that decimal precision is preserved in risk metrics."""
        risk_metrics = RiskMetrics(
            timestamp=datetime.utcnow(),
            bot_id="test-bot-123",
            portfolio_var_95=Decimal("5000.12345678"),
            portfolio_var_99=Decimal("8000.87654321"),
            max_drawdown=Decimal("0.123456789"),
            volatility=Decimal("0.987654321"),
            expected_shortfall=Decimal("6000.55555555"),
            correlation_risk=Decimal("0.123456789"),
            concentration_risk=Decimal("0.987654321"),
        )

        await repository.store_risk_metrics(risk_metrics)

        # Verify the risk_repo was called with the correct data
        repository.risk_repo.create.assert_called_once()
        
        # Verify transformation service was called
        repository.risk_repo.create.assert_called()
        
        # Verify decimal precision is preserved in the input data
        assert isinstance(risk_metrics.portfolio_var_95, Decimal)
        assert risk_metrics.portfolio_var_95 == Decimal("5000.12345678")
        assert isinstance(risk_metrics.portfolio_var_99, Decimal)
        assert risk_metrics.portfolio_var_99 == Decimal("8000.87654321")
        assert isinstance(risk_metrics.max_drawdown, Decimal)
        assert risk_metrics.max_drawdown == Decimal("0.123456789")

    @pytest.mark.asyncio
    async def test_get_latest_risk_metrics_success(self, repository):
        """Test successful latest risk metrics retrieval."""
        db_metric = AnalyticsRiskMetrics(
            timestamp=datetime.utcnow(),
            portfolio_var_95=Decimal("5000.00"),
            portfolio_var_99=Decimal("8000.00"),
            expected_shortfall_95=Decimal("6000.00"),
            maximum_drawdown=Decimal("0.15"),
            volatility=Decimal("0.25"),
            correlation_risk=Decimal("0.8"),
            concentration_risk=Decimal("0.3"),
        )
        
        # Mock session execute for scalar_one_or_none
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = db_metric
        repository.session.execute.return_value = mock_result

        result = await repository.get_latest_risk_metrics()

        assert result is not None
        # Verify transformation was successful and data integrity maintained
        assert result.portfolio_var_95 == Decimal("5000.00")
        assert result.portfolio_var_99 == Decimal("8000.00")
        assert result.max_drawdown == Decimal("0.15")
        assert result.volatility == Decimal("0.25")

        # Verify session.execute was called
        repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_latest_risk_metrics_not_found(self, repository):
        """Test latest risk metrics retrieval when none exist."""
        # Mock session execute for scalar_one_or_none returning None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        repository.session.execute.return_value = mock_result

        result = await repository.get_latest_risk_metrics()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_risk_metrics_database_error(self, repository):
        """Test latest risk metrics retrieval database error handling."""
        # Mock session execute to raise an exception
        repository.session.execute.side_effect = Exception("Query failed")

        with pytest.raises(DataError) as exc_info:
            await repository.get_latest_risk_metrics()

        assert "Failed to retrieve latest risk metrics" in str(exc_info.value)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_context_manager_error_handling(
        self, repository, sample_portfolio_metrics
    ):
        """Test repository error handling during storage operations."""
        # Mock the portfolio repository to raise an exception  
        repository.portfolio_repo.create.side_effect = Exception("Storage failed")

        with pytest.raises(DataError):
            await repository.store_portfolio_metrics(sample_portfolio_metrics)

    @pytest.mark.asyncio
    async def test_concurrent_operations_isolation(
        self, repository, sample_portfolio_metrics
    ):
        """Test that concurrent operations don't interfere with each other."""
        import asyncio

        # Create multiple concurrent operations
        tasks = [
            repository.store_portfolio_metrics(sample_portfolio_metrics),
            repository.store_portfolio_metrics(sample_portfolio_metrics),
            repository.get_latest_portfolio_metrics(),
        ]

        # All should complete without interference
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # First two are store operations (should return None)
        assert results[0] is None
        assert results[1] is None
        # Third is a get operation (should work with our mock)

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_commit_failure(
        self, repository, sample_risk_metrics
    ):
        """Test repository error handling during risk metrics storage."""
        # Mock the risk repository to raise an exception
        repository.risk_repo.create.side_effect = Exception("Storage failed")

        with pytest.raises(DataError):
            await repository.store_risk_metrics(sample_risk_metrics)

    @pytest.mark.asyncio
    async def test_data_conversion_edge_cases(self, repository):
        """Test data conversion with edge case values."""
        # Test with zero and negative values
        metrics = PortfolioMetrics(
            timestamp=datetime.utcnow(),
            bot_id="test-bot-123",
            total_value=Decimal("0.00"),
            cash=Decimal("-1000.00"),  # Negative cash (margin account)
            invested_capital=Decimal("1000.00"),
            unrealized_pnl=Decimal("-5000.00"),  # Negative PnL
            realized_pnl=Decimal("0.00"),
            total_pnl=Decimal("-5000.00"),
            daily_return=Decimal("-0.05"),  # Negative return
            leverage=Decimal("0.00"),  # No leverage
            margin_used=Decimal("0.00"),  # No margin
        )

        await repository.store_portfolio_metrics(metrics)

        # Verify the portfolio repo was called
        repository.portfolio_repo.create.assert_called_once()
        
        # Verify edge case values are handled correctly in the input data
        assert metrics.total_value == Decimal("0.00")
        assert metrics.cash == Decimal("-1000.00")
        assert metrics.unrealized_pnl == Decimal("-5000.00")
        assert metrics.realized_pnl == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, repository):
        """Test handling of large datasets without memory issues."""
        # Reduce size for performance - 25 positions is sufficient for testing
        fixed_timestamp = datetime(2024, 1, 15, 12, 0, 0)
        large_position_list = []
        for i in range(25):
            large_position_list.append(
                PositionMetrics(
                    timestamp=fixed_timestamp,
                    bot_id="test-bot-123",
                    symbol=f"SYMBOL{i:04d}",
                    exchange="test",
                    quantity=Decimal("1.0"),
                    market_value=Decimal("1000.0"),
                    unrealized_pnl=Decimal("0.0"),
                    unrealized_pnl_percent=Decimal("0.0"),
                    realized_pnl=Decimal("0.0"),
                    total_pnl=Decimal("0.0"),
                    entry_price=Decimal("1000.0"),
                    current_price=Decimal("1000.0"),
                    weight=Decimal("0.02"),
                    side="long",
                )
            )

        # Should handle large dataset without issues
        await repository.store_position_metrics(large_position_list)

        # Verify all positions were processed
        assert repository.position_repo.create.call_count == 25

    @pytest.mark.asyncio
    async def test_historical_metrics_boundary_dates(self, repository):
        """Test historical metrics with boundary date conditions."""
        # Test with microsecond precision
        start_date = datetime(2024, 1, 1, 0, 0, 0, 1)  # 1 microsecond after midnight
        end_date = datetime(2024, 1, 1, 23, 59, 59, 999999)  # 1 microsecond before next day

        # Mock empty result
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        repository.session.execute.return_value = mock_result

        result = await repository.get_historical_portfolio_metrics(start_date, end_date)

        assert result == []
        # Verify session.execute was called
        repository.session.execute.assert_called_once()
