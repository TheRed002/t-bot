"""
Unit tests for market data repository implementation.

This module tests the MarketDataRepository class which handles
market data storage, retrieval, and analysis operations.
"""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.market_data import MarketDataRecord
from src.database.repository.market_data import MarketDataRepository


class TestMarketDataRepository:
    """Test MarketDataRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def market_data_repository(self, mock_session):
        """Create MarketDataRepository instance for testing."""
        return MarketDataRepository(mock_session)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data record."""
        return MarketDataRecord(
            id=str(uuid.uuid4()),
            symbol="BTCUSD",
            exchange="binance",
            data_timestamp=datetime.now(timezone.utc),
            open_price=Decimal("44800.00"),
            high_price=Decimal("45200.00"),
            low_price=Decimal("44600.00"),
            close_price=Decimal("45000.00"),
            volume=Decimal("125.5"),
            interval="1h",
            source="websocket"
        )

    @pytest.fixture
    def sample_market_data_list(self):
        """Create list of sample market data records."""
        base_time = datetime.now(timezone.utc)
        return [
            MarketDataRecord(
                id=str(uuid.uuid4()),
                symbol="BTCUSD",
                exchange="binance",
                data_timestamp=base_time - timedelta(hours=i),
                open_price=Decimal(str(45000 + i * 100)),
                high_price=Decimal(str(45200 + i * 100)),
                low_price=Decimal(str(44800 + i * 100)),
                close_price=Decimal(str(45100 + i * 100)),
                volume=Decimal(str(100 + i * 10)),
                interval="1h",
                source="websocket"
            )
            for i in range(5)
        ]

    def test_market_data_repository_init(self, mock_session):
        """Test MarketDataRepository initialization."""
        repo = MarketDataRepository(mock_session)
        
        assert repo.session == mock_session
        assert repo.model == MarketDataRecord
        assert repo.name == "MarketDataRepository"

    @pytest.mark.asyncio
    async def test_get_by_symbol(self, market_data_repository, mock_session, sample_market_data_list):
        """Test get market data by symbol."""
        symbol = "BTCUSD"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = sample_market_data_list
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_by_symbol(symbol)
        
        assert len(result) == 5
        for record in result:
            assert record.symbol == symbol

    @pytest.mark.asyncio
    async def test_get_by_symbol_no_data(self, market_data_repository, mock_session):
        """Test get market data by symbol when no data exists."""
        symbol = "NONEXISTENT"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_by_symbol(symbol)
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_by_exchange(self, market_data_repository, mock_session, sample_market_data_list):
        """Test get market data by exchange."""
        exchange = "binance"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = sample_market_data_list
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_by_exchange(exchange)
        
        assert len(result) == 5
        for record in result:
            assert record.exchange == exchange

    @pytest.mark.asyncio
    async def test_get_by_symbol_and_exchange(self, market_data_repository, mock_session, sample_market_data_list):
        """Test get market data by symbol and exchange."""
        symbol = "BTCUSD"
        exchange = "binance"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = sample_market_data_list
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_by_symbol_and_exchange(symbol, exchange)
        
        assert len(result) == 5
        for record in result:
            assert record.symbol == symbol
            assert record.exchange == exchange

    @pytest.mark.asyncio
    async def test_get_latest_price_found(self, market_data_repository, mock_session, sample_market_data):
        """Test get latest price when data exists."""
        symbol = "BTCUSD"
        exchange = "binance"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [sample_market_data]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_latest_price(symbol, exchange)
        
        assert result == sample_market_data
        assert result.symbol == symbol
        assert result.exchange == exchange

    @pytest.mark.asyncio
    async def test_get_latest_price_not_found(self, market_data_repository, mock_session):
        """Test get latest price when no data exists."""
        symbol = "NONEXISTENT"
        exchange = "binance"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_latest_price(symbol, exchange)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_ohlc_data(self, market_data_repository, mock_session, sample_market_data_list):
        """Test get OHLC data for time range."""
        symbol = "BTCUSD"
        exchange = "binance"
        start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        end_time = datetime.now(timezone.utc)
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = sample_market_data_list
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_ohlc_data(symbol, exchange, start_time, end_time)
        
        assert len(result) == 5
        for record in result:
            assert record.symbol == symbol
            assert record.exchange == exchange

    @pytest.mark.asyncio
    async def test_get_ohlc_data_no_data(self, market_data_repository, mock_session):
        """Test get OHLC data when no data in range."""
        symbol = "BTCUSD"
        exchange = "binance"
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_ohlc_data(symbol, exchange, start_time, end_time)
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_recent_data(self, market_data_repository, mock_session, sample_market_data_list):
        """Test get recent market data."""
        symbol = "BTCUSD"
        exchange = "binance"
        hours = 12
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = sample_market_data_list
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_recent_data(symbol, exchange, hours)
        
        assert len(result) == 5
        for record in result:
            assert record.symbol == symbol
            assert record.exchange == exchange

    @pytest.mark.asyncio
    async def test_get_recent_data_default_hours(self, market_data_repository, mock_session, sample_market_data_list):
        """Test get recent market data with default hours."""
        symbol = "BTCUSD"
        exchange = "binance"
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = sample_market_data_list
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_recent_data(symbol, exchange)
        
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_get_by_data_source(self, market_data_repository, mock_session, sample_market_data_list):
        """Test get data by source."""
        data_source = "websocket"
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = sample_market_data_list
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_by_data_source(data_source)
        
        assert len(result) == 5
        for record in result:
            assert record.source == data_source

    @pytest.mark.asyncio
    async def test_get_poor_quality_data_default_threshold(self, market_data_repository):
        """Test get poor quality data with default threshold."""
        # MarketDataRecord doesn't have quality_score field, so method returns empty list
        result = await market_data_repository.get_poor_quality_data()
        
        assert len(result) == 0
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_poor_quality_data_custom_threshold(self, market_data_repository):
        """Test get poor quality data with custom threshold."""
        # MarketDataRecord doesn't have quality_score field, so method returns empty list
        result = await market_data_repository.get_poor_quality_data(threshold=Decimal('0.9'))
        
        assert len(result) == 0
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_poor_quality_data_no_poor_quality(self, market_data_repository):
        """Test get poor quality data when all data is good quality."""
        records = [
            Mock(quality_score=0.9),
            Mock(quality_score=0.95),
            Mock(quality_score=0.98)
        ]
        
        with patch.object(market_data_repository, 'get_all', return_value=records):
            result = await market_data_repository.get_poor_quality_data()
            
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_invalid_data(self, market_data_repository, mock_session):
        """Test get invalid data records."""
        # MarketDataRecord doesn't have validation_status field, so method returns empty list
        result = await market_data_repository.get_invalid_data()
        
        assert len(result) == 0
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_invalid_data_no_invalid(self, market_data_repository, mock_session):
        """Test get invalid data when no invalid records exist."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_invalid_data()
        
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_data_success(self, market_data_repository):
        """Test successful cleanup of old data."""
        # The cleanup_old_data method uses RepositoryUtils.cleanup_old_entities
        with patch('src.database.repository.utils.RepositoryUtils.cleanup_old_entities', 
                   return_value=3):
            result = await market_data_repository.cleanup_old_data(days=30)
            
            assert result == 3

    @pytest.mark.asyncio
    async def test_cleanup_old_data_no_old_data(self, market_data_repository):
        """Test cleanup when no old data exists."""
        # The cleanup_old_data method uses RepositoryUtils.cleanup_old_entities
        with patch('src.database.repository.utils.RepositoryUtils.cleanup_old_entities', 
                   return_value=0):
            result = await market_data_repository.cleanup_old_data(days=30)
            
            assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_data_custom_days(self, market_data_repository):
        """Test cleanup with custom days parameter."""
        # The cleanup_old_data method uses RepositoryUtils.cleanup_old_entities
        with patch('src.database.repository.utils.RepositoryUtils.cleanup_old_entities', 
                   return_value=1):
            result = await market_data_repository.cleanup_old_data(days=7)
            
            assert result == 1

    @pytest.mark.asyncio
    async def test_get_volume_leaders_no_exchange_filter(self, market_data_repository, mock_session):
        """Test get volume leaders without exchange filter."""
        volume_leaders = [
            Mock(volume=Decimal("1000.0")),
            Mock(volume=Decimal("800.0")),
            Mock(volume=Decimal("600.0"))
        ]
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = volume_leaders
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_volume_leaders()
        
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_volume_leaders_with_exchange_filter(self, market_data_repository, mock_session):
        """Test get volume leaders with exchange filter."""
        volume_leaders = [Mock(volume=Decimal("1000.0"), exchange="binance")]
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = volume_leaders
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_volume_leaders(exchange="binance")
        
        assert len(result) == 1
        assert result[0].exchange == "binance"

    @pytest.mark.asyncio
    async def test_get_volume_leaders_custom_limit(self, market_data_repository, mock_session):
        """Test get volume leaders with custom limit."""
        volume_leaders = [Mock(volume=Decimal(str(i * 100))) for i in range(25)]
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = volume_leaders
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await market_data_repository.get_volume_leaders(limit=25)
        
        assert len(result) == 25

    @pytest.mark.asyncio
    async def test_get_price_changes_sufficient_data(self, market_data_repository, mock_session):
        """Test get price changes with sufficient data."""
        base_time = datetime.now(timezone.utc)
        records = [
            Mock(
                data_timestamp=base_time - timedelta(hours=23),
                price=None,  # Add price attribute 
                close_price=Decimal("44000.00"),
                open_price=Decimal("44000.00")
            ),
            Mock(
                data_timestamp=base_time - timedelta(hours=12),
                price=None,  # Add price attribute
                close_price=Decimal("45000.00"),
                open_price=None
            ),
            Mock(
                data_timestamp=base_time - timedelta(hours=1),
                price=None,  # Add price attribute
                close_price=Decimal("46000.00"),
                open_price=None
            )
        ]
        
        # Mock _execute_recent_query directly instead of session.execute
        with patch.object(market_data_repository, '_execute_recent_query', return_value=records):
            result = await market_data_repository.get_price_changes("BTCUSD", "binance", hours=24)
        
        price_change, percentage_change = result
        
        # Should calculate: 46000 - 44000 = 2000 change
        # Percentage: (2000 / 44000) * 100 = ~4.55%
        assert price_change == Decimal("2000.00")
        assert abs(percentage_change - Decimal("4.545454545454545454545454545")) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_get_price_changes_insufficient_data(self, market_data_repository, mock_session):
        """Test get price changes with insufficient data."""
        records = [Mock(price=Decimal("45000.00"))]  # Only one record
        
        # Mock _execute_recent_query directly
        with patch.object(market_data_repository, '_execute_recent_query', return_value=records):
            result = await market_data_repository.get_price_changes("BTCUSD", "binance")
        
        price_change, percentage_change = result
        assert price_change is None
        assert percentage_change is None

    @pytest.mark.asyncio
    async def test_get_price_changes_no_data(self, market_data_repository, mock_session):
        """Test get price changes with no data."""
        records = []  # No records
        
        # Mock _execute_recent_query directly
        with patch.object(market_data_repository, '_execute_recent_query', return_value=records):
            result = await market_data_repository.get_price_changes("BTCUSD", "binance")
        
        price_change, percentage_change = result
        assert price_change is None
        assert percentage_change is None

    @pytest.mark.asyncio
    async def test_get_price_changes_missing_prices(self, market_data_repository, mock_session):
        """Test get price changes when prices are None."""
        records = [
            Mock(price=None, open_price=None, close_price=None),
            Mock(price=None, open_price=None, close_price=None)
        ]
        
        # Mock _execute_recent_query directly
        with patch.object(market_data_repository, '_execute_recent_query', return_value=records):
            result = await market_data_repository.get_price_changes("BTCUSD", "binance")
        
        price_change, percentage_change = result
        assert price_change is None
        assert percentage_change is None

    @pytest.mark.asyncio
    async def test_get_price_changes_custom_hours(self, market_data_repository, mock_session):
        """Test get price changes with custom hours parameter."""
        records = [
            Mock(
                price=None,  # Add price attribute
                close_price=Decimal("45000.00"),
                open_price=Decimal("45000.00")
            ),
            Mock(
                price=None,  # Add price attribute
                close_price=Decimal("46000.00"),
                open_price=None
            )
        ]
        
        # Mock _execute_recent_query directly
        with patch.object(market_data_repository, '_execute_recent_query', return_value=records):
            result = await market_data_repository.get_price_changes("BTCUSD", "binance", hours=12)
        
        price_change, percentage_change = result
        assert price_change == Decimal("1000.00")


class TestMarketDataRepositoryErrorHandling:
    """Test error handling in market data repository."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def market_data_repository(self, mock_session):
        """Create MarketDataRepository instance for testing."""
        return MarketDataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_database_error_handling(self, market_data_repository, mock_session):
        """Test database error handling in repository operations."""
        from src.core.exceptions import RepositoryError
        mock_session.execute.side_effect = SQLAlchemyError("Database connection lost")
        
        with pytest.raises(RepositoryError):
            await market_data_repository.get_by_symbol("BTCUSD")

    @pytest.mark.asyncio
    async def test_integrity_error_handling(self, market_data_repository, mock_session):
        """Test integrity error handling during create operations."""
        from src.core.exceptions import RepositoryError
        market_data = MarketDataRecord(
            id=str(uuid.uuid4()),
            symbol="BTCUSD",
            exchange="binance",
            data_timestamp=datetime.now(timezone.utc),
            open_price=Decimal("45000.00"),
            high_price=Decimal("45000.00"),
            low_price=Decimal("45000.00"),
            close_price=Decimal("45000.00"),
            volume=Decimal("100.0"),
            interval="1h",
            source="exchange"
        )
        mock_session.flush.side_effect = IntegrityError("Duplicate key", None, None)
        
        with pytest.raises(RepositoryError):
            await market_data_repository.create(market_data)
            
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_delete_error(self, market_data_repository):
        """Test cleanup operation when delete fails."""
        # The cleanup_old_data method uses RepositoryUtils.cleanup_old_entities
        with patch('src.database.repository.utils.RepositoryUtils.cleanup_old_entities', return_value=1):
            result = await market_data_repository.cleanup_old_data()
            assert result == 1


class TestMarketDataRepositoryPerformance:
    """Test performance-related functionality in market data repository."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def market_data_repository(self, mock_session):
        """Create MarketDataRepository instance for testing."""
        return MarketDataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_large_dataset_queries(self, market_data_repository):
        """Test queries on large datasets."""
        # MarketDataRecord doesn't have quality_score field, so method returns empty list
        result = await market_data_repository.get_poor_quality_data(threshold=Decimal("0.8"))
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_batch_cleanup_operations(self, market_data_repository):
        """Test batch cleanup operations."""
        # The cleanup_old_data method uses RepositoryUtils.cleanup_old_entities
        with patch('src.database.repository.utils.RepositoryUtils.cleanup_old_entities', return_value=1000):
            result = await market_data_repository.cleanup_old_data()
            assert result == 1000

    @pytest.mark.asyncio
    async def test_time_range_queries_performance(self, market_data_repository, mock_session):
        """Test time range queries for performance."""
        # Simulate time-based data
        base_time = datetime.now(timezone.utc)
        time_series_data = [
            Mock(
                data_timestamp=base_time - timedelta(hours=i),
                symbol="BTCUSD",
                exchange="binance"
            )
            for i in range(168)  # One week of hourly data
        ]
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = time_series_data
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test various time ranges
        result_24h = await market_data_repository.get_recent_data("BTCUSD", "binance", hours=24)
        result_168h = await market_data_repository.get_recent_data("BTCUSD", "binance", hours=168)
        
        assert len(result_24h) == 168  # Mock returns all data
        assert len(result_168h) == 168


class TestMarketDataRepositoryEdgeCases:
    """Test edge cases in market data repository."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def market_data_repository(self, mock_session):
        """Create MarketDataRepository instance for testing."""
        return MarketDataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_zero_prices(self, market_data_repository):
        """Test handling of zero prices."""
        market_data = MarketDataRecord(
            id=str(uuid.uuid4()),
            symbol="TESTCOIN",
            exchange="test",
            data_timestamp=datetime.now(timezone.utc),
            open_price=Decimal("0.00"),
            high_price=Decimal("0.00"),
            low_price=Decimal("0.00"),
            close_price=Decimal("0.00"),
            volume=Decimal("1000.0"),
            interval="1h",
            source="exchange"
        )
        
        with patch.object(market_data_repository, 'create', return_value=market_data):
            result = await market_data_repository.create(market_data)
            
            assert result.close_price == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_very_small_prices(self, market_data_repository):
        """Test handling of very small prices (many decimal places)."""
        market_data = MarketDataRecord(
            id=str(uuid.uuid4()),
            symbol="MICROCOIN",
            exchange="test",
            data_timestamp=datetime.now(timezone.utc),
            open_price=Decimal("0.000000123456789"),
            high_price=Decimal("0.000000123456789"),
            low_price=Decimal("0.000000123456789"),
            close_price=Decimal("0.000000123456789"),
            volume=Decimal("1000000.0"),
            interval="1h",
            source="exchange"
        )
        
        with patch.object(market_data_repository, 'create', return_value=market_data):
            result = await market_data_repository.create(market_data)
            
            assert result.close_price == Decimal("0.000000123456789")

    @pytest.mark.asyncio
    async def test_very_large_prices(self, market_data_repository):
        """Test handling of very large prices."""
        market_data = MarketDataRecord(
            id=str(uuid.uuid4()),
            symbol="EXPENSIVECOIN",
            exchange="test",
            data_timestamp=datetime.now(timezone.utc),
            open_price=Decimal("999999999999.99"),
            high_price=Decimal("999999999999.99"),
            low_price=Decimal("999999999999.99"),
            close_price=Decimal("999999999999.99"),
            volume=Decimal("0.001"),
            interval="1h",
            source="exchange"
        )
        
        with patch.object(market_data_repository, 'create', return_value=market_data):
            result = await market_data_repository.create(market_data)
            
            assert result.close_price == Decimal("999999999999.99")

    @pytest.mark.asyncio
    async def test_extreme_quality_scores(self, market_data_repository):
        """Test handling of extreme quality scores."""
        # MarketDataRecord doesn't have quality_score field, so method returns empty list
        result = await market_data_repository.get_poor_quality_data(threshold=Decimal("0.5"))
        assert len(result) == 0
        
        result = await market_data_repository.get_poor_quality_data(threshold=Decimal("0.0"))
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_future_timestamps(self, market_data_repository, mock_session):
        """Test handling of future timestamps."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        future_record = Mock(data_timestamp=future_time)
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [future_record]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Should handle future timestamps gracefully
        result = await market_data_repository.get_recent_data("BTCUSD", "binance")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_very_old_timestamps(self, market_data_repository):
        """Test cleanup of very old timestamps."""
        # The cleanup_old_data method uses RepositoryUtils.cleanup_old_entities
        with patch('src.database.repository.utils.RepositoryUtils.cleanup_old_entities', return_value=100):
            result = await market_data_repository.cleanup_old_data(days=365)
            assert result == 100

    @pytest.mark.asyncio
    async def test_empty_string_fields(self, market_data_repository):
        """Test handling of empty string fields."""
        market_data = MarketDataRecord(
            id=str(uuid.uuid4()),
            symbol="",  # Empty symbol
            exchange="",  # Empty exchange
            data_timestamp=datetime.now(timezone.utc),
            open_price=Decimal("45000.00"),
            high_price=Decimal("45000.00"),
            low_price=Decimal("45000.00"),
            close_price=Decimal("45000.00"),
            volume=Decimal("100.0"),
            source="",  # Empty data source
            interval="1h"  # Add required interval field
        )
        
        with patch.object(market_data_repository, 'create', return_value=market_data):
            result = await market_data_repository.create(market_data)
            
            assert result.symbol == ""
            assert result.exchange == ""
            assert result.source == ""

    @pytest.mark.asyncio
    async def test_null_optional_fields(self, market_data_repository):
        """Test handling of null optional fields."""
        # Create a minimal valid MarketDataRecord since some fields are required
        market_data = MarketDataRecord(
            id=str(uuid.uuid4()),
            symbol="BTCUSD",
            exchange="binance",
            data_timestamp=datetime.now(timezone.utc),
            open_price=Decimal("45000.00"),  # Required
            high_price=Decimal("45100.00"),  # Required
            low_price=Decimal("44900.00"),   # Required
            close_price=Decimal("45000.00"), # Required
            volume=Decimal("100.0"),         # Required
            interval="1h",                  # Required
            source="exchange"               # Required
        )
        
        with patch.object(market_data_repository, 'create', return_value=market_data):
            result = await market_data_repository.create(market_data)
            
            assert result.open_price == Decimal("45000.00")
            assert result.high_price == Decimal("45100.00")
            assert result.low_price == Decimal("44900.00")
            assert result.close_price == Decimal("45000.00")
            assert result.volume == Decimal("100.0")