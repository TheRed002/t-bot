"""
Unit tests for new database query methods.

These tests verify the new query methods added to DatabaseQueries:
- Market data record operations
- Feature record operations
- Data quality record operations
- Data pipeline record operations
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

from src.database.models import (
    DataPipelineRecord,
    DataQualityRecord,
    FeatureRecord,
    MarketDataRecord,
)
from src.database.queries import DatabaseQueries


class TestMarketDataRecordQueries:
    """Test market data record query methods."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.delete = MagicMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest_asyncio.fixture
    async def mock_db_queries(self, mock_session):
        """Create DatabaseQueries instance with properly mocked database."""
        # Create an async context manager for the mock session
        @asynccontextmanager
        async def mock_get_session():
            yield mock_session
        
        # Pass the mock session directly to DatabaseQueries
        queries = DatabaseQueries(mock_session)
        
        # Patch get_async_session in connection module
        with patch('src.database.connection.get_async_session', mock_get_session):
            yield queries

    @pytest.fixture
    def sample_market_data_record(self):
        """Create sample market data record."""
        return MarketDataRecord(
            symbol="BTCUSDT",
            exchange="binance",
            data_timestamp=datetime.now(timezone.utc),
            open_price=50000.0,
            high_price=51000.0,
            low_price=49000.0,
            close_price=50500.0,
            volume=100.0,
            interval="1h",
            source="exchange"
        )

    @pytest.mark.asyncio
    async def test_create_market_data_record(self, mock_db_queries, sample_market_data_record):
        """Test creating a single market data record."""
        mock_db_queries.session.add = MagicMock()

        result = await mock_db_queries.create_market_data_record(sample_market_data_record)

        assert result == sample_market_data_record
        mock_db_queries.session.add.assert_called_once_with(sample_market_data_record)
        mock_db_queries.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_create_market_data_records(self, mock_db_queries, sample_market_data_record):
        """Test bulk creating market data records."""
        records = [sample_market_data_record, sample_market_data_record]
        mock_db_queries.session.add_all = MagicMock()

        result = await mock_db_queries.bulk_create_market_data_records(records)

        assert result == records
        mock_db_queries.session.add_all.assert_called_once_with(records)
        mock_db_queries.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_market_data_records(self, mock_db_queries):
        """Test retrieving market data records."""
        mock_records = [MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        result = await mock_db_queries.get_market_data_records(
            symbol="BTCUSDT",
            exchange="binance"
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_market_data_by_quality(self, mock_db_queries):
        """Test retrieving market data by quality score."""
        mock_records = [MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        result = await mock_db_queries.get_market_data_by_quality(
            min_quality_score=0.8
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_old_market_data(self, mock_db_queries):
        """Test deleting old market data records."""
        cutoff_date = datetime.now(timezone.utc)
        mock_records = [MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        result = await mock_db_queries.delete_old_market_data(cutoff_date)

        assert result == 2
        mock_db_queries.session.execute.assert_called_once()
        assert mock_db_queries.session.delete.call_count == 2
        mock_db_queries.session.commit.assert_called_once()


class TestFeatureRecordQueries:
    """Test feature record query methods."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()
        session.add_all = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest_asyncio.fixture
    async def mock_db_queries(self, mock_session):
        """Create DatabaseQueries instance with properly mocked database."""
        @asynccontextmanager
        async def mock_get_session():
            yield mock_session
        
        # Pass the mock session directly to DatabaseQueries
        queries = DatabaseQueries(mock_session)
        
        # Patch get_async_session in connection module
        with patch('src.database.connection.get_async_session', mock_get_session):
            yield queries

    @pytest.fixture
    def sample_feature_record(self):
        """Create sample feature record."""
        return FeatureRecord(
            symbol="BTCUSDT",
            feature_type="technical",
            feature_name="sma_20",
            calculation_timestamp=datetime.now(timezone.utc),
            feature_value=49500.0,
            confidence_score=0.85,
            calculation_method="standard"
        )

    @pytest.mark.asyncio
    async def test_create_feature_record(self, mock_db_queries, sample_feature_record):
        """Test creating a feature record."""
        result = await mock_db_queries.create_feature_record(sample_feature_record)

        assert result == sample_feature_record
        mock_db_queries.session.add.assert_called_once_with(sample_feature_record)
        mock_db_queries.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_create_feature_records(self, mock_db_queries, sample_feature_record):
        """Test bulk creating feature records."""
        records = [sample_feature_record, sample_feature_record]
        mock_db_queries.session.add_all = MagicMock()

        result = await mock_db_queries.bulk_create_feature_records(records)

        assert result == records
        mock_db_queries.session.add_all.assert_called_once_with(records)
        mock_db_queries.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_feature_records(self, mock_db_queries):
        """Test retrieving feature records."""
        mock_records = [MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        result = await mock_db_queries.get_feature_records(
            symbol="BTCUSDT",
            feature_type="technical"
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()


class TestDataQualityRecordQueries:
    """Test data quality record query methods."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest_asyncio.fixture
    async def mock_db_queries(self, mock_session):
        """Create DatabaseQueries instance with properly mocked database."""
        @asynccontextmanager
        async def mock_get_session():
            yield mock_session
        
        # Pass the mock session directly to DatabaseQueries
        queries = DatabaseQueries(mock_session)
        
        # Patch get_async_session in connection module
        with patch('src.database.connection.get_async_session', mock_get_session):
            yield queries

    @pytest.fixture
    def sample_quality_record(self):
        """Create sample data quality record."""
        return DataQualityRecord(
            symbol="BTCUSDT",
            data_source="exchange",
            quality_check_timestamp=datetime.now(timezone.utc),
            completeness_score=0.95,
            accuracy_score=0.98,
            consistency_score=0.92,
            timeliness_score=0.99,
            overall_score=0.96,
            missing_data_count=5,
            outlier_count=2,
            duplicate_count=0,
            check_type="comprehensive"
        )

    @pytest.mark.asyncio
    async def test_create_data_quality_record(self, mock_db_queries, sample_quality_record):
        """Test creating a data quality record."""
        result = await mock_db_queries.create_data_quality_record(sample_quality_record)

        assert result == sample_quality_record
        mock_db_queries.session.add.assert_called_once_with(sample_quality_record)
        mock_db_queries.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_quality_records(self, mock_db_queries):
        """Test retrieving data quality records."""
        mock_records = [MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        result = await mock_db_queries.get_data_quality_records(
            symbol="BTCUSDT",
            data_source="exchange"
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()


class TestDataPipelineRecordQueries:
    """Test data pipeline record query methods."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest_asyncio.fixture
    async def mock_db_queries(self, mock_session):
        """Create DatabaseQueries instance with properly mocked database."""
        @asynccontextmanager
        async def mock_get_session():
            yield mock_session
        
        # Pass the mock session directly to DatabaseQueries
        queries = DatabaseQueries(mock_session)
        
        # Patch get_async_session in connection module
        with patch('src.database.connection.get_async_session', mock_get_session):
            yield queries

    @pytest.fixture
    def sample_pipeline_record(self):
        """Create sample data pipeline record."""
        return DataPipelineRecord(
            pipeline_name="market_data_ingestion",
            execution_id="exec_001",
            execution_timestamp=datetime.now(timezone.utc),
            status="running",
            stage="started"
        )

    @pytest.mark.asyncio
    async def test_create_data_pipeline_record(self, mock_db_queries, sample_pipeline_record):
        """Test creating a data pipeline record."""
        result = await mock_db_queries.create_data_pipeline_record(sample_pipeline_record)

        assert result == sample_pipeline_record
        mock_db_queries.session.add.assert_called_once_with(sample_pipeline_record)
        mock_db_queries.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_data_pipeline_status(self, mock_db_queries):
        """Test updating data pipeline status."""
        execution_id = "exec_001"
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db_queries.session.execute.return_value = mock_result

        result = await mock_db_queries.update_data_pipeline_status(
            execution_id=execution_id,
            status="completed",
            stage="finished"
        )

        assert result is True
        mock_db_queries.session.execute.assert_called_once()
        mock_db_queries.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_data_pipeline_status_no_rows_affected(self, mock_db_queries):
        """Test updating data pipeline status when no rows are affected."""
        execution_id = "exec_001"
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db_queries.session.execute.return_value = mock_result

        result = await mock_db_queries.update_data_pipeline_status(
            execution_id=execution_id,
            status="completed"
        )

        assert result is False
        mock_db_queries.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_pipeline_records(self, mock_db_queries):
        """Test retrieving data pipeline records."""
        mock_records = [MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        result = await mock_db_queries.get_data_pipeline_records(
            pipeline_name="market_data_ingestion"
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()


class TestQueryErrorHandling:
    """Test error handling in query methods."""

    @pytest.fixture
    def mock_session_with_error(self):
        """Create a mock database session that raises errors."""
        session = MagicMock()
        session.execute = AsyncMock(side_effect=Exception("Database error"))
        session.commit = AsyncMock(side_effect=Exception("Commit error"))
        session.add = MagicMock(side_effect=Exception("Add error"))
        session.add_all = MagicMock(side_effect=Exception("Add error"))
        return session

    @pytest_asyncio.fixture
    async def mock_db_queries_with_error(self, mock_session_with_error):
        """Create DatabaseQueries instance with error-prone mock session."""
        # Create an async context manager for the mock session
        @asynccontextmanager
        async def mock_get_session():
            yield mock_session_with_error
        
        # Pass the mock session directly to DatabaseQueries
        queries = DatabaseQueries(mock_session_with_error)
        
        # Patch get_async_session in connection module
        with patch('src.database.connection.get_async_session', mock_get_session):
            yield queries

    @pytest.mark.asyncio
    async def test_create_market_data_record_error(self, mock_db_queries_with_error):
        """Test error handling when creating market data record fails."""
        record = MarketDataRecord(
            symbol="BTCUSDT",
            exchange="binance",
            data_timestamp=datetime.now(timezone.utc),
            interval="1h",
            source="exchange"
        )

        with pytest.raises(Exception, match="Add error"):
            await mock_db_queries_with_error.create_market_data_record(record)

    @pytest.mark.asyncio
    async def test_bulk_create_market_data_records_error(self, mock_db_queries_with_error):
        """Test error handling when bulk creating market data records fails."""
        records = [
            MarketDataRecord(
                symbol="BTCUSDT",
                exchange="binance",
                data_timestamp=datetime.now(timezone.utc),
                interval="1h",
                source="exchange"
            )
        ]

        with pytest.raises(Exception, match="Add error"):
            await mock_db_queries_with_error.bulk_create_market_data_records(records)

    @pytest.mark.asyncio
    async def test_get_market_data_records_error(self, mock_db_queries_with_error):
        """Test error handling when retrieving market data records fails."""
        with pytest.raises(Exception, match="Database error"):
            await mock_db_queries_with_error.get_market_data_records(
                symbol="BTCUSDT",
                exchange="binance"
            )

    @pytest.mark.asyncio
    async def test_delete_old_market_data_error(self, mock_db_queries_with_error):
        """Test error handling when deleting old market data fails."""
        cutoff_date = datetime.now(timezone.utc)

        with pytest.raises(Exception, match="Database error"):
            await mock_db_queries_with_error.delete_old_market_data(cutoff_date)


class TestQueryParameterHandling:
    """Test query parameter handling and filtering."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest_asyncio.fixture
    async def mock_db_queries(self, mock_session):
        """Create DatabaseQueries instance with properly mocked database."""
        @asynccontextmanager
        async def mock_get_session():
            yield mock_session
        
        # Pass the mock session directly to DatabaseQueries
        queries = DatabaseQueries(mock_session)
        
        # Patch get_async_session in connection module
        with patch('src.database.connection.get_async_session', mock_get_session):
            yield queries

    @pytest.mark.asyncio
    async def test_get_market_data_records_with_filters(self, mock_db_queries):
        """Test retrieving market data records with various filters."""
        mock_records = [MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        # Test with all filters
        result = await mock_db_queries.get_market_data_records(
            symbol="BTCUSDT",
            exchange="binance",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            limit=100
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_feature_records_with_filters(self, mock_db_queries):
        """Test retrieving feature records with various filters."""
        mock_records = [MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        # Test with all filters
        result = await mock_db_queries.get_feature_records(
            symbol="BTCUSDT",
            feature_type="technical",
            feature_name="sma_20",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc)
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_quality_records_with_filters(self, mock_db_queries):
        """Test retrieving data quality records with various filters."""
        mock_records = [MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        # Test with all filters
        result = await mock_db_queries.get_data_quality_records(
            symbol="BTCUSDT",
            data_source="exchange",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc)
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_pipeline_records_with_filters(self, mock_db_queries):
        """Test retrieving data pipeline records with various filters."""
        mock_records = [MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_records
        mock_db_queries.session.execute.return_value = mock_result

        # Test with all filters
        result = await mock_db_queries.get_data_pipeline_records(
            pipeline_name="market_data_ingestion",
            status="running"
        )

        assert result == mock_records
        mock_db_queries.session.execute.assert_called_once()
