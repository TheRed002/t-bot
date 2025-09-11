"""
Tests for DatabaseStorage

Comprehensive test coverage for database storage implementation including:
- Initialization and configuration
- Record storage operations
- Record retrieval with filtering
- Record counting
- Health checks
- Error handling and cleanup
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.config import Config
from src.core.exceptions import DatabaseError, DataError
from src.data.storage.database_storage import DatabaseStorage
from src.data.types import DataRequest
from src.database.models import MarketDataRecord


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def mock_database_service():
    """Create mock database service."""
    service = Mock()
    service.bulk_create = AsyncMock()
    service.list_entities = AsyncMock()
    service.count_entities = AsyncMock()
    service.get_health_status = AsyncMock()
    return service


@pytest.fixture
def sample_records():
    """Create sample market data records."""
    now = datetime.now(timezone.utc)
    return [
        MarketDataRecord(
            symbol="BTC/USD",
            exchange="binance",
            price=Decimal("50000.00"),
            volume=Decimal("100.0"),
            data_timestamp=now,
        ),
        MarketDataRecord(
            symbol="ETH/USD",
            exchange="binance",
            price=Decimal("3000.00"),
            volume=Decimal("200.0"),
            data_timestamp=now,
        ),
    ]


@pytest.fixture
def sample_data_request():
    """Create sample data request."""
    return DataRequest(
        symbol="BTC/USD",
        exchange="binance",
        start_time=datetime.now(timezone.utc),
        limit=100,
    )


class TestDatabaseStorageInitialization:
    """Test database storage initialization."""

    def test_initialization_basic(self, config):
        """Test basic initialization."""
        storage = DatabaseStorage(config)

        assert storage.config == config
        assert storage.database_service is None
        assert storage._initialized is False

    def test_initialization_with_database_service(self, config, mock_database_service):
        """Test initialization with injected database service."""
        storage = DatabaseStorage(config, mock_database_service)

        assert storage.database_service == mock_database_service

    @pytest.mark.asyncio
    async def test_initialize_success(self, config):
        """Test successful initialization."""
        storage = DatabaseStorage(config)

        await storage.initialize()

        assert storage._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, config):
        """Test initialization when already initialized."""
        storage = DatabaseStorage(config)
        storage._initialized = True

        await storage.initialize()  # Should not cause issues

        assert storage._initialized is True


class TestDatabaseStorageStoreRecords:
    """Test record storage operations."""

    @pytest.mark.asyncio
    async def test_store_records_success(self, config, mock_database_service, sample_records):
        """Test successful record storage."""
        storage = DatabaseStorage(config, mock_database_service)

        result = await storage.store_records(sample_records)

        assert result is True
        mock_database_service.bulk_create.assert_called_once_with(sample_records)

    @pytest.mark.asyncio
    async def test_store_records_no_database_service(self, config, sample_records):
        """Test store records without database service."""
        storage = DatabaseStorage(config)

        with pytest.raises(DataError, match="Database service not available"):
            await storage.store_records(sample_records)

    @pytest.mark.asyncio
    async def test_store_records_database_error(self, config, mock_database_service, sample_records):
        """Test store records with database error."""
        mock_database_service.bulk_create.side_effect = DatabaseError("Database connection failed")
        storage = DatabaseStorage(config, mock_database_service)

        with pytest.raises(DataError, match="Database storage failed.*Database connection failed"):
            await storage.store_records(sample_records)

    @pytest.mark.asyncio
    async def test_store_records_generic_exception(self, config, mock_database_service, sample_records):
        """Test store records with generic exception."""
        mock_database_service.bulk_create.side_effect = Exception("Unexpected error")
        storage = DatabaseStorage(config, mock_database_service)

        with pytest.raises(DataError, match="Database storage failed.*Unexpected error"):
            await storage.store_records(sample_records)

    @pytest.mark.asyncio
    async def test_store_records_empty_list(self, config, mock_database_service):
        """Test storing empty record list."""
        storage = DatabaseStorage(config, mock_database_service)

        result = await storage.store_records([])

        assert result is True
        mock_database_service.bulk_create.assert_called_once_with([])


class TestDatabaseStorageRetrieveRecords:
    """Test record retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_records_success(self, config, mock_database_service, sample_data_request, sample_records):
        """Test successful record retrieval."""
        mock_database_service.list_entities.return_value = sample_records
        storage = DatabaseStorage(config, mock_database_service)

        result = await storage.retrieve_records(sample_data_request)

        assert result == sample_records
        mock_database_service.list_entities.assert_called_once()

        # Verify call arguments
        call_args = mock_database_service.list_entities.call_args
        assert call_args.kwargs["model_class"] == MarketDataRecord
        assert call_args.kwargs["order_by"] == "data_timestamp"
        assert call_args.kwargs["order_desc"] is True
        assert call_args.kwargs["limit"] == sample_data_request.limit

    @pytest.mark.asyncio
    async def test_retrieve_records_no_database_service(self, config, sample_data_request):
        """Test retrieve records without database service."""
        storage = DatabaseStorage(config)

        with pytest.raises(DataError, match="Database service not available"):
            await storage.retrieve_records(sample_data_request)

    @pytest.mark.asyncio
    async def test_retrieve_records_with_symbol_filter(self, config, mock_database_service):
        """Test retrieve records with symbol filter."""
        mock_database_service.list_entities.return_value = []
        storage = DatabaseStorage(config, mock_database_service)

        request = DataRequest(symbol="BTC/USD", exchange="binance")
        await storage.retrieve_records(request)

        call_args = mock_database_service.list_entities.call_args
        assert call_args.kwargs["filters"]["symbol"] == "BTC/USD"

    @pytest.mark.asyncio
    async def test_retrieve_records_with_exchange_filter(self, config, mock_database_service):
        """Test retrieve records with exchange filter."""
        mock_database_service.list_entities.return_value = []
        storage = DatabaseStorage(config, mock_database_service)

        request = DataRequest(symbol="BTC/USD", exchange="binance")
        await storage.retrieve_records(request)

        call_args = mock_database_service.list_entities.call_args
        assert call_args.kwargs["filters"]["exchange"] == "binance"

    @pytest.mark.asyncio
    async def test_retrieve_records_with_time_filter(self, config, mock_database_service):
        """Test retrieve records with time filter."""
        mock_database_service.list_entities.return_value = []
        storage = DatabaseStorage(config, mock_database_service)

        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=1)
        request = DataRequest(symbol="BTC/USD", exchange="binance", start_time=start_time, end_time=end_time)

        await storage.retrieve_records(request)

        call_args = mock_database_service.list_entities.call_args
        time_filter = call_args.kwargs["filters"]["data_timestamp"]
        assert time_filter["gte"] == start_time
        assert time_filter["lte"] == end_time

    @pytest.mark.asyncio
    async def test_retrieve_records_with_start_time_only(self, config, mock_database_service):
        """Test retrieve records with only start time."""
        mock_database_service.list_entities.return_value = []
        storage = DatabaseStorage(config, mock_database_service)

        start_time = datetime.now(timezone.utc)
        request = DataRequest(symbol="BTC/USD", exchange="binance", start_time=start_time)

        await storage.retrieve_records(request)

        call_args = mock_database_service.list_entities.call_args
        time_filter = call_args.kwargs["filters"]["data_timestamp"]
        assert time_filter["gte"] == start_time
        assert "lte" not in time_filter

    @pytest.mark.asyncio
    async def test_retrieve_records_with_end_time_only(self, config, mock_database_service):
        """Test retrieve records with only end time."""
        mock_database_service.list_entities.return_value = []
        storage = DatabaseStorage(config, mock_database_service)

        end_time = datetime.now(timezone.utc)
        request = DataRequest(symbol="BTC/USD", exchange="binance", end_time=end_time)

        await storage.retrieve_records(request)

        call_args = mock_database_service.list_entities.call_args
        time_filter = call_args.kwargs["filters"]["data_timestamp"]
        assert time_filter["lte"] == end_time
        assert "gte" not in time_filter

    @pytest.mark.asyncio
    async def test_retrieve_records_no_time_filters(self, config, mock_database_service):
        """Test retrieve records with no time filters."""
        mock_database_service.list_entities.return_value = []
        storage = DatabaseStorage(config, mock_database_service)

        request = DataRequest(symbol="BTC/USD", exchange="binance")
        await storage.retrieve_records(request)

        call_args = mock_database_service.list_entities.call_args
        filters = call_args.kwargs["filters"]
        # Should have symbol and exchange but no timestamp filter
        assert "symbol" in filters
        assert "exchange" in filters
        assert "data_timestamp" not in filters

    @pytest.mark.asyncio
    async def test_retrieve_records_database_error(self, config, mock_database_service, sample_data_request):
        """Test retrieve records with database error."""
        mock_database_service.list_entities.side_effect = DatabaseError("Database query failed")
        storage = DatabaseStorage(config, mock_database_service)

        with pytest.raises(DataError, match="Database retrieval failed.*Database query failed"):
            await storage.retrieve_records(sample_data_request)

    @pytest.mark.asyncio
    async def test_retrieve_records_generic_exception(self, config, mock_database_service, sample_data_request):
        """Test retrieve records with generic exception."""
        mock_database_service.list_entities.side_effect = Exception("Unexpected error")
        storage = DatabaseStorage(config, mock_database_service)

        with pytest.raises(DataError, match="Database retrieval failed.*Unexpected error"):
            await storage.retrieve_records(sample_data_request)


class TestDatabaseStorageRecordCount:
    """Test record counting operations."""

    @pytest.mark.asyncio
    async def test_get_record_count_success(self, config, mock_database_service):
        """Test successful record count retrieval."""
        mock_database_service.count_entities.return_value = 1500
        storage = DatabaseStorage(config, mock_database_service)

        count = await storage.get_record_count("BTC/USD", "binance")

        assert count == 1500
        mock_database_service.count_entities.assert_called_once_with(
            model_class=MarketDataRecord,
            filters={"symbol": "BTC/USD", "exchange": "binance"}
        )

    @pytest.mark.asyncio
    async def test_get_record_count_no_database_service(self, config):
        """Test get record count without database service."""
        storage = DatabaseStorage(config)

        count = await storage.get_record_count("BTC/USD", "binance")

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_record_count_database_error(self, config, mock_database_service):
        """Test get record count with database error."""
        mock_database_service.count_entities.side_effect = DatabaseError("Database count failed")
        storage = DatabaseStorage(config, mock_database_service)

        count = await storage.get_record_count("BTC/USD", "binance")

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_record_count_generic_exception(self, config, mock_database_service):
        """Test get record count with generic exception."""
        mock_database_service.count_entities.side_effect = Exception("Unexpected error")
        storage = DatabaseStorage(config, mock_database_service)

        count = await storage.get_record_count("BTC/USD", "binance")

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_record_count_zero_result(self, config, mock_database_service):
        """Test get record count with zero result."""
        mock_database_service.count_entities.return_value = 0
        storage = DatabaseStorage(config, mock_database_service)

        count = await storage.get_record_count("NONEXISTENT/USD", "binance")

        assert count == 0


class TestDatabaseStorageHealthCheck:
    """Test health check operations."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, config, mock_database_service):
        """Test health check when database is healthy."""
        mock_health_status = Mock()
        mock_health_status.name = "HEALTHY"
        mock_database_service.get_health_status.return_value = mock_health_status

        storage = DatabaseStorage(config, mock_database_service)

        health = await storage.health_check()

        assert health["status"] == "healthy"
        assert health["component"] == "database_storage"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, config, mock_database_service):
        """Test health check when database is unhealthy."""
        mock_health_status = Mock()
        mock_health_status.name = "UNHEALTHY"
        mock_database_service.get_health_status.return_value = mock_health_status

        storage = DatabaseStorage(config, mock_database_service)

        health = await storage.health_check()

        assert health["status"] == "unhealthy"
        assert health["component"] == "database_storage"

    @pytest.mark.asyncio
    async def test_health_check_no_database_service(self, config):
        """Test health check without database service."""
        storage = DatabaseStorage(config)

        health = await storage.health_check()

        assert health["status"] == "unhealthy"
        assert health["component"] == "database_storage"
        assert "Database service not available" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_exception(self, config, mock_database_service):
        """Test health check with exception."""
        mock_database_service.get_health_status.side_effect = Exception("Health check failed")
        storage = DatabaseStorage(config, mock_database_service)

        health = await storage.health_check()

        assert health["status"] == "unhealthy"
        assert health["component"] == "database_storage"
        assert "Health check failed" in health["error"]


class TestDatabaseStorageCleanup:
    """Test cleanup operations."""

    @pytest.mark.asyncio
    async def test_cleanup_success(self, config, mock_database_service):
        """Test successful cleanup."""
        storage = DatabaseStorage(config, mock_database_service)
        storage._initialized = True

        await storage.cleanup()

        assert storage._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_not_initialized(self, config):
        """Test cleanup when not initialized."""
        storage = DatabaseStorage(config)
        storage._initialized = False

        await storage.cleanup()

        assert storage._initialized is False


class TestDatabaseStorageIntegration:
    """Integration tests for database storage."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, config, mock_database_service, sample_records):
        """Test full workflow: store, retrieve, count."""
        mock_database_service.list_entities.return_value = sample_records
        mock_database_service.count_entities.return_value = len(sample_records)

        storage = DatabaseStorage(config, mock_database_service)
        await storage.initialize()

        # Store records
        store_result = await storage.store_records(sample_records)
        assert store_result is True

        # Retrieve records
        request = DataRequest(symbol="BTC/USD", exchange="binance")
        retrieved = await storage.retrieve_records(request)
        assert len(retrieved) == len(sample_records)

        # Count records
        count = await storage.get_record_count("BTC/USD", "binance")
        assert count == len(sample_records)

        # Health check
        mock_health_status = Mock()
        mock_health_status.name = "HEALTHY"
        mock_database_service.get_health_status.return_value = mock_health_status

        health = await storage.health_check()
        assert health["status"] == "healthy"

        # Cleanup
        await storage.cleanup()
        assert storage._initialized is False

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, config, mock_database_service, sample_records):
        """Test error handling across operations."""
        storage = DatabaseStorage(config, mock_database_service)

        # Test store error
        mock_database_service.bulk_create.side_effect = DatabaseError("Store failed")
        with pytest.raises(DataError):
            await storage.store_records(sample_records)

        # Reset mock
        mock_database_service.bulk_create.side_effect = None

        # Test retrieve error
        mock_database_service.list_entities.side_effect = DatabaseError("Retrieve failed")
        request = DataRequest(symbol="BTC/USD", exchange="binance")
        with pytest.raises(DataError):
            await storage.retrieve_records(request)

        # Test count error (should return 0, not raise)
        mock_database_service.count_entities.side_effect = DatabaseError("Count failed")
        count = await storage.get_record_count("BTC/USD", "binance")
        assert count == 0


class TestDatabaseStorageEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_store_large_record_batch(self, config, mock_database_service):
        """Test storing large batch of records."""
        # Create large batch
        large_batch = []
        for i in range(10000):
            record = MarketDataRecord(
                symbol=f"COIN{i}/USD",
                exchange="binance",
                price=Decimal("100.00"),
                volume=Decimal("10.0"),
                data_timestamp=datetime.now(timezone.utc),
            )
            large_batch.append(record)

        storage = DatabaseStorage(config, mock_database_service)

        result = await storage.store_records(large_batch)

        assert result is True
        mock_database_service.bulk_create.assert_called_once_with(large_batch)

    @pytest.mark.asyncio
    async def test_retrieve_with_complex_filters(self, config, mock_database_service):
        """Test retrieval with all possible filters."""
        mock_database_service.list_entities.return_value = []
        storage = DatabaseStorage(config, mock_database_service)

        request = DataRequest(
            symbol="BTC/USD",
            exchange="binance",
            start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2023, 12, 31, tzinfo=timezone.utc),
            limit=5000
        )

        await storage.retrieve_records(request)

        call_args = mock_database_service.list_entities.call_args
        filters = call_args.kwargs["filters"]

        assert filters["symbol"] == "BTC/USD"
        assert filters["exchange"] == "binance"
        assert filters["data_timestamp"]["gte"] == request.start_time
        assert filters["data_timestamp"]["lte"] == request.end_time
        assert call_args.kwargs["limit"] == 5000

    @pytest.mark.asyncio
    async def test_count_with_special_symbols(self, config, mock_database_service):
        """Test counting records with special characters in symbols."""
        mock_database_service.count_entities.return_value = 100
        storage = DatabaseStorage(config, mock_database_service)

        special_symbols = [
            "BTC-USD",
            "BTC.USD",
            "BTC_USD",
            "BTC/USD",
            "BTC@USD",
            "1INCH/USD",
        ]

        for symbol in special_symbols:
            count = await storage.get_record_count(symbol, "binance")
            assert count == 100

            # Verify correct symbol was passed
            call_args = mock_database_service.count_entities.call_args
            assert call_args.kwargs["filters"]["symbol"] == symbol

    @pytest.mark.asyncio
    async def test_concurrent_operations_simulation(self, config, mock_database_service, sample_records):
        """Test simulated concurrent operations."""
        storage = DatabaseStorage(config, mock_database_service)

        # Simulate multiple operations happening "concurrently"
        operations = []

        # Store operations
        for _ in range(5):
            operations.append(storage.store_records(sample_records))

        # Retrieve operations
        mock_database_service.list_entities.return_value = sample_records
        for _ in range(5):
            request = DataRequest(symbol="BTC/USD", exchange="binance")
            operations.append(storage.retrieve_records(request))

        # Count operations
        mock_database_service.count_entities.return_value = 100
        for _ in range(5):
            operations.append(storage.get_record_count("BTC/USD", "binance"))

        # Execute all operations
        results = []
        for operation in operations:
            result = await operation
            results.append(result)

        # Verify all operations completed
        assert len(results) == 15

        # Check store results
        store_results = results[:5]
        assert all(result is True for result in store_results)

        # Check retrieve results
        retrieve_results = results[5:10]
        assert all(len(result) == len(sample_records) for result in retrieve_results)

        # Check count results
        count_results = results[10:15]
        assert all(result == 100 for result in count_results)

    @pytest.mark.asyncio
    async def test_health_status_edge_cases(self, config, mock_database_service):
        """Test health check with various status values."""
        storage = DatabaseStorage(config, mock_database_service)

        # Test various health status names
        status_cases = [
            ("HEALTHY", "healthy"),
            ("UNHEALTHY", "unhealthy"),
            ("DEGRADED", "unhealthy"),
            ("UNKNOWN", "unhealthy"),
            ("", "unhealthy"),
        ]

        for status_name, expected_status in status_cases:
            mock_health_status = Mock()
            mock_health_status.name = status_name
            mock_database_service.get_health_status.return_value = mock_health_status

            health = await storage.health_check()

            assert health["status"] == expected_status
            assert health["component"] == "database_storage"
