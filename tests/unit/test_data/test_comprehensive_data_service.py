"""
Test suite for DataService.

This module contains tests for the simplified DataService
including initialization, caching, data storage, and retrieval.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.config import Config
from src.core.types import MarketData
from src.data.services.data_service import DataService
from src.data.types import DataRequest
from src.database.interfaces import DatabaseServiceInterface
from src.database.models import MarketDataRecord
from src.monitoring import MetricsCollector


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=Config)

    # Data service configuration - use dict for proper .get() behavior
    config.data_service = {
        "l1_cache_max_size": 1000,
        "l1_cache_ttl": 300,
        "l2_cache_ttl": 600,
    }

    return config


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for testing."""
    collector = Mock(spec=MetricsCollector)
    collector.increment_counter = Mock()
    collector.record_histogram = Mock()
    collector.record_gauge = Mock()
    return collector


@pytest.fixture
def mock_database_service():
    """Mock database service for testing."""
    service = Mock(spec=DatabaseServiceInterface)
    service.bulk_create = AsyncMock()
    service.list_entities = AsyncMock()
    service.count_entities = AsyncMock()
    service.get_health_status = AsyncMock()
    return service


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return [
        MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            price=Decimal("50000.00"),
            volume=Decimal("1.5"),
            bid_price=Decimal("49999.50"),
            ask_price=Decimal("50000.50"),
            high=Decimal("50500.00"),
            low=Decimal("49500.00"),
            open=Decimal("50200.00"),
            close=Decimal("50000.00"),
            exchange="binance",
        ),
        MarketData(
            symbol="ETHUSDT",
            timestamp=datetime.now(timezone.utc),
            price=Decimal("3000.00"),
            volume=Decimal("2.0"),
            bid_price=Decimal("2999.50"),
            ask_price=Decimal("3000.50"),
            high=Decimal("3100.00"),
            low=Decimal("2950.00"),
            open=Decimal("3050.00"),
            close=Decimal("3000.00"),
            exchange="binance",
        ),
    ]


class TestDataServiceInitialization:
    """Test DataService initialization."""

    def test_initialization_basic(self, mock_config, mock_metrics_collector, mock_database_service):
        """Test basic initialization."""
        service = DataService(
            config=mock_config,
            metrics_collector=mock_metrics_collector,
            database_service=mock_database_service,
        )

        assert service.config is mock_config
        assert service.database_service is mock_database_service
        assert isinstance(service._memory_cache, dict)
        assert service.cache_service is None
        assert hasattr(service, "metrics_collector")

    def test_initialization_minimal(self, mock_config, mock_database_service):
        """Test initialization with minimal parameters."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        assert service.config is mock_config
        assert service.database_service is mock_database_service
        assert isinstance(service._memory_cache, dict)

    def test_setup_configuration(self, mock_config, mock_database_service):
        """Test configuration setup."""
        # Mock missing attributes to test defaults
        mock_config.data_service = None

        service = DataService(config=mock_config, database_service=mock_database_service)

        # Should not crash and should handle missing configuration gracefully
        assert service.config is mock_config

    def test_initialization_without_database_service(self, mock_config):
        """Test initialization fails without database service."""
        from src.core.exceptions import DataError
        with pytest.raises(DataError, match="database_service is required"):
            DataService(config=mock_config, database_service=None)

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_database_service):
        """Test successful initialization."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        await service.initialize()

        assert service._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_redis_success(self, mock_config, mock_database_service):
        """Test cache service initialization."""
        mock_cache_service = Mock()
        mock_cache_service.initialize = AsyncMock()

        service = DataService(
            config=mock_config,
            database_service=mock_database_service,
            cache_service=mock_cache_service
        )

        await service.initialize()

        assert service.cache_service is mock_cache_service
        mock_cache_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_twice(self, mock_config, mock_database_service):
        """Test that initialize can be called multiple times safely."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        await service.initialize()
        assert service._initialized is True

        # Should not fail when called again
        await service.initialize()
        assert service._initialized is True


class TestDataServiceCaching:
    """Test DataService caching functionality."""

    def test_memory_cache_initialization(self, mock_config, mock_database_service):
        """Test memory cache is initialized."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        assert isinstance(service._memory_cache, dict)
        assert len(service._memory_cache) == 0

    def test_cache_manager_integration(self, mock_config, mock_database_service):
        """Test cache service integration."""
        mock_cache_service = Mock()
        service = DataService(
            config=mock_config,
            database_service=mock_database_service,
            cache_service=mock_cache_service
        )

        assert service.cache_service is mock_cache_service


class TestDataServiceStorage:
    """Test DataService storage functionality."""

    @pytest.mark.asyncio
    async def test_store_market_data_success(
        self, mock_config, mock_database_service, sample_market_data
    ):
        """Test successful market data storage."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        result = await service.store_market_data(sample_market_data, "binance", validate=False)

        assert result is True
        mock_database_service.bulk_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_market_data_empty_list(self, mock_config, mock_database_service):
        """Test storing empty market data list."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        result = await service.store_market_data([], "binance")
        assert result is False

    @pytest.mark.asyncio
    async def test_store_market_data_single_item(
        self, mock_config, mock_database_service, sample_market_data
    ):
        """Test storing single market data item."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        result = await service.store_market_data(sample_market_data[0], "binance", validate=False)

        assert result is True
        mock_database_service.bulk_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_market_data_database_error(
        self, mock_config, mock_database_service, sample_market_data
    ):
        """Test market data storage with database error."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        mock_database_service.bulk_create.side_effect = Exception("Database error")

        result = await service.store_market_data(sample_market_data, "binance", validate=False)

        assert result is False


class TestDataServiceRetrieval:
    """Test DataService data retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_market_data_success(self, mock_config, mock_database_service):
        """Test successful market data retrieval."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        mock_records = [Mock(spec=MarketDataRecord) for _ in range(2)]
        mock_database_service.list_entities.return_value = mock_records

        request = DataRequest(symbol="BTCUSDT", exchange="binance", use_cache=False)
        result = await service.get_market_data(request)

        # Data service reverses records to chronological order
        assert result == list(reversed(mock_records))
        mock_database_service.list_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_recent_data_success(self, mock_config, mock_database_service):
        """Test get recent data."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        # Mock database records
        mock_record = Mock(spec=MarketDataRecord)
        mock_record.symbol = "BTCUSDT"
        mock_record.exchange = "binance"
        mock_record.timestamp = datetime.now(timezone.utc)
        mock_record.data_timestamp = datetime.now(timezone.utc)
        mock_record.open_price = Decimal("50000")
        mock_record.high_price = Decimal("51000")
        mock_record.low_price = Decimal("49000")
        mock_record.close_price = Decimal("50500")
        mock_record.volume = Decimal("1.0")

        mock_database_service.list_entities.return_value = [mock_record]

        result = await service.get_recent_data("BTCUSDT", limit=10, exchange="binance")

        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_data_count_success(self, mock_config, mock_database_service):
        """Test get data count."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        mock_database_service.count_entities.return_value = 100

        result = await service.get_data_count("BTCUSDT", "binance")

        assert result == 100
        mock_database_service.count_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_count_error(self, mock_config, mock_database_service):
        """Test get data count with error."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        mock_database_service.count_entities.side_effect = Exception("Database error")

        result = await service.get_data_count("BTCUSDT", "binance")

        assert result == 0


class TestDataServiceHealthCheck:
    """Test DataService health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_config, mock_database_service):
        """Test healthy status."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        # Mock healthy database
        from src.core import HealthStatus
        mock_health = Mock()
        mock_health.name = "HEALTHY"
        mock_database_service.get_health_status.return_value = mock_health

        result = await service.health_check()

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, mock_config, mock_database_service):
        """Test degraded status."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        # Mock degraded database
        from src.core import HealthStatus
        mock_health = Mock()
        mock_health.name = "DEGRADED"
        mock_database_service.get_health_status.return_value = mock_health

        result = await service.health_check()

        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_cleanup_success(self, mock_config, mock_database_service):
        """Test successful cleanup."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        # Add some data to cache
        service._memory_cache["test"] = "data"
        service._initialized = True

        await service.cleanup()

        assert len(service._memory_cache) == 0
        assert service._initialized is False


class TestDataServiceValidation:
    """Test DataService validation functionality."""

    def test_validate_market_data_success(self, mock_config, mock_database_service, sample_market_data):
        """Test successful market data validation."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        # Mock validation to return valid data
        with patch("src.utils.validation.market_data_validation.MarketDataValidator") as mock_validator:
            mock_validator_instance = mock_validator.return_value
            mock_validator_instance.validate_market_data_batch.return_value = sample_market_data
            
            # Validation should not raise exceptions for valid data
            result = service._validate_market_data(sample_market_data)

            # Should return the same data or validated data
            assert len(result) == len(sample_market_data)

    def test_validate_market_data_fallback(self, mock_config, mock_database_service, sample_market_data):
        """Test market data validation fallback on error."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        # Mock validation to raise exception
        with patch("src.utils.validation.market_data_validation.MarketDataValidator") as mock_validator:
            mock_validator.side_effect = Exception("Validation error")

            result = service._validate_market_data(sample_market_data)

            # Should fallback to returning original data
            assert result == sample_market_data


class TestDataServiceTransformation:
    """Test DataService data transformation functionality."""

    def test_transform_to_db_records(self, mock_config, mock_database_service, sample_market_data):
        """Test transformation of MarketData to database records."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        records = service._transform_to_db_records(sample_market_data, "binance")

        assert len(records) == len(sample_market_data)
        for record in records:
            assert isinstance(record, MarketDataRecord)
            assert record.exchange == "binance"
            assert record.symbol in ["BTCUSDT", "ETHUSDT"]

    def test_build_cache_key(self, mock_config, mock_database_service):
        """Test cache key building."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        request = DataRequest(
            symbol="BTCUSDT",
            exchange="binance",
            limit=100,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2)
        )

        key = service._build_cache_key(request)

        assert "BTCUSDT" in key
        assert "binance" in key
        assert "limit:100" in key


class TestDataServiceCacheOperations:
    """Test DataService cache operations."""

    @pytest.mark.asyncio
    async def test_update_l1_cache(self, mock_config, mock_database_service, sample_market_data):
        """Test L1 cache update."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        await service._update_l1_cache(sample_market_data)

        # Should have cache entries for both symbols
        assert len(service._memory_cache) > 0

        # Check cache entries exist
        btc_key = "market_data:BTCUSDT:latest"
        eth_key = "market_data:ETHUSDT:latest"

        if btc_key in service._memory_cache:
            assert "data" in service._memory_cache[btc_key]
            assert "timestamp" in service._memory_cache[btc_key]
            assert "ttl" in service._memory_cache[btc_key]

    def test_get_from_l1_cache_hit(self, mock_config, mock_database_service):
        """Test L1 cache hit."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        # Pre-populate cache
        request = DataRequest(symbol="BTCUSDT", exchange="binance")
        cache_key = service._build_cache_key(request)
        test_data = [Mock(spec=MarketDataRecord)]

        service._memory_cache[cache_key] = {
            "data": test_data,
            "timestamp": datetime.now(timezone.utc),
            "ttl": 300
        }

        result = service._get_from_l1_cache(request)

        assert result == test_data

    def test_get_from_l1_cache_miss(self, mock_config, mock_database_service):
        """Test L1 cache miss."""
        service = DataService(config=mock_config, database_service=mock_database_service)

        request = DataRequest(symbol="BTCUSDT", exchange="binance")
        result = service._get_from_l1_cache(request)

        assert result is None


class TestDataServiceErrorHandling:
    """Test DataService error handling."""

    @pytest.mark.asyncio
    async def test_store_market_data_database_error(
        self, mock_config, mock_database_service, sample_market_data, mock_metrics_collector
    ):
        """Test market data storage with database error."""
        service = DataService(
            config=mock_config,
            database_service=mock_database_service,
            metrics_collector=mock_metrics_collector
        )

        mock_database_service.bulk_create.side_effect = Exception("Database connection failed")

        result = await service.store_market_data(sample_market_data, "binance", validate=False)

        assert result is False
        mock_metrics_collector.increment_counter.assert_called()

    @pytest.mark.asyncio
    async def test_get_market_data_database_error(
        self, mock_config, mock_database_service, mock_metrics_collector
    ):
        """Test market data retrieval with database error."""
        service = DataService(
            config=mock_config,
            database_service=mock_database_service,
            metrics_collector=mock_metrics_collector
        )

        mock_database_service.list_entities.side_effect = Exception("Database error")

        request = DataRequest(symbol="BTCUSDT", exchange="binance")
        result = await service.get_market_data(request)

        assert result == []
        mock_metrics_collector.increment_counter.assert_called()
