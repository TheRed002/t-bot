"""
Comprehensive test suite for data service.

This module contains comprehensive tests for the DataService
including initialization, caching, data storage, validation, and pipeline processing.
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid

from src.core.config import Config
from src.core.exceptions import DataError, DataValidationError
from src.core.types import MarketData
from src.data.types import (
    CacheLevel,
    DataMetrics,
    DataPipelineStage,
    DataRequest,
)
from src.data.services.data_service import DataService
from src.database.interfaces import DatabaseServiceInterface
from src.database.models import MarketDataRecord
from src.monitoring import MetricsCollector, Status, StatusCode


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=Config)
    
    # Redis configuration
    config.redis = Mock()
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.redis.db = 0
    config.redis.password = None
    config.redis.ssl = False
    config.redis.max_connections = 20
    config.redis.socket_timeout = 5
    
    # Data service configuration - use dict for proper .get() behavior
    config.data_service = {
        "cache_ttl": 300,
        "batch_size": 1000,
        "max_retries": 3,
        "enable_stream_processing": True,
        "enable_feature_store": True,
        "enable_validation": True,
        "max_financial_value": 1e15,
        "decimal_precision": 8,
        "processing_mode": "batch"
    }
    
    # Feature store configuration
    config.feature_store = Mock()
    config.feature_store.cache_size = 10000
    config.feature_store.ttl = 300
    
    return config


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for testing."""
    collector = Mock(spec=MetricsCollector)
    collector.record_counter = Mock()
    collector.record_histogram = Mock()
    collector.record_gauge = Mock()
    collector.get_status = Mock(return_value=Status(StatusCode.OK, "OK"))
    return collector


@pytest.fixture
def mock_database_service():
    """Mock database service for testing."""
    service = Mock(spec=DatabaseServiceInterface)
    service.store_market_data = AsyncMock()
    service.get_market_data = AsyncMock()
    service.health_check = AsyncMock(return_value={"status": "healthy"})
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
            bid=Decimal("49999.50"),
            ask=Decimal("50000.50"),
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
            bid=Decimal("2999.50"),
            ask=Decimal("3000.50"),
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
        assert service._redis_client is None
        assert hasattr(service, 'cache_manager')

    def test_initialization_minimal(self, mock_config):
        """Test initialization with minimal parameters."""
        service = DataService(config=mock_config)
        
        assert service.config is mock_config
        assert service.database_service is None
        assert isinstance(service._memory_cache, dict)

    def test_setup_configuration(self, mock_config):
        """Test configuration setup."""
        # Mock missing attributes to test defaults
        mock_config.data_service = None
        mock_config.redis = None
        mock_config.feature_store = None
        
        service = DataService(config=mock_config)
        
        # Should not crash and should handle missing configuration gracefully
        assert service.config is mock_config

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_database_service):
        """Test successful initialization."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock the initialization methods
        with patch.object(service, '_initialize_redis', new_callable=AsyncMock) as mock_redis, \
             patch.object(service, '_initialize_feature_store', new_callable=AsyncMock) as mock_feature, \
             patch.object(service, '_initialize_validators', new_callable=AsyncMock) as mock_validators, \
             patch.object(service, '_setup_metrics_collection', new_callable=AsyncMock) as mock_metrics:
            
            await service.initialize()
            
            mock_redis.assert_called_once()
            mock_feature.assert_called_once()
            mock_validators.assert_called_once()
            mock_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_redis_success(self, mock_config):
        """Test Redis initialization."""
        service = DataService(config=mock_config)
        
        # Mock Redis client creation
        mock_redis_client = Mock()
        mock_redis_client.ping = AsyncMock(return_value=True)
        
        with patch('src.data.services.data_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_client
            
            await service._initialize_redis()
            
            assert service._redis_client is mock_redis_client
            mock_redis_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_redis_failure(self, mock_config):
        """Test Redis initialization failure."""
        service = DataService(config=mock_config)
        
        # Mock Redis client that fails ping
        mock_redis_client = Mock()
        mock_redis_client.ping = AsyncMock(side_effect=Exception("Redis connection failed"))
        
        with patch('src.data.services.data_service.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_client
            
            # Should not raise exception, just log warning
            await service._initialize_redis()
            
            # Redis client should be None on failure
            assert service._redis_client is None

    @pytest.mark.asyncio
    async def test_initialize_feature_store(self, mock_config):
        """Test feature store initialization."""
        service = DataService(config=mock_config)
        
        # The actual implementation just logs initialization - no external class required
        await service._initialize_feature_store()
        
        # Should not raise any exceptions and just complete successfully
        assert service.config is mock_config

    @pytest.mark.asyncio
    async def test_initialize_validators(self, mock_config):
        """Test validators initialization."""
        service = DataService(config=mock_config)
        
        # The actual implementation just logs initialization - no external classes required
        await service._initialize_validators()
        
        # Should not raise any exceptions and just complete successfully
        assert service.config is mock_config

    @pytest.mark.asyncio
    async def test_setup_metrics_collection(self, mock_config, mock_metrics_collector):
        """Test metrics collection setup."""
        service = DataService(config=mock_config, metrics_collector=mock_metrics_collector)
        
        await service._setup_metrics_collection()
        
        # Should not raise any exceptions
        assert service.config is mock_config


class TestDataServiceCaching:
    """Test DataService caching functionality."""

    def test_memory_cache_initialization(self, mock_config):
        """Test memory cache is initialized."""
        service = DataService(config=mock_config)
        
        assert isinstance(service._memory_cache, dict)
        assert len(service._memory_cache) == 0

    def test_cache_manager_integration(self, mock_config):
        """Test cache manager integration."""
        with patch('src.data.services.data_service.get_cache_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            service = DataService(config=mock_config)
            
            assert service.cache_manager is mock_manager
            mock_get_manager.assert_called_once_with(config=mock_config)


class TestDataServiceStorage:
    """Test DataService storage functionality."""

    @pytest.mark.asyncio
    async def test_store_market_data_success(self, mock_config, mock_database_service, sample_market_data):
        """Test successful market data storage."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock validation and pipeline execution
        with patch.object(service, '_validate_market_data', new_callable=AsyncMock) as mock_validate, \
             patch.object(service, '_execute_storage_pipeline', new_callable=AsyncMock) as mock_pipeline, \
             patch.object(service, '_validate_data_at_boundary', new_callable=AsyncMock) as mock_boundary_validate, \
             patch.object(service, '_update_caches', new_callable=AsyncMock) as mock_update_caches:
            
            mock_validate.return_value = sample_market_data
            mock_pipeline.return_value = "pipeline_123"
            mock_boundary_validate.return_value = None  # No return value
            mock_update_caches.return_value = None
            
            result = await service.store_market_data(sample_market_data, "binance")
            
            assert result is True  # store_market_data returns bool
            mock_validate.assert_called_once_with(sample_market_data)
            mock_pipeline.assert_called_once_with(sample_market_data, "binance")

    @pytest.mark.asyncio
    async def test_store_market_data_empty_list(self, mock_config, mock_database_service):
        """Test storing empty market data list."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        with pytest.raises(DataValidationError, match="Empty data list provided"):
            await service.store_market_data([], "binance")

    @pytest.mark.asyncio
    async def test_store_market_data_validation_error(self, mock_config, mock_database_service, sample_market_data):
        """Test market data storage with validation error."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock validation to raise error
        with patch.object(service, '_validate_market_data', new_callable=AsyncMock) as mock_validate:
            mock_validate.side_effect = DataValidationError("Validation failed")
            
            with pytest.raises(DataValidationError, match="Validation failed"):
                await service.store_market_data(sample_market_data, "binance")

    @pytest.mark.asyncio
    async def test_validate_market_data_success(self, mock_config, sample_market_data):
        """Test successful market data validation."""
        service = DataService(config=mock_config)
        
        # Override the validation config with real values
        service.validation_config = {
            "max_financial_value": 1e15,
            "decimal_precision": 8
        }
        
        # Mock the validate_market_data utility function
        with patch('src.data.services.data_service.validate_market_data') as mock_validate, \
             patch('src.data.services.data_service.validate_decimal_precision') as mock_decimal_validate, \
             patch('src.utils.decimal_utils.to_decimal') as mock_to_decimal:
            
            mock_validate.return_value = True
            mock_decimal_validate.return_value = True
            mock_to_decimal.side_effect = lambda x: Decimal(str(x))
            
            result = await service._validate_market_data(sample_market_data)
            
            assert result == sample_market_data
            assert mock_validate.call_count == len(sample_market_data)

    @pytest.mark.asyncio
    async def test_validate_market_data_failure(self, mock_config, sample_market_data):
        """Test market data validation failure."""
        service = DataService(config=mock_config)
        
        # Mock validation to fail
        with patch('src.data.services.data_service.validate_market_data') as mock_validate:
            mock_validate.return_value = False
            
            # Should return empty list when all validation fails
            result = await service._validate_market_data(sample_market_data)
            assert result == []

    @pytest.mark.asyncio
    async def test_execute_storage_pipeline_batch(self, mock_config, mock_database_service, sample_market_data):
        """Test storage pipeline execution with batch processing."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        service.config.data_service["batch_size"] = 1  # Force batch processing
        service._setup_configuration()  # Re-read configuration
        
        # Mock pipeline methods
        with patch.object(service, '_execute_batch_pipeline', new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = "batch_pipeline_123"
            
            result = await service._execute_storage_pipeline(sample_market_data, "binance")
            
            assert result == "batch_pipeline_123"
            mock_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_storage_pipeline_stream(self, mock_config, mock_database_service, sample_market_data):
        """Test storage pipeline execution with stream processing."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        service.config.data_service["processing_mode"] = "stream"
        service.config.data_service["batch_size"] = 10000  # Large batch size to avoid batch mode
        service._setup_configuration()  # Re-read configuration
        
        # Mock pipeline methods
        with patch.object(service, '_execute_stream_pipeline', new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value = "stream_pipeline_123"
            
            result = await service._execute_storage_pipeline(sample_market_data, "binance")
            
            assert result == "stream_pipeline_123"
            mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_storage_pipeline_default(self, mock_config, mock_database_service, sample_market_data):
        """Test storage pipeline execution with default processing."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        service.config.data_service["processing_mode"] = "default"  # This will hit the else clause
        service.config.data_service["batch_size"] = 10000  # Large batch size to avoid batch mode
        service._setup_configuration()  # Re-read configuration
        
        # Mock pipeline methods
        with patch.object(service, '_execute_default_pipeline', new_callable=AsyncMock) as mock_default:
            mock_default.return_value = "default_pipeline_123"
            
            result = await service._execute_storage_pipeline(sample_market_data, "binance")
            
            assert result == "default_pipeline_123"
            mock_default.assert_called_once()


class TestDataServicePipelines:
    """Test DataService pipeline functionality."""

    @pytest.mark.asyncio
    async def test_execute_batch_pipeline(self, mock_config, mock_database_service, sample_market_data):
        """Test batch pipeline execution."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock database and cache operations
        with patch.object(service, '_transform_to_db_records', new_callable=AsyncMock) as mock_transform, \
             patch.object(service, '_update_pipeline_stage', new_callable=AsyncMock) as mock_update, \
             patch.object(service, '_store_to_database', new_callable=AsyncMock) as mock_store, \
             patch.object(service, '_update_indexes', new_callable=AsyncMock) as mock_index:
            
            mock_db_records = [Mock(spec=MarketDataRecord) for _ in sample_market_data]
            mock_transform.return_value = mock_db_records
            mock_store.return_value = None
            mock_index.return_value = None
            
            pipeline_id = await service._execute_batch_pipeline(sample_market_data, "binance", "test_pipeline_id")
            
            assert isinstance(pipeline_id, str)
            mock_transform.assert_called_once_with(sample_market_data, "binance")
            mock_store.assert_called_once_with(mock_db_records)
            assert mock_update.call_count >= 2  # Should update stage multiple times

    @pytest.mark.asyncio
    async def test_execute_stream_pipeline(self, mock_config, mock_database_service, sample_market_data):
        """Test stream pipeline execution."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock stream processing
        with patch.object(service, '_transform_to_db_records', new_callable=AsyncMock) as mock_transform, \
             patch.object(service, '_store_to_database', new_callable=AsyncMock) as mock_store, \
             patch.object(service, '_update_pipeline_stage', new_callable=AsyncMock):
            
            mock_db_records = [Mock(spec=MarketDataRecord) for _ in sample_market_data]
            mock_transform.return_value = mock_db_records
            mock_store.return_value = None
            
            pipeline_id = await service._execute_stream_pipeline(sample_market_data, "binance", "test_pipeline_id")
            
            assert isinstance(pipeline_id, str)
            mock_transform.assert_called()  # May be called multiple times depending on batching
            mock_store.assert_called()  # May be called multiple times depending on batching

    @pytest.mark.asyncio
    async def test_execute_default_pipeline(self, mock_config, mock_database_service, sample_market_data):
        """Test default pipeline execution."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock database operations
        with patch.object(service, '_transform_to_db_records', new_callable=AsyncMock) as mock_transform, \
             patch.object(service, '_store_to_database', new_callable=AsyncMock) as mock_store, \
             patch.object(service, '_update_pipeline_stage', new_callable=AsyncMock):
            
            mock_db_records = [Mock(spec=MarketDataRecord) for _ in sample_market_data]
            mock_transform.return_value = mock_db_records
            mock_store.return_value = None
            
            pipeline_id = await service._execute_default_pipeline(sample_market_data, "binance", "test_pipeline_id")
            
            assert isinstance(pipeline_id, str)
            mock_transform.assert_called_once_with(sample_market_data, "binance")
            mock_store.assert_called_once_with(mock_db_records)

    @pytest.mark.asyncio
    async def test_process_stream_batch(self, mock_config, mock_database_service, sample_market_data):
        """Test stream batch processing."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock real-time processing
        with patch.object(service, '_transform_to_db_records', new_callable=AsyncMock) as mock_transform, \
             patch.object(service, '_store_to_database', new_callable=AsyncMock) as mock_store, \
             patch.object(service, '_update_pipeline_stage', new_callable=AsyncMock):
            
            mock_db_records = [Mock(spec=MarketDataRecord) for _ in sample_market_data]
            mock_transform.return_value = mock_db_records
            mock_store.return_value = None
            
            await service._process_stream_batch(sample_market_data, "binance", "test_pipeline_id")
            
            mock_transform.assert_called_once_with(sample_market_data, "binance")
            mock_store.assert_called_once_with(mock_db_records)

    @pytest.mark.asyncio
    async def test_update_pipeline_stage(self, mock_config):
        """Test pipeline stage update."""
        service = DataService(config=mock_config)
        
        # Should not raise any exceptions
        await service._update_pipeline_stage("test_pipeline", DataPipelineStage.PROCESSING)
        await service._update_pipeline_stage("test_pipeline", DataPipelineStage.INDEXING)


class TestDataServiceTransformation:
    """Test DataService data transformation."""

    @pytest.mark.asyncio
    async def test_transform_to_db_records(self, mock_config, sample_market_data):
        """Test transformation of market data to database records."""
        service = DataService(config=mock_config)
        
        # Mock MarketDataRecord creation
        with patch('src.data.services.data_service.MarketDataRecord') as mock_record_class:
            mock_records = [Mock(spec=MarketDataRecord) for _ in sample_market_data]
            mock_record_class.side_effect = mock_records
            
            result = await service._transform_to_db_records(sample_market_data, "binance")
            
            assert len(result) == len(sample_market_data)
            assert all(isinstance(r, Mock) for r in result)
            assert mock_record_class.call_count == len(sample_market_data)


class TestDataServiceValidation:
    """Test DataService validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_data_at_boundary_all_layers(self, mock_config):
        """Test data validation at all boundary layers."""
        service = DataService(config=mock_config)
        
        test_data = {"symbol": "BTCUSDT", "price": 50000.00}
        
        # Mock validation methods
        with patch.object(service, '_validate_input_boundary') as mock_input, \
             patch.object(service, '_validate_database_boundary') as mock_db, \
             patch.object(service, '_validate_cache_boundary') as mock_cache:
            
            await service._validate_data_at_boundary(test_data, validate_input=True, 
                                                   validate_database=True, validate_cache=True)
            
            mock_input.assert_called_once_with(test_data)
            mock_db.assert_called_once_with(test_data)
            mock_cache.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_validate_data_at_boundary_selective(self, mock_config):
        """Test selective data validation at boundaries."""
        service = DataService(config=mock_config)
        
        test_data = {"symbol": "BTCUSDT", "price": 50000.00}
        
        # Mock validation methods
        with patch.object(service, '_validate_input_boundary') as mock_input, \
             patch.object(service, '_validate_database_boundary') as mock_db, \
             patch.object(service, '_validate_cache_boundary') as mock_cache:
            
            await service._validate_data_at_boundary(test_data, validate_input=True, 
                                                   validate_database=False, validate_cache=False)
            
            mock_input.assert_called_once_with(test_data)
            mock_db.assert_not_called()
            mock_cache.assert_not_called()

    def test_validate_input_boundary(self, mock_config):
        """Test input boundary validation."""
        service = DataService(config=mock_config)
        
        # Valid data should not raise
        valid_data = {
            "symbol": "BTCUSDT", 
            "price": 50000.00, 
            "timestamp": datetime.now(timezone.utc), 
            "exchange": "binance"
        }
        service._validate_input_boundary(valid_data)
        
        # Test missing required field raises DataValidationError
        invalid_data = {"symbol": "BTCUSDT", "price": 50000.00}
        with pytest.raises(DataValidationError, match="Required field timestamp missing"):
            service._validate_input_boundary(invalid_data)

    def test_validate_database_boundary(self, mock_config):
        """Test database boundary validation."""
        service = DataService(config=mock_config)
        
        # Override the validation config with real values
        service.validation_config = {
            "max_financial_value": 1e15,
            "decimal_precision": 8
        }
        
        # Valid data should not raise
        valid_data = {"symbol": "BTCUSDT", "price": 50000.00}
        with patch('src.utils.decimal_utils.to_decimal') as mock_to_decimal:
            mock_to_decimal.side_effect = lambda x: Decimal(str(x))
            service._validate_database_boundary(valid_data)

    def test_validate_cache_boundary(self, mock_config):
        """Test cache boundary validation."""
        service = DataService(config=mock_config)
        
        # Valid data should not raise (must include cache key fields)
        valid_data = {
            "symbol": "BTCUSDT", 
            "price": 50000.00,
            "exchange": "binance",
            "timestamp": datetime.now(timezone.utc)
        }
        service._validate_cache_boundary(valid_data)


class TestDataServiceErrorHandling:
    """Test DataService error handling."""

    @pytest.mark.asyncio
    async def test_store_market_data_database_error(self, mock_config, mock_database_service, sample_market_data):
        """Test handling database errors during storage."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock store_to_database to raise error since that's what actually gets called
        with patch.object(service, '_validate_market_data', new_callable=AsyncMock) as mock_validate, \
             patch.object(service, '_store_to_database', new_callable=AsyncMock) as mock_store:
            
            mock_validate.return_value = sample_market_data
            mock_store.side_effect = Exception("Database error")
            
            # The method catches exceptions and returns False instead of re-raising
            result = await service.store_market_data(sample_market_data, "binance")
            assert result is False  # Should return False on database error

    @pytest.mark.asyncio
    async def test_pipeline_execution_error_handling(self, mock_config, mock_database_service, sample_market_data):
        """Test error handling in pipeline execution."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock transform method to raise error
        with patch.object(service, '_transform_to_db_records') as mock_transform:
            mock_transform.side_effect = Exception("Transform error")
            
            with pytest.raises(Exception, match="Transform error"):
                await service._execute_default_pipeline(sample_market_data, "binance", "test_pipeline_id")


class TestDataServiceIntegration:
    """Test DataService integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_storage_workflow(self, mock_config, mock_database_service, sample_market_data):
        """Test complete storage workflow."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock all dependencies
        with patch.object(service, '_validate_market_data', new_callable=AsyncMock) as mock_validate, \
             patch.object(service, '_transform_to_db_records', new_callable=AsyncMock) as mock_transform, \
             patch.object(service, '_store_to_database', new_callable=AsyncMock) as mock_store, \
             patch('src.data.services.data_service.validate_market_data') as mock_validate_util:
            
            mock_validate.return_value = sample_market_data
            mock_db_records = [Mock(spec=MarketDataRecord) for _ in sample_market_data]
            mock_transform.return_value = mock_db_records
            mock_store.return_value = None
            mock_validate_util.return_value = True
            
            result = await service.store_market_data(sample_market_data, "binance")
            
            # Should return True on success
            assert result is True
            
            # Should have called validation
            mock_validate.assert_called_once_with(sample_market_data)
            
            # Should have stored in database
            mock_store.assert_called_once_with(mock_db_records)

    @pytest.mark.asyncio
    async def test_concurrent_storage_requests(self, mock_config, mock_database_service, sample_market_data):
        """Test handling concurrent storage requests."""
        service = DataService(config=mock_config, database_service=mock_database_service)
        
        # Mock validation and transformation
        with patch.object(service, '_validate_market_data', new_callable=AsyncMock) as mock_validate, \
             patch.object(service, '_transform_to_db_records', new_callable=AsyncMock) as mock_transform, \
             patch.object(service, '_store_to_database', new_callable=AsyncMock) as mock_store, \
             patch('src.data.services.data_service.validate_market_data') as mock_validate_util:
            
            mock_validate.return_value = sample_market_data
            mock_db_records = [Mock(spec=MarketDataRecord) for _ in sample_market_data]
            mock_transform.return_value = mock_db_records
            mock_store.return_value = None
            mock_validate_util.return_value = True
            
            # Make multiple concurrent requests
            tasks = [
                service.store_market_data(sample_market_data, f"exchange_{i}")
                for i in range(3)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed and return True
            assert len(results) == 3
            assert all(r is True for r in results)
            
            # All should have succeeded
            assert len(results) == 3


class TestDataServiceConfiguration:
    """Test DataService configuration handling."""

    def test_missing_configuration_sections(self):
        """Test handling missing configuration sections."""
        config = Mock(spec=Config)
        # Remove all optional config sections
        del config.redis
        del config.data_service
        del config.feature_store
        
        # Should not crash during initialization
        service = DataService(config=config)
        assert service.config is config

    def test_partial_configuration(self, mock_config):
        """Test handling partial configuration."""
        # Remove some optional attributes
        del mock_config.data_service["enable_stream_processing"]
        del mock_config.redis.ssl
        
        # Should handle missing attributes gracefully
        service = DataService(config=mock_config)
        assert service.config is mock_config

    def test_configuration_validation(self, mock_config):
        """Test configuration validation during setup."""
        service = DataService(config=mock_config)
        
        # Configuration should be set up during initialization
        assert service.config is mock_config
        # Cache manager should be initialized
        assert hasattr(service, 'cache_manager')


class TestDataServiceMetrics:
    """Test DataService metrics collection."""

    def test_metrics_collector_integration(self, mock_config, mock_metrics_collector):
        """Test metrics collector integration."""
        service = DataService(config=mock_config, metrics_collector=mock_metrics_collector)
        
        # Metrics collector should be available
        # This would be used throughout the service for recording metrics

    def test_metrics_without_collector(self, mock_config):
        """Test service works without metrics collector."""
        service = DataService(config=mock_config)
        
        # Should work fine without metrics collector
        assert service.config is mock_config