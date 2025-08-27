"""
Unit tests for DataIntegrationService.

These tests verify the data integration service functionality
without external dependencies like databases or InfluxDB.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import MarketData, StorageMode
from src.data.services.data_integration_service import DataIntegrationService


class TestDataIntegrationService:
    """Test DataIntegrationService functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.data_storage = {
            "mode": "batch",
            "batch_size": 100,
            "cleanup_interval": 3600
        }
        config.influxdb = {
            "url": "http://localhost:8086",
            "token": "test_token",
            "org": "test_org",
            "bucket": "test_bucket"
        }
        return config

    @pytest.fixture
    def mock_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("49900.00"),
            high=Decimal("50100.00"),
            low=Decimal("49800.00"),
            close=Decimal("50000.00"),
            volume=Decimal("100.0"),
            exchange="binance",
            bid_price=Decimal("49999.00"),
            ask_price=Decimal("50001.00")
        )

    @pytest.fixture
    def service(self, mock_config):
        """Create DataIntegrationService instance with mocked dependencies."""
        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.DataService') as mock_data_service:
                # Mock the DataService instance
                mock_ds_instance = AsyncMock()
                mock_ds_instance.initialize = AsyncMock()
                mock_ds_instance.store_market_data = AsyncMock(return_value=True)
                mock_ds_instance.get_market_data = AsyncMock(return_value=[])
                mock_ds_instance.cleanup_old_data = AsyncMock(return_value=0)
                mock_data_service.return_value = mock_ds_instance
                
                service = DataIntegrationService(mock_config)
                return service

    def test_initialization(self, mock_config):
        """Test service initialization."""
        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.DataService'):
                service = DataIntegrationService(mock_config)

                assert service.storage_mode == StorageMode.BATCH
                assert service.batch_size == 100
                assert service.cleanup_interval == 3600

    def test_initialization_with_defaults(self):
        """Test service initialization with default values."""
        config = MagicMock()
        config.data_storage = {}
        config.influxdb = {}

        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.DataService'):
                service = DataIntegrationService(config)

                assert service.storage_mode == StorageMode.BATCH
                assert service.batch_size == 100
                assert service.cleanup_interval == 3600

    @pytest.mark.asyncio
    async def test_store_single_market_data(self, service, mock_market_data):
        """Test storing single market data."""
        # The service fixture already mocks the DataService
        service._data_service.store_market_data.return_value = True
        
        result = await service.store_market_data(mock_market_data, "binance")

        assert result is True
        service._data_service.store_market_data.assert_called_once_with(mock_market_data, "binance", validate=True)

    @pytest.mark.asyncio
    async def test_store_market_data_batch(self, service, mock_market_data):
        """Test storing batch market data."""
        market_data_list = [mock_market_data, mock_market_data]
        
        service._data_service.store_market_data.return_value = True
        
        result = await service.store_market_data(market_data_list, "binance")

        assert result is True
        service._data_service.store_market_data.assert_called_once_with(market_data_list, "binance", validate=True)

    @pytest.mark.asyncio
    async def test_store_market_data_failure(self, service, mock_market_data):
        """Test market data storage failure handling."""
        service._data_service.store_market_data.side_effect = Exception("Database error")
        
        result = await service.store_market_data(mock_market_data, "binance")

        assert result is False

    @pytest.mark.asyncio
    async def test_store_market_data_with_influxdb(self, mock_config):
        """Test market data storage with InfluxDB enabled."""
        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.DataService') as mock_data_service:
                mock_ds_instance = AsyncMock()
                mock_ds_instance.initialize = AsyncMock()
                mock_ds_instance.store_market_data = AsyncMock(return_value=True)
                mock_data_service.return_value = mock_ds_instance
                
                service = DataIntegrationService(mock_config)

                market_data = MarketData(
                    symbol="BTCUSDT",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal("49900.00"),
                    high=Decimal("50100.00"),
                    low=Decimal("49800.00"),
                    close=Decimal("50000.00"),
                    volume=Decimal("100.0"),
                    exchange="binance"
                )

                result = await service.store_market_data(market_data, "binance")

                assert result is True
                mock_ds_instance.store_market_data.assert_called_once_with(market_data, "binance", validate=True)

    @pytest.mark.skip(reason="store_feature method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_store_feature(self, service):
        """Test feature storage."""
        pass

    @pytest.mark.skip(reason="store_feature method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_store_feature_failure(self, service):
        """Test feature storage failure handling."""
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.create_feature_record.side_effect = Exception("Database error")

                result = await service.store_feature(
                    symbol="BTCUSDT",
                    feature_type="technical",
                    feature_name="sma_20",
                    feature_value=49500.0
                )

                assert result is False

    @pytest.mark.skip(reason="store_data_quality_metrics method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_store_data_quality_metrics(self, service):
        """Test data quality metrics storage."""
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.create_data_quality_record = AsyncMock()

                result = await service.store_data_quality_metrics(
                    symbol="BTCUSDT",
                    data_source="exchange",
                    completeness_score=0.95,
                    accuracy_score=0.98,
                    consistency_score=0.92,
                    timeliness_score=0.99,
                    overall_score=0.96
                )

                assert result is True
                mock_db.create_data_quality_record.assert_called_once()

    @pytest.mark.skip(reason="track_pipeline_execution method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_track_pipeline_execution(self, service):
        """Test pipeline execution tracking."""
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.create_data_pipeline_record = AsyncMock()

                execution_id = await service.track_pipeline_execution(
                    pipeline_name="market_data_ingestion",
                    configuration={"batch_size": 100}
                )

                assert execution_id is not None
                mock_db.create_data_pipeline_record.assert_called_once()

    @pytest.mark.skip(reason="track_pipeline_execution method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_track_pipeline_execution_with_custom_id(self, service):
        """Test pipeline execution tracking with custom execution ID."""
        custom_id = "custom_exec_001"

        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.create_data_pipeline_record = AsyncMock()

                execution_id = await service.track_pipeline_execution(
                    pipeline_name="market_data_ingestion",
                    execution_id=custom_id
                )

                assert execution_id == custom_id

    @pytest.mark.skip(reason="update_pipeline_status method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_update_pipeline_status(self, service):
        """Test pipeline status update."""
        execution_id = "test_exec_001"

        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.update_data_pipeline_status = AsyncMock(return_value=True)

                result = await service.update_pipeline_status(
                    execution_id=execution_id,
                    status="completed",
                    stage="finished"
                )

                assert result is True
                mock_db.update_data_pipeline_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_market_data(self, service):
        """Test market data retrieval."""
        # The DataIntegrationService.get_market_data delegates to DataService
        service._data_service.get_market_data = AsyncMock(return_value=[])
        
        result = await service.get_market_data(
            symbol="BTCUSDT",
            exchange="binance"
        )

        assert result == []
        service._data_service.get_market_data.assert_called_once()

    @pytest.mark.skip(reason="get_features method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_get_features(self, service):
        """Test feature retrieval."""
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.get_feature_records = AsyncMock(return_value=[])

                result = await service.get_features(
                    symbol="BTCUSDT",
                    feature_type="technical"
                )

                assert result == []
                mock_db.get_feature_records.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, service):
        """Test old data cleanup."""
        # DataIntegrationService.cleanup_old_data is deprecated and returns 0
        result = await service.cleanup_old_data(days_to_keep=30)
        
        assert result == 0  # Always returns 0 as it's deprecated

    @pytest.mark.skip(reason="get_data_quality_summary method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_get_data_quality_summary(self, service):
        """Test data quality summary retrieval."""
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.get_data_quality_records = AsyncMock(return_value=[])

                result = await service.get_data_quality_summary(
                    symbol="BTCUSDT",
                    data_source="exchange",
                    days=7
                )

                assert result["total_records"] == 0
                assert result["average_overall_score"] == 0.0
                mock_db.get_data_quality_records.assert_called_once()

    @pytest.mark.skip(reason="get_data_quality_summary method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_get_data_quality_summary_with_data(self, service):
        """Test data quality summary with actual data."""
        from src.database.models import DataQualityRecord

        # Create mock quality records
        mock_records = [
            MagicMock(overall_score=0.95, missing_data_count=2, outlier_count=1, duplicate_count=0),
            MagicMock(overall_score=0.85, missing_data_count=5, outlier_count=3, duplicate_count=1),
            MagicMock(overall_score=0.75, missing_data_count=8, outlier_count=4, duplicate_count=2)
        ]

        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.get_data_quality_records = AsyncMock(return_value=mock_records)

                result = await service.get_data_quality_summary(
                    symbol="BTCUSDT",
                    data_source="exchange",
                    days=7
                )

                assert result["total_records"] == 3
                assert result["average_overall_score"] == 0.85
                assert result["quality_distribution"]["excellent"] == 1
                assert result["quality_distribution"]["good"] == 2
                assert result["quality_distribution"]["fair"] == 0
                assert result["top_issues"][0]["count"] == 15  # Total missing data

    @pytest.mark.asyncio
    async def test_cleanup(self, service):
        """Test service cleanup."""
        # DataIntegrationService.cleanup delegates to DataService
        service._data_service.cleanup = AsyncMock()
        
        await service.cleanup()
        
        service._data_service.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_storage_mode_handling(self, mock_config):
        """Test different storage mode handling."""
        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.DataService'):
                # Test BATCH mode
                config = MagicMock()
                config.data_storage = {"mode": "batch"}
                config.influxdb = {}

                service = DataIntegrationService(config)
                assert service.storage_mode == StorageMode.BATCH

                # Test STREAM mode
                config.data_storage = {"mode": "stream"}
                service = DataIntegrationService(config)
                assert service.storage_mode == StorageMode.STREAM

                # Test ARCHIVE mode
                config.data_storage = {"mode": "archive"}
                service = DataIntegrationService(config)
                assert service.storage_mode == StorageMode.ARCHIVE

    @pytest.mark.skip(reason="store_feature method no longer exists in refactored DataIntegrationService")
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, service):
        """Test error handling integration."""
        pass

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, service):
        """Test concurrent operations handling."""
        import asyncio
        
        # Test concurrent market data storage
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("49900.00"),
            high=Decimal("50100.00"),
            low=Decimal("49800.00"),
            close=Decimal("50000.00"),
            volume=Decimal("100.0"),
            exchange="binance"
        )
        
        service._data_service.store_market_data = AsyncMock(return_value=True)
        
        # Run multiple concurrent operations
        tasks = [service.store_market_data(market_data, "binance") for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(result is True for result in results)
        assert service._data_service.store_market_data.call_count == 5
