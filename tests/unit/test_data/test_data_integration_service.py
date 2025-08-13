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
            price=Decimal("50000.00"),
            volume=Decimal("100.0"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999.00"),
            ask=Decimal("50001.00"),
            open_price=Decimal("49900.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49800.00")
        )

    @pytest.fixture
    def service(self, mock_config):
        """Create DataIntegrationService instance with mocked dependencies."""
        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.InfluxDBClientWrapper'):
                service = DataIntegrationService(mock_config)
                service.influx_client = None  # Disable InfluxDB for unit tests
                return service

    def test_initialization(self, mock_config):
        """Test service initialization."""
        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.InfluxDBClientWrapper') as mock_influx:
                # Mock the InfluxDB client to return None on connect failure
                mock_influx.return_value.connect.side_effect = Exception("Connection failed")

                service = DataIntegrationService(mock_config)

                assert service.storage_mode == StorageMode.BATCH
                assert service.batch_size == 100
                assert service.cleanup_interval == 3600
                assert service.influx_client is None

    def test_initialization_with_defaults(self):
        """Test service initialization with default values."""
        config = MagicMock()
        config.data_storage = {}
        config.influxdb = {}

        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.InfluxDBClientWrapper'):
                service = DataIntegrationService(config)

                assert service.storage_mode == StorageMode.BATCH
                assert service.batch_size == 100
                assert service.cleanup_interval == 3600

    @pytest.mark.asyncio
    async def test_store_single_market_data(self, service, mock_market_data):
        """Test storing single market data."""
        with patch.object(service, '_store_market_data_to_postgresql', new_callable=AsyncMock) as mock_store:
            mock_store.return_value = True

            result = await service.store_market_data(mock_market_data, "binance")

            assert result is True
            mock_store.assert_called_once_with([mock_market_data], "binance")

    @pytest.mark.asyncio
    async def test_store_market_data_batch(self, service, mock_market_data):
        """Test storing batch market data."""
        market_data_list = [mock_market_data, mock_market_data]

        with patch.object(service, '_store_market_data_to_postgresql', new_callable=AsyncMock) as mock_store:
            mock_store.return_value = True

            result = await service.store_market_data(market_data_list, "binance")

            assert result is True
            mock_store.assert_called_once_with(market_data_list, "binance")

    @pytest.mark.asyncio
    async def test_store_market_data_failure(self, service, mock_market_data):
        """Test market data storage failure handling."""
        with patch.object(service, '_store_market_data_to_postgresql', new_callable=AsyncMock) as mock_store:
            mock_store.side_effect = Exception("Database error")

            result = await service.store_market_data(mock_market_data, "binance")

            assert result is False

    @pytest.mark.asyncio
    async def test_store_market_data_with_influxdb(self, mock_config):
        """Test market data storage with InfluxDB enabled."""
        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.InfluxDBClientWrapper') as mock_influx:
                mock_client = MagicMock()
                mock_influx.return_value = mock_client
                mock_client.connect.return_value = None

                service = DataIntegrationService(mock_config)
                service.influx_client = mock_client

                market_data = MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50000.00"),
                    volume=Decimal("100.0"),
                    timestamp=datetime.now(timezone.utc)
                )

                with patch.object(service, '_store_market_data_to_postgresql', new_callable=AsyncMock) as mock_store:
                    mock_store.return_value = True

                    result = await service.store_market_data(market_data, "binance")

                    assert result is True
                    mock_client.write_market_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_feature(self, service):
        """Test feature storage."""
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.create_feature_record = AsyncMock()

                result = await service.store_feature(
                    symbol="BTCUSDT",
                    feature_type="technical",
                    feature_name="sma_20",
                    feature_value=49500.0
                )

                assert result is True
                mock_db.create_feature_record.assert_called_once()

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
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.get_market_data_records = AsyncMock(return_value=[])

                result = await service.get_market_data(
                    symbol="BTCUSDT",
                    exchange="binance"
                )

                assert result == []
                mock_db.get_market_data_records.assert_called_once()

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
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.return_value.__aenter__.return_value = MagicMock()

            with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                mock_db = MagicMock()
                mock_queries.return_value = mock_db
                mock_db.delete_old_market_data = AsyncMock(return_value=10)

                result = await service.cleanup_old_data(days_to_keep=30)

                assert result == 10
                mock_db.delete_old_market_data.assert_called_once()

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
        # Test cleanup without InfluxDB client
        await service.cleanup()

        # Test cleanup with InfluxDB client
        mock_influx = MagicMock()
        service.influx_client = mock_influx

        await service.cleanup()
        mock_influx.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_storage_mode_handling(self, mock_config):
        """Test different storage mode handling."""
        with patch('src.data.services.data_integration_service.ErrorHandler'):
            with patch('src.data.services.data_integration_service.InfluxDBClientWrapper'):
                # Test BATCH mode
                config = MagicMock()
                config.data_storage = {"mode": "batch"}
                config.influxdb = {}

                service = DataIntegrationService(config)
                assert service.storage_mode == StorageMode.BATCH

                # Test BUFFER mode
                config.data_storage = {"mode": "buffer"}
                service = DataIntegrationService(config)
                assert service.storage_mode == StorageMode.BUFFER

                # Test REAL_TIME mode
                config.data_storage = {"mode": "real_time"}
                service = DataIntegrationService(config)
                assert service.storage_mode == StorageMode.REAL_TIME

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, service):
        """Test error handling integration."""
        with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
            mock_session.side_effect = Exception("Connection failed")

            result = await service.store_feature(
                symbol="BTCUSDT",
                feature_type="technical",
                feature_name="sma_20",
                feature_value=49500.0
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, service):
        """Test concurrent operations handling."""
        import asyncio

        async def store_feature_concurrent():
            with patch('src.data.services.data_integration_service.get_async_session') as mock_session:
                mock_session.return_value.__aenter__.return_value = MagicMock()

                with patch('src.data.services.data_integration_service.DatabaseQueries') as mock_queries:
                    mock_db = MagicMock()
                    mock_queries.return_value = mock_db
                    mock_db.create_feature_record = AsyncMock()

                    return await service.store_feature(
                        symbol="BTCUSDT",
                        feature_type="technical",
                        feature_name="sma_20",
                        feature_value=49500.0
                    )

        # Run multiple concurrent operations
        tasks = [store_feature_concurrent() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(result is True for result in results)
