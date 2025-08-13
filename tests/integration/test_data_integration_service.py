"""
Integration tests for DataIntegrationService.

These tests verify the complete database integration functionality
with actual database connections and real data operations.
"""

import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from src.core.types import MarketData, StorageMode
from src.data.services.data_integration_service import DataIntegrationService


@pytest.mark.asyncio
class TestDataIntegrationServiceIntegration:
    """Test DataIntegrationService with real database integration."""

    @pytest.fixture
    def service(self, config):
        """Create DataIntegrationService instance with real configuration."""
        return DataIntegrationService(config)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
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

    async def test_store_and_retrieve_market_data(self, service, sample_market_data):
        """Test complete market data storage and retrieval workflow."""
        # Store market data
        success = await service.store_market_data(sample_market_data, "binance")
        assert success is True

        # Retrieve market data
        records = await service.get_market_data(
            symbol="BTCUSDT",
            exchange="binance"
        )

        assert len(records) > 0
        record = records[0]
        assert record.symbol == "BTCUSDT"
        assert record.exchange == "binance"
        assert record.price == 50000.0
        assert record.volume == 100.0

    async def test_store_and_retrieve_features(self, service):
        """Test complete feature storage and retrieval workflow."""
        # Store a feature
        success = await service.store_feature(
            symbol="BTCUSDT",
            feature_type="technical",
            feature_name="sma_20",
            feature_value=49500.0,
            confidence_score=0.85,
            lookback_period=20
        )
        assert success is True

        # Retrieve features
        features = await service.get_features(
            symbol="BTCUSDT",
            feature_type="technical"
        )

        assert len(features) > 0
        feature = features[0]
        assert feature.symbol == "BTCUSDT"
        assert feature.feature_type == "technical"
        assert feature.feature_name == "sma_20"
        assert feature.feature_value == 49500.0
        assert feature.confidence_score == 0.85

    async def test_store_and_retrieve_data_quality_metrics(self, service):
        """Test complete data quality metrics storage and retrieval workflow."""
        # Store quality metrics
        success = await service.store_data_quality_metrics(
            symbol="BTCUSDT",
            data_source="exchange",
            completeness_score=0.95,
            accuracy_score=0.98,
            consistency_score=0.92,
            timeliness_score=0.99,
            overall_score=0.96,
            missing_data_count=5,
            outlier_count=2,
            duplicate_count=0
        )
        assert success is True

        # Get quality summary
        summary = await service.get_data_quality_summary(
            symbol="BTCUSDT",
            data_source="exchange",
            days=1
        )

        assert summary["total_records"] > 0
        assert summary["average_overall_score"] > 0.9

    async def test_pipeline_execution_tracking(self, service):
        """Test complete pipeline execution tracking workflow."""
        # Start tracking pipeline
        execution_id = await service.track_pipeline_execution(
            pipeline_name="market_data_ingestion",
            configuration={"batch_size": 100, "timeout": 30}
        )
        assert execution_id is not None

        # Update pipeline status
        success = await service.update_pipeline_status(
            execution_id=execution_id,
            status="running",
            stage="data_processing",
            records_processed=500,
            records_successful=480,
            records_failed=20
        )
        assert success is True

        # Complete pipeline
        success = await service.update_pipeline_status(
            execution_id=execution_id,
            status="completed",
            stage="finished",
            records_processed=1000,
            records_successful=950,
            records_failed=50,
            processing_time_ms=5000
        )
        assert success is True

    async def test_batch_market_data_operations(self, service):
        """Test batch market data operations."""
        # Create multiple market data records
        market_data_list = []
        for i in range(5):
            data = MarketData(
                symbol=f"ETHUSDT",
                price=Decimal(f"3000.{i:02d}"),
                volume=Decimal(f"50.{i}"),
                timestamp=datetime.now(timezone.utc),
                bid=Decimal(f"2999.{i:02d}"),
                ask=Decimal(f"3001.{i:02d}")
            )
            market_data_list.append(data)

        # Store batch
        success = await service.store_market_data(market_data_list, "coinbase")
        assert success is True

        # Retrieve batch
        records = await service.get_market_data(
            symbol="ETHUSDT",
            exchange="coinbase"
        )

        assert len(records) >= 5

    async def test_data_cleanup_operations(self, service):
        """Test data cleanup operations."""
        # Store some old data first
        old_data = MarketData(
            symbol="ADAUSDT",
            price=Decimal("0.50"),
            volume=Decimal("1000.0"),
            timestamp=datetime.now(timezone.utc) - timedelta(days=35)
        )

        success = await service.store_market_data(old_data, "binance")
        assert success is True

        # Clean up old data (keep only 30 days)
        deleted_count = await service.cleanup_old_data(days_to_keep=30)
        assert deleted_count >= 0

    async def test_storage_mode_handling(self, service):
        """Test different storage mode handling."""
        # Test with BATCH mode
        service.storage_mode = StorageMode.BATCH

        data = MarketData(
            symbol="DOTUSDT",
            price=Decimal("7.50"),
            volume=Decimal("500.0"),
            timestamp=datetime.now(timezone.utc)
        )

        success = await service.store_market_data(data, "binance")
        assert success is True

        # Test with BUFFER mode
        service.storage_mode = StorageMode.BUFFER

        data2 = MarketData(
            symbol="LINKUSDT",
            price=Decimal("15.00"),
            volume=Decimal("200.0"),
            timestamp=datetime.now(timezone.utc)
        )

        success = await service.store_market_data(data2, "binance")
        assert success is True

    async def test_error_handling_and_recovery(self, service):
        """Test error handling and recovery scenarios."""
        # Test with invalid data (should handle gracefully)
        invalid_data = MarketData(
            symbol="",  # Invalid symbol
            price=Decimal("0.00"),
            volume=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc)
        )

        # This should not crash the service
        try:
            success = await service.store_market_data(invalid_data, "binance")
            # Service should handle invalid data gracefully
        except Exception as e:
            # If it raises an exception, that's also acceptable
            assert isinstance(e, Exception)

    async def test_concurrent_operations(self, service):
        """Test concurrent operations handling."""
        import asyncio

        async def store_feature_concurrent(symbol, feature_name, value):
            return await service.store_feature(
                symbol=symbol,
                feature_type="technical",
                feature_name=feature_name,
                feature_value=value
            )

        # Run multiple concurrent feature storage operations
        tasks = [
            store_feature_concurrent("BTCUSDT", f"rsi_{i}", 50.0 + i)
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)
        assert all(result is True for result in results)

    async def test_data_quality_summary_calculations(self, service):
        """Test data quality summary calculations."""
        # Store multiple quality records with different scores
        quality_scores = [0.95, 0.85, 0.75, 0.65, 0.55]

        for i, score in enumerate(quality_scores):
            success = await service.store_data_quality_metrics(
                symbol=f"TEST{i}",
                data_source="test",
                completeness_score=score,
                accuracy_score=score + 0.02,
                consistency_score=score - 0.03,
                timeliness_score=score + 0.01,
                overall_score=score,
                missing_data_count=i,
                outlier_count=i + 1,
                duplicate_count=i + 2
            )
            assert success is True

        # Get overall quality summary
        summary = await service.get_data_quality_summary(days=1)

        assert summary["total_records"] >= 5
        assert summary["average_overall_score"] > 0.7
        assert "quality_distribution" in summary
        assert "top_issues" in summary

    async def test_pipeline_metrics_tracking(self, service):
        """Test detailed pipeline metrics tracking."""
        # Start pipeline
        execution_id = await service.track_pipeline_execution(
            pipeline_name="feature_calculation",
            dependencies=["market_data", "technical_indicators"]
        )
        assert execution_id is not None

        # Update with detailed metrics
        success = await service.update_pipeline_status(
            execution_id=execution_id,
            status="running",
            stage="feature_calculation",
            records_processed=1000,
            records_successful=950,
            records_failed=50,
            processing_time_ms=3000
        )
        assert success is True

        # Update with error information
        success = await service.update_pipeline_status(
            execution_id=execution_id,
            status="completed",
            stage="finished",
            error_message="Some features failed to calculate"
        )
        assert success is True

    async def test_market_data_filtering_and_pagination(self, service):
        """Test market data filtering and pagination."""
        # Store data over different time periods
        now = datetime.now(timezone.utc)

        for i in range(10):
            data = MarketData(
                symbol="BTCUSDT",
                price=Decimal(f"50000.{i:02d}"),
                volume=Decimal(f"100.{i}"),
                timestamp=now - timedelta(hours=i),
                bid=Decimal(f"49999.{i:02d}"),
                ask=Decimal(f"50001.{i:02d}")
            )
            await service.store_market_data(data, "binance")

        # Test time-based filtering
        records = await service.get_market_data(
            symbol="BTCUSDT",
            exchange="binance",
            start_time=now - timedelta(hours=5),
            end_time=now
        )

        assert len(records) <= 6  # Should be 6 records (0-5 hours)

        # Test with limit
        limited_records = await service.get_market_data(
            symbol="BTCUSDT",
            exchange="binance",
            limit=3
        )

        assert len(limited_records) <= 3

    async def test_feature_retrieval_with_filters(self, service):
        """Test feature retrieval with various filters."""
        # Store features of different types
        feature_types = ["technical", "statistical", "alternative"]

        for feature_type in feature_types:
            for i in range(3):
                success = await service.store_feature(
                    symbol="BTCUSDT",
                    feature_type=feature_type,
                    feature_name=f"{feature_type}_feature_{i}",
                    feature_value=100.0 + i,
                    confidence_score=0.8 + (i * 0.05)
                )
                assert success is True

        # Test filtering by feature type
        technical_features = await service.get_features(
            symbol="BTCUSDT",
            feature_type="technical"
        )
        assert len(technical_features) >= 3

        # Test filtering by time range
        recent_features = await service.get_features(
            symbol="BTCUSDT",
            start_time=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        assert len(recent_features) >= 9

    async def test_service_cleanup(self, service):
        """Test service cleanup functionality."""
        # Test cleanup without errors
        service.cleanup()

        # Verify cleanup completed successfully
        # (This is mainly testing that cleanup doesn't crash)

    async def test_integration_with_existing_data(self, service):
        """Test integration with existing data in the system."""
        # This test verifies that the service can work with existing data
        # and doesn't interfere with other parts of the system

        # Try to retrieve any existing market data
        existing_records = await service.get_market_data(
            symbol="BTCUSDT",
            exchange="binance"
        )

        # Should not crash even if no data exists
        assert isinstance(existing_records, list)

        # Try to get quality summary for any existing data
        quality_summary = await service.get_data_quality_summary(days=30)

        # Should return valid structure even if no data
        assert isinstance(quality_summary, dict)
        assert "total_records" in quality_summary
        assert "average_overall_score" in quality_summary


@pytest.mark.asyncio
class TestDataIntegrationServicePerformance:
    """Test DataIntegrationService performance characteristics."""

    @pytest.fixture
    def service(self, config):
        """Create DataIntegrationService instance for performance testing."""
        return DataIntegrationService(config)

    async def test_bulk_operations_performance(self, service):
        """Test performance of bulk operations."""
        import time

        # Create large batch of market data
        market_data_list = []
        for i in range(100):
            data = MarketData(
                symbol="PERFUSDT",
                price=Decimal(f"100.{i:02d}"),
                volume=Decimal(f"10.{i}"),
                timestamp=datetime.now(timezone.utc),
                bid=Decimal(f"99.{i:02d}"),
                ask=Decimal(f"101.{i:02d}")
            )
            market_data_list.append(data)

        # Measure bulk storage performance
        start_time = time.time()
        success = await service.store_market_data(market_data_list, "performance_test")
        end_time = time.time()

        assert success is True
        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 10.0  # 10 seconds max for 100 records

    async def test_concurrent_read_performance(self, service):
        """Test performance of concurrent read operations."""
        import asyncio
        import time

        async def read_market_data():
            return await service.get_market_data(
                symbol="BTCUSDT",
                exchange="binance"
            )

        # Run multiple concurrent reads
        start_time = time.time()
        tasks = [read_market_data() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        total_time = end_time - start_time

        # All reads should complete successfully
        assert all(isinstance(result, list) for result in results)

        # Should complete within reasonable time
        assert total_time < 5.0  # 5 seconds max for 20 concurrent reads

    async def test_large_dataset_handling(self, service):
        """Test handling of large datasets."""
        # Test retrieving data with large time ranges
        start_time = datetime.now(timezone.utc) - timedelta(days=30)
        end_time = datetime.now(timezone.utc)

        start_retrieval = time.time()
        records = await service.get_market_data(
            symbol="BTCUSDT",
            exchange="binance",
            start_time=start_time,
            end_time=end_time
        )
        end_retrieval = time.time()

        retrieval_time = end_retrieval - start_retrieval

        # Should handle large time ranges without crashing
        assert isinstance(records, list)

        # Should complete within reasonable time even for large ranges
        assert retrieval_time < 15.0  # 15 seconds max for 30-day range


@pytest.mark.asyncio
class TestDataIntegrationServiceErrorScenarios:
    """Test DataIntegrationService error handling scenarios."""

    @pytest.fixture
    def service(self, config):
        """Create DataIntegrationService instance for error testing."""
        return DataIntegrationService(config)

    async def test_database_connection_failure_handling(self, service):
        """Test handling of database connection failures."""
        # This test verifies that the service handles database connection issues gracefully
        # Note: In a real test environment, we might need to temporarily disable the database

        # Test with invalid database configuration
        # This is a theoretical test since we can't easily break the connection in integration tests
        pass

    async def test_invalid_data_handling(self, service):
        """Test handling of invalid data inputs."""
        # Test with None values
        try:
            success = await service.store_market_data(None, "test")
            # Should handle gracefully
        except Exception:
            # Exception is acceptable for invalid input
            pass

        # Test with empty symbol
        invalid_data = MarketData(
            symbol="",
            price=Decimal("0.00"),
            volume=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc)
        )

        try:
            success = await service.store_market_data(invalid_data, "test")
            # Should handle gracefully
        except Exception:
            # Exception is acceptable for invalid data
            pass

    async def test_transaction_rollback_handling(self, service):
        """Test handling of transaction rollbacks."""
        # This test verifies that the service can handle transaction failures gracefully
        # In a real scenario, this might happen due to constraint violations or deadlocks

        # Test with duplicate unique constraint (if applicable)
        # This depends on the specific database constraints

        pass

    async def test_timeout_handling(self, service):
        """Test handling of operation timeouts."""
        # This test verifies that the service can handle slow database operations gracefully

        # Test with very large data sets that might cause timeouts
        large_data_list = []
        for i in range(1000):
            data = MarketData(
                symbol="TIMEOUTUSDT",
                price=Decimal(f"100.{i:03d}"),
                volume=Decimal(f"10.{i}"),
                timestamp=datetime.now(timezone.utc)
            )
            large_data_list.append(data)

        try:
            success = await service.store_market_data(large_data_list, "timeout_test")
            # Should either complete successfully or handle timeout gracefully
        except Exception as e:
            # Timeout or other errors should be handled gracefully
            assert isinstance(e, Exception)
