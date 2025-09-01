"""Test suite for data types."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from src.data.types import (
    CacheLevel,
    DataMetrics,
    DataPipelineStage,
    DataRequest,
)


class TestCacheLevel:
    """Test suite for CacheLevel enum."""

    def test_cache_level_values(self):
        """Test cache level enum values."""
        assert CacheLevel.L1_MEMORY.value == "l1_memory"
        assert CacheLevel.L2_REDIS.value == "l2_redis"
        assert CacheLevel.L3_DATABASE.value == "l3_database"

    def test_cache_level_is_enum(self):
        """Test that CacheLevel is an enum."""
        assert issubclass(CacheLevel, Enum)

    def test_cache_level_members(self):
        """Test cache level enum members."""
        levels = list(CacheLevel)
        assert len(levels) == 3
        assert CacheLevel.L1_MEMORY in levels
        assert CacheLevel.L2_REDIS in levels
        assert CacheLevel.L3_DATABASE in levels


class TestDataPipelineStage:
    """Test suite for DataPipelineStage enum."""

    def test_data_pipeline_stage_values(self):
        """Test data pipeline stage enum values."""
        assert DataPipelineStage.INGESTION.value == "ingestion"
        assert DataPipelineStage.VALIDATION.value == "validation"
        assert DataPipelineStage.TRANSFORMATION.value == "transformation"
        assert DataPipelineStage.STORAGE.value == "storage"
        assert DataPipelineStage.INDEXING.value == "indexing"

    def test_data_pipeline_stage_is_enum(self):
        """Test that DataPipelineStage is an enum."""
        assert issubclass(DataPipelineStage, Enum)

    def test_data_pipeline_stage_members(self):
        """Test data pipeline stage enum members."""
        stages = list(DataPipelineStage)
        assert len(stages) >= 5  # At least the core stages
        assert DataPipelineStage.INGESTION in stages
        assert DataPipelineStage.VALIDATION in stages
        assert DataPipelineStage.TRANSFORMATION in stages
        assert DataPipelineStage.STORAGE in stages
        assert DataPipelineStage.INDEXING in stages


class TestDataMetrics:
    """Test suite for DataMetrics."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        metrics = DataMetrics()
        
        assert metrics.records_processed == 0
        assert metrics.records_valid == 0
        assert metrics.records_invalid == 0
        assert metrics.processing_time_ms == 0
        assert metrics.throughput_per_second == Decimal("0.0")
        assert metrics.error_rate == Decimal("0.0")
        assert metrics.cache_hit_rate == Decimal("0.0")

    def test_initialization_with_values(self):
        """Test initialization with custom values."""
        metrics = DataMetrics(
            records_processed=1000,
            records_valid=950,
            records_invalid=50,
            processing_time_ms=250,
            throughput_per_second=Decimal("100.0"),
            error_rate=Decimal("0.05"),
            cache_hit_rate=Decimal("0.85")
        )
        
        assert metrics.records_processed == 1000
        assert metrics.records_valid == 950
        assert metrics.records_invalid == 50
        assert metrics.processing_time_ms == 250
        assert metrics.throughput_per_second == Decimal("100.0")
        assert metrics.error_rate == Decimal("0.05")
        assert metrics.cache_hit_rate == Decimal("0.85")

    def test_metrics_are_numeric(self):
        """Test that all metrics are numeric types."""
        metrics = DataMetrics()
        
        assert isinstance(metrics.records_processed, int)
        assert isinstance(metrics.records_valid, int)
        assert isinstance(metrics.records_invalid, int)
        assert isinstance(metrics.processing_time_ms, int)
        assert isinstance(metrics.throughput_per_second, Decimal)
        assert isinstance(metrics.error_rate, Decimal)
        assert isinstance(metrics.cache_hit_rate, Decimal)


class TestDataRequest:
    """Test suite for DataRequest."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        request = DataRequest(
            symbol="BTCUSDT",
            exchange="binance"
        )
        
        assert request.symbol == "BTCUSDT"
        assert request.exchange == "binance"
        assert request.start_time is None
        assert request.end_time is None
        assert request.limit is None
        assert request.use_cache is True
        assert request.cache_ttl is None

    def test_initialization_full(self):
        """Test full initialization."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)
        
        request = DataRequest(
            symbol="ETHUSD",
            exchange="coinbase",
            start_time=start_time,
            end_time=end_time,
            limit=100,
            use_cache=False,
            cache_ttl=3600
        )
        
        assert request.symbol == "ETHUSD"
        assert request.exchange == "coinbase"
        assert request.start_time == start_time
        assert request.end_time == end_time
        assert request.limit == 100
        assert request.use_cache is False
        assert request.cache_ttl == 3600

    def test_symbol_validation(self):
        """Test symbol is required."""
        # Symbol is required, so this should work
        request = DataRequest(symbol="BTCUSDT", exchange="binance")
        assert request.symbol == "BTCUSDT"

    def test_limit_validation(self):
        """Test limit validation if implemented."""
        request = DataRequest(symbol="BTCUSDT", exchange="binance", limit=1000)
        assert request.limit == 1000

    def test_cache_settings(self):
        """Test cache-related settings."""
        # Test with cache enabled
        request1 = DataRequest(symbol="BTCUSDT", exchange="binance", use_cache=True, cache_ttl=7200)
        assert request1.use_cache is True
        assert request1.cache_ttl == 7200
        
        # Test with cache disabled
        request2 = DataRequest(symbol="BTCUSDT", exchange="binance", use_cache=False)
        assert request2.use_cache is False
        assert request2.cache_ttl is None

    def test_time_range_settings(self):
        """Test time range settings."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        request = DataRequest(
            symbol="BTCUSDT",
            exchange="binance",
            start_time=start,
            end_time=end
        )
        
        assert request.start_time == start
        assert request.end_time == end

    def test_exchange_setting(self):
        """Test exchange setting."""
        exchanges = ["binance", "coinbase", "okx"]
        
        for exchange in exchanges:
            request = DataRequest(symbol="BTCUSDT", exchange=exchange)
            assert request.exchange == exchange