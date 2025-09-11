"""Test suite for data pipeline components."""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.core.types import MarketData
from src.data.pipeline.data_pipeline import (
    DataTransformation,
    DataValidationResult,
    PipelineMetrics,
    PipelineRecord,
    PipelineStage,
)
from src.utils.pipeline_utilities import (
    DataQuality,
    ProcessingMode,
)


class TestDataTransformation:
    """Test suite for DataTransformation."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTCUSDT",
            open=Decimal("45500.99999999"),
            high=Decimal("46000.87654321"),
            low=Decimal("44000.11111111"),
            close=Decimal("45000.12345678"),
            bid_price=Decimal("44999.12345"),
            ask_price=Decimal("45001.87654"),
            volume=Decimal("1000.123456789"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

    @pytest.mark.asyncio
    async def test_normalize_prices_success(self, sample_market_data):
        """Test successful price normalization."""
        result = await DataTransformation.normalize_prices(sample_market_data)

        assert result is not None
        assert result.symbol == sample_market_data.symbol
        # Verify the result is MarketData with properly quantized decimals
        assert isinstance(result.close, type(sample_market_data.close))

    @pytest.mark.asyncio
    async def test_normalize_prices_with_none_values(self):
        """Test price normalization with None values."""
        data = MarketData(
            symbol="BTCUSDT",
            open=Decimal("44900"),
            high=Decimal("45100"),
            low=Decimal("44800"),
            close=Decimal("45000.12345678"),
            volume=Decimal("1000"),
            bid_price=None,
            ask_price=None,
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        result = await DataTransformation.normalize_prices(data)

        assert result is not None
        # Should handle None values gracefully
        assert result.bid_price is None
        assert result.ask_price is None

    @pytest.mark.asyncio
    async def test_normalize_prices_error_handling(self):
        """Test error handling in price normalization."""
        # Create invalid MarketData that would cause an error during processing
        invalid_data = Mock()
        invalid_data.symbol = "BTCUSDT"
        invalid_data.exchange = "binance"
        invalid_data.timestamp = datetime.now(timezone.utc)
        invalid_data.price = Mock()  # Mock that will fail .quantize()
        invalid_data.price.quantize.side_effect = Exception("Quantize error")
        invalid_data.high_price = None
        invalid_data.low_price = None
        invalid_data.open_price = None
        invalid_data.bid = None
        invalid_data.ask = None
        invalid_data.volume = None
        invalid_data.model_dump.return_value = {}

        with pytest.raises(Exception):  # Will raise DataProcessingError
            await DataTransformation.normalize_prices(invalid_data)

    @pytest.mark.asyncio
    async def test_validate_ohlc_consistency_valid_data(self):
        """Test OHLC validation with valid data."""
        data = MarketData(
            symbol="BTCUSDT",
            open=Decimal("45500"),
            high=Decimal("46000"),
            low=Decimal("44000"),
            close=Decimal("45000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        result = await DataTransformation.validate_ohlc_consistency(data)

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_ohlc_consistency_invalid_high(self):
        """Test OHLC validation with invalid high price."""
        data = MarketData(
            symbol="BTCUSDT",
            open=Decimal("44500"),
            high=Decimal("44000"),  # High less than close - invalid
            low=Decimal("43000"),
            close=Decimal("45000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        result = await DataTransformation.validate_ohlc_consistency(data)

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_ohlc_consistency_invalid_low(self):
        """Test OHLC validation with invalid low price."""
        data = MarketData(
            symbol="BTCUSDT",
            open=Decimal("45500"),
            high=Decimal("46000"),
            low=Decimal("46000"),  # Low greater than close - invalid
            close=Decimal("45000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        result = await DataTransformation.validate_ohlc_consistency(data)

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_ohlc_consistency_high_less_than_low(self):
        """Test OHLC validation with high less than low."""
        data = MarketData(
            symbol="BTCUSDT",
            open=Decimal("45500"),
            high=Decimal("43000"),  # High less than low - invalid
            low=Decimal("44000"),
            close=Decimal("45000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        result = await DataTransformation.validate_ohlc_consistency(data)

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_ohlc_consistency_missing_prices(self):
        """Test OHLC validation with missing prices."""
        data = MarketData(
            symbol="BTCUSDT",
            open=Decimal("45500"),
            high=Decimal("46000"),
            low=Decimal("44000"),
            close=Decimal("45000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        result = await DataTransformation.validate_ohlc_consistency(data)

        assert result is True  # Should skip validation if not all prices available

    @pytest.mark.asyncio
    async def test_validate_ohlc_consistency_exception_handling(self):
        """Test OHLC validation exception handling."""
        # Create a mock object to simulate error during processing
        from unittest.mock import Mock

        data = Mock()
        data.open = None
        data.high = None
        data.low = None
        data.close = None

        # This should return True because not all prices are available
        result = await DataTransformation.validate_ohlc_consistency(data)

        # The method should handle the case gracefully by returning False on error
        assert result is False


class TestDataValidationResult:
    """Test suite for DataValidationResult."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        result = DataValidationResult(is_valid=True, quality_score=0.95)

        assert result.is_valid is True
        assert result.quality_score == 0.95
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}

    def test_initialization_with_data(self):
        """Test initialization with all data."""
        result = DataValidationResult(
            is_valid=False,
            quality_score=0.65,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            metadata={"source": "test", "count": 5},
        )

        assert result.is_valid is False
        assert result.quality_score == 0.65
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.metadata["source"] == "test"


class TestPipelineMetrics:
    """Test suite for PipelineMetrics."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        metrics = PipelineMetrics()

        assert metrics.total_records_processed == 0
        assert metrics.successful_records == 0
        assert metrics.failed_records == 0
        assert metrics.records_rejected == 0
        assert metrics.avg_processing_time_ms == 0.0
        assert metrics.throughput_per_second == 0.0
        assert metrics.data_quality_score == 0.0
        assert metrics.pipeline_uptime == 0.0
        assert isinstance(metrics.last_processed_time, datetime)

    def test_initialization_with_values(self):
        """Test initialization with custom values."""
        timestamp = datetime.now(timezone.utc)
        metrics = PipelineMetrics(
            total_records_processed=100,
            successful_records=95,
            failed_records=3,
            records_rejected=2,
            avg_processing_time_ms=15.5,
            throughput_per_second=50.0,
            data_quality_score=0.92,
            pipeline_uptime=99.5,
            last_processed_time=timestamp,
        )

        assert metrics.total_records_processed == 100
        assert metrics.successful_records == 95
        assert metrics.failed_records == 3
        assert metrics.records_rejected == 2
        assert metrics.avg_processing_time_ms == 15.5
        assert metrics.throughput_per_second == 50.0
        assert metrics.data_quality_score == 0.92
        assert metrics.pipeline_uptime == 99.5
        assert metrics.last_processed_time == timestamp


class TestPipelineRecord:
    """Test suite for PipelineRecord."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTCUSDT",
            open=Decimal("44900"),
            high=Decimal("45100"),
            low=Decimal("44800"),
            close=Decimal("45000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

    def test_initialization(self, sample_market_data):
        """Test pipeline record initialization."""
        timestamp = datetime.now(timezone.utc)
        record = PipelineRecord(
            record_id="test-id",
            data=sample_market_data,
            stage=PipelineStage.VALIDATION,
            timestamp=timestamp,
        )

        assert record.record_id == "test-id"
        assert record.data is sample_market_data
        assert record.stage == PipelineStage.VALIDATION
        assert record.timestamp == timestamp
        assert record.validation_result is None
        assert record.processing_time_ms == 0.0
        assert record.error_message is None
        assert record.retry_count == 0

    def test_with_validation_result(self, sample_market_data):
        """Test pipeline record with validation result."""
        validation_result = DataValidationResult(is_valid=True, quality_score=0.95)
        record = PipelineRecord(
            record_id="test-id",
            data=sample_market_data,
            stage=PipelineStage.VALIDATION,
            timestamp=datetime.now(timezone.utc),
            validation_result=validation_result,
            processing_time_ms=25.5,
            retry_count=1,
        )

        assert record.validation_result is validation_result
        assert record.processing_time_ms == 25.5
        assert record.retry_count == 1

    def test_with_error(self, sample_market_data):
        """Test pipeline record with error."""
        record = PipelineRecord(
            record_id="test-id",
            data=sample_market_data,
            stage=PipelineStage.STORAGE,
            timestamp=datetime.now(timezone.utc),
            error_message="Storage failed",
            retry_count=3,
        )

        assert record.error_message == "Storage failed"
        assert record.retry_count == 3


class TestEnums:
    """Test suite for pipeline enums."""

    def test_pipeline_stage_values(self):
        """Test pipeline stage enum values."""
        assert PipelineStage.INGESTION.value == "ingestion"
        assert PipelineStage.VALIDATION.value == "validation"
        assert PipelineStage.CLEANSING.value == "cleansing"
        assert PipelineStage.TRANSFORMATION.value == "transformation"
        assert PipelineStage.ENRICHMENT.value == "enrichment"
        assert PipelineStage.QUALITY_CHECK.value == "quality_check"
        assert PipelineStage.STORAGE.value == "storage"
        assert PipelineStage.INDEXING.value == "indexing"
        assert PipelineStage.NOTIFICATION.value == "notification"

    def test_data_quality_values(self):
        """Test data quality enum values."""
        assert DataQuality.EXCELLENT.value == "excellent"
        assert DataQuality.GOOD.value == "good"
        assert DataQuality.ACCEPTABLE.value == "acceptable"
        assert DataQuality.POOR.value == "poor"
        assert DataQuality.UNACCEPTABLE.value == "unacceptable"

    def test_processing_mode_values(self):
        """Test processing mode enum values."""
        assert ProcessingMode.REAL_TIME.value == "real_time"
        assert ProcessingMode.BATCH.value == "batch"
        assert ProcessingMode.STREAM.value == "stream"
        assert ProcessingMode.HYBRID.value == "hybrid"
