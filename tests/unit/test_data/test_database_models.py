"""
Unit tests for new database models.

These tests verify the new data management models:
- MarketDataRecord
- FeatureRecord  
- DataQualityRecord
- DataPipelineRecord
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import (
    Base,
    DataPipelineRecord,
    DataQualityRecord,
    FeatureRecord,
    MarketDataRecord,
)


class TestMarketDataRecord:
    """Test MarketDataRecord model."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data record."""
        return MarketDataRecord(
            symbol="BTCUSDT",
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
            open_price=50000.0,
            high_price=51000.0,
            low_price=49000.0,
            close_price=50500.0,
            price=50500.0,
            volume=100.0,
            bid=50499.0,
            ask=50501.0,
            data_source="exchange",
            quality_score=0.95,
            validation_status="valid"
        )

    def test_market_data_record_creation(self, sample_market_data):
        """Test market data record creation."""
        assert sample_market_data.symbol == "BTCUSDT"
        assert sample_market_data.exchange == "binance"
        assert sample_market_data.open_price == 50000.0
        assert sample_market_data.high_price == 51000.0
        assert sample_market_data.low_price == 49000.0
        assert sample_market_data.close_price == 50500.0
        assert sample_market_data.price == 50500.0
        assert sample_market_data.volume == 100.0
        assert sample_market_data.bid == 50499.0
        assert sample_market_data.ask == 50501.0
        assert sample_market_data.data_source == "exchange"
        assert sample_market_data.quality_score == 0.95
        assert sample_market_data.validation_status == "valid"

    def test_market_data_record_defaults(self):
        """Test market data record default values."""
        record = MarketDataRecord(
            symbol="ETHUSDT",
            exchange="coinbase",
            timestamp=datetime.now(timezone.utc)
        )

        assert record.id is not None
        assert record.data_source == "exchange"
        assert record.quality_score is None
        assert record.validation_status == "valid"
        assert record.created_at is not None
        assert record.updated_at is not None

    def test_market_data_record_id_generation(self, sample_market_data):
        """Test that ID is automatically generated."""
        assert sample_market_data.id is not None
        assert isinstance(sample_market_data.id, str)
        assert len(sample_market_data.id) > 0

    def test_market_data_record_timestamps(self, sample_market_data):
        """Test timestamp fields."""
        assert sample_market_data.created_at is not None
        assert sample_market_data.updated_at is not None
        assert isinstance(sample_market_data.created_at, datetime)
        assert isinstance(sample_market_data.updated_at, datetime)

    def test_market_data_record_optional_fields(self):
        """Test optional fields can be None."""
        record = MarketDataRecord(
            symbol="ADAUSDT",
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
            open_price=None,
            high_price=None,
            low_price=None,
            close_price=None,
            price=None,
            volume=None,
            bid=None,
            ask=None
        )

        assert record.open_price is None
        assert record.high_price is None
        assert record.low_price is None
        assert record.close_price is None
        assert record.price is None
        assert record.volume is None
        assert record.bid is None
        assert record.ask is None


class TestFeatureRecord:
    """Test FeatureRecord model."""

    @pytest.fixture
    def sample_feature(self):
        """Create sample feature record."""
        return FeatureRecord(
            symbol="BTCUSDT",
            feature_type="technical",
            feature_name="sma_20",
            calculation_timestamp=datetime.now(timezone.utc),
            feature_value=49500.0,
            confidence_score=0.85,
            lookback_period=20,
            parameters={"window": 20, "min_periods": 1},
            calculation_method="standard",
            source_data_start=datetime.now(timezone.utc),
            source_data_end=datetime.now(timezone.utc)
        )

    def test_feature_record_creation(self, sample_feature):
        """Test feature record creation."""
        assert sample_feature.symbol == "BTCUSDT"
        assert sample_feature.feature_type == "technical"
        assert sample_feature.feature_name == "sma_20"
        assert sample_feature.feature_value == 49500.0
        assert sample_feature.confidence_score == 0.85
        assert sample_feature.lookback_period == 20
        assert sample_feature.parameters == {"window": 20, "min_periods": 1}
        assert sample_feature.calculation_method == "standard"

    def test_feature_record_defaults(self):
        """Test feature record default values."""
        record = FeatureRecord(
            symbol="ETHUSDT",
            feature_type="statistical",
            feature_name="volatility",
            calculation_timestamp=datetime.now(timezone.utc),
            feature_value=0.15
        )

        assert record.id is not None
        assert record.calculation_method == "standard"
        assert record.created_at is not None

    def test_feature_record_optional_fields(self):
        """Test optional fields can be None."""
        record = FeatureRecord(
            symbol="ADAUSDT",
            feature_type="alternative",
            feature_name="sentiment_score",
            calculation_timestamp=datetime.now(timezone.utc),
            feature_value=0.75,
            confidence_score=None,
            lookback_period=None,
            parameters=None,
            source_data_start=None,
            source_data_end=None
        )

        assert record.confidence_score is None
        assert record.lookback_period is None
        assert record.parameters is None
        assert record.source_data_start is None
        assert record.source_data_end is None


class TestDataQualityRecord:
    """Test DataQualityRecord model."""

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
            validation_errors=["missing_timestamp", "invalid_price"],
            check_type="comprehensive",
            data_period_start=datetime.now(timezone.utc),
            data_period_end=datetime.now(timezone.utc)
        )

    def test_data_quality_record_creation(self, sample_quality_record):
        """Test data quality record creation."""
        assert sample_quality_record.symbol == "BTCUSDT"
        assert sample_quality_record.data_source == "exchange"
        assert sample_quality_record.completeness_score == 0.95
        assert sample_quality_record.accuracy_score == 0.98
        assert sample_quality_record.consistency_score == 0.92
        assert sample_quality_record.timeliness_score == 0.99
        assert sample_quality_record.overall_score == 0.96
        assert sample_quality_record.missing_data_count == 5
        assert sample_quality_record.outlier_count == 2
        assert sample_quality_record.duplicate_count == 0
        assert sample_quality_record.validation_errors == ["missing_timestamp", "invalid_price"]
        assert sample_quality_record.check_type == "comprehensive"

    def test_data_quality_record_defaults(self):
        """Test data quality record default values."""
        record = DataQualityRecord(
            symbol="ETHUSDT",
            data_source="api",
            quality_check_timestamp=datetime.now(timezone.utc),
            completeness_score=0.90,
            accuracy_score=0.95,
            consistency_score=0.88,
            timeliness_score=0.97,
            overall_score=0.93
        )

        assert record.id is not None
        assert record.missing_data_count == 0
        assert record.outlier_count == 0
        assert record.duplicate_count == 0
        assert record.validation_errors is None
        assert record.check_type == "comprehensive"
        assert record.created_at is not None

    def test_data_quality_record_optional_fields(self):
        """Test optional fields can be None."""
        record = DataQualityRecord(
            symbol="ADAUSDT",
            data_source="websocket",
            quality_check_timestamp=datetime.now(timezone.utc),
            completeness_score=0.85,
            accuracy_score=0.90,
            consistency_score=0.82,
            timeliness_score=0.95,
            overall_score=0.88,
            validation_errors=None,
            data_period_start=None,
            data_period_end=None
        )

        assert record.validation_errors is None
        assert record.data_period_start is None
        assert record.data_period_end is None


class TestDataPipelineRecord:
    """Test DataPipelineRecord model."""

    @pytest.fixture
    def sample_pipeline_record(self):
        """Create sample data pipeline record."""
        return DataPipelineRecord(
            pipeline_name="market_data_ingestion",
            execution_id="exec_001",
            execution_timestamp=datetime.now(timezone.utc),
            status="running",
            stage="data_processing",
            records_processed=1000,
            records_successful=950,
            records_failed=50,
            processing_time_ms=5000,
            error_count=3,
            error_messages=["timeout_error", "connection_failed"],
            last_error="Connection timeout after 30 seconds",
            configuration={"batch_size": 100, "timeout": 30},
            dependencies=["market_data_source", "validation_service"]
        )

    def test_data_pipeline_record_creation(self, sample_pipeline_record):
        """Test data pipeline record creation."""
        assert sample_pipeline_record.pipeline_name == "market_data_ingestion"
        assert sample_pipeline_record.execution_id == "exec_001"
        assert sample_pipeline_record.status == "running"
        assert sample_pipeline_record.stage == "data_processing"
        assert sample_pipeline_record.records_processed == 1000
        assert sample_pipeline_record.records_successful == 950
        assert sample_pipeline_record.records_failed == 50
        assert sample_pipeline_record.processing_time_ms == 5000
        assert sample_pipeline_record.error_count == 3
        assert sample_pipeline_record.error_messages == ["timeout_error", "connection_failed"]
        assert sample_pipeline_record.last_error == "Connection timeout after 30 seconds"
        assert sample_pipeline_record.configuration == {"batch_size": 100, "timeout": 30}
        assert sample_pipeline_record.dependencies == ["market_data_source", "validation_service"]

    def test_data_pipeline_record_defaults(self):
        """Test data pipeline record default values."""
        record = DataPipelineRecord(
            pipeline_name="feature_calculation",
            execution_id="exec_002",
            execution_timestamp=datetime.now(timezone.utc)
        )

        assert record.id is not None
        assert record.status == "running"
        assert record.stage == "started"
        assert record.records_processed == 0
        assert record.records_successful == 0
        assert record.records_failed == 0
        assert record.error_count == 0
        assert record.processing_time_ms is None
        assert record.error_messages is None
        assert record.last_error is None
        assert record.configuration is None
        assert record.dependencies is None
        assert record.started_at is not None
        assert record.completed_at is None
        assert record.created_at is not None
        assert record.updated_at is not None

    def test_data_pipeline_record_optional_fields(self):
        """Test optional fields can be None."""
        record = DataPipelineRecord(
            pipeline_name="data_cleanup",
            execution_id="exec_003",
            execution_timestamp=datetime.now(timezone.utc),
            processing_time_ms=None,
            error_messages=None,
            last_error=None,
            configuration=None,
            dependencies=None
        )

        assert record.processing_time_ms is None
        assert record.error_messages is None
        assert record.last_error is None
        assert record.configuration is None
        assert record.dependencies is None

    def test_data_pipeline_record_timestamps(self, sample_pipeline_record):
        """Test timestamp fields."""
        assert sample_pipeline_record.started_at is not None
        assert sample_pipeline_record.completed_at is None
        assert sample_pipeline_record.created_at is not None
        assert sample_pipeline_record.updated_at is not None


class TestDatabaseModelRelationships:
    """Test relationships between database models."""

    def test_model_inheritance(self):
        """Test that all models inherit from Base."""
        assert issubclass(MarketDataRecord, Base)
        assert issubclass(FeatureRecord, Base)
        assert issubclass(DataQualityRecord, Base)
        assert issubclass(DataPipelineRecord, Base)

    def test_model_metadata(self):
        """Test model metadata and table names."""
        assert MarketDataRecord.__tablename__ == "market_data_records"
        assert FeatureRecord.__tablename__ == "feature_records"
        assert DataQualityRecord.__tablename__ == "data_quality_records"
        assert DataPipelineRecord.__tablename__ == "data_pipeline_records"

    def test_model_columns(self):
        """Test that all expected columns exist."""
        market_data_columns = {col.name for col in MarketDataRecord.__table__.columns}
        expected_market_columns = {
            "id", "symbol", "exchange", "timestamp", "open_price", "high_price",
            "low_price", "close_price", "price", "volume", "quote_volume",
            "trades_count", "bid", "ask", "bid_volume", "ask_volume",
            "data_source", "quality_score", "validation_status", "created_at", "updated_at"
        }
        assert expected_market_columns.issubset(market_data_columns)

        feature_columns = {col.name for col in FeatureRecord.__table__.columns}
        expected_feature_columns = {
            "id", "symbol", "feature_type", "feature_name", "calculation_timestamp",
            "feature_value", "confidence_score", "lookback_period", "parameters",
            "calculation_method", "source_data_start", "source_data_end", "created_at"
        }
        assert expected_feature_columns.issubset(feature_columns)

        quality_columns = {col.name for col in DataQualityRecord.__table__.columns}
        expected_quality_columns = {
            "id", "symbol", "data_source", "quality_check_timestamp",
            "completeness_score", "accuracy_score", "consistency_score",
            "timeliness_score", "overall_score", "missing_data_count",
            "outlier_count", "duplicate_count", "validation_errors",
            "check_type", "data_period_start", "data_period_end", "created_at"
        }
        assert expected_quality_columns.issubset(quality_columns)

        pipeline_columns = {col.name for col in DataPipelineRecord.__table__.columns}
        expected_pipeline_columns = {
            "id", "pipeline_name", "execution_id", "execution_timestamp",
            "status", "stage", "records_processed", "records_successful",
            "records_failed", "processing_time_ms", "error_count",
            "error_messages", "last_error", "configuration", "dependencies",
            "started_at", "completed_at", "created_at", "updated_at"
        }
        assert expected_pipeline_columns.issubset(pipeline_columns)


class TestDatabaseModelValidation:
    """Test database model validation and constraints."""

    def test_market_data_record_constraints(self):
        """Test market data record constraints."""
        # Test required fields
        with pytest.raises(Exception):
            MarketDataRecord()  # Missing required fields

        # Test valid record
        record = MarketDataRecord(
            symbol="BTCUSDT",
            exchange="binance",
            timestamp=datetime.now(timezone.utc)
        )
        assert record.symbol == "BTCUSDT"
        assert record.exchange == "binance"

    def test_feature_record_constraints(self):
        """Test feature record constraints."""
        # Test required fields
        with pytest.raises(Exception):
            FeatureRecord()  # Missing required fields

        # Test valid record
        record = FeatureRecord(
            symbol="BTCUSDT",
            feature_type="technical",
            feature_name="sma_20",
            calculation_timestamp=datetime.now(timezone.utc),
            feature_value=49500.0
        )
        assert record.symbol == "BTCUSDT"
        assert record.feature_type == "technical"
        assert record.feature_name == "sma_20"

    def test_data_quality_record_constraints(self):
        """Test data quality record constraints."""
        # Test required fields
        with pytest.raises(Exception):
            DataQualityRecord()  # Missing required fields

        # Test valid record
        record = DataQualityRecord(
            symbol="BTCUSDT",
            data_source="exchange",
            quality_check_timestamp=datetime.now(timezone.utc),
            completeness_score=0.95,
            accuracy_score=0.98,
            consistency_score=0.92,
            timeliness_score=0.99,
            overall_score=0.96
        )
        assert record.symbol == "BTCUSDT"
        assert record.data_source == "exchange"

    def test_data_pipeline_record_constraints(self):
        """Test data pipeline record constraints."""
        # Test required fields
        with pytest.raises(Exception):
            DataPipelineRecord()  # Missing required fields

        # Test valid record
        record = DataPipelineRecord(
            pipeline_name="test_pipeline",
            execution_id="exec_001",
            execution_timestamp=datetime.now(timezone.utc)
        )
        assert record.pipeline_name == "test_pipeline"
        assert record.execution_id == "exec_001"
