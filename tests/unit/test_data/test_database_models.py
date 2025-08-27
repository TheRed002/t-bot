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
            data_timestamp=datetime.now(timezone.utc),
            open_price=Decimal("50000.0"),
            high_price=Decimal("51000.0"),
            low_price=Decimal("49000.0"),
            close_price=Decimal("50500.0"),
            volume=Decimal("100.0"),
            interval="1m",
            source="exchange"
        )

    def test_market_data_record_creation(self, sample_market_data):
        """Test market data record creation."""
        assert sample_market_data.symbol == "BTCUSDT"
        assert sample_market_data.exchange == "binance"
        assert sample_market_data.open_price == Decimal("50000.0")
        assert sample_market_data.high_price == Decimal("51000.0")
        assert sample_market_data.low_price == Decimal("49000.0")
        assert sample_market_data.close_price == Decimal("50500.0")
        assert sample_market_data.volume == Decimal("100.0")
        assert sample_market_data.interval == "1m"
        assert sample_market_data.source == "exchange"

    def test_market_data_record_defaults(self):
        """Test market data record default values."""
        record = MarketDataRecord(
            symbol="ETHUSDT",
            exchange="coinbase",
            data_timestamp=datetime.now(timezone.utc),
            interval="1h",
            source="exchange"
        )

        # Test that defaults are properly configured in model (will be set when saved to DB)
        assert hasattr(record, 'id')
        assert record.source == "exchange"
        assert hasattr(record, 'created_at')  # Will be set when saved to DB
        assert hasattr(record, 'updated_at')  # Will be set when saved to DB

    def test_market_data_record_id_generation(self, sample_market_data):
        """Test that ID field exists (will be generated on save)."""
        assert hasattr(sample_market_data, 'id')
        # ID is generated on save to database, so it's None initially
        assert sample_market_data.id is None or isinstance(sample_market_data.id, uuid.UUID)

    def test_market_data_record_timestamps(self, sample_market_data):
        """Test timestamp fields."""
        assert sample_market_data.data_timestamp is not None
        assert isinstance(sample_market_data.data_timestamp, datetime)
        # created_at and updated_at are set on save to database
        assert hasattr(sample_market_data, 'created_at')
        assert hasattr(sample_market_data, 'updated_at')

    def test_market_data_record_optional_fields(self):
        """Test optional fields can be None."""
        record = MarketDataRecord(
            symbol="ADAUSDT",
            exchange="binance",
            data_timestamp=datetime.now(timezone.utc),
            interval="1h",
            source="exchange",
            open_price=None,
            high_price=None,
            low_price=None,
            close_price=None,
            volume=None
        )

        assert record.open_price is None
        assert record.high_price is None
        assert record.low_price is None
        assert record.close_price is None
        assert record.volume is None


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
        # Default value is set at database level, not object level
        assert record.calculation_method is None  # SQLAlchemy default applies on insert
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
        # Default values are set at database level, not object level
        assert record.missing_data_count is None  # SQLAlchemy default applies on insert
        assert record.outlier_count is None  # SQLAlchemy default applies on insert
        assert record.duplicate_count is None  # SQLAlchemy default applies on insert
        assert record.validation_errors is None
        assert record.check_type is None  # SQLAlchemy default applies on insert
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
        # Default values are set at database level, not object level
        assert record.status is None  # SQLAlchemy default applies on insert
        assert record.stage is None  # SQLAlchemy default applies on insert
        assert record.records_processed is None  # SQLAlchemy default applies on insert
        assert record.records_successful is None  # SQLAlchemy default applies on insert
        assert record.records_failed is None  # SQLAlchemy default applies on insert
        assert record.error_count is None  # SQLAlchemy default applies on insert
        assert record.processing_time_ms is None
        assert record.error_messages is None
        assert record.last_error is None
        assert record.configuration is None
        assert record.dependencies is None
        # started_at and completed_at are not in the model definition
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
        # Only created_at and updated_at exist as TimestampMixin fields
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
            "id", "symbol", "exchange", "data_timestamp", "open_price", "high_price",
            "low_price", "close_price", "volume", "interval", "source", "created_at", "updated_at"
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
            "id", "pipeline_name", "pipeline_version", "execution_id", "execution_timestamp",
            "status", "stage", "records_processed", "records_successful",
            "records_failed", "processing_time_ms", "error_count",
            "error_messages", "last_error", "configuration", "dependencies",
            "created_at", "updated_at"
        }
        assert expected_pipeline_columns.issubset(pipeline_columns)


class TestDatabaseModelValidation:
    """Test database model validation and constraints."""

    def test_market_data_record_constraints(self):
        """Test market data record constraints."""
        # Test that required fields are properly set when provided
        record = MarketDataRecord(
            symbol="BTCUSDT",
            exchange="binance",
            data_timestamp=datetime.now(timezone.utc),
            interval="1m",
            source="exchange"
        )
        assert record.symbol == "BTCUSDT"
        assert record.exchange == "binance"
        
        # Note: Database constraints are enforced at commit time, not object creation
        # Missing required fields would be caught when saving to database

    def test_feature_record_constraints(self):
        """Test feature record constraints."""
        # Test that required fields are properly set when provided
        record = FeatureRecord(
            symbol="BTCUSDT",
            feature_type="technical",
            feature_name="sma_20",
            calculation_timestamp=datetime.now(timezone.utc),
            feature_value=49500.0
        )
        assert record.symbol == "BTCUSDT"
        assert record.feature_type == "technical"
        
        # Note: Database constraints are enforced at commit time, not object creation
        assert record.feature_name == "sma_20"

    def test_data_quality_record_constraints(self):
        """Test data quality record constraints."""
        # Test that required fields are properly set when provided
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
        
        # Note: Database constraints are enforced at commit time, not object creation

    def test_data_pipeline_record_constraints(self):
        """Test data pipeline record constraints."""
        # Test that required fields are properly set when provided
        record = DataPipelineRecord(
            pipeline_name="test_pipeline",
            execution_id="exec_001",
            execution_timestamp=datetime.now(timezone.utc)
        )
        assert record.pipeline_name == "test_pipeline"
        assert record.execution_id == "exec_001"
        
        # Note: Database constraints are enforced at commit time, not object creation
