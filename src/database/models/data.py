"""Data pipeline and feature management models."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class FeatureRecord(Base, TimestampMixin):
    """Feature record model for ML feature storage."""

    __tablename__ = "feature_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20))
    feature_type = Column(String(50), nullable=False)
    feature_name = Column(String(100), nullable=False)
    calculation_timestamp = Column(DateTime(timezone=True), nullable=False)
    feature_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))
    confidence_score: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 6))  # Confidence score (0-1)
    lookback_period = Column(Integer)
    parameters = Column(JSONB)
    calculation_method = Column(String(50), default="standard")
    source_data_start = Column(DateTime(timezone=True))
    source_data_end = Column(DateTime(timezone=True))

    # Foreign key relationship to market data
    market_data_id = Column(UUID(as_uuid=True), ForeignKey("market_data_records.id", ondelete="SET NULL"))

    # Relationships
    market_data = relationship("MarketDataRecord", back_populates="feature_records")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_feature_name", "feature_name"),
        Index("idx_feature_type", "feature_type"),
        Index("idx_feature_symbol", "symbol"),
        Index("idx_feature_timestamp", "calculation_timestamp"),
        Index("idx_feature_market_data", "market_data_id"),
        Index("idx_feature_composite", "feature_name", "symbol", "calculation_timestamp"),
        Index("idx_feature_type_symbol", "feature_type", "symbol"),  # Performance optimization
        Index("idx_feature_confidence", "confidence_score"),  # Quality filtering
        Index("idx_feature_market_data_symbol", "market_data_id", "symbol"),  # Market data relationship optimization
        Index("idx_feature_symbol_timestamp", "symbol", "calculation_timestamp"),  # Time-series queries
        Index("idx_feature_type_confidence", "feature_type", "confidence_score"),  # Quality-based feature selection
        Index(
            "idx_feature_calculation_method_timestamp",
            "calculation_method",
            "calculation_timestamp",
        ),  # Method-based analysis
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_feature_confidence_range"),
        CheckConstraint("lookback_period IS NULL OR lookback_period > 0", name="check_lookback_period_positive"),
        CheckConstraint(
            "source_data_end IS NULL OR source_data_start IS NULL OR " "source_data_end >= source_data_start",
            name="check_feature_data_period_order",
        ),
        CheckConstraint(
            "calculation_method IN ('standard', 'rolling', 'exponential', 'custom')",
            name="check_feature_calculation_method",
        ),
    )

    def __init__(self, **kwargs):
        """Initialize with auto-generated ID."""
        super().__init__(**kwargs)
        if not self.id:
            self.id = uuid.uuid4()
        # Set created_at for tests
        if not hasattr(self, "created_at") or not self.created_at:
            self.created_at = datetime.now(timezone.utc)
        if not hasattr(self, "updated_at") or not self.updated_at:
            self.updated_at = datetime.now(timezone.utc)

    def __repr__(self):
        return f"<FeatureRecord {self.feature_name}: {self.feature_value}>"


class DataQualityRecord(Base, TimestampMixin):
    """Data quality tracking model."""

    __tablename__ = "data_quality_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20))
    data_source = Column(String(100), nullable=False)
    quality_check_timestamp = Column(DateTime(timezone=True), nullable=False)

    # Foreign key relationship to market data
    market_data_id = Column(UUID(as_uuid=True), ForeignKey("market_data_records.id", ondelete="CASCADE"))

    # Quality scores
    completeness_score: Mapped[Decimal] = mapped_column(DECIMAL(8, 6), default=0)  # Completeness score (0-1)
    accuracy_score: Mapped[Decimal] = mapped_column(DECIMAL(8, 6), default=0)  # Accuracy score (0-1)
    consistency_score: Mapped[Decimal] = mapped_column(DECIMAL(8, 6), default=0)  # Consistency score (0-1)
    timeliness_score: Mapped[Decimal] = mapped_column(DECIMAL(8, 6), default=0)  # Timeliness score (0-1)
    overall_score: Mapped[Decimal] = mapped_column(DECIMAL(8, 6), default=0)  # Overall quality score (0-1)

    # Quality check details
    missing_data_count = Column(Integer, default=0)
    outlier_count = Column(Integer, default=0)
    duplicate_count = Column(Integer, default=0)
    validation_errors = Column(JSONB, default=[])

    # Check metadata
    check_type = Column(String(50), default="scheduled")
    data_period_start = Column(DateTime(timezone=True))
    data_period_end = Column(DateTime(timezone=True))

    # Relationships
    market_data = relationship("MarketDataRecord", back_populates="data_quality_records")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_data_quality_source", "data_source"),
        Index("idx_data_quality_symbol", "symbol"),
        Index("idx_data_quality_score", "overall_score"),
        Index("idx_data_quality_timestamp", "quality_check_timestamp"),
        Index("idx_data_quality_market_data", "market_data_id"),
        Index("idx_data_quality_source_symbol", "data_source", "symbol"),  # Performance optimization
        Index("idx_data_quality_period", "data_period_start", "data_period_end"),  # Period queries
        Index("idx_data_quality_market_data_score", "market_data_id", "overall_score"),  # Quality filtering
        Index("idx_data_quality_symbol_timestamp", "symbol", "quality_check_timestamp"),  # Symbol quality history
        Index("idx_data_quality_source_timestamp", "data_source", "quality_check_timestamp"),  # Source monitoring
        Index("idx_data_quality_score_timestamp", "overall_score", "quality_check_timestamp"),  # Quality trends
        CheckConstraint(
            "completeness_score >= 0 AND completeness_score <= 1",
            name="check_completeness_score_range",
        ),
        CheckConstraint("accuracy_score >= 0 AND accuracy_score <= 1", name="check_accuracy_score_range"),
        CheckConstraint(
            "consistency_score >= 0 AND consistency_score <= 1",
            name="check_consistency_score_range",
        ),
        CheckConstraint("timeliness_score >= 0 AND timeliness_score <= 1", name="check_timeliness_score_range"),
        CheckConstraint("overall_score >= 0 AND overall_score <= 1", name="check_overall_score_range"),
        CheckConstraint("missing_data_count >= 0", name="check_missing_data_count_non_negative"),
        CheckConstraint("outlier_count >= 0", name="check_outlier_count_non_negative"),
        CheckConstraint("duplicate_count >= 0", name="check_duplicate_count_non_negative"),
        CheckConstraint(
            "data_period_end IS NULL OR data_period_start IS NULL OR " "data_period_end >= data_period_start",
            name="check_quality_data_period_order",
        ),
        CheckConstraint(
            "check_type IN ('scheduled', 'triggered', 'manual', 'continuous')",
            name="check_quality_check_type",
        ),
    )

    def __init__(self, **kwargs):
        """Initialize with auto-generated ID."""
        super().__init__(**kwargs)
        if not self.id:
            self.id = uuid.uuid4()
        # Set created_at for tests
        if not hasattr(self, "created_at") or not self.created_at:
            self.created_at = datetime.now(timezone.utc)
        if not hasattr(self, "updated_at") or not self.updated_at:
            self.updated_at = datetime.now(timezone.utc)

    def __repr__(self):
        return f"<DataQualityRecord {self.data_source}: {self.overall_score:.2f}>"


class DataPipelineRecord(Base, TimestampMixin):
    """Data pipeline execution tracking model."""

    __tablename__ = "data_pipeline_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_name = Column(String(100), nullable=False)
    pipeline_version = Column(String(20), default="1.0")

    # Foreign key relationship to bot for ownership tracking
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="SET NULL"))

    # Execution details
    execution_id = Column(String(100), nullable=False)
    execution_timestamp = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(20), default="running")
    stage = Column(String(50))

    # Records tracking
    records_processed = Column(Integer, default=0)
    records_successful = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)

    # Performance metrics
    processing_time_ms = Column(Integer)

    # Error tracking
    error_count = Column(Integer, default=0)
    error_messages = Column(JSONB, default=[])
    last_error = Column(Text)

    # Pipeline metadata
    configuration = Column(JSONB, default={})
    dependencies = Column(JSONB, default=[])

    # Relationships
    bot = relationship("Bot", back_populates="data_pipeline_records")

    # Indexes and constraints
    __table_args__ = (
        Index("idx_pipeline_name", "pipeline_name"),
        Index("idx_pipeline_status", "status"),
        Index("idx_pipeline_execution", "execution_id"),
        Index("idx_pipeline_timestamp", "execution_timestamp"),
        Index("idx_pipeline_bot_id", "bot_id"),
        Index("idx_pipeline_bot_status", "bot_id", "status"),  # Bot pipeline monitoring
        Index("idx_pipeline_name_version", "pipeline_name", "pipeline_version"),  # Version management
        Index("idx_pipeline_execution_status", "execution_id", "status"),  # Execution tracking
        Index("idx_pipeline_name_timestamp", "pipeline_name", "execution_timestamp"),  # Pipeline history
        CheckConstraint("records_processed >= 0", name="check_records_processed_non_negative"),
        CheckConstraint("records_successful >= 0", name="check_records_successful_non_negative"),
        CheckConstraint("records_failed >= 0", name="check_records_failed_non_negative"),
        CheckConstraint(
            "records_successful + records_failed <= records_processed",
            name="check_records_consistency",
        ),
        CheckConstraint("processing_time_ms >= 0", name="check_processing_time_non_negative"),
        CheckConstraint("error_count >= 0", name="check_error_count_non_negative"),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'cancelled')",
            name="check_pipeline_status",
        ),
        CheckConstraint(
            r"pipeline_version ~ '^[0-9]+\.[0-9]+(\.[0-9]+)*$'",
            name="check_pipeline_version_format",
        ),
    )

    def __init__(self, **kwargs):
        """Initialize with auto-generated ID."""
        super().__init__(**kwargs)
        if not self.id:
            self.id = uuid.uuid4()
        # Set created_at for tests
        if not hasattr(self, "created_at") or not self.created_at:
            self.created_at = datetime.now(timezone.utc)
        if not hasattr(self, "updated_at") or not self.updated_at:
            self.updated_at = datetime.now(timezone.utc)

    def __repr__(self):
        return f"<DataPipelineRecord {self.pipeline_name}: {self.status}>"
