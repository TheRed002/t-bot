"""Data pipeline and feature management models."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .base import Base, TimestampMixin


class FeatureRecord(Base, TimestampMixin):
    """Feature record model for ML feature storage."""

    __tablename__ = "feature_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20))
    feature_type = Column(String(50), nullable=False)
    feature_name = Column(String(100), nullable=False)
    calculation_timestamp = Column(DateTime(timezone=True), nullable=False)
    feature_value = Column(Float)
    confidence_score = Column(Float)
    lookback_period = Column(Integer)
    parameters = Column(JSONB)
    calculation_method = Column(String(50), default="standard")
    source_data_start = Column(DateTime(timezone=True))
    source_data_end = Column(DateTime(timezone=True))

    # Indexes
    __table_args__ = (
        Index("idx_feature_name", "feature_name"),
        Index("idx_feature_type", "feature_type"),
        Index("idx_feature_symbol", "symbol"),
        Index("idx_feature_timestamp", "calculation_timestamp"),
        Index("idx_feature_composite", "feature_name", "symbol", "calculation_timestamp"),
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

    # Quality scores
    completeness_score = Column(Float, default=0)
    accuracy_score = Column(Float, default=0)
    consistency_score = Column(Float, default=0)
    timeliness_score = Column(Float, default=0)
    overall_score = Column(Float, default=0)

    # Quality check details
    missing_data_count = Column(Integer, default=0)
    outlier_count = Column(Integer, default=0)
    duplicate_count = Column(Integer, default=0)
    validation_errors = Column(JSONB, default=[])

    # Check metadata
    check_type = Column(String(50), default="scheduled")
    data_period_start = Column(DateTime(timezone=True))
    data_period_end = Column(DateTime(timezone=True))

    # Indexes
    __table_args__ = (
        Index("idx_data_quality_source", "data_source"),
        Index("idx_data_quality_symbol", "symbol"),
        Index("idx_data_quality_score", "overall_score"),
        Index("idx_data_quality_timestamp", "quality_check_timestamp"),
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

    # Indexes
    __table_args__ = (
        Index("idx_pipeline_name", "pipeline_name"),
        Index("idx_pipeline_status", "status"),
        Index("idx_pipeline_execution", "execution_id"),
        Index("idx_pipeline_timestamp", "execution_timestamp"),
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
