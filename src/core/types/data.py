"""Data pipeline and quality types for the T-Bot trading system."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QualityLevel(Enum):
    """Data quality level classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class DriftType(Enum):
    """Data drift type classification."""

    CONCEPT = "concept"
    FEATURE = "feature"
    PREDICTION = "prediction"
    LABEL = "label"
    SCHEMA = "schema"


class IngestionMode(Enum):
    """Data ingestion mode."""

    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"
    MANUAL = "manual"


class PipelineStatus(Enum):
    """Data pipeline status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    RETRYING = "retrying"


class ProcessingStep(Enum):
    """Data processing pipeline steps."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    AGGREGATION = "aggregation"
    STORAGE = "storage"
    DISTRIBUTION = "distribution"


class StorageMode(Enum):
    """Data storage mode."""

    HOT = "hot"  # Frequently accessed
    WARM = "warm"  # Occasionally accessed
    COLD = "cold"  # Rarely accessed
    BATCH = "batch"  # Batch processing mode
    STREAM = "stream"  # Stream processing mode
    ARCHIVE = "archive"  # Long-term storage


class ErrorPattern:
    """Common error patterns in data processing."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_DATA_TYPE = "invalid_data_type"
    OUT_OF_RANGE = "out_of_range"
    DUPLICATE_RECORD = "duplicate_record"
    SCHEMA_MISMATCH = "schema_mismatch"
    TIMESTAMP_ERROR = "timestamp_error"
    ENCODING_ERROR = "encoding_error"
    PARSING_ERROR = "parsing_error"


# ML-specific data types
class MLMarketData(BaseModel):
    """Market data structure for ML processing."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    """ML prediction result structure."""

    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_id: str
    predictions: list[float]
    probabilities: list[list[float]] | None = None
    confidence_scores: list[float] | None = None
    processing_time_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureSet(BaseModel):
    """Feature set for ML models."""

    feature_set_id: str
    symbol: str
    features: dict[str, Any]
    feature_names: list[str]
    computation_time_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
