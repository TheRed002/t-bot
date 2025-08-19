"""Data pipeline and quality types for the T-Bot trading system."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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