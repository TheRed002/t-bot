"""
Data module type definitions.

This module contains shared type definitions for the data module to avoid
circular dependencies and provide a centralized location for data types.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CacheLevel(Enum):
    """Cache level enumeration."""

    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


class DataPipelineStage(Enum):
    """Data pipeline stage enumeration."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    STORAGE = "storage"
    INDEXING = "indexing"


@dataclass
class DataMetrics:
    """Data processing metrics."""

    records_processed: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    processing_time_ms: int = 0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0


class DataRequest(BaseModel):
    """Data request model with validation."""

    symbol: str = Field(..., min_length=1, max_length=20)
    exchange: str = Field(..., min_length=1, max_length=50)
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int | None = Field(None, ge=1, le=10000)
    data_types: list[str] = Field(default_factory=list)
    use_cache: bool = True
    cache_ttl: int | None = Field(None, ge=1, le=86400)

    @field_validator("end_time")
    @classmethod
    def validate_time_range(cls, v, info):
        if v and hasattr(info, "data") and "start_time" in info.data and info.data["start_time"]:
            if v <= info.data["start_time"]:
                raise ValueError("end_time must be after start_time")
        return v


class FeatureRequest(BaseModel):
    """Feature calculation request model."""

    symbol: str = Field(..., min_length=1, max_length=20)
    feature_types: list[str] = Field(..., min_length=1)
    lookback_period: int = Field(..., ge=1, le=1000)
    parameters: dict[str, Any] = Field(default_factory=dict)
    force_recalculation: bool = False
    cache_result: bool = True
