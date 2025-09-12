"""
Data module type definitions.

This module contains shared type definitions for the data module to avoid
circular dependencies and provide a centralized location for data types.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.core.exceptions import ValidationError
from src.data.constants import (
    MAX_CACHE_TTL_SECONDS,
    MAX_DATA_LIMIT,
    MAX_EXCHANGE_LENGTH,
    MAX_LOOKBACK_PERIOD,
    MAX_SYMBOL_LENGTH,
    MIN_CACHE_TTL_SECONDS,
    MIN_DATA_LIMIT,
    MIN_EXCHANGE_LENGTH,
    MIN_LOOKBACK_PERIOD,
    MIN_SYMBOL_LENGTH,
)


class CacheLevel(Enum):
    """Cache level enumeration."""

    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


class DataPipelineStage(Enum):
    """Data pipeline stage enumeration."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    PROCESSING = "processing"
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
    throughput_per_second: Decimal = Decimal("0.0")
    error_rate: Decimal = Decimal("0.0")
    cache_hit_rate: Decimal = Decimal("0.0")


class DataRequest(BaseModel):
    """Data request model with validation."""

    symbol: str = Field(..., min_length=MIN_SYMBOL_LENGTH, max_length=MAX_SYMBOL_LENGTH)
    exchange: str = Field(..., min_length=MIN_EXCHANGE_LENGTH, max_length=MAX_EXCHANGE_LENGTH)
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int | None = Field(None, ge=MIN_DATA_LIMIT, le=MAX_DATA_LIMIT)
    data_types: list[str] = Field(default_factory=list)
    use_cache: bool = True
    cache_ttl: int | None = Field(None, ge=MIN_CACHE_TTL_SECONDS, le=MAX_CACHE_TTL_SECONDS)
    # Add alignment fields for consistency between backtesting and data modules
    processing_mode: str = Field(default="hybrid", description="Processing mode: batch, stream, hybrid")
    message_pattern: str = Field(default="pub_sub", description="Message pattern: pub_sub, req_reply")

    @field_validator("end_time")
    @classmethod
    def validate_time_range(cls, v: datetime | None, info: Any) -> datetime | None:
        if v and hasattr(info, "data") and "start_time" in info.data and info.data["start_time"]:
            if v <= info.data["start_time"]:
                raise ValidationError("end_time must be after start_time", field_name="end_time")
        return v


class FeatureRequest(BaseModel):
    """Feature calculation request model."""

    symbol: str = Field(..., min_length=MIN_SYMBOL_LENGTH, max_length=MAX_SYMBOL_LENGTH)
    feature_types: list[str] = Field(..., min_length=1)
    lookback_period: int = Field(..., ge=MIN_LOOKBACK_PERIOD, le=MAX_LOOKBACK_PERIOD)
    parameters: dict[str, Any] = Field(default_factory=dict)
    force_recalculation: bool = False
    cache_result: bool = True
