"""
DataValidator - Enterprise Data Quality and Validation System

This module implements comprehensive data validation and quality monitoring
for financial data, using consolidated validation utilities to eliminate code duplication.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models, queries, and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core import BaseComponent, Config, HealthCheckResult, HealthStatus, MarketData
from src.utils.decorators import time_execution

# Use consolidated validation utilities
from src.utils.validation.market_data_validation import (
    MarketDataValidator,
)


def _get_utc_now():
    """Get current UTC datetime."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


class MarketDataValidationResult(BaseModel):
    """Market data validation result."""

    symbol: str
    is_valid: bool
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    error_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    validation_timestamp: Any = Field(default_factory=_get_utc_now)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DataValidator(BaseComponent):
    """
    Comprehensive data validator for financial data quality assurance.

    This validator uses consolidated validation utilities to ensure consistency
    and eliminate code duplication across the data module.
    """

    def __init__(self, config: Config):
        """Initialize the data validator."""
        super().__init__()
        self.config = config

        # Initialize consolidated market data validator
        self._validator = MarketDataValidator(
            enable_precision_validation=True,
            enable_consistency_validation=True,
            enable_timestamp_validation=True,
            max_decimal_places=getattr(config, "decimal_precision", 8),
            max_future_seconds=getattr(config, "max_future_seconds", 300),
            max_age_seconds=getattr(config, "max_data_age_seconds", 3600),
        )

        # Configuration
        self._setup_configuration()
        self._initialized = False

    def _setup_configuration(self) -> None:
        """Setup validator configuration."""
        validator_config = getattr(self.config, "data_validator", {}) or {}

        self.validation_config = {
            "enable_schema_validation": validator_config.get("enable_schema", True),
            "enable_business_validation": validator_config.get("enable_business", True),
            "enable_statistical_validation": validator_config.get("enable_statistical", True),
            "enable_temporal_validation": validator_config.get("enable_temporal", True),
            "enable_regulatory_validation": validator_config.get("enable_regulatory", True),
            # Quality scoring weights
            "quality_weights": validator_config.get(
                "quality_weights",
                {
                    "completeness": 0.25,
                    "accuracy": 0.25,
                    "consistency": 0.20,
                    "timeliness": 0.15,
                    "validity": 0.10,
                    "uniqueness": 0.05,
                },
            ),
        }

    @time_execution
    async def validate_market_data(
        self, data: MarketData | list[MarketData], include_statistical: bool = True
    ) -> MarketDataValidationResult | list[MarketDataValidationResult]:
        """
        Validate market data with comprehensive checks using consolidated validator.

        Args:
            data: Market data to validate
            include_statistical: Whether to include statistical validation (legacy parameter)

        Returns:
            Validation result(s)
        """
        try:
            if isinstance(data, list):
                return await self._validate_batch(data)
            else:
                return await self._validate_single(data)

        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            raise

    async def _validate_single(self, data: MarketData) -> MarketDataValidationResult:
        """Validate single market data record using consolidated validator."""
        # Use consolidated validator
        is_valid = self._validator.validate_market_data_record(data)
        errors = self._validator.get_validation_errors()

        # Calculate quality score based on validation results
        quality_score = self._calculate_quality_score(data, errors)

        return MarketDataValidationResult(
            symbol=data.symbol,
            is_valid=is_valid,
            quality_score=quality_score,
            error_count=len(errors),
            errors=errors,
            metadata={
                "validator_type": "consolidated",
                "validation_features": {
                    "precision": self._validator.enable_precision_validation,
                    "consistency": self._validator.enable_consistency_validation,
                    "timestamp": self._validator.enable_timestamp_validation,
                },
            },
        )

    async def _validate_batch(self, data_list: list[MarketData]) -> list[MarketDataValidationResult]:
        """Validate batch of market data records using consolidated validator."""
        results = []

        for data in data_list:
            result = await self._validate_single(data)
            results.append(result)

        return results

    def _calculate_quality_score(self, data: MarketData, errors: list[str]) -> float:
        """Calculate data quality score based on validation results."""
        if not errors:
            return 1.0

        # Calculate base completeness score
        required_fields = ["symbol", "timestamp"]
        optional_fields = ["close", "open", "high", "low", "volume", "bid_price", "ask_price"]

        required_present = sum(
            1 for field in required_fields if getattr(data, field, None) is not None
        )
        optional_present = sum(
            1 for field in optional_fields if getattr(data, field, None) is not None
        )

        completeness_base = required_present / len(required_fields)
        completeness_bonus = (optional_present / len(optional_fields)) * 0.2
        completeness_score = min(1.0, completeness_base + completeness_bonus)

        # Deduct points for errors
        error_penalty = min(0.9, len(errors) * 0.1)  # Max 90% penalty
        overall_score = max(0.0, completeness_score - error_penalty)

        return overall_score

    async def add_custom_rule(self, rule_name: str, rule_config: dict[str, Any]) -> bool:
        """Add custom validation rule (placeholder for future extension)."""
        try:
            self.logger.info(f"Custom rule '{rule_name}' configuration noted: {rule_config}")
            # Custom rules would be implemented in consolidated validator
            return True
        except Exception as e:
            self.logger.error(f"Failed to add custom rule: {e}")
            return False

    async def disable_rule(self, rule_name: str) -> bool:
        """Disable validation rule (placeholder for future extension)."""
        self.logger.info(f"Rule disable request noted for: {rule_name}")
        return True

    async def enable_rule(self, rule_name: str) -> bool:
        """Enable validation rule (placeholder for future extension)."""
        self.logger.info(f"Rule enable request noted for: {rule_name}")
        return True

    async def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        return {
            "validator_type": "consolidated",
            "validation_features": {
                "precision_validation": self._validator.enable_precision_validation,
                "consistency_validation": self._validator.enable_consistency_validation,
                "timestamp_validation": self._validator.enable_timestamp_validation,
                "max_decimal_places": self._validator.max_decimal_places,
                "max_future_seconds": self._validator.max_future_seconds,
                "max_age_seconds": self._validator.max_age_seconds,
            },
            "configuration": self.validation_config,
        }

    async def health_check(self) -> HealthCheckResult:
        """Perform validator health check."""
        status = HealthStatus.HEALTHY if self._initialized else HealthStatus.DEGRADED

        details = {
            "status": "healthy" if self._initialized else "degraded",
            "initialized": self._initialized,
            "validation_stats": await self.get_validation_stats(),
            "configuration": self.validation_config,
        }

        return HealthCheckResult(
            status=status, details=details, message="DataValidator health check"
        )

    async def cleanup(self) -> None:
        """Cleanup validator resources."""
        try:
            # Reset consolidated validator
            self._validator.reset()

            self.logger.info("DataValidator cleanup completed")

        except Exception as e:
            self.logger.error(f"DataValidator cleanup error: {e}")
