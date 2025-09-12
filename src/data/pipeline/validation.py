"""
Pipeline Data Validation

This module provides validation for data pipeline operations:
- Schema validation for pipeline data
- Data quality checks for pipeline integrity
- Pipeline configuration validation
- Real-time validation monitoring

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.core.config import Config

# Import from P-001 core components
from src.core.types import MarketData, ValidationLevel
from src.data.quality.validation import ValidationIssue  # reuse issue structure

# Import from P-002A error handling
from src.error_handling import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution

# ValidationLevel is now imported from core.types


@dataclass
class PipelineValidationIssue:
    """Pipeline-specific validation issue"""

    field: str
    value: Any
    expected: Any
    message: str
    level: ValidationLevel
    timestamp: datetime
    pipeline_stage: str


class PipelineValidator:
    """
    Pipeline-specific data validator for data integrity and quality.

    This class provides validation specifically for data pipeline operations,
    ensuring data integrity throughout the ingestion and processing stages.
    """

    def __init__(self, config: Config):
        """Initialize pipeline validator."""
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Validation statistics
        self.stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "critical_issues": 0,
            "last_validation_time": None,
        }

        self.logger.info("PipelineValidator initialized")

    @time_execution
    async def validate_pipeline_data(
        self, data: Any, data_type: str, pipeline_stage: str
    ) -> tuple[bool, list[PipelineValidationIssue]]:
        """
        Validate data for pipeline processing.

        Args:
            data: Data to validate
            data_type: Type of data
            pipeline_stage: Current pipeline stage

        Returns:
            Tuple of (is_valid, validation_issues)
        """
        issues = []

        try:
            self.stats["total_validations"] += 1

            if data_type == "market_data":
                issues.extend(await self._validate_market_data_pipeline(data, pipeline_stage))

            # Check for critical issues
            critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
            is_valid = len(critical_issues) == 0

            if is_valid:
                self.stats["passed_validations"] += 1
            else:
                self.stats["failed_validations"] += 1
                self.stats["critical_issues"] += len(critical_issues)

            self.stats["last_validation_time"] = datetime.now(timezone.utc)

            return is_valid, issues

        except Exception as e:
            self.logger.error(f"Pipeline validation failed: {e!s}")
            self.stats["failed_validations"] += 1

            issue = PipelineValidationIssue(
                field="validation_system",
                value="exception",
                expected="successful_validation",
                message=f"Pipeline validation error: {e!s}",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                pipeline_stage=pipeline_stage,
            )
            return False, [issue]

    async def _validate_market_data_pipeline(
        self, data: MarketData, pipeline_stage: str
    ) -> list[PipelineValidationIssue]:
        """Validate market data for pipeline operations."""
        issues = []

        # Pipeline-specific validations
        if pipeline_stage == "ingestion":
            issues.extend(self._validate_ingestion_data(data))
        elif pipeline_stage == "processing":
            issues.extend(self._validate_processing_data(data))
        elif pipeline_stage == "storage":
            issues.extend(self._validate_storage_data(data))

        return issues

    def _validate_ingestion_data(self, data: MarketData) -> list[ValidationIssue]:
        """Validate data during ingestion stage."""
        issues = []

        # Check required fields for ingestion
        if not data.symbol:
            issues.append(
                PipelineValidationIssue(
                    field="symbol",
                    value=data.symbol,
                    expected="non_empty_string",
                    message="Symbol is required for ingestion",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    pipeline_stage="ingestion",
                )
            )

        if not data.timestamp:
            issues.append(
                PipelineValidationIssue(
                    field="timestamp",
                    value=data.timestamp,
                    expected="valid_datetime",
                    message="Timestamp is required for ingestion",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    pipeline_stage="ingestion",
                )
            )

        return issues

    def _validate_processing_data(self, data: MarketData) -> list[ValidationIssue]:
        """Validate data during processing stage."""
        issues = []

        # Check data consistency after processing
        if data.bid and data.ask and data.bid >= data.ask:
            issues.append(
                PipelineValidationIssue(
                    field="bid_ask_spread",
                    value=f"bid={data.bid}, ask={data.ask}",
                    expected="bid < ask",
                    message="Invalid bid/ask spread after processing",
                    level=ValidationLevel.HIGH,
                    timestamp=datetime.now(timezone.utc),
                    pipeline_stage="processing",
                )
            )

        return issues

    def _validate_storage_data(self, data: MarketData) -> list[ValidationIssue]:
        """Validate data before storage."""
        issues = []

        # Check data completeness for storage
        if not data.price:
            issues.append(
                PipelineValidationIssue(
                    field="price",
                    value=data.price,
                    expected="valid_price",
                    message="Price is required for storage",
                    level=ValidationLevel.HIGH,
                    timestamp=datetime.now(timezone.utc),
                    pipeline_stage="storage",
                )
            )

        return issues

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get pipeline validation statistics."""
        success_rate = (
            self.stats["passed_validations"] / self.stats["total_validations"]
            if self.stats["total_validations"] > 0
            else 0.0
        )

        return {
            "total_validations": self.stats["total_validations"],
            "passed_validations": self.stats["passed_validations"],
            "failed_validations": self.stats["failed_validations"],
            "success_rate": success_rate,
            "critical_issues": self.stats["critical_issues"],
            "last_validation_time": self.stats["last_validation_time"],
        }
