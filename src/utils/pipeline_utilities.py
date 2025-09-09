"""
Shared Pipeline Utilities

This module consolidates common data pipeline patterns and functionality
to eliminate duplication between various pipeline implementations in the
data module, providing shared enums, metrics, and processing patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from src.core.logging import get_logger

logger = get_logger(__name__)


class PipelineStage(Enum):
    """Common pipeline stage enumeration."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    CLEANSING = "cleansing"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    QUALITY_CHECK = "quality_check"
    STORAGE = "storage"
    INDEXING = "indexing"
    NOTIFICATION = "notification"
    COMPLETED = "completed"


class ProcessingMode(Enum):
    """Data processing mode enumeration."""

    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAM = "stream"
    HYBRID = "hybrid"


class DataQuality(Enum):
    """Data quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class PipelineAction(Enum):
    """Pipeline action enumeration."""

    ACCEPT = "accept"
    ACCEPT_WITH_WARNING = "accept_with_warning"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    RETRY = "retry"
    SKIP = "skip"


@dataclass
class PipelineMetrics:
    """Common pipeline processing metrics."""

    total_records_processed: int = 0
    successful_records: int = 0
    failed_records: int = 0
    records_rejected: int = 0
    records_quarantined: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    data_quality_score: float = 0.0
    pipeline_uptime: float = 0.0
    last_processed_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_stages_completed: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    average_quality_score: float = 0.0

    def calculate_success_rate(self) -> float:
        """Calculate processing success rate."""
        if self.total_records_processed == 0:
            return 0.0
        return self.successful_records / self.total_records_processed

    def calculate_failure_rate(self) -> float:
        """Calculate processing failure rate."""
        if self.total_records_processed == 0:
            return 0.0
        return self.failed_records / self.total_records_processed

    def update_processing_time(self, processing_time_ms: float) -> None:
        """Update average processing time with new measurement."""
        if self.total_records_processed == 0:
            self.avg_processing_time_ms = processing_time_ms
        else:
            # Use weighted average
            weight = 1.0 / (self.total_records_processed + 1)
            self.avg_processing_time_ms = (
                1 - weight
            ) * self.avg_processing_time_ms + weight * processing_time_ms

    def calculate_throughput(self, time_window_seconds: float = 1.0) -> float:
        """Calculate throughput per second."""
        if time_window_seconds <= 0:
            return 0.0

        # Estimate throughput based on average processing time
        if self.avg_processing_time_ms > 0:
            records_per_ms = 1.0 / self.avg_processing_time_ms
            self.throughput_per_second = records_per_ms * 1000

        return self.throughput_per_second


@dataclass
class PipelineRecord:
    """Common pipeline record structure."""

    id: str
    data: Any
    stage: PipelineStage
    timestamp: datetime
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0

    def add_error(self, error: str, stage: PipelineStage | None = None) -> None:
        """Add error message to record."""
        if stage:
            error = f"[{stage.value}] {error}"
        self.errors.append(error)

    def add_warning(self, warning: str, stage: PipelineStage | None = None) -> None:
        """Add warning message to record."""
        if stage:
            warning = f"[{stage.value}] {warning}"
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if record has any errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if record has any warnings."""
        return len(self.warnings) > 0


class PipelineStageProcessor(Protocol):
    """Protocol for pipeline stage processors."""

    async def process(self, record: PipelineRecord) -> PipelineRecord:
        """Process a pipeline record."""
        ...

    def get_stage_name(self) -> str:
        """Get the stage name."""
        ...


class PipelineUtils:
    """Shared pipeline utility functions."""

    @staticmethod
    def validate_pipeline_config(config: dict[str, Any]) -> list[str]:
        """
        Validate pipeline configuration and return errors.

        Args:
            config: Pipeline configuration

        Returns:
            List of validation errors
        """
        errors = []

        # Check required fields
        required_fields = ["batch_size", "max_workers", "timeout_seconds"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate batch_size
        if "batch_size" in config:
            batch_size = config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append("batch_size must be positive integer")

        # Validate max_workers
        if "max_workers" in config:
            max_workers = config["max_workers"]
            if not isinstance(max_workers, int) or max_workers <= 0:
                errors.append("max_workers must be positive integer")

        # Validate timeout_seconds
        if "timeout_seconds" in config:
            timeout = config["timeout_seconds"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append("timeout_seconds must be positive number")

        return errors

    @staticmethod
    def calculate_data_quality_score(
        record: PipelineRecord, error_weight: float = 0.5, warning_weight: float = 0.2
    ) -> float:
        """
        Calculate data quality score for a record.

        Args:
            record: Pipeline record
            error_weight: Weight for errors (0-1)
            warning_weight: Weight for warnings (0-1)

        Returns:
            Quality score (0-1)
        """
        base_score = 1.0

        # Reduce score based on errors and warnings
        error_penalty = len(record.errors) * error_weight
        warning_penalty = len(record.warnings) * warning_weight

        total_penalty = error_penalty + warning_penalty
        quality_score = max(0.0, base_score - total_penalty)

        return quality_score

    @staticmethod
    def determine_pipeline_action(
        quality_score: float,
        error_count: int,
        warning_count: int,
        min_quality_threshold: float = 0.7,
        quarantine_threshold: float = 0.5,
        max_errors: int = 0,
        max_warnings: int = 10,
    ) -> PipelineAction:
        """
        Determine what action to take based on data quality.

        Args:
            quality_score: Data quality score (0-1)
            error_count: Number of errors
            warning_count: Number of warnings
            min_quality_threshold: Minimum quality to accept
            quarantine_threshold: Minimum quality to quarantine
            max_errors: Maximum allowed errors
            max_warnings: Maximum allowed warnings

        Returns:
            Pipeline action to take
        """
        # Reject if too many critical errors
        if error_count > max_errors:
            return PipelineAction.REJECT

        # Quarantine if quality is poor but not unacceptable
        if quality_score < quarantine_threshold:
            return PipelineAction.QUARANTINE

        # Accept with warning if quality is acceptable but has issues
        if quality_score < min_quality_threshold or warning_count > max_warnings:
            return PipelineAction.ACCEPT_WITH_WARNING

        # Accept if quality is good
        return PipelineAction.ACCEPT

    @staticmethod
    def create_processing_summary(
        metrics: PipelineMetrics, duration_seconds: float
    ) -> dict[str, Any]:
        """
        Create processing summary from metrics.

        Args:
            metrics: Pipeline metrics
            duration_seconds: Total processing duration

        Returns:
            Processing summary dictionary
        """
        return {
            "total_records": metrics.total_records_processed,
            "successful_records": metrics.successful_records,
            "failed_records": metrics.failed_records,
            "rejected_records": metrics.records_rejected,
            "quarantined_records": metrics.records_quarantined,
            "success_rate": metrics.calculate_success_rate(),
            "failure_rate": metrics.calculate_failure_rate(),
            "avg_processing_time_ms": metrics.avg_processing_time_ms,
            "throughput_per_second": metrics.calculate_throughput(),
            "data_quality_score": metrics.data_quality_score,
            "processing_duration_seconds": duration_seconds,
            "critical_issues": metrics.critical_issues,
            "error_issues": metrics.error_issues,
            "warning_issues": metrics.warning_issues,
        }

    @staticmethod
    def log_pipeline_summary(
        pipeline_name: str, summary: dict[str, Any], logger_instance: Any = None
    ) -> None:
        """
        Log pipeline processing summary.

        Args:
            pipeline_name: Name of the pipeline
            summary: Processing summary
            logger_instance: Logger instance to use
        """
        log = logger_instance or logger

        log.info(
            f"{pipeline_name} Pipeline Summary: "
            f"Processed {summary['total_records']} records, "
            f"{summary['successful_records']} successful, "
            f"{summary['failed_records']} failed, "
            f"Success rate: {summary['success_rate']:.2%}, "
            f"Avg processing time: {summary['avg_processing_time_ms']:.2f}ms, "
            f"Throughput: {summary['throughput_per_second']:.1f} records/sec"
        )

        if summary["critical_issues"] > 0 or summary["error_issues"] > 0:
            log.warning(
                f"{pipeline_name} Pipeline Issues: "
                f"{summary['critical_issues']} critical, "
                f"{summary['error_issues']} errors, "
                f"{summary['warning_issues']} warnings"
            )
