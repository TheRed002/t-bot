"""Metrics collection utilities for analytics module."""

import time

from src.core.base import BaseComponent


class MetricsHelper(BaseComponent):
    """Centralized metrics collection utilities for analytics."""

    def __init__(self):
        super().__init__()
        self._metrics_collector = None

    def setup_metrics_collector(self, metrics_collector) -> None:
        """Setup metrics collector instance."""
        self._metrics_collector = metrics_collector

    def record_operation_metrics(
        self,
        operation_name: str,
        start_time: float,
        success: bool = True,
        additional_tags: dict[str, str] | None = None,
    ) -> None:
        """Record standard operation metrics.

        Args:
            operation_name: Name of the operation
            start_time: Start timestamp from time.time()
            success: Whether operation succeeded
            additional_tags: Additional metric tags
        """
        if not self._metrics_collector:
            return

        duration = time.time() - start_time
        tags = {"operation": operation_name, "success": str(success).lower()}

        if additional_tags:
            tags.update(additional_tags)

        self._metrics_collector.record_histogram(
            "analytics_operation_duration_seconds", duration, tags=tags
        )

        self._metrics_collector.increment("analytics_operation_total", tags=tags)

    def record_data_processing_metrics(
        self, data_type: str, records_processed: int, processing_time: float, errors_count: int = 0
    ) -> None:
        """Record data processing metrics.

        Args:
            data_type: Type of data being processed
            records_processed: Number of records processed
            processing_time: Time taken to process
            errors_count: Number of errors encountered
        """
        if not self._metrics_collector:
            return

        tags = {"data_type": data_type}

        self._metrics_collector.record_gauge(
            "analytics_records_processed", records_processed, tags=tags
        )

        self._metrics_collector.record_histogram(
            "analytics_processing_time_seconds", processing_time, tags=tags
        )

        if errors_count > 0:
            self._metrics_collector.record_gauge(
                "analytics_processing_errors", errors_count, tags=tags
            )

    def record_calculation_metrics(
        self,
        calculation_type: str,
        input_size: int,
        calculation_time: float,
        result_value: float | None = None,
    ) -> None:
        """Record calculation-specific metrics.

        Args:
            calculation_type: Type of calculation performed
            input_size: Size of input data
            calculation_time: Time taken for calculation
            result_value: Calculated result value (if numeric)
        """
        if not self._metrics_collector:
            return

        tags = {"calculation_type": calculation_type}

        self._metrics_collector.record_histogram(
            "analytics_calculation_time_seconds", calculation_time, tags=tags
        )

        self._metrics_collector.record_gauge(
            "analytics_calculation_input_size", input_size, tags=tags
        )

        if result_value is not None:
            self._metrics_collector.record_gauge(
                "analytics_calculation_result", result_value, tags=tags
            )

    def create_timing_context(
        self, operation_name: str, additional_tags: dict[str, str] | None = None
    ):
        """Create a context manager for timing operations."""
        return MetricsTimingContext(self, operation_name, additional_tags)

    def record_alert_metrics(
        self, alert_type: str, severity: str, source: str, triggered: bool = True
    ) -> None:
        """Record alert-related metrics.

        Args:
            alert_type: Type of alert
            severity: Alert severity level
            source: Source that generated the alert
            triggered: Whether alert was triggered or cleared
        """
        if not self._metrics_collector:
            return

        tags = {
            "alert_type": alert_type,
            "severity": severity,
            "source": source,
            "action": "triggered" if triggered else "cleared",
        }

        self._metrics_collector.increment("analytics_alerts_total", tags=tags)

    def record_export_metrics(
        self, export_format: str, data_size: int, export_time: float, success: bool = True
    ) -> None:
        """Record data export metrics.

        Args:
            export_format: Format of the export
            data_size: Size of exported data
            export_time: Time taken for export
            success: Whether export succeeded
        """
        if not self._metrics_collector:
            return

        tags = {"format": export_format.lower(), "success": str(success).lower()}

        self._metrics_collector.record_histogram(
            "analytics_export_time_seconds", export_time, tags=tags
        )

        self._metrics_collector.record_gauge("analytics_export_size_bytes", data_size, tags=tags)

        self._metrics_collector.increment("analytics_exports_total", tags=tags)


class MetricsTimingContext:
    """Context manager for timing operations with metrics recording."""

    def __init__(
        self,
        metrics_helper: MetricsHelper,
        operation_name: str,
        additional_tags: dict[str, str] | None = None,
    ):
        self.metrics_helper = metrics_helper
        self.operation_name = operation_name
        self.additional_tags = additional_tags or {}
        self.start_time = None
        self.success = True

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False

        self.metrics_helper.record_operation_metrics(
            self.operation_name, self.start_time, self.success, self.additional_tags
        )

    def mark_failed(self):
        """Mark the operation as failed."""
        self.success = False
