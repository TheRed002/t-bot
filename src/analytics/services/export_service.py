"""
Export Service.

This service provides a proper service layer implementation for data export,
following service layer patterns and using dependency injection.
"""

from typing import Any

from src.analytics.export.data_exporter import DataExporter
from src.analytics.interfaces import ExportServiceProtocol
from src.analytics.types import (
    AnalyticsReport,
    OperationalMetrics,
    PositionMetrics,
    StrategyMetrics,
)
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, ValidationError


class ExportService(BaseService, ExportServiceProtocol):
    """
    Service layer implementation for data export.

    This service acts as a facade over the DataExporter,
    providing proper service layer abstraction and dependency injection.
    """

    def __init__(self, data_exporter: DataExporter | None = None):
        """
        Initialize the export service.

        Args:
            data_exporter: Injected data exporter engine (optional)
        """
        super().__init__()

        # Use dependency injection - data_exporter must be injected
        if data_exporter is None:
            raise ComponentError(
                "data_exporter must be injected via dependency injection",
                component="ExportService",
                operation="__init__",
                context={"missing_dependency": "data_exporter"},
            )

        self._exporter = data_exporter

        self.logger.info("ExportService initialized")

    async def export_portfolio_data(
        self, format: str = "json", include_metadata: bool = True
    ) -> str:
        """
        Export portfolio data in specified format.

        Args:
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported portfolio data

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If export fails
        """
        valid_formats = ["json", "csv", "excel"]
        if format not in valid_formats:
            raise ValidationError(
                "Invalid format parameter",
                field_name="format",
                field_value=format,
                validation_rule=f"must be one of {valid_formats}",
            )

        try:
            return await self._exporter.export_portfolio_data(format, include_metadata)
        except Exception as e:
            raise ComponentError(
                f"Failed to export portfolio data: {e}",
                component="ExportService",
                operation="export_portfolio_data",
                context={"format": format, "include_metadata": include_metadata},
            ) from e

    async def export_risk_data(self, format: str = "json", include_metadata: bool = True) -> str:
        """
        Export risk data in specified format.

        Args:
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported risk data

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If export fails
        """
        valid_formats = ["json", "csv", "excel"]
        if format not in valid_formats:
            raise ValidationError(
                "Invalid format parameter",
                field_name="format",
                field_value=format,
                validation_rule=f"must be one of {valid_formats}",
            )

        try:
            return await self._exporter.export_risk_data(format, include_metadata)
        except Exception as e:
            raise ComponentError(
                f"Failed to export risk data: {e}",
                component="ExportService",
                operation="export_risk_data",
                context={"format": format, "include_metadata": include_metadata},
            ) from e

    async def export_position_metrics(
        self,
        metrics: list[PositionMetrics],
        format: str = "json",
        include_metadata: bool = True,
    ) -> str:
        """
        Export position metrics data.

        Args:
            metrics: Position metrics to export
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported position metrics data

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If export fails
        """
        if not isinstance(metrics, list):
            raise ValidationError(
                "Invalid metrics parameter",
                field_name="metrics",
                field_value=type(metrics),
                expected_type="List[PositionMetrics]",
            )

        valid_formats = ["json", "csv", "excel"]
        if format not in valid_formats:
            raise ValidationError(
                "Invalid format parameter",
                field_name="format",
                field_value=format,
                validation_rule=f"must be one of {valid_formats}",
            )

        try:
            return await self._exporter.export_position_metrics(metrics, format, include_metadata)
        except Exception as e:
            raise ComponentError(
                f"Failed to export position metrics: {e}",
                component="ExportService",
                operation="export_position_metrics",
                context={"format": format, "metrics_count": len(metrics)},
            ) from e

    async def export_strategy_metrics(
        self,
        metrics: list[StrategyMetrics],
        format: str = "json",
        include_metadata: bool = True,
    ) -> str:
        """
        Export strategy metrics data.

        Args:
            metrics: Strategy metrics to export
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported strategy metrics data

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If export fails
        """
        if not isinstance(metrics, list):
            raise ValidationError(
                "Invalid metrics parameter",
                field_name="metrics",
                field_value=type(metrics),
                expected_type="List[StrategyMetrics]",
            )

        valid_formats = ["json", "csv", "excel"]
        if format not in valid_formats:
            raise ValidationError(
                "Invalid format parameter",
                field_name="format",
                field_value=format,
                validation_rule=f"must be one of {valid_formats}",
            )

        try:
            return await self._exporter.export_strategy_metrics(metrics, format, include_metadata)
        except Exception as e:
            raise ComponentError(
                f"Failed to export strategy metrics: {e}",
                component="ExportService",
                operation="export_strategy_metrics",
                context={"format": format, "metrics_count": len(metrics)},
            ) from e

    async def export_operational_metrics(
        self,
        metrics: OperationalMetrics,
        format: str = "json",
        include_metadata: bool = True,
    ) -> str:
        """
        Export operational metrics data.

        Args:
            metrics: Operational metrics to export
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported operational metrics data

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If export fails
        """
        if not isinstance(metrics, OperationalMetrics):
            raise ValidationError(
                "Invalid metrics parameter",
                field_name="metrics",
                field_value=type(metrics),
                expected_type="OperationalMetrics",
            )

        valid_formats = ["json", "csv", "excel"]
        if format not in valid_formats:
            raise ValidationError(
                "Invalid format parameter",
                field_name="format",
                field_value=format,
                validation_rule=f"must be one of {valid_formats}",
            )

        try:
            return await self._exporter.export_operational_metrics(
                metrics, format, include_metadata
            )
        except Exception as e:
            raise ComponentError(
                f"Failed to export operational metrics: {e}",
                component="ExportService",
                operation="export_operational_metrics",
                context={"format": format},
            ) from e

    async def export_complete_report(
        self, report: AnalyticsReport, format: str = "json", include_charts: bool = False
    ) -> str:
        """
        Export complete analytics report.

        Args:
            report: Analytics report to export
            format: Export format
            include_charts: Whether to include chart data

        Returns:
            Exported report data

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If export fails
        """
        if not isinstance(report, AnalyticsReport):
            raise ValidationError(
                "Invalid report parameter",
                field_name="report",
                field_value=type(report),
                expected_type="AnalyticsReport",
            )

        valid_formats = ["json", "excel"]
        if format not in valid_formats:
            raise ValidationError(
                "Invalid format parameter",
                field_name="format",
                field_value=format,
                validation_rule=f"must be one of {valid_formats}",
            )

        try:
            return await self._exporter.export_complete_report(report, format, include_charts)
        except Exception as e:
            raise ComponentError(
                f"Failed to export complete report: {e}",
                component="ExportService",
                operation="export_complete_report",
                context={"format": format, "include_charts": include_charts},
            ) from e

    def get_export_statistics(self) -> dict[str, Any]:
        """
        Get export usage statistics.

        Returns:
            Export statistics

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return self._exporter.get_export_statistics()
        except Exception as e:
            raise ComponentError(
                f"Failed to get export statistics: {e}",
                component="ExportService",
                operation="get_export_statistics",
            ) from e

    # Advanced export methods that delegate to the exporter

    async def export_to_prometheus(self, data: dict[str, Any]) -> bool:
        """Export data to Prometheus format."""
        try:
            return await self._exporter.export_to_prometheus(data)
        except Exception as e:
            raise ComponentError(
                f"Failed to export to Prometheus: {e}",
                component="ExportService",
                operation="export_to_prometheus",
            ) from e

    async def export_to_influxdb_line_protocol(self, data: dict[str, Any]) -> str:
        """Export data to InfluxDB line protocol format."""
        try:
            return await self._exporter.export_to_influxdb_line_protocol(data)
        except Exception as e:
            raise ComponentError(
                f"Failed to export to InfluxDB: {e}",
                component="ExportService",
                operation="export_to_influxdb_line_protocol",
            ) from e

    async def export_to_kafka(
        self, topic: str, data: dict[str, Any], kafka_config: dict[str, Any]
    ) -> bool:
        """Export data to Kafka topic."""
        try:
            return await self._exporter.export_to_kafka(topic, data, kafka_config)
        except Exception as e:
            raise ComponentError(
                f"Failed to export to Kafka: {e}",
                component="ExportService",
                operation="export_to_kafka",
                context={"topic": topic},
            ) from e

    async def export_to_rest_api(
        self,
        endpoint: str,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
        auth: dict[str, str] | None = None,
    ) -> bool:
        """Export data to REST API endpoint."""
        try:
            return await self._exporter.export_to_rest_api(endpoint, data, headers, auth)
        except Exception as e:
            raise ComponentError(
                f"Failed to export to REST API: {e}",
                component="ExportService",
                operation="export_to_rest_api",
                context={"endpoint": endpoint},
            ) from e

    async def export_regulatory_report(
        self, report_type: str, data: dict[str, Any], template: str | None = None
    ) -> str:
        """Export regulatory compliance report."""
        try:
            return await self._exporter.export_regulatory_report(report_type, data, template)
        except Exception as e:
            raise ComponentError(
                f"Failed to export regulatory report: {e}",
                component="ExportService",
                operation="export_regulatory_report",
                context={"report_type": report_type},
            ) from e
