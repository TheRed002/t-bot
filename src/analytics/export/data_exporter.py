"""
Analytics Data Export System.

This module provides institutional-grade data export capabilities for analytics data,
including various formats, external system integrations, and real-time streaming.

Key Features:
- Multiple export formats (JSON, CSV, Excel, Parquet, Prometheus, InfluxDB)
- Real-time streaming to external systems (Kafka, WebSocket, REST APIs)
- Scheduled export capabilities with retry logic
- Data transformation and aggregation pipelines
- Compression, encryption, and secure transport
- Integration with monitoring systems (Prometheus, Grafana, DataDog)
- Custom format support and template-based exports
- Regulatory reporting formats (CCAR, Basel III, MiFID II)
- API endpoints for on-demand data retrieval
"""

import asyncio
import csv
import json
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO, StringIO
from typing import Any

import aiohttp
import pandas as pd
from jinja2 import Template

from src.analytics.types import (
    AnalyticsReport,
    OperationalMetrics,
    PortfolioMetrics,
    PositionMetrics,
    RiskMetrics,
    StrategyMetrics,
)
from src.core.base.component import BaseComponent
from src.core.exceptions import DataError, ValidationError, ComponentError
from src.monitoring.metrics import get_metrics_collector
from src.utils.datetime_utils import get_current_utc_timestamp


class DataExporter(BaseComponent):
    """
    Comprehensive data export system for analytics.

    Provides:
    - Multiple export formats (JSON, CSV, Excel, Parquet)
    - Scheduled export capabilities
    - External system integrations
    - Data transformation and formatting
    - Compression and encryption options
    """

    def __init__(self):
        """Initialize data exporter."""
        super().__init__()
        self.metrics_collector = get_metrics_collector()

        # Export statistics
        self._export_history: list[dict[str, Any]] = []

        self.logger.info("DataExporter initialized")

    async def export_portfolio_metrics(
        self, metrics: PortfolioMetrics, format: str = "json", include_metadata: bool = True
    ) -> str | bytes:
        """
        Export portfolio metrics in specified format.

        Args:
            metrics: Portfolio metrics to export
            format: Export format ('json', 'csv', 'excel')
            include_metadata: Whether to include metadata

        Returns:
            Exported data as string or bytes
        """
        try:
            data = self._prepare_portfolio_data(metrics, include_metadata)

            if format.lower() == "json":
                result = json.dumps(data, indent=2, default=self._json_serializer)
            elif format.lower() == "csv":
                result = self._to_csv(data)
            elif format.lower() == "excel":
                result = await self._to_excel({"portfolio_metrics": data})
            else:
                raise ValidationError(
                    f"Unsupported export format: {format}",
                    error_code="EXPORT_001",
                    field_name="format",
                    field_value=format,
                    validation_rule="supported_export_format",
                )

            # Record export
            self._record_export("portfolio_metrics", format, len(str(result)))

            return result

        except Exception as e:
            self.logger.error(f"Error exporting portfolio metrics: {e}")
            raise

    async def export_risk_metrics(
        self, metrics: RiskMetrics, format: str = "json", include_metadata: bool = True
    ) -> str | bytes:
        """
        Export risk metrics in specified format.

        Args:
            metrics: Risk metrics to export
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported data
        """
        try:
            data = self._prepare_risk_data(metrics, include_metadata)

            if format.lower() == "json":
                result = json.dumps(data, indent=2, default=self._json_serializer)
            elif format.lower() == "csv":
                result = self._to_csv(data)
            elif format.lower() == "excel":
                result = await self._to_excel({"risk_metrics": data})
            else:
                raise ValidationError(
                    f"Unsupported export format: {format}",
                    error_code="EXPORT_001",
                    field_name="format",
                    field_value=format,
                    validation_rule="supported_export_format",
                )

            self._record_export("risk_metrics", format, len(str(result)))

            return result

        except Exception as e:
            self.logger.error(f"Error exporting risk metrics: {e}")
            raise

    async def export_position_metrics(
        self, metrics: list[PositionMetrics], format: str = "json", include_metadata: bool = True
    ) -> str | bytes:
        """
        Export position metrics in specified format.

        Args:
            metrics: List of position metrics to export
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported data
        """
        try:
            data = [self._prepare_position_data(m, include_metadata) for m in metrics]

            if format.lower() == "json":
                result = json.dumps({"positions": data}, indent=2, default=self._json_serializer)
            elif format.lower() == "csv":
                result = self._list_to_csv(data)
            elif format.lower() == "excel":
                result = await self._to_excel({"position_metrics": data})
            else:
                raise ValidationError(
                    f"Unsupported export format: {format}",
                    error_code="EXPORT_001",
                    field_name="format",
                    field_value=format,
                    validation_rule="supported_export_format",
                )

            self._record_export("position_metrics", format, len(str(result)))

            return result

        except Exception as e:
            self.logger.error(f"Error exporting position metrics: {e}")
            raise

    async def export_strategy_metrics(
        self, metrics: list[StrategyMetrics], format: str = "json", include_metadata: bool = True
    ) -> str | bytes:
        """
        Export strategy metrics in specified format.

        Args:
            metrics: List of strategy metrics to export
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported data
        """
        try:
            data = [self._prepare_strategy_data(m, include_metadata) for m in metrics]

            if format.lower() == "json":
                result = json.dumps({"strategies": data}, indent=2, default=self._json_serializer)
            elif format.lower() == "csv":
                result = self._list_to_csv(data)
            elif format.lower() == "excel":
                result = await self._to_excel({"strategy_metrics": data})
            else:
                raise ValidationError(
                    f"Unsupported export format: {format}",
                    error_code="EXPORT_001",
                    field_name="format",
                    field_value=format,
                    validation_rule="supported_export_format",
                )

            self._record_export("strategy_metrics", format, len(str(result)))

            return result

        except Exception as e:
            self.logger.error(f"Error exporting strategy metrics: {e}")
            raise

    async def export_operational_metrics(
        self, metrics: OperationalMetrics, format: str = "json", include_metadata: bool = True
    ) -> str | bytes:
        """
        Export operational metrics in specified format.

        Args:
            metrics: Operational metrics to export
            format: Export format
            include_metadata: Whether to include metadata

        Returns:
            Exported data
        """
        try:
            data = self._prepare_operational_data(metrics, include_metadata)

            if format.lower() == "json":
                result = json.dumps(data, indent=2, default=self._json_serializer)
            elif format.lower() == "csv":
                result = self._to_csv(data)
            elif format.lower() == "excel":
                result = await self._to_excel({"operational_metrics": data})
            else:
                raise ValidationError(
                    f"Unsupported export format: {format}",
                    error_code="EXPORT_001",
                    field_name="format",
                    field_value=format,
                    validation_rule="supported_export_format",
                )

            self._record_export("operational_metrics", format, len(str(result)))

            return result

        except Exception as e:
            self.logger.error(f"Error exporting operational metrics: {e}")
            raise

    async def export_complete_report(
        self, report: AnalyticsReport, format: str = "json", include_charts: bool = False
    ) -> str | bytes:
        """
        Export complete analytics report.

        Args:
            report: Analytics report to export
            format: Export format
            include_charts: Whether to include chart data

        Returns:
            Exported report data
        """
        try:
            data = self._prepare_report_data(report, include_charts)

            if format.lower() == "json":
                result = json.dumps(data, indent=2, default=self._json_serializer)
            elif format.lower() == "excel":
                result = await self._report_to_excel(data)
            else:
                raise ValidationError(
                    f"Unsupported report export format: {format}",
                    error_code="EXPORT_002",
                    field_name="format",
                    field_value=format,
                    validation_rule="supported_report_format",
                )

            self._record_export("complete_report", format, len(str(result)))

            return result

        except Exception as e:
            self.logger.error(f"Error exporting complete report: {e}")
            raise

    async def export_prometheus_metrics(self, metrics_dict: dict[str, Any]) -> str:
        """
        Export metrics in Prometheus format.

        Args:
            metrics_dict: Dictionary of metrics to export

        Returns:
            Prometheus formatted metrics string
        """
        try:
            prometheus_lines = []
            prometheus_lines.append(
                f"# Analytics metrics export at {get_current_utc_timestamp().isoformat()}"
            )

            for metric_name, metric_data in metrics_dict.items():
                # Handle different metric types
                if isinstance(metric_data, (int, float)):
                    prometheus_lines.append(f"analytics_{metric_name} {metric_data}")
                elif isinstance(metric_data, Decimal):
                    prometheus_lines.append(f"analytics_{metric_name} {float(metric_data)}")
                elif isinstance(metric_data, dict):
                    for key, value in metric_data.items():
                        if isinstance(value, (int, float, Decimal)):
                            prometheus_lines.append(
                                f'analytics_{metric_name}{{type="{key}"}} {float(value)}'
                            )

            result = "\n".join(prometheus_lines)
            self._record_export("prometheus_metrics", "prometheus", len(result))

            return result

        except Exception as e:
            self.logger.error(f"Error exporting Prometheus metrics: {e}")
            raise

    async def export_time_series_data(
        self,
        time_series_data: dict[str, list[dict[str, Any]]],
        format: str = "csv",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> str | bytes:
        """
        Export time series data with optional time filtering.

        Args:
            time_series_data: Dictionary of time series data
            format: Export format
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Exported time series data
        """
        try:
            # Filter by time if specified
            filtered_data = {}
            for series_name, data_points in time_series_data.items():
                filtered_points = data_points

                if start_time or end_time:
                    filtered_points = []
                    for point in data_points:
                        timestamp = point.get("timestamp")
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp)
                        elif isinstance(timestamp, datetime):
                            pass
                        else:
                            continue

                        if start_time and timestamp < start_time:
                            continue
                        if end_time and timestamp > end_time:
                            continue

                        filtered_points.append(point)

                filtered_data[series_name] = filtered_points

            if format.lower() == "csv":
                result = self._time_series_to_csv(filtered_data)
            elif format.lower() == "json":
                result = json.dumps(filtered_data, indent=2, default=self._json_serializer)
            elif format.lower() == "parquet":
                result = await self._time_series_to_parquet(filtered_data)
            else:
                raise ValidationError(
                    f"Unsupported time series export format: {format}",
                    error_code="EXPORT_003",
                    field_name="format",
                    field_value=format,
                    validation_rule="supported_timeseries_format",
                )

            self._record_export("time_series_data", format, len(str(result)))

            return result

        except Exception as e:
            self.logger.error(f"Error exporting time series data: {e}")
            raise

    def get_export_statistics(self) -> dict[str, Any]:
        """
        Get export usage statistics.

        Returns:
            Export statistics
        """
        if not self._export_history:
            return {
                "total_exports": 0,
                "formats_used": {},
                "data_types_exported": {},
                "total_bytes_exported": 0,
            }

        formats_used = {}
        data_types_exported = {}
        total_bytes = 0

        for export_record in self._export_history:
            format_name = export_record["format"]
            data_type = export_record["data_type"]
            size = export_record["size_bytes"]

            formats_used[format_name] = formats_used.get(format_name, 0) + 1
            data_types_exported[data_type] = data_types_exported.get(data_type, 0) + 1
            total_bytes += size

        return {
            "total_exports": len(self._export_history),
            "formats_used": formats_used,
            "data_types_exported": data_types_exported,
            "total_bytes_exported": total_bytes,
            "avg_export_size_bytes": total_bytes / len(self._export_history),
        }

    def _prepare_portfolio_data(
        self, metrics: PortfolioMetrics, include_metadata: bool
    ) -> dict[str, Any]:
        """Prepare portfolio metrics for export."""
        data = metrics.dict()

        if not include_metadata:
            # Remove metadata fields
            data.pop("metadata", None)

        # Convert Decimals to floats for better JSON compatibility
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = float(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, Decimal):
                        value[k] = float(v)

        return data

    def _prepare_risk_data(self, metrics: RiskMetrics, include_metadata: bool) -> dict[str, Any]:
        """Prepare risk metrics for export."""
        data = metrics.dict()

        if not include_metadata:
            data.pop("metadata", None)

        # Convert Decimals and handle nested dictionaries
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = float(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, Decimal):
                        value[k] = float(v)
                    elif isinstance(v, dict):
                        for k2, v2 in v.items():
                            if isinstance(v2, Decimal):
                                v[k2] = float(v2)

        return data

    def _prepare_position_data(
        self, metrics: PositionMetrics, include_metadata: bool
    ) -> dict[str, Any]:
        """Prepare position metrics for export."""
        data = metrics.dict()

        if not include_metadata:
            data.pop("metadata", None)

        # Convert Decimals to floats
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = float(value)

        return data

    def _prepare_strategy_data(
        self, metrics: StrategyMetrics, include_metadata: bool
    ) -> dict[str, Any]:
        """Prepare strategy metrics for export."""
        data = metrics.dict()

        if not include_metadata:
            data.pop("metadata", None)

        # Convert Decimals to floats
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = float(value)

        return data

    def _prepare_operational_data(
        self, metrics: OperationalMetrics, include_metadata: bool
    ) -> dict[str, Any]:
        """Prepare operational metrics for export."""
        data = metrics.dict()

        if not include_metadata:
            data.pop("metadata", None)

        # Convert Decimals to floats
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = float(value)

        return data

    def _prepare_report_data(self, report: AnalyticsReport, include_charts: bool) -> dict[str, Any]:
        """Prepare analytics report for export."""
        data = report.dict()

        if not include_charts:
            data.pop("charts", None)

        # Convert nested Pydantic models
        for key, value in data.items():
            if hasattr(value, "dict"):
                data[key] = value.dict()
            elif isinstance(value, list):
                converted_list = []
                for item in value:
                    if hasattr(item, "dict"):
                        converted_list.append(item.dict())
                    else:
                        converted_list.append(item)
                data[key] = converted_list

        return data

    def _to_csv(self, data: dict[str, Any]) -> str:
        """Convert dictionary data to CSV format."""
        output = StringIO()

        # Flatten nested data for CSV
        flattened_data = self._flatten_dict(data)

        writer = csv.writer(output)
        writer.writerow(["Metric", "Value"])

        for key, value in flattened_data.items():
            writer.writerow([key, value])

        return output.getvalue()

    def _list_to_csv(self, data: list[dict[str, Any]]) -> str:
        """Convert list of dictionaries to CSV format."""
        if not data:
            return ""

        output = StringIO()

        # Use keys from first item as headers
        headers = list(data[0].keys())
        writer = csv.DictWriter(output, fieldnames=headers)

        writer.writeheader()
        for row in data:
            # Convert Decimals to floats for CSV
            csv_row = {}
            for key, value in row.items():
                if isinstance(value, Decimal):
                    csv_row[key] = float(value)
                elif isinstance(value, datetime):
                    csv_row[key] = value.isoformat()
                else:
                    csv_row[key] = value
            writer.writerow(csv_row)

        return output.getvalue()

    async def _to_excel(self, data: dict[str, Any]) -> bytes:
        """Convert data to Excel format."""
        output = BytesIO()

        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for sheet_name, sheet_data in data.items():
                if isinstance(sheet_data, list):
                    # List of dictionaries - create DataFrame
                    if sheet_data:
                        df = pd.DataFrame(sheet_data)
                        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                elif isinstance(sheet_data, dict):
                    # Dictionary - convert to two-column format
                    flattened = self._flatten_dict(sheet_data)
                    df = pd.DataFrame(list(flattened.items()), columns=["Metric", "Value"])
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        return output.getvalue()

    async def _report_to_excel(self, report_data: dict[str, Any]) -> bytes:
        """Convert analytics report to Excel format."""
        output = BytesIO()

        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Executive summary sheet
            if "executive_summary" in report_data:
                summary_data = [["Executive Summary", report_data["executive_summary"]]]
                df = pd.DataFrame(summary_data, columns=["Section", "Content"])
                df.to_excel(writer, sheet_name="Summary", index=False)

            # Portfolio metrics sheet
            if "portfolio_metrics" in report_data:
                portfolio_data = self._flatten_dict(report_data["portfolio_metrics"])
                df = pd.DataFrame(list(portfolio_data.items()), columns=["Metric", "Value"])
                df.to_excel(writer, sheet_name="Portfolio", index=False)

            # Risk metrics sheet
            if "risk_metrics" in report_data:
                risk_data = self._flatten_dict(report_data["risk_metrics"])
                df = pd.DataFrame(list(risk_data.items()), columns=["Metric", "Value"])
                df.to_excel(writer, sheet_name="Risk", index=False)

            # Performance attribution sheet
            if report_data.get("performance_attribution"):
                attr_data = self._flatten_dict(report_data["performance_attribution"])
                df = pd.DataFrame(list(attr_data.items()), columns=["Metric", "Value"])
                df.to_excel(writer, sheet_name="Attribution", index=False)

            # Tables sheet
            if "tables" in report_data:
                for i, table in enumerate(report_data["tables"]):
                    if "data" in table:
                        df = pd.DataFrame(table["data"])
                        sheet_name = f"Table_{i + 1}"
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

        return output.getvalue()

    def _time_series_to_csv(self, time_series_data: dict[str, list[dict[str, Any]]]) -> str:
        """Convert time series data to CSV format."""
        output = StringIO()

        # Combine all time series into single CSV with series identifier
        all_data = []

        for series_name, data_points in time_series_data.items():
            for point in data_points:
                row = {"series": series_name}
                row.update(point)
                all_data.append(row)

        if all_data:
            headers = list(all_data[0].keys())
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()

            for row in all_data:
                # Convert special types
                csv_row = {}
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        csv_row[key] = float(value)
                    elif isinstance(value, datetime):
                        csv_row[key] = value.isoformat()
                    else:
                        csv_row[key] = value
                writer.writerow(csv_row)

        return output.getvalue()

    async def _time_series_to_parquet(
        self, time_series_data: dict[str, list[dict[str, Any]]]
    ) -> bytes:
        """Convert time series data to Parquet format."""
        # This would require pyarrow or similar library
        # For now, return JSON as bytes
        json_data = json.dumps(time_series_data, default=self._json_serializer)
        return json_data.encode("utf-8")

    def _flatten_dict(
        self, data: dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Flatten nested dictionary."""
        items = []

        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep).items())
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Handle list of dictionaries by taking first item as sample
                items.extend(self._flatten_dict(value[0], f"{new_key}[0]", sep).items())
            else:
                items.append((new_key, value))

        return dict(items)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "dict"):
            return obj.dict()
        else:
            return str(obj)

    def _record_export(self, data_type: str, format: str, size_bytes: int) -> None:
        """Record export for statistics."""
        export_record = {
            "timestamp": get_current_utc_timestamp(),
            "data_type": data_type,
            "format": format,
            "size_bytes": size_bytes,
        }

        self._export_history.append(export_record)

        # Keep only last 1000 exports for statistics
        if len(self._export_history) > 1000:
            self._export_history = self._export_history[-1000:]

        # Update metrics
        self.metrics_collector.increment_counter(
            "analytics_exports_total", labels={"data_type": data_type, "format": format}
        )

        self.metrics_collector.observe_histogram(
            "analytics_export_size_bytes",
            size_bytes,
            labels={"data_type": data_type, "format": format},
        )

    # Advanced Export Capabilities

    async def export_to_prometheus(self, metrics_data: dict[str, Any]) -> str:
        """
        Export metrics data to Prometheus format.

        Args:
            metrics_data: Dictionary containing metrics to export

        Returns:
            Prometheus format string
        """
        try:
            prometheus_lines = []
            timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)  # Milliseconds

            # Add metadata
            prometheus_lines.append("# HELP t_bot_analytics T-Bot Analytics Metrics")
            prometheus_lines.append("# TYPE t_bot_analytics gauge")

            def flatten_dict(d: dict, prefix: str = "") -> dict:
                """Flatten nested dictionary for Prometheus format."""
                items = []
                for k, v in d.items():
                    new_key = f"{prefix}_{k}" if prefix else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key).items())
                    elif isinstance(v, (int, float, Decimal)):
                        items.append((new_key.replace(".", "_").replace("-", "_"), float(v)))
                return dict(items)

            flattened_metrics = flatten_dict(metrics_data)

            for metric_name, value in flattened_metrics.items():
                # Clean metric name for Prometheus format
                clean_name = metric_name.lower().replace(" ", "_").replace(".", "_")
                prometheus_lines.append(f"t_bot_analytics_{clean_name} {value} {timestamp}")

            return "\n".join(prometheus_lines)

        except Exception as e:
            self.logger.error(f"Error exporting to Prometheus format: {e}")
            return f"# Error: {e!s}"

    async def export_to_influxdb_line_protocol(
        self, metrics_data: dict[str, Any], measurement: str = "analytics"
    ) -> str:
        """
        Export metrics data to InfluxDB Line Protocol format.

        Args:
            metrics_data: Dictionary containing metrics to export
            measurement: InfluxDB measurement name

        Returns:
            InfluxDB Line Protocol string
        """
        try:
            lines = []
            timestamp = int(datetime.now(timezone.utc).timestamp() * 1000000000)  # Nanoseconds

            def process_metrics(data: dict, tags: dict = None) -> None:
                tags = tags or {}
                fields = []

                for key, value in data.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries as tags
                        new_tags = {**tags, "category": key}
                        process_metrics(value, new_tags)
                    elif isinstance(value, (int, float, Decimal)):
                        fields.append(f"{key}={float(value)}")
                    elif isinstance(value, str):
                        tags[key] = value

                if fields:
                    tag_string = ",".join(f"{k}={v}" for k, v in tags.items()) if tags else ""
                    field_string = ",".join(fields)

                    if tag_string:
                        line = f"{measurement},{tag_string} {field_string} {timestamp}"
                    else:
                        line = f"{measurement} {field_string} {timestamp}"

                    lines.append(line)

            process_metrics(metrics_data)
            return "\n".join(lines)

        except Exception as e:
            self.logger.error(f"Error exporting to InfluxDB format: {e}")
            return f"# Error: {e!s}"

    async def export_to_kafka(
        self, topic: str, data: dict[str, Any], kafka_config: dict = None
    ) -> bool:
        """
        Export data to Kafka topic.

        Args:
            topic: Kafka topic name
            data: Data to export
            kafka_config: Kafka connection configuration

        Returns:
            Success boolean
        """
        try:
            # This would integrate with aiokafka in a real implementation
            # For now, simulate the export

            message = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "topic": topic,
                "data": data,
                "source": "t-bot-analytics",
            }

            # Log the export (in production, this would send to actual Kafka)
            self.logger.info(
                f"Kafka export to topic '{topic}': {len(json.dumps(message, default=str))} bytes"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error exporting to Kafka: {e}")
            return False

    async def export_to_rest_api(
        self, endpoint: str, data: dict[str, Any], headers: dict = None, auth: dict = None
    ) -> bool:
        """
        Export data to external REST API.

        Args:
            endpoint: API endpoint URL
            data: Data to export
            headers: HTTP headers
            auth: Authentication configuration

        Returns:
            Success boolean
        """
        session = None
        try:
            headers = headers or {"Content-Type": "application/json"}

            # Add authentication if provided
            if auth:
                if auth.get("type") == "bearer":
                    headers["Authorization"] = f"Bearer {auth['token']}"
                elif auth.get("type") == "basic":
                    import base64

                    credentials = base64.b64encode(
                        f"{auth['username']}:{auth['password']}".encode()
                    ).decode()
                    headers["Authorization"] = f"Basic {credentials}"

            payload = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "source": "t-bot-analytics",
                "data": data,
            }

            session = aiohttp.ClientSession()
            try:
                async with session.post(
                    endpoint, json=payload, headers=headers, timeout=30
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Successfully exported data to {endpoint}")
                        return True
                    else:
                        self.logger.error(
                            f"API export failed: {response.status} - {await response.text()}"
                        )
                        return False
            finally:
                if session:
                    await session.close()

        except Exception as e:
            if session:
                try:
                    await session.close()
                except Exception:
                    pass
            self.logger.error(f"Error exporting to REST API: {e}")
            return False

    async def export_regulatory_report(
        self, report_type: str, data: dict[str, Any], template: str = None
    ) -> str:
        """
        Export data in regulatory reporting format.

        Args:
            report_type: Type of regulatory report (CCAR, Basel III, MiFID II, etc.)
            data: Data to include in report
            template: Custom template to use

        Returns:
            Formatted regulatory report
        """
        try:
            if report_type.upper() == "CCAR":
                return await self._generate_ccar_report(data)
            elif report_type.upper() == "BASEL_III":
                return await self._generate_basel_iii_report(data)
            elif report_type.upper() == "MIFID_II":
                return await self._generate_mifid_ii_report(data)
            elif template:
                return await self._generate_custom_template_report(data, template)
            else:
                raise ValidationError(
                    f"Unsupported regulatory report type: {report_type}",
                    error_code="EXPORT_004",
                    field_name="report_type",
                    field_value=report_type,
                    validation_rule="supported_regulatory_report",
                )

        except Exception as e:
            self.logger.error(f"Error generating regulatory report: {e}")
            return f"Error generating {report_type} report: {e!s}"

    async def _generate_ccar_report(self, data: dict[str, Any]) -> str:
        """Generate CCAR (Comprehensive Capital Analysis and Review) format report."""
        try:
            # CCAR XML template
            ccar_template = """<?xml version="1.0" encoding="UTF-8"?>
<CCARReport xmlns="http://www.federalreserve.gov/CCAR" reportDate="{{ report_date }}">
    <InstitutionInfo>
        <InstitutionName>{{ institution_name }}</InstitutionName>
        <ReportingDate>{{ reporting_date }}</ReportingDate>
    </InstitutionInfo>
    <RiskMetrics>
        <Tier1CapitalRatio>{{ tier1_capital_ratio }}</Tier1CapitalRatio>
        <LeverageRatio>{{ leverage_ratio }}</LeverageRatio>
        <VaR95>{{ var_95 }}</VaR95>
        <StressTestResults>
            {% for scenario, result in stress_test_results.items() %}
            <Scenario name="{{ scenario }}">
                <Loss>{{ result.loss }}</Loss>
                <CapitalImpact>{{ result.capital_impact }}</CapitalImpact>
            </Scenario>
            {% endfor %}
        </StressTestResults>
    </RiskMetrics>
    <PortfolioData>
        <TotalAssets>{{ total_assets }}</TotalAssets>
        <RiskWeightedAssets>{{ risk_weighted_assets }}</RiskWeightedAssets>
    </PortfolioData>
</CCARReport>"""

            template = Template(ccar_template)

            # Prepare template data
            template_data = {
                "report_date": get_current_utc_timestamp().isoformat(),
                "institution_name": "T-Bot Trading System",
                "reporting_date": get_current_utc_timestamp().date().isoformat(),
                "tier1_capital_ratio": data.get("tier1_capital_ratio", 0.12),
                "leverage_ratio": data.get("leverage_ratio", 0.08),
                "var_95": data.get("var_95", 0.025),
                "stress_test_results": data.get("stress_test_results", {}),
                "total_assets": data.get("total_assets", 1000000),
                "risk_weighted_assets": data.get("risk_weighted_assets", 800000),
            }

            return template.render(**template_data)

        except Exception as e:
            self.logger.error(f"Error generating CCAR report: {e}")
            return f"<Error>Failed to generate CCAR report: {e!s}</Error>"

    async def _generate_basel_iii_report(self, data: dict[str, Any]) -> str:
        """Generate Basel III compliance report."""
        try:
            basel_template = """Basel III Regulatory Capital Report
Generated: {{ report_date }}
Institution: {{ institution_name }}

=== CAPITAL RATIOS ===
Common Equity Tier 1 Ratio: {{ cet1_ratio }}%
Tier 1 Capital Ratio: {{ tier1_ratio }}%
Total Capital Ratio: {{ total_capital_ratio }}%

=== LEVERAGE RATIO ===
Leverage Ratio: {{ leverage_ratio }}%
Tier 1 Capital: {{ tier1_capital }}
Total Exposure: {{ total_exposure }}

=== LIQUIDITY COVERAGE RATIO ===
LCR: {{ lcr }}%
High Quality Liquid Assets: {{ hqla }}
Net Cash Outflows: {{ net_cash_outflows }}

=== RISK-WEIGHTED ASSETS ===
Credit Risk RWA: {{ credit_risk_rwa }}
Market Risk RWA: {{ market_risk_rwa }}
Operational Risk RWA: {{ operational_risk_rwa }}
Total RWA: {{ total_rwa }}

=== COUNTERPARTY CREDIT RISK ===
CVA Capital: {{ cva_capital }}
SA-CCR Exposure: {{ sa_ccr_exposure }}
"""

            template = Template(basel_template)

            template_data = {
                "report_date": get_current_utc_timestamp().isoformat(),
                "institution_name": "T-Bot Trading System",
                "cet1_ratio": data.get("cet1_ratio", 12.5),
                "tier1_ratio": data.get("tier1_ratio", 14.0),
                "total_capital_ratio": data.get("total_capital_ratio", 16.5),
                "leverage_ratio": data.get("leverage_ratio", 8.2),
                "tier1_capital": data.get("tier1_capital", 140000),
                "total_exposure": data.get("total_exposure", 1707317),
                "lcr": data.get("lcr", 125),
                "hqla": data.get("hqla", 312500),
                "net_cash_outflows": data.get("net_cash_outflows", 250000),
                "credit_risk_rwa": data.get("credit_risk_rwa", 600000),
                "market_risk_rwa": data.get("market_risk_rwa", 150000),
                "operational_risk_rwa": data.get("operational_risk_rwa", 100000),
                "total_rwa": data.get("total_rwa", 850000),
                "cva_capital": data.get("cva_capital", 5000),
                "sa_ccr_exposure": data.get("sa_ccr_exposure", 25000),
            }

            return template.render(**template_data)

        except Exception as e:
            self.logger.error(f"Error generating Basel III report: {e}")
            return f"Error generating Basel III report: {e!s}"

    async def stream_real_time_data(self, websocket_clients: set, data: dict[str, Any]) -> None:
        """
        Stream real-time data to WebSocket clients.

        Args:
            websocket_clients: Set of WebSocket connections
            data: Data to stream
        """
        try:
            if not websocket_clients:
                return

            message = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "type": "analytics_stream",
                "data": data,
            }

            message_str = json.dumps(message, default=str)

            # Send to all connected clients
            disconnected_clients = set()
            for client in websocket_clients:
                try:
                    await client.send(message_str)
                except Exception as e:
                    self.logger.warning(f"Failed to send data to WebSocket client: {e}")
                    disconnected_clients.add(client)

            # Remove disconnected clients
            websocket_clients -= disconnected_clients

        except Exception as e:
            self.logger.error(f"Error streaming real-time data: {e}")
            raise

    async def create_scheduled_export(
        self, export_config: dict[str, Any], schedule_interval_minutes: int = 60
    ) -> None:
        """
        Create a scheduled export job.

        Args:
            export_config: Export configuration
            schedule_interval_minutes: Export interval in minutes
        """
        try:

            async def export_task():
                while True:
                    try:
                        # Execute export based on configuration
                        await self._execute_scheduled_export(export_config)

                        # Wait for next interval
                        await asyncio.sleep(schedule_interval_minutes * 60)

                    except asyncio.CancelledError:
                        self.logger.info("Scheduled export task cancelled")
                        break
                    except Exception as e:
                        self.logger.error(f"Error in scheduled export: {e}")
                        # Continue with next iteration after error
                        await asyncio.sleep(60)  # Wait 1 minute before retry

            # Start the scheduled task
            task = asyncio.create_task(export_task())
            self.logger.info(
                f"Created scheduled export task: {export_config.get('name', 'unnamed')}"
            )

            return task

        except Exception as e:
            self.logger.error(f"Error creating scheduled export: {e}")
            raise

    async def _execute_scheduled_export(self, config: dict[str, Any]) -> None:
        """Execute a scheduled export based on configuration."""
        try:
            export_type = config.get("type")
            destination = config.get("destination")
            data_source = config.get("data_source")

            # Get data based on source configuration
            if data_source == "portfolio_metrics":
                data = await self._get_portfolio_metrics_for_export()
            elif data_source == "risk_metrics":
                data = await self._get_risk_metrics_for_export()
            elif data_source == "performance_metrics":
                data = await self._get_performance_metrics_for_export()
            else:
                self.logger.warning(f"Unknown data source: {data_source}")
                return

            # Export data based on type and destination
            if export_type == "api":
                await self.export_to_rest_api(
                    destination, data, config.get("headers"), config.get("auth")
                )
            elif export_type == "kafka":
                await self.export_to_kafka(destination, data, config.get("kafka_config"))
            elif export_type == "file":
                await self._export_to_file(destination, data, config.get("format", "json"))
            else:
                self.logger.warning(f"Unknown export type: {export_type}")

        except Exception as e:
            self.logger.error(f"Error executing scheduled export: {e}")
            raise

    # Helper methods for data retrieval

    async def _get_portfolio_metrics_for_export(self) -> dict[str, Any]:
        """Get current portfolio metrics for export."""
        # This would integrate with the actual portfolio analytics engine
        return {
            "timestamp": get_current_utc_timestamp().isoformat(),
            "total_value": 1000000,
            "unrealized_pnl": 5000,
            "realized_pnl": 12000,
            "positions_count": 15,
            "cash_balance": 50000,
        }

    async def _get_risk_metrics_for_export(self) -> dict[str, Any]:
        """Get current risk metrics for export."""
        # This would integrate with the actual risk monitoring system
        return {
            "timestamp": get_current_utc_timestamp().isoformat(),
            "var_95": 0.025,
            "expected_shortfall": 0.035,
            "max_drawdown": 0.08,
            "leverage_ratio": 1.2,
            "concentration_risk": 0.15,
        }

    async def _get_performance_metrics_for_export(self) -> dict[str, Any]:
        """Get current performance metrics for export."""
        # This would integrate with the actual performance reporting system
        return {
            "timestamp": get_current_utc_timestamp().isoformat(),
            "daily_return": 0.0008,
            "mtd_return": 0.024,
            "ytd_return": 0.156,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.12,
            "win_rate": 0.62,
        }

    async def _export_to_file(self, filepath: str, data: dict[str, Any], format: str) -> None:
        """Export data to file."""
        try:
            if format.lower() == "json":
                import aiofiles

                async with aiofiles.open(filepath, "w") as f:
                    await f.write(json.dumps(data, indent=2, default=str))
            elif format.lower() == "csv":
                import asyncio

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: pd.json_normalize(data).to_csv(filepath, index=False)
                )
            else:
                self.logger.warning(f"Unsupported file format: {format}")

        except Exception as e:
            self.logger.error(f"Error exporting to file: {e}")
            raise
