"""
InfluxDB client for the trading bot framework.

This module provides InfluxDB client for time series data storage including
market data, trades, performance metrics, and system monitoring data.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for time series data storage.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import SYNCHRONOUS

from src.core.base import BaseComponent

# Import core components from P-001
from src.core.config import Config
from src.core.exceptions import DataError, DataSourceError

# Import error handling from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import NetworkDisconnectionRecovery

# Import utils from P-007A
from src.utils.decorators import circuit_breaker, retry, time_execution
from src.utils.formatters import format_api_response


class InfluxDBClientWrapper(BaseComponent):
    """InfluxDB client wrapper with trading-specific utilities."""

    def __init__(self, url: str, token: str, org: str, bucket: str, config: Config | None = None):
        super().__init__()  # Initialize BaseComponent
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client: InfluxDBClient | None = None
        self.write_api: Any | None = None
        self.query_api: QueryApi | None = None
        self.config = config
        self.error_handler = ErrorHandler(config) if config else None

    @time_execution
    @retry(max_attempts=3)
    @circuit_breaker(failure_threshold=3, recovery_timeout=30)
    async def connect(self) -> None:
        """Connect to InfluxDB."""
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

            # Initialize APIs
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()

            # Test connection
            try:
                self.client.ping()
                self.logger.info("InfluxDB connection established")
            except Exception as e:
                raise DataSourceError(f"InfluxDB health check failed: {e!s}") from e

        except Exception as e:
            if self.error_handler and self.config:
                # Create error context for comprehensive error handling
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="influxdb_client",
                    operation="connect",
                    details={"influxdb_url": self.url, "org": self.org, "bucket": self.bucket},
                )

                # Use ErrorHandler for sophisticated error management
                recovery_scenario = NetworkDisconnectionRecovery(self.config)
                handled = await self.error_handler.handle_error(e, error_context, recovery_scenario)

                if not handled:
                    self.logger.error("InfluxDB connection failed", error=str(e))
                    raise DataSourceError(f"InfluxDB connection failed: {e!s}") from e
                else:
                    self.logger.info("InfluxDB connection recovered after error handling")
                    # Retry connection after recovery
                    await self.connect()
            else:
                self.logger.error("InfluxDB connection failed", error=str(e))
                raise DataSourceError(f"InfluxDB connection failed: {e!s}") from e

    def disconnect(self) -> None:
        """Disconnect from InfluxDB."""
        if self.client:
            self.client.close()
            self.logger.info("InfluxDB connection closed")

    def _decimal_to_float(self, value: Any) -> float:
        """
        Convert Decimal to float for InfluxDB storage.

        WARNING: This method intentionally loses precision for time-series storage.
        InfluxDB requires float types, so we accept precision loss for historical data.
        Financial calculations should always use Decimal types before calling this.
        """
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(value or 0)

    def _create_point(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> Point:
        """Create an InfluxDB point."""
        point = Point(measurement)

        # Add tags
        for key, value in tags.items():
            point.tag(key, value)

        # Add fields
        for key, value in fields.items():
            if isinstance(value, int | float):
                point.field(key, value)
            elif isinstance(value, bool):
                point.field(key, value)
            elif isinstance(value, str):
                point.field(key, value)
            else:
                # Convert complex types to string
                point.field(key, str(value))

        # Add timestamp
        if timestamp:
            point.time(timestamp, WritePrecision.NS)
        else:
            point.time(datetime.now(timezone.utc), WritePrecision.NS)

        return point

    def write_point(self, point: Point) -> None:
        """Write a single point to InfluxDB."""
        try:
            self.write_api.write(bucket=self.bucket, record=point)
        except Exception as e:
            self.logger.error("Failed to write point to InfluxDB", error=str(e))
            raise DataError(f"Failed to write point to InfluxDB: {e!s}")

    def write_points(self, points: list[Point]) -> None:
        """Write multiple points to InfluxDB."""
        try:
            self.write_api.write(bucket=self.bucket, record=points)
        except Exception as e:
            self.logger.error("Failed to write points to InfluxDB", error=str(e))
            raise DataError(f"Failed to write points to InfluxDB: {e!s}")

    # Market data utilities
    def write_market_data(
        self, symbol: str, data: dict[str, Any], timestamp: datetime | None = None
    ) -> None:
        """Write market data point."""
        tags = {"symbol": symbol, "data_type": "market_data"}
        fields = {
            "price": self._decimal_to_float(data.get("price", 0)),
            "volume": self._decimal_to_float(data.get("volume", 0)),
            "bid": self._decimal_to_float(data.get("bid", 0)),
            "ask": self._decimal_to_float(data.get("ask", 0)),
            "open": self._decimal_to_float(data.get("open", 0)),
            "high": self._decimal_to_float(data.get("high", 0)),
            "low": self._decimal_to_float(data.get("low", 0)),
        }

        point = self._create_point("market_data", tags, fields, timestamp)
        self.write_point(point)

    def write_market_data_batch(
        self, data_list: list[dict[str, Any]], timestamp: datetime | None = None
    ) -> None:
        """Write multiple market data points in batch."""
        points = []

        for data in data_list:
            if isinstance(data, dict):
                # Handle dict format
                symbol = data.get("symbol", "")
                fields = {
                    "price": self._decimal_to_float(data.get("price", 0)),
                    "volume": self._decimal_to_float(data.get("volume", 0)),
                    "bid": self._decimal_to_float(data.get("bid", 0)),
                    "ask": self._decimal_to_float(data.get("ask", 0)),
                    "open": self._decimal_to_float(data.get("open", 0)),
                    "high": self._decimal_to_float(data.get("high", 0)),
                    "low": self._decimal_to_float(data.get("low", 0)),
                }
            else:
                # Handle MarketData object format
                symbol = getattr(data, "symbol", "")
                fields = {
                    "price": self._decimal_to_float(getattr(data, "price", 0)),
                    "volume": self._decimal_to_float(getattr(data, "volume", 0) or 0),
                    "bid": self._decimal_to_float(getattr(data, "bid", 0) or 0),
                    "ask": self._decimal_to_float(getattr(data, "ask", 0) or 0),
                    "high": self._decimal_to_float(getattr(data, "high_price", 0) or 0),
                    "low": self._decimal_to_float(getattr(data, "low_price", 0) or 0),
                    "open": self._decimal_to_float(getattr(data, "open_price", 0) or 0),
                }

            tags = {"symbol": symbol, "data_type": "market_data"}
            point = self._create_point("market_data", tags, fields, timestamp)
            points.append(point)

        if points:
            self.write_points(points)

    def write_trade(self, trade_data: dict[str, Any], timestamp: datetime | None = None) -> None:
        """Write trade data point."""
        tags = {
            "symbol": trade_data.get("symbol", ""),
            "exchange": trade_data.get("exchange", ""),
            "side": trade_data.get("side", ""),
            "order_type": trade_data.get("order_type", ""),
            "bot_id": trade_data.get("bot_id", ""),
        }

        fields = {
            "quantity": self._decimal_to_float(trade_data.get("quantity", 0)),
            "price": self._decimal_to_float(trade_data.get("price", 0)),
            "executed_price": self._decimal_to_float(trade_data.get("executed_price", 0)),
            "fee": self._decimal_to_float(trade_data.get("fee", 0)),
            "pnl": self._decimal_to_float(trade_data.get("pnl", 0)),
        }

        point = self._create_point("trades", tags, fields, timestamp)
        self.write_point(point)

    def write_performance_metrics(
        self, bot_id: str, metrics: dict[str, Any], timestamp: datetime | None = None
    ) -> None:
        """Write performance metrics point."""
        tags = {"bot_id": bot_id, "data_type": "performance"}

        fields = {
            "total_trades": int(metrics.get("total_trades", 0)),
            "winning_trades": int(metrics.get("winning_trades", 0)),
            "losing_trades": int(metrics.get("losing_trades", 0)),
            "total_pnl": self._decimal_to_float(metrics.get("total_pnl", 0)),
            "realized_pnl": self._decimal_to_float(metrics.get("realized_pnl", 0)),
            "unrealized_pnl": self._decimal_to_float(metrics.get("unrealized_pnl", 0)),
            "win_rate": self._decimal_to_float(metrics.get("win_rate", 0)),
            "profit_factor": self._decimal_to_float(metrics.get("profit_factor", 0)),
            "sharpe_ratio": self._decimal_to_float(metrics.get("sharpe_ratio", 0)),
            "max_drawdown": self._decimal_to_float(metrics.get("max_drawdown", 0)),
        }

        point = self._create_point("performance_metrics", tags, fields, timestamp)
        self.write_point(point)

    def write_system_metrics(
        self, metrics: dict[str, Any], timestamp: datetime | None = None
    ) -> None:
        """Write system monitoring metrics."""
        tags = {"data_type": "system_metrics"}

        fields = {
            "cpu_usage": float(metrics.get("cpu_usage", 0)),
            "memory_usage": float(metrics.get("memory_usage", 0)),
            "disk_usage": float(metrics.get("disk_usage", 0)),
            "network_latency": float(metrics.get("network_latency", 0)),
            "active_connections": int(metrics.get("active_connections", 0)),
            "error_rate": float(metrics.get("error_rate", 0)),
        }

        point = self._create_point("system_metrics", tags, fields, timestamp)
        self.write_point(point)

    def write_risk_metrics(
        self, bot_id: str, risk_data: dict[str, Any], timestamp: datetime | None = None
    ) -> None:
        """Write risk management metrics."""
        tags = {"bot_id": bot_id, "data_type": "risk_metrics"}

        fields = {
            "var_1d": self._decimal_to_float(risk_data.get("var_1d", 0)),
            "var_5d": self._decimal_to_float(risk_data.get("var_5d", 0)),
            "expected_shortfall": self._decimal_to_float(risk_data.get("expected_shortfall", 0)),
            "max_drawdown": self._decimal_to_float(risk_data.get("max_drawdown", 0)),
            "current_drawdown": self._decimal_to_float(risk_data.get("current_drawdown", 0)),
            "position_count": int(risk_data.get("position_count", 0)),
            "portfolio_exposure": self._decimal_to_float(risk_data.get("portfolio_exposure", 0)),
        }

        point = self._create_point("risk_metrics", tags, fields, timestamp)
        self.write_point(point)

    # Query utilities
    def query_market_data(
        self, symbol: str, start_time: datetime, end_time: datetime, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Query market data for a symbol within a time range."""
        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "market_data")
            |> filter(fn: (r) => r["symbol"] == "{symbol}")
            |> limit(n: {limit})
        """

        try:
            result = self.query_api.query(query, org=self.org)
            return self._parse_query_result(result)
        except Exception as e:
            self.logger.error("Failed to query market data", error=str(e))
            raise DataError(f"Failed to query market data: {e!s}")

    def query_trades(
        self, bot_id: str, start_time: datetime, end_time: datetime, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Query trades for a bot within a time range."""
        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "trades")
            |> filter(fn: (r) => r["bot_id"] == "{bot_id}")
            |> limit(n: {limit})
        """

        try:
            result = self.query_api.query(query, org=self.org)
            return self._parse_query_result(result)
        except Exception as e:
            self.logger.error("Failed to query trades", error=str(e))
            raise DataError(f"Failed to query trades: {e!s}")

    def query_performance_metrics(
        self, bot_id: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Query performance metrics for a bot within a time range."""
        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
            |> filter(fn: (r) => r["bot_id"] == "{bot_id}")
        """

        try:
            result = self.query_api.query(query, org=self.org)
            return self._parse_query_result(result)
        except Exception as e:
            self.logger.error("Failed to query performance metrics", error=str(e))
            raise DataError(f"Failed to query performance metrics: {e!s}")

    def _parse_query_result(self, result) -> list[dict[str, Any]]:
        """Parse InfluxDB query result into list of dictionaries."""
        parsed_data = []

        for table in result:
            for record in table.records:
                data = {
                    "time": record.get_time(),
                    "measurement": record.get_measurement(),
                    "field": record.get_field(),
                    "value": record.get_value(),
                }

                # Add tags
                for key, value in record.values.items():
                    if key.startswith("_"):
                        continue
                    data[key] = value

                parsed_data.append(data)

        return parsed_data

    # Aggregation queries
    def get_daily_pnl(self, bot_id: str, date: datetime) -> dict[str, float]:
        """Get daily P&L summary for a bot."""
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time.replace(hour=23, minute=59, second=59, microsecond=999999)

        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "trades")
            |> filter(fn: (r) => r["bot_id"] == "{bot_id}")
            |> filter(fn: (r) => r["_field"] == "pnl")
            |> sum()
        """

        try:
            result = self.query_api.query(query, org=self.org)
            total_pnl = Decimal("0")

            for table in result:
                for record in table.records:
                    total_pnl = Decimal(str(record.get_value()))

            return {"total_pnl": total_pnl}
        except Exception as e:
            self.logger.error("Failed to get daily P&L", error=str(e))
            return {"total_pnl": Decimal("0")}

    def get_win_rate(self, bot_id: str, start_time: datetime, end_time: datetime) -> float:
        """Get win rate for a bot within a time range."""
        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "trades")
            |> filter(fn: (r) => r["bot_id"] == "{bot_id}")
            |> filter(fn: (r) => r["_field"] == "pnl")
            |> filter(fn: (r) => r["_value"] > 0)
            |> count()
        """

        try:
            result = self.query_api.query(query, org=self.org)
            winning_trades = 0

            for table in result:
                for record in table.records:
                    winning_trades = int(record.get_value())

            # Get total trades
            total_query = f"""
            from(bucket: "{self.bucket}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r["_measurement"] == "trades")
                |> filter(fn: (r) => r["bot_id"] == "{bot_id}")
                |> filter(fn: (r) => r["_field"] == "pnl")
                |> count()
            """

            total_result = self.query_api.query(total_query, org=self.org)
            total_trades = 0

            for table in total_result:
                for record in table.records:
                    total_trades = int(record.get_value())

            if total_trades > 0:
                return winning_trades / total_trades
            else:
                return 0.0

        except Exception as e:
            self.logger.error("Failed to get win rate", error=str(e))
            return 0.0

    # Health check
    def health_check(self) -> bool:
        """Check InfluxDB health."""
        try:
            self.client.ping()
            return True
        except Exception as e:
            self.logger.error("InfluxDB health check failed", error=str(e))
            return False

    # TODO: Remove in production - Debug functions
    def debug_info(self) -> dict[str, Any]:
        """Get debug information about InfluxDB."""
        try:
            self.client.ping()
            debug_data = {
                "status": "pass",
                "message": "Connection successful",
                "version": "2.x",
                "uptime": "N/A",
            }
            # Use utils formatter for consistent API response
            return format_api_response(
                debug_data, success=True, message="InfluxDB debug info retrieved"
            )
        except Exception as e:
            return format_api_response(
                {}, success=False, message=f"Failed to get InfluxDB info: {e!s}"
            )


# Export wrapper class
__all__ = ["InfluxDBClientWrapper"]
