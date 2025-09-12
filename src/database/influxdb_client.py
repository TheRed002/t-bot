"""
InfluxDB client for the trading bot framework.

This module provides InfluxDB client for time series data storage including
market data, trades, performance metrics, and system monitoring data.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for time series data storage.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from influxdb_client import InfluxDBClient as InfluxClient, Point, WritePrecision
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.write_api import SYNCHRONOUS

from src.core.base import BaseComponent

# Import core components from P-001
from src.core.config import Config
from src.core.exceptions import DataError

# Error handling is provided by decorators
# Import utils from P-007A
from src.error_handling.decorators import with_circuit_breaker, with_retry
from src.utils.decorators import time_execution


class InfluxDBClientWrapper(BaseComponent):
    """InfluxDB client wrapper with trading-specific utilities."""

    def __init__(self, url: str, token: str, org: str, bucket: str, config: Config | None = None):
        super().__init__()  # Initialize BaseComponent
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client: InfluxClient | None = None
        self.write_api: Any | None = None
        self.query_api: QueryApi | None = None
        self.config = config

    @time_execution
    @with_retry(max_attempts=3)
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=30)
    async def connect(self) -> None:
        """Connect to InfluxDB."""
        self.client = InfluxClient(url=self.url, token=self.token, org=self.org)

        # Initialize APIs
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

        # Test connection with timeout
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, self.client.ping), timeout=10.0
        )
        self.logger.info("InfluxDB connection established")

    async def disconnect(self) -> None:
        """Disconnect from InfluxDB."""
        client = None
        try:
            client = self.client
            if client:
                # Run blocking close operation in executor to avoid blocking
                await asyncio.get_event_loop().run_in_executor(None, client.close)
                self.logger.info("InfluxDB connection closed")
        except Exception as e:
            self.logger.error(f"Error closing InfluxDB connection: {e}")
        finally:
            # Ensure references are cleared even if close fails
            self.client = None
            self.write_api = None
            self.query_api = None

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
            if isinstance(value, (int, float)):
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

    async def write_point(self, point: Point) -> None:
        """Write a single point to InfluxDB."""
        write_api = None
        try:
            write_api = self.write_api
            if not write_api:
                raise DataError("InfluxDB write API not initialized")
            # Run blocking write operation in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, write_api.write, self.bucket, point
            )
        except Exception as e:
            self.logger.error("Failed to write point to InfluxDB", error=str(e))
            raise DataError(f"Failed to write point to InfluxDB: {e!s}") from e
        finally:
            # Ensure write API is flushed if needed
            if write_api and hasattr(write_api, "flush"):
                try:
                    await asyncio.get_event_loop().run_in_executor(None, write_api.flush)
                except Exception as flush_error:
                    self.logger.warning(f"Write API flush failed: {flush_error}")
                    # Try to close write API if flush fails to prevent resource leaks
                    try:
                        if hasattr(write_api, "close"):
                            await asyncio.get_event_loop().run_in_executor(None, write_api.close)
                    except Exception as close_error:
                        self.logger.error(f"Write API close failed: {close_error}")

    async def write_points(self, points: list[Point]) -> None:
        """Write multiple points to InfluxDB."""
        write_api = None
        try:
            write_api = self.write_api
            if not write_api:
                raise DataError("InfluxDB write API not initialized")
            # Run blocking write operation in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, write_api.write, self.bucket, points
            )
        except Exception as e:
            self.logger.error("Failed to write points to InfluxDB", error=str(e))
            raise DataError(f"Failed to write points to InfluxDB: {e!s}") from e
        finally:
            # Ensure write API is flushed if needed
            if write_api and hasattr(write_api, "flush"):
                try:
                    await asyncio.get_event_loop().run_in_executor(None, write_api.flush)
                except Exception as flush_error:
                    self.logger.warning(f"Write API flush failed: {flush_error}")
                    # Try to close write API if flush fails to prevent resource leaks
                    try:
                        if hasattr(write_api, "close"):
                            await asyncio.get_event_loop().run_in_executor(None, write_api.close)
                    except Exception as close_error:
                        self.logger.error(f"Write API close failed: {close_error}")

    # Market data utilities
    async def write_market_data(
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
        await self.write_point(point)

    async def write_market_data_batch(
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
            await self.write_points(points)

    async def write_trade(
        self, trade_data: dict[str, Any], timestamp: datetime | None = None
    ) -> None:
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
        await self.write_point(point)

    async def write_performance_metrics(
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
        await self.write_point(point)

    async def write_system_metrics(
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
        await self.write_point(point)

    async def write_risk_metrics(
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
        await self.write_point(point)

    # Query utilities
    async def query_market_data(
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

        query_api = None
        try:
            query_api = self.query_api
            if not query_api:
                raise DataError("InfluxDB query API not initialized")
            # Run blocking query operation in executor to avoid blocking
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, query_api.query, query, self.org),
                timeout=30.0,
            )
            return self._parse_query_result(result)
        except asyncio.TimeoutError as e:
            self.logger.error("Query market data timed out")
            raise DataError("Query market data timed out") from e
        except Exception as e:
            self.logger.error("Failed to query market data", error=str(e))
            raise DataError(f"Failed to query market data: {e!s}") from e
        finally:
            # Ensure query resources are properly released
            if query_api:
                try:
                    # Some query APIs may have cleanup methods
                    if hasattr(query_api, "close") and callable(query_api.close):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.close)
                    elif hasattr(query_api, "__del__") and callable(query_api.__del__):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.__del__)
                except Exception as close_error:
                    self.logger.warning(f"Query API cleanup failed: {close_error}")

    async def query_trades(
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

        query_api = None
        try:
            query_api = self.query_api
            if not query_api:
                raise DataError("InfluxDB query API not initialized")
            # Run blocking query operation in executor to avoid blocking
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, query_api.query, query, self.org),
                timeout=30.0,
            )
            return self._parse_query_result(result)
        except asyncio.TimeoutError as e:
            self.logger.error("Query trades timed out")
            raise DataError("Query trades timed out") from e
        except Exception as e:
            self.logger.error("Failed to query trades", error=str(e))
            raise DataError(f"Failed to query trades: {e!s}") from e
        finally:
            # Ensure query resources are properly released
            if query_api:
                try:
                    # Some query APIs may have cleanup methods
                    if hasattr(query_api, "close") and callable(query_api.close):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.close)
                    elif hasattr(query_api, "__del__") and callable(query_api.__del__):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.__del__)
                except Exception as close_error:
                    self.logger.warning(f"Query API cleanup failed: {close_error}")

    async def query_performance_metrics(
        self, bot_id: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Query performance metrics for a bot within a time range."""
        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
            |> filter(fn: (r) => r["bot_id"] == "{bot_id}")
        """

        query_api = None
        try:
            query_api = self.query_api
            if not query_api:
                raise DataError("InfluxDB query API not initialized")
            # Run blocking query operation in executor to avoid blocking
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, query_api.query, query, self.org),
                timeout=30.0,
            )
            return self._parse_query_result(result)
        except asyncio.TimeoutError as e:
            self.logger.error("Query performance metrics timed out")
            raise DataError("Query performance metrics timed out") from e
        except Exception as e:
            self.logger.error("Failed to query performance metrics", error=str(e))
            raise DataError(f"Failed to query performance metrics: {e!s}") from e
        finally:
            # Ensure query resources are properly released
            if query_api:
                try:
                    # Some query APIs may have cleanup methods
                    if hasattr(query_api, "close") and callable(query_api.close):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.close)
                    elif hasattr(query_api, "__del__") and callable(query_api.__del__):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.__del__)
                except Exception as close_error:
                    self.logger.warning(f"Query API cleanup failed: {close_error}")

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
    async def get_daily_pnl(self, bot_id: str, date: datetime) -> dict[str, Decimal]:
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

        query_api = None
        try:
            query_api = self.query_api
            if not query_api:
                self.logger.warning("InfluxDB query API not initialized")
                return {"total_pnl": Decimal("0")}
            # Run blocking query operation in executor to avoid blocking
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, query_api.query, query, self.org),
                timeout=30.0,
            )
            total_pnl = Decimal("0")

            for table in result:
                for record in table.records:
                    total_pnl = Decimal(str(record.get_value()))

            return {"total_pnl": total_pnl}
        except asyncio.TimeoutError:
            self.logger.error("Get daily P&L timed out")
            return {"total_pnl": Decimal("0")}
        except Exception as e:
            self.logger.error("Failed to get daily P&L", error=str(e))
            return {"total_pnl": Decimal("0")}
        finally:
            # Ensure query resources are properly released
            if query_api:
                try:
                    # Some query APIs may have cleanup methods
                    if hasattr(query_api, "close") and callable(query_api.close):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.close)
                    elif hasattr(query_api, "__del__") and callable(query_api.__del__):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.__del__)
                except Exception as close_error:
                    self.logger.warning(f"Query API cleanup failed: {close_error}")

    async def get_win_rate(self, bot_id: str, start_time: datetime, end_time: datetime) -> Decimal:
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

        query_api = None
        try:
            query_api = self.query_api
            if not query_api:
                self.logger.warning("InfluxDB query API not initialized")
                return Decimal("0.0")
            # Run blocking query operations in executor to avoid blocking
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, query_api.query, query, self.org),
                timeout=30.0,
            )
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

            total_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, query_api.query, total_query, self.org
                ),
                timeout=30.0,
            )
            total_trades = 0

            for table in total_result:
                for record in table.records:
                    total_trades = int(record.get_value())

            if total_trades > 0:
                return Decimal(str(winning_trades)) / Decimal(str(total_trades))
            else:
                return Decimal("0.0")

        except asyncio.TimeoutError:
            self.logger.error("Get win rate timed out")
            return Decimal("0.0")
        except Exception as e:
            self.logger.error("Failed to get win rate", error=str(e))
            return Decimal("0.0")
        finally:
            # Ensure query resources are properly released
            if query_api:
                try:
                    # Some query APIs may have cleanup methods
                    if hasattr(query_api, "close") and callable(query_api.close):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.close)
                    elif hasattr(query_api, "__del__") and callable(query_api.__del__):
                        await asyncio.get_event_loop().run_in_executor(None, query_api.__del__)
                except Exception as close_error:
                    self.logger.warning(f"Query API cleanup failed: {close_error}")

    # Health check
    async def health_check(self) -> bool:
        """Check InfluxDB health."""
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.client.ping), timeout=5.0
            )
            return True
        except asyncio.TimeoutError:
            self.logger.error("InfluxDB health check timed out")
            return False
        except Exception as e:
            self.logger.error("InfluxDB health check failed", error=str(e))
            return False


# Export wrapper class
__all__ = ["InfluxDBClientWrapper"]
