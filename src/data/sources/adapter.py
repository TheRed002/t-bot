"""Adapter to standardize different data source interfaces."""

import asyncio
import re
from collections.abc import AsyncIterator
from decimal import Decimal, getcontext
from typing import Any

from src.core import BaseComponent
from src.core.exceptions import ConfigurationError, DataError
from src.data.interfaces import DataSourceInterface


class DataSourceAdapter(BaseComponent):
    """
    Adapter to standardize different data source interfaces.

    This class provides a unified interface for different data sources,
    handling the conversion of parameters and responses to a standard format.
    """

    def __init__(self, source_type: str, **config: Any) -> None:
        """
        Initialize data source adapter.

        Args:
            source_type: Type of data source ('binance', 'coinbase', 'okx', etc.)
            **config: Source-specific configuration
        """
        super().__init__()  # Initialize BaseComponent
        self.source_type = source_type.lower()
        self.config = config
        self.source = self._create_source()

    def _create_source(self) -> DataSourceInterface:
        """
        Create the appropriate data source instance.

        Returns:
            Data source instance

        Raises:
            ValueError: If source type is not supported
        """
        # Import here to avoid circular dependencies
        if self.source_type == "binance":
            from src.data.sources.binance import BinanceDataSource

            return BinanceDataSource(**self.config)
        elif self.source_type == "coinbase":
            from src.data.sources.coinbase import CoinbaseDataSource

            return CoinbaseDataSource(**self.config)
        elif self.source_type == "okx":
            from src.data.sources.okx import OKXDataSource

            return OKXDataSource(**self.config)
        else:
            raise ConfigurationError(f"Unsupported data source type: {self.source_type}")

    async def fetch_market_data(
        self, symbol: str, timeframe: str = "1h", limit: int = 100, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Unified interface for fetching market data.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '5m', '1d')
            limit: Number of records to fetch
            **kwargs: Additional parameters

        Returns:
            Standardized list of market data records
        """
        # Convert parameters for specific sources
        adapted_params = self._adapt_fetch_params(symbol, timeframe, limit, **kwargs)

        # Fetch data from source
        raw_data = await self.source.fetch(**adapted_params)

        # Standardize response
        standardized_data = self._standardize_response(raw_data)

        return standardized_data

    async def stream_market_data(self, symbol: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """
        Unified interface for streaming market data.

        Args:
            symbol: Trading symbol
            **kwargs: Additional parameters

        Yields:
            Standardized market data records
        """
        # Convert parameters
        adapted_params = self._adapt_stream_params(symbol, **kwargs)

        # Stream from source
        stream_iter = await self.source.stream(**adapted_params)
        async for raw_record in stream_iter:
            # Standardize each record
            standardized = self._standardize_record(raw_record)
            yield standardized

    def _adapt_fetch_params(
        self, symbol: str, timeframe: str, limit: int, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Adapt parameters for specific data source.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Record limit
            **kwargs: Additional parameters

        Returns:
            Adapted parameters dictionary
        """
        if self.source_type == "binance":
            # Binance uses 'interval' instead of 'timeframe'
            return {
                "symbol": self._symbol_to_binance(symbol),
                "interval": self._timeframe_to_binance_interval(timeframe),
                "limit": limit,
                **kwargs,
            }

        elif self.source_type == "coinbase":
            # Coinbase uses different parameter names
            return {
                "product_id": self._symbol_to_coinbase_pair(symbol),
                "granularity": self._timeframe_to_coinbase_granularity(timeframe),
                "limit": limit,
                **kwargs,
            }

        elif self.source_type == "okx":
            # OKX format
            return {
                "instId": self._symbol_to_okx_inst(symbol),
                "bar": self._timeframe_to_okx_bar(timeframe),
                "limit": str(limit),  # OKX expects string
                **kwargs,
            }

        # Default: pass through as-is
        return {"symbol": symbol, "timeframe": timeframe, "limit": limit, **kwargs}

    def _adapt_stream_params(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        """Adapt streaming parameters for specific source."""
        if self.source_type == "binance":
            return {"symbol": self._symbol_to_binance(symbol), **kwargs}
        elif self.source_type == "coinbase":
            return {"product_ids": [self._symbol_to_coinbase_pair(symbol)], **kwargs}
        elif self.source_type == "okx":
            return {"instId": self._symbol_to_okx_inst(symbol), **kwargs}

        return {"symbol": symbol, **kwargs}

    def _standardize_response(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Standardize response from different sources.

        Args:
            raw_data: Raw data from source

        Returns:
            Standardized data
        """
        standardized = []
        for record in raw_data:
            standardized.append(self._standardize_record(record))
        return standardized

    def _standardize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Standardize a single data record.

        Args:
            record: Raw record from source

        Returns:
            Standardized record with common fields
        """
        # Define standard fields
        standard_record = {}

        if self.source_type == "binance":
            # Binance kline format: [timestamp, open, high, low, close, volume, ...]
            if isinstance(record, list):
                # Set precision context for financial calculations (8 decimal places for crypto)
                getcontext().prec = 16
                standard_record = {
                    "timestamp": record[0],
                    "open": Decimal(str(record[1])).quantize(Decimal("0.00000001")),
                    "high": Decimal(str(record[2])).quantize(Decimal("0.00000001")),
                    "low": Decimal(str(record[3])).quantize(Decimal("0.00000001")),
                    "close": Decimal(str(record[4])).quantize(Decimal("0.00000001")),
                    "volume": Decimal(str(record[5])).quantize(Decimal("0.00000001")),
                    "source": "binance",
                }
            else:
                # Dictionary format
                getcontext().prec = 16
                standard_record = {
                    "timestamp": record.get("openTime", record.get("t")),
                    "open": Decimal(str(record.get("open", record.get("o", 0)))).quantize(
                        Decimal("0.00000001")
                    ),
                    "high": Decimal(str(record.get("high", record.get("h", 0)))).quantize(
                        Decimal("0.00000001")
                    ),
                    "low": Decimal(str(record.get("low", record.get("l", 0)))).quantize(
                        Decimal("0.00000001")
                    ),
                    "close": Decimal(str(record.get("close", record.get("c", 0)))).quantize(
                        Decimal("0.00000001")
                    ),
                    "volume": Decimal(str(record.get("volume", record.get("v", 0)))).quantize(
                        Decimal("0.00000001")
                    ),
                    "source": "binance",
                }

        elif self.source_type == "coinbase":
            # Coinbase format
            getcontext().prec = 16
            standard_record = {
                "timestamp": record.get("time"),
                "open": Decimal(str(record.get("open", 0))).quantize(Decimal("0.00000001")),
                "high": Decimal(str(record.get("high", 0))).quantize(Decimal("0.00000001")),
                "low": Decimal(str(record.get("low", 0))).quantize(Decimal("0.00000001")),
                "close": Decimal(str(record.get("close", 0))).quantize(Decimal("0.00000001")),
                "volume": Decimal(str(record.get("volume", 0))).quantize(Decimal("0.00000001")),
                "source": "coinbase",
            }

        elif self.source_type == "okx":
            # OKX format
            if isinstance(record, list):
                # OKX returns arrays: [timestamp, open, high, low, close, volume, ...]
                getcontext().prec = 16
                standard_record = {
                    "timestamp": int(record[0]),
                    "open": Decimal(str(record[1])).quantize(Decimal("0.00000001")),
                    "high": Decimal(str(record[2])).quantize(Decimal("0.00000001")),
                    "low": Decimal(str(record[3])).quantize(Decimal("0.00000001")),
                    "close": Decimal(str(record[4])).quantize(Decimal("0.00000001")),
                    "volume": Decimal(str(record[5] if len(record) > 5 else 0)).quantize(
                        Decimal("0.00000001")
                    ),
                    "source": "okx",
                }
            else:
                standard_record = record  # Assume already in correct format
                standard_record["source"] = "okx"

        else:
            # Unknown source, pass through
            standard_record = record
            standard_record["source"] = self.source_type

        return standard_record

    # Symbol conversion methods
    def _symbol_to_binance(self, symbol: str) -> str:
        """Convert symbol to Binance format (BTCUSDT)."""
        # Remove common separators
        return symbol.upper().replace("/", "").replace("-", "").replace("_", "")

    def _symbol_to_coinbase_pair(self, symbol: str) -> str:
        """Convert symbol to Coinbase format (BTC-USD)."""
        # Split and rejoin with hyphen
        parts = re.split(r"[/_]", symbol.upper())
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}"
        return symbol.upper().replace("/", "-")

    def _symbol_to_okx_inst(self, symbol: str) -> str:
        """Convert symbol to OKX format (BTC-USDT)."""
        return self._symbol_to_coinbase_pair(symbol)  # Same format as Coinbase

    # Timeframe conversion methods
    def _timeframe_to_binance_interval(self, timeframe: str) -> str:
        """Convert timeframe to Binance interval format."""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "1M",
        }
        return mapping.get(timeframe, "1h")

    def _timeframe_to_coinbase_granularity(self, timeframe: str) -> int:
        """Convert timeframe to Coinbase granularity (seconds)."""
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "1d": 86400,
        }
        return mapping.get(timeframe, 3600)

    def _timeframe_to_okx_bar(self, timeframe: str) -> str:
        """Convert timeframe to OKX bar format."""
        # OKX uses same format as input mostly
        return timeframe

    # Connection management
    async def connect(self) -> None:
        """Connect to data source."""
        try:
            await asyncio.wait_for(self.source.connect(), timeout=30.0)
            self.logger.info(f"Connected to {self.source_type} data source")
        except asyncio.TimeoutError:
            raise DataError(f"Connection timeout for {self.source_type} data source")

    async def disconnect(self) -> None:
        """Disconnect from data source."""
        try:
            await asyncio.wait_for(self.source.disconnect(), timeout=10.0)
            self.logger.info(f"Disconnected from {self.source_type} data source")
        except asyncio.TimeoutError:
            self.logger.warning(f"Disconnect timeout for {self.source_type} data source")
        except Exception as e:
            self.logger.warning(f"Error disconnecting from {self.source_type}: {e}")

    def is_connected(self) -> bool:
        """Check if connected to data source."""
        return self.source.is_connected()
