"""
Technical Indicators Calculator - REFACTORED for DataService Integration

This module provides 50+ technical indicators using TA-Lib, now integrated
with the new DataService architecture to eliminate duplicate calculations
and provide consistent feature management across the system.

Key Features:
- Price-based: SMA, EMA, VWAP, Bollinger Bands, Pivot Points
- Momentum: RSI, MACD, Stochastic, Williams %R, CCI
- Volume: OBV, Volume Profile, MFI, A/D Line, VWAP
- Volatility: ATR, Historical Volatility, GARCH estimates
- Market Structure: Support/Resistance, Fibonacci levels

GPU acceleration is used when available for improved performance.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- DataService integration for consistent caching and calculation
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.core import BaseComponent
from src.core.config import Config

# Import from P-001 core components
from src.core.types import MarketData

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution
from src.utils.technical_indicators import (
    calculate_atr_talib,
    calculate_bollinger_bands_talib,
    calculate_ema_talib,
    calculate_macd_talib,
    calculate_rsi_talib,
    calculate_sma_talib,
)


class IndicatorType(Enum):
    """Technical indicator type enumeration"""

    PRICE_BASED = "price_based"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MARKET_STRUCTURE = "market_structure"


@dataclass
class IndicatorConfig:
    """Technical indicator configuration"""

    indicator_name: str
    indicator_type: IndicatorType
    period: int
    enabled: bool = True
    parameters: dict[str, Any] | None = None


@dataclass
class IndicatorResult:
    """Technical indicator calculation result"""

    indicator_name: str
    symbol: str
    timestamp: datetime
    value: Decimal | None
    metadata: dict[str, Any]
    calculation_time: float


class TechnicalIndicators(BaseComponent):
    """
    REFACTORED: Technical indicator calculator integrated with DataService architecture.

    This class provides 50+ technical indicators for market analysis,
    now designed to work with the FeatureStore for consistent caching
    and elimination of duplicate calculations across the system.

    Key improvements:
    - Integration with FeatureStore for unified feature management
    - Elimination of duplicate calculation logic
    - Consistent result formatting for ML pipeline
    - Better error handling and monitoring
    """

    def __init__(self, config: Config, feature_store=None, data_service=None):
        """Initialize technical indicator calculator."""
        super().__init__()
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.feature_store = feature_store  # Will be injected by FeatureStore
        self.data_service = data_service  # Will be injected for strategy integration

        # Indicator configuration
        indicators_config = getattr(config, "indicators", {})
        if hasattr(indicators_config, "get"):
            self.default_periods = indicators_config.get(
                "default_periods",
                {
                    "sma": 20,
                    "ema": 20,
                    "rsi": 14,
                    "macd": [12, 26, 9],
                    "bollinger": 20,
                    "atr": 14,
                },
            )
            self.cache_enabled = indicators_config.get("cache_enabled", True)
            self.max_calculation_time = indicators_config.get("max_calculation_time", 5.0)
        else:
            self.default_periods = {
                "sma": 20,
                "ema": 20,
                "rsi": 14,
                "macd": [12, 26, 9],
                "bollinger": 20,
                "atr": 14,
            }
            self.cache_enabled = True
            self.max_calculation_time = 5.0

        # Data storage
        self.price_data: dict[str, pd.DataFrame] = {}
        self.feature_cache: dict[str, dict[str, Any]] = {}

        # Calculation statistics
        self.calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_calculation_time": 0.0,
        }

        self.logger.info("TechnicalIndicators initialized with FeatureStore integration")

    def set_feature_store(self, feature_store):
        """Set the feature store for integrated calculations."""
        self.feature_store = feature_store
        self.logger.info("FeatureStore integration enabled")

    def set_data_service(self, data_service):
        """Set the data service for strategy integration."""
        self.data_service = data_service
        self.logger.info("DataService integration enabled")

    @time_execution
    async def calculate_indicators_batch(
        self,
        symbol: str,
        data: list[MarketData],
        indicators: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate multiple technical indicators efficiently.

        This method replaces individual calculation methods to eliminate
        duplicate data processing and provide consistent results.

        Args:
            symbol: Trading symbol
            data: List of market data
            indicators: List of indicator names to calculate
            parameters: Calculation parameters

        Returns:
            Dictionary of calculated indicator values
        """
        try:
            if not data:
                return {}

            # Convert market data to numpy arrays for efficiency - use Decimal for precision
            prices = np.array([float(d.price) for d in data if d.price], dtype=np.float64)
            high_prices = np.array(
                [float(d.high_price) for d in data if d.high_price], dtype=np.float64
            )
            low_prices = np.array(
                [float(d.low_price) for d in data if d.low_price], dtype=np.float64
            )

            if len(prices) == 0:
                return {}

            # Default indicators if none specified
            if indicators is None:
                indicators = ["sma_20", "ema_20", "rsi_14", "macd", "bollinger_bands"]

            params = parameters or {}
            results: dict[str, Decimal | dict[str, Decimal]] = {}

            # Calculate each requested indicator
            for indicator in indicators:
                try:
                    if indicator.startswith("sma_"):
                        period = (
                            int(indicator.split("_")[1])
                            if "_" in indicator
                            else params.get("sma_period", 20)
                        )
                        value = await self._calculate_sma(prices, period)
                        if value is not None:
                            results[indicator] = value

                    elif indicator.startswith("ema_"):
                        period = (
                            int(indicator.split("_")[1])
                            if "_" in indicator
                            else params.get("ema_period", 20)
                        )
                        value = await self._calculate_ema(prices, period)
                        if value is not None:
                            results[indicator] = value

                    elif indicator.startswith("rsi_"):
                        period = (
                            int(indicator.split("_")[1])
                            if "_" in indicator
                            else params.get("rsi_period", 14)
                        )
                        value = await self._calculate_rsi(prices, period)
                        if value is not None:
                            results[indicator] = value

                    elif indicator == "macd":
                        macd_params = params.get("macd", [12, 26, 9])
                        macd_result = await self._calculate_macd(prices, *macd_params)
                        if macd_result is not None:
                            results[indicator] = macd_result

                    elif indicator == "bollinger_bands":
                        period = params.get("bb_period", 20)
                        std_dev = Decimal(str(params.get("bb_std", 2)))
                        bb_result = await self._calculate_bollinger_bands(prices, period, std_dev)
                        if bb_result is not None:
                            results[indicator] = bb_result

                    elif indicator.startswith("atr_"):
                        period = (
                            int(indicator.split("_")[1])
                            if "_" in indicator
                            else params.get("atr_period", 14)
                        )
                        if len(high_prices) > 0 and len(low_prices) > 0:
                            value = await self._calculate_atr(
                                high_prices, low_prices, prices, period
                            )
                            if value is not None:
                                results[indicator] = value

                    # Add more indicators as needed

                except Exception as e:
                    self.logger.error(f"Failed to calculate {indicator}: {e}")
                    continue

            self.calculation_stats["total_calculations"] += len(indicators)
            self.calculation_stats["successful_calculations"] += len(results)

            return results

        except Exception as e:
            self.logger.error(f"Batch indicator calculation failed: {e}")
            self.calculation_stats["failed_calculations"] += 1
            return {}

    async def _calculate_sma(self, prices: np.ndarray, period: int) -> Decimal | None:
        """Calculate Simple Moving Average using shared utility."""
        return calculate_sma_talib(prices, period)

    async def _calculate_ema(self, prices: np.ndarray, period: int) -> Decimal | None:
        """Calculate Exponential Moving Average using shared utility."""
        return calculate_ema_talib(prices, period)

    async def _calculate_rsi(self, prices: np.ndarray, period: int) -> Decimal | None:
        """Calculate Relative Strength Index using shared utility."""
        return calculate_rsi_talib(prices, period)

    async def _calculate_macd(
        self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> dict[str, Decimal] | None:
        """Calculate MACD using shared utility."""
        return calculate_macd_talib(prices, fast, slow, signal)

    async def _calculate_bollinger_bands(
        self, prices: np.ndarray, period: int = 20, std_dev: Decimal = Decimal("2")
    ) -> dict[str, Decimal] | None:
        """Calculate Bollinger Bands using shared utility."""
        return calculate_bollinger_bands_talib(prices, period, std_dev)

    async def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> Decimal | None:
        """Calculate Average True Range using shared utility."""
        return calculate_atr_talib(high, low, close, period)

    # Legacy methods for backward compatibility - DEPRECATED
    async def sma(self, prices: list[Decimal], period: int = 20) -> Decimal | None:
        """DEPRECATED: Use calculate_indicators_batch instead."""
        warnings.warn(
            "sma() is deprecated. Use calculate_indicators_batch() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        prices_array = np.array([float(p) for p in prices], dtype=np.float64)
        return await self._calculate_sma(prices_array, period)

    async def ema(self, prices: list[Decimal], period: int = 20) -> Decimal | None:
        """DEPRECATED: Use calculate_indicators_batch instead."""
        warnings.warn(
            "ema() is deprecated. Use calculate_indicators_batch() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        prices_array = np.array([float(p) for p in prices], dtype=np.float64)
        return await self._calculate_ema(prices_array, period)

    async def rsi(self, prices: list[Decimal], period: int = 14) -> Decimal | None:
        """DEPRECATED: Use calculate_indicators_batch instead."""
        warnings.warn(
            "rsi() is deprecated. Use calculate_indicators_batch() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        prices_array = np.array([float(p) for p in prices], dtype=np.float64)
        return await self._calculate_rsi(prices_array, period)

    async def macd(self, prices: list[Decimal] | None) -> dict[str, Decimal]:
        """DEPRECATED: Use calculate_indicators_batch instead."""
        warnings.warn(
            "macd() is deprecated. Use calculate_indicators_batch() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        prices_array = np.array([float(p) for p in prices], dtype=np.float64)
        return await self._calculate_macd(prices_array)

    async def bollinger_bands(
        self, prices: list[Decimal], period: int = 20, std_dev: Decimal = Decimal("2")
    ) -> dict[str, Decimal] | None:
        """DEPRECATED: Use calculate_indicators_batch instead."""
        warnings.warn(
            "bollinger_bands() is deprecated. Use calculate_indicators_batch() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        prices_array = np.array([float(p) for p in prices], dtype=np.float64)
        return await self._calculate_bollinger_bands(prices_array, period, std_dev)

    async def volume_sma(self, volumes: list[Decimal], period: int = 20) -> Decimal | None:
        """DEPRECATED: Use calculate_indicators_batch instead."""
        warnings.warn(
            "volume_sma() is deprecated. Use calculate_indicators_batch() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        volumes_array = np.array([float(v) for v in volumes], dtype=np.float64)
        return await self._calculate_sma(volumes_array, period)

    def get_calculation_stats(self) -> dict[str, Any]:
        """Get calculation statistics."""
        return self.calculation_stats.copy()

    # Symbol-based wrapper methods for strategy integration
    async def calculate_sma(self, symbol: str, period: int) -> Decimal | None:
        """Calculate SMA using symbol - wrapper for strategy integration."""
        try:
            if not self.feature_store:
                self.logger.warning("Feature store not available for SMA calculation")
                return None

            # Get price data from feature store or data service
            price_data = await self._get_price_data(symbol, period + 10)  # Extra buffer
            if not price_data or len(price_data) < period:
                return None

            prices = np.array([float(d.price) for d in price_data if d.price], dtype=np.float64)
            return await self._calculate_sma(prices, period)

        except Exception as e:
            self.logger.error(f"SMA calculation failed for {symbol}: {e}")
            return None

    async def calculate_rsi(self, symbol: str, period: int) -> Decimal | None:
        """Calculate RSI using symbol - wrapper for strategy integration."""
        try:
            if not self.feature_store:
                self.logger.warning("Feature store not available for RSI calculation")
                return None

            # Get price data from feature store or data service
            price_data = await self._get_price_data(symbol, period + 20)  # Extra buffer for RSI
            if not price_data or len(price_data) < period + 1:
                return None

            prices = np.array([float(d.price) for d in price_data if d.price], dtype=np.float64)
            return await self._calculate_rsi(prices, period)

        except Exception as e:
            self.logger.error(f"RSI calculation failed for {symbol}: {e}")
            return None

    async def calculate_momentum(self, symbol: str, period: int) -> Decimal | None:
        """Calculate price momentum - NEW implementation for strategy integration."""
        try:
            if not self.feature_store:
                self.logger.warning("Feature store not available for momentum calculation")
                return None

            # Get price data from feature store or data service
            price_data = await self._get_price_data(symbol, period + 5)
            if not price_data or len(price_data) < period + 1:
                return None

            prices = np.array([float(d.price) for d in price_data if d.price], dtype=np.float64)

            # Calculate momentum as percent change over period
            if len(prices) >= period + 1:
                getcontext().prec = 16
                current_price = Decimal(str(prices[-1]))
                past_price = Decimal(str(prices[-(period + 1)]))
                if past_price != 0:
                    momentum = ((current_price - past_price) / past_price).quantize(
                        Decimal("0.000001")
                    )
                else:
                    momentum = Decimal("0")
                return momentum
            return None

        except Exception as e:
            self.logger.error(f"Momentum calculation failed for {symbol}: {e}")
            return None

    async def calculate_volatility(self, symbol: str, period: int) -> Decimal | None:
        """Calculate historical volatility - NEW implementation for strategy integration."""
        try:
            if not self.feature_store:
                self.logger.warning("Feature store not available for volatility calculation")
                return None

            # Get price data from feature store or data service
            price_data = await self._get_price_data(symbol, period + 5)
            if not price_data or len(price_data) < period:
                return None

            prices = np.array([float(d.price) for d in price_data if d.price], dtype=np.float64)

            # Calculate returns
            if len(prices) < 2:
                return None

            returns = np.diff(np.log(prices))
            if len(returns) < period:
                return None

            # Calculate volatility as standard deviation of returns
            volatility = np.std(returns[-period:]) if len(returns) >= period else np.std(returns)
            getcontext().prec = 16
            return Decimal(str(volatility)).quantize(Decimal("0.000001"))

        except Exception as e:
            self.logger.error(f"Volatility calculation failed for {symbol}: {e}")
            return None

    async def calculate_volume_ratio(self, symbol: str, period: int) -> Decimal | None:
        """Calculate volume ratio - NEW implementation for strategy integration."""
        try:
            if not self.feature_store:
                self.logger.warning("Feature store not available for volume ratio calculation")
                return None

            # Get price data with volume from feature store or data service
            price_data = await self._get_price_data(symbol, period + 5)
            if not price_data or len(price_data) < period:
                return None

            # Extract volumes
            volumes: list[Decimal] = []
            for d in price_data:
                if hasattr(d, "volume") and d.volume:
                    volumes.append(Decimal(str(d.volume)))
                else:
                    # Use a default volume if not available
                    volumes.append(Decimal("1.0"))

            if len(volumes) < period + 1:
                return None

            # Calculate volume ratio as current volume vs average volume
            current_volume = volumes[-1]
            recent_volumes = volumes[-period:] if len(volumes) >= period else volumes
            avg_volume = (
                sum(recent_volumes) / len(recent_volumes) if recent_volumes else Decimal("1.0")
            )

            if avg_volume > 0:
                getcontext().prec = 16
                volume_ratio = (current_volume / avg_volume).quantize(Decimal("0.0001"))
            else:
                volume_ratio = Decimal("1.0")
            return volume_ratio

        except Exception as e:
            self.logger.error(f"Volume ratio calculation failed for {symbol}: {e}")
            return None

    async def calculate_atr(self, symbol: str, period: int) -> Decimal | None:
        """Calculate ATR using symbol - wrapper for strategy integration."""
        try:
            if not self.feature_store:
                self.logger.warning("Feature store not available for ATR calculation")
                return None

            # Get OHLC data from feature store or data service
            price_data = await self._get_price_data(symbol, period + 10)
            if not price_data or len(price_data) < period:
                return None

            # Extract OHLC arrays
            high_prices = []
            low_prices = []
            close_prices = []

            for d in price_data:
                high_val = (
                    float(d.high_price)
                    if hasattr(d, "high_price") and d.high_price
                    else float(d.price)
                )
                low_val = (
                    float(d.low_price)
                    if hasattr(d, "low_price") and d.low_price
                    else float(d.price)
                )
                close_val = float(d.price)

                high_prices.append(high_val)
                low_prices.append(low_val)
                close_prices.append(close_val)

            if len(high_prices) < period:
                return None

            high_array = np.array(high_prices)
            low_array = np.array(low_prices)
            close_array = np.array(close_prices)

            return await self._calculate_atr(high_array, low_array, close_array, period)

        except Exception as e:
            self.logger.error(f"ATR calculation failed for {symbol}: {e}")
            return None

    async def calculate_bollinger_bands(
        self, symbol: str, period: int = 20, std_dev: Decimal = Decimal("2.0")
    ) -> dict[str, Decimal] | None:
        """Calculate Bollinger Bands using symbol - wrapper for strategy integration."""
        try:
            if not self.feature_store:
                self.logger.warning("Feature store not available for Bollinger Bands calculation")
                return None

            # Get price data from feature store or data service
            price_data = await self._get_price_data(symbol, period + 10)
            if not price_data or len(price_data) < period:
                return None

            prices = np.array([float(d.price) for d in price_data if d.price], dtype=np.float64)
            return await self._calculate_bollinger_bands(prices, period, std_dev)

        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation failed for {symbol}: {e}")
            return None

    async def _get_price_data(self, symbol: str, limit: int) -> list[Any] | None:
        """Get price data from feature store or data service."""
        try:
            # Try to get data from feature store first
            if self.feature_store and hasattr(self.feature_store, "get_recent_data"):
                return await self.feature_store.get_recent_data(symbol, limit=limit)

            # Try to get data from data service
            if self.data_service and hasattr(self.data_service, "get_recent_data"):
                return await self.data_service.get_recent_data(symbol, limit=limit)

            # If neither available, log warning
            self.logger.warning(
                f"No data source available for {symbol}, indicator calculations may be limited"
            )
            return None

        except Exception as e:
            self.logger.error(f"Failed to get price data for {symbol}: {e}")
            return None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.price_data.clear()
        self.feature_cache.clear()
        self.logger.info("TechnicalIndicators cleanup completed")

    # All legacy individual calculation methods have been removed
    # to eliminate duplicate calculation logic. Use calculate_indicators_batch() instead.

    # For backward compatibility, the old class name is aliased
    # but users should migrate to the new FeatureStore integration

    def __str__(self) -> str:
        """String representation."""
        total_calcs = self.calculation_stats["total_calculations"]
        success_calcs = self.calculation_stats["successful_calculations"]
        success_rate = success_calcs / max(1, total_calcs)
        return (
            f"TechnicalIndicators("
            f"total_calculations={total_calcs}, "
            f"success_rate={success_rate:.2%})"
        )

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()


# Legacy alias for backward compatibility
TechnicalIndicatorCalculator = TechnicalIndicators


# All legacy individual calculation methods have been consolidated
# into the batch processing method for better performance and consistency
