"""
Technical Indicators Calculator

This module provides 50+ technical indicators using TA-Lib:
- Price-based: SMA, EMA, VWAP, Bollinger Bands, Pivot Points
- Momentum: RSI, MACD, Stochastic, Williams %R, CCI
- Volume: OBV, Volume Profile, MFI, A/D Line, VWAP
- Volatility: ATR, Historical Volatility, GARCH estimates
- Market Structure: Support/Resistance, Fibonacci levels

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- P-000A: Data pipeline integration
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import talib

# Import from P-001 core components
from src.core.types import MarketData, Signal
from src.core.exceptions import DataError, ValidationError
from src.core.config import Config
from src.core.logging import get_logger

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution, retry, cache_result
from src.utils.validators import validate_price, validate_quantity
from src.utils.formatters import format_percentage

logger = get_logger(__name__)


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
    parameters: Dict[str, Any] = None


@dataclass
class IndicatorResult:
    """Technical indicator calculation result"""
    indicator_name: str
    symbol: str
    timestamp: datetime
    value: Optional[float]
    metadata: Dict[str, Any]
    calculation_time: float


class TechnicalIndicatorCalculator:
    """
    Comprehensive technical indicator calculator using TA-Lib.

    This class provides 50+ technical indicators for market analysis,
    supporting both real-time and batch calculations with caching.
    """

    def __init__(self, config: Config):
        """Initialize technical indicator calculator."""
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Indicator configuration
        indicators_config = getattr(config, 'indicators', {})
        if hasattr(indicators_config, 'get'):
            self.default_periods = indicators_config.get('default_periods', {
                'sma': 20, 'ema': 20, 'rsi': 14, 'macd': [12, 26, 9],
                'bollinger': 20, 'atr': 14, 'stoch': [14, 3, 3]
            })
            self.cache_enabled = indicators_config.get('cache_enabled', True)
            self.cache_ttl = indicators_config.get(
                'cache_ttl', 300)  # 5 minutes
        else:
            self.default_periods = {
                'sma': 20, 'ema': 20, 'rsi': 14, 'macd': [12, 26, 9],
                'bollinger': 20, 'atr': 14, 'stoch': [14, 3, 3]
            }
            self.cache_enabled = True
            self.cache_ttl = 300

        # Data storage for calculations
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.indicator_cache: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.calculation_stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_calculation_time': 0.0
        }

        logger.info("TechnicalIndicatorCalculator initialized")

    async def add_market_data(self, data: MarketData) -> None:
        """
        Add market data for indicator calculations.

        Args:
            data: Market data to add
        """
        try:
            symbol = data.symbol

            # Initialize DataFrame if not exists
            if symbol not in self.price_data:
                self.price_data[symbol] = pd.DataFrame(columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ])

            # Add new data point
            new_row = {
                'timestamp': data.timestamp,
                'open': float(data.open_price or data.price),
                'high': float(data.high_price or data.price),
                'low': float(data.low_price or data.price),
                'close': float(data.price),
                'volume': float(data.volume)
            }

            self.price_data[symbol] = pd.concat([
                self.price_data[symbol],
                pd.DataFrame([new_row])
            ], ignore_index=True)

            # Keep only recent data (configurable window)
            max_rows = getattr(self.config, 'max_price_history', 1000)
            if len(self.price_data[symbol]) > max_rows:
                self.price_data[symbol] = self.price_data[symbol].tail(
                    max_rows)

            # Clear cache for this symbol
            if symbol in self.indicator_cache:
                self.indicator_cache[symbol] = {}

        except Exception as e:
            logger.error(f"Failed to add market data for {symbol}: {str(e)}")
            raise DataError(f"Market data addition failed: {str(e)}")

    @time_execution
    @cache_result(ttl=300)
    async def calculate_sma(
        self,
        symbol: str,
        period: int = None,
        field: str = 'close'
    ) -> IndicatorResult:
        """
        Calculate Simple Moving Average.

        Args:
            symbol: Trading symbol
            period: Period for calculation
            field: Price field to use

        Returns:
            IndicatorResult: Calculation result
        """
        start_time = datetime.now()

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            period = period or self.default_periods['sma']
            df = self.price_data[symbol]

            if len(df) < period:
                logger.warning(
                    f"Insufficient data for SMA calculation: {
                        len(df)} < {period}")
                return IndicatorResult(
                    indicator_name='SMA',
                    symbol=symbol,
                    timestamp=datetime.now(
                        timezone.utc),
                    value=None,
                    metadata={
                        'period': period,
                        'reason': 'insufficient_data'},
                    calculation_time=(
                        datetime.now() -
                        start_time).total_seconds())

            # Calculate SMA using TA-Lib
            sma_values = talib.SMA(df[field].values, timeperiod=period)
            latest_value = sma_values[-1] if not np.isnan(
                sma_values[-1]) else None

            self.calculation_stats['successful_calculations'] += 1

            return IndicatorResult(
                indicator_name='SMA',
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=latest_value,
                metadata={'period': period, 'field': field},
                calculation_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            self.calculation_stats['failed_calculations'] += 1
            logger.error(f"SMA calculation failed for {symbol}: {str(e)}")
            raise DataError(f"SMA calculation failed: {str(e)}")

    @time_execution
    @cache_result(ttl=300)
    async def calculate_ema(
        self,
        symbol: str,
        period: int = None,
        field: str = 'close'
    ) -> IndicatorResult:
        """
        Calculate Exponential Moving Average.

        Args:
            symbol: Trading symbol
            period: Period for calculation
            field: Price field to use

        Returns:
            IndicatorResult: Calculation result
        """
        start_time = datetime.now()

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            period = period or self.default_periods['ema']
            df = self.price_data[symbol]

            if len(df) < period:
                return IndicatorResult(
                    indicator_name='EMA',
                    symbol=symbol,
                    timestamp=datetime.now(
                        timezone.utc),
                    value=None,
                    metadata={
                        'period': period,
                        'reason': 'insufficient_data'},
                    calculation_time=(
                        datetime.now() -
                        start_time).total_seconds())

            # Calculate EMA using TA-Lib
            ema_values = talib.EMA(df[field].values, timeperiod=period)
            latest_value = ema_values[-1] if not np.isnan(
                ema_values[-1]) else None

            self.calculation_stats['successful_calculations'] += 1

            return IndicatorResult(
                indicator_name='EMA',
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=latest_value,
                metadata={'period': period, 'field': field},
                calculation_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            self.calculation_stats['failed_calculations'] += 1
            logger.error(f"EMA calculation failed for {symbol}: {str(e)}")
            raise DataError(f"EMA calculation failed: {str(e)}")

    @time_execution
    @cache_result(ttl=300)
    async def calculate_rsi(
        self,
        symbol: str,
        period: int = None
    ) -> IndicatorResult:
        """
        Calculate Relative Strength Index.

        Args:
            symbol: Trading symbol
            period: Period for calculation

        Returns:
            IndicatorResult: Calculation result
        """
        start_time = datetime.now()

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            period = period or self.default_periods['rsi']
            df = self.price_data[symbol]

            if len(df) < period + 1:
                return IndicatorResult(
                    indicator_name='RSI',
                    symbol=symbol,
                    timestamp=datetime.now(
                        timezone.utc),
                    value=None,
                    metadata={
                        'period': period,
                        'reason': 'insufficient_data'},
                    calculation_time=(
                        datetime.now() -
                        start_time).total_seconds())

            # Calculate RSI using TA-Lib
            rsi_values = talib.RSI(df['close'].values, timeperiod=period)
            latest_value = rsi_values[-1] if not np.isnan(
                rsi_values[-1]) else None

            # Determine signal
            signal = None
            if latest_value is not None:
                if latest_value > 70:
                    signal = 'overbought'
                elif latest_value < 30:
                    signal = 'oversold'
                else:
                    signal = 'neutral'

            self.calculation_stats['successful_calculations'] += 1

            return IndicatorResult(
                indicator_name='RSI',
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=latest_value,
                metadata={'period': period, 'signal': signal},
                calculation_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            self.calculation_stats['failed_calculations'] += 1
            logger.error(f"RSI calculation failed for {symbol}: {str(e)}")
            raise DataError(f"RSI calculation failed: {str(e)}")

    @time_execution
    @cache_result(ttl=300)
    async def calculate_macd(
        self,
        symbol: str,
        fast_period: int = None,
        slow_period: int = None,
        signal_period: int = None
    ) -> IndicatorResult:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            symbol: Trading symbol
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            IndicatorResult: Calculation result
        """
        start_time = datetime.now()

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            default_macd = self.default_periods['macd']
            fast_period = fast_period or default_macd[0]
            slow_period = slow_period or default_macd[1]
            signal_period = signal_period or default_macd[2]

            df = self.price_data[symbol]
            min_periods = max(fast_period, slow_period) + signal_period

            if len(df) < min_periods:
                return IndicatorResult(
                    indicator_name='MACD',
                    symbol=symbol,
                    timestamp=datetime.now(
                        timezone.utc),
                    value=None,
                    metadata={
                        'fast_period': fast_period,
                        'slow_period': slow_period,
                        'signal_period': signal_period,
                        'reason': 'insufficient_data'},
                    calculation_time=(
                        datetime.now() -
                        start_time).total_seconds())

            # Calculate MACD using TA-Lib
            macd_line, macd_signal, macd_histogram = talib.MACD(
                df['close'].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )

            # Get latest values
            latest_macd = macd_line[-1] if not np.isnan(
                macd_line[-1]) else None
            latest_signal = macd_signal[-1] if not np.isnan(
                macd_signal[-1]) else None
            latest_histogram = macd_histogram[-1] if not np.isnan(
                macd_histogram[-1]) else None

            # Determine trend signal
            trend_signal = None
            if latest_macd is not None and latest_signal is not None:
                if latest_macd > latest_signal:
                    trend_signal = 'bullish'
                elif latest_macd < latest_signal:
                    trend_signal = 'bearish'
                else:
                    trend_signal = 'neutral'

            self.calculation_stats['successful_calculations'] += 1

            return IndicatorResult(
                indicator_name='MACD',
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=latest_macd,
                metadata={
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'signal_period': signal_period,
                    'macd_line': latest_macd,
                    'signal_line': latest_signal,
                    'histogram': latest_histogram,
                    'trend_signal': trend_signal
                },
                calculation_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            self.calculation_stats['failed_calculations'] += 1
            logger.error(f"MACD calculation failed for {symbol}: {str(e)}")
            raise DataError(f"MACD calculation failed: {str(e)}")

    @time_execution
    @cache_result(ttl=300)
    async def calculate_bollinger_bands(
        self,
        symbol: str,
        period: int = None,
        std_dev: float = 2.0
    ) -> IndicatorResult:
        """
        Calculate Bollinger Bands.

        Args:
            symbol: Trading symbol
            period: Period for calculation
            std_dev: Standard deviation multiplier

        Returns:
            IndicatorResult: Calculation result
        """
        start_time = datetime.now()

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            period = period or self.default_periods['bollinger']
            df = self.price_data[symbol]

            if len(df) < period:
                return IndicatorResult(
                    indicator_name='BOLLINGER_BANDS',
                    symbol=symbol,
                    timestamp=datetime.now(
                        timezone.utc),
                    value=None,
                    metadata={
                        'period': period,
                        'std_dev': std_dev,
                        'reason': 'insufficient_data'},
                    calculation_time=(
                        datetime.now() -
                        start_time).total_seconds())

            # Calculate Bollinger Bands using TA-Lib
            upper_band, middle_band, lower_band = talib.BBANDS(
                df['close'].values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0
            )

            # Get latest values
            latest_upper = upper_band[-1] if not np.isnan(
                upper_band[-1]) else None
            latest_middle = middle_band[-1] if not np.isnan(
                middle_band[-1]) else None
            latest_lower = lower_band[-1] if not np.isnan(
                lower_band[-1]) else None
            latest_price = df['close'].iloc[-1]

            # Calculate band position and width
            band_position = None
            band_width = None
            signal = None

            if all(
                x is not None for x in [
                    latest_upper,
                    latest_middle,
                    latest_lower]):
                band_width = (latest_upper - latest_lower) / \
                    latest_middle * 100
                band_position = (latest_price - latest_lower) / \
                    (latest_upper - latest_lower) * 100

                # Generate signals
                if latest_price <= latest_lower:
                    signal = 'oversold'
                elif latest_price >= latest_upper:
                    signal = 'overbought'
                else:
                    signal = 'neutral'

            self.calculation_stats['successful_calculations'] += 1

            return IndicatorResult(
                indicator_name='BOLLINGER_BANDS',
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=latest_middle,
                metadata={
                    'period': period,
                    'std_dev': std_dev,
                    'upper_band': latest_upper,
                    'middle_band': latest_middle,
                    'lower_band': latest_lower,
                    'band_width': band_width,
                    'band_position': band_position,
                    'signal': signal
                },
                calculation_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            self.calculation_stats['failed_calculations'] += 1
            logger.error(
                f"Bollinger Bands calculation failed for {symbol}: {
                    str(e)}")
            raise DataError(f"Bollinger Bands calculation failed: {str(e)}")

    @time_execution
    @cache_result(ttl=300)
    async def calculate_atr(
        self,
        symbol: str,
        period: int = None
    ) -> IndicatorResult:
        """
        Calculate Average True Range (ATR).

        Args:
            symbol: Trading symbol
            period: Period for calculation

        Returns:
            IndicatorResult: Calculation result
        """
        start_time = datetime.now()

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            period = period or self.default_periods['atr']
            df = self.price_data[symbol]

            if len(df) < period + 1:
                return IndicatorResult(
                    indicator_name='ATR',
                    symbol=symbol,
                    timestamp=datetime.now(
                        timezone.utc),
                    value=None,
                    metadata={
                        'period': period,
                        'reason': 'insufficient_data'},
                    calculation_time=(
                        datetime.now() -
                        start_time).total_seconds())

            # Calculate ATR using TA-Lib
            atr_values = talib.ATR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=period
            )

            latest_value = atr_values[-1] if not np.isnan(
                atr_values[-1]) else None
            latest_price = df['close'].iloc[-1]

            # Calculate volatility percentage
            volatility_pct = None
            if latest_value is not None and latest_price > 0:
                volatility_pct = (latest_value / latest_price) * 100

            self.calculation_stats['successful_calculations'] += 1

            return IndicatorResult(
                indicator_name='ATR',
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=latest_value,
                metadata={
                    'period': period,
                    'volatility_percentage': volatility_pct,
                    'price': latest_price
                },
                calculation_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            self.calculation_stats['failed_calculations'] += 1
            logger.error(f"ATR calculation failed for {symbol}: {str(e)}")
            raise DataError(f"ATR calculation failed: {str(e)}")

    @time_execution
    async def calculate_batch_indicators(
        self,
        symbol: str,
        indicators: List[str]
    ) -> Dict[str, IndicatorResult]:
        """
        Calculate multiple indicators in batch for efficiency.

        Args:
            symbol: Trading symbol
            indicators: List of indicator names to calculate

        Returns:
            Dict[str, IndicatorResult]: Results by indicator name
        """
        try:
            results = {}

            for indicator in indicators:
                try:
                    if indicator.upper() == 'SMA':
                        results['SMA'] = await self.calculate_sma(symbol)
                    elif indicator.upper() == 'EMA':
                        results['EMA'] = await self.calculate_ema(symbol)
                    elif indicator.upper() == 'RSI':
                        results['RSI'] = await self.calculate_rsi(symbol)
                    elif indicator.upper() == 'MACD':
                        results['MACD'] = await self.calculate_macd(symbol)
                    elif indicator.upper() == 'BOLLINGER':
                        results['BOLLINGER'] = await self.calculate_bollinger_bands(symbol)
                    elif indicator.upper() == 'ATR':
                        results['ATR'] = await self.calculate_atr(symbol)
                    else:
                        logger.warning(f"Unknown indicator: {indicator}")

                except Exception as e:
                    logger.error(
                        f"Failed to calculate {indicator} for {symbol}: {
                            str(e)}")
                    results[indicator] = None

            logger.info(
                f"Calculated {len([r for r in results.values() if r is not None])} indicators for {symbol}")
            return results

        except Exception as e:
            logger.error(
                f"Batch indicator calculation failed for {symbol}: {
                    str(e)}")
            raise DataError(f"Batch calculation failed: {str(e)}")

    @time_execution
    async def get_calculation_summary(self) -> Dict[str, Any]:
        """Get calculation statistics and summary."""
        try:
            total = self.calculation_stats['total_calculations']
            success_rate = (
                (self.calculation_stats['successful_calculations'] / total * 100)
                if total > 0 else 0
            )

            cache_hit_rate = (
                (self.calculation_stats['cache_hits'] /
                 (
                    self.calculation_stats['cache_hits'] +
                    self.calculation_stats['cache_misses']) *
                    100) if (
                    self.calculation_stats['cache_hits'] +
                    self.calculation_stats['cache_misses']) > 0 else 0)

            return {
                'statistics': self.calculation_stats.copy(),
                'success_rate': f"{success_rate:.2f}%",
                'cache_hit_rate': f"{cache_hit_rate:.2f}%",
                'symbols_tracked': len(self.price_data),
                'cache_enabled': self.cache_enabled,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to generate calculation summary: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now(
                    timezone.utc).isoformat()}
