"""
Statistical Features Calculator - REFACTORED for FeatureStore Integration

This module provides statistical feature extraction for market analysis,
now integrated with the FeatureStore architecture to eliminate duplicate
calculations and provide consistent feature management.

Key Features:
- Rolling statistics (mean, std, skewness, kurtosis)
- Autocorrelation features
- Cross-correlation between assets
- Regime indicators (trending vs ranging)
- Seasonality and cyclical features

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- FeatureStore: For centralized feature management
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import periodogram

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import DataError

# Import from P-001 core components
from src.core.types import MarketData

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import cache_result, time_execution


class StatFeatureType(Enum):
    """Statistical feature type enumeration"""

    ROLLING_STATS = "rolling_stats"
    AUTOCORRELATION = "autocorrelation"
    CROSS_CORRELATION = "cross_correlation"
    REGIME_DETECTION = "regime_detection"
    SEASONALITY = "seasonality"


class RegimeType(Enum):
    """Market regime type enumeration"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class StatisticalConfig:
    """Statistical feature calculation configuration"""

    feature_name: str
    feature_type: StatFeatureType
    window_size: int
    enabled: bool = True
    parameters: dict[str, Any] = None


@dataclass
class StatisticalResult:
    """Statistical feature calculation result"""

    feature_name: str
    symbol: str
    timestamp: datetime
    value: float | dict[str, float] | None
    metadata: dict[str, Any]
    calculation_time: float


class StatisticalFeatures(BaseComponent):
    """
    REFACTORED: Statistical feature calculator integrated with FeatureStore architecture.

    This class provides advanced statistical analysis including regime detection,
    autocorrelation analysis, and cross-asset correlation studies, now designed
    to work seamlessly with the FeatureStore for efficient feature management.

    Key improvements:
    - Integration with FeatureStore for centralized calculations
    - Elimination of duplicate statistical computations
    - Consistent result formatting for ML pipelines
    - Better error handling and monitoring
    """

    def __init__(self, config: Config, feature_store=None):
        """Initialize statistical feature calculator."""
        super().__init__()
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.feature_store = feature_store  # Will be injected by FeatureStore

        # Statistical configuration
        stats_config = getattr(config, "statistical_features", {})
        if hasattr(stats_config, "get"):
            self.default_windows = stats_config.get(
                "default_windows",
                {
                    "rolling_stats": 20,
                    "autocorr": 50,
                    "regime": 100,
                    "seasonality": 252,  # Trading days in a year
                },
            )
            self.regime_threshold = stats_config.get("regime_threshold", 0.02)
            self.correlation_threshold = stats_config.get("correlation_threshold", 0.7)
        else:
            self.default_windows = {
                "rolling_stats": 20,
                "autocorr": 50,
                "regime": 100,
                "seasonality": 252,
            }
            self.regime_threshold = 0.02
            self.correlation_threshold = 0.7

        # Data storage
        self.price_data: dict[str, pd.DataFrame] = {}
        self.feature_cache: dict[str, dict[str, Any]] = {}

        # Statistics
        self.calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "avg_calculation_time": 0.0,
        }

        self.logger.info("StatisticalFeatureCalculator initialized")

    async def add_market_data(self, data: MarketData) -> None:
        """
        Add market data for statistical calculations.

        Args:
            data: Market data to add
        """
        try:
            symbol = data.symbol

            # Initialize DataFrame if not exists
            if symbol not in self.price_data:
                self.price_data[symbol] = pd.DataFrame(
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "returns",
                        "log_returns",
                    ]
                )

            # Calculate returns
            close_price = float(data.price)
            returns = 0.0
            log_returns = 0.0

            if len(self.price_data[symbol]) > 0:
                prev_close = self.price_data[symbol]["close"].iloc[-1]
                if prev_close > 0:
                    returns = (close_price - prev_close) / prev_close
                    log_returns = np.log(close_price / prev_close)

            # Add new data point
            new_row = {
                "timestamp": data.timestamp,
                "open": float(data.open_price or data.price),
                "high": float(data.high_price or data.price),
                "low": float(data.low_price or data.price),
                "close": close_price,
                "volume": float(data.volume),
                "returns": returns,
                "log_returns": log_returns,
            }

            # More efficient append using loc instead of concat
            new_index = len(self.price_data[symbol])
            self.price_data[symbol].loc[new_index] = new_row

            # Keep only recent data
            max_rows = getattr(self.config, "max_price_history", 2000)
            if len(self.price_data[symbol]) > max_rows:
                self.price_data[symbol] = self.price_data[symbol].tail(max_rows)

            # Clear cache for this symbol
            if symbol in self.feature_cache:
                self.feature_cache[symbol] = {}

        except Exception as e:
            self.logger.error(f"Failed to add market data for {symbol}: {e!s}")
            raise DataError(f"Market data addition failed: {e!s}")

    @time_execution
    @cache_result(ttl=300)
    async def calculate_rolling_stats(
        self, symbol: str, window: int | None = None, field: str = "returns"
    ) -> StatisticalResult:
        """
        Calculate rolling statistical features.

        Args:
            symbol: Trading symbol
            window: Rolling window size
            field: Data field to analyze

        Returns:
            StatisticalResult: Calculation result
        """
        start_time = datetime.now(timezone.utc)

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            window = window or self.default_windows["rolling_stats"]
            df = self.price_data[symbol]

            if len(df) < window:
                return StatisticalResult(
                    feature_name="ROLLING_STATS",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"window": window, "field": field, "reason": "insufficient_data"},
                    calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                )

            # Calculate rolling statistics
            data_series = df[field]
            rolling_mean = data_series.rolling(window=window).mean().iloc[-1]
            rolling_std = data_series.rolling(window=window).std().iloc[-1]
            rolling_skew = data_series.rolling(window=window).skew().iloc[-1]
            rolling_kurt = data_series.rolling(window=window).kurt().iloc[-1]
            rolling_min = data_series.rolling(window=window).min().iloc[-1]
            rolling_max = data_series.rolling(window=window).max().iloc[-1]

            # Calculate additional metrics
            rolling_median = data_series.rolling(window=window).median().iloc[-1]
            rolling_quantile_25 = data_series.rolling(window=window).quantile(0.25).iloc[-1]
            rolling_quantile_75 = data_series.rolling(window=window).quantile(0.75).iloc[-1]

            # Calculate Z-score of latest value
            latest_value = data_series.iloc[-1]
            z_score = ((latest_value - rolling_mean) / rolling_std) if rolling_std > 0 else 0

            statistical_values = {
                "mean": rolling_mean,
                "std": rolling_std,
                "skewness": rolling_skew,
                "kurtosis": rolling_kurt,
                "min": rolling_min,
                "max": rolling_max,
                "median": rolling_median,
                "q25": rolling_quantile_25,
                "q75": rolling_quantile_75,
                "z_score": z_score,
                "latest_value": latest_value,
            }

            self.calculation_stats["successful_calculations"] += 1

            return StatisticalResult(
                feature_name="ROLLING_STATS",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=statistical_values,
                metadata={"window": window, "field": field},
                calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            self.logger.error(f"Rolling stats calculation failed for {symbol}: {e!s}")
            raise DataError(f"Rolling stats calculation failed: {e!s}")

    @time_execution
    @cache_result(ttl=600)
    async def calculate_autocorrelation(
        self, symbol: str, max_lags: int | None = None, field: str = "returns"
    ) -> StatisticalResult:
        """
        Calculate autocorrelation features.

        Args:
            symbol: Trading symbol
            max_lags: Maximum number of lags to analyze
            field: Data field to analyze

        Returns:
            StatisticalResult: Calculation result
        """
        start_time = datetime.now(timezone.utc)

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            max_lags = max_lags or min(50, len(self.price_data[symbol]) // 4)
            df = self.price_data[symbol]

            if len(df) < max_lags * 2:
                return StatisticalResult(
                    feature_name="AUTOCORRELATION",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"max_lags": max_lags, "field": field, "reason": "insufficient_data"},
                    calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                )

            # Calculate autocorrelations
            data_series = df[field].dropna()
            autocorrs = []

            for lag in range(1, max_lags + 1):
                autocorr = data_series.autocorr(lag=lag)
                autocorrs.append(autocorr if not np.isnan(autocorr) else 0.0)

            # Find significant autocorrelations (using 95% confidence interval)
            n = len(data_series)
            confidence_interval = 1.96 / np.sqrt(n)
            significant_lags = [
                i + 1 for i, ac in enumerate(autocorrs) if abs(ac) > confidence_interval
            ]

            # Calculate Ljung-Box test statistic for serial correlation
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox

                ljung_box_stat = acorr_ljungbox(data_series, lags=min(10, max_lags), return_df=True)
                ljung_box_pvalue = ljung_box_stat["lb_pvalue"].min()
            except ImportError:
                ljung_box_pvalue = None
                self.logger.warning("statsmodels not available for Ljung-Box test")

            autocorr_values = {
                "autocorrelations": autocorrs[:10],  # Return first 10 lags
                "max_autocorr": max(autocorrs),
                "min_autocorr": min(autocorrs),
                "mean_autocorr": np.mean(autocorrs),
                # Top 5 significant lags
                "significant_lags": significant_lags[:5],
                "ljung_box_pvalue": ljung_box_pvalue,
            }

            self.calculation_stats["successful_calculations"] += 1

            return StatisticalResult(
                feature_name="AUTOCORRELATION",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=autocorr_values,
                metadata={"max_lags": max_lags, "field": field, "n_observations": len(data_series)},
                calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            self.logger.error(f"Autocorrelation calculation failed for {symbol}: {e!s}")
            raise DataError(f"Autocorrelation calculation failed: {e!s}")

    @time_execution
    @cache_result(ttl=600)
    async def detect_regime(
        self, symbol: str, window: int | None = None, field: str = "returns"
    ) -> StatisticalResult:
        """
        Detect market regime (trending vs ranging).

        Args:
            symbol: Trading symbol
            window: Analysis window size
            field: Data field to analyze

        Returns:
            StatisticalResult: Calculation result
        """
        start_time = datetime.now(timezone.utc)

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            window = window or self.default_windows["regime"]
            df = self.price_data[symbol]

            if len(df) < window:
                return StatisticalResult(
                    feature_name="REGIME_DETECTION",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"window": window, "field": field, "reason": "insufficient_data"},
                    calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                )

            # Get recent data
            recent_data = df.tail(window)
            prices = recent_data["close"]
            returns = recent_data["returns"]

            # Calculate trend indicators
            price_trend = (
                (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] if prices.iloc[0] != 0 else 0
            )
            returns_mean = returns.mean()
            returns_std = returns.std()

            # Calculate volatility regime
            volatility_percentile = stats.percentileofscore(
                df["returns"].rolling(window).std().dropna(), returns_std
            )

            # Determine regime
            regime = RegimeType.UNKNOWN
            confidence = 0.0

            # Trending regimes
            if abs(price_trend) > self.regime_threshold:
                if price_trend > 0:
                    regime = RegimeType.TRENDING_UP
                else:
                    regime = RegimeType.TRENDING_DOWN
                confidence = min(abs(price_trend) / self.regime_threshold, 1.0)

            # Volatility regimes
            elif volatility_percentile > 80:
                regime = RegimeType.HIGH_VOLATILITY
                confidence = (volatility_percentile - 50) / 50
            elif volatility_percentile < 20:
                regime = RegimeType.LOW_VOLATILITY
                confidence = (50 - volatility_percentile) / 50
            else:
                regime = RegimeType.RANGING
                confidence = 1.0 - abs(price_trend) / self.regime_threshold

            # Calculate additional regime metrics
            # Avoid division by zero and handle empty returns
            if len(returns) > 0:
                directional_movement = np.sum(np.sign(returns) == np.sign(returns.shift(1))) / len(
                    returns
                )
            else:
                directional_movement = 0.0
            trending_strength = abs(returns_mean) / returns_std if returns_std > 0 else 0

            regime_values = {
                "regime": regime.value,
                "confidence": confidence,
                "price_trend": price_trend,
                "returns_mean": returns_mean,
                "returns_std": returns_std,
                "volatility_percentile": volatility_percentile,
                "directional_movement": directional_movement,
                "trending_strength": trending_strength,
            }

            self.calculation_stats["successful_calculations"] += 1

            return StatisticalResult(
                feature_name="REGIME_DETECTION",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=regime_values,
                metadata={"window": window, "field": field, "threshold": self.regime_threshold},
                calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            self.logger.error(f"Regime detection failed for {symbol}: {e!s}")
            raise DataError(f"Regime detection failed: {e!s}")

    @time_execution
    @cache_result(ttl=900)
    async def calculate_cross_correlation(
        self, symbol1: str, symbol2: str, max_lags: int = 20, field: str = "returns"
    ) -> StatisticalResult:
        """
        Calculate cross-correlation between two assets.

        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            max_lags: Maximum number of lags to analyze
            field: Data field to analyze

        Returns:
            StatisticalResult: Calculation result
        """
        start_time = datetime.now(timezone.utc)

        try:
            if symbol1 not in self.price_data or symbol2 not in self.price_data:
                missing = [s for s in [symbol1, symbol2] if s not in self.price_data]
                raise DataError(f"No price data available for symbols: {missing}")

            df1 = self.price_data[symbol1]
            df2 = self.price_data[symbol2]

            # Align timestamps and get common data
            df1["timestamp"] = pd.to_datetime(df1["timestamp"])
            df2["timestamp"] = pd.to_datetime(df2["timestamp"])

            merged = pd.merge(df1, df2, on="timestamp", suffixes=("_1", "_2"))

            if len(merged) < max_lags * 2:
                return StatisticalResult(
                    feature_name="CROSS_CORRELATION",
                    symbol=f"{symbol1}_{symbol2}",
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "max_lags": max_lags,
                        "field": field,
                        "reason": "insufficient_common_data",
                    },
                    calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                )

            # Get data series
            series1 = merged[f"{field}_1"].dropna()
            series2 = merged[f"{field}_2"].dropna()

            # Calculate cross-correlations
            cross_correlations = []
            for lag in range(-max_lags, max_lags + 1):
                if lag == 0:
                    ccorr = series1.corr(series2)
                elif lag > 0:
                    ccorr = series1[:-lag].corr(series2[lag:])
                else:  # lag < 0
                    ccorr = series1[-lag:].corr(series2[:lag])

                cross_correlations.append(ccorr if not np.isnan(ccorr) else 0.0)

            # Find maximum correlation and its lag
            max_corr_idx = np.argmax(np.abs(cross_correlations))
            max_corr = cross_correlations[max_corr_idx]
            max_corr_lag = max_corr_idx - max_lags

            # Calculate lead-lag relationship
            # symbol1 leads symbol2
            positive_lags = cross_correlations[max_lags + 1 :]
            # symbol2 leads symbol1
            negative_lags = cross_correlations[:max_lags]

            max_positive_lag_corr = max(positive_lags) if positive_lags else 0
            max_negative_lag_corr = max(negative_lags) if negative_lags else 0

            cross_corr_values = {
                "contemporaneous_correlation": cross_correlations[max_lags],
                "max_correlation": max_corr,
                "max_correlation_lag": max_corr_lag,
                "max_positive_lag_correlation": max_positive_lag_corr,
                "max_negative_lag_correlation": max_negative_lag_corr,
                "lead_lag_asymmetry": max_positive_lag_corr - max_negative_lag_corr,
                "correlation_strength": abs(max_corr),
            }

            self.calculation_stats["successful_calculations"] += 1

            return StatisticalResult(
                feature_name="CROSS_CORRELATION",
                symbol=f"{symbol1}_{symbol2}",
                timestamp=datetime.now(timezone.utc),
                value=cross_corr_values,
                metadata={
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "max_lags": max_lags,
                    "field": field,
                    "n_observations": len(merged),
                },
                calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            self.logger.error(
                f"Cross-correlation calculation failed for {symbol1}-{symbol2}: {e!s}"
            )
            raise DataError(f"Cross-correlation calculation failed: {e!s}")

    @time_execution
    @cache_result(ttl=1800)
    async def detect_seasonality(self, symbol: str, field: str = "returns") -> StatisticalResult:
        """
        Detect seasonal patterns in price data.

        Args:
            symbol: Trading symbol
            field: Data field to analyze

        Returns:
            StatisticalResult: Calculation result
        """
        start_time = datetime.now(timezone.utc)

        try:
            if symbol not in self.price_data:
                raise DataError(f"No price data available for {symbol}")

            df = self.price_data[symbol]
            min_observations = self.default_windows["seasonality"]

            if len(df) < min_observations:
                return StatisticalResult(
                    feature_name="SEASONALITY",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"field": field, "reason": "insufficient_data"},
                    calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                )

            # Prepare time series with datetime index
            df_copy = df.copy()
            df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
            df_copy.set_index("timestamp", inplace=True)

            data_series = df_copy[field].dropna()

            # Extract time components
            df_copy["hour"] = df_copy.index.hour
            df_copy["day_of_week"] = df_copy.index.dayofweek
            df_copy["day_of_month"] = df_copy.index.day
            df_copy["month"] = df_copy.index.month

            # Calculate seasonal patterns
            hourly_pattern = data_series.groupby(df_copy["hour"]).mean().to_dict()
            daily_pattern = data_series.groupby(df_copy["day_of_week"]).mean().to_dict()
            monthly_pattern = data_series.groupby(df_copy["month"]).mean().to_dict()

            # Spectral analysis for dominant frequencies
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    frequencies, power = periodogram(data_series.values)
                    dominant_freq_idx = np.argmax(power[1:]) + 1  # Skip DC component
                    dominant_frequency = frequencies[dominant_freq_idx]
                    dominant_period = 1 / dominant_frequency if dominant_frequency > 0 else None
            except Exception as e:
                self.logger.warning(f"FFT frequency analysis failed: {e}")
                dominant_frequency = None
                dominant_period = None

            # Calculate pattern strength
            hourly_variance = np.var(list(hourly_pattern.values()))
            daily_variance = np.var(list(daily_pattern.values()))
            monthly_variance = np.var(list(monthly_pattern.values()))

            seasonality_values = {
                "hourly_pattern": hourly_pattern,
                "daily_pattern": daily_pattern,
                "monthly_pattern": monthly_pattern,
                "hourly_variance": hourly_variance,
                "daily_variance": daily_variance,
                "monthly_variance": monthly_variance,
                "dominant_frequency": dominant_frequency,
                "dominant_period_days": dominant_period,
                "strongest_pattern": max(
                    [
                        ("hourly", hourly_variance),
                        ("daily", daily_variance),
                        ("monthly", monthly_variance),
                    ],
                    key=lambda x: x[1],
                )[0],
            }

            self.calculation_stats["successful_calculations"] += 1

            return StatisticalResult(
                feature_name="SEASONALITY",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=seasonality_values,
                metadata={"field": field, "n_observations": len(data_series)},
                calculation_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            self.logger.error(f"Seasonality detection failed for {symbol}: {e!s}")
            raise DataError(f"Seasonality detection failed: {e!s}")

    @time_execution
    async def calculate_batch_features(
        self, symbol: str, features: list[str]
    ) -> dict[str, StatisticalResult]:
        """
        Calculate multiple statistical features in batch.

        Args:
            symbol: Trading symbol
            features: List of feature names to calculate

        Returns:
            Dict[str, StatisticalResult]: Results by feature name
        """
        try:
            results = {}

            for feature in features:
                try:
                    if feature.upper() == "ROLLING_STATS":
                        results["ROLLING_STATS"] = await self.calculate_rolling_stats(symbol)
                    elif feature.upper() == "AUTOCORRELATION":
                        results["AUTOCORRELATION"] = await self.calculate_autocorrelation(symbol)
                    elif feature.upper() == "REGIME":
                        results["REGIME"] = await self.detect_regime(symbol)
                    elif feature.upper() == "SEASONALITY":
                        results["SEASONALITY"] = await self.detect_seasonality(symbol)
                    else:
                        self.logger.warning(f"Unknown statistical feature: {feature}")

                except Exception as e:
                    self.logger.error(f"Failed to calculate {feature} for {symbol}: {e!s}")
                    results[feature] = None

            successful_count = len([r for r in results.values() if r is not None])
            self.logger.info(f"Calculated {successful_count} statistical features for {symbol}")
            return results

        except Exception as e:
            self.logger.error(f"Batch statistical feature calculation failed for {symbol}: {e!s}")
            raise DataError(f"Batch calculation failed: {e!s}")

    @time_execution
    async def get_calculation_summary(self) -> dict[str, Any]:
        """Get calculation statistics and summary."""
        try:
            total = self.calculation_stats["total_calculations"]
            success_rate = (
                (self.calculation_stats["successful_calculations"] / total * 100)
                if total > 0
                else 0
            )

            return {
                "statistics": self.calculation_stats.copy(),
                "success_rate": f"{success_rate:.2f}%",
                "symbols_tracked": len(self.price_data),
                "regime_threshold": self.regime_threshold,
                "correlation_threshold": self.correlation_threshold,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to generate calculation summary: {e!s}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


# Alias for backward compatibility
StatisticalFeatureCalculator = StatisticalFeatures
