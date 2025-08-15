"""
Feature Engineering for ML Models in Trading.

This module provides comprehensive feature creation, selection, and transformation
capabilities for ML models in the trading system.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.data.features.statistical_features import StatisticalFeatureCalculator
from src.data.features.technical_indicators import TechnicalIndicatorCalculator
from src.utils.decorators import cache_result, log_calls, time_execution

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for creating, selecting, and transforming features.

    This class provides comprehensive feature engineering capabilities including
    technical indicators, statistical features, feature selection, and preprocessing
    for ML models in the trading system.

    Attributes:
        config: Application configuration
        technical_indicators: Technical indicators calculator
        statistical_features: Statistical features calculator
        feature_cache: Cache for computed features
        scalers: Dictionary of fitted scalers
        selectors: Dictionary of fitted feature selectors
    """

    def __init__(self, config: Config):
        """
        Initialize the feature engineer.

        Args:
            config: Application configuration
        """
        self.config = config
        self.technical_indicators = TechnicalIndicatorCalculator(config)
        self.statistical_features = StatisticalFeatureCalculator(config)

        # Feature processing state
        self.feature_cache: dict[str, pd.DataFrame] = {}
        self.scalers: dict[str, Any] = {}
        self.selectors: dict[str, Any] = {}

        # Configuration
        self.max_features = config.ml.max_features
        self.feature_selection_threshold = config.ml.feature_selection_threshold
        self.cache_ttl_hours = config.ml.feature_cache_ttl_hours

        logger.info(
            "Feature engineer initialized",
            max_features=self.max_features,
            selection_threshold=self.feature_selection_threshold,
            cache_ttl_hours=self.cache_ttl_hours,
        )

    @time_execution
    @log_calls
    def create_features(
        self, market_data: pd.DataFrame, symbol: str, feature_types: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Create comprehensive features from market data.

        Args:
            market_data: OHLCV market data
            symbol: Trading symbol
            feature_types: List of feature types to create (None for all)

        Returns:
            DataFrame with created features

        Raises:
            ValidationError: If input data is invalid
        """
        if market_data.empty:
            raise ValidationError("Market data cannot be empty")

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = set(required_columns) - set(market_data.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")

        # Default feature types
        if feature_types is None:
            feature_types = [
                "price_features",
                "technical_indicators",
                "statistical_features",
                "volume_features",
                "volatility_features",
                "momentum_features",
                "trend_features",
            ]

        features_list = []

        try:
            # Price-based features
            if "price_features" in feature_types:
                price_features = self._create_price_features(market_data)
                features_list.append(price_features)

            # Technical indicators
            if "technical_indicators" in feature_types:
                tech_features = self._create_technical_features(market_data, symbol)
                features_list.append(tech_features)

            # Statistical features
            if "statistical_features" in feature_types:
                stat_features = self._create_statistical_features(market_data)
                features_list.append(stat_features)

            # Volume features
            if "volume_features" in feature_types:
                volume_features = self._create_volume_features(market_data)
                features_list.append(volume_features)

            # Volatility features
            if "volatility_features" in feature_types:
                volatility_features = self._create_volatility_features(market_data)
                features_list.append(volatility_features)

            # Momentum features
            if "momentum_features" in feature_types:
                momentum_features = self._create_momentum_features(market_data)
                features_list.append(momentum_features)

            # Trend features
            if "trend_features" in feature_types:
                trend_features = self._create_trend_features(market_data)
                features_list.append(trend_features)

            # Combine all features
            features_df = pd.concat(features_list, axis=1)

            # Remove duplicate columns
            features_df = features_df.loc[:, ~features_df.columns.duplicated()]

            # Handle missing values
            features_df = self._handle_missing_values(features_df)

            # Cache features
            cache_key = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H')}"
            self.feature_cache[cache_key] = features_df

            logger.info(
                "Features created successfully",
                symbol=symbol,
                feature_count=len(features_df.columns),
                data_points=len(features_df),
                feature_types=feature_types,
            )

            return features_df

        except Exception as e:
            logger.error("Feature creation failed", symbol=symbol, error=str(e))
            raise ValidationError(f"Failed to create features for {symbol}: {e}") from e

    @time_execution
    @log_calls
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info",
        k_features: int | None = None,
        percentile: float | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Select the most important features using various methods.

        Args:
            X: Feature data
            y: Target data
            method: Selection method ('mutual_info', 'f_test', 'percentile')
            k_features: Number of features to select
            percentile: Percentile of features to select

        Returns:
            Tuple of (selected features DataFrame, selected feature names)

        Raises:
            ValidationError: If selection parameters are invalid
        """
        try:
            if X.empty or y.empty:
                raise ValidationError("Feature and target data cannot be empty")

            # Determine selection parameters
            if k_features is None and percentile is None:
                k_features = min(self.max_features, len(X.columns))

            # Choose selection method
            if method == "mutual_info":
                if y.dtype in ["object", "category"] or y.nunique() < 10:
                    score_func = mutual_info_classif
                else:
                    score_func = mutual_info_regression
            elif method == "f_test":
                if y.dtype in ["object", "category"] or y.nunique() < 10:
                    score_func = f_classif
                else:
                    score_func = f_regression
            else:
                raise ValidationError(f"Unknown selection method: {method}")

            # Create selector
            if k_features is not None:
                selector = SelectKBest(score_func=score_func, k=k_features)
            else:
                selector = SelectPercentile(score_func=score_func, percentile=percentile)

            # Fit selector and transform features
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            # Create DataFrame with selected features
            X_selected_df = pd.DataFrame(X_selected, index=X.index, columns=selected_features)

            # Store selector for future use
            selector_key = f"{method}_{k_features or percentile}"
            self.selectors[selector_key] = selector

            logger.info(
                "Feature selection completed",
                method=method,
                original_features=len(X.columns),
                selected_features=len(selected_features),
                k_features=k_features,
                percentile=percentile,
            )

            return X_selected_df, selected_features

        except Exception as e:
            logger.error("Feature selection failed", method=method, error=str(e))
            raise ValidationError(f"Feature selection failed: {e}") from e

    @time_execution
    @log_calls
    def preprocess_features(
        self, X: pd.DataFrame, scaling_method: str = "standard", fit_scalers: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess features using scaling and normalization.

        Args:
            X: Feature data
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
            fit_scalers: Whether to fit new scalers or use existing ones

        Returns:
            Preprocessed feature data

        Raises:
            ValidationError: If preprocessing fails
        """
        try:
            if X.empty:
                raise ValidationError("Feature data cannot be empty")

            # Choose scaler
            if scaling_method == "standard":
                scaler_class = StandardScaler
            elif scaling_method == "minmax":
                scaler_class = MinMaxScaler
            elif scaling_method == "robust":
                scaler_class = RobustScaler
            else:
                raise ValidationError(f"Unknown scaling method: {scaling_method}")

            # Get or create scaler
            if fit_scalers or scaling_method not in self.scalers:
                scaler = scaler_class()
                scaler.fit(X)
                self.scalers[scaling_method] = scaler
            else:
                scaler = self.scalers[scaling_method]

            # Transform features
            X_scaled = scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

            logger.info(
                "Feature preprocessing completed",
                scaling_method=scaling_method,
                feature_count=len(X.columns),
                fit_scalers=fit_scalers,
            )

            return X_scaled_df

        except Exception as e:
            logger.error(
                "Feature preprocessing failed", scaling_method=scaling_method, error=str(e)
            )
            raise ValidationError(f"Feature preprocessing failed: {e}") from e

    def _create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        features = pd.DataFrame(index=data.index)

        # Basic price features
        features["price_range"] = data["high"] - data["low"]
        features["price_change"] = data["close"] - data["open"]
        features["price_change_pct"] = data["close"].pct_change()
        features["hl_ratio"] = data["high"] / data["low"]
        features["oc_ratio"] = data["open"] / data["close"]

        # Price position within range
        features["close_position"] = (data["close"] - data["low"]) / (data["high"] - data["low"])
        features["open_position"] = (data["open"] - data["low"]) / (data["high"] - data["low"])

        # Multi-period returns
        for period in [1, 2, 3, 5, 10, 20]:
            features[f"return_{period}d"] = data["close"].pct_change(period)
            features[f"log_return_{period}d"] = np.log(data["close"] / data["close"].shift(period))

        return features

    def _create_technical_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create technical indicator features."""
        try:
            # Use the technical indicators module
            tech_features = self.technical_indicators.calculate_all_indicators(data, symbol)
            return tech_features
        except Exception as e:
            logger.warning(f"Failed to create technical features: {e}")
            return pd.DataFrame(index=data.index)

    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        try:
            # Use the statistical features module
            stat_features = self.statistical_features.calculate_all_features(data)
            return stat_features
        except Exception as e:
            logger.warning(f"Failed to create statistical features: {e}")
            return pd.DataFrame(index=data.index)

    def _create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        features = pd.DataFrame(index=data.index)

        # Volume features
        features["volume_sma_10"] = data["volume"].rolling(10).mean()
        features["volume_sma_20"] = data["volume"].rolling(20).mean()
        features["volume_ratio"] = data["volume"] / features["volume_sma_20"]
        features["volume_change"] = data["volume"].pct_change()

        # Volume-price features
        features["vwap"] = (data["close"] * data["volume"]).rolling(20).sum() / data[
            "volume"
        ].rolling(20).sum()
        features["price_volume"] = data["close"] * data["volume"]
        features["volume_price_trend"] = (
            features["price_volume"] / features["price_volume"].shift(1)
        ) - 1

        return features

    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        features = pd.DataFrame(index=data.index)

        # Price volatility
        features["close_volatility_5"] = data["close"].rolling(5).std()
        features["close_volatility_10"] = data["close"].rolling(10).std()
        features["close_volatility_20"] = data["close"].rolling(20).std()

        # True Range and Average True Range
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        features["true_range"] = np.maximum(high_low, np.maximum(high_close, low_close))
        features["atr_14"] = features["true_range"].rolling(14).mean()

        # Volatility ratios
        features["volatility_ratio"] = (
            features["close_volatility_5"] / features["close_volatility_20"]
        )

        return features

    def _create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features."""
        features = pd.DataFrame(index=data.index)

        # Rate of change
        for period in [5, 10, 14, 20]:
            features[f"roc_{period}"] = (
                (data["close"] - data["close"].shift(period)) / data["close"].shift(period)
            ) * 100

        # RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic oscillator
        lowest_low = data["low"].rolling(14).min()
        highest_high = data["high"].rolling(14).max()
        features["stoch_k"] = ((data["close"] - lowest_low) / (highest_high - lowest_low)) * 100
        features["stoch_d"] = features["stoch_k"].rolling(3).mean()

        return features

    def _create_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create trend-based features."""
        features = pd.DataFrame(index=data.index)

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f"sma_{period}"] = data["close"].rolling(period).mean()
            features[f"ema_{period}"] = data["close"].ewm(span=period).mean()

        # Moving average crossovers
        features["sma_5_20_ratio"] = features["sma_5"] / features["sma_20"]
        features["sma_10_50_ratio"] = features["sma_10"] / features["sma_50"]
        features["ema_12_26_ratio"] = features["ema_12"] / features["ema_26"]

        # Price vs moving average
        features["price_vs_sma_20"] = data["close"] / features["sma_20"]
        features["price_vs_sma_50"] = data["close"] / features["sma_50"]
        features["price_vs_ema_20"] = data["close"] / features["ema_20"]

        # Trend strength
        features["trend_strength"] = (features["sma_5"] - features["sma_20"]) / features["sma_20"]

        return features

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Forward fill first
        features = features.fillna(method="ffill")

        # Then backward fill
        features = features.fillna(method="bfill")

        # Finally, fill any remaining NaN with 0
        features = features.fillna(0)

        # Remove infinite values
        features = features.replace([np.inf, -np.inf], 0)

        return features

    @cache_result(ttl_seconds=3600)  # Cache for 1 hour
    def get_feature_importance(
        self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info"
    ) -> pd.Series:
        """
        Calculate feature importance scores.

        Args:
            X: Feature data
            y: Target data
            method: Importance calculation method

        Returns:
            Series with feature importance scores
        """
        try:
            if method == "mutual_info":
                if y.dtype in ["object", "category"] or y.nunique() < 10:
                    scores = mutual_info_classif(X, y)
                else:
                    scores = mutual_info_regression(X, y)
            elif method == "f_test":
                if y.dtype in ["object", "category"] or y.nunique() < 10:
                    scores, _ = f_classif(X, y)
                else:
                    scores, _ = f_regression(X, y)
            else:
                raise ValidationError(f"Unknown importance method: {method}")

            importance_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)

            return importance_series

        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return pd.Series(dtype=float)

    def clean_cache(self, hours_old: int = 24) -> int:
        """
        Clean old entries from feature cache.

        Args:
            hours_old: Remove entries older than this many hours

        Returns:
            Number of entries removed
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)
        removed_count = 0

        keys_to_remove = []
        for key in self.feature_cache:
            try:
                # Extract timestamp from key
                timestamp_str = key.split("_")[-1]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H")

                if timestamp < cutoff_time:
                    keys_to_remove.append(key)
            except (ValueError, IndexError):
                # If timestamp parsing fails, remove the key
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.feature_cache[key]
            removed_count += 1

        logger.info(f"Cleaned {removed_count} entries from feature cache")
        return removed_count
