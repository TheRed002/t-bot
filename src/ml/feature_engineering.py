"""
Feature Engineering Service for ML Models in Trading.

This module provides comprehensive feature creation, selection, and transformation
capabilities for ML models using the service layer pattern without direct database access.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
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

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.core.types.data import FeatureSet
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering service."""

    max_features: int = Field(default=100, description="Maximum number of features to use")
    feature_selection_threshold: float = Field(
        default=0.01, description="Feature selection threshold"
    )
    cache_ttl_hours: int = Field(default=4, description="Feature cache TTL in hours")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")
    enable_feature_selection: bool = Field(default=True, description="Enable feature selection")
    enable_preprocessing: bool = Field(default=True, description="Enable feature preprocessing")
    computation_workers: int = Field(default=4, description="Number of computation workers")
    default_scaling_method: str = Field(default="standard", description="Default scaling method")
    feature_types: list[str] = Field(
        default_factory=lambda: [
            "price_features",
            "technical_indicators",
            "statistical_features",
            "volume_features",
            "volatility_features",
            "momentum_features",
            "trend_features",
        ],
        description="Default feature types to compute",
    )


class FeatureRequest(BaseModel):
    """Request for feature computation."""

    market_data: dict[str, Any]
    symbol: str
    feature_types: list[str] | None = None
    enable_selection: bool = True
    enable_preprocessing: bool = True
    scaling_method: str = "standard"
    max_features: int | None = None


class FeatureResponse(BaseModel):
    """Response from feature computation."""

    feature_set: FeatureSet
    selected_features: list[str] | None = None
    feature_importance: dict[str, float] | None = None
    preprocessing_info: dict[str, Any] = Field(default_factory=dict)
    computation_time_ms: float
    error: str | None = None


class FeatureEngineeringService(BaseService):
    """
    Feature engineering service for creating, selecting, and transforming features.

    This service provides comprehensive feature engineering capabilities including
    technical indicators, statistical features, feature selection, and preprocessing
    for ML models using proper service patterns without direct database access.

    All data access goes through DataService dependency.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the feature engineering service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="FeatureEngineeringService",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse feature engineering configuration
        fe_config_dict = (config or {}).get("feature_engineering", {})
        self.fe_config = FeatureEngineeringConfig(**fe_config_dict)

        # Service dependencies - resolved during startup
        self.data_service: Any = None
        self.technical_calculator: Any = None
        self.statistical_calculator: Any = None

        # Internal state
        self._scalers: dict[str, Any] = {}
        self._selectors: dict[str, Any] = {}
        self._feature_cache: dict[str, tuple[pd.DataFrame, datetime]] = {}

        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=self.fe_config.computation_workers)

        # Add required dependencies
        self.add_dependency("DataService")
        self.add_dependency("TechnicalIndicatorCalculator")
        self.add_dependency("StatisticalFeatureCalculator")

    async def _do_start(self) -> None:
        """Start the feature engineering service."""
        await super()._do_start()

        # Resolve dependencies
        self.data_service = self.resolve_dependency("DataService")
        self.technical_calculator = self.resolve_dependency("TechnicalIndicatorCalculator")
        self.statistical_calculator = self.resolve_dependency("StatisticalFeatureCalculator")

        self._logger.info(
            "Feature engineering service started successfully",
            config=self.fe_config.dict(),
            dependencies_resolved=3,
        )

    async def _do_stop(self) -> None:
        """Stop the feature engineering service."""
        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        await super()._do_stop()

    # Core Feature Engineering Operations
    @dec.enhance(log=True, monitor=True, log_level="info")
    async def compute_features(self, request: FeatureRequest) -> FeatureResponse:
        """
        Compute features from market data.

        Args:
            request: Feature computation request

        Returns:
            Feature computation response

        Raises:
            ModelError: If feature computation fails
        """
        return await self.execute_with_monitoring(
            "compute_features",
            self._compute_features_impl,
            request,
        )

    async def _compute_features_impl(self, request: FeatureRequest) -> FeatureResponse:
        """Internal feature computation implementation."""
        computation_start = datetime.utcnow()
        warnings = []

        try:
            # Convert market data to DataFrame
            market_data = pd.DataFrame(request.market_data)

            # Validate market data
            if market_data.empty:
                raise ValidationError("Market data cannot be empty")

            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = set(required_columns) - set(market_data.columns)
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")

            # Determine feature types to compute
            feature_types = request.feature_types or self.fe_config.feature_types

            # Validate feature types
            valid_feature_types = {
                "price_features",
                "technical_indicators",
                "statistical_features",
                "volume_features",
                "volatility_features",
                "momentum_features",
                "trend_features",
            }
            invalid_types = set(feature_types) - valid_feature_types
            if invalid_types:
                warnings.append(f"Invalid feature types will be ignored: {invalid_types}")
                feature_types = [ft for ft in feature_types if ft in valid_feature_types]

            # Generate cache key
            cache_key = self._generate_cache_key(request.symbol, feature_types, market_data)

            # Check cache
            cached_features = await self._get_cached_features(cache_key)
            if cached_features is not None:
                computation_time = (datetime.utcnow() - computation_start).total_seconds() * 1000
                return FeatureResponse(
                    feature_set=FeatureSet(
                        feature_set_id=cache_key,
                        symbol=request.symbol,
                        features=cached_features.to_dict("records"),
                        feature_names=list(cached_features.columns),
                        computation_time_ms=0.0,  # Cached
                    ),
                    computation_time_ms=computation_time,
                )

            # Compute all requested features
            features_df = await self._compute_all_feature_types(
                market_data, request.symbol, feature_types
            )

            # Handle missing values
            features_df = self._handle_missing_values(features_df)

            # Feature selection if requested
            selected_features = None
            feature_importance = None
            if request.enable_selection and self.fe_config.enable_feature_selection:
                # Note: For feature selection we'd need target data, which isn't provided
                # This would need to be handled differently in practice
                self._logger.debug("Feature selection skipped - no target data provided")

            # Preprocessing if requested
            preprocessing_info = {}
            if request.enable_preprocessing and self.fe_config.enable_preprocessing:
                features_df, preprocessing_info = await self._preprocess_features(
                    features_df, request.scaling_method
                )

            # Cache the results
            await self._cache_features(cache_key, features_df)

            # Create feature set
            feature_set = FeatureSet(
                feature_set_id=cache_key,
                symbol=request.symbol,
                features=features_df.to_dict("records"),
                feature_names=list(features_df.columns),
                computation_time_ms=(datetime.utcnow() - computation_start).total_seconds() * 1000,
            )

            computation_time = (datetime.utcnow() - computation_start).total_seconds() * 1000

            self._logger.info(
                "Features computed successfully",
                symbol=request.symbol,
                feature_types=feature_types,
                feature_count=len(features_df.columns),
                data_points=len(features_df),
                computation_time_ms=computation_time,
            )

            return FeatureResponse(
                feature_set=feature_set,
                selected_features=selected_features,
                feature_importance=feature_importance,
                preprocessing_info=preprocessing_info,
                computation_time_ms=computation_time,
            )

        except Exception as e:
            computation_time = (datetime.utcnow() - computation_start).total_seconds() * 1000
            error_msg = f"Feature computation failed for {request.symbol}: {e}"

            self._logger.error(
                "Feature computation failed",
                symbol=request.symbol,
                error=str(e),
            )

            return FeatureResponse(
                feature_set=FeatureSet(
                    feature_set_id="error",
                    symbol=request.symbol,
                    features=[],
                    feature_names=[],
                    computation_time_ms=computation_time,
                ),
                computation_time_ms=computation_time,
                error=error_msg,
            )

    async def _compute_all_feature_types(
        self, market_data: pd.DataFrame, symbol: str, feature_types: list[str]
    ) -> pd.DataFrame:
        """Compute all requested feature types."""
        feature_dfs = []

        # Execute feature computations concurrently
        tasks = []

        for feature_type in feature_types:
            if feature_type == "price_features":
                task = asyncio.create_task(self._compute_price_features_async(market_data))
            elif feature_type == "technical_indicators":
                task = asyncio.create_task(
                    self._compute_technical_features_async(market_data, symbol)
                )
            elif feature_type == "statistical_features":
                task = asyncio.create_task(self._compute_statistical_features_async(market_data))
            elif feature_type == "volume_features":
                task = asyncio.create_task(self._compute_volume_features_async(market_data))
            elif feature_type == "volatility_features":
                task = asyncio.create_task(self._compute_volatility_features_async(market_data))
            elif feature_type == "momentum_features":
                task = asyncio.create_task(self._compute_momentum_features_async(market_data))
            elif feature_type == "trend_features":
                task = asyncio.create_task(self._compute_trend_features_async(market_data))
            else:
                self._logger.warning(f"Unknown feature type: {feature_type}")
                continue

            tasks.append((feature_type, task))

        # Wait for all computations to complete
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Collect successful results
        for i, result in enumerate(results):
            feature_type = tasks[i][0]
            if isinstance(result, Exception):
                self._logger.warning(f"Feature computation failed for {feature_type}: {result}")
                continue

            if result is not None and not result.empty:
                feature_dfs.append(result)

        if not feature_dfs:
            raise ModelError("No features could be computed")

        # Combine all features
        combined_features = pd.concat(feature_dfs, axis=1)

        # Remove duplicate columns
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

        return combined_features

    # Feature Computation Methods (now async)
    async def _compute_price_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute price-based features asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._compute_price_features, market_data)

    def _compute_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute price-based features."""
        features = pd.DataFrame(index=data.index)

        # Basic price features
        features["price_range"] = data["high"] - data["low"]
        features["price_change"] = data["close"] - data["open"]
        features["price_change_pct"] = data["close"].pct_change()

        # Safe division - avoid division by zero
        low_mask = data["low"] != 0
        close_mask = data["close"] != 0

        features["hl_ratio"] = np.nan
        features.loc[low_mask, "hl_ratio"] = data.loc[low_mask, "high"] / data.loc[low_mask, "low"]

        features["oc_ratio"] = np.nan
        features.loc[close_mask, "oc_ratio"] = (
            data.loc[close_mask, "open"] / data.loc[close_mask, "close"]
        )

        # Price position within range
        range_mask = (data["high"] - data["low"]) != 0
        features.loc[range_mask, "close_position"] = (
            data.loc[range_mask, "close"] - data.loc[range_mask, "low"]
        ) / (data.loc[range_mask, "high"] - data.loc[range_mask, "low"])
        features.loc[range_mask, "open_position"] = (
            data.loc[range_mask, "open"] - data.loc[range_mask, "low"]
        ) / (data.loc[range_mask, "high"] - data.loc[range_mask, "low"])

        # Multi-period returns
        for period in [1, 2, 3, 5, 10, 20]:
            features[f"return_{period}d"] = data["close"].pct_change(period)
            features[f"log_return_{period}d"] = np.log(data["close"] / data["close"].shift(period))

        return features

    async def _compute_technical_features_async(
        self, market_data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Compute technical indicator features asynchronously."""
        try:
            return await self.technical_calculator.calculate_all_indicators(market_data, symbol)
        except Exception as e:
            self._logger.warning(f"Technical features computation failed: {e}")
            return pd.DataFrame(index=market_data.index)

    async def _compute_statistical_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute statistical features asynchronously."""
        try:
            return await self.statistical_calculator.calculate_all_features(market_data)
        except Exception as e:
            self._logger.warning(f"Statistical features computation failed: {e}")
            return pd.DataFrame(index=market_data.index)

    async def _compute_volume_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._compute_volume_features, market_data
        )

    def _compute_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features."""
        features = pd.DataFrame(index=data.index)

        # Volume features
        features["volume_sma_10"] = data["volume"].rolling(10).mean()
        features["volume_sma_20"] = data["volume"].rolling(20).mean()

        # Safe division
        volume_sma_20_mask = features["volume_sma_20"] != 0
        features["volume_ratio"] = np.nan
        features.loc[volume_sma_20_mask, "volume_ratio"] = (
            data.loc[volume_sma_20_mask, "volume"]
            / features.loc[volume_sma_20_mask, "volume_sma_20"]
        )
        features["volume_change"] = data["volume"].pct_change()

        # Volume-price features
        volume_sum = data["volume"].rolling(20).sum()
        volume_sum_mask = volume_sum != 0
        features["vwap"] = np.nan
        features.loc[volume_sum_mask, "vwap"] = (
            data.loc[volume_sum_mask, "close"] * data.loc[volume_sum_mask, "volume"]
        ).rolling(20).sum() / volume_sum.loc[volume_sum_mask]
        features["price_volume"] = data["close"] * data["volume"]
        features["volume_price_trend"] = (
            features["price_volume"] / features["price_volume"].shift(1)
        ) - 1

        return features

    async def _compute_volatility_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based features asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._compute_volatility_features, market_data
        )

    def _compute_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based features."""
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
        vol_20_mask = features["close_volatility_20"] != 0
        features["volatility_ratio"] = np.nan
        features.loc[vol_20_mask, "volatility_ratio"] = (
            features.loc[vol_20_mask, "close_volatility_5"]
            / features.loc[vol_20_mask, "close_volatility_20"]
        )

        return features

    async def _compute_momentum_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum-based features asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._compute_momentum_features, market_data
        )

    def _compute_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum-based features."""
        features = pd.DataFrame(index=data.index)

        # Rate of change
        for period in [5, 10, 14, 20]:
            close_shifted = data["close"].shift(period)
            close_shifted_mask = close_shifted != 0
            features[f"roc_{period}"] = np.nan
            features.loc[close_shifted_mask, f"roc_{period}"] = (
                (data.loc[close_shifted_mask, "close"] - close_shifted.loc[close_shifted_mask])
                / close_shifted.loc[close_shifted_mask]
            ) * 100

        # RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        avg_loss_mask = avg_loss != 0
        rs = np.nan
        rs_values = np.full_like(avg_gain, np.nan)
        rs_values[avg_loss_mask] = avg_gain[avg_loss_mask] / avg_loss[avg_loss_mask]
        features["rsi_14"] = 100 - (100 / (1 + pd.Series(rs_values, index=data.index)))

        # Stochastic oscillator
        lowest_low = data["low"].rolling(14).min()
        highest_high = data["high"].rolling(14).max()

        hh_ll_diff = highest_high - lowest_low
        hh_ll_diff_mask = hh_ll_diff != 0
        features["stoch_k"] = np.nan
        features.loc[hh_ll_diff_mask, "stoch_k"] = (
            (data.loc[hh_ll_diff_mask, "close"] - lowest_low.loc[hh_ll_diff_mask])
            / hh_ll_diff.loc[hh_ll_diff_mask]
        ) * 100
        features["stoch_d"] = features["stoch_k"].rolling(3).mean()

        return features

    async def _compute_trend_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute trend-based features asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._compute_trend_features, market_data)

    def _compute_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute trend-based features."""
        features = pd.DataFrame(index=data.index)

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f"sma_{period}"] = data["close"].rolling(period).mean()
            features[f"ema_{period}"] = data["close"].ewm(span=period).mean()

        # Moving average crossovers
        for short, long in [(5, 20), (10, 50), (12, 26)]:
            if f"sma_{long}" in features.columns and f"sma_{short}" in features.columns:
                sma_long_mask = features[f"sma_{long}"] != 0
                features[f"sma_{short}_{long}_ratio"] = np.nan
                features.loc[sma_long_mask, f"sma_{short}_{long}_ratio"] = (
                    features.loc[sma_long_mask, f"sma_{short}"]
                    / features.loc[sma_long_mask, f"sma_{long}"]
                )

            if f"ema_{long}" in features.columns and f"ema_{short}" in features.columns:
                ema_long_mask = features[f"ema_{long}"] != 0
                features[f"ema_{short}_{long}_ratio"] = np.nan
                features.loc[ema_long_mask, f"ema_{short}_{long}_ratio"] = (
                    features.loc[ema_long_mask, f"ema_{short}"]
                    / features.loc[ema_long_mask, f"ema_{long}"]
                )

        # Price vs moving average
        for period in [20, 50]:
            if f"sma_{period}" in features.columns:
                sma_mask = features[f"sma_{period}"] != 0
                features[f"price_vs_sma_{period}"] = np.nan
                features.loc[sma_mask, f"price_vs_sma_{period}"] = (
                    data.loc[sma_mask, "close"] / features.loc[sma_mask, f"sma_{period}"]
                )

            if f"ema_{period}" in features.columns:
                ema_mask = features[f"ema_{period}"] != 0
                features[f"price_vs_ema_{period}"] = np.nan
                features.loc[ema_mask, f"price_vs_ema_{period}"] = (
                    data.loc[ema_mask, "close"] / features.loc[ema_mask, f"ema_{period}"]
                )

        # Trend strength
        if "sma_5" in features.columns and "sma_20" in features.columns:
            sma_20_mask = features["sma_20"] != 0
            features["trend_strength"] = np.nan
            features.loc[sma_20_mask, "trend_strength"] = (
                features.loc[sma_20_mask, "sma_5"] - features.loc[sma_20_mask, "sma_20"]
            ) / features.loc[sma_20_mask, "sma_20"]

        return features

    # Feature Selection and Preprocessing
    async def select_features(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        method: str = "mutual_info",
        max_features: int | None = None,
        percentile: float | None = None,
    ) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
        """
        Select the most important features.

        Args:
            features_df: Feature data
            target_series: Target data
            method: Selection method ('mutual_info', 'f_test')
            max_features: Number of features to select
            percentile: Percentile of features to select

        Returns:
            Tuple of (selected features DataFrame, selected feature names, importance scores)
        """
        return await self.execute_with_monitoring(
            "select_features",
            self._select_features_impl,
            features_df,
            target_series,
            method,
            max_features,
            percentile,
        )

    async def _select_features_impl(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        method: str,
        max_features: int | None,
        percentile: float | None,
    ) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
        """Internal feature selection implementation."""
        try:
            if features_df.empty or target_series.empty:
                raise ValidationError("Feature and target data cannot be empty")

            # Determine selection parameters
            if max_features is None and percentile is None:
                max_features = min(self.fe_config.max_features, len(features_df.columns))

            # Choose selection method
            if method == "mutual_info":
                if target_series.dtype in ["object", "category"] or target_series.nunique() < 10:
                    score_func = mutual_info_classif
                else:
                    score_func = mutual_info_regression
            elif method == "f_test":
                if target_series.dtype in ["object", "category"] or target_series.nunique() < 10:
                    score_func = f_classif
                else:
                    score_func = f_regression
            else:
                raise ValidationError(f"Unknown selection method: {method}")

            # Create selector
            if max_features is not None:
                selector = SelectKBest(score_func=score_func, k=max_features)
            else:
                selector = SelectPercentile(score_func=score_func, percentile=percentile)

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            selected_features_array = await loop.run_in_executor(
                self._executor, selector.fit_transform, features_df.values, target_series.values
            )

            selected_feature_names = features_df.columns[selector.get_support()].tolist()
            selected_features_df = pd.DataFrame(
                selected_features_array, index=features_df.index, columns=selected_feature_names
            )

            # Get feature scores
            scores = selector.scores_
            importance_scores = {
                name: float(score)
                for name, score in zip(
                    selected_feature_names, scores[selector.get_support()]
                )
            }

            # Store selector for future use
            selector_key = f"{method}_{max_features or percentile}"
            self._selectors[selector_key] = selector

            self._logger.info(
                "Feature selection completed",
                method=method,
                original_features=len(features_df.columns),
                selected_features=len(selected_feature_names),
                max_features=max_features,
                percentile=percentile,
            )

            return selected_features_df, selected_feature_names, importance_scores

        except Exception as e:
            self._logger.error("Feature selection failed", method=method, error=str(e))
            raise ModelError(f"Feature selection failed: {e}")

    async def _preprocess_features(
        self, features_df: pd.DataFrame, scaling_method: str = "standard"
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Preprocess features using scaling and normalization."""
        try:
            if features_df.empty:
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
            if scaling_method not in self._scalers:
                scaler = scaler_class()

                # Fit scaler in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, scaler.fit, features_df.values)

                self._scalers[scaling_method] = scaler
            else:
                scaler = self._scalers[scaling_method]

            # Transform features in thread pool
            loop = asyncio.get_event_loop()
            scaled_features_array = await loop.run_in_executor(
                self._executor, scaler.transform, features_df.values
            )

            scaled_features_df = pd.DataFrame(
                scaled_features_array, index=features_df.index, columns=features_df.columns
            )

            preprocessing_info = {
                "scaling_method": scaling_method,
                "feature_count": len(features_df.columns),
                "scaler_fitted": True,
            }

            self._logger.info(
                "Feature preprocessing completed",
                scaling_method=scaling_method,
                feature_count=len(features_df.columns),
            )

            return scaled_features_df, preprocessing_info

        except Exception as e:
            self._logger.error(
                "Feature preprocessing failed", scaling_method=scaling_method, error=str(e)
            )
            raise ModelError(f"Feature preprocessing failed: {e}")

    # Utility Methods
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Forward fill first
        features = features.ffill()

        # Then backward fill
        features = features.bfill()

        # Finally, fill any remaining NaN with 0
        features = features.fillna(0)

        # Remove infinite values
        features = features.replace([np.inf, -np.inf], 0)

        return features

    def _generate_cache_key(
        self, symbol: str, feature_types: list[str], market_data: pd.DataFrame
    ) -> str:
        """Generate cache key for feature computation."""
        import hashlib

        # Create hash from symbol, feature types, and data shape/checksum
        feature_types_str = "_".join(sorted(feature_types))
        data_shape = f"{len(market_data)}x{len(market_data.columns)}"
        data_checksum = (
            str(hash(tuple(market_data.iloc[-1].values))) if not market_data.empty else "empty"
        )

        cache_str = f"{symbol}_{feature_types_str}_{data_shape}_{data_checksum}"
        return hashlib.md5(cache_str.encode()).hexdigest()[:16]

    @dec.enhance(cache=True, cache_ttl=3600)  # 1 hour cache
    async def _get_cached_features(self, cache_key: str) -> pd.DataFrame | None:
        """Get cached features."""
        if cache_key in self._feature_cache:
            features_df, timestamp = self._feature_cache[cache_key]
            ttl_hours = self.fe_config.cache_ttl_hours

            if datetime.utcnow() - timestamp < timedelta(hours=ttl_hours):
                return features_df

        return None

    async def _cache_features(self, cache_key: str, features_df: pd.DataFrame) -> None:
        """Cache computed features."""
        # Check cache size limit
        if len(self._feature_cache) >= self.fe_config.cache_max_size:
            # Remove oldest entries
            sorted_items = sorted(self._feature_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[
                : len(self._feature_cache) - self.fe_config.cache_max_size + 1
            ]:
                del self._feature_cache[key]

        self._feature_cache[cache_key] = (features_df.copy(), datetime.utcnow())

        # Clean old cache entries to prevent memory issues
        await self._clean_feature_cache()

    async def _clean_feature_cache(self) -> None:
        """Clean expired feature cache entries."""
        ttl_hours = self.fe_config.cache_ttl_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=ttl_hours)

        expired_keys = [
            key for key, (_, timestamp) in self._feature_cache.items() if timestamp < cutoff_time
        ]

        for key in expired_keys:
            del self._feature_cache[key]

        if expired_keys:
            self._logger.debug(f"Cleaned {len(expired_keys)} expired feature cache entries")

    # Service Health and Metrics
    async def _service_health_check(self) -> "HealthStatus":
        """Feature engineering service specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check dependencies
            if not all([self.data_service, self.technical_calculator, self.statistical_calculator]):
                return HealthStatus.UNHEALTHY

            # Check cache size
            cache_size = len(self._feature_cache)
            if cache_size > self.fe_config.cache_max_size:  # Too many cached items
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Feature engineering service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_feature_engineering_metrics(self) -> dict[str, Any]:
        """Get feature engineering service metrics."""
        return {
            "cached_features": len(self._feature_cache),
            "fitted_scalers": len(self._scalers),
            "fitted_selectors": len(self._selectors),
            "executor_workers": self.fe_config.computation_workers,
        }

    async def clear_cache(self) -> dict[str, int]:
        """Clear feature engineering cache."""
        cache_size = len(self._feature_cache)
        scaler_count = len(self._scalers)
        selector_count = len(self._selectors)

        self._feature_cache.clear()
        self._scalers.clear()
        self._selectors.clear()

        self._logger.info(
            "Feature engineering cache cleared",
            cached_features_removed=cache_size,
            scalers_removed=scaler_count,
            selectors_removed=selector_count,
        )

        return {
            "cached_features_removed": cache_size,
            "scalers_removed": scaler_count,
            "selectors_removed": selector_count,
        }

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate feature engineering service configuration."""
        try:
            fe_config_dict = config.get("feature_engineering", {})
            FeatureEngineeringConfig(**fe_config_dict)
            return True
        except Exception as e:
            self._logger.error(
                "Feature engineering service configuration validation failed", error=str(e)
            )
            return False
