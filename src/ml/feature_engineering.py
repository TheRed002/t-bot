"""
Feature Engineering Service for ML Models in Trading.

This module provides comprehensive feature creation, selection, and transformation
capabilities for ML models using the service layer pattern without direct database access.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal

# Import for type annotations
from typing import TYPE_CHECKING, Any

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
from src.core.exceptions import DataError, ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.core.types.data import FeatureSet
from src.core.types import MarketData
from src.utils.constants import ML_FEATURE_CONSTANTS, ML_MODEL_CONSTANTS
from src.utils.decorators import UnifiedDecorator
from src.utils.ml_cache import FeatureCache, generate_feature_cache_key
from src.utils.ml_validation import validate_market_data

if TYPE_CHECKING:
    from src.core.base.interfaces import HealthStatus

# Initialize decorator instance
dec = UnifiedDecorator()


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering service."""

    max_features: int = Field(
        default=ML_FEATURE_CONSTANTS["max_features"],
        description="Maximum number of features to use",
    )
    feature_selection_threshold: float = Field(
        default=ML_FEATURE_CONSTANTS["feature_selection_threshold"],
        description="Feature selection threshold",
    )
    cache_ttl_hours: int = Field(
        default=ML_FEATURE_CONSTANTS["cache_ttl_hours"], description="Feature cache TTL in hours"
    )
    cache_max_size: int = Field(
        default=ML_FEATURE_CONSTANTS["cache_max_size"], description="Maximum cache size"
    )
    enable_feature_selection: bool = Field(default=True, description="Enable feature selection")
    enable_preprocessing: bool = Field(default=True, description="Enable feature preprocessing")
    computation_workers: int = Field(
        default=ML_FEATURE_CONSTANTS["computation_workers"],
        description="Number of computation workers",
    )
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

    market_data: dict[str, Any] | list[dict[str, Any]]
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
    ) -> None:
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
        self.fe_config = FeatureEngineeringConfig(
            **(fe_config_dict if isinstance(fe_config_dict, dict) else {})
        )

        # Service dependencies - resolved during startup
        self.data_service: Any = None
        self.technical_calculator: Any = None
        self.statistical_calculator: Any = None

        # Internal state
        self._scalers: dict[str, Any] = {}
        self._selectors: dict[str, Any] = {}
        self._feature_cache: FeatureCache | None = None

        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=self.fe_config.computation_workers)

        # Add required dependencies
        self.add_dependency("MLDataService")
        self.add_dependency("TechnicalIndicatorService")
        self.add_dependency("StatisticalFeatureService")

    async def _do_start(self) -> None:
        """Start the feature engineering service."""
        await super()._do_start()

        # Resolve dependencies
        self.ml_data_service = self.resolve_dependency("MLDataService")

        # Try to resolve feature calculators from DI, fallback to direct creation if not available
        try:
            self.technical_calculator = self.resolve_dependency("TechnicalIndicatorService")
        except Exception as e:
            self._logger.warning(f"TechnicalIndicatorService not available, creating directly: {e}")
            from src.data.features.technical_indicators import TechnicalIndicators
            self.technical_calculator = TechnicalIndicators()

        try:
            self.statistical_calculator = self.resolve_dependency("StatisticalFeatureService")
        except Exception as e:
            self._logger.warning(f"StatisticalFeatureService not available, creating directly: {e}")
            from src.data.features.statistical_features import StatisticalFeatures
            self.statistical_calculator = StatisticalFeatures()

        # Initialize feature cache
        if self.fe_config.cache_ttl_hours > 0:
            self._feature_cache = FeatureCache(
                ttl_hours=self.fe_config.cache_ttl_hours,
                max_feature_sets=self.fe_config.cache_max_size,
            )

        self._logger.info(
            "Feature engineering service started successfully",
            config=self.fe_config.dict(),
            dependencies_resolved=1,
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
        computation_start = datetime.now(timezone.utc)
        warnings = []

        try:
            # Apply consistent boundary validation at ML module boundary
            from src.utils.decimal_utils import to_decimal
            from src.utils.messaging_patterns import BoundaryValidator

            # Validate request data at feature engineering boundary
            if not request.market_data:
                raise ValidationError("Market data is required for feature computation")
            if not request.symbol:
                raise ValidationError("Symbol is required for feature computation")

            # Apply consistent data transformation aligned with data module
            transformed_records = []
            if isinstance(request.market_data, list):
                for record in request.market_data:
                    if isinstance(record, dict):
                        transformed_record = {}
                        for key, value in record.items():
                            if (
                                key
                                in [
                                    "price",
                                    "close",
                                    "open",
                                    "high",
                                    "low",
                                    "open_price",
                                    "high_price",
                                    "low_price",
                                    "close_price",
                                    "volume",
                                    "bid",
                                    "ask",
                                ]
                                and value is not None
                            ):
                                transformed_record[key] = to_decimal(value)
                            else:
                                transformed_record[key] = value

                        # Apply boundary validation
                        try:
                            BoundaryValidator.validate_database_entity(
                                transformed_record, "validate"
                            )
                        except Exception as e:
                            warnings.append(f"Feature engineering boundary validation warning: {e}")

                        transformed_records.append(transformed_record)
                    else:
                        transformed_records.append(record)
            else:
                # Handle dict format
                transformed_record = {}
                for key, value in request.market_data.items():
                    if (
                        key
                        in [
                            "price",
                            "close",
                            "open",
                            "high",
                            "low",
                            "open_price",
                            "high_price",
                            "low_price",
                            "close_price",
                            "volume",
                            "bid",
                            "ask",
                        ]
                        and value is not None
                    ):
                        transformed_record[key] = to_decimal(value)
                    else:
                        transformed_record[key] = value
                transformed_records = [transformed_record]

            # Convert and validate market data using utils
            market_data = pd.DataFrame(transformed_records)
            market_data = validate_market_data(market_data)

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

            # Generate cache key using utils
            cache_key = generate_feature_cache_key(
                request.symbol, feature_types, str(hash(str(market_data.values.tobytes())))[:ML_MODEL_CONSTANTS["hash_digest_length"]]
            )

            # Check cache using utils
            if self._feature_cache:
                cached_features_dict = await self._feature_cache.get_features(cache_key)
                if cached_features_dict is not None:
                    computation_time = (
                        datetime.now(timezone.utc) - computation_start
                    ).total_seconds() * ML_MODEL_CONSTANTS["time_to_milliseconds"]

                    cached_features_df = pd.DataFrame(cached_features_dict["features"])
                    return FeatureResponse(
                        feature_set=FeatureSet(
                            feature_set_id=cache_key,
                            symbol=request.symbol,
                            features=cached_features_df.to_dict("records"),
                            feature_names=list(cached_features_df.columns),
                            computation_time_ms=ML_MODEL_CONSTANTS["cached_computation_time"],
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
                self._logger.info("Feature selection skipped - no target data provided")

            # Preprocessing if requested
            preprocessing_info: dict[str, Any] = {}
            if request.enable_preprocessing and self.fe_config.enable_preprocessing:
                features_df, preprocessing_info = await self._preprocess_features(
                    features_df, request.scaling_method
                )

            # Cache the results using utils
            if self._feature_cache:
                await self._feature_cache.cache_features(
                    cache_key,
                    {
                        "features": features_df.to_dict("records"),
                        "feature_names": list(features_df.columns),
                        "computation_time": (
                            datetime.now(timezone.utc) - computation_start
                        ).total_seconds()
                        * ML_MODEL_CONSTANTS["time_to_milliseconds"],
                    },
                )

            # Create feature set
            feature_set = FeatureSet(
                feature_set_id=cache_key,
                symbol=request.symbol,
                features=features_df.to_dict("records"),
                feature_names=list(features_df.columns),
                computation_time_ms=(datetime.now(timezone.utc) - computation_start).total_seconds()
                * ML_MODEL_CONSTANTS["time_to_milliseconds"],
            )

            computation_time = (
                datetime.now(timezone.utc) - computation_start
            ).total_seconds() * ML_MODEL_CONSTANTS["time_to_milliseconds"]

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
            # Apply consistent error propagation patterns
            from src.utils.messaging_patterns import ErrorPropagationMixin

            computation_time = (
                datetime.now(timezone.utc) - computation_start
            ).total_seconds() * ML_MODEL_CONSTANTS["time_to_milliseconds"]
            error_msg = f"Feature computation failed for {request.symbol}: {e}"

            # Use consistent error propagation
            error_propagator = ErrorPropagationMixin()
            try:
                if isinstance(e, ValidationError):
                    error_propagator.propagate_validation_error(
                        e, "feature_engineering.compute_features"
                    )
                elif isinstance(e, DataError):
                    error_propagator.propagate_service_error(
                        e, "feature_engineering.compute_features"
                    )
                else:
                    error_propagator.propagate_service_error(
                        e, "feature_engineering.compute_features"
                    )
            except Exception as prop_error:
                # Fallback if error propagation fails
                self._logger.warning(f"Error propagation failed: {prop_error}")

            self._logger.error(
                "Feature computation failed",
                symbol=request.symbol,
                error=str(e),
            )

            return FeatureResponse(
                feature_set=FeatureSet(
                    feature_set_id="error",
                    symbol=request.symbol,
                    features={},
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

        # Multi-period returns with Decimal precision
        for period in ML_FEATURE_CONSTANTS["price_return_periods"]:
            # Calculate returns with Decimal precision
            returns = pd.Series(index=data.index, dtype=object)
            log_returns = pd.Series(index=data.index, dtype=object)

            for i in range(period, len(data)):
                current_price = data["close"].iloc[i]
                past_price = data["close"].iloc[i - period]

                if pd.notna(current_price) and pd.notna(past_price) and past_price != 0:
                    current_decimal = Decimal(str(current_price))
                    past_decimal = Decimal(str(past_price))

                    # Calculate return with Decimal precision
                    return_decimal = (current_decimal / past_decimal) - Decimal("1")
                    returns.iloc[i] = return_decimal

                    # Calculate log return with Decimal precision
                    ratio = current_decimal / past_decimal
                    log_returns.iloc[i] = Decimal(str(np.log(float(ratio))))
                else:
                    returns.iloc[i] = None
                    log_returns.iloc[i] = None

            features[f"return_{period}d"] = returns
            features[f"log_return_{period}d"] = log_returns

        return features

    async def _compute_technical_features_async(
        self, market_data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Compute technical indicator features asynchronously."""
        try:
            # Convert DataFrame to list of MarketData objects as expected by data module
            market_data_list = []
            for _, row in market_data.iterrows():

                market_data_obj = MarketData(
                    symbol=symbol,
                    timestamp=row.get("timestamp", datetime.now(timezone.utc)),
                    price=row.get("close", row.get("price", 0)),
                    high_price=row.get("high", row.get("close", row.get("price", 0))),
                    low_price=row.get("low", row.get("close", row.get("price", 0))),
                    volume=row.get("volume", 0),
                )
                market_data_list.append(market_data_obj)

            # Use the correct method signature from TechnicalIndicators
            indicators = ["sma_20", "ema_20", "rsi_14", "macd", "bollinger_bands", "atr_14"]
            result_dict = await self.technical_calculator.calculate_indicators_batch(
                symbol=symbol, data=market_data_list, indicators=indicators
            )

            # Convert result dictionary to DataFrame format
            features_dict = {}
            for indicator, value in result_dict.items():
                if isinstance(value, dict):
                    # Handle MACD and Bollinger Bands which return dicts
                    for sub_key, sub_value in value.items():
                        # Convert to Decimal for financial precision
                        decimal_value = Decimal(str(sub_value)) if not isinstance(sub_value, Decimal) else sub_value
                        features_dict[f"{indicator}_{sub_key}"] = [decimal_value] * len(market_data)
                else:
                    # Handle single values - Convert to Decimal for financial precision
                    decimal_value = Decimal(str(value)) if not isinstance(value, Decimal) else value
                    features_dict[indicator] = [decimal_value] * len(market_data)

            return pd.DataFrame(features_dict, index=market_data.index)

        except Exception as e:
            self._logger.warning(f"Technical features computation failed: {e}")
            return pd.DataFrame(index=market_data.index)

    async def _compute_statistical_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute statistical features asynchronously."""
        try:
            # Get a dummy symbol from the first row or use default
            symbol = "STATS"

            # Add market data to the statistical calculator first
            for _, row in market_data.iterrows():

                market_data_obj = MarketData(
                    symbol=symbol,
                    timestamp=row.get("timestamp", datetime.now(timezone.utc)),
                    price=row.get("close", row.get("price", 0)),
                    open_price=row.get("open", row.get("close", row.get("price", 0))),
                    high_price=row.get("high", row.get("close", row.get("price", 0))),
                    low_price=row.get("low", row.get("close", row.get("price", 0))),
                    volume=row.get("volume", 0),
                )
                await self.statistical_calculator.add_market_data(market_data_obj)

            # Use the correct method signature from StatisticalFeatures
            features = ["ROLLING_STATS", "AUTOCORRELATION", "REGIME"]
            result_dict = await self.statistical_calculator.calculate_batch_features(
                symbol=symbol, features=features
            )

            # Convert result dictionary to DataFrame format
            features_dict = {}
            for feature_name, result in result_dict.items():
                if result and result.value:
                    if isinstance(result.value, dict):
                        # Handle dict results like rolling stats
                        for sub_key, sub_value in result.value.items():
                            if isinstance(sub_value, (int, float, Decimal)):
                                # Convert to Decimal for financial precision
                                decimal_value = Decimal(str(sub_value)) if not isinstance(sub_value, Decimal) else sub_value
                                features_dict[f"{feature_name.lower()}_{sub_key}"] = [decimal_value] * len(market_data)
                    elif isinstance(result.value, (int, float, Decimal)):
                        # Handle single numeric values - Convert to Decimal for financial precision
                        decimal_value = Decimal(str(result.value)) if not isinstance(result.value, Decimal) else result.value
                        features_dict[feature_name.lower()] = [decimal_value] * len(market_data)

            return pd.DataFrame(features_dict, index=market_data.index)

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
        sma_periods = ML_FEATURE_CONSTANTS["technical_periods"]["volume_sma_periods"]
        features[f"volume_sma_{sma_periods[0]}"] = data["volume"].rolling(sma_periods[0]).mean()
        features[f"volume_sma_{sma_periods[1]}"] = data["volume"].rolling(sma_periods[1]).mean()

        # Safe division
        volume_sma_col = f"volume_sma_{sma_periods[1]}"
        volume_sma_mask = features[volume_sma_col] != 0
        features["volume_ratio"] = np.nan
        features.loc[volume_sma_mask, "volume_ratio"] = (
            data.loc[volume_sma_mask, "volume"] / features.loc[volume_sma_mask, volume_sma_col]
        )
        features["volume_change"] = data["volume"].pct_change()

        # Volume-price features
        vwap_period = sma_periods[1]
        volume_sum = data["volume"].rolling(vwap_period).sum()
        volume_sum_mask = volume_sum != 0
        features["vwap"] = np.nan
        features.loc[volume_sum_mask, "vwap"] = (
            data.loc[volume_sum_mask, "close"] * data.loc[volume_sum_mask, "volume"]
        ).rolling(vwap_period).sum() / volume_sum.loc[volume_sum_mask]
        # Volume-price features - use Decimal for financial precision
        close_decimal = data["close"].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
        volume_decimal = data["volume"].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)

        features["price_volume"] = close_decimal * volume_decimal
        price_volume_shift = features["price_volume"].shift(1)
        features["volume_price_trend"] = (
            features["price_volume"] / price_volume_shift - Decimal("1")
        ).where(price_volume_shift.notna() & (price_volume_shift != 0))

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
        vol_periods = ML_FEATURE_CONSTANTS["technical_periods"]["volatility_periods"]
        for period in vol_periods:
            features[f"close_volatility_{period}"] = data["close"].rolling(period).std()

        # True Range and Average True Range
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        features["true_range"] = np.maximum(high_low, np.maximum(high_close, low_close))
        atr_period = ML_FEATURE_CONSTANTS["technical_periods"]["atr_period"]
        features[f"atr_{atr_period}"] = features["true_range"].rolling(atr_period).mean()

        # Volatility ratios
        short_vol_col = f"close_volatility_{vol_periods[0]}"
        long_vol_col = f"close_volatility_{vol_periods[2]}"
        vol_mask = features[long_vol_col] != 0
        features["volatility_ratio"] = np.nan
        features.loc[vol_mask, "volatility_ratio"] = (
            features.loc[vol_mask, short_vol_col] / features.loc[vol_mask, long_vol_col]
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
        roc_periods = ML_FEATURE_CONSTANTS["technical_periods"]["momentum_roc_periods"]
        for period in roc_periods:
            close_shifted = data["close"].shift(period)
            close_shifted_mask = close_shifted != 0
            features[f"roc_{period}"] = np.nan
            features.loc[close_shifted_mask, f"roc_{period}"] = (
                (data.loc[close_shifted_mask, "close"] - close_shifted.loc[close_shifted_mask])
                / close_shifted.loc[close_shifted_mask]
            ) * ML_FEATURE_CONSTANTS["percentage_multiplier"]

        # RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rsi_period = ML_FEATURE_CONSTANTS["technical_periods"]["rsi_period"]
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()

        avg_loss_mask = avg_loss != 0
        rs_values = np.full_like(avg_gain, np.nan)
        rs_values[avg_loss_mask] = avg_gain[avg_loss_mask] / avg_loss[avg_loss_mask]
        multiplier = ML_FEATURE_CONSTANTS["percentage_multiplier"]
        features[f"rsi_{rsi_period}"] = multiplier - (
            multiplier / (1 + pd.Series(rs_values, index=data.index))
        )

        # Stochastic oscillator
        stoch_period = ML_FEATURE_CONSTANTS["technical_periods"]["stoch_period"]
        stoch_smooth = ML_FEATURE_CONSTANTS["technical_periods"]["stoch_smooth"]
        lowest_low = data["low"].rolling(stoch_period).min()
        highest_high = data["high"].rolling(stoch_period).max()

        hh_ll_diff = highest_high - lowest_low
        hh_ll_diff_mask = hh_ll_diff != 0
        features["stoch_k"] = np.nan
        features.loc[hh_ll_diff_mask, "stoch_k"] = (
            (data.loc[hh_ll_diff_mask, "close"] - lowest_low.loc[hh_ll_diff_mask])
            / hh_ll_diff.loc[hh_ll_diff_mask]
        ) * ML_FEATURE_CONSTANTS["percentage_multiplier"]
        features["stoch_d"] = features["stoch_k"].rolling(stoch_smooth).mean()

        return features

    async def _compute_trend_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute trend-based features asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._compute_trend_features, market_data)

    def _compute_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute trend-based features."""
        features = pd.DataFrame(index=data.index)

        # Moving averages
        all_periods = (
            ML_FEATURE_CONSTANTS["technical_periods"]["short"]
            + ML_FEATURE_CONSTANTS["technical_periods"]["medium"]
            + ML_FEATURE_CONSTANTS["technical_periods"]["long"]
        )
        for period in all_periods:
            features[f"sma_{period}"] = data["close"].rolling(period).mean()
            features[f"ema_{period}"] = data["close"].ewm(span=period).mean()

        # Moving average crossovers
        for short, long in ML_FEATURE_CONSTANTS["ma_crossover_pairs"]:
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
        for period in ML_FEATURE_CONSTANTS["price_vs_ma_periods"]:
            if f"sma_{period}" in features.columns:
                sma_mask = features[f"sma_{period}"] != 0
                features[f"price_vs_sma_{period}"] = None
                # Use Decimal for financial precision
                close_decimal = data.loc[sma_mask, "close"].apply(lambda x: Decimal(str(x)))
                sma_decimal = features.loc[sma_mask, f"sma_{period}"].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
                features.loc[sma_mask, f"price_vs_sma_{period}"] = close_decimal / sma_decimal

            if f"ema_{period}" in features.columns:
                ema_mask = features[f"ema_{period}"] != 0
                features[f"price_vs_ema_{period}"] = None
                # Use Decimal for financial precision
                close_decimal = data.loc[ema_mask, "close"].apply(lambda x: Decimal(str(x)))
                ema_decimal = features.loc[ema_mask, f"ema_{period}"].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
                features.loc[ema_mask, f"price_vs_ema_{period}"] = close_decimal / ema_decimal

        # Trend strength
        short_ma_period = ML_FEATURE_CONSTANTS["trend_comparison_periods"]["short"]
        medium_ma_period = ML_FEATURE_CONSTANTS["trend_comparison_periods"]["medium"]
        short_ma_col = f"sma_{short_ma_period}"
        medium_ma_col = f"sma_{medium_ma_period}"

        if short_ma_col in features.columns and medium_ma_col in features.columns:
            ma_mask = features[medium_ma_col] != 0
            features["trend_strength"] = np.nan
            features.loc[ma_mask, "trend_strength"] = (
                features.loc[ma_mask, short_ma_col] - features.loc[ma_mask, medium_ma_col]
            ) / features.loc[ma_mask, medium_ma_col]

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
            classification_threshold = ML_FEATURE_CONSTANTS["classification_threshold"]
            is_categorical = (
                target_series.dtype in ["object", "category"]
                or target_series.nunique() < classification_threshold
            )

            if method == "mutual_info":
                if is_categorical:
                    score_func = mutual_info_classif
                else:
                    score_func = mutual_info_regression
            elif method == "f_test":
                if is_categorical:
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
                name: Decimal(str(score))
                for name, score in zip(
                    selected_feature_names, scores[selector.get_support()], strict=False
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
            raise ModelError(f"Feature selection failed: {e}") from e

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
            raise ModelError(f"Feature preprocessing failed: {e}") from e

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

    # Cache management now handled by utils

    # Service Health and Metrics
    async def _service_health_check(self) -> "HealthStatus":
        """Feature engineering service specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check dependencies
            if not all([self.data_service, self.technical_calculator, self.statistical_calculator]):
                return HealthStatus.UNHEALTHY

            # Check cache size using utils
            if self._feature_cache:
                cache_size = await self._feature_cache.size()
                if cache_size > self.fe_config.cache_max_size:
                    return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Feature engineering service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    async def get_feature_engineering_metrics(self) -> dict[str, Any]:
        """Get feature engineering service metrics."""
        cache_size = 0
        if self._feature_cache:
            cache_size = await self._feature_cache.size()

        return {
            "cached_features": cache_size,
            "fitted_scalers": len(self._scalers),
            "fitted_selectors": len(self._selectors),
            "executor_workers": self.fe_config.computation_workers,
        }

    async def clear_cache(self) -> dict[str, int]:
        """Clear feature engineering cache."""
        cache_size = 0
        if self._feature_cache:
            cache_size = await self._feature_cache.clear()

        scaler_count = len(self._scalers)
        selector_count = len(self._selectors)

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
            FeatureEngineeringConfig(**(fe_config_dict if isinstance(fe_config_dict, dict) else {}))
            return True
        except Exception as e:
            self._logger.error(
                "Feature engineering service configuration validation failed", error=str(e)
            )
            return False
