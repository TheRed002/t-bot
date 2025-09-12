"""
FeatureStore - Enterprise-Grade Feature Management System

This module provides a comprehensive feature store for managing financial features,
calculations, and ML pipelines. It eliminates duplicate calculations across the
system and provides consistent feature management for trading strategies.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- DataService: For data access and caching
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.core import BaseComponent, HealthCheckResult, HealthStatus
from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.types import MarketData

# Import from P-002A error handling
from src.error_handling import ErrorHandler

# Import from P-007A utilities
from src.utils.decimal_utils import to_decimal
from src.utils.decorators import time_execution


class FeatureType(Enum):
    """Feature type enumeration."""

    TECHNICAL_INDICATOR = "technical_indicator"
    STATISTICAL_FEATURE = "statistical_feature"
    ALTERNATIVE_FEATURE = "alternative_feature"
    DERIVED_FEATURE = "derived_feature"
    ML_FEATURE = "ml_feature"


class CalculationStatus(Enum):
    """Feature calculation status."""

    PENDING = "pending"
    CALCULATING = "calculating"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class FeatureMetadata:
    """Feature metadata for tracking and versioning."""

    feature_id: str
    feature_name: str
    feature_type: FeatureType
    version: str = "1.0.0"
    dependencies: list[str] = field(default_factory=list)
    calculation_cost: float = 1.0  # Relative cost (1.0 = baseline)
    cache_ttl: int = 300  # Cache TTL in seconds
    data_requirements: dict[str, Any] = field(default_factory=dict)
    calculation_parameters: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeatureValue:
    """Feature calculation result."""

    feature_id: str
    symbol: str
    value: Decimal | dict[str, Decimal] | None
    timestamp: datetime
    status: CalculationStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    calculation_time_ms: float = 0.0
    cache_hit: bool = False


class FeatureRequest(BaseModel):
    """Feature calculation request model."""

    symbol: str = Field(..., min_length=1, max_length=20)
    feature_types: list[str] = Field(..., min_length=1)  # Changed from feature_names to match data.types
    lookback_period: int = Field(default=100, ge=1, le=5000)
    parameters: dict[str, Any] = Field(default_factory=dict)
    cache_result: bool = True  # Changed from use_cache to match data.types
    force_recalculation: bool = False
    priority: int = Field(default=5, ge=1, le=10)

    @field_validator("feature_types")
    @classmethod
    def validate_feature_types(cls, v):
        if not v:
            raise ValidationError("At least one feature name is required")
        return v


class FeatureCalculationPipeline:
    """Feature calculation pipeline for efficient batch processing."""

    def __init__(self, feature_store: "FeatureStore"):
        self.feature_store = feature_store
        self.logger = feature_store.logger
        self.active_calculations: dict[str, asyncio.Task] = {}

    async def calculate_batch(
        self, requests: list[FeatureRequest]
    ) -> dict[str, list[FeatureValue]]:
        """Calculate features in batch for efficiency."""
        results = {}

        # Group requests by symbol for data efficiency
        symbol_groups = {}
        for request in requests:
            if request.symbol not in symbol_groups:
                symbol_groups[request.symbol] = []
            symbol_groups[request.symbol].append(request)

        # Process each symbol group
        for symbol, symbol_requests in symbol_groups.items():
            try:
                symbol_results = await self._calculate_symbol_batch(symbol, symbol_requests)
                results[symbol] = symbol_results
            except Exception as e:
                self.logger.error(f"Batch calculation failed for {symbol}: {e}")
                results[symbol] = []

        return results

    async def _calculate_symbol_batch(
        self, symbol: str, requests: list[FeatureRequest]
    ) -> list[FeatureValue]:
        """Calculate features for a single symbol."""
        results = []

        # Get all unique feature names
        all_features = set()
        for request in requests:
            all_features.update(request.feature_types)

        # Get market data once for all calculations
        max_lookback = max(request.lookback_period for request in requests)
        market_data = await self.feature_store._get_market_data(symbol, max_lookback)

        if not market_data:
            self.logger.warning(f"No market data available for {symbol}")
            return results

        # Calculate each feature
        for feature_name in all_features:
            try:
                feature_value = await self.feature_store._calculate_single_feature(
                    symbol, feature_name, market_data, {}
                )
                if feature_value:
                    results.append(feature_value)
            except Exception as e:
                self.logger.error(f"Feature calculation failed for {feature_name}: {e}")

        return results


class FeatureStore(BaseComponent):
    """
    Enterprise-grade FeatureStore for financial feature management.

    This service provides:
    - Centralized feature calculation and caching
    - Elimination of duplicate calculations
    - Consistent feature versioning and metadata
    - High-performance batch processing
    - Dependency graph management
    - ML pipeline integration
    """

    def __init__(self, config: Config, data_service=None):
        """Initialize the FeatureStore."""
        super().__init__()
        self.config = config
        self.data_service = data_service
        self.error_handler = ErrorHandler(config)

        # Configuration
        self._setup_configuration()

        # Feature registry
        self._features: dict[str, FeatureMetadata] = {}
        self._calculators: dict[str, Callable] = {}

        # Cache and computation tracking
        self._feature_cache: dict[str, FeatureValue] = {}
        self._calculation_locks: dict[str, asyncio.Lock] = {}

        # Pipeline for batch processing
        self.calculation_pipeline = FeatureCalculationPipeline(self)

        # Metrics
        self._metrics = {
            "total_calculations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failed_calculations": 0,
            "avg_calculation_time": 0.0,
            "duplicate_calculations_avoided": 0,
        }

        # Background task management
        self._background_tasks: list[asyncio.Task] = []

        self._initialized = False

    def _setup_configuration(self) -> None:
        """Setup FeatureStore configuration."""
        feature_config = getattr(self.config, "feature_store", {})

        self.cache_config = {
            "max_cache_size": feature_config.get("max_cache_size", 10000),
            "default_ttl": feature_config.get("default_ttl", 300),
            "cleanup_interval": feature_config.get("cleanup_interval", 3600),
        }

        self.calculation_config = {
            "max_concurrent_calculations": feature_config.get("max_concurrent_calculations", 10),
            "calculation_timeout": feature_config.get("calculation_timeout", 30),
            "batch_size": feature_config.get("batch_size", 100),
        }

    async def initialize(self) -> None:
        """Initialize the FeatureStore."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing FeatureStore...")

            # Register built-in features
            await self._register_builtin_features()

            # Initialize technical indicators integration
            await self._initialize_technical_indicators()

            # Initialize statistical features
            await self._initialize_statistical_features()

            # Initialize alternative features
            await self._initialize_alternative_features()

            # Start cache cleanup task
            cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            self._background_tasks.append(cleanup_task)

            self._initialized = True
            self.logger.info("FeatureStore initialized successfully")

        except Exception as e:
            self.logger.error(f"FeatureStore initialization failed: {e}")
            raise

    async def _register_builtin_features(self) -> None:
        """Register built-in feature definitions."""
        # Technical indicators
        await self.register_feature(
            "sma_20",
            FeatureType.TECHNICAL_INDICATOR,
            self._calculate_sma,
            {"period": 20, "min_data_points": 20},
        )

        await self.register_feature(
            "ema_20",
            FeatureType.TECHNICAL_INDICATOR,
            self._calculate_ema,
            {"period": 20, "min_data_points": 20},
        )

        await self.register_feature(
            "rsi_14",
            FeatureType.TECHNICAL_INDICATOR,
            self._calculate_rsi,
            {"period": 14, "min_data_points": 15},
        )

        await self.register_feature(
            "macd",
            FeatureType.TECHNICAL_INDICATOR,
            self._calculate_macd,
            {"fast": 12, "slow": 26, "signal": 9, "min_data_points": 35},
        )

        await self.register_feature(
            "bollinger_bands",
            FeatureType.TECHNICAL_INDICATOR,
            self._calculate_bollinger_bands,
            {"period": 20, "std_dev": 2, "min_data_points": 20},
        )

        # Statistical features
        await self.register_feature(
            "price_volatility",
            FeatureType.STATISTICAL_FEATURE,
            self._calculate_volatility,
            {"window": 20, "min_data_points": 20},
        )

        await self.register_feature(
            "price_momentum",
            FeatureType.STATISTICAL_FEATURE,
            self._calculate_momentum,
            {"window": 10, "min_data_points": 11},
        )

        await self.register_feature(
            "volume_trend",
            FeatureType.STATISTICAL_FEATURE,
            self._calculate_volume_trend,
            {"window": 20, "min_data_points": 20},
        )

        self.logger.info("Built-in features registered")

    async def _initialize_technical_indicators(self) -> None:
        """Initialize technical indicators integration."""
        try:
            from src.data.features.technical_indicators import TechnicalIndicators

            self.technical_indicators = TechnicalIndicators(self.config, feature_store=self)
            self.logger.info("Technical indicators integration enabled")
        except ImportError as e:
            self.logger.warning(f"Technical indicators not available: {e}")
            self.technical_indicators = None

    async def _initialize_statistical_features(self) -> None:
        """Initialize statistical features integration."""
        try:
            from src.data.features.statistical_features import StatisticalFeatures

            self.statistical_features = StatisticalFeatures(self.config, feature_store=self)
            self.logger.info("Statistical features integration enabled")
        except ImportError as e:
            self.logger.warning(f"Statistical features not available: {e}")
            self.statistical_features = None

    async def _initialize_alternative_features(self) -> None:
        """Initialize alternative features integration."""
        try:
            from src.data.features.alternative_features import AlternativeFeatures

            self.alternative_features = AlternativeFeatures(self.config, feature_store=self)
            self.logger.info("Alternative features integration enabled")
        except ImportError as e:
            self.logger.warning(f"Alternative features not available: {e}")
            self.alternative_features = None

    async def register_feature(
        self,
        feature_name: str,
        feature_type: FeatureType,
        calculator: Callable,
        parameters: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        cache_ttl: int | None = None,
    ) -> None:
        """Register a new feature with the store."""
        feature_id = f"{feature_type.value}:{feature_name}"

        metadata = FeatureMetadata(
            feature_id=feature_id,
            feature_name=feature_name,
            feature_type=feature_type,
            dependencies=dependencies or [],
            cache_ttl=cache_ttl or self.cache_config["default_ttl"],
            calculation_parameters=parameters or {},
        )

        self._features[feature_name] = metadata
        self._calculators[feature_name] = calculator

        self.logger.debug(f"Registered feature: {feature_name}")

    @time_execution
    async def calculate_features(self, request: FeatureRequest) -> list[FeatureValue]:
        """Calculate requested features with intelligent caching."""
        if not self._initialized:
            await self.initialize()

        results = []

        # Check cache first
        if request.cache_result and not request.force_recalculation:
            cached_results = await self._get_cached_features(request)
            if cached_results:
                self._metrics["cache_hits"] += len(cached_results)
                results.extend(cached_results)

                # Remove cached features from request
                cached_names = {result.feature_id.split(":")[-1] for result in cached_results}
                request.feature_types = [
                    name for name in request.feature_types if name not in cached_names
                ]

        # Calculate remaining features
        if request.feature_types:
            calculated_results = await self._calculate_features_batch(request)
            results.extend(calculated_results)

            # Cache results
            if request.cache_result:
                await self._cache_results(calculated_results)

        return results

    async def _get_cached_features(self, request: FeatureRequest) -> list[FeatureValue]:
        """Get features from cache if available and valid."""
        cached_results = []

        for feature_name in request.feature_types:
            cache_key = self._build_cache_key(request.symbol, feature_name, request.parameters)

            if cache_key in self._feature_cache:
                cached_value = self._feature_cache[cache_key]

                # Check if cache is still valid
                if self._is_cache_valid(cached_value, feature_name):
                    cached_value.cache_hit = True
                    cached_results.append(cached_value)

        return cached_results

    def _is_cache_valid(self, cached_value: FeatureValue, feature_name: str) -> bool:
        """Check if cached value is still valid."""
        if feature_name not in self._features:
            return False

        metadata = self._features[feature_name]
        age = (datetime.now(timezone.utc) - cached_value.timestamp).total_seconds()

        return age <= metadata.cache_ttl

    async def _calculate_features_batch(self, request: FeatureRequest) -> list[FeatureValue]:
        """Calculate features in batch for efficiency."""
        self._metrics["cache_misses"] += len(request.feature_types)

        # Get market data once for all calculations
        market_data = await self._get_market_data(request.symbol, request.lookback_period)

        if not market_data:
            self.logger.warning(f"No market data available for {request.symbol}")
            return []

        results = []
        calculation_tasks = []

        # Create calculation tasks
        for feature_name in request.feature_types:
            task = asyncio.create_task(
                self._calculate_single_feature(
                    request.symbol, feature_name, market_data, request.parameters
                )
            )
            calculation_tasks.append(task)

        # Wait for all calculations to complete
        completed_results = await asyncio.gather(*calculation_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(completed_results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Feature calculation failed for {request.feature_types[i]}: {result}"
                )
                self._metrics["failed_calculations"] += 1
            elif result:
                results.append(result)
                self._metrics["total_calculations"] += 1

        return results

    async def _calculate_single_feature(
        self,
        symbol: str,
        feature_name: str,
        market_data: list[MarketData],
        parameters: dict[str, Any],
    ) -> FeatureValue | None:
        """Calculate a single feature."""
        start_time = datetime.now(timezone.utc)

        try:
            # Check if feature is registered
            if feature_name not in self._features:
                self.logger.error(f"Unknown feature: {feature_name}")
                return None

            metadata = self._features[feature_name]
            calculator = self._calculators[feature_name]

            # Merge parameters
            calc_params = metadata.calculation_parameters.copy()
            calc_params.update(parameters)

            # Check minimum data requirements
            min_data_points = calc_params.get("min_data_points", 1)
            if len(market_data) < min_data_points:
                self.logger.warning(
                    f"Insufficient data for {feature_name}: {len(market_data)} < {min_data_points}"
                )
                return None

            # Use lock to prevent duplicate calculations
            lock_key = f"{symbol}:{feature_name}"
            if lock_key not in self._calculation_locks:
                self._calculation_locks[lock_key] = asyncio.Lock()

            async with self._calculation_locks[lock_key]:
                # Check cache again after acquiring lock
                cache_key = self._build_cache_key(symbol, feature_name, parameters)
                if cache_key in self._feature_cache:
                    cached_value = self._feature_cache[cache_key]
                    if self._is_cache_valid(cached_value, feature_name):
                        self._metrics["duplicate_calculations_avoided"] += 1
                        return cached_value

                # Perform calculation
                value = await calculator(market_data, **calc_params)

                # Calculate processing time
                calculation_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                # Create feature value
                feature_value = FeatureValue(
                    feature_id=metadata.feature_id,
                    symbol=symbol,
                    value=value,
                    timestamp=datetime.now(timezone.utc),
                    status=(
                        CalculationStatus.COMPLETED
                        if value is not None
                        else CalculationStatus.FAILED
                    ),
                    calculation_time_ms=calculation_time,
                    metadata={"parameters": calc_params},
                )

                return feature_value

        except Exception as e:
            self.logger.error(f"Feature calculation error for {feature_name}: {e}")
            return FeatureValue(
                feature_id=f"error:{feature_name}",
                symbol=symbol,
                value=None,
                timestamp=datetime.now(timezone.utc),
                status=CalculationStatus.FAILED,
                metadata={"error": str(e)},
            )

    async def _get_market_data(self, symbol: str, lookback_period: int) -> list[MarketData]:
        """Get market data for feature calculations."""
        try:
            if not self.data_service:
                self.logger.error("DataService not available for market data retrieval")
                return []

            # Create data request
            from src.data.types import DataRequest

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=lookback_period)

            request = DataRequest(
                symbol=symbol,
                exchange="all",  # Get from all exchanges
                start_time=start_time,
                end_time=end_time,
                limit=lookback_period * 1440,  # Assume minute data
                cache_result=True,
            )

            db_records = await self.data_service.get_market_data(request)

            # Convert to MarketData objects
            market_data = []
            for record in db_records:
                market_data.append(
                    MarketData(
                        symbol=record.symbol,
                        close=to_decimal(record.price) if record.price else to_decimal(0),
                        volume=to_decimal(record.volume) if record.volume else to_decimal(0),
                        timestamp=record.timestamp,
                        high=to_decimal(record.high_price) if record.high_price else to_decimal(0),
                        low=to_decimal(record.low_price) if record.low_price else to_decimal(0),
                        open=to_decimal(record.open_price) if record.open_price else to_decimal(0),
                        bid_price=to_decimal(record.bid) if record.bid else None,
                        ask_price=to_decimal(record.ask) if record.ask else None,
                        exchange=getattr(record, "exchange", "unknown"),
                    )
                )

            return market_data

        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return []

    def _build_cache_key(self, symbol: str, feature_name: str, parameters: dict[str, Any]) -> str:
        """Build cache key for feature value."""
        try:
            param_hash = hash(json.dumps(parameters, sort_keys=True, default=str))
        except (TypeError, ValueError) as e:
            self.logger.warning(
                f"Failed to serialize parameters for cache key, using fallback: {e}"
            )
            param_hash = hash(str(sorted(parameters.items())))
        return f"{symbol}:{feature_name}:{param_hash}"

    async def _cache_results(self, results: list[FeatureValue]) -> None:
        """Cache calculation results."""
        for result in results:
            if result.status == CalculationStatus.COMPLETED:
                cache_key = self._build_cache_key(
                    result.symbol,
                    result.feature_id.split(":")[-1],
                    result.metadata.get("parameters", {}),
                )

                # Check cache size limit
                if len(self._feature_cache) >= self.cache_config["max_cache_size"]:
                    await self._cleanup_old_cache_entries()

                self._feature_cache[cache_key] = result

    async def _cleanup_old_cache_entries(self) -> None:
        """Cleanup old cache entries."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []

        for cache_key, cached_value in self._feature_cache.items():
            feature_name = cached_value.feature_id.split(":")[-1]
            if feature_name in self._features:
                metadata = self._features[feature_name]
                age = (current_time - cached_value.timestamp).total_seconds()

                if age > metadata.cache_ttl:
                    expired_keys.append(cache_key)

        # Remove expired entries
        for key in expired_keys:
            del self._feature_cache[key]

        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                await asyncio.sleep(self.cache_config["cleanup_interval"])
                await self._cleanup_old_cache_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")

    # Built-in feature calculators
    async def _calculate_sma(
        self, market_data: list[MarketData], period: int = 20, **kwargs
    ) -> Decimal | None:
        """Calculate Simple Moving Average."""
        try:
            prices = [to_decimal(data.price) for data in market_data[-period:] if data.price]
            if len(prices) < period:
                return None
            getcontext().prec = 16
            result = sum(prices) / to_decimal(len(prices))
            return result.quantize(Decimal("0.00000001"))
        except Exception as e:
            self.logger.error(f"Simple moving average calculation failed: {e}")
            return None

    async def _calculate_ema(
        self, market_data: list[MarketData], period: int = 20, **kwargs
    ) -> Decimal | None:
        """Calculate Exponential Moving Average."""
        try:
            prices = [to_decimal(data.price) for data in market_data if data.price]
            if len(prices) < period:
                return None

            # Calculate EMA
            getcontext().prec = 16
            multiplier = Decimal("2") / (to_decimal(period) + Decimal("1"))
            ema = prices[0]

            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (Decimal("1") - multiplier))

            return ema.quantize(Decimal("0.00000001"))
        except Exception as e:
            self.logger.error(f"Exponential moving average calculation failed: {e}")
            return None

    async def _calculate_rsi(
        self, market_data: list[MarketData], period: int = 14, **kwargs
    ) -> Decimal | None:
        """Calculate Relative Strength Index."""
        try:
            prices = [Decimal(str(data.price)) for data in market_data if data.price]
            if len(prices) < period + 1:
                return None

            # Calculate price changes
            gains = []
            losses = []

            for i in range(1, len(prices)):
                change = prices[i] - prices[i - 1]
                if change >= 0:
                    gains.append(change)
                    losses.append(Decimal("0"))
                else:
                    gains.append(Decimal("0"))
                    losses.append(abs(change))

            # Calculate average gains and losses
            getcontext().prec = 16
            avg_gain = sum(gains[-period:]) / Decimal(str(period))
            avg_loss = sum(losses[-period:]) / Decimal(str(period))

            if avg_loss == 0:
                return Decimal("100.0")

            rs = avg_gain / avg_loss
            rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

            return rsi.quantize(Decimal("0.0001"))
        except Exception as e:
            self.logger.error(f"Relative strength index calculation failed: {e}")
            return None

    async def _calculate_macd(
        self,
        market_data: list[MarketData],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        **kwargs,
    ) -> dict[str, Decimal] | None:
        """Calculate MACD."""
        try:
            prices = [Decimal(str(data.price)) for data in market_data if data.price]
            if len(prices) < slow + signal:
                return None

            # Calculate EMAs for all prices using Decimal precision
            getcontext().prec = 16
            fast_ema_values = []
            slow_ema_values = []

            # Calculate EMA values for each price point
            for i in range(len(prices)):
                if i < fast - 1:
                    fast_ema_values.append(None)
                else:
                    # Calculate fast EMA up to this point
                    ema = prices[0]
                    multiplier = Decimal("2") / (Decimal(str(fast)) + Decimal("1"))
                    for j in range(1, i + 1):
                        ema = (prices[j] * multiplier) + (ema * (Decimal("1") - multiplier))
                    fast_ema_values.append(ema)

                if i < slow - 1:
                    slow_ema_values.append(None)
                else:
                    # Calculate slow EMA up to this point
                    ema = prices[0]
                    multiplier = Decimal("2") / (Decimal(str(slow)) + Decimal("1"))
                    for j in range(1, i + 1):
                        ema = (prices[j] * multiplier) + (ema * (Decimal("1") - multiplier))
                    slow_ema_values.append(ema)

            # Calculate MACD line values
            macd_values = []
            for i in range(len(prices)):
                if fast_ema_values[i] is not None and slow_ema_values[i] is not None:
                    macd_values.append(fast_ema_values[i] - slow_ema_values[i])
                else:
                    macd_values.append(None)

            # Find first valid MACD value index
            first_valid_idx = next((i for i, v in enumerate(macd_values) if v is not None), None)
            if first_valid_idx is None or len(macd_values) - first_valid_idx < signal:
                return None

            # Calculate signal line (EMA of MACD line)
            valid_macd = [v for v in macd_values if v is not None]
            signal_ema = valid_macd[0]
            multiplier = Decimal("2") / (Decimal(str(signal)) + Decimal("1"))

            for i in range(1, len(valid_macd)):
                signal_ema = (valid_macd[i] * multiplier) + (
                    signal_ema * (Decimal("1") - multiplier)
                )

            # Use latest values
            macd_line = macd_values[-1]
            signal_line = signal_ema
            histogram = macd_line - signal_line

            return {
                "macd": macd_line.quantize(Decimal("0.00000001")),
                "signal": signal_line.quantize(Decimal("0.00000001")),
                "histogram": histogram.quantize(Decimal("0.00000001")),
            }
        except Exception as e:
            self.logger.error(f"Simple moving average calculation failed: {e}")
            return None

    async def _calculate_bollinger_bands(
        self,
        market_data: list[MarketData],
        period: int = 20,
        std_dev: Decimal = Decimal("2"),
        **kwargs,
    ) -> dict[str, Decimal] | None:
        """Calculate Bollinger Bands."""
        try:
            prices = [Decimal(str(data.price)) for data in market_data[-period:] if data.price]
            if len(prices) < period:
                return None

            # Calculate SMA and standard deviation using Decimal precision
            getcontext().prec = 16
            sma = sum(prices) / Decimal(str(len(prices)))

            # Calculate variance using Decimal arithmetic
            variance_sum = sum((price - sma) ** 2 for price in prices)
            variance = variance_sum / Decimal(str(len(prices)))

            # For standard deviation, we need to convert to float temporarily for sqrt
            std = Decimal(str(float(variance) ** 0.5))

            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)
            width = upper - lower

            return {
                "upper": upper.quantize(Decimal("0.00000001")),
                "middle": sma.quantize(Decimal("0.00000001")),
                "lower": lower.quantize(Decimal("0.00000001")),
                "width": width.quantize(Decimal("0.00000001")),
                "position": (
                    (prices[-1] - lower) / width if width != 0 else Decimal("0.5")
                ).quantize(Decimal("0.0001")),
            }
        except Exception as e:
            self.logger.error(f"Simple moving average calculation failed: {e}")
            return None

    async def _calculate_volatility(
        self, market_data: list[MarketData], window: int = 20, **kwargs
    ) -> Decimal | None:
        """Calculate price volatility."""
        try:
            prices = [Decimal(str(data.price)) for data in market_data[-window:] if data.price]
            if len(prices) < window:
                return None

            # Calculate returns using Decimal precision
            getcontext().prec = 16
            returns = []
            for i in range(1, len(prices)):
                if prices[i - 1] != 0:
                    ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                    returns.append(ret)

            # Calculate volatility (standard deviation of returns)
            if not returns:
                return None

            mean_return = sum(returns) / Decimal(str(len(returns)))
            variance_sum = sum((ret - mean_return) ** 2 for ret in returns)
            variance = variance_sum / Decimal(str(len(returns)))

            # For standard deviation, convert to float temporarily for sqrt
            volatility = Decimal(str(float(variance) ** 0.5))

            return volatility.quantize(Decimal("0.000001"))
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}")
            return None

    async def _calculate_momentum(
        self, market_data: list[MarketData], window: int = 10, **kwargs
    ) -> Decimal | None:
        """Calculate price momentum."""
        try:
            prices = [Decimal(str(data.price)) for data in market_data if data.price]
            if len(prices) < window + 1:
                return None

            getcontext().prec = 16
            current_price = prices[-1]
            past_price = prices[-(window + 1)]

            if past_price != 0:
                momentum = (current_price - past_price) / past_price
                return momentum.quantize(Decimal("0.000001"))
            else:
                return Decimal("0")
        except Exception as e:
            self.logger.error(f"Momentum calculation failed: {e}")
            return None

    async def _calculate_volume_trend(
        self, market_data: list[MarketData], window: int = 20, **kwargs
    ) -> Decimal | None:
        """Calculate volume trend."""
        try:
            volumes = [Decimal(str(data.volume)) for data in market_data[-window:] if data.volume]
            if len(volumes) < window:
                return None

            # Simple volume trend calculation using Decimal precision
            getcontext().prec = 16
            half_window = window // 2
            recent_avg = sum(volumes[-half_window:]) / Decimal(str(half_window))
            past_avg = sum(volumes[:half_window]) / Decimal(str(half_window))

            if past_avg == 0:
                return Decimal("0")

            trend = (recent_avg - past_avg) / past_avg
            return trend.quantize(Decimal("0.000001"))
        except Exception as e:
            self.logger.error(f"Volume trend calculation failed: {e}")
            return None

    def get_metrics(self) -> dict[str, Any]:
        """Get FeatureStore metrics."""
        total_requests = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        cache_hit_rate = self._metrics["cache_hits"] / max(1, total_requests)

        success_rate = self._metrics["total_calculations"] / max(
            1, self._metrics["total_calculations"] + self._metrics["failed_calculations"]
        )

        return {
            "total_calculations": self._metrics["total_calculations"],
            "cache_hits": self._metrics["cache_hits"],
            "cache_misses": self._metrics["cache_misses"],
            "failed_calculations": self._metrics["failed_calculations"],
            "cache_hit_rate": cache_hit_rate,
            "success_rate": success_rate,
            "duplicate_calculations_avoided": self._metrics["duplicate_calculations_avoided"],
            "registered_features": len(self._features),
            "cached_values": len(self._feature_cache),
            "active_locks": len(self._calculation_locks),
        }

    async def health_check(self) -> HealthCheckResult:
        """Perform FeatureStore health check."""
        status = HealthStatus.HEALTHY
        details = {
            "initialized": self._initialized,
            "metrics": self.get_metrics(),
            "components": {},
        }

        # Check data service connectivity
        if self.data_service:
            try:
                await self.data_service.health_check()
                details["components"]["data_service"] = "healthy"
            except Exception as e:
                details["components"]["data_service"] = f"unhealthy: {e}"
                status = HealthStatus.DEGRADED
        else:
            details["components"]["data_service"] = "not_configured"
            status = HealthStatus.DEGRADED

        return HealthCheckResult(
            status=status, details=details, message=f"FeatureStore health: {status.value}"
        )

    async def cleanup(self) -> None:
        """Cleanup FeatureStore resources."""
        try:
            # Cancel background tasks
            if self._background_tasks:
                for task in self._background_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.warning(f"Error cancelling feature store task: {e}")

                self._background_tasks.clear()

            # Clear caches
            self._feature_cache.clear()
            self._calculation_locks.clear()

            self._initialized = False
            self.logger.info("FeatureStore cleanup completed")

        except Exception as e:
            self.logger.error(f"FeatureStore cleanup error: {e}")
