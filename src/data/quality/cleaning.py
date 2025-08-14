"""
Data Cleaning Pipeline

This module provides comprehensive data cleaning and preprocessing functionality:
- Missing data imputation strategies
- Outlier handling (remove vs adjust)
- Data smoothing for noisy signals
- Duplicate detection and removal
- Data normalization and standardization

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np

from src.core.config import Config
from src.core.logging import get_logger

# Import from P-001 core components
from src.core.types import MarketData, Signal

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class CleaningStrategy(Enum):
    """Data cleaning strategy enumeration"""

    REMOVE = "remove"
    ADJUST = "adjust"
    IMPUTE = "impute"
    SMOOTH = "smooth"


class OutlierMethod(Enum):
    """Outlier detection method enumeration"""

    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"


@dataclass
class CleaningResult:
    """Data cleaning result record"""

    original_data: Any
    cleaned_data: Any
    applied_strategies: list[str]
    removed_count: int
    adjusted_count: int
    imputed_count: int
    timestamp: datetime
    metadata: dict[str, Any]


class DataCleaner:
    """
    Comprehensive data cleaning system for market data preprocessing.

    This class provides various cleaning strategies for handling missing data,
    outliers, noise, and data quality issues.
    """

    def __init__(self, config: Config | dict[str, Any]):
        """
        Initialize the data cleaner with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        # Allow dict-like configs in tests
        cfg_get = config.get if isinstance(config, dict) else getattr
        self.error_handler = ErrorHandler(config if not isinstance(config, dict) else Config())

        # Cleaning thresholds
        self.outlier_threshold = (
            cfg_get("outlier_threshold", 3.0)
            if isinstance(config, dict)
            else getattr(config, "outlier_threshold", 3.0)
        )
        self.missing_threshold = (
            cfg_get("missing_threshold", 0.1)
            if isinstance(config, dict)
            else getattr(config, "missing_threshold", 0.1)
        )
        self.smoothing_window = (
            cfg_get("smoothing_window", 5)
            if isinstance(config, dict)
            else getattr(config, "smoothing_window", 5)
        )
        self.duplicate_threshold = (
            cfg_get("duplicate_threshold", 1.0)
            if isinstance(config, dict)
            else getattr(config, "duplicate_threshold", 1.0)
        )

        # Data history for cleaning
        self.price_history: dict[str, list[float]] = {}
        self.volume_history: dict[str, list[float]] = {}
        self.max_history_size = (
            cfg_get("max_history_size", 1000)
            if isinstance(config, dict)
            else getattr(config, "max_history_size", 1000)
        )

        # Cleaning statistics
        self.cleaning_stats = {
            "total_processed": 0,
            "total_cleaned": 0,
            "outliers_removed": 0,
            "outliers_adjusted": 0,
            "missing_imputed": 0,
            "duplicates_removed": 0,
        }

        logger.info("DataCleaner initialized", config=config)

    @time_execution
    async def clean_market_data(self, data: MarketData) -> tuple[MarketData, CleaningResult]:
        """
        Clean market data using comprehensive cleaning pipeline.

        Args:
            data: Market data to clean

        Returns:
            Tuple of (cleaned_data, cleaning_result)
        """
        original_data = data
        applied_strategies = []
        removed_count = 0
        adjusted_count = 0
        imputed_count = 0

        try:
            # Handle None data
            if data is None:
                cleaning_result = CleaningResult(
                    original_data=None,
                    cleaned_data=None,
                    applied_strategies=[],
                    removed_count=0,
                    adjusted_count=0,
                    imputed_count=0,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "error": "Data is None",
                        "cleaning_stats": self.cleaning_stats.copy(),
                    },
                )
                return None, cleaning_result

            # Step 1: Handle missing data
            data, imputed_count = await self._handle_missing_data(data)
            if imputed_count > 0:
                applied_strategies.append("missing_data_imputation")

            # Step 2: Detect and handle outliers
            data, outlier_removed, outlier_adjusted = await self._handle_outliers(data)
            removed_count += outlier_removed
            adjusted_count += outlier_adjusted
            if outlier_removed > 0 or outlier_adjusted > 0:
                applied_strategies.append("outlier_handling")

            # Step 3: Smooth noisy data
            data = await self._smooth_data(data)
            applied_strategies.append("data_smoothing")

            # Step 4: Remove duplicates
            data, duplicate_removed = await self._remove_duplicates(data)
            removed_count += duplicate_removed
            if duplicate_removed > 0:
                applied_strategies.append("duplicate_removal")

            # Step 5: Normalize data
            data = await self._normalize_data(data)
            applied_strategies.append("data_normalization")

            # Update statistics
            self.cleaning_stats["total_processed"] += 1
            if applied_strategies:
                self.cleaning_stats["total_cleaned"] += 1
            self.cleaning_stats["outliers_removed"] += outlier_removed
            self.cleaning_stats["outliers_adjusted"] += outlier_adjusted
            self.cleaning_stats["missing_imputed"] += imputed_count
            self.cleaning_stats["duplicates_removed"] += duplicate_removed

            # Create cleaning result
            cleaning_result = CleaningResult(
                original_data=original_data,
                cleaned_data=data,
                applied_strategies=applied_strategies,
                removed_count=removed_count,
                adjusted_count=adjusted_count,
                imputed_count=imputed_count,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "symbol": data.symbol if data else "unknown",
                    "cleaning_stats": self.cleaning_stats.copy(),
                },
            )

            if applied_strategies:
                logger.info(
                    "Market data cleaned successfully",
                    symbol=data.symbol if data else "unknown",
                    applied_strategies=applied_strategies,
                    removed_count=removed_count,
                    adjusted_count=adjusted_count,
                    imputed_count=imputed_count,
                )
            else:
                logger.debug(
                    "Market data passed cleaning without changes",
                    symbol=data.symbol if data else "unknown",
                )

            return data, cleaning_result

        except Exception as e:
            logger.error(
                "Market data cleaning failed",
                symbol=data.symbol if data and data.symbol else "unknown",
                error=str(e),
            )
            cleaning_result = CleaningResult(
                original_data=original_data,
                cleaned_data=original_data,
                applied_strategies=[],
                removed_count=0,
                adjusted_count=0,
                imputed_count=0,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "cleaning_stats": self.cleaning_stats.copy(),
                },
            )
            return original_data, cleaning_result

    @time_execution
    async def clean_signal_data(self, signals: list[Signal]) -> tuple[list[Signal], CleaningResult]:
        """
        Clean trading signal data.

        Args:
            signals: List of trading signals to clean

        Returns:
            Tuple of (cleaned_signals, cleaning_result)
        """
        original_signals = signals
        cleaned_signals = []
        applied_strategies = []
        removed_count = 0
        adjusted_count = 0
        imputed_count = 0

        try:
            for signal in signals:
                # Validate signal
                if not await self._is_valid_signal(signal):
                    removed_count += 1
                    continue

                # Clean signal confidence
                cleaned_confidence = await self._clean_confidence(signal.confidence)
                if cleaned_confidence != signal.confidence:
                    adjusted_count += 1
                    signal.confidence = cleaned_confidence

                # Remove duplicate signals (same symbol, direction, timestamp)
                if not await self._is_duplicate_signal(signal, cleaned_signals):
                    cleaned_signals.append(signal)
                else:
                    removed_count += 1

            if removed_count > 0:
                applied_strategies.append("duplicate_signal_removal")
            if adjusted_count > 0:
                applied_strategies.append("signal_adjustment")

            # Update statistics
            self.cleaning_stats["total_processed"] += len(signals)
            if applied_strategies:
                self.cleaning_stats["total_cleaned"] += 1
            self.cleaning_stats["outliers_removed"] += removed_count
            self.cleaning_stats["outliers_adjusted"] += adjusted_count

            cleaning_result = CleaningResult(
                original_data=original_signals,
                cleaned_data=cleaned_signals,
                applied_strategies=applied_strategies,
                removed_count=removed_count,
                adjusted_count=adjusted_count,
                imputed_count=imputed_count,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "signal_count": len(cleaned_signals),
                    "cleaning_stats": self.cleaning_stats.copy(),
                },
            )

            if applied_strategies:
                logger.info(
                    "Signal data cleaned successfully",
                    original_count=len(signals),
                    cleaned_count=len(cleaned_signals),
                    applied_strategies=applied_strategies,
                )
            else:
                logger.debug("Signal data passed cleaning without changes")

            return cleaned_signals, cleaning_result

        except Exception as e:
            logger.error("Signal data cleaning failed", error=str(e))
            cleaning_result = CleaningResult(
                original_data=original_signals,
                cleaned_data=original_signals,
                applied_strategies=[],
                removed_count=0,
                adjusted_count=0,
                imputed_count=0,
                timestamp=datetime.now(timezone.utc),
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
            return original_signals, cleaning_result

    async def _handle_missing_data(self, data: MarketData) -> tuple[MarketData, int]:
        """Handle missing data using imputation strategies"""
        imputed_count = 0

        if data is None:
            return None, imputed_count

        # Price imputation
        if not data.price or data.price == 0:
            imputed_price = await self._impute_price(data.symbol)
            if imputed_price:
                data.price = imputed_price
                imputed_count += 1

        # Volume imputation
        if not data.volume or data.volume == 0:
            imputed_volume = await self._impute_volume(data.symbol)
            if imputed_volume:
                data.volume = imputed_volume
                imputed_count += 1

        # OHLC imputation
        if not data.open_price and data.price:
            data.open_price = data.price
            imputed_count += 1

        if not data.high_price and data.price:
            data.high_price = data.price
            imputed_count += 1

        if not data.low_price and data.price:
            data.low_price = data.price
            imputed_count += 1

        # Bid/Ask imputation
        if not data.bid and data.price:
            spread = data.price * Decimal("0.001")  # 0.1% spread
            data.bid = data.price - spread
            imputed_count += 1

        if not data.ask and data.price:
            spread = data.price * Decimal("0.001")  # 0.1% spread
            data.ask = data.price + spread
            imputed_count += 1

        return data, imputed_count

    async def _handle_outliers(self, data: MarketData) -> tuple[MarketData, int, int]:
        """Detect and handle outliers in price and volume data"""
        removed_count = 0
        adjusted_count = 0

        if not data.symbol or not data.price:
            return data, removed_count, adjusted_count

        # Initialize history if needed
        if data.symbol not in self.price_history:
            self.price_history[data.symbol] = []

        price_history = self.price_history[data.symbol]
        current_price = float(data.price)

        # Add current price to history
        price_history.append(current_price)

        # Maintain history size
        if len(price_history) > self.max_history_size:
            price_history.pop(0)

        # Outlier detection (need at least 10 data points)
        if len(price_history) >= 10:
            mean_price = statistics.mean(price_history[:-1])  # Exclude current price
            std_price = statistics.stdev(price_history[:-1]) if len(price_history) > 1 else 0

            if std_price > 0:
                z_score = abs(current_price - mean_price) / std_price

                if z_score > self.outlier_threshold:
                    # Strategy: adjust to mean + threshold * std
                    if self.config.get("outlier_strategy") == CleaningStrategy.ADJUST:
                        adjusted_price = mean_price + (
                            self.outlier_threshold * std_price * np.sign(current_price - mean_price)
                        )
                        data.price = Decimal(str(adjusted_price))
                        adjusted_count += 1
                        logger.warning(
                            "Price outlier adjusted",
                            symbol=data.symbol,
                            original_price=current_price,
                            adjusted_price=adjusted_price,
                            z_score=z_score,
                        )
                    else:
                        # Strategy: remove by setting to None
                        data.price = None
                        removed_count += 1
                        logger.warning(
                            "Price outlier removed",
                            symbol=data.symbol,
                            original_price=current_price,
                            z_score=z_score,
                        )

        # Volume outlier detection
        if data.volume and data.symbol not in self.volume_history:
            self.volume_history[data.symbol] = []

        if data.volume and data.symbol in self.volume_history:
            volume_history = self.volume_history[data.symbol]
            current_volume = float(data.volume)

            volume_history.append(current_volume)

            if len(volume_history) > self.max_history_size:
                volume_history.pop(0)

            if len(volume_history) >= 10:
                mean_volume = statistics.mean(volume_history[:-1])
                std_volume = statistics.stdev(volume_history[:-1]) if len(volume_history) > 1 else 0

                if std_volume > 0:
                    z_score = abs(current_volume - mean_volume) / std_volume

                    if z_score > self.outlier_threshold:
                        if self.config.get("outlier_strategy") == CleaningStrategy.ADJUST:
                            adjusted_volume = mean_volume + (
                                self.outlier_threshold
                                * std_volume
                                * np.sign(current_volume - mean_volume)
                            )
                            data.volume = Decimal(str(adjusted_volume))
                            adjusted_count += 1
                        else:
                            data.volume = None
                            removed_count += 1

        return data, removed_count, adjusted_count

    async def _smooth_data(self, data: MarketData) -> MarketData:
        """Apply smoothing to reduce noise in data"""
        if not data.symbol or not data.price:
            return data

        # Initialize history if needed
        if data.symbol not in self.price_history:
            self.price_history[data.symbol] = []

        price_history = self.price_history[data.symbol]
        current_price = float(data.price)

        # Add current price to history
        price_history.append(current_price)

        # Maintain history size
        if len(price_history) > self.max_history_size:
            price_history.pop(0)

        # Apply smoothing if enough data points
        if len(price_history) >= self.smoothing_window:
            # Simple moving average smoothing
            smoothed_price = statistics.mean(price_history[-self.smoothing_window :])
            data.price = Decimal(str(smoothed_price))

            # Smooth volume if available
            if data.volume and data.symbol in self.volume_history:
                volume_history = self.volume_history[data.symbol]
                if len(volume_history) >= self.smoothing_window:
                    smoothed_volume = statistics.mean(volume_history[-self.smoothing_window :])
                    data.volume = Decimal(str(smoothed_volume))

        return data

    async def _remove_duplicates(self, data: MarketData) -> tuple[MarketData, int]:
        """Remove duplicate data points"""
        removed_count = 0

        # Create data hash for duplicate detection
        data_hash = self._create_data_hash(data)

        # Check if this data point is a duplicate
        if hasattr(self, "_last_data_hash") and self._last_data_hash == data_hash:
            # This is a duplicate, mark for removal
            removed_count = 1
            data = None  # Mark for removal
        else:
            # Update last data hash
            self._last_data_hash = data_hash

        return data, removed_count

    async def _normalize_data(self, data: MarketData) -> MarketData:
        """Normalize data for consistent format"""
        if not data:
            return data

        # Ensure timestamp is timezone-aware
        if data.timestamp and data.timestamp.tzinfo is None:
            data.timestamp = data.timestamp.replace(tzinfo=timezone.utc)

        # Normalize symbol format
        if data.symbol:
            data.symbol = data.symbol.upper()

        # Ensure price precision
        if data.price:
            # Round to 8 decimal places for crypto
            data.price = Decimal(str(float(data.price))).quantize(Decimal("0.00000001"))

        # Ensure volume precision
        if data.volume:
            data.volume = Decimal(str(float(data.volume))).quantize(Decimal("0.00000001"))

        return data

    async def _impute_price(self, symbol: str) -> Decimal | None:
        """Impute missing price using historical data"""
        if symbol not in self.price_history or not self.price_history[symbol]:
            # Return default price if no history available
            return Decimal("50000.00")  # Default BTC price

        # Use median of recent prices
        recent_prices = self.price_history[symbol][-10:]  # Last 10 prices
        if recent_prices:
            median_price = statistics.median(recent_prices)
            return Decimal(str(median_price))

        return Decimal("50000.00")  # Default price

    async def _impute_volume(self, symbol: str) -> Decimal | None:
        """Impute missing volume using historical data"""
        if symbol not in self.volume_history or not self.volume_history[symbol]:
            # Return default volume if no history available
            return Decimal("100.0")  # Default volume

        # Use median of recent volumes
        recent_volumes = self.volume_history[symbol][-10:]  # Last 10 volumes
        if recent_volumes:
            median_volume = statistics.median(recent_volumes)
            return Decimal(str(median_volume))

        return Decimal("100.0")  # Default volume

    async def _is_valid_signal(self, signal: Signal) -> bool:
        """Check if signal is valid for cleaning"""
        if not signal.direction or not signal.symbol:
            return False

        if not (0.0 <= signal.confidence <= 1.0):
            return False

        if signal.timestamp > datetime.now(timezone.utc) + timedelta(seconds=60):
            return False

        return True

    async def _clean_confidence(self, confidence: float) -> float:
        """Clean signal confidence value"""
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))

        # Round to 3 decimal places
        return round(confidence, 3)

    async def _is_duplicate_signal(self, signal: Signal, existing_signals: list[Signal]) -> bool:
        """Check if signal is a duplicate"""
        for existing_signal in existing_signals:
            if (
                signal.symbol == existing_signal.symbol
                and signal.direction == existing_signal.direction
                and abs((signal.timestamp - existing_signal.timestamp).total_seconds())
                < self.duplicate_threshold
            ):
                return True
        return False

    def _create_data_hash(self, data: MarketData) -> str:
        """Create hash for duplicate detection"""
        if not data:
            return ""

        # Create hash from key fields
        hash_data = {
            "symbol": data.symbol,
            "price": float(data.price) if data.price else 0,
            "volume": float(data.volume) if data.volume else 0,
            "timestamp": data.timestamp.isoformat() if data.timestamp else "",
        }

        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()

    @time_execution
    async def get_cleaning_summary(self) -> dict[str, Any]:
        """Get cleaning statistics and summary"""
        return {
            "cleaning_stats": self.cleaning_stats.copy(),
            "price_history_size": {
                symbol: len(history) for symbol, history in self.price_history.items()
            },
            "volume_history_size": {
                symbol: len(history) for symbol, history in self.volume_history.items()
            },
            "cleaning_config": {
                "outlier_threshold": self.outlier_threshold,
                "missing_threshold": self.missing_threshold,
                "smoothing_window": self.smoothing_window,
                "duplicate_threshold": self.duplicate_threshold,
            },
        }
