"""
Historical Data Replay Manager for Backtesting.

This module manages the replay of historical market data with support for
multiple timeframes, data sources, and replay modes.
"""

import asyncio
import os
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import DataError
from src.utils.backtesting_decorators import data_loading_operation
from src.utils.config_conversion import convert_config_to_dict
from src.utils.synthetic_data_generator import generate_synthetic_ohlcv_data, validate_ohlcv_data

# Configuration constants
DEFAULT_CACHE_SIZE = 10000
DEFAULT_MAX_CACHE_SIZE = 1000
DEFAULT_BACKPRESSURE_LIMIT = 100
DEFAULT_CONCURRENT_NOTIFICATION_LIMIT = 50
DEFAULT_SPEED_MULTIPLIER = 1.0
DEFAULT_CALLBACK_TIMEOUT = 5.0
DEFAULT_CLEANUP_TIMEOUT = 5.0
DEFAULT_SLEEP_INTERVAL = 0.001
DEFAULT_CLEANUP_WAIT_INTERVAL = 0.1


class ReplayMode(Enum):
    """Data replay modes."""

    SEQUENTIAL = "sequential"  # Play data in order
    RANDOM_WALK = "random_walk"  # Random sampling with temporal coherence
    BOOTSTRAP = "bootstrap"  # Bootstrap resampling
    SHUFFLE = "shuffle"  # Shuffle historical periods


class DataReplayManager(BaseComponent):
    """
    Manages historical data replay for backtesting.

    Features:
    - Multi-timeframe support
    - Data synchronization across symbols
    - Event-driven replay
    - Data quality validation
    """

    def __init__(
        self,
        config: Config | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """
        Initialize data replay manager.

        Args:
            config: Configuration object
            cache_size: Maximum number of records to cache in memory
        """
        # Convert config to dict using shared utility
        config_dict = convert_config_to_dict(config)
        super().__init__(name="DataReplayManager", config=config_dict)  # type: ignore
        self.config = config
        self.cache_size = cache_size

        # Data storage
        self._data_cache: dict[str, pd.DataFrame] = {}
        # Use self._config from BaseComponent, with proper fallback
        if self._config and isinstance(self._config, dict):
            max_cache_size_value = self._config.get("max_cache_size", DEFAULT_MAX_CACHE_SIZE)
            self._max_cache_size = (
                int(max_cache_size_value)
                if isinstance(max_cache_size_value, int | str)
                else DEFAULT_MAX_CACHE_SIZE
            )
        else:
            self._max_cache_size = DEFAULT_MAX_CACHE_SIZE  # Max dataframes in cache
        self._current_index: dict[str, int] = {}
        self._subscribers: list[Callable] = []

        # Replay state
        self._replay_mode = ReplayMode.SEQUENTIAL
        self._current_timestamp: datetime | None = None
        self._speed_multiplier = DEFAULT_SPEED_MULTIPLIER

        # Backpressure and connection management
        self._max_pending_notifications = DEFAULT_BACKPRESSURE_LIMIT  # Backpressure limit
        self._pending_notifications = 0
        # Concurrent notification limit
        self._notification_semaphore = asyncio.Semaphore(DEFAULT_CONCURRENT_NOTIFICATION_LIMIT)
        self._connection_active = False

        self.logger.info("DataReplayManager initialized", cache_size=cache_size)

    async def __aenter__(self) -> "DataReplayManager":
        """Async context manager entry."""
        self._connection_active = True
        self.logger.info("DataReplayManager connection opened")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with proper cleanup."""
        self._connection_active = False
        await self.cleanup()
        if exc_type:
            self.logger.error(
                "DataReplayManager exiting with exception",
                exc_type=exc_type.__name__,
                exc_val=str(exc_val),
            )
        else:
            self.logger.info("DataReplayManager connection closed")

    @data_loading_operation(operation="replay_load_data")
    async def load_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h",
        source: str = "database",
    ) -> None:
        """
        Load historical data for replay.

        Args:
            symbols: List of symbols to load
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe
            source: Data source (database, csv, api)
        """
        self.logger.info(
            "Loading historical data",
            symbols=symbols,
            start=start_date,
            end=end_date,
            timeframe=timeframe,
        )

        for symbol in symbols:
            try:
                if source == "csv":
                    data = await self._load_from_csv(symbol, start_date, end_date)
                else:
                    # Generate synthetic data as default
                    data = await self._generate_synthetic_data(
                        symbol, start_date, end_date, timeframe
                    )

                # Validate and clean data
                data = self._validate_data(data)

                # Cache data
                # Implement cache eviction if needed
                if len(self._data_cache) >= self._max_cache_size:
                    # Remove oldest entry (simple FIFO eviction)
                    oldest_symbol = next(iter(self._data_cache))
                    del self._data_cache[oldest_symbol]
                    if oldest_symbol in self._current_index:
                        del self._current_index[oldest_symbol]

                self._data_cache[symbol] = data
                self._current_index[symbol] = 0

                self.logger.info(f"Loaded data for {symbol}", records=len(data))

            except DataError:
                # Re-raise DataError as-is
                raise
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}", error=str(e))
                raise DataError(f"Failed to load data for {symbol}: {e!s}") from e

    async def _load_from_csv(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        file_path = f"data/{symbol}.csv"

        if not os.path.exists(file_path):
            raise DataError(f"CSV file not found: {file_path}")

        try:
            with open(file_path) as file_handle:
                df = pd.read_csv(file_handle, parse_dates=["timestamp"], index_col="timestamp")

                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]

                return df
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise DataError(f"Failed to load CSV file {file_path}: {e}") from e
        except Exception as e:
            raise DataError(f"Failed to load CSV file {file_path}: {e}") from e

    async def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Generate synthetic data for testing."""
        # Use shared synthetic data generator
        return generate_synthetic_ohlcv_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
        )

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data."""
        # Use shared validation utility
        is_valid, errors = validate_ohlcv_data(data)

        if not is_valid:
            for error in errors:
                self.logger.warning(f"Data validation error: {error}")

        # Clean data by removing invalid rows
        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing = set(required_columns) - set(data.columns)

        if missing:
            raise DataError(f"Missing required columns: {missing}")

        # Remove NaN values
        before_rows = len(data)
        data = data.dropna()
        after_rows = len(data)

        if after_rows < before_rows:
            self.logger.warning(f"Removed {before_rows - after_rows} rows with NaN values")

        # Validate OHLC relationships
        invalid_ohlc = (
            (data["high"] < data["low"])
            | (data["high"] < data["open"])
            | (data["high"] < data["close"])
            | (data["low"] > data["open"])
            | (data["low"] > data["close"])
        )

        if invalid_ohlc.any():
            self.logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
            data = data[~invalid_ohlc]

        # Validate positive values
        negative_values = (data[required_columns] < 0).any(axis=1)

        if negative_values.any():
            self.logger.warning(f"Found {negative_values.sum()} rows with negative values")
            data = data[~negative_values]

        return data

    async def start_replay(
        self,
        mode: ReplayMode = ReplayMode.SEQUENTIAL,
        speed: float = 1.0,
        callback: Callable | None = None,
    ) -> None:
        """
        Start data replay.

        Args:
            mode: Replay mode
            speed: Speed multiplier for replay
            callback: Optional callback for each data point
        """
        self._replay_mode = mode
        self._speed_multiplier = speed

        if callback:
            self._subscribers.append(callback)

        self.logger.info("Starting data replay", mode=mode.value, speed=speed)

        if mode == ReplayMode.SEQUENTIAL:
            await self._replay_sequential()
        elif mode == ReplayMode.RANDOM_WALK:
            await self._replay_random_walk()
        elif mode == ReplayMode.BOOTSTRAP:
            await self._replay_bootstrap()
        elif mode == ReplayMode.SHUFFLE:
            await self._replay_shuffle()

    async def _replay_sequential(self) -> None:
        """Replay data in sequential order."""
        # Get all unique timestamps across symbols
        all_timestamps = set()

        for data in self._data_cache.values():
            all_timestamps.update(data.index.tolist())

        sorted_timestamps = sorted(all_timestamps)

        for timestamp in sorted_timestamps:
            self._current_timestamp = timestamp

            # Get data for this timestamp
            current_data = {}

            for symbol, data in self._data_cache.items():
                if timestamp in data.index:
                    current_data[symbol] = data.loc[timestamp]
                    self._current_index[symbol] = data.index.get_loc(timestamp)

            # Notify subscribers
            await self._notify_subscribers(timestamp, current_data)

            # Control replay speed
            if self._speed_multiplier > 0:
                await asyncio.sleep(DEFAULT_SLEEP_INTERVAL / self._speed_multiplier)

    async def _replay_random_walk(self) -> None:
        """Replay with random walk through historical periods."""
        import random

        # Get all timestamps
        all_timestamps = []
        for data in self._data_cache.values():
            all_timestamps.extend(data.index.tolist())

        all_timestamps = sorted(set(all_timestamps))

        if not all_timestamps:
            return

        # Start from random position
        current_idx = random.randint(0, len(all_timestamps) - 1)

        while True:
            timestamp = all_timestamps[current_idx]
            self._current_timestamp = timestamp

            # Get data
            current_data = {}
            for symbol, data in self._data_cache.items():
                if timestamp in data.index:
                    current_data[symbol] = data.loc[timestamp]

            # Notify subscribers
            await self._notify_subscribers(timestamp, current_data)

            # Random walk to next position
            step = random.choice([-2, -1, 1, 2])  # Can go backward
            current_idx = max(0, min(len(all_timestamps) - 1, current_idx + step))

            # Control speed
            if self._speed_multiplier > 0:
                await asyncio.sleep(DEFAULT_SLEEP_INTERVAL / self._speed_multiplier)

    async def _replay_bootstrap(self) -> None:
        """Replay with bootstrap resampling of historical periods."""
        import random

        # Collect all data points
        all_data_points = []

        for symbol, data in self._data_cache.items():
            for timestamp, row in data.iterrows():
                all_data_points.append((timestamp, symbol, row))

        if not all_data_points:
            return

        # Bootstrap resample
        n_samples = len(all_data_points)

        for _ in range(n_samples):
            # Random sample with replacement
            timestamp, symbol, row = random.choice(all_data_points)
            self._current_timestamp = timestamp

            # Notify subscribers
            await self._notify_subscribers(timestamp, {symbol: row})

            # Control speed
            if self._speed_multiplier > 0:
                await asyncio.sleep(DEFAULT_SLEEP_INTERVAL / self._speed_multiplier)

    async def _replay_shuffle(self) -> None:
        """Replay with shuffled historical periods."""
        import random

        # Group data by periods (e.g., daily)
        period_data: dict[Any, dict[str, Any]] = {}

        for symbol, data in self._data_cache.items():
            # Group by date
            for date, group in data.groupby(data.index.date):
                if date not in period_data:
                    period_data[date] = {}
                period_data[date][symbol] = group

        # Shuffle periods
        shuffled_dates = list(period_data.keys())
        random.shuffle(shuffled_dates)

        # Replay shuffled periods
        for date in shuffled_dates:
            period = period_data[date]

            # Get all timestamps in this period
            timestamps = set()
            for symbol_data in period.values():
                timestamps.update(symbol_data.index.tolist())

            # Play period sequentially
            for timestamp in sorted(timestamps):
                self._current_timestamp = timestamp

                current_data = {}
                for symbol, data in period.items():
                    if timestamp in data.index:
                        current_data[symbol] = data.loc[timestamp]

                await self._notify_subscribers(timestamp, current_data)

                if self._speed_multiplier > 0:
                    await asyncio.sleep(DEFAULT_SLEEP_INTERVAL / self._speed_multiplier)

    async def _notify_subscribers(self, timestamp: datetime, data: dict[str, pd.Series]) -> None:
        """
        Notify all subscribers of new data with concurrent execution and backpressure handling.
        """
        if not self._subscribers:
            return

        # Apply backpressure - skip notifications if too many are pending
        if self._pending_notifications >= self._max_pending_notifications:
            self.logger.warning(
                "Notification backpressure applied - dropping message",
                pending=self._pending_notifications,
            )
            return

        self._pending_notifications += 1

        try:
            # Use semaphore to limit concurrent notifications
            async with self._notification_semaphore:
                # Separate async and sync callbacks for proper handling
                async_callbacks = []
                sync_callbacks = []

                for callback in self._subscribers:
                    if asyncio.iscoroutinefunction(callback):
                        async_callbacks.append(callback)
                    else:
                        sync_callbacks.append(callback)

                # Execute sync callbacks first (they're typically fast)
                for callback in sync_callbacks:
                    try:
                        callback(timestamp, data)
                    except Exception as e:
                        self.logger.error("Sync subscriber callback failed", error=str(e))
                        # Don't re-raise to avoid breaking other subscribers

                # Execute async callbacks concurrently with timeout protection
                if async_callbacks:
                    tasks = []
                    for callback in async_callbacks:
                        task = asyncio.create_task(
                            self._execute_callback_with_timeout(callback, timestamp, data)
                        )
                        tasks.append(task)

                    # Wait for all async callbacks to complete, but don't fail if some timeout
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._pending_notifications = max(0, self._pending_notifications - 1)

    async def _execute_callback_with_timeout(
        self, callback: Callable, timestamp: datetime, data: dict[str, pd.Series]
    ) -> None:
        """Execute async callback with timeout and error handling."""
        try:
            # Set reasonable timeout for callback execution
            await asyncio.wait_for(callback(timestamp, data), timeout=DEFAULT_CALLBACK_TIMEOUT)
        except asyncio.TimeoutError:
            self.logger.warning("Subscriber callback timed out", callback=str(callback))
        except Exception as e:
            self.logger.error(
                "Async subscriber callback failed", callback=str(callback), error=str(e)
            )

    def get_current_data(self, symbol: str) -> pd.Series | None:
        """Get current data point for a symbol."""
        if symbol not in self._data_cache:
            return None

        idx = self._current_index.get(symbol, 0)

        if idx < len(self._data_cache[symbol]):
            return self._data_cache[symbol].iloc[idx]

        return None

    def get_historical_data(self, symbol: str, lookback: int) -> pd.DataFrame | None:
        """Get historical data with lookback."""
        if symbol not in self._data_cache:
            return None

        current_idx = self._current_index.get(symbol, 0)
        start_idx = max(0, current_idx - lookback)

        return self._data_cache[symbol].iloc[start_idx : current_idx + 1]

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to data updates."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from data updates."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def reset(self) -> None:
        """Reset replay state."""
        for symbol in self._current_index:
            self._current_index[symbol] = 0

        self._current_timestamp = None
        self.logger.info("Data replay reset")

    def get_statistics(self) -> dict[str, Any]:
        """Get data statistics."""
        stats = {}

        for symbol, data in self._data_cache.items():
            stats[symbol] = {
                "total_records": len(data),
                "start_date": data.index[0] if len(data) > 0 else None,
                "end_date": data.index[-1] if len(data) > 0 else None,
                "current_position": self._current_index.get(symbol, 0),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            }

        return stats

    async def cleanup(self) -> None:
        """Cleanup data replay manager resources with proper connection handling."""
        try:
            # Mark connection as inactive
            self._connection_active = False

            # Wait for pending notifications to complete (with timeout)
            max_wait = DEFAULT_CLEANUP_TIMEOUT  # Maximum wait time in seconds
            wait_interval = DEFAULT_CLEANUP_WAIT_INTERVAL
            waited = 0.0

            while self._pending_notifications > 0 and waited < max_wait:
                await asyncio.sleep(wait_interval)
                waited += wait_interval

            if self._pending_notifications > 0:
                self.logger.warning(
                    "Cleanup proceeding with pending notifications",
                    pending=self._pending_notifications,
                )

            # Clear all cached data
            self._data_cache.clear()
            self._current_index.clear()
            self._subscribers.clear()

            # Reset state
            self._current_timestamp = None
            self._pending_notifications = 0

            self.logger.info("DataReplayManager cleanup completed")
        except Exception as e:
            self.logger.error(f"DataReplayManager cleanup error: {e}")
            # Don't re-raise cleanup errors to avoid masking original issues
