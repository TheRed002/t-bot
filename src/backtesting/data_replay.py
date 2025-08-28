"""
Historical Data Replay Manager for Backtesting.

This module manages the replay of historical market data with support for
multiple timeframes, data sources, and replay modes.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import DataError
from src.error_handling.decorators import with_circuit_breaker, with_error_context
from src.utils.decorators import time_execution


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
        cache_size: int = 10000,
    ) -> None:
        """
        Initialize data replay manager.

        Args:
            config: Configuration object
            cache_size: Maximum number of records to cache in memory
        """
        # Convert Config to dict if needed
        config_dict = None
        if config:
            if hasattr(config, "model_dump"):
                config_dict = config.model_dump()
            elif hasattr(config, "dict"):
                config_dict = config.dict()
            elif isinstance(config, dict):
                config_dict = config
            else:
                config_dict = {}

        super().__init__(name="DataReplayManager", config=config_dict)
        self.config = config
        self.cache_size = cache_size

        # Data storage
        self._data_cache: dict[str, pd.DataFrame] = {}
        # Use self._config from BaseComponent, with proper fallback
        if self._config and isinstance(self._config, dict):
            max_cache_size_value = self._config.get("max_cache_size", 1000)
            self._max_cache_size = (
                int(max_cache_size_value) if isinstance(max_cache_size_value, int | str) else 1000
            )
        else:
            self._max_cache_size = 1000  # Max dataframes in cache
        self._current_index: dict[str, int] = {}
        self._subscribers: list[Callable] = []

        # Replay state
        self._replay_mode = ReplayMode.SEQUENTIAL
        self._current_timestamp: datetime | None = None
        self._speed_multiplier = 1.0

        self.logger.info("DataReplayManager initialized", cache_size=cache_size)

    @time_execution
    @with_error_context(component="data_loading", operation="replay_load_data")
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
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
        import os

        file_path = f"data/{symbol}.csv"

        if not os.path.exists(file_path):
            raise DataError(f"CSV file not found: {file_path}")

        df = None
        try:
            df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            return df
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise DataError(f"Failed to load CSV file {file_path}: {e}") from e
        except Exception as e:
            raise DataError(f"Failed to load CSV file {file_path}: {e}") from e
        finally:
            # pandas handles file closing automatically, but ensure proper cleanup
            if df is not None:
                pass  # DataFrame cleanup is handled by pandas/garbage collector

    async def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Generate synthetic data for testing."""
        import numpy as np

        # Parse timeframe
        freq_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
        }

        freq = freq_map.get(timeframe, "1h")
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Generate price data using geometric Brownian motion
        np.random.seed(hash(symbol) % 2**32)  # Consistent per symbol

        n = len(dates)
        dt = 1 / 252  # Daily time step
        mu = 0.1  # Annual drift
        sigma = 0.2  # Annual volatility

        # Generate returns
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLC from prices
        df = pd.DataFrame(index=dates)
        df["close"] = prices
        df["open"] = prices * (1 + np.random.normal(0, 0.001, n))
        df["high"] = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
        df["low"] = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
        df["volume"] = np.random.uniform(1000, 10000, n)

        return df

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data."""
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
                await asyncio.sleep(0.001 / self._speed_multiplier)

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
                await asyncio.sleep(0.001 / self._speed_multiplier)

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
                await asyncio.sleep(0.001 / self._speed_multiplier)

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
                    await asyncio.sleep(0.001 / self._speed_multiplier)

    async def _notify_subscribers(self, timestamp: datetime, data: dict[str, pd.Series]) -> None:
        """Notify all subscribers of new data."""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(timestamp, data)
                else:
                    callback(timestamp, data)
            except Exception as e:
                self.logger.error("Subscriber callback failed", error=str(e))
                # Don't re-raise to avoid breaking other subscribers

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

    def cleanup(self) -> None:
        """Cleanup data replay manager resources."""
        try:
            # Clear all cached data
            self._data_cache.clear()
            self._current_index.clear()
            self._subscribers.clear()

            # Reset state
            self._current_timestamp = None

            self.logger.info("DataReplayManager cleanup completed")
        except Exception as e:
            self.logger.error(f"DataReplayManager cleanup error: {e}")
            # Don't re-raise cleanup errors to avoid masking original issues
