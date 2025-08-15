"""
Historical Data Replay Manager for Backtesting.

This module manages the replay of historical market data with support for
multiple timeframes, data sources, and replay modes.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field

from src.core.exceptions import DataError
from src.core.logging import get_logger
from src.database.manager import DatabaseManager
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class ReplayMode(Enum):
    """Data replay modes."""

    SEQUENTIAL = "sequential"  # Play data in order
    RANDOM_WALK = "random_walk"  # Random sampling with temporal coherence
    BOOTSTRAP = "bootstrap"  # Bootstrap resampling
    SHUFFLE = "shuffle"  # Shuffle historical periods


class DataReplayManager:
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
        db_manager: Optional[DatabaseManager] = None,
        cache_size: int = 10000,
    ):
        """
        Initialize data replay manager.

        Args:
            db_manager: Optional database manager for data loading
            cache_size: Maximum number of records to cache in memory
        """
        self.db_manager = db_manager
        self.cache_size = cache_size
        
        # Data storage
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._current_index: Dict[str, int] = {}
        self._subscribers: List[Callable] = []
        
        # Replay state
        self._replay_mode = ReplayMode.SEQUENTIAL
        self._current_timestamp: Optional[datetime] = None
        self._speed_multiplier = 1.0
        
        logger.info("DataReplayManager initialized", cache_size=cache_size)

    @time_execution
    async def load_data(
        self,
        symbols: List[str],
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
        logger.info(
            "Loading historical data",
            symbols=symbols,
            start=start_date,
            end=end_date,
            timeframe=timeframe,
        )

        for symbol in symbols:
            try:
                if source == "database" and self.db_manager:
                    data = await self._load_from_database(
                        symbol, start_date, end_date, timeframe
                    )
                elif source == "csv":
                    data = await self._load_from_csv(symbol, start_date, end_date)
                else:
                    data = await self._generate_synthetic_data(
                        symbol, start_date, end_date, timeframe
                    )

                # Validate and clean data
                data = self._validate_data(data)
                
                # Cache data
                self._data_cache[symbol] = data
                self._current_index[symbol] = 0
                
                logger.info(f"Loaded data for {symbol}", records=len(data))

            except Exception as e:
                logger.error(f"Failed to load data for {symbol}", error=str(e))
                raise DataError(f"Failed to load data for {symbol}: {str(e)}")

    async def _load_from_database(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Load data from database."""
        if not self.db_manager:
            raise DataError("Database manager not configured")

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = $1 
            AND timestamp >= $2 
            AND timestamp <= $3
            AND timeframe = $4
            ORDER BY timestamp
        """

        rows = await self.db_manager.fetch_all(
            query, symbol, start_date, end_date, timeframe
        )

        if not rows:
            raise DataError(f"No data found for {symbol}")

        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        return df

    async def _load_from_csv(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        import os

        file_path = f"data/{symbol}.csv"
        
        if not os.path.exists(file_path):
            raise DataError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        return df

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
            logger.warning(
                f"Removed {before_rows - after_rows} rows with NaN values"
            )

        # Validate OHLC relationships
        invalid_ohlc = (
            (data["high"] < data["low"]) |
            (data["high"] < data["open"]) |
            (data["high"] < data["close"]) |
            (data["low"] > data["open"]) |
            (data["low"] > data["close"])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
            data = data[~invalid_ohlc]

        # Validate positive values
        negative_values = (data[required_columns] < 0).any(axis=1)
        
        if negative_values.any():
            logger.warning(f"Found {negative_values.sum()} rows with negative values")
            data = data[~negative_values]

        return data

    async def start_replay(
        self,
        mode: ReplayMode = ReplayMode.SEQUENTIAL,
        speed: float = 1.0,
        callback: Optional[Callable] = None,
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

        logger.info("Starting data replay", mode=mode.value, speed=speed)

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
        period_data = {}
        
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

    async def _notify_subscribers(
        self, timestamp: datetime, data: Dict[str, pd.Series]
    ) -> None:
        """Notify all subscribers of new data."""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(timestamp, data)
                else:
                    callback(timestamp, data)
            except Exception as e:
                logger.error(f"Subscriber callback failed", error=str(e))

    def get_current_data(self, symbol: str) -> Optional[pd.Series]:
        """Get current data point for a symbol."""
        if symbol not in self._data_cache:
            return None

        idx = self._current_index.get(symbol, 0)
        
        if idx < len(self._data_cache[symbol]):
            return self._data_cache[symbol].iloc[idx]
        
        return None

    def get_historical_data(
        self, symbol: str, lookback: int
    ) -> Optional[pd.DataFrame]:
        """Get historical data with lookback."""
        if symbol not in self._data_cache:
            return None

        current_idx = self._current_index.get(symbol, 0)
        start_idx = max(0, current_idx - lookback)
        
        return self._data_cache[symbol].iloc[start_idx:current_idx + 1]

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
        logger.info("Data replay reset")

    def get_statistics(self) -> Dict[str, Any]:
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