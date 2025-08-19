"""Base implementation for data sources with common functionality."""

from typing import Any, Dict, List, Optional, AsyncIterator
import asyncio
from datetime import datetime, timedelta

from src.data.interfaces import DataSourceInterface, DataCacheInterface
from src.core.logging import get_logger


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 10):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry."""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_call
            
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            
            self.last_call = asyncio.get_event_loop().time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class SimpleCache(DataCacheInterface):
    """Simple in-memory cache implementation."""
    
    def __init__(self):
        """Initialize cache."""
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache."""
        if key not in self._cache:
            return None
        
        # Check if expired
        if key in self._timestamps:
            timestamp = self._timestamps[key]
            # Simple expiry check (could be improved)
            if datetime.utcnow() > timestamp:
                del self._cache[key]
                del self._timestamps[key]
                return None
        
        return self._cache[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set in cache."""
        self._cache[key] = value
        if ttl:
            self._timestamps[key] = datetime.utcnow() + timedelta(seconds=ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._timestamps:
                del self._timestamps[key]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._timestamps.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cache


class BaseDataSource(DataSourceInterface):
    """Base implementation with common functionality for data sources."""
    
    def __init__(
        self,
        cache: Optional[DataCacheInterface] = None,
        rate_limit: float = 10
    ):
        """
        Initialize base data source.
        
        Args:
            cache: Optional cache implementation
            rate_limit: API calls per second
        """
        self.cache = cache or SimpleCache()
        self.rate_limiter = RateLimiter(rate_limit)
        self._connected = False
        self._logger = get_logger(self.__class__.__module__)
    
    async def fetch_with_cache(
        self,
        cache_key: str,
        fetch_func: callable,
        ttl: int = 60
    ) -> Any:
        """
        Fetch data with caching.
        
        Args:
            cache_key: Key for caching
            fetch_func: Function to fetch data if not cached
            ttl: Cache time to live
            
        Returns:
            Fetched or cached data
        """
        # Check cache first
        cached = await self.cache.get(cache_key)
        if cached is not None:
            self._logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        # Fetch with rate limiting
        async with self.rate_limiter:
            self._logger.debug(f"Fetching data for {cache_key}")
            data = await fetch_func()
            
            # Cache the result
            await self.cache.set(cache_key, data, ttl)
            
            return data
    
    async def connect(self) -> None:
        """Establish connection."""
        self._connected = True
        self._logger.info(f"{self.__class__.__name__} connected")
    
    async def disconnect(self) -> None:
        """Close connection."""
        self._connected = False
        self._logger.info(f"{self.__class__.__name__} disconnected")
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected
    
    # Abstract methods to be implemented by subclasses
    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fetch historical data - to be implemented by subclasses."""
        raise NotImplementedError
    
    async def stream(
        self,
        symbol: str,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream real-time data - to be implemented by subclasses."""
        raise NotImplementedError