"""
Data interfaces consolidated to avoid circular dependencies.

All data-related interfaces are defined here instead of being split across
multiple files to prevent circular import issues.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.types import MarketData
    from src.data.types import CacheLevel, DataMetrics, DataRequest
    from src.database.models import MarketDataRecord


class DataSourceInterface(ABC):
    """Common interface for all data sources."""

    @abstractmethod
    async def fetch(
        self, symbol: str, timeframe: str, limit: int = 100, **kwargs
    ) -> list[dict[str, Any]]:
        """Fetch historical data."""
        pass

    @abstractmethod
    async def stream(self, symbol: str, **kwargs) -> AsyncIterator[dict[str, Any]]:
        """Stream real-time data."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to data source."""
        pass


class DataValidatorInterface(ABC):
    """Common interface for data validators."""

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data."""
        pass

    @abstractmethod
    def get_errors(self) -> list[str]:
        """Get validation errors from last validation."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset validator state."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform validator health check."""
        pass


class DataCacheInterface(ABC):
    """Interface for data caching."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get data from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set data in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached data."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform cache health check."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the cache."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        pass


class DataServiceInterface(ABC):
    """Abstract interface for data services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the data service."""
        pass

    @abstractmethod
    async def store_market_data(
        self,
        data: "MarketData | list[MarketData]",
        exchange: str,
        validate: bool = True,
        cache_levels: "list[CacheLevel] | None" = None,
    ) -> bool:
        """Store market data with validation and caching."""
        pass

    @abstractmethod
    async def get_market_data(self, request: "DataRequest") -> "list[MarketDataRecord]":
        """Retrieve market data with intelligent caching."""
        pass

    @abstractmethod
    async def get_data_count(self, symbol: str, exchange: str = "binance") -> int:
        """Get count of available data points for a symbol."""
        pass

    @abstractmethod
    async def get_recent_data(
        self, symbol: str, limit: int = 100, exchange: str = "binance"
    ) -> "list[MarketData]":
        """Get recent market data for a symbol."""
        pass

    @abstractmethod
    async def get_metrics(self) -> "DataMetrics":
        """Get current data service metrics."""
        pass

    @abstractmethod
    async def reset_metrics(self) -> None:
        """Reset metrics counters."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass


class DataStorageInterface(ABC):
    """Abstract interface for data storage operations."""

    @abstractmethod
    async def store_records(self, records: "list[MarketDataRecord]") -> bool:
        """Store market data records."""
        pass

    @abstractmethod
    async def retrieve_records(self, request: "DataRequest") -> "list[MarketDataRecord]":
        """Retrieve market data records."""
        pass

    @abstractmethod
    async def get_record_count(self, symbol: str, exchange: str) -> int:
        """Get count of records for symbol and exchange."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform storage health check."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup storage resources."""
        pass


class ServiceDataValidatorInterface(ABC):
    """Extended data validator interface for service layer operations."""

    @abstractmethod
    async def validate_market_data(self, data_list: "list[MarketData]") -> "list[MarketData]":
        """Validate market data with comprehensive checks."""
        pass

    @abstractmethod
    def get_validation_errors(self) -> list[str]:
        """Get validation errors from last validation."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform validator health check."""
        pass


__all__ = [
    "DataCacheInterface",
    "DataServiceInterface",
    "DataSourceInterface",
    "DataStorageInterface",
    "DataValidatorInterface",
    "ServiceDataValidatorInterface",
]
