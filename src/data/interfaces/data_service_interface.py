"""
Data Service Interface

Abstract interface for data services, providing contracts for data operations
without tight coupling to specific infrastructure implementations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.core.types import MarketData
from src.data.types import CacheLevel, DataMetrics, DataRequest
from src.database.models import MarketDataRecord


class DataServiceInterface(ABC):
    """
    Abstract interface for data services.
    
    This interface defines the contract for data operations without
    coupling to specific infrastructure implementations (Redis, database, etc.).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the data service."""
        pass

    @abstractmethod
    async def store_market_data(
        self,
        data: MarketData | list[MarketData],
        exchange: str,
        validate: bool = True,
        cache_levels: list[CacheLevel] | None = None,
    ) -> bool:
        """
        Store market data with validation and caching.

        Args:
            data: Single MarketData or list of MarketData objects
            exchange: Exchange name
            validate: Whether to perform data validation
            cache_levels: Cache levels to update

        Returns:
            bool: Success status
        """
        pass

    @abstractmethod
    async def get_market_data(self, request: DataRequest) -> list[MarketDataRecord]:
        """
        Retrieve market data with intelligent caching.

        Args:
            request: Data request with filters and options

        Returns:
            List[MarketDataRecord]: Retrieved market data
        """
        pass

    @abstractmethod
    async def get_data_count(self, symbol: str, exchange: str = "binance") -> int:
        """
        Get count of available data points for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            int: Number of data points
        """
        pass

    @abstractmethod
    async def get_recent_data(
        self, symbol: str, limit: int = 100, exchange: str = "binance"
    ) -> list[MarketData]:
        """
        Get recent market data for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of records to return
            exchange: Exchange name

        Returns:
            List[MarketData]: Recent market data
        """
        pass

    @abstractmethod
    async def get_metrics(self) -> DataMetrics:
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
    """
    Abstract interface for data storage operations.
    
    Decouples storage logic from service layer.
    """

    @abstractmethod
    async def store_records(self, records: list[MarketDataRecord]) -> bool:
        """
        Store market data records.

        Args:
            records: List of records to store

        Returns:
            bool: Success status
        """
        pass

    @abstractmethod
    async def retrieve_records(self, request: DataRequest) -> list[MarketDataRecord]:
        """
        Retrieve market data records.

        Args:
            request: Data request with filters

        Returns:
            List[MarketDataRecord]: Retrieved records
        """
        pass

    @abstractmethod
    async def get_record_count(self, symbol: str, exchange: str) -> int:
        """
        Get count of records for symbol and exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            int: Number of records
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform storage health check."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup storage resources."""
        pass


class DataCacheInterface(ABC):
    """
    Abstract interface for data caching operations.
    
    Decouples caching logic from service layer.
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """
        Get data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set data in cache.

        Args:
            key: Cache key
            value: Data to cache
            ttl: Time to live in seconds
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete data from cache.

        Args:
            key: Cache key

        Returns:
            bool: True if deleted
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached data."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            bool: True if exists
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform cache health check."""
        pass


class DataValidatorInterface(ABC):
    """
    Abstract interface for data validation operations.
    
    Decouples validation logic from service layer.
    """

    @abstractmethod
    async def validate_market_data(self, data_list: list[MarketData]) -> list[MarketData]:
        """
        Validate market data with comprehensive checks.

        Args:
            data_list: List of market data to validate

        Returns:
            List[MarketData]: Valid market data
        """
        pass

    @abstractmethod
    def get_validation_errors(self) -> list[str]:
        """
        Get validation errors from last validation.

        Returns:
            List[str]: Error messages
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform validator health check."""
        pass