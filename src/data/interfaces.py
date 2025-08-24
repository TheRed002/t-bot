"""Common interfaces for all data sources, validators, and pipelines."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class DataSourceInterface(ABC):
    """Common interface for all data sources."""

    @abstractmethod
    async def fetch(
        self, symbol: str, timeframe: str, limit: int = 100, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Fetch historical data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            limit: Number of records to fetch
            **kwargs: Additional source-specific parameters

        Returns:
            List of data records
        """
        pass

    @abstractmethod
    async def stream(self, symbol: str, **kwargs) -> AsyncIterator[dict[str, Any]]:
        """
        Stream real-time data.

        Args:
            symbol: Trading symbol
            **kwargs: Additional source-specific parameters

        Yields:
            Data records as they arrive
        """
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
        """
        Validate data.

        Args:
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_errors(self) -> list[str]:
        """
        Get validation errors from last validation.

        Returns:
            List of error messages
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset validator state."""
        pass


class DataPipelineInterface(ABC):
    """Common interface for data processing pipelines."""

    @abstractmethod
    async def process(self, data: Any) -> Any:
        """
        Process data through the pipeline.

        Args:
            data: Input data

        Returns:
            Processed data
        """
        pass

    @abstractmethod
    def add_stage(self, stage: "PipelineStage") -> "DataPipelineInterface":
        """
        Add a processing stage to the pipeline.

        Args:
            stage: Pipeline stage to add

        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def remove_stage(self, stage_name: str) -> bool:
        """
        Remove a stage from the pipeline.

        Args:
            stage_name: Name of stage to remove

        Returns:
            True if removed, False if not found
        """
        pass

    @abstractmethod
    def get_stages(self) -> list["PipelineStage"]:
        """
        Get all pipeline stages.

        Returns:
            List of pipeline stages
        """
        pass


class PipelineStage(ABC):
    """Interface for pipeline processing stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for identification."""
        pass

    @abstractmethod
    async def process(self, data: Any) -> Any:
        """
        Process data in this stage.

        Args:
            data: Input data

        Returns:
            Processed data
        """
        pass

    @abstractmethod
    def can_process(self, data: Any) -> bool:
        """
        Check if this stage can process the data.

        Args:
            data: Data to check

        Returns:
            True if can process
        """
        pass


class DataCacheInterface(ABC):
    """Interface for data caching."""

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
            True if deleted
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
            True if exists
        """
        pass
