"""
Data Integration Service - DEPRECATED

This module is deprecated in favor of the new comprehensive DataService.
It remains for backward compatibility but should not be used for new implementations.

Please use src.data.services.data_service.DataService instead.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models, queries, and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import warnings
from datetime import datetime
from typing import Any

from src.base import BaseComponent
from src.core.config import Config
from src.core.types import MarketData, StorageMode

# Import from new DataService
from src.data.services.data_service import DataRequest, DataService

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution


class DataIntegrationService(BaseComponent):
    """
    DEPRECATED: Legacy data integration service.

    This class now acts as a wrapper around the new DataService to maintain
    backward compatibility. All direct database access has been removed
    and delegated to the proper DataService.

    Use DataService directly for new implementations.
    """

    def __init__(self, config: Config):
        """Initialize the legacy data integration service."""
        super().__init__()

        # Issue deprecation warning
        warnings.warn(
            "DataIntegrationService is deprecated. Use DataService instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.config = config
        self.error_handler = ErrorHandler(config)

        # Initialize new DataService
        self._data_service = DataService(config)

        # Legacy configuration for compatibility
        storage_config = getattr(config, "data_storage", {})
        if hasattr(storage_config, "get"):
            self.storage_mode = StorageMode(storage_config.get("mode", "batch"))
            self.batch_size = storage_config.get("batch_size", 100)
            self.cleanup_interval = storage_config.get("cleanup_interval", 3600)
        else:
            self.storage_mode = StorageMode.BATCH
            self.batch_size = 100
            self.cleanup_interval = 3600

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the legacy service."""
        if not self._initialized:
            await self._data_service.initialize()
            self._initialized = True
            self.logger.info(
                "Legacy DataIntegrationService initialized (delegating to DataService)"
            )

    @time_execution
    async def store_market_data(
        self, market_data: MarketData | list[MarketData], exchange: str = "unknown"
    ) -> bool:
        """
        DEPRECATED: Store market data using DataService.

        Args:
            market_data: Single MarketData object or list of MarketData objects
            exchange: Exchange name for the data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Delegate to DataService
            return await self._data_service.store_market_data(market_data, exchange, validate=True)

        except Exception as e:
            self.logger.error(f"Market data storage failed: {e}")
            return False

    async def get_market_data(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        """
        DEPRECATED: Retrieve market data using DataService.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            limit: Maximum number of records to return

        Returns:
            List of market data records
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Create DataRequest and delegate to DataService
            request = DataRequest(
                symbol=symbol,
                exchange=exchange,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

            return await self._data_service.get_market_data(request)

        except Exception as e:
            self.logger.error(f"Market data retrieval failed: {e}")
            return []

    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        DEPRECATED: Clean up old data using DataService.

        Args:
            days_to_keep: Number of days of data to keep

        Returns:
            Number of records cleaned up
        """
        try:
            if not self._initialized:
                await self.initialize()

            # DataService doesn't have cleanup_old_data method
            # This is a no-op for backward compatibility
            self.logger.warning("cleanup_old_data is deprecated and no longer implemented")
            return 0

        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """
        DEPRECATED: Perform health check using DataService.

        Returns:
            Health check results
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Get DataService health and add deprecation warning
            health = await self._data_service.health_check()
            health["deprecated"] = True
            health["message"] = "This service is deprecated. Use DataService directly."

            return health

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "deprecated": True,
                "message": "This service is deprecated. Use DataService directly.",
            }

    async def cleanup(self) -> None:
        """Cleanup legacy service resources."""
        try:
            if self._data_service:
                await self._data_service.cleanup()
            self.logger.info("Legacy DataIntegrationService cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during legacy DataIntegrationService cleanup: {e}")

    # All other legacy methods have been removed
    # Use DataService directly for new implementations
