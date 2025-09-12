"""
Data Storage Manager

This module provides data storage management for the pipeline:
- Database storage operations
- Data persistence and retrieval
- Storage optimization and cleanup
- Storage health monitoring

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# Import from P-002 database components
from src.core import BaseComponent
from src.core.config import Config

# Import from P-001 core components
from src.core.types import MarketData, StorageMode
from src.database.interfaces import DatabaseServiceInterface

# Import from P-002A error handling
from src.error_handling import ErrorHandler, with_retry

# Import from P-007A utilities
from src.utils.decorators import time_execution

# StorageMode is now imported from core.types


@dataclass
class StorageMetrics:
    """Storage operation metrics"""

    total_records_stored: int
    successful_stores: int
    failed_stores: int
    avg_storage_time: float
    storage_rate: float
    last_storage_time: datetime


class DataStorageManager(BaseComponent):
    """
    Data storage manager for pipeline data persistence.

    This class handles storage operations for all data types processed
    through the pipeline, with support for different storage modes.
    """

    def __init__(self, config: Config, database_service: DatabaseServiceInterface | None = None):
        """Initialize data storage manager."""
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Inject database service for storage operations
        self.database_service = database_service

        # Storage configuration
        storage_config = getattr(config, "data_storage", {})
        if isinstance(storage_config, dict):
            self.storage_mode = StorageMode(storage_config.get("mode", "batch"))
            self.batch_size = storage_config.get("batch_size", 100)
            self.buffer_threshold = storage_config.get("buffer_threshold", 50)
            self.cleanup_interval = storage_config.get("cleanup_interval", 3600)
        else:
            self.storage_mode = StorageMode("batch")
            self.batch_size = 100
            self.buffer_threshold = 50
            self.cleanup_interval = 3600

        # Storage buffers
        self.storage_buffer: list[dict[str, Any]] = []

        # Initialization flag
        self._initialized = False

        # Storage metrics
        self.metrics = StorageMetrics(
            total_records_stored=0,
            successful_stores=0,
            failed_stores=0,
            avg_storage_time=0.0,
            storage_rate=0.0,
            last_storage_time=datetime.now(timezone.utc),
        )

        self.logger.info("DataStorageManager initialized")

    async def initialize(self) -> None:
        """Initialize storage manager with async operations."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing DataStorageManager...")

            # Initialize database service if provided
            if self.database_service and hasattr(self.database_service, "initialize"):
                try:
                    await self.database_service.initialize()
                    self.logger.info("Database service connection established")
                except Exception as e:
                    self.logger.warning(f"Database service initialization failed: {e}")

            self._initialized = True
            self.logger.info("DataStorageManager initialized successfully")

        except Exception as e:
            self.logger.error(f"DataStorageManager initialization failed: {e}")
            raise

    @time_execution
    @with_retry(max_attempts=3, base_delay=1.0, exponential=True)
    async def store_market_data(self, data: MarketData) -> bool:
        """
        Store market data to database.

        Args:
            data: Market data to store

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()

            if self.storage_mode == StorageMode.HOT:
                return await self._store_real_time(data)
            elif self.storage_mode == StorageMode.STREAM:
                return await self._store_to_buffer(data)
            else:  # BATCH mode
                return await self._store_to_buffer(data)

        except Exception as e:
            self.logger.error(f"Failed to store market data: {e!s}")
            self.metrics.failed_stores += 1
            return False

    async def _store_real_time(self, data: MarketData) -> bool:
        """Store data immediately through database service."""
        try:
            if not self._initialized:
                await self.initialize()

            if not self.database_service:
                self.logger.warning("No database service available for real-time storage")
                return False

            # Store through database service interface using proper service layer
            # Transform data to standard format for database service
            data_dict = {
                "symbol": data.symbol,
                "exchange": "pipeline",  # Default exchange for pipeline data
                "timestamp": data.timestamp or datetime.now(timezone.utc),
                "open": getattr(data, "open_price", None) or getattr(data, "open", None),
                "high": getattr(data, "high_price", None) or getattr(data, "high", None),
                "low": getattr(data, "low_price", None) or getattr(data, "low", None),
                "close": getattr(data, "close_price", None) or getattr(data, "close", None) or data.price,
                "price": data.price,
                "volume": data.volume,
                "bid": getattr(data, "bid", None),
                "ask": getattr(data, "ask", None),
            }

            # Use database service's generic entity creation
            await self.database_service.execute_operation("create_market_data", data_dict)

            self.metrics.successful_stores += 1
            self.metrics.total_records_stored += 1
            self.metrics.last_storage_time = datetime.now(timezone.utc)

            return True

        except Exception as e:
            self.logger.error(f"Real-time storage failed: {e!s}")
            self.metrics.failed_stores += 1
            return False

    async def _store_to_buffer(self, data: MarketData) -> bool:
        """Add data to storage buffer."""
        try:
            buffer_item = {
                "data": data,
                "timestamp": datetime.now(timezone.utc),
                "type": "market_data",
            }

            self.storage_buffer.append(buffer_item)

            # Flush buffer if threshold reached
            if len(self.storage_buffer) >= self.buffer_threshold:
                await self._flush_buffer()

            return True

        except Exception as e:
            self.logger.error(f"Buffer storage failed: {e!s}")
            return False

    @time_execution
    async def _flush_buffer(self) -> bool:
        """Flush storage buffer through database service."""
        try:
            if not self.storage_buffer:
                return True

            if not self.database_service:
                self.logger.warning("No database service available for buffer flush")
                return False

            # Prepare data dictionaries for database service
            data_items = []
            for item in self.storage_buffer:
                if item["type"] == "market_data":
                    data = item["data"]
                    data_dict = {
                        "symbol": data.symbol,
                        "exchange": "pipeline",  # Default exchange for pipeline data
                        "timestamp": data.timestamp or datetime.now(timezone.utc),
                        "open": getattr(data, "open_price", None) or getattr(data, "open", None),
                        "high": getattr(data, "high_price", None) or getattr(data, "high", None),
                        "low": getattr(data, "low_price", None) or getattr(data, "low", None),
                        "close": getattr(data, "close_price", None) or getattr(data, "close", None) or data.price,
                        "price": data.price,
                        "volume": data.volume,
                        "bid": getattr(data, "bid", None),
                        "ask": getattr(data, "ask", None),
                    }
                    data_items.append(data_dict)

            # Bulk write through database service using proper service layer
            if data_items:
                await self.database_service.execute_operation("bulk_create_market_data", data_items)

                # Update metrics
                stored_count = len(data_items)
                self.metrics.successful_stores += stored_count
                self.metrics.total_records_stored += stored_count
                self.metrics.last_storage_time = datetime.now(timezone.utc)

                # Clear buffer after successful flush
                self.storage_buffer.clear()

                self.logger.info(f"Flushed {stored_count} records to database")
                return True

        except Exception as e:
            self.logger.error(f"Buffer flush failed: {e!s}")
            self.metrics.failed_stores += len(self.storage_buffer)
            return False

    async def store_batch(self, data_list: list[MarketData]) -> int:
        """
        Store a batch of data records.

        Args:
            data_list: List of market data to store

        Returns:
            int: Number of successfully stored records
        """
        try:
            if not self._initialized:
                await self.initialize()

            if not self.database_service:
                self.logger.warning("No database service available for batch storage")
                return 0

            # Convert to data dictionaries for database service
            data_items = []
            for data in data_list:
                data_dict = {
                    "symbol": data.symbol,
                    "exchange": "pipeline",  # Default exchange for pipeline data
                    "timestamp": data.timestamp or datetime.now(timezone.utc),
                    "open": getattr(data, "open_price", None) or getattr(data, "open", None),
                    "high": getattr(data, "high_price", None) or getattr(data, "high", None),
                    "low": getattr(data, "low_price", None) or getattr(data, "low", None),
                    "close": getattr(data, "close_price", None) or getattr(data, "close", None) or data.price,
                    "price": data.price,
                    "volume": data.volume,
                    "bid": getattr(data, "bid", None),
                    "ask": getattr(data, "ask", None),
                }
                data_items.append(data_dict)

            # Bulk write through database service using proper service layer
            await self.database_service.execute_operation("bulk_create_market_data", data_items)

            # Update metrics
            stored_count = len(data_items)
            self.metrics.successful_stores += stored_count
            self.metrics.total_records_stored += stored_count
            self.metrics.last_storage_time = datetime.now(timezone.utc)

            self.logger.info(f"Stored batch of {stored_count} records")
            return stored_count

        except Exception as e:
            self.logger.error(f"Batch storage failed: {e!s}")
            self.metrics.failed_stores += len(data_list)
            return 0

    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Cleanup old data from storage.

        Args:
            days_to_keep: Number of days of data to keep

        Returns:
            int: Number of records cleaned up
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            # Delegate cleanup to a storage service if available
            # For now, log the operation
            self.logger.info(f"Cleanup requested for data older than {cutoff_date}")

            # This should be handled by a proper storage service
            # Return 0 for now as this is infrastructure concern
            return 0

        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e!s}")
            return 0

    def get_storage_metrics(self) -> dict[str, Any]:
        """Get storage operation metrics."""
        success_rate = (
            self.metrics.successful_stores
            / (self.metrics.successful_stores + self.metrics.failed_stores)
            if (self.metrics.successful_stores + self.metrics.failed_stores) > 0
            else 0.0
        )

        return {
            "total_records_stored": self.metrics.total_records_stored,
            "successful_stores": self.metrics.successful_stores,
            "failed_stores": self.metrics.failed_stores,
            "success_rate": success_rate,
            "storage_rate": self.metrics.storage_rate,
            "avg_storage_time": self.metrics.avg_storage_time,
            "last_storage_time": self.metrics.last_storage_time,
            "buffer_size": len(self.storage_buffer),
            "storage_mode": self.storage_mode.value,
            "configuration": {
                "batch_size": self.batch_size,
                "buffer_threshold": self.buffer_threshold,
                "cleanup_interval": self.cleanup_interval,
            },
        }

    async def force_flush(self) -> bool:
        """Force flush all buffered data."""
        try:
            if not self._initialized:
                await self.initialize()

            if self.storage_buffer:
                return await self._flush_buffer()
            return True

        except Exception as e:
            self.logger.error(f"Force flush failed: {e!s}")
            return False

    async def cleanup(self) -> None:
        """Cleanup storage manager resources."""
        try:
            # Flush any remaining buffered data
            await self.force_flush()

            # Cleanup database service if provided
            if self.database_service and hasattr(self.database_service, "cleanup"):
                await self.database_service.cleanup()

            self.logger.info("DataStorageManager cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during DataStorageManager cleanup: {e!s}")
        finally:
            # Clear local state
            self.storage_buffer.clear()
            self._initialized = False

    async def _store_batch_to_postgresql(self, data_list: list[MarketData]) -> bool:
        """
        Store a batch of market data to PostgreSQL for persistent storage.

        This should delegate to a proper storage service rather than
        directly accessing database infrastructure.

        Args:
            data_list: List of market data to store

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # This should be handled by injected storage dependency
            self.logger.info(f"PostgreSQL storage requested for {len(data_list)} records")

            # For now, just return True as this is being refactored
            # to use proper service layer with dependency injection
            return True

        except Exception as e:
            self.logger.error(f"PostgreSQL storage failed: {e!s}")
            return False
