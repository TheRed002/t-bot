"""
Refactored Data Service - Service Layer Without Infrastructure Coupling

This module provides a clean service layer that depends on abstractions
rather than concrete infrastructure implementations.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.types import MarketData
from src.data.interfaces import (
    DataCacheInterface,
    DataServiceInterface,
    DataStorageInterface,
    DataValidatorInterface,
)
from src.data.types import CacheLevel, DataMetrics, DataRequest
from src.database.models import MarketDataRecord
from src.monitoring import MetricsCollector


class RefactoredDataService(BaseComponent, DataServiceInterface):
    """
    Refactored data service that uses dependency injection.
    
    This service depends on abstractions rather than concrete implementations,
    allowing for better testability and flexibility.
    """

    def __init__(
        self,
        config: Config,
        storage: DataStorageInterface,
        cache: DataCacheInterface | None = None,
        validator: DataValidatorInterface | None = None,
        metrics_collector: MetricsCollector | None = None,
    ):
        """
        Initialize the refactored data service.

        Args:
            config: Service configuration
            storage: Storage implementation
            cache: Optional cache implementation
            validator: Optional validator implementation
            metrics_collector: Optional metrics collector
        """
        super().__init__()
        self.config = config
        self.storage = storage
        self.cache = cache
        self.validator = validator
        self.metrics_collector = metrics_collector

        # Metrics tracking
        self._metrics = DataMetrics()
        self._last_metrics_reset = datetime.now(timezone.utc)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the data service and its dependencies."""
        if self._initialized:
            return

        self.logger.info("Initializing RefactoredDataService...")

        # Initialize storage
        if hasattr(self.storage, 'initialize'):
            await self.storage.initialize()

        # Initialize cache if available
        if self.cache and hasattr(self.cache, 'initialize'):
            await self.cache.initialize()

        self._initialized = True
        self.logger.info("RefactoredDataService initialized successfully")

    async def store_market_data(
        self,
        data: MarketData | list[MarketData],
        exchange: str,
        validate: bool = True,
        cache_levels: list[CacheLevel] | None = None,
    ) -> bool:
        """Store market data with validation and caching."""
        try:
            if not self._initialized:
                await self.initialize()

            # Normalize input
            data_list = data if isinstance(data, list) else [data]

            # Validate data if validator is available
            if validate and self.validator:
                data_list = await self.validator.validate_market_data(data_list)

            if not data_list:
                self.logger.warning("No valid market data to store")
                return False

            # Transform to database records
            db_records = await self._transform_to_db_records(data_list, exchange)

            # Store to database
            success = await self.storage.store_records(db_records)

            if success:
                # Update caches if specified and cache is available
                if cache_levels and self.cache:
                    await self._update_caches(data_list, cache_levels)

                # Update metrics
                self._metrics.records_processed += len(data_list)
                self._metrics.records_valid += len(data_list)

                # Record metrics to collector
                if self.metrics_collector:
                    self.metrics_collector.increment_counter(
                        "data_records_processed_total",
                        value=len(data_list),
                        labels={"exchange": exchange, "data_type": "market_data"},
                    )

                self.logger.info(f"Successfully stored {len(data_list)} market data records")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Market data storage failed: {e}")
            self._metrics.records_invalid += len(data_list) if 'data_list' in locals() else 1
            return False

    async def get_market_data(self, request: DataRequest) -> list[MarketDataRecord]:
        """Retrieve market data with intelligent caching."""
        try:
            if not self._initialized:
                await self.initialize()

            # Try cache first if enabled
            if request.use_cache and self.cache:
                cache_key = self._build_cache_key(request)
                cached_data = await self.cache.get(cache_key)
                if cached_data:
                    self._metrics.cache_hit_rate += 1
                    return [MarketDataRecord(**record) for record in cached_data]

            # Fetch from storage
            data = await self.storage.retrieve_records(request)

            # Update cache if enabled
            if request.use_cache and self.cache and data:
                await self._cache_data(request, data)

            return data

        except Exception as e:
            self.logger.error(f"Market data retrieval failed: {e}")
            return []

    async def get_data_count(self, symbol: str, exchange: str = "binance") -> int:
        """Get count of available data points for a symbol."""
        try:
            if not self._initialized:
                await self.initialize()

            return await self.storage.get_record_count(symbol, exchange)

        except Exception as e:
            self.logger.error(f"Data count retrieval failed for {symbol}: {e}")
            return 0

    async def get_recent_data(
        self, symbol: str, limit: int = 100, exchange: str = "binance"
    ) -> list[MarketData]:
        """Get recent market data for a symbol."""
        try:
            if not self._initialized:
                await self.initialize()

            # Create a data request
            request = DataRequest(
                symbol=symbol, 
                exchange=exchange, 
                limit=limit, 
                use_cache=True, 
                cache_ttl=3600
            )

            # Get records from storage
            records = await self.get_market_data(request)

            # Convert to MarketData objects
            market_data = []
            for record in records:
                data = MarketData(
                    symbol=record.symbol,
                    timestamp=record.timestamp or record.data_timestamp,
                    open=record.open_price or record.close_price or Decimal("0"),
                    high=record.high_price or record.close_price or Decimal("0"),
                    low=record.low_price or record.close_price or Decimal("0"),
                    close=record.close_price or Decimal("0"),
                    volume=record.volume or Decimal("0"),
                )
                market_data.append(data)

            # Sort by timestamp descending (most recent first)
            market_data.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)

            return market_data[:limit]

        except Exception as e:
            self.logger.error(f"Recent data retrieval failed for {symbol}: {e}")
            return []

    async def get_metrics(self) -> DataMetrics:
        """Get current data service metrics."""
        # Calculate derived metrics
        elapsed = (datetime.now(timezone.utc) - self._last_metrics_reset).total_seconds()

        if elapsed > 0:
            self._metrics.throughput_per_second = self._metrics.records_processed / elapsed

        if self._metrics.records_processed > 0:
            self._metrics.error_rate = (
                self._metrics.records_invalid / self._metrics.records_processed
            )

        return self._metrics

    async def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._metrics = DataMetrics()
        self._last_metrics_reset = datetime.now(timezone.utc)

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "components": {},
            "metrics": await self.get_metrics(),
        }

        # Check storage health
        storage_health = await self.storage.health_check()
        health["components"]["storage"] = storage_health["status"]
        if storage_health["status"] != "healthy":
            health["status"] = "degraded"

        # Check cache health if available
        if self.cache:
            cache_health = await self.cache.health_check()
            health["components"]["cache"] = cache_health["status"]
        else:
            health["components"]["cache"] = "disabled"

        # Check validator health if available
        if self.validator:
            validator_health = await self.validator.health_check()
            health["components"]["validator"] = validator_health["status"]
        else:
            health["components"]["validator"] = "disabled"

        return health

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        try:
            self.logger.info("Starting RefactoredDataService cleanup...")

            # Cleanup storage
            await self.storage.cleanup()

            # Cleanup cache if available
            if self.cache:
                await self.cache.cleanup()

            self._initialized = False
            self.logger.info("RefactoredDataService cleanup completed")

        except Exception as e:
            self.logger.error(f"RefactoredDataService cleanup error: {e}")

    async def _transform_to_db_records(
        self, data_list: list[MarketData], exchange: str
    ) -> list[MarketDataRecord]:
        """Transform MarketData to database records."""
        records = []

        for data in data_list:
            record = MarketDataRecord(
                symbol=data.symbol,
                exchange=exchange,
                timestamp=data.timestamp or datetime.now(timezone.utc),
                open_price=float(data.open_price) if data.open_price else None,
                high_price=float(data.high_price) if data.high_price else None,
                low_price=float(data.low_price) if data.low_price else None,
                close_price=float(data.price) if data.price else None,
                price=float(data.price) if data.price else None,
                volume=float(data.volume) if data.volume else None,
                bid=float(data.bid) if data.bid else None,
                ask=float(data.ask) if data.ask else None,
                data_source="exchange",
                quality_score=1.0,
                validation_status="valid",
            )
            records.append(record)

        return records

    async def _update_caches(
        self, data_list: list[MarketData], cache_levels: list[CacheLevel]
    ) -> None:
        """Update specified cache levels."""
        if not self.cache:
            return

        for data in data_list:
            cache_key = f"market_data:{data.symbol}:latest"
            cache_data = {
                "symbol": data.symbol,
                "price": str(data.price) if data.price else None,
                "volume": str(data.volume) if data.volume else None,
                "timestamp": data.timestamp.isoformat() if data.timestamp else None,
            }

            # Use default TTL of 300 seconds
            await self.cache.set(cache_key, cache_data, 300)

    async def _cache_data(self, request: DataRequest, data: list[MarketDataRecord]) -> None:
        """Cache data for future retrieval."""
        if not self.cache:
            return

        cache_key = self._build_cache_key(request)

        # Convert SQLAlchemy models to dictionaries
        cache_data = [
            {
                "id": str(record.id) if record.id else None,
                "symbol": record.symbol,
                "exchange": record.exchange,
                "open_price": str(record.open_price) if record.open_price else None,
                "high_price": str(record.high_price) if record.high_price else None,
                "low_price": str(record.low_price) if record.low_price else None,
                "close_price": str(record.close_price) if record.close_price else None,
                "volume": str(record.volume) if record.volume else None,
                "data_timestamp": (
                    record.data_timestamp.isoformat() if record.data_timestamp else None
                ),
                "interval": record.interval,
                "source": record.source,
            }
            for record in data
        ]

        ttl = request.cache_ttl or 3600  # Default 1 hour
        await self.cache.set(cache_key, cache_data, ttl)

    def _build_cache_key(self, request: DataRequest) -> str:
        """Build cache key from request parameters."""
        key_parts = [
            "market_data",
            f"symbol:{request.symbol}",
            f"exchange:{request.exchange}",
        ]

        if request.start_time:
            key_parts.append(f"start:{request.start_time.isoformat()}")
        if request.end_time:
            key_parts.append(f"end:{request.end_time.isoformat()}")
        if request.limit:
            key_parts.append(f"limit:{request.limit}")

        return ":".join(key_parts)