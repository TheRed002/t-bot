"""
DataService - Simplified Data Management Service

This service provides core data functionality without unnecessary complexity:
- Store and retrieve market data
- Basic L1/L2 caching
- Database operations through service layer
- Simple validation and error handling

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models, queries, and connections
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.core import (
    BaseComponent,
    Config,
    DataError,
    DataValidationError,
    HealthCheckResult,
    HealthStatus,
)
from src.core.exceptions import DatabaseError
from src.core.types import MarketData
from src.data.constants import (
    DEFAULT_DATA_INTERVAL,
    DEFAULT_DATA_LIMIT,
    DEFAULT_DATA_SOURCE,
    DEFAULT_EXCHANGE,
    DEFAULT_L1_CACHE_MAX_SIZE,
    DEFAULT_L1_CACHE_TTL_SECONDS,
    DEFAULT_L2_CACHE_TTL_SECONDS,
    DEFAULT_QUALITY_SCORE,
    DEFAULT_VALIDATION_STATUS,
)
from src.data.types import DataRequest
from src.database.interfaces import DatabaseServiceInterface
from src.database.models import MarketDataRecord
from src.monitoring import MetricsCollector
from src.utils.decimal_utils import to_decimal
from src.utils.decorators import time_execution


class DataService(BaseComponent):
    """
    Simplified DataService for core data management.
    
    Provides:
    - Market data storage and retrieval
    - Basic L1 memory cache and L2 Redis cache
    - Database operations through service layer
    - Simple validation
    """

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface,
        cache_service=None,
        metrics_collector: MetricsCollector | None = None,
    ):
        """Initialize the DataService."""
        super().__init__()
        self.config = config

        # Required dependencies
        if database_service is None:
            raise ValueError("database_service is required and must be injected")
        self.database_service = database_service
        self.cache_service = cache_service
        self.metrics_collector = metrics_collector

        # Simple configuration
        data_config = getattr(self.config, "data_service", {}) or {}
        self.cache_config = {
            "l1_max_size": data_config.get("l1_cache_max_size", DEFAULT_L1_CACHE_MAX_SIZE),
            "l1_ttl": data_config.get("l1_cache_ttl", DEFAULT_L1_CACHE_TTL_SECONDS),
            "l2_ttl": data_config.get("l2_cache_ttl", DEFAULT_L2_CACHE_TTL_SECONDS),
        }

        # L1 memory cache
        self._memory_cache: dict[str, dict] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the DataService."""
        if self._initialized:
            return

        self.logger.info("Initializing DataService...")

        # Initialize cache service if provided
        if self.cache_service and hasattr(self.cache_service, "initialize"):
            await self.cache_service.initialize()

        self._initialized = True
        self.logger.info("DataService initialized successfully")

    @time_execution
    async def store_market_data(
        self,
        data: MarketData | list[MarketData],
        exchange: str,
        validate: bool = True,
    ) -> bool:
        """
        Store market data with validation.
        
        Args:
            data: Single MarketData or list of MarketData objects
            exchange: Exchange name
            validate: Whether to perform data validation
            
        Returns:
            bool: Success status
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Normalize input
            data_list = data if isinstance(data, list) else [data]

            if not data_list:
                raise DataValidationError(
                    "Empty data list provided",
                    field_name="data_list",
                    field_value="[]",
                    validation_rule="non_empty_list"
                )

            # Validate if requested
            if validate:
                data_list = self._validate_market_data(data_list)

            if not data_list:
                self.logger.warning("No valid market data to store")
                return False

            # Transform to database records
            db_records = self._transform_to_db_records(data_list, exchange)

            # Store to database
            await self._store_to_database(db_records)

            # Update L1 cache
            await self._update_l1_cache(data_list)

            self.logger.info(f"Successfully stored {len(data_list)} market data records")
            return True

        except DatabaseError as e:
            self.logger.error(f"Database error during market data storage: {e}")
            if self.metrics_collector:
                self.metrics_collector.increment_counter("data_service.storage.database_error")
            return False
        except DataValidationError as e:
            self.logger.error(f"Validation error during market data storage: {e}")
            if self.metrics_collector:
                self.metrics_collector.increment_counter("data_service.storage.validation_error")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during market data storage: {e}", exc_info=True)
            if self.metrics_collector:
                self.metrics_collector.increment_counter("data_service.storage.unexpected_error")
            return False

    def _validate_market_data(self, data_list: list[MarketData]) -> list[MarketData]:
        """Validate market data using consolidated validation utilities."""
        try:
            from src.utils.validation.market_data_validation import MarketDataValidator

            # Use consolidated validator
            validator = MarketDataValidator(
                enable_precision_validation=True,
                enable_consistency_validation=True,
                enable_timestamp_validation=False,  # May be None for some data
            )

            # Validate batch and return valid data
            return validator.validate_market_data_batch(data_list)

        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            # Fallback to returning original data without validation
            return data_list

    def _transform_to_db_records(self, data_list: list[MarketData], exchange: str) -> list[MarketDataRecord]:
        """Transform MarketData to database records."""
        records = []

        for data in data_list:
            try:
                record = MarketDataRecord(
                    symbol=data.symbol,
                    exchange=exchange,
                    data_timestamp=data.timestamp or datetime.now(timezone.utc),
                    timestamp=data.timestamp or datetime.now(timezone.utc),
                    open_price=to_decimal(data.open) if hasattr(data, "open") and data.open is not None else None,
                    high_price=to_decimal(data.high) if hasattr(data, "high") and data.high is not None else None,
                    low_price=to_decimal(data.low) if hasattr(data, "low") and data.low is not None else None,
                    close_price=to_decimal(data.close) if hasattr(data, "close") and data.close is not None else to_decimal(data.price) if data.price is not None else None,
                    price=to_decimal(data.price) if data.price is not None else None,
                    volume=to_decimal(data.volume) if data.volume is not None else None,
                    bid=to_decimal(data.bid) if hasattr(data, "bid") and data.bid is not None else None,
                    ask=to_decimal(data.ask) if hasattr(data, "ask") and data.ask is not None else None,
                    interval=DEFAULT_DATA_INTERVAL,
                    source=DEFAULT_DATA_SOURCE,
                    data_source=DEFAULT_DATA_SOURCE,
                    quality_score=to_decimal(DEFAULT_QUALITY_SCORE),
                    validation_status=DEFAULT_VALIDATION_STATUS,
                )
                records.append(record)

            except Exception as e:
                self.logger.error(f"Failed to transform data for {data.symbol}: {e}")
                continue

        return records

    async def _store_to_database(self, records: list[MarketDataRecord]) -> None:
        """Store records to database."""
        try:
            if not self.database_service:
                raise DataError("Database service not available")

            await self.database_service.bulk_create(records)
            self.logger.debug(f"Stored {len(records)} records to database")

        except DatabaseError as e:
            self.logger.error(f"Database storage failed: {e}")
            raise DataError(f"Database storage failed: {e}") from e

    async def _update_l1_cache(self, data_list: list[MarketData]) -> None:
        """Update L1 memory cache."""
        for data in data_list:
            cache_key = f"market_data:{data.symbol}:latest"
            self._memory_cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now(timezone.utc),
                "ttl": self.cache_config["l1_ttl"],
            }

    @time_execution
    async def get_market_data(self, request: DataRequest) -> list[MarketDataRecord]:
        """
        Retrieve market data with simple caching.
        
        Args:
            request: Data request with filters
            
        Returns:
            List[MarketDataRecord]: Retrieved market data
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Try L1 cache first
            if request.use_cache:
                cached_data = self._get_from_l1_cache(request)
                if cached_data:
                    return cached_data

                # Try L2 cache
                cached_data = await self._get_from_l2_cache(request)
                if cached_data:
                    return cached_data

            # Fetch from database
            data = await self._get_from_database(request)

            # Cache the data
            if request.use_cache and data:
                await self._cache_data(request, data)

            return data

        except DatabaseError as e:
            self.logger.error(f"Database error during market data retrieval: {e}")
            if self.metrics_collector:
                self.metrics_collector.increment_counter("data_service.retrieval.database_error")
            return []
        except DataError as e:
            self.logger.error(f"Data error during market data retrieval: {e}")
            if self.metrics_collector:
                self.metrics_collector.increment_counter("data_service.retrieval.data_error")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error during market data retrieval: {e}", exc_info=True)
            if self.metrics_collector:
                self.metrics_collector.increment_counter("data_service.retrieval.unexpected_error")
            return []

    def _get_from_l1_cache(self, request: DataRequest) -> list[MarketDataRecord] | None:
        """Get data from L1 memory cache."""
        cache_key = self._build_cache_key(request)

        if cache_key in self._memory_cache:
            cache_entry = self._memory_cache[cache_key]

            # Check TTL
            age = (datetime.now(timezone.utc) - cache_entry["timestamp"]).total_seconds()
            if age <= cache_entry["ttl"]:
                return cache_entry["data"]
            else:
                # Remove expired entry
                del self._memory_cache[cache_key]

        return None

    async def _get_from_l2_cache(self, request: DataRequest) -> list[MarketDataRecord] | None:
        """Get data from L2 cache."""
        if not self.cache_service:
            return None

        try:
            cache_key = self._build_cache_key(request)
            cached_data = await self.cache_service.get(cache_key)

            if cached_data:
                if isinstance(cached_data, list):
                    return [MarketDataRecord(**record) for record in cached_data]

        except Exception as e:
            self.logger.error(f"L2 cache retrieval failed: {e}")

        return None

    async def _get_from_database(self, request: DataRequest) -> list[MarketDataRecord]:
        """Get data from database."""
        try:
            if not self.database_service:
                raise DataError("Database service not available")

            # Build filters
            filters = {}
            if request.symbol:
                filters["symbol"] = request.symbol
            if request.exchange:
                filters["exchange"] = request.exchange
            if request.start_time or request.end_time:
                time_filter = {}
                if request.start_time:
                    time_filter["gte"] = request.start_time
                if request.end_time:
                    time_filter["lte"] = request.end_time
                filters["data_timestamp"] = time_filter

            return await self.database_service.list_entities(
                model_class=MarketDataRecord,
                filters=filters,
                order_by="data_timestamp",
                order_desc=True,
                limit=request.limit,
            )

        except Exception as e:
            self.logger.error(f"Database retrieval failed: {e}")
            raise DataError(f"Failed to retrieve data from database: {e}")

    async def _cache_data(self, request: DataRequest, data: list[MarketDataRecord]) -> None:
        """Cache data in L1 and L2."""
        cache_key = self._build_cache_key(request)

        # Update L1 cache
        if len(self._memory_cache) < self.cache_config["l1_max_size"]:
            self._memory_cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now(timezone.utc),
                "ttl": self.cache_config["l1_ttl"],
            }

        # Update L2 cache
        if self.cache_service:
            try:
                cache_data = [
                    {
                        "id": str(record.id),
                        "symbol": record.symbol,
                        "exchange": record.exchange,
                        "open_price": str(record.open_price) if record.open_price else None,
                        "high_price": str(record.high_price) if record.high_price else None,
                        "low_price": str(record.low_price) if record.low_price else None,
                        "close_price": str(record.close_price) if record.close_price else None,
                        "volume": str(record.volume) if record.volume else None,
                        "data_timestamp": record.data_timestamp.isoformat() if record.data_timestamp else None,
                        "interval": record.interval,
                        "source": record.source,
                    }
                    for record in data
                ]
                await self.cache_service.set(cache_key, cache_data, self.cache_config["l2_ttl"])

            except Exception as e:
                self.logger.error(f"L2 cache storage failed: {e}")

    def _build_cache_key(self, request: DataRequest) -> str:
        """Build cache key from request."""
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

    async def get_recent_data(
        self, symbol: str, limit: int = DEFAULT_DATA_LIMIT, exchange: str = DEFAULT_EXCHANGE
    ) -> list[MarketData]:
        """Get recent market data for a symbol."""
        try:
            request = DataRequest(
                symbol=symbol,
                exchange=exchange,
                limit=limit,
                use_cache=True
            )

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
                    exchange=record.exchange or exchange,
                )
                market_data.append(data)

            # Sort by timestamp descending
            market_data.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)
            return market_data[:limit]

        except Exception as e:
            self.logger.error(f"Recent data retrieval failed for {symbol}: {e}")
            return []

    async def get_data_count(self, symbol: str, exchange: str = DEFAULT_EXCHANGE) -> int:
        """Get count of available data points for a symbol."""
        try:
            if not self.database_service:
                return 0

            filters = {"symbol": symbol, "exchange": exchange}
            return await self.database_service.count_entities(
                model_class=MarketDataRecord, filters=filters
            )

        except Exception as e:
            self.logger.error(f"Data count retrieval failed for {symbol}: {e}")
            return 0

    async def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        status = HealthStatus.HEALTHY
        components = {}

        # Check cache service
        if self.cache_service:
            try:
                cache_health = await self.cache_service.health_check()
                components["cache"] = cache_health.get("status", "unknown")
                if cache_health.get("status") != "healthy":
                    status = HealthStatus.DEGRADED
            except Exception as e:
                components["cache"] = f"unhealthy: {e}"
                status = HealthStatus.DEGRADED
        else:
            components["cache"] = "disabled"

        # Check database
        try:
            if self.database_service:
                db_health = await self.database_service.get_health_status()
                if db_health.name == "HEALTHY":
                    components["database"] = "healthy"
                elif db_health.name == "DEGRADED":
                    components["database"] = "degraded"
                    status = HealthStatus.DEGRADED
                else:
                    components["database"] = "unhealthy"
                    status = HealthStatus.UNHEALTHY
            else:
                components["database"] = "unavailable"
                status = HealthStatus.DEGRADED
        except Exception as e:
            components["database"] = f"unhealthy: {e}"
            status = HealthStatus.UNHEALTHY

        details = {
            "initialized": self._initialized,
            "components": components,
        }

        return HealthCheckResult(
            status=status,
            details=details,
            message="DataService health check"
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.cache_service and hasattr(self.cache_service, "cleanup"):
                await self.cache_service.cleanup()

            self._memory_cache.clear()
            self._initialized = False

            self.logger.info("DataService cleanup completed")

        except Exception as e:
            self.logger.error(f"DataService cleanup error: {e}")
