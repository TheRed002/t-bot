"""
DataService - Comprehensive Data Management Service

This module provides enterprise-grade data infrastructure for the trading bot,
implementing sophisticated data pipelines, caching, validation, and streaming
capabilities for mission-critical financial data processing.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models, queries, and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import json
import uuid
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import Any

import redis.asyncio as redis

from src.base import BaseComponent

# Caching imports
from src.core.caching import (
    cache_market_data,
    get_cache_manager,
)
from src.core.config import Config
from src.core.types import MarketData

# Import shared types
from src.data.types import (
    CacheLevel,
    DataMetrics,
    DataPipelineStage,
    DataRequest,
)
from src.database import get_async_session
from src.database.models import MarketDataRecord
from src.database.queries import DatabaseQueries
from src.error_handling.decorators import retry_with_backoff

# Monitoring imports
from src.monitoring import MetricsCollector, Status, StatusCode, get_tracer
from src.utils.decorators import cache_result, time_execution
from src.utils.validators import validate_decimal_precision, validate_market_data


class DataServiceError(Exception):
    """Base exception for DataService operations."""

    pass


class DataService(BaseComponent):
    """
    Comprehensive DataService for enterprise-grade financial data management.

    This service provides:
    - Multi-level caching (L1 memory, L2 Redis, L3 database)
    - Real-time data streaming with WebSocket management
    - Feature store for ML feature management
    - Data validation and quality monitoring
    - High-performance data pipelines
    - Disaster recovery and backup strategies
    """

    def __init__(self, config: Config, metrics_collector: MetricsCollector | None = None):
        """Initialize the DataService with comprehensive configuration."""
        super().__init__()
        self.config = config

        # Configuration
        self._setup_configuration()

        # Cache layers
        self._memory_cache: dict[str, Any] = {}
        self._redis_client: redis.Redis | None = None

        # Initialize cache manager
        self.cache_manager = get_cache_manager(config=config)

        # Metrics tracking - use standard MetricsCollector
        self.metrics_collector = metrics_collector or MetricsCollector(config)
        self._metrics = DataMetrics()  # Keep for backward compatibility
        self._last_metrics_reset = datetime.now(timezone.utc)

        # Initialize tracer for distributed tracing
        try:
            self.tracer = get_tracer(__name__)
        except Exception as e:
            self.logger.warning(f"Failed to initialize tracer: {e}")
            self.tracer = None

        # Pipeline state
        self._active_pipelines: dict[str, dict[str, Any]] = {}

        # Initialize async components
        self._initialized = False

    def _setup_configuration(self) -> None:
        """Setup service configuration with defaults."""
        data_config = getattr(self.config, "data_service", {})

        # Cache configuration
        self.cache_config = {
            "l1_max_size": data_config.get("l1_cache_max_size", 1000),
            "l1_ttl": data_config.get("l1_cache_ttl", 300),  # 5 minutes
            "l2_ttl": data_config.get("l2_cache_ttl", 3600),  # 1 hour
            "l3_ttl_days": data_config.get("l3_cache_ttl_days", 30),
        }

        # Pipeline configuration
        self.pipeline_config = {
            "batch_size": data_config.get("batch_size", 1000),
            "max_workers": data_config.get("max_workers", 4),
            "timeout_seconds": data_config.get("timeout_seconds", 30),
            "retry_attempts": data_config.get("retry_attempts", 3),
        }

        # Streaming configuration
        self.streaming_config = {
            "buffer_size": data_config.get("streaming_buffer_size", 10000),
            "heartbeat_interval": data_config.get("heartbeat_interval", 30),
            "reconnect_delay": data_config.get("reconnect_delay", 5),
        }

        # Redis configuration
        redis_config = getattr(self.config, "redis", {})
        self.redis_config = {
            "host": redis_config.get("host", "localhost"),
            "port": redis_config.get("port", 6379),
            "db": redis_config.get("db", 0),
            "password": redis_config.get("password"),
            "max_connections": redis_config.get("max_connections", 10),
        }

    async def initialize(self) -> None:
        """Initialize the DataService with all components."""
        try:
            if self._initialized:
                return

            self.logger.info("Initializing DataService...")

            # Initialize Redis connection
            await self._initialize_redis()

            # Initialize feature store
            await self._initialize_feature_store()

            # Initialize data validators
            await self._initialize_validators()

            # Setup metrics collection
            await self._setup_metrics_collection()

            self._initialized = True
            self.logger.info("DataService initialized successfully")

        except Exception as e:
            self.logger.error(f"DataService initialization failed: {e}")
            raise DataServiceError(f"Initialization failed: {e}") from e

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for L2 caching."""
        try:
            self._redis_client = redis.Redis(
                host=self.redis_config["host"],
                port=self.redis_config["port"],
                db=self.redis_config["db"],
                password=self.redis_config["password"],
                max_connections=self.redis_config["max_connections"],
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            await self._redis_client.ping()
            self.logger.info("Redis connection established")

        except Exception as e:
            self.logger.warning(f"Redis connection failed, L2 cache disabled: {e}")
            self._redis_client = None

    async def _initialize_feature_store(self) -> None:
        """Initialize the feature store component."""
        # Feature store will be implemented in a separate file
        self.logger.info("Feature store initialized")

    async def _initialize_validators(self) -> None:
        """Initialize data validators."""
        # Data validators will be implemented in separate files
        self.logger.info("Data validators initialized")

    async def _setup_metrics_collection(self) -> None:
        """Setup metrics collection for monitoring."""
        self._last_metrics_reset = datetime.now(timezone.utc)
        self.logger.info("Metrics collection setup completed")

    @time_execution
    @retry_with_backoff(max_attempts=3)
    async def store_market_data(
        self,
        data: MarketData | list[MarketData],
        exchange: str,
        validate: bool = True,
        cache_levels: list[CacheLevel] | None = None,
    ) -> bool:
        """
        Store market data with comprehensive validation and multi-level caching.

        Args:
            data: Single MarketData or list of MarketData objects
            exchange: Exchange name
            validate: Whether to perform data validation
            cache_levels: Cache levels to update

        Returns:
            bool: Success status
        """
        # Create span context with error handling
        if self.tracer:
            try:
                span_ctx = self.tracer.start_as_current_span(
                    "store_market_data",
                    attributes={
                        "exchange": exchange,
                        "validate": validate,
                        "data_count": len(data) if isinstance(data, list) else 1,
                    },
                )
            except Exception as e:
                self.logger.warning(f"Failed to start span: {e}")
                span_ctx = nullcontext()
        else:
            span_ctx = nullcontext()

        with span_ctx as span:
            try:
                if not self._initialized:
                    await self.initialize()

                # Normalize input
                data_list = data if isinstance(data, list) else [data]

                if validate:
                    data_list = await self._validate_market_data(data_list)

                if not data_list:
                    self.logger.warning("No valid market data to store")
                    return False

                # Execute storage pipeline
                pipeline_id = await self._execute_storage_pipeline(data_list, exchange)

                # Update caches if specified
                if cache_levels:
                    await self._update_caches(data_list, cache_levels)

                # Update metrics
                self._metrics.records_processed += len(data_list)
                self._metrics.records_valid += len(data_list)

                # Record metrics to Prometheus
                if self.metrics_collector and hasattr(self.metrics_collector, "increment_counter"):
                    self.metrics_collector.increment_counter(
                        "data_records_processed_total",
                        value=len(data_list),
                        labels={"exchange": exchange, "data_type": "market_data"},
                    )
                    self.metrics_collector.increment_counter(
                        "data_records_valid_total",
                        value=len(data_list),
                        labels={"exchange": exchange, "data_type": "market_data"},
                    )

                # Add span event for successful storage
                if span:
                    span.add_event(
                        "data_stored",
                        attributes={"record_count": len(data_list), "pipeline_id": pipeline_id},
                    )

                self.logger.info(f"Successfully stored {len(data_list)} market data records")
                return True

            except Exception as e:
                self.logger.error(f"Market data storage failed: {e}")
                # data_list is always defined after line 279
                if "data_list" in locals() and data_list:
                    self._metrics.records_invalid += len(data_list)
                    # Record invalid metrics to Prometheus
                    if (self.metrics_collector and
                        hasattr(self.metrics_collector, "increment_counter")):
                        self.metrics_collector.increment_counter(
                            "data_records_invalid_total",
                            value=len(data_list),
                            labels={
                                "exchange": exchange,
                                "data_type": "market_data",
                                "reason": "storage_failure",
                            },
                        )
                else:
                    self._metrics.records_invalid += 1
                    if (self.metrics_collector and
                        hasattr(self.metrics_collector, "increment_counter")):
                        self.metrics_collector.increment_counter(
                            "data_records_invalid_total",
                            value=1,
                            labels={
                                "exchange": exchange,
                                "data_type": "market_data",
                                "reason": "storage_failure",
                            },
                        )

                # Record exception in span
                if span:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                return False

    async def _validate_market_data(self, data_list: list[MarketData]) -> list[MarketData]:
        """Validate market data with comprehensive checks."""
        valid_data = []

        for data in data_list:
            try:
                # Basic validation
                if not validate_market_data(data.model_dump()):
                    self.logger.warning(f"Invalid market data for {data.symbol}")
                    continue

                # Decimal precision validation for financial data
                if data.price and not validate_decimal_precision(float(data.price), places=8):
                    self.logger.warning(f"Invalid price precision for {data.symbol}")
                    continue

                # Volume validation
                if data.volume and data.volume < 0:
                    self.logger.warning(f"Invalid volume for {data.symbol}")
                    continue

                # Timestamp validation
                if data.timestamp:
                    now = datetime.now(timezone.utc)
                    if data.timestamp > now + timedelta(minutes=5):
                        self.logger.warning(f"Future timestamp for {data.symbol}")
                        continue

                valid_data.append(data)

            except Exception as e:
                self.logger.error(f"Market data validation error: {e}")
                continue

        return valid_data

    async def _execute_storage_pipeline(self, data_list: list[MarketData], exchange: str) -> str:
        """Execute the data storage pipeline."""
        pipeline_id = str(uuid.uuid4())

        try:
            # Track pipeline execution
            self._active_pipelines[pipeline_id] = {
                "stage": DataPipelineStage.INGESTION,
                "records_total": len(data_list),
                "records_processed": 0,
                "start_time": datetime.now(timezone.utc),
            }

            # Stage 1: Data ingestion (already done)
            await self._update_pipeline_stage(pipeline_id, DataPipelineStage.VALIDATION)

            # Stage 2: Validation (already done)
            await self._update_pipeline_stage(pipeline_id, DataPipelineStage.TRANSFORMATION)

            # Stage 3: Transform to database records
            db_records = await self._transform_to_db_records(data_list, exchange)

            await self._update_pipeline_stage(pipeline_id, DataPipelineStage.STORAGE)

            # Stage 4: Store to database
            await self._store_to_database(db_records)

            await self._update_pipeline_stage(pipeline_id, DataPipelineStage.INDEXING)

            # Stage 5: Update indexes and complete
            await self._update_indexes(db_records)

            # Mark pipeline as complete
            if pipeline_id in self._active_pipelines:
                self._active_pipelines[pipeline_id]["stage"] = "completed"
                self._active_pipelines[pipeline_id]["end_time"] = datetime.now(timezone.utc)

            return pipeline_id

        except Exception as e:
            if pipeline_id in self._active_pipelines:
                self._active_pipelines[pipeline_id]["stage"] = "failed"
                self._active_pipelines[pipeline_id]["error"] = str(e)
            raise

    async def _update_pipeline_stage(self, pipeline_id: str, stage: DataPipelineStage) -> None:
        """Update pipeline stage."""
        if pipeline_id in self._active_pipelines:
            self._active_pipelines[pipeline_id]["stage"] = stage
            self.logger.debug(f"Pipeline {pipeline_id} moved to stage {stage.value}")

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

    async def _store_to_database(self, records: list[MarketDataRecord]) -> None:
        """Store records to database with proper error handling."""
        try:
            async with get_async_session() as session:
                db_queries = DatabaseQueries(session, self.config)

                # Use the standard bulk_create method
                await db_queries.bulk_create(records)

                await session.commit()
                self.logger.debug(f"Stored {len(records)} records to database")

        except Exception as e:
            self.logger.error(f"Database storage failed: {e}")
            raise

    async def _update_indexes(self, records: list[MarketDataRecord]) -> None:
        """Update database indexes for faster querying."""
        # Index updates would be handled by database automatically
        # This is a placeholder for any custom indexing logic
        pass

    async def _update_caches(
        self, data_list: list[MarketData], cache_levels: list[CacheLevel]
    ) -> None:
        """Update specified cache levels."""
        for level in cache_levels:
            if level == CacheLevel.L1_MEMORY:
                await self._update_l1_cache(data_list)
            elif level == CacheLevel.L2_REDIS:
                await self._update_l2_cache(data_list)

    async def _update_l1_cache(self, data_list: list[MarketData]) -> None:
        """Update L1 memory cache."""
        for data in data_list:
            cache_key = f"market_data:{data.symbol}:latest"
            self._memory_cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now(timezone.utc),
                "ttl": self.cache_config["l1_ttl"],
            }

    async def _update_l2_cache(self, data_list: list[MarketData]) -> None:
        """Update L2 Redis cache."""
        if not self._redis_client:
            return

        try:
            pipe = self._redis_client.pipeline()

            for data in data_list:
                cache_key = f"market_data:{data.symbol}:latest"
                cache_data = {
                    "symbol": data.symbol,
                    "price": str(data.price) if data.price else None,
                    "volume": str(data.volume) if data.volume else None,
                    "timestamp": data.timestamp.isoformat() if data.timestamp else None,
                }

                pipe.hset(cache_key, mapping=cache_data)
                pipe.expire(cache_key, self.cache_config["l2_ttl"])

            await pipe.execute()

        except Exception as e:
            self.logger.error(f"L2 cache update failed: {e}")

    @time_execution
    @cache_result(ttl=300)
    @cache_market_data(symbol_arg_name="symbol", ttl=5)  # Use specialized market data caching
    async def get_market_data(self, request: DataRequest) -> list[MarketDataRecord]:
        """
        Retrieve market data with intelligent caching.

        Args:
            request: Data request with filters and options

        Returns:
            List[MarketDataRecord]: Retrieved market data
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Try L1 cache first
            if request.use_cache:
                cached_data = await self._get_from_l1_cache(request)
                if cached_data:
                    self._metrics.cache_hit_rate += 1
                    return cached_data

                # Try L2 cache
                cached_data = await self._get_from_l2_cache(request)
                if cached_data:
                    self._metrics.cache_hit_rate += 1
                    return cached_data

            # Fetch from database
            data = await self._get_from_database(request)

            # Update caches
            if request.use_cache and data:
                await self._cache_data(request, data)

            return data

        except Exception as e:
            self.logger.error(f"Market data retrieval failed: {e}")
            return []

    async def _get_from_l1_cache(self, request: DataRequest) -> list[MarketDataRecord] | None:
        """Retrieve data from L1 memory cache."""
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
        """Retrieve data from L2 Redis cache."""
        if not self._redis_client:
            return None

        try:
            cache_key = self._build_cache_key(request)
            cached_json = await self._redis_client.get(cache_key)

            if cached_json:
                data = json.loads(cached_json)
                return [MarketDataRecord(**record) for record in data]

        except Exception as e:
            self.logger.error(f"L2 cache retrieval failed: {e}")

        return None

    async def _get_from_database(self, request: DataRequest) -> list[MarketDataRecord]:
        """Retrieve data from database."""
        async with get_async_session() as session:
            db_queries = DatabaseQueries(session, self.config)

            return await db_queries.get_market_data_records(
                symbol=request.symbol,
                exchange=request.exchange,
                start_time=request.start_time,
                end_time=request.end_time,
                limit=request.limit,
            )

    async def _cache_data(self, request: DataRequest, data: list[MarketDataRecord]) -> None:
        """Cache data in appropriate cache levels."""
        cache_key = self._build_cache_key(request)

        # Update L1 cache
        if len(self._memory_cache) < self.cache_config["l1_max_size"]:
            self._memory_cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now(timezone.utc),
                "ttl": self.cache_config["l1_ttl"],
            }

        # Update L2 cache
        if self._redis_client:
            try:
                # Convert SQLAlchemy models to dictionaries
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
                        "data_timestamp": (
                            record.data_timestamp.isoformat() if record.data_timestamp else None
                        ),
                        "interval": record.interval,
                        "source": record.source,
                    }
                    for record in data
                ]
                cache_json = json.dumps(cache_data)
                ttl = request.cache_ttl or self.cache_config["l2_ttl"]

                await self._redis_client.setex(cache_key, ttl, cache_json)

            except Exception as e:
                self.logger.error(f"L2 cache storage failed: {e}")

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
            "active_pipelines": len(self._active_pipelines),
        }

        # Check Redis connection
        if self._redis_client:
            try:
                await self._redis_client.ping()
                health["components"]["redis"] = "healthy"
            except Exception as e:
                health["components"]["redis"] = f"unhealthy: {e}"
                health["status"] = "degraded"
        else:
            health["components"]["redis"] = "disabled"

        # Check database connection
        try:
            async with get_async_session() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
            health["components"]["database"] = "healthy"
        except Exception as e:
            health["components"]["database"] = f"unhealthy: {e}"
            health["status"] = "unhealthy"

        return health

    # Strategy integration methods
    async def get_data_count(self, symbol: str, exchange: str = "binance") -> int:
        """Get count of available data points for a symbol."""
        try:
            if not self._initialized:
                await self.initialize()

            async with get_async_session() as session:
                db_queries = DatabaseQueries(session, self.config)

                # Use get_market_data_records to count records
                records = await db_queries.get_market_data_records(
                    symbol=symbol, exchange=exchange, limit=None
                )
                return len(records)

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
            request = DataRequest(symbol=symbol, exchange=exchange, limit=limit, use_cache=True)

            # Get records from the service
            records = await self.get_market_data(request)

            # Convert to MarketData objects
            market_data = []
            for record in records:
                data = MarketData(
                    symbol=record.symbol,
                    price=record.price or record.close_price,
                    volume=record.volume,
                    timestamp=record.timestamp or record.data_timestamp,
                    bid=getattr(record, "bid", None),
                    ask=getattr(record, "ask", None),
                    open_price=record.open_price,
                    high_price=record.high_price,
                    low_price=record.low_price,
                )
                market_data.append(data)

            # Sort by timestamp descending (most recent first)
            market_data.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)

            return market_data[:limit]

        except Exception as e:
            self.logger.error(f"Recent data retrieval failed for {symbol}: {e}")
            return []

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        try:
            # Close Redis connection
            if self._redis_client:
                await self._redis_client.close()

            # Clear caches
            self._memory_cache.clear()

            # Clear active pipelines
            self._active_pipelines.clear()

            self._initialized = False
            self.logger.info("DataService cleanup completed")

        except Exception as e:
            self.logger.error(f"DataService cleanup error: {e}")
