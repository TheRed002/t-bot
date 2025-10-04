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
    DatabaseError,
    DataError,
    DataValidationError,
    HealthCheckResult,
    HealthStatus,
    MarketData,
)
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
            raise DataError("database_service is required and must be injected")
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

        # Metrics tracking (required by interface)
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_operations = 0

        # Technical indicators (lazy initialization)
        self._technical_indicators = None

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
        cache_levels: list | None = None,
        processing_mode: str = "stream",
    ) -> bool:
        """
        Store market data with validation using consistent processing paradigms aligned with database service.
        
        Args:
            data: Single MarketData or list of MarketData objects
            exchange: Exchange name
            validate: Whether to perform data validation
            cache_levels: Cache levels to use (from interface contract)
            processing_mode: Processing paradigm ("stream" for real-time, "batch" for bulk operations)
            
        Returns:
            bool: Success status
        """
        self._total_operations += 1
        try:
            if not self._initialized:
                await self.initialize()

            # Normalize input and apply processing paradigm alignment
            data_list = data if isinstance(data, list) else [data]

            if not data_list:
                raise DataValidationError(
                    "Empty data list provided",
                    field_name="data_list",
                    field_value="[]",
                    validation_rule="non_empty_list"
                )

            # Set processing mode optimization
            if processing_mode == "batch":
                batch_size = min(len(data_list), 1000)  # Batch processing limit
                self.logger.debug(f"Processing {len(data_list)} records in batch mode (size: {batch_size})")
            else:
                # Stream processing (default)
                self.logger.debug(f"Processing {len(data_list)} records in stream mode")

            # Validate if requested
            if validate:
                data_list = self._validate_market_data(data_list, processing_mode)

            if not data_list:
                self.logger.warning("No valid market data to store")
                return False

            # Apply processing paradigm alignment with error_handling module
            aligned_processing_mode = self._align_processing_paradigm(processing_mode, len(data_list))

            # Transform to database records with aligned processing mode context
            db_records = self._transform_to_db_records(data_list, exchange, aligned_processing_mode)

            # Store to database with aligned processing mode
            await self._store_to_database(db_records, aligned_processing_mode)

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

    def _validate_market_data(self, data_list: list[MarketData], processing_mode: str = "stream") -> list[MarketData]:
        """Validate market data using consolidated validation utilities."""
        try:
            from src.utils.validation.market_data_validation import MarketDataValidator

            # Use consolidated validator
            # Use 18 decimal places (Ethereum/ERC-20 standard) as per project requirements
            validator = MarketDataValidator(
                enable_precision_validation=True,
                enable_consistency_validation=True,
                enable_timestamp_validation=False,  # May be None for some data
                max_decimal_places=18,  # Ethereum/ERC-20 standard for crypto precision
            )

            # Validate batch and return valid data
            return validator.validate_market_data_batch(data_list)

        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            # Fallback to returning original data without validation
            return data_list

    def _transform_to_db_records(self, data_list: list[MarketData], exchange: str, processing_mode: str = "stream") -> list[MarketDataRecord]:
        """Transform MarketData to database records using consistent patterns aligned with error_handling module."""
        records = []

        for data in data_list:
            try:
                # Apply consistent data transformation patterns matching error_handling module
                transformed_data = self._apply_consistent_data_transformation(data, processing_mode)

                # Apply consistent decimal transformation matching database service _transform_entity_data
                record = MarketDataRecord(
                    symbol=transformed_data.symbol,
                    exchange=exchange,
                    data_timestamp=transformed_data.timestamp or datetime.now(timezone.utc),
                    timestamp=transformed_data.timestamp or datetime.now(timezone.utc),
                    open_price=to_decimal(transformed_data.open) if hasattr(transformed_data, "open") and transformed_data.open is not None else None,
                    high_price=to_decimal(transformed_data.high) if hasattr(transformed_data, "high") and transformed_data.high is not None else None,
                    low_price=to_decimal(transformed_data.low) if hasattr(transformed_data, "low") and transformed_data.low is not None else None,
                    close_price=to_decimal(transformed_data.close) if hasattr(transformed_data, "close") and transformed_data.close is not None else to_decimal(transformed_data.price) if transformed_data.price is not None else None,
                    price=to_decimal(transformed_data.price) if transformed_data.price is not None else None,
                    volume=to_decimal(transformed_data.volume) if transformed_data.volume is not None else None,
                    bid=to_decimal(transformed_data.bid_price) if hasattr(transformed_data, "bid_price") and transformed_data.bid_price is not None else None,
                    ask=to_decimal(transformed_data.ask_price) if hasattr(transformed_data, "ask_price") and transformed_data.ask_price is not None else None,
                    interval=DEFAULT_DATA_INTERVAL,
                    source=DEFAULT_DATA_SOURCE,
                    data_source=DEFAULT_DATA_SOURCE,
                    quality_score=to_decimal(DEFAULT_QUALITY_SCORE),
                    validation_status=DEFAULT_VALIDATION_STATUS,
                )

                # Processing metadata is already stored in transformed_data.metadata
                # No need to set additional attributes on the record

                records.append(record)

            except Exception as e:
                # Use consistent error propagation patterns
                from src.utils.messaging_patterns import ErrorPropagationMixin
                mixin = ErrorPropagationMixin()
                try:
                    mixin.propagate_service_error(e, f"data_transformation.{data.symbol}")
                except Exception:
                    # Fallback to regular logging if propagation fails
                    self.logger.error(f"Failed to transform data for {data.symbol}: {e}")
                    continue

        return records

    def _apply_consistent_data_transformation(self, data: MarketData, processing_mode: str) -> MarketData:
        """Apply consistent data transformation patterns matching error_handling module."""
        # Add processing metadata to the metadata dictionary instead of trying to set attributes
        metadata = getattr(data, "metadata", {}).copy()
        metadata.update({
            "processing_mode": processing_mode,
            "data_format": "event_data_v1",  # Align with error_handling format
            "transformation_timestamp": datetime.now(timezone.utc).isoformat(),
            "boundary_crossed": True
        })
        
        # Create transformed data with consistent metadata
        transformed_data = MarketData(
            symbol=data.symbol,
            timestamp=data.timestamp or datetime.now(timezone.utc),
            open=getattr(data, "open", data.price),
            high=getattr(data, "high", data.price),
            low=getattr(data, "low", data.price),
            close=getattr(data, "close", data.price),
            volume=data.volume,
            quote_volume=getattr(data, "quote_volume", None),
            trades_count=getattr(data, "trades_count", None),
            vwap=getattr(data, "vwap", None),
            exchange=getattr(data, "exchange", "unknown"),
            metadata=metadata,
            bid_price=getattr(data, "bid_price", None) or getattr(data, "bid", None),
            ask_price=getattr(data, "ask_price", None) or getattr(data, "ask", None),
        )

        return transformed_data

    def _align_processing_paradigm(self, processing_mode: str, data_count: int) -> str:
        """Align processing paradigm with error_handling module patterns."""
        # Apply paradigm alignment logic matching error_handling module
        if processing_mode == "batch" and data_count == 1:
            # Single item should use stream processing for consistency with error_handling
            return "stream"
        elif processing_mode == "stream" and data_count > 100:
            # Large data sets can benefit from batch processing
            return "batch"
        else:
            # Keep original mode if alignment not needed
            return processing_mode

    async def _store_to_database(self, records: list[MarketDataRecord], processing_mode: str = "stream") -> None:
        """Store records to database with processing paradigm alignment."""
        try:
            if not self.database_service:
                raise DataError("Database service not available")

            # Store to database using service layer
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
        self._total_operations += 1
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
                self._cache_hits += 1
                return cache_entry["data"]
            else:
                # Remove expired entry
                del self._memory_cache[cache_key]

        self._cache_misses += 1
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

            # Get most recent data first (DESC order), then reverse to chronological order
            # This ensures we get the LATEST N records, not the OLDEST N records
            records = await self.database_service.list_entities(
                model_class=MarketDataRecord,
                filters=filters,
                order_by="data_timestamp",
                order_desc=True,  # DESC order to get most recent records first
                limit=request.limit,
            )

            # Reverse to chronological order (oldest first) for technical indicator calculations
            return list(reversed(records))

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

            # Data is already sorted by database query (ascending order, oldest first)
            # No need to re-sort
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

    async def get_volatility(
        self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE
    ) -> Decimal | None:
        """Get volatility for a symbol."""
        try:
            request = DataRequest(
                symbol=symbol,
                exchange=exchange,
                limit=period,
                data_types=["volatility"]
            )

            records = await self.get_market_data(request)
            if not records:
                return None

            # Calculate simple volatility from price changes
            if len(records) < 2:
                return None

            prices = [record.close_price for record in records]
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] and prices[i]:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)

            if not returns:
                return None

            # Calculate standard deviation
            mean_return = sum(returns) / len(returns)
            variance = sum((ret - mean_return) ** 2 for ret in returns) / len(returns)
            volatility = variance ** Decimal("0.5")

            return volatility

        except Exception as e:
            self.logger.error(f"Failed to get volatility for {symbol}: {e}")
            raise DataError(f"Failed to get volatility for {symbol}: {e}")

    def _get_technical_indicators(self):
        """Get or create technical indicators instance (lazy initialization)."""
        if self._technical_indicators is None:
            from src.data.features.technical_indicators import TechnicalIndicators
            self._technical_indicators = TechnicalIndicators(
                config=self.config,
                data_service=self
            )
        return self._technical_indicators

    async def get_rsi(self, symbol: str, period: int = 14, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None:
        """Calculate RSI (Relative Strength Index) for a symbol.

        Args:
            symbol: Trading symbol
            period: RSI period (default 14)
            exchange: Exchange name

        Returns:
            RSI value as Decimal or None if insufficient data
        """
        try:
            indicators = self._get_technical_indicators()
            return await indicators.calculate_rsi(symbol, period)
        except Exception as e:
            self.logger.error(f"Failed to calculate RSI for {symbol}: {e}")
            return None

    async def get_sma(self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None:
        """Calculate SMA (Simple Moving Average) for a symbol.

        Args:
            symbol: Trading symbol
            period: SMA period (default 20)
            exchange: Exchange name

        Returns:
            SMA value as Decimal or None if insufficient data
        """
        try:
            indicators = self._get_technical_indicators()
            return await indicators.calculate_sma(symbol, period)
        except Exception as e:
            self.logger.error(f"Failed to calculate SMA for {symbol}: {e}")
            return None

    async def get_ema(self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None:
        """Calculate EMA (Exponential Moving Average) for a symbol.

        Args:
            symbol: Trading symbol
            period: EMA period (default 20)
            exchange: Exchange name

        Returns:
            EMA value as Decimal or None if insufficient data
        """
        try:
            indicators = self._get_technical_indicators()
            return await indicators.calculate_ema(symbol, period)
        except Exception as e:
            self.logger.error(f"Failed to calculate EMA for {symbol}: {e}")
            return None

    async def get_macd(self, symbol: str, exchange: str = DEFAULT_EXCHANGE) -> dict[str, Decimal] | None:
        """Calculate MACD (Moving Average Convergence Divergence) for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dictionary with 'macd', 'signal', 'histogram' values or None if insufficient data
        """
        try:
            indicators = self._get_technical_indicators()
            # Get recent prices for MACD calculation
            recent_data = await self.get_recent_data(symbol, limit=50, exchange=exchange)
            if not recent_data:
                return None

            prices = [data.close for data in recent_data]
            return await indicators.macd(prices)
        except Exception as e:
            self.logger.error(f"Failed to calculate MACD for {symbol}: {e}")
            return None

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

    def _validate_data_to_database_boundary(self, data_list: list[MarketData], processing_mode: str) -> None:
        """Validate data compatibility for database storage."""
        for i, data in enumerate(data_list):
            try:
                # Basic validation for database compatibility
                if not data.symbol:
                    raise DataValidationError(
                        f"Symbol is required for database storage (record {i})",
                        field_name="symbol",
                        field_value=None,
                        validation_rule="required"
                    )

                # Validate price/volume are valid numbers if present
                if data.price is not None and float(data.price) < 0:
                    raise DataValidationError(
                        f"Price cannot be negative (record {i})",
                        field_name="price",
                        field_value=str(data.price),
                        validation_rule="non_negative"
                    )

                if data.volume is not None and float(data.volume) < 0:
                    raise DataValidationError(
                        f"Volume cannot be negative (record {i})",
                        field_name="volume",
                        field_value=str(data.volume),
                        validation_rule="non_negative"
                    )

            except Exception as e:
                self.logger.error(f"Data validation failed for record {i}: {e}")
                if not isinstance(e, DataValidationError):
                    raise DataValidationError(
                        f"Data validation failed for record {i}",
                        field_name=f"record_{i}",
                        field_value=str(data),
                        validation_rule="database_compatibility"
                    ) from e
                raise

    async def get_metrics(self):
        """Get current data service metrics (required by interface)."""
        return {
            "cache_size": len(self._memory_cache),
            "initialized": self._initialized,
            "cache_hits": getattr(self, "_cache_hits", 0),
            "cache_misses": getattr(self, "_cache_misses", 0),
            "total_operations": getattr(self, "_total_operations", 0),
        }

    async def reset_metrics(self) -> None:
        """Reset metrics counters (required by interface)."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_operations = 0

    async def store_market_data_batch(self, market_data_list: list[MarketData]) -> bool:
        """
        Store multiple market data records in batch.

        Args:
            market_data_list: List of market data to store

        Returns:
            bool: True if all stored successfully
        """
        try:
            if not market_data_list:
                return True

            # Convert to database records
            records = []
            for data in market_data_list:
                record = MarketDataRecord(
                    symbol=data.symbol,
                    open=data.open,
                    high=data.high,
                    low=data.low,
                    close=data.close,
                    volume=data.volume,
                    timestamp=data.timestamp,
                    exchange=data.exchange or DEFAULT_EXCHANGE,
                    bid_price=data.bid_price,
                    ask_price=data.ask_price,
                    validation_status=DEFAULT_VALIDATION_STATUS,
                    quality_score=DEFAULT_QUALITY_SCORE,
                    data_source=DEFAULT_DATA_SOURCE,
                )
                records.append(record)

            # Bulk insert to database
            await self._store_to_database(records, processing_mode="batch")

            # Update cache with latest data
            await self._update_l1_cache(market_data_list)

            self.logger.info(f"Stored {len(records)} market data records in batch")
            return True

        except Exception as e:
            self.logger.error(f"Batch storage failed: {e}")
            raise DataError(f"Batch storage failed: {e}") from e

    async def aggregate_market_data(
        self,
        symbol: str,
        source_timeframe: str,
        target_timeframe: str,
        periods: int,
        exchange: str = DEFAULT_EXCHANGE,
    ) -> list[MarketData]:
        """
        Aggregate market data from source timeframe to target timeframe.

        Args:
            symbol: Trading symbol
            source_timeframe: Source timeframe (e.g., '1h')
            target_timeframe: Target timeframe (e.g., '4h')
            periods: Number of target periods to generate
            exchange: Exchange name

        Returns:
            List of aggregated MarketData
        """
        try:
            # Parse timeframe multiplier (simple implementation)
            timeframe_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

            if source_timeframe not in timeframe_map or target_timeframe not in timeframe_map:
                raise DataError(f"Unsupported timeframe: {source_timeframe} or {target_timeframe}")

            source_minutes = timeframe_map[source_timeframe]
            target_minutes = timeframe_map[target_timeframe]

            if target_minutes <= source_minutes:
                raise DataError("Target timeframe must be larger than source timeframe")

            # Calculate how many source periods per target period
            periods_per_aggregation = target_minutes // source_minutes

            # Fetch source data
            total_source_periods = periods * periods_per_aggregation
            request = DataRequest(
                symbol=symbol, exchange=exchange, limit=total_source_periods, interval=source_timeframe
            )

            source_records = await self.get_market_data(request)

            if len(source_records) < total_source_periods:
                self.logger.warning(
                    f"Insufficient data for aggregation: got {len(source_records)}, need {total_source_periods}"
                )

            # Aggregate data
            aggregated = []
            for i in range(0, len(source_records), periods_per_aggregation):
                chunk = source_records[i : i + periods_per_aggregation]
                if len(chunk) == periods_per_aggregation:
                    # Create aggregated candle
                    agg_data = MarketData(
                        symbol=symbol,
                        open=chunk[0].open_price,
                        high=max(r.high_price for r in chunk),
                        low=min(r.low_price for r in chunk),
                        close=chunk[-1].close_price,
                        volume=sum(r.volume for r in chunk),
                        timestamp=chunk[-1].timestamp or chunk[-1].data_timestamp,
                        exchange=exchange,
                        bid_price=chunk[-1].bid if hasattr(chunk[-1], 'bid') else None,
                        ask_price=chunk[-1].ask if hasattr(chunk[-1], 'ask') else None,
                    )
                    aggregated.append(agg_data)

            self.logger.info(
                f"Aggregated {len(source_records)} {source_timeframe} records into {len(aggregated)} {target_timeframe} records"
            )
            return aggregated

        except Exception as e:
            self.logger.error(f"Market data aggregation failed: {e}")
            raise DataError(f"Market data aggregation failed: {e}") from e

    async def get_market_data_history(
        self, symbol: str, limit: int = 100, exchange: str = DEFAULT_EXCHANGE
    ) -> list[MarketData]:
        """
        Get market data history in chronological order.

        Args:
            symbol: Trading symbol
            limit: Number of records to retrieve
            exchange: Exchange name

        Returns:
            List of MarketData in chronological order (oldest first)
        """
        try:
            request = DataRequest(symbol=symbol, exchange=exchange, limit=limit)
            records = await self.get_market_data(request)

            # Convert records to MarketData objects
            market_data_list = [
                MarketData(
                    symbol=r.symbol,
                    open=r.open_price,
                    high=r.high_price,
                    low=r.low_price,
                    close=r.close_price,
                    volume=r.volume,
                    timestamp=r.timestamp or r.data_timestamp,
                    exchange=r.exchange,
                    bid_price=r.bid if hasattr(r, 'bid') else None,
                    ask_price=r.ask if hasattr(r, 'ask') else None,
                )
                for r in records
            ]

            return market_data_list

        except Exception as e:
            self.logger.error(f"Failed to get market data history: {e}")
            raise DataError(f"Failed to get market data history: {e}") from e

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
