"""
Environment-aware Data Pipeline Integration.

This module extends the Data service with environment awareness,
providing different data handling, validation, and storage strategies
for sandbox vs live trading environments.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from src.core.integration.environment_aware_service import (
    EnvironmentAwareServiceMixin,
    EnvironmentContext,
)
from src.core.logging import get_logger
from src.core.types import MarketData

logger = get_logger(__name__)


class DataQualityLevel(Enum):
    """Data quality levels for different environments."""
    BASIC = "basic"           # Basic validation for sandbox
    STANDARD = "standard"     # Standard validation
    STRICT = "strict"         # Strict validation for production
    ULTRA_STRICT = "ultra_strict"  # Ultra-strict for critical production


class DataStorageStrategy(Enum):
    """Data storage strategies for different environments."""
    IN_MEMORY = "in_memory"           # In-memory only for testing
    CACHE_ONLY = "cache_only"         # Cache-based storage
    DATABASE = "database"             # Full database persistence
    HYBRID = "hybrid"                 # Cache + database hybrid


class EnvironmentAwareDataConfiguration:
    """Environment-specific data configuration."""

    @staticmethod
    def get_sandbox_data_config() -> dict[str, Any]:
        """Get data configuration for sandbox environment."""
        return {
            "data_quality_level": DataQualityLevel.BASIC,
            "storage_strategy": DataStorageStrategy.CACHE_ONLY,
            "enable_data_validation": True,
            "enable_anomaly_detection": False,  # Disabled for performance
            "enable_real_time_validation": False,
            "cache_ttl_seconds": 300,  # 5 minutes
            "batch_size": 1000,
            "max_data_age_minutes": 60,  # 1 hour
            "enable_data_compression": False,
            "enable_data_encryption": False,
            "enable_audit_logging": False,
            "max_memory_usage_mb": 512,
            "websocket_buffer_size": 10000,
            "enable_data_replay": True,  # For testing
            "enable_synthetic_data": True,  # For testing scenarios
            "data_retention_days": 7,  # Short retention for sandbox
            "enable_cross_validation": False,
            "parallel_processing": True,
            "max_concurrent_feeds": 10,
            "enable_data_profiling": True,  # For analysis in sandbox
        }

    @staticmethod
    def get_live_data_config() -> dict[str, Any]:
        """Get data configuration for live/production environment."""
        return {
            "data_quality_level": DataQualityLevel.STRICT,
            "storage_strategy": DataStorageStrategy.HYBRID,
            "enable_data_validation": True,
            "enable_anomaly_detection": True,  # Critical for production
            "enable_real_time_validation": True,
            "cache_ttl_seconds": 60,  # 1 minute for fresher data
            "batch_size": 500,  # Smaller batches for reliability
            "max_data_age_minutes": 5,  # 5 minutes max age
            "enable_data_compression": True,  # Save storage space
            "enable_data_encryption": True,  # Security requirement
            "enable_audit_logging": True,  # Compliance requirement
            "max_memory_usage_mb": 1024,  # Higher limit for production
            "websocket_buffer_size": 50000,
            "enable_data_replay": False,  # Not needed in production
            "enable_synthetic_data": False,  # Only real data in production
            "data_retention_days": 365,  # Long retention for production
            "enable_cross_validation": True,  # Data quality assurance
            "parallel_processing": True,
            "max_concurrent_feeds": 20,
            "enable_data_profiling": False,  # Disabled for performance
        }


class EnvironmentAwareDataManager(EnvironmentAwareServiceMixin):
    """
    Environment-aware data management functionality.
    
    This mixin adds environment-specific data handling, validation,
    and storage strategies to the Data service.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._environment_data_configs: dict[str, dict[str, Any]] = {}
        self._data_quality_metrics: dict[str, dict[str, Any]] = {}
        self._data_feeds: dict[str, list[str]] = {}
        self._anomaly_detectors: dict[str, Any] = {}

    async def _update_service_environment(self, context: EnvironmentContext) -> None:
        """Update data handling settings based on environment context."""
        await super()._update_service_environment(context)

        # Get environment-specific data configuration
        if context.is_production:
            data_config = EnvironmentAwareDataConfiguration.get_live_data_config()
            logger.info(f"Applied live data configuration for {context.exchange_name}")
        else:
            data_config = EnvironmentAwareDataConfiguration.get_sandbox_data_config()
            logger.info(f"Applied sandbox data configuration for {context.exchange_name}")

        self._environment_data_configs[context.exchange_name] = data_config

        # Initialize data quality tracking
        self._data_quality_metrics[context.exchange_name] = {
            "total_data_points": 0,
            "validation_failures": 0,
            "anomalies_detected": 0,
            "data_gaps": 0,
            "average_latency_ms": 0,
            "last_quality_check": None,
            "quality_score": Decimal("1.0"),
            "uptime_percentage": Decimal("100.0"),
            "error_rate": Decimal("0.0"),
        }

        # Initialize data feeds list
        self._data_feeds[context.exchange_name] = []

        # Setup environment-specific components
        await self._setup_environment_data_components(context.exchange_name, data_config)

    def get_environment_data_config(self, exchange: str) -> dict[str, Any]:
        """Get data configuration for a specific exchange environment."""
        if exchange not in self._environment_data_configs:
            # Initialize with default config based on current environment
            context = self.get_environment_context(exchange)
            if context.is_production:
                config = EnvironmentAwareDataConfiguration.get_live_data_config()
            else:
                config = EnvironmentAwareDataConfiguration.get_sandbox_data_config()
            self._environment_data_configs[exchange] = config

        return self._environment_data_configs[exchange]

    async def validate_market_data_for_environment(
        self,
        market_data: MarketData,
        exchange: str
    ) -> bool:
        """Validate market data with environment-specific quality checks."""
        context = self.get_environment_context(exchange)
        data_config = self.get_environment_data_config(exchange)
        quality_level = data_config.get("data_quality_level", DataQualityLevel.STANDARD)

        # Basic validation (all environments)
        if not await self._validate_basic_data_quality(market_data, exchange):
            return False

        # Environment-specific validation levels
        if quality_level == DataQualityLevel.BASIC:
            # Minimal validation for sandbox
            return await self._validate_sandbox_data(market_data, exchange)

        elif quality_level == DataQualityLevel.STRICT:
            # Strict validation for production
            return await self._validate_production_data(market_data, exchange, data_config)

        elif quality_level == DataQualityLevel.ULTRA_STRICT:
            # Ultra-strict validation for critical production
            return await self._validate_ultra_strict_data(market_data, exchange, data_config)

        else:
            # Standard validation
            return await self._validate_standard_data(market_data, exchange)

    async def _validate_basic_data_quality(self, market_data: MarketData, exchange: str) -> bool:
        """Basic data quality validation for all environments."""

        # Check required fields
        if not market_data.symbol:
            await self._log_data_issue("Missing symbol", exchange, "error")
            return False

        if not market_data.price or market_data.price <= 0:
            await self._log_data_issue("Invalid price", exchange, "error")
            return False

        if not market_data.timestamp:
            await self._log_data_issue("Missing timestamp", exchange, "warning")
            # Allow for basic validation but warn

        return True

    async def _validate_sandbox_data(self, market_data: MarketData, exchange: str) -> bool:
        """Sandbox-specific data validation (lenient)."""

        # Allow older data in sandbox for testing
        max_age = self.get_environment_data_config(exchange).get("max_data_age_minutes", 60)

        if market_data.timestamp:
            age_minutes = (datetime.now(timezone.utc) - market_data.timestamp).total_seconds() / 60
            if age_minutes > max_age:
                await self._log_data_issue(f"Data age {age_minutes:.1f}min exceeds limit", exchange, "info")
                # Allow but log for sandbox

        return True

    async def _validate_production_data(
        self,
        market_data: MarketData,
        exchange: str,
        data_config: dict[str, Any]
    ) -> bool:
        """Production-specific data validation (strict)."""

        # Strict timestamp validation
        if not market_data.timestamp:
            await self._log_data_issue("Timestamp required in production", exchange, "error")
            return False

        # Check data freshness
        max_age_minutes = data_config.get("max_data_age_minutes", 5)
        age_minutes = (datetime.now(timezone.utc) - market_data.timestamp).total_seconds() / 60

        if age_minutes > max_age_minutes:
            await self._log_data_issue(
                f"Data too old: {age_minutes:.1f}min > {max_age_minutes}min",
                exchange, "error"
            )
            return False

        # Price sanity checks
        if await self._is_price_anomalous(market_data, exchange):
            await self._log_data_issue("Anomalous price detected", exchange, "warning")
            # In production, we might reject or flag for review
            return await self._handle_anomalous_data(market_data, exchange)

        # Volume validation (if available)
        if hasattr(market_data, "volume") and market_data.volume is not None:
            if not await self._validate_volume_data(market_data, exchange):
                return False

        return True

    async def _validate_ultra_strict_data(
        self,
        market_data: MarketData,
        exchange: str,
        data_config: dict[str, Any]
    ) -> bool:
        """Ultra-strict data validation for critical production systems."""

        # All production validations plus additional checks
        if not await self._validate_production_data(market_data, exchange, data_config):
            return False

        # Cross-validation with multiple sources
        if data_config.get("enable_cross_validation"):
            if not await self._cross_validate_market_data(market_data, exchange):
                await self._log_data_issue("Cross-validation failed", exchange, "error")
                return False

        # Additional security checks
        if not await self._validate_data_integrity(market_data, exchange):
            await self._log_data_issue("Data integrity check failed", exchange, "error")
            return False

        return True

    async def _validate_standard_data(self, market_data: MarketData, exchange: str) -> bool:
        """Standard data validation."""

        # Check timestamp freshness (moderate requirements)
        if market_data.timestamp:
            age_minutes = (datetime.now(timezone.utc) - market_data.timestamp).total_seconds() / 60
            if age_minutes > 30:  # 30 minutes max for standard
                await self._log_data_issue(f"Data moderately old: {age_minutes:.1f}min", exchange, "warning")

        # Basic price validation
        if market_data.price and market_data.price > 0:
            # Check for reasonable price ranges (implement based on symbol)
            if await self._is_price_unreasonable(market_data, exchange):
                await self._log_data_issue("Unreasonable price detected", exchange, "warning")

        return True

    async def store_market_data_by_environment(
        self,
        market_data: MarketData,
        exchange: str
    ) -> bool:
        """Store market data using environment-specific storage strategy."""
        context = self.get_environment_context(exchange)
        data_config = self.get_environment_data_config(exchange)
        storage_strategy = data_config.get("storage_strategy", DataStorageStrategy.HYBRID)

        # Validate data before storage
        if not await self.validate_market_data_for_environment(market_data, exchange):
            logger.warning(f"Data validation failed for {exchange} - skipping storage")
            return False

        try:
            # Apply environment-specific storage
            if storage_strategy == DataStorageStrategy.IN_MEMORY:
                return await self._store_in_memory(market_data, exchange)

            elif storage_strategy == DataStorageStrategy.CACHE_ONLY:
                return await self._store_in_cache(market_data, exchange, data_config)

            elif storage_strategy == DataStorageStrategy.DATABASE:
                return await self._store_in_database(market_data, exchange, data_config)

            elif storage_strategy == DataStorageStrategy.HYBRID:
                # Store in both cache and database
                cache_success = await self._store_in_cache(market_data, exchange, data_config)
                db_success = await self._store_in_database(market_data, exchange, data_config)
                return cache_success and db_success

            else:
                logger.error(f"Unknown storage strategy: {storage_strategy}")
                return False

        except Exception as e:
            await self._log_data_issue(f"Storage failed: {e}", exchange, "error")
            return False

    async def _store_in_memory(self, market_data: MarketData, exchange: str) -> bool:
        """Store data in memory only (for testing)."""
        # Implementation would store in memory structure
        logger.debug(f"Stored {market_data.symbol} data in memory for {exchange}")
        return True

    async def _store_in_cache(
        self,
        market_data: MarketData,
        exchange: str,
        data_config: dict[str, Any]
    ) -> bool:
        """Store data in cache with environment-specific TTL."""
        ttl = data_config.get("cache_ttl_seconds", 300)
        compression = data_config.get("enable_data_compression", False)

        # Implementation would store in Redis/cache
        logger.debug(f"Stored {market_data.symbol} data in cache for {exchange} (TTL: {ttl}s)")
        return True

    async def _store_in_database(
        self,
        market_data: MarketData,
        exchange: str,
        data_config: dict[str, Any]
    ) -> bool:
        """Store data in database with environment-specific settings."""
        encryption = data_config.get("enable_data_encryption", False)
        audit_logging = data_config.get("enable_audit_logging", False)

        # Implementation would store in PostgreSQL/TimescaleDB
        logger.debug(f"Stored {market_data.symbol} data in database for {exchange}")

        if audit_logging:
            await self._log_data_audit_event(market_data, exchange, "stored")

        return True

    async def get_environment_aware_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str | None = None,
        limit: int | None = None
    ) -> list[MarketData]:
        """Retrieve market data with environment-specific filtering and validation."""
        context = self.get_environment_context(exchange)
        data_config = self.get_environment_data_config(exchange)

        # Determine data source based on environment
        if context.is_production:
            # Use high-quality, validated data sources
            data = await self._get_production_market_data(symbol, exchange, timeframe, limit)
        else:
            # Allow more flexible data sources for sandbox
            data = await self._get_sandbox_market_data(symbol, exchange, timeframe, limit)

        # Apply environment-specific post-processing
        filtered_data = await self._apply_environment_data_filters(data, exchange, data_config)

        # Update quality metrics
        await self._update_data_quality_metrics(exchange, len(filtered_data))

        return filtered_data

    async def _get_production_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str | None,
        limit: int | None
    ) -> list[MarketData]:
        """Get market data for production environment (strict quality)."""
        # Implementation would query verified, high-quality data sources
        logger.debug(f"Retrieving production market data for {symbol} on {exchange}")
        return []  # Mock implementation

    async def _get_sandbox_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str | None,
        limit: int | None
    ) -> list[MarketData]:
        """Get market data for sandbox environment (more flexible)."""
        data_config = self.get_environment_data_config(exchange)

        # Allow synthetic data generation for testing
        if data_config.get("enable_synthetic_data"):
            logger.debug(f"Including synthetic data for {symbol} on {exchange}")
            # Could generate test data here

        # Allow data replay for testing scenarios
        if data_config.get("enable_data_replay"):
            logger.debug(f"Data replay enabled for {symbol} on {exchange}")
            # Could replay historical data here

        logger.debug(f"Retrieving sandbox market data for {symbol} on {exchange}")
        return []  # Mock implementation

    async def _apply_environment_data_filters(
        self,
        data: list[MarketData],
        exchange: str,
        data_config: dict[str, Any]
    ) -> list[MarketData]:
        """Apply environment-specific filters to market data."""
        context = self.get_environment_context(exchange)

        filtered_data = []

        for item in data:
            # Age-based filtering
            max_age_minutes = data_config.get("max_data_age_minutes", 60)
            if item.timestamp:
                age_minutes = (datetime.now(timezone.utc) - item.timestamp).total_seconds() / 60
                if age_minutes > max_age_minutes:
                    continue

            # Anomaly detection (if enabled)
            if data_config.get("enable_anomaly_detection"):
                if await self._is_data_anomalous(item, exchange):
                    await self._log_data_issue(f"Anomaly detected in {item.symbol}", exchange, "warning")
                    if context.is_production:
                        continue  # Skip anomalous data in production

            filtered_data.append(item)

        return filtered_data

    async def _setup_environment_data_components(
        self,
        exchange: str,
        data_config: dict[str, Any]
    ) -> None:
        """Setup environment-specific data components."""

        # Setup anomaly detection if enabled
        if data_config.get("enable_anomaly_detection"):
            self._anomaly_detectors[exchange] = await self._create_anomaly_detector(exchange)
            logger.info(f"Anomaly detection enabled for {exchange}")

        # Setup data profiling if enabled
        if data_config.get("enable_data_profiling"):
            logger.info(f"Data profiling enabled for {exchange}")

        # Configure memory limits
        max_memory_mb = data_config.get("max_memory_usage_mb", 512)
        logger.info(f"Data memory limit set to {max_memory_mb}MB for {exchange}")

    async def _create_anomaly_detector(self, exchange: str) -> Any:
        """Create anomaly detector for the exchange."""
        # This would create actual anomaly detection instance
        return {"exchange": exchange, "type": "statistical"}

    async def _is_price_anomalous(self, market_data: MarketData, exchange: str) -> bool:
        """Check if price is anomalous using statistical methods."""
        # Implementation would use actual anomaly detection
        return False  # Mock: no anomaly detected

    async def _is_data_anomalous(self, market_data: MarketData, exchange: str) -> bool:
        """Check if data point is anomalous."""
        detector = self._anomaly_detectors.get(exchange)
        if not detector:
            return False

        # Implementation would run actual anomaly detection
        return False  # Mock: no anomaly

    async def _is_price_unreasonable(self, market_data: MarketData, exchange: str) -> bool:
        """Check if price is unreasonable (basic sanity check)."""
        # Implementation would check against reasonable price ranges
        # For crypto, prices can vary widely, so this would be symbol-specific
        return False  # Mock: price is reasonable

    async def _validate_volume_data(self, market_data: MarketData, exchange: str) -> bool:
        """Validate volume data if present."""
        if hasattr(market_data, "volume") and market_data.volume is not None:
            if market_data.volume < 0:
                await self._log_data_issue("Negative volume detected", exchange, "error")
                return False
        return True

    async def _cross_validate_market_data(self, market_data: MarketData, exchange: str) -> bool:
        """Cross-validate market data with multiple sources."""
        # Implementation would compare with other data sources
        return True  # Mock: validation passed

    async def _validate_data_integrity(self, market_data: MarketData, exchange: str) -> bool:
        """Validate data integrity and authenticity."""
        # Implementation would check data signatures, checksums, etc.
        return True  # Mock: integrity check passed

    async def _handle_anomalous_data(self, market_data: MarketData, exchange: str) -> bool:
        """Handle anomalous data detection in production."""
        context = self.get_environment_context(exchange)

        if context.is_production:
            # In production, might quarantine or require manual review
            logger.critical(f"Anomalous data detected in production for {exchange}: {market_data.symbol}")
            # Could implement quarantine logic here
            return False  # Reject anomalous data in production
        else:
            # In sandbox, might allow with warning
            logger.warning(f"Anomalous data in sandbox for {exchange}: {market_data.symbol}")
            return True  # Allow in sandbox for testing

    async def _log_data_issue(self, message: str, exchange: str, severity: str) -> None:
        """Log data quality issues with appropriate severity."""
        logger_method = getattr(logger, severity, logger.info)
        logger_method(f"Data quality issue for {exchange}: {message}")

        # Update quality metrics
        if exchange in self._data_quality_metrics:
            if severity == "error":
                self._data_quality_metrics[exchange]["validation_failures"] += 1
            elif severity == "warning" and "anomal" in message.lower():
                self._data_quality_metrics[exchange]["anomalies_detected"] += 1

    async def _log_data_audit_event(self, market_data: MarketData, exchange: str, action: str) -> None:
        """Log data audit events for compliance."""
        logger.info(
            f"Data audit: {action} {market_data.symbol} data for {exchange}",
            extra={
                "audit": True,
                "exchange": exchange,
                "symbol": market_data.symbol,
                "action": action,
                "timestamp": market_data.timestamp.isoformat() if market_data.timestamp else None,
                "price": str(market_data.price) if market_data.price else None,
            }
        )

    async def _update_data_quality_metrics(self, exchange: str, data_points_count: int) -> None:
        """Update data quality metrics."""
        if exchange not in self._data_quality_metrics:
            return

        metrics = self._data_quality_metrics[exchange]
        metrics["total_data_points"] += data_points_count
        metrics["last_quality_check"] = datetime.now().isoformat()

        # Calculate quality score
        total_points = metrics["total_data_points"]
        failures = metrics["validation_failures"]

        if total_points > 0:
            metrics["quality_score"] = max(Decimal("0"), Decimal("1.0") - (Decimal(failures) / Decimal(total_points)))
            metrics["error_rate"] = Decimal(failures) / Decimal(total_points) * 100

    def get_environment_data_metrics(self, exchange: str) -> dict[str, Any]:
        """Get data quality metrics for an exchange environment."""
        context = self.get_environment_context(exchange)
        data_config = self.get_environment_data_config(exchange)
        metrics = self._data_quality_metrics.get(exchange, {})

        return {
            "exchange": exchange,
            "environment": context.environment.value,
            "is_production": context.is_production,
            "data_quality_level": data_config.get("data_quality_level", DataQualityLevel.STANDARD).value,
            "storage_strategy": data_config.get("storage_strategy", DataStorageStrategy.HYBRID).value,
            "total_data_points": metrics.get("total_data_points", 0),
            "validation_failures": metrics.get("validation_failures", 0),
            "anomalies_detected": metrics.get("anomalies_detected", 0),
            "data_gaps": metrics.get("data_gaps", 0),
            "quality_score": str(metrics.get("quality_score", Decimal("1.0"))),
            "error_rate": str(metrics.get("error_rate", Decimal("0.0"))),
            "uptime_percentage": str(metrics.get("uptime_percentage", Decimal("100.0"))),
            "average_latency_ms": metrics.get("average_latency_ms", 0),
            "enable_anomaly_detection": data_config.get("enable_anomaly_detection", False),
            "enable_real_time_validation": data_config.get("enable_real_time_validation", False),
            "cache_ttl_seconds": data_config.get("cache_ttl_seconds", 300),
            "max_data_age_minutes": data_config.get("max_data_age_minutes", 60),
            "data_retention_days": data_config.get("data_retention_days", 30),
            "active_feeds": len(self._data_feeds.get(exchange, [])),
            "last_quality_check": metrics.get("last_quality_check"),
            "last_updated": datetime.now().isoformat()
        }
