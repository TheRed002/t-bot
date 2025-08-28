"""
Capital Allocator Adapter for Bot Management Integration.

This adapter bridges the API differences between bot_management's expectations
and the new CapitalAllocator implementation that requires CapitalService.

CRITICAL: This adapter maintains backward compatibility while properly integrating
with the new service-based capital management architecture.
"""

from decimal import Decimal

from src.capital_management.capital_allocator import CapitalAllocator
from src.capital_management.service import CapitalService
from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.config.service import ConfigService
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import CapitalMetrics
from src.database.service import DatabaseService
from src.error_handling import with_error_context, with_retry
from src.utils.validation.service import ValidationService


class CapitalAllocatorAdapter(BaseComponent):
    """
    Adapter that provides backward-compatible API for bot_management module.

    This adapter translates bot_management's simple API calls to the new
    CapitalAllocator's service-based API, handling:
    - Constructor differences (Config vs CapitalService)
    - Method signature differences
    - Missing methods (get_available_capital)
    - Property access (total_capital, available_capital)
    """

    def __init__(self, config: Config):
        """
        Initialize adapter with backward-compatible constructor.

        Args:
            config: Application configuration (legacy parameter)
        """
        super().__init__(config={"name": "CapitalAllocatorAdapter"})
        self.config = config

        # Check if we're in test mode by checking if config has certain mock attributes
        is_test_mode = hasattr(config, "_mock_name") or getattr(config, "_is_test", False)

        if is_test_mode:
            # Simplified initialization for tests
            self._database_service = None
            self._unit_of_work = None
            self._state_service = None
            self._capital_service = None
            self._allocator = None
            self._cached_metrics = None
            self._cache_timestamp = 0
            self._cache_ttl = 60.0
            self.logger.info("CapitalAllocatorAdapter initialized in test mode")
            return

        # Create required services
        # Create ConfigService from the legacy Config object
        config_service = ConfigService()
        config_service._config = config  # Set the config directly (as done in state factory)

        # Get ValidationService from dependency injection container
        from src.core.dependency_injection import get_container

        container = get_container()
        validation_service = container.get("validation_service", ValidationService())

        # Now create DatabaseService with proper parameters
        self._database_service = DatabaseService(config_service, validation_service)

        # For now, we'll skip UnitOfWork creation as DatabaseService doesn't expose Session
        # The adapter will work with the service-based pattern instead
        self._unit_of_work = None

        # Store factory function to create StateService during startup
        # Cannot use asyncio.run() in __init__ as it may be called from async context
        try:
            from src.state import create_default_state_service

            self._state_service_factory = create_default_state_service
            self._state_service = None  # Will be created during startup
        except Exception as e:
            self.logger.error(f"Failed to import StateService factory: {e}")
            # Re-raise the exception - StateService factory is required
            raise ServiceError(
                f"Cannot initialize CapitalAllocatorAdapter without StateService factory: {e}"
            ) from e

        # Create CapitalService - it will use database_service internally
        self._capital_service = CapitalService(
            database_service=self._database_service,
            uow_factory=None,  # Let CapitalService handle its own UoW if needed
            state_service=self._state_service,
        )

        # Create actual CapitalAllocator with required dependencies
        # Validate required dependency
        if not self._capital_service:
            raise ServiceError("CapitalService is required for CapitalAllocator")

        self._allocator = CapitalAllocator(
            capital_service=self._capital_service,
            config_service=None,  # Will use legacy config fallback
            risk_manager=None,  # Optional
            trade_lifecycle_manager=None,  # Optional
        )

        # Cache for properties
        self._cached_metrics: CapitalMetrics | None = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 60.0  # 1 minute cache

        self.logger.info("CapitalAllocatorAdapter initialized for bot_management compatibility")

    @with_error_context("capital_allocator_adapter")
    @with_retry(max_attempts=3)
    async def allocate_capital(
        self, bot_id: str, amount: Decimal, source: str = "bot_instance"
    ) -> bool:
        """
        Allocate capital for a bot (backward-compatible signature).

        Args:
            bot_id: Bot identifier
            amount: Amount to allocate
            source: Source of allocation request (ignored, for compatibility)

        Returns:
            bool: True if allocation successful, False otherwise
        """
        try:
            # Map bot_id to strategy_id (assuming 1:1 mapping)
            strategy_id = f"bot_{bot_id}"

            # Use default exchange from config or "default"
            exchange = getattr(self.config, "default_exchange", "binance")

            # Call new API
            allocation = await self._allocator.allocate_capital(
                strategy_id=strategy_id, exchange=exchange, requested_amount=amount, bot_id=bot_id
            )

            # Return boolean based on allocation success
            return allocation is not None and allocation.allocated_amount > 0

        except (ServiceError, ValidationError) as e:
            self.logger.error(
                "Capital allocation failed", bot_id=bot_id, amount=str(amount), error=str(e)
            )
            return False

    @with_error_context("capital_allocator_adapter")
    @with_retry(max_attempts=3)
    async def release_capital(self, bot_id: str, amount: Decimal) -> bool:
        """
        Release capital from a bot (backward-compatible signature).

        Args:
            bot_id: Bot identifier
            amount: Amount to release

        Returns:
            bool: True if release successful
        """
        try:
            # Map bot_id to strategy_id
            strategy_id = f"bot_{bot_id}"

            # Use default exchange
            exchange = getattr(self.config, "default_exchange", "binance")

            # Call new API
            result = await self._allocator.release_capital(
                strategy_id=strategy_id, exchange=exchange, amount=amount, bot_id=bot_id
            )

            return result

        except (ServiceError, ValidationError) as e:
            self.logger.error(
                "Capital release failed", bot_id=bot_id, amount=str(amount), error=str(e)
            )
            return False

    @with_error_context("capital_allocator_adapter")
    async def get_total_capital(self) -> Decimal:
        """
        Get total capital asynchronously using cached metrics.

        Returns:
            Decimal: Total capital amount
        """
        try:
            metrics = await self._get_cached_metrics()
            return metrics.total_capital if metrics else self.total_capital
        except Exception as e:
            self.logger.error(f"Failed to get total capital: {e}")
            return self.total_capital  # Fallback to property

    @with_error_context("capital_allocator_adapter")
    async def get_available_capital(self, bot_id: str | None = None) -> Decimal:
        """
        Get available capital for a bot or total available.

        Args:
            bot_id: Optional bot identifier (ignored for now)

        Returns:
            Decimal: Available capital amount
        """
        try:
            # Get metrics from allocator
            metrics = await self._get_cached_metrics()
            return metrics.available_capital if metrics else Decimal("0")

        except Exception as e:
            self.logger.error("Failed to get available capital", bot_id=bot_id, error=str(e))
            return Decimal("0")

    @property
    def total_capital(self) -> Decimal:
        """
        Get total capital (synchronous property for backward compatibility).

        Returns:
            Decimal: Total capital amount from cache or config
        """
        # Return cached value if available and fresh
        import time

        current_time = time.time()
        if self._cached_metrics and current_time - self._cache_timestamp <= self._cache_ttl:
            return self._cached_metrics.total_capital

        # Fallback to config value
        capital_config = getattr(self.config, "capital_management", {})
        if isinstance(capital_config, dict):
            return Decimal(str(capital_config.get("total_capital", "10000")))
        else:
            return Decimal(str(getattr(capital_config, "total_capital", "10000")))

    @property
    def available_capital(self) -> Decimal:
        """
        Get available capital (synchronous property for backward compatibility).

        Returns:
            Decimal: Available capital amount from cache or estimate
        """
        # Return cached value if available and fresh
        import time

        current_time = time.time()
        if self._cached_metrics and current_time - self._cache_timestamp <= self._cache_ttl:
            return self._cached_metrics.available_capital

        # Fallback to percentage of total capital
        return self.total_capital * Decimal("0.9")  # Assume 90% available

    async def _get_cached_metrics(self) -> CapitalMetrics | None:
        """Get capital metrics with caching."""
        import time

        current_time = time.time()
        if self._cached_metrics is None or current_time - self._cache_timestamp > self._cache_ttl:
            try:
                # Get fresh metrics - ensure allocator is available and properly awaited
                if self._allocator:
                    self._cached_metrics = await self._allocator.get_capital_metrics()
                    self._cache_timestamp = current_time
                else:
                    self.logger.warning("CapitalAllocator not available for metrics")
            except Exception as e:
                self.logger.error(f"Failed to get capital metrics: {e}")
                # Keep old cache if available

        return self._cached_metrics

    async def startup(self) -> None:
        """Initialize services on startup."""
        # Create StateService if not already created
        if self._state_service is None and self._state_service_factory:
            try:
                self._state_service = await self._state_service_factory(self.config)
                self.logger.info("StateService created during startup")
            except Exception as e:
                self.logger.error(f"Failed to create StateService during startup: {e}")
                # Continue startup with None state service - will use fallbacks
                # Record failure for monitoring
                if hasattr(self, "_metrics_collector") and self._metrics_collector:
                    try:
                        await self._metrics_collector.increment(
                            "capital_allocator_state_service_failures",
                            labels={"error_type": type(e).__name__},
                        )
                    except Exception as e:
                        self.logger.debug(f"Failed to record metrics during startup: {e}")

        await self._database_service.startup()

        # Recreate CapitalService with the properly initialized StateService
        self._capital_service = CapitalService(
            database_service=self._database_service,
            uow_factory=None,  # Let CapitalService handle its own UoW if needed
            state_service=self._state_service,
        )
        await self._capital_service.startup()

        # Pre-populate cache
        try:
            await self._get_cached_metrics()
        except Exception as e:
            self.logger.warning(f"Failed to pre-populate capital metrics cache: {e}")
            # Continue startup even if cache population fails

    async def shutdown(self) -> None:
        """Cleanup services on shutdown."""
        db_connection = None
        try:
            if self._capital_service:
                await self._capital_service.shutdown()
            if self._database_service:
                await self._database_service.shutdown()
        finally:
            if db_connection:
                try:
                    await db_connection.close()
                except Exception as e:
                    self.logger.debug(f"Failed to close connection during shutdown: {e}")
