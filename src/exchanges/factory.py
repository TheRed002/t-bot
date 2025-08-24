"""
Enhanced Exchange Factory with Dependency Injection.

This module provides a factory pattern with dependency injection service container
for creating exchange instances dynamically from configuration, with support for
hot-swapping, connection pooling, and service registration.

CRITICAL: This integrates with P-001 (core types, exceptions, config)
and P-002A (error handling) components.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from src.core.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import ExchangeError, ValidationError

# MANDATORY: Import from P-001
from src.core.types import ExchangeInfo, ExchangeStatus

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# Import base exchange interface
from src.exchanges.base import BaseExchange

# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import log_calls, retry, time_execution

T = TypeVar("T")


class ServiceContainer:
    """
    Dependency injection container for managing services and their dependencies.
    """

    def __init__(self):
        self._services: dict[str, Any] = {}
        self._factories: dict[str, Callable] = {}
        self._singletons: dict[str, Any] = {}
        self._lifecycle: dict[str, str] = {}  # singleton, transient, scoped
        self._dependencies: dict[str, list[str]] = {}

    def register_singleton(
        self, service_name: str, factory: Callable, dependencies: list[str] | None = None
    ) -> None:
        """Register a singleton service."""
        self._factories[service_name] = factory
        self._lifecycle[service_name] = "singleton"
        self._dependencies[service_name] = dependencies or []

    def register_transient(
        self, service_name: str, factory: Callable, dependencies: list[str] | None = None
    ) -> None:
        """Register a transient service (new instance each time)."""
        self._factories[service_name] = factory
        self._lifecycle[service_name] = "transient"
        self._dependencies[service_name] = dependencies or []

    def register_instance(self, service_name: str, instance: Any) -> None:
        """Register a pre-created instance."""
        self._services[service_name] = instance
        self._lifecycle[service_name] = "singleton"

    def resolve(self, service_name: str) -> Any:
        """Resolve a service and its dependencies."""
        # Check if already resolved for singletons
        if service_name in self._singletons:
            return self._singletons[service_name]

        # Check if pre-registered instance
        if service_name in self._services:
            return self._services[service_name]

        # Check if factory exists
        if service_name not in self._factories:
            raise ValidationError(f"Service '{service_name}' not registered")

        # Resolve dependencies first
        resolved_dependencies = {}
        for dep in self._dependencies.get(service_name, []):
            resolved_dependencies[dep] = self.resolve(dep)

        # Create the service instance
        factory = self._factories[service_name]
        instance = factory(**resolved_dependencies)

        # Store singleton instances
        if self._lifecycle.get(service_name) == "singleton":
            self._singletons[service_name] = instance

        return instance

    def has_service(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self._services or service_name in self._factories

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._lifecycle.clear()
        self._dependencies.clear()


class HealthMonitor:
    """
    Health monitoring service for exchanges.
    """

    def __init__(self, config: Config):
        self.config = config
        self.health_status: dict[str, dict] = {}
        self.monitoring_tasks: dict[str, asyncio.Task] = {}

        # Load health monitoring configuration
        health_config = getattr(config, "exchange_health", {})
        if hasattr(health_config, "__dict__"):
            # Extract values safely
            self.health_check_interval = getattr(health_config, "health_check_interval_seconds", 30)
            self.unhealthy_threshold = getattr(health_config, "unhealthy_threshold", 3)
            self.recovery_check_interval = getattr(health_config, "recovery_check_interval_seconds", 60)
            self.timeout_seconds = getattr(health_config, "timeout_seconds", 10)
        else:
            # Use defaults if not an object
            self.health_check_interval = 30
            self.unhealthy_threshold = 3
            self.recovery_check_interval = 60
            self.timeout_seconds = 10

    async def start_monitoring(self, exchange_name: str, exchange: BaseExchange) -> None:
        """Start health monitoring for an exchange."""
        if exchange_name in self.monitoring_tasks:
            return

        task = asyncio.create_task(self._monitor_exchange_health(exchange_name, exchange))
        self.monitoring_tasks[exchange_name] = task

    async def stop_monitoring(self, exchange_name: str) -> None:
        """Stop health monitoring for an exchange."""
        if exchange_name in self.monitoring_tasks:
            task = self.monitoring_tasks[exchange_name]
            task.cancel()
            del self.monitoring_tasks[exchange_name]

        if exchange_name in self.health_status:
            del self.health_status[exchange_name]

    async def _monitor_exchange_health(self, exchange_name: str, exchange: BaseExchange) -> None:
        """Monitor exchange health continuously."""
        failure_count = 0

        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Perform health check
                is_healthy = await exchange.health_check()

                if is_healthy:
                    failure_count = 0
                    self.health_status[exchange_name] = {
                        "status": "healthy",
                        "last_check": datetime.now(),
                        "consecutive_failures": 0,
                    }
                else:
                    failure_count += 1
                    self.health_status[exchange_name] = {
                        "status": (
                            "unhealthy" if failure_count >= self.unhealthy_threshold else "degraded"
                        ),
                        "last_check": datetime.now(),
                        "consecutive_failures": failure_count,
                    }

            except asyncio.CancelledError:
                break
            except Exception as e:
                failure_count += 1
                self.health_status[exchange_name] = {
                    "status": "error",
                    "last_check": datetime.now(),
                    "consecutive_failures": failure_count,
                    "error": str(e),
                }

    def get_health_status(self, exchange_name: str) -> dict:
        """Get health status for an exchange."""
        return self.health_status.get(exchange_name, {"status": "unknown"})

    def get_all_health_status(self) -> dict[str, dict]:
        """Get health status for all monitored exchanges."""
        return self.health_status.copy()


class ExchangeFactory(BaseComponent):
    """
    Enhanced Factory for creating and managing exchange instances with dependency injection.

    This class provides a centralized way to create exchange instances from configuration,
    with support for connection pooling, hot-swapping capabilities, service registration,
    health monitoring, and dependency injection.
    """

    def __init__(self, config: Config):
        """
        Initialize the enhanced exchange factory with dependency injection.

        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.error_handler = ErrorHandler(config.error_handling)

        # Dependency injection container
        self.container = ServiceContainer()

        # Register core services
        self._register_core_services()

        # Registry of supported exchanges
        self._exchange_registry: dict[str, type[BaseExchange]] = {}
        self._exchange_configurations: dict[str, dict] = {}

        # Active exchange instances with metadata
        self._active_exchanges: dict[str, BaseExchange] = {}
        self._exchange_metadata: dict[str, dict] = {}

        # Connection pool for each exchange
        self._connection_pools: dict[str, list[BaseExchange]] = {}
        self._pool_configurations: dict[str, dict] = {}

        # Health monitoring
        self.health_monitor = HealthMonitor(config)

        # Performance metrics
        self._performance_metrics: dict[str, dict] = {}
        self._last_metrics_update: datetime | None = None

        self.initialize()
        self.logger.info("Initialized enhanced exchange factory with dependency injection")

    def _register_core_services(self) -> None:
        """Register core services in the dependency injection container."""
        # Register configuration
        self.container.register_instance("config", self.config)

        # Register error handler
        self.container.register_instance("error_handler", self.error_handler)

        # Register health monitor factory
        self.container.register_singleton(
            "health_monitor", lambda config: HealthMonitor(config), ["config"]
        )
        
        # Register state service and trade lifecycle manager as None initially
        # They will be set later via register_state_services method
        self.container.register_instance("state_service", None)
        self.container.register_instance("trade_lifecycle_manager", None)

    def register_exchange(
        self,
        exchange_name: str,
        exchange_class: type[BaseExchange],
        configuration: dict | None = None,
        pool_config: dict | None = None,
    ) -> None:
        """
        Register an exchange implementation with optional configuration.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance')
            exchange_class: Exchange class that inherits from BaseExchange
            configuration: Optional exchange-specific configuration
            pool_config: Optional connection pool configuration
        """
        if not issubclass(exchange_class, BaseExchange):
            raise ValidationError("Exchange class must inherit from BaseExchange")

        self._exchange_registry[exchange_name] = exchange_class
        self._exchange_configurations[exchange_name] = configuration or {}
        self._pool_configurations[exchange_name] = pool_config or {
            "min_connections": 1,
            "max_connections": 5,
            "health_check_interval": 60,
        }

        # Register exchange factory in container with state services
        self.container.register_transient(
            f"exchange_{exchange_name}",
            lambda config, error_handler, state_service, trade_lifecycle_manager: exchange_class(
                config, exchange_name, state_service, trade_lifecycle_manager
            ),
            ["config", "error_handler", "state_service", "trade_lifecycle_manager"],
        )

        self.logger.info(f"Registered exchange with DI container: {exchange_name}")

    def register_state_services(
        self, state_service: Any | None = None, trade_lifecycle_manager: Any | None = None
    ) -> None:
        """
        Register state management services for exchange integration.
        
        Args:
            state_service: StateService instance for state persistence
            trade_lifecycle_manager: TradeLifecycleManager instance for trade tracking
        """
        # Validate and update container with state services
        if state_service is not None:
            # Validate state_service has required methods
            if not hasattr(state_service, 'set_state') or not hasattr(state_service, 'get_state'):
                raise ValueError(
                    "Invalid state_service: must have 'set_state' and 'get_state' methods"
                )
            self.container._services["state_service"] = state_service
            self.logger.info("Registered StateService with exchange factory")
        else:
            self.logger.warning(
                "No StateService registered - exchanges will use in-memory fallback"
            )
            
        if trade_lifecycle_manager is not None:
            # Validate trade_lifecycle_manager has required methods
            if not hasattr(trade_lifecycle_manager, 'update_trade_event'):
                raise ValueError(
                    "Invalid trade_lifecycle_manager: must have 'update_trade_event' method"
                )
            self.container._services["trade_lifecycle_manager"] = trade_lifecycle_manager
            self.logger.info("Registered TradeLifecycleManager with exchange factory")
        else:
            self.logger.warning(
                "No TradeLifecycleManager registered - trade lifecycle tracking disabled"
            )

    def get_supported_exchanges(self) -> list[str]:
        """
        Get list of supported exchanges.

        Returns:
            List[str]: List of supported exchange names
        """
        return list(self._exchange_registry.keys())

    def get_available_exchanges(self) -> list[str]:
        """
        Get list of available exchanges from configuration.

        Returns:
            List[str]: List of configured exchange names
        """
        available = []
        if hasattr(self.config, "exchanges") and self.config.exchanges:
            # Check which exchanges have API keys configured
            if self.config.exchanges.binance_api_key:
                available.append("binance")
            if self.config.exchanges.okx_api_key:
                available.append("okx")
            if self.config.exchanges.coinbase_api_key:
                available.append("coinbase")
        return available

    def is_exchange_supported(self, exchange_name: str) -> bool:
        """
        Check if an exchange is supported.

        Args:
            exchange_name: Name of the exchange to check

        Returns:
            bool: True if supported, False otherwise
        """
        return exchange_name in self._exchange_registry

    @retry(max_attempts=3, base_delay=1.0)
    @log_calls
    @time_execution
    async def create_exchange(self, exchange_name: str, use_container: bool = True) -> BaseExchange:
        """
        Create a new exchange instance with optional dependency injection.

        Args:
            exchange_name: Name of the exchange to create
            use_container: Whether to use dependency injection container

        Returns:
            BaseExchange: Exchange instance

        Raises:
            ValidationError: If exchange is not supported
            ExchangeError: If exchange creation fails
        """
        if not self.is_exchange_supported(exchange_name):
            raise ValidationError(
                f"Exchange '{exchange_name}' is not supported. "
                f"Supported exchanges: {self.get_supported_exchanges()}"
            )

        try:
            if use_container and self.container.has_service(f"exchange_{exchange_name}"):
                # Use dependency injection container
                exchange = self.container.resolve(f"exchange_{exchange_name}")
                self.logger.debug(f"Created exchange using DI container: {exchange_name}")
            else:
                # Fallback to direct instantiation with state services
                exchange_class = self._exchange_registry[exchange_name]
                state_service = self.container._services.get("state_service")
                trade_lifecycle_manager = self.container._services.get("trade_lifecycle_manager")
                exchange = exchange_class(
                    self.config, exchange_name, state_service, trade_lifecycle_manager
                )
                self.logger.debug(f"Created exchange using direct instantiation: {exchange_name}")

            # Connect to the exchange
            connected = await exchange.connect()
            if not connected:
                raise ExchangeError(f"Failed to connect to {exchange_name}")

            # Store exchange metadata
            self._exchange_metadata[exchange_name] = {
                "created_at": datetime.now(),
                "connection_method": "container" if use_container else "direct",
                "status": "connected",
            }

            # Start health monitoring
            await self.health_monitor.start_monitoring(exchange_name, exchange)

            self.logger.info(f"Created and connected to exchange: {exchange_name}")
            return exchange

        except Exception as e:
            # Update metadata on failure
            self._exchange_metadata[exchange_name] = {
                "created_at": datetime.now(),
                "status": "failed",
                "error": str(e),
            }
            self.logger.error(f"Failed to create exchange {exchange_name}: {e!s}")
            raise ExchangeError(f"Exchange creation failed: {e!s}")

    async def get_exchange(
        self, exchange_name: str, create_if_missing: bool = True, force_recreate: bool = False
    ) -> BaseExchange | None:
        """
        Get an existing exchange instance or create a new one with lazy loading.

        Args:
            exchange_name: Name of the exchange
            create_if_missing: Whether to create a new instance if not found
            force_recreate: Force recreation of existing instance

        Returns:
            Optional[BaseExchange]: Exchange instance or None if not found and
            create_if_missing=False
        """
        # Force recreation if requested
        if force_recreate and exchange_name in self._active_exchanges:
            await self.remove_exchange(exchange_name)

        # Check if we already have an active instance
        if exchange_name in self._active_exchanges:
            exchange = self._active_exchanges[exchange_name]

            # Enhanced health check with cached results
            health_status = self.health_monitor.get_health_status(exchange_name)
            if health_status.get("status") == "healthy" or await exchange.health_check():
                # Update last accessed time
                if exchange_name in self._exchange_metadata:
                    self._exchange_metadata[exchange_name]["last_accessed"] = datetime.now()
                return exchange
            else:
                self.logger.warning(
                    "Exchange failed health check, removing from active pool",
                    exchange=exchange_name,
                )
                await self.remove_exchange(exchange_name)

        # Create new instance if requested (lazy loading)
        if create_if_missing:
            try:
                exchange = await self.create_exchange(exchange_name)
                self._active_exchanges[exchange_name] = exchange

                # Update performance metrics
                await self._update_performance_metrics(exchange_name, "created")

                return exchange
            except Exception as e:
                msg = "Failed to get/create exchange"
                self.logger.error(msg, exchange=exchange_name, error=str(e))
                await self._update_performance_metrics(exchange_name, "failed")
                return None

        return None

    async def get_or_create_pool(
        self, exchange_name: str, pool_size: int = 3
    ) -> list[BaseExchange]:
        """
        Get or create a connection pool for an exchange.

        Args:
            exchange_name: Name of the exchange
            pool_size: Number of connections in the pool

        Returns:
            List[BaseExchange]: List of exchange instances in the pool
        """
        if exchange_name not in self._connection_pools:
            self._connection_pools[exchange_name] = []

        pool = self._connection_pools[exchange_name]

        # Create instances up to pool_size
        while len(pool) < pool_size:
            try:
                exchange = await self.create_exchange(exchange_name)
                pool.append(exchange)
                msg = f"Added instance to pool for {exchange_name}: {len(pool)}/{pool_size}"
                self.logger.debug(msg)
            except Exception as e:
                self.logger.error(f"Failed to create pool instance for {exchange_name}: {e}")
                break

        return pool

    async def get_pooled_exchange(self, exchange_name: str) -> BaseExchange | None:
        """
        Get an exchange instance from the connection pool.

        Args:
            exchange_name: Name of the exchange

        Returns:
            Optional[BaseExchange]: Available exchange instance from pool
        """
        if exchange_name not in self._connection_pools:
            # Create initial pool
            await self.get_or_create_pool(exchange_name, 3)

        pool = self._connection_pools[exchange_name]

        # Find a healthy instance in the pool
        for exchange in pool:
            health_status = self.health_monitor.get_health_status(exchange_name)
            if health_status.get("status") == "healthy" or await exchange.health_check():
                return exchange

        # If no healthy instances, try to create a new one
        try:
            exchange = await self.create_exchange(exchange_name)
            pool.append(exchange)
            return exchange
        except Exception as e:
            self.logger.error(f"Failed to create new pool instance for {exchange_name}: {e}")
            return None

    async def _update_performance_metrics(self, exchange_name: str, operation: str) -> None:
        """
        Update performance metrics for exchange operations.

        Args:
            exchange_name: Name of the exchange
            operation: Type of operation (created, failed, removed)
        """
        now = datetime.now()

        if exchange_name not in self._performance_metrics:
            self._performance_metrics[exchange_name] = {
                "total_created": 0,
                "total_failed": 0,
                "total_removed": 0,
                "average_creation_time": 0.0,
                "last_updated": now,
            }

        metrics = self._performance_metrics[exchange_name]

        if operation == "created":
            metrics["total_created"] += 1
        elif operation == "failed":
            metrics["total_failed"] += 1
        elif operation == "removed":
            metrics["total_removed"] += 1

        metrics["last_updated"] = now
        self._last_metrics_update = now

    async def remove_exchange(self, exchange_name: str, remove_from_pool: bool = False) -> bool:
        """
        Remove an exchange instance from the active pool and optionally from connection pool.

        Args:
            exchange_name: Name of the exchange to remove
            remove_from_pool: Whether to also remove from connection pool

        Returns:
            bool: True if removed successfully, False otherwise
        """
        success = True

        # Remove from active exchanges
        if exchange_name in self._active_exchanges:
            try:
                exchange = self._active_exchanges[exchange_name]
                await exchange.disconnect()
                del self._active_exchanges[exchange_name]

                # Stop health monitoring
                await self.health_monitor.stop_monitoring(exchange_name)

                # Update metadata
                if exchange_name in self._exchange_metadata:
                    self._exchange_metadata[exchange_name]["removed_at"] = datetime.now()
                    self._exchange_metadata[exchange_name]["status"] = "removed"

                # Update performance metrics
                await self._update_performance_metrics(exchange_name, "removed")

                self.logger.info("Removed exchange from active pool", exchange=exchange_name)
            except Exception as e:
                self.logger.error("Failed to remove exchange", exchange=exchange_name, error=str(e))
                # Remove from active exchanges even if disconnect fails
                del self._active_exchanges[exchange_name]
                success = False

        # Remove from connection pool if requested
        if remove_from_pool and exchange_name in self._connection_pools:
            try:
                pool = self._connection_pools[exchange_name]
                for exchange in pool:
                    try:
                        await exchange.disconnect()
                    except Exception as e:
                        self.logger.warning(f"Failed to disconnect pooled exchange: {e}")

                del self._connection_pools[exchange_name]
                self.logger.info(f"Removed connection pool for {exchange_name}")
            except Exception as e:
                self.logger.error(f"Failed to remove connection pool for {exchange_name}: {e}")
                success = False

        return success

    async def get_all_active_exchanges(self) -> dict[str, BaseExchange]:
        """
        Get all active exchange instances.

        Returns:
            Dict[str, BaseExchange]: Dictionary of active exchanges
        """
        return self._active_exchanges.copy()

    async def health_check_all(self, include_pools: bool = True) -> dict[str, dict]:
        """
        Perform comprehensive health check on all active exchanges and pools.

        Args:
            include_pools: Whether to include connection pools in health check

        Returns:
            Dict[str, dict]: Dictionary mapping exchange names to detailed health status
        """
        health_status = {}

        # Check active exchanges
        for exchange_name, exchange in self._active_exchanges.items():
            try:
                is_healthy = await exchange.health_check()
                cached_status = self.health_monitor.get_health_status(exchange_name)

                health_status[exchange_name] = {
                    "active_instance": {
                        "healthy": is_healthy,
                        "last_check": datetime.now(),
                        "cached_status": cached_status,
                    }
                }
            except Exception as e:
                self.logger.error(f"Health check failed for {exchange_name}: {e!s}")
                health_status[exchange_name] = {
                    "active_instance": {
                        "healthy": False,
                        "error": str(e),
                        "last_check": datetime.now(),
                    }
                }

        # Check connection pools if requested
        if include_pools:
            for exchange_name, pool in self._connection_pools.items():
                if exchange_name not in health_status:
                    health_status[exchange_name] = {}

                pool_health = []
                for i, exchange in enumerate(pool):
                    try:
                        is_healthy = await exchange.health_check()
                        pool_health.append({"index": i, "healthy": is_healthy})
                    except Exception as e:
                        pool_health.append({"index": i, "healthy": False, "error": str(e)})

                health_status[exchange_name]["connection_pool"] = {
                    "size": len(pool),
                    "instances": pool_health,
                    "healthy_count": sum(1 for h in pool_health if h["healthy"]),
                }

        return health_status

    async def disconnect_all(self, include_pools: bool = True) -> None:
        """
        Disconnect all active exchange instances and optionally connection pools.

        Args:
            include_pools: Whether to also disconnect connection pools
        """
        # Disconnect active exchanges
        for exchange_name in list(self._active_exchanges.keys()):
            await self.remove_exchange(exchange_name, remove_from_pool=include_pools)

        # Clear all performance metrics and metadata
        self._performance_metrics.clear()
        self._exchange_metadata.clear()

        # Clear service container
        self.container.clear()

        # Re-register core services
        self._register_core_services()

        self.logger.info("Disconnected all active exchanges and cleared resources")

    def get_performance_metrics(self, exchange_name: str | None = None) -> dict:
        """
        Get performance metrics for exchanges.

        Args:
            exchange_name: Optional specific exchange name, None for all

        Returns:
            dict: Performance metrics
        """
        if exchange_name:
            return self._performance_metrics.get(exchange_name, {})
        return self._performance_metrics.copy()

    def get_exchange_metadata(self, exchange_name: str | None = None) -> dict:
        """
        Get metadata for exchanges.

        Args:
            exchange_name: Optional specific exchange name, None for all

        Returns:
            dict: Exchange metadata
        """
        if exchange_name:
            return self._exchange_metadata.get(exchange_name, {})
        return self._exchange_metadata.copy()

    def register_service(
        self,
        service_name: str,
        factory: Callable,
        dependencies: list[str] | None = None,
        lifecycle: str = "transient",
    ) -> None:
        """
        Register a custom service in the dependency injection container.

        Args:
            service_name: Name of the service
            factory: Factory function to create the service
            dependencies: List of dependency service names
            lifecycle: Service lifecycle ('singleton', 'transient')
        """
        if lifecycle == "singleton":
            self.container.register_singleton(service_name, factory, dependencies)
        else:
            self.container.register_transient(service_name, factory, dependencies)

        self.logger.debug(f"Registered custom service: {service_name} ({lifecycle})")

    def get_service(self, service_name: str) -> Any:
        """
        Get a service from the dependency injection container.

        Args:
            service_name: Name of the service to resolve

        Returns:
            Any: The resolved service instance
        """
        return self.container.resolve(service_name)

    def get_exchange_info(self, exchange_name: str) -> ExchangeInfo | None:
        """
        Get information about a supported exchange.

        Args:
            exchange_name: Name of the exchange

        Returns:
            Optional[ExchangeInfo]: Exchange information or None if not found
        """
        if not self.is_exchange_supported(exchange_name):
            return None

        # Get rate limits from config
        rate_limits = self.config.exchanges.rate_limits.get(exchange_name, {})

        return ExchangeInfo(
            name=exchange_name,
            supported_symbols=[],  # Will be populated by actual exchange implementation
            rate_limits=rate_limits,
            features=["spot_trading"],  # Default features
            api_version="1.0",
        )

    def get_exchange_status(self, exchange_name: str) -> ExchangeStatus:
        """
        Get the status of an exchange.

        Args:
            exchange_name: Name of the exchange

        Returns:
            ExchangeStatus: Current status of the exchange
        """
        if exchange_name not in self._active_exchanges:
            return ExchangeStatus.OFFLINE

        exchange = self._active_exchanges[exchange_name]
        if exchange.connected:
            return ExchangeStatus.ONLINE
        else:
            return ExchangeStatus.OFFLINE

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()
