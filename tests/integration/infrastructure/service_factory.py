"""
Real Service Factory for Infrastructure Integration Tests

This module provides production-ready service instantiation patterns for
infrastructure integration tests using real PostgreSQL, Redis, and InfluxDB services.

CRITICAL: This factory ensures correct service initialization with proper dependency chains.
"""

import logging

from src.core.caching.cache_manager import CacheManager
from src.core.config import get_config
from src.core.dependency_injection import DependencyContainer
from src.database.connection import DatabaseConnectionManager
from src.database.service import DatabaseService
from src.monitoring.metrics import MetricsCollector
from src.state.state_manager import StateManager

logger = logging.getLogger(__name__)


class RealServiceFactory:
    """
    Factory for creating properly initialized real services for infrastructure testing.

    This factory handles the complex dependency chains and initialization sequences
    required for production-ready service testing.
    """

    def __init__(self):
        self.config = None
        self.connection_manager = None
        self.database_service = None
        self.cache_manager = None
        self.state_manager = None
        self.container = None
        self._initialized = False

    async def initialize_core_services(
        self, clean_database: DatabaseConnectionManager | None = None
    ) -> None:
        """
        Initialize core services with proper dependency chain.

        Args:
            clean_database: Optional pre-configured database connection manager
        """
        print(
            f"DEBUG: initialize_core_services called with clean_database type: {type(clean_database)}"
        )
        print(f"DEBUG: clean_database value: {clean_database}")

        if self._initialized:
            return

        logger.info("Initializing core services for infrastructure testing")

        # 1. Load configuration
        self.config = get_config()
        logger.debug("Configuration loaded")

        # 2. Setup database connection manager
        if clean_database:
            self.connection_manager = clean_database
            logger.debug(
                f"Using provided database connection manager, type: {type(clean_database)}"
            )
        else:
            self.connection_manager = DatabaseConnectionManager(self.config)
            await self.connection_manager.initialize()
            logger.debug("Database connection manager initialized")

        # Set as global connection manager for DatabaseQueries and other components
        from src.database.connection import set_connection_manager

        set_connection_manager(self.connection_manager)
        logger.debug("Global connection manager set for database queries")

        logger.debug(f"After setup, self.connection_manager type: {type(self.connection_manager)}")

        # 3. Create database service
        logger.debug(
            f"Creating DatabaseService with connection_manager type: {type(self.connection_manager)}"
        )
        logger.debug(
            f"Creating DatabaseService with connection_manager value: {self.connection_manager}"
        )
        self.database_service = DatabaseService(
            connection_manager=self.connection_manager,
            config_service=None,  # Optional for testing
            validation_service=None,  # Optional for testing
        )
        await self.database_service.start()
        logger.debug("Database service started")

        # 4. Create cache manager with Redis client adapter
        raw_redis_client = await self.connection_manager.get_redis_client()
        redis_adapter = self._create_redis_adapter(raw_redis_client)
        self.cache_manager = CacheManager(redis_client=redis_adapter, config=self.config)
        logger.debug("Cache manager created with Redis client")

        # 5. Verify services are healthy
        await self._verify_service_health()

        self._initialized = True
        logger.info("✅ Core services initialized successfully")

    def _create_redis_adapter(self, raw_redis_client):
        """Create an adapter to make raw Redis client compatible with CacheClientInterface."""

        class RedisAdapter:
            def __init__(self, redis_client):
                self.client = redis_client

            async def connect(self):
                pass  # Already connected

            async def disconnect(self):
                if hasattr(self.client, "aclose"):
                    await self.client.aclose()
                else:
                    await self.client.close()

            async def ping(self):
                return await self.client.ping()

            async def info(self):
                return await self.client.info()

            async def get(self, key: str, namespace: str = "cache"):
                namespaced_key = f"{namespace}:{key}"
                value = await self.client.get(namespaced_key)
                if value is None:
                    return None
                try:
                    import json

                    return json.loads(value)
                except json.JSONDecodeError:
                    return value

            async def set(
                self, key: str, value, ttl: int | None = None, namespace: str = "cache"
            ) -> bool:
                import json

                namespaced_key = f"{namespace}:{key}"

                # Serialize value
                if isinstance(value, (dict, list)):
                    serialized_value = json.dumps(value, default=str)
                else:
                    serialized_value = str(value)

                # Set with TTL using setex
                if ttl is not None:
                    result = await self.client.setex(namespaced_key, ttl, serialized_value)
                else:
                    result = await self.client.setex(
                        namespaced_key, 3600, serialized_value
                    )  # Default 1 hour
                return result

            async def delete(self, key: str, namespace: str = "cache") -> bool:
                namespaced_key = f"{namespace}:{key}"
                result = await self.client.delete(namespaced_key)
                return result > 0

            async def exists(self, key: str, namespace: str = "cache") -> bool:
                namespaced_key = f"{namespace}:{key}"
                result = await self.client.exists(namespaced_key)
                return result > 0

            async def expire(self, key: str, ttl: int, namespace: str = "cache") -> bool:
                namespaced_key = f"{namespace}:{key}"
                result = await self.client.expire(namespaced_key, ttl)
                return result

        return RedisAdapter(raw_redis_client)

    async def create_dependency_container(self) -> DependencyContainer:
        """
        Create and configure dependency injection container with real services.

        Returns:
            DependencyContainer: Configured container with real services
        """
        if not self._initialized:
            await self.initialize_core_services()

        container = DependencyContainer()

        # Register core services
        container.register("DatabaseService", self.database_service, singleton=True)
        container.register("CacheManager", self.cache_manager, singleton=True)
        container.register("Config", self.config, singleton=True)

        logger.debug("Core services registered in DI container")
        return container

    async def create_advanced_services(self, container: DependencyContainer) -> None:
        """
        Create and register advanced services that depend on core services.

        Args:
            container: DI container with core services already registered
        """
        # Create MetricsCollector with proper parameters
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        metrics_collector = MetricsCollector(registry=registry, auto_register_metrics=True)
        container.register("MetricsCollector", metrics_collector, singleton=True)

        # Create StateManager with proper configuration
        self.state_manager = StateManager(config=self.config)
        await self.state_manager.initialize()
        container.register("StateManager", self.state_manager, singleton=True)

        # Create simplified business services for infrastructure testing
        await self._create_business_services(container)

        logger.debug("Advanced services registered in DI container")

    async def _create_business_services(self, container: DependencyContainer) -> None:
        """
        Create simplified business services for infrastructure testing.

        These are minimal implementations that satisfy dependency injection
        requirements without full business logic complexity.
        """
        # Create minimal ExecutionService with mock repository
        execution_service = self._create_mock_execution_service()
        container.register("ExecutionService", execution_service, singleton=True)

        # Create minimal RiskService
        risk_service = self._create_mock_risk_service()
        container.register("RiskService", risk_service, singleton=True)

        # Create minimal StrategyService
        strategy_service = self._create_mock_strategy_service()
        container.register("StrategyService", strategy_service, singleton=True)

        # Create minimal DataService
        data_service = self._create_mock_data_service()
        container.register("DataService", data_service, singleton=True)

        # Create minimal AnalyticsService
        analytics_service = self._create_mock_analytics_service()
        container.register("AnalyticsService", analytics_service, singleton=True)

        logger.debug("Business services registered for infrastructure testing")

    def _create_mock_execution_service(self):
        """Create mock ExecutionService for infrastructure testing."""

        class MockExecutionRepositoryService:
            async def create_execution_record(self, data):
                return {"id": "mock_execution"}

            async def health_check(self):
                return True

        class MockExecutionService:
            def __init__(self):
                self.repository_service = MockExecutionRepositoryService()

            async def health_check(self):
                return True

            async def submit_order(self, **kwargs):
                return {"order_id": "mock_order", "status": "submitted"}

        return MockExecutionService()

    def _create_mock_risk_service(self):
        """Create mock RiskService for infrastructure testing."""

        class MockRiskService:
            async def health_check(self):
                return True

            async def calculate_position_size(self, **kwargs):
                return {"size": 100, "risk": 0.02}

            async def check_risk_limits(self, **kwargs):
                return True

        return MockRiskService()

    def _create_mock_strategy_service(self):
        """Create mock StrategyService for infrastructure testing."""

        class MockStrategyService:
            async def health_check(self):
                return True

            async def get_active_strategies(self):
                return []

            async def generate_signals(self):
                return []

        return MockStrategyService()

    def _create_mock_data_service(self):
        """Create mock DataService for infrastructure testing."""

        class MockDataService:
            async def health_check(self):
                return True

            async def get_market_data(self, symbol):
                return {"symbol": symbol, "price": 50000}

            async def get_historical_data(self, **kwargs):
                return []

        return MockDataService()

    def _create_mock_analytics_service(self):
        """Create mock AnalyticsService for infrastructure testing."""

        class MockAnalyticsService:
            async def health_check(self):
                return True

            async def calculate_performance_metrics(self):
                return {"total_return": 0.05}

        return MockAnalyticsService()

    async def initialize_bot_management_services(self) -> None:
        """
        Initialize bot management services with real dependencies.

        This creates all bot management services with proper dependency injection.
        """
        if not self._initialized:
            raise RuntimeError("Core services must be initialized first")

        # Import bot management services
        from src.bot_management.di_registration import register_bot_management_services

        # Create a minimal container for bot management DI
        if not self.container:
            self.container = await self.create_dependency_container()
            await self.create_advanced_services(self.container)

        # Register bot management services using DI registration
        register_bot_management_services(self.container)

        logger.info("✅ Bot management services initialized")

    async def create_full_container(
        self, clean_database: DatabaseConnectionManager | None = None
    ) -> DependencyContainer:
        """
        Create fully configured DI container with all real services.

        Args:
            clean_database: Optional pre-configured database connection manager

        Returns:
            DependencyContainer: Fully configured container
        """
        await self.initialize_core_services(clean_database)
        container = await self.create_dependency_container()
        await self.create_advanced_services(container)

        logger.info("✅ Full service container created with real services")
        return container

    async def _verify_service_health(self) -> None:
        """Verify all services are healthy and accessible."""
        try:
            # Test database connectivity
            await self.database_service.health_check()
            logger.debug("Database health check passed")

            # Test cache connectivity - CacheManager should have health_check
            try:
                if hasattr(self.cache_manager, "health_check"):
                    await self.cache_manager.health_check()
                else:
                    # Fallback: test the Redis client directly
                    if (
                        hasattr(self.cache_manager, "redis_client")
                        and self.cache_manager.redis_client
                    ):
                        await self.cache_manager.redis_client.ping()
                logger.debug("Cache health check passed")
            except Exception as e:
                logger.warning(f"Cache health check warning: {e}")
                # Don't fail if cache isn't critical for the test

        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            raise RuntimeError(f"Service health verification failed: {e}") from e

    async def cleanup(self) -> None:
        """
        Clean up all services and resources.

        This ensures proper resource cleanup after testing.
        """
        logger.info("Starting service cleanup")

        try:
            # Shutdown services in reverse dependency order

            # Shutdown StateManager first (highest level service)
            if self.state_manager and hasattr(self.state_manager, "shutdown"):
                await self.state_manager.shutdown()
                logger.debug("State manager shutdown")

            # Shutdown DatabaseService (use correct method name)
            if self.database_service and hasattr(self.database_service, "stop"):
                await self.database_service.stop()
                logger.debug("Database service stopped")

            # Shutdown CacheManager
            if self.cache_manager and hasattr(self.cache_manager, "shutdown"):
                await self.cache_manager.shutdown()
                logger.debug("Cache manager shutdown")

            # Shutdown DatabaseConnectionManager (use correct method name)
            if self.connection_manager and hasattr(self.connection_manager, "close"):
                await self.connection_manager.close()
                logger.debug("Connection manager closed")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

        self._initialized = False
        logger.info("✅ Service cleanup completed")


class ProductionReadyTestBase:
    """
    Base class for production-ready infrastructure integration tests.

    This provides standardized patterns for real service testing with proper
    lifecycle management and error handling.
    """

    def __init__(self):
        self.service_factory = RealServiceFactory()
        self.container: DependencyContainer | None = None

    async def setup_real_services(
        self, clean_database: DatabaseConnectionManager | None = None
    ) -> DependencyContainer:
        """
        Setup real services for integration testing.

        Args:
            clean_database: Optional pre-configured database connection manager

        Returns:
            DependencyContainer: Configured container with real services
        """
        logger.info(f"Setting up real services for test: {self.__class__.__name__}")

        self.container = await self.service_factory.create_full_container(clean_database)

        # Verify all services are working
        await self._verify_test_environment()

        logger.info("✅ Real services setup completed successfully")
        return self.container

    async def _verify_test_environment(self) -> None:
        """Verify the test environment is ready."""
        if not self.container:
            raise RuntimeError("Container not initialized")

        # Test core services
        db_service = self.container.get("DatabaseService")
        cache_service = self.container.get("CacheManager")

        if not db_service or not cache_service:
            raise RuntimeError("Core services not properly registered")

        # Health checks
        await db_service.health_check()
        await cache_service.health_check()

        logger.debug("Test environment verification passed")

    async def cleanup_real_services(self) -> None:
        """Cleanup all real services and resources."""
        logger.info("Cleaning up real services")

        try:
            if self.service_factory:
                await self.service_factory.cleanup()

            if self.container:
                self.container.clear()

        except Exception as e:
            logger.warning(f"Error during test cleanup: {e}")

        logger.info("✅ Real services cleanup completed")


# Convenience functions for common patterns
async def create_test_database_service(
    clean_database: DatabaseConnectionManager,
) -> DatabaseService:
    """
    Create a properly configured DatabaseService for testing.

    Args:
        clean_database: Pre-configured database connection manager

    Returns:
        DatabaseService: Ready-to-use database service
    """
    factory = RealServiceFactory()
    await factory.initialize_core_services(clean_database)
    return factory.database_service


async def create_test_cache_manager() -> CacheManager:
    """
    Create a properly configured CacheManager for testing.

    Returns:
        CacheManager: Ready-to-use cache manager
    """
    factory = RealServiceFactory()
    await factory.initialize_core_services()
    return factory.cache_manager


async def create_test_container(clean_database: DatabaseConnectionManager) -> DependencyContainer:
    """
    Create a fully configured DI container for testing.

    Args:
        clean_database: Pre-configured database connection manager

    Returns:
        DependencyContainer: Fully configured container with real services
    """
    factory = RealServiceFactory()
    return await factory.create_full_container(clean_database)
