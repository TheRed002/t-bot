"""
State Service Factory for dependency injection and service composition.

This module provides factory functions for creating StateService instances
with proper dependency injection, configuration, and component wiring.
"""

import asyncio
from typing import Any, ClassVar, Protocol, cast

from src.core.config.main import Config
from src.core.config.service import ConfigService
from src.database.service import DatabaseService
from src.monitoring import MetricsCollector

from .monitoring_integration import create_integrated_monitoring_service
from .state_service import StateService

# Import utilities through centralized import handler
from .utils_imports import ValidationService


class DatabaseServiceProtocol(Protocol):
    """Protocol defining the interface for database services."""

    async def start(self) -> None:
        """Start the database service."""
        ...

    async def stop(self) -> None:
        """Stop the database service."""
        ...

    async def create_entity(self, entity: Any) -> Any:
        """Create a new entity."""
        ...

    async def get_entity_by_id(self, model_class: type, entity_id: Any) -> Any | None:
        """Get entity by ID."""
        ...

    async def health_check(self) -> Any:
        """Perform health check."""
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        ...


class DatabaseServiceWrapper:
    """Wrapper to add compatibility for DatabaseService with StateService."""

    def __init__(self, database_service: DatabaseService):
        """Initialize wrapper with DatabaseService instance."""
        self._service = database_service
        self._initialized = False

    async def start(self) -> None:
        """Start the wrapped database service."""
        await self._service.start()
        self._initialized = True

    async def stop(self) -> None:
        """Stop the wrapped database service."""
        await self._service.stop()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped service."""
        return getattr(self._service, name)


# Create a type alias for backward compatibility
DatabaseServiceInterface = DatabaseService | DatabaseServiceProtocol | DatabaseServiceWrapper


class MockDatabaseService:
    """Mock database service implementation for testing."""

    def __init__(self):
        self._started = False
        self._entities = {}
        self._metrics = {"total_queries": 0, "successful_queries": 0, "failed_queries": 0}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Check if mock service is initialized."""
        return self._initialized

    async def start(self) -> None:
        """Start the mock database service."""
        self._started = True
        self._initialized = True

    async def stop(self) -> None:
        """Stop the mock database service."""
        self._started = False
        self._initialized = False
        self._entities.clear()

    async def create_entity(self, entity: Any) -> Any:
        """Create entity in mock storage."""
        self._metrics["total_queries"] += 1
        entity_id = getattr(entity, "id", str(len(self._entities)))
        self._entities[entity_id] = entity
        self._metrics["successful_queries"] += 1
        return entity

    async def get_entity_by_id(self, model_class: type, entity_id: Any) -> Any | None:
        """Get entity by ID from mock storage."""
        self._metrics["total_queries"] += 1
        entity = self._entities.get(entity_id)
        if entity:
            self._metrics["successful_queries"] += 1
        else:
            self._metrics["failed_queries"] += 1
        return entity

    async def health_check(self) -> dict[str, Any]:
        """Return mock health status."""
        return {
            "status": "healthy" if self._started else "unhealthy",
            "started": self._started,
            "entities_count": len(self._entities),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return mock metrics."""
        return self._metrics.copy()


class StateServiceFactory:
    """
    Factory for creating and configuring StateService instances.

    Provides dependency injection and proper component wiring for
    enterprise-grade state management with all required services.
    """

    @staticmethod
    async def create_state_service(
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        auto_start: bool = True,
    ) -> StateService:
        """
        Create a fully configured StateService with all dependencies.

        Args:
            config: Application configuration
            database_service: DatabaseService instance (optional)
            auto_start: Whether to automatically start the service

        Returns:
            Configured and optionally started StateService
        """
        # Create database service if not provided
        if database_service is None:
            database_service = await StateServiceFactory._create_database_service(config)

        # Create state service with dependency injection
        # Cast to expected type for StateService compatibility
        state_service = StateService(config, cast(DatabaseService, database_service))

        # Create integrated monitoring service
        metrics_collector = MetricsCollector()
        _ = create_integrated_monitoring_service(state_service, metrics_collector)

        # Note: monitoring_service integration is handled by the monitoring_integration module
        # The state_service doesn't need a direct reference as it uses the integrated patterns

        # Initialize if requested
        if auto_start:
            await state_service.initialize()

        return state_service

    @staticmethod
    async def _create_database_service(config: Config) -> DatabaseServiceInterface:
        """Create and configure DatabaseService using dependency injection."""
        # Create dependencies through proper factory methods
        config_service = await StateServiceFactory._create_config_service(config)
        validation_service = await StateServiceFactory._create_validation_service()

        # Create database service with injected dependencies
        database_service = DatabaseService(
            config_service=config_service, validation_service=validation_service
        )

        # Wrap the service to add compatibility properties
        wrapped_service = DatabaseServiceWrapper(database_service)

        # Start the service
        await wrapped_service.start()

        return wrapped_service

    @staticmethod
    async def _create_config_service(config: Config) -> ConfigService:
        """Create and configure ConfigService."""
        config_service = ConfigService()
        # Use the existing config object directly instead of load_config
        config_service._config = config
        return config_service

    @staticmethod
    async def _create_validation_service() -> ValidationService:
        """Create and configure ValidationService."""
        from src.core.dependency_injection import get_container

        container = get_container()
        validation_service = container.get("validation_service")
        if not validation_service:
            validation_service = ValidationService()
            container.register("validation_service", validation_service, singleton=True)
        await validation_service.initialize()
        return validation_service

    @staticmethod
    async def create_state_service_for_testing(
        config: Config | None = None, mock_database: bool = False
    ) -> StateService:
        """
        Create StateService configured for testing.

        Args:
            config: Test configuration (default provided if None)
            mock_database: Whether to use mock database service

        Returns:
            StateService configured for testing
        """
        # Use default test config if none provided
        if config is None:
            config = StateServiceFactory._create_test_config()

        # Create mock or real database service
        if mock_database:
            database_service = StateServiceFactory._create_mock_database_service()
        else:
            database_service = await StateServiceFactory._create_database_service(config)

        # Create state service (don't auto-start for testing)
        # Cast to expected type for StateService compatibility
        state_service = StateService(config, cast(DatabaseService, database_service))

        return state_service

    @staticmethod
    def _create_test_config() -> Config:
        """Create default configuration for testing."""
        # Create a test config without any arguments
        config = Config()

        # Set test-specific values directly
        test_config_dict = {
            "database": {"url": "sqlite:///:memory:"},
            "redis": {"url": "redis://localhost:6379/15"},  # Use test DB
            "influxdb": {"url": "http://localhost:8086", "database": "test"},
            "state_management": {
                "cache_ttl_seconds": 60,
                "max_change_log_size": 100,
                "sync_interval_seconds": 5,
                "enable_compression": False,
            },
        }
        config.__dict__.update(test_config_dict)

        return config

    @staticmethod
    def _create_mock_database_service() -> DatabaseServiceInterface:
        """Create mock database service for testing."""
        return MockDatabaseService()


class StateServiceRegistry:
    """
    Registry for managing StateService instances across the application.

    Provides singleton access and proper lifecycle management for
    StateService instances.
    """

    _instances: ClassVar[dict[str, StateService]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get_instance(
        cls,
        name: str = "default",
        config: Config | None = None,
        database_service: DatabaseServiceInterface | None = None,
    ) -> StateService:
        """
        Get or create a StateService instance.

        Args:
            name: Instance name (for multiple services)
            config: Configuration (required for new instances)
            database_service: Database service dependency

        Returns:
            StateService instance
        """
        async with cls._lock:
            if name not in cls._instances:
                if config is None:
                    raise ValueError(
                        f"Config required to create new StateService instance '{name}'"
                    )

                cls._instances[name] = await StateServiceFactory.create_state_service(
                    config=config, database_service=database_service, auto_start=True
                )

            return cls._instances[name]

    @classmethod
    async def register_instance(cls, name: str, instance: StateService) -> None:
        """
        Register a pre-configured StateService instance.

        Args:
            name: Instance name
            instance: StateService instance
        """
        async with cls._lock:
            cls._instances[name] = instance

    @classmethod
    async def remove_instance(cls, name: str) -> None:
        """
        Remove and cleanup a StateService instance.

        Args:
            name: Instance name
        """
        async with cls._lock:
            if name in cls._instances:
                instance = cls._instances.pop(name)
                await instance.cleanup()

    @classmethod
    async def cleanup_all(cls) -> None:
        """Cleanup all registered StateService instances."""
        async with cls._lock:
            for instance in cls._instances.values():
                await instance.cleanup()
            cls._instances.clear()

    @classmethod
    def list_instances(cls) -> list[str]:
        """Get list of registered instance names."""
        return list(cls._instances.keys())

    @classmethod
    async def get_health_status(cls) -> dict[str, dict]:
        """Get health status of all registered instances."""
        status = {}
        for name, instance in cls._instances.items():
            try:
                instance_status = await instance.get_health_status()
                status[name] = instance_status
            except Exception as e:
                status[name] = {"overall_status": "unhealthy", "error": str(e)}
        return status


# Convenience functions for common patterns


async def create_default_state_service(config: Config) -> StateService:
    """Create default StateService with standard configuration."""
    return await StateServiceFactory.create_state_service(config)


async def get_state_service(name: str = "default") -> StateService:
    """Get StateService instance from registry."""
    return await StateServiceRegistry.get_instance(name)


async def create_test_state_service() -> StateService:
    """Create StateService configured for testing."""
    return await StateServiceFactory.create_state_service_for_testing()
