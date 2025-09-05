"""
State Service Factory for dependency injection and service composition.

This module provides factory functions for creating StateService instances
with proper dependency injection, configuration, and component wiring.
"""

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Union, cast

from src.core.config.main import Config
from src.core.config.service import ConfigService
from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import DependencyError, ServiceError
from src.database.service import DatabaseService
from src.monitoring import MetricsCollector

from .interfaces import StateServiceFactoryInterface
from .monitoring_integration import create_integrated_monitoring_service

# Import utilities through centralized import handler
from .utils_imports import ValidationService

if TYPE_CHECKING:
    from .state_service import StateService


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

    async def get_entity_by_id(self, model_class: type, entity_id: Any) -> Union[Any, None]:
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
DatabaseServiceInterface = Union[DatabaseService, DatabaseServiceProtocol, DatabaseServiceWrapper]


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

    async def get_entity_by_id(self, model_class: type, entity_id: Any) -> Union[Any, None]:
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


class StateServiceFactory(StateServiceFactoryInterface):
    """
    Factory for creating and configuring StateService instances.

    Provides dependency injection and proper component wiring for
    enterprise-grade state management with all required services.
    """

    def __init__(self, injector: Union[DependencyInjector, None] = None):
        """
        Initialize factory with dependency injector.
        
        Args:
            injector: Optional dependency injector instance
        """
        from src.core.dependency_injection import get_container
        from .di_registration import register_state_services
        
        # Use provided injector or get global container
        if injector is not None:
            self.injector = injector
            self.container = injector.get_container()
        else:
            from src.core.dependency_injection import injector as global_injector
            self.injector = global_injector
            self.container = get_container()
        
        # Ensure state services are registered
        try:
            register_state_services(self.container)
        except (DependencyError, ServiceError):
            # Services may already be registered
            pass

    async def create_state_service(
        self,
        config: Config,
        database_service: Union[DatabaseServiceInterface, None] = None,
        auto_start: bool = True,
    ) -> "StateService":
        """
        Create a fully configured StateService with all dependencies.

        Args:
            config: Application configuration
            database_service: DatabaseService instance (optional)
            auto_start: Whether to automatically start the service

        Returns:
            Configured and optionally started StateService
        """
        # Register dependencies in container using proper factory patterns
        self.container.register("Config", lambda: config, singleton=True)
        
        if database_service:
            self.container.register("DatabaseService", lambda: database_service, singleton=True)
        else:
            # Create database service using dependency injection factory
            database_service = await self._create_database_service_with_di(config)
            self.container.register("DatabaseService", lambda: database_service, singleton=True)

        # Use factory pattern through DI container
        state_service = self.container.get("StateService")

        # Create integrated monitoring service
        metrics_collector = MetricsCollector()
        _ = create_integrated_monitoring_service(state_service, metrics_collector)


        # Initialize if requested
        if auto_start:
            await state_service.initialize()

        return state_service

    async def _create_database_service_with_di(self, config: Config) -> DatabaseServiceInterface:
        """Create and configure DatabaseService using dependency injection factory pattern."""
        # Register config in container for factory usage
        self.container.register("Config", lambda: config, singleton=True)
        
        # Use dependency injection factory to create config service
        config_service = await self._create_config_service_factory()
        validation_service = await self._create_validation_service_factory()
        
        # Register services using factory pattern
        self.container.register("ConfigService", lambda: config_service, singleton=True)
        if validation_service:
            self.container.register("ValidationService", lambda: validation_service, singleton=True)

        # Create database service using dependency injection factory
        try:
            database_service = self.container.get("DatabaseService")
        except (DependencyError, ServiceError):
            # Use factory pattern for fallback creation
            database_service = await self._create_database_service_factory(config_service, validation_service)

        # Wrap the service using factory pattern
        wrapped_service = DatabaseServiceWrapper(database_service)

        # Start the service
        await wrapped_service.start()

        return wrapped_service

    async def _create_config_service_factory(self) -> ConfigService:
        """Create and configure ConfigService using factory pattern."""
        # Use dependency injection to resolve config
        try:
            config = self.container.get("Config")
        except (DependencyError, ServiceError):
            # Create default config if not available
            config = Config()
        
        config_service = ConfigService()
        config_service._config = config
        return config_service

    async def _create_validation_service_factory(self) -> Union[ValidationService, None]:
        """Create and configure ValidationService using factory pattern."""
        # Try to resolve from container using factory pattern
        try:
            validation_service = self.container.get("ValidationService")
            return validation_service
        except (DependencyError, ServiceError):
            # Use factory pattern to check if ValidationFramework is available
            try:
                # Check if we can create ValidationService through DI
                from src.utils.validators import ValidationFramework
                framework = ValidationFramework()
                validation_service = ValidationService(framework)
                return validation_service
            except ImportError:
                # ValidationService dependencies not available
                return None

    async def create_state_service_for_testing(
        self,
        config: Union[Config, None] = None, mock_database: bool = False
    ) -> "StateService":
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
            config = self._create_test_config()

        # Create mock or real database service
        if mock_database:
            database_service = self._create_mock_database_service()
        else:
            database_service = await self._create_database_service_with_di(config)

        # Create state service for testing
        from .state_service import DatabaseServiceProtocol, StateService
        
        # Register test dependencies in container
        self.container.register("Config", lambda: config, singleton=True)
        self.container.register("DatabaseService", lambda: database_service, singleton=True)
        
        # Use dependency injection factory to create service
        try:
            state_service = self.container.get("StateService")
        except (DependencyError, ServiceError):
            # Fallback to direct creation with proper casting
            state_service = StateService(config, cast(DatabaseServiceProtocol, database_service))

        return state_service

    def _create_test_config(self) -> Config:
        """Create default configuration for testing."""
        # Create a test config with default values
        config = Config()
        return config

    def _create_mock_database_service(self) -> DatabaseServiceInterface:
        """Create mock database service using factory pattern."""
        # Use factory method to create mock service
        return MockDatabaseService()
        
    async def _create_database_service_factory(self, config_service: ConfigService, validation_service: Union[ValidationService, None]) -> DatabaseService:
        """Factory method to create DatabaseService with injected dependencies."""
        return DatabaseService(
            config_service=config_service, 
            validation_service=validation_service
        )


class StateServiceRegistry:
    """
    Registry for managing StateService instances across the application.

    Provides singleton access and proper lifecycle management for
    StateService instances using dependency injection.
    """

    _factory: Union[StateServiceFactory, None] = None

    _instances: ClassVar[dict[str, "StateService"]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get_instance(
        cls,
        name: str = "default",
        config: Union[Config, None] = None,
        database_service: Union[DatabaseServiceInterface, None] = None,
    ) -> "StateService":
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

                if cls._factory is None:
                    # Use dependency injection to create factory
                    from src.core.dependency_injection import get_container
                    container = get_container()
                    try:
                        cls._factory = container.get("StateServiceFactory")
                    except (DependencyError, ServiceError):
                        # Fallback to direct factory creation
                        cls._factory = StateServiceFactory()
                    
                cls._instances[name] = await cls._factory.create_state_service(
                    config=config, database_service=database_service, auto_start=True
                )

            return cls._instances[name]

    @classmethod
    async def register_instance(cls, name: str, instance: "StateService") -> None:
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


async def create_default_state_service(config: Config) -> "StateService":
    """Create default StateService using factory pattern with dependency injection."""
    from src.core.dependency_injection import get_container
    
    # Use dependency injection container to get factory
    container = get_container()
    try:
        factory = container.get("StateServiceFactory")
    except (DependencyError, ServiceError):
        # Fallback to direct factory creation
        factory = StateServiceFactory()
    
    return await factory.create_state_service(config)


async def get_state_service(name: str = "default") -> "StateService":
    """Get StateService instance from registry."""
    return await StateServiceRegistry.get_instance(name)


async def create_test_state_service() -> "StateService":
    """Create StateService configured for testing using factory pattern."""
    from src.core.dependency_injection import get_container
    
    # Use dependency injection container to get factory
    container = get_container()
    try:
        factory = container.get("StateServiceFactory")
    except (DependencyError, ServiceError):
        # Fallback to direct factory creation
        factory = StateServiceFactory()
    
    return await factory.create_state_service_for_testing()
