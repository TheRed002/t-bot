"""State services dependency injection registration."""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyContainer, DependencyInjector
from src.core.exceptions import DependencyError, ServiceError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.core.config.main import Config
    from src.database.service import DatabaseService

    from .interfaces import StateServiceFactoryInterface
    from .services.state_business_service import StateBusinessService
    from .services.state_persistence_service import StatePersistenceService
    from .services.state_synchronization_service import StateSynchronizationService
    from .services.state_validation_service import StateValidationService
    from .state_service import StateService

logger = get_logger(__name__)


def register_state_services(container: DependencyContainer) -> None:
    """
    Register state services with the dependency injection container.

    Args:
        container: DI container instance
    """
    try:
        # Register StateBusinessService using factory pattern with dependency injection
        def state_business_service_factory() -> "StateBusinessService":
            from .services.state_business_service import StateBusinessService

            # Use dependency injection to get config
            config = None
            try:
                config = container.get("Config")
            except (DependencyError, ServiceError):
                pass

            return StateBusinessService(config=config)

        container.register("StateBusinessService", state_business_service_factory, singleton=True)

        # Register StatePersistenceService using factory pattern with dependency injection
        def state_persistence_service_factory() -> "StatePersistenceService":
            from .services.state_persistence_service import StatePersistenceService

            # Use dependency injection to get database service
            database_service = None
            try:
                database_service = container.get("DatabaseService")
            except (DependencyError, ServiceError):
                pass

            return StatePersistenceService(database_service=database_service)

        container.register("StatePersistenceService", state_persistence_service_factory, singleton=True)

        # Register StateValidationService using factory pattern with dependency injection
        def state_validation_service_factory() -> "StateValidationService":
            from .services.state_validation_service import StateValidationService

            # Use dependency injection to get validation service
            validation_service = None
            try:
                validation_service = container.get("ValidationService")
            except (DependencyError, ServiceError):
                pass

            return StateValidationService(validation_service=validation_service)

        container.register("StateValidationService", state_validation_service_factory, singleton=True)

        # Register StateSynchronizationService using factory pattern with dependency injection
        def state_synchronization_service_factory() -> "StateSynchronizationService":
            from .services.state_synchronization_service import StateSynchronizationService

            # Use dependency injection to get event service
            event_service = None
            try:
                event_service = container.get("EventService")
            except (DependencyError, ServiceError):
                pass

            return StateSynchronizationService(event_service=event_service)

        container.register("StateSynchronizationService", state_synchronization_service_factory, singleton=True)

        # Register StateService using comprehensive factory pattern with dependency injection
        def state_service_factory() -> "StateService":
            from .state_service import StateService

            # Use dependency injection to resolve all dependencies
            config = None
            try:
                config = container.get("Config")
            except (DependencyError, ServiceError):
                pass

            business_service = None
            try:
                business_service = container.get("StateBusinessService")
            except (DependencyError, ServiceError):
                pass

            persistence_service = None
            try:
                persistence_service = container.get("StatePersistenceService")
            except (DependencyError, ServiceError):
                pass

            validation_service = None
            try:
                validation_service = container.get("StateValidationService")
            except (DependencyError, ServiceError):
                pass

            synchronization_service = None
            try:
                synchronization_service = container.get("StateSynchronizationService")
            except (DependencyError, ServiceError):
                pass

            # Create StateService using factory pattern with injected dependencies
            return StateService(
                config=config,
                business_service=business_service,
                persistence_service=persistence_service,
                validation_service=validation_service,
                synchronization_service=synchronization_service
            )

        container.register("StateService", state_service_factory, singleton=True)

        # Register factory using proper dependency injection pattern
        def state_service_factory_factory() -> "StateServiceFactoryInterface":
            from src.core.dependency_injection import DependencyInjector

            from .factory import StateServiceFactory

            # Create injector instance for factory
            injector = DependencyInjector()
            injector._container = container  # Link to current container
            return StateServiceFactory(injector=injector)

        container.register("StateServiceFactory", state_service_factory_factory, singleton=True)
        container.register(
            "StateServiceFactoryInterface",
            lambda: container.get("StateServiceFactory"),
            singleton=True
        )

        # Register interface bindings
        container.register(
            "StateBusinessServiceInterface",
            lambda: container.get("StateBusinessService"),
            singleton=True
        )
        container.register(
            "StatePersistenceServiceInterface",
            lambda: container.get("StatePersistenceService"),
            singleton=True
        )
        container.register(
            "StateValidationServiceInterface",
            lambda: container.get("StateValidationService"),
            singleton=True
        )
        container.register(
            "StateSynchronizationServiceInterface",
            lambda: container.get("StateSynchronizationService"),
            singleton=True
        )

        logger.info("State services registered with DI container")

    except Exception as e:
        logger.error(f"Failed to register state services: {e}")
        raise


async def create_state_service_with_dependencies(
    config: "Config", database_service: "DatabaseService", injector: DependencyInjector | None = None
) -> "StateService":
    """
    Create StateService with proper dependency injection using factory pattern.

    Args:
        config: Application configuration
        database_service: Database service instance
        injector: Optional dependency injector

    Returns:
        Configured StateService with all dependencies injected
    """
    if injector is None:
        from src.core.dependency_injection import DependencyInjector

        injector = DependencyInjector()
        register_state_services(injector.get_container())

    # Register dependencies in container using factory pattern
    container = injector.get_container()
    container.register("Config", lambda: config, singleton=True)
    container.register("DatabaseService", lambda: database_service, singleton=True)

    # Create service using factory with dependency injection
    from .factory import StateServiceFactory

    factory = StateServiceFactory(injector=injector)
    return await factory.create_state_service(
        config=config, database_service=database_service, auto_start=False
    )
