"""State services dependency injection registration."""

from typing import TYPE_CHECKING, Any

from src.core.dependency_injection import DependencyContainer
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
            # Use dependency injection to resolve config
            try:
                config = container.get("Config")
            except (DependencyError, ServiceError):
                config = None
            return StateBusinessService(config=config)

        container.register("StateBusinessService", state_business_service_factory, singleton=True)

        # Register StatePersistenceService using factory pattern with dependency injection
        def state_persistence_service_factory() -> "StatePersistenceService":
            from .services.state_persistence_service import StatePersistenceService
            # Use dependency injection to resolve database service
            try:
                database_service = container.get("DatabaseService")
            except (DependencyError, ServiceError):
                database_service = None
            return StatePersistenceService(database_service=database_service)

        container.register("StatePersistenceService", state_persistence_service_factory, singleton=True)

        # Register StateValidationService using factory pattern with dependency injection
        def state_validation_service_factory() -> "StateValidationService":
            from .services.state_validation_service import StateValidationService
            # Use dependency injection to resolve validation service
            try:
                validation_service = container.get("ValidationService")
            except (DependencyError, ServiceError):
                validation_service = None
            return StateValidationService(validation_service=validation_service)

        container.register("StateValidationService", state_validation_service_factory, singleton=True)

        # Register StateSynchronizationService using factory pattern with dependency injection
        def state_synchronization_service_factory() -> "StateSynchronizationService":
            from .services.state_synchronization_service import StateSynchronizationService
            # Use dependency injection to resolve event service
            try:
                event_service = container.get("EventService")
            except (DependencyError, ServiceError):
                event_service = None
            return StateSynchronizationService(event_service=event_service)

        container.register("StateSynchronizationService", state_synchronization_service_factory, singleton=True)

        # Register StateService using comprehensive factory pattern with dependency injection
        def state_service_factory() -> "StateService":
            from .state_service import StateService
            
            # Use dependency injection to resolve all dependencies
            try:
                config = container.get("Config")
            except (DependencyError, ServiceError):
                config = None
                
            try:
                database_service = container.get("DatabaseService") 
            except (DependencyError, ServiceError):
                database_service = None
            
            # Create StateService using factory pattern with injected dependencies
            state_service = StateService(config=config, database_service=database_service)
            
            # Use factory pattern to inject service layer dependencies
            try:
                state_service._business_service = container.get("StateBusinessService")
            except (DependencyError, ServiceError):
                pass
                
            try:
                state_service._persistence_service = container.get("StatePersistenceService")
            except (DependencyError, ServiceError):
                pass
                
            try:
                state_service._validation_service = container.get("StateValidationService")
            except (DependencyError, ServiceError):
                pass
                
            try:
                state_service._synchronization_service = container.get("StateSynchronizationService")
            except (DependencyError, ServiceError):
                pass
            
            return state_service

        container.register("StateService", state_service_factory, singleton=True)

        # Register factory using proper dependency injection pattern
        def state_service_factory_factory() -> "StateServiceFactoryInterface":
            from .factory import StateServiceFactory
            from src.core.dependency_injection import DependencyInjector
            # Create injector instance for factory
            injector = DependencyInjector()
            injector._container = container  # Link to current container
            return StateServiceFactory(injector=injector)

        container.register("StateServiceFactory", state_service_factory_factory, singleton=True)
        container.register("StateServiceFactoryInterface", 
                          lambda: container.get("StateServiceFactory"), singleton=True)

        # Register interface bindings
        container.register("StateBusinessServiceInterface", 
                          lambda: container.get("StateBusinessService"), singleton=True)
        container.register("StatePersistenceServiceInterface",
                          lambda: container.get("StatePersistenceService"), singleton=True)
        container.register("StateValidationServiceInterface", 
                          lambda: container.get("StateValidationService"), singleton=True)
        container.register("StateSynchronizationServiceInterface",
                          lambda: container.get("StateSynchronizationService"), singleton=True)

        logger.info("State services registered with DI container")

    except Exception as e:
        logger.error(f"Failed to register state services: {e}")
        raise


def create_state_service_with_dependencies(config: "Config", database_service: "DatabaseService") -> "StateService":
    """
    Create StateService with proper dependency injection using factory pattern.
    
    Args:
        config: Application configuration
        database_service: Database service instance
        
    Returns:
        Configured StateService with all dependencies injected
    """
    from src.core.dependency_injection import get_container
    
    try:
        # Use factory pattern through DI container
        container = get_container()
        factory = container.get("StateServiceFactory")
        
        # Register dependencies in container using factory pattern
        container.register("Config", lambda: config, singleton=True)
        container.register("DatabaseService", lambda: database_service, singleton=True)
        
        # Create service using factory with dependency injection
        return factory.create_state_service(config=config, database_service=database_service, auto_start=False)
        
    except (DependencyError, ServiceError):
        # Use factory pattern for fallback creation with dependency injection
        from .services.state_business_service import StateBusinessService
        from .services.state_persistence_service import StatePersistenceService
        from .services.state_synchronization_service import StateSynchronizationService
        from .services.state_validation_service import StateValidationService
        from .state_service import StateService
        
        # Create service dependencies using factory pattern
        business_service = StateBusinessService(config=config)
        persistence_service = StatePersistenceService(database_service=database_service)
        
        # Use dependency injection factory to get validation service
        try:
            container = get_container()
            validation_service_dep = container.get("ValidationService")
        except (DependencyError, ServiceError):
            validation_service_dep = None
            
        validation_service = StateValidationService(validation_service=validation_service_dep)
        
        # Use dependency injection factory to get event service
        try:
            container = get_container()
            event_service = container.get("EventService")
        except (DependencyError, ServiceError):
            event_service = None
            
        synchronization_service = StateSynchronizationService(event_service=event_service)
        
        # Create StateService using factory pattern
        from typing import cast
        from .state_service import DatabaseServiceProtocol
        state_service = StateService(config=config, database_service=cast(DatabaseServiceProtocol, database_service))
        
        # Use factory pattern to inject dependencies
        state_service._business_service = business_service
        state_service._persistence_service = persistence_service
        state_service._validation_service = validation_service
        state_service._synchronization_service = synchronization_service
        
        return state_service