"""Bot Management services dependency injection registration - Simplified."""

from typing import TYPE_CHECKING, Any

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from .service import BotService

logger = get_logger(__name__)


def register_bot_management_services(injector: DependencyInjector) -> None:
    """
    Register bot management services with proper service layer architecture.

    Args:
        injector: Dependency injector instance
    """
    try:
        # Register service interfaces and implementations
        from .instance_service import BotInstanceService
        from .lifecycle_service import BotLifecycleService
        from .coordination_service import BotCoordinationService
        from .monitoring_service import BotMonitoringService
        from .resource_service import BotResourceService
        from .controller import BotManagementController
        from .interfaces import (
            IBotInstanceService,
            IBotLifecycleService,
            IBotCoordinationService,
            IBotMonitoringService,
            IResourceManagementService,
        )

        # Register service implementations
        injector.register_singleton(IBotInstanceService, BotInstanceService)
        injector.register_singleton(IBotLifecycleService, BotLifecycleService)
        injector.register_singleton(IBotCoordinationService, BotCoordinationService)
        injector.register_singleton(IBotMonitoringService, BotMonitoringService)
        injector.register_singleton(IResourceManagementService, BotResourceService)

        # Register concrete implementations
        injector.register_singleton("BotInstanceService", BotInstanceService)
        injector.register_singleton("BotLifecycleService", BotLifecycleService)
        injector.register_singleton("BotCoordinationService", BotCoordinationService)
        injector.register_singleton("BotMonitoringService", BotMonitoringService)
        injector.register_singleton("BotResourceService", BotResourceService)

        # Register controller
        injector.register_singleton("BotManagementController", BotManagementController)

        logger.info("Bot management services registered successfully")

    except Exception as e:
        logger.error(f"Failed to register bot management services: {e}")
        raise


def configure_bot_management_dependencies(
    injector: DependencyInjector | None = None,
) -> DependencyInjector:
    """
    Configure bot management dependencies with proper registration.

    Args:
        injector: Optional dependency injector instance

    Returns:
        Configured dependency injector
    """
    if injector is None:
        injector = DependencyInjector()

    # Register core bot management services
    register_bot_management_services(injector)

    logger.info("Bot management dependencies configured")
    return injector


def get_bot_service(injector: DependencyInjector) -> "BotService":
    """Get BotService from DI container."""
    return injector.resolve("BotService")


def get_bot_instance_service(injector: DependencyInjector) -> 'IBotInstanceService':
    """Get bot instance service from injector."""
    from .interfaces import IBotInstanceService
    return injector.resolve(IBotInstanceService)


def get_bot_lifecycle_service(injector: DependencyInjector) -> 'IBotLifecycleService':
    """Get bot lifecycle service from injector."""
    from .interfaces import IBotLifecycleService
    return injector.resolve(IBotLifecycleService)


def get_bot_coordination_service(injector: DependencyInjector) -> 'IBotCoordinationService':
    """Get bot coordination service from injector."""
    from .interfaces import IBotCoordinationService
    return injector.resolve(IBotCoordinationService)


def get_bot_monitoring_service(injector: DependencyInjector) -> 'IBotMonitoringService':
    """Get bot monitoring service from injector."""
    from .interfaces import IBotMonitoringService
    return injector.resolve(IBotMonitoringService)


def get_bot_resource_service(injector: DependencyInjector) -> 'IResourceManagementService':
    """Get bot resource service from injector."""
    from .interfaces import IResourceManagementService
    return injector.resolve(IResourceManagementService)


def get_bot_management_controller(injector: DependencyInjector) -> 'BotManagementController':
    """Get bot management controller from injector."""
    from .controller import BotManagementController
    return injector.resolve(BotManagementController)


def initialize_bot_management_services(injector: DependencyInjector) -> dict[str, Any]:
    """
    Initialize simplified bot management services.

    Args:
        injector: Dependency injector instance

    Returns:
        Dictionary of initialized services
    """
    services = {}

    try:
        services["bot_service"] = injector.resolve("BotService")
        services["bot_management_factory"] = injector.resolve("BotManagementFactory")

        logger.info("Bot management services initialized successfully")
        return services

    except Exception as e:
        logger.error(f"Failed to initialize bot management services: {e}")
        raise
