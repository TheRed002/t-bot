"""Bot Management services dependency injection registration - Simplified."""

from typing import TYPE_CHECKING, Any

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from .service import BotService

logger = get_logger(__name__)


def register_bot_management_services(injector: DependencyInjector) -> None:
    """
    Register simplified bot management services.

    Args:
        injector: Dependency injector instance
    """

    # Register repository factories
    def bot_repository_factory():
        """Factory for BotRepository with proper session injection."""
        from src.bot_management.repository import BotRepository

        session = injector.resolve("AsyncSession")
        return BotRepository(session)

    def bot_instance_repository_factory():
        """Factory for BotInstanceRepository with proper session injection."""
        from src.bot_management.repository import BotInstanceRepository

        session = injector.resolve("AsyncSession")
        return BotInstanceRepository(session)

    def bot_metrics_repository_factory():
        """Factory for BotMetricsRepository with proper session injection."""
        from src.bot_management.repository import BotMetricsRepository

        session = injector.resolve("AsyncSession")
        return BotMetricsRepository(session)

    # Note: Repository singletons should be per session, not global instances
    injector.register_factory("BotRepository", bot_repository_factory, singleton=False)
    injector.register_factory(
        "BotInstanceRepository", bot_instance_repository_factory, singleton=False
    )
    injector.register_factory(
        "BotMetricsRepository", bot_metrics_repository_factory, singleton=False
    )

    # Register services using simplified factory pattern
    from .factory import register_bot_management_services as register_factory_services

    register_factory_services(injector)

    logger.info("Bot management services registered successfully with simplified factories")


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
