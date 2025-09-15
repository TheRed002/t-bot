"""
Bot Management Factory for creating main bot management components.

Simplified factory that creates the core bot management classes directly,
without unnecessary service wrappers and interfaces.
"""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import DependencyError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from .bot_coordinator import BotCoordinator
    from .resource_manager import ResourceManager
    from .service import BotService

logger = get_logger(__name__)


class BotManagementFactory:
    """Simplified factory for creating bot management components."""

    def __init__(self, injector: DependencyInjector = None):
        """Initialize factory with dependency injector."""
        self._injector = injector
        self.logger = get_logger(self.__class__.__name__)

    def create_bot_service(self) -> "BotService":
        """Create main BotService with simplified dependencies."""
        from .service import BotService

        if not self._injector:
            raise DependencyError("Injector required for BotService creation")

        try:
            exchange_service = self._injector.resolve("ExchangeService")
            capital_service = self._injector.resolve("CapitalService")
        except DependencyError as e:
            logger.error(f"Required dependencies not available: {e}")
            raise

        # Optional dependencies
        kwargs = {
            "exchange_service": exchange_service,
            "capital_service": capital_service,
        }

        for service_name, param_name in [
            ("ExecutionServiceInterface", "execution_service"),
            ("RiskServiceInterface", "risk_service"),
            ("StateService", "state_service"),
            ("StrategyServiceInterface", "strategy_service"),
            ("ConfigService", "config_service"),
            ("BotRepository", "bot_repository"),
            ("BotInstanceRepository", "bot_instance_repository"),
            ("BotMetricsRepository", "bot_metrics_repository"),
        ]:
            try:
                kwargs[param_name] = self._injector.resolve(service_name)
            except DependencyError:
                kwargs[param_name] = None

        return BotService(**kwargs)

    def create_bot_coordinator(self) -> "BotCoordinator":
        """Create BotCoordinator component."""
        from .bot_coordinator import BotCoordinator

        if not self._injector:
            raise DependencyError("Injector required for BotCoordinator creation")

        try:
            config = self._injector.resolve("Config")
            return BotCoordinator(config)
        except DependencyError as e:
            logger.error(f"Config dependency not available: {e}")
            raise

    def create_resource_manager(self) -> "ResourceManager":
        """Create ResourceManager component."""
        from .resource_manager import ResourceManager

        if not self._injector:
            raise DependencyError("Injector required for ResourceManager creation")

        try:
            config = self._injector.resolve("Config")
            capital_service = self._injector.resolve("CapitalService")
            return ResourceManager(config, capital_service)
        except DependencyError as e:
            logger.error(f"Required dependencies not available: {e}")
            raise


def register_bot_management_services(injector: DependencyInjector) -> None:
    """Register simplified bot management services."""
    logger.info("Registering simplified bot management services")

    factory = BotManagementFactory(injector)

    # Register main BotService
    injector.register_factory("BotService", lambda: factory.create_bot_service(), singleton=True)

    # Register factory itself
    injector.register_factory("BotManagementFactory", lambda: factory, singleton=True)

    logger.info("Bot management services registered successfully")


def create_bot_management_factory(injector: DependencyInjector = None) -> BotManagementFactory:
    """Create bot management factory."""
    return BotManagementFactory(injector)
