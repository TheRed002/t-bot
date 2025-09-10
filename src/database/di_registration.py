"""Database services dependency injection registration."""

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger
from src.database.connection import (
    DatabaseConnectionManager,
)

# Import classes that tests need to patch
try:
    from src.database.manager import DatabaseManager
except ImportError:
    DatabaseManager = None

try:
    from src.database.repository_factory import RepositoryFactory
except ImportError:
    RepositoryFactory = None

try:
    from src.database.service import DatabaseService
except ImportError:
    DatabaseService = None

try:
    from src.database.uow import UnitOfWorkFactory
except ImportError:
    UnitOfWorkFactory = None

if TYPE_CHECKING:
    from src.database.services import BotService, MLService, TradingService
    from src.database.uow import AsyncUnitOfWork, UnitOfWork

logger = get_logger(__name__)


def register_database_services(injector: DependencyInjector) -> None:
    """
    Register database services with the dependency injector.

    Args:
        injector: Dependency injector instance
    """
    _register_connection_manager(injector)
    _register_repository_factory(injector)
    _register_session_factories(injector)
    _register_database_service(injector)
    _register_interface_implementations(injector)
    _register_database_manager(injector)
    _register_unit_of_work_factory(injector)
    _register_unit_of_work_instances(injector)
    _register_specialized_services(injector)
    _register_repository_instances(injector)


def _register_connection_manager(injector: DependencyInjector) -> None:
    """Register DatabaseConnectionManager as singleton with proper factory pattern."""

    def connection_manager_factory() -> DatabaseConnectionManager:
        # Use dependency injection to get config
        if injector.has_service("ConfigService"):
            config_service = injector.resolve("ConfigService")
            config = config_service.get_config()
        else:
            from src.core.config import Config

            config = Config().to_dict()

        return DatabaseConnectionManager(config)

    injector.register_factory(
        "DatabaseConnectionManager", connection_manager_factory, singleton=True
    )

    # Register interface implementation
    injector.register_service(
        "ConnectionManagerInterface",
        lambda: injector.resolve("DatabaseConnectionManager"),
        singleton=True,
    )


def _register_repository_factory(injector: DependencyInjector) -> None:
    """Register RepositoryFactory as singleton using proper dependency injection."""

    def repository_factory_factory() -> "RepositoryFactory":
        from src.database.repository_factory import RepositoryFactory as RepoFactory

        return RepoFactory(dependency_injector=injector)

    injector.register_factory("RepositoryFactory", repository_factory_factory, singleton=True)

    # Register interface implementation
    injector.register_service(
        "RepositoryFactoryInterface", lambda: injector.resolve("RepositoryFactory"), singleton=True
    )


def _register_session_factories(injector: DependencyInjector) -> None:
    """Register session factories as transient."""
    _register_async_session_factory(injector)
    _register_sync_session_factory(injector)
    _register_database_service_factory(injector)
    _register_database_interfaces(injector)


def _register_async_session_factory(injector: DependencyInjector) -> None:
    """Register async session factory."""

    def async_session_factory() -> AsyncSession:
        connection_manager = injector.resolve("DatabaseConnectionManager")
        if not connection_manager.async_session_maker:
            raise RuntimeError("Async session maker not initialized")
        return connection_manager.async_session_maker()

    injector.register_factory("AsyncSession", async_session_factory, singleton=False)


def _register_sync_session_factory(injector: DependencyInjector) -> None:
    """Register sync session factory."""

    def sync_session_factory() -> Session:
        connection_manager = injector.resolve("DatabaseConnectionManager")
        if not connection_manager.sync_session_maker:
            raise RuntimeError("Sync session maker not initialized")
        return connection_manager.sync_session_maker()

    injector.register_factory("Session", sync_session_factory, singleton=False)


def _register_database_service_factory(injector: DependencyInjector) -> None:
    """Register database service factory with simplified dependency injection."""

    def database_service_factory() -> "DatabaseService":
        return _create_database_service_with_deps(injector)

    injector.register_factory("DatabaseService", database_service_factory, singleton=True)

    # Register interface mapping to the same singleton instance
    injector.register_service(
        "DatabaseServiceInterface", lambda: injector.resolve("DatabaseService"), singleton=True
    )


def _create_database_service_with_deps(injector: DependencyInjector) -> "DatabaseService":
    """Create database service with dependency injection."""
    from src.database.service import DatabaseService

    # Use dependency injection to resolve required services
    config_service = (
        injector.resolve("ConfigService") if injector.has_service("ConfigService") else None
    )
    validation_service = (
        injector.resolve("ValidationService") if injector.has_service("ValidationService") else None
    )
    connection_manager = injector.resolve("DatabaseConnectionManager")

    return DatabaseService(
        config_service=config_service,
        validation_service=validation_service,
        connection_manager=connection_manager,
        dependency_injector=injector,
    )


def _register_database_interfaces(injector: DependencyInjector) -> None:
    """Register database interface implementations."""
    interfaces = [
        "DatabaseServiceInterface",
        "TradingDataServiceInterface",
        "BotMetricsServiceInterface",
        "HealthAnalyticsServiceInterface",
    ]

    for interface in interfaces:
        injector.register_service(
            interface, lambda: injector.resolve("DatabaseService"), singleton=True
        )

    injector.register_service(
        "ResourceManagementServiceInterface",
        lambda: injector.resolve("DatabaseService"),
        singleton=True,
    )

    # Register DatabaseManager as singleton
    def database_manager_factory() -> "DatabaseManager":
        from src.database.manager import DatabaseManager as _db_manager

        database_service = injector.resolve("DatabaseService")
        return _db_manager(database_service=database_service)

    injector.register_factory("DatabaseManager", database_manager_factory, singleton=True)

    # Register UnitOfWorkFactory as singleton
    def uow_factory_factory() -> "UnitOfWorkFactory":
        from sqlalchemy.ext.asyncio import async_sessionmaker
        from sqlalchemy.orm import sessionmaker

        from src.database.uow import UnitOfWorkFactory as _uow_factory

        # Create proper session makers using connection manager
        connection_manager = injector.resolve("DatabaseConnectionManager")

        sync_session_maker = sessionmaker(bind=connection_manager.sync_engine)
        async_session_maker = async_sessionmaker(
            bind=connection_manager.async_engine, expire_on_commit=False
        )

        return _uow_factory(sync_session_maker, async_session_maker, dependency_injector=injector)

    injector.register_factory("UnitOfWorkFactory", uow_factory_factory, singleton=True)

    # Register interface implementation
    injector.register_service(
        "UnitOfWorkFactoryInterface", lambda: injector.resolve("UnitOfWorkFactory"), singleton=True
    )

    # Register UoW instances as transient (new per request)
    def uow_factory() -> "UnitOfWork":
        factory = injector.resolve("UnitOfWorkFactory")
        return factory.create()

    def async_uow_factory() -> "AsyncUnitOfWork":
        factory = injector.resolve("UnitOfWorkFactory")
        return factory.create_async()

    injector.register_factory("UnitOfWork", uow_factory, singleton=False)
    injector.register_factory("AsyncUnitOfWork", async_uow_factory, singleton=False)

    # Register repository factories using simplified dependency injection
    def register_repository_factory(repository_name: str, repository_class):
        """Register repository factory using dependency injection pattern."""

        def factory():
            session_factory = injector.resolve("AsyncSession")
            session = session_factory()
            return repository_class(session)

        return factory

    # Register trading repository factories
    from src.database.repository.trading import (
        OrderFillRepository,
        OrderRepository,
        PositionRepository,
        TradeRepository,
    )

    injector.register_factory(
        "OrderRepository",
        register_repository_factory("OrderRepository", OrderRepository),
        singleton=False,
    )
    injector.register_factory(
        "PositionRepository",
        register_repository_factory("PositionRepository", PositionRepository),
        singleton=False,
    )
    injector.register_factory(
        "TradeRepository",
        register_repository_factory("TradeRepository", TradeRepository),
        singleton=False,
    )
    injector.register_factory(
        "OrderFillRepository",
        register_repository_factory("OrderFillRepository", OrderFillRepository),
        singleton=False,
    )

    # Register ML repository factories
    from src.database.repository.ml import (
        MLModelMetadataRepository,
        MLPredictionRepository,
        MLTrainingJobRepository,
    )

    injector.register_factory(
        "MLPredictionRepository",
        register_repository_factory("MLPredictionRepository", MLPredictionRepository),
        singleton=False,
    )
    injector.register_factory(
        "MLModelMetadataRepository",
        register_repository_factory("MLModelMetadataRepository", MLModelMetadataRepository),
        singleton=False,
    )
    injector.register_factory(
        "MLTrainingJobRepository",
        register_repository_factory("MLTrainingJobRepository", MLTrainingJobRepository),
        singleton=False,
    )

    # Register Bot repository factory
    from src.database.repository.bot import BotRepository

    injector.register_factory(
        "BotRepository",
        register_repository_factory("BotRepository", BotRepository),
        singleton=False,
    )

    # Register specialized services for business logic
    def trading_service_factory() -> "TradingService":
        """Create TradingService with injected repositories."""
        from src.database.services import TradingService

        order_repo = injector.resolve("OrderRepository")
        position_repo = injector.resolve("PositionRepository")
        trade_repo = injector.resolve("TradeRepository")
        fill_repo = injector.resolve("OrderFillRepository")

        return TradingService(
            order_repo=order_repo,
            position_repo=position_repo,
            trade_repo=trade_repo,
            fill_repo=fill_repo,
        )

    injector.register_factory("TradingService", trading_service_factory, singleton=True)

    def ml_service_factory() -> "MLService":
        """Create MLService with injected repositories."""
        from src.database.services import MLService

        prediction_repo = injector.resolve("MLPredictionRepository")
        model_repo = injector.resolve("MLModelMetadataRepository")
        training_repo = injector.resolve("MLTrainingJobRepository")

        return MLService(
            prediction_repo=prediction_repo, model_repo=model_repo, training_repo=training_repo
        )

    injector.register_factory("MLService", ml_service_factory, singleton=True)

    def bot_service_factory() -> "BotService":
        """Create BotService with injected repositories."""
        from src.database.services import BotService

        bot_repo = injector.resolve("BotRepository")
        return BotService(bot_repo=bot_repo)

    injector.register_factory("BotService", bot_service_factory, singleton=True)

    # Register specialized service interfaces
    injector.register_service(
        "TradingServiceInterface", lambda: injector.resolve("TradingService"), singleton=True
    )

    injector.register_service(
        "MLServiceInterface", lambda: injector.resolve("MLService"), singleton=True
    )

    injector.register_service(
        "BotMetricsServiceInterface", lambda: injector.resolve("BotService"), singleton=True
    )

    logger.info("Database services and specialized services registered with dependency injector")


def configure_database_dependencies(
    injector: DependencyInjector | None = None,
) -> DependencyInjector:
    """
    Configure database dependencies with proper service lifetimes.

    Args:
        injector: Optional existing injector instance

    Returns:
        Configured dependency injector
    """
    if injector is None:
        # Use global injector instance
        try:
            injector = DependencyInjector.get_instance()
        except (RuntimeError, AttributeError) as e:
            # Fallback to new instance if global injector unavailable
            logger.warning(f"Could not get global injector: {e}. Creating new instance.")
            injector = DependencyInjector()

    register_database_services(injector)

    return injector


# Convenience function to get database services from DI container
def get_database_service(injector: DependencyInjector) -> "DatabaseService":
    """Get DatabaseService from DI container."""
    return injector.resolve("DatabaseService")


def get_database_manager(injector: DependencyInjector) -> "DatabaseManager":
    """Get DatabaseManager from DI container."""
    return injector.resolve("DatabaseManager")


def get_uow_factory(injector: DependencyInjector) -> "UnitOfWorkFactory":
    """Get UnitOfWorkFactory from DI container."""
    return injector.resolve("UnitOfWorkFactory")


# Helper functions now implemented in _register_database_interfaces
def _register_database_service(injector: DependencyInjector) -> None:
    """Register database service - implemented in _register_session_factories."""
    pass


def _register_interface_implementations(injector: DependencyInjector) -> None:
    """Register interface implementations - implemented in _register_database_interfaces."""
    pass


def _register_database_manager(injector: DependencyInjector) -> None:
    """Register database manager - implemented in _register_database_interfaces."""
    pass


def _register_unit_of_work_factory(injector: DependencyInjector) -> None:
    """Register unit of work factory - implemented in _register_database_interfaces."""
    pass


def _register_unit_of_work_instances(injector: DependencyInjector) -> None:
    """Register unit of work instances - implemented in _register_database_interfaces."""
    pass


def _register_specialized_services(injector: DependencyInjector) -> None:
    """Register specialized services - implemented in _register_database_interfaces."""
    pass


def _register_repository_instances(injector: DependencyInjector) -> None:
    """Register repository instances - implemented in _register_database_interfaces."""
    pass
