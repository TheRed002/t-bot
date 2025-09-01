"""Data services dependency injection registration."""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.data.interfaces import (
        DataCacheInterface,
        DataServiceInterface,
        DataStorageInterface,
        DataValidatorInterface,
    )
    from src.data.services.data_service import DataService
    from src.data.sources.market_data import MarketDataSource
    from src.data.vectorized_processor import VectorizedProcessor

logger = get_logger(__name__)


def _resolve_optional_dependency(
    injector: DependencyInjector, service_name: str, default_factory=None
):
    """Helper to resolve optional dependencies with fallback."""
    try:
        return injector.resolve(service_name)
    except Exception as e:
        logger.debug(f"{service_name} not available: {e}")
        if default_factory:
            return default_factory()
        return None


def _get_config(injector: DependencyInjector):
    """Get config with fallback to default."""
    config = _resolve_optional_dependency(injector, "ConfigService")
    if config:
        return config.get_config()

    # Fallback to default config
    from src.core.config import Config
    return Config()


def register_data_services(injector: DependencyInjector) -> None:
    """
    Register data services with the dependency injector.

    Args:
        injector: Dependency injector instance
    """

    # Register DataStorageInterface implementation
    def database_storage_factory() -> "DataStorageInterface":
        from src.data.storage.database_storage import DatabaseStorage

        config = _get_config(injector)
        database_service = _resolve_optional_dependency(injector, "DatabaseService")

        return DatabaseStorage(config=config, database_service=database_service)

    injector.register_factory("DataStorageInterface", database_storage_factory, singleton=True)

    # Register DataCacheInterface implementation
    def redis_cache_factory() -> "DataCacheInterface":
        from src.data.cache.redis_cache import RedisCache

        config = _get_config(injector)
        return RedisCache(config=config)

    injector.register_factory("DataCacheInterface", redis_cache_factory, singleton=True)

    # Register DataValidatorInterface implementation
    def market_data_validator_factory() -> "DataValidatorInterface":
        from src.data.validation.market_data_validator import MarketDataValidator

        return MarketDataValidator()

    injector.register_factory(
        "DataValidatorInterface", market_data_validator_factory, singleton=True
    )
    injector.register_factory(
        "ServiceDataValidatorInterface", market_data_validator_factory, singleton=True
    )

    # Register RefactoredDataService as the main DataService implementation
    def refactored_data_service_factory() -> "DataServiceInterface":
        from src.data.services.refactored_data_service import RefactoredDataService

        config = _get_config(injector)
        storage = injector.resolve("DataStorageInterface")
        cache = injector.resolve("DataCacheInterface")
        validator = injector.resolve("ServiceDataValidatorInterface")
        metrics_collector = _resolve_optional_dependency(injector, "MetricsCollector")

        return RefactoredDataService(
            config=config,
            storage=storage,
            cache=cache,
            validator=validator,
            metrics_collector=metrics_collector,
        )

    injector.register_factory(
        "DataServiceInterface", refactored_data_service_factory, singleton=True
    )

    # Register DataService (legacy service) with proper dependencies
    def data_service_factory() -> "DataService":
        from src.data.services.data_service import DataService

        config = _get_config(injector)
        database_service = _resolve_optional_dependency(injector, "DatabaseService")
        metrics_collector = _resolve_optional_dependency(injector, "MetricsCollector")

        return DataService(
            config=config,
            metrics_collector=metrics_collector,
            database_service=database_service,
        )

    injector.register_factory("DataService", data_service_factory, singleton=True)

    # Register aliases for different service names
    injector.register_factory(
        "data_service", lambda: injector.resolve("DataServiceInterface"), singleton=True
    )

    # Register MarketDataSource factory
    def market_data_source_factory() -> "MarketDataSource":
        from src.data.sources.market_data import MarketDataSource

        config = _get_config(injector)
        exchange_factory = _resolve_optional_dependency(injector, "ExchangeFactory")

        return MarketDataSource(config=config, exchange_factory=exchange_factory)

    injector.register_factory("MarketDataSource", market_data_source_factory, singleton=True)

    # Register VectorizedProcessor factory
    def vectorized_processor_factory() -> "VectorizedProcessor":
        from src.data.vectorized_processor import VectorizedProcessor

        config = _get_config(injector)
        return VectorizedProcessor(config=config)

    injector.register_factory("VectorizedProcessor", vectorized_processor_factory, singleton=True)

    # Register DataServiceFactory itself for factory pattern access
    def data_service_factory_factory():
        # Import inside function to avoid circular dependency
        from src.data.factory import DataServiceFactory
        return DataServiceFactory(injector)

    injector.register_factory("DataServiceFactory", data_service_factory_factory, singleton=True)

    logger.info("Data services registered with dependency injector")


def configure_data_dependencies(
    injector: DependencyInjector | None = None,
) -> DependencyInjector:
    """
    Configure data dependencies with proper service lifetimes.

    Args:
        injector: Optional existing injector instance

    Returns:
        Configured dependency injector
    """
    if injector is None:
        injector = DependencyInjector()

    register_data_services(injector)

    return injector


# Convenience functions to get data services from DI container
def get_data_service(injector: DependencyInjector) -> "DataServiceInterface":
    """Get DataService from DI container."""
    return injector.resolve("DataServiceInterface")


def get_data_storage(injector: DependencyInjector) -> "DataStorageInterface":
    """Get DataStorageInterface from DI container."""
    return injector.resolve("DataStorageInterface")


def get_data_cache(injector: DependencyInjector) -> "DataCacheInterface":
    """Get DataCacheInterface from DI container."""
    return injector.resolve("DataCacheInterface")


def get_data_validator(injector: DependencyInjector) -> "DataValidatorInterface":
    """Get DataValidatorInterface from DI container."""
    return injector.resolve("DataValidatorInterface")


def get_market_data_source(injector: DependencyInjector) -> "MarketDataSource":
    """Get MarketDataSource from DI container."""
    return injector.resolve("MarketDataSource")


def get_vectorized_processor(injector: DependencyInjector) -> "VectorizedProcessor":
    """Get VectorizedProcessor from DI container."""
    return injector.resolve("VectorizedProcessor")


def get_service_data_validator(injector: DependencyInjector):
    """Get ServiceDataValidatorInterface from DI container."""
    return injector.resolve("ServiceDataValidatorInterface")


def get_data_service_factory(injector: DependencyInjector):
    """Get DataServiceFactory from DI container."""
    return injector.resolve("DataServiceFactory")
