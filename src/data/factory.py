"""
Data Service Factory

Factory for creating data services with proper dependency injection.
"""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.data.interfaces import DataServiceInterface

if TYPE_CHECKING:
    from src.core.config import Config


class DataServiceFactory:
    """Factory for creating data services with proper dependency injection."""

    def __init__(self, injector: DependencyInjector | None = None):
        """
        Initialize factory with dependency injector.

        Args:
            injector: Optional dependency injector instance
        """
        # Don't create default injector - require injection
        if injector is None:
            from src.core.exceptions import ComponentError

            raise ComponentError(
                "Injector must be provided to factory",
                component="DataServiceFactory",
                operation="__init__",
                context={"missing_dependency": "injector"},
            )
        self._injector = injector

    def create_data_service(
        self,
        use_cache: bool = True,
        use_validator: bool = True,
    ) -> DataServiceInterface:
        """
        Create a fully configured data service using dependency injection.

        Args:
            use_cache: Whether to enable caching
            use_validator: Whether to enable validation

        Returns:
            DataServiceInterface: Configured data service
        """
        return self._injector.resolve("DataServiceInterface")

    def create_minimal_data_service(self) -> DataServiceInterface:
        """
        Create a minimal data service using dependency injection.

        Returns:
            DataServiceInterface: Minimal data service
        """
        return self._injector.resolve("DataServiceInterface")

    def create_testing_data_service(
        self,
        mock_storage=None,
        mock_cache=None,
        mock_validator=None,
    ) -> DataServiceInterface:
        """
        Create a data service for testing with mock dependencies using factory patterns.

        Args:
            mock_storage: Mock storage implementation
            mock_cache: Mock cache implementation
            mock_validator: Mock validator implementation

        Returns:
            DataServiceInterface: Testing data service
        """
        # Create a separate injector for testing to avoid conflicts
        from src.data.di_registration import configure_data_dependencies

        test_injector = configure_data_dependencies()

        # Override services with mocks if provided
        if mock_storage:
            test_injector.register_factory(
                "DataStorageInterface", lambda: mock_storage, singleton=True
            )
        if mock_cache:
            test_injector.register_factory("DataCacheInterface", lambda: mock_cache, singleton=True)
        if mock_validator:
            test_injector.register_factory(
                "DataValidatorInterface", lambda: mock_validator, singleton=True
            )

        return test_injector.resolve("DataServiceInterface")

    def create_data_storage(self) -> "DataStorageInterface":
        """Create data storage service using DI."""
        return self._injector.resolve("DataStorageInterface")

    def create_data_cache(self) -> "DataCacheInterface":
        """Create data cache service using DI."""
        return self._injector.resolve("DataCacheInterface")

    def create_data_validator(self) -> "DataValidatorInterface":
        """Create data validator service using DI."""
        return self._injector.resolve("DataValidatorInterface")

    def create_market_data_source(self) -> "MarketDataSource":
        """Create market data source using DI."""
        return self._injector.resolve("MarketDataSource")

    def create_vectorized_processor(self) -> "VectorizedProcessor":
        """Create vectorized processor using DI."""
        return self._injector.resolve("VectorizedProcessor")


def create_default_data_service(
    config: "Config | None" = None,
    injector: DependencyInjector | None = None,
) -> DataServiceInterface:
    """
    Create data service with default dependencies.

    Args:
        config: Service configuration
        injector: Required dependency injector

    Returns:
        Configured data service
    """
    if injector is None:
        from src.data.di_registration import configure_data_dependencies

        injector = configure_data_dependencies()

    # Register config if provided
    if config:
        injector.register_factory("Config", lambda: config, singleton=True)

    factory = DataServiceFactory(injector=injector)
    return factory.create_data_service()
