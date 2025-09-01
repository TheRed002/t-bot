"""
Data Service Factory

Factory for creating data services with proper dependency injection.
"""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.data.interfaces import DataServiceInterface

if TYPE_CHECKING:
    from src.core.config import Config
    from src.monitoring import MetricsCollector


class DataServiceFactory:
    """Factory for creating data services with proper dependency injection."""

    def __init__(self, injector: DependencyInjector | None = None):
        """
        Initialize factory with dependency injector.

        Args:
            injector: Optional dependency injector instance
        """
        if injector is None:
            from src.data.di_registration import configure_data_dependencies
            injector = configure_data_dependencies()

        self.injector = injector

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
        return self.injector.resolve("DataServiceInterface")

    def create_minimal_data_service(self) -> DataServiceInterface:
        """
        Create a minimal data service using dependency injection.

        Returns:
            DataServiceInterface: Minimal data service
        """
        return self.injector.resolve("DataServiceInterface")

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

    @staticmethod
    def create_data_service_from_config(
        config: "Config",
        use_cache: bool = True,
        use_validator: bool = True,
        metrics_collector: "MetricsCollector | None" = None,
    ) -> DataServiceInterface:
        """
        Legacy method for backward compatibility.

        Creates a data service with dependency injection patterns.

        Args:
            config: Service configuration
            use_cache: Whether to enable caching
            use_validator: Whether to enable validation
            metrics_collector: Optional metrics collector

        Returns:
            DataServiceInterface: Configured data service
        """
        from src.data.di_registration import configure_data_dependencies

        injector = configure_data_dependencies()

        # Register config if provided
        injector.register_factory("Config", lambda: config, singleton=True)

        # Register metrics collector if provided
        if metrics_collector:
            injector.register_factory("MetricsCollector", lambda: metrics_collector, singleton=True)

        return injector.resolve("DataServiceInterface")
