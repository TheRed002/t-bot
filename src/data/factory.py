"""
Data Service Factory

Factory for creating data services with proper dependency injection.
"""

from src.core.config import Config
from src.data.cache.redis_cache import RedisCache
from src.data.interfaces import DataServiceInterface
from src.data.services.refactored_data_service import RefactoredDataService
from src.data.storage.database_storage import DatabaseStorage
from src.data.validation.market_data_validator import MarketDataValidator
from src.monitoring import MetricsCollector


class DataServiceFactory:
    """Factory for creating data services with dependencies."""

    @staticmethod
    def create_data_service(
        config: Config,
        use_cache: bool = True,
        use_validator: bool = True,
        metrics_collector: MetricsCollector | None = None,
    ) -> DataServiceInterface:
        """
        Create a fully configured data service.

        Args:
            config: Service configuration
            use_cache: Whether to enable caching
            use_validator: Whether to enable validation
            metrics_collector: Optional metrics collector

        Returns:
            DataServiceInterface: Configured data service
        """
        # Create storage implementation
        storage = DatabaseStorage(config)

        # Create cache implementation if requested
        cache = RedisCache(config) if use_cache else None

        # Create validator if requested
        validator = MarketDataValidator() if use_validator else None

        # Create the service with dependencies
        return RefactoredDataService(
            config=config,
            storage=storage,
            cache=cache,
            validator=validator,
            metrics_collector=metrics_collector,
        )

    @staticmethod
    def create_minimal_data_service(config: Config) -> DataServiceInterface:
        """
        Create a minimal data service with just storage.

        Args:
            config: Service configuration

        Returns:
            DataServiceInterface: Minimal data service
        """
        storage = DatabaseStorage(config)
        return RefactoredDataService(config=config, storage=storage)

    @staticmethod
    def create_testing_data_service(
        config: Config,
        mock_storage=None,
        mock_cache=None,
        mock_validator=None,
    ) -> DataServiceInterface:
        """
        Create a data service for testing with mock dependencies.

        Args:
            config: Service configuration
            mock_storage: Mock storage implementation
            mock_cache: Mock cache implementation
            mock_validator: Mock validator implementation

        Returns:
            DataServiceInterface: Testing data service
        """
        storage = mock_storage or DatabaseStorage(config)
        
        return RefactoredDataService(
            config=config,
            storage=storage,
            cache=mock_cache,
            validator=mock_validator,
        )