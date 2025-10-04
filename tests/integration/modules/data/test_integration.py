"""Integration tests for the data module.

Tests focus on dependency injection configuration, service registration,
and interface compliance without requiring database connections.
"""

from decimal import Decimal
from datetime import datetime, timezone

import pytest

from src.core.config import Config
from src.core.dependency_injection import DependencyInjector
from src.data.di_registration import configure_data_dependencies
from src.data.interfaces import DataServiceInterface, DataStorageInterface
from src.core.types.market import MarketData


@pytest.mark.asyncio
class TestDataModuleIntegration:
    """Test data module integration with proper DI patterns."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        # Configure database settings for data service
        config.database.redis_host = "localhost"
        config.database.redis_port = 6379
        config.database.redis_db = 1  # Use test database

        # Configure PostgreSQL for test environment
        config.database.postgresql_host = "localhost"
        config.database.postgresql_port = 5432
        config.database.postgresql_database = "tbot_test"
        config.database.postgresql_username = "tbot"
        config.database.postgresql_password = "password"

        return config

    @pytest.fixture
    def injector(self, config):
        """Create properly configured dependency injector."""
        injector = DependencyInjector()

        # Register config first
        injector.register_singleton("ConfigService", config)

        # Configure data dependencies (without database dependencies to avoid connection issues)
        return configure_data_dependencies(injector)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("49900.00"),
            high=Decimal("50100.00"),
            low=Decimal("49800.00"),
            close=Decimal("50000.00"),
            volume=Decimal("100.0"),
            quote_volume=Decimal("5000000.00"),
            trades_count=1500,
            exchange="binance",
        )

    @pytest.mark.asyncio
    async def test_dependency_injection_configuration(self, injector):
        """Test that all required dependencies are properly configured."""
        # Test that core interfaces are registered
        assert injector.is_registered("DataServiceInterface")
        assert injector.is_registered("DataStorageInterface")
        assert injector.is_registered("DataCacheInterface")
        assert injector.is_registered("DataValidatorInterface")

        # Test factory patterns are properly configured
        assert injector.is_registered("MarketDataSource")
        assert injector.is_registered("VectorizedProcessor")
        assert injector.is_registered("DataServiceFactory")

    @pytest.mark.asyncio
    async def test_data_service_factory_creation(self, injector):
        """Test that data service factories are properly configured."""
        # Test that all required factories are registered
        required_factories = [
            "DataServiceInterface",
            "DataStorageInterface",
            "DataCacheInterface",
            "DataValidatorInterface",
            "MarketDataSource",
            "VectorizedProcessor",
            "DataServiceFactory"
        ]

        for factory_name in required_factories:
            assert injector.is_registered(factory_name), f"Factory {factory_name} not registered"

    @pytest.mark.asyncio
    async def test_data_module_configuration_completeness(self, injector):
        """Test that the data module has all necessary configuration."""
        # Verify all data interfaces are properly registered
        data_interfaces = [
            "DataServiceInterface",
            "DataStorageInterface",
            "DataCacheInterface",
            "DataValidatorInterface",
            "ServiceDataValidatorInterface"
        ]

        for interface in data_interfaces:
            assert injector.is_registered(interface), f"Interface {interface} not found"

        # Test that legacy and new service names are both available
        assert injector.is_registered("DataService")  # Legacy name
        assert injector.is_registered("data_service")  # Lowercase alias

    @pytest.mark.asyncio
    async def test_specialized_services_registration(self, injector):
        """Test that specialized data services are properly registered."""
        specialized_services = [
            "DataPipelineIngestion",
            "StreamingDataService",
            "DataServiceRegistry"
        ]

        for service_name in specialized_services:
            assert injector.is_registered(service_name), f"Specialized service {service_name} not registered"

    @pytest.mark.asyncio
    async def test_market_data_model_validation(self, sample_market_data):
        """Test that market data model is properly structured."""
        # Test that sample data has all required fields
        assert sample_market_data.symbol == "BTCUSDT"
        assert sample_market_data.close == Decimal("50000.00")
        assert sample_market_data.volume == Decimal("100.0")
        assert isinstance(sample_market_data.timestamp, datetime)
        assert sample_market_data.open == Decimal("49900.00")
        assert sample_market_data.high == Decimal("50100.00")
        assert sample_market_data.low == Decimal("49800.00")
        assert sample_market_data.exchange == "binance"

        # Test decimal precision preservation
        assert isinstance(sample_market_data.close, Decimal)
        assert isinstance(sample_market_data.volume, Decimal)
        assert isinstance(sample_market_data.open, Decimal)
        assert isinstance(sample_market_data.high, Decimal)
        assert isinstance(sample_market_data.low, Decimal)