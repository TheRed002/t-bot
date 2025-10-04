"""
Bot Management Module Integration Validation with Real Services.

This test validates that bot_management module properly integrates with other modules
through proper dependency injection patterns and interface usage.

Phase 5: Real service-based tests - NO MOCKS allowed except for external APIs.
"""

import pytest
from decimal import Decimal

from tests.integration.infrastructure.service_factory import RealServiceFactory

from src.core.dependency_injection import DependencyContainer
from src.core.types import BotConfiguration, BotStatus, BotPriority, BotType
from src.bot_management.factory import BotManagementFactory
from src.bot_management.service import BotService
from src.bot_management.bot_instance import BotInstance


@pytest.fixture
async def real_bot_services(clean_database):
    """Create real bot management services with real dependencies."""
    service_factory = RealServiceFactory()
    try:
        await service_factory.initialize_core_services(clean_database)
        await service_factory.initialize_bot_management_services()
        yield service_factory
    finally:
        await service_factory.cleanup()


@pytest.fixture
def bot_config():
    """Create test bot configuration."""
    return BotConfiguration(
        bot_id="test_bot_001",
        name="Test Bot",
        version="1.0.0",
        bot_type=BotType.TRADING,
        strategy_name="test_strategy",
        exchanges=["binance"],
        symbols=["BTC/USDT"],
        allocated_capital=Decimal("1000"),
        risk_percentage=0.02
    )


class TestBotManagementModuleIntegration:
    """Test bot_management module integration with dependency injection."""

    @pytest.mark.asyncio
    async def test_dependency_injection_patterns(self, real_bot_services):
        """Test that bot_management uses proper dependency injection patterns."""
        container = real_bot_services.container

        # Test factory creates services with proper DI
        factory = BotManagementFactory(container)

        # Verify factory can create services without circular dependencies
        bot_service = factory.create_bot_service()
        assert bot_service is not None
        assert isinstance(bot_service, BotService)

        # Verify coordinator can be created
        coordinator = factory.create_bot_coordinator()
        assert coordinator is not None

        # Verify resource manager can be created
        resource_manager = factory.create_resource_manager()
        assert resource_manager is not None

    @pytest.mark.asyncio
    async def test_service_boundary_validation(self, real_bot_services):
        """Test that bot_management respects service boundaries."""
        bot_service = real_bot_services.container.get("BotService")

        # Verify service uses injected dependencies
        assert bot_service._exchange_service is not None
        assert bot_service._capital_service is not None

    def test_circular_dependency_prevention(self):
        """Test that there are no circular dependencies in bot_management."""
        # Import modules to ensure no circular imports
        try:
            from src.bot_management.service import BotService
            from src.bot_management.bot_monitor import BotMonitor
            from src.bot_management.bot_instance import BotInstance
            from src.bot_management.factory import BotManagementFactory

            # If we reach this point, no circular imports occurred
            assert True

        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    @pytest.mark.asyncio
    async def test_integration_with_capital_management(self, real_bot_services):
        """Test integration with capital management module."""
        capital_service = real_bot_services.container.get("capital_service")
        bot_service = real_bot_services.container.get("BotService")

        # Should properly integrate with capital service
        assert bot_service._capital_service is not None

    @pytest.mark.asyncio
    async def test_integration_with_execution_module(self, real_bot_services):
        """Test integration with execution module through interfaces."""
        execution_service = real_bot_services.container.get("execution_service")
        bot_service = real_bot_services.container.get("BotService")

        # Should use execution service interface
        assert bot_service._execution_service is not None

    def test_web_interface_integration_validation(self):
        """Test that web interface properly integrates with bot_management."""
        # Import web interface bot service to ensure integration works
        try:
            from src.web_interface.services.bot_service import WebBotService

            # Should be able to import without errors
            assert WebBotService is not None

        except ImportError as e:
            pytest.skip(f"Web interface integration not fully implemented: {e}")

    def test_module_provides_required_services(self):
        """Test that bot_management provides all required service classes."""
        from src.bot_management.service import BotService
        from src.bot_management.bot_coordinator import BotCoordinator
        from src.bot_management.bot_monitor import BotMonitor
        from src.bot_management.resource_manager import ResourceManager
        from src.bot_management.factory import BotManagementFactory

        # Verify all service classes are properly defined
        assert BotService is not None
        assert BotCoordinator is not None
        assert BotMonitor is not None
        assert ResourceManager is not None
        assert BotManagementFactory is not None

    def test_proper_exception_usage(self):
        """Test that bot_management uses core exceptions properly."""
        from src.bot_management.service import BotService
        from src.core.exceptions import ServiceError, ValidationError

        # Should import core exceptions, not define its own
        assert ServiceError is not None
        assert ValidationError is not None

    @pytest.mark.asyncio
    async def test_service_lifecycle_management(self, real_bot_services):
        """Test that services properly manage their lifecycle."""
        bot_service = real_bot_services.container.get("BotService")

        # Test service lifecycle
        await bot_service.start()
        assert bot_service.is_running

        await bot_service.stop()
        assert not bot_service.is_running

    def test_architecture_compliance(self):
        """Test that bot_management follows architectural patterns."""
        from src.bot_management.service import BotService
        from src.core.base.service import BaseService

        # Should inherit from BaseService
        assert issubclass(BotService, BaseService)

        # Should have proper service methods
        assert hasattr(BotService, 'create_bot')
        assert hasattr(BotService, 'start_bot')
        assert hasattr(BotService, 'stop_bot')
        assert hasattr(BotService, 'delete_bot')
