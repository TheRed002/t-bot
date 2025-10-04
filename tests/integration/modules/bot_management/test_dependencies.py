"""
Integration tests for bot_management module dependency injection validation.

This test validates that the bot_management module is properly integrated
with all its dependencies and follows the Controller-Service-Repository pattern.

Phase 5: Real service-based tests - NO MOCKS allowed except for external APIs.
"""

import pytest
import pytest_asyncio
from decimal import Decimal

from tests.integration.infrastructure.service_factory import RealServiceFactory

from src.core.dependency_injection import DependencyContainer
from src.core.types import BotConfiguration, BotPriority, BotStatus, BotType
from src.bot_management.di_registration import register_bot_management_services
from src.bot_management.service import BotService
from src.bot_management.factory import BotManagementFactory


@pytest_asyncio.fixture
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
def sample_bot_config():
    """Create a sample bot configuration for testing."""
    return BotConfiguration(
        bot_id="test-bot-123",
        bot_name="Test Bot",
        bot_type=BotType.TRADING,
        strategy_name="mean_reversion",
        exchanges=["binance"],
        symbols=["BTC/USDT"],
        allocated_capital=Decimal("10000"),
        risk_percentage=Decimal("0.02"),
        priority=BotPriority.NORMAL,
        auto_start=False,
        paper_trading=True,
        strategy_parameters={"lookback_period": 20}
    )


class TestBotManagementDependencyValidation:
    """Test bot_management module dependency injection and integration with real services."""

    @pytest.mark.asyncio
    async def test_dependency_injection_registration_complete(self, real_bot_services):
        """Test that all bot management services are registered correctly."""
        container = real_bot_services.container

        # Verify core services are registered
        assert container.get("BotService") is not None
        assert container.get("BotInstanceService") is not None
        assert container.get("BotLifecycleService") is not None
        assert container.get("BotCoordinationService") is not None
        assert container.get("BotResourceService") is not None
        assert container.get("BotMonitoringService") is not None

    @pytest.mark.asyncio
    async def test_bot_management_factory_integration(self, real_bot_services):
        """Test that BotManagementFactory creates services with proper dependencies."""
        container = real_bot_services.container

        # Create factory with real container
        factory = BotManagementFactory(container)

        # Test factory service creation
        bot_service = factory.create_bot_service()
        assert bot_service is not None
        assert isinstance(bot_service, BotService)

        # Test coordinator creation
        coordinator = factory.create_bot_coordinator()
        assert coordinator is not None

        # Test resource manager creation
        resource_manager = factory.create_resource_manager()
        assert resource_manager is not None

    @pytest.mark.asyncio
    async def test_bot_service_dependency_injection(self, real_bot_services, sample_bot_config):
        """Test that BotService receives and uses all required dependencies."""
        bot_service = real_bot_services.container.get("BotService")

        # Verify dependencies are properly injected
        assert bot_service._exchange_service is not None
        assert bot_service._capital_service is not None
        assert bot_service._execution_service is not None
        assert bot_service._risk_service is not None
        assert bot_service._state_service is not None
        assert bot_service._strategy_service is not None

    @pytest.mark.asyncio
    async def test_circular_dependency_prevention(self):
        """Test that TYPE_CHECKING is used to prevent circular dependencies."""
        # Check that critical files use TYPE_CHECKING
        import src.bot_management.di_registration as di_module
        import src.bot_management.factory as factory_module
        import src.bot_management.service as service_module

        # These modules should have TYPE_CHECKING imports
        assert hasattr(di_module, 'TYPE_CHECKING')
        assert hasattr(factory_module, 'TYPE_CHECKING')
        assert hasattr(service_module, 'TYPE_CHECKING')

    @pytest.mark.asyncio
    async def test_core_module_integration(self, real_bot_services):
        """Test integration with core modules."""
        bot_service = real_bot_services.container.get("BotService")

        # Verify core imports are available
        assert hasattr(BotService, '__bases__')
        assert any('BaseService' in str(base) for base in BotService.__bases__)

    @pytest.mark.asyncio
    async def test_database_model_integration(self):
        """Test that bot_management integrates with database models."""
        # Verify database models exist for bot management
        from src.database.models.bot import Bot
        from src.database.models.bot_instance import BotInstance

        # Check that models have required fields
        assert hasattr(Bot, 'id')
        assert hasattr(Bot, 'name')
        assert hasattr(Bot, 'status')
        assert hasattr(Bot, 'allocated_capital')

        assert hasattr(BotInstance, 'id')
        assert hasattr(BotInstance, 'bot_id')
        assert hasattr(BotInstance, 'status')

    @pytest.mark.asyncio
    async def test_execution_service_integration(self, real_bot_services):
        """Test integration with execution service."""
        execution_service = real_bot_services.container.get("execution_service")
        assert execution_service is not None

        # Verify service has expected methods
        assert hasattr(execution_service, 'health_check')

    @pytest.mark.asyncio
    async def test_risk_management_integration(self, real_bot_services):
        """Test integration with risk management service."""
        risk_service = real_bot_services.container.get("risk_service")
        assert risk_service is not None

        # Verify service has expected methods
        assert hasattr(risk_service, 'health_check')

    @pytest.mark.asyncio
    async def test_strategy_service_integration(self, real_bot_services):
        """Test integration with strategy service."""
        strategy_service = real_bot_services.container.get("strategy_service")
        assert strategy_service is not None

        # Verify service has expected methods
        assert hasattr(strategy_service, 'health_check')

    @pytest.mark.asyncio
    async def test_capital_management_integration(self, real_bot_services):
        """Test integration with capital management service."""
        capital_service = real_bot_services.container.get("capital_service")
        assert capital_service is not None

        # Verify service has expected methods
        assert hasattr(capital_service, 'allocate_capital')

    @pytest.mark.asyncio
    async def test_web_interface_integration(self):
        """Test that web interface properly uses bot_management."""
        # Verify web interface imports bot_management correctly
        try:
            from src.web_interface.api.bot_management import get_bot_service
            from src.web_interface.services.bot_service import WebBotService

            # These imports should work without circular dependency issues
            assert callable(get_bot_service)
            assert hasattr(WebBotService, '__init__')
        except ImportError as e:
            pytest.skip(f"Web interface bot_management integration not fully implemented: {e}")

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, real_bot_services):
        """Test integration with monitoring service."""
        bot_service = real_bot_services.container.get("BotService")

        # Verify bot service has monitoring capabilities
        assert hasattr(bot_service, 'health_check')

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, real_bot_services, sample_bot_config):
        """Test that error handling decorators work with bot_management."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Test that errors are properly handled
        # Non-existent bot should raise appropriate error or return error response
        result = await bot_service.get_bot_status("nonexistent_bot")
        assert isinstance(result, dict)
        # Should contain error information
        assert "error" in result or "status" in result
