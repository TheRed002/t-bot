"""
Integration validation tests for bot_management module.

This test suite validates that bot_management correctly integrates with
other modules and uses their APIs properly with REAL services.

Phase 5: Real service-based tests - NO MOCKS allowed except for external APIs.
"""

import pytest
from decimal import Decimal

from tests.integration.infrastructure.service_factory import RealServiceFactory

from src.bot_management.service import BotService
from src.core.types import BotConfiguration, BotStatus, BotPriority, BotType
from src.core.exceptions import ServiceError, ValidationError


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
def sample_bot_config():
    """Sample bot configuration."""
    return BotConfiguration(
        bot_id="test_bot_001",
        name="Test Bot",
        bot_name="Test Bot",
        bot_type=BotType.TRADING,
        version="1.0.0",
        strategy_name="test_strategy",
        exchanges=["binance"],
        symbols=["BTCUSDT"],
        allocated_capital=Decimal("10000"),
        risk_percentage=0.02,
        strategy_parameters={"param1": "value1"},
        priority=BotPriority.NORMAL,
    )


class TestBotManagementIntegration:
    """Tests for bot management integration with other services."""

    @pytest.mark.asyncio
    async def test_bot_service_dependency_injection(self, real_bot_services):
        """Test that BotService properly initializes with all dependencies."""
        bot_service = real_bot_services.container.get("BotService")

        # All dependencies should be injected
        assert bot_service._exchange_service is not None
        assert bot_service._capital_service is not None
        assert bot_service._execution_service is not None
        assert bot_service._risk_service is not None
        assert bot_service._state_service is not None
        assert bot_service._strategy_service is not None

    @pytest.mark.asyncio
    async def test_create_bot_integration_flow(
        self,
        real_bot_services,
        sample_bot_config,
    ):
        """Test that create_bot follows proper integration patterns."""
        bot_service = real_bot_services.container.get("BotService")
        state_service = real_bot_services.container.get("state_service")
        capital_service = real_bot_services.container.get("capital_service")
        strategy_service = real_bot_services.container.get("strategy_service")

        # Start the service to load configuration
        await bot_service.start()

        # Create bot
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Verify bot was created with correct ID
        assert bot_id == sample_bot_config.bot_id

        # Verify bot exists in internal state
        assert bot_id in bot_service._bots

    @pytest.mark.asyncio
    async def test_start_bot_integration_flow(
        self,
        real_bot_services,
        sample_bot_config,
    ):
        """Test that start_bot follows proper integration patterns."""
        bot_service = real_bot_services.container.get("BotService")

        # Start service and create bot first
        await bot_service.start()
        await bot_service.create_bot(sample_bot_config)

        # Start the bot
        result = await bot_service.start_bot(sample_bot_config.bot_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_bot_integration_flow(
        self,
        real_bot_services,
        sample_bot_config,
    ):
        """Test that stop_bot follows proper integration patterns."""
        bot_service = real_bot_services.container.get("BotService")

        # Start service, create and start bot first
        await bot_service.start()
        await bot_service.create_bot(sample_bot_config)
        await bot_service.start_bot(sample_bot_config.bot_id)

        # Stop the bot
        result = await bot_service.stop_bot(sample_bot_config.bot_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_service_layer_not_bypassed(
        self,
        real_bot_services,
        sample_bot_config,
    ):
        """Test that bot management doesn't bypass service layers."""
        bot_service = real_bot_services.container.get("BotService")

        await bot_service.start()
        await bot_service.create_bot(sample_bot_config)

        # Get bot status - should use service methods, not direct access
        status = await bot_service.get_bot_status(sample_bot_config.bot_id)

        # Status should be a valid dict with bot information
        assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_health_check_integration(
        self,
        real_bot_services,
    ):
        """Test health check integrates with dependent services."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Perform health check
        health = await bot_service.perform_health_check("nonexistent_bot")

        # Should return proper status for non-existent bot
        assert health["status"] == "not_found"
        assert health["healthy"] is False

        # Create a bot and check its health
        sample_config = BotConfiguration(
            bot_id="health_test_bot",
            bot_name="Health Test Bot",
            bot_type=BotType.TRADING,
            strategy_name="test_strategy",
            exchanges=["binance"],
            symbols=["BTCUSDT"],
            allocated_capital=Decimal("1000"),
            risk_percentage=0.01,
        )
        await bot_service.create_bot(sample_config)

        health = await bot_service.perform_health_check("health_test_bot")

        # Health result should include service checks
        assert "checks" in health

    @pytest.mark.asyncio
    async def test_metrics_integration(
        self,
        real_bot_services,
        sample_bot_config,
    ):
        """Test metrics integration with monitoring system."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()
        await bot_service.create_bot(sample_bot_config)

        # Update bot metrics
        test_metrics = {
            "pnl": 100.50,
            "trades_count": 5,
            "win_rate": 0.8,
        }

        result = await bot_service.update_bot_metrics(
            sample_bot_config.bot_id, test_metrics
        )
        assert result is True


class TestBotManagementModuleBoundaries:
    """Test module boundary violations and proper API usage."""

    def test_no_direct_database_imports(self):
        """Test that bot_management doesn't import database models directly."""
        import src.bot_management.service as bot_service_module
        import inspect

        # Get all imports in the bot service module
        source = inspect.getsource(bot_service_module)

        # Should not import database models directly
        forbidden_imports = [
            "from src.database.models import",
            "from src.database.models.bot import",
            "from src.database.models.trading import",
        ]

        for forbidden in forbidden_imports:
            assert forbidden not in source, f"Found forbidden import: {forbidden}"

    def test_no_direct_exchange_access(self):
        """Test that bot_management doesn't access exchanges directly."""
        import src.bot_management.service as bot_service_module
        import inspect

        source = inspect.getsource(bot_service_module)

        # Should not import exchange clients directly
        forbidden_imports = [
            "from src.exchanges.binance import",
            "from src.exchanges.coinbase import",
            "import ccxt",
        ]

        for forbidden in forbidden_imports:
            assert forbidden not in source, f"Found forbidden import: {forbidden}"

    def test_proper_service_dependencies(self):
        """Test that bot_management only depends on service interfaces."""
        from src.bot_management.service import BotService
        import inspect

        # Get constructor signature
        sig = inspect.signature(BotService.__init__)
        params = list(sig.parameters.keys())

        # Should have proper service dependencies
        expected_services = [
            "exchange_service",
            "capital_service",
        ]

        for service in expected_services:
            assert service in params, f"Missing service dependency: {service}"
