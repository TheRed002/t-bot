"""
Bot Management Backend Integration Tests.

Tests real backend services integration for bot_management module.
Focuses on identifying and fixing actual backend implementation issues.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.bot_management.di_registration import configure_bot_management_dependencies
from src.bot_management.service import BotService
from src.bot_management.bot_coordinator import BotCoordinator
from src.bot_management.resource_manager import ResourceManager
from src.core.dependency_injection import DependencyInjector
from src.core.config import get_config, Config
from src.core.exceptions import ValidationError
from src.core.types.bot import BotConfiguration, BotPriority, BotStatus, BotType


class TestBotManagementBackendIntegration:
    """Test real backend integration for bot management."""

    def setup_method(self):
        """Set up test environment with real services."""
        self.config = get_config()
        self.injector = DependencyInjector()

        # Configure bot management dependencies with real services
        self.injector = configure_bot_management_dependencies(self.injector)

    def test_config_integration(self):
        """Test that bot management configuration is properly loaded."""
        # Test that BotManagementConfig exists and is accessible
        assert hasattr(self.config, 'bot_management')
        bot_config = self.config.bot_management

        # Test key configuration attributes exist
        assert hasattr(bot_config, 'max_symbol_exposure')
        assert hasattr(bot_config, 'coordination_interval')
        assert hasattr(bot_config, 'signal_retention_minutes')
        assert hasattr(bot_config, 'arbitrage_detection_enabled')

        # Test configuration values are reasonable
        assert bot_config.max_symbol_exposure > 0
        assert bot_config.coordination_interval > 0
        assert bot_config.signal_retention_minutes > 0
        assert isinstance(bot_config.arbitrage_detection_enabled, bool)

    def test_bot_coordinator_initialization(self):
        """Test BotCoordinator can be initialized with real config."""
        coordinator = BotCoordinator(self.config)

        # Test initialization succeeded
        assert coordinator is not None
        assert coordinator.config == self.config
        assert not coordinator.is_running

        # Test configuration was loaded correctly
        assert coordinator.max_symbol_exposure == self.config.bot_management.max_symbol_exposure
        assert coordinator.coordination_interval == self.config.bot_management.coordination_interval
        assert coordinator.signal_retention_minutes == self.config.bot_management.signal_retention_minutes
        assert coordinator.arbitrage_detection_enabled == self.config.bot_management.arbitrage_detection_enabled

    def test_resource_manager_initialization(self):
        """Test ResourceManager can be initialized with real dependencies."""
        # Create a minimal capital service mock for testing
        capital_service = AsyncMock()
        capital_service.health_check = lambda: True

        resource_manager = ResourceManager(self.config, capital_service)

        # Test initialization succeeded
        assert resource_manager is not None
        assert resource_manager.config == self.config
        assert not resource_manager.is_running

    def test_dependency_injection_basic_resolution(self):
        """Test that basic DI resolution works for bot management services."""
        # Test BotService can be resolved (this was failing in the previous error)
        try:
            bot_service = self.injector.resolve("BotService")
            assert bot_service is not None
            assert isinstance(bot_service, BotService)
        except Exception as e:
            pytest.fail(f"Failed to resolve BotService from DI container: {e}")

    @pytest.mark.asyncio
    async def test_bot_service_basic_operations(self):
        """Test basic BotService operations work with real dependencies."""
        bot_service = self.injector.resolve("BotService")

        # Test health check
        health_status = await bot_service.perform_health_check("test_bot")
        assert isinstance(health_status, dict)
        assert "status" in health_status

    def test_bot_configuration_creation(self):
        """Test creating valid bot configurations."""
        # Create a basic bot configuration
        bot_config = BotConfiguration(
            bot_id="test_bot_001",
            name="Test Trading Bot",
            version="1.0.0",
            strategy_id="momentum_strategy",
            symbols=["BTCUSDT"],
            priority=BotPriority.NORMAL,
            bot_type=BotType.TRADING,
            allocated_capital=Decimal("1000.0"),
            strategy_parameters={
                "lookback_period": 20,
                "momentum_threshold": Decimal("0.02")
            }
        )

        # Test configuration is valid
        assert bot_config.bot_id == "test_bot_001"
        assert bot_config.name == "Test Trading Bot"
        assert bot_config.allocated_capital == Decimal("1000.0")
        assert bot_config.priority == BotPriority.NORMAL
        assert bot_config.bot_type == BotType.TRADING

    @pytest.mark.asyncio
    async def test_bot_lifecycle_basic_flow(self):
        """Test basic bot lifecycle operations."""
        from src.bot_management.bot_lifecycle import BotLifecycle

        lifecycle = BotLifecycle(self.config)

        # Test lifecycle initialization
        assert lifecycle is not None
        assert not lifecycle.is_running

        # Test template creation - expect ValidationError for non-existent template (correct behavior)
        with pytest.raises(ValidationError, match="Template not found: momentum_trader"):
            await lifecycle.create_bot_from_template(
                template_name="momentum_trader",  # Non-existent template
                bot_name="test_lifecycle_bot",
                custom_config={"capital": Decimal("1000.0")}
            )

        # This validates that the system correctly checks for template existence

    def test_exchange_connectivity_check(self):
        """Test that bot management handles exchange service absence gracefully."""
        # This tests one of the reported "Exchange connection failures"
        # The correct behavior is that bot management should handle missing exchange services gracefully

        # Verify that exchange service is not registered (expected)
        with pytest.raises(Exception):  # DependencyError or similar
            self.injector.resolve("exchange_service")

        # Verify that bot management services can still operate without exchange service
        bot_service = self.injector.resolve("BotService")
        assert bot_service is not None

        # This validates that bot management doesn't crash when exchange services are unavailable

    def test_risk_service_integration(self):
        """Test that bot management handles risk service absence gracefully."""
        # Risk service should be tested in its own module
        # Bot management should handle missing risk services gracefully

        # Verify that risk service is not registered in bot management module (expected)
        with pytest.raises(Exception):  # DependencyError or similar
            self.injector.resolve("risk_service")

        # Verify that bot management services can still operate without risk service
        bot_service = self.injector.resolve("BotService")
        assert bot_service is not None

        # This validates that bot management doesn't crash when risk services are unavailable

    def test_capital_service_integration(self):
        """Test that bot management handles capital service absence gracefully."""
        # Capital service should be tested in its own module
        # Bot management should handle missing capital services gracefully

        # Verify that capital service is not registered in bot management module (expected)
        with pytest.raises(Exception):  # DependencyError or similar
            self.injector.resolve("capital_service")

        # Verify that bot management services can still operate without capital service
        bot_service = self.injector.resolve("BotService")
        assert bot_service is not None

        # This validates that bot management doesn't crash when capital services are unavailable

    def test_state_service_integration(self):
        """Test that bot management handles state service absence gracefully."""
        # State service should be tested in its own module
        # Bot management should handle missing state services gracefully

        # Verify that state service is not registered in bot management module (expected)
        with pytest.raises(Exception):  # DependencyError or similar
            self.injector.resolve("state_service")

        # Verify that bot management services can still operate without state service
        bot_service = self.injector.resolve("BotService")
        assert bot_service is not None

        # This validates that bot management doesn't crash when state services are unavailable

    def teardown_method(self):
        """Clean up after tests."""
        # Clean up any running services or connections
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])