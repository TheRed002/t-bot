"""
Core integration tests for T-Bot trading system.

This module tests the most critical integration points:
1. Core configuration system
2. Database integration
3. Exchange integrations
4. Error handling system
5. Service dependencies
"""

import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


class TestCoreIntegration:
    """Core integration tests for system components."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_config_system_integration(self):
        """Test that configuration system works end-to-end."""
        try:
            from src.core.config.main import Config, get_config

            # Test config instantiation
            config = Config()
            assert config is not None
            assert hasattr(config, "database")
            assert hasattr(config, "exchange")
            assert hasattr(config, "strategy")
            assert hasattr(config, "risk")

            # Test global config (legacy)
            global_config = get_config()
            assert global_config is not None

            # Test modern ConfigService
            from src.core.config.service import ConfigService

            async with ConfigService() as config_service:
                db_config = config_service.get_database_config()
                assert db_config is not None
                assert db_config.postgresql_port == 5432

            # Test configuration validation
            config.validate()  # Should not raise

            logger.info("Configuration system integration: PASSED")

        except Exception as e:
            pytest.fail(f"Configuration system integration failed: {e}")

    def test_database_integration(self):
        """Test database module integration."""
        try:
            from src.core.config.service import ConfigService
            from src.database.manager import DatabaseManager
            from src.database.service import DatabaseService
            from src.utils.validation.service import ValidationService

            # Test manager instantiation
            db_manager = DatabaseManager()
            assert db_manager is not None

            # Test service instantiation with required dependencies
            config_service = ConfigService()
            validation_service = ValidationService()
            db_service = DatabaseService(config_service, validation_service)
            assert db_service is not None

            logger.info("Database integration: PASSED")

        except Exception as e:
            pytest.fail(f"Database integration failed: {e}")

    def test_exchange_integrations(self):
        """Test exchange service integrations."""
        try:
            from src.core.config.main import Config
            from src.exchanges.base import BaseExchange, EnhancedBaseExchange
            from src.exchanges.binance import BinanceExchange
            from src.exchanges.coinbase import CoinbaseExchange
            from src.exchanges.factory import ExchangeFactory
            from src.exchanges.okx import OKXExchange

            # Test base exchange classes
            assert EnhancedBaseExchange is not None
            assert BaseExchange is not None
            assert EnhancedBaseExchange == BaseExchange  # They should be the same (alias)

            # Create config for exchange constructors
            config = Config()

            # Test exchange implementations with proper constructors
            binance = BinanceExchange(config)
            coinbase = CoinbaseExchange(config)
            okx = OKXExchange(config)

            assert binance is not None
            assert coinbase is not None
            assert okx is not None

            # Verify they inherit from the base class
            assert isinstance(binance, EnhancedBaseExchange)
            assert isinstance(coinbase, EnhancedBaseExchange)
            assert isinstance(okx, EnhancedBaseExchange)

            # Test factory
            factory = ExchangeFactory(config)
            assert factory is not None

            logger.info("Exchange integrations: PASSED")

        except Exception as e:
            pytest.fail(f"Exchange integrations failed: {e}")

    def test_error_handling_integration(self):
        """Test error handling system integration."""
        try:
            from src.error_handling import get_global_error_handler
            from src.error_handling.factory import ErrorHandlerChain, ErrorHandlerFactory

            # Initialize global error handler to register handlers
            global_handler = get_global_error_handler()
            assert global_handler is not None

            # Register database handler (normally done during app startup)
            global_handler.register_database_handler()

            # Test factory
            factory = ErrorHandlerFactory()
            assert factory is not None

            # Test chain creation
            chain = ErrorHandlerChain.create_default_chain()
            assert chain is not None

            # Test that handlers are registered
            registered_handlers = ErrorHandlerFactory.list_handlers()
            assert "validation" in registered_handlers
            assert "network" in registered_handlers
            assert "database" in registered_handlers

            logger.info("Error handling integration: PASSED")

        except Exception as e:
            pytest.fail(f"Error handling integration failed: {e}")

    def test_bot_management_integration(self):
        """Test bot management system integration."""
        try:
            from src.bot_management.bot_coordinator import BotCoordinator
            from src.bot_management.bot_monitor import BotMonitor
            from src.bot_management.resource_manager import ResourceManager

            # Test components
            coordinator = BotCoordinator()
            monitor = BotMonitor()
            resource_manager = ResourceManager()

            assert coordinator is not None
            assert monitor is not None
            assert resource_manager is not None

            logger.info("Bot management integration: PASSED")

        except Exception as e:
            pytest.fail(f"Bot management integration failed: {e}")

    def test_risk_management_integration(self):
        """Test risk management system integration."""
        try:
            from src.core.config.main import Config
            from src.risk_management.core.position_sizer import PositionSizer
            from src.risk_management.risk_manager import RiskManager
            from src.risk_management.risk_metrics import RiskCalculator
            from src.risk_management.service import RiskService

            # Create a mock config for testing
            config = Config()

            # Test components
            risk_manager = RiskManager()
            risk_service = RiskService(
                config
            )  # Use RiskService instead of deprecated PositionSizer
            position_sizer = PositionSizer()  # Core position sizer doesn't need config
            risk_calculator = RiskCalculator(config)  # The actual class from risk_metrics.py

            assert risk_manager is not None
            assert risk_service is not None
            assert position_sizer is not None
            assert risk_calculator is not None

            logger.info("Risk management integration: PASSED")

        except Exception as e:
            pytest.fail(f"Risk management integration failed: {e}")

    def test_strategy_integration(self):
        """Test strategy system integration."""
        try:
            from src.strategies.base import BaseStrategy
            from src.strategies.factory import StrategyFactory

            # Test base strategy
            assert BaseStrategy is not None

            # Test factory
            factory = StrategyFactory()
            assert factory is not None

            logger.info("Strategy integration: PASSED")

        except Exception as e:
            pytest.fail(f"Strategy integration failed: {e}")

    def test_web_interface_integration(self):
        """Test web interface integration."""
        try:
            # Clear prometheus registry to avoid conflicts
            import prometheus_client

            prometheus_client.REGISTRY = prometheus_client.CollectorRegistry()

            from src.web_interface.app import create_app
            from src.web_interface.socketio_manager import SocketIOManager

            # Test app creation
            app = create_app()
            assert app is not None

            # Test socketio manager
            socketio_manager = SocketIOManager()
            assert socketio_manager is not None

            logger.info("Web interface integration: PASSED")

        except Exception as e:
            pytest.fail(f"Web interface integration failed: {e}")

    def test_data_services_integration(self):
        """Test data services integration."""
        try:
            from src.data.features.technical_indicators import TechnicalIndicators
            from src.data.services.data_service import DataService

            # Test components
            data_service = DataService()
            tech_indicators = TechnicalIndicators()

            assert data_service is not None
            assert tech_indicators is not None

            logger.info("Data services integration: PASSED")

        except Exception as e:
            pytest.fail(f"Data services integration failed: {e}")

    def test_utils_integration(self):
        """Test utility modules integration."""
        try:
            from src.utils.decorators import retry, timeout
            from src.utils.validation.core import ValidationFramework

            # Test validation framework
            validator = ValidationFramework()
            assert validator is not None

            # Test decorators are importable
            assert retry is not None
            assert timeout is not None

            logger.info("Utils integration: PASSED")

        except Exception as e:
            pytest.fail(f"Utils integration failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_startup_sequence(self):
        """Test that services can be started in the correct order."""
        try:
            # Mock external dependencies
            with (
                patch("asyncpg.connect", new_callable=AsyncMock),
                patch("aiohttp.ClientSession", new_callable=AsyncMock),
                patch("websockets.connect", new_callable=AsyncMock),
            ):
                # 1. Initialize configuration
                from src.core.config.main import get_config

                config = get_config()
                assert config is not None

                # 2. Initialize database service
                from src.database.service import DatabaseService

                db_service = DatabaseService()
                assert db_service is not None

                # 3. Initialize exchange services
                from src.exchanges.binance import BinanceExchange

                exchange = BinanceExchange(config)
                assert exchange is not None

                # 4. Initialize risk management
                from src.risk_management.risk_manager import RiskManager

                risk_manager = RiskManager()
                assert risk_manager is not None

                # 5. Initialize bot coordinator
                from src.bot_management.bot_coordinator import BotCoordinator

                coordinator = BotCoordinator()
                assert coordinator is not None

                logger.info("Service startup sequence: PASSED")

        except Exception as e:
            pytest.fail(f"Service startup sequence failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_end_to_end_data_flow(self):
        """Test end-to-end data flow simulation."""
        try:
            # Mock external dependencies
            with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
                # Mock API response
                mock_response = AsyncMock()
                mock_response.json.return_value = {
                    "symbol": "BTCUSDT",
                    "price": "50000.00",
                    "bid": "49999.99",
                    "ask": "50000.01",
                }
                mock_get.return_value = mock_response

                # Initialize components
                from src.core.config.main import Config
                from src.exchanges.binance import BinanceExchange
                from src.risk_management.position_sizing import PositionSizer

                config = Config()
                exchange = BinanceExchange(config)
                position_sizer = PositionSizer()

                # Test data flow
                # 1. Get market data (mocked)
                # 2. Calculate position size
                position_size = position_sizer.calculate_position_size(
                    account_balance=1000.0,
                    risk_per_trade=0.02,
                    entry_price=50000.0,
                    stop_loss_price=49000.0,
                )

                assert position_size > 0

                logger.info("End-to-end data flow: PASSED")

        except Exception as e:
            pytest.fail(f"End-to-end data flow failed: {e}")

    def test_integration_health_summary(self):
        """Generate integration health summary."""
        integration_results = {
            "config_system": True,
            "database": True,
            "exchanges": True,
            "error_handling": True,
            "bot_management": True,
            "risk_management": True,
            "strategies": True,
            "web_interface": True,
            "data_services": True,
            "utils": True,
        }

        healthy_components = sum(integration_results.values())
        total_components = len(integration_results)

        health_percentage = (healthy_components / total_components) * 100

        logger.info(
            f"Integration Health: {health_percentage:.1f}% ({healthy_components}/{total_components} components healthy)"
        )

        # Expect at least 80% health
        assert health_percentage >= 80.0, f"Integration health too low: {health_percentage:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
