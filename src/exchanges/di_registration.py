"""
Dependency Injection Registration for Exchange Module

This module provides simplified dependency injection registration for exchange services.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from src.core.config import Config
from src.core.dependency_injection import DependencyContainer
from src.core.logging import get_logger

# Import exchange implementations
from src.exchanges.binance import BinanceExchange
from src.exchanges.coinbase import CoinbaseExchange
from src.exchanges.factory import ExchangeFactory
from src.exchanges.mock_exchange import MockExchange
from src.exchanges.okx import OKXExchange
from src.exchanges.service import ExchangeService

logger = get_logger(__name__)


def register_exchange_dependencies(
    container: DependencyContainer, config: Config
) -> None:
    """
    Register exchange dependencies in the DI container.

    Args:
        container: The dependency injection container
        config: Application configuration
    """
    # Register configuration
    container.register("config", config, singleton=True)

    # Register factory
    def create_exchange_factory():
        factory = ExchangeFactory(config, container)
        # Register default exchanges during creation
        try:
            factory.register_default_exchanges()
        except Exception as e:
            logger.warning(f"Could not register default exchanges: {e}")

        return factory

    container.register("exchange_factory", create_exchange_factory, singleton=True)

    # Register service
    def create_exchange_service():
        factory = container.get("exchange_factory")
        return ExchangeService(factory, config)

    container.register("exchange_service", create_exchange_service, singleton=True)

    # Register individual exchanges
    container.register("binance_exchange", lambda: BinanceExchange("binance", config))
    container.register("coinbase_exchange", lambda: CoinbaseExchange("coinbase", config))
    container.register("okx_exchange", lambda: OKXExchange("okx", config))
    container.register("mock_exchange", lambda: MockExchange(config))

    logger.info("Exchange dependencies registered successfully")


def setup_exchange_services(config: Config) -> DependencyContainer:
    """
    Set up exchange services with dependency injection.

    Args:
        config: Application configuration

    Returns:
        DependencyContainer: Configured container with exchange services
    """
    container = DependencyContainer()
    register_exchange_dependencies(container, config)
    return container
