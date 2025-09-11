"""
Simplified Exchange Factory.

This module provides a simple factory for creating exchange instances.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.base import BaseService
from src.core.config import Config
from src.core.dependency_injection import DependencyContainer
from src.core.exceptions import ExchangeError, ValidationError

# Logging provided by BaseService
# Import base exchange interface
from src.exchanges.base import BaseExchange
from src.exchanges.interfaces import IExchange, IExchangeFactory


class ExchangeFactory(BaseService, IExchangeFactory):
    """
    Simple factory for creating exchange instances.
    """

    def __init__(self, config: Config, container: DependencyContainer | None = None) -> None:
        """
        Initialize the exchange factory.

        Args:
            config: Application configuration
            container: Dependency container instance (optional, will create if None)
        """
        super().__init__(name="exchange_factory")
        self.config = config
        self.container = container or DependencyContainer()

        # Registry of supported exchanges
        self._exchange_registry: dict[str, type[BaseExchange]] = {}

        self.logger.info("Initialized exchange factory")

    async def _do_start(self) -> None:
        """Initialize factory service."""
        self.logger.info("Exchange factory started")

    async def _do_stop(self) -> None:
        """Cleanup factory resources."""
        self.logger.info("Exchange factory stopped")

    def register_exchange(
        self,
        exchange_name: str,
        exchange_class: type[BaseExchange],
    ) -> None:
        """
        Register an exchange implementation.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance')
            exchange_class: Exchange class that inherits from BaseExchange
        """
        if not issubclass(exchange_class, BaseExchange):
            raise ValidationError("Exchange class must inherit from BaseExchange")

        self._exchange_registry[exchange_name] = exchange_class
        self.logger.info(f"Registered exchange: {exchange_name}")

    def get_supported_exchanges(self) -> list[str]:
        """
        Get list of supported exchanges.

        Returns:
            List[str]: List of supported exchange names
        """
        return list(self._exchange_registry.keys())

    def is_exchange_supported(self, exchange_name: str) -> bool:
        """
        Check if an exchange is supported.

        Args:
            exchange_name: Name of the exchange to check

        Returns:
            bool: True if supported, False otherwise
        """
        return exchange_name in self._exchange_registry

    async def create_exchange(self, exchange_name: str) -> BaseExchange:
        """
        Create a new exchange instance.

        Args:
            exchange_name: Name of the exchange to create

        Returns:
            BaseExchange: Exchange instance

        Raises:
            ValidationError: If exchange is not supported
            ExchangeError: If exchange creation fails
        """
        if not self.is_exchange_supported(exchange_name):
            raise ValidationError(
                f"Exchange '{exchange_name}' is not supported. "
                f"Supported exchanges: {self.get_supported_exchanges()}"
            )

        try:
            exchange_class = self._exchange_registry[exchange_name]

            # Handle different constructor signatures
            if exchange_name == "mock":
                exchange = exchange_class(self.config)
            else:
                exchange = exchange_class(exchange_name, self.config)

            self.logger.info(f"Created exchange: {exchange_name}")
            return exchange

        except Exception as e:
            self.logger.error(f"Failed to create exchange {exchange_name}: {e}")
            raise ExchangeError(f"Exchange creation failed: {e}") from e

    def get_available_exchanges(self) -> list[str]:
        """
        Get list of configured exchanges.
        
        Returns:
            List[str]: List of configured exchange names
        """
        # Return supported exchanges as they are configured
        return self.get_supported_exchanges()

    async def get_exchange(
        self, exchange_name: str, create_if_missing: bool = True, force_recreate: bool = False
    ) -> IExchange | None:
        """
        Get or create exchange instance.
        
        Args:
            exchange_name: Name of the exchange
            create_if_missing: Whether to create if doesn't exist
            force_recreate: Whether to force recreation
            
        Returns:
            Optional[IExchange]: Exchange instance or None if not available
        """
        try:
            # Check if exchange is supported
            if not self.is_exchange_supported(exchange_name):
                if not create_if_missing:
                    return None
                raise ValidationError(f"Exchange '{exchange_name}' is not supported")

            # Create new exchange instance - factory doesn't cache instances
            # Caching is handled by the service layer
            if create_if_missing:
                return await self.create_exchange(exchange_name)

            return None

        except (ValidationError, ExchangeError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to get exchange {exchange_name}: {e}")
            raise ExchangeError(f"Exchange retrieval failed: {e}") from e

    async def remove_exchange(self, exchange_name: str) -> bool:
        """
        Remove exchange instance.
        
        Args:
            exchange_name: Name of the exchange to remove
            
        Returns:
            bool: True if removed successfully
        """
        # Factory doesn't cache instances, so nothing to remove
        # This is handled by the service layer
        self.logger.debug(f"Remove exchange called for {exchange_name} (no-op in factory)")
        return True

    async def health_check_all(self) -> dict[str, Any]:
        """
        Health check all exchanges.
        
        Returns:
            dict[str, Any]: Health check results
        """
        health_results = {
            "factory_status": "healthy",
            "supported_exchanges": self.get_supported_exchanges(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return health_results

    async def disconnect_all(self) -> None:
        """
        Disconnect all exchanges.
        
        Factory doesn't manage connections - this is a no-op.
        Connection management is handled by the service layer.
        """
        self.logger.debug("Disconnect all called on factory (no-op)")

    def register_default_exchanges(self) -> None:
        """Register default exchange implementations."""
        try:
            # Import exchange implementations
            from src.exchanges.binance import BinanceExchange
            from src.exchanges.coinbase import CoinbaseExchange
            from src.exchanges.mock_exchange import MockExchange
            from src.exchanges.okx import OKXExchange

            # Register implementations
            self.register_exchange("binance", BinanceExchange)
            self.register_exchange("coinbase", CoinbaseExchange)
            self.register_exchange("okx", OKXExchange)
            self.register_exchange("mock", MockExchange)

            self.logger.info("Default exchanges registered successfully")

        except Exception as e:
            self.logger.error(f"Failed to register default exchanges: {e}")
            raise ExchangeError(f"Default exchange registration failed: {e}") from e
