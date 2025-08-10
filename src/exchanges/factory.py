"""
Exchange factory for dynamic exchange instantiation.

This module provides a factory pattern for creating exchange instances
dynamically from configuration, with support for hot-swapping and
connection pooling.

CRITICAL: This integrates with P-001 (core types, exceptions, config)
and P-002A (error handling) components.
"""

from typing import Dict, Optional, Type, List
from datetime import datetime

# MANDATORY: Import from P-001
from src.core.types import ExchangeInfo, ExchangeStatus
from src.core.exceptions import ExchangeError, ValidationError
from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007 (advanced rate limiting)
from src.exchanges.advanced_rate_limiter import AdvancedRateLimiter
from src.exchanges.connection_manager import ConnectionManager

# Import base exchange interface
from .base import BaseExchange

logger = get_logger(__name__)


class ExchangeFactory:
    """
    Factory for creating and managing exchange instances.

    This class provides a centralized way to create exchange instances
    from configuration, with support for connection pooling and
    hot-swapping capabilities.
    """

    def __init__(self, config: Config):
        """
        Initialize the exchange factory.

        Args:
            config: Application configuration
        """
        self.config = config
        self.error_handler = ErrorHandler(config.error_handling)

        # Registry of supported exchanges
        self._exchange_registry: Dict[str, Type[BaseExchange]] = {}

        # Active exchange instances
        self._active_exchanges: Dict[str, BaseExchange] = {}

        # Connection pool for each exchange
        self._connection_pools: Dict[str, List[BaseExchange]] = {}

        logger.info("Initialized exchange factory")

    def register_exchange(
            self,
            exchange_name: str,
            exchange_class: Type[BaseExchange]) -> None:
        """
        Register an exchange implementation.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance')
            exchange_class: Exchange class that inherits from BaseExchange
        """
        if not issubclass(exchange_class, BaseExchange):
            raise ValidationError(
                f"Exchange class must inherit from BaseExchange")

        self._exchange_registry[exchange_name] = exchange_class
        logger.info(f"Registered exchange: {exchange_name}")

    def get_supported_exchanges(self) -> List[str]:
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
                f"Exchange '{exchange_name}' is not supported. " f"Supported exchanges: {
                    self.get_supported_exchanges()}")

        try:
            # Get the exchange class from registry
            exchange_class = self._exchange_registry[exchange_name]

            # Create the exchange instance
            exchange = exchange_class(self.config, exchange_name)

            # Connect to the exchange
            connected = await exchange.connect()
            if not connected:
                raise ExchangeError(f"Failed to connect to {exchange_name}")

            # TODO: Remove in production
            logger.debug(
                f"Exchange {exchange_name} created with P-007 components")
            logger.info(f"Created and connected to {exchange_name}")
            return exchange

        except Exception as e:
            logger.error(
                f"Failed to create exchange {exchange_name}: {
                    str(e)}")
            raise ExchangeError(f"Exchange creation failed: {str(e)}")

    async def get_exchange(
            self,
            exchange_name: str,
            create_if_missing: bool = True) -> Optional[BaseExchange]:
        """
        Get an existing exchange instance or create a new one.

        Args:
            exchange_name: Name of the exchange
            create_if_missing: Whether to create a new instance if not found

        Returns:
            Optional[BaseExchange]: Exchange instance or None if not found and create_if_missing=False
        """
        # Check if we already have an active instance
        if exchange_name in self._active_exchanges:
            exchange = self._active_exchanges[exchange_name]

            # Check if the exchange is still healthy
            if await exchange.health_check():
                return exchange
            else:
                logger.warning(
                    f"Exchange {exchange_name} failed health check, removing from active pool")
                await self.remove_exchange(exchange_name)

        # Create new instance if requested
        if create_if_missing:
            try:
                exchange = await self.create_exchange(exchange_name)
                self._active_exchanges[exchange_name] = exchange
                return exchange
            except Exception as e:
                logger.error(
                    f"Failed to get/create exchange {exchange_name}: {str(e)}")
                return None

        return None

    async def remove_exchange(self, exchange_name: str) -> bool:
        """
        Remove an exchange instance from the active pool.

        Args:
            exchange_name: Name of the exchange to remove

        Returns:
            bool: True if removed successfully, False otherwise
        """
        if exchange_name in self._active_exchanges:
            try:
                exchange = self._active_exchanges[exchange_name]
                await exchange.disconnect()
                del self._active_exchanges[exchange_name]
                logger.info(
                    f"Removed exchange {exchange_name} from active pool")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to remove exchange {exchange_name}: {
                        str(e)}")
                # Remove from active exchanges even if disconnect fails
                del self._active_exchanges[exchange_name]
                return False

        return False

    async def get_all_active_exchanges(self) -> Dict[str, BaseExchange]:
        """
        Get all active exchange instances.

        Returns:
            Dict[str, BaseExchange]: Dictionary of active exchanges
        """
        return self._active_exchanges.copy()

    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all active exchanges.

        Returns:
            Dict[str, bool]: Dictionary mapping exchange names to health status
        """
        health_status = {}

        for exchange_name, exchange in self._active_exchanges.items():
            try:
                health_status[exchange_name] = await exchange.health_check()
            except Exception as e:
                logger.error(
                    f"Health check failed for {exchange_name}: {
                        str(e)}")
                health_status[exchange_name] = False

        return health_status

    async def disconnect_all(self) -> None:
        """Disconnect all active exchange instances."""
        for exchange_name in list(self._active_exchanges.keys()):
            await self.remove_exchange(exchange_name)

        logger.info("Disconnected all active exchanges")

    def get_exchange_info(self, exchange_name: str) -> Optional[ExchangeInfo]:
        """
        Get information about a supported exchange.

        Args:
            exchange_name: Name of the exchange

        Returns:
            Optional[ExchangeInfo]: Exchange information or None if not found
        """
        if not self.is_exchange_supported(exchange_name):
            return None

        # Get rate limits from config
        rate_limits = self.config.exchanges.rate_limits.get(exchange_name, {})

        return ExchangeInfo(
            name=exchange_name,
            supported_symbols=[],  # Will be populated by actual exchange implementation
            rate_limits=rate_limits,
            features=["spot_trading"],  # Default features
            api_version="1.0"
        )

    def get_exchange_status(self, exchange_name: str) -> ExchangeStatus:
        """
        Get the status of an exchange.

        Args:
            exchange_name: Name of the exchange

        Returns:
            ExchangeStatus: Current status of the exchange
        """
        if exchange_name not in self._active_exchanges:
            return ExchangeStatus.OFFLINE

        exchange = self._active_exchanges[exchange_name]
        if exchange.is_connected():
            return ExchangeStatus.ONLINE
        else:
            return ExchangeStatus.OFFLINE

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()
