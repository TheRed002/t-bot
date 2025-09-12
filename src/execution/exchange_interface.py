"""
Exchange Interface Protocol for Execution Module.

This module defines the protocol that exchanges must implement
for use by the execution module. This provides type safety and
clear contract definition.
"""

from typing import Protocol

from src.core.types import (
    MarketData,
    OrderRequest,
    OrderResponse,
)


class ExchangeInterface(Protocol):
    """Protocol defining the exchange interface required by execution module."""

    @property
    def exchange_name(self) -> str:
        """Get the exchange name."""
        ...

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place an order on the exchange.

        Args:
            order: Order request details

        Returns:
            OrderResponse: Order response with ID and status

        Raises:
            ExchangeError: If order placement fails
            ExchangeConnectionError: If connection fails
            ExchangeRateLimitError: If rate limit exceeded
            ValidationError: If order validation fails
        """
        ...

    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse:
        """
        Get the status of an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to check

        Returns:
            OrderResponse: Order response with status and details

        Raises:
            ExchangeError: If status check fails
            ExchangeConnectionError: If connection fails
        """
        ...

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            OrderResponse: Cancellation response with status

        Raises:
            ExchangeError: If cancellation fails
            ExchangeConnectionError: If connection fails
        """
        ...

    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """
        Get market data for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            MarketData: Current market data

        Raises:
            ExchangeError: If data fetch fails
            ExchangeConnectionError: If connection fails
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the exchange connection is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        ...


class ExchangeFactoryInterface(Protocol):
    """Protocol defining the exchange factory interface."""

    async def get_exchange(self, exchange_name: str) -> ExchangeInterface:
        """
        Get an exchange instance by name.

        Args:
            exchange_name: Name of the exchange

        Returns:
            ExchangeInterface: Exchange instance

        Raises:
            ValidationError: If exchange not found
        """
        ...

    def get_available_exchanges(self) -> list[str]:
        """
        Get list of available exchange names.

        Returns:
            list[str]: Available exchange names
        """
        ...
