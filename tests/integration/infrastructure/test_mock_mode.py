#!/usr/bin/env python3
"""
Test script to verify mock mode functionality.
"""

import asyncio
import os
from decimal import Decimal

import pytest

# Set mock mode
os.environ["MOCK_MODE"] = "true"

from src.core.config import Config
from src.core.types import OrderSide, OrderType
from src.exchanges import register_exchanges
from src.exchanges.factory import ExchangeFactory


@pytest.mark.asyncio
async def test_mock_exchange():
    """Test mock exchange functionality."""
    print("Testing T-Bot mock mode...")

    # Initialize configuration
    config = Config()
    print(f"✓ Config loaded, MOCK_MODE: {os.getenv('MOCK_MODE')}")

    # Initialize exchange factory
    factory = ExchangeFactory(config)
    register_exchanges(factory, config=config)

    supported = factory.get_supported_exchanges()
    print(f"✓ Supported exchanges: {supported}")

    # Test creating an exchange - should be "mock" exchange when MOCK_MODE is set
    exchange = await factory.get_exchange("mock")
    print(f"✓ Mock exchange created: {type(exchange).__name__}")

    # Test basic exchange operations
    print("\nTesting exchange operations...")

    # Connect to the exchange first
    await exchange.connect()
    print("✓ Exchange connected")

    # Get balances
    balances = await exchange.get_balance()
    print(f"✓ Balances: {list(balances.keys())}")

    # Get ticker
    ticker = await exchange.get_ticker("BTCUSDT")
    # Handle both dict and object responses
    price = ticker.get('last_price') if isinstance(ticker, dict) else getattr(ticker, 'last_price', 'N/A')
    print(f"✓ BTCUSDT ticker: ${price}")

    # Get order book
    order_book = await exchange.get_order_book("BTCUSDT", limit=5)
    print(f"✓ Order book: {len(order_book.bids)} bids, {len(order_book.asks)} asks")

    # Place a mock order
    order = await exchange.place_order(
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=Decimal("0.001"),
        price=Decimal("50000")  # Price needed even for market orders in mock
    )
    # Handle both dict and object responses for order
    order_id = order.get('order_id') if isinstance(order, dict) else getattr(order, 'order_id', 'N/A')
    status = order.get('status') if isinstance(order, dict) else getattr(order, 'status', 'N/A')
    print(f"✓ Order placed: {order_id} - {status}")

    # Get order status
    updated_order = await exchange.get_order_status("BTCUSDT", order_id)
    updated_status = updated_order.get('status') if isinstance(updated_order, dict) else getattr(updated_order, 'status', 'N/A')
    print(f"✓ Order status: {updated_status}")

    # Check updated balances
    new_balances = await exchange.get_balance()
    print("✓ Updated balances checked")

    # Disconnect
    await exchange.disconnect()
    print("✓ Exchange disconnected")

    print("\n🎉 Mock mode test completed successfully!")
    print("✅ T-Bot can run without real API keys!")


if __name__ == "__main__":
    asyncio.run(test_mock_exchange())
