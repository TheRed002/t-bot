#!/usr/bin/env python3
"""
Test script to verify mock mode functionality.
"""
import os
import asyncio
from decimal import Decimal
import pytest

# Set mock mode
os.environ["MOCK_MODE"] = "true"

from src.exchanges.factory import ExchangeFactory
from src.exchanges import register_exchanges
from src.core.config import Config
from src.core.types import OrderSide, OrderType


@pytest.mark.asyncio
async def test_mock_exchange():
    """Test mock exchange functionality."""
    print("Testing T-Bot mock mode...")
    
    # Initialize configuration
    config = Config()
    print(f"✓ Config loaded, MOCK_MODE: {os.getenv('MOCK_MODE')}")
    
    # Initialize exchange factory
    factory = ExchangeFactory(config)
    register_exchanges(factory)
    
    supported = factory.get_supported_exchanges()
    print(f"✓ Supported exchanges: {supported}")
    
    # Test creating a mock exchange
    exchange = await factory.get_exchange("mock")
    print(f"✓ Mock exchange created: {type(exchange).__name__}")
    
    # Test basic exchange operations
    print("\nTesting exchange operations...")
    
    # Get balances
    balances = await exchange.get_balance()
    print(f"✓ Balances: {list(balances.keys())}")
    
    # Get ticker
    ticker = await exchange.get_ticker("BTC/USDT")
    print(f"✓ BTC/USDT ticker: ${ticker.last_price}")
    
    # Get order book
    order_book = await exchange.get_order_book("BTC/USDT", limit=5)
    print(f"✓ Order book: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
    
    # Place a mock order
    order = await exchange.place_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.001")
    )
    print(f"✓ Order placed: {order.id} - {order.status}")
    
    # Get order status
    updated_order = await exchange.get_order(order.id)
    print(f"✓ Order status: {updated_order.status}")
    
    # Check updated balances
    new_balances = await exchange.get_balance()
    print(f"✓ Updated balances checked")
    
    # Disconnect
    await exchange.disconnect()
    print("✓ Exchange disconnected")
    
    print("\n🎉 Mock mode test completed successfully!")
    print("✅ T-Bot can run without real API keys!")


if __name__ == "__main__":
    asyncio.run(test_mock_exchange())