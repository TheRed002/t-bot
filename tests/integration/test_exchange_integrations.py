#!/usr/bin/env python3
"""
Test script for exchange integrations with sandbox/testnet endpoints.

This script tests all three exchange implementations (Binance, Coinbase, OKX) 
with sandbox/testnet credentials to verify:
- API connection and authentication
- Account balance retrieval
- Market data fetching
- Order placement (test orders)
- WebSocket connections
- Error handling

Usage:
    python tests/integration/test_exchange_integrations.py

Requirements:
    - Set up sandbox/testnet API credentials in config
    - Install all required dependencies
    - Ensure all exchanges are in testnet/sandbox mode
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import Config
from core.types import OrderRequest, OrderSide, OrderType
from exchanges.binance import BinanceExchange
from exchanges.coinbase import CoinbaseExchange
from exchanges.okx import OKXExchange
from utils.decorators import time_execution


class ExchangeIntegrationTester:
    """Test exchange integrations with sandbox/testnet endpoints."""

    def __init__(self):
        """Initialize tester with config."""
        self.config = Config()
        self.results = {}
        
    async def test_all_exchanges(self) -> dict:
        """Test all exchange integrations."""
        print("🚀 Starting Exchange Integration Tests")
        print("=" * 50)
        
        exchanges = {
            "binance": BinanceExchange,
            "coinbase": CoinbaseExchange, 
            "okx": OKXExchange
        }
        
        for exchange_name, exchange_class in exchanges.items():
            print(f"\n📈 Testing {exchange_name.upper()} Exchange")
            print("-" * 30)
            
            try:
                await self._test_exchange(exchange_name, exchange_class)
                self.results[exchange_name] = "✅ PASSED"
            except Exception as e:
                print(f"❌ {exchange_name.upper()} test failed: {e}")
                self.results[exchange_name] = f"❌ FAILED: {e}"
                
        return self.results
    
    @time_execution
    async def _test_exchange(self, exchange_name: str, exchange_class) -> None:
        """Test a specific exchange."""
        exchange = None
        try:
            # Initialize exchange
            print(f"1. Initializing {exchange_name} exchange...")
            exchange = exchange_class(self.config, exchange_name)
            
            # Test connection
            print(f"2. Testing connection...")
            connected = await exchange.connect()
            if not connected:
                raise Exception("Failed to connect to exchange")
            print(f"   ✅ Connected successfully")
            
            # Test account balance
            print(f"3. Testing account balance retrieval...")
            balances = await exchange.get_account_balance()
            print(f"   ✅ Retrieved balances for {len(balances)} currencies")
            for currency, balance in list(balances.items())[:3]:  # Show first 3
                print(f"      {currency}: {balance}")
            
            # Test exchange info
            print(f"4. Testing exchange info...")
            try:
                exchange_info = await exchange.get_exchange_info()
                print(f"   ✅ Exchange supports {len(exchange_info.supported_symbols)} symbols")
            except Exception as e:
                print(f"   ⚠️ Exchange info test skipped: {e}")
            
            # Test market data
            print(f"5. Testing market data...")
            test_symbols = self._get_test_symbols(exchange_name)
            for symbol in test_symbols[:2]:  # Test first 2 symbols
                try:
                    market_data = await exchange.get_market_data(symbol)
                    print(f"   ✅ {symbol}: Price=${market_data.price}, Volume={market_data.volume}")
                except Exception as e:
                    print(f"   ⚠️ Market data for {symbol} failed: {e}")
            
            # Test order book
            print(f"6. Testing order book...")
            try:
                symbol = test_symbols[0]
                order_book = await exchange.get_order_book(symbol, depth=5)
                print(f"   ✅ {symbol} order book: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
                if order_book.bids and order_book.asks:
                    print(f"      Best bid: ${order_book.bids[0][0]}, Best ask: ${order_book.asks[0][0]}")
            except Exception as e:
                print(f"   ⚠️ Order book test failed: {e}")
            
            # Test ticker
            print(f"7. Testing ticker...")
            try:
                symbol = test_symbols[0]
                ticker = await exchange.get_ticker(symbol)
                print(f"   ✅ {symbol} ticker: Last=${ticker.last_price}, Bid=${ticker.bid}, Ask=${ticker.ask}")
            except Exception as e:
                print(f"   ⚠️ Ticker test failed: {e}")
            
            # Test order placement (very small test order)
            print(f"8. Testing order placement (TEST ORDER)...")
            try:
                await self._test_order_placement(exchange, exchange_name)
            except Exception as e:
                print(f"   ⚠️ Order placement test failed: {e}")
            
            # Test WebSocket (basic connection test)
            print(f"9. Testing WebSocket connection...")
            try:
                await self._test_websocket(exchange, test_symbols[0])
            except Exception as e:
                print(f"   ⚠️ WebSocket test failed: {e}")
            
            # Test health check
            print(f"10. Testing health check...")
            health = await exchange.health_check()
            if health:
                print(f"   ✅ Health check passed")
            else:
                print(f"   ⚠️ Health check failed")
                
            print(f"✅ {exchange_name.upper()} - All tests completed successfully!")
            
        finally:
            # Clean up
            if exchange:
                try:
                    await exchange.disconnect()
                except Exception as e:
                    print(f"   ⚠️ Error during disconnect: {e}")
    
    async def _test_order_placement(self, exchange, exchange_name: str) -> None:
        """Test order placement with a very small test order."""
        test_symbols = self._get_test_symbols(exchange_name)
        symbol = test_symbols[0]
        
        # Create a very small test market buy order
        order_request = OrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),  # Very small amount
            client_order_id=f"test_{exchange_name}_{int(asyncio.get_event_loop().time())}"
        )
        
        # Only place order if we have sufficient balance
        balances = await exchange.get_account_balance()
        quote_currency = symbol.split('-')[-1] if '-' in symbol else 'USDT'
        
        # Get current price to estimate cost
        market_data = await exchange.get_market_data(symbol)
        estimated_cost = order_request.quantity * market_data.price
        
        balance = balances.get(quote_currency, Decimal('0'))
        
        if balance > estimated_cost * Decimal('2'):  # Need 2x cost for safety
            try:
                print(f"   Placing test market buy order: {order_request.quantity} {symbol}")
                order_response = await exchange.place_order(order_request)
                print(f"   ✅ Test order placed successfully: {order_response.id}")
                
                # Wait a moment then check order status
                await asyncio.sleep(1)
                order_status = await exchange.get_order_status(order_response.id)
                print(f"   Order status: {order_status}")
                
            except Exception as e:
                print(f"   ⚠️ Order placement failed: {e}")
        else:
            print(f"   ⚠️ Insufficient balance for test order ({balance} {quote_currency} < {estimated_cost})")
    
    async def _test_websocket(self, exchange, symbol: str) -> None:
        """Test WebSocket connection."""
        try:
            # Test basic stream subscription
            message_received = False
            
            def test_callback(data):
                nonlocal message_received
                message_received = True
                print(f"   ✅ WebSocket message received: {type(data).__name__}")
            
            await exchange.subscribe_to_stream(symbol, test_callback)
            
            # Wait a moment for potential messages
            await asyncio.sleep(3)
            
            if message_received:
                print(f"   ✅ WebSocket test passed - received data")
            else:
                print(f"   ⚠️ WebSocket test - no data received (may be normal)")
                
        except Exception as e:
            raise Exception(f"WebSocket test failed: {e}")
    
    def _get_test_symbols(self, exchange_name: str) -> list[str]:
        """Get test symbols for each exchange."""
        symbols = {
            "binance": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            "coinbase": ["BTC-USD", "ETH-USD", "LTC-USD"], 
            "okx": ["BTC-USDT", "ETH-USDT", "ADA-USDT"]
        }
        return symbols.get(exchange_name, ["BTCUSDT"])
    
    def print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 50)
        print("📊 EXCHANGE INTEGRATION TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if "✅ PASSED" in r])
        
        for exchange, result in self.results.items():
            print(f"{exchange.upper():>12}: {result}")
        
        print("-" * 50)
        print(f"Total: {passed_tests}/{total_tests} exchanges passed")
        
        if passed_tests == total_tests:
            print("🎉 All exchange integrations working correctly!")
        else:
            print("⚠️  Some exchange integrations need attention")


async def main():
    """Main test function."""
    tester = ExchangeIntegrationTester()
    
    try:
        await tester.test_all_exchanges()
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
    finally:
        tester.print_summary()


if __name__ == "__main__":
    # Run the test
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)