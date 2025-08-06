"""
Coinbase Exchange Implementation (P-006)

This module implements the Coinbase-specific exchange client with full API integration,
including REST API client, WebSocket streams, and rate limiting.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.
"""

import asyncio
import json
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Callable, Any
from decimal import Decimal
from datetime import datetime, timezone

# MANDATORY: Import from P-001
from src.core.types import (
    OrderRequest, OrderResponse, MarketData, Position,
    Signal, TradingMode, OrderSide, OrderType,
    ExchangeInfo, Ticker, OrderBook, Trade, OrderStatus
)
from src.core.exceptions import (
    ExchangeError, ExchangeConnectionError, ExchangeRateLimitError,
    ExchangeInsufficientFundsError, ValidationError, ExecutionError
)
from src.core.config import Config

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-003
from src.exchanges.base import BaseExchange
from src.exchanges.rate_limiter import RateLimiter
from src.exchanges.connection_manager import ConnectionManager

# MANDATORY: Import from P-007A (utils)
from src.utils.constants import API_ENDPOINTS, RATE_LIMITS, TIMEOUTS

# Coinbase-specific imports
import aiohttp
import websockets
from coinbase.rest import RESTClient
from coinbase.websocket import WSClient
# Note: Using generic Exception handling for REST API as no specific exceptions are documented
# For WebSocket, use WSClientException and WSClientConnectionClosedException as per documentation
from src.core.logging import get_logger

logger = get_logger(__name__)


class CoinbaseExchange(BaseExchange):
    """
    Coinbase exchange implementation.
    
    Implements the unified exchange interface for Coinbase Advanced API, providing:
    - REST API client with async support
    - WebSocket stream management
    - Rate limiting and error handling
    - Order management and balance tracking
    
    CRITICAL: This class must inherit from BaseExchange and implement all abstract methods.
    """
    
    def __init__(self, config: Config, exchange_name: str = "coinbase"):
        """
        Initialize Coinbase exchange.
        
        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "coinbase")
        """
        super().__init__(config, exchange_name)
        
        # Coinbase-specific configuration
        self.api_key = config.exchanges.coinbase_api_key
        self.api_secret = config.exchanges.coinbase_api_secret
        self.sandbox = config.exchanges.coinbase_sandbox
        
        # Coinbase API URLs from constants
        coinbase_config = API_ENDPOINTS["coinbase"]
        if self.sandbox:
            self.base_url = coinbase_config["sandbox_url"]
            self.ws_url = coinbase_config["ws_url"]
        else:
            self.base_url = coinbase_config["base_url"]
            self.ws_url = coinbase_config["ws_url"]
        
        # Initialize Coinbase client
        self.client: Optional[RESTClient] = None
        self.ws_client: Optional[WSClient] = None
        
        # Initialize rate limiter for Coinbase-specific limits
        self.rate_limiter = RateLimiter(config, "coinbase")
        
        # Initialize connection manager
        self.connection_manager = ConnectionManager(config, exchange_name)
        
        # WebSocket streams
        self.active_streams: Dict[str, Any] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Order tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.order_status_cache: Dict[str, OrderStatus] = {}
        
        # Balance cache
        self.balance_cache: Dict[str, Decimal] = {}
        self.last_balance_update = None
        
        logger.info(f"Initialized {exchange_name} exchange interface")
    
    async def connect(self) -> bool:
        """
        Establish connection to Coinbase exchange.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize REST client with sandbox support
            base_url = "api-public.sandbox.exchange.coinbase.com" if self.sandbox else "api.coinbase.com"
            self.client = RESTClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_url=base_url
            )
            
            # Test connection by getting account info
            await self._test_connection()
            
            # Initialize WebSocket client with sandbox support
            ws_base_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com" if self.sandbox else "wss://advanced-trade-ws.coinbase.com"
            self.ws_client = WSClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_url=ws_base_url
            )
            
            # Initialize WebSocket connection
            await self._initialize_websocket()
            
            self.connected = True
            self.status = "connected"
            self.last_heartbeat = datetime.now(timezone.utc)
            
            logger.info(f"Successfully connected to {self.exchange_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_name}: {str(e)}")
            self.connected = False
            self.status = "connection_failed"
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Coinbase exchange."""
        try:
            # Close WebSocket connections
            if self.ws_client:
                await self.ws_client.close()
            
            # Close REST client
            if self.client:
                await self.client.close()
            
            self.connected = False
            self.status = "disconnected"
            
            logger.info(f"Disconnected from {self.exchange_name}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.exchange_name}: {str(e)}")
    
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """
        Get all asset balances from Coinbase exchange.
        
        Returns:
            Dict[str, Decimal]: Dictionary mapping asset symbols to balances
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Apply rate limiting
            await self.rate_limiter.acquire("requests_per_minute", 1)
            
            # Get accounts from Coinbase
            accounts = await self.client.get_accounts()
            
            balances = {}
            for account in accounts:
                currency = account['currency']
                available = Decimal(str(account.get('available_balance', {}).get('value', '0')))
                hold = Decimal(str(account.get('hold', {}).get('value', '0')))
                total = available + hold
                
                if total > 0:
                    balances[currency] = total
            
            # Update cache
            self.balance_cache = balances
            self.last_balance_update = datetime.now(timezone.utc)
            
            logger.debug(f"Retrieved balances for {len(balances)} currencies")
            return balances
            
        except ExchangeConnectionError:
            # Re-raise connection errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to get account balance: {str(e)}")
            raise ExchangeError(f"Failed to get account balance: {str(e)}")
    
    @time_execution
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place an order on Coinbase exchange.
        
        Args:
            order: Order request with all necessary details
            
        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Validate order
            if not await self.pre_trade_validation(order):
                raise ValidationError("Order validation failed")
            
            # Apply rate limiting
            await self.rate_limiter.acquire("orders_per_second", 1)
            
            # Convert order to Coinbase format
            coinbase_order = self._convert_order_to_coinbase(order)
            
            # Place order
            result = await self.client.create_order(**coinbase_order)
            
            # Convert response to unified format
            order_response = self._convert_coinbase_order_to_response(result)
            
            # Track order
            self.pending_orders[order_response.id] = order
            self.order_status_cache[order_response.id] = OrderStatus.PENDING
            
            # Post-trade processing
            await self.post_trade_processing(order_response)
            
            logger.info(f"Placed order {order_response.id} for {order.symbol}")
            return order_response
            
        except ExchangeConnectionError:
            # Re-raise connection errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            raise ExecutionError(f"Failed to place order: {str(e)}")
    
    @time_execution
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on Coinbase exchange.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Apply rate limiting
            await self.rate_limiter.acquire("orders_per_second", 1)
            
            # Cancel order
            result = await self.client.cancel_orders([order_id])
            
            # Update cache
            if order_id in self.order_status_cache:
                self.order_status_cache[order_id] = OrderStatus.CANCELLED
            
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
            logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    @time_execution
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get the status of an order on Coinbase exchange.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            OrderStatus: Current status of the order
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get order details
            order = await self.client.get_order(order_id)
            
            # Convert status
            status = self._convert_coinbase_status_to_order_status(order['status'])
            
            # Update cache
            self.order_status_cache[order_id] = status
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            return OrderStatus.UNKNOWN
    
    @time_execution
    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """
        Get market data for a symbol from Coinbase exchange.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            timeframe: Timeframe for the data (e.g., "1m", "1h", "1d")
            
        Returns:
            MarketData: Market data with price and volume information
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get product details
            product = await self.client.get_product(symbol)
            
            # Get latest ticker
            ticker = await self.client.get_product_ticker(symbol)
            
            # Get recent candles for OHLCV data
            candles = await self.client.get_product_candles(
                product_id=symbol,
                granularity=self._convert_timeframe_to_granularity(timeframe),
                limit=1
            )
            
            # Build market data
            market_data = MarketData(
                symbol=symbol,
                price=Decimal(str(ticker['price'])),
                volume=Decimal(str(ticker['volume_24h'])),
                timestamp=datetime.fromisoformat(ticker['time'].replace('Z', '+00:00')),
                bid=Decimal(str(ticker['bid'])) if ticker.get('bid') else None,
                ask=Decimal(str(ticker['ask'])) if ticker.get('ask') else None,
                open_price=Decimal(str(candles[0]['open'])) if candles else None,
                high_price=Decimal(str(candles[0]['high'])) if candles else None,
                low_price=Decimal(str(candles[0]['low'])) if candles else None
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to get market data: {str(e)}")
    
    async def subscribe_to_stream(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to real-time data stream for a symbol.
        
        Args:
            symbol: Trading symbol to subscribe to
            callback: Callback function to handle stream data
        """
        try:
            if not self.ws_client:
                raise ExchangeConnectionError("WebSocket not connected")
            
            # Subscribe to ticker channel
            await self.ws_client.ticker(symbol, callback)
            
            # Track subscription
            if symbol not in self.callbacks:
                self.callbacks[symbol] = []
            self.callbacks[symbol].append(callback)
            
            logger.info(f"Subscribed to {symbol} stream")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol} stream: {str(e)}")
            raise ExchangeError(f"Failed to subscribe to stream: {str(e)}")
    
    @time_execution
    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """
        Get order book for a symbol from Coinbase exchange.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            depth: Number of levels to retrieve
            
        Returns:
            OrderBook: Order book with bid and ask levels
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get order book
            book = await self.client.get_product_book(symbol, depth)
            
            # Convert to unified format
            order_book = OrderBook(
                symbol=symbol,
                bids=[[Decimal(str(level[0])), Decimal(str(level[1]))] for level in book['bids'][:depth]],
                asks=[[Decimal(str(level[0])), Decimal(str(level[1]))] for level in book['asks'][:depth]],
                timestamp=datetime.now(timezone.utc)
            )
            
            return order_book
            
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to get order book: {str(e)}")
    
    @time_execution
    async def get_trade_history(self, symbol: str, limit: int = 100) -> List[Trade]:
        """
        Get trade history for a symbol from Coinbase exchange.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            limit: Maximum number of trades to retrieve
            
        Returns:
            List[Trade]: List of recent trades
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get recent trades
            trades_data = await self.client.get_product_trades(symbol, limit)
            
            # Convert to unified format
            trades = []
            for trade_data in trades_data:
                trade = Trade(
                    id=trade_data['trade_id'],
                    symbol=symbol,
                    side=OrderSide.BUY if trade_data['side'] == 'buy' else OrderSide.SELL,
                    quantity=Decimal(str(trade_data['size'])),
                    price=Decimal(str(trade_data['price'])),
                    timestamp=datetime.fromisoformat(trade_data['time'].replace('Z', '+00:00')),
                    fee=Decimal("0"),  # Coinbase doesn't provide fee in trade data
                    fee_currency="USD"
                )
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get trade history for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to get trade history: {str(e)}")
    
    @time_execution
    async def get_exchange_info(self) -> ExchangeInfo:
        """
        Get exchange information including supported symbols and features.
        
        Returns:
            ExchangeInfo: Exchange information
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get all products
            products = await self.client.get_products()
            
            # Extract supported symbols
            supported_symbols = [product['product_id'] for product in products if product['status'] == 'online']
            
            # Build exchange info
            exchange_info = ExchangeInfo(
                name="coinbase",
                supported_symbols=supported_symbols,
                rate_limits={
                    "requests_per_minute": 600,
                    "orders_per_second": 15,
                    "websocket_connections": 4
                },
                features=["spot_trading", "websocket_streams", "order_book", "trade_history"],
                api_version="v3"
            )
            
            return exchange_info
            
        except Exception as e:
            logger.error(f"Failed to get exchange info: {str(e)}")
            raise ExchangeError(f"Failed to get exchange info: {str(e)}")
    
    @time_execution
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get real-time ticker information for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            
        Returns:
            Ticker: Real-time ticker information
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get ticker
            ticker_data = await self.client.get_product_ticker(symbol)
            
            # Convert to unified format
            ticker = Ticker(
                symbol=symbol,
                bid=Decimal(str(ticker_data['bid'])) if ticker_data.get('bid') else Decimal("0"),
                ask=Decimal(str(ticker_data['ask'])) if ticker_data.get('ask') else Decimal("0"),
                last_price=Decimal(str(ticker_data['price'])),
                volume_24h=Decimal(str(ticker_data['volume_24h'])),
                price_change_24h=Decimal(str(ticker_data.get('price_change_24h', '0'))),
                timestamp=datetime.fromisoformat(ticker_data['time'].replace('Z', '+00:00'))
            )
            
            return ticker
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to get ticker: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Perform health check on Coinbase connection.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Test connection by getting products (this should always work)
            await self.client.get_products()
            
            self.last_heartbeat = datetime.now(timezone.utc)
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            return False
    
    def get_rate_limits(self) -> Dict[str, int]:
        """
        Get current rate limits for Coinbase.
        
        Returns:
            Dict[str, int]: Rate limits dictionary
        """
        return {
            "requests_per_minute": 600,
            "orders_per_second": 15,
            "websocket_connections": 4
        }
    
    # Helper methods
    
    async def _test_connection(self) -> None:
        """Test connection to Coinbase API."""
        try:
            # Test connection by getting products (this should always work)
            await self.client.get_products()
        except Exception as e:
            raise ExchangeConnectionError(f"Failed to connect to Coinbase: {str(e)}")
    
    async def _initialize_websocket(self) -> None:
        """Initialize WebSocket connection."""
        try:
            await self.ws_client.open()
            logger.info("WebSocket connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {str(e)}")
    
    def _convert_order_to_coinbase(self, order: OrderRequest) -> Dict[str, Any]:
        """Convert unified order to Coinbase format."""
        coinbase_order = {
            "product_id": order.symbol,
            "side": order.side.value,
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": str(order.quantity)
                } if order.order_type == OrderType.MARKET else {
                    "limit_limit_gtc": {
                        "base_size": str(order.quantity),
                        "limit_price": str(order.price)
                    }
                }
            }
        }
        
        if order.client_order_id:
            coinbase_order["client_order_id"] = order.client_order_id
        
        return coinbase_order
    
    def _convert_coinbase_order_to_response(self, result: Dict) -> OrderResponse:
        """Convert Coinbase order response to unified format."""
        return OrderResponse(
            id=result['order_id'],
            client_order_id=result.get('client_order_id'),
            symbol=result['product_id'],
            side=OrderSide.BUY if result['side'] == 'BUY' else OrderSide.SELL,
            order_type=OrderType.MARKET if result['order_configuration'].get('market_market_ioc') else OrderType.LIMIT,
            quantity=Decimal(str(result.get('filled_size', '0'))),
            price=Decimal(str(result.get('limit_price', '0'))) if result.get('limit_price') else None,
            filled_quantity=Decimal(str(result.get('filled_size', '0'))),
            status=result['status'],
            timestamp=datetime.fromisoformat(result['created_time'].replace('Z', '+00:00'))
        )
    
    def _convert_coinbase_status_to_order_status(self, status: str) -> OrderStatus:
        """Convert Coinbase order status to unified OrderStatus."""
        status_mapping = {
            'OPEN': OrderStatus.PENDING,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'EXPIRED': OrderStatus.EXPIRED,
            'REJECTED': OrderStatus.REJECTED,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED
        }
        return status_mapping.get(status, OrderStatus.UNKNOWN)
    
    def _convert_timeframe_to_granularity(self, timeframe: str) -> str:
        """Convert unified timeframe to Coinbase granularity."""
        mapping = {
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "1h": "ONE_HOUR",
            "6h": "SIX_HOUR",
            "1d": "ONE_DAY"
        }
        return mapping.get(timeframe, "ONE_MINUTE") 