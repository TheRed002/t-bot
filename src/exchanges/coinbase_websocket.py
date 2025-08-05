"""
Coinbase WebSocket Handler (P-006)

This module implements the Coinbase-specific WebSocket client for real-time data streaming,
including ticker updates, order book changes, and trade notifications.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any
from decimal import Decimal
from datetime import datetime, timezone
import logging

# MANDATORY: Import from P-001
from src.core.types import (
    MarketData, Ticker, OrderBook, Trade, OrderStatus
)
from src.core.exceptions import (
    ExchangeError, ExchangeConnectionError, ExchangeRateLimitError
)
from src.core.config import Config

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# Coinbase-specific imports
from coinbase.websocket import WSClient
from coinbase.websocket import WSClientException, WSClientConnectionClosedException

logger = logging.getLogger(__name__)


class CoinbaseWebSocketHandler:
    """
    Coinbase WebSocket handler for real-time data streaming.
    
    Provides real-time market data, order updates, and account notifications
    through Coinbase's WebSocket API.
    """
    
    def __init__(self, config: Config, exchange_name: str = "coinbase"):
        """
        Initialize Coinbase WebSocket handler.
        
        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "coinbase")
        """
        self.config = config
        self.exchange_name = exchange_name
        self.error_handler = ErrorHandler(config.error_handling)
        
        # Coinbase-specific configuration
        self.api_key = config.exchanges.coinbase_api_key
        self.api_secret = config.exchanges.coinbase_api_secret
        self.sandbox = config.exchanges.coinbase_sandbox
        
        # WebSocket client
        self.ws_client: Optional[WSClient] = None
        
        # Stream management
        self.active_streams: Dict[str, Any] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.connected = False
        self.last_heartbeat = None
        
        # Message queue for reconnection
        self.message_queue: List[Dict] = []
        self.max_queue_size = 1000
        
        logger.info(f"Initialized {exchange_name} WebSocket handler")
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Coinbase.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize WebSocket client
            self.ws_client = WSClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                sandbox=self.sandbox
            )
            
            # Open connection
            await self.ws_client.open()
            
            self.connected = True
            self.last_heartbeat = datetime.now(timezone.utc)
            
            logger.info(f"Successfully connected to {self.exchange_name} WebSocket")
            return True
            
        except (WSClientException, WSClientConnectionClosedException) as e:
            logger.error(f"Failed to connect to {self.exchange_name} WebSocket: {str(e)}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to {self.exchange_name} WebSocket: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Coinbase WebSocket."""
        try:
            if self.ws_client:
                await self.ws_client.close()
            
            self.connected = False
            self.active_streams.clear()
            
            logger.info(f"Disconnected from {self.exchange_name} WebSocket")
            
        except (WSClientException, WSClientConnectionClosedException) as e:
            logger.error(f"Error disconnecting from {self.exchange_name} WebSocket: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error disconnecting from {self.exchange_name} WebSocket: {str(e)}")
    
    async def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to ticker updates for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            callback: Callback function to handle ticker updates
        """
        try:
            if not self.ws_client:
                raise ExchangeConnectionError("WebSocket not connected")
            
            # Subscribe to ticker channel
            await self.ws_client.ticker(symbol, callback)
            
            # Track subscription
            stream_key = f"ticker_{symbol}"
            self.active_streams[stream_key] = {
                "type": "ticker",
                "symbol": symbol,
                "callback": callback
            }
            
            if symbol not in self.callbacks:
                self.callbacks[symbol] = []
            self.callbacks[symbol].append(callback)
            
            logger.info(f"Subscribed to ticker stream for {symbol}")
            
        except (WSClientException, WSClientConnectionClosedException) as e:
            logger.error(f"Failed to subscribe to ticker for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to subscribe to ticker: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error subscribing to ticker for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to subscribe to ticker: {str(e)}")
    
    async def subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to order book updates for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            callback: Callback function to handle order book updates
        """
        try:
            if not self.ws_client:
                raise ExchangeConnectionError("WebSocket not connected")
            
            # Subscribe to level2 channel (order book)
            await self.ws_client.level2(symbol, callback)
            
            # Track subscription
            stream_key = f"orderbook_{symbol}"
            self.active_streams[stream_key] = {
                "type": "orderbook",
                "symbol": symbol,
                "callback": callback
            }
            
            if symbol not in self.callbacks:
                self.callbacks[symbol] = []
            self.callbacks[symbol].append(callback)
            
            logger.info(f"Subscribed to order book stream for {symbol}")
            
        except (WSClientException, WSClientConnectionClosedException) as e:
            logger.error(f"Failed to subscribe to order book for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to subscribe to order book: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error subscribing to order book for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to subscribe to order book: {str(e)}")
    
    async def subscribe_to_trades(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to trade updates for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            callback: Callback function to handle trade updates
        """
        try:
            if not self.ws_client:
                raise ExchangeConnectionError("WebSocket not connected")
            
            # Subscribe to matches channel (trades)
            await self.ws_client.matches(symbol, callback)
            
            # Track subscription
            stream_key = f"trades_{symbol}"
            self.active_streams[stream_key] = {
                "type": "trades",
                "symbol": symbol,
                "callback": callback
            }
            
            if symbol not in self.callbacks:
                self.callbacks[symbol] = []
            self.callbacks[symbol].append(callback)
            
            logger.info(f"Subscribed to trades stream for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to trades for {symbol}: {str(e)}")
            raise ExchangeError(f"Failed to subscribe to trades: {str(e)}")
    
    async def subscribe_to_user_data(self, callback: Callable) -> None:
        """
        Subscribe to user data updates (orders, balances).
        
        Args:
            callback: Callback function to handle user data updates
        """
        try:
            if not self.ws_client:
                raise ExchangeConnectionError("WebSocket not connected")
            
            # Subscribe to user channel
            await self.ws_client.user(callback)
            
            # Track subscription
            stream_key = "user_data"
            self.active_streams[stream_key] = {
                "type": "user_data",
                "callback": callback
            }
            
            logger.info("Subscribed to user data stream")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to user data: {str(e)}")
            raise ExchangeError(f"Failed to subscribe to user data: {str(e)}")
    
    async def unsubscribe_from_stream(self, stream_key: str) -> bool:
        """
        Unsubscribe from a specific stream.
        
        Args:
            stream_key: Key of the stream to unsubscribe from
            
        Returns:
            bool: True if unsubscribed successfully, False otherwise
        """
        try:
            if stream_key not in self.active_streams:
                logger.warning(f"Stream {stream_key} not found in active streams")
                return False
            
            stream_info = self.active_streams[stream_key]
            
            # Unsubscribe based on stream type
            if stream_info["type"] == "ticker":
                await self.ws_client.ticker_unsubscribe(stream_info["symbol"])
            elif stream_info["type"] == "orderbook":
                await self.ws_client.level2_unsubscribe(stream_info["symbol"])
            elif stream_info["type"] == "trades":
                await self.ws_client.matches_unsubscribe(stream_info["symbol"])
            elif stream_info["type"] == "user_data":
                await self.ws_client.user_unsubscribe()
            
            # Remove from tracking
            del self.active_streams[stream_key]
            
            logger.info(f"Unsubscribed from {stream_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {stream_key}: {str(e)}")
            return False
    
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all active streams."""
        try:
            for stream_key in list(self.active_streams.keys()):
                await self.unsubscribe_from_stream(stream_key)
            
            self.active_streams.clear()
            self.callbacks.clear()
            
            logger.info("Unsubscribed from all streams")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from all streams: {str(e)}")
    
    async def handle_ticker_message(self, message: Dict) -> None:
        """
        Handle ticker message from WebSocket.
        
        Args:
            message: Ticker message from Coinbase
        """
        try:
            # Convert to unified Ticker format
            ticker = Ticker(
                symbol=message.get('product_id', ''),
                bid=Decimal(str(message.get('bid', '0'))),
                ask=Decimal(str(message.get('ask', '0'))),
                last_price=Decimal(str(message.get('price', '0'))),
                volume_24h=Decimal(str(message.get('volume_24h', '0'))),
                price_change_24h=Decimal(str(message.get('price_change_24h', '0'))),
                timestamp=datetime.fromisoformat(message.get('time', '').replace('Z', '+00:00'))
            )
            
            # Call registered callbacks
            symbol = message.get('product_id', '')
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        await callback(ticker)
                    except Exception as e:
                        logger.error(f"Error in ticker callback: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error handling ticker message: {str(e)}")
    
    async def handle_orderbook_message(self, message: Dict) -> None:
        """
        Handle order book message from WebSocket.
        
        Args:
            message: Order book message from Coinbase
        """
        try:
            # Convert to unified OrderBook format
            order_book = OrderBook(
                symbol=message.get('product_id', ''),
                bids=[[Decimal(str(level[0])), Decimal(str(level[1]))] for level in message.get('bids', [])],
                asks=[[Decimal(str(level[0])), Decimal(str(level[1]))] for level in message.get('asks', [])],
                timestamp=datetime.now(timezone.utc)
            )
            
            # Call registered callbacks
            symbol = message.get('product_id', '')
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        await callback(order_book)
                    except Exception as e:
                        logger.error(f"Error in order book callback: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error handling order book message: {str(e)}")
    
    async def handle_trade_message(self, message: Dict) -> None:
        """
        Handle trade message from WebSocket.
        
        Args:
            message: Trade message from Coinbase
        """
        try:
            # Convert to unified Trade format
            trade = Trade(
                id=message.get('trade_id', ''),
                symbol=message.get('product_id', ''),
                side=message.get('side', 'buy').lower(),  # Convert to lowercase for consistency
                quantity=Decimal(str(message.get('size', '0'))),
                price=Decimal(str(message.get('price', '0'))),
                timestamp=datetime.fromisoformat(message.get('time', '').replace('Z', '+00:00')),
                fee=Decimal("0"),  # Coinbase doesn't provide fee in trade data
                fee_currency="USD"
            )
            
            # Call registered callbacks
            symbol = message.get('product_id', '')
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        await callback(trade)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error handling trade message: {str(e)}")
    
    async def handle_user_message(self, message: Dict) -> None:
        """
        Handle user data message from WebSocket.
        
        Args:
            message: User data message from Coinbase
        """
        try:
            # Handle different types of user messages
            message_type = message.get('channel', {}).get('name', '')
            
            if message_type == 'orders':
                # Handle order updates
                await self._handle_order_update(message)
            elif message_type == 'accounts':
                # Handle account updates
                await self._handle_account_update(message)
            
            # Call registered callbacks
            if 'user_data' in self.callbacks:
                for callback in self.callbacks['user_data']:
                    try:
                        await callback(message)
                    except Exception as e:
                        logger.error(f"Error in user data callback: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error handling user message: {str(e)}")
    
    async def _handle_order_update(self, message: Dict) -> None:
        """Handle order update message."""
        try:
            order_id = message.get('order_id', '')
            status = message.get('status', '')
            
            logger.info(f"Order {order_id} status updated to {status}")
            
        except Exception as e:
            logger.error(f"Error handling order update: {str(e)}")
    
    async def _handle_account_update(self, message: Dict) -> None:
        """Handle account update message."""
        try:
            account_id = message.get('account_id', '')
            balance = message.get('balance', {})
            
            logger.info(f"Account {account_id} balance updated")
            
        except Exception as e:
            logger.error(f"Error handling account update: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Perform health check on WebSocket connection.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self.ws_client or not self.connected:
                return False
            
            # Check if connection is still alive
            # Coinbase WebSocket doesn't have a specific health check method
            # We'll rely on the connection state and last heartbeat
            
            if self.last_heartbeat:
                time_since_heartbeat = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
                if time_since_heartbeat > 60:  # 1 minute timeout
                    logger.warning("WebSocket heartbeat timeout")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"WebSocket health check failed: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if WebSocket is connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and self.ws_client is not None
    
    def get_active_streams(self) -> Dict[str, Any]:
        """
        Get information about active streams.
        
        Returns:
            Dict[str, Any]: Dictionary of active streams
        """
        return self.active_streams.copy()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect() 