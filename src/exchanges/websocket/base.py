"""Base WebSocket manager for exchanges."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Set, Optional, Callable, Any
import websockets
from websockets.client import WebSocketClientProtocol

from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseWebSocketManager(ABC):
    """
    Shared WebSocket management logic for all exchanges.
    
    This eliminates duplication of WebSocket connection handling,
    reconnection logic, and subscription management.
    """
    
    def __init__(
        self,
        url: str,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 10,
        ping_interval: int = 30
    ):
        """
        Initialize WebSocket manager.
        
        Args:
            url: WebSocket URL
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
            ping_interval: Ping interval in seconds
        """
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.ping_interval = ping_interval
        
        self.ws: Optional[WebSocketClientProtocol] = None
        self.subscriptions: Set[str] = set()
        self.handlers: Dict[str, Callable] = {}
        self._running = False
        self._reconnect_count = 0
        self._logger = logger
        self._tasks: Set[asyncio.Task] = set()
    
    async def connect(self) -> None:
        """
        Connect to WebSocket with automatic reconnection.
        """
        self._running = True
        self._reconnect_count = 0
        
        while self._running and self._reconnect_count < self.max_reconnect_attempts:
            try:
                self._logger.info(f"Connecting to WebSocket: {self.url}")
                self.ws = await websockets.connect(self.url)
                self._reconnect_count = 0  # Reset on successful connection
                
                # Call connection hook
                await self._on_connect()
                
                # Resubscribe to channels
                await self._resubscribe()
                
                # Start ping task
                ping_task = asyncio.create_task(self._ping_loop())
                self._tasks.add(ping_task)
                
                # Start listening
                await self._listen()
                
            except websockets.exceptions.WebSocketException as e:
                self._logger.error(f"WebSocket error: {e}")
                await self._handle_disconnect()
                
            except Exception as e:
                self._logger.error(f"Unexpected error: {e}")
                await self._handle_disconnect()
            
            if self._running and self._reconnect_count < self.max_reconnect_attempts:
                self._logger.info(
                    f"Reconnecting in {self.reconnect_delay}s "
                    f"(attempt {self._reconnect_count}/{self.max_reconnect_attempts})"
                )
                await asyncio.sleep(self.reconnect_delay)
        
        if self._reconnect_count >= self.max_reconnect_attempts:
            self._logger.error("Max reconnection attempts reached")
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        if self.ws:
            await self.ws.close()
            self.ws = None
        
        self._logger.info("Disconnected from WebSocket")
    
    async def _listen(self) -> None:
        """Listen for messages from WebSocket."""
        if not self.ws:
            return
        
        try:
            async for message in self.ws:
                if not self._running:
                    break
                
                try:
                    # Parse message
                    data = self._parse_message(message)
                    
                    # Route to appropriate handler
                    await self._route_message(data)
                    
                except Exception as e:
                    self._logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self._logger.warning("WebSocket connection closed")
            await self._handle_disconnect()
        
        except Exception as e:
            self._logger.error(f"Listen error: {e}")
            await self._handle_disconnect()
    
    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while self._running:
            try:
                if self.ws and not self.ws.closed:
                    await self.ws.ping()
                    self._logger.debug("Ping sent")
                
                await asyncio.sleep(self.ping_interval)
                
            except Exception as e:
                self._logger.error(f"Ping error: {e}")
                break
    
    async def _handle_disconnect(self) -> None:
        """Handle disconnection and prepare for reconnection."""
        self._reconnect_count += 1
        
        if self.ws:
            await self.ws.close()
            self.ws = None
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        
        await self._on_disconnect()
    
    async def _resubscribe(self) -> None:
        """Resubscribe to all channels after reconnection."""
        for channel in self.subscriptions.copy():
            await self._send_subscribe(channel)
            self._logger.debug(f"Resubscribed to {channel}")
    
    async def subscribe(self, channel: str, handler: Callable) -> None:
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel to subscribe to
            handler: Handler function for channel messages
        """
        self.subscriptions.add(channel)
        self.handlers[channel] = handler
        
        if self.ws and not self.ws.closed:
            await self._send_subscribe(channel)
        
        self._logger.info(f"Subscribed to {channel}")
    
    async def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from a channel.
        
        Args:
            channel: Channel to unsubscribe from
        """
        self.subscriptions.discard(channel)
        self.handlers.pop(channel, None)
        
        if self.ws and not self.ws.closed:
            await self._send_unsubscribe(channel)
        
        self._logger.info(f"Unsubscribed from {channel}")
    
    async def _route_message(self, data: Dict[str, Any]) -> None:
        """
        Route message to appropriate handler.
        
        Args:
            data: Parsed message data
        """
        channel = self._extract_channel(data)
        
        if channel and channel in self.handlers:
            handler = self.handlers[channel]
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
                    
            except Exception as e:
                self._logger.error(f"Handler error for {channel}: {e}")
        else:
            # Default handler for unmatched messages
            await self._on_message(data)
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """
        Send a message through WebSocket.
        
        Args:
            message: Message to send
        """
        if not self.ws or self.ws.closed:
            raise ConnectionError("WebSocket not connected")
        
        message_str = json.dumps(message)
        await self.ws.send(message_str)
        self._logger.debug(f"Sent message: {message_str}")
    
    # Abstract methods for exchange-specific implementation
    
    @abstractmethod
    def _parse_message(self, message: str) -> Dict[str, Any]:
        """
        Parse raw message from WebSocket.
        
        Args:
            message: Raw message string
            
        Returns:
            Parsed message data
        """
        pass
    
    @abstractmethod
    def _extract_channel(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract channel from message data.
        
        Args:
            data: Message data
            
        Returns:
            Channel name or None
        """
        pass
    
    @abstractmethod
    async def _send_subscribe(self, channel: str) -> None:
        """
        Send subscription message for a channel.
        
        Args:
            channel: Channel to subscribe to
        """
        pass
    
    @abstractmethod
    async def _send_unsubscribe(self, channel: str) -> None:
        """
        Send unsubscription message for a channel.
        
        Args:
            channel: Channel to unsubscribe from
        """
        pass
    
    # Hooks for subclasses
    
    async def _on_connect(self) -> None:
        """Hook called when connected."""
        pass
    
    async def _on_disconnect(self) -> None:
        """Hook called when disconnected."""
        pass
    
    async def _on_message(self, data: Dict[str, Any]) -> None:
        """
        Hook for handling unmatched messages.
        
        Args:
            data: Message data
        """
        self._logger.debug(f"Unhandled message: {data}")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.ws is not None and not self.ws.closed
    
    def get_subscriptions(self) -> Set[str]:
        """Get current subscriptions."""
        return self.subscriptions.copy()