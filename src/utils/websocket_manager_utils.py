"""
WebSocket Management Utilities for T-Bot Trading System.

This module provides common utilities for WebSocket connection management,
subscription handling, message broadcasting, and reconnection logic across 
different WebSocket handlers and exchange implementations.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket

from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseWebSocketManager:
    """
    Base WebSocket connection manager with common functionality.

    This class provides the foundation for managing WebSocket connections
    with subscription support and message broadcasting capabilities.
    """

    def __init__(self, manager_name: str = "WebSocket"):
        self.manager_name = manager_name
        self.active_connections: dict[str, WebSocket] = {}
        self.user_subscriptions: dict[str, set[str]] = {}  # user_id -> set of subscription_keys
        self.subscription_users: dict[str, set[str]] = {}  # subscription_key -> set of user_ids

    async def connect(self, websocket: WebSocket, user_id: str) -> bool:
        """
        Accept WebSocket connection.

        Args:
            websocket: The WebSocket connection
            user_id: Unique user identifier

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            await websocket.accept()
            self.active_connections[user_id] = websocket
            self.user_subscriptions[user_id] = set()
            logger.info(f"{self.manager_name} WebSocket connected", user_id=user_id)
            return True
        except Exception as e:
            logger.error(
                f"Failed to accept {self.manager_name} WebSocket connection: {e}", user_id=user_id
            )
            return False

    def disconnect(self, user_id: str):
        """
        Disconnect WebSocket and cleanup subscriptions.

        Args:
            user_id: User identifier to disconnect
        """
        if user_id not in self.active_connections:
            return

        # Remove from all subscriptions
        for subscription_key in self.user_subscriptions.get(user_id, set()).copy():
            self.unsubscribe(user_id, subscription_key)

        # Remove user connection and subscriptions
        del self.active_connections[user_id]
        if user_id in self.user_subscriptions:
            del self.user_subscriptions[user_id]

        logger.info(f"{self.manager_name} WebSocket disconnected", user_id=user_id)

    def subscribe(self, user_id: str, subscription_key: str):
        """
        Subscribe user to updates for a specific key.

        Args:
            user_id: User identifier
            subscription_key: Key to subscribe to (e.g., symbol, bot_id)
        """
        if user_id not in self.active_connections:
            logger.warning(f"Cannot subscribe inactive user {user_id} to {subscription_key}")
            return

        # Add to user subscriptions
        if user_id not in self.user_subscriptions:
            self.user_subscriptions[user_id] = set()
        self.user_subscriptions[user_id].add(subscription_key)

        # Add to subscription users
        if subscription_key not in self.subscription_users:
            self.subscription_users[subscription_key] = set()
        self.subscription_users[subscription_key].add(user_id)

        logger.info(
            f"User subscribed to {subscription_key}",
            user_id=user_id,
            subscription_key=subscription_key,
        )

    def unsubscribe(self, user_id: str, subscription_key: str):
        """
        Unsubscribe user from updates for a specific key.

        Args:
            user_id: User identifier
            subscription_key: Key to unsubscribe from
        """
        # Remove from user subscriptions
        if user_id in self.user_subscriptions:
            self.user_subscriptions[user_id].discard(subscription_key)

        # Remove from subscription users
        if subscription_key in self.subscription_users:
            self.subscription_users[subscription_key].discard(user_id)
            if not self.subscription_users[subscription_key]:
                del self.subscription_users[subscription_key]

        logger.info(
            f"User unsubscribed from {subscription_key}",
            user_id=user_id,
            subscription_key=subscription_key,
        )

    async def send_personal_message(self, message: dict[str, Any], user_id: str) -> bool:
        """
        Send message to a specific user.

        Args:
            message: Message to send
            user_id: Target user identifier

        Returns:
            bool: True if sent successfully, False otherwise
        """
        if user_id not in self.active_connections:
            logger.warning(f"Cannot send message to inactive user {user_id}")
            return False

        try:
            websocket = self.active_connections[user_id]
            # Add timeout to prevent hanging on send
            await asyncio.wait_for(websocket.send_text(json.dumps(message)), timeout=10.0)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout sending message to user {user_id}")
            # Connection might be broken, disconnect user
            self.disconnect(user_id)
            return False
        except Exception as e:
            logger.error(f"Failed to send message to user {user_id}: {e}")
            # Connection might be broken, disconnect user
            self.disconnect(user_id)
            return False

    async def broadcast_to_subscription(self, message: dict[str, Any], subscription_key: str):
        """
        Broadcast message to all users subscribed to a specific key.

        Args:
            message: Message to broadcast
            subscription_key: Subscription key to broadcast to
        """
        subscribers = self.subscription_users.get(subscription_key, set())
        if not subscribers:
            return

        # Send message to all subscribers with concurrency limit
        max_concurrent = min(len(subscribers), 50)  # Limit concurrent sends
        semaphore = asyncio.Semaphore(max_concurrent)

        async def send_with_semaphore(user_id: str) -> bool:
            async with semaphore:
                return await self.send_personal_message(message, user_id)

        # Create tasks for concurrent sending
        tasks = [send_with_semaphore(user_id) for user_id in subscribers.copy()]

        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                failed_count = sum(
                    1 for result in results if isinstance(result, Exception) or not result
                )
                if failed_count > 0:
                    logger.warning(
                        f"Failed to send broadcast to {failed_count} users for {subscription_key}"
                    )
            except Exception as e:
                logger.error(f"Error during broadcast for {subscription_key}: {e}")

    async def broadcast_to_all(self, message: dict[str, Any]):
        """
        Broadcast message to all connected users.

        Args:
            message: Message to broadcast
        """
        if not self.active_connections:
            return

        # Get user list to avoid modification during iteration
        user_ids = list(self.active_connections.keys())

        # Send message to all users with concurrency limit
        max_concurrent = min(len(user_ids), 50)  # Limit concurrent sends
        semaphore = asyncio.Semaphore(max_concurrent)

        async def send_with_semaphore(user_id: str) -> bool:
            async with semaphore:
                return await self.send_personal_message(message, user_id)

        # Create tasks for concurrent sending
        tasks = [send_with_semaphore(user_id) for user_id in user_ids]

        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                failed_count = sum(
                    1 for result in results if isinstance(result, Exception) or not result
                )
                if failed_count > 0:
                    logger.warning(f"Failed to broadcast to {failed_count} users")
            except Exception as e:
                logger.error(f"Error during broadcast to all users: {e}")

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_subscription_count(self, subscription_key: str) -> int:
        """Get number of users subscribed to a specific key."""
        return len(self.subscription_users.get(subscription_key, set()))

    def get_user_subscriptions(self, user_id: str) -> set[str]:
        """Get all subscriptions for a user."""
        return self.user_subscriptions.get(user_id, set()).copy()

    def is_user_subscribed(self, user_id: str, subscription_key: str) -> bool:
        """Check if user is subscribed to a specific key."""
        return subscription_key in self.user_subscriptions.get(user_id, set())


class MarketDataWebSocketManager(BaseWebSocketManager):
    """WebSocket manager specifically for market data streaming."""

    def __init__(self):
        super().__init__("Market Data")

    def subscribe_to_symbol(self, user_id: str, symbol: str):
        """Subscribe user to symbol updates."""
        self.subscribe(user_id, f"symbol:{symbol}")

    def unsubscribe_from_symbol(self, user_id: str, symbol: str):
        """Unsubscribe user from symbol updates."""
        self.unsubscribe(user_id, f"symbol:{symbol}")

    async def broadcast_market_data(self, symbol: str, data: dict[str, Any]):
        """Broadcast market data to all symbol subscribers with consistent data transformation."""
        # Apply consistent data transformation matching messaging patterns
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        # Align processing mode for websocket streaming
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode="batch", target_mode="stream", data=data
        )

        message = {
            "type": "market_data",
            "symbol": symbol,
            "data": aligned_data,
            "processing_mode": "stream",  # Consistent with core events
            "message_pattern": "pub_sub",  # WebSocket broadcasts use pub/sub
            "data_format": "market_data_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.broadcast_to_subscription(message, f"symbol:{symbol}")


class BotStatusWebSocketManager(BaseWebSocketManager):
    """WebSocket manager specifically for bot status updates."""

    def __init__(self):
        super().__init__("Bot Status")

    def subscribe_to_bot(self, user_id: str, bot_id: str):
        """Subscribe user to bot status updates."""
        self.subscribe(user_id, f"bot:{bot_id}")

    def unsubscribe_from_bot(self, user_id: str, bot_id: str):
        """Unsubscribe user from bot status updates."""
        self.unsubscribe(user_id, f"bot:{bot_id}")

    async def broadcast_bot_status(self, bot_id: str, status_data: dict[str, Any]):
        """Broadcast bot status to all bot subscribers with consistent data transformation."""
        # Apply consistent data transformation matching messaging patterns
        from src.utils.messaging_patterns import BoundaryValidator, ProcessingParadigmAligner

        # Align processing mode for websocket streaming
        aligned_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode="async", target_mode="stream", data=status_data
        )

        # Apply boundary validation for web_interface data
        validation_data = aligned_data.copy()
        validation_data.update(
            {
                "component": "websocket_bot_manager",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_mode": "stream",
            }
        )

        try:
            BoundaryValidator.validate_web_interface_to_error_boundary(validation_data)
        except Exception as e:
            # Continue if validation fails - don't break websocket functionality
            logger.debug(f"WebSocket validation failed: {e}")

        message = {
            "type": "bot_status",
            "bot_id": bot_id,
            "data": aligned_data,
            "processing_mode": "stream",  # Consistent with core events
            "message_pattern": "pub_sub",  # WebSocket broadcasts use pub/sub
            "data_format": "bot_status_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.broadcast_to_subscription(message, f"bot:{bot_id}")


async def authenticate_websocket(websocket: WebSocket, token: str | None = None) -> str | None:
    """
    Authenticate WebSocket connection and return user_id if successful.

    Args:
        websocket: The WebSocket connection
        token: Optional authentication token

    Returns:
        str: User ID if authentication successful, None otherwise
    """
    try:
        # Authentication logic to be implemented with proper JWT validation
        # Current implementation provides mock authentication for development
        if not token:
            # Add timeout to WebSocket close to prevent hanging
            await asyncio.wait_for(
                websocket.close(code=4001, reason="Authentication required"), timeout=5.0
            )
            return None

        # Mock authentication - replace with real authentication
        user_id = f"user_{token[:8]}" if len(token) >= 8 else "anonymous"
        return user_id

    except asyncio.TimeoutError:
        logger.error("WebSocket authentication close timed out")
        return None
    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        try:
            # Add timeout to close operation
            await asyncio.wait_for(
                websocket.close(code=4000, reason="Authentication failed"), timeout=5.0
            )
        except (Exception, asyncio.TimeoutError) as close_error:
            logger.debug(f"Failed to close WebSocket connection: {close_error}")
        return None


async def handle_websocket_disconnect(
    websocket: WebSocket, manager: BaseWebSocketManager, user_id: str
):
    """
    Handle WebSocket disconnection cleanup.

    Args:
        websocket: The WebSocket connection
        manager: The WebSocket manager instance
        user_id: User identifier
    """
    try:
        manager.disconnect(user_id)
        logger.info(f"WebSocket connection cleaned up for user {user_id}")
    except Exception as e:
        logger.error(f"Error cleaning up WebSocket connection for user {user_id}: {e}")


# Exchange WebSocket Reconnection Utilities

class ExchangeWebSocketReconnectionManager:
    """Common reconnection logic for exchange WebSocket connections."""

    def __init__(self, exchange_name: str, max_reconnect_attempts: int = 5):
        self.exchange_name = exchange_name
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_attempts = 0
        self.reconnect_task: asyncio.Task | None = None

    def reset_reconnect_attempts(self) -> None:
        """Reset reconnection attempt counter."""
        self.reconnect_attempts = 0

    def should_attempt_reconnect(self) -> bool:
        """Check if reconnection should be attempted."""
        return self.reconnect_attempts < self.max_reconnect_attempts

    def calculate_reconnect_delay(self) -> float:
        """Calculate backoff delay for next reconnection attempt."""
        import random
        base_delay = min(2.0 ** self.reconnect_attempts, 30.0)  # Max 30s
        jitter = random.uniform(0.1, 0.3) * base_delay
        return base_delay + jitter

    async def schedule_reconnect(self, reconnect_callback) -> None:
        """
        Schedule a reconnection attempt with exponential backoff.
        
        Args:
            reconnect_callback: Async function to call for reconnection
        """
        if not self.should_attempt_reconnect():
            logger.error(f"Max reconnection attempts reached for {self.exchange_name}")
            return

        if self.reconnect_task and not self.reconnect_task.done():
            logger.debug(f"Reconnection already scheduled for {self.exchange_name}")
            return

        self.reconnect_task = asyncio.create_task(
            self._reconnect_with_delay(reconnect_callback)
        )

    async def _reconnect_with_delay(self, reconnect_callback) -> None:
        """Execute reconnection with delay."""
        try:
            delay = self.calculate_reconnect_delay()
            self.reconnect_attempts += 1

            logger.info(f"Scheduling reconnection attempt {self.reconnect_attempts} "
                       f"for {self.exchange_name} in {delay:.1f}s")

            await asyncio.sleep(delay)
            await reconnect_callback()

        except Exception as e:
            logger.error(f"Reconnection failed for {self.exchange_name}: {e}")

    def cancel_reconnect(self) -> None:
        """Cancel any pending reconnection."""
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
            self.reconnect_task = None
