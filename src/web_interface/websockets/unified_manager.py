"""
Unified WebSocket Manager for T-Bot Trading System.

This module consolidates all WebSocket handling into a single,
comprehensive manager that handles all real-time communications.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import socketio
from socketio import AsyncNamespace, AsyncServer

from src.base import BaseComponent
from src.core.logging import correlation_context
from src.web_interface.facade import get_api_facade


class ChannelType(Enum):
    """WebSocket channel types."""

    MARKET_DATA = "market_data"
    BOT_STATUS = "bot_status"
    PORTFOLIO = "portfolio"
    TRADES = "trades"
    ORDERS = "orders"
    ALERTS = "alerts"
    LOGS = "logs"
    RISK_METRICS = "risk_metrics"
    SYSTEM_STATUS = "system_status"


class SubscriptionLevel(Enum):
    """Subscription permission levels."""

    PUBLIC = "public"  # No auth required
    USER = "user"  # User auth required
    TRADING = "trading"  # Trading permission required
    ADMIN = "admin"  # Admin permission required


class WebSocketEventHandler:
    """Base class for WebSocket event handlers."""

    def __init__(
        self, channel: ChannelType, subscription_level: SubscriptionLevel = SubscriptionLevel.USER
    ):
        self.channel = channel
        self.subscription_level = subscription_level
        self.subscribers: set[str] = set()
        self._running = False

    async def subscribe(self, session_id: str) -> bool:
        """Subscribe a session to this channel."""
        self.subscribers.add(session_id)
        return True

    async def unsubscribe(self, session_id: str) -> bool:
        """Unsubscribe a session from this channel."""
        self.subscribers.discard(session_id)
        return True

    async def start(self) -> None:
        """Start the event handler."""
        self._running = True

    async def stop(self) -> None:
        """Stop the event handler."""
        self._running = False

    async def emit_to_channel(self, sio: AsyncServer, event: str, data: Any) -> None:
        """Emit data to all subscribers of this channel."""
        if self.subscribers:
            await sio.emit(event, data, room=list(self.subscribers))


class MarketDataHandler(WebSocketEventHandler):
    """Handler for market data WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.MARKET_DATA, SubscriptionLevel.PUBLIC)
        self.update_interval = 1.0  # 1 second updates

    async def start(self) -> None:
        """Start market data broadcasting."""
        await super().start()
        asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Market data broadcast loop."""
        while self._running:
            try:
                # Get market data from API facade
                get_api_facade()

                # Mock data for now - in production, get from facade
                market_data = {
                    "type": "market_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "BTC/USDT": {"price": 45000.00, "volume": 1234567890, "change_24h": 2.5},
                        "ETH/USDT": {"price": 2500.00, "volume": 987654321, "change_24h": -1.2},
                    },
                }

                if self.subscribers:
                    # This will be called by the unified manager
                    await self._emit_data("market_data", market_data)

                await asyncio.sleep(self.update_interval)

            except Exception:
                # Log error but continue
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission - will be set by unified manager."""
        pass


class BotStatusHandler(WebSocketEventHandler):
    """Handler for bot status WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.BOT_STATUS, SubscriptionLevel.USER)
        self.update_interval = 2.0  # 2 second updates

    async def start(self) -> None:
        """Start bot status broadcasting."""
        await super().start()
        asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Bot status broadcast loop."""
        while self._running:
            try:
                # Get bot status from API facade
                get_api_facade()

                # Mock data for now
                bot_status = {
                    "type": "bot_status_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "active_bots": 3,
                        "total_profit": 1234.56,
                        "bots": [
                            {
                                "id": "bot-1",
                                "name": "BTC Trader",
                                "status": "running",
                                "profit": 456.78,
                            }
                        ],
                    },
                }

                if self.subscribers:
                    await self._emit_data("bot_status", bot_status)

                await asyncio.sleep(self.update_interval)

            except Exception:
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission."""
        pass


class PortfolioHandler(WebSocketEventHandler):
    """Handler for portfolio WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.PORTFOLIO, SubscriptionLevel.USER)
        self.update_interval = 5.0  # 5 second updates

    async def start(self) -> None:
        """Start portfolio broadcasting."""
        await super().start()
        asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Portfolio broadcast loop."""
        while self._running:
            try:
                # Mock portfolio data
                portfolio_data = {
                    "type": "portfolio_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "total_value": 10000.00,
                        "daily_pnl": 123.45,
                        "daily_pnl_percent": 1.25,
                        "positions": [
                            {
                                "symbol": "BTC/USDT",
                                "size": 0.1,
                                "entry_price": 45000,
                                "current_price": 46000,
                                "pnl": 100.00,
                            }
                        ],
                    },
                }

                if self.subscribers:
                    await self._emit_data("portfolio_update", portfolio_data)

                await asyncio.sleep(self.update_interval)

            except Exception:
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission."""
        pass


class UnifiedWebSocketNamespace(AsyncNamespace):
    """Unified namespace for all WebSocket communications."""

    def __init__(self, namespace: str = "/"):
        super().__init__(namespace)
        self.connected_clients: dict[str, dict[str, Any]] = {}
        self.authenticated_sessions: set[str] = set()
        self.session_subscriptions: dict[str, set[ChannelType]] = {}

        # Initialize handlers
        self.handlers: dict[ChannelType, WebSocketEventHandler] = {
            ChannelType.MARKET_DATA: MarketDataHandler(),
            ChannelType.BOT_STATUS: BotStatusHandler(),
            ChannelType.PORTFOLIO: PortfolioHandler(),
        }

        # Set emit functions for handlers
        for handler in self.handlers.values():
            handler._emit_data = self._create_emit_function(handler.channel)

    def _create_emit_function(self, channel: ChannelType) -> Callable:
        """Create an emit function for a specific channel."""

        async def emit_data(event: str, data: Any):
            channel_room = f"channel_{channel.value}"
            await self.emit(event, data, room=channel_room)

        return emit_data

    async def on_connect(
        self, sid: str, environ: dict[str, Any], auth: dict[str, Any] | None = None
    ):
        """Handle client connection."""
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            self.logger.info(f"Client connected: {sid}")

        # Store client info
        self.connected_clients[sid] = {
            "connected_at": datetime.now(timezone.utc),
            "authenticated": False,
            "user_id": None,
            "permissions": set(),
            "last_activity": datetime.now(timezone.utc),
        }

        self.session_subscriptions[sid] = set()

        # Handle authentication if provided
        if auth and "token" in auth:
            await self._authenticate_session(sid, auth["token"])

        # Send welcome message
        await self.emit(
            "welcome",
            {
                "message": "Connected to T-Bot Trading System",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "2.0.0",
                "session_id": sid,
            },
            room=sid,
        )

        return True

    async def on_disconnect(self, sid: str):
        """Handle client disconnection."""
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            self.logger.info(f"Client disconnected: {sid}")

        # Unsubscribe from all channels
        if sid in self.session_subscriptions:
            for channel in list(self.session_subscriptions[sid]):
                await self._unsubscribe_from_channel(sid, channel)

        # Clean up client data
        self.connected_clients.pop(sid, None)
        self.authenticated_sessions.discard(sid)
        self.session_subscriptions.pop(sid, None)

    async def on_authenticate(self, sid: str, data: dict[str, Any]):
        """Handle authentication request."""
        token = data.get("token")
        if not token:
            await self.emit("auth_error", {"error": "Token required"}, room=sid)
            return

        await self._authenticate_session(sid, token)

    async def _authenticate_session(self, sid: str, token: str) -> bool:
        """Authenticate a session with the provided token."""
        try:
            # TODO: Implement proper JWT validation
            # For now, accept any non-empty token
            if token:
                self.connected_clients[sid]["authenticated"] = True
                self.connected_clients[sid]["user_id"] = "user123"  # Mock user ID
                self.connected_clients[sid]["permissions"] = {"user", "trading"}  # Mock permissions
                self.authenticated_sessions.add(sid)

                await self.emit(
                    "authenticated",
                    {
                        "status": "success",
                        "user_id": "user123",
                        "permissions": list(self.connected_clients[sid]["permissions"]),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    room=sid,
                )

                return True
        except Exception as e:
            await self.emit("auth_error", {"error": f"Authentication failed: {e!s}"}, room=sid)

        return False

    async def on_subscribe(self, sid: str, data: dict[str, Any]):
        """Handle subscription to channels."""
        if not self._is_authenticated(sid):
            await self.emit("error", {"error": "Authentication required"}, room=sid)
            return

        channels = data.get("channels", [])
        successful_subscriptions = []
        failed_subscriptions = []

        for channel_name in channels:
            try:
                channel = ChannelType(channel_name)
                if await self._subscribe_to_channel(sid, channel):
                    successful_subscriptions.append(channel_name)
                else:
                    failed_subscriptions.append(
                        {"channel": channel_name, "error": "Subscription failed"}
                    )
            except ValueError:
                failed_subscriptions.append({"channel": channel_name, "error": "Unknown channel"})

        await self.emit(
            "subscription_result",
            {
                "successful": successful_subscriptions,
                "failed": failed_subscriptions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            room=sid,
        )

    async def on_unsubscribe(self, sid: str, data: dict[str, Any]):
        """Handle unsubscription from channels."""
        channels = data.get("channels", [])

        for channel_name in channels:
            try:
                channel = ChannelType(channel_name)
                await self._unsubscribe_from_channel(sid, channel)
            except ValueError:
                pass  # Ignore unknown channels

        await self.emit(
            "unsubscribed",
            {"channels": channels, "timestamp": datetime.now(timezone.utc).isoformat()},
            room=sid,
        )

    async def _subscribe_to_channel(self, sid: str, channel: ChannelType) -> bool:
        """Subscribe a session to a specific channel."""
        # Check permissions
        if not self._has_channel_permission(sid, channel):
            return False

        # Add to handler's subscribers
        handler = self.handlers.get(channel)
        if handler:
            await handler.subscribe(sid)
            # Add to socket.io room
            channel_room = f"channel_{channel.value}"
            await self.enter_room(sid, channel_room)

            # Track subscription
            self.session_subscriptions[sid].add(channel)

            return True

        return False

    async def _unsubscribe_from_channel(self, sid: str, channel: ChannelType) -> bool:
        """Unsubscribe a session from a specific channel."""
        # Remove from handler's subscribers
        handler = self.handlers.get(channel)
        if handler:
            await handler.unsubscribe(sid)
            # Remove from socket.io room
            channel_room = f"channel_{channel.value}"
            await self.leave_room(sid, channel_room)

            # Remove from tracking
            self.session_subscriptions.get(sid, set()).discard(channel)

            return True

        return False

    def _is_authenticated(self, sid: str) -> bool:
        """Check if a session is authenticated."""
        return sid in self.authenticated_sessions

    def _has_channel_permission(self, sid: str, channel: ChannelType) -> bool:
        """Check if a session has permission for a channel."""
        client_info = self.connected_clients.get(sid, {})
        permissions = client_info.get("permissions", set())

        handler = self.handlers.get(channel)
        if not handler:
            return False

        required_level = handler.subscription_level

        # Check permission levels
        if required_level == SubscriptionLevel.PUBLIC:
            return True
        elif required_level == SubscriptionLevel.USER:
            return "user" in permissions
        elif required_level == SubscriptionLevel.TRADING:
            return "trading" in permissions
        elif required_level == SubscriptionLevel.ADMIN:
            return "admin" in permissions

        return False

    async def on_ping(self, sid: str, data: dict[str, Any] | None = None):
        """Handle ping for connection health check."""
        self.connected_clients[sid]["last_activity"] = datetime.now(timezone.utc)

        await self.emit(
            "pong",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency": data.get("timestamp") if data else None,
            },
            room=sid,
        )

    async def start_handlers(self):
        """Start all event handlers."""
        for handler in self.handlers.values():
            await handler.start()

    async def stop_handlers(self):
        """Stop all event handlers."""
        for handler in self.handlers.values():
            await handler.stop()


class UnifiedWebSocketManager(BaseComponent):
    """Unified manager for all WebSocket communications."""

    def __init__(self):
        super().__init__()
        self.sio: AsyncServer | None = None
        self.namespace: UnifiedWebSocketNamespace | None = None
        self._running = False

    def create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer:
        """Create and configure unified Socket.IO server."""
        if cors_allowed_origins is None:
            cors_allowed_origins = ["http://localhost:3000", "http://localhost:8000"]

        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=cors_allowed_origins,
            logger=False,  # Use our own logging
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=1000000,
            allow_upgrades=True,
            compression_threshold=1024,
        )

        # Create and register unified namespace
        self.namespace = UnifiedWebSocketNamespace("/")
        self.sio.register_namespace(self.namespace)

        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            self.logger.info("Unified WebSocket server created")

        return self.sio

    async def start(self):
        """Start the unified WebSocket manager."""
        if self._running or not self.namespace:
            return

        self._running = True
        await self.namespace.start_handlers()

        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            self.logger.info("Unified WebSocket manager started")

    async def stop(self):
        """Stop the unified WebSocket manager."""
        if not self._running or not self.namespace:
            return

        self._running = False
        await self.namespace.stop_handlers()

        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            self.logger.info("Unified WebSocket manager stopped")

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        if not self.namespace:
            return {}

        return {
            "total_connections": len(self.namespace.connected_clients),
            "authenticated_connections": len(self.namespace.authenticated_sessions),
            "channel_subscriptions": {
                channel.value: len(handler.subscribers)
                for channel, handler in self.namespace.handlers.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Global unified manager
_unified_manager: UnifiedWebSocketManager | None = None


def get_unified_websocket_manager() -> UnifiedWebSocketManager:
    """Get or create the unified WebSocket manager."""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedWebSocketManager()
    return _unified_manager
