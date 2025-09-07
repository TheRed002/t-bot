"""
Socket.IO manager for T-Bot Trading System.

This module provides centralized Socket.IO management with:
- Authentication and session management
- Room-based broadcasting for different data streams
- Error handling and reconnection logic
- Rate limiting and security features
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import socketio
from socketio import AsyncNamespace, AsyncServer

from src.core.base import BaseComponent


from src.core.exceptions import AuthenticationError
from src.core.logging import correlation_context
from src.error_handling import (
    ConnectionManager,
    ConnectionState,
    with_error_context,
)
from src.error_handling.recovery_scenarios import NetworkDisconnectionRecovery


class TradingNamespace(AsyncNamespace):
    """Main namespace for trading-related Socket.IO events."""

    def __init__(self, namespace: str = "/", jwt_handler=None):
        super().__init__(namespace)
        self.connected_clients: dict[str, dict[str, Any]] = {}
        self.authenticated_sessions: set[str] = set()
        self.jwt_handler = jwt_handler
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize connection manager for resilient connections
        try:
            from src.core.config import Config

            config = Config()
            self.connection_manager = ConnectionManager(config)
            self.network_recovery = NetworkDisconnectionRecovery(config)
        except Exception as e:
            # Fallback - create mock managers if Config is not available
            self.logger.warning(f"Failed to initialize connection manager: {e}")
            self.connection_manager = None
            self.network_recovery = None

        # Rate limiting: max connections per IP
        self.connection_counts: dict[str, int] = {}
        self.max_connections_per_ip = 10

    async def on_connect(
        self, sid: str, environ: dict[str, Any], auth: dict[str, Any] | None = None
    ):
        """Handle client connection with security checks."""
        # Get client IP for rate limiting
        client_ip = environ.get("REMOTE_ADDR", "unknown")

        # Register connection with connection manager
        if self.connection_manager:
            try:
                await self.connection_manager.add_connection(
                    connection_id=sid,
                    connection_type="socketio",
                    metadata={"client_ip": client_ip, "namespace": self.namespace},
                )
                await self.connection_manager.update_state(sid, ConnectionState.CONNECTING)
            except Exception as e:
                self.logger.warning(f"Failed to register connection {sid} with manager: {e}")

        # Rate limiting by IP
        if client_ip != "unknown":
            current_connections = self.connection_counts.get(client_ip, 0)
            if current_connections >= self.max_connections_per_ip:
                self.logger.warning(f"Connection limit exceeded for IP {client_ip}")
                return False

            self.connection_counts[client_ip] = current_connections + 1

        self.logger.info(f"Client connected: {sid} from IP: {client_ip}")

        # Store client info
        self.connected_clients[sid] = {
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "authenticated": False,
            "user_id": None,
            "username": None,
            "scopes": [],
            "subscriptions": set(),
            "client_ip": client_ip,
        }

        # Check for authentication token
        if auth and "token" in auth:
            try:
                token_data = await self._validate_token(auth["token"])
                if token_data:
                    self.connected_clients[sid]["authenticated"] = True
                    self.connected_clients[sid]["user_id"] = token_data["user_id"]
                    self.connected_clients[sid]["username"] = token_data["username"]
                    self.connected_clients[sid]["scopes"] = token_data["scopes"]
                    self.authenticated_sessions.add(sid)

                    await self.emit("authenticated", {"status": "success"}, room=sid)
                    self.logger.info(f"Client {sid} authenticated as {token_data['username']}")

                    # Update connection state to active after successful auth
                    if self.connection_manager:
                        try:
                            await self.connection_manager.update_state(sid, ConnectionState.ACTIVE)
                        except Exception as e:
                            self.logger.warning(f"Failed to update connection state to active: {e}")
                else:
                    await self.emit("auth_error", {"error": "Invalid or expired token"}, room=sid)
                    self.logger.warning(f"Authentication failed for client {sid}")
            except Exception as e:
                self.logger.error(f"Authentication error for client {sid}: {e}")
                await self.emit("auth_error", {"error": "Authentication service error"}, room=sid)
        else:
            # No authentication provided - client must authenticate before using secured features
            await self.emit(
                "auth_required",
                {
                    "message": "Authentication required for secure features",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                room=sid,
            )

        # Send welcome message with limited info
        await self.emit(
            "welcome",
            {
                "message": "Connected to T-Bot Trading System",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "authenticated": self.connected_clients[sid]["authenticated"],
            },
            room=sid,
        )

        return True

    @with_error_context(operation="socketio_disconnect")
    async def on_disconnect(self, sid: str):
        """Handle client disconnection."""
        self.logger.info(f"Client disconnected: {sid}")

        # Update connection state in connection manager with error handling
        if self.connection_manager:
            try:
                await self.connection_manager.update_state(sid, ConnectionState.DISCONNECTED)
            except Exception as e:
                self.logger.warning(f"Failed to update connection state for {sid}: {e}")

        # Clean up client data and decrement connection count
        if sid in self.connected_clients:
            client_data = self.connected_clients[sid]
            client_ip = client_data.get("client_ip")

            # Decrement connection count for IP
            if client_ip and client_ip != "unknown":
                current_count = self.connection_counts.get(client_ip, 0)
                if current_count > 0:
                    self.connection_counts[client_ip] = current_count - 1
                    if self.connection_counts[client_ip] == 0:
                        del self.connection_counts[client_ip]

            del self.connected_clients[sid]

        if sid in self.authenticated_sessions:
            self.authenticated_sessions.remove(sid)

        # Remove connection from manager
        if self.connection_manager:
            try:
                await self.connection_manager.remove_connection(sid)
            except Exception as e:
                self.logger.warning(f"Failed to remove connection {sid} from manager: {e}")

    @with_error_context(operation="socketio_authenticate")
    async def on_authenticate(self, sid: str, data: dict[str, Any]):
        """Handle authentication request."""
        token = data.get("token")
        if not token:
            await self.emit("auth_error", {"error": "Token required"}, room=sid)
            return

        token_data = await self._validate_token(token)
        if token_data:
            self.connected_clients[sid]["authenticated"] = True
            self.authenticated_sessions.add(sid)
            await self.emit(
                "authenticated",
                {"status": "success", "timestamp": datetime.now(timezone.utc).isoformat()},
                room=sid,
            )

            # Join authenticated room for broadcasts
            await self.enter_room(sid, "authenticated")
        else:
            await self.emit("auth_error", {"error": "Invalid token"}, room=sid)

    async def on_subscribe(self, sid: str, data: dict[str, Any]):
        """Handle subscription to data streams."""
        if sid not in self.authenticated_sessions:
            await self.emit("error", {"error": "Authentication required"}, room=sid)
            return

        channels = data.get("channels", [])
        for channel in channels:
            if channel in ["market_data", "bot_status", "portfolio", "alerts", "logs"]:
                await self.enter_room(sid, channel)
                self.connected_clients[sid]["subscriptions"].add(channel)
                self.logger.info(f"Client {sid} subscribed to {channel}")

        await self.emit(
            "subscribed",
            {
                "channels": list(self.connected_clients[sid]["subscriptions"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            room=sid,
        )

    async def on_unsubscribe(self, sid: str, data: dict[str, Any]):
        """Handle unsubscription from data streams."""
        channels = data.get("channels", [])
        for channel in channels:
            await self.leave_room(sid, channel)
            if sid in self.connected_clients:
                self.connected_clients[sid]["subscriptions"].discard(channel)

        await self.emit(
            "unsubscribed",
            {"channels": channels, "timestamp": datetime.now(timezone.utc).isoformat()},
            room=sid,
        )

    async def on_ping(self, sid: str, data: dict[str, Any] | None = None):
        """Handle ping for connection health check."""
        # Handle both cases: with and without data
        latency = None
        if data and isinstance(data, dict):
            latency = data.get("timestamp")

        await self.emit(
            "pong",
            {"timestamp": datetime.now(timezone.utc).isoformat(), "latency": latency},
            room=sid,
        )

    async def on_execute_order(self, sid: str, data: dict[str, Any]):
        """Handle order execution requests."""
        if sid not in self.authenticated_sessions:
            await self.emit("error", {"error": "Authentication required"}, room=sid)
            return

        # Validate order data
        order_type = data.get("type")
        symbol = data.get("symbol")
        side = data.get("side")
        amount = data.get("amount")

        if not all([order_type, symbol, side, amount]):
            await self.emit("order_error", {"error": "Missing required order fields"}, room=sid)
            return

        # Execute order through trading engine
        # TBD: Integrate with actual trading engine when execution service is available
        await self.emit(
            "order_submitted",
            {
                "order_id": f"ORD-{sid}-{datetime.now(timezone.utc).timestamp()}",
                "status": "pending",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            room=sid,
        )

    async def on_get_portfolio(self, sid: str, data: dict[str, Any]):
        """Handle portfolio data requests."""
        if sid not in self.authenticated_sessions:
            await self.emit("error", {"error": "Authentication required"}, room=sid)
            return

        # TBD: Integrate with portfolio service for real data when service is available
        portfolio_data = {
            "total_value": Decimal("10000.00"),
            "available_balance": Decimal("5000.00"),
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "amount": Decimal("0.1"),
                    "entry_price": Decimal("45000"),
                    "current_price": Decimal("46000"),
                    "pnl": Decimal("100.00"),
                }
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self.emit("portfolio_data", portfolio_data, room=sid)

    async def _validate_token(self, token: str) -> dict[str, Any] | None:
        """Validate JWT authentication token."""
        try:
            if not self.jwt_handler:
                self.logger.error("JWT handler not configured for WebSocket authentication")
                return None

            # Validate the JWT token - check if it's async
            if callable(self.jwt_handler.validate_token) and asyncio.iscoroutinefunction(
                self.jwt_handler.validate_token
            ):
                token_data = await self.jwt_handler.validate_token(token)
            else:
                token_data = self.jwt_handler.validate_token(token)

            if token_data:
                return {
                    "user_id": token_data.user_id,
                    "username": token_data.username,
                    "scopes": token_data.scopes,
                    "expires_at": (
                        token_data.expires_at.isoformat() if token_data.expires_at else None
                    ),
                }

            return None

        except AuthenticationError as e:
            self.logger.warning(f"JWT token validation failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return None

    def _require_scope(self, sid: str, required_scope: str) -> bool:
        """Check if authenticated client has required scope."""
        if sid not in self.connected_clients:
            return False

        client = self.connected_clients[sid]
        if not client["authenticated"]:
            return False

        user_scopes = client.get("scopes", [])

        # Admin scope grants all permissions
        if "admin" in user_scopes:
            return True

        return required_scope in user_scopes


class SocketIOManager(BaseComponent):
    """Manager for Socket.IO server and connections."""

    def __init__(self):
        super().__init__()
        self.sio: AsyncServer | None = None
        self.app = None
        self.background_tasks: list[asyncio.Task] = []

    def create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer:
        """Create and configure Socket.IO server."""
        if cors_allowed_origins is None:
            cors_allowed_origins = ["http://localhost:3000", "http://localhost:8000"]

        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=cors_allowed_origins,
            logger=self.logger,
            engineio_logger=False,  # Reduce verbosity
            ping_timeout=30,  # Reduced timeout for faster detection
            ping_interval=15,  # More frequent pings
            max_http_buffer_size=1000000,
            allow_upgrades=True,
            compression_threshold=1024,
            # Connection timeout and heartbeat
            always_connect=False,
            transports=["websocket", "polling"],  # Prefer WebSocket
        )

        # Register namespaces
        self.sio.register_namespace(TradingNamespace("/"))

        # Generate correlation ID for server creation
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            self.logger.info("Socket.IO server created")
        return self.sio

    def create_app(self):
        """Create ASGI app for Socket.IO."""
        if not self.sio:
            raise RuntimeError("Socket.IO server not created. Call create_server() first.")

        self.app = socketio.ASGIApp(self.sio, socketio_path="/socket.io/")
        return self.app

    async def start_background_tasks(self):
        """Start background tasks for data broadcasting."""
        if self.is_running:
            return

        self._is_running = True

        # Generate correlation ID for startup tasks
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            # Start market data broadcaster
            self.background_tasks.append(asyncio.create_task(self._broadcast_market_data()))

            # Start bot status broadcaster
            self.background_tasks.append(asyncio.create_task(self._broadcast_bot_status()))

            # Start portfolio updates
            self.background_tasks.append(asyncio.create_task(self._broadcast_portfolio_updates()))

            self.logger.info("Background tasks started")

    async def stop_background_tasks(self):
        """Stop all background tasks."""
        self._is_running = False

        # Generate correlation ID for shutdown tasks
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            try:
                # Cancel all background tasks
                for task in self.background_tasks:
                    if not task.done():
                        task.cancel()

                # Wait for all tasks to complete or be cancelled
                if self.background_tasks:
                    await asyncio.gather(*self.background_tasks, return_exceptions=True)

            except Exception as e:
                self.logger.error(f"Error stopping background tasks: {e}")
            finally:
                self.background_tasks.clear()
                self.logger.info("Background tasks stopped")

    async def _broadcast_market_data(self):
        """Broadcast market data to subscribed clients."""
        while self.is_running:
            try:
                # Generate correlation ID for this background task
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    # TBD: Integrate with market data service for real-time feeds
                    market_data = {
                        "type": "market_update",
                        "data": {
                            "BTC/USDT": {
                                "price": 45000 + (asyncio.get_event_loop().time() % 1000),
                                "volume": 1234567890,
                                "change_24h": Decimal("2.5"),
                            },
                            "ETH/USDT": {
                                "price": 2500 + (asyncio.get_event_loop().time() % 100),
                                "volume": 987654321,
                                "change_24h": Decimal("-1.2"),
                            },
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    if self.sio:
                        await self.sio.emit("market_data", market_data, room="market_data")
                        self.logger.debug('emitting event "market_data" to market_data [/]')

                    await asyncio.sleep(5)  # Update every 5 seconds to reduce log volume

            except Exception as e:
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    self.logger.error(f"Error broadcasting market data: {e}")
                await asyncio.sleep(5)

    async def _broadcast_bot_status(self):
        """Broadcast bot status updates."""
        while self.is_running:
            try:
                # Generate correlation ID for this background task
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    # TBD: Integrate with bot management service for real bot status
                    bot_status = {
                        "type": "bot_status",
                        "data": {
                            "active_bots": 3,
                            "total_profit": Decimal("1234.56"),
                            "bots": [
                                {
                                    "id": "bot-1",
                                    "name": "BTC Trader",
                                    "status": "running",
                                    "profit": Decimal("456.78"),
                                }
                            ],
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    if self.sio:
                        await self.sio.emit("bot_status", bot_status, room="bot_status")
                        self.logger.debug('emitting event "bot_status" to bot_status [/]')

                    await asyncio.sleep(5)  # Update every 5 seconds

            except Exception as e:
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    self.logger.error(f"Error broadcasting bot status: {e}")
                await asyncio.sleep(10)

    async def _broadcast_portfolio_updates(self):
        """Broadcast portfolio updates."""
        while self.is_running:
            try:
                # Generate correlation ID for this background task
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    # TBD: Integrate with portfolio service for real portfolio updates
                    portfolio_update = {
                        "type": "portfolio_update",
                        "data": {
                            "total_value": 10000 + (asyncio.get_event_loop().time() % 1000),
                            "daily_pnl": Decimal("123.45"),
                            "daily_pnl_percent": Decimal("1.25"),
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    if self.sio:
                        await self.sio.emit("portfolio_update", portfolio_update, room="portfolio")
                        self.logger.debug('emitting event "portfolio_update" to portfolio [/]')

                    await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    self.logger.error(f"Error broadcasting portfolio updates: {e}")
                await asyncio.sleep(15)

    async def emit_to_user(self, user_id: str, event: str, data: Any):
        """Emit event to specific user."""
        # Find user's session and emit to their room
        user_room = f"user_{user_id}"
        if self.sio:
            await self.sio.emit(event, data, room=user_room)

    async def broadcast(self, event: str, data: Any, room: str | None = None):
        """Broadcast event to all clients or specific room."""
        if self.sio:
            await self.sio.emit(event, data, room=room)


# Global instance
socketio_manager = SocketIOManager()
