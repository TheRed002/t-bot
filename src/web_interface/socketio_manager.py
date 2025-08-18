"""
Socket.IO manager for T-Bot Trading System.

This module provides centralized Socket.IO management with:
- Authentication and session management
- Room-based broadcasting for different data streams
- Error handling and reconnection logic
- Rate limiting and security features
"""

import asyncio
from datetime import datetime
from typing import Any

import socketio
from socketio import AsyncNamespace, AsyncServer

from src.core.logging import correlation_context, get_logger

logger = get_logger(__name__)


class TradingNamespace(AsyncNamespace):
    """Main namespace for trading-related Socket.IO events."""

    def __init__(self, namespace: str = "/"):
        super().__init__(namespace)
        self.connected_clients: dict[str, dict[str, Any]] = {}
        self.authenticated_sessions: set[str] = set()

    async def on_connect(
        self, sid: str, environ: dict[str, Any], auth: dict[str, Any] | None = None
    ):
        """Handle client connection."""
        logger.info(f"Client connected: {sid}")

        # Store client info
        self.connected_clients[sid] = {
            "connected_at": datetime.utcnow().isoformat(),
            "authenticated": False,
            "user_id": None,
            "subscriptions": set(),
        }

        # Check for authentication token
        if auth and "token" in auth:
            # Validate token (simplified for now)
            if await self._validate_token(auth["token"]):
                self.connected_clients[sid]["authenticated"] = True
                self.authenticated_sessions.add(sid)
                await self.emit("authenticated", {"status": "success"}, room=sid)
                logger.info(f"Client {sid} authenticated successfully")
            else:
                await self.emit(
                    "authenticated", {"status": "failed", "error": "Invalid token"}, room=sid
                )

        # Send welcome message
        await self.emit(
            "welcome",
            {
                "message": "Connected to T-Bot Trading System",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
            },
            room=sid,
        )

        return True

    async def on_disconnect(self, sid: str):
        """Handle client disconnection."""
        logger.info(f"Client disconnected: {sid}")

        # Clean up client data
        if sid in self.connected_clients:
            del self.connected_clients[sid]
        if sid in self.authenticated_sessions:
            self.authenticated_sessions.remove(sid)

    async def on_authenticate(self, sid: str, data: dict[str, Any]):
        """Handle authentication request."""
        token = data.get("token")
        if not token:
            await self.emit("auth_error", {"error": "Token required"}, room=sid)
            return

        if await self._validate_token(token):
            self.connected_clients[sid]["authenticated"] = True
            self.authenticated_sessions.add(sid)
            await self.emit(
                "authenticated",
                {"status": "success", "timestamp": datetime.utcnow().isoformat()},
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
                logger.info(f"Client {sid} subscribed to {channel}")

        await self.emit(
            "subscribed",
            {
                "channels": list(self.connected_clients[sid]["subscriptions"]),
                "timestamp": datetime.utcnow().isoformat(),
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
            {"channels": channels, "timestamp": datetime.utcnow().isoformat()},
            room=sid,
        )

    async def on_ping(self, sid: str, data: dict[str, Any]):
        """Handle ping for connection health check."""
        await self.emit(
            "pong",
            {"timestamp": datetime.utcnow().isoformat(), "latency": data.get("timestamp")},
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

        # TODO: Execute order through trading engine
        # For now, send confirmation
        await self.emit(
            "order_submitted",
            {
                "order_id": f"ORD-{sid}-{datetime.utcnow().timestamp()}",
                "status": "pending",
                "timestamp": datetime.utcnow().isoformat(),
            },
            room=sid,
        )

    async def on_get_portfolio(self, sid: str, data: dict[str, Any]):
        """Handle portfolio data requests."""
        if sid not in self.authenticated_sessions:
            await self.emit("error", {"error": "Authentication required"}, room=sid)
            return

        # TODO: Fetch real portfolio data
        # Mock data for now
        portfolio_data = {
            "total_value": 10000.00,
            "available_balance": 5000.00,
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "amount": 0.1,
                    "entry_price": 45000,
                    "current_price": 46000,
                    "pnl": 100.00,
                }
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.emit("portfolio_data", portfolio_data, room=sid)

    async def _validate_token(self, token: str) -> bool:
        """Validate authentication token."""
        # TODO: Implement proper JWT validation
        # For now, accept any non-empty token in development mode
        # In production, this should validate JWT tokens properly
        return bool(token)


class SocketIOManager:
    """Manager for Socket.IO server and connections."""

    def __init__(self):
        self.sio: AsyncServer | None = None
        self.app = None
        self.background_tasks: list[asyncio.Task] = []
        self.is_running = False

    def create_server(self, cors_allowed_origins: list[str] = None) -> AsyncServer:
        """Create and configure Socket.IO server."""
        if cors_allowed_origins is None:
            cors_allowed_origins = ["http://localhost:3000", "http://localhost:8000"]

        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=cors_allowed_origins,
            logger=logger,
            engineio_logger=False,  # Reduce verbosity
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=1000000,
            allow_upgrades=True,
            compression_threshold=1024,
        )

        # Register namespaces
        self.sio.register_namespace(TradingNamespace("/"))

        # Generate correlation ID for server creation
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            logger.info("Socket.IO server created")
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

        self.is_running = True

        # Generate correlation ID for startup tasks
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            # Start market data broadcaster
            self.background_tasks.append(asyncio.create_task(self._broadcast_market_data()))

            # Start bot status broadcaster
            self.background_tasks.append(asyncio.create_task(self._broadcast_bot_status()))

            # Start portfolio updates
            self.background_tasks.append(asyncio.create_task(self._broadcast_portfolio_updates()))

            logger.info("Background tasks started")

    async def stop_background_tasks(self):
        """Stop all background tasks."""
        self.is_running = False

        # Generate correlation ID for shutdown tasks
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            for task in self.background_tasks:
                task.cancel()

            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)

            self.background_tasks.clear()
            logger.info("Background tasks stopped")

    async def _broadcast_market_data(self):
        """Broadcast market data to subscribed clients."""
        while self.is_running:
            try:
                # Generate correlation ID for this background task
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    # TODO: Get real market data
                    # Mock data for now
                    market_data = {
                        "type": "market_update",
                        "data": {
                            "BTC/USDT": {
                                "price": 45000 + (asyncio.get_event_loop().time() % 1000),
                                "volume": 1234567890,
                                "change_24h": 2.5,
                            },
                            "ETH/USDT": {
                                "price": 2500 + (asyncio.get_event_loop().time() % 100),
                                "volume": 987654321,
                                "change_24h": -1.2,
                            },
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    if self.sio:
                        await self.sio.emit("market_data", market_data, room="market_data")
                        logger.debug('emitting event "market_data" to market_data [/]')

                    await asyncio.sleep(5)  # Update every 5 seconds to reduce log volume

            except Exception as e:
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    logger.error(f"Error broadcasting market data: {e}")
                await asyncio.sleep(5)

    async def _broadcast_bot_status(self):
        """Broadcast bot status updates."""
        while self.is_running:
            try:
                # Generate correlation ID for this background task
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    # TODO: Get real bot status
                    # Mock data for now
                    bot_status = {
                        "type": "bot_status",
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
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    if self.sio:
                        await self.sio.emit("bot_status", bot_status, room="bot_status")
                        logger.debug('emitting event "bot_status" to bot_status [/]')

                    await asyncio.sleep(5)  # Update every 5 seconds

            except Exception as e:
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    logger.error(f"Error broadcasting bot status: {e}")
                await asyncio.sleep(10)

    async def _broadcast_portfolio_updates(self):
        """Broadcast portfolio updates."""
        while self.is_running:
            try:
                # Generate correlation ID for this background task
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    # TODO: Get real portfolio updates
                    # Mock data for now
                    portfolio_update = {
                        "type": "portfolio_update",
                        "data": {
                            "total_value": 10000 + (asyncio.get_event_loop().time() % 1000),
                            "daily_pnl": 123.45,
                            "daily_pnl_percent": 1.25,
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    if self.sio:
                        await self.sio.emit("portfolio_update", portfolio_update, room="portfolio")
                        logger.debug('emitting event "portfolio_update" to portfolio [/]')

                    await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                correlation_id = correlation_context.generate_correlation_id()
                with correlation_context.correlation_context(correlation_id):
                    logger.error(f"Error broadcasting portfolio updates: {e}")
                await asyncio.sleep(15)

    async def emit_to_user(self, user_id: str, event: str, data: Any):
        """Emit event to specific user."""
        # TODO: Implement user-specific room management
        pass

    async def broadcast(self, event: str, data: Any, room: str | None = None):
        """Broadcast event to all clients or specific room."""
        if self.sio:
            await self.sio.emit(event, data, room=room)


# Global instance
socketio_manager = SocketIOManager()
