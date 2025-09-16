"""
Unified WebSocket Manager for T-Bot Trading System.

This module consolidates all WebSocket handling into a single,
comprehensive manager that handles all real-time communications.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import socketio
from socketio import AsyncNamespace, AsyncServer

from src.core.base import BaseComponent
from src.core.logging import correlation_context

if TYPE_CHECKING:
    from src.web_interface.facade import APIFacade


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
        self._task: asyncio.Task | None = None
        self.logger = logging.getLogger(self.__class__.__name__)

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
        """Stop the event handler with proper cleanup and timeout."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                # Use async context manager for timeout
                try:
                    await asyncio.wait_for(self._task, timeout=5.0)
                except asyncio.CancelledError:
                    # Task was cancelled - this is expected
                    self.logger.debug(f"Event handler task cancelled for {self.channel.value}")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout stopping event handler task for {self.channel.value}")
                # Force cancellation if timeout
                self._task.cancel()
                try:
                    await asyncio.wait_for(self._task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            except Exception as e:
                self.logger.warning(f"Error stopping event handler task: {e}")
        self._task = None

    async def emit_to_channel(self, sio: AsyncServer, event: str, data: Any) -> None:
        """Emit data to all subscribers of this channel with timeout and error handling."""
        if not self.subscribers:
            return

        try:
            # Use timeout to prevent hanging on emit operations
            try:
                subscriber_list = list(self.subscribers)  # Create snapshot to avoid race conditions
                if subscriber_list:
                    await asyncio.wait_for(sio.emit(event, data, room=subscriber_list), timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timeout emitting to channel {self.channel.value}, clearing subscribers"
                )
                # Clear subscribers on timeout to prevent future hangs
                self.subscribers.clear()
        except Exception as e:
            self.logger.error(f"Failed to emit to channel {self.channel.value}: {e}")
            # Only clear subscribers on connection-related errors, not data errors
            if "connection" in str(e).lower() or "socket" in str(e).lower():
                self.subscribers.clear()


class MarketDataHandler(WebSocketEventHandler):
    """Handler for market data WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.MARKET_DATA, SubscriptionLevel.PUBLIC)
        self.update_interval = 1.0  # 1 second updates

    async def start(self) -> None:
        """Start market data broadcasting."""
        await super().start()
        self._task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Market data broadcast loop with consistent data transformation."""
        while self._running:
            try:
                # Mock data for now - in production, get from API facade
                # NOTE: Real implementation would use: facade = get_api_facade()

                # Apply consistent data transformation patterns
                market_data = {
                    "type": "market_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "processing_mode": "stream",  # Align with core events
                    "data_format": "event_data_v1",
                    "message_pattern": "pub_sub",  # WebSocket uses pub_sub pattern
                    "data": {
                        "BTC/USDT": {"price": 45000.00, "volume": 1234567890, "change_24h": 2.5},
                        "ETH/USDT": {"price": 2500.00, "volume": 987654321, "change_24h": -1.2},
                    },
                }

                if self.subscribers:
                    # This will be called by the unified manager
                    await self._emit_data("market_data", market_data)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in market data broadcast loop: {e}")
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
        self._task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Bot status broadcast loop."""
        while self._running:
            try:
                # Mock data for now - in production, get from API facade
                # NOTE: Real implementation would use: facade = get_api_facade()
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

            except Exception as e:
                self.logger.error(f"Error in bot status broadcast loop: {e}")
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
        self._task = asyncio.create_task(self._broadcast_loop())

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

            except Exception as e:
                self.logger.error(f"Error in portfolio broadcast loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission."""
        pass


class TradesHandler(WebSocketEventHandler):
    """Handler for trade execution WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.TRADES, SubscriptionLevel.TRADING)
        self.update_interval = 0.5  # 500ms updates for trades

    async def start(self) -> None:
        """Start trade execution broadcasting."""
        await super().start()
        self._task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Trade execution broadcast loop."""
        while self._running:
            try:
                # Mock trade data - in production, get from API facade
                trade_data = {
                    "type": "trade_execution",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "processing_mode": "realtime",
                    "data_format": "trade_event_v1",
                    "message_pattern": "event_stream",
                    "data": {
                        "trade_id": f"trade_{datetime.now().strftime('%H%M%S')}",
                        "symbol": "BTC/USDT",
                        "side": "buy",
                        "quantity": "0.001",
                        "price": "45123.50",
                        "value": "45.12",
                        "commission": "0.045",
                        "execution_time": datetime.now(timezone.utc).isoformat(),
                        "status": "filled",
                        "strategy": "momentum_v1",
                    },
                }

                if self.subscribers:
                    await self._emit_data("trade_executed", trade_data)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in trades broadcast loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission."""
        pass


class OrdersHandler(WebSocketEventHandler):
    """Handler for order status WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.ORDERS, SubscriptionLevel.TRADING)
        self.update_interval = 1.0  # 1 second updates for orders

    async def start(self) -> None:
        """Start order status broadcasting."""
        await super().start()
        self._task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Order status broadcast loop."""
        while self._running:
            try:
                # Mock order data - in production, get from API facade
                order_data = {
                    "type": "order_status_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "processing_mode": "realtime",
                    "data_format": "order_event_v1",
                    "message_pattern": "state_change",
                    "data": {
                        "order_id": f"order_{datetime.now().strftime('%H%M%S')}",
                        "client_order_id": f"client_{datetime.now().strftime('%H%M%S')}",
                        "symbol": "ETH/USDT",
                        "side": "sell",
                        "type": "limit",
                        "quantity": "0.5",
                        "price": "2501.25",
                        "filled_quantity": "0.2",
                        "remaining_quantity": "0.3",
                        "status": "partially_filled",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }

                if self.subscribers:
                    await self._emit_data("order_update", order_data)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in orders broadcast loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission."""
        pass


class AlertsHandler(WebSocketEventHandler):
    """Handler for alerts and notifications WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.ALERTS, SubscriptionLevel.USER)
        self.update_interval = 3.0  # 3 second check for alerts

    async def start(self) -> None:
        """Start alerts broadcasting."""
        await super().start()
        self._task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Alerts broadcast loop."""
        while self._running:
            try:
                # Mock alert data - in production, get from API facade
                alert_data = {
                    "type": "system_alert",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "processing_mode": "alert",
                    "data_format": "alert_event_v1",
                    "message_pattern": "notification",
                    "data": {
                        "alert_id": f"alert_{datetime.now().strftime('%H%M%S')}",
                        "severity": "medium",
                        "category": "risk_management",
                        "title": "Position Size Warning",
                        "message": "BTC position approaching 5% portfolio limit",
                        "symbol": "BTC/USDT",
                        "current_exposure": "4.8%",
                        "threshold": "5.0%",
                        "action_required": False,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "expires_at": (datetime.now(timezone.utc).timestamp() + 3600),
                    },
                }

                if self.subscribers:
                    await self._emit_data("new_alert", alert_data)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in alerts broadcast loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission."""
        pass


class LogsHandler(WebSocketEventHandler):
    """Handler for system logs WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.LOGS, SubscriptionLevel.ADMIN)
        self.update_interval = 2.0  # 2 second updates for logs

    async def start(self) -> None:
        """Start logs broadcasting."""
        await super().start()
        self._task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Logs broadcast loop."""
        while self._running:
            try:
                # Mock log data - in production, get from API facade
                log_data = {
                    "type": "system_log",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "processing_mode": "logging",
                    "data_format": "log_event_v1",
                    "message_pattern": "stream",
                    "data": {
                        "log_id": f"log_{datetime.now().strftime('%H%M%S')}",
                        "level": "INFO",
                        "component": "execution_engine",
                        "message": "Order execution completed successfully",
                        "context": {
                            "order_id": "order_123456",
                            "symbol": "BTC/USDT",
                            "execution_time_ms": 142,
                        },
                        "correlation_id": f"corr_{datetime.now().strftime('%H%M%S')}",
                        "thread_id": "exec_thread_1",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                }

                if self.subscribers:
                    await self._emit_data("system_log", log_data)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in logs broadcast loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission."""
        pass


class RiskMetricsHandler(WebSocketEventHandler):
    """Handler for risk metrics WebSocket events."""

    def __init__(self):
        super().__init__(ChannelType.RISK_METRICS, SubscriptionLevel.USER)
        self.update_interval = 10.0  # 10 second updates for risk metrics

    async def start(self) -> None:
        """Start risk metrics broadcasting."""
        await super().start()
        self._task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Risk metrics broadcast loop."""
        while self._running:
            try:
                # Mock risk metrics data - in production, get from API facade
                risk_data = {
                    "type": "risk_metrics_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "processing_mode": "analytics",
                    "data_format": "risk_metrics_v1",
                    "message_pattern": "periodic_update",
                    "data": {
                        "portfolio_var_1d": "1234.56",  # Using strings for Decimal values
                        "portfolio_var_5d": "2456.78",
                        "max_drawdown": "0.08",
                        "sharpe_ratio": "1.24",
                        "total_exposure": "0.78",
                        "risk_score": "6.5",
                        "position_risks": {
                            "BTC/USDT": {
                                "position_size": "0.1",
                                "var_1d": "234.56",
                                "correlation_score": "0.85",
                            },
                            "ETH/USDT": {
                                "position_size": "2.5",
                                "var_1d": "456.78",
                                "correlation_score": "0.72",
                            },
                        },
                        "risk_limits": {
                            "portfolio_limit": "10000.00",
                            "position_limit": "2000.00",
                            "var_limit": "500.00",
                        },
                        "calculated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }

                if self.subscribers:
                    await self._emit_data("risk_metrics", risk_data)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in risk metrics broadcast loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _emit_data(self, event: str, data: Any) -> None:
        """Placeholder for data emission."""
        pass


class UnifiedWebSocketNamespace(AsyncNamespace):
    """Unified namespace for all WebSocket communications."""

    def __init__(self, namespace: str = "/"):
        super().__init__(namespace)
        self.logger = BaseComponent().logger
        self.connected_clients: dict[str, dict[str, Any]] = {}
        self.authenticated_sessions: set[str] = set()
        self.session_subscriptions: dict[str, set[ChannelType]] = {}

        # Initialize handlers
        self.handlers: dict[ChannelType, WebSocketEventHandler] = {
            ChannelType.MARKET_DATA: MarketDataHandler(),
            ChannelType.BOT_STATUS: BotStatusHandler(),
            ChannelType.PORTFOLIO: PortfolioHandler(),
            ChannelType.TRADES: TradesHandler(),
            ChannelType.ORDERS: OrdersHandler(),
            ChannelType.ALERTS: AlertsHandler(),
            ChannelType.LOGS: LogsHandler(),
            ChannelType.RISK_METRICS: RiskMetricsHandler(),
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
        self, sid: str, environ: dict[str, Any] | None = None, auth: dict[str, Any] | None = None
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
        """Handle client disconnection with proper cleanup and concurrency."""
        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            self.logger.info(f"Client disconnected: {sid}")

        # Unsubscribe from all channels concurrently
        cleanup_tasks = []
        if sid in self.session_subscriptions:
            channels_to_unsubscribe = list(
                self.session_subscriptions[sid]
            )  # Snapshot to avoid race conditions
            for channel in channels_to_unsubscribe:
                cleanup_tasks.append(self._unsubscribe_from_channel(sid, channel))

        # Execute all unsubscribe operations concurrently with timeout
        if cleanup_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout during channel cleanup for client {sid}")
            except Exception as e:
                self.logger.warning(f"Error during channel cleanup for client {sid}: {e}")

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
        """Authenticate a session with the provided token with timeout and proper error handling."""
        try:
            # Use timeout for authentication process
            try:
                # JWT validation implementation needed
                # For now, accept any non-empty token
                if token:
                    self.connected_clients[sid]["authenticated"] = True
                    self.connected_clients[sid]["user_id"] = "user123"  # Mock user ID
                    self.connected_clients[sid]["permissions"] = {
                        "user",
                        "trading",
                    }  # Mock permissions
                    self.authenticated_sessions.add(sid)

                    # Use timeout for emit operation
                    try:
                        await asyncio.wait_for(
                            self.emit(
                                "authenticated",
                                {
                                    "status": "success",
                                    "user_id": "user123",
                                    "permissions": list(self.connected_clients[sid]["permissions"]),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                                room=sid,
                            ),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        self.logger.error(f"Failed to send auth success message to {sid}")

                    return True
            except asyncio.TimeoutError:
                self.logger.warning(f"Authentication timeout for session {sid}")
                try:
                    await asyncio.wait_for(
                        self.emit("auth_error", {"error": "Authentication timeout"}, room=sid),
                        timeout=3.0
                    )
                except asyncio.TimeoutError:
                    self.logger.error(f"Failed to send auth timeout message to {sid}")
        except Exception as e:
            try:
                await asyncio.wait_for(
                    self.emit(
                        "auth_error", {"error": f"Authentication failed: {e!s}"}, room=sid
                    ),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Failed to send auth error message to {sid}")

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
        """Stop all event handlers with proper timeout and error handling."""
        # Stop all handlers concurrently with timeout
        stop_tasks = [handler.stop() for handler in self.handlers.values()]
        if stop_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*stop_tasks, return_exceptions=True), timeout=30.0)  # Overall timeout for stopping all handlers
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Timeout stopping WebSocket handlers, some handlers may not have stopped cleanly"
                )
                # Force stop any remaining handlers
                for handler in self.handlers.values():
                    if handler._task and not handler._task.done():
                        handler._task.cancel()
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket handlers: {e}")


class UnifiedWebSocketManager(BaseComponent):
    """Unified manager for all WebSocket communications."""

    def __init__(self, api_facade=None):
        super().__init__()
        self.sio: AsyncServer | None = None
        self.namespace: UnifiedWebSocketNamespace | None = None
        self.api_facade = api_facade
        self._running = False

    def configure_dependencies(self, injector):
        """Configure dependencies through DI if not provided in constructor."""
        if self.api_facade is None:
            try:
                self.api_facade = injector.resolve("APIFacade")
            except Exception as e:
                # API facade is optional for fallback
                self.logger.debug(f"Could not resolve APIFacade from DI container: {e}")
                pass

    def create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer:
        """Create and configure unified Socket.IO server."""
        if cors_allowed_origins is None:
            cors_allowed_origins = ["http://localhost:3000", "http://localhost:8000"]

        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=cors_allowed_origins,
            logger=False,  # Use our own logging
            engineio_logger=False,
            ping_timeout=30,  # Reduced timeout for faster detection
            ping_interval=15,  # More frequent pings
            max_http_buffer_size=1000000,
            allow_upgrades=True,
            compression_threshold=1024,
            # Connection management
            always_connect=False,
            transports=["websocket", "polling"],
        )

        # Create and register unified namespace
        self.namespace = UnifiedWebSocketNamespace("/")
        self.sio.register_namespace(self.namespace)

        correlation_id = correlation_context.generate_correlation_id()
        with correlation_context.correlation_context(correlation_id):
            self.logger.info("Unified WebSocket server created")

        return self.sio

    async def start(self):
        """Start the unified WebSocket manager with proper error handling."""
        if self._running or not self.namespace:
            return

        self._running = True

        try:
            # Start handlers with timeout
            await asyncio.wait_for(self.namespace.start_handlers(), timeout=20.0)

            correlation_id = correlation_context.generate_correlation_id()
            with correlation_context.correlation_context(correlation_id):
                self.logger.info("Unified WebSocket manager started")
        except asyncio.TimeoutError:
            self.logger.error("Timeout starting WebSocket handlers")
            self._running = False
            raise
        except Exception as e:
            self.logger.error(f"Error starting unified WebSocket manager: {e}")
            self._running = False
            raise

    async def stop(self):
        """Stop the unified WebSocket manager with proper cleanup."""
        if not self._running or not self.namespace:
            return

        self._running = False

        try:
            # Stop handlers with timeout
            await asyncio.wait_for(self.namespace.stop_handlers(), timeout=30.0)
        except asyncio.TimeoutError:
            self.logger.warning("Timeout stopping WebSocket handlers during shutdown")
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket handlers: {e}")
        finally:
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


def get_unified_websocket_manager(api_facade: "APIFacade" = None) -> UnifiedWebSocketManager:
    """Get or create the unified WebSocket manager."""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedWebSocketManager(api_facade=api_facade)
    return _unified_manager
