"""
Market Data WebSocket handlers for T-Bot web interface.

This module provides real-time market data streaming including prices,
order book updates, trade feeds, and market statistics.
"""

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from src.core.logging import get_logger
from src.web_interface.security.auth import User

logger = get_logger(__name__)
router = APIRouter()


# Connection manager for WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections for market data streaming."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.subscriptions: dict[str, set[str]] = {}  # user_id -> set of symbols
        self.symbol_subscribers: dict[str, set[str]] = {}  # symbol -> set of user_ids

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept WebSocket connection with timeout and proper error handling."""
        try:
            # Use timeout for WebSocket accept to prevent hanging
            await asyncio.wait_for(websocket.accept(), timeout=10.0)
            self.active_connections[user_id] = websocket
            self.subscriptions[user_id] = set()
            logger.info("WebSocket connected", user_id=user_id)
        except asyncio.TimeoutError:
            logger.error(f"Timeout accepting market data WebSocket connection for {user_id}")
            raise
        except Exception as e:
            logger.error(f"Failed to accept market data WebSocket connection: {e}", user_id=user_id)
            raise

    def disconnect(self, user_id: str):
        """Disconnect WebSocket and cleanup subscriptions."""
        if user_id in self.active_connections:
            # Remove from symbol subscriptions
            for symbol in self.subscriptions.get(user_id, set()):
                if symbol in self.symbol_subscribers:
                    self.symbol_subscribers[symbol].discard(user_id)
                    if not self.symbol_subscribers[symbol]:
                        del self.symbol_subscribers[symbol]

            # Remove user subscriptions
            del self.active_connections[user_id]
            if user_id in self.subscriptions:
                del self.subscriptions[user_id]

            logger.info("WebSocket disconnected", user_id=user_id)

    def subscribe_to_symbol(self, user_id: str, symbol: str):
        """Subscribe user to symbol updates."""
        if user_id in self.subscriptions:
            self.subscriptions[user_id].add(symbol)

            if symbol not in self.symbol_subscribers:
                self.symbol_subscribers[symbol] = set()
            self.symbol_subscribers[symbol].add(user_id)

            logger.info("User subscribed to symbol", user_id=user_id, symbol=symbol)

    def unsubscribe_from_symbol(self, user_id: str, symbol: str):
        """Unsubscribe user from symbol updates."""
        if user_id in self.subscriptions:
            self.subscriptions[user_id].discard(symbol)

        if symbol in self.symbol_subscribers:
            self.symbol_subscribers[symbol].discard(user_id)
            if not self.symbol_subscribers[symbol]:
                del self.symbol_subscribers[symbol]

        logger.info("User unsubscribed from symbol", user_id=user_id, symbol=symbol)

    async def send_to_user(self, user_id: str, message: dict):
        """Send message to specific user with proper error handling and cleanup."""
        if user_id not in self.active_connections:
            return

        websocket = self.active_connections[user_id]
        try:
            # Use asyncio timeout to prevent blocking
            # Send message with timeout for better responsiveness
            message_json = json.dumps(message, default=str)  # Handle Decimal serialization
            await asyncio.wait_for(websocket.send_text(message_json), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout sending message to user", user_id=user_id)
            self.disconnect(user_id)
        except Exception as e:
            logger.error("Failed to send message to user", user_id=user_id, error=str(e))
            # Remove broken connection
            self.disconnect(user_id)

    async def broadcast_to_symbol_subscribers(self, symbol: str, message: dict):
        """Broadcast message to all subscribers of a symbol with backpressure handling."""
        if symbol not in self.symbol_subscribers or not self.symbol_subscribers[symbol]:
            return

        # Create snapshot to avoid race conditions during iteration
        subscribers_snapshot = list(self.symbol_subscribers[symbol])
        if not subscribers_snapshot:
            return

        # Prepare message once for all subscribers
        try:
            message_json = json.dumps(message, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize message for symbol {symbol}: {e}")
            return

        send_tasks = []
        # Collect all send tasks for active connections
        for user_id in subscribers_snapshot:
            websocket = self.active_connections.get(user_id)
            if websocket:
                send_tasks.append(
                    self._send_with_timeout_broadcast(websocket, message_json, user_id)
                )

        # Execute all sends concurrently with overall timeout
        if send_tasks:
            try:
                # Execute with overall broadcast timeout
                results = await asyncio.wait_for(asyncio.gather(*send_tasks, return_exceptions=True), timeout=10.0)

                # Clean up failed connections
                for i, result in enumerate(results):
                    if isinstance(result, Exception) and i < len(subscribers_snapshot):
                        user_id = subscribers_snapshot[i]
                        logger.debug(
                            f"Disconnecting user {user_id} due to send failure: {result}"
                        )
                        self.disconnect(user_id)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout broadcasting to {symbol} subscribers, cleaning up all connections"
                )
                # Clean up all subscribers on broadcast timeout
                for user_id in subscribers_snapshot:
                    self.disconnect(user_id)

    async def _send_with_timeout(self, websocket: WebSocket, message: dict, user_id: str):
        """Send message with timeout handling."""
        try:
            message_json = json.dumps(message, default=str)
            await asyncio.wait_for(websocket.send_text(message_json), timeout=3.0)  # Shorter timeout for individual sends
        except Exception as e:
            logger.error(f"Failed to send to user {user_id}: {e}")
            raise

    async def _send_with_timeout_broadcast(
        self, websocket: WebSocket, message_json: str, user_id: str
    ):
        """Send pre-serialized message with timeout handling for broadcasts."""
        try:
            await asyncio.wait_for(websocket.send_text(message_json), timeout=3.0)
        except Exception as e:
            logger.debug(f"Failed to broadcast to user {user_id}: {e}")
            raise


# Global connection manager
manager = ConnectionManager()


class MarketDataMessage(BaseModel):
    """Base model for market data messages."""

    type: str
    symbol: str
    timestamp: datetime
    data: dict


class SubscriptionMessage(BaseModel):
    """Model for subscription messages."""

    action: str  # 'subscribe' or 'unsubscribe'
    symbols: list[str]
    data_types: list[str] = ["ticker", "orderbook", "trades"]


async def authenticate_websocket(websocket: WebSocket) -> User | None:
    """Authenticate WebSocket connection with proper async handling and timeouts."""
    try:
        # Use timeout for the entire authentication process
        # Get token from query parameters or headers
        token = websocket.query_params.get("token")
        if not token:
            # Try to get from headers (if supported by client)
            headers = websocket.headers
            auth_header = headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]

        if not token:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required"
            )
            return None

        # Validate token with proper async handling
        from src.web_interface.security.auth import jwt_handler

        if jwt_handler:
            # Check if validate_token is async
            if callable(jwt_handler.validate_token) and asyncio.iscoroutinefunction(
                jwt_handler.validate_token
            ):
                token_data = await jwt_handler.validate_token(token)
            else:
                # Run synchronous validation in thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                token_data = await loop.run_in_executor(None, jwt_handler.validate_token, token)

            if token_data:
                # Create user object
                user = User(
                    id=token_data.user_id,
                    username=token_data.username,
                    email="user@example.com",  # Simplified
                    is_active=True,
                    is_verified=True,  # Add required field
                    scopes=token_data.scopes,
                )
                return user

        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed"
        )
        return None

    except asyncio.TimeoutError:
        logger.warning("WebSocket authentication timeout")
        try:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason="Authentication timeout"
            )
        except Exception:
            pass
        return None
    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        try:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason="Authentication error"
            )
        except Exception:
            pass
        return None


@router.websocket("/market-data")
async def market_data_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time market data streaming.

    Args:
        websocket: WebSocket connection
    """
    # Authenticate connection
    user = await authenticate_websocket(websocket)
    if not user:
        return

    # Connect to manager
    await manager.connect(websocket, user.user_id)

    try:
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "message": "Connected to T-Bot market data stream",
            "user_id": user.user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "available_data_types": ["ticker", "orderbook", "trades", "candles"],
            "instructions": {
                "subscribe": 'Send {"action": "subscribe", "symbols": ["BTCUSDT"], "data_types": ["ticker"]}',
                "unsubscribe": 'Send {"action": "unsubscribe", "symbols": ["BTCUSDT"]}',
            },
        }
        await manager.send_to_user(user.user_id, welcome_message)

        # Listen for subscription messages with timeout
        while True:
            try:
                # Add timeout to prevent hanging on receive
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=300.0
                )  # 5 min timeout
                message = json.loads(data)

                if message.get("action") == "subscribe":
                    symbols = message.get("symbols", [])
                    data_types = message.get("data_types", ["ticker"])

                    for symbol in symbols:
                        manager.subscribe_to_symbol(user.user_id, symbol)

                        # Send initial data for subscribed symbol
                        await send_initial_market_data(user.user_id, symbol, data_types)

                    # Confirm subscription
                    response = {
                        "type": "subscription_confirmed",
                        "symbols": symbols,
                        "data_types": data_types,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await manager.send_to_user(user.user_id, response)

                elif message.get("action") == "unsubscribe":
                    symbols = message.get("symbols", [])

                    for symbol in symbols:
                        manager.unsubscribe_from_symbol(user.user_id, symbol)

                    # Confirm unsubscription
                    response = {
                        "type": "unsubscription_confirmed",
                        "symbols": symbols,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await manager.send_to_user(user.user_id, response)

                elif message.get("action") == "ping":
                    # Respond to ping with pong
                    pong_message = {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await manager.send_to_user(user.user_id, pong_message)

            except asyncio.TimeoutError:
                # Send ping to check connection
                ping_message = {
                    "type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await manager.send_to_user(user.user_id, ping_message)

            except json.JSONDecodeError:
                error_message = {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await manager.send_to_user(user.user_id, error_message)

            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}", user_id=user.user_id)
                error_message = {
                    "type": "error",
                    "message": f"Message processing error: {e!s}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await manager.send_to_user(user.user_id, error_message)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally", user_id=user.user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", user_id=user.user_id)
    finally:
        manager.disconnect(user.user_id)


async def send_initial_market_data(user_id: str, symbol: str, data_types: list[str]):
    """Send initial market data for newly subscribed symbol."""
    try:
        # Mock initial data (in production, get from exchange APIs)
        import random

        base_price = 45000.0 if "BTC" in symbol else 3000.0
        current_price = base_price * (1 + random.uniform(-0.02, 0.02))

        if "ticker" in data_types:
            ticker_data = {
                "type": "ticker",
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "price": current_price,
                    "bid": current_price * 0.9999,
                    "ask": current_price * 1.0001,
                    "volume_24h": random.uniform(1000000, 5000000),
                    "change_24h": random.uniform(-5, 5),
                    "change_24h_percentage": random.uniform(-5, 5),
                    "high_24h": current_price * 1.05,
                    "low_24h": current_price * 0.95,
                },
            }
            await manager.send_to_user(user_id, ticker_data)

        if "orderbook" in data_types:
            # Generate mock order book
            bids = []
            asks = []

            for i in range(10):
                bid_price = current_price * (1 - 0.0001 * (i + 1))
                ask_price = current_price * (1 + 0.0001 * (i + 1))

                bids.append([bid_price, random.uniform(0.1, 5.0)])
                asks.append([ask_price, random.uniform(0.1, 5.0)])

            orderbook_data = {
                "type": "orderbook",
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"bids": bids, "asks": asks},
            }
            await manager.send_to_user(user_id, orderbook_data)

        if "trades" in data_types:
            # Generate mock recent trades
            trades = []
            for i in range(5):
                trade_time = datetime.now(timezone.utc)
                trades.append(
                    {
                        "id": f"trade_{i + 1}",
                        "price": current_price * (1 + random.uniform(-0.001, 0.001)),
                        "quantity": random.uniform(0.1, 2.0),
                        "side": random.choice(["buy", "sell"]),
                        "timestamp": trade_time.isoformat(),
                    }
                )

            trades_data = {
                "type": "trades",
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"trades": trades},
            }
            await manager.send_to_user(user_id, trades_data)

    except Exception as e:
        logger.error(f"Failed to send initial market data: {e}", user_id=user_id, symbol=symbol)


# Background task to simulate real-time market data updates
async def market_data_simulator():
    """Simulate real-time market data updates."""
    import random

    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT"]

    while True:
        try:
            for symbol in symbols:
                if manager.symbol_subscribers.get(symbol):
                    # Generate mock price update
                    base_price = (
                        45000.0 if "BTC" in symbol else (3000.0 if "ETH" in symbol else 500.0)
                    )
                    current_price = base_price * (1 + random.uniform(-0.001, 0.001))

                    ticker_update = {
                        "type": "ticker_update",
                        "symbol": symbol,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "price": current_price,
                            "bid": current_price * 0.9999,
                            "ask": current_price * 1.0001,
                            "volume_24h": random.uniform(1000000, 5000000),
                            "change_24h": random.uniform(-0.5, 0.5),
                        },
                    }

                    await manager.broadcast_to_symbol_subscribers(symbol, ticker_update)

            # Wait before next update
            await asyncio.sleep(1)  # Update every second

        except Exception as e:
            logger.error(f"Market data simulator error: {e}")
            await asyncio.sleep(5)


# Start background task (in production, this would be managed by the application lifecycle)


@router.get("/market-data/status")
async def get_market_data_status():
    """Get market data WebSocket status."""
    return {
        "active_connections": len(manager.active_connections),
        "total_subscriptions": sum(len(subs) for subs in manager.subscriptions.values()),
        "symbols_with_subscribers": list(manager.symbol_subscribers.keys()),
        "connection_details": {
            user_id: list(symbols) for user_id, symbols in manager.subscriptions.items()
        },
    }
