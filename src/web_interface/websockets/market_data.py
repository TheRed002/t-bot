"""
Market Data WebSocket handlers for T-Bot web interface.

This module provides real-time market data streaming including prices,
order book updates, trade feeds, and market statistics.
"""

import asyncio
import json
from datetime import datetime

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
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.subscriptions[user_id] = set()
        logger.info("WebSocket connected", user_id=user_id)

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
        """Send message to specific user."""
        if user_id in self.active_connections:
            try:
                websocket = self.active_connections[user_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to send message to user", user_id=user_id, error=str(e))
                # Remove broken connection
                self.disconnect(user_id)

    async def broadcast_to_symbol_subscribers(self, symbol: str, message: dict):
        """Broadcast message to all subscribers of a symbol."""
        if symbol in self.symbol_subscribers:
            disconnected_users = []

            for user_id in self.symbol_subscribers[symbol]:
                try:
                    websocket = self.active_connections.get(user_id)
                    if websocket:
                        await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error("Failed to broadcast to user", user_id=user_id, error=str(e))
                    disconnected_users.append(user_id)

            # Clean up disconnected users
            for user_id in disconnected_users:
                self.disconnect(user_id)


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
    """Authenticate WebSocket connection."""
    try:
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

        # Validate token (simplified - in production use proper JWT validation)
        from src.web_interface.security.auth import jwt_handler

        if jwt_handler:
            token_data = jwt_handler.validate_token(token)

            # Create user object
            user = User(
                user_id=token_data.user_id,
                username=token_data.username,
                email="user@example.com",  # Simplified
                is_active=True,
                scopes=token_data.scopes,
            )
            return user

        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return None

    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication error")
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
            "timestamp": datetime.utcnow().isoformat(),
            "available_data_types": ["ticker", "orderbook", "trades", "candles"],
            "instructions": {
                "subscribe": 'Send {"action": "subscribe", "symbols": ["BTCUSDT"], "data_types": ["ticker"]}',
                "unsubscribe": 'Send {"action": "unsubscribe", "symbols": ["BTCUSDT"]}',
            },
        }
        await manager.send_to_user(user.user_id, welcome_message)

        # Listen for subscription messages
        while True:
            try:
                data = await websocket.receive_text()
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
                        "timestamp": datetime.utcnow().isoformat(),
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
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await manager.send_to_user(user.user_id, response)

                elif message.get("action") == "ping":
                    # Respond to ping with pong
                    pong_message = {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                    await manager.send_to_user(user.user_id, pong_message)

            except json.JSONDecodeError:
                error_message = {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await manager.send_to_user(user.user_id, error_message)

            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}", user_id=user.user_id)
                error_message = {
                    "type": "error",
                    "message": f"Message processing error: {e!s}",
                    "timestamp": datetime.utcnow().isoformat(),
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
                "timestamp": datetime.utcnow().isoformat(),
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
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"bids": bids, "asks": asks},
            }
            await manager.send_to_user(user_id, orderbook_data)

        if "trades" in data_types:
            # Generate mock recent trades
            trades = []
            for i in range(5):
                trade_time = datetime.utcnow()
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
                "timestamp": datetime.utcnow().isoformat(),
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
                        "timestamp": datetime.utcnow().isoformat(),
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
# asyncio.create_task(market_data_simulator())


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
