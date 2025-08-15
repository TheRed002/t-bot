"""
Bot Status WebSocket handlers for T-Bot web interface.

This module provides real-time bot status updates including state changes,
performance metrics, trade notifications, and bot health monitoring.
"""

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Import authentication function from market_data module
from .market_data import authenticate_websocket


class BotStatusManager:
    """Manages WebSocket connections for bot status updates."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.bot_subscriptions: dict[str, set[str]] = {}  # user_id -> set of bot_ids
        self.user_subscriptions: dict[str, set[str]] = {}  # bot_id -> set of user_ids

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.bot_subscriptions[user_id] = set()
        logger.info("Bot status WebSocket connected", user_id=user_id)

    def disconnect(self, user_id: str):
        """Disconnect WebSocket and cleanup subscriptions."""
        if user_id in self.active_connections:
            # Remove from bot subscriptions
            for bot_id in self.bot_subscriptions.get(user_id, set()):
                if bot_id in self.user_subscriptions:
                    self.user_subscriptions[bot_id].discard(user_id)
                    if not self.user_subscriptions[bot_id]:
                        del self.user_subscriptions[bot_id]

            # Remove user subscriptions
            del self.active_connections[user_id]
            if user_id in self.bot_subscriptions:
                del self.bot_subscriptions[user_id]

            logger.info("Bot status WebSocket disconnected", user_id=user_id)

    def subscribe_to_bot(self, user_id: str, bot_id: str):
        """Subscribe user to bot status updates."""
        if user_id in self.bot_subscriptions:
            self.bot_subscriptions[user_id].add(bot_id)

            if bot_id not in self.user_subscriptions:
                self.user_subscriptions[bot_id] = set()
            self.user_subscriptions[bot_id].add(user_id)

            logger.info("User subscribed to bot", user_id=user_id, bot_id=bot_id)

    def unsubscribe_from_bot(self, user_id: str, bot_id: str):
        """Unsubscribe user from bot status updates."""
        if user_id in self.bot_subscriptions:
            self.bot_subscriptions[user_id].discard(bot_id)

        if bot_id in self.user_subscriptions:
            self.user_subscriptions[bot_id].discard(user_id)
            if not self.user_subscriptions[bot_id]:
                del self.user_subscriptions[bot_id]

        logger.info("User unsubscribed from bot", user_id=user_id, bot_id=bot_id)

    def subscribe_to_all_bots(self, user_id: str):
        """Subscribe user to all bot updates."""
        # This would typically get all bot IDs from the orchestrator
        # For now, use mock bot IDs
        mock_bot_ids = ["bot_001", "bot_002", "bot_003"]

        for bot_id in mock_bot_ids:
            self.subscribe_to_bot(user_id, bot_id)

    async def send_to_user(self, user_id: str, message: dict):
        """Send message to specific user."""
        if user_id in self.active_connections:
            try:
                websocket = self.active_connections[user_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(
                    "Failed to send bot status message to user", user_id=user_id, error=str(e)
                )
                self.disconnect(user_id)

    async def broadcast_to_bot_subscribers(self, bot_id: str, message: dict):
        """Broadcast message to all subscribers of a bot."""
        if bot_id in self.user_subscriptions:
            disconnected_users = []

            for user_id in self.user_subscriptions[bot_id]:
                try:
                    websocket = self.active_connections.get(user_id)
                    if websocket:
                        await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(
                        "Failed to broadcast bot status to user", user_id=user_id, error=str(e)
                    )
                    disconnected_users.append(user_id)

            # Clean up disconnected users
            for user_id in disconnected_users:
                self.disconnect(user_id)


# Global bot status manager
bot_manager = BotStatusManager()


class BotStatusMessage(BaseModel):
    """Base model for bot status messages."""

    type: str
    bot_id: str
    timestamp: datetime
    data: dict


class BotSubscriptionMessage(BaseModel):
    """Model for bot subscription messages."""

    action: str  # 'subscribe', 'unsubscribe', 'subscribe_all'
    bot_ids: list[str] | None = None
    update_types: list[str] = ["status", "metrics", "trades", "errors"]


@router.websocket("/bot-status")
async def bot_status_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bot status updates.

    Args:
        websocket: WebSocket connection
    """
    # Authenticate connection
    user = await authenticate_websocket(websocket)
    if not user:
        return

    # Connect to manager
    await bot_manager.connect(websocket, user.user_id)

    try:
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "message": "Connected to T-Bot bot status stream",
            "user_id": user.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "available_update_types": ["status", "metrics", "trades", "errors", "alerts"],
            "instructions": {
                "subscribe_all": 'Send {"action": "subscribe_all"}',
                "subscribe_specific": 'Send {"action": "subscribe", "bot_ids": ["bot_001"]}',
                "unsubscribe": 'Send {"action": "unsubscribe", "bot_ids": ["bot_001"]}',
            },
        }
        await bot_manager.send_to_user(user.user_id, welcome_message)

        # Listen for subscription messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("action") == "subscribe_all":
                    bot_manager.subscribe_to_all_bots(user.user_id)

                    # Send initial status for all bots
                    await send_all_bot_status(user.user_id)

                    response = {
                        "type": "subscription_confirmed",
                        "scope": "all_bots",
                        "update_types": message.get("update_types", ["status", "metrics"]),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await bot_manager.send_to_user(user.user_id, response)

                elif message.get("action") == "subscribe":
                    bot_ids = message.get("bot_ids", [])
                    update_types = message.get("update_types", ["status", "metrics"])

                    for bot_id in bot_ids:
                        bot_manager.subscribe_to_bot(user.user_id, bot_id)
                        await send_initial_bot_status(user.user_id, bot_id, update_types)

                    response = {
                        "type": "subscription_confirmed",
                        "bot_ids": bot_ids,
                        "update_types": update_types,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await bot_manager.send_to_user(user.user_id, response)

                elif message.get("action") == "unsubscribe":
                    bot_ids = message.get("bot_ids", [])

                    for bot_id in bot_ids:
                        bot_manager.unsubscribe_from_bot(user.user_id, bot_id)

                    response = {
                        "type": "unsubscription_confirmed",
                        "bot_ids": bot_ids,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await bot_manager.send_to_user(user.user_id, response)

                elif message.get("action") == "ping":
                    pong_message = {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                    await bot_manager.send_to_user(user.user_id, pong_message)

            except json.JSONDecodeError:
                error_message = {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await bot_manager.send_to_user(user.user_id, error_message)

            except Exception as e:
                logger.error(
                    f"Error processing bot status WebSocket message: {e}", user_id=user.user_id
                )
                error_message = {
                    "type": "error",
                    "message": f"Message processing error: {e!s}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await bot_manager.send_to_user(user.user_id, error_message)

    except WebSocketDisconnect:
        logger.info("Bot status WebSocket disconnected normally", user_id=user.user_id)
    except Exception as e:
        logger.error(f"Bot status WebSocket error: {e}", user_id=user.user_id)
    finally:
        bot_manager.disconnect(user.user_id)


async def send_initial_bot_status(user_id: str, bot_id: str, update_types: list[str]):
    """Send initial status for a specific bot."""
    try:
        import random

        if "status" in update_types:
            status_data = {
                "type": "bot_status",
                "bot_id": bot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "status": random.choice(["running", "stopped", "paused", "error"]),
                    "uptime": f"{random.randint(1, 168)} hours",
                    "last_trade": (datetime.utcnow()).isoformat(),
                    "strategy": "trend_following",
                    "exchange": "binance",
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "health": "healthy",
                },
            }
            await bot_manager.send_to_user(user_id, status_data)

        if "metrics" in update_types:
            metrics_data = {
                "type": "bot_metrics",
                "bot_id": bot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "total_trades": random.randint(10, 100),
                    "winning_trades": random.randint(5, 70),
                    "total_pnl": random.uniform(-1000, 5000),
                    "daily_pnl": random.uniform(-200, 800),
                    "win_rate": random.uniform(0.5, 0.8),
                    "average_trade_duration": f"{random.uniform(1, 8):.1f} hours",
                    "allocated_capital": 10000.0,
                    "current_exposure": random.uniform(5000, 9000),
                    "max_drawdown": random.uniform(-500, -100),
                },
            }
            await bot_manager.send_to_user(user_id, metrics_data)

        if "trades" in update_types:
            # Send recent trades
            trades = []
            for i in range(3):
                trades.append(
                    {
                        "trade_id": f"trade_{bot_id}_{i + 1}",
                        "symbol": random.choice(["BTCUSDT", "ETHUSDT"]),
                        "side": random.choice(["buy", "sell"]),
                        "quantity": random.uniform(0.1, 2.0),
                        "price": random.uniform(40000, 50000),
                        "pnl": random.uniform(-100, 300),
                        "executed_at": datetime.utcnow().isoformat(),
                    }
                )

            trades_data = {
                "type": "bot_trades",
                "bot_id": bot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"recent_trades": trades},
            }
            await bot_manager.send_to_user(user_id, trades_data)

    except Exception as e:
        logger.error(f"Failed to send initial bot status: {e}", user_id=user_id, bot_id=bot_id)


async def send_all_bot_status(user_id: str):
    """Send initial status for all bots."""
    mock_bot_ids = ["bot_001", "bot_002", "bot_003"]

    for bot_id in mock_bot_ids:
        await send_initial_bot_status(user_id, bot_id, ["status", "metrics"])


# Background task to simulate bot status updates
async def bot_status_simulator():
    """Simulate real-time bot status updates."""
    import random

    mock_bot_ids = ["bot_001", "bot_002", "bot_003"]

    while True:
        try:
            for bot_id in mock_bot_ids:
                if bot_manager.user_subscriptions.get(bot_id):
                    # Randomly send different types of updates
                    update_type = random.choice(["metrics", "trade", "status_change", "alert"])

                    if update_type == "metrics":
                        metrics_update = {
                            "type": "bot_metrics_update",
                            "bot_id": bot_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {
                                "daily_pnl": random.uniform(-200, 800),
                                "total_pnl": random.uniform(-1000, 5000),
                                "current_exposure": random.uniform(5000, 9000),
                                "recent_performance": random.uniform(-5, 15),
                            },
                        }
                        await bot_manager.broadcast_to_bot_subscribers(bot_id, metrics_update)

                    elif update_type == "trade":
                        trade_update = {
                            "type": "bot_trade_executed",
                            "bot_id": bot_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {
                                "trade_id": f"trade_{bot_id}_{random.randint(1000, 9999)}",
                                "symbol": random.choice(["BTCUSDT", "ETHUSDT"]),
                                "side": random.choice(["buy", "sell"]),
                                "quantity": random.uniform(0.1, 2.0),
                                "price": random.uniform(40000, 50000),
                                "pnl": random.uniform(-100, 300),
                                "strategy_signal": random.choice(
                                    ["trend_up", "trend_down", "reversion"]
                                ),
                            },
                        }
                        await bot_manager.broadcast_to_bot_subscribers(bot_id, trade_update)

                    elif update_type == "status_change":
                        if random.random() < 0.1:  # 10% chance of status change
                            status_update = {
                                "type": "bot_status_changed",
                                "bot_id": bot_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": {
                                    "old_status": "running",
                                    "new_status": random.choice(["paused", "error", "stopped"]),
                                    "reason": random.choice(
                                        [
                                            "User requested pause",
                                            "Risk limit exceeded",
                                            "Exchange connection error",
                                            "Manual intervention",
                                        ]
                                    ),
                                },
                            }
                            await bot_manager.broadcast_to_bot_subscribers(bot_id, status_update)

                    elif update_type == "alert":
                        if random.random() < 0.05:  # 5% chance of alert
                            alert_update = {
                                "type": "bot_alert",
                                "bot_id": bot_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": {
                                    "alert_type": random.choice(["warning", "error", "info"]),
                                    "message": random.choice(
                                        [
                                            "High volatility detected in market",
                                            "Profit target reached",
                                            "Stop loss triggered",
                                            "Low liquidity warning",
                                        ]
                                    ),
                                    "severity": random.choice(["low", "medium", "high"]),
                                },
                            }
                            await bot_manager.broadcast_to_bot_subscribers(bot_id, alert_update)

            # Wait before next update cycle
            await asyncio.sleep(random.uniform(2, 8))  # Variable interval

        except Exception as e:
            logger.error(f"Bot status simulator error: {e}")
            await asyncio.sleep(5)


@router.get("/bot-status/connections")
async def get_bot_status_connections():
    """Get bot status WebSocket connection information."""
    return {
        "active_connections": len(bot_manager.active_connections),
        "total_bot_subscriptions": sum(
            len(subs) for subs in bot_manager.bot_subscriptions.values()
        ),
        "bots_with_subscribers": list(bot_manager.user_subscriptions.keys()),
        "subscription_details": {
            user_id: list(bot_ids) for user_id, bot_ids in bot_manager.bot_subscriptions.items()
        },
    }
