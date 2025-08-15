"""
Portfolio WebSocket handlers for T-Bot web interface.

This module provides real-time portfolio updates including position changes,
P&L updates, balance changes, and portfolio performance metrics.
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


class PortfolioManager:
    """Manages WebSocket connections for portfolio updates."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.subscriptions: dict[str, set[str]] = {}  # user_id -> set of update_types

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.subscriptions[user_id] = set()
        logger.info("Portfolio WebSocket connected", user_id=user_id)

    def disconnect(self, user_id: str):
        """Disconnect WebSocket and cleanup subscriptions."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            if user_id in self.subscriptions:
                del self.subscriptions[user_id]

            logger.info("Portfolio WebSocket disconnected", user_id=user_id)

    def subscribe_to_updates(self, user_id: str, update_types: list[str]):
        """Subscribe user to portfolio update types."""
        if user_id in self.subscriptions:
            self.subscriptions[user_id].update(update_types)
            logger.info(
                "User subscribed to portfolio updates", user_id=user_id, update_types=update_types
            )

    def unsubscribe_from_updates(self, user_id: str, update_types: list[str]):
        """Unsubscribe user from portfolio update types."""
        if user_id in self.subscriptions:
            for update_type in update_types:
                self.subscriptions[user_id].discard(update_type)
            logger.info(
                "User unsubscribed from portfolio updates",
                user_id=user_id,
                update_types=update_types,
            )

    async def send_to_user(self, user_id: str, message: dict):
        """Send message to specific user."""
        if user_id in self.active_connections:
            try:
                websocket = self.active_connections[user_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(
                    "Failed to send portfolio message to user", user_id=user_id, error=str(e)
                )
                self.disconnect(user_id)

    async def broadcast_to_all(self, message: dict, update_type: str):
        """Broadcast message to all users subscribed to update type."""
        disconnected_users = []

        for user_id, user_subscriptions in self.subscriptions.items():
            if update_type in user_subscriptions:
                try:
                    websocket = self.active_connections.get(user_id)
                    if websocket:
                        await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(
                        "Failed to broadcast portfolio update to user",
                        user_id=user_id,
                        error=str(e),
                    )
                    disconnected_users.append(user_id)

        # Clean up disconnected users
        for user_id in disconnected_users:
            self.disconnect(user_id)


# Global portfolio manager
portfolio_manager = PortfolioManager()


class PortfolioMessage(BaseModel):
    """Base model for portfolio messages."""

    type: str
    timestamp: datetime
    data: dict


class PortfolioSubscriptionMessage(BaseModel):
    """Model for portfolio subscription messages."""

    action: str  # 'subscribe' or 'unsubscribe'
    update_types: list[str] = ["positions", "pnl", "balances", "performance"]


@router.websocket("/portfolio")
async def portfolio_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time portfolio updates.

    Args:
        websocket: WebSocket connection
    """
    # Authenticate connection
    user = await authenticate_websocket(websocket)
    if not user:
        return

    # Connect to manager
    await portfolio_manager.connect(websocket, user.user_id)

    try:
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "message": "Connected to T-Bot portfolio stream",
            "user_id": user.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "available_update_types": ["positions", "pnl", "balances", "performance", "alerts"],
            "instructions": {
                "subscribe": 'Send {"action": "subscribe", "update_types": ["positions", "pnl"]}',
                "unsubscribe": 'Send {"action": "unsubscribe", "update_types": ["positions"]}',
            },
        }
        await portfolio_manager.send_to_user(user.user_id, welcome_message)

        # Listen for subscription messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("action") == "subscribe":
                    update_types = message.get("update_types", ["positions", "pnl"])
                    portfolio_manager.subscribe_to_updates(user.user_id, update_types)

                    # Send initial portfolio data
                    await send_initial_portfolio_data(user.user_id, update_types)

                    response = {
                        "type": "subscription_confirmed",
                        "update_types": update_types,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await portfolio_manager.send_to_user(user.user_id, response)

                elif message.get("action") == "unsubscribe":
                    update_types = message.get("update_types", [])
                    portfolio_manager.unsubscribe_from_updates(user.user_id, update_types)

                    response = {
                        "type": "unsubscription_confirmed",
                        "update_types": update_types,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await portfolio_manager.send_to_user(user.user_id, response)

                elif message.get("action") == "ping":
                    pong_message = {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                    await portfolio_manager.send_to_user(user.user_id, pong_message)

            except json.JSONDecodeError:
                error_message = {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await portfolio_manager.send_to_user(user.user_id, error_message)

            except Exception as e:
                logger.error(
                    f"Error processing portfolio WebSocket message: {e}", user_id=user.user_id
                )
                error_message = {
                    "type": "error",
                    "message": f"Message processing error: {e!s}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await portfolio_manager.send_to_user(user.user_id, error_message)

    except WebSocketDisconnect:
        logger.info("Portfolio WebSocket disconnected normally", user_id=user.user_id)
    except Exception as e:
        logger.error(f"Portfolio WebSocket error: {e}", user_id=user.user_id)
    finally:
        portfolio_manager.disconnect(user.user_id)


async def send_initial_portfolio_data(user_id: str, update_types: list[str]):
    """Send initial portfolio data for subscribed update types."""
    try:
        import random

        if "positions" in update_types:
            positions_data = {
                "type": "positions",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "positions": [
                        {
                            "symbol": "BTCUSDT",
                            "exchange": "binance",
                            "side": "long",
                            "quantity": 2.0,
                            "entry_price": 45000.0,
                            "current_price": 47000.0,
                            "unrealized_pnl": 4000.0,
                            "unrealized_pnl_percentage": 4.44,
                            "market_value": 94000.0,
                        },
                        {
                            "symbol": "ETHUSDT",
                            "exchange": "binance",
                            "side": "long",
                            "quantity": 15.0,
                            "entry_price": 3000.0,
                            "current_price": 3100.0,
                            "unrealized_pnl": 1500.0,
                            "unrealized_pnl_percentage": 3.33,
                            "market_value": 46500.0,
                        },
                    ],
                    "total_positions": 2,
                    "total_market_value": 140500.0,
                    "total_unrealized_pnl": 5500.0,
                },
            }
            await portfolio_manager.send_to_user(user_id, positions_data)

        if "pnl" in update_types:
            pnl_data = {
                "type": "pnl",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "daily_pnl": random.uniform(-1000, 2000),
                    "weekly_pnl": random.uniform(-3000, 8000),
                    "monthly_pnl": random.uniform(-5000, 15000),
                    "total_pnl": random.uniform(-2000, 25000),
                    "realized_pnl": random.uniform(1000, 20000),
                    "unrealized_pnl": 5500.0,
                    "daily_return_percentage": random.uniform(-2, 5),
                    "total_return_percentage": random.uniform(-5, 25),
                },
            }
            await portfolio_manager.send_to_user(user_id, pnl_data)

        if "balances" in update_types:
            balances_data = {
                "type": "balances",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "balances": [
                        {
                            "exchange": "binance",
                            "currency": "USDT",
                            "total_balance": 50000.0,
                            "available_balance": 45000.0,
                            "locked_balance": 5000.0,
                            "usd_value": 50000.0,
                        },
                        {
                            "exchange": "binance",
                            "currency": "BTC",
                            "total_balance": 2.0,
                            "available_balance": 1.8,
                            "locked_balance": 0.2,
                            "usd_value": 94000.0,
                        },
                        {
                            "exchange": "coinbase",
                            "currency": "USD",
                            "total_balance": 25000.0,
                            "available_balance": 23000.0,
                            "locked_balance": 2000.0,
                            "usd_value": 25000.0,
                        },
                    ],
                    "total_usd_value": 169000.0,
                    "total_available_usd": 158000.0,
                },
            }
            await portfolio_manager.send_to_user(user_id, balances_data)

        if "performance" in update_types:
            performance_data = {
                "type": "performance",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "portfolio_value": 169000.0,
                    "total_exposure": 140500.0,
                    "leverage": 0.83,
                    "win_rate": 0.68,
                    "profit_factor": 2.15,
                    "sharpe_ratio": 1.45,
                    "max_drawdown": -12500.0,
                    "max_drawdown_percentage": -7.4,
                    "volatility": 0.15,
                    "beta": 1.2,
                    "correlation_btc": 0.85,
                },
            }
            await portfolio_manager.send_to_user(user_id, performance_data)

    except Exception as e:
        logger.error(f"Failed to send initial portfolio data: {e}", user_id=user_id)


# Background task to simulate portfolio updates
async def portfolio_simulator():
    """Simulate real-time portfolio updates."""
    import random

    while True:
        try:
            # Generate different types of updates randomly
            update_type = random.choice(
                ["position_update", "pnl_update", "balance_update", "performance_update"]
            )

            if update_type == "position_update":
                # Simulate position changes
                position_update = {
                    "type": "position_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "symbol": random.choice(["BTCUSDT", "ETHUSDT"]),
                        "change_type": random.choice(
                            ["price_update", "quantity_change", "new_position", "closed_position"]
                        ),
                        "new_unrealized_pnl": random.uniform(-500, 1000),
                        "price_change": random.uniform(-2, 3),
                        "total_portfolio_change": random.uniform(-1000, 1500),
                    },
                }
                await portfolio_manager.broadcast_to_all(position_update, "positions")

            elif update_type == "pnl_update":
                # Simulate P&L updates
                pnl_update = {
                    "type": "pnl_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "daily_pnl": random.uniform(-1000, 2000),
                        "realized_pnl_change": random.uniform(-200, 500),
                        "unrealized_pnl_change": random.uniform(-300, 800),
                        "total_return_change": random.uniform(-1, 2),
                    },
                }
                await portfolio_manager.broadcast_to_all(pnl_update, "pnl")

            elif update_type == "balance_update":
                # Simulate balance changes (less frequent)
                if random.random() < 0.3:  # 30% chance
                    balance_update = {
                        "type": "balance_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "exchange": random.choice(["binance", "coinbase"]),
                            "currency": random.choice(["USDT", "USD", "BTC", "ETH"]),
                            "balance_change": random.uniform(-1000, 1000),
                            "reason": random.choice(
                                ["trade_execution", "deposit", "withdrawal", "fee"]
                            ),
                        },
                    }
                    await portfolio_manager.broadcast_to_all(balance_update, "balances")

            elif update_type == "performance_update":
                # Simulate performance metric updates
                if random.random() < 0.2:  # 20% chance
                    performance_update = {
                        "type": "performance_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "portfolio_value": 169000.0 + random.uniform(-5000, 5000),
                            "daily_return": random.uniform(-2, 3),
                            "volatility": 0.15 + random.uniform(-0.02, 0.02),
                            "sharpe_ratio": 1.45 + random.uniform(-0.1, 0.1),
                            "win_rate": 0.68 + random.uniform(-0.05, 0.05),
                        },
                    }
                    await portfolio_manager.broadcast_to_all(performance_update, "performance")

            # Generate alerts occasionally
            if random.random() < 0.05:  # 5% chance
                alert_update = {
                    "type": "portfolio_alert",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "alert_type": random.choice(
                            ["profit_target", "stop_loss", "drawdown_warning", "exposure_limit"]
                        ),
                        "severity": random.choice(["info", "warning", "critical"]),
                        "message": random.choice(
                            [
                                "Portfolio reached daily profit target",
                                "Maximum drawdown threshold exceeded",
                                "High correlation detected in positions",
                                "Low cash balance warning",
                            ]
                        ),
                        "affected_positions": random.choice(
                            [["BTCUSDT"], ["ETHUSDT"], ["BTCUSDT", "ETHUSDT"]]
                        ),
                    },
                }
                await portfolio_manager.broadcast_to_all(alert_update, "alerts")

            # Wait before next update
            await asyncio.sleep(random.uniform(3, 10))  # Variable interval

        except Exception as e:
            logger.error(f"Portfolio simulator error: {e}")
            await asyncio.sleep(5)


@router.get("/portfolio/connections")
async def get_portfolio_connections():
    """Get portfolio WebSocket connection information."""
    return {
        "active_connections": len(portfolio_manager.active_connections),
        "subscription_details": {
            user_id: list(update_types)
            for user_id, update_types in portfolio_manager.subscriptions.items()
        },
        "total_subscriptions": sum(len(subs) for subs in portfolio_manager.subscriptions.values()),
        "available_update_types": ["positions", "pnl", "balances", "performance", "alerts"],
    }
