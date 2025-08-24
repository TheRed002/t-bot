"""
WebSocket handlers for T-Bot Trading System.

This module provides real-time WebSocket communication for:
- Market data streaming (prices, order book, trades)
- Bot status updates (state changes, performance metrics)
- Portfolio updates (positions, P&L, balances)
- Alerts and notifications
- Log streaming

All WebSocket handlers implement proper authentication, error handling,
and connection management for reliable real-time communication.
"""

from .unified_manager import UnifiedWebSocketManager, get_unified_websocket_manager

__all__ = ["UnifiedWebSocketManager", "get_unified_websocket_manager"]
