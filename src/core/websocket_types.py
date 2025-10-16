"""
WebSocket Types and Enums for T-Bot Trading System.

Provides types and enums for the consolidated WebSocket infrastructure.
"""

from enum import Enum


class StreamType(Enum):
    """WebSocket stream types for different data subscriptions."""

    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    USER_DATA = "user_data"
    CANDLES = "candles"
    DEPTH = "depth"


class MessagePriority(Enum):
    """Message priority levels for WebSocket message handling."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        """Compare priorities (lower value = higher priority)."""
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        """Compare priorities (lower value = higher priority)."""
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other):
        """Compare priorities (lower value = higher priority)."""
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        """Compare priorities (lower value = higher priority)."""
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value >= other.value
