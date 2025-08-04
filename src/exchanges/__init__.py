"""
Exchange integration module for the trading bot framework.

This module provides unified interfaces for multiple cryptocurrency exchanges,
abstracting their differences to enable consistent trading logic across all
supported exchanges.

CRITICAL: This module integrates with P-001 (core types, exceptions, config)
and P-002A (error handling) components.
"""

from .base import BaseExchange
from .factory import ExchangeFactory
from .types import ExchangeTypes
from .rate_limiter import RateLimiter
from .connection_manager import ConnectionManager

__all__ = [
    "BaseExchange",
    "ExchangeFactory", 
    "ExchangeTypes",
    "RateLimiter",
    "ConnectionManager"
] 