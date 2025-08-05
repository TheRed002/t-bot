"""
Exchange module initialization.

This module initializes all exchange implementations and registers them
with the exchange factory for dynamic instantiation.

CRITICAL: This integrates with P-001 (core types, exceptions, config),
P-002A (error handling), and P-003 (base exchange interface) components.
"""

# Import base exchange interface
from .base import BaseExchange

# Import exchange factory
from .factory import ExchangeFactory

# Import rate limiter and connection manager
from .rate_limiter import RateLimiter
from .connection_manager import ConnectionManager

# Import exchange types
from .types import (
    ExchangeInfo, Ticker, OrderBook, Trade, OrderStatus,
    ExchangeCapability, ExchangeTradingPair, ExchangeRateLimit
)

# Import Binance exchange implementation (P-004)
try:
    from .binance import BinanceExchange
    BINANCE_AVAILABLE = True
except ImportError as e:
    # TODO: Remove in production
    print(f"Warning: Binance exchange not available: {e}")
    BINANCE_AVAILABLE = False

# Import WebSocket handler (P-004)
try:
    from .binance_websocket import BinanceWebSocketHandler
    BINANCE_WEBSOCKET_AVAILABLE = True
except ImportError as e:
    # TODO: Remove in production
    print(f"Warning: Binance WebSocket handler not available: {e}")
    BINANCE_WEBSOCKET_AVAILABLE = False

# Import order manager (P-004)
try:
    from .binance_orders import BinanceOrderManager
    BINANCE_ORDERS_AVAILABLE = True
except ImportError as e:
    # TODO: Remove in production
    print(f"Warning: Binance order manager not available: {e}")
    BINANCE_ORDERS_AVAILABLE = False


def register_exchanges(factory: ExchangeFactory) -> None:
    """
    Register all available exchanges with the factory.
    
    Args:
        factory: Exchange factory instance
    """
    # Register Binance exchange (P-004)
    if BINANCE_AVAILABLE:
        factory.register_exchange("binance", BinanceExchange)
        print("Registered Binance exchange")
    else:
        print("Warning: Binance exchange not registered - dependencies missing")
    
    # TODO: Register other exchanges as they are implemented
    # - P-005: OKX exchange
    # - P-006: Coinbase exchange
    
    print(f"Registered {len(factory.get_supported_exchanges())} exchanges")


# Export main classes
__all__ = [
    # Base classes
    'BaseExchange',
    'ExchangeFactory',
    'RateLimiter',
    'ConnectionManager',
    
    # Types
    'ExchangeInfo',
    'Ticker',
    'OrderBook',
    'Trade',
    'OrderStatus',
    'ExchangeCapability',
    'ExchangeTradingPair',
    'ExchangeRateLimit',
    
    # Binance implementation (P-004)
    'BinanceExchange',
    'BinanceWebSocketHandler',
    'BinanceOrderManager',
    
    # Utility function
    'register_exchanges',
] 