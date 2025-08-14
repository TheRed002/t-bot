"""
Exchange module initialization.

This module initializes all exchange implementations and registers them
with the exchange factory for dynamic instantiation.

CRITICAL: This integrates with P-001 (core types, exceptions, config),
P-002A (error handling), and P-003 (base exchange interface) components.
"""

from src.core.logging import get_logger

logger = get_logger(__name__)

# Import base exchange interface
# Import exchange-related types
from src.core.types import (
    ExchangeInfo,
    OrderBook,
    OrderStatus,
    Ticker,
    Trade,
)

from .base import BaseExchange
from .connection_manager import ConnectionManager

# Import exchange factory
from .factory import ExchangeFactory

# Import rate limiter and connection manager
from .rate_limiter import RateLimiter
from .types import (
    ExchangeCapability,
    ExchangeRateLimit,
    ExchangeTradingPair,
)

# Import Binance exchange implementation (P-004)
try:
    from .binance import BinanceExchange

    BINANCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Binance exchange not available: {e}")
    BINANCE_AVAILABLE = False

# Import WebSocket handler (P-004)
try:
    from .binance_websocket import BinanceWebSocketHandler

    BINANCE_WEBSOCKET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Binance WebSocket handler not available: {e}")
    BINANCE_WEBSOCKET_AVAILABLE = False

# Import order manager (P-004)
try:
    from .binance_orders import BinanceOrderManager

    BINANCE_ORDERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Binance order manager not available: {e}")
    BINANCE_ORDERS_AVAILABLE = False

# Import OKX exchange implementation (P-005)
try:
    from .okx import OKXExchange

    OKX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OKX exchange not available: {e}")
    OKX_AVAILABLE = False

# Import OKX WebSocket handler (P-005)
try:
    from .okx_websocket import OKXWebSocketManager

    OKX_WEBSOCKET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OKX WebSocket handler not available: {e}")
    OKX_WEBSOCKET_AVAILABLE = False

# Import OKX order manager (P-005)
try:
    from .okx_orders import OKXOrderManager

    OKX_ORDERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OKX order manager not available: {e}")
    OKX_ORDERS_AVAILABLE = False

# Import Coinbase exchange implementation (P-006)
try:
    from .coinbase import CoinbaseExchange

    COINBASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Coinbase exchange not available: {e}")
    COINBASE_AVAILABLE = False

# Import Coinbase WebSocket handler (P-006)
try:
    from .coinbase_websocket import CoinbaseWebSocketHandler

    COINBASE_WEBSOCKET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Coinbase WebSocket handler not available: {e}")
    COINBASE_WEBSOCKET_AVAILABLE = False

# Import Coinbase order manager (P-006)
try:
    from .coinbase_orders import CoinbaseOrderManager

    COINBASE_ORDERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Coinbase order manager not available: {e}")
    COINBASE_ORDERS_AVAILABLE = False


def register_exchanges(factory: ExchangeFactory) -> None:
    """
    Register all available exchanges with the factory.

    Args:
        factory: Exchange factory instance
    """
    # Register Binance exchange (P-004)
    if BINANCE_AVAILABLE:
        factory.register_exchange("binance", BinanceExchange)
        logger.info("Registered Binance exchange")
    else:
        logger.warning("Binance exchange not registered - dependencies missing")

    # Register OKX exchange (P-005)
    if OKX_AVAILABLE:
        factory.register_exchange("okx", OKXExchange)
        logger.info("Registered OKX exchange")
    else:
        logger.warning("OKX exchange not registered - dependencies missing")

    # Register Coinbase exchange (P-006)
    if COINBASE_AVAILABLE:
        factory.register_exchange("coinbase", CoinbaseExchange)
        logger.info("Registered Coinbase exchange")
    else:
        logger.warning("Coinbase exchange not registered - dependencies missing")

    # TODO: Register other exchanges as they are implemented

    logger.info(f"Registered {len(factory.get_supported_exchanges())} exchanges")


# Export main classes
__all__ = [
    # Base classes
    "BaseExchange",
    "ExchangeFactory",
    "RateLimiter",
    "ConnectionManager",
    # Types
    "ExchangeInfo",
    "Ticker",
    "OrderBook",
    "Trade",
    "OrderStatus",
    "ExchangeCapability",
    "ExchangeTradingPair",
    "ExchangeRateLimit",
    # Binance implementation (P-004)
    "BinanceExchange",
    "BinanceWebSocketHandler",
    "BinanceOrderManager",
    # OKX implementation (P-005)
    "OKXExchange",
    "OKXWebSocketManager",
    "OKXOrderManager",
    # Coinbase implementation (P-006)
    "CoinbaseExchange",
    "CoinbaseWebSocketHandler",
    "CoinbaseOrderManager",
    # Utility function
    "register_exchanges",
]
