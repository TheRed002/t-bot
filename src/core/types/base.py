"""Base types and common enums for the T-Bot trading system."""

from enum import Enum
from pydantic import BaseModel


class TradingMode(Enum):
    """Trading mode enumeration for different execution environments.

    Attributes:
        LIVE: Live trading with real money
        PAPER: Paper trading for testing
        BACKTEST: Historical backtesting mode
    """

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class ExchangeType(Enum):
    """Exchange types for rate limiting and coordination."""

    BINANCE = "binance"
    OKX = "okx"
    COINBASE = "coinbase"


class RequestType(Enum):
    """Request types for global coordination and rate limiting."""

    MARKET_DATA = "market_data"
    ORDER_PLACEMENT = "order_placement"
    ORDER_CANCELLATION = "order_cancellation"
    BALANCE_QUERY = "balance_query"
    POSITION_QUERY = "position_query"
    HISTORICAL_DATA = "historical_data"
    WEBSOCKET_CONNECTION = "websocket_connection"


class ConnectionType(Enum):
    """Connection types for different stream types."""

    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    USER_DATA = "user_data"
    MARKET_DATA = "market_data"


class ValidationLevel(Enum):
    """Data validation severity levels used across quality and pipeline validation"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationResult(Enum):
    """Data validation result enumeration"""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"