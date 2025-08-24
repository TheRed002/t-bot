"""Base types and foundational enums for the T-Bot trading system.

This module provides the core type system with comprehensive validation,
immutability where appropriate, serialization support, and rich comparison methods.

Key Features:
- Full input validation on construction
- Immutable types for critical data
- Rich comparison and equality methods
- JSON serialization/deserialization
- Type conversion utilities
- Comprehensive documentation and examples

TODO: Update the following modules to use consolidated types:
- src/exchanges/types.py (consolidate ExchangeCapability)
- src/risk_management/emergency_controls.py (use base ValidationLevel)
- src/state/quality_controller.py (remove duplicate ValidationLevel)
- All modules using duplicate Status enums
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TradingMode(Enum):
    """Trading mode enumeration for different execution environments.

    Defines the operational mode of the trading system, affecting
    order execution, risk management, and data handling.

    Attributes:
        LIVE: Live trading with real money and actual market execution
        PAPER: Paper trading for testing strategies without real money
        BACKTEST: Historical backtesting mode for strategy validation
        SIMULATION: Real-time simulation mode for testing

    Example:
        >>> mode = TradingMode.LIVE
        >>> print(f"Trading in {mode.value} mode")
        Trading in live mode
    """

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"

    def is_real_money(self) -> bool:
        """Check if this mode involves real money."""
        return self == TradingMode.LIVE

    def allows_testing(self) -> bool:
        """Check if this mode allows safe testing."""
        return self in [TradingMode.PAPER, TradingMode.BACKTEST, TradingMode.SIMULATION]

    @classmethod
    def from_string(cls, value: str) -> "TradingMode":
        """Create TradingMode from string with validation.

        Args:
            value: String representation of trading mode

        Returns:
            TradingMode instance

        Raises:
            ValueError: If value is not a valid trading mode
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_modes = [mode.value for mode in cls]
            raise ValueError(f"Invalid trading mode '{value}'. Valid modes: {valid_modes}")


class ExchangeType(Enum):
    """Exchange types for API integration and rate limiting coordination.

    Defines supported cryptocurrency exchanges with their specific
    characteristics and connection requirements.

    Attributes:
        BINANCE: Binance exchange (spot and futures)
        OKX: OKX exchange (formerly OKEx)
        COINBASE: Coinbase Pro exchange
        KRAKEN: Kraken exchange
        BYBIT: Bybit exchange

    Example:
        >>> exchange = ExchangeType.BINANCE
        >>> print(f"Rate limit: {exchange.get_rate_limit()} req/min")
        Rate limit: 1200 req/min
    """

    BINANCE = "binance"
    OKX = "okx"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"

    def get_rate_limit(self) -> int:
        """Get the standard rate limit for this exchange (requests per minute)."""
        limits = {
            ExchangeType.BINANCE: 1200,
            ExchangeType.OKX: 300,
            ExchangeType.COINBASE: 100,
            ExchangeType.KRAKEN: 60,
            ExchangeType.BYBIT: 120,
        }
        return limits.get(self, 100)

    def supports_websocket(self) -> bool:
        """Check if exchange supports WebSocket streaming."""
        return True  # All supported exchanges have WebSocket support

    def get_base_url(self) -> str:
        """Get the base API URL for this exchange."""
        urls = {
            ExchangeType.BINANCE: "https://api.binance.com",
            ExchangeType.OKX: "https://www.okx.com",
            ExchangeType.COINBASE: "https://api.exchange.coinbase.com",
            ExchangeType.KRAKEN: "https://api.kraken.com",
            ExchangeType.BYBIT: "https://api.bybit.com",
        }
        return urls.get(self, "")


class MarketType(Enum):
    """Market types for different trading venues and instruments.

    Defines the type of financial instrument being traded, affecting
    order execution, margin requirements, and settlement.

    Attributes:
        SPOT: Spot trading (immediate settlement)
        FUTURES: Futures contracts (future settlement)
        OPTIONS: Options contracts
        PERPETUAL: Perpetual swap contracts
        MARGIN: Margin trading
    """

    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    PERPETUAL = "perpetual"
    MARGIN = "margin"

    def requires_margin(self) -> bool:
        """Check if this market type requires margin."""
        return self in [MarketType.FUTURES, MarketType.PERPETUAL, MarketType.MARGIN]

    def has_expiration(self) -> bool:
        """Check if instruments in this market have expiration dates."""
        return self in [MarketType.FUTURES, MarketType.OPTIONS]

    def supports_leverage(self) -> bool:
        """Check if this market type supports leverage."""
        return self in [MarketType.FUTURES, MarketType.PERPETUAL, MarketType.MARGIN]


class RequestType(Enum):
    """Request types for API coordination and rate limiting.

    Categorizes different types of API requests for proper rate limiting,
    priority handling, and resource allocation.

    Attributes:
        MARKET_DATA: Public market data requests
        ORDER_PLACEMENT: Order placement requests
        ORDER_CANCELLATION: Order cancellation requests
        ORDER_MODIFICATION: Order modification requests
        BALANCE_QUERY: Account balance queries
        POSITION_QUERY: Position information queries
        HISTORICAL_DATA: Historical data requests
        WEBSOCKET_CONNECTION: WebSocket connection establishment
        ACCOUNT_INFO: Account information requests
        TRADE_HISTORY: Trade history requests
    """

    MARKET_DATA = "market_data"
    ORDER_PLACEMENT = "order_placement"
    ORDER_CANCELLATION = "order_cancellation"
    ORDER_MODIFICATION = "order_modification"
    BALANCE_QUERY = "balance_query"
    POSITION_QUERY = "position_query"
    HISTORICAL_DATA = "historical_data"
    WEBSOCKET_CONNECTION = "websocket_connection"
    ACCOUNT_INFO = "account_info"
    TRADE_HISTORY = "trade_history"

    def get_priority(self) -> int:
        """Get the priority level for this request type (lower = higher priority)."""
        priorities = {
            RequestType.ORDER_CANCELLATION: 1,  # Highest priority
            RequestType.ORDER_PLACEMENT: 2,
            RequestType.ORDER_MODIFICATION: 3,
            RequestType.POSITION_QUERY: 4,
            RequestType.BALANCE_QUERY: 5,
            RequestType.MARKET_DATA: 6,
            RequestType.ACCOUNT_INFO: 7,
            RequestType.TRADE_HISTORY: 8,
            RequestType.HISTORICAL_DATA: 9,
            RequestType.WEBSOCKET_CONNECTION: 10,  # Lowest priority
        }
        return priorities.get(self, 5)

    def is_modifying_operation(self) -> bool:
        """Check if this request type modifies account state."""
        return self in [
            RequestType.ORDER_PLACEMENT,
            RequestType.ORDER_CANCELLATION,
            RequestType.ORDER_MODIFICATION,
        ]


class ConnectionType(Enum):
    """WebSocket connection types for different data streams.

    Defines the type of real-time data stream being subscribed to,
    affecting subscription management and data parsing.

    Attributes:
        TICKER: Price ticker updates
        ORDERBOOK: Order book depth updates
        TRADES: Recent trades stream
        USER_DATA: User-specific data (orders, balances)
        MARKET_DATA: General market data
        KLINES: Candlestick/OHLCV data
        LIQUIDATIONS: Liquidation events
        FUNDING_RATES: Funding rate updates (for perpetuals)
    """

    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    USER_DATA = "user_data"
    MARKET_DATA = "market_data"
    KLINES = "klines"
    LIQUIDATIONS = "liquidations"
    FUNDING_RATES = "funding_rates"

    def is_public_stream(self) -> bool:
        """Check if this is a public data stream."""
        return self in [
            ConnectionType.TICKER,
            ConnectionType.ORDERBOOK,
            ConnectionType.TRADES,
            ConnectionType.MARKET_DATA,
            ConnectionType.KLINES,
            ConnectionType.LIQUIDATIONS,
            ConnectionType.FUNDING_RATES,
        ]

    def requires_authentication(self) -> bool:
        """Check if this stream requires authentication."""
        return not self.is_public_stream()

    def get_update_frequency(self) -> str:
        """Get typical update frequency for this stream type."""
        frequencies = {
            ConnectionType.TICKER: "1s",
            ConnectionType.ORDERBOOK: "100ms",
            ConnectionType.TRADES: "real-time",
            ConnectionType.USER_DATA: "real-time",
            ConnectionType.MARKET_DATA: "1s",
            ConnectionType.KLINES: "1s",
            ConnectionType.LIQUIDATIONS: "real-time",
            ConnectionType.FUNDING_RATES: "8h",
        }
        return frequencies.get(self, "unknown")


class ValidationLevel(Enum):
    """Data validation severity levels used across the system.

    Provides consistent validation severity classification for
    data quality, pipeline validation, and error handling.

    Attributes:
        CRITICAL: Critical validation failure (system halt)
        HIGH: High severity (significant impact)
        MEDIUM: Medium severity (monitoring required)
        LOW: Low severity (informational)
        INFO: Informational only

    Example:
        >>> level = ValidationLevel.CRITICAL
        >>> if level.should_halt_system():
        ...     print("System halt required")
        System halt required
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def should_halt_system(self) -> bool:
        """Check if this validation level should halt system operation."""
        return self == ValidationLevel.CRITICAL

    def requires_immediate_attention(self) -> bool:
        """Check if this level requires immediate attention."""
        return self in [ValidationLevel.CRITICAL, ValidationLevel.HIGH]

    def get_numeric_value(self) -> int:
        """Get numeric representation for comparison and sorting."""
        values = {
            ValidationLevel.INFO: 1,
            ValidationLevel.LOW: 2,
            ValidationLevel.MEDIUM: 3,
            ValidationLevel.HIGH: 4,
            ValidationLevel.CRITICAL: 5,
        }
        return values.get(self, 0)

    def __lt__(self, other: "ValidationLevel") -> bool:
        """Enable comparison of validation levels."""
        return self.get_numeric_value() < other.get_numeric_value()


class ValidationResult(Enum):
    """Data validation result enumeration with enhanced functionality.

    Provides standardized validation outcomes with additional context
    and utility methods for result processing.

    Attributes:
        PASS: Validation passed successfully
        FAIL: Validation failed
        WARNING: Validation passed with warnings
        SKIP: Validation was skipped
        ERROR: Validation could not be completed due to error
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"

    def is_success(self) -> bool:
        """Check if validation was successful."""
        return self in [ValidationResult.PASS, ValidationResult.WARNING]

    def is_failure(self) -> bool:
        """Check if validation failed."""
        return self in [ValidationResult.FAIL, ValidationResult.ERROR]

    def should_proceed(self) -> bool:
        """Check if operation should proceed after this validation result."""
        return self in [ValidationResult.PASS, ValidationResult.WARNING, ValidationResult.SKIP]

    def get_severity(self) -> ValidationLevel:
        """Get the severity level associated with this result."""
        severities = {
            ValidationResult.PASS: ValidationLevel.INFO,
            ValidationResult.WARNING: ValidationLevel.LOW,
            ValidationResult.SKIP: ValidationLevel.MEDIUM,
            ValidationResult.FAIL: ValidationLevel.HIGH,
            ValidationResult.ERROR: ValidationLevel.CRITICAL,
        }
        return severities.get(self, ValidationLevel.MEDIUM)


# =============================================================================
# BASE MODEL CLASSES
# =============================================================================


class BaseValidatedModel(BaseModel):
    """Enhanced base model with comprehensive validation and utilities.

    Provides common functionality for all domain models including:
    - Automatic timestamp generation
    - Enhanced serialization
    - Validation utilities
    - Equality and comparison methods
    """

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_updated(self) -> None:
        """Mark the model as updated with current timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary with proper serialization."""
        return json.loads(self.model_dump_json())

    def to_json(self) -> str:
        """Convert model to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseValidatedModel":
        """Create model instance from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseValidatedModel":
        """Create model instance from JSON string."""
        return cls.model_validate_json(json_str)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the model."""
        self.metadata[key] = value
        self.mark_updated()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)

    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self.metadata

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
