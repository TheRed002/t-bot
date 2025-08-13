"""
Exchange-specific types and data structures.

This module defines exchange-specific data structures that are used
across different exchange implementations for consistent data handling.

CRITICAL: This integrates with P-001 (core types) and provides
exchange-specific extensions.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
import re
from typing import Any

from pydantic import BaseModel, Field

# MANDATORY: Import from P-001

# Use centralized validators to avoid duplication
from src.utils.validators import (
    validate_symbol as core_validate_symbol,
    validate_price as core_validate_price,
    validate_quantity as core_validate_quantity,
)


class ExchangeTypes:
    """
    Exchange-specific type definitions and utilities.

    This class provides exchange-specific data structures and validation
    methods that are used across different exchange implementations.
    """

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate a trading symbol format.

        Delegates to core validator, then applies exchange-specific stricter
        rule used in integration tests: only uppercase alphanumerics (no
        separators like '-' or '/').
        """
        try:
            normalized = core_validate_symbol(symbol)
            # Exchange-specific stricter rule: alphanumeric only
            return bool(re.match(r"^[A-Z0-9]+$", normalized))
        except Exception:
            return False

    @staticmethod
    def validate_quantity(quantity: Decimal, min_quantity: Decimal = Decimal("0")) -> bool:
        """Validate order quantity using core validator with minimal duplication."""
        if not isinstance(quantity, Decimal):
            return False
        try:
            # Symbol context impacts precision only; boolean validity here
            # depends on positivity and minimum threshold
            normalized = core_validate_quantity(
                float(quantity), "BTCUSDT", float(min_quantity))
            return Decimal(normalized) > min_quantity
        except Exception:
            return False

    @staticmethod
    def validate_price(price: Decimal, min_price: Decimal = Decimal("0")) -> bool:
        """Validate order price using core validator; return boolean outcome."""
        if not isinstance(price, Decimal):
            return False
        try:
            normalized = core_validate_price(float(price), "BTCUSDT")
            return Decimal(normalized) > min_price
        except Exception:
            return False


class ExchangeCapability(Enum):
    """Exchange capabilities enumeration."""

    SPOT_TRADING = "spot_trading"
    FUTURES_TRADING = "futures_trading"
    MARGIN_TRADING = "margin_trading"
    STAKING = "staking"
    LENDING = "lending"
    DERIVATIVES = "derivatives"


class ExchangeTradingPair(BaseModel):
    """Trading pair information."""

    symbol: str
    base_asset: str
    quote_asset: str
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal
    min_price: Decimal
    max_price: Decimal
    tick_size: Decimal
    is_active: bool = True


class ExchangeFee(BaseModel):
    """Exchange fee structure."""

    maker_fee: Decimal = Field(default=Decimal(
        "0.001"), description="Maker fee rate")
    taker_fee: Decimal = Field(default=Decimal(
        "0.001"), description="Taker fee rate")
    min_fee: Decimal = Field(default=Decimal("0"), description="Minimum fee")
    max_fee: Decimal = Field(default=Decimal(
        "0.01"), description="Maximum fee")


class ExchangeRateLimit(BaseModel):
    """Exchange rate limit configuration."""

    requests_per_minute: int
    orders_per_second: int
    websocket_connections: int
    weight_per_request: int = 1


class ExchangeConnectionConfig(BaseModel):
    """Exchange connection configuration."""

    base_url: str
    websocket_url: str
    api_key: str
    api_secret: str
    passphrase: str | None = None
    timeout: int = 30
    max_retries: int = 3
    testnet: bool = True


class ExchangeOrderBookLevel(BaseModel):
    """Order book level information."""

    price: Decimal
    quantity: Decimal
    total_quantity: Decimal = Decimal("0")


class ExchangeOrderBookSnapshot(BaseModel):
    """Order book snapshot."""

    symbol: str
    bids: list[ExchangeOrderBookLevel]
    asks: list[ExchangeOrderBookLevel]
    timestamp: datetime
    sequence_number: int | None = None


class ExchangeTrade(BaseModel):
    """Exchange trade information."""

    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    fee: Decimal = Decimal("0")
    fee_currency: str = "USDT"
    is_maker: bool = False


class ExchangeBalance(BaseModel):
    """Exchange balance information."""

    asset: str
    free_balance: Decimal
    locked_balance: Decimal
    total_balance: Decimal
    usd_value: Decimal | None = None
    last_updated: datetime


class ExchangePosition(BaseModel):
    """Exchange position information."""

    symbol: str
    side: str  # 'long' or 'short'
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    margin_type: str = "isolated"
    leverage: Decimal = Decimal("1")
    liquidation_price: Decimal | None = None


class ExchangeOrder(BaseModel):
    """Exchange order information."""

    id: str
    client_order_id: str | None
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop', etc.
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    status: str
    time_in_force: str = "GTC"
    created_at: datetime
    updated_at: datetime
    fees: Decimal = Decimal("0")


class ExchangeWebSocketMessage(BaseModel):
    """WebSocket message structure."""

    channel: str
    symbol: str | None = None
    data: dict[str, Any]
    timestamp: datetime


class ExchangeErrorResponse(BaseModel):
    """Exchange error response structure."""

    code: int
    message: str
    details: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ExchangeHealthStatus(BaseModel):
    """Exchange health status."""

    exchange_name: str
    status: str  # 'online', 'offline', 'maintenance'
    latency_ms: float | None = None
    last_heartbeat: datetime | None = None
    error_count: int = 0
    success_rate: float = 1.0
