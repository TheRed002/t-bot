"""
Capital Management Specific Type Definitions

This module provides type definitions specifically tailored for the capital management
module implementation, extending the base types with additional fields needed by
the capital management components.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class CapitalFundFlow(BaseModel):
    """Extended fund flow for capital management operations."""

    from_strategy: str | None = None
    to_strategy: str | None = None
    from_exchange: str | None = None
    to_exchange: str | None = None
    amount: Decimal
    currency: str = "USDT"
    converted_amount: Decimal | None = None
    exchange_rate: Decimal | None = None
    reason: str  # "deposit", "withdrawal", "reallocation", "auto_compound", "currency_conversion"
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    fees: Decimal | None = None
    fee_amount: Decimal | None = None


class CapitalCurrencyExposure(BaseModel):
    """Extended currency exposure for capital management."""

    currency: str
    total_exposure: Decimal  # Total amount in this currency
    base_currency_equivalent: Decimal  # Equivalent in base currency
    exposure_percentage: Decimal  # Percentage of total portfolio
    hedging_required: bool = False
    hedge_amount: Decimal = Decimal("0")
    timestamp: datetime


class CapitalExchangeAllocation(BaseModel):
    """Extended exchange allocation for capital management."""

    exchange: str
    allocated_amount: Decimal
    utilized_amount: Decimal = Decimal("0")
    available_amount: Decimal
    utilization_rate: Decimal = Decimal("0.0")
    liquidity_score: Decimal = Decimal("0.5")
    fee_efficiency: Decimal = Decimal("0.5")
    reliability_score: Decimal = Decimal("0.5")
    last_rebalance: datetime


class ExtendedCapitalProtection(BaseModel):
    """Extended capital protection with additional fields."""

    protection_id: str
    enabled: bool = True
    min_capital_threshold: Decimal
    stop_trading_threshold: Decimal
    reduce_size_threshold: Decimal
    size_reduction_factor: Decimal
    max_daily_loss: Decimal
    max_weekly_loss: Decimal
    max_monthly_loss: Decimal
    emergency_threshold: Decimal

    # Additional fields used by fund flow manager
    emergency_reserve_pct: Decimal = Decimal("0.1")
    max_daily_loss_pct: Decimal = Decimal("0.05")
    max_weekly_loss_pct: Decimal = Decimal("0.10")
    max_monthly_loss_pct: Decimal = Decimal("0.20")
    profit_lock_pct: Decimal = Decimal("0.5")
    auto_compound_enabled: bool = True


class ExtendedWithdrawalRule(BaseModel):
    """Extended withdrawal rule for fund flow manager."""

    name: str
    description: str = ""
    enabled: bool = True
    threshold: Decimal | None = None
    min_amount: Decimal | None = None
    max_percentage: Decimal | None = None
    cooldown_hours: int | None = None
