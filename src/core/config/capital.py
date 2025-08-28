"""Capital management configuration for the T-Bot trading system."""

import os
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from ..types import AllocationStrategy


class CapitalManagementConfig(BaseModel):
    """Capital management configuration settings."""

    base_currency: str = Field(
        default="USDT", description="Base currency for all calculations", pattern=r"^[A-Z]{3,10}$"
    )

    total_capital: Decimal = Field(
        default_factory=lambda: Decimal(os.getenv("CAPITAL_TOTAL", "100000.0")),
        description="Total capital available for trading",
        gt=0,
    )

    emergency_reserve_pct: Decimal = Field(
        default_factory=lambda: Decimal(os.getenv("CAPITAL_EMERGENCY_RESERVE_PCT", "0.1")),
        description="Percentage of capital to keep as emergency reserve",
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
    )

    allocation_strategy: AllocationStrategy = Field(
        default=AllocationStrategy.EQUAL_WEIGHT,
        description="Strategy for allocating capital across exchanges/strategies",
    )

    rebalance_frequency_hours: int = Field(
        default_factory=lambda: int(os.getenv("CAPITAL_REBALANCE_HOURS", "24")),
        description="How often to rebalance allocations (in hours)",
        gt=0,
    )

    min_allocation_pct: Decimal = Field(
        default_factory=lambda: Decimal(os.getenv("CAPITAL_MIN_ALLOCATION_PCT", "0.05")),
        description="Minimum allocation percentage per exchange/strategy",
        ge=0.0,
        le=1.0,
    )

    max_allocation_pct: Decimal = Field(
        default_factory=lambda: Decimal(os.getenv("CAPITAL_MAX_ALLOCATION_PCT", "0.5")),
        description="Maximum allocation percentage per exchange/strategy",
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
    )

    max_drawdown_threshold: Decimal = Field(
        default_factory=lambda: Decimal(os.getenv("CAPITAL_MAX_DRAWDOWN", "0.2")),
        description="Maximum drawdown before triggering emergency controls",
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
    )

    funding_threshold: Decimal = Field(
        default_factory=lambda: Decimal(os.getenv("CAPITAL_FUNDING_THRESHOLD", "0.8")),
        description="Threshold for triggering funding operations",
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
    )

    withdrawal_limits: dict[str, float] = Field(
        default_factory=dict, description="Per-exchange withdrawal limits"
    )

    enable_auto_rebalance: bool = Field(
        default_factory=lambda: os.getenv("CAPITAL_AUTO_REBALANCE", "true").lower() == "true",
        description="Enable automatic rebalancing",
    )

    enable_cross_exchange_transfers: bool = Field(
        default_factory=lambda: os.getenv("CAPITAL_CROSS_EXCHANGE_TRANSFERS", "false").lower()
        == "true",
        description="Enable transfers between exchanges",
    )

    # Currency management settings
    supported_currencies: list[str] = Field(
        default=["USDT", "BTC", "ETH", "USD"],
        description="List of supported currencies for trading",
    )

    hedging_enabled: bool = Field(default=True, description="Enable currency hedging")

    hedging_threshold: float = Field(
        default=0.2, description="Threshold for triggering hedging operations", ge=0.0, le=1.0
    )

    hedge_ratio: float = Field(
        default=0.5, description="Ratio of exposure to hedge", ge=0.0, le=1.0
    )

    # Capital protection settings
    max_daily_loss_pct: float = Field(
        default=0.05, description="Maximum daily loss percentage", ge=0.0, le=1.0
    )

    max_weekly_loss_pct: float = Field(
        default=0.15, description="Maximum weekly loss percentage", ge=0.0, le=1.0
    )

    max_monthly_loss_pct: float = Field(
        default=0.25, description="Maximum monthly loss percentage", ge=0.0, le=1.0
    )

    profit_lock_pct: float = Field(
        default=0.1, description="Percentage of profit to lock in", ge=0.0, le=1.0
    )

    auto_compound_enabled: bool = Field(
        default=True, description="Enable automatic profit compounding"
    )

    auto_compound_frequency: str = Field(
        default="daily", description="Frequency of auto-compounding (daily, weekly, monthly)"
    )

    profit_threshold: float = Field(
        default=0.05, description="Minimum profit threshold for operations", ge=0.0
    )

    # Fund flow settings
    min_deposit_amount: float = Field(default=1000.0, description="Minimum deposit amount", gt=0)

    min_withdrawal_amount: float = Field(
        default=100.0, description="Minimum withdrawal amount", gt=0
    )

    max_withdrawal_pct: float = Field(
        default=0.2, description="Maximum withdrawal percentage per transaction", ge=0.0, le=1.0
    )

    max_daily_reallocation_pct: float = Field(
        default=0.1, description="Maximum daily reallocation percentage", ge=0.0, le=1.0
    )

    fund_flow_cooldown_minutes: int = Field(
        default=30, description="Cooldown period for fund flows in minutes", gt=0
    )

    withdrawal_rules: dict[str, Any] = Field(
        default_factory=dict, description="Withdrawal rules configuration"
    )

    per_strategy_minimum: dict[str, float] = Field(
        default_factory=dict, description="Minimum allocation per strategy type"
    )

    def get_available_capital(self) -> Decimal:
        """Get capital available for trading (total - emergency reserve)."""
        reserve_amount = self.total_capital * self.emergency_reserve_pct
        return Decimal(str(self.total_capital - reserve_amount))

    def get_emergency_reserve(self) -> Decimal:
        """Get emergency reserve amount."""
        return Decimal(str(self.total_capital * self.emergency_reserve_pct))

    def get_max_allocation_for_strategy(self) -> Decimal:
        """Get maximum allocation amount for a single strategy."""
        available = self.get_available_capital()
        return available * Decimal(str(self.max_allocation_pct))

    def get_min_allocation_for_strategy(self) -> Decimal:
        """Get minimum allocation amount for a single strategy."""
        available = self.get_available_capital()
        return available * Decimal(str(self.min_allocation_pct))

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "total_capital": self.total_capital,
            "emergency_reserve_pct": self.emergency_reserve_pct,
            "allocation_strategy": self.allocation_strategy.value,
            "rebalance_frequency_hours": self.rebalance_frequency_hours,
            "min_allocation_pct": self.min_allocation_pct,
            "max_allocation_pct": self.max_allocation_pct,
            "max_drawdown_threshold": self.max_drawdown_threshold,
            "funding_threshold": self.funding_threshold,
            "withdrawal_limits": self.withdrawal_limits,
            "enable_auto_rebalance": self.enable_auto_rebalance,
            "enable_cross_exchange_transfers": self.enable_cross_exchange_transfers,
        }
