"""Execution configuration for the T-Bot trading system."""

from decimal import Decimal
from typing import Any

from pydantic import Field, field_validator

from .base import BaseConfig


class ExecutionConfig(BaseConfig):
    """Execution-specific configuration."""

    # Order management
    order_timeout_minutes: int = Field(
        default=60, ge=1, le=1440, description="Default order timeout in minutes"
    )
    status_check_interval_seconds: int = Field(
        default=5, ge=1, le=60, description="Order status check interval in seconds"
    )
    max_concurrent_orders: int = Field(
        default=100, ge=1, le=1000, description="Maximum concurrent orders"
    )
    order_history_retention_hours: int = Field(
        default=168, ge=1, le=8760, description="Order history retention in hours (default: 1 week)"
    )

    # Performance and threading
    max_workers: int = Field(
        default=4, ge=1, le=16, description="Maximum worker threads for execution"
    )
    timeout: int = Field(
        default=30, ge=1, le=300, description="Default execution timeout in seconds"
    )

    # Order size limits
    max_order_size: Decimal | None = Field(default=None, description="Maximum order size allowed")
    min_order_size: Decimal | None = Field(default=None, description="Minimum order size allowed")

    # Volume and market impact
    large_order_threshold: str = Field(
        default="10000", description="Threshold for considering an order large (USD value)"
    )
    volume_significance_threshold: float = Field(
        default=0.05, ge=0.01, le=0.5, description="Volume significance threshold for market impact"
    )
    default_daily_volume: str = Field(
        default="1000000", description="Default daily volume for calculations"
    )
    default_portfolio_value: str = Field(
        default="100000", description="Default portfolio value for calculations"
    )
    estimated_market_volume_default: str = Field(
        default="500000", description="Default estimated market volume"
    )

    # Exchange routing
    exchanges: list[str] = Field(
        default_factory=lambda: ["binance", "coinbase", "okx"],
        description="Available exchanges for routing",
    )

    # Routing configuration
    routing_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "cost_optimization": True,
            "latency_optimization": True,
            "liquidity_optimization": True,
            "smart_order_routing": True,
            "max_exchange_splits": 3,
            "min_order_size_for_routing": "100",
        },
        description="Order routing configuration",
    )

    @field_validator("max_order_size", "min_order_size", mode="before")
    @classmethod
    def validate_decimal_fields(cls, v):
        """Convert string values to Decimal."""
        if v is None:
            return v
        if isinstance(v, str):
            return Decimal(v)
        return v

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback to default."""
        return getattr(self, key, default)

    def get_routing_config(self) -> dict[str, Any]:
        """Get routing configuration."""
        return self.routing_config

    def get_order_size_limits(self) -> dict[str, Decimal | None]:
        """Get order size limits."""
        return {"max_order_size": self.max_order_size, "min_order_size": self.min_order_size}

    def get_performance_settings(self) -> dict[str, int]:
        """Get performance-related settings."""
        return {
            "max_workers": self.max_workers,
            "timeout": self.timeout,
            "max_concurrent_orders": self.max_concurrent_orders,
        }
