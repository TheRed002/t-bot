"""Sandbox configuration for the T-Bot trading system."""

from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from .base import BaseConfig


class SandboxEnvironment(str, Enum):
    """Sandbox environment types."""

    PRODUCTION = "production"
    SANDBOX = "sandbox"
    TESTNET = "testnet"
    LOCAL_MOCK = "local_mock"


class SandboxExchangeConfig(BaseConfig):
    """Sandbox-specific exchange configuration."""

    # Environment detection
    environment: SandboxEnvironment = Field(
        default=SandboxEnvironment.SANDBOX,
        description="Current environment mode",
        alias="SANDBOX_ENVIRONMENT"
    )

    # Sandbox URLs for each exchange
    binance_sandbox_url: str = Field(
        default="https://testnet.binance.vision",
        description="Binance sandbox API URL",
        alias="BINANCE_SANDBOX_URL"
    )
    binance_sandbox_ws_url: str = Field(
        default="wss://testnet.binance.vision",
        description="Binance sandbox WebSocket URL",
        alias="BINANCE_SANDBOX_WS_URL"
    )

    coinbase_sandbox_url: str = Field(
        default="https://api-public.sandbox.exchange.coinbase.com",
        description="Coinbase sandbox API URL",
        alias="COINBASE_SANDBOX_URL"
    )
    coinbase_sandbox_ws_url: str = Field(
        default="wss://ws-feed-public.sandbox.exchange.coinbase.com",
        description="Coinbase sandbox WebSocket URL",
        alias="COINBASE_SANDBOX_WS_URL"
    )

    okx_sandbox_url: str = Field(
        default="https://aws.okx.com",
        description="OKX sandbox API URL",
        alias="OKX_SANDBOX_URL"
    )
    okx_sandbox_ws_url: str = Field(
        default="wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999",
        description="OKX sandbox WebSocket URL",
        alias="OKX_SANDBOX_WS_URL"
    )

    # Sandbox-specific settings
    sandbox_rate_limit_multiplier: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Rate limit multiplier for sandbox (0.5 = 50% of production limits)"
    )

    sandbox_timeout_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Timeout multiplier for sandbox (1.5 = 150% of production timeouts)"
    )

    # Mock data settings
    enable_mock_data: bool = Field(
        default=True,
        description="Enable mock data generation for testing"
    )

    mock_balance_btc: str = Field(
        default="1.5",
        description="Mock BTC balance for testing"
    )

    mock_balance_eth: str = Field(
        default="10.0",
        description="Mock ETH balance for testing"
    )

    mock_balance_usdt: str = Field(
        default="10000.0",
        description="Mock USDT balance for testing"
    )

    # Sandbox validation settings
    validate_sandbox_credentials: bool = Field(
        default=True,
        description="Validate sandbox credentials on startup"
    )

    sandbox_health_check_interval: int = Field(
        default=30,
        description="Sandbox health check interval in seconds"
    )

    # Error simulation for testing
    simulate_errors: bool = Field(
        default=False,
        description="Enable error simulation for testing"
    )

    error_simulation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Error simulation rate (0.1 = 10% of requests fail)"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: SandboxEnvironment) -> SandboxEnvironment:
        """Validate sandbox environment."""
        if not isinstance(v, SandboxEnvironment):
            try:
                return SandboxEnvironment(v)
            except ValueError:
                raise ValueError(f"Invalid sandbox environment: {v}")
        return v

    def get_sandbox_credentials(self, exchange: str) -> dict[str, Any]:
        """Get sandbox credentials for a specific exchange."""
        exchange = exchange.lower()

        base_config = {
            "environment": self.environment.value,
            "validate_credentials": self.validate_sandbox_credentials,
            "rate_limit_multiplier": self.sandbox_rate_limit_multiplier,
            "timeout_multiplier": self.sandbox_timeout_multiplier,
        }

        if exchange == "binance":
            return {
                **base_config,
                "api_url": self.binance_sandbox_url,
                "ws_url": self.binance_sandbox_ws_url,
                "testnet": True,
            }
        elif exchange == "coinbase":
            return {
                **base_config,
                "api_url": self.coinbase_sandbox_url,
                "ws_url": self.coinbase_sandbox_ws_url,
                "sandbox": True,
            }
        elif exchange == "okx":
            return {
                **base_config,
                "api_url": self.okx_sandbox_url,
                "ws_url": self.okx_sandbox_ws_url,
                "testnet": True,
            }
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

    def get_mock_balances(self) -> dict[str, str]:
        """Get mock balances for testing."""
        return {
            "BTC": self.mock_balance_btc,
            "ETH": self.mock_balance_eth,
            "USDT": self.mock_balance_usdt,
        }

    def is_sandbox_environment(self) -> bool:
        """Check if running in sandbox environment."""
        return self.environment != SandboxEnvironment.PRODUCTION

    def get_environment_config(self) -> dict[str, Any]:
        """Get environment-specific configuration."""
        return {
            "environment": self.environment.value,
            "mock_data_enabled": self.enable_mock_data,
            "error_simulation": {
                "enabled": self.simulate_errors,
                "rate": self.error_simulation_rate,
            },
            "health_check_interval": self.sandbox_health_check_interval,
        }
