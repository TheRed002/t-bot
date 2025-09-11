"""Environment configuration for trading networks (sandbox/live)."""

from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from .base import BaseConfig


class TradingEnvironment(Enum):
    """Trading environment types."""

    SANDBOX = "sandbox"
    LIVE = "live"
    MOCK = "mock"
    HYBRID = "hybrid"  # Mix of sandbox/live for different exchanges


class ExchangeEnvironment(Enum):
    """Exchange-specific environment types."""

    SANDBOX = "sandbox"
    TESTNET = "testnet"
    LIVE = "live"
    PRODUCTION = "production"


class EnvironmentConfig(BaseConfig):
    """Configuration for trading environment switching."""

    # Global environment settings
    global_environment: TradingEnvironment = Field(
        default=TradingEnvironment.SANDBOX,
        description="Global trading environment",
        alias="GLOBAL_ENVIRONMENT"
    )

    # Environment confirmation required for production
    production_confirmation: bool = Field(
        default=True,
        description="Require confirmation for production trading",
        alias="PRODUCTION_CONFIRMATION"
    )

    # Safety checks
    enable_production_safeguards: bool = Field(
        default=True,
        description="Enable additional safeguards for production trading",
        alias="ENABLE_PRODUCTION_SAFEGUARDS"
    )

    # Per-exchange environment overrides
    binance_environment: ExchangeEnvironment | None = Field(
        default=None,
        description="Binance environment override",
        alias="BINANCE_ENVIRONMENT"
    )

    coinbase_environment: ExchangeEnvironment | None = Field(
        default=None,
        description="Coinbase environment override",
        alias="COINBASE_ENVIRONMENT"
    )

    okx_environment: ExchangeEnvironment | None = Field(
        default=None,
        description="OKX environment override",
        alias="OKX_ENVIRONMENT"
    )

    # Production endpoint configurations
    binance_live_api_url: str = Field(
        default="https://api.binance.com",
        description="Binance production API URL",
        alias="BINANCE_LIVE_API_URL"
    )

    binance_live_ws_url: str = Field(
        default="wss://stream.binance.com:9443",
        description="Binance production WebSocket URL",
        alias="BINANCE_LIVE_WS_URL"
    )

    coinbase_live_api_url: str = Field(
        default="https://api.exchange.coinbase.com",
        description="Coinbase production API URL",
        alias="COINBASE_LIVE_API_URL"
    )

    coinbase_live_ws_url: str = Field(
        default="wss://ws-feed.exchange.coinbase.com",
        description="Coinbase production WebSocket URL",
        alias="COINBASE_LIVE_WS_URL"
    )

    okx_live_api_url: str = Field(
        default="https://www.okx.com/api/v5",
        description="OKX production API URL",
        alias="OKX_LIVE_API_URL"
    )

    okx_live_ws_url: str = Field(
        default="wss://ws.okx.com:8443/ws/v5/public",
        description="OKX production WebSocket URL",
        alias="OKX_LIVE_WS_URL"
    )

    # Live credential configurations
    binance_live_api_key: str = Field(
        default="",
        description="Binance production API key",
        alias="BINANCE_LIVE_API_KEY"
    )

    binance_live_api_secret: str = Field(
        default="",
        description="Binance production API secret",
        alias="BINANCE_LIVE_API_SECRET"
    )

    coinbase_live_api_key: str = Field(
        default="",
        description="Coinbase production API key",
        alias="COINBASE_LIVE_API_KEY"
    )

    coinbase_live_api_secret: str = Field(
        default="",
        description="Coinbase production API secret",
        alias="COINBASE_LIVE_API_SECRET"
    )

    coinbase_live_passphrase: str = Field(
        default="",
        description="Coinbase production API passphrase",
        alias="COINBASE_LIVE_PASSPHRASE"
    )

    okx_live_api_key: str = Field(
        default="",
        description="OKX production API key",
        alias="OKX_LIVE_API_KEY"
    )

    okx_live_api_secret: str = Field(
        default="",
        description="OKX production API secret",
        alias="OKX_LIVE_API_SECRET"
    )

    okx_live_passphrase: str = Field(
        default="",
        description="OKX production API passphrase",
        alias="OKX_LIVE_PASSPHRASE"
    )

    # Environment validation settings
    require_credentials_validation: bool = Field(
        default=True,
        description="Require credentials validation before switching environments",
        alias="REQUIRE_CREDENTIALS_VALIDATION"
    )

    max_environment_switches_per_hour: int = Field(
        default=5,
        description="Maximum environment switches allowed per hour",
        alias="MAX_ENVIRONMENT_SWITCHES_PER_HOUR"
    )

    @field_validator("global_environment", mode="before")
    @classmethod
    def validate_global_environment(cls, v):
        """Validate global environment value."""
        if isinstance(v, str):
            try:
                return TradingEnvironment(v.lower())
            except ValueError:
                raise ValueError(f"Invalid global environment: {v}")
        return v

    @field_validator("binance_environment", "coinbase_environment", "okx_environment", mode="before")
    @classmethod
    def validate_exchange_environment(cls, v):
        """Validate exchange environment value."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return ExchangeEnvironment(v.lower())
            except ValueError:
                raise ValueError(f"Invalid exchange environment: {v}")
        return v

    def get_exchange_environment(self, exchange_name: str) -> ExchangeEnvironment:
        """Get effective environment for a specific exchange."""
        exchange_name = exchange_name.lower()

        # Check for exchange-specific override
        if exchange_name == "binance" and self.binance_environment:
            return self.binance_environment
        elif exchange_name == "coinbase" and self.coinbase_environment:
            return self.coinbase_environment
        elif exchange_name == "okx" and self.okx_environment:
            return self.okx_environment

        # Map global environment to exchange environment
        if self.global_environment == TradingEnvironment.SANDBOX:
            return ExchangeEnvironment.SANDBOX
        elif self.global_environment == TradingEnvironment.LIVE:
            return ExchangeEnvironment.LIVE
        else:
            # Default to sandbox for mock/hybrid
            return ExchangeEnvironment.SANDBOX

    def get_exchange_endpoints(self, exchange_name: str) -> dict[str, str]:
        """Get API and WebSocket endpoints for an exchange based on environment."""
        exchange_name = exchange_name.lower()
        env = self.get_exchange_environment(exchange_name)

        if exchange_name == "binance":
            if env in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION):
                return {
                    "api_url": self.binance_live_api_url,
                    "ws_url": self.binance_live_ws_url,
                    "environment": "production"
                }
            else:
                return {
                    "api_url": "https://testnet.binance.vision/api",
                    "ws_url": "wss://testnet.binance.vision/ws",
                    "environment": "sandbox"
                }

        elif exchange_name == "coinbase":
            if env in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION):
                return {
                    "api_url": self.coinbase_live_api_url,
                    "ws_url": self.coinbase_live_ws_url,
                    "environment": "production"
                }
            else:
                return {
                    "api_url": "https://api-public.sandbox.exchange.coinbase.com",
                    "ws_url": "wss://ws-feed-public.sandbox.exchange.coinbase.com",
                    "environment": "sandbox"
                }

        elif exchange_name == "okx":
            if env in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION):
                return {
                    "api_url": self.okx_live_api_url,
                    "ws_url": self.okx_live_ws_url,
                    "environment": "production"
                }
            else:
                return {
                    "api_url": "https://www.okx.com/api/v5",  # OKX uses same URL with testnet flag
                    "ws_url": "wss://wspap.okx.com:8443/ws/v5/public",
                    "environment": "sandbox"
                }

        else:
            raise ValueError(f"Unknown exchange: {exchange_name}")

    def get_exchange_credentials(self, exchange_name: str) -> dict[str, Any]:
        """Get credentials for an exchange based on environment."""
        exchange_name = exchange_name.lower()
        env = self.get_exchange_environment(exchange_name)

        is_production = env in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION)

        if exchange_name == "binance":
            if is_production:
                return {
                    "api_key": self.binance_live_api_key,
                    "api_secret": self.binance_live_api_secret,
                    "testnet": False,
                }
            else:
                # Return sandbox credentials - these would come from sandbox config
                return {
                    "api_key": "",  # Will be filled by sandbox config
                    "api_secret": "",  # Will be filled by sandbox config
                    "testnet": True,
                }

        elif exchange_name == "coinbase":
            if is_production:
                return {
                    "api_key": self.coinbase_live_api_key,
                    "api_secret": self.coinbase_live_api_secret,
                    "passphrase": self.coinbase_live_passphrase,
                    "sandbox": False,
                }
            else:
                return {
                    "api_key": "",  # Will be filled by sandbox config
                    "api_secret": "",  # Will be filled by sandbox config
                    "passphrase": "",  # Will be filled by sandbox config
                    "sandbox": True,
                }

        elif exchange_name == "okx":
            if is_production:
                return {
                    "api_key": self.okx_live_api_key,
                    "api_secret": self.okx_live_api_secret,
                    "passphrase": self.okx_live_passphrase,
                    "testnet": False,
                }
            else:
                return {
                    "api_key": "",  # Will be filled by sandbox config
                    "api_secret": "",  # Will be filled by sandbox config
                    "passphrase": "",  # Will be filled by sandbox config
                    "testnet": True,
                }

        else:
            raise ValueError(f"Unknown exchange: {exchange_name}")

    def is_production_environment(self, exchange_name: str) -> bool:
        """Check if an exchange is configured for production trading."""
        env = self.get_exchange_environment(exchange_name)
        return env in (ExchangeEnvironment.LIVE, ExchangeEnvironment.PRODUCTION)

    def validate_production_credentials(self, exchange_name: str) -> bool:
        """Validate that production credentials are configured for an exchange."""
        if not self.is_production_environment(exchange_name):
            return True  # Sandbox doesn't require production credentials

        creds = self.get_exchange_credentials(exchange_name)

        # Check required fields based on exchange
        if exchange_name.lower() == "binance":
            return bool(creds.get("api_key") and creds.get("api_secret"))
        elif exchange_name.lower() == "coinbase":
            return bool(
                creds.get("api_key") and
                creds.get("api_secret") and
                creds.get("passphrase")
            )
        elif exchange_name.lower() == "okx":
            return bool(
                creds.get("api_key") and
                creds.get("api_secret") and
                creds.get("passphrase")
            )

        return False

    def get_environment_summary(self) -> dict[str, Any]:
        """Get a summary of current environment configuration."""
        return {
            "global_environment": self.global_environment.value,
            "production_safeguards_enabled": self.enable_production_safeguards,
            "exchanges": {
                "binance": {
                    "environment": self.get_exchange_environment("binance").value,
                    "is_production": self.is_production_environment("binance"),
                    "credentials_configured": self.validate_production_credentials("binance"),
                },
                "coinbase": {
                    "environment": self.get_exchange_environment("coinbase").value,
                    "is_production": self.is_production_environment("coinbase"),
                    "credentials_configured": self.validate_production_credentials("coinbase"),
                },
                "okx": {
                    "environment": self.get_exchange_environment("okx").value,
                    "is_production": self.is_production_environment("okx"),
                    "credentials_configured": self.validate_production_credentials("okx"),
                },
            },
        }
