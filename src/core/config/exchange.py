"""Exchange configuration for the T-Bot trading system."""

from typing import Any

from pydantic import Field, field_validator

from .base import BaseConfig


class ExchangeConfig(BaseConfig):
    """Exchange-specific configuration."""

    # Binance configuration
    binance_api_key: str = Field(default="", description="Binance API key", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(
        default="", description="Binance API secret", alias="BINANCE_API_SECRET"
    )
    binance_testnet: bool = Field(
        default=True, description="Use Binance testnet", alias="BINANCE_TESTNET"
    )
    binance_base_url: str = Field(
        default="https://api.binance.com",
        description="Binance API base URL",
        alias="BINANCE_BASE_URL",
    )
    binance_ws_url: str = Field(
        default="wss://stream.binance.com:9443",
        description="Binance WebSocket URL",
        alias="BINANCE_WS_URL",
    )

    # Coinbase configuration
    coinbase_api_key: str = Field(
        default="", description="Coinbase API key", alias="COINBASE_API_KEY"
    )
    coinbase_api_secret: str = Field(
        default="", description="Coinbase API secret", alias="COINBASE_API_SECRET"
    )
    coinbase_passphrase: str = Field(
        default="", description="Coinbase API passphrase", alias="COINBASE_PASSPHRASE"
    )
    coinbase_sandbox: bool = Field(
        default=True, description="Use Coinbase sandbox", alias="COINBASE_SANDBOX"
    )
    coinbase_base_url: str = Field(
        default="https://api.exchange.coinbase.com",
        description="Coinbase API base URL",
        alias="COINBASE_BASE_URL",
    )
    coinbase_ws_url: str = Field(
        default="wss://ws-feed.exchange.coinbase.com",
        description="Coinbase WebSocket URL",
        alias="COINBASE_WS_URL",
    )

    # OKX configuration
    okx_api_key: str = Field(default="", description="OKX API key", alias="OKX_API_KEY")
    okx_api_secret: str = Field(default="", description="OKX API secret", alias="OKX_API_SECRET")
    okx_passphrase: str = Field(
        default="", description="OKX API passphrase", alias="OKX_PASSPHRASE"
    )
    okx_testnet: bool = Field(default=True, description="Use OKX testnet", alias="OKX_TESTNET")
    okx_base_url: str = Field(
        default="https://www.okx.com", description="OKX API base URL", alias="OKX_BASE_URL"
    )
    okx_ws_url: str = Field(
        default="wss://ws.okx.com:8443", description="OKX WebSocket URL", alias="OKX_WS_URL"
    )

    # Common exchange settings
    default_exchange: str = Field(
        default="binance", description="Default exchange to use", alias="DEFAULT_EXCHANGE"
    )
    enabled_exchanges: list[str] = Field(
        default_factory=lambda: ["binance"], description="List of enabled exchanges"
    )
    testnet_mode: bool = Field(default=False, description="Enable testnet mode for all exchanges")
    rate_limit_per_second: int = Field(default=10, description="Rate limit per second")

    # Rate limiting
    rate_limit_buffer: float = Field(
        default=0.9, ge=0.1, le=1.0, description="Rate limit buffer (0.9 = use 90% of limit)"
    )

    # Connection settings
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")

    @field_validator("default_exchange")
    @classmethod
    def validate_default_exchange(cls, v: str) -> str:
        """Validate default exchange is supported."""
        valid_exchanges = ["binance", "coinbase", "okx"]
        if v.lower() not in valid_exchanges:
            raise ValueError(f"Exchange must be one of {valid_exchanges}, got {v}")
        return v.lower()

    @field_validator("enabled_exchanges")
    @classmethod
    def validate_enabled_exchanges(cls, v: list[str]) -> list[str]:
        """Validate all enabled exchanges are supported."""
        valid_exchanges = ["binance", "coinbase", "okx"]
        for exchange in v:
            if exchange.lower() not in valid_exchanges:
                raise ValueError(f"Exchange {exchange} not supported")
        return [ex.lower() for ex in v]

    def get_exchange_credentials(self, exchange: str) -> dict[str, Any]:
        """Get credentials for a specific exchange."""
        exchange = exchange.lower()

        if exchange == "binance":
            return {
                "api_key": self.binance_api_key,
                "api_secret": self.binance_api_secret,
                "testnet": self.testnet_mode,
                "base_url": self.binance_base_url,
                "ws_url": self.binance_ws_url,
            }
        elif exchange == "coinbase":
            return {
                "api_key": self.coinbase_api_key,
                "api_secret": self.coinbase_api_secret,
                "passphrase": self.coinbase_passphrase,
                "sandbox": self.coinbase_sandbox,
                "base_url": self.coinbase_base_url,
                "ws_url": self.coinbase_ws_url,
            }
        elif exchange == "okx":
            return {
                "api_key": self.okx_api_key,
                "api_secret": self.okx_api_secret,
                "passphrase": self.okx_passphrase,
                "testnet": self.okx_testnet,
                "base_url": self.okx_base_url,
                "ws_url": self.okx_ws_url,
            }
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

    def is_exchange_configured(self, exchange: str) -> bool:
        """Check if an exchange has valid credentials."""
        try:
            creds = self.get_exchange_credentials(exchange)
            return bool(creds.get("api_key") and creds.get("api_secret"))
        except ValueError:
            return False

    def get_websocket_config(self, exchange: str) -> dict:
        """Get WebSocket configuration for a specific exchange."""
        base_config = {
            "reconnect_attempts": 5,
            "ping_interval": 30,
            "timeout": self.connection_timeout,
        }

        exchange = exchange.lower()
        if exchange == "binance":
            base_config["url"] = self.binance_ws_url
        elif exchange == "coinbase":
            base_config["url"] = self.coinbase_ws_url
        elif exchange == "okx":
            base_config["url"] = self.okx_ws_url
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

        return base_config
