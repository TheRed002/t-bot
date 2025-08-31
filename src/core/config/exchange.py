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

    # WebSocket settings
    websocket_message_timeout: int = Field(
        default=60, description="WebSocket message timeout in seconds"
    )
    websocket_ping_interval: int = Field(
        default=30, description="WebSocket ping interval in seconds"
    )
    websocket_ping_timeout: int = Field(default=10, description="WebSocket ping timeout in seconds")
    websocket_close_timeout: int = Field(
        default=10, description="WebSocket close timeout in seconds"
    )
    websocket_max_reconnect_attempts: int = Field(
        default=10, description="Maximum WebSocket reconnection attempts"
    )
    websocket_reconnect_delay: float = Field(
        default=1.0, description="WebSocket reconnection delay in seconds"
    )
    websocket_max_reconnect_delay: float = Field(
        default=60.0, description="Maximum WebSocket reconnection delay in seconds"
    )
    websocket_heartbeat_interval: float = Field(
        default=30.0, description="WebSocket heartbeat interval in seconds"
    )
    websocket_health_check_interval: float = Field(
        default=30.0, description="WebSocket health check interval in seconds"
    )

    # Connection pool settings
    connection_pool_size: int = Field(default=3, description="Default connection pool size")
    connection_pool_max_size: int = Field(default=10, description="Maximum connection pool size")
    connection_pool_keepalive_timeout: int = Field(
        default=300, description="Connection keepalive timeout in seconds"
    )
    connection_pool_health_check_interval: int = Field(
        default=60, description="Connection pool health check interval in seconds"
    )
    connection_pool_retry_attempts: int = Field(
        default=3, description="Connection pool retry attempts"
    )
    connection_pool_circuit_breaker_timeout: int = Field(
        default=60, description="Circuit breaker timeout in seconds"
    )

    # Rate limiting settings
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")
    rate_limit_max_queue_size: int = Field(
        default=1000, description="Maximum rate limit queue size"
    )

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

    def get_websocket_config(self, exchange: str) -> dict[str, Any]:
        """Get WebSocket configuration for a specific exchange."""
        base_config: dict[str, Any] = {
            "reconnect_attempts": self.websocket_max_reconnect_attempts,
            "ping_interval": self.websocket_ping_interval,
            "ping_timeout": self.websocket_ping_timeout,
            "close_timeout": self.websocket_close_timeout,
            "message_timeout": self.websocket_message_timeout,
            "timeout": self.connection_timeout,
            "reconnect_delay": self.websocket_reconnect_delay,
            "max_reconnect_delay": self.websocket_max_reconnect_delay,
            "heartbeat_interval": self.websocket_heartbeat_interval,
            "health_check_interval": self.websocket_health_check_interval,
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

    def get_connection_pool_config(self) -> dict[str, Any]:
        """Get connection pool configuration."""
        return {
            "pool_size": self.connection_pool_size,
            "max_pool_size": self.connection_pool_max_size,
            "keepalive_timeout": self.connection_pool_keepalive_timeout,
            "health_check_interval": self.connection_pool_health_check_interval,
            "retry_attempts": self.connection_pool_retry_attempts,
            "circuit_breaker_timeout": self.connection_pool_circuit_breaker_timeout,
            "connection_timeout": self.connection_timeout,
            "request_timeout": self.request_timeout,
        }

    def get_rate_limit_config(self) -> dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            "requests_per_second": self.rate_limit_per_second,
            "window_seconds": self.rate_limit_window_seconds,
            "buffer": self.rate_limit_buffer,
            "max_queue_size": self.rate_limit_max_queue_size,
        }
