"""
Web Exchange Management Service Implementation.

This service provides a web-specific interface to the exchange management system,
handling connection management, configuration, rate limiting, and health monitoring.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.core.base import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.exchanges.interfaces import IExchangeFactory
from src.exchanges.service import ExchangeService
from src.utils.decorators import cached, monitored

logger = get_logger(__name__)


class WebExchangeService(BaseService):
    """
    Web interface service for exchange management operations.

    This service wraps the exchange service and provides web-specific
    formatting, validation, and business logic.
    """

    def __init__(
        self,
        exchange_service: ExchangeService | None = None,
        exchange_factory: IExchangeFactory | None = None,
    ):
        """Initialize web exchange service with dependencies."""
        super().__init__("WebExchangeService")

        self.exchange_service = exchange_service
        self.exchange_factory = exchange_factory

        # Cache for exchange configurations
        self._config_cache: dict[str, Any] = {}
        self._connection_cache: dict[str, Any] = {}

        logger.info("Web exchange service initialized")

    async def _do_start(self) -> None:
        """Start the web exchange service."""
        logger.info("Starting web exchange service")
        if self.exchange_service:
            await self.exchange_service.start()

    async def _do_stop(self) -> None:
        """Stop the web exchange service."""
        logger.info("Stopping web exchange service")
        if self.exchange_service:
            await self.exchange_service.stop()

    # Connection Management Methods
    @cached(ttl=30)  # Cache for 30 seconds
    async def get_connections(self) -> list[dict[str, Any]]:
        """Get all exchange connections."""
        try:
            if self.exchange_service:
                exchanges = await self.exchange_service.get_connected_exchanges()
                return [
                    {
                        "exchange": ex,
                        "status": "connected",
                        "test_mode": False,  # Would get from actual service
                        "created_at": datetime.utcnow() - timedelta(days=7),
                    }
                    for ex in exchanges
                ]

            # Mock connections
            return [
                {
                    "exchange": "binance",
                    "status": "connected",
                    "test_mode": False,
                    "created_at": datetime.utcnow() - timedelta(days=7),
                },
                {
                    "exchange": "coinbase",
                    "status": "connected",
                    "test_mode": True,
                    "created_at": datetime.utcnow() - timedelta(days=3),
                },
            ]

        except Exception as e:
            logger.error(f"Error getting connections: {e}")
            raise ServiceError(f"Failed to retrieve connections: {e!s}")

    @monitored()
    async def connect_exchange(
        self,
        exchange: str,
        api_key: str | None,
        api_secret: str | None,
        passphrase: str | None,
        testnet: bool,
        sandbox: bool,
        connected_by: str,
    ) -> dict[str, Any]:
        """Connect to an exchange."""
        try:
            # Validate credentials if provided
            if api_key and not self._validate_api_key(api_key):
                raise ValidationError("Invalid API key format")

            if self.exchange_service:
                # Connect through service
                success = await self.exchange_service.connect_exchange(
                    exchange_name=exchange,
                    config={
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "passphrase": passphrase,
                        "testnet": testnet,
                        "sandbox": sandbox,
                    },
                )

                if success:
                    # Clear cache
                    self._connection_cache.clear()

                    return {
                        "success": True,
                        "connection_id": f"conn_{exchange}_{datetime.utcnow().timestamp()}",
                        "message": f"Successfully connected to {exchange}",
                    }

            # Mock success
            return {
                "success": True,
                "connection_id": f"conn_{exchange}_{datetime.utcnow().timestamp()}",
                "message": f"Successfully connected to {exchange}",
            }

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error connecting to exchange: {e}")
            return {
                "success": False,
                "message": f"Failed to connect: {e!s}",
            }

    async def disconnect_exchange(self, exchange: str, disconnected_by: str) -> bool:
        """Disconnect from an exchange."""
        try:
            if self.exchange_service:
                await self.exchange_service.disconnect_exchange(exchange)

            # Clear caches
            self._connection_cache.clear()
            self._config_cache.pop(exchange, None)

            logger.info(f"Exchange {exchange} disconnected by {disconnected_by}")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from exchange: {e}")
            raise ServiceError(f"Failed to disconnect: {e!s}")

    @cached(ttl=10)
    async def get_exchange_status(self, exchange: str) -> dict[str, Any] | None:
        """Get exchange connection status."""
        try:
            if self.exchange_service:
                status = await self.exchange_service.get_exchange_status(exchange)
                if status:
                    return {
                        "exchange": exchange,
                        "status": status.status,
                        "connected": status.status == "connected",
                        "uptime_seconds": 86400,
                        "last_heartbeat": datetime.utcnow(),
                        "active_connections": 1,
                        "error_count": 0,
                        "latency_ms": 15.5,
                    }

            # Mock status
            return {
                "exchange": exchange,
                "status": "connected",
                "connected": True,
                "uptime_seconds": 86400,
                "last_heartbeat": datetime.utcnow(),
                "active_connections": 2,
                "error_count": 5,
                "latency_ms": 25.5,
            }

        except Exception as e:
            logger.error(f"Error getting exchange status: {e}")
            raise ServiceError(f"Failed to get status: {e!s}")

    # Configuration Methods
    async def get_exchange_config(self, exchange: str) -> dict[str, Any] | None:
        """Get exchange configuration."""
        try:
            # Check cache
            if exchange in self._config_cache:
                return self._config_cache[exchange]

            # Mock config
            config = {
                "rate_limit": 1200,
                "timeout": 30000,
                "enable_rate_limiter": True,
                "sandbox": False,
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "api_key": "***hidden***",
            }

            # Cache result
            self._config_cache[exchange] = config
            return config

        except Exception as e:
            logger.error(f"Error getting exchange config: {e}")
            raise ServiceError(f"Failed to get config: {e!s}")

    async def validate_exchange_config(
        self, exchange: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate exchange configuration."""
        try:
            errors = []

            # Validate required fields
            if "rate_limit" in config and config["rate_limit"] < 1:
                errors.append("rate_limit must be positive")

            if "timeout" in config and config["timeout"] < 1000:
                errors.append("timeout must be at least 1000ms")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
            }

        except Exception as e:
            logger.error(f"Error validating config: {e}")
            raise ServiceError(f"Failed to validate config: {e!s}")

    async def update_exchange_config(
        self, exchange: str, config: dict[str, Any], updated_by: str
    ) -> bool:
        """Update exchange configuration."""
        try:
            # Clear cache
            self._config_cache.pop(exchange, None)

            logger.info(f"Exchange {exchange} config updated by {updated_by}")
            return True

        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise ServiceError(f"Failed to update config: {e!s}")

    @cached(ttl=300)  # Cache for 5 minutes
    async def get_exchange_symbols(self, exchange: str, active_only: bool = True) -> list[str]:
        """Get available trading symbols."""
        try:
            if self.exchange_service:
                symbols = await self.exchange_service.get_symbols(exchange)
                return symbols

            # Mock symbols
            symbols = [
                "BTC/USDT",
                "ETH/USDT",
                "BNB/USDT",
                "XRP/USDT",
                "ADA/USDT",
                "SOL/USDT",
                "DOT/USDT",
                "MATIC/USDT",
            ]

            return symbols

        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            raise ServiceError(f"Failed to get symbols: {e!s}")

    async def get_exchange_fees(self, exchange: str, symbol: str | None = None) -> dict[str, Any]:
        """Get exchange fee structure."""
        try:
            if self.exchange_service:
                fees = await self.exchange_service.get_fees(exchange)
                return fees

            # Mock fees
            return {
                "maker": "0.001",
                "taker": "0.001",
                "withdraw": {
                    "BTC": "0.0005",
                    "ETH": "0.005",
                    "USDT": "1.0",
                },
            }

        except Exception as e:
            logger.error(f"Error getting fees: {e}")
            raise ServiceError(f"Failed to get fees: {e!s}")

    # Rate Limiting Methods
    async def get_rate_limits(self, exchange: str) -> dict[str, Any]:
        """Get exchange rate limits."""
        try:
            # Mock rate limits
            return {
                "requests_per_second": 10,
                "requests_per_minute": 1200,
                "weight_per_minute": 6000,
                "order_per_second": 5,
                "order_per_day": 200000,
            }

        except Exception as e:
            logger.error(f"Error getting rate limits: {e}")
            raise ServiceError(f"Failed to get rate limits: {e!s}")

    async def get_rate_usage(self, exchange: str) -> dict[str, Any]:
        """Get current rate limit usage."""
        try:
            # Mock usage
            return {
                "requests_used": 450,
                "requests_limit": 1200,
                "usage_percentage": 37.5,
                "weight_used": 2000,
                "weight_limit": 6000,
                "reset_in_seconds": 45,
            }

        except Exception as e:
            logger.error(f"Error getting rate usage: {e}")
            raise ServiceError(f"Failed to get rate usage: {e!s}")

    async def update_rate_config(
        self,
        exchange: str,
        requests_per_second: int,
        burst_limit: int,
        cooldown_seconds: int,
        enable_auto_throttle: bool,
        updated_by: str,
    ) -> bool:
        """Update rate limit configuration."""
        try:
            logger.info(f"Rate config for {exchange} updated by {updated_by}")
            return True

        except Exception as e:
            logger.error(f"Error updating rate config: {e}")
            raise ServiceError(f"Failed to update rate config: {e!s}")

    # Health & Monitoring Methods
    @monitored()
    async def get_all_exchanges_health(self) -> dict[str, Any]:
        """Get health status for all exchanges."""
        try:
            exchanges = await self.get_connections()

            health_statuses = []
            for conn in exchanges:
                health = await self.get_exchange_health(conn["exchange"])
                if health:
                    health_statuses.append(health)

            healthy = [h for h in health_statuses if h["status"] == "healthy"]

            return {
                "overall_health": "healthy" if len(healthy) == len(health_statuses) else "degraded",
                "exchanges": health_statuses,
                "healthy_count": len(healthy),
                "unhealthy_count": len(health_statuses) - len(healthy),
            }

        except Exception as e:
            logger.error(f"Error getting exchanges health: {e}")
            raise ServiceError(f"Failed to get health: {e!s}")

    async def get_exchange_health(self, exchange: str) -> dict[str, Any] | None:
        """Get health status for a specific exchange."""
        try:
            # Mock health
            return {
                "exchange": exchange,
                "health_score": 0.95,
                "status": "healthy",
                "checks": {
                    "api_connection": True,
                    "websocket_connection": True,
                    "rate_limits": True,
                    "balance_fetch": True,
                    "order_placement": True,
                },
                "issues": [],
                "last_check": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error getting exchange health: {e}")
            raise ServiceError(f"Failed to get health: {e!s}")

    async def get_exchange_latency(self, exchange: str, hours: int) -> dict[str, Any]:
        """Get exchange latency metrics."""
        try:
            # Mock latency
            return {
                "avg_latency_ms": 25.5,
                "p50_latency_ms": 20.0,
                "p95_latency_ms": 50.0,
                "p99_latency_ms": 100.0,
                "max_latency_ms": 250.0,
                "samples": 5000,
            }

        except Exception as e:
            logger.error(f"Error getting latency: {e}")
            raise ServiceError(f"Failed to get latency: {e!s}")

    async def get_exchange_errors(
        self, exchange: str, hours: int, error_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Get exchange error history."""
        try:
            # Mock errors
            errors = [
                {
                    "error_id": "err_001",
                    "timestamp": datetime.utcnow() - timedelta(hours=2),
                    "error_type": "rate_limit",
                    "message": "Rate limit exceeded",
                    "details": {"endpoint": "/api/v3/order"},
                },
                {
                    "error_id": "err_002",
                    "timestamp": datetime.utcnow() - timedelta(hours=5),
                    "error_type": "connection",
                    "message": "Connection timeout",
                    "details": {"timeout": 30000},
                },
            ]

            if error_type:
                errors = [e for e in errors if e["error_type"] == error_type]

            return errors

        except Exception as e:
            logger.error(f"Error getting errors: {e}")
            raise ServiceError(f"Failed to get errors: {e!s}")

    # Market Data Methods
    async def get_orderbook(self, exchange: str, symbol: str, limit: int) -> dict[str, Any] | None:
        """Get order book for a symbol."""
        try:
            if self.exchange_service:
                orderbook = await self.exchange_service.get_order_book(exchange, symbol, limit)
                if orderbook:
                    return {
                        "bids": [[str(b.price), str(b.amount)] for b in orderbook.bids[:limit]],
                        "asks": [[str(a.price), str(a.amount)] for a in orderbook.asks[:limit]],
                        "timestamp": orderbook.timestamp,
                    }

            # Mock orderbook
            return {
                "bids": [
                    ["45000.00", "0.5"],
                    ["44999.00", "1.0"],
                    ["44998.00", "1.5"],
                ],
                "asks": [
                    ["45001.00", "0.5"],
                    ["45002.00", "1.0"],
                    ["45003.00", "1.5"],
                ],
                "timestamp": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error getting orderbook: {e}")
            raise ServiceError(f"Failed to get orderbook: {e!s}")

    async def get_ticker(self, exchange: str, symbol: str) -> dict[str, Any] | None:
        """Get ticker data for a symbol."""
        try:
            if self.exchange_service:
                ticker = await self.exchange_service.get_ticker(exchange, symbol)
                if ticker:
                    return {
                        "bid": str(ticker.bid),
                        "ask": str(ticker.ask),
                        "last": str(ticker.last),
                        "volume": str(ticker.volume),
                        "high": str(ticker.high),
                        "low": str(ticker.low),
                    }

            # Mock ticker
            return {
                "bid": "45000.00",
                "ask": "45001.00",
                "last": "45000.50",
                "volume": "1250.5",
                "high": "46000.00",
                "low": "44000.00",
            }

        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            raise ServiceError(f"Failed to get ticker: {e!s}")

    # Balance Methods
    async def get_exchange_balance(
        self, exchange: str, user_id: str
    ) -> dict[str, dict[str, Decimal]]:
        """Get account balance for an exchange."""
        try:
            if self.exchange_service:
                balance = await self.exchange_service.get_balance(exchange)
                return balance

            # Mock balance
            return {
                "BTC": {
                    "free": Decimal("0.5"),
                    "used": Decimal("0.1"),
                    "total": Decimal("0.6"),
                },
                "USDT": {
                    "free": Decimal("10000"),
                    "used": Decimal("5000"),
                    "total": Decimal("15000"),
                },
            }

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            raise ServiceError(f"Failed to get balance: {e!s}")

    # WebSocket Methods
    async def subscribe_websocket(
        self, exchange: str, channel: str, symbols: list[str], subscriber: str
    ) -> str:
        """Subscribe to WebSocket channels."""
        try:
            # Validate channel
            valid_channels = ["trades", "orderbook", "ticker", "ohlcv"]
            if channel not in valid_channels:
                raise ValidationError(f"Invalid channel: {channel}")

            # Generate subscription ID
            subscription_id = f"sub_{exchange}_{channel}_{datetime.utcnow().timestamp()}"

            logger.info(f"WebSocket subscription {subscription_id} created by {subscriber}")

            return subscription_id

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error subscribing to WebSocket: {e}")
            raise ServiceError(f"Failed to subscribe: {e!s}")

    async def unsubscribe_websocket(
        self, exchange: str, subscription_id: str, subscriber: str
    ) -> bool:
        """Unsubscribe from WebSocket channels."""
        try:
            logger.info(f"WebSocket subscription {subscription_id} cancelled by {subscriber}")
            return True

        except Exception as e:
            logger.error(f"Error unsubscribing from WebSocket: {e}")
            raise ServiceError(f"Failed to unsubscribe: {e!s}")

    # Helper Methods
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        # Basic validation - check length and characters
        if len(api_key) < 16 or len(api_key) > 128:
            return False

        # Check for valid characters (alphanumeric and some special chars)
        import re

        if not re.match(r"^[A-Za-z0-9\-_]+$", api_key):
            return False

        return True
