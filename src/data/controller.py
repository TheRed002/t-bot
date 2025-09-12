"""
Data Controller - Orchestrate Data Service Operations

This controller handles HTTP/API requests and orchestrates calls to data services.
Controllers should only handle request/response logic and delegate business logic to services.
"""

from typing import Any

from src.core import BaseComponent, Config
from src.core.exceptions import ValidationError
from src.core.types import MarketData
from src.data.constants import (
    DEFAULT_DATA_LIMIT,
    DEFAULT_EXCHANGE,
    DEFAULT_L1_CACHE_TTL_SECONDS,
    MAX_DATA_LIMIT,
)
from src.data.interfaces import DataServiceInterface
from src.data.types import DataRequest
from src.error_handling import FallbackStrategy, with_fallback


class DataController(BaseComponent):
    """
    Data controller for handling data operations requests.

    This controller follows proper service layer architecture:
    - Controllers handle request/response and input validation
    - Business logic is delegated to services
    - No direct database or cache access
    """

    def __init__(self, config: Config, data_service: DataServiceInterface):
        """
        Initialize data controller.

        Args:
            config: Application configuration
            data_service: Injected data service interface
        """
        super().__init__()
        self.config = config

        # Required dependency - must be injected
        if data_service is None:
            raise ValidationError("data_service is required and must be injected")
        self.data_service = data_service

    async def initialize(self) -> None:
        """Initialize the data controller."""
        await self.data_service.initialize()
        self.logger.info("DataController initialized")

    @with_fallback(
        strategy=FallbackStrategy.RETURN_EMPTY,
        default_value={
            "success": False,
            "error": "Storage operation failed",
            "error_code": "STORAGE_ERROR",
        },
    )
    async def store_market_data_request(
        self, data: MarketData | list[MarketData], exchange: str, validate: bool = True
    ) -> dict[str, Any]:
        """
        Handle market data storage request.

        Args:
            data: Market data to store
            exchange: Exchange name
            validate: Whether to validate data

        Returns:
            Response with success status and metrics
        """
        # Input validation (controller responsibility)
        if not data:
            return {"success": False, "error": "No data provided", "error_code": "INVALID_REQUEST"}

        if not exchange:
            return {
                "success": False,
                "error": "Exchange not specified",
                "error_code": "INVALID_REQUEST",
            }

        # Delegate to service layer (no business logic in controller)
        success = await self.data_service.store_market_data(
            data=data, exchange=exchange, validate=validate
        )

        # Format response
        return {
            "success": success,
            "records_count": len(data) if isinstance(data, list) else 1,
            "exchange": exchange,
        }

    @with_fallback(
        strategy=FallbackStrategy.RETURN_EMPTY,
        default_value={
                "success": False,
                "error": "Retrieval operation failed",
                "error_code": "RETRIEVAL_ERROR",
                "data": [],
                "count": 0,
        },
    )
    async def get_market_data_request(
        self, symbol: str, exchange: str, limit: int = DEFAULT_DATA_LIMIT, use_cache: bool = True
    ) -> dict[str, Any]:
        """
        Handle market data retrieval request.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            limit: Maximum records to return
            use_cache: Whether to use cached data

        Returns:
            Response with market data and metadata
        """
        # Input validation (controller responsibility)
        if not symbol:
            return {
                "success": False,
                "error": "Symbol not provided",
                "error_code": "INVALID_REQUEST",
            }

        if not exchange:
            return {
                "success": False,
                "error": "Exchange not provided",
                "error_code": "INVALID_REQUEST",
            }

        # Create request object
        request = DataRequest(symbol=symbol, exchange=exchange, limit=limit, use_cache=use_cache, cache_ttl=DEFAULT_L1_CACHE_TTL_SECONDS)

        # Delegate to service layer
        records = await self.data_service.get_market_data(request)

        # Format response (delegate financial data formatting to service layer)
        return {
            "success": True,
            "data": [
                {
                    "symbol": record.symbol,
                    "exchange": record.exchange,
                    "timestamp": record.data_timestamp.isoformat()
                    if record.data_timestamp
                    else None,
                    "price": str(record.close_price) if record.close_price else None,
                    "volume": str(record.volume) if record.volume else None,
                }
                for record in records
            ],
            "count": len(records),
            "symbol": symbol,
            "exchange": exchange,
        }

    @with_fallback(
        strategy=FallbackStrategy.RETURN_EMPTY,
        default_value={
                "success": False,
                "error": "Count operation failed",
                "error_code": "COUNT_ERROR",
                "count": 0,
        },
    )
    async def get_data_count_request(
        self, symbol: str, exchange: str = "binance"
    ) -> dict[str, Any]:
        """
        Handle data count request.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Response with data count
        """
        # Input validation (controller responsibility)
        if not symbol:
            return {
                "success": False,
                "error": "Symbol not provided",
                "error_code": "INVALID_REQUEST",
            }

        # Delegate to service layer
        count = await self.data_service.get_data_count(symbol, exchange)

        # Format response
        return {"success": True, "count": count, "symbol": symbol, "exchange": exchange}

    @with_fallback(
        strategy=FallbackStrategy.RETURN_EMPTY,
        default_value={
                "success": False,
                "error": "Recent data operation failed",
                "error_code": "RECENT_DATA_ERROR",
                "data": [],
                "count": 0,
        },
    )
    async def get_recent_data_request(
        self, symbol: str, limit: int = DEFAULT_DATA_LIMIT, exchange: str = DEFAULT_EXCHANGE
    ) -> dict[str, Any]:
        """
        Handle recent data request.

        Args:
            symbol: Trading symbol
            limit: Number of recent records
            exchange: Exchange name

        Returns:
            Response with recent market data
        """
        # Input validation (controller responsibility)
        if not symbol:
            return {
                "success": False,
                "error": "Symbol not provided",
                "error_code": "INVALID_REQUEST",
            }

        if limit <= 0 or limit > MAX_DATA_LIMIT:
            return {
                "success": False,
                "error": f"Invalid limit (must be between 1 and {MAX_DATA_LIMIT})",
                "error_code": "INVALID_REQUEST",
            }

        # Delegate to service layer
        data = await self.data_service.get_recent_data(symbol, limit, exchange)

        # Format response (delegate financial data formatting to service layer)
        return {
            "success": True,
            "data": [
                {
                    "symbol": item.symbol,
                    "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                    "open": str(item.open) if item.open else None,
                    "high": str(item.high) if item.high else None,
                    "low": str(item.low) if item.low else None,
                    "close": str(item.close) if item.close else None,
                    "volume": str(item.volume) if item.volume else None,
                }
                for item in data
            ],
            "count": len(data),
            "symbol": symbol,
            "exchange": exchange,
        }

    @with_fallback(
        strategy=FallbackStrategy.RETURN_EMPTY,
        default_value={
                "success": False,
                "status": "unhealthy",
                "error": "Health check failed",
                "error_code": "HEALTH_CHECK_ERROR",
        },
    )
    async def get_health_status_request(self) -> dict[str, Any]:
        """
        Handle health check request.

        Returns:
            Health status response
        """
        # Delegate to service layer
        health = await self.data_service.health_check()

        # Format response for API
        return {
            "success": True,
            "status": health.get("status", "unknown"),
            "details": health.get("details", {}),
            "timestamp": health.get("timestamp"),
        }

    @with_fallback(strategy=FallbackStrategy.RETURN_NONE)
    async def cleanup(self) -> None:
        """Cleanup controller resources."""
        await self.data_service.cleanup()
        self.logger.info("DataController cleanup completed")
