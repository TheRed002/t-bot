"""
Exchange Management API endpoints for the web interface.

This module provides REST API endpoints for exchange connection management,
configuration, rate limiting, and health monitoring.
All endpoints follow proper service layer patterns.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.utils.decorators import monitored
from src.web_interface.auth.middleware import get_current_user
from src.web_interface.dependencies import get_web_exchange_service, get_web_auth_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/exchanges", tags=["exchanges"])


# Request/Response Models
class ExchangeConnectionRequest(BaseModel):
    """Request model for exchange connection."""

    exchange: str
    api_key: str | None = None
    api_secret: str | None = None
    passphrase: str | None = None  # For exchanges that require it
    testnet: bool = False
    sandbox: bool = False


class ExchangeConfigRequest(BaseModel):
    """Request model for exchange configuration."""

    config: dict[str, Any]
    validate: bool = True


class RateLimitConfigRequest(BaseModel):
    """Request model for rate limit configuration."""

    requests_per_second: int = Field(ge=1, le=1000)
    burst_limit: int = Field(ge=1, le=5000)
    cooldown_seconds: int = Field(ge=1, le=300)
    enable_auto_throttle: bool = True


class ExchangeStatusResponse(BaseModel):
    """Response model for exchange status."""

    exchange: str
    status: str
    connected: bool
    uptime_seconds: int
    last_heartbeat: datetime
    active_connections: int
    error_count: int
    latency_ms: float


class ExchangeHealthResponse(BaseModel):
    """Response model for exchange health."""

    exchange: str
    health_score: float
    status: str
    checks: dict[str, bool]
    issues: list[str]
    last_check: datetime


# Connection Management Endpoints
@router.get("/connections")
@monitored()
async def get_exchange_connections(
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get all exchange connections."""
    try:
        connections = await web_exchange_service.get_connections()
        return {"connections": connections, "count": len(connections)}

    except Exception as e:
        logger.error(f"Error getting exchange connections: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve connections")


@router.post("/connect/{exchange}")
@monitored()
async def connect_exchange(
    exchange: str,
    request: ExchangeConnectionRequest,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Connect to an exchange."""
    try:
        # Use auth service for authorization
        web_auth_service.require_trading_permission(current_user)

        # Validate exchange name using service
        supported_exchanges = web_exchange_service.get_supported_exchanges()
        if exchange.lower() not in [ex.lower() for ex in supported_exchanges]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported exchange: {exchange}. Supported: {supported_exchanges}",
            )

        result = await web_exchange_service.connect_exchange(
            exchange=exchange,
            api_key=request.api_key,
            api_secret=request.api_secret,
            passphrase=request.passphrase,
            testnet=request.testnet,
            sandbox=request.sandbox,
            connected_by=current_user["user_id"],
        )

        return {
            "status": "connected" if result["success"] else "failed",
            "exchange": exchange,
            "connection_id": result.get("connection_id"),
            "message": result.get("message"),
        }

    except HTTPException:
        raise
    except ServiceError as e:
        logger.error(f"Service error in exchange connection: {e}")
        if "Insufficient permissions" in str(e):
            raise HTTPException(status_code=403, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        logger.error(f"Validation error in exchange connection: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error connecting to exchange: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to exchange")


@router.post("/disconnect/{exchange}")
@monitored()
async def disconnect_exchange(
    exchange: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Disconnect from an exchange."""
    try:
        # Use auth service for authorization
        web_auth_service.require_trading_permission(current_user)

        success = await web_exchange_service.disconnect_exchange(
            exchange=exchange, disconnected_by=current_user["user_id"]
        )

        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to disconnect from {exchange}")

        return {"status": "disconnected", "exchange": exchange}

    except HTTPException:
        raise
    except ServiceError as e:
        logger.error(f"Service error in exchange disconnection: {e}")
        if "Insufficient permissions" in str(e):
            raise HTTPException(status_code=403, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error disconnecting from exchange: {e}")
        raise HTTPException(status_code=500, detail="Failed to disconnect from exchange")


@router.get("/{exchange}/status", response_model=ExchangeStatusResponse)
@monitored()
async def get_exchange_status(
    exchange: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get exchange connection status."""
    try:
        status = await web_exchange_service.get_exchange_status(exchange)

        if not status:
            raise HTTPException(status_code=404, detail=f"Exchange {exchange} not found")

        return ExchangeStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting exchange status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange status")


# Configuration Endpoints
@router.get("/{exchange}/config")
@monitored()
async def get_exchange_config(
    exchange: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get exchange configuration."""
    try:
        config = await web_exchange_service.get_exchange_config(exchange)

        if not config:
            raise HTTPException(status_code=404, detail=f"Configuration for {exchange} not found")

        # Remove sensitive information
        if "api_key" in config:
            config["api_key"] = "***hidden***"
        if "api_secret" in config:
            config["api_secret"] = "***hidden***"

        return {"exchange": exchange, "config": config}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting exchange config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange configuration")


@router.put("/{exchange}/config")
@monitored()
async def update_exchange_config(
    exchange: str,
    request: ExchangeConfigRequest,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Update exchange configuration."""
    try:
        # Use auth service for authorization
        web_auth_service.require_admin_or_developer_permission(current_user)

        # Validate configuration if requested
        if request.validate:
            validation_result = await web_exchange_service.validate_exchange_config(
                exchange, request.config
            )
            if not validation_result["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration: {validation_result['errors']}",
                )

        success = await web_exchange_service.update_exchange_config(
            exchange=exchange,
            config=request.config,
            updated_by=current_user["user_id"],
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to update configuration")

        return {"status": "updated", "exchange": exchange}

    except HTTPException:
        raise
    except ServiceError as e:
        logger.error(f"Service error in exchange config update: {e}")
        if "Insufficient permissions" in str(e):
            raise HTTPException(status_code=403, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating exchange config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update exchange configuration")


@router.get("/{exchange}/symbols")
@monitored()
async def get_exchange_symbols(
    exchange: str,
    active_only: bool = True,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get available trading symbols for an exchange."""
    try:
        symbols = await web_exchange_service.get_exchange_symbols(
            exchange=exchange, active_only=active_only
        )

        return {
            "exchange": exchange,
            "symbols": symbols,
            "count": len(symbols),
            "active_only": active_only,
        }

    except Exception as e:
        logger.error(f"Error getting exchange symbols: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange symbols")


@router.get("/{exchange}/fees")
@monitored()
async def get_exchange_fees(
    exchange: str,
    symbol: str | None = None,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get exchange fee structure."""
    try:
        fees = await web_exchange_service.get_exchange_fees(exchange=exchange, symbol=symbol)

        return {"exchange": exchange, "symbol": symbol, "fees": fees}

    except Exception as e:
        logger.error(f"Error getting exchange fees: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange fees")


# Rate Limiting Endpoints
@router.get("/{exchange}/rate-limits")
@monitored()
async def get_rate_limits(
    exchange: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get exchange rate limits."""
    try:
        limits = await web_exchange_service.get_rate_limits(exchange)

        return {"exchange": exchange, "rate_limits": limits}

    except Exception as e:
        logger.error(f"Error getting rate limits: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve rate limits")


@router.get("/{exchange}/rate-usage")
@monitored()
async def get_rate_usage(
    exchange: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get current rate limit usage."""
    try:
        usage = await web_exchange_service.get_rate_usage(exchange)

        return {
            "exchange": exchange,
            "usage": usage,
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Error getting rate usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve rate usage")


@router.put("/{exchange}/rate-config")
@monitored()
async def update_rate_config(
    exchange: str,
    request: RateLimitConfigRequest,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Update rate limit configuration."""
    try:
        # Use auth service for authorization
        web_auth_service.require_admin_or_developer_permission(current_user)

        success = await web_exchange_service.update_rate_config(
            exchange=exchange,
            requests_per_second=request.requests_per_second,
            burst_limit=request.burst_limit,
            cooldown_seconds=request.cooldown_seconds,
            enable_auto_throttle=request.enable_auto_throttle,
            updated_by=current_user["user_id"],
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to update rate configuration")

        return {"status": "updated", "exchange": exchange}

    except HTTPException:
        raise
    except ServiceError as e:
        logger.error(f"Service error in rate config update: {e}")
        if "Insufficient permissions" in str(e):
            raise HTTPException(status_code=403, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating rate config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update rate configuration")


# Health & Monitoring Endpoints
@router.get("/health")
@monitored()
async def get_exchanges_health(
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get health status for all exchanges."""
    try:
        health = await web_exchange_service.get_all_exchanges_health()

        return {
            "overall_health": health["overall_health"],
            "exchanges": health["exchanges"],
            "healthy_count": health["healthy_count"],
            "unhealthy_count": health["unhealthy_count"],
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Error getting exchanges health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchanges health")


@router.get("/{exchange}/health", response_model=ExchangeHealthResponse)
@monitored()
async def get_exchange_health(
    exchange: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get health status for a specific exchange."""
    try:
        health = await web_exchange_service.get_exchange_health(exchange)

        if not health:
            raise HTTPException(status_code=404, detail=f"Exchange {exchange} not found")

        return ExchangeHealthResponse(**health)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting exchange health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange health")


@router.get("/{exchange}/latency")
@monitored()
async def get_exchange_latency(
    exchange: str,
    hours: int = Query(default=1, ge=1, le=24),
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get exchange latency metrics."""
    try:
        latency = await web_exchange_service.get_exchange_latency(exchange=exchange, hours=hours)

        return {
            "exchange": exchange,
            "time_window_hours": hours,
            "latency_metrics": latency,
        }

    except Exception as e:
        logger.error(f"Error getting exchange latency: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange latency")


@router.get("/{exchange}/errors")
@monitored()
async def get_exchange_errors(
    exchange: str,
    hours: int = Query(default=24, ge=1, le=168),
    error_type: str | None = None,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get exchange error history."""
    try:
        errors = await web_exchange_service.get_exchange_errors(
            exchange=exchange, hours=hours, error_type=error_type
        )

        return {
            "exchange": exchange,
            "errors": errors,
            "count": len(errors),
            "time_window_hours": hours,
            "error_type_filter": error_type,
        }

    except Exception as e:
        logger.error(f"Error getting exchange errors: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange errors")


# Order Book & Market Data Endpoints
@router.get("/{exchange}/orderbook")
@monitored()
async def get_orderbook(
    exchange: str,
    symbol: str,
    limit: int = Query(default=20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get order book for a symbol."""
    try:
        orderbook = await web_exchange_service.get_orderbook(
            exchange=exchange, symbol=symbol, limit=limit
        )

        if not orderbook:
            raise HTTPException(status_code=404, detail=f"Order book not available for {symbol}")

        return {
            "exchange": exchange,
            "symbol": symbol,
            "bids": orderbook["bids"],
            "asks": orderbook["asks"],
            "timestamp": orderbook["timestamp"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting orderbook: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve orderbook")


@router.get("/{exchange}/ticker")
@monitored()
async def get_ticker(
    exchange: str,
    symbol: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Get ticker data for a symbol."""
    try:
        ticker = await web_exchange_service.get_ticker(exchange=exchange, symbol=symbol)

        if not ticker:
            raise HTTPException(status_code=404, detail=f"Ticker not available for {symbol}")

        return {
            "exchange": exchange,
            "symbol": symbol,
            "ticker": ticker,
            "timestamp": datetime.utcnow(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ticker: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve ticker")


# Balance & Account Endpoints
@router.get("/{exchange}/balance")
@monitored()
async def get_exchange_balance(
    exchange: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Get account balance for an exchange."""
    try:
        # Use auth service for authorization
        web_auth_service.require_management_permission(current_user)

        balance = await web_exchange_service.get_exchange_balance(
            exchange=exchange, user_id=current_user["user_id"]
        )

        # Format balances as strings for Decimal precision
        formatted_balance = {
            currency: {
                "free": str(amounts["free"]),
                "used": str(amounts["used"]),
                "total": str(amounts["total"]),
            }
            for currency, amounts in balance.items()
        }

        return {
            "exchange": exchange,
            "balance": formatted_balance,
            "timestamp": datetime.utcnow(),
        }

    except HTTPException:
        raise
    except ServiceError as e:
        logger.error(f"Service error in exchange balance retrieval: {e}")
        if "Insufficient permissions" in str(e):
            raise HTTPException(status_code=403, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting exchange balance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange balance")


# WebSocket Management Endpoints
@router.post("/{exchange}/websocket/subscribe")
@monitored()
async def subscribe_websocket(
    exchange: str,
    channel: str,
    symbols: list[str],
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Subscribe to WebSocket channels."""
    try:
        subscription_id = await web_exchange_service.subscribe_websocket(
            exchange=exchange,
            channel=channel,
            symbols=symbols,
            subscriber=current_user["user_id"],
        )

        return {
            "status": "subscribed",
            "exchange": exchange,
            "channel": channel,
            "symbols": symbols,
            "subscription_id": subscription_id,
        }

    except ValidationError as e:
        logger.error(f"Validation error in WebSocket subscription: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error subscribing to WebSocket: {e}")
        raise HTTPException(status_code=500, detail="Failed to subscribe to WebSocket")


@router.post("/{exchange}/websocket/unsubscribe")
@monitored()
async def unsubscribe_websocket(
    exchange: str,
    subscription_id: str,
    current_user: dict = Depends(get_current_user),
    web_exchange_service=Depends(get_web_exchange_service),
):
    """Unsubscribe from WebSocket channels."""
    try:
        success = await web_exchange_service.unsubscribe_websocket(
            exchange=exchange,
            subscription_id=subscription_id,
            subscriber=current_user["user_id"],
        )

        if not success:
            raise HTTPException(status_code=404, detail="Subscription not found")

        return {
            "status": "unsubscribed",
            "exchange": exchange,
            "subscription_id": subscription_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsubscribing from WebSocket: {e}")
        raise HTTPException(status_code=500, detail="Failed to unsubscribe from WebSocket")
