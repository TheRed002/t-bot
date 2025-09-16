"""
Trading API endpoints for T-Bot web interface.

This module provides trading operations including order placement, cancellation,
order history, and trade analysis functionality.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.exceptions import ErrorSeverity, NetworkError, TimeoutError
from src.core.logging import get_logger
from src.core.types import OrderSide, OrderType
from src.error_handling import (
    get_global_error_handler,
    with_error_context,
    with_retry,
)
from src.error_handling.context import ErrorContext
from src.utils.web_interface_utils import handle_api_error
from src.web_interface.di_registration import get_web_trading_service
from src.web_interface.security.auth import User, get_current_user, get_trading_user

logger = get_logger(__name__)
router = APIRouter()


def get_trading_service() -> Any:
    """Get trading service through web service layer."""
    # Controllers should only use web services, not facades directly
    # The web service will handle facade interactions internally
    return get_web_trading_service_instance()


def get_web_trading_service_instance():
    """Get web trading service for business logic through DI."""
    return get_web_trading_service()


# Deprecated function for backward compatibility
def set_dependencies(engine: Any, orchestrator: Any) -> None:
    """DEPRECATED: Use service registry instead."""
    logger.warning("set_dependencies is deprecated. Use service registry instead.")


class PlaceOrderRequest(BaseModel):
    """Request model for placing an order."""

    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    order_type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    price: Decimal | None = Field(None, description="Order price (for limit orders)")
    stop_price: Decimal | None = Field(None, description="Stop price (for stop orders)")
    exchange: str = Field(..., description="Exchange to place order on")
    bot_id: str | None = Field(None, description="Bot ID (if order is for specific bot)")
    time_in_force: str = Field(default="GTC", description="Time in force")
    client_order_id: str | None = Field(None, description="Client-specified order ID")


class CancelOrderRequest(BaseModel):
    """Request model for cancelling an order."""

    order_id: str | None = None
    client_order_id: str | None = None
    symbol: str | None = None
    exchange: str


class OrderResponse(BaseModel):
    """Response model for order information."""

    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    price: Decimal | None
    stop_price: Decimal | None
    status: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_fill_price: Decimal | None
    commission: Decimal | None
    commission_asset: str | None
    exchange: str
    bot_id: str | None
    created_at: datetime
    updated_at: datetime
    fills: list[dict[str, Any]] | None = None


class TradeResponse(BaseModel):
    """Response model for trade information."""

    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    value: Decimal
    commission: Decimal
    commission_asset: str
    exchange: str
    bot_id: str | None
    executed_at: datetime


class OrderBookResponse(BaseModel):
    """Response model for order book data."""

    symbol: str
    exchange: str
    bids: list[list[Decimal]]  # [price, quantity]
    asks: list[list[Decimal]]  # [price, quantity]
    timestamp: datetime


class MarketDataResponse(BaseModel):
    """Response model for market data."""

    symbol: str
    exchange: str
    price: Decimal
    bid: Decimal | None
    ask: Decimal | None
    volume_24h: Decimal
    change_24h: Decimal
    change_24h_percentage: Decimal
    high_24h: Decimal
    low_24h: Decimal
    timestamp: datetime


@router.post("/orders", response_model=dict[str, Any])
@with_error_context(
    component="trading",
    operation="place_order",
)
@with_retry(
    max_attempts=3,
    exceptions=(NetworkError, TimeoutError),
    backoff_factor=1.5,
)
async def place_order(
    order_request: PlaceOrderRequest, current_user: User = Depends(get_trading_user)
):
    """
    Place a new trading order.

    Args:
        order_request: Order placement parameters
        current_user: Current authenticated user with trading permissions

    Returns:
        Dict: Order placement result

    Raises:
        HTTPException: If order placement fails
    """
    try:
        # Get global error handler for integrated recovery
        global_error_handler = get_global_error_handler()
    except Exception as e:
        logger.warning(f"Could not get global error handler: {e}")
        global_error_handler = None

    try:
        web_trading_service = get_web_trading_service_instance()
        # Controllers should not access facades directly - use service layer
        # All operations should go through web_trading_service

        # Apply boundary validation before service call
        from src.utils.messaging_patterns import BoundaryValidator

        boundary_data = {
            "component": "web_interface_trading",
            "operation": "POST /orders",
            "processing_mode": "stream",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_context": True,
            "symbol": order_request.symbol,
            "quantity": str(order_request.quantity),
            "price": str(order_request.price) if order_request.price else None,
        }

        # Validate web_interface to execution boundary
        BoundaryValidator.validate_web_interface_to_error_boundary(boundary_data)

        # Validate order request through web trading service (business logic moved to service)
        validation_result = await web_trading_service.validate_order_request(
            symbol=order_request.symbol,
            side=order_request.side.value,
            order_type=order_request.order_type.value,
            quantity=order_request.quantity,
            price=order_request.price,
        )

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation errors: {', '.join(validation_result['errors'])}",
            )

        validated_data = validation_result["validated_data"]

        # Place order through service layer - service will handle facade calls
        order_result = await web_trading_service.place_order_through_service(
            symbol=validated_data["symbol"],
            side=validated_data["side"],
            order_type=validated_data["order_type"],
            quantity=validated_data["quantity"],
            price=validated_data["price"],
        )
        order_id = order_result.get("order_id")

        logger.info(
            "Order placed successfully",
            order_id=order_id,
            symbol=validated_data["symbol"],
            side=validated_data["side"],
            quantity=validated_data["quantity"],
            exchange=order_request.exchange,
            user=current_user.username,
        )

        # Format response through web trading service (business logic moved to service)
        request_data = {
            "symbol": validated_data["symbol"],
            "side": validated_data["side"],
            "order_type": validated_data["order_type"],
            "quantity": validated_data["quantity"],
            "price": validated_data["price"],
            "client_order_id": order_request.client_order_id,
        }

        order_result = {"order_id": order_id}
        formatted_response = await web_trading_service.format_order_response(
            order_result, request_data
        )

        return formatted_response

    except HTTPException:
        # Re-raise FastAPI exceptions
        raise
    except Exception as e:
        # Create error context with consistent data transformation
        from src.execution.data_transformer import ExecutionDataTransformer

        error_data = {
            "component": "trading_api",
            "operation": "place_order",
            "severity": "high",
            "user": current_user.username,
            "symbol": order_request.symbol,
            "side": order_request.side.value,
            "quantity": str(order_request.quantity),
            "exchange": order_request.exchange,
            "processing_mode": "stream",
            "data_format": "event_data_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Apply consistent cross-module error transformation
        error_data = ExecutionDataTransformer.apply_cross_module_validation(
            error_data, source_module="web_interface", target_module="error_handling"
        )

        error_context = ErrorContext.from_exception(
            error=e,
            component="trading_api",
            operation="place_order",
            severity=ErrorSeverity.HIGH,
            user=current_user.username,
            symbol=order_request.symbol,
            side=order_request.side.value,
            quantity=str(order_request.quantity),
            exchange=order_request.exchange,
        )

        # Try to handle through global error handler
        if global_error_handler:
            try:
                # Use consistent error data with execution module patterns
                result = await global_error_handler.handle_error(
                    error=e,
                    context=error_data,  # Use transformed error data
                    severity="high",
                )
                if result.get("recovery_attempted"):
                    # Recovery was successful
                    logger.info(
                        "Order error recovered by global handler",
                        recovery_method=result.get("recovery_result", {}).get("method", "unknown"),
                    )
                    return {
                        "success": False,
                        "message": "Order processed with recovery",
                        "recovery_attempted": True,
                        "recovery_method": result.get("recovery_result", {}).get(
                            "method", "unknown"
                        ),
                    }
            except Exception as handler_error:
                logger.warning(f"Error handler failed: {handler_error}")

        logger.error(
            f"Order placement failed: {e}",
            user=current_user.username,
            error_type=error_context.error.__class__.__name__,
            severity=error_context.severity.value,
            operation=error_context.operation,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Order placement failed: {e!s}"
        )


@router.delete("/orders/{order_id}")
@with_error_context(
    component="trading",
    operation="cancel_order",
)
@with_retry(
    max_attempts=2,
    exceptions=(NetworkError, TimeoutError),
    backoff_factor=1.1,
)
async def cancel_order(
    order_id: str,
    exchange: str = Query(..., description="Exchange where order was placed"),
    current_user: User = Depends(get_trading_user),
):
    """
    Cancel a specific order.

    Args:
        order_id: Order ID to cancel
        exchange: Exchange where order was placed
        current_user: Current authenticated user with trading permissions

    Returns:
        Dict: Cancellation result

    Raises:
        HTTPException: If cancellation fails
    """
    try:
        web_trading_service = get_web_trading_service_instance()

        # Cancel order through service layer - service will handle facade calls
        success = await web_trading_service.cancel_order_through_service(order_id)

        if success:
            logger.info(
                "Order cancelled successfully",
                order_id=order_id,
                exchange=exchange,
                user=current_user.username,
            )
            return {
                "success": True,
                "message": f"Order {order_id} cancelled successfully",
                "order_id": order_id,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to cancel order {order_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        # Create error context
        error_context = ErrorContext.from_exception(
            error=e,
            component="trading_api",
            operation="cancel_order",
            severity=ErrorSeverity.MEDIUM,
            user=current_user.username,
            order_id=order_id,
            exchange=exchange,
        )

        logger.error(
            f"Order cancellation failed: {e}",
            order_id=order_id,
            user=current_user.username,
            error_type=error_context.error.__class__.__name__,
            severity=error_context.severity.value,
            operation=error_context.operation,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Order cancellation failed: {e!s}"
        )


@router.get("/orders", response_model=list[OrderResponse])
async def get_orders(
    symbol: str | None = Query(None, description="Filter by symbol"),
    exchange: str | None = Query(None, description="Filter by exchange"),
    status: str | None = Query(None, description="Filter by status"),
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    limit: int = Query(50, ge=1, le=500, description="Number of orders to return"),
    current_user: User = Depends(get_current_user),
):
    """
    Get order history with optional filtering.

    Args:
        symbol: Optional symbol filter
        exchange: Optional exchange filter
        status: Optional status filter
        bot_id: Optional bot ID filter
        limit: Maximum number of orders to return
        current_user: Current authenticated user

    Returns:
        List[OrderResponse]: List of orders

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_trading_service = get_web_trading_service_instance()

        # Prepare filters for service layer (business logic moved to service)
        filters = {
            "symbol": symbol,
            "exchange": exchange,
            "status": status,
            "bot_id": bot_id,
            "limit": limit,
        }

        # Get formatted orders through service layer
        order_data_list = await web_trading_service.get_formatted_orders(filters)

        # Convert to response models
        mock_orders = []
        for order_data in order_data_list:
            order = OrderResponse(**order_data)
            mock_orders.append(order)

        return mock_orders

    except Exception as e:
        raise handle_api_error(e, "Orders retrieval", user=current_user.username)


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    exchange: str = Query(..., description="Exchange where order was placed"),
    current_user: User = Depends(get_current_user),
):
    """
    Get details of a specific order.

    Args:
        order_id: Order ID
        exchange: Exchange where order was placed
        current_user: Current authenticated user

    Returns:
        OrderResponse: Order details

    Raises:
        HTTPException: If order not found or retrieval fails
    """
    try:
        web_trading_service = get_web_trading_service_instance()

        # Get order data through service layer (business logic moved to service)
        order_data = await web_trading_service.get_order_details(order_id, exchange)

        # Convert to response model
        order = OrderResponse(**order_data)
        return order

    except Exception as e:
        raise handle_api_error(
            e, "Order retrieval", user=current_user.username, context={"order_id": order_id}
        )


@router.get("/trades", response_model=list[TradeResponse])
async def get_trades(
    symbol: str | None = Query(None, description="Filter by symbol"),
    exchange: str | None = Query(None, description="Filter by exchange"),
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    start_date: datetime | None = Query(None, description="Start date"),
    end_date: datetime | None = Query(None, description="End date"),
    limit: int = Query(50, ge=1, le=500, description="Number of trades to return"),
    current_user: User = Depends(get_current_user),
):
    """
    Get trade history with optional filtering.

    Args:
        symbol: Optional symbol filter
        exchange: Optional exchange filter
        bot_id: Optional bot ID filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Maximum number of trades to return
        current_user: Current authenticated user

    Returns:
        List[TradeResponse]: List of trades

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_trading_service = get_web_trading_service_instance()

        # Prepare filters for service layer (business logic moved to service)
        filters = {
            "symbol": symbol,
            "exchange": exchange,
            "bot_id": bot_id,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        }

        # Get formatted trades through service layer
        trade_data_list = await web_trading_service.get_formatted_trades(filters)

        # Convert to response models
        mock_trades = []
        for trade_data in trade_data_list:
            trade = TradeResponse(**trade_data)
            mock_trades.append(trade)

        return mock_trades

    except Exception as e:
        raise handle_api_error(e, "Trades retrieval", user=current_user.username)


@router.get("/market-data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(
    symbol: str,
    exchange: str = Query("binance", description="Exchange to get data from"),
    current_user: User = Depends(get_current_user),
):
    """
    Get current market data for a symbol.

    Args:
        symbol: Trading symbol
        exchange: Exchange to get data from
        current_user: Current authenticated user

    Returns:
        MarketDataResponse: Current market data

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_trading_service = get_web_trading_service_instance()

        # Get market data through service layer (business logic moved to service)
        market_data = await web_trading_service.get_market_data_with_context(symbol, exchange)

        # Convert to response model
        mock_data = MarketDataResponse(
            symbol=market_data["symbol"],
            exchange=market_data["exchange"],
            price=market_data["price"],
            bid=market_data["bid"],
            ask=market_data["ask"],
            volume_24h=market_data["volume_24h"],
            change_24h=market_data["change_24h"],
            change_24h_percentage=market_data["change_24h_percentage"],
            high_24h=market_data["high_24h"],
            low_24h=market_data["low_24h"],
            timestamp=market_data["timestamp"],
        )

        return mock_data

    except Exception as e:
        raise handle_api_error(
            e, "Market data retrieval", user=current_user.username, context={"symbol": symbol}
        )


@router.get("/orderbook/{symbol}", response_model=OrderBookResponse)
async def get_order_book(
    symbol: str,
    exchange: str = Query("binance", description="Exchange to get data from"),
    depth: int = Query(20, ge=1, le=100, description="Number of price levels"),
    current_user: User = Depends(get_current_user),
):
    """
    Get order book for a symbol.

    Args:
        symbol: Trading symbol
        exchange: Exchange to get data from
        depth: Number of price levels to return
        current_user: Current authenticated user

    Returns:
        OrderBookResponse: Order book data

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_trading_service = get_web_trading_service_instance()

        # Generate order book through service layer (business logic moved to service)
        order_book_data = await web_trading_service.generate_order_book_data(
            symbol, exchange, depth
        )

        return OrderBookResponse(
            symbol=order_book_data["symbol"],
            exchange=order_book_data["exchange"],
            bids=order_book_data["bids"],
            asks=order_book_data["asks"],
            timestamp=order_book_data["timestamp"],
        )

    except Exception as e:
        raise handle_api_error(
            e, "Order book retrieval", user=current_user.username, context={"symbol": symbol}
        )


@router.get("/execution/status")
async def get_execution_status(current_user: User = Depends(get_current_user)):
    """
    Get execution engine status and statistics.

    Args:
        current_user: Current authenticated user

    Returns:
        Dict: Execution engine status
    """
    try:
        web_trading_service = get_web_trading_service_instance()

        # Get service health status through service layer
        health_status = (
            await web_trading_service.get_service_health()
            if hasattr(web_trading_service, "get_service_health")
            else {"status": "unknown"}
        )

        return {
            "success": True,
            "status": {
                "service_health": health_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Trading service operational",
            },
        }

    except Exception as e:
        raise handle_api_error(e, "Execution status retrieval", user=current_user.username)


@router.get("/orders/active")
async def get_active_orders(current_user: User = Depends(get_current_user)):
    """Get active/open orders only."""
    try:
        # Mock active orders for testing
        return [
            {
                "order_id": "ord_12345",
                "symbol": "BTCUSDT",
                "side": "buy",
                "type": "limit",
                "quantity": "0.001",
                "price": "45000.00",
                "status": "open",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "filled_quantity": "0.0000",
                "remaining_quantity": "0.001",
            },
            {
                "order_id": "ord_67890",
                "symbol": "ETHUSDT",
                "side": "sell",
                "type": "limit",
                "quantity": "0.1",
                "price": "3200.00",
                "status": "partially_filled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "filled_quantity": "0.05",
                "remaining_quantity": "0.05",
            },
        ]

    except Exception as e:
        logger.error(f"Active orders retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get active orders"
        )


@router.get("/positions")
async def get_trading_positions(current_user: User = Depends(get_current_user)):
    """Get current trading positions."""
    try:
        from decimal import Decimal

        # Mock positions for testing
        return [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "size": str(Decimal("0.005")),
                "entry_price": str(Decimal("44500.00")),
                "current_price": str(Decimal("45200.00")),
                "unrealized_pnl": str(Decimal("3.50")),
                "realized_pnl": str(Decimal("0.00")),
                "percentage": "1.57%",
                "margin_used": str(Decimal("225.00")),
                "liquidation_price": str(Decimal("40000.00")),
                "opened_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "symbol": "ETHUSDT",
                "side": "short",
                "size": str(Decimal("0.1")),
                "entry_price": str(Decimal("3250.00")),
                "current_price": str(Decimal("3200.00")),
                "unrealized_pnl": str(Decimal("5.00")),
                "realized_pnl": str(Decimal("0.00")),
                "percentage": "1.54%",
                "margin_used": str(Decimal("162.50")),
                "liquidation_price": str(Decimal("3600.00")),
                "opened_at": datetime.now(timezone.utc).isoformat(),
            },
        ]

    except Exception as e:
        logger.error(f"Positions retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get positions"
        )


@router.get("/balance")
async def get_trading_balance(current_user: User = Depends(get_current_user)):
    """Get trading account balance."""
    try:
        from decimal import Decimal

        # Mock trading balance
        return {
            "success": True,
            "balance": {
                "total_balance": str(Decimal("50000.00")),
                "available_balance": str(Decimal("48750.25")),
                "used_margin": str(Decimal("1249.75")),
                "free_margin": str(Decimal("47500.50")),
                "equity": str(Decimal("50008.50")),
                "margin_level": "3996.68%",
                "currency": "USDT",
            },
            "balances_by_asset": [
                {
                    "asset": "USDT",
                    "free": str(Decimal("48750.25")),
                    "locked": str(Decimal("1249.75")),
                    "total": str(Decimal("50000.00")),
                },
                {
                    "asset": "BTC",
                    "free": str(Decimal("0.00018750")),
                    "locked": str(Decimal("0.0000")),
                    "total": str(Decimal("0.00018750")),
                },
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Balance retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trading balance",
        )
