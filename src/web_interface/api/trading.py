"""
Trading API endpoints for T-Bot web interface.

This module provides trading operations including order placement, cancellation,
order history, and trade analysis functionality.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.core.types import ExecutionInstruction, OrderRequest, OrderSide, OrderType
from src.web_interface.security.auth import User, get_current_user, get_trading_user

logger = get_logger(__name__)
router = APIRouter()

# Global references (set by app startup)
execution_engine = None
bot_orchestrator = None


def set_dependencies(engine, orchestrator):
    """Set global dependencies."""
    global execution_engine, bot_orchestrator
    execution_engine = engine
    bot_orchestrator = orchestrator


class PlaceOrderRequest(BaseModel):
    """Request model for placing an order."""

    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
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
    change_24h_percentage: float
    high_24h: Decimal
    low_24h: Decimal
    timestamp: datetime


@router.post("/orders", response_model=dict[str, Any])
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
        if not execution_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Execution engine not available",
            )

        # Generate client order ID if not provided
        client_order_id = order_request.client_order_id or f"web_{uuid4().hex[:8]}"

        # Create order request
        order = OrderRequest(
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            time_in_force=order_request.time_in_force,
            client_order_id=client_order_id,
        )

        # Create execution instruction
        execution_instruction = ExecutionInstruction(
            order=order,
            preferred_exchanges=[order_request.exchange],
            is_urgent=False,
            allow_partial_fills=True,
        )

        # Get exchange factory (mock for now)
        from src.exchanges.factory import ExchangeFactory

        exchange_factory = ExchangeFactory()

        # Execute order
        execution_result = await execution_engine.execute_order(
            execution_instruction, exchange_factory
        )

        logger.info(
            "Order placed successfully",
            order_id=execution_result.execution_id,
            symbol=order_request.symbol,
            side=order_request.side.value,
            quantity=float(order_request.quantity),
            exchange=order_request.exchange,
            user=current_user.username,
        )

        return {
            "success": True,
            "message": "Order placed successfully",
            "execution_id": execution_result.execution_id,
            "order_id": execution_result.execution_id,  # Simplified
            "client_order_id": client_order_id,
            "status": execution_result.status.value,
            "filled_quantity": float(execution_result.total_filled_quantity),
            "average_fill_price": float(execution_result.average_fill_price)
            if execution_result.average_fill_price
            else None,
        }

    except Exception as e:
        logger.error(f"Order placement failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Order placement failed: {e!s}"
        )


@router.delete("/orders/{order_id}")
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
        if not execution_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Execution engine not available",
            )

        # Cancel order through execution engine
        success = await execution_engine.cancel_execution(order_id)

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
        logger.error(
            f"Order cancellation failed: {e}", order_id=order_id, user=current_user.username
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
        # Mock order data (in production, get from database/exchange)
        mock_orders = []

        # Generate some mock orders
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"] if not symbol else [symbol]
        exchanges = ["binance", "coinbase"] if not exchange else [exchange]
        statuses = ["filled", "cancelled", "partially_filled"] if not status else [status]

        for i in range(min(limit, 20)):  # Limit mock data
            order_symbol = symbols[i % len(symbols)]
            order_exchange = exchanges[i % len(exchanges)]
            order_status = statuses[i % len(statuses)]

            quantity = Decimal("1.0") + Decimal(str(i * 0.1))
            filled_qty = quantity if order_status == "filled" else quantity * Decimal("0.5")

            mock_order = OrderResponse(
                order_id=f"order_{i + 1:03d}",
                client_order_id=f"web_{uuid4().hex[:8]}",
                symbol=order_symbol,
                side="buy" if i % 2 == 0 else "sell",
                order_type="limit",
                quantity=quantity,
                price=Decimal("45000.00") + Decimal(str(i * 100)),
                stop_price=None,
                status=order_status,
                filled_quantity=filled_qty,
                remaining_quantity=quantity - filled_qty,
                average_fill_price=Decimal("45000.00") + Decimal(str(i * 50))
                if filled_qty > 0
                else None,
                commission=Decimal("0.1") if filled_qty > 0 else None,
                commission_asset="USDT",
                exchange=order_exchange,
                bot_id=f"bot_{(i % 3) + 1:03d}" if i % 4 != 0 else None,
                created_at=datetime.utcnow() - timedelta(hours=i),
                updated_at=datetime.utcnow() - timedelta(hours=i, minutes=30),
            )

            # Apply bot_id filter
            if bot_id and mock_order.bot_id != bot_id:
                continue

            mock_orders.append(mock_order)

        return mock_orders[:limit]

    except Exception as e:
        logger.error(f"Orders retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get orders: {e!s}"
        )


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
        # Mock order data (in production, get from database/exchange)
        mock_order = OrderResponse(
            order_id=order_id,
            client_order_id=f"web_{uuid4().hex[:8]}",
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("1.0"),
            price=Decimal("45000.00"),
            stop_price=None,
            status="filled",
            filled_quantity=Decimal("1.0"),
            remaining_quantity=Decimal("0.0"),
            average_fill_price=Decimal("44950.00"),
            commission=Decimal("0.1"),
            commission_asset="USDT",
            exchange=exchange,
            bot_id="bot_001",
            created_at=datetime.utcnow() - timedelta(hours=2),
            updated_at=datetime.utcnow() - timedelta(hours=1),
            fills=[
                {
                    "trade_id": "trade_001",
                    "quantity": "1.0",
                    "price": "44950.00",
                    "commission": "0.1",
                    "commission_asset": "USDT",
                    "executed_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                }
            ],
        )

        return mock_order

    except Exception as e:
        logger.error(f"Order retrieval failed: {e}", order_id=order_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get order: {e!s}"
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
        # Mock trade data (in production, get from database)
        mock_trades = []

        # Generate some mock trades
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"] if not symbol else [symbol]
        exchanges = ["binance", "coinbase"] if not exchange else [exchange]

        for i in range(min(limit, 30)):  # Limit mock data
            trade_symbol = symbols[i % len(symbols)]
            trade_exchange = exchanges[i % len(exchanges)]

            executed_at = datetime.utcnow() - timedelta(hours=i, minutes=i * 5)

            # Apply date filters
            if start_date and executed_at < start_date:
                continue
            if end_date and executed_at > end_date:
                continue

            quantity = Decimal("0.5") + Decimal(str(i * 0.1))
            price = Decimal("45000.00") + Decimal(str(i * 50))
            value = quantity * price

            mock_trade = TradeResponse(
                trade_id=f"trade_{i + 1:03d}",
                order_id=f"order_{i + 1:03d}",
                symbol=trade_symbol,
                side="buy" if i % 2 == 0 else "sell",
                quantity=quantity,
                price=price,
                value=value,
                commission=value * Decimal("0.001"),  # 0.1% commission
                commission_asset="USDT",
                exchange=trade_exchange,
                bot_id=f"bot_{(i % 3) + 1:03d}" if i % 4 != 0 else None,
                executed_at=executed_at,
            )

            # Apply bot_id filter
            if bot_id and mock_trade.bot_id != bot_id:
                continue

            mock_trades.append(mock_trade)

        return mock_trades[:limit]

    except Exception as e:
        logger.error(f"Trades retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get trades: {e!s}"
        )


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
        # Mock market data (in production, get from exchange API)
        base_price = Decimal("45000.00") if "BTC" in symbol else Decimal("3000.00")

        # Add some realistic variation
        import random

        price_variation = Decimal(str(random.uniform(-0.02, 0.02)))  # ±2%
        current_price = base_price * (Decimal("1") + price_variation)

        bid = current_price * Decimal("0.9999")
        ask = current_price * Decimal("1.0001")

        mock_data = MarketDataResponse(
            symbol=symbol,
            exchange=exchange,
            price=current_price,
            bid=bid,
            ask=ask,
            volume_24h=Decimal("1234567.89"),
            change_24h=current_price - base_price,
            change_24h_percentage=float((current_price - base_price) / base_price * 100),
            high_24h=current_price * Decimal("1.05"),
            low_24h=current_price * Decimal("0.95"),
            timestamp=datetime.utcnow(),
        )

        return mock_data

    except Exception as e:
        logger.error(
            f"Market data retrieval failed: {e}", symbol=symbol, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market data: {e!s}",
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
        # Mock order book data (in production, get from exchange)
        base_price = Decimal("45000.00") if "BTC" in symbol else Decimal("3000.00")

        # Generate mock bids (below current price)
        bids = []
        for i in range(depth):
            price = base_price * (Decimal("1") - Decimal(str(0.0001 * (i + 1))))
            quantity = Decimal(str(random.uniform(0.1, 5.0)))
            bids.append([price, quantity])

        # Generate mock asks (above current price)
        asks = []
        for i in range(depth):
            price = base_price * (Decimal("1") + Decimal(str(0.0001 * (i + 1))))
            quantity = Decimal(str(random.uniform(0.1, 5.0)))
            asks.append([price, quantity])

        return OrderBookResponse(
            symbol=symbol, exchange=exchange, bids=bids, asks=asks, timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Order book retrieval failed: {e}", symbol=symbol, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get order book: {e!s}",
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
        if not execution_engine:
            return {"status": "unavailable", "message": "Execution engine not available"}

        engine_summary = await execution_engine.get_engine_summary()
        return {"success": True, "status": engine_summary}

    except Exception as e:
        logger.error(f"Execution status retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution status: {e!s}",
        )
