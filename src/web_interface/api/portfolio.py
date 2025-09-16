"""
Portfolio API endpoints for T-Bot web interface.

This module provides portfolio tracking, analysis, and reporting functionality
including positions, balances, P&L, and performance metrics.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from src.core.logging import get_logger
from src.web_interface.di_registration import get_web_portfolio_service
from src.web_interface.security.auth import User, get_current_user, get_trading_user

if TYPE_CHECKING:
    from src.web_interface.services.portfolio_service import WebPortfolioService

logger = get_logger(__name__)
router = APIRouter()


def get_portfolio_service():
    """Get portfolio service through web service layer."""
    # Controllers should only use web services, not facades directly
    # The web service will handle facade interactions internally
    return get_web_portfolio_service_instance()


def get_web_portfolio_service_instance() -> "WebPortfolioService":
    """Get web portfolio service for business logic through DI."""
    return get_web_portfolio_service()


# Deprecated function for backward compatibility
def set_dependencies(orchestrator, engine):
    """DEPRECATED: Use service registry instead."""
    logger.warning("set_dependencies is deprecated. Use service registry instead.")


class PositionResponse(BaseModel):
    """Response model for position information."""

    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percentage: Decimal
    cost_basis: Decimal
    created_at: datetime
    updated_at: datetime
    bot_id: str | None = None


class BalanceResponse(BaseModel):
    """Response model for balance information."""

    exchange: str
    currency: str
    total_balance: Decimal
    available_balance: Decimal
    locked_balance: Decimal
    usd_value: Decimal | None = None
    updated_at: datetime


class PnLResponse(BaseModel):
    """Response model for P&L information."""

    period: str
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_return_percentage: Decimal
    number_of_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    sharpe_ratio: Decimal | None = None
    max_drawdown: Decimal
    max_drawdown_percentage: Decimal


class PortfolioSummaryResponse(BaseModel):
    """Response model for portfolio summary."""

    total_value: Decimal
    total_pnl: Decimal
    total_pnl_percentage: Decimal
    positions_count: int
    active_bots: int
    last_updated: datetime
    daily_pnl: PnLResponse
    weekly_pnl: PnLResponse
    monthly_pnl: PnLResponse


class AssetAllocationResponse(BaseModel):
    """Response model for asset allocation."""

    asset: str
    value: Decimal
    percentage: Decimal
    positions: int


@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(current_user: User = Depends(get_current_user)):
    """
    Get portfolio summary with key metrics.

    Args:
        current_user: Current authenticated user

    Returns:
        PortfolioSummaryResponse: Portfolio summary

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_portfolio_service = get_web_portfolio_service_instance()

        # Get processed data through service layer (business logic moved to service)
        summary_data = await web_portfolio_service.get_portfolio_summary_data()

        total_value = summary_data["total_value"]
        total_pnl = summary_data["total_pnl"]
        total_pnl_percentage = summary_data["total_pnl_percentage"]
        positions_count = summary_data["positions_count"]
        active_bots = summary_data["active_bots"]

        raw_data = summary_data["raw_data"]
        total_trades = raw_data.get("total_trades", 0)
        win_rate = raw_data.get("average_win_rate", 0.0)

        # Get P&L periods through service layer (business logic moved to service)
        pnl_periods = await web_portfolio_service.calculate_pnl_periods(
            total_pnl, total_trades, win_rate
        )

        # Convert to response models
        daily_pnl = PnLResponse(period="1d", **pnl_periods["daily"])
        weekly_pnl = PnLResponse(period="7d", **pnl_periods["weekly"])
        monthly_pnl = PnLResponse(period="30d", **pnl_periods["monthly"])

        return PortfolioSummaryResponse(
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_percentage,
            positions_count=positions_count,
            active_bots=active_bots,
            last_updated=datetime.now(timezone.utc),
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            monthly_pnl=monthly_pnl,
        )

    except Exception as e:
        logger.error(f"Portfolio summary retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get portfolio summary: {e!s}",
        )


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions(
    exchange: str | None = Query(None, description="Filter by exchange"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    current_user: User = Depends(get_current_user),
):
    """
    Get current positions across all bots and exchanges.

    Args:
        exchange: Optional exchange filter
        symbol: Optional symbol filter
        bot_id: Optional bot ID filter
        current_user: Current authenticated user

    Returns:
        List[PositionResponse]: List of current positions

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_portfolio_service = get_web_portfolio_service_instance()

        # Build filters from request parameters
        filters = {}
        if exchange:
            filters["exchange"] = exchange
        if symbol:
            filters["symbol"] = symbol
        if bot_id:
            filters["bot_id"] = bot_id

        # Get processed positions through service layer (business logic moved to service)
        processed_positions = await web_portfolio_service.get_processed_positions(filters)

        # Convert to response models
        positions = []
        for position_data in processed_positions:
            position_response = PositionResponse(**position_data)
            positions.append(position_response)

        return positions

    except Exception as e:
        logger.error(f"Positions retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get positions: {e!s}",
        )


@router.get("/balances", response_model=list[BalanceResponse])
async def get_balances(
    exchange: str | None = Query(None, description="Filter by exchange"),
    currency: str | None = Query(None, description="Filter by currency"),
    current_user: User = Depends(get_current_user),
):
    """
    Get account balances across all exchanges.

    Args:
        exchange: Optional exchange filter
        currency: Optional currency filter
        current_user: Current authenticated user

    Returns:
        List[BalanceResponse]: List of account balances

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_portfolio_service = get_web_portfolio_service_instance()

        # Build filters from request parameters
        filters = {}
        if exchange:
            filters["exchange"] = exchange
        if currency:
            filters["currency"] = currency

        # Get balance data through service layer (business logic moved to service)
        balance_data_list = web_portfolio_service.generate_mock_balances(filters)

        # Convert to response models
        balances = []
        for balance_data in balance_data_list:
            balance = BalanceResponse(
                exchange=balance_data["exchange"],
                currency=balance_data["currency"],
                total_balance=balance_data["total_balance"],
                available_balance=balance_data["available_balance"],
                locked_balance=balance_data["locked_balance"],
                usd_value=balance_data.get("usd_value"),
                updated_at=datetime.now(timezone.utc),
            )
            balances.append(balance)

        return balances

    except Exception as e:
        logger.error(f"Balances retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get balances: {e!s}",
        )


@router.get("/pnl", response_model=PnLResponse)
async def get_pnl(
    period: str = Query("30d", description="Period: 1d, 7d, 30d, 90d, 1y"),
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    current_user: User = Depends(get_current_user),
):
    """
    Get P&L analysis for specified period.

    Args:
        period: Analysis period
        bot_id: Optional bot ID filter
        current_user: Current authenticated user

    Returns:
        PnLResponse: P&L analysis

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_portfolio_service = get_web_portfolio_service_instance()

        # Get P&L metrics through service layer (business logic moved to service)
        pnl_metrics = await web_portfolio_service.calculate_pnl_metrics(period)

        return PnLResponse(**pnl_metrics)

    except Exception as e:
        logger.error(f"P&L retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get P&L: {e!s}"
        )


@router.get("/allocation", response_model=list[AssetAllocationResponse])
async def get_asset_allocation(current_user: User = Depends(get_current_user)):
    """
    Get asset allocation breakdown.

    Args:
        current_user: Current authenticated user

    Returns:
        List[AssetAllocationResponse]: Asset allocation breakdown

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_portfolio_service = get_web_portfolio_service_instance()

        # Get asset allocation through service layer (business logic moved to service)
        allocation_data = web_portfolio_service.calculate_asset_allocation()

        # Convert to response models
        allocations = []
        for alloc_data in allocation_data:
            allocation = AssetAllocationResponse(**alloc_data)
            allocations.append(allocation)

        return allocations

    except Exception as e:
        logger.error(f"Asset allocation retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get asset allocation: {e!s}",
        )


@router.get("/performance/chart")
async def get_performance_chart(
    period: str = Query("30d", description="Chart period: 1d, 7d, 30d, 90d, 1y"),
    resolution: str = Query("1h", description="Chart resolution: 5m, 15m, 1h, 4h, 1d"),
    current_user: User = Depends(get_current_user),
):
    """
    Get portfolio performance chart data.

    Args:
        period: Chart period
        resolution: Chart resolution
        current_user: Current authenticated user

    Returns:
        Dict: Chart data with timestamps and values

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_portfolio_service = get_web_portfolio_service_instance()

        # Get performance chart data through service layer (business logic moved to service)
        chart_data = web_portfolio_service.generate_performance_chart_data(period, resolution)

        return chart_data

    except Exception as e:
        logger.error(f"Performance chart retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance chart: {e!s}",
        )


@router.get("/history")
async def get_portfolio_history(
    period: str = Query("30d", description="History period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
):
    """Get portfolio history data."""
    try:
        from datetime import datetime, timedelta, timezone

        # Mock historical data for testing
        now = datetime.now(timezone.utc)
        days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(period, 30)

        history = []
        for i in range(days):
            date = now - timedelta(days=i)
            history.append(
                {
                    "date": date.isoformat(),
                    "total_value": 50000 + (i * 100),  # Mock growing value
                    "pnl": 1000 - (i * 10),
                    "positions": 5,
                }
            )

        return {
            "success": True,
            "period": period,
            "history": history[::-1],  # Reverse to get chronological order
            "total_points": len(history),
        }

    except Exception as e:
        logger.error(f"Portfolio history retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get portfolio history",
        )


@router.get("/performance")
async def get_portfolio_performance(current_user: User = Depends(get_current_user)):
    """Get portfolio performance metrics."""
    try:
        from decimal import Decimal

        return {
            "success": True,
            "performance": {
                "total_return": "15.25%",
                "annual_return": "12.80%",
                "sharpe_ratio": 1.85,
                "max_drawdown": "-8.50%",
                "volatility": "18.30%",
                "win_rate": "62.5%",
                "profit_factor": 1.45,
                "total_trades": 156,
                "winning_trades": 98,
                "losing_trades": 58,
                "average_win": str(Decimal("450.25")),
                "average_loss": str(Decimal("-285.75")),
                "best_trade": str(Decimal("1250.00")),
                "worst_trade": str(Decimal("-680.50")),
            },
            "benchmark_comparison": {
                "portfolio_return": "15.25%",
                "benchmark_return": "8.75%",
                "outperformance": "6.50%",
            },
        }

    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance metrics",
        )


@router.post("/rebalance")
async def rebalance_portfolio(
    request: dict[str, Any],
    trading_user: User = Depends(get_trading_user),
):
    """Rebalance portfolio according to target allocation."""
    try:
        target_allocation = request.get("allocations", {})
        rebalance_type = request.get("type", "gradual")  # gradual, immediate

        if not target_allocation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Target allocation required"
            )

        # Validate allocation adds to 1.0
        total_allocation = sum(Decimal(str(v)) for v in target_allocation.values())
        if abs(total_allocation - Decimal("1.0")) > Decimal("0.01"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Allocation must sum to 1.0 (100%)"
            )

        # Mock rebalancing result
        return {
            "success": True,
            "rebalance_id": f"rebal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "in_progress",
            "target_allocation": target_allocation,
            "rebalance_type": rebalance_type,
            "estimated_completion": "2-5 minutes",
            "trades_required": len(target_allocation),
            "message": f"Rebalancing to {len(target_allocation)} assets using {rebalance_type} strategy",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio rebalancing failed: {e}", user=trading_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start rebalancing"
        )


@router.get("/holdings")
async def get_portfolio_holdings(current_user: User = Depends(get_current_user)):
    """Get detailed portfolio holdings (alias for positions)."""
    return await get_positions(current_user)
