"""
Portfolio API endpoints for T-Bot web interface.

This module provides portfolio tracking, analysis, and reporting functionality
including positions, balances, P&L, and performance metrics.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from src.core.logging import get_logger
from src.web_interface.security.auth import User, get_current_user

logger = get_logger(__name__)
router = APIRouter()

# Global references (set by app startup)
bot_orchestrator = None
execution_engine = None


def set_dependencies(orchestrator, engine):
    """Set global dependencies."""
    global bot_orchestrator, execution_engine
    bot_orchestrator = orchestrator
    execution_engine = engine


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
    unrealized_pnl_percentage: float
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
    total_return_percentage: float
    number_of_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: Decimal
    average_loss: Decimal
    profit_factor: float
    sharpe_ratio: float | None = None
    max_drawdown: Decimal
    max_drawdown_percentage: float


class PortfolioSummaryResponse(BaseModel):
    """Response model for portfolio summary."""

    total_value: Decimal
    total_pnl: Decimal
    total_pnl_percentage: float
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
    percentage: float
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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        # Get orchestrator status for portfolio metrics
        orchestrator_status = await bot_orchestrator.get_orchestrator_status()
        global_metrics = orchestrator_status.get("global_metrics", {})

        # Calculate portfolio values
        total_value = global_metrics.get("total_allocated_capital", Decimal("0"))
        total_pnl = global_metrics.get("total_pnl", Decimal("0"))
        total_pnl_percentage = float(total_pnl / total_value * 100) if total_value > 0 else 0.0

        # Get bot information
        bot_summaries = orchestrator_status.get("bots", {})
        positions_count = sum(
            1
            for bot in bot_summaries.values()
            if bot.get("metrics", {}).get("active_positions", 0) > 0
        )
        active_bots = global_metrics.get("running_bots", 0)

        # Mock P&L periods (in production, calculate from actual data)
        daily_pnl = PnLResponse(
            period="1d",
            total_pnl=total_pnl * Decimal("0.1"),  # Mock daily portion
            realized_pnl=total_pnl * Decimal("0.08"),
            unrealized_pnl=total_pnl * Decimal("0.02"),
            total_return_percentage=total_pnl_percentage * 0.1,
            number_of_trades=global_metrics.get("total_trades", 0) // 7,
            winning_trades=int(global_metrics.get("total_trades", 0) * 0.6 // 7),
            losing_trades=int(global_metrics.get("total_trades", 0) * 0.4 // 7),
            win_rate=global_metrics.get("average_win_rate", 0.0),
            average_win=Decimal("150.00"),
            average_loss=Decimal("-75.00"),
            profit_factor=2.0,
            sharpe_ratio=1.2,
            max_drawdown=Decimal("-500.00"),
            max_drawdown_percentage=-2.5,
        )

        weekly_pnl = PnLResponse(
            period="7d",
            total_pnl=total_pnl * Decimal("0.3"),
            realized_pnl=total_pnl * Decimal("0.25"),
            unrealized_pnl=total_pnl * Decimal("0.05"),
            total_return_percentage=total_pnl_percentage * 0.3,
            number_of_trades=global_metrics.get("total_trades", 0) // 4,
            winning_trades=int(global_metrics.get("total_trades", 0) * 0.6 // 4),
            losing_trades=int(global_metrics.get("total_trades", 0) * 0.4 // 4),
            win_rate=global_metrics.get("average_win_rate", 0.0),
            average_win=Decimal("180.00"),
            average_loss=Decimal("-90.00"),
            profit_factor=2.2,
            sharpe_ratio=1.35,
            max_drawdown=Decimal("-1200.00"),
            max_drawdown_percentage=-4.2,
        )

        monthly_pnl = PnLResponse(
            period="30d",
            total_pnl=total_pnl,
            realized_pnl=total_pnl * Decimal("0.8"),
            unrealized_pnl=total_pnl * Decimal("0.2"),
            total_return_percentage=total_pnl_percentage,
            number_of_trades=global_metrics.get("total_trades", 0),
            winning_trades=int(global_metrics.get("total_trades", 0) * 0.6),
            losing_trades=int(global_metrics.get("total_trades", 0) * 0.4),
            win_rate=global_metrics.get("average_win_rate", 0.0),
            average_win=Decimal("200.00"),
            average_loss=Decimal("-100.00"),
            profit_factor=2.1,
            sharpe_ratio=1.45,
            max_drawdown=Decimal("-2500.00"),
            max_drawdown_percentage=-8.3,
        )

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
        positions = []

        if not bot_orchestrator:
            return positions

        # Get positions from all bots
        orchestrator_status = await bot_orchestrator.get_orchestrator_status()
        bot_summaries = orchestrator_status.get("bots", {})

        for current_bot_id, bot_data in bot_summaries.items():
            # Apply bot filter
            if bot_id and current_bot_id != bot_id:
                continue

            # Mock position data (in production, get from actual bot positions)
            bot_positions = bot_data.get("positions", [])

            # Mock some positions for demonstration
            if not bot_positions and current_bot_id in ["bot_001", "bot_002"]:
                mock_positions = [
                    {
                        "symbol": "BTCUSDT",
                        "exchange": "binance",
                        "side": "long",
                        "quantity": Decimal("0.1"),
                        "entry_price": Decimal("45000.00"),
                        "current_price": Decimal("47000.00"),
                        "created_at": datetime.now(timezone.utc) - timedelta(hours=2),
                    },
                    {
                        "symbol": "ETHUSDT",
                        "exchange": "binance",
                        "side": "long",
                        "quantity": Decimal("2.5"),
                        "entry_price": Decimal("3000.00"),
                        "current_price": Decimal("3100.00"),
                        "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
                    },
                ]
                bot_positions = mock_positions

            for position in bot_positions:
                # Apply filters
                if exchange and position.get("exchange") != exchange:
                    continue
                if symbol and position.get("symbol") != symbol:
                    continue

                # Calculate position metrics
                quantity = position.get("quantity", Decimal("0"))
                entry_price = position.get("entry_price", Decimal("0"))
                current_price = position.get("current_price", Decimal("0"))

                market_value = quantity * current_price
                cost_basis = quantity * entry_price
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_percentage = (
                    float(unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
                )

                position_response = PositionResponse(
                    symbol=position.get("symbol", ""),
                    exchange=position.get("exchange", ""),
                    side=position.get("side", "long"),
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percentage=unrealized_pnl_percentage,
                    cost_basis=cost_basis,
                    created_at=position.get("created_at", datetime.now(timezone.utc)),
                    updated_at=datetime.now(timezone.utc),
                    bot_id=current_bot_id,
                )
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
        # Mock balance data (in production, get from actual exchange balances)
        mock_balances = [
            {
                "exchange": "binance",
                "currency": "USDT",
                "total_balance": Decimal("50000.00"),
                "available_balance": Decimal("45000.00"),
                "locked_balance": Decimal("5000.00"),
                "usd_value": Decimal("50000.00"),
            },
            {
                "exchange": "binance",
                "currency": "BTC",
                "total_balance": Decimal("1.5"),
                "available_balance": Decimal("1.3"),
                "locked_balance": Decimal("0.2"),
                "usd_value": Decimal("70500.00"),
            },
            {
                "exchange": "binance",
                "currency": "ETH",
                "total_balance": Decimal("10.0"),
                "available_balance": Decimal("8.5"),
                "locked_balance": Decimal("1.5"),
                "usd_value": Decimal("31000.00"),
            },
            {
                "exchange": "coinbase",
                "currency": "USD",
                "total_balance": Decimal("25000.00"),
                "available_balance": Decimal("23000.00"),
                "locked_balance": Decimal("2000.00"),
                "usd_value": Decimal("25000.00"),
            },
        ]

        balances = []
        for balance_data in mock_balances:
            # Apply filters
            if exchange and balance_data["exchange"] != exchange:
                continue
            if currency and balance_data["currency"] != currency:
                continue

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
        if not bot_orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Bot orchestrator not available",
            )

        # Get portfolio metrics
        orchestrator_status = await bot_orchestrator.get_orchestrator_status()
        global_metrics = orchestrator_status.get("global_metrics", {})

        # Calculate period-specific metrics (mock implementation)
        total_pnl = global_metrics.get("total_pnl", Decimal("0"))
        total_trades = global_metrics.get("total_trades", 0)
        win_rate = global_metrics.get("average_win_rate", 0.0)

        # Adjust metrics based on period
        period_multipliers = {"1d": 0.1, "7d": 0.3, "30d": 1.0, "90d": 3.0, "1y": 12.0}

        multiplier = period_multipliers.get(period, 1.0)
        period_pnl = total_pnl * Decimal(str(multiplier))
        period_trades = int(total_trades * multiplier)

        # Calculate metrics
        winning_trades = int(period_trades * win_rate)
        losing_trades = period_trades - winning_trades

        # Mock additional metrics
        realized_pnl = period_pnl * Decimal("0.8")
        unrealized_pnl = period_pnl * Decimal("0.2")

        allocated_capital = global_metrics.get("total_allocated_capital", Decimal("100000"))
        total_return_percentage = (
            float(period_pnl / allocated_capital * 100) if allocated_capital > 0 else 0.0
        )

        average_win = Decimal("200.00") * Decimal(str(multiplier))
        average_loss = Decimal("-100.00") * Decimal(str(multiplier))
        profit_factor = abs(average_win / average_loss) if average_loss != 0 else 0.0

        max_drawdown = period_pnl * Decimal("-0.15")  # 15% of gains as max drawdown
        max_drawdown_percentage = (
            float(max_drawdown / allocated_capital * 100) if allocated_capital > 0 else 0.0
        )

        return PnLResponse(
            period=period,
            total_pnl=period_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_return_percentage=total_return_percentage,
            number_of_trades=period_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=float(profit_factor),
            sharpe_ratio=1.2 + (multiplier * 0.1),  # Mock Sharpe ratio
            max_drawdown=max_drawdown,
            max_drawdown_percentage=max_drawdown_percentage,
        )

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
        # Mock asset allocation (in production, calculate from actual positions)
        total_value = Decimal("176500.00")  # Sum of all asset values

        allocations = [
            AssetAllocationResponse(
                asset="BTC",
                value=Decimal("70500.00"),
                percentage=float(Decimal("70500.00") / total_value * 100),
                positions=2,
            ),
            AssetAllocationResponse(
                asset="ETH",
                value=Decimal("31000.00"),
                percentage=float(Decimal("31000.00") / total_value * 100),
                positions=3,
            ),
            AssetAllocationResponse(
                asset="USDT",
                value=Decimal("50000.00"),
                percentage=float(Decimal("50000.00") / total_value * 100),
                positions=1,
            ),
            AssetAllocationResponse(
                asset="USD",
                value=Decimal("25000.00"),
                percentage=float(Decimal("25000.00") / total_value * 100),
                positions=1,
            ),
        ]

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
        # Mock chart data (in production, get from time series database)
        import random
        from datetime import datetime, timedelta, timezone

        # Calculate number of data points
        resolution_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

        period_hours = {"1d": 24, "7d": 168, "30d": 720, "90d": 2160, "1y": 8760}

        minutes_per_point = resolution_minutes.get(resolution, 60)
        hours_in_period = period_hours.get(period, 720)
        points = hours_in_period * 60 // minutes_per_point

        # Generate mock data
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours_in_period)
        start_value = 100000.0  # Starting portfolio value

        data_points = []
        current_value = start_value

        for i in range(points):
            timestamp = start_time + timedelta(minutes=i * minutes_per_point)

            # Simulate portfolio value changes
            change_percent = random.uniform(-0.5, 0.7)  # Slight upward bias
            current_value *= 1 + change_percent / 100

            data_points.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "portfolio_value": round(current_value, 2),
                    "pnl": round(current_value - start_value, 2),
                    "pnl_percentage": round((current_value - start_value) / start_value * 100, 4),
                }
            )

        return {
            "period": period,
            "resolution": resolution,
            "data_points": len(data_points),
            "start_value": start_value,
            "end_value": current_value,
            "total_return": round(current_value - start_value, 2),
            "total_return_percentage": round((current_value - start_value) / start_value * 100, 4),
            "data": data_points,
        }

    except Exception as e:
        logger.error(f"Performance chart retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance chart: {e!s}",
        )
