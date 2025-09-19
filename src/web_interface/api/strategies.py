"""
Strategy Management API endpoints for T-Bot web interface.

This module provides strategy configuration, deployment, and management
functionality for trading strategies.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.exceptions import ServiceError
from src.core.logging import get_logger
from src.web_interface.security.auth import User, get_admin_user, get_current_user

if TYPE_CHECKING:
    from src.web_interface.services.strategy_service import WebStrategyService

logger = get_logger(__name__)
router = APIRouter()


def get_strategy_service():
    """Get strategy service through web service layer."""
    # Controllers should only use web services, not facades directly
    return get_web_strategy_service_instance()


def get_web_strategy_service_instance() -> "WebStrategyService":
    """Get web strategy service for business logic through DI."""
    try:
        # Use proper dependency injection
        from src.web_interface.di_registration import get_web_interface_factory
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

        # Check if service is already registered
        if injector.has_service("WebStrategyService"):
            return injector.resolve("WebStrategyService")

        # If not registered, get it through factory
        factory = get_web_interface_factory(injector)
        strategy_service = factory.create_strategy_service()

        # Register for future use
        injector.register_service("WebStrategyService", strategy_service, singleton=True)

        return strategy_service
    except Exception as e:
        logger.error(f"Error getting web strategy service: {e}")
        raise ServiceError(f"Web strategy service not available: {e}")


# Deprecated function for backward compatibility
def set_dependencies(factory):
    """DEPRECATED: Use service registry instead."""
    logger.warning("set_dependencies is deprecated. Use service registry instead.")


class StrategyResponse(BaseModel):
    """Response model for strategy information."""

    strategy_name: str
    strategy_type: str
    description: str
    category: str
    supported_exchanges: list[str]
    supported_symbols: list[str]
    risk_level: str
    minimum_capital: Decimal
    recommended_timeframes: list[str]
    parameters: dict[str, Any]
    performance_metrics: dict[str, Any] | None = None
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None


class StrategyConfigRequest(BaseModel):
    """Request model for strategy configuration."""

    strategy_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk_settings: dict[str, Any] = Field(default_factory=dict)
    exchanges: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)


class StrategyPerformanceResponse(BaseModel):
    """Response model for strategy performance."""

    strategy_name: str
    period: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    total_pnl: Decimal
    total_return_percentage: Decimal
    average_trade_duration: Decimal | None = None
    max_drawdown: Decimal
    max_drawdown_percentage: Decimal
    sharpe_ratio: Decimal | None = None
    sortino_ratio: Decimal | None = None
    profit_factor: Decimal
    expectancy: Decimal
    largest_win: Decimal
    largest_loss: Decimal


class BacktestRequest(BaseModel):
    """Request model for strategy backtesting."""

    strategy_name: str
    symbols: list[str]
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Field(default=Decimal("100000"))
    parameters: dict[str, Any] = Field(default_factory=dict)
    benchmark: str | None = Field(default="BTCUSDT")


class BacktestResponse(BaseModel):
    """Response model for backtest results."""

    backtest_id: str
    strategy_name: str
    symbols: list[str]
    period: str
    initial_capital: Decimal
    final_capital: Decimal
    total_return: Decimal
    total_return_percentage: Decimal
    benchmark_return: Decimal | None = None
    alpha: Decimal | None = None
    beta: Decimal | None = None
    max_drawdown: Decimal
    sharpe_ratio: float
    trades_count: int
    win_rate: Decimal
    profit_factor: Decimal
    started_at: datetime
    completed_at: datetime | None = None
    status: str


@router.get("/", response_model=list[StrategyResponse])
async def list_strategies(
    category: str | None = Query(None, description="Filter by category"),
    risk_level: str | None = Query(None, description="Filter by risk level"),
    exchange: str | None = Query(None, description="Filter by supported exchange"),
    current_user: User = Depends(get_current_user),
):
    """
    List available trading strategies.

    Args:
        category: Optional category filter
        risk_level: Optional risk level filter
        exchange: Optional exchange filter
        current_user: Current authenticated user

    Returns:
        List[StrategyResponse]: List of available strategies

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        web_strategy_service = get_web_strategy_service_instance()

        # Get strategies through service layer - service handles facade calls and formatting
        strategies = await web_strategy_service.get_formatted_strategies()

        # Filter strategies based on parameters
        filtered_strategies = []
        for strategy_data in strategies:
            # Apply category filter
            if category and strategy_data.get("category") != category:
                continue

            # Apply risk level filter
            if risk_level and strategy_data.get("risk_level") != risk_level:
                continue

            # Apply exchange filter (check if strategy supports the exchange)
            if exchange and exchange not in strategy_data.get("supported_exchanges", []):
                continue

            # Convert to response format
            strategy_response = StrategyResponse(
                strategy_name=strategy_data.get("name", ""),
                strategy_type=strategy_data.get("display_name", ""),
                description=strategy_data.get("description", ""),
                category=strategy_data.get("category", "general"),
                supported_exchanges=strategy_data.get("supported_exchanges", ["binance"]),
                supported_symbols=strategy_data.get("supported_symbols", ["BTCUSDT"]),
                risk_level=strategy_data.get("risk_level", "medium"),
                minimum_capital=Decimal(str(strategy_data.get("minimum_capital", 10000))),
                recommended_timeframes=strategy_data.get("recommended_timeframes", ["1h"]),
                parameters=strategy_data.get("parameters", {}),
                performance_metrics=strategy_data.get("performance_metrics"),
                is_active=strategy_data.get("is_active", True),
                created_at=strategy_data.get("created_at"),
                updated_at=strategy_data.get("updated_at"),
            )
            filtered_strategies.append(strategy_response)

        return filtered_strategies

    except Exception as e:
        logger.error(f"Strategy listing failed: {e}", user=current_user.username)
        # Service layer already handles fallbacks, so just re-raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list strategies: {e!s}",
        )


@router.get("/{strategy_name}", response_model=StrategyResponse)
async def get_strategy(strategy_name: str, current_user: User = Depends(get_current_user)):
    """
    Get detailed information about a specific strategy.

    Args:
        strategy_name: Strategy name
        current_user: Current authenticated user

    Returns:
        StrategyResponse: Strategy details

    Raises:
        HTTPException: If strategy not found
    """
    try:
        web_strategy_service = get_web_strategy_service_instance()

        # Get strategy configuration through service layer
        strategy_config = await web_strategy_service.get_strategy_config_through_service(
            strategy_name
        )

        if strategy_config:
            # Convert service response to API response format
            strategy_response = StrategyResponse(
                strategy_name=strategy_name,
                strategy_type=strategy_config.get("display_name", strategy_name),
                description=strategy_config.get("description", f"Strategy: {strategy_name}"),
                category=strategy_config.get("category", "general"),
                supported_exchanges=strategy_config.get("supported_exchanges", ["binance"]),
                supported_symbols=strategy_config.get("supported_symbols", ["BTCUSDT"]),
                risk_level=strategy_config.get("risk_level", "medium"),
                minimum_capital=Decimal(str(strategy_config.get("minimum_capital", 10000))),
                recommended_timeframes=strategy_config.get("recommended_timeframes", ["1h"]),
                parameters=strategy_config.get("parameters", {}),
                performance_metrics=strategy_config.get("performance_metrics"),
                is_active=strategy_config.get("is_active", True),
                created_at=strategy_config.get("created_at"),
                updated_at=strategy_config.get("updated_at"),
            )
            return strategy_response

        # Fallback to mock data for known strategies
        elif strategy_name == "trend_following":
            strategy_data = {
                "strategy_name": "trend_following",
                "strategy_type": "momentum",
                "description": "Advanced trend following strategy using multiple timeframe analysis",
                "category": "trend",
                "supported_exchanges": ["binance", "coinbase", "okx"],
                "supported_symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT"],
                "risk_level": "medium",
                "minimum_capital": Decimal("10000"),
                "recommended_timeframes": ["15m", "1h", "4h", "1d"],
                "parameters": {
                    "fast_ma_period": 12,
                    "slow_ma_period": 26,
                    "signal_period": 9,
                    "risk_per_trade": 0.02,
                    "max_positions": 3,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.15,
                    "trailing_stop": True,
                    "volatility_filter": True,
                    "volume_filter": True,
                },
                "performance_metrics": {
                    "30d_return": 15.2,
                    "90d_return": 45.8,
                    "1y_return": 187.5,
                    "win_rate": 0.65,
                    "profit_factor": 2.15,
                    "sharpe_ratio": 1.45,
                    "max_drawdown": -12.3,
                    "avg_trade_duration": 4.2,
                    "total_trades": 156,
                    "largest_win": 850.0,
                    "largest_loss": -425.0,
                },
                "is_active": True,
                "created_at": datetime.now(timezone.utc) - timedelta(days=30),
                "updated_at": datetime.now(timezone.utc) - timedelta(days=1),
            }
            return StrategyResponse(**strategy_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy not found: {strategy_name}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Strategy retrieval failed: {e}", strategy=strategy_name, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get strategy: {e!s}",
        )


@router.post("/{strategy_name}/configure")
async def configure_strategy(
    strategy_name: str,
    config_request: StrategyConfigRequest,
    current_user: User = Depends(get_admin_user),
):
    """
    Configure strategy parameters (admin only).

    Args:
        strategy_name: Strategy name
        config_request: Configuration parameters
        current_user: Current admin user

    Returns:
        Dict: Configuration result

    Raises:
        HTTPException: If configuration fails
    """
    try:
        web_strategy_service = get_web_strategy_service_instance()

        # Validate strategy configuration through service layer
        is_valid = await web_strategy_service.validate_strategy_config_through_service(
            strategy_name, config_request.parameters
        )

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration for strategy {strategy_name}",
            )

        # Note: Actual configuration update would need to be implemented in the service layer
        logger.info(
            "Strategy configured",
            strategy=strategy_name,
            parameters=config_request.parameters,
            admin=current_user.username,
        )

        return {
            "success": True,
            "message": f"Strategy {strategy_name} configured successfully",
            "strategy_name": strategy_name,
            "updated_parameters": config_request.parameters,
            "updated_by": current_user.username,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(
            f"Strategy configuration failed: {e}",
            strategy=strategy_name,
            user=current_user.username,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Strategy configuration failed: {e!s}"
        )


@router.get("/{strategy_name}/performance", response_model=StrategyPerformanceResponse)
async def get_strategy_performance(
    strategy_name: str,
    period: str = Query("30d", description="Performance period: 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
):
    """
    Get strategy performance metrics.

    Args:
        strategy_name: Strategy name
        period: Performance period
        current_user: Current authenticated user

    Returns:
        StrategyPerformanceResponse: Performance metrics

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Mock performance data (in production, calculate from trade history)
        period_multipliers = {"7d": 0.25, "30d": 1.0, "90d": 3.0, "1y": 12.0}

        multiplier = period_multipliers.get(period, 1.0)
        base_trades = 50
        base_pnl = Decimal("5000")

        total_trades = int(base_trades * multiplier)
        winning_trades = int(total_trades * 0.65)
        losing_trades = total_trades - winning_trades

        performance = StrategyPerformanceResponse(
            strategy_name=strategy_name,
            period=period,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=0.65,
            total_pnl=base_pnl * Decimal(str(multiplier)),
            total_return_percentage=float(5.0 * multiplier),
            average_trade_duration=4.2,
            max_drawdown=base_pnl * Decimal("-0.15") * Decimal(str(multiplier)),
            max_drawdown_percentage=-7.5,
            sharpe_ratio=1.45,
            sortino_ratio=1.89,
            profit_factor=2.15,
            expectancy=base_pnl / total_trades if total_trades > 0 else Decimal("0"),
            largest_win=Decimal("850.00") * Decimal(str(multiplier)),
            largest_loss=Decimal("-425.00") * Decimal(str(multiplier)),
        )

        return performance

    except Exception as e:
        logger.error(
            f"Strategy performance retrieval failed: {e}",
            strategy=strategy_name,
            user=current_user.username,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get strategy performance: {e!s}",
        )


@router.post("/{strategy_name}/backtest", response_model=BacktestResponse)
async def start_backtest(
    strategy_name: str,
    backtest_request: BacktestRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Start a strategy backtest.

    Args:
        strategy_name: Strategy name
        backtest_request: Backtest parameters
        current_user: Current authenticated user

    Returns:
        BacktestResponse: Backtest result

    Raises:
        HTTPException: If backtest fails
    """
    try:
        import uuid

        # Mock backtest execution (in production, queue backtest job)
        backtest_id = f"bt_{uuid.uuid4().hex[:8]}"

        # Calculate mock results
        days = (backtest_request.end_date - backtest_request.start_date).days
        period_str = f"{days}d"

        # Mock performance calculation
        daily_return = 0.0015  # 0.15% daily return
        total_return_pct = daily_return * days * 100
        final_capital = backtest_request.initial_capital * (
            Decimal("1") + Decimal(str(total_return_pct / 100))
        )
        total_return = final_capital - backtest_request.initial_capital

        # Mock benchmark comparison
        benchmark_return = total_return * Decimal("0.8")  # Strategy outperforms benchmark

        trades_count = max(1, days // 3)  # Roughly one trade every 3 days

        backtest_result = BacktestResponse(
            backtest_id=backtest_id,
            strategy_name=strategy_name,
            symbols=backtest_request.symbols,
            period=period_str,
            initial_capital=backtest_request.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_percentage=float(total_return_pct),
            benchmark_return=benchmark_return,
            alpha=float(total_return_pct * 0.2),  # Alpha vs benchmark
            beta=0.85,  # Beta vs benchmark
            max_drawdown=total_return * Decimal("-0.12"),  # 12% max drawdown
            sharpe_ratio=1.65,
            trades_count=trades_count,
            win_rate=0.68,
            profit_factor=2.25,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc) + timedelta(seconds=30),  # Mock completion
            status="completed",
        )

        logger.info(
            "Backtest completed",
            backtest_id=backtest_id,
            strategy=strategy_name,
            return_pct=total_return_pct,
            user=current_user.username,
        )

        return backtest_result

    except Exception as e:
        logger.error(f"Backtest failed: {e}", strategy=strategy_name, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Backtest failed: {e!s}"
        )


@router.get("/categories/list")
async def list_strategy_categories(current_user: User = Depends(get_current_user)):
    """
    List available strategy categories.

    Args:
        current_user: Current authenticated user

    Returns:
        Dict: List of strategy categories
    """
    try:
        categories = {
            "trend": {
                "name": "Trend Following",
                "description": "Strategies that follow market trends and momentum",
                "risk_level": "medium",
                "strategies": ["trend_following", "momentum_breakout"],
            },
            "reversion": {
                "name": "Mean Reversion",
                "description": "Strategies that trade against temporary price movements",
                "risk_level": "high",
                "strategies": ["mean_reversion", "bollinger_bounce"],
            },
            "arbitrage": {
                "name": "Arbitrage",
                "description": "Strategies that exploit price differences",
                "risk_level": "low",
                "strategies": ["arbitrage_scanner", "triangular_arbitrage"],
            },
            "liquidity": {
                "name": "Market Making",
                "description": "Strategies that provide liquidity to markets",
                "risk_level": "medium",
                "strategies": ["market_making", "spread_capture"],
            },
            "ml": {
                "name": "Machine Learning",
                "description": "AI-powered trading strategies",
                "risk_level": "variable",
                "strategies": ["price_predictor", "regime_detector"],
            },
        }

        return {
            "categories": categories,
            "total_categories": len(categories),
            "total_strategies": sum(len(cat["strategies"]) for cat in categories.values()),
        }

    except Exception as e:
        logger.error(f"Category listing failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list categories: {e!s}",
        )


@router.get("/{strategy_name}/optimize")
async def optimize_strategy(
    strategy_name: str,
    symbols: list[str] = Query(..., description="Symbols to optimize for"),
    start_date: datetime = Query(..., description="Optimization start date"),
    end_date: datetime = Query(..., description="Optimization end date"),
    current_user: User = Depends(get_current_user),
):
    """
    Optimize strategy parameters.

    Args:
        strategy_name: Strategy name
        symbols: Symbols to optimize for
        start_date: Optimization start date
        end_date: Optimization end date
        current_user: Current authenticated user

    Returns:
        Dict: Optimization results
    """
    try:
        # Mock optimization results (in production, run parameter optimization)
        optimization_results = {
            "strategy_name": strategy_name,
            "symbols": symbols,
            "optimization_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days,
            },
            "optimized_parameters": {
                "fast_ma_period": 10,  # Optimized from 12
                "slow_ma_period": 28,  # Optimized from 26
                "signal_period": 8,  # Optimized from 9
                "risk_per_trade": 0.025,  # Optimized from 0.02
                "stop_loss_pct": 0.045,  # Optimized from 0.05
            },
            "original_performance": {
                "total_return": 15.2,
                "sharpe_ratio": 1.45,
                "max_drawdown": -12.3,
                "win_rate": 0.65,
            },
            "optimized_performance": {
                "total_return": 22.8,
                "sharpe_ratio": 1.78,
                "max_drawdown": -9.5,
                "win_rate": 0.71,
            },
            "improvement": {
                "return_improvement": 7.6,
                "sharpe_improvement": 0.33,
                "drawdown_improvement": 2.8,
                "win_rate_improvement": 0.06,
            },
            "confidence_score": 0.87,
            "optimization_method": "genetic_algorithm",
            "iterations": 1000,
            "optimization_time_seconds": 45.2,
        }

        logger.info(
            "Strategy optimization completed",
            strategy=strategy_name,
            return_improvement=optimization_results["improvement"]["return_improvement"],
            user=current_user.username,
        )

        return optimization_results

    except Exception as e:
        logger.error(
            f"Strategy optimization failed: {e}", strategy=strategy_name, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy optimization failed: {e!s}",
        )
