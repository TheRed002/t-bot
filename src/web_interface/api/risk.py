"""
Risk Management API endpoints for T-Bot web interface.

This module provides risk monitoring, limits management, and risk analysis
functionality for the trading system.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.web_interface.security.auth import User, get_admin_user, get_current_user

logger = get_logger(__name__)
router = APIRouter()

# Global references (set by app startup)
risk_manager = None


def set_dependencies(manager):
    """Set global dependencies."""
    global risk_manager
    risk_manager = manager


class RiskMetricsResponse(BaseModel):
    """Response model for risk metrics."""

    portfolio_value: Decimal
    total_exposure: Decimal
    leverage: Decimal
    var_1d: Decimal  # Value at Risk 1 day
    var_5d: Decimal  # Value at Risk 5 days
    expected_shortfall: Decimal
    max_drawdown: Decimal
    max_drawdown_percentage: Decimal
    volatility: Decimal
    beta: Decimal | None = None
    correlation_btc: Decimal | None = None
    risk_score: Decimal
    risk_level: str
    last_updated: datetime


class RiskLimitsResponse(BaseModel):
    """Response model for risk limits."""

    max_portfolio_risk: Decimal
    max_position_size: Decimal
    max_leverage: Decimal
    max_daily_loss: Decimal
    max_drawdown_limit: Decimal
    concentration_limit: Decimal
    correlation_limit: Decimal
    var_limit: Decimal
    stop_loss_required: bool
    position_sizing_method: str
    risk_per_trade: Decimal


class UpdateRiskLimitsRequest(BaseModel):
    """Request model for updating risk limits."""

    max_portfolio_risk: Decimal | None = Field(None, ge=0, le=1)
    max_position_size: Decimal | None = Field(None, gt=0)
    max_leverage: Decimal | None = Field(None, ge=1, le=10)
    max_daily_loss: Decimal | None = Field(None, gt=0)
    max_drawdown_limit: Decimal | None = Field(None, ge=0, le=1)
    concentration_limit: Decimal | None = Field(None, ge=0, le=1)
    var_limit: Decimal | None = Field(None, gt=0)
    risk_per_trade: Decimal | None = Field(None, ge=0, le=0.1)


class RiskAlertResponse(BaseModel):
    """Response model for risk alerts."""

    alert_id: str
    alert_type: str
    severity: str
    message: str
    triggered_at: datetime
    current_value: Decimal | None = None
    threshold_value: Decimal | None = None
    affected_positions: list[str] | None = None
    recommended_action: str
    is_resolved: bool
    resolved_at: datetime | None = None


class PositionRiskResponse(BaseModel):
    """Response model for individual position risk."""

    position_id: str
    symbol: str
    exchange: str
    side: str
    quantity: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    risk_amount: Decimal
    risk_percentage: Decimal
    var_contribution: Decimal
    beta: Decimal | None = None
    correlation: Decimal | None = None
    concentration_risk: Decimal
    liquidity_risk: str
    time_decay_risk: Decimal | None = None


class StressTestRequest(BaseModel):
    """Request model for stress testing."""

    test_name: str
    scenarios: list[dict[str, Any]]
    confidence_levels: list[Decimal] = Field(default=[Decimal("0.95"), Decimal("0.99")])
    time_horizons: list[int] = Field(default=[1, 5, 10])  # days


class StressTestResponse(BaseModel):
    """Response model for stress test results."""

    test_id: str
    test_name: str
    scenarios_tested: int
    worst_case_loss: Decimal
    worst_case_scenario: str
    confidence_levels: dict[float, Decimal]
    time_horizons: dict[int, Decimal]
    portfolio_resilience_score: Decimal
    recommendations: list[str]
    completed_at: datetime


@router.get("/metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics(current_user: User = Depends(get_current_user)):
    """
    Get current portfolio risk metrics.

    Args:
        current_user: Current authenticated user

    Returns:
        RiskMetricsResponse: Current risk metrics

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Mock risk metrics (in production, calculate from actual portfolio)
        portfolio_value = Decimal("150000.00")
        total_exposure = Decimal("135000.00")
        leverage = total_exposure / portfolio_value

        # Mock VaR calculations
        var_1d = portfolio_value * Decimal("0.025")  # 2.5% VaR
        var_5d = var_1d * Decimal("2.236")  # sqrt(5) scaling

        max_drawdown = Decimal("-12500.00")
        max_drawdown_pct = abs(max_drawdown) / portfolio_value * 100

        # Risk scoring (0-100, where 100 is highest risk)
        risk_score = min(Decimal("100"), leverage * 20 + max_drawdown_pct * 2)

        if risk_score < 30:
            risk_level = "low"
        elif risk_score < 60:
            risk_level = "medium"
        elif risk_score < 80:
            risk_level = "high"
        else:
            risk_level = "critical"

        return RiskMetricsResponse(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            leverage=leverage,
            var_1d=var_1d,
            var_5d=var_5d,
            expected_shortfall=var_1d * Decimal("1.3"),  # ES typically 1.3x VaR
            max_drawdown=max_drawdown,
            max_drawdown_percentage=max_drawdown_pct,
            volatility=0.15,  # 15% annualized volatility
            beta=1.2,  # Beta vs BTC
            correlation_btc=0.85,  # Correlation with BTC
            risk_score=risk_score,
            risk_level=risk_level,
            last_updated=datetime.now(timezone.utc),
        )

    except Exception as e:
        logger.error(f"Risk metrics retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get risk metrics: {e!s}",
        )


@router.get("/limits", response_model=RiskLimitsResponse)
async def get_risk_limits(current_user: User = Depends(get_current_user)):
    """
    Get current risk limits configuration.

    Args:
        current_user: Current authenticated user

    Returns:
        RiskLimitsResponse: Current risk limits

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Mock risk limits (in production, get from configuration/database)
        return RiskLimitsResponse(
            max_portfolio_risk=0.20,  # 20% max portfolio risk
            max_position_size=Decimal("25000.00"),  # $25k max position
            max_leverage=3.0,  # 3x max leverage
            max_daily_loss=Decimal("5000.00"),  # $5k max daily loss
            max_drawdown_limit=0.15,  # 15% max drawdown
            concentration_limit=0.30,  # 30% max in single asset
            correlation_limit=0.70,  # 70% max correlation
            var_limit=Decimal("7500.00"),  # $7.5k VaR limit
            stop_loss_required=True,
            position_sizing_method="kelly_criterion",
            risk_per_trade=0.02,  # 2% risk per trade
        )

    except Exception as e:
        logger.error(f"Risk limits retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get risk limits: {e!s}",
        )


@router.put("/limits")
async def update_risk_limits(
    limits_request: UpdateRiskLimitsRequest, current_user: User = Depends(get_admin_user)
):
    """
    Update risk limits configuration (admin only).

    Args:
        limits_request: Updated risk limits
        current_user: Current admin user

    Returns:
        Dict: Update result

    Raises:
        HTTPException: If update fails
    """
    try:
        # Mock risk limits update (in production, update configuration/database)
        updated_fields = {}

        if limits_request.max_portfolio_risk is not None:
            updated_fields["max_portfolio_risk"] = limits_request.max_portfolio_risk

        if limits_request.max_position_size is not None:
            updated_fields["max_position_size"] = limits_request.max_position_size

        if limits_request.max_leverage is not None:
            updated_fields["max_leverage"] = limits_request.max_leverage

        if limits_request.max_daily_loss is not None:
            updated_fields["max_daily_loss"] = limits_request.max_daily_loss

        if limits_request.max_drawdown_limit is not None:
            updated_fields["max_drawdown_limit"] = limits_request.max_drawdown_limit

        if limits_request.risk_per_trade is not None:
            updated_fields["risk_per_trade"] = limits_request.risk_per_trade

        logger.info(
            "Risk limits updated", updated_fields=updated_fields, admin=current_user.username
        )

        return {
            "success": True,
            "message": "Risk limits updated successfully",
            "updated_fields": updated_fields,
            "updated_by": current_user.username,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Risk limits update failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Risk limits update failed: {e!s}"
        )


@router.get("/alerts", response_model=list[RiskAlertResponse])
async def get_risk_alerts(
    severity: str | None = Query(None, description="Filter by severity"),
    unresolved_only: bool = Query(True, description="Show only unresolved alerts"),
    limit: int = Query(50, ge=1, le=200, description="Number of alerts to return"),
    current_user: User = Depends(get_current_user),
):
    """
    Get risk alerts with optional filtering.

    Args:
        severity: Optional severity filter
        unresolved_only: Show only unresolved alerts
        limit: Maximum number of alerts to return
        current_user: Current authenticated user

    Returns:
        List[RiskAlertResponse]: List of risk alerts

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Mock risk alerts (in production, get from alerting system)
        mock_alerts = [
            {
                "alert_id": "alert_001",
                "alert_type": "drawdown_limit",
                "severity": "high",
                "message": "Portfolio drawdown exceeded 12% threshold",
                "triggered_at": datetime.now(timezone.utc) - timedelta(hours=2),
                "current_value": 13.2,
                "threshold_value": 12.0,
                "affected_positions": ["BTCUSDT", "ETHUSDT"],
                "recommended_action": "Reduce position sizes or hedge portfolio",
                "is_resolved": False,
                "resolved_at": None,
            },
            {
                "alert_id": "alert_002",
                "alert_type": "var_limit",
                "severity": "medium",
                "message": "Daily VaR approaching limit",
                "triggered_at": datetime.now(timezone.utc) - timedelta(hours=4),
                "current_value": 7200.0,
                "threshold_value": 7500.0,
                "affected_positions": None,
                "recommended_action": "Monitor closely and consider reducing exposure",
                "is_resolved": False,
                "resolved_at": None,
            },
            {
                "alert_id": "alert_003",
                "alert_type": "concentration_risk",
                "severity": "medium",
                "message": "BTC concentration exceeds 35% of portfolio",
                "triggered_at": datetime.now(timezone.utc) - timedelta(hours=6),
                "current_value": 37.5,
                "threshold_value": 35.0,
                "affected_positions": ["BTCUSDT"],
                "recommended_action": "Diversify holdings or reduce BTC exposure",
                "is_resolved": True,
                "resolved_at": datetime.now(timezone.utc) - timedelta(hours=1),
            },
            {
                "alert_id": "alert_004",
                "alert_type": "leverage_limit",
                "severity": "low",
                "message": "Portfolio leverage increased to 2.8x",
                "triggered_at": datetime.now(timezone.utc) - timedelta(hours=8),
                "current_value": 2.8,
                "threshold_value": 3.0,
                "affected_positions": None,
                "recommended_action": "Monitor leverage and avoid additional exposure",
                "is_resolved": False,
                "resolved_at": None,
            },
        ]

        # Apply filters
        filtered_alerts = []
        for alert_data in mock_alerts:
            # Apply severity filter
            if severity and alert_data["severity"] != severity:
                continue

            # Apply unresolved filter
            if unresolved_only and alert_data["is_resolved"]:
                continue

            alert = RiskAlertResponse(**alert_data)
            filtered_alerts.append(alert)

        return filtered_alerts[:limit]

    except Exception as e:
        logger.error(f"Risk alerts retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get risk alerts: {e!s}",
        )


@router.get("/positions", response_model=list[PositionRiskResponse])
async def get_position_risks(
    exchange: str | None = Query(None, description="Filter by exchange"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    current_user: User = Depends(get_current_user),
):
    """
    Get risk analysis for individual positions.

    Args:
        exchange: Optional exchange filter
        symbol: Optional symbol filter
        current_user: Current authenticated user

    Returns:
        List[PositionRiskResponse]: Position risk analysis

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Mock position risk data (in production, calculate from actual positions)
        mock_positions = [
            {
                "position_id": "pos_001",
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "side": "long",
                "quantity": Decimal("2.0"),
                "market_value": Decimal("94000.00"),
                "unrealized_pnl": Decimal("4000.00"),
                "risk_amount": Decimal("1880.00"),  # 2% of market value
                "risk_percentage": 2.0,
                "var_contribution": Decimal("2350.00"),
                "beta": 1.0,  # BTC beta vs itself
                "correlation": 1.0,  # BTC correlation vs itself
                "concentration_risk": 62.7,  # High concentration
                "liquidity_risk": "low",
                "time_decay_risk": None,
            },
            {
                "position_id": "pos_002",
                "symbol": "ETHUSDT",
                "exchange": "binance",
                "side": "long",
                "quantity": Decimal("15.0"),
                "market_value": Decimal("46500.00"),
                "unrealized_pnl": Decimal("1500.00"),
                "risk_amount": Decimal("930.00"),
                "risk_percentage": 2.0,
                "var_contribution": Decimal("1162.50"),
                "beta": 1.3,  # ETH beta vs BTC
                "correlation": 0.85,  # ETH correlation vs BTC
                "concentration_risk": 31.0,
                "liquidity_risk": "low",
                "time_decay_risk": None,
            },
            {
                "position_id": "pos_003",
                "symbol": "ADAUSDT",
                "exchange": "coinbase",
                "side": "long",
                "quantity": Decimal("25000.0"),
                "market_value": Decimal("9500.00"),
                "unrealized_pnl": Decimal("-500.00"),
                "risk_amount": Decimal("190.00"),
                "risk_percentage": 2.0,
                "var_contribution": Decimal("285.00"),
                "beta": 1.5,  # ADA beta vs BTC
                "correlation": 0.72,  # ADA correlation vs BTC
                "concentration_risk": 6.3,
                "liquidity_risk": "medium",
                "time_decay_risk": None,
            },
        ]

        # Apply filters
        filtered_positions = []
        for pos_data in mock_positions:
            # Apply exchange filter
            if exchange and pos_data["exchange"] != exchange:
                continue

            # Apply symbol filter
            if symbol and pos_data["symbol"] != symbol:
                continue

            position = PositionRiskResponse(**pos_data)
            filtered_positions.append(position)

        return filtered_positions

    except Exception as e:
        logger.error(f"Position risks retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get position risks: {e!s}",
        )


@router.post("/stress-test", response_model=StressTestResponse)
async def run_stress_test(
    stress_test_request: StressTestRequest, current_user: User = Depends(get_current_user)
):
    """
    Run portfolio stress test.

    Args:
        stress_test_request: Stress test parameters
        current_user: Current authenticated user

    Returns:
        StressTestResponse: Stress test results

    Raises:
        HTTPException: If stress test fails
    """
    try:
        import uuid

        # Mock stress test execution (in production, run actual stress testing)
        test_id = f"stress_{uuid.uuid4().hex[:8]}"

        # Mock scenarios and results
        worst_case_loss = Decimal("-25000.00")  # 25k loss in worst case
        worst_case_scenario = "BTC -50%, Market Crash"

        # Mock confidence levels (VaR at different confidence levels)
        confidence_levels = {
            0.95: Decimal("-7500.00"),  # 95% confidence
            0.99: Decimal("-15000.00"),  # 99% confidence
        }

        # Mock time horizons (losses over different time periods)
        time_horizons = {
            1: Decimal("-3750.00"),  # 1 day
            5: Decimal("-8400.00"),  # 5 days
            10: Decimal("-11900.00"),  # 10 days
        }

        # Portfolio resilience score (0-100, higher is better)
        resilience_score = 75.5

        recommendations = [
            "Consider reducing BTC concentration below 50%",
            "Add hedging positions during high volatility periods",
            "Maintain higher cash reserves for extreme scenarios",
            "Implement dynamic position sizing based on volatility",
            "Consider correlation breakdown scenarios in risk models",
        ]

        result = StressTestResponse(
            test_id=test_id,
            test_name=stress_test_request.test_name,
            scenarios_tested=len(stress_test_request.scenarios),
            worst_case_loss=worst_case_loss,
            worst_case_scenario=worst_case_scenario,
            confidence_levels=confidence_levels,
            time_horizons=time_horizons,
            portfolio_resilience_score=resilience_score,
            recommendations=recommendations,
            completed_at=datetime.now(timezone.utc),
        )

        logger.info(
            "Stress test completed",
            test_id=test_id,
            test_name=stress_test_request.test_name,
            worst_case_loss=float(worst_case_loss),
            resilience_score=resilience_score,
            user=current_user.username,
        )

        return result

    except Exception as e:
        logger.error(f"Stress test failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Stress test failed: {e!s}"
        )


@router.get("/correlation-matrix")
async def get_correlation_matrix(
    symbols: list[str] = Query(..., description="Symbols to include in correlation matrix"),
    period: str = Query("30d", description="Correlation period: 7d, 30d, 90d"),
    current_user: User = Depends(get_current_user),
):
    """
    Get correlation matrix for specified symbols.

    Args:
        symbols: List of symbols
        period: Correlation calculation period
        current_user: Current authenticated user

    Returns:
        Dict: Correlation matrix data

    Raises:
        HTTPException: If calculation fails
    """
    try:
        # Mock correlation matrix (in production, calculate from price data)
        import random

        correlation_matrix = {}
        for i, symbol1 in enumerate(symbols):
            correlation_matrix[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    # Generate realistic correlation (higher for crypto pairs)
                    base_correlation = 0.6 if "USD" in symbol1 and "USD" in symbol2 else 0.3
                    noise = random.uniform(-0.2, 0.2)
                    corr = max(-1.0, min(1.0, base_correlation + noise))
                    correlation_matrix[symbol1][symbol2] = round(corr, 3)

        # Calculate average correlations
        avg_correlations = {}
        for symbol in symbols:
            other_correlations = [
                abs(correlation_matrix[symbol][other]) for other in symbols if other != symbol
            ]
            avg_correlations[symbol] = round(
                sum(other_correlations) / len(other_correlations) if other_correlations else 0.0, 3
            )

        return {
            "symbols": symbols,
            "period": period,
            "correlation_matrix": correlation_matrix,
            "average_correlations": avg_correlations,
            "highest_correlation": max(
                correlation_matrix[s1][s2] for s1 in symbols for s2 in symbols if s1 != s2
            ),
            "lowest_correlation": min(
                correlation_matrix[s1][s2] for s1 in symbols for s2 in symbols if s1 != s2
            ),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Correlation matrix calculation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate correlation matrix: {e!s}",
        )
