"""
Risk Management API endpoints for T-Bot web interface.

This module provides risk monitoring, limits management, and risk analysis
functionality for the trading system.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.web_interface.di_registration import get_web_risk_service
from src.web_interface.security.auth import User, get_admin_user, get_current_user

logger = get_logger(__name__)
router = APIRouter()


def get_web_risk_service_instance():
    """Get web risk service for business logic through DI."""
    return get_web_risk_service()


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
        risk_service = get_web_risk_service_instance()
        risk_data = await risk_service.get_risk_dashboard_data()

        overview = risk_data.get("overview", {})
        var_metrics = risk_data.get("var_metrics", {})
        portfolio_risk = risk_data.get("portfolio_risk", {})

        return RiskMetricsResponse(
            portfolio_value=overview.get("portfolio_value", Decimal("0")),
            total_exposure=overview.get("total_exposure", Decimal("0")),
            leverage=overview.get("leverage", Decimal("1")),
            var_1d=var_metrics.get("var_1d", Decimal("0")),
            var_5d=var_metrics.get("var_5d", Decimal("0")),
            expected_shortfall=var_metrics.get("expected_shortfall", Decimal("0")),
            max_drawdown=portfolio_risk.get("max_drawdown", Decimal("0")),
            max_drawdown_percentage=portfolio_risk.get("max_drawdown_percentage", Decimal("0")),
            volatility=portfolio_risk.get("volatility", Decimal("0")),
            beta=portfolio_risk.get("beta"),
            correlation_btc=portfolio_risk.get("correlation_btc"),
            risk_score=overview.get("risk_score", Decimal("0")),
            risk_level=overview.get("risk_level", "unknown"),
            last_updated=datetime.now(timezone.utc),
        )

    except ServiceError as e:
        logger.error(f"Risk service error: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Risk service unavailable: {e!s}",
        )
    except ValidationError as e:
        logger.error(f"Risk data validation error: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid risk data: {e!s}",
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
        # Business logic moved to service layer - mock data for development
        return RiskLimitsResponse(
            max_portfolio_risk=Decimal("0.20"),  # 20% max portfolio risk
            max_position_size=Decimal("25000.00"),  # $25k max position
            max_leverage=Decimal("3.0"),  # 3x max leverage
            max_daily_loss=Decimal("5000.00"),  # $5k max daily loss
            max_drawdown_limit=Decimal("0.15"),  # 15% max drawdown
            concentration_limit=Decimal("0.30"),  # 30% max in single asset
            correlation_limit=Decimal("0.70"),  # 70% max correlation
            var_limit=Decimal("7500.00"),  # $7.5k VaR limit
            stop_loss_required=True,
            position_sizing_method="kelly_criterion",
            risk_per_trade=Decimal("0.02"),  # 2% risk per trade
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
        risk_service = get_web_risk_service_instance()

        # Prepare parameters for validation
        parameters = limits_request.model_dump(exclude_none=True)

        # Validate parameters through service
        validation_result = await risk_service.validate_risk_parameters(parameters)

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation errors: {validation_result['errors']}",
            )

        # Note: Actual risk limits update should go through risk management service
        # This is currently validation only - full implementation would require
        # integration with risk_management.service to update actual limits

        logger.info("Risk limits validated", updated_fields=parameters, admin=current_user.username)

        return {
            "success": True,
            "message": "Risk limits validated successfully (update not implemented)",
            "validated_fields": parameters,
            "validated_by": current_user.username,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "recommendations": validation_result.get("recommendations", []),
            "note": "Full risk limits update requires integration with risk management service",
        }

    except ValidationError as e:
        logger.error(f"Risk parameter validation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid risk parameters: {e!s}"
        )
    except ServiceError as e:
        logger.error(f"Risk service error: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Risk service unavailable: {e!s}",
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Risk limits validation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk limits validation failed: {e!s}",
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
        risk_service = get_web_risk_service_instance()

        # Prepare filters for service layer (business logic moved to service)
        filters = {
            "severity": severity,
            "unresolved_only": unresolved_only,
            "limit": limit,
        }

        # Get alerts through service layer
        alert_data_list = risk_service.generate_mock_risk_alerts(filters)

        # Convert to response models
        filtered_alerts = []
        for alert_data in alert_data_list:
            try:
                alert = RiskAlertResponse(**alert_data)
                filtered_alerts.append(alert)
            except Exception as model_error:
                logger.warning(f"Failed to convert alert data to response model: {model_error}")
                continue  # Skip invalid alert data

        return filtered_alerts

    except ServiceError as e:
        logger.error(f"Risk service error: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Risk service unavailable: {e!s}",
        )
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
        risk_service = get_web_risk_service_instance()

        # Prepare filters for service layer (business logic moved to service)
        filters = {
            "exchange": exchange,
            "symbol": symbol,
        }

        # Get position risk data through service layer
        position_data_list = risk_service.generate_mock_position_risks(filters)

        # Convert to response models
        filtered_positions = []
        for pos_data in position_data_list:
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
        risk_service = get_web_risk_service_instance()

        # Convert request to dict for service layer (business logic moved to service)
        test_request_data = {
            "test_name": stress_test_request.test_name,
            "scenarios": stress_test_request.scenarios,
            "confidence_levels": stress_test_request.confidence_levels,
            "time_horizons": stress_test_request.time_horizons,
        }

        # Generate stress test results through service layer
        result_data = risk_service.generate_mock_stress_test_results(test_request_data)

        # Convert to response model
        result = StressTestResponse(**result_data)

        logger.info(
            "Stress test completed",
            test_id=result.test_id,
            test_name=result.test_name,
            worst_case_loss=str(result.worst_case_loss),
            resilience_score=str(result.portfolio_resilience_score),
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
        risk_service = get_web_risk_service_instance()

        # Generate correlation matrix through service layer (business logic moved to service)
        correlation_data = risk_service.generate_mock_correlation_matrix(symbols, period)

        return correlation_data

    except Exception as e:
        logger.error(f"Correlation matrix calculation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate correlation matrix: {e!s}",
        )
