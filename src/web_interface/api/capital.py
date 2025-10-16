"""
Capital Management API endpoints for the web interface.

This module provides REST API endpoints for capital management operations including
allocation, release, utilization tracking, currency management, and fund flows.
All endpoints follow proper service layer patterns and use Decimal for financial precision.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.core.exceptions import CapitalAllocationError, ServiceError, ValidationError
from src.core.logging import get_logger
from src.utils.decorators import monitored
from src.utils.pydantic_validators import validate_amount, validate_utilized_amount
from src.utils.web_interface_utils import async_api_error_handler
from src.web_interface.security.auth import get_current_user
from src.web_interface.dependencies import get_web_capital_service, get_web_auth_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/capital", tags=["capital"])




# Request/Response Models
class CapitalAllocationRequest(BaseModel):
    """Request model for capital allocation."""

    strategy_id: str
    exchange: str
    amount: str  # Decimal as string
    bot_id: str | None = None
    risk_context: dict[str, Any] | None = None
    notes: str | None = None

    # Use shared validator from utilities
    _validate_amount = validate_amount


class CapitalReleaseRequest(BaseModel):
    """Request model for capital release."""

    strategy_id: str
    exchange: str
    amount: str  # Decimal as string
    bot_id: str | None = None
    reason: str | None = None

    # Use shared validator from utilities
    _validate_amount = validate_amount


class CapitalUtilizationUpdate(BaseModel):
    """Request model for utilization update."""

    strategy_id: str
    exchange: str
    utilized_amount: str  # Decimal as string
    bot_id: str | None = None

    # Use shared validator from utilities
    _validate_utilized_amount = validate_utilized_amount


class CurrencyHedgeRequest(BaseModel):
    """Request model for currency hedging."""

    base_currency: str
    quote_currency: str
    exposure_amount: str  # Decimal as string
    hedge_ratio: float = Field(ge=0.0, le=1.0)
    strategy: str = Field(default="natural", pattern="^(natural|forward|option)$")


class FundFlowRequest(BaseModel):
    """Request model for recording fund flows."""

    flow_type: str = Field(pattern="^(deposit|withdrawal|transfer)$")
    amount: str  # Decimal as string
    currency: str
    source: str | None = None
    destination: str | None = None
    reference: str | None = None
    notes: str | None = None

    # Use shared validator from utilities
    _validate_amount = validate_amount


class CapitalLimitsUpdate(BaseModel):
    """Request model for updating capital limits."""

    limit_type: str = Field(pattern="^(strategy|exchange|global|currency)$")
    limit_id: str
    max_allocation: str | None = None  # Decimal as string
    max_utilization_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    max_concentration: float | None = Field(default=None, ge=0.0, le=1.0)
    enabled: bool = True


# Allocation Response Models
class CapitalAllocationResponse(BaseModel):
    """Response model for capital allocation."""

    allocation_id: str
    strategy_id: str
    exchange: str
    allocated_amount: str
    utilized_amount: str
    available_amount: str
    utilization_ratio: float
    created_at: datetime
    last_updated: datetime
    status: str


class CapitalMetricsResponse(BaseModel):
    """Response model for capital metrics."""

    total_capital: str
    allocated_capital: str
    utilized_capital: str
    available_capital: str
    allocation_ratio: float
    utilization_ratio: float
    currency: str
    last_updated: datetime


# Capital Allocation Endpoints
@router.post("/allocate", response_model=CapitalAllocationResponse)
@monitored()
async def allocate_capital(
    request: CapitalAllocationRequest,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Allocate capital to a strategy."""
    try:
        logger.info(
            f"User {current_user['user_id']} allocating {request.amount} to "
            f"strategy {request.strategy_id} on {request.exchange}"
        )

        # Use auth service for authorization
        web_auth_service.require_permission(current_user, ["admin", "trader", "manager", "trading"])

        allocation = await web_capital_service.allocate_capital(
            strategy_id=request.strategy_id,
            exchange=request.exchange,
            amount=Decimal(request.amount),
            bot_id=request.bot_id,
            user_id=current_user["user_id"],
            risk_context=request.risk_context,
        )

        return CapitalAllocationResponse(**allocation)

    except CapitalAllocationError as e:
        logger.error(f"Capital allocation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        logger.error(f"Validation error in capital allocation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in capital allocation: {e}")
        if "Insufficient permissions" in str(e):
            raise HTTPException(status_code=403, detail=str(e))
        raise HTTPException(status_code=500, detail="Failed to allocate capital")


@router.post("/release")
@monitored()
@async_api_error_handler("release capital")
async def release_capital(
    request: CapitalReleaseRequest,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Release allocated capital from a strategy."""
    logger.info(
        f"User {current_user['user_id']} releasing {request.amount} from "
        f"strategy {request.strategy_id} on {request.exchange}"
    )

    success = await web_capital_service.release_capital(
        strategy_id=request.strategy_id,
        exchange=request.exchange,
        amount=Decimal(request.amount),
        bot_id=request.bot_id,
        user_id=current_user["user_id"],
        reason=request.reason,
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to release capital")

    return {"status": "success", "message": "Capital released successfully"}


@router.put("/update-utilization")
@monitored()
@async_api_error_handler("update utilization")
async def update_utilization(
    request: CapitalUtilizationUpdate,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Update capital utilization for a strategy."""
    success = await web_capital_service.update_utilization(
        strategy_id=request.strategy_id,
        exchange=request.exchange,
        utilized_amount=Decimal(request.utilized_amount),
        bot_id=request.bot_id,
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to update utilization")

    return {"status": "success", "message": "Utilization updated successfully"}


@router.get("/allocations")
@monitored()
@async_api_error_handler("get allocations")
async def get_allocations(
    strategy_id: str | None = None,
    exchange: str | None = None,
    active_only: bool = True,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get capital allocations."""
    allocations = await web_capital_service.get_allocations(
        strategy_id=strategy_id, exchange=exchange, active_only=active_only
    )

    return {"allocations": allocations, "count": len(allocations)}


@router.get("/allocations/{strategy_id}")
@monitored()
async def get_strategy_allocation(
    strategy_id: str,
    exchange: str | None = None,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get capital allocation for a specific strategy."""
    try:
        allocation = await web_capital_service.get_strategy_allocation(
            strategy_id=strategy_id, exchange=exchange
        )

        if not allocation:
            raise HTTPException(status_code=404, detail="Allocation not found")

        return allocation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy allocation: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve allocation")


# Capital Metrics Endpoints
@router.get("/metrics", response_model=CapitalMetricsResponse)
@monitored()
async def get_capital_metrics(
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get overall capital metrics."""
    try:
        metrics = await web_capital_service.get_capital_metrics()
        return CapitalMetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Error getting capital metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve capital metrics")


@router.get("/utilization")
@monitored()
async def get_utilization(
    by: str = Query(default="strategy", pattern="^(strategy|exchange|global)$"),
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get capital utilization breakdown."""
    try:
        utilization = await web_capital_service.get_utilization_breakdown(by=by)
        return {"breakdown_by": by, "utilization": utilization}

    except Exception as e:
        logger.error(f"Error getting utilization: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve utilization")


@router.get("/available")
@monitored()
async def get_available_capital(
    strategy_id: str | None = None,
    exchange: str | None = None,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get available capital for allocation."""
    try:
        available = await web_capital_service.get_available_capital(
            strategy_id=strategy_id, exchange=exchange
        )

        return {
            "available_capital": str(available["amount"]),
            "currency": available["currency"],
            "filters": {"strategy_id": strategy_id, "exchange": exchange},
        }

    except Exception as e:
        logger.error(f"Error getting available capital: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available capital")


@router.get("/exposure")
@monitored()
async def get_capital_exposure(
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get capital exposure analysis."""
    try:
        exposure = await web_capital_service.get_capital_exposure()
        return exposure

    except Exception as e:
        logger.error(f"Error getting capital exposure: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve capital exposure")


# Currency Management Endpoints
@router.get("/currency/exposure")
@monitored()
async def get_currency_exposure(
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get currency exposure breakdown."""
    try:
        exposure = await web_capital_service.get_currency_exposure()
        return exposure

    except Exception as e:
        logger.error(f"Error getting currency exposure: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve currency exposure")


@router.post("/currency/hedge")
@monitored()
async def create_currency_hedge(
    request: CurrencyHedgeRequest,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Create a currency hedge."""
    try:
        # Use auth service for authorization
        web_auth_service.require_risk_manager_permission(current_user)

        hedge = await web_capital_service.create_currency_hedge(
            base_currency=request.base_currency,
            quote_currency=request.quote_currency,
            exposure_amount=Decimal(request.exposure_amount),
            hedge_ratio=request.hedge_ratio,
            strategy=request.strategy,
            created_by=current_user["user_id"],
        )

        return {"status": "success", "hedge": hedge}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating currency hedge: {e}")
        raise HTTPException(status_code=500, detail="Failed to create currency hedge")


@router.get("/currency/rates")
@monitored()
async def get_exchange_rates(
    base_currency: str = Query(default="USDT"),
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get current currency exchange rates."""
    try:
        rates = await web_capital_service.get_exchange_rates(base_currency=base_currency)
        return rates

    except Exception as e:
        logger.error(f"Error getting exchange rates: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exchange rates")


@router.post("/currency/convert")
@monitored()
async def convert_currency(
    request: dict,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Convert currency."""
    try:
        conversion = await web_capital_service.convert_currency(
            from_currency=request["from_currency"],
            to_currency=request["to_currency"],
            amount=Decimal(request["amount"]),
        )
        return conversion

    except Exception as e:
        logger.error(f"Error converting currency: {e}")
        raise HTTPException(status_code=500, detail="Failed to convert currency")


# Fund Flow Endpoints
@router.get("/flows")
@monitored()
async def get_fund_flows(
    days: int = Query(default=30, ge=1, le=365),
    flow_type: str | None = None,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get fund flow history."""
    try:
        flows = await web_capital_service.get_fund_flows(days=days, flow_type=flow_type)

        return {
            "flows": flows,
            "count": len(flows),
            "period_days": days,
            "flow_type": flow_type,
        }

    except Exception as e:
        logger.error(f"Error getting fund flows: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve fund flows")


@router.post("/flows/record")
@monitored()
async def record_fund_flow(
    request: FundFlowRequest,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Record a fund flow transaction."""
    try:
        # Use auth service for authorization
        web_auth_service.require_treasurer_permission(current_user)

        flow = await web_capital_service.record_fund_flow(
            flow_type=request.flow_type,
            amount=Decimal(request.amount),
            currency=request.currency,
            source=request.source,
            destination=request.destination,
            reference=request.reference,
            notes=request.notes,
            recorded_by=current_user["user_id"],
        )

        return {"status": "success", "flow_id": flow["id"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording fund flow: {e}")
        raise HTTPException(status_code=500, detail="Failed to record fund flow")


@router.get("/flows/report")
@monitored()
async def get_fund_flow_report(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    format: str = Query(default="json", pattern="^(json|csv|pdf)$"),
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Generate fund flow report."""
    try:
        report = await web_capital_service.generate_fund_flow_report(
            start_date=start_date, end_date=end_date, format=format
        )

        return {"report": report, "format": format}

    except Exception as e:
        logger.error(f"Error generating fund flow report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate fund flow report")


# Limits & Controls Endpoints
@router.get("/limits")
@monitored()
async def get_allocation_limits(
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get configured allocation limits."""
    try:
        limits = await web_capital_service.get_allocation_limits()
        return limits

    except Exception as e:
        logger.error(f"Error getting allocation limits: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve allocation limits")


@router.post("/limits")
@monitored()
async def set_allocation_limits(
    request: dict,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Set allocation limits."""
    try:
        # Use auth service for authorization
        web_auth_service.require_admin(current_user)

        limits = await web_capital_service.set_allocation_limits(**request)
        return limits

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting allocation limits: {e}")
        raise HTTPException(status_code=500, detail="Failed to set allocation limits")


@router.put("/limits/update")
@monitored()
async def update_capital_limits(
    request: CapitalLimitsUpdate,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Update capital limits configuration."""
    try:
        # Use auth service for authorization
        web_auth_service.require_risk_manager_permission(current_user)

        updated = await web_capital_service.update_capital_limits(
            limit_type=request.limit_type,
            limit_id=request.limit_id,
            max_allocation=Decimal(request.max_allocation) if request.max_allocation else None,
            max_utilization_ratio=request.max_utilization_ratio,
            max_concentration=request.max_concentration,
            enabled=request.enabled,
            updated_by=current_user["user_id"],
        )

        if not updated:
            raise HTTPException(status_code=400, detail="Failed to update limits")

        return {"status": "success", "message": "Capital limits updated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating capital limits: {e}")
        raise HTTPException(status_code=500, detail="Failed to update capital limits")


@router.get("/breaches")
@monitored()
async def get_limit_breaches(
    hours: int = Query(default=24, ge=1, le=168),
    severity: str | None = None,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get recent capital limit breaches."""
    try:
        breaches = await web_capital_service.get_limit_breaches(hours=hours, severity=severity)

        return {
            "breaches": breaches,
            "count": len(breaches),
            "time_window_hours": hours,
            "severity_filter": severity,
        }

    except Exception as e:
        logger.error(f"Error getting limit breaches: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve limit breaches")


# Portfolio Management Endpoints
@router.post("/rebalance")
@monitored()
async def rebalance_portfolio(
    request: dict,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
    web_auth_service=Depends(get_web_auth_service),
):
    """Rebalance portfolio allocations."""
    try:
        # Use auth service for authorization
        web_auth_service.require_admin(current_user)

        result = await web_capital_service.rebalance_portfolio(
            target_allocations=request.get("target_allocations"),
            dry_run=request.get("dry_run", False),
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebalancing portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to rebalance portfolio")


# Capital Optimization Endpoints
@router.get("/optimization/calculate")
@monitored()
async def calculate_optimal_allocation(
    risk_tolerance: str,
    optimization_method: str,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Calculate optimal capital allocation."""
    try:
        result = await web_capital_service.calculate_optimal_allocation(
            risk_tolerance=risk_tolerance, optimization_method=optimization_method
        )
        return result

    except Exception as e:
        logger.error(f"Error calculating optimal allocation: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate optimal allocation")


@router.get("/efficiency")
@monitored()
async def get_capital_efficiency(
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get capital efficiency metrics."""
    try:
        efficiency = await web_capital_service.get_capital_efficiency()
        return efficiency

    except Exception as e:
        logger.error(f"Error getting capital efficiency: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve capital efficiency")


# Capital Reservation Endpoints
@router.post("/reserve")
@monitored()
async def reserve_capital(
    request: dict,
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Reserve capital for future use."""
    try:
        reservation = await web_capital_service.reserve_capital(
            amount=Decimal(request["amount"]),
            currency=request["currency"],
            purpose=request["purpose"],
            duration_minutes=request.get("duration_minutes", 60),
        )
        return reservation

    except Exception as e:
        logger.error(f"Error reserving capital: {e}")
        raise HTTPException(status_code=500, detail="Failed to reserve capital")


@router.get("/reserved")
@monitored()
async def get_reserved_capital(
    current_user: dict = Depends(get_current_user),
    web_capital_service=Depends(get_web_capital_service),
):
    """Get reserved capital information."""
    try:
        reserved = await web_capital_service.get_reserved_capital()
        return reserved

    except Exception as e:
        logger.error(f"Error getting reserved capital: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve reserved capital")
