"""
Data Management API endpoints for the web interface.

This module provides REST API endpoints for data pipeline management, quality monitoring,
feature store access, market data configuration, and data monitoring operations.
All endpoints follow proper service layer patterns.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.utils.decorators import monitored
from src.web_interface.auth.middleware import get_current_user
from src.web_interface.dependencies import get_web_data_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


# Request/Response Models
class PipelineControlRequest(BaseModel):
    """Request model for pipeline control."""

    action: str = Field(pattern="^(start|stop|restart|pause|resume)$")
    pipeline_id: str | None = None
    parameters: dict[str, Any] | None = None


class DataValidationRequest(BaseModel):
    """Request model for data validation."""

    data_source: str
    symbol: str | None = None
    timeframe: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    validation_rules: list[str] | None = None


class FeatureComputeRequest(BaseModel):
    """Request model for feature computation."""

    feature_ids: list[str]
    symbols: list[str]
    timeframe: str = "1h"
    lookback_periods: int = Field(default=100, ge=10, le=1000)
    include_derived: bool = True


class DataSourceConfigRequest(BaseModel):
    """Request model for data source configuration."""

    source_type: str = Field(pattern="^(market|news|social|alternative)$")
    provider: str
    config: dict[str, Any]
    enabled: bool = True
    priority: int = Field(default=1, ge=1, le=10)


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status."""

    pipeline_id: str
    status: str
    uptime_seconds: int
    messages_processed: int
    error_count: int
    latency_ms: float
    throughput_per_second: float
    last_error: str | None
    last_updated: datetime


class DataQualityMetricsResponse(BaseModel):
    """Response model for data quality metrics."""

    completeness: float
    accuracy: float
    consistency: float
    timeliness: float
    validity: float
    uniqueness: float
    total_records: int
    valid_records: int
    invalid_records: int
    missing_fields: dict[str, int]
    timestamp: datetime


# Data Pipeline Endpoints
@router.get("/pipeline/status")
@monitored()
async def get_pipeline_status(
    pipeline_id: str | None = None,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get data pipeline status."""
    try:
        status = await web_data_service.get_pipeline_status(pipeline_id)

        if pipeline_id and not status:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

        return {"pipelines": status if isinstance(status, list) else [status]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pipeline status")


@router.post("/pipeline/control")
@monitored()
async def control_pipeline(
    request: PipelineControlRequest,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Control data pipeline operations."""
    try:
        if current_user.get("role") not in ["admin", "operator"]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        result = await web_data_service.control_pipeline(
            action=request.action,
            pipeline_id=request.pipeline_id,
            parameters=request.parameters,
        )

        return {
            "status": "success",
            "action": request.action,
            "pipeline_id": request.pipeline_id,
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling pipeline: {e}")
        raise HTTPException(status_code=500, detail="Failed to control pipeline")


@router.get("/pipeline/metrics")
@monitored()
async def get_pipeline_metrics(
    hours: int = Query(default=1, ge=1, le=24),
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get data pipeline performance metrics."""
    try:
        metrics = await web_data_service.get_pipeline_metrics(hours)
        return metrics

    except Exception as e:
        logger.error(f"Error getting pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pipeline metrics")


# Data Quality Endpoints
@router.get("/quality/metrics", response_model=DataQualityMetricsResponse)
@monitored()
async def get_data_quality_metrics(
    data_source: str | None = None,
    symbol: str | None = None,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get data quality metrics."""
    try:
        metrics = await web_data_service.get_data_quality_metrics(
            data_source=data_source, symbol=symbol
        )
        return DataQualityMetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Error getting data quality metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality metrics")


@router.get("/quality/validation-report")
@monitored()
async def get_validation_report(
    days: int = Query(default=7, ge=1, le=30),
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get data validation report."""
    try:
        report = await web_data_service.get_validation_report(days)
        return report

    except Exception as e:
        logger.error(f"Error getting validation report: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve validation report")


@router.post("/quality/validate")
@monitored()
async def validate_data(
    request: DataValidationRequest,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Validate data against quality rules."""
    try:
        validation_result = await web_data_service.validate_data(
            data_source=request.data_source,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            validation_rules=request.validation_rules,
        )

        return {
            "status": "validated",
            "is_valid": validation_result["is_valid"],
            "errors": validation_result.get("errors", []),
            "warnings": validation_result.get("warnings", []),
            "stats": validation_result.get("stats", {}),
        }

    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate data")


@router.get("/quality/anomalies")
@monitored()
async def get_data_anomalies(
    hours: int = Query(default=24, ge=1, le=168),
    severity: str | None = Query(default=None, pattern="^(low|medium|high|critical)$"),
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get detected data anomalies."""
    try:
        anomalies = await web_data_service.get_data_anomalies(hours=hours, severity=severity)

        return {
            "anomalies": anomalies,
            "count": len(anomalies),
            "time_window_hours": hours,
            "severity_filter": severity,
        }

    except Exception as e:
        logger.error(f"Error getting data anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data anomalies")


# Feature Store Endpoints
@router.get("/features/list")
@monitored()
async def list_features(
    category: str | None = None,
    active_only: bool = True,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """List available features in the feature store."""
    try:
        features = await web_data_service.list_features(category=category, active_only=active_only)

        return {"features": features, "count": len(features)}

    except Exception as e:
        logger.error(f"Error listing features: {e}")
        raise HTTPException(status_code=500, detail="Failed to list features")


@router.get("/features/{feature_id}")
@monitored()
async def get_feature_details(
    feature_id: str,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get detailed information about a specific feature."""
    try:
        feature = await web_data_service.get_feature_details(feature_id)

        if not feature:
            raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")

        return feature

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feature details")


@router.post("/features/compute")
@monitored()
async def compute_features(
    request: FeatureComputeRequest,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Compute features for specified symbols."""
    try:
        results = await web_data_service.compute_features(
            feature_ids=request.feature_ids,
            symbols=request.symbols,
            timeframe=request.timeframe,
            lookback_periods=request.lookback_periods,
            include_derived=request.include_derived,
        )

        return {
            "status": "computed",
            "features_computed": len(request.feature_ids),
            "symbols_processed": len(request.symbols),
            "results": results,
        }

    except ValidationError as e:
        logger.error(f"Validation error in feature computation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error computing features: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute features")


@router.get("/features/metadata")
@monitored()
async def get_feature_metadata(
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get feature store metadata."""
    try:
        metadata = await web_data_service.get_feature_metadata()
        return metadata

    except Exception as e:
        logger.error(f"Error getting feature metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feature metadata")


# Market Data Configuration Endpoints
@router.get("/sources")
@monitored()
async def list_data_sources(
    source_type: str | None = None,
    enabled_only: bool = True,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """List configured data sources."""
    try:
        sources = await web_data_service.list_data_sources(
            source_type=source_type, enabled_only=enabled_only
        )

        return {"sources": sources, "count": len(sources)}

    except Exception as e:
        logger.error(f"Error listing data sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to list data sources")


@router.post("/sources/configure")
@monitored()
async def configure_data_source(
    request: DataSourceConfigRequest,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Configure a new data source."""
    try:
        if current_user.get("role") not in ["admin", "developer"]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        source_id = await web_data_service.configure_data_source(
            source_type=request.source_type,
            provider=request.provider,
            config=request.config,
            enabled=request.enabled,
            priority=request.priority,
            configured_by=current_user["user_id"],
        )

        return {
            "status": "configured",
            "source_id": source_id,
            "message": f"Data source {request.provider} configured successfully",
        }

    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Validation error in source configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error configuring data source: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure data source")


@router.put("/sources/{source_id}/update")
@monitored()
async def update_data_source(
    source_id: str,
    config: dict[str, Any],
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Update data source configuration."""
    try:
        if current_user.get("role") not in ["admin", "developer"]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        success = await web_data_service.update_data_source(
            source_id=source_id, config=config, updated_by=current_user["user_id"]
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"Data source {source_id} not found")

        return {"status": "updated", "source_id": source_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating data source: {e}")
        raise HTTPException(status_code=500, detail="Failed to update data source")


@router.delete("/sources/{source_id}")
@monitored()
async def delete_data_source(
    source_id: str,
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Delete a data source."""
    try:
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        success = await web_data_service.delete_data_source(
            source_id=source_id, deleted_by=current_user["user_id"]
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"Data source {source_id} not found")

        return {"status": "deleted", "source_id": source_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data source: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete data source")


# Data Monitoring Endpoints
@router.get("/monitoring/health")
@monitored()
async def get_data_health(
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get data system health status."""
    try:
        health = await web_data_service.get_data_health()

        return {
            "status": health["overall_status"],
            "components": health["components"],
            "checks_passed": health["checks_passed"],
            "checks_failed": health["checks_failed"],
            "last_check": health["last_check"],
        }

    except Exception as e:
        logger.error(f"Error getting data health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data health")


@router.get("/monitoring/latency")
@monitored()
async def get_data_latency(
    source: str | None = None,
    hours: int = Query(default=1, ge=1, le=24),
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get data latency metrics."""
    try:
        latency = await web_data_service.get_data_latency(source=source, hours=hours)

        return {
            "source": source or "all",
            "time_window_hours": hours,
            "latency_metrics": latency,
        }

    except Exception as e:
        logger.error(f"Error getting data latency: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data latency")


@router.get("/monitoring/throughput")
@monitored()
async def get_data_throughput(
    source: str | None = None,
    hours: int = Query(default=1, ge=1, le=24),
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get data throughput metrics."""
    try:
        throughput = await web_data_service.get_data_throughput(source=source, hours=hours)

        return {
            "source": source or "all",
            "time_window_hours": hours,
            "throughput_metrics": throughput,
        }

    except Exception as e:
        logger.error(f"Error getting data throughput: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data throughput")


# Data Cache Management Endpoints
@router.post("/cache/clear")
@monitored()
async def clear_data_cache(
    cache_type: str | None = Query(default=None, pattern="^(market|feature|validation)$"),
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Clear data cache."""
    try:
        if current_user.get("role") not in ["admin", "developer"]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        cleared = await web_data_service.clear_data_cache(cache_type=cache_type)

        return {
            "status": "cleared",
            "cache_type": cache_type or "all",
            "entries_cleared": cleared,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing data cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear data cache")


@router.get("/cache/stats")
@monitored()
async def get_cache_statistics(
    current_user: dict = Depends(get_current_user),
    web_data_service=Depends(get_web_data_service),
):
    """Get data cache statistics."""
    try:
        stats = await web_data_service.get_cache_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")
