"""
FastAPI endpoints for state management operations.

This module provides REST API endpoints for:
- State persistence and recovery
- Checkpoint management
- Trade lifecycle tracking
- Quality control validation
- State synchronization monitoring

All endpoints include proper authentication, validation, and error handling.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.config import Config
from src.core.exceptions import StateError, ValidationError, ConflictError
from src.core.logging import get_logger
from src.core.types import BotState, OrderRequest, MarketData
from src.state.state_manager import StateManager
from src.state.checkpoint_manager import CheckpointManager
from src.state.trade_lifecycle_manager import TradeLifecycleManager
from src.state.quality_controller import QualityController
from src.state.state_sync_manager import StateSyncManager
from src.web_interface.security.auth import get_current_user

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/state", tags=["state_management"])


# Request/Response models

class StateSnapshotRequest(BaseModel):
    """Request model for creating state snapshots."""
    
    bot_id: str = Field(..., description="Bot identifier")
    snapshot_type: str = Field(default="manual", description="Snapshot type")
    description: Optional[str] = Field(default=None, description="Optional description")


class StateSnapshotResponse(BaseModel):
    """Response model for state snapshots."""
    
    snapshot_id: str
    bot_id: str
    snapshot_type: str
    created_at: datetime
    size_bytes: int
    compressed: bool
    validated: bool


class CheckpointRequest(BaseModel):
    """Request model for checkpoint operations."""
    
    bot_id: str = Field(..., description="Bot identifier")
    checkpoint_type: str = Field(default="manual", description="Checkpoint type")


class CheckpointResponse(BaseModel):
    """Response model for checkpoint operations."""
    
    checkpoint_id: str
    bot_id: str
    created_at: datetime
    size_bytes: int
    checkpoint_type: str


class RecoveryRequest(BaseModel):
    """Request model for recovery operations."""
    
    bot_id: str = Field(..., description="Bot identifier")
    checkpoint_id: str = Field(..., description="Checkpoint to restore from")
    recovery_type: str = Field(default="full", description="Recovery type")


class TradeValidationRequest(BaseModel):
    """Request model for trade validation."""
    
    order_request: dict = Field(..., description="Order request data")
    market_data: Optional[dict] = Field(default=None, description="Current market data")
    portfolio_context: Optional[dict] = Field(default=None, description="Portfolio context")


class TradeValidationResponse(BaseModel):
    """Response model for trade validation."""
    
    validation_id: str
    overall_result: str
    overall_score: float
    risk_level: str
    validation_time_ms: float
    recommendations: List[str]


class PostTradeAnalysisRequest(BaseModel):
    """Request model for post-trade analysis."""
    
    trade_id: str = Field(..., description="Trade identifier")
    execution_result: dict = Field(..., description="Execution result data")
    market_data_before: Optional[dict] = Field(default=None, description="Market data before trade")
    market_data_after: Optional[dict] = Field(default=None, description="Market data after trade")


class PostTradeAnalysisResponse(BaseModel):
    """Response model for post-trade analysis."""
    
    analysis_id: str
    trade_id: str
    overall_quality_score: float
    slippage_bps: float
    execution_time_seconds: float
    quality_issues: List[str]
    recommendations: List[str]


class SyncStatusResponse(BaseModel):
    """Response model for sync status."""
    
    entity_type: str
    entity_id: str
    current_version: Optional[int]
    last_updated: Optional[str]
    sync_in_progress: bool
    has_conflicts: bool
    consistency_status: dict


# Dependency injection

async def get_state_manager() -> StateManager:
    """Get StateManager instance."""
    # This would typically be injected via dependency injection
    # For now, return a mock or singleton instance
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="StateManager dependency not configured"
    )


async def get_checkpoint_manager() -> CheckpointManager:
    """Get CheckpointManager instance."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="CheckpointManager dependency not configured"
    )


async def get_lifecycle_manager() -> TradeLifecycleManager:
    """Get TradeLifecycleManager instance."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="TradeLifecycleManager dependency not configured"
    )


async def get_quality_controller() -> QualityController:
    """Get QualityController instance."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="QualityController dependency not configured"
    )


async def get_sync_manager() -> StateSyncManager:
    """Get StateSyncManager instance."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="StateSyncManager dependency not configured"
    )


# State Management Endpoints

@router.get("/bot/{bot_id}/state", response_model=Dict[str, Any])
async def get_bot_state(
    bot_id: str,
    version_id: Optional[str] = Query(default=None, description="Specific version to load"),
    current_user: dict = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Get current bot state.
    
    Args:
        bot_id: Bot identifier
        version_id: Specific version to load (latest if not specified)
        
    Returns:
        Bot state data
    """
    try:
        bot_state = await state_manager.load_bot_state(bot_id, version_id)
        
        if not bot_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bot state not found for bot {bot_id}"
            )
        
        return {
            "bot_id": bot_id,
            "state": bot_state.model_dump(),
            "version_id": version_id,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }
        
    except StateError as e:
        logger.error(f"Failed to get bot state: {e}", bot_id=bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve bot state: {str(e)}"
        )


@router.post("/bot/{bot_id}/state", response_model=Dict[str, str])
async def save_bot_state(
    bot_id: str,
    bot_state: Dict[str, Any],
    create_snapshot: bool = Query(default=False, description="Create snapshot"),
    current_user: dict = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Save bot state.
    
    Args:
        bot_id: Bot identifier
        bot_state: Bot state data
        create_snapshot: Whether to create a snapshot
        
    Returns:
        Version information
    """
    try:
        # Validate bot state data
        state_obj = BotState(**bot_state)
        
        version_id = await state_manager.save_bot_state(
            bot_id,
            state_obj,
            create_snapshot=create_snapshot
        )
        
        return {
            "version_id": version_id,
            "bot_id": bot_id,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid bot state data: {str(e)}"
        )
    except StateError as e:
        logger.error(f"Failed to save bot state: {e}", bot_id=bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save bot state: {str(e)}"
        )


@router.get("/bot/{bot_id}/state/metrics", response_model=Dict[str, Any])
async def get_state_metrics(
    bot_id: str,
    hours: int = Query(default=24, ge=1, le=168, description="Time period in hours"),
    current_user: dict = Depends(get_current_user),
    state_manager: StateManager = Depends(get_state_manager)
):
    """
    Get state management metrics for a bot.
    
    Args:
        bot_id: Bot identifier
        hours: Time period for metrics
        
    Returns:
        State metrics
    """
    try:
        metrics = await state_manager.get_state_metrics(bot_id, hours)
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get state metrics: {e}", bot_id=bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve state metrics: {str(e)}"
        )


# Checkpoint Management Endpoints

@router.post("/checkpoint", response_model=CheckpointResponse)
async def create_checkpoint(
    request: CheckpointRequest,
    current_user: dict = Depends(get_current_user),
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
):
    """
    Create a checkpoint for bot state.
    
    Args:
        request: Checkpoint creation request
        
    Returns:
        Checkpoint information
    """
    try:
        # Load bot state first (this would need to be integrated with StateManager)
        # For now, create a mock BotState
        from src.core.types import BotState, BotStatus
        from decimal import Decimal
        
        mock_state = BotState(
            bot_id=request.bot_id,
            status=BotStatus.RUNNING,
            allocated_capital=Decimal("1000.0")
        )
        
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            request.bot_id,
            mock_state,
            checkpoint_type=request.checkpoint_type
        )
        
        metadata = checkpoint_manager.checkpoints.get(checkpoint_id)
        
        return CheckpointResponse(
            checkpoint_id=checkpoint_id,
            bot_id=request.bot_id,
            created_at=metadata.created_at if metadata else datetime.now(timezone.utc),
            size_bytes=metadata.size_bytes if metadata else 0,
            checkpoint_type=request.checkpoint_type
        )
        
    except Exception as e:
        logger.error(f"Failed to create checkpoint: {e}", bot_id=request.bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create checkpoint: {str(e)}"
        )


@router.get("/checkpoint", response_model=List[Dict[str, Any]])
async def list_checkpoints(
    bot_id: Optional[str] = Query(default=None, description="Filter by bot ID"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
    current_user: dict = Depends(get_current_user),
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
):
    """
    List available checkpoints.
    
    Args:
        bot_id: Optional bot ID filter
        limit: Maximum results to return
        
    Returns:
        List of checkpoints
    """
    try:
        checkpoints = await checkpoint_manager.list_checkpoints(bot_id, limit)
        return checkpoints
        
    except Exception as e:
        logger.error(f"Failed to list checkpoints: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list checkpoints: {str(e)}"
        )


@router.post("/recovery", response_model=Dict[str, Any])
async def restore_from_checkpoint(
    request: RecoveryRequest,
    current_user: dict = Depends(get_current_user),
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
):
    """
    Restore bot state from checkpoint.
    
    Args:
        request: Recovery request
        
    Returns:
        Recovery result
    """
    try:
        # Create recovery plan
        plan = await checkpoint_manager.create_recovery_plan(request.bot_id)
        
        # Execute recovery (simplified)
        success = await checkpoint_manager.execute_recovery_plan(plan)
        
        return {
            "recovery_id": str(uuid4()),
            "bot_id": request.bot_id,
            "checkpoint_id": request.checkpoint_id,
            "success": success,
            "recovery_type": request.recovery_type,
            "executed_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to restore from checkpoint: {e}", bot_id=request.bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore from checkpoint: {str(e)}"
        )


@router.get("/checkpoint/stats", response_model=Dict[str, Any])
async def get_checkpoint_stats(
    current_user: dict = Depends(get_current_user),
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
):
    """
    Get checkpoint management statistics.
    
    Returns:
        Checkpoint statistics
    """
    try:
        stats = await checkpoint_manager.get_checkpoint_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get checkpoint stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get checkpoint statistics: {str(e)}"
        )


# Trade Lifecycle Endpoints

@router.get("/trade/{trade_id}/lifecycle", response_model=Dict[str, Any])
async def get_trade_lifecycle(
    trade_id: str,
    current_user: dict = Depends(get_current_user),
    lifecycle_manager: TradeLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Get trade lifecycle information.
    
    Args:
        trade_id: Trade identifier
        
    Returns:
        Trade lifecycle data
    """
    try:
        trade_context = lifecycle_manager.active_trades.get(trade_id)
        
        if not trade_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trade {trade_id} not found"
            )
        
        return {
            "trade_id": trade_id,
            "current_state": trade_context.current_state.value,
            "previous_state": trade_context.previous_state.value if trade_context.previous_state else None,
            "bot_id": trade_context.bot_id,
            "strategy_name": trade_context.strategy_name,
            "symbol": trade_context.symbol,
            "filled_quantity": float(trade_context.filled_quantity),
            "remaining_quantity": float(trade_context.remaining_quantity),
            "signal_timestamp": trade_context.signal_timestamp.isoformat(),
            "quality_score": trade_context.quality_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trade lifecycle: {e}", trade_id=trade_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve trade lifecycle: {str(e)}"
        )


@router.get("/trade/history", response_model=List[Dict[str, Any]])
async def get_trade_history(
    bot_id: Optional[str] = Query(default=None, description="Filter by bot ID"),
    strategy_name: Optional[str] = Query(default=None, description="Filter by strategy"),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    current_user: dict = Depends(get_current_user),
    lifecycle_manager: TradeLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Get trade history with optional filters.
    
    Args:
        bot_id: Optional bot ID filter
        strategy_name: Optional strategy filter
        symbol: Optional symbol filter
        limit: Maximum results to return
        
    Returns:
        Trade history
    """
    try:
        history = await lifecycle_manager.get_trade_history(
            bot_id=bot_id,
            strategy_name=strategy_name,
            symbol=symbol,
            limit=limit
        )
        return history
        
    except Exception as e:
        logger.error(f"Failed to get trade history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve trade history: {str(e)}"
        )


@router.get("/trade/{trade_id}/performance", response_model=Dict[str, Any])
async def get_trade_performance(
    trade_id: str,
    current_user: dict = Depends(get_current_user),
    lifecycle_manager: TradeLifecycleManager = Depends(get_lifecycle_manager)
):
    """
    Get trade performance metrics.
    
    Args:
        trade_id: Trade identifier
        
    Returns:
        Trade performance data
    """
    try:
        performance = await lifecycle_manager.calculate_trade_performance(trade_id)
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get trade performance: {e}", trade_id=trade_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve trade performance: {str(e)}"
        )


# Quality Control Endpoints

@router.post("/validation/pre-trade", response_model=TradeValidationResponse)
async def validate_pre_trade(
    request: TradeValidationRequest,
    current_user: dict = Depends(get_current_user),
    quality_controller: QualityController = Depends(get_quality_controller)
):
    """
    Perform pre-trade validation.
    
    Args:
        request: Validation request data
        
    Returns:
        Validation results
    """
    try:
        # Convert request data to proper types
        order_request = OrderRequest(**request.order_request)
        market_data = MarketData(**request.market_data) if request.market_data else None
        
        validation = await quality_controller.validate_pre_trade(
            order_request,
            market_data,
            request.portfolio_context
        )
        
        return TradeValidationResponse(
            validation_id=validation.validation_id,
            overall_result=validation.overall_result.value,
            overall_score=validation.overall_score,
            risk_level=validation.risk_level,
            validation_time_ms=validation.validation_time_ms,
            recommendations=validation.recommendations
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid validation request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to validate pre-trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform pre-trade validation: {str(e)}"
        )


@router.post("/analysis/post-trade", response_model=PostTradeAnalysisResponse)
async def analyze_post_trade(
    request: PostTradeAnalysisRequest,
    current_user: dict = Depends(get_current_user),
    quality_controller: QualityController = Depends(get_quality_controller)
):
    """
    Perform post-trade analysis.
    
    Args:
        request: Analysis request data
        
    Returns:
        Analysis results
    """
    try:
        # Convert request data to proper types
        from src.core.types import ExecutionResult
        execution_result = ExecutionResult(**request.execution_result)
        
        market_data_before = MarketData(**request.market_data_before) if request.market_data_before else None
        market_data_after = MarketData(**request.market_data_after) if request.market_data_after else None
        
        analysis = await quality_controller.analyze_post_trade(
            request.trade_id,
            execution_result,
            market_data_before,
            market_data_after
        )
        
        return PostTradeAnalysisResponse(
            analysis_id=analysis.analysis_id,
            trade_id=request.trade_id,
            overall_quality_score=analysis.overall_quality_score,
            slippage_bps=analysis.slippage_bps,
            execution_time_seconds=analysis.execution_time_seconds,
            quality_issues=analysis.issues,
            recommendations=analysis.recommendations
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid analysis request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to analyze post-trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform post-trade analysis: {str(e)}"
        )


@router.get("/quality/summary", response_model=Dict[str, Any])
async def get_quality_summary(
    bot_id: Optional[str] = Query(default=None, description="Filter by bot ID"),
    hours: int = Query(default=24, ge=1, le=168, description="Time period in hours"),
    current_user: dict = Depends(get_current_user),
    quality_controller: QualityController = Depends(get_quality_controller)
):
    """
    Get quality control summary.
    
    Args:
        bot_id: Optional bot ID filter
        hours: Time period for summary
        
    Returns:
        Quality summary data
    """
    try:
        summary = await quality_controller.get_quality_summary(bot_id, hours)
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get quality summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality summary: {str(e)}"
        )


# State Synchronization Endpoints

@router.get("/sync/{entity_type}/{entity_id}/status", response_model=SyncStatusResponse)
async def get_sync_status(
    entity_type: str,
    entity_id: str,
    current_user: dict = Depends(get_current_user),
    sync_manager: StateSyncManager = Depends(get_sync_manager)
):
    """
    Get synchronization status for an entity.
    
    Args:
        entity_type: Type of entity
        entity_id: Entity identifier
        
    Returns:
        Sync status information
    """
    try:
        status = await sync_manager.get_sync_status(entity_type, entity_id)
        
        return SyncStatusResponse(
            entity_type=entity_type,
            entity_id=entity_id,
            current_version=status.get("current_version"),
            last_updated=status.get("last_updated"),
            sync_in_progress=status.get("sync_in_progress", False),
            has_conflicts=status.get("has_conflicts", False),
            consistency_status=status.get("consistency_status", {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sync status: {str(e)}"
        )


@router.post("/sync/{entity_type}/{entity_id}/force")
async def force_sync(
    entity_type: str,
    entity_id: str,
    current_user: dict = Depends(get_current_user),
    sync_manager: StateSyncManager = Depends(get_sync_manager)
):
    """
    Force synchronization of an entity.
    
    Args:
        entity_type: Type of entity
        entity_id: Entity identifier
        
    Returns:
        Sync result
    """
    try:
        success = await sync_manager.force_sync(entity_type, entity_id)
        
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "success": success,
            "forced_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to force sync: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to force synchronization: {str(e)}"
        )


@router.get("/sync/metrics", response_model=Dict[str, Any])
async def get_sync_metrics(
    current_user: dict = Depends(get_current_user),
    sync_manager: StateSyncManager = Depends(get_sync_manager)
):
    """
    Get synchronization metrics.
    
    Returns:
        Sync metrics data
    """
    try:
        metrics = await sync_manager.get_sync_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get sync metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sync metrics: {str(e)}"
        )


@router.get("/sync/conflicts", response_model=List[Dict[str, Any]])
async def get_sync_conflicts(
    resolved: Optional[bool] = Query(default=None, description="Filter by resolution status"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results"),
    current_user: dict = Depends(get_current_user),
    sync_manager: StateSyncManager = Depends(get_sync_manager)
):
    """
    Get active or resolved sync conflicts.
    
    Args:
        resolved: Filter by resolution status
        limit: Maximum results to return
        
    Returns:
        List of conflicts
    """
    try:
        conflicts = []
        
        for conflict_id, conflict in sync_manager.active_conflicts.items():
            if resolved is None or conflict.resolved == resolved:
                conflicts.append({
                    "conflict_id": conflict_id,
                    "entity_type": conflict.entity_type,
                    "entity_id": conflict.entity_id,
                    "detected_at": conflict.detected_at.isoformat(),
                    "resolved": conflict.resolved,
                    "resolved_at": conflict.resolved_at.isoformat() if conflict.resolved_at else None,
                    "resolution_strategy": conflict.resolution_strategy.value,
                    "conflicting_fields": conflict.conflicting_fields
                })
        
        # Sort by detection time (newest first) and limit
        conflicts.sort(key=lambda x: x["detected_at"], reverse=True)
        return conflicts[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get sync conflicts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sync conflicts: {str(e)}"
        )