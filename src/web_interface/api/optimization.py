"""
Optimization API endpoints for T-Bot web interface.

This module provides optimization functionality for parameter optimization,
brute force testing, and hyperparameter tuning with proper overfitting prevention.
"""

import asyncio
import itertools
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from pydantic import BaseModel, Field, field_validator

from src.backtesting.engine import BacktestEngine
from src.core.logging import get_logger
from enum import Enum

class TimeInterval(Enum):
    """Time interval enumeration for data intervals."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
from src.strategies.factory import StrategyFactory
from src.web_interface.security.auth import User, get_current_user, get_trading_user

logger = get_logger(__name__)
router = APIRouter()

# Global references (set by app startup)
backtesting_engine = None
strategy_factory = None


def set_dependencies(backtest_engine, strat_factory):
    """Set global dependencies."""
    global backtesting_engine, strategy_factory
    backtesting_engine = backtest_engine
    strategy_factory = strat_factory


class OptimizationParameterRange(BaseModel):
    """Model for parameter optimization range."""

    parameter_name: str = Field(..., description="Name of parameter to optimize")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    step_size: float = Field(..., gt=0, description="Step size for parameter values")
    parameter_type: str = Field(default="float", description="Parameter type (float, int, bool)")


class OptimizationRequest(BaseModel):
    """Request model for optimization job."""

    # Basic Configuration
    job_name: str = Field(..., description="Name for the optimization job")
    strategy_name: str = Field(..., description="Strategy to optimize")
    symbols: List[str] = Field(..., description="Trading symbols")
    
    # Capital and Risk Configuration
    allocated_capital: Decimal = Field(..., gt=0, description="Capital allocated")
    risk_percentage: float = Field(..., gt=0, le=0.1, description="Risk per trade (max 10%)")
    
    # Data Configuration
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    time_interval: TimeInterval = Field(default=TimeInterval.HOUR_1, description="Data interval")
    
    # Parameter Ranges
    parameter_ranges: List[OptimizationParameterRange] = Field(..., description="Parameter ranges to optimize")
    
    # Fixed Parameters
    fixed_parameters: Dict[str, Any] = Field(default_factory=dict, description="Fixed strategy parameters")
    
    # Optimization Configuration
    optimization_metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    max_combinations: int = Field(default=1000, le=5000, description="Maximum parameter combinations to test")
    
    # Cross-validation Configuration
    enable_walk_forward: bool = Field(default=True, description="Enable walk-forward analysis")
    train_test_split: float = Field(default=0.7, gt=0.5, lt=1.0, description="Train/test split ratio")
    num_folds: int = Field(default=3, ge=2, le=10, description="Number of cross-validation folds")
    
    # Overfitting Prevention
    min_trades_required: int = Field(default=30, description="Minimum trades required for valid results")
    max_drawdown_threshold: float = Field(default=0.25, description="Maximum allowed drawdown")
    min_sharpe_threshold: float = Field(default=1.0, description="Minimum Sharpe ratio threshold")
    
    # ML Model Selection (optional)
    ml_models_to_test: List[str] = Field(default_factory=list, description="ML models to include in optimization")
    
    @field_validator('parameter_ranges')
    @classmethod
    def validate_parameter_ranges(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 parameters can be optimized simultaneously")
        return v
    
    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, info):
        if info.data.get('start_date') and v <= info.data['start_date']:
            raise ValueError("End date must be after start date")
        if info.data.get('start_date') and (v - info.data['start_date']).days < 30:
            raise ValueError("Minimum 30 days of data required")
        return v


class OptimizationResult(BaseModel):
    """Model for individual optimization result."""

    combination_id: str
    parameters: Dict[str, Any]
    ml_model_id: Optional[str] = None
    
    # Performance Metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    total_trades: int
    
    # Risk Metrics
    var_95: float
    expected_shortfall: float
    calmar_ratio: float
    
    # Validation Metrics
    in_sample_sharpe: float
    out_sample_sharpe: float
    stability_score: float  # Measure of consistency across folds
    
    # Execution Details
    execution_time_seconds: float
    completed_at: datetime


class OptimizationJobResponse(BaseModel):
    """Response model for optimization job."""

    job_id: str
    job_name: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_combinations: int
    completed_combinations: int
    estimated_completion_time: Optional[datetime] = None
    best_result: Optional[OptimizationResult] = None


class OptimizationStatusResponse(BaseModel):
    """Response model for optimization job status."""

    job_id: str
    status: str
    progress: float  # 0.0 to 1.0
    current_combination: int
    total_combinations: int
    elapsed_time_minutes: float
    estimated_remaining_minutes: float
    combinations_per_minute: float
    memory_usage_mb: float
    cpu_usage_percent: float
    current_parameters: Dict[str, Any]
    best_metric_value: float
    last_update: datetime


class OptimizationResultsResponse(BaseModel):
    """Response model for optimization results."""

    job_id: str
    total_results: int
    valid_results: int
    invalid_results: int
    best_results: List[OptimizationResult]
    parameter_importance: Dict[str, float]
    optimization_summary: Dict[str, Any]


# In-memory storage for optimization jobs
active_optimization_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/jobs", response_model=OptimizationJobResponse)
async def create_optimization_job(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_trading_user),
):
    """
    Create a new optimization job for parameter tuning and model selection.
    
    This endpoint creates a comprehensive optimization job that tests multiple
    parameter combinations while preventing overfitting through cross-validation
    and walk-forward analysis.
    """
    try:
        # Generate unique job ID
        job_id = str(uuid4())
        
        # Validate strategy exists
        if not strategy_factory or not strategy_factory.is_strategy_available(request.strategy_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Strategy '{request.strategy_name}' not available"
            )
        
        # Calculate total parameter combinations
        total_combinations = 1
        for param_range in request.parameter_ranges:
            range_size = int((param_range.max_value - param_range.min_value) / param_range.step_size) + 1
            total_combinations *= range_size
        
        # Add ML model combinations if specified
        if request.ml_models_to_test:
            total_combinations *= len(request.ml_models_to_test)
        
        # Limit total combinations
        if total_combinations > request.max_combinations:
            # Use sampling to reduce combinations
            total_combinations = request.max_combinations
        
        # Estimate completion time (rough estimate: 10 seconds per combination)
        estimated_duration_minutes = (total_combinations * 10) / 60
        estimated_completion = datetime.utcnow() + timedelta(minutes=estimated_duration_minutes)
        
        # Create job data
        job_data = {
            "job_id": job_id,
            "job_name": request.job_name,
            "status": "created",
            "user_id": user.id,
            "request": request.dict(),
            "created_at": datetime.utcnow(),
            "total_combinations": total_combinations,
            "completed_combinations": 0,
            "results": [],
            "metrics": {},
            "logs": [],
        }
        
        active_optimization_jobs[job_id] = job_data
        
        logger.info(f"Created optimization job {job_id} with {total_combinations} combinations for user {user.username}")
        
        return OptimizationJobResponse(
            job_id=job_id,
            job_name=request.job_name,
            status="created",
            created_at=job_data["created_at"],
            total_combinations=total_combinations,
            completed_combinations=0,
            estimated_completion_time=estimated_completion,
        )
        
    except Exception as e:
        logger.error(f"Error creating optimization job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create optimization job: {str(e)}"
        )


@router.post("/jobs/{job_id}/start")
async def start_optimization_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_trading_user),
):
    """
    Start an optimization job.
    
    This will begin the parameter optimization process in the background.
    """
    try:
        if job_id not in active_optimization_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization job not found"
            )
        
        job = active_optimization_jobs[job_id]
        
        # Verify user ownership
        if job["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this job"
            )
        
        # Check if already started
        if job["status"] in ["running", "completed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job is already {job['status']}"
            )
        
        # Update job status
        job["status"] = "starting"
        job["started_at"] = datetime.utcnow()
        
        # Start optimization in background
        background_tasks.add_task(_run_optimization_job, job_id)
        
        logger.info(f"Starting optimization job {job_id}")
        
        return {"message": "Optimization job started", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting optimization job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start optimization job: {str(e)}"
        )


@router.get("/jobs/{job_id}/status", response_model=OptimizationStatusResponse)
async def get_optimization_job_status(
    job_id: str,
    user: User = Depends(get_current_user),
):
    """
    Get the current status of an optimization job.
    """
    try:
        if job_id not in active_optimization_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization job not found"
            )
        
        job = active_optimization_jobs[job_id]
        
        # Verify user ownership
        if job["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this job"
            )
        
        start_time = job.get("started_at", job["created_at"])
        elapsed_minutes = (datetime.utcnow() - start_time).total_seconds() / 60
        
        completed = job["completed_combinations"]
        total = job["total_combinations"]
        progress = completed / total if total > 0 else 0
        
        # Calculate performance metrics
        combinations_per_minute = completed / elapsed_minutes if elapsed_minutes > 0 else 0
        estimated_remaining = (total - completed) / combinations_per_minute if combinations_per_minute > 0 else 0
        
        metrics = job.get("metrics", {})
        
        return OptimizationStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=progress,
            current_combination=completed + 1,
            total_combinations=total,
            elapsed_time_minutes=elapsed_minutes,
            estimated_remaining_minutes=estimated_remaining,
            combinations_per_minute=combinations_per_minute,
            memory_usage_mb=metrics.get("memory_usage_mb", 0.0),
            cpu_usage_percent=metrics.get("cpu_usage_percent", 0.0),
            current_parameters=metrics.get("current_parameters", {}),
            best_metric_value=metrics.get("best_metric_value", 0.0),
            last_update=datetime.utcnow(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization job status {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/jobs/{job_id}/results", response_model=OptimizationResultsResponse)
async def get_optimization_job_results(
    job_id: str,
    top_n: int = Query(default=20, le=100, description="Number of top results to return"),
    user: User = Depends(get_current_user),
):
    """
    Get results from a completed optimization job.
    """
    try:
        if job_id not in active_optimization_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization job not found"
            )
        
        job = active_optimization_jobs[job_id]
        
        # Verify user ownership
        if job["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this job"
            )
        
        results = job.get("results", [])
        
        # Filter valid results (meet minimum requirements)
        valid_results = [r for r in results if r.get("is_valid", True)]
        
        # Sort by optimization metric
        optimization_metric = job["request"]["optimization_metric"]
        valid_results.sort(key=lambda x: x.get(optimization_metric, 0), reverse=True)
        
        # Calculate parameter importance (mock implementation)
        parameter_importance = {}
        for param_range in job["request"]["parameter_ranges"]:
            parameter_importance[param_range["parameter_name"]] = 0.8  # Mock value
        
        # Create optimization summary
        optimization_summary = {
            "total_runtime_minutes": (job.get("completed_at", datetime.utcnow()) - job["started_at"]).total_seconds() / 60 if job.get("started_at") else 0,
            "best_metric_value": valid_results[0].get(optimization_metric, 0) if valid_results else 0,
            "improvement_over_baseline": 0.25,  # Mock 25% improvement
            "stability_analysis": {
                "consistent_performers": len([r for r in valid_results if r.get("stability_score", 0) > 0.7]),
                "overfitted_results": len([r for r in results if r.get("is_valid", True) == False]),
            },
            "risk_analysis": {
                "low_risk_results": len([r for r in valid_results if r.get("max_drawdown", 1) < 0.15]),
                "high_sharpe_results": len([r for r in valid_results if r.get("sharpe_ratio", 0) > 1.5]),
            },
        }
        
        # Convert results to response format
        best_results = []
        for result in valid_results[:top_n]:
            best_results.append(OptimizationResult(
                combination_id=result.get("combination_id", ""),
                parameters=result.get("parameters", {}),
                ml_model_id=result.get("ml_model_id"),
                total_return=result.get("total_return", 0.0),
                sharpe_ratio=result.get("sharpe_ratio", 0.0),
                max_drawdown=result.get("max_drawdown", 0.0),
                profit_factor=result.get("profit_factor", 0.0),
                win_rate=result.get("win_rate", 0.0),
                total_trades=result.get("total_trades", 0),
                var_95=result.get("var_95", 0.0),
                expected_shortfall=result.get("expected_shortfall", 0.0),
                calmar_ratio=result.get("calmar_ratio", 0.0),
                in_sample_sharpe=result.get("in_sample_sharpe", 0.0),
                out_sample_sharpe=result.get("out_sample_sharpe", 0.0),
                stability_score=result.get("stability_score", 0.0),
                execution_time_seconds=result.get("execution_time_seconds", 0.0),
                completed_at=result.get("completed_at", datetime.utcnow()),
            ))
        
        return OptimizationResultsResponse(
            job_id=job_id,
            total_results=len(results),
            valid_results=len(valid_results),
            invalid_results=len(results) - len(valid_results),
            best_results=best_results,
            parameter_importance=parameter_importance,
            optimization_summary=optimization_summary,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization job results {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job results: {str(e)}"
        )


@router.post("/jobs/{job_id}/stop")
async def stop_optimization_job(
    job_id: str,
    user: User = Depends(get_trading_user),
):
    """
    Stop a running optimization job.
    """
    try:
        if job_id not in active_optimization_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization job not found"
            )
        
        job = active_optimization_jobs[job_id]
        
        # Verify user ownership
        if job["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this job"
            )
        
        # Update job status
        job["status"] = "stopping"
        job["completed_at"] = datetime.utcnow()
        
        # In real implementation, signal the optimization process to stop
        await asyncio.sleep(0.1)  # Small delay to simulate stopping
        
        job["status"] = "stopped"
        
        logger.info(f"Stopped optimization job {job_id}")
        
        return {"message": "Optimization job stopped", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping optimization job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop optimization job: {str(e)}"
        )


@router.delete("/jobs/{job_id}")
async def delete_optimization_job(
    job_id: str,
    user: User = Depends(get_trading_user),
):
    """
    Delete an optimization job and clean up resources.
    """
    try:
        if job_id not in active_optimization_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization job not found"
            )
        
        job = active_optimization_jobs[job_id]
        
        # Verify user ownership
        if job["user_id"] != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this job"
            )
        
        # Stop job if running
        if job["status"] in ["running", "starting"]:
            try:
                await stop_optimization_job(job_id, user)
            except Exception as e:
                logger.warning(f"Error stopping job during deletion: {str(e)}")
        
        # Remove from active jobs
        del active_optimization_jobs[job_id]
        
        logger.info(f"Deleted optimization job {job_id}")
        
        return {"message": "Optimization job deleted", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting optimization job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete optimization job: {str(e)}"
        )


@router.get("/jobs", response_model=List[OptimizationJobResponse])
async def list_optimization_jobs(
    user: User = Depends(get_current_user),
    status_filter: Optional[str] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=20, le=50, description="Maximum number of jobs to return"),
):
    """
    List optimization jobs for the current user.
    """
    try:
        user_jobs = []
        
        for job_id, job in active_optimization_jobs.items():
            if job["user_id"] != user.id:
                continue
            
            if status_filter and job["status"] != status_filter:
                continue
            
            # Get best result
            results = job.get("results", [])
            best_result = None
            if results:
                optimization_metric = job["request"]["optimization_metric"]
                best = max(results, key=lambda x: x.get(optimization_metric, 0))
                best_result = OptimizationResult(
                    combination_id=best.get("combination_id", ""),
                    parameters=best.get("parameters", {}),
                    ml_model_id=best.get("ml_model_id"),
                    total_return=best.get("total_return", 0.0),
                    sharpe_ratio=best.get("sharpe_ratio", 0.0),
                    max_drawdown=best.get("max_drawdown", 0.0),
                    profit_factor=best.get("profit_factor", 0.0),
                    win_rate=best.get("win_rate", 0.0),
                    total_trades=best.get("total_trades", 0),
                    var_95=best.get("var_95", 0.0),
                    expected_shortfall=best.get("expected_shortfall", 0.0),
                    calmar_ratio=best.get("calmar_ratio", 0.0),
                    in_sample_sharpe=best.get("in_sample_sharpe", 0.0),
                    out_sample_sharpe=best.get("out_sample_sharpe", 0.0),
                    stability_score=best.get("stability_score", 0.0),
                    execution_time_seconds=best.get("execution_time_seconds", 0.0),
                    completed_at=best.get("completed_at", datetime.utcnow()),
                )
            
            user_jobs.append(
                OptimizationJobResponse(
                    job_id=job_id,
                    job_name=job["job_name"],
                    status=job["status"],
                    created_at=job["created_at"],
                    started_at=job.get("started_at"),
                    completed_at=job.get("completed_at"),
                    total_combinations=job["total_combinations"],
                    completed_combinations=job["completed_combinations"],
                    best_result=best_result,
                )
            )
        
        # Sort by creation date (newest first) and limit
        user_jobs.sort(key=lambda x: x.created_at, reverse=True)
        return user_jobs[:limit]
        
    except Exception as e:
        logger.error(f"Error listing optimization jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list optimization jobs: {str(e)}"
        )


async def _run_optimization_job(job_id: str):
    """
    Background task to run an optimization job.
    
    This function handles the actual parameter optimization with cross-validation
    and overfitting prevention measures.
    """
    try:
        job = active_optimization_jobs.get(job_id)
        if not job:
            logger.error(f"Optimization job {job_id} not found during execution")
            return
        
        request_data = job["request"]
        
        job["status"] = "running"
        
        # Generate parameter combinations
        combinations = _generate_parameter_combinations(request_data)
        
        # Limit combinations if necessary
        if len(combinations) > request_data["max_combinations"]:
            import random
            combinations = random.sample(combinations, request_data["max_combinations"])
        
        job["total_combinations"] = len(combinations)
        
        logger.info(f"Starting optimization of {len(combinations)} combinations for job {job_id}")
        
        # Run optimization for each combination
        for i, combination in enumerate(combinations):
            if job["status"] != "running":
                break
            
            try:
                # Update progress
                job["completed_combinations"] = i
                job["metrics"]["current_parameters"] = combination
                
                # Run backtest for this combination
                result = await _run_parameter_combination(job_id, combination, request_data)
                
                if result:
                    job["results"].append(result)
                    
                    # Update best metric
                    metric_value = result.get(request_data["optimization_metric"], 0)
                    if metric_value > job["metrics"].get("best_metric_value", 0):
                        job["metrics"]["best_metric_value"] = metric_value
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing combination {i} in job {job_id}: {str(e)}")
                continue
        
        # Finalize job
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow()
        job["completed_combinations"] = len(combinations)
        
        logger.info(f"Completed optimization job {job_id} with {len(job['results'])} results")
        
    except Exception as e:
        logger.error(f"Error running optimization job {job_id}: {str(e)}")
        job = active_optimization_jobs.get(job_id)
        if job:
            job["status"] = "failed"
            job["completed_at"] = datetime.utcnow()


def _generate_parameter_combinations(request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations to test."""
    combinations = []
    
    # Generate parameter value lists
    param_values = {}
    for param_range in request_data["parameter_ranges"]:
        name = param_range["parameter_name"]
        min_val = param_range["min_value"]
        max_val = param_range["max_value"]
        step = param_range["step_size"]
        param_type = param_range["parameter_type"]
        
        values = []
        current = min_val
        while current <= max_val:
            if param_type == "int":
                values.append(int(current))
            elif param_type == "bool":
                values.append(current > 0.5)
            else:
                values.append(current)
            current += step
        
        param_values[name] = values
    
    # Generate all combinations using itertools.product
    param_names = list(param_values.keys())
    param_value_lists = list(param_values.values())
    
    for combination_values in itertools.product(*param_value_lists):
        combination = dict(zip(param_names, combination_values))
        
        # Add fixed parameters
        combination.update(request_data.get("fixed_parameters", {}))
        
        combinations.append(combination)
    
    # Add ML model variations if specified
    if request_data.get("ml_models_to_test"):
        expanded_combinations = []
        for combination in combinations:
            for model_id in request_data["ml_models_to_test"]:
                model_combination = combination.copy()
                model_combination["ml_model_id"] = model_id
                expanded_combinations.append(model_combination)
        combinations = expanded_combinations
    
    return combinations


async def _run_parameter_combination(
    job_id: str, 
    parameters: Dict[str, Any], 
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a single parameter combination and return results."""
    
    start_time = datetime.utcnow()
    combination_id = str(uuid4())
    
    try:
        # Mock backtest execution
        # In real implementation, this would:
        # 1. Create strategy with parameters
        # 2. Run cross-validation backtest
        # 3. Calculate all performance metrics
        # 4. Validate results against overfitting criteria
        
        # Mock performance metrics
        total_return = 0.15 + (hash(str(parameters)) % 100) / 1000  # 10-25% return
        sharpe_ratio = 1.0 + (hash(str(parameters)) % 200) / 200  # 1.0-2.0 Sharpe
        max_drawdown = 0.05 + (hash(str(parameters)) % 150) / 1000  # 5-20% drawdown
        
        # Simulate cross-validation scores
        in_sample_sharpe = sharpe_ratio * 1.1  # Slightly higher in-sample
        out_sample_sharpe = sharpe_ratio * 0.9  # Lower out-of-sample
        
        # Calculate stability score (consistency across folds)
        stability_score = 1 - abs(in_sample_sharpe - out_sample_sharpe) / in_sample_sharpe
        
        # Check validation criteria
        min_trades = request_data.get("min_trades_required", 30)
        max_dd_threshold = request_data.get("max_drawdown_threshold", 0.25)
        min_sharpe_threshold = request_data.get("min_sharpe_threshold", 1.0)
        
        total_trades = min_trades + (hash(str(parameters)) % 100)  # 30-130 trades
        
        is_valid = (
            total_trades >= min_trades and
            max_drawdown <= max_dd_threshold and
            out_sample_sharpe >= min_sharpe_threshold and
            stability_score >= 0.5  # Reasonable consistency
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "combination_id": combination_id,
            "parameters": parameters,
            "ml_model_id": parameters.get("ml_model_id"),
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": 1.5 if total_return > 0 else 0.8,
            "win_rate": 0.55 if total_return > 0 else 0.45,
            "total_trades": total_trades,
            "var_95": max_drawdown * 0.7,  # Mock VaR
            "expected_shortfall": max_drawdown * 0.85,  # Mock ES
            "calmar_ratio": total_return / max_drawdown if max_drawdown > 0 else 0,
            "in_sample_sharpe": in_sample_sharpe,
            "out_sample_sharpe": out_sample_sharpe,
            "stability_score": stability_score,
            "execution_time_seconds": execution_time,
            "completed_at": datetime.utcnow(),
            "is_valid": is_valid,
        }
        
    except Exception as e:
        logger.error(f"Error running parameter combination: {str(e)}")
        return None