"""
Machine Learning Models API endpoints for T-Bot web interface.

This module provides ML model management, training, deployment, and inference
functionality for AI-powered trading strategies.
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
model_manager = None


def set_dependencies(manager):
    """Set global dependencies."""
    global model_manager
    model_manager = manager


class ModelResponse(BaseModel):
    """Response model for ML model information."""

    model_id: str
    model_name: str
    model_type: str
    version: str
    description: str | None = None
    status: str
    deployment_stage: str
    accuracy: Decimal | None = None
    precision: Decimal | None = None
    recall: Decimal | None = None
    f1_score: Decimal | None = None
    training_samples: int | None = None
    validation_samples: int | None = None
    features_count: int | None = None
    target_symbol: str | None = None
    created_at: datetime
    updated_at: datetime
    last_prediction_at: datetime | None = None
    prediction_count: int | None = None


class CreateModelRequest(BaseModel):
    """Request model for creating a new ML model."""

    model_name: str = Field(..., min_length=1, max_length=100)
    model_type: str = Field(
        ...,
        description="Model type: price_predictor, direction_classifier, volatility_forecaster, regime_detector",
    )
    target_symbol: str = Field(..., description="Target trading symbol")
    description: str | None = Field(None, max_length=500)
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific parameters"
    )
    training_config: dict[str, Any] = Field(
        default_factory=dict, description="Training configuration"
    )


class TrainModelRequest(BaseModel):
    """Request model for training a model."""

    start_date: datetime = Field(..., description="Training data start date")
    end_date: datetime = Field(..., description="Training data end date")
    validation_split: Decimal = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation data split"
    )
    test_split: Decimal = Field(
        default=Decimal("0.1"), ge=Decimal("0.1"), le=Decimal("0.3"), description="Test data split"
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )


class PredictionRequest(BaseModel):
    """Request model for model predictions."""

    symbol: str = Field(..., description="Symbol to predict")
    features: dict[str, Decimal] = Field(..., description="Input features")
    prediction_horizon: int | None = Field(
        default=1, description="Prediction horizon in time units"
    )


class PredictionResponse(BaseModel):
    """Response model for model predictions."""

    model_id: str
    model_name: str
    symbol: str
    prediction: Decimal
    confidence: Decimal
    prediction_type: str
    prediction_horizon: int
    features_used: list[str]
    created_at: datetime
    metadata: dict[str, Any] | None = None


class TrainingJobResponse(BaseModel):
    """Response model for training job status."""

    job_id: str
    model_id: str
    model_name: str
    status: str
    progress: Decimal
    started_at: datetime
    estimated_completion: datetime | None = None
    completed_at: datetime | None = None
    training_samples: int | None = None
    validation_samples: int | None = None
    current_epoch: int | None = None
    total_epochs: int | None = None
    current_loss: Decimal | None = None
    best_accuracy: Decimal | None = None
    error_message: str | None = None


class ModelPerformanceResponse(BaseModel):
    """Response model for model performance metrics."""

    model_id: str
    model_name: str
    evaluation_period: str
    accuracy: Decimal
    precision: Decimal
    recall: Decimal
    f1_score: Decimal
    auc_score: Decimal | None = None
    confusion_matrix: list[list[int]] | None = None
    feature_importance: dict[str, Decimal] | None = None
    prediction_distribution: dict[str, int] | None = None
    error_analysis: dict[str, Any] | None = None
    drift_detected: bool
    last_evaluated: datetime


class DeploymentRequest(BaseModel):
    """Request model for model deployment."""

    deployment_stage: str = Field(
        ..., description="Deployment stage: development, staging, production"
    )
    auto_fallback: bool = Field(default=True, description="Enable automatic fallback on errors")
    performance_threshold: Decimal | None = Field(None, description="Minimum performance threshold")


@router.get("/", response_model=list[ModelResponse])
async def list_models(
    model_type: str | None = Query(None, description="Filter by model type"),
    status: str | None = Query(None, description="Filter by status"),
    deployment_stage: str | None = Query(None, description="Filter by deployment stage"),
    current_user: User = Depends(get_current_user),
):
    """
    List all ML models with optional filtering.

    Args:
        model_type: Optional model type filter
        status: Optional status filter
        deployment_stage: Optional deployment stage filter
        current_user: Current authenticated user

    Returns:
        List[ModelResponse]: List of ML models

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Mock model data (in production, get from model registry)
        mock_models = [
            {
                "model_id": "model_001",
                "model_name": "btc_price_predictor_v2",
                "model_type": "price_predictor",
                "version": "2.1.0",
                "description": "LSTM-based Bitcoin price predictor with attention mechanism",
                "status": "active",
                "deployment_stage": "production",
                "accuracy": 0.78,
                "precision": 0.82,
                "recall": 0.74,
                "f1_score": 0.78,
                "training_samples": 50000,
                "validation_samples": 10000,
                "features_count": 25,
                "target_symbol": "BTCUSDT",
                "created_at": datetime.now(timezone.utc) - timedelta(days=30),
                "updated_at": datetime.now(timezone.utc) - timedelta(days=2),
                "last_prediction_at": datetime.now(timezone.utc) - timedelta(minutes=5),
                "prediction_count": 12543,
            },
            {
                "model_id": "model_002",
                "model_name": "eth_direction_classifier",
                "model_type": "direction_classifier",
                "version": "1.3.0",
                "description": "Random Forest classifier for ETH price direction",
                "status": "active",
                "deployment_stage": "production",
                "accuracy": 0.65,
                "precision": 0.68,
                "recall": 0.62,
                "f1_score": 0.65,
                "training_samples": 75000,
                "validation_samples": 15000,
                "features_count": 18,
                "target_symbol": "ETHUSDT",
                "created_at": datetime.now(timezone.utc) - timedelta(days=45),
                "updated_at": datetime.now(timezone.utc) - timedelta(days=5),
                "last_prediction_at": datetime.now(timezone.utc) - timedelta(minutes=2),
                "prediction_count": 8921,
            },
            {
                "model_id": "model_003",
                "model_name": "volatility_forecaster_ensemble",
                "model_type": "volatility_forecaster",
                "version": "1.0.0",
                "description": "Ensemble model for crypto volatility forecasting",
                "status": "training",
                "deployment_stage": "development",
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "training_samples": 40000,
                "validation_samples": 8000,
                "features_count": 32,
                "target_symbol": "BTCUSDT",
                "created_at": datetime.now(timezone.utc) - timedelta(days=3),
                "updated_at": datetime.now(timezone.utc) - timedelta(hours=2),
                "last_prediction_at": None,
                "prediction_count": 0,
            },
            {
                "model_id": "model_004",
                "model_name": "market_regime_detector",
                "model_type": "regime_detector",
                "version": "2.0.0",
                "description": "HMM-based market regime classification",
                "status": "active",
                "deployment_stage": "staging",
                "accuracy": 0.71,
                "precision": 0.73,
                "recall": 0.69,
                "f1_score": 0.71,
                "training_samples": 100000,
                "validation_samples": 20000,
                "features_count": 15,
                "target_symbol": None,  # Multi-symbol model
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                "updated_at": datetime.now(timezone.utc) - timedelta(days=10),
                "last_prediction_at": datetime.now(timezone.utc) - timedelta(minutes=15),
                "prediction_count": 3456,
            },
        ]

        # Apply filters
        filtered_models = []
        for model_data in mock_models:
            # Apply model type filter
            if model_type and model_data["model_type"] != model_type:
                continue

            # Apply status filter
            if status and model_data["status"] != status:
                continue

            # Apply deployment stage filter
            if deployment_stage and model_data["deployment_stage"] != deployment_stage:
                continue

            model = ModelResponse(**model_data)
            filtered_models.append(model)

        return filtered_models

    except Exception as e:
        logger.error(f"Models listing failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {e!s}",
        )


@router.post("/", response_model=dict[str, Any])
async def create_model(
    model_request: CreateModelRequest, current_user: User = Depends(get_admin_user)
):
    """
    Create a new ML model (admin only).

    Args:
        model_request: Model creation parameters
        current_user: Current admin user

    Returns:
        Dict: Model creation result

    Raises:
        HTTPException: If creation fails
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        # Mock model creation (in production, use actual model manager)
        import uuid

        model_id = f"model_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Model creation started",
            model_id=model_id,
            model_name=model_request.model_name,
            model_type=model_request.model_type,
            admin=current_user.username,
        )

        return {
            "success": True,
            "message": "Model created successfully",
            "model_id": model_id,
            "model_name": model_request.model_name,
            "model_type": model_request.model_type,
            "status": "created",
            "created_by": current_user.username,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Model creation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model creation failed: {e!s}"
        )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, current_user: User = Depends(get_current_user)):
    """
    Get detailed information about a specific model.

    Args:
        model_id: Model identifier
        current_user: Current authenticated user

    Returns:
        ModelResponse: Model details

    Raises:
        HTTPException: If model not found
    """
    try:
        # Mock model lookup (in production, get from model registry)
        if model_id == "model_001":
            model_data = {
                "model_id": "model_001",
                "model_name": "btc_price_predictor_v2",
                "model_type": "price_predictor",
                "version": "2.1.0",
                "description": "Advanced LSTM-based Bitcoin price predictor with attention mechanism and multi-timeframe analysis",
                "status": "active",
                "deployment_stage": "production",
                "accuracy": 0.78,
                "precision": 0.82,
                "recall": 0.74,
                "f1_score": 0.78,
                "training_samples": 50000,
                "validation_samples": 10000,
                "features_count": 25,
                "target_symbol": "BTCUSDT",
                "created_at": datetime.now(timezone.utc) - timedelta(days=30),
                "updated_at": datetime.now(timezone.utc) - timedelta(days=2),
                "last_prediction_at": datetime.now(timezone.utc) - timedelta(minutes=5),
                "prediction_count": 12543,
            }
            return ModelResponse(**model_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Model not found: {model_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model retrieval failed: {e}", model_id=model_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get model: {e!s}"
        )


@router.post("/{model_id}/train", response_model=TrainingJobResponse)
async def train_model(
    model_id: str, training_request: TrainModelRequest, current_user: User = Depends(get_admin_user)
):
    """
    Start training a model (admin only).

    Args:
        model_id: Model identifier
        training_request: Training parameters
        current_user: Current admin user

    Returns:
        TrainingJobResponse: Training job status

    Raises:
        HTTPException: If training start fails
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        # Mock training job start (in production, queue training job)
        import uuid

        job_id = f"job_{uuid.uuid4().hex[:8]}"

        # Calculate training samples
        training_days = (training_request.end_date - training_request.start_date).days
        training_samples = training_days * 1440  # 1 sample per minute
        validation_samples = int(training_samples * training_request.validation_split)

        job_response = TrainingJobResponse(
            job_id=job_id,
            model_id=model_id,
            model_name="btc_price_predictor_v2",  # Mock model name
            status="starting",
            progress=0.0,
            started_at=datetime.now(timezone.utc),
            estimated_completion=datetime.now(timezone.utc) + timedelta(hours=2),
            completed_at=None,
            training_samples=training_samples,
            validation_samples=validation_samples,
            current_epoch=0,
            total_epochs=100,
            current_loss=None,
            best_accuracy=None,
            error_message=None,
        )

        logger.info(
            "Model training started",
            job_id=job_id,
            model_id=model_id,
            training_samples=training_samples,
            admin=current_user.username,
        )

        return job_response

    except Exception as e:
        logger.error(
            f"Model training start failed: {e}", model_id=model_id, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model training start failed: {e!s}"
        )


@router.get("/{model_id}/training-jobs", response_model=list[TrainingJobResponse])
async def get_training_jobs(
    model_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of jobs to return"),
    current_user: User = Depends(get_current_user),
):
    """
    Get training job history for a model.

    Args:
        model_id: Model identifier
        limit: Maximum number of jobs to return
        current_user: Current authenticated user

    Returns:
        List[TrainingJobResponse]: Training job history

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Mock training job history (in production, get from database)
        mock_jobs = []

        for i in range(min(limit, 5)):  # Limit mock data
            job_id = f"job_{i + 1:03d}"
            status = ["completed", "failed", "running"][i % 3]

            start_time = datetime.now(timezone.utc) - timedelta(days=i * 7, hours=i * 2)
            completion_time = start_time + timedelta(hours=2) if status == "completed" else None

            mock_job = TrainingJobResponse(
                job_id=job_id,
                model_id=model_id,
                model_name="btc_price_predictor_v2",
                status=status,
                progress=(
                    100.0 if status == "completed" else (50.0 if status == "running" else 85.0)
                ),
                started_at=start_time,
                estimated_completion=(
                    start_time + timedelta(hours=2) if status == "running" else None
                ),
                completed_at=completion_time,
                training_samples=50000,
                validation_samples=10000,
                current_epoch=100 if status == "completed" else (50 if status == "running" else 85),
                total_epochs=100,
                current_loss=0.025 if status != "failed" else None,
                best_accuracy=(
                    0.78 if status == "completed" else (0.72 if status == "running" else None)
                ),
                error_message="CUDA out of memory" if status == "failed" else None,
            )
            mock_jobs.append(mock_job)

        return mock_jobs

    except Exception as e:
        logger.error(
            f"Training jobs retrieval failed: {e}", model_id=model_id, user=current_user.username
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training jobs: {e!s}",
        )


@router.post("/{model_id}/predict", response_model=PredictionResponse)
async def predict(
    model_id: str,
    prediction_request: PredictionRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Get prediction from a model.

    Args:
        model_id: Model identifier
        prediction_request: Prediction parameters
        current_user: Current authenticated user

    Returns:
        PredictionResponse: Model prediction

    Raises:
        HTTPException: If prediction fails
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        # Mock prediction (in production, use actual model inference)
        import random

        # Generate mock prediction based on model type
        if model_id == "model_001":  # Price predictor
            base_price = prediction_request.features.get("current_price", 45000.0)
            price_change = random.uniform(-0.05, 0.05)  # ±5% prediction
            prediction_value = base_price * (1 + price_change)
            prediction_type = "price"
            confidence = random.uniform(0.6, 0.9)
        elif model_id == "model_002":  # Direction classifier
            prediction_value = random.choice([0, 1])  # 0=down, 1=up
            prediction_type = "direction"
            confidence = random.uniform(0.55, 0.85)
        else:
            prediction_value = random.uniform(0, 1)
            prediction_type = "probability"
            confidence = random.uniform(0.5, 0.8)

        prediction_response = PredictionResponse(
            model_id=model_id,
            model_name="btc_price_predictor_v2",  # Mock model name
            symbol=prediction_request.symbol,
            prediction=prediction_value,
            confidence=confidence,
            prediction_type=prediction_type,
            prediction_horizon=prediction_request.prediction_horizon,
            features_used=list(prediction_request.features.keys()),
            created_at=datetime.now(timezone.utc),
            metadata={
                "model_version": "2.1.0",
                "feature_count": len(prediction_request.features),
                "preprocessing_time": 0.005,
                "inference_time": 0.012,
            },
        )

        logger.info(
            "Model prediction generated",
            model_id=model_id,
            symbol=prediction_request.symbol,
            prediction=prediction_value,
            confidence=confidence,
            user=current_user.username,
        )

        return prediction_response

    except Exception as e:
        logger.error(f"Model prediction failed: {e}", model_id=model_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model prediction failed: {e!s}"
        )


@router.get("/{model_id}/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(
    model_id: str,
    period: str = Query("30d", description="Evaluation period: 7d, 30d, 90d"),
    current_user: User = Depends(get_current_user),
):
    """
    Get model performance metrics.

    Args:
        model_id: Model identifier
        period: Evaluation period
        current_user: Current authenticated user

    Returns:
        ModelPerformanceResponse: Performance metrics

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Mock performance metrics (in production, calculate from actual predictions vs outcomes)
        performance = ModelPerformanceResponse(
            model_id=model_id,
            model_name="btc_price_predictor_v2",
            evaluation_period=period,
            accuracy=0.78,
            precision=0.82,
            recall=0.74,
            f1_score=0.78,
            auc_score=0.85,
            confusion_matrix=[[450, 120], [150, 480]],  # Mock confusion matrix
            feature_importance={
                "price_ma_20": 0.25,
                "volume_ma_10": 0.18,
                "rsi_14": 0.15,
                "macd_signal": 0.12,
                "bollinger_position": 0.10,
                "volume_ratio": 0.08,
                "price_momentum": 0.07,
                "volatility_20": 0.05,
            },
            prediction_distribution={"buy": 520, "sell": 480, "hold": 200},
            error_analysis={
                "mean_absolute_error": 0.023,
                "root_mean_squared_error": 0.035,
                "directional_accuracy": 0.72,
                "false_positive_rate": 0.12,
                "false_negative_rate": 0.16,
            },
            drift_detected=False,
            last_evaluated=datetime.now(timezone.utc),
        )

        return performance

    except Exception as e:
        logger.error(
            f"Model performance retrieval failed: {e}",
            model_id=model_id,
            user=current_user.username,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model performance: {e!s}",
        )


@router.post("/{model_id}/deploy", response_model=dict[str, Any])
async def deploy_model(
    model_id: str,
    deployment_request: DeploymentRequest,
    current_user: User = Depends(get_admin_user),
):
    """
    Deploy a model to specified stage (admin only).

    Args:
        model_id: Model identifier
        deployment_request: Deployment parameters
        current_user: Current admin user

    Returns:
        Dict: Deployment result

    Raises:
        HTTPException: If deployment fails
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        # Mock deployment (in production, use actual model manager)
        logger.info(
            "Model deployment started",
            model_id=model_id,
            deployment_stage=deployment_request.deployment_stage,
            admin=current_user.username,
        )

        return {
            "success": True,
            "message": f"Model deployed to {deployment_request.deployment_stage} successfully",
            "model_id": model_id,
            "deployment_stage": deployment_request.deployment_stage,
            "auto_fallback": deployment_request.auto_fallback,
            "deployed_by": current_user.username,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "health_check_url": f"/api/ml/{model_id}/health",
        }

    except Exception as e:
        logger.error(f"Model deployment failed: {e}", model_id=model_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model deployment failed: {e!s}"
        )


@router.delete("/{model_id}")
async def retire_model(
    model_id: str,
    reason: str = Query("replaced", description="Retirement reason"),
    current_user: User = Depends(get_admin_user),
):
    """
    Retire a model (admin only).

    Args:
        model_id: Model identifier
        reason: Retirement reason
        current_user: Current admin user

    Returns:
        Dict: Retirement result

    Raises:
        HTTPException: If retirement fails
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        # Mock model retirement (in production, use actual model manager)
        logger.info(
            "Model retirement started",
            model_id=model_id,
            reason=reason,
            admin=current_user.username,
        )

        return {
            "success": True,
            "message": f"Model {model_id} retired successfully",
            "model_id": model_id,
            "reason": reason,
            "retired_by": current_user.username,
            "retired_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Model retirement failed: {e}", model_id=model_id, user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model retirement failed: {e!s}"
        )


# Feature Engineering Endpoints


class FeatureEngineeringRequest(BaseModel):
    """Request model for feature engineering."""

    data_source: str = Field(..., description="Data source identifier")
    feature_types: list[str] = Field(..., description="Types of features to engineer")
    lookback_period: int = Field(default=30, description="Lookback period in days")
    target_symbols: list[str] = Field(..., description="Target symbols")
    advanced_features: bool = Field(default=False, description="Include advanced features")


class FeatureSelectionRequest(BaseModel):
    """Request model for feature selection."""

    feature_set: str = Field(..., description="Feature set identifier")
    selection_method: str = Field(default="mutual_info", description="Selection method")
    max_features: int = Field(default=50, description="Maximum number of features")
    target_variable: str = Field(..., description="Target variable name")


@router.post("/features/engineer", response_model=dict)
async def engineer_features(
    request: FeatureEngineeringRequest, current_user: User = Depends(get_admin_user)
):
    """
    Engineer features for ML models (admin only).

    Args:
        request: Feature engineering parameters
        current_user: Current admin user

    Returns:
        Dict: Feature engineering results
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        # Mock feature engineering (in production, use actual feature engineering service)
        results = {
            "job_id": f"feature_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "data_source": request.data_source,
            "feature_types": request.feature_types,
            "lookback_period": request.lookback_period,
            "target_symbols": request.target_symbols,
            "features_generated": len(request.feature_types) * len(request.target_symbols) * 5,
            "advanced_features_included": request.advanced_features,
            "estimated_completion": datetime.now(timezone.utc) + timedelta(minutes=15),
            "status": "started",
        }

        logger.info(
            "Feature engineering started", user=current_user.username, job_id=results["job_id"]
        )
        return results

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature engineering failed: {e!s}",
        )


@router.post("/features/select", response_model=dict)
async def select_features(
    request: FeatureSelectionRequest, current_user: User = Depends(get_admin_user)
):
    """
    Select optimal features for ML models (admin only).

    Args:
        request: Feature selection parameters
        current_user: Current admin user

    Returns:
        Dict: Feature selection results
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        # Mock feature selection results
        selected_features = [f"feature_{i}" for i in range(1, min(request.max_features + 1, 51))]

        results = {
            "job_id": f"selection_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "feature_set": request.feature_set,
            "selection_method": request.selection_method,
            "target_variable": request.target_variable,
            "selected_features": selected_features,
            "feature_importance_scores": {
                feature: round(Decimal(str(1.0 - i * 0.01)), 4)
                for i, feature in enumerate(selected_features)
            },
            "cross_validation_score": Decimal("0.785"),
            "status": "completed",
        }

        logger.info(
            "Feature selection completed", user=current_user.username, job_id=results["job_id"]
        )
        return results

    except Exception as e:
        logger.error(f"Feature selection failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature selection failed: {e!s}",
        )


# A/B Testing Endpoints


class ABTestRequest(BaseModel):
    """Request model for A/B test creation."""

    test_name: str = Field(..., description="Test name")
    control_model: str = Field(..., description="Control model ID")
    treatment_model: str = Field(..., description="Treatment model ID")
    traffic_split: Decimal = Field(default=Decimal("0.5"), description="Treatment traffic split")
    target_symbols: list[str] = Field(..., description="Target trading symbols")
    duration_days: int = Field(default=7, description="Test duration in days")
    success_metrics: list[str] = Field(
        default=["accuracy", "profit"], description="Success metrics"
    )


@router.post("/ab-test/create", response_model=dict)
async def create_ab_test(request: ABTestRequest, current_user: User = Depends(get_admin_user)):
    """
    Create a new A/B test for model comparison (admin only).

    Args:
        request: A/B test parameters
        current_user: Current admin user

    Returns:
        Dict: A/B test configuration
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        test_config = {
            "test_id": test_id,
            "test_name": request.test_name,
            "control_model": request.control_model,
            "treatment_model": request.treatment_model,
            "traffic_split": request.traffic_split,
            "target_symbols": request.target_symbols,
            "duration_days": request.duration_days,
            "success_metrics": request.success_metrics,
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc) + timedelta(days=request.duration_days),
            "status": "active",
            "participants": 0,
            "control_participants": 0,
            "treatment_participants": 0,
        }

        logger.info("A/B test created", user=current_user.username, test_id=test_id)
        return test_config

    except Exception as e:
        logger.error(f"A/B test creation failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test creation failed: {e!s}",
        )


@router.get("/ab-test/{test_id}/results", response_model=dict)
async def get_ab_test_results(test_id: str, current_user: User = Depends(get_current_user)):
    """
    Get A/B test results.

    Args:
        test_id: Test identifier
        current_user: Current authenticated user

    Returns:
        Dict: A/B test results
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        # Mock A/B test results
        results = {
            "test_id": test_id,
            "status": "completed",
            "duration_actual_days": 7,
            "total_participants": 1250,
            "control_group": {
                "participants": 625,
                "accuracy": Decimal("0.742"),
                "profit_pct": Decimal("2.34"),
                "trades_executed": 1890,
                "avg_trade_duration_hours": Decimal("4.2"),
            },
            "treatment_group": {
                "participants": 625,
                "accuracy": Decimal("0.768"),
                "profit_pct": Decimal("2.89"),
                "trades_executed": 1823,
                "avg_trade_duration_hours": Decimal("3.8"),
            },
            "statistical_significance": {
                "accuracy_p_value": Decimal("0.032"),
                "profit_p_value": Decimal("0.018"),
                "confidence_level": Decimal("0.95"),
                "significant": True,
            },
            "recommendation": "promote_treatment",
        }

        logger.info("A/B test results retrieved", user=current_user.username, test_id=test_id)
        return results

    except Exception as e:
        logger.error(f"A/B test results retrieval failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get A/B test results: {e!s}",
        )


@router.post("/ab-test/{test_id}/promote", response_model=dict)
async def promote_ab_test_winner(
    test_id: str,
    promote_treatment: bool = Query(True, description="Promote treatment model"),
    current_user: User = Depends(get_admin_user),
):
    """
    Promote the winning model from A/B test (admin only).

    Args:
        test_id: Test identifier
        promote_treatment: Whether to promote treatment model
        current_user: Current admin user

    Returns:
        Dict: Promotion results
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        promoted_model = "treatment_model" if promote_treatment else "control_model"

        promotion_result = {
            "test_id": test_id,
            "promoted_model": promoted_model,
            "promotion_time": datetime.now(timezone.utc),
            "rollout_percentage": Decimal("100.0"),
            "previous_model_status": "retired",
            "deployment_status": "successful",
            "performance_impact": {
                "expected_accuracy_improvement": Decimal("2.6")
                if promote_treatment
                else Decimal("0.0"),
                "expected_profit_improvement": Decimal("23.5")
                if promote_treatment
                else Decimal("0.0"),
            },
        }

        logger.info(
            "A/B test model promoted",
            user=current_user.username,
            test_id=test_id,
            promoted_model=promoted_model,
        )
        return promotion_result

    except Exception as e:
        logger.error(f"A/B test promotion failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test promotion failed: {e!s}",
        )


# Hyperparameter Optimization Endpoints


class HyperparameterOptimizationRequest(BaseModel):
    """Request model for hyperparameter optimization."""

    optimization_method: str = Field(default="bayesian", description="Optimization method")
    parameter_space: dict[str, Any] = Field(..., description="Parameter search space")
    max_trials: int = Field(default=50, description="Maximum optimization trials")
    objective_metric: str = Field(
        default="val_accuracy", description="Objective metric to optimize"
    )
    cv_folds: int = Field(default=5, description="Cross-validation folds")


@router.post("/models/{model_id}/optimize", response_model=dict)
async def optimize_hyperparameters(
    model_id: str,
    request: HyperparameterOptimizationRequest,
    current_user: User = Depends(get_admin_user),
):
    """
    Start hyperparameter optimization for a model (admin only).

    Args:
        model_id: Model identifier
        request: Optimization parameters
        current_user: Current admin user

    Returns:
        Dict: Optimization job details
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        optimization_job = {
            "job_id": f"optim_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_id": model_id,
            "optimization_method": request.optimization_method,
            "parameter_space": request.parameter_space,
            "max_trials": request.max_trials,
            "objective_metric": request.objective_metric,
            "cv_folds": request.cv_folds,
            "status": "started",
            "current_trial": 0,
            "best_score": None,
            "best_parameters": None,
            "estimated_completion": datetime.now(timezone.utc) + timedelta(hours=2),
            "created_at": datetime.now(timezone.utc),
        }

        logger.info(
            "Hyperparameter optimization started",
            user=current_user.username,
            model_id=model_id,
            job_id=optimization_job["job_id"],
        )
        return optimization_job

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hyperparameter optimization failed: {e!s}",
        )


# Model Export and Comparison Endpoints


@router.get("/models/{model_id}/export", response_model=dict)
async def export_model(
    model_id: str,
    export_format: str = Query("onnx", description="Export format (onnx, pmml, pickle)"),
    include_metadata: bool = Query(True, description="Include model metadata"),
    current_user: User = Depends(get_admin_user),
):
    """
    Export a model in specified format (admin only).

    Args:
        model_id: Model identifier
        export_format: Export format
        include_metadata: Whether to include metadata
        current_user: Current admin user

    Returns:
        Dict: Export details
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        export_result = {
            "model_id": model_id,
            "export_format": export_format,
            "export_path": f"/exports/{model_id}.{export_format}",
            "file_size_mb": Decimal("12.4"),
            "include_metadata": include_metadata,
            "export_time": datetime.now(timezone.utc),
            "checksum": "sha256:abc123...",
            "compatibility": {
                "python_version": ">=3.8",
                "dependencies": ["numpy>=1.20.0", "pandas>=1.3.0"],
                "hardware_requirements": "CPU: 2+ cores, RAM: 4GB+",
            },
        }

        logger.info(
            "Model export completed",
            user=current_user.username,
            model_id=model_id,
            export_format=export_format,
        )
        return export_result

    except Exception as e:
        logger.error(f"Model export failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model export failed: {e!s}"
        )


@router.post("/models/compare", response_model=dict)
async def compare_models(
    model_ids: list[str] = Query(..., description="Model IDs to compare"),
    comparison_metrics: list[str] = Query(
        default=["accuracy", "precision", "recall"], description="Metrics to compare"
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Compare multiple models across specified metrics.

    Args:
        model_ids: List of model identifiers
        comparison_metrics: Metrics to compare
        current_user: Current authenticated user

    Returns:
        Dict: Model comparison results
    """
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model manager not available",
            )

        if len(model_ids) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 models required for comparison",
            )

        # Mock comparison results
        comparison_results = {
            "comparison_id": f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "models_compared": model_ids,
            "comparison_metrics": comparison_metrics,
            "results": {
                model_id: {
                    "accuracy": round(Decimal(str(0.7 + i * 0.05)), 4),
                    "precision": round(Decimal(str(0.72 + i * 0.03)), 4),
                    "recall": round(Decimal(str(0.68 + i * 0.04)), 4),
                    "f1_score": round(Decimal(str(0.70 + i * 0.035)), 4),
                    "inference_time_ms": Decimal(str(10 + i * 2)),
                    "model_size_mb": Decimal(str(15 + i * 3)),
                }
                for i, model_id in enumerate(model_ids)
            },
            "ranking": {
                metric: sorted(model_ids, key=lambda x: model_ids.index(x), reverse=True)[:3]
                for metric in comparison_metrics
            },
            "recommendation": model_ids[0],  # Best overall
            "comparison_time": datetime.now(timezone.utc),
        }

        logger.info(
            "Model comparison completed",
            user=current_user.username,
            models=len(model_ids),
            comparison_id=comparison_results["comparison_id"],
        )
        return comparison_results

    except Exception as e:
        logger.error(f"Model comparison failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model comparison failed: {e!s}",
        )


@router.get("/health/status")
async def get_ml_system_health(current_user: User = Depends(get_current_user)):
    """
    Get ML system health status.

    Args:
        current_user: Current authenticated user

    Returns:
        Dict: ML system health status
    """
    try:
        if not model_manager:
            return {"status": "unavailable", "message": "Model manager not available"}

        # Mock health check (in production, check actual system health)
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "model_manager": "healthy",
                "inference_engine": "healthy",
                "training_queue": "healthy",
                "model_registry": "healthy",
                "feature_store": "healthy",
            },
            "statistics": {
                "total_models": 12,
                "active_models": 8,
                "training_models": 2,
                "retired_models": 2,
                "total_predictions_24h": 15420,
                "average_inference_time_ms": 12.5,
                "model_accuracy_average": 0.74,
            },
            "alerts": [
                {
                    "type": "performance_degradation",
                    "model": "model_003",
                    "message": "Model accuracy dropped below threshold",
                    "severity": "medium",
                }
            ],
        }

        return health_status

    except Exception as e:
        logger.error(f"ML system health check failed: {e}", user=current_user.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ML system health: {e!s}",
        )
