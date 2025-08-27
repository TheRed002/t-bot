"""
Machine Learning models for database.

This module contains all ML-related database models for storing
predictions, model metadata, and training results.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)

from .base import Base


class MLPrediction(Base):
    """
    Model for storing ML prediction results.

    This table stores individual predictions made by ML models,
    including metadata about the prediction context and confidence scores.
    """

    __tablename__ = "ml_predictions"

    id = Column(BigInteger, primary_key=True, index=True)

    # Model information
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)

    # Trading context
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(50), nullable=True)

    # Timing information
    timestamp = Column(DateTime, nullable=False, index=True)  # Data timestamp
    prediction_timestamp = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )  # When prediction was made

    # Prediction results
    prediction_value = Column(Float, nullable=True)  # Main prediction value
    confidence_score = Column(Float, nullable=True)  # Confidence/probability score
    prediction_type = Column(String(50), nullable=True)  # e.g., 'price', 'direction', 'volatility'

    # Feature information
    features_hash = Column(String(64), nullable=True)  # Hash of input features for reproducibility
    features_metadata = Column(Text, nullable=True)  # JSON string of feature names/types

    # Performance tracking
    actual_value = Column(Float, nullable=True)  # Actual value (filled later for evaluation)
    prediction_error = Column(Float, nullable=True)  # Calculated error when actual is known

    # Additional metadata
    market_conditions = Column(Text, nullable=True)  # JSON string of market context
    prediction_horizon = Column(Integer, nullable=True)  # Prediction timeframe in minutes

    def __repr__(self) -> str:
        return (
            f"<MLPrediction(id={self.id}, model={self.model_name}, "
            f"symbol={self.symbol}, timestamp={self.timestamp}, "
            f"prediction={self.prediction_value}, confidence={self.confidence_score})>"
        )


class MLModelMetadata(Base):
    """
    Model for storing ML model metadata and versioning information.

    This table tracks different versions of ML models, their performance
    metrics, and deployment status.
    """

    __tablename__ = "ml_model_metadata"

    id = Column(BigInteger, primary_key=True, index=True)

    # Model identification
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(
        String(50), nullable=False
    )  # e.g., 'price_predictor', 'direction_classifier'

    # Model details
    model_path = Column(String(500), nullable=True)  # Path to saved model file
    model_size_mb = Column(Float, nullable=True)
    training_dataset_hash = Column(String(64), nullable=True)

    # Performance metrics
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)

    # Training information
    training_start_time = Column(DateTime, nullable=True)
    training_end_time = Column(DateTime, nullable=True)
    training_duration_seconds = Column(Integer, nullable=True)

    # Deployment status
    is_active = Column(String(10), default="false")  # 'true' or 'false' as string
    deployment_date = Column(DateTime, nullable=True)
    deprecation_date = Column(DateTime, nullable=True)

    # Hyperparameters and configuration
    hyperparameters = Column(Text, nullable=True)  # JSON string
    feature_config = Column(Text, nullable=True)  # JSON string

    # Performance tracking
    prediction_count = Column(BigInteger, default=0)
    average_prediction_time_ms = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<MLModelMetadata(id={self.id}, name={self.model_name}, "
            f"version={self.model_version}, type={self.model_type}, "
            f"active={self.is_active})>"
        )


class MLTrainingJob(Base):
    """
    Model for tracking ML model training jobs.

    This table stores information about training jobs including their
    status, parameters, and results.
    """

    __tablename__ = "ml_training_jobs"

    id = Column(BigInteger, primary_key=True, index=True)

    # Job identification
    job_id = Column(String(100), nullable=False, unique=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)

    # Job status
    status = Column(
        String(20), nullable=False, default="pending"
    )  # pending, running, completed, failed
    progress_percentage = Column(Float, default=0.0)

    # Training configuration
    training_config = Column(Text, nullable=True)  # JSON string of training parameters
    dataset_config = Column(Text, nullable=True)  # JSON string of dataset parameters

    # Resource usage
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    gpu_usage_percent = Column(Float, nullable=True)

    # Results
    final_model_path = Column(String(500), nullable=True)
    training_logs = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Performance metrics
    best_validation_score = Column(Float, nullable=True)
    final_training_score = Column(Float, nullable=True)
    epochs_completed = Column(Integer, nullable=True)
    early_stopping_epoch = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<MLTrainingJob(id={self.id}, job_id={self.job_id}, "
            f"model={self.model_name}, status={self.status}, "
            f"progress={self.progress_percentage}%)>"
        )
