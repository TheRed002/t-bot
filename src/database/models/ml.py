"""
Machine Learning models for database.

This module contains all ML-related database models for storing
predictions, model metadata, and training results.
"""

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

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
    prediction_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))  # Main prediction value
    confidence_score: Mapped[Decimal | None] = mapped_column(
        DECIMAL(8, 6)
    )  # Confidence/probability score (0-1 with 6 decimals)
    prediction_type = Column(String(50), nullable=True)  # e.g., 'price', 'direction', 'volatility'

    # Feature information
    features_hash = Column(String(64), nullable=True)  # Hash of input features for reproducibility
    features_metadata = Column(Text, nullable=True)  # JSON string of feature names/types

    # Performance tracking
    actual_value: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))  # Actual value (filled later for evaluation)
    prediction_error: Mapped[Decimal | None] = mapped_column(DECIMAL(20, 8))  # Calculated error when actual is known

    # Additional metadata
    market_conditions = Column(Text, nullable=True)  # JSON string of market context
    prediction_horizon = Column(Integer, nullable=True)  # Prediction timeframe in minutes

    # Indexes and constraints
    __table_args__ = (
        Index("idx_ml_predictions_model_symbol", "model_name", "symbol"),
        Index("idx_ml_predictions_timestamp", "timestamp"),
        Index("idx_ml_predictions_prediction_timestamp", "prediction_timestamp"),
        Index("idx_ml_predictions_model_timestamp", "model_name", "timestamp"),
        Index("idx_ml_predictions_confidence", "confidence_score"),  # Performance filtering
        Index("idx_ml_predictions_prediction_type", "prediction_type"),  # Type-based queries
        Index("idx_ml_predictions_symbol_type", "symbol", "prediction_type"),  # Symbol-specific predictions
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="check_confidence_score_range",
        ),
        CheckConstraint(
            "prediction_horizon IS NULL OR prediction_horizon > 0",
            name="check_prediction_horizon_positive",
        ),
    )

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
    model_type = Column(String(50), nullable=False)  # e.g., 'price_predictor', 'direction_classifier'

    # Model details
    model_path = Column(String(500), nullable=True)  # Path to saved model file
    model_size_mb: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 2), nullable=True
    )  # Model size in MB with 2 decimal precision
    training_dataset_hash = Column(String(64), nullable=True)

    # Performance metrics
    training_accuracy: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 6))  # Accuracy score (0-1)
    validation_accuracy: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 6))  # Validation accuracy (0-1)
    test_accuracy: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 6))  # Test accuracy (0-1)

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
    average_prediction_time_ms: Mapped[Decimal | None] = mapped_column(
        DECIMAL(10, 3)
    )  # Average prediction time in milliseconds

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Indexes and constraints
    __table_args__ = (
        Index("idx_ml_model_metadata_name_version", "model_name", "model_version"),
        Index("idx_ml_model_metadata_active", "is_active"),
        Index("idx_ml_model_metadata_created", "created_at"),
        CheckConstraint(
            "training_accuracy >= 0 AND training_accuracy <= 1",
            name="check_training_accuracy_range",
        ),
        CheckConstraint(
            "validation_accuracy >= 0 AND validation_accuracy <= 1",
            name="check_validation_accuracy_range",
        ),
        CheckConstraint("test_accuracy >= 0 AND test_accuracy <= 1", name="check_test_accuracy_range"),
        CheckConstraint("model_size_mb >= 0", name="check_model_size_non_negative"),
        CheckConstraint("training_duration_seconds >= 0", name="check_training_duration_non_negative"),
        CheckConstraint("prediction_count >= 0", name="check_prediction_count_non_negative"),
        CheckConstraint("is_active IN ('true', 'false')", name="check_is_active_boolean_string"),
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
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    progress_percentage: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=0.0)  # Progress percentage (0-100)

    # Training configuration
    training_config = Column(Text, nullable=True)  # JSON string of training parameters
    dataset_config = Column(Text, nullable=True)  # JSON string of dataset parameters

    # Resource usage
    cpu_usage_percent: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 2))  # CPU usage percentage (0-100)
    memory_usage_mb: Mapped[Decimal | None] = mapped_column(DECIMAL(10, 2))  # Memory usage in MB
    gpu_usage_percent: Mapped[Decimal | None] = mapped_column(DECIMAL(5, 2))  # GPU usage percentage (0-100)

    # Results
    final_model_path = Column(String(500), nullable=True)
    training_logs = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Performance metrics
    best_validation_score: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 6))  # Best validation score (0-1)
    final_training_score: Mapped[Decimal | None] = mapped_column(DECIMAL(8, 6))  # Final training score (0-1)
    epochs_completed = Column(Integer, nullable=True)
    early_stopping_epoch = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Indexes and constraints
    __table_args__ = (
        Index("idx_ml_training_jobs_job_id", "job_id"),
        Index("idx_ml_training_jobs_model_name", "model_name"),
        Index("idx_ml_training_jobs_status", "status"),
        Index("idx_ml_training_jobs_created", "created_at"),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="check_training_job_status",
        ),
        CheckConstraint(
            "progress_percentage >= 0 AND progress_percentage <= 100",
            name="check_progress_percentage_range",
        ),
        CheckConstraint("cpu_usage_percent >= 0 AND cpu_usage_percent <= 100", name="check_cpu_usage_range"),
        CheckConstraint("memory_usage_mb >= 0", name="check_memory_usage_non_negative"),
        CheckConstraint("gpu_usage_percent >= 0 AND gpu_usage_percent <= 100", name="check_gpu_usage_range"),
        CheckConstraint(
            "best_validation_score >= 0 AND best_validation_score <= 1",
            name="check_validation_score_range",
        ),
        CheckConstraint(
            "final_training_score >= 0 AND final_training_score <= 1",
            name="check_training_score_range",
        ),
        CheckConstraint("epochs_completed >= 0", name="check_epochs_completed_non_negative"),
        CheckConstraint("early_stopping_epoch >= 0", name="check_early_stopping_epoch_non_negative"),
    )

    def __repr__(self) -> str:
        return (
            f"<MLTrainingJob(id={self.id}, job_id={self.job_id}, "
            f"model={self.model_name}, status={self.status}, "
            f"progress={self.progress_percentage}%)>"
        )
