"""
ML Configuration Utilities.

This module provides shared configuration patterns to eliminate
duplicate configuration code across ML services and models.
"""

from pydantic import BaseModel, Field

from src.utils.constants import ML_MODEL_CONSTANTS


class BaseMLConfig(BaseModel):
    """Base configuration for ML components."""

    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_validation: bool = Field(default=True, description="Enable data validation")
    log_level: str = Field(default="info", description="Logging level")
    correlation_id: str | None = Field(default=None, description="Request correlation ID")


class MLModelConfig(BaseMLConfig):
    """Configuration for ML models."""

    enable_model_validation: bool = Field(default=True, description="Enable model validation")
    enable_feature_selection: bool = Field(default=True, description="Enable feature selection")
    enable_model_persistence: bool = Field(default=True, description="Enable model persistence")
    model_storage_backend: str = Field(default="joblib", description="Model storage backend")
    training_validation_split: float = Field(default=0.2, description="Validation split ratio")
    enable_training_history: bool = Field(
        default=True, description="Enable training history tracking"
    )
    max_training_history_length: int = Field(
        default=100, description="Maximum training history length"
    )
    random_state: int | None = Field(default=42, description="Random state for reproducibility")


class MLServiceConfig(BaseMLConfig):
    """Configuration for ML service."""

    enable_feature_engineering: bool = Field(default=True, description="Enable feature engineering")
    enable_model_registry: bool = Field(default=True, description="Enable model registry")
    enable_inference: bool = Field(default=True, description="Enable inference service")
    enable_feature_store: bool = Field(default=True, description="Enable feature store")
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing")
    enable_pipeline_caching: bool = Field(default=True, description="Enable pipeline caching")
    max_concurrent_operations: int = Field(
        default=ML_MODEL_CONSTANTS["max_concurrent_operations"],
        description="Maximum concurrent ML operations",
    )
    pipeline_timeout_seconds: int = Field(
        default=ML_MODEL_CONSTANTS["pipeline_timeout_seconds"],
        description="Pipeline timeout in seconds",
    )
    cache_ttl_minutes: int = Field(
        default=ML_MODEL_CONSTANTS["cache_ttl_minutes"], description="Cache TTL in minutes"
    )
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )


class ModelManagerConfig(BaseMLConfig):
    """Configuration for model manager service."""

    enable_model_monitoring: bool = Field(default=True, description="Enable model monitoring")
    default_validation_threshold: float = Field(
        default=ML_MODEL_CONSTANTS["default_validation_threshold"],
        description="Default validation accuracy threshold",
    )
    max_active_models: int = Field(
        default=ML_MODEL_CONSTANTS["max_active_models"],
        description="Maximum number of active models",
    )
    model_retirement_days: int = Field(
        default=ML_MODEL_CONSTANTS["model_retirement_days"],
        description="Days after which unused models are retired",
    )
    enable_auto_retraining: bool = Field(
        default=False, description="Enable automatic model retraining"
    )
    drift_threshold: float = Field(
        default=ML_MODEL_CONSTANTS["drift_threshold"], description="Drift detection threshold"
    )
    performance_decline_threshold: float = Field(
        default=ML_MODEL_CONSTANTS["performance_decline_threshold"],
        description="Performance decline threshold",
    )
    # Classifier thresholds
    min_accuracy_threshold: float = Field(
        default=ML_MODEL_CONSTANTS["min_accuracy_threshold"],
        description="Minimum accuracy for classifiers",
    )
    min_f1_score_threshold: float = Field(
        default=ML_MODEL_CONSTANTS["min_f1_score_threshold"],
        description="Minimum F1 score for classifiers",
    )
    # Regressor thresholds
    max_mse_threshold: float = Field(
        default=ML_MODEL_CONSTANTS["max_mse_threshold"], description="Maximum MSE for regressors"
    )
    min_r2_threshold: float = Field(
        default=ML_MODEL_CONSTANTS["min_r2_threshold"],
        description="Minimum R² score for regressors",
    )


class PredictorConfig(MLModelConfig):
    """Configuration for predictor models."""

    algorithm: str = Field(default="random_forest", description="Algorithm to use")
    prediction_horizon: int = Field(default=1, description="Prediction horizon")
    enable_gpu_acceleration: bool = Field(default=True, description="Enable GPU acceleration if available")

    # Algorithm-specific defaults
    n_estimators: int = Field(default=100, description="Number of estimators for tree models")
    max_depth: int | None = Field(default=10, description="Maximum tree depth")
    learning_rate: float = Field(default=0.1, description="Learning rate for boosting models")


class ClassifierConfig(MLModelConfig):
    """Configuration for classifier models."""

    algorithm: str = Field(default="random_forest", description="Algorithm to use")
    direction_threshold: float = Field(default=0.01, description="Direction classification threshold")
    prediction_horizon: int = Field(default=1, description="Prediction horizon")
    class_weights: str | dict | None = Field(default="balanced", description="Class weights")

    # Algorithm-specific defaults
    n_estimators: int = Field(default=100, description="Number of estimators for tree models")
    max_depth: int | None = Field(default=10, description="Maximum tree depth")
    max_iter: int = Field(default=1000, description="Maximum iterations for iterative algorithms")


class MLCacheConfig(BaseModel):
    """Configuration for ML caching."""

    model_cache_ttl_hours: int = Field(default=24, description="Model cache TTL in hours")
    prediction_cache_ttl_minutes: int = Field(default=5, description="Prediction cache TTL in minutes")
    feature_cache_ttl_hours: int = Field(default=4, description="Feature cache TTL in hours")
    max_cached_models: int = Field(default=100, description="Maximum cached models")
    max_cached_predictions: int = Field(default=10000, description="Maximum cached predictions")
    max_cached_feature_sets: int = Field(default=1000, description="Maximum cached feature sets")
    cleanup_interval_minutes: int = Field(default=30, description="Cache cleanup interval in minutes")


def create_ml_service_config(config_dict: dict | None = None) -> MLServiceConfig:
    """Create ML service configuration from dictionary."""
    config_dict = config_dict or {}
    return MLServiceConfig(**config_dict)


def create_model_manager_config(config_dict: dict | None = None) -> ModelManagerConfig:
    """Create model manager configuration from dictionary."""
    config_dict = config_dict or {}
    return ModelManagerConfig(**config_dict)


def create_predictor_config(config_dict: dict | None = None) -> PredictorConfig:
    """Create predictor configuration from dictionary."""
    config_dict = config_dict or {}
    return PredictorConfig(**config_dict)


def create_classifier_config(config_dict: dict | None = None) -> ClassifierConfig:
    """Create classifier configuration from dictionary."""
    config_dict = config_dict or {}
    return ClassifierConfig(**config_dict)


def create_cache_config(config_dict: dict | None = None) -> MLCacheConfig:
    """Create cache configuration from dictionary."""
    config_dict = config_dict or {}
    return MLCacheConfig(**config_dict)
