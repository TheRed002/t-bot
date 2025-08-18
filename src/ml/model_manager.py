"""
Central ML Model Lifecycle Management.

This module provides centralized management for the entire ML model lifecycle
including training, validation, deployment, monitoring, and retirement.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.ml.feature_engineering import FeatureEngineer
from src.ml.inference.batch_predictor import BatchPredictor
from src.ml.inference.inference_engine import InferenceEngine
from src.ml.models.base_model import BaseModel
from src.ml.models.direction_classifier import DirectionClassifier
from src.ml.models.price_predictor import PricePredictor
from src.ml.models.regime_detector import RegimeDetector
from src.ml.models.volatility_forecaster import VolatilityForecaster
from src.ml.registry.model_registry import ModelRegistry
from src.ml.training.trainer import Trainer
from src.ml.validation.drift_detector import DriftDetector
from src.ml.validation.model_validator import ModelValidator
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class ModelManager:
    """
    Central manager for ML model lifecycle.

    This class orchestrates all aspects of the ML pipeline:
    - Model creation and training
    - Model validation and testing
    - Model deployment and versioning
    - Performance monitoring and drift detection
    - Model retirement and replacement

    Attributes:
        config: Application configuration
        model_registry: Model registry for storage and versioning
        feature_engineer: Feature engineering pipeline
        trainer: Model training orchestrator
        validator: Model validation system
        drift_detector: Drift detection system
        inference_engine: Real-time inference engine
        batch_predictor: Batch prediction system
    """

    def __init__(self, config: Config):
        """
        Initialize the model manager.

        Args:
            config: Application configuration
        """
        self.config = config

        # Initialize ML components
        self.model_registry = ModelRegistry(config)
        self.feature_engineer = FeatureEngineer(config)
        self.trainer = Trainer(config)
        self.validator = ModelValidator(config)
        self.drift_detector = DriftDetector(config)
        self.inference_engine = InferenceEngine(config)
        self.batch_predictor = BatchPredictor(config)

        # Model types
        self.model_types = {
            "price_predictor": PricePredictor,
            "direction_classifier": DirectionClassifier,
            "volatility_forecaster": VolatilityForecaster,
            "regime_detector": RegimeDetector,
        }

        # Lifecycle state
        self.active_models = {}
        self.training_jobs = {}
        self.monitoring_jobs = {}

        logger.info(
            "Model manager initialized", available_model_types=list(self.model_types.keys())
        )

    @time_execution
    @log_calls
    async def create_and_train_model(
        self,
        model_type: str,
        model_name: str,
        training_data: pd.DataFrame,
        symbol: str,
        model_params: dict[str, Any] | None = None,
        training_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create and train a new model.

        Args:
            model_type: Type of model to create
            model_name: Name for the model
            training_data: Training data
            symbol: Trading symbol
            model_params: Model-specific parameters
            training_params: Training-specific parameters

        Returns:
            Training results and model information

        Raises:
            ValidationError: If model creation or training fails
        """
        try:
            if model_type not in self.model_types:
                raise ValidationError(f"Unknown model type: {model_type}")

            if training_data.empty:
                raise ValidationError("Training data cannot be empty")

            logger.info(
                "Starting model creation and training",
                model_type=model_type,
                model_name=model_name,
                symbol=symbol,
                training_samples=len(training_data),
            )

            # Create model instance
            model_class = self.model_types[model_type]
            model_params = model_params or {}
            model = model_class(self.config, **model_params)
            model.model_name = model_name

            # Prepare training data
            training_result = await self._prepare_and_train_model(
                model, training_data, symbol, training_params
            )

            # Validate model
            validation_result = await self._validate_trained_model(
                model, training_result["test_data"]
            )

            # Register model if validation passes
            if validation_result["overall_pass"]:
                model_info = await self._register_model(
                    model, training_result, validation_result, symbol
                )

                # Add to active models
                self.active_models[model_name] = {
                    "model": model,
                    "model_info": model_info,
                    "created_at": datetime.utcnow(),
                    "symbol": symbol,
                    "status": "active",
                }

                logger.info(
                    "Model creation and training completed successfully",
                    model_name=model_name,
                    model_id=model_info["id"],
                )

                return {
                    "success": True,
                    "model_name": model_name,
                    "model_info": model_info,
                    "training_result": training_result,
                    "validation_result": validation_result,
                }

            else:
                logger.warning("Model validation failed, not registering", model_name=model_name)

                return {
                    "success": False,
                    "model_name": model_name,
                    "training_result": training_result,
                    "validation_result": validation_result,
                    "reason": "validation_failed",
                }

        except Exception as e:
            logger.error(
                "Model creation and training failed",
                model_type=model_type,
                model_name=model_name,
                error=str(e),
            )
            raise ValidationError(f"Model creation and training failed: {e}") from e

    @time_execution
    @log_calls
    async def deploy_model(
        self, model_name: str, deployment_stage: str = "production"
    ) -> dict[str, Any]:
        """
        Deploy a model to a specific stage.

        Args:
            model_name: Name of the model to deploy
            deployment_stage: Deployment stage ('development', 'staging', 'production')

        Returns:
            Deployment result

        Raises:
            ValidationError: If deployment fails
        """
        try:
            logger.info(
                "Starting model deployment",
                model_name=model_name,
                deployment_stage=deployment_stage,
            )

            # Get model from registry
            model_info = self.model_registry.get_latest_model(model_name)
            if not model_info:
                raise ValidationError(f"Model {model_name} not found in registry")

            # Load model
            model = self.model_registry.load_model(model_info["id"])

            # Perform pre-deployment validation
            pre_deployment_checks = await self._pre_deployment_validation(model, model_info)

            if not pre_deployment_checks["ready_for_deployment"]:
                raise ValidationError(
                    f"Model {model_name} failed pre-deployment validation: "
                    f"{pre_deployment_checks['issues']}"
                )

            # Promote model in registry
            promotion_result = self.model_registry.promote_model(model_info["id"], deployment_stage)

            # Update inference engine if deploying to production
            if deployment_stage == "production":
                await self.inference_engine.load_model(model_name, model)

                # Start monitoring
                await self._start_model_monitoring(model_name, model)

            deployment_result = {
                "success": True,
                "model_name": model_name,
                "model_id": model_info["id"],
                "deployment_stage": deployment_stage,
                "deployment_timestamp": datetime.utcnow(),
                "pre_deployment_checks": pre_deployment_checks,
                "promotion_result": promotion_result,
            }

            logger.info(
                "Model deployment completed",
                model_name=model_name,
                deployment_stage=deployment_stage,
                model_id=model_info["id"],
            )

            return deployment_result

        except Exception as e:
            logger.error(
                "Model deployment failed",
                model_name=model_name,
                deployment_stage=deployment_stage,
                error=str(e),
            )
            raise ValidationError(f"Model deployment failed: {e}") from e

    @time_execution
    @log_calls
    async def monitor_model_performance(
        self,
        model_name: str,
        monitoring_data: pd.DataFrame,
        true_labels: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Monitor model performance and detect drift.

        Args:
            model_name: Name of the model to monitor
            monitoring_data: New data for monitoring
            true_labels: True labels for performance monitoring

        Returns:
            Monitoring results

        Raises:
            ValidationError: If monitoring fails
        """
        try:
            logger.info(
                "Starting model performance monitoring",
                model_name=model_name,
                monitoring_samples=len(monitoring_data),
            )

            # Get model
            if model_name not in self.active_models:
                model_info = self.model_registry.get_latest_model(model_name)
                if not model_info:
                    raise ValidationError(f"Model {model_name} not found")
                model = self.model_registry.load_model(model_info["id"])
            else:
                model = self.active_models[model_name]["model"]

            monitoring_results = {}

            # 1. Feature drift detection
            reference_data = self.drift_detector.get_reference_data("features")
            if reference_data is not None:
                feature_drift = self.drift_detector.detect_feature_drift(
                    reference_data, monitoring_data
                )
                monitoring_results["feature_drift"] = feature_drift
            else:
                logger.warning("No reference data for feature drift detection")

            # 2. Prediction drift detection
            predictions = model.predict(monitoring_data)
            reference_predictions = self.drift_detector.get_reference_data("predictions")
            if reference_predictions is not None:
                prediction_drift = self.drift_detector.detect_prediction_drift(
                    reference_predictions, predictions, model_name
                )
                monitoring_results["prediction_drift"] = prediction_drift
            else:
                # Set current predictions as reference if none exist
                self.drift_detector.set_reference_data(
                    pd.DataFrame({"predictions": predictions}), "predictions"
                )

            # 3. Performance monitoring (if true labels available)
            if true_labels is not None:
                performance_metrics = model.evaluate(monitoring_data, true_labels)
                monitoring_results["current_performance"] = performance_metrics

                # Performance drift detection
                reference_performance = self.drift_detector.reference_data.get(
                    "performance", {}
                ).get("data")
                if reference_performance is not None:
                    performance_drift = self.drift_detector.detect_performance_drift(
                        reference_performance, performance_metrics, model_name
                    )
                    monitoring_results["performance_drift"] = performance_drift

            # 4. Overall monitoring assessment
            overall_drift_detected = any(
                result.get("drift_detected", False) for result in monitoring_results.values()
            )

            # 5. Generate alerts if necessary
            alerts = await self._generate_monitoring_alerts(
                model_name, monitoring_results, overall_drift_detected
            )

            final_monitoring_result = {
                "timestamp": datetime.utcnow(),
                "model_name": model_name,
                "monitoring_samples": len(monitoring_data),
                "overall_drift_detected": overall_drift_detected,
                "monitoring_results": monitoring_results,
                "alerts": alerts,
            }

            logger.info(
                "Model performance monitoring completed",
                model_name=model_name,
                overall_drift_detected=overall_drift_detected,
                alerts_generated=len(alerts),
            )

            return final_monitoring_result

        except Exception as e:
            logger.error("Model performance monitoring failed", model_name=model_name, error=str(e))
            raise ValidationError(f"Model performance monitoring failed: {e}") from e

    @time_execution
    @log_calls
    async def retire_model(self, model_name: str, reason: str = "replaced") -> dict[str, Any]:
        """
        Retire a model from active service.

        Args:
            model_name: Name of the model to retire
            reason: Reason for retirement

        Returns:
            Retirement result

        Raises:
            ValidationError: If retirement fails
        """
        try:
            logger.info("Starting model retirement", model_name=model_name, reason=reason)

            # Remove from inference engine
            await self.inference_engine.unload_model(model_name)

            # Stop monitoring
            if model_name in self.monitoring_jobs:
                await self._stop_model_monitoring(model_name)

            # Update registry status
            model_info = self.model_registry.get_latest_model(model_name)
            if model_info:
                self.model_registry.update_model_metadata(
                    model_info["id"],
                    {
                        "status": "retired",
                        "retirement_reason": reason,
                        "retired_at": datetime.utcnow(),
                    },
                )

            # Remove from active models
            if model_name in self.active_models:
                self.active_models[model_name]["status"] = "retired"
                del self.active_models[model_name]

            retirement_result = {
                "success": True,
                "model_name": model_name,
                "retirement_timestamp": datetime.utcnow(),
                "reason": reason,
            }

            logger.info("Model retirement completed", model_name=model_name, reason=reason)

            return retirement_result

        except Exception as e:
            logger.error("Model retirement failed", model_name=model_name, error=str(e))
            raise ValidationError(f"Model retirement failed: {e}") from e

    async def _prepare_and_train_model(
        self,
        model: BaseModel,
        training_data: pd.DataFrame,
        symbol: str,
        training_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Prepare data and train model."""
        try:
            # Use trainer for comprehensive training
            training_result = await self.trainer.train_model(
                model, training_data, symbol, training_params
            )

            return training_result

        except Exception as e:
            logger.error(f"Model training preparation failed: {e}")
            raise ValidationError(f"Model training preparation failed: {e}") from e

    async def _validate_trained_model(
        self, model: BaseModel, test_data: tuple[pd.DataFrame, pd.Series]
    ) -> dict[str, Any]:
        """Validate trained model."""
        try:
            X_test, y_test = test_data

            # Production readiness validation
            validation_result = self.validator.validate_production_readiness(model, X_test, y_test)

            return validation_result

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise ValidationError(f"Model validation failed: {e}") from e

    async def _register_model(
        self,
        model: BaseModel,
        training_result: dict[str, Any],
        validation_result: dict[str, Any],
        symbol: str,
    ) -> dict[str, Any]:
        """Register model in registry."""
        try:
            # Save model artifacts
            model_artifacts = await self.trainer.save_artifacts(model, training_result, {}, {})

            # Register in model registry
            model_info = self.model_registry.register_model(
                model_name=model.model_name,
                model_type=model.model_type,
                version=model.version,
                model_path=model_artifacts["model_path"],
                metrics=training_result.get("training_metrics", {}),
                metadata={
                    "symbol": symbol,
                    "training_samples": training_result.get("training_samples", 0),
                    "validation_passed": validation_result["overall_pass"],
                    "created_by": "model_manager",
                },
            )

            return model_info

        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise ValidationError(f"Model registration failed: {e}") from e

    async def _pre_deployment_validation(
        self, model: BaseModel, model_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform pre-deployment validation."""
        try:
            checks = {
                "model_trained": model.is_trained,
                "has_required_methods": all(
                    hasattr(model, method) for method in ["predict", "evaluate"]
                ),
                "model_registered": model_info is not None,
                "recent_validation": True,  # Placeholder for validation recency check
            }

            issues = [check_name for check_name, passed in checks.items() if not passed]

            return {"ready_for_deployment": len(issues) == 0, "checks": checks, "issues": issues}

        except Exception as e:
            logger.error(f"Pre-deployment validation failed: {e}")
            return {
                "ready_for_deployment": False,
                "checks": {},
                "issues": [f"Validation error: {e}"],
            }

    async def _start_model_monitoring(self, model_name: str, model: BaseModel):
        """Start monitoring for a deployed model."""
        try:
            # Create monitoring job (placeholder implementation)
            self.monitoring_jobs[model_name] = {
                "model": model,
                "start_time": datetime.utcnow(),
                "status": "active",
            }

            logger.info(f"Monitoring started for model {model_name}")

        except Exception as e:
            logger.error(f"Failed to start monitoring for {model_name}: {e}")

    async def _stop_model_monitoring(self, model_name: str):
        """Stop monitoring for a model."""
        try:
            if model_name in self.monitoring_jobs:
                self.monitoring_jobs[model_name]["status"] = "stopped"
                self.monitoring_jobs[model_name]["stop_time"] = datetime.utcnow()
                del self.monitoring_jobs[model_name]

            logger.info(f"Monitoring stopped for model {model_name}")

        except Exception as e:
            logger.error(f"Failed to stop monitoring for {model_name}: {e}")

    async def _generate_monitoring_alerts(
        self, model_name: str, monitoring_results: dict[str, Any], overall_drift_detected: bool
    ) -> list[dict[str, Any]]:
        """Generate alerts based on monitoring results."""
        try:
            alerts = []

            if overall_drift_detected:
                alerts.append(
                    {
                        "type": "drift_detected",
                        "model_name": model_name,
                        "timestamp": datetime.utcnow(),
                        "severity": "high",
                        "message": f"Drift detected for model {model_name}",
                        "details": monitoring_results,
                    }
                )

            # Add more alert types as needed

            return alerts

        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            return []

    def get_active_models(self) -> dict[str, Any]:
        """Get information about active models."""
        return {
            name: {
                "model_type": info["model"].model_type,
                "created_at": info["created_at"],
                "symbol": info["symbol"],
                "status": info["status"],
            }
            for name, info in self.active_models.items()
        }

    def get_model_status(self, model_name: str) -> dict[str, Any] | None:
        """Get status information for a specific model."""
        if model_name in self.active_models:
            return {"status": "active", "details": self.active_models[model_name]}

        # Check registry
        model_info = self.model_registry.get_latest_model(model_name)
        if model_info:
            return {"status": "registered", "details": model_info}

        return None

    async def health_check(self) -> dict[str, Any]:
        """Perform health check of the ML system."""
        try:
            health_status = {
                "timestamp": datetime.utcnow(),
                "status": "healthy",
                "components": {},
                "active_models": len(self.active_models),
                "monitoring_jobs": len(self.monitoring_jobs),
            }

            # Check component health
            components = [
                ("model_registry", self.model_registry),
                ("feature_engineer", self.feature_engineer),
                ("trainer", self.trainer),
                ("validator", self.validator),
                ("drift_detector", self.drift_detector),
                ("inference_engine", self.inference_engine),
                ("batch_predictor", self.batch_predictor),
            ]

            for name, component in components:
                try:
                    # Basic health check - component exists and has required attributes
                    component_healthy = hasattr(component, "config")
                    health_status["components"][name] = (
                        "healthy" if component_healthy else "unhealthy"
                    )
                except Exception as e:
                    health_status["components"][name] = f"error: {e}"
                    health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"timestamp": datetime.utcnow(), "status": "unhealthy", "error": str(e)}
