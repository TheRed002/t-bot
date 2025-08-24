"""
Central ML Model Lifecycle Management.

This module provides centralized management for the entire ML model lifecycle
including training, validation, deployment, monitoring, and retirement.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field

from src.core.base.service import BaseService
from src.core.exceptions import ValidationError
from src.core.types.base import ConfigDict
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class ModelManagerConfig(PydanticBaseModel):
    """Configuration for model manager service."""

    enable_model_monitoring: bool = Field(default=True, description="Enable model monitoring")
    default_validation_threshold: float = Field(
        default=0.6, description="Default validation accuracy threshold"
    )
    max_active_models: int = Field(default=10, description="Maximum number of active models")
    model_retirement_days: int = Field(
        default=90, description="Days after which unused models are retired"
    )
    enable_auto_retraining: bool = Field(
        default=False, description="Enable automatic model retraining"
    )
    drift_threshold: float = Field(default=0.1, description="Drift detection threshold")
    performance_decline_threshold: float = Field(
        default=0.05, description="Performance decline threshold"
    )


class ModelManagerService(BaseService):
    """
    Central manager for ML model lifecycle.

    This service orchestrates all aspects of the ML pipeline using proper service patterns:
    - Model creation and training
    - Model validation and testing
    - Model deployment and versioning
    - Performance monitoring and drift detection
    - Model retirement and replacement

    All operations go through service dependencies without direct database access.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the model manager service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="ModelManagerService",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse model manager configuration
        mm_config_dict = (config or {}).get("model_manager", {})
        self.mm_config = ModelManagerConfig(**mm_config_dict)

        # Service dependencies - resolved during startup
        self.model_registry_service: Any = None
        self.feature_engineering_service: Any = None
        self.training_service: Any = None
        self.validation_service: Any = None
        self.drift_detection_service: Any = None
        self.inference_service: Any = None
        self.batch_prediction_service: Any = None
        
        # Legacy service references (will be mapped to actual services)
        self.trainer: Any = None
        self.validator: Any = None
        self.drift_detector: Any = None
        self.model_registry: Any = None
        self.inference_engine: Any = None
        self.feature_engineer: Any = None
        self.batch_predictor: Any = None

        # Model type registry - populated during startup
        self.model_types = {}

        # Lifecycle state
        self.active_models = {}
        self.training_jobs = {}
        self.monitoring_jobs = {}

        # Add required dependencies
        self.add_dependency("ModelRegistryService")
        self.add_dependency("FeatureEngineeringService")
        self.add_dependency("TrainingService")
        self.add_dependency("ModelValidationService")
        self.add_dependency("DriftDetectionService")
        self.add_dependency("InferenceService")
        self.add_dependency("BatchPredictionService")

    async def _do_start(self) -> None:
        """Start the model manager service."""
        await super()._do_start()

        # Resolve dependencies
        self.model_registry_service = self.resolve_dependency("ModelRegistryService")
        self.feature_engineering_service = self.resolve_dependency("FeatureEngineeringService")
        self.training_service = self.resolve_dependency("TrainingService")
        self.validation_service = self.resolve_dependency("ModelValidationService")
        self.drift_detection_service = self.resolve_dependency("DriftDetectionService")
        self.inference_service = self.resolve_dependency("InferenceService")
        self.batch_prediction_service = self.resolve_dependency("BatchPredictionService")
        
        # Map legacy references to actual services
        self.trainer = self.training_service
        self.validator = self.validation_service
        self.drift_detector = self.drift_detection_service
        self.model_registry = self.model_registry_service
        self.inference_engine = self.inference_service
        self.feature_engineer = self.feature_engineering_service
        self.batch_predictor = self.batch_prediction_service

        # Initialize available model types (these would be registered dynamically)
        self.model_types = {
            "price_predictor": "PricePredictor",
            "direction_classifier": "DirectionClassifier",
            "volatility_forecaster": "VolatilityForecaster",
            "regime_detector": "RegimeDetector",
        }

        self._logger.info(
            "Model manager service started successfully",
            config=self.mm_config.dict(),
            available_model_types=list(self.model_types.keys()),
            dependencies_resolved=7,
        )

    async def _do_stop(self) -> None:
        """Stop the model manager service."""
        # Clean up active monitoring jobs
        for model_name in list(self.monitoring_jobs.keys()):
            await self._stop_model_monitoring(model_name)

        await super()._do_stop()

    @dec.enhance(log=True, monitor=True, log_level="info")
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

            self._logger.info(
                "Starting model creation and training",
                model_type=model_type,
                model_name=model_name,
                symbol=symbol,
                training_samples=len(training_data),
            )

            # Create model instance (this would be handled by a model factory service)
            # For now, we'll use a placeholder implementation
            model_params = model_params or {}

            # This should be replaced with proper model factory service
            model_instance_info = await self._create_model_instance(model_type, model_name, model_params)
            
            # For now, create a placeholder model object
            # In production, this would come from a model factory service
            model = model_instance_info  # Placeholder model

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

                self._logger.info(
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
                self._logger.warning(
                    "Model validation failed, not registering", model_name=model_name
                )

                return {
                    "success": False,
                    "model_name": model_name,
                    "training_result": training_result,
                    "validation_result": validation_result,
                    "reason": "validation_failed",
                }

        except Exception as e:
            self._logger.error(
                "Model creation and training failed",
                model_type=model_type,
                model_name=model_name,
                error=str(e),
            )
            raise ValidationError(f"Model creation and training failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
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
            self._logger.info(
                "Starting model deployment",
                model_name=model_name,
                deployment_stage=deployment_stage,
            )

            # Get model from registry service
            model_info = await self.model_registry_service.get_latest_model(model_name)
            if not model_info:
                raise ValidationError(f"Model {model_name} not found in registry")

            # Load model
            model = await self.model_registry_service.load_model(model_info["id"])

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

            self._logger.info(
                "Model deployment completed",
                model_name=model_name,
                deployment_stage=deployment_stage,
                model_id=model_info["id"],
            )

            return deployment_result

        except Exception as e:
            self._logger.error(
                "Model deployment failed",
                model_name=model_name,
                deployment_stage=deployment_stage,
                error=str(e),
            )
            raise ValidationError(f"Model deployment failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
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
            self._logger.info(
                "Starting model performance monitoring",
                model_name=model_name,
                monitoring_samples=len(monitoring_data),
            )

            # Get model
            if model_name not in self.active_models:
                model_info = await self.model_registry_service.get_latest_model(model_name)
                if not model_info:
                    raise ValidationError(f"Model {model_name} not found")
                model = await self.model_registry_service.load_model(model_info["id"])
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
                self._logger.warning("No reference data for feature drift detection")

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
                reference_performance = None
                if hasattr(self.drift_detector, "reference_data"):
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

            self._logger.info(
                "Model performance monitoring completed",
                model_name=model_name,
                overall_drift_detected=overall_drift_detected,
                alerts_generated=len(alerts),
            )

            return final_monitoring_result

        except Exception as e:
            self._logger.error(
                "Model performance monitoring failed", model_name=model_name, error=str(e)
            )
            raise ValidationError(f"Model performance monitoring failed: {e}")

    @dec.enhance(log=True, monitor=True, log_level="info")
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
            self._logger.info("Starting model retirement", model_name=model_name, reason=reason)

            # Remove from inference engine
            await self.inference_engine.unload_model(model_name)

            # Stop monitoring
            if model_name in self.monitoring_jobs:
                await self._stop_model_monitoring(model_name)

            # Update registry status
            model_info = await self.model_registry_service.get_latest_model(model_name)
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

            self._logger.info("Model retirement completed", model_name=model_name, reason=reason)

            return retirement_result

        except Exception as e:
            self._logger.error("Model retirement failed", model_name=model_name, error=str(e))
            raise ValidationError(f"Model retirement failed: {e}")

    async def _prepare_and_train_model(
        self,
        model: Any,
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
            self._logger.error(f"Model training preparation failed: {e}")
            raise ValidationError(f"Model training preparation failed: {e}")

    async def _validate_trained_model(
        self, model: Any, test_data: tuple[pd.DataFrame, pd.Series]
    ) -> dict[str, Any]:
        """Validate trained model."""
        try:
            X_test, y_test = test_data

            # Production readiness validation
            validation_result = self.validator.validate_production_readiness(model, X_test, y_test)

            return validation_result

        except Exception as e:
            self._logger.error(f"Model validation failed: {e}")
            raise ValidationError(f"Model validation failed: {e}")

    async def _register_model(
        self,
        model: Any,
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
            self._logger.error(f"Model registration failed: {e}")
            raise ValidationError(f"Model registration failed: {e}")

    async def _pre_deployment_validation(
        self, model: Any, model_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform pre-deployment validation."""
        try:
            checks = {
                "model_trained": model.is_trained,
                "has_required_methods": all(
                    hasattr(model, method) for method in ["predict", "evaluate", "prepare_data"]
                ),
                "model_registered": model_info is not None,
                "has_feature_names": bool(getattr(model, "feature_names", [])),
                "has_metrics": bool(getattr(model, "metrics", {})),
            }

            # Check validation recency
            validation_timestamp = model_info.get("metadata", {}).get("last_validation", None)
            if validation_timestamp:
                validation_age = (
                    datetime.utcnow() - datetime.fromisoformat(validation_timestamp)
                ).days
                checks["recent_validation"] = validation_age <= 7  # Validated within last 7 days
            else:
                checks["recent_validation"] = False

            # Check minimum performance thresholds
            metrics = getattr(model, "metrics", {})
            if model.model_type.endswith("classifier"):
                checks["min_accuracy"] = metrics.get("val_accuracy", 0) >= 0.6
                checks["min_f1_score"] = metrics.get("val_f1_score", 0) >= 0.5
            else:
                checks["max_mse"] = metrics.get("val_mse", float("inf")) <= 1.0
                checks["min_r2"] = metrics.get("val_r2_score", 0) >= 0.3

            issues = [check_name for check_name, passed in checks.items() if not passed]

            return {"ready_for_deployment": len(issues) == 0, "checks": checks, "issues": issues}

        except Exception as e:
            self._logger.error(f"Pre-deployment validation failed: {e}")
            return {
                "ready_for_deployment": False,
                "checks": {},
                "issues": [f"Validation error: {e}"],
            }

    async def _start_model_monitoring(self, model_name: str, model: Any):
        """Start monitoring for a deployed model."""
        try:
            # Create monitoring job (placeholder implementation)
            self.monitoring_jobs[model_name] = {
                "model": model,
                "start_time": datetime.utcnow(),
                "status": "active",
            }

            self._logger.info(f"Monitoring started for model {model_name}")

        except Exception as e:
            self._logger.error(f"Failed to start monitoring for {model_name}: {e}")

    async def _stop_model_monitoring(self, model_name: str):
        """Stop monitoring for a model."""
        try:
            if model_name in self.monitoring_jobs:
                self.monitoring_jobs[model_name]["status"] = "stopped"
                self.monitoring_jobs[model_name]["stop_time"] = datetime.utcnow()
                del self.monitoring_jobs[model_name]

            self._logger.info(f"Monitoring stopped for model {model_name}")

        except Exception as e:
            self._logger.error(f"Failed to stop monitoring for {model_name}: {e}")

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
            self._logger.error(f"Alert generation failed: {e}")
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

    async def get_model_status(self, model_name: str) -> dict[str, Any] | None:
        """Get status information for a specific model."""
        if model_name in self.active_models:
            return {"status": "active", "details": self.active_models[model_name]}

        # Check registry
        model_info = await self.model_registry_service.get_latest_model(model_name)
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
            self._logger.error(f"Health check failed: {e}")
            return {"timestamp": datetime.utcnow(), "status": "unhealthy", "error": str(e)}

    # Helper Methods
    async def _create_model_instance(
        self, model_type: str, model_name: str, model_params: dict[str, Any]
    ) -> Any:
        """Create model instance through proper factory service."""
        # This should be implemented with a proper model factory service
        # For now, return a placeholder
        return {
            "model_type": model_type,
            "model_name": model_name,
            "params": model_params,
            "status": "created",
        }

    # Service Health and Metrics
    async def _service_health_check(self) -> "HealthStatus":
        """Model manager service specific health check."""
        from src.core.types import HealthStatus

        try:
            # Check dependencies
            required_services = [
                self.model_registry_service,
                self.feature_engineering_service,
                self.training_service,
                self.validation_service,
                self.drift_detection_service,
                self.inference_service,
                self.batch_prediction_service,
            ]

            if not all(required_services):
                return HealthStatus.UNHEALTHY

            # Check if we have too many active models
            if len(self.active_models) > self.mm_config.max_active_models:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Model manager service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_model_manager_metrics(self) -> dict[str, Any]:
        """Get model manager service metrics."""
        return {
            "active_models_count": len(self.active_models),
            "training_jobs_count": len(self.training_jobs),
            "monitoring_jobs_count": len(self.monitoring_jobs),
            "available_model_types": list(self.model_types.keys()),
            "model_monitoring_enabled": self.mm_config.enable_model_monitoring,
            "auto_retraining_enabled": self.mm_config.enable_auto_retraining,
            "max_active_models": self.mm_config.max_active_models,
        }

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate model manager service configuration."""
        try:
            mm_config_dict = config.get("model_manager", {})
            ModelManagerConfig(**mm_config_dict)
            return True
        except Exception as e:
            self._logger.error(
                "Model manager service configuration validation failed", error=str(e)
            )
            return False
