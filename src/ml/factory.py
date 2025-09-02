"""
ML Model Factory.

This module provides factory pattern implementation for creating ML models
with proper dependency injection and configuration management.
"""

from typing import Any

from src.core.base.factory import BaseFactory
from src.core.exceptions import CreationError, RegistrationError
from src.core.types.base import ConfigDict
from src.ml.interfaces import IModelFactory
from src.ml.models.base_model import BaseMLModel
from src.ml.models.direction_classifier import DirectionClassifier
from src.ml.models.price_predictor import PricePredictor
from src.ml.models.regime_detector import RegimeDetector
from src.ml.models.volatility_forecaster import VolatilityForecaster
from src.utils.constants import ML_MODEL_CONSTANTS


class ModelFactory(BaseFactory[BaseMLModel], IModelFactory):
    """
    Factory for creating ML model instances.

    This factory provides a centralized way to create ML models with proper
    dependency injection and configuration management.
    """

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        """
        Initialize the model factory.

        Args:
            config: Factory configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            product_type=BaseMLModel,
            name="ModelFactory",
            config=config,
            correlation_id=correlation_id,
        )

        # Register default model creators
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Register default ML model creators."""
        try:
            # Register direction classifier
            self.register(
                "DirectionClassifier",
                DirectionClassifier,
                singleton=False,
                metadata={"description": "Binary direction prediction model"},
            )

            # Register price predictor
            self.register(
                "PricePredictor",
                PricePredictor,
                singleton=False,
                metadata={"description": "Continuous price prediction model"},
            )

            # Register volatility forecaster
            self.register(
                "VolatilityForecaster",
                VolatilityForecaster,
                singleton=False,
                metadata={"description": "Volatility forecasting model"},
            )

            # Register regime detector
            self.register(
                "RegimeDetector",
                RegimeDetector,
                singleton=False,
                metadata={"description": "Market regime detection model"},
            )

            self._logger.info(
                "Default ML models registered", factory=self._name, model_count=len(self._creators)
            )

        except Exception as e:
            self._logger.error(
                "Failed to register default models", factory=self._name, error=str(e)
            )
            raise RegistrationError(f"Failed to register default models: {e}") from e

    def create_model(
        self,
        model_type: str,
        model_name: str | None = None,
        version: str = ML_MODEL_CONSTANTS["default_model_version"],
        config: ConfigDict | None = None,
        **kwargs: Any,
    ) -> BaseMLModel:
        """
        Create a model instance of the specified type.

        Args:
            model_type: Type of model to create
            model_name: Name for the model instance
            version: Model version
            config: Model-specific configuration
            **kwargs: Additional creation parameters

        Returns:
            Created model instance

        Raises:
            CreationError: If model creation fails
        """
        try:
            # Generate model name if not provided
            if model_name is None:
                model_name = f"{model_type}_model"

            # Merge configuration
            creation_config = self._config.copy() if self._config else {}
            if config:
                creation_config.update(config)

            # Create model with proper parameters
            model = self.create(
                model_type, model_name=model_name, version=version, config=creation_config, **kwargs
            )

            self._logger.info(
                "Model created successfully",
                factory=self._name,
                model_type=model_type,
                model_name=model_name,
                version=version,
            )

            return model

        except Exception as e:
            self._logger.error(
                "Model creation failed",
                factory=self._name,
                model_type=model_type,
                model_name=model_name,
                error=str(e),
            )
            raise CreationError(f"Failed to create model '{model_type}': {e}") from e

    def get_available_models(self) -> list[str]:
        """
        Get list of available model types.

        Returns:
            List of registered model type names
        """
        return self.list_registered()

    def get_model_info(self, model_type: str) -> dict[str, Any] | None:
        """
        Get information about a specific model type.

        Args:
            model_type: Model type name

        Returns:
            Model type information or None if not found
        """
        return self.get_creator_info(model_type)

    def register_custom_model(
        self,
        name: str,
        model_class: type[BaseMLModel],
        config: dict[str, Any] | None = None,
        singleton: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a custom model type.

        Args:
            name: Model type name
            model_class: Model class to register
            config: Default configuration for this model type
            singleton: Whether to create singleton instances
            metadata: Additional metadata about the model type

        Raises:
            RegistrationError: If registration fails
        """
        try:
            # Validate model class
            if not issubclass(model_class, BaseMLModel):
                raise RegistrationError(
                    f"Model class {model_class.__name__} must inherit from BaseMLModel"
                )

            self.register(name, model_class, config, singleton, metadata)

            self._logger.info(
                "Custom model registered",
                factory=self._name,
                model_type=name,
                model_class=model_class.__name__,
            )

        except Exception as e:
            self._logger.error(
                "Failed to register custom model", factory=self._name, model_type=name, error=str(e)
            )
            raise RegistrationError(f"Failed to register custom model '{name}': {e}") from e
