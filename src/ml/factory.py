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

        # Container for dependency injection
        self._container = None

        # Register default model creators
        self._register_default_models()

    def set_container(self, container: Any) -> None:
        """
        Set the dependency injection container.
        
        Args:
            container: DI container for resolving dependencies
        """
        self._container = container

    def _register_default_models(self) -> None:
        """Register default ML model creators."""
        try:
            # Register direction classifier with dependency injection
            self.register(
                "DirectionClassifier",
                self._create_direction_classifier,
                singleton=False,
                metadata={"description": "Binary direction prediction model"},
            )

            # Register price predictor with dependency injection
            self.register(
                "PricePredictor",
                self._create_price_predictor,
                singleton=False,
                metadata={"description": "Continuous price prediction model"},
            )

            # Register volatility forecaster with dependency injection
            self.register(
                "VolatilityForecaster",
                self._create_volatility_forecaster,
                singleton=False,
                metadata={"description": "Volatility forecasting model"},
            )

            # Register regime detector with dependency injection
            self.register(
                "RegimeDetector",
                self._create_regime_detector,
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

    def _create_direction_classifier(self, **kwargs: Any) -> DirectionClassifier:
        """Create DirectionClassifier with dependency injection."""
        config = self._get_injected_config(**kwargs)
        return DirectionClassifier(config=config, **kwargs)

    def _create_price_predictor(self, **kwargs: Any) -> PricePredictor:
        """Create PricePredictor with dependency injection."""
        config = self._get_injected_config(**kwargs)
        return PricePredictor(config=config, **kwargs)

    def _create_volatility_forecaster(self, **kwargs: Any) -> VolatilityForecaster:
        """Create VolatilityForecaster with dependency injection."""
        config = self._get_injected_config(**kwargs)
        return VolatilityForecaster(config=config, **kwargs)

    def _create_regime_detector(self, **kwargs: Any) -> RegimeDetector:
        """Create RegimeDetector with dependency injection."""
        config = self._get_injected_config(**kwargs)
        return RegimeDetector(config=config, **kwargs)

    def _get_injected_config(self, **kwargs: Any) -> Any:
        """Get config with dependency injection."""
        # Use provided config or get from container
        if "config" in kwargs:
            return kwargs.pop("config")

        if self._container and hasattr(self._container, "get"):
            try:
                return self._container.get("Config", self._config)
            except Exception:
                pass

        return self._config

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

            # Add model metadata to kwargs
            kwargs["model_name"] = model_name
            kwargs["version"] = version
            if config:
                kwargs["config"] = config

            # Use base factory create method with injected dependencies
            model = self.create(model_type, **kwargs)

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

            # Create injected creator function
            def create_custom_model(**kwargs: Any) -> BaseMLModel:
                injected_config = self._get_injected_config(**kwargs)
                return model_class(config=injected_config, **kwargs)

            self.register(name, create_custom_model, config, singleton, metadata)

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
