"""
Dependency Injection Registration for ML Module.

This module registers all ML services with the DI container to ensure
proper service layer separation and dependency management.
"""

from typing import TYPE_CHECKING

from src.core.types.base import ConfigDict

if TYPE_CHECKING:
    from src.core.di import DependencyContainer

from src.ml.factory import ModelFactory
from src.ml.feature_engineering import FeatureEngineeringService
from src.ml.inference.inference_engine import InferenceService
from src.ml.model_manager import ModelManagerService
from src.ml.registry.model_registry import ModelRegistryService
from src.ml.repository import MLRepository
from src.ml.service import MLService
from src.ml.services import (
    BatchPredictionService,
    DriftDetectionService,
    ModelValidationService,
    TrainingService,
)


def register_ml_services(container: "DependencyContainer", config: ConfigDict) -> None:
    """
    Register all ML services with the dependency injection container.

    Args:
        container: The DI container instance
        config: Application configuration
    """
    # Register repository first as singleton
    container.register("MLRepository", lambda: MLRepository(config=config), singleton=True)

    # Register model factory as singleton
    container.register("ModelFactory", lambda: ModelFactory(config=config), singleton=True)

    # Register core ML services as singletons (stateful services)
    container.register(
        "FeatureEngineeringService",
        lambda: FeatureEngineeringService(config=config),
        singleton=True,
    )

    container.register(
        "ModelRegistryService", lambda: ModelRegistryService(config=config), singleton=True
    )

    container.register("InferenceService", lambda: InferenceService(config=config), singleton=True)

    container.register(
        "ModelManagerService", lambda: ModelManagerService(config=config), singleton=True
    )

    container.register("MLService", lambda: MLService(config=config), singleton=True)

    # Register mock services as singletons
    container.register(
        "ModelValidationService", lambda: ModelValidationService(config=config), singleton=True
    )

    container.register(
        "DriftDetectionService", lambda: DriftDetectionService(config=config), singleton=True
    )

    container.register("TrainingService", lambda: TrainingService(config=config), singleton=True)

    container.register(
        "BatchPredictionService", lambda: BatchPredictionService(config=config), singleton=True
    )

    # Note: Service aliases would need to be handled by the container implementation
    # if such functionality is required


def get_ml_service_dependencies() -> dict[str, list[str]]:
    """
    Get ML service dependencies for proper initialization order.

    Returns:
        Dictionary mapping service names to their dependencies
    """
    return {
        "MLRepository": [],
        "ModelFactory": [],
        "FeatureEngineeringService": [
            "DataService",
        ],
        "ModelRegistryService": ["DataService", "MLRepository"],
        "InferenceService": ["ModelRegistryService", "FeatureEngineeringService"],
        "ModelValidationService": [],
        "DriftDetectionService": [],
        "TrainingService": [],
        "BatchPredictionService": [],
        "ModelManagerService": [
            "ModelRegistryService",
            "FeatureEngineeringService",
            "TrainingService",
            "ModelValidationService",
            "DriftDetectionService",
            "InferenceService",
            "BatchPredictionService",
            "ModelFactory",
        ],
        "MLService": [
            "DataService",
            "FeatureEngineeringService",
            "ModelRegistryService",
            "InferenceService",
            "FeatureStoreService",
        ],
    }
