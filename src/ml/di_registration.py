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
from src.ml.inference.model_cache import ModelCacheService
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
from src.ml.store.feature_store import FeatureStoreService
from src.ml.validation_service import MLValidationService
from src.ml.integration_service import MLIntegrationService


def register_ml_services(container: "DependencyContainer", config: ConfigDict) -> None:
    """
    Register all ML services with the dependency injection container.

    Args:
        container: The DI container instance
        config: Application configuration
    """
    # Convert Config object to dict if needed (for Pydantic Config objects)
    if hasattr(config, 'model_dump'):
        config_dict = config.model_dump()
    elif hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif hasattr(config, 'dict'):
        config_dict = config.dict()
    else:
        config_dict = config

    # Create a minimal injector wrapper for dependency resolution
    # Services need an object with a resolve() method, not just a container with get()
    class InjectorWrapper:
        def __init__(self, container):
            self._container = container

        def resolve(self, name: str):
            """Resolve a dependency by name."""
            return self._container.get(name)

    injector = InjectorWrapper(container)

    # Register ML repository as singleton (uses DataService internally)
    def create_ml_repository():
        return MLRepository(config=config_dict)
    container.register("MLRepository", create_ml_repository, singleton=True)

    # Register model factory as singleton
    def create_model_factory():
        factory = ModelFactory(config=config_dict)
        if hasattr(factory, 'configure_dependencies'):
            factory.configure_dependencies(container)  # Inject container for dependency resolution
        return factory
    container.register("ModelFactory", create_model_factory, singleton=True)

    # Register cache service as singleton (stateful)
    def create_model_cache_service():
        service = ModelCacheService(config=config_dict)
        service.configure_dependencies(injector)  # Inject wrapper for dependency resolution
        return service
    container.register("ModelCacheService", create_model_cache_service, singleton=True)

    # Register feature store service as singleton (stateful)
    def create_feature_store_service():
        return FeatureStoreService(config=config_dict)
    container.register("FeatureStoreService", create_feature_store_service, singleton=True)

    # Register core ML services as singletons (stateful services)
    def create_feature_engineering_service():
        service = FeatureEngineeringService(config=config_dict)
        service.configure_dependencies(injector)  # Inject wrapper for dependency resolution
        return service
    container.register("FeatureEngineeringService", create_feature_engineering_service, singleton=True)

    def create_model_registry_service():
        service = ModelRegistryService(config=config_dict)
        service.configure_dependencies(injector)  # Inject wrapper for dependency resolution
        return service
    container.register("ModelRegistryService", create_model_registry_service, singleton=True)

    def create_inference_service():
        service = InferenceService(config=config_dict)
        service.configure_dependencies(injector)  # Inject wrapper for dependency resolution
        return service
    container.register("InferenceService", create_inference_service, singleton=True)

    def create_model_manager_service():
        return ModelManagerService(config=config_dict)
    container.register("ModelManagerService", create_model_manager_service, singleton=True)

    def create_ml_service():
        service = MLService(config=config_dict)
        service.configure_dependencies(injector)  # Inject wrapper for dependency resolution
        return service
    container.register("MLService", create_ml_service, singleton=True)

    # Register mock services as singletons
    def create_model_validation_service():
        return ModelValidationService(config=config_dict)
    container.register("ModelValidationService", create_model_validation_service, singleton=True)

    def create_drift_detection_service():
        return DriftDetectionService(config=config_dict)
    container.register("DriftDetectionService", create_drift_detection_service, singleton=True)

    def create_training_service():
        return TrainingService(config=config_dict)
    container.register("TrainingService", create_training_service, singleton=True)

    def create_batch_prediction_service():
        return BatchPredictionService(config=config_dict)
    container.register("BatchPredictionService", create_batch_prediction_service, singleton=True)

    # Register ML validation service as singleton (stateless but configuration-dependent)
    def create_ml_validation_service():
        return MLValidationService(config=config_dict)
    container.register("MLValidationService", create_ml_validation_service, singleton=True)

    # Register ML integration service as singleton (stateless but configuration-dependent)
    def create_ml_integration_service():
        return MLIntegrationService(config=config_dict)
    container.register("MLIntegrationService", create_ml_integration_service, singleton=True)

    # Note: Service aliases would need to be handled by the container implementation
    # if such functionality is required


def get_ml_service_dependencies() -> dict[str, list[str]]:
    """
    Get ML service dependencies for proper initialization order.

    Returns:
        Dictionary mapping service names to their dependencies
    """
    return {
        "MLRepository": ["MLDataService"],
        "ModelFactory": [],
        "ModelCacheService": ["MLDataService"],
        "FeatureStoreService": ["MLDataService"],
        "FeatureEngineeringService": [
            "MLDataService",
        ],
        "ModelRegistryService": ["MLDataService", "MLRepository"],
        "InferenceService": ["ModelCacheService", "ModelRegistryService", "FeatureEngineeringService"],
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
        "MLValidationService": [],  # No dependencies - stateless validation service
        "MLIntegrationService": [],  # No dependencies - stateless integration service
        "MLService": [
            "MLDataService",
            "MLValidationService",
            "MLIntegrationService",
            "FeatureEngineeringService",
            "ModelRegistryService",
            "InferenceService",
            "FeatureStoreService",
        ],
    }
