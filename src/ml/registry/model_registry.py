"""
Model Registry Service for ML Model Versioning and Storage Management.

This module provides comprehensive model versioning, storage, and lifecycle management
for ML models using proper service patterns without direct database access.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class ModelRegistryConfig(BaseModel):
    """Configuration for model registry service."""

    registry_path: str = Field(
        default="./data/models/registry", description="Model registry storage path"
    )
    artifact_path: str = Field(
        default="./data/models/artifacts", description="Model artifacts storage path"
    )
    enable_versioning: bool = Field(default=True, description="Enable model versioning")
    enable_persistence: bool = Field(default=True, description="Enable model persistence")
    max_versions_per_model: int = Field(
        default=10, description="Maximum versions to keep per model"
    )
    cache_ttl_hours: int = Field(default=24, description="Model cache TTL in hours")
    enable_audit_trail: bool = Field(default=True, description="Enable audit trail logging")
    compression_enabled: bool = Field(default=True, description="Enable model compression")
    background_cleanup_interval: int = Field(
        default=3600, description="Background cleanup interval in seconds"
    )


class ModelMetadata(BaseModel):
    """Model metadata structure."""

    model_id: str
    name: str
    model_type: str
    version: str
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    stage: str = "development"  # development, staging, production
    metrics: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    training_data_info: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    file_path: str | None = None


class ModelRegistrationRequest(BaseModel):
    """Request for model registration."""

    model: Any  # The actual model object
    name: str
    model_type: str
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    stage: str = "development"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelLoadRequest(BaseModel):
    """Request for model loading."""

    model_id: str | None = None
    model_name: str | None = None
    model_type: str | None = None
    version: str | None = None
    stage: str | None = None
    use_cache: bool = True


class ModelRegistryService(BaseService):
    """
    Model registry service for managing ML model versions and storage.

    This service provides centralized management of ML models including versioning,
    storage, retrieval, and lifecycle management using proper service patterns
    without direct database access.

    All data persistence goes through DataService dependency.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the model registry service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="ModelRegistryService",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse model registry configuration
        registry_config_dict = (config or {}).get("model_registry", {})
        self.registry_config = ModelRegistryConfig(**registry_config_dict)

        # Service dependencies - resolved during startup
        self.data_service: Any = None
        self.ml_repository: Any = None

        # Internal state
        self._model_cache: dict[str, tuple[ModelMetadata, Any, datetime]] = {}
        self._model_metadata_cache: dict[str, tuple[ModelMetadata, datetime]] = {}

        # Thread pool for I/O operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Background cleanup task
        self._cleanup_task: asyncio.Task | None = None

        # Setup storage paths
        if self.registry_config.enable_persistence:
            self._registry_path = Path(self.registry_config.registry_path)
            self._artifact_path = Path(self.registry_config.artifact_path)
            self._registry_path.mkdir(parents=True, exist_ok=True)
            self._artifact_path.mkdir(parents=True, exist_ok=True)

        # Add required dependencies
        self.add_dependency("DataService")
        self.add_dependency("MLRepository")

    async def _do_start(self) -> None:
        """Start the model registry service."""
        await super()._do_start()

        # Resolve dependencies
        self.data_service = self.resolve_dependency("DataService")
        self.ml_repository = self.resolve_dependency("MLRepository")

        # Load existing model metadata if persistence is enabled
        if self.registry_config.enable_persistence:
            await self._load_existing_metadata()

        # Start background cleanup task
        if self.registry_config.background_cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())

        self._logger.info(
            "Model registry service started successfully",
            config=self.registry_config.dict(),
            cached_models=len(self._model_cache),
        )

    async def _do_stop(self) -> None:
        """Stop the model registry service."""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        await super()._do_stop()

    # Core Model Registry Operations
    @dec.enhance(log=True, monitor=True, log_level="info")
    async def register_model(self, request: ModelRegistrationRequest) -> str:
        """
        Register a new model version in the registry.

        Args:
            request: Model registration request

        Returns:
            Model ID in the registry

        Raises:
            ModelError: If registration fails
            ValidationError: If model validation fails
        """
        return await self.execute_with_monitoring(
            "register_model",
            self._register_model_impl,
            request,
        )

    async def _register_model_impl(self, request: ModelRegistrationRequest) -> str:
        """Internal model registration implementation."""
        try:
            # Validate model
            if not hasattr(request.model, "predict"):
                raise ValidationError("Model must have a predict method")

            # Generate model ID and version
            model_id, version = await self._generate_model_id_and_version(
                request.name, request.model_type
            )

            # Create model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=request.name,
                model_type=request.model_type,
                version=version,
                description=request.description,
                tags=request.tags,
                stage=request.stage,
                parameters=request.metadata.get("hyperparameters", {}),
                feature_names=getattr(request.model, "feature_names_", []),
                training_data_info=request.metadata.get("training_data", {}),
            )

            # Extract model metrics if available
            if hasattr(request.model, "metrics_") and request.model.metrics_:
                metadata.metrics = request.model.metrics_

            # Save model to file
            model_file_path = None
            if self.registry_config.enable_persistence:
                model_filename = f"{request.name}_{request.model_type}_{version}.pkl"
                model_file_path = self._artifact_path / model_filename
                await self._save_model_to_file(request.model, model_file_path)
                metadata.file_path = str(model_file_path)

            # Store model metadata through repository
            await self._store_model_metadata(metadata)

            # Cache the model and metadata
            await self._cache_model(model_id, metadata, request.model)

            # Save registry entry to file for backup
            if self.registry_config.enable_persistence:
                await self._save_registry_entry(metadata)

            # Log audit trail
            if self.registry_config.enable_audit_trail:
                await self._log_audit_event(
                    "model_registered",
                    model_id,
                    {
                        "name": request.name,
                        "type": request.model_type,
                        "version": version,
                        "stage": request.stage,
                    },
                )

            self._logger.info(
                "Model registered successfully",
                model_id=model_id,
                model_name=request.name,
                version=version,
                stage=request.stage,
            )

            return model_id

        except Exception as e:
            self._logger.error(
                "Model registration failed",
                model_name=request.name,
                model_type=request.model_type,
                error=str(e),
            )
            raise ModelError(f"Failed to register model {request.name}: {e}") from e

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def load_model(self, request: ModelLoadRequest) -> dict[str, Any]:
        """
        Load a model from the registry.

        Args:
            request: Model load request

        Returns:
            Dictionary containing model and metadata

        Raises:
            ModelError: If model loading fails
        """
        return await self.execute_with_monitoring(
            "load_model",
            self._load_model_impl,
            request,
        )

    async def _load_model_impl(self, request: ModelLoadRequest) -> dict[str, Any]:
        """Internal model loading implementation."""
        try:
            # Find model metadata
            metadata = await self._find_model_metadata(
                model_id=request.model_id,
                model_name=request.model_name,
                model_type=request.model_type,
                version=request.version,
                stage=request.stage,
            )

            if not metadata:
                criteria = (
                    f"id={request.model_id}, name={request.model_name}, "
                    f"type={request.model_type}, version={request.version}, stage={request.stage}"
                )
                raise ModelError(f"Model not found with criteria: {criteria}")

            # Check cache first
            if request.use_cache:
                cached_model = await self._get_cached_model(metadata.model_id)
                if cached_model is not None:
                    return {
                        "model_id": metadata.model_id,
                        "model": cached_model,
                        "metadata": metadata.dict(),
                        "source": "cache",
                    }

            # Load model from file
            if not metadata.file_path:
                raise ModelError(f"Model file path not found for model {metadata.model_id}")

            model_path = Path(metadata.file_path)
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")

            # Load model in thread pool to avoid blocking
            model = await self._load_model_from_file(model_path)

            # Cache the loaded model
            if request.use_cache:
                await self._cache_model(metadata.model_id, metadata, model)

            # Log audit trail
            if self.registry_config.enable_audit_trail:
                await self._log_audit_event(
                    "model_loaded",
                    metadata.model_id,
                    {
                        "version": metadata.version,
                        "stage": metadata.stage,
                    },
                )

            self._logger.info(
                "Model loaded successfully",
                model_id=metadata.model_id,
                model_name=metadata.name,
                version=metadata.version,
            )

            return {
                "model_id": metadata.model_id,
                "model": model,
                "metadata": metadata.dict(),
                "source": "file",
            }

        except Exception as e:
            self._logger.error(
                "Model loading failed",
                model_id=request.model_id,
                model_name=request.model_name,
                error=str(e),
            )
            raise ModelError(f"Failed to load model: {e}") from e

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def list_models(
        self,
        model_type: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List all models in the registry.

        Args:
            model_type: Filter by model type
            stage: Filter by stage
            active_only: Only return active models

        Returns:
            List of model metadata dictionaries
        """
        return await self.execute_with_monitoring(
            "list_models",
            self._list_models_impl,
            model_type,
            stage,
            active_only,
        )

    async def _list_models_impl(
        self, model_type: str | None, stage: str | None, active_only: bool
    ) -> list[dict[str, Any]]:
        """Internal model listing implementation."""
        try:
            # Get all model metadata from repository
            models_data = await self.ml_repository.get_all_models(
                model_type=model_type,
                stage=stage,
                active_only=active_only,
            )

            # Convert to list of dictionaries
            models_list = []
            for model_data in models_data:
                model_info = model_data.copy()
                models_list.append(model_info)

            self._logger.info(
                "Models listed successfully",
                total_models=len(models_list),
                model_type=model_type,
                stage=stage,
                active_only=active_only,
            )

            return models_list

        except Exception as e:
            self._logger.error("Failed to list models", error=str(e))
            raise ModelError(f"Failed to list models: {e}") from e

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def promote_model(self, model_id: str, stage: str, description: str = "") -> bool:
        """
        Promote a model to a different stage.

        Args:
            model_id: Model ID to promote
            stage: Target stage (staging, production)
            description: Promotion description

        Returns:
            True if promotion successful

        Raises:
            ModelError: If promotion fails
        """
        return await self.execute_with_monitoring(
            "promote_model",
            self._promote_model_impl,
            model_id,
            stage,
            description,
        )

    async def _promote_model_impl(self, model_id: str, stage: str, description: str) -> bool:
        """Internal model promotion implementation."""
        try:
            # Get current model metadata
            metadata = await self._get_model_metadata(model_id)
            if not metadata:
                raise ModelError(f"Model not found: {model_id}")

            old_stage = metadata.stage

            # Update stage
            metadata.stage = stage
            metadata.updated_at = datetime.now(timezone.utc)

            # Update in data service
            await self._update_model_metadata(metadata)

            # Update cache
            if model_id in self._model_metadata_cache:
                self._model_metadata_cache[model_id] = (metadata, datetime.now(timezone.utc))

            # Update registry file
            if self.registry_config.enable_persistence:
                await self._save_registry_entry(metadata)

            # Log audit trail
            if self.registry_config.enable_audit_trail:
                await self._log_audit_event(
                    "model_promoted",
                    model_id,
                    {
                        "old_stage": old_stage,
                        "new_stage": stage,
                        "description": description,
                    },
                )

            self._logger.info(
                "Model promoted successfully",
                model_id=model_id,
                old_stage=old_stage,
                new_stage=stage,
            )

            return True

        except Exception as e:
            self._logger.error(
                "Model promotion failed",
                model_id=model_id,
                stage=stage,
                error=str(e),
            )
            raise ModelError(f"Failed to promote model {model_id}: {e}") from e

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def deactivate_model(self, model_id: str, reason: str = "") -> bool:
        """
        Deactivate a model in the registry.

        Args:
            model_id: Model ID to deactivate
            reason: Reason for deactivation

        Returns:
            True if deactivation successful

        Raises:
            ModelError: If deactivation fails
        """
        return await self.execute_with_monitoring(
            "deactivate_model",
            self._deactivate_model_impl,
            model_id,
            reason,
        )

    async def _deactivate_model_impl(self, model_id: str, reason: str) -> bool:
        """Internal model deactivation implementation."""
        try:
            # Get current model metadata
            metadata = await self._get_model_metadata(model_id)
            if not metadata:
                raise ModelError(f"Model not found: {model_id}")

            # Update active status
            metadata.is_active = False
            metadata.updated_at = datetime.now(timezone.utc)

            # Update in data service
            await self._update_model_metadata(metadata)

            # Update cache
            if model_id in self._model_metadata_cache:
                self._model_metadata_cache[model_id] = (metadata, datetime.now(timezone.utc))

            # Remove from model cache since it's no longer active
            if model_id in self._model_cache:
                del self._model_cache[model_id]

            # Update registry file
            if self.registry_config.enable_persistence:
                await self._save_registry_entry(metadata)

            # Log audit trail
            if self.registry_config.enable_audit_trail:
                await self._log_audit_event(
                    "model_deactivated",
                    model_id,
                    {
                        "reason": reason,
                    },
                )

            self._logger.info("Model deactivated successfully", model_id=model_id, reason=reason)

            return True

        except Exception as e:
            self._logger.error("Model deactivation failed", model_id=model_id, error=str(e))
            raise ModelError(f"Failed to deactivate model {model_id}: {e}") from e

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def delete_model(self, model_id: str, remove_files: bool = True) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model ID to delete
            remove_files: Whether to remove model files

        Returns:
            True if deletion successful

        Raises:
            ModelError: If deletion fails
        """
        return await self.execute_with_monitoring(
            "delete_model",
            self._delete_model_impl,
            model_id,
            remove_files,
        )

    async def _delete_model_impl(self, model_id: str, remove_files: bool) -> bool:
        """Internal model deletion implementation."""
        try:
            # Get model metadata
            metadata = await self._get_model_metadata(model_id)
            if not metadata:
                raise ModelError(f"Model not found: {model_id}")

            # Remove files if requested
            if remove_files and self.registry_config.enable_persistence:
                if metadata.file_path:
                    model_path = Path(metadata.file_path)
                    if model_path.exists():
                        await asyncio.get_event_loop().run_in_executor(
                            self._executor, model_path.unlink
                        )

                # Remove registry file
                registry_file = self._registry_path / f"{model_id}.json"
                if registry_file.exists():
                    await asyncio.get_event_loop().run_in_executor(
                        self._executor, registry_file.unlink
                    )

            # Remove from repository
            await self.ml_repository.delete_model(model_id)

            # Remove from caches
            if model_id in self._model_cache:
                del self._model_cache[model_id]
            if model_id in self._model_metadata_cache:
                del self._model_metadata_cache[model_id]

            # Log audit trail
            if self.registry_config.enable_audit_trail:
                await self._log_audit_event(
                    "model_deleted",
                    model_id,
                    {
                        "remove_files": remove_files,
                        "model_name": metadata.name,
                        "version": metadata.version,
                    },
                )

            self._logger.info(
                "Model deleted successfully",
                model_id=model_id,
                remove_files=remove_files,
            )

            return True

        except Exception as e:
            self._logger.error("Model deletion failed", model_id=model_id, error=str(e))
            raise ModelError(f"Failed to delete model {model_id}: {e}") from e

    async def get_model_metrics(self, model_id: str) -> dict[str, Any]:
        """
        Get detailed metrics for a model.

        Args:
            model_id: Model ID

        Returns:
            Dictionary with model metrics and information
        """
        return await self.execute_with_monitoring(
            "get_model_metrics",
            self._get_model_metrics_impl,
            model_id,
        )

    async def _get_model_metrics_impl(self, model_id: str) -> dict[str, Any]:
        """Internal get model metrics implementation."""
        try:
            metadata = await self._get_model_metadata(model_id)
            if not metadata:
                raise ModelError(f"Model not found: {model_id}")

            # Get additional registry info if available
            registry_info = {}
            if self.registry_config.enable_persistence:
                registry_file = self._registry_path / f"{model_id}.json"
                if registry_file.exists():
                    loop = asyncio.get_event_loop()
                    registry_data = await loop.run_in_executor(
                        self._executor, self._load_json_file, registry_file
                    )
                    registry_info = registry_data.get("additional_info", {})

            metrics = {
                "model_id": metadata.model_id,
                "name": metadata.name,
                "type": metadata.model_type,
                "version": metadata.version,
                "stage": metadata.stage,
                "is_active": metadata.is_active,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat(),
                "metrics": metadata.metrics,
                "parameters": metadata.parameters,
                "training_data_info": metadata.training_data_info,
                "feature_count": len(metadata.feature_names),
                "feature_names": metadata.feature_names,
                "registry_info": registry_info,
            }

            return metrics

        except Exception as e:
            self._logger.error("Failed to get model metrics", model_id=model_id, error=str(e))
            raise ModelError(f"Failed to get metrics for model {model_id}: {e}") from e

    # Helper Methods
    async def _generate_model_id_and_version(self, name: str, model_type: str) -> tuple[str, str]:
        """Generate unique model ID and version."""
        # Check existing models
        existing_models = await self.ml_repository.get_models_by_name_and_type(name, model_type)

        if existing_models:
            # Get latest version and increment
            versions = [model.get("version", "1.0.0") for model in existing_models]
            latest_version = max(versions)
            version_parts = latest_version.split(".")
            major, minor, patch = map(int, version_parts)
            new_version = f"{major}.{minor}.{patch + 1}"
        else:
            new_version = "1.0.0"

        # Generate model ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_id = f"{name}_{model_type}_{new_version}_{timestamp}"

        return model_id, new_version

    async def _find_model_metadata(
        self,
        model_id: str | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
    ) -> ModelMetadata | None:
        """Find model metadata based on criteria."""
        if model_id:
            return await self._get_model_metadata(model_id)

        # Search by other criteria
        models_data = await self.ml_repository.find_models(
            name=model_name,
            model_type=model_type,
            version=version,
            stage=stage,
            active_only=True,
        )

        if not models_data:
            return None

        # Return the most recent one
        latest_model = max(models_data, key=lambda x: x.get("created_at", datetime.min))
        return ModelMetadata(**latest_model)

    async def _get_model_metadata(self, model_id: str) -> ModelMetadata | None:
        """Get model metadata by ID."""
        # Check cache first
        if model_id in self._model_metadata_cache:
            metadata, timestamp = self._model_metadata_cache[model_id]
            ttl_hours = self.registry_config.cache_ttl_hours
            if (datetime.now(timezone.utc) - timestamp).total_seconds() < ttl_hours * 3600:
                return metadata
            else:
                del self._model_metadata_cache[model_id]

        # Get from repository
        model_data = await self.ml_repository.get_model_by_id(model_id)
        if not model_data:
            return None

        metadata = ModelMetadata(**model_data)

        # Cache it
        self._model_metadata_cache[model_id] = (metadata, datetime.now(timezone.utc))

        return metadata

    async def _store_model_metadata(self, metadata: ModelMetadata) -> None:
        """Store model metadata through repository."""
        await self.ml_repository.store_model_metadata(metadata.dict())

    async def _update_model_metadata(self, metadata: ModelMetadata) -> None:
        """Update model metadata through repository."""
        await self.ml_repository.update_model_metadata(metadata.model_id, metadata.dict())

    async def _save_model_to_file(self, model: Any, file_path: Path) -> None:
        """Save model to file."""

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._save_pickle_file,
            model,
            file_path,
        )

    def _save_pickle_file(self, obj: Any, file_path: Path) -> None:
        """Save object as pickle file."""
        import pickle

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    async def _load_model_from_file(self, file_path: Path) -> Any:
        """Load model from file."""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._load_pickle_file,
            file_path,
        )

    def _load_pickle_file(self, file_path: Path) -> Any:
        """Load object from pickle file."""
        import pickle

        with open(file_path, "rb") as f:
            return pickle.load(f)

    def _load_json_file(self, file_path: Path) -> dict[str, Any]:
        """Load JSON file."""
        with open(file_path) as f:
            return json.load(f)

    async def _save_registry_entry(self, metadata: ModelMetadata) -> None:
        """Save registry entry to file."""
        if not self.registry_config.enable_persistence:
            return

        registry_entry = {
            **metadata.dict(),
            "registry_version": "1.0",
            "created_by": "ModelRegistryService",
        }

        registry_file = self._registry_path / f"{metadata.model_id}.json"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._save_json_file,
            registry_entry,
            registry_file,
        )

    def _save_json_file(self, data: dict[str, Any], file_path: Path) -> None:
        """Save data as JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    async def _load_existing_metadata(self) -> None:
        """Load existing model metadata from files."""
        if not self._registry_path.exists():
            return

        try:
            json_files = list(self._registry_path.glob("*.json"))

            for file_path in json_files:
                try:
                    loop = asyncio.get_event_loop()
                    registry_data = await loop.run_in_executor(
                        self._executor, self._load_json_file, file_path
                    )

                    # Create metadata object
                    metadata = ModelMetadata(**registry_data)

                    # Cache metadata
                    self._model_metadata_cache[metadata.model_id] = (
                        metadata,
                        datetime.now(timezone.utc),
                    )

                except Exception as e:
                    self._logger.warning(f"Failed to load registry entry from {file_path}: {e}")

        except Exception as e:
            self._logger.error(f"Failed to load existing metadata: {e}")

    async def _cache_model(self, model_id: str, metadata: ModelMetadata, model: Any) -> None:
        """Cache model and metadata."""
        self._model_cache[model_id] = (metadata, model, datetime.now(timezone.utc))
        self._model_metadata_cache[model_id] = (metadata, datetime.now(timezone.utc))

    async def _get_cached_model(self, model_id: str) -> Any | None:
        """Get cached model."""
        if model_id in self._model_cache:
            metadata, model, timestamp = self._model_cache[model_id]
            ttl_hours = self.registry_config.cache_ttl_hours

            if (datetime.now(timezone.utc) - timestamp).total_seconds() < ttl_hours * 3600:
                return model
            else:
                # Remove expired entry
                del self._model_cache[model_id]

        return None

    async def _log_audit_event(
        self, event_type: str, model_id: str, details: dict[str, Any]
    ) -> None:
        """Log audit event."""
        if not self.registry_config.enable_audit_trail:
            return

        audit_entry = {
            "event_type": event_type,
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "service": "ModelRegistryService",
        }

        # Store audit entry through repository
        try:
            await self.ml_repository.store_audit_entry("model_registry", audit_entry)
        except Exception as e:
            self._logger.warning(f"Failed to log audit event: {e}")

    # Background Tasks
    async def _background_cleanup(self) -> None:
        """Background task for cleanup and maintenance."""
        while True:
            try:
                await asyncio.sleep(self.registry_config.background_cleanup_interval)

                # Clean expired cache entries
                await self._clean_expired_cache()

                # Clean old model versions if enabled
                if self.registry_config.max_versions_per_model > 0:
                    await self._cleanup_old_versions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Background cleanup error: {e}")

    async def _clean_expired_cache(self) -> None:
        """Clean expired cache entries."""
        ttl_hours = self.registry_config.cache_ttl_hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)

        # Clean model cache
        expired_model_keys = [
            key for key, (_, _, timestamp) in self._model_cache.items() if timestamp < cutoff_time
        ]

        for key in expired_model_keys:
            del self._model_cache[key]

        # Clean metadata cache
        expired_metadata_keys = [
            key
            for key, (_, timestamp) in self._model_metadata_cache.items()
            if timestamp < cutoff_time
        ]

        for key in expired_metadata_keys:
            del self._model_metadata_cache[key]

        if expired_model_keys or expired_metadata_keys:
            self._logger.debug(
                f"Cleaned expired cache entries: {len(expired_model_keys)} models, "
                f"{len(expired_metadata_keys)} metadata"
            )

    async def _cleanup_old_versions(self) -> None:
        """Clean up old model versions."""
        try:
            # Get all models grouped by name and type
            all_models = await self.ml_repository.get_all_models()

            # Group by name and type
            model_groups: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
            for model in all_models:
                key = (model.get("name"), model.get("model_type"))
                if key not in model_groups:
                    model_groups[key] = []
                model_groups[key].append(model)

            # Clean up old versions for each group
            for (_name, _model_type), models in model_groups.items():
                if len(models) > self.registry_config.max_versions_per_model:
                    # Sort by creation date
                    models.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

                    # Keep only the most recent versions
                    models_to_remove = models[self.registry_config.max_versions_per_model :]

                    for model in models_to_remove:
                        try:
                            await self.delete_model(model["model_id"], remove_files=True)
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to clean up old model version {model['model_id']}: {e}"
                            )

        except Exception as e:
            self._logger.error(f"Old version cleanup failed: {e}")

    # Service Health and Metrics
    async def _service_health_check(self) -> Any:
        """Model registry service specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check dependencies
            if not self.data_service:
                return HealthStatus.UNHEALTHY

            # Check storage paths if persistence is enabled
            if self.registry_config.enable_persistence:
                if not self._registry_path.exists() or not self._artifact_path.exists():
                    return HealthStatus.DEGRADED

            # Check cache sizes
            if len(self._model_cache) > 1000 or len(self._model_metadata_cache) > 10000:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Model registry service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_model_registry_metrics(self) -> dict[str, Any]:
        """Get model registry service metrics."""
        return {
            "cached_models": len(self._model_cache),
            "cached_metadata": len(self._model_metadata_cache),
            "persistence_enabled": self.registry_config.enable_persistence,
            "versioning_enabled": self.registry_config.enable_versioning,
            "audit_trail_enabled": self.registry_config.enable_audit_trail,
            "max_versions_per_model": self.registry_config.max_versions_per_model,
        }

    async def clear_cache(self) -> dict[str, int]:
        """Clear model registry caches."""
        model_cache_size = len(self._model_cache)
        metadata_cache_size = len(self._model_metadata_cache)

        self._model_cache.clear()
        self._model_metadata_cache.clear()

        self._logger.info(
            "Model registry caches cleared",
            models_removed=model_cache_size,
            metadata_removed=metadata_cache_size,
        )

        return {
            "models_removed": model_cache_size,
            "metadata_removed": metadata_cache_size,
        }

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate model registry service configuration."""
        try:
            registry_config_dict = config.get("model_registry", {})
            ModelRegistryConfig(**registry_config_dict)
            return True
        except Exception as e:
            self._logger.error(
                "Model registry service configuration validation failed", error=str(e)
            )
            return False
