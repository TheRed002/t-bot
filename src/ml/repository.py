"""
ML Repository Layer.

This module provides repository implementations for ML data persistence
following the repository pattern to separate data access from business logic.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.core.base.repository import BaseRepository
from src.core.exceptions import DatabaseError, ServiceError
from src.core.types.base import ConfigDict
from src.utils.constants import ML_MODEL_CONSTANTS


class IMLRepository(ABC):
    """Interface for ML data repository."""

    @abstractmethod
    async def store_model_metadata(self, metadata: dict[str, Any]) -> str:
        """Store model metadata."""
        pass

    @abstractmethod
    async def get_model_by_id(self, model_id: str) -> dict[str, Any] | None:
        """Get model metadata by ID."""
        pass

    @abstractmethod
    async def get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]:
        """Get models by name and type."""
        pass

    @abstractmethod
    async def find_models(
        self,
        name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Find models by criteria."""
        pass

    @abstractmethod
    async def get_all_models(
        self,
        model_type: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Get all models matching criteria."""
        pass

    @abstractmethod
    async def update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> bool:
        """Update model metadata."""
        pass

    @abstractmethod
    async def delete_model(self, model_id: str) -> bool:
        """Delete model metadata."""
        pass

    @abstractmethod
    async def store_prediction(self, prediction_data: dict[str, Any]) -> str:
        """Store ML prediction."""
        pass

    @abstractmethod
    async def get_predictions(
        self,
        model_id: str | None = None,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = ML_MODEL_CONSTANTS["default_query_limit"],
    ) -> list[dict[str, Any]]:
        """Get ML predictions with filters."""
        pass

    @abstractmethod
    async def store_training_job(self, job_data: dict[str, Any]) -> str:
        """Store training job information."""
        pass

    @abstractmethod
    async def get_training_job(self, job_id: str) -> dict[str, Any] | None:
        """Get training job by ID."""
        pass

    @abstractmethod
    async def update_training_progress(self, job_id: str, progress: dict[str, Any]) -> bool:
        """Update training job progress."""
        pass


class MLRepository(BaseRepository, IMLRepository):
    """ML repository implementation using actual database models."""

    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None):
        super().__init__(
            entity_type=dict,  # Use generic dict since we use data service
            key_type=str,
            name="MLRepository",
            config=config,
            correlation_id=correlation_id,
        )
        # Service dependencies - resolved during startup
        self.data_service: Any = None

        # Add required dependencies
        self.add_dependency("DataServiceInterface")

        # In-memory storage for abstract methods compatibility
        self._models: dict[str, dict[str, Any]] = {}
        self._predictions: dict[str, dict[str, Any]] = {}
        self._training_jobs: dict[str, dict[str, Any]] = {}
        self._audit_entries: list[dict[str, Any]] = []

    async def _do_start(self) -> None:
        """Start the repository and resolve dependencies."""
        await super()._do_start()

        # Resolve data service dependency
        try:
            self.data_service = self.resolve_dependency("DataServiceInterface")
        except Exception as e:
            self._logger.warning(f"Failed to resolve DataServiceInterface: {e}")
            self.data_service = None

        self._logger.info(
            "ML repository started successfully",
            data_service_available=bool(self.data_service),
        )

    async def store_model_metadata(self, metadata: dict[str, Any]) -> str:
        """Store model metadata through data service."""
        try:
            # Apply consistent boundary validation at repository boundary
            from src.utils.messaging_patterns import BoundaryValidator

            try:
                BoundaryValidator.validate_database_entity(metadata, "create")
            except ImportError:
                # Fallback validation if messaging patterns not available
                if not metadata.get("model_name"):
                    raise DatabaseError("model_name is required")

            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Store through data service
            stored_metadata = await self.data_service.store_ml_model_metadata(metadata)
            model_id = stored_metadata.get("id") or str(len(self._models))

            # Keep local copy for abstract methods compatibility
            self._models[model_id] = stored_metadata

            return model_id
        except Exception as e:
            # Apply consistent error propagation patterns
            from src.utils.messaging_patterns import ErrorPropagationMixin

            error_propagator = ErrorPropagationMixin()
            try:
                error_propagator.propagate_database_error(e, "ml_repository.store_model_metadata")
            except ImportError:
                # Fallback to existing pattern if propagation fails
                raise DatabaseError(f"Failed to store model metadata: {e}") from e

    async def get_model_by_id(self, model_id: str) -> dict[str, Any] | None:
        """Get model metadata by ID through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Get through data service
            model_metadata = await self.data_service.get_ml_model_by_id(model_id)

            if model_metadata:
                # Keep local copy for abstract methods compatibility
                self._models[model_id] = model_metadata

            return model_metadata
        except Exception as e:
            raise DatabaseError(f"Failed to get model by ID: {e}") from e

    async def get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]:
        """Get models by name and type through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Get through data service
            models = await self.data_service.get_ml_models_by_criteria(
                {"model_name": name, "model_type": model_type}
            )

            # Keep local copies for abstract methods compatibility
            for model in models:
                model_id = model.get("model_id") or model.get("id")
                if model_id:
                    self._models[str(model_id)] = model

            return models
        except Exception as e:
            raise DatabaseError(f"Failed to get models by name and type: {e}") from e

    async def find_models(
        self,
        name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Find models by criteria through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Build criteria for data service
            criteria: dict[str, Any] = {}
            if active_only:
                criteria["is_active"] = active_only
            if name:
                criteria["model_name"] = name
            if model_type:
                criteria["model_type"] = model_type
            if version:
                criteria["model_version"] = version
            if stage:
                criteria["stage"] = stage

            # Get through data service
            models = await self.data_service.get_ml_models_by_criteria(criteria)

            # Keep local copies for abstract methods compatibility
            for model in models:
                model_id = model.get("model_id") or model.get("id")
                if model_id:
                    self._models[str(model_id)] = model

            return models
        except Exception as e:
            raise DatabaseError(f"Failed to find models: {e}") from e

    async def get_all_models(
        self,
        model_type: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Get all models matching criteria through data service."""
        try:
            # Use find_models with appropriate criteria
            return await self.find_models(
                model_type=model_type, stage=stage, active_only=active_only
            )
        except Exception as e:
            raise DatabaseError(f"Failed to get all models: {e}") from e

    async def update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> bool:
        """Update model metadata through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Update through data service
            success = await self.data_service.update_ml_model_metadata(model_id, metadata)

            if success:
                # Update local copy if exists
                if model_id in self._models:
                    self._models[model_id].update(metadata)

            return success
        except Exception as e:
            raise DatabaseError(f"Failed to update model metadata: {e}") from e

    async def delete_model(self, model_id: str) -> bool:
        """Delete model metadata through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Delete through data service
            success = await self.data_service.delete_ml_model(model_id)

            if success:
                # Remove from local copy
                self._models.pop(model_id, None)

            return success
        except Exception as e:
            raise DatabaseError(f"Failed to delete model: {e}") from e

    async def store_prediction(self, prediction_data: dict[str, Any]) -> str:
        """Store ML prediction through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Store through data service
            stored_prediction = await self.data_service.store_ml_prediction(prediction_data)
            prediction_id = stored_prediction.get("id") or str(len(self._predictions))

            # Keep local copy for abstract methods compatibility
            self._predictions[prediction_id] = stored_prediction

            return prediction_id
        except Exception as e:
            raise DatabaseError(f"Failed to store prediction: {e}") from e

    async def get_predictions(
        self,
        model_id: str | None = None,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = ML_MODEL_CONSTANTS["default_query_limit"],
    ) -> list[dict[str, Any]]:
        """Get ML predictions with filters through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Build criteria for data service
            criteria: dict[str, Any] = {"limit": limit}
            if model_id:
                criteria["model_id"] = model_id
            if symbol:
                criteria["symbol"] = symbol
            if start_time:
                criteria["start_time"] = start_time
            if end_time:
                criteria["end_time"] = end_time

            # Get through data service
            predictions = await self.data_service.get_ml_predictions(criteria)

            # Keep local copies for abstract methods compatibility
            for prediction in predictions:
                prediction_id = prediction.get("id")
                if prediction_id:
                    self._predictions[str(prediction_id)] = prediction

            return predictions
        except Exception as e:
            raise DatabaseError(f"Failed to get predictions: {e}") from e

    async def store_training_job(self, job_data: dict[str, Any]) -> str:
        """Store training job information through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Store through data service
            stored_job = await self.data_service.store_ml_training_job(job_data)
            job_id = stored_job.get("id") or str(len(self._training_jobs))

            # Keep local copy for abstract methods compatibility
            self._training_jobs[job_id] = stored_job

            return job_id
        except Exception as e:
            raise DatabaseError(f"Failed to store training job: {e}") from e

    async def get_training_job(self, job_id: str) -> dict[str, Any] | None:
        """Get training job by ID through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Get through data service
            training_job = await self.data_service.get_ml_training_job(job_id)

            if training_job:
                # Keep local copy for abstract methods compatibility
                self._training_jobs[job_id] = training_job

            return training_job
        except Exception as e:
            raise DatabaseError(f"Failed to get training job: {e}") from e

    async def update_training_progress(self, job_id: str, progress: dict[str, Any]) -> bool:
        """Update training job progress through data service."""
        try:
            # Use data service instead of direct database access
            if not self.data_service:
                raise ServiceError("Data service not available")

            # Update through data service
            success = await self.data_service.update_ml_training_job(job_id, progress)

            if success:
                # Update local copy if exists
                if job_id in self._training_jobs:
                    self._training_jobs[job_id].update(progress)

            return success
        except Exception as e:
            raise DatabaseError(f"Failed to update training progress: {e}") from e

    async def store_audit_entry(self, category: str, entry: dict[str, Any]) -> bool:
        """Store audit trail entry."""
        audit_entry = {"category": category, "entry": entry}
        self._audit_entries.append(audit_entry)
        return True

    # BaseRepository abstract methods implementation
    async def _create_entity(self, entity: dict[str, Any]) -> str:
        """Create a new entity through data service."""
        return await self.store_model_metadata(entity)

    async def _get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        """Get entity by ID through data service."""
        return await self.get_model_by_id(entity_id)

    async def _update_entity(self, entity_id: str, entity: dict[str, Any]) -> bool:
        """Update entity through data service."""
        return await self.update_model_metadata(entity_id, entity)

    async def _delete_entity(self, entity_id: str) -> bool:
        """Delete entity through data service."""
        return await self.delete_model(entity_id)

    async def _list_entities(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """List entities with optional filters through data service."""
        if not filters:
            return await self.get_all_models(active_only=False)

        # Extract standard filter parameters
        name = filters.get("model_name")
        model_type = filters.get("model_type")
        version = filters.get("model_version")
        stage = filters.get("stage")
        active_only = filters.get("is_active", filters.get("active_only", True))

        return await self.find_models(
            name=name, model_type=model_type, version=version, stage=stage, active_only=active_only
        )

    async def _count_entities(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities with optional filters through data service."""
        entities = await self._list_entities(filters)
        return len(entities)
