"""
ML Repository Layer.

This module provides repository implementations for ML data persistence
following the repository pattern to separate data access from business logic.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.core.base.repository import BaseRepository
from src.core.types.base import ConfigDict


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
    async def store_audit_entry(self, category: str, entry: dict[str, Any]) -> bool:
        """Store audit trail entry."""
        pass


class MLRepository(BaseRepository, IMLRepository):
    """ML repository implementation."""

    def __init__(self, config: ConfigDict | None = None):
        super().__init__(
            entity_type=dict,  # ML models are stored as dict
            key_type=str,  # Model IDs are strings
            name="MLRepository",
            config=config,
        )

        # In-memory storage for demonstration
        # In production, this would connect to a real database
        self._models: dict[str, dict[str, Any]] = {}
        self._audit_entries: list[dict[str, Any]] = []

    async def store_model_metadata(self, metadata: dict[str, Any]) -> str:
        """Store model metadata."""
        model_id = metadata.get("model_id")
        if not model_id:
            raise ValueError("Model ID is required")

        self._models[model_id] = metadata
        return model_id

    async def get_model_by_id(self, model_id: str) -> dict[str, Any] | None:
        """Get model metadata by ID."""
        return self._models.get(model_id)

    async def get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]:
        """Get models by name and type."""
        return [
            model
            for model in self._models.values()
            if model.get("name") == name and model.get("model_type") == model_type
        ]

    async def find_models(
        self,
        name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Find models by criteria."""
        results = []

        for model in self._models.values():
            if active_only and not model.get("active", True):
                continue

            if name and model.get("name") != name:
                continue

            if model_type and model.get("model_type") != model_type:
                continue

            if version and model.get("version") != version:
                continue

            if stage and model.get("stage") != stage:
                continue

            results.append(model)

        return results

    async def get_all_models(
        self,
        model_type: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Get all models matching criteria."""
        results = []

        for model in self._models.values():
            if active_only and not model.get("active", True):
                continue

            if model_type and model.get("model_type") != model_type:
                continue

            if stage and model.get("stage") != stage:
                continue

            results.append(model)

        return results

    async def update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> bool:
        """Update model metadata."""
        if model_id in self._models:
            self._models[model_id].update(metadata)
            return True
        return False

    async def delete_model(self, model_id: str) -> bool:
        """Delete model metadata."""
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False

    async def store_audit_entry(self, category: str, entry: dict[str, Any]) -> bool:
        """Store audit trail entry."""
        audit_entry = {"category": category, "entry": entry}
        self._audit_entries.append(audit_entry)
        return True

    # BaseRepository abstract methods implementation
    async def _create_entity(self, entity: dict[str, Any]) -> str:
        """Create a new entity."""
        entity_id = entity.get("model_id") or entity.get("id", str(len(self._models)))
        self._models[entity_id] = entity
        return entity_id

    async def _get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        """Get entity by ID."""
        return self._models.get(entity_id)

    async def _update_entity(self, entity_id: str, entity: dict[str, Any]) -> bool:
        """Update entity."""
        if entity_id in self._models:
            self._models[entity_id].update(entity)
            return True
        return False

    async def _delete_entity(self, entity_id: str) -> bool:
        """Delete entity."""
        if entity_id in self._models:
            del self._models[entity_id]
            return True
        return False

    async def _list_entities(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """List entities with optional filters."""
        if not filters:
            return list(self._models.values())

        results = []
        for entity in self._models.values():
            match = True
            for key, value in filters.items():
                if entity.get(key) != value:
                    match = False
                    break
            if match:
                results.append(entity)
        return results

    async def _count_entities(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities with optional filters."""
        if not filters:
            return len(self._models)

        count = 0
        for entity in self._models.values():
            match = True
            for key, value in filters.items():
                if entity.get(key) != value:
                    match = False
                    break
            if match:
                count += 1
        return count
