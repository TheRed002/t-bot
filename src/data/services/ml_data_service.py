"""
ML Data Service - ML-specific data operations

This module provides ML-specific data operations including model metadata,
feature set storage, and prediction persistence.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.base.interfaces import HealthStatus
from src.core.base.service import BaseService
from src.core.exceptions import DataError
from src.core.types.base import ConfigDict
from src.utils.decorators import UnifiedDecorator


class MLDataService(BaseService):
    """
    ML-specific data service providing storage and retrieval for ML artifacts.

    This service extends the base DataService with ML-specific operations
    for model registry, feature store, and prediction storage.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """Initialize ML Data Service."""
        super().__init__(
            name="MLDataService",
            config=config,
            correlation_id=correlation_id,
        )

        # In-memory storage for MVP
        # In production, these would be persisted to database
        self._model_metadata: dict[str, dict[str, Any]] = {}
        self._feature_sets: dict[str, dict[str, Any]] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._predictions: list[dict[str, Any]] = []
        self._audit_log: list[dict[str, Any]] = []

    async def _do_start(self) -> None:
        """Start the ML data service."""
        await super()._do_start()
        self.logger.info("ML Data Service started successfully")

    async def _do_stop(self) -> None:
        """Stop the ML data service."""
        await super()._do_stop()

    # Model Registry Operations
    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def store_model_metadata(self, metadata: dict[str, Any]) -> None:
        """Store model metadata."""
        model_id = metadata.get("model_id")
        if not model_id:
            raise DataError("model_id is required")

        self._model_metadata[model_id] = {
            **metadata,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        self.logger.info(f"Stored model metadata for {model_id}")

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> None:
        """Update model metadata."""
        if model_id not in self._model_metadata:
            raise DataError(f"Model {model_id} not found")

        self._model_metadata[model_id].update(
            {
                **metadata,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.logger.info(f"Updated model metadata for {model_id}")

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def get_model_by_id(self, model_id: str) -> dict[str, Any] | None:
        """Get model by ID."""
        return self._model_metadata.get(model_id)

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def get_all_models(
        self,
        model_type: str | None = None,
        stage: str | None = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Get all models with optional filtering."""
        models = list(self._model_metadata.values())

        if model_type:
            models = [m for m in models if m.get("model_type") == model_type]

        if stage:
            models = [m for m in models if m.get("stage") == stage]

        if not include_archived:
            models = [m for m in models if not m.get("archived", False)]

        return models

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]:
        """Get models by name and type."""
        return [
            m
            for m in self._model_metadata.values()
            if m.get("name") == name and m.get("model_type") == model_type
        ]

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def find_models(
        self,
        name: str | None = None,
        model_type: str | None = None,
        tags: dict[str, str] | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find models by criteria."""
        models = list(self._model_metadata.values())

        if name:
            models = [m for m in models if m.get("name") == name]

        if model_type:
            models = [m for m in models if m.get("model_type") == model_type]

        if stage:
            models = [m for m in models if m.get("stage") == stage]

        if tags:
            models = [
                m for m in models if all(m.get("tags", {}).get(k) == v for k, v in tags.items())
            ]

        return models

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def delete_model(self, model_id: str) -> None:
        """Delete model metadata."""
        if model_id in self._model_metadata:
            del self._model_metadata[model_id]
            self.logger.info(f"Deleted model {model_id}")

    # Feature Store Operations
    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def store_feature_set(
        self,
        feature_set_id: str,
        symbol: str,
        feature_data: dict[str, Any],
        metadata: dict[str, Any],
        version: str | None = None,
    ) -> None:
        """Store feature set."""
        key = f"{symbol}:{feature_set_id}:{version or 'latest'}"
        self._feature_sets[key] = {
            "feature_set_id": feature_set_id,
            "symbol": symbol,
            "feature_data": feature_data,
            "metadata": metadata,
            "version": version or "latest",
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        self.logger.info(f"Stored feature set {key}")

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def get_feature_set(
        self,
        symbol: str,
        feature_set_id: str,
        version: str | None = None,
    ) -> dict[str, Any] | None:
        """Get feature set."""
        key = f"{symbol}:{feature_set_id}:{version or 'latest'}"
        return self._feature_sets.get(key)

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def update_feature_set_metadata(
        self,
        feature_set_id: str,
        metadata_updates: dict[str, Any],
    ) -> None:
        """Update feature set metadata."""
        for key, data in self._feature_sets.items():
            if data["feature_set_id"] == feature_set_id:
                data["metadata"].update(metadata_updates)
                data["updated_at"] = datetime.now(timezone.utc).isoformat()

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def list_feature_sets(
        self,
        symbol: str | None = None,
        include_expired: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List feature sets."""
        feature_sets = list(self._feature_sets.values())

        if symbol:
            feature_sets = [fs for fs in feature_sets if fs["symbol"] == symbol]

        if not include_expired:
            # Filter expired based on metadata TTL
            feature_sets = [
                fs for fs in feature_sets if not fs.get("metadata", {}).get("expired", False)
            ]

        if limit:
            feature_sets = feature_sets[:limit]

        return feature_sets

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def delete_feature_set(
        self,
        symbol: str,
        feature_set_id: str,
        version: str | None = None,
        delete_all_versions: bool = False,
    ) -> int:
        """Delete feature set."""
        deleted_count = 0

        if delete_all_versions:
            keys_to_delete = [
                k for k in self._feature_sets.keys() if k.startswith(f"{symbol}:{feature_set_id}:")
            ]
        else:
            keys_to_delete = [f"{symbol}:{feature_set_id}:{version or 'latest'}"]

        for key in keys_to_delete:
            if key in self._feature_sets:
                del self._feature_sets[key]
                deleted_count += 1

        self.logger.info(f"Deleted {deleted_count} feature sets")
        return deleted_count

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def get_feature_set_versions(
        self,
        symbol: str,
        feature_set_id: str,
    ) -> list[str]:
        """Get all versions of a feature set."""
        versions = []
        prefix = f"{symbol}:{feature_set_id}:"

        for key in self._feature_sets.keys():
            if key.startswith(prefix):
                version = key.split(":")[-1]
                versions.append(version)

        return sorted(versions)

    # Artifact Store Operations
    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def store_artifact_info(self, artifact_metadata: dict[str, Any]) -> None:
        """Store artifact information."""
        artifact_key = (
            f"{artifact_metadata.get('model_id')}:"
            f"{artifact_metadata.get('artifact_name')}:"
            f"{artifact_metadata.get('artifact_type', 'unknown')}:"
            f"{artifact_metadata.get('version', 'latest')}"
        )

        self._artifacts[artifact_key] = {
            **artifact_metadata,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        self.logger.info(f"Stored artifact info {artifact_key}")

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def get_artifact_info(
        self,
        artifact_name: str,
        model_id: str,
        artifact_type: str,
        version: str | None = None,
    ) -> dict[str, Any] | None:
        """Get artifact information."""
        artifact_key = f"{model_id}:{artifact_name}:{artifact_type}:{version or 'latest'}"
        return self._artifacts.get(artifact_key)

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def list_artifacts(
        self,
        model_id: str | None = None,
        artifact_type: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """List artifacts."""
        artifacts = list(self._artifacts.values())

        if model_id:
            artifacts = [a for a in artifacts if a.get("model_id") == model_id]

        if artifact_type:
            artifacts = [a for a in artifacts if a.get("artifact_type") == artifact_type]

        if tags:
            artifacts = [
                a for a in artifacts if all(a.get("tags", {}).get(k) == v for k, v in tags.items())
            ]

        return artifacts

    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def delete_artifact_info(
        self,
        artifact_name: str,
        model_id: str,
        artifact_type: str,
        version: str | None = None,
    ) -> None:
        """Delete artifact information."""
        artifact_key = f"{model_id}:{artifact_name}:{artifact_type}:{version or 'latest'}"
        if artifact_key in self._artifacts:
            del self._artifacts[artifact_key]
            self.logger.info(f"Deleted artifact {artifact_key}")

    # Prediction Storage Operations
    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def save_ml_predictions(self, prediction_data: dict[str, Any]) -> None:
        """Save ML predictions."""
        self._predictions.append(
            {
                **prediction_data,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.logger.info(f"Saved predictions for model {prediction_data.get('model_id')}")

    # Audit Operations
    @UnifiedDecorator.enhance(log=True, monitor=True)
    async def store_audit_entry(self, service: str, audit_entry: dict[str, Any]) -> None:
        """Store audit log entry."""
        self._audit_log.append(
            {
                "service": service,
                **audit_entry,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    # Service Health
    async def _service_health_check(self) -> HealthStatus:
        """ML data service specific health check."""

        try:
            # Check storage capacity
            total_items = (
                len(self._model_metadata)
                + len(self._feature_sets)
                + len(self._artifacts)
                + len(self._predictions)
            )

            # Warn if too many items in memory
            if total_items > 10000:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self.logger.error("ML data service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_ml_data_metrics(self) -> dict[str, Any]:
        """Get ML data service metrics."""
        return {
            "models_stored": len(self._model_metadata),
            "feature_sets_stored": len(self._feature_sets),
            "artifacts_stored": len(self._artifacts),
            "predictions_stored": len(self._predictions),
            "audit_entries": len(self._audit_log),
        }
