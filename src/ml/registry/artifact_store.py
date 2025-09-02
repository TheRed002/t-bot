"""
Artifact Store for ML Model Artifact Management.

This module provides comprehensive artifact management for ML models including
storage, versioning, compression, and metadata tracking using proper service patterns.
"""

import asyncio
import gzip
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.base.service import BaseService
from src.core.exceptions import ModelError, ValidationError
from src.core.types.base import ConfigDict
from src.utils.decorators import UnifiedDecorator

# Initialize decorator instance
dec = UnifiedDecorator()


class ArtifactStoreConfig(BaseModel):
    """Configuration for artifact store service."""

    artifact_store_path: str = Field(
        default="./data/models/artifacts", description="Artifact storage path"
    )
    compression_enabled: bool = Field(default=True, description="Enable artifact compression")
    max_artifact_size_mb: int = Field(default=500, description="Maximum artifact size in MB")
    enable_persistence: bool = Field(default=True, description="Enable artifact persistence")
    cache_ttl_hours: int = Field(default=24, description="Artifact cache TTL in hours")
    cleanup_interval_hours: int = Field(
        default=168, description="Cleanup interval in hours (7 days)"
    )
    retention_days: int = Field(default=30, description="Artifact retention period in days")
    enable_audit_trail: bool = Field(default=True, description="Enable audit trail logging")
    computation_workers: int = Field(default=4, description="Number of computation workers")


class ArtifactStore(BaseService):
    """
    Artifact store service for managing ML model artifacts.

    This service provides comprehensive artifact management including storage,
    compression, versioning, and metadata tracking for ML models and associated
    files like training data, feature transformers, and evaluation reports.

    Uses proper service patterns without direct database access.
    """

    def __init__(
        self,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize the artifact store service.

        Args:
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="ArtifactStore",
            config=config,
            correlation_id=correlation_id,
        )

        # Parse artifact store configuration
        artifact_config_dict = (config or {}).get("artifact_store", {})
        self.artifact_config = ArtifactStoreConfig(**artifact_config_dict)

        # Service dependencies - resolved during startup
        self.data_service: Any = None

        # Thread pool for I/O operations
        self._executor = ThreadPoolExecutor(max_workers=self.artifact_config.computation_workers)

        # Background cleanup task
        self._cleanup_task: asyncio.Task | None = None

        # Initialize paths
        self.base_path = Path(self.artifact_config.artifact_store_path)

        # Add required dependencies
        self.add_dependency("DataService")

    async def _do_start(self) -> None:
        """Start the artifact store service."""
        await super()._do_start()

        # Resolve dependencies
        self.data_service = self.resolve_dependency("DataService")

        # Create necessary directories if persistence is enabled
        if self.artifact_config.enable_persistence:
            self.base_path.mkdir(parents=True, exist_ok=True)
            (self.base_path / "models").mkdir(exist_ok=True)
            (self.base_path / "data").mkdir(exist_ok=True)
            (self.base_path / "reports").mkdir(exist_ok=True)
            (self.base_path / "transformers").mkdir(exist_ok=True)
            (self.base_path / "metadata").mkdir(exist_ok=True)

        # Start background cleanup task if enabled
        if self.artifact_config.cleanup_interval_hours > 0:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())

        self._logger.info(
            "Artifact store service started successfully",
            config=self.artifact_config.dict(),
            base_path=str(self.base_path),
        )

    async def _do_stop(self) -> None:
        """Stop the artifact store service."""
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

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def store_artifact(
        self,
        artifact_data: Any,
        artifact_type: str,
        artifact_name: str,
        model_id: str,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
        compress: bool = True,
    ) -> str:
        """
        Store an artifact in the artifact store.

        Args:
            artifact_data: The artifact data to store
            artifact_type: Type of artifact (model, data, report, transformer)
            artifact_name: Name of the artifact
            model_id: Associated model ID
            version: Artifact version
            metadata: Additional metadata
            compress: Whether to compress the artifact

        Returns:
            Path to the stored artifact

        Raises:
            ModelError: If storage fails
            ValidationError: If artifact validation fails
        """
        return await self.execute_with_monitoring(
            "store_artifact",
            self._store_artifact_impl,
            artifact_data,
            artifact_type,
            artifact_name,
            model_id,
            version,
            metadata,
            compress,
        )

    async def _store_artifact_impl(
        self,
        artifact_data: Any,
        artifact_type: str,
        artifact_name: str,
        model_id: str,
        version: str,
        metadata: dict[str, Any] | None,
        compress: bool,
    ) -> str:
        """Internal artifact storage implementation."""
        try:
            # Validate inputs
            if artifact_type not in ["model", "data", "report", "transformer"]:
                raise ValidationError(f"Invalid artifact type: {artifact_type}")

            if not self.artifact_config.enable_persistence:
                raise ModelError("Artifact persistence is disabled")

            # Create artifact directory structure
            artifact_dir = self.base_path / artifact_type / model_id / version
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Generate artifact filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            base_filename = f"{artifact_name}_{timestamp}"

            # Save artifact to file in thread pool
            artifact_path = await self._save_artifact_data(
                artifact_data, artifact_dir, base_filename
            )

            # Check file size
            file_size_mb = artifact_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.artifact_config.max_artifact_size_mb:
                artifact_path.unlink()  # Remove the file
                raise ValidationError(
                    f"Artifact size ({file_size_mb:.2f}MB) exceeds maximum "
                    f"allowed size ({self.artifact_config.max_artifact_size_mb}MB)"
                )

            # Compress if requested and enabled
            final_path = artifact_path
            if compress and self.artifact_config.compression_enabled:
                final_path = await self._compress_artifact(artifact_path)

            # Calculate file hash for integrity checking
            file_hash = await self._calculate_file_hash(final_path)

            # Create metadata
            artifact_metadata = {
                "artifact_name": artifact_name,
                "artifact_type": artifact_type,
                "model_id": model_id,
                "version": version,
                "file_path": str(final_path),
                "file_size_bytes": final_path.stat().st_size,
                "file_hash": file_hash,
                "compressed": compress and self.artifact_config.compression_enabled,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            # Store metadata
            await self._save_artifact_metadata(artifact_dir, base_filename, artifact_metadata)

            # Store artifact info in data service
            if self.data_service:
                await self.data_service.store_artifact_info(artifact_metadata)

            # Log audit trail
            if self.artifact_config.enable_audit_trail:
                await self._log_audit_event("artifact_stored", artifact_metadata)

            self._logger.info(
                "Artifact stored successfully",
                artifact_type=artifact_type,
                artifact_name=artifact_name,
                model_id=model_id,
                version=version,
                file_path=str(final_path),
                file_size_mb=round(file_size_mb, 2),
            )

            return str(final_path)

        except Exception as e:
            self._logger.error(
                "Artifact storage failed",
                artifact_type=artifact_type,
                artifact_name=artifact_name,
                model_id=model_id,
                error=str(e),
            )
            raise ModelError(f"Failed to store artifact {artifact_name}: {e}") from e

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def retrieve_artifact(
        self,
        artifact_name: str,
        model_id: str,
        artifact_type: str = "model",
        version: str | None = None,
    ) -> Any:
        """
        Retrieve an artifact from the store.

        Args:
            artifact_name: Name of the artifact to retrieve
            model_id: Associated model ID
            artifact_type: Type of artifact
            version: Specific version (latest if not specified)

        Returns:
            The retrieved artifact data

        Raises:
            ModelError: If retrieval fails
        """
        return await self.execute_with_monitoring(
            "retrieve_artifact",
            self._retrieve_artifact_impl,
            artifact_name,
            model_id,
            artifact_type,
            version,
        )

    async def _retrieve_artifact_impl(
        self,
        artifact_name: str,
        model_id: str,
        artifact_type: str,
        version: str | None,
    ) -> Any:
        """Internal artifact retrieval implementation."""
        try:
            if not self.artifact_config.enable_persistence:
                # Try to get artifact info from data service
                if self.data_service:
                    artifact_info = await self.data_service.get_artifact_info(
                        artifact_name, model_id, artifact_type, version
                    )
                    if artifact_info:
                        raise ModelError("Artifact found in registry but persistence is disabled")
                raise ModelError("Artifact persistence is disabled")

            # Find artifact directory
            type_dir = self.base_path / artifact_type / model_id
            if not type_dir.exists():
                raise ModelError(f"No artifacts found for model {model_id}")

            # Get version directory
            if version:
                version_dir = type_dir / version
                if not version_dir.exists():
                    raise ModelError(f"Version {version} not found for model {model_id}")
            else:
                # Get latest version
                version_dirs = [d for d in type_dir.iterdir() if d.is_dir()]
                if not version_dirs:
                    raise ModelError(f"No versions found for model {model_id}")
                version_dir = max(version_dirs, key=lambda x: x.name)

            # Find artifact files matching the name
            artifact_files = list(version_dir.glob(f"{artifact_name}_*"))
            metadata_files = [f for f in artifact_files if f.name.endswith("_metadata.json")]
            data_files = [f for f in artifact_files if not f.name.endswith("_metadata.json")]

            if not data_files:
                raise ModelError(f"Artifact {artifact_name} not found")

            # Get the most recent artifact file
            data_file = max(data_files, key=lambda x: x.stat().st_mtime)

            # Load metadata if available
            metadata = {}
            metadata_file = await self._find_metadata_file(metadata_files, data_file)
            if metadata_file:
                metadata = await self._load_json_file(metadata_file)

            # Verify file integrity if hash is available
            if metadata.get("file_hash"):
                current_hash = await self._calculate_file_hash(data_file)
                if current_hash != metadata["file_hash"]:
                    self._logger.warning(
                        "Artifact file hash mismatch",
                        artifact_name=artifact_name,
                        expected_hash=metadata["file_hash"],
                        actual_hash=current_hash,
                    )

            # Load the artifact data
            artifact_data = await self._load_artifact_file(
                data_file, metadata.get("compressed", False)
            )

            # Log audit trail
            if self.artifact_config.enable_audit_trail:
                await self._log_audit_event(
                    "artifact_retrieved",
                    {
                        "artifact_name": artifact_name,
                        "model_id": model_id,
                        "artifact_type": artifact_type,
                        "version": version_dir.name,
                    },
                )

            self._logger.info(
                "Artifact retrieved successfully",
                artifact_name=artifact_name,
                model_id=model_id,
                artifact_type=artifact_type,
                version=version_dir.name,
                file_path=str(data_file),
            )

            return artifact_data

        except Exception as e:
            self._logger.error(
                "Artifact retrieval failed",
                artifact_name=artifact_name,
                model_id=model_id,
                error=str(e),
            )
            raise ModelError(f"Failed to retrieve artifact {artifact_name}: {e}") from e

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def list_artifacts(
        self,
        model_id: str | None = None,
        artifact_type: str | None = None,
        version: str | None = None,
    ) -> pd.DataFrame:
        """
        List all artifacts in the store.

        Args:
            model_id: Filter by model ID
            artifact_type: Filter by artifact type
            version: Filter by version

        Returns:
            DataFrame with artifact information
        """
        return await self.execute_with_monitoring(
            "list_artifacts",
            self._list_artifacts_impl,
            model_id,
            artifact_type,
            version,
        )

    async def _list_artifacts_impl(
        self,
        model_id: str | None,
        artifact_type: str | None,
        version: str | None,
    ) -> pd.DataFrame:
        """Internal artifact listing implementation."""
        try:
            if not self.artifact_config.enable_persistence:
                # Get from data service if available
                if self.data_service:
                    artifacts_data = await self.data_service.list_artifacts(
                        model_id=model_id,
                        artifact_type=artifact_type,
                        version=version,
                    )
                    return pd.DataFrame(artifacts_data)
                else:
                    return pd.DataFrame()

            artifacts = []

            # Determine search paths
            if artifact_type:
                type_dirs = [self.base_path / artifact_type]
            else:
                type_dirs = [
                    self.base_path / "models",
                    self.base_path / "data",
                    self.base_path / "reports",
                    self.base_path / "transformers",
                ]

            # Scan directories in thread pool
            for type_dir in type_dirs:
                if not type_dir.exists():
                    continue

                type_artifacts = await self._scan_artifact_directory(type_dir, model_id, version)
                artifacts.extend(type_artifacts)

            df = pd.DataFrame(artifacts)

            self._logger.info(
                "Artifacts listed successfully",
                total_artifacts=len(df),
                model_id=model_id,
                artifact_type=artifact_type,
            )

            return df

        except Exception as e:
            self._logger.error("Failed to list artifacts", error=str(e))
            raise ModelError(f"Failed to list artifacts: {e}") from e

    @dec.enhance(log=True, monitor=True, log_level="info")
    async def delete_artifact(
        self,
        artifact_name: str,
        model_id: str,
        artifact_type: str = "model",
        version: str | None = None,
    ) -> bool:
        """
        Delete an artifact from the store.

        Args:
            artifact_name: Name of the artifact to delete
            model_id: Associated model ID
            artifact_type: Type of artifact
            version: Specific version (all versions if not specified)

        Returns:
            True if deletion successful

        Raises:
            ModelError: If deletion fails
        """
        return await self.execute_with_monitoring(
            "delete_artifact",
            self._delete_artifact_impl,
            artifact_name,
            model_id,
            artifact_type,
            version,
        )

    async def _delete_artifact_impl(
        self,
        artifact_name: str,
        model_id: str,
        artifact_type: str,
        version: str | None,
    ) -> bool:
        """Internal artifact deletion implementation."""
        try:
            deleted_count = 0

            # Delete from data service
            if self.data_service:
                await self.data_service.delete_artifact_info(
                    artifact_name, model_id, artifact_type, version
                )

            # Delete files if persistence is enabled
            if self.artifact_config.enable_persistence:
                type_dir = self.base_path / artifact_type / model_id

                if not type_dir.exists():
                    raise ModelError(f"No artifacts found for model {model_id}")

                # Get version directories to search
                if version:
                    version_dirs = [type_dir / version] if (type_dir / version).exists() else []
                else:
                    version_dirs = [d for d in type_dir.iterdir() if d.is_dir()]

                for version_dir in version_dirs:
                    # Find and delete artifact files
                    artifact_files = list(version_dir.glob(f"{artifact_name}_*"))

                    for artifact_file in artifact_files:
                        await asyncio.get_event_loop().run_in_executor(
                            self._executor, artifact_file.unlink
                        )
                        deleted_count += 1

                    # Clean up empty directories
                    if not any(version_dir.iterdir()):
                        await asyncio.get_event_loop().run_in_executor(
                            self._executor, version_dir.rmdir
                        )

                # Clean up empty model directory
                if not any(type_dir.iterdir()):
                    await asyncio.get_event_loop().run_in_executor(self._executor, type_dir.rmdir)

            # Log audit trail
            if self.artifact_config.enable_audit_trail:
                await self._log_audit_event(
                    "artifact_deleted",
                    {
                        "artifact_name": artifact_name,
                        "model_id": model_id,
                        "artifact_type": artifact_type,
                        "version": version,
                        "deleted_files": deleted_count,
                    },
                )

            self._logger.info(
                "Artifact deleted successfully",
                artifact_name=artifact_name,
                model_id=model_id,
                artifact_type=artifact_type,
                version=version,
                deleted_files=deleted_count,
            )

            return deleted_count > 0 or self.data_service is not None

        except Exception as e:
            self._logger.error(
                "Artifact deletion failed",
                artifact_name=artifact_name,
                model_id=model_id,
                error=str(e),
            )
            raise ModelError(f"Failed to delete artifact {artifact_name}: {e}") from e

    async def cleanup_old_artifacts(self, days_to_keep: int | None = None) -> int:
        """
        Clean up old artifacts beyond the retention period.

        Args:
            days_to_keep: Number of days to keep artifacts (uses config default if not specified)

        Returns:
            Number of artifacts cleaned up
        """
        return await self.execute_with_monitoring(
            "cleanup_old_artifacts",
            self._cleanup_old_artifacts_impl,
            days_to_keep or self.artifact_config.retention_days,
        )

    async def _cleanup_old_artifacts_impl(self, days_to_keep: int) -> int:
        """Internal cleanup implementation."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            cleanup_count = 0

            if not self.artifact_config.enable_persistence:
                return 0

            # Iterate through all artifact directories
            for artifact_type in ["models", "data", "reports", "transformers"]:
                type_dir = self.base_path / artifact_type
                if not type_dir.exists():
                    continue

                cleanup_count += await self._cleanup_type_directory(type_dir, cutoff_date)

            # Log audit trail
            if self.artifact_config.enable_audit_trail:
                await self._log_audit_event(
                    "artifacts_cleaned",
                    {
                        "cleanup_count": cleanup_count,
                        "days_to_keep": days_to_keep,
                    },
                )

            self._logger.info(
                "Artifact cleanup completed",
                cleanup_count=cleanup_count,
                days_to_keep=days_to_keep,
            )

            return cleanup_count

        except Exception as e:
            self._logger.error("Artifact cleanup failed", error=str(e))
            return 0

    # Helper Methods
    async def _save_artifact_data(
        self, artifact_data: Any, artifact_dir: Path, base_filename: str
    ) -> Path:
        """Save artifact data to file."""
        loop = asyncio.get_event_loop()

        # Determine file extension and save method based on data type
        if isinstance(artifact_data, pd.DataFrame | pd.Series):
            file_extension = ".parquet"
            artifact_path = artifact_dir / f"{base_filename}{file_extension}"
            await loop.run_in_executor(
                self._executor, lambda: artifact_data.to_parquet(artifact_path)
            )
        elif isinstance(artifact_data, np.ndarray):
            file_extension = ".npy"
            artifact_path = artifact_dir / f"{base_filename}{file_extension}"

            def save_numpy():
                from src.core.exceptions import DataError

                try:
                    np.save(artifact_path, artifact_data)
                except Exception as e:
                    raise DataError(
                        f"Failed to save numpy array to {artifact_path}",
                        error_code="DATA_003",
                        data_type="numpy",
                        file_path=str(artifact_path),
                        original_error=str(e),
                    ) from e

            await loop.run_in_executor(self._executor, save_numpy)
        elif isinstance(artifact_data, dict):
            file_extension = ".json"
            artifact_path = artifact_dir / f"{base_filename}{file_extension}"
            await loop.run_in_executor(
                self._executor, self._save_json_data, artifact_data, artifact_path
            )
        elif isinstance(artifact_data, str):
            file_extension = ".txt"
            artifact_path = artifact_dir / f"{base_filename}{file_extension}"
            await loop.run_in_executor(
                self._executor, self._save_text_data, artifact_data, artifact_path
            )
        else:
            # For other objects, use pickle-like storage
            file_extension = ".joblib"
            artifact_path = artifact_dir / f"{base_filename}{file_extension}"
            await loop.run_in_executor(
                self._executor, self._save_joblib_data, artifact_data, artifact_path
            )

        return artifact_path

    def _save_json_data(self, data: dict, path: Path) -> None:
        """Save JSON data to file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_text_data(self, data: str, path: Path) -> None:
        """Save text data to file."""
        with open(path, "w") as f:
            f.write(data)

    def _save_joblib_data(self, data: Any, path: Path) -> None:
        """Save object using joblib."""
        import joblib

        from src.core.exceptions import DataError

        try:
            joblib.dump(data, path)
        except Exception as e:
            raise DataError(
                f"Failed to save joblib data to {path}",
                error_code="DATA_003",
                data_type="joblib",
                file_path=str(path),
                original_error=str(e),
            ) from e

    async def _compress_artifact(self, artifact_path: Path) -> Path:
        """Compress artifact file."""
        compressed_path = artifact_path.with_suffix(f"{artifact_path.suffix}.gz")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor, self._compress_file, artifact_path, compressed_path
        )

        # Remove original file
        artifact_path.unlink()
        return compressed_path

    def _compress_file(self, input_path: Path, output_path: Path) -> None:
        """Compress file using gzip."""
        with open(input_path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._calculate_hash_sync, file_path)

    def _calculate_hash_sync(self, file_path: Path) -> str:
        """Calculate SHA-256 hash synchronously."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
            return sha256_hash.hexdigest()

    async def _save_artifact_metadata(
        self, artifact_dir: Path, base_filename: str, metadata: dict[str, Any]
    ) -> None:
        """Save artifact metadata to file."""
        metadata_path = artifact_dir / f"{base_filename}_metadata.json"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._save_json_data, metadata, metadata_path)

    async def _load_json_file(self, file_path: Path) -> dict[str, Any]:
        """Load JSON file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._load_json_sync, file_path)

    def _load_json_sync(self, file_path: Path) -> dict[str, Any]:
        """Load JSON file synchronously."""
        with open(file_path) as f:
            return json.load(f)

    async def _find_metadata_file(self, metadata_files: list[Path], data_file: Path) -> Path | None:
        """Find metadata file corresponding to data file."""
        for mf in metadata_files:
            if mf.stem.replace("_metadata", "") == data_file.stem:
                return mf
        return None

    async def _load_artifact_file(self, file_path: Path, compressed: bool = False) -> Any:
        """Load artifact data from file."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._load_artifact_sync, file_path, compressed
        )

    def _load_artifact_sync(self, file_path: Path, compressed: bool) -> Any:
        """Load artifact data synchronously."""
        # Handle compressed files
        if compressed or file_path.suffix == ".gz":
            if file_path.suffix == ".gz":
                # Extract the original extension
                original_suffix = file_path.stem.split(".")[-1]
            else:
                original_suffix = file_path.suffix

            if compressed:
                with gzip.open(file_path, "rb") as f:
                    content = f.read()
                # Save to temporary file for processing
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=f".{original_suffix}", delete=False) as tmp:
                    tmp.write(content)
                    temp_path = Path(tmp.name)

                try:
                    result = self._load_by_extension(temp_path)
                finally:
                    temp_path.unlink()
                return result
            else:
                return self._load_by_extension(file_path)
        else:
            return self._load_by_extension(file_path)

    def _load_by_extension(self, file_path: Path) -> Any:
        """Load file based on its extension."""
        suffix = file_path.suffix.lower()

        if suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif suffix == ".npy":
            from src.core.exceptions import DataError

            try:
                return np.load(file_path)
            except Exception as e:
                raise DataError(
                    f"Failed to load numpy array from {file_path}",
                    error_code="DATA_002",
                    data_type="numpy",
                    file_path=str(file_path),
                    original_error=str(e),
                ) from e
        elif suffix == ".json":
            with open(file_path) as f:
                return json.load(f)
        elif suffix == ".txt":
            with open(file_path) as f:
                return f.read()
        elif suffix == ".joblib":
            import joblib

            from src.core.exceptions import DataError

            try:
                return joblib.load(file_path)
            except Exception as e:
                raise DataError(
                    f"Failed to load joblib data from {file_path}",
                    error_code="DATA_002",
                    data_type="joblib",
                    file_path=str(file_path),
                    original_error=str(e),
                ) from e
        else:
            # Default to binary read
            with open(file_path, "rb") as f:
                return f.read()

    async def _scan_artifact_directory(
        self, type_dir: Path, model_id: str | None, version: str | None
    ) -> list[dict[str, Any]]:
        """Scan artifact directory for metadata."""
        artifacts = []
        current_type = type_dir.name

        # Get model directories
        if model_id:
            model_dirs = [type_dir / model_id] if (type_dir / model_id).exists() else []
        else:
            model_dirs = [d for d in type_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            current_model_id = model_dir.name

            # Get version directories
            if version:
                version_dirs = [model_dir / version] if (model_dir / version).exists() else []
            else:
                version_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

            for version_dir in version_dirs:
                current_version = version_dir.name

                # Get metadata files
                metadata_files = list(version_dir.glob("*_metadata.json"))

                for metadata_file in metadata_files:
                    try:
                        metadata = await self._load_json_file(metadata_file)
                        artifacts.append(
                            {
                                "artifact_name": metadata.get("artifact_name", "unknown"),
                                "artifact_type": current_type,
                                "model_id": current_model_id,
                                "version": current_version,
                                "file_path": metadata.get("file_path", ""),
                                "file_size_bytes": metadata.get("file_size_bytes", 0),
                                "file_size_mb": round(
                                    metadata.get("file_size_bytes", 0) / (1024 * 1024), 2
                                ),
                                "compressed": metadata.get("compressed", False),
                                "created_at": metadata.get("created_at", ""),
                                "file_hash": metadata.get("file_hash", ""),
                            }
                        )
                    except Exception as e:
                        self._logger.warning(
                            "Failed to read metadata file",
                            metadata_file=str(metadata_file),
                            error=str(e),
                        )

        return artifacts

    async def _cleanup_type_directory(self, type_dir: Path, cutoff_date: datetime) -> int:
        """Clean up artifacts in a type directory."""
        cleanup_count = 0

        for model_dir in type_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                # Check if any file in the version directory is older than cutoff
                should_delete = False
                for file_path in version_dir.rglob("*"):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            should_delete = True
                            break

                if should_delete:
                    await asyncio.get_event_loop().run_in_executor(
                        self._executor, lambda vd=version_dir: shutil.rmtree(vd)
                    )
                    cleanup_count += 1

        return cleanup_count

    async def _log_audit_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Log audit event."""
        if not self.artifact_config.enable_audit_trail:
            return

        audit_entry = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "service": "ArtifactStore",
        }

        # Store audit entry through data service
        try:
            if self.data_service:
                await self.data_service.store_audit_entry("artifact_store", audit_entry)
        except Exception as e:
            self._logger.warning(f"Failed to log audit event: {e}")

    # Background Tasks
    async def _background_cleanup(self) -> None:
        """Background task for cleanup and maintenance."""
        while True:
            try:
                await asyncio.sleep(self.artifact_config.cleanup_interval_hours * 3600)

                # Clean old artifacts
                await self.cleanup_old_artifacts()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Background cleanup error: {e}")

    # Service Health and Metrics
    async def _service_health_check(self) -> Any:
        """Artifact store service specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check dependencies
            if self.artifact_config.enable_persistence and not self.data_service:
                return HealthStatus.DEGRADED

            # Check storage paths if persistence is enabled
            if self.artifact_config.enable_persistence:
                if not self.base_path.exists():
                    return HealthStatus.UNHEALTHY

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error("Artifact store service health check failed", error=str(e))
            return HealthStatus.UNHEALTHY

    def get_artifact_store_metrics(self) -> dict[str, Any]:
        """Get artifact store service metrics."""
        metrics = {
            "persistence_enabled": self.artifact_config.enable_persistence,
            "compression_enabled": self.artifact_config.compression_enabled,
            "max_artifact_size_mb": self.artifact_config.max_artifact_size_mb,
            "retention_days": self.artifact_config.retention_days,
        }

        if self.artifact_config.enable_persistence and self.base_path.exists():
            # Count artifacts in each category
            for artifact_type in ["models", "data", "reports", "transformers"]:
                type_dir = self.base_path / artifact_type
                if type_dir.exists():
                    count = len(list(type_dir.rglob("*_metadata.json")))
                    metrics[f"{artifact_type}_count"] = count

        return metrics

    # Configuration validation
    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Validate artifact store service configuration."""
        try:
            artifact_config_dict = config.get("artifact_store", {})
            ArtifactStoreConfig(**artifact_config_dict)
            return True
        except Exception as e:
            self._logger.error(
                "Artifact store service configuration validation failed", error=str(e)
            )
            return False
