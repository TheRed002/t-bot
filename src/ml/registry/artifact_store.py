"""
Artifact Store for ML Model Artifact Management.

This module provides comprehensive artifact management for ML models including
storage, versioning, compression, and metadata tracking with database integration.
"""

import gzip
import hashlib
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.exceptions import ModelError, ValidationError
from src.core.logging import get_logger
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class ArtifactStore:
    """
    Artifact store for managing ML model artifacts.

    This class provides comprehensive artifact management including storage,
    compression, versioning, and metadata tracking for ML models and associated
    files like training data, feature transformers, and evaluation reports.

    Attributes:
        config: Application configuration
        base_path: Base path for artifact storage
        compression_enabled: Whether to enable artifact compression
        max_artifact_size_mb: Maximum artifact size in MB
    """

    def __init__(self, config: Config):
        """
        Initialize the artifact store.

        Args:
            config: Application configuration
        """
        self.config = config
        self.base_path = Path(config.ml.artifact_store_path)
        self.compression_enabled = True  # Enable compression by default
        self.max_artifact_size_mb = 500  # 500MB limit

        # Create necessary directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
        (self.base_path / "data").mkdir(exist_ok=True)
        (self.base_path / "reports").mkdir(exist_ok=True)
        (self.base_path / "transformers").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)

        logger.info(
            "Artifact store initialized",
            base_path=str(self.base_path),
            compression_enabled=self.compression_enabled,
            max_size_mb=self.max_artifact_size_mb,
        )

    @time_execution
    @log_calls
    def store_artifact(
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
        try:
            # Validate inputs
            if artifact_type not in ["model", "data", "report", "transformer"]:
                raise ValidationError(f"Invalid artifact type: {artifact_type}")

            # Create artifact directory structure
            artifact_dir = self.base_path / artifact_type / model_id / version
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Generate artifact filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{artifact_name}_{timestamp}"

            # Determine file extension based on data type
            if isinstance(artifact_data, pd.DataFrame | pd.Series):
                file_extension = ".parquet"
                artifact_path = artifact_dir / f"{base_filename}{file_extension}"
                artifact_data.to_parquet(artifact_path)
            elif isinstance(artifact_data, np.ndarray):
                file_extension = ".npy"
                artifact_path = artifact_dir / f"{base_filename}{file_extension}"
                np.save(artifact_path, artifact_data)
            elif isinstance(artifact_data, dict):
                file_extension = ".json"
                artifact_path = artifact_dir / f"{base_filename}{file_extension}"
                with open(artifact_path, "w") as f:
                    json.dump(artifact_data, f, indent=2, default=str)
            elif isinstance(artifact_data, str):
                file_extension = ".txt"
                artifact_path = artifact_dir / f"{base_filename}{file_extension}"
                with open(artifact_path, "w") as f:
                    f.write(artifact_data)
            else:
                # For other objects, use pickle-like storage
                import joblib

                file_extension = ".joblib"
                artifact_path = artifact_dir / f"{base_filename}{file_extension}"
                joblib.dump(artifact_data, artifact_path)

            # Check file size
            file_size_mb = artifact_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_artifact_size_mb:
                artifact_path.unlink()  # Remove the file
                raise ValidationError(
                    f"Artifact size ({file_size_mb:.2f}MB) exceeds maximum "
                    f"allowed size ({self.max_artifact_size_mb}MB)"
                )

            # Compress if requested and enabled
            final_path = artifact_path
            if compress and self.compression_enabled:
                compressed_path = artifact_path.with_suffix(f"{artifact_path.suffix}.gz")
                with open(artifact_path, "rb") as f_in:
                    with gzip.open(compressed_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove original file and use compressed version
                artifact_path.unlink()
                final_path = compressed_path

            # Calculate file hash for integrity checking
            file_hash = self._calculate_file_hash(final_path)

            # Create metadata
            artifact_metadata = {
                "artifact_name": artifact_name,
                "artifact_type": artifact_type,
                "model_id": model_id,
                "version": version,
                "file_path": str(final_path),
                "file_size_bytes": final_path.stat().st_size,
                "file_hash": file_hash,
                "compressed": compress and self.compression_enabled,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }

            # Store metadata
            metadata_path = artifact_dir / f"{base_filename}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(artifact_metadata, f, indent=2)

            logger.info(
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
            logger.error(
                "Artifact storage failed",
                artifact_type=artifact_type,
                artifact_name=artifact_name,
                model_id=model_id,
                error=str(e),
            )
            raise ModelError(f"Failed to store artifact {artifact_name}: {e}") from e

    @time_execution
    @log_calls
    def retrieve_artifact(
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
        try:
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
            metadata_file = None
            for mf in metadata_files:
                if mf.stem.replace("_metadata", "") == data_file.stem:
                    metadata_file = mf
                    break

            if metadata_file:
                with open(metadata_file) as f:
                    metadata = json.load(f)

            # Verify file integrity if hash is available
            if metadata.get("file_hash"):
                current_hash = self._calculate_file_hash(data_file)
                if current_hash != metadata["file_hash"]:
                    logger.warning(
                        "Artifact file hash mismatch",
                        artifact_name=artifact_name,
                        expected_hash=metadata["file_hash"],
                        actual_hash=current_hash,
                    )

            # Load the artifact data
            artifact_data = self._load_artifact_file(data_file, metadata.get("compressed", False))

            logger.info(
                "Artifact retrieved successfully",
                artifact_name=artifact_name,
                model_id=model_id,
                artifact_type=artifact_type,
                version=version_dir.name,
                file_path=str(data_file),
            )

            return artifact_data

        except Exception as e:
            logger.error(
                "Artifact retrieval failed",
                artifact_name=artifact_name,
                model_id=model_id,
                error=str(e),
            )
            raise ModelError(f"Failed to retrieve artifact {artifact_name}: {e}") from e

    @time_execution
    @log_calls
    def list_artifacts(
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
        try:
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

            for type_dir in type_dirs:
                if not type_dir.exists():
                    continue

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
                        version_dirs = (
                            [model_dir / version] if (model_dir / version).exists() else []
                        )
                    else:
                        version_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

                    for version_dir in version_dirs:
                        current_version = version_dir.name

                        # Get metadata files
                        metadata_files = list(version_dir.glob("*_metadata.json"))

                        for metadata_file in metadata_files:
                            try:
                                with open(metadata_file) as f:
                                    metadata = json.load(f)

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
                                logger.warning(
                                    "Failed to read metadata file",
                                    metadata_file=str(metadata_file),
                                    error=str(e),
                                )

            df = pd.DataFrame(artifacts)

            logger.info(
                "Artifacts listed successfully",
                total_artifacts=len(df),
                model_id=model_id,
                artifact_type=artifact_type,
            )

            return df

        except Exception as e:
            logger.error("Failed to list artifacts", error=str(e))
            raise ModelError(f"Failed to list artifacts: {e}") from e

    @time_execution
    @log_calls
    def delete_artifact(
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
        try:
            deleted_count = 0
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
                    artifact_file.unlink()
                    deleted_count += 1

                # Clean up empty directories
                if not any(version_dir.iterdir()):
                    version_dir.rmdir()

            # Clean up empty model directory
            if not any(type_dir.iterdir()):
                type_dir.rmdir()

            logger.info(
                "Artifact deleted successfully",
                artifact_name=artifact_name,
                model_id=model_id,
                artifact_type=artifact_type,
                version=version,
                deleted_files=deleted_count,
            )

            return deleted_count > 0

        except Exception as e:
            logger.error(
                "Artifact deletion failed",
                artifact_name=artifact_name,
                model_id=model_id,
                error=str(e),
            )
            raise ModelError(f"Failed to delete artifact {artifact_name}: {e}") from e

    @time_execution
    @log_calls
    def create_bundle(self, model_id: str, version: str, bundle_name: str | None = None) -> str:
        """
        Create a bundle containing all artifacts for a model version.

        Args:
            model_id: Model ID to bundle
            version: Model version to bundle
            bundle_name: Optional bundle name

        Returns:
            Path to the created bundle

        Raises:
            ModelError: If bundle creation fails
        """
        try:
            if not bundle_name:
                bundle_name = f"{model_id}_{version}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            bundle_path = self.base_path / "bundles" / f"{bundle_name}.tar.gz"
            bundle_path.parent.mkdir(parents=True, exist_ok=True)

            # Create tar archive
            with tarfile.open(bundle_path, "w:gz") as tar:
                # Add all artifacts for this model version
                for artifact_type in ["models", "data", "reports", "transformers"]:
                    type_dir = self.base_path / artifact_type / model_id / version
                    if type_dir.exists():
                        tar.add(type_dir, arcname=f"{artifact_type}/{version}")

            bundle_size_mb = bundle_path.stat().st_size / (1024 * 1024)

            # Create bundle metadata
            bundle_metadata = {
                "bundle_name": bundle_name,
                "model_id": model_id,
                "version": version,
                "bundle_path": str(bundle_path),
                "bundle_size_mb": round(bundle_size_mb, 2),
                "created_at": datetime.utcnow().isoformat(),
            }

            metadata_path = bundle_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(bundle_metadata, f, indent=2)

            logger.info(
                "Bundle created successfully",
                bundle_name=bundle_name,
                model_id=model_id,
                version=version,
                bundle_size_mb=round(bundle_size_mb, 2),
            )

            return str(bundle_path)

        except Exception as e:
            logger.error("Bundle creation failed", model_id=model_id, version=version, error=str(e))
            raise ModelError(f"Failed to create bundle for {model_id}: {e}") from e

    def cleanup_old_artifacts(self, days_to_keep: int = 30) -> int:
        """
        Clean up old artifacts beyond the retention period.

        Args:
            days_to_keep: Number of days to keep artifacts

        Returns:
            Number of artifacts cleaned up
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            cleanup_count = 0

            # Iterate through all artifact directories
            for artifact_type in ["models", "data", "reports", "transformers"]:
                type_dir = self.base_path / artifact_type
                if not type_dir.exists():
                    continue

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
                            shutil.rmtree(version_dir)
                            cleanup_count += 1

            logger.info(
                "Artifact cleanup completed", cleanup_count=cleanup_count, days_to_keep=days_to_keep
            )

            return cleanup_count

        except Exception as e:
            logger.error("Artifact cleanup failed", error=str(e))
            return 0

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _load_artifact_file(self, file_path: Path, compressed: bool = False) -> Any:
        """Load artifact data from file."""
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
            return np.load(file_path)
        elif suffix == ".json":
            with open(file_path) as f:
                return json.load(f)
        elif suffix == ".txt":
            with open(file_path) as f:
                return f.read()
        elif suffix == ".joblib":
            import joblib

            return joblib.load(file_path)
        else:
            # Default to binary read
            with open(file_path, "rb") as f:
                return f.read()
