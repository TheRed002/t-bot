"""
Model storage abstraction layer.

This module provides an abstraction for model persistence to avoid
direct coupling to specific serialization libraries.
"""

import abc
from pathlib import Path
from typing import Any

import joblib


class ModelStorageBackend(abc.ABC):
    """Abstract base class for model storage backends."""

    @abc.abstractmethod
    def save(self, model_data: dict[str, Any], filepath: Path) -> None:
        """Save model data to storage."""
        pass

    @abc.abstractmethod
    def load(self, filepath: Path) -> dict[str, Any]:
        """Load model data from storage."""
        pass


class JoblibStorageBackend(ModelStorageBackend):
    """Joblib-based storage backend for sklearn models."""

    def save(self, model_data: dict[str, Any], filepath: Path) -> None:
        """Save model data using joblib."""
        from src.core.exceptions import ModelError

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_data, filepath)
        except Exception as e:
            raise ModelError(
                f"Failed to save model to {filepath}",
                error_code="MODEL_001",
                model_path=str(filepath),
                original_error=str(e),
            ) from e

    def load(self, filepath: Path) -> dict[str, Any]:
        """Load model data using joblib."""
        from src.core.exceptions import ModelLoadError

        if not filepath.exists():
            raise ModelLoadError(f"Model file not found: {filepath}", model_path=str(filepath))

        try:
            return joblib.load(filepath)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model from {filepath}",
                model_path=str(filepath),
                original_error=str(e),
            ) from e


class PickleStorageBackend(ModelStorageBackend):
    """Pickle-based storage backend for general Python objects."""

    def save(self, model_data: dict[str, Any], filepath: Path) -> None:
        """Save model data using pickle."""
        import pickle

        from src.core.exceptions import ModelError

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
        except Exception as e:
            raise ModelError(
                f"Failed to save model to {filepath}",
                error_code="MODEL_001",
                model_path=str(filepath),
                original_error=str(e),
            ) from e

    def load(self, filepath: Path) -> dict[str, Any]:
        """Load model data using pickle."""
        import pickle

        from src.core.exceptions import ModelLoadError

        if not filepath.exists():
            raise ModelLoadError(f"Model file not found: {filepath}", model_path=str(filepath))

        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model from {filepath}",
                model_path=str(filepath),
                original_error=str(e),
            ) from e


class ModelStorageManager:
    """Manager for model storage operations."""

    def __init__(self, backend: str = "joblib"):
        """
        Initialize storage manager.

        Args:
            backend: Storage backend to use ('joblib' or 'pickle')
        """
        self._backends = {
            "joblib": JoblibStorageBackend(),
            "pickle": PickleStorageBackend(),
        }

        if backend not in self._backends:
            raise ValueError(
                f"Unknown backend: {backend}. Available: {list(self._backends.keys())}"
            )

        self.backend = self._backends[backend]

    def save_model(self, model_data: dict[str, Any], filepath: str | Path) -> Path:
        """
        Save model data.

        Args:
            model_data: Model data to save
            filepath: Path to save to

        Returns:
            Path where model was saved
        """
        filepath = Path(filepath)
        self.backend.save(model_data, filepath)
        return filepath

    def load_model(self, filepath: str | Path) -> dict[str, Any]:
        """
        Load model data.

        Args:
            filepath: Path to load from

        Returns:
            Loaded model data
        """
        filepath = Path(filepath)
        return self.backend.load(filepath)
