"""
Unit tests for ML model storage functionality.
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.core.exceptions import ModelError, ModelLoadError
from src.ml.models.storage import (
    JoblibStorageBackend,
    ModelStorageBackend,
    ModelStorageManager,
    PickleStorageBackend,
)


class TestModelStorageBackend:
    """Test abstract base class for model storage backends."""

    def test_abstract_methods(self):
        """Test that abstract methods cannot be instantiated."""
        with pytest.raises(TypeError):
            ModelStorageBackend()

    def test_subclass_implementation(self):
        """Test that subclasses must implement abstract methods."""

        class IncompleteBackend(ModelStorageBackend):
            pass

        with pytest.raises(TypeError):
            IncompleteBackend()


class TestJoblibStorageBackend:
    """Test joblib storage backend."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = JoblibStorageBackend()
        assert isinstance(backend, ModelStorageBackend)

    @patch("joblib.dump")
    @patch("pathlib.Path.mkdir")
    def test_save_success(self, mock_mkdir, mock_dump):
        """Test successful model saving."""
        backend = JoblibStorageBackend()
        model_data = {"model": "test_model", "metadata": {"version": "1.0"}}
        filepath = Path("/tmp/test_model.joblib")

        backend.save(model_data, filepath)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_dump.assert_called_once_with(model_data, filepath)

    @patch("joblib.dump")
    @patch("pathlib.Path.mkdir")
    def test_save_mkdir_exception(self, mock_mkdir, mock_dump):
        """Test save when mkdir fails."""
        backend = JoblibStorageBackend()
        model_data = {"model": "test_model"}
        filepath = Path("/tmp/test_model.joblib")

        mock_mkdir.side_effect = OSError("Permission denied")

        with pytest.raises(ModelError) as exc_info:
            backend.save(model_data, filepath)

        assert "Failed to save model to" in str(exc_info.value)
        assert exc_info.value.error_code == "MODEL_001"

    @patch("joblib.dump")
    @patch("pathlib.Path.mkdir")
    def test_save_dump_exception(self, mock_mkdir, mock_dump):
        """Test save when joblib.dump fails."""
        backend = JoblibStorageBackend()
        model_data = {"model": "test_model"}
        filepath = Path("/tmp/test_model.joblib")

        mock_dump.side_effect = RuntimeError("Serialization failed")

        with pytest.raises(ModelError) as exc_info:
            backend.save(model_data, filepath)

        assert "Failed to save model to" in str(exc_info.value)
        assert exc_info.value.error_code == "MODEL_001"

    @patch("joblib.load")
    @patch("pathlib.Path.exists")
    def test_load_success(self, mock_exists, mock_load):
        """Test successful model loading."""
        backend = JoblibStorageBackend()
        filepath = Path("/tmp/test_model.joblib")
        expected_data = {"model": "test_model", "metadata": {"version": "1.0"}}

        mock_exists.return_value = True
        mock_load.return_value = expected_data

        result = backend.load(filepath)

        assert result == expected_data
        mock_exists.assert_called_once()
        mock_load.assert_called_once_with(filepath)

    @patch("pathlib.Path.exists")
    def test_load_file_not_found(self, mock_exists):
        """Test load when file doesn't exist."""
        backend = JoblibStorageBackend()
        filepath = Path("/tmp/nonexistent.joblib")

        mock_exists.return_value = False

        with pytest.raises(ModelLoadError) as exc_info:
            backend.load(filepath)

        assert "Model file not found" in str(exc_info.value)

    @patch("joblib.load")
    @patch("pathlib.Path.exists")
    def test_load_joblib_exception(self, mock_exists, mock_load):
        """Test load when joblib.load fails."""
        backend = JoblibStorageBackend()
        filepath = Path("/tmp/test_model.joblib")

        mock_exists.return_value = True
        mock_load.side_effect = RuntimeError("Deserialization failed")

        with pytest.raises(ModelLoadError) as exc_info:
            backend.load(filepath)

        assert "Failed to load model from" in str(exc_info.value)


class TestPickleStorageBackend:
    """Test pickle storage backend."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = PickleStorageBackend()
        assert isinstance(backend, ModelStorageBackend)

    @patch("builtins.open", mock_open())
    @patch("pickle.dump")
    @patch("pathlib.Path.mkdir")
    def test_save_success(self, mock_mkdir, mock_dump):
        """Test successful model saving."""
        backend = PickleStorageBackend()
        model_data = {"model": "test_model", "metadata": {"version": "1.0"}}
        filepath = Path("/tmp/test_model.pkl")

        backend.save(model_data, filepath)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_dump.assert_called_once()

    @patch("builtins.open", mock_open())
    @patch("pickle.dump")
    @patch("pathlib.Path.mkdir")
    def test_save_mkdir_exception(self, mock_mkdir, mock_dump):
        """Test save when mkdir fails."""
        backend = PickleStorageBackend()
        model_data = {"model": "test_model"}
        filepath = Path("/tmp/test_model.pkl")

        mock_mkdir.side_effect = OSError("Permission denied")

        with pytest.raises(ModelError) as exc_info:
            backend.save(model_data, filepath)

        assert "Failed to save model to" in str(exc_info.value)
        assert exc_info.value.error_code == "MODEL_001"

    @patch("builtins.open", side_effect=OSError("Write failed"))
    @patch("pathlib.Path.mkdir")
    def test_save_file_exception(self, mock_mkdir, mock_open):
        """Test save when file opening fails."""
        backend = PickleStorageBackend()
        model_data = {"model": "test_model"}
        filepath = Path("/tmp/test_model.pkl")

        with pytest.raises(ModelError) as exc_info:
            backend.save(model_data, filepath)

        assert "Failed to save model to" in str(exc_info.value)
        assert exc_info.value.error_code == "MODEL_001"

    @patch("builtins.open", mock_open())
    @patch("pickle.load")
    @patch("pathlib.Path.exists")
    def test_load_success(self, mock_exists, mock_load):
        """Test successful model loading."""
        backend = PickleStorageBackend()
        filepath = Path("/tmp/test_model.pkl")
        expected_data = {"model": "test_model", "metadata": {"version": "1.0"}}

        mock_exists.return_value = True
        mock_load.return_value = expected_data

        result = backend.load(filepath)

        assert result == expected_data
        mock_exists.assert_called_once()
        mock_load.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_load_file_not_found(self, mock_exists):
        """Test load when file doesn't exist."""
        backend = PickleStorageBackend()
        filepath = Path("/tmp/nonexistent.pkl")

        mock_exists.return_value = False

        with pytest.raises(ModelLoadError) as exc_info:
            backend.load(filepath)

        assert "Model file not found" in str(exc_info.value)

    @patch("builtins.open", side_effect=OSError("Read failed"))
    @patch("pathlib.Path.exists")
    def test_load_file_exception(self, mock_exists, mock_open):
        """Test load when file opening fails."""
        backend = PickleStorageBackend()
        filepath = Path("/tmp/test_model.pkl")

        mock_exists.return_value = True

        with pytest.raises(ModelLoadError) as exc_info:
            backend.load(filepath)

        assert "Failed to load model from" in str(exc_info.value)

    @patch("builtins.open", mock_open())
    @patch("pickle.load", side_effect=pickle.UnpicklingError("Invalid format"))
    @patch("pathlib.Path.exists")
    def test_load_pickle_exception(self, mock_exists, mock_load):
        """Test load when pickle.load fails."""
        backend = PickleStorageBackend()
        filepath = Path("/tmp/test_model.pkl")

        mock_exists.return_value = True

        with pytest.raises(ModelLoadError) as exc_info:
            backend.load(filepath)

        assert "Failed to load model from" in str(exc_info.value)


class TestModelStorageManager:
    """Test model storage manager."""

    def test_initialization_default_backend(self):
        """Test initialization with default backend."""
        manager = ModelStorageManager()
        assert isinstance(manager.backend, JoblibStorageBackend)

    def test_initialization_joblib_backend(self):
        """Test initialization with joblib backend."""
        manager = ModelStorageManager(backend="joblib")
        assert isinstance(manager.backend, JoblibStorageBackend)

    def test_initialization_pickle_backend(self):
        """Test initialization with pickle backend."""
        manager = ModelStorageManager(backend="pickle")
        assert isinstance(manager.backend, PickleStorageBackend)

    def test_initialization_invalid_backend(self):
        """Test initialization with invalid backend."""
        with pytest.raises(ValueError) as exc_info:
            ModelStorageManager(backend="invalid")

        assert "Unknown backend: invalid" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    @patch.object(JoblibStorageBackend, "save")
    def test_save_model_string_path(self, mock_save):
        """Test save model with string path."""
        manager = ModelStorageManager()
        model_data = {"model": "test_model"}
        filepath = "/tmp/test_model.joblib"

        result = manager.save_model(model_data, filepath)

        assert isinstance(result, Path)
        assert str(result) == filepath
        mock_save.assert_called_once_with(model_data, Path(filepath))

    @patch.object(JoblibStorageBackend, "save")
    def test_save_model_path_object(self, mock_save):
        """Test save model with Path object."""
        manager = ModelStorageManager()
        model_data = {"model": "test_model"}
        filepath = Path("/tmp/test_model.joblib")

        result = manager.save_model(model_data, filepath)

        assert result == filepath
        mock_save.assert_called_once_with(model_data, filepath)

    @patch.object(JoblibStorageBackend, "load")
    def test_load_model_string_path(self, mock_load):
        """Test load model with string path."""
        manager = ModelStorageManager()
        expected_data = {"model": "test_model"}
        filepath = "/tmp/test_model.joblib"

        mock_load.return_value = expected_data

        result = manager.load_model(filepath)

        assert result == expected_data
        mock_load.assert_called_once_with(Path(filepath))

    @patch.object(JoblibStorageBackend, "load")
    def test_load_model_path_object(self, mock_load):
        """Test load model with Path object."""
        manager = ModelStorageManager()
        expected_data = {"model": "test_model"}
        filepath = Path("/tmp/test_model.joblib")

        mock_load.return_value = expected_data

        result = manager.load_model(filepath)

        assert result == expected_data
        mock_load.assert_called_once_with(filepath)

    def test_backend_switching(self):
        """Test switching between different backends."""
        # Start with joblib
        manager = ModelStorageManager(backend="joblib")
        assert isinstance(manager.backend, JoblibStorageBackend)

        # Switch to pickle
        manager = ModelStorageManager(backend="pickle")
        assert isinstance(manager.backend, PickleStorageBackend)

    def test_backends_registry(self):
        """Test that backends registry is properly populated."""
        manager = ModelStorageManager()

        assert "joblib" in manager._backends
        assert "pickle" in manager._backends
        assert isinstance(manager._backends["joblib"], JoblibStorageBackend)
        assert isinstance(manager._backends["pickle"], PickleStorageBackend)


class TestIntegration:
    """Integration tests using real temporary files."""

    def test_joblib_backend_integration(self):
        """Test joblib backend with real files."""
        backend = JoblibStorageBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.joblib"
            model_data = {"model": "test_data", "version": 1}

            # Save
            backend.save(model_data, filepath)
            assert filepath.exists()

            # Load
            loaded_data = backend.load(filepath)
            assert loaded_data == model_data

    def test_pickle_backend_integration(self):
        """Test pickle backend with real files."""
        backend = PickleStorageBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.pkl"
            model_data = {"model": "test_data", "version": 1}

            # Save
            backend.save(model_data, filepath)
            assert filepath.exists()

            # Load
            loaded_data = backend.load(filepath)
            assert loaded_data == model_data

    def test_storage_manager_integration(self):
        """Test storage manager with real files."""
        manager = ModelStorageManager(backend="joblib")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.joblib"
            model_data = {"model": "test_data", "version": 1}

            # Save
            saved_path = manager.save_model(model_data, filepath)
            assert saved_path.exists()

            # Load
            loaded_data = manager.load_model(filepath)
            assert loaded_data == model_data
