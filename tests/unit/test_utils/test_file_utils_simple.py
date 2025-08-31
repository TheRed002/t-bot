"""
Unit tests for file_utils module (simplified).

Tests file utility functions that actually exist in the module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock

from src.utils.file_utils import (
    safe_read_file,
    safe_write_file,
    ensure_directory_exists,
    get_file_size,
    load_config_file,
    save_config_file,
    delete_file,
    list_files,
)
from src.core.exceptions import ValidationError


class TestSafeReadFile:
    """Test safe_read_file function."""

    def test_read_existing_file(self):
        """Test reading an existing file."""
        mock_content = "Test file content"
        
        with patch("builtins.open", mock_open(read_data=mock_content)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.is_file", return_value=True):
                    result = safe_read_file("test.txt")
                    assert result == mock_content

    def test_read_with_encoding(self):
        """Test reading file with specific encoding."""
        content = "Test with special chars: àéîôù"
        
        with patch("builtins.open", mock_open(read_data=content)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.is_file", return_value=True):
                    result = safe_read_file("test.txt", encoding="utf-8")
                    assert result == content

    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValidationError, match="File does not exist"):
                safe_read_file("nonexistent.txt")

    def test_read_permission_error(self):
        """Test reading file with permission error."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("builtins.open", side_effect=PermissionError("No permission")):
                    with pytest.raises(ValidationError):
                        safe_read_file("protected.txt")


class TestSafeWriteFile:
    """Test safe_write_file function."""

    def test_write_file_success(self):
        """Test successful file writing."""
        content = "Test content to write"
        
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.parent") as mock_parent:
                with patch("pathlib.Path.replace") as mock_replace:
                    with patch("pathlib.Path.exists", return_value=False) as mock_exists:
                        mock_parent.mkdir = MagicMock()
                        safe_write_file("test.txt", content)
                        mock_file.assert_called()
                        mock_replace.assert_called_once()

    def test_write_file_permission_error(self):
        """Test writing file with permission error."""
        content = "Test content"
        
        with patch("builtins.open", side_effect=PermissionError("No permission")):
            with pytest.raises(ValidationError):
                safe_write_file("protected.txt", content)

    def test_write_with_encoding(self):
        """Test writing file with specific encoding."""
        content = "Test with special chars: àéîôù"
        
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.parent") as mock_parent:
                with patch("pathlib.Path.replace") as mock_replace:
                    mock_parent.mkdir = MagicMock()
                    safe_write_file("test.txt", content, encoding="utf-8")
                    mock_file.assert_called()
                    mock_replace.assert_called_once()


class TestEnsureDirectoryExists:
    """Test ensure_directory_exists function."""

    def test_create_new_directory(self):
        """Test creating new directory."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            ensure_directory_exists("new/directory")
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_directory_creation_error(self):
        """Test directory creation with error."""
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("No permission")):
            with pytest.raises(ValidationError):
                ensure_directory_exists("protected/directory")


class TestGetFileSize:
    """Test get_file_size function."""

    def test_get_size_success(self):
        """Test getting file size successfully."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1024
                result = get_file_size("test.txt")
                assert result == 1024

    def test_get_size_file_not_found(self):
        """Test getting size of non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValidationError, match="File does not exist"):
                get_file_size("nonexistent.txt")

    def test_get_size_permission_error(self):
        """Test getting size with permission error."""
        with patch("pathlib.Path.stat", side_effect=PermissionError("No permission")):
            with pytest.raises(ValidationError):
                get_file_size("protected.txt")


class TestLoadConfigFile:
    """Test load_config_file function."""

    def test_load_json_config(self):
        """Test loading JSON config file."""
        config_data = {"key": "value", "number": 123}
        json_content = json.dumps(config_data)
        
        with patch("src.utils.file_utils.safe_read_file", return_value=json_content):
            result = load_config_file("config.json")
            assert result == config_data

    def test_load_yaml_config(self):
        """Test loading YAML config file."""
        config_data = {"key": "value", "number": 123}
        yaml_content = "key: value\nnumber: 123"
        
        with patch("src.utils.file_utils.safe_read_file", return_value=yaml_content):
            with patch("yaml.safe_load", return_value=config_data):
                result = load_config_file("config.yaml")
                assert result == config_data

    def test_load_config_unsupported_format(self):
        """Test loading config with unsupported format."""
        with patch("src.utils.file_utils.safe_read_file", return_value="content"):
            with pytest.raises(ValidationError):
                load_config_file("config.txt")

    def test_load_config_invalid_json(self):
        """Test loading invalid JSON config."""
        invalid_json = "{ invalid json }"
        
        with patch("src.utils.file_utils.safe_read_file", return_value=invalid_json):
            with pytest.raises(ValidationError):
                load_config_file("config.json")

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML config."""
        invalid_yaml = "key: value\n  invalid: yaml: structure"
        
        with patch("src.utils.file_utils.safe_read_file", return_value=invalid_yaml):
            with patch("yaml.safe_load", side_effect=Exception("Invalid YAML")):
                with pytest.raises(ValidationError):
                    load_config_file("config.yml")


class TestSaveConfigFile:
    """Test save_config_file function."""

    def test_save_json_config(self):
        """Test saving JSON config file."""
        config_data = {"key": "value", "number": 123}
        
        with patch("src.utils.file_utils.safe_write_file") as mock_write:
            save_config_file("config.json", config_data)
            mock_write.assert_called_once()
            # Check that JSON was written
            written_content = mock_write.call_args[0][1]
            assert isinstance(written_content, str)

    def test_save_yaml_config(self):
        """Test saving YAML config file."""
        config_data = {"key": "value", "number": 123}
        
        with patch("src.utils.file_utils.safe_write_file") as mock_write:
            with patch("yaml.dump", return_value="key: value\nnumber: 123\n"):
                save_config_file("config.yaml", config_data)
                mock_write.assert_called_once()

    def test_save_config_unsupported_format(self):
        """Test saving config with unsupported format."""
        config_data = {"key": "value"}
        
        with pytest.raises(ValidationError):
            save_config_file("config.txt", config_data)


class TestDeleteFile:
    """Test delete_file function."""

    def test_delete_file_success(self):
        """Test successful file deletion."""
        with patch("pathlib.Path.unlink") as mock_unlink:
            with patch("pathlib.Path.exists", return_value=True):
                delete_file("test.txt")
                mock_unlink.assert_called_once()

    def test_delete_nonexistent_file(self):
        """Test deleting non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            # Should not raise error for non-existent file
            delete_file("nonexistent.txt")

    def test_delete_file_permission_error(self):
        """Test deleting file with permission error."""
        with patch("pathlib.Path.unlink", side_effect=PermissionError("No permission")):
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(ValidationError):
                    delete_file("protected.txt")


class TestListFiles:
    """Test list_files function."""

    def test_list_files_success(self):
        """Test listing files successfully."""
        mock_files = ["file1.txt", "file2.txt", "file3.txt"]
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.glob") as mock_glob:
                    # Create mock Path objects with is_file() method
                    mock_path_objects = []
                    for f in mock_files:
                        mock_path = MagicMock()
                        # Fix: Set return_value directly instead of lambda function
                        mock_path.__str__.return_value = f
                        mock_path.is_file.return_value = True
                        mock_path_objects.append(mock_path)
                    mock_glob.return_value = mock_path_objects
                    result = list_files("/test/directory")
                    assert len(result) == 3
                    assert all(isinstance(f, str) for f in result)

    def test_list_files_with_pattern(self):
        """Test listing files with specific pattern."""
        mock_files = ["file1.txt", "file2.txt"]
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.glob") as mock_glob:
                    # Create mock Path objects with is_file() method
                    mock_path_objects = []
                    for f in mock_files:
                        mock_path = MagicMock()
                        # Fix: Set return_value directly instead of lambda function
                        mock_path.__str__.return_value = f
                        mock_path.is_file.return_value = True
                        mock_path_objects.append(mock_path)
                    mock_glob.return_value = mock_path_objects
                    result = list_files("/test/directory", pattern="*.txt")
                    assert len(result) == 2

    def test_list_files_empty_directory(self):
        """Test listing files in empty directory."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.glob") as mock_glob:
                    mock_glob.return_value = []
                    result = list_files("/test/empty")
                    assert result == []

    def test_list_files_permission_error(self):
        """Test listing files with permission error."""
        with patch("pathlib.Path.glob", side_effect=PermissionError("No permission")):
            with pytest.raises(ValidationError):
                list_files("/protected/directory")

    def test_list_files_directory_not_found(self):
        """Test listing files in non-existent directory."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValidationError, match="Directory does not exist"):
                list_files("/nonexistent/directory")


class TestFileUtilsIntegration:
    """Test integration between different file utilities."""

    def test_config_round_trip(self):
        """Test complete config save/load cycle."""
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432
            },
            "api": {
                "timeout": 30,
                "retries": 3
            }
        }
        
        filename = "test_config.json"
        
        # Mock the write operation
        with patch("src.utils.file_utils.safe_write_file") as mock_write:
            save_config_file(filename, config_data)
            mock_write.assert_called_once()
            
            # Get the written JSON content
            written_content = mock_write.call_args[0][1]
            
            # Mock the read operation to return the same content
            with patch("src.utils.file_utils.safe_read_file", return_value=written_content):
                loaded_config = load_config_file(filename)
                assert loaded_config == config_data

    def test_file_workflow_with_cleanup(self):
        """Test complete file workflow with cleanup."""
        content = "Test file workflow"
        filename = "workflow_test.txt"
        
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("src.utils.file_utils.Path") as mock_path:
                mock_path_instance = MagicMock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.stat.return_value.st_size = len(content)
                mock_path_instance.exists.return_value = True
                mock_path_instance.parent.mkdir = MagicMock()
                mock_path_instance.replace = MagicMock()  # Add this for atomic write
                mock_path_instance.with_suffix.return_value = mock_path_instance  # For temp file
                
                # Write file
                safe_write_file(filename, content)
                
                # Get file size
                size = get_file_size(filename)
                assert size == len(content)
                
                # Delete file
                delete_file(filename)
                mock_path_instance.unlink.assert_called()

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across functions."""
        # All functions should raise ValidationError for common issues
        
        # Permission errors
        with patch("builtins.open", side_effect=PermissionError("No permission")):
            with pytest.raises(ValidationError):
                safe_read_file("protected.txt")
        
        with patch("builtins.open", side_effect=PermissionError("No permission")):
            with pytest.raises(ValidationError):
                safe_write_file("protected.txt", "content")
        
        with patch("pathlib.Path.stat", side_effect=PermissionError("No permission")):
            with pytest.raises(ValidationError):
                get_file_size("protected.txt")

    def test_directory_operations(self):
        """Test directory-related operations."""
        directory_path = "test/nested/directory"
        
        # Ensure directory exists
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            ensure_directory_exists(directory_path)
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)
        
        # List files in directory
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.glob") as mock_glob:
                    # Create proper mock objects
                    mock_files = []
                    for filename in ["file1.txt", "file2.txt"]:
                        mock_path = MagicMock()
                        mock_path.__str__.return_value = filename
                        mock_path.is_file.return_value = True
                        mock_files.append(mock_path)
                    mock_glob.return_value = mock_files
                    files = list_files(directory_path)
            assert len(files) == 2