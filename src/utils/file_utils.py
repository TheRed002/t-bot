"""File operations utilities for the T-Bot trading system."""

import json
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# Module level logger for static methods
logger = get_logger(__name__)


def safe_read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Safely read a file with error handling.

    Args:
        file_path: Path to file to read
        encoding: File encoding (default "utf-8")

    Returns:
        File contents as string

    Raises:
        ValidationError: If file cannot be read
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")

        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        with open(path, encoding=encoding) as f:
            content = f.read()

        logger.debug(f"Successfully read file: {file_path}")
        return content

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e!s}")
        raise ValidationError(f"Cannot read file '{file_path}': {e!s}") from e


def safe_write_file(file_path: str, content: str, encoding: str = "utf-8") -> None:
    """
    Safely write content to a file with error handling.

    Args:
        file_path: Path to file to write
        content: Content to write
        encoding: File encoding (default "utf-8")

    Raises:
        ValidationError: If file cannot be written
    """
    path = Path(file_path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write content atomically using temporary file
        with open(temp_path, "w", encoding=encoding) as f:
            f.write(content)

        # Atomic move
        temp_path.replace(path)

        logger.debug(f"Successfully wrote file: {file_path}")

    except Exception as e:
        # Clean up temp file on failure
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except (OSError, PermissionError) as cleanup_e:
                # Log cleanup errors but don't fail the operation
                logger.debug(f"Failed to cleanup temp file {temp_path}: {cleanup_e}")

        logger.error(f"Failed to write file {file_path}: {e!s}")
        raise ValidationError(f"Cannot write file '{file_path}': {e!s}") from e


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to directory to create

    Raises:
        ValidationError: If directory cannot be created
    """
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory_path}")

    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise ValidationError(f"Cannot create directory '{directory_path}': {e}") from e


def load_config_file(file_path: str) -> dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        file_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        ValidationError: If file cannot be loaded or parsed
    """
    try:
        content = safe_read_file(file_path)
        path = Path(file_path)

        if path.suffix.lower() in [".yaml", ".yml"]:
            config = yaml.safe_load(content)
        elif path.suffix.lower() == ".json":
            config = json.loads(content)
        else:
            raise ValidationError(f"Unsupported file format: {path.suffix}")

        if not isinstance(config, dict):
            raise ValidationError("Configuration must be a dictionary")

        logger.debug(f"Successfully loaded config file: {file_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load config file {file_path}: {e!s}")
        raise ValidationError(f"Cannot load config file '{file_path}': {e!s}") from e


def save_config_file(file_path: str, config: dict[str, Any]) -> None:
    """
    Save configuration to YAML or JSON file.

    Args:
        file_path: Path to save configuration
        config: Configuration dictionary

    Raises:
        ValidationError: If file cannot be saved
    """
    try:
        path = Path(file_path)

        if path.suffix.lower() in [".yaml", ".yml"]:
            content = yaml.dump(config, default_flow_style=False)
        elif path.suffix.lower() == ".json":
            content = json.dumps(config, indent=2)
        else:
            raise ValidationError(f"Unsupported file format: {path.suffix}")

        safe_write_file(file_path, content)
        logger.debug(f"Successfully saved config file: {file_path}")

    except Exception as e:
        logger.error(f"Failed to save config file {file_path}: {e!s}")
        raise ValidationError(f"Cannot save config file '{file_path}': {e!s}") from e


def delete_file(file_path: str) -> None:
    """
    Safely delete a file.

    Args:
        file_path: Path to file to delete

    Raises:
        ValidationError: If file cannot be deleted
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.debug(f"Successfully deleted file: {file_path}")
        else:
            logger.warning(f"File does not exist: {file_path}")

    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e!s}")
        raise ValidationError(f"Cannot delete file '{file_path}': {e!s}") from e


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Raises:
        ValidationError: If file doesn't exist
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")

        return path.stat().st_size

    except Exception as e:
        raise ValidationError(f"Cannot get file size for '{file_path}': {e!s}") from e


def list_files(directory: str, pattern: str = "*") -> list[str]:
    """
    List files in directory matching pattern.

    Args:
        directory: Directory path
        pattern: Glob pattern (default "*")

    Returns:
        List of file paths

    Raises:
        ValidationError: If directory doesn't exist
    """
    try:
        path = Path(directory)
        if not path.exists():
            raise ValidationError(f"Directory does not exist: {directory}")

        if not path.is_dir():
            raise ValidationError(f"Path is not a directory: {directory}")

        files = [str(f) for f in path.glob(pattern) if f.is_file()]
        return sorted(files)

    except Exception as e:
        raise ValidationError(f"Cannot list files in '{directory}': {e!s}") from e
