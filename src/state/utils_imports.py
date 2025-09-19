"""
Centralized utils imports for the state module.

This module provides safe imports from the utils module with proper error handling
and fallback mechanisms to ensure the state module can function even if some
utils features are unavailable.
"""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Handle logging import gracefully for testing environments
try:
    from src.core.logging import get_logger
except (ImportError, AttributeError):
    # Fallback for test environments where logging might be mocked
    def get_logger(name):
        from unittest.mock import Mock
        # Create a mock logger with all the necessary methods
        mock_logger = Mock()
        mock_logger.debug = Mock()
        mock_logger.info = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        mock_logger.critical = Mock()
        mock_logger.setLevel = Mock()
        mock_logger.level = 50  # CRITICAL level
        return mock_logger

logger = get_logger(__name__)


# Try to import time_execution decorator with fallback
try:
    from src.utils.decorators import time_execution
except ImportError as e:
    logger.error(f"Failed to import time_execution decorator: {e}")

    # Provide a fallback no-op decorator
    def time_execution(func: Callable) -> Callable:
        """Fallback time_execution decorator."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not hasattr(wrapper, "_warned"):
                logger.debug(f"Performance monitoring not available for {func.__name__}")
                wrapper._warned = True
            return func(*args, **kwargs)

        return wrapper


# Try to import ValidationService with error handling
try:
    from src.utils.validation.service import ValidationService
except ImportError as e:
    logger.error(f"Failed to import ValidationService: {e}")
    # Re-raise as this is critical for factory.py
    raise ImportError(
        "ValidationService is required for state module operation. "
        "Please ensure utils.validation module is properly installed."
    ) from e


# Import utilities from centralized modules
try:
    from src.utils.state_utils import ensure_directory_exists
except ImportError as e:
    logger.error(f"Failed to import state utilities: {e}")

    # Provide a fallback implementation that uses standard library
    def ensure_directory_exists(directory_path: str | Path) -> None:
        """Fallback ensure_directory_exists using standard library."""
        from pathlib import Path

        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_error:
            logger.error(f"Failed to create directory {directory_path}: {mkdir_error}")
            raise


# Import state constants
try:
    from src.utils.state_constants import (
        CHECKPOINT_FILE_EXTENSION,
        DEFAULT_CACHE_TTL,
        DEFAULT_CLEANUP_INTERVAL,
        DEFAULT_COMPRESSION_THRESHOLD,
        DEFAULT_MAX_CHECKPOINTS,
        DEFAULT_TRADE_STALENESS_THRESHOLD,
    )
except ImportError as e:
    logger.warning(f"Failed to import state constants: {e}")
    # Set fallback values
    DEFAULT_CACHE_TTL = 300
    DEFAULT_COMPRESSION_THRESHOLD = 1024
    DEFAULT_MAX_CHECKPOINTS = 50
    CHECKPOINT_FILE_EXTENSION = ".checkpoint"
    DEFAULT_CLEANUP_INTERVAL = 3600  # 1 hour
    DEFAULT_TRADE_STALENESS_THRESHOLD = 3600  # 1 hour


# Try to import validation utilities
try:
    from src.utils.validation_utilities import AuditEventType
except ImportError as e:
    logger.warning(f"Failed to import validation utilities: {e}")
    # Provide fallback enum
    from enum import Enum

    class AuditEventType(Enum):
        STATE_UPDATED = "state_updated"
        STATE_CREATED = "state_created"
        STATE_DELETED = "state_deleted"


# Export all imported utilities
__all__ = [
    "CHECKPOINT_FILE_EXTENSION",
    "DEFAULT_CACHE_TTL",
    "DEFAULT_CLEANUP_INTERVAL",
    "DEFAULT_COMPRESSION_THRESHOLD",
    "DEFAULT_MAX_CHECKPOINTS",
    "DEFAULT_TRADE_STALENESS_THRESHOLD",
    "AuditEventType",
    "ValidationService",
    "ensure_directory_exists",
    "logger",
    "time_execution",
]
