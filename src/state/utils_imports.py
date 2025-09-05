"""
Centralized utils imports for the state module.

This module provides safe imports from the utils module with proper error handling
and fallback mechanisms to ensure the state module can function even if some
utils features are unavailable.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


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
    def ensure_directory_exists(directory_path: str) -> None:
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
        DEFAULT_CACHE_TTL,
        DEFAULT_COMPRESSION_THRESHOLD,
        DEFAULT_MAX_CHECKPOINTS,
        CHECKPOINT_FILE_EXTENSION
    )
except ImportError as e:
    logger.warning(f"Failed to import state constants: {e}")
    # Set fallback values
    DEFAULT_CACHE_TTL = 300
    DEFAULT_COMPRESSION_THRESHOLD = 1024
    DEFAULT_MAX_CHECKPOINTS = 50
    CHECKPOINT_FILE_EXTENSION = ".checkpoint"


# Export all imported utilities
__all__ = [
    "ValidationService", 
    "ensure_directory_exists", 
    "time_execution",
    "DEFAULT_CACHE_TTL",
    "DEFAULT_COMPRESSION_THRESHOLD", 
    "DEFAULT_MAX_CHECKPOINTS",
    "CHECKPOINT_FILE_EXTENSION"
]
