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
    from ..utils.decorators import time_execution
except ImportError as e:
    logger.warning(f"Failed to import time_execution decorator: {e}")

    # Provide a no-op fallback decorator
    def time_execution(func: Callable) -> Callable:
        """Fallback time_execution decorator that does nothing."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper


# Try to import ValidationService with error handling
try:
    from ..utils.validation.service import ValidationService
except ImportError as e:
    logger.error(f"Failed to import ValidationService: {e}")
    # Re-raise as this is critical for factory.py
    raise ImportError(
        "ValidationService is required for state module operation. "
        "Please ensure utils.validation module is properly installed."
    ) from e


# Import file utilities
try:
    from ..utils.file_utils import ensure_directory_exists
except ImportError as e:
    logger.warning(f"Failed to import file utilities: {e}")

    # Provide a fallback implementation
    def ensure_directory_exists(path: str) -> None:
        """Fallback ensure_directory_exists that does nothing."""
        pass


# Export all imported utilities
__all__ = ["ValidationService", "ensure_directory_exists", "time_execution"]
