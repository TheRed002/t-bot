"""
Configuration Conversion Utilities.

Common utility functions for converting configuration objects to dictionaries
across different formats (Pydantic, attrs, plain dicts, etc.).
"""

from typing import Any


def convert_config_to_dict(config: Any) -> dict[str, Any]:
    """
    Convert configuration object to dictionary format.

    Supports multiple config object types:
    - Pydantic models (model_dump, dict methods)
    - Dict-like objects (get method)
    - Plain dictionaries
    - Other objects (fallback to empty dict)

    Args:
        config: Configuration object of any supported type

    Returns:
        Dictionary representation of the config, or empty dict if conversion fails
    """
    if config is None:
        return {}

    # Pydantic v2 models
    if hasattr(config, "model_dump"):
        return config.model_dump()

    # Pydantic v1 models or other dict-convertible objects
    elif hasattr(config, "dict"):
        return config.dict()

    # Already a dictionary
    elif isinstance(config, dict):
        return config

    # Dict-like objects with get method
    elif hasattr(config, "get"):
        # Try to convert to dict if possible
        try:
            return dict(config)
        except (TypeError, ValueError):
            return {}

    # Fallback for unsupported config types
    else:
        return {}
