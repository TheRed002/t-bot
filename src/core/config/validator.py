"""Configuration validation utilities."""

import os
from typing import Dict


def check_env_pollution() -> Dict[str, str]:
    """
    Check for placeholder environment variables.

    Returns:
        Dictionary of placeholder variables found
    """
    placeholders = {}
    placeholder_patterns = ['your_', 'changeme', 'placeholder', 'example_', 'demo_']

    for key, value in os.environ.items():
        if not value:
            continue

        value_lower = value.lower()
        if any(pattern in value_lower for pattern in placeholder_patterns):
            placeholders[key] = value

    return placeholders


def validate_environment(strict: bool = False) -> None:
    """
    Validate environment before loading config.

    Args:
        strict: If True, raise error on placeholders. If False, just warn.

    Raises:
        EnvironmentError: If placeholders found and strict=True
    """
    polluted = check_env_pollution()

    if not polluted:
        return

    message = (
        f"⚠️  Placeholder environment variables detected ({len(polluted)} total):\n"
    )
    for key, value in list(polluted.items())[:5]:  # Show first 5
        message += f"  - {key}={value[:50]}...\n" if len(value) > 50 else f"  - {key}={value}\n"

    if len(polluted) > 5:
        message += f"  ... and {len(polluted) - 5} more\n"

    message += "\n💡 Solution: Unset these variables or update .env file:\n"
    message += "  unset " + " ".join(list(polluted.keys())[:5])

    if strict:
        raise EnvironmentError(message)
    else:
        import warnings
        warnings.warn(message, UserWarning, stacklevel=2)


def validate_credentials(config_dict: Dict[str, str], source: str = "config") -> None:
    """
    Validate that credentials are not placeholders.

    Args:
        config_dict: Dictionary of config values to validate
        source: Source name for error messages

    Raises:
        ValueError: If placeholder credentials found
    """
    placeholders = {}
    placeholder_patterns = ['your_', 'changeme', 'placeholder', 'example_']

    for key, value in config_dict.items():
        if not isinstance(value, str) or not value:
            continue

        value_lower = value.lower()
        if any(pattern in value_lower for pattern in placeholder_patterns):
            placeholders[key] = value

    if placeholders:
        raise ValueError(
            f"Placeholder values detected in {source}:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in placeholders.items())
            + "\n\nUpdate your .env file with real credentials."
        )
