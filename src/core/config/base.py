"""Base configuration class for the T-Bot trading system."""

from collections.abc import Callable

from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    """Base configuration class with common patterns.

    Provides common Pydantic settings configuration with environment
    variable support, case insensitive matching, and validation.
    """

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "validate_assignment": True,
        "extra": "ignore",
        "populate_by_name": True,  # Allow both field names and aliases
    }

    def __init__(self, **kwargs):
        """Initialize with validators list for extensibility."""
        super().__init__(**kwargs)
        self._validators: list[Callable] = []

    def run_validators(self) -> None:
        """Run all registered validators."""
        for validator in self._validators:
            validator(self)

    def add_validator(self, validator: Callable) -> None:
        """Add a custom validator."""
        self._validators.append(validator)
