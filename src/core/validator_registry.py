"""Central registry for all validators to eliminate duplication."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from src.core.dependency_injection import injectable
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class ValidatorInterface(ABC):
    """Base interface for all validators."""

    @abstractmethod
    def validate(self, data: Any, **kwargs) -> bool:
        """
        Validate data.

        Args:
            data: Data to validate
            **kwargs: Additional validation parameters

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        pass


class CompositeValidator(ValidatorInterface):
    """Composite validator that chains multiple validators."""

    def __init__(self, validators: list[ValidatorInterface]):
        """
        Initialize composite validator.

        Args:
            validators: List of validators to chain
        """
        self.validators = validators

    def validate(self, data: Any, **kwargs) -> bool:
        """Validate data through all validators."""
        for validator in self.validators:
            validator.validate(data, **kwargs)
        return True


@injectable(singleton=True)
class ValidatorRegistry:
    """
    Central registry for all validators.

    This eliminates duplication of validation logic across modules.
    """

    def __init__(self):
        """Initialize validator registry."""
        self._validators: dict[str, ValidatorInterface] = {}
        self._validator_classes: dict[str, type[ValidatorInterface]] = {}
        self._validation_rules: dict[str, list[Callable]] = {}
        self._logger = logger

    def register_validator(self, name: str, validator: ValidatorInterface) -> None:
        """
        Register a validator instance.

        Args:
            name: Validator name
            validator: Validator instance
        """
        self._validators[name] = validator
        self._logger.debug(f"Registered validator: {name}")

    def register_validator_class(
        self, name: str, validator_class: type[ValidatorInterface]
    ) -> None:
        """
        Register a validator class for lazy instantiation.

        Args:
            name: Validator name
            validator_class: Validator class
        """
        self._validator_classes[name] = validator_class
        self._logger.debug(f"Registered validator class: {name}")

    def register_rule(
        self, data_type: str, rule: Callable[[Any], bool], error_message: str | None = None
    ) -> None:
        """
        Register a validation rule for a data type.

        Args:
            data_type: Type of data this rule applies to
            rule: Validation rule function
            error_message: Error message if rule fails
        """
        if data_type not in self._validation_rules:
            self._validation_rules[data_type] = []

        # Wrap rule to include error message
        def wrapped_rule(data):
            if not rule(data):
                raise ValidationError(error_message or f"Validation failed for {data_type}")
            return True

        self._validation_rules[data_type].append(wrapped_rule)
        self._logger.debug(f"Registered validation rule for {data_type}")

    def get_validator(self, name: str) -> ValidatorInterface:
        """
        Get a validator by name.

        Args:
            name: Validator name

        Returns:
            Validator instance

        Raises:
            KeyError: If validator not found
        """
        # Check if instance exists
        if name in self._validators:
            return self._validators[name]

        # Check if class is registered
        if name in self._validator_classes:
            validator = self._validator_classes[name]()
            self._validators[name] = validator  # Cache instance
            return validator

        raise KeyError(f"Validator '{name}' not registered")

    def validate(
        self, data_type: str, data: Any, validator_name: str | None = None, **kwargs
    ) -> bool:
        """
        Validate data using registered validators and rules.

        Args:
            data_type: Type of data being validated
            data: Data to validate
            validator_name: Specific validator to use
            **kwargs: Additional validation parameters

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Use specific validator if provided
        if validator_name:
            validator = self.get_validator(validator_name)
            return validator.validate(data, **kwargs)

        # Apply validation rules for data type
        if data_type in self._validation_rules:
            for rule in self._validation_rules[data_type]:
                rule(data)

        # Try to find a validator by data type name
        try:
            validator = self.get_validator(data_type)
            return validator.validate(data, **kwargs)
        except KeyError:
            # No specific validator, validation passes if rules pass
            return True

    def create_composite_validator(self, validator_names: list[str]) -> CompositeValidator:
        """
        Create a composite validator from multiple validators.

        Args:
            validator_names: Names of validators to compose

        Returns:
            Composite validator
        """
        validators = [self.get_validator(name) for name in validator_names]
        return CompositeValidator(validators)

    def clear(self) -> None:
        """Clear all registered validators."""
        self._validators.clear()
        self._validator_classes.clear()
        self._validation_rules.clear()


# Built-in validators
class RangeValidator(ValidatorInterface):
    """Validator for numeric ranges."""

    def __init__(self, min_value: float | None = None, max_value: float | None = None):
        """
        Initialize range validator.

        Args:
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        """
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, data: Any, **kwargs) -> bool:
        """Validate that data is within range."""
        if not isinstance(data, int | float):
            raise ValidationError(f"Expected numeric value, got {type(data)}")

        if self.min_value is not None and data < self.min_value:
            raise ValidationError(f"Value {data} is below minimum {self.min_value}")

        if self.max_value is not None and data > self.max_value:
            raise ValidationError(f"Value {data} is above maximum {self.max_value}")

        return True


class LengthValidator(ValidatorInterface):
    """Validator for string/collection length."""

    def __init__(self, min_length: int | None = None, max_length: int | None = None):
        """
        Initialize length validator.

        Args:
            min_length: Minimum length
            max_length: Maximum length
        """
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, data: Any, **kwargs) -> bool:
        """Validate length."""
        if not hasattr(data, "__len__"):
            raise ValidationError(f"Data type {type(data)} has no length")

        length = len(data)

        if self.min_length is not None and length < self.min_length:
            raise ValidationError(f"Length {length} is below minimum {self.min_length}")

        if self.max_length is not None and length > self.max_length:
            raise ValidationError(f"Length {length} is above maximum {self.max_length}")

        return True


class PatternValidator(ValidatorInterface):
    """Validator for regex patterns."""

    def __init__(self, pattern: str):
        """
        Initialize pattern validator.

        Args:
            pattern: Regex pattern
        """
        import re

        self.pattern = re.compile(pattern)

    def validate(self, data: Any, **kwargs) -> bool:
        """Validate against pattern."""
        if not isinstance(data, str):
            raise ValidationError(f"Expected string, got {type(data)}")

        if not self.pattern.match(data):
            raise ValidationError(f"Value '{data}' does not match pattern {self.pattern.pattern}")

        return True


class TypeValidator(ValidatorInterface):
    """Validator for type checking."""

    def __init__(self, expected_type: type):
        """
        Initialize type validator.

        Args:
            expected_type: Expected type
        """
        self.expected_type = expected_type

    def validate(self, data: Any, **kwargs) -> bool:
        """Validate type."""
        if not isinstance(data, self.expected_type):
            raise ValidationError(
                f"Expected type {self.expected_type.__name__}, got {type(data).__name__}"
            )
        return True


# Global registry instance
_validator_registry = ValidatorRegistry()


# Register built-in validators
_validator_registry.register_validator_class("range", RangeValidator)
_validator_registry.register_validator_class("length", LengthValidator)
_validator_registry.register_validator_class("pattern", PatternValidator)
_validator_registry.register_validator_class("type", TypeValidator)


# Common validation rules
_validator_registry.register_rule(
    "price", lambda x: isinstance(x, int | float) and x > 0, "Price must be a positive number"
)

_validator_registry.register_rule(
    "quantity",
    lambda x: isinstance(x, int | float) and x > 0,
    "Quantity must be a positive number",
)

_validator_registry.register_rule(
    "percentage",
    lambda x: isinstance(x, int | float) and 0 <= x <= 100,
    "Percentage must be between 0 and 100",
)

_validator_registry.register_rule(
    "symbol",
    lambda x: isinstance(x, str) and len(x) > 0 and "/" in x,
    "Symbol must be a non-empty string in format BASE/QUOTE",
)


# Convenience functions
def register_validator(name: str, validator: ValidatorInterface) -> None:
    """Register a validator."""
    _validator_registry.register_validator(name, validator)


def register_validator_class(name: str, validator_class: type[ValidatorInterface]) -> None:
    """Register a validator class."""
    _validator_registry.register_validator_class(name, validator_class)


def register_rule(data_type: str, rule: Callable, error_message: str | None = None) -> None:
    """Register a validation rule."""
    _validator_registry.register_rule(data_type, rule, error_message)


def validate(data_type: str, data: Any, **kwargs) -> bool:
    """Validate data."""
    return _validator_registry.validate(data_type, data, **kwargs)


def get_validator(name: str) -> ValidatorInterface:
    """Get a validator."""
    return _validator_registry.get_validator(name)
