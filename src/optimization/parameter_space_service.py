"""
Parameter space service for optimization module.

This module provides specialized services for parameter space operations,
separating these concerns from the main optimization service.
"""

from typing import Any

from src.core.base import BaseService
from src.core.exceptions import ValidationError
from src.optimization.parameter_space import ParameterSpace, ParameterSpaceBuilder


class ParameterSpaceService(BaseService):
    """
    Service for parameter space operations.

    Handles parameter space building, validation, and transformations
    to keep optimization service focused on coordination.
    """

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize parameter space service.

        Args:
            name: Service name for identification
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name or "ParameterSpaceService", config, correlation_id)
        self._logger.info("ParameterSpaceService initialized")

    def build_parameter_space(self, config: dict[str, Any]) -> ParameterSpace:
        """
        Build parameter space from configuration.

        Args:
            config: Parameter space configuration

        Returns:
            Built parameter space
        """
        builder = ParameterSpaceBuilder()

        for param_name, param_config in config.items():
            param_type = param_config.get("type")

            if param_type == "continuous":
                builder.add_continuous(
                    name=param_name,
                    min_value=param_config["min_value"],
                    max_value=param_config["max_value"],
                    precision=param_config.get("precision", 3),
                )
            elif param_type == "discrete":
                builder.add_discrete(
                    name=param_name,
                    min_value=param_config["min_value"],
                    max_value=param_config["max_value"],
                    step_size=param_config.get("step_size", 1),
                )
            elif param_type == "categorical":
                builder.add_categorical(
                    name=param_name,
                    values=param_config["values"],
                )
            elif param_type == "boolean":
                builder.add_boolean(
                    name=param_name,
                    true_probability=param_config.get("true_probability", 0.5),
                )
            else:
                raise ValidationError(
                    f"Invalid parameter type: {param_type}",
                    error_code="OPT_006",
                    field_name="parameter_type",
                    field_value=param_type,
                )

        return builder.build()

    def build_parameter_space_from_current(
        self, current_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build parameter space configuration from current parameters.

        Args:
            current_parameters: Current parameter values

        Returns:
            Parameter space configuration
        """
        parameter_space_config = {}

        for param_name, param_value in current_parameters.items():
            if isinstance(param_value, (int, float)):
                # Create continuous parameter space around current value
                min_val = float(param_value) * 0.5  # 50% lower
                max_val = float(param_value) * 2.0  # 100% higher
                parameter_space_config[param_name] = {
                    "type": "continuous",
                    "min_value": min_val,
                    "max_value": max_val,
                    "precision": 4,
                }
            elif isinstance(param_value, bool):
                parameter_space_config[param_name] = {
                    "type": "boolean",
                    "true_probability": 0.5,
                }
            elif isinstance(param_value, str):
                # For string parameters, treat as categorical with current value
                parameter_space_config[param_name] = {
                    "type": "categorical",
                    "values": [param_value],
                }

        # If no parameters to optimize, add default trading parameters
        if not parameter_space_config:
            parameter_space_config = self._get_default_trading_parameter_space()

        return parameter_space_config

    def _get_default_trading_parameter_space(self) -> dict[str, Any]:
        """Get default trading parameter space configuration."""
        return {
            "position_size_pct": {
                "type": "continuous",
                "min_value": 0.01,
                "max_value": 0.05,
                "precision": 4,
            },
            "stop_loss_pct": {
                "type": "continuous",
                "min_value": 0.01,
                "max_value": 0.05,
                "precision": 4,
            },
            "take_profit_pct": {
                "type": "continuous",
                "min_value": 0.02,
                "max_value": 0.10,
                "precision": 4,
            },
        }

    def validate_parameter_space_config(self, config: dict[str, Any]) -> bool:
        """
        Validate parameter space configuration.

        Args:
            config: Parameter space configuration to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError(
                "Parameter space configuration must be a dictionary",
                field_name="parameter_space_config",
                field_value=type(config).__name__,
            )

        if not config:
            raise ValidationError(
                "Parameter space configuration cannot be empty",
                field_name="parameter_space_config",
            )

        # Validate each parameter configuration
        for param_name, param_config in config.items():
            if not isinstance(param_config, dict):
                raise ValidationError(
                    f"Parameter configuration for '{param_name}' must be a dictionary",
                    field_name=f"parameter_config.{param_name}",
                    field_value=type(param_config).__name__,
                )

            param_type = param_config.get("type")
            if not param_type:
                raise ValidationError(
                    f"Parameter '{param_name}' missing required 'type' field",
                    field_name=f"parameter_config.{param_name}.type",
                )

            # Validate type-specific requirements
            self._validate_parameter_type_config(param_name, param_type, param_config)

        return True

    def _validate_parameter_type_config(
        self, param_name: str, param_type: str, param_config: dict[str, Any]
    ) -> None:
        """Validate parameter type-specific configuration."""
        if param_type == "continuous":
            required_fields = ["min_value", "max_value"]
            for field in required_fields:
                if field not in param_config:
                    raise ValidationError(
                        f"Continuous parameter '{param_name}' missing required field '{field}'",
                        field_name=f"parameter_config.{param_name}.{field}",
                    )

        elif param_type == "discrete":
            required_fields = ["min_value", "max_value"]
            for field in required_fields:
                if field not in param_config:
                    raise ValidationError(
                        f"Discrete parameter '{param_name}' missing required field '{field}'",
                        field_name=f"parameter_config.{param_name}.{field}",
                    )

        elif param_type == "categorical":
            if "values" not in param_config:
                raise ValidationError(
                    f"Categorical parameter '{param_name}' missing required field 'values'",
                    field_name=f"parameter_config.{param_name}.values",
                )
            if not isinstance(param_config["values"], list) or not param_config["values"]:
                raise ValidationError(
                    f"Categorical parameter '{param_name}' 'values' must be a non-empty list",
                    field_name=f"parameter_config.{param_name}.values",
                )

        elif param_type not in ["boolean"]:
            raise ValidationError(
                f"Invalid parameter type '{param_type}' for parameter '{param_name}'",
                field_name=f"parameter_config.{param_name}.type",
                field_value=param_type,
            )