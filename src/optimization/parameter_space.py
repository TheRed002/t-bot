"""
Parameter Space Definition and Management.

This module provides comprehensive parameter space definition capabilities
for optimization algorithms, including support for continuous, discrete,
categorical, and conditional parameters with hierarchical dependencies.

Key Features:
- Multiple parameter types (continuous, discrete, categorical, boolean)
- Hierarchical parameter dependencies and constraints
- Intelligent sampling strategies
- Parameter validation and bounds checking
- Support for conditional parameter spaces
- Integration with all optimization algorithms

Critical for Financial Applications:
- Decimal precision for financial parameters
- Proper bounds checking for risk parameters
- Validation of parameter combinations
- Support for complex trading strategy parameters
"""

import random
from abc import ABC, abstractmethod
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ParameterType(Enum):
    """Parameter type enumeration."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    CONDITIONAL = "conditional"


class SamplingStrategy(Enum):
    """Sampling strategy for parameter space exploration."""

    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    LOG_UNIFORM = "log_uniform"
    GRID = "grid"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL = "sobol"
    HALTON = "halton"


class ParameterDefinition(BaseModel, ABC):
    """
    Abstract base class for parameter definitions.

    Defines the interface for all parameter types and provides
    common functionality for validation and sampling.
    """

    name: str = Field(description="Parameter name")
    description: str = Field(default="", description="Parameter description")
    parameter_type: ParameterType = Field(description="Parameter type")
    is_conditional: bool = Field(default=False, description="Whether parameter is conditional")
    conditions: dict[str, Any] = Field(
        default_factory=dict, description="Conditions for parameter activation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional parameter metadata"
    )

    @abstractmethod
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any:
        """
        Sample a value from this parameter's space.

        Args:
            strategy: Sampling strategy to use

        Returns:
            Sampled parameter value
        """
        pass

    @abstractmethod
    def validate_value(self, value: Any) -> bool:
        """
        Validate if a value is valid for this parameter.

        Args:
            value: Value to validate

        Returns:
            True if value is valid
        """
        pass

    @abstractmethod
    def clip_value(self, value: Any) -> Any:
        """
        Clip a value to be within valid bounds.

        Args:
            value: Value to clip

        Returns:
            Clipped value
        """
        pass

    @abstractmethod
    def get_bounds(self) -> tuple[Any, Any]:
        """
        Get parameter bounds.

        Returns:
            Tuple of (min_value, max_value)
        """
        pass

    def is_active(self, context: dict[str, Any]) -> bool:
        """
        Check if parameter is active given the context.

        Args:
            context: Current parameter context

        Returns:
            True if parameter should be active
        """
        if not self.is_conditional or not self.conditions:
            return True

        for condition_param, condition_value in self.conditions.items():
            if condition_param not in context:
                return False

            context_value = context[condition_param]

            # Handle different condition types
            if isinstance(condition_value, list | tuple | set):
                if context_value not in condition_value:
                    return False
            elif isinstance(condition_value, dict):
                # Range condition
                if "min" in condition_value and context_value < condition_value["min"]:
                    return False
                if "max" in condition_value and context_value > condition_value["max"]:
                    return False
            else:
                if context_value != condition_value:
                    return False

        return True


class ContinuousParameter(ParameterDefinition):
    """
    Continuous parameter definition for real-valued parameters.

    Supports different distributions and sampling strategies
    with proper decimal precision handling.
    """

    parameter_type: ParameterType = Field(default=ParameterType.CONTINUOUS)
    min_value: Decimal = Field(description="Minimum parameter value")
    max_value: Decimal = Field(description="Maximum parameter value")
    precision: int | None = Field(
        default=None, ge=0, description="Decimal precision (None for arbitrary precision)"
    )
    log_scale: bool = Field(
        default=False, description="Whether parameter should be sampled on log scale"
    )
    default_value: Decimal | None = Field(default=None, description="Default parameter value")

    @field_validator("max_value")
    @classmethod
    def validate_bounds(cls, v, values):
        """Validate that max_value > min_value."""
        if "min_value" in values.data and v <= values.data["min_value"]:
            raise ValueError("max_value must be greater than min_value")
        return v

    @field_validator("default_value")
    @classmethod
    def validate_default(cls, v, values):
        """Validate default value is within bounds."""
        if v is not None:
            if "min_value" in values.data and v < values.data["min_value"]:
                raise ValueError("default_value must be >= min_value")
            if "max_value" in values.data and v > values.data["max_value"]:
                raise ValueError("default_value must be <= max_value")
        return v

    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Decimal:
        """Sample a value from the continuous parameter space."""
        if strategy == SamplingStrategy.UNIFORM:
            if self.log_scale:
                # Log-uniform sampling
                log_min = float(self.min_value.ln())
                log_max = float(self.max_value.ln())
                log_value = random.uniform(log_min, log_max)
                value = Decimal(str(np.exp(log_value)))
            else:
                # Linear uniform sampling
                range_size = self.max_value - self.min_value
                random_factor = Decimal(str(random.random()))
                value = self.min_value + range_size * random_factor

        elif strategy == SamplingStrategy.GAUSSIAN:
            # Gaussian sampling centered at midpoint
            center = (self.min_value + self.max_value) / 2
            std_dev = (self.max_value - self.min_value) / 6  # 3-sigma rule
            value = center + Decimal(str(random.gauss(0, float(std_dev))))

        elif strategy == SamplingStrategy.LOG_UNIFORM:
            # Force log-uniform sampling regardless of log_scale setting
            log_min = float(self.min_value.ln())
            log_max = float(self.max_value.ln())
            log_value = random.uniform(log_min, log_max)
            value = Decimal(str(np.exp(log_value)))

        else:
            # Default to uniform sampling
            range_size = self.max_value - self.min_value
            random_factor = Decimal(str(random.random()))
            value = self.min_value + range_size * random_factor

        # Clip to bounds and apply precision
        value = self.clip_value(value)

        if self.precision is not None:
            value = value.quantize(Decimal(10) ** -self.precision)

        return value

    def validate_value(self, value: Any) -> bool:
        """Validate if value is within parameter bounds."""
        try:
            decimal_value = Decimal(str(value))
            return self.min_value <= decimal_value <= self.max_value
        except (ValueError, TypeError):
            return False

    def clip_value(self, value: Any) -> Decimal:
        """Clip value to parameter bounds."""
        try:
            decimal_value = Decimal(str(value))
            return max(self.min_value, min(self.max_value, decimal_value))
        except (ValueError, TypeError):
            return self.default_value or self.min_value

    def get_bounds(self) -> tuple[Decimal, Decimal]:
        """Get parameter bounds."""
        return (self.min_value, self.max_value)

    def get_range(self) -> Decimal:
        """Get parameter range."""
        return self.max_value - self.min_value


class DiscreteParameter(ParameterDefinition):
    """
    Discrete parameter definition for integer-valued parameters.

    Supports step sizes and different sampling strategies
    while maintaining integer constraints.
    """

    parameter_type: ParameterType = Field(default=ParameterType.DISCRETE)
    min_value: int = Field(description="Minimum parameter value")
    max_value: int = Field(description="Maximum parameter value")
    step_size: int = Field(default=1, ge=1, description="Step size between valid values")
    default_value: int | None = Field(default=None, description="Default parameter value")

    @field_validator("max_value")
    @classmethod
    def validate_bounds(cls, v, values):
        """Validate that max_value > min_value."""
        if "min_value" in values.data and v <= values.data["min_value"]:
            raise ValueError("max_value must be greater than min_value")
        return v

    @field_validator("default_value")
    @classmethod
    def validate_default(cls, v, values):
        """Validate default value is within bounds and aligned with step size."""
        if v is not None:
            min_val = values.data.get("min_value")
            max_val = values.data.get("max_value")
            step = values.data.get("step_size", 1)

            if min_val is not None and v < min_val:
                raise ValueError("default_value must be >= min_value")
            if max_val is not None and v > max_val:
                raise ValueError("default_value must be <= max_value")
            if min_val is not None and (v - min_val) % step != 0:
                raise ValueError("default_value must be aligned with step_size")
        return v

    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> int:
        """Sample a value from the discrete parameter space."""
        valid_values = list(range(self.min_value, self.max_value + 1, self.step_size))

        if strategy == SamplingStrategy.UNIFORM:
            return random.choice(valid_values)
        elif strategy == SamplingStrategy.GAUSSIAN:
            # Gaussian sampling around center
            center_index = len(valid_values) // 2
            std_dev = len(valid_values) / 6  # 3-sigma rule
            index = int(random.gauss(center_index, std_dev))
            index = max(0, min(len(valid_values) - 1, index))
            return valid_values[index]
        else:
            return random.choice(valid_values)

    def validate_value(self, value: Any) -> bool:
        """Validate if value is a valid discrete parameter value."""
        try:
            int_value = int(value)
            if not (self.min_value <= int_value <= self.max_value):
                return False
            return (int_value - self.min_value) % self.step_size == 0
        except (ValueError, TypeError):
            return False

    def clip_value(self, value: Any) -> int:
        """Clip value to nearest valid discrete value."""
        try:
            int_value = int(value)
            int_value = max(self.min_value, min(self.max_value, int_value))

            # Align to step size
            offset = (int_value - self.min_value) % self.step_size
            if offset != 0:
                # Round to nearest step
                if offset <= self.step_size / 2:
                    int_value -= offset
                else:
                    int_value += self.step_size - offset

            return max(self.min_value, min(self.max_value, int_value))
        except (ValueError, TypeError):
            return self.default_value or self.min_value

    def get_bounds(self) -> tuple[int, int]:
        """Get parameter bounds."""
        return (self.min_value, self.max_value)

    def get_valid_values(self) -> list[int]:
        """Get all valid values for this parameter."""
        return list(range(self.min_value, self.max_value + 1, self.step_size))


class CategoricalParameter(ParameterDefinition):
    """
    Categorical parameter definition for discrete choice parameters.

    Supports any hashable values with optional weights for sampling.
    """

    parameter_type: ParameterType = Field(default=ParameterType.CATEGORICAL)
    choices: list[Any] = Field(description="List of valid parameter choices")
    weights: list[float] | None = Field(
        default=None, description="Sampling weights for choices (must match choices length)"
    )
    default_value: Any | None = Field(default=None, description="Default parameter value")

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v):
        """Validate choices are non-empty and unique."""
        if not v:
            raise ValueError("choices cannot be empty")
        if len(v) != len(set(map(str, v))):  # Convert to string for comparison
            raise ValueError("choices must be unique")
        return v

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v, values):
        """Validate weights match choices length and are positive."""
        if v is not None:
            choices = values.data.get("choices", [])
            if len(v) != len(choices):
                raise ValueError("weights must have same length as choices")
            if any(w <= 0 for w in v):
                raise ValueError("all weights must be positive")
        return v

    @field_validator("default_value")
    @classmethod
    def validate_default(cls, v, values):
        """Validate default value is in choices."""
        if v is not None:
            choices = values.data.get("choices", [])
            if v not in choices:
                raise ValueError("default_value must be one of the choices")
        return v

    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any:
        """Sample a value from the categorical parameter space."""
        if self.weights is not None:
            # Weighted sampling
            return random.choices(self.choices, weights=self.weights, k=1)[0]
        else:
            # Uniform sampling
            return random.choice(self.choices)

    def validate_value(self, value: Any) -> bool:
        """Validate if value is a valid categorical choice."""
        return value in self.choices

    def clip_value(self, value: Any) -> Any:
        """Return closest valid choice or default."""
        if value in self.choices:
            return value
        return self.default_value or self.choices[0]

    def get_bounds(self) -> tuple[Any, Any]:
        """Get parameter bounds (first and last choice)."""
        return (self.choices[0], self.choices[-1])

    def get_choice_index(self, value: Any) -> int:
        """Get index of choice value."""
        try:
            return self.choices.index(value)
        except ValueError:
            return 0


class BooleanParameter(ParameterDefinition):
    """
    Boolean parameter definition for binary choice parameters.

    Specialized categorical parameter for boolean values.
    """

    parameter_type: ParameterType = Field(default=ParameterType.BOOLEAN)
    default_value: bool | None = Field(default=None, description="Default parameter value")
    true_probability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of sampling True"
    )

    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> bool:
        """Sample a boolean value."""
        return random.random() < self.true_probability

    def validate_value(self, value: Any) -> bool:
        """Validate if value is boolean."""
        return isinstance(value, bool)

    def clip_value(self, value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool):
            return value
        if value in (0, "false", "False", "FALSE", "f", "F"):
            return False
        return bool(value)

    def get_bounds(self) -> tuple[bool, bool]:
        """Get parameter bounds."""
        return (False, True)


class ConditionalParameter(ParameterDefinition):
    """
    Conditional parameter that depends on other parameters.

    Enables complex parameter spaces with hierarchical dependencies.
    """

    parameter_type: ParameterType = Field(default=ParameterType.CONDITIONAL)
    base_parameter: ParameterDefinition = Field(description="Base parameter definition")
    activation_conditions: dict[str, Any] = Field(description="Conditions for parameter activation")

    def __init__(self, **data):
        super().__init__(**data)
        self.is_conditional = True
        self.conditions = self.activation_conditions

    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any:
        """Sample from base parameter."""
        return self.base_parameter.sample(strategy)

    def validate_value(self, value: Any) -> bool:
        """Validate using base parameter."""
        return self.base_parameter.validate_value(value)

    def clip_value(self, value: Any) -> Any:
        """Clip using base parameter."""
        return self.base_parameter.clip_value(value)

    def get_bounds(self) -> tuple[Any, Any]:
        """Get bounds from base parameter."""
        return self.base_parameter.get_bounds()


class ParameterSpace(BaseModel):
    """
    Complete parameter space definition.

    Manages a collection of parameters with their relationships,
    dependencies, and sampling strategies.
    """

    parameters: dict[str, ParameterDefinition] = Field(
        description="Dictionary of parameter definitions"
    )
    constraints: list[str] = Field(
        default_factory=list, description="List of parameter constraints (expressions)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Parameter space metadata")

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v):
        """Validate parameter definitions."""
        if not v:
            raise ValueError("At least one parameter must be defined")

        # Check for circular dependencies in conditional parameters
        dependencies = {}
        for name, param in v.items():
            if param.is_conditional:
                dependencies[name] = set(param.conditions.keys())

        # Simple cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in dependencies.get(node, []):
                if has_cycle(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for param_name in dependencies:
            if has_cycle(param_name):
                raise ValueError(f"Circular dependency detected involving parameter: {param_name}")

        return v

    def sample(
        self,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Sample a complete parameter set.

        Args:
            strategy: Sampling strategy to use
            context: Context for conditional parameters

        Returns:
            Dictionary of sampled parameter values
        """
        if context is None:
            context = {}

        sampled = {}

        # Sort parameters to handle dependencies
        sorted_params = self._topological_sort()

        for param_name in sorted_params:
            param = self.parameters[param_name]

            # Check if parameter is active
            if param.is_active({**context, **sampled}):
                value = param.sample(strategy)
                sampled[param_name] = value
            else:
                # Use default value for inactive parameters
                if hasattr(param, "default_value") and param.default_value is not None:
                    sampled[param_name] = param.default_value

        return sampled

    def validate_parameters(self, parameters: dict[str, Any]) -> dict[str, bool]:
        """
        Validate a parameter set.

        Args:
            parameters: Parameter values to validate

        Returns:
            Dictionary of validation results for each parameter
        """
        results = {}

        for param_name, param_def in self.parameters.items():
            if param_name in parameters:
                results[param_name] = param_def.validate_value(parameters[param_name])
            else:
                # Missing parameter
                results[param_name] = False

        return results

    def clip_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Clip parameters to valid ranges.

        Args:
            parameters: Parameter values to clip

        Returns:
            Dictionary of clipped parameter values
        """
        clipped = {}

        for param_name, param_def in self.parameters.items():
            if param_name in parameters:
                clipped[param_name] = param_def.clip_value(parameters[param_name])
            elif hasattr(param_def, "default_value") and param_def.default_value is not None:
                clipped[param_name] = param_def.default_value

        return clipped

    def get_active_parameters(self, context: dict[str, Any]) -> set[str]:
        """
        Get set of active parameters given context.

        Args:
            context: Current parameter context

        Returns:
            Set of active parameter names
        """
        active = set()

        for param_name, param_def in self.parameters.items():
            if param_def.is_active(context):
                active.add(param_name)

        return active

    def get_bounds(self) -> dict[str, tuple[Any, Any]]:
        """
        Get bounds for all parameters.

        Returns:
            Dictionary of parameter bounds
        """
        bounds = {}

        for param_name, param_def in self.parameters.items():
            bounds[param_name] = param_def.get_bounds()

        return bounds

    def _topological_sort(self) -> list[str]:
        """
        Topologically sort parameters to handle dependencies.

        Returns:
            List of parameter names in dependency order
        """
        # Build dependency graph
        dependencies = {}
        for name, param in self.parameters.items():
            dependencies[name] = set()
            if param.is_conditional:
                dependencies[name].update(param.conditions.keys())

        # Kahn's algorithm
        in_degree = {name: 0 for name in self.parameters}
        for name, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for name, deps in dependencies.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        return result

    def get_dimensionality(self) -> int:
        """Get effective dimensionality of parameter space."""
        return len(self.parameters)

    def get_parameter_info(self) -> dict[str, dict[str, Any]]:
        """
        Get comprehensive information about all parameters.

        Returns:
            Dictionary with parameter information
        """
        info = {}

        for param_name, param_def in self.parameters.items():
            info[param_name] = {
                "type": param_def.parameter_type.value,
                "bounds": param_def.get_bounds(),
                "description": param_def.description,
                "is_conditional": param_def.is_conditional,
                "conditions": param_def.conditions if param_def.is_conditional else {},
                "metadata": param_def.metadata,
            }

            # Add type-specific information
            if isinstance(param_def, ContinuousParameter):
                info[param_name].update(
                    {
                        "precision": param_def.precision,
                        "log_scale": param_def.log_scale,
                        "default_value": param_def.default_value,
                    }
                )
            elif isinstance(param_def, DiscreteParameter):
                info[param_name].update(
                    {
                        "step_size": param_def.step_size,
                        "valid_values": param_def.get_valid_values(),
                        "default_value": param_def.default_value,
                    }
                )
            elif isinstance(param_def, CategoricalParameter):
                info[param_name].update(
                    {
                        "choices": param_def.choices,
                        "weights": param_def.weights,
                        "default_value": param_def.default_value,
                    }
                )
            elif isinstance(param_def, BooleanParameter):
                info[param_name].update(
                    {
                        "true_probability": param_def.true_probability,
                        "default_value": param_def.default_value,
                    }
                )

        return info


class ParameterSpaceBuilder:
    """
    Builder class for constructing parameter spaces.

    Provides a fluent interface for building complex parameter spaces
    with proper validation and dependency handling.
    """

    def __init__(self):
        """Initialize parameter space builder."""
        self.parameters: dict[str, ParameterDefinition] = {}
        self.constraints: list[str] = []
        self.metadata: dict[str, Any] = {}

    def add_continuous(
        self,
        name: str,
        min_value: float | Decimal,
        max_value: float | Decimal,
        precision: int | None = None,
        log_scale: bool = False,
        default_value: float | Decimal | None = None,
        description: str = "",
        **kwargs,
    ) -> "ParameterSpaceBuilder":
        """Add a continuous parameter."""
        param = ContinuousParameter(
            name=name,
            min_value=Decimal(str(min_value)),
            max_value=Decimal(str(max_value)),
            precision=precision,
            log_scale=log_scale,
            default_value=Decimal(str(default_value)) if default_value is not None else None,
            description=description,
            **kwargs,
        )
        self.parameters[name] = param
        return self

    def add_discrete(
        self,
        name: str,
        min_value: int,
        max_value: int,
        step_size: int = 1,
        default_value: int | None = None,
        description: str = "",
        **kwargs,
    ) -> "ParameterSpaceBuilder":
        """Add a discrete parameter."""
        param = DiscreteParameter(
            name=name,
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            default_value=default_value,
            description=description,
            **kwargs,
        )
        self.parameters[name] = param
        return self

    def add_categorical(
        self,
        name: str,
        choices: list[Any],
        weights: list[float] | None = None,
        default_value: Any | None = None,
        description: str = "",
        **kwargs,
    ) -> "ParameterSpaceBuilder":
        """Add a categorical parameter."""
        param = CategoricalParameter(
            name=name,
            choices=choices,
            weights=weights,
            default_value=default_value,
            description=description,
            **kwargs,
        )
        self.parameters[name] = param
        return self

    def add_boolean(
        self,
        name: str,
        true_probability: float = 0.5,
        default_value: bool | None = None,
        description: str = "",
        **kwargs,
    ) -> "ParameterSpaceBuilder":
        """Add a boolean parameter."""
        param = BooleanParameter(
            name=name,
            true_probability=true_probability,
            default_value=default_value,
            description=description,
            **kwargs,
        )
        self.parameters[name] = param
        return self

    def add_conditional(
        self,
        name: str,
        base_parameter: ParameterDefinition,
        conditions: dict[str, Any],
        description: str = "",
        **kwargs,
    ) -> "ParameterSpaceBuilder":
        """Add a conditional parameter."""
        param = ConditionalParameter(
            name=name,
            base_parameter=base_parameter,
            activation_conditions=conditions,
            description=description,
            **kwargs,
        )
        self.parameters[name] = param
        return self

    def add_constraint(self, constraint: str) -> "ParameterSpaceBuilder":
        """Add a parameter constraint."""
        self.constraints.append(constraint)
        return self

    def set_metadata(self, key: str, value: Any) -> "ParameterSpaceBuilder":
        """Set metadata for the parameter space."""
        self.metadata[key] = value
        return self

    def build(self) -> ParameterSpace:
        """Build the parameter space."""
        return ParameterSpace(
            parameters=self.parameters, constraints=self.constraints, metadata=self.metadata
        )


# Factory functions for common trading parameter patterns


def create_trading_strategy_space() -> ParameterSpace:
    """Create a standard trading strategy parameter space."""
    builder = ParameterSpaceBuilder()

    # Position sizing parameters
    builder.add_continuous(
        "position_size_pct",
        min_value=0.01,
        max_value=0.10,
        precision=4,
        default_value=0.02,
        description="Position size as percentage of portfolio",
    )

    # Risk management parameters
    builder.add_continuous(
        "stop_loss_pct",
        min_value=0.005,
        max_value=0.05,
        precision=4,
        default_value=0.02,
        description="Stop loss percentage",
    )

    builder.add_continuous(
        "take_profit_pct",
        min_value=0.01,
        max_value=0.10,
        precision=4,
        default_value=0.04,
        description="Take profit percentage",
    )

    # Timeframe selection
    builder.add_categorical(
        "timeframe",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        default_value="1h",
        description="Trading timeframe",
    )

    # Strategy-specific parameters
    builder.add_discrete(
        "lookback_period",
        min_value=5,
        max_value=50,
        default_value=20,
        description="Lookback period for indicators",
    )

    builder.add_continuous(
        "confidence_threshold",
        min_value=0.5,
        max_value=0.9,
        precision=3,
        default_value=0.7,
        description="Minimum confidence threshold for signals",
    )

    return builder.build()


def create_ml_model_space() -> ParameterSpace:
    """Create a parameter space for ML model hyperparameters."""
    builder = ParameterSpaceBuilder()

    # Model architecture
    builder.add_categorical(
        "model_type",
        choices=["random_forest", "xgboost", "neural_network", "svm"],
        default_value="random_forest",
        description="Machine learning model type",
    )

    # Random Forest parameters (conditional)
    rf_n_estimators = DiscreteParameter(
        name="rf_n_estimators",
        min_value=50,
        max_value=500,
        step_size=50,
        default_value=100,
        description="Number of trees in random forest",
    )

    builder.add_conditional(
        "rf_n_estimators",
        rf_n_estimators,
        {"model_type": "random_forest"},
        description="Random forest: number of estimators",
    )

    rf_max_depth = DiscreteParameter(
        name="rf_max_depth",
        min_value=3,
        max_value=20,
        default_value=10,
        description="Maximum depth of trees",
    )

    builder.add_conditional(
        "rf_max_depth",
        rf_max_depth,
        {"model_type": "random_forest"},
        description="Random forest: maximum tree depth",
    )

    # XGBoost parameters (conditional)
    xgb_learning_rate = ContinuousParameter(
        name="xgb_learning_rate",
        min_value=Decimal("0.01"),
        max_value=Decimal("0.3"),
        precision=3,
        log_scale=True,
        default_value=Decimal("0.1"),
        description="XGBoost learning rate",
    )

    builder.add_conditional(
        "xgb_learning_rate",
        xgb_learning_rate,
        {"model_type": "xgboost"},
        description="XGBoost: learning rate",
    )

    # General parameters
    builder.add_continuous(
        "validation_split",
        min_value=0.1,
        max_value=0.3,
        precision=2,
        default_value=0.2,
        description="Validation split ratio",
    )

    return builder.build()


def create_risk_management_space() -> ParameterSpace:
    """Create parameter space for risk management settings."""
    builder = ParameterSpaceBuilder()

    # Portfolio limits
    builder.add_continuous(
        "max_portfolio_exposure",
        min_value=0.5,
        max_value=0.95,
        precision=3,
        default_value=0.8,
        description="Maximum portfolio exposure",
    )

    builder.add_discrete(
        "max_positions",
        min_value=1,
        max_value=20,
        default_value=5,
        description="Maximum number of positions",
    )

    # Risk metrics
    builder.add_continuous(
        "max_drawdown_limit",
        min_value=0.05,
        max_value=0.25,
        precision=3,
        default_value=0.15,
        description="Maximum allowed drawdown",
    )

    builder.add_continuous(
        "var_confidence_level",
        min_value=0.9,
        max_value=0.99,
        precision=3,
        default_value=0.95,
        description="VaR confidence level",
    )

    # Circuit breakers
    builder.add_boolean(
        "enable_correlation_breaker",
        true_probability=0.8,
        default_value=True,
        description="Enable correlation-based circuit breaker",
    )

    builder.add_continuous(
        "correlation_threshold",
        min_value=0.7,
        max_value=0.95,
        precision=3,
        default_value=0.85,
        description="Correlation threshold for circuit breaker",
    )

    return builder.build()
