"""
Unit tests for optimization core module.

Tests the foundational types and base classes for the optimization
framework, ensuring type validation and proper behavior.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import patch

from src.optimization.core import (
    OptimizationStatus,
    ObjectiveDirection,
    OptimizationObjective,
    OptimizationConstraint,
    OptimizationConfig,
    OptimizationResult,
    OptimizationEngine,
)
from src.optimization.parameter_space import ParameterType
from src.core.exceptions import ValidationError
from src.core.types import TradingMode


@pytest.fixture(scope="module")
def sample_objective():
    """Create sample optimization objective."""
    return OptimizationObjective(
        name="return",
        direction=ObjectiveDirection.MAXIMIZE,
        weight=Decimal("1.0"),
    )


@pytest.fixture(scope="module") 
def sample_config():
    """Create sample optimization config."""
    return OptimizationConfig(
        max_iterations=100,
        max_evaluations=1000,
        timeout_seconds=300,
        convergence_tolerance=Decimal("1e-4"),
    )


class TestEnumerations:
    """Test optimization enumeration types."""

    def test_optimization_status_enum(self):
        """Test OptimizationStatus enum values."""
        assert OptimizationStatus.PENDING.value == "pending"
        assert OptimizationStatus.INITIALIZING.value == "initializing"
        assert OptimizationStatus.RUNNING.value == "running"
        assert OptimizationStatus.PAUSED.value == "paused"
        assert OptimizationStatus.COMPLETED.value == "completed"
        assert OptimizationStatus.FAILED.value == "failed"
        assert OptimizationStatus.CANCELLED.value == "cancelled"

    def test_objective_direction_enum(self):
        """Test ObjectiveDirection enum values."""
        assert ObjectiveDirection.MAXIMIZE.value == "maximize"
        assert ObjectiveDirection.MINIMIZE.value == "minimize"

    def test_parameter_type_enum(self):
        """Test ParameterType enum values."""
        assert ParameterType.CONTINUOUS.value == "continuous"
        assert ParameterType.DISCRETE.value == "discrete"
        assert ParameterType.CATEGORICAL.value == "categorical"
        assert ParameterType.BOOLEAN.value == "boolean"


class TestOptimizationObjective:
    """Test OptimizationObjective class."""

    def test_objective_creation(self):
        """Test basic objective creation."""
        objective = OptimizationObjective(
            name="return",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=Decimal("1.0"),
        )
        
        assert objective.name == "return"
        assert objective.direction == ObjectiveDirection.MAXIMIZE
        assert objective.weight == Decimal("1.0")
        assert objective.target_value is None
        assert objective.constraint_min is None
        assert objective.constraint_max is None
        assert objective.description == ""
        assert objective.is_primary is False

    def test_objective_with_all_fields(self):
        """Test objective with all fields specified."""
        objective = OptimizationObjective(
            name="sharpe_ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=Decimal("2.5"),
            target_value=Decimal("1.5"),
            constraint_min=Decimal("0.0"),
            constraint_max=Decimal("5.0"),
            description="Risk-adjusted return",
            is_primary=True,
        )
        
        assert objective.name == "sharpe_ratio"
        assert objective.direction == ObjectiveDirection.MAXIMIZE
        assert objective.weight == Decimal("2.5")
        assert objective.target_value == Decimal("1.5")
        assert objective.constraint_min == Decimal("0.0")
        assert objective.constraint_max == Decimal("5.0")
        assert objective.description == "Risk-adjusted return"
        assert objective.is_primary is True

    def test_objective_weight_validation(self):
        """Test weight validation."""
        # Valid weight
        objective = OptimizationObjective(
            name="return",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=Decimal("0.0"),
        )
        assert objective.weight == Decimal("0.0")
        
        # Invalid negative weight
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            OptimizationObjective(
                name="return",
                direction=ObjectiveDirection.MAXIMIZE,
                weight=Decimal("-1.0"),
            )

    def test_is_better_method(self):
        """Test is_better comparison method."""
        # Maximize objective
        max_obj = OptimizationObjective(
            name="return",
            direction=ObjectiveDirection.MAXIMIZE,
        )
        assert max_obj.is_better(Decimal("2.0"), Decimal("1.0")) is True
        assert max_obj.is_better(Decimal("1.0"), Decimal("2.0")) is False
        assert max_obj.is_better(Decimal("1.0"), Decimal("1.0")) is False
        
        # Minimize objective
        min_obj = OptimizationObjective(
            name="risk",
            direction=ObjectiveDirection.MINIMIZE,
        )
        assert min_obj.is_better(Decimal("1.0"), Decimal("2.0")) is True
        assert min_obj.is_better(Decimal("2.0"), Decimal("1.0")) is False
        assert min_obj.is_better(Decimal("1.0"), Decimal("1.0")) is False

    def test_satisfies_constraints_method(self):
        """Test constraint satisfaction method."""
        objective = OptimizationObjective(
            name="return",
            direction=ObjectiveDirection.MAXIMIZE,
            constraint_min=Decimal("0.0"),
            constraint_max=Decimal("10.0"),
        )
        
        # Within constraints
        assert objective.satisfies_constraints(Decimal("5.0")) is True
        assert objective.satisfies_constraints(Decimal("0.0")) is True
        assert objective.satisfies_constraints(Decimal("10.0")) is True
        
        # Outside constraints
        assert objective.satisfies_constraints(Decimal("-1.0")) is False
        assert objective.satisfies_constraints(Decimal("11.0")) is False
        
        # No constraints
        no_constraint_obj = OptimizationObjective(
            name="return",
            direction=ObjectiveDirection.MAXIMIZE,
        )
        assert no_constraint_obj.satisfies_constraints(Decimal("-100.0")) is True
        assert no_constraint_obj.satisfies_constraints(Decimal("100.0")) is True

    def test_distance_to_target_method(self):
        """Test distance to target calculation."""
        objective = OptimizationObjective(
            name="return",
            direction=ObjectiveDirection.MAXIMIZE,
            target_value=Decimal("5.0"),
        )
        
        assert objective.distance_to_target(Decimal("5.0")) == Decimal("0.0")
        assert objective.distance_to_target(Decimal("3.0")) == Decimal("2.0")
        assert objective.distance_to_target(Decimal("7.0")) == Decimal("2.0")

        # Test without target value
        no_target_obj = OptimizationObjective(
            name="return",
            direction=ObjectiveDirection.MAXIMIZE,
        )
        assert no_target_obj.distance_to_target(Decimal("100.0")) == Decimal("0.0")
        assert no_target_obj.distance_to_target(Decimal("-100.0")) == Decimal("0.0")


class TestOptimizationConstraint:
    """Test OptimizationConstraint class."""

    def test_constraint_creation(self):
        """Test basic constraint creation."""
        constraint = OptimizationConstraint(
            name="max_drawdown",
            expression="drawdown <= 0.1",
            constraint_type="inequality",
        )
        
        assert constraint.name == "max_drawdown"
        assert constraint.expression == "drawdown <= 0.1"
        assert constraint.constraint_type == "inequality"
        assert constraint.tolerance == Decimal("1e-6")
        assert constraint.penalty_weight == Decimal("1000")
        assert constraint.is_hard is True

    def test_constraint_with_custom_values(self):
        """Test constraint with custom values."""
        constraint = OptimizationConstraint(
            name="sharpe_ratio",
            expression="sharpe_ratio >= 0.5",
            constraint_type="inequality",
            tolerance=Decimal("0.01"),
            penalty_weight=Decimal("500"),
            is_hard=False,
            description="Minimum Sharpe ratio constraint"
        )
        
        assert constraint.name == "sharpe_ratio"
        assert constraint.expression == "sharpe_ratio >= 0.5"
        assert constraint.tolerance == Decimal("0.01")
        assert constraint.penalty_weight == Decimal("500")
        assert constraint.is_hard is False
        assert constraint.description == "Minimum Sharpe ratio constraint"

    def test_constraint_description_auto_generation(self):
        """Test automatic description generation."""
        constraint = OptimizationConstraint(
            name="position_limit",
            expression="position_size <= 0.02",
        )
        
        # Should auto-generate description from expression
        assert constraint.description == "Constraint: position_size <= 0.02"


class TestOptimizationConfig:
    """Test OptimizationConfig class."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert config.max_iterations == 1000
        assert config.max_evaluations == 10000
        assert config.timeout_seconds is None
        assert config.convergence_tolerance == Decimal("1e-6")
        assert config.max_stagnation_iterations == 50
        assert config.parallel_evaluations is True
        assert config.trading_mode == TradingMode.BACKTEST

    def test_config_with_values(self):
        """Test configuration with custom values."""
        config = OptimizationConfig(
            max_iterations=500,
            max_evaluations=5000,
            timeout_seconds=3600,
            convergence_tolerance=Decimal("1e-4"),
            max_stagnation_iterations=25,
            parallel_evaluations=False,
            trading_mode=TradingMode.PAPER,
        )
        
        assert config.max_iterations == 500
        assert config.max_evaluations == 5000
        assert config.timeout_seconds == 3600
        assert config.convergence_tolerance == Decimal("1e-4")
        assert config.max_stagnation_iterations == 25
        assert config.parallel_evaluations is False
        assert config.trading_mode == TradingMode.PAPER

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configurations
        OptimizationConfig(max_iterations=1)
        OptimizationConfig(max_evaluations=1)
        OptimizationConfig(timeout_seconds=1)
        OptimizationConfig(convergence_tolerance=Decimal("1e-10"))
        
        # Invalid configurations
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            OptimizationConfig(max_iterations=0)
        
        with pytest.raises(PydanticValidationError):
            OptimizationConfig(max_evaluations=0)
            
        with pytest.raises(PydanticValidationError):
            OptimizationConfig(timeout_seconds=0)
            
        with pytest.raises(PydanticValidationError):
            OptimizationConfig(convergence_tolerance=Decimal("0"))


class TestOptimizationResult:
    """Test OptimizationResult class."""

    def test_result_creation(self):
        """Test optimization result creation."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        result = OptimizationResult(
            optimization_id="test_123",
            algorithm_name="BruteForceOptimizer",
            optimal_parameters={"param1": Decimal("1.0")},
            optimal_objective_value=Decimal("0.85"),
            objective_values={"return": Decimal("0.85")},
            iterations_completed=10,
            evaluations_completed=50,
            convergence_achieved=True,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=Decimal("120.5"),
            config_used={},
        )
        
        assert result.optimization_id == "test_123"
        assert result.algorithm_name == "BruteForceOptimizer"
        assert result.optimal_parameters == {"param1": Decimal("1.0")}
        assert result.optimal_objective_value == Decimal("0.85")
        assert result.objective_values == {"return": Decimal("0.85")}

    def test_result_defaults(self):
        """Test result default values."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        result = OptimizationResult(
            optimization_id="test_123",
            algorithm_name="TestOptimizer",
            optimal_parameters={},
            optimal_objective_value=Decimal("0.0"),
            objective_values={},
            iterations_completed=0,
            evaluations_completed=0,
            convergence_achieved=False,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=Decimal("0.0"),
            config_used={},
        )
        
        assert result.validation_score is None
        assert result.overfitting_score is None
        assert result.robustness_score is None


class TestFinancialEdgeCases:
    """Test financial-specific edge cases."""

    def test_decimal_precision_preservation(self):
        """Test that decimal precision is preserved."""
        high_precision_value = Decimal("1.123456789012345678901234567890")
        
        objective = OptimizationObjective(
            name="return",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=high_precision_value,
            target_value=high_precision_value,
            constraint_min=high_precision_value,
            constraint_max=high_precision_value * 2,
        )
        
        assert objective.weight == high_precision_value
        assert objective.target_value == high_precision_value
        assert objective.constraint_min == high_precision_value
        assert objective.constraint_max == high_precision_value * 2

    def test_financial_constraint_validation(self):
        """Test financial constraint validation."""
        # Maximum drawdown constraint
        drawdown_obj = OptimizationObjective(
            name="max_drawdown",
            direction=ObjectiveDirection.MINIMIZE,
            constraint_max=Decimal("0.20"),  # 20% max drawdown
        )
        
        assert drawdown_obj.satisfies_constraints(Decimal("0.15")) is True
        assert drawdown_obj.satisfies_constraints(Decimal("0.25")) is False
        
        # Sharpe ratio constraint
        sharpe_obj = OptimizationObjective(
            name="sharpe_ratio",
            direction=ObjectiveDirection.MAXIMIZE,
            constraint_min=Decimal("0.5"),  # Minimum Sharpe ratio
        )
        
        assert sharpe_obj.satisfies_constraints(Decimal("0.8")) is True
        assert sharpe_obj.satisfies_constraints(Decimal("0.3")) is False

    def test_risk_parameter_bounds(self):
        """Test risk parameter boundary validation."""
        # Position sizing should be constrained
        position_obj = OptimizationObjective(
            name="position_size",
            direction=ObjectiveDirection.MAXIMIZE,
            constraint_min=Decimal("0.0"),  # No negative positions
            constraint_max=Decimal("0.02"),  # Max 2% per trade
        )
        
        assert position_obj.satisfies_constraints(Decimal("0.01")) is True
        assert position_obj.satisfies_constraints(Decimal("0.03")) is False
        assert position_obj.satisfies_constraints(Decimal("-0.01")) is False