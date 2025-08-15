"""
Core optimization types and base classes.

This module provides the foundational types and base classes for the optimization
framework, ensuring consistent interfaces across all optimization algorithms.

Key Features:
- Base optimization engine interface
- Standard objective and constraint definitions
- Status tracking and progress monitoring
- Configuration management
- Error handling and logging integration

Critical for Financial Applications:
- Decimal precision preservation
- Proper error handling for trading systems
- Comprehensive logging for audit trails
- Type safety and validation
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

from pydantic import BaseModel, Field, field_validator

from src.core.exceptions import OptimizationError, ValidationError
from src.core.logging import get_logger
from src.core.types import TradingMode

logger = get_logger(__name__)


class OptimizationStatus(Enum):
    """Status enumeration for optimization processes."""
    
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ObjectiveDirection(Enum):
    """Direction for optimization objectives."""
    
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ParameterType(Enum):
    """Parameter type enumeration."""
    
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class OptimizationObjective(BaseModel):
    """
    Optimization objective definition.
    
    Defines a single objective to optimize with its direction, weight,
    and constraints. Used across all optimization algorithms.
    """
    
    name: str = Field(description="Objective name")
    direction: ObjectiveDirection = Field(description="Optimization direction")
    weight: Decimal = Field(
        default=Decimal("1.0"), 
        ge=Decimal("0"), 
        description="Objective weight in multi-objective optimization"
    )
    target_value: Optional[Decimal] = Field(
        default=None, 
        description="Target value for objective (if applicable)"
    )
    constraint_min: Optional[Decimal] = Field(
        default=None, 
        description="Minimum constraint value"
    )
    constraint_max: Optional[Decimal] = Field(
        default=None, 
        description="Maximum constraint value"
    )
    description: str = Field(
        default="", 
        description="Objective description"
    )
    is_primary: bool = Field(
        default=False, 
        description="Whether this is the primary objective"
    )
    
    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: Decimal) -> Decimal:
        """Validate weight is non-negative."""
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v
    
    @field_validator("constraint_min", "constraint_max")
    @classmethod
    def validate_constraints(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate constraint values."""
        # Constraints can be any value, validation happens during optimization
        return v
    
    def is_better(self, value1: Decimal, value2: Decimal) -> bool:
        """
        Check if value1 is better than value2 for this objective.
        
        Args:
            value1: First value to compare
            value2: Second value to compare
            
        Returns:
            True if value1 is better than value2
        """
        if self.direction == ObjectiveDirection.MAXIMIZE:
            return value1 > value2
        else:
            return value1 < value2
    
    def satisfies_constraints(self, value: Decimal) -> bool:
        """
        Check if value satisfies this objective's constraints.
        
        Args:
            value: Value to check
            
        Returns:
            True if value satisfies constraints
        """
        if self.constraint_min is not None and value < self.constraint_min:
            return False
        if self.constraint_max is not None and value > self.constraint_max:
            return False
        return True
    
    def distance_to_target(self, value: Decimal) -> Decimal:
        """
        Calculate distance to target value.
        
        Args:
            value: Current value
            
        Returns:
            Distance to target (0 if no target set)
        """
        if self.target_value is None:
            return Decimal("0")
        return abs(value - self.target_value)


class OptimizationConstraint(BaseModel):
    """
    Optimization constraint definition.
    
    Defines constraints that must be satisfied by optimization solutions.
    Separate from objective constraints for clarity and flexibility.
    """
    
    name: str = Field(description="Constraint name")
    expression: str = Field(description="Constraint expression or description")
    constraint_type: str = Field(
        default="inequality", 
        description="Type of constraint (equality, inequality, etc.)"
    )
    tolerance: Decimal = Field(
        default=Decimal("1e-6"), 
        ge=Decimal("0"), 
        description="Constraint violation tolerance"
    )
    penalty_weight: Decimal = Field(
        default=Decimal("1000"), 
        ge=Decimal("0"), 
        description="Penalty weight for constraint violations"
    )
    is_hard: bool = Field(
        default=True, 
        description="Whether this is a hard constraint (vs soft constraint)"
    )
    description: str = Field(
        default="", 
        description="Constraint description"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.description:
            self.description = f"Constraint: {self.expression}"


class OptimizationProgress(BaseModel):
    """
    Progress tracking for optimization processes.
    
    Provides detailed progress information including current status,
    completion percentage, and performance metrics.
    """
    
    optimization_id: str = Field(description="Optimization identifier")
    status: OptimizationStatus = Field(description="Current status")
    current_iteration: int = Field(default=0, ge=0, description="Current iteration")
    total_iterations: int = Field(default=0, ge=0, description="Total iterations")
    completion_percentage: Decimal = Field(
        default=Decimal("0"), 
        ge=Decimal("0"), 
        le=Decimal("100"), 
        description="Completion percentage"
    )
    
    # Performance metrics
    best_objective_value: Optional[Decimal] = Field(
        default=None, 
        description="Best objective value found so far"
    )
    current_objective_value: Optional[Decimal] = Field(
        default=None, 
        description="Current objective value"
    )
    evaluations_completed: int = Field(
        default=0, 
        ge=0, 
        description="Number of evaluations completed"
    )
    
    # Timing information
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Optimization start time"
    )
    last_update_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last progress update time"
    )
    estimated_completion_time: Optional[datetime] = Field(
        default=None, 
        description="Estimated completion time"
    )
    
    # Additional information
    current_parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Current best parameters"
    )
    message: str = Field(
        default="", 
        description="Current status message"
    )
    warnings: List[str] = Field(
        default_factory=list, 
        description="List of warnings encountered"
    )
    
    def update_progress(
        self, 
        iteration: int, 
        objective_value: Optional[Decimal] = None,
        parameters: Optional[Dict[str, Any]] = None,
        message: str = ""
    ) -> None:
        """
        Update progress information.
        
        Args:
            iteration: Current iteration number
            objective_value: Current objective value
            parameters: Current parameters
            message: Status message
        """
        self.current_iteration = iteration
        self.last_update_time = datetime.now(timezone.utc)
        
        if self.total_iterations > 0:
            self.completion_percentage = Decimal(str(
                (iteration / self.total_iterations) * 100
            ))
        
        if objective_value is not None:
            self.current_objective_value = objective_value
            if (self.best_objective_value is None or 
                objective_value > self.best_objective_value):
                self.best_objective_value = objective_value
        
        if parameters is not None:
            self.current_parameters = parameters
        
        if message:
            self.message = message
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the progress tracker."""
        self.warnings.append(warning)
        logger.warning(f"Optimization warning: {warning}", optimization_id=self.optimization_id)
    
    def estimate_completion_time(self) -> None:
        """Estimate completion time based on current progress."""
        if self.completion_percentage <= 0:
            return
        
        elapsed_time = (
            datetime.now(timezone.utc) - self.start_time
        ).total_seconds()
        
        estimated_total_time = elapsed_time / (float(self.completion_percentage) / 100)
        remaining_time = estimated_total_time - elapsed_time
        
        self.estimated_completion_time = (
            datetime.now(timezone.utc) + 
            timedelta(seconds=remaining_time)
        )


class OptimizationConfig(BaseModel):
    """
    Base configuration for optimization algorithms.
    
    Provides common configuration options that apply across different
    optimization algorithms.
    """
    
    # Basic configuration
    max_iterations: int = Field(
        default=1000, 
        ge=1, 
        description="Maximum number of iterations"
    )
    max_evaluations: int = Field(
        default=10000, 
        ge=1, 
        description="Maximum number of function evaluations"
    )
    timeout_seconds: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Maximum optimization time in seconds"
    )
    
    # Convergence criteria
    convergence_tolerance: Decimal = Field(
        default=Decimal("1e-6"), 
        gt=Decimal("0"), 
        description="Convergence tolerance"
    )
    max_stagnation_iterations: int = Field(
        default=50, 
        ge=1, 
        description="Maximum iterations without improvement"
    )
    
    # Parallel processing
    parallel_evaluations: bool = Field(
        default=True, 
        description="Enable parallel evaluation of candidates"
    )
    max_workers: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Maximum number of parallel workers"
    )
    
    # Validation and robustness
    validation_enabled: bool = Field(
        default=True, 
        description="Enable validation during optimization"
    )
    cross_validation_folds: int = Field(
        default=5, 
        ge=2, 
        description="Number of cross-validation folds"
    )
    
    # Logging and monitoring
    log_level: str = Field(
        default="INFO", 
        description="Logging level for optimization"
    )
    progress_callback_interval: int = Field(
        default=10, 
        ge=1, 
        description="Interval for progress callbacks"
    )
    save_intermediate_results: bool = Field(
        default=True, 
        description="Save intermediate optimization results"
    )
    
    # Resource management
    memory_limit_mb: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Memory limit in megabytes"
    )
    disk_space_limit_mb: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Disk space limit in megabytes"
    )
    
    # Trading-specific settings
    trading_mode: TradingMode = Field(
        default=TradingMode.BACKTEST, 
        description="Trading mode for optimization"
    )
    preserve_decimal_precision: bool = Field(
        default=True, 
        description="Preserve decimal precision in calculations"
    )
    risk_free_rate: Decimal = Field(
        default=Decimal("0.02"), 
        ge=Decimal("0"), 
        description="Risk-free rate for performance calculations"
    )


class OptimizationResult(BaseModel):
    """
    Result of an optimization process.
    
    Contains the optimal parameters found, objective values achieved,
    and comprehensive metadata about the optimization process.
    """
    
    # Identification
    optimization_id: str = Field(description="Optimization identifier")
    algorithm_name: str = Field(description="Optimization algorithm used")
    
    # Results
    optimal_parameters: Dict[str, Any] = Field(
        description="Optimal parameters found"
    )
    optimal_objective_value: Decimal = Field(
        description="Optimal objective value achieved"
    )
    objective_values: Dict[str, Decimal] = Field(
        description="All objective values at optimal point"
    )
    
    # Validation results
    validation_score: Optional[Decimal] = Field(
        default=None, 
        description="Validation score (out-of-sample performance)"
    )
    overfitting_score: Optional[Decimal] = Field(
        default=None, 
        description="Overfitting detection score"
    )
    robustness_score: Optional[Decimal] = Field(
        default=None, 
        description="Robustness analysis score"
    )
    
    # Optimization statistics
    iterations_completed: int = Field(
        ge=0, 
        description="Number of iterations completed"
    )
    evaluations_completed: int = Field(
        ge=0, 
        description="Number of function evaluations completed"
    )
    convergence_achieved: bool = Field(
        description="Whether convergence was achieved"
    )
    
    # Timing information
    start_time: datetime = Field(description="Optimization start time")
    end_time: datetime = Field(description="Optimization end time")
    total_duration_seconds: Decimal = Field(
        ge=Decimal("0"), 
        description="Total optimization duration in seconds"
    )
    
    # Quality metrics
    parameter_stability: Dict[str, Decimal] = Field(
        default_factory=dict, 
        description="Parameter stability scores"
    )
    sensitivity_analysis: Dict[str, Decimal] = Field(
        default_factory=dict, 
        description="Parameter sensitivity analysis"
    )
    
    # Additional metadata
    warnings: List[str] = Field(
        default_factory=list, 
        description="Warnings encountered during optimization"
    )
    config_used: Dict[str, Any] = Field(
        description="Configuration used for optimization"
    )
    
    # Statistical significance
    statistical_significance: Optional[Decimal] = Field(
        default=None, 
        description="Statistical significance of results"
    )
    confidence_interval: Optional[Tuple[Decimal, Decimal]] = Field(
        default=None, 
        description="Confidence interval for optimal value"
    )
    
    def is_statistically_significant(
        self, 
        significance_level: Decimal = Decimal("0.05")
    ) -> bool:
        """
        Check if the optimization result is statistically significant.
        
        Args:
            significance_level: Required significance level
            
        Returns:
            True if result is statistically significant
        """
        if self.statistical_significance is None:
            return False
        return self.statistical_significance < significance_level
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization results.
        
        Returns:
            Dictionary containing result summary
        """
        return {
            "optimization_id": self.optimization_id,
            "algorithm": self.algorithm_name,
            "optimal_objective": float(self.optimal_objective_value),
            "converged": self.convergence_achieved,
            "iterations": self.iterations_completed,
            "evaluations": self.evaluations_completed,
            "duration_seconds": float(self.total_duration_seconds),
            "validation_score": float(self.validation_score) if self.validation_score else None,
            "statistically_significant": self.is_statistically_significant(),
            "warnings_count": len(self.warnings)
        }


class OptimizationEngine(ABC):
    """
    Abstract base class for optimization engines.
    
    Defines the interface that all optimization algorithms must implement,
    ensuring consistency across different optimization approaches.
    """
    
    def __init__(
        self, 
        objectives: List[OptimizationObjective],
        constraints: Optional[List[OptimizationConstraint]] = None,
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize optimization engine.
        
        Args:
            objectives: List of optimization objectives
            constraints: List of optimization constraints
            config: Optimization configuration
        """
        self.objectives = objectives
        self.constraints = constraints or []
        self.config = config or OptimizationConfig()
        
        # Generate unique optimization ID
        self.optimization_id = str(uuid.uuid4())
        
        # Initialize progress tracking
        self.progress = OptimizationProgress(
            optimization_id=self.optimization_id,
            status=OptimizationStatus.PENDING
        )
        
        # Initialize result storage
        self.result: Optional[OptimizationResult] = None
        
        # Validation
        self._validate_configuration()
        
        logger.info(
            "Optimization engine initialized",
            optimization_id=self.optimization_id,
            engine_type=self.__class__.__name__,
            objective_count=len(objectives),
            constraint_count=len(constraints)
        )
    
    def _validate_configuration(self) -> None:
        """Validate the optimization configuration."""
        if not self.objectives:
            raise ValidationError("At least one objective must be specified")
        
        # Check for duplicate objective names
        objective_names = [obj.name for obj in self.objectives]
        if len(objective_names) != len(set(objective_names)):
            raise ValidationError("Duplicate objective names found")
        
        # Check for duplicate constraint names
        if self.constraints:
            constraint_names = [const.name for const in self.constraints]
            if len(constraint_names) != len(set(constraint_names)):
                raise ValidationError("Duplicate constraint names found")
        
        # Validate configuration values
        if self.config.max_iterations <= 0:
            raise ValidationError("max_iterations must be positive")
        
        if self.config.max_evaluations <= 0:
            raise ValidationError("max_evaluations must be positive")
    
    @abstractmethod
    async def optimize(
        self, 
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        initial_parameters: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Run the optimization algorithm.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Definition of parameter space
            initial_parameters: Initial parameter values (optional)
            
        Returns:
            Optimization result
        """
        pass
    
    @abstractmethod
    def get_next_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Get the next set of parameters to evaluate.
        
        Returns:
            Next parameter set or None if optimization is complete
        """
        pass
    
    def update_progress(
        self, 
        iteration: int,
        objective_value: Optional[Decimal] = None,
        parameters: Optional[Dict[str, Any]] = None,
        message: str = ""
    ) -> None:
        """
        Update optimization progress.
        
        Args:
            iteration: Current iteration
            objective_value: Current objective value
            parameters: Current parameters
            message: Status message
        """
        self.progress.update_progress(iteration, objective_value, parameters, message)
        
        # Log progress periodically
        if iteration % self.config.progress_callback_interval == 0:
            logger.info(
                "Optimization progress update",
                optimization_id=self.optimization_id,
                iteration=iteration,
                completion_pct=float(self.progress.completion_percentage),
                best_value=float(self.progress.best_objective_value) if self.progress.best_objective_value else None
            )
    
    def check_convergence(self, recent_values: List[Decimal]) -> bool:
        """
        Check if optimization has converged.
        
        Args:
            recent_values: Recent objective values
            
        Returns:
            True if converged
        """
        if len(recent_values) < 2:
            return False
        
        # Check if improvement is below tolerance
        recent_improvement = abs(recent_values[-1] - recent_values[-2])
        return recent_improvement < self.config.convergence_tolerance
    
    def evaluate_constraints(self, parameters: Dict[str, Any]) -> Dict[str, Decimal]:
        """
        Evaluate constraint violations for given parameters.
        
        Args:
            parameters: Parameters to evaluate
            
        Returns:
            Dictionary of constraint violations
        """
        violations = {}
        
        for constraint in self.constraints:
            # This is a simplified constraint evaluation
            # In practice, this would evaluate the actual constraint expression
            violation = Decimal("0")  # Placeholder
            violations[constraint.name] = violation
        
        return violations
    
    def is_feasible(self, parameters: Dict[str, Any]) -> bool:
        """
        Check if parameters satisfy all constraints.
        
        Args:
            parameters: Parameters to check
            
        Returns:
            True if parameters are feasible
        """
        violations = self.evaluate_constraints(parameters)
        total_violation = sum(violations.values())
        return total_violation <= self.config.convergence_tolerance
    
    def get_progress(self) -> OptimizationProgress:
        """Get current optimization progress."""
        return self.progress
    
    def get_result(self) -> Optional[OptimizationResult]:
        """Get optimization result (None if not completed)."""
        return self.result
    
    async def stop(self) -> None:
        """Stop the optimization process."""
        self.progress.status = OptimizationStatus.CANCELLED
        logger.info(
            "Optimization stopped",
            optimization_id=self.optimization_id
        )


# Utility functions for common optimization patterns

def create_profit_maximization_objective() -> OptimizationObjective:
    """Create a standard profit maximization objective."""
    return OptimizationObjective(
        name="total_return",
        direction=ObjectiveDirection.MAXIMIZE,
        weight=Decimal("1.0"),
        constraint_min=Decimal("0"),
        description="Maximize total portfolio return",
        is_primary=True
    )


def create_risk_minimization_objective() -> OptimizationObjective:
    """Create a standard risk minimization objective."""
    return OptimizationObjective(
        name="max_drawdown",
        direction=ObjectiveDirection.MINIMIZE,
        weight=Decimal("1.0"),
        constraint_max=Decimal("0.2"),
        description="Minimize maximum drawdown",
        is_primary=False
    )


def create_sharpe_ratio_objective() -> OptimizationObjective:
    """Create a Sharpe ratio optimization objective."""
    return OptimizationObjective(
        name="sharpe_ratio",
        direction=ObjectiveDirection.MAXIMIZE,
        weight=Decimal("1.0"),
        constraint_min=Decimal("1.0"),
        description="Maximize risk-adjusted returns (Sharpe ratio)",
        is_primary=True
    )


def create_standard_trading_objectives() -> List[OptimizationObjective]:
    """Create a standard set of trading optimization objectives."""
    return [
        create_profit_maximization_objective(),
        create_sharpe_ratio_objective(),
        create_risk_minimization_objective()
    ]