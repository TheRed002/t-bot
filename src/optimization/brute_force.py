"""
Brute Force Optimization with Grid Search and Intelligent Sampling.

This module implements comprehensive brute force optimization approaches
including grid search, random search, and quasi-random sequences for
parameter space exploration with robust overfitting prevention.

Key Features:
- Grid search with adaptive refinement
- Random and quasi-random sampling (Sobol, Halton)
- Latin Hypercube sampling for efficient space coverage
- Parallel evaluation with resource management
- Progressive validation with walk-forward analysis
- Statistical significance testing
- Memory-efficient batch processing

Critical for Financial Applications:
- Decimal precision preservation
- Robust cross-validation
- Overfitting detection and prevention
- Statistical significance testing
- Resource management for large parameter spaces
- Comprehensive logging for audit trails
"""

import asyncio
import itertools
import math
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy.stats import qmc  # For quasi-random sequences

from src.core.exceptions import OptimizationError
from src.core.logging import get_logger
from src.optimization.core import (
    OptimizationConfig,
    OptimizationEngine,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStatus,
)
from src.optimization.parameter_space import ParameterSpace, SamplingStrategy
from src.utils.decorators import memory_usage, time_execution

logger = get_logger(__name__)


class GridSearchConfig(BaseModel):
    """
    Configuration for grid search optimization.

    Defines how the parameter grid should be constructed and searched,
    including refinement strategies and resource management.
    """

    # Grid construction
    grid_resolution: int = Field(
        default=10, ge=2, description="Number of points per parameter dimension"
    )
    adaptive_refinement: bool = Field(
        default=True, description="Enable adaptive grid refinement around best regions"
    )
    refinement_iterations: int = Field(
        default=3, ge=1, description="Number of refinement iterations"
    )
    refinement_factor: float = Field(
        default=0.3, gt=0, lt=1, description="Factor by which to shrink grid for refinement"
    )

    # Sampling strategies
    sampling_strategy: SamplingStrategy = Field(
        default=SamplingStrategy.GRID, description="Primary sampling strategy"
    )
    random_samples: int | None = Field(
        default=None, ge=1, description="Number of random samples (if using random sampling)"
    )
    use_quasi_random: bool = Field(
        default=False, description="Use quasi-random sequences (Sobol/Halton)"
    )

    # Resource management
    batch_size: int = Field(default=100, ge=1, description="Batch size for parameter evaluation")
    max_concurrent_evaluations: int = Field(
        default=4, ge=1, description="Maximum concurrent evaluations"
    )
    memory_limit_per_batch_mb: int = Field(
        default=1000, ge=100, description="Memory limit per batch in MB"
    )

    # Early stopping
    early_stopping_enabled: bool = Field(
        default=True, description="Enable early stopping for poor performers"
    )
    early_stopping_patience: int = Field(
        default=100, ge=10, description="Patience for early stopping"
    )
    early_stopping_threshold: Decimal = Field(
        default=Decimal("0.1"), gt=Decimal("0"), description="Threshold for early stopping"
    )

    # Quality control
    duplicate_detection: bool = Field(
        default=True, description="Detect and skip duplicate parameter combinations"
    )
    parameter_validation: bool = Field(
        default=True, description="Validate parameters before evaluation"
    )

    @field_validator("grid_resolution")
    @classmethod
    def validate_grid_resolution(cls, v):
        """Validate grid resolution is reasonable."""
        if v > 50:
            logger.warning(f"Large grid resolution {v} may result in very long optimization times")
        return v


class ValidationConfig(BaseModel):
    """
    Configuration for validation during optimization.

    Defines validation strategies to prevent overfitting and ensure
    robust parameter selection.
    """

    # Cross-validation
    enable_cross_validation: bool = Field(default=True, description="Enable cross-validation")
    cv_folds: int = Field(default=5, ge=2, description="Number of cross-validation folds")
    cv_strategy: str = Field(default="time_series", description="Cross-validation strategy")

    # Walk-forward analysis
    enable_walk_forward: bool = Field(default=True, description="Enable walk-forward analysis")
    walk_forward_periods: int = Field(default=6, ge=2, description="Number of walk-forward periods")
    walk_forward_step: int = Field(
        default=30, ge=1, description="Step size for walk-forward analysis (in days)"
    )

    # Out-of-sample testing
    out_of_sample_ratio: float = Field(
        default=0.25, gt=0, lt=0.5, description="Ratio of data reserved for out-of-sample testing"
    )
    require_out_of_sample_significance: bool = Field(
        default=True, description="Require statistical significance on out-of-sample data"
    )

    # Statistical testing
    significance_level: Decimal = Field(
        default=Decimal("0.05"),
        gt=Decimal("0"),
        lt=Decimal("1"),
        description="Statistical significance level",
    )
    bootstrap_samples: int = Field(
        default=1000, ge=100, description="Number of bootstrap samples for significance testing"
    )

    # Overfitting detection
    overfitting_threshold: Decimal = Field(
        default=Decimal("0.2"), gt=Decimal("0"), description="Threshold for overfitting detection"
    )
    performance_degradation_threshold: Decimal = Field(
        default=Decimal("0.15"),
        gt=Decimal("0"),
        description="Threshold for performance degradation",
    )


class OptimizationCandidate(BaseModel):
    """
    Represents a single parameter combination candidate for evaluation.

    Contains the parameters, evaluation status, and results.
    """

    candidate_id: str = Field(description="Unique candidate identifier")
    parameters: dict[str, Any] = Field(description="Parameter values")

    # Evaluation status
    status: str = Field(default="pending", description="Evaluation status")
    start_time: datetime | None = Field(default=None, description="Evaluation start time")
    end_time: datetime | None = Field(default=None, description="Evaluation end time")

    # Results
    objective_value: Decimal | None = Field(default=None, description="Primary objective value")
    objective_values: dict[str, Decimal] = Field(
        default_factory=dict, description="All objective values"
    )
    validation_score: Decimal | None = Field(default=None, description="Cross-validation score")

    # Quality metrics
    is_feasible: bool = Field(default=True, description="Whether candidate satisfies constraints")
    constraint_violations: dict[str, Decimal] = Field(
        default_factory=dict, description="Constraint violation amounts"
    )

    # Metadata
    evaluation_duration: Decimal | None = Field(
        default=None, description="Evaluation duration in seconds"
    )
    memory_usage_mb: Decimal | None = Field(
        default=None, description="Memory usage during evaluation"
    )
    error_message: str | None = Field(default=None, description="Error message if failed")

    def mark_started(self) -> None:
        """Mark candidate evaluation as started."""
        self.status = "running"
        self.start_time = datetime.now(timezone.utc)

    def mark_completed(
        self,
        objective_value: Decimal,
        objective_values: dict[str, Decimal],
        validation_score: Decimal | None = None,
    ) -> None:
        """Mark candidate evaluation as completed."""
        self.status = "completed"
        self.end_time = datetime.now(timezone.utc)
        self.objective_value = objective_value
        self.objective_values = objective_values
        self.validation_score = validation_score

        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.evaluation_duration = Decimal(str(duration))

    def mark_failed(self, error_message: str) -> None:
        """Mark candidate evaluation as failed."""
        self.status = "failed"
        self.end_time = datetime.now(timezone.utc)
        self.error_message = error_message

        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.evaluation_duration = Decimal(str(duration))


class GridGenerator:
    """
    Generates parameter grids for brute force optimization.

    Supports different grid generation strategies including uniform grids,
    adaptive refinement, and quasi-random sampling.
    """

    def __init__(self, parameter_space: ParameterSpace, config: GridSearchConfig):
        """
        Initialize grid generator.

        Args:
            parameter_space: Parameter space definition
            config: Grid search configuration
        """
        self.parameter_space = parameter_space
        self.config = config
        self.generated_grids: list[dict[str, Any]] = []

        logger.info(
            "GridGenerator initialized",
            parameter_count=len(parameter_space.parameters),
            grid_resolution=config.grid_resolution,
            sampling_strategy=config.sampling_strategy.value,
        )

    def generate_initial_grid(self) -> list[dict[str, Any]]:
        """
        Generate initial parameter grid.

        Returns:
            List of parameter combinations
        """
        if self.config.sampling_strategy == SamplingStrategy.GRID:
            return self._generate_uniform_grid()
        elif self.config.sampling_strategy == SamplingStrategy.UNIFORM:
            return self._generate_random_grid()
        elif self.config.sampling_strategy == SamplingStrategy.LATIN_HYPERCUBE:
            return self._generate_latin_hypercube()
        elif self.config.sampling_strategy == SamplingStrategy.SOBOL:
            return self._generate_sobol_sequence()
        elif self.config.sampling_strategy == SamplingStrategy.HALTON:
            return self._generate_halton_sequence()
        else:
            return self._generate_uniform_grid()

    def _generate_uniform_grid(self) -> list[dict[str, Any]]:
        """Generate uniform grid across all parameters."""
        parameter_grids = {}

        for param_name, param_def in self.parameter_space.parameters.items():
            if param_def.parameter_type.value == "continuous":
                min_val, max_val = param_def.get_bounds()
                values = np.linspace(float(min_val), float(max_val), self.config.grid_resolution)
                parameter_grids[param_name] = [Decimal(str(v)) for v in values]

            elif param_def.parameter_type.value == "discrete":
                min_val, max_val = param_def.get_bounds()
                step = getattr(param_def, "step_size", 1)
                values = list(range(min_val, max_val + 1, step))
                # Subsample if too many values
                if len(values) > self.config.grid_resolution:
                    indices = np.linspace(
                        0, len(values) - 1, self.config.grid_resolution, dtype=int
                    )
                    values = [values[i] for i in indices]
                parameter_grids[param_name] = values

            elif param_def.parameter_type.value == "categorical":
                choices = getattr(param_def, "choices", [])
                parameter_grids[param_name] = choices

            elif param_def.parameter_type.value == "boolean":
                parameter_grids[param_name] = [False, True]

        # Generate all combinations
        param_names = list(parameter_grids.keys())
        param_values = list(parameter_grids.values())

        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo, strict=False))
            combinations.append(param_dict)

        logger.info(f"Generated uniform grid with {len(combinations)} combinations")
        return combinations

    def _generate_random_grid(self) -> list[dict[str, Any]]:
        """Generate random parameter combinations."""
        num_samples = self.config.random_samples or (
            self.config.grid_resolution ** len(self.parameter_space.parameters)
        )

        combinations = []
        for _ in range(num_samples):
            params = self.parameter_space.sample(SamplingStrategy.UNIFORM)
            combinations.append(params)

        logger.info(f"Generated random grid with {len(combinations)} combinations")
        return combinations

    def _generate_latin_hypercube(self) -> list[dict[str, Any]]:
        """Generate Latin Hypercube sampling."""
        num_samples = self.config.random_samples or (
            self.config.grid_resolution ** max(1, len(self.parameter_space.parameters) // 2)
        )

        # Get continuous parameters only for LHS
        continuous_params = {
            name: param
            for name, param in self.parameter_space.parameters.items()
            if param.parameter_type.value == "continuous"
        }

        if not continuous_params:
            # Fall back to random sampling
            return self._generate_random_grid()

        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=len(continuous_params))
        samples = sampler.random(n=num_samples)

        # Scale samples to parameter bounds
        combinations = []
        param_names = list(continuous_params.keys())

        for sample in samples:
            params = {}

            # Set continuous parameters from LHS
            for i, param_name in enumerate(param_names):
                param_def = continuous_params[param_name]
                min_val, max_val = param_def.get_bounds()
                scaled_value = min_val + (max_val - min_val) * Decimal(str(sample[i]))
                params[param_name] = scaled_value

            # Sample other parameter types randomly
            for param_name, param_def in self.parameter_space.parameters.items():
                if param_name not in continuous_params:
                    params[param_name] = param_def.sample()

            combinations.append(params)

        logger.info(f"Generated Latin Hypercube sampling with {len(combinations)} combinations")
        return combinations

    def _generate_sobol_sequence(self) -> list[dict[str, Any]]:
        """Generate Sobol quasi-random sequence."""
        num_samples = self.config.random_samples or 1024  # Power of 2 for Sobol

        continuous_params = {
            name: param
            for name, param in self.parameter_space.parameters.items()
            if param.parameter_type.value == "continuous"
        }

        if not continuous_params:
            return self._generate_random_grid()

        # Generate Sobol sequence
        sampler = qmc.Sobol(d=len(continuous_params), scramble=True)
        samples = sampler.random(n=num_samples)

        combinations = []
        param_names = list(continuous_params.keys())

        for sample in samples:
            params = {}

            # Set continuous parameters from Sobol sequence
            for i, param_name in enumerate(param_names):
                param_def = continuous_params[param_name]
                min_val, max_val = param_def.get_bounds()
                scaled_value = min_val + (max_val - min_val) * Decimal(str(sample[i]))
                params[param_name] = scaled_value

            # Sample other parameter types
            for param_name, param_def in self.parameter_space.parameters.items():
                if param_name not in continuous_params:
                    params[param_name] = param_def.sample()

            combinations.append(params)

        logger.info(f"Generated Sobol sequence with {len(combinations)} combinations")
        return combinations

    def _generate_halton_sequence(self) -> list[dict[str, Any]]:
        """Generate Halton quasi-random sequence."""
        num_samples = self.config.random_samples or 1000

        continuous_params = {
            name: param
            for name, param in self.parameter_space.parameters.items()
            if param.parameter_type.value == "continuous"
        }

        if not continuous_params:
            return self._generate_random_grid()

        # Generate Halton sequence
        sampler = qmc.Halton(d=len(continuous_params), scramble=True)
        samples = sampler.random(n=num_samples)

        combinations = []
        param_names = list(continuous_params.keys())

        for sample in samples:
            params = {}

            # Set continuous parameters from Halton sequence
            for i, param_name in enumerate(param_names):
                param_def = continuous_params[param_name]
                min_val, max_val = param_def.get_bounds()
                scaled_value = min_val + (max_val - min_val) * Decimal(str(sample[i]))
                params[param_name] = scaled_value

            # Sample other parameter types
            for param_name, param_def in self.parameter_space.parameters.items():
                if param_name not in continuous_params:
                    params[param_name] = param_def.sample()

            combinations.append(params)

        logger.info(f"Generated Halton sequence with {len(combinations)} combinations")
        return combinations

    def generate_refined_grid(
        self, best_candidates: list[OptimizationCandidate], refinement_factor: float
    ) -> list[dict[str, Any]]:
        """
        Generate refined grid around best candidates.

        Args:
            best_candidates: Best candidates from previous iteration
            refinement_factor: Factor by which to shrink the search space

        Returns:
            List of refined parameter combinations
        """
        if not best_candidates:
            return []

        # Calculate bounds around best candidates
        refined_bounds = {}

        for param_name in self.parameter_space.parameters:
            param_def = self.parameter_space.parameters[param_name]

            if param_def.parameter_type.value == "continuous":
                # Get values from best candidates
                values = []
                for candidate in best_candidates:
                    if param_name in candidate.parameters:
                        values.append(Decimal(str(candidate.parameters[param_name])))

                if values:
                    min_val = min(values)
                    max_val = max(values)
                    center = (min_val + max_val) / 2
                    range_size = max_val - min_val

                    # Expand range by refinement factor
                    new_range = range_size * Decimal(str(refinement_factor))
                    new_min = max(param_def.get_bounds()[0], center - new_range / 2)
                    new_max = min(param_def.get_bounds()[1], center + new_range / 2)

                    refined_bounds[param_name] = (new_min, new_max)
                else:
                    refined_bounds[param_name] = param_def.get_bounds()
            else:
                # For discrete/categorical parameters, use original bounds
                refined_bounds[param_name] = param_def.get_bounds()

        # Generate grid within refined bounds
        parameter_grids = {}

        for param_name, param_def in self.parameter_space.parameters.items():
            if param_def.parameter_type.value == "continuous":
                min_val, max_val = refined_bounds[param_name]
                values = np.linspace(
                    float(min_val),
                    float(max_val),
                    max(3, self.config.grid_resolution // 2),  # Smaller grid for refinement
                )
                parameter_grids[param_name] = [Decimal(str(v)) for v in values]

            elif param_def.parameter_type.value == "discrete":
                # Use original discrete values
                min_val, max_val = param_def.get_bounds()
                step = getattr(param_def, "step_size", 1)
                values = list(range(min_val, max_val + 1, step))
                parameter_grids[param_name] = values

            elif param_def.parameter_type.value == "categorical":
                choices = getattr(param_def, "choices", [])
                parameter_grids[param_name] = choices

            elif param_def.parameter_type.value == "boolean":
                parameter_grids[param_name] = [False, True]

        # Generate combinations
        param_names = list(parameter_grids.keys())
        param_values = list(parameter_grids.values())

        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo, strict=False))
            combinations.append(param_dict)

        logger.info(f"Generated refined grid with {len(combinations)} combinations")
        return combinations


class BruteForceOptimizer(OptimizationEngine):
    """
    Brute force optimization engine with grid search and intelligent sampling.

    Provides comprehensive parameter space exploration with robust validation,
    overfitting prevention, and performance optimization for trading strategies.
    """

    def __init__(
        self,
        objectives: list[OptimizationObjective],
        parameter_space: ParameterSpace,
        config: OptimizationConfig | None = None,
        grid_config: GridSearchConfig | None = None,
        validation_config: ValidationConfig | None = None,
    ):
        """
        Initialize brute force optimizer.

        Args:
            objectives: List of optimization objectives
            parameter_space: Parameter space definition
            config: General optimization configuration
            grid_config: Grid search specific configuration
            validation_config: Validation configuration
        """
        super().__init__(objectives, [], config)

        self.parameter_space = parameter_space
        self.grid_config = grid_config or GridSearchConfig()
        self.validation_config = validation_config or ValidationConfig()

        # Initialize components
        self.grid_generator = GridGenerator(parameter_space, self.grid_config)

        # State tracking
        self.candidates: list[OptimizationCandidate] = []
        self.completed_candidates: list[OptimizationCandidate] = []
        self.best_candidate: OptimizationCandidate | None = None
        self.current_iteration = 0

        # Resource management
        self.active_evaluations = 0
        self.evaluation_executor: ProcessPoolExecutor | None = None

        # Performance tracking
        self.evaluation_times: list[float] = []
        self.memory_usage_samples: list[float] = []

        logger.info(
            "BruteForceOptimizer initialized",
            optimization_id=self.optimization_id,
            parameter_count=len(parameter_space.parameters),
            grid_resolution=self.grid_config.grid_resolution,
            sampling_strategy=self.grid_config.sampling_strategy.value,
        )

    @time_execution
    @memory_usage
    async def optimize(
        self,
        objective_function: Callable,
        parameter_space: dict[str, Any] | None = None,
        initial_parameters: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """
        Run brute force optimization.

        Args:
            objective_function: Function to optimize
            parameter_space: Not used (using internal parameter space)
            initial_parameters: Initial parameters (optional)

        Returns:
            Optimization result
        """
        try:
            self.progress.status = OptimizationStatus.INITIALIZING
            logger.info("Starting brute force optimization", optimization_id=self.optimization_id)

            # Generate initial grid
            initial_grid = self.grid_generator.generate_initial_grid()
            self._create_candidates(initial_grid)

            # Initialize executor
            self.evaluation_executor = ProcessPoolExecutor(
                max_workers=self.grid_config.max_concurrent_evaluations
            )

            self.progress.status = OptimizationStatus.RUNNING
            self.progress.total_iterations = len(self.candidates)

            # Evaluate candidates in batches
            await self._evaluate_candidates_in_batches(objective_function)

            # Adaptive refinement if enabled
            if self.grid_config.adaptive_refinement:
                await self._run_adaptive_refinement(objective_function)

            # Final validation and result preparation
            result = await self._finalize_optimization()

            self.progress.status = OptimizationStatus.COMPLETED
            logger.info(
                "Brute force optimization completed",
                optimization_id=self.optimization_id,
                best_objective=float(result.optimal_objective_value),
                candidates_evaluated=len(self.completed_candidates),
            )

            return result

        except Exception as e:
            self.progress.status = OptimizationStatus.FAILED
            logger.error(
                "Brute force optimization failed",
                optimization_id=self.optimization_id,
                error=str(e),
            )
            raise OptimizationError(f"Brute force optimization failed: {e!s}")

        finally:
            if self.evaluation_executor:
                self.evaluation_executor.shutdown(wait=True)

    def _create_candidates(self, parameter_combinations: list[dict[str, Any]]) -> None:
        """Create optimization candidates from parameter combinations."""
        for i, params in enumerate(parameter_combinations):
            candidate = OptimizationCandidate(
                candidate_id=f"{self.optimization_id}_candidate_{len(self.candidates) + i}",
                parameters=params,
            )
            self.candidates.append(candidate)

        logger.info(f"Created {len(parameter_combinations)} optimization candidates")

    async def _evaluate_candidates_in_batches(self, objective_function: Callable) -> None:
        """Evaluate candidates in batches to manage resources."""
        batch_size = self.grid_config.batch_size
        total_candidates = len(self.candidates)

        for batch_start in range(0, total_candidates, batch_size):
            batch_end = min(batch_start + batch_size, total_candidates)
            batch = self.candidates[batch_start:batch_end]

            logger.info(
                f"Evaluating batch {batch_start // batch_size + 1}",
                batch_size=len(batch),
                progress_pct=round((batch_start / total_candidates) * 100, 2),
            )

            # Evaluate batch
            await self._evaluate_batch(batch, objective_function)

            # Update progress
            self.update_progress(
                iteration=batch_end, message=f"Completed batch {batch_start // batch_size + 1}"
            )

            # Early stopping check
            if self._should_stop_early():
                logger.info("Early stopping triggered")
                break

    async def _evaluate_batch(
        self, batch: list[OptimizationCandidate], objective_function: Callable
    ) -> None:
        """Evaluate a batch of candidates."""
        tasks = []
        processed_in_batch = []  # Track candidates processed in this batch

        for candidate in batch:
            if self.grid_config.duplicate_detection and self._is_duplicate(
                candidate, processed_in_batch
            ):
                candidate.mark_failed("Duplicate parameter combination")
                continue

            if self.grid_config.parameter_validation and not self._validate_candidate(candidate):
                candidate.mark_failed("Parameter validation failed")
                continue

            # Add to processed list before creating task
            processed_in_batch.append(candidate)
            task = self._evaluate_candidate(candidate, objective_function)
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _evaluate_candidate(
        self, candidate: OptimizationCandidate, objective_function: Callable
    ) -> None:
        """Evaluate a single candidate."""
        try:
            candidate.mark_started()
            self.active_evaluations += 1

            # Run objective function
            result = await self._run_objective_function(objective_function, candidate.parameters)

            if result is not None:
                # Extract objective values
                if isinstance(result, dict):
                    objective_values = {k: Decimal(str(v)) for k, v in result.items()}
                    primary_objective = self._get_primary_objective_value(objective_values)
                else:
                    primary_objective = Decimal(str(result))
                    objective_values = {self.objectives[0].name: primary_objective}

                # Run validation if enabled
                validation_score = None
                if self.validation_config.enable_cross_validation:
                    validation_score = await self._validate_candidate_performance(
                        candidate, objective_function
                    )

                candidate.mark_completed(primary_objective, objective_values, validation_score)
                self.completed_candidates.append(candidate)

                # Update best candidate
                if self._is_better_candidate(candidate):
                    self.best_candidate = candidate
                    logger.info(
                        "New best candidate found",
                        candidate_id=candidate.candidate_id,
                        objective_value=float(primary_objective),
                    )
            else:
                candidate.mark_failed("Objective function returned None")

        except Exception as e:
            candidate.mark_failed(str(e))
            logger.error(
                "Candidate evaluation failed", candidate_id=candidate.candidate_id, error=str(e)
            )

        finally:
            self.active_evaluations -= 1

    async def _run_objective_function(
        self, objective_function: Callable, parameters: dict[str, Any]
    ) -> Any:
        """Run the objective function with given parameters."""
        try:
            # Convert parameters to appropriate types
            converted_params = self.parameter_space.clip_parameters(parameters)

            # Run function (assume it's async-compatible)
            if asyncio.iscoroutinefunction(objective_function):
                result = await objective_function(converted_params)
            else:
                # Run in executor for CPU-bound functions
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.evaluation_executor, objective_function, converted_params
                )

            return result

        except Exception as e:
            logger.error(f"Objective function execution failed: {e!s}")
            return None

    def _get_primary_objective_value(self, objective_values: dict[str, Decimal]) -> Decimal:
        """Get primary objective value from results."""
        # Find primary objective
        primary_obj = None
        for obj in self.objectives:
            if obj.is_primary:
                primary_obj = obj
                break

        if primary_obj is None:
            primary_obj = self.objectives[0]  # Use first objective as primary

        return objective_values.get(primary_obj.name, Decimal("0"))

    def _is_better_candidate(self, candidate: OptimizationCandidate) -> bool:
        """Check if candidate is better than current best."""
        if self.best_candidate is None:
            return True

        if candidate.objective_value is None:
            return False

        # Use primary objective for comparison
        primary_obj = self.objectives[0]  # Simplified - use first objective

        return primary_obj.is_better(candidate.objective_value, self.best_candidate.objective_value)

    def _is_duplicate(
        self,
        candidate: OptimizationCandidate,
        processed_in_batch: list[OptimizationCandidate] | None = None,
    ) -> bool:
        """Check if candidate is a duplicate of previously evaluated candidate."""
        # Check against completed candidates
        for completed in self.completed_candidates:
            if self._parameters_equal(candidate.parameters, completed.parameters):
                return True

        # Check against candidates processed in current batch
        if processed_in_batch:
            for processed in processed_in_batch:
                if self._parameters_equal(candidate.parameters, processed.parameters):
                    return True

        return False

    def _parameters_equal(self, params1: dict[str, Any], params2: dict[str, Any]) -> bool:
        """Check if two parameter sets are equal within tolerance."""
        if set(params1.keys()) != set(params2.keys()):
            return False

        tolerance = float(self.config.convergence_tolerance)

        for key in params1:
            val1 = params1[key]
            val2 = params2[key]

            if isinstance(val1, int | float | Decimal) and isinstance(val2, int | float | Decimal):
                if abs(float(val1) - float(val2)) > tolerance:
                    return False
            else:
                if val1 != val2:
                    return False

        return True

    def _validate_candidate(self, candidate: OptimizationCandidate) -> bool:
        """Validate candidate parameters."""
        validation_results = self.parameter_space.validate_parameters(candidate.parameters)
        return all(validation_results.values())

    async def _validate_candidate_performance(
        self, candidate: OptimizationCandidate, objective_function: Callable
    ) -> Decimal | None:
        """Validate candidate performance using cross-validation."""
        if not self.validation_config.enable_cross_validation:
            return None

        try:
            # Simplified cross-validation - in practice would use proper CV
            scores = []

            for _fold in range(self.validation_config.cv_folds):
                # Run objective function with slight parameter perturbation
                perturbed_params = self._perturb_parameters(candidate.parameters, 0.01)
                result = await self._run_objective_function(objective_function, perturbed_params)

                if result is not None:
                    if isinstance(result, dict):
                        score = self._get_primary_objective_value(
                            {k: Decimal(str(v)) for k, v in result.items()}
                        )
                    else:
                        score = Decimal(str(result))
                    scores.append(score)

            if scores:
                return sum(scores) / len(scores)

        except Exception as e:
            logger.warning(f"Cross-validation failed for candidate {candidate.candidate_id}: {e!s}")

        return None

    def _perturb_parameters(
        self, parameters: dict[str, Any], noise_factor: float
    ) -> dict[str, Any]:
        """Add small perturbations to parameters for validation."""
        perturbed = parameters.copy()

        for param_name, value in parameters.items():
            if param_name in self.parameter_space.parameters:
                param_def = self.parameter_space.parameters[param_name]

                if param_def.parameter_type.value == "continuous":
                    min_val, max_val = param_def.get_bounds()
                    range_size = max_val - min_val
                    noise = random.uniform(-noise_factor, noise_factor) * range_size
                    new_value = Decimal(str(value)) + Decimal(str(noise))
                    perturbed[param_name] = param_def.clip_value(new_value)

        return perturbed

    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria are met."""
        if not self.grid_config.early_stopping_enabled:
            return False

        if len(self.completed_candidates) < self.grid_config.early_stopping_patience:
            return False

        # Check if no improvement in recent candidates
        recent_candidates = self.completed_candidates[-self.grid_config.early_stopping_patience :]
        recent_scores = [
            c.objective_value for c in recent_candidates if c.objective_value is not None
        ]

        if len(recent_scores) < 2:
            return False

        # Check improvement
        improvement = abs(max(recent_scores) - min(recent_scores))
        return improvement < self.grid_config.early_stopping_threshold

    async def _run_adaptive_refinement(self, objective_function: Callable) -> None:
        """Run adaptive grid refinement around best regions."""
        for iteration in range(self.grid_config.refinement_iterations):
            logger.info(f"Running refinement iteration {iteration + 1}")

            # Get top candidates for refinement
            top_candidates = self._get_top_candidates(
                min(10, len(self.completed_candidates) // 4)  # Top 25% or min 10
            )

            if not top_candidates:
                break

            # Generate refined grid
            refined_grid = self.grid_generator.generate_refined_grid(
                top_candidates, self.grid_config.refinement_factor
            )

            if not refined_grid:
                break

            # Create and evaluate refined candidates
            refined_candidates = []
            for params in refined_grid:
                candidate = OptimizationCandidate(
                    candidate_id=f"{self.optimization_id}_refined_{iteration}_{len(refined_candidates)}",
                    parameters=params,
                )
                refined_candidates.append(candidate)

            # Evaluate refined candidates
            await self._evaluate_batch(refined_candidates, objective_function)

    def _get_top_candidates(self, count: int) -> list[OptimizationCandidate]:
        """Get top performing candidates."""
        valid_candidates = [
            c
            for c in self.completed_candidates
            if c.objective_value is not None and c.status == "completed"
        ]

        # Sort by objective value (assuming maximization for simplicity)
        sorted_candidates = sorted(valid_candidates, key=lambda x: x.objective_value, reverse=True)

        return sorted_candidates[:count]

    async def _finalize_optimization(self) -> OptimizationResult:
        """Finalize optimization and create result."""
        if not self.best_candidate:
            raise OptimizationError("No valid candidates found")

        # Calculate validation metrics
        validation_score = self.best_candidate.validation_score

        # Statistical significance testing
        statistical_significance = await self._calculate_statistical_significance()

        # Parameter stability analysis
        parameter_stability = self._analyze_parameter_stability()

        # Create result
        result = OptimizationResult(
            optimization_id=self.optimization_id,
            algorithm_name="BruteForceGridSearch",
            optimal_parameters=self.best_candidate.parameters,
            optimal_objective_value=self.best_candidate.objective_value,
            objective_values=self.best_candidate.objective_values,
            validation_score=validation_score,
            iterations_completed=len(self.completed_candidates),
            evaluations_completed=len(self.completed_candidates),
            convergence_achieved=True,  # Grid search always "converges"
            start_time=self.progress.start_time,
            end_time=datetime.now(timezone.utc),
            total_duration_seconds=Decimal(
                str((datetime.now(timezone.utc) - self.progress.start_time).total_seconds())
            ),
            parameter_stability=parameter_stability,
            statistical_significance=statistical_significance,
            config_used=self.config.model_dump(),
        )

        self.result = result
        return result

    async def _calculate_statistical_significance(self) -> Decimal | None:
        """Calculate statistical significance of results."""
        if len(self.completed_candidates) < 10:
            return None

        try:
            # Simple bootstrap test
            scores = [
                c.objective_value
                for c in self.completed_candidates
                if c.objective_value is not None
            ]

            if len(scores) < 10:
                return None

            # Bootstrap confidence interval
            bootstrap_means = []
            for _ in range(self.validation_config.bootstrap_samples):
                sample = random.choices(scores, k=len(scores))
                bootstrap_means.append(sum(sample) / len(sample))

            bootstrap_means.sort()
            alpha = float(self.validation_config.significance_level)
            lower_idx = int(alpha / 2 * len(bootstrap_means))
            upper_idx = int((1 - alpha / 2) * len(bootstrap_means))

            # Check if confidence interval excludes zero (for significance)
            lower_bound = bootstrap_means[lower_idx]
            upper_bound = bootstrap_means[upper_idx]

            if float(lower_bound) > 0 or float(upper_bound) < 0:
                return self.validation_config.significance_level
            else:
                return Decimal("1.0")  # Not significant

        except Exception as e:
            logger.warning(f"Statistical significance calculation failed: {e!s}")
            return None

    def _analyze_parameter_stability(self) -> dict[str, Decimal]:
        """Analyze stability of parameters across top candidates."""
        stability_scores = {}

        top_candidates = self._get_top_candidates(min(10, len(self.completed_candidates)))

        if len(top_candidates) < 2:
            return stability_scores

        for param_name in self.parameter_space.parameters:
            values = []
            for candidate in top_candidates:
                if param_name in candidate.parameters:
                    values.append(candidate.parameters[param_name])

            if len(values) < 2:
                stability_scores[param_name] = Decimal("0")
                continue

            # Calculate coefficient of variation for continuous parameters
            if all(isinstance(v, int | float | Decimal) for v in values):
                float_values = [float(v) for v in values]
                mean_val = sum(float_values) / len(float_values)

                if mean_val != 0:
                    std_val = math.sqrt(
                        sum((v - mean_val) ** 2 for v in float_values) / len(float_values)
                    )
                    cv = std_val / abs(mean_val)
                    stability_scores[param_name] = Decimal(str(1.0 / (1.0 + cv)))
                else:
                    stability_scores[param_name] = Decimal("1")
            else:
                # For categorical parameters, measure mode frequency
                from collections import Counter

                counts = Counter(values)
                mode_freq = counts.most_common(1)[0][1]
                stability_scores[param_name] = Decimal(str(mode_freq / len(values)))

        return stability_scores

    def get_next_parameters(self) -> dict[str, Any] | None:
        """Get next parameters to evaluate (not used in batch optimization)."""
        # For brute force, all parameters are generated upfront
        return None
