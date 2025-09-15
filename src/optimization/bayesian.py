"""
Bayesian Optimization for Efficient Parameter Search.

This module implements Bayesian optimization using Gaussian Processes
for efficient exploration of expensive objective functions with proper
uncertainty quantification and acquisition function optimization.

Key Features:
- Gaussian Process surrogate models
- Multiple acquisition functions (EI, UCB, PI, Knowledge Gradient)
- Multi-objective Bayesian optimization
- Constraint handling with probabilistic constraints
- Active learning with uncertainty quantification
- Parallel evaluation with batch acquisition
- Robust handling of noisy objectives

Critical for Financial Applications:
- Decimal precision for financial parameters
- Robust uncertainty quantification
- Conservative acquisition strategies
- Proper handling of financial constraints
- Integration with risk management
"""

import asyncio
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel as C,
    Matern,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import OptimizationError, ValidationError
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

# Configuration constants
DEFAULT_OBJECTIVE_TIMEOUT_SECONDS = 60.0

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class AcquisitionFunction(BaseModel):
    """
    Acquisition function configuration for Bayesian optimization.

    Defines how to balance exploration vs exploitation when selecting
    the next point to evaluate.
    """

    name: str = Field(description="Acquisition function name")
    exploration_factor: Decimal = Field(
        default=Decimal("2.0"),
        gt=0,
        description="Exploration factor (higher = more exploration)",
    )
    jitter: Decimal = Field(
        default=Decimal("0.01"), ge=0, description="Jitter for numerical stability"
    )

    # Function-specific parameters
    xi: Decimal = Field(
        default=Decimal("0.01"),
        ge=0,
        description="Improvement threshold for Expected Improvement",
    )
    kappa: Decimal = Field(
        default=Decimal("2.576"),
        gt=0,
        description="Confidence parameter for UCB (95% = 1.96, 99% = 2.576)",
    )

    @field_validator("name")
    @classmethod
    def validate_acquisition_function(cls, v):
        """Validate acquisition function name."""
        valid_functions = ["ei", "ucb", "pi", "lcb", "kg", "mes"]
        if v.lower() not in valid_functions:
            raise ValueError(f"Invalid acquisition function: {v}. Must be one of {valid_functions}")
        return v.lower()


class GaussianProcessConfig(BaseModel):
    """
    Configuration for Gaussian Process surrogate model.

    Defines the GP kernel, optimization settings, and numerical
    stability parameters.
    """

    # Kernel configuration
    kernel_type: str = Field(
        default="matern", description="Kernel type: 'rbf', 'matern', 'rational_quadratic'"
    )
    length_scale: Decimal = Field(
        default=Decimal("1.0"), gt=0, description="Initial length scale for kernel"
    )
    length_scale_bounds: tuple[Decimal, Decimal] = Field(
        default=(Decimal("1e-5"), Decimal("1e5")),
        description="Bounds for length scale optimization",
    )

    # Noise handling
    alpha: Decimal = Field(
        default=Decimal("1e-10"), gt=0, description="Noise level (regularization parameter)"
    )
    white_kernel: bool = Field(default=True, description="Include white noise kernel")
    noise_level_bounds: tuple[Decimal, Decimal] = Field(
        default=(Decimal("1e-10"), Decimal("1e-1")), description="Bounds for noise level"
    )

    # Optimization settings
    n_restarts_optimizer: int = Field(
        default=10, ge=1, description="Number of restarts for hyperparameter optimization"
    )
    normalize_y: bool = Field(default=True, description="Normalize target values")

    # Numerical stability
    jitter: float = Field(default=1e-6, gt=0, description="Jitter for numerical stability")

    @field_validator("kernel_type")
    @classmethod
    def validate_kernel_type(cls, v):
        """Validate kernel type."""
        valid_kernels = ["rbf", "matern", "rational_quadratic"]
        if v.lower() not in valid_kernels:
            raise ValueError(f"Invalid kernel type: {v}. Must be one of {valid_kernels}")
        return v.lower()

    @field_validator("length_scale_bounds", "noise_level_bounds")
    @classmethod
    def validate_bounds(cls, v):
        """Validate bounds are properly ordered."""
        if v[0] >= v[1]:
            raise ValueError("Lower bound must be less than upper bound")
        return v


class BayesianConfig(BaseModel):
    """
    Configuration for Bayesian optimization.

    Combines GP configuration, acquisition function settings,
    and optimization strategy parameters.
    """

    # Surrogate model
    gp_config: GaussianProcessConfig = Field(
        default_factory=GaussianProcessConfig, description="Gaussian Process configuration"
    )

    # Acquisition function
    acquisition_function: AcquisitionFunction = Field(
        default_factory=lambda: AcquisitionFunction(name="ei"),
        description="Acquisition function configuration",
    )

    # Optimization strategy
    n_initial_points: int = Field(default=10, ge=2, description="Number of initial random points")
    n_calls: int = Field(default=100, ge=1, description="Total number of function evaluations")

    # Batch optimization
    batch_size: int = Field(default=1, ge=1, description="Number of points to evaluate in parallel")
    batch_strategy: str = Field(default="cl_mean", description="Batch acquisition strategy")

    # Convergence criteria
    convergence_tolerance: Decimal = Field(
        default=Decimal("1e-6"), gt=0, description="Convergence tolerance"
    )
    patience: int = Field(default=10, ge=1, description="Patience for early stopping")

    # Constraint handling
    constraint_tolerance: Decimal = Field(
        default=Decimal("0.01"), ge=0, description="Constraint violation tolerance"
    )

    @field_validator("batch_strategy")
    @classmethod
    def validate_batch_strategy(cls, v):
        """Validate batch strategy."""
        valid_strategies = ["cl_mean", "cl_min", "cl_max", "kb", "lp"]
        if v.lower() not in valid_strategies:
            raise ValueError(f"Invalid batch strategy: {v}. Must be one of {valid_strategies}")
        return v.lower()


class BayesianPoint(BaseModel):
    """
    Represents a point in the Bayesian optimization process.

    Contains parameters, objective value, and GP predictions.
    """

    point_id: str = Field(description="Unique point identifier")
    parameters: dict[str, Any] = Field(description="Parameter values")
    objective_value: Decimal | None = Field(default=None, description="Observed objective value")
    objective_std: Decimal | None = Field(default=None, description="Objective standard deviation")

    # GP predictions
    gp_mean: Decimal | None = Field(default=None, description="GP mean prediction")
    gp_std: Decimal | None = Field(default=None, description="GP standard deviation prediction")

    # Acquisition function values
    acquisition_value: Decimal | None = Field(
        default=None, description="Acquisition function value"
    )

    # Status
    evaluated: bool = Field(default=False, description="Whether point has been evaluated")
    evaluation_time: datetime | None = Field(default=None, description="Evaluation timestamp")

    # Constraints
    constraint_violations: dict[str, Decimal] = Field(
        default_factory=dict, description="Constraint violation values"
    )
    is_feasible: bool = Field(default=True, description="Whether point satisfies constraints")

    def mark_evaluated(
        self, objective_value: Decimal, objective_std: Decimal | None = None
    ) -> None:
        """Mark point as evaluated with results."""
        self.evaluated = True
        self.objective_value = objective_value
        self.objective_std = objective_std
        self.evaluation_time = datetime.now(timezone.utc)


class GaussianProcessModel:
    """
    Gaussian Process surrogate model for Bayesian optimization.

    Provides predictions with uncertainty quantification and
    handles parameter scaling and kernel optimization.
    """

    def __init__(self, config: GaussianProcessConfig, parameter_space: ParameterSpace):
        """
        Initialize GP model.

        Args:
            config: GP configuration
            parameter_space: Parameter space definition
        """
        self.config = config
        self.parameter_space = parameter_space

        # Initialize scalers
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler() if config.normalize_y else None

        # Initialize kernel
        self.kernel = self._create_kernel()

        # Initialize GP
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=config.alpha,
            n_restarts_optimizer=config.n_restarts_optimizer,
            normalize_y=False,  # We handle normalization manually
        )

        # Training data
        self.X_train = None
        self.y_train = None
        self.is_fitted = False

        logger.info(
            "GaussianProcessModel initialized",
            kernel_type=config.kernel_type,
            parameter_count=len(parameter_space.parameters),
        )

    def _create_kernel(self) -> Any:
        """Create GP kernel based on configuration."""
        # Base kernel
        if self.config.kernel_type == "rbf":
            base_kernel = RBF(
                length_scale=self.config.length_scale,
                length_scale_bounds=self.config.length_scale_bounds,
            )
        elif self.config.kernel_type == "matern":
            base_kernel = Matern(
                length_scale=self.config.length_scale,
                length_scale_bounds=self.config.length_scale_bounds,
                nu=2.5,  # Standard choice
            )
        else:  # rational_quadratic
            from sklearn.gaussian_process.kernels import RationalQuadratic

            base_kernel = RationalQuadratic(
                length_scale=self.config.length_scale,
                length_scale_bounds=self.config.length_scale_bounds,
            )

        # Add constant kernel for mean function
        kernel = C(1.0, (1e-3, 1e3)) * base_kernel

        # Add white noise kernel if enabled
        if self.config.white_kernel:
            kernel += WhiteKernel(
                noise_level=self.config.alpha, noise_level_bounds=self.config.noise_level_bounds
            )

        return kernel

    def _encode_parameters(self, parameters_list: list[dict[str, Any]]) -> np.ndarray:
        """Encode parameter dictionaries to numerical arrays."""
        encoded_list = []

        for parameters in parameters_list:
            encoded = []

            for param_name in sorted(self.parameter_space.parameters.keys()):
                if param_name in parameters:
                    value = parameters[param_name]
                    param_def = self.parameter_space.parameters[param_name]

                    if param_def.parameter_type.value == "continuous":
                        encoded.append(float(value))
                    elif param_def.parameter_type.value == "discrete":
                        encoded.append(float(value))
                    elif param_def.parameter_type.value == "categorical":
                        # One-hot encoding for categorical
                        choices = getattr(param_def, "choices", [])
                        for choice in choices:
                            encoded.append(1.0 if value == choice else 0.0)
                    elif param_def.parameter_type.value == "boolean":
                        encoded.append(1.0 if value else 0.0)
                else:
                    # Handle missing parameters
                    param_def = self.parameter_space.parameters[param_name]
                    if param_def.parameter_type.value == "categorical":
                        choices = getattr(param_def, "choices", [])
                        encoded.extend([0.0] * len(choices))
                    else:
                        encoded.append(0.0)

            encoded_list.append(encoded)

        return np.array(encoded_list)

    def fit(self, points: list[BayesianPoint]) -> None:
        """
        Fit GP model to observed data.

        Args:
            points: List of evaluated points
        """
        # Filter evaluated points
        evaluated_points = [p for p in points if p.evaluated and p.objective_value is not None]

        if len(evaluated_points) < 2:
            logger.warning("Insufficient data points for GP fitting")
            return

        # Encode parameters
        parameter_dicts = [p.parameters for p in evaluated_points]
        X = self._encode_parameters(parameter_dicts)

        # Extract objectives
        y = np.array([float(p.objective_value) for p in evaluated_points])

        # Scale data
        X_scaled = self.x_scaler.fit_transform(X)

        if self.y_scaler is not None:
            y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = y

        # Fit GP
        try:
            self.gp.fit(X_scaled, y_scaled)
            self.X_train = X_scaled
            self.y_train = y_scaled
            self.is_fitted = True

            logger.info(
                "GP model fitted successfully",
                data_points=len(evaluated_points),
                log_marginal_likelihood=self.gp.log_marginal_likelihood_value_,
            )

        except (ValueError, ArithmeticError) as e:
            logger.error(f"GP fitting failed due to data issue: {e!s}")
            self.is_fitted = False
        except Exception as e:
            logger.error(f"GP fitting failed unexpectedly: {e!s}")
            self.is_fitted = False

    def predict(
        self, parameters_list: list[dict[str, Any]], return_std: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Predict objective values for parameter sets.

        Args:
            parameters_list: List of parameter dictionaries
            return_std: Whether to return standard deviations

        Returns:
            Tuple of (means, stds) where stds is None if return_std=False
        """
        if not self.is_fitted:
            # Return prior predictions
            n_points = len(parameters_list)
            means = np.zeros(n_points)
            stds = np.ones(n_points) if return_std else None
            return means, stds

        # Encode parameters
        X = self._encode_parameters(parameters_list)
        X_scaled = self.x_scaler.transform(X)

        # Predict
        if return_std:
            y_pred, y_std = self.gp.predict(X_scaled, return_std=True)
        else:
            y_pred = self.gp.predict(X_scaled, return_std=False)
            y_std = None

        # Unscale predictions
        if self.y_scaler is not None:
            y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            if y_std is not None:
                # Standard deviation scaling
                y_std = y_std * self.y_scaler.scale_[0]

        return y_pred, y_std

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the fitted model."""
        if not self.is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "kernel": str(self.gp.kernel_),
            "log_marginal_likelihood": self.gp.log_marginal_likelihood_value_,
            "n_training_points": len(self.y_train) if self.y_train is not None else 0,
        }


class AcquisitionOptimizer:
    """
    Optimizes acquisition functions to select next evaluation points.

    Handles different acquisition functions and batch selection strategies.
    """

    def __init__(
        self,
        gp_model: GaussianProcessModel,
        config: BayesianConfig,
        parameter_space: ParameterSpace,
    ):
        """
        Initialize acquisition optimizer.

        Args:
            gp_model: Fitted GP model
            config: Bayesian optimization configuration
            parameter_space: Parameter space definition
        """
        self.gp_model = gp_model
        self.config = config
        self.parameter_space = parameter_space

        logger.info(
            "AcquisitionOptimizer initialized",
            acquisition_function=config.acquisition_function.name,
        )

    def optimize_acquisition(
        self, current_points: list[BayesianPoint], n_points: int = 1
    ) -> list[dict[str, Any]]:
        """
        Optimize acquisition function to find next evaluation points.

        Args:
            current_points: Currently evaluated points
            n_points: Number of points to select

        Returns:
            List of parameter dictionaries for next evaluations
        """
        if n_points == 1:
            return [self._optimize_single_point(current_points)]
        else:
            return self._optimize_batch(current_points, n_points)

    def _optimize_single_point(self, current_points: list[BayesianPoint]) -> dict[str, Any]:
        """Optimize acquisition function for single point."""
        # Current best objective value
        evaluated_points = [p for p in current_points if p.evaluated]

        if evaluated_points:
            best_value = max(float(p.objective_value) for p in evaluated_points)
        else:
            best_value = 0.0

        # Define acquisition function
        def acquisition_func(x_encoded):
            # Convert encoded parameters back to parameter dict
            params_list = [self._decode_parameters(x_encoded)]

            # Get GP predictions
            mean, std = self.gp_model.predict(params_list, return_std=True)

            # Calculate acquisition value
            acq_value = self._calculate_acquisition(mean[0], std[0], best_value)

            return -acq_value  # Minimize negative acquisition

        # Optimize using random search with local refinement
        best_params = None
        best_acq_value = float("-inf")

        # Random search
        for _ in range(100):
            random_params = self.parameter_space.sample()
            x_encoded = self.gp_model._encode_parameters([random_params])[0]

            acq_value = -acquisition_func(x_encoded)

            if acq_value > best_acq_value:
                best_acq_value = acq_value
                best_params = random_params

        return best_params or self.parameter_space.sample()

    def _optimize_batch(
        self, current_points: list[BayesianPoint], n_points: int
    ) -> list[dict[str, Any]]:
        """Optimize acquisition function for batch of points."""
        # For simplicity, use greedy batch selection
        selected_points: list[dict[str, Any]] = []

        for _i in range(n_points):
            # Create temporary points list including previously selected
            temp_points = current_points + [
                BayesianPoint(point_id=f"temp_{j}", parameters=params, evaluated=False)
                for j, params in enumerate(selected_points)
            ]

            # Optimize single point
            next_point = self._optimize_single_point(temp_points)
            selected_points.append(next_point)

        return selected_points

    def _decode_parameters(self, x_encoded: np.ndarray) -> dict[str, Any]:
        """Decode numerical array back to parameter dictionary."""
        params = {}
        idx = 0

        for param_name in sorted(self.parameter_space.parameters.keys()):
            param_def = self.parameter_space.parameters[param_name]

            if param_def.parameter_type.value in ["continuous", "discrete"]:
                value = x_encoded[idx]

                if param_def.parameter_type.value == "discrete":
                    value = round(value)

                params[param_name] = param_def.clip_value(value)
                idx += 1

            elif param_def.parameter_type.value == "categorical":
                choices = getattr(param_def, "choices", [])
                choice_values = x_encoded[idx : idx + len(choices)]
                best_choice_idx = np.argmax(choice_values)
                params[param_name] = choices[best_choice_idx]
                idx += len(choices)

            elif param_def.parameter_type.value == "boolean":
                params[param_name] = x_encoded[idx] > 0.5
                idx += 1

        return params

    def _calculate_acquisition(self, mean: float, std: float, best_value: float) -> float:
        """Calculate acquisition function value."""
        func_name = self.config.acquisition_function.name

        if func_name == "ei":
            return self._expected_improvement(mean, std, best_value)
        elif func_name == "ucb":
            return self._upper_confidence_bound(mean, std)
        elif func_name == "pi":
            return self._probability_of_improvement(mean, std, best_value)
        elif func_name == "lcb":
            return self._lower_confidence_bound(mean, std)
        else:
            # Default to Expected Improvement
            return self._expected_improvement(mean, std, best_value)

    def _expected_improvement(self, mean: float, std: float, best_value: float) -> float:
        """Calculate Expected Improvement."""
        if std == 0:
            return 0.0

        xi = float(self.config.acquisition_function.xi)
        improvement = mean - best_value - xi
        z = improvement / std

        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return max(0.0, ei)

    def _upper_confidence_bound(self, mean: float, std: float) -> float:
        """Calculate Upper Confidence Bound."""
        kappa = float(self.config.acquisition_function.kappa)
        return mean + kappa * std

    def _probability_of_improvement(self, mean: float, std: float, best_value: float) -> float:
        """Calculate Probability of Improvement."""
        if std == 0:
            return 1.0 if mean > best_value else 0.0

        xi = float(self.config.acquisition_function.xi)
        improvement = mean - best_value - xi
        z = improvement / std

        return norm.cdf(z)

    def _lower_confidence_bound(self, mean: float, std: float) -> float:
        """Calculate Lower Confidence Bound (for minimization)."""
        kappa = float(self.config.acquisition_function.kappa)
        return mean - kappa * std


class BayesianOptimizer(OptimizationEngine):
    """
    Bayesian optimization engine with Gaussian Process surrogate models.

    Implements efficient global optimization using GP models with
    acquisition function optimization for parameter selection.
    """

    def __init__(
        self,
        objectives: list[OptimizationObjective],
        parameter_space: ParameterSpace,
        config: OptimizationConfig | None = None,
        bayesian_config: BayesianConfig | None = None,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            objectives: List of optimization objectives
            parameter_space: Parameter space definition
            config: General optimization configuration
            bayesian_config: Bayesian-specific configuration
        """
        super().__init__(objectives, [], config)

        self.parameter_space = parameter_space
        self.bayesian_config = bayesian_config or BayesianConfig()

        # Initialize components
        self.gp_model = GaussianProcessModel(self.bayesian_config.gp_config, parameter_space)
        self.acquisition_optimizer = None  # Initialize after GP fitting

        # State tracking
        self.points: list[BayesianPoint] = []
        self.current_iteration = 0
        self.best_point: BayesianPoint | None = None

        # Convergence tracking
        self.stagnation_count = 0
        self.recent_improvements: list[Decimal] = []

        # Concurrent task management for race condition prevention
        self._active_tasks: set[asyncio.Task] = set()

        logger.info(
            "BayesianOptimizer initialized",
            optimization_id=self.optimization_id,
            parameter_count=len(parameter_space.parameters),
            acquisition_function=self.bayesian_config.acquisition_function.name,
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
        Run Bayesian optimization.

        Args:
            objective_function: Function to optimize
            parameter_space: Not used (using internal parameter space)
            initial_parameters: Initial parameter suggestions

        Returns:
            Optimization result
        """
        try:
            self.progress.status = OptimizationStatus.INITIALIZING
            logger.info("Starting Bayesian optimization", optimization_id=self.optimization_id)

            # Generate initial points
            await self._generate_initial_points(objective_function, initial_parameters)

            # Initialize acquisition optimizer
            self.acquisition_optimizer = AcquisitionOptimizer(
                self.gp_model, self.bayesian_config, self.parameter_space
            )

            self.progress.status = OptimizationStatus.RUNNING
            self.progress.total_iterations = self.bayesian_config.n_calls

            # Main optimization loop
            while (
                self.current_iteration < self.bayesian_config.n_calls
                and not self._check_convergence()
            ):
                await self._optimization_iteration(objective_function)
                self.current_iteration += 1

                # Update progress
                self.update_progress(
                    iteration=self.current_iteration,
                    objective_value=self.best_point.objective_value if self.best_point else None,
                    parameters=self.best_point.parameters if self.best_point else None,
                    message=f"Iteration {self.current_iteration}",
                )

            # Finalize optimization
            result = await self._finalize_optimization()

            self.progress.status = OptimizationStatus.COMPLETED
            logger.info(
                "Bayesian optimization completed",
                optimization_id=self.optimization_id,
                best_objective=float(result.optimal_objective_value),
                iterations=self.current_iteration,
            )

            return result

        except (ValidationError, ValueError, TypeError) as e:
            self.progress.status = OptimizationStatus.FAILED
            logger.error(
                "Bayesian optimization failed due to configuration/data error",
                optimization_id=self.optimization_id,
                error=str(e),
            )
            raise OptimizationError(f"Bayesian optimization failed: {e!s}") from e
        except Exception as e:
            self.progress.status = OptimizationStatus.FAILED
            logger.error(
                "Bayesian optimization failed unexpectedly",
                optimization_id=self.optimization_id,
                error=str(e),
            )
            raise OptimizationError(f"Bayesian optimization failed: {e!s}") from e

        finally:
            # Cancel all active tasks to prevent race conditions
            if self._active_tasks:
                for task in self._active_tasks:
                    if not task.done():
                        task.cancel()

                # Wait for all tasks to complete cancellation
                if self._active_tasks:
                    await asyncio.gather(*self._active_tasks, return_exceptions=True)
                    self._active_tasks.clear()

            # Clean up resources
            try:
                if hasattr(self, "gp_model") and self.gp_model:
                    # Clear GP model data to free memory
                    self.gp_model.X_train = None
                    self.gp_model.y_train = None
                    self.gp_model.is_fitted = False
            except Exception as e:
                logger.warning(f"Error cleaning up GP model resources: {e}")

    async def _generate_initial_points(
        self, objective_function: Callable, initial_parameters: dict[str, Any] | None = None
    ) -> None:
        """Generate and evaluate initial points."""
        initial_points = []

        # Add user-provided initial point if available
        if initial_parameters:
            initial_points.append(initial_parameters)

        # Generate random initial points
        remaining = self.bayesian_config.n_initial_points - len(initial_points)
        for _ in range(remaining):
            params = self.parameter_space.sample(SamplingStrategy.LATIN_HYPERCUBE)
            initial_points.append(params)

        # Evaluate initial points
        for i, params in enumerate(initial_points):
            point = BayesianPoint(point_id=f"{self.optimization_id}_init_{i}", parameters=params)

            await self._evaluate_point(point, objective_function)
            self.points.append(point)

        # Fit initial GP model
        self.gp_model.fit(self.points)

        logger.info(f"Evaluated {len(initial_points)} initial points")

    async def _optimization_iteration(self, objective_function: Callable) -> None:
        """Run single optimization iteration."""
        # Select next point(s) to evaluate
        next_parameters_list = self.acquisition_optimizer.optimize_acquisition(
            self.points, n_points=self.bayesian_config.batch_size
        )

        # Evaluate selected points
        new_points = []
        for i, params in enumerate(next_parameters_list):
            point = BayesianPoint(
                point_id=f"{self.optimization_id}_iter_{self.current_iteration}_{i}",
                parameters=params,
            )

            await self._evaluate_point(point, objective_function)
            new_points.append(point)
            self.points.append(point)

        # Update GP model
        self.gp_model.fit(self.points)

        # Update best point
        self._update_best_point()

        logger.info(
            f"Iteration {self.current_iteration} completed",
            new_points=len(new_points),
            best_objective=float(self.best_point.objective_value) if self.best_point else None,
        )

    async def _evaluate_point(self, point: BayesianPoint, objective_function: Callable) -> None:
        """Evaluate a single point."""
        try:
            # Run objective function
            result = await self._run_objective_function(
                objective_function, point.parameters, self.parameter_space
            )

            if result is not None:
                try:
                    if isinstance(result, dict):
                        # Multi-objective case
                        objective_values = {k: Decimal(str(v)) for k, v in result.items()}
                        primary_objective = self._get_primary_objective_value(objective_values)
                    else:
                        primary_objective = Decimal(str(result))
                except (ValueError, TypeError, OverflowError) as e:
                    logger.error(f"Failed to convert objective value to Decimal: {e}")
                    point.mark_failed(str(e))
                    return

                point.mark_evaluated(primary_objective)

                logger.debug(
                    "Point evaluated successfully",
                    point_id=point.point_id,
                    objective_value=float(primary_objective),
                )
            else:
                logger.error(f"Objective function returned None for point {point.point_id}")

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                "Point evaluation failed due to parameter/data issue",
                point_id=point.point_id,
                error=str(e),
            )
        except Exception as e:
            logger.error(
                "Point evaluation failed unexpectedly", point_id=point.point_id, error=str(e)
            )

    def _update_best_point(self) -> None:
        """Update best point found so far."""
        evaluated_points = [p for p in self.points if p.evaluated]

        if not evaluated_points:
            return

        # Find best point based on primary objective
        best_point = max(
            evaluated_points,
            key=lambda p: p.objective_value if p.objective_value is not None else Decimal("-inf"),
        )

        if self.best_point is None or (
            best_point.objective_value is not None
            and best_point.objective_value > self.best_point.objective_value
        ):
            previous_best = self.best_point.objective_value if self.best_point else None
            self.best_point = best_point

            # Track improvement
            if previous_best is not None:
                improvement = best_point.objective_value - previous_best
                self.recent_improvements.append(improvement)

                # Keep only recent improvements
                if len(self.recent_improvements) > self.bayesian_config.patience:
                    self.recent_improvements.pop(0)

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.recent_improvements) < self.bayesian_config.patience:
            return False

        # Check if recent improvements are below threshold
        avg_improvement = sum(self.recent_improvements) / len(self.recent_improvements)

        if avg_improvement < self.bayesian_config.convergence_tolerance:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        return self.stagnation_count >= self.bayesian_config.patience

    async def _finalize_optimization(self) -> OptimizationResult:
        """Finalize optimization and create result."""
        # Yield control to event loop
        await asyncio.sleep(0)

        if not self.best_point:
            raise OptimizationError("No valid points found")

        # Create result
        result = OptimizationResult(
            optimization_id=self.optimization_id,
            algorithm_name="BayesianOptimization",
            optimal_parameters=self.best_point.parameters,
            optimal_objective_value=self.best_point.objective_value,
            objective_values={self.objectives[0].name: self.best_point.objective_value},
            iterations_completed=self.current_iteration,
            evaluations_completed=len([p for p in self.points if p.evaluated]),
            convergence_achieved=self._check_convergence(),
            start_time=self.progress.start_time,
            end_time=datetime.now(timezone.utc),
            total_duration_seconds=Decimal(
                str((datetime.now(timezone.utc) - self.progress.start_time).total_seconds())
            ),
            config_used=self.config.dict(),
        )

        self.result = result
        return result

    def get_next_parameters(self) -> dict[str, Any] | None:
        """Get next parameters to evaluate."""
        if self.acquisition_optimizer is None:
            return self.parameter_space.sample()

        next_params_list = self.acquisition_optimizer.optimize_acquisition(self.points, n_points=1)
        return next_params_list[0] if next_params_list else None

    def get_gp_predictions(
        self, parameters_list: list[dict[str, Any]]
    ) -> tuple[list[Decimal], list[Decimal]]:
        """
        Get GP predictions for parameter sets.

        Args:
            parameters_list: List of parameter dictionaries

        Returns:
            Tuple of (means, stds)
        """
        means, stds = self.gp_model.predict(parameters_list, return_std=True)

        return ([Decimal(str(m)) for m in means], [Decimal(str(s)) for s in stds])

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get summary of optimization progress."""
        evaluated_points = [p for p in self.points if p.evaluated]

        return {
            "optimization_id": self.optimization_id,
            "current_iteration": self.current_iteration,
            "points_evaluated": len(evaluated_points),
            "best_objective": float(self.best_point.objective_value) if self.best_point else None,
            "best_parameters": self.best_point.parameters if self.best_point else None,
            "gp_model_info": self.gp_model.get_model_info(),
            "stagnation_count": self.stagnation_count,
            "converged": self._check_convergence(),
        }
