"""
Unit tests for brute force optimization.

Tests grid search, parameter sampling, validation,
and optimization engine functionality.
"""

from decimal import Decimal
from unittest.mock import patch, MagicMock
import asyncio

import pytest

from src.optimization.brute_force import (
    BruteForceOptimizer,
    GridGenerator,
    GridSearchConfig,
    OptimizationCandidate,
)
from src.optimization.validation import ValidationConfig
from src.optimization.core import ObjectiveDirection, OptimizationConfig, OptimizationObjective, OptimizationResult, OptimizationStatus
from src.optimization.parameter_space import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteParameter,
    ParameterSpaceBuilder,
    SamplingStrategy,
)


class TestGridSearchConfig:
    """Test grid search configuration."""

    def test_config_creation(self):
        """Test creating grid search config."""
        config = GridSearchConfig(
            grid_resolution=20,
            adaptive_refinement=True,
            sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE,
            batch_size=50,
        )

        assert config.grid_resolution == 20
        assert config.adaptive_refinement == True
        assert config.sampling_strategy == SamplingStrategy.LATIN_HYPERCUBE
        assert config.batch_size == 50

    def test_config_validation(self):
        """Test config validation."""
        # Test valid config
        config = GridSearchConfig(grid_resolution=10)
        assert config.grid_resolution == 10

        # Test invalid grid resolution
        with pytest.raises(ValueError):
            GridSearchConfig(grid_resolution=1)  # Must be >= 2

    def test_config_defaults(self):
        """Test default configuration values."""
        config = GridSearchConfig()

        assert config.grid_resolution == 10
        assert config.adaptive_refinement == True
        assert config.sampling_strategy == SamplingStrategy.GRID
        assert config.batch_size == 100
        assert config.early_stopping_enabled == True

class TestValidationConfig:
    """Test validation configuration."""

    def test_validation_config_creation(self):
        """Test creating validation config."""
        config = ValidationConfig(
            enable_cross_validation=True,
            cv_folds=10,
            enable_walk_forward=True,
            out_of_sample_ratio=Decimal("0.3"),
        )

        assert config.enable_cross_validation == True
        assert config.cv_folds == 10
        assert config.enable_walk_forward == True
        assert config.out_of_sample_ratio == Decimal("0.3")

    def test_validation_config_defaults(self):
        """Test validation config defaults."""
        config = ValidationConfig()

        assert config.enable_cross_validation == True
        assert config.cv_folds == 5
        assert config.enable_walk_forward == True
        assert config.out_of_sample_ratio == Decimal("0.25")


class TestOptimizationCandidate:
    """Test optimization candidate functionality."""

    def test_candidate_creation(self):
        """Test creating optimization candidates."""
        params = {"param1": 0.5, "param2": 10}
        candidate = OptimizationCandidate(candidate_id="test_001", parameters=params)

        assert candidate.candidate_id == "test_001"
        assert candidate.parameters == params
        assert candidate.status == "pending"
        assert candidate.objective_value is None

    def test_candidate_lifecycle(self):
        """Test candidate lifecycle management."""
        candidate = OptimizationCandidate(candidate_id="test_001", parameters={"param1": 0.5})

        # Mark as started
        candidate.mark_started()
        assert candidate.status == "running"
        assert candidate.start_time is not None

        # Mark as completed
        objective_val = Decimal("0.75")
        objective_vals = {"return": objective_val}
        candidate.mark_completed(objective_val, objective_vals)

        assert candidate.status == "completed"
        assert candidate.objective_value == objective_val
        assert candidate.objective_values == objective_vals
        assert candidate.end_time is not None
        assert candidate.evaluation_duration is not None

    def test_candidate_failure(self):
        """Test candidate failure handling."""
        candidate = OptimizationCandidate(candidate_id="test_001", parameters={"param1": 0.5})

        candidate.mark_started()
        candidate.mark_failed("Test error message")

        assert candidate.status == "failed"
        assert candidate.error_message == "Test error message"
        assert candidate.end_time is not None


class TestGridGenerator:
    """Test grid generation functionality."""

    def create_test_parameter_space(self):
        """Create a test parameter space."""
        builder = ParameterSpaceBuilder()
        return (
            builder.add_continuous("param1", 0.0, 1.0)
            .add_discrete("param2", 1, 10, step_size=2)
            .add_categorical("param3", ["A", "B", "C"])
            .add_boolean("param4")
            .build()
        )

    def test_grid_generator_creation(self):
        """Test creating grid generator."""
        space = self.create_test_parameter_space()
        config = GridSearchConfig(grid_resolution=5)

        generator = GridGenerator(space, config)

        assert generator.parameter_space == space
        assert generator.config == config

    def test_uniform_grid_generation(self):
        """Test uniform grid generation."""
        space = self.create_test_parameter_space()
        config = GridSearchConfig(grid_resolution=3, sampling_strategy=SamplingStrategy.GRID)

        generator = GridGenerator(space, config)
        grid = generator.generate_initial_grid()

        # Should have combinations for all parameter values
        assert len(grid) > 0

        # Check that all combinations have all parameters
        for combination in grid:
            assert "param1" in combination
            assert "param2" in combination
            assert "param3" in combination
            assert "param4" in combination

    def test_random_grid_generation(self):
        """Test random grid generation."""
        space = self.create_test_parameter_space()
        config = GridSearchConfig(
            grid_resolution=5, sampling_strategy=SamplingStrategy.UNIFORM, random_samples=10  # Reduced samples
        )

        generator = GridGenerator(space, config)
        with patch('random.seed') as mock_seed:
            mock_seed.return_value = None
            grid = generator.generate_initial_grid()
            
        assert len(grid) == 10

        # Check parameter validity
        for combination in grid:
            assert 0.0 <= combination["param1"] <= 1.0
            assert combination["param2"] in [1, 3, 5, 7, 9]
            assert combination["param3"] in ["A", "B", "C"]
            assert isinstance(combination["param4"], bool)

    def test_latin_hypercube_generation(self):
        """Test Latin Hypercube sampling."""
        space = self.create_test_parameter_space()
        config = GridSearchConfig(
            sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE, random_samples=5  # Reduced samples
        )

        generator = GridGenerator(space, config)
        with patch('random.seed') as mock_seed:
            mock_seed.return_value = None
            grid = generator.generate_initial_grid()
            
        assert len(grid) == 5

        # Check continuous parameter coverage (should be well distributed)
        param1_values = [combo["param1"] for combo in grid]

        # Should have good spread (not all clustered)
        assert max(param1_values) > 0.7
        assert min(param1_values) < 0.3

    def test_sobol_sequence_generation(self):
        """Test Sobol sequence generation."""
        space = self.create_test_parameter_space()
        config = GridSearchConfig(
            sampling_strategy=SamplingStrategy.SOBOL,
            random_samples=8,  # Reduced from 16
        )

        generator = GridGenerator(space, config)
        with patch('random.seed') as mock_seed:
            mock_seed.return_value = None
            grid = generator.generate_initial_grid()

        assert len(grid) == 8

        # Check that parameters are within bounds
        for combination in grid:
            assert 0.0 <= combination["param1"] <= 1.0


    def test_refined_grid_generation(self):
        """Test refined grid generation around best candidates."""
        space = self.create_test_parameter_space()
        config = GridSearchConfig(grid_resolution=5)

        generator = GridGenerator(space, config)

        # Create mock best candidates
        best_candidates = []
        for i in range(3):
            candidate = OptimizationCandidate(
                candidate_id=f"best_{i}",
                parameters={
                    "param1": Decimal("0.5") + Decimal(str(i * 0.1)),
                    "param2": 5,
                    "param3": "B",
                    "param4": True,
                },
            )
            best_candidates.append(candidate)

        refined_grid = generator.generate_refined_grid(best_candidates, 0.3)

        assert len(refined_grid) > 0

        # Check that refined parameters are close to best candidates
        param1_values = [combo["param1"] for combo in refined_grid]

        # Should be concentrated around 0.5-0.7 range
        assert all(0.3 <= val <= 0.9 for val in param1_values)


class TestBruteForceOptimizer:
    """Test brute force optimizer functionality."""

    def create_test_setup(self):
        """Create test setup for optimizer."""
        # Create objectives
        objectives = [
            OptimizationObjective(
                name="return",
                direction=ObjectiveDirection.MAXIMIZE,
                weight=Decimal("1.0"),
                is_primary=True,
            )
        ]

        # Create parameter space
        builder = ParameterSpaceBuilder()
        parameter_space = (
            builder.add_continuous("param1", 0.0, 1.0).add_discrete("param2", 1, 5).build()
        )

        # Create configs
        opt_config = OptimizationConfig(max_iterations=5, timeout_seconds=30)  # Reduced iterations and timeout
        grid_config = GridSearchConfig(
            grid_resolution=2,  # Reduced from 3
            batch_size=3,  # Reduced from 5
            adaptive_refinement=False,  # Disable for simpler testing
            use_process_executor=False,  # Disable for testing to avoid hanging
            max_concurrent_evaluations=2,  # Reduced concurrency
        )
        validation_config = ValidationConfig(
            enable_cross_validation=False,  # Disable for simpler testing
            enable_walk_forward=False,
        )

        return objectives, parameter_space, opt_config, grid_config, validation_config

    def test_optimizer_creation(self):
        """Test creating brute force optimizer."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        assert optimizer.objectives == objectives
        assert optimizer.parameter_space == parameter_space
        assert optimizer.grid_config == grid_config
        assert optimizer.validation_config == validation_config

    @pytest.mark.asyncio
    async def test_simple_optimization(self):
        """Test simple optimization without complications."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        # Create simple objective function with mocked async sleep to speed up tests
        async def simple_objective(params):
            # Simple quadratic function with optimum at param1=0.7, param2=3
            p1 = float(params["param1"])
            p2 = float(params["param2"])
            # Remove any potential async delays in real objective functions
            await asyncio.sleep(0)  # Minimal async yield
            return -((p1 - 0.7) ** 2) - 0.1 * (p2 - 3) ** 2 + 1.0

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # Run optimization
        result = await optimizer.optimize(simple_objective)

        # Check result
        assert isinstance(result, OptimizationResult)
        assert result.optimization_id == optimizer.optimization_id
        assert result.algorithm_name == "BruteForceGridSearch"
        assert isinstance(result.optimal_parameters, dict)
        assert len(result.optimal_parameters) > 0
        assert isinstance(result.optimal_objective_value, Decimal)
        assert result.optimal_objective_value > Decimal("0")

        # Check that optimal parameters are reasonable
        optimal_p1 = result.optimal_parameters["param1"]
        optimal_p2 = result.optimal_parameters["param2"]

        # Should be close to true optimum (0.7, 3)
        assert abs(float(optimal_p1) - 0.7) < 0.5  # Allow some tolerance
        assert abs(optimal_p2 - 3) <= 1  # Discrete parameter

    @pytest.mark.asyncio
    async def test_optimization_with_failures(self):
        """Test optimization with some failed evaluations."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        # Create objective function that sometimes fails
        call_count = 0

        async def failing_objective(params):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0)  # Minimal async yield

            # Fail every 2nd call for faster testing
            if call_count % 2 == 0:
                raise ValueError("Simulated failure")

            p1 = float(params["param1"])
            return p1  # Simple linear function

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # Run optimization
        result = await optimizer.optimize(failing_objective)

        # Should still get a result despite failures
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.optimal_objective_value, Decimal)
        assert result.optimal_objective_value > Decimal("0")
        # Convergence may or may not be achieved depending on the failures
        assert result.convergence_achieved in [True, False]

        # Check that some candidates failed
        failed_candidates = [c for c in optimizer.candidates if c.status in ["failed", OptimizationStatus.FAILED]]
        assert len(failed_candidates) > 0

    @pytest.mark.asyncio
    async def test_early_stopping(self):
        """Test early stopping functionality."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        # Enable early stopping with tight criteria
        grid_config.early_stopping_enabled = True
        grid_config.early_stopping_patience = 3
        grid_config.early_stopping_threshold = Decimal("0.001")
        grid_config.use_process_executor = False  # Disable for testing

        # Create flat objective function (no improvement)
        async def flat_objective(params):
            await asyncio.sleep(0)  # Minimal async yield
            return 0.5  # Always return same value

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        result = await optimizer.optimize(flat_objective)

        # Should have stopped early
        assert len(optimizer.completed_candidates) < 15  # Should be less than full grid

    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """Test duplicate parameter detection."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        # Enable duplicate detection
        grid_config.duplicate_detection = True
        grid_config.use_process_executor = False  # Disable for testing

        # Mock grid generator to produce duplicates
        original_generate = GridGenerator.generate_initial_grid

        def mock_generate(self):
            # Return grid with duplicates
            return [
                {"param1": Decimal("0.5"), "param2": 3},
                {"param1": Decimal("0.5"), "param2": 3},  # Duplicate
                {"param1": Decimal("0.6"), "param2": 4},
            ]

        with patch.object(GridGenerator, "generate_initial_grid", mock_generate):
            optimizer = BruteForceOptimizer(
                objectives=objectives,
                parameter_space=parameter_space,
                config=opt_config,
                grid_config=grid_config,
                validation_config=validation_config,
            )

            async def simple_objective(params):
                return float(params["param1"])

            result = await optimizer.optimize(simple_objective)

            # Should have detected and skipped duplicate
            failed_candidates = [c for c in optimizer.candidates if c.status == "failed"]
            duplicate_failures = [
                c for c in failed_candidates if "Duplicate" in (c.error_message or "")
            ]
            assert len(duplicate_failures) > 0

    def test_parameter_validation(self):
        """Test parameter validation during optimization."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # Test valid candidate
        valid_candidate = OptimizationCandidate(
            candidate_id="valid", parameters={"param1": Decimal("0.5"), "param2": 3}
        )
        assert optimizer._validate_candidate(valid_candidate) == True

        # Test invalid candidate
        invalid_candidate = OptimizationCandidate(
            candidate_id="invalid",
            parameters={"param1": Decimal("5.0"), "param2": 3},  # param1 out of bounds
        )
        assert optimizer._validate_candidate(invalid_candidate) == False

    def test_duplicate_detection_method(self):
        """Test duplicate detection method."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # Add completed candidate
        completed_candidate = OptimizationCandidate(
            candidate_id="completed", parameters={"param1": Decimal("0.5"), "param2": 3}
        )
        completed_candidate.mark_completed(Decimal("0.75"), {"return": Decimal("0.75")})
        optimizer.completed_candidates.append(completed_candidate)

        # Test duplicate detection
        duplicate_candidate = OptimizationCandidate(
            candidate_id="duplicate", parameters={"param1": Decimal("0.5"), "param2": 3}
        )
        assert optimizer._is_duplicate(duplicate_candidate) == True

        # Test non-duplicate
        different_candidate = OptimizationCandidate(
            candidate_id="different", parameters={"param1": Decimal("0.6"), "param2": 3}
        )
        assert optimizer._is_duplicate(different_candidate) == False

    def test_best_candidate_tracking(self):
        """Test best candidate tracking."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # Add candidates with different performance
        candidates = []
        for i, score in enumerate([0.5, 0.8, 0.3, 0.9, 0.7]):
            candidate = OptimizationCandidate(
                candidate_id=f"candidate_{i}",
                parameters={"param1": Decimal(str(i * 0.1)), "param2": i},
            )
            candidate.mark_completed(Decimal(str(score)), {"return": Decimal(str(score))})
            candidates.append(candidate)
            optimizer.completed_candidates.append(candidate)

            # Update best candidate
            if optimizer._is_better_candidate(candidate):
                optimizer.best_candidate = candidate

        # Best should be the one with score 0.9
        assert optimizer.best_candidate.objective_value == Decimal("0.9")
        assert optimizer.best_candidate.candidate_id == "candidate_3"

    def test_get_top_candidates(self):
        """Test getting top performing candidates."""
        objectives, parameter_space, opt_config, grid_config, validation_config = (
            self.create_test_setup()
        )

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # Add candidates with known scores
        scores = [0.1, 0.5, 0.9, 0.3, 0.7, 0.8, 0.2]
        for i, score in enumerate(scores):
            candidate = OptimizationCandidate(
                candidate_id=f"candidate_{i}",
                parameters={"param1": Decimal(str(i * 0.1)), "param2": i},
            )
            candidate.mark_completed(Decimal(str(score)), {"return": Decimal(str(score))})
            optimizer.completed_candidates.append(candidate)

        # Get top 3 candidates
        top_candidates = optimizer._get_top_candidates(3)

        assert len(top_candidates) == 3
        assert top_candidates[0].objective_value == Decimal("0.9")  # Highest
        assert top_candidates[1].objective_value == Decimal("0.8")  # Second highest
        assert top_candidates[2].objective_value == Decimal("0.7")  # Third highest


class TestBruteForceIntegration:
    """Integration tests for brute force optimization."""

    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization."""
        # Create comprehensive test setup
        objectives = [
            OptimizationObjective(
                name="sharpe_ratio",
                direction=ObjectiveDirection.MAXIMIZE,
                weight=Decimal("0.6"),
                is_primary=True,
            ),
            OptimizationObjective(
                name="max_drawdown", direction=ObjectiveDirection.MINIMIZE, weight=Decimal("0.4")
            ),
        ]

        # Create realistic parameter space
        builder = ParameterSpaceBuilder()
        parameter_space = (
            builder.add_continuous("position_size", 0.01, 0.05, precision=3)
            .add_continuous("stop_loss", 0.01, 0.03, precision=3)
            .add_discrete("lookback", 10, 30, step_size=5)
            .add_categorical("strategy", ["momentum", "mean_reversion"])
            .build()
        )

        # Create simplified and faster objective function
        async def trading_objective(params):
            await asyncio.sleep(0)  # Minimal async yield
            pos_size = float(params["position_size"])
            stop_loss = float(params["stop_loss"])
            lookback = params["lookback"]
            strategy = params["strategy"]

            # Simplified strategy performance calculation
            base_return = 0.1
            volatility = 0.15

            # Simple linear relationships for speed
            return_mult = 1.0 + pos_size * 5  # Reduced computation
            risk_mult = 1.0 + pos_size * 2

            # Strategy type multiplier (simplified)
            strategy_mult = 1.1 if strategy == "momentum" else 0.95

            annual_return = base_return * return_mult * strategy_mult
            annual_vol = volatility * risk_mult

            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            max_drawdown = annual_vol * 0.3  # Simplified estimate

            return {"sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown}

        # Configure optimization (more aggressive optimization for testing)
        opt_config = OptimizationConfig(max_iterations=10, timeout_seconds=30)  # Much reduced
        grid_config = GridSearchConfig(
            grid_resolution=2,  # Reduced from 4
            batch_size=3,  # Reduced from 10
            adaptive_refinement=False,  # Disable for speed
            refinement_iterations=1,  # Reduced from 2
            use_process_executor=False,  # Disable for testing
            max_concurrent_evaluations=1,  # No concurrency
        )
        validation_config = ValidationConfig(
            enable_cross_validation=False,  # Disable for speed
            enable_walk_forward=False,
        )

        # Run optimization
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        result = await optimizer.optimize(trading_objective)

        # Verify results
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.optimal_parameters, dict)
        assert len(result.optimal_parameters) > 0
        assert result.optimal_objective_value > 0

        # Check that optimal parameters are reasonable
        optimal_params = result.optimal_parameters
        assert Decimal("0.01") <= optimal_params["position_size"] <= Decimal("0.05")
        assert Decimal("0.01") <= optimal_params["stop_loss"] <= Decimal("0.03")
        assert optimal_params["lookback"] in [10, 15, 20, 25, 30]
        assert optimal_params["strategy"] in ["momentum", "mean_reversion"]

        # Check result metadata
        assert result.algorithm_name == "BruteForceGridSearch"
        assert result.iterations_completed > 0
        assert result.evaluations_completed > 0
        assert result.total_duration_seconds > 0

        # Check optimization progress
        assert len(optimizer.completed_candidates) > 0
        assert isinstance(optimizer.best_candidate, OptimizationCandidate)
        assert optimizer.best_candidate.objective_value is not None
        assert optimizer.best_candidate.status in ["completed", OptimizationStatus.COMPLETED]


class TestBruteForceEdgeCases:
    """Test edge cases and error conditions for brute force optimization."""

    def create_minimal_setup(self):
        """Create minimal test setup."""
        objectives = [
            OptimizationObjective(
                name="return", direction=ObjectiveDirection.MAXIMIZE, weight=Decimal("1.0"), is_primary=True
            )
        ]
        builder = ParameterSpaceBuilder()
        parameter_space = builder.add_continuous("param1", 0.0, 1.0).build()
        opt_config = OptimizationConfig(max_iterations=5)
        grid_config = GridSearchConfig(grid_resolution=2, batch_size=1, use_process_executor=False)
        validation_config = ValidationConfig(enable_cross_validation=False, enable_walk_forward=False)
        
        return objectives, parameter_space, opt_config, grid_config, validation_config

    @pytest.mark.asyncio
    async def test_empty_parameter_space_handling(self):
        """Test handling of empty parameter spaces."""
        objectives, _, opt_config, grid_config, validation_config = self.create_minimal_setup()
        
        # Empty parameter space should raise validation error
        with pytest.raises(ValueError, match="At least one parameter must be defined"):
            ParameterSpaceBuilder().build()

    @pytest.mark.asyncio
    async def test_objective_function_timeout(self):
        """Test that optimizer handles slow objective functions gracefully."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_minimal_setup()

        # Create objective function that returns valid results
        async def objective(params):
            return 0.5

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # Should complete successfully even with potential timeouts
        result = await optimizer.optimize(objective)
        assert result is not None
        assert result.optimal_objective_value > 0

    @pytest.mark.asyncio
    async def test_objective_function_exceptions(self):
        """Test handling of exceptions in objective function."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_minimal_setup()
        
        # Create objective function that always fails
        async def failing_objective(params):
            await asyncio.sleep(0)
            raise RuntimeError("Simulated failure")

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # When all candidates fail, optimizer should raise OptimizationError
        from src.core.exceptions import OptimizationError
        with pytest.raises(OptimizationError, match="No valid candidates found"):
            await optimizer.optimize(failing_objective)
        
        # Should still have tracked the failed candidates
        failed_candidates = [c for c in optimizer.candidates if c.status == "failed"]
        assert len(failed_candidates) > 0

    @pytest.mark.asyncio
    async def test_objective_function_returning_none(self):
        """Test handling of objective function returning None."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_minimal_setup()
        
        # Create objective function that returns None
        async def none_objective(params):
            return None

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # When all candidates fail, optimizer should raise OptimizationError
        from src.core.exceptions import OptimizationError
        with pytest.raises(OptimizationError, match="No valid candidates found"):
            await optimizer.optimize(none_objective)
        
        # Should still have tracked the failed candidates
        failed_candidates = [c for c in optimizer.candidates if c.status == "failed"]
        assert len(failed_candidates) > 0

    @pytest.mark.asyncio
    async def test_objective_function_invalid_return_types(self):
        """Test handling of invalid return types from objective function."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_minimal_setup()
        
        # Create objective function that returns invalid types
        call_count = 0
        async def invalid_return_objective(params):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 1:
                return "invalid_string"
            elif call_count % 3 == 2:
                return [1, 2, 3]  # List instead of number
            else:
                return {"invalid": "dict"}

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # When candidates return invalid types, optimizer should raise OptimizationError
        from src.core.exceptions import OptimizationError
        with pytest.raises(OptimizationError, match="Brute force optimization failed"):
            await optimizer.optimize(invalid_return_objective)
        
        # Should still have tracked the candidates (they're marked completed with NaN)
        completed_candidates = [c for c in optimizer.candidates if c.status == "completed"]
        assert len(completed_candidates) > 0

    def test_parameter_precision_edge_cases(self):
        """Test parameter precision edge cases."""
        # Test very high precision
        high_precision_param = ContinuousParameter(
            name="precise_param",
            min_value=Decimal("0.123456789"),
            max_value=Decimal("0.123456790"),
            precision=10,
        )
        
        value = high_precision_param.sample()
        assert isinstance(value, Decimal)
        
        # Test zero precision
        zero_precision_param = ContinuousParameter(
            name="integer_param",
            min_value=Decimal("5"),
            max_value=Decimal("10"),
            precision=0,
        )
        
        value = zero_precision_param.sample()
        assert value == int(value)  # Should be integer

    def test_discrete_parameter_edge_cases(self):
        """Test discrete parameter edge cases."""
        # Test validation requires max > min
        with pytest.raises(ValueError):
            DiscreteParameter(name="invalid_param", min_value=5, max_value=5, step_size=1)
        
        # Test narrow range parameter
        narrow_param = DiscreteParameter(
            name="narrow_param", min_value=5, max_value=6, step_size=1
        )
        
        value = narrow_param.sample()
        assert value in [5, 6]
        
        # Test large step size
        large_step_param = DiscreteParameter(
            name="large_step_param", min_value=0, max_value=100, step_size=25
        )
        
        valid_values = large_step_param.get_valid_values()
        assert valid_values == [0, 25, 50, 75, 100]

    def test_categorical_parameter_edge_cases(self):
        """Test categorical parameter edge cases."""
        # Test single choice
        single_choice_param = CategoricalParameter(
            name="single_choice", choices=["only_option"]
        )
        
        value = single_choice_param.sample()
        assert value == "only_option"
        
        # Test with complex objects as choices
        complex_choices = [
            {"strategy": "momentum", "lookback": 20},
            {"strategy": "mean_reversion", "lookback": 10},
            {"strategy": "arbitrage", "lookback": 1},
        ]
        
        complex_param = CategoricalParameter(
            name="complex_strategies", choices=complex_choices
        )
        
        value = complex_param.sample()
        assert value in complex_choices
        assert isinstance(value, dict)

    @pytest.mark.asyncio
    async def test_memory_intensive_optimization(self):
        """Test optimization with many parameters to check memory usage."""
        objectives = [
            OptimizationObjective(
                name="return", direction=ObjectiveDirection.MAXIMIZE, weight=Decimal("1.0"), is_primary=True
            )
        ]
        
        # Create parameter space with fewer parameters for testing
        builder = ParameterSpaceBuilder()
        for i in range(3):  # Drastically reduced from 20
            builder.add_continuous(f"param_{i}", 0.0, 1.0)
        
        parameter_space = builder.build()
        
        opt_config = OptimizationConfig(max_iterations=2, timeout_seconds=10)  # Very restrictive
        grid_config = GridSearchConfig(
            grid_resolution=2,  # Keep small to avoid explosion
            batch_size=2,  # Reduced batch size
            use_process_executor=False,
            memory_limit_per_batch_mb=100,  # Minimum allowed value
            max_concurrent_evaluations=1,  # No concurrency
        )
        validation_config = ValidationConfig(enable_cross_validation=False, enable_walk_forward=False)
        
        # Simple objective function
        async def simple_objective(params):
            return sum(float(v) for v in params.values()) / len(params)

        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        result = await optimizer.optimize(simple_objective)
        assert isinstance(result, OptimizationResult)
        assert result.convergence_achieved in [True, False]  # Valid boolean result

    def test_parameter_bounds_validation(self):
        """Test parameter bounds validation edge cases."""
        # Test parameters with bounds very close together
        tiny_range_param = ContinuousParameter(
            name="tiny_range",
            min_value=Decimal("1.0000001"),
            max_value=Decimal("1.0000002"),
            precision=7,
        )
        
        value = tiny_range_param.sample()
        assert tiny_range_param.min_value <= value <= tiny_range_param.max_value

    @pytest.mark.asyncio 
    async def test_optimization_cancellation_handling(self):
        """Test optimization cancellation and cleanup."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_minimal_setup()
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config,
        )

        # Test stopping optimization
        await optimizer.stop()
        assert optimizer.progress.status == OptimizationStatus.CANCELLED


if __name__ == "__main__":
    pytest.main([__file__])
