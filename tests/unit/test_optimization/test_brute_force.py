"""
Unit tests for brute force optimization.

Tests grid search, parameter sampling, validation,
and optimization engine functionality.
"""

import asyncio
import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.optimization.brute_force import (
    BruteForceOptimizer,
    GridSearchConfig,
    ValidationConfig,
    GridGenerator,
    OptimizationCandidate
)
from src.optimization.core import (
    OptimizationObjective,
    OptimizationConfig,
    ObjectiveDirection
)
from src.optimization.parameter_space import (
    ParameterSpace,
    ParameterSpaceBuilder,
    SamplingStrategy
)
from src.core.exceptions import OptimizationError


class TestGridSearchConfig:
    """Test grid search configuration."""
    
    def test_config_creation(self):
        """Test creating grid search config."""
        config = GridSearchConfig(
            grid_resolution=20,
            adaptive_refinement=True,
            sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE,
            batch_size=50
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
            out_of_sample_ratio=0.3
        )
        
        assert config.enable_cross_validation == True
        assert config.cv_folds == 10
        assert config.enable_walk_forward == True
        assert config.out_of_sample_ratio == 0.3
    
    def test_validation_config_defaults(self):
        """Test validation config defaults."""
        config = ValidationConfig()
        
        assert config.enable_cross_validation == True
        assert config.cv_folds == 5
        assert config.enable_walk_forward == True
        assert config.out_of_sample_ratio == 0.25


class TestOptimizationCandidate:
    """Test optimization candidate functionality."""
    
    def test_candidate_creation(self):
        """Test creating optimization candidates."""
        params = {"param1": 0.5, "param2": 10}
        candidate = OptimizationCandidate(
            candidate_id="test_001",
            parameters=params
        )
        
        assert candidate.candidate_id == "test_001"
        assert candidate.parameters == params
        assert candidate.status == "pending"
        assert candidate.objective_value is None
    
    def test_candidate_lifecycle(self):
        """Test candidate lifecycle management."""
        candidate = OptimizationCandidate(
            candidate_id="test_001",
            parameters={"param1": 0.5}
        )
        
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
        candidate = OptimizationCandidate(
            candidate_id="test_001",
            parameters={"param1": 0.5}
        )
        
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
        return (builder
                .add_continuous("param1", 0.0, 1.0)
                .add_discrete("param2", 1, 10, step_size=2)
                .add_categorical("param3", ["A", "B", "C"])
                .add_boolean("param4")
                .build())
    
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
        config = GridSearchConfig(
            grid_resolution=3,
            sampling_strategy=SamplingStrategy.GRID
        )
        
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
            grid_resolution=5,
            sampling_strategy=SamplingStrategy.UNIFORM,
            random_samples=20
        )
        
        generator = GridGenerator(space, config)
        grid = generator.generate_initial_grid()
        
        assert len(grid) == 20
        
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
            sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE,
            random_samples=10
        )
        
        generator = GridGenerator(space, config)
        grid = generator.generate_initial_grid()
        
        assert len(grid) == 10
        
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
            random_samples=16  # Power of 2 for Sobol
        )
        
        generator = GridGenerator(space, config)
        grid = generator.generate_initial_grid()
        
        assert len(grid) == 16
        
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
                    "param4": True
                }
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
                is_primary=True
            )
        ]
        
        # Create parameter space
        builder = ParameterSpaceBuilder()
        parameter_space = (builder
                          .add_continuous("param1", 0.0, 1.0)
                          .add_discrete("param2", 1, 5)
                          .build())
        
        # Create configs
        opt_config = OptimizationConfig(max_iterations=10)
        grid_config = GridSearchConfig(
            grid_resolution=3,
            batch_size=5,
            adaptive_refinement=False  # Disable for simpler testing
        )
        validation_config = ValidationConfig(
            enable_cross_validation=False,  # Disable for simpler testing
            enable_walk_forward=False
        )
        
        return objectives, parameter_space, opt_config, grid_config, validation_config
    
    def test_optimizer_creation(self):
        """Test creating brute force optimizer."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        assert optimizer.objectives == objectives
        assert optimizer.parameter_space == parameter_space
        assert optimizer.grid_config == grid_config
        assert optimizer.validation_config == validation_config
    
    @pytest.mark.asyncio
    async def test_simple_optimization(self):
        """Test simple optimization without complications."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        # Create simple objective function
        async def simple_objective(params):
            # Simple quadratic function with optimum at param1=0.7, param2=3
            p1 = float(params["param1"])
            p2 = float(params["param2"])
            
            return -(p1 - 0.7)**2 - 0.1 * (p2 - 3)**2 + 1.0
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        # Run optimization
        result = await optimizer.optimize(simple_objective)
        
        # Check result
        assert result is not None
        assert result.optimization_id == optimizer.optimization_id
        assert result.algorithm_name == "BruteForceGridSearch"
        assert result.optimal_parameters is not None
        assert result.optimal_objective_value is not None
        
        # Check that optimal parameters are reasonable
        optimal_p1 = result.optimal_parameters["param1"]
        optimal_p2 = result.optimal_parameters["param2"]
        
        # Should be close to true optimum (0.7, 3)
        assert abs(float(optimal_p1) - 0.7) < 0.5  # Allow some tolerance
        assert abs(optimal_p2 - 3) <= 1  # Discrete parameter
    
    @pytest.mark.asyncio
    async def test_optimization_with_failures(self):
        """Test optimization with some failed evaluations."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        # Create objective function that sometimes fails
        call_count = 0
        
        async def failing_objective(params):
            nonlocal call_count
            call_count += 1
            
            # Fail every 3rd call
            if call_count % 3 == 0:
                raise ValueError("Simulated failure")
            
            p1 = float(params["param1"])
            return p1  # Simple linear function
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        # Run optimization
        result = await optimizer.optimize(failing_objective)
        
        # Should still get a result despite failures
        assert result is not None
        assert result.optimal_objective_value is not None
        
        # Check that some candidates failed
        failed_candidates = [c for c in optimizer.candidates if c.status == "failed"]
        assert len(failed_candidates) > 0
    
    @pytest.mark.asyncio
    async def test_early_stopping(self):
        """Test early stopping functionality."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        # Enable early stopping with tight criteria
        grid_config.early_stopping_enabled = True
        grid_config.early_stopping_patience = 3
        grid_config.early_stopping_threshold = Decimal("0.001")
        
        # Create flat objective function (no improvement)
        async def flat_objective(params):
            return 0.5  # Always return same value
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        result = await optimizer.optimize(flat_objective)
        
        # Should have stopped early
        assert len(optimizer.completed_candidates) < 15  # Should be less than full grid
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """Test duplicate parameter detection."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        # Enable duplicate detection
        grid_config.duplicate_detection = True
        
        # Mock grid generator to produce duplicates
        original_generate = GridGenerator.generate_initial_grid
        
        def mock_generate(self):
            # Return grid with duplicates
            return [
                {"param1": Decimal("0.5"), "param2": 3},
                {"param1": Decimal("0.5"), "param2": 3},  # Duplicate
                {"param1": Decimal("0.6"), "param2": 4},
            ]
        
        with patch.object(GridGenerator, 'generate_initial_grid', mock_generate):
            optimizer = BruteForceOptimizer(
                objectives=objectives,
                parameter_space=parameter_space,
                config=opt_config,
                grid_config=grid_config,
                validation_config=validation_config
            )
            
            async def simple_objective(params):
                return float(params["param1"])
            
            result = await optimizer.optimize(simple_objective)
            
            # Should have detected and skipped duplicate
            failed_candidates = [c for c in optimizer.candidates if c.status == "failed"]
            duplicate_failures = [c for c in failed_candidates if "Duplicate" in (c.error_message or "")]
            assert len(duplicate_failures) > 0
    
    def test_parameter_validation(self):
        """Test parameter validation during optimization."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        # Test valid candidate
        valid_candidate = OptimizationCandidate(
            candidate_id="valid",
            parameters={"param1": Decimal("0.5"), "param2": 3}
        )
        assert optimizer._validate_candidate(valid_candidate) == True
        
        # Test invalid candidate
        invalid_candidate = OptimizationCandidate(
            candidate_id="invalid",
            parameters={"param1": Decimal("5.0"), "param2": 3}  # param1 out of bounds
        )
        assert optimizer._validate_candidate(invalid_candidate) == False
    
    def test_duplicate_detection_method(self):
        """Test duplicate detection method."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        # Add completed candidate
        completed_candidate = OptimizationCandidate(
            candidate_id="completed",
            parameters={"param1": Decimal("0.5"), "param2": 3}
        )
        completed_candidate.mark_completed(Decimal("0.75"), {"return": Decimal("0.75")})
        optimizer.completed_candidates.append(completed_candidate)
        
        # Test duplicate detection
        duplicate_candidate = OptimizationCandidate(
            candidate_id="duplicate",
            parameters={"param1": Decimal("0.5"), "param2": 3}
        )
        assert optimizer._is_duplicate(duplicate_candidate) == True
        
        # Test non-duplicate
        different_candidate = OptimizationCandidate(
            candidate_id="different",
            parameters={"param1": Decimal("0.6"), "param2": 3}
        )
        assert optimizer._is_duplicate(different_candidate) == False
    
    def test_best_candidate_tracking(self):
        """Test best candidate tracking."""
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        # Add candidates with different performance
        candidates = []
        for i, score in enumerate([0.5, 0.8, 0.3, 0.9, 0.7]):
            candidate = OptimizationCandidate(
                candidate_id=f"candidate_{i}",
                parameters={"param1": Decimal(str(i * 0.1)), "param2": i}
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
        objectives, parameter_space, opt_config, grid_config, validation_config = self.create_test_setup()
        
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        # Add candidates with known scores
        scores = [0.1, 0.5, 0.9, 0.3, 0.7, 0.8, 0.2]
        for i, score in enumerate(scores):
            candidate = OptimizationCandidate(
                candidate_id=f"candidate_{i}",
                parameters={"param1": Decimal(str(i * 0.1)), "param2": i}
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
                is_primary=True
            ),
            OptimizationObjective(
                name="max_drawdown",
                direction=ObjectiveDirection.MINIMIZE,
                weight=Decimal("0.4")
            )
        ]
        
        # Create realistic parameter space
        builder = ParameterSpaceBuilder()
        parameter_space = (builder
                          .add_continuous("position_size", 0.01, 0.05, precision=3)
                          .add_continuous("stop_loss", 0.01, 0.03, precision=3)
                          .add_discrete("lookback", 10, 30, step_size=5)
                          .add_categorical("strategy", ["momentum", "mean_reversion"])
                          .build())
        
        # Create realistic objective function
        async def trading_objective(params):
            pos_size = float(params["position_size"])
            stop_loss = float(params["stop_loss"])
            lookback = params["lookback"]
            strategy = params["strategy"]
            
            # Simulate strategy performance
            base_return = 0.1
            volatility = 0.15
            
            # Position size affects both return and risk
            return_mult = 1.0 + pos_size * 10
            risk_mult = 1.0 + pos_size * 5
            
            # Stop loss reduces risk but also return
            return_mult *= (1.0 - stop_loss * 2)
            risk_mult *= (1.0 - stop_loss * 3)
            
            # Lookback affects performance
            return_mult *= (1.0 + (lookback - 20) * 0.01)
            
            # Strategy type affects base performance
            if strategy == "momentum":
                return_mult *= 1.1
            else:
                return_mult *= 0.95
                risk_mult *= 0.9
            
            annual_return = base_return * return_mult
            annual_vol = volatility * risk_mult
            
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            max_drawdown = annual_vol * 0.5  # Simplified drawdown estimate
            
            return {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown
            }
        
        # Configure optimization
        opt_config = OptimizationConfig(max_iterations=50)
        grid_config = GridSearchConfig(
            grid_resolution=4,  # Small for testing
            batch_size=10,
            adaptive_refinement=True,
            refinement_iterations=2
        )
        validation_config = ValidationConfig(
            enable_cross_validation=False,  # Disable for speed
            enable_walk_forward=False
        )
        
        # Run optimization
        optimizer = BruteForceOptimizer(
            objectives=objectives,
            parameter_space=parameter_space,
            config=opt_config,
            grid_config=grid_config,
            validation_config=validation_config
        )
        
        result = await optimizer.optimize(trading_objective)
        
        # Verify results
        assert result is not None
        assert result.optimal_parameters is not None
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
        assert optimizer.best_candidate is not None


if __name__ == "__main__":
    pytest.main([__file__])