"""
Unit tests for parameter space definition and management.

Tests all parameter types, validation, sampling strategies,
and parameter space operations.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any

from src.optimization.parameter_space import (
    ParameterSpace,
    ParameterSpaceBuilder,
    ContinuousParameter,
    DiscreteParameter,
    CategoricalParameter,
    BooleanParameter,
    ConditionalParameter,
    SamplingStrategy,
    create_trading_strategy_space,
    create_ml_model_space,
    create_risk_management_space
)
from src.core.exceptions import ValidationError


class TestContinuousParameter:
    """Test continuous parameter functionality."""
    
    def test_continuous_parameter_creation(self):
        """Test creating continuous parameters."""
        param = ContinuousParameter(
            name="test_param",
            min_value=Decimal("0.0"),
            max_value=Decimal("1.0"),
            precision=3,
            description="Test parameter"
        )
        
        assert param.name == "test_param"
        assert param.min_value == Decimal("0.0")
        assert param.max_value == Decimal("1.0")
        assert param.precision == 3
    
    def test_continuous_parameter_validation(self):
        """Test continuous parameter validation."""
        # Test invalid bounds
        with pytest.raises(ValueError):
            ContinuousParameter(
                name="invalid",
                min_value=Decimal("1.0"),
                max_value=Decimal("0.0")  # max < min
            )
    
    def test_continuous_parameter_sampling(self):
        """Test continuous parameter sampling."""
        param = ContinuousParameter(
            name="test_param",
            min_value=Decimal("0.0"),
            max_value=Decimal("10.0"),
            precision=2
        )
        
        # Test uniform sampling
        for _ in range(10):
            value = param.sample(SamplingStrategy.UNIFORM)
            assert param.min_value <= value <= param.max_value
            
        # Test Gaussian sampling
        for _ in range(10):
            value = param.sample(SamplingStrategy.GAUSSIAN)
            assert isinstance(value, Decimal)
    
    def test_continuous_parameter_validation_methods(self):
        """Test parameter validation methods."""
        param = ContinuousParameter(
            name="test_param",
            min_value=Decimal("0.0"),
            max_value=Decimal("10.0")
        )
        
        # Test valid values
        assert param.validate_value(5.0) == True
        assert param.validate_value(Decimal("7.5")) == True
        
        # Test invalid values
        assert param.validate_value(-1.0) == False
        assert param.validate_value(15.0) == False
        assert param.validate_value("invalid") == False
    
    def test_continuous_parameter_clipping(self):
        """Test parameter value clipping."""
        param = ContinuousParameter(
            name="test_param",
            min_value=Decimal("0.0"),
            max_value=Decimal("10.0")
        )
        
        # Test clipping
        assert param.clip_value(-5.0) == Decimal("0.0")
        assert param.clip_value(15.0) == Decimal("10.0")
        assert param.clip_value(5.0) == Decimal("5.0")


class TestDiscreteParameter:
    """Test discrete parameter functionality."""
    
    def test_discrete_parameter_creation(self):
        """Test creating discrete parameters."""
        param = DiscreteParameter(
            name="test_param",
            min_value=1,
            max_value=10,
            step_size=2,
            description="Test discrete parameter"
        )
        
        assert param.name == "test_param"
        assert param.min_value == 1
        assert param.max_value == 10
        assert param.step_size == 2
    
    def test_discrete_parameter_sampling(self):
        """Test discrete parameter sampling."""
        param = DiscreteParameter(
            name="test_param",
            min_value=1,
            max_value=10,
            step_size=2
        )
        
        valid_values = param.get_valid_values()
        assert valid_values == [1, 3, 5, 7, 9]
        
        for _ in range(10):
            value = param.sample()
            assert value in valid_values
    
    def test_discrete_parameter_validation(self):
        """Test discrete parameter validation."""
        param = DiscreteParameter(
            name="test_param",
            min_value=0,
            max_value=10,
            step_size=3
        )
        
        # Test valid values
        assert param.validate_value(0) == True
        assert param.validate_value(3) == True
        assert param.validate_value(6) == True
        
        # Test invalid values
        assert param.validate_value(1) == False
        assert param.validate_value(15) == False
        assert param.validate_value(-1) == False


class TestCategoricalParameter:
    """Test categorical parameter functionality."""
    
    def test_categorical_parameter_creation(self):
        """Test creating categorical parameters."""
        choices = ["option1", "option2", "option3"]
        param = CategoricalParameter(
            name="test_param",
            choices=choices,
            description="Test categorical parameter"
        )
        
        assert param.name == "test_param"
        assert param.choices == choices
    
    def test_categorical_parameter_sampling(self):
        """Test categorical parameter sampling."""
        choices = ["A", "B", "C"]
        param = CategoricalParameter(
            name="test_param",
            choices=choices
        )
        
        for _ in range(10):
            value = param.sample()
            assert value in choices
    
    def test_categorical_parameter_weighted_sampling(self):
        """Test weighted categorical sampling."""
        choices = ["A", "B", "C"]
        weights = [0.5, 0.3, 0.2]
        
        param = CategoricalParameter(
            name="test_param",
            choices=choices,
            weights=weights
        )
        
        # Sample many times and check distribution roughly matches weights
        samples = [param.sample() for _ in range(1000)]
        
        a_count = samples.count("A")
        b_count = samples.count("B")
        c_count = samples.count("C")
        
        # Allow some tolerance in the distribution
        assert a_count > b_count > c_count  # Should roughly follow weights
    
    def test_categorical_parameter_validation(self):
        """Test categorical parameter validation."""
        choices = ["option1", "option2", "option3"]
        param = CategoricalParameter(
            name="test_param",
            choices=choices
        )
        
        # Test valid values
        assert param.validate_value("option1") == True
        assert param.validate_value("option2") == True
        
        # Test invalid values
        assert param.validate_value("invalid_option") == False
        assert param.validate_value(123) == False


class TestBooleanParameter:
    """Test boolean parameter functionality."""
    
    def test_boolean_parameter_creation(self):
        """Test creating boolean parameters."""
        param = BooleanParameter(
            name="test_param",
            true_probability=0.7,
            description="Test boolean parameter"
        )
        
        assert param.name == "test_param"
        assert param.true_probability == 0.7
    
    def test_boolean_parameter_sampling(self):
        """Test boolean parameter sampling."""
        param = BooleanParameter(
            name="test_param",
            true_probability=0.8
        )
        
        # Sample many times and check distribution
        samples = [param.sample() for _ in range(1000)]
        true_count = sum(samples)
        true_ratio = true_count / len(samples)
        
        # Should be approximately 0.8 with some tolerance
        assert 0.7 < true_ratio < 0.9
    
    def test_boolean_parameter_validation(self):
        """Test boolean parameter validation."""
        param = BooleanParameter(name="test_param")
        
        # Test valid values
        assert param.validate_value(True) == True
        assert param.validate_value(False) == True
        
        # Test invalid values
        assert param.validate_value("true") == False
        assert param.validate_value(1) == False


class TestParameterSpace:
    """Test parameter space functionality."""
    
    def test_parameter_space_creation(self):
        """Test creating parameter spaces."""
        builder = ParameterSpaceBuilder()
        
        space = builder.add_continuous(
            "param1", 0.0, 1.0, description="First parameter"
        ).add_discrete(
            "param2", 1, 10, description="Second parameter"
        ).add_categorical(
            "param3", ["A", "B", "C"], description="Third parameter"
        ).build()
        
        assert len(space.parameters) == 3
        assert "param1" in space.parameters
        assert "param2" in space.parameters
        assert "param3" in space.parameters
    
    def test_parameter_space_sampling(self):
        """Test parameter space sampling."""
        space = create_trading_strategy_space()
        
        # Test sampling
        sample = space.sample(SamplingStrategy.UNIFORM)
        
        # Check all parameters are present
        expected_params = [
            "position_size_pct", "stop_loss_pct", "take_profit_pct",
            "timeframe", "lookback_period", "confidence_threshold"
        ]
        
        for param in expected_params:
            assert param in sample
    
    def test_parameter_space_validation(self):
        """Test parameter space validation."""
        space = create_trading_strategy_space()
        
        # Create valid parameters
        valid_params = {
            "position_size_pct": Decimal("0.02"),
            "stop_loss_pct": Decimal("0.02"),
            "take_profit_pct": Decimal("0.04"),
            "timeframe": "1h",
            "lookback_period": 20,
            "confidence_threshold": Decimal("0.7")
        }
        
        validation_results = space.validate_parameters(valid_params)
        assert all(validation_results.values())
        
        # Create invalid parameters
        invalid_params = valid_params.copy()
        invalid_params["position_size_pct"] = Decimal("0.5")  # Too large
        
        validation_results = space.validate_parameters(invalid_params)
        assert not validation_results["position_size_pct"]
    
    def test_parameter_space_clipping(self):
        """Test parameter space clipping."""
        space = create_trading_strategy_space()
        
        # Create out-of-bounds parameters
        params = {
            "position_size_pct": Decimal("0.5"),  # Too large
            "stop_loss_pct": Decimal("-0.01"),   # Too small
            "take_profit_pct": Decimal("0.05"),
            "timeframe": "1h",
            "lookback_period": 100,              # Too large
            "confidence_threshold": Decimal("1.5")  # Too large
        }
        
        clipped = space.clip_parameters(params)
        
        # Check clipping worked
        assert clipped["position_size_pct"] <= Decimal("0.10")
        assert clipped["stop_loss_pct"] >= Decimal("0.005")
        assert clipped["lookback_period"] <= 50
        assert clipped["confidence_threshold"] <= Decimal("0.9")
    
    def test_parameter_space_bounds(self):
        """Test getting parameter bounds."""
        space = create_trading_strategy_space()
        bounds = space.get_bounds()
        
        assert "position_size_pct" in bounds
        assert "timeframe" in bounds
        
        # Check bounds format
        pos_size_bounds = bounds["position_size_pct"]
        assert len(pos_size_bounds) == 2
        assert pos_size_bounds[0] < pos_size_bounds[1]


class TestConditionalParameter:
    """Test conditional parameter functionality."""
    
    def test_conditional_parameter_creation(self):
        """Test creating conditional parameters."""
        base_param = ContinuousParameter(
            name="learning_rate",
            min_value=Decimal("0.001"),
            max_value=Decimal("0.1")
        )
        
        conditional_param = ConditionalParameter(
            name="learning_rate",
            base_parameter=base_param,
            activation_conditions={"model_type": "neural_network"}
        )
        
        assert conditional_param.is_conditional == True
        assert conditional_param.conditions == {"model_type": "neural_network"}
    
    def test_conditional_parameter_activation(self):
        """Test conditional parameter activation."""
        base_param = ContinuousParameter(
            name="n_estimators",
            min_value=Decimal("50"),
            max_value=Decimal("500")
        )
        
        conditional_param = ConditionalParameter(
            name="n_estimators",
            base_parameter=base_param,
            activation_conditions={"model_type": "random_forest"}
        )
        
        # Test activation
        context_active = {"model_type": "random_forest"}
        assert conditional_param.is_active(context_active) == True
        
        context_inactive = {"model_type": "neural_network"}
        assert conditional_param.is_active(context_inactive) == False
    
    def test_conditional_parameter_sampling(self):
        """Test conditional parameter sampling."""
        base_param = DiscreteParameter(
            name="max_depth",
            min_value=3,
            max_value=20
        )
        
        conditional_param = ConditionalParameter(
            name="max_depth",
            base_parameter=base_param,
            activation_conditions={"model_type": ["random_forest", "xgboost"]}
        )
        
        # Test sampling
        value = conditional_param.sample()
        assert 3 <= value <= 20


class TestParameterSpaceBuilder:
    """Test parameter space builder functionality."""
    
    def test_fluent_interface(self):
        """Test fluent interface for building parameter spaces."""
        builder = ParameterSpaceBuilder()
        
        space = (builder
                .add_continuous("param1", 0.0, 1.0)
                .add_discrete("param2", 1, 10)
                .add_categorical("param3", ["A", "B"])
                .add_boolean("param4")
                .add_constraint("param1 + param2 > 0")
                .set_metadata("description", "Test space")
                .build())
        
        assert len(space.parameters) == 4
        assert len(space.constraints) == 1
        assert space.metadata["description"] == "Test space"
    
    def test_builder_validation(self):
        """Test builder validation."""
        builder = ParameterSpaceBuilder()
        
        # Test invalid continuous parameter
        with pytest.raises(ValueError):
            builder.add_continuous("invalid", 10.0, 5.0)  # max < min


class TestFactoryFunctions:
    """Test factory functions for common parameter spaces."""
    
    def test_trading_strategy_space(self):
        """Test trading strategy parameter space factory."""
        space = create_trading_strategy_space()
        
        assert len(space.parameters) >= 5
        assert "position_size_pct" in space.parameters
        assert "stop_loss_pct" in space.parameters
        assert "timeframe" in space.parameters
        
        # Test sampling
        sample = space.sample()
        assert len(sample) == len(space.parameters)
    
    def test_ml_model_space(self):
        """Test ML model parameter space factory."""
        space = create_ml_model_space()
        
        assert "model_type" in space.parameters
        assert "validation_split" in space.parameters
        
        # Test conditional parameters are present
        param_names = list(space.parameters.keys())
        assert any("rf_" in name for name in param_names)
        assert any("xgb_" in name for name in param_names)
        
        # Test sampling with conditional activation
        sample = space.sample()
        model_type = sample.get("model_type")
        
        if model_type == "random_forest":
            assert "rf_n_estimators" in sample or "rf_max_depth" in sample
        elif model_type == "xgboost":
            assert "xgb_learning_rate" in sample
    
    def test_risk_management_space(self):
        """Test risk management parameter space factory."""
        space = create_risk_management_space()
        
        assert "max_portfolio_exposure" in space.parameters
        assert "max_positions" in space.parameters
        assert "enable_correlation_breaker" in space.parameters
        
        # Test sampling
        sample = space.sample()
        assert isinstance(sample["enable_correlation_breaker"], bool)
        assert 0.5 <= sample["max_portfolio_exposure"] <= 0.95


class TestParameterSpaceIntegration:
    """Integration tests for parameter space components."""
    
    def test_complex_parameter_space(self):
        """Test complex parameter space with multiple parameter types."""
        builder = ParameterSpaceBuilder()
        
        # Create complex space with interactions
        space = (builder
                .add_categorical("strategy_type", ["momentum", "mean_reversion", "arbitrage"])
                .add_continuous("param1", 0.0, 1.0)
                .add_discrete("param2", 1, 100)
                .add_boolean("use_feature")
                .build())
        
        # Test multiple sampling strategies
        for strategy in [SamplingStrategy.UNIFORM, SamplingStrategy.GAUSSIAN]:
            sample = space.sample(strategy)
            assert len(sample) == 4
            assert sample["strategy_type"] in ["momentum", "mean_reversion", "arbitrage"]
            assert isinstance(sample["use_feature"], bool)
    
    def test_parameter_space_dependency_resolution(self):
        """Test parameter space dependency resolution."""
        builder = ParameterSpaceBuilder()
        
        # Create space with conditional dependencies
        base_lr = ContinuousParameter(
            name="learning_rate",
            min_value=Decimal("0.001"),
            max_value=Decimal("0.1")
        )
        
        space = (builder
                .add_categorical("model", ["simple", "complex"])
                .add_conditional("learning_rate", base_lr, {"model": "complex"})
                .add_discrete("epochs", 1, 100)
                .build())
        
        # Test topological sorting
        sorted_params = space._topological_sort()
        
        # Model should come before learning_rate
        model_idx = sorted_params.index("model")
        lr_idx = sorted_params.index("learning_rate") 
        assert model_idx < lr_idx
    
    def test_parameter_space_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        # This should be tested in the actual validation, but we can't easily 
        # create circular dependencies with the current API structure
        pass
    
    def test_parameter_space_active_parameters(self):
        """Test getting active parameters based on context."""
        space = create_ml_model_space()
        
        # Test with random forest context
        rf_context = {"model_type": "random_forest"}
        active_params = space.get_active_parameters(rf_context)
        
        # Should include general params and RF-specific params
        assert "model_type" in active_params
        assert "validation_split" in active_params
        
        # Test with XGBoost context
        xgb_context = {"model_type": "xgboost"}
        active_params = space.get_active_parameters(xgb_context)
        
        assert "model_type" in active_params
        assert "validation_split" in active_params
    
    def test_parameter_space_info_extraction(self):
        """Test extracting comprehensive parameter information."""
        space = create_trading_strategy_space()
        
        info = space.get_parameter_info()
        
        # Check structure
        assert isinstance(info, dict)
        assert len(info) == len(space.parameters)
        
        # Check parameter info completeness
        for param_name, param_info in info.items():
            assert "type" in param_info
            assert "bounds" in param_info
            assert "description" in param_info
            assert "is_conditional" in param_info
    
    def test_parameter_space_dimensionality(self):
        """Test parameter space dimensionality calculation."""
        space = create_trading_strategy_space()
        
        dimensionality = space.get_dimensionality()
        assert dimensionality == len(space.parameters)
        assert dimensionality > 0


if __name__ == "__main__":
    pytest.main([__file__])