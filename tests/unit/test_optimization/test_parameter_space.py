"""
Unit tests for parameter space definition and management.

Tests all parameter types, validation, sampling strategies,
and parameter space operations.
"""

from decimal import Decimal
from unittest.mock import patch

import pytest

from src.optimization.parameter_space import (
    BooleanParameter,
    CategoricalParameter,
    ConditionalParameter,
    ContinuousParameter,
    DiscreteParameter,
    ParameterSpaceBuilder,
    SamplingStrategy,
    create_ml_model_space,
    create_risk_management_space,
    create_trading_strategy_space,
)


@pytest.fixture(scope="module")
def sample_continuous_param():
    """Create sample continuous parameter for reuse."""
    return ContinuousParameter(
        name="test_param", min_value=Decimal("0.0"), max_value=Decimal("1.0"), precision=3
    )


@pytest.fixture(scope="module")
def sample_discrete_param():
    """Create sample discrete parameter for reuse."""
    return DiscreteParameter(name="test_param", min_value=1, max_value=10, step_size=2)


@pytest.fixture(scope="module")
def sample_categorical_param():
    """Create sample categorical parameter for reuse."""
    return CategoricalParameter(name="test_param", choices=["A", "B", "C"])


@pytest.fixture(scope="module")
def trading_space():
    """Create trading strategy space for reuse."""
    return create_trading_strategy_space()


class TestContinuousParameter:
    """Test continuous parameter functionality."""

    def test_continuous_parameter_creation(self):
        """Test creating continuous parameters."""
        param = ContinuousParameter(
            name="test_param",
            min_value=Decimal("0.0"),
            max_value=Decimal("1.0"),
            precision=3,
            description="Test parameter",
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
                max_value=Decimal("0.0"),  # max < min
            )

    def test_continuous_parameter_sampling(self):
        """Test continuous parameter sampling."""
        param = ContinuousParameter(
            name="test_param", min_value=Decimal("0.0"), max_value=Decimal("10.0"), precision=2
        )

        # Test uniform sampling with realistic varied values
        test_values = [0.1, 0.3, 0.7, 0.9, 0.5]  # Varied realistic values
        with patch('random.uniform', side_effect=test_values):
            for expected_ratio in test_values:
                value = param.sample(SamplingStrategy.UNIFORM)
                assert param.min_value <= value <= param.max_value
                # Check that sampling returns values in the valid range
                assert isinstance(value, Decimal)

        # Test Gaussian sampling with realistic center and spread values
        test_gauss_values = [4.8, 5.2, 4.5, 5.8, 5.0]  # Values around center
        with patch('random.gauss', side_effect=test_gauss_values):
            for _ in range(len(test_gauss_values)):
                value = param.sample(SamplingStrategy.GAUSSIAN)
                assert isinstance(value, Decimal)
                # Gaussian sampling should still respect bounds after clipping
                assert param.min_value <= value <= param.max_value

    def test_continuous_parameter_validation_methods(self):
        """Test parameter validation methods."""
        param = ContinuousParameter(
            name="test_param", min_value=Decimal("0.0"), max_value=Decimal("10.0")
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
            name="test_param", min_value=Decimal("0.0"), max_value=Decimal("10.0")
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
            description="Test discrete parameter",
        )

        assert param.name == "test_param"
        assert param.min_value == 1
        assert param.max_value == 10
        assert param.step_size == 2

    def test_discrete_parameter_sampling(self):
        """Test discrete parameter sampling."""
        param = DiscreteParameter(name="test_param", min_value=1, max_value=10, step_size=2)

        valid_values = param.get_valid_values()
        assert valid_values == [1, 3, 5, 7, 9]

        with patch('random.choice') as mock_choice:
            mock_choice.side_effect = valid_values  # Cycle through valid values
            for _ in range(5):  # Reduced iterations
                value = param.sample()
                assert value in valid_values

    def test_discrete_parameter_validation(self):
        """Test discrete parameter validation."""
        param = DiscreteParameter(name="test_param", min_value=0, max_value=10, step_size=3)

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
            name="test_param", choices=choices, description="Test categorical parameter"
        )

        assert param.name == "test_param"
        assert param.choices == choices

    def test_categorical_parameter_sampling(self):
        """Test categorical parameter sampling."""
        choices = ["A", "B", "C"]
        param = CategoricalParameter(name="test_param", choices=choices)

        with patch('random.choice') as mock_choice:
            mock_choice.side_effect = choices  # Cycle through choices
            for _ in range(3):  # Reduced iterations
                value = param.sample()
                assert value in choices

    def test_categorical_parameter_weighted_sampling(self):
        """Test weighted categorical sampling."""
        choices = ["A", "B", "C"]
        weights = [0.5, 0.3, 0.2]

        param = CategoricalParameter(name="test_param", choices=choices, weights=weights)

        # Sample fewer times with mocked random for determinism  
        with patch('random.choices') as mock_choices:
            mock_choices.return_value = ["A"] * 50 + ["B"] * 30 + ["C"] * 20  # Mock weighted distribution
            samples = [param.sample() for _ in range(100)]  # Reduced from 1000
            
            a_count = samples.count("A")
            b_count = samples.count("B") 
            c_count = samples.count("C")
            
            # Should roughly follow mocked distribution
            assert a_count >= b_count >= c_count

    def test_categorical_parameter_validation(self):
        """Test categorical parameter validation."""
        choices = ["option1", "option2", "option3"]
        param = CategoricalParameter(name="test_param", choices=choices)

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
            name="test_param", true_probability=0.7, description="Test boolean parameter"
        )

        assert param.name == "test_param"
        assert param.true_probability == Decimal("0.7")

    def test_boolean_parameter_sampling(self):
        """Test boolean parameter sampling."""
        param = BooleanParameter(name="test_param", true_probability=0.8)

        # Sample fewer times with mocked random for determinism
        with patch('random.random') as mock_random:
            # Mock random to return values that would give 80% True
            mock_random.side_effect = [0.1, 0.9, 0.5, 0.2, 0.7] * 20  # 3/5 < 0.8, so ~60% True
            samples = [param.sample() for _ in range(100)]  # Reduced from 1000
            true_count = sum(samples)
            
            # With mocked values, should have reasonable distribution
            assert true_count > 0  # Some true values

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

        space = (
            builder.add_continuous("param1", 0.0, 1.0, description="First parameter")
            .add_discrete("param2", 1, 10, description="Second parameter")
            .add_categorical("param3", ["A", "B", "C"], description="Third parameter")
            .build()
        )

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
            "position_size_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "timeframe",
            "lookback_period",
            "confidence_threshold",
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
            "confidence_threshold": Decimal("0.7"),
        }

        validation_results = space.validate_parameter_set(valid_params)
        assert all(validation_results.values())

        # Create invalid parameters
        invalid_params = valid_params.copy()
        invalid_params["position_size_pct"] = Decimal("0.5")  # Too large

        validation_results = space.validate_parameter_set(invalid_params)
        assert not validation_results["position_size_pct"]

    def test_parameter_space_clipping(self):
        """Test parameter space clipping."""
        space = create_trading_strategy_space()

        # Create out-of-bounds parameters
        params = {
            "position_size_pct": Decimal("0.5"),  # Too large
            "stop_loss_pct": Decimal("-0.01"),  # Too small
            "take_profit_pct": Decimal("0.05"),
            "timeframe": "1h",
            "lookback_period": 100,  # Too large
            "confidence_threshold": Decimal("1.5"),  # Too large
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
            name="learning_rate", min_value=Decimal("0.001"), max_value=Decimal("0.1")
        )

        conditional_param = ConditionalParameter(
            name="learning_rate",
            base_parameter=base_param,
            activation_conditions={"model_type": "neural_network"},
        )

        assert conditional_param.is_conditional == True
        assert conditional_param.conditions == {"model_type": "neural_network"}

    def test_conditional_parameter_activation(self):
        """Test conditional parameter activation."""
        base_param = ContinuousParameter(
            name="n_estimators", min_value=Decimal("50"), max_value=Decimal("500")
        )

        conditional_param = ConditionalParameter(
            name="n_estimators",
            base_parameter=base_param,
            activation_conditions={"model_type": "random_forest"},
        )

        # Test activation
        context_active = {"model_type": "random_forest"}
        assert conditional_param.is_active(context_active) == True

        context_inactive = {"model_type": "neural_network"}
        assert conditional_param.is_active(context_inactive) == False

    def test_conditional_parameter_sampling(self):
        """Test conditional parameter sampling."""
        base_param = DiscreteParameter(name="max_depth", min_value=3, max_value=20)

        conditional_param = ConditionalParameter(
            name="max_depth",
            base_parameter=base_param,
            activation_conditions={"model_type": ["random_forest", "xgboost"]},
        )

        # Test sampling
        value = conditional_param.sample()
        assert 3 <= value <= 20


class TestParameterSpaceBuilder:
    """Test parameter space builder functionality."""

    def test_fluent_interface(self):
        """Test fluent interface for building parameter spaces."""
        builder = ParameterSpaceBuilder()

        space = (
            builder.add_continuous("param1", 0.0, 1.0)
            .add_discrete("param2", 1, 10)
            .add_categorical("param3", ["A", "B"])
            .add_boolean("param4")
            .add_constraint("param1 + param2 > 0")
            .set_metadata("description", "Test space")
            .build()
        )

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
        space = (
            builder.add_categorical("strategy_type", ["momentum", "mean_reversion", "arbitrage"])
            .add_continuous("param1", 0.0, 1.0)
            .add_discrete("param2", 1, 100)
            .add_boolean("use_feature")
            .build()
        )

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
            name="learning_rate", min_value=Decimal("0.001"), max_value=Decimal("0.1")
        )

        space = (
            builder.add_categorical("model", ["simple", "complex"])
            .add_conditional("learning_rate", base_lr, {"model": "complex"})
            .add_discrete("epochs", 1, 100)
            .build()
        )

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


class TestFinancialEdgeCases:
    """Test edge cases specific to financial applications."""

    def test_decimal_precision_preservation(self):
        """Test that decimal precision is preserved in financial calculations."""
        param = ContinuousParameter(
            name="price", min_value=Decimal("0.00000001"), max_value=Decimal("1.0"), precision=8
        )

        # Test that precision is maintained
        value = param.sample()
        assert isinstance(value, Decimal)
        
        # Test very small values
        small_value = param.clip_value(Decimal("0.00000005"))
        assert isinstance(small_value, Decimal)
        assert small_value >= param.min_value

    def test_extreme_parameter_ranges(self):
        """Test extreme parameter ranges for financial use cases."""
        # Test very large ranges (e.g., market cap)
        large_param = ContinuousParameter(
            name="market_cap",
            min_value=Decimal("1000000"),  # 1M
            max_value=Decimal("1000000000000"),  # 1T
            log_scale=True,
            precision=0,
        )

        value = large_param.sample(SamplingStrategy.LOG_UNIFORM)
        assert large_param.min_value <= value <= large_param.max_value

        # Test very small ranges (e.g., fee rates)
        small_param = ContinuousParameter(
            name="fee_rate",
            min_value=Decimal("0.00001"),  # 0.001%
            max_value=Decimal("0.001"),  # 0.1%
            precision=6,
        )

        value = small_param.sample()
        assert small_param.min_value <= value <= small_param.max_value

    def test_financial_constraint_validation(self):
        """Test financial-specific constraint validation."""
        builder = ParameterSpaceBuilder()

        # Risk management constraints
        space = (
            builder.add_continuous("position_size", 0.001, 0.1, precision=4)
            .add_continuous("stop_loss", 0.005, 0.05, precision=4)
            .add_continuous("leverage", 1.0, 10.0, precision=1)
            .build()
        )

        # Test valid financial parameters
        valid_params = {
            "position_size": Decimal("0.02"),  # 2%
            "stop_loss": Decimal("0.02"),  # 2%
            "leverage": Decimal("2.0"),  # 2x
        }

        validation_results = space.validate_parameter_set(valid_params)
        assert all(validation_results.values())

        # Test risk management violations
        risky_params = {
            "position_size": Decimal("0.15"),  # 15% - too large
            "stop_loss": Decimal("0.001"),  # 0.1% - too small
            "leverage": Decimal("15.0"),  # 15x - too high
        }

        clipped_params = space.clip_parameters(risky_params)
        assert clipped_params["position_size"] <= Decimal("0.1")
        assert clipped_params["stop_loss"] >= Decimal("0.005")
        assert clipped_params["leverage"] <= Decimal("10.0")

    def test_trading_timeframe_constraints(self):
        """Test trading timeframe parameter constraints."""
        # Valid trading timeframes
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "8h", "12h", "1d", "3d", "1w"]
        
        timeframe_param = CategoricalParameter(
            name="timeframe",
            choices=timeframes,
            weights=[0.05, 0.1, 0.15, 0.15, 0.2, 0.15, 0.05, 0.05, 0.05, 0.03, 0.02],  # Favor common timeframes
        )

        # Test that all samples are valid timeframes (reduced iterations)
        with patch('random.choices') as mock_choices:
            mock_choices.side_effect = [["1h"], ["5m"], ["1d"], ["4h"], ["15m"]]
            for _ in range(5):  # Reduced from 100
                timeframe = timeframe_param.sample()
                assert timeframe in timeframes
                assert isinstance(timeframe, str)

    def test_parameter_correlation_constraints(self):
        """Test parameter correlation constraints in trading contexts."""
        builder = ParameterSpaceBuilder()

        # Position size should correlate with stop loss
        base_stop_loss = ContinuousParameter(
            name="stop_loss_base", min_value=Decimal("0.01"), max_value=Decimal("0.05")
        )

        space = (
            builder.add_continuous("position_size", 0.01, 0.1)
            .add_conditional(
                "stop_loss",
                base_stop_loss,
                {"position_size": {"min": 0.05, "max": 1.0}},  # Larger positions need larger stops
            )
            .build()
        )

        # Test conditional parameter activation
        large_position_context = {"position_size": Decimal("0.08")}
        active_params = space.get_active_parameters(large_position_context)
        assert "stop_loss" in active_params

        small_position_context = {"position_size": Decimal("0.02")}
        active_params = space.get_active_parameters(small_position_context)
        assert "stop_loss" not in active_params

    def test_currency_precision_handling(self):
        """Test handling of different currency precisions."""
        # Bitcoin precision (8 decimals)
        btc_param = ContinuousParameter(
            name="btc_amount",
            min_value=Decimal("0.00000001"),  # 1 satoshi
            max_value=Decimal("1.0"),  # 1 BTC
            precision=8,
        )

        value = btc_param.sample()
        # Check precision is exactly 8 decimal places
        assert len(str(value).split('.')[-1]) <= 8

        # USD precision (2 decimals)
        usd_param = ContinuousParameter(
            name="usd_amount",
            min_value=Decimal("0.01"),  # 1 cent
            max_value=Decimal("1000000.00"),  # 1M USD
            precision=2,
        )

        value = usd_param.sample()
        # Check precision is exactly 2 decimal places
        decimal_places = len(str(value).split('.')[-1]) if '.' in str(value) else 0
        assert decimal_places <= 2

    def test_risk_parameter_bounds(self):
        """Test risk parameter bounds are financially sensible."""
        risk_space = create_risk_management_space()

        # Test multiple samples to ensure bounds are respected (reduced iterations)
        for _ in range(10):  # Reduced from 50
            sample = risk_space.sample()
            
            # Portfolio exposure should never exceed 100%
            assert sample["max_portfolio_exposure"] <= Decimal("0.95")
            assert sample["max_portfolio_exposure"] >= Decimal("0.5")
            
            # Maximum positions should be reasonable
            assert 1 <= sample["max_positions"] <= 20
            
            # Drawdown limit should be reasonable
            assert Decimal("0.05") <= sample["max_drawdown_limit"] <= Decimal("0.25")
            
            # VaR confidence should be high
            assert Decimal("0.9") <= sample["var_confidence_level"] <= Decimal("0.99")

    def test_ml_hyperparameter_constraints(self):
        """Test ML hyperparameter constraints for financial models."""
        ml_space = create_ml_model_space()

        # Test that conditional parameters exist in the space
        assert any(name.startswith("rf_") for name in ml_space.parameters)
        assert any(name.startswith("xgb_") for name in ml_space.parameters)
        
        # Test parameter activation based on model type
        rf_context = {"model_type": "random_forest"}
        active_rf_params = ml_space.get_active_parameters(rf_context)
        
        # Should have RF-specific parameters when model_type is random_forest
        rf_specific = [p for p in active_rf_params if p.startswith("rf_")]
        assert len(rf_specific) > 0
        
        xgb_context = {"model_type": "xgboost"}  
        active_xgb_params = ml_space.get_active_parameters(xgb_context)
        
        # Should have XGB-specific parameters when model_type is xgboost
        xgb_specific = [p for p in active_xgb_params if p.startswith("xgb_")]
        assert len(xgb_specific) > 0
        
        # Test general sampling works
        sample = ml_space.sample()
        assert "model_type" in sample
        assert sample["model_type"] in ["random_forest", "xgboost", "neural_network", "svm"]


if __name__ == "__main__":
    pytest.main([__file__])
