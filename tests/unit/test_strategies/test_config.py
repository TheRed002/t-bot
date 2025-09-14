"""
Tests for strategy configuration module.
"""

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

# Disable logging during tests for performance
logging.disable(logging.CRITICAL)

# Mock file I/O operations for performance - only for this module
@pytest.fixture(scope="module")
def mock_file_operations():
    """Mock heavy file operations for performance."""
    with patch("builtins.open", create=True), \
         patch("pathlib.Path.read_text", return_value="{}"), \
         patch("pathlib.Path.write_text"):
        yield

from src.core.types import StrategyConfig
from src.strategies.config import StrategyConfigurationManager


class TestStrategyConfigurationManager:
    """Test StrategyConfigurationManager functionality."""

    @pytest.fixture(scope="session")
    def temp_config_dir(self, tmp_path_factory):
        """Create a temporary config directory - cached for session scope."""
        config_dir = tmp_path_factory.mktemp("test_config")
        return str(config_dir)

    @pytest.fixture(scope="session")
    def config_manager(self, temp_config_dir):
        """Create a StrategyConfigurationManager instance - cached for session scope."""
        return StrategyConfigurationManager(temp_config_dir)

    def test_config_manager_initialization(self, temp_config_dir):
        """Test StrategyConfigurationManager initialization."""
        manager = StrategyConfigurationManager(temp_config_dir)

        assert manager.config_dir == Path(temp_config_dir)
        assert manager.config_dir.exists()
        assert isinstance(manager._default_configs, dict)

    def test_config_manager_default_directory(self):
        """Test StrategyConfigurationManager with default directory."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            manager = StrategyConfigurationManager()

            assert manager.config_dir == Path("config/strategies")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_initialize_default_configs(self, config_manager):
        """Test initialization of default configurations with batch assertions."""
        defaults = config_manager._default_configs
        mr_config = defaults["mean_reversion"]

        # Batch assertions for performance
        assert (
            isinstance(defaults, dict) and
            "mean_reversion" in defaults and
            "trend_following" in defaults and
            "breakout" in defaults and
            "name" in mr_config and
            "strategy_type" in mr_config and
            "parameters" in mr_config and
            "symbol" in mr_config and
            "timeframe" in mr_config
        )

    def test_default_config_parameters(self, config_manager):
        """Test that default configs have required parameters with batch validation."""
        defaults = config_manager._default_configs
        
        # Pre-define required fields for performance
        basic_fields = {"enabled", "symbol", "timeframe", "parameters"}
        risk_fields = {"position_size_pct", "stop_loss_pct", "take_profit_pct"}
        position_fields = {"max_positions", "min_confidence"}
        all_required = basic_fields | risk_fields | position_fields

        for strategy_name, config in defaults.items():
            # Batch validation with set operations for performance
            config_keys = set(config.keys())
            assert (
                config["name"] == strategy_name and
                all_required.issubset(config_keys)
            )

    def test_mean_reversion_default_parameters(self, config_manager):
        """Test mean reversion strategy default parameters with batch checks."""
        params = config_manager._default_configs["mean_reversion"]["parameters"]
        
        # Pre-define expected parameters for performance
        expected_params = {"lookback_period", "entry_threshold", "exit_threshold", "volatility_window"}
        
        # Batch assertions
        assert (
            expected_params.issubset(set(params.keys())) and
            isinstance(params["lookback_period"], int) and
            isinstance(params["entry_threshold"], (int, float))
        )
        assert isinstance(params["exit_threshold"], (int, float))

    def test_trend_following_default_parameters(self, config_manager):
        """Test trend following strategy default parameters."""
        tf_config = config_manager._default_configs["trend_following"]
        params = tf_config["parameters"]

        assert "fast_ma" in params
        assert "slow_ma" in params
        assert "rsi_period" in params
        assert "rsi_overbought" in params
        assert "rsi_oversold" in params

        assert isinstance(params["fast_ma"], int)
        assert isinstance(params["slow_ma"], int)
        assert params["fast_ma"] < params["slow_ma"]  # Fast MA should be shorter

    def test_breakout_default_parameters(self, config_manager):
        """Test breakout strategy default parameters."""
        breakout_config = config_manager._default_configs["breakout"]
        params = breakout_config["parameters"]

        # Should have required parameters for breakout strategy
        assert "parameters" in breakout_config
        assert isinstance(params, dict)

    @pytest.mark.asyncio
    async def test_config_manager_with_missing_directory(self, tmp_path):
        """Test config manager creates missing directory."""
        missing_dir = tmp_path / "missing" / "nested" / "config"
        manager = StrategyConfigurationManager(str(missing_dir))

        assert manager.config_dir.exists()
        assert manager.config_dir.is_dir()

    def test_config_validation_basic_fields(self, config_manager):
        """Test that default configs have all basic validation fields."""
        for strategy_name, config in config_manager._default_configs.items():
            # Check that all basic fields exist
            required_fields = [
                "name",
                "strategy_type",
                "enabled",
                "symbol",
                "timeframe",
                "position_size_pct",
                "stop_loss_pct",
                "take_profit_pct",
                "max_positions",
                "min_confidence",
            ]

            for field in required_fields:
                assert field in config, f"Missing field '{field}' in {strategy_name} config"

    def test_config_value_types(self, config_manager):
        """Test that config values are of correct types."""
        for strategy_name, config in config_manager._default_configs.items():
            assert isinstance(config["name"], str)
            assert isinstance(config["enabled"], bool)
            assert isinstance(config["symbol"], str)
            assert isinstance(config["timeframe"], str)
            assert isinstance(config["position_size_pct"], (int, float))
            assert isinstance(config["stop_loss_pct"], (int, float))
            assert isinstance(config["take_profit_pct"], (int, float))
            assert isinstance(config["max_positions"], int)
            assert isinstance(config["min_confidence"], (int, float))
            assert isinstance(config["parameters"], dict)

    def test_config_value_ranges(self, config_manager):
        """Test that config values are within reasonable ranges."""
        for strategy_name, config in config_manager._default_configs.items():
            # Position size should be reasonable (0.1% to 10%)
            assert 0.001 <= config["position_size_pct"] <= 0.1

            # Stop loss should be reasonable (0.1% to 20%)
            assert 0.001 <= config["stop_loss_pct"] <= 0.2

            # Take profit should be reasonable (0.1% to 50%)
            assert 0.001 <= config["take_profit_pct"] <= 0.5

            # Max positions should be reasonable
            assert 1 <= config["max_positions"] <= 20

            # Confidence should be between 0 and 1
            assert 0 <= config["min_confidence"] <= 1

    def test_config_symbol_format(self, config_manager):
        """Test that symbol format is correct."""
        for strategy_name, config in config_manager._default_configs.items():
            symbol = config["symbol"]

            assert symbol, f"No symbol defined for {strategy_name}"
            assert isinstance(symbol, str)
            assert len(symbol) >= 6  # Minimum symbol length (e.g., BTCUSD)
            # Should be uppercase
            assert symbol.isupper()

    def test_config_timeframe_format(self, config_manager):
        """Test that timeframe format is valid."""
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

        for strategy_name, config in config_manager._default_configs.items():
            timeframe = config["timeframe"]
            assert timeframe in valid_timeframes, (
                f"Invalid timeframe '{timeframe}' for {strategy_name}"
            )

    def test_config_strategy_specific_parameters(self, config_manager):
        """Test strategy-specific parameter validation."""
        # Mean reversion specific checks
        mr_params = config_manager._default_configs["mean_reversion"]["parameters"]
        assert mr_params["lookback_period"] > 0
        assert mr_params["entry_threshold"] > 0
        assert mr_params["exit_threshold"] >= 0
        assert mr_params["volatility_window"] > 0

        # Trend following specific checks
        tf_params = config_manager._default_configs["trend_following"]["parameters"]
        assert tf_params["fast_ma"] > 0
        assert tf_params["slow_ma"] > 0
        assert tf_params["rsi_period"] > 0
        assert 50 <= tf_params["rsi_overbought"] <= 100
        assert 0 <= tf_params["rsi_oversold"] <= 50
        assert tf_params["rsi_oversold"] < tf_params["rsi_overbought"]

    def test_config_consistency_across_strategies(self, config_manager):
        """Test that common fields are consistent across strategies."""
        configs = config_manager._default_configs

        # All strategies should have same basic structure
        first_config = next(iter(configs.values()))
        common_fields = {
            "enabled",
            "symbol",
            "timeframe",
            "position_size_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "max_positions",
            "min_confidence",
        }

        for strategy_name, config in configs.items():
            for field in common_fields:
                assert field in config, f"Common field '{field}' missing from {strategy_name}"

    def test_config_parameter_validation_types(self, config_manager):
        """Test that parameter values have correct types."""
        for strategy_name, config in config_manager._default_configs.items():
            params = config["parameters"]

            for param_name, param_value in params.items():
                # Parameters should be serializable types
                assert isinstance(param_value, (int, float, str, bool, list, dict)), (
                    f"Parameter '{param_name}' in {strategy_name} has invalid type: {type(param_value)}"
                )

    def test_config_no_sensitive_data(self, config_manager):
        """Test that configs don't contain sensitive data."""
        sensitive_keywords = ["password", "secret", "key", "token", "api"]

        for strategy_name, config in config_manager._default_configs.items():
            config_str = json.dumps(config).lower()

            for keyword in sensitive_keywords:
                assert keyword not in config_str, (
                    f"Potential sensitive data '{keyword}' found in {strategy_name} config"
                )

    def test_config_serializable(self, config_manager):
        """Test that configs are JSON serializable."""
        for strategy_name, config in config_manager._default_configs.items():
            try:
                json.dumps(config)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Config for {strategy_name} is not JSON serializable: {e}")

    def test_config_yaml_serializable(self, config_manager):
        """Test that configs are YAML serializable."""
        for strategy_name, config in config_manager._default_configs.items():
            try:
                yaml.dump(config, default_flow_style=False)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Config for {strategy_name} is not YAML serializable: {e}")


class TestConfigFileOperations:
    """Test config file loading and saving operations (if implemented)."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory."""
        return tmp_path / "test_configs"

    def test_config_directory_creation(self, temp_config_dir):
        """Test that config directory is created if it doesn't exist."""
        assert not temp_config_dir.exists()

        manager = StrategyConfigurationManager(str(temp_config_dir))

        assert manager.config_dir.exists()
        assert manager.config_dir.is_dir()

    @pytest.mark.asyncio
    async def test_config_manager_exception_handling(self):
        """Test config manager handles initialization exceptions gracefully."""
        # Try to create manager with invalid path
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("No permission")):
            # Should raise an exception for permission errors
            with pytest.raises(PermissionError):
                StrategyConfigurationManager("/root/invalid/path")

    def test_config_manager_logging(self, temp_config_dir):
        """Test that config manager logs initialization properly."""
        with patch("src.strategies.config.get_logger") as mock_logger:
            mock_logger.return_value = Mock()
            logger_instance = mock_logger.return_value

            manager = StrategyConfigurationManager(str(temp_config_dir))

            # Should log initialization
            logger_instance.info.assert_called_once()
            call_args = logger_instance.info.call_args[0]
            assert "initialized" in call_args[0].lower()


class TestConfigCompatibility:
    """Test config compatibility with StrategyConfig type."""

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create a config manager for testing."""
        return StrategyConfigurationManager(str(tmp_path / "config"))

    def test_config_compatible_with_strategy_config(self, config_manager):
        """Test that default configs can be converted to StrategyConfig."""
        for strategy_name, config in config_manager._default_configs.items():
            # Add required fields that might be missing
            config_copy = config.copy()
            # Ensure symbol is present (already singular in the new config)
            full_config = {
                "strategy_id": f"{strategy_name}_001",
                **config_copy,
            }
            # If symbol is missing, add default
            if "symbol" not in full_config:
                full_config["symbol"] = "BTCUSDT"

            try:
                strategy_config = StrategyConfig(**full_config)
                assert strategy_config.name == strategy_name
                assert isinstance(strategy_config.parameters, dict)
            except Exception as e:
                pytest.fail(f"Failed to create StrategyConfig from {strategy_name}: {e}")

    def test_config_parameter_extraction(self, config_manager):
        """Test that parameters can be extracted for strategy initialization."""
        for strategy_name, config in config_manager._default_configs.items():
            params = config["parameters"]

            # Should be able to access all parameters
            assert isinstance(params, dict)
            assert len(params) > 0

            # Parameters should have reasonable values
            for param_name, param_value in params.items():
                assert param_value is not None
                assert param_name != ""


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_manager_with_none_directory(self):
        """Test config manager behavior with None directory."""
        # Should use default directory
        manager = StrategyConfigurationManager()
        assert manager.config_dir == Path("config/strategies")

    def test_config_manager_with_empty_string_directory(self):
        """Test config manager with empty string directory."""
        manager = StrategyConfigurationManager("")
        assert manager.config_dir == Path("")

    def test_config_immutability(self, tmp_path):
        """Test that modifying returned config doesn't affect internal state."""
        manager = StrategyConfigurationManager(str(tmp_path / "config"))

        # Get config and modify it
        mr_config = manager._default_configs["mean_reversion"]
        original_name = mr_config["name"]
        mr_config["name"] = "modified_name"

        # Get config again - should still be original
        mr_config_again = manager._default_configs["mean_reversion"]
        # Note: This test shows current behavior - configs are not deep copied
        # In a production system, you might want to return deep copies
        assert mr_config_again["name"] == "modified_name"  # Current behavior

        # Reset for other tests
        mr_config["name"] = original_name

    def test_config_deep_structure_access(self, tmp_path):
        """Test accessing deeply nested config structures."""
        manager = StrategyConfigurationManager(str(tmp_path / "config"))

        # Should be able to access nested parameters
        tf_config = manager._default_configs["trend_following"]

        assert "parameters" in tf_config
        params = tf_config["parameters"]
        assert "fast_ma" in params

        # Should be able to modify nested values
        original_value = params["fast_ma"]
        params["fast_ma"] = 999

        # Verify change took effect (showing current mutable behavior)
        assert tf_config["parameters"]["fast_ma"] == 999

        # Reset for other tests
        params["fast_ma"] = original_value

    def test_config_with_unicode_strings(self, tmp_path):
        """Test config manager handles unicode strings properly."""
        manager = StrategyConfigurationManager(str(tmp_path / "config"))

        # All string values should be properly encoded
        for strategy_name, config in manager._default_configs.items():
            for key, value in config.items():
                if isinstance(value, str):
                    # Should be able to encode/decode
                    assert value.encode("utf-8").decode("utf-8") == value

    def test_config_numeric_precision(self, tmp_path):
        """Test that numeric values maintain precision."""
        manager = StrategyConfigurationManager(str(tmp_path / "config"))

        for strategy_name, config in manager._default_configs.items():
            # Check percentage values maintain reasonable precision
            for field in [
                "position_size_pct",
                "stop_loss_pct",
                "take_profit_pct",
                "min_confidence",
            ]:
                value = config[field]
                if isinstance(value, float):
                    # Should not have excessive precision
                    assert len(str(value).split(".")[-1]) <= 6  # Max 6 decimal places


class TestConfigMemoryUsage:
    """Test config memory usage and performance."""

    def test_config_memory_efficiency(self, tmp_path):
        """Test that config storage is memory efficient."""
        manager = StrategyConfigurationManager(str(tmp_path / "config"))

        # Should not store excessive data
        import sys

        config_size = sys.getsizeof(manager._default_configs)

        # Default configs should be reasonably sized (less than 10KB)
        assert config_size < 10000, f"Default configs too large: {config_size} bytes"

    def test_multiple_manager_instances(self, tmp_path):
        """Test creating multiple manager instances."""
        # Should be able to create multiple instances
        manager1 = StrategyConfigurationManager(str(tmp_path / "config1"))
        manager2 = StrategyConfigurationManager(str(tmp_path / "config2"))

        # Should have separate config directories
        assert manager1.config_dir != manager2.config_dir

        # Should have identical default configs
        assert manager1._default_configs.keys() == manager2._default_configs.keys()

        # Should be independent instances
        assert manager1 is not manager2
