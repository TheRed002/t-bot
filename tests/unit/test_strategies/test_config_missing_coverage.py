"""Tests for missing coverage in StrategyConfigurationManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest
import yaml

from src.core.exceptions import ConfigurationError
from src.core.types import StrategyConfig, StrategyType
from src.strategies.config import StrategyConfigurationManager


class TestStrategyConfigurationManagerMissingCoverage:
    """Test cases for uncovered functionality in StrategyConfigurationManager."""

    def test_load_strategy_config_yaml_file(self, tmp_path):
        """Test loading strategy config from YAML file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create test YAML config
        config_data = {
            "strategy_id": "test_yaml_001",
            "name": "test_strategy",
            "strategy_type": "mean_reversion",
            "enabled": True,
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {"lookback_period": 20}
        }
        
        config_file = config_dir / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        config = manager.load_strategy_config("test_strategy")
        
        assert config.name == "test_strategy"
        assert config.strategy_type == StrategyType.MEAN_REVERSION

    def test_load_strategy_config_yml_file(self, tmp_path):
        """Test loading strategy config from .yml file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_data = {
            "strategy_id": "test_yml_001",
            "name": "test_strategy",
            "strategy_type": "trend_following",
            "enabled": True,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {"fast_ma": 20}
        }
        
        config_file = config_dir / "test_strategy.yml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        config = manager.load_strategy_config("test_strategy")
        
        assert config.name == "test_strategy"
        assert config.strategy_type == StrategyType.TREND_FOLLOWING

    def test_load_strategy_config_json_file(self, tmp_path):
        """Test loading strategy config from JSON file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_data = {
            "strategy_id": "test_json_001",
            "name": "test_strategy",
            "strategy_type": "momentum",
            "enabled": True,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {"breakout_confirmation_periods": 3}
        }
        
        config_file = config_dir / "test_strategy.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        config = manager.load_strategy_config("test_strategy")
        
        assert config.name == "test_strategy"
        assert config.strategy_type == StrategyType.MOMENTUM

    def test_load_strategy_config_nested_strategy_key(self, tmp_path):
        """Test loading config with nested 'strategy' key."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        nested_config = {
            "strategy": {
                "strategy_id": "nested_001",
                "name": "nested_strategy",
                "strategy_type": "mean_reversion",
                "enabled": True,
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "min_confidence": 0.6,
                "max_positions": 5,
                "position_size_pct": 0.02,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "parameters": {"lookback_period": 20}
            }
        }
        
        config_file = config_dir / "nested_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(nested_config, f)
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        config = manager.load_strategy_config("nested_strategy")
        
        assert config.name == "nested_strategy"

    def test_load_strategy_config_unsupported_file_extension(self, tmp_path):
        """Test loading config with unsupported file extension."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create file with unsupported extension
        unsupported_file = config_dir / "test_strategy.txt"
        unsupported_file.write_text("some content")
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        with pytest.raises(ConfigurationError, match="Unsupported config file format"):
            manager.load_strategy_config("test_strategy")

    def test_load_strategy_config_file_not_found_exception(self, tmp_path):
        """Test FileNotFoundError during config loading."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        # Mock file exists check to pass but file read to fail
        with patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            
            with pytest.raises(ConfigurationError, match="Configuration file not found"):
                manager.load_strategy_config("nonexistent")

    def test_load_strategy_config_yaml_error(self, tmp_path):
        """Test YAMLError during config loading."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create invalid YAML file
        config_file = config_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            manager.load_strategy_config("invalid")

    def test_load_strategy_config_json_decode_error(self, tmp_path):
        """Test JSONDecodeError during config loading."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create invalid JSON file
        config_file = config_dir / "invalid.json"
        config_file.write_text('{"invalid": json}')
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        with pytest.raises(ConfigurationError, match="Invalid JSON"):
            manager.load_strategy_config("invalid")

    def test_load_config_file_unsupported_extension(self, tmp_path):
        """Test _load_config_file with unsupported extension."""
        manager = StrategyConfigurationManager()
        
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("content")
        
        with pytest.raises(ConfigurationError, match="Unsupported config file format"):
            manager._load_config_file(unsupported_file)

    def test_load_config_file_file_not_found_error(self):
        """Test _load_config_file with FileNotFoundError."""
        manager = StrategyConfigurationManager()
        
        nonexistent_file = Path("/nonexistent/file.yaml")
        
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            manager._load_config_file(nonexistent_file)

    def test_load_config_file_yaml_error(self, tmp_path):
        """Test _load_config_file with YAMLError."""
        manager = StrategyConfigurationManager()
        
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: [")
        
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            manager._load_config_file(invalid_yaml)

    def test_load_config_file_json_error(self, tmp_path):
        """Test _load_config_file with JSONDecodeError."""
        manager = StrategyConfigurationManager()
        
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text('{"invalid": json}')
        
        with pytest.raises(ConfigurationError, match="Invalid JSON"):
            manager._load_config_file(invalid_json)

    def test_get_default_config_invalid_strategy(self):
        """Test _get_default_config with invalid strategy name."""
        manager = StrategyConfigurationManager()
        
        with pytest.raises(ConfigurationError, match="No default configuration"):
            manager._get_default_config("nonexistent_strategy")

    def test_save_strategy_config_permission_error(self, tmp_path):
        """Test save_strategy_config with PermissionError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        config = StrategyConfig(
            strategy_id="test_001",
            name="test",
            strategy_type=StrategyType.MEAN_REVERSION,
            enabled=True,
            symbol="BTCUSDT",
            timeframe="5m",
            min_confidence=0.6,
            max_positions=5,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            parameters={"lookback_period": 20}
        )
        
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(ConfigurationError, match="Permission denied saving configuration"):
                manager.save_strategy_config("test", config)

    def test_save_strategy_config_os_error(self, tmp_path):
        """Test save_strategy_config with OSError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        config = StrategyConfig(
            strategy_id="test_001",
            name="test",
            strategy_type=StrategyType.MEAN_REVERSION,
            enabled=True,
            symbol="BTCUSDT",
            timeframe="5m",
            min_confidence=0.6,
            max_positions=5,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            parameters={"lookback_period": 20}
        )
        
        with patch("builtins.open", side_effect=OSError("OS error")):
            with pytest.raises(ConfigurationError, match="OS error saving configuration"):
                manager.save_strategy_config("test", config)

    def test_save_strategy_config_yaml_error(self, tmp_path):
        """Test save_strategy_config with YAMLError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        config = StrategyConfig(
            strategy_id="test_001",
            name="test",
            strategy_type=StrategyType.MEAN_REVERSION,
            enabled=True,
            symbol="BTCUSDT",
            timeframe="5m",
            min_confidence=0.6,
            max_positions=5,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            parameters={"lookback_period": 20}
        )
        
        with patch("yaml.dump", side_effect=yaml.YAMLError("YAML error")):
            with pytest.raises(ConfigurationError, match="Failed to serialize YAML"):
                manager.save_strategy_config("test", config)

    def test_validate_config_value_error(self):
        """Test validate_config with ValueError."""
        manager = StrategyConfigurationManager()
        
        invalid_config = {
            "strategy_id": "test_001",
            "name": "test",
            "strategy_type": "invalid_type",  # This will cause ValueError
            "enabled": True,
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {"lookback_period": 20}
        }
        
        result = manager.validate_config(invalid_config)
        assert result is False

    def test_validate_config_type_error(self):
        """Test validate_config with TypeError."""
        manager = StrategyConfigurationManager()
        
        invalid_config = {
            "strategy_id": "test_001",
            "name": "test",
            "strategy_type": "mean_reversion",
            "enabled": "not_boolean",  # This will cause TypeError
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "min_confidence": 0.6,
            "max_positions": 5,
            "position_size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "parameters": {"lookback_period": 20}
        }
        
        result = manager.validate_config(invalid_config)
        assert result is False

    def test_validate_config_general_exception(self):
        """Test validate_config with general Exception."""
        manager = StrategyConfigurationManager()
        
        with patch("src.core.types.StrategyConfig", side_effect=Exception("General error")):
            result = manager.validate_config({"test": "config"})
            assert result is False

    def test_get_available_strategies(self, tmp_path):
        """Test get_available_strategies method."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create some config files
        (config_dir / "strategy1.yaml").write_text("test")
        (config_dir / "strategy2.yaml").write_text("test")
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        strategies = manager.get_available_strategies()
        
        # Should include file-based strategies and default configs
        assert "strategy1" in strategies
        assert "strategy2" in strategies
        assert "mean_reversion" in strategies
        assert "trend_following" in strategies
        assert "breakout" in strategies

    def test_get_config_schema(self):
        """Test get_config_schema method."""
        manager = StrategyConfigurationManager()
        schema = manager.get_config_schema()
        
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_update_config_parameter_success(self, tmp_path):
        """Test successful parameter update."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        # This will use default config since no file exists
        result = manager.update_config_parameter("mean_reversion", "min_confidence", 0.7)
        assert result is True

    def test_update_config_parameter_validation_failure(self, tmp_path):
        """Test parameter update with validation failure."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        # Try to set invalid value
        result = manager.update_config_parameter("mean_reversion", "min_confidence", "invalid")
        assert result is False

    def test_update_config_parameter_not_found(self, tmp_path):
        """Test parameter update with parameter not found."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        result = manager.update_config_parameter("mean_reversion", "nonexistent_param", 0.7)
        assert result is False

    def test_update_config_parameter_exception(self, tmp_path):
        """Test parameter update with exception."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        with patch.object(manager, "load_strategy_config", side_effect=Exception("Error")):
            result = manager.update_config_parameter("mean_reversion", "min_confidence", 0.7)
            assert result is False

    def test_create_strategy_config_momentum_mapping(self):
        """Test creating config with momentum strategy type mapping."""
        manager = StrategyConfigurationManager()
        
        config = manager.create_strategy_config(
            "test_momentum",
            StrategyType.MOMENTUM,
            "BTCUSDT",
            timeframe="1h"
        )
        
        assert config.name == "test_momentum"
        assert config.strategy_type == StrategyType.MOMENTUM
        assert config.symbol == "BTCUSDT"
        assert config.timeframe == "1h"

    def test_create_strategy_config_string_strategy_type(self):
        """Test creating config with string strategy type."""
        manager = StrategyConfigurationManager()
        
        # Use string instead of enum for strategy type
        config = manager.create_strategy_config(
            "test_string_type",
            "mean_reversion",  # String instead of enum
            "BTCUSDT"
        )
        
        assert config.name == "test_string_type"
        assert config.symbol == "BTCUSDT"

    def test_create_strategy_config_custom_strategy_id(self):
        """Test create_strategy_config with custom strategy_id."""
        manager = StrategyConfigurationManager()
        
        config = manager.create_strategy_config(
            "test_custom",
            StrategyType.MEAN_REVERSION,
            "BTCUSDT",
            strategy_id="custom_id_123"
        )
        
        assert config.strategy_id == "custom_id_123"
        assert config.name == "test_custom"

    def test_delete_strategy_config_success(self, tmp_path):
        """Test successful deletion of strategy config."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create a config file
        config_file = config_dir / "test_strategy.yaml"
        config_file.write_text("test config")
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        result = manager.delete_strategy_config("test_strategy")
        
        assert result is True
        assert not config_file.exists()

    def test_delete_strategy_config_file_not_found(self, tmp_path):
        """Test deletion of non-existent config file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        result = manager.delete_strategy_config("nonexistent")
        
        assert result is False

    def test_delete_strategy_config_permission_error(self, tmp_path):
        """Test deletion with PermissionError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink", side_effect=PermissionError("Permission denied")):
            
            result = manager.delete_strategy_config("test")
            assert result is False

    def test_delete_strategy_config_os_error(self, tmp_path):
        """Test deletion with OSError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink", side_effect=OSError("OS error")):
            
            result = manager.delete_strategy_config("test")
            assert result is False

    def test_delete_strategy_config_general_exception(self, tmp_path):
        """Test deletion with general exception."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink", side_effect=Exception("General error")):
            
            result = manager.delete_strategy_config("test")
            assert result is False

    def test_get_config_summary(self, tmp_path):
        """Test get_config_summary method."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create some config files
        config_file1 = config_dir / "strategy1.yaml"
        config_file2 = config_dir / "strategy2.yaml"
        config_file1.write_text("config1")
        config_file2.write_text("config2")
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        summary = manager.get_config_summary()
        
        assert summary["config_directory"] == str(config_dir)
        assert summary["total_strategies"] >= 5  # At least the defaults + file configs
        assert len(summary["config_files"]) == 2
        assert "strategy1" in [f["name"] for f in summary["config_files"]]
        assert "strategy2" in [f["name"] for f in summary["config_files"]]
        assert "mean_reversion" in summary["default_configs"]
    
    def test_load_strategy_config_general_exception(self, tmp_path):
        """Test general exception during config loading."""
        config_dir = tmp_path / "config"  
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        # Mock StrategyConfig to raise general exception
        with patch("src.strategies.config.StrategyConfig", side_effect=Exception("General error")):
            with pytest.raises(ConfigurationError, match="Failed to load configuration"):
                manager.load_strategy_config("mean_reversion")
    
    def test_save_strategy_config_general_exception(self, tmp_path):
        """Test general exception during config saving."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        config = StrategyConfig(
            strategy_id="test_001",
            name="test",
            strategy_type=StrategyType.MEAN_REVERSION,
            enabled=True,
            symbol="BTCUSDT",
            timeframe="5m",
            min_confidence=0.6,
            max_positions=5,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            parameters={"lookbook_period": 20}
        )
        
        # Mock datetime to raise exception
        with patch("src.strategies.config.datetime", side_effect=Exception("DateTime error")):
            with pytest.raises(ConfigurationError, match="Failed to save configuration"):
                manager.save_strategy_config("test", config)

    def test_create_strategy_config_general_exception(self):
        """Test general exception during config creation."""
        manager = StrategyConfigurationManager()
        
        # Mock save_strategy_config to raise exception
        with patch.object(manager, "save_strategy_config", side_effect=Exception("Save error")):
            with pytest.raises(ConfigurationError, match="Failed to create configuration"):
                manager.create_strategy_config("test", StrategyType.MEAN_REVERSION, "BTCUSDT")
    
    def test_update_config_parameter_configuration_error_reraise(self, tmp_path):
        """Test that ConfigurationError is re-raised in update_config_parameter."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        manager = StrategyConfigurationManager(config_dir=str(config_dir))
        
        # Mock load_strategy_config to raise ConfigurationError
        with patch.object(manager, "load_strategy_config", side_effect=ConfigurationError("Config error")):
            with pytest.raises(ConfigurationError, match="Config error"):
                manager.update_config_parameter("test", "min_confidence", 0.7)
    
    def test_create_strategy_config_configuration_error_reraise(self):
        """Test that ConfigurationError is re-raised in create_strategy_config."""
        manager = StrategyConfigurationManager()
        
        # Mock save_strategy_config to raise ConfigurationError  
        with patch.object(manager, "save_strategy_config", side_effect=ConfigurationError("Save error")):
            with pytest.raises(ConfigurationError, match="Save error"):
                manager.create_strategy_config("test", StrategyType.MEAN_REVERSION, "BTCUSDT")
    
    def test_load_config_file_json_nested_strategy(self, tmp_path):
        """Test loading JSON config file with nested strategy key."""
        manager = StrategyConfigurationManager()
        
        nested_json = {
            "strategy": {
                "strategy_id": "nested_001",
                "name": "nested_test",
                "strategy_type": "mean_reversion"
            }
        }
        
        json_file = tmp_path / "nested.json"
        with open(json_file, "w") as f:
            json.dump(nested_json, f)
        
        result = manager._load_config_file(json_file)
        assert result["name"] == "nested_test"
        assert "strategy_id" in result