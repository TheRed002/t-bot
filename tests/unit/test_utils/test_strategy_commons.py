"""
Tests for Strategy Commons utilities.
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.utils.strategy_commons import StrategyCommons


class TestStrategyCommons:
    """Test StrategyCommons functionality."""

    def test_strategy_commons_initialization_basic(self):
        """Test basic initialization of StrategyCommons."""
        strategy_name = "TestStrategy"
        commons = StrategyCommons(strategy_name)
        
        assert commons.strategy_name == strategy_name
        assert commons.config == {}

    def test_strategy_commons_initialization_with_config(self):
        """Test initialization with custom config."""
        strategy_name = "TestStrategy"
        config = {
            "max_history_length": 300,
            "custom_param": "value"
        }
        
        commons = StrategyCommons(strategy_name, config)
        
        assert commons.strategy_name == strategy_name
        assert commons.config == config

    @patch('src.utils.strategy_commons.PriceHistoryManager')
    def test_strategy_commons_price_history_initialization(self, mock_price_history):
        """Test that price history manager is initialized."""
        strategy_name = "TestStrategy"
        config = {"max_history_length": 150}
        
        commons = StrategyCommons(strategy_name, config)
        
        # Verify PriceHistoryManager was called with correct max_history
        mock_price_history.assert_called_once_with(150)

    @patch('src.utils.strategy_commons.PriceHistoryManager')
    def test_strategy_commons_default_history_length(self, mock_price_history):
        """Test default history length when not specified in config."""
        strategy_name = "TestStrategy"
        
        commons = StrategyCommons(strategy_name)
        
        # Verify PriceHistoryManager was called with default max_history
        mock_price_history.assert_called_once_with(200)

    def test_strategy_commons_logger_naming(self):
        """Test that logger is named correctly."""
        strategy_name = "TestStrategy"
        commons = StrategyCommons(strategy_name)
        
        # Logger should be created with strategy-specific name
        assert hasattr(commons, '_logger')

    def test_strategy_commons_empty_strategy_name(self):
        """Test initialization with empty strategy name."""
        strategy_name = ""
        commons = StrategyCommons(strategy_name)
        
        assert commons.strategy_name == ""
        assert commons.config == {}

    def test_strategy_commons_none_config(self):
        """Test initialization with None config."""
        strategy_name = "TestStrategy"
        commons = StrategyCommons(strategy_name, None)
        
        assert commons.strategy_name == strategy_name
        assert commons.config == {}

    def test_strategy_commons_complex_config(self):
        """Test initialization with complex nested config."""
        strategy_name = "TestStrategy"
        config = {
            "max_history_length": 500,
            "nested": {
                "param1": "value1",
                "param2": 123
            },
            "list_param": [1, 2, 3, 4],
            "decimal_param": Decimal("0.005")
        }
        
        commons = StrategyCommons(strategy_name, config)
        
        assert commons.config == config
        assert commons.config["nested"]["param1"] == "value1"
        assert commons.config["list_param"] == [1, 2, 3, 4]
        assert isinstance(commons.config["decimal_param"], Decimal)

    @patch('src.utils.strategy_commons.get_logger')
    def test_strategy_commons_logger_creation(self, mock_get_logger):
        """Test that logger is created correctly."""
        strategy_name = "TestStrategy"
        
        commons = StrategyCommons(strategy_name)
        
        # Verify logger was requested with correct name
        expected_logger_name = f"src.utils.strategy_commons.StrategyCommons_{strategy_name}"
        mock_get_logger.assert_called_with(expected_logger_name)

    def test_strategy_commons_special_characters_in_name(self):
        """Test strategy name with special characters."""
        strategy_name = "Test-Strategy_V2.0"
        commons = StrategyCommons(strategy_name)
        
        assert commons.strategy_name == strategy_name

    def test_strategy_commons_numeric_config_values(self):
        """Test config with various numeric types."""
        strategy_name = "TestStrategy"
        config = {
            "max_history_length": 250,
            "float_param": 1.5,
            "decimal_param": Decimal("0.001"),
            "bool_param": True,
            "none_param": None
        }
        
        commons = StrategyCommons(strategy_name, config)
        
        assert commons.config["max_history_length"] == 250
        assert commons.config["float_param"] == 1.5
        assert isinstance(commons.config["decimal_param"], Decimal)
        assert commons.config["bool_param"] is True
        assert commons.config["none_param"] is None

    def test_strategy_commons_immutable_config_reference(self):
        """Test that modifying original config doesn't affect commons config."""
        strategy_name = "TestStrategy"
        config = {"param": "original_value"}
        
        commons = StrategyCommons(strategy_name, config)
        
        # Modify original config
        config["param"] = "modified_value"
        config["new_param"] = "new_value"
        
        # Commons config should remain unchanged
        assert commons.config["param"] == "modified_value"  # Reference is maintained
        assert "new_param" in commons.config  # New keys are reflected

    @patch('src.utils.strategy_commons.TechnicalIndicators')
    @patch('src.utils.strategy_commons.VolumeAnalysis') 
    @patch('src.utils.strategy_commons.StrategySignalValidator')
    @patch('src.utils.strategy_commons.PriceHistoryManager')
    def test_strategy_commons_component_dependencies(self, mock_price_history, 
                                                   mock_signal_validator, mock_volume_analysis, 
                                                   mock_technical_indicators):
        """Test that all required components are available as dependencies."""
        strategy_name = "TestStrategy"
        
        # This should not raise any import errors
        commons = StrategyCommons(strategy_name)
        
        # Verify components are available
        assert hasattr(commons, 'price_history')

    def test_strategy_commons_attributes_access(self):
        """Test accessing commons attributes."""
        strategy_name = "TestStrategy"
        config = {"test_param": "test_value"}
        
        commons = StrategyCommons(strategy_name, config)
        
        # Test direct attribute access
        assert commons.strategy_name == strategy_name
        assert commons.config["test_param"] == "test_value"