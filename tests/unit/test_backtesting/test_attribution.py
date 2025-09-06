"""Tests for backtesting attribution module."""

import pytest
from unittest.mock import MagicMock, patch
from decimal import Decimal

from src.backtesting.attribution import PerformanceAttributor
from src.core.config import Config


class TestPerformanceAttributor:
    """Test PerformanceAttributor."""

    def test_performance_attributor_creation(self):
        """Test creating PerformanceAttributor."""
        config = MagicMock(spec=Config)
        
        attributor = PerformanceAttributor(config=config)
        
        assert attributor.config == config

    def test_performance_attributor_attributes(self):
        """Test PerformanceAttributor has expected attributes."""
        config = MagicMock(spec=Config)
        
        attributor = PerformanceAttributor(config=config)
        
        # Should have basic attributes
        assert hasattr(attributor, 'config')

    def test_performance_attributor_initialization_variations(self):
        """Test different ways to initialize PerformanceAttributor."""
        # Test with mock config
        config1 = MagicMock(spec=Config)
        attributor1 = PerformanceAttributor(config=config1)
        assert attributor1.config == config1
        
        # Test with dict config
        config2 = {"attribution": {"enabled": True}}
        attributor2 = PerformanceAttributor(config=config2)
        assert attributor2.config == config2


class TestAttributionModuleImports:
    """Test attribution module imports."""

    def test_performance_attributor_import(self):
        """Test PerformanceAttributor can be imported."""
        from src.backtesting.attribution import PerformanceAttributor
        
        assert PerformanceAttributor is not None
        assert hasattr(PerformanceAttributor, '__init__')

    def test_module_attributes(self):
        """Test module has expected attributes."""
        import src.backtesting.attribution as attribution_module
        
        assert hasattr(attribution_module, 'PerformanceAttributor')

    def test_attribution_class_properties(self):
        """Test PerformanceAttributor class properties."""
        from src.backtesting.attribution import PerformanceAttributor
        
        assert PerformanceAttributor.__name__ == "PerformanceAttributor"
        assert hasattr(PerformanceAttributor, '__init__')


class TestAttributionIntegration:
    """Test attribution integration scenarios."""

    def test_attributor_with_different_configs(self):
        """Test attributor works with different config types."""
        # Test with mock config
        mock_config = MagicMock(spec=Config)
        attributor1 = PerformanceAttributor(config=mock_config)
        assert attributor1.config == mock_config
        
        # Test with None config
        attributor2 = PerformanceAttributor(config=None)
        assert attributor2.config is None
        
        # Test with empty dict config
        attributor3 = PerformanceAttributor(config={})
        assert attributor3.config == {}

    def test_multiple_attributor_instances(self):
        """Test creating multiple attributor instances."""
        config1 = {"type": "config1"}
        config2 = {"type": "config2"}
        
        attributor1 = PerformanceAttributor(config=config1)
        attributor2 = PerformanceAttributor(config=config2)
        
        assert attributor1 != attributor2
        assert attributor1.config != attributor2.config
        assert attributor1.config == config1
        assert attributor2.config == config2

    def test_attributor_config_assignment(self):
        """Test that config is properly assigned."""
        test_config = {"attribution_settings": {"risk_free_rate": 0.02}}
        
        attributor = PerformanceAttributor(config=test_config)
        
        assert attributor.config is test_config
        assert attributor.config["attribution_settings"]["risk_free_rate"] == 0.02


class TestPerformanceAttributorEdgeCases:
    """Test edge cases for PerformanceAttributor."""

    def test_attributor_with_complex_config(self):
        """Test attributor with complex config structure."""
        complex_config = {
            "attribution": {
                "methods": ["brinson", "fama_french"],
                "risk_model": {
                    "type": "factor_model",
                    "factors": ["market", "value", "momentum"],
                    "lookback_days": 252
                },
                "benchmark": "SPY",
                "currency": "USD"
            },
            "other_settings": {
                "debug": True,
                "cache_results": False
            }
        }
        
        attributor = PerformanceAttributor(config=complex_config)
        
        assert attributor.config == complex_config
        assert attributor.config["attribution"]["methods"] == ["brinson", "fama_french"]
        assert attributor.config["attribution"]["benchmark"] == "SPY"

    def test_attributor_config_access_patterns(self):
        """Test different config access patterns."""
        config_with_methods = MagicMock()
        config_with_methods.get.return_value = {"default": "value"}
        config_with_methods.attribution_config = {"enabled": True}
        
        attributor = PerformanceAttributor(config=config_with_methods)
        
        assert attributor.config == config_with_methods
        # Should be able to access config methods if they exist
        if hasattr(attributor.config, 'get'):
            assert callable(attributor.config.get)
        if hasattr(attributor.config, 'attribution_config'):
            assert attributor.config.attribution_config == {"enabled": True}

    def test_attributor_instantiation_validation(self):
        """Test that attributor can be instantiated with various inputs."""
        # Test with string config (unusual but should work)
        string_config = "test_config"
        attributor1 = PerformanceAttributor(config=string_config)
        assert attributor1.config == string_config
        
        # Test with numeric config (unusual but should work)
        numeric_config = 123
        attributor2 = PerformanceAttributor(config=numeric_config)
        assert attributor2.config == numeric_config
        
        # Test with list config (unusual but should work)
        list_config = ["config1", "config2"]
        attributor3 = PerformanceAttributor(config=list_config)
        assert attributor3.config == list_config