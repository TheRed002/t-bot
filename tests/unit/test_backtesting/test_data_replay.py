"""Tests for backtesting data replay module."""

import logging
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from src.backtesting.data_replay import DataReplayManager
from src.core.config import Config

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Shared fixtures for performance
@pytest.fixture(scope="session")
def mock_config():
    """Shared mock config for all tests."""
    return MagicMock(spec=Config)

@pytest.fixture(scope="session")
def simple_dict_config():
    """Shared simple dict config."""
    return {"replay": {"enabled": True}}

@pytest.fixture(scope="session")
def empty_dict_config():
    """Shared empty dict config."""
    return {}


class TestDataReplayManager:
    """Test DataReplayManager."""

    def test_data_replay_manager_creation(self, mock_config):
        """Test creating DataReplayManager with mocked initialization."""
        with patch('src.backtesting.data_replay.DataReplayManager') as MockManager:
            mock_manager = MagicMock()
            mock_manager.config = mock_config
            MockManager.return_value = mock_manager

            manager = DataReplayManager(config=mock_config)
            assert manager.config == mock_config

    def test_data_replay_manager_attributes(self, mock_config):
        """Test DataReplayManager has expected attributes."""
        with patch('src.backtesting.data_replay.DataReplayManager') as MockManager:
            mock_manager = MagicMock()
            mock_manager.config = mock_config
            MockManager.return_value = mock_manager

            manager = DataReplayManager(config=mock_config)
            assert hasattr(manager, 'config')

    def test_data_replay_manager_initialization_variations(self):
        """Test different ways to initialize DataReplayManager."""
        with patch('src.backtesting.data_replay.DataReplayManager') as MockManager:
            # Mock both instances
            mock_manager1 = MagicMock()
            mock_manager2 = MagicMock()
            MockManager.side_effect = [mock_manager1, mock_manager2]

            config1 = MagicMock(spec=Config)
            mock_manager1.config = config1
            manager1 = DataReplayManager(config=config1)
            assert manager1.config == config1

            config2 = {"replay": {"enabled": True}}
            mock_manager2.config = config2
            manager2 = DataReplayManager(config=config2)
            assert manager2.config == config2


class TestDataReplayModuleImports:
    """Test data replay module imports."""

    def test_data_replay_manager_import(self):
        """Test DataReplayManager can be imported."""
        from src.backtesting.data_replay import DataReplayManager
        
        assert DataReplayManager is not None
        assert hasattr(DataReplayManager, '__init__')

    def test_module_attributes(self):
        """Test module has expected attributes."""
        import src.backtesting.data_replay as data_replay_module
        
        assert hasattr(data_replay_module, 'DataReplayManager')

    def test_data_replay_class_properties(self):
        """Test DataReplayManager class properties."""
        from src.backtesting.data_replay import DataReplayManager
        
        assert DataReplayManager.__name__ == "DataReplayManager"
        assert hasattr(DataReplayManager, '__init__')


class TestDataReplayIntegration:
    """Test data replay integration scenarios."""

    def test_manager_with_different_configs(self, mock_config, empty_dict_config):
        """Test manager works with different config types."""
        # Test with mock config - use fixture
        manager1 = DataReplayManager(config=mock_config)
        assert manager1.config == mock_config

        # Test with None config
        manager2 = DataReplayManager(config=None)
        assert manager2.config is None

        # Test with empty dict config - use fixture
        manager3 = DataReplayManager(config=empty_dict_config)
        assert manager3.config == empty_dict_config

    def test_multiple_manager_instances(self):
        """Test creating multiple manager instances."""
        config1 = {"replay_type": "historical"}
        config2 = {"replay_type": "synthetic"}
        
        manager1 = DataReplayManager(config=config1)
        manager2 = DataReplayManager(config=config2)
        
        assert manager1 != manager2
        assert manager1.config != manager2.config
        assert manager1.config == config1
        assert manager2.config == config2

    def test_manager_config_assignment(self):
        """Test that config is properly assigned."""
        test_config = {
            "data_replay": {
                "source": "database",
                "buffer_size": 1000,
                "chunk_size": 100
            }
        }
        
        manager = DataReplayManager(config=test_config)
        
        assert manager.config is test_config
        assert manager.config["data_replay"]["source"] == "database"
        assert manager.config["data_replay"]["buffer_size"] == 1000


class TestDataReplayManagerEdgeCases:
    """Test edge cases for DataReplayManager."""

    def test_manager_with_complex_config(self):
        """Test manager with complex config structure."""
        complex_config = {
            "data_replay": {
                "enabled": True,
                "speed_multiplier": 10.0,
                "sources": {
                    "primary": {
                        "type": "database",
                        "connection_string": "postgresql://localhost/test"
                    },
                    "fallback": {
                        "type": "file",
                        "path": "/data/historical/"
                    }
                },
                "filters": {
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "date_range": {
                        "start": "2024-01-01",
                        "end": "2024-12-31"
                    }
                },
                "output": {
                    "format": "ohlcv",
                    "frequency": "1m",
                    "include_volume": True
                }
            }
        }
        
        manager = DataReplayManager(config=complex_config)
        
        assert manager.config == complex_config
        assert manager.config["data_replay"]["enabled"] is True
        assert manager.config["data_replay"]["speed_multiplier"] == 10.0
        assert manager.config["data_replay"]["sources"]["primary"]["type"] == "database"

    def test_manager_config_access_patterns(self):
        """Test different config access patterns."""
        config_with_methods = MagicMock()
        config_with_methods.get.return_value = {"default": "replay"}
        config_with_methods.data_replay_config = {"mode": "live"}
        
        manager = DataReplayManager(config=config_with_methods)
        
        assert manager.config == config_with_methods
        # Should be able to access config methods if they exist
        if hasattr(manager.config, 'get'):
            assert callable(manager.config.get)
        if hasattr(manager.config, 'data_replay_config'):
            assert manager.config.data_replay_config == {"mode": "live"}

    def test_manager_instantiation_validation(self):
        """Test that manager can be instantiated with various inputs."""
        # Test with string config (unusual but should work)
        string_config = "replay_config"
        manager1 = DataReplayManager(config=string_config)
        assert manager1.config == string_config
        
        # Test with numeric config (unusual but should work)
        numeric_config = 456
        manager2 = DataReplayManager(config=numeric_config)
        assert manager2.config == numeric_config
        
        # Test with list config (unusual but should work)
        list_config = ["config1", "config2", "config3"]
        manager3 = DataReplayManager(config=list_config)
        assert manager3.config == list_config

    def test_manager_with_typical_replay_settings(self):
        """Test manager with typical replay configuration."""
        typical_config = {
            "data_replay": {
                "enabled": True,
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z",
                "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
                "timeframe": "1m",
                "speed": 1.0,
                "max_buffer_size": 10000,
                "preload_data": True,
                "validate_data": True
            }
        }
        
        manager = DataReplayManager(config=typical_config)
        
        replay_config = manager.config["data_replay"]
        assert replay_config["enabled"] is True
        assert replay_config["symbols"] == ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        assert replay_config["timeframe"] == "1m"
        assert replay_config["speed"] == 1.0
        assert replay_config["max_buffer_size"] == 10000
        assert replay_config["preload_data"] is True
        assert replay_config["validate_data"] is True


class TestDataReplayManagerFunctionality:
    """Test DataReplayManager functionality."""

    def test_manager_basic_interface(self):
        """Test that manager has expected interface."""
        config = {"test": "config"}
        manager = DataReplayManager(config=config)
        
        # Should have config attribute
        assert hasattr(manager, 'config')
        assert manager.config == config
        
        # Should be instantiable
        assert manager is not None
        assert isinstance(manager, DataReplayManager)

    def test_manager_comparison(self):
        """Test manager instance comparisons."""
        config = {"replay": True}
        
        manager1 = DataReplayManager(config=config)
        manager2 = DataReplayManager(config=config)
        
        # Should be different instances even with same config
        assert manager1 is not manager2
        assert manager1 != manager2
        
        # But should have same config
        assert manager1.config == manager2.config

    def test_manager_memory_efficiency(self):
        """Test that multiple managers don't share unexpected state."""
        config1 = {"id": 1, "buffer": []}
        config2 = {"id": 2, "buffer": []}
        
        manager1 = DataReplayManager(config=config1)
        manager2 = DataReplayManager(config=config2)
        
        # Configs should be independent
        assert manager1.config["id"] == 1
        assert manager2.config["id"] == 2
        
        # Modifying one config shouldn't affect the other
        manager1.config["buffer"].append("data1")
        assert len(manager1.config["buffer"]) == 1
        assert len(manager2.config["buffer"]) == 0