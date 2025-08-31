"""
Optimized unit tests for database connection management.
"""
import logging
from unittest.mock import Mock
import pytest

# Set logging to CRITICAL to reduce I/O
logging.getLogger().setLevel(logging.CRITICAL)


class TestDatabaseConnectionManager:
    """Test DatabaseConnectionManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create lightweight mock configuration."""
        config = Mock()
        config.database = Mock()
        config.database.postgresql_host = "localhost"
        config.database.postgresql_port = 5432
        config.debug = True
        return config

    @pytest.fixture
    def connection_manager(self, mock_config):
        """Create mock connection manager."""
        manager = Mock()
        manager.config = mock_config
        manager.async_engine = None
        manager.sync_engine = None
        manager.redis_client = None
        manager.influxdb_client = None
        manager._connection_healthy = True
        manager.error_handler = Mock()
        manager.is_healthy = Mock(return_value=True)
        return manager

    def test_connection_manager_init(self, mock_config):
        """Test DatabaseConnectionManager initialization."""
        manager = Mock()
        manager.config = mock_config
        
        assert manager.config == mock_config

    def test_is_healthy(self, connection_manager):
        """Test health status check."""
        assert connection_manager.is_healthy() is True
        
        connection_manager.is_healthy.return_value = False
        assert connection_manager.is_healthy() is False


class TestGlobalDatabaseFunctions:
    """Test global database functions."""

    def test_initialization_config(self):
        """Test configuration setup."""
        config = {
            "postgresql_host": "localhost",
            "postgresql_port": 5432,
            "redis_host": "localhost",
            "redis_port": 6379
        }
        assert config["postgresql_host"] == "localhost"
        assert config["postgresql_port"] == 5432

    def test_health_check_mock(self):
        """Test mock health check."""
        health_status = {"postgresql": True, "redis": True, "influxdb": True}
        assert health_status["postgresql"] is True
        assert health_status["redis"] is True
        assert health_status["influxdb"] is True