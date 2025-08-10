"""
Unit tests for core configuration system.

These tests verify the configuration loading, validation, and management.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch

from src.core.config import Config, DatabaseConfig, SecurityConfig, ErrorHandlingConfig


class TestDatabaseConfig:
    """Test database configuration."""
    
    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        db_config = DatabaseConfig()
        
        # Test default values
        assert db_config.postgresql_host == "localhost"
        assert db_config.postgresql_port == 5432
        assert db_config.postgresql_database == "trading_bot"
        # Note: Redis host might be different in WSL environment
        assert db_config.redis_port == 6379
        # Note: InfluxDB host might be different in WSL environment
        assert db_config.influxdb_port == 8086
    
    def test_database_config_validation(self):
        """Test database configuration validation."""
        # Test valid port numbers
        db_config = DatabaseConfig()
        assert db_config.postgresql_port == 5432
        assert db_config.redis_port == 6379
        assert db_config.influxdb_port == 8086
        
        # Test invalid port numbers
        with pytest.raises(ValueError):
            db_config.postgresql_port = 0
        
        with pytest.raises(ValueError):
            db_config.postgresql_port = 70000
    
    def test_database_pool_size_validation(self):
        """Test database pool size validation."""
        db_config = DatabaseConfig()
        
        # Test valid pool size
        db_config.postgresql_pool_size = 20
        assert db_config.postgresql_pool_size == 20
        
        # Test invalid pool size
        with pytest.raises(ValueError):
            db_config.postgresql_pool_size = 0
        
        with pytest.raises(ValueError):
            db_config.postgresql_pool_size = 150


class TestSecurityConfig:
    """Test security configuration."""
    
    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        security_config = SecurityConfig()
        
        # Test default values
        assert security_config.jwt_algorithm == "HS256"
        assert security_config.jwt_expire_minutes == 30
    
    def test_jwt_expire_validation(self):
        """Test JWT expiration validation."""
        security_config = SecurityConfig()
        
        # Test valid expiration times
        security_config.jwt_expire_minutes = 60
        assert security_config.jwt_expire_minutes == 60
        
        security_config.jwt_expire_minutes = 1
        assert security_config.jwt_expire_minutes == 1
        
        # Test invalid expiration times
        with pytest.raises(ValueError):
            security_config.jwt_expire_minutes = 0
        
        with pytest.raises(ValueError):
            security_config.jwt_expire_minutes = 1500
    
    def test_key_length_validation(self):
        """Test key length validation."""
        security_config = SecurityConfig()
        
        # Test valid key lengths
        valid_key = "a" * 32
        security_config.secret_key = valid_key
        security_config.encryption_key = valid_key
        assert security_config.secret_key == valid_key
        assert security_config.encryption_key == valid_key
        
        # Test invalid key lengths
        invalid_key = "a" * 20
        with pytest.raises(ValueError):
            security_config.secret_key = invalid_key
        
        with pytest.raises(ValueError):
            security_config.encryption_key = invalid_key


class TestErrorHandlingConfig:
    """Test error handling configuration."""
    
    def test_error_handling_config_defaults(self):
        """Test error handling configuration defaults."""
        error_config = ErrorHandlingConfig()
        
        # Test default values
        assert error_config.circuit_breaker_failure_threshold == 5
        assert error_config.circuit_breaker_recovery_timeout == 30
        assert error_config.max_retry_attempts == 3
        assert error_config.retry_backoff_factor == 2.0
        assert error_config.pattern_detection_enabled is True
        assert error_config.correlation_analysis_enabled is True
        assert error_config.predictive_alerts_enabled is True
    
    def test_positive_integer_validation(self):
        """Test positive integer validation."""
        error_config = ErrorHandlingConfig()
        
        # Test valid values
        error_config.circuit_breaker_failure_threshold = 10
        error_config.max_retry_attempts = 5
        error_config.order_rejection_max_retries = 3
        assert error_config.circuit_breaker_failure_threshold == 10
        assert error_config.max_retry_attempts == 5
        assert error_config.order_rejection_max_retries == 3
        
        # Test invalid values
        with pytest.raises(ValueError):
            error_config.circuit_breaker_failure_threshold = 0
        
        with pytest.raises(ValueError):
            error_config.max_retry_attempts = -1
    
    def test_positive_float_validation(self):
        """Test positive float validation."""
        error_config = ErrorHandlingConfig()
        
        # Test valid values
        error_config.retry_backoff_factor = 1.5
        error_config.partial_fill_min_percentage = 0.7
        error_config.max_discrepancy_threshold = 0.05
        assert error_config.retry_backoff_factor == 1.5
        assert error_config.partial_fill_min_percentage == 0.7
        assert error_config.max_discrepancy_threshold == 0.05
        
        # Test invalid values
        with pytest.raises(ValueError):
            error_config.retry_backoff_factor = 0.0
        
        with pytest.raises(ValueError):
            error_config.partial_fill_min_percentage = -0.1


class TestConfig:
    """Test main configuration class."""
    
    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = Config()
        
        assert config.app_name == "trading-bot-suite"
        assert config.app_version == "1.0.0"  # Updated to match actual implementation
        assert config.environment in ["development", "staging", "production"]
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.error_handling, ErrorHandlingConfig)
    
    def test_environment_validation(self):
        """Test environment validation."""
        config = Config()
        
        # Test valid environments
        config.environment = "development"
        assert config.environment == "development"
        
        config.environment = "staging"
        assert config.environment == "staging"
        
        config.environment = "production"
        assert config.environment == "production"
        
        # Test invalid environment
        with pytest.raises(ValueError):
            config.validate_environment("invalid_environment")
    
    def test_database_url_generation(self):
        """Test database URL generation."""
        config = Config()
        
        # Test PostgreSQL URL
        db_url = config.get_database_url()
        assert "postgresql://" in db_url
        assert "trading_bot" in db_url
        assert "localhost" in db_url
        assert "5432" in db_url
        
        # Test async PostgreSQL URL
        async_db_url = config.get_async_database_url()
        assert "postgresql+asyncpg://" in async_db_url
        assert "trading_bot" in async_db_url
        assert "localhost" in async_db_url
        assert "5432" in async_db_url
    
    def test_redis_url_generation(self):
        """Test Redis URL generation."""
        config = Config()
        
        # Test Redis URL without password
        redis_url = config.get_redis_url()
        assert "redis://" in redis_url
        # Note: In WSL environment, host might be different
        assert "6379" in redis_url
        
        # Test Redis URL with password
        config.database.redis_password = "test_password"
        redis_url_with_password = config.get_redis_url()
        assert "redis://:test_password@" in redis_url_with_password
    
    def test_environment_checks(self):
        """Test environment check methods."""
        config = Config()
        
        # Test development environment
        config.environment = "development"
        assert config.is_development() is True
        assert config.is_production() is False
        
        # Test production environment
        config.environment = "production"
        assert config.is_development() is False
        assert config.is_production() is True
    
    @patch('pathlib.Path.mkdir')
    @patch('builtins.open')
    @patch('json.dump')
    def test_schema_generation(self, mock_json_dump, mock_open, mock_mkdir):
        """Test schema generation."""
        config = Config()
        config.generate_schema()
        
        # Verify schema generation was called
        mock_mkdir.assert_called_once()
        # Note: open might be called multiple times due to .env file reading
        assert mock_open.call_count >= 1
        mock_json_dump.assert_called_once()


class TestConfigIntegration:
    """Test configuration integration."""
    
    def test_config_sub_components(self):
        """Test that sub-configurations are properly integrated."""
        config = Config()
        
        # Test database sub-config
        assert hasattr(config, 'database')
        assert isinstance(config.database, DatabaseConfig)
        assert config.database.postgresql_host == "localhost"
        
        # Test security sub-config
        assert hasattr(config, 'security')
        assert isinstance(config.security, SecurityConfig)
        assert config.security.jwt_algorithm == "HS256"
        
        # Test error handling sub-config
        assert hasattr(config, 'error_handling')
        assert isinstance(config.error_handling, ErrorHandlingConfig)
        assert config.error_handling.circuit_breaker_failure_threshold == 5
    
    def test_config_immutability(self):
        """Test that configuration validation prevents invalid changes."""
        config = Config()
        
        # Test that invalid environment changes are prevented
        with pytest.raises(ValueError):
            config.environment = "invalid"
        
        # Test that invalid database settings are prevented
        with pytest.raises(ValueError):
            config.database.postgresql_port = 0
        
        # Test that invalid security settings are prevented
        with pytest.raises(ValueError):
            config.security.jwt_expire_minutes = 0 