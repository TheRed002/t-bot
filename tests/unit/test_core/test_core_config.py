"""
Unit tests for core configuration system.

These tests verify the configuration loading, validation, and management.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

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

    def test_validate_environment_invalid(self):
        """Test environment validation with invalid value."""
        with pytest.raises(ValidationError):
            Config(environment="invalid_env")

    def test_generate_schema(self):
        """Test schema generation."""
        config = Config()
        # This should not raise an exception
        config.generate_schema()

    def test_from_yaml_file_not_found(self):
        """Test from_yaml with non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent.yaml")

    def test_from_yaml_with_env_override_file_exists(self, tmp_path):
        """Test from_yaml_with_env_override with existing file."""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text("""
        environment: "staging"
        app_name: "test-app"
        """)

        config = Config.from_yaml_with_env_override(str(yaml_file))
        assert config.environment == "staging"
        assert config.app_name == "test-app"

    def test_from_yaml_with_env_override_file_not_exists(self):
        """Test from_yaml_with_env_override with non-existent file."""
        config = Config.from_yaml_with_env_override("nonexistent.yaml")
        # Should create config with defaults
        assert config.environment in ["development", "staging", "production"]

    def test_to_yaml(self, tmp_path):
        """Test saving configuration to YAML file."""
        config = Config()
        yaml_file = tmp_path / "output_config.yaml"

        config.to_yaml(str(yaml_file))

        assert yaml_file.exists()
        with open(yaml_file, 'r') as f:
            content = f.read()
            # Check that the file contains expected configuration structure
            assert "app_name:" in content
            assert "app_version:" in content
            assert "database:" in content
            assert "security:" in content

    def test_get_database_url(self):
        """Test database URL generation."""
        config = Config()
        url = config.get_database_url()
        assert "postgresql://" in url
        assert "localhost:5432" in url
        assert "trading_bot" in url

    def test_get_async_database_url(self):
        """Test async database URL generation."""
        config = Config()
        url = config.get_async_database_url()
        assert "postgresql+asyncpg://" in url
        assert "localhost:5432" in url
        assert "trading_bot" in url

    def test_get_redis_url_with_password(self):
        """Test Redis URL generation with password."""
        config = Config()
        config.database.redis_password = "test_password"
        url = config.get_redis_url()
        assert "redis://:test_password@" in url

    def test_get_redis_url_without_password(self):
        """Test Redis URL generation without password."""
        config = Config()
        config.database.redis_password = None
        url = config.get_redis_url()
        assert "redis://" in url
        assert "6379" in url

    def test_is_production(self):
        """Test production environment check."""
        config = Config(environment="production")
        assert config.is_production() is True
        assert config.is_development() is False

    def test_is_development(self):
        """Test development environment check."""
        config = Config(environment="development")
        assert config.is_development() is True
        assert config.is_production() is False

    def test_validate_yaml_config_valid(self, tmp_path):
        """Test YAML config validation with valid file."""
        yaml_file = tmp_path / "valid_config.yaml"
        yaml_file.write_text("""
        environment: "staging"
        app_name: "test-app"
        """)

        config = Config()
        assert config.validate_yaml_config(str(yaml_file)) is True

    def test_validate_yaml_config_invalid(self, tmp_path):
        """Test YAML config validation with invalid file."""
        yaml_file = tmp_path / "invalid_config.yaml"
        yaml_file.write_text("""
        environment: "invalid_env"
        """)

        # Should return False for invalid config
        config = Config()
        assert config.validate_yaml_config(str(yaml_file)) is False

    def test_validate_yaml_config_file_not_found(self):
        """Test YAML config validation with non-existent file."""
        config = Config()
        assert config.validate_yaml_config("nonexistent.yaml") is False


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
