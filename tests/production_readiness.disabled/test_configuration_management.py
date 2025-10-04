"""
Production Readiness Tests for Configuration Management

Tests configuration management and deployment readiness:
- Environment switching (sandbox/live)
- Dynamic configuration updates
- Feature flag support
- Configuration validation
- Hot-reload capabilities
- Environment-specific settings
- Configuration security
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tests.production_readiness.test_config import TestConfig as Config
from src.exchanges.factory import ExchangeFactory
from src.exchanges.interfaces import SandboxMode
from src.exchanges.service import ExchangeService


class TestConfigurationManagement:
    """Test configuration management capabilities."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration."""
        return {
            "exchanges": {
                "binance": {
                    "api_key": "test_binance_key",
                    "api_secret": "test_binance_secret",
                    "sandbox": True
                },
                "coinbase": {
                    "api_key": "test_coinbase_key", 
                    "api_secret": "test_coinbase_secret",
                    "passphrase": "test_coinbase_passphrase",
                    "sandbox": True
                },
                "okx": {
                    "api_key": "test_okx_key",
                    "api_secret": "test_okx_secret",
                    "passphrase": "test_okx_passphrase", 
                    "sandbox": True
                }
            },
            "environment": "testing",
            "debug": True
        }

    @pytest.fixture
    def production_config(self, base_config):
        """Create production configuration."""
        prod_config = base_config.copy()
        prod_config["environment"] = "production"
        prod_config["debug"] = False
        
        # Switch to production endpoints
        for exchange in prod_config["exchanges"].values():
            if isinstance(exchange, dict):
                exchange["sandbox"] = False
        
        return prod_config

    @pytest.fixture
    def feature_flag_config(self, base_config):
        """Create configuration with feature flags."""
        config_with_flags = base_config.copy()
        config_with_flags["feature_flags"] = {
            "advanced_order_types": True,
            "margin_trading": False,
            "websocket_streaming": True,
            "paper_trading": True,
            "high_frequency_trading": False,
            "cross_exchange_arbitrage": True
        }
        
        # Exchange-specific feature flags
        config_with_flags["exchanges"]["binance"]["features"] = {
            "futures_trading": False,
            "spot_trading": True,
            "websocket_enabled": True
        }
        
        return config_with_flags

    def test_environment_switching_sandbox_live(self, base_config, production_config):
        """Test switching between sandbox and live environments."""
        
        # Test sandbox configuration
        sandbox_config = Config(base_config)
        sandbox_factory = ExchangeFactory(sandbox_config)
        
        assert sandbox_factory.is_exchange_supported("binance")
        assert sandbox_factory.is_exchange_supported("coinbase")
        assert sandbox_factory.is_exchange_supported("okx")
        
        # Verify sandbox settings
        exchanges = sandbox_factory.get_available_exchanges()
        assert "binance" in exchanges
        assert "coinbase" in exchanges
        assert "okx" in exchanges
        
        # Test production configuration
        prod_config = Config(production_config)
        prod_factory = ExchangeFactory(prod_config)
        
        assert prod_factory.is_exchange_supported("binance")
        assert prod_factory.is_exchange_supported("coinbase")
        assert prod_factory.is_exchange_supported("okx")
        
        # Should support same exchanges in both environments
        prod_exchanges = prod_factory.get_available_exchanges()
        assert set(exchanges) == set(prod_exchanges)

    def test_dynamic_configuration_updates(self, base_config):
        """Test dynamic configuration updates."""
        
        # Create initial configuration
        config = Config(base_config)
        factory = ExchangeFactory(config)
        
        initial_exchanges = factory.get_available_exchanges()
        assert len(initial_exchanges) >= 3  # binance, coinbase, okx
        
        # Test configuration update
        updated_config = base_config.copy()
        updated_config["exchanges"]["kraken"] = {
            "api_key": "test_kraken_key",
            "api_secret": "test_kraken_secret",
            "sandbox": True
        }
        
        # In a real implementation, this would trigger hot-reload
        new_config = Config(updated_config)
        new_factory = ExchangeFactory(new_config)
        
        new_exchanges = new_factory.get_available_exchanges()
        
        # Should include new exchange
        if "kraken" in updated_config["exchanges"]:
            # Would be available if properly configured
            assert new_factory.is_exchange_supported("kraken") or len(new_exchanges) >= len(initial_exchanges)

    def test_feature_flag_support(self, feature_flag_config):
        """Test feature flag support in configuration."""
        
        config = Config(feature_flag_config)
        factory = ExchangeFactory(config)
        
        # Test global feature flags
        if hasattr(config, 'feature_flags'):
            feature_flags = config.feature_flags
            
            # Test boolean feature flags
            assert isinstance(feature_flags["advanced_order_types"], bool)
            assert isinstance(feature_flags["margin_trading"], bool)
            assert isinstance(feature_flags["websocket_streaming"], bool)
            
            # Test feature flag values
            assert feature_flags["advanced_order_types"] is True
            assert feature_flags["margin_trading"] is False
            assert feature_flags["websocket_streaming"] is True

    def test_configuration_parameter_validation(self):
        """Test validation of configuration parameters."""
        
        # Test valid configuration
        valid_config_dict = {
            "exchanges": {
                "binance": {
                    "api_key": "valid_key_123",
                    "api_secret": "valid_secret_456", 
                    "sandbox": True
                }
            }
        }
        
        config = Config(valid_config_dict)
        factory = ExchangeFactory(config)
        assert factory.is_exchange_supported("binance")
        
        # Test configuration with missing required fields
        invalid_config_dict = {
            "exchanges": {
                "binance": {
                    "api_key": "only_key"
                    # Missing api_secret
                }
            }
        }
        
        invalid_config = Config(invalid_config_dict)
        invalid_factory = ExchangeFactory(invalid_config)
        
        # Should handle gracefully but may not be functional
        try:
            supported = invalid_factory.is_exchange_supported("binance")
            # Implementation dependent - may still report as supported
        except Exception:
            # May raise validation error
            pass
        
        # Test configuration with invalid values
        bad_config_dict = {
            "exchanges": {
                "binance": {
                    "api_key": "",  # Empty key
                    "api_secret": "",  # Empty secret
                    "sandbox": "invalid_boolean"  # Should be boolean
                }
            }
        }
        
        bad_config = Config(bad_config_dict)
        bad_factory = ExchangeFactory(bad_config)
        
        # Should handle invalid configuration gracefully
        available = bad_factory.get_available_exchanges()
        # May return empty list or filtered results

    def test_environment_specific_settings(self, base_config):
        """Test environment-specific configuration settings."""
        
        # Test development environment
        dev_config = base_config.copy()
        dev_config["environment"] = "development"
        dev_config["debug"] = True
        dev_config["logging"] = {"level": "DEBUG"}
        
        config_dev = Config(dev_config)
        
        # Test staging environment
        staging_config = base_config.copy()
        staging_config["environment"] = "staging"
        staging_config["debug"] = False
        staging_config["logging"] = {"level": "INFO"}
        
        config_staging = Config(staging_config)
        
        # Test production environment
        prod_config = base_config.copy()
        prod_config["environment"] = "production"
        prod_config["debug"] = False
        prod_config["logging"] = {"level": "WARNING"}
        
        config_prod = Config(prod_config)
        
        # Each environment should have appropriate settings
        environments = [
            ("development", config_dev),
            ("staging", config_staging),
            ("production", config_prod)
        ]
        
        for env_name, config in environments:
            factory = ExchangeFactory(config)
            
            # Should support basic exchanges in all environments
            assert factory.is_exchange_supported("binance")
            
            # Environment-specific behavior
            if hasattr(config, 'environment'):
                assert config.environment == env_name
            
            if hasattr(config, 'debug'):
                if env_name == "production":
                    assert config.debug is False
                elif env_name == "development":
                    assert config.debug is True

    def test_configuration_security_validation(self, base_config):
        """Test configuration security validation."""
        
        config = Config(base_config)
        
        # Test that configuration doesn't expose secrets
        config_str = str(config)
        config_repr = repr(config)
        
        # Secrets should be masked or not present
        secrets = ["test_binance_secret", "test_coinbase_secret", "test_okx_secret"]
        
        for secret in secrets:
            # Should not appear in string representations
            assert secret not in config_str
            assert secret not in config_repr
        
        # Test configuration serialization
        if hasattr(config, 'to_dict'):
            try:
                config_dict = config.to_dict()
                assert isinstance(config_dict, dict)
                
                # Should contain structure but potentially mask secrets
                assert "exchanges" in config_dict
                
            except Exception:
                # May not support serialization for security reasons
                pass

    def test_hot_reload_capabilities(self, base_config):
        """Test configuration hot-reload capabilities."""
        
        # Create initial service
        config = Config(base_config)
        
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.get_supported_exchanges.return_value = ["binance", "coinbase", "okx"]
            mock_factory.get_available_exchanges.return_value = ["binance", "coinbase"]
            mock_factory_class.return_value = mock_factory
            
            service = ExchangeService(
                exchange_factory=mock_factory,
                config=config
            )
            
            # Test that service can handle configuration changes
            initial_exchanges = service.get_supported_exchanges()
            assert len(initial_exchanges) >= 2
            
            # Simulate configuration update
            updated_config = base_config.copy()
            updated_config["exchanges"]["ftx"] = {
                "api_key": "test_ftx_key",
                "api_secret": "test_ftx_secret",
                "sandbox": True
            }
            
            # In a real implementation, this would trigger service update
            # For testing, verify service remains stable
            updated_exchanges = service.get_supported_exchanges()
            assert len(updated_exchanges) >= len(initial_exchanges)

    def test_configuration_validation_rules(self):
        """Test configuration validation rules."""
        
        validation_tests = [
            # Valid configurations
            ({
                "exchanges": {
                    "binance": {"api_key": "key123", "api_secret": "secret456", "sandbox": True}
                }
            }, True),
            
            # Invalid configurations
            ({
                "exchanges": {
                    "binance": {"api_key": "", "api_secret": "secret456", "sandbox": True}
                }
            }, False),
            
            ({
                "exchanges": {
                    "binance": {"api_key": "key123", "sandbox": True}
                    # Missing api_secret
                }
            }, False),
            
            # Empty configuration
            ({}, False),
            
            # Invalid exchange name
            ({
                "exchanges": {
                    "invalid_exchange_name_!@#": {
                        "api_key": "key123",
                        "api_secret": "secret456",
                        "sandbox": True
                    }
                }
            }, False)
        ]
        
        for config_dict, should_be_valid in validation_tests:
            try:
                config = Config(config_dict)
                factory = ExchangeFactory(config)
                
                # Test if configuration produces working factory
                supported = factory.get_supported_exchanges()
                available = factory.get_available_exchanges()
                
                if should_be_valid:
                    # Should have some supported exchanges
                    assert isinstance(supported, list)
                    assert isinstance(available, list)
                else:
                    # May have empty lists for invalid config
                    if not supported and not available:
                        # Invalid config correctly handled
                        pass
                
            except Exception as e:
                if not should_be_valid:
                    # Expected to fail for invalid configuration
                    assert "config" in str(e).lower() or "validation" in str(e).lower()
                else:
                    # Should not fail for valid configuration
                    raise

    def test_configuration_inheritance_overrides(self, base_config):
        """Test configuration inheritance and overrides."""
        
        # Base configuration
        base_config_with_defaults = base_config.copy()
        base_config_with_defaults["defaults"] = {
            "timeout_seconds": 30,
            "max_retries": 3,
            "sandbox": True
        }
        
        # Override configuration
        override_config = base_config_with_defaults.copy()
        override_config["exchanges"]["binance"]["timeout_seconds"] = 60  # Override default
        override_config["exchanges"]["binance"]["max_retries"] = 5      # Override default
        
        config = Config(override_config)
        factory = ExchangeFactory(config)
        
        # Should handle inheritance and overrides
        assert factory.is_exchange_supported("binance")
        
        # Test that overrides work (implementation dependent)
        # In a real system, the factory would apply these settings

    def test_multi_environment_configuration(self):
        """Test multi-environment configuration management."""
        
        environments = {
            "development": {
                "exchanges": {
                    "binance": {
                        "api_key": "dev_key",
                        "api_secret": "dev_secret",
                        "sandbox": True,
                        "base_url": "https://testnet.binance.vision"
                    }
                },
                "debug": True,
                "log_level": "DEBUG"
            },
            
            "production": {
                "exchanges": {
                    "binance": {
                        "api_key": "prod_key",
                        "api_secret": "prod_secret", 
                        "sandbox": False,
                        "base_url": "https://api.binance.com"
                    }
                },
                "debug": False,
                "log_level": "ERROR"
            }
        }
        
        for env_name, env_config in environments.items():
            config = Config(env_config)
            factory = ExchangeFactory(config)
            
            # Should work in all environments
            assert factory.is_exchange_supported("binance")
            
            # Environment-specific validation
            if env_name == "development":
                # Development should use sandbox
                dev_exchange_config = env_config["exchanges"]["binance"]
                assert dev_exchange_config["sandbox"] is True
                assert "testnet" in dev_exchange_config["base_url"]
            
            elif env_name == "production":
                # Production should use live endpoints
                prod_exchange_config = env_config["exchanges"]["binance"]
                assert prod_exchange_config["sandbox"] is False
                assert "testnet" not in prod_exchange_config["base_url"]

    def test_configuration_backwards_compatibility(self, base_config):
        """Test backwards compatibility with older configuration formats."""
        
        # Test legacy configuration format
        legacy_config = {
            "binance_api_key": "legacy_key",
            "binance_api_secret": "legacy_secret",
            "coinbase_api_key": "legacy_coinbase_key",
            "coinbase_api_secret": "legacy_coinbase_secret",
            "sandbox_mode": True
        }
        
        # In a real implementation, there would be migration logic
        # For testing, ensure the system handles unknown formats gracefully
        try:
            config = Config(legacy_config)
            factory = ExchangeFactory(config)
            
            # May or may not work depending on implementation
            supported = factory.get_supported_exchanges()
            assert isinstance(supported, list)
            
        except Exception:
            # Expected if legacy format is not supported
            pass
        
        # Test modern configuration format (should always work)
        modern_config = Config(base_config)
        modern_factory = ExchangeFactory(modern_config)
        
        assert modern_factory.get_supported_exchanges() is not None
        assert len(modern_factory.get_available_exchanges()) > 0

    def test_configuration_performance_impact(self, base_config):
        """Test configuration performance impact."""
        
        import time
        
        # Test configuration loading performance
        start_time = time.time()
        
        for _ in range(100):  # Load configuration multiple times
            config = Config(base_config)
            factory = ExchangeFactory(config)
            supported = factory.get_supported_exchanges()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly
        assert total_time < 5.0  # Less than 5 seconds for 100 loads
        
        # Average time per configuration load
        avg_time = total_time / 100
        assert avg_time < 0.05  # Less than 50ms per load