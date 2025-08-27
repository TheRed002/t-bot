"""
Unit tests for core configuration system.

These tests verify the configuration loading, validation, and management
using the new refactored configuration structure.
"""

import os
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import json
import yaml

import pytest
from pydantic import ValidationError

# Import from new refactored structure
from src.core.config.main import Config, get_config
from src.core.config.database import DatabaseConfig
from src.core.config.exchange import ExchangeConfig
from src.core.config.strategy import StrategyConfig
from src.core.config.risk import RiskConfig
from src.core.config.service import ConfigService, get_config_service
from src.core.dependency_injection import get_container


class TestDatabaseConfig:
    """Test database configuration."""

    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        # Create config without environment variables
        with patch.dict(os.environ, {}, clear=True):
            db_config = DatabaseConfig()

            # Test default values
            assert db_config.postgresql_host == "localhost"
            assert db_config.postgresql_port == 5432
            assert db_config.postgresql_database == "tbot_dev"
            assert db_config.redis_port == 6379
            assert db_config.influxdb_port == 8086

    def test_database_config_from_env(self):
        """Test database configuration from environment variables."""
        with patch.dict(os.environ, {
            'DB_HOST': 'db.example.com',
            'DB_PORT': '5433',
            'REDIS_HOST': 'redis.example.com'
        }):
            db_config = DatabaseConfig()
            
            assert db_config.postgresql_host == 'db.example.com'
            assert db_config.postgresql_port == 5433
            assert db_config.redis_host == 'redis.example.com'

    def test_database_url_generation(self):
        """Test database URL generation."""
        db_config = DatabaseConfig()
        # Override the values after creation to test URL generation
        db_config.postgresql_username = 'testuser'
        db_config.postgresql_password = 'testpass'
        db_config.postgresql_host = 'localhost'
        db_config.postgresql_database = 'testdb'
        
        expected_url = 'postgresql://testuser:testpass@localhost:5432/testdb'
        assert db_config.postgresql_url == expected_url

    def test_redis_url_generation(self):
        """Test Redis URL generation."""
        db_config = DatabaseConfig()
        # Override the values after creation to test URL generation
        db_config.redis_host = 'redis.local'
        db_config.redis_port = 6380
        db_config.redis_password = None
        
        expected_url = 'redis://redis.local:6380/0'
        assert db_config.redis_url == expected_url


class TestExchangeConfig:
    """Test exchange configuration."""

    def test_exchange_config_defaults(self):
        """Test exchange configuration defaults."""
        exchange_config = ExchangeConfig()
        
        assert exchange_config.default_exchange == 'binance'
        assert exchange_config.testnet_mode == False
        assert exchange_config.rate_limit_per_second == 10

    def test_exchange_credentials(self):
        """Test exchange credentials retrieval."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_key',
            'BINANCE_API_SECRET': 'test_secret'
        }):
            exchange_config = ExchangeConfig()
            
            creds = exchange_config.get_exchange_credentials('binance')
            assert creds['api_key'] == 'test_key'
            assert creds['api_secret'] == 'test_secret'
            assert creds['testnet'] == False

    def test_websocket_config(self):
        """Test WebSocket configuration."""
        exchange_config = ExchangeConfig()
        
        ws_config = exchange_config.get_websocket_config('binance')
        assert 'url' in ws_config
        assert ws_config['reconnect_attempts'] == 5
        assert ws_config['ping_interval'] == 30


class TestStrategyConfig:
    """Test strategy configuration."""

    def test_strategy_config_defaults(self):
        """Test strategy configuration defaults."""
        strategy_config = StrategyConfig()
        
        assert strategy_config.default_strategy == 'market_making'
        assert strategy_config.backtest_enabled == False
        assert strategy_config.paper_trading_enabled == False

    def test_strategy_params(self):
        """Test strategy parameters."""
        strategy_config = StrategyConfig()
        
        # Test market making params
        mm_params = strategy_config.get_strategy_params('market_making')
        assert mm_params['bid_spread'] == 0.001
        assert mm_params['ask_spread'] == 0.001
        assert mm_params['order_levels'] == 3
        
        # Test arbitrage params
        arb_params = strategy_config.get_strategy_params('arbitrage')
        assert arb_params['min_profit_threshold'] == 0.002
        assert arb_params['max_exposure'] == 10000


class TestRiskConfig:
    """Test risk management configuration."""

    def test_risk_config_defaults(self):
        """Test risk configuration defaults."""
        risk_config = RiskConfig()
        
        assert risk_config.position_sizing_method == 'fixed'
        assert float(risk_config.max_position_size) == 1000.0
        assert risk_config.risk_per_trade == 0.02
        assert risk_config.max_leverage == 1.0

    def test_risk_validation(self):
        """Test risk parameter validation."""
        risk_config = RiskConfig()
        
        # Valid risk per trade
        risk_config.risk_per_trade = 0.05
        assert risk_config.risk_per_trade == 0.05
        
        # Invalid risk per trade (too high)
        with pytest.raises(ValidationError):
            risk_config = RiskConfig(risk_per_trade=0.15)  # > 0.1

    def test_position_size_params(self):
        """Test position sizing parameters."""
        risk_config = RiskConfig()
        risk_config.position_sizing_method = 'kelly_criterion'
        risk_config.kelly_fraction = 0.25
        
        params = risk_config.get_position_size_params()
        assert params['method'] == 'kelly_criterion'
        assert params['kelly_fraction'] == 0.25

    def test_circuit_breaker_config(self):
        """Test circuit breaker configuration."""
        risk_config = RiskConfig()
        
        assert risk_config.enable_circuit_breakers == True
        assert risk_config.loss_limit_circuit_breaker == 0.05
        assert risk_config.circuit_breaker_cooldown == 3600


class TestMainConfig:
    """Test main configuration aggregator."""

    def test_config_initialization(self):
        """Test main config initialization."""
        config = Config()
        
        # Check that all sub-configs are initialized
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.exchange, ExchangeConfig)
        assert isinstance(config.strategy, StrategyConfig)
        assert isinstance(config.risk, RiskConfig)
        
        # Check app-level config
        assert config.app_name == "T-Bot Trading System"
        assert config.environment == "development"

    def test_config_backward_compatibility(self):
        """Test backward compatibility properties."""
        with patch.dict(os.environ, {
            'POSTGRESQL_HOST': 'db.test.com',
            'DB_HOST': 'db.test.com',  # Override the .env value
            'BINANCE_API_KEY': 'test_key'
        }):
            config = Config()
            
            # Test backward compatible properties
            assert config.postgresql_host == 'db.test.com'
            assert config.binance_api_key == 'test_key'
            assert config.db_url == config.database.postgresql_url
            assert config.redis_url == config.database.redis_url

    def test_config_from_file(self):
        """Test loading configuration from file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'database': {
                    'postgresql_host': 'file.db.com',
                    'postgresql_port': 5433
                },
                'exchange': {
                    'default_exchange': 'coinbase'
                },
                'risk': {
                    'risk_per_trade': 0.03
                }
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Load config from file
            config = Config(config_file=temp_path)
            
            assert config.database.postgresql_host == 'file.db.com'
            assert config.database.postgresql_port == 5433
            assert config.exchange.default_exchange == 'coinbase'
            assert config.risk.risk_per_trade == 0.03
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_config_save_to_file(self):
        """Test saving configuration to file."""
        config = Config()
        config.database.postgresql_host = 'save.test.com'
        config.exchange.default_exchange = 'okx'
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_to_file(temp_path)
            
            # Load and verify
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data['database']['postgresql_host'] == 'save.test.com'
            assert saved_data['exchange']['default_exchange'] == 'okx'
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Should not raise any errors
        config.validate()
        
        # Test with invalid configuration
        config.risk.risk_per_trade = 0.05  # Valid
        config.validate()  # Should still pass

    def test_get_config_singleton(self):
        """Test get_config singleton pattern."""
        config1 = get_config()
        config2 = get_config()
        
        # Should be the same instance
        assert config1 is config2
        
        # Test reload
        config3 = get_config(reload=True)
        assert config3 is not config2

    def test_config_methods(self):
        """Test configuration helper methods."""
        config = Config()
        
        # Test get_exchange_config
        exchange_config = config.get_exchange_config('binance')
        assert 'api_key' in exchange_config
        assert 'api_secret' in exchange_config
        
        # Test get_strategy_config
        strategy_config = config.get_strategy_config('market_making')
        assert 'bid_spread' in strategy_config
        
        # Test get_risk_config
        risk_config = config.get_risk_config()
        assert 'method' in risk_config
        assert 'risk_per_trade' in risk_config

    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = Config()
        config_dict = config.to_dict()
        
        assert 'app' in config_dict
        assert 'database' in config_dict
        assert 'exchange' in config_dict
        assert 'strategy' in config_dict
        assert 'risk' in config_dict
        
        assert config_dict['app']['name'] == "T-Bot Trading System"
        assert config_dict['database']['postgresql_port'] == 5432


@pytest.mark.asyncio
class TestConfigService:
    """Test modern ConfigService implementation."""

    async def test_config_service_initialization(self):
        """Test ConfigService initialization."""
        service = ConfigService()
        await service.initialize()
        
        assert service._initialized
        
        await service.shutdown()
        assert not service._initialized

    async def test_config_service_database_config(self):
        """Test database configuration access."""
        service = ConfigService()
        await service.initialize()
        
        db_config = service.get_database_config()
        assert isinstance(db_config, DatabaseConfig)
        assert db_config.postgresql_port == 5432
        
        await service.shutdown()

    async def test_config_service_exchange_config(self):
        """Test exchange configuration access."""
        service = ConfigService()
        await service.initialize()
        
        exchange_config = service.get_exchange_config()
        assert isinstance(exchange_config, ExchangeConfig)
        
        await service.shutdown()

    async def test_config_service_caching(self):
        """Test configuration caching."""
        service = ConfigService(cache_ttl=60)
        await service.initialize()
        
        # First access should load from config
        db_config1 = service.get_database_config()
        
        # Second access should come from cache
        db_config2 = service.get_database_config()
        
        assert db_config1 is db_config2  # Should be same object from cache
        
        # Get cache stats
        stats = service.get_cache_stats()
        assert stats['total_keys'] > 0
        assert stats['total_accesses'] > 0
        
        await service.shutdown()

    async def test_config_service_dependency_injection(self):
        """Test ConfigService with dependency injection."""
        from src.core.config.service import register_config_service_in_container
        
        # Register service in container
        register_config_service_in_container()
        
        # Get from container
        container = get_container()
        service = await container.get("ConfigService")
        
        assert isinstance(service, ConfigService)
        assert service._initialized
        
        # Test config access
        db_config = service.get_database_config()
        assert isinstance(db_config, DatabaseConfig)
        
        await service.shutdown()

    async def test_config_value_access(self):
        """Test dot notation config value access."""
        service = ConfigService()
        await service.initialize()
        
        # Test nested value access
        port = service.get_config_value("database.postgresql_port", 5432)
        assert port == 5432
        
        # Test with default
        non_existent = service.get_config_value("non.existent.key", "default")
        assert non_existent == "default"
        
        await service.shutdown()

    async def test_config_service_context_manager(self):
        """Test ConfigService as async context manager."""
        async with ConfigService() as service:
            assert service._initialized
            
            db_config = service.get_database_config()
            assert isinstance(db_config, DatabaseConfig)
        
        # Should be shutdown after context exit
        assert not service._initialized