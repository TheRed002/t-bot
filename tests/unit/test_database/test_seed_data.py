"""
Unit tests for database seed data functionality.

Tests DatabaseSeeder class and all seeding functions to ensure proper
database initialization for development and testing environments.
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.core.config import Config
from src.core.types import StrategyStatus, StrategyType, OrderSide, OrderType, OrderStatus
from src.database.seed_data import DatabaseSeeder, run_seed, main
from src.database.models import User, BotInstance, Strategy, Trade


class TestDatabaseSeeder:
    """Test cases for DatabaseSeeder class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.debug = True
        config.environment = "development"
        config.jwt_secret = "test_secret_key"
        config.jwt_algorithm = "HS256"
        config.jwt_access_token_expire_minutes = 30
        return config

    @pytest.fixture
    def mock_production_config(self):
        """Create mock production configuration."""
        config = Mock(spec=Config)
        config.debug = False
        config.environment = "production"
        return config

    @pytest.fixture
    def mock_jwt_handler(self):
        """Create mock JWT handler."""
        handler = Mock()
        handler.hash_password.return_value = "hashed_password"
        return handler

    @pytest.fixture
    def database_seeder(self, mock_config):
        """Create DatabaseSeeder instance for testing."""
        with patch('src.database.seed_data.JWTHandler') as mock_jwt:
            mock_jwt.return_value.hash_password.return_value = "hashed_password"
            return DatabaseSeeder(mock_config)

    def test_database_seeder_init_success(self, mock_config):
        """Test DatabaseSeeder successful initialization."""
        with patch('src.database.seed_data.JWTHandler'):
            seeder = DatabaseSeeder(mock_config)
            assert seeder.config == mock_config
            assert seeder.seed_data == {}

    def test_database_seeder_init_production_error(self, mock_production_config):
        """Test DatabaseSeeder fails in production."""
        with pytest.raises(ValueError, match="Database seeding is only allowed in development mode"):
            DatabaseSeeder(mock_production_config)

    def test_database_seeder_init_debug_false_error(self):
        """Test DatabaseSeeder fails when debug is False."""
        config = Mock(spec=Config)
        config.debug = False
        config.environment = "development"
        
        with pytest.raises(ValueError, match="Database seeding is only allowed in development mode"):
            DatabaseSeeder(config)

    def test_load_seed_data(self, database_seeder):
        """Test loading seed data configuration."""
        seed_data = database_seeder._load_seed_data()
        
        # Verify structure exists
        assert 'users' in seed_data
        assert 'bot_instances' in seed_data
        assert 'strategies' in seed_data
        assert 'exchange_credentials' in seed_data
        
        # Verify data types
        assert isinstance(seed_data['users'], list)
        assert isinstance(seed_data['bot_instances'], list)
        assert isinstance(seed_data['strategies'], list)
        assert isinstance(seed_data['exchange_credentials'], list)
        
        # Verify user data structure
        assert len(seed_data['users']) > 0
        user = seed_data['users'][0]
        assert 'username' in user
        assert 'email' in user
        assert 'password' in user
        assert 'is_active' in user
        assert 'is_verified' in user

    @pytest.mark.asyncio
    async def test_seed_users_success(self, database_seeder):
        """Test successful user seeding."""
        mock_session = AsyncMock()
        # Mock synchronous methods to return None instead of coroutine
        mock_session.add = Mock()
        
        # Mock database result - no existing users
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        users = await database_seeder.seed_users(mock_session)
        
        assert len(users) > 0
        assert mock_session.add.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_seed_users_existing_user_skip(self, database_seeder):
        """Test user seeding skips existing users."""
        mock_session = AsyncMock()
        # Mock synchronous methods to return None instead of coroutine
        mock_session.add = Mock()
        
        # Mock existing user
        existing_user = Mock()
        existing_user.username = "admin"
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = existing_user
        mock_session.execute.return_value = mock_result
        
        users = await database_seeder.seed_users(mock_session)
        
        # Should return existing user without creating new one
        assert len(users) > 0
        assert users[0] == existing_user

    @pytest.mark.asyncio
    async def test_seed_bot_instances_success(self, database_seeder):
        """Test successful bot instance seeding."""
        mock_session = AsyncMock()
        # Mock synchronous methods to return None instead of coroutine
        mock_session.add = Mock()
        
        # Create mock users
        mock_users = [Mock() for _ in range(2)]
        for i, user in enumerate(mock_users):
            user.id = f"user_{i}"
            user.username = f"user_{i}"
        
        # Mock database result - no existing bots
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        bots = await database_seeder.seed_bot_instances(mock_session, mock_users)
        
        assert len(bots) > 0
        assert mock_session.add.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_seed_strategies_success(self, database_seeder):
        """Test successful strategy seeding."""
        mock_session = AsyncMock()
        # Mock synchronous methods to return None instead of coroutine
        mock_session.add = Mock()
        
        # Create mock bots
        mock_bots = [Mock() for _ in range(2)]
        for i, bot in enumerate(mock_bots):
            bot.id = f"bot_{i}"
            bot.name = f"bot_{i}"
        
        # Mock database result - no existing strategies
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        strategies = await database_seeder.seed_strategies(mock_session, mock_bots)
        
        assert len(strategies) > 0
        assert mock_session.add.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_seed_exchange_credentials_success(self, database_seeder):
        """Test successful exchange credential seeding."""
        mock_session = AsyncMock()
        
        # Create mock users
        mock_users = [Mock() for _ in range(3)]
        for i, user in enumerate(mock_users):
            user.username = ["admin", "trader1", "trader2"][i]
        
        credentials = await database_seeder.seed_exchange_credentials(mock_session, mock_users)
        
        assert len(credentials) > 0
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_seed_sample_trades_success(self, database_seeder):
        """Test successful sample trade seeding."""
        mock_session = AsyncMock()
        # Mock synchronous methods to return None instead of coroutine
        mock_session.add = Mock()
        
        # Create mock active bots
        mock_bots = [Mock() for _ in range(2)]
        for i, bot in enumerate(mock_bots):
            bot.id = uuid.uuid4()
            bot.name = f"bot_{i}"
            bot.is_active = True
            bot.exchange = "binance"
            bot.config = {"trading_pair": "BTC/USDT"}
        
        # Mock database result - no existing trades
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        await database_seeder.seed_sample_trades(mock_session, mock_bots)
        
        assert mock_session.add.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_seed_all_success(self, database_seeder):
        """Test successful complete seeding."""
        with patch('src.database.seed_data.get_db_session') as mock_get_session:
            mock_session = AsyncMock()
            # Mock synchronous methods to return None instead of coroutine
            mock_session.add = Mock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Mock all database results
            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = mock_result
            
            await database_seeder.seed_all()
            
            assert mock_session.add.called
            assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_seed_all_non_development_skip(self, mock_config):
        """Test seed_all skips in non-development environment."""
        mock_config.environment = "production"
        
        with patch('src.database.seed_data.JWTHandler'):
            seeder = DatabaseSeeder.__new__(DatabaseSeeder)
            seeder.config = mock_config
            
            # Should not raise an error, just skip seeding
            await seeder.seed_all()

    @pytest.mark.asyncio
    async def test_seed_all_database_error(self, database_seeder):
        """Test seed_all handles database errors."""
        with patch('src.database.seed_data.get_db_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("Database error")
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Database error"):
                await database_seeder.seed_all()
            
            assert mock_session.rollback.called


class TestSeedDataFunctions:
    """Test standalone seeding functions."""

    @pytest.mark.asyncio
    async def test_run_seed_with_config(self):
        """Test run_seed function with provided config."""
        mock_config = Mock(spec=Config)
        mock_config.debug = True
        mock_config.environment = "development"
        
        with patch('src.database.seed_data.DatabaseSeeder') as mock_seeder_class:
            mock_seeder = Mock()
            mock_seeder.seed_all = AsyncMock()
            mock_seeder_class.return_value = mock_seeder
            
            await run_seed(mock_config)
            
            mock_seeder_class.assert_called_once_with(mock_config)
            mock_seeder.seed_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_seed_without_config(self):
        """Test run_seed function without config (creates default)."""
        with patch('src.database.seed_data.Config') as mock_config_class, \
             patch('src.database.seed_data.DatabaseSeeder') as mock_seeder_class:
            
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_seeder = Mock()
            mock_seeder.seed_all = AsyncMock()
            mock_seeder_class.return_value = mock_seeder
            
            await run_seed()
            
            mock_config_class.assert_called_once()
            mock_seeder_class.assert_called_once_with(mock_config)

    def test_main_function(self):
        """Test main function."""
        with patch('src.database.seed_data.Config') as mock_config_class, \
             patch('src.database.seed_data.asyncio.run') as mock_run:
            
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            # Configure mock_run to properly handle the coroutine argument
            def mock_run_handler(coro):
                if hasattr(coro, 'close'):
                    coro.close()  # Close the coroutine to avoid unawaited warning
                
            mock_run.side_effect = mock_run_handler
            
            main()
            
            mock_config_class.assert_called_once()
            assert mock_run.called


class TestSeedDataEdgeCases:
    """Test edge cases and error handling."""

    def test_database_seeder_init_jwt_error(self):
        """Test DatabaseSeeder initialization handles JWT handler errors."""
        config = Mock(spec=Config)
        config.debug = True
        config.environment = "development"
        
        with patch('src.database.seed_data.JWTHandler') as mock_jwt:
            mock_jwt.side_effect = Exception("JWT initialization error")
            
            with pytest.raises(Exception, match="JWT initialization error"):
                DatabaseSeeder(config)

    @pytest.mark.asyncio
    async def test_seed_users_database_commit_error(self):
        """Test seed_users handles database commit errors."""
        config = Mock(spec=Config)
        config.debug = True
        config.environment = "development"
        
        with patch('src.database.seed_data.JWTHandler'):
            seeder = DatabaseSeeder(config)
            
            mock_session = AsyncMock()
            # Mock synchronous methods to return None instead of coroutine
            mock_session.add = Mock()
            mock_session.commit.side_effect = Exception("Commit error")
            
            # Mock database result - no existing users
            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = mock_result
            
            with pytest.raises(Exception, match="Commit error"):
                await seeder.seed_users(mock_session)

    def test_seed_with_invalid_data_types(self):
        """Test seeder handles invalid data types in seed data."""
        config = Mock(spec=Config)
        config.debug = True
        config.environment = "development"
        
        with patch('src.database.seed_data.JWTHandler'):
            seeder = DatabaseSeeder(config)
            seed_data = seeder._load_seed_data()
            
            # Verify data types are correct
            assert isinstance(seed_data['users'], list)
            assert all(isinstance(user, dict) for user in seed_data['users'])

    def test_load_seed_data_structure_validation(self):
        """Test seed data structure validation."""
        config = Mock(spec=Config)
        config.debug = True
        config.environment = "development"
        
        with patch('src.database.seed_data.JWTHandler'):
            seeder = DatabaseSeeder(config)
            seed_data = seeder._load_seed_data()
            
            # Validate required keys exist
            required_keys = ['users', 'bot_instances', 'strategies', 'exchange_credentials']
            for key in required_keys:
                assert key in seed_data
                assert isinstance(seed_data[key], list)

    def test_financial_precision_in_seed_data(self):
        """Test that financial values maintain precision."""
        config = Mock(spec=Config)
        config.debug = True
        config.environment = "development"
        
        with patch('src.database.seed_data.JWTHandler'):
            seeder = DatabaseSeeder(config)
            seed_data = seeder._load_seed_data()
            
            # Check bot instances have proper financial values
            for bot_data in seed_data['bot_instances']:
                assert 'initial_balance' in bot_data
                assert 'current_balance' in bot_data
                assert isinstance(bot_data['initial_balance'], Decimal)
                assert isinstance(bot_data['current_balance'], Decimal)