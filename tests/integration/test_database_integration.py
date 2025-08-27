"""
Integration tests for database functionality.

These tests verify database connections, models, and operations
with actual database instances.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.database.connection import (
    close_database,
    get_async_session,
    health_check,
    initialize_database,
)
from src.database.influxdb_client import InfluxDBClientWrapper
from src.database.models import BalanceSnapshot, BotInstance, Position, Trade, User
from src.database.queries import DatabaseQueries
from src.database.redis_client import RedisClient

# Database setup is handled by conftest.py fixtures


@pytest.mark.asyncio
class TestDatabaseConnection:
    """Test database connection and health check."""

    async def test_database_connection(self, config, database_setup):
        """Test database connection and health check."""

        try:
            await initialize_database(database_setup)

            # The database_setup fixture should have already initialized the database
            # and run a health check. Let's verify it's working.
            health_status = await health_check()
            assert health_status is not None
            # Check if any database is available
            if isinstance(health_status, dict):
                # New format with individual database status
                assert any(health_status.values()), "No databases are available"
            else:
                # Old format with overall status
                assert "status" in health_status
                assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        finally:
            await close_database()


@pytest.mark.asyncio
class TestDatabaseModels:
    """Test database model creation and validation."""

    async def test_user_creation(self, config, database_setup):
        """Test user creation and validation."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test user
                unique_id = str(uuid.uuid4())[:8]
                user = User(
                    username=f"test_user_{unique_id}",
                    email=f"test_{unique_id}@example.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                )

                created_user = await queries.create(user)
                assert created_user.id is not None
                assert created_user.username.startswith("test_user_")
                assert created_user.email.startswith("test_")
                assert created_user.is_active is True
                assert created_user.is_verified is True
        finally:
            await close_database()

    async def test_bot_instance_creation(self, config, database_setup):
        """Test bot instance creation and validation."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test user first
                unique_id = str(uuid.uuid4())[:8]
                user = User(
                    username=f"test_user_{unique_id}",
                    email=f"test_{unique_id}@example.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                )
                created_user = await queries.create(user)

                # Create bot instance
                bot_instance = BotInstance(
                    name="test_bot",
                    user_id=created_user.id,
                    strategy_type="mean_reversion",
                    exchange="binance",
                    status="stopped",
                    config={"param1": "value1"},
                )

                created_bot = await queries.create(bot_instance)
                assert created_bot.id is not None
                assert created_bot.name == "test_bot"
                assert created_bot.user_id == created_user.id
                assert created_bot.strategy_type == "mean_reversion"
                assert created_bot.exchange == "binance"
                assert created_bot.status == "stopped"
        finally:
            await close_database()

    async def test_trade_creation(self, config, database_setup):
        """Test trade creation and validation."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

            # Create test user and bot
            unique_id = str(uuid.uuid4())[:8]
            user = User(
                username=f"test_user_{unique_id}",
                email=f"test_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )
            created_user = await queries.create(user)

            bot_instance = BotInstance(
                name="test_bot",
                user_id=created_user.id,
                strategy_type="mean_reversion",
                exchange="binance",
                status="stopped",
                config={"param1": "value1"},
            )
            created_bot = await queries.create(bot_instance)

            # Create trade
            trade = Trade(
                bot_id=created_bot.id,
                exchange_order_id="test_order_123",
                exchange="binance",
                symbol="BTCUSDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.001"),
                price=Decimal("50000.00"),
                executed_price=Decimal("50000.00"),
                status="filled",
                fee=Decimal("0.0001"),
                fee_currency="USDT",
            )

            created_trade = await queries.create(trade)
            assert created_trade.id is not None
            assert created_trade.bot_id == created_bot.id
            assert created_trade.exchange_order_id == "test_order_123"
            assert created_trade.symbol == "BTCUSDT"
            assert created_trade.side == "buy"
            assert created_trade.status == "filled"
        finally:
            await close_database()

    async def test_position_creation(self, config, database_setup):
        """Test position creation and validation."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

            # Create test user and bot
            unique_id = str(uuid.uuid4())[:8]
            user = User(
                username=f"test_user_{unique_id}",
                email=f"test_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )
            created_user = await queries.create(user)

            bot_instance = BotInstance(
                name="test_bot",
                user_id=created_user.id,
                strategy_type="mean_reversion",
                exchange="binance",
                status="stopped",
                config={"param1": "value1"},
            )
            created_bot = await queries.create(bot_instance)

            # Create position
            position = Position(
                bot_id=created_bot.id,
                exchange="binance",
                symbol="BTCUSDT",
                side="long",
                quantity=Decimal("0.001"),
                entry_price=Decimal("50000.00"),
                current_price=Decimal("51000.00"),
                unrealized_pnl=Decimal("10.00"),
                realized_pnl=Decimal("0.00"),
            )

            created_position = await queries.create(position)
            assert created_position.id is not None
            assert created_position.bot_id == created_bot.id
            assert created_position.symbol == "BTCUSDT"
            assert created_position.side == "long"
            assert created_position.quantity == Decimal("0.001")
            assert created_position.entry_price == Decimal("50000.00")
        finally:
            await close_database()

    async def test_balance_snapshot_creation(self, config, database_setup):
        """Test balance snapshot creation and validation."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

            # Create test user
            unique_id = str(uuid.uuid4())[:8]
            user = User(
                username=f"test_user_{unique_id}",
                email=f"test_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )
            created_user = await queries.create(user)

            # Create balance snapshot
            balance_snapshot = BalanceSnapshot(
                user_id=created_user.id,
                exchange="binance",
                currency="USDT",
                free_balance=Decimal("1000.00"),
                locked_balance=Decimal("0.00"),
                total_balance=Decimal("1000.00"),
            )

            created_balance = await queries.create(balance_snapshot)
            assert created_balance.id is not None
            assert created_balance.user_id == created_user.id
            assert created_balance.exchange == "binance"
            assert created_balance.currency == "USDT"
            assert created_balance.free_balance == Decimal("1000.00")
            assert created_balance.total_balance == Decimal("1000.00")
        finally:
            await close_database()


@pytest.mark.asyncio
class TestDatabaseQueries:
    """Test database query operations."""

    async def test_get_by_id(self, config, database_setup):
        """Test get_by_id query."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

            # Create test user
            unique_id = str(uuid.uuid4())[:8]
            user = User(
                username=f"test_user_{unique_id}",
                email=f"test_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )
            created_user = await queries.create(user)

            # Test get_by_id
            retrieved_user = await queries.get_by_id(User, created_user.id)
            assert retrieved_user is not None
            assert retrieved_user.id == created_user.id
            assert retrieved_user.username == created_user.username
        finally:
            await close_database()

    async def test_get_all(self, config, database_setup):
        """Test get_all query."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

            # Create test users
            unique_id = str(uuid.uuid4())[:8]
            user1 = User(
                username=f"test_user1_{unique_id}",
                email=f"test1_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )
            user2 = User(
                username=f"test_user2_{unique_id}",
                email=f"test2_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )

            await queries.create(user1)
            await queries.create(user2)

            # Test get_all
            all_users = await queries.get_all(User)
            assert len(all_users) >= 2
        finally:
            await close_database()

    async def test_get_trades_by_bot(self, config, database_setup):
        """Test get_trades_by_bot query."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

            # Create test user and bot
            unique_id = str(uuid.uuid4())[:8]
            user = User(
                username=f"test_user_{unique_id}",
                email=f"test_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )
            created_user = await queries.create(user)

            bot_instance = BotInstance(
                name="test_bot",
                user_id=created_user.id,
                strategy_type="mean_reversion",
                exchange="binance",
                status="stopped",
                config={"param1": "value1"},
            )
            created_bot = await queries.create(bot_instance)

            # Create trades
            trade1 = Trade(
                bot_id=created_bot.id,
                exchange_order_id="test_order_1",
                exchange="binance",
                symbol="BTCUSDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.001"),
                price=Decimal("50000.00"),
                executed_price=Decimal("50000.00"),
                status="filled",
                fee=Decimal("0.0001"),
                fee_currency="USDT",
            )

            trade2 = Trade(
                bot_id=created_bot.id,
                exchange_order_id="test_order_2",
                exchange="binance",
                symbol="BTCUSDT",
                side="sell",
                order_type="market",
                quantity=Decimal("0.001"),
                price=Decimal("51000.00"),
                executed_price=Decimal("51000.00"),
                status="filled",
                fee=Decimal("0.0001"),
                fee_currency="USDT",
            )

            await queries.create(trade1)
            await queries.create(trade2)

            # Test get_trades_by_bot
            trades = await queries.get_trades_by_bot(created_bot.id)
            assert len(trades) >= 2
            for trade in trades:
                assert trade.bot_id == created_bot.id
        finally:
            await close_database()

    async def test_get_positions_by_bot(self, config, database_setup):
        """Test get_positions_by_bot query."""

        try:
            await initialize_database(database_setup)

            async with get_async_session() as session:
                queries = DatabaseQueries(session)

            # Create test user and bot
            unique_id = str(uuid.uuid4())[:8]
            user = User(
                username=f"test_user_{unique_id}",
                email=f"test_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )
            created_user = await queries.create(user)

            bot_instance = BotInstance(
                name="test_bot",
                user_id=created_user.id,
                strategy_type="mean_reversion",
                exchange="binance",
                status="stopped",
                config={"param1": "value1"},
            )
            created_bot = await queries.create(bot_instance)

            # Create positions
            position1 = Position(
                bot_id=created_bot.id,
                exchange="binance",
                symbol="BTCUSDT",
                side="long",
                quantity=Decimal("0.001"),
                entry_price=Decimal("50000.00"),
                current_price=Decimal("51000.00"),
                unrealized_pnl=Decimal("10.00"),
                realized_pnl=Decimal("0.00"),
            )

            position2 = Position(
                bot_id=created_bot.id,
                exchange="binance",
                symbol="ETHUSDT",
                side="short",
                quantity=Decimal("0.01"),
                entry_price=Decimal("3000.00"),
                current_price=Decimal("2900.00"),
                unrealized_pnl=Decimal("10.00"),
                realized_pnl=Decimal("0.00"),
            )

            await queries.create(position1)
            await queries.create(position2)

            # Test get_positions_by_bot
            positions = await queries.get_positions_by_bot(created_bot.id)
            assert len(positions) >= 2
            for position in positions:
                assert position.bot_id == created_bot.id
        finally:
            await close_database()


@pytest.mark.asyncio
class TestRedisIntegration:
    """Test Redis client integration."""

    async def test_redis_basic_operations(self, config):
        """Test Redis basic operations."""
        redis_client = RedisClient(config)

        # Connect to Redis
        await redis_client.connect()

        # Test basic operations
        await redis_client.set("test_key", "test_value")
        value = await redis_client.get("test_key")
        assert value == "test_value"

        # Test hash operations
        await redis_client.hset("test_hash", "field1", "value1")
        hash_value = await redis_client.hget("test_hash", "field1")
        assert hash_value == "value1"

    async def test_redis_trading_operations(self, config):
        """Test Redis trading-specific operations."""
        redis_client = RedisClient(config)

        await redis_client.connect()

        # Test market data storage
        market_data = {
            "symbol": "BTCUSDT",
            "price": "50000.00",
            "volume": "100.5",
            "timestamp": datetime.now().isoformat(),
        }

        await redis_client.store_market_data("BTCUSDT", market_data)
        retrieved_data = await redis_client.get_market_data("BTCUSDT")
        assert retrieved_data is not None
        assert retrieved_data["symbol"] == "BTCUSDT"

        # Test position storage
        position_data = {
            "symbol": "BTCUSDT",
            "quantity": "0.001",
            "side": "long",
            "entry_price": "50000.00",
        }

        await redis_client.store_position("test_bot", position_data)
        retrieved_position = await redis_client.get_position("test_bot")
        assert retrieved_position is not None
        assert retrieved_position["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
class TestInfluxDBIntegration:
    """Test InfluxDB client integration."""

    async def test_influxdb_connection(self, config):
        """Test InfluxDB connection."""
        influx_client = InfluxDBClientWrapper(
            url=f"http://{config.database.influxdb_host}:{config.database.influxdb_port}",
            token=config.database.influxdb_token,
            org=config.database.influxdb_org,
            bucket=config.database.influxdb_bucket,
        )

        # Test connection
        influx_client.connect()

        # Test health check
        health_status = influx_client.health_check()
        assert health_status is not None

        influx_client.disconnect()

    async def test_influxdb_data_writing(self, config):
        """Test InfluxDB data writing."""
        influx_client = InfluxDBClientWrapper(
            url=f"http://{config.database.influxdb_host}:{config.database.influxdb_port}",
            token=config.database.influxdb_token,
            org=config.database.influxdb_org,
            bucket=config.database.influxdb_bucket,
        )

        influx_client.connect()

        # Test market data writing
        market_data_point = {
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "volume": 100.5,
            "timestamp": datetime.now(timezone.utc),
        }

        influx_client.write_market_data("BTCUSDT", market_data_point)

        # Test trade data writing
        trade_data_point = {
            "bot_id": "test_bot",
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.001,
            "price": 50000.0,
            "commission": 0.0001,
            "timestamp": datetime.now(timezone.utc),
        }

        influx_client.write_trade(trade_data_point)

        # Test performance metrics writing
        performance_point = {
            "bot_id": "test_bot",
            "total_pnl": 100.0,
            "win_rate": 0.65,
            "sharpe_ratio": 1.2,
            "max_drawdown": -50.0,
            "timestamp": datetime.now(timezone.utc),
        }

        influx_client.write_performance_metrics("test_bot", performance_point)

        influx_client.disconnect()


@pytest.mark.asyncio
class TestMigrationSystem:
    """Test database migration system."""

    async def test_migration_configuration(self):
        """Test migration configuration."""
        from alembic.config import Config as AlembicConfig

        # Test migration configuration
        alembic_cfg = AlembicConfig("alembic.ini")

        # Test current revision
        from alembic.script import ScriptDirectory

        script_dir = ScriptDirectory.from_config(alembic_cfg)
        current_revision = script_dir.get_current_head()

        assert current_revision is not None

        # Test migration history
        revisions = list(script_dir.walk_revisions())
        assert len(revisions) > 0

    async def test_migration_file_structure(self):
        """Test migration file structure."""
        migration_file = "src/database/migrations/versions/001_initial_schema.py"

        # Test migration file exists
        import os

        assert os.path.exists(migration_file)

        # Test migration content
        with open(migration_file) as f:
            content = f.read()
            assert "def upgrade()" in content
            assert "def downgrade()" in content
            assert "op.create_table" in content
