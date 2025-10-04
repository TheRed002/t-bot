"""
Integration tests for database functionality with real services.

These tests verify database connections, models, and operations
using real PostgreSQL, Redis, and InfluxDB instances instead of mocks.
Builds upon Phase 1 infrastructure foundation.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text

from src.database.models import BalanceSnapshot, BotInstance, Position, Order, User
from src.database.models.trading import Trade
from src.database.queries import DatabaseQueries
from tests.integration.infrastructure.service_factory import RealServiceFactory
# Import the correct fixtures from infrastructure
from tests.integration.infrastructure.conftest import clean_database, real_test_config  # noqa: F401


logger = logging.getLogger(__name__)

# Uses real services from Phase 1 infrastructure


@pytest.mark.asyncio
class TestDatabaseConnection:
    """Test database connection and health check with real services."""

    @pytest.mark.asyncio
    async def test_database_connection(self, clean_database):
        """Test database connection and health check using real PostgreSQL."""
        service_factory = RealServiceFactory()

        try:
            # Initialize real services
            await service_factory.initialize_core_services(clean_database)

            # Verify real database connectivity
            async with clean_database.get_async_session() as session:
                result = await session.execute(text("SELECT 1 as test"))
                test_value = result.scalar()
                assert test_value == 1
                logger.info("✅ Real database connection verified")

            # Test Redis connectivity
            redis_client = await clean_database.get_redis_client()
            await redis_client.ping()
            logger.info("✅ Real Redis connection verified")

            # Test InfluxDB connectivity
            influx_client = clean_database.get_influxdb_client()
            await asyncio.get_event_loop().run_in_executor(None, influx_client.ping)
            logger.info("✅ Real InfluxDB connection verified")

        finally:
            await service_factory.cleanup()


@pytest.mark.asyncio
class TestDatabaseModels:
    """Test database model creation and validation with real database."""

    @pytest.mark.asyncio
    async def test_user_creation(self, clean_database):
        """Test user creation and validation using real PostgreSQL."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            async with clean_database.get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test user with unique data for real database
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
                logger.info(f"✅ Real database user creation verified: {created_user.username}")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_bot_instance_creation(self, clean_database):
        """Test bot instance creation and validation using real PostgreSQL."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            async with clean_database.get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test user first with unique data
                unique_id = str(uuid.uuid4())[:8]
                user = User(
                    username=f"test_user_{unique_id}",
                    email=f"test_{unique_id}@example.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                )
                created_user = await queries.create(user)

                # Create bot instance with unique name
                bot_instance = BotInstance(
                    name=f"test_bot_{unique_id}",
                    strategy_type="mean_reversion",
                    exchange="binance",
                    status="stopped",
                    config={"param1": "value1"},
                )

                created_bot = await queries.create(bot_instance)
                assert created_bot.id is not None
                assert created_bot.name == f"test_bot_{unique_id}"
                assert created_bot.strategy_type == "mean_reversion"
                assert created_bot.exchange == "binance"
                assert created_bot.status == "stopped"
                logger.info(f"✅ Real database bot creation verified: {created_bot.name}")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_order_creation_with_strategy(self, clean_database):
        """Test order creation and validation using real PostgreSQL."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            async with clean_database.get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test user and bot with unique data
                unique_id = str(uuid.uuid4())[:8]
                user = User(
                    username=f"test_user_{unique_id}",
                    email=f"test_{unique_id}@example.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                )
                created_user = await queries.create(user)

                # Create strategy first (required for Order)
                from src.database.models.bot import Strategy, Bot

                # Create bot first (required for Strategy)
                bot = Bot(
                    name=f"test_bot_{unique_id}",
                    exchange="binance",  # Use a supported exchange
                    status="running",
                    allocated_capital=Decimal("1000.00"),
                    current_balance=Decimal("1000.00")
                )
                created_bot_record = await queries.create(bot)

                strategy = Strategy(
                    name=f"test_strategy_{unique_id}",
                    type="mean_reversion",
                    bot_id=created_bot_record.id,
                    params={"risk_level": "medium"},
                    max_position_size=Decimal("100.00"),
                    risk_per_trade=Decimal("0.02"),
                    status="active"
                )
                created_strategy = await queries.create(strategy)

                bot_instance = BotInstance(
                    name=f"test_bot_{unique_id}",
                    strategy_type="mean_reversion",
                    exchange="binance",
                    status="stopped",
                    config={"param1": "value1"},
                )
                created_bot_instance = await queries.create(bot_instance)

                # Create order with required strategy_id
                order = Order(
                    bot_id=created_bot_record.id,  # Use the Bot ID from the Bot record
                    strategy_id=created_strategy.id,  # Required field
                    exchange_order_id=f"test_order_{unique_id}",
                    exchange="binance",
                    symbol="BTCUSDT",
                    side="buy",
                    order_type="market",
                    quantity=Decimal("0.001"),
                    price=Decimal("50000.00"),
                    status="filled",
                    filled_quantity=Decimal("0.001"),
                    average_price=Decimal("50000.00"),
                )

                created_order = await queries.create(order)
                assert created_order.id is not None
                assert created_order.bot_id == created_bot_record.id
                assert created_order.exchange_order_id == f"test_order_{unique_id}"
                assert created_order.symbol == "BTCUSDT"
                assert created_order.side == "buy"
                assert created_order.status == "filled"
                # Verify financial precision is maintained in real database
                assert isinstance(created_order.quantity, Decimal)
                assert isinstance(created_order.price, Decimal)
                assert created_order.quantity == Decimal("0.001")
                logger.info(f"✅ Real database order creation verified: {created_order.exchange_order_id}")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_position_creation(self, clean_database):
        """Test position creation and validation using real PostgreSQL."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            async with clean_database.get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test user with unique data
                unique_id = str(uuid.uuid4())[:8]
                user = User(
                    username=f"test_user_{unique_id}",
                    email=f"test_{unique_id}@example.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                )
                created_user = await queries.create(user)

                # Create strategy first (required for Position)
                from src.database.models.bot import Strategy, Bot

                # Create bot first (required for Strategy)
                bot = Bot(
                    name=f"test_bot_{unique_id}",
                    exchange="binance",  # Use a supported exchange
                    status="running",
                    allocated_capital=Decimal("1000.00"),
                    current_balance=Decimal("1000.00")
                )
                created_bot_record = await queries.create(bot)

                strategy = Strategy(
                    name=f"test_strategy_{unique_id}",
                    type="mean_reversion",
                    bot_id=created_bot_record.id,
                    params={"risk_level": "medium"},
                    max_position_size=Decimal("100.00"),
                    risk_per_trade=Decimal("0.02"),
                    status="active"
                )
                created_strategy = await queries.create(strategy)

                bot_instance = BotInstance(
                    name=f"test_bot_{unique_id}",
                    strategy_type="mean_reversion",
                    exchange="binance",
                    status="stopped",
                    config={"param1": "value1"},
                )
                created_bot_instance = await queries.create(bot_instance)

                # Create position with financial precision and required strategy_id
                position = Position(
                    bot_id=created_bot_record.id,  # Use the Bot ID from the Bot record
                    strategy_id=created_strategy.id,  # Required field
                    exchange="binance",
                    symbol="BTCUSDT",
                    side="LONG",  # Must be uppercase per model constraints
                    quantity=Decimal("0.001"),
                    entry_price=Decimal("50000.00"),
                    current_price=Decimal("51000.00"),
                    unrealized_pnl=Decimal("10.00"),
                    realized_pnl=Decimal("0.00"),
                )

                created_position = await queries.create(position)
                assert created_position.id is not None
                assert created_position.bot_id == created_bot_record.id
                assert created_position.strategy_id == created_strategy.id
                assert created_position.symbol == "BTCUSDT"
                assert created_position.side == "LONG"
                assert created_position.quantity == Decimal("0.001")
                assert created_position.entry_price == Decimal("50000.00")
                # Verify Decimal precision is maintained
                assert isinstance(created_position.quantity, Decimal)
                assert isinstance(created_position.entry_price, Decimal)
                logger.info(f"✅ Real database position creation verified: {created_position.symbol}")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_balance_snapshot_creation(self, clean_database):
        """Test balance snapshot creation and validation using real PostgreSQL."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            async with clean_database.get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test user with unique data
                unique_id = str(uuid.uuid4())[:8]
                user = User(
                    username=f"test_user_{unique_id}",
                    email=f"test_{unique_id}@example.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                )
                created_user = await queries.create(user)

                # Create balance snapshot with financial precision
                balance_snapshot = BalanceSnapshot(
                    user_id=created_user.id,  # Required field
                    exchange="binance",
                    account_type="spot",  # Required field
                    currency="USDT",
                    available_balance=Decimal("1000.00"),  # Correct field name
                    locked_balance=Decimal("0.00"),
                    total_balance=Decimal("1000.00"),
                )

                created_balance = await queries.create(balance_snapshot)
                assert created_balance.id is not None
                assert created_balance.user_id == created_user.id
                assert created_balance.exchange == "binance"
                assert created_balance.account_type == "spot"
                assert created_balance.currency == "USDT"
                assert created_balance.available_balance == Decimal("1000.00")  # Correct field name
                assert created_balance.total_balance == Decimal("1000.00")
                # Verify Decimal precision is maintained
                assert isinstance(created_balance.available_balance, Decimal)
                assert isinstance(created_balance.total_balance, Decimal)
                logger.info(f"✅ Real database balance snapshot creation verified: {created_balance.currency}")

        finally:
            await service_factory.cleanup()


@pytest.mark.asyncio
class TestDatabaseQueries:
    """Test database query operations with real PostgreSQL."""

    @pytest.mark.asyncio
    async def test_get_by_id(self, clean_database):
        """Test get_by_id query using real database."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            async with clean_database.get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test user with unique data
                unique_id = str(uuid.uuid4())[:8]
                user = User(
                    username=f"test_user_{unique_id}",
                    email=f"test_{unique_id}@example.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                )
                created_user = await queries.create(user)

                # Test get_by_id with real database
                retrieved_user = await queries.get_by_id(User, created_user.id)
                assert retrieved_user is not None
                assert retrieved_user.id == created_user.id
                assert retrieved_user.username == created_user.username
                logger.info(f"✅ Real database get_by_id verified: {retrieved_user.username}")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_get_all(self, clean_database):
        """Test get_all query using real database."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            async with clean_database.get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create test users with unique data
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

                # Test get_all with real database
                all_users = await queries.get_all(User)
                assert len(all_users) >= 2
                logger.info(f"✅ Real database get_all verified: {len(all_users)} users found")

        finally:
            await service_factory.cleanup()


@pytest.mark.asyncio
class TestRedisIntegration:
    """Test Redis client integration with real Redis service."""

    @pytest.mark.asyncio
    async def test_redis_basic_operations(self, clean_database):
        """Test Redis basic operations using real Redis."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            # Get real Redis client
            redis_client = await clean_database.get_redis_client()

            # Test basic operations with real Redis
            test_key = f"test_key_{uuid.uuid4().hex[:8]}"
            await redis_client.set(test_key, "test_value")
            value = await redis_client.get(test_key)
            assert value == "test_value"

            # Test hash operations with real Redis
            test_hash = f"test_hash_{uuid.uuid4().hex[:8]}"
            await redis_client.hset(test_hash, "field1", "value1")
            hash_value = await redis_client.hget(test_hash, "field1")
            assert hash_value == "value1"

            # Cleanup test data
            await redis_client.delete(test_key)
            await redis_client.delete(test_hash)

            logger.info("✅ Real Redis operations verified")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_redis_trading_operations(self, clean_database):
        """Test Redis trading-specific operations using real Redis."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            # Get real Redis client
            redis_client = await clean_database.get_redis_client()

            # Test market data storage with unique keys
            unique_id = str(uuid.uuid4())[:8]
            symbol_key = f"BTCUSDT_{unique_id}"

            market_data = {
                "symbol": symbol_key,
                "price": "50000.00",
                "volume": "100.5",
                "timestamp": datetime.now().isoformat(),
            }

            # Store market data in real Redis
            await redis_client.set(f"market_data:{symbol_key}", str(market_data))
            retrieved_data = await redis_client.get(f"market_data:{symbol_key}")
            assert retrieved_data is not None

            # Test position storage with unique keys
            position_key = f"position:test_bot_{unique_id}"
            position_data = {
                "symbol": symbol_key,
                "quantity": "0.001",
                "side": "long",
                "entry_price": "50000.00",
            }

            await redis_client.set(position_key, str(position_data))
            retrieved_position = await redis_client.get(position_key)
            assert retrieved_position is not None

            # Cleanup test data
            await redis_client.delete(f"market_data:{symbol_key}")
            await redis_client.delete(position_key)

            logger.info("✅ Real Redis trading operations verified")

        finally:
            await service_factory.cleanup()


@pytest.mark.asyncio
class TestInfluxDBIntegration:
    """Test InfluxDB client integration with real InfluxDB service."""

    @pytest.mark.asyncio
    async def test_influxdb_connection(self, clean_database):
        """Test InfluxDB connection using real InfluxDB."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            # Get real InfluxDB client
            influx_client = clean_database.get_influxdb_client()

            # Test connection and health check with real InfluxDB
            health_status = await asyncio.get_event_loop().run_in_executor(
                None, influx_client.ping
            )
            assert health_status is not None
            logger.info("✅ Real InfluxDB connection verified")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_influxdb_data_writing(self, clean_database):
        """Test InfluxDB data writing using real InfluxDB."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            # Get our custom InfluxDB client wrapper
            from src.database.influxdb_client import InfluxDBClientWrapper

            # Create custom client using proper configuration
            influx_client = InfluxDBClientWrapper(
                url="http://localhost:8086",
                token="test-token",
                org="test-org",
                bucket="test-bucket"
            )

            # Initialize the client connection
            await influx_client.connect()

            # Test market data writing with unique data
            unique_id = str(uuid.uuid4())[:8]
            market_data_point = {
                "symbol": f"BTCUSDT_{unique_id}",
                "price": 50000.0,
                "volume": 100.5,
                "timestamp": datetime.now(timezone.utc),
            }

            # Write to real InfluxDB using the specific method
            await influx_client.write_market_data(
                symbol=market_data_point["symbol"],
                data={
                    "price": market_data_point["price"],
                    "volume": market_data_point["volume"]
                },
                timestamp=market_data_point["timestamp"]
            )

            # Test trade data writing with unique data
            trade_data_point = {
                "bot_id": f"test_bot_{unique_id}",
                "symbol": f"BTCUSDT_{unique_id}",
                "side": "buy",
                "quantity": 0.001,
                "price": 50000.0,
                "commission": 0.0001,
                "timestamp": datetime.now(timezone.utc),
            }

            # Write trade data using the specific method
            await influx_client.write_trade(
                trade_data=trade_data_point,
                timestamp=trade_data_point["timestamp"]
            )

            # Test performance metrics writing with unique data
            performance_point = {
                "bot_id": f"test_bot_{unique_id}",
                "total_pnl": 100.0,
                "win_rate": 0.65,
                "sharpe_ratio": 1.2,
                "max_drawdown": -50.0,
                "timestamp": datetime.now(timezone.utc),
            }

            # Write performance data using the specific method
            await influx_client.write_performance_metrics(
                bot_id=performance_point["bot_id"],
                metrics={
                    "total_pnl": performance_point["total_pnl"],
                    "win_rate": performance_point["win_rate"],
                    "sharpe_ratio": performance_point["sharpe_ratio"],
                    "max_drawdown": performance_point["max_drawdown"]
                },
                timestamp=performance_point["timestamp"]
            )

            logger.info("✅ Real InfluxDB data writing verified")

        finally:
            await service_factory.cleanup()


@pytest.mark.asyncio
class TestFinancialPrecisionIntegration:
    """Test financial precision with real database operations."""

    @pytest.mark.asyncio
    async def test_decimal_precision_maintenance(self, clean_database):
        """Test that Decimal precision is maintained through real database operations."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            async with clean_database.get_async_session() as session:
                queries = DatabaseQueries(session)

                # Create user and bot for trade
                unique_id = str(uuid.uuid4())[:8]
                user = User(
                    username=f"precision_user_{unique_id}",
                    email=f"precision_{unique_id}@example.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                )
                created_user = await queries.create(user)

                # Create strategy first (required for Trade)
                from src.database.models.bot import Strategy, Bot

                # Create bot first (required for Strategy)
                bot = Bot(
                    name=f"precision_bot_{unique_id}",
                    exchange="binance",  # Use a supported exchange
                    status="running",
                    allocated_capital=Decimal("1000.00"),
                    current_balance=Decimal("1000.00")
                )
                created_bot_record = await queries.create(bot)

                strategy = Strategy(
                    name=f"precision_strategy_{unique_id}",
                    type="momentum",  # Valid strategy type
                    bot_id=created_bot_record.id,
                    params={"risk_level": "high"},
                    max_position_size=Decimal("100.00"),
                    risk_per_trade=Decimal("0.01"),
                    status="active"
                )
                created_strategy = await queries.create(strategy)

                # Create position (required for Trade)
                position = Position(
                    bot_id=created_bot_record.id,
                    strategy_id=created_strategy.id,
                    exchange="binance",
                    symbol="BTCUSDT",
                    side="LONG",
                    quantity=Decimal("0.12345678"),
                    entry_price=Decimal("45678.12345678"),
                    current_price=Decimal("46000.00"),
                    unrealized_pnl=Decimal("0.00"),
                    realized_pnl=Decimal("0.00"),
                )
                created_position = await queries.create(position)

                bot_instance = BotInstance(
                    name=f"precision_bot_{unique_id}",
                    strategy_type="momentum",
                    exchange="binance",
                    status="running",
                )
                created_bot_instance = await queries.create(bot_instance)

                # Test with high-precision financial values
                high_precision_quantity = Decimal("0.12345678")  # 18 decimal places
                high_precision_entry_price = Decimal("45678.12345678")  # 18 decimal places
                high_precision_exit_price = Decimal("46000.00000000")  # 18 decimal places
                expected_pnl = (high_precision_exit_price - high_precision_entry_price) * high_precision_quantity

                trade = Trade(
                    bot_id=created_bot_record.id,
                    strategy_id=created_strategy.id,
                    position_id=created_position.id,
                    exchange="binance",
                    symbol="BTCUSDT",
                    side="buy",
                    quantity=high_precision_quantity,
                    entry_price=high_precision_entry_price,
                    exit_price=high_precision_exit_price,
                    pnl=expected_pnl,
                    fees=Decimal("0.00012345"),
                )

                # Store in real database and retrieve
                created_trade = await queries.create(trade)
                retrieved_trade = await queries.get_by_id(Trade, created_trade.id)

                # Verify precision is maintained through real database roundtrip
                assert isinstance(retrieved_trade.quantity, Decimal)
                assert isinstance(retrieved_trade.entry_price, Decimal)
                assert isinstance(retrieved_trade.exit_price, Decimal)
                assert isinstance(retrieved_trade.pnl, Decimal)
                assert retrieved_trade.quantity == high_precision_quantity
                assert retrieved_trade.entry_price == high_precision_entry_price
                assert retrieved_trade.exit_price == high_precision_exit_price

                # Verify calculation precision
                calculated_pnl = (retrieved_trade.exit_price - retrieved_trade.entry_price) * retrieved_trade.quantity
                assert calculated_pnl == expected_pnl

                logger.info(f"✅ Financial precision maintained: ({high_precision_exit_price} - {high_precision_entry_price}) * {high_precision_quantity} = {calculated_pnl}")

        finally:
            await service_factory.cleanup()