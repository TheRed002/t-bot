"""
Database Integration Tests - Real Services (Fixed)

Tests that work with real Docker services with proper setup.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select, text

from src.database.models import Bot, Order, Position, Strategy, User


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_database_connections_basic(clean_database):
    """Test basic database connections and table creation."""
    # Initialize real database service with clean database
    from src.database.service import DatabaseService

    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        # Verify tables exist in the test schema
        test_schema = getattr(clean_database, "_test_schema", "test_unknown")

        async with db_service.get_session() as session:
            # Check that tables exist
            result = await session.execute(
                text(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = '{test_schema}'
                ORDER BY table_name
            """)
            )
            tables = [row[0] for row in result.fetchall()]

            print(f"Tables in schema {test_schema}: {tables}")
            assert "users" in tables, f"Users table not found in {test_schema}. Available: {tables}"
            assert "bots" in tables, f"Bots table not found in {test_schema}. Available: {tables}"

            # Test basic insertion
            unique_id = uuid.uuid4().hex[:8]
            user = User(
                id=uuid.uuid4(),
                username=f"test_user_{unique_id}",
                email=f"test_{unique_id}@example.com",
                password_hash="hashed_password",
                created_at=datetime.now(timezone.utc),
            )
            session.add(user)
            await session.commit()

        # Verify the user was inserted in a new session
        async with db_service.get_session() as session:
            result = await session.execute(select(User).where(User.id == user.id))
            retrieved_user = result.scalar_one()
            assert retrieved_user.username == user.username

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_model_relationships(clean_database):
    """Test model relationships with real database."""
    # Initialize real database service with clean database
    from src.database.service import DatabaseService

    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        async with db_service.get_session() as session:
            # Create a user
            user = User(
                id=uuid.uuid4(),
                username=f"trader_{uuid.uuid4().hex[:8]}",
                email=f"trader_{uuid.uuid4().hex[:8]}@test.com",
                password_hash="hashed",
                created_at=datetime.now(timezone.utc),
            )
            session.add(user)
            await session.flush()

            # Create a bot
            bot = Bot(
                id=uuid.uuid4(),
                name="Test Bot",
                exchange="binance",
                status="running",
                allocated_capital=Decimal("10000.00"),
                current_balance=Decimal("10000.00"),
                created_at=datetime.now(timezone.utc),
            )
            session.add(bot)
            await session.flush()

            # Create a strategy
            strategy = Strategy(
                id=uuid.uuid4(),
                name="Test Strategy",
                type="momentum",
                bot_id=bot.id,
                status="active",
                params={"param": "value"},
                max_position_size=Decimal("1000.0"),
                risk_per_trade=Decimal("0.02"),
                created_at=datetime.now(timezone.utc),
            )
            session.add(strategy)
            await session.flush()

            # Create a position
            position = Position(
                id=uuid.uuid4(),
                bot_id=bot.id,
                strategy_id=strategy.id,
                exchange="binance",
                symbol="BTC/USDT",
                side="LONG",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                status="OPEN",
                created_at=datetime.now(timezone.utc),
            )
            session.add(position)
            await session.flush()

            # Create an order
            order = Order(
                id=uuid.uuid4(),
                bot_id=bot.id,
                strategy_id=strategy.id,
                position_id=position.id,
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                status="filled",
                filled_quantity=Decimal("1.0"),
                average_price=Decimal("50000.00"),
                created_at=datetime.now(timezone.utc),
            )
            session.add(order)

            await session.commit()

        # Test relationships in a new session
        async with db_service.get_session() as session:
            result = await session.execute(select(Position).where(Position.id == position.id))
            retrieved_position = result.scalar_one()
            assert retrieved_position.bot_id == bot.id
            assert retrieved_position.strategy_id == strategy.id

            result = await session.execute(select(Order).where(Order.position_id == position.id))
            retrieved_order = result.scalar_one()
            assert retrieved_order.bot_id == bot.id

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_transactions_and_rollback(clean_database):
    """Test database transactions with rollback."""
    # Initialize real database service with clean database
    from src.database.service import DatabaseService

    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        # Create required dependencies first
        bot_id = None
        strategy_id = None

        async with db_service.get_session() as session:
            # Create bot
            bot = Bot(
                id=uuid.uuid4(),
                name="Rollback Test Bot",
                exchange="binance",
                status="running",
                allocated_capital=Decimal("10000.00"),
                current_balance=Decimal("10000.00"),
                created_at=datetime.now(timezone.utc),
            )
            session.add(bot)
            await session.flush()
            bot_id = bot.id

            # Create strategy
            strategy = Strategy(
                id=uuid.uuid4(),
                name="Rollback Test Strategy",
                type="momentum",
                bot_id=bot.id,
                status="active",
                params={"test": "value"},
                max_position_size=Decimal("1000.0"),
                risk_per_trade=Decimal("0.02"),
                created_at=datetime.now(timezone.utc),
            )
            session.add(strategy)
            await session.flush()
            strategy_id = strategy.id

            await session.commit()

        # Test successful transaction
        async with db_service.get_session() as session:
            position1 = Position(
                id=uuid.uuid4(),
                bot_id=bot_id,
                strategy_id=strategy_id,
                exchange="binance",
                symbol="BTC/USDT",
                side="LONG",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                status="OPEN",
                created_at=datetime.now(timezone.utc),
            )
            session.add(position1)
            await session.commit()

        # Verify position exists
        async with db_service.get_session() as session:
            result = await session.execute(select(Position))
            positions = result.scalars().all()
            assert len(positions) == 1

        # Test rollback
        try:
            async with db_service.get_session() as session:
                position2 = Position(
                    id=uuid.uuid4(),
                    bot_id=bot_id,
                    strategy_id=strategy_id,
                    exchange="binance",
                    symbol="ETH/USDT",
                    side="SHORT",
                    quantity=Decimal("10.0"),
                    entry_price=Decimal("3000.00"),
                    status="OPEN",
                    created_at=datetime.now(timezone.utc),
                )
                session.add(position2)
                await session.flush()

                # Force an error
                raise ValueError("Intentional error")

        except ValueError:
            pass  # Expected

        # Verify rollback worked
        async with db_service.get_session() as session:
            result = await session.execute(select(Position))
            positions = result.scalars().all()
            assert len(positions) == 1  # Only first position
            assert positions[0].symbol == "BTC/USDT"

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_redis_operations(clean_database):
    """Test Redis operations if available."""
    try:
        redis_client = await clean_database.get_redis_client()

        # Generate unique test keys to avoid conflicts
        import time

        test_run_id = f"test_{int(time.time())}"
        test_key = f"{test_run_id}_key"
        expiring_key = f"{test_run_id}_expiring"
        hash_key = f"{test_run_id}_hash"

        # Test basic operations
        await redis_client.set(test_key, "test_value")
        value = await redis_client.get(test_key)
        # Handle both bytes and string return types
        assert (value.decode("utf-8") if isinstance(value, bytes) else value) == "test_value"

        # Test expiration
        await redis_client.setex(expiring_key, 60, "expires_later")
        value = await redis_client.get(expiring_key)
        # Handle both bytes and string return types
        assert (value.decode("utf-8") if isinstance(value, bytes) else value) == "expires_later"

        # Test hash operations
        await redis_client.hset(hash_key, "field1", "value1")
        await redis_client.hset(hash_key, "field2", "value2")

        field1_value = await redis_client.hget(hash_key, "field1")
        # Handle both bytes and string return types
        assert (
            field1_value.decode("utf-8") if isinstance(field1_value, bytes) else field1_value
        ) == "value1"

        # Test cleanup
        await redis_client.delete(test_key, expiring_key, hash_key)

    except Exception as e:
        # Redis might not be available - let test fail naturally
        raise AssertionError(f"Redis operations failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_complex_queries(clean_database):
    """Test complex database queries."""
    # Initialize real database service with clean database
    from src.database.service import DatabaseService

    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        bots = []
        strategies = []
        async with db_service.get_session() as session:
            # Create multiple bots
            for i in range(3):
                bot = Bot(
                    id=uuid.uuid4(),
                    name=f"Bot {i}",
                    exchange="binance" if i % 2 == 0 else "coinbase",
                    status="running",
                    allocated_capital=Decimal("10000.00"),
                    current_balance=Decimal(f"{10000 + i * 1000}.00"),
                    created_at=datetime.now(timezone.utc),
                )
                bots.append(bot)
                session.add(bot)

            await session.flush()

            # Create strategies for each bot
            for i, bot in enumerate(bots):
                strategy = Strategy(
                    id=uuid.uuid4(),
                    name=f"Strategy {i}",
                    type="momentum",
                    bot_id=bot.id,
                    status="active",
                    params={"param": f"value{i}"},
                    max_position_size=Decimal("1000.0"),
                    risk_per_trade=Decimal("0.02"),
                    created_at=datetime.now(timezone.utc),
                )
                strategies.append(strategy)
                session.add(strategy)

            await session.flush()

            # Create positions for each bot
            for i, bot in enumerate(bots):
                strategy = strategies[i]
                for j in range(2):
                    position = Position(
                        id=uuid.uuid4(),
                        bot_id=bot.id,
                        strategy_id=strategy.id,
                        exchange="binance",
                        symbol=f"{'BTC' if j == 0 else 'ETH'}/USDT",
                        side="LONG",
                        quantity=Decimal("1.0"),
                        entry_price=Decimal(f"{50000 + i * 1000}.00"),
                        status="OPEN" if j == 0 else "CLOSED",
                        created_at=datetime.now(timezone.utc),
                    )
                    session.add(position)

            await session.commit()

        # Test complex queries in a new session
        async with db_service.get_session() as session:
            # Query bots by exchange
            result = await session.execute(select(Bot).where(Bot.exchange == "binance"))
            binance_bots = result.scalars().all()
            assert len(binance_bots) == 2

            # Query open positions
            result = await session.execute(select(Position).where(Position.status == "OPEN"))
            open_positions = result.scalars().all()
            assert len(open_positions) == 3

            # Query positions for specific bot
            result = await session.execute(select(Position).where(Position.bot_id == bots[0].id))
            bot_positions = result.scalars().all()
            assert len(bot_positions) == 2

    finally:
        await db_service.stop()
