"""
Database Integration Tests - Dual Mode (Real/Mock Services)

Tests database operations using either real Docker services or mock services
based on the USE_MOCK_SERVICES environment variable.
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select

# Set environment variable before imports
os.environ.setdefault("USE_MOCK_SERVICES", "true")

# Add infrastructure path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from infrastructure.dual_mode_conftest import (
    database_service,
    influxdb_client,
    redis_client,
    test_services,
)

from src.database.models import Bot, Order, Position, Strategy, User


@pytest.mark.integration
async def test_database_connections(test_services):
    """Test database connections and health checks."""
    # Test database health
    db_service = test_services["database"]
    health_status = await db_service.get_health_status()
    assert health_status.value == "healthy"

    # Test Redis connection
    redis = test_services["redis"]
    assert await redis.ping()

    # Test InfluxDB connection
    influx = test_services["influxdb"]
    assert await influx.health_check()


@pytest.mark.integration
async def test_model_persistence_and_relationships(database_service):
    """Test SQLAlchemy models with database persistence."""
    async with database_service.get_session() as session:
        # Create user
        user = User(
            id=uuid.uuid4(),
            username=f"trader_{uuid.uuid4().hex[:8]}",
            email=f"trader_{uuid.uuid4().hex[:8]}@test.com",
            password_hash="hashed_password",
            created_at=datetime.now(timezone.utc),
        )
        session.add(user)
        await session.flush()

        # Create bot
        bot = Bot(
            id=uuid.uuid4(),
            name="Test Trading Bot",
            exchange="binance",
            status="ready",
            allocated_capital=Decimal("10000.00"),
            current_balance=Decimal("10000.00"),
            created_at=datetime.now(timezone.utc),
        )
        session.add(bot)
        await session.flush()

        # Create strategy
        strategy = Strategy(
            id=uuid.uuid4(),
            name="Momentum Strategy",
            type="momentum",
            bot_id=bot.id,
            params={"lookback": 20, "threshold": 0.02},
            status="active",
            max_position_size=Decimal("1000.00"),
            risk_per_trade=Decimal("0.02"),
            created_at=datetime.now(timezone.utc),
        )
        session.add(strategy)
        await session.flush()

        # Create position with relationships
        position = Position(
            id=uuid.uuid4(),
            bot_id=bot.id,
            strategy_id=strategy.id,
            exchange="binance",
            symbol="BTC/USDT",
            side="LONG",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00"),
            unrealized_pnl=Decimal("1000.00"),
            status="OPEN",
            created_at=datetime.now(timezone.utc),
        )
        session.add(position)
        await session.flush()

        # Create order linked to position
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

        # Verify persistence and relationships
        result = await session.execute(
            select(Position).where(Position.id == position.id)
        )
        retrieved_position = result.scalar_one()
        assert retrieved_position.bot_id == bot.id
        assert retrieved_position.strategy_id == strategy.id
        assert retrieved_position.symbol == "BTC/USDT"

        result = await session.execute(
            select(Order).where(Order.position_id == position.id)
        )
        retrieved_order = result.scalar_one()
        assert retrieved_order.bot_id == bot.id
        assert retrieved_order.status == "filled"


@pytest.mark.integration
async def test_database_transactions_and_rollback(database_service):
    """Test database transactions with rollback scenarios."""
    # Test successful transaction
    async with database_service.get_session() as session:
        # Create required parent objects first
        bot = Bot(
            id=uuid.uuid4(),
            name="Test Bot",
            exchange="binance",
            status="ready",
            allocated_capital=Decimal("10000.00"),
            current_balance=Decimal("10000.00"),
        )
        session.add(bot)
        await session.flush()

        strategy = Strategy(
            id=uuid.uuid4(),
            name="Test Strategy",
            type="momentum",
            bot_id=bot.id,
            params={"test": "value"},
            status="active",
            max_position_size=Decimal("1000.00"),
            risk_per_trade=Decimal("0.02"),
        )
        session.add(strategy)
        await session.flush()

        position1 = Position(
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
        session.add(position1)
        await session.commit()

    # Verify persistence
    async with database_service.get_session() as session:
        result = await session.execute(select(Position))
        positions = result.scalars().all()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"

    # Test rollback on error
    try:
        async with database_service.get_session() as session:
            # Get the existing bot and strategy for the second position
            result = await session.execute(select(Bot).limit(1))
            bot = result.scalar_one()
            result = await session.execute(select(Strategy).limit(1))
            strategy = result.scalar_one()

            position2 = Position(
                id=uuid.uuid4(),
                bot_id=bot.id,
                strategy_id=strategy.id,
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

            # Force an error to trigger rollback
            raise ValueError("Intentional error for rollback test")

    except ValueError:
        pass  # Expected error

    # Verify rollback worked - only first position should exist
    async with database_service.get_session() as session:
        result = await session.execute(select(Position))
        positions = result.scalars().all()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"


@pytest.mark.integration
async def test_redis_cache_operations(redis_client):
    """Test Redis cache operations."""
    # Test basic operations
    await redis_client.set("test_key", "test_value")
    value = await redis_client.get("test_key")
    assert value == "test_value"

    # Test expiration
    await redis_client.setex("expiring_key", 2, "expires_soon")
    value = await redis_client.get("expiring_key")
    assert value == "expires_soon"

    # Test TTL
    ttl = await redis_client.ttl("expiring_key")
    assert 0 < ttl <= 2

    # Test hash operations
    await redis_client.hset("hash_key", mapping={"field1": "value1", "field2": "value2"})
    hash_data = await redis_client.hgetall("hash_key")
    assert hash_data["field1"] == "value1"
    assert hash_data["field2"] == "value2"

    # Test list operations
    await redis_client.lpush("list_key", "item1", "item2", "item3")
    items = await redis_client.lrange("list_key", 0, -1)
    assert len(items) == 3

    # Test deletion
    deleted = await redis_client.delete("test_key")
    assert deleted == 1
    value = await redis_client.get("test_key")
    assert value is None


@pytest.mark.integration
async def test_influxdb_time_series_operations(influxdb_client):
    """Test InfluxDB time-series operations."""
    # Write market data points
    now = datetime.now(timezone.utc)

    for i in range(10):
        await influxdb_client.write_point(
            measurement="market_data",
            tags={"symbol": "BTC/USDT", "exchange": "binance"},
            fields={"price": float(50000 + i * 100), "volume": float(100 + i * 10)},
            timestamp=now - timedelta(minutes=10 - i),
        )

    # Query data range
    points = await influxdb_client.query_range(
        measurement="market_data",
        start_time=now - timedelta(minutes=15),
        tags={"symbol": "BTC/USDT"},
    )

    assert len(points) == 10
    assert all(p["tags"]["symbol"] == "BTC/USDT" for p in points)

    # Verify data ordering
    prices = [p["fields"]["price"] for p in points]
    assert prices[0] == 50000.0
    assert prices[-1] == 50900.0


@pytest.mark.integration
async def test_complex_queries(database_service):
    """Test complex database queries."""
    async with database_service.get_session() as session:
        # Create multiple bots with positions
        bots = []
        for i in range(3):
            bot = Bot(
                id=uuid.uuid4(),
                name=f"Bot {i}",
                exchange="binance" if i % 2 == 0 else "coinbase",
                status="ready",
                allocated_capital=Decimal("10000.00"),
                current_balance=Decimal(f"{10000 + i * 1000}.00"),
                created_at=datetime.now(timezone.utc),
            )
            bots.append(bot)
            session.add(bot)

        await session.flush()

        # Create strategies for each bot
        strategies = []
        for i, bot in enumerate(bots):
            strategy = Strategy(
                id=uuid.uuid4(),
                name=f"Strategy {i}",
                type="momentum",
                bot_id=bot.id,
                params={"test": f"value_{i}"},
                status="active",
                max_position_size=Decimal("1000.00"),
                risk_per_trade=Decimal("0.02"),
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
                    exchange=bot.exchange,
                    symbol=f"{'BTC' if j == 0 else 'ETH'}/USDT",
                    side="LONG",
                    quantity=Decimal("1.0"),
                    entry_price=Decimal(f"{50000 + i * 1000}.00"),
                    status="OPEN" if j == 0 else "CLOSED",
                    created_at=datetime.now(timezone.utc),
                )
                session.add(position)

        await session.commit()

    # Test complex queries
    async with database_service.get_session() as session:
        # Query bots by exchange
        result = await session.execute(select(Bot).where(Bot.exchange == "binance"))
        binance_bots = result.scalars().all()
        assert len(binance_bots) == 2

        # Query open positions
        result = await session.execute(
            select(Position).where(Position.status == "OPEN")
        )
        open_positions = result.scalars().all()
        assert len(open_positions) == 3

        # Query positions for specific bot
        result = await session.execute(
            select(Position).where(Position.bot_id == bots[0].id)
        )
        bot_positions = result.scalars().all()
        assert len(bot_positions) == 2


@pytest.mark.integration
async def test_batch_operations(database_service):
    """Test batch database operations."""
    async with database_service.get_session() as session:
        # Batch insert users
        users = []
        for i in range(10):
            user = User(
                id=uuid.uuid4(),
                username=f"batch_user_{i}",
                email=f"batch_{i}@test.com",
                password_hash="hashed",
                created_at=datetime.now(timezone.utc),
            )
            users.append(user)

        session.add_all(users)
        await session.commit()

        # Verify batch insert
        result = await session.execute(select(User))
        all_users = result.scalars().all()
        assert len(all_users) >= 10

        # Batch update
        for user in users[:5]:
            user.email = f"updated_{user.username}@test.com"

        await session.commit()

        # Verify batch update
        result = await session.execute(
            select(User).where(User.email.like("updated_%"))
        )
        updated_users = result.scalars().all()
        assert len(updated_users) == 5


@pytest.mark.integration
async def test_concurrent_operations(test_services):
    """Test concurrent database and cache operations."""
    db_service = test_services["database"]
    redis = test_services["redis"]

    async def db_operation(index: int):
        async with db_service.get_session() as session:
            user = User(
                id=uuid.uuid4(),
                username=f"concurrent_user_{index}",
                email=f"concurrent_{index}@test.com",
                password_hash="hashed",
                created_at=datetime.now(timezone.utc),
            )
            session.add(user)
            await session.commit()
            return user.id

    async def cache_operation(index: int):
        key = f"concurrent_key_{index}"
        value = f"concurrent_value_{index}"
        await redis.set(key, value)
        retrieved = await redis.get(key)
        return retrieved == value

    # Run concurrent operations
    tasks = []
    for i in range(5):
        tasks.append(db_operation(i))
        tasks.append(cache_operation(i))

    results = await asyncio.gather(*tasks)

    # Verify all operations succeeded
    user_ids = [r for r in results if isinstance(r, uuid.UUID)]
    cache_results = [r for r in results if isinstance(r, bool)]

    assert len(user_ids) == 5
    assert all(cache_results)


@pytest.mark.integration
async def test_cache_with_database_consistency(test_services):
    """Test cache consistency with database operations."""
    db_service = test_services["database"]
    redis = test_services["redis"]

    # Create position in database
    async with db_service.get_session() as session:
        # Create required parent objects
        bot = Bot(
            id=uuid.uuid4(),
            name="Test Cache Bot",
            exchange="binance",
            status="ready",
            allocated_capital=Decimal("10000.00"),
            current_balance=Decimal("10000.00"),
        )
        session.add(bot)
        await session.flush()

        strategy = Strategy(
            id=uuid.uuid4(),
            name="Test Cache Strategy",
            type="momentum",
            bot_id=bot.id,
            params={"test": "cache"},
            status="active",
            max_position_size=Decimal("1000.00"),
            risk_per_trade=Decimal("0.02"),
        )
        session.add(strategy)
        await session.flush()

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
        await session.commit()
        position_id = str(position.id)

    # Cache position data
    cache_key = f"position:{position_id}"
    cache_data = {
        "symbol": "BTC/USDT",
        "side": "LONG",
        "quantity": "1.0",
        "entry_price": "50000.00",
    }
    await redis.hset(cache_key, mapping=cache_data)

    # Verify cache matches database
    cached = await redis.hgetall(cache_key)
    assert cached["symbol"] == "BTC/USDT"
    assert cached["quantity"] == "1.0"

    # Update database
    async with db_service.get_session() as session:
        result = await session.execute(
            select(Position).where(Position.id == uuid.UUID(position_id))
        )
        position = result.scalar_one()
        position.current_price = Decimal("51000.00")
        position.unrealized_pnl = Decimal("1000.00")
        await session.commit()

    # Update cache
    await redis.hset(cache_key, "current_price", "51000.00")
    await redis.hset(cache_key, "unrealized_pnl", "1000.00")

    # Verify consistency
    cached = await redis.hgetall(cache_key)
    assert cached["current_price"] == "51000.00"
    assert cached["unrealized_pnl"] == "1000.00"