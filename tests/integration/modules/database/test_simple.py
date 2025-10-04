"""
Phase 2: Simple Database Integration Tests

Basic database operations with real services:
1. Simple CRUD operations with real PostgreSQL
2. Basic Redis cache operations
3. Simple health checks and connectivity
4. Basic model validation

NO MOCKS - All operations use real database services.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select

from src.database.service import DatabaseService
from src.database.models import User, Bot, Strategy, Position
from src.database.redis_client import RedisClient
from src.core.config import get_config
from tests.integration.infrastructure.conftest import clean_database


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_database_crud_operations(clean_database):
    """Test basic CRUD operations with real database."""
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        async with db_service.get_session() as session:
            # CREATE - Insert a real user
            user_id = str(uuid.uuid4())
            unique_id = uuid.uuid4().hex[:8]
            user = User(
                id=user_id,
                username=f"simple_test_user_{unique_id}",
                email=f"simple_{unique_id}@test.com",
                password_hash="hashed_password_test",
                created_at=datetime.now(timezone.utc)
            )
            session.add(user)
            await session.commit()

            # READ - Query the user back
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            retrieved_user = result.scalar_one()
            assert retrieved_user.username == f"simple_test_user_{unique_id}"
            assert retrieved_user.email == f"simple_{unique_id}@test.com"

            # UPDATE - Modify the user
            retrieved_user.email = "updated@test.com"
            await session.commit()

            # Verify update
            updated_result = await session.execute(
                select(User).where(User.id == user_id)
            )
            updated_user = updated_result.scalar_one()
            assert updated_user.email == "updated@test.com"

            # DELETE - Remove the user
            await session.delete(retrieved_user)
            await session.commit()

            # Verify deletion
            deleted_result = await session.execute(
                select(User).where(User.id == user_id)
            )
            deleted_user = deleted_result.scalar_one_or_none()
            assert deleted_user is None

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_redis_cache_operations(clean_database):
    """Test basic Redis cache operations."""
    # Use the Redis client from the clean database connection manager
    redis_client = await clean_database.get_redis_client()

    # Generate unique test keys to avoid conflicts
    test_run_id = "simple_" + str(hash(str(clean_database)))[-8:]
    test_key = f"test_{test_run_id}_key"
    expire_key = f"expire_{test_run_id}_test"

    # Simple set/get operations
    await redis_client.set(test_key, "test_value")
    value = await redis_client.get(test_key)
    assert value == "test_value"

    # Test exists
    exists = await redis_client.exists(test_key)
    assert exists

    # Test delete
    await redis_client.delete(test_key)
    deleted_value = await redis_client.get(test_key)
    assert deleted_value is None

    # Test with expiry
    await redis_client.setex(expire_key, 1, "will_expire")
    immediate_value = await redis_client.get(expire_key)
    assert immediate_value == "will_expire"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_database_health_check(clean_database):
    """Test simple database health checks."""
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        # Test health status
        health_status = await db_service.get_health_status()
        assert health_status.value == "healthy"

        # Test database connectivity
        async with db_service.get_session() as session:
            # Simple connectivity test
            result = await session.execute(select(1))
            assert result.scalar() == 1

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_model_validation(clean_database):
    """Test basic model validation with real database."""
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        async with db_service.get_session() as session:
            # Create bot first
            bot = Bot(
                id=uuid.uuid4(),
                name="Simple Test Bot",
                exchange="binance",
                status="running",
                created_at=datetime.now(timezone.utc)
            )
            session.add(bot)
            await session.flush()

            # Create strategy with valid bot_id reference
            strategy = Strategy(
                id=uuid.uuid4(),
                name="Simple Test Strategy",
                type="custom",
                status="active",
                bot_id=bot.id,
                max_position_size=Decimal("1000.0"),
                risk_per_trade=Decimal("0.02"),
                created_at=datetime.now(timezone.utc)
            )
            session.add(strategy)
            await session.flush()

            # Test valid position creation
            position = Position(
                id=uuid.uuid4(),
                bot_id=bot.id,
                strategy_id=strategy.id,
                exchange="binance",
                symbol="BTC/USDT",
                side="LONG",
                status="OPEN",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.0"),
                current_price=Decimal("50000.0"),
                unrealized_pnl=Decimal("0.0"),
                created_at=datetime.now(timezone.utc)
            )
            session.add(position)
            await session.commit()

            # Verify position was saved
            result = await session.execute(
                select(Position).where(Position.id == position.id)
            )
            saved_position = result.scalar_one()
            assert saved_position.symbol == "BTC/USDT"
            assert saved_position.quantity == Decimal("1.0")

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_session_management(clean_database):
    """Test basic database session management."""
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        # Test session context manager
        async with db_service.get_session() as session:
            # Session should be active and available
            assert session is not None
            assert hasattr(session, 'execute')  # Verify it's a valid session

            # Simple query to verify session works
            result = await session.execute(select(1))
            assert result.scalar() == 1

        # Session should be closed after context manager

        # Test another session
        async with db_service.get_session() as session2:
            result2 = await session2.execute(select(2))
            assert result2.scalar() == 2

    finally:
        await db_service.stop()