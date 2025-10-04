"""
Phase 2: Database Module Integration Tests

Tests database module integration with other system modules:
1. Database service integration with dependency injection
2. Module boundary respect and service layer patterns
3. Real database operations through service interfaces
4. Cross-module database usage patterns

NO MOCKS - All operations use real database services and interfaces.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.core.dependency_injection import DependencyContainer
from src.database.service import DatabaseService
from src.database.models import User, Bot, Strategy, Position
from src.database.interfaces import DatabaseServiceInterface
from tests.integration.infrastructure.conftest import clean_database


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_module_dependency_injection(clean_database):
    """Test database module integrates properly with DI container."""
    # Create DI container
    container = DependencyContainer()

    # Register real database service
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        container.register("DatabaseService", db_service, singleton=True)
        container.register("DatabaseServiceInterface", db_service, singleton=True)

        # Test service resolution
        resolved_service = container.get("DatabaseService")
        assert resolved_service is db_service

        resolved_interface = container.get("DatabaseServiceInterface")
        assert resolved_interface is db_service
        assert isinstance(resolved_interface, DatabaseServiceInterface)

        # Test service functionality through DI
        async with resolved_service.get_session() as session:
            unique_id = uuid.uuid4().hex[:8]
            user = User(
                id=uuid.uuid4(),
                username=f"di_test_user_{unique_id}",
                email=f"di_{unique_id}@test.com",
                password_hash="hashed_password_test",
                created_at=datetime.now(timezone.utc)
            )
            session.add(user)
            await session.commit()

            # Verify through interface
            result = await session.execute(
                select(User).where(User.username == f"di_test_user_{unique_id}")
            )
            retrieved_user = result.scalar_one()
            assert retrieved_user.email == f"di_{unique_id}@test.com"

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_service_interface_compliance(clean_database):
    """Test database service implements required interfaces correctly."""
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        # Test interface compliance
        assert isinstance(db_service, DatabaseServiceInterface)

        # Test required interface methods
        assert hasattr(db_service, "get_session")
        assert hasattr(db_service, "get_health_status")
        assert hasattr(db_service, "start")
        assert hasattr(db_service, "stop")

        # Test methods work as expected
        health_status = await db_service.get_health_status()
        assert health_status.value == "healthy"

        async with db_service.get_session() as session:
            assert session is not None

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_module_boundaries(clean_database):
    """Test database module respects proper boundaries."""
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        # Database module should provide clean interfaces
        # Other modules should not access database internals directly

        async with db_service.get_session() as session:
            # This is the proper way - through service interface
            unique_id = uuid.uuid4().hex[:8]
            user = User(
                id=uuid.uuid4(),
                username=f"boundary_test_{unique_id}",
                email=f"boundary_{unique_id}@test.com",
                password_hash="hashed_password_test",
                created_at=datetime.now(timezone.utc)
            )
            session.add(user)
            await session.commit()

            # Verify the operation worked
            result = await session.execute(
                select(User).where(User.username == f"boundary_test_{unique_id}")
            )
            retrieved_user = result.scalar_one()
            assert retrieved_user is not None

        # Test that service provides proper abstractions
        health_status = await db_service.get_health_status()
        assert health_status is not None

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_service_error_handling(clean_database):
    """Test database service error handling and resilience."""
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        async with db_service.get_session() as session:
            # Test constraint violation handling
            unique_id_1 = uuid.uuid4().hex[:8]
            duplicate_uuid = uuid.uuid4()
            user1 = User(
                id=duplicate_uuid,
                username=f"user1_{unique_id_1}",
                email=f"user1_{unique_id_1}@test.com",
                password_hash="hashed_password_test",
                created_at=datetime.now(timezone.utc)
            )
            session.add(user1)
            await session.commit()

        # Test constraint violation within the same session
        async with db_service.get_session() as test_session:
            unique_id_2 = uuid.uuid4().hex[:8]
            user2 = User(
                id=duplicate_uuid,  # Same ID
                username=f"user2_{unique_id_2}",
                email=f"user2_{unique_id_2}@test.com",
                password_hash="hashed_password_test",
                created_at=datetime.now(timezone.utc)
            )
            test_session.add(user2)

            # Should raise constraint violation when trying to commit duplicate UUID
            with pytest.raises((Exception, IntegrityError)):
                await test_session.commit()

        # Test successful operation after error
        async with db_service.get_session() as session:
            unique_id_valid = uuid.uuid4().hex[:8]
            valid_user = User(
                id=uuid.uuid4(),
                username=f"valid_user_{unique_id_valid}",
                email=f"valid_{unique_id_valid}@test.com",
                password_hash="hashed_password_test",
                created_at=datetime.now(timezone.utc)
            )
            session.add(valid_user)
            await session.commit()

            # Verify it worked
            result = await session.execute(
                select(User).where(User.username == f"valid_user_{unique_id_valid}")
            )
            retrieved_user = result.scalar_one()
            assert retrieved_user.email == f"valid_{unique_id_valid}@test.com"

    finally:
        await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_service_lifecycle_management(clean_database):
    """Test database service lifecycle management."""
    db_service = DatabaseService(clean_database)

    # Test start
    await db_service.start()
    health_status = await db_service.get_health_status()
    assert health_status.value == "healthy"

    # Test service is functional
    async with db_service.get_session() as session:
        result = await session.execute(select(1))
        assert result.scalar() == 1

    # Test stop
    await db_service.stop()

    # Test restart
    await db_service.start()
    health_status = await db_service.get_health_status()
    assert health_status.value == "healthy"

    # Clean shutdown
    await db_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_models_with_service_layer(clean_database):
    """Test database models work correctly with service layer."""
    db_service = DatabaseService(clean_database)
    await db_service.start()

    try:
        async with db_service.get_session() as session:
            # Create bot first
            bot = Bot(
                id=uuid.uuid4(),
                name="Module Test Bot",
                exchange="binance",
                status="running",
                created_at=datetime.now(timezone.utc)
            )
            session.add(bot)
            await session.flush()

            # Create strategy with valid bot_id reference
            strategy = Strategy(
                id=uuid.uuid4(),
                name="Module Test Strategy",
                type="custom",
                status="active",
                bot_id=bot.id,
                max_position_size=Decimal("1000.0"),
                risk_per_trade=Decimal("0.02"),
                created_at=datetime.now(timezone.utc)
            )
            session.add(strategy)
            await session.flush()

            # Test model creation through service
            # Use unique symbol for this test
            unique_symbol = f"TEST_{uuid.uuid4().hex[:8]}/USDT"
            position = Position(
                id=uuid.uuid4(),
                bot_id=bot.id,
                strategy_id=strategy.id,
                exchange="binance",
                symbol=unique_symbol,
                side="LONG",
                status="OPEN",
                quantity=Decimal("10.0"),
                entry_price=Decimal("100.0"),
                current_price=Decimal("105.0"),
                unrealized_pnl=Decimal("50.0"),
                created_at=datetime.now(timezone.utc)
            )
            session.add(position)
            await session.commit()

            # Test model retrieval
            result = await session.execute(
                select(Position).where(Position.symbol == unique_symbol)
            )
            retrieved_position = result.scalar_one()

            # Verify model attributes
            assert retrieved_position.side == "LONG"
            assert retrieved_position.quantity == Decimal("10.0")
            assert retrieved_position.unrealized_pnl == Decimal("50.0")

            # Test model updates
            retrieved_position.current_price = Decimal("110.0")
            retrieved_position.unrealized_pnl = Decimal("100.0")
            await session.commit()

            # Verify update
            updated_result = await session.execute(
                select(Position).where(Position.id == position.id)
            )
            updated_position = updated_result.scalar_one()
            assert updated_position.current_price == Decimal("110.0")
            assert updated_position.unrealized_pnl == Decimal("100.0")

    finally:
        await db_service.stop()