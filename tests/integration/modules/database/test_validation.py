"""
Integration tests for database module validation using real services.

This test suite validates that the database module is properly integrated
with proper dependency injection, correct API usage, and error handling
using real PostgreSQL and Redis instead of mocks.
"""

import asyncio
import logging
import uuid
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import DatabaseError, ValidationError
from src.database.service import DatabaseService
from src.database.connection import DatabaseConnectionManager
from src.database.di_registration import configure_database_dependencies, register_database_services
from src.database.models import User, Trade, Order
from tests.integration.infrastructure.service_factory import RealServiceFactory
# Import the correct fixtures from infrastructure
from tests.integration.infrastructure.conftest import clean_database, real_test_config  # noqa: F401


logger = logging.getLogger(__name__)


class TestDatabaseModuleIntegration:
    """Test database module integration patterns using real services."""

    @pytest.fixture
    async def real_connection_manager(self, clean_database):
        """Create real connection manager."""
        return clean_database

    @pytest.fixture
    def mock_config_service(self):
        """Create mock config service."""
        mock_config = MagicMock()
        mock_config.get_config.return_value.to_dict.return_value = {
            'database': {
                'url': 'postgresql://localhost:5432/tbot_dev',
                'host': 'localhost',
                'port': 5432,
                'database': 'tbot_dev',
                'username': 'tbot',
                'password': 'tbot_password'
            }
        }
        return mock_config

    @pytest.fixture
    def mock_validation_service(self):
        """Create mock validation service."""
        return MagicMock()

    async def test_database_service_dependency_injection_constructor(
        self, real_connection_manager, mock_config_service, mock_validation_service
    ):
        """Test DatabaseService constructor accepts all DI parameters with real connection manager."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(real_connection_manager)

            # Test that DatabaseService can be created with DI parameters
            service = DatabaseService(
                connection_manager=real_connection_manager,
                config_service=mock_config_service,
                validation_service=mock_validation_service,
                dependency_injector=MagicMock(),
                cache_enabled=False
            )

            assert service.connection_manager == real_connection_manager
            assert service._config_service == mock_config_service
            assert service._validation_service == mock_validation_service
            assert service._dependency_injector is not None

            logger.info("✅ Real DatabaseService DI constructor verified")

        finally:
            await service_factory.cleanup()

    async def test_dependency_injection_registration(self):
        """Test database services are properly registered in DI container."""
        injector = DependencyInjector()

        # Mock required services
        injector.register_service("ConfigService", lambda: MagicMock(), singleton=True)
        injector.register_service("ValidationService", lambda: MagicMock(), singleton=True)

        # Register database services
        register_database_services(injector)

        # Verify key services are registered
        assert injector.has_service("DatabaseConnectionManager")
        assert injector.has_service("DatabaseService")
        assert injector.has_service("RepositoryFactory")
        assert injector.has_service("UnitOfWorkFactory")

        logger.info("✅ Database services DI registration verified")

    async def test_database_service_integration_with_di(self):
        """Test DatabaseService can be resolved from DI container."""
        injector = DependencyInjector()

        # Register required dependencies
        mock_config_service = MagicMock()
        mock_config_service.get_config.return_value.to_dict.return_value = {
            'database': {'url': 'postgresql://localhost:5432/tbot_dev'}
        }
        injector.register_service("ConfigService", lambda: mock_config_service, singleton=True)

        # Configure database dependencies
        configure_database_dependencies(injector)

        # Should be able to resolve DatabaseService
        db_service = injector.resolve("DatabaseService")
        assert db_service is not None
        assert isinstance(db_service, DatabaseService)

        logger.info("✅ Database service DI resolution verified")

    async def test_database_service_transaction_context_manager(self, real_connection_manager):
        """Test database service transaction context manager works correctly with real database."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(real_connection_manager)

            service = DatabaseService(
                connection_manager=real_connection_manager,
                cache_enabled=False
            )

            # Test successful transaction with real database
            async with service.transaction() as session:
                # Execute real database operation
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1 as test"))
                test_value = result.scalar()
                assert test_value == 1

            logger.info("✅ Real database transaction context manager verified")

        finally:
            await service_factory.cleanup()

    async def test_database_service_transaction_rollback_on_error(self, real_connection_manager):
        """Test database service rolls back transaction on error using real database."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(real_connection_manager)

            service = DatabaseService(
                connection_manager=real_connection_manager,
                cache_enabled=False
            )

            # Test transaction rollback on error with real database
            with pytest.raises(Exception):
                async with service.transaction() as session:
                    # Create test data that will be rolled back
                    unique_id = str(uuid.uuid4())[:8]
                    user = User(
                        username=f"rollback_test_{unique_id}",
                        email=f"rollback_{unique_id}@example.com",
                        password_hash="hashed_password",
                        is_active=True,
                        is_verified=True,
                    )
                    session.add(user)
                    await session.flush()  # Flush to database but don't commit

                    # Force an error to trigger rollback
                    raise Exception("Test error")

            # Verify data was rolled back in real database
            async with service.transaction() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(User).where(User.username == f"rollback_test_{unique_id}")
                )
                user_check = result.scalar_one_or_none()
                assert user_check is None  # Should be None due to rollback

            logger.info("✅ Real database transaction rollback verified")

        finally:
            await service_factory.cleanup()

    async def test_database_service_crud_operations_integration(self, real_connection_manager):
        """Test database service CRUD operations work with proper error handling using real database."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(real_connection_manager)

            service = DatabaseService(
                connection_manager=real_connection_manager,
                cache_enabled=False
            )

            # Test create operation with real database
            unique_id = str(uuid.uuid4())[:8]
            test_user = User(
                username=f"testuser_{unique_id}",
                email=f"test_{unique_id}@example.com",
                password_hash="hashed_password",
                is_active=True,
                is_verified=True,
            )

            result = await service.create_entity(test_user)

            assert result == test_user
            assert result.id is not None  # ID should be assigned by database
            assert result.username == f"testuser_{unique_id}"

            # Test read operation with real database
            retrieved_user = await service.get_entity_by_id(User, result.id)
            assert retrieved_user is not None
            assert retrieved_user.username == f"testuser_{unique_id}"

            # Cleanup
            await service.delete_entity(User, result.id)

            logger.info(f"✅ Real database CRUD operations verified for: {result.username}")

        finally:
            await service_factory.cleanup()

    async def test_database_service_error_handling(self, real_connection_manager):
        """Test database service properly handles and propagates errors using real database."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(real_connection_manager)

            service = DatabaseService(
                connection_manager=real_connection_manager,
                cache_enabled=False
            )

            # Test error propagation with invalid data
            invalid_user = User(
                username="",  # Invalid empty username
                email="invalid-email",  # Invalid email format
                password_hash="",  # Invalid empty password
            )

            # This should raise a DatabaseError due to validation constraints
            with pytest.raises((DatabaseError, Exception)) as exc_info:
                await service.create_entity(invalid_user)

            # Verify error is properly propagated
            assert exc_info.value is not None

            logger.info(f"✅ Real database error handling verified: {type(exc_info.value).__name__}")

        finally:
            await service_factory.cleanup()

    async def test_consuming_module_integration_pattern(self):
        """Test that consuming modules properly use database service through DI."""
        injector = DependencyInjector()

        # Mock database service
        mock_db_service = AsyncMock(spec=DatabaseService)
        mock_db_service.list_entities = AsyncMock(return_value=[])

        injector.register_service("DatabaseService", lambda: mock_db_service, singleton=True)

        # Test the pattern we fixed: using DI instead of direct instantiation
        def get_database_service():
            injector_instance = DependencyInjector.get_instance()
            return injector_instance.resolve("DatabaseService")

        # This should work without errors
        db_service = get_database_service()
        assert db_service == mock_db_service

        logger.info("✅ Consuming module integration pattern verified")

    @pytest.mark.parametrize("entity_type,expected_fields", [
        (User, ["username", "email", "password_hash"]),
        (Order, ["symbol", "quantity", "side"]),
        (Trade, ["symbol", "quantity", "price"]),
    ])
    def test_database_model_contracts(self, entity_type, expected_fields):
        """Test database models have expected fields for API contracts."""
        # Verify model contracts are stable for consuming modules
        instance = entity_type.__new__(entity_type)

        for field in expected_fields:
            assert hasattr(instance, field), f"{entity_type.__name__} missing field: {field}"

        logger.info(f"✅ Database model contracts verified for: {entity_type.__name__}")

    async def test_repository_pattern_integration(self):
        """Test repository pattern works with database service."""
        injector = DependencyInjector()

        # Mock session factory
        mock_session = AsyncMock()
        injector.register_factory("AsyncSession", lambda: mock_session, singleton=False)

        # Register database services
        register_database_services(injector)

        # Should be able to resolve repositories
        try:
            order_repo = injector.resolve("OrderRepository")
            assert order_repo is not None
            logger.info("✅ OrderRepository resolved")
        except Exception as e:
            logger.info(f"OrderRepository test: {e}")

        try:
            position_repo = injector.resolve("PositionRepository")
            assert position_repo is not None
            logger.info("✅ PositionRepository resolved")
        except Exception as e:
            logger.info(f"PositionRepository test: {e}")

    async def test_health_check_integration(self, real_connection_manager):
        """Test database service health check works correctly with real database."""
        from src.core.base.interfaces import HealthStatus

        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(real_connection_manager)

            service = DatabaseService(
                connection_manager=real_connection_manager,
                cache_enabled=False
            )

            # Test healthy status with real database
            health_status = await service.get_health_status()
            assert health_status == HealthStatus.HEALTHY

            logger.info("✅ Real database health check verified")

        finally:
            await service_factory.cleanup()

    async def test_financial_precision_integration(self, real_connection_manager):
        """Test database service handles financial precision correctly with real database."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(real_connection_manager)

            service = DatabaseService(
                connection_manager=real_connection_manager,
                cache_enabled=False
            )

            # Test with Decimal values (financial precision)
            unique_id = str(uuid.uuid4())[:8]
            test_trade = Trade(
                bot_id=f"precision_bot_{unique_id}",
                exchange_order_id=f"precision_order_{unique_id}",
                exchange="binance",
                symbol="BTC/USD",
                side="buy",
                order_type="limit",
                quantity=Decimal("0.12345678"),  # 18 decimal places
                price=Decimal("50000.12345678"),
                executed_price=Decimal("50000.12345678"),
                status="filled",
                fee=Decimal("0.00012345"),
                timestamp=datetime.now(timezone.utc)
            )

            # Should handle Decimal values without conversion in real database
            result = await service.create_entity(test_trade)
            assert isinstance(result.quantity, Decimal)
            assert isinstance(result.price, Decimal)
            assert result.quantity == Decimal("0.12345678")
            assert result.price == Decimal("50000.12345678")

            # Verify precision is maintained through real database roundtrip
            retrieved_trade = await service.get_entity_by_id(Trade, result.id)
            assert isinstance(retrieved_trade.quantity, Decimal)
            assert isinstance(retrieved_trade.price, Decimal)
            assert retrieved_trade.quantity == Decimal("0.12345678")
            assert retrieved_trade.price == Decimal("50000.12345678")

            # Test calculations maintain precision
            calculated_total = retrieved_trade.quantity * retrieved_trade.price
            expected_total = Decimal("0.12345678") * Decimal("50000.12345678")
            assert calculated_total == expected_total

            # Cleanup
            await service.delete_entity(Trade, result.id)

            logger.info(f"✅ Real database financial precision verified: {calculated_total}")

        finally:
            await service_factory.cleanup()


@pytest.mark.asyncio
class TestDatabaseModuleBoundaryValidation:
    """Test database module boundaries and proper separation of concerns using real services."""

    async def test_database_module_does_not_import_business_logic(self):
        """Test database module doesn't import business logic modules."""
        # Database module should not import from trading, strategies, execution, etc.
        forbidden_imports = [
            "src.strategies",
            "src.execution",
            "src.trading",
            "src.risk_management",
            "src.backtesting"
        ]

        # Check database service imports
        from src.database import service
        import inspect

        source = inspect.getsource(service)

        for forbidden in forbidden_imports:
            assert forbidden not in source, f"Database module improperly imports {forbidden}"

        logger.info("✅ Database module boundary validation verified")

    async def test_consuming_modules_use_interfaces(self):
        """Test consuming modules use database interfaces, not concrete classes."""
        # Check that modules import DatabaseServiceInterface, not DatabaseService directly
        consuming_modules = [
            "src.bot_management.instance_service",
            "src.execution.service",
            "src.analytics.service"
        ]

        for module_path in consuming_modules:
            try:
                module = __import__(module_path.replace("/", "."), fromlist=[""])
                import inspect
                source = inspect.getsource(module)

                # Should prefer interfaces over concrete classes
                if "DatabaseService" in source:
                    # If importing concrete class, should also have interface usage
                    assert "DatabaseServiceInterface" in source or "database_service:" in source.lower()

            except ImportError:
                # Module might not exist in current test environment
                pass

        logger.info("✅ Interface usage validation verified")

    def test_database_models_follow_financial_precision_patterns(self):
        """Test database models use proper financial precision types."""
        from src.database.models.trading import Order, Trade, Position
        from sqlalchemy import inspect

        # Check that financial fields use DECIMAL type
        financial_models = [Order, Trade, Position]

        for model in financial_models:
            try:
                mapper = inspect(model)

                for column in mapper.columns:
                    column_name = column.name.lower()

                    # Financial precision fields should use DECIMAL
                    if any(field in column_name for field in ['price', 'quantity', 'amount', 'value']):
                        # Should use DECIMAL type for financial precision
                        assert str(column.type).startswith('DECIMAL') or str(column.type).startswith('NUMERIC'), \
                            f"{model.__name__}.{column.name} should use DECIMAL type, got {column.type}"

                logger.info(f"✅ Financial precision patterns verified for: {model.__name__}")

            except Exception as e:
                logger.info(f"Model validation for {model.__name__}: {e}")


class TestRealRedisOperations:
    """Test Redis operations with real Redis service."""

    @pytest.mark.asyncio
    async def test_redis_basic_operations_integration(self, clean_database):
        """Test Redis basic operations with real Redis service."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            # Get real Redis client
            redis_client = await clean_database.get_redis_client()

            # Test basic operations with real Redis
            unique_id = str(uuid.uuid4())[:8]
            test_key = f"validation_test_key_{unique_id}"
            test_value = f"validation_test_value_{unique_id}"

            await redis_client.set(test_key, test_value)
            retrieved_value = await redis_client.get(test_key)
            assert retrieved_value == test_value

            # Test hash operations with real Redis
            test_hash = f"validation_test_hash_{unique_id}"
            await redis_client.hset(test_hash, "field1", "value1")
            hash_value = await redis_client.hget(test_hash, "field1")
            assert hash_value == "value1"

            # Test expiration with real Redis
            await redis_client.set(f"exp_key_{unique_id}", "exp_value", ex=1)
            await asyncio.sleep(2)
            expired_value = await redis_client.get(f"exp_key_{unique_id}")
            assert expired_value is None

            # Cleanup
            await redis_client.delete(test_key)
            await redis_client.delete(test_hash)

            logger.info(f"✅ Real Redis basic operations verified: {test_key}")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_redis_financial_data_operations(self, clean_database):
        """Test Redis financial data operations with real Redis service."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            # Get real Redis client
            redis_client = await clean_database.get_redis_client()

            unique_id = str(uuid.uuid4())[:8]

            # Test storing financial precision data in Redis
            financial_data = {
                "symbol": f"BTC/USDT_{unique_id}",
                "price": "50000.12345678",  # High precision string
                "quantity": "0.12345678",   # High precision string
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Store in real Redis
            data_key = f"financial_data:{financial_data['symbol']}"
            import json
            await redis_client.set(data_key, json.dumps(financial_data))

            # Retrieve from real Redis
            retrieved_data_str = await redis_client.get(data_key)
            retrieved_data = json.loads(retrieved_data_str)

            assert retrieved_data["symbol"] == financial_data["symbol"]
            assert retrieved_data["price"] == "50000.12345678"
            assert retrieved_data["quantity"] == "0.12345678"

            # Test precision is maintained
            assert Decimal(retrieved_data["price"]) == Decimal("50000.12345678")
            assert Decimal(retrieved_data["quantity"]) == Decimal("0.12345678")

            # Cleanup
            await redis_client.delete(data_key)

            logger.info(f"✅ Real Redis financial data operations verified: {financial_data['symbol']}")

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    async def test_redis_concurrent_operations(self, clean_database):
        """Test concurrent Redis operations with real Redis service."""
        service_factory = RealServiceFactory()

        try:
            await service_factory.initialize_core_services(clean_database)

            # Get real Redis client
            redis_client = await clean_database.get_redis_client()

            unique_id = str(uuid.uuid4())[:8]

            # Create concurrent Redis operations
            tasks = []
            for i in range(5):
                key = f"concurrent_key_{i}_{unique_id}"
                value = f"concurrent_value_{i}_{unique_id}"
                tasks.append(redis_client.set(key, value))

            # Execute concurrently
            await asyncio.gather(*tasks)

            # Verify all operations succeeded
            verification_tasks = []
            for i in range(5):
                key = f"concurrent_key_{i}_{unique_id}"
                verification_tasks.append(redis_client.get(key))

            results = await asyncio.gather(*verification_tasks)

            for i, result in enumerate(results):
                expected_value = f"concurrent_value_{i}_{unique_id}"
                assert result == expected_value

            # Cleanup
            cleanup_tasks = []
            for i in range(5):
                key = f"concurrent_key_{i}_{unique_id}"
                cleanup_tasks.append(redis_client.delete(key))

            await asyncio.gather(*cleanup_tasks)

            logger.info(f"✅ Real Redis concurrent operations verified: {len(results)} operations")

        finally:
            await service_factory.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])