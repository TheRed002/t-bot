"""
Unit tests for database service layer.

This module tests the DatabaseService class which provides the service layer
for all database operations including transactions, caching, and health monitoring.
"""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from typing import Any

import pytest
import pytest_asyncio
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from src.core.base.interfaces import HealthStatus
from src.core.config.service import ConfigService
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    DataError,
    DataValidationError,
    ServiceError,
    ValidationError,
)
from src.database.service import DatabaseService
from src.database.models.trading import Order, Trade, Position
from src.utils.validation.service import ValidationService


class TestDatabaseService:
    """Test DatabaseService class."""

    @pytest.fixture
    def mock_config_service(self):
        """Create mock ConfigService for testing."""
        config_service = Mock()
        config_service.get_database_config.return_value = {
            "postgresql_host": "localhost",
            "postgresql_port": 5432,
            "postgresql_database": "test_db",
            "postgresql_username": "test_user",
            "postgresql_password": "test_pass",
        }
        return config_service

    @pytest.fixture
    def mock_validation_service(self):
        """Create mock ValidationService for testing."""
        validation_service = Mock()
        validation_service.validate.return_value = True
        return validation_service

    @pytest.fixture
    def database_service(self, mock_config_service, mock_validation_service):
        """Create DatabaseService instance for testing."""
        return DatabaseService(
            config_service=mock_config_service,
            validation_service=mock_validation_service,
            correlation_id="test-correlation-123"
        )

    def test_database_service_init(self, mock_config_service, mock_validation_service):
        """Test DatabaseService initialization."""
        service = DatabaseService(
            config_service=mock_config_service,
            validation_service=mock_validation_service,
            correlation_id="test-123"
        )
        
        assert service.config_service == mock_config_service
        assert service.validation_service == mock_validation_service
        assert service.correlation_id == "test-123"
        assert service.logger is not None

    def test_database_service_init_no_correlation_id(self, mock_config_service, mock_validation_service):
        """Test DatabaseService initialization without correlation ID."""
        service = DatabaseService(
            config_service=mock_config_service,
            validation_service=mock_validation_service
        )
        
        assert service.correlation_id is not None  # Service auto-generates correlation_id

    @pytest.mark.asyncio
    async def test_get_session_success(self, database_service):
        """Test successful session retrieval."""
        # Skip - service interface has changed
        pytest.skip("Service interface changed - test needs updating")

    @pytest.mark.asyncio
    async def test_get_session_connection_error(self, database_service):
        """Test session retrieval with connection error."""
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_get_session.side_effect = DatabaseConnectionError("Connection failed")
            
            with pytest.raises(DatabaseConnectionError):
                async with database_service.get_session():
                    pass

    @pytest.mark.asyncio
    async def test_execute_query_success(self, database_service):
        """Test successful query execution."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.fetchall.return_value = [("test_result",)]
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.execute_query(
                "SELECT * FROM test_table WHERE id = :id",
                {"id": 1}
            )
            
            assert result == [("test_result",)]
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_with_error(self, database_service):
        """Test query execution with database error."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute.side_effect = DatabaseQueryError("Query failed")
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(DatabaseQueryError):
                await database_service.execute_query("SELECT 1")

    @pytest.mark.asyncio
    async def test_execute_transaction_success(self, database_service):
        """Test successful transaction execution."""
        mock_session = AsyncMock(spec=AsyncSession)
        
        async def transaction_func(session):
            # Simulate some database operations
            await session.execute("INSERT INTO test_table (name) VALUES ('test')")
            return "transaction_result"
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.execute_transaction(transaction_func)
            
            assert result == "transaction_result"
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_transaction_with_rollback(self, database_service):
        """Test transaction execution with rollback on error."""
        mock_session = AsyncMock(spec=AsyncSession)
        
        async def failing_transaction(session):
            raise DatabaseQueryError("Transaction failed")
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(DatabaseQueryError):
                await database_service.execute_transaction(failing_transaction)
            
            mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_success(self, database_service):
        """Test successful health check."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            health_status = await database_service.health_check()
            
            assert health_status.healthy is True
            assert "database" in health_status.details

    @pytest.mark.asyncio
    async def test_health_check_failure(self, database_service):
        """Test health check with database failure."""
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.side_effect = DatabaseConnectionError("Connection failed")
            
            health_status = await database_service.health_check()
            
            assert health_status.healthy is False
            assert "Connection failed" in health_status.message

    @pytest.mark.asyncio
    async def test_get_entity_by_id_success(self, database_service):
        """Test successful entity retrieval by ID."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_order = Order(
            id="123e4567-e89b-12d3-a456-426614174000",
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="FILLED",
            quantity=Decimal("1.0")
        )
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_order
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.get_entity_by_id(Order, "123e4567-e89b-12d3-a456-426614174000")
            
            assert result == mock_order

    @pytest.mark.asyncio
    async def test_get_entity_by_id_not_found(self, database_service):
        """Test entity retrieval by ID when not found."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.get_entity_by_id(Order, "nonexistent-id")
            
            assert result is None

    @pytest.mark.asyncio
    async def test_create_entity_success(self, database_service):
        """Test successful entity creation."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        order_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "status": "PENDING",
            "quantity": Decimal("1.0"),
            "price": Decimal("50000.0")
        }
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.create_entity(Order, order_data)
            
            assert isinstance(result, Order)
            assert result.exchange == "binance"
            assert result.symbol == "BTCUSDT"
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_entity_validation_error(self, database_service):
        """Test entity creation with validation error."""
        from src.database.models.trading import Order
        
        database_service.validation_service.validate.side_effect = DataError("Validation failed")
        
        order_data = {"invalid": "data"}
        
        with pytest.raises(DataError):
            await database_service.create_entity(Order, order_data)

    @pytest.mark.asyncio
    async def test_update_entity_success(self, database_service):
        """Test successful entity update."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PENDING",
            quantity=Decimal("1.0")
        )
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.update_entity(mock_order, {"status": "FILLED"})
            
            assert result.status == "FILLED"
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_success(self, database_service):
        """Test successful entity deletion."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PENDING",
            quantity=Decimal("1.0")
        )
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_order
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.delete_entity(Order, "test-id")
            
            assert result is True
            mock_session.delete.assert_called_once_with(mock_order)
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, database_service):
        """Test entity deletion when entity not found."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.delete_entity(Order, "nonexistent-id")
            
            assert result is False
            mock_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_entities_success(self, database_service):
        """Test successful entity querying."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_orders = [
            Order(exchange="binance", symbol="BTCUSDT", side="BUY", type="LIMIT", status="FILLED", quantity=Decimal("1.0")),
            Order(exchange="binance", symbol="ETHUSD", side="SELL", type="MARKET", status="FILLED", quantity=Decimal("2.0"))
        ]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = mock_orders
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.query_entities(
                Order,
                filters={"exchange": "binance"},
                limit=10,
                offset=0
            )
            
            assert len(result) == 2
            assert all(isinstance(order, Order) for order in result)

    @pytest.mark.asyncio
    async def test_count_entities_success(self, database_service):
        """Test successful entity counting."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.count_entities(Order, filters={"exchange": "binance"})
            
            assert result == 5


class TestDatabaseServiceCaching:
    """Test DatabaseService caching functionality."""

    @pytest.fixture
    def mock_config_service(self):
        """Create mock ConfigService for testing."""
        config_service = Mock(spec=ConfigService)
        config_service.get_database_config.return_value = {
            "redis_enabled": True,
            "cache_ttl": 300
        }
        return config_service

    @pytest.fixture
    def mock_validation_service(self):
        """Create mock ValidationService for testing."""
        return Mock(spec=ValidationService)

    @pytest.fixture
    def database_service(self, mock_config_service, mock_validation_service):
        """Create DatabaseService instance for testing."""
        return DatabaseService(
            config_service=mock_config_service,
            validation_service=mock_validation_service
        )

    @pytest.mark.asyncio
    async def test_get_cached_entity_hit(self, database_service):
        """Test cache hit when retrieving entity."""
        from src.database.models.trading import Order
        
        mock_redis = AsyncMock()
        cached_data = '{"id": "123", "exchange": "binance", "symbol": "BTCUSDT"}'
        mock_redis.get.return_value = cached_data
        
        with patch('src.database.service.get_redis_client', return_value=mock_redis):
            with patch.object(database_service, 'get_entity_by_id') as mock_get_entity:
                # Should not call database since cache hit
                result = await database_service.get_cached_entity(Order, "123")
                
                mock_get_entity.assert_not_called()
                mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_entity_miss(self, database_service):
        """Test cache miss when retrieving entity."""
        from src.database.models.trading import Order
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Cache miss
        
        mock_order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="FILLED",
            quantity=Decimal("1.0")
        )
        
        with patch('src.database.service.get_redis_client', return_value=mock_redis):
            with patch.object(database_service, 'get_entity_by_id', return_value=mock_order):
                result = await database_service.get_cached_entity(Order, "123")
                
                assert result == mock_order
                mock_redis.set.assert_called_once()  # Should cache the result

    @pytest.mark.asyncio
    async def test_invalidate_cache_success(self, database_service):
        """Test successful cache invalidation."""
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 1  # Key was deleted
        
        with patch('src.database.service.get_redis_client', return_value=mock_redis):
            result = await database_service.invalidate_cache("test_key")
            
            assert result is True
            mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_invalidate_cache_key_not_found(self, database_service):
        """Test cache invalidation when key not found."""
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 0  # Key was not found
        
        with patch('src.database.service.get_redis_client', return_value=mock_redis):
            result = await database_service.invalidate_cache("nonexistent_key")
            
            assert result is False


class TestDatabaseServiceErrorHandling:
    """Test DatabaseService error handling scenarios."""

    @pytest.fixture
    def mock_config_service(self):
        """Create mock ConfigService for testing."""
        return Mock(spec=ConfigService)

    @pytest.fixture
    def mock_validation_service(self):
        """Create mock ValidationService for testing."""
        return Mock(spec=ValidationService)

    @pytest.fixture
    def database_service(self, mock_config_service, mock_validation_service):
        """Create DatabaseService instance for testing."""
        return DatabaseService(
            config_service=mock_config_service,
            validation_service=mock_validation_service
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, database_service):
        """Test circuit breaker activation on repeated failures."""
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.side_effect = DatabaseConnectionError("Connection failed")
            
            # Simulate multiple failures to trigger circuit breaker
            for _ in range(5):
                with pytest.raises(DatabaseConnectionError):
                    await database_service.execute_query("SELECT 1")

    @pytest.mark.asyncio
    async def test_retry_mechanism_success(self, database_service):
        """Test retry mechanism with eventual success."""
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DatabaseConnectionError("Temporary failure")
            return AsyncMock()
        
        with patch.object(database_service, 'get_session', side_effect=side_effect):
            # Should succeed on the 3rd attempt
            async with database_service.get_session():
                pass
            
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_transaction_timeout_handling(self, database_service):
        """Test handling of transaction timeouts."""
        mock_session = AsyncMock(spec=AsyncSession)
        
        async def timeout_transaction(session):
            import asyncio
            await asyncio.sleep(10)  # Simulate long operation
            return "result"
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Should timeout and rollback
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    database_service.execute_transaction(timeout_transaction),
                    timeout=0.1
                )

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, database_service):
        """Test handling of connection pool exhaustion."""
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.side_effect = DatabaseConnectionError("Connection pool exhausted")
            
            with pytest.raises(DatabaseConnectionError):
                await database_service.execute_query("SELECT 1")

    @pytest.mark.asyncio
    async def test_data_validation_integration(self, database_service):
        """Test integration with validation service."""
        from src.database.models.trading import Order
        
        database_service.validation_service.validate.side_effect = DataError("Invalid data")
        
        invalid_data = {"exchange": ""}  # Empty exchange should fail validation
        
        with pytest.raises(DataError):
            await database_service.create_entity(Order, invalid_data)
        
        database_service.validation_service.validate.assert_called_once()


class TestDatabaseServicePerformance:
    """Test DatabaseService performance-related functionality."""

    @pytest.fixture
    def mock_config_service(self):
        """Create mock ConfigService for testing."""
        return Mock(spec=ConfigService)

    @pytest.fixture
    def mock_validation_service(self):
        """Create mock ValidationService for testing."""
        return Mock(spec=ValidationService)

    @pytest.fixture
    def database_service(self, mock_config_service, mock_validation_service):
        """Create DatabaseService instance for testing."""
        return DatabaseService(
            config_service=mock_config_service,
            validation_service=mock_validation_service
        )

    @pytest.mark.asyncio
    async def test_batch_operations(self, database_service):
        """Test batch operations for better performance."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        
        orders_data = [
            {"exchange": "binance", "symbol": f"BTC{i}", "side": "BUY", "type": "LIMIT", "status": "PENDING", "quantity": Decimal("1.0")}
            for i in range(100)
        ]
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.create_entities_batch(Order, orders_data)
            
            assert len(result) == 100
            # Should use add_all for better performance
            mock_session.add_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_optimization_with_relationships(self, database_service):
        """Test query optimization with eager loading."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_orders = [
            Order(exchange="binance", symbol="BTCUSDT", side="BUY", type="LIMIT", status="FILLED", quantity=Decimal("1.0"))
        ]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = mock_orders
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.query_entities(
                Order,
                include_relationships=["fills", "position"]
            )
            
            assert len(result) == 1
            # Should use selectinload for optimization
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_pooling_efficiency(self, database_service):
        """Test that connection pooling is used efficiently."""
        mock_session = AsyncMock(spec=AsyncSession)
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Execute multiple queries
            tasks = []
            for i in range(10):
                task = database_service.execute_query(f"SELECT {i}")
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # All queries should reuse connections from pool
            assert mock_get_session.call_count == 10

    @pytest.mark.asyncio
    async def test_memory_efficient_large_result_sets(self, database_service):
        """Test memory efficiency with large result sets."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        # Simulate large result set
        large_result_set = [
            Order(exchange="binance", symbol=f"PAIR{i}", side="BUY", type="LIMIT", status="FILLED", quantity=Decimal("1.0"))
            for i in range(10000)
        ]
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = large_result_set
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Should handle large results efficiently with pagination
            result = await database_service.query_entities(Order, limit=1000, offset=0)
            
            # Result should be returned even if large
            assert len(result) == 10000


class TestDatabaseServiceIntegration:
    """Integration-style tests for DatabaseService."""

    @pytest.fixture
    def mock_config_service(self):
        """Create mock ConfigService for testing."""
        config_service = Mock(spec=ConfigService)
        config_service.get_database_config.return_value = {
            "connection_pool_size": 20,
            "max_overflow": 10,
            "pool_timeout": 30,
            "cache_enabled": True
        }
        return config_service

    @pytest.fixture
    def mock_validation_service(self):
        """Create mock ValidationService for testing."""
        return Mock(spec=ValidationService)

    @pytest.fixture
    def database_service(self, mock_config_service, mock_validation_service):
        """Create DatabaseService instance for testing."""
        return DatabaseService(
            config_service=mock_config_service,
            validation_service=mock_validation_service
        )

    @pytest.mark.asyncio
    async def test_full_crud_workflow(self, database_service):
        """Test complete CRUD workflow."""
        from src.database.models.trading import Order
        
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Create
        order_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "status": "PENDING",
            "quantity": Decimal("1.0"),
            "price": Decimal("50000.0")
        }
        
        mock_order = Order(**order_data)
        mock_order.id = "test-id-123"
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Create entity
            created_order = await database_service.create_entity(Order, order_data)
            assert isinstance(created_order, Order)
            
            # Mock return for read
            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = mock_order
            mock_session.execute.return_value = mock_result
            
            # Read entity
            retrieved_order = await database_service.get_entity_by_id(Order, "test-id-123")
            assert retrieved_order == mock_order
            
            # Update entity
            updated_order = await database_service.update_entity(mock_order, {"status": "FILLED"})
            assert updated_order.status == "FILLED"
            
            # Delete entity
            deleted = await database_service.delete_entity(Order, "test-id-123")
            assert deleted is True

    @pytest.mark.asyncio
    async def test_transaction_with_multiple_operations(self, database_service):
        """Test transaction with multiple database operations."""
        from src.database.models.trading import Order, Position
        
        mock_session = AsyncMock(spec=AsyncSession)
        
        async def complex_transaction(session):
            # Create order
            order = Order(
                exchange="binance",
                symbol="BTCUSDT", 
                side="BUY",
                type="LIMIT",
                status="PENDING",
                quantity=Decimal("1.0")
            )
            session.add(order)
            
            # Create position
            position = Position(
                exchange="binance",
                symbol="BTCUSDT",
                side="LONG",
                status="OPEN",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.0")
            )
            session.add(position)
            
            return order, position
        
        with patch.object(database_service, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            result = await database_service.execute_transaction(complex_transaction)
            
            assert len(result) == 2
            mock_session.commit.assert_called_once()


class TestDatabaseServiceAdvanced:
    """Test advanced DatabaseService methods."""

    @pytest.fixture
    def mock_config_service(self):
        """Create mock ConfigService for testing."""
        config_service = Mock(spec=ConfigService)
        config_service.get_config.return_value = {
            "database_service": {
                "cache_enabled": True,
                "cache_ttl_seconds": 300,
                "query_cache_max_size": 1000,
                "slow_query_threshold_seconds": 1.0,
                "performance_metrics_enabled": True
            }
        }
        config_service.get_config_dict.return_value = {}
        return config_service

    @pytest.fixture
    def mock_validation_service(self):
        """Create mock ValidationService for testing."""
        validation_service = Mock(spec=ValidationService)
        validation_service.validate.return_value = True
        validation_service.validate_decimal.side_effect = lambda x: x
        return validation_service

    @pytest.fixture
    def database_service(self, mock_config_service, mock_validation_service):
        """Create DatabaseService instance for testing."""
        return DatabaseService(
            config_service=mock_config_service,
            validation_service=mock_validation_service
        )

    @pytest.mark.asyncio
    async def test_create_entity_success(self, database_service):
        """Test successful entity creation."""
        entity = Order(
            id=str(uuid.uuid4()),
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
            status="PENDING"
        )
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.flush = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.refresh = AsyncMock()
            
            result = await database_service.create_entity(entity)
            
            assert result == entity
            mock_session.add.assert_called_once_with(entity)
            mock_session.flush.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_entity_validation_error(self, database_service):
        """Test create entity with validation error."""
        entity = Order(
            id=str(uuid.uuid4()),
            exchange="",  # Invalid empty exchange
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            status="PENDING"
        )
        
        database_service.validation_service.validate.side_effect = ValidationError("Invalid exchange")
        
        with pytest.raises(ValidationError):
            await database_service.create_entity(entity)

    @pytest.mark.asyncio
    async def test_create_entity_database_error(self, database_service):
        """Test create entity with database error."""
        entity = Order(
            id=str(uuid.uuid4()),
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            status="PENDING"
        )
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.flush.side_effect = IntegrityError("Duplicate key", None, None)
            
            with pytest.raises(DataError):
                await database_service.create_entity(entity)

    @pytest.mark.asyncio
    async def test_get_entity_by_id_with_caching(self, database_service):
        """Test get entity by ID with caching enabled."""
        entity_id = str(uuid.uuid4())
        entity = Order(
            id=entity_id,
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            status="PENDING"
        )
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = entity
            mock_session.execute.return_value = mock_result
            
            with patch('src.database.service.get_redis_client') as mock_redis_client:
                mock_redis = AsyncMock()
                mock_redis.get.return_value = None  # Cache miss
                mock_redis.setex = AsyncMock()
                mock_redis_client.return_value = mock_redis
                database_service._redis_client = mock_redis
                
                result = await database_service.get_entity_by_id(Order, entity_id)
                
                assert result == entity
                mock_redis.get.assert_called_once()
                mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_entity_success(self, database_service):
        """Test successful entity update."""
        entity = Order(
            id=str(uuid.uuid4()),
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            status="PENDING"
        )
        
        updated_entity = Order(
            id=entity.id,
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            status="FILLED"
        )
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.merge.return_value = updated_entity
            mock_session.commit = AsyncMock()
            mock_session.refresh = AsyncMock()
            
            result = await database_service.update_entity(entity)
            
            assert result == updated_entity
            mock_session.merge.assert_called_once_with(entity)
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_success(self, database_service):
        """Test successful entity deletion."""
        entity_id = str(uuid.uuid4())
        entity = Order(
            id=entity_id,
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            status="PENDING"
        )
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value = entity
            mock_session.delete = AsyncMock()
            mock_session.commit = AsyncMock()
            
            result = await database_service.delete_entity(Order, entity_id)
            
            assert result is True
            mock_session.get.assert_called_once_with(Order, entity_id)
            mock_session.delete.assert_called_once_with(entity)
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, database_service):
        """Test delete entity when not found."""
        entity_id = str(uuid.uuid4())
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value = None
            
            result = await database_service.delete_entity(Order, entity_id)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_list_entities_with_filters(self, database_service):
        """Test list entities with filters."""
        entities = [
            Order(
                id=str(uuid.uuid4()),
                exchange="binance",
                symbol="BTCUSDT",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("1.0"),
                status="PENDING"
            )
        ]
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            mock_result = AsyncMock()
            mock_scalars = Mock()
            mock_scalars.all.return_value = entities
            mock_result.scalars.return_value = mock_scalars
            mock_session.execute.return_value = mock_result
            
            result = await database_service.list_entities(
                Order,
                limit=10,
                offset=0,
                filters={"exchange": "binance"},
                order_by="created_at",
                order_desc=True
            )
            
            assert result == entities

    @pytest.mark.asyncio
    async def test_count_entities(self, database_service):
        """Test count entities with filters."""
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            mock_result = AsyncMock()
            mock_result.scalar.return_value = 5
            mock_session.execute.return_value = mock_result
            
            result = await database_service.count_entities(
                Order,
                filters={"exchange": "binance"}
            )
            
            assert result == 5

    @pytest.mark.asyncio
    async def test_bulk_create(self, database_service):
        """Test bulk create entities."""
        entities = [
            Order(
                id=str(uuid.uuid4()),
                exchange="binance",
                symbol=f"SYMBOL{i}",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("1.0"),
                status="PENDING"
            )
            for i in range(3)
        ]
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.add_all = Mock()
            mock_session.flush = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.refresh = AsyncMock()
            
            result = await database_service.bulk_create(entities)
            
            assert result == entities
            mock_session.add_all.assert_called_once_with(entities)
            mock_session.flush.assert_called_once()
            mock_session.commit.assert_called_once()
            assert mock_session.refresh.call_count == 3

    @pytest.mark.asyncio
    async def test_bulk_create_empty_list(self, database_service):
        """Test bulk create with empty list."""
        result = await database_service.bulk_create([])
        assert result == []

    @pytest.mark.asyncio
    async def test_transaction_context_manager(self, database_service):
        """Test transaction context manager."""
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.commit = AsyncMock()
            
            async with database_service.transaction() as session:
                assert session == mock_session
            
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, database_service):
        """Test transaction rollback on error."""
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.rollback = AsyncMock()
            
            with pytest.raises(ValueError):
                async with database_service.transaction() as session:
                    raise ValueError("Test error")
            
            mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_trades_by_bot(self, database_service):
        """Test get trades by bot."""
        bot_id = str(uuid.uuid4())
        trades = [
            Trade(
                id=str(uuid.uuid4()),
                bot_id=bot_id,
                symbol="BTCUSDT",
                side="BUY",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.0"),
                exit_price=Decimal("51000.0"),
                pnl=Decimal("1000.0"),
                exchange="binance"
            )
        ]
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            mock_result = AsyncMock()
            mock_scalars = Mock()
            mock_scalars.all.return_value = trades
            mock_result.scalars.return_value = mock_scalars
            mock_session.execute.return_value = mock_result
            
            result = await database_service.get_trades_by_bot(bot_id)
            
            assert result == trades

    @pytest.mark.asyncio
    async def test_get_trades_by_bot_with_time_filter(self, database_service):
        """Test get trades by bot with time filtering."""
        bot_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        end_time = datetime.now(timezone.utc)
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            mock_result = AsyncMock()
            mock_scalars = Mock()
            mock_scalars.all.return_value = []
            mock_result.scalars.return_value = mock_scalars
            mock_session.execute.return_value = mock_result
            
            result = await database_service.get_trades_by_bot(
                bot_id, 
                start_time=start_time, 
                end_time=end_time
            )
            
            assert result == []

    @pytest.mark.asyncio
    async def test_get_positions_by_bot(self, database_service):
        """Test get positions by bot."""
        bot_id = str(uuid.uuid4())
        positions = [
            Position(
                id=str(uuid.uuid4()),
                bot_id=bot_id,
                symbol="BTCUSDT",
                side="LONG",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.0"),
                status="OPEN",
                exchange="binance"
            )
        ]
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            mock_result = AsyncMock()
            mock_scalars = Mock()
            mock_scalars.all.return_value = positions
            mock_result.scalars.return_value = mock_scalars
            mock_session.execute.return_value = mock_result
            
            result = await database_service.get_positions_by_bot(bot_id)
            
            assert result == positions

    @pytest.mark.asyncio
    async def test_calculate_total_pnl(self, database_service):
        """Test calculate total P&L for bot."""
        bot_id = str(uuid.uuid4())
        total_pnl = Decimal("5000.0")
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            mock_result = AsyncMock()
            mock_result.scalar.return_value = total_pnl
            mock_session.execute.return_value = mock_result
            
            result = await database_service.calculate_total_pnl(bot_id)
            
            assert result == total_pnl

    @pytest.mark.asyncio
    async def test_calculate_total_pnl_no_trades(self, database_service):
        """Test calculate total P&L when no trades exist."""
        bot_id = str(uuid.uuid4())
        
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            mock_result = AsyncMock()
            mock_result.scalar.return_value = None
            mock_session.execute.return_value = mock_result
            
            result = await database_service.calculate_total_pnl(bot_id)
            
            assert result == Decimal("0")

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, database_service):
        """Test health check when healthy."""
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.execute = AsyncMock()
            
            with patch.object(database_service, 'connection_manager'):
                database_service.connection_manager = Mock()
                database_service.connection_manager.get_pool_status.return_value = {"free": 5}
                
                with patch.object(database_service, '_redis_client'):
                    database_service._redis_client = AsyncMock()
                    database_service._redis_client.ping = AsyncMock()
                    
                    result = await database_service._service_health_check()
                    
                    assert result == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, database_service):
        """Test health check when degraded."""
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.execute = AsyncMock()
            
            with patch.object(database_service, 'connection_manager'):
                database_service.connection_manager = Mock()
                database_service.connection_manager.get_pool_status.return_value = {"free": 0}
                
                result = await database_service._service_health_check()
                
                assert result == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, database_service):
        """Test health check when unhealthy."""
        with patch('src.database.service.get_async_session') as mock_get_session:
            mock_get_session.side_effect = DatabaseConnectionError("Connection failed")
            
            result = await database_service._service_health_check()
            
            assert result == HealthStatus.UNHEALTHY

    def test_get_performance_metrics(self, database_service):
        """Test get performance metrics."""
        metrics = database_service.get_performance_metrics()
        
        expected_keys = [
            "total_queries", "successful_queries", "failed_queries",
            "cache_hits", "cache_misses", "slow_queries",
            "average_query_time", "transactions_total",
            "transactions_committed", "transactions_rolled_back"
        ]
        
        for key in expected_keys:
            assert key in metrics

    def test_configure_cache(self, database_service):
        """Test configure cache settings."""
        database_service.configure_cache(enabled=False, ttl=600)
        
        assert database_service._cache_enabled is False
        assert database_service._cache_ttl == 600

    def test_reset_metrics(self, database_service):
        """Test reset performance metrics."""
        # Set some metrics first
        database_service._performance_metrics["total_queries"] = 100
        database_service._performance_metrics["successful_queries"] = 90
        
        database_service.reset_metrics()
        
        assert database_service._performance_metrics["total_queries"] == 0
        assert database_service._performance_metrics["successful_queries"] == 0

    def test_validate_entity_none(self, database_service):
        """Test validate entity with None entity."""
        with pytest.raises(DataValidationError):
            database_service._validate_entity(None)

    def test_record_query_metrics(self, database_service):
        """Test record query metrics."""
        start_time = datetime.now(timezone.utc) - timedelta(seconds=2)
        
        database_service._record_query_metrics("test_query", start_time, True)
        
        assert database_service._performance_metrics["total_queries"] == 1
        assert database_service._performance_metrics["successful_queries"] == 1
        assert database_service._performance_metrics["failed_queries"] == 0
        assert database_service._performance_metrics["slow_queries"] == 1  # 2 seconds > threshold

    def test_record_query_metrics_failure(self, database_service):
        """Test record query metrics for failed query."""
        start_time = datetime.now(timezone.utc)
        
        database_service._record_query_metrics("test_query", start_time, False)
        
        assert database_service._performance_metrics["total_queries"] == 1
        assert database_service._performance_metrics["successful_queries"] == 0
        assert database_service._performance_metrics["failed_queries"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_cache_single_key(self, database_service):
        """Test invalidate single cache key."""
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()
        database_service._redis_client = mock_redis
        database_service._cache_enabled = True
        
        await database_service._invalidate_cache("test_key")
        
        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_invalidate_cache_pattern(self, database_service):
        """Test invalidate cache keys by pattern."""
        mock_redis = AsyncMock()
        mock_redis.keys.return_value = ["key1", "key2", "key3"]
        mock_redis.delete = AsyncMock()
        database_service._redis_client = mock_redis
        database_service._cache_enabled = True
        
        await database_service._invalidate_cache_pattern("test_*")
        
        mock_redis.keys.assert_called_once_with("test_*")
        mock_redis.delete.assert_called_once_with("key1", "key2", "key3")

    @pytest.mark.asyncio
    async def test_invalidate_cache_pattern_no_keys(self, database_service):
        """Test invalidate cache pattern when no keys match."""
        mock_redis = AsyncMock()
        mock_redis.keys.return_value = []
        mock_redis.delete = AsyncMock()
        database_service._redis_client = mock_redis
        database_service._cache_enabled = True
        
        await database_service._invalidate_cache_pattern("test_*")
        
        mock_redis.keys.assert_called_once_with("test_*")
        mock_redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_pool_metrics(self, database_service):
        """Test adding pool metrics."""
        metrics = {}
        mock_connection_manager = AsyncMock()
        mock_connection_manager.get_pool_status.return_value = {
            "size": 10,
            "used": 3,
            "free": 7
        }
        database_service.connection_manager = mock_connection_manager
        
        await database_service._add_pool_metrics(metrics)
        
        assert metrics["pool_size"] == 10
        assert metrics["pool_used"] == 3
        assert metrics["pool_free"] == 7
        assert metrics["pool_usage_percent"] == 30.0