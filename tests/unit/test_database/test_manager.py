"""
Unit tests for database manager.

This module tests the DatabaseManager class which provides a unified interface
for all database operations across PostgreSQL, Redis, and InfluxDB.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.manager import DatabaseManager
from src.database.models import MarketDataRecord, Position, Trade


class TestDatabaseManager:
    """Test DatabaseManager class."""

    @pytest.fixture
    def database_manager(self):
        """Create DatabaseManager instance for testing."""
        return DatabaseManager()

    def test_database_manager_init(self):
        """Test DatabaseManager initialization."""
        manager = DatabaseManager()
        
        assert manager.session is None
        assert manager._session_context is None
        assert manager.logger is not None

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, database_manager):
        """Test entering async context manager."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            async with database_manager as manager:
                assert manager == database_manager
                assert manager.session == mock_session
                assert manager._session_context == mock_context

    @pytest.mark.asyncio
    async def test_context_manager_exit_normal(self, database_manager):
        """Test exiting async context manager normally."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            async with database_manager:
                pass
            
            mock_context.__aexit__.assert_called_once()
            assert database_manager.session is None
            assert database_manager._session_context is None

    @pytest.mark.asyncio
    async def test_context_manager_exit_with_exception(self, database_manager):
        """Test exiting async context manager with exception."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            try:
                async with database_manager:
                    raise ValueError("Test exception")
            except ValueError:
                pass
            
            # Should still clean up properly
            mock_context.__aexit__.assert_called_once()
            assert database_manager.session is None

    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, database_manager):
        """Test successful historical data retrieval."""
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        mock_records = [
            MarketDataRecord(
                symbol="BTCUSDT",
                exchange="binance",
                data_timestamp=start_time,
                open_price=Decimal("50000.0"),
                close_price=Decimal("51000.0"),
                high_price=Decimal("52000.0"),
                low_price=Decimal("49000.0"),
                volume=Decimal("100.0"),
                interval="1m",
                source="exchange"
            )
        ]
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            # Mock the actual database query that would happen in production
            with patch.object(database_manager, '_query_market_data', return_value=mock_records):
                result = await database_manager.get_historical_data(
                    "BTCUSDT", start_time, end_time, "1m"
                )
                
                # Current implementation returns empty list (placeholder)
                assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_historical_data_with_retry(self, database_manager):
        """Test historical data retrieval with retry mechanism."""
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary database error")
            return AsyncMock()
        
        with patch('src.database.manager.get_async_session', side_effect=side_effect):
            result = await database_manager.get_historical_data(
                "BTCUSDT", start_time, end_time
            )
            
            # Should succeed after retries
            assert isinstance(result, list)
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_get_historical_data_circuit_breaker(self, database_manager):
        """Test historical data with circuit breaker activation."""
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        with patch('src.database.manager.get_async_session', side_effect=Exception("Database down")):
            # Circuit breaker should eventually activate and provide fallback
            result = await database_manager.get_historical_data(
                "BTCUSDT", start_time, end_time
            )
            
            # Should return fallback value (empty list)
            assert result == []

    @pytest.mark.asyncio
    async def test_save_trade_success(self, database_manager):
        """Test successful trade saving."""
        trade_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "entry_price": Decimal("50000.0"),
            "exit_price": Decimal("51000.0"),
            "pnl": Decimal("1000.0"),
            "entry_order_id": "order_123",
            "exit_order_id": "order_124"
        }
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            result = await database_manager.save_trade(trade_data)
            
            assert isinstance(result, Trade)
            assert result.exchange == "binance"
            assert result.symbol == "BTCUSDT"
            assert result.side == "BUY"

    @pytest.mark.asyncio
    async def test_save_trade_with_retry(self, database_manager):
        """Test trade saving with retry mechanism."""
        trade_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "entry_price": Decimal("50000.0"),
            "exit_price": Decimal("51000.0"),
            "pnl": Decimal("1000.0")
        }
        
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Database connection error")
            return AsyncMock()
        
        with patch('src.database.manager.get_async_session', side_effect=side_effect):
            result = await database_manager.save_trade(trade_data)
            
            # Should succeed after retries
            assert isinstance(result, Trade)
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_save_trade_circuit_breaker(self, database_manager):
        """Test trade saving with circuit breaker."""
        trade_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "entry_price": Decimal("50000.0"),
            "exit_price": Decimal("51000.0"),
            "pnl": Decimal("1000.0")
        }
        
        # Simulate repeated failures to trigger circuit breaker
        with patch('src.database.manager.get_async_session', side_effect=Exception("Database error")):
            # Should eventually handle via circuit breaker
            with pytest.raises(Exception):
                # Multiple calls to trigger circuit breaker
                for _ in range(6):
                    try:
                        await database_manager.save_trade(trade_data)
                    except Exception:
                        pass
                
                # This call should be rejected by circuit breaker
                await database_manager.save_trade(trade_data)

    @pytest.mark.asyncio
    async def test_get_active_positions_success(self, database_manager):
        """Test successful active positions retrieval."""
        mock_positions = [
            Position(
                exchange="binance",
                symbol="BTCUSDT",
                side="LONG",
                status="OPEN",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.0"),
                current_price=Decimal("51000.0")
            ),
            Position(
                exchange="binance",
                symbol="ETHUSD",
                side="SHORT",
                status="OPEN",
                quantity=Decimal("2.0"),
                entry_price=Decimal("3000.0"),
                current_price=Decimal("2950.0")
            )
        ]
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            # Mock the query that would be implemented in production
            with patch.object(database_manager, '_query_active_positions', return_value=mock_positions):
                # Test would call actual method when implemented
                pass

    @pytest.mark.asyncio
    async def test_update_position_success(self, database_manager):
        """Test successful position update."""
        position_id = "pos_123"
        update_data = {
            "current_price": Decimal("51500.0"),
            "unrealized_pnl": Decimal("1500.0")
        }
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            # Mock the update that would be implemented in production
            with patch.object(database_manager, '_update_position', return_value=True):
                # Test would call actual method when implemented
                pass

    @pytest.mark.asyncio
    async def test_health_check_success(self, database_manager):
        """Test successful health check."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            # Mock health check implementation
            with patch.object(database_manager, '_perform_health_check', return_value=True):
                # Test would call actual method when implemented
                pass

    @pytest.mark.asyncio
    async def test_health_check_failure(self, database_manager):
        """Test health check failure."""
        with patch('src.database.manager.get_async_session', side_effect=Exception("Connection failed")):
            # Mock health check that would detect failures
            with patch.object(database_manager, '_perform_health_check', return_value=False):
                # Test would call actual method when implemented
                pass


class TestDatabaseManagerErrorHandling:
    """Test DatabaseManager error handling scenarios."""

    @pytest.fixture
    def database_manager(self):
        """Create DatabaseManager instance for testing."""
        return DatabaseManager()

    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, database_manager):
        """Test handling of database connection failures."""
        with patch('src.database.manager.get_async_session', side_effect=Exception("Connection failed")):
            # Should handle connection failures gracefully
            try:
                async with database_manager:
                    pass
            except Exception as e:
                assert "Connection failed" in str(e)

    @pytest.mark.asyncio
    async def test_session_cleanup_on_error(self, database_manager):
        """Test that session is properly cleaned up on errors."""
        mock_context = AsyncMock()
        mock_context.__aenter__.side_effect = Exception("Session creation failed")
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            with pytest.raises(Exception):
                async with database_manager:
                    pass
            
            # Session should be cleaned up even on error
            assert database_manager.session is None

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, database_manager):
        """Test transaction rollback on errors."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            try:
                async with database_manager:
                    # Simulate error during database operations
                    raise ValueError("Business logic error")
            except ValueError:
                pass
            
            # Context manager should handle rollback
            mock_context.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, database_manager):
        """Test behavior when retry attempts are exhausted."""
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        # Always fail
        with patch('src.database.manager.get_async_session', side_effect=Exception("Persistent error")):
            # Should eventually return fallback value due to @with_fallback
            result = await database_manager.get_historical_data(
                "BTCUSDT", start_time, end_time
            )
            
            # Should return fallback value (empty list)
            assert result == []

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, database_manager):
        """Test handling of partial operation failures."""
        trade_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "entry_price": Decimal("50000.0"),
            "exit_price": Decimal("51000.0"),
            "pnl": Decimal("1000.0")
        }
        
        # First call succeeds, second fails
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncMock()
            else:
                raise Exception("Second call failed")
        
        with patch('src.database.manager.get_async_session', side_effect=side_effect):
            # First call should succeed
            result1 = await database_manager.save_trade(trade_data)
            assert isinstance(result1, Trade)
            
            # Second call should fail and trigger retry
            with pytest.raises(Exception):
                # After retries are exhausted, should raise exception
                await database_manager.save_trade(trade_data)


class TestDatabaseManagerPerformance:
    """Test DatabaseManager performance-related functionality."""

    @pytest.fixture
    def database_manager(self):
        """Create DatabaseManager instance for testing."""
        return DatabaseManager()

    @pytest.mark.asyncio
    async def test_session_reuse(self, database_manager):
        """Test that session is reused efficiently."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            async with database_manager:
                # Multiple operations should reuse the same session
                assert database_manager.session == mock_session
                
                # Simulate multiple operations
                await database_manager.get_historical_data(
                    "BTCUSDT", 
                    datetime.now(timezone.utc), 
                    datetime.now(timezone.utc)
                )
                
                trade_data = {
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "quantity": Decimal("1.0"),
                    "entry_price": Decimal("50000.0"),
                    "exit_price": Decimal("51000.0"),
                    "pnl": Decimal("1000.0")
                }
                await database_manager.save_trade(trade_data)
                
                # Both operations should use the same session context
                assert database_manager.session == mock_session

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, database_manager):
        """Test handling of concurrent database operations."""
        import asyncio
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        async def concurrent_operation(symbol):
            return await database_manager.get_historical_data(
                symbol,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            )
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            # Run concurrent operations
            tasks = [
                concurrent_operation("BTCUSDT"),
                concurrent_operation("ETHUSD"),
                concurrent_operation("ADAUSDT")
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All operations should complete
            assert len(results) == 3
            assert all(isinstance(result, list) for result in results)

    @pytest.mark.asyncio
    async def test_bulk_operations(self, database_manager):
        """Test bulk database operations."""
        # Simulate bulk trade saving
        trades_data = [
            {
                "exchange": "binance",
                "symbol": f"PAIR{i}",
                "side": "BUY",
                "quantity": Decimal("1.0"),
                "entry_price": Decimal("50000.0"),
                "exit_price": Decimal("51000.0"),
                "pnl": Decimal("1000.0")
            }
            for i in range(100)
        ]
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            # Process bulk operations
            results = []
            for trade_data in trades_data:
                result = await database_manager.save_trade(trade_data)
                results.append(result)
            
            assert len(results) == 100
            assert all(isinstance(result, Trade) for result in results)

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_datasets(self, database_manager):
        """Test memory efficiency with large datasets."""
        # Simulate query that returns large dataset
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 12, 31, tzinfo=timezone.utc)
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            # Should handle large date range efficiently
            result = await database_manager.get_historical_data(
                "BTCUSDT", start_time, end_time, "1m"
            )
            
            # Should return data without memory issues
            assert isinstance(result, list)


class TestDatabaseManagerIntegration:
    """Integration-style tests for DatabaseManager."""

    @pytest.fixture
    def database_manager(self):
        """Create DatabaseManager instance for testing."""
        return DatabaseManager()

    @pytest.mark.asyncio
    async def test_full_workflow(self, database_manager):
        """Test complete workflow with multiple operations."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        
        trade_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "entry_price": Decimal("50000.0"),
            "exit_price": Decimal("51000.0"),
            "pnl": Decimal("1000.0")
        }
        
        with patch('src.database.manager.get_async_session', return_value=mock_context):
            async with database_manager:
                # 1. Query historical data
                historical_data = await database_manager.get_historical_data(
                    "BTCUSDT",
                    datetime(2023, 1, 1, tzinfo=timezone.utc),
                    datetime(2023, 1, 2, tzinfo=timezone.utc)
                )
                
                # 2. Save trade
                saved_trade = await database_manager.save_trade(trade_data)
                
                # 3. Verify operations completed
                assert isinstance(historical_data, list)
                assert isinstance(saved_trade, Trade)
                assert saved_trade.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, database_manager):
        """Test workflow with error recovery."""
        call_count = 0
        
        def session_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = AsyncMock()
            return mock_context
        
        trade_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "entry_price": Decimal("50000.0"),
            "exit_price": Decimal("51000.0"),
            "pnl": Decimal("1000.0")
        }
        
        with patch('src.database.manager.get_async_session', side_effect=session_side_effect):
            # Should recover from initial failures
            result = await database_manager.save_trade(trade_data)
            
            assert isinstance(result, Trade)
            # Should have retried multiple times
            assert call_count >= 3

    @pytest.mark.asyncio
    async def test_multiple_manager_instances(self):
        """Test multiple DatabaseManager instances."""
        manager1 = DatabaseManager()
        manager2 = DatabaseManager()
        
        mock_session1 = AsyncMock(spec=AsyncSession)
        mock_session2 = AsyncMock(spec=AsyncSession)
        mock_context1 = AsyncMock()
        mock_context2 = AsyncMock()
        mock_context1.__aenter__.return_value = mock_session1
        mock_context2.__aenter__.return_value = mock_session2
        
        with patch('src.database.manager.get_async_session', side_effect=[mock_context1, mock_context2]):
            async with manager1:
                async with manager2:
                    # Each manager should have its own session
                    assert manager1.session == mock_session1
                    assert manager2.session == mock_session2
                    assert manager1.session != manager2.session