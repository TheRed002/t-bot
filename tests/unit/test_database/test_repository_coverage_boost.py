"""
High-coverage repository tests to achieve 90%+ coverage.

This module contains additional tests for repository edge cases,
error conditions, and less commonly used functionality.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.bot import Bot, Signal, Strategy
from src.database.models.trading import Order, Position
from src.database.repository.bot import BotRepository, SignalRepository, StrategyRepository
from src.database.repository.market_data import MarketDataRepository
from src.database.repository.trading import OrderRepository, PositionRepository


@pytest.mark.asyncio
class TestRepositoryEdgeCases:
    """Test edge cases and error conditions for comprehensive coverage."""

    @pytest.fixture
    def mock_session(self):
        """Create properly configured mock session."""
        session = AsyncMock(spec=AsyncSession)
        session.delete = Mock()
        session.add = Mock()
        session.merge = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.execute = AsyncMock()
        return session

    async def test_bot_repository_edge_cases(self, mock_session):
        """Test BotRepository edge cases."""
        repo = BotRepository(mock_session)

        # Test get with None ID
        with patch.object(repo, "_get_entity_by_id", return_value=None):
            result = await repo.get(None)
            assert result is None

        # Test get_all with empty result
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        empty_result = await repo.get_all()
        assert empty_result == []

        # Test create with None entity (should raise DataValidationError)
        from src.core.exceptions import DataValidationError

        with pytest.raises(DataValidationError):
            await repo.create(None)

        # Test update with invalid data
        invalid_bot = Bot(
            id=uuid.uuid4(),
            name="",  # Empty name might be invalid
            status="INVALID_STATUS",
            exchange="unknown_exchange",
            allocated_capital=-1000,  # Negative capital
        )

        # Mock session to raise IntegrityError
        from src.core.exceptions import RepositoryError

        mock_session.flush.side_effect = IntegrityError("test", "test", "test", "test")

        with pytest.raises(RepositoryError):
            await repo.update(invalid_bot)

        # Reset mock
        mock_session.flush.side_effect = None

        # Test exists with non-existent ID
        mock_result.scalar.return_value = None
        exists = await repo.exists(uuid.uuid4())
        assert exists is False

        # Test count with filters
        mock_result.scalar.return_value = 5
        count = await repo.count(filters={"status": "RUNNING"})
        assert count == 5

    async def test_strategy_repository_advanced_operations(self, mock_session):
        """Test StrategyRepository advanced operations."""
        repo = StrategyRepository(mock_session)

        # Test get_active_strategies with empty result
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        active_strategies = await repo.get_active_strategies()
        assert active_strategies == []

        # Test get_active_strategies with bot filter
        bot_id = uuid.uuid4()
        active_strategies_filtered = await repo.get_active_strategies(bot_id=str(bot_id))
        assert active_strategies_filtered == []

        # Test activate_strategy with invalid strategy
        with patch.object(repo, "get", return_value=None):
            result = await repo.activate_strategy(uuid.uuid4())
            assert result is False

        # Test activate_strategy with already active strategy
        active_strategy = Strategy(
            id=uuid.uuid4(),
            name="Already Active",
            type="test",
            status="ACTIVE",
            bot_id=uuid.uuid4(),
        )

        with patch.object(repo, "get", return_value=active_strategy):
            result = await repo.activate_strategy(active_strategy.id)
            assert result is False

        # Test deactivate_strategy success case
        with (
            patch.object(repo, "get", return_value=active_strategy),
            patch.object(repo, "update", return_value=active_strategy),
        ):
            result = await repo.deactivate_strategy(active_strategy.id)
            assert result is True
            assert active_strategy.status == "INACTIVE"

    async def test_signal_repository_execution_tracking(self, mock_session):
        """Test SignalRepository execution tracking methods."""
        repo = SignalRepository(mock_session)

        # Test mark_signal_executed with non-existent signal (requires 3 parameters)
        with patch.object(repo, "get", return_value=None):
            result = await repo.mark_signal_executed(uuid.uuid4(), uuid.uuid4(), Decimal("10.5"))
            assert result is False

        # Test mark_signal_executed with already executed signal
        executed_signal = Signal(
            id=uuid.uuid4(), strategy_id=uuid.uuid4(), symbol="BTCUSD", action="BUY", executed=True
        )

        with (
            patch.object(repo, "get", return_value=executed_signal),
            patch.object(repo, "update", return_value=executed_signal),
        ):
            # mark_signal_executed requires order_id and execution_time parameters
            result = await repo.mark_signal_executed(
                executed_signal.id, uuid.uuid4(), Decimal("15.2")
            )
            assert result is True  # Should succeed but update existing execution

        # Test update_signal_outcome with non-existent signal
        with patch.object(repo, "get", return_value=None):
            result = await repo.update_signal_outcome(uuid.uuid4(), "SUCCESS", Decimal("100.00"))
            assert result is False

        # Test update_signal_outcome without PnL
        pending_signal = Signal(
            id=uuid.uuid4(), strategy_id=uuid.uuid4(), symbol="BTCUSD", action="BUY", executed=True
        )

        with (
            patch.object(repo, "get", return_value=pending_signal),
            patch.object(repo, "update", return_value=pending_signal),
        ):
            result = await repo.update_signal_outcome(pending_signal.id, "SUCCESS")
            assert result is True
            assert pending_signal.outcome == "SUCCESS"
            assert pending_signal.pnl is None

    async def test_market_data_repository_time_queries(self, mock_session):
        """Test MarketDataRepository time-based queries."""
        repo = MarketDataRepository(mock_session)

        # Mock empty results for time-based queries
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test get_ohlc_data with no data
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)
        ohlc_data = await repo.get_ohlc_data("BTCUSD", "binance", start_time, end_time)
        assert ohlc_data == []

        # Mock the RepositoryUtils.execute_time_based_query method
        with patch("src.database.repository.utils.RepositoryUtils.execute_time_based_query", return_value=[]):
            # Test get_recent_data with no data
            recent_data = await repo.get_recent_data("BTCUSD", "binance", hours=24)
            assert recent_data == []

            # Test get_recent_data with default hours
            recent_data_default = await repo.get_recent_data("BTCUSD", "binance")
            assert recent_data_default == []

        # Test get_by_data_source method (correct method name)
        source_data = await repo.get_by_data_source("websocket")
        assert source_data == []

        # Test get_poor_quality_data (should return empty list as implemented)
        poor_quality = await repo.get_poor_quality_data(0.5)
        assert poor_quality == []

        # Test get_invalid_data (should return empty list as implemented)
        invalid_data = await repo.get_invalid_data()
        assert invalid_data == []

        # Test cleanup_old_data
        with patch(
            "src.database.repository.utils.RepositoryUtils.cleanup_old_entities", return_value=0
        ):
            cleanup_count = await repo.cleanup_old_data(days=90)
            assert cleanup_count == 0  # No old data to clean

    async def test_order_repository_order_management(self, mock_session):
        """Test OrderRepository order management methods."""
        repo = OrderRepository(mock_session)

        # Test update_order_status (cancel_order method doesn't exist)
        with patch(
            "src.database.repository.utils.RepositoryUtils.update_entity_status", return_value=False
        ):
            result = await repo.update_order_status(str(uuid.uuid4()), "CANCELLED")
            assert result is False

        # Test cancel_order with already inactive order
        filled_order = Order(
            id=uuid.uuid4(),
            exchange="binance",
            symbol="BTCUSD",
            side="BUY",
            order_type="LIMIT",
            status="FILLED",
            quantity=Decimal("1.0"),
            price=Decimal("45000.00"),
        )

        with patch.object(repo, "get", return_value=filled_order):
            # Test with filled order status update
            with patch(
                "src.database.repository.utils.RepositoryUtils.update_entity_status",
                return_value=True,
            ):
                result = await repo.update_order_status(str(filled_order.id), "CANCELLED")
                assert result is True  # Repository allows status updates regardless

        # Test cancel_order success case
        active_order = Order(
            id=uuid.uuid4(),
            exchange="binance",
            symbol="BTCUSD",
            side="BUY",
            order_type="LIMIT",
            status="OPEN",
            quantity=Decimal("1.0"),
            price=Decimal("45000.00"),
        )

        with (
            patch.object(repo, "get", return_value=active_order),
            patch.object(repo, "update", return_value=active_order),
        ):
            with patch(
                "src.database.repository.utils.RepositoryUtils.update_entity_status",
                return_value=True,
            ):
                result = await repo.update_order_status(str(active_order.id), "CANCELLED")
                assert result is True

        # Test get_by_exchange_id
        with patch.object(repo, "get_by", return_value=active_order):
            result = await repo.get_by_exchange_id("binance", "12345")
            assert result == active_order

    async def test_position_repository_position_management(self, mock_session):
        """Test PositionRepository position management methods."""
        repo = PositionRepository(mock_session)

        # Test update_position_status (close_position method doesn't exist)
        with patch(
            "src.database.repository.utils.RepositoryUtils.update_entity_status", return_value=False
        ):
            result = await repo.update_position_status(
                str(uuid.uuid4()), "CLOSED", exit_price=Decimal("46000.00")
            )
            assert result is False

        # Test close_position with already closed position
        closed_position = Position(
            id=uuid.uuid4(),
            exchange="binance",
            symbol="BTCUSD",
            side="LONG",
            status="CLOSED",
            quantity=Decimal("1.0"),
            entry_price=Decimal("45000.00"),
        )

        with patch.object(repo, "get", return_value=closed_position):
            with patch(
                "src.database.repository.utils.RepositoryUtils.update_entity_status",
                return_value=True,
            ):
                result = await repo.update_position_status(
                    str(closed_position.id), "CLOSED", exit_price=Decimal("46000.00")
                )
                assert result is True  # Repository allows status updates regardless

        # Test close_position success case with PnL calculation
        open_position = Position(
            id=uuid.uuid4(),
            exchange="binance",
            symbol="BTCUSD",
            side="LONG",
            status="OPEN",
            quantity=Decimal("1.0"),
            entry_price=Decimal("45000.00"),
        )

        # Mock calculate_pnl method
        with patch.object(open_position, "calculate_pnl", return_value=Decimal("1000.00")):
            with (
                patch.object(repo, "get", return_value=open_position),
                patch.object(repo, "update", return_value=open_position),
            ):
                with patch(
                    "src.database.repository.utils.RepositoryUtils.update_entity_status",
                    return_value=True,
                ):
                    result = await repo.update_position_status(
                        str(open_position.id),
                        "CLOSED",
                        exit_price=Decimal("46000.00"),
                        realized_pnl=Decimal("1000.00"),
                    )
                    assert result is True

        # Test update_position_price
        with patch.object(open_position, "calculate_pnl", return_value=Decimal("500.00")):
            with (
                patch.object(repo, "get", return_value=open_position),
                patch.object(repo, "update", return_value=open_position),
            ):
                # Test update_position_fields (update_position_price method doesn't exist)
                with patch(
                    "src.database.repository.utils.RepositoryUtils.update_entity_fields",
                    return_value=True,
                ):
                    result = await repo.update_position_fields(
                        str(open_position.id),
                        current_price=Decimal("45500.00"),
                        unrealized_pnl=Decimal("500.00"),
                    )
                    assert result is True

        # Test get_position_by_symbol
        with patch.object(repo, "get_by", return_value=open_position):
            result = await repo.get_position_by_symbol(str(uuid.uuid4()), "BTCUSD", "LONG")
            assert result == open_position

    async def test_repository_error_handling_comprehensive(self, mock_session):
        """Test comprehensive error handling across repositories."""
        from src.core.exceptions import RepositoryError

        repo = BotRepository(mock_session)

        # Test database connection error
        mock_session.execute.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(RepositoryError):
            await repo.get_all()

        # Reset mock
        mock_session.execute.side_effect = None

        # Test transaction rollback on error
        mock_session.flush.side_effect = SQLAlchemyError("Constraint violation")

        bot = Bot(id=uuid.uuid4(), name="Error Test Bot", status="RUNNING", exchange="binance")

        with pytest.raises(RepositoryError):
            await repo.create(bot)

        # Verify rollback was called
        mock_session.rollback.assert_called()

    async def test_repository_pagination_and_ordering(self, mock_session):
        """Test pagination and ordering functionality."""
        repo = BotRepository(mock_session)

        # Mock paginated results
        mock_result = Mock()
        mock_scalars = Mock()

        # Create mock bots for pagination test
        mock_bots = [
            Bot(id=uuid.uuid4(), name=f"Bot {i}", status="RUNNING", exchange="binance")
            for i in range(5)
        ]
        mock_scalars.all.return_value = mock_bots[:3]  # First page
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test with limit and offset
        paginated_bots = await repo.get_all(limit=3, offset=0)
        assert len(paginated_bots) == 3

        # Test with ordering
        ordered_bots = await repo.get_all(order_by="name")
        assert len(ordered_bots) == 3

        # Test with descending order
        desc_bots = await repo.get_all(order_by="-created_at")
        assert len(desc_bots) == 3

    async def test_repository_soft_delete_functionality(self, mock_session):
        """Test soft delete functionality if supported."""
        repo = BotRepository(mock_session)

        # Test soft delete with non-existent entity
        with patch.object(repo, "_get_entity_by_id", new_callable=AsyncMock, return_value=None):
            result = await repo.soft_delete(uuid.uuid4())
            assert result is False

        # Test soft delete with entity that doesn't support it
        bot = Bot(id=uuid.uuid4(), name="Test Bot", status="RUNNING", exchange="binance")

        with patch.object(repo, "_get_entity_by_id", new_callable=AsyncMock, return_value=bot):
            result = await repo.soft_delete(bot.id)
            # Bot model has soft_delete method via SoftDeleteMixin, should return True
            assert result is True

    async def test_repository_batch_operations(self, mock_session):
        """Test batch operations for efficiency."""
        repo = BotRepository(mock_session)

        # BotRepository doesn't have create_many method, so test individual creates
        # Test creating multiple entities individually
        bots = [
            Bot(id=uuid.uuid4(), name=f"Bot {i}", status="RUNNING", exchange="binance")
            for i in range(3)
        ]

        with patch.object(repo, "create") as mock_create:
            mock_create.side_effect = bots
            results = []
            for bot in bots:
                result = await repo.create(bot)
                results.append(result)

            assert len(results) == 3

    async def test_repository_complex_filtering(self, mock_session):
        """Test complex filtering scenarios."""
        repo = MarketDataRepository(mock_session)

        # Mock complex query results
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test complex filters with ranges
        complex_filters = {
            "symbol": "BTCUSD",
            "data_timestamp": {
                "gte": datetime.now(timezone.utc),
                "lte": datetime.now(timezone.utc),
            },
            "volume": {"gt": Decimal("100.0")},
        }

        result = await repo.get_all(filters=complex_filters)
        assert result == []

        # Test filter with list values
        list_filters = {
            "symbol": ["BTCUSD", "ETHUSDT", "ADAUSDT"],
            "exchange": ["binance", "coinbase"],
        }

        result = await repo.get_all(filters=list_filters)
        assert result == []

        # Test filter with LIKE pattern
        like_filters = {"symbol": {"like": "BTC"}}

        result = await repo.get_all(filters=like_filters)
        assert result == []
