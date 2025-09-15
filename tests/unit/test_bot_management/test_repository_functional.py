"""Functional tests for bot management repository classes."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock

from src.bot_management.repository import BotRepository, BotInstanceRepository, BotMetricsRepository
from src.core.exceptions import DatabaseError, EntityNotFoundError
from src.core.types import BotStatus, BotMetrics
from src.database.models.bot import Bot
from src.database.models.bot_instance import BotInstance


class TestBotRepositoryFunctional:
    """Functional tests for BotRepository."""

    def test_repository_initialization_legacy(self):
        """Test repository initialization with legacy interface."""
        # Create a mock that doesn't have execute attribute
        mock_service = Mock()
        if hasattr(mock_service, 'execute'):
            delattr(mock_service, 'execute')

        repo = BotRepository(mock_service)

        assert repo.db_service == mock_service
        assert repo.session == mock_service

    @pytest.mark.asyncio
    async def test_create_bot_configuration_success(self):
        """Test create bot configuration success."""
        mock_service = AsyncMock()
        mock_service.execute.return_value = True

        repo = BotRepository(mock_service)
        result = await repo.create_bot_configuration({"bot_id": "test"})

        assert result is True

    @pytest.mark.asyncio
    async def test_create_bot_configuration_failure(self):
        """Test create bot configuration failure."""
        mock_service = AsyncMock()
        mock_service.execute.side_effect = Exception("Database error")

        repo = BotRepository(mock_service)
        result = await repo.create_bot_configuration({"bot_id": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_get_bot_configuration_success(self):
        """Test get bot configuration success."""
        mock_service = AsyncMock()
        mock_result = Mock()
        mock_result.first.return_value = {"bot_id": "test", "name": "Test Bot"}
        mock_service.execute.return_value = mock_result

        repo = BotRepository(mock_service)
        result = await repo.get_bot_configuration("test")

        assert result == {"bot_id": "test", "name": "Test Bot"}

    @pytest.mark.asyncio
    async def test_get_bot_configuration_not_found(self):
        """Test get bot configuration not found."""
        mock_service = AsyncMock()
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_service.execute.return_value = mock_result

        repo = BotRepository(mock_service)
        result = await repo.get_bot_configuration("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_bot_configuration_no_first_method(self):
        """Test get bot configuration when result has no first method."""
        mock_service = AsyncMock()
        mock_result = Mock(spec=[])  # Empty spec means no methods
        mock_service.execute.return_value = mock_result

        repo = BotRepository(mock_service)
        result = await repo.get_bot_configuration("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_bot_configuration_error(self):
        """Test get bot configuration with database error."""
        mock_service = AsyncMock()
        mock_service.execute.side_effect = Exception("Database connection lost")

        repo = BotRepository(mock_service)
        result = await repo.get_bot_configuration("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_bot_configuration_success(self):
        """Test update bot configuration success."""
        mock_service = AsyncMock()
        mock_service.execute.return_value = True

        repo = BotRepository(mock_service)
        result = await repo.update_bot_configuration({"bot_id": "test", "name": "Updated"})

        assert result is True

    @pytest.mark.asyncio
    async def test_update_bot_configuration_failure(self):
        """Test update bot configuration failure."""
        mock_service = AsyncMock()
        mock_service.execute.side_effect = Exception("Update failed")

        repo = BotRepository(mock_service)
        result = await repo.update_bot_configuration({"bot_id": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_bot_configuration_success(self):
        """Test delete bot configuration success."""
        mock_service = AsyncMock()
        mock_service.execute.return_value = True

        repo = BotRepository(mock_service)
        result = await repo.delete_bot_configuration("test")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_bot_configuration_failure(self):
        """Test delete bot configuration failure."""
        mock_service = AsyncMock()
        mock_service.execute.side_effect = Exception("Delete failed")

        repo = BotRepository(mock_service)
        result = await repo.delete_bot_configuration("test")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_bot_configurations_success(self):
        """Test list bot configurations success."""
        mock_service = AsyncMock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [{"bot_id": "test1"}, {"bot_id": "test2"}]
        mock_service.execute.return_value = mock_result

        repo = BotRepository(mock_service)
        result = await repo.list_bot_configurations()

        assert result == [{"bot_id": "test1"}, {"bot_id": "test2"}]

    @pytest.mark.asyncio
    async def test_list_bot_configurations_no_fetchall(self):
        """Test list bot configurations when result has no fetchall method."""
        mock_service = AsyncMock()
        mock_result = Mock(spec=[])  # Empty spec means no methods
        mock_service.execute.return_value = mock_result

        repo = BotRepository(mock_service)
        result = await repo.list_bot_configurations()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_bot_configurations_error(self):
        """Test list bot configurations with error."""
        mock_service = AsyncMock()
        mock_service.execute.side_effect = Exception("Query failed")

        repo = BotRepository(mock_service)
        result = await repo.list_bot_configurations()

        assert result == []

    @pytest.mark.asyncio
    async def test_store_bot_metrics_success(self):
        """Test store bot metrics success."""
        mock_service = AsyncMock()
        mock_service.execute.return_value = True

        repo = BotRepository(mock_service)
        result = await repo.store_bot_metrics({"bot_id": "test", "total_trades": 10})

        assert result is True

    @pytest.mark.asyncio
    async def test_store_bot_metrics_failure(self):
        """Test store bot metrics failure."""
        mock_service = AsyncMock()
        mock_service.execute.side_effect = Exception("Insert failed")

        repo = BotRepository(mock_service)
        result = await repo.store_bot_metrics({"bot_id": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_get_bot_metrics_success(self):
        """Test get bot metrics success."""
        mock_service = AsyncMock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [{"metric1": "value1"}]
        mock_service.execute.return_value = mock_result

        repo = BotRepository(mock_service)
        result = await repo.get_bot_metrics("test")

        assert result == [{"metric1": "value1"}]

    @pytest.mark.asyncio
    async def test_get_bot_metrics_no_fetchall(self):
        """Test get bot metrics when result has no fetchall method."""
        mock_service = AsyncMock()
        mock_result = Mock(spec=[])
        mock_service.execute.return_value = mock_result

        repo = BotRepository(mock_service)
        result = await repo.get_bot_metrics("test")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_bot_metrics_error(self):
        """Test get bot metrics with error."""
        mock_service = AsyncMock()
        mock_service.execute.side_effect = Exception("Query failed")

        repo = BotRepository(mock_service)
        result = await repo.get_bot_metrics("test")

        assert result == []

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test health check success."""
        mock_service = AsyncMock()
        mock_service.execute.return_value = True

        repo = BotRepository(mock_service)
        result = await repo.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        mock_service = AsyncMock()
        mock_service.execute.side_effect = Exception("Connection failed")

        repo = BotRepository(mock_service)
        result = await repo.health_check()

        assert result is False


class TestBotInstanceRepositoryFunctional:
    """Functional tests for BotInstanceRepository."""

    def test_repository_initialization(self):
        """Test repository initialization."""
        mock_session = AsyncMock()
        repo = BotInstanceRepository(mock_session)

        assert repo.session == mock_session
        assert repo.model == BotInstance

    @pytest.mark.asyncio
    async def test_update_metrics_without_total_pnl(self):
        """Test update metrics when BotMetrics has no total_pnl attribute."""
        mock_session = AsyncMock()
        mock_instance = BotInstance()
        mock_instance.id = "test-instance"

        repo = BotInstanceRepository(mock_session)

        # Mock the get method to return the instance
        async def mock_get(instance_id):
            return mock_instance

        repo.get = mock_get

        # Create metrics without total_pnl attribute
        mock_metrics = Mock()
        mock_metrics.total_trades = 10
        mock_metrics.profitable_trades = 6
        mock_metrics.losing_trades = 4
        # Ensure total_pnl attribute doesn't exist
        if hasattr(mock_metrics, 'total_pnl'):
            delattr(mock_metrics, 'total_pnl')

        result = await repo.update_metrics("test-instance", mock_metrics)

        assert result.total_pnl == 0.0  # Should default to 0.0
        assert result.total_trades == 10
        assert result.profitable_trades == 6

    @pytest.mark.asyncio
    async def test_get_performance_stats_with_null_aggregations(self):
        """Test get performance stats when aggregations return None."""
        mock_session = AsyncMock()
        mock_result = Mock()

        # Create a mock row with None values
        mock_row = Mock()
        mock_row.total_instances = None
        mock_row.total_trades = None
        mock_row.profitable_trades = None
        mock_row.total_pnl = None
        mock_row.avg_pnl = None

        mock_result.first.return_value = mock_row
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_performance_stats("test-bot")

        # Should handle None values gracefully
        assert result["total_instances"] == 0
        assert result["total_trades"] == 0
        assert result["profitable_trades"] == 0
        assert result["total_pnl"] == Decimal("0")
        assert result["avg_pnl"] == Decimal("0")


class TestBotMetricsRepositoryFunctional:
    """Functional tests for BotMetricsRepository."""

    def test_repository_initialization(self):
        """Test repository initialization."""
        mock_session = AsyncMock()
        repo = BotMetricsRepository(mock_session)

        assert repo.session == mock_session
        assert repo.model == BotInstance

    @pytest.mark.asyncio
    async def test_save_metrics_instance_not_found(self):
        """Test save metrics when no bot instance found."""
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = BotMetricsRepository(mock_session)

        mock_metrics = BotMetrics(
            bot_id="nonexistent-bot",
            created_at=datetime.utcnow(),
            total_trades=10,
            profitable_trades=6,
            losing_trades=4
        )

        # Should not raise exception, just log warning
        await repo.save_metrics(mock_metrics)

        # Commit should not be called since no instance was found
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_latest_metrics_instance_found(self):
        """Test get latest metrics when instance is found."""
        mock_session = AsyncMock()
        mock_result = Mock()

        mock_instance = BotInstance()
        mock_instance.bot_id = "test-bot"
        mock_instance.total_trades = 15
        mock_instance.profitable_trades = 9
        mock_instance.losing_trades = 6
        mock_instance.updated_at = datetime.utcnow()

        mock_result.scalar_one_or_none.return_value = mock_instance
        mock_session.execute.return_value = mock_result

        repo = BotMetricsRepository(mock_session)
        result = await repo.get_latest_metrics("test-bot")

        assert result is not None
        assert result.bot_id == "test-bot"
        assert result.total_trades == 15
        assert result.profitable_trades == 9
        assert result.losing_trades == 6

    @pytest.mark.asyncio
    async def test_get_latest_metrics_instance_not_found(self):
        """Test get latest metrics when no instance found."""
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = BotMetricsRepository(mock_session)
        result = await repo.get_latest_metrics("nonexistent-bot")

        assert result is None


class TestRepositoryConnectionHandling:
    """Test connection handling edge cases."""

    @pytest.mark.asyncio
    async def test_connection_close_without_close_method(self):
        """Test connection close when connection has no close method."""
        mock_service = AsyncMock()
        mock_connection = Mock(spec=[])  # No close method
        mock_service.execute.return_value = mock_connection

        repo = BotRepository(mock_service)
        # Should not raise exception even if connection has no close method
        result = await repo.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_result_close_without_close_method(self):
        """Test result close when result has no close method."""
        mock_service = AsyncMock()
        mock_result = Mock(spec=['first'])  # Has first but no close method
        mock_result.first.return_value = {"data": "test"}
        mock_service.execute.return_value = mock_result

        repo = BotRepository(mock_service)
        result = await repo.get_bot_configuration("test")

        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_session_close_without_close_method(self):
        """Test session close when session has no close method."""
        mock_session = Mock(spec=['commit', 'rollback'])  # No close method
        mock_session.commit.side_effect = Exception("Commit failed")

        repo = BotRepository(mock_session)

        # Mock the get method
        async def mock_get(bot_id):
            return None
        repo.get = mock_get

        with pytest.raises(DatabaseError, match="Failed to update bot status.*Bot .* not found"):
            await repo.update_status("test-bot", BotStatus.STOPPED)