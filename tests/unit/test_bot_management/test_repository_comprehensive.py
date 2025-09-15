"""Comprehensive tests for bot management repository classes."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from src.bot_management.repository import BotRepository, BotInstanceRepository, BotMetricsRepository
from src.core.exceptions import DatabaseError, EntityNotFoundError
from src.core.types import BotStatus, BotMetrics
from src.database.models.bot import Bot
from src.database.models.bot_instance import BotInstance


@pytest.fixture
def mock_session():
    """Create mock async session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.scalar_one_or_none = AsyncMock()
    session.scalars = AsyncMock()
    return session


@pytest.fixture
def mock_db_service():
    """Create mock db service for legacy interface."""
    service = AsyncMock()
    service.execute = AsyncMock()
    return service


@pytest.fixture
def sample_bot():
    """Create sample bot entity."""
    bot = Bot()
    bot.id = "test-bot-123"
    bot.name = "Test Bot"
    bot.status = "running"
    bot.created_at = datetime.utcnow()
    bot.updated_at = datetime.utcnow()
    return bot


@pytest.fixture
def sample_bot_instance():
    """Create sample bot instance."""
    instance = BotInstance()
    instance.id = "instance-123"
    instance.bot_id = "test-bot-123"
    instance.status = "running"
    instance.total_trades = 10
    instance.profitable_trades = 6
    instance.losing_trades = 4
    instance.total_pnl = 150.50
    instance.created_at = datetime.utcnow()
    instance.updated_at = datetime.utcnow()
    return instance


@pytest.fixture
def sample_bot_metrics():
    """Create sample bot metrics."""
    return BotMetrics(
        bot_id="test-bot-123",
        created_at=datetime.utcnow(),
        total_trades=15,
        profitable_trades=9,
        losing_trades=6,
        total_pnl=Decimal("200.75"),
        win_rate=0.60,
        avg_trade_pnl=Decimal("13.38")
    )


class TestBotRepository:
    """Test cases for BotRepository."""

    def test_init_with_async_session(self, mock_session):
        """Test initialization with AsyncSession."""
        repo = BotRepository(mock_session)
        assert repo.db_service == mock_session
        assert hasattr(repo, 'session')

    def test_init_with_legacy_service(self, mock_db_service):
        """Test initialization with legacy db service."""
        repo = BotRepository(mock_db_service)
        assert repo.db_service == mock_db_service
        assert repo.session == mock_db_service

    @pytest.mark.asyncio
    async def test_get_by_name_success(self, mock_session, sample_bot):
        """Test successful get by name."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_bot
        mock_session.execute.return_value = mock_result

        repo = BotRepository(mock_session)
        result = await repo.get_by_name("Test Bot")

        assert result == sample_bot
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_name_not_found(self, mock_session):
        """Test get by name when bot not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = BotRepository(mock_session)
        result = await repo.get_by_name("Nonexistent Bot")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_name_database_error(self, mock_session):
        """Test get by name with database error."""
        mock_session.execute.side_effect = Exception("Database connection failed")

        repo = BotRepository(mock_session)
        with pytest.raises(DatabaseError, match="Failed to get bot by name"):
            await repo.get_by_name("Test Bot")

    @pytest.mark.asyncio
    async def test_get_active_bots_success(self, mock_session, sample_bot):
        """Test successful get active bots."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [sample_bot]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = BotRepository(mock_session)
        result = await repo.get_active_bots()

        assert result == [sample_bot]
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_bots_empty(self, mock_session):
        """Test get active bots when none exist."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = BotRepository(mock_session)
        result = await repo.get_active_bots()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_active_bots_database_error(self, mock_session):
        """Test get active bots with database error."""
        mock_session.execute.side_effect = Exception("Connection timeout")

        repo = BotRepository(mock_session)
        with pytest.raises(DatabaseError, match="Failed to get active bots"):
            await repo.get_active_bots()

    @pytest.mark.asyncio
    async def test_update_status_success(self, mock_session, sample_bot):
        """Test successful status update."""
        repo = BotRepository(mock_session)
        repo.get = AsyncMock(return_value=sample_bot)

        result = await repo.update_status("test-bot-123", BotStatus.STOPPED)

        assert result.status == BotStatus.STOPPED.value
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_bot_not_found(self, mock_session):
        """Test status update when bot not found."""
        repo = BotRepository(mock_session)
        with patch.object(repo, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            with pytest.raises(DatabaseError, match="Failed to update bot status.*Bot .* not found"):
                await repo.update_status("nonexistent-bot", BotStatus.STOPPED)

    @pytest.mark.asyncio
    async def test_update_status_database_error(self, mock_session, sample_bot):
        """Test status update with database error."""
        repo = BotRepository(mock_session)
        repo.get = AsyncMock(return_value=sample_bot)
        mock_session.commit.side_effect = Exception("Commit failed")

        with pytest.raises(DatabaseError, match="Failed to update bot status"):
            await repo.update_status("test-bot-123", BotStatus.STOPPED)

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_bot_configuration_success(self, mock_db_service):
        """Test successful bot configuration creation."""
        repo = BotRepository(mock_db_service)

        result = await repo.create_bot_configuration({"bot_id": "test", "name": "Test"})

        assert result is True
        mock_db_service.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_bot_configuration_error(self, mock_db_service):
        """Test bot configuration creation error."""
        mock_db_service.execute.side_effect = Exception("Insert failed")
        repo = BotRepository(mock_db_service)

        result = await repo.create_bot_configuration({"bot_id": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_get_bot_configuration_success(self, mock_db_service):
        """Test successful get bot configuration."""
        mock_result = Mock()
        mock_result.first.return_value = {"bot_id": "test", "name": "Test"}
        mock_db_service.execute.return_value = mock_result

        repo = BotRepository(mock_db_service)
        result = await repo.get_bot_configuration("test")

        assert result == {"bot_id": "test", "name": "Test"}

    @pytest.mark.asyncio
    async def test_get_bot_configuration_not_found(self, mock_db_service):
        """Test get bot configuration when not found."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_db_service.execute.return_value = mock_result

        repo = BotRepository(mock_db_service)
        result = await repo.get_bot_configuration("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_bot_configuration_error(self, mock_db_service):
        """Test get bot configuration with error."""
        mock_db_service.execute.side_effect = Exception("Query failed")
        repo = BotRepository(mock_db_service)

        result = await repo.get_bot_configuration("test")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_bot_configuration_success(self, mock_db_service):
        """Test successful bot configuration update."""
        repo = BotRepository(mock_db_service)

        result = await repo.update_bot_configuration({"bot_id": "test", "name": "Updated"})

        assert result is True
        mock_db_service.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_bot_configuration_error(self, mock_db_service):
        """Test bot configuration update error."""
        mock_db_service.execute.side_effect = Exception("Update failed")
        repo = BotRepository(mock_db_service)

        result = await repo.update_bot_configuration({"bot_id": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_bot_configuration_success(self, mock_db_service):
        """Test successful bot configuration deletion."""
        repo = BotRepository(mock_db_service)

        result = await repo.delete_bot_configuration("test")

        assert result is True
        mock_db_service.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_bot_configuration_error(self, mock_db_service):
        """Test bot configuration deletion error."""
        mock_db_service.execute.side_effect = Exception("Delete failed")
        repo = BotRepository(mock_db_service)

        result = await repo.delete_bot_configuration("test")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_bot_configurations_success(self, mock_db_service):
        """Test successful list bot configurations."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [{"bot_id": "test1"}, {"bot_id": "test2"}]
        mock_db_service.execute.return_value = mock_result

        repo = BotRepository(mock_db_service)
        result = await repo.list_bot_configurations()

        assert result == [{"bot_id": "test1"}, {"bot_id": "test2"}]

    @pytest.mark.asyncio
    async def test_list_bot_configurations_empty(self, mock_db_service):
        """Test list bot configurations when empty."""
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_db_service.execute.return_value = mock_result

        repo = BotRepository(mock_db_service)
        result = await repo.list_bot_configurations()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_bot_configurations_error(self, mock_db_service):
        """Test list bot configurations with error."""
        mock_db_service.execute.side_effect = Exception("Query failed")
        repo = BotRepository(mock_db_service)

        result = await repo.list_bot_configurations()

        assert result == []

    @pytest.mark.asyncio
    async def test_store_bot_metrics_success(self, mock_db_service):
        """Test successful store bot metrics."""
        repo = BotRepository(mock_db_service)
        metrics = {"bot_id": "test", "total_trades": 10}

        result = await repo.store_bot_metrics(metrics)

        assert result is True
        mock_db_service.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_bot_metrics_error(self, mock_db_service):
        """Test store bot metrics error."""
        mock_db_service.execute.side_effect = Exception("Insert failed")
        repo = BotRepository(mock_db_service)

        result = await repo.store_bot_metrics({"bot_id": "test"})

        assert result is False

    @pytest.mark.asyncio
    async def test_get_bot_metrics_success(self, mock_db_service):
        """Test successful get bot metrics."""
        mock_result = Mock()
        mock_result.fetchall.return_value = [{"metric1": "value1"}]
        mock_db_service.execute.return_value = mock_result

        repo = BotRepository(mock_db_service)
        result = await repo.get_bot_metrics("test")

        assert result == [{"metric1": "value1"}]

    @pytest.mark.asyncio
    async def test_get_bot_metrics_error(self, mock_db_service):
        """Test get bot metrics error."""
        mock_db_service.execute.side_effect = Exception("Query failed")
        repo = BotRepository(mock_db_service)

        result = await repo.get_bot_metrics("test")

        assert result == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_db_service):
        """Test successful health check."""
        repo = BotRepository(mock_db_service)

        result = await repo.health_check()

        assert result is True
        mock_db_service.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_db_service):
        """Test health check failure."""
        mock_db_service.execute.side_effect = Exception("Connection failed")
        repo = BotRepository(mock_db_service)

        result = await repo.health_check()

        assert result is False


class TestBotInstanceRepository:
    """Test cases for BotInstanceRepository."""

    def test_init(self, mock_session):
        """Test initialization."""
        repo = BotInstanceRepository(mock_session)
        assert repo.session == mock_session
        assert repo.model == BotInstance

    @pytest.mark.asyncio
    async def test_get_by_bot_id_success(self, mock_session, sample_bot_instance):
        """Test successful get by bot id."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [sample_bot_instance]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_by_bot_id("test-bot-123")

        assert result == [sample_bot_instance]

    @pytest.mark.asyncio
    async def test_get_by_bot_id_empty(self, mock_session):
        """Test get by bot id when empty."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_by_bot_id("test-bot-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_by_bot_id_error(self, mock_session):
        """Test get by bot id with error."""
        mock_session.execute.side_effect = Exception("Query failed")
        repo = BotInstanceRepository(mock_session)

        with pytest.raises(DatabaseError, match="Failed to get bot instances"):
            await repo.get_by_bot_id("test-bot-123")

    @pytest.mark.asyncio
    async def test_get_active_instance_success(self, mock_session, sample_bot_instance):
        """Test successful get active instance."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_bot_instance
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_active_instance("test-bot-123")

        assert result == sample_bot_instance

    @pytest.mark.asyncio
    async def test_get_active_instance_not_found(self, mock_session):
        """Test get active instance when not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_active_instance("test-bot-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_active_instance_error(self, mock_session):
        """Test get active instance with error."""
        mock_session.execute.side_effect = Exception("Query failed")
        repo = BotInstanceRepository(mock_session)

        with pytest.raises(DatabaseError, match="Failed to get active instance"):
            await repo.get_active_instance("test-bot-123")

    @pytest.mark.asyncio
    async def test_get_active_instances_success(self, mock_session, sample_bot_instance):
        """Test successful get active instances."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [sample_bot_instance]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_active_instances()

        assert result == [sample_bot_instance]

    @pytest.mark.asyncio
    async def test_get_active_instances_error(self, mock_session):
        """Test get active instances with error."""
        mock_session.execute.side_effect = Exception("Query failed")
        repo = BotInstanceRepository(mock_session)

        with pytest.raises(DatabaseError, match="Failed to get active instances"):
            await repo.get_active_instances()

    @pytest.mark.asyncio
    async def test_update_metrics_success(self, mock_session, sample_bot_instance, sample_bot_metrics):
        """Test successful metrics update."""
        repo = BotInstanceRepository(mock_session)
        repo.get = AsyncMock(return_value=sample_bot_instance)

        result = await repo.update_metrics("instance-123", sample_bot_metrics)

        assert result.total_trades == sample_bot_metrics.total_trades
        assert result.profitable_trades == sample_bot_metrics.profitable_trades
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_metrics_instance_not_found(self, mock_session, sample_bot_metrics):
        """Test metrics update when instance not found."""
        repo = BotInstanceRepository(mock_session)
        with patch.object(repo, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            with pytest.raises(DatabaseError, match="Failed to update instance metrics.*Bot instance .* not found"):
                await repo.update_metrics("nonexistent", sample_bot_metrics)

    @pytest.mark.asyncio
    async def test_update_metrics_database_error(self, mock_session, sample_bot_instance, sample_bot_metrics):
        """Test metrics update with database error."""
        repo = BotInstanceRepository(mock_session)
        repo.get = AsyncMock(return_value=sample_bot_instance)
        mock_session.commit.side_effect = Exception("Commit failed")

        with pytest.raises(DatabaseError, match="Failed to update instance metrics"):
            await repo.update_metrics("instance-123", sample_bot_metrics)

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_performance_stats_success(self, mock_session):
        """Test successful get performance stats."""
        mock_result = Mock()
        mock_row = MagicMock()
        mock_row.total_instances = 5
        mock_row.total_trades = 100
        mock_row.profitable_trades = 60
        mock_row.total_pnl = 1000.0
        mock_row.avg_pnl = 200.0
        mock_result.first.return_value = mock_row
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_performance_stats("test-bot-123")

        assert result["total_instances"] == 5
        assert result["total_trades"] == 100
        assert result["profitable_trades"] == 60
        assert result["total_pnl"] == Decimal("1000.0")

    @pytest.mark.asyncio
    async def test_get_performance_stats_no_data(self, mock_session):
        """Test get performance stats when no data."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_performance_stats("test-bot-123")

        assert result["total_instances"] == 0
        assert result["total_trades"] == 0
        assert result["total_pnl"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_performance_stats_error(self, mock_session):
        """Test get performance stats with error."""
        mock_session.execute.side_effect = Exception("Query failed")
        repo = BotInstanceRepository(mock_session)

        with pytest.raises(DatabaseError, match="Failed to get performance stats"):
            await repo.get_performance_stats("test-bot-123")


class TestBotMetricsRepository:
    """Test cases for BotMetricsRepository."""

    def test_init(self, mock_session):
        """Test initialization."""
        repo = BotMetricsRepository(mock_session)
        assert repo.session == mock_session
        assert repo.model == BotInstance

    @pytest.mark.asyncio
    async def test_save_metrics_success(self, mock_session, sample_bot_instance, sample_bot_metrics):
        """Test successful save metrics."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_bot_instance
        mock_session.execute.return_value = mock_result

        repo = BotMetricsRepository(mock_session)
        await repo.save_metrics(sample_bot_metrics)

        assert sample_bot_instance.total_trades == sample_bot_metrics.total_trades
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_metrics_no_instance(self, mock_session, sample_bot_metrics):
        """Test save metrics when no instance found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = BotMetricsRepository(mock_session)
        # Should not raise exception, just log warning
        await repo.save_metrics(sample_bot_metrics)

        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_metrics_database_error(self, mock_session, sample_bot_instance, sample_bot_metrics):
        """Test save metrics with database error."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_bot_instance
        mock_session.execute.return_value = mock_result
        mock_session.commit.side_effect = Exception("Commit failed")

        repo = BotMetricsRepository(mock_session)

        with pytest.raises(DatabaseError, match="Failed to save metrics"):
            await repo.save_metrics(sample_bot_metrics)

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_latest_metrics_success(self, mock_session, sample_bot_instance):
        """Test successful get latest metrics."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_bot_instance
        mock_session.execute.return_value = mock_result

        repo = BotMetricsRepository(mock_session)
        result = await repo.get_latest_metrics("test-bot-123")

        assert result is not None
        assert result.bot_id == "test-bot-123"
        assert result.total_trades == sample_bot_instance.total_trades

    @pytest.mark.asyncio
    async def test_get_latest_metrics_not_found(self, mock_session):
        """Test get latest metrics when not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = BotMetricsRepository(mock_session)
        result = await repo.get_latest_metrics("test-bot-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_metrics_database_error(self, mock_session):
        """Test get latest metrics with database error."""
        mock_session.execute.side_effect = Exception("Query failed")
        repo = BotMetricsRepository(mock_session)

        with pytest.raises(DatabaseError, match="Failed to get latest metrics"):
            await repo.get_latest_metrics("test-bot-123")