"""Coverage boost tests for bot management repository classes."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.bot_management.repository import BotRepository, BotInstanceRepository, BotMetricsRepository
from src.core.exceptions import DatabaseError, EntityNotFoundError
from src.core.types import BotStatus, BotMetrics
from src.database.models.bot import Bot
from src.database.models.bot_instance import BotInstance


class TestBotRepositoryCoverage:
    """Test cases to boost BotRepository coverage."""

    @pytest.mark.asyncio
    async def test_repository_session_close_handling(self):
        """Test repository session close handling in finally blocks."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Test error")
        mock_session.close.side_effect = Exception("Close error")

        repo = BotRepository(mock_session)

        # Test update_status finally block
        with patch.object(repo, 'get', AsyncMock(side_effect=Exception("Error"))):
            with pytest.raises(DatabaseError):
                await repo.update_status("test-bot", BotStatus.STOPPED)

    @pytest.mark.asyncio
    async def test_health_check_connection_close(self):
        """Test health check with connection close in finally block."""
        mock_db_service = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.close.side_effect = Exception("Close failed")
        mock_db_service.execute.return_value = mock_connection

        repo = BotRepository(mock_db_service)
        result = await repo.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_create_bot_configuration_connection_close(self):
        """Test create bot configuration with connection close error."""
        mock_db_service = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.close.side_effect = Exception("Close failed")
        mock_db_service.execute.return_value = mock_connection

        repo = BotRepository(mock_db_service)
        result = await repo.create_bot_configuration({"test": "data"})

        assert result is True

    @pytest.mark.asyncio
    async def test_get_bot_configuration_result_close(self):
        """Test get bot configuration with result close error."""
        mock_db_service = AsyncMock()
        mock_result = Mock()
        mock_result.first.return_value = {"data": "test"}
        mock_result.close.side_effect = Exception("Close failed")
        mock_db_service.execute.return_value = mock_result

        repo = BotRepository(mock_db_service)
        result = await repo.get_bot_configuration("test-bot")

        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_list_bot_configurations_result_missing_attrs(self):
        """Test list bot configurations when result doesn't have fetchall."""
        mock_db_service = AsyncMock()
        mock_result = Mock()
        # Remove fetchall attribute to trigger else path
        delattr(mock_result, 'fetchall') if hasattr(mock_result, 'fetchall') else None
        mock_db_service.execute.return_value = mock_result

        repo = BotRepository(mock_db_service)
        result = await repo.list_bot_configurations()

        assert result == []

    @pytest.mark.asyncio
    async def test_store_bot_metrics_with_connection_handling(self):
        """Test store bot metrics with connection handling."""
        mock_db_service = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.close.side_effect = Exception("Close failed")
        mock_db_service.execute.return_value = mock_connection

        repo = BotRepository(mock_db_service)
        result = await repo.store_bot_metrics({"bot_id": "test"})

        assert result is True

    @pytest.mark.asyncio
    async def test_get_bot_metrics_result_missing_attrs(self):
        """Test get bot metrics when result doesn't have fetchall."""
        mock_db_service = AsyncMock()
        mock_result = Mock()
        # Remove fetchall attribute to trigger else path
        delattr(mock_result, 'fetchall') if hasattr(mock_result, 'fetchall') else None
        mock_db_service.execute.return_value = mock_result

        repo = BotRepository(mock_db_service)
        result = await repo.get_bot_metrics("test-bot")

        assert result == []


class TestBotInstanceRepositoryCoverage:
    """Test cases to boost BotInstanceRepository coverage."""

    @pytest.mark.asyncio
    async def test_update_metrics_session_close_handling(self):
        """Test update metrics with session close handling in finally block."""
        mock_session = AsyncMock()
        mock_instance = BotInstance()
        mock_instance.id = "test-instance"
        mock_session.commit.side_effect = Exception("Commit error")
        mock_session.close.side_effect = Exception("Close error")

        repo = BotInstanceRepository(mock_session)

        with patch.object(repo, 'get', AsyncMock(return_value=mock_instance)):
            mock_metrics = BotMetrics(
                bot_id="test-bot",
                created_at=datetime.utcnow(),
                total_trades=10,
                profitable_trades=6,
                losing_trades=4
            )

            with pytest.raises(DatabaseError):
                await repo.update_metrics("test-instance", mock_metrics)

    @pytest.mark.asyncio
    async def test_update_metrics_metrics_without_total_pnl(self):
        """Test update metrics when metrics doesn't have total_pnl."""
        mock_session = AsyncMock()
        mock_instance = BotInstance()
        mock_instance.id = "test-instance"

        repo = BotInstanceRepository(mock_session)

        with patch.object(repo, 'get', AsyncMock(return_value=mock_instance)):
            # Create metrics without total_pnl attribute
            mock_metrics = MagicMock()
            mock_metrics.total_trades = 10
            mock_metrics.profitable_trades = 6
            mock_metrics.losing_trades = 4
            # Explicitly remove total_pnl to test else path
            mock_metrics.total_pnl = None
            delattr(mock_metrics, 'total_pnl')

            result = await repo.update_metrics("test-instance", mock_metrics)

            assert result.total_pnl == 0.0

    @pytest.mark.asyncio
    async def test_get_performance_stats_with_null_values(self):
        """Test get performance stats when database returns null values."""
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_row = MagicMock()
        # Set all values to None to test fallback logic
        mock_row.total_instances = None
        mock_row.total_trades = None
        mock_row.profitable_trades = None
        mock_row.total_pnl = None
        mock_row.avg_pnl = None
        mock_result.first.return_value = mock_row
        mock_session.execute.return_value = mock_result

        repo = BotInstanceRepository(mock_session)
        result = await repo.get_performance_stats("test-bot")

        # Should handle None values with fallbacks
        assert result["total_instances"] == 0
        assert result["total_trades"] == 0
        assert result["profitable_trades"] == 0
        assert result["total_pnl"] == Decimal("0")
        assert result["avg_pnl"] == Decimal("0")


class TestBotMetricsRepositoryCoverage:
    """Test cases to boost BotMetricsRepository coverage."""

    @pytest.mark.asyncio
    async def test_save_metrics_session_close_handling(self):
        """Test save metrics with session close handling in finally block."""
        mock_session = AsyncMock()
        mock_instance = BotInstance()
        mock_session.commit.side_effect = Exception("Commit error")
        mock_session.close.side_effect = Exception("Close error")

        repo = BotMetricsRepository(mock_session)

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_instance
        mock_session.execute.return_value = mock_result

        mock_metrics = BotMetrics(
            bot_id="test-bot",
            created_at=datetime.utcnow(),
            total_trades=10,
            profitable_trades=6,
            losing_trades=4
        )

        with pytest.raises(DatabaseError):
            await repo.save_metrics(mock_metrics)

    @pytest.mark.asyncio
    async def test_get_latest_metrics_session_close_handling(self):
        """Test get latest metrics with session close handling in finally block."""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Query error")
        mock_session.close.side_effect = Exception("Close error")

        repo = BotMetricsRepository(mock_session)

        with pytest.raises(DatabaseError):
            await repo.get_latest_metrics("test-bot")


class TestRepositoryInitializationEdgeCases:
    """Test edge cases in repository initialization."""

    def test_bot_repository_init_with_execute_attribute(self):
        """Test BotRepository init when session has execute attribute."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        repo = BotRepository(mock_session)

        assert repo.db_service == mock_session
        assert hasattr(repo, 'session')

    def test_bot_repository_init_without_execute_attribute(self):
        """Test BotRepository init when session doesn't have execute attribute."""
        mock_service = MagicMock()
        # Remove execute attribute to trigger legacy path
        if hasattr(mock_service, 'execute'):
            delattr(mock_service, 'execute')

        repo = BotRepository(mock_service)

        assert repo.db_service == mock_service
        assert repo.session == mock_service
        assert repo.model == Bot

    @pytest.mark.asyncio
    async def test_rollback_session_already_closed(self):
        """Test rollback when session is already closed."""
        mock_session = AsyncMock()
        mock_session.commit.side_effect = Exception("Commit failed")
        mock_session.rollback.side_effect = Exception("Session already closed")

        repo = BotRepository(mock_session)

        with patch.object(repo, 'get', AsyncMock(return_value=None)):
            with pytest.raises(DatabaseError, match="Failed to update bot status.*Bot .* not found"):
                await repo.update_status("test-bot", BotStatus.STOPPED)


class TestErrorHandlingBranches:
    """Test error handling branches for coverage."""

    @pytest.mark.asyncio
    async def test_update_status_rollback_error(self):
        """Test update status when rollback also fails."""
        mock_session = AsyncMock()
        mock_bot = Bot()
        mock_bot.id = "test-bot"

        # Make commit fail
        mock_session.commit.side_effect = Exception("Commit failed")
        # Make rollback also fail
        mock_session.rollback.side_effect = Exception("Rollback failed")

        repo = BotRepository(mock_session)

        with patch.object(repo, 'get', AsyncMock(return_value=mock_bot)):
            with pytest.raises(DatabaseError):
                await repo.update_status("test-bot", BotStatus.STOPPED)

    @pytest.mark.asyncio
    async def test_bot_instance_update_metrics_rollback_error(self):
        """Test bot instance update metrics when rollback fails."""
        mock_session = AsyncMock()
        mock_instance = BotInstance()

        mock_session.commit.side_effect = Exception("Commit failed")
        mock_session.rollback.side_effect = Exception("Rollback failed")

        repo = BotInstanceRepository(mock_session)

        with patch.object(repo, 'get', AsyncMock(return_value=mock_instance)):
            mock_metrics = BotMetrics(
                bot_id="test-bot",
                created_at=datetime.utcnow(),
                total_trades=10,
                profitable_trades=6,
                losing_trades=4
            )

            with pytest.raises(DatabaseError):
                await repo.update_metrics("test-instance", mock_metrics)

    @pytest.mark.asyncio
    async def test_bot_metrics_save_rollback_error(self):
        """Test bot metrics save when rollback fails."""
        mock_session = AsyncMock()
        mock_instance = BotInstance()

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_instance
        mock_session.execute.return_value = mock_result
        mock_session.commit.side_effect = Exception("Commit failed")
        mock_session.rollback.side_effect = Exception("Rollback failed")

        repo = BotMetricsRepository(mock_session)

        mock_metrics = BotMetrics(
            bot_id="test-bot",
            created_at=datetime.utcnow(),
            total_trades=10,
            profitable_trades=6,
            losing_trades=4
        )

        with pytest.raises(DatabaseError):
            await repo.save_metrics(mock_metrics)