"""Tests for database bot service."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from typing import Any

from src.core.exceptions import ServiceError, ValidationError
from src.database.services.bot_service import BotService


class TestBotService:
    """Test the BotService class."""

    @pytest.fixture
    def mock_bot_repo(self):
        """Create a mock bot repository."""
        return AsyncMock()

    @pytest.fixture
    def bot_service(self, mock_bot_repo):
        """Create a BotService instance with mocked dependencies."""
        return BotService(bot_repo=mock_bot_repo)

    @pytest.fixture
    def mock_bot(self):
        """Create a mock bot object."""
        bot = Mock()
        bot.id = "test-bot-123"
        bot.name = "Test Bot"
        bot.status = "ACTIVE"
        bot.created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        return bot

    async def test_init(self, mock_bot_repo):
        """Test BotService initialization."""
        service = BotService(bot_repo=mock_bot_repo)
        assert service.bot_repo == mock_bot_repo
        assert service.name == "BotService"

    async def test_init_without_repo(self):
        """Test BotService initialization without repository raises ValueError."""
        with pytest.raises(ValueError, match="bot_repo must be injected via dependency injection"):
            BotService(bot_repo=None)

    async def test_get_active_bots_success(self, bot_service, mock_bot_repo, mock_bot):
        """Test getting active bots successfully."""
        # Arrange
        mock_bot_repo.get_all.return_value = [mock_bot]

        # Act
        result = await bot_service.get_active_bots()

        # Assert
        assert len(result) == 1
        assert result[0]["bot_id"] == "test-bot-123"
        assert result[0]["name"] == "Test Bot"
        assert result[0]["status"] == "ACTIVE"
        assert result[0]["is_healthy"] is True
        assert result[0]["uptime_hours"] > 0
        
        mock_bot_repo.get_all.assert_called_once_with(
            filters={"status": "ACTIVE"}, order_by="-created_at"
        )

    async def test_get_active_bots_no_repository(self):
        """Test getting active bots without repository raises ValueError at init."""
        # Act & Assert - should fail at initialization
        with pytest.raises(ValueError, match="bot_repo must be injected via dependency injection"):
            BotService(bot_repo=None)

    async def test_get_active_bots_repository_error(self, bot_service, mock_bot_repo):
        """Test getting active bots with repository error."""
        # Arrange
        mock_bot_repo.get_all.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Get active bots failed"):
            await bot_service.get_active_bots()

    async def test_archive_bot_record_success(self, bot_service, mock_bot_repo, mock_bot):
        """Test archiving bot record successfully."""
        # Arrange
        mock_bot.status = "STOPPED"  # Make bot archivable
        mock_bot_repo.get_by_id.return_value = mock_bot
        mock_bot_repo.soft_delete.return_value = True

        # Act
        result = await bot_service.archive_bot_record("test-bot-123")

        # Assert
        assert result is True
        mock_bot_repo.get_by_id.assert_called_once_with("test-bot-123")
        mock_bot_repo.soft_delete.assert_called_once_with("test-bot-123", deleted_by="system")

    async def test_archive_bot_record_empty_id(self, bot_service):
        """Test archiving bot record with empty ID."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Bot ID is required"):
            await bot_service.archive_bot_record("")

    async def test_archive_bot_record_whitespace_id(self, bot_service):
        """Test archiving bot record with whitespace ID."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Bot ID is required"):
            await bot_service.archive_bot_record("   ")

    async def test_archive_bot_record_no_repository(self):
        """Test archiving bot record without repository raises ValueError at init."""
        # Act & Assert - should fail at initialization
        with pytest.raises(ValueError, match="bot_repo must be injected via dependency injection"):
            BotService(bot_repo=None)

    async def test_archive_bot_record_bot_not_found(self, bot_service, mock_bot_repo):
        """Test archiving bot record when bot not found."""
        # Arrange
        mock_bot_repo.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ValidationError, match="Bot test-bot-123 not found"):
            await bot_service.archive_bot_record("test-bot-123")

    async def test_archive_bot_record_cannot_archive_active_bot(self, bot_service, mock_bot_repo, mock_bot):
        """Test archiving active bot that cannot be archived."""
        # Arrange
        mock_bot.status = "ACTIVE"  # Active bot cannot be archived
        mock_bot_repo.get_by_id.return_value = mock_bot

        # Act & Assert
        with pytest.raises(ValidationError, match="cannot be archived in current state"):
            await bot_service.archive_bot_record("test-bot-123")

    async def test_archive_bot_record_repository_error(self, bot_service, mock_bot_repo, mock_bot):
        """Test archiving bot record with repository error."""
        # Arrange
        mock_bot.status = "STOPPED"
        mock_bot_repo.get_by_id.return_value = mock_bot
        mock_bot_repo.soft_delete.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Bot archival failed"):
            await bot_service.archive_bot_record("test-bot-123")

    async def test_get_bot_metrics_success(self, bot_service):
        """Test getting bot metrics successfully."""
        # Act
        result = await bot_service.get_bot_metrics("test-bot-123", limit=5)

        # Assert
        assert len(result) == 2  # Two default metrics
        assert result[0]["bot_id"] == "test-bot-123"
        assert result[0]["metric_type"] == "performance"
        assert result[0]["value"] == 95.5
        assert result[1]["metric_type"] == "uptime"
        assert result[1]["value"] == 99.9

    async def test_get_bot_metrics_with_limit(self, bot_service):
        """Test getting bot metrics with limit."""
        # Act
        result = await bot_service.get_bot_metrics("test-bot-123", limit=1)

        # Assert
        assert len(result) == 1
        assert result[0]["metric_type"] == "performance"

    async def test_get_bot_metrics_empty_bot_id(self, bot_service):
        """Test getting bot metrics with empty bot ID."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Bot ID is required"):
            await bot_service.get_bot_metrics("")

    async def test_get_bot_metrics_invalid_limit(self, bot_service):
        """Test getting bot metrics with invalid limit."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Limit must be positive"):
            await bot_service.get_bot_metrics("test-bot-123", limit=0)

    async def test_get_bot_metrics_negative_limit(self, bot_service):
        """Test getting bot metrics with negative limit."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Limit must be positive"):
            await bot_service.get_bot_metrics("test-bot-123", limit=-1)

    async def test_store_bot_metrics_success(self, bot_service):
        """Test storing bot metrics successfully."""
        # Arrange
        metrics_record = {
            "bot_id": "test-bot-123",
            "metric_type": "performance",
            "value": 85.5
        }

        # Act
        result = await bot_service.store_bot_metrics(metrics_record)

        # Assert
        assert result is True

    async def test_store_bot_metrics_missing_bot_id(self, bot_service):
        """Test storing bot metrics without bot_id."""
        # Arrange
        metrics_record = {
            "metric_type": "performance",
            "value": 85.5
        }

        # Act & Assert
        with pytest.raises(ValidationError, match="bot_id is required"):
            await bot_service.store_bot_metrics(metrics_record)

    async def test_store_bot_metrics_missing_metric_type(self, bot_service):
        """Test storing bot metrics without metric_type."""
        # Arrange
        metrics_record = {
            "bot_id": "test-bot-123",
            "value": 85.5
        }

        # Act & Assert
        with pytest.raises(ValidationError, match="metric_type is required"):
            await bot_service.store_bot_metrics(metrics_record)

    async def test_store_bot_metrics_missing_value(self, bot_service):
        """Test storing bot metrics without value."""
        # Arrange
        metrics_record = {
            "bot_id": "test-bot-123",
            "metric_type": "performance"
        }

        # Act & Assert
        with pytest.raises(ValidationError, match="value is required"):
            await bot_service.store_bot_metrics(metrics_record)

    async def test_store_bot_metrics_invalid_performance_value_negative(self, bot_service):
        """Test storing bot metrics with invalid negative performance value."""
        # Arrange
        metrics_record = {
            "bot_id": "test-bot-123",
            "metric_type": "performance",
            "value": -5.0
        }

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid value for performance: must be 0-100"):
            await bot_service.store_bot_metrics(metrics_record)

    async def test_store_bot_metrics_invalid_performance_value_over_100(self, bot_service):
        """Test storing bot metrics with invalid over-100 performance value."""
        # Arrange
        metrics_record = {
            "bot_id": "test-bot-123",
            "metric_type": "performance",
            "value": 150.0
        }

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid value for performance: must be 0-100"):
            await bot_service.store_bot_metrics(metrics_record)

    async def test_store_bot_metrics_invalid_uptime_value(self, bot_service):
        """Test storing bot metrics with invalid uptime value."""
        # Arrange
        metrics_record = {
            "bot_id": "test-bot-123",
            "metric_type": "uptime",
            "value": 105.0
        }

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid value for uptime: must be 0-100"):
            await bot_service.store_bot_metrics(metrics_record)

    async def test_store_bot_metrics_valid_custom_metric(self, bot_service):
        """Test storing bot metrics with custom metric type (no validation)."""
        # Arrange
        metrics_record = {
            "bot_id": "test-bot-123",
            "metric_type": "custom_metric",
            "value": 999.0  # Should be allowed for custom metrics
        }

        # Act
        result = await bot_service.store_bot_metrics(metrics_record)

        # Assert
        assert result is True

    def test_assess_bot_health_active(self, bot_service, mock_bot):
        """Test assessing bot health for active bot."""
        # Arrange
        mock_bot.status = "ACTIVE"

        # Act
        result = bot_service._assess_bot_health(mock_bot)

        # Assert
        assert result is True

    def test_assess_bot_health_inactive(self, bot_service, mock_bot):
        """Test assessing bot health for inactive bot."""
        # Arrange
        mock_bot.status = "STOPPED"

        # Act
        result = bot_service._assess_bot_health(mock_bot)

        # Assert
        assert result is False

    def test_calculate_uptime_hours_with_created_at(self, bot_service, mock_bot):
        """Test calculating uptime hours when bot has created_at."""
        # Arrange
        with patch('datetime.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 2, tzinfo=timezone.utc)  # 1 day later
            mock_datetime.now.return_value = mock_now
            mock_bot.created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)

            # Act
            result = bot_service._calculate_uptime_hours(mock_bot)

            # Assert
            assert result == 24.0  # 1 day = 24 hours

    def test_calculate_uptime_hours_without_created_at(self, bot_service, mock_bot):
        """Test calculating uptime hours when bot has no created_at."""
        # Arrange
        mock_bot.created_at = None

        # Act
        result = bot_service._calculate_uptime_hours(mock_bot)

        # Assert
        assert result == 0.0

    def test_can_archive_bot_active(self, bot_service, mock_bot):
        """Test checking if active bot can be archived."""
        # Arrange
        mock_bot.status = "ACTIVE"

        # Act
        result = bot_service._can_archive_bot(mock_bot)

        # Assert
        assert result is False

    def test_can_archive_bot_starting(self, bot_service, mock_bot):
        """Test checking if starting bot can be archived."""
        # Arrange
        mock_bot.status = "STARTING"

        # Act
        result = bot_service._can_archive_bot(mock_bot)

        # Assert
        assert result is False

    def test_can_archive_bot_stopping(self, bot_service, mock_bot):
        """Test checking if stopping bot can be archived."""
        # Arrange
        mock_bot.status = "STOPPING"

        # Act
        result = bot_service._can_archive_bot(mock_bot)

        # Assert
        assert result is False

    def test_can_archive_bot_stopped(self, bot_service, mock_bot):
        """Test checking if stopped bot can be archived."""
        # Arrange
        mock_bot.status = "STOPPED"

        # Act
        result = bot_service._can_archive_bot(mock_bot)

        # Assert
        assert result is True

    def test_can_archive_bot_error(self, bot_service, mock_bot):
        """Test checking if error bot can be archived."""
        # Arrange
        mock_bot.status = "ERROR"

        # Act
        result = bot_service._can_archive_bot(mock_bot)

        # Assert
        assert result is True


class TestBotServiceErrorHandling:
    """Test error handling in BotService."""

    @pytest.fixture
    def bot_service(self):
        """Create a BotService instance with mocked dependencies."""
        mock_repo = Mock()
        return BotService(bot_repo=mock_repo)

    async def test_get_active_bots_exception_propagation(self, bot_service):
        """Test that exceptions in get_active_bots are properly wrapped."""
        # Arrange
        bot_service.bot_repo.get_all.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(ServiceError):
            await bot_service.get_active_bots()

    async def test_archive_bot_record_exception_propagation(self, bot_service):
        """Test that exceptions in archive_bot_record are properly wrapped."""
        # Arrange
        bot_service.bot_repo.get_by_id.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(ServiceError):
            await bot_service.archive_bot_record("test-bot-123")

    async def test_get_bot_metrics_exception_propagation(self, bot_service):
        """Test that exceptions in get_bot_metrics are properly wrapped."""
        # Act & Assert
        with pytest.raises(ValidationError):
            await bot_service.get_bot_metrics("")

    async def test_store_bot_metrics_exception_propagation(self, bot_service):
        """Test that exceptions in store_bot_metrics are properly wrapped."""
        # Act & Assert
        with pytest.raises(ValidationError):
            await bot_service.store_bot_metrics({})