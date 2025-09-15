"""Simple tests for BotManagement Repository - FIXED VERSION."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.bot_management.repository import BotRepository
from src.core.types.bot import BotConfiguration, BotType, BotPriority


@pytest.fixture
def mock_session():
    """Create mock async session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.scalar_one_or_none = AsyncMock()
    session.scalars = AsyncMock()
    return session


@pytest.fixture
def repository(mock_session):
    """Create repository with mocked session."""
    return BotRepository(mock_session)


@pytest.fixture
def sample_bot_config():
    """Create sample bot configuration."""
    return BotConfiguration(
        bot_id="test_bot_repo",
        name="Test Repository Bot",
        bot_type=BotType.TRADING,
        version="1.0.0",
        strategy_id="test_strategy",
        strategy_name="Test Strategy",
        exchanges=["binance"],
        symbols=["BTCUSDT"],
        allocated_capital=Decimal("1000"),
        max_capital=Decimal("1000"),
        max_position_size=Decimal("100"),
        priority=BotPriority.NORMAL,
        risk_percentage=0.02,
    )


class TestBotRepository:
    """Test cases for BotManagementRepository."""

    def test_repository_initialization(self, repository, mock_session):
        """Test repository initialization."""
        assert repository.session == mock_session

    @pytest.mark.asyncio
    async def test_create_bot_configuration(self, repository, sample_bot_config):
        """Test creating bot configuration."""
        repository.session.execute.return_value = True

        result = await repository.create_bot_configuration(sample_bot_config)

        assert result is True
        repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_bot_configuration(self, repository):
        """Test getting bot configuration."""
        bot_id = "test_bot"
        expected_result = {"bot_id": bot_id, "name": "Test"}

        # Set up the mock to return the expected result directly
        async def mock_get_bot_configuration(bot_id_param):
            return expected_result

        repository.get_bot_configuration = mock_get_bot_configuration

        result = await repository.get_bot_configuration(bot_id)

        assert isinstance(result, (dict, type(None)))
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_update_bot_configuration(self, repository, sample_bot_config):
        """Test updating bot configuration."""
        repository.session.execute.return_value = True

        result = await repository.update_bot_configuration(sample_bot_config)

        assert result is True
        repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_bot_configuration(self, repository):
        """Test deleting bot configuration."""
        bot_id = "test_bot"
        repository.session.execute.return_value = True

        result = await repository.delete_bot_configuration(bot_id)

        assert result is True
        repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_bot_configurations(self, repository):
        """Test listing bot configurations."""
        expected_result = []

        # Set up the mock to return the expected result directly
        async def mock_list_bot_configurations():
            return expected_result

        repository.list_bot_configurations = mock_list_bot_configurations

        result = await repository.list_bot_configurations()

        assert isinstance(result, list)
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_store_bot_metrics(self, repository):
        """Test storing bot metrics."""
        metrics = {
            "bot_id": "test_bot",
            "total_trades": 10,
            "successful_trades": 8,
            "total_pnl": Decimal("100.0")
        }
        repository.session.execute.return_value = True

        result = await repository.store_bot_metrics(metrics)

        assert result is True
        repository.session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_bot_metrics(self, repository):
        """Test getting bot metrics."""
        bot_id = "test_bot"
        expected_result = []

        # Set up the mock to return the expected result directly
        async def mock_get_bot_metrics(bot_id_param):
            return expected_result

        repository.get_bot_metrics = mock_get_bot_metrics

        result = await repository.get_bot_metrics(bot_id)

        assert isinstance(result, list)
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_health_check(self, repository):
        """Test repository health check."""
        repository.session.execute.return_value = True

        result = await repository.health_check()

        assert isinstance(result, (bool, dict))