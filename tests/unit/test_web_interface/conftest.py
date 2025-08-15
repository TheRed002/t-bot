"""
Test configuration and fixtures for web interface tests.

This module provides pytest fixtures and configuration for testing
the T-Bot FastAPI web interface components.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.core.config import Config
from src.web_interface.app import create_app
from src.web_interface.security.auth import init_auth


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = Config()
    config.debug = True
    
    # Configure security for testing
    config.security.secret_key = "test_secret_key_for_testing_only_do_not_use_in_production"
    config.security.jwt_algorithm = "HS256"
    config.security.jwt_expire_minutes = 30
    
    return config


@pytest.fixture
def mock_bot_orchestrator():
    """Create mock bot orchestrator."""
    mock = AsyncMock()
    
    # Mock methods
    mock.start = AsyncMock(return_value=None)
    mock.stop = AsyncMock(return_value=None)
    mock.create_bot = AsyncMock(return_value="bot_test_001")
    mock.start_bot = AsyncMock(return_value=True)
    mock.stop_bot = AsyncMock(return_value=True)
    mock.pause_bot = AsyncMock(return_value=True)
    mock.resume_bot = AsyncMock(return_value=True)
    mock.delete_bot = AsyncMock(return_value=True)
    mock.get_bot_list = AsyncMock(return_value=[])
    mock.get_orchestrator_status = AsyncMock(return_value={
        "orchestrator": {"is_running": True, "total_bots": 0},
        "global_metrics": {"total_pnl": 0, "total_trades": 0}
    })
    
    # Mock attributes
    mock.bot_instances = {}
    mock.bot_configurations = {}
    
    return mock


@pytest.fixture
def mock_execution_engine():
    """Create mock execution engine."""
    mock = AsyncMock()
    
    # Mock methods
    mock.start = AsyncMock(return_value=None)
    mock.stop = AsyncMock(return_value=None)
    mock.execute_order = AsyncMock()
    mock.cancel_execution = AsyncMock(return_value=True)
    mock.get_execution_status = AsyncMock(return_value=None)
    mock.get_engine_summary = AsyncMock(return_value={
        "engine_status": {"is_running": True},
        "performance_statistics": {"total_executions": 0}
    })
    
    return mock


@pytest.fixture
def mock_model_manager():
    """Create mock model manager."""
    mock = AsyncMock()
    
    # Mock methods
    mock.create_and_train_model = AsyncMock()
    mock.deploy_model = AsyncMock()
    mock.monitor_model_performance = AsyncMock()
    mock.retire_model = AsyncMock()
    mock.get_active_models = AsyncMock(return_value={})
    mock.health_check = AsyncMock(return_value={"status": "healthy"})
    
    return mock


@pytest.fixture
def test_app(test_config, mock_bot_orchestrator, mock_execution_engine, mock_model_manager):
    """Create test FastAPI application."""
    app = create_app(
        config=test_config,
        bot_orchestrator_instance=mock_bot_orchestrator,
        execution_engine_instance=mock_execution_engine,
        model_manager_instance=mock_model_manager
    )
    
    # Initialize authentication for testing
    init_auth(test_config)
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
async def async_client(test_app):
    """Create async test client."""
    async with AsyncClient(app=test_app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    # This would normally create a valid JWT token
    # For testing, we'll use a mock token
    return {
        "Authorization": "Bearer test_token_for_testing"
    }


@pytest.fixture
def admin_auth_headers():
    """Create admin authentication headers for testing."""
    return {
        "Authorization": "Bearer admin_test_token_for_testing"
    }


@pytest.fixture
def mock_jwt_handler():
    """Create mock JWT handler."""
    from src.web_interface.security.jwt_handler import TokenData
    from datetime import datetime, timezone
    
    mock = MagicMock()
    
    # Mock token validation
    mock.validate_token.return_value = TokenData(
        username="testuser",
        user_id="test_user_001",
        scopes=["read", "write", "trade"],
        issued_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc)
    )
    
    # Mock password operations
    mock.hash_password.return_value = "hashed_password"
    mock.verify_password.return_value = True
    
    # Mock token creation
    mock.create_access_token.return_value = "test_access_token"
    mock.create_refresh_token.return_value = "test_refresh_token"
    
    return mock


@pytest.fixture
def sample_bot_config():
    """Create sample bot configuration for testing."""
    from src.core.types import BotConfiguration, BotType, BotPriority
    from decimal import Decimal
    from datetime import datetime
    
    return BotConfiguration(
        bot_id="test_bot_001",
        bot_name="Test Bot",
        bot_type=BotType.STRATEGY,
        strategy_name="trend_following",
        exchanges=["binance"],
        symbols=["BTCUSDT"],
        allocated_capital=Decimal("10000"),
        risk_percentage=0.02,
        priority=BotPriority.NORMAL,
        auto_start=False,
        configuration={},
        created_by="test_user",
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_order_request():
    """Create sample order request for testing."""
    from src.core.types import OrderRequest, OrderSide, OrderType
    from decimal import Decimal
    
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("45000.0"),
        time_in_force="GTC",
        client_order_id="test_order_001"
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    from src.core.types import MarketData
    from decimal import Decimal
    from datetime import datetime
    
    return MarketData(
        symbol="BTCUSDT",
        price=Decimal("45000.0"),
        bid=Decimal("44995.0"),
        ask=Decimal("45005.0"),
        volume=Decimal("1000.0"),
        high_price=Decimal("46000.0"),
        low_price=Decimal("44000.0"),
        timestamp=datetime.utcnow()
    )