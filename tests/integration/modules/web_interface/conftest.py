"""
Web Interface Integration Test Fixtures.

Provides REAL services for integration testing - NO MOCKS for internal services.
Uses real PostgreSQL, Redis, and all internal services.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration for web interface."""
    config = Config()
    config.environment = "test"
    config.debug = True

    # Ensure security config exists
    if not hasattr(config, 'security') or config.security is None:
        from src.core.config import SecurityConfig
        config.security = SecurityConfig(
            secret_key="test_jwt_secret_key_for_integration_tests",
            jwt_algorithm="HS256",
            jwt_expire_minutes=15,
            refresh_token_expire_days=7
        )

    return config


@pytest.fixture(scope="session")
def real_app(test_config):
    """
    Create REAL FastAPI application with all real services.
    NO MOCKS - uses actual DI container with real services.

    Using direct FastAPI creation without lifespan for testing.
    """
    from fastapi import FastAPI
    from src.web_interface.app import _register_routes, _setup_monitoring
    from fastapi.middleware.cors import CORSMiddleware

    # Create minimal FastAPI app for testing
    app = FastAPI(
        title="T-Bot Trading System API - Test",
        description="Test instance",
        version="1.0.0-test",
        docs_url="/docs",
        redoc_url="/redoc",
        debug=True
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add security middleware
    try:
        from src.web_interface.middleware.security import SecurityMiddleware
        app.add_middleware(SecurityMiddleware, enable_csp=True, enable_input_validation=False)
    except Exception as e:
        logger.warning(f"Security middleware setup failed: {e}")

    # Add versioning middleware
    try:
        from src.web_interface.versioning import VersioningMiddleware
        app.add_middleware(VersioningMiddleware)
    except Exception as e:
        logger.warning(f"Versioning middleware setup failed: {e}")

    # Register routes
    _register_routes(app)

    # Setup monitoring (non-critical for tests)
    try:
        _setup_monitoring(app, test_config)
    except Exception as e:
        logger.warning(f"Monitoring setup skipped in tests: {e}")

    return app


@pytest.fixture
def test_client(real_app):
    """Create synchronous test client for FastAPI app."""
    with TestClient(real_app) as client:
        yield client


@pytest_asyncio.fixture
async def async_test_client(real_app):
    """Create async test client for FastAPI app."""
    async with AsyncClient(app=real_app, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def authenticated_client(test_client, test_user):
    """Create authenticated test client with valid JWT token."""
    # Login to get real JWT token
    response = test_client.post(
        "/api/auth/login",
        json={
            "username": test_user["username"],
            "password": test_user["password"]
        }
    )

    if response.status_code == 200:
        token_data = response.json()
        token = token_data.get("access_token")
    else:
        # Fallback: create user and try again
        from src.web_interface.services.auth_service import AuthService
        from src.core.dependency_injection import injector

        auth_service = injector.resolve("AuthService")
        user_result = await auth_service.create_user(
            username=test_user["username"],
            email=test_user["email"],
            password=test_user["password"]
        )

        # Login again
        response = test_client.post(
            "/api/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        token_data = response.json()
        token = token_data.get("access_token")

    # Add authorization header to client
    test_client.headers.update({"Authorization": f"Bearer {token}"})

    yield test_client


@pytest_asyncio.fixture
async def admin_client(test_client, admin_user):
    """Create admin authenticated test client with valid JWT token."""
    # Login to get real JWT token
    response = test_client.post(
        "/api/auth/login",
        json={
            "username": admin_user["username"],
            "password": admin_user["password"]
        }
    )

    if response.status_code == 200:
        token_data = response.json()
        token = token_data.get("access_token")
    else:
        # Fallback: create admin user
        from src.web_interface.services.auth_service import AuthService
        from src.core.dependency_injection import injector

        auth_service = injector.resolve("AuthService")
        user_result = await auth_service.create_user(
            username=admin_user["username"],
            email=admin_user["email"],
            password=admin_user["password"],
            role="admin"
        )

        # Login again
        response = test_client.post(
            "/api/auth/login",
            json={
                "username": admin_user["username"],
                "password": admin_user["password"]
            }
        )
        token_data = response.json()
        token = token_data.get("access_token")

    # Add authorization header to client
    test_client.headers.update({"Authorization": f"Bearer {token}"})

    yield test_client


@pytest.fixture
def test_user():
    """Test user credentials."""
    return {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "TestPassword123!",
        "role": "user"
    }


@pytest.fixture
def admin_user():
    """Admin user credentials."""
    return {
        "username": "admin",
        "email": "admin@example.com",
        "password": "AdminPassword123!",
        "role": "admin"
    }


@pytest.fixture
def auth_headers(test_user):
    """
    Create authentication headers with mock token for endpoints that don't validate.
    For real validation, use authenticated_client fixture.
    """
    return {"Authorization": "Bearer test_jwt_token"}


@pytest.fixture
def admin_headers(admin_user):
    """
    Create admin authentication headers with mock token.
    For real validation, use admin_client fixture.
    """
    return {"Authorization": "Bearer admin_jwt_token"}


@pytest_asyncio.fixture
async def real_bot_service(clean_database):
    """Get real BotManagementService from DI container."""
    from src.core.dependency_injection import injector

    service = injector.resolve("BotManagementService")
    yield service


@pytest_asyncio.fixture
async def real_trading_service(clean_database):
    """Get real TradingService from DI container."""
    from src.core.dependency_injection import injector

    service = injector.resolve("TradingService")
    yield service


@pytest_asyncio.fixture
async def real_portfolio_service(clean_database):
    """Get real PortfolioService from DI container."""
    from src.core.dependency_injection import injector

    service = injector.resolve("PortfolioService")
    yield service


@pytest_asyncio.fixture
async def real_risk_service(clean_database):
    """Get real RiskService from DI container."""
    from src.core.dependency_injection import injector

    service = injector.resolve("RiskService")
    yield service


@pytest_asyncio.fixture
async def real_strategy_service(clean_database):
    """Get real StrategyService from DI container."""
    from src.core.dependency_injection import injector

    service = injector.resolve("StrategyServiceImpl")
    yield service


@pytest_asyncio.fixture
async def real_auth_service(clean_database):
    """Get real AuthService from DI container."""
    from src.core.dependency_injection import injector

    service = injector.resolve("AuthService")
    yield service


@pytest_asyncio.fixture
async def real_market_data_service(clean_database):
    """Get real MarketDataService from DI container."""
    from src.core.dependency_injection import injector

    service = injector.resolve("MarketDataService")
    yield service


@pytest.fixture
def sample_bot_config():
    """Sample bot configuration for testing."""
    return {
        "name": "Test Bot",
        "strategy": "momentum",
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "exchanges": ["binance"],
        "capital_allocation": "1000.00",
        "risk_profile": "medium",
        "parameters": {
            "lookback_period": 14,
            "threshold": 0.02,
            "max_position_size": 0.1
        }
    }


@pytest.fixture
def sample_order_request():
    """Sample order request for testing."""
    return {
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "limit",
        "quantity": "0.001",
        "price": "45000.00",
        "exchange": "binance",
        "time_in_force": "GTC"
    }


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing."""
    return {
        "total_value": Decimal("100000.00"),
        "positions": [
            {
                "symbol": "BTC/USDT",
                "quantity": Decimal("1.5"),
                "average_price": Decimal("45000.00"),
                "current_price": Decimal("47000.00"),
                "unrealized_pnl": Decimal("3000.00")
            },
            {
                "symbol": "ETH/USDT",
                "quantity": Decimal("20.0"),
                "average_price": Decimal("2800.00"),
                "current_price": Decimal("3000.00"),
                "unrealized_pnl": Decimal("4000.00")
            }
        ],
        "cash_balance": Decimal("30000.00"),
        "daily_pnl": Decimal("1500.00"),
        "total_pnl": Decimal("7000.00")
    }


@pytest.fixture
def sample_risk_metrics():
    """Sample risk metrics for testing."""
    return {
        "portfolio_var_1d": Decimal("2500.00"),
        "portfolio_var_5d": Decimal("5500.00"),
        "max_drawdown": Decimal("0.15"),
        "sharpe_ratio": Decimal("1.85"),
        "total_risk_score": 6.5,
        "position_concentrations": {
            "BTC/USDT": 0.45,
            "ETH/USDT": 0.35
        },
        "risk_limits_status": "within_limits"
    }


@pytest_asyncio.fixture
async def websocket_test_context(real_app):
    """Context for WebSocket testing."""
    from src.web_interface.websockets import get_unified_websocket_manager

    ws_manager = get_unified_websocket_manager()

    # Ensure WebSocket manager is started
    if not ws_manager.sio:
        ws_manager.create_server()

    yield ws_manager

    # Cleanup handled by app lifespan


@pytest.fixture
def api_endpoints():
    """List of API endpoints for testing."""
    return {
        "health": {
            "basic": "/health",
            "detailed": "/api/health/detailed"
        },
        "auth": {
            "login": "/api/auth/login",
            "logout": "/api/auth/logout",
            "refresh": "/api/auth/refresh",
            "register": "/api/auth/register"
        },
        "bot": {
            "list": "/api/bot",
            "create": "/api/bot",
            "get": "/api/bot/{bot_id}",
            "start": "/api/bot/{bot_id}/start",
            "stop": "/api/bot/{bot_id}/stop"
        },
        "trading": {
            "orders": "/api/trading/orders",
            "create_order": "/api/trading/orders",
            "positions": "/api/trading/positions",
            "trades": "/api/trading/trades"
        },
        "portfolio": {
            "summary": "/api/portfolio",
            "positions": "/api/portfolio/positions",
            "performance": "/api/portfolio/performance"
        },
        "risk": {
            "metrics": "/api/risk/metrics",
            "limits": "/api/risk/limits",
            "alerts": "/api/risk/alerts"
        },
        "strategies": {
            "list": "/api/strategies",
            "get": "/api/strategies/{strategy_id}",
            "backtest": "/api/strategies/{strategy_id}/backtest"
        }
    }


@pytest.fixture
def expected_response_schemas():
    """Expected response schemas for validation."""
    return {
        "health": ["status", "service", "version"],
        "bot_list": ["bots"],
        "bot_detail": ["id", "name", "status", "strategy"],
        "order": ["id", "symbol", "side", "type", "status"],
        "portfolio": ["total_value", "positions", "cash_balance"],
        "risk_metrics": ["portfolio_var_1d", "total_risk_score"]
    }


@pytest_asyncio.fixture
async def cleanup_test_data(clean_database):
    """Cleanup test data after tests."""
    yield

    # Cleanup is handled by clean_database fixture
    logger.info("Test data cleanup completed")


@pytest.fixture(autouse=True)
async def reset_service_state():
    """Reset service state between tests."""
    # This runs before each test
    yield
    # This runs after each test

    # Services are managed by DI container and clean_database fixture
    # No manual cleanup needed here


@pytest_asyncio.fixture
async def di_container():
    """
    Provide fully configured DI container with all services registered.

    Uses master DI registration to ensure all dependencies are properly configured
    in the correct order without circular dependency issues.
    """
    from tests.integration.conftest import register_all_services_for_testing

    container = register_all_services_for_testing()
    yield container

    # Cleanup is handled by individual services
