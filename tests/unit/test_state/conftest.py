"""
State test configuration optimized for ultra-fast execution.

This conftest.py specifically optimizes state tests by:
1. Disabling ALL logging during test collection and execution
2. Mocking ALL expensive service initialization
3. Using session-scoped fixtures to minimize repeated setup
4. Preventing database connections, file I/O, and network operations
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

# ULTRA-AGGRESSIVE logging suppression
logging.disable(logging.CRITICAL)

# Set environment variables needed for testing
# Note: We don't set TESTING here as some tests need to control it
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_LOGGING", "1")
os.environ.setdefault("DISABLE_ERROR_HANDLER_LOGGING", "1")
os.environ.setdefault("DISABLE_HANDLER_POOL", "1")
os.environ.setdefault("DISABLE_VALIDATOR_REGISTRY", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("ASYNCIO_DEBUG", "0")

# Suppress ALL loggers immediately
for logger_name in ["src", "root", "__main__", "asyncio", "pytest"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True
    logger.propagate = False


# SELECTIVE module mocking - avoid mocking core DI functionality
MOCK_MODULES = {
    "src.core.logging": Mock(get_logger=Mock(return_value=Mock(level=50, setLevel=Mock()))),
    "src.error_handling.service": Mock(ErrorHandlingService=Mock()),
    "src.error_handling.decorators": Mock(
        with_error_handling=lambda func: func, shutdown_all_error_handlers=AsyncMock()
    ),
    "src.error_handling.handler_pool": Mock(HandlerPool=Mock()),
    "src.database.redis_client": Mock(RedisClient=Mock()),
    "src.database.manager": Mock(DatabaseManager=Mock()),
    "src.database.influxdb_client": Mock(InfluxDBClient=Mock()),
    "src.database.connection": Mock(
        initialize_database=AsyncMock(), close_database=AsyncMock(), get_async_session=Mock()
    ),
    "src.monitoring.telemetry": Mock(get_tracer=Mock(return_value=Mock())),
    "src.utils.validation.core": Mock(ValidatorRegistry=Mock()),
    # Remove utils_imports mock to allow testing of actual module
    "src.utils.file_utils": Mock(ensure_directory_exists=Mock()),
    "redis": Mock(),
    "influxdb_client": Mock(),
    "psycopg2": Mock(),
    "sqlalchemy": Mock(),
}


# Create proper mock services for dependency injection
class MockDatabaseService:
    """Mock database service for testing."""

    def __init__(self):
        self.initialized = True
        self.is_healthy = True

    async def start(self):
        pass

    async def stop(self):
        pass

    async def health_check(self):
        return {"status": "healthy"}


class MockValidationService:
    """Mock validation service for testing."""

    def __init__(self):
        self.initialized = True

    async def validate(self, data):
        return True


# Apply all mocks immediately
for module_name, mock_module in MOCK_MODULES.items():
    sys.modules[module_name] = mock_module


@pytest.fixture(autouse=True)
def ultra_aggressive_performance_optimization():
    """Ultra-aggressive performance optimization for state tests."""
    # Create a proper mock logger with level attribute
    mock_logger = Mock()
    mock_logger.level = logging.CRITICAL
    mock_logger.setLevel = Mock()
    mock_logger.debug = Mock()
    mock_logger.info = Mock()
    mock_logger.warning = Mock()
    mock_logger.error = Mock()
    mock_logger.critical = Mock()

    # Mock only non-asyncio operations to prevent interference
    with (
        patch("time.sleep"),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.exists", return_value=True),
        patch("os.makedirs"),
        patch("tempfile.mkdtemp", return_value="/tmp/test"),
        patch("json.dump"),
        patch("json.load", return_value={}),
        patch("pickle.dump"),
        patch("pickle.load", return_value=Mock()),
        patch("logging.getLogger", return_value=mock_logger),
        patch("structlog.get_logger", return_value=Mock()),
    ):
        yield


@pytest.fixture(scope="session")
def ultra_fast_config():
    """Ultra-fast mock configuration optimized for maximum speed."""
    config = Mock()

    # State management with minimal values
    config.state_management = Mock()
    config.state_management.max_snapshots_per_bot = 1
    config.state_management.snapshot_interval_minutes = 0.001
    config.state_management.redis_ttl_seconds = 0.1
    config.state_management.enable_compression = False
    config.state_management.checkpoints = Mock()
    config.state_management.checkpoints.directory = "/tmp/checkpoints"

    # Quality controls with minimal values
    config.quality_controls = Mock()
    config.quality_controls.min_quality_score = 50.0
    config.quality_controls.slippage_threshold_bps = 50.0
    config.quality_controls.execution_time_threshold_seconds = 1.0
    config.quality_controls.market_impact_threshold_bps = 50.0

    # State sync with minimal values
    config.state_sync = Mock()
    config.state_sync.sync_interval_seconds = 0.001
    config.state_sync.conflict_resolution_timeout_seconds = 0.01
    config.state_sync.max_sync_retries = 1
    config.state_sync.default_resolution_strategy = "last_write_wins"

    # Database with in-memory alternatives
    config.database = Mock()
    config.database.url = "sqlite:///:memory:"
    config.database.postgresql_host = "localhost"
    config.database.postgresql_port = 5432
    config.database.redis_host = "localhost"
    config.database.redis_port = 6379

    return config


@pytest.fixture(scope="session")
def sample_bot_state():
    """Minimal bot state for ultra-fast testing."""
    from src.core.types import BotPriority, BotState, BotStatus

    return BotState(
        bot_id="test",
        status=BotStatus.RUNNING,
        priority=BotPriority.NORMAL,
        allocated_capital=Decimal("100"),
        used_capital=Decimal("50"),
        open_positions=[],
        active_orders=[],
        metadata={},
    )


@pytest.fixture(scope="session")
def minimal_order_request():
    """Minimal order request for ultra-fast testing."""
    from src.core.types import OrderRequest, OrderSide, OrderType

    return OrderRequest(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.001"),
        price=Decimal("100"),
        client_order_id="test",
    )


@pytest.fixture(scope="session")
def minimal_market_data():
    """Minimal market data for ultra-fast testing."""
    from src.core.types import MarketData

    return MarketData(
        symbol="BTC/USDT",
        timestamp=datetime(2023, 1, 1),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("10"),
        exchange="test",
    )


@pytest.fixture(scope="session")
def minimal_execution_result():
    """Minimal execution result for ultra-fast testing."""
    from src.core.types import ExecutionResult, ExecutionStatus

    return ExecutionResult(
        instruction_id="exec",
        symbol="BTC/USDT",
        status=ExecutionStatus.COMPLETED,
        target_quantity=Decimal("0.001"),
        filled_quantity=Decimal("0.001"),
        remaining_quantity=Decimal("0"),
        average_price=Decimal("100"),
        worst_price=Decimal("101"),
        best_price=Decimal("99"),
        expected_cost=Decimal("0.1"),
        actual_cost=Decimal("0.1"),
        slippage_bps=1.0,
        slippage_amount=Decimal("0.001"),
        fill_rate=1.0,
        execution_time=1,
        num_fills=1,
        num_orders=1,
        total_fees=Decimal("0.001"),
        maker_fees=Decimal("0"),
        taker_fees=Decimal("0.001"),
        started_at=datetime(2023, 1, 1),
        completed_at=datetime(2023, 1, 1),
        metadata={},
    )


@pytest.fixture(scope="session")
def mock_config(ultra_fast_config):
    """Alias for ultra_fast_config for backward compatibility."""
    return ultra_fast_config


# Session-scoped cleanup to prevent async warnings
@pytest.fixture(autouse=True, scope="session")
def cleanup_session():
    """Clean up session-level resources."""
    yield

    # Cancel any pending tasks and wait for them to complete
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in tasks:
                if not task.cancelled():
                    task.cancel()
            # Wait a bit for tasks to be cancelled
            if tasks:
                try:
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                except Exception:
                    pass
    except Exception:
        pass


@pytest.fixture(autouse=True)
def cleanup_test():
    """Clean up after each test to prevent task leakage."""
    yield

    # Cancel any tasks created during the test
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in tasks:
                if not task.cancelled():
                    task.cancel()
    except Exception:
        pass
