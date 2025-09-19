"""
Optimized fixtures for bot management tests with proper isolation and cleanup.
This conftest implements production-grade test isolation to prevent resource exhaustion.
"""

import logging
import gc
import weakref
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import ExitStack

import pytest

from src.core.config import Config
from src.core.types.bot import BotPriority, BotConfiguration, BotType, BotMetrics

# Disable logging during tests
logging.disable(logging.CRITICAL)

# Environment setup
import os
os.environ['TESTING'] = '1'
os.environ['DISABLE_REAL_NETWORK'] = '1'
os.environ['MOCK_DATABASE'] = '1'

# Constants
_MINIMAL_CAPITAL = Decimal("100")
_MINIMAL_POSITION = Decimal("10")
_TEST_SYMBOL = "BTCUSDT"
_TEST_EXCHANGE = "binance"
_BASE_TIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_START_TIME = _BASE_TIME - timedelta(hours=1)
_LAST_TRADE_TIME = _BASE_TIME - timedelta(minutes=5)

# Global registry for tracking created resources
_RESOURCE_REGISTRY = weakref.WeakSet()
_PATCH_STACK = None


@pytest.fixture(scope="session", autouse=True)
def global_test_setup():
    """Global setup and teardown for the entire test session."""
    global _PATCH_STACK
    _PATCH_STACK = ExitStack()
    
    # Apply global patches that are safe for the entire session
    _PATCH_STACK.enter_context(patch('psutil.cpu_percent', return_value=25.0))
    _PATCH_STACK.enter_context(patch('psutil.virtual_memory', MagicMock(return_value=MagicMock(percent=35.0))))
    _PATCH_STACK.enter_context(patch('psutil.boot_time', return_value=1609459200.0))
    
    yield
    
    # Cleanup
    _PATCH_STACK.close()
    _PATCH_STACK = None
    gc.collect()


@pytest.fixture(autouse=True)
def test_isolation():
    """Ensure proper isolation between tests."""
    # Clean up DI container before test
    from src.core.dependency_injection import DependencyInjector
    try:
        DependencyInjector.reset_instance()
    except Exception:
        pass
        
    # Clear any accumulated state before test
    gc.collect()
    
    # Track initial state
    import sys
    initial_modules = set(sys.modules.keys())
    
    yield
    
    # Cleanup after test
    # Clean up DI container after test
    try:
        DependencyInjector.reset_instance()
    except Exception:
        pass
        
    # Clear any resources tracked during test
    _RESOURCE_REGISTRY.clear()
    
    # Force garbage collection
    gc.collect()
    
    # Clear any singleton instances that might have been created
    from src.core.dependency_injection import DependencyContainer
    if hasattr(DependencyContainer, '_instance'):
        DependencyContainer._instance = None
    
    # Clear any cached imports from singleton services
    modules_to_clear = [
        'src.utils.validation.service',
        'src.risk_management.factory',
    ]
    for module in modules_to_clear:
        if module in sys.modules:
            # Don't delete, just clear the cache
            if hasattr(sys.modules[module], '__dict__'):
                for key in list(sys.modules[module].__dict__.keys()):
                    if key.startswith('_cached_') or key.startswith('_singleton_'):
                        del sys.modules[module].__dict__[key]


@pytest.fixture
def mock_async_operations():
    """Mock async operations with proper cleanup - not autouse."""
    with ExitStack() as stack:
        # Mock sleep operations
        stack.enter_context(patch('time.sleep', return_value=None))
        stack.enter_context(patch('asyncio.sleep', new_callable=AsyncMock, return_value=None))
        
        # Mock asyncio.create_task to prevent background tasks
        def mock_create_task(coro, **kwargs):
            mock_task = MagicMock()
            mock_task.cancel = MagicMock()
            mock_task.done = MagicMock(return_value=True)
            mock_task.cancelled = MagicMock(return_value=False)
            
            # Close coroutine to prevent warnings
            if hasattr(coro, 'close'):
                try:
                    coro.close()
                except Exception:
                    pass
            
            _RESOURCE_REGISTRY.add(mock_task)
            return mock_task
        
        stack.enter_context(patch('asyncio.create_task', side_effect=mock_create_task))
        
        yield
        
        # Cleanup tracked tasks
        for obj in _RESOURCE_REGISTRY:
            if hasattr(obj, 'cancel'):
                obj.cancel()


@pytest.fixture
def mock_background_loops():
    """Mock background loop methods - not autouse, use when needed."""
    with ExitStack() as stack:
        # Only mock the specific loops that are problematic
        loop_methods = [
            'src.bot_management.bot_coordinator.BotCoordinator._coordination_loop',
            'src.bot_management.bot_coordinator.BotCoordinator._signal_distribution_loop',
            'src.bot_management.bot_instance.BotInstance._trading_loop',
            'src.bot_management.bot_instance.BotInstance._heartbeat_loop',
            'src.bot_management.bot_monitor.BotMonitor._monitoring_loop',
            'src.bot_management.bot_lifecycle.BotLifecycle._lifecycle_loop',
        ]
        
        for method_path in loop_methods:
            stack.enter_context(patch(method_path, new_callable=AsyncMock))
        
        yield


@pytest.fixture(scope="function")
def base_config():
    """Create base test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    return config


@pytest.fixture(scope="function")
def lightweight_bot_config():
    """Create a lightweight bot configuration for tests."""
    import uuid
    return BotConfiguration(
        bot_id=f"test_bot_{uuid.uuid4().hex[:8]}",
        name="Test Bot",
        bot_type=BotType.TRADING,
        version="1.0.0",
        strategy_id="mean_reversion",
        strategy_name="Test Strategy",
        exchanges=[_TEST_EXCHANGE],
        symbols=[_TEST_SYMBOL],
        allocated_capital=_MINIMAL_CAPITAL,
        max_capital=_MINIMAL_CAPITAL,
        max_position_size=_MINIMAL_POSITION,
        priority=BotPriority.NORMAL,
        risk_percentage=0.02,
    )


@pytest.fixture
def lightweight_bot_metrics():
    """Create lightweight bot metrics."""
    return BotMetrics(
        bot_id="test_bot_001",
        total_trades=3,
        successful_trades=2,
        failed_trades=1,
        profitable_trades=2,
        losing_trades=1,
        total_pnl=Decimal("30.0"),
        win_rate=0.67,
        average_trade_pnl=Decimal("10.0"),
        uptime_percentage=0.9,
        error_count=0,
        last_heartbeat=_BASE_TIME,
        cpu_usage=25.0,
        memory_usage=35.0,
        api_calls_made=20,
        start_time=_START_TIME,
        last_trade_time=_LAST_TRADE_TIME,
        metrics_updated_at=_BASE_TIME,
    )


# Service mocks - created fresh for each test
@pytest.fixture(scope="function")
def mock_database_service():
    """Create a mock database service."""
    service = AsyncMock()
    service.store_bot_metrics = AsyncMock(return_value=True)
    service.get_bot_metrics = AsyncMock(return_value=None)
    service.create = AsyncMock(return_value=True)
    service.read = AsyncMock(return_value=None)
    service.update = AsyncMock(return_value=True)
    service.delete = AsyncMock(return_value=True)
    _RESOURCE_REGISTRY.add(service)
    return service


@pytest.fixture
def mock_bot_repository():
    """Mock bot repository with memory cleanup."""
    mock = AsyncMock()
    _RESOURCE_REGISTRY.add(mock)
    return mock


@pytest.fixture
def mock_bot_instance_repository():
    """Mock bot instance repository with memory cleanup."""
    mock = AsyncMock()
    _RESOURCE_REGISTRY.add(mock)
    return mock


@pytest.fixture
def mock_bot_metrics_repository():
    """Mock bot metrics repository with memory cleanup."""
    mock = AsyncMock()
    _RESOURCE_REGISTRY.add(mock)
    return mock


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector with memory cleanup."""
    mock = MagicMock()  # Use MagicMock since MetricsCollector might not be async
    _RESOURCE_REGISTRY.add(mock)
    return mock


@pytest.fixture(scope="function")
def mock_exchange_service():
    """Mock exchange service with memory cleanup."""
    mock = AsyncMock()
    _RESOURCE_REGISTRY.add(mock)
    return mock


@pytest.fixture
def full_bot_service_deps(
    base_config, mock_state_service, mock_risk_service,
    mock_execution_service, mock_strategy_service, mock_capital_service,
    mock_metrics_collector, mock_exchange_service
):
    """All dependencies needed for BotService initialization."""
    return {
        'exchange_service': mock_exchange_service,
        'capital_service': mock_capital_service,
        'execution_service': mock_execution_service,
        'risk_service': mock_risk_service,
        'state_service': mock_state_service,
        'strategy_service': mock_strategy_service,
        'metrics_collector': mock_metrics_collector,
        'config_service': base_config,
        'analytics_service': None,  # Optional parameter
    }


@pytest.fixture(scope="function")
def mock_execution_service():
    """Create a mock execution service."""
    service = AsyncMock()
    service.execute_order = AsyncMock(return_value=None)
    service.cancel_order = AsyncMock(return_value=True)
    service.get_order_status = AsyncMock(return_value="completed")
    _RESOURCE_REGISTRY.add(service)
    return service


@pytest.fixture(scope="function")
def mock_strategy_service():
    """Create a mock strategy service."""
    service = AsyncMock()
    service.get_strategy = AsyncMock(return_value=None)
    service.create_strategy = AsyncMock(return_value=None)
    service.start_strategy = AsyncMock(return_value=None)
    service.stop_strategy = AsyncMock(return_value=None)
    _RESOURCE_REGISTRY.add(service)
    return service


@pytest.fixture(scope="function")
def mock_risk_service():
    """Create a mock risk management service."""
    service = AsyncMock()
    service.validate_position = AsyncMock(return_value=True)
    service.calculate_position_size = AsyncMock(return_value=_MINIMAL_POSITION)
    service.check_risk_limits = AsyncMock(return_value=True)
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    _RESOURCE_REGISTRY.add(service)
    return service


@pytest.fixture(scope="function")
def mock_capital_service():
    """Create a mock capital management service."""
    service = AsyncMock()
    service.get_available_capital = AsyncMock(return_value=Decimal("1000"))
    service.allocate_capital = AsyncMock(return_value=True)
    service.release_capital = AsyncMock(return_value=True)
    _RESOURCE_REGISTRY.add(service)
    return service


@pytest.fixture(scope="function")
def mock_state_service():
    """Create a mock state management service."""
    service = AsyncMock()
    service.save_state = AsyncMock(return_value=None)
    service.load_state = AsyncMock(return_value={})
    service.get_bot_state = AsyncMock(return_value={})
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    _RESOURCE_REGISTRY.add(service)
    return service


@pytest.fixture
def mock_bot_instance_service():
    """Create a mock bot instance service."""
    service = AsyncMock()
    service.create_bot_instance = AsyncMock(return_value="test_bot_id")
    service.start_bot = AsyncMock(return_value=True)
    service.stop_bot = AsyncMock(return_value=True)
    service.get_bot_instance = AsyncMock(return_value=None)
    _RESOURCE_REGISTRY.add(service)
    return service


@pytest.fixture
def bot_config_factory():
    """Factory for creating bot configurations."""
    def _create_bot_config(bot_id: str = "test_bot", **overrides):
        config = {
            "bot_id": bot_id,
            "name": f"Test Bot {bot_id}",
            "bot_type": BotType.TRADING,
            "version": "1.0.0",
            "strategy_id": "mean_reversion",
            "strategy_name": "Test Strategy",
            "exchanges": [_TEST_EXCHANGE],
            "symbols": [_TEST_SYMBOL],
            "allocated_capital": _MINIMAL_CAPITAL,
            "max_capital": _MINIMAL_CAPITAL,
            "max_position_size": _MINIMAL_POSITION,
            "priority": BotPriority.NORMAL,
            "risk_percentage": 0.02,
        }
        config.update(overrides)
        return BotConfiguration(**config)
    
    return _create_bot_config


@pytest.fixture
def metrics_factory():
    """Factory for creating metrics."""
    def _create_metrics(bot_id: str = "test_bot", **overrides):
        metrics = {
            "bot_id": bot_id,
            "total_trades": 3,
            "successful_trades": 2,
            "failed_trades": 1,
            "profitable_trades": 2,
            "losing_trades": 1,
            "total_pnl": Decimal("30.0"),
            "win_rate": 0.67,
            "average_trade_pnl": Decimal("10.0"),
            "uptime_percentage": 0.9,
            "error_count": 0,
            "last_heartbeat": _BASE_TIME,
            "cpu_usage": 25.0,
            "memory_usage": 35.0,
            "api_calls_made": 20,
            "start_time": _START_TIME,
            "last_trade_time": _LAST_TRADE_TIME,
            "metrics_updated_at": _BASE_TIME,
        }
        metrics.update(overrides)
        return BotMetrics(**metrics)
    
    return _create_metrics


# Execution result factory
@pytest.fixture
def execution_result_factory():
    """Factory for creating ExecutionResult objects."""
    def _create_execution_result(status="COMPLETED", **overrides):
        from src.core.types import ExecutionResult, ExecutionStatus
        
        data = {
            "instruction_id": "exec_test_123",
            "symbol": _TEST_SYMBOL,
            "target_quantity": _MINIMAL_POSITION,
            "filled_quantity": _MINIMAL_POSITION,
            "remaining_quantity": Decimal("0"),
            "average_price": Decimal("50000"),
            "worst_price": Decimal("50100"),
            "best_price": Decimal("49900"),
            "expected_cost": Decimal("500"),
            "actual_cost": Decimal("500"),
            "slippage_bps": 0.0,
            "slippage_amount": Decimal("0"),
            "fill_rate": 1.0,
            "execution_time": 30,
            "num_fills": 1,
            "num_orders": 1,
            "total_fees": Decimal("1"),
            "maker_fees": Decimal("0.5"),
            "taker_fees": Decimal("0.5"),
            "started_at": _BASE_TIME,
            "completed_at": _BASE_TIME,
        }
        data.update(overrides)
        
        if isinstance(status, str):
            status = getattr(ExecutionStatus, status, ExecutionStatus.COMPLETED)
        
        return ExecutionResult(status=status, **data)
    
    return _create_execution_result


# Order request factory
@pytest.fixture
def order_request_factory():
    """Factory for creating OrderRequest objects."""
    def _create_order_request(side="BUY", order_type="MARKET", **overrides):
        from src.core.types import OrderRequest, OrderSide, OrderType
        
        data = {
            "symbol": _TEST_SYMBOL,
            "quantity": _MINIMAL_POSITION,
            "price": Decimal("50000"),
            "timestamp": _BASE_TIME,
        }
        data.update(overrides)
        
        if isinstance(side, str):
            side = getattr(OrderSide, side, OrderSide.BUY)
        if isinstance(order_type, str):
            order_type = getattr(OrderType, order_type, OrderType.MARKET)
        
        return OrderRequest(side=side, order_type=order_type, **data)
    
    return _create_order_request


# Minimal test data generators
@pytest.fixture
def minimal_order_data():
    """Provide minimal order data for testing."""
    return {
        "symbol": _TEST_SYMBOL,
        "side": "BUY",
        "quantity": _MINIMAL_POSITION,
        "price": Decimal("50000"),
        "order_type": "MARKET"
    }


@pytest.fixture
def minimal_position_data():
    """Provide minimal position data for testing."""
    return {
        "symbol": _TEST_SYMBOL,
        "side": "BUY",
        "quantity": _MINIMAL_POSITION,
        "average_price": Decimal("50000"),
        "unrealized_pnl": Decimal("100"),
        "realized_pnl": Decimal("0"),
        "timestamp": _BASE_TIME,
    }