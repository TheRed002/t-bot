"""Unit tests for strategies controller."""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timezone

from src.core.exceptions import ValidationError, ServiceError
from src.core.types import MarketData, StrategyConfig, StrategyType
from src.strategies.controller import StrategyController


@pytest.fixture
def mock_strategy_service():
    """Mock strategy service."""
    service = AsyncMock()
    return service


@pytest.fixture
def controller(mock_strategy_service):
    """Controller instance."""
    return StrategyController(mock_strategy_service)


@pytest.mark.asyncio
async def test_register_strategy_success(controller, mock_strategy_service):
    """Test successful strategy registration."""
    request_data = {
        "strategy_id": "test_strategy",
        "config": {
            "strategy_id": "test_strategy",
            "name": "test",
            "strategy_type": StrategyType.TREND_FOLLOWING.value,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "parameters": {}
        },
        "strategy_instance": Mock()
    }
    
    result = await controller.register_strategy(request_data)
    
    assert result["success"] is True
    assert result["strategy_id"] == "test_strategy"
    assert "registered successfully" in result["message"]
    mock_strategy_service.register_strategy.assert_called_once()


@pytest.mark.asyncio
async def test_register_strategy_missing_id(controller):
    """Test registration with missing strategy_id."""
    request_data = {"config": {"name": "test"}}
    
    result = await controller.register_strategy(request_data)
    
    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "strategy_id is required" in result["error"]


@pytest.mark.asyncio
async def test_register_strategy_missing_config(controller):
    """Test registration with missing config."""
    request_data = {"strategy_id": "test"}
    
    result = await controller.register_strategy(request_data)
    
    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "config is required" in result["error"]


@pytest.mark.asyncio
async def test_register_strategy_service_error(controller, mock_strategy_service):
    """Test registration with service error."""
    mock_strategy_service.register_strategy.side_effect = ServiceError("Service failed")
    
    request_data = {
        "strategy_id": "test_strategy",
        "config": {
            "strategy_id": "test_strategy",
            "name": "test",
            "strategy_type": StrategyType.TREND_FOLLOWING.value,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "parameters": {}
        }
    }
    
    result = await controller.register_strategy(request_data)
    
    assert result["success"] is False
    assert result["error_type"] == "ServiceError"
    assert "Service failed" in result["error"]


@pytest.mark.asyncio
async def test_start_strategy_success(controller, mock_strategy_service):
    """Test successful strategy start."""
    result = await controller.start_strategy("test_strategy")
    
    assert result["success"] is True
    assert result["strategy_id"] == "test_strategy"
    assert "started successfully" in result["message"]
    mock_strategy_service.start_strategy.assert_called_once_with("test_strategy")


@pytest.mark.asyncio
async def test_start_strategy_empty_id(controller):
    """Test start with empty strategy_id."""
    result = await controller.start_strategy("")
    
    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "strategy_id is required" in result["error"]


@pytest.mark.asyncio
async def test_start_strategy_service_error(controller, mock_strategy_service):
    """Test start with service error."""
    mock_strategy_service.start_strategy.side_effect = ServiceError("Start failed")
    
    result = await controller.start_strategy("test_strategy")
    
    assert result["success"] is False
    assert result["error_type"] == "ServiceError"
    assert "Start failed" in result["error"]


@pytest.mark.asyncio
async def test_stop_strategy_success(controller, mock_strategy_service):
    """Test successful strategy stop."""
    result = await controller.stop_strategy("test_strategy")
    
    assert result["success"] is True
    assert result["strategy_id"] == "test_strategy"
    assert "stopped successfully" in result["message"]
    mock_strategy_service.stop_strategy.assert_called_once_with("test_strategy")


@pytest.mark.asyncio
async def test_stop_strategy_empty_id(controller):
    """Test stop with empty strategy_id."""
    result = await controller.stop_strategy("")
    
    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "strategy_id is required" in result["error"]


@pytest.mark.asyncio
async def test_stop_strategy_service_error(controller, mock_strategy_service):
    """Test stop with service error."""
    mock_strategy_service.stop_strategy.side_effect = ServiceError("Stop failed")
    
    result = await controller.stop_strategy("test_strategy")
    
    assert result["success"] is False
    assert result["error_type"] == "ServiceError"
    assert "Stop failed" in result["error"]


@pytest.mark.asyncio
async def test_process_market_data_success(controller, mock_strategy_service):
    """Test successful market data processing."""
    market_data_dict = {
        "symbol": "BTCUSDT",
        "open": 49800.0,
        "high": 50200.0,
        "low": 49500.0,
        "close": 50000.0,
        "volume": 100.0,
        "exchange": "binance",
        "timestamp": datetime.now(timezone.utc)
    }
    
    mock_signals = [{"direction": "BUY", "confidence": 0.8}]
    mock_strategy_service.process_market_data.return_value = mock_signals
    
    result = await controller.process_market_data(market_data_dict)
    
    assert result["success"] is True
    assert result["signals"] == mock_signals
    assert result["symbol"] == "BTCUSDT"
    assert "processed_at" in result
    mock_strategy_service.process_market_data.assert_called_once()


@pytest.mark.asyncio
async def test_process_market_data_empty(controller):
    """Test processing empty market data."""
    result = await controller.process_market_data({})
    
    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "market_data is required" in result["error"]


@pytest.mark.asyncio
async def test_process_market_data_none(controller):
    """Test processing None market data."""
    result = await controller.process_market_data(None)
    
    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "market_data is required" in result["error"]


@pytest.mark.asyncio
async def test_process_market_data_service_error(controller, mock_strategy_service):
    """Test market data processing with service error."""
    mock_strategy_service.process_market_data.side_effect = ServiceError("Processing failed")
    
    market_data_dict = {
        "symbol": "BTCUSDT",
        "open": 49800.0,
        "high": 50200.0,
        "low": 49500.0,
        "close": 50000.0,
        "volume": 100.0,
        "exchange": "binance",
        "timestamp": datetime.now(timezone.utc)
    }
    
    result = await controller.process_market_data(market_data_dict)
    
    assert result["success"] is False
    assert result["error_type"] == "ServiceError"
    assert "Processing failed" in result["error"]


@pytest.mark.asyncio
async def test_get_strategy_performance_success(controller, mock_strategy_service):
    """Test successful performance retrieval."""
    mock_performance = {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.05
    }
    mock_strategy_service.get_strategy_performance.return_value = mock_performance
    
    result = await controller.get_strategy_performance("test_strategy")
    
    assert result["success"] is True
    assert result["performance"] == mock_performance
    assert result["strategy_id"] == "test_strategy"
    mock_strategy_service.get_strategy_performance.assert_called_once_with("test_strategy")


@pytest.mark.asyncio
async def test_get_strategy_performance_empty_id(controller):
    """Test performance retrieval with empty id."""
    result = await controller.get_strategy_performance("")
    
    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "strategy_id is required" in result["error"]


@pytest.mark.asyncio
async def test_get_strategy_performance_service_error(controller, mock_strategy_service):
    """Test performance retrieval with service error."""
    mock_strategy_service.get_strategy_performance.side_effect = ServiceError("Performance failed")
    
    result = await controller.get_strategy_performance("test_strategy")
    
    assert result["success"] is False
    assert result["error_type"] == "ServiceError"
    assert "Performance failed" in result["error"]


@pytest.mark.asyncio
async def test_get_all_strategies_success(controller, mock_strategy_service):
    """Test successful retrieval of all strategies."""
    mock_strategies = [
        {"id": "strategy1", "name": "Test Strategy 1"},
        {"id": "strategy2", "name": "Test Strategy 2"}
    ]
    mock_strategy_service.get_all_strategies.return_value = mock_strategies
    
    result = await controller.get_all_strategies()
    
    assert result["success"] is True
    assert result["strategies"] == mock_strategies
    assert result["count"] == 2
    mock_strategy_service.get_all_strategies.assert_called_once()


@pytest.mark.asyncio
async def test_get_all_strategies_empty_list(controller, mock_strategy_service):
    """Test retrieval with empty strategy list."""
    mock_strategy_service.get_all_strategies.return_value = []
    
    result = await controller.get_all_strategies()
    
    assert result["success"] is True
    assert result["strategies"] == []
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_get_all_strategies_service_error(controller, mock_strategy_service):
    """Test all strategies retrieval with service error."""
    mock_strategy_service.get_all_strategies.side_effect = ServiceError("Retrieval failed")
    
    result = await controller.get_all_strategies()
    
    assert result["success"] is False
    assert result["error_type"] == "ServiceError"
    assert "Retrieval failed" in result["error"]


@pytest.mark.asyncio
async def test_cleanup_strategy_success(controller, mock_strategy_service):
    """Test successful strategy cleanup."""
    result = await controller.cleanup_strategy("test_strategy")
    
    assert result["success"] is True
    assert result["strategy_id"] == "test_strategy"
    assert "cleaned up successfully" in result["message"]
    mock_strategy_service.cleanup_strategy.assert_called_once_with("test_strategy")


@pytest.mark.asyncio
async def test_cleanup_strategy_empty_id(controller):
    """Test cleanup with empty strategy_id."""
    result = await controller.cleanup_strategy("")
    
    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "strategy_id is required" in result["error"]


@pytest.mark.asyncio
async def test_cleanup_strategy_service_error(controller, mock_strategy_service):
    """Test cleanup with service error."""
    mock_strategy_service.cleanup_strategy.side_effect = ServiceError("Cleanup failed")
    
    result = await controller.cleanup_strategy("test_strategy")
    
    assert result["success"] is False
    assert result["error_type"] == "ServiceError"
    assert "Cleanup failed" in result["error"]


@pytest.mark.asyncio
async def test_controller_initialization(mock_strategy_service):
    """Test controller initialization."""
    controller = StrategyController(mock_strategy_service)
    
    assert controller._strategy_service == mock_strategy_service
    assert hasattr(controller, "register_strategy")
    assert hasattr(controller, "start_strategy")
    assert hasattr(controller, "stop_strategy")
    assert hasattr(controller, "process_market_data")
    assert hasattr(controller, "get_strategy_performance")
    assert hasattr(controller, "get_all_strategies")
    assert hasattr(controller, "cleanup_strategy")


@pytest.mark.asyncio 
async def test_register_strategy_with_instance(controller, mock_strategy_service):
    """Test registration with strategy instance."""
    mock_instance = Mock()
    request_data = {
        "strategy_id": "test_strategy",
        "config": {
            "strategy_id": "test_strategy",
            "name": "test",
            "strategy_type": StrategyType.TREND_FOLLOWING.value,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "parameters": {}
        },
        "strategy_instance": mock_instance
    }
    
    result = await controller.register_strategy(request_data)
    
    assert result["success"] is True
    call_args = mock_strategy_service.register_strategy.call_args[0]
    assert call_args[0] == "test_strategy"  # strategy_id
    assert call_args[1] == mock_instance  # strategy_instance
    assert isinstance(call_args[2], StrategyConfig)  # config


@pytest.mark.asyncio
async def test_register_strategy_without_instance(controller, mock_strategy_service):
    """Test registration without strategy instance."""
    request_data = {
        "strategy_id": "test_strategy",
        "config": {
            "strategy_id": "test_strategy",
            "name": "test",
            "strategy_type": StrategyType.TREND_FOLLOWING.value,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "parameters": {}
        }
    }
    
    result = await controller.register_strategy(request_data)
    
    assert result["success"] is True
    call_args = mock_strategy_service.register_strategy.call_args[0]
    assert call_args[0] == "test_strategy"  # strategy_id
    assert call_args[1] is None  # strategy_instance
    assert isinstance(call_args[2], StrategyConfig)  # config