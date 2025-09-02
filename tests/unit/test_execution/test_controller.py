"""Unit tests for ExecutionController."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from decimal import Decimal
from datetime import datetime, timezone

from src.core.types import OrderRequest, OrderSide, OrderType
from src.execution.controller import ExecutionController
from src.execution.types import ExecutionInstruction


class TestExecutionController:
    """Test cases for ExecutionController."""

    @pytest.fixture
    def mock_execution_service(self):
        """Create mock execution service."""
        service = AsyncMock()
        service.execute_order = AsyncMock()
        service.cancel_execution = AsyncMock()
        service.get_execution_status = AsyncMock()
        return service

    @pytest.fixture
    def mock_orchestration_service(self):
        """Create mock orchestration service."""
        service = AsyncMock()
        service.execute_order = AsyncMock()
        service.cancel_execution = AsyncMock()
        service.get_comprehensive_metrics = AsyncMock()
        return service

    @pytest.fixture
    def execution_controller(self, mock_orchestration_service, mock_execution_service):
        """Create ExecutionController instance."""
        return ExecutionController(mock_orchestration_service, mock_execution_service)

    @pytest.fixture
    def sample_order_request(self):
        """Create sample order request."""
        return {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": "1.0",
            "price": "50000.0",
            "time_in_force": "GTC",
            "client_order_id": "test_order_001"
        }

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return {
            "symbol": "BTC/USDT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open": "49900.0",
            "high": "50100.0", 
            "low": "49800.0",
            "close": "50000.0",  # This is what should be used as 'price'
            "volume": "100.0",
            "exchange": "binance",
            "bid_price": "49950.0",
            "ask_price": "50050.0"
        }

    def test_initialization(self, execution_controller, mock_execution_service):
        """Test controller initialization."""
        assert execution_controller.execution_service == mock_execution_service

    @pytest.mark.asyncio
    async def test_execute_order(self, execution_controller, mock_orchestration_service, sample_order_request, sample_market_data):
        """Test executing an order."""
        # Setup mock ExecutionResultWrapper response
        from src.execution.execution_result_wrapper import ExecutionResultWrapper
        from src.core.types.execution import ExecutionResult, ExecutionStatus, ExecutionAlgorithm
        from decimal import Decimal
        from datetime import datetime, timezone
        from unittest.mock import patch
        
        mock_core_result = ExecutionResult(
            instruction_id="exec_001",
            symbol="BTC/USDT",
            status=ExecutionStatus.PENDING,
            target_quantity=Decimal("1.0"),
            filled_quantity=Decimal("0.0"),
            remaining_quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
            worst_price=Decimal("50000.0"),
            best_price=Decimal("50000.0"),
            expected_cost=Decimal("50000.0"),
            actual_cost=Decimal("0.0"),
            slippage_bps=Decimal("0.0"),
            slippage_amount=Decimal("0.0"),
            fill_rate=Decimal("0.0"),
            execution_time=0,
            num_fills=0,
            num_orders=1,
            total_fees=Decimal("0.0"),
            maker_fees=Decimal("0.0"),
            taker_fees=Decimal("0.0"),
            started_at=datetime.now(timezone.utc)
        )
        
        mock_execution_result = ExecutionResultWrapper(
            core_result=mock_core_result,
            algorithm=ExecutionAlgorithm.LIMIT
        )
        
        mock_orchestration_service.execute_order.return_value = mock_execution_result

        # Mock the _convert_to_market_data method to avoid the validation issue
        with patch.object(execution_controller, '_convert_to_market_data') as mock_convert:
            mock_market_data = MagicMock()
            mock_convert.return_value = mock_market_data
            
            result = await execution_controller.execute_order(sample_order_request, sample_market_data)
            
            assert result["success"] is True
            assert result["execution_id"] == "exec_001" 
            assert result["status"] == "pending"
            # Verify the orchestration service was called
            assert mock_orchestration_service.execute_order.called

    @pytest.mark.asyncio
    async def test_cancel_execution(self, execution_controller, mock_orchestration_service):
        """Test canceling an execution."""
        execution_id = "exec_001"
        mock_orchestration_service.cancel_execution.return_value = True
        
        result = await execution_controller.cancel_execution(execution_id)
        
        assert result["success"] is True
        assert result["cancelled"] is True
        assert result["execution_id"] == execution_id
        mock_orchestration_service.cancel_execution.assert_called_once_with(
            execution_id=execution_id, reason="user_request"
        )

    @pytest.mark.asyncio 
    async def test_get_execution_metrics(self, execution_controller, mock_orchestration_service):
        """Test getting execution metrics."""
        mock_metrics = {
            "total_executions": 10,
            "success_rate": 0.95,
            "average_execution_time": 1.5
        }
        mock_orchestration_service.get_comprehensive_metrics.return_value = mock_metrics
        
        result = await execution_controller.get_execution_metrics(bot_id="bot_001")
        
        assert result["success"] is True
        assert result["data"] == mock_metrics
        mock_orchestration_service.get_comprehensive_metrics.assert_called_once_with(
            bot_id="bot_001", symbol=None, time_range_hours=24
        )

    @pytest.mark.asyncio
    async def test_execute_order_error_handling(self, execution_controller, mock_orchestration_service, sample_order_request, sample_market_data):
        """Test error handling in execute_order."""
        from unittest.mock import patch
        mock_orchestration_service.execute_order.side_effect = Exception("Execution failed")
        
        # Mock the _convert_to_market_data method to avoid validation issues
        with patch.object(execution_controller, '_convert_to_market_data') as mock_convert:
            mock_market_data = MagicMock()
            mock_convert.return_value = mock_market_data
            
            result = await execution_controller.execute_order(sample_order_request, sample_market_data)
            
            assert result["success"] is False
            assert result["error"] == "internal_error"
            assert result["message"] == "An unexpected error occurred"

    @pytest.mark.asyncio
    async def test_cancel_execution_error_handling(self, execution_controller, mock_orchestration_service):
        """Test error handling in cancel_execution."""
        mock_orchestration_service.cancel_execution.side_effect = Exception("Cancel failed")
        
        result = await execution_controller.cancel_execution("exec_001")
        
        assert result["success"] is False
        assert "Cancel failed" in result["error"]

    def test_controller_properties(self, execution_controller, mock_execution_service, mock_orchestration_service):
        """Test controller properties are accessible."""
        assert hasattr(execution_controller, 'execution_service')
        assert hasattr(execution_controller, 'orchestration_service')
        assert execution_controller.execution_service is mock_execution_service
        assert execution_controller.orchestration_service is mock_orchestration_service