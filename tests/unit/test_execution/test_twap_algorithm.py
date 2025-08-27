"""Unit tests for TWAP algorithm."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from src.core.config import Config
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
)
from src.execution.algorithms.twap import TWAPAlgorithm


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    config.execution = {"default_portfolio_value": "100000"}
    return config


@pytest.fixture
def twap_algorithm(config):
    """Create TWAP algorithm instance."""
    return TWAPAlgorithm(config)


@pytest.fixture
def sample_order_request():
    """Create sample order request."""
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("10.0"),
        price=Decimal("50000")
    )


@pytest.fixture
def sample_execution_instruction(sample_order_request):
    """Create sample execution instruction."""
    return ExecutionInstruction(
        order=sample_order_request,
        algorithm=ExecutionAlgorithm.TWAP,
        time_horizon_minutes=60,
        participation_rate=0.2,
        max_slices=10
    )


@pytest.fixture
def mock_exchange():
    """Create mock exchange."""
    exchange = AsyncMock()
    exchange.exchange_name = "binance"
    exchange.place_order = AsyncMock()
    exchange.get_order_status = AsyncMock()
    return exchange


@pytest.fixture
def mock_exchange_factory(mock_exchange):
    """Create mock exchange factory."""
    factory = AsyncMock()
    factory.get_exchange = AsyncMock(return_value=mock_exchange)
    return factory


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    manager = AsyncMock()
    manager.validate_order = AsyncMock(return_value=True)
    return manager


class TestTWAPAlgorithm:
    """Test cases for TWAP algorithm."""

    def test_initialization(self, twap_algorithm):
        """Test TWAP algorithm initialization."""
        assert twap_algorithm is not None
        assert twap_algorithm.get_algorithm_type() == ExecutionAlgorithm.TWAP
        assert not twap_algorithm.is_running
        assert twap_algorithm.total_executions == 0

    @pytest.mark.asyncio
    async def test_validate_algorithm_parameters_valid(self, twap_algorithm, sample_execution_instruction):
        """Test validation of valid TWAP parameters."""
        await twap_algorithm._validate_algorithm_parameters(sample_execution_instruction)
        # Should not raise any exception

    @pytest.mark.asyncio
    async def test_validate_algorithm_parameters_invalid_time_horizon(self, twap_algorithm, sample_execution_instruction):
        """Test validation with invalid time horizon."""
        sample_execution_instruction.time_horizon_minutes = -10
        
        with pytest.raises(Exception):
            await twap_algorithm._validate_algorithm_parameters(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_validate_algorithm_parameters_excessive_time_horizon(self, twap_algorithm, sample_execution_instruction):
        """Test validation with excessive time horizon."""
        sample_execution_instruction.time_horizon_minutes = 25 * 60  # 25 hours
        
        with pytest.raises(Exception):
            await twap_algorithm._validate_algorithm_parameters(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_validate_algorithm_parameters_invalid_participation_rate(self, twap_algorithm, sample_execution_instruction):
        """Test validation with invalid participation rate."""
        sample_execution_instruction.participation_rate = 1.5  # > 1.0
        
        with pytest.raises(Exception):
            await twap_algorithm._validate_algorithm_parameters(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_validate_algorithm_parameters_invalid_max_slices(self, twap_algorithm, sample_execution_instruction):
        """Test validation with invalid max slices."""
        sample_execution_instruction.max_slices = -5
        
        with pytest.raises(Exception):
            await twap_algorithm._validate_algorithm_parameters(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_validate_algorithm_parameters_excessive_max_slices(self, twap_algorithm, sample_execution_instruction):
        """Test validation with excessive max slices."""
        sample_execution_instruction.max_slices = 200  # > 100
        
        with pytest.raises(Exception):
            await twap_algorithm._validate_algorithm_parameters(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_validate_algorithm_parameters_invalid_slice_size(self, twap_algorithm, sample_execution_instruction):
        """Test validation with invalid slice size."""
        sample_execution_instruction.slice_size = Decimal("0")
        
        with pytest.raises(Exception):
            await twap_algorithm._validate_algorithm_parameters(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_validate_algorithm_parameters_slice_size_exceeds_quantity(self, twap_algorithm, sample_execution_instruction):
        """Test validation with slice size exceeding order quantity."""
        sample_execution_instruction.slice_size = Decimal("20.0")  # > order quantity of 10.0
        
        with pytest.raises(Exception):
            await twap_algorithm._validate_algorithm_parameters(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_create_execution_plan_default_parameters(self, twap_algorithm, sample_execution_instruction):
        """Test execution plan creation with default parameters."""
        plan = await twap_algorithm._create_execution_plan(sample_execution_instruction)
        
        assert plan["total_quantity"] == sample_execution_instruction.order.quantity
        assert plan["time_horizon_minutes"] == 60
        assert plan["participation_rate"] == 0.2
        assert len(plan["slices"]) <= 10  # max_slices
        assert all(slice_info["quantity"] > 0 for slice_info in plan["slices"])

    @pytest.mark.asyncio
    async def test_create_execution_plan_custom_parameters(self, twap_algorithm, sample_execution_instruction):
        """Test execution plan creation with custom parameters."""
        sample_execution_instruction.time_horizon_minutes = 30
        sample_execution_instruction.participation_rate = 0.15
        sample_execution_instruction.max_slices = 5
        
        plan = await twap_algorithm._create_execution_plan(sample_execution_instruction)
        
        assert plan["time_horizon_minutes"] == 30
        assert plan["participation_rate"] == 0.15
        assert len(plan["slices"]) <= 5

    @pytest.mark.asyncio
    async def test_create_execution_plan_fixed_slice_size(self, twap_algorithm, sample_execution_instruction):
        """Test execution plan creation with fixed slice size."""
        sample_execution_instruction.slice_size = Decimal("2.0")
        
        plan = await twap_algorithm._create_execution_plan(sample_execution_instruction)
        
        # With quantity 10.0 and slice size 2.0, should have 5 slices
        assert len(plan["slices"]) == 5
        assert all(slice_info["quantity"] == Decimal("2.0") for slice_info in plan["slices"])

    @pytest.mark.asyncio
    async def test_create_execution_plan_slice_timing(self, twap_algorithm, sample_execution_instruction):
        """Test execution plan slice timing."""
        plan = await twap_algorithm._create_execution_plan(sample_execution_instruction)
        
        # Check that slices have proper timing
        slices = plan["slices"]
        for i in range(1, len(slices)):
            current_time = slices[i]["execution_time"]
            previous_time = slices[i-1]["execution_time"]
            assert current_time > previous_time

    @pytest.mark.asyncio
    async def test_execute_successful(self, twap_algorithm, sample_execution_instruction, 
                                     mock_exchange_factory, mock_risk_manager):
        """Test successful TWAP execution."""
        # Mock order responses
        mock_order_responses = []
        for i in range(5):  # Assuming 5 slices
            response = OrderResponse(
                id=f"order_{i}",
                client_order_id=f"client_{i}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("2.0"),
                price=Decimal("50000"),
                filled_quantity=Decimal("2.0"),
                status="filled",
                timestamp=datetime.now(timezone.utc)
            )
            mock_order_responses.append(response)
        
        mock_exchange_factory.get_exchange.return_value.place_order.side_effect = mock_order_responses
        
        # Mock order status
        mock_exchange_factory.get_exchange.return_value.get_order_status.return_value = "filled"
        
        # Use a small time horizon to speed up test
        sample_execution_instruction.time_horizon_minutes = 1
        sample_execution_instruction.max_slices = 5
        
        with patch('asyncio.sleep', new_callable=AsyncMock):  # Mock sleep to speed up test
            result = await twap_algorithm.execute(
                sample_execution_instruction, mock_exchange_factory, mock_risk_manager
            )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.algorithm == ExecutionAlgorithm.TWAP
        assert result.total_filled_quantity == Decimal("10.0")
        assert len(result.child_orders) == 5

    @pytest.mark.asyncio
    async def test_execute_with_risk_manager_rejection(self, twap_algorithm, sample_execution_instruction, 
                                                      mock_exchange_factory, mock_risk_manager):
        """Test TWAP execution with risk manager rejecting orders."""
        mock_risk_manager.validate_order.return_value = False
        
        sample_execution_instruction.time_horizon_minutes = 1
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await twap_algorithm.execute(
                sample_execution_instruction, mock_exchange_factory, mock_risk_manager
            )
        
        # Should complete but with no fills due to risk rejections
        assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
        assert result.total_filled_quantity == Decimal("0")

    @pytest.mark.asyncio
    async def test_execute_exchange_error(self, twap_algorithm, sample_execution_instruction, 
                                         mock_exchange_factory, mock_risk_manager):
        """Test TWAP execution with exchange errors."""
        mock_exchange_factory.get_exchange.return_value.place_order.side_effect = Exception("Exchange error")
        
        sample_execution_instruction.time_horizon_minutes = 1
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await twap_algorithm.execute(
                sample_execution_instruction, mock_exchange_factory, mock_risk_manager
            )
            
            # When all slices fail, execution should complete with status="failed"
            assert result.status == ExecutionStatus.FAILED
            assert result.total_filled_quantity == Decimal("0")
            assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_execute_no_exchange_factory(self, twap_algorithm, sample_execution_instruction):
        """Test TWAP execution without exchange factory."""
        with pytest.raises(Exception):
            await twap_algorithm.execute(sample_execution_instruction, None, None)

    @pytest.mark.asyncio
    async def test_cancel_execution_running(self, twap_algorithm, sample_execution_instruction, 
                                           mock_exchange_factory, mock_risk_manager):
        """Test cancellation of running TWAP execution."""
        # Start an execution
        execution_result = await twap_algorithm._create_execution_result(sample_execution_instruction)
        execution_result.status = ExecutionStatus.RUNNING
        twap_algorithm.current_executions[execution_result.execution_id] = execution_result
        
        # Cancel the execution
        success = await twap_algorithm.cancel_execution(execution_result.execution_id)
        
        assert success is True
        assert execution_result.status == ExecutionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_execution_not_found(self, twap_algorithm):
        """Test cancellation of non-existent execution."""
        success = await twap_algorithm.cancel_execution("non_existent_id")
        assert success is False

    @pytest.mark.asyncio
    async def test_cancel_execution_already_completed(self, twap_algorithm, sample_execution_instruction):
        """Test cancellation of already completed execution."""
        execution_result = await twap_algorithm._create_execution_result(sample_execution_instruction)
        execution_result.status = ExecutionStatus.COMPLETED
        twap_algorithm.current_executions[execution_result.execution_id] = execution_result
        
        success = await twap_algorithm.cancel_execution(execution_result.execution_id)
        assert success is False

    @pytest.mark.asyncio
    async def test_validate_instruction_invalid_order(self, twap_algorithm):
        """Test instruction validation with invalid order."""
        # Creating ExecutionInstruction with None order will fail Pydantic validation
        # So we test that the Pydantic validation itself raises an error
        with pytest.raises(ValidationError):
            instruction = ExecutionInstruction(
                order=None,
                algorithm=ExecutionAlgorithm.TWAP
            )

    @pytest.mark.asyncio
    async def test_validate_instruction_missing_symbol(self, twap_algorithm):
        """Test instruction validation with missing symbol."""
        order = OrderRequest(
            symbol="",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        instruction = ExecutionInstruction(order=order, algorithm=ExecutionAlgorithm.TWAP)
        
        with pytest.raises(Exception):
            await twap_algorithm.validate_instruction(instruction)

    @pytest.mark.asyncio
    async def test_validate_instruction_invalid_quantity(self, twap_algorithm):
        """Test instruction validation with invalid quantity."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0")
        )
        instruction = ExecutionInstruction(order=order, algorithm=ExecutionAlgorithm.TWAP)
        
        with pytest.raises(Exception):
            await twap_algorithm.validate_instruction(instruction)

    @pytest.mark.asyncio
    async def test_get_algorithm_summary(self, twap_algorithm):
        """Test algorithm summary generation."""
        summary = await twap_algorithm.get_algorithm_summary()
        
        assert summary["algorithm_name"] == "TWAPAlgorithm"
        assert summary["algorithm_type"] == "twap"
        assert summary["total_executions"] == 0
        assert summary["successful_executions"] == 0
        assert summary["failed_executions"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["is_running"] is False
        assert summary["active_executions"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_completed_executions(self, twap_algorithm, sample_execution_instruction):
        """Test cleanup of completed executions."""
        # Create multiple completed executions
        for i in range(5):
            execution_result = await twap_algorithm._create_execution_result(sample_execution_instruction)
            execution_result.status = ExecutionStatus.COMPLETED
            twap_algorithm.current_executions[f"execution_{i}"] = execution_result
            
        # Create one running execution
        running_execution = await twap_algorithm._create_execution_result(sample_execution_instruction)
        running_execution.status = ExecutionStatus.RUNNING
        twap_algorithm.current_executions["running_execution"] = running_execution
        
        # Cleanup with max_history = 3
        await twap_algorithm.cleanup_completed_executions(max_history=3)
        
        # Should keep 3 completed + 1 running = 4 total
        assert len(twap_algorithm.current_executions) == 4
        assert "running_execution" in twap_algorithm.current_executions

    @pytest.mark.asyncio
    async def test_execution_result_creation(self, twap_algorithm, sample_execution_instruction):
        """Test execution result creation."""
        execution_result = await twap_algorithm._create_execution_result(sample_execution_instruction)
        
        assert execution_result.execution_id.startswith("twapalgorithm_")
        assert execution_result.original_order == sample_execution_instruction.order
        assert execution_result.algorithm == ExecutionAlgorithm.TWAP
        assert execution_result.status == ExecutionStatus.PENDING
        assert execution_result.total_filled_quantity == Decimal("0")

    @pytest.mark.asyncio
    async def test_execution_result_update_with_child_order(self, twap_algorithm, sample_execution_instruction):
        """Test execution result update with child order."""
        execution_result = await twap_algorithm._create_execution_result(sample_execution_instruction)
        
        child_order = OrderResponse(
            id="order_123",
            client_order_id="client_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
            filled_quantity=Decimal("2.0"),
            status="filled",
            timestamp=datetime.now(timezone.utc)
        )
        
        updated_result = await twap_algorithm._update_execution_result(
            execution_result,
            status=ExecutionStatus.RUNNING,
            child_order=child_order
        )
        
        assert updated_result.status == ExecutionStatus.RUNNING
        assert len(updated_result.child_orders) == 1
        assert updated_result.total_filled_quantity == Decimal("2.0")
        assert updated_result.average_fill_price == Decimal("50000")
        assert updated_result.number_of_trades == 1

    @pytest.mark.asyncio
    async def test_execution_result_multiple_fills(self, twap_algorithm, sample_execution_instruction):
        """Test execution result with multiple fills."""
        execution_result = await twap_algorithm._create_execution_result(sample_execution_instruction)
        
        # First fill
        child_order_1 = OrderResponse(
            id="order_1",
            client_order_id="client_1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
            filled_quantity=Decimal("2.0"),
            status="filled",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Second fill at different price
        child_order_2 = OrderResponse(
            id="order_2",
            client_order_id="client_2",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("3.0"),
            price=Decimal("50100"),
            filled_quantity=Decimal("3.0"),
            status="filled",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Update with both orders
        await twap_algorithm._update_execution_result(execution_result, child_order=child_order_1)
        await twap_algorithm._update_execution_result(execution_result, child_order=child_order_2)
        
        assert execution_result.total_filled_quantity == Decimal("5.0")
        assert execution_result.number_of_trades == 2
        
        # Check volume-weighted average price
        expected_avg = (Decimal("2.0") * Decimal("50000") + Decimal("3.0") * Decimal("50100")) / Decimal("5.0")
        assert execution_result.average_fill_price == expected_avg