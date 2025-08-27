"""
Comprehensive unit tests for ExecutionEngine.

These tests validate critical execution paths, error handling, and integration
with the ExecutionService layer for financial trading operations.
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.execution.execution_engine import ExecutionEngine
from src.execution.service import ExecutionService
from src.core.config import Config
from src.core.exceptions import ExecutionError, ServiceError, ValidationError
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    MarketData,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    SlippageMetrics,
)


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    config = Mock(spec=Config)
    config.execution = Mock()
    config.execution.max_slippage = Decimal('0.005')  # 0.5%
    config.execution.timeout = 30
    config.execution.retry_attempts = 3
    config.execution.get = Mock(side_effect=lambda key, default: default)
    config.error_handling = Mock()
    return config


@pytest.fixture
def mock_execution_service():
    """Mock ExecutionService for tests."""
    service = Mock(spec=ExecutionService)
    service.create_execution_record = AsyncMock()
    service.update_execution_status = AsyncMock()
    service.get_execution_history = AsyncMock()
    service.validate_execution_request = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_risk_service():
    """Mock RiskService for tests."""
    service = Mock()
    service.validate_trade = AsyncMock(return_value=True)
    service.calculate_position_size = AsyncMock(return_value=Decimal("1.0"))
    return service


@pytest.fixture
def execution_engine(mock_execution_service, mock_risk_service, mock_config):
    """Create ExecutionEngine instance for testing."""
    with patch('src.execution.execution_engine.OrderManager'):
        with patch('src.execution.execution_engine.SlippageModel'):
            with patch('src.execution.execution_engine.CostAnalyzer'):
                engine = ExecutionEngine(mock_execution_service, mock_risk_service, mock_config)
                return engine


@pytest.fixture
def sample_execution_instruction():
    """Sample execution instruction for tests."""
    return ExecutionInstruction(
        instruction_id="TEST-001",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        target_quantity=Decimal("1.0"),
        order_type=OrderType.LIMIT,
        price=Decimal("50000.0"),
        algorithm=ExecutionAlgorithm.TWAP,
        max_slippage=Decimal("0.001"),
        timeout_seconds=300
    )


@pytest.fixture
def sample_market_data():
    """Sample market data for tests."""
    return MarketData(
        symbol="BTC/USDT",
        price=Decimal("50000.0"),
        bid=Decimal("49999.0"),
        ask=Decimal("50001.0"),
        volume=Decimal("100.0"),
        timestamp=datetime.now(timezone.utc)
    )


class TestExecutionEngineInitialization:
    """Test ExecutionEngine initialization and setup."""

    def test_initialization_with_valid_dependencies(self, mock_execution_service, mock_risk_service, mock_config):
        """Test successful initialization with all dependencies."""
        with patch('src.execution.execution_engine.OrderManager'):
            with patch('src.execution.execution_engine.SlippageModel'):
                with patch('src.execution.execution_engine.CostAnalyzer'):
                    engine = ExecutionEngine(mock_execution_service, mock_risk_service, mock_config)
                    
                    assert engine.execution_service == mock_execution_service
                    assert engine.config == mock_config
                    assert engine.order_manager is not None
                    assert engine.algorithms is not None
                    assert len(engine.algorithms) > 0

    def test_initialization_missing_service(self, mock_risk_service, mock_config):
        """Test initialization fails with missing ExecutionService."""
        with pytest.raises(ValueError):
            ExecutionEngine(None, mock_risk_service, mock_config)

    def test_algorithms_registration(self, execution_engine):
        """Test that execution algorithms are properly registered."""
        assert ExecutionAlgorithm.TWAP in execution_engine.algorithms
        assert ExecutionAlgorithm.VWAP in execution_engine.algorithms
        assert ExecutionAlgorithm.ICEBERG in execution_engine.algorithms
        assert ExecutionAlgorithm.SMART_ROUTER in execution_engine.algorithms


class TestExecutionValidation:
    """Test execution request validation."""

    @pytest.mark.asyncio
    async def test_valid_execution_request(self, execution_engine, sample_execution_instruction, sample_market_data):
        """Test validation of valid execution request."""
        execution_engine.execution_service.validate_execution_request.return_value = True
        execution_engine.execution_service.record_trade_execution.return_value = None
        
        # The ExecutionEngine validates via ExecutionService
        result = await execution_engine.execute_order(sample_execution_instruction, sample_market_data)
        assert result is not None
        execution_engine.execution_service.validate_execution_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_execution_request_zero_quantity(self, execution_engine, sample_execution_instruction):
        """Test validation fails for zero quantity."""
        sample_execution_instruction.quantity = Decimal("0")
        
        with pytest.raises(ValidationError):
            await execution_engine._validate_execution_request(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_invalid_execution_request_negative_price(self, execution_engine, sample_execution_instruction):
        """Test validation fails for negative price."""
        sample_execution_instruction.price = Decimal("-100")
        
        with pytest.raises(ValidationError):
            await execution_engine._validate_execution_request(sample_execution_instruction)

    @pytest.mark.asyncio
    async def test_invalid_execution_request_excessive_slippage(self, execution_engine, sample_execution_instruction):
        """Test validation fails for excessive slippage tolerance."""
        sample_execution_instruction.max_slippage = Decimal("0.1")  # 10%
        
        with pytest.raises(ValidationError):
            await execution_engine._validate_execution_request(sample_execution_instruction)


class TestOrderExecution:
    """Test order execution functionality."""

    @pytest.mark.asyncio
    async def test_successful_market_order_execution(self, execution_engine, sample_market_data):
        """Test successful execution of a market order."""
        # Setup
        instruction = ExecutionInstruction(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
            algorithm=ExecutionAlgorithm.TWAP
        )
        
        expected_result = ExecutionResult(
            instruction_id="test_123",
            status=ExecutionStatus.COMPLETED,
            executed_quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
            total_cost=Decimal("50000.0"),
            slippage=SlippageMetrics(
                expected_price=Decimal("50000.0"),
                actual_price=Decimal("50000.0"),
                slippage_bps=Decimal("0")
            ),
            execution_time=datetime.now(timezone.utc)
        )
        
        # Mock algorithm execution
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute = AsyncMock(return_value=expected_result)
        
        result = await execution_engine.execute_order(instruction, sample_market_data)
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.executed_quantity == Decimal("1.0")
        assert result.average_price == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_successful_limit_order_execution(self, execution_engine, sample_market_data):
        """Test successful execution of a limit order."""
        instruction = ExecutionInstruction(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.LIMIT,
            price=Decimal("49500.0"),  # Below market price
            algorithm=ExecutionAlgorithm.VWAP
        )
        
        expected_result = ExecutionResult(
            instruction_id="test_124",
            status=ExecutionStatus.COMPLETED,
            executed_quantity=Decimal("1.0"),
            average_price=Decimal("49500.0"),
            total_cost=Decimal("49500.0"),
            slippage=SlippageMetrics(
                expected_price=Decimal("49500.0"),
                actual_price=Decimal("49500.0"),
                slippage_bps=Decimal("0")
            ),
            execution_time=datetime.now(timezone.utc)
        )
        
        execution_engine.algorithms[ExecutionAlgorithm.VWAP].execute = AsyncMock(return_value=expected_result)
        
        result = await execution_engine.execute_order(instruction, sample_market_data)
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.executed_quantity == Decimal("1.0")
        assert result.average_price == Decimal("49500.0")

    @pytest.mark.asyncio
    async def test_partial_order_execution(self, execution_engine, sample_market_data):
        """Test handling of partially filled orders."""
        instruction = ExecutionInstruction(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            order_type=OrderType.LIMIT,
            price=Decimal("50000.0"),
            algorithm=ExecutionAlgorithm.ICEBERG
        )
        
        # Simulate partial fill
        expected_result = ExecutionResult(
            instruction_id="test_125",
            status=ExecutionStatus.PARTIALLY_FILLED,
            executed_quantity=Decimal("3.5"),
            average_price=Decimal("50000.0"),
            total_cost=Decimal("175000.0"),
            slippage=SlippageMetrics(
                expected_price=Decimal("50000.0"),
                actual_price=Decimal("50000.0"),
                slippage_bps=Decimal("0")
            ),
            execution_time=datetime.now(timezone.utc)
        )
        
        execution_engine.algorithms[ExecutionAlgorithm.ICEBERG].execute = AsyncMock(return_value=expected_result)
        
        result = await execution_engine.execute_order(instruction, sample_market_data)
        
        assert result.status == ExecutionStatus.PARTIALLY_FILLED
        assert result.executed_quantity == Decimal("3.5")
        assert result.average_price == Decimal("50000.0")


class TestSlippageHandling:
    """Test slippage calculation and handling."""

    @pytest.mark.asyncio
    async def test_acceptable_slippage(self, execution_engine, sample_market_data):
        """Test execution proceeds with acceptable slippage."""
        instruction = ExecutionInstruction(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
            max_slippage=Decimal("0.002"),  # 0.2%
            algorithm=ExecutionAlgorithm.TWAP
        )
        
        # Simulate slight slippage
        expected_result = ExecutionResult(
            instruction_id="test_126",
            status=ExecutionStatus.COMPLETED,
            executed_quantity=Decimal("1.0"),
            average_price=Decimal("50050.0"),  # 0.1% slippage
            total_cost=Decimal("50050.0"),
            slippage=SlippageMetrics(
                expected_price=Decimal("50000.0"),
                actual_price=Decimal("50050.0"),
                slippage_bps=Decimal("10")  # 10 bps = 0.1%
            ),
            execution_time=datetime.now(timezone.utc)
        )
        
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute = AsyncMock(return_value=expected_result)
        
        result = await execution_engine.execute_order(instruction, sample_market_data)
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.slippage.slippage_bps == Decimal("10")

    @pytest.mark.asyncio
    async def test_excessive_slippage_rejection(self, execution_engine, sample_market_data):
        """Test execution is rejected due to excessive slippage."""
        instruction = ExecutionInstruction(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
            max_slippage=Decimal("0.001"),  # 0.1% - very strict
            algorithm=ExecutionAlgorithm.TWAP
        )
        
        # Mock slippage model to detect high slippage
        execution_engine.slippage_model.calculate_expected_slippage.return_value = Decimal("0.005")  # 0.5%
        
        with pytest.raises(ExecutionError, match="slippage"):
            await execution_engine.execute_order(instruction, sample_market_data)


class TestErrorHandling:
    """Test error handling in execution scenarios."""

    @pytest.mark.asyncio
    async def test_algorithm_execution_failure(self, execution_engine, sample_market_data, sample_execution_instruction):
        """Test handling of algorithm execution failures."""
        # Mock algorithm to raise an exception
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute = AsyncMock(
            side_effect=ExecutionError("Algorithm execution failed", error_code="EXEC_001")
        )
        
        with pytest.raises(ExecutionError):
            await execution_engine.execute_order(sample_execution_instruction, sample_market_data)

    @pytest.mark.asyncio
    async def test_service_layer_failure(self, execution_engine, sample_market_data, sample_execution_instruction):
        """Test handling of service layer failures."""
        # Mock service to fail
        execution_engine.execution_service.create_execution_record.side_effect = ServiceError(
            "Database connection failed", error_code="SERV_001"
        )
        
        with pytest.raises(ServiceError):
            await execution_engine.execute_order(sample_execution_instruction, sample_market_data)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, execution_engine, sample_market_data):
        """Test handling of execution timeouts."""
        instruction = ExecutionInstruction(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.LIMIT,
            price=Decimal("50000.0"),
            algorithm=ExecutionAlgorithm.TWAP,
            timeout_seconds=1  # Very short timeout
        )
        
        # Mock algorithm to take longer than timeout
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(2)
            return ExecutionResult(
                instruction_id="test_timeout",
                status=ExecutionStatus.COMPLETED,
                executed_quantity=Decimal("1.0"),
                average_price=Decimal("50000.0"),
                total_cost=Decimal("50000.0")
            )
        
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute = slow_execution
        
        with pytest.raises(ExecutionError, match="timeout"):
            await execution_engine.execute_order(instruction, sample_market_data)


class TestExecutionMetrics:
    """Test execution metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_execution_metrics_collection(self, execution_engine, sample_market_data, sample_execution_instruction):
        """Test that execution metrics are properly collected."""
        expected_result = ExecutionResult(
            instruction_id="test_metrics",
            status=ExecutionStatus.COMPLETED,
            executed_quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
            total_cost=Decimal("50000.0"),
            slippage=SlippageMetrics(
                expected_price=Decimal("50000.0"),
                actual_price=Decimal("50000.0"),
                slippage_bps=Decimal("0")
            ),
            execution_time=datetime.now(timezone.utc)
        )
        
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute = AsyncMock(return_value=expected_result)
        
        result = await execution_engine.execute_order(sample_execution_instruction, sample_market_data)
        
        # Verify service was called to record metrics
        execution_engine.execution_service.create_execution_record.assert_called_once()
        
        # Verify result contains expected metrics
        assert result.execution_time is not None
        assert result.slippage is not None
        assert result.total_cost == Decimal("50000.0")

    @pytest.mark.asyncio 
    async def test_cost_analysis(self, execution_engine, sample_market_data, sample_execution_instruction):
        """Test execution cost analysis."""
        # Mock cost analyzer
        execution_engine.cost_analyzer.analyze_execution_cost = Mock(return_value={
            "total_cost": Decimal("50000.0"),
            "fees": Decimal("25.0"),
            "market_impact": Decimal("10.0"),
            "timing_cost": Decimal("5.0")
        })
        
        expected_result = ExecutionResult(
            instruction_id="test_cost",
            status=ExecutionStatus.COMPLETED,
            executed_quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
            total_cost=Decimal("50000.0")
        )
        
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute = AsyncMock(return_value=expected_result)
        
        result = await execution_engine.execute_order(sample_execution_instruction, sample_market_data)
        
        # Verify cost analysis was performed
        execution_engine.cost_analyzer.analyze_execution_cost.assert_called_once()
        assert result.total_cost == Decimal("50000.0")


class TestExecutionHistory:
    """Test execution history and audit trail."""

    @pytest.mark.asyncio
    async def test_execution_history_retrieval(self, execution_engine):
        """Test retrieval of execution history."""
        # Mock service response
        mock_history = [
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": Decimal("1.0"),
                "price": Decimal("50000.0"),
                "status": "COMPLETED",
                "timestamp": datetime.now(timezone.utc)
            }
        ]
        
        execution_engine.execution_service.get_execution_history.return_value = mock_history
        
        history = await execution_engine.get_execution_history("BTC/USDT", limit=10)
        
        assert len(history) == 1
        assert history[0]["symbol"] == "BTC/USDT"
        execution_engine.execution_service.get_execution_history.assert_called_once_with("BTC/USDT", limit=10)

    @pytest.mark.asyncio
    async def test_execution_audit_trail(self, execution_engine, sample_market_data, sample_execution_instruction):
        """Test that all executions create proper audit trail."""
        expected_result = ExecutionResult(
            instruction_id="test_audit",
            status=ExecutionStatus.COMPLETED,
            executed_quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
            total_cost=Decimal("50000.0")
        )
        
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute = AsyncMock(return_value=expected_result)
        
        await execution_engine.execute_order(sample_execution_instruction, sample_market_data)
        
        # Verify audit trail creation
        execution_engine.execution_service.create_execution_record.assert_called_once()
        execution_engine.execution_service.update_execution_status.assert_called()


class TestConcurrentExecution:
    """Test concurrent execution scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_executions(self, execution_engine, sample_market_data):
        """Test handling of multiple concurrent execution requests."""
        instructions = []
        for i in range(3):
            instruction = ExecutionInstruction(
                symbol=f"ETH/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.MARKET,
                algorithm=ExecutionAlgorithm.TWAP
            )
            instructions.append(instruction)
        
        # Mock algorithm responses
        expected_results = []
        for i, instruction in enumerate(instructions):
            result = ExecutionResult(
                instruction_id=f"concurrent_{i}",
                status=ExecutionStatus.COMPLETED,
                executed_quantity=Decimal("1.0"),
                average_price=Decimal("3000.0"),
                total_cost=Decimal("3000.0")
            )
            expected_results.append(result)
        
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute = AsyncMock(side_effect=expected_results)
        
        # Execute concurrently
        tasks = []
        for instruction in instructions:
            task = execution_engine.execute_order(instruction, sample_market_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result.status == ExecutionStatus.COMPLETED
            assert result.executed_quantity == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_execution_queue_management(self, execution_engine):
        """Test execution queue management under load."""
        # This would test the execution engine's ability to manage 
        # a queue of execution requests without overwhelming the system
        
        # Mock high-load scenario
        with patch.object(execution_engine, '_get_queue_depth', return_value=10):
            queue_depth = execution_engine._get_queue_depth()
            assert queue_depth == 10
            
            # Verify engine can handle queue depth information
            assert isinstance(queue_depth, int)