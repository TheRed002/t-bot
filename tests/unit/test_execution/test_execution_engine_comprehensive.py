"""
Optimized comprehensive unit tests for ExecutionEngine.

These tests validate critical execution paths, error handling, and integration
with the ExecutionService layer for financial trading operations.
"""

import asyncio
import os
import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from src.core.config import Config
from src.core.exceptions import ExecutionError, ServiceError, ValidationError
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionStatus,
    MarketData,
    OrderSide,
    OrderType,
)

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Create fixed datetime instances to avoid repeated datetime.now() calls
FIXED_DATETIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FIXED_TIMESTAMP = FIXED_DATETIME.timestamp()

# Pre-defined constants for faster test data creation
TEST_DECIMALS = {
    "ZERO": Decimal("0.0"),
    "ONE": Decimal("1.0"),
    "TEN": Decimal("10.0"),
    "PRICE_50K": Decimal("50000.0"),
    "FEE_25": Decimal("25.0"),
    "VOLUME_100": Decimal("100.0"),
    "VOLUME_5M": Decimal("5000000.0"),
    "PRICE_49950": Decimal("49950.0"),
    "PRICE_50100": Decimal("50100.0"),
    "PRICE_49900": Decimal("49900.0"),
    "PRICE_49999": Decimal("49999.0"),
    "PRICE_50001": Decimal("50001.0")
}

# Cache common mock configurations
COMMON_MOCK_ATTRS = {
    "symbol": "BTC/USDT",
    "side": OrderSide.BUY,
    "quantity": TEST_DECIMALS["ONE"],
    "price": TEST_DECIMALS["PRICE_50K"]
}
from src.execution.execution_engine import ExecutionEngine
from src.execution.service import ExecutionService
from src.execution.types import ExecutionInstruction


@pytest.fixture(scope="session")
def mock_config():
    """Mock configuration for tests."""
    config = Mock(spec=Config)
    config.execution = Mock()
    config.execution.max_slippage = Decimal("0.005")  # 0.5%
    config.execution.timeout = 30
    config.execution.retry_attempts = 3
    config.execution.get = Mock(side_effect=lambda key, default: default)
    config.error_handling = Mock()
    # Add all required attributes
    config.database = Mock()
    config.monitoring = Mock()
    config.redis = Mock()
    return config


@pytest.fixture(scope="session")
def mock_execution_service():
    """Mock ExecutionService for tests."""
    service = Mock(spec=ExecutionService)
    service.create_execution_record = AsyncMock(return_value="exec_123")
    service.update_execution_status = AsyncMock()
    service.get_execution_history = AsyncMock(return_value=[])
    service.validate_execution_request = AsyncMock(return_value=True)
    service.get_execution_status = AsyncMock(return_value=ExecutionStatus.COMPLETED)
    service.record_execution_metrics = AsyncMock()
    service.validate_order_pre_execution = AsyncMock(return_value={"overall_result": "passed", "errors": []})
    service.record_trade_execution = AsyncMock()
    service.get_execution_metrics = AsyncMock(return_value={
        "total_executions": 10,
        "successful_executions": 9,
        "recent_executions": []
    })
    return service


@pytest.fixture(scope="session") 
def mock_risk_service():
    """Mock RiskService for tests."""
    service = Mock()
    service.validate_trade = AsyncMock(return_value=True)
    service.calculate_position_size = AsyncMock(return_value=Decimal("1.0"))
    service.check_risk_limits = AsyncMock(return_value=True)
    service.assess_market_risk = AsyncMock(return_value=Decimal("0.01"))
    service.validate_signal = AsyncMock(return_value=True)
    return service


@pytest.fixture(scope="session")
def execution_engine(mock_execution_service, mock_risk_service, mock_config):
    """Create ExecutionEngine instance for testing."""
    # Mock all required dependencies
    mock_order_manager = Mock()
    mock_order_manager.submit_order = AsyncMock(return_value=Mock())
    mock_order_manager.cancel_order = AsyncMock(return_value=True)
    mock_order_manager.get_order_status = AsyncMock(return_value="filled")
    
    mock_slippage_model = Mock()
    mock_slippage_model.calculate_slippage = Mock(return_value=Decimal("0.001"))
    mock_slippage_model.estimate_execution_price = Mock(return_value=Decimal("50000"))
    mock_slippage_model.calculate_expected_slippage = Mock(return_value=Decimal("5"))
    mock_slippage_model.predict_slippage = AsyncMock()
    
    mock_cost_analyzer = Mock()
    mock_cost_analyzer.calculate_execution_cost = AsyncMock(return_value=Decimal("10"))
    mock_cost_analyzer.analyze_execution_cost = Mock(return_value={
        "total_cost": TEST_DECIMALS["PRICE_50K"],
        "fees": TEST_DECIMALS["FEE_25"],
        "market_impact": TEST_DECIMALS["TEN"],
        "timing_cost": Decimal("5.0"),
    })
    
    mock_algorithms = {
        ExecutionAlgorithm.TWAP: Mock(),
        ExecutionAlgorithm.VWAP: Mock(), 
        ExecutionAlgorithm.ICEBERG: Mock(),
        ExecutionAlgorithm.SMART_ROUTER: Mock()
    }
    
    # Mock all algorithm executions
    for algo in mock_algorithms.values():
        algo.execute = AsyncMock()
        algo.cancel_execution = AsyncMock(return_value=True)
    
    # Create simple pass-through decorator
    def passthrough_decorator(**kwargs):
        def decorator(func):
            return func
        return decorator
    
    def simple_passthrough(func):
        return func
    
    with patch.multiple(
        "src.execution.execution_engine",
        get_tracer=Mock(return_value=Mock()),
        with_circuit_breaker=passthrough_decorator,
        with_error_context=passthrough_decorator, 
        with_retry=passthrough_decorator,
        log_calls=simple_passthrough,
        time_execution=simple_passthrough
    ):
        engine = ExecutionEngine(
            mock_execution_service,
            mock_risk_service,
            mock_config,
            order_manager=mock_order_manager,
            slippage_model=mock_slippage_model,
            cost_analyzer=mock_cost_analyzer,
            algorithms=mock_algorithms
        )
        return engine


@pytest.fixture(scope="session")
def sample_execution_instruction():
    """Sample execution instruction for tests using pre-defined constants."""
    from src.core.types import OrderRequest

    order = OrderRequest(
        symbol=COMMON_MOCK_ATTRS["symbol"],
        side=COMMON_MOCK_ATTRS["side"],
        quantity=COMMON_MOCK_ATTRS["quantity"],
        order_type=OrderType.LIMIT,
        price=COMMON_MOCK_ATTRS["price"],
    )
    return ExecutionInstruction(
        order=order, algorithm=ExecutionAlgorithm.TWAP, max_slippage_bps=TEST_DECIMALS["TEN"]
    )


@pytest.fixture(scope="session")
def sample_market_data():
    """Sample market data for tests using pre-defined constants."""
    return MarketData(
        symbol="BTC/USDT",
        open=TEST_DECIMALS["PRICE_49950"],
        high=TEST_DECIMALS["PRICE_50100"],
        low=TEST_DECIMALS["PRICE_49900"],
        close=TEST_DECIMALS["PRICE_50K"],
        volume=TEST_DECIMALS["VOLUME_100"],
        quote_volume=TEST_DECIMALS["VOLUME_5M"],
        timestamp=FIXED_DATETIME,
        exchange="binance",
        bid_price=TEST_DECIMALS["PRICE_49999"],
        ask_price=TEST_DECIMALS["PRICE_50001"],
    )


def create_mock_execution_result(execution_id="test_123", symbol="BTC/USDT", status=ExecutionStatus.COMPLETED):
    """Helper to create mock execution results quickly using pre-defined constants."""
    from src.core.types import ExecutionResult as CoreExecutionResult
    from src.execution.execution_result_wrapper import ExecutionResultWrapper
    from src.core.types import OrderRequest
    
    # Use lightweight mock without spec for better performance
    mock_core_result = Mock()
    mock_core_result.instruction_id = execution_id
    mock_core_result.symbol = symbol
    mock_core_result.status = status
    mock_core_result.target_quantity = TEST_DECIMALS["ONE"]
    mock_core_result.filled_quantity = TEST_DECIMALS["ONE"]
    mock_core_result.remaining_quantity = TEST_DECIMALS["ZERO"]
    mock_core_result.average_price = TEST_DECIMALS["PRICE_50K"]
    mock_core_result.worst_price = TEST_DECIMALS["PRICE_50K"]
    mock_core_result.best_price = TEST_DECIMALS["PRICE_50K"]
    mock_core_result.expected_cost = TEST_DECIMALS["PRICE_50K"]
    mock_core_result.actual_cost = TEST_DECIMALS["PRICE_50K"]
    mock_core_result.slippage_bps = 0.0
    mock_core_result.slippage_amount = TEST_DECIMALS["ZERO"]
    mock_core_result.fill_rate = 1.0
    mock_core_result.execution_time = 2
    mock_core_result.num_fills = 1
    mock_core_result.num_orders = 1
    mock_core_result.total_fees = TEST_DECIMALS["FEE_25"]
    mock_core_result.maker_fees = TEST_DECIMALS["ZERO"]
    mock_core_result.taker_fees = TEST_DECIMALS["FEE_25"]
    mock_core_result.started_at = FIXED_DATETIME
    mock_core_result.completed_at = FIXED_DATETIME
    mock_core_result.fills = []
    mock_core_result.metadata = {}
    
    mock_order = Mock()
    mock_order.symbol = symbol
    mock_order.side = COMMON_MOCK_ATTRS["side"]
    mock_order.quantity = TEST_DECIMALS["ONE"]
    
    return ExecutionResultWrapper(
        core_result=mock_core_result,
        original_order=mock_order,
        algorithm=ExecutionAlgorithm.TWAP,
    )


class TestExecutionEngineInitialization:
    """Test ExecutionEngine initialization and setup."""

    def test_initialization_with_valid_dependencies(
        self, mock_execution_service, mock_risk_service, mock_config
    ):
        """Test successful initialization with all dependencies."""
        # Use lightweight mocks with minimal configuration
        mock_dependencies = {
            'order_manager': Mock(),
            'slippage_model': Mock(),
            'cost_analyzer': Mock(),
            'algorithms': {ExecutionAlgorithm.TWAP: Mock()}
        }
        
        # Batch patch for better performance
        patches = {
            "get_tracer": Mock(return_value=Mock()),
            "with_circuit_breaker": lambda **kwargs: lambda func: func,
            "with_error_context": lambda **kwargs: lambda func: func, 
            "with_retry": lambda **kwargs: lambda func: func,
            "log_calls": lambda func: func,
            "time_execution": lambda func: func
        }
        
        with patch.multiple("src.execution.execution_engine", **patches):
            engine = ExecutionEngine(
                mock_execution_service,
                mock_risk_service,
                mock_config,
                **mock_dependencies
            )

            # Batch assertions for better performance
            assertions = [
                (engine.execution_service, mock_execution_service),
                (engine.config, mock_config),
                (bool(engine.order_manager), True),
                (bool(engine.algorithms), True),
                (len(engine.algorithms) > 0, True)
            ]
            
            for actual, expected in assertions:
                assert actual == expected

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
    async def test_valid_execution_request(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test validation of valid execution request."""
        # Mock engine as running
        execution_engine._is_running = True
        
        # Mock slippage model
        from src.core.types import SlippageMetrics
        mock_slippage = Mock()
        mock_slippage.total_slippage_bps = Decimal("5")
        execution_engine.slippage_model.predict_slippage.return_value = mock_slippage

        # Mock algorithm execution
        mock_result = create_mock_execution_result()
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.return_value = mock_result

        result = await execution_engine.execute_order(
            sample_execution_instruction, sample_market_data
        )
        assert result is not None
        execution_engine.execution_service.validate_order_pre_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_execution_request_zero_quantity(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test validation fails for zero quantity."""
        execution_engine._is_running = True

        # Create a new order with zero quantity
        from src.core.types import OrderRequest, OrderSide, OrderType
        bad_order = Mock()
        bad_order.symbol = "BTC/USDT"
        bad_order.side = OrderSide.BUY
        bad_order.order_type = OrderType.LIMIT
        bad_order.quantity = Decimal("0")
        bad_order.price = Decimal("50000.0")
        sample_execution_instruction.order = bad_order

        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "failed",
            "errors": ["Zero quantity not allowed"],
        }

        from src.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="Zero quantity not allowed"):
            await execution_engine.execute_order(
                sample_execution_instruction, sample_market_data
            )

    @pytest.mark.asyncio
    async def test_invalid_execution_request_negative_price(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test validation fails for negative price."""
        execution_engine._is_running = True

        # Create a new order with negative price
        from src.core.types import OrderRequest, OrderSide, OrderType
        bad_order = Mock()
        bad_order.symbol = "BTC/USDT"
        bad_order.side = OrderSide.BUY
        bad_order.order_type = OrderType.LIMIT
        bad_order.quantity = Decimal("1.0")
        bad_order.price = Decimal("-100")
        sample_execution_instruction.order = bad_order

        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "failed",
            "errors": ["Negative price not allowed"],
        }

        with pytest.raises(ValidationError):
            await execution_engine.execute_order(
                sample_execution_instruction, sample_market_data
            )

    @pytest.mark.asyncio
    async def test_invalid_execution_request_excessive_slippage(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test validation fails for excessive slippage tolerance."""
        execution_engine._is_running = True

        sample_execution_instruction.max_slippage_bps = Decimal("1000")  # 10%
        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "failed",
            "errors": ["Excessive slippage tolerance"],
        }

        with pytest.raises(ValidationError):
            await execution_engine.execute_order(
                sample_execution_instruction, sample_market_data
            )


class TestOrderExecution:
    """Test order execution functionality."""

    @pytest.mark.asyncio
    async def test_successful_market_order_execution(self, execution_engine, sample_market_data):
        """Test successful execution of a market order."""
        from src.core.types import OrderRequest

        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )
        instruction = ExecutionInstruction(order=order, algorithm=ExecutionAlgorithm.TWAP)

        execution_engine._is_running = True
        
        # Mock validation to pass
        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "passed", 
            "errors": []
        }

        # Mock algorithm execution
        expected_result = create_mock_execution_result()
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.return_value = expected_result

        result = await execution_engine.execute_order(instruction, sample_market_data)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.total_filled_quantity == Decimal("1.0")
        assert result.average_fill_price == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_successful_limit_order_execution(self, execution_engine, sample_market_data):
        """Test successful execution of a limit order."""
        from src.core.types import OrderRequest

        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.LIMIT,
            price=Decimal("49500.0"),
        )
        instruction = ExecutionInstruction(order=order, algorithm=ExecutionAlgorithm.VWAP)

        execution_engine._is_running = True
        
        # Mock validation to pass
        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "passed", 
            "errors": []
        }

        # Mock algorithm execution - create custom result with different price
        expected_result = create_mock_execution_result("test_124", "BTC/USDT", ExecutionStatus.COMPLETED)
        expected_result._core.average_price = Decimal("49500.0")
        execution_engine.algorithms[ExecutionAlgorithm.VWAP].execute.return_value = expected_result

        result = await execution_engine.execute_order(instruction, sample_market_data)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.total_filled_quantity == Decimal("1.0")
        assert result.average_fill_price == Decimal("49500.0")

    @pytest.mark.asyncio
    async def test_partial_order_execution(self, execution_engine, sample_market_data):
        """Test handling of partially filled orders."""
        from src.core.types import OrderRequest

        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            order_type=OrderType.LIMIT,
            price=Decimal("50000.0"),
        )
        instruction = ExecutionInstruction(order=order, algorithm=ExecutionAlgorithm.ICEBERG)

        execution_engine._is_running = True

        # Mock algorithm execution - partial fill
        expected_result = create_mock_execution_result("test_125", "BTC/USDT", ExecutionStatus.PARTIAL)
        expected_result.total_filled_quantity = Decimal("3.5")
        execution_engine.algorithms[ExecutionAlgorithm.ICEBERG].execute.return_value = expected_result

        result = await execution_engine.execute_order(instruction, sample_market_data)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.total_filled_quantity == Decimal("3.5")


class TestSlippageHandling:
    """Test slippage calculation and handling."""

    @pytest.mark.asyncio
    async def test_acceptable_slippage(self, execution_engine, sample_market_data):
        """Test execution proceeds with acceptable slippage."""
        from src.core.types import OrderRequest

        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )
        instruction = ExecutionInstruction(
            order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            max_slippage_bps=Decimal("20"),
        )

        execution_engine._is_running = True
        
        # Mock validation to pass
        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "passed", 
            "errors": []
        }

        # Mock algorithm execution with slight slippage
        expected_result = create_mock_execution_result("test_126", "BTC/USDT", ExecutionStatus.COMPLETED)
        expected_result._core.average_price = Decimal("50050.0")
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.return_value = expected_result

        result = await execution_engine.execute_order(instruction, sample_market_data)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.average_fill_price == Decimal("50050.0")

    @pytest.mark.asyncio
    async def test_excessive_slippage_rejection(self, execution_engine, sample_market_data):
        """Test execution is rejected due to excessive slippage."""
        from src.core.types import OrderRequest

        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )
        instruction = ExecutionInstruction(
            order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            max_slippage_bps=Decimal("10"),
        )

        execution_engine._is_running = True
        
        # Mock validation to fail due to excessive slippage
        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "failed",
            "errors": ["Excessive slippage tolerance"],
        }

        # Mock slippage model to detect high slippage
        execution_engine.slippage_model.calculate_expected_slippage.return_value = Decimal("50")

        with pytest.raises(ValidationError):
            await execution_engine.execute_order(instruction, sample_market_data)


class TestErrorHandling:
    """Test error handling in execution scenarios."""

    @pytest.mark.asyncio
    async def test_algorithm_execution_failure(
        self, execution_engine, sample_market_data, sample_execution_instruction
    ):
        """Test handling of algorithm execution failures."""
        execution_engine._is_running = True
        
        # Mock validation to pass
        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "passed", 
            "errors": []
        }

        # Mock algorithm to raise an exception
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.side_effect = ExecutionError(
            "Algorithm execution failed"
        )

        # Since algorithm execution fails, we expect an ExecutionError to be raised
        with pytest.raises(ExecutionError):
            await execution_engine.execute_order(
                sample_execution_instruction, sample_market_data
            )

    @pytest.mark.asyncio
    async def test_service_layer_failure(
        self, execution_engine, sample_market_data, sample_execution_instruction
    ):
        """Test handling of service layer failures."""
        execution_engine._is_running = True

        # Mock ExecutionService validation to fail with ServiceError
        execution_engine.execution_service.validate_order_pre_execution.side_effect = ServiceError(
            "Database connection failed"
        )

        # Service layer failures should raise ServiceError
        with pytest.raises(ServiceError):
            await execution_engine.execute_order(
                sample_execution_instruction, sample_market_data
            )

    @pytest.mark.asyncio
    async def test_timeout_handling(self, execution_engine, sample_market_data):
        """Test handling of execution timeouts."""
        from src.core.types import OrderRequest

        # Reset any side effects from previous tests
        execution_engine.execution_service.validate_order_pre_execution.side_effect = None
        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "passed", 
            "errors": []
        }

        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.LIMIT,
            price=Decimal("50000.0"),
        )
        instruction = ExecutionInstruction(order=order, algorithm=ExecutionAlgorithm.TWAP)

        execution_engine._is_running = True

        # Mock algorithm to timeout with fast execution
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.side_effect = asyncio.TimeoutError(
            "Algorithm execution timed out"
        )

        # Timeout should raise ExecutionError - mock time operations for speed
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError("Timeout")), \
             pytest.raises(ExecutionError, match="Execution timeout after 30 seconds"):
            await execution_engine.execute_order(instruction, sample_market_data)


class TestExecutionMetrics:
    """Test execution metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_execution_metrics_collection(
        self, execution_engine, sample_market_data, sample_execution_instruction
    ):
        """Test that execution metrics are properly collected."""
        execution_engine._is_running = True

        # Reset any side effects from previous tests
        execution_engine.execution_service.validate_order_pre_execution.side_effect = None
        execution_engine.execution_service.validate_order_pre_execution.return_value = {
            "overall_result": "passed", 
            "errors": []
        }

        # Mock slippage model
        from src.core.types import SlippageMetrics
        mock_slippage = Mock()
        mock_slippage.total_slippage_bps = Decimal("5")
        execution_engine.slippage_model.predict_slippage.return_value = mock_slippage

        # Reset algorithm side effects and set return value
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.side_effect = None
        expected_result = create_mock_execution_result("test_metrics")
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.return_value = expected_result

        # Reset mock call history before test execution
        execution_engine.execution_service.record_trade_execution.reset_mock()

        result = await execution_engine.execute_order(
            sample_execution_instruction, sample_market_data
        )

        # Verify service was called to record metrics
        execution_engine.execution_service.record_trade_execution.assert_called_once()

        # Verify result contains expected metrics
        assert result.execution_duration is not None
        assert result.total_filled_quantity == Decimal("1.0")
        assert result.average_fill_price == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_cost_analysis(
        self, execution_engine, sample_market_data, sample_execution_instruction
    ):
        """Test execution cost analysis."""
        execution_engine._is_running = True

        # Mock slippage model
        from src.core.types import SlippageMetrics
        mock_slippage = Mock()
        mock_slippage.total_slippage_bps = Decimal("5")
        execution_engine.slippage_model.predict_slippage.return_value = mock_slippage

        # Mock algorithm execution
        expected_result = create_mock_execution_result("test_cost")
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.return_value = expected_result

        result = await execution_engine.execute_order(
            sample_execution_instruction, sample_market_data
        )

        assert result.total_filled_quantity == Decimal("1.0")
        assert result.average_fill_price == Decimal("50000.0")


class TestExecutionHistory:
    """Test execution history and audit trail."""

    @pytest.mark.asyncio
    async def test_execution_history_retrieval(self, execution_engine):
        """Test retrieval of execution history through metrics."""
        mock_metrics = {
            "service_metrics": {
                "total_executions": 10,
                "successful_executions": 9,
                "recent_executions": [
                    {
                        "id": 1,
                        "symbol": "BTC/USDT",
                        "side": "BUY",
                        "quantity": Decimal("1.0"),
                        "price": Decimal("50000.0"),
                        "status": "COMPLETED",
                        "timestamp": datetime.now(timezone.utc),
                    }
                ],
            },
            "engine_metrics": {
                "engine_status": "running",
                "active_executions": 0,
                "algorithms_available": 4,
            },
        }

        execution_engine.execution_service.get_execution_metrics.return_value = mock_metrics["service_metrics"]

        metrics = await execution_engine.get_execution_metrics()

        assert metrics["service_metrics"]["total_executions"] == 10
        assert len(metrics["service_metrics"]["recent_executions"]) == 1
        assert metrics["service_metrics"]["recent_executions"][0]["symbol"] == "BTC/USDT"
        execution_engine.execution_service.get_execution_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_audit_trail(
        self, execution_engine, sample_market_data, sample_execution_instruction
    ):
        """Test that all executions create proper audit trail."""
        execution_engine._is_running = True

        # Mock slippage model
        from src.core.types import SlippageMetrics
        mock_slippage = Mock()
        mock_slippage.total_slippage_bps = Decimal("5")
        execution_engine.slippage_model.predict_slippage.return_value = mock_slippage

        # Mock algorithm execution
        expected_result = create_mock_execution_result("test_audit")
        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.return_value = expected_result

        # Reset mock call history before test execution
        execution_engine.execution_service.record_trade_execution.reset_mock()

        await execution_engine.execute_order(sample_execution_instruction, sample_market_data)

        # Verify audit trail creation
        execution_engine.execution_service.record_trade_execution.assert_called_once()
        call_args = execution_engine.execution_service.record_trade_execution.call_args
        assert call_args is not None
        assert call_args.kwargs["execution_result"] == expected_result


class TestConcurrentExecution:
    """Test concurrent execution scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_executions(self, execution_engine, sample_market_data):
        """Test handling of multiple concurrent execution requests."""
        # Pre-create order template for reuse
        from src.core.types import OrderRequest
        base_order = OrderRequest(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=TEST_DECIMALS["ONE"],
            order_type=OrderType.MARKET,
        )

        # Create instructions using template
        instructions = [ExecutionInstruction(order=base_order, algorithm=ExecutionAlgorithm.TWAP) for _ in range(2)]
        execution_engine._is_running = True

        # Pre-create expected results
        expected_results = [
            create_mock_execution_result(f"concurrent_{i}", "ETH/USDT", ExecutionStatus.COMPLETED)
            for i in range(2)
        ]
        # Batch update core prices
        for result in expected_results:
            result._core.average_price = Decimal("3000.0")

        execution_engine.algorithms[ExecutionAlgorithm.TWAP].execute.side_effect = expected_results

        # Execute concurrently - mock timeout for faster execution
        async def mock_wait_for(coro, timeout):
            return await coro
        
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            results = await asyncio.gather(*[
                execution_engine.execute_order(instruction, sample_market_data) 
                for instruction in instructions
            ])

        assert len(results) == 2
        assert all(result.status == ExecutionStatus.COMPLETED for result in results)
        assert all(result.total_filled_quantity == TEST_DECIMALS["ONE"] for result in results)

    @pytest.mark.asyncio
    async def test_execution_queue_management(self, execution_engine):
        """Test execution queue management under load."""
        assert len(execution_engine.active_executions) == 0

        # Create lightweight mock execution using pre-defined constants
        from src.execution.execution_result_wrapper import ExecutionResultWrapper

        mock_core_result = Mock(
            instruction_id="queue_test_1",
            symbol=COMMON_MOCK_ATTRS["symbol"],
            status=ExecutionStatus.RUNNING,
            filled_quantity=TEST_DECIMALS["ZERO"],
            total_fees=TEST_DECIMALS["ZERO"]
        )

        mock_order = Mock(
            symbol=COMMON_MOCK_ATTRS["symbol"],
            side=COMMON_MOCK_ATTRS["side"],
            quantity=COMMON_MOCK_ATTRS["quantity"]
        )

        execution_result = ExecutionResultWrapper(
            core_result=mock_core_result,
            original_order=mock_order,
            algorithm=ExecutionAlgorithm.TWAP,
        )

        # Simulate active execution
        execution_engine.active_executions["queue_test_1"] = execution_result

        # Batch assertions for better performance
        queue_depth = len(execution_engine.active_executions)
        active = await execution_engine.get_active_executions()
        
        assert queue_depth == 1
        assert len(active) == 1
        assert "queue_test_1" in active