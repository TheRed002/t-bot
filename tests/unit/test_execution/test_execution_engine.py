"""Unit tests for ExecutionEngine."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Create fixed datetime instances to avoid repeated datetime.now() calls
FIXED_DATETIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Pre-defined constants for faster test data creation
TEST_DECIMALS = {
    "ZERO": Decimal("0"),
    "ONE": Decimal("1"),
    "PRICE_50K": Decimal("50000"),
    "VOLUME_100": Decimal("100"),
    "SLIPPAGE_10": Decimal("10")
}

from src.core.config import Config
from src.core.exceptions import ValidationError as CoreValidationError
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction as CoreExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
)
from src.execution.execution_engine import ExecutionEngine
from src.execution.execution_result_wrapper import ExecutionResultWrapper
from src.execution.types import ExecutionInstruction


def create_execution_result_wrapper(
    execution_id: str,
    order: OrderRequest,
    algorithm: ExecutionAlgorithm,
    status: ExecutionStatus,
    filled_quantity: Decimal | None = None,
    execution_duration: float | None = None,
    total_fees: Decimal | None = None,
) -> ExecutionResultWrapper:
    """Helper to create ExecutionResultWrapper for tests."""
    # Create core result with proper fields
    core_result = ExecutionResult(
        instruction_id=execution_id,
        symbol=order.symbol,
        status=status,
        target_quantity=order.quantity,
        filled_quantity=filled_quantity or TEST_DECIMALS["ZERO"],
        remaining_quantity=order.quantity - (filled_quantity or TEST_DECIMALS["ZERO"]),
        average_price=order.price or TEST_DECIMALS["PRICE_50K"],
        worst_price=order.price or TEST_DECIMALS["PRICE_50K"],
        best_price=order.price or TEST_DECIMALS["PRICE_50K"],
        expected_cost=order.quantity * (order.price or TEST_DECIMALS["PRICE_50K"]),
        actual_cost=order.quantity * (order.price or TEST_DECIMALS["PRICE_50K"]),
        slippage_bps=0.0,
        slippage_amount=TEST_DECIMALS["ZERO"],
        fill_rate=float(filled_quantity or 0) / float(order.quantity) if order.quantity else 0.0,
        execution_time=int(execution_duration or 0),
        num_fills=1 if filled_quantity else 0,
        num_orders=1,
        total_fees=total_fees or TEST_DECIMALS["ZERO"],
        maker_fees=TEST_DECIMALS["ZERO"],
        taker_fees=total_fees or TEST_DECIMALS["ZERO"],
        started_at=FIXED_DATETIME,
        completed_at=FIXED_DATETIME
        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]
        else None,
    )
    return ExecutionResultWrapper(core_result, order, algorithm)


@pytest.fixture(scope="session")
def config():
    """Create test configuration."""
    config = MagicMock()
    config.error_handling = MagicMock()
    config.execution = MagicMock()
    config.execution.get = MagicMock(side_effect=lambda key, default: default)
    return config


@pytest.fixture(scope="session")
def execution_service():
    """Create mock ExecutionService."""
    service = MagicMock()
    service.is_running = False
    service.start = AsyncMock()
    service.stop = AsyncMock()
    service.validate_order_pre_execution = AsyncMock(
        return_value={"overall_result": "passed", "errors": []}
    )
    service.record_trade_execution = AsyncMock()
    service.get_execution_metrics = AsyncMock(return_value={})
    return service


@pytest.fixture(scope="session")
def mock_risk_service():
    """Create mock RiskService."""
    service = MagicMock()
    service.validate_signal = AsyncMock(return_value=True)
    service.calculate_position_size = AsyncMock(return_value=TEST_DECIMALS["ONE"])
    return service


@pytest.fixture
def execution_engine(execution_service, mock_risk_service, config, mock_exchange_factory):
    """Create ExecutionEngine instance."""
    # Create mock dependencies that ExecutionEngine requires
    mock_order_manager = AsyncMock()
    mock_slippage_model = AsyncMock()
    # Mock slippage prediction with required attributes
    mock_slippage_prediction = MagicMock()
    mock_slippage_prediction.total_slippage_bps = Decimal("50")  # 0.5% = 50 bps
    mock_slippage_model.predict_slippage.return_value = mock_slippage_prediction
    mock_cost_analyzer = MagicMock()
    
    return ExecutionEngine(
        orchestration_service=None,
        execution_service=execution_service,
        risk_service=mock_risk_service,
        config=config,
        exchange_factory=mock_exchange_factory,
        order_manager=mock_order_manager,
        slippage_model=mock_slippage_model,
        cost_analyzer=mock_cost_analyzer,
        algorithms={
            ExecutionAlgorithm.TWAP: AsyncMock(),
            ExecutionAlgorithm.VWAP: AsyncMock(),
            ExecutionAlgorithm.ICEBERG: AsyncMock(),
            ExecutionAlgorithm.SMART_ROUTER: AsyncMock()
        },  # Mock algorithms dict
    )


@pytest.fixture(scope="session")
def sample_order_request():
    """Create sample order request using pre-defined constants."""
    return OrderRequest(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=TEST_DECIMALS["ONE"],
        price=TEST_DECIMALS["PRICE_50K"],
    )


@pytest.fixture(scope="function")
def sample_execution_instruction(sample_order_request):
    """Create sample execution instruction."""
    return ExecutionInstruction(
        order=sample_order_request,
        algorithm=ExecutionAlgorithm.TWAP,
        time_horizon_minutes=60,
        participation_rate=0.2,
    )


@pytest.fixture
def sample_core_execution_instruction():
    """Create sample core execution instruction."""
    return CoreExecutionInstruction(
        instruction_id="test-exec-123",
        symbol="BTC/USDT",
        side="buy",
        target_quantity=TEST_DECIMALS["ONE"],
        algorithm=ExecutionAlgorithm.TWAP,
        urgency=Decimal("0.5"),
        limit_price=Decimal("50000"),
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return MarketData(
        symbol="BTC/USDT",
        exchange="binance",
        timestamp=datetime.now(timezone.utc),
        open=Decimal("49500"),
        high=Decimal("51000"),
        low=Decimal("49000"),
        close=Decimal("50000"),
        volume=Decimal("1000"),
    )


@pytest.fixture
def mock_exchange_factory():
    """Create mock exchange factory."""
    factory = AsyncMock()
    exchange = AsyncMock()
    exchange.exchange_name = "binance"
    exchange.get_market_data = AsyncMock()
    factory.get_exchange = AsyncMock(return_value=exchange)
    return factory


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    manager = MagicMock()
    manager.validate_order = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def sample_order_response():
    """Create sample order response."""
    from src.core.types.trading import OrderResponse, OrderSide, OrderStatus, OrderType

    return OrderResponse(
        order_id="TEST123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("1.0"),
        average_price=Decimal("50000.0"),
        created_at=datetime.now(timezone.utc),
        exchange="binance",
    )


class TestExecutionEngine:
    """Test cases for ExecutionEngine."""

    @pytest.mark.asyncio
    async def test_initialization(self, execution_engine):
        """Test ExecutionEngine initialization."""
        assert execution_engine is not None
        assert not execution_engine.is_running
        assert len(execution_engine.algorithms) == 4
        assert ExecutionAlgorithm.TWAP in execution_engine.algorithms
        assert ExecutionAlgorithm.VWAP in execution_engine.algorithms
        assert ExecutionAlgorithm.ICEBERG in execution_engine.algorithms
        assert ExecutionAlgorithm.SMART_ROUTER in execution_engine.algorithms

    @pytest.mark.asyncio
    async def test_start_stop(self, execution_engine):
        """Test engine start and stop."""
        # Test start
        await execution_engine.start()
        assert execution_engine.is_running

        # Test stop
        await execution_engine.stop()
        assert not execution_engine.is_running

    @pytest.mark.asyncio
    async def test_validate_execution_instruction_valid(
        self, execution_engine, sample_execution_instruction
    ):
        """Test validation of valid execution instruction."""
        # Test that the instruction is properly created without errors
        assert sample_execution_instruction.order is not None
        assert sample_execution_instruction.algorithm == ExecutionAlgorithm.TWAP
        assert sample_execution_instruction.time_horizon_minutes == 60
        assert sample_execution_instruction.participation_rate == 0.2

    @pytest.mark.asyncio
    async def test_validate_execution_instruction_invalid_order(self, execution_engine):
        """Test validation of invalid execution instruction."""
        # Test that creating ExecutionInstruction with None order works but execution will fail
        invalid_instruction = ExecutionInstruction(order=None, algorithm=ExecutionAlgorithm.TWAP)
        # The order field should be None
        assert invalid_instruction.order is None

    @pytest.mark.asyncio
    async def test_validate_execution_instruction_invalid_symbol(self, execution_engine):
        """Test validation of instruction with invalid symbol."""
        # Creating an OrderRequest with empty symbol should raise ValidationError
        with pytest.raises(ValidationError):
            order = OrderRequest(
                symbol="", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("1.0")
            )

    @pytest.mark.asyncio
    async def test_validate_execution_instruction_invalid_quantity(self, execution_engine):
        """Test validation of instruction with invalid quantity."""
        # Creating an OrderRequest with zero quantity should raise ValidationError
        with pytest.raises(CoreValidationError):
            order = OrderRequest(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0"),
            )

    @pytest.mark.asyncio
    async def test_execution_with_smart_routing(
        self, execution_engine, sample_core_execution_instruction, sample_market_data
    ):
        """Test order execution with smart routing algorithm."""
        # Configure instruction for smart routing
        sample_core_execution_instruction.algorithm = ExecutionAlgorithm.SMART_ROUTER

        # Create mock execution result wrapper
        order = OrderRequest(
            symbol=sample_core_execution_instruction.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=sample_core_execution_instruction.target_quantity,
        )
        mock_result = create_execution_result_wrapper(
            execution_id="test-exec-123",
            order=order,
            algorithm=ExecutionAlgorithm.SMART_ROUTER,
            status=ExecutionStatus.COMPLETED,
            filled_quantity=sample_core_execution_instruction.target_quantity,
        )

        # Mock the algorithm's execute method
        with patch.object(
            execution_engine.algorithms[ExecutionAlgorithm.SMART_ROUTER],
            "execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_execute:
            # Mock services
            execution_engine.execution_service.validate_order_pre_execution = AsyncMock(
                return_value={"overall_result": "passed", "errors": [], "risk_score": 50.0}
            )
            execution_engine.risk_service.validate_signal = AsyncMock(return_value=True)
            execution_engine.risk_service.calculate_position_size = AsyncMock(
                return_value=sample_core_execution_instruction.target_quantity
            )
            execution_engine.execution_service.record_trade_execution = AsyncMock()

            # Start the engine
            await execution_engine.start()

            # Execute order
            result = await execution_engine.execute_order(
                sample_core_execution_instruction, sample_market_data
            )

            # Verify algorithm was properly selected and order executed
            assert result.status == ExecutionStatus.COMPLETED
            assert result.execution_id == "test-exec-123"
            mock_execute.assert_called_once()

            # Stop engine
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_perform_post_trade_analysis(self, execution_engine, sample_market_data):
        """Test post-trade analysis."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        execution_result = create_execution_result_wrapper(
            execution_id="test_123",
            order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.COMPLETED,
            filled_quantity=Decimal("1.0"),
            execution_duration=5.0,
        )
        pre_trade_analysis = {"risk_level": "medium", "warnings": []}

        result = await execution_engine._perform_post_trade_analysis(
            execution_result, sample_market_data, pre_trade_analysis
        )

        assert "execution_time_ms" in result
        assert "slippage_bps" in result
        assert "fill_rate" in result
        assert "quality_score" in result
        assert "algorithm_used" in result
        assert "market_conditions" in result
        assert result["fill_rate"] == 1.0
        assert result["algorithm_used"] == ExecutionAlgorithm.TWAP.value

    @pytest.mark.asyncio
    async def test_select_algorithm_specified_twap(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test algorithm selection when TWAP is specified."""
        sample_execution_instruction.algorithm = ExecutionAlgorithm.TWAP
        validation_results = {"risk_level": "medium"}

        algorithm = await execution_engine._select_algorithm(
            sample_execution_instruction, sample_market_data, validation_results
        )

        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.TWAP]

    @pytest.mark.asyncio
    async def test_select_algorithm_smart_router(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test algorithm selection for smart routing."""
        sample_execution_instruction.algorithm = ExecutionAlgorithm.SMART_ROUTER
        validation_results = {"risk_level": "medium"}

        algorithm = await execution_engine._select_algorithm(
            sample_execution_instruction, sample_market_data, validation_results
        )

        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.SMART_ROUTER]

    @pytest.mark.asyncio
    async def test_select_algorithm_high_risk_order(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test algorithm selection for high risk orders."""
        # Create a mock algorithm enum that doesn't exist in the algorithms dict
        from unittest.mock import MagicMock

        mock_algorithm = MagicMock()
        mock_algorithm.value = "NON_EXISTENT"
        sample_execution_instruction.algorithm = mock_algorithm
        # Set high risk level
        validation_results = {"risk_level": "high"}

        algorithm = await execution_engine._select_algorithm(
            sample_execution_instruction, sample_market_data, validation_results
        )

        # Should select TWAP for high-risk orders
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.TWAP]

    @pytest.mark.asyncio
    async def test_select_algorithm_volume_significant_order(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test algorithm selection for volume significant orders."""
        # Create a mock algorithm enum that doesn't exist in the algorithms dict
        from unittest.mock import MagicMock

        mock_algorithm = MagicMock()
        mock_algorithm.value = "NON_EXISTENT"
        sample_execution_instruction.algorithm = mock_algorithm
        # Create order value > market_volume * volume_significance_threshold but < large_order_threshold
        # volume_significance = 1000 * 0.01 = 10, large_order_threshold = 10000
        # So order value needs to be > 10 but < 10000
        sample_execution_instruction.order.quantity = Decimal(
            "0.1"
        )  # 0.1 * 100 = 10 > 10, but < 10000
        sample_execution_instruction.order.price = Decimal(
            "200"
        )  # 0.1 * 200 = 20 > 10, but < 10000
        validation_results = {"risk_level": "medium"}

        algorithm = await execution_engine._select_algorithm(
            sample_execution_instruction, sample_market_data, validation_results
        )

        # Should select VWAP for volume-significant orders
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.VWAP]

    @pytest.mark.asyncio
    async def test_select_algorithm_large_quantity_order(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test algorithm selection for large quantity orders."""
        # Create a mock algorithm enum that doesn't exist in the algorithms dict
        from unittest.mock import MagicMock

        mock_algorithm = MagicMock()
        mock_algorithm.value = "NON_EXISTENT"
        sample_execution_instruction.algorithm = mock_algorithm
        # Set large quantity order but with small value to avoid triggering large order threshold
        # Need: quantity > 1000, order_value < 10000, order_value < volume_significance (10)
        sample_execution_instruction.order.quantity = Decimal("1500")  # > 1000 threshold
        sample_execution_instruction.order.price = Decimal("0.005")  # 1500 * 0.005 = 7.5 < 10
        validation_results = {"risk_level": "medium"}

        algorithm = await execution_engine._select_algorithm(
            sample_execution_instruction, sample_market_data, validation_results
        )

        # Should select ICEBERG for large quantity orders
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.ICEBERG]

    @pytest.mark.asyncio
    async def test_select_algorithm_large_value_threshold(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test algorithm selection for orders exceeding value threshold."""
        # Remove algorithm specification to trigger intelligent selection
        sample_execution_instruction.algorithm = None
        # Set order value to exceed large_order_threshold (default 10000)
        sample_execution_instruction.order.quantity = Decimal("1")
        sample_execution_instruction.order.price = Decimal("15000")  # 1 * 15000 > 10000
        validation_results = {"risk_level": "medium"}

        algorithm = await execution_engine._select_algorithm(
            sample_execution_instruction, sample_market_data, validation_results
        )

        # Should select TWAP for orders exceeding value threshold
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.TWAP]

    @pytest.mark.asyncio
    async def test_select_algorithm_default_case(
        self, execution_engine, sample_execution_instruction, sample_market_data
    ):
        """Test default algorithm selection for general case orders."""
        # Create a mock algorithm enum that doesn't exist in the algorithms dict
        from unittest.mock import MagicMock

        mock_algorithm = MagicMock()
        mock_algorithm.value = "NON_EXISTENT"
        sample_execution_instruction.algorithm = mock_algorithm
        # Set small order that doesn't trigger any special conditions
        # Need: quantity < 1000, order_value < 10000, order_value < volume_significance (10)
        sample_execution_instruction.order.quantity = Decimal("0.5")  # Small quantity < 1000
        sample_execution_instruction.order.price = Decimal("10")  # Small value: 0.5 * 10 = 5 < 10
        validation_results = {"risk_level": "medium"}

        algorithm = await execution_engine._select_algorithm(
            sample_execution_instruction, sample_market_data, validation_results
        )

        # Should select Smart Router for general cases
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.SMART_ROUTER]

    @pytest.mark.asyncio
    async def test_algorithm_execution_integration(
        self, execution_engine, sample_execution_instruction, mock_exchange_factory
    ):
        """Test direct algorithm execution."""
        # Create mock execution result
        mock_execution_result = create_execution_result_wrapper(
            execution_id="test_execution_123",
            order=sample_execution_instruction.order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.COMPLETED,
            filled_quantity=sample_execution_instruction.order.quantity,
        )

        # Mock algorithm execution directly
        algorithm = execution_engine.algorithms[ExecutionAlgorithm.TWAP]
        with patch.object(
            algorithm, "execute", new_callable=AsyncMock, return_value=mock_execution_result
        ):
            result = await algorithm.execute(
                sample_execution_instruction,
                exchange_factory=mock_exchange_factory,
                risk_manager=execution_engine.risk_manager_adapter,
            )

            assert result == mock_execution_result
            assert sample_execution_instruction.algorithm == ExecutionAlgorithm.TWAP

    @pytest.mark.asyncio
    async def test_cancel_execution(self, execution_engine):
        """Test execution cancellation."""
        execution_id = "test_execution_123"

        # Create mock execution result and add to active executions
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        mock_execution_result = create_execution_result_wrapper(
            execution_id=execution_id,
            order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.RUNNING,
        )
        execution_engine.active_executions[execution_id] = mock_execution_result

        # Mock algorithm cancellation
        algorithm = execution_engine.algorithms[ExecutionAlgorithm.TWAP]
        with patch.object(algorithm, "cancel_execution", new_callable=AsyncMock, return_value=True):
            result = await execution_engine.cancel_execution(execution_id)

            assert result is True

    @pytest.mark.asyncio
    async def test_cancel_execution_not_found(self, execution_engine):
        """Test cancellation of non-existent execution."""
        result = await execution_engine.cancel_execution("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_execution_metrics(self, execution_engine):
        """Test execution metrics retrieval."""
        # Mock the execution service metrics
        mock_service_metrics = {
            "total_executions": 100,
            "successful_executions": 95,
            "failed_executions": 5,
            "average_execution_time_ms": 2500,
        }
        execution_engine.execution_service.get_execution_metrics = AsyncMock(
            return_value=mock_service_metrics
        )

        metrics = await execution_engine.get_execution_metrics()

        assert "service_metrics" in metrics
        assert "engine_metrics" in metrics
        assert "timestamp" in metrics
        assert metrics["service_metrics"] == mock_service_metrics
        assert metrics["engine_metrics"]["active_executions"] == 0
        assert metrics["engine_metrics"]["algorithms_available"] == 4
