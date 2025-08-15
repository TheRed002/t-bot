"""Unit tests for ExecutionEngine."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
)
from src.execution.execution_engine import ExecutionEngine


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    return config


@pytest.fixture
def execution_engine(config):
    """Create ExecutionEngine instance."""
    return ExecutionEngine(config)


@pytest.fixture
def sample_order_request():
    """Create sample order request."""
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
        price=Decimal("50000")
    )


@pytest.fixture
def sample_execution_instruction(sample_order_request):
    """Create sample execution instruction."""
    return ExecutionInstruction(
        order=sample_order_request,
        algorithm=ExecutionAlgorithm.TWAP,
        time_horizon_minutes=60,
        participation_rate=0.2
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return MarketData(
        symbol="BTCUSDT",
        price=Decimal("50000"),
        volume=Decimal("1000"),
        timestamp=datetime.now(timezone.utc),
        bid=Decimal("49995"),
        ask=Decimal("50005"),
        high_price=Decimal("51000"),
        low_price=Decimal("49000")
    )


@pytest.fixture
def mock_exchange_factory():
    """Create mock exchange factory."""
    factory = MagicMock()
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
    async def test_validate_execution_instruction_valid(self, execution_engine, sample_execution_instruction):
        """Test validation of valid execution instruction."""
        await execution_engine._validate_execution_instruction(sample_execution_instruction)
        # Should not raise any exception

    @pytest.mark.asyncio
    async def test_validate_execution_instruction_invalid_order(self, execution_engine):
        """Test validation of invalid execution instruction."""
        invalid_instruction = ExecutionInstruction(
            order=None,
            algorithm=ExecutionAlgorithm.TWAP
        )
        
        with pytest.raises(Exception):
            await execution_engine._validate_execution_instruction(invalid_instruction)

    @pytest.mark.asyncio
    async def test_validate_execution_instruction_invalid_symbol(self, execution_engine):
        """Test validation of instruction with invalid symbol."""
        order = OrderRequest(
            symbol="",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        instruction = ExecutionInstruction(order=order, algorithm=ExecutionAlgorithm.TWAP)
        
        with pytest.raises(Exception):
            await execution_engine._validate_execution_instruction(instruction)

    @pytest.mark.asyncio
    async def test_validate_execution_instruction_invalid_quantity(self, execution_engine):
        """Test validation of instruction with invalid quantity."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0")
        )
        instruction = ExecutionInstruction(order=order, algorithm=ExecutionAlgorithm.TWAP)
        
        with pytest.raises(Exception):
            await execution_engine._validate_execution_instruction(instruction)

    @pytest.mark.asyncio
    async def test_get_market_data(self, execution_engine, sample_execution_instruction, 
                                  sample_market_data, mock_exchange_factory):
        """Test market data retrieval."""
        mock_exchange_factory.get_exchange.return_value.get_market_data.return_value = sample_market_data
        
        result = await execution_engine._get_market_data(sample_execution_instruction, mock_exchange_factory)
        
        assert result == sample_market_data
        mock_exchange_factory.get_exchange.assert_called_once_with("binance")

    @pytest.mark.asyncio
    async def test_get_market_data_with_preferred_exchange(self, execution_engine, 
                                                          sample_execution_instruction, sample_market_data, 
                                                          mock_exchange_factory):
        """Test market data retrieval with preferred exchange."""
        sample_execution_instruction.preferred_exchanges = ["okx"]
        mock_exchange_factory.get_exchange.return_value.get_market_data.return_value = sample_market_data
        
        result = await execution_engine._get_market_data(sample_execution_instruction, mock_exchange_factory)
        
        assert result == sample_market_data
        mock_exchange_factory.get_exchange.assert_called_once_with("okx")

    @pytest.mark.asyncio
    async def test_perform_pre_trade_analysis(self, execution_engine, sample_execution_instruction, 
                                            sample_market_data):
        """Test pre-trade analysis."""
        with patch.object(execution_engine.slippage_model, 'predict_slippage', new_callable=AsyncMock) as mock_predict:
            mock_slippage = MagicMock()
            mock_predict.return_value = mock_slippage
            
            result = await execution_engine._perform_pre_trade_analysis(
                sample_execution_instruction, sample_market_data
            )
            
            assert "slippage_prediction" in result
            assert "order_value" in result
            assert "market_conditions" in result
            assert "analysis_timestamp" in result
            
            mock_predict.assert_called_once_with(
                sample_execution_instruction.order,
                sample_market_data,
                sample_execution_instruction.participation_rate,
                sample_execution_instruction.time_horizon_minutes
            )

    @pytest.mark.asyncio
    async def test_select_optimal_algorithm_specified_twap(self, execution_engine, sample_execution_instruction, 
                                                         sample_market_data):
        """Test algorithm selection when TWAP is specified."""
        sample_execution_instruction.algorithm = ExecutionAlgorithm.TWAP
        pre_trade_analysis = {"order_value": 50000, "market_conditions": {"volume_ratio": 0.01}}
        
        with patch.object(execution_engine.algorithms[ExecutionAlgorithm.TWAP], 'validate_instruction',
                         new_callable=AsyncMock, return_value=True):
            algorithm = await execution_engine._select_optimal_algorithm(
                sample_execution_instruction, sample_market_data, pre_trade_analysis
            )
            
            assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.TWAP]

    @pytest.mark.asyncio
    async def test_select_optimal_algorithm_urgent(self, execution_engine, sample_execution_instruction, 
                                                  sample_market_data):
        """Test algorithm selection for urgent orders."""
        sample_execution_instruction.is_urgent = True
        sample_execution_instruction.algorithm = ExecutionAlgorithm.SMART_ROUTER
        pre_trade_analysis = {"order_value": 50000, "market_conditions": {"volume_ratio": 0.01}}
        
        algorithm = await execution_engine._select_optimal_algorithm(
            sample_execution_instruction, sample_market_data, pre_trade_analysis
        )
        
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.SMART_ROUTER]

    @pytest.mark.asyncio
    async def test_select_optimal_algorithm_large_volume(self, execution_engine, sample_execution_instruction, 
                                                        sample_market_data):
        """Test algorithm selection for large volume orders."""
        sample_execution_instruction.algorithm = ExecutionAlgorithm.SMART_ROUTER
        pre_trade_analysis = {"order_value": 50000, "market_conditions": {"volume_ratio": 0.1}}  # High volume ratio
        
        algorithm = await execution_engine._select_optimal_algorithm(
            sample_execution_instruction, sample_market_data, pre_trade_analysis
        )
        
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.ICEBERG]

    @pytest.mark.asyncio
    async def test_select_optimal_algorithm_multi_exchange(self, execution_engine, sample_execution_instruction, 
                                                          sample_market_data):
        """Test algorithm selection for multi-exchange routing."""
        sample_execution_instruction.algorithm = ExecutionAlgorithm.SMART_ROUTER
        pre_trade_analysis = {"order_value": 100000, "market_conditions": {"volume_ratio": 0.01}}  # Large order value
        
        algorithm = await execution_engine._select_optimal_algorithm(
            sample_execution_instruction, sample_market_data, pre_trade_analysis
        )
        
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.SMART_ROUTER]

    @pytest.mark.asyncio
    async def test_select_optimal_algorithm_time_sensitive(self, execution_engine, sample_execution_instruction, 
                                                          sample_market_data):
        """Test algorithm selection for time-sensitive orders."""
        sample_execution_instruction.algorithm = ExecutionAlgorithm.SMART_ROUTER
        sample_execution_instruction.time_horizon_minutes = 15  # Short time horizon
        pre_trade_analysis = {"order_value": 10000, "market_conditions": {"volume_ratio": 0.01}}
        
        algorithm = await execution_engine._select_optimal_algorithm(
            sample_execution_instruction, sample_market_data, pre_trade_analysis
        )
        
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.TWAP]

    @pytest.mark.asyncio
    async def test_select_optimal_algorithm_large_order(self, execution_engine, sample_execution_instruction, 
                                                       sample_market_data):
        """Test algorithm selection for large orders."""
        sample_execution_instruction.algorithm = ExecutionAlgorithm.SMART_ROUTER
        pre_trade_analysis = {"order_value": 20000, "market_conditions": {"volume_ratio": 0.01}}  # Large order
        
        algorithm = await execution_engine._select_optimal_algorithm(
            sample_execution_instruction, sample_market_data, pre_trade_analysis
        )
        
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.VWAP]

    @pytest.mark.asyncio
    async def test_select_optimal_algorithm_default(self, execution_engine, sample_execution_instruction, 
                                                   sample_market_data):
        """Test default algorithm selection."""
        sample_execution_instruction.algorithm = ExecutionAlgorithm.SMART_ROUTER
        pre_trade_analysis = {"order_value": 5000, "market_conditions": {"volume_ratio": 0.01}}  # Small order
        
        algorithm = await execution_engine._select_optimal_algorithm(
            sample_execution_instruction, sample_market_data, pre_trade_analysis
        )
        
        assert algorithm == execution_engine.algorithms[ExecutionAlgorithm.TWAP]

    @pytest.mark.asyncio
    async def test_execute_with_algorithm(self, execution_engine, sample_execution_instruction, 
                                         mock_exchange_factory, mock_risk_manager):
        """Test execution with algorithm."""
        # Create mock execution result
        mock_execution_result = ExecutionResult(
            execution_id="test_execution_123",
            original_order=sample_execution_instruction.order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.COMPLETED,
            start_time=datetime.now(timezone.utc)
        )
        
        # Mock algorithm execution
        algorithm = execution_engine.algorithms[ExecutionAlgorithm.TWAP]
        with patch.object(algorithm, 'execute', new_callable=AsyncMock, return_value=mock_execution_result):
            result = await execution_engine._execute_with_algorithm(
                sample_execution_instruction, algorithm, mock_exchange_factory, mock_risk_manager
            )
            
            assert result == mock_execution_result
            assert sample_execution_instruction.algorithm == ExecutionAlgorithm.TWAP
            assert execution_engine.execution_statistics["algorithm_usage"]["twap"] == 1

    @pytest.mark.asyncio
    async def test_update_execution_statistics_successful(self, execution_engine):
        """Test statistics update for successful execution."""
        execution_result = ExecutionResult(
            execution_id="test_123",
            original_order=OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0")
            ),
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            total_filled_quantity=Decimal("1.0"),
            execution_duration=30.0
        )
        
        initial_total = execution_engine.execution_statistics["total_executions"]
        initial_successful = execution_engine.execution_statistics["successful_executions"]
        
        await execution_engine._update_execution_statistics(execution_result)
        
        assert execution_engine.execution_statistics["total_executions"] == initial_total + 1
        assert execution_engine.execution_statistics["successful_executions"] == initial_successful + 1
        assert execution_engine.execution_statistics["total_volume"] == Decimal("1.0")
        assert execution_engine.execution_statistics["average_execution_time_seconds"] == 30.0

    @pytest.mark.asyncio
    async def test_update_execution_statistics_failed(self, execution_engine):
        """Test statistics update for failed execution."""
        execution_result = ExecutionResult(
            execution_id="test_123",
            original_order=OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0")
            ),
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.FAILED,
            start_time=datetime.now(timezone.utc)
        )
        
        initial_total = execution_engine.execution_statistics["total_executions"]
        initial_failed = execution_engine.execution_statistics["failed_executions"]
        
        await execution_engine._update_execution_statistics(execution_result)
        
        assert execution_engine.execution_statistics["total_executions"] == initial_total + 1
        assert execution_engine.execution_statistics["failed_executions"] == initial_failed + 1

    @pytest.mark.asyncio
    async def test_cancel_execution(self, execution_engine):
        """Test execution cancellation."""
        execution_id = "test_execution_123"
        
        # Create mock execution result and add to active executions
        mock_execution_result = ExecutionResult(
            execution_id=execution_id,
            original_order=OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0")
            ),
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now(timezone.utc)
        )
        execution_engine.active_executions[execution_id] = mock_execution_result
        
        # Mock algorithm cancellation
        algorithm = execution_engine.algorithms[ExecutionAlgorithm.TWAP]
        with patch.object(algorithm, 'cancel_execution', new_callable=AsyncMock, return_value=True):
            result = await execution_engine.cancel_execution(execution_id)
            
            assert result is True

    @pytest.mark.asyncio
    async def test_cancel_execution_not_found(self, execution_engine):
        """Test cancellation of non-existent execution."""
        result = await execution_engine.cancel_execution("non_existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_execution_status(self, execution_engine):
        """Test getting execution status."""
        execution_id = "test_execution_123"
        
        # Test with active execution
        mock_execution_result = ExecutionResult(
            execution_id=execution_id,
            original_order=OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0")
            ),
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now(timezone.utc)
        )
        execution_engine.active_executions[execution_id] = mock_execution_result
        
        status = await execution_engine.get_execution_status(execution_id)
        assert status == ExecutionStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_execution_status_from_algorithm(self, execution_engine):
        """Test getting execution status from algorithm."""
        execution_id = "test_execution_123"
        
        # Mock algorithm status check
        algorithm = execution_engine.algorithms[ExecutionAlgorithm.TWAP]
        with patch.object(algorithm, 'get_execution_status', new_callable=AsyncMock, 
                         return_value=ExecutionStatus.COMPLETED):
            status = await execution_engine.get_execution_status(execution_id)
            assert status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_estimate_volatility(self, execution_engine, sample_market_data):
        """Test volatility estimation."""
        volatility = await execution_engine._estimate_volatility(sample_market_data)
        
        # With high and low prices set, should calculate actual volatility
        expected_volatility = (51000 - 49000) / 50000  # (high - low) / price
        assert volatility == expected_volatility

    @pytest.mark.asyncio
    async def test_estimate_volatility_no_range_data(self, execution_engine):
        """Test volatility estimation without range data."""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        volatility = await execution_engine._estimate_volatility(market_data)
        assert volatility == 0.02  # Default volatility

    @pytest.mark.asyncio
    async def test_estimate_liquidity(self, execution_engine, sample_market_data):
        """Test liquidity estimation."""
        liquidity = await execution_engine._estimate_liquidity(sample_market_data)
        
        # Volume of 1000 should give liquidity score of min(1.0, 1000/1000000) = 0.001
        expected_liquidity = min(1.0, 1000 / 1000000)
        assert liquidity == expected_liquidity

    @pytest.mark.asyncio
    async def test_calculate_spread_bps(self, execution_engine, sample_market_data):
        """Test spread calculation in basis points."""
        spread_bps = await execution_engine._calculate_spread_bps(sample_market_data)
        
        # Spread = 50005 - 49995 = 10, spread_bps = (10 / 50000) * 10000 = 2
        expected_spread_bps = Decimal("2")
        assert spread_bps == expected_spread_bps

    @pytest.mark.asyncio
    async def test_calculate_spread_bps_no_bid_ask(self, execution_engine):
        """Test spread calculation without bid/ask data."""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        spread_bps = await execution_engine._calculate_spread_bps(market_data)
        assert spread_bps == Decimal("20")  # Default spread

    @pytest.mark.asyncio
    async def test_get_engine_summary(self, execution_engine):
        """Test engine summary generation."""
        with patch.object(execution_engine.order_manager, 'get_order_manager_summary',
                         new_callable=AsyncMock, return_value={"orders": 0}):
            with patch.object(execution_engine.slippage_model, 'get_model_summary',
                             new_callable=AsyncMock, return_value={"models": 0}):
                with patch.object(execution_engine.cost_analyzer, 'get_performance_report',
                                 new_callable=AsyncMock, return_value={"reports": 0}):
                    
                    summary = await execution_engine.get_engine_summary()
                    
                    assert "engine_status" in summary
                    assert "performance_statistics" in summary
                    assert "component_summaries" in summary
                    assert "configuration" in summary
                    
                    assert summary["engine_status"]["is_running"] is False
                    assert len(summary["engine_status"]["supported_algorithms"]) == 4