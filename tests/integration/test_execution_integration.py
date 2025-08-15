"""Integration tests for execution engine."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionStatus,
    MarketData,
    OrderRequest,
    OrderResponse,
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
def sample_execution_instruction():
    """Create sample execution instruction."""
    order = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
        price=Decimal("50000")
    )
    
    return ExecutionInstruction(
        order=order,
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
        volume=Decimal("100000"),
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
    exchange.place_order = AsyncMock()
    exchange.get_order_status = AsyncMock(return_value="filled")
    factory.get_exchange = AsyncMock(return_value=exchange)
    return factory


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    manager = MagicMock()
    manager.validate_order = AsyncMock(return_value=True)
    return manager


class TestExecutionEngineIntegration:
    """Integration test cases for ExecutionEngine."""

    @pytest.mark.asyncio
    async def test_full_execution_workflow(self, execution_engine, sample_execution_instruction, 
                                          sample_market_data, mock_exchange_factory, mock_risk_manager):
        """Test complete execution workflow from instruction to result."""
        # Setup mocks
        mock_exchange_factory.get_exchange.return_value.get_market_data.return_value = sample_market_data
        
        # Mock successful order response
        order_response = OrderResponse(
            id="order_123",
            client_order_id="client_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            filled_quantity=Decimal("1.0"),
            status="filled",
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange_factory.get_exchange.return_value.place_order.return_value = order_response
        
        # Start engine
        await execution_engine.start()
        
        try:
            # Mock fast execution for testing
            with patch('asyncio.sleep', new_callable=AsyncMock):
                # Execute order
                result = await execution_engine.execute_order(
                    sample_execution_instruction,
                    mock_exchange_factory,
                    mock_risk_manager,
                    sample_market_data
                )
            
            # Verify result
            assert result is not None
            assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.PARTIALLY_FILLED]
            assert result.algorithm == ExecutionAlgorithm.TWAP
            assert result.original_order == sample_execution_instruction.order
            
            # Verify engine statistics updated
            assert execution_engine.execution_statistics["total_executions"] >= 1
            
        finally:
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_algorithm_selection_integration(self, execution_engine, mock_exchange_factory, mock_risk_manager):
        """Test algorithm selection integration."""
        await execution_engine.start()
        
        try:
            # Test urgent order -> Smart Router
            urgent_instruction = ExecutionInstruction(
                order=OrderRequest(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("1.0")
                ),
                algorithm=ExecutionAlgorithm.SMART_ROUTER,
                is_urgent=True
            )
            
            market_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                volume=Decimal("100000"),
                timestamp=datetime.now(timezone.utc)
            )
            
            mock_exchange_factory.get_exchange.return_value.get_market_data.return_value = market_data
            
            # Mock the algorithm selection
            pre_trade_analysis = {"order_value": 50000, "market_conditions": {"volume_ratio": 0.01}}
            
            selected_algorithm = await execution_engine._select_optimal_algorithm(
                urgent_instruction, market_data, pre_trade_analysis
            )
            
            assert selected_algorithm == execution_engine.algorithms[ExecutionAlgorithm.SMART_ROUTER]
            
        finally:
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_pre_post_trade_analysis_integration(self, execution_engine, sample_execution_instruction, 
                                                     sample_market_data, mock_exchange_factory, mock_risk_manager):
        """Test pre-trade and post-trade analysis integration."""
        await execution_engine.start()
        
        try:
            # Mock successful execution
            order_response = OrderResponse(
                id="order_123",
                client_order_id="client_123",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
                filled_quantity=Decimal("1.0"),
                status="filled",
                timestamp=datetime.now(timezone.utc)
            )
            mock_exchange_factory.get_exchange.return_value.place_order.return_value = order_response
            mock_exchange_factory.get_exchange.return_value.get_market_data.return_value = sample_market_data
            
            # Perform pre-trade analysis
            pre_trade_analysis = await execution_engine._perform_pre_trade_analysis(
                sample_execution_instruction, sample_market_data
            )
            
            assert "slippage_prediction" in pre_trade_analysis
            assert "order_value" in pre_trade_analysis
            assert "market_conditions" in pre_trade_analysis
            
            # Execute with mocked sleep
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await execution_engine.execute_order(
                    sample_execution_instruction,
                    mock_exchange_factory,
                    mock_risk_manager,
                    sample_market_data
                )
            
            # Check that post-trade analysis was performed
            assert "post_trade_analysis" in result.metadata
            assert "pre_trade_analysis" in result.metadata
            
        finally:
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_engine_summary_integration(self, execution_engine, sample_execution_instruction, 
                                            sample_market_data, mock_exchange_factory, mock_risk_manager):
        """Test engine summary generation integration."""
        await execution_engine.start()
        
        try:
            # Mock all component summaries
            with patch.object(execution_engine.order_manager, 'get_order_manager_summary',
                             new_callable=AsyncMock, return_value={"test": "order_manager"}):
                with patch.object(execution_engine.slippage_model, 'get_model_summary',
                                 new_callable=AsyncMock, return_value={"test": "slippage_model"}):
                    with patch.object(execution_engine.cost_analyzer, 'get_performance_report',
                                     new_callable=AsyncMock, return_value={"test": "cost_analyzer"}):
                        
                        summary = await execution_engine.get_engine_summary()
                        
                        # Verify summary structure
                        assert "engine_status" in summary
                        assert "performance_statistics" in summary
                        assert "component_summaries" in summary
                        assert "configuration" in summary
                        
                        # Verify component summaries included
                        assert summary["component_summaries"]["order_manager"]["test"] == "order_manager"
                        assert summary["component_summaries"]["slippage_model"]["test"] == "slippage_model"
                        assert summary["component_summaries"]["cost_analyzer"]["test"] == "cost_analyzer"
                        
                        # Verify algorithm summaries
                        assert "algorithms" in summary["component_summaries"]
                        assert len(summary["component_summaries"]["algorithms"]) == 4
                        
        finally:
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_execution_cancellation_integration(self, execution_engine, sample_execution_instruction, 
                                                     mock_exchange_factory, mock_risk_manager):
        """Test execution cancellation integration."""
        await execution_engine.start()
        
        try:
            # Create a mock long-running execution
            execution_id = "test_execution_123"
            
            # Mock the algorithm to support cancellation
            with patch.object(execution_engine.algorithms[ExecutionAlgorithm.TWAP], 'cancel_execution',
                             new_callable=AsyncMock, return_value=True):
                
                # Add execution to active executions
                from src.core.types import ExecutionResult
                mock_result = ExecutionResult(
                    execution_id=execution_id,
                    original_order=sample_execution_instruction.order,
                    algorithm=ExecutionAlgorithm.TWAP,
                    status=ExecutionStatus.RUNNING,
                    start_time=datetime.now(timezone.utc)
                )
                execution_engine.active_executions[execution_id] = mock_result
                
                # Test cancellation
                success = await execution_engine.cancel_execution(execution_id)
                assert success is True
                
        finally:
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, execution_engine, sample_execution_instruction, 
                                            mock_exchange_factory, mock_risk_manager):
        """Test error handling integration throughout the execution flow."""
        await execution_engine.start()
        
        try:
            # Test with exchange failure
            mock_exchange_factory.get_exchange.return_value.get_market_data.side_effect = Exception("Exchange down")
            
            with pytest.raises(Exception):
                await execution_engine.execute_order(
                    sample_execution_instruction,
                    mock_exchange_factory,
                    mock_risk_manager
                )
            
            # Verify failure statistics updated
            assert execution_engine.execution_statistics["failed_executions"] >= 1
            assert execution_engine.execution_statistics["total_executions"] >= 1
            
        finally:
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_risk_manager_integration(self, execution_engine, sample_execution_instruction, 
                                          sample_market_data, mock_exchange_factory):
        """Test risk manager integration."""
        await execution_engine.start()
        
        try:
            # Create risk manager that rejects orders
            rejecting_risk_manager = MagicMock()
            rejecting_risk_manager.validate_order = AsyncMock(return_value=False)
            
            mock_exchange_factory.get_exchange.return_value.get_market_data.return_value = sample_market_data
            
            # Mock fast execution
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await execution_engine.execute_order(
                    sample_execution_instruction,
                    mock_exchange_factory,
                    rejecting_risk_manager,
                    sample_market_data
                )
            
            # Should complete but with limited or no fills due to risk rejections
            assert result is not None
            
        finally:
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_multiple_algorithm_integration(self, execution_engine, mock_exchange_factory, mock_risk_manager):
        """Test integration with multiple different algorithms."""
        await execution_engine.start()
        
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        mock_exchange_factory.get_exchange.return_value.get_market_data.return_value = market_data
        
        try:
            algorithms_to_test = [
                ExecutionAlgorithm.TWAP,
                ExecutionAlgorithm.VWAP,
                ExecutionAlgorithm.ICEBERG,
                ExecutionAlgorithm.SMART_ROUTER
            ]
            
            for algorithm in algorithms_to_test:
                instruction = ExecutionInstruction(
                    order=OrderRequest(
                        symbol="BTCUSDT",
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=Decimal("1.0")
                    ),
                    algorithm=algorithm,
                    time_horizon_minutes=30
                )
                
                # Mock algorithm validation to pass
                with patch.object(execution_engine.algorithms[algorithm], 'validate_instruction',
                                 new_callable=AsyncMock, return_value=True):
                    
                    pre_trade_analysis = {"order_value": 50000, "market_conditions": {"volume_ratio": 0.01}}
                    
                    selected_algorithm = await execution_engine._select_optimal_algorithm(
                        instruction, market_data, pre_trade_analysis
                    )
                    
                    # Verify the correct algorithm was selected
                    assert selected_algorithm == execution_engine.algorithms[algorithm]
                    
        finally:
            await execution_engine.stop()

    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self, execution_engine, sample_execution_instruction,
                                                   sample_market_data, mock_exchange_factory, mock_risk_manager):
        """Test performance tracking integration."""
        await execution_engine.start()
        
        try:
            initial_stats = execution_engine.execution_statistics.copy()
            
            # Mock successful execution
            order_response = OrderResponse(
                id="order_123",
                client_order_id="client_123",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
                filled_quantity=Decimal("1.0"),
                status="filled",
                timestamp=datetime.now(timezone.utc)
            )
            mock_exchange_factory.get_exchange.return_value.place_order.return_value = order_response
            mock_exchange_factory.get_exchange.return_value.get_market_data.return_value = sample_market_data
            
            # Execute with fast timing
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await execution_engine.execute_order(
                    sample_execution_instruction,
                    mock_exchange_factory,
                    mock_risk_manager,
                    sample_market_data
                )
            
            # Verify statistics were updated
            final_stats = execution_engine.execution_statistics
            
            assert final_stats["total_executions"] > initial_stats["total_executions"]
            assert final_stats["total_volume"] >= initial_stats["total_volume"]
            assert final_stats["algorithm_usage"]["twap"] > initial_stats["algorithm_usage"]["twap"]
            
        finally:
            await execution_engine.stop()