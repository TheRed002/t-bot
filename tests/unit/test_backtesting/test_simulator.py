"""Tests for backtesting simulator module."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

from src.backtesting.simulator import (
    SimulationConfig,
    SimulatedOrder,
    TradeSimulator
)
from src.core.types import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    ExecutionAlgorithm
)


class TestSimulationConfig:
    """Test SimulationConfig model."""

    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = SimulationConfig()
        
        assert config.initial_capital == Decimal("10000")
        assert config.commission_rate == Decimal("0.001")
        assert config.slippage_rate == Decimal("0.0005")
        assert config.enable_shorting is False
        assert config.max_positions == 5
        assert config.enable_partial_fills is True
        assert config.enable_market_impact is True
        assert config.enable_latency is True
        assert config.latency_ms == (10, 100)
        assert config.market_impact_factor == Decimal("0.0001")
        assert config.liquidity_factor == Decimal("0.1")
        assert config.rejection_probability == Decimal("0.01")

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = SimulationConfig(
            initial_capital=Decimal("50000"),
            commission_rate=Decimal("0.002"),
            slippage_rate=Decimal("0.001"),
            enable_shorting=True,
            max_positions=10,
            enable_partial_fills=False,
            enable_market_impact=False,
            enable_latency=False,
            latency_ms=(5, 50),
            market_impact_factor=Decimal("0.0002"),
            liquidity_factor=Decimal("0.2"),
            rejection_probability=Decimal("0.02")
        )
        
        assert config.initial_capital == Decimal("50000")
        assert config.commission_rate == Decimal("0.002")
        assert config.slippage_rate == Decimal("0.001")
        assert config.enable_shorting is True
        assert config.max_positions == 10
        assert config.enable_partial_fills is False
        assert config.enable_market_impact is False
        assert config.enable_latency is False
        assert config.latency_ms == (5, 50)
        assert config.market_impact_factor == Decimal("0.0002")
        assert config.liquidity_factor == Decimal("0.2")
        assert config.rejection_probability == Decimal("0.02")

    def test_config_validation(self):
        """Test config validation."""
        # Should allow valid values
        config = SimulationConfig(
            commission_rate=Decimal("0"),
            slippage_rate=Decimal("0"),
            max_positions=1,
            market_impact_factor=Decimal("0"),
            liquidity_factor=Decimal("1.0"),
            rejection_probability=Decimal("0")
        )
        
        assert config.commission_rate == Decimal("0")
        assert config.rejection_probability == Decimal("0")


class TestSimulatedOrder:
    """Test SimulatedOrder model."""

    def create_sample_order_request(self):
        """Create sample order request for testing."""
        return OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            exchange="binance"
        )

    def test_simulated_order_creation(self):
        """Test creating simulated order."""
        request = self.create_sample_order_request()
        order_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)
        
        order = SimulatedOrder(
            request=request,
            order_id=order_id,
            timestamp=timestamp
        )
        
        assert order.request == request
        assert order.order_id == order_id
        assert order.timestamp == timestamp
        assert order.filled_quantity == Decimal("0")
        assert order.status == OrderStatus.PENDING
        assert order.average_fill_price is None
        assert order.fill_time is None
        assert order.execution_fees == Decimal("0")
        assert order.execution_algorithm == ExecutionAlgorithm.MARKET
        assert order.algorithm_params == {}

    def test_simulated_order_with_custom_values(self):
        """Test creating simulated order with custom values."""
        request = self.create_sample_order_request()
        order_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)
        fill_time = datetime.now(timezone.utc)
        
        order = SimulatedOrder(
            request=request,
            order_id=order_id,
            timestamp=timestamp,
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PARTIALLY_FILLED,
            average_fill_price=Decimal("50100"),
            fill_time=fill_time,
            execution_fees=Decimal("25.05"),
            execution_algorithm=ExecutionAlgorithm.TWAP,
            algorithm_params={"duration": 300, "slices": 10}
        )
        
        assert order.filled_quantity == Decimal("0.5")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.average_fill_price == Decimal("50100")
        assert order.fill_time == fill_time
        assert order.execution_fees == Decimal("25.05")
        assert order.execution_algorithm == ExecutionAlgorithm.TWAP
        assert order.algorithm_params == {"duration": 300, "slices": 10}


class TestTradeSimulator:
    """Test TradeSimulator functionality."""

    def create_default_config(self):
        """Create default simulation config."""
        return SimulationConfig()

    def create_sample_order_request(self, symbol="BTCUSDT", quantity="1.0", price="50000"):
        """Create sample order request."""
        return OrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal(quantity),
            price=Decimal(price),
            exchange="binance"
        )

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        config = self.create_default_config()
        simulator = TradeSimulator(config)
        
        assert simulator.config == config
        assert simulator.slippage_model is None
        assert isinstance(simulator._order_book, dict)
        assert len(simulator._order_book) == 0
        assert isinstance(simulator._executed_trades, list)
        assert len(simulator._executed_trades) == 0
        assert isinstance(simulator._pending_orders, dict)
        assert len(simulator._pending_orders) == 0

    def test_simulator_initialization_with_slippage_model(self):
        """Test simulator initialization with slippage model."""
        config = self.create_default_config()
        slippage_model = MagicMock()
        simulator = TradeSimulator(config, slippage_model)
        
        assert simulator.slippage_model == slippage_model

    def test_simulator_state_initialization(self):
        """Test that simulator initializes proper state."""
        config = self.create_default_config()
        simulator = TradeSimulator(config)
        
        # Check internal state is properly initialized
        assert hasattr(simulator, '_order_book')
        assert hasattr(simulator, '_executed_trades') 
        assert hasattr(simulator, '_pending_orders')
        
        # Should be empty initially
        assert not simulator._order_book
        assert not simulator._executed_trades
        assert not simulator._pending_orders

    def test_simulator_config_access(self):
        """Test that simulator properly accesses config values."""
        config = SimulationConfig(
            initial_capital=Decimal("25000"),
            commission_rate=Decimal("0.0015"),
            enable_shorting=True,
            max_positions=8
        )
        simulator = TradeSimulator(config)
        
        assert simulator.config.initial_capital == Decimal("25000")
        assert simulator.config.commission_rate == Decimal("0.0015")
        assert simulator.config.enable_shorting is True
        assert simulator.config.max_positions == 8

    def test_simulator_order_book_structure(self):
        """Test simulator order book structure."""
        config = self.create_default_config()
        simulator = TradeSimulator(config)
        
        # Order book should be dict[str, dict[str, list[SimulatedOrder]]]
        assert isinstance(simulator._order_book, dict)
        
        # Should be empty initially
        assert len(simulator._order_book) == 0

    def test_simulator_executed_trades_structure(self):
        """Test simulator executed trades structure."""
        config = self.create_default_config()
        simulator = TradeSimulator(config)
        
        # Executed trades should be list[dict[str, Any]]
        assert isinstance(simulator._executed_trades, list)
        
        # Should be empty initially
        assert len(simulator._executed_trades) == 0

    def test_simulator_pending_orders_structure(self):
        """Test simulator pending orders structure."""
        config = self.create_default_config()
        simulator = TradeSimulator(config)
        
        # Pending orders should be dict[str, SimulatedOrder]
        assert isinstance(simulator._pending_orders, dict)
        
        # Should be empty initially
        assert len(simulator._pending_orders) == 0

    def test_simulator_configuration_variations(self):
        """Test simulator with different configurations."""
        # Test with minimal config
        minimal_config = SimulationConfig(
            enable_partial_fills=False,
            enable_market_impact=False,
            enable_latency=False
        )
        simulator1 = TradeSimulator(minimal_config)
        
        assert simulator1.config.enable_partial_fills is False
        assert simulator1.config.enable_market_impact is False
        assert simulator1.config.enable_latency is False
        
        # Test with maximal config
        maximal_config = SimulationConfig(
            enable_partial_fills=True,
            enable_market_impact=True,
            enable_latency=True,
            enable_shorting=True,
            max_positions=20,
            rejection_probability=Decimal("0.05")
        )
        simulator2 = TradeSimulator(maximal_config)
        
        assert simulator2.config.enable_partial_fills is True
        assert simulator2.config.enable_market_impact is True
        assert simulator2.config.enable_latency is True
        assert simulator2.config.enable_shorting is True
        assert simulator2.config.max_positions == 20
        assert simulator2.config.rejection_probability == Decimal("0.05")

    def test_simulator_with_custom_slippage_model(self):
        """Test simulator with custom slippage model."""
        config = self.create_default_config()
        
        # Create mock slippage model with specific methods
        slippage_model = MagicMock()
        slippage_model.calculate_slippage.return_value = Decimal("0.001")
        
        simulator = TradeSimulator(config, slippage_model)
        
        assert simulator.slippage_model == slippage_model
        assert hasattr(simulator.slippage_model, 'calculate_slippage')

    def test_simulator_order_book_initialization(self):
        """Test order book initialization for different symbols."""
        config = self.create_default_config()
        simulator = TradeSimulator(config)
        
        # Order book should be able to handle multiple symbols
        # Test that it's properly structured for extension
        assert isinstance(simulator._order_book, dict)
        
        # Should support nested structure: symbol -> side -> orders
        # This is tested implicitly by the type annotation


class TestSimulatorIntegration:
    """Integration tests for simulator components."""

    def test_config_order_simulator_integration(self):
        """Test that config, order, and simulator work together."""
        # Create config
        config = SimulationConfig(
            initial_capital=Decimal("100000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            max_positions=5
        )
        
        # Create order request
        request = OrderRequest(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            price=Decimal("3000"),
            exchange="binance"
        )
        
        # Create simulated order
        order = SimulatedOrder(
            request=request,
            order_id="test-order-123",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Create simulator
        simulator = TradeSimulator(config)
        
        # Verify all components are properly created and configured
        assert simulator.config.initial_capital == Decimal("100000")
        assert order.request.symbol == "ETHUSDT"
        assert order.request.side == OrderSide.SELL
        assert order.request.quantity == Decimal("10.0")
        assert order.status == OrderStatus.PENDING

    def test_simulator_edge_cases(self):
        """Test simulator edge cases and boundary conditions."""
        # Test with zero values
        config = SimulationConfig(
            initial_capital=Decimal("0"),
            commission_rate=Decimal("0"),
            slippage_rate=Decimal("0"),
            max_positions=1,
            rejection_probability=Decimal("0")
        )
        
        simulator = TradeSimulator(config)
        
        assert simulator.config.initial_capital == Decimal("0")
        assert simulator.config.commission_rate == Decimal("0")
        assert simulator.config.max_positions == 1
        
        # Test with maximum reasonable values
        config2 = SimulationConfig(
            initial_capital=Decimal("1000000"),
            commission_rate=Decimal("0.01"),  # 1%
            slippage_rate=Decimal("0.01"),    # 1%
            max_positions=100,
            rejection_probability=Decimal("0.1")  # 10%
        )
        
        simulator2 = TradeSimulator(config2)
        
        assert simulator2.config.initial_capital == Decimal("1000000")
        assert simulator2.config.commission_rate == Decimal("0.01")
        assert simulator2.config.max_positions == 100

    def test_order_status_lifecycle(self):
        """Test order status progression."""
        # Test all possible order statuses
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            exchange="binance"
        )
        
        # Test PENDING status (default)
        order1 = SimulatedOrder(
            request=request,
            order_id="pending-order",
            timestamp=datetime.now(timezone.utc)
        )
        assert order1.status == OrderStatus.PENDING
        
        # Test PARTIALLY_FILLED status
        order2 = SimulatedOrder(
            request=request,
            order_id="partial-order",
            timestamp=datetime.now(timezone.utc),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.5")
        )
        assert order2.status == OrderStatus.PARTIALLY_FILLED
        assert order2.filled_quantity == Decimal("0.5")
        
        # Test FILLED status
        order3 = SimulatedOrder(
            request=request,
            order_id="filled-order",
            timestamp=datetime.now(timezone.utc),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            average_fill_price=Decimal("50100")
        )
        assert order3.status == OrderStatus.FILLED
        assert order3.filled_quantity == Decimal("1.0")
        assert order3.average_fill_price == Decimal("50100")

    def test_execution_algorithms(self):
        """Test different execution algorithms."""
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5.0"),
            price=Decimal("45000"),
            exchange="coinbase"
        )
        
        # Test MARKET algorithm (default)
        order1 = SimulatedOrder(
            request=request,
            order_id="market-order",
            timestamp=datetime.now(timezone.utc)
        )
        assert order1.execution_algorithm == ExecutionAlgorithm.MARKET
        
        # Test TWAP algorithm
        order2 = SimulatedOrder(
            request=request,
            order_id="twap-order",
            timestamp=datetime.now(timezone.utc),
            execution_algorithm=ExecutionAlgorithm.TWAP,
            algorithm_params={"duration": 600, "slices": 20}
        )
        assert order2.execution_algorithm == ExecutionAlgorithm.TWAP
        assert order2.algorithm_params["duration"] == 600
        assert order2.algorithm_params["slices"] == 20
        
        # Test VWAP algorithm
        order3 = SimulatedOrder(
            request=request,
            order_id="vwap-order", 
            timestamp=datetime.now(timezone.utc),
            execution_algorithm=ExecutionAlgorithm.VWAP,
            algorithm_params={"participation_rate": 0.1, "min_fill_size": 0.1}
        )
        assert order3.execution_algorithm == ExecutionAlgorithm.VWAP
        assert order3.algorithm_params["participation_rate"] == 0.1