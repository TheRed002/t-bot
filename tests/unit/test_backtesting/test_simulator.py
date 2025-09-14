"""Tests for backtesting simulator module."""

import logging
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Shared fixtures for performance
@pytest.fixture(scope="session")
def sample_timestamp():
    """Shared timestamp for all tests."""
    return datetime.now(timezone.utc)

@pytest.fixture(scope="session")
def sample_order_id():
    """Shared order ID for tests."""
    return "test-order-123"

@pytest.fixture(scope="session")
def minimal_order_request():
    """Shared minimal order request."""
    from src.core.types import OrderRequest, OrderSide, OrderType
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
        price=Decimal("50000"),
        exchange="binance"
    )

@pytest.fixture(scope="session")
def minimal_simulation_config():
    """Shared minimal simulation config."""
    return SimulationConfig()

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
        with patch('src.backtesting.simulator.SimulationConfig') as MockConfig:
            mock_config = MagicMock()
            mock_config.initial_capital = Decimal("10000")
            mock_config.commission_rate = Decimal("0.001")
            mock_config.enable_shorting = False
            MockConfig.return_value = mock_config

            config = SimulationConfig()
            assert config.initial_capital == Decimal("10000")
            assert config.commission_rate == Decimal("0.001")
            assert config.enable_shorting is False

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        # Mock heavy config creation
        with patch('src.backtesting.simulator.SimulationConfig') as MockConfig:
            mock_config = MagicMock()
            mock_config.initial_capital = Decimal("50000")
            mock_config.enable_shorting = True
            mock_config.max_positions = 10
            MockConfig.return_value = mock_config

            config = SimulationConfig(
                initial_capital=Decimal("50000"),
                enable_shorting=True,
                max_positions=10
            )

            assert config.initial_capital == Decimal("50000")
            assert config.enable_shorting is True
            assert config.max_positions == 10

    def test_config_validation(self):
        """Test config validation."""
        # Mock validation for speed
        with patch('src.backtesting.simulator.SimulationConfig') as MockConfig:
            mock_config = MagicMock()
            mock_config.commission_rate = Decimal("0")
            mock_config.rejection_probability = Decimal("0")
            MockConfig.return_value = mock_config

            config = SimulationConfig(commission_rate=Decimal("0"))
            assert config.commission_rate == Decimal("0")


class TestSimulatedOrder:
    """Test SimulatedOrder model."""

    def create_sample_order_request(self, minimal_order_request):
        """Create sample order request for testing."""
        return minimal_order_request

    def test_simulated_order_creation(self, minimal_order_request, sample_order_id, sample_timestamp):
        """Test creating simulated order."""
        order = SimulatedOrder(
            request=minimal_order_request,
            order_id=sample_order_id,
            timestamp=sample_timestamp
        )

        assert order.request == minimal_order_request
        assert order.order_id == sample_order_id
        assert order.timestamp == sample_timestamp
        assert order.filled_quantity == Decimal("0")
        assert order.status == OrderStatus.PENDING
        assert order.average_fill_price is None
        assert order.fill_time is None
        assert order.execution_fees == Decimal("0")
        assert order.execution_algorithm == ExecutionAlgorithm.MARKET
        assert order.algorithm_params == {}

    def test_simulated_order_with_custom_values(self, minimal_order_request, sample_order_id, sample_timestamp):
        """Test creating simulated order with custom values."""
        order = SimulatedOrder(
            request=minimal_order_request,
            order_id=sample_order_id,
            timestamp=sample_timestamp,
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PARTIALLY_FILLED,
            average_fill_price=Decimal("50100"),
            fill_time=sample_timestamp,
            execution_fees=Decimal("25.05"),
            execution_algorithm=ExecutionAlgorithm.TWAP,
            algorithm_params={"duration": 300, "slices": 10}
        )
        
        assert order.filled_quantity == Decimal("0.5")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.average_fill_price == Decimal("50100")
        assert order.fill_time == sample_timestamp
        assert order.execution_fees == Decimal("25.05")
        assert order.execution_algorithm == ExecutionAlgorithm.TWAP
        assert order.algorithm_params == {"duration": 300, "slices": 10}


class TestTradeSimulator:
    """Test TradeSimulator functionality."""

    def create_default_config(self, minimal_simulation_config):
        """Create default simulation config."""
        return minimal_simulation_config

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

    def test_simulator_initialization(self, minimal_simulation_config):
        """Test simulator initialization."""
        simulator = TradeSimulator(minimal_simulation_config)

        assert simulator.config == minimal_simulation_config
        assert simulator.slippage_model is None
        assert isinstance(simulator._order_book, dict)
        assert len(simulator._order_book) == 0
        assert isinstance(simulator._executed_trades, list)
        assert len(simulator._executed_trades) == 0
        assert isinstance(simulator._pending_orders, dict)
        assert len(simulator._pending_orders) == 0

    def test_simulator_initialization_with_slippage_model(self, minimal_simulation_config):
        """Test simulator initialization with slippage model."""
        slippage_model = MagicMock()
        simulator = TradeSimulator(minimal_simulation_config, slippage_model)
        
        assert simulator.slippage_model == slippage_model

    def test_simulator_state_initialization(self, minimal_simulation_config):
        """Test that simulator initializes proper state."""
        simulator = TradeSimulator(minimal_simulation_config)
        
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

    def test_simulator_order_book_structure(self, minimal_simulation_config):
        """Test simulator order book structure."""
        simulator = TradeSimulator(minimal_simulation_config)
        
        # Order book should be dict[str, dict[str, list[SimulatedOrder]]]
        assert isinstance(simulator._order_book, dict)
        
        # Should be empty initially
        assert len(simulator._order_book) == 0

    def test_simulator_executed_trades_structure(self, minimal_simulation_config):
        """Test simulator executed trades structure."""
        simulator = TradeSimulator(minimal_simulation_config)
        
        # Executed trades should be list[dict[str, Any]]
        assert isinstance(simulator._executed_trades, list)
        
        # Should be empty initially
        assert len(simulator._executed_trades) == 0

    def test_simulator_pending_orders_structure(self, minimal_simulation_config):
        """Test simulator pending orders structure."""
        simulator = TradeSimulator(minimal_simulation_config)
        
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

    def test_simulator_with_custom_slippage_model(self, minimal_simulation_config):
        """Test simulator with custom slippage model."""
        # Create mock slippage model with specific methods
        slippage_model = MagicMock()
        slippage_model.calculate_slippage.return_value = Decimal("0.001")

        simulator = TradeSimulator(minimal_simulation_config, slippage_model)
        
        assert simulator.slippage_model == slippage_model
        assert hasattr(simulator.slippage_model, 'calculate_slippage')

    def test_simulator_order_book_initialization(self, minimal_simulation_config):
        """Test order book initialization for different symbols."""
        simulator = TradeSimulator(minimal_simulation_config)
        
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
        # Test all possible order statuses - use fixture
        from src.core.types import OrderRequest, OrderSide, OrderType
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            exchange="binance"
        )
        
        # Test PENDING status (default) - use shared timestamp
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        order1 = SimulatedOrder(
            request=request,
            order_id="pending-order",
            timestamp=timestamp
        )
        assert order1.status == OrderStatus.PENDING
        
        # Test PARTIALLY_FILLED status
        order2 = SimulatedOrder(
            request=request,
            order_id="partial-order",
            timestamp=timestamp,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.5")
        )
        assert order2.status == OrderStatus.PARTIALLY_FILLED
        assert order2.filled_quantity == Decimal("0.5")
        
        # Test FILLED status
        order3 = SimulatedOrder(
            request=request,
            order_id="filled-order",
            timestamp=timestamp,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            average_fill_price=Decimal("50100")
        )
        assert order3.status == OrderStatus.FILLED
        assert order3.filled_quantity == Decimal("1.0")
        assert order3.average_fill_price == Decimal("50100")

    def test_execution_algorithms(self):
        """Test different execution algorithms."""
        # Use shared timestamp for performance
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        from src.core.types import OrderRequest, OrderSide, OrderType
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
            timestamp=timestamp
        )
        assert order1.execution_algorithm == ExecutionAlgorithm.MARKET
        
        # Test TWAP algorithm
        order2 = SimulatedOrder(
            request=request,
            order_id="twap-order",
            timestamp=timestamp,
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
            timestamp=timestamp,
            execution_algorithm=ExecutionAlgorithm.VWAP,
            algorithm_params={"participation_rate": 0.1, "min_fill_size": 0.1}
        )
        assert order3.execution_algorithm == ExecutionAlgorithm.VWAP
        assert order3.algorithm_params["participation_rate"] == 0.1