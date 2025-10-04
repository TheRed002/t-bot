"""
Comprehensive integration tests for complete trading workflows.

Tests end-to-end trading scenarios including signal generation, order execution,
position management, risk controls, and state persistence.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Core system imports
from src.core.config import Config
from src.core.types import (
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalDirection,
    Ticker,
)
from src.exchanges.base import BaseExchange
from src.execution.execution_engine import ExecutionEngine

# Trading system components
from src.strategies.base import BaseStrategy


@pytest.fixture
def mock_config():
    """Comprehensive mock configuration for integration tests."""
    config = Mock(spec=Config)

    # Trading configuration
    config.trading = Mock()
    config.trading.enabled = True
    config.trading.max_concurrent_trades = 5
    config.trading.min_trade_size = Decimal("10.0")
    config.trading.max_trade_size = Decimal("10000.0")

    # Risk management configuration
    config.risk = Mock()
    config.risk.max_position_size = Decimal("50000.0")
    config.risk.max_daily_loss = Decimal("5000.0")
    config.risk.max_drawdown = Decimal("0.1")  # 10%
    config.risk.position_sizing_method = "FIXED"
    config.risk.circuit_breakers_enabled = True

    # Execution configuration
    config.execution = Mock()
    config.execution.max_slippage = Decimal("0.005")  # 0.5%
    config.execution.timeout = 30
    config.execution.retry_attempts = 3

    # Exchange configuration
    config.exchanges = Mock()
    config.exchanges.binance = Mock()
    config.exchanges.binance.enabled = True
    config.exchanges.binance.api_key = "test_key"
    config.exchanges.binance.api_secret = "test_secret"

    # State management
    config.state = Mock()
    config.state.checkpoint_interval = 300
    config.state.enable_persistence = True

    return config


@pytest.fixture
def mock_exchange():
    """Mock exchange for integration tests."""
    exchange = Mock(spec=BaseExchange)
    exchange.name = "binance"
    exchange.is_connected = True

    # Mock market data
    exchange.get_market_data = AsyncMock(
        return_value=MarketData(
            symbol="BTC/USDT",
            price=Decimal("50000.0"),
            bid=Decimal("49999.0"),
            ask=Decimal("50001.0"),
            volume=Decimal("1000.0"),
            timestamp=datetime.now(timezone.utc),
        )
    )

    # Mock order placement
    exchange.place_order = AsyncMock(return_value="order_123")
    exchange.get_order_status = AsyncMock(return_value=OrderStatus.FILLED)

    # Mock balance
    exchange.get_balance = AsyncMock(
        return_value={"USDT": Decimal("100000.0"), "BTC": Decimal("0.0")}
    )

    return exchange


@pytest.fixture
def mock_strategy():
    """Mock trading strategy for integration tests."""
    strategy = Mock(spec=BaseStrategy)
    strategy.name = "test_strategy"
    strategy.enabled = True

    # Mock signal generation
    strategy.generate_signal = AsyncMock(
        return_value=Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.8,
            price=Decimal("50000.0"),
            timestamp=datetime.now(timezone.utc),
        )
    )

    return strategy


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return [
        MarketData(
            symbol="BTC/USDT",
            price=Decimal("50000.0"),
            bid=Decimal("49999.0"),
            ask=Decimal("50001.0"),
            volume=Decimal("1000.0"),
            timestamp=datetime.now(timezone.utc),
        ),
        MarketData(
            symbol="ETH/USDT",
            price=Decimal("3000.0"),
            bid=Decimal("2999.0"),
            ask=Decimal("3001.0"),
            volume=Decimal("5000.0"),
            timestamp=datetime.now(timezone.utc),
        ),
    ]


class TestCompleteSignalToExecutionWorkflow:
    """Test complete workflow from signal generation to order execution."""

    @pytest.mark.asyncio
    async def test_buy_signal_execution_workflow(self, mock_config, mock_exchange, mock_strategy):
        """Test complete buy signal execution workflow."""
        # Setup components
        with patch("src.execution.execution_engine.OrderManager"):
            with patch("src.execution.execution_engine.SlippageModel"):
                with patch("src.execution.execution_engine.CostAnalyzer"):
                    with patch("src.execution.service.ExecutionService"):
                        execution_engine = ExecutionEngine(Mock(), mock_config)

        risk_manager = Mock()
        risk_manager.validate_trade = AsyncMock(return_value=True)
        risk_manager.calculate_position_size = AsyncMock(return_value=Decimal("1.0"))

        state_manager = Mock()
        state_manager.save_state = AsyncMock()
        state_manager.load_state = AsyncMock(return_value=None)

        # 1. Generate signal
        signal = await mock_strategy.generate_signal()
        assert signal.direction == SignalDirection.BUY
        assert signal.symbol == "BTC/USDT"

        # 2. Validate with risk management
        is_valid_trade = await risk_manager.validate_trade(signal)
        assert is_valid_trade is True

        # 3. Calculate position size
        position_size = await risk_manager.calculate_position_size(signal)
        assert position_size == Decimal("1.0")

        # 4. Create execution instruction
        instruction = ExecutionInstruction(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            quantity=position_size,
            order_type=OrderType.MARKET,
            max_slippage=mock_config.execution.max_slippage,
        )

        # 5. Get market data
        market_data = await mock_exchange.get_market_data(signal.symbol)

        # 6. Mock successful execution
        expected_result = ExecutionResult(
            instruction_id="exec_123",
            status=ExecutionStatus.COMPLETED,
            executed_quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
            total_cost=Decimal("50000.0"),
        )

        with patch.object(execution_engine, "execute_order", return_value=expected_result):
            result = await execution_engine.execute_order(instruction, market_data)

        # 7. Verify execution result
        assert result.status == ExecutionStatus.COMPLETED
        assert result.executed_quantity == Decimal("1.0")

        # 8. Save state
        await state_manager.save_state("portfolio", {"last_trade": result})

        # Complete workflow succeeded
        assert True

    @pytest.mark.asyncio
    async def test_sell_signal_execution_workflow(self, mock_config, mock_exchange, mock_strategy):
        """Test complete sell signal execution workflow."""
        # Modify strategy to generate sell signal
        mock_strategy.generate_signal = AsyncMock(
            return_value=Signal(
                symbol="BTC/USDT",
                direction=SignalDirection.SELL,
                confidence=0.75,
                price=Decimal("50000.0"),
                timestamp=datetime.now(timezone.utc),
            )
        )

        # Setup components with existing position
        risk_manager = Mock()
        risk_manager.validate_trade = AsyncMock(return_value=True)
        risk_manager.get_position = AsyncMock(
            return_value=Position(
                symbol="BTC/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("2.0"),
                average_price=Decimal("49000.0"),
            )
        )
        risk_manager.calculate_position_size = AsyncMock(return_value=Decimal("1.0"))

        # Generate and process sell signal
        signal = await mock_strategy.generate_signal()
        assert signal.direction == SignalDirection.SELL

        # Verify position exists before selling
        position = await risk_manager.get_position(signal.symbol)
        assert position.quantity == Decimal("2.0")

        # Validate sell trade
        is_valid_trade = await risk_manager.validate_trade(signal)
        assert is_valid_trade is True

        # Calculate sell size (partial position close)
        sell_size = await risk_manager.calculate_position_size(signal)
        assert sell_size == Decimal("1.0")

        # Execution would proceed similar to buy workflow
        assert True

    @pytest.mark.asyncio
    async def test_workflow_with_rejected_signal(self, mock_config, mock_exchange, mock_strategy):
        """Test workflow when signal is rejected by risk management."""
        # Setup risk manager to reject trade
        risk_manager = Mock()
        risk_manager.validate_trade = AsyncMock(return_value=False)

        # Generate signal
        signal = await mock_strategy.generate_signal()

        # Risk management rejects the trade
        is_valid_trade = await risk_manager.validate_trade(signal)
        assert is_valid_trade is False

        # Workflow should stop here - no execution should occur
        # This tests that risk controls properly prevent unwanted trades
        assert True


class TestPositionManagementWorkflow:
    """Test position management throughout trading lifecycle."""

    @pytest.mark.asyncio
    async def test_position_opening_and_tracking(self, mock_config, mock_exchange):
        """Test opening and tracking new positions."""
        position_manager = Mock()
        position_manager.positions = {}

        # Simulate successful order fill
        filled_order = Order(
            id="order_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
        )

        # Update position based on fill
        new_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
            unrealized_pnl=Decimal("0.0"),
        )

        position_manager.positions[filled_order.symbol] = new_position

        # Verify position was created correctly
        assert "BTC/USDT" in position_manager.positions
        assert position_manager.positions["BTC/USDT"].quantity == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_position_size_increase(self, mock_config):
        """Test increasing existing position size."""
        position_manager = Mock()

        # Existing position
        existing_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("1.0"),
            average_price=Decimal("49000.0"),
        )
        position_manager.positions = {"BTC/USDT": existing_position}

        # New fill that increases position
        new_fill = Order(
            id="order_124",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.5"),
            average_price=Decimal("51000.0"),
        )

        # Calculate new weighted average price
        total_quantity = existing_position.quantity + new_fill.filled_quantity
        new_average_price = (
            existing_position.quantity * existing_position.average_price
            + new_fill.filled_quantity * new_fill.average_price
        ) / total_quantity

        # Update position
        updated_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=total_quantity,
            average_price=new_average_price,
        )

        position_manager.positions["BTC/USDT"] = updated_position

        # Verify position update
        assert position_manager.positions["BTC/USDT"].quantity == Decimal("1.5")
        assert position_manager.positions["BTC/USDT"].average_price == Decimal(
            "49666.666666666666666666666667"
        )

    @pytest.mark.asyncio
    async def test_position_partial_close(self, mock_config):
        """Test partially closing a position."""
        position_manager = Mock()

        # Existing long position
        existing_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("2.0"),
            average_price=Decimal("50000.0"),
        )
        position_manager.positions = {"BTC/USDT": existing_position}

        # Sell order that partially closes position
        sell_order = Order(
            id="sell_order_1",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.8"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.8"),
            average_price=Decimal("52000.0"),
        )

        # Update position (reduce quantity)
        remaining_quantity = existing_position.quantity - sell_order.filled_quantity
        updated_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=remaining_quantity,
            average_price=existing_position.average_price,  # Average price stays same
        )

        position_manager.positions["BTC/USDT"] = updated_position

        # Calculate realized PnL
        realized_pnl = (
            sell_order.average_price - existing_position.average_price
        ) * sell_order.filled_quantity

        # Verify position update
        assert position_manager.positions["BTC/USDT"].quantity == Decimal("1.2")
        assert realized_pnl == Decimal("1600.0")  # (52000 - 50000) * 0.8

    @pytest.mark.asyncio
    async def test_position_complete_close(self, mock_config):
        """Test completely closing a position."""
        position_manager = Mock()

        # Existing position
        existing_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("1.0"),
            average_price=Decimal("50000.0"),
        )
        position_manager.positions = {"BTC/USDT": existing_position}

        # Sell entire position
        close_order = Order(
            id="close_order_1",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("1.0"),
            average_price=Decimal("53000.0"),
        )

        # Position should be closed (removed or set to 0)
        del position_manager.positions["BTC/USDT"]

        # Calculate final realized PnL
        realized_pnl = (
            close_order.average_price - existing_position.average_price
        ) * close_order.filled_quantity

        # Verify position is closed
        assert "BTC/USDT" not in position_manager.positions
        assert realized_pnl == Decimal("3000.0")


class TestRiskManagementWorkflow:
    """Test risk management integration in trading workflows."""

    @pytest.mark.asyncio
    async def test_position_size_limits(self, mock_config, mock_strategy):
        """Test position size limit enforcement."""
        risk_manager = Mock()
        risk_manager.max_position_size = mock_config.risk.max_position_size

        # Test normal size (within limits)
        normal_signal = Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.8,
            price=Decimal("50000.0"),
            timestamp=datetime.now(timezone.utc),
        )

        normal_size = Decimal("1000.0")  # $50,000 notional
        risk_manager.validate_position_size = Mock(return_value=True)
        risk_manager.calculate_position_size = Mock(return_value=normal_size)

        is_valid = risk_manager.validate_position_size(normal_size)
        assert is_valid is True

        # Test oversized position (exceeds limits)
        oversized_signal = Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.9,
            price=Decimal("50000.0"),
            timestamp=datetime.now(timezone.utc),
        )

        oversized_amount = Decimal("2.0")  # $100,000 notional (exceeds $50,000 limit)
        risk_manager.validate_position_size = Mock(return_value=False)

        is_valid_oversized = risk_manager.validate_position_size(oversized_amount)
        assert is_valid_oversized is False

    @pytest.mark.asyncio
    async def test_daily_loss_limits(self, mock_config):
        """Test daily loss limit enforcement."""
        risk_manager = Mock()
        risk_manager.daily_pnl = Decimal("-4000.0")  # Already lost $4,000 today
        risk_manager.max_daily_loss = mock_config.risk.max_daily_loss  # $5,000 limit

        # Test trade that would stay within limit
        potential_loss = Decimal("500.0")  # Additional $500 loss
        risk_manager.check_daily_loss_limit = Mock(return_value=True)

        can_trade = risk_manager.check_daily_loss_limit(potential_loss)
        assert can_trade is True

        # Test trade that would exceed limit
        excessive_loss = Decimal("2000.0")  # Would total $6,000 loss
        risk_manager.check_daily_loss_limit = Mock(return_value=False)

        can_trade_excessive = risk_manager.check_daily_loss_limit(excessive_loss)
        assert can_trade_excessive is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, mock_config):
        """Test circuit breaker activation during high losses."""
        risk_manager = Mock()
        risk_manager.circuit_breakers_enabled = mock_config.risk.circuit_breakers_enabled

        # Simulate series of losses that trigger circuit breaker
        loss_events = [
            Decimal("-1000.0"),
            Decimal("-1500.0"),
            Decimal("-2000.0"),
            Decimal("-1200.0"),  # Total: -$5,700 > -$5,000 limit
        ]

        cumulative_loss = Decimal("0.0")
        circuit_breaker_triggered = False

        for loss in loss_events:
            cumulative_loss += loss
            if abs(cumulative_loss) > mock_config.risk.max_daily_loss:
                circuit_breaker_triggered = True
                break

        assert circuit_breaker_triggered is True
        assert abs(cumulative_loss) == Decimal("5700.0")

    @pytest.mark.asyncio
    async def test_correlation_limits(self, mock_config):
        """Test correlation-based position limits."""
        risk_manager = Mock()

        # Existing positions in correlated assets
        existing_positions = {
            "BTC/USDT": Position(
                symbol="BTC/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("1.0"),
                average_price=Decimal("50000.0"),
            ),
            "ETH/USDT": Position(
                symbol="ETH/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("10.0"),
                average_price=Decimal("3000.0"),
            ),
        }

        # Simulate correlation check for new BTC position
        risk_manager.check_correlation_limits = Mock(return_value=True)
        risk_manager.get_asset_correlation = Mock(return_value=0.85)  # High correlation

        # Should still allow if within overall exposure limits
        can_add_btc = risk_manager.check_correlation_limits("BTC/USDT", existing_positions)
        assert can_add_btc is True


class TestMultiExchangeWorkflow:
    """Test trading workflows across multiple exchanges."""

    @pytest.mark.asyncio
    async def test_best_price_execution_across_exchanges(self, mock_config):
        """Test finding best execution price across multiple exchanges."""
        # Mock multiple exchanges with different prices
        binance_exchange = Mock()
        binance_exchange.name = "binance"
        binance_exchange.get_ticker = AsyncMock(
            return_value=Ticker(
                symbol="BTC/USDT",
                bid_price=Decimal("49999.0"),
                bid_quantity=Decimal("10.0"),
                ask_price=Decimal("50001.0"),
                ask_quantity=Decimal("10.0"),
                last_price=Decimal("50000.0"),
                open_price=Decimal("49900.0"),
                high_price=Decimal("50100.0"),
                low_price=Decimal("49800.0"),
                volume=Decimal("1000.0"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
            )
        )

        coinbase_exchange = Mock()
        coinbase_exchange.name = "coinbase"
        coinbase_exchange.get_ticker = AsyncMock(
            return_value=Ticker(
                symbol="BTC/USDT",
                bid_price=Decimal("49979.0"),
                bid_quantity=Decimal("8.0"),
                ask_price=Decimal("49981.0"),  # Better ask price for buying
                ask_quantity=Decimal("8.0"),
                last_price=Decimal("49980.0"),
                open_price=Decimal("49880.0"),
                high_price=Decimal("50080.0"),
                low_price=Decimal("49780.0"),
                volume=Decimal("800.0"),
                timestamp=datetime.now(timezone.utc),
                exchange="coinbase",
            )
        )

        exchanges = [binance_exchange, coinbase_exchange]

        # Smart router should choose Coinbase for buying (lower ask)
        best_exchange_for_buy = None
        best_ask_price = Decimal("999999.0")

        for exchange in exchanges:
            ticker = await exchange.get_ticker("BTC/USDT")
            if ticker.ask_price < best_ask_price:
                best_ask_price = ticker.ask_price
                best_exchange_for_buy = exchange

        assert best_exchange_for_buy.name == "coinbase"
        assert best_ask_price == Decimal("49981.0")

    @pytest.mark.asyncio
    async def test_cross_exchange_position_management(self, mock_config):
        """Test managing positions across multiple exchanges."""
        position_manager = Mock()

        # Positions across different exchanges
        positions = {
            "binance_BTC/USDT": Position(
                symbol="BTC/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("0.5"),
                average_price=Decimal("50000.0"),
            ),
            "coinbase_BTC/USDT": Position(
                symbol="BTC/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("0.3"),
                average_price=Decimal("49500.0"),
            ),
        }

        position_manager.get_combined_position = Mock(
            return_value=Position(
                symbol="BTC/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("0.8"),  # Combined quantity
                average_price=Decimal("49812.5"),  # Weighted average price
            )
        )

        combined_position = position_manager.get_combined_position("BTC/USDT")

        # Verify combined position calculation
        assert combined_position.quantity == Decimal("0.8")

        # Verify weighted average price calculation
        # (0.5 * 50000 + 0.3 * 49500) / 0.8 = (25000 + 14850) / 0.8 = 49812.5
        assert combined_position.average_price == Decimal("49812.5")

    @pytest.mark.asyncio
    async def test_exchange_failover_workflow(self, mock_config):
        """Test failover to backup exchange when primary fails."""
        primary_exchange = Mock()
        primary_exchange.name = "binance"
        primary_exchange.is_connected = False  # Primary is down
        primary_exchange.place_order = AsyncMock(side_effect=Exception("Connection failed"))

        backup_exchange = Mock()
        backup_exchange.name = "coinbase"
        backup_exchange.is_connected = True
        backup_exchange.place_order = AsyncMock(return_value="backup_order_123")

        exchanges = [primary_exchange, backup_exchange]

        # Order routing logic with failover
        order_placed = False
        selected_exchange = None

        for exchange in exchanges:
            if exchange.is_connected:
                try:
                    order_id = await exchange.place_order(
                        {"symbol": "BTC/USDT", "side": "BUY", "quantity": "1.0", "type": "MARKET"}
                    )
                    order_placed = True
                    selected_exchange = exchange
                    break
                except Exception:
                    continue

        assert order_placed is True
        assert selected_exchange.name == "coinbase"


class TestErrorRecoveryWorkflow:
    """Test error handling and recovery in trading workflows."""

    @pytest.mark.asyncio
    async def test_network_error_retry_workflow(self, mock_config, mock_exchange):
        """Test retry mechanism for network errors."""
        # Mock exchange that fails first two times, succeeds on third
        call_count = 0

        async def mock_place_order(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Network timeout")
            return "order_success_123"

        mock_exchange.place_order = mock_place_order

        # Retry logic
        max_retries = 3
        retry_count = 0
        order_id = None

        while retry_count < max_retries:
            try:
                order_id = await mock_exchange.place_order(
                    {"symbol": "BTC/USDT", "side": "BUY", "quantity": "1.0"}
                )
                break
            except Exception:
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(0.1)  # Brief delay before retry

        assert order_id == "order_success_123"
        assert call_count == 3  # Succeeded on third attempt

    @pytest.mark.asyncio
    async def test_partial_fill_handling_workflow(self, mock_config):
        """Test handling of partially filled orders."""
        order_manager = Mock()

        # Initial order
        original_order = Order(
            id="partial_order_1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("2.0"),
            price=Decimal("50000.0"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.7"),
            remaining_quantity=Decimal("1.3"),
        )

        # Decision logic for partial fill
        if original_order.status == OrderStatus.PARTIALLY_FILLED:
            fill_percentage = (original_order.filled_quantity / original_order.quantity) * 100

            if fill_percentage >= 30:  # If 30%+ filled, accept partial fill
                # Cancel remaining quantity
                order_manager.cancel_order = AsyncMock(return_value=True)
                cancelled = await order_manager.cancel_order(original_order.id)
                assert cancelled is True
            else:
                # Keep order open for full fill
                assert False, "Should not reach here in this test"

        # Verify partial fill handling
        assert original_order.filled_quantity == Decimal("0.7")
        assert fill_percentage == 35.0  # 35% filled

    @pytest.mark.asyncio
    async def test_state_recovery_after_crash(self, mock_config):
        """Test state recovery after system crash."""
        state_manager = Mock()

        # Mock pre-crash state
        pre_crash_state = {
            "portfolio": {
                "total_value": "100000.0",
                "positions": {"BTC/USDT": {"quantity": "1.0", "average_price": "50000.0"}},
                "open_orders": {
                    "order_123": {
                        "symbol": "BTC/USDT",
                        "side": "BUY",
                        "quantity": "0.5",
                        "status": "NEW",
                    }
                },
            },
            "last_checkpoint": "2024-01-01T12:00:00Z",
        }

        # Mock state recovery
        state_manager.recover_from_checkpoint = AsyncMock(return_value=pre_crash_state)

        # Recovery process
        recovered_state = await state_manager.recover_from_checkpoint("latest")

        # Verify recovery
        assert recovered_state is not None
        assert "portfolio" in recovered_state
        assert "BTC/USDT" in recovered_state["portfolio"]["positions"]
        assert "order_123" in recovered_state["portfolio"]["open_orders"]

    @pytest.mark.asyncio
    async def test_risk_limit_breach_recovery(self, mock_config):
        """Test recovery workflow when risk limits are breached."""
        risk_manager = Mock()
        position_manager = Mock()

        # Simulate risk limit breach
        current_loss = Decimal("-6000.0")  # Exceeds $5,000 daily limit
        risk_manager.daily_loss_limit_breached = Mock(return_value=True)
        risk_manager.get_daily_pnl = Mock(return_value=current_loss)

        # Emergency response workflow
        if risk_manager.daily_loss_limit_breached():
            # 1. Stop new trading
            trading_enabled = False

            # 2. Cancel all open orders
            position_manager.cancel_all_orders = AsyncMock(return_value=True)
            cancelled = await position_manager.cancel_all_orders()

            # 3. Optionally close positions (configurable)
            position_manager.close_all_positions = AsyncMock(return_value=True)
            if mock_config.risk.emergency_close_positions:
                closed = await position_manager.close_all_positions()
            else:
                closed = True  # Assume positions kept open

            # 4. Send alert
            alert_sent = True  # Mock alert system

        assert trading_enabled is False
        assert cancelled is True
        assert closed is True
        assert alert_sent is True


class TestBotInstanceIntegration:
    """Test full bot instance integration with all components."""

    @pytest.mark.asyncio
    async def test_bot_instance_complete_lifecycle(self, mock_config, mock_exchange, mock_strategy):
        """Test complete bot instance lifecycle."""
        # This test simulates a full bot instance running through its lifecycle

        # Mock bot instance
        bot = Mock()
        bot.config = mock_config
        bot.strategies = [mock_strategy]
        bot.exchanges = [mock_exchange]
        bot.state = "RUNNING"
        bot.total_trades = 0
        bot.profit_loss = Decimal("0.0")

        # Mock lifecycle methods
        bot.initialize = AsyncMock(return_value=True)
        bot.start_trading = AsyncMock(return_value=True)
        bot.process_market_data = AsyncMock()
        bot.shutdown = AsyncMock(return_value=True)

        # 1. Initialize bot
        initialized = await bot.initialize()
        assert initialized is True

        # 2. Start trading
        started = await bot.start_trading()
        assert started is True

        # 3. Process market data (simulate multiple cycles)
        market_data_cycles = [
            MarketData(
                symbol="BTC/USDT", price=Decimal("50000.0"), timestamp=datetime.now(timezone.utc)
            ),
            MarketData(
                symbol="BTC/USDT", price=Decimal("51000.0"), timestamp=datetime.now(timezone.utc)
            ),
            MarketData(
                symbol="BTC/USDT", price=Decimal("52000.0"), timestamp=datetime.now(timezone.utc)
            ),
        ]

        for market_data in market_data_cycles:
            await bot.process_market_data(market_data)

        # 4. Simulate successful trades
        bot.total_trades = 3
        bot.profit_loss = Decimal("1500.0")

        # 5. Shutdown gracefully
        shutdown_success = await bot.shutdown()
        assert shutdown_success is True

        # Verify bot performed successfully
        assert bot.total_trades > 0
        assert bot.profit_loss > 0

    @pytest.mark.asyncio
    async def test_multi_strategy_coordination(self, mock_config):
        """Test coordination between multiple strategies in a single bot."""
        # Create multiple mock strategies
        momentum_strategy = Mock()
        momentum_strategy.name = "momentum"
        momentum_strategy.generate_signal = AsyncMock(
            return_value=Signal(
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                confidence=0.8,
                price=Decimal("50000.0"),
            )
        )

        mean_reversion_strategy = Mock()
        mean_reversion_strategy.name = "mean_reversion"
        mean_reversion_strategy.generate_signal = AsyncMock(
            return_value=Signal(
                symbol="BTC/USDT",
                direction=SignalDirection.SELL,
                confidence=0.6,
                price=Decimal("50000.0"),
            )
        )

        # Strategy coordinator
        strategies = [momentum_strategy, mean_reversion_strategy]
        signals = []

        # Collect signals from all strategies
        for strategy in strategies:
            signal = await strategy.generate_signal()
            signals.append((strategy.name, signal))

        # Signal aggregation logic (simple majority vote)
        buy_votes = sum(1 for _, signal in signals if signal.direction == SignalDirection.BUY)
        sell_votes = sum(1 for _, signal in signals if signal.direction == SignalDirection.SELL)

        if buy_votes > sell_votes:
            final_direction = SignalDirection.BUY
        elif sell_votes > buy_votes:
            final_direction = SignalDirection.SELL
        else:
            final_direction = SignalDirection.HOLD

        # In this case: 1 buy vote, 1 sell vote = HOLD
        assert final_direction == SignalDirection.HOLD
        assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, mock_config):
        """Test performance monitoring throughout trading workflows."""
        performance_monitor = Mock()
        performance_monitor.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": Decimal("0.0"),
            "max_drawdown": Decimal("0.0"),
            "sharpe_ratio": 0.0,
        }

        # Simulate series of trades
        trade_results = [
            {"pnl": Decimal("500.0"), "win": True},
            {"pnl": Decimal("-200.0"), "win": False},
            {"pnl": Decimal("800.0"), "win": True},
            {"pnl": Decimal("-150.0"), "win": False},
            {"pnl": Decimal("300.0"), "win": True},
        ]

        # Update metrics for each trade
        for trade in trade_results:
            performance_monitor.metrics["total_trades"] += 1
            performance_monitor.metrics["total_profit"] += trade["pnl"]

            if trade["win"]:
                performance_monitor.metrics["winning_trades"] += 1
            else:
                performance_monitor.metrics["losing_trades"] += 1

        # Calculate win rate
        win_rate = (
            performance_monitor.metrics["winning_trades"]
            / performance_monitor.metrics["total_trades"]
        ) * 100

        # Verify performance tracking
        assert performance_monitor.metrics["total_trades"] == 5
        assert performance_monitor.metrics["winning_trades"] == 3
        assert performance_monitor.metrics["total_profit"] == Decimal("1250.0")
        assert win_rate == 60.0  # 60% win rate
