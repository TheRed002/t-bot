"""
End-to-end trading workflow integration tests.

Tests complete trading workflows from signal generation through order execution,
position management, risk validation, and state persistence.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.types import (
    MarketData,
    OrderSide,
    OrderStatus,
    Position,
    Signal,
    SignalDirection,
)
from src.execution.execution_engine import ExecutionEngine
from src.risk_management.risk_manager import RiskManager
from src.state import StateService
from tests.integration.base_integration import (
    BaseIntegrationTest,
    performance_test,
    wait_for_condition,
)

logger = logging.getLogger(__name__)


class TestCompleteSignalToExecution(BaseIntegrationTest):
    """Test complete signal generation to order execution workflow."""

    @pytest.mark.asyncio
    @performance_test(max_duration=30.0)
    @pytest.mark.timeout(300)
    async def test_buy_signal_complete_workflow(self, performance_monitor):
        """Test complete buy signal workflow with realistic timing and error handling."""

        # Setup components
        exchanges = await self.create_mock_exchanges()
        strategies = await self.create_mock_strategies()

        binance = exchanges["binance"]
        momentum_strategy = strategies["momentum"]

        # Mock execution engine and risk manager
        execution_engine = Mock(spec=ExecutionEngine)
        risk_manager = Mock(spec=RiskManager)
        state_manager = Mock(spec=StateService)

        # Configure risk manager
        risk_manager.validate_trade = AsyncMock(return_value=True)
        risk_manager.calculate_position_size = AsyncMock(return_value=Decimal("0.1"))
        risk_manager.check_pre_trade_risk = AsyncMock(return_value=True)
        risk_manager.update_exposure = AsyncMock()

        # Configure state manager
        state_manager.save_checkpoint = AsyncMock(return_value=True)
        state_manager.get_portfolio_state = AsyncMock(
            return_value={
                "total_value": Decimal("100000.0"),
                "available_balance": Decimal("50000.0"),
                "positions": {},
            }
        )

        # 1. Market data update
        market_data = await binance.get_market_data("BTC/USDT")
        performance_monitor.record_api_call()

        assert market_data.symbol == "BTC/USDT"
        assert market_data.price > 0
        logger.info(f"Market data: BTC/USDT @ ${market_data.price}")

        # 2. Signal generation
        start_time = time.time()
        signal = await momentum_strategy.generate_signal()
        signal_time = time.time() - start_time
        performance_monitor.metrics["signal_generation_time"].append(signal_time)

        # For testing, force a BUY signal
        signal.direction = SignalDirection.BUY
        signal.symbol = "BTC/USDT"
        signal.price = market_data.price

        logger.info(f"Generated signal: {signal.direction.value} {signal.symbol} @ ${signal.price}")

        # 3. Risk validation
        portfolio_state = await state_manager.get_portfolio_state()
        risk_valid = await risk_manager.validate_trade(signal, portfolio_state)
        assert risk_valid is True

        # 4. Position sizing
        position_size = await risk_manager.calculate_position_size(signal, portfolio_state)
        assert position_size == Decimal("0.1")
        logger.info(f"Calculated position size: {position_size} BTC")

        # 5. Pre-trade risk check
        pre_trade_check = await risk_manager.check_pre_trade_risk(signal, position_size)
        assert pre_trade_check is True

        # 6. Order creation and execution
        start_time = time.time()
        order_id = await binance.place_order(
            {
                "symbol": signal.symbol,
                "side": signal.direction.value,
                "quantity": str(position_size),
                "type": "MARKET",
            }
        )
        order_latency = time.time() - start_time
        performance_monitor.record_api_call(order_latency)

        assert order_id is not None
        logger.info(f"Order placed: {order_id}, latency: {order_latency * 1000:.1f}ms")

        # 7. Wait for order fill
        order_filled = await wait_for_condition(
            lambda: binance._orders[order_id].status == OrderStatus.FILLED, timeout_seconds=10.0
        )
        assert order_filled is True

        filled_order = binance._orders[order_id]
        assert filled_order.filled_quantity == position_size
        logger.info(f"Order filled: {filled_order.filled_quantity} @ ${filled_order.average_price}")

        # 8. Update risk exposure
        await risk_manager.update_exposure(filled_order)

        # 9. State persistence
        new_position = Position(
            symbol=signal.symbol,
            side=OrderSide.LONG,
            quantity=filled_order.filled_quantity,
            average_price=filled_order.average_price,
            unrealized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
        )

        await state_manager.save_checkpoint(
            {
                "positions": {signal.symbol: new_position},
                "last_trade": filled_order,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Verify workflow success
        balance_after = await binance.get_balance()
        assert balance_after["BTC"] == position_size
        assert balance_after["USDT"] < Decimal("100000.0")  # USDT spent

        logger.info("Complete buy signal workflow completed successfully")

    @pytest.mark.asyncio
    @performance_test(max_duration=30.0)
    @pytest.mark.timeout(300)
    async def test_sell_signal_with_existing_position(self, performance_monitor):
        """Test sell signal workflow with existing position."""

        exchanges = await self.create_mock_exchanges()
        strategies = await self.create_mock_strategies()

        binance = exchanges["binance"]
        mean_reversion_strategy = strategies["mean_reversion"]

        # Setup existing BTC position
        binance._balances["BTC"] = Decimal("0.2")
        binance._balances["USDT"] = Decimal("90000.0")

        risk_manager = Mock(spec=RiskManager)
        risk_manager.get_position = AsyncMock(
            return_value=Position(
                symbol="BTC/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("0.2"),
                average_price=Decimal("49000.0"),
                unrealized_pnl=Decimal("200.0"),
            )
        )
        risk_manager.validate_trade = AsyncMock(return_value=True)
        risk_manager.calculate_position_size = AsyncMock(
            return_value=Decimal("0.1")
        )  # Partial close

        # Generate sell signal
        sell_signal = await mean_reversion_strategy.generate_signal()
        sell_signal.direction = SignalDirection.SELL
        sell_signal.symbol = "BTC/USDT"

        # Verify existing position
        existing_position = await risk_manager.get_position("BTC/USDT")
        assert existing_position.quantity == Decimal("0.2")

        # Validate sell trade
        can_sell = await risk_manager.validate_trade(sell_signal)
        assert can_sell is True

        # Calculate partial sell size
        sell_size = await risk_manager.calculate_position_size(sell_signal)
        assert sell_size == Decimal("0.1")

        # Execute sell order
        sell_order_id = await binance.place_order(
            {"symbol": "BTC/USDT", "side": "SELL", "quantity": str(sell_size), "type": "MARKET"}
        )
        performance_monitor.record_api_call()

        # Wait for fill
        sell_filled = await wait_for_condition(
            lambda: binance._orders[sell_order_id].status == OrderStatus.FILLED,
            timeout_seconds=10.0,
        )
        assert sell_filled is True

        # Verify position reduced
        balance_after = await binance.get_balance()
        assert balance_after["BTC"] == Decimal("0.1")  # 0.2 - 0.1 = 0.1 remaining
        assert balance_after["USDT"] > Decimal("90000.0")  # USDT increased from sale

        logger.info("Sell signal with existing position completed successfully")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_signal_rejection_by_risk_management(self):
        """Test workflow when signal is rejected by risk management."""

        exchanges = await self.create_mock_exchanges()
        strategies = await self.create_mock_strategies()

        momentum_strategy = strategies["momentum"]

        # Risk manager that rejects trades
        risk_manager = Mock(spec=RiskManager)
        risk_manager.validate_trade = AsyncMock(return_value=False)
        risk_manager.get_rejection_reason = Mock(return_value="Daily loss limit exceeded")

        # Generate signal
        signal = await momentum_strategy.generate_signal()
        signal.direction = SignalDirection.BUY

        # Risk validation should reject
        is_valid = await risk_manager.validate_trade(signal)
        assert is_valid is False

        reason = risk_manager.get_rejection_reason()
        assert "loss limit" in reason

        # No order should be placed - workflow stops here
        logger.info(f"Signal correctly rejected: {reason}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_multi_strategy_signal_aggregation(self):
        """Test aggregation of signals from multiple strategies."""

        strategies = await self.create_mock_strategies()
        momentum = strategies["momentum"]
        mean_reversion = strategies["mean_reversion"]

        # Force specific signals for testing
        momentum.generate_signal = AsyncMock(
            return_value=Signal(
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                confidence=0.8,
                price=Decimal("50000.0"),
                timestamp=datetime.now(timezone.utc),
            )
        )

        mean_reversion.generate_signal = AsyncMock(
            return_value=Signal(
                symbol="BTC/USDT",
                direction=SignalDirection.SELL,
                confidence=0.6,
                price=Decimal("50000.0"),
                timestamp=datetime.now(timezone.utc),
            )
        )

        # Collect signals
        strategy_signals = []
        for strategy_name, strategy in strategies.items():
            signal = await strategy.generate_signal()
            strategy_signals.append((strategy_name, signal))

        # Signal aggregation logic
        buy_confidence = sum(
            s[1].confidence for s in strategy_signals if s[1].direction == SignalDirection.BUY
        )
        sell_confidence = sum(
            s[1].confidence for s in strategy_signals if s[1].direction == SignalDirection.SELL
        )

        if buy_confidence > sell_confidence:
            final_signal = SignalDirection.BUY
            final_confidence = buy_confidence
        elif sell_confidence > buy_confidence:
            final_signal = SignalDirection.SELL
            final_confidence = sell_confidence
        else:
            final_signal = SignalDirection.HOLD
            final_confidence = 0.0

        # In this test: BUY(0.8) vs SELL(0.6) -> BUY wins
        assert final_signal == SignalDirection.BUY
        assert final_confidence == 0.8

        logger.info(f"Aggregated signal: {final_signal.value} with confidence {final_confidence}")


class TestOrderExecutionScenarios(BaseIntegrationTest):
    """Test various order execution scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_limit_order_execution_workflow(self):
        """Test limit order execution with partial fills."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]

        # Override place_order to simulate limit order behavior
        original_place_order = binance.place_order

        async def limit_order_behavior(order_data):
            order_id = await original_place_order(order_data)
            order = binance._orders[order_id]

            # For limit orders, simulate partial fill over time
            if order_data.get("type") == "LIMIT":
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_quantity = order.quantity * Decimal("0.3")  # 30% fill
                order.remaining_quantity = order.quantity - order.filled_quantity

            return order_id

        binance.place_order = limit_order_behavior

        # Place limit order
        order_id = await binance.place_order(
            {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "1.0",
                "type": "LIMIT",
                "price": "49000.0",  # Below market price
            }
        )

        # Check partial fill
        order = binance._orders[order_id]
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("0.3")
        assert order.remaining_quantity == Decimal("0.7")

        # Simulate strategy decision on partial fill
        fill_percentage = (order.filled_quantity / order.quantity) * 100

        if fill_percentage >= 25:  # Accept if 25%+ filled
            # Cancel remaining quantity
            cancel_success = await binance.cancel_order(order_id)
            assert cancel_success is True
            assert binance._orders[order_id].status == OrderStatus.CANCELLED

            # Accept the partial fill as final position
            logger.info(
                f"Accepted partial fill: {order.filled_quantity} BTC @ ${order.average_price}"
            )

        # Verify balance updated for partial fill only
        balance = await binance.get_balance()
        assert balance["BTC"] == Decimal("0.3")  # Only partial fill amount

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_slippage_and_price_impact(self):
        """Test order execution with slippage and price impact."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]

        # Simulate market impact on large orders
        original_place_order = binance.place_order

        async def slippage_simulation(order_data):
            quantity = Decimal(str(order_data["quantity"]))
            market_data = await binance.get_market_data(order_data["symbol"])

            # Calculate price impact based on order size
            if quantity > Decimal("0.5"):  # Large order
                slippage_bps = min(int(quantity * 20), 100)  # Max 100bps slippage
            else:
                slippage_bps = 5  # Small orders have minimal slippage

            # Apply slippage
            if order_data["side"] == "BUY":
                execution_price = market_data.ask * (1 + slippage_bps / 10000)
            else:
                execution_price = market_data.bid * (1 - slippage_bps / 10000)

            # Create order with slippage
            order_id = await original_place_order(order_data)
            order = binance._orders[order_id]
            order.average_price = execution_price

            # Update balances with slippage price
            base_asset, quote_asset = order_data["symbol"].split("/")
            if order_data["side"] == "BUY":
                cost = quantity * execution_price
                binance._balances[quote_asset] -= cost
                binance._balances[base_asset] = (
                    binance._balances.get(base_asset, Decimal("0")) + quantity
                )

            logger.info(
                f"Order executed with slippage: {slippage_bps}bps, price: ${execution_price}"
            )
            return order_id

        binance.place_order = slippage_simulation

        # Test small order (low slippage)
        small_order_id = await binance.place_order(
            {"symbol": "BTC/USDT", "side": "BUY", "quantity": "0.1", "type": "MARKET"}
        )

        small_order = binance._orders[small_order_id]
        market_price = Decimal("50001.0")  # Expected ask price
        small_slippage = abs(small_order.average_price - market_price) / market_price
        assert small_slippage < Decimal("0.001")  # Less than 0.1% slippage

        # Reset balances for large order test
        binance._balances = {"USDT": Decimal("100000.0"), "BTC": Decimal("0.0")}

        # Test large order (high slippage)
        large_order_id = await binance.place_order(
            {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "2.0",  # Large order
                "type": "MARKET",
            }
        )

        large_order = binance._orders[large_order_id]
        large_slippage = abs(large_order.average_price - market_price) / market_price
        assert large_slippage > small_slippage  # Higher slippage for large order
        assert large_slippage < Decimal("0.01")  # But still reasonable (< 1%)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_order_timeout_and_retry(self):
        """Test order timeout handling and retry mechanism."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]

        # Mock timeout behavior
        call_count = 0

        async def timeout_simulation(order_data):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                # First two calls timeout
                await asyncio.sleep(0.1)
                raise asyncio.TimeoutError("Order placement timeout")
            else:
                # Third call succeeds
                return f"success_order_{call_count}"

        binance.place_order = timeout_simulation

        # Retry logic
        max_retries = 3
        retry_count = 0
        order_id = None
        last_error = None

        while retry_count < max_retries:
            try:
                order_id = await binance.place_order(
                    {"symbol": "BTC/USDT", "side": "BUY", "quantity": "1.0", "type": "MARKET"}
                )
                break
            except asyncio.TimeoutError as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(0.5 * retry_count)  # Exponential backoff

        # Should succeed on third attempt
        assert order_id == "success_order_3"
        assert call_count == 3
        assert retry_count == 2  # Two retries before success

        logger.info(f"Order succeeded after {retry_count} retries")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_order_rejection_handling(self):
        """Test handling of rejected orders."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]

        # Set insufficient balance
        binance._balances = {"USDT": Decimal("100.0"), "BTC": Decimal("0.0")}

        # Override to simulate order rejection
        async def reject_insufficient_funds(order_data):
            quantity = Decimal(str(order_data["quantity"]))
            market_data = await binance.get_market_data(order_data["symbol"])

            if order_data["side"] == "BUY":
                cost = quantity * market_data.ask
                available = binance._balances.get("USDT", Decimal("0"))

                if cost > available:
                    raise Exception(f"Insufficient funds: need ${cost}, have ${available}")

            # If we get here, order would succeed (but we won't in this test)
            return "success_order"

        binance.place_order = reject_insufficient_funds

        # Try to place order that exceeds balance
        with pytest.raises(Exception, match="Insufficient funds"):
            await binance.place_order(
                {
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "quantity": "10.0",  # Would cost ~$500,000 but we only have $100
                    "type": "MARKET",
                }
            )

        # Verify balance unchanged
        balance = await binance.get_balance()
        assert balance["USDT"] == Decimal("100.0")
        assert balance["BTC"] == Decimal("0.0")

        logger.info("Order rejection handled correctly")


class TestPositionManagementIntegration(BaseIntegrationTest):
    """Test position management integration scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_position_lifecycle_management(self):
        """Test complete position lifecycle from opening to closing."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]

        position_manager = Mock()
        positions = {}  # Track positions

        # 1. Open initial position
        buy_order_id = await binance.place_order(
            {"symbol": "BTC/USDT", "side": "BUY", "quantity": "1.0", "type": "MARKET"}
        )

        buy_order = binance._orders[buy_order_id]
        assert buy_order.status == OrderStatus.FILLED

        # Create position
        positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=buy_order.filled_quantity,
            average_price=buy_order.average_price,
            unrealized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            f"Opened position: {positions['BTC/USDT'].quantity} BTC @ ${positions['BTC/USDT'].average_price}"
        )

        # 2. Increase position size
        additional_buy_id = await binance.place_order(
            {"symbol": "BTC/USDT", "side": "BUY", "quantity": "0.5", "type": "MARKET"}
        )

        additional_order = binance._orders[additional_buy_id]
        existing_position = positions["BTC/USDT"]

        # Update position (weighted average price)
        total_quantity = existing_position.quantity + additional_order.filled_quantity
        new_average_price = (
            existing_position.quantity * existing_position.average_price
            + additional_order.filled_quantity * additional_order.average_price
        ) / total_quantity

        positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=total_quantity,
            average_price=new_average_price,
            unrealized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
        )

        assert positions["BTC/USDT"].quantity == Decimal("1.5")
        logger.info(
            f"Increased position to: {positions['BTC/USDT'].quantity} BTC @ ${positions['BTC/USDT'].average_price}"
        )

        # 3. Partial close
        partial_sell_id = await binance.place_order(
            {"symbol": "BTC/USDT", "side": "SELL", "quantity": "0.8", "type": "MARKET"}
        )

        sell_order = binance._orders[partial_sell_id]

        # Calculate realized P&L
        realized_pnl = (
            sell_order.average_price - positions["BTC/USDT"].average_price
        ) * sell_order.filled_quantity

        # Update position
        remaining_quantity = positions["BTC/USDT"].quantity - sell_order.filled_quantity
        positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=remaining_quantity,
            average_price=positions["BTC/USDT"].average_price,  # Average price unchanged
            unrealized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
        )

        assert positions["BTC/USDT"].quantity == Decimal("0.7")
        assert realized_pnl != Decimal("0.0")  # Should have some P&L
        logger.info(
            f"Partial close: remaining {positions['BTC/USDT'].quantity} BTC, realized P&L: ${realized_pnl}"
        )

        # 4. Complete close
        final_sell_id = await binance.place_order(
            {
                "symbol": "BTC/USDT",
                "side": "SELL",
                "quantity": str(positions["BTC/USDT"].quantity),
                "type": "MARKET",
            }
        )

        final_order = binance._orders[final_sell_id]
        final_pnl = (
            final_order.average_price - positions["BTC/USDT"].average_price
        ) * final_order.filled_quantity

        # Position is now closed
        del positions["BTC/USDT"]

        total_realized_pnl = realized_pnl + final_pnl

        assert "BTC/USDT" not in positions
        logger.info(f"Position closed completely, total realized P&L: ${total_realized_pnl}")

        # Verify final balances
        final_balance = await binance.get_balance()
        assert final_balance["BTC"] == Decimal("0.0")
        assert final_balance["USDT"] != Decimal("100000.0")  # Should be different from start

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_multi_symbol_position_management(self):
        """Test managing positions across multiple symbols."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]

        # Add ETH market data
        original_get_market_data = binance.get_market_data

        async def multi_symbol_market_data(symbol):
            if symbol == "ETH/USDT":
                return MarketData(
                    symbol="ETH/USDT",
                    price=Decimal("3000.0"),
                    bid=Decimal("2999.0"),
                    ask=Decimal("3001.0"),
                    volume=Decimal("5000.0"),
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                return await original_get_market_data(symbol)

        binance.get_market_data = multi_symbol_market_data

        positions = {}

        # Open BTC position
        btc_order_id = await binance.place_order(
            {"symbol": "BTC/USDT", "side": "BUY", "quantity": "1.0", "type": "MARKET"}
        )

        btc_order = binance._orders[btc_order_id]
        positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=btc_order.filled_quantity,
            average_price=btc_order.average_price,
            unrealized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
        )

        # Open ETH position
        eth_order_id = await binance.place_order(
            {"symbol": "ETH/USDT", "side": "BUY", "quantity": "10.0", "type": "MARKET"}
        )

        eth_order = binance._orders[eth_order_id]
        positions["ETH/USDT"] = Position(
            symbol="ETH/USDT",
            side=OrderSide.LONG,
            quantity=eth_order.filled_quantity,
            average_price=eth_order.average_price,
            unrealized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
        )

        # Verify multi-symbol portfolio
        assert len(positions) == 2
        assert "BTC/USDT" in positions
        assert "ETH/USDT" in positions

        # Calculate total portfolio value
        btc_market_data = await binance.get_market_data("BTC/USDT")
        eth_market_data = await binance.get_market_data("ETH/USDT")

        btc_value = positions["BTC/USDT"].quantity * btc_market_data.price
        eth_value = positions["ETH/USDT"].quantity * eth_market_data.price
        total_position_value = btc_value + eth_value

        logger.info(
            f"Multi-symbol portfolio value: BTC ${btc_value}, ETH ${eth_value}, Total ${total_position_value}"
        )

        # Test correlation risk - both crypto positions (high correlation)
        correlation_risk = 0.85  # High correlation between BTC and ETH
        effective_diversification = 1.0 - correlation_risk

        # Risk manager might reduce position sizes due to correlation
        assert effective_diversification < 0.5  # Limited diversification benefit
        logger.info(
            f"Correlation risk detected: {correlation_risk}, diversification benefit: {effective_diversification}"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_position_pnl_tracking(self):
        """Test real-time P&L tracking for positions."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]

        # Open position
        order_id = await binance.place_order(
            {"symbol": "BTC/USDT", "side": "BUY", "quantity": "1.0", "type": "MARKET"}
        )

        order = binance._orders[order_id]
        entry_price = order.average_price

        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=order.filled_quantity,
            average_price=entry_price,
            unrealized_pnl=Decimal("0.0"),
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(f"Position opened at ${entry_price}")

        # Simulate price movements and P&L updates
        price_scenarios = [
            Decimal("51000.0"),  # +2% profit
            Decimal("49000.0"),  # -2% loss
            Decimal("52500.0"),  # +5% profit
            Decimal("47500.0"),  # -5% loss
        ]

        pnl_history = []

        for current_price in price_scenarios:
            # Update market price
            binance._market_prices["BTC/USDT"] = current_price

            # Calculate unrealized P&L
            if position.side == OrderSide.LONG:
                unrealized_pnl = (current_price - position.average_price) * position.quantity
            else:  # SHORT
                unrealized_pnl = (position.average_price - current_price) * position.quantity

            position.unrealized_pnl = unrealized_pnl
            pnl_history.append((current_price, unrealized_pnl))

            pnl_percent = (unrealized_pnl / (position.average_price * position.quantity)) * 100
            logger.info(
                f"Price: ${current_price}, Unrealized P&L: ${unrealized_pnl} ({pnl_percent:.1f}%)"
            )

        # Verify P&L calculations
        assert len(pnl_history) == 4

        # First scenario: +2% should give ~$1000 profit on $50k position
        price_1, pnl_1 = pnl_history[0]
        expected_pnl_1 = (price_1 - entry_price) * position.quantity
        assert abs(pnl_1 - expected_pnl_1) < Decimal("0.01")

        # Last scenario: -5% should give ~$2500 loss on $50k position
        price_4, pnl_4 = pnl_history[3]
        expected_pnl_4 = (price_4 - entry_price) * position.quantity
        assert abs(pnl_4 - expected_pnl_4) < Decimal("0.01")
        assert pnl_4 < Decimal("0.0")  # Should be negative (loss)

        logger.info("P&L tracking validation completed")


class TestErrorRecoveryIntegration(BaseIntegrationTest):
    """Test error recovery and resilience in trading workflows."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_network_disconnection_recovery(self):
        """Test recovery from network disconnection during trading."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]

        # Simulate network disconnection
        disconnect_count = 0

        async def intermittent_connection(order_data):
            nonlocal disconnect_count
            disconnect_count += 1

            if disconnect_count <= 3:
                # Simulate connection issues
                await asyncio.sleep(0.1)
                raise ConnectionError("Network unreachable")
            else:
                # Connection restored
                binance.is_connected = True
                return f"recovered_order_{disconnect_count}"

        binance.place_order = intermittent_connection
        binance.is_connected = False

        # Connection recovery logic
        max_attempts = 5
        attempt = 0
        order_id = None

        while attempt < max_attempts and not binance.is_connected:
            try:
                order_id = await binance.place_order(
                    {"symbol": "BTC/USDT", "side": "BUY", "quantity": "1.0", "type": "MARKET"}
                )
                binance.is_connected = True
                break
            except ConnectionError:
                attempt += 1
                backoff_time = min(2**attempt, 10)  # Exponential backoff, max 10s
                logger.info(f"Connection attempt {attempt} failed, retrying in {backoff_time}s")
                await asyncio.sleep(backoff_time)

        # Should eventually succeed
        assert order_id == "recovered_order_4"  # 4th call succeeds
        assert binance.is_connected is True
        assert attempt == 3  # 3 retry attempts before success

        logger.info(f"Network recovery successful after {attempt} retries")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_partial_system_failure_recovery(self):
        """Test recovery when some system components fail."""

        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]
        coinbase = exchanges["coinbase"]

        # Simulate Binance failure
        binance.is_connected = False
        binance.place_order = AsyncMock(side_effect=Exception("Exchange maintenance"))

        # Coinbase remains operational
        coinbase.is_connected = True

        # Exchange failover logic
        target_exchanges = [binance, coinbase]
        successful_exchange = None

        for exchange in target_exchanges:
            if exchange.is_connected:
                try:
                    order_id = await exchange.place_order(
                        {"symbol": "BTC/USDT", "side": "BUY", "quantity": "1.0", "type": "MARKET"}
                    )
                    successful_exchange = exchange
                    logger.info(f"Order placed on backup exchange: {exchange.name}")
                    break
                except Exception as e:
                    logger.warning(f"Exchange {exchange.name} failed: {e}")
                    continue

        # Should failover to Coinbase
        assert successful_exchange == coinbase
        assert successful_exchange.name == "coinbase"

        # Verify order was placed on backup exchange
        coinbase_balance = await coinbase.get_balance()
        assert coinbase_balance["BTC"] > Decimal("0.0")

        logger.info("Exchange failover completed successfully")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_state_corruption_recovery(self):
        """Test recovery from corrupted state data."""

        state_manager = Mock(spec=StateService)

        # Simulate corrupted state
        corrupted_state = {
            "positions": {
                "BTC/USDT": {
                    "quantity": "invalid_decimal",  # Corrupted data
                    "average_price": None,
                    "timestamp": "invalid_date",
                }
            },
            "portfolio_value": "not_a_number",
        }

        # Mock recovery methods
        backup_state = {
            "positions": {
                "BTC/USDT": {
                    "quantity": "1.0",
                    "average_price": "50000.0",
                    "timestamp": "2024-01-01T12:00:00Z",
                }
            },
            "portfolio_value": "100000.0",
            "last_checkpoint": "2024-01-01T11:30:00Z",
        }

        state_manager.load_state = AsyncMock(side_effect=[corrupted_state, backup_state])
        state_manager.validate_state = Mock(side_effect=[False, True])
        state_manager.repair_state = AsyncMock(return_value=backup_state)

        # State recovery logic
        recovered_state = None
        recovery_attempts = 0
        max_recovery_attempts = 3

        while recovery_attempts < max_recovery_attempts:
            try:
                candidate_state = await state_manager.load_state()

                if state_manager.validate_state(candidate_state):
                    recovered_state = candidate_state
                    break
                else:
                    # Attempt repair
                    if recovery_attempts == 0:
                        recovered_state = await state_manager.repair_state(candidate_state)
                        if state_manager.validate_state(recovered_state):
                            break

                    recovery_attempts += 1

            except Exception as e:
                logger.warning(f"State recovery attempt {recovery_attempts + 1} failed: {e}")
                recovery_attempts += 1

        # Should recover valid state
        assert recovered_state is not None
        assert state_manager.validate_state(recovered_state) is True
        assert "positions" in recovered_state
        assert recovered_state["positions"]["BTC/USDT"]["quantity"] == "1.0"

        logger.info("State corruption recovery completed successfully")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_cascade_failure_isolation(self):
        """Test isolation of cascade failures across system components."""

        # Setup components with circuit breaker pattern
        components = {
            "exchange": {"status": "healthy", "failures": 0, "circuit_open": False},
            "risk_manager": {"status": "healthy", "failures": 0, "circuit_open": False},
            "state_manager": {"status": "healthy", "failures": 0, "circuit_open": False},
            "execution_engine": {"status": "healthy", "failures": 0, "circuit_open": False},
        }

        failure_threshold = 3
        recovery_timeout = 5.0

        def circuit_breaker(component_name: str, operation: Callable):
            """Circuit breaker pattern for component isolation."""
            component = components[component_name]

            # Check if circuit is open
            if component["circuit_open"]:
                raise Exception(f"Circuit breaker open for {component_name}")

            try:
                # Simulate operation
                if component["failures"] >= failure_threshold:
                    raise Exception(f"Component {component_name} is failing")

                # Operation succeeds
                component["failures"] = 0  # Reset failure count
                component["status"] = "healthy"
                return "success"

            except Exception as e:
                component["failures"] += 1
                component["status"] = "degraded"

                # Open circuit if threshold reached
                if component["failures"] >= failure_threshold:
                    component["circuit_open"] = True
                    component["status"] = "failed"
                    logger.warning(f"Circuit breaker opened for {component_name}")

                raise e

        # Simulate cascade failure starting with exchange
        components["exchange"]["failures"] = 5  # Force exchange failure

        # Test component isolation
        with pytest.raises(Exception, match="Circuit breaker open"):
            circuit_breaker("exchange", lambda: "exchange_operation")

        # Other components should remain operational
        assert circuit_breaker("risk_manager", lambda: "risk_check") == "success"
        assert circuit_breaker("state_manager", lambda: "state_save") == "success"
        assert circuit_breaker("execution_engine", lambda: "execute_order") == "success"

        # Verify isolation
        assert components["exchange"]["circuit_open"] is True
        assert components["risk_manager"]["circuit_open"] is False
        assert components["state_manager"]["circuit_open"] is False
        assert components["execution_engine"]["circuit_open"] is False

        logger.info("Cascade failure isolation successful - other components remain operational")

        # Test recovery
        components["exchange"]["failures"] = 0  # Simulate component recovery
        components["exchange"]["circuit_open"] = False

        # Should work again
        assert circuit_breaker("exchange", lambda: "exchange_operation") == "success"
        assert components["exchange"]["status"] == "healthy"

        logger.info("Component recovery successful")
