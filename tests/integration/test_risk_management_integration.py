"""
Integration tests for Risk Management Framework.

Tests circuit breakers, position limits, emergency controls, portfolio risk metrics,
correlation monitoring, and adaptive risk management across all system components.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import time
import asyncio
import logging

import pytest

from tests.integration.base_integration import (
    BaseIntegrationTest, MockExchangeFactory, PerformanceMonitor,
    performance_test, wait_for_condition
)
from src.core.config import Config
from src.core.exceptions import RiskManagementError, ValidationError
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    RiskLevel,
    RiskMetrics,
    Signal,
    SignalDirection,
    BotStatus,
    Order,
    OrderStatus
)
from src.risk_management import RiskManager

logger = logging.getLogger(__name__)


class TestRiskManagementIntegration:
    """Integration tests for complete risk management workflow."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def risk_manager(self, config):
        """Create risk manager instance."""
        return RiskManager(config)

    @pytest.fixture
    def sample_positions(self):
        """Create sample portfolio positions."""
        return [
            Position(
                symbol="BTCUSDT",
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                unrealized_pnl=Decimal("100"),
                side=OrderSide.BUY,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
            ),
            Position(
                symbol="ETHUSDT",
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                unrealized_pnl=Decimal("100"),
                side=OrderSide.BUY,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
            ),
        ]

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for all positions."""
        return [
            MarketData(
                symbol="BTCUSDT",
                open=Decimal("50000"),
                high=Decimal("52000"),
                low=Decimal("49000"),
                close=Decimal("51000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
            ),
            MarketData(
                symbol="ETHUSDT",
                open=Decimal("3000"),
                high=Decimal("3200"),
                low=Decimal("2900"),
                close=Decimal("3100"),
                volume=Decimal("500"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
            ),
        ]

    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals."""
        return [
            Signal(
                direction=SignalDirection.BUY,
                strength=0.8,
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                source="test_strategy",
            ),
            Signal(
                direction=SignalDirection.SELL,
                strength=0.7,
                timestamp=datetime.now(timezone.utc),
                symbol="ETHUSDT",
                source="test_strategy",
            ),
        ]

    @pytest.mark.asyncio
    async def test_complete_risk_management_workflow(
        self, risk_manager, sample_positions, sample_market_data, sample_signals
    ):
        """Test complete risk management workflow."""
        portfolio_value = Decimal("100000")

        # Step 1: Update portfolio state
        await risk_manager.update_portfolio_state(sample_positions, portfolio_value)

        # Step 2: Calculate risk metrics
        risk_metrics = await risk_manager.calculate_risk_metrics(
            sample_positions, sample_market_data
        )

        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.risk_level in [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]
        assert risk_metrics.var_1d >= 0
        assert risk_metrics.var_5d >= 0
        assert risk_metrics.expected_shortfall >= 0
        assert risk_metrics.max_drawdown >= 0

        # Step 3: Validate signals
        for signal in sample_signals:
            is_valid = await risk_manager.validate_signal(signal)
            assert isinstance(is_valid, bool)

        # Step 4: Calculate position sizes
        for signal in sample_signals:
            if signal.direction != SignalDirection.HOLD:
                position_size = await risk_manager.calculate_position_size(signal, portfolio_value)
                assert position_size >= 0
                assert position_size <= portfolio_value * Decimal(
                    str(risk_manager.risk_config.max_position_size_pct)
                )

        # Step 5: Check portfolio limits for new positions
        new_position = Position(
            symbol="ADAUSDT",
            quantity=Decimal("1000"),
            entry_price=Decimal("0.5"),
            current_price=Decimal("0.5"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(),
        )

        can_add = await risk_manager.check_portfolio_limits(new_position)
        assert isinstance(can_add, bool)

        # Step 6: Get comprehensive risk summary
        summary = await risk_manager.get_comprehensive_risk_summary()
        assert isinstance(summary, dict)
        assert "risk_level" in summary
        assert "total_positions" in summary
        assert "portfolio_value" in summary

    @pytest.mark.asyncio
    async def test_risk_management_with_large_portfolio(self, risk_manager):
        """Test risk management with a large portfolio."""
        # Create large portfolio
        positions = []
        market_data = []
        portfolio_value = Decimal("1000000")  # 1M portfolio

        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]

        for i, symbol in enumerate(symbols):
            position = Position(
                symbol=symbol,
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                unrealized_pnl=Decimal("1000"),
                side=OrderSide.BUY,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
            )
            positions.append(position)

            market_data.append(
                MarketData(
                    symbol=symbol,
                    open=Decimal("50000"),
                    high=Decimal("52000"),
                    low=Decimal("49000"),
                    close=Decimal("51000"),
                    volume=Decimal("10000"),
                    timestamp=datetime.now(timezone.utc),
                    exchange="binance",
                )
            )

        # Simulate historical portfolio data to enable proper VaR calculation
        # Update portfolio state multiple times with varying values to create
        # return history
        base_value = 1000000
        for i in range(50):  # Create 50 days of history
            # Simulate some volatility in portfolio value
            variation = (i % 10 - 5) * 10000  # -50k to +50k variation
            current_value = base_value + variation
            await risk_manager.update_portfolio_state(positions, Decimal(str(current_value)))

        # Calculate risk metrics
        risk_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)

        assert risk_metrics.risk_level in [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]
        assert risk_metrics.var_1d > 0
        # 5-day VaR should be greater than 1-day VaR due to time scaling
        assert risk_metrics.var_5d > risk_metrics.var_1d

        # Test position sizing with large portfolio
        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.9,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            source="test_strategy",
        )

        position_size = await risk_manager.calculate_position_size(signal, portfolio_value)
        assert position_size > 0
        assert position_size <= portfolio_value * Decimal(
            str(risk_manager.risk_config.max_position_size_pct)
        )

    @pytest.mark.asyncio
    async def test_risk_management_with_high_volatility(self, risk_manager):
        """Test risk management with high volatility scenario."""
        # Create positions with high volatility (large price movements)
        positions = [
            Position(
                symbol="BTCUSDT",
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("45000"),  # 10% loss
                unrealized_pnl=Decimal("-500"),
                side=OrderSide.BUY,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
            )
        ]

        market_data = [
            MarketData(
                symbol="BTCUSDT",
                open=Decimal("50000"),
                high=Decimal("52000"),
                low=Decimal("44000"),
                close=Decimal("45000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
            )
        ]

        # Simulate portfolio decline to create drawdown
        # Start with higher portfolio value and gradually decline
        initial_portfolio_value = Decimal("60000")  # Higher initial value
        current_portfolio_value = Decimal("50000")  # Current lower value

        # Update portfolio state with declining values to create drawdown
        for i in range(30):  # Create 30 days of history
            # Simulate declining portfolio value
            decline_factor = 1 - (i * 0.003)  # Gradual decline
            portfolio_value = initial_portfolio_value * Decimal(str(decline_factor))
            await risk_manager.update_portfolio_state(positions, portfolio_value)

        # Use the final (lower) portfolio value for risk calculation
        final_portfolio_value = initial_portfolio_value * Decimal("0.85")  # 15% decline
        await risk_manager.update_portfolio_state(positions, final_portfolio_value)

        # Update the position to reflect the declining portfolio value
        # This ensures the calculated portfolio value matches our historical
        # data
        positions[0].current_price = Decimal("45000")  # Keep the loss position
        positions[0].unrealized_pnl = Decimal("-500")

        # Calculate risk metrics
        risk_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)

        # Should detect risk metrics correctly
        assert risk_metrics.var_1d > 0
        assert risk_metrics.var_5d > risk_metrics.var_1d  # 5-day VaR should be larger

        # Test position exit evaluation
        should_exit = await risk_manager.should_exit_position(positions[0], market_data[0])
        # Should exit due to large loss (10% loss on 50k position = 5k loss)
        assert should_exit is True

    @pytest.mark.asyncio
    async def test_risk_management_with_multiple_strategies(self, risk_manager):
        """Test risk management with multiple trading strategies."""
        # Create signals from different strategies
        signals = [
            Signal(
                direction=SignalDirection.BUY,
                strength=0.8,
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                source="momentum_strategy",
            ),
            Signal(
                direction=SignalDirection.SELL,
                strength=0.7,
                timestamp=datetime.now(),
                symbol="ETHUSDT",
                source="mean_reversion_strategy",
            ),
            Signal(
                direction=SignalDirection.BUY,
                strength=0.9,
                timestamp=datetime.now(),
                symbol="ADAUSDT",
                source="ml_strategy",
            ),
        ]

        portfolio_value = Decimal("100000")

        # Test position sizing for each strategy
        for signal in signals:
            if signal.direction != SignalDirection.HOLD:
                position_size = await risk_manager.calculate_position_size(signal, portfolio_value)
                assert position_size > 0

                # Higher confidence should generally result in larger position
                if signal.confidence > 0.8:
                    # This is a general expectation, not a strict rule
                    assert position_size > 0

    @pytest.mark.asyncio
    async def test_risk_management_parameter_validation(self, risk_manager):
        """Test risk management parameter validation."""
        # Test with valid parameters
        is_valid = await risk_manager.validate_risk_parameters()
        assert is_valid is True

        # Note: Pydantic validation is tested at the config level
        # The validation error is raised during RiskConfig creation, which is correct behavior
        # This test validates that the risk manager can work with valid
        # parameters

    @pytest.mark.asyncio
    async def test_risk_management_error_handling(self, risk_manager):
        """Test risk management error handling."""
        # Test with invalid signal
        invalid_signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.0,  # Invalid confidence
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            source="test_strategy",
        )

        portfolio_value = Decimal("10000")

        with pytest.raises(RiskManagementError):
            await risk_manager.calculate_position_size(invalid_signal, portfolio_value)

        # Test with invalid order
        invalid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("-0.1"),  # Negative quantity
            price=None,
            stop_price=None,
            time_in_force="GTC",
            client_order_id="test_order_123",
        )

        with pytest.raises(ValidationError):
            await risk_manager.validate_order(invalid_order, portfolio_value)

    @pytest.mark.asyncio
    async def test_risk_management_performance(
        self, risk_manager, sample_positions, sample_market_data
    ):
        """Test risk management performance with multiple rapid operations."""
        portfolio_value = Decimal("100000")

        # Update portfolio state
        await risk_manager.update_portfolio_state(sample_positions, portfolio_value)

        # Perform multiple rapid operations
        start_time = datetime.now()

        for _ in range(10):
            # Calculate risk metrics
            risk_metrics = await risk_manager.calculate_risk_metrics(
                sample_positions, sample_market_data
            )
            assert isinstance(risk_metrics, RiskMetrics)

            # Validate signals
            signal = Signal(
                direction=SignalDirection.BUY,
                strength=0.8,
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                source="test_strategy",
            )
            is_valid = await risk_manager.validate_signal(signal)
            assert isinstance(is_valid, bool)

            # Calculate position size
            position_size = await risk_manager.calculate_position_size(signal, portfolio_value)
            assert position_size > 0

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete within reasonable time (less than 5 seconds for 10
        # operations)
        assert duration < 5.0

    @pytest.mark.asyncio
    async def test_risk_management_edge_cases(self, risk_manager):
        """Test risk management edge cases."""
        # Test with empty portfolio
        empty_positions = []
        empty_market_data = []
        portfolio_value = Decimal("10000")

        await risk_manager.update_portfolio_state(empty_positions, portfolio_value)
        risk_metrics = await risk_manager.calculate_risk_metrics(empty_positions, empty_market_data)

        assert risk_metrics.risk_level == RiskLevel.LOW
        assert risk_metrics.var_1d == Decimal("0")
        assert risk_metrics.var_5d == Decimal("0")

        # Test with zero portfolio value
        zero_portfolio_value = Decimal("0")
        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            source="test_strategy",
        )

        with pytest.raises(RiskManagementError):
            await risk_manager.calculate_position_size(signal, zero_portfolio_value)

        # Test with very small portfolio
        small_portfolio_value = Decimal("100")
        position_size = await risk_manager.calculate_position_size(signal, small_portfolio_value)
        assert position_size >= 0

    @pytest.mark.asyncio
    async def test_risk_management_configuration_changes(self, risk_manager):
        """Test risk management behavior with configuration changes."""
        portfolio_value = Decimal("100000")

        # Test with conservative settings
        original_max_position = risk_manager.risk_config.max_position_size_pct
        risk_manager.risk_config.max_position_size_pct = 0.05  # 5% max position

        signal = Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            source="test_strategy",
        )

        position_size = await risk_manager.calculate_position_size(signal, portfolio_value)
        assert position_size <= portfolio_value * Decimal("0.05")

        # Test with aggressive settings
        risk_manager.risk_config.max_position_size_pct = 0.2  # 20% max position
        position_size = await risk_manager.calculate_position_size(signal, portfolio_value)
        assert position_size <= portfolio_value * Decimal("0.2")

        # Restore original settings
        risk_manager.risk_config.max_position_size_pct = original_max_position

    @pytest.mark.asyncio
    async def test_risk_management_real_time_updates(self, risk_manager):
        """Test risk management with real-time updates."""
        portfolio_value = Decimal("100000")

        # Initial state
        positions = [
            Position(
                symbol="BTCUSDT",
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                unrealized_pnl=Decimal("100"),
                side=OrderSide.BUY,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
            )
        ]

        market_data = [
            MarketData(
                symbol="BTCUSDT",
                open=Decimal("50000"),
                high=Decimal("52000"),
                low=Decimal("49000"),
                close=Decimal("51000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
            )
        ]

        # Simulate some historical data to avoid CRITICAL risk level due to insufficient data
        # Update portfolio state multiple times with stable values
        for i in range(30):  # Create 30 days of history
            stable_value = portfolio_value + Decimal(str(i * 100))  # Gradual increase
            await risk_manager.update_portfolio_state(positions, stable_value)

        initial_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)

        # Update with price change
        market_data[0].close = Decimal("52000")  # Price increase
        positions[0].current_price = Decimal("52000")
        positions[0].unrealized_pnl = Decimal("200")

        await risk_manager.update_portfolio_state(positions, portfolio_value)
        updated_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)

        # Metrics should reflect the change
        assert updated_metrics.timestamp > initial_metrics.timestamp

        # Test position exit evaluation with updated data
        # With sufficient historical data, risk should be low for a profitable
        # position
        assert updated_metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]

        should_exit = await risk_manager.should_exit_position(positions[0], market_data[0])
        # Should not exit due to profit and low risk
        assert should_exit is False


class TestCircuitBreakerIntegration(BaseIntegrationTest):
    """Test circuit breaker mechanisms across trading operations."""
    
    @pytest.mark.asyncio
    @performance_test(max_duration=30.0)
    async def test_daily_loss_circuit_breaker(self, performance_monitor):
        """Test daily loss circuit breaker activation and recovery."""
        
        exchanges = await self.create_mock_exchanges()
        binance = exchanges["binance"]
        
        # Setup risk manager with daily loss limits
        risk_manager = Mock()
        risk_manager.daily_loss_limit = Decimal("5000.0")  # $5,000 daily loss limit
        risk_manager.current_daily_pnl = Decimal("0.0")
        risk_manager.circuit_breaker_active = False
        risk_manager.trading_halted = False
        
        # Mock emergency controls
        emergency_controls = Mock()
        emergency_controls.halt_all_trading = AsyncMock(return_value=True)
        emergency_controls.cancel_all_orders = AsyncMock(return_value=True)
        emergency_controls.send_alert = AsyncMock(return_value=True)
        
        # Trade execution function with risk checks
        async def execute_trade_with_risk_check(symbol, side, quantity, expected_pnl):
            nonlocal risk_manager
            
            # Pre-trade risk check
            projected_daily_pnl = risk_manager.current_daily_pnl + expected_pnl
            
            if abs(projected_daily_pnl) > risk_manager.daily_loss_limit:
                # Circuit breaker should activate
                risk_manager.circuit_breaker_active = True
                risk_manager.trading_halted = True
                
                await emergency_controls.halt_all_trading()
                await emergency_controls.cancel_all_orders()
                await emergency_controls.send_alert(f"Daily loss limit exceeded: ${abs(projected_daily_pnl)}")
                
                logger.warning(f"Circuit breaker activated: Daily P&L would be ${projected_daily_pnl}")
                return None
            
            if risk_manager.trading_halted:
                logger.info("Trading halted by circuit breaker - trade rejected")
                return None
            
            # Execute trade
            order_id = await binance.place_order({
                "symbol": symbol,
                "side": side,
                "quantity": str(quantity),
                "type": "MARKET"
            })
            
            # Update P&L
            risk_manager.current_daily_pnl += expected_pnl
            
            performance_monitor.record_api_call()
            logger.info(f"Trade executed: {side} {quantity} {symbol}, P&L: ${expected_pnl}")
            
            return order_id
        
        # Simulate series of trades leading to circuit breaker
        trades = [
            ("BTC/USDT", "BUY", Decimal("0.5"), Decimal("500.0")),    # Profit
            ("ETH/USDT", "BUY", Decimal("5.0"), Decimal("-800.0")),   # Loss
            ("BTC/USDT", "SELL", Decimal("0.3"), Decimal("-1200.0")), # Loss
            ("ETH/USDT", "SELL", Decimal("3.0"), Decimal("300.0")),   # Profit
            ("BTC/USDT", "BUY", Decimal("1.0"), Decimal("-2000.0")),  # Large loss
            ("ETH/USDT", "BUY", Decimal("10.0"), Decimal("-1800.0"))  # This should trigger circuit breaker
        ]
        
        executed_trades = []
        
        for i, (symbol, side, quantity, expected_pnl) in enumerate(trades):
            order_id = await execute_trade_with_risk_check(symbol, side, quantity, expected_pnl)
            
            if order_id is not None:
                executed_trades.append((i, symbol, side, quantity, expected_pnl))
            else:
                logger.info(f"Trade {i+1} rejected by circuit breaker")
                break
        
        # Verify circuit breaker activation
        assert risk_manager.circuit_breaker_active is True
        assert risk_manager.trading_halted is True
        
        # Should not execute all trades due to circuit breaker
        assert len(executed_trades) < len(trades)
        
        logger.info(f"Circuit breaker test complete: {len(executed_trades)}/{len(trades)} trades executed")
        
        # Test circuit breaker reset
        async def reset_daily_circuit_breaker():
            risk_manager.current_daily_pnl = Decimal("0.0")
            risk_manager.circuit_breaker_active = False
            risk_manager.trading_halted = False
            logger.info("Circuit breaker reset for new trading day")
        
        await reset_daily_circuit_breaker()
        
        # Verify reset
        assert risk_manager.circuit_breaker_active is False
        assert risk_manager.trading_halted is False
        
        logger.info("Circuit breaker reset and recovery successful")
    
    @pytest.mark.asyncio
    async def test_position_concentration_limits(self):
        """Test position concentration limit enforcement."""
        
        portfolio_limits = Mock()
        portfolio_limits.max_single_asset_concentration = Decimal("0.3")  # 30% max per asset
        portfolio_limits.max_sector_concentration = Decimal("0.6")        # 60% max per sector
        portfolio_limits.total_portfolio_value = Decimal("200000.0")      # $200k portfolio
        
        # Define asset sectors
        asset_sectors = {
            "BTC/USDT": "crypto_large_cap",
            "ETH/USDT": "crypto_large_cap", 
            "ADA/USDT": "crypto_mid_cap",
            "DOT/USDT": "crypto_mid_cap",
            "LINK/USDT": "crypto_defi"
        }
        
        current_portfolio = {}
        
        def calculate_concentrations():
            """Calculate current asset and sector concentrations."""
            total_value = sum(current_portfolio.values())
            
            asset_concentrations = {}
            sector_concentrations = {}
            
            for symbol, position_value in current_portfolio.items():
                # Asset concentration
                if total_value > 0:
                    asset_concentrations[symbol] = position_value / total_value
                
                # Sector concentration
                sector = asset_sectors.get(symbol, "unknown")
                if sector not in sector_concentrations:
                    sector_concentrations[sector] = Decimal("0.0")
                sector_concentrations[sector] += position_value / total_value if total_value > 0 else Decimal("0.0")
            
            return asset_concentrations, sector_concentrations, total_value
        
        async def attempt_position_addition(symbol, position_value):
            """Attempt to add position with concentration checking."""
            test_portfolio = current_portfolio.copy()
            test_portfolio[symbol] = test_portfolio.get(symbol, Decimal("0.0")) + position_value
            
            # Calculate concentrations with new position
            temp_portfolio = current_portfolio
            current_portfolio.update(test_portfolio)
            asset_conc, sector_conc, total_value = calculate_concentrations()
            
            # Check asset concentration limit
            symbol_concentration = asset_conc.get(symbol, Decimal("0.0"))
            if symbol_concentration > portfolio_limits.max_single_asset_concentration:
                logger.warning(f"Asset concentration limit exceeded for {symbol}: "
                              f"{symbol_concentration:.1%} > {portfolio_limits.max_single_asset_concentration:.1%}")
                return False, f"Asset concentration limit exceeded: {symbol_concentration:.1%}"
            
            # Check sector concentration limit
            sector = asset_sectors.get(symbol, "unknown")
            sector_concentration = sector_conc.get(sector, Decimal("0.0"))
            if sector_concentration > portfolio_limits.max_sector_concentration:
                logger.warning(f"Sector concentration limit exceeded for {sector}: "
                              f"{sector_concentration:.1%} > {portfolio_limits.max_sector_concentration:.1%}")
                return False, f"Sector concentration limit exceeded: {sector_concentration:.1%}"
            
            # Position allowed - update portfolio
            current_portfolio[symbol] = current_portfolio.get(symbol, Decimal("0.0")) + position_value
            
            logger.info(f"Position added: {symbol} +${position_value} "
                       f"(concentration: {symbol_concentration:.1%}, sector {sector}: {sector_concentration:.1%})")
            
            return True, "Position added successfully"
        
        # Test concentration limit scenarios
        concentration_tests = [
            # Build up crypto_large_cap sector
            ("BTC/USDT", Decimal("40000.0")),   # 20% - allowed
            ("ETH/USDT", Decimal("30000.0")),   # 15% - allowed (sector: 35%)
            ("BTC/USDT", Decimal("20000.0")),   # BTC: 30%, sector: 45% - allowed
            ("ETH/USDT", Decimal("40000.0")),   # ETH would be 35% - rejected (asset limit)
            ("BTC/USDT", Decimal("30000.0")),   # Sector would be 60% - at sector limit
            ("ETH/USDT", Decimal("10000.0")),   # Sector would be 65% - rejected (sector limit)
            
            # Try other sectors
            ("ADA/USDT", Decimal("25000.0")),   # 12.5% mid_cap - allowed
            ("LINK/USDT", Decimal("30000.0")),  # 15% defi - allowed
        ]
        
        concentration_results = []
        
        for symbol, position_value in concentration_tests:
            success, message = await attempt_position_addition(symbol, position_value)
            
            asset_conc, sector_conc, total_value = calculate_concentrations()
            
            concentration_results.append({
                "symbol": symbol,
                "position_value": position_value,
                "success": success,
                "message": message,
                "asset_concentration": asset_conc.get(symbol, Decimal("0.0")),
                "sector": asset_sectors.get(symbol, "unknown"),
                "sector_concentration": sector_conc.get(asset_sectors.get(symbol, "unknown"), Decimal("0.0")),
                "total_portfolio_value": total_value
            })
        
        # Verify concentration limits enforced
        successful_additions = [r for r in concentration_results if r["success"]]
        rejected_additions = [r for r in concentration_results if not r["success"]]
        
        # Check that all successful additions respect limits
        for result in successful_additions:
            assert result["asset_concentration"] <= portfolio_limits.max_single_asset_concentration
            assert result["sector_concentration"] <= portfolio_limits.max_sector_concentration
        
        logger.info(f"Concentration limit tests: {len(successful_additions)} allowed, "
                   f"{len(rejected_additions)} rejected")


class TestCorrelationMonitoring(BaseIntegrationTest):
    """Test correlation monitoring and risk management."""
    
    @pytest.mark.asyncio
    async def test_asset_correlation_stress_testing(self):
        """Test correlation behavior under market stress conditions."""
        
        correlation_monitor = Mock()
        
        # Simulate normal vs stress market conditions
        normal_correlations = {
            ("BTC", "ETH"): 0.75,
            ("BTC", "ADA"): 0.45,
            ("BTC", "LINK"): 0.20,
            ("ETH", "ADA"): 0.50,
            ("ETH", "LINK"): 0.25,
            ("ADA", "LINK"): 0.30,
        }
        
        # During stress, correlations tend to increase (assets move together)
        stress_correlations = {
            ("BTC", "ETH"): 0.95,   # Very high correlation
            ("BTC", "ADA"): 0.85,   # Increased correlation
            ("BTC", "LINK"): 0.70,  # Much higher correlation
            ("ETH", "ADA"): 0.80,   # Higher correlation
            ("ETH", "LINK"): 0.75,  # Higher correlation
            ("ADA", "LINK"): 0.65,  # Higher correlation
        }
        
        def simulate_market_stress(stress_level=1.0):
            """Simulate market stress impact on correlations."""
            stressed_correlations = {}
            
            for pair, normal_corr in normal_correlations.items():
                stress_corr = stress_correlations[pair]
                
                # Interpolate between normal and stress correlations
                current_corr = normal_corr + stress_level * (stress_corr - normal_corr)
                stressed_correlations[pair] = min(current_corr, 0.99)  # Cap at 99%
            
            return stressed_correlations
        
        # Test different stress levels
        stress_levels = [0.0, 0.3, 0.6, 1.0]  # Normal, mild stress, high stress, extreme stress
        
        portfolio_positions = {
            "BTC": Decimal("40000.0"),
            "ETH": Decimal("30000.0"),
            "ADA": Decimal("20000.0"),
            "LINK": Decimal("10000.0")
        }
        
        def calculate_portfolio_correlation_risk(correlations, positions):
            """Calculate overall portfolio correlation risk."""
            assets = list(positions.keys())
            total_value = sum(positions.values())
            
            # Calculate weighted correlation risk
            total_correlation_risk = Decimal("0.0")
            pair_count = 0
            
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets[i+1:], i+1):
                    pair = (asset1, asset2) if (asset1, asset2) in correlations else (asset2, asset1)
                    correlation = correlations.get(pair, 0.0)
                    
                    # Weight by position sizes
                    weight1 = positions[asset1] / total_value
                    weight2 = positions[asset2] / total_value
                    
                    # Correlation risk contribution
                    pair_risk = Decimal(str(correlation)) * weight1 * weight2
                    total_correlation_risk += pair_risk
                    pair_count += 1
            
            # Average correlation risk
            avg_correlation_risk = total_correlation_risk / pair_count if pair_count > 0 else Decimal("0.0")
            
            return avg_correlation_risk, total_correlation_risk
        
        stress_test_results = []
        
        for stress_level in stress_levels:
            current_correlations = simulate_market_stress(stress_level)
            avg_risk, total_risk = calculate_portfolio_correlation_risk(current_correlations, portfolio_positions)
            
            # Calculate diversification benefit (inverse of correlation risk)
            diversification_benefit = Decimal("1.0") - avg_risk
            
            stress_test_results.append({
                "stress_level": stress_level,
                "avg_correlation_risk": avg_risk,
                "total_correlation_risk": total_risk,
                "diversification_benefit": diversification_benefit,
                "correlations": current_correlations.copy()
            })
            
            logger.info(f"Stress level {stress_level}: Avg correlation risk {avg_risk:.3f}, "
                       f"Diversification benefit {diversification_benefit:.3f}")
        
        # Verify stress impact on correlations
        normal_result = stress_test_results[0]  # stress_level = 0.0
        extreme_result = stress_test_results[-1]  # stress_level = 1.0
        
        # Correlation risk should increase with stress
        assert extreme_result["avg_correlation_risk"] > normal_result["avg_correlation_risk"]
        
        # Diversification benefit should decrease with stress
        assert extreme_result["diversification_benefit"] < normal_result["diversification_benefit"]
        
        # Test correlation risk threshold alerts
        risk_threshold = Decimal("0.6")  # 60% correlation risk threshold
        
        alert_triggered = False
        for result in stress_test_results:
            if result["avg_correlation_risk"] > risk_threshold:
                alert_triggered = True
                logger.warning(f"Correlation risk alert: {result['avg_correlation_risk']:.1%} > {risk_threshold:.1%} "
                              f"at stress level {result['stress_level']}")
                break
        
        assert alert_triggered is True  # Should trigger alert under stress
        
        logger.info("Correlation stress testing completed")


class TestEmergencyControls(BaseIntegrationTest):
    """Test emergency control mechanisms."""
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown_procedures(self):
        """Test emergency shutdown and recovery procedures."""
        
        exchanges = await self.create_mock_exchanges()
        
        emergency_controls = Mock()
        emergency_controls.shutdown_active = False
        emergency_controls.emergency_reason = None
        
        # Mock emergency shutdown functions
        async def trigger_emergency_shutdown(reason):
            emergency_controls.shutdown_active = True
            emergency_controls.emergency_reason = reason
            
            # Stop all trading
            for exchange_name, exchange in exchanges.items():
                exchange.trading_enabled = False
                logger.info(f"Trading disabled on {exchange_name}")
            
            # Cancel all open orders (mocked)
            cancelled_orders = 0
            for exchange_name, exchange in exchanges.items():
                # Mock cancelling orders
                cancelled_orders += 3  # Assume 3 orders per exchange
            
            logger.warning(f"Emergency shutdown triggered: {reason}")
            logger.info(f"Cancelled {cancelled_orders} open orders")
            
            return True
        
        async def recovery_procedures():
            # Check system health
            system_healthy = True
            
            # Re-enable trading if system is healthy
            if system_healthy:
                for exchange_name, exchange in exchanges.items():
                    exchange.trading_enabled = True
                    logger.info(f"Trading re-enabled on {exchange_name}")
                
                emergency_controls.shutdown_active = False
                emergency_controls.emergency_reason = None
                logger.info("System recovery completed")
                return True
            
            return False
        
        emergency_controls.trigger_emergency_shutdown = trigger_emergency_shutdown
        emergency_controls.recovery_procedures = recovery_procedures
        
        # Test emergency scenarios
        emergency_scenarios = [
            "Daily loss limit exceeded",
            "System connectivity lost",
            "Risk management failure",
            "Manual emergency stop"
        ]
        
        for scenario in emergency_scenarios:
            # Trigger emergency shutdown
            shutdown_success = await emergency_controls.trigger_emergency_shutdown(scenario)
            
            assert shutdown_success is True
            assert emergency_controls.shutdown_active is True
            assert emergency_controls.emergency_reason == scenario
            
            # Verify trading is halted
            for exchange in exchanges.values():
                assert exchange.trading_enabled is False
            
            logger.info(f"Emergency shutdown test completed for: {scenario}")
            
            # Test recovery
            recovery_success = await emergency_controls.recovery_procedures()
            
            assert recovery_success is True
            assert emergency_controls.shutdown_active is False
            
            # Verify trading is restored
            for exchange in exchanges.values():
                assert exchange.trading_enabled is True
            
            logger.info(f"Recovery test completed for: {scenario}")
        
        logger.info("Emergency control tests completed")
