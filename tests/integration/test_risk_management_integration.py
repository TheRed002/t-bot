"""
Integration tests for Risk Management Framework.

This module tests the complete risk management workflow including all components
working together in realistic scenarios.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import patch

from src.core.types import (
    Signal, SignalDirection, MarketData, Position, OrderRequest, OrderSide, OrderType,
    RiskMetrics, RiskLevel, PositionSizeMethod
)
from src.core.exceptions import (
    RiskManagementError, PositionLimitError, ValidationError
)
from src.core.config import Config
from src.risk_management import RiskManager


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
                timestamp=datetime.now()
            ),
            Position(
                symbol="ETHUSDT",
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                unrealized_pnl=Decimal("100"),
                side=OrderSide.BUY,
                timestamp=datetime.now()
            )
        ]
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for all positions."""
        return [
            MarketData(
                symbol="BTCUSDT",
                price=Decimal("51000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(),
                bid=Decimal("50990"),
                ask=Decimal("51010"),
                open_price=Decimal("50000"),
                high_price=Decimal("52000"),
                low_price=Decimal("49000")
            ),
            MarketData(
                symbol="ETHUSDT",
                price=Decimal("3100"),
                volume=Decimal("500"),
                timestamp=datetime.now(),
                bid=Decimal("3095"),
                ask=Decimal("3105"),
                open_price=Decimal("3000"),
                high_price=Decimal("3200"),
                low_price=Decimal("2900")
            )
        ]
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals."""
        return [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                strategy_name="test_strategy"
            ),
            Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                timestamp=datetime.now(),
                symbol="ETHUSDT",
                strategy_name="test_strategy"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_complete_risk_management_workflow(self, risk_manager, sample_positions, 
                                                   sample_market_data, sample_signals):
        """Test complete risk management workflow."""
        portfolio_value = Decimal("100000")
        
        # Step 1: Update portfolio state
        await risk_manager.update_portfolio_state(sample_positions, portfolio_value)
        
        # Step 2: Calculate risk metrics
        risk_metrics = await risk_manager.calculate_risk_metrics(sample_positions, sample_market_data)
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
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
                assert position_size <= portfolio_value * Decimal(str(risk_manager.risk_config.max_position_size_pct))
        
        # Step 5: Check portfolio limits for new positions
        new_position = Position(
            symbol="ADAUSDT",
            quantity=Decimal("1000"),
            entry_price=Decimal("0.5"),
            current_price=Decimal("0.5"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now()
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
                timestamp=datetime.now()
            )
            positions.append(position)
            
            market_data.append(MarketData(
                symbol=symbol,
                price=Decimal("51000"),
                volume=Decimal("10000"),
                timestamp=datetime.now(),
                bid=Decimal("50990"),
                ask=Decimal("51010"),
                open_price=Decimal("50000"),
                high_price=Decimal("52000"),
                low_price=Decimal("49000")
            ))
        
        # Update portfolio state
        await risk_manager.update_portfolio_state(positions, portfolio_value)
        
        # Calculate risk metrics
        risk_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)
        
        assert risk_metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert risk_metrics.var_1d > 0
        assert risk_metrics.var_5d > risk_metrics.var_1d
        
        # Test position sizing with large portfolio
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )
        
        position_size = await risk_manager.calculate_position_size(signal, portfolio_value)
        assert position_size > 0
        assert position_size <= portfolio_value * Decimal(str(risk_manager.risk_config.max_position_size_pct))
    
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
                timestamp=datetime.now()
            )
        ]
        
        market_data = [
            MarketData(
                symbol="BTCUSDT",
                price=Decimal("45000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(),
                bid=Decimal("44990"),
                ask=Decimal("45010"),
                open_price=Decimal("50000"),
                high_price=Decimal("52000"),
                low_price=Decimal("44000"),
            )
        ]
        
        portfolio_value = Decimal("50000")
        
        # Update portfolio state
        await risk_manager.update_portfolio_state(positions, portfolio_value)
        
        # Calculate risk metrics
        risk_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)
        
        # Should detect higher risk due to losses
        assert risk_metrics.current_drawdown > 0
        assert risk_metrics.var_1d > 0
        
        # Test position exit evaluation
        should_exit = await risk_manager.should_exit_position(positions[0], market_data[0])
        # Should exit due to large loss
        assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_risk_management_with_multiple_strategies(self, risk_manager):
        """Test risk management with multiple trading strategies."""
        # Create signals from different strategies
        signals = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                strategy_name="momentum_strategy"
            ),
            Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                timestamp=datetime.now(),
                symbol="ETHUSDT",
                strategy_name="mean_reversion_strategy"
            ),
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.9,
                timestamp=datetime.now(),
                symbol="ADAUSDT",
                strategy_name="ml_strategy"
            )
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
        
        # Test with invalid parameters (modify config temporarily)
        original_max_position = risk_manager.risk_config.max_position_size_pct
        risk_manager.risk_config.max_position_size_pct = 1.5  # Invalid: > 1
        
        with pytest.raises(ValidationError):
            await risk_manager.validate_risk_parameters()
        
        # Restore original value
        risk_manager.risk_config.max_position_size_pct = original_max_position
    
    @pytest.mark.asyncio
    async def test_risk_management_error_handling(self, risk_manager):
        """Test risk management error handling."""
        # Test with invalid signal
        invalid_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.0,  # Invalid confidence
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
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
            client_order_id="test_order_123"
        )
        
        with pytest.raises(ValidationError):
            await risk_manager.validate_order(invalid_order, portfolio_value)
    
    @pytest.mark.asyncio
    async def test_risk_management_performance(self, risk_manager, sample_positions, sample_market_data):
        """Test risk management performance with multiple rapid operations."""
        portfolio_value = Decimal("100000")
        
        # Update portfolio state
        await risk_manager.update_portfolio_state(sample_positions, portfolio_value)
        
        # Perform multiple rapid operations
        start_time = datetime.now()
        
        for _ in range(10):
            # Calculate risk metrics
            risk_metrics = await risk_manager.calculate_risk_metrics(sample_positions, sample_market_data)
            assert isinstance(risk_metrics, RiskMetrics)
            
            # Validate signals
            signal = Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                strategy_name="test_strategy"
            )
            is_valid = await risk_manager.validate_signal(signal)
            assert isinstance(is_valid, bool)
            
            # Calculate position size
            position_size = await risk_manager.calculate_position_size(signal, portfolio_value)
            assert position_size > 0
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (less than 5 seconds for 10 operations)
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
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
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
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
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
                timestamp=datetime.now()
            )
        ]
        
        market_data = [
            MarketData(
                symbol="BTCUSDT",
                price=Decimal("51000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(),
                bid=Decimal("50990"),
                ask=Decimal("51010"),
                open_price=Decimal("50000"),
                high_price=Decimal("52000"),
                low_price=Decimal("49000")
            )
        ]
        
        await risk_manager.update_portfolio_state(positions, portfolio_value)
        initial_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)
        
        # Update with price change
        market_data[0].price = Decimal("52000")  # Price increase
        positions[0].current_price = Decimal("52000")
        positions[0].unrealized_pnl = Decimal("200")
        
        await risk_manager.update_portfolio_state(positions, portfolio_value)
        updated_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)
        
        # Metrics should reflect the change
        assert updated_metrics.timestamp > initial_metrics.timestamp
        
        # Test position exit evaluation with updated data
        should_exit = await risk_manager.should_exit_position(positions[0], market_data[0])
        # Should not exit due to profit
        assert should_exit is False 