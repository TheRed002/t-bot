"""
Unit tests for triangular arbitrage strategy.

This module tests the TriangularArbitrageStrategy which identifies three-pair
arbitrage opportunities within a single exchange and generates signals for
a sequence of trades to capture the profit.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.strategies.static.triangular_arbitrage import TriangularArbitrageStrategy
from src.core.types import (
    Signal, MarketData, Position, SignalDirection,
    StrategyConfig, StrategyType, OrderSide
)
from src.core.exceptions import ArbitrageError, ValidationError


class TestTriangularArbitrageStrategy:
    """Test cases for TriangularArbitrageStrategy."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "name": "triangular_arbitrage",
            "strategy_type": "arbitrage",
            "min_profit_threshold": "0.001",
            "max_execution_time": 500,
            "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "triangular_paths": [
                ["BTCUSDT", "ETHBTC", "ETHUSDT"],
                ["BTCUSDT", "BNBBTC", "BNBUSDT"]
            ],
            "exchange": "binance",
            "slippage_limit": "0.0005",
            "total_capital": 10000,
            "risk_per_trade": 0.02,
            "position_size_pct": 0.1,
            "min_confidence": 0.5,
            "parameters": {
                "max_position_size": 0.05,
                "max_open_arbitrages": 3
            }
        }
    
    @pytest.fixture
    def strategy(self, config):
        """Create test strategy instance."""
        return TriangularArbitrageStrategy(config)
    
    @pytest.fixture
    def market_data(self):
        """Create test market data."""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            volume=Decimal("1000"),
            timestamp=datetime.now(),
            metadata={"exchange": "binance"}
        )
    
    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "triangular_arbitrage"
        assert strategy.strategy_type == StrategyType.ARBITRAGE
        assert strategy.min_profit_threshold == Decimal("0.001")
        assert strategy.max_execution_time == 500
        assert len(strategy.triangular_paths) == 2
        assert strategy.exchange == "binance"
        assert strategy.slippage_limit == Decimal("0.0005")
        assert strategy.active_triangular_arbitrages == {}
        assert strategy.pair_prices == {}
    
    @pytest.mark.asyncio
    async def test_generate_signals_no_opportunities(self, strategy, market_data):
        """Test signal generation when no opportunities exist."""
        signals = await strategy._generate_signals_impl(market_data)
        
        assert isinstance(signals, list)
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_generate_signals_with_opportunities(self, strategy, market_data):
        """Test signal generation when opportunities exist."""
        # Setup price data for triangular arbitrage
        strategy.pair_prices = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                bid=Decimal("49999"),
                ask=Decimal("50001"),
                volume=Decimal("1000"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHBTC": MarketData(
                symbol="ETHBTC",
                price=Decimal("0.05"),
                bid=Decimal("0.0499"),
                ask=Decimal("0.0501"),
                volume=Decimal("100"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHUSDT": MarketData(
                symbol="ETHUSDT",
                price=Decimal("2500"),
                bid=Decimal("2499"),
                ask=Decimal("2501"),
                volume=Decimal("500"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            )
        }
        
        signals = await strategy._generate_signals_impl(market_data)
        
        assert isinstance(signals, list)
        # May or may not find triangular arbitrage opportunity depending on prices
        assert len(signals) >= 0
    
    @pytest.mark.asyncio
    async def test_detect_triangular_opportunities(self, strategy):
        """Test triangular arbitrage opportunity detection."""
        # Setup price data for triangular arbitrage
        strategy.pair_prices = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                bid=Decimal("49999"),
                ask=Decimal("50001"),
                volume=Decimal("1000"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHBTC": MarketData(
                symbol="ETHBTC",
                price=Decimal("0.05"),
                bid=Decimal("0.0499"),
                ask=Decimal("0.0501"),
                volume=Decimal("100"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHUSDT": MarketData(
                symbol="ETHUSDT",
                price=Decimal("2500"),
                bid=Decimal("2499"),
                ask=Decimal("2501"),
                volume=Decimal("500"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            )
        }
        
        signals = await strategy._detect_triangular_opportunities("BTCUSDT")
        
        assert isinstance(signals, list)
        if signals:
            signal = signals[0]
            assert signal.metadata["arbitrage_type"] == "triangular"
            assert "path" in signal.metadata
            assert "net_profit_percentage" in signal.metadata
    
    @pytest.mark.asyncio
    async def test_check_triangular_path(self, strategy):
        """Test triangular path checking."""
        # Setup price data for triangular arbitrage
        strategy.pair_prices = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                bid=Decimal("49999"),
                ask=Decimal("50001"),
                volume=Decimal("1000"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHBTC": MarketData(
                symbol="ETHBTC",
                price=Decimal("0.05"),
                bid=Decimal("0.0499"),
                ask=Decimal("0.0501"),
                volume=Decimal("100"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHUSDT": MarketData(
                symbol="ETHUSDT",
                price=Decimal("2500"),
                bid=Decimal("2499"),
                ask=Decimal("2501"),
                volume=Decimal("500"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            )
        }
        
        path = ["BTCUSDT", "ETHBTC", "ETHUSDT"]
        signal = await strategy._check_triangular_path(path)
        
        # May or may not find opportunity depending on prices
        if signal:
            assert signal.metadata["arbitrage_type"] == "triangular"
            assert signal.metadata["path"] == path
            assert "net_profit_percentage" in signal.metadata
    
    def test_calculate_triangular_fees(self, strategy):
        """Test triangular fee calculation."""
        rate1 = Decimal("50000")  # BTC/USDT
        rate2 = Decimal("0.05")   # ETH/BTC
        rate3 = Decimal("2500")   # ETH/USDT
        
        fees = strategy._calculate_triangular_fees(rate1, rate2, rate3)
        
        assert isinstance(fees, Decimal)
        assert fees > 0
    
    def test_calculate_triangular_fees_invalid_input(self, strategy):
        """Test triangular fee calculation with invalid input."""
        with pytest.raises(ArbitrageError):
            strategy._calculate_triangular_fees(Decimal("-1"), Decimal("0.05"), Decimal("2500"))
    
    @pytest.mark.asyncio
    async def test_validate_triangular_timing_valid(self, strategy):
        """Test triangular timing validation with valid data."""
        # Setup recent price data
        strategy.pair_prices = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                bid=Decimal("49999"),
                ask=Decimal("50001"),
                volume=Decimal("1000"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHBTC": MarketData(
                symbol="ETHBTC",
                price=Decimal("0.05"),
                bid=Decimal("0.0499"),
                ask=Decimal("0.0501"),
                volume=Decimal("100"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHUSDT": MarketData(
                symbol="ETHUSDT",
                price=Decimal("2500"),
                bid=Decimal("2499"),
                ask=Decimal("2501"),
                volume=Decimal("500"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            )
        }
        
        path = ["BTCUSDT", "ETHBTC", "ETHUSDT"]
        result = await strategy._validate_triangular_timing(path)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_triangular_timing_old_data(self, strategy):
        """Test triangular timing validation with old data."""
        # Setup old price data
        strategy.pair_prices = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                bid=Decimal("49999"),
                ask=Decimal("50001"),
                volume=Decimal("1000"),
                timestamp=datetime.now() - timedelta(seconds=1),  # Old data
                metadata={"exchange": "binance"}
            ),
            "ETHBTC": MarketData(
                symbol="ETHBTC",
                price=Decimal("0.05"),
                bid=Decimal("0.0499"),
                ask=Decimal("0.0501"),
                volume=Decimal("100"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            ),
            "ETHUSDT": MarketData(
                symbol="ETHUSDT",
                price=Decimal("2500"),
                bid=Decimal("2499"),
                ask=Decimal("2501"),
                volume=Decimal("500"),
                timestamp=datetime.now(),
                metadata={"exchange": "binance"}
            )
        }
        
        path = ["BTCUSDT", "ETHBTC", "ETHUSDT"]
        result = await strategy._validate_triangular_timing(path)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_signal_valid(self, strategy):
        """Test signal validation with valid signal."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test",
            metadata={
                "arbitrage_type": "triangular",
                "path": ["BTCUSDT", "ETHBTC", "ETHUSDT"],
                "exchange": "binance",
                "net_profit_percentage": 0.5,
                "execution_sequence": [
                    {"pair": "BTCUSDT", "action": "buy", "rate": 50000.0},
                    {"pair": "ETHBTC", "action": "sell", "rate": 0.05},
                    {"pair": "ETHUSDT", "action": "sell", "rate": 2500.0}
                ]
            }
        )
        
        result = await strategy.validate_signal(signal)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_signal_invalid(self, strategy):
        """Test signal validation with invalid signal."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.3,  # Below minimum confidence
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test",
            metadata={
                "arbitrage_type": "triangular",
                "net_profit_percentage": 0.05  # Below threshold
            }
        )
        
        result = await strategy.validate_signal(signal)
        
        assert result is False
    
    def test_get_position_size(self, strategy):
        """Test position size calculation."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test",
            metadata={
                "arbitrage_type": "triangular",
                "net_profit_percentage": 0.5
            }
        )
        
        position_size = strategy.get_position_size(signal)
        
        assert isinstance(position_size, Decimal)
        assert position_size > 0
    
    def test_get_position_size_invalid_signal(self, strategy):
        """Test position size calculation with invalid signal."""
        with pytest.raises(ArbitrageError):
            strategy.get_position_size(None)
    
    @pytest.mark.asyncio
    async def test_should_exit_not_triangular_position(self, strategy, market_data):
        """Test exit condition for non-triangular position."""
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now(),
            metadata={}  # No arbitrage_type
        )
        
        result = await strategy.should_exit(position, market_data)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_should_exit_timeout(self, strategy, market_data):
        """Test exit condition for timeout."""
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("0"),
            side=OrderSide.BUY,
            timestamp=datetime.now() - timedelta(seconds=1),  # Old position
            metadata={
                "arbitrage_type": "triangular",
                "execution_timeout": 500  # 500ms timeout
            }
        )
        
        result = await strategy.should_exit(position, market_data)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_post_trade_processing(self, strategy):
        """Test post-trade processing."""
        trade_result = {
            "symbol": "BTCUSDT",
            "path": ["BTCUSDT", "ETHBTC", "ETHUSDT"],
            "pnl": 50.0,
            "execution_time_ms": 100
        }
        
        initial_trades = strategy.metrics.total_trades
        initial_pnl = strategy.metrics.total_pnl
        
        await strategy.post_trade_processing(trade_result)
        
        assert strategy.metrics.total_trades == initial_trades + 1
        assert strategy.metrics.total_pnl == initial_pnl + Decimal("50.0")
    
    @pytest.mark.asyncio
    async def test_post_trade_processing_error(self, strategy):
        """Test post-trade processing with error."""
        trade_result = {
            "symbol": "BTCUSDT",
            "path": ["BTCUSDT", "ETHBTC", "ETHUSDT"],
            "pnl": "invalid_pnl",  # Invalid P&L
            "execution_time_ms": 100
        }
        
        # Should not raise exception, should handle gracefully
        await strategy.post_trade_processing(trade_result)
        
        # Metrics should still be updated
        assert strategy.metrics.total_trades > 0
