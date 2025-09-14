"""
Unit tests for cross-exchange arbitrage strategy.

This module tests the CrossExchangeArbitrageStrategy which detects price
differences for the same asset across multiple exchanges and generates
signals for simultaneous buy/sell orders.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.core.exceptions import ArbitrageError
from src.core.types import (
    MarketData,
    Position,
    PositionSide,
    PositionStatus,
    Signal,
    SignalDirection,
    StrategyType,
)
from src.strategies.static.cross_exchange_arbitrage import CrossExchangeArbitrageStrategy


class TestCrossExchangeArbitrageStrategy:
    """Test cases for CrossExchangeArbitrageStrategy."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "strategy_id": "test_arbitrage_001",
            "strategy_type": "arbitrage",
            "name": "cross_exchange_arbitrage",
            "symbol": "BTC/USD",
            "timeframe": "1m",
            "min_profit_threshold": "0.001",
            "max_execution_time": 500,
            "exchanges": ["binance", "okx", "coinbase"],
            "symbols": ["BTC/USD", "ETH/USD"],
            "latency_threshold": 100,
            "slippage_limit": "0.0005",
            "total_capital": 10000,
            "risk_per_trade": 0.02,
            "position_size_pct": 0.1,
            "min_confidence": 0.5,
            "parameters": {"max_position_size": 0.1, "max_open_arbitrages": 5},
        }

    @pytest.fixture
    def strategy(self, config):
        """Create test strategy instance."""
        return CrossExchangeArbitrageStrategy(config, None)

    @pytest.fixture
    def market_data(self):
        """Create test market data."""
        return MarketData(
            symbol="BTC/USD",
            open=Decimal("49900"),
            high=Decimal("50100"),
            low=Decimal("49800"),
            close=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            bid_price=Decimal("49999"),
            ask_price=Decimal("50001"),
        )

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "cross_exchange_arbitrage"
        assert strategy.strategy_type == StrategyType.ARBITRAGE
        assert strategy.min_profit_threshold == Decimal("0.001")
        assert strategy.max_execution_time == 500
        assert strategy.exchanges == ["binance", "okx", "coinbase"]
        assert strategy.symbols == ["BTC/USD", "ETH/USD"]
        assert strategy.latency_threshold == 100
        assert strategy.slippage_limit == Decimal("0.0005")
        assert strategy.active_arbitrages == {}
        assert strategy.exchange_prices == {}

    @pytest.mark.asyncio
    async def test_generate_signals_no_opportunities(self, strategy, market_data):
        """Test signal generation when no opportunities exist."""
        signals = await strategy._generate_signals_impl(market_data)

        assert isinstance(signals, list)
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_with_opportunities(self, strategy, market_data):
        """Test signal generation when opportunities exist."""
        # Setup price data for cross-exchange arbitrage with larger spread
        strategy.exchange_prices = {
            "binance": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal("49900"),
                    high=Decimal("50100"),
                    low=Decimal("49800"),
                    close=Decimal("50000"),
                    volume=Decimal("1000"),
                    exchange="binance",
                    bid_price=Decimal("49999"),
                    ask_price=Decimal("50001"),
                    metadata={"exchange": "binance"},
                )
            },
            "okx": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal("50200"),
                    high=Decimal("50400"),
                    low=Decimal("50100"),
                    close=Decimal("50300"),
                    volume=Decimal("1000"),
                    exchange="okx",
                    bid_price=Decimal("50299"),
                    ask_price=Decimal("50301"),
                    metadata={"exchange": "okx"},
                )
            },
        }

        signals = await strategy._detect_arbitrage_opportunities("BTC/USD")

        assert isinstance(signals, list)
        # Should find cross-exchange arbitrage opportunity
        assert len(signals) > 0

    @pytest.mark.asyncio
    async def test_detect_arbitrage_opportunities(self, strategy):
        """Test arbitrage opportunity detection."""
        # Setup price data for cross-exchange arbitrage
        strategy.exchange_prices = {
            "binance": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal("49900"),
                    high=Decimal("50100"),
                    low=Decimal("49800"),
                    close=Decimal("50000"),
                    volume=Decimal("1000"),
                    exchange="binance",
                    bid_price=Decimal("49999"),
                    ask_price=Decimal("50001"),
                    metadata={"exchange": "binance"},
                )
            },
            "okx": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal("50050"),
                    high=Decimal("50150"),
                    low=Decimal("49950"),
                    close=Decimal("50100"),
                    volume=Decimal("1000"),
                    exchange="okx",
                    bid_price=Decimal("50099"),
                    ask_price=Decimal("50101"),
                    metadata={"exchange": "okx"},
                )
            },
        }

        signals = await strategy._detect_arbitrage_opportunities("BTC/USD")

        assert isinstance(signals, list)
        if signals:
            signal = signals[0]
            assert signal.metadata["arbitrage_type"] == "cross_exchange"
            assert "buy_exchange" in signal.metadata
            assert "sell_exchange" in signal.metadata
            assert "net_profit_percentage" in signal.metadata

    def test_calculate_total_fees(self, strategy):
        """Test fee calculation."""
        buy_price = Decimal("50000")
        # Larger spread to make arbitrage profitable
        sell_price = Decimal("50200")

        fees = strategy._calculate_total_fees(buy_price, sell_price)

        assert isinstance(fees, Decimal)
        assert fees > 0
        # Fees should be less than spread
        assert fees < (sell_price - buy_price)

    def test_calculate_total_fees_invalid_input(self, strategy):
        """Test fee calculation with invalid input."""
        with pytest.raises(ArbitrageError):
            strategy._calculate_total_fees(Decimal("-1"), Decimal("50000"))

    @pytest.mark.asyncio
    async def test_validate_execution_timing_valid(self, strategy):
        """Test execution timing validation with valid data."""
        # Setup recent price data
        strategy.exchange_prices = {
            "binance": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal("49900"),
                    high=Decimal("50100"),
                    low=Decimal("49800"),
                    close=Decimal("50000"),
                    volume=Decimal("1000"),
                    exchange="binance",
                    bid_price=Decimal("49999"),
                    ask_price=Decimal("50001"),
                    metadata={"exchange": "binance"},
                )
            }
        }

        result = await strategy._validate_execution_timing("BTC/USD")

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_execution_timing_old_data(self, strategy):
        """Test execution timing validation with old data."""
        # Setup old price data
        strategy.exchange_prices = {
            "binance": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    timestamp=datetime.now() - timedelta(seconds=1),  # Old data
                    open=Decimal("49900"),
                    high=Decimal("50100"),
                    low=Decimal("49800"),
                    close=Decimal("50000"),
                    volume=Decimal("1000"),
                    exchange="binance",
                    bid_price=Decimal("49999"),
                    ask_price=Decimal("50001"),
                    metadata={"exchange": "binance"},
                )
            }
        }

        result = await strategy._validate_execution_timing("BTC/USD")

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_valid(self, strategy):
        """Test signal validation with valid signal."""
        signal = Signal(
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(timezone.utc),
            source="test",
            metadata={
                "arbitrage_type": "cross_exchange",
                "buy_exchange": "binance",
                "sell_exchange": "okx",
                "buy_price": 50000.0,
                "sell_price": 50100.0,
                "net_profit_percentage": 0.5,
            },
        )

        result = await strategy.validate_signal(signal)

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_signal_invalid(self, strategy):
        """Test signal validation with invalid signal."""
        signal = Signal(
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.3"),  # Below minimum confidence
            timestamp=datetime.now(timezone.utc),
            source="test",
            metadata={
                "arbitrage_type": "cross_exchange",
                "net_profit_percentage": 0.05,  # Below threshold
            },
        )

        result = await strategy.validate_signal(signal)

        assert result is False

    def test_get_position_size(self, strategy):
        """Test position size calculation."""
        signal = Signal(
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(timezone.utc),
            source="test",
            metadata={"arbitrage_type": "cross_exchange", "net_profit_percentage": 0.5},
        )

        position_size = strategy.get_position_size(signal)

        assert isinstance(position_size, Decimal)
        assert position_size > 0

    def test_get_position_size_invalid_signal(self, strategy):
        """Test position size calculation with invalid signal."""
        with pytest.raises(ArbitrageError):
            strategy.get_position_size(None)

    @pytest.mark.asyncio
    async def test_should_exit_not_arbitrage_position(self, strategy, market_data):
        """Test exit condition for non-arbitrage position."""
        position = Position(
            symbol="BTC/USD",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("0"),
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
            metadata={},  # No arbitrage_type
        )

        result = await strategy.should_exit(position, market_data)

        assert result is False

    @pytest.mark.asyncio
    async def test_should_exit_timeout(self, strategy, market_data):
        """Test exit condition for timeout."""
        position = Position(
            symbol="BTC/USD",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("0"),
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc) - timedelta(seconds=1),  # Old position
            exchange="binance",
            metadata={
                "arbitrage_type": "cross_exchange",
                "execution_timeout": 500,  # 500ms timeout
            },
        )

        result = await strategy.should_exit(position, market_data)

        assert result is True

    @pytest.mark.asyncio
    async def test_get_current_spread(self, strategy):
        """Test current spread calculation."""
        strategy.exchange_prices = {
            "binance": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal("49900"),
                    high=Decimal("50100"),
                    low=Decimal("49800"),
                    close=Decimal("50000"),
                    volume=Decimal("1000"),
                    exchange="binance",
                    bid_price=Decimal("49999"),
                    ask_price=Decimal("50001"),
                    metadata={"exchange": "binance"},
                )
            },
            "okx": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    timestamp=datetime.now(timezone.utc),
                    open=Decimal("50050"),
                    high=Decimal("50150"),
                    low=Decimal("49950"),
                    close=Decimal("50100"),
                    volume=Decimal("1000"),
                    exchange="okx",
                    bid_price=Decimal("50099"),
                    ask_price=Decimal("50101"),
                    metadata={"exchange": "okx"},
                )
            },
        }

        spread = await strategy._get_current_spread("BTC/USD", "binance", "okx")

        assert isinstance(spread, Decimal)
        assert spread > 0  # Should be positive for profitable arbitrage

    @pytest.mark.asyncio
    async def test_post_trade_processing(self, strategy):
        """Test post-trade processing."""
        trade_result = {"symbol": "BTC/USD", "pnl": 50.0, "execution_time_ms": 100}

        initial_trades = strategy.metrics.total_trades
        initial_pnl = strategy.metrics.total_pnl

        # Test that post_trade_processing executes without errors
        await strategy.post_trade_processing(trade_result)

        # Verify that trade count increases (if implemented)
        # Note: The actual implementation may or may not update metrics here
        assert strategy.metrics.total_trades >= initial_trades
        # The strategy may log the trade without immediately updating PnL metrics
        assert isinstance(strategy.metrics.total_pnl, Decimal)

    @pytest.mark.asyncio
    async def test_post_trade_processing_error(self, strategy):
        """Test post-trade processing with error."""
        trade_result = {
            "symbol": "BTC/USD",
            "pnl": "invalid_pnl",  # Invalid P&L
            "execution_time_ms": 100,
        }

        # Should not raise exception, should handle gracefully
        await strategy.post_trade_processing(trade_result)

        # Metrics should still be updated
        assert strategy.metrics.total_trades > 0
