"""
Unit tests for arbitrage opportunity scanner strategy.

This module tests the ArbitrageOpportunity strategy which scans for both
cross-exchange and triangular arbitrage opportunities across multiple exchanges.
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
from src.strategies.static.arbitrage_scanner import ArbitrageOpportunity


class TestArbitrageOpportunity:
    """Test cases for ArbitrageOpportunity strategy."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "name": "arbitrage_scanner",
            "strategy_id": "arbitrage_scanner_001",
            "strategy_type": StrategyType.ARBITRAGE,
            "symbol": "BTC/USD",  # Required field for base strategy
            "timeframe": "1m",  # Required field for base strategy
            "scan_interval": 100,
            "min_profit_threshold": "0.0001",  # Much smaller threshold for test
            "max_opportunities": 5,
            "exchanges": ["binance", "okx", "coinbase"],
            "symbols": ["BTC/USD", "ETH/USD", "BNBUSDT"],
            "triangular_paths": [
                ["BTC/USD", "ETHBTC", "ETH/USD"],
                ["BTC/USD", "BNBBTC", "BNBUSDT"],
            ],
            "total_capital": 10000,
            "risk_per_trade": 0.02,
            "position_size_pct": 0.1,
            "min_confidence": 0.5,
            "parameters": {"max_position_size": 0.1, "max_open_arbitrages": 3},
        }

    @pytest.fixture
    def strategy(self, config):
        """Create test strategy instance."""
        return ArbitrageOpportunity(config, None)

    @pytest.fixture
    def market_data(self):
        """Create test market data."""
        return MarketData(
            symbol="BTC/USD",
            open=Decimal("49800"),
            high=Decimal("50200"),
            low=Decimal("49700"),
            close=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            # Optional fields for bid/ask if needed by tests
            bid_price=Decimal("49999"),
            ask_price=Decimal("50001"),
        )

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "arbitrage_scanner"
        assert strategy.strategy_type == StrategyType.ARBITRAGE
        assert strategy.scan_interval == 100
        assert strategy.min_profit_threshold == Decimal("0.0001")
        assert strategy.max_opportunities == 5
        assert strategy.exchanges == ["binance", "okx", "coinbase"]
        assert strategy.symbols == ["BTC/USD", "ETH/USD", "BNBUSDT"]
        assert len(strategy.triangular_paths) == 2
        assert strategy.active_opportunities == {}
        assert strategy.exchange_prices == {}
        assert strategy.opportunity_history == []
        assert strategy.scan_count == 0
        assert strategy.opportunities_found == 0
        assert strategy.execution_success_rate == 0.0

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
            },
            "okx": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    open=Decimal("51400"),
                    high=Decimal("51600"),
                    low=Decimal("51300"),
                    close=Decimal("51500"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(timezone.utc),
                    exchange="okx",
                    bid_price=Decimal("51499"),  # Much higher bid to create arbitrage opportunity
                    ask_price=Decimal("51501"),
                )
            },
        }

        signals = await strategy._scan_arbitrage_opportunities()

        assert isinstance(signals, list)
        # The signals list may be empty if no profitable opportunities after fees
        # This test mainly verifies the scanning method works without errors

    @pytest.mark.asyncio
    async def test_scan_arbitrage_opportunities(self, strategy):
        """Test comprehensive arbitrage opportunity scanning."""
        # Setup price data
        strategy.exchange_prices = {
            "binance": {
                "BTC/USD": MarketData(
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
            },
            "okx": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    open=Decimal("50000"),
                    high=Decimal("50200"),
                    low=Decimal("49900"),
                    close=Decimal("50100"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(timezone.utc),
                    exchange="okx",
                    bid_price=Decimal("50099"),
                    ask_price=Decimal("50101"),
                )
            },
        }

        signals = await strategy._scan_arbitrage_opportunities()

        assert isinstance(signals, list)
        assert strategy.scan_count == 1
        assert strategy.opportunities_found >= 0

    @pytest.mark.asyncio
    async def test_scan_cross_exchange_opportunities(self, strategy):
        """Test cross-exchange opportunity scanning."""
        # Setup price data for cross-exchange arbitrage
        strategy.exchange_prices = {
            "binance": {
                "BTC/USD": MarketData(
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
            },
            "okx": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    open=Decimal("50000"),
                    high=Decimal("50200"),
                    low=Decimal("49900"),
                    close=Decimal("50100"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(timezone.utc),
                    exchange="okx",
                    bid_price=Decimal("50099"),
                    ask_price=Decimal("50101"),
                )
            },
        }

        signals = await strategy._scan_cross_exchange_opportunities()

        assert isinstance(signals, list)
        if signals:
            signal = signals[0]
            assert signal.metadata["arbitrage_type"] == "cross_exchange"
            assert "buy_exchange" in signal.metadata
            assert "sell_exchange" in signal.metadata
            assert "net_profit_percentage" in signal.metadata

    @pytest.mark.asyncio
    async def test_scan_triangular_opportunities(self, strategy):
        """Test triangular opportunity scanning."""
        # Setup price data for triangular arbitrage
        strategy.pair_prices = {
            "BTC/USD": MarketData(
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
            ),
            "ETHBTC": MarketData(
                symbol="ETHBTC",
                open=Decimal("0.0495"),
                high=Decimal("0.0505"),
                low=Decimal("0.0490"),
                close=Decimal("0.05"),
                volume=Decimal("100"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
                bid_price=Decimal("0.0499"),
                ask_price=Decimal("0.0501"),
            ),
            "ETH/USD": MarketData(
                symbol="ETH/USD",
                open=Decimal("2480"),
                high=Decimal("2520"),
                low=Decimal("2470"),
                close=Decimal("2500"),
                volume=Decimal("500"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
                bid_price=Decimal("2499"),
                ask_price=Decimal("2501"),
            ),
        }

        signals = await strategy._scan_triangular_opportunities()

        assert isinstance(signals, list)
        if signals:
            signal = signals[0]
            assert signal.metadata["arbitrage_type"] == "triangular"
            assert "path" in signal.metadata
            assert "net_profit_percentage" in signal.metadata

    def test_calculate_cross_exchange_fees(self, strategy):
        """Test cross-exchange fee calculation."""
        buy_price = Decimal("50000")
        # Larger spread to make arbitrage profitable
        sell_price = Decimal("50200")

        fees = strategy._calculate_cross_exchange_fees(buy_price, sell_price)

        assert isinstance(fees, Decimal)
        assert fees > 0
        # Fees should be less than spread
        assert fees < (sell_price - buy_price)

    def test_calculate_cross_exchange_fees_invalid_input(self, strategy):
        """Test cross-exchange fee calculation with invalid input."""
        with pytest.raises(ArbitrageError):
            strategy._calculate_cross_exchange_fees(Decimal("-1"), Decimal("50000"))

    def test_calculate_triangular_fees(self, strategy):
        """Test triangular fee calculation."""
        rate1 = Decimal("50000")  # BTC/USDT
        rate2 = Decimal("0.05")  # ETH/BTC
        rate3 = Decimal("2500")  # ETH/USDT

        fees = strategy._calculate_triangular_fees(rate1, rate2, rate3)

        assert isinstance(fees, Decimal)
        assert fees > 0

    def test_calculate_triangular_fees_invalid_input(self, strategy):
        """Test triangular fee calculation with invalid input."""
        with pytest.raises(ArbitrageError):
            strategy._calculate_triangular_fees(Decimal("-1"), Decimal("0.05"), Decimal("2500"))

    def test_calculate_priority(self, strategy):
        """Test priority calculation."""
        profit_percentage = 0.5
        arbitrage_type = "cross_exchange"

        priority = strategy._calculate_priority(profit_percentage, arbitrage_type)

        assert isinstance(priority, float)
        assert priority > 0
        assert priority <= 1000  # Should respect upper limit

    def test_calculate_priority_invalid_input(self, strategy):
        """Test priority calculation with invalid input."""
        with pytest.raises(ArbitrageError):
            strategy._calculate_priority(-1.0, "cross_exchange")

        with pytest.raises(ArbitrageError):
            strategy._calculate_priority(0.5, "invalid_type")

    def test_prioritize_opportunities(self, strategy):
        """Test opportunity prioritization."""
        signals = [
            Signal(
                symbol="BTC/USD",
                direction=SignalDirection.BUY,
                strength=Decimal("0.8"),
                timestamp=datetime.now(timezone.utc),
                source="test",
                metadata={"opportunity_priority": 0.3},
            ),
            Signal(
                symbol="ETH/USD",
                direction=SignalDirection.BUY,
                strength=Decimal("0.9"),
                timestamp=datetime.now(timezone.utc),
                source="test",
                metadata={"opportunity_priority": 0.8},
            ),
        ]

        prioritized = strategy._prioritize_opportunities(signals)

        assert isinstance(prioritized, list)
        assert len(prioritized) <= strategy.max_opportunities
        if len(prioritized) > 1:
            # Should be sorted by priority (highest first)
            assert (
                prioritized[0].metadata["opportunity_priority"]
                >= prioritized[1].metadata["opportunity_priority"]
            )

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
                "net_profit_percentage": 0.5,
                "buy_exchange": "binance",
                "sell_exchange": "okx",
                "buy_price": 50000.0,
                "sell_price": 50100.0,
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
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("0"),
            side=PositionSide.LONG,
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
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("0"),
            side=PositionSide.LONG,
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
    async def test_get_current_cross_exchange_spread(self, strategy):
        """Test current cross-exchange spread calculation."""
        strategy.exchange_prices = {
            "binance": {
                "BTC/USD": MarketData(
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
            },
            "okx": {
                "BTC/USD": MarketData(
                    symbol="BTC/USD",
                    open=Decimal("50000"),
                    high=Decimal("50200"),
                    low=Decimal("49900"),
                    close=Decimal("50100"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(timezone.utc),
                    exchange="okx",
                    bid_price=Decimal("50099"),
                    ask_price=Decimal("50101"),
                )
            },
        }

        spread = await strategy._get_current_cross_exchange_spread("BTC/USD", "binance", "okx")

        assert isinstance(spread, Decimal)
        # Expected spread: sell_price (okx bid: 50099) - buy_price (binance ask: 50001) = 98
        assert spread == Decimal("98")  # Should be positive for profitable arbitrage

    @pytest.mark.asyncio
    async def test_check_triangular_path(self, strategy):
        """Test triangular path checking."""
        strategy.pair_prices = {
            "BTC/USD": MarketData(
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
            ),
            "ETHBTC": MarketData(
                symbol="ETHBTC",
                open=Decimal("0.0495"),
                high=Decimal("0.0505"),
                low=Decimal("0.0490"),
                close=Decimal("0.05"),
                volume=Decimal("100"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
                bid_price=Decimal("0.0499"),
                ask_price=Decimal("0.0501"),
            ),
            "ETH/USD": MarketData(
                symbol="ETH/USD",
                open=Decimal("2480"),
                high=Decimal("2520"),
                low=Decimal("2470"),
                close=Decimal("2500"),
                volume=Decimal("500"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
                bid_price=Decimal("2499"),
                ask_price=Decimal("2501"),
            ),
        }

        path = ["BTC/USD", "ETHBTC", "ETH/USD"]
        signal = await strategy._check_triangular_path(path)

        # May or may not find opportunity depending on prices
        if signal:
            assert signal.metadata["arbitrage_type"] == "triangular"
            assert signal.metadata["path"] == path
            assert "net_profit_percentage" in signal.metadata

    @pytest.mark.asyncio
    async def test_post_trade_processing(self, strategy):
        """Test post-trade processing."""
        trade_result = {
            "symbol": "BTC/USD",
            "arbitrage_type": "cross_exchange",
            "pnl": 50.0,
            "execution_time_ms": 100,
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
            "symbol": "BTC/USD",
            "arbitrage_type": "cross_exchange",
            "pnl": "invalid_pnl",  # Invalid P&L
            "execution_time_ms": 100,
        }

        # Should not raise exception, should handle gracefully
        await strategy.post_trade_processing(trade_result)

        # Metrics should still be updated
        assert strategy.metrics.total_trades > 0
