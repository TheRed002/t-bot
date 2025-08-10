"""
Unit tests for arbitrage opportunity scanner strategy.

This module tests the ArbitrageOpportunity strategy which scans for both
cross-exchange and triangular arbitrage opportunities across multiple exchanges.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.strategies.static.arbitrage_scanner import ArbitrageOpportunity
from src.core.types import (
    Signal, MarketData, Position, SignalDirection,
    StrategyConfig, StrategyType, OrderSide
)
from src.core.exceptions import ArbitrageError, ValidationError


class TestArbitrageOpportunity:
    """Test cases for ArbitrageOpportunity strategy."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "name": "arbitrage_scanner",
            "strategy_type": "arbitrage",
            "scan_interval": 100,
            "min_profit_threshold": "0.001",
            "max_opportunities": 5,
            "exchanges": ["binance", "okx", "coinbase"],
            "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "triangular_paths": [
                ["BTCUSDT", "ETHBTC", "ETHUSDT"],
                ["BTCUSDT", "BNBBTC", "BNBUSDT"]
            ],
            "total_capital": 10000,
            "risk_per_trade": 0.02,
            "position_size_pct": 0.1,
            "min_confidence": 0.5,
            "parameters": {
                "max_position_size": 0.1,
                "max_open_arbitrages": 3
            }
        }

    @pytest.fixture
    def strategy(self, config):
        """Create test strategy instance."""
        return ArbitrageOpportunity(config)

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
        assert strategy.name == "arbitrage_scanner"
        assert strategy.strategy_type == StrategyType.ARBITRAGE
        assert strategy.scan_interval == 100
        assert strategy.min_profit_threshold == Decimal("0.001")
        assert strategy.max_opportunities == 5
        assert strategy.exchanges == ["binance", "okx", "coinbase"]
        assert strategy.symbols == ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        assert len(strategy.triangular_paths) == 2
        assert strategy.active_opportunities == {}
        assert strategy.exchange_prices == {}
        assert strategy.opportunity_history == []
        assert strategy.scan_count == 0
        assert strategy.opportunities_found == 0
        assert strategy.execution_success_rate == 0.0

    @pytest.mark.asyncio
    async def test_generate_signals_no_opportunities(
            self, strategy, market_data):
        """Test signal generation when no opportunities exist."""
        signals = await strategy._generate_signals_impl(market_data)

        assert isinstance(signals, list)
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_with_opportunities(
            self, strategy, market_data):
        """Test signal generation when opportunities exist."""
        # Setup price data for cross-exchange arbitrage with larger spread
        strategy.exchange_prices = {
            "binance": {
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50000"),
                    bid=Decimal("49999"),
                    ask=Decimal("50001"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(),
                    metadata={"exchange": "binance"}
                )
            },
            "okx": {
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50500"),
                    bid=Decimal("50499"),
                    ask=Decimal("50501"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(),
                    metadata={"exchange": "okx"}
                )
            }
        }

        signals = await strategy._scan_arbitrage_opportunities()

        assert isinstance(signals, list)
        # Should find cross-exchange arbitrage opportunity
        assert len(signals) > 0

    @pytest.mark.asyncio
    async def test_scan_arbitrage_opportunities(self, strategy):
        """Test comprehensive arbitrage opportunity scanning."""
        # Setup price data
        strategy.exchange_prices = {
            "binance": {
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50000"),
                    bid=Decimal("49999"),
                    ask=Decimal("50001"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(),
                    metadata={"exchange": "binance"}
                )
            },
            "okx": {
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50100"),
                    bid=Decimal("50099"),
                    ask=Decimal("50101"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(),
                    metadata={"exchange": "okx"}
                )
            }
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
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50000"),
                    bid=Decimal("49999"),
                    ask=Decimal("50001"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(),
                    metadata={"exchange": "binance"}
                )
            },
            "okx": {
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50100"),
                    bid=Decimal("50099"),
                    ask=Decimal("50101"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(),
                    metadata={"exchange": "okx"}
                )
            }
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
            strategy._calculate_cross_exchange_fees(
                Decimal("-1"), Decimal("50000"))

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
            strategy._calculate_triangular_fees(
                Decimal("-1"), Decimal("0.05"), Decimal("2500"))

    def test_calculate_priority(self, strategy):
        """Test priority calculation."""
        profit_percentage = 0.5
        arbitrage_type = "cross_exchange"

        priority = strategy._calculate_priority(
            profit_percentage, arbitrage_type)

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
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                strategy_name="test",
                metadata={"opportunity_priority": 0.3}
            ),
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.9,
                timestamp=datetime.now(),
                symbol="ETHUSDT",
                strategy_name="test",
                metadata={"opportunity_priority": 0.8}
            )
        ]

        prioritized = strategy._prioritize_opportunities(signals)

        assert isinstance(prioritized, list)
        assert len(prioritized) <= strategy.max_opportunities
        if len(prioritized) > 1:
            # Should be sorted by priority (highest first)
            assert prioritized[0].metadata["opportunity_priority"] >= prioritized[1].metadata["opportunity_priority"]

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
                "arbitrage_type": "cross_exchange",
                "net_profit_percentage": 0.5,
                "buy_exchange": "binance",
                "sell_exchange": "okx",
                "buy_price": 50000.0,
                "sell_price": 50100.0
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
                "arbitrage_type": "cross_exchange",
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
                "arbitrage_type": "cross_exchange",
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
    async def test_should_exit_not_arbitrage_position(
            self, strategy, market_data):
        """Test exit condition for non-arbitrage position."""
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
                "arbitrage_type": "cross_exchange",
                "execution_timeout": 500  # 500ms timeout
            }
        )

        result = await strategy.should_exit(position, market_data)

        assert result is True

    @pytest.mark.asyncio
    async def test_get_current_cross_exchange_spread(self, strategy):
        """Test current cross-exchange spread calculation."""
        strategy.exchange_prices = {
            "binance": {
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50000"),
                    bid=Decimal("49999"),
                    ask=Decimal("50001"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(),
                    metadata={"exchange": "binance"}
                )
            },
            "okx": {
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    price=Decimal("50100"),
                    bid=Decimal("50099"),
                    ask=Decimal("50101"),
                    volume=Decimal("1000"),
                    timestamp=datetime.now(),
                    metadata={"exchange": "okx"}
                )
            }
        }

        spread = await strategy._get_current_cross_exchange_spread(
            "BTCUSDT", "binance", "okx"
        )

        assert isinstance(spread, Decimal)
        assert spread > 0  # Should be positive for profitable arbitrage

    @pytest.mark.asyncio
    async def test_check_triangular_path(self, strategy):
        """Test triangular path checking."""
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

    @pytest.mark.asyncio
    async def test_post_trade_processing(self, strategy):
        """Test post-trade processing."""
        trade_result = {
            "symbol": "BTCUSDT",
            "arbitrage_type": "cross_exchange",
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
            "arbitrage_type": "cross_exchange",
            "pnl": "invalid_pnl",  # Invalid P&L
            "execution_time_ms": 100
        }

        # Should not raise exception, should handle gracefully
        await strategy.post_trade_processing(trade_result)

        # Metrics should still be updated
        assert strategy.metrics.total_trades > 0
