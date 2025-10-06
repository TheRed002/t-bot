"""
Integration tests for P-013C Backtesting Framework.

Tests cover:
- End-to-end backtesting with real strategy integration
- Database integration for historical data loading
- Risk manager integration
- Metrics calculation integration
- Performance with realistic data volumes
- Error recovery and resilience
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import MetricsCalculator
from src.core.types import MarketData, OrderSide, Position, Signal, SignalDirection
from src.database.manager import DatabaseManager
from src.strategies.base import BaseStrategy, StrategyType


class SimpleMovingAverageStrategy(BaseStrategy):
    """Simple MA crossover strategy for integration testing."""

    def __init__(self, **config):
        super().__init__(config)
        self.short_period = config.get("short_period", 10)
        self.long_period = config.get("long_period", 20)
        self.ma_short_history = {}
        self.ma_long_history = {}

    async def initialize(self, symbol: str, data: pd.DataFrame):
        """Initialize strategy with warm-up data."""
        if len(data) >= self.long_period:
            self.ma_short_history[symbol] = data["close"].rolling(self.short_period).mean().iloc[-1]
            self.ma_long_history[symbol] = data["close"].rolling(self.long_period).mean().iloc[-1]

    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> SignalDirection:
        """Generate signal based on MA crossover."""
        if len(data) < self.long_period:
            return SignalDirection.HOLD

        # Calculate current MAs
        ma_short = data["close"].rolling(self.short_period).mean().iloc[-1]
        ma_long = data["close"].rolling(self.long_period).mean().iloc[-1]

        # Get previous MAs
        prev_ma_short = self.ma_short_history.get(symbol, ma_short)
        prev_ma_long = self.ma_long_history.get(symbol, ma_long)

        # Update history
        self.ma_short_history[symbol] = ma_short
        self.ma_long_history[symbol] = ma_long

        # Generate signals based on crossover
        if ma_short > ma_long and prev_ma_short <= prev_ma_long:
            return SignalDirection.BUY  # Bullish crossover
        elif ma_short < ma_long and prev_ma_short >= prev_ma_long:
            return SignalDirection.SELL  # Bearish crossover
        else:
            return SignalDirection.HOLD

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.TREND_FOLLOWING

    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Internal signal generation implementation."""
        # Convert MarketData to DataFrame for processing
        df = pd.DataFrame([data.__dict__])
        signal_direction = await self.generate_signal(data.symbol, df)

        if signal_direction != SignalDirection.HOLD:
            signal = Signal(
                strategy_name=self.name,
                symbol=data.symbol,
                direction=signal_direction,
                confidence=0.8,
                timestamp=data.timestamp,
            )
            return [signal]
        return []

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal before execution."""
        # Basic validation for test
        return signal.confidence > 0.5

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size for signal."""
        # Simple fixed position size for test
        return Decimal("0.1")

    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed."""
        # Simple exit logic for test - exit after 5% profit or 2% loss
        current_price = data.close
        entry_price = position.entry_price
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        if position.side == OrderSide.BUY:
            return pnl_pct >= 5 or pnl_pct <= -2
        else:  # SELL position
            return pnl_pct <= -5 or pnl_pct >= 2


@pytest.mark.integration
class TestBacktestingIntegration:
    """Integration tests for backtesting framework."""

    @pytest.fixture
    def backtest_config(self):
        """Create realistic backtest configuration."""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),  # 3 months
            symbols=["BTC/USD", "ETH/USD"],
            initial_capital=Decimal("100000"),
            commission=Decimal("0.001"),  # 0.1%
            slippage=Decimal("0.0005"),  # 0.05%
            max_open_positions=3,
            warm_up_period=30,
        )

    @pytest.fixture
    def strategy(self):
        """Create test strategy."""
        return SimpleMovingAverageStrategy(
            name="TestMAStrategy",
            strategy_id="test-ma-strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="BTCUSDT",
            timeframe="1h",
            short_period=10,
            long_period=20,
            position_size_pct=0.1,
        )

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager with realistic data."""
        db_manager = AsyncMock(spec=DatabaseManager)

        # Generate realistic market data
        def generate_market_data(symbol: str, start_date: datetime, end_date: datetime):
            dates = pd.date_range(start=start_date, end=end_date, freq="1H")
            np.random.seed(42 if symbol == "BTC/USD" else 123)

            # Base prices
            base_price = 45000 if symbol == "BTC/USD" else 3000

            # Generate price series with realistic movements
            returns = np.random.normal(0.0001, 0.02, len(dates))
            prices = [base_price]

            for ret in returns:
                prices.append(prices[-1] * (1 + ret))

            prices = prices[1:]  # Remove initial price

            # Create OHLCV data
            data = []
            for i, (date, close_price) in enumerate(zip(dates, prices, strict=False)):
                # Add some intrabar movement
                high = close_price * (1 + abs(np.random.normal(0, 0.005)))
                low = close_price * (1 - abs(np.random.normal(0, 0.005)))
                open_price = prices[i - 1] if i > 0 else close_price
                volume = np.random.uniform(100, 1000)

                data.append(
                    {
                        "timestamp": date,
                        "open": open_price,
                        "high": max(open_price, close_price, high),
                        "low": min(open_price, close_price, low),
                        "close": close_price,
                        "volume": volume,
                    }
                )

            return data

        async def mock_fetch_all(query, symbol, start_date, end_date):
            return generate_market_data(symbol, start_date, end_date)

        db_manager.fetch_all = mock_fetch_all
        return db_manager

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_end_to_end_backtest_single_symbol(
        self, backtest_config, strategy, mock_db_manager
    ):
        """Test complete backtest process with single symbol."""
        # Modify config for single symbol
        config = backtest_config
        config.symbols = ["BTC/USD"]

        engine = BacktestEngine(config=config, strategy=strategy, db_manager=mock_db_manager)

        result = await engine.run()

        # Verify result structure
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0
        assert len(result.trades) >= 0
        assert len(result.daily_returns) >= 0

        # Verify financial metrics
        assert isinstance(result.total_return, Decimal)
        assert isinstance(result.annual_return, Decimal)
        assert isinstance(result.max_drawdown, Decimal)
        assert 0 <= result.win_rate <= 100
        assert result.profit_factor >= 0

        # Verify equity curve progression
        assert all(isinstance(point["equity"], (int, float)) for point in result.equity_curve)
        assert all("timestamp" in point for point in result.equity_curve)

        # Verify trade data integrity
        for trade in result.trades:
            assert "symbol" in trade
            assert "entry_time" in trade
            assert "exit_time" in trade
            assert "pnl" in trade
            assert trade["exit_time"] >= trade["entry_time"]

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_end_to_end_backtest_multi_symbol(
        self, backtest_config, strategy, mock_db_manager
    ):
        """Test complete backtest process with multiple symbols."""
        engine = BacktestEngine(
            config=backtest_config, strategy=strategy, db_manager=mock_db_manager
        )

        result = await engine.run()

        # Should handle multiple symbols
        symbols_traded = set(trade["symbol"] for trade in result.trades)
        assert len(symbols_traded) <= len(backtest_config.symbols)

        # Verify no symbol-specific errors
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backtest_with_risk_manager_integration(
        self, backtest_config, strategy, mock_db_manager
    ):
        """Test backtest integration with risk manager."""
        from src.risk_management.risk_manager import RiskManager

        # Create mock risk manager
        risk_manager = AsyncMock(spec=RiskManager)
        risk_manager.calculate_position_size = AsyncMock(return_value=Decimal("5000"))

        engine = BacktestEngine(
            config=backtest_config,
            strategy=strategy,
            risk_manager=risk_manager,
            db_manager=mock_db_manager,
        )

        result = await engine.run()

        # Risk manager should have been called for position sizing
        if result.total_trades > 0:
            risk_manager.calculate_position_size.assert_called()

        # Verify results
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backtest_metrics_integration(self, backtest_config, strategy, mock_db_manager):
        """Test integration with metrics calculator."""
        engine = BacktestEngine(
            config=backtest_config, strategy=strategy, db_manager=mock_db_manager
        )

        result = await engine.run()

        # Test metrics calculation directly
        calculator = MetricsCalculator(risk_free_rate=0.02)

        additional_metrics = calculator.calculate_all(
            equity_curve=result.equity_curve,
            trades=result.trades,
            daily_returns=result.daily_returns,
            initial_capital=float(backtest_config.initial_capital),
        )

        # Verify additional metrics are reasonable
        assert "volatility" in additional_metrics
        assert "sharpe_ratio" in additional_metrics
        assert "sortino_ratio" in additional_metrics

        # Cross-verify some metrics
        if result.equity_curve:
            initial_equity = result.equity_curve[0]["equity"]
            final_equity = result.equity_curve[-1]["equity"]
            calculated_return = (final_equity - initial_equity) / initial_equity * 100

            # Should be close to backtest result
            assert abs(float(result.total_return) - calculated_return) < 1.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backtest_error_recovery(self, backtest_config, strategy, mock_db_manager):
        """Test backtest error recovery and resilience."""
        # Modify strategy to occasionally fail
        original_generate_signal = strategy.generate_signal
        call_count = 0

        async def failing_generate_signal(symbol, data):
            nonlocal call_count
            call_count += 1
            if call_count == 5:  # Fail on 5th call
                raise Exception("Simulated strategy error")
            return await original_generate_signal(symbol, data)

        strategy.generate_signal = failing_generate_signal

        engine = BacktestEngine(
            config=backtest_config, strategy=strategy, db_manager=mock_db_manager
        )

        # Should handle strategy errors gracefully
        result = await engine.run()

        # Should complete despite error
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backtest_commission_slippage_accuracy(
        self, backtest_config, strategy, mock_db_manager
    ):
        """Test accurate application of commissions and slippage."""
        # Set known commission and slippage
        backtest_config.commission = Decimal("0.002")  # 0.2%
        backtest_config.slippage = Decimal("0.001")  # 0.1%

        engine = BacktestEngine(
            config=backtest_config, strategy=strategy, db_manager=mock_db_manager
        )

        result = await engine.run()

        if result.total_trades > 0:
            # Calculate expected commission costs
            total_position_value = 0
            for trade in result.trades:
                position_value = abs(trade["size"])
                total_position_value += position_value

            # Expected total commission (both entry and exit)
            expected_commission = total_position_value * float(backtest_config.commission) * 2

            # Total return should account for commission costs
            # This is a rough check - actual calculation is more complex
            gross_return = sum(trade["pnl"] for trade in result.trades)

            # Net return should be less than gross return due to costs
            assert (
                float(result.total_return) * float(backtest_config.initial_capital) / 100
                <= gross_return
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backtest_position_management(self, backtest_config, strategy, mock_db_manager):
        """Test position management and limits."""
        backtest_config.max_open_positions = 2  # Limit positions

        engine = BacktestEngine(
            config=backtest_config, strategy=strategy, db_manager=mock_db_manager
        )

        result = await engine.run()

        # Verify position limits were respected
        # Track concurrent positions by analyzing trade times
        open_positions = []
        max_concurrent = 0

        for trade in result.trades:
            # Remove closed positions
            open_positions = [
                pos for pos in open_positions if pos["exit_time"] > trade["entry_time"]
            ]

            # Add new position
            open_positions.append(trade)
            max_concurrent = max(max_concurrent, len(open_positions))

        assert max_concurrent <= backtest_config.max_open_positions

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backtest_drawdown_limits(self, backtest_config, strategy, mock_db_manager):
        """Test drawdown limit enforcement."""
        engine = BacktestEngine(
            config=backtest_config, strategy=strategy, db_manager=mock_db_manager
        )

        # Patch the risk limit check to simulate high drawdown
        original_check_risk_limits = engine._check_risk_limits
        risk_limit_triggered = False

        async def mock_check_risk_limits():
            nonlocal risk_limit_triggered
            if len(engine._equity_curve) > 10 and not risk_limit_triggered:
                # Simulate 25% drawdown
                engine._equity_curve.append(
                    {
                        "timestamp": engine._current_time,
                        "equity": float(backtest_config.initial_capital) * 0.75,
                    }
                )
                risk_limit_triggered = True
            await original_check_risk_limits()

        engine._check_risk_limits = mock_check_risk_limits

        result = await engine.run()

        # Should complete even with risk limit enforcement
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backtest_performance_large_dataset(self, strategy, mock_db_manager):
        """Test backtest performance with large dataset."""
        # Large backtest configuration
        large_config = BacktestConfig(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2023, 12, 31),  # 2 years
            symbols=["BTC/USD", "ETH/USD", "ADA/USD"],
            initial_capital=Decimal("1000000"),
            warm_up_period=50,
        )

        engine = BacktestEngine(config=large_config, strategy=strategy, db_manager=mock_db_manager)

        import time

        start_time = time.time()

        result = await engine.run()

        execution_time = time.time() - start_time

        # Should complete within reasonable time (2 minutes for 2 years)
        assert execution_time < 120.0

        # Should produce meaningful results
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 100  # Should have many data points
        assert len(result.daily_returns) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backtest_data_quality_handling(self, backtest_config, strategy):
        """Test handling of poor quality market data."""
        # Create mock DB manager with problematic data
        db_manager = AsyncMock(spec=DatabaseManager)

        async def mock_fetch_with_gaps(query, symbol, start_date, end_date):
            # Generate data with gaps and inconsistencies
            dates = pd.date_range(start=start_date, end=end_date, freq="1H")
            data = []

            for i, date in enumerate(dates):
                # Introduce random gaps
                if np.random.random() < 0.05:  # 5% chance of missing data
                    continue

                # Introduce price inconsistencies
                if i == 0:
                    price = 45000 if symbol == "BTC/USD" else 3000
                else:
                    # Occasional large jumps
                    if np.random.random() < 0.02:  # 2% chance of large jump
                        price = data[-1]["close"] * (1 + np.random.uniform(-0.1, 0.1))
                    else:
                        price = data[-1]["close"] * (1 + np.random.normal(0, 0.01))

                # Sometimes high < low (invalid data)
                high = price * (1 + abs(np.random.normal(0, 0.005)))
                low = price * (1 - abs(np.random.normal(0, 0.005)))

                # Introduce occasional invalid data
                if np.random.random() < 0.01:  # 1% chance
                    high, low = low, high  # Swap to create invalid data

                data.append(
                    {
                        "timestamp": date,
                        "open": price,
                        "high": high,
                        "low": low,
                        "close": price,
                        "volume": max(0, np.random.normal(500, 100)),  # Ensure positive volume
                    }
                )

            return data

        db_manager.fetch_all = mock_fetch_with_gaps

        engine = BacktestEngine(config=backtest_config, strategy=strategy, db_manager=db_manager)

        # Should handle data quality issues gracefully
        result = await engine.run()

        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0
        # Results might be impacted by data quality but shouldn't crash

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_multiple_strategies_comparison(self, backtest_config, mock_db_manager):
        """Test running multiple strategies for comparison."""
        strategies = [
            SimpleMovingAverageStrategy(name="MA_Fast", short_period=5, long_period=15),
            SimpleMovingAverageStrategy(name="MA_Slow", short_period=15, long_period=30),
        ]

        results = []

        for strategy in strategies:
            engine = BacktestEngine(
                config=backtest_config, strategy=strategy, db_manager=mock_db_manager
            )

            result = await engine.run()
            results.append((strategy.name, result))

        # Should have results for both strategies
        assert len(results) == 2

        # Compare results
        for name, result in results:
            assert result.total_trades >= 0
            assert len(result.equity_curve) > 0
            assert result.metadata["strategy"] == name

        # Results should be different (different strategies)
        if all(result.total_trades > 0 for _, result in results):
            assert results[0][1].total_return != results[1][1].total_return


@pytest.mark.integration
class TestBacktestingDatabaseIntegration:
    """Test backtesting integration with real database components."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_connection_handling(self, clean_database):
        """Test proper database connection handling during backtest."""
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 7),  # 1 week
            symbols=["BTC/USD"],
            initial_capital=Decimal("10000"),
        )

        strategy = SimpleMovingAverageStrategy(name="TestStrategy")

        # Use real database manager
        from src.database.manager import DatabaseManager

        db_manager = DatabaseManager()

        # Insert some test data
        await db_manager.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP,
                symbol VARCHAR(20),
                open DECIMAL(20,8),
                high DECIMAL(20,8),
                low DECIMAL(20,8),
                close DECIMAL(20,8),
                volume DECIMAL(20,8)
            )
        """)

        # Insert sample data
        test_data = []
        base_time = datetime(2023, 1, 1)
        for i in range(168):  # 1 week of hourly data
            timestamp = base_time + timedelta(hours=i)
            price = 45000 + i * 10 + np.random.uniform(-50, 50)
            test_data.append(
                (timestamp, "BTC/USD", price, price + 50, price - 50, price + 25, 1000)
            )

        await db_manager.execute_many(
            """
            INSERT INTO market_data (timestamp, symbol, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
            test_data,
        )

        engine = BacktestEngine(config=config, strategy=strategy, db_manager=db_manager)

        result = await engine.run()

        # Should successfully load and process data
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

        # Clean up
        await db_manager.execute("DROP TABLE IF EXISTS market_data")


# Stress tests
@pytest.mark.stress
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_backtest_memory_usage():
    """Test backtest memory usage with large datasets."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),  # 4 years
        symbols=["BTC/USD", "ETH/USD", "LTC/USD", "ADA/USD"],
        initial_capital=Decimal("1000000"),
    )

    strategy = SimpleMovingAverageStrategy(name="MemoryTestStrategy")

    # Mock large dataset
    db_manager = AsyncMock(spec=DatabaseManager)

    async def large_dataset_fetch(query, symbol, start_date, end_date):
        # Generate 4 years of hourly data (35k+ points per symbol)
        dates = pd.date_range(start=start_date, end=end_date, freq="1H")
        np.random.seed(42)

        data = []
        price = 45000 if symbol == "BTC/USD" else 3000

        for date in dates:
            price *= 1 + np.random.normal(0, 0.01)
            data.append(
                {
                    "timestamp": date,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000,
                }
            )

        return data

    db_manager.fetch_all = large_dataset_fetch

    engine = BacktestEngine(config=config, strategy=strategy, db_manager=db_manager)

    result = await engine.run()

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    # Memory increase should be reasonable (< 500MB for 4 years of 4 symbols)
    assert memory_increase < 500

    # Should still produce valid results
    assert result.total_trades >= 0
    assert len(result.equity_curve) > 1000  # Many data points
