"""
Backtesting Module Integration Test Configuration.

Provides pytest fixtures for backtesting integration tests with full DI support.
"""

import logging
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from src.utils.synthetic_data_generator import generate_synthetic_ohlcv_data

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def di_container():
    """
    Provide fully configured DI container with all services registered.

    Uses master DI registration to ensure all dependencies are properly configured
    in the correct order without circular dependency issues.
    """
    from tests.integration.conftest import cleanup_di_container, register_all_services_for_testing

    container = register_all_services_for_testing()
    yield container

    # Cleanup to prevent resource leaks
    await cleanup_di_container(container)


@pytest.fixture
def sample_historical_data():
    """Generate sample historical market data for tests."""

    def _generate_data(
        symbol: str = "BTCUSDT",
        days: int = 30,
        interval_minutes: int = 60,
        initial_price: float = 50000.0,
    ):
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        return generate_synthetic_ohlcv_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval_minutes=interval_minutes,
            initial_price=initial_price,
        )

    return _generate_data


@pytest.fixture
def backtest_request_factory():
    """Factory for creating backtest requests."""

    def _create_request(
        symbols: list[str] = None,
        days: int = 30,
        initial_capital: float = 10000.0,
        strategy_config: dict = None,
    ):
        from src.backtesting.service import BacktestRequest
        from src.utils.decimal_utils import to_decimal

        if symbols is None:
            symbols = ["BTCUSDT"]

        if strategy_config is None:
            strategy_config = {"type": "MA_Crossover", "short_period": 10, "long_period": 20}

        return BacktestRequest(
            strategy_config=strategy_config,
            symbols=symbols,
            exchange="binance",
            start_date=datetime.now(timezone.utc) - timedelta(days=days),
            end_date=datetime.now(timezone.utc),
            initial_capital=to_decimal(str(initial_capital)),
            timeframe="1h",
            commission_rate=to_decimal("0.001"),
            slippage_rate=to_decimal("0.0005"),
        )

    return _create_request


@pytest.fixture
def backtest_result_factory():
    """Factory for creating backtest results."""

    def _create_result(
        total_trades: int = 10,
        win_rate: float = 60.0,
        total_return: float = 15.0,
        initial_capital: float = 10000.0,
    ):
        from src.backtesting.engine import BacktestResult
        from src.utils.decimal_utils import to_decimal

        final_capital = initial_capital * (1 + total_return / 100)

        return BacktestResult(
            total_return_pct=to_decimal(str(total_return)),
            annual_return_pct=to_decimal(str(total_return * 2)),
            sharpe_ratio=to_decimal("1.5"),
            max_drawdown_pct=to_decimal("10.0"),
            win_rate_pct=to_decimal(str(win_rate)),
            total_trades=total_trades,
            winning_trades=int(total_trades * win_rate / 100),
            losing_trades=total_trades - int(total_trades * win_rate / 100),
            initial_capital=to_decimal(str(initial_capital)),
            final_capital=to_decimal(str(final_capital)),
            peak_capital=to_decimal(str(final_capital * 1.1)),
            lowest_capital=to_decimal(str(initial_capital * 0.9)),
        )

    return _create_result
