"""
Real Service Fixtures for Strategy Integration Tests.

This module provides fixtures for creating real service instances
with proper dependency injection and database integration.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator

import pytest

import pytest_asyncio
from src.core.config import Config
from src.core.types import MarketData, StrategyConfig, StrategyType
from src.strategies.dependencies import StrategyServiceContainer, create_strategy_service_container
from src.strategies.service import StrategyService
from src.database.connection import DatabaseConnectionManager
from src.database.models import Base


@pytest_asyncio.fixture
async def real_database_connection() -> AsyncGenerator[DatabaseConnectionManager, None]:
    """Create real database connection for testing."""
    from tests.conftest import clean_database

    # Use the clean_database fixture approach that's working in other tests
    config = Config()
    db_connection = DatabaseConnectionManager(config)
    await db_connection.initialize()

    # Create tables if they don't exist using the async engine directly
    async with db_connection.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield db_connection

    # Cleanup
    await db_connection.cleanup()


@pytest_asyncio.fixture
async def real_strategy_service_container(
    real_database_connection,
) -> AsyncGenerator[StrategyServiceContainer, None]:
    """Create real strategy service container with all dependencies."""
    config = Config()

    # Create container builder
    container_builder = create_strategy_service_container()

    # Build container with real services
    container = await container_builder.build_container(
        config=config,
        database=real_database_connection
    )

    # Verify container is ready
    assert container.is_ready()

    yield container

    # Cleanup
    await container_builder.cleanup()


@pytest_asyncio.fixture
async def real_risk_manager(real_strategy_service_container):
    """Get real risk manager from container."""
    return real_strategy_service_container.risk_service


@pytest_asyncio.fixture
async def real_data_service(real_strategy_service_container):
    """Get real data service from container."""
    return real_strategy_service_container.data_service


@pytest_asyncio.fixture
async def real_execution_service(real_strategy_service_container):
    """Get real execution service from container."""
    return real_strategy_service_container.execution_service


@pytest.fixture
def real_strategy_configs():
    """Create real strategy configurations for testing."""
    return {
        "mean_reversion": StrategyConfig(
            strategy_id="test_mean_reversion_001",
            name="test_mean_reversion_strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            enabled=True,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.6"),
            max_positions=5,
            position_size_pct=Decimal("0.02"),
            stop_loss_pct=Decimal("0.02"),
            take_profit_pct=Decimal("0.04"),
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
                "exit_threshold": Decimal("0.5"),
                "volume_filter": True,
                "min_volume_ratio": Decimal("1.5"),
                "atr_period": 14,
                "atr_multiplier": Decimal("2.0"),
            },
        ),
        "trend_following": StrategyConfig(
            strategy_id="test_trend_following_001",
            name="test_trend_following_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            enabled=True,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.7"),
            max_positions=3,
            position_size_pct=Decimal("0.03"),
            stop_loss_pct=Decimal("0.03"),
            take_profit_pct=Decimal("0.06"),
            parameters={
                "fast_ma_period": 10,
                "slow_ma_period": 20,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "volume_confirmation": True,
                "min_volume_ratio": Decimal("1.2"),
                "trailing_stop_enabled": True,
                "trailing_stop_pct": Decimal("0.01"),
            },
        ),
        "breakout": StrategyConfig(
            strategy_id="test_breakout_001",
            name="test_breakout_strategy",
            strategy_type=StrategyType.BREAKOUT,
            enabled=True,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.65"),
            max_positions=4,
            position_size_pct=Decimal("0.025"),
            stop_loss_pct=Decimal("0.025"),
            take_profit_pct=Decimal("0.05"),
            parameters={
                "lookback_period": 20,
                "consolidation_period": 5,
                "volume_confirmation": True,
                "min_volume_ratio": Decimal("1.5"),
                "false_breakout_filter": True,
                "false_breakout_threshold": Decimal("0.02"),
                "atr_period": 14,
                "atr_multiplier": Decimal("2.0"),
                "target_multiplier": Decimal("3.0"),
            },
        ),
    }


def generate_realistic_market_data_sequence(
    symbol: str = "BTC/USDT",
    periods: int = 50,
    base_price: Decimal = Decimal("50000.00"),
    base_volume: Decimal = Decimal("1000.00"),
    pattern: str = "mixed"
) -> list[MarketData]:
    """
    Generate realistic market data sequence for testing.

    Args:
        symbol: Trading symbol
        periods: Number of periods to generate
        base_price: Starting price
        base_volume: Base volume
        pattern: Market pattern ('uptrend', 'downtrend', 'sideways', 'mixed')

    Returns:
        List of MarketData objects with realistic price movements
    """
    market_data_sequence = []
    current_price = base_price

    for i in range(periods):
        timestamp = datetime.now(timezone.utc) - timedelta(hours=periods-i)

        # Generate price movement based on pattern
        if pattern == "uptrend":
            price_change = Decimal(str(i * 50 + (i % 3 - 1) * 100))
        elif pattern == "downtrend":
            price_change = Decimal(str(-i * 30 + (i % 3 - 1) * 80))
        elif pattern == "sideways":
            price_change = Decimal(str((i % 7 - 3) * 150))
        else:  # mixed pattern
            if i < periods // 3:
                # Initial uptrend
                price_change = Decimal(str(i * 40 + (i % 3 - 1) * 120))
            elif i < 2 * periods // 3:
                # Sideways movement
                price_change = Decimal(str((i % 5 - 2) * 200))
            else:
                # Final downtrend
                price_change = Decimal(str((2 * periods // 3 - i) * 35))

        # Calculate OHLC prices
        close_price = current_price + price_change
        open_price = current_price

        # Add realistic intraday volatility
        volatility = Decimal(str(abs(i % 11) * 50))
        high_price = max(open_price, close_price) + volatility
        low_price = min(open_price, close_price) - volatility

        # Generate realistic volume
        volume_factor = Decimal("1.0") + (Decimal(str(i % 13)) / Decimal("10"))
        volume = base_volume * volume_factor

        market_data = MarketData(
            symbol=symbol,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timestamp=timestamp,
            exchange="binance",
            bid_price=close_price - Decimal("1.0"),
            ask_price=close_price + Decimal("1.0"),
        )

        market_data_sequence.append(market_data)
        current_price = close_price

    return market_data_sequence


def generate_mean_reversion_scenario() -> list[MarketData]:
    """Generate market data that should trigger mean reversion signals."""
    base_price = Decimal("50000.00")
    sequence = []

    # Establish a mean price
    for i in range(20):
        price = base_price + Decimal(str((i % 5 - 2) * 100))  # Oscillate around mean
        sequence.append(MarketData(
            symbol="BTC/USDT",
            open=price - Decimal("50"),
            high=price + Decimal("75"),
            low=price - Decimal("75"),
            close=price,
            volume=Decimal("2000.00"),
            timestamp=datetime.now(timezone.utc) - timedelta(hours=25-i),
            exchange="binance"
        ))

    # Add extreme deviation (should trigger mean reversion signal)
    extreme_price = base_price - Decimal("2500.00")  # Significant deviation
    sequence.append(MarketData(
        symbol="BTC/USDT",
        open=base_price - Decimal("1000"),
        high=base_price - Decimal("500"),
        low=extreme_price - Decimal("100"),
        close=extreme_price,
        volume=Decimal("5000.00"),  # High volume for confirmation
        timestamp=datetime.now(timezone.utc),
        exchange="binance"
    ))

    return sequence


def generate_trend_following_scenario() -> list[MarketData]:
    """Generate market data that should trigger trend following signals."""
    base_price = Decimal("50000.00")
    sequence = []

    # Build strong uptrend
    for i in range(25):
        trend_price = base_price + Decimal(str(i * 200))  # Strong upward trend
        sequence.append(MarketData(
            symbol="BTC/USDT",
            open=trend_price - Decimal("50"),
            high=trend_price + Decimal("150"),
            low=trend_price - Decimal("100"),
            close=trend_price,
            volume=Decimal("3000.00") + Decimal(str(i * 50)),  # Increasing volume
            timestamp=datetime.now(timezone.utc) - timedelta(hours=30-i),
            exchange="binance"
        ))

    return sequence


def generate_breakout_scenario() -> list[MarketData]:
    """Generate market data that should trigger breakout signals."""
    base_price = Decimal("50000.00")
    sequence = []

    # Consolidation phase
    for i in range(20):
        consolidation_price = base_price + Decimal(str((i % 3 - 1) * 50))  # Tight range
        sequence.append(MarketData(
            symbol="BTC/USDT",
            open=consolidation_price - Decimal("25"),
            high=consolidation_price + Decimal("75"),
            low=consolidation_price - Decimal("75"),
            close=consolidation_price,
            volume=Decimal("1500.00"),  # Lower volume during consolidation
            timestamp=datetime.now(timezone.utc) - timedelta(hours=25-i),
            exchange="binance"
        ))

    # Breakout
    breakout_price = base_price + Decimal("1500.00")  # Strong breakout
    sequence.append(MarketData(
        symbol="BTC/USDT",
        open=base_price + Decimal("100"),
        high=breakout_price + Decimal("200"),
        low=base_price + Decimal("50"),
        close=breakout_price,
        volume=Decimal("6000.00"),  # High volume breakout
        timestamp=datetime.now(timezone.utc),
        exchange="binance"
    ))

    return sequence


@pytest.fixture
def market_data_scenarios():
    """Provide different market data scenarios for testing."""
    return {
        "mean_reversion": generate_mean_reversion_scenario(),
        "trend_following": generate_trend_following_scenario(),
        "breakout": generate_breakout_scenario(),
        "mixed": generate_realistic_market_data_sequence(pattern="mixed"),
        "uptrend": generate_realistic_market_data_sequence(pattern="uptrend"),
        "downtrend": generate_realistic_market_data_sequence(pattern="downtrend"),
        "sideways": generate_realistic_market_data_sequence(pattern="sideways"),
    }