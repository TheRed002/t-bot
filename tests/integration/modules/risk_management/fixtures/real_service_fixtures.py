"""
Real Service Fixtures for Risk Management Integration Tests.

This module provides fixtures for creating real risk management service instances
with proper dependency injection, database integration, and NO MOCKS.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio

from src.core.config import Config
from src.core.types import (
    MarketData,
    Position,
    PositionSide,
    PositionStatus,
    OrderRequest,
    OrderSide,
    OrderType,
    Signal,
    SignalDirection,
    RiskLevel,
)
from src.risk_management.service import RiskService
from src.risk_management.factory import RiskManagementFactory


@pytest_asyncio.fixture
async def real_risk_service(clean_database) -> AsyncGenerator[RiskService, None]:
    """Create real risk management service with all dependencies."""
    from src.core.dependency_injection import DependencyInjector
    from src.database.repository.risk import RiskMetricsRepository, PortfolioRepository

    config = Config()

    # Create DI container
    injector = DependencyInjector()

    # Register dependencies
    injector.register_singleton("Config", lambda: config)
    injector.register_singleton("DatabaseService", lambda: clean_database)

    # Create repositories
    risk_metrics_repo = RiskMetricsRepository(clean_database)
    portfolio_repo = PortfolioRepository(clean_database)

    # Create risk service directly (bypass factory complexity for tests)
    risk_service = RiskService(
        risk_metrics_repository=risk_metrics_repo,
        portfolio_repository=portfolio_repo,
        state_service=None,  # Optional in constructor
        analytics_service=None,  # Optional in constructor
        config=config,
        correlation_id="test_correlation_id"
    )

    await risk_service.initialize()

    # Service is ready after initialization
    yield risk_service

    # Cleanup
    await risk_service.cleanup()


@pytest_asyncio.fixture
async def minimal_state_service(clean_database) -> AsyncGenerator:
    """Create minimal StateService for testing."""
    from src.state.state_service import StateService

    config = Config()
    # StateService only needs config - internal services are created automatically
    state_service = StateService(config=config)

    await state_service.initialize()
    yield state_service
    await state_service.cleanup()


@pytest_asyncio.fixture
async def real_risk_factory(clean_database, minimal_state_service) -> AsyncGenerator[RiskManagementFactory, None]:
    """Create real risk management factory."""
    from src.core.dependency_injection import DependencyInjector
    from src.risk_management.di_registration import register_risk_management_services

    config = Config()

    # Create DI container
    injector = DependencyInjector()

    # Register dependencies
    injector.register_singleton("Config", lambda: config)
    injector.register_singleton("DatabaseService", lambda: clean_database)
    injector.register_singleton("StateService", lambda: minimal_state_service)

    # Register risk management services (CRITICAL - this registers RiskService)
    register_risk_management_services(injector)

    # Create factory
    factory = RiskManagementFactory(injector=injector)

    yield factory

    # Cleanup
    await factory.stop_services()


def generate_realistic_price_sequence(
    base_price: Decimal,
    periods: int = 30,
    volatility: Decimal = Decimal("0.02"),
    trend: Decimal = Decimal("0.0")
) -> list[Decimal]:
    """
    Generate realistic price sequence with volatility and trend.

    Uses geometric Brownian motion for realistic price movements.
    """
    import numpy as np

    prices = [base_price]
    dt = Decimal("1")  # Time step (1 period)

    for _ in range(periods - 1):
        # Random return based on volatility
        random_return = Decimal(str(np.random.normal(0, float(volatility))))

        # Price change = drift + diffusion
        price_change = prices[-1] * (trend * dt + random_return)
        new_price = prices[-1] + price_change

        # Ensure price stays positive
        new_price = max(new_price, base_price * Decimal("0.5"))
        prices.append(new_price)

    return prices


def generate_realistic_market_data_sequence(
    symbol: str = "BTC/USDT",
    base_price: Decimal = Decimal("50000"),
    periods: int = 30,
    volatility: Decimal = Decimal("0.02"),
    trend: Decimal = Decimal("0.0"),
    exchange: str = "binance"
) -> list[MarketData]:
    """Generate realistic market data sequence for testing."""
    prices = generate_realistic_price_sequence(base_price, periods, volatility, trend)

    market_data_list = []
    start_time = datetime.now(timezone.utc) - timedelta(hours=periods)

    for i, close_price in enumerate(prices):
        # Add some intraday variation
        high_price = close_price * Decimal("1.01")
        low_price = close_price * Decimal("0.99")
        open_price = prices[i-1] if i > 0 else close_price

        # Volume varies with volatility
        base_volume = Decimal("1000")
        volume = base_volume * (Decimal("1") + abs(close_price - open_price) / close_price * Decimal("10"))

        market_data = MarketData(
            symbol=symbol,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timestamp=start_time + timedelta(hours=i),
            exchange=exchange
        )
        market_data_list.append(market_data)

    return market_data_list


def generate_bull_market_scenario(symbol: str = "BTC/USDT") -> list[MarketData]:
    """Generate bull market scenario with strong uptrend."""
    return generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=Decimal("50000"),
        periods=50,
        volatility=Decimal("0.015"),
        trend=Decimal("0.002")  # 0.2% daily uptrend
    )


def generate_bear_market_scenario(symbol: str = "BTC/USDT") -> list[MarketData]:
    """Generate bear market scenario with downtrend."""
    return generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=Decimal("50000"),
        periods=50,
        volatility=Decimal("0.025"),
        trend=Decimal("-0.0015")  # -0.15% daily downtrend
    )


def generate_high_volatility_scenario(symbol: str = "BTC/USDT") -> list[MarketData]:
    """Generate high volatility scenario."""
    return generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=Decimal("50000"),
        periods=50,
        volatility=Decimal("0.05"),  # High volatility
        trend=Decimal("0.0")
    )


def generate_crash_scenario(symbol: str = "BTC/USDT") -> list[MarketData]:
    """Generate market crash scenario with sudden drop."""
    # Normal market first
    normal_data = generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=Decimal("50000"),
        periods=20,
        volatility=Decimal("0.015"),
        trend=Decimal("0.001")
    )

    # Then crash
    crash_data = generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=normal_data[-1].close,
        periods=10,
        volatility=Decimal("0.08"),  # Extreme volatility
        trend=Decimal("-0.05")  # Severe downtrend
    )

    # Then recovery
    recovery_data = generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=crash_data[-1].close,
        periods=20,
        volatility=Decimal("0.03"),
        trend=Decimal("0.002")
    )

    return normal_data + crash_data[1:] + recovery_data[1:]


@pytest.fixture
def sample_position() -> Position:
    """Create a sample position for testing."""
    return Position(
        position_id=str(uuid4()),
        symbol="BTC/USDT",
        quantity=Decimal("0.1"),
        entry_price=Decimal("50000"),
        current_price=Decimal("51000"),
        unrealized_pnl=Decimal("100"),
        realized_pnl=Decimal("0"),
        side=PositionSide.LONG,
        status=PositionStatus.OPEN,
        opened_at=datetime.now(timezone.utc),
        exchange="binance",
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000")
    )


@pytest.fixture
def sample_positions() -> list[Position]:
    """Create sample portfolio positions."""
    return [
        Position(
            position_id=str(uuid4()),
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            realized_pnl=Decimal("0"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
            stop_loss=Decimal("49000"),
            take_profit=Decimal("52000")
        ),
        Position(
            position_id=str(uuid4()),
            symbol="ETH/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("3000"),
            current_price=Decimal("3100"),
            unrealized_pnl=Decimal("100"),
            realized_pnl=Decimal("0"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
            stop_loss=Decimal("2900"),
            take_profit=Decimal("3200")
        ),
    ]


@pytest.fixture
def sample_signal() -> Signal:
    """Create a sample trading signal."""
    return Signal(
        signal_id=str(uuid4()),
        strategy_id="test_strategy",
        strategy_name="Test Strategy",
        symbol="BTC/USDT",
        direction=SignalDirection.BUY,
        confidence=Decimal("0.85"),
        strength=Decimal("0.75"),
        source="test",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "entry_price": Decimal("50000"),
            "stop_loss": Decimal("49000"),
            "take_profit": Decimal("52000"),
            "risk_reward_ratio": Decimal("2.0")
        }
    )


@pytest.fixture
def sample_order_request() -> OrderRequest:
    """Create a sample order request."""
    return OrderRequest(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
        exchange="binance",
        strategy_id="test_strategy",
        signal_id=str(uuid4())
    )
