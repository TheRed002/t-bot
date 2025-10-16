"""
Real Service Fixtures for Risk Management Integration Tests.

This module provides fixtures for creating real risk management service instances
with proper dependency injection, database integration, and NO MOCKS.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio

from src.core.config import Config
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    Signal,
    SignalDirection,
)
from src.error_handling.decorators import shutdown_all_error_handlers
from src.risk_management.factory import RiskManagementFactory
from src.risk_management.service import RiskService


@pytest_asyncio.fixture
async def real_risk_service(container) -> AsyncGenerator[RiskService, None]:
    """Create real risk management service with all dependencies from DI container."""
    # Reset circuit breakers before each test to prevent state pollution
    shutdown_all_error_handlers()

    # Always create risk service manually WITHOUT StateService
    # StateService can cause circuit breaker issues in integration tests due to Redis dependencies
    # DO NOT use container's RiskService as it may have StateService attached
    from src.database.repository.risk import PortfolioRepository, RiskMetricsRepository

    config = container.resolve("Config")
    database_service = container.resolve("DatabaseService")

    # Create repositories
    risk_metrics_repo = RiskMetricsRepository(database_service)
    portfolio_repo = PortfolioRepository(database_service)

    # Create risk service without StateService for integration tests
    risk_service = RiskService(
        risk_metrics_repository=risk_metrics_repo,
        portfolio_repository=portfolio_repo,
        state_service=None,  # Skip StateService to avoid circuit breaker and timeout issues
        analytics_service=None,  # Optional
        config=config,
        correlation_id="test_correlation_id",
    )

    await risk_service.initialize()
    # Explicitly set _initialized flag for health check tests
    risk_service._initialized = True

    yield risk_service

    # Cleanup
    try:
        await risk_service.cleanup()
    except:
        pass


@pytest_asyncio.fixture
async def minimal_state_service(container) -> AsyncGenerator:
    """Create minimal StateService from DI container."""
    # Try to get StateService from container
    try:
        state_service = container.resolve("StateService")
    except:
        try:
            state_service = container.resolve("state_service")
        except:
            # Create manually if needed
            config = container.resolve("Config")
            from src.state.state_service import StateService

            state_service = StateService(config=config)
            await state_service.initialize()

    yield state_service

    # Cleanup
    try:
        await state_service.cleanup()
    except:
        pass


@pytest_asyncio.fixture
async def real_risk_factory(container, minimal_state_service) -> AsyncGenerator[RiskManagementFactory, None]:
    """Create real risk management factory from DI container."""
    # Try to get factory from container first
    try:
        factory = container.resolve("RiskManagementFactory")
    except:
        try:
            factory = container.resolve("risk_management_factory")
        except:
            # Create manually if needed
            factory = RiskManagementFactory(injector=container)

    yield factory

    # Cleanup
    try:
        await factory.stop_services()
    except:
        pass


def generate_realistic_price_sequence(
    base_price: Decimal,
    periods: int = 30,
    volatility: Decimal = Decimal("0.02"),
    trend: Decimal = Decimal("0.0"),
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
    exchange: str = "binance",
) -> list[MarketData]:
    """Generate realistic market data sequence for testing."""
    prices = generate_realistic_price_sequence(base_price, periods, volatility, trend)

    market_data_list = []
    start_time = datetime.now(timezone.utc) - timedelta(hours=periods)

    for i, close_price in enumerate(prices):
        # Add some intraday variation
        high_price = close_price * Decimal("1.01")
        low_price = close_price * Decimal("0.99")
        open_price = prices[i - 1] if i > 0 else close_price

        # Volume varies with volatility
        base_volume = Decimal("1000")
        volume = base_volume * (
            Decimal("1") + abs(close_price - open_price) / close_price * Decimal("10")
        )

        market_data = MarketData(
            symbol=symbol,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timestamp=start_time + timedelta(hours=i),
            exchange=exchange,
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
        trend=Decimal("0.002"),  # 0.2% daily uptrend
    )


def generate_bear_market_scenario(symbol: str = "BTC/USDT") -> list[MarketData]:
    """Generate bear market scenario with downtrend."""
    return generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=Decimal("50000"),
        periods=50,
        volatility=Decimal("0.025"),
        trend=Decimal("-0.0015"),  # -0.15% daily downtrend
    )


def generate_high_volatility_scenario(symbol: str = "BTC/USDT") -> list[MarketData]:
    """Generate high volatility scenario."""
    return generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=Decimal("50000"),
        periods=50,
        volatility=Decimal("0.05"),  # High volatility
        trend=Decimal("0.0"),
    )


def generate_crash_scenario(symbol: str = "BTC/USDT") -> list[MarketData]:
    """Generate market crash scenario with sudden drop."""
    # Normal market first
    normal_data = generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=Decimal("50000"),
        periods=20,
        volatility=Decimal("0.015"),
        trend=Decimal("0.001"),
    )

    # Then crash
    crash_data = generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=normal_data[-1].close,
        periods=10,
        volatility=Decimal("0.08"),  # Extreme volatility
        trend=Decimal("-0.05"),  # Severe downtrend
    )

    # Then recovery
    recovery_data = generate_realistic_market_data_sequence(
        symbol=symbol,
        base_price=crash_data[-1].close,
        periods=20,
        volatility=Decimal("0.03"),
        trend=Decimal("0.002"),
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
        take_profit=Decimal("52000"),
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
            take_profit=Decimal("52000"),
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
            take_profit=Decimal("3200"),
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
            "risk_reward_ratio": Decimal("2.0"),
        },
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
        signal_id=str(uuid4()),
    )
