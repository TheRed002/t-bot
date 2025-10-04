"""
Strategy Module Integration Test Configuration.

This module provides pytest configuration and fixtures specifically
for strategy integration tests using real service implementations.

Key Features:
- Real service fixtures with dependency injection
- Comprehensive test data generation
- Performance benchmarking utilities
- Database cleanup and isolation
- Error handling test helpers
"""

import asyncio
from decimal import Decimal

import pytest
import pytest_asyncio

# Import base fixtures from main conftest
from tests.conftest import clean_database

# Import core components
from src.core.config import Config
from src.core.types import StrategyConfig, StrategyType
from src.strategies.factory import StrategyFactory
from src.strategies.dependencies import create_strategy_service_container
from src.strategies.service import StrategyService

# Import strategy-specific fixtures
from tests.integration.modules.strategies.fixtures.market_data_generators import (
    MarketDataGenerator,
    create_test_market_data_suite,
)

from tests.integration.modules.strategies.helpers.indicator_validators import (
    IndicatorValidator,
    PerformanceBenchmarker,
    create_known_test_cases,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Real Service Fixtures - Shared across all strategy tests
# ============================================================================


@pytest_asyncio.fixture
async def real_config():
    """Create real configuration for testing."""
    # Use default config - no overrides needed for integration tests
    config = Config()
    return config


@pytest_asyncio.fixture
async def real_database_service_fixture(real_config, clean_database):
    """
    Create real DatabaseService for testing.

    Depends on clean_database to ensure database is clean before initialization.
    """
    from src.database.connection import DatabaseConnectionManager
    from src.database.service import DatabaseService

    # Create database connection manager
    connection_manager = DatabaseConnectionManager(real_config)
    await connection_manager.initialize()

    # Create and initialize database service
    db_service = DatabaseService(connection_manager=connection_manager)
    await db_service.initialize()

    yield db_service

    # Cleanup
    await db_service.cleanup()
    await connection_manager.close()


# Alias for compatibility with both naming conventions
@pytest_asyncio.fixture
async def real_database_service(real_database_service_fixture):
    """Alias for real_database_service_fixture."""
    return real_database_service_fixture


@pytest_asyncio.fixture
async def real_data_service(real_config, real_database_service_fixture):
    """Create real DataService with database integration."""
    from src.data.services.data_service import DataService

    data_service = DataService(
        config=real_config,
        database_service=real_database_service_fixture,
        cache_service=None,  # Optional
        metrics_collector=None,  # Optional
    )

    await data_service.initialize()
    yield data_service
    await data_service.cleanup()


@pytest_asyncio.fixture
async def real_risk_service(real_config, real_database_service_fixture):
    """Create real RiskService with database integration."""
    from src.database.repository.risk import PortfolioRepository, RiskMetricsRepository
    from src.risk_management.service import RiskService

    # Create repositories
    risk_metrics_repo = RiskMetricsRepository(real_database_service_fixture)
    portfolio_repo = PortfolioRepository(real_database_service_fixture)

    # Create risk service
    risk_service = RiskService(
        risk_metrics_repository=risk_metrics_repo,
        portfolio_repository=portfolio_repo,
        state_service=None,  # Optional
        analytics_service=None,  # Optional
        config=real_config,
        correlation_id="test_strategy_correlation",
    )

    await risk_service.initialize()
    yield risk_service
    await risk_service.cleanup()


@pytest_asyncio.fixture
async def strategy_service_container(
    real_config, real_database_service_fixture, real_data_service, real_risk_service
):
    """Create real strategy service container with all dependencies."""
    # Create container with real services
    container = create_strategy_service_container(
        risk_service=real_risk_service,
        data_service=real_data_service,
        execution_service=None,  # Not needed for most tests
        monitoring_service=None,  # Optional
        state_service=None,  # Optional
        capital_service=None,  # Optional
        ml_service=None,  # Optional
        analytics_service=None,  # Optional
        optimization_service=None,  # Optional
    )

    # Verify container was created with required services
    assert container is not None
    assert container.data_service is not None
    assert container.risk_service is not None

    yield container

    # No cleanup needed - container doesn't require explicit cleanup


@pytest_asyncio.fixture
async def real_strategy_service(strategy_service_container):
    """Create real StrategyService with injected dependencies."""
    # Extract services from container to pass to StrategyService
    strategy_service = StrategyService(
        name="RealStrategyService",
        risk_manager=strategy_service_container.risk_service,
        data_service=strategy_service_container.data_service,
    )

    # Start the service (this calls initialize internally)
    await strategy_service.start()
    yield strategy_service
    await strategy_service.stop()


@pytest_asyncio.fixture
async def strategy_factory(strategy_service_container):
    """Create StrategyFactory for creating strategy instances."""
    factory = StrategyFactory(
        service_container=strategy_service_container,
        risk_manager=strategy_service_container.risk_service,
        data_service=strategy_service_container.data_service,
    )
    yield factory


@pytest.fixture
def real_mean_reversion_config():
    """Create real mean reversion strategy configuration."""
    return StrategyConfig(
        strategy_id="real_mean_reversion_001",
        name="real_mean_reversion_test",
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
            # Required parameters for factory validation
            "mean_period": 20,
            "deviation_threshold": Decimal("2.0"),
            "reversion_strength": Decimal("0.5"),
            # Additional parameters for strategy implementation
            "lookback_period": 20,
            "entry_threshold": Decimal("2.0"),
            "exit_threshold": Decimal("0.5"),
            "volume_filter": True,
            "min_volume_ratio": Decimal("1.5"),
            "atr_period": 14,
            "atr_multiplier": Decimal("2.0"),
        },
    )


@pytest.fixture
def real_trend_following_config():
    """Create real trend following strategy configuration."""
    return StrategyConfig(
        strategy_id="real_trend_following_001",
        name="real_trend_following_test",
        strategy_type=StrategyType.TREND_FOLLOWING,
        enabled=True,
        symbol="BTC/USDT",
        timeframe="1h",
        min_confidence=Decimal("0.7"),
        max_positions=3,
        position_size_pct=Decimal("0.025"),
        stop_loss_pct=Decimal("0.015"),
        take_profit_pct=Decimal("0.045"),
        parameters={
            # Required parameters for factory validation
            "fast_ma_period": 10,
            "slow_ma_period": 30,
            "rsi_period": 14,
            "volume_confirmation": True,
        },
    )


@pytest.fixture
def real_breakout_config():
    """Create real breakout strategy configuration."""
    return StrategyConfig(
        strategy_id="real_breakout_001",
        name="real_breakout_test",
        strategy_type=StrategyType.BREAKOUT,
        enabled=True,
        symbol="BTC/USDT",
        timeframe="1h",
        min_confidence=Decimal("0.75"),
        max_positions=2,
        position_size_pct=Decimal("0.03"),
        stop_loss_pct=Decimal("0.02"),
        take_profit_pct=Decimal("0.06"),
        parameters={
            # Required parameters for factory validation
            "lookback_period": 20,
            "volume_confirmation": True,
            "min_volume_ratio": Decimal("1.5"),
        },
    )


@pytest_asyncio.fixture
async def real_mean_reversion_strategy(real_mean_reversion_config, strategy_factory):
    """Create real mean reversion strategy instance."""
    strategy = await strategy_factory.create_strategy(
        strategy_type=StrategyType.MEAN_REVERSION, config=real_mean_reversion_config
    )
    yield strategy
    strategy.cleanup()


@pytest_asyncio.fixture
async def real_trend_following_strategy(real_trend_following_config, strategy_factory):
    """Create real trend following strategy instance."""
    strategy = await strategy_factory.create_strategy(
        strategy_type=StrategyType.TREND_FOLLOWING, config=real_trend_following_config
    )
    yield strategy
    strategy.cleanup()


@pytest_asyncio.fixture
async def real_breakout_strategy(real_breakout_config, strategy_factory):
    """Create real breakout strategy instance."""
    strategy = await strategy_factory.create_strategy(
        strategy_type=StrategyType.BREAKOUT, config=real_breakout_config
    )
    yield strategy
    strategy.cleanup()


@pytest_asyncio.fixture
async def real_technical_indicators(real_data_service):
    """
    Create real technical indicators service using DataService.

    Note: DataService provides indicator methods like get_sma, get_ema, etc.
    This fixture wraps it to match the expected interface for tests.
    """
    # DataService itself provides technical indicator methods
    # We can return it directly or wrap it
    yield real_data_service


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def market_data_generator():
    """Provide a market data generator for tests."""
    return MarketDataGenerator(seed=42)  # Deterministic for reproducible tests


@pytest.fixture
def test_market_data_suite():
    """Provide a comprehensive suite of test market data scenarios."""
    return create_test_market_data_suite()


@pytest.fixture
def indicator_validator():
    """Provide an indicator validator for mathematical accuracy testing."""
    return IndicatorValidator()


@pytest.fixture
def performance_benchmarker():
    """Provide a performance benchmarker for strategy testing."""
    return PerformanceBenchmarker()


@pytest.fixture
def known_test_cases():
    """Provide known test cases with expected results for validation."""
    return create_known_test_cases()


@pytest_asyncio.fixture
async def strategy_test_environment(integrated_strategy_environment):
    """
    Complete test environment for strategy integration tests.

    Provides all necessary services and utilities for comprehensive testing:
    - Real services with dependency injection
    - Market data generators
    - Validation utilities
    - Performance benchmarking tools
    """
    environment = integrated_strategy_environment.copy()

    # Add additional test utilities
    environment.update({
        "market_data_generator": MarketDataGenerator(seed=42),
        "test_data_suite": create_test_market_data_suite(),
        "indicator_validator": IndicatorValidator(),
        "performance_benchmarker": PerformanceBenchmarker(),
        "known_test_cases": create_known_test_cases(),
    })

    yield environment

    # Cleanup is handled by the integrated_strategy_environment fixture


# Performance testing configuration
@pytest.fixture
def performance_test_config():
    """Configuration for performance testing."""
    return {
        "indicator_benchmark_iterations": 100,
        "signal_generation_iterations": 50,
        "database_operation_iterations": 20,
        "concurrent_strategy_count": 5,
        "performance_targets": {
            "rsi_calculation_ms": 10.0,
            "sma_calculation_ms": 5.0,
            "ema_calculation_ms": 5.0,
            "macd_calculation_ms": 15.0,
            "signal_generation_ms": 50.0,
            "database_operation_ms": 100.0,
            "memory_usage_mb": 50.0,
        },
        "tolerance_settings": {
            "rsi_tolerance": "0.01",  # 1%
            "sma_tolerance": "0.001",  # 0.1%
            "ema_tolerance": "0.01",  # 1%
            "macd_tolerance": "0.02",  # 2%
        },
    }


# Test data configuration
@pytest.fixture
def test_data_config():
    """Configuration for test data generation."""
    return {
        "default_periods": 100,
        "trend_strength": 0.001,  # 0.1% per period
        "volatility": 0.02,  # 2% volatility
        "base_price": "50000.00",
        "base_volume": "500.00",
        "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    }


# Strategy configuration templates
@pytest.fixture
def strategy_config_templates():
    """Provide strategy configuration templates for testing."""
    from src.core.types import StrategyType
    from decimal import Decimal

    return {
        "trend_following": {
            "strategy_type": StrategyType.TREND_FOLLOWING,
            "min_confidence": Decimal("0.7"),
            "max_positions": 3,
            "position_size_pct": Decimal("0.02"),
            "stop_loss_pct": Decimal("0.02"),
            "take_profit_pct": Decimal("0.04"),
            "parameters": {
                "fast_ma": 20,
                "slow_ma": 50,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "volume_confirmation": True,
            },
        },
        "mean_reversion": {
            "strategy_type": StrategyType.MEAN_REVERSION,
            "min_confidence": Decimal("0.6"),
            "max_positions": 2,
            "position_size_pct": Decimal("0.03"),
            "stop_loss_pct": Decimal("0.015"),
            "take_profit_pct": Decimal("0.03"),
            "parameters": {
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "rsi_period": 14,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "mean_reversion_threshold": Decimal("0.8"),
            },
        },
        "breakout": {
            "strategy_type": StrategyType.BREAKOUT,
            "min_confidence": Decimal("0.75"),
            "max_positions": 1,
            "position_size_pct": Decimal("0.04"),
            "stop_loss_pct": Decimal("0.025"),
            "take_profit_pct": Decimal("0.06"),
            "parameters": {
                "lookback_period": 50,
                "breakout_threshold": Decimal("0.02"),
                "volume_threshold": Decimal("1.5"),
                "momentum_confirmation": True,
            },
        },
    }


# Error testing utilities
@pytest.fixture
def error_test_scenarios():
    """Provide error scenarios for testing error handling."""
    from src.core.types import StrategyType, SignalDirection
    from src.core.types import Signal
    from decimal import Decimal
    from datetime import datetime, timezone

    return {
        "invalid_strategy_config": {
            "strategy_id": "invalid_test",
            "name": "InvalidTest",
            "strategy_type": StrategyType.CUSTOM,
            "symbol": "INVALID/PAIR",
            "timeframe": "invalid_timeframe",
            "enabled": True,
            "min_confidence": Decimal("1.5"),  # Invalid > 1.0
            "max_positions": -1,  # Invalid negative
            "position_size_pct": Decimal("2.0"),  # Invalid > 1.0
            "stop_loss_pct": Decimal("-0.1"),  # Invalid negative
            "take_profit_pct": Decimal("0"),  # Invalid zero
            "parameters": {},
        },
        "invalid_signal": Signal(
            direction=SignalDirection.BUY,
            confidence=Decimal("1.5"),  # Invalid > 1.0
            strength=Decimal("-0.5"),  # Invalid negative
            source="InvalidTest",
            timestamp=datetime.now(timezone.utc),
            symbol="INVALID/PAIR",
            strategy_name="InvalidTest",
            metadata={},
        ),
        "edge_case_market_data": {
            "zero_volume": True,
            "zero_price_change": True,
            "extreme_volatility": True,
            "missing_data_points": True,
        },
    }


# Async fixture helpers
@pytest_asyncio.fixture
async def async_test_cleanup():
    """Provide async cleanup utilities for tests."""
    cleanup_tasks = []

    def add_cleanup(coro):
        """Add a cleanup coroutine to be executed after test."""
        cleanup_tasks.append(coro)

    yield add_cleanup

    # Execute all cleanup tasks
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)


# Note: Database cleanup is handled by individual test classes using autouse fixtures
# See test_real_integration.py::TestRealStrategyFrameworkIntegration for example