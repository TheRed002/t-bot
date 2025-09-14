"""
Pytest fixtures for strategy tests.

This module provides common fixtures for all strategy test modules including
mock services, sample data, and helper utilities.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
import logging

# Disable logging during tests for better performance
logging.disable(logging.CRITICAL)

from src.core.types import (
    MarketData,
    MarketRegime,
    Position,
    PositionSide,
    PositionStatus,
    StrategyConfig,
    StrategyType,
)
from src.data.features.technical_indicators import TechnicalIndicators
from src.risk_management.adaptive_risk import AdaptiveRiskManager
from src.risk_management.regime_detection import MarketRegimeDetector
from src.strategies.service import StrategyService


@pytest.fixture(scope="session")
def mock_technical_indicators():
    """Create mock technical indicators service."""
    mock_indicators = Mock(spec=TechnicalIndicators)
    
    # Pre-computed return values for performance
    atr_value = Decimal("1000.0")
    volatility_value = Decimal("0.05")
    bollinger_result = {
        "upper": Decimal("51500.0"),
        "middle": Decimal("50000.0"),
        "lower": Decimal("48500.0"),
        "bandwidth": Decimal("0.06"),
        "squeeze": False
    }
    consolidation_result = {
        "consolidation_score": Decimal("0.7"),
        "price_range_ratio": Decimal("0.05"),
        "volume_stability": Decimal("0.8"),
        "bollinger_squeeze": False
    }
    rsi_value = Decimal("65.0")
    macd_result = {
        "macd": Decimal("150.0"),
        "signal": Decimal("120.0"),
        "histogram": Decimal("30.0")
    }
    volume_ratio_value = Decimal("1.5")
    
    # Use AsyncMock for async methods that are awaited in the strategy code
    mock_indicators.calculate_atr = AsyncMock(return_value=atr_value)
    mock_indicators.calculate_volatility = AsyncMock(return_value=volatility_value)
    mock_indicators.calculate_bollinger_bands = AsyncMock(return_value=bollinger_result)
    mock_indicators.calculate_consolidation_score = AsyncMock(return_value=consolidation_result)
    mock_indicators.calculate_rsi = AsyncMock(return_value=rsi_value)
    mock_indicators.calculate_macd = AsyncMock(return_value=macd_result)
    mock_indicators.calculate_volume_ratio = AsyncMock(return_value=volume_ratio_value)
    
    return mock_indicators


@pytest.fixture(scope="session")
def mock_regime_detector():
    """Create mock regime detector."""
    mock_detector = Mock(spec=MarketRegimeDetector)
    # Use synchronous mock for better performance
    mock_detector.detect_regime = Mock(return_value=MarketRegime.TRENDING_UP)
    mock_detector.get_regime_confidence = Mock(return_value=Decimal("0.8"))
    return mock_detector


@pytest.fixture(scope="session")
def mock_adaptive_risk_manager():
    """Create mock adaptive risk manager."""
    mock_manager = Mock(spec=AdaptiveRiskManager)
    mock_manager.get_dynamic_position_size = Mock(return_value=Decimal("0.02"))
    mock_manager.get_regime_risk_adjustment = Mock(return_value=Decimal("1.0"))
    return mock_manager


@pytest.fixture(scope="session")
def mock_strategy_service():
    """Create mock strategy service."""
    mock_service = Mock(spec=StrategyService)
    # Pre-computed return values for performance
    empty_state = {}
    mock_service.get_strategy_state = Mock(return_value=empty_state)
    mock_service.persist_strategy_state = Mock(return_value=True)
    mock_service.validate_strategy_config = Mock(return_value=True)
    mock_service.register_strategy = Mock(return_value=True)
    mock_service.start_strategy = Mock(return_value=True)
    mock_service.stop_strategy = Mock(return_value=True)
    return mock_service


@pytest.fixture(scope="session")
def sample_market_data():
    """Create sample market data for testing."""
    # Use fixed timestamp for performance
    fixed_timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return MarketData(
        symbol="BTC/USD",
        timestamp=fixed_timestamp,
        open=Decimal("49500.0"),
        high=Decimal("51000.0"),
        low=Decimal("49000.0"),
        close=Decimal("50000.0"),
        volume=Decimal("1000.0"),
        exchange="binance",
        vwap=Decimal("50200.0"),
        trades_count=500,
        quote_volume=Decimal("50000000.0"),
        bid_price=Decimal("49999.0"),
        ask_price=Decimal("50001.0")
    )


@pytest.fixture(scope="session")
def sample_position():
    """Create sample position for testing."""
    # Use fixed timestamp for performance
    fixed_timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return Position(
        symbol="BTC/USD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),  # Changed from size to quantity
        entry_price=Decimal("49000.0"),
        current_price=Decimal("50000.0"),
        unrealized_pnl=Decimal("1000.0"),
        realized_pnl=Decimal("0.0"),
        opened_at=fixed_timestamp,  # Changed from timestamp to opened_at
        status=PositionStatus.OPEN,
        exchange="binance"
        # Removed position_id and strategy_id as they're not in the Position model
    )




@pytest.fixture(scope="session")
def strategy_config_dict():
    """Create strategy configuration dictionary."""
    return {
        "name": "test_strategy",
        "strategy_id": "test_strategy_001",
        "strategy_type": StrategyType.MOMENTUM,
        "symbol": "BTC/USD",
        "timeframe": "1h",
        "position_size_pct": 0.02,
        "min_confidence": 0.6,
        "max_positions": 5,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.05,
        "parameters": {
            "atr_period": 14,
            "breakout_multiplier": 2.0,
            "consolidation_period": 20,
            "volume_confirmation": True,
            "min_consolidation_ratio": 0.8,
            "max_consolidation_ratio": 1.2,
            "time_decay_factor": 0.95,
        },
        "risk_parameters": {
            "max_drawdown": 0.1,
        },
    }


@pytest.fixture(scope="session")
def strategy_config(strategy_config_dict):
    """Create strategy configuration object."""
    return StrategyConfig(**strategy_config_dict)


@pytest.fixture(scope="session") 
def mock_exchange():
    """Create mock exchange - cached for session scope."""
    exchange = Mock()
    exchange.place_order = AsyncMock(return_value={"id": "test_order_123", "status": "filled"})
    exchange.cancel_order = AsyncMock(return_value=True)
    exchange.get_balance = AsyncMock(return_value={"BTC": Decimal("1.0"), "USD": Decimal("50000.0")})
    exchange.fetch_ticker = AsyncMock(return_value={
        "symbol": "BTC/USD",
        "bid": Decimal("49999.0"),
        "ask": Decimal("50001.0"),
        "last": Decimal("50000.0")
    })
    return exchange


@pytest.fixture(scope="session")
def mock_risk_manager():
    """Create mock risk manager - cached for session scope."""
    risk_manager = Mock()
    risk_manager.validate_signal = AsyncMock(return_value=True)
    risk_manager.calculate_position_size = Mock(return_value=Decimal("0.02"))
    risk_manager.check_risk_limits = AsyncMock(return_value=True)
    return risk_manager


@pytest.fixture(scope="session")
def mock_portfolio_manager():
    """Create mock portfolio manager - cached for session scope."""
    portfolio = Mock()
    portfolio.get_available_capital = Mock(return_value=Decimal("50000.0"))
    portfolio.get_current_positions = Mock(return_value=[])
    portfolio.calculate_portfolio_value = Mock(return_value=Decimal("100000.0"))
    return portfolio


@pytest.fixture(scope="session")
def fast_market_data_batch():
    """Create batch of market data for performance testing."""
    fixed_timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    base_data = {
        "symbol": "BTC/USD",
        "timestamp": fixed_timestamp,
        "open": Decimal("49500.0"),
        "high": Decimal("51000.0"),
        "low": Decimal("49000.0"),
        "close": Decimal("50000.0"),
        "volume": Decimal("1000.0"),
        "exchange": "binance",
        "vwap": Decimal("50200.0"),
        "trades_count": 500,
        "quote_volume": Decimal("50000000.0"),
        "bid_price": Decimal("49999.0"),
        "ask_price": Decimal("50001.0")
    }
    
    # Pre-create smaller batch (5 items) for better memory usage
    return [MarketData(**base_data) for _ in range(5)]


@pytest.fixture(scope="session")
def mock_async_database():
    """Mock async database operations for performance."""
    db = Mock()
    # Use sync mocks for even better performance
    db.save = Mock(return_value=True)
    db.load = Mock(return_value={})
    db.update = Mock(return_value=True) 
    db.delete = Mock(return_value=True)
    db.query = Mock(return_value=[])
    return db


@pytest.fixture(scope="session", autouse=True)
def no_sleep():
    """Mock sleep functions to eliminate delays - auto-applied to all tests."""
    with patch("time.sleep"), patch("asyncio.sleep", new_callable=AsyncMock), \
         patch("numpy.random.seed", return_value=None), \
         patch("random.seed", return_value=None):
        yield