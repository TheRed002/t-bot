"""
Common test fixtures and configuration for capital management tests.

This file contains shared fixtures and test configuration to optimize test performance.
"""

import logging
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Disable logging during tests for better performance
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("pytest").setLevel(logging.CRITICAL)

# Disable warnings for cleaner output
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Disable asyncio debug mode for performance
import asyncio
import os

# Set environment variables to speed up tests
os.environ["PYTHONASYNCIODEBUG"] = "0"
os.environ["PYTHONPATH"] = "/mnt/e/Work/P-41 Trading/code/t-bot"

# Configure asyncio for performance
if hasattr(asyncio, "set_event_loop_policy"):
    policy = asyncio.DefaultEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)


@pytest.fixture(scope="session")
def base_config():
    """Base configuration used across multiple test modules."""
    return {
        "total_capital": 1000.0,  # Very small amounts for fastest tests
        "emergency_reserve_pct": 0.1,
        "max_allocation_pct": 0.2,
        "rebalance_frequency_hours": 1,  # Very short for faster tests
        "min_allocation_pct": 0.05,
        "max_daily_reallocation_pct": 0.1,
        "hedging_enabled": False,  # Disable for faster tests
        "hedging_threshold": 0.2,
        "hedge_ratio": 0.5,
        "min_deposit_amount": 10.0,  # Very small for faster tests
        "min_withdrawal_amount": 1.0,  # Very small for faster tests
        "auto_compound_enabled": False,  # Disable for faster tests
        "fund_flow_cooldown_minutes": 0,  # No cooldown for faster tests
    }


@pytest.fixture(scope="session")
def test_amounts():
    """Common test amounts to ensure consistency."""
    return {
        "small": Decimal("10"),
        "medium": Decimal("50"),
        "large": Decimal("100"),
        "very_large": Decimal("500"),
    }


@pytest.fixture(scope="session")
def mock_exchanges():
    """Mock exchange instances for testing."""
    return {"binance": Mock(), "okx": Mock(), "coinbase": Mock()}


@pytest.fixture(scope="session")
def mock_supported_currencies():
    """Mock supported currencies list."""
    return ["USDT", "BTC", "ETH", "USD"]


@pytest.fixture(scope="module")
def optimized_mock_service():
    """Optimized mock service with common async methods."""
    service = Mock()
    service.start = AsyncMock()
    service.stop = AsyncMock()
    service.is_running = True
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    return service


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup optimized test environment."""
    # Mock time.sleep to make tests instant
    import time

    original_sleep = time.sleep
    time.sleep = Mock()

    # Mock datetime.now for consistency in tests

    with patch("time.sleep"):
        yield

    # Restore original functions
    time.sleep = original_sleep


def pytest_collection_modifyitems(config, items):
    """Modify test collection to optimize execution."""
    # All unit tests should run - no skipping based on performance
    pass
