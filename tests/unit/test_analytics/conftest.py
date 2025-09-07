"""
Shared fixtures for analytics tests.

Optimized fixtures to improve test performance and reduce redundancy.
"""

# Disable logging during tests for performance and set optimization flags
import logging
import os
import warnings
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "0"  # Deterministic hash for faster dict operations
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # Skip bytecode generation

# Fixed timestamp for consistent testing
FIXED_TIMESTAMP = datetime(2024, 1, 15, 12, 0, 0)


@pytest.fixture(scope="session")
def fixed_timestamp():
    """Fixed timestamp for consistent test results."""
    return FIXED_TIMESTAMP


# Pre-computed config for faster test execution
_cached_config = None


@pytest.fixture(scope="session")
def mock_analytics_config():
    """Mock analytics configuration for tests (cached)."""
    global _cached_config
    if _cached_config is None:
        from src.analytics.types import AnalyticsConfiguration

        _cached_config = AnalyticsConfiguration(
            enable_realtime=True,
            calculation_interval=60,
            risk_metrics_enabled=True,
            portfolio_analytics_enabled=True,
            attribution_enabled=True,
            factor_analysis_enabled=True,
            stress_testing_enabled=True,
            benchmark_comparison_enabled=True,
            alternative_metrics_enabled=True,
            var_confidence_level=Decimal("0.95"),
            max_drawdown_threshold=Decimal("0.20"),
            correlation_threshold=Decimal("0.80"),
            concentration_threshold=Decimal("0.25"),
        )
    return _cached_config


@pytest.fixture(scope="session")
def mock_metrics_collector():
    """Mock metrics collector for tests."""
    collector = Mock()
    collector.increment = Mock()
    collector.gauge = Mock()
    collector.histogram = Mock()
    collector.timer = Mock()
    return collector


@pytest.fixture(scope="session")
def mock_logger():
    """Mock logger to disable logging during tests."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def mock_uow():
    """Mock unit of work for database operations."""
    uow = Mock()
    uow.analytics_repository = AsyncMock()
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Create an async context manager mock
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)

    return uow


# Pytest configuration for better performance
def pytest_configure(config):
    """Configure pytest for optimal performance."""
    # Disable warnings and optimize Python runtime
    import sys
    import warnings

    warnings.filterwarnings("ignore")

    # Optimize Python performance for tests
    sys.dont_write_bytecode = True

    # Set up asyncio mode and test markers
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "slow: mark test as slow")

    # Enable fast test discovery
    config.option.tb = "short"
    config.option.maxfail = 1  # Stop on first failure for faster feedback
