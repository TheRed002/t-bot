"""
ML Module Integration Test Configuration.

Provides pytest fixtures for ML integration tests with full DI support.
"""

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import AsyncGenerator, Tuple


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


@pytest_asyncio.fixture
async def clean_database() -> AsyncGenerator:
    """
    Provide clean database for testing.

    Creates a clean database connection manager for ML tests.
    Uses the infrastructure clean_database fixture pattern.
    """
    from src.database.connection import DatabaseConnectionManager
    from src.core.config import get_config
    import uuid
    from sqlalchemy import text

    # Create unique test schema for isolation
    test_id = str(uuid.uuid4())[:8]
    test_schema = f"test_{test_id}"

    config = get_config()
    manager = DatabaseConnectionManager(config)

    # Configure for test schema
    manager._test_schema = test_schema
    manager.set_test_schema(test_schema)

    try:
        await manager.initialize()

        # Create test schema
        async with manager.get_async_session() as session:
            await session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {test_schema}"))
            await session.commit()

        # Set search path
        async with manager.get_async_session() as session:
            await session.execute(text(f"SET search_path TO {test_schema}, public"))
            await session.commit()

        # Create tables in test schema
        from src.database.models.base import Base
        from src.database.models import (  # noqa: F401
            analytics, audit, backtesting, bot, bot_instance, capital,
            data, exchange, market_data, ml, optimization, risk,
            state, system, trading, user
        )

        if manager.async_engine:
            async with manager.async_engine.begin() as conn:
                await conn.execute(text(f"SET search_path TO {test_schema}, public"))
                await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=False))

        yield manager

    finally:
        # Cleanup: drop test schema
        try:
            async with manager.get_async_session() as session:
                await session.execute(text(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"))
                await session.commit()
        except Exception:
            pass  # Ignore cleanup errors

        await manager.close()


@pytest.fixture
def generate_market_data():
    """
    Factory fixture for generating realistic market data.

    Returns a function that generates market data with specified parameters.
    """
    def _generate(
        symbol: str = 'BTC/USD',
        n_points: int = 100,
        start_price: float = 50000.0,
        volatility: float = 0.02,
        trend: float = 0.0001,
        start_date: datetime | None = None
    ) -> list[dict]:
        """
        Generate realistic market data.

        Args:
            symbol: Trading symbol
            n_points: Number of data points
            start_price: Starting price
            volatility: Price volatility (standard deviation)
            trend: Price trend (drift per period)
            start_date: Starting timestamp

        Returns:
            List of market data dictionaries
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(hours=n_points)

        np.random.seed(42)

        # Generate price path with trend and volatility
        returns = np.random.normal(trend, volatility, n_points)
        prices = start_price * np.exp(returns.cumsum())

        # Generate OHLCV data
        data = []
        for i in range(n_points):
            timestamp = start_date + timedelta(hours=i)
            close_price = Decimal(str(prices[i]))

            # Generate realistic OHLC around close
            volatility_range = close_price * Decimal(str(volatility * 2))
            open_price = close_price + Decimal(str(np.random.uniform(-1, 1))) * volatility_range
            high_price = max(open_price, close_price) + Decimal(str(abs(np.random.uniform(0, 1)))) * volatility_range
            low_price = min(open_price, close_price) - Decimal(str(abs(np.random.uniform(0, 1)))) * volatility_range
            volume = Decimal(str(np.random.uniform(50, 200)))

            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        return data

    return _generate


@pytest.fixture
def generate_training_data():
    """
    Factory fixture for generating ML training data.

    Returns a function that generates features and targets for model training.
    """
    def _generate(
        n_samples: int = 200,
        n_features: int = 5,
        task_type: str = 'classification',
        noise: float = 0.1,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate training data for ML models.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            task_type: 'classification' or 'regression'
            noise: Noise level in data
            seed: Random seed for reproducibility

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        np.random.seed(seed)

        # Generate features
        feature_data = {}
        for i in range(n_features):
            feature_data[f'feature_{i}'] = np.random.randn(n_samples)

        features = pd.DataFrame(feature_data)

        # Generate target based on task type
        if task_type == 'classification':
            # Binary classification based on linear combination of features
            linear_combination = sum(
                features[f'feature_{i}'] * (0.5 + i * 0.1)
                for i in range(min(3, n_features))
            )
            target = (linear_combination + np.random.randn(n_samples) * noise > 0).astype(int)
        else:  # regression
            # Regression target with some non-linear relationships
            target = (
                features['feature_0'] * 2.0 +
                features['feature_1'] ** 2 +
                np.random.randn(n_samples) * noise
            )

        return features, pd.Series(target, name='target')

    return _generate
