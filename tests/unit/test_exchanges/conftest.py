"""
Configuration for exchanges module tests - PERFORMANCE OPTIMIZED.

Basic pytest configuration with essential mocking for speed.
All tests should work properly with mocking and run in <1s per test.
"""

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


# Global mock instances (module-scoped for better performance)
_global_mock_time = time.time()
_global_mock_datetime = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def mock_time_fixed():
    """Session-scoped fixed time for consistent tests."""
    return _global_mock_time


@pytest.fixture(scope="session") 
def mock_datetime_fixed():
    """Session-scoped fixed datetime for consistent tests."""
    return _global_mock_datetime


@pytest.fixture(scope="session")
def mock_exchange_config():
    """Session-scoped standard mock exchange configuration."""
    return {
        "api_key": "test_key",
        "api_secret": "test_secret", 
        "sandbox": True,
        "testnet": True,
        "rate_limit": {"requests_per_second": 1000},
        "timeout": 1,
        "max_retries": 1,
    }


@pytest.fixture(scope="function")
def mock_exchange_base():
    """Base mock exchange for testing."""
    exchange = MagicMock()
    exchange.exchange_name = "mock_exchange"
    exchange.connected = False
    exchange.connect = AsyncMock(return_value=True)
    exchange.disconnect = AsyncMock()
    exchange.is_healthy = AsyncMock(return_value=True)
    exchange.ping = AsyncMock(return_value=True)
    exchange.get_ticker = AsyncMock(return_value=_create_mock_ticker())
    exchange.get_order_book = AsyncMock(return_value=_create_mock_order_book())
    exchange.get_account_balance = AsyncMock(return_value=_create_mock_balance())
    return exchange


@pytest.fixture(scope="function")
def fast_retry_config():
    """Fast retry configuration for testing advanced rate limiter."""
    config = Mock()
    config.retry_attempts = 1  # Reduced from 3
    config.retry_delay = 0.001  # Reduced from 0.1
    config.backoff_factor = 1.0  # Reduced from 1.5 
    return config


@pytest.fixture(scope="session")
def mock_config():
    """Session-scoped mock configuration."""
    config = Mock()
    config.exchange_service = Mock()
    config.exchange_service.default_timeout_seconds = 1
    config.exchange_service.max_retries = 1
    config.exchange_service.health_check_interval_seconds = 1
    return config


# Critical performance-optimized auto-use fixtures
@pytest.fixture(autouse=True)
async def mock_sleep_in_decorators():
    """Automatically mock asyncio.sleep in decorators to prevent delays in tests."""
    # Create a simple async mock that returns immediately
    mock_sleep = AsyncMock(return_value=None)
    
    with patch('src.utils.decorators.asyncio.sleep', mock_sleep):
        yield


@pytest.fixture(autouse=True)
def mock_retry_delays():
    """Mock retry delays to speed up tests."""
    with patch('src.utils.decorators.ExceptionCategory.get_retry_delay', return_value=0.001):
        yield


@pytest.fixture(autouse=True) 
def mock_timing():
    """Mock time operations to speed up tests."""
    with patch('asyncio.sleep', new_callable=lambda: AsyncMock(return_value=None)), \
         patch('time.sleep', return_value=None), \
         patch('time.time', return_value=_global_mock_time):
        yield


# Helper functions for creating mock data
def _create_mock_ticker():
    """Create mock ticker data."""
    from src.core.types import Ticker
    return Ticker(
        symbol="BTCUSDT",
        last_price=Decimal("50000.00"),
        bid_price=Decimal("49999.00"),
        ask_price=Decimal("50001.00"),
        volume=Decimal("1000.00"),
        timestamp=_global_mock_datetime
    )


def _create_mock_order_book():
    """Create mock order book data."""
    from src.core.types import OrderBook, OrderBookLevel
    return OrderBook(
        symbol="BTCUSDT",
        bids=[OrderBookLevel(price=Decimal("49999.00"), quantity=Decimal("1.00"))],
        asks=[OrderBookLevel(price=Decimal("50001.00"), quantity=Decimal("1.00"))],
        timestamp=_global_mock_datetime
    )


def _create_mock_balance():
    """Create mock balance data.""" 
    return {
        "BTC": Decimal("1.00"),
        "USDT": Decimal("10000.00"),
    }


def _create_mock_exchange_info():
    """Create mock Binance exchange info."""
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING", 
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "filters": []
            }
        ]
    }


def _create_mock_binance_ticker():
    """Create mock Binance ticker response."""
    return {
        "symbol": "BTCUSDT",
        "lastPrice": "50000.00",
        "bidPrice": "49999.00", 
        "askPrice": "50001.00",
        "volume": "1000.00"
    }


def _create_mock_binance_order_book():
    """Create mock Binance order book response."""
    return {
        "bids": [["49999.00", "1.00"]],
        "asks": [["50001.00", "1.00"]]
    }


def _create_mock_binance_account():
    """Create mock Binance account response."""
    return {
        "balances": [
            {"asset": "BTC", "free": "1.00", "locked": "0.00"},
            {"asset": "USDT", "free": "10000.00", "locked": "0.00"}
        ]
    }


def _create_mock_binance_order():
    """Create mock Binance order response."""
    return {
        "symbol": "BTCUSDT",
        "orderId": 12345,
        "status": "FILLED",
        "executedQty": "0.1",
        "price": "50000.00"
    }


def _create_mock_binance_cancel():
    """Create mock Binance cancel response."""
    return {
        "symbol": "BTCUSDT", 
        "orderId": 12345,
        "status": "CANCELED"
    }


def _create_mock_coinbase_product():
    """Create mock Coinbase product response."""
    return {
        "id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "status": "online"
    }


def _create_mock_coinbase_ticker():
    """Create mock Coinbase ticker response."""
    return {
        "price": "50000.00",
        "size": "1.00",
        "bid": "49999.00",
        "ask": "50001.00"
    }


def _create_mock_coinbase_order_book():
    """Create mock Coinbase order book response."""
    return {
        "bids": [["49999.00", "1.00", 1]],
        "asks": [["50001.00", "1.00", 1]]
    }


def _create_mock_okx_instrument():
    """Create mock OKX instrument response."""
    return {
        "instId": "BTC-USDT",
        "baseCcy": "BTC", 
        "quoteCcy": "USDT",
        "state": "live"
    }


def _create_mock_okx_ticker():
    """Create mock OKX ticker response."""
    return {
        "instId": "BTC-USDT",
        "last": "50000.00",
        "bidPx": "49999.00",
        "askPx": "50001.00", 
        "vol24h": "1000.00"
    }
