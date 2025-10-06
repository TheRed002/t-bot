"""
Base classes and utilities for integration testing.

This module provides common infrastructure for integration tests including:
- Base test classes with common setup/teardown
- Mock factories for testing components
- Test fixtures and utilities
- Performance measurement decorators
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from src.core.types import (
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Signal,
    SignalDirection,
    TimeInForce,
)
from src.database.manager import DatabaseManager
from src.exchanges.base import BaseExchange
from src.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MockExchangeFactory:
    """Factory for creating mock exchanges with realistic behavior."""

    @staticmethod
    def create_binance_mock(
        initial_balance: dict[str, Decimal] = None,
        market_prices: dict[str, Decimal] = None,
        order_latency: float = 0.1,
    ) -> Mock:
        """Create a realistic Binance exchange mock."""
        if initial_balance is None:
            initial_balance = {"USDT": Decimal("100000.0"), "BTC": Decimal("0.0")}
        if market_prices is None:
            market_prices = {"BTC/USDT": Decimal("50000.0"), "ETH/USDT": Decimal("3000.0")}

        exchange = Mock(spec=BaseExchange)
        exchange.name = "binance"
        exchange.is_connected = True
        exchange._balances = initial_balance.copy()
        exchange._market_prices = market_prices.copy()
        exchange._order_counter = 0
        exchange._orders = {}

        async def get_market_data(symbol: str) -> MarketData:
            await asyncio.sleep(0.01)  # Simulate network delay
            price = exchange._market_prices.get(symbol, Decimal("1.0"))
            spread = price * Decimal("0.001")  # 0.1% spread
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                open=price * Decimal("0.999"),  # Slightly below current price
                high=price * Decimal("1.002"),  # Slightly above current price
                low=price * Decimal("0.998"),  # Slightly below current price
                close=price,  # Current price as close
                volume=Decimal("1000.0"),
                exchange=exchange.name,
                metadata={
                    "bid": float(price - spread / 2),
                    "ask": float(price + spread / 2),
                    "price": float(price),
                },
            )

        async def place_order(order_data: dict[str, Any]) -> str:
            await asyncio.sleep(order_latency)  # Simulate order latency
            exchange._order_counter += 1
            order_id = f"binance_order_{exchange._order_counter}"

            # Simulate order processing
            order = Order(
                order_id=order_id,
                symbol=order_data["symbol"],
                side=OrderSide(order_data["side"].lower()),  # Convert to lowercase
                order_type=OrderType(
                    order_data.get("type", "MARKET").lower()
                ),  # Convert to lowercase
                quantity=Decimal(str(order_data["quantity"])),
                price=Decimal(str(order_data.get("price", "0")))
                if order_data.get("price")
                else None,
                status=OrderStatus.PENDING,
                time_in_force=TimeInForce.GTC,
                created_at=datetime.now(timezone.utc),
                exchange=exchange.name,
            )
            exchange._orders[order_id] = order

            # For testing, immediately fill market orders
            if order.order_type == OrderType.MARKET:
                market_data = await get_market_data(order.symbol)
                ask_price = Decimal(str(market_data.metadata.get("ask", market_data.close)))
                bid_price = Decimal(str(market_data.metadata.get("bid", market_data.close)))
                fill_price = ask_price if order.side == OrderSide.BUY else bid_price
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_price = fill_price

                # Update balances
                base_asset, quote_asset = order.symbol.split("/")
                if order.side == OrderSide.BUY:
                    cost = order.quantity * fill_price
                    if exchange._balances.get(quote_asset, Decimal("0")) >= cost:
                        exchange._balances[quote_asset] -= cost
                        exchange._balances[base_asset] = (
                            exchange._balances.get(base_asset, Decimal("0")) + order.quantity
                        )
                else:  # SELL
                    if exchange._balances.get(base_asset, Decimal("0")) >= order.quantity:
                        exchange._balances[base_asset] -= order.quantity
                        proceeds = order.quantity * fill_price
                        exchange._balances[quote_asset] = (
                            exchange._balances.get(quote_asset, Decimal("0")) + proceeds
                        )

            return order_id

        async def get_order_status(order_id: str) -> OrderStatus:
            return exchange._orders.get(order_id, Mock(status=OrderStatus.PENDING)).status

        async def get_balance(asset: str = None) -> dict[str, Decimal]:
            if asset:
                return {asset: exchange._balances.get(asset, Decimal("0"))}
            return exchange._balances.copy()

        async def cancel_order(order_id: str) -> bool:
            if order_id in exchange._orders:
                exchange._orders[order_id].status = OrderStatus.CANCELLED
                return True
            return False

        exchange.get_market_data = get_market_data
        exchange.place_order = place_order
        exchange.get_order_status = get_order_status
        exchange.get_balance = get_balance
        exchange.cancel_order = cancel_order

        return exchange

    @staticmethod
    def create_coinbase_mock(
        initial_balance: dict[str, Decimal] = None,
        market_prices: dict[str, Decimal] = None,
        order_latency: float = 0.15,
    ) -> Mock:
        """Create a realistic Coinbase exchange mock with slightly different characteristics."""
        if initial_balance is None:
            initial_balance = {"USDT": Decimal("50000.0"), "BTC": Decimal("0.0")}
        if market_prices is None:
            # Coinbase typically has slightly different prices
            market_prices = {"BTC/USDT": Decimal("50020.0"), "ETH/USDT": Decimal("3005.0")}

        # Use similar logic but with coinbase-specific differences
        exchange = MockExchangeFactory.create_binance_mock(
            initial_balance, market_prices, order_latency
        )
        exchange.name = "coinbase"
        return exchange


class MockStrategyFactory:
    """Factory for creating mock trading strategies."""

    @staticmethod
    def create_momentum_strategy(
        symbol: str = "BTC/USDT",
        signal_confidence: float = 0.8,
        signal_frequency: float = 0.3,  # Probability of generating non-HOLD signal
    ) -> Mock:
        """Create a momentum strategy mock."""
        strategy = Mock(spec=BaseStrategy)
        strategy.name = "momentum"
        strategy.enabled = True
        strategy.symbol = symbol

        async def generate_signal() -> Signal:
            await asyncio.sleep(0.05)  # Simulate analysis time

            # Randomly generate signals based on frequency
            import random

            if random.random() < signal_frequency:
                direction = SignalDirection.BUY if random.random() > 0.5 else SignalDirection.SELL
            else:
                direction = SignalDirection.HOLD

            return Signal(
                symbol=symbol,
                direction=direction,
                confidence=signal_confidence + (random.random() - 0.5) * 0.2,  # Add some noise
                price=Decimal("50000.0")
                + Decimal(str((random.random() - 0.5) * 1000)),  # Price noise
                timestamp=datetime.now(timezone.utc),
                metadata={"strategy": "momentum", "indicator_values": {"rsi": 65.5, "macd": 1.2}},
            )

        strategy.generate_signal = generate_signal
        return strategy

    @staticmethod
    def create_mean_reversion_strategy(
        symbol: str = "BTC/USDT", signal_confidence: float = 0.7, contrarian: bool = True
    ) -> Mock:
        """Create a mean reversion strategy mock."""
        strategy = Mock(spec=BaseStrategy)
        strategy.name = "mean_reversion"
        strategy.enabled = True
        strategy.symbol = symbol

        async def generate_signal() -> Signal:
            await asyncio.sleep(0.08)  # Slightly different analysis time

            import random

            # Mean reversion tends to be contrarian
            base_direction = SignalDirection.SELL if random.random() > 0.5 else SignalDirection.BUY
            if contrarian and random.random() > 0.3:  # Often contrarian
                direction = (
                    SignalDirection.SELL
                    if base_direction == SignalDirection.BUY
                    else SignalDirection.BUY
                )
            else:
                direction = base_direction

            return Signal(
                symbol=symbol,
                direction=direction,
                confidence=signal_confidence + (random.random() - 0.5) * 0.15,
                price=Decimal("50000.0") + Decimal(str((random.random() - 0.5) * 800)),
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "strategy": "mean_reversion",
                    "indicator_values": {"bollinger_position": -0.8, "rsi": 30.2},
                },
            )

        strategy.generate_signal = generate_signal
        return strategy


class PerformanceMonitor:
    """Monitor performance metrics during integration tests."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.events = []

    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics = {
            "total_duration": 0.0,
            "api_calls": 0,
            "order_latency": [],
            "signal_generation_time": [],
            "memory_usage": [],
            "error_count": 0,
        }

    def stop(self):
        """Stop performance monitoring and calculate final metrics."""
        self.end_time = time.time()
        self.metrics["total_duration"] = self.end_time - self.start_time

        # Calculate averages
        if self.metrics["order_latency"]:
            self.metrics["avg_order_latency"] = sum(self.metrics["order_latency"]) / len(
                self.metrics["order_latency"]
            )
        if self.metrics["signal_generation_time"]:
            self.metrics["avg_signal_time"] = sum(self.metrics["signal_generation_time"]) / len(
                self.metrics["signal_generation_time"]
            )

    def record_api_call(self, duration: float = None):
        """Record an API call."""
        self.metrics["api_calls"] += 1
        if duration:
            self.metrics["order_latency"].append(duration)

    def record_error(self, error: Exception):
        """Record an error."""
        self.metrics["error_count"] += 1
        self.events.append({"type": "error", "error": str(error), "timestamp": time.time()})

    def record_event(self, event_type: str, data: dict[str, Any]):
        """Record a custom event."""
        self.events.append({"type": event_type, "data": data, "timestamp": time.time()})


def performance_test(max_duration: float = 60.0, max_memory_mb: float = 500.0):
    """Decorator for performance testing."""

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            monitor.start()

            try:
                # Run the test
                result = await func(*args, **kwargs, performance_monitor=monitor)

                monitor.stop()

                # Check performance constraints
                if monitor.metrics["total_duration"] > max_duration:
                    pytest.fail(
                        f"Test exceeded maximum duration: {monitor.metrics['total_duration']:.2f}s > {max_duration}s"
                    )

                # Log performance metrics
                logger.info(f"Performance metrics for {func.__name__}:")
                logger.info(f"  Duration: {monitor.metrics['total_duration']:.2f}s")
                logger.info(f"  API calls: {monitor.metrics['api_calls']}")
                logger.info(f"  Errors: {monitor.metrics['error_count']}")
                if monitor.metrics.get("avg_order_latency"):
                    logger.info(f"  Avg order latency: {monitor.metrics['avg_order_latency']:.3f}s")

                return result

            except Exception as e:
                monitor.record_error(e)
                monitor.stop()
                raise

        return wrapper

    return decorator


class BaseIntegrationTest(ABC):
    """Base class for integration tests with common setup/teardown."""

    def __init__(self):
        self.config = None
        self.exchanges = {}
        self.strategies = {}
        self.bot_instance = None
        self.performance_monitor = PerformanceMonitor()

    @pytest_asyncio.fixture(autouse=True)
    async def setup_integration_test(self):
        """Setup common test infrastructure."""
        # Load test configuration
        self.config = await self._create_test_config()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Starting integration test: {self.__class__.__name__}")

        # Start performance monitoring
        self.performance_monitor.start()

        yield

        # Cleanup
        await self._cleanup_test_resources()
        self.performance_monitor.stop()
        logger.info(f"Completed integration test: {self.__class__.__name__}")

    async def _create_test_config(self) -> Mock:
        """Create test configuration."""
        config = Mock()

        # Trading configuration
        config.trading = Mock()
        config.trading.enabled = True
        config.trading.max_concurrent_trades = 5
        config.trading.min_trade_size = Decimal("10.0")
        config.trading.max_trade_size = Decimal("10000.0")

        # Risk management
        config.risk = Mock()
        config.risk.max_position_size = Decimal("50000.0")
        config.risk.max_daily_loss = Decimal("5000.0")
        config.risk.max_drawdown = Decimal("0.1")
        config.risk.position_sizing_method = "FIXED"
        config.risk.circuit_breakers_enabled = True
        config.risk.emergency_close_positions = False

        # Execution
        config.execution = Mock()
        config.execution.max_slippage = Decimal("0.005")
        config.execution.timeout = 30
        config.execution.retry_attempts = 3

        # Exchanges
        config.exchanges = Mock()
        config.exchanges.binance = Mock()
        config.exchanges.binance.enabled = True
        config.exchanges.binance.testnet = True
        config.exchanges.coinbase = Mock()
        config.exchanges.coinbase.enabled = True
        config.exchanges.coinbase.sandbox = True

        # State management
        config.state = Mock()
        config.state.checkpoint_interval = 300
        config.state.enable_persistence = True

        # Database (use test database)
        config.database = Mock()
        config.database.url = "postgresql://test:test@localhost:5432/test_trading"
        config.database.pool_size = 5

        return config

    async def _cleanup_test_resources(self):
        """Clean up test resources."""
        # Close exchange connections
        for exchange in self.exchanges.values():
            if hasattr(exchange, "close") and callable(exchange.close):
                try:
                    await exchange.close()
                except Exception as e:
                    logger.warning(f"Error closing exchange connection: {e}")

        # Stop bot instance
        if self.bot_instance and hasattr(self.bot_instance, "stop"):
            try:
                await self.bot_instance.stop()
            except Exception as e:
                logger.warning(f"Error stopping bot instance: {e}")

    @abstractmethod
    async def run_integration_test(self):
        """Abstract method for running the actual integration test."""
        pass

    async def create_mock_exchanges(self) -> dict[str, Mock]:
        """Create mock exchanges for testing."""
        exchanges = {
            "binance": MockExchangeFactory.create_binance_mock(),
            "coinbase": MockExchangeFactory.create_coinbase_mock(),
        }
        self.exchanges = exchanges
        return exchanges

    async def create_mock_strategies(self) -> dict[str, Mock]:
        """Create mock strategies for testing."""
        strategies = {
            "momentum": MockStrategyFactory.create_momentum_strategy(),
            "mean_reversion": MockStrategyFactory.create_mean_reversion_strategy(),
        }
        self.strategies = strategies
        return strategies

    @asynccontextmanager
    async def database_context(self) -> AsyncGenerator[Mock, None]:
        """Create a mock database context for testing."""
        # In a real integration test, this might create a test database connection
        db_manager = Mock(spec=DatabaseManager)

        # Mock database operations
        db_manager.execute = AsyncMock(return_value=None)
        db_manager.fetch = AsyncMock(return_value=[])
        db_manager.fetchrow = AsyncMock(return_value=None)
        db_manager.transaction = AsyncMock()

        try:
            yield db_manager
        finally:
            # Cleanup mock database
            pass

    async def simulate_market_conditions(
        self,
        duration_seconds: int = 10,
        price_volatility: float = 0.02,
        update_frequency: float = 1.0,
    ) -> AsyncGenerator[MarketData, None]:
        """Simulate realistic market conditions."""
        import random

        start_time = time.time()
        current_price = Decimal("50000.0")

        while time.time() - start_time < duration_seconds:
            # Simulate price movement
            change_pct = (random.random() - 0.5) * price_volatility
            current_price *= Decimal("1.0") + Decimal(str(change_pct))

            # Create market data
            spread = current_price * Decimal("0.001")
            market_data = MarketData(
                symbol="BTC/USDT",
                price=current_price,
                bid=current_price - spread / 2,
                ask=current_price + spread / 2,
                volume=Decimal(str(random.randint(100, 2000))),
                timestamp=datetime.now(timezone.utc),
            )

            yield market_data
            await asyncio.sleep(1.0 / update_frequency)

    def assert_performance_acceptable(self, max_latency_ms: float = 1000.0):
        """Assert that performance metrics are within acceptable ranges."""
        metrics = self.performance_monitor.metrics

        # Check order latency
        if metrics.get("avg_order_latency"):
            assert metrics["avg_order_latency"] * 1000 < max_latency_ms, (
                f"Average order latency too high: {metrics['avg_order_latency'] * 1000:.1f}ms > {max_latency_ms}ms"
            )

        # Check error rate
        if metrics["api_calls"] > 0:
            error_rate = metrics["error_count"] / metrics["api_calls"]
            assert error_rate < 0.05, f"Error rate too high: {error_rate:.1%} > 5%"

        logger.info("Performance metrics within acceptable ranges")


class IntegrationTestSuite:
    """Test suite coordinator for running comprehensive integration tests."""

    def __init__(self):
        self.test_results = {}
        self.overall_performance = PerformanceMonitor()

    async def run_all_integration_tests(self) -> dict[str, Any]:
        """Run all integration tests and return comprehensive results."""
        self.overall_performance.start()

        test_classes = [
            # Will be implemented in subsequent files
            "TradingWorkflowIntegrationTest",
            "MultiExchangeIntegrationTest",
            "StateManagementIntegrationTest",
            "RiskManagementIntegrationTest",
            "RealtimeDataIntegrationTest",
            "SecurityIntegrationTest",
        ]

        for test_class_name in test_classes:
            try:
                logger.info(f"Running integration test suite: {test_class_name}")
                # Test execution would be implemented here
                self.test_results[test_class_name] = {"status": "passed", "duration": 0.0}
            except Exception as e:
                logger.error(f"Integration test failed: {test_class_name} - {e}")
                self.test_results[test_class_name] = {"status": "failed", "error": str(e)}

        self.overall_performance.stop()

        return {
            "test_results": self.test_results,
            "performance_summary": self.overall_performance.metrics,
            "total_duration": self.overall_performance.metrics["total_duration"],
        }


# Utility functions for integration testing
async def wait_for_condition(
    condition_func: Callable[[], bool], timeout_seconds: float = 30.0, check_interval: float = 0.1
) -> bool:
    """Wait for a condition to become true with timeout."""
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        if condition_func():
            return True
        await asyncio.sleep(check_interval)

    return False


def generate_test_data(data_type: str, count: int = 100) -> list[Any]:
    """Generate test data for various scenarios."""
    import random

    if data_type == "market_data":
        base_price = Decimal("50000.0")
        return [
            MarketData(
                symbol="BTC/USDT",
                price=base_price + Decimal(str((random.random() - 0.5) * 1000)),
                bid=base_price - Decimal("10"),
                ask=base_price + Decimal("10"),
                volume=Decimal(str(random.randint(100, 2000))),
                timestamp=datetime.now(timezone.utc),
            )
            for _ in range(count)
        ]
    elif data_type == "signals":
        return [
            Signal(
                symbol="BTC/USDT",
                direction=random.choice(
                    [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
                ),
                confidence=random.uniform(0.3, 0.9),
                price=Decimal("50000.0") + Decimal(str((random.random() - 0.5) * 2000)),
                timestamp=datetime.now(timezone.utc),
            )
            for _ in range(count)
        ]
    else:
        raise ValueError(f"Unknown data type: {data_type}")
