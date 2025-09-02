"""
Tests for Correlation Spike Circuit Breaker.

This module provides comprehensive tests for correlation-based circuit breaker
functionality, including edge cases and integration scenarios.

CRITICAL: Tests must verify proper triggering, graduated responses, and integration
with existing circuit breaker system.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import Config
from src.core.exceptions import CircuitBreakerTriggeredError
from src.core.types.market import MarketData
from src.core.types.risk import CircuitBreakerStatus, CircuitBreakerType
from src.core.types.trading import OrderSide, Position, PositionSide, PositionStatus
from src.risk_management.base import BaseRiskManager
from src.risk_management.circuit_breakers import CircuitBreakerManager, CorrelationSpikeBreaker


@pytest.fixture
def sample_config():
    """Create sample configuration for tests."""
    config = Config()
    return config


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    risk_manager = MagicMock(spec=BaseRiskManager)
    return risk_manager


@pytest.fixture
def correlation_breaker(sample_config, mock_risk_manager):
    """Create correlation spike breaker for testing."""
    return CorrelationSpikeBreaker(sample_config, mock_risk_manager)


@pytest.fixture
def circuit_breaker_manager(sample_config, mock_risk_manager):
    """Create circuit breaker manager with correlation breaker."""
    return CircuitBreakerManager(sample_config, mock_risk_manager)


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    timestamp = datetime.now(timezone.utc)
    return [
        Position(
            symbol="BTC/USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("50100.00"),
            unrealized_pnl=Decimal("100.00"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=timestamp,
            exchange="binance",
            metadata={},
        ),
        Position(
            symbol="ETH/USD",
            quantity=Decimal("10.0"),
            entry_price=Decimal("3000.00"),
            current_price=Decimal("3030.00"),
            unrealized_pnl=Decimal("300.00"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=timestamp,
            exchange="binance",
            metadata={},
        ),
        Position(
            symbol="ADA/USD",
            quantity=Decimal("1000.0"),
            entry_price=Decimal("1.00"),
            current_price=Decimal("1.05"),
            unrealized_pnl=Decimal("50.00"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=timestamp,
            exchange="binance",
            metadata={},
        ),
    ]


@pytest.fixture
def correlated_market_data():
    """Create correlated market data for testing."""
    base_time = datetime.now(timezone.utc)
    data = []

    # Create highly correlated price movements
    prices = [
        (50000, 3000, 1.00),  # Base prices
        (50500, 3030, 1.01),  # All up 1%
        (49750, 2985, 0.995),  # All down ~0.5%
        (50750, 3045, 1.015),  # All up 1.5%
        (49500, 2970, 0.99),  # All down 1%
        (51000, 3060, 1.02),  # All up 2%
    ]

    for i, (btc_price, eth_price, ada_price) in enumerate(prices):
        timestamp = base_time - timedelta(minutes=10 - i)

        data.extend(
            [
                MarketData(
                    symbol="BTC/USD",
                    timestamp=timestamp,
                    open=Decimal(str(btc_price)),
                    high=Decimal(str(btc_price)),
                    low=Decimal(str(btc_price)),
                    close=Decimal(str(btc_price)),
                    volume=Decimal("1000.0"),
                    exchange="test_exchange"
                ),
                MarketData(
                    symbol="ETH/USD",
                    timestamp=timestamp,
                    open=Decimal(str(eth_price)),
                    high=Decimal(str(eth_price)),
                    low=Decimal(str(eth_price)),
                    close=Decimal(str(eth_price)),
                    volume=Decimal("2000.0"),
                    exchange="test_exchange"
                ),
                MarketData(
                    symbol="ADA/USD",
                    timestamp=timestamp,
                    open=Decimal(str(ada_price)),
                    high=Decimal(str(ada_price)),
                    low=Decimal(str(ada_price)),
                    close=Decimal(str(ada_price)),
                    volume=Decimal("5000.0"),
                    exchange="test_exchange"
                ),
            ]
        )

    return data


class TestCorrelationSpikeBreaker:
    """Test correlation spike circuit breaker functionality."""

    def test_initialization(self, correlation_breaker):
        """Test correlation spike breaker initialization."""
        assert correlation_breaker.breaker_type == CircuitBreakerType.CORRELATION_SPIKE
        assert correlation_breaker.state == CircuitBreakerStatus.ACTIVE
        assert correlation_breaker.correlation_spike_count == 0
        assert correlation_breaker.consecutive_high_correlation_periods == 0
        assert correlation_breaker.max_consecutive_periods == 3
        assert correlation_breaker.thresholds.warning_threshold == Decimal("0.6")
        assert correlation_breaker.thresholds.critical_threshold == Decimal("0.8")

    @pytest.mark.asyncio
    async def test_get_threshold_value(self, correlation_breaker):
        """Test getting correlation threshold value."""
        threshold = await correlation_breaker.get_threshold_value()
        assert threshold == Decimal("0.8")  # Critical threshold

    @pytest.mark.asyncio
    async def test_get_current_value_no_data(self, correlation_breaker):
        """Test getting current correlation value with no data."""
        data = {"positions": [], "market_data": []}
        current_value = await correlation_breaker.get_current_value(data)
        assert current_value == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_get_current_value_insufficient_positions(
        self, correlation_breaker, sample_positions
    ):
        """Test getting current correlation value with insufficient positions."""
        data = {"positions": [sample_positions[0]], "market_data": []}
        current_value = await correlation_breaker.get_current_value(data)
        assert current_value == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_get_current_value_with_positions(
        self, correlation_breaker, sample_positions, correlated_market_data
    ):
        """Test getting current correlation value with positions and market data."""
        data = {"positions": sample_positions, "market_data": correlated_market_data}

        current_value = await correlation_breaker.get_current_value(data)

        # Should return the maximum pairwise correlation
        assert isinstance(current_value, Decimal)
        assert current_value >= Decimal("0.0")

    @pytest.mark.asyncio
    async def test_check_condition_no_data(self, correlation_breaker):
        """Test condition checking with no correlation data."""
        data = {"positions": [], "market_data": []}

        should_trigger = await correlation_breaker.check_condition(data)

        assert should_trigger is False
        assert correlation_breaker.consecutive_high_correlation_periods == 0

    @pytest.mark.asyncio
    async def test_check_condition_low_correlation(self, correlation_breaker):
        """Test condition checking with low correlation."""
        # Mock the get_current_value to return low correlation
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("0.3"))

        data = {"positions": [], "market_data": []}
        should_trigger = await correlation_breaker.check_condition(data)

        assert should_trigger is False
        assert correlation_breaker.consecutive_high_correlation_periods == 0

    @pytest.mark.asyncio
    async def test_check_condition_warning_level(self, correlation_breaker):
        """Test condition checking with warning level correlation."""
        # Mock moderate correlation (above warning, below critical)
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("0.7"))

        data = {"positions": [], "market_data": []}

        # Should not trigger on first occurrence
        should_trigger = await correlation_breaker.check_condition(data)
        assert should_trigger is False
        assert correlation_breaker.consecutive_high_correlation_periods == 1

        # Should not trigger on second occurrence
        should_trigger = await correlation_breaker.check_condition(data)
        assert should_trigger is False
        assert correlation_breaker.consecutive_high_correlation_periods == 2

        # Should trigger on third consecutive occurrence
        should_trigger = await correlation_breaker.check_condition(data)
        assert should_trigger is True
        assert correlation_breaker.consecutive_high_correlation_periods == 3
        assert correlation_breaker.correlation_spike_count == 1

    @pytest.mark.asyncio
    async def test_check_condition_critical_level(self, correlation_breaker):
        """Test condition checking with critical level correlation."""
        # Mock critical correlation (above critical threshold)
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("0.9"))

        data = {"positions": [], "market_data": []}

        # Should trigger immediately at critical level
        should_trigger = await correlation_breaker.check_condition(data)

        assert should_trigger is True
        assert correlation_breaker.consecutive_high_correlation_periods == 1
        assert correlation_breaker.correlation_spike_count == 1

    @pytest.mark.asyncio
    async def test_check_condition_reset_counter(self, correlation_breaker):
        """Test that consecutive period counter resets when correlation drops."""
        data = {"positions": [], "market_data": []}

        # Start with warning level correlation
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("0.7"))
        await correlation_breaker.check_condition(data)
        await correlation_breaker.check_condition(data)

        assert correlation_breaker.consecutive_high_correlation_periods == 2

        # Drop below warning level
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("0.5"))
        should_trigger = await correlation_breaker.check_condition(data)

        assert should_trigger is False
        assert correlation_breaker.consecutive_high_correlation_periods == 0

    @pytest.mark.asyncio
    async def test_check_condition_concentration_risk(self, correlation_breaker):
        """Test condition checking with high concentration risk."""
        from src.risk_management.correlation_monitor import CorrelationMetrics

        # Mock warning level correlation with high concentration risk
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("0.7"))
        correlation_breaker.last_correlation_metrics = CorrelationMetrics(
            average_correlation=Decimal("0.6"),
            max_pairwise_correlation=Decimal("0.7"),
            correlation_spike=True,
            correlated_pairs_count=3,
            portfolio_concentration_risk=Decimal("0.8"),  # High concentration risk
            timestamp=datetime.now(timezone.utc),
            correlation_matrix={},
        )

        data = {"positions": [], "market_data": []}

        # Should not trigger on first occurrence
        await correlation_breaker.check_condition(data)
        assert correlation_breaker.consecutive_high_correlation_periods == 1

        # Should trigger on second occurrence due to high concentration risk
        should_trigger = await correlation_breaker.check_condition(data)
        assert should_trigger is True
        assert correlation_breaker.consecutive_high_correlation_periods == 2

    @pytest.mark.asyncio
    async def test_evaluate_triggers_exception(
        self, correlation_breaker, correlated_market_data, sample_positions
    ):
        """Test that evaluate method triggers CircuitBreakerTriggeredError."""
        # Mock critical correlation to trigger circuit breaker
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("0.9"))

        data = {"positions": sample_positions, "market_data": correlated_market_data}

        with pytest.raises(CircuitBreakerTriggeredError) as exc_info:
            await correlation_breaker.evaluate(data)

        assert "Circuit breaker triggered" in str(exc_info.value)
        assert correlation_breaker.state == CircuitBreakerStatus.TRIGGERED
        assert correlation_breaker.trigger_count == 1

    def test_get_correlation_metrics_none(self, correlation_breaker):
        """Test getting correlation metrics when none available."""
        metrics = correlation_breaker.get_correlation_metrics()
        assert metrics is None

    def test_get_correlation_metrics_available(self, correlation_breaker):
        """Test getting correlation metrics when available."""
        from src.risk_management.correlation_monitor import CorrelationMetrics

        correlation_breaker.last_correlation_metrics = CorrelationMetrics(
            average_correlation=Decimal("0.6"),
            max_pairwise_correlation=Decimal("0.7"),
            correlation_spike=True,
            correlated_pairs_count=2,
            portfolio_concentration_risk=Decimal("0.4"),
            timestamp=datetime.now(timezone.utc),
            correlation_matrix={},
        )

        metrics = correlation_breaker.get_correlation_metrics()

        assert metrics is not None
        assert metrics["average_correlation"] == "0.6"
        assert metrics["max_pairwise_correlation"] == "0.7"
        assert metrics["correlation_spike"] is True
        assert metrics["correlated_pairs_count"] == 2
        assert metrics["portfolio_concentration_risk"] == "0.4"
        assert "timestamp" in metrics

    @pytest.mark.asyncio
    async def test_get_position_limits_none(self, correlation_breaker):
        """Test getting position limits when no metrics available."""
        limits = await correlation_breaker.get_position_limits()

        assert limits["max_positions"] is None
        assert limits["reduction_factor"] == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_get_position_limits_with_metrics(self, correlation_breaker):
        """Test getting position limits with correlation metrics."""
        from src.risk_management.correlation_monitor import CorrelationMetrics

        correlation_breaker.last_correlation_metrics = CorrelationMetrics(
            average_correlation=Decimal("0.5"),
            max_pairwise_correlation=Decimal("0.7"),  # Warning level
            correlation_spike=True,
            correlated_pairs_count=2,
            portfolio_concentration_risk=Decimal("0.4"),
            timestamp=datetime.now(timezone.utc),
            correlation_matrix={},
        )

        limits = await correlation_breaker.get_position_limits()

        assert "max_positions" in limits
        assert "correlation_based_reduction" in limits
        assert "warning_level" in limits

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, correlation_breaker):
        """Test cleanup of old correlation data."""
        cutoff_time = datetime.now(timezone.utc)

        # Should not raise any exceptions
        await correlation_breaker.cleanup_old_data(cutoff_time)

    def test_reset(self, correlation_breaker):
        """Test resetting correlation circuit breaker state."""
        # Set some state first
        correlation_breaker.correlation_spike_count = 5
        correlation_breaker.consecutive_high_correlation_periods = 3
        correlation_breaker.state = CircuitBreakerStatus.TRIGGERED

        correlation_breaker.reset()

        assert correlation_breaker.state == CircuitBreakerStatus.ACTIVE
        assert correlation_breaker.correlation_spike_count == 0
        assert correlation_breaker.consecutive_high_correlation_periods == 0
        assert correlation_breaker.last_correlation_metrics is None


class TestCorrelationBreakerIntegration:
    """Test integration of correlation breaker with circuit breaker manager."""

    def test_circuit_breaker_manager_includes_correlation(self, circuit_breaker_manager):
        """Test that circuit breaker manager includes correlation spike breaker."""
        assert "correlation_spike" in circuit_breaker_manager.circuit_breakers

        correlation_breaker = circuit_breaker_manager.circuit_breakers["correlation_spike"]
        assert isinstance(correlation_breaker, CorrelationSpikeBreaker)
        assert correlation_breaker.breaker_type == CircuitBreakerType.CORRELATION_SPIKE

    @pytest.mark.asyncio
    async def test_evaluate_all_includes_correlation(
        self, circuit_breaker_manager, sample_positions
    ):
        """Test that evaluate_all includes correlation circuit breaker."""
        data = {
            "positions": sample_positions,
            "market_data": [],
            "portfolio_value": Decimal("100000.00"),
            "daily_pnl": Decimal("-1000.00"),
            "current_portfolio_value": Decimal("99000.00"),
            "peak_portfolio_value": Decimal("100000.00"),
            "price_history": [50000, 50100, 49900],
            "model_confidence": Decimal("0.8"),
            "error_occurred": False,
            "total_requests": 100,
        }

        # Should not raise exception with normal data
        results = await circuit_breaker_manager.evaluate_all(data)

        assert "correlation_spike" in results
        assert isinstance(results["correlation_spike"], bool)

    @pytest.mark.asyncio
    async def test_evaluate_all_correlation_triggered(
        self, circuit_breaker_manager, sample_positions, correlated_market_data
    ):
        """Test evaluate_all when correlation circuit breaker triggers."""
        # Mock the correlation breaker to trigger
        correlation_breaker = circuit_breaker_manager.circuit_breakers["correlation_spike"]
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("0.9"))

        data = {
            "positions": sample_positions,
            "market_data": correlated_market_data,
            "portfolio_value": Decimal("100000.00"),
            "daily_pnl": Decimal("500.00"),  # Positive to avoid daily loss trigger
            "current_portfolio_value": Decimal("100500.00"),
            "peak_portfolio_value": Decimal("100000.00"),
            "price_history": [50000, 50100, 50200],  # Low volatility
            "model_confidence": Decimal("0.8"),  # High confidence
            "error_occurred": False,
            "total_requests": 100,
        }

        with pytest.raises(CircuitBreakerTriggeredError) as exc_info:
            await circuit_breaker_manager.evaluate_all(data)

        assert "correlation_spike" in str(exc_info.value)

    def test_get_status_includes_correlation(self, circuit_breaker_manager):
        """Test that get_status includes correlation circuit breaker."""
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            status = loop.run_until_complete(circuit_breaker_manager.get_status())

            assert "correlation_spike" in status
            assert "state" in status["correlation_spike"]
            assert "trigger_count" in status["correlation_spike"]
        finally:
            loop.close()

    def test_reset_all_includes_correlation(self, circuit_breaker_manager):
        """Test that reset_all resets correlation circuit breaker."""
        # Set correlation breaker state
        correlation_breaker = circuit_breaker_manager.circuit_breakers["correlation_spike"]
        correlation_breaker.state = CircuitBreakerStatus.TRIGGERED
        correlation_breaker.correlation_spike_count = 3

        # Reset all
        circuit_breaker_manager.reset_all()

        assert correlation_breaker.state == CircuitBreakerStatus.ACTIVE
        assert correlation_breaker.correlation_spike_count == 0

    def test_get_triggered_breakers_includes_correlation(self, circuit_breaker_manager):
        """Test that get_triggered_breakers includes correlation when triggered."""
        # Trigger correlation breaker
        correlation_breaker = circuit_breaker_manager.circuit_breakers["correlation_spike"]
        correlation_breaker.state = CircuitBreakerStatus.TRIGGERED

        triggered = circuit_breaker_manager.get_triggered_breakers()

        assert "correlation_spike" in triggered

    def test_is_trading_allowed_considers_correlation(self, circuit_breaker_manager):
        """Test that is_trading_allowed considers correlation circuit breaker."""
        # Initially should allow trading
        assert circuit_breaker_manager.is_trading_allowed() is True

        # Trigger correlation breaker
        correlation_breaker = circuit_breaker_manager.circuit_breakers["correlation_spike"]
        correlation_breaker.state = CircuitBreakerStatus.TRIGGERED

        # Should not allow trading
        assert circuit_breaker_manager.is_trading_allowed() is False


class TestCorrelationEdgeCases:
    """Test edge cases for correlation circuit breaker."""

    @pytest.mark.asyncio
    async def test_correlation_calculation_error_handling(self, correlation_breaker):
        """Test error handling in correlation calculations."""
        # Mock correlation monitor to raise exception
        correlation_breaker.correlation_monitor.calculate_portfolio_correlation = AsyncMock(
            side_effect=Exception("Calculation error")
        )

        data = {"positions": [MagicMock()], "market_data": []}
        current_value = await correlation_breaker.get_current_value(data)

        # Should return 0 on error and log warning
        assert current_value == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_extreme_correlation_values(self, correlation_breaker):
        """Test handling of extreme correlation values."""
        # Test perfect positive correlation
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("1.0"))

        data = {"positions": [], "market_data": []}
        should_trigger = await correlation_breaker.check_condition(data)

        assert should_trigger is True

        # Reset and test perfect negative correlation (should also trigger)
        correlation_breaker.reset()
        correlation_breaker.get_current_value = AsyncMock(return_value=Decimal("-1.0"))

        should_trigger = await correlation_breaker.check_condition(data)
        assert should_trigger is True

    @pytest.mark.asyncio
    async def test_market_data_without_positions(self, correlation_breaker, correlated_market_data):
        """Test market data updates without positions."""
        data = {"positions": [], "market_data": correlated_market_data}

        current_value = await correlation_breaker.get_current_value(data)

        # Should return 0 when no positions but still process market data
        assert current_value == Decimal("0.0")

        # Verify market data was processed (correlation monitor should have data)
        assert len(correlation_breaker.correlation_monitor.price_history) > 0

    @pytest.mark.asyncio
    async def test_positions_without_market_data(self, correlation_breaker, sample_positions):
        """Test positions without market data updates."""
        data = {"positions": sample_positions, "market_data": []}

        current_value = await correlation_breaker.get_current_value(data)

        # Should return 0 when no market data available for correlation calculation
        assert current_value == Decimal("0.0")

    def test_concurrent_access_safety(self, correlation_breaker):
        """Test concurrent access to correlation breaker state."""
        import threading
        import time

        results = []

        def update_counter():
            for _ in range(100):
                correlation_breaker.consecutive_high_correlation_periods += 1
                time.sleep(0.001)  # Small delay to increase chance of race conditions
                results.append(correlation_breaker.consecutive_high_correlation_periods)

        # Create multiple threads updating the counter
        threads = [threading.Thread(target=update_counter) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Final value should be consistent (300 if no race conditions)
        # Note: This test might be flaky due to the nature of testing race conditions
        assert correlation_breaker.consecutive_high_correlation_periods == 300
