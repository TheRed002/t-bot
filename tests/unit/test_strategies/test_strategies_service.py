"""
Tests for strategy service module.
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from functools import lru_cache

import pytest

# Disable logging during tests for performance
logging.disable(logging.CRITICAL)

# Fast mock time for deterministic tests
FIXED_TIME = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

from src.core.exceptions import ServiceError, StrategyError, ValidationError
from src.core.types import (
    MarketData,
    Signal,
    SignalDirection,
    StrategyConfig,
    StrategyStatus,
    StrategyType,
)
from src.strategies.service import StrategyService


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, strategy_id: str = "test_strategy"):
        self.strategy_id = strategy_id
        self.status = StrategyStatus.STOPPED
        self._signals = []

    async def start(self):
        self.status = StrategyStatus.ACTIVE

    async def stop(self):
        self.status = StrategyStatus.STOPPED

    async def generate_signals(self, data: MarketData) -> list[Signal]:
        return self._signals

    async def validate_signal(self, signal: Signal) -> bool:
        return signal.strength >= Decimal("0.5")

    def set_risk_manager(self, risk_manager):
        self.risk_manager = risk_manager

    def set_exchange(self, exchange):
        self.exchange = exchange

    def set_data_service(self, data_service):
        self.data_service = data_service

    def set_validation_framework(self, validation_framework):
        self.validation_framework = validation_framework

    def set_metrics_collector(self, metrics_collector):
        self.metrics_collector = metrics_collector

    def get_performance_summary(self) -> dict[str, Any]:
        return {"total_trades": 10, "win_rate": 0.7, "total_pnl": Decimal("1000.0")}

    def cleanup(self):
        pass


@pytest.fixture(scope="session")
@lru_cache(maxsize=1)
def mock_strategy():
    """Create a mock strategy - cached for session scope."""
    return MockStrategy()


@pytest.fixture(scope="session")
@lru_cache(maxsize=1)
def mock_config():
    """Create a mock strategy config - cached for session scope."""
    return StrategyConfig(
        strategy_id="test_strategy",
        strategy_type=StrategyType.MOMENTUM,
        name="Test Strategy",
        symbol="BTC/USD",
        timeframe="1h",
        enabled=True,
        parameters={"param1": "value1"},
        exchange_type="binance",
    )


@pytest.fixture(scope="session")
@lru_cache(maxsize=1)
def mock_market_data():
    """Create mock market data - cached for session scope with fixed time."""
    return MarketData(
        symbol="BTC/USD",
        open=Decimal("49900"),
        high=Decimal("50100"),
        low=Decimal("49800"),
        close=Decimal("50000"),
        volume=Decimal("100"),
        timestamp=FIXED_TIME,  # Use fixed time for performance
        exchange="binance",
        vwap=Decimal("50000"),
        trades_count=100,
        quote_volume=Decimal("5000000"),
        metadata={}
    )


@pytest.fixture(scope="session")
@lru_cache(maxsize=1)
def mock_signal():
    """Create a mock signal - cached for session scope with fixed time."""
    return Signal(
        signal_id="test_signal_1",
        strategy_id="test_strategy_1",
        strategy_name="test_strategy",
        symbol="BTC/USD",
        direction=SignalDirection.BUY,
        strength=Decimal("0.8"),
        timestamp=FIXED_TIME,  # Use fixed time for performance
        source="test_strategy",
        metadata={}
    )


@pytest.fixture(scope="function")
def mock_dependencies():
    """Create mock dependencies - fresh for each test."""
    exchange_factory = Mock()
    exchange_factory.get_exchange = AsyncMock()
    exchange_factory.is_exchange_supported = Mock(return_value=True)
    data_service = Mock()
    data_service.get_market_data = AsyncMock()

    # Create properly mocked service manager with async get_service method
    service_manager = Mock()
    service_manager.get_service = AsyncMock(return_value=Mock())

    return {
        "repository": Mock(),
        "risk_manager": AsyncMock(),
        "exchange_factory": exchange_factory,
        "data_service": data_service,
        "service_manager": service_manager,
    }


class TestStrategyServiceInitialization:
    """Test StrategyService initialization."""

    def test_init_with_default_values(self):
        """Test initialization with default values."""
        service = StrategyService()

        assert service.name == "StrategyService"
        assert isinstance(service._active_strategies, dict)
        assert isinstance(service._strategy_configs, dict)
        assert isinstance(service._strategy_metrics, dict)
        assert len(service._active_strategies) == 0

    def test_init_with_dependencies(self, mock_dependencies):
        """Test initialization with dependencies."""
        service = StrategyService(
            name="TestService", config={"test": "config"}, **mock_dependencies
        )

        assert service.name == "TestService"
        assert service._repository is mock_dependencies["repository"]
        assert service._risk_manager is mock_dependencies["risk_manager"]

    @pytest.mark.asyncio
    async def test_start_service_with_missing_dependencies(self):
        """Test starting service with missing dependencies."""
        service = StrategyService()

        # Should start with warnings but not fail
        await service.start()
        assert service.is_running  # Service should still start

    @pytest.mark.asyncio
    async def test_start_service_with_exception(self):
        """Test service start failure."""
        service = StrategyService()

        # Mock an exception during start
        with patch.object(service, "_logger") as mock_logger:
            mock_logger.info.side_effect = Exception("Test error")

            with pytest.raises(ServiceError):
                await service._do_start()


class TestStrategyRegistration:
    """Test strategy registration functionality."""

    @pytest.mark.asyncio
    async def test_register_strategy_success(self, mock_dependencies, mock_strategy, mock_config):
        """Test successful strategy registration."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Mock config validation
        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        assert "test_id" in service._active_strategies
        assert "test_id" in service._strategy_configs
        assert "test_id" in service._strategy_metrics
        assert service._active_strategies["test_id"] == mock_strategy

    @pytest.mark.asyncio
    async def test_register_duplicate_strategy(self, mock_dependencies, mock_strategy, mock_config):
        """Test registering duplicate strategy fails."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

            with pytest.raises(ServiceError):
                await service.register_strategy("test_id", mock_strategy, mock_config)

    @pytest.mark.asyncio
    async def test_register_strategy_invalid_config(
        self, mock_dependencies, mock_strategy, mock_config
    ):
        """Test registering strategy with invalid config fails."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=False):
            with pytest.raises(ServiceError):
                await service.register_strategy("test_id", mock_strategy, mock_config)

    @pytest.mark.asyncio
    async def test_register_strategy_with_risk_manager(
        self, mock_dependencies, mock_strategy, mock_config
    ):
        """Test strategy registration with risk manager injection."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        # Verify risk manager was injected
        assert hasattr(mock_strategy, "risk_manager")

    @pytest.mark.asyncio
    async def test_register_strategy_with_exchange_factory(
        self, mock_dependencies, mock_strategy, mock_config
    ):
        """Test strategy registration with exchange factory."""
        mock_exchange = Mock()
        mock_dependencies["exchange_factory"].get_exchange.return_value = mock_exchange

        service = StrategyService(**mock_dependencies)
        await service.start()

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        # Verify exchange was set
        mock_dependencies["exchange_factory"].get_exchange.assert_called_once()
        assert hasattr(mock_strategy, "exchange")


class TestStrategyLifecycle:
    """Test strategy lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_strategy_success(self, mock_dependencies, mock_strategy, mock_config):
        """Test successful strategy start."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Register strategy first
        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        with patch.object(service, "_validate_start_conditions", return_value=True):
            await service.start_strategy("test_id")

        assert mock_strategy.status == StrategyStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_start_unregistered_strategy(self, mock_dependencies):
        """Test starting unregistered strategy fails."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with pytest.raises(ServiceError):
            await service.start_strategy("nonexistent")

    @pytest.mark.asyncio
    async def test_start_strategy_conditions_not_met(
        self, mock_dependencies, mock_strategy, mock_config
    ):
        """Test strategy start fails when conditions not met."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        with patch.object(service, "_validate_start_conditions", return_value=False):
            with pytest.raises(ServiceError):
                await service.start_strategy("test_id")

    @pytest.mark.asyncio
    async def test_stop_strategy_success(self, mock_dependencies, mock_strategy, mock_config):
        """Test successful strategy stop."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Register and start strategy
        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        await service.stop_strategy("test_id")
        assert mock_strategy.status == StrategyStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_unregistered_strategy(self, mock_dependencies):
        """Test stopping unregistered strategy fails."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with pytest.raises(ServiceError):
            await service.stop_strategy("nonexistent")


class TestSignalProcessing:
    """Test signal processing functionality."""

    @pytest.mark.asyncio
    async def test_process_market_data_success(
        self, mock_dependencies, mock_strategy, mock_config, mock_market_data, mock_signal
    ):
        """Test successful market data processing."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Setup strategy with signal
        mock_strategy._signals = [mock_signal]
        mock_strategy.status = StrategyStatus.ACTIVE

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        with patch.object(service, "validate_signal", return_value=True):
            result = await service.process_market_data(mock_market_data)

        assert "test_id" in result
        assert len(result["test_id"]) == 1
        assert result["test_id"][0] == mock_signal

    @pytest.mark.asyncio
    async def test_process_market_data_inactive_strategy(
        self, mock_dependencies, mock_strategy, mock_config, mock_market_data
    ):
        """Test market data processing skips inactive strategies."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        mock_strategy.status = StrategyStatus.STOPPED  # Inactive

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        result = await service.process_market_data(mock_market_data)

        # Should skip inactive strategy
        assert "test_id" not in result or len(result.get("test_id", [])) == 0

    @pytest.mark.asyncio
    async def test_process_market_data_validation_failure(
        self, mock_dependencies, mock_strategy, mock_config, mock_market_data, mock_signal
    ):
        """Test signal validation failure during market data processing."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        mock_strategy._signals = [mock_signal]
        mock_strategy.status = StrategyStatus.ACTIVE

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        with patch.object(service, "validate_signal", return_value=False):
            result = await service.process_market_data(mock_market_data)

        assert "test_id" in result
        assert len(result["test_id"]) == 0  # Signal was filtered out

    @pytest.mark.asyncio
    async def test_process_market_data_strategy_exception(
        self, mock_dependencies, mock_strategy, mock_config, mock_market_data
    ):
        """Test handling strategy exception during market data processing."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        mock_strategy.status = StrategyStatus.ACTIVE
        mock_strategy.generate_signals = AsyncMock(side_effect=Exception("Strategy error"))

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        result = await service.process_market_data(mock_market_data)

        assert "test_id" in result
        assert len(result["test_id"]) == 0  # Exception handled gracefully


class TestSignalValidation:
    """Test signal validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_signal_success(
        self, mock_dependencies, mock_strategy, mock_config, mock_signal
    ):
        """Test successful signal validation."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        result = await service.validate_signal("test_id", mock_signal)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_signal_missing_symbol(self, mock_dependencies):
        """Test signal creation fails for missing symbol."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Signal creation should fail with empty symbol
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            Signal(
                signal_id="test_signal_2",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol="",  # Empty symbol
                direction=SignalDirection.BUY,
                strength=Decimal("0.8"),
                timestamp=FIXED_TIME,
                source="test",
            )

    @pytest.mark.asyncio
    async def test_validate_signal_low_strength(self, mock_dependencies):
        """Test signal validation fails for low strength."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        weak_signal = Signal(
            signal_id="test_signal_3",
            strategy_id="test_strategy_1",
            strategy_name="test_strategy",
            symbol="BTC/USD",
            direction=SignalDirection.BUY,
            strength=Decimal("0.05"),  # Below threshold
            timestamp=FIXED_TIME,
            source="test",
        )

        result = await service.validate_signal("test_id", weak_signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_with_risk_manager(self, mock_dependencies, mock_signal):
        """Test signal validation with risk manager."""
        mock_dependencies["risk_manager"].validate_signal.return_value = False

        service = StrategyService(**mock_dependencies)
        await service.start()

        result = await service.validate_signal("test_id", mock_signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_exception_handling(self, mock_dependencies, mock_signal):
        """Test signal validation exception handling."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Make strategy validation raise exception
        bad_strategy = Mock()
        bad_strategy.validate_signal = AsyncMock(side_effect=Exception("Validation error"))
        service._active_strategies["test_id"] = bad_strategy

        result = await service.validate_signal("test_id", mock_signal)
        assert result is False


class TestConfigurationValidation:
    """Test configuration validation."""

    @pytest.mark.asyncio
    async def test_validate_strategy_config_success(self, mock_dependencies, mock_config):
        """Test successful config validation."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        result = await service.validate_strategy_config(mock_config)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_strategy_config_missing_name(self, mock_dependencies):
        """Test config validation fails for missing name."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        bad_config = StrategyConfig(
            strategy_id="test",
            strategy_type=StrategyType.MOMENTUM,
            name="",  # Empty name
            symbol="BTC/USD",
            timeframe="1h",
            enabled=True,
            parameters={},
        )

        result = await service.validate_strategy_config(bad_config)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_strategy_config_missing_parameters(self, mock_dependencies):
        """Test config validation fails for missing parameters."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        bad_config = StrategyConfig(
            strategy_id="test",
            strategy_type=StrategyType.MOMENTUM,
            name="Test",
            symbol="BTC/USD",
            timeframe="1h",
            enabled=True,
            parameters={},  # Empty parameters
        )

        result = await service.validate_strategy_config(bad_config)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_strategy_config_unsupported_exchange(self, mock_dependencies):
        """Test config validation fails for unsupported exchange."""
        mock_dependencies["exchange_factory"].is_exchange_supported.return_value = False

        service = StrategyService(**mock_dependencies)
        await service.start()

        config_with_exchange = StrategyConfig(
            strategy_id="test",
            strategy_type=StrategyType.MOMENTUM,
            name="Test",
            symbol="BTC/USD",
            timeframe="1h",
            enabled=True,
            parameters={"param": "value"},
            exchange_type="unsupported_exchange",
        )

        result = await service.validate_strategy_config(config_with_exchange)
        assert result is False


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""

    @pytest.mark.asyncio
    async def test_get_strategy_performance_success(
        self, mock_dependencies, mock_strategy, mock_config
    ):
        """Test successful performance retrieval."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        result = await service.get_strategy_performance("test_id")

        assert result["strategy_id"] == "test_id"
        assert "config" in result
        assert "metrics" in result
        assert "total_trades" in result  # From mock strategy performance

    @pytest.mark.asyncio
    async def test_get_strategy_performance_not_found(self, mock_dependencies):
        """Test performance retrieval for non-existent strategy."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with pytest.raises(ServiceError):
            await service.get_strategy_performance("nonexistent")

    @pytest.mark.asyncio
    async def test_get_all_strategies(self, mock_dependencies, mock_strategy, mock_config):
        """Test getting all strategies."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        result = await service.get_all_strategies()

        assert "test_id" in result
        assert isinstance(result["test_id"], dict)


class TestStrategyCleanup:
    """Test strategy cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_strategy_success(self, mock_dependencies, mock_strategy, mock_config):
        """Test successful strategy cleanup."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Register and setup strategy
        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        mock_strategy.status = StrategyStatus.ACTIVE

        await service.cleanup_strategy("test_id")

        # Verify cleanup
        assert "test_id" not in service._active_strategies
        assert "test_id" not in service._strategy_configs
        assert "test_id" not in service._strategy_metrics

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_strategy(self, mock_dependencies):
        """Test cleanup of non-existent strategy."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Should not raise error
        await service.cleanup_strategy("nonexistent")


class TestServiceMetrics:
    """Test service metrics and health checks."""

    def test_get_metrics(self, mock_dependencies):
        """Test getting service metrics."""
        service = StrategyService(**mock_dependencies)

        metrics = service.get_metrics()

        assert "total_strategies" in metrics
        assert "running_strategies" in metrics
        assert "total_signals_generated" in metrics
        assert "strategies_by_status" in metrics

    @pytest.mark.asyncio
    async def test_service_health_check_healthy(
        self, mock_dependencies, mock_strategy, mock_config
    ):
        """Test healthy service health check."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        # Add active strategy
        mock_strategy.status = StrategyStatus.ACTIVE
        with patch.object(service, "validate_strategy_config", new_callable=AsyncMock, return_value=True):
            await service.register_strategy("test_id", mock_strategy, mock_config)

        health = await service._service_health_check()

        from src.core.base.interfaces import HealthStatus

        assert health == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_service_health_check_degraded_no_strategies(self, mock_dependencies):
        """Test degraded health with no strategies."""
        service = StrategyService(**mock_dependencies)
        await service.start()

        health = await service._service_health_check()

        from src.core.base.interfaces import HealthStatus

        assert health == HealthStatus.DEGRADED


class TestDependencyResolution:
    """Test dependency resolution functionality."""

    def test_resolve_dependency_success(self, mock_dependencies):
        """Test successful dependency resolution."""
        service = StrategyService(**mock_dependencies)

        repository = service.resolve_dependency("StrategyRepository")
        assert repository is mock_dependencies["repository"]

    def test_resolve_dependency_not_found(self, mock_dependencies):
        """Test dependency resolution failure."""
        service = StrategyService(**mock_dependencies)

        with pytest.raises(KeyError):
            service.resolve_dependency("NonExistentDependency")

    def test_resolve_dependency_none_value(self):
        """Test resolving None dependency."""
        service = StrategyService(repository=None)

        with pytest.raises(KeyError):
            service.resolve_dependency("StrategyRepository")


# TestBacktestingIntegration removed - StrategyService doesn't have run_backtest method
# Backtesting is handled by the BacktestingService in the backtesting module
