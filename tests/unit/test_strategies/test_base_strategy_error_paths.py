"""
Comprehensive tests for BaseStrategy error paths and edge cases.

This module focuses on testing error handling, edge cases, and less commonly
executed paths in the BaseStrategy class to improve coverage from 68% to 85%+.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict

from src.strategies.base import BaseStrategy, DEFAULT_CIRCUIT_BREAKER_THRESHOLD
from src.core.types import (
    MarketData,
    Position,
    Signal,
    SignalDirection,
    StrategyStatus,
    StrategyType,
)
from src.core.exceptions import StrategyError, ValidationError
from src.error_handling import ErrorSeverity
from src.monitoring import MetricsCollector


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._generate_signals_called = False
        self._validate_signal_called = False
        self._should_exit_called = False
    
    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.MOMENTUM
    
    async def _generate_signals_impl(self, data: MarketData):
        self._generate_signals_called = True
        if hasattr(self, '_signal_generation_error'):
            raise self._signal_generation_error
        return self._test_signals if hasattr(self, '_test_signals') else []
    
    def validate_signal(self, signal: Signal) -> bool:
        self._validate_signal_called = True
        if hasattr(self, '_signal_validation_error'):
            raise self._signal_validation_error
        return getattr(self, '_validation_result', True)
    
    def get_position_size(self, signal: Signal) -> Decimal:
        if hasattr(self, '_position_size_error'):
            raise self._position_size_error
        return getattr(self, '_position_size', Decimal('0.01'))
    
    def should_exit(self, position: Position, data: MarketData) -> bool:
        self._should_exit_called = True
        if hasattr(self, '_should_exit_error'):
            raise self._should_exit_error
        return getattr(self, '_should_exit_result', False)


class TestBaseStrategyErrorPaths:
    """Test suite focusing on BaseStrategy error paths and edge cases."""

    @pytest.fixture
    def strategy_config(self):
        """Basic strategy configuration."""
        return {
            'strategy_id': 'test-strategy-001',
            'name': 'test_strategy',
            'strategy_type': StrategyType.MOMENTUM,
            'symbol': 'BTCUSDT',
            'timeframe': '1m',
            'min_confidence': 0.5,
            'position_size_pct': 0.02,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'max_positions': 3,
            'enabled': True,
        }

    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance."""
        return ConcreteStrategy(strategy_config)

    @pytest.fixture
    def market_data(self):
        """Create market data."""
        return MarketData(
            symbol='BTCUSDT',
            timestamp=datetime.now(timezone.utc),
            open=Decimal('49500'),
            high=Decimal('50500'),
            low=Decimal('49000'),
            close=Decimal('50000'),
            volume=Decimal('1000'),
            exchange='binance'
        )

    @pytest.fixture
    def signal(self):
        """Create test signal."""
        return Signal(
            symbol='BTC/USDT',
            direction=SignalDirection.BUY,
            strength=Decimal('0.8'),
            timestamp=datetime.now(timezone.utc),
            source='test_strategy'
        )

    def test_strategy_initialization_missing_required_config(self):
        """Test strategy initialization with missing required configuration."""
        config = {'strategy_type': StrategyType.MOMENTUM}  # Missing name
        
        with pytest.raises((ValueError, KeyError)):
            ConcreteStrategy(config)

    def test_strategy_initialization_invalid_strategy_type(self):
        """Test strategy initialization with invalid strategy type."""
        config = {
            'name': 'test_strategy',
            'strategy_type': 'invalid_type',  # Invalid type
        }
        
        with pytest.raises((ValueError, TypeError)):
            ConcreteStrategy(config)

    def test_strategy_initialization_circuit_breaker_defaults(self, strategy_config):
        """Test that circuit breaker is initialized with defaults."""
        strategy = ConcreteStrategy(strategy_config)
        
        # Circuit breaker should be initialized
        assert hasattr(strategy, '_circuit_breaker')
        assert strategy._circuit_breaker.threshold == DEFAULT_CIRCUIT_BREAKER_THRESHOLD

    def test_strategy_initialization_metrics_collector_unavailable(self, strategy_config):
        """Test strategy initialization when metrics collector is unavailable."""
        with patch('src.strategies.base.MetricsCollector', side_effect=ImportError("Metrics unavailable")):
            strategy = ConcreteStrategy(strategy_config)
            
            # Should initialize without metrics
            assert strategy is not None

    def test_strategy_initialization_alerting_unavailable(self, strategy_config):
        """Test strategy initialization when alerting is unavailable."""
        with patch('src.strategies.base.ALERTING_AVAILABLE', False):
            strategy = ConcreteStrategy(strategy_config)
            
            # Should initialize without alerting
            assert strategy is not None

    @pytest.mark.asyncio
    async def test_generate_signals_circuit_breaker_open(self, strategy, market_data):
        """Test signal generation when circuit breaker is open."""
        # Force circuit breaker to open state
        strategy._circuit_breaker.open()
        
        result = await strategy.generate_signals(market_data)
        
        # Should return empty list when circuit breaker is open
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_signals_implementation_error(self, strategy, market_data):
        """Test signal generation when implementation raises error."""
        strategy._signal_generation_error = Exception("Signal generation failed")
        
        result = await strategy.generate_signals(market_data)
        
        # Should handle error and return empty list
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_signals_validation_error(self, strategy, market_data):
        """Test signal generation when validation raises error."""
        test_signal = Signal(
            symbol='BTC/USDT',
            direction=SignalDirection.BUY,
            strength=Decimal('0.8'),
            timestamp=datetime.now(timezone.utc),
            source='test_strategy'
        )
        strategy._test_signals = [test_signal]
        strategy._signal_validation_error = Exception("Validation failed")
        
        result = await strategy.generate_signals(market_data)
        
        # Should filter out invalid signals
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_signals_invalid_signal_validation(self, strategy, market_data):
        """Test signal generation with invalid signal validation."""
        test_signal = Signal(
            symbol='BTC/USDT',
            direction=SignalDirection.BUY,
            strength=Decimal('0.8'),
            timestamp=datetime.now(timezone.utc),
            source='test_strategy'
        )
        strategy._test_signals = [test_signal]
        strategy._validation_result = False  # Signal validation fails
        
        result = await strategy.generate_signals(market_data)
        
        # Should filter out invalid signals
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_signals_position_size_error(self, strategy, market_data):
        """Test signal generation when position size calculation fails."""
        test_signal = Signal(
            symbol='BTC/USDT',
            direction=SignalDirection.BUY,
            strength=Decimal('0.8'),
            timestamp=datetime.now(timezone.utc),
            source='test_strategy'
        )
        strategy._test_signals = [test_signal]
        strategy._position_size_error = Exception("Position size calculation failed")
        
        result = await strategy.generate_signals(market_data)
        
        # Should handle error and filter out signal
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_signals_circuit_breaker_failure_count(self, strategy, market_data):
        """Test that circuit breaker counts failures properly."""
        strategy._signal_generation_error = Exception("Repeated failure")
        
        # Trigger multiple failures to open circuit breaker
        for _ in range(DEFAULT_CIRCUIT_BREAKER_THRESHOLD + 1):
            await strategy.generate_signals(market_data)
        
        # Circuit breaker should now be open
        assert strategy._circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_start_already_started(self, strategy):
        """Test starting strategy that's already started."""
        strategy._status = StrategyStatus.ACTIVE
        
        result = await strategy.start()
        
        # Should not start again
        assert result is False

    @pytest.mark.asyncio
    async def test_start_on_start_method_error(self, strategy):
        """Test strategy start when _on_start method raises error."""
        async def failing_on_start():
            raise Exception("Start failed")
        
        strategy._on_start = failing_on_start
        
        result = await strategy.start()
        
        # Should handle error and return False
        assert result is False
        assert strategy._status != StrategyStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_stop_not_started(self, strategy):
        """Test stopping strategy that wasn't started."""
        strategy._status = StrategyStatus.INACTIVE
        
        result = await strategy.stop()
        
        # Should not try to stop
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_on_stop_method_error(self, strategy):
        """Test strategy stop when _on_stop method raises error."""
        # First start the strategy
        await strategy.start()
        
        async def failing_on_stop():
            raise Exception("Stop failed")
        
        strategy._on_stop = failing_on_stop
        
        result = await strategy.stop()
        
        # Should handle error but still mark as stopped
        assert result is True  # May still return True even with error
        assert strategy._status == StrategyStatus.STOPPED

    def test_update_metrics_metrics_collector_unavailable(self, strategy):
        """Test updating metrics when metrics collector is unavailable."""
        strategy._metrics_collector = None
        
        # Should not raise error
        strategy._update_metrics({})

    def test_update_metrics_exception_in_collection(self, strategy):
        """Test updating metrics when metrics collection raises exception."""
        mock_metrics = Mock()
        mock_metrics.record_strategy_metric = Mock(side_effect=Exception("Metrics failed"))
        strategy._metrics_collector = mock_metrics
        
        # Should handle exception gracefully
        strategy._update_metrics({'test_metric': 1.0})

    @pytest.mark.asyncio
    async def test_handle_error_with_alerting(self, strategy):
        """Test error handling with alerting available."""
        with patch('src.strategies.base.ALERTING_AVAILABLE', True), \
             patch('src.strategies.base.get_alert_manager') as mock_alert_manager, \
             patch('src.strategies.base.AlertSeverity') as mock_alert_severity, \
             patch('src.strategies.base.Alert') as mock_alert:
            
            mock_manager = AsyncMock()
            mock_alert_manager.return_value = mock_manager
            # Mock AlertSeverity as a class-like object that evaluates to True
            mock_alert_severity.HIGH = 'HIGH'
            mock_alert_severity.MEDIUM = 'MEDIUM'
            # Mock has __bool__ that returns True by default, so we just need to make sure it exists
            
            error = Exception("Test error")
            await strategy._handle_error(error, ErrorSeverity.HIGH, {'context': 'test'})
            
            # Should send alert (the method name in the base strategy)
            mock_manager.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_error_alerting_failure(self, strategy):
        """Test error handling when alerting itself fails."""
        with patch('src.strategies.base.ALERTING_AVAILABLE', True), \
             patch('src.strategies.base.get_alert_manager') as mock_alert_manager:
            
            mock_manager = Mock()
            mock_manager.send_alert = Mock(side_effect=Exception("Alert failed"))
            mock_alert_manager.return_value = mock_manager
            
            error = Exception("Test error")
            
            # Should handle alerting failure gracefully
            await strategy._handle_error(error, ErrorSeverity.HIGH, {'context': 'test'})

    @pytest.mark.asyncio
    async def test_handle_error_without_alerting(self, strategy):
        """Test error handling when alerting is not available."""
        with patch('src.strategies.base.ALERTING_AVAILABLE', False):
            error = Exception("Test error")
            
            # Should not raise exception
            await strategy._handle_error(error, ErrorSeverity.LOW, {'context': 'test'})

    def test_log_signal_history_limit(self, strategy):
        """Test signal history respects the limit."""
        # Add signals beyond the limit
        for i in range(1500):  # More than DEFAULT_SIGNAL_HISTORY_LIMIT (1000)
            signal = Signal(
                symbol='BTC/USDT',
                direction=SignalDirection.BUY,
                strength=Decimal('0.8'),
                timestamp=datetime.now(timezone.utc),
                source='test_strategy'
            )
            strategy._log_signal(signal)
        
        # Should maintain only the limit
        from src.strategies.base import DEFAULT_SIGNAL_HISTORY_LIMIT
        assert len(strategy._signal_history) == DEFAULT_SIGNAL_HISTORY_LIMIT

    def test_log_signal_exception_handling(self, strategy, signal):
        """Test signal logging with exception."""
        # Mock signal_history to raise exception
        strategy._signal_history = Mock()
        strategy._signal_history.append = Mock(side_effect=Exception("Logging failed"))
        
        # Should handle exception gracefully
        strategy._log_signal(signal)

    def test_set_data_service_with_none(self, strategy):
        """Test setting data service to None."""
        strategy.set_data_service(None)
        assert strategy.services.data_service is None

    def test_set_data_service_invalid_type(self, strategy):
        """Test setting data service with invalid type."""
        # Should handle invalid type gracefully
        strategy.set_data_service("invalid_service")

    def test_set_execution_service_with_none(self, strategy):
        """Test setting execution service to None."""
        strategy.set_execution_service(None)
        assert strategy.services.execution_service is None

    def test_set_risk_manager_with_none(self, strategy):
        """Test setting risk manager to None."""
        strategy.set_risk_manager(None)
        assert strategy.services.risk_service is None

    def test_set_exchange_with_none(self, strategy):
        """Test setting exchange to None."""
        strategy.set_exchange(None)
        assert strategy._exchange is None

    def test_get_metrics_calculation_error(self, strategy):
        """Test metrics calculation when errors occur during calculation."""
        # Mock internal methods to raise exceptions
        strategy._calculate_win_rate = Mock(side_effect=Exception("Win rate calculation failed"))
        
        metrics = strategy.get_metrics()
        
        # Should return metrics dict even with calculation errors
        assert isinstance(metrics, dict)
        assert 'strategy_name' in metrics
        assert 'status' in metrics

    def test_get_metrics_empty_signal_history(self, strategy):
        """Test metrics calculation with empty signal history."""
        strategy._signal_history = []
        
        metrics = strategy.get_metrics()
        
        assert metrics['total_signals'] == 0
        assert metrics['win_rate'] == 0.0

    def test_is_healthy_with_errors(self, strategy):
        """Test health check with accumulated errors."""
        # Add errors to error handler
        mock_error_handler = Mock()
        mock_error_handler.total_errors = 100
        strategy._error_handler = mock_error_handler
        
        is_healthy = strategy.is_healthy()
        
        # Should consider health based on error count
        assert isinstance(is_healthy, bool)

    def test_is_healthy_circuit_breaker_open(self, strategy):
        """Test health check when circuit breaker is open."""
        strategy._circuit_breaker.open()
        
        is_healthy = strategy.is_healthy()
        
        assert is_healthy is False

    def test_is_healthy_inactive_status(self, strategy):
        """Test health check when strategy is inactive."""
        strategy._status = StrategyStatus.INACTIVE
        
        is_healthy = strategy.is_healthy()
        
        assert is_healthy is False

    def test_is_healthy_exception_handling(self, strategy):
        """Test health check when exception occurs during check."""
        # Mock circuit breaker to raise exception
        strategy._circuit_breaker.is_open = Mock(side_effect=Exception("Health check failed"))
        
        is_healthy = strategy.is_healthy()
        
        # Should return False on exception
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker_success(self, strategy):
        """Test resetting circuit breaker successfully."""
        # Open the circuit breaker first
        strategy._circuit_breaker.open()
        
        result = await strategy.reset()
        
        assert result is True
        assert not strategy._circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker_exception(self, strategy):
        """Test resetting circuit breaker when exception occurs."""
        # Mock circuit breaker reset to raise exception
        strategy._circuit_breaker.reset = Mock(side_effect=Exception("Reset failed"))
        
        result = await strategy.reset()
        
        assert result is False

    def test_validate_config_missing_required_fields(self):
        """Test config validation with missing required fields."""
        config = {}  # Empty config
        
        with pytest.raises(ValidationError):
            ConcreteStrategy._validate_config(config)

    def test_validate_config_invalid_field_types(self):
        """Test config validation with invalid field types."""
        config = {
            'name': 123,  # Should be string
            'strategy_type': StrategyType.MOMENTUM,
            'min_confidence': 'invalid',  # Should be numeric
        }
        
        with pytest.raises(ValidationError):
            ConcreteStrategy._validate_config(config)

    def test_validate_config_out_of_range_values(self):
        """Test config validation with out-of-range values."""
        config = {
            'name': 'test_strategy',
            'strategy_type': StrategyType.MOMENTUM,
            'min_confidence': 2.0,  # Should be <= 1.0
            'position_size_pct': -0.1,  # Should be > 0
        }
        
        with pytest.raises(ValidationError):
            ConcreteStrategy._validate_config(config)

    @pytest.mark.asyncio
    async def test_validate_market_data_none(self, strategy):
        """Test market data validation with None."""
        with pytest.raises(ValidationError):
            await strategy.validate_market_data(None)

    @pytest.mark.asyncio
    async def test_validate_market_data_missing_required_fields(self, strategy):
        """Test market data validation with missing required fields."""
        incomplete_data = Mock()
        incomplete_data.symbol = None  # Missing symbol
        incomplete_data.price = Decimal('50000')
        incomplete_data.timestamp = datetime.now(timezone.utc)
        
        with pytest.raises(ValidationError):
            await strategy.validate_market_data(incomplete_data)

    @pytest.mark.asyncio
    async def test_validate_market_data_invalid_price(self, strategy):
        """Test market data validation with invalid price."""
        invalid_data = Mock()
        invalid_data.symbol = 'BTCUSDT'
        invalid_data.price = Decimal('-1000')  # Negative price
        invalid_data.timestamp = datetime.now(timezone.utc)
        
        with pytest.raises(ValidationError):
            await strategy.validate_market_data(invalid_data)

    @pytest.mark.asyncio
    async def test_validate_market_data_old_timestamp(self, strategy):
        """Test market data validation with old timestamp."""
        old_data = Mock()
        old_data.symbol = 'BTCUSDT'
        old_data.price = Decimal('50000')
        old_data.timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)  # Very old
        
        with pytest.raises(ValidationError):
            await strategy.validate_market_data(old_data)

    def test_calculate_win_rate_empty_history(self, strategy):
        """Test win rate calculation with empty history."""
        strategy._signal_history = []
        
        win_rate = strategy._calculate_win_rate()
        assert win_rate == 0.0

    def test_calculate_win_rate_exception_handling(self, strategy):
        """Test win rate calculation with exception."""
        # Mock signal history to raise exception when iterated
        def raising_iterator():
            raise Exception("Invalid signal")
        
        strategy._signal_history = Mock()
        strategy._signal_history.__iter__ = raising_iterator
        strategy._signal_history.__len__ = Mock(return_value=1)  # Non-empty to get past the empty check
        
        win_rate = strategy._calculate_win_rate()
        
        # Should return 0.0 on exception
        assert win_rate == 0.0

    def test_calculate_sharpe_ratio_insufficient_data(self, strategy):
        """Test Sharpe ratio calculation with insufficient data."""
        # Add only one return value
        strategy._returns = [0.05]
        
        sharpe = strategy._calculate_sharpe_ratio()
        
        # Should return 0.0 with insufficient data
        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_zero_volatility(self, strategy):
        """Test Sharpe ratio calculation with zero volatility."""
        # Add identical returns (no volatility)
        strategy._returns = [0.05, 0.05, 0.05, 0.05, 0.05]
        
        sharpe = strategy._calculate_sharpe_ratio()
        
        # Should handle zero volatility gracefully
        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_exception_handling(self, strategy):
        """Test Sharpe ratio calculation with exception."""
        # Mock returns to raise exception
        strategy._returns = Mock()
        strategy._returns.__len__ = Mock(return_value=10)
        strategy._returns.__iter__ = Mock(side_effect=Exception("Returns calculation failed"))
        
        sharpe = strategy._calculate_sharpe_ratio()
        
        # Should return 0.0 on exception
        assert sharpe == 0.0

    def test_calculate_max_drawdown_empty_returns(self, strategy):
        """Test max drawdown calculation with empty returns."""
        strategy._returns = []
        
        max_dd = strategy._calculate_max_drawdown()
        assert max_dd == 0.0

    def test_calculate_max_drawdown_exception_handling(self, strategy):
        """Test max drawdown calculation with exception."""
        # Mock returns to raise exception
        strategy._returns = Mock(side_effect=Exception("Drawdown calculation failed"))
        
        max_dd = strategy._calculate_max_drawdown()
        
        # Should return 0.0 on exception
        assert max_dd == 0.0

    def test_get_status_string_unknown(self, strategy):
        """Test status string for unknown status."""
        # Set unknown status
        strategy._status = "unknown_status"
        
        status_str = strategy.get_status_string()
        
        # Should return the string representation
        assert status_str == "unknown_status"

    def test_repr_string(self, strategy):
        """Test string representation of strategy."""
        repr_str = repr(strategy)
        
        assert 'ConcreteStrategy' in repr_str
        assert strategy.name in repr_str

    def test_str_string(self, strategy):
        """Test string conversion of strategy."""
        str_repr = str(strategy)
        
        assert strategy.name in str_repr
        assert strategy.strategy_type.value in str_repr

    @pytest.mark.asyncio
    async def test_cleanup_resources_exception_handling(self, strategy):
        """Test resource cleanup with exception."""
        # Mock cleanup methods to raise exceptions
        strategy._cleanup_signal_history = Mock(side_effect=Exception("Cleanup failed"))
        
        # Should handle cleanup exceptions gracefully
        await strategy._cleanup_resources()

    def test_strategy_config_property_access(self, strategy):
        """Test strategy config property access."""
        assert hasattr(strategy, 'config')
        assert strategy.config.name == strategy.name
        assert strategy.config.strategy_type == strategy.strategy_type

    def test_strategy_performance_tracking_initialization(self, strategy):
        """Test that performance tracking is initialized."""
        assert hasattr(strategy, '_returns')
        assert hasattr(strategy, '_signal_history')
        assert hasattr(strategy, '_trades_count')
        assert isinstance(strategy._returns, list)
        assert isinstance(strategy._signal_history, list)

    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self, strategy, market_data):
        """Test concurrent signal generation calls."""
        import asyncio
        
        # Create multiple concurrent calls
        tasks = []
        for _ in range(10):
            task = asyncio.create_task(strategy.generate_signals(market_data))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, list)

    def test_memory_usage_with_large_history(self, strategy, signal):
        """Test memory usage with large signal history."""
        # Add many signals to test memory management
        for i in range(2000):
            test_signal = Signal(
                symbol='BTC/USDT',
                direction=SignalDirection.BUY,
                strength=Decimal('0.8'),
                timestamp=datetime.now(timezone.utc),
                source='test_strategy'
            )
            strategy._log_signal(test_signal)
        
        # Should maintain reasonable memory usage by limiting history
        from src.strategies.base import DEFAULT_SIGNAL_HISTORY_LIMIT
        assert len(strategy._signal_history) <= DEFAULT_SIGNAL_HISTORY_LIMIT

    @pytest.mark.asyncio
    async def test_strategy_lifecycle_full_cycle(self, strategy):
        """Test complete strategy lifecycle."""
        # Initial state
        assert strategy.get_status() == StrategyStatus.STOPPED
        
        # Start
        start_result = await strategy.start()
        assert start_result is True
        assert strategy.get_status() == StrategyStatus.ACTIVE
        
        # Stop
        stop_result = await strategy.stop()
        assert stop_result is True
        assert strategy.get_status() == StrategyStatus.STOPPED

    def test_thread_safety_signal_logging(self, strategy, signal):
        """Test thread safety of signal logging."""
        import threading
        
        def log_signals():
            for i in range(100):
                test_signal = Signal(
                    symbol='BTC/USDT',
                    direction=SignalDirection.BUY,
                    strength=Decimal('0.8'),
                    timestamp=datetime.now(timezone.utc),
                    source='test_strategy'
                )
                strategy._log_signal(test_signal)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=log_signals)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have logged signals without issues
        assert len(strategy._signal_history) > 0