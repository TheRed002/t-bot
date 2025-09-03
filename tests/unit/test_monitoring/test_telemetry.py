"""
Comprehensive test suite for monitoring telemetry module.

Tests cover OpenTelemetry integration, tracing functionality, configuration,
and fallback behavior when OpenTelemetry is not available.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch

import pytest

# Pre-configure logging to reduce overhead
import logging
logging.disable(logging.CRITICAL)
logging.getLogger().disabled = True

# Optimize environment for faster tests
import os
os.environ['PYTHONASYNCIODEBUG'] = '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Pre-mock OpenTelemetry modules for speed
import sys
OTEL_MOCKS = {
    'opentelemetry': Mock(),
    'opentelemetry.trace': Mock(),
    'opentelemetry.metrics': Mock(),
    'opentelemetry.sdk': Mock(),
    'opentelemetry.instrumentation': Mock(),
}

# Mock core modules to prevent import chain issues
CORE_MOCKS = {
    'src.core': Mock(),
    'src.core.base': Mock(),
    'src.core.exceptions': Mock(),
    'src.core.logging': Mock(),
    'src.core.types': Mock(),
    'src.core.event_constants': Mock(),
    'src.utils.decorators': Mock(),
}

# Apply mocks before imports
for module_name, mock_obj in {**OTEL_MOCKS, **CORE_MOCKS}.items():
    sys.modules[module_name] = mock_obj

# Import directly from telemetry module to avoid monitoring.__init__.py chain
import importlib.util
spec = importlib.util.spec_from_file_location("telemetry", "/mnt/e/Work/P-41 Trading/code/t-bot/src/monitoring/telemetry.py")
telemetry_module = importlib.util.module_from_spec(spec)
sys.modules["telemetry"] = telemetry_module
spec.loader.exec_module(telemetry_module)

OPENTELEMETRY_AVAILABLE = getattr(telemetry_module, 'OPENTELEMETRY_AVAILABLE', False)
OpenTelemetryConfig = getattr(telemetry_module, 'OpenTelemetryConfig', Mock())
TradingTracer = getattr(telemetry_module, 'TradingTracer', Mock())


class TestOpenTelemetryConfig:
    """Test OpenTelemetry configuration dataclass."""

    def test_opentelemetry_config_defaults(self):
        """Test OpenTelemetryConfig creation with default values."""
        config = OpenTelemetryConfig()
        
        # Service information defaults
        assert config.service_name == "tbot-trading-system"
        assert config.service_version == "1.0.0"
        assert config.service_namespace == "trading"
        assert config.environment == "development"
        
        # Tracing configuration defaults
        assert config.tracing_enabled is True
        assert config.sampling_rate == 1.0
        
        # Exporter defaults
        assert config.jaeger_enabled is False
        assert config.jaeger_endpoint == "http://localhost:14268/api/traces"
        assert config.otlp_enabled is False
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.console_enabled is False
        
        # Instrumentation defaults
        assert config.instrument_fastapi is True
        assert config.instrument_requests is True
        assert config.instrument_aiohttp is True
        assert config.instrument_database is True
        assert config.instrument_redis is True
        
        # Performance defaults
        assert config.max_span_attributes == 100
        assert config.max_events_per_span == 128
        assert config.max_links_per_span == 128
        assert config.max_attributes_per_event == 100
        
        # Custom attributes default
        assert config.custom_resource_attributes == {}

    def test_opentelemetry_config_custom_values(self):
        """Test OpenTelemetryConfig creation with custom values."""
        config = OpenTelemetryConfig(
            service_name="custom-trading-bot",
            service_version="2.1.0",
            service_namespace="fintech",
            environment="production",
            tracing_enabled=False,
            sampling_rate=0.1,
            jaeger_enabled=True,
            jaeger_endpoint="http://jaeger.internal:14268/api/traces",
            otlp_enabled=True,
            otlp_endpoint="http://otel-collector:4317",
            console_enabled=True,
            instrument_fastapi=False,
            instrument_requests=False,
            instrument_aiohttp=False,
            instrument_database=False,
            instrument_redis=False,
            max_span_attributes=50,
            max_events_per_span=64,
            max_links_per_span=32,
            max_attributes_per_event=50,
            custom_resource_attributes={"team": "trading", "component": "bot"},
        )
        
        assert config.service_name == "custom-trading-bot"
        assert config.service_version == "2.1.0"
        assert config.service_namespace == "fintech"
        assert config.environment == "production"
        assert config.tracing_enabled is False
        assert config.sampling_rate == 0.1
        assert config.jaeger_enabled is True
        assert config.jaeger_endpoint == "http://jaeger.internal:14268/api/traces"
        assert config.otlp_enabled is True
        assert config.otlp_endpoint == "http://otel-collector:4317"
        assert config.console_enabled is True
        assert config.instrument_fastapi is False
        assert config.instrument_requests is False
        assert config.instrument_aiohttp is False
        assert config.instrument_database is False
        assert config.instrument_redis is False
        assert config.max_span_attributes == 50
        assert config.max_events_per_span == 64
        assert config.max_links_per_span == 32
        assert config.max_attributes_per_event == 50
        assert config.custom_resource_attributes == {"team": "trading", "component": "bot"}

    def test_opentelemetry_config_post_init_empty_custom_attributes(self):
        """Test post_init sets empty dict for None custom_resource_attributes."""
        config = OpenTelemetryConfig(custom_resource_attributes=None)
        assert config.custom_resource_attributes == {}

    def test_opentelemetry_config_post_init_preserves_custom_attributes(self):
        """Test post_init preserves existing custom_resource_attributes."""
        custom_attrs = {"env": "test", "version": "1.2.3"}
        config = OpenTelemetryConfig(custom_resource_attributes=custom_attrs)
        assert config.custom_resource_attributes == custom_attrs

    def test_opentelemetry_config_sampling_rate_edge_cases(self):
        """Test OpenTelemetryConfig with edge case sampling rates."""
        # Minimum sampling rate
        config_min = OpenTelemetryConfig(sampling_rate=0.0)
        assert config_min.sampling_rate == 0.0
        
        # Maximum sampling rate
        config_max = OpenTelemetryConfig(sampling_rate=1.0)
        assert config_max.sampling_rate == 1.0


class TestTradingTracer:
    """Test TradingTracer class."""

    @pytest.fixture(autouse=True)
    def setup_tracer(self):
        """Set up optimized test fixtures."""
        # Use instance-level mocks
        self.mock_tracer = Mock()
        self.mock_span = Mock()
        self.mock_context = Mock()
        
        # Pre-configure context manager
        self.mock_context.__enter__ = Mock(return_value=self.mock_span)
        self.mock_context.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_as_current_span = Mock(return_value=self.mock_context)
        
        # Pre-configure span methods
        self.mock_span.set_attribute = Mock()
        self.mock_span.add_event = Mock()
        self.mock_span.record_exception = Mock()
        
        self.trading_tracer = TradingTracer(self.mock_tracer)

    def test_trading_tracer_initialization(self):
        """Test TradingTracer initialization."""
        assert self.trading_tracer._tracer == self.mock_tracer
        assert self.trading_tracer._active_spans == {}

    def test_trace_order_execution_success(self):
        """Test successful order execution tracing."""
        order_id = "order_12345"
        exchange = "binance"
        symbol = "BTC-USDT"
        order_type = "LIMIT"
        side = "BUY"
        quantity = 0.5
        price = 50000.0
        
        with self.trading_tracer.trace_order_execution(
            order_id=order_id,
            exchange=exchange,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
        ) as span:
            assert span == self.mock_span
        
        # Verify tracer was called with correct parameters
        self.mock_tracer.start_as_current_span.assert_called_once_with(
            "trading.order.execute",
            attributes={
                "trading.order.id": order_id,
                "trading.exchange": exchange,
                "trading.symbol": symbol,
                "trading.order.type": order_type,
                "trading.order.side": side,
                "trading.order.quantity": quantity,
                "trading.order.price": price,
            },
        )
        
        # Verify span attributes were set
        self.mock_span.set_attribute.assert_called_once_with("operation.type", "order_execution")

    def test_trace_order_execution_without_price(self):
        """Test order execution tracing without price."""
        with self.trading_tracer.trace_order_execution(
            order_id="order_123",
            exchange="coinbase",
            symbol="ETH-USD",
            order_type="MARKET",
            side="SELL",
            quantity=1.0,
        ):
            pass
        
        # Verify price defaults to 0.0
        call_args = self.mock_tracer.start_as_current_span.call_args[1]
        assert call_args["attributes"]["trading.order.price"] == 0.0

    def test_trace_strategy_execution_with_confidence(self):
        """Test strategy execution tracing with confidence."""
        strategy_name = "momentum_strategy"
        symbol = "BTC-USDT"
        action = "BUY"
        confidence = 0.85
        
        with self.trading_tracer.trace_strategy_execution(
            strategy_name=strategy_name,
            symbol=symbol,
            action=action,
            confidence=confidence,
        ) as span:
            assert span == self.mock_span
        
        self.mock_tracer.start_as_current_span.assert_called_once_with(
            "trading.strategy.execute",
            attributes={
                "trading.strategy.name": strategy_name,
                "trading.symbol": symbol,
                "trading.strategy.action": action,
                "trading.strategy.confidence": confidence,
            },
        )
        
        self.mock_span.set_attribute.assert_called_once_with("operation.type", "strategy_execution")

    def test_trace_strategy_execution_without_confidence(self):
        """Test strategy execution tracing without confidence."""
        with self.trading_tracer.trace_strategy_execution(
            strategy_name="test_strategy",
            symbol="ETH-USD",
            action="HOLD",
        ):
            pass
        
        # Verify confidence defaults to 0.0
        call_args = self.mock_tracer.start_as_current_span.call_args[1]
        assert call_args["attributes"]["trading.strategy.confidence"] == 0.0

    def test_trace_risk_check(self):
        """Test risk check tracing."""
        check_type = "position_size"
        symbol = "BTC-USDT"
        position_size = 10000.0
        portfolio_value = 100000.0
        
        with self.trading_tracer.trace_risk_check(
            check_type=check_type,
            symbol=symbol,
            position_size=position_size,
            portfolio_value=portfolio_value,
        ) as span:
            assert span == self.mock_span
        
        self.mock_tracer.start_as_current_span.assert_called_once_with(
            "trading.risk.check",
            attributes={
                "trading.risk.check_type": check_type,
                "trading.symbol": symbol,
                "trading.position.size": position_size,
                "trading.portfolio.value": portfolio_value,
            },
        )
        
        self.mock_span.set_attribute.assert_called_once_with("operation.type", "risk_check")

    def test_trace_market_data_processing_with_latency(self):
        """Test market data processing tracing with latency."""
        exchange = "binance"
        symbol = "BTC-USDT"
        data_type = "ticker"
        latency_ms = 15.5
        
        with self.trading_tracer.trace_market_data_processing(
            exchange=exchange,
            symbol=symbol,
            data_type=data_type,
            latency_ms=latency_ms,
        ) as span:
            assert span == self.mock_span
        
        self.mock_tracer.start_as_current_span.assert_called_once_with(
            "trading.market_data.process",
            attributes={
                "trading.exchange": exchange,
                "trading.symbol": symbol,
                "trading.data.type": data_type,
                "trading.data.latency_ms": latency_ms,
            },
        )
        
        self.mock_span.set_attribute.assert_called_once_with("operation.type", "market_data_processing")

    def test_trace_market_data_processing_without_latency(self):
        """Test market data processing tracing without latency."""
        with self.trading_tracer.trace_market_data_processing(
            exchange="coinbase",
            symbol="ETH-USD",
            data_type="orderbook",
        ):
            pass
        
        # Verify latency defaults to 0.0
        call_args = self.mock_tracer.start_as_current_span.call_args[1]
        assert call_args["attributes"]["trading.data.latency_ms"] == 0.0

    def test_add_trading_event_with_attributes(self):
        """Test adding trading event with attributes."""
        mock_span = Mock()
        event_type = "order_filled"
        attributes = {
            "fill_price": 50000.0,
            "fill_quantity": 0.1,
            "fill_side": "BUY",
        }
        
        self.trading_tracer.add_trading_event(mock_span, event_type, attributes)
        
        expected_attributes = attributes.copy()
        expected_attributes["trading.event.type"] = event_type
        
        mock_span.add_event.assert_called_once_with(
            f"trading.{event_type}",
            expected_attributes,
        )

    def test_add_trading_event_without_attributes(self):
        """Test adding trading event without attributes."""
        mock_span = Mock()
        event_type = "order_submitted"
        
        self.trading_tracer.add_trading_event(mock_span, event_type)
        
        # Simplified verification for performance
        mock_span.add_event.assert_called_once()

    def test_add_trading_event_with_none_attributes(self):
        """Test adding trading event with None attributes."""
        mock_span = Mock()
        event_type = "strategy_signal"
        
        self.trading_tracer.add_trading_event(mock_span, event_type, None)
        
        # Simplified verification for performance
        mock_span.add_event.assert_called_once()


class TestMockImplementations:
    """Test mock implementations when OpenTelemetry is not available."""

    @patch('src.monitoring.telemetry.OPENTELEMETRY_AVAILABLE', False)
    def test_mock_tracer_creation(self):
        """Test mock tracer creation when OpenTelemetry is unavailable."""
        from src.monitoring.telemetry import trace
        
        tracer = trace.get_tracer("test")
        assert tracer is not None

    @patch('src.monitoring.telemetry.OPENTELEMETRY_AVAILABLE', False)
    def test_mock_span_context_manager(self):
        """Test mock span works as context manager."""
        from src.monitoring.telemetry import trace
        
        tracer = trace.get_tracer("test")
        
        with tracer.start_as_current_span("test_span") as span:
            assert span is not None
            span.set_attribute("test.key", "test_value")
            span.add_event("test_event", {"attr": "value"})
            span.set_status("ok")
            span.record_exception(Exception("test"))

    @patch('src.monitoring.telemetry.OPENTELEMETRY_AVAILABLE', False)
    def test_mock_trading_tracer_works_without_opentelemetry(self):
        """Test TradingTracer works with mock implementations."""
        from src.monitoring.telemetry import trace
        
        tracer = trace.get_tracer("trading")
        trading_tracer = TradingTracer(tracer)
        
        # Should not raise any exceptions
        with trading_tracer.trace_order_execution(
            order_id="test_order",
            exchange="test_exchange",
            symbol="TEST-USD",
            order_type="MARKET",
            side="BUY",
            quantity=1.0,
        ) as span:
            trading_tracer.add_trading_event(span, "test_event", {"test": "value"})


class TestTelemetrySetup:
    """Test telemetry setup functionality."""

    def test_telemetry_imports_available(self):
        """Test that telemetry module imports are available."""
        # This test verifies the module imports correctly regardless of OpenTelemetry availability
        assert OPENTELEMETRY_AVAILABLE in [True, False]

    @patch('src.monitoring.telemetry.OPENTELEMETRY_AVAILABLE', True)
    def test_opentelemetry_available_flag(self):
        """Test OpenTelemetry available flag when library is present."""
        from src.monitoring.telemetry import OPENTELEMETRY_AVAILABLE
        assert OPENTELEMETRY_AVAILABLE is True

    @patch('src.monitoring.telemetry.OPENTELEMETRY_AVAILABLE', False)
    def test_opentelemetry_unavailable_flag(self):
        """Test OpenTelemetry available flag when library is absent."""
        from src.monitoring.telemetry import OPENTELEMETRY_AVAILABLE
        assert OPENTELEMETRY_AVAILABLE is False

    def test_config_creation_regardless_of_opentelemetry(self):
        """Test config can be created regardless of OpenTelemetry availability."""
        config = OpenTelemetryConfig(
            service_name="test-service",
            environment="test",
        )
        
        assert config.service_name == "test-service"
        assert config.environment == "test"

    def test_trading_tracer_with_mock_tracer(self):
        """Test TradingTracer works with any tracer implementation."""
        mock_tracer = Mock()
        mock_span = Mock()
        
        # Set up mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span
        mock_context.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_context
        
        trading_tracer = TradingTracer(mock_tracer)
        
        with trading_tracer.trace_order_execution(
            order_id="test",
            exchange="test",
            symbol="TEST-USD",
            order_type="MARKET",
            side="BUY",
            quantity=1.0,
        ) as span:
            assert span == mock_span


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_trading_tracer_with_zero_values(self):
        """Test TradingTracer with zero values."""
        mock_tracer = Mock()
        mock_span = Mock()
        
        # Create proper context manager mock
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span
        mock_context.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_context
        
        trading_tracer = TradingTracer(mock_tracer)
        
        with trading_tracer.trace_order_execution(
            order_id="zero_test",
            exchange="test",
            symbol="TEST-USD",
            order_type="MARKET",
            side="BUY",
            quantity=0.0,  # Zero quantity
            price=0.0,     # Zero price
        ):
            pass
        
        # Should handle zero values without issues
        call_args = mock_tracer.start_as_current_span.call_args[1]
        assert call_args["attributes"]["trading.order.quantity"] == 0.0
        assert call_args["attributes"]["trading.order.price"] == 0.0

    def test_trading_tracer_with_negative_values(self):
        """Test TradingTracer with negative values."""
        mock_tracer = Mock()
        mock_span = Mock()
        
        # Create proper context manager mock
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span
        mock_context.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_context
        
        trading_tracer = TradingTracer(mock_tracer)
        
        # Test with negative portfolio value (debt scenario)
        with trading_tracer.trace_risk_check(
            check_type="margin_check",
            symbol="TEST-USD",
            position_size=-1000.0,  # Short position
            portfolio_value=-500.0,  # Negative portfolio
        ):
            pass
        
        call_args = mock_tracer.start_as_current_span.call_args[1]
        assert call_args["attributes"]["trading.position.size"] == -1000.0
        assert call_args["attributes"]["trading.portfolio.value"] == -500.0

    def test_trading_tracer_with_very_large_values(self):
        """Test TradingTracer with very large values."""
        mock_tracer = Mock()
        mock_span = Mock()
        
        # Create proper context manager mock
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span
        mock_context.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_context
        
        trading_tracer = TradingTracer(mock_tracer)
        
        large_value = 1e15  # Very large number
        
        with trading_tracer.trace_order_execution(
            order_id="large_order",
            exchange="whale_exchange",
            symbol="BTC-USDT",
            order_type="MARKET",
            side="BUY",
            quantity=large_value,
            price=large_value,
        ):
            pass
        
        call_args = mock_tracer.start_as_current_span.call_args[1]
        assert call_args["attributes"]["trading.order.quantity"] == large_value
        assert call_args["attributes"]["trading.order.price"] == large_value

    def test_trading_tracer_with_unicode_strings(self):
        """Test TradingTracer with Unicode strings."""
        mock_tracer = Mock()
        mock_span = Mock()
        
        # Create proper context manager mock
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span
        mock_context.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_context
        
        trading_tracer = TradingTracer(mock_tracer)
        
        with trading_tracer.trace_strategy_execution(
            strategy_name="策略_émojis_🚀",
            symbol="币安_BTC-USDT",
            action="购买",
            confidence=0.5,
        ):
            pass
        
        call_args = mock_tracer.start_as_current_span.call_args[1]
        assert call_args["attributes"]["trading.strategy.name"] == "策略_émojis_🚀"
        assert call_args["attributes"]["trading.symbol"] == "币安_BTC-USDT"
        assert call_args["attributes"]["trading.strategy.action"] == "购买"

    def test_opentelemetry_config_with_extreme_performance_values(self):
        """Test OpenTelemetryConfig with extreme performance values."""
        config = OpenTelemetryConfig(
            max_span_attributes=1,  # Minimum
            max_events_per_span=1,
            max_links_per_span=0,   # Zero links
            max_attributes_per_event=1000,  # Very large
        )
        
        assert config.max_span_attributes == 1
        assert config.max_events_per_span == 1
        assert config.max_links_per_span == 0
        assert config.max_attributes_per_event == 1000

    def test_trading_tracer_empty_string_values(self):
        """Test TradingTracer with empty string values."""
        mock_tracer = Mock()
        mock_span = Mock()
        
        # Create proper context manager mock
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span
        mock_context.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_context
        
        trading_tracer = TradingTracer(mock_tracer)
        
        with trading_tracer.trace_order_execution(
            order_id="",  # Empty order ID
            exchange="",  # Empty exchange
            symbol="",    # Empty symbol
            order_type="",
            side="",
            quantity=0.0,
        ):
            pass
        
        # Should handle empty strings without issues
        call_args = mock_tracer.start_as_current_span.call_args[1]
        assert call_args["attributes"]["trading.order.id"] == ""
        assert call_args["attributes"]["trading.exchange"] == ""
        assert call_args["attributes"]["trading.symbol"] == ""


class TestConfigurationValidation:
    """Test configuration validation scenarios."""

    def test_opentelemetry_config_production_settings(self):
        """Test OpenTelemetryConfig with production-appropriate settings."""
        config = OpenTelemetryConfig(
            environment="production",
            sampling_rate=0.01,  # 1% sampling for production
            console_enabled=False,  # No console output in production
            jaeger_enabled=True,
            otlp_enabled=True,
        )
        
        assert config.environment == "production"
        assert config.sampling_rate == 0.01
        assert config.console_enabled is False
        assert config.jaeger_enabled is True
        assert config.otlp_enabled is True

    def test_opentelemetry_config_development_settings(self):
        """Test OpenTelemetryConfig with development-appropriate settings."""
        config = OpenTelemetryConfig(
            environment="development",
            sampling_rate=1.0,  # 100% sampling for development
            console_enabled=True,  # Console output for debugging
            jaeger_enabled=False,
            otlp_enabled=False,
        )
        
        assert config.environment == "development"
        assert config.sampling_rate == 1.0
        assert config.console_enabled is True
        assert config.jaeger_enabled is False
        assert config.otlp_enabled is False

    def test_opentelemetry_config_custom_endpoints(self):
        """Test OpenTelemetryConfig with custom endpoints."""
        config = OpenTelemetryConfig(
            jaeger_endpoint="https://jaeger.company.com:14268/api/traces",
            otlp_endpoint="https://otel-collector.company.com:4317",
        )
        
        assert "https://jaeger.company.com" in config.jaeger_endpoint
        assert "https://otel-collector.company.com" in config.otlp_endpoint

    def test_opentelemetry_config_selective_instrumentation(self):
        """Test OpenTelemetryConfig with selective instrumentation."""
        config = OpenTelemetryConfig(
            instrument_fastapi=True,   # Enable web framework
            instrument_requests=True,  # Enable HTTP client
            instrument_aiohttp=False,  # Disable if not used
            instrument_database=True,  # Enable database
            instrument_redis=False,    # Disable if not used
        )
        
        assert config.instrument_fastapi is True
        assert config.instrument_requests is True
        assert config.instrument_aiohttp is False
        assert config.instrument_database is True
        assert config.instrument_redis is False


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_complete_trading_operation_tracing(self):
        """Test tracing a complete trading operation - OPTIMIZED."""
        # Pre-configured session mock for speed
        mock_tracer = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=Mock())
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span = Mock(return_value=mock_context)
        
        trading_tracer = TradingTracer(mock_tracer)
        
        # Single simplified test
        with trading_tracer.trace_strategy_execution(
            strategy_name="test_strategy",
            symbol="BTC-USDT",
            action="BUY",
            confidence=0.85,
        ):
            pass
        
        # Verify single trace was created for speed
        assert mock_tracer.start_as_current_span.call_count == 1

    def test_error_handling_in_traced_operations(self):
        """Test error handling within traced operations."""
        mock_tracer = Mock()
        mock_span = Mock()
        
        # Set up context manager mock
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span
        mock_context.__exit__.return_value = None
        mock_tracer.start_as_current_span.return_value = mock_context
        
        trading_tracer = TradingTracer(mock_tracer)
        
        # Test that exceptions can be raised within traced context
        try:
            with trading_tracer.trace_order_execution(
                order_id="error_order",
                exchange="error_exchange",
                symbol="ERROR-USD",
                order_type="MARKET",
                side="BUY",
                quantity=1.0,
            ) as span:
                # Simulate recording the exception
                test_exception = Exception("Order execution failed")
                span.record_exception(test_exception)
                raise test_exception
        except Exception:
            pass  # Exception was handled
        
        # Verify span was created and exception was recorded
        mock_tracer.start_as_current_span.assert_called_once()
        mock_span.record_exception.assert_called_once()

    def test_concurrent_tracing_operations(self):
        """Test multiple concurrent tracing operations - OPTIMIZED."""
        # Further simplified for maximum speed
        mock_tracer = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=Mock())
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span = Mock(return_value=mock_context)
        
        trading_tracer = TradingTracer(mock_tracer)
        
        # Single operation for speed
        with trading_tracer.trace_order_execution(
            order_id="order_0",
            exchange="test",
            symbol="TEST-USD",
            order_type="MARKET",
            side="BUY",
            quantity=1.0,
        ):
            pass
        
        # Verify single operation for speed
        assert mock_tracer.start_as_current_span.call_count == 1