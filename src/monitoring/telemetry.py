"""
OpenTelemetry instrumentation and tracing for T-Bot Trading System.

This module provides comprehensive observability through OpenTelemetry including:
- Distributed tracing for request flows
- Automatic FastAPI instrumentation
- Custom trading operation spans
- Performance monitoring and profiling
- Integration with Jaeger and other backends

Key Features:
- Low-overhead tracing with sampling
- Automatic HTTP request tracing
- Database operation tracing
- Exchange API call tracing
- Custom trading operation spans
"""

from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

# Try to import OpenTelemetry components, fall back gracefully if not available
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import Span, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.semconv.trace import SpanAttributes
    from opentelemetry.trace.status import Status, StatusCode
    from opentelemetry.util.http import get_excluded_urls

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Mock classes for when OpenTelemetry is not available
    OPENTELEMETRY_AVAILABLE = False

    class MockTracer:
        def start_as_current_span(self, name, **kwargs):
            return MockSpan()

    class MockSpan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, key, value):
            pass

        def add_event(self, name, attributes=None):
            pass

        def set_status(self, status):
            pass

        def record_exception(self, exception):
            pass

    class MockTrace:
        Tracer = MockTracer

        def get_tracer(self, *args):
            return MockTracer()

        def set_tracer_provider(self, provider):
            pass

        def get_tracer_provider(self):
            return None

    trace: Any = MockTrace()
    metrics: Any = None
    Status: Any = type("MockStatus", (), {})()
    StatusCode: Any = type("MockStatusCode", (), {"OK": "ok", "ERROR": "error"})()

    # Mock other classes
    TracerProvider: Any = None
    Span: Any = None
    BatchSpanProcessor: Any = None
    ConsoleSpanExporter: Any = None
    MeterProvider: Any = None
    Resource: Any = None
    JaegerExporter: Any = None
    OTLPSpanExporter: Any = None
    FastAPIInstrumentor: Any = None
    RequestsInstrumentor: Any = None
    AioHttpClientInstrumentor: Any = None
    AsyncPGInstrumentor: Any = None
    RedisInstrumentor: Any = None
    SQLAlchemyInstrumentor: Any = None
    get_excluded_urls: Any = None
    SpanAttributes: Any = None

from src.core import MonitoringError, get_logger


from src.error_handling import (
    ErrorContext,
    get_global_error_handler,
    with_error_context,
    with_retry,
)

# Initialize logger with error handling
try:
    logger = get_logger(__name__)
except Exception:
    # Fallback to basic logging if get_logger fails
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class OpenTelemetryConfig:
    """Configuration for OpenTelemetry setup."""

    # Service information
    service_name: str = "tbot-trading-system"
    service_version: str = "1.0.0"
    service_namespace: str = "trading"
    environment: str = "development"

    # Tracing configuration
    tracing_enabled: bool = True
    sampling_rate: float = 1.0  # 100% for development, reduce for production

    # Exporters
    jaeger_enabled: bool = False  # Disabled by default - requires Jaeger server
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    otlp_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    console_enabled: bool = False  # Disabled by default to prevent log pollution

    # Instrumentation
    instrument_fastapi: bool = True
    instrument_requests: bool = True
    instrument_aiohttp: bool = True
    instrument_database: bool = True
    instrument_redis: bool = True

    # Performance
    max_span_attributes: int = 100
    max_events_per_span: int = 128
    max_links_per_span: int = 128
    max_attributes_per_event: int = 100

    # Custom attributes
    custom_resource_attributes: dict[str, str] = None

    def __post_init__(self):
        """Post-initialization setup."""
        if self.custom_resource_attributes is None:
            self.custom_resource_attributes = {}


class TradingTracer:
    """
    Custom tracer for trading operations with financial context.

    Provides specialized tracing for trading operations with relevant
    attributes and metrics for financial analysis.
    """

    def __init__(self, tracer):
        """
        Initialize trading tracer.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self._tracer = tracer
        self._active_spans: dict[str, Any] = {}

    @contextmanager
    def trace_order_execution(
        self,
        order_id: str,
        exchange: str,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None = None,
    ):
        """
        Trace order execution with trading-specific attributes.

        Args:
            order_id: Unique order identifier
            exchange: Exchange name
            symbol: Trading symbol
            order_type: Order type (market, limit, etc.)
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Order price (if applicable)
        """
        with self._tracer.start_as_current_span(
            "trading.order.execute",
            attributes={
                "trading.order.id": order_id,
                "trading.exchange": exchange,
                "trading.symbol": symbol,
                "trading.order.type": order_type,
                "trading.order.side": side,
                "trading.order.quantity": quantity,
                "trading.order.price": price or 0.0,
            },
        ) as span:
            span.set_attribute("operation.type", "order_execution")
            yield span

    @contextmanager
    def trace_strategy_execution(
        self, strategy_name: str, symbol: str, action: str, confidence: float | None = None
    ):
        """
        Trace strategy execution.

        Args:
            strategy_name: Name of the trading strategy
            symbol: Trading symbol
            action: Strategy action (buy/sell/hold)
            confidence: Signal confidence (0-1)
        """
        with self._tracer.start_as_current_span(
            "trading.strategy.execute",
            attributes={
                "trading.strategy.name": strategy_name,
                "trading.symbol": symbol,
                "trading.strategy.action": action,
                "trading.strategy.confidence": confidence or 0.0,
            },
        ) as span:
            span.set_attribute("operation.type", "strategy_execution")
            yield span

    @contextmanager
    def trace_risk_check(
        self, check_type: str, symbol: str, position_size: float, portfolio_value: float
    ):
        """
        Trace risk management checks.

        Args:
            check_type: Type of risk check
            symbol: Trading symbol
            position_size: Position size
            portfolio_value: Current portfolio value
        """
        with self._tracer.start_as_current_span(
            "trading.risk.check",
            attributes={
                "trading.risk.check_type": check_type,
                "trading.symbol": symbol,
                "trading.position.size": position_size,
                "trading.portfolio.value": portfolio_value,
            },
        ) as span:
            span.set_attribute("operation.type", "risk_check")
            yield span

    @contextmanager
    def trace_market_data_processing(
        self, exchange: str, symbol: str, data_type: str, latency_ms: float | None = None
    ):
        """
        Trace market data processing.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Type of market data (ticker, orderbook, trades)
            latency_ms: Data latency in milliseconds
        """
        with self._tracer.start_as_current_span(
            "trading.market_data.process",
            attributes={
                "trading.exchange": exchange,
                "trading.symbol": symbol,
                "trading.data.type": data_type,
                "trading.data.latency_ms": latency_ms or 0.0,
            },
        ) as span:
            span.set_attribute("operation.type", "market_data_processing")
            yield span

    def add_trading_event(self, span, event_type: str, attributes: dict[str, Any] | None = None):
        """
        Add a trading-specific event to a span.

        Args:
            span: Active span
            event_type: Type of trading event
            attributes: Event attributes
        """
        event_attributes = attributes or {}
        event_attributes["trading.event.type"] = event_type
        span.add_event(f"trading.{event_type}", event_attributes)


@with_retry(max_attempts=3, backoff_factor=2.0, exceptions=(MonitoringError,))
@with_error_context("telemetry_setup")
def setup_telemetry(config: OpenTelemetryConfig) -> TradingTracer:
    """
    Setup OpenTelemetry instrumentation for the trading system.

    Args:
        config: OpenTelemetry configuration

    Returns:
        Configured TradingTracer instance

    Raises:
        MonitoringError: If telemetry setup fails
    """
    try:
        error_handler = get_global_error_handler()
        # Create resource with service information
        resource_attributes = {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "service.namespace": config.service_namespace,
            "deployment.environment": config.environment,
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        }

        # Add custom resource attributes
        resource_attributes.update(config.custom_resource_attributes)

        resource = Resource.create(resource_attributes)

        # Setup tracing
        if config.tracing_enabled:
            tracer_provider = TracerProvider(
                resource=resource, sampler=TraceIdRatioBased(config.sampling_rate)
            )

            # Add span processors/exporters
            exporters = []

            # Jaeger exporter
            if config.jaeger_enabled:
                try:
                    jaeger_exporter = JaegerExporter(
                        agent_host_name="localhost",
                        agent_port=6831,
                        collector_endpoint=config.jaeger_endpoint,
                    )
                    exporters.append(jaeger_exporter)
                    logger.info("Jaeger exporter configured")
                except (ImportError, ConnectionError, OSError) as e:
                    if error_handler:
                        error_handler.handle_error_sync(
                            e,
                            component="Telemetry",
                            operation="configure_jaeger_exporter",
                            details={"error_type": type(e).__name__},
                        )
                    logger.warning(f"Failed to configure Jaeger exporter: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error configuring Jaeger exporter: {e}")
                    if error_handler:
                        error_handler.handle_error_sync(
                            e,
                            component="Telemetry",
                            operation="configure_jaeger_exporter",
                            details={
                                "error_type": "unexpected",
                                "error_class": type(e).__name__,
                            },
                        )

            # OTLP exporter
            if config.otlp_enabled:
                try:
                    otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=True)
                    exporters.append(otlp_exporter)
                    logger.info("OTLP exporter configured")
                except Exception as e:
                    if error_handler:
                        error_handler.handle_error_sync(
                            e,
                            component="Telemetry",
                            operation="configure_otlp_exporter",
                        )
                    logger.warning(f"Failed to configure OTLP exporter: {e}")

            # Console exporter (for debugging)
            if config.console_enabled:
                console_exporter = ConsoleSpanExporter()
                exporters.append(console_exporter)
                logger.info("Console exporter configured")

            # Add exporters to tracer provider
            for exporter in exporters:
                span_processor = BatchSpanProcessor(exporter)
                tracer_provider.add_span_processor(span_processor)

            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)

            logger.info(f"OpenTelemetry tracing configured with {len(exporters)} exporters")

        # Setup metrics (placeholder for future implementation)
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)

        # Get tracer
        tracer = trace.get_tracer(config.service_name, config.service_version)

        # Setup automatic instrumentation
        _setup_auto_instrumentation(config)

        # Create trading tracer
        trading_tracer = TradingTracer(tracer)

        logger.info("OpenTelemetry setup completed successfully")
        return trading_tracer

    except Exception as e:
        if error_handler:
            error_handler.handle_error_sync(
                e,
                component="Telemetry",
                operation="setup_telemetry",
                service_name=config.service_name,
            )
        raise MonitoringError(
            f"Failed to setup OpenTelemetry: {e}", error_code="MONITORING_003"
        ) from e


def _setup_auto_instrumentation(config: OpenTelemetryConfig) -> None:
    """
    Setup automatic instrumentation for various libraries.

    Args:
        config: OpenTelemetry configuration
    """
    try:
        # FastAPI instrumentation
        if config.instrument_fastapi:
            # Get excluded URLs safely with deprecation handling
            try:
                # Handle both old and new API formats
                if hasattr(FastAPIInstrumentor(), "instrument"):
                    excluded_urls = get_excluded_urls("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS") or ""

                    # Convert to proper format based on OpenTelemetry version
                    try:
                        if hasattr(excluded_urls, "split"):
                            excluded_urls_param = excluded_urls
                        else:
                            # Handle ExcludeList or other objects
                            excluded_urls_param = str(excluded_urls) if excluded_urls else ""

                        FastAPIInstrumentor().instrument(excluded_urls=excluded_urls_param)
                    except TypeError:
                        # Fallback for newer versions that may have different parameter format
                        try:
                            FastAPIInstrumentor().instrument()
                        except Exception as fallback_error:
                            logger.debug(
                                f"FastAPI instrumentation fallback failed: {fallback_error}"
                            )

            except Exception as e:
                logger.debug(f"FastAPI instrumentation configuration failed: {e}")
                # Try basic instrumentation without excluded URLs
                try:
                    FastAPIInstrumentor().instrument()
                except Exception as basic_error:
                    logger.warning(f"Basic FastAPI instrumentation failed: {basic_error}")
            logger.debug("FastAPI instrumentation enabled")

        # HTTP client instrumentation
        if config.instrument_requests:
            RequestsInstrumentor().instrument()
            logger.debug("Requests instrumentation enabled")

        if config.instrument_aiohttp:
            AioHttpClientInstrumentor().instrument()
            logger.debug("AioHTTP client instrumentation enabled")

        # Database instrumentation
        if config.instrument_database:
            try:
                AsyncPGInstrumentor().instrument()
                logger.debug("AsyncPG instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument AsyncPG: {e}")

            try:
                SQLAlchemyInstrumentor().instrument()
                logger.debug("SQLAlchemy instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument SQLAlchemy: {e}")

        # Redis instrumentation
        if config.instrument_redis:
            try:
                RedisInstrumentor().instrument()
                logger.debug("Redis instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Redis: {e}")

    except Exception as e:
        logger.error(f"Error setting up auto instrumentation: {e}")


def get_tracer(name: str = "tbot-trading") -> trace.Tracer:
    """
    Get a tracer instance.

    Args:
        name: Tracer name

    Returns:
        OpenTelemetry tracer instance
    """
    return trace.get_tracer(name)


def instrument_fastapi(app, config: OpenTelemetryConfig):
    """
    Instrument a FastAPI application with OpenTelemetry.

    Args:
        app: FastAPI application instance
        config: OpenTelemetry configuration
    """
    try:
        if config.instrument_fastapi:
            # Get excluded URLs safely
            try:
                excluded_urls = get_excluded_urls("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS") or ""
                if hasattr(excluded_urls, "split"):
                    excluded_urls_param = excluded_urls
                else:
                    # Convert ExcludeList or other objects to string
                    excluded_urls_param = str(excluded_urls) if excluded_urls else ""
            except Exception:
                excluded_urls_param = ""

            FastAPIInstrumentor.instrument_app(
                app,
                excluded_urls=excluded_urls_param,
                tracer_provider=trace.get_tracer_provider(),
            )
            logger.info("FastAPI application instrumented with OpenTelemetry")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI app: {e}")


def trace_async_function(operation_name: str, attributes: dict[str, Any] | None = None):
    """
    Decorator to trace async functions.

    Args:
        operation_name: Name of the operation being traced
        attributes: Additional span attributes

    Returns:
        Decorated function
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_attributes = attributes or {}

            # Add function information
            span_attributes.update(
                {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }
            )

            with tracer.start_as_current_span(operation_name, attributes=span_attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def trace_function(operation_name: str, attributes: dict[str, Any] | None = None):
    """
    Decorator to trace synchronous functions.

    Args:
        operation_name: Name of the operation being traced
        attributes: Additional span attributes

    Returns:
        Decorated function
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_attributes = attributes or {}

            # Add function information
            span_attributes.update(
                {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }
            )

            with tracer.start_as_current_span(operation_name, attributes=span_attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


@asynccontextmanager
async def trace_async_context(operation_name: str, attributes: dict[str, Any] | None = None):
    """
    Async context manager for tracing operations.

    Args:
        operation_name: Name of the operation being traced
        attributes: Additional span attributes

    Yields:
        Active span
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(operation_name, attributes=attributes) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# Global trading tracer instance
_global_trading_tracer: TradingTracer | None = None


def get_trading_tracer() -> TradingTracer | None:
    """
    Get the global trading tracer instance.

    Returns:
        Global TradingTracer instance or None if not initialized
    """
    return _global_trading_tracer


def set_global_trading_tracer(tracer: TradingTracer) -> None:
    """
    Set the global trading tracer instance.

    Args:
        tracer: TradingTracer instance
    """
    global _global_trading_tracer
    _global_trading_tracer = tracer
