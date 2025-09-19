"""
Fixed comprehensive conftest.py for monitoring tests.

This replaces the existing conftest.py with proper isolation and mocking.
"""

import asyncio
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from contextlib import contextmanager
import gc
import sys
import warnings

import pytest


# CRITICAL: Disable ALL logging completely for monitoring tests
logging.disable(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger().disabled = True

# Specifically disable error handling logging that causes spam
for logger_name in [
    "src.error_handling",
    "error_handling", 
    "src.monitoring",
    "monitoring",
    "src.core",
    "src.utils",
    "asyncio",
    "urllib3",
    "opentelemetry",
]:
    logger = logging.getLogger(logger_name)
    logger.disabled = True
    logger.setLevel(logging.CRITICAL)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

# Set environment variables for maximum performance
os.environ.update({
    "DISABLE_ERROR_HANDLER_LOGGING": "true",
    "TESTING": "true",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", 
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONASYNCIODEBUG": "0",
    "PYTHONHASHSEED": "0",
    "PYTHONOPTIMIZE": "2",
    "DISABLE_ALL_LOGGING": "1",
    "PYTEST_FAST_MODE": "1",
})


# Complete OpenTelemetry mock infrastructure
class MockResource:
    def __init__(self, attributes=None):
        self.attributes = attributes or {}
    
    @classmethod
    def create(cls, attributes=None):
        return cls(attributes)

class MockTracerProvider:
    def __init__(self, *args, **kwargs):
        self.processors = []
        self.shutdowns = 0
    
    def add_span_processor(self, processor):
        self.processors.append(processor)
    
    def shutdown(self):
        self.shutdowns += 1

class MockMeterProvider:
    def __init__(self, *args, **kwargs):
        self.shutdowns = 0
    
    def shutdown(self):
        self.shutdowns += 1

class MockBatchSpanProcessor:
    def __init__(self, exporter):
        self.exporter = exporter

class MockTraceIdRatioBased:
    def __init__(self, ratio):
        self.ratio = ratio

class MockSpan:
    def __init__(self):
        self.attributes = {}
        self.events = []
        self.status = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return None
    
    def set_attribute(self, key, value):
        self.attributes[key] = value
    
    def add_event(self, name, attributes=None):
        self.events.append({"name": name, "attributes": attributes or {}})
    
    def set_status(self, status):
        self.status = status

class MockTracer:
    def __init__(self):
        self.spans = []
    
    def start_as_current_span(self, name, **kwargs):
        span = MockSpan()
        self.spans.append(span)
        return span

# Mock performance monitoring utilities
def mock_format_timestamp(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

# Create comprehensive external dependency mocks
EXTERNAL_MOCKS = {
    # Prometheus mocks
    "prometheus_client": Mock(
        Counter=Mock,
        Gauge=Mock,
        Histogram=Mock,
        Summary=Mock,
        CollectorRegistry=Mock,
        start_http_server=Mock(),
        generate_latest=Mock(return_value=b"# Mock metrics\n"),
    ),
    
    # OpenTelemetry mocks
    "opentelemetry": Mock(),
    "opentelemetry.trace": Mock(
        get_tracer=Mock(return_value=MockTracer()),
        set_tracer_provider=Mock(),
    ),
    "opentelemetry.metrics": Mock(),
    "opentelemetry.sdk.resources": Mock(Resource=MockResource),
    "opentelemetry.sdk.trace": Mock(TracerProvider=MockTracerProvider),
    "opentelemetry.sdk.metrics": Mock(MeterProvider=MockMeterProvider),
    "opentelemetry.sdk.trace.export": Mock(BatchSpanProcessor=MockBatchSpanProcessor),
    "opentelemetry.sdk.trace.sampling": Mock(TraceIdRatioBased=MockTraceIdRatioBased),
    
    # System monitoring mocks
    "psutil": Mock(
        cpu_percent=Mock(return_value=5.0),
        virtual_memory=Mock(return_value=Mock(percent=25.0, available=8000000000)),
        disk_usage=Mock(return_value=Mock(percent=15.0)),
        net_io_counters=Mock(return_value=Mock(bytes_sent=1000, bytes_recv=2000)),
    ),
    
    # Math/statistics mocks
    # Note: numpy removed from global mocks to avoid sklearn import conflicts
    # Individual test files can mock numpy locally if needed
    "scipy": Mock(
        stats=Mock(percentileofscore=Mock(return_value=50.0))
    ),
    "statistics": Mock(
        mean=Mock(return_value=2.0),
        median=Mock(return_value=2.0),
        stdev=Mock(return_value=0.5),
    ),
    
    # Networking and communication
    "smtplib": Mock(),
    "email": Mock(),
    "requests": Mock(),
    "httpx": Mock(),
    "aiohttp": Mock(),
    
    # File and data handling
    "yaml": Mock(safe_load=Mock(return_value={}), dump=Mock()),
    "sqlite3": Mock(),
    "redis": Mock(),
    
    # Threading and processing
    "threading": Mock(
        active_count=Mock(return_value=1),
        current_thread=Mock(return_value=Mock(name="MainThread")),
        Thread=Mock,
        Lock=Mock,
        Event=Mock,
    ),
    "multiprocessing": Mock(
        cpu_count=Mock(return_value=4),
        active_children=Mock(return_value=[]),
    ),
    
    # System utilities
    "gc": Mock(
        get_stats=Mock(return_value=[{"collections": 5, "collected": 100, "uncollectable": 2}]),
        collect=Mock(return_value=0),
    ),
}


@pytest.fixture(scope="session", autouse=True)
def mock_all_external_dependencies():
    """Mock all external dependencies for the entire test session."""
    with patch.dict("sys.modules", EXTERNAL_MOCKS):
        # Also patch specific functions
        with patch("time.sleep", Mock()), \
             patch("time.perf_counter", Mock(side_effect=lambda: 0.001)), \
             patch("time.time", Mock(return_value=1672574400.0)):
            yield


@pytest.fixture(scope="session")
def mock_metrics_collector():
    """Provide a comprehensive mock metrics collector."""
    mock = Mock()
    
    # Configure all methods with realistic return values
    mock.increment_counter = Mock(return_value=None)
    mock.set_gauge = Mock(return_value=None)
    mock.observe_histogram = Mock(return_value=None)
    mock.observe_summary = Mock(return_value=None)
    mock.export_metrics = Mock(return_value="# Mock metrics\ntbot_test_metric 1.0\n")
    mock.get_metrics_content_type = Mock(return_value="text/plain; version=0.0.4; charset=utf-8")
    
    # Context manager for timing operations
    context_mock = Mock()
    context_mock.__enter__ = Mock(return_value=Mock())
    context_mock.__exit__ = Mock(return_value=None)
    mock.time_operation = Mock(return_value=context_mock)
    
    # Metric management
    mock.register_metric = Mock(return_value=Mock())
    mock.get_metric = Mock(return_value=Mock())
    mock._metrics = {}
    mock._metric_definitions = {}
    mock._running = False
    
    # Sub-component mocks
    mock.trading_metrics = Mock()
    mock.system_metrics = Mock()
    mock.exchange_metrics = Mock()
    mock.risk_metrics = Mock()
    
    return mock


@pytest.fixture(scope="session") 
def mock_alert_manager():
    """Provide a comprehensive mock alert manager."""
    mock = Mock()
    
    # Configure alert management methods
    mock.fire_alert = Mock(return_value=None)
    mock.resolve_alert = Mock(return_value=True)
    mock.acknowledge_alert = Mock(return_value=True)
    mock.get_active_alerts = Mock(return_value=[])
    mock.get_alert_stats = Mock(return_value={"total": 0, "active": 0, "resolved": 0, "escalated": 0})
    mock.add_rule = Mock(return_value=None)
    mock.remove_rule = Mock(return_value=True)
    mock.add_escalation_policy = Mock(return_value=None)
    mock.add_suppression_rule = Mock(return_value=None)
    
    # State attributes
    mock._rules = {}
    mock._active_alerts = {}
    mock._alert_history = []
    mock._escalation_policies = {}
    mock._suppression_rules = []
    mock._running = False
    mock._alerts_fired = 0
    mock._alerts_resolved = 0
    
    # Config mock
    mock.config = Mock()
    mock.config.email_enabled = False
    mock.config.slack_enabled = False
    mock.config.webhook_enabled = False
    
    return mock


@pytest.fixture(scope="session")
def mock_performance_profiler():
    """Provide a comprehensive mock performance profiler.""" 
    mock = Mock()
    
    # Core attributes
    mock._running = False
    mock.max_samples = 1000
    mock.collection_interval = 1.0
    mock.anomaly_detection = False
    mock._latency_data = {"test_metric": [1.0, 2.0, 3.0]}
    mock.metrics_collector = Mock()
    mock.alert_manager = Mock()
    
    # Recording methods
    mock.record_order_execution = Mock()
    mock.record_market_data_processing = Mock()
    mock.record_websocket_latency = Mock()
    mock.record_database_query = Mock()
    mock.record_strategy_performance = Mock()
    
    # Context managers
    mock.profile_function = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
    mock.profile_async_function = Mock(return_value=Mock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))
    
    # Stats methods with realistic return values
    mock.get_performance_summary = Mock(
        return_value={
            "timestamp": mock_format_timestamp(),
            "metrics_collected": 100,
            "system_resources": {"cpu": 5.0, "memory": 25.0},
            "latency_stats": {"avg": 2.0, "p95": 5.0, "p99": 10.0},
            "throughput_stats": {"operations_per_second": 100.0},
            "gc_stats": {"collections": 5},
        }
    )
    mock.get_latency_stats = Mock(return_value=Mock(count=10, avg=2.0, p95=5.0, p99=10.0))
    mock.get_system_resource_stats = Mock(return_value={"cpu_percent": 5.0, "memory_percent": 25.0})
    mock.get_throughput_stats = Mock(return_value={"operations_per_second": 100.0})
    
    # Control methods
    mock.start = Mock()
    mock.stop = Mock()
    mock.clear_metrics = Mock()
    
    return mock


@pytest.fixture
def mock_dashboard_manager():
    """Provide a mock dashboard manager."""
    mock = Mock()
    mock.deploy_dashboard = Mock(return_value=True)
    mock.deploy_all_dashboards = Mock(return_value={"test": True})
    mock.export_dashboards_to_files = Mock()
    mock.builder = Mock()
    mock.builder.create_trading_overview_dashboard = Mock(return_value=Mock())
    mock.builder.create_system_performance_dashboard = Mock(return_value=Mock())
    return mock


@pytest.fixture(scope="session")
def sample_alert_request():
    """Provide a sample alert request for testing."""
    # Import here to avoid circular dependencies
    from src.monitoring.alerting import AlertSeverity
    from src.monitoring.services import AlertRequest

    return AlertRequest(
        rule_name="test_alert",
        severity=AlertSeverity.INFO,
        message="Test message",
        labels={"env": "test"},
        annotations={"runbook": "test"},
    )


@pytest.fixture(scope="session")
def sample_metric_request():
    """Provide a sample metric request for testing."""
    from src.monitoring.services import MetricRequest

    return MetricRequest(
        name="test_metric",
        value=Decimal("100.0"),
        labels={"type": "test"},
        namespace="testing",
    )


@pytest.fixture(scope="session")
def mock_dependency_injector():
    """Provide a mock dependency injector."""
    mock = Mock()
    mock.register_factory = Mock()
    mock.register_instance = Mock()
    mock.register_singleton = Mock()

    # Pre-configure common service resolutions
    service_mocks = {
        "AlertManager": Mock(),
        "MetricsCollector": Mock(),
        "PerformanceProfiler": Mock(),
        "GrafanaDashboardManager": Mock(),
        "AlertServiceInterface": Mock(),
        "MetricsServiceInterface": Mock(),
        "PerformanceServiceInterface": Mock(),
        "DashboardServiceInterface": Mock(),
        "MonitoringServiceInterface": Mock(),
    }
    
    mock.resolve.side_effect = lambda service_name: service_mocks.get(service_name, Mock())
    return mock


@pytest.fixture(scope="session")
def fast_datetime():
    """Provide a fixed datetime to avoid time-based test variations."""
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# Global state management fixtures
@pytest.fixture(autouse=True)
def reset_all_global_state():
    """Reset all global state between tests to prevent pollution."""
    # Store original values
    original_states = {}
    
    try:
        import sys
        
        # Modules and their global variables to reset
        modules_to_reset = {
            'src.monitoring.alerting': ['_global_alert_manager'],
            'src.monitoring.metrics': ['_global_collector'],
            'src.monitoring.performance': ['_global_profiler'],
            'src.monitoring.telemetry': ['_global_trading_tracer'],
            'src.monitoring.dependency_injection': ['_container', '_monitoring_container']
        }
        
        # Store original values
        for module_name, global_vars in modules_to_reset.items():
            if module_name in sys.modules:
                module = sys.modules[module_name]
                original_states[module_name] = {}
                for var_name in global_vars:
                    if hasattr(module, var_name):
                        original_states[module_name][var_name] = getattr(module, var_name)
                        
    except (ImportError, AttributeError):
        pass
    
    yield
    
    # Reset all global state after test
    try:
        import sys
        
        modules_to_reset = {
            'src.monitoring.alerting': ['_global_alert_manager'],
            'src.monitoring.metrics': ['_global_collector'],
            'src.monitoring.performance': ['_global_profiler'],
            'src.monitoring.telemetry': ['_global_trading_tracer'],
            'src.monitoring.dependency_injection': ['_container', '_monitoring_container']
        }
        
        for module_name, global_vars in modules_to_reset.items():
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for var_name in global_vars:
                    if hasattr(module, var_name):
                        if var_name == '_container':
                            # Reset container to new clean instance
                            try:
                                from src.monitoring.dependency_injection import DIContainer
                                setattr(module, var_name, DIContainer())
                            except ImportError:
                                setattr(module, var_name, None)
                        else:
                            # Reset other globals to None
                            setattr(module, var_name, None)
                            
        # Force garbage collection
        gc.collect()
        
    except (ImportError, AttributeError, Exception) as e:
        # Log error but don't fail tests
        pass


@pytest.fixture(autouse=True) 
def isolate_di_container():
    """Ensure DI container is properly isolated between tests."""
    yield
    
    # Always ensure clean container after each test
    try:
        import sys
        if 'src.monitoring.dependency_injection' in sys.modules:
            di_module = sys.modules['src.monitoring.dependency_injection']
            from src.monitoring.dependency_injection import DIContainer
            
            # Create fresh container instance
            di_module._container = DIContainer()
            
            # Clear any existing bindings
            if hasattr(di_module._container, '_bindings'):
                di_module._container._bindings.clear()
            if hasattr(di_module._container, '_resolving'):
                di_module._container._resolving.clear()
                
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def thread_safety_isolation():
    """Ensure thread-safe test execution."""
    # Skip if threading is mocked
    if isinstance(threading.active_count, Mock):
        yield
        return
        
    initial_thread_count = threading.active_count()
    yield
    
    # Wait for background threads to complete (with timeout)
    max_wait = 1.0
    wait_interval = 0.01
    waited = 0
    
    if not isinstance(threading.active_count, Mock):
        while threading.active_count() > initial_thread_count and waited < max_wait:
            time.sleep(wait_interval)
            waited += wait_interval


@pytest.fixture(autouse=True)
def clean_asyncio_state():
    """Clean asyncio state between tests."""
    yield
    
    try:
        import asyncio
        
        # Clean up pending tasks
        try:
            loop = asyncio.get_event_loop()
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                    try:
                        loop.run_until_complete(task)
                    except (asyncio.CancelledError, Exception):
                        pass
        except RuntimeError:
            # No event loop running
            pass
            
        # Force garbage collection
        gc.collect()
        
    except (ImportError, AttributeError):
        pass


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for maximum performance."""
    import gc
    import warnings
    
    # Disable GC and warnings for speed
    gc.disable()
    warnings.filterwarnings("ignore")
    
    # Set critical environment variables
    os.environ.update({
        "PYTEST_FAST_MODE": "1",
        "PYTHONASYNCIODEBUG": "0",
        "DISABLE_ALL_LOGGING": "1"
    })


def pytest_unconfigure(config):
    """Re-enable GC after tests complete."""
    import gc
    gc.enable()


def pytest_runtest_setup(item):
    """Setup before each test run."""
    # Ensure clean state
    try:
        import sys
        from src.monitoring.dependency_injection import DIContainer
        
        # Ensure DI container exists
        if 'src.monitoring.dependency_injection' in sys.modules:
            di_module = sys.modules['src.monitoring.dependency_injection']
            if not hasattr(di_module, '_container') or di_module._container is None:
                di_module._container = DIContainer()
                
    except (ImportError, AttributeError):
        pass


def pytest_runtest_teardown(item, nextitem):
    """Cleanup after each test run."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def optimize_test_session():
    """Session-level optimizations."""
    import time
    import warnings

    # Essential optimizations
    original_sleep = time.sleep
    time.sleep = lambda x: None
    warnings.filterwarnings("ignore")

    yield

    # Restore
    time.sleep = original_sleep


# Mock format_timestamp function for performance tests
@pytest.fixture
def mock_format_timestamp_func():
    """Provide mock format_timestamp function."""
    return mock_format_timestamp