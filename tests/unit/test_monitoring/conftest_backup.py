"""
Shared fixtures for optimized monitoring tests.

This file provides lightweight, optimized fixtures to reduce test setup overhead
and eliminate I/O operations during testing.
"""

import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch

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
]:
    logger = logging.getLogger(logger_name)
    logger.disabled = True
    logger.setLevel(logging.CRITICAL)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

# Set environment variables for maximum performance
os.environ.update(
    {
        "DISABLE_ERROR_HANDLER_LOGGING": "true",
        "TESTING": "true",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONASYNCIODEBUG": "0",
        "PYTHONHASHSEED": "0",
        "PYTHONOPTIMIZE": "2",
        "DISABLE_ALL_LOGGING": "1",
        "PYTEST_FAST_MODE": "1",
    }
)

# Mock OpenTelemetry Resource class with proper create method
class MockResource:
    def __init__(self, attributes=None):
        self.attributes = attributes or {}
    
    @classmethod
    def create(cls, attributes=None):
        return cls(attributes)

# Mock OpenTelemetry classes with proper methods
class MockTracerProvider:
    def __init__(self, *args, **kwargs):
        pass
    
    def add_span_processor(self, processor):
        pass
        
    def shutdown(self):
        pass

class MockMeterProvider:
    def __init__(self, *args, **kwargs):
        pass
        
    def shutdown(self):
        pass

class MockTraceIdRatioBased:
    def __init__(self, ratio):
        self.ratio = ratio

class MockBatchSpanProcessor:
    def __init__(self, exporter):
        self.exporter = exporter

# Optimized mock patches - only essential heavy dependencies
MOCK_PATCHES = {
    "prometheus_client": Mock(),
    "prometheus_client.generate_latest": Mock(return_value=b"# Mock\n"),
    "opentelemetry": Mock(),
    "psutil": Mock(cpu_percent=Mock(return_value=0.0)),
    "smtplib": Mock(),
    "requests": Mock(),
    "httpx": Mock(),
    "time.sleep": Mock(),
    "yaml": Mock(safe_load=Mock(return_value={})),
}


@pytest.fixture(scope="session", autouse=True)
def mock_external_dependencies():
    """Mock heavy external dependencies for all monitoring tests."""
    # Only mock external libraries, not our own modules
    external_patches = {k: v for k, v in MOCK_PATCHES.items() if not k.startswith("src.")}
    
    # Create comprehensive OpenTelemetry mocks
    opentelemetry_mocks = {
        "opentelemetry.sdk.resources": Mock(Resource=MockResource),
        "opentelemetry.sdk.trace": Mock(TracerProvider=MockTracerProvider),
        "opentelemetry.sdk.metrics": Mock(MeterProvider=MockMeterProvider),
        "opentelemetry.sdk.trace.sampling": Mock(TraceIdRatioBased=MockTraceIdRatioBased),
        "opentelemetry.sdk.trace.export": Mock(BatchSpanProcessor=MockBatchSpanProcessor),
    }
    
    # Combine all mocks
    all_patches = {**external_patches, **opentelemetry_mocks}
    
    with patch.dict("sys.modules", all_patches):
        # Also patch common slow functions
        with patch("time.sleep") as mock_sleep:
            mock_sleep.return_value = None
            yield


@pytest.fixture(scope="session")
def mock_metrics_collector():
    """Provide a lightweight mock metrics collector - SESSION CACHED."""
    mock = Mock()
    # Pre-configure all methods for consistent behavior
    mock.increment_counter = Mock(return_value=None)
    mock.set_gauge = Mock(return_value=None)
    mock.observe_histogram = Mock(return_value=None)
    mock.observe_summary = Mock(return_value=None)
    mock.export_metrics = Mock(return_value="# Mock metrics\ntbot_test_metric 1.0\n")
    mock.get_metrics_content_type = Mock(return_value="text/plain; version=0.0.4; charset=utf-8")

    # Pre-create context manager mock to avoid repeated creation
    context_mock = Mock()
    context_mock.__enter__ = Mock(return_value=Mock())
    context_mock.__exit__ = Mock(return_value=None)
    mock.time_operation = Mock(return_value=context_mock)

    mock.register_metric = Mock(return_value=Mock())
    mock.get_metric = Mock(return_value=Mock())
    mock._metrics = {}
    mock._metric_definitions = {}
    mock._running = False

    # Pre-create sub-component mocks
    mock.trading_metrics = Mock()
    mock.system_metrics = Mock()
    mock.exchange_metrics = Mock()
    mock.risk_metrics = Mock()
    return mock


@pytest.fixture(scope="session")
def mock_alert_manager():
    """Provide a lightweight mock alert manager - SESSION CACHED."""
    mock = Mock()
    # Pre-configure all methods with consistent return values
    mock.fire_alert = Mock(return_value=None)
    mock.resolve_alert = Mock(return_value=True)
    mock.acknowledge_alert = Mock(return_value=True)
    mock.get_active_alerts = Mock(return_value=[])
    mock.get_alert_stats = Mock(return_value={"total": 0, "active": 0, "resolved": 0, "escalated": 0})
    mock.add_rule = Mock(return_value=None)
    mock.remove_rule = Mock(return_value=True)
    mock.add_escalation_policy = Mock(return_value=None)
    mock.add_suppression_rule = Mock(return_value=None)

    # Pre-initialize state attributes
    mock._rules = {}
    mock._active_alerts = {}
    mock._alert_history = []
    mock._escalation_policies = {}
    mock._suppression_rules = []
    mock._running = False
    mock._alerts_fired = 0
    mock._alerts_resolved = 0

    # Pre-create config mock
    mock.config = Mock()
    mock.config.email_enabled = False
    mock.config.slack_enabled = False
    mock.config.webhook_enabled = False

    return mock


@pytest.fixture(scope="session")
def mock_performance_profiler():
    """Provide a lightweight mock performance profiler with all expected methods."""
    mock = Mock()
    # Core profiler attributes
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
    mock.profile_async_function = Mock(return_value=Mock(__aenter__=Mock(), __aexit__=Mock()))

    # Stats methods
    mock.get_performance_summary = Mock(
        return_value={
            "timestamp": "2023-01-01T00:00:00Z",
            "metrics_collected": 0,
            "system_resources": {"cpu": 0.0, "memory": 0.0},
            "latency_stats": {},
            "throughput_stats": {},
            "gc_stats": {"collections": 0},
        }
    )
    mock.get_latency_stats = Mock(return_value=Mock(count=5, avg=3.0, p95=0.0, p99=0.0))
    mock.get_system_resource_stats = Mock(return_value={"cpu_percent": 0.0, "memory_percent": 0.0})
    mock.get_throughput_stats = Mock(return_value={"operations_per_second": 100.0})

    # Control methods
    mock.start = Mock()
    mock.stop = Mock()
    mock.clear_metrics = Mock()

    return mock


@pytest.fixture
def mock_dashboard_manager():
    """Provide a lightweight mock dashboard manager."""
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
    """Provide a mock dependency injector with comprehensive service resolution."""
    mock = Mock()
    mock.register_factory = Mock()
    mock.register_instance = Mock()
    mock.register_singleton = Mock()

    # Pre-configure common service resolutions
    mock.resolve.side_effect = lambda service_name: {
        "AlertManager": Mock(),
        "MetricsCollector": Mock(),
        "PerformanceProfiler": Mock(),
        "GrafanaDashboardManager": Mock(),
        "AlertServiceInterface": Mock(),
        "MetricsServiceInterface": Mock(),
        "PerformanceServiceInterface": Mock(),
        "DashboardServiceInterface": Mock(),
        "MonitoringServiceInterface": Mock(),
    }.get(service_name, Mock())

    return mock


@pytest.fixture(scope="session")
def fast_datetime():
    """Provide a fixed datetime to avoid time-based test variations."""

    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def mock_time_operations():
    """Mock all time-related operations for faster tests."""

    with patch("time.time", return_value=1672574400.0):  # Fixed timestamp
        with patch("time.perf_counter", side_effect=lambda: 0.001):  # Minimal timing
            yield


# Pytest configuration for fastest test execution
def pytest_configure(config):
    """Configure pytest for maximum performance."""
    # Essential optimization only
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
    # Ensure clean state before each test
    try:
        import sys
        from src.monitoring.dependency_injection import DIContainer
        
        # Ensure DI container is clean
        if 'src.monitoring.dependency_injection' in sys.modules:
            di_module = sys.modules['src.monitoring.dependency_injection']
            if not hasattr(di_module, '_container') or di_module._container is None:
                di_module._container = DIContainer()
                
    except (ImportError, AttributeError):
        pass


def pytest_runtest_teardown(item, nextitem):
    """Cleanup after each test run."""
    # Force cleanup of any remaining global state
    try:
        import gc
        gc.collect()  # Force garbage collection between tests
    except Exception:
        pass


# Removed custom event_loop fixture - pytest-asyncio handles this automatically
# The custom fixture was conflicting with pytest-asyncio's built-in event loop management


@pytest.fixture
def fresh_di_container():
    """Provide a fresh DI container for tests that need one."""
    from src.monitoring.dependency_injection import DIContainer
    
    container = DIContainer()
    yield container
    
    # Clean up after test
    if hasattr(container, '_bindings'):
        container._bindings.clear()
    if hasattr(container, '_resolving'):
        container._resolving.clear()


@pytest.fixture
def isolated_monitoring_state():
    """Provide completely isolated monitoring state for complex tests."""
    # Store all current state
    saved_state = {}
    
    try:
        import sys
        modules_to_isolate = [
            'src.monitoring.alerting',
            'src.monitoring.metrics',
            'src.monitoring.performance',
            'src.monitoring.telemetry',
            'src.monitoring.dependency_injection'
        ]
        
        for module_name in modules_to_isolate:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                saved_state[module_name] = {}
                
                # Save all global variables that start with _global
                for attr_name in dir(module):
                    if attr_name.startswith('_global') or attr_name == '_container':
                        if hasattr(module, attr_name):
                            saved_state[module_name][attr_name] = getattr(module, attr_name)
                            
    except (ImportError, AttributeError):
        pass
    
    yield
    
    # Restore original state
    try:
        import sys
        
        for module_name, module_state in saved_state.items():
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for attr_name, attr_value in module_state.items():
                    if hasattr(module, attr_name):
                        setattr(module, attr_name, attr_value)
                        
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset all global state between tests to prevent pollution."""
    # Store original values to restore clean state
    original_states = {}
    
    try:
        import sys
        from unittest.mock import Mock
        
        # Import modules to access their global state
        modules_to_reset = {
            'src.monitoring.alerting': ['_global_alert_manager'],
            'src.monitoring.metrics': ['_global_collector'],
            'src.monitoring.performance': ['_global_profiler'],
            'src.monitoring.telemetry': ['_global_trading_tracer'],
            'src.monitoring.dependency_injection': ['_container']
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
    
    # Reset all global state to clean initial values after test
    try:
        import sys
        from unittest.mock import Mock
        
        # Reset monitoring module globals properly
        modules_to_reset = {
            'src.monitoring.alerting': ['_global_alert_manager'],
            'src.monitoring.metrics': ['_global_collector'],
            'src.monitoring.performance': ['_global_profiler'],
            'src.monitoring.telemetry': ['_global_trading_tracer'],
            'src.monitoring.dependency_injection': ['_container']
        }
        
        for module_name, global_vars in modules_to_reset.items():
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for var_name in global_vars:
                    if hasattr(module, var_name):
                        if var_name == '_container':
                            # Reset container to new clean instance, not None
                            from src.monitoring.dependency_injection import DIContainer
                            setattr(module, var_name, DIContainer())
                        else:
                            # Reset other globals to None
                            setattr(module, var_name, None)
                            
        # Special handling for DI container - ensure it's properly reset
        try:
            import src.monitoring.dependency_injection as di_module
            if hasattr(di_module, '_container'):
                from src.monitoring.dependency_injection import DIContainer
                di_module._container = DIContainer()
        except ImportError:
            pass
            
        # Reset any cached instances in factory functions
        try:
            import src.monitoring.dependency_injection as di_module
            if hasattr(di_module, '_monitoring_container'):
                di_module._monitoring_container = None
        except ImportError:
            pass
            
    except (ImportError, AttributeError, Exception) as e:
        # Log error but don't fail tests
        import logging
        logging.getLogger(__name__).debug(f"Error resetting global state: {e}")


@pytest.fixture(scope="session", autouse=True)
def optimize_test_session():
    """Session-level optimizations."""
    import time
    import warnings

    # Essential optimizations only
    original_sleep = time.sleep
    time.sleep = lambda x: None
    warnings.filterwarnings("ignore")

    yield

    # Restore
    time.sleep = original_sleep


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks between tests."""
    yield
    
    # Reset mock call counts and side effects
    try:
        from unittest.mock import _mock_registry
        # Clear the global mock registry to prevent cross-test pollution
        _mock_registry.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)  
def isolate_di_container():
    """Ensure DI container is properly isolated between tests."""
    # Save original container state
    original_container = None
    try:
        import src.monitoring.dependency_injection as di_module
        if hasattr(di_module, '_container'):
            original_container = di_module._container
    except ImportError:
        pass
    
    yield
    
    # Always ensure we have a clean container after each test
    try:
        import src.monitoring.dependency_injection as di_module
        from src.monitoring.dependency_injection import DIContainer
        
        # Create fresh container instance
        di_module._container = DIContainer()
        
        # Clear any existing bindings
        if hasattr(di_module._container, '_bindings'):
            di_module._container._bindings.clear()
        if hasattr(di_module._container, '_resolving'):
            di_module._container._resolving.clear()
            
    except (ImportError, AttributeError) as e:
        # Log but don't fail test
        import logging
        logging.getLogger(__name__).debug(f"Error isolating DI container: {e}")


@pytest.fixture(autouse=True)
def thread_safety_isolation():
    """Ensure thread-safe test execution."""
    import threading
    from unittest.mock import Mock
    
    # Check if threading is mocked
    if isinstance(threading.active_count, Mock):
        # Skip thread safety checks if threading is mocked
        yield
        return
        
    # Get current thread count
    initial_thread_count = threading.active_count()
    
    yield
    
    # Wait for any background threads to complete
    import time
    max_wait = 1.0  # Maximum 1 second wait
    wait_interval = 0.01  # Check every 10ms
    waited = 0
    
    # Check again if threading is mocked after yield
    if isinstance(threading.active_count, Mock):
        return
        
    while threading.active_count() > initial_thread_count and waited < max_wait:
        time.sleep(wait_interval)
        waited += wait_interval


@pytest.fixture(autouse=True)
def clean_asyncio_state():
    """Clean asyncio state between tests."""
    yield
    
    try:
        import asyncio
        import gc
        
        # Clean up any pending tasks
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
            
        # Force garbage collection to clean up async resources
        gc.collect()
        
    except (ImportError, AttributeError):
        pass
