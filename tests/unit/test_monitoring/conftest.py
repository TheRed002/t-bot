"""
Shared fixtures for optimized monitoring tests.

This file provides lightweight, optimized fixtures to reduce test setup overhead
and eliminate I/O operations during testing.
"""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal
from datetime import datetime, timezone
import logging
import os

# CRITICAL: Disable ALL logging completely for monitoring tests
logging.disable(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger().disabled = True

# Specifically disable error handling logging that causes spam
for logger_name in ['src.error_handling', 'error_handling', 'src.monitoring', 'monitoring', 'src.core', 'src.utils', 'asyncio', 'urllib3']:
    logger = logging.getLogger(logger_name)
    logger.disabled = True
    logger.setLevel(logging.CRITICAL)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

# Set environment variables for maximum performance
os.environ.update({
    'DISABLE_ERROR_HANDLER_LOGGING': 'true',
    'TESTING': 'true',
    'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONASYNCIODEBUG': '0',
    'PYTHONHASHSEED': '0',
    'PYTHONOPTIMIZE': '2',
    'DISABLE_ALL_LOGGING': '1',
    'PYTEST_FAST_MODE': '1'
})

# Mock heavy external dependencies globally - comprehensive optimization
MOCK_PATCHES = {
    'prometheus_client': Mock(),
    'prometheus_client.Counter': Mock,
    'prometheus_client.Gauge': Mock,
    'prometheus_client.Histogram': Mock,
    'prometheus_client.Summary': Mock,
    'prometheus_client.CollectorRegistry': Mock,
    'prometheus_client.start_http_server': Mock,
    'prometheus_client.generate_latest': Mock(return_value=b'# Mock metrics\n'),
    'opentelemetry': Mock(),
    'opentelemetry.trace': Mock(),
    'opentelemetry.metrics': Mock(),
    'opentelemetry.sdk': Mock(),
    'opentelemetry.instrumentation': Mock(),
    'psutil': Mock(cpu_percent=Mock(return_value=0.0), virtual_memory=Mock(return_value=Mock(percent=0.0))),
    'aiohttp': Mock(),
    'smtplib': Mock(),
    'requests': Mock(),
    'httpx': Mock(),
    'websockets': Mock(),
    # Don't mock asyncio.sleep - causes event loop issues
    'time.sleep': Mock(),
    'numpy': Mock(),
    'scipy': Mock(),
    'yaml': Mock(safe_load=Mock(return_value={})),
    'email.mime': Mock(),
    'fastapi': Mock(),
    'urllib3': Mock(),
    'sqlite3': Mock(),
    'threading': Mock(),
    'multiprocessing': Mock(),
}

@pytest.fixture(scope="session", autouse=True)
def mock_external_dependencies():
    """Mock heavy external dependencies for all monitoring tests."""
    with patch.dict('sys.modules', MOCK_PATCHES):
        # Also patch common slow functions
        with patch('time.sleep') as mock_sleep:
            mock_sleep.return_value = None
            yield

@pytest.fixture(scope="session")
def mock_metrics_collector():
    """Provide a lightweight mock metrics collector."""
    mock = Mock()
    mock.increment_counter = Mock()
    mock.set_gauge = Mock()
    mock.observe_histogram = Mock()
    mock.observe_summary = Mock()
    mock.export_metrics = Mock(return_value="# Mock metrics")
    mock.get_metrics_content_type = Mock(return_value="text/plain; version=0.0.4; charset=utf-8")
    mock.time_operation = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
    mock.register_metric = Mock()
    mock.get_metric = Mock()
    mock._metrics = {}
    mock._metric_definitions = {}
    mock._running = False
    mock.trading_metrics = Mock()
    mock.system_metrics = Mock()
    mock.exchange_metrics = Mock()
    mock.risk_metrics = Mock()
    return mock

@pytest.fixture(scope="session")
def mock_alert_manager():
    """Provide a lightweight mock alert manager."""
    mock = Mock()
    mock.fire_alert = Mock()
    mock.resolve_alert = Mock(return_value=True)
    mock.acknowledge_alert = Mock(return_value=True)
    mock.get_active_alerts = Mock(return_value=[])
    mock.get_alert_stats = Mock(return_value={"total": 0, "active": 0})
    mock.add_rule = Mock()
    mock.remove_rule = Mock(return_value=True)
    mock.add_escalation_policy = Mock()
    mock.add_suppression_rule = Mock()
    mock._rules = {}
    mock._active_alerts = {}
    mock._alert_history = []
    mock._running = False
    mock.config = Mock()
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
    mock._latency_data = {'test_metric': [1.0, 2.0, 3.0]}
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
    mock.get_performance_summary = Mock(return_value={
        "timestamp": "2023-01-01T00:00:00Z",
        "metrics_collected": 0,
        "system_resources": {"cpu": 0.0, "memory": 0.0},
        "latency_stats": {},
        "throughput_stats": {},
        "gc_stats": {"collections": 0}
    })
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
        'AlertManager': Mock(),
        'MetricsCollector': Mock(),
        'PerformanceProfiler': Mock(),
        'GrafanaDashboardManager': Mock(),
        'AlertServiceInterface': Mock(),
        'MetricsServiceInterface': Mock(),
        'PerformanceServiceInterface': Mock(),
        'DashboardServiceInterface': Mock(),
        'MonitoringServiceInterface': Mock(),
    }.get(service_name, Mock())
    
    return mock

@pytest.fixture(scope="session")
def fast_datetime():
    """Provide a fixed datetime to avoid time-based test variations."""
    from datetime import datetime, timezone
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

@pytest.fixture(scope="session")
def mock_time_operations():
    """Mock all time-related operations for faster tests."""
    import time
    with patch('time.time', return_value=1672574400.0):  # Fixed timestamp
        with patch('time.perf_counter', side_effect=lambda: 0.001):  # Minimal timing
            with patch('datetime.datetime') as mock_dt:
                mock_dt.now.return_value = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
                mock_dt.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
                yield

# Pytest configuration for fastest test execution
def pytest_configure(config):
    """Configure pytest for maximum performance."""
    # Disable warnings and optimize output
    config.option.disable_warnings = True
    config.option.tb = 'no'  # No tracebacks for speed
    config.option.capture = 'no'  # Disable output capture
    config.option.quiet = True
    
    # Set fastest asyncio event loop - but handle it properly
    import asyncio
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use the default event loop policy without nest_asyncio
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except RuntimeError:
        pass
    except Exception:
        # Silently handle any asyncio policy issues
        pass
    
    # Optimize garbage collection for tests
    import gc
    gc.disable()  # Disable GC during tests for maximum speed
    gc.set_threshold(0, 0, 0)  # Disable automatic GC completely
    
    # Set environment variables for maximum performance
    os.environ.update({
        'PYTEST_FAST_MODE': '1',
        'PYTHONASYNCIODEBUG': '0',
        'PYTHONHASHSEED': '0',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONOPTIMIZE': '2',
        'DISABLE_ALL_LOGGING': '1',
        'PYTHONUNBUFFERED': '1',
        'PYTHONNOUSERSITE': '1'
    })
    
    # Disable pytest plugins for speed
    try:
        config.pluginmanager.set_blocked('cacheprovider')
        config.pluginmanager.set_blocked('junitxml')
        config.pluginmanager.set_blocked('resultlog')
    except:
        pass

def pytest_unconfigure(config):
    """Re-enable GC after tests complete."""
    import gc
    gc.enable()

# Removed custom event_loop fixture - pytest-asyncio handles this automatically
# The custom fixture was conflicting with pytest-asyncio's built-in event loop management

@pytest.fixture(scope="session", autouse=True)
def optimize_test_session():
    """Session-level optimizations."""
    # Patch time.sleep and asyncio.sleep globally to be no-ops
    import time
    import asyncio
    import warnings
    
    # Disable nest_asyncio warnings properly
    try:
        import nest_asyncio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nest_asyncio.apply()
    except ImportError:
        pass
    except Exception:
        # Silently handle any nest_asyncio issues
        pass
    
    # Replace time functions with no-ops
    original_sleep = time.sleep
    time.sleep = lambda x: None
    
    # Don't mock asyncio.sleep globally - it can cause issues
    # Let individual tests mock it if needed
    
    # Disable all warnings more comprehensively
    import warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', PendingDeprecationWarning) 
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', FutureWarning)
    warnings.simplefilter('ignore', ImportWarning)
    
    try:
        import urllib3
        urllib3.disable_warnings()
    except ImportError:
        pass
    
    # Disable expensive network operations
    import socket
    socket.setdefaulttimeout(0.1)
    
    yield
    
    # Restore original functions
    time.sleep = original_sleep
    
    # Cleanup after session
    import gc
    gc.collect()