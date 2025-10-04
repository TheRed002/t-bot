"""
Minimal conftest for monitoring tests - fixes hanging issues.

The original conftest.py had complex session fixtures and pytest hooks
that caused hanging during test collection. This minimal version provides
only essential fixtures.
"""

import pytest
from unittest.mock import Mock, MagicMock
from decimal import Decimal
from datetime import datetime, timezone

# Simple mocks without complex initialization
@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    return Mock()

@pytest.fixture
def mock_alert_manager():
    """Mock alert manager."""
    return Mock()

@pytest.fixture
def sample_metric_request():
    """Provide a sample metric request."""
    return {
        "name": "test_metric",
        "value": Decimal("100.0"),
        "labels": {"type": "test"},
        "namespace": "testing",
    }

@pytest.fixture(scope="session")
def fast_datetime():
    """Provide a fixed datetime to avoid time-based test variations."""
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

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
    }

    mock.resolve = Mock()
    mock.resolve.side_effect = lambda service_name: service_mocks.get(service_name, Mock())
    return mock
