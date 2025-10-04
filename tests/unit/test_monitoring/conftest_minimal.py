"""
Minimal conftest for monitoring tests - fixes hanging issues.

The original conftest.py had complex session fixtures and pytest hooks
that caused hanging during test collection. This minimal version provides
only essential fixtures.
"""

import pytest
from unittest.mock import Mock, MagicMock
from decimal import Decimal

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
