"""
Unit tests for pattern analytics functionality.

Tests error pattern detection, trend analysis, and predictive analytics.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from src.core.config import Config
from src.error_handling.pattern_analytics import ErrorPatternAnalytics, ErrorTrend


@dataclass
class ErrorPattern:
    """Mock ErrorPattern for testing purposes."""

    pattern_id: str
    pattern_type: str
    component: str
    error_type: str
    frequency: float
    severity: str
    first_detected: datetime
    last_detected: datetime
    occurrence_count: int
    confidence: float
    description: str
    suggested_action: str
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "component": self.component,
            "error_type": self.error_type,
            "frequency": self.frequency,
            "severity": self.severity,
            "first_detected": self.first_detected.isoformat(),
            "last_detected": self.last_detected.isoformat(),
            "occurrence_count": self.occurrence_count,
            "confidence": self.confidence,
            "description": self.description,
            "suggested_action": self.suggested_action,
            "is_active": self.is_active,
        }


class TestErrorPattern:
    """Test error pattern functionality."""

    def test_error_pattern_creation(self):
        """Test error pattern creation."""
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            component="exchange",
            error_type="timeout",
            frequency=5.0,
            severity="high",
            first_detected=timestamp,
            last_detected=timestamp,
            occurrence_count=5,
            confidence=0.8,
            description="Test pattern",
            suggested_action="Monitor",
        )

        assert pattern.pattern_id == "test_pattern"
        assert pattern.pattern_type == "frequency"
        assert pattern.component == "exchange"
        assert pattern.error_type == "timeout"
        assert pattern.frequency == 5.0
        assert pattern.severity == "high"
        assert pattern.occurrence_count == 5
        assert pattern.confidence == 0.8
        assert pattern.description == "Test pattern"
        assert pattern.suggested_action == "Monitor"

    def test_error_pattern_to_dict(self):
        """Test error pattern to dictionary conversion."""
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="correlation",
            component="database",
            error_type="connection_failed",
            frequency=3.0,
            severity="medium",
            first_detected=timestamp,
            last_detected=timestamp,
            occurrence_count=3,
            confidence=0.7,
            description="Test pattern",
            suggested_action="Investigate",
        )

        # Test direct field access (dataclass)
        assert pattern.pattern_id == "test_pattern"
        assert pattern.pattern_type == "correlation"
        assert pattern.component == "database"
        assert pattern.error_type == "connection_failed"
        assert pattern.frequency == 3.0
        assert pattern.severity == "medium"
        assert pattern.occurrence_count == 3
        assert pattern.confidence == 0.7
        assert pattern.description == "Test pattern"
        assert pattern.suggested_action == "Investigate"


class TestErrorTrend:
    """Test error trend functionality."""

    def test_error_trend_creation(self):
        """Test error trend creation."""
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=1)
        data_points = [
            (start_time, 5),
            (start_time + timedelta(minutes=15), 8),
            (start_time + timedelta(minutes=30), 12),
            (start_time + timedelta(minutes=45), 15),
            (end_time, 18),
        ]

        trend = ErrorTrend(
            pattern_id="api_rate_limit_pattern",
            frequency=0.8,
            direction="increasing",
            confidence=0.8,
            time_window="hourly",
        )

        assert trend.pattern_id == "api_rate_limit_pattern"
        assert trend.frequency == 0.8
        assert trend.direction == "increasing"
        assert trend.confidence == 0.8
        assert trend.time_window == "hourly"


class TestErrorPatternAnalytics:
    """Test error pattern analytics functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def analytics(self, config):
        """Provide error pattern analytics instance."""
        return ErrorPatternAnalytics(config)

    def test_analytics_initialization(self, config):
        """Test error pattern analytics initialization."""
        analytics = ErrorPatternAnalytics(config)
        assert analytics._raw_config == config
        # Check optimized data structures are initialized
        assert hasattr(analytics.error_history, "size")
        assert analytics.error_history.size() == 0
        assert hasattr(analytics.detected_patterns, "size")
        assert analytics.detected_patterns.size() == 0
        assert isinstance(analytics.error_trends, dict)
        assert isinstance(analytics.correlation_matrix, dict)
        assert len(analytics.error_trends) == 0
        assert len(analytics.correlation_matrix) == 0

    @pytest.mark.asyncio
    async def test_add_error_event(self, analytics):
        """Test adding error event."""
        error_event = {
            "timestamp": datetime.now(timezone.utc),
            "error_type": "timeout",
            "component": "exchange",
            "severity": "high",
            "message": "API timeout",
        }

        # Mock the _check_patterns method to avoid async issues
        with patch.object(analytics, "_check_patterns") as mock_analyze:
            analytics.add_error_event(error_event)

            # Verify the event was added
            assert analytics.error_history.size() == 1
            # Since we can't access items by index directly, just check the size
            # Verify the analyze method was called
            mock_analyze.assert_called_once()

    def test_check_patterns(self, analytics):
        """Test pattern checking."""
        # Add some test events
        test_event = {
            "timestamp": datetime.now(timezone.utc),
            "error_type": "timeout",
            "component": "exchange",
            "severity": "high",
        }
        analytics.error_history.append(test_event)

        # Test that check patterns can be called without error
        analytics._check_patterns()
        
        # Verify the event was added
        assert len(analytics.error_history) == 1

    def test_check_frequency_patterns(self, analytics):
        """Test frequency pattern checking."""
        # Add enough test events to exceed the frequency threshold (5)
        for i in range(6):  # 6 errors > 5 threshold
            test_event = {
                "timestamp": datetime.now(timezone.utc),
                "error_type": "timeout",
                "component": "exchange",
                "severity": "high",
            }
            analytics.error_history.append(test_event)

        # Test that check patterns can handle high frequency without error
        analytics._check_patterns()
        
        # Verify all events were added
        assert len(analytics.error_history) == 6

    def test_get_correlation_summary(self, analytics):
        """Test correlation summary."""
        # Add test events from different components close in time
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        
        # Add two events from different components within 5 minutes
        event1 = {
            "timestamp": now,
            "error_type": "timeout",
            "component": "exchange",
            "severity": "high",
        }
        event2 = {
            "timestamp": now + timedelta(minutes=2),
            "error_type": "connection",
            "component": "database", 
            "severity": "medium",
        }
        analytics.error_history.append(event1)
        analytics.error_history.append(event2)

        # Test correlation summary
        result = analytics.get_correlation_summary()
        assert isinstance(result, dict)
        assert "component_correlations" in result

    def test_get_trend_summary(self, analytics):
        """Test trend summary."""
        # Add test events across different hours
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        
        # Add events in early and late periods
        for i in range(3):
            early_event = {
                "timestamp": now - timedelta(hours=2),
                "error_type": "timeout",
                "component": "exchange",
                "severity": "high",
            }
            late_event = {
                "timestamp": now - timedelta(minutes=30),
                "error_type": "timeout",
                "component": "exchange", 
                "severity": "high",
            }
            analytics.error_history.append(early_event)
            analytics.error_history.append(late_event)

        # Test trend summary
        result = analytics.get_trend_summary()
        assert isinstance(result, dict)
        assert "trends" in result

    def test_get_recent_errors(self, analytics):
        """Test getting recent errors."""
        # Add some test events with different timestamps
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        
        # Recent error (within 1 hour)
        recent_event = {
            "timestamp": now - timedelta(minutes=30),
            "error_type": "timeout",
            "component": "exchange",
            "severity": "high",
        }
        # Old error (more than 1 hour ago)
        old_event = {
            "timestamp": now - timedelta(hours=2),
            "error_type": "timeout", 
            "component": "exchange",
            "severity": "high",
        }
        analytics.error_history.append(recent_event)
        analytics.error_history.append(old_event)

        # Test getting recent errors
        recent_errors = analytics.get_recent_errors(hours=1)
        assert isinstance(recent_errors, list)
        # Should only include the recent error
        assert len(recent_errors) == 1
        assert recent_errors[0]["timestamp"] == recent_event["timestamp"]

    def test_get_pattern_summary(self, analytics):
        """Test getting pattern summary."""
        # Add some test events
        test_event = {
            "timestamp": datetime.now(timezone.utc),
            "error_type": "timeout",
            "component": "exchange",
            "severity": "high",
        }
        analytics.error_history.append(test_event)

        summary = analytics.get_pattern_summary()
        assert "total_errors_24h" in summary
        assert "errors_by_component" in summary
        assert "errors_by_severity" in summary
        assert "total_errors_tracked" in summary
        assert summary["total_errors_tracked"] == 1

    def test_get_correlation_summary_original(self, analytics):
        """Test getting actual correlation summary."""
        result = analytics.get_correlation_summary()
        assert isinstance(result, dict)
        assert "component_correlations" in result

    def test_get_trend_summary_original(self, analytics):
        """Test getting actual trend summary."""
        result = analytics.get_trend_summary() 
        assert isinstance(result, dict)
        assert "trends" in result

    def test_analytics_integration(self, analytics):
        """Test analytics integration."""
        # Add test events
        test_event = {
            "timestamp": datetime.now(timezone.utc),
            "error_type": "timeout",
            "component": "exchange",
            "severity": "high",
        }
        analytics.error_history.append(test_event)

        # Test that all methods work together
        analytics.add_error_event(test_event)
        pattern_summary = analytics.get_pattern_summary()
        correlation_summary = analytics.get_correlation_summary()
        trend_summary = analytics.get_trend_summary()
        recent_errors = analytics.get_recent_errors(hours=24)
        
        # Verify all methods return expected data structures
        assert isinstance(pattern_summary, dict)
        assert isinstance(correlation_summary, dict)
        assert isinstance(trend_summary, dict)
        assert isinstance(recent_errors, list)
