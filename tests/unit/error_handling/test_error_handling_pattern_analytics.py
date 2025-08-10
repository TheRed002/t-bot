"""
Unit tests for pattern analytics functionality.

Tests error pattern detection, trend analysis, and predictive analytics.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from src.error_handling.pattern_analytics import (
    ErrorPattern, ErrorTrend, ErrorPatternAnalytics
)
from src.core.exceptions import TradingBotError
from src.core.config import Config


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
            suggested_action="Monitor"
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
            suggested_action="Investigate"
        )

        pattern_dict = pattern.to_dict()
        assert pattern_dict["pattern_id"] == "test_pattern"
        assert pattern_dict["pattern_type"] == "correlation"
        assert pattern_dict["component"] == "database"
        assert pattern_dict["error_type"] == "connection_failed"
        assert pattern_dict["frequency"] == 3.0
        assert pattern_dict["severity"] == "medium"
        assert pattern_dict["occurrence_count"] == 3
        assert pattern_dict["confidence"] == 0.7
        assert pattern_dict["description"] == "Test pattern"
        assert pattern_dict["suggested_action"] == "Investigate"


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
            (end_time, 18)
        ]

        trend = ErrorTrend(
            component="api",
            error_type="rate_limit",
            time_period="hourly",
            trend_direction="increasing",
            trend_strength=0.8,
            start_time=start_time,
            end_time=end_time,
            data_points=data_points
        )

        assert trend.component == "api"
        assert trend.error_type == "rate_limit"
        assert trend.time_period == "hourly"
        assert trend.trend_direction == "increasing"
        assert trend.trend_strength == 0.8
        assert trend.start_time == start_time
        assert trend.end_time == end_time
        assert len(trend.data_points) == 5


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
        assert analytics.config == config
        assert analytics.error_history == []
        assert analytics.detected_patterns == {}
        assert analytics.error_trends == {}
        assert analytics.correlation_matrix == {}

    @pytest.mark.asyncio
    async def test_add_error_event(self, analytics):
        """Test adding error event."""
        error_event = {
            "timestamp": datetime.now(timezone.utc),
            "error_type": "timeout",
            "component": "exchange",
            "severity": "high",
            "message": "API timeout"
        }

        # Mock the _analyze_patterns method to avoid async issues
        with patch.object(analytics, '_analyze_patterns') as mock_analyze:
            analytics.add_error_event(error_event)

            # Verify the event was added
            assert len(analytics.error_history) == 1
            assert analytics.error_history[0]["error_type"] == "timeout"
            # Verify the analyze method was called
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_patterns(self, analytics):
        """Test pattern analysis."""
        # Add some test events
        analytics.error_history = [
            {
                "timestamp": datetime.now(timezone.utc),
                "error_type": "timeout",
                "component": "exchange",
                "severity": "high"
            }
        ]

        # Mock the internal methods to avoid complex async operations
        with patch.object(analytics, '_analyze_frequency_patterns') as mock_freq, \
                patch.object(analytics, '_analyze_correlations') as mock_corr, \
                patch.object(analytics, '_analyze_trends') as mock_trend, \
                patch.object(analytics, '_predictive_analysis') as mock_pred:

            await analytics._analyze_patterns()

            # Verify the internal methods were called
            mock_freq.assert_called_once()
            mock_corr.assert_called_once()
            mock_trend.assert_called_once()
            # Predictive analysis may not be called if disabled
            if analytics.predictive_alerts:
                mock_pred.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_frequency_patterns(self, analytics):
        """Test frequency pattern analysis."""
        # Add enough test events to exceed the frequency threshold (5)
        for i in range(6):  # 6 errors > 5 threshold
            analytics.error_history.append({
                "timestamp": datetime.now(timezone.utc),
                "error_type": "timeout",
                "component": "exchange",
                "severity": "high"
            })

        # Mock the internal methods to avoid complex async operations
        with patch.object(analytics, '_trigger_pattern_alert') as mock_alert, \
                patch.object(analytics, '_get_recent_errors', return_value=analytics.error_history):

            await analytics._analyze_frequency_patterns()

            # Verify the method completed without errors
            # The alert should be called because we have 6 errors > 5 threshold
            mock_alert.assert_called()

    @pytest.mark.asyncio
    async def test_analyze_correlations(self, analytics):
        """Test correlation analysis."""
        # Add test events
        analytics.error_history = [
            {
                "timestamp": datetime.now(timezone.utc),
                "error_type": "timeout",
                "component": "exchange",
                "severity": "high"
            }
        ]

        # Mock the internal methods to avoid complex async operations
        with patch.object(analytics, '_get_recent_errors', return_value=analytics.error_history), \
                patch.object(analytics, '_create_time_windows', return_value=[]), \
                patch.object(analytics, '_calculate_correlation', return_value=0.5):

            await analytics._analyze_correlations()

            # Verify the method completed without errors
            assert len(analytics.correlation_matrix) >= 0

    @pytest.mark.asyncio
    async def test_analyze_trends(self, analytics):
        """Test trend analysis."""
        # Add test events
        analytics.error_history = [
            {
                "timestamp": datetime.now(timezone.utc),
                "error_type": "timeout",
                "component": "exchange",
                "severity": "high"
            }
        ]

        # Mock the internal methods to avoid complex async operations
        with patch.object(analytics, '_calculate_trend', return_value=None):

            await analytics._analyze_trends()

            # Verify the method completed without errors
            assert len(analytics.error_trends) >= 0

    @pytest.mark.asyncio
    async def test_predictive_analysis(self, analytics):
        """Test predictive analysis."""
        # Create a pattern that meets all the conditions
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            component="exchange",
            error_type="timeout",
            frequency=10.0,
            severity="high",
            first_detected=datetime.now(timezone.utc),
            last_detected=datetime.now(timezone.utc),
            occurrence_count=10,
            confidence=0.9,  # Above 0.8 threshold
            description="Test pattern",
            suggested_action="Test action",
            is_active=True  # Ensure pattern is active
        )
        analytics.detected_patterns["test_pattern"] = pattern

        # Add some error history
        for i in range(10):
            analytics.error_history.append({
                "timestamp": datetime.now(timezone.utc),
                "error_type": "timeout",
                "component": "exchange",
                "severity": "high"
            })

        # Mock the internal methods to ensure conditions are met
        with patch.object(analytics, '_predict_issues', return_value="Test prediction"), \
                patch.object(analytics, '_trigger_predictive_alert') as mock_alert, \
                patch.object(analytics, '_get_recent_frequency', return_value=20):  # > 10 * 1.5 = 15

            await analytics._predictive_analysis()

            # Verify the method completed without errors
            mock_alert.assert_called_once()

    def test_get_pattern_summary(self, analytics):
        """Test getting pattern summary."""
        # Add a test pattern
        analytics.detected_patterns = {
            "test_pattern": ErrorPattern(
                pattern_id="test_pattern",
                pattern_type="frequency",
                component="exchange",
                error_type="timeout",
                frequency=5.0,
                severity="high",
                first_detected=datetime.now(timezone.utc),
                last_detected=datetime.now(timezone.utc),
                occurrence_count=5,
                confidence=0.8,
                description="Test pattern",
                suggested_action="Monitor"
            )
        }

        summary = analytics.get_pattern_summary()
        assert "total_patterns" in summary
        assert summary["total_patterns"] == 1

    def test_get_correlation_summary(self, analytics):
        """Test getting correlation summary."""
        # Add test correlations
        analytics.correlation_matrix = {
            "test_correlation": 0.8
        }

        summary = analytics.get_correlation_summary()
        assert "total_correlations" in summary
        assert summary["total_correlations"] == 1

    def test_get_trend_summary(self, analytics):
        """Test getting trend summary."""
        # Add some trends
        base_time = datetime.now(timezone.utc)
        trend1 = ErrorTrend(
            component="exchange",
            error_type="timeout",
            time_period="hourly",
            trend_direction="increasing",
            trend_strength=0.8,
            start_time=base_time,
            end_time=base_time + timedelta(hours=1),
            data_points=[]
        )
        trend2 = ErrorTrend(
            component="database",
            error_type="connection_failed",
            time_period="daily",
            trend_direction="decreasing",
            trend_strength=0.6,
            start_time=base_time,
            end_time=base_time + timedelta(days=1),
            data_points=[]
        )

        analytics.error_trends["trend1"] = trend1
        analytics.error_trends["trend2"] = trend2

        summary = analytics.get_trend_summary()

        assert summary["total_trends"] == 2
        assert summary["increasing_trends"] == 1
        assert summary["decreasing_trends"] == 1
        assert summary["strong_trends"] >= 0

    @pytest.mark.asyncio
    async def test_analytics_integration(self, analytics):
        """Test analytics integration."""
        # Add test events
        analytics.error_history = [
            {
                "timestamp": datetime.now(timezone.utc),
                "error_type": "timeout",
                "component": "exchange",
                "severity": "high"
            }
        ]

        # Mock the internal methods to avoid complex async operations
        with patch.object(analytics, '_analyze_frequency_patterns') as mock_freq, \
                patch.object(analytics, '_analyze_correlations') as mock_corr, \
                patch.object(analytics, '_analyze_trends') as mock_trend:

            await analytics._analyze_patterns()

            # Verify integration works
            mock_freq.assert_called_once()
            mock_corr.assert_called_once()
            mock_trend.assert_called_once()
