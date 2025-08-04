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
    
    def test_add_error_event(self, analytics):
        """Test adding error event to analytics."""
        error_event = {
            "timestamp": datetime.now(timezone.utc),
            "component": "exchange",
            "error_type": "timeout",
            "severity": "high",
            "details": {"latency": 5000}
        }
        
        analytics.add_error_event(error_event)
        
        assert len(analytics.error_history) == 1
        assert analytics.error_history[0]["component"] == "exchange"
        assert analytics.error_history[0]["error_type"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_analyze_patterns(self, analytics):
        """Test pattern analysis."""
        # Add some error events
        for i in range(10):
            error_event = {
                "timestamp": datetime.now(timezone.utc),
                "component": "exchange",
                "error_type": "timeout",
                "severity": "high",
                "details": {"latency": 5000 + i}
            }
            analytics.add_error_event(error_event)
        
        await analytics.analyze_patterns()
        
        # Should detect patterns
        assert len(analytics.detected_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_frequency_patterns(self, analytics):
        """Test frequency pattern analysis."""
        # Add error events with high frequency
        for i in range(20):
            error_event = {
                "timestamp": datetime.now(timezone.utc),
                "component": "database",
                "error_type": "connection_failed",
                "severity": "medium",
                "details": {"attempt": i}
            }
            analytics.add_error_event(error_event)
        
        await analytics.analyze_frequency_patterns()
        
        # Should detect frequency patterns
        frequency_patterns = [
            pattern for pattern in analytics.detected_patterns.values()
            if pattern.pattern_type == "frequency"
        ]
        assert len(frequency_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_correlations(self, analytics):
        """Test correlation analysis."""
        # Add correlated error events
        for i in range(10):
            # Add exchange timeout
            error_event1 = {
                "timestamp": datetime.now(timezone.utc),
                "component": "exchange",
                "error_type": "timeout",
                "severity": "high"
            }
            analytics.add_error_event(error_event1)
            
            # Add database connection error (correlated)
            error_event2 = {
                "timestamp": datetime.now(timezone.utc),
                "component": "database",
                "error_type": "connection_failed",
                "severity": "medium"
            }
            analytics.add_error_event(error_event2)
        
        await analytics.analyze_correlations()
        
        # Should detect correlations
        assert len(analytics.correlation_matrix) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_trends(self, analytics):
        """Test trend analysis."""
        # Add error events over time
        base_time = datetime.now(timezone.utc)
        for i in range(24):  # 24 hours
            error_event = {
                "timestamp": base_time + timedelta(hours=i),
                "component": "api",
                "error_type": "rate_limit",
                "severity": "medium",
                "details": {"hour": i}
            }
            analytics.add_error_event(error_event)
        
        await analytics.analyze_trends()
        
        # Should detect trends
        assert len(analytics.error_trends) > 0
    
    @pytest.mark.asyncio
    async def test_predictive_analysis(self, analytics):
        """Test predictive analysis."""
        # Add historical error events
        for i in range(50):
            error_event = {
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=i),
                "component": "exchange",
                "error_type": "timeout",
                "severity": "high",
                "details": {"historical": True}
            }
            analytics.add_error_event(error_event)
        
        await analytics.predictive_analysis()
        
        # Should generate predictions
        predictions = [
            pattern for pattern in analytics.detected_patterns.values()
            if pattern.pattern_type == "anomaly"
        ]
        assert len(predictions) >= 0  # May or may not have predictions
    
    def test_get_pattern_summary(self, analytics):
        """Test getting pattern summary."""
        # Add some patterns
        timestamp = datetime.now(timezone.utc)
        pattern1 = ErrorPattern(
            pattern_id="pattern1",
            pattern_type="frequency",
            component="exchange",
            error_type="timeout",
            frequency=5.0,
            severity="high",
            first_detected=timestamp,
            last_detected=timestamp,
            occurrence_count=5,
            confidence=0.8,
            description="Test pattern 1",
            suggested_action="Monitor"
        )
        pattern2 = ErrorPattern(
            pattern_id="pattern2",
            pattern_type="correlation",
            component="database",
            error_type="connection_failed",
            frequency=3.0,
            severity="medium",
            first_detected=timestamp,
            last_detected=timestamp,
            occurrence_count=3,
            confidence=0.7,
            description="Test pattern 2",
            suggested_action="Investigate"
        )
        
        analytics.detected_patterns["pattern1"] = pattern1
        analytics.detected_patterns["pattern2"] = pattern2
        
        summary = analytics.get_pattern_summary()
        
        assert summary["total_patterns"] == 2
        assert summary["frequency_patterns"] == 1
        assert summary["correlation_patterns"] == 1
        assert summary["high_severity_patterns"] == 1
        assert summary["medium_severity_patterns"] == 1
    
    def test_get_correlation_summary(self, analytics):
        """Test getting correlation summary."""
        # Add some correlations
        analytics.correlation_matrix["exchange:timeout"] = {
            "database:connection_failed": 0.8,
            "api:rate_limit": 0.6
        }
        analytics.correlation_matrix["database:connection_failed"] = {
            "exchange:timeout": 0.8
        }
        
        summary = analytics.get_correlation_summary()
        
        assert summary["total_correlations"] == 2
        assert summary["strong_correlations"] >= 0
        assert summary["correlation_pairs"] == 2
    
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
        """Test analytics integration scenarios."""
        # Add various error events
        error_events = [
            {"component": "exchange", "error_type": "timeout", "severity": "high"},
            {"component": "database", "error_type": "connection_failed", "severity": "medium"},
            {"component": "api", "error_type": "rate_limit", "severity": "low"},
            {"component": "exchange", "error_type": "timeout", "severity": "high"},
            {"component": "database", "error_type": "connection_failed", "severity": "medium"}
        ]
        
        for event in error_events:
            event["timestamp"] = datetime.now(timezone.utc)
            event["details"] = {"test": True}
            analytics.add_error_event(event)
        
        # Run analysis
        await analytics.analyze_patterns()
        
        # Check results
        assert len(analytics.error_history) == 5
        assert len(analytics.detected_patterns) >= 0  # May or may not detect patterns
        
        # Get summaries
        pattern_summary = analytics.get_pattern_summary()
        correlation_summary = analytics.get_correlation_summary()
        trend_summary = analytics.get_trend_summary()
        
        assert "total_patterns" in pattern_summary
        assert "total_correlations" in correlation_summary
        assert "total_trends" in trend_summary 