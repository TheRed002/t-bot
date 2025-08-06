"""
Error pattern analytics for detecting and analyzing error patterns.

This module provides error frequency analysis, root cause analysis automation,
predictive error detection, error correlation analysis, and automated error
reporting and escalation.

CRITICAL: This module integrates with P-001 core framework and P-002 database
for error logging and will be used by all subsequent prompts.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from src.core.logging import get_logger

# MANDATORY: Import from P-001 core framework
from src.core.exceptions import TradingBotError
from src.core.config import Config

logger = get_logger(__name__)


@dataclass
class ErrorPattern:
    """Represents a detected error pattern."""
    pattern_id: str
    pattern_type: str  # frequency, correlation, trend, anomaly
    component: str
    error_type: str
    frequency: float  # errors per hour
    severity: str
    first_detected: datetime
    last_detected: datetime
    occurrence_count: int
    confidence: float  # 0.0 to 1.0
    description: str
    suggested_action: str
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
            "is_active": self.is_active
        }


@dataclass
class ErrorTrend:
    """Represents an error trend over time."""
    component: str
    error_type: str
    time_period: str  # hourly, daily, weekly
    trend_direction: str  # increasing, decreasing, stable
    trend_strength: float  # 0.0 to 1.0
    start_time: datetime
    end_time: datetime
    data_points: List[Tuple[datetime, int]] = field(default_factory=list)


class ErrorPatternAnalytics:
    """Analyzes error patterns for predictive detection and root cause analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.error_analytics_config = config.error_handling
        self.pattern_detection = self.error_analytics_config.pattern_detection_enabled
        self.correlation_analysis = self.error_analytics_config.correlation_analysis_enabled
        self.predictive_alerts = self.error_analytics_config.predictive_alerts_enabled
        
        # Error tracking
        self.error_history: List[Dict[str, Any]] = []
        self.detected_patterns: Dict[str, ErrorPattern] = {}
        self.error_trends: Dict[str, ErrorTrend] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        # Analytics settings
        self.frequency_threshold = 5  # errors per hour to trigger pattern detection
        self.correlation_threshold = 0.7  # minimum correlation coefficient
        self.trend_window_hours = 24  # hours to analyze for trends
        self.pattern_confidence_threshold = 0.8  # minimum confidence for pattern detection
    
    def add_error_event(self, error_context: Dict[str, Any]):
        """Add an error event to the analytics system."""
        
        error_event = {
            "timestamp": datetime.now(timezone.utc),
            "component": error_context.get("component", "unknown"),
            "error_type": error_context.get("error_type", "unknown"),
            "severity": error_context.get("severity", "medium"),
            "operation": error_context.get("operation", "unknown"),
            "details": error_context.get("details", {}),
            "error_id": error_context.get("error_id", "unknown")
        }
        
        self.error_history.append(error_event)
        
        # Keep only last 10000 error events
        if len(self.error_history) > 10000:
            self.error_history = self.error_history[-10000:]
        
        # Trigger pattern analysis
        asyncio.create_task(self._analyze_patterns())
    
    async def _analyze_patterns(self):
        """Analyze error patterns asynchronously."""
        
        try:
            # Frequency analysis
            if self.pattern_detection:
                await self._analyze_frequency_patterns()
            
            # Correlation analysis
            if self.correlation_analysis:
                await self._analyze_correlations()
            
            # Trend analysis
            await self._analyze_trends()
            
            # Predictive analysis
            if self.predictive_alerts:
                await self._predictive_analysis()
                
        except Exception as e:
            logger.error("Error pattern analysis failed", error=str(e))
    
    async def _analyze_frequency_patterns(self):
        """Analyze error frequency patterns."""
        
        # Group errors by component and type
        error_counts = defaultdict(lambda: defaultdict(int))
        recent_errors = self._get_recent_errors(hours=1)
        
        for error in recent_errors:
            component = error["component"]
            error_type = error["error_type"]
            error_counts[component][error_type] += 1
        
        # Detect high-frequency patterns
        for component, error_types in error_counts.items():
            for error_type, count in error_types.items():
                if count >= self.frequency_threshold:
                    pattern_id = f"{component}:{error_type}:frequency"
                    
                    if pattern_id not in self.detected_patterns:
                        # New pattern detected
                        pattern = ErrorPattern(
                            pattern_id=pattern_id,
                            pattern_type="frequency",
                            component=component,
                            error_type=error_type,
                            frequency=count,
                            severity=self._determine_severity(count),
                            first_detected=datetime.now(timezone.utc),
                            last_detected=datetime.now(timezone.utc),
                            occurrence_count=count,
                            confidence=min(count / self.frequency_threshold, 1.0),
                            description=f"High frequency {error_type} errors in {component}",
                            suggested_action=f"Investigate {component} {error_type} errors"
                        )
                        
                        self.detected_patterns[pattern_id] = pattern
                        
                        logger.warning(
                            "Error pattern detected",
                            pattern_id=pattern_id,
                            frequency=count,
                            severity=pattern.severity,
                            confidence=pattern.confidence
                        )
                        
                        # Trigger alert
                        await self._trigger_pattern_alert(pattern)
                    else:
                        # Update existing pattern
                        pattern = self.detected_patterns[pattern_id]
                        pattern.last_detected = datetime.now(timezone.utc)
                        pattern.occurrence_count += count
                        pattern.frequency = count
                        pattern.confidence = min(count / self.frequency_threshold, 1.0)
    
    async def _analyze_correlations(self):
        """Analyze correlations between different error types."""
        
        recent_errors = self._get_recent_errors(hours=6)
        
        # Create time windows for correlation analysis
        time_windows = self._create_time_windows(recent_errors, window_minutes=30)
        
        # Calculate correlations between error types
        error_types = set()
        for error in recent_errors:
            error_types.add(error["error_type"])
        
        for error_type1 in error_types:
            for error_type2 in error_types:
                if error_type1 < error_type2:  # Avoid duplicate pairs
                    correlation = self._calculate_correlation(
                        error_type1, error_type2, time_windows
                    )
                    
                    if correlation >= self.correlation_threshold:
                        correlation_key = f"{error_type1}:{error_type2}"
                        self.correlation_matrix[correlation_key] = correlation
                        
                        logger.info(
                            "Error correlation detected",
                            error_type1=error_type1,
                            error_type2=error_type2,
                            correlation=correlation
                        )
    
    async def _analyze_trends(self):
        """Analyze error trends over time."""
        
        # Analyze trends for each component and error type
        components = set(error["component"] for error in self.error_history)
        error_types = set(error["error_type"] for error in self.error_history)
        
        for component in components:
            for error_type in error_types:
                trend = self._calculate_trend(component, error_type)
                
                if trend and trend.trend_strength > 0.5:
                    trend_key = f"{component}:{error_type}:trend"
                    self.error_trends[trend_key] = trend
                    
                    logger.info(
                        "Error trend detected",
                        component=component,
                        error_type=error_type,
                        direction=trend.trend_direction,
                        strength=trend.trend_strength
                    )
    
    async def _predictive_analysis(self):
        """Perform predictive analysis for potential future errors."""
        
        # Analyze patterns that might indicate future problems
        for pattern in self.detected_patterns.values():
            if pattern.is_active and pattern.confidence > self.pattern_confidence_threshold:
                
                # Check if pattern is worsening
                recent_frequency = self._get_recent_frequency(
                    pattern.component, pattern.error_type, hours=1
                )
                
                if recent_frequency > pattern.frequency * 1.5:
                    # Pattern is worsening, predict potential issues
                    prediction = await self._predict_issues(pattern)
                    
                    if prediction:
                        logger.warning(
                            "Predictive alert",
                            pattern_id=pattern.pattern_id,
                            prediction=prediction,
                            confidence=pattern.confidence
                        )
                        
                        # Trigger predictive alert
                        await self._trigger_predictive_alert(pattern, prediction)
    
    def _get_recent_errors(self, hours: int) -> List[Dict[str, Any]]:
        """Get errors from the last specified hours."""
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            error for error in self.error_history
            if error["timestamp"] > cutoff
        ]
    
    def _create_time_windows(self, errors: List[Dict[str, Any]], window_minutes: int) -> List[List[Dict[str, Any]]]:
        """Create time windows for correlation analysis."""
        
        if not errors:
            return []
        
        windows = []
        current_window = []
        window_start = errors[0]["timestamp"]
        
        for error in errors:
            if (error["timestamp"] - window_start).total_seconds() <= window_minutes * 60:
                current_window.append(error)
            else:
                if current_window:
                    windows.append(current_window)
                current_window = [error]
                window_start = error["timestamp"]
        
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def _calculate_correlation(self, error_type1: str, error_type2: str, time_windows: List[List[Dict[str, Any]]]) -> float:
        """Calculate correlation between two error types."""
        
        if len(time_windows) < 2:
            return 0.0
        
        # Count occurrences in each window
        counts1 = []
        counts2 = []
        
        for window in time_windows:
            count1 = sum(1 for error in window if error["error_type"] == error_type1)
            count2 = sum(1 for error in window if error["error_type"] == error_type2)
            
            counts1.append(count1)
            counts2.append(count2)
        
        # Calculate Pearson correlation
        if len(counts1) < 2:
            return 0.0
        
        mean1 = sum(counts1) / len(counts1)
        mean2 = sum(counts2) / len(counts2)
        
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(counts1, counts2))
        denominator1 = sum((x - mean1) ** 2 for x in counts1)
        denominator2 = sum((y - mean2) ** 2 for y in counts2)
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        correlation = numerator / (denominator1 ** 0.5 * denominator2 ** 0.5)
        return abs(correlation)  # Return absolute correlation
    
    def _calculate_trend(self, component: str, error_type: str) -> Optional[ErrorTrend]:
        """Calculate trend for a specific component and error type."""
        
        recent_errors = self._get_recent_errors(hours=self.trend_window_hours)
        component_errors = [
            error for error in recent_errors
            if error["component"] == component and error["error_type"] == error_type
        ]
        
        if len(component_errors) < 3:
            return None
        
        # Group by hour
        hourly_counts = defaultdict(int)
        for error in component_errors:
            hour = error["timestamp"].replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour] += 1
        
        # Calculate trend
        sorted_hours = sorted(hourly_counts.keys())
        if len(sorted_hours) < 2:
            return None
        
        # Simple linear trend calculation
        x_values = [(hour - sorted_hours[0]).total_seconds() / 3600 for hour in sorted_hours]
        y_values = [hourly_counts[hour] for hour in sorted_hours]
        
        # Calculate slope
        n = len(x_values)
        if n < 2:
            return None
        
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x ** 2 for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Determine trend direction and strength
        if abs(slope) < 0.1:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = min(abs(slope) / 2.0, 1.0)
        else:
            direction = "decreasing"
            strength = min(abs(slope) / 2.0, 1.0)
        
        return ErrorTrend(
            component=component,
            error_type=error_type,
            time_period="hourly",
            trend_direction=direction,
            trend_strength=strength,
            start_time=sorted_hours[0],
            end_time=sorted_hours[-1],
            data_points=[(hour, hourly_counts[hour]) for hour in sorted_hours]
        )
    
    def _get_recent_frequency(self, component: str, error_type: str, hours: int) -> int:
        """Get frequency of errors for a component and type in recent hours."""
        
        recent_errors = self._get_recent_errors(hours=hours)
        return sum(
            1 for error in recent_errors
            if error["component"] == component and error["error_type"] == error_type
        )
    
    def _determine_severity(self, frequency: int) -> str:
        """Determine severity based on error frequency."""
        
        if frequency >= 20:
            return "critical"
        elif frequency >= 10:
            return "high"
        elif frequency >= 5:
            return "medium"
        else:
            return "low"
    
    async def _predict_issues(self, pattern: ErrorPattern) -> Optional[str]:
        """Predict potential issues based on error pattern."""
        
        # Simple prediction logic based on pattern characteristics
        if pattern.frequency > 20:
            return f"Critical system degradation in {pattern.component}"
        elif pattern.frequency > 10:
            return f"Performance issues in {pattern.component}"
        elif pattern.frequency > 5:
            return f"Minor issues in {pattern.component}"
        else:
            return None
    
    async def _trigger_pattern_alert(self, pattern: ErrorPattern):
        """Trigger alert for detected pattern."""
        
        alert_message = {
            "type": "error_pattern",
            "pattern_id": pattern.pattern_id,
            "component": pattern.component,
            "error_type": pattern.error_type,
            "frequency": pattern.frequency,
            "severity": pattern.severity,
            "confidence": pattern.confidence,
            "description": pattern.description,
            "suggested_action": pattern.suggested_action,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.warning(
            "Error pattern alert",
            alert_data=alert_message
        )
        
        # TODO: Send alert to notification system
        # This will be implemented in P-031 (Alerting and Notification System)
    
    async def _trigger_predictive_alert(self, pattern: ErrorPattern, prediction: str):
        """Trigger predictive alert."""
        
        alert_message = {
            "type": "predictive_alert",
            "pattern_id": pattern.pattern_id,
            "component": pattern.component,
            "error_type": pattern.error_type,
            "prediction": prediction,
            "confidence": pattern.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.warning(
            "Predictive alert",
            alert_data=alert_message
        )
        
        # TODO: Send alert to notification system
        # This will be implemented in P-031 (Alerting and Notification System)
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns."""
        
        active_patterns = [p for p in self.detected_patterns.values() if p.is_active]
        
        return {
            "total_patterns": len(self.detected_patterns),
            "active_patterns": len(active_patterns),
            "patterns_by_severity": Counter(p.severity for p in active_patterns),
            "patterns_by_component": Counter(p.component for p in active_patterns),
            "recent_patterns": [
                p.to_dict() for p in active_patterns
                if (datetime.now(timezone.utc) - p.last_detected).total_seconds() < 3600
            ]
        }
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of error correlations."""
        
        strong_correlations = [
            (key, value) for key, value in self.correlation_matrix.items()
            if value >= self.correlation_threshold
        ]
        
        return {
            "total_correlations": len(self.correlation_matrix),
            "strong_correlations": len(strong_correlations),
            "correlation_threshold": self.correlation_threshold,
            "top_correlations": sorted(strong_correlations, key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of error trends."""
        
        increasing_trends = [t for t in self.error_trends.values() if t.trend_direction == "increasing"]
        decreasing_trends = [t for t in self.error_trends.values() if t.trend_direction == "decreasing"]
        
        return {
            "total_trends": len(self.error_trends),
            "increasing_trends": len(increasing_trends),
            "decreasing_trends": len(decreasing_trends),
            "strong_trends": len([t for t in self.error_trends.values() if t.trend_strength > 0.8]),
            "recent_trends": [
                {
                    "component": t.component,
                    "error_type": t.error_type,
                    "direction": t.trend_direction,
                    "strength": t.trend_strength
                }
                for t in self.error_trends.values()
                if (datetime.now(timezone.utc) - t.end_time).total_seconds() < 3600
            ]
        } 