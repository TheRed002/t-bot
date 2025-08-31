"""
Error pattern analytics for detecting and analyzing error patterns.

This module provides error frequency analysis, root cause analysis automation,
predictive error detection, error correlation analysis, and automated error
reporting and escalation.

CRITICAL: This module integrates with P-001 core framework and P-002 database
for error logging and will be used by all subsequent prompts.
"""

import asyncio
import threading
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Sized
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# MANDATORY: Import from P-001 core framework
# Import ErrorPattern using TYPE_CHECKING to avoid circular dependency
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types.data import ErrorPattern

from src.core.base.service import BaseService
from src.core.config import Config

# MANDATORY: Import from P-007A utils framework
# Import security components
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
)
from src.utils.decorators import time_execution
from src.utils.error_categorization import is_financial_component


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
    data_points: list[tuple[datetime, int]] = field(default_factory=list)


class OptimizedErrorHistory:
    """Memory-efficient error history with size limits and automatic cleanup."""

    def __init__(self, max_size: int = 5000, cleanup_interval_minutes: int = 30):
        self.max_size = max_size
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self._history: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._last_cleanup = datetime.now(timezone.utc)
        self._lock = threading.RLock()
        self._total_events = 0

    def add_event(self, event: dict[str, Any]) -> None:
        """Add error event with automatic cleanup."""
        with self._lock:
            # Add timestamp if not present
            if "timestamp" not in event:
                event["timestamp"] = datetime.now(timezone.utc)

            self._history.append(event)
            self._total_events += 1

            # Periodic cleanup
            if self._should_cleanup():
                self._cleanup_old_events()

    def get_recent_events(self, hours: int) -> list[dict[str, Any]]:
        """Get events from the last specified hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self._lock:
            return [event for event in self._history if event["timestamp"] > cutoff]

    def get_all_events(self) -> list[dict[str, Any]]:
        """Get all events in history."""
        with self._lock:
            return list(self._history)

    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        minutes_since_cleanup = (
            datetime.now(timezone.utc) - self._last_cleanup
        ).total_seconds() / 60
        return minutes_since_cleanup >= self.cleanup_interval_minutes

    def _cleanup_old_events(self) -> None:
        """Remove events older than 48 hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        with self._lock:
            # Since we're using deque, we can only remove from left
            # This is acceptable since we want to remove oldest events
            original_size = len(self._history)

            # Convert to list, filter, and recreate deque
            filtered_events = [event for event in self._history if event["timestamp"] > cutoff]

            self._history.clear()
            self._history.extend(filtered_events)

            removed_count = original_size - len(self._history)
            if removed_count > 0:
                # Update logger reference from parent class
                pass  # Will be logged by parent

            self._last_cleanup = datetime.now(timezone.utc)

    def size(self) -> int:
        """Get current history size."""
        return len(self._history)

    def total_events(self) -> int:
        """Get total events processed."""
        return self._total_events

    def __iter__(self):
        """Make the history iterable."""
        with self._lock:
            return iter(list(self._history))

    def __len__(self):
        """Get current history size."""
        return len(self._history)


class OptimizedPatternCache(Sized):
    """LRU cache for error patterns with TTL and memory limits."""

    def __init__(self, max_patterns: int = 1000, ttl_hours: int = 48):
        self.max_patterns = max_patterns
        self.ttl_hours = ttl_hours
        self._patterns: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def add_pattern(self, pattern: dict[str, Any] | Any) -> None:
        """Add or update pattern."""
        with self._lock:
            # Remove expired patterns
            self._cleanup_expired()

            # Remove LRU patterns if at capacity
            while len(self._patterns) >= self.max_patterns:
                self._patterns.popitem(last=False)

            # Convert to dict if it's a dataclass or object with to_dict method
            if hasattr(pattern, "to_dict"):
                pattern_dict = pattern.to_dict()
            elif hasattr(pattern, "__dict__"):
                # Convert object to dict
                pattern_dict = {
                    key: value for key, value in pattern.__dict__.items() if not key.startswith("_")
                }
            else:
                pattern_dict = pattern

            # Add/update pattern
            self._patterns[pattern_dict["pattern_id"]] = pattern_dict
            self._patterns.move_to_end(pattern_dict["pattern_id"])

    def get_pattern(self, pattern_id: str) -> dict[str, Any] | None:
        """Get pattern and mark as recently used."""
        with self._lock:
            if pattern_id not in self._patterns:
                return None

            pattern = self._patterns[pattern_id]
            # Check expiration
            age_hours = (
                datetime.now(timezone.utc) - pattern["first_detected"]
            ).total_seconds() / 3600
            if age_hours > self.ttl_hours:
                del self._patterns[pattern_id]
                return None

            # Mark as recently used
            self._patterns.move_to_end(pattern_id)
            return pattern

    def get_all_patterns(self) -> dict[str, dict[str, Any]]:
        """Get all active patterns."""
        with self._lock:
            self._cleanup_expired()
            return dict(self._patterns)

    def _cleanup_expired(self) -> None:
        """Remove expired patterns."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []

        for pattern_id, pattern in self._patterns.items():
            age_hours = (current_time - pattern["first_detected"]).total_seconds() / 3600
            if age_hours > self.ttl_hours:
                expired_keys.append(pattern_id)

        for key in expired_keys:
            del self._patterns[key]

    def size(self) -> int:
        """Get current pattern count."""
        return len(self._patterns)

    def __len__(self) -> int:
        """Get current pattern count for Sized protocol."""
        return len(self._patterns)

    def values(self) -> list["ErrorPattern"]:
        """Get all pattern values."""
        with self._lock:
            return list(self._patterns.values())

    def __setitem__(self, key: str, pattern: "ErrorPattern") -> None:
        """Support dictionary-style assignment."""
        self.add_pattern(pattern)

    def __getitem__(self, key: str) -> "ErrorPattern":
        """Support dictionary-style access."""
        pattern = self.get_pattern(key)
        if pattern is None:
            raise KeyError(key)
        return pattern

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return self.get_pattern(key) is not None


class ErrorPatternAnalytics(BaseService):
    """Optimized error pattern analyzer with memory-efficient data structures."""

    def __init__(self, config: Config):
        super().__init__(
            name="ErrorPatternAnalytics",
            config=config.model_dump() if hasattr(config, "model_dump") else None,
        )

        # Store original config for component initialization
        self._raw_config = config

        # Handle missing error_handling config gracefully
        self.error_analytics_config = getattr(config, "error_handling", None)
        if self.error_analytics_config:
            self.pattern_detection = getattr(
                self.error_analytics_config, "pattern_detection_enabled", True
            )
            self.correlation_analysis = getattr(
                self.error_analytics_config, "correlation_analysis_enabled", True
            )
            self.predictive_alerts = getattr(
                self.error_analytics_config, "predictive_alerts_enabled", True
            )
        else:
            # Default values when error_handling config is not available
            self.pattern_detection = True
            self.correlation_analysis = True
            self.predictive_alerts = True

        # Optimized error tracking with memory limits
        self.error_history = OptimizedErrorHistory(max_size=5000, cleanup_interval_minutes=30)
        self.detected_patterns = OptimizedPatternCache(max_patterns=1000, ttl_hours=48)

        # Limited-size data structures
        self.error_trends: OrderedDict[str, ErrorTrend] = OrderedDict()
        self.correlation_matrix: dict[str, float] = {}  # Flattened for efficiency

        # Performance tracking
        self._analytics_stats: dict[str, int | datetime] = {
            "total_events_processed": 0,
            "patterns_detected": 0,
            "correlations_found": 0,
            "last_analytics_run": datetime.now(timezone.utc),
        }

        # Analytics settings
        self.frequency_threshold = 5  # errors per hour to trigger pattern detection
        self.correlation_threshold = 0.7  # minimum correlation coefficient
        self.trend_window_hours = 24  # hours to analyze for trends
        self.pattern_confidence_threshold = 0.8
        self._pattern_analysis_task: asyncio.Task | None = None

        # Memory management
        self.max_trends = 500
        self.max_correlations = 1000

        # Cleanup task - will be started when needed
        self._cleanup_task: asyncio.Task | None = None
        # Don't start cleanup task immediately - start it when first pattern is added

    def configure_dependencies(self, injector) -> None:
        """Configure dependencies via dependency injector."""
        super().configure_dependencies(injector)

        try:
            # Try to resolve security sanitizer from DI container
            if injector.has_service("SecuritySanitizer"):
                self._sanitizer = injector.resolve("SecuritySanitizer")
                self.logger.debug("SecuritySanitizer resolved from DI container")
            else:
                raise ValueError("SecuritySanitizer service not registered in DI container")

            self.logger.debug("ErrorPatternAnalytics dependencies configured via DI container")
        except Exception as e:
            self.logger.error(f"Failed to configure ErrorPatternAnalytics dependencies via DI: {e}")
            raise

    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup task."""
        try:
            loop = asyncio.get_running_loop()
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = loop.create_task(self._periodic_cleanup())
                # Add done callback to handle any exceptions
                self._cleanup_task.add_done_callback(self._cleanup_task_done_callback)
        except RuntimeError:
            # No event loop running, cleanup task will be started on first use
            pass

    def _cleanup_task_done_callback(self, task: asyncio.Task) -> None:
        """Handle cleanup task completion."""
        try:
            exception = task.exception()
            if exception and not isinstance(exception, asyncio.CancelledError):
                self.logger.error(f"Cleanup task failed: {exception}")
        except asyncio.CancelledError:
            # Task was cancelled, this is expected during shutdown
            pass
        # Reset task reference so it can be restarted if needed
        if self._cleanup_task is task:
            self._cleanup_task = None

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of analytics data."""
        while True:
            try:
                await asyncio.sleep(1800)  # Cleanup every 30 minutes

                # Cleanup trends
                if len(self.error_trends) > self.max_trends:
                    # Remove oldest trends
                    items_to_remove = len(self.error_trends) - self.max_trends
                    for _ in range(items_to_remove):
                        self.error_trends.popitem(last=False)

                # Cleanup correlations
                if len(self.correlation_matrix) > self.max_correlations:
                    # Remove lowest correlations
                    sorted_corrs = sorted(self.correlation_matrix.items(), key=lambda x: x[1])
                    items_to_remove = len(self.correlation_matrix) - self.max_correlations
                    for key, _ in sorted_corrs[:items_to_remove]:
                        del self.correlation_matrix[key]

                # Log statistics
                self.logger.info(
                    "Pattern analytics cleanup completed",
                    history_size=self.error_history.size(),
                    patterns_count=self.detected_patterns.size(),
                    trends_count=len(self.error_trends),
                    correlations_count=len(self.correlation_matrix),
                    total_events=self.error_history.total_events(),
                )

            except Exception as e:
                self.logger.error("Error in pattern analytics cleanup", error=str(e))
                await asyncio.sleep(1800)

    @time_execution
    def add_error_event(self, error_context: dict[str, Any]) -> None:
        """Add an error event to the analytics system with consistent data transformation."""
        # Start cleanup task if not already running
        self._start_cleanup_task()

        # Apply consistent data transformation patterns
        transformed_context = self._transform_error_event_data(error_context)

        # Sanitize details based on sensitivity level
        component = transformed_context.get("component", "unknown")
        severity = transformed_context.get("severity", "medium")
        sensitivity_level = self._get_sensitivity_level(component, severity)

        # Use injected sanitizer - should be configured via DI
        sanitizer = getattr(self, "_sanitizer", None)
        if sanitizer is None:
            raise ValueError("SecuritySanitizer not configured - ensure dependency injection is set up")

        # Sanitize the details before storing
        raw_details = transformed_context.get("details", {})
        sanitized_details = sanitizer.sanitize_context(raw_details, sensitivity_level)

        error_event = {
            "timestamp": datetime.now(timezone.utc),
            "component": component,
            "error_type": transformed_context.get("error_type", "unknown"),
            "severity": severity,
            "operation": transformed_context.get("operation", "unknown"),
            "details": sanitized_details,
            "error_id": transformed_context.get("error_id", "unknown"),
            "processing_mode": transformed_context.get("processing_mode", "sync"),
        }

        # Add to optimized history
        self.error_history.add_event(error_event)
        current_count = self._analytics_stats["total_events_processed"]
        if isinstance(current_count, int):
            self._analytics_stats["total_events_processed"] = current_count + 1

        # Trigger pattern analysis (non-blocking)
        try:
            loop = asyncio.get_running_loop()
            if self._pattern_analysis_task is None or self._pattern_analysis_task.done():
                self._pattern_analysis_task = loop.create_task(self._analyze_patterns())
                # Add done callback to handle any exceptions
                self._pattern_analysis_task.add_done_callback(self._pattern_analysis_done_callback)
        except RuntimeError:
            # No event loop running, skip async pattern analysis
            # This can happen in synchronous contexts like tests
            pass

    def _pattern_analysis_done_callback(self, task: asyncio.Task) -> None:
        """Handle pattern analysis task completion."""
        try:
            exception = task.exception()
            if exception and not isinstance(exception, asyncio.CancelledError):
                self.logger.error(f"Pattern analysis task failed: {exception}")
        except asyncio.CancelledError:
            # Task was cancelled, this is expected during shutdown
            pass
        # Reset task reference so it can be restarted if needed
        if self._pattern_analysis_task is task:
            self._pattern_analysis_task = None

    @time_execution
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
            self.logger.error("Error pattern analysis failed", error=str(e))

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
                        # New pattern detected - create dict-based pattern
                        pattern = {
                            "pattern_id": pattern_id,
                            "pattern_type": "frequency",
                            "component": component,
                            "error_type": error_type,
                            "frequency": count,
                            "severity": self._determine_severity(count),
                            "first_detected": datetime.now(timezone.utc),
                            "last_detected": datetime.now(timezone.utc),
                            "occurrence_count": count,
                            "confidence": min(count / self.frequency_threshold, 1.0),
                            "description": f"High frequency {error_type} errors in {component}",
                            "suggested_action": f"Investigate {component} {error_type} errors",
                            "is_active": True,
                        }

                        self.detected_patterns[pattern_id] = pattern

                        self.logger.warning(
                            "Error pattern detected",
                            pattern_id=pattern_id,
                            frequency=count,
                            severity=pattern["severity"],
                            confidence=pattern["confidence"],
                        )

                        # Trigger alert
                        await self._trigger_pattern_alert(pattern)
                    else:
                        # Update existing pattern
                        pattern = self.detected_patterns[pattern_id]
                        pattern["last_detected"] = datetime.now(timezone.utc)
                        pattern["occurrence_count"] += count
                        pattern["frequency"] = count
                        pattern["confidence"] = min(count / self.frequency_threshold, 1.0)

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

                        self.logger.info(
                            "Error correlation detected",
                            error_type1=error_type1,
                            error_type2=error_type2,
                            correlation=correlation,
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

                    self.logger.info(
                        "Error trend detected",
                        component=component,
                        error_type=error_type,
                        direction=trend.trend_direction,
                        strength=trend.trend_strength,
                    )

    async def _predictive_analysis(self):
        """Perform predictive analysis for potential future errors."""

        # Analyze patterns that might indicate future problems
        for pattern in self.detected_patterns.values():
            if (
                pattern.get("is_active", True)
                and pattern.get("confidence", 0.0) > self.pattern_confidence_threshold
            ):
                # Check if pattern is worsening
                recent_frequency = self._get_recent_frequency(
                    pattern["component"], pattern["error_type"], hours=1
                )

                if recent_frequency > pattern["frequency"] * 1.5:
                    # Pattern is worsening, predict potential issues
                    prediction = await self._predict_issues(pattern)

                    if prediction:
                        self.logger.warning(
                            "Predictive alert",
                            pattern_id=pattern["pattern_id"],
                            prediction=prediction,
                            confidence=pattern["confidence"],
                        )

                        # Trigger predictive alert
                        await self._trigger_predictive_alert(pattern, prediction)

    def _get_recent_errors(self, hours: int) -> list[dict[str, Any]]:
        """Get errors from the last specified hours."""
        return self.error_history.get_recent_events(hours)

    def _create_time_windows(
        self, errors: list[dict[str, Any]], window_minutes: int
    ) -> list[list[dict[str, Any]]]:
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

    def _calculate_correlation(
        self, error_type1: str, error_type2: str, time_windows: list[list[dict[str, Any]]]
    ) -> float:
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

        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(counts1, counts2, strict=False))
        denominator1 = sum((x - mean1) ** 2 for x in counts1)
        denominator2 = sum((y - mean2) ** 2 for y in counts2)

        if denominator1 == 0 or denominator2 == 0:
            return 0.0

        correlation = numerator / (denominator1**0.5 * denominator2**0.5)
        return abs(correlation)  # Return absolute correlation

    def _calculate_trend(self, component: str, error_type: str) -> ErrorTrend | None:
        """Calculate trend for a specific component and error type."""

        recent_errors = self._get_recent_errors(hours=self.trend_window_hours)
        component_errors = [
            error
            for error in recent_errors
            if error["component"] == component and error["error_type"] == error_type
        ]

        if len(component_errors) < 3:
            return None

        # Group by hour
        hourly_counts: dict[datetime, int] = defaultdict(int)
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
        sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
        sum_x2 = sum(x**2 for x in x_values)

        # Calculate slope with division by zero protection
        denominator = n * sum_x2 - sum_x**2
        if denominator == 0:
            return None  # Cannot calculate slope with zero denominator

        slope = (n * sum_xy - sum_x * sum_y) / denominator

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
            data_points=[(hour, hourly_counts[hour]) for hour in sorted_hours],
        )

    def _get_recent_frequency(self, component: str, error_type: str, hours: int) -> int:
        """Get frequency of errors for a component and type in recent hours."""

        recent_errors = self._get_recent_errors(hours=hours)
        return sum(
            1
            for error in recent_errors
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

    async def _predict_issues(self, pattern: dict[str, Any]) -> str | None:
        """Predict potential issues based on error pattern."""

        # Simple prediction logic based on pattern characteristics
        if pattern["frequency"] > 20:
            return f"Critical system degradation in {pattern['component']}"
        elif pattern["frequency"] > 10:
            return f"Performance issues in {pattern['component']}"
        elif pattern["frequency"] > 5:
            return f"Minor issues in {pattern['component']}"
        else:
            return None

    async def _trigger_pattern_alert(self, pattern: dict[str, Any]) -> None:
        """Trigger alert for detected pattern."""

        alert_message = {
            "type": "error_pattern",
            "pattern_id": pattern["pattern_id"],
            "component": pattern["component"],
            "error_type": pattern["error_type"],
            "frequency": pattern["frequency"],
            "severity": pattern["severity"],
            "confidence": pattern["confidence"],
            "description": pattern["description"],
            "suggested_action": pattern["suggested_action"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.logger.warning("Error pattern alert", alert_data=alert_message)

        # Send alert to monitoring module with boundary validation
        await self._send_pattern_to_monitoring(pattern)

    async def _trigger_predictive_alert(self, pattern: dict[str, Any], prediction: str) -> None:
        """Trigger predictive alert."""

        alert_message = {
            "type": "predictive_alert",
            "pattern_id": pattern["pattern_id"],
            "component": pattern["component"],
            "error_type": pattern["error_type"],
            "prediction": prediction,
            "confidence": pattern["confidence"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.logger.warning("Predictive alert", alert_data=alert_message)

        # Alert notification placeholder - implement with notification service

    def get_pattern_summary(self) -> dict[str, Any]:
        """Get summary of detected patterns."""

        # Handle both dict and object patterns
        active_patterns = []
        for p in self.detected_patterns.values():
            if isinstance(p, dict):
                if p.get("is_active", True):
                    active_patterns.append(p)
            else:
                # Handle object patterns
                if getattr(p, "is_active", True):
                    active_patterns.append(p)

        def get_value(pattern, key, default=None):
            """Safely get value from pattern dict or object."""
            if isinstance(pattern, dict):
                return pattern.get(key, default)
            else:
                return getattr(pattern, key, default)

        return {
            "total_patterns": len(self.detected_patterns),
            "active_patterns": len(active_patterns),
            "patterns_by_severity": Counter(
                get_value(p, "severity", "unknown") for p in active_patterns
            ),
            "patterns_by_component": Counter(
                get_value(p, "component", "unknown") for p in active_patterns
            ),
            "recent_patterns": [
                p.to_dict() if hasattr(p, "to_dict") else p
                for p in active_patterns
                if (
                    datetime.now(timezone.utc) - get_value(p, "last_detected", datetime.min.replace(tzinfo=timezone.utc))
                ).total_seconds() < 3600
            ],
        }

    def get_correlation_summary(self) -> dict[str, Any]:
        """Get summary of error correlations."""

        strong_correlations = [
            (key, value)
            for key, value in self.correlation_matrix.items()
            if value >= self.correlation_threshold
        ]

        return {
            "total_correlations": len(self.correlation_matrix),
            "strong_correlations": len(strong_correlations),
            "correlation_threshold": self.correlation_threshold,
            "top_correlations": sorted(strong_correlations, key=lambda x: x[1], reverse=True)[:10],
        }

    def get_trend_summary(self) -> dict[str, Any]:
        """Get summary of error trends."""

        increasing_trends = [
            t for t in self.error_trends.values() if t.trend_direction == "increasing"
        ]
        decreasing_trends = [
            t for t in self.error_trends.values() if t.trend_direction == "decreasing"
        ]

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
                    "strength": t.trend_strength,
                }
                for t in self.error_trends.values()
                if (datetime.now(timezone.utc) - t.end_time).total_seconds() < 3600
            ],
        }

    def _get_sensitivity_level(self, component: str, severity: str) -> SensitivityLevel:
        """Determine sensitivity level for error analytics data."""
        # Financial/trading components always get high sensitivity
        if is_financial_component(component):
            return SensitivityLevel.CRITICAL

        # Map severity to sensitivity level
        severity_mapping = {
            "critical": SensitivityLevel.CRITICAL,
            "high": SensitivityLevel.HIGH,
            "medium": SensitivityLevel.MEDIUM,
            "low": SensitivityLevel.LOW,
        }

        return severity_mapping.get(severity.lower(), SensitivityLevel.MEDIUM)

    async def cleanup(self) -> None:
        """Cleanup resources including async tasks."""

        # Cancel and wait for cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error(f"Failed to await cleanup task: {e}")

        # Cancel and wait for pattern analysis task
        if self._pattern_analysis_task and not self._pattern_analysis_task.done():
            self._pattern_analysis_task.cancel()
            try:
                await self._pattern_analysis_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error(f"Failed to await pattern analysis task: {e}")

        # Clear references
        self._cleanup_task = None
        self._pattern_analysis_task = None

    def _transform_error_event_data(self, error_context: dict[str, Any]) -> dict[str, Any]:
        """Transform error event data consistently across operations."""
        # Apply consistent data transformation patterns matching database module
        transformed_context = error_context.copy()

        # Apply consistent Decimal transformation for financial data
        from src.utils.decimal_utils import to_decimal

        financial_fields = [
            "price",
            "quantity",
            "amount",
            "value",
            "profit",
            "loss",
            "balance",
            "cost",
            "fee",
        ]
        for financial_field in financial_fields:
            if (
                financial_field in transformed_context
                and transformed_context[financial_field] is not None
            ):
                transformed_context[financial_field] = to_decimal(
                    transformed_context[financial_field]
                )

        # Set processing mode for consistent paradigm alignment
        if "processing_mode" not in transformed_context:
            transformed_context["processing_mode"] = "stream"

        # Set audit fields consistently
        if "processed_at" not in transformed_context:
            transformed_context["processed_at"] = datetime.now(timezone.utc).isoformat()

        return transformed_context

    async def add_batch_error_events(self, error_contexts: list[dict[str, Any]]) -> None:
        """Add multiple error events in batch for consistent processing paradigm."""
        if not error_contexts:
            return

        for error_context in error_contexts:
            try:
                self.add_error_event(error_context)
            except Exception as e:
                self.logger.error(f"Failed to add error event in batch: {e}")
                # Continue with other events

    async def _send_pattern_to_monitoring(self, pattern: dict[str, Any]) -> None:
        """Send error pattern to monitoring module via service interface."""
        # Don't directly integrate with monitoring - leave for service layer
        self.logger.info(
            "Pattern detected - should be sent to monitoring via service layer",
            pattern_id=pattern["pattern_id"],
            component=pattern["component"],
            severity=pattern["severity"],
        )
