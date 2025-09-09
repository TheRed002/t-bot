"""
Secure pattern analytics backward compatibility.

Redirects to simplified security validator.
"""

import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.utils.security_types import (
    SeverityLevel as PatternSeverity,  # Alias for backward compatibility
    ThreatType,
)

from .security_validator import get_security_sanitizer

DEFAULT_MIN_OCCURRENCES = 5
DEFAULT_ANALYSIS_WINDOW_HOURS = 24
DEFAULT_PATTERN_SIMILARITY_THRESHOLD = 0.8
DEFAULT_THREAT_SCORE_THRESHOLD = Decimal("7.0")
DEFAULT_MAX_EVENTS = 100000


@dataclass
class ErrorPattern:
    """Enhanced error pattern with security analytics."""

    pattern_id: str
    pattern_type: str
    severity: PatternSeverity
    frequency: int
    first_seen: datetime
    last_seen: datetime
    error_signature: str
    component_hash: str
    common_context: dict[str, Any]

    # Optional fields
    risk_score: Decimal = Decimal("0.0")
    affected_users: int = 0
    affected_ips: int = 0
    threat_types: list[ThreatType] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    geographic_distribution: dict[str, int] = field(default_factory=dict)
    time_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class AnalyticsConfig:
    """Analytics configuration for backward compatibility."""

    min_occurrences: int = DEFAULT_MIN_OCCURRENCES
    analysis_window_hours: int = DEFAULT_ANALYSIS_WINDOW_HOURS
    pattern_similarity_threshold: Decimal = Decimal(str(DEFAULT_PATTERN_SIMILARITY_THRESHOLD))
    enable_threat_detection: bool = True
    auto_threat_scoring: bool = True
    threat_score_threshold: Decimal = DEFAULT_THREAT_SCORE_THRESHOLD


class SecurePatternAnalytics:
    """Simplified pattern analytics for backward compatibility."""

    def __init__(self, config: AnalyticsConfig | None = None):
        self.config = config or AnalyticsConfig()
        self.sanitizer = get_security_sanitizer()
        from src.core.logging import get_logger

        self.logger = get_logger(__name__)
        self._error_events = deque(maxlen=DEFAULT_MAX_EVENTS)
        self._detected_patterns = {}
        self._start_background_analysis()

    def _start_background_analysis(self) -> None:
        """Start background analysis (stub for backward compatibility)."""
        # Background analysis not implemented in current version
        return

    def record_error_event(
        self, error: Exception, component: str, security_context, error_context: dict[str, Any]
    ) -> None:
        """Record an error event for analysis."""
        sanitized_event = self._create_sanitized_event(
            error, component, security_context, error_context
        )
        self._error_events.append(sanitized_event)

    def _create_sanitized_event(
        self, error: Exception, component: str, security_context, additional_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a sanitized error event."""
        return {
            "timestamp": datetime.now(timezone.utc),
            "error_type": type(error).__name__,
            "error_signature": self._create_error_signature(error, additional_context),
            "sanitized_message": self.sanitizer.sanitize_error_message(str(error)),
            "component_hash": hashlib.sha256(component.encode()).hexdigest()[:16],
            "context": self.sanitizer.sanitize_context(additional_context),
            "user_hash": getattr(security_context, "user_id", "anonymous"),
            "ip_hash": getattr(security_context, "client_ip", "unknown"),
        }

    def _create_error_signature(self, error: Exception, context: dict[str, Any]) -> str:
        """Create a signature for error pattern matching."""
        signature_parts = [
            type(error).__name__,
            context.get("component", "unknown"),
            context.get("operation", "unknown"),
        ]
        signature_string = "|".join(signature_parts)
        return hashlib.sha256(signature_string.encode()).hexdigest()[:16]

    async def analyze_patterns(self) -> list[ErrorPattern]:
        """Analyze error patterns from recorded events."""
        if len(self._error_events) < self.config.min_occurrences:
            return []

        patterns = []
        pattern_counts = {}

        # Count error patterns
        for event in self._error_events:
            signature = event["error_signature"]
            if signature not in pattern_counts:
                pattern_counts[signature] = {
                    "count": 0,
                    "first_seen": event["timestamp"],
                    "last_seen": event["timestamp"],
                    "events": [],
                }
            pattern_counts[signature]["count"] += 1
            pattern_counts[signature]["last_seen"] = event["timestamp"]
            pattern_counts[signature]["events"].append(event)

        # Create patterns for frequent errors
        for signature, data in pattern_counts.items():
            if data["count"] >= self.config.min_occurrences:
                pattern = ErrorPattern(
                    pattern_id=signature,
                    pattern_type="frequency",
                    severity=self._determine_severity(data["count"]),
                    frequency=data["count"],
                    first_seen=data["first_seen"],
                    last_seen=data["last_seen"],
                    error_signature=signature,
                    component_hash=data["events"][0]["component_hash"],
                    common_context={"error_type": data["events"][0]["error_type"]},
                )

                # Add threat detection if enabled
                if self.config.enable_threat_detection:
                    pattern.threat_types = self._detect_threats_in_pattern(pattern, data["events"])

                # Calculate risk score if auto-scoring is enabled
                if self.config.auto_threat_scoring:
                    pattern.risk_score = self._calculate_risk_score(pattern, data["events"])

                patterns.append(pattern)
                self._detected_patterns[signature] = pattern

        return patterns

    def _determine_severity(self, frequency: int) -> PatternSeverity:
        """Determine pattern severity based on frequency."""
        if frequency >= 100:
            return PatternSeverity.CRITICAL
        elif frequency >= 50:
            return PatternSeverity.HIGH
        elif frequency >= 10:
            return PatternSeverity.MEDIUM
        else:
            return PatternSeverity.LOW

    def _detect_threats_in_pattern(
        self, pattern: ErrorPattern, events: list[dict[str, Any]]
    ) -> list[ThreatType]:
        """Detect potential threats in the pattern."""
        threats = []

        # Check for brute force patterns (many failures from few IPs)
        unique_ips = set(event["ip_hash"] for event in events)
        unique_users = set(event["user_hash"] for event in events)

        if len(unique_ips) == 1 and len(unique_users) > 5:
            threats.append(ThreatType.BRUTE_FORCE)

        # Check for credential stuffing (many users, few IPs)
        if len(unique_ips) < 5 and len(unique_users) > 10:
            threats.append(ThreatType.CREDENTIAL_STUFFING)

        # Check for DoS patterns (high frequency)
        if pattern.frequency > 1000:
            threats.append(ThreatType.DOS_ATTACK)

        return threats

    def _calculate_risk_score(self, pattern: ErrorPattern, events: list[dict[str, Any]]) -> Decimal:
        """Calculate risk score for the pattern."""
        base_score = Decimal("0.0")

        # Frequency-based scoring
        if pattern.frequency > 1000:
            base_score += Decimal("5.0")
        elif pattern.frequency > 100:
            base_score += Decimal("3.0")
        elif pattern.frequency > 10:
            base_score += Decimal("1.0")

        # Threat-based scoring
        threat_scores = {
            ThreatType.BRUTE_FORCE: Decimal("3.0"),
            ThreatType.CREDENTIAL_STUFFING: Decimal("4.0"),
            ThreatType.DOS_ATTACK: Decimal("5.0"),
            ThreatType.DATA_EXFILTRATION: Decimal("8.0"),
            ThreatType.PRIVILEGE_ESCALATION: Decimal("9.0"),
        }

        for threat in pattern.threat_types:
            base_score += threat_scores.get(threat, Decimal("1.0"))

        # Affected users/IPs multiplier
        if pattern.affected_users > 100:
            base_score *= Decimal("1.5")
        elif pattern.affected_users > 10:
            base_score *= Decimal("1.2")

        return min(base_score, Decimal("10.0"))  # Cap at 10.0

    def _generate_recommendations(self) -> None:
        """Generate security recommendations (stub)."""
        # Recommendation generation not implemented in current version
        return

    async def get_patterns_summary(
        self, severity_filter: PatternSeverity | None = None
    ) -> dict[str, Any]:
        """Get a summary of detected patterns."""
        patterns = list(self._detected_patterns.values())

        if severity_filter:
            patterns = [p for p in patterns if p.severity == severity_filter]

        # Count by severity
        severity_counts = {}
        for pattern in patterns:
            severity_key = pattern.severity.value
            severity_counts[severity_key] = severity_counts.get(severity_key, 0) + 1

        # Count by threat type
        threat_counts = {}
        for pattern in patterns:
            for threat in pattern.threat_types:
                threat_key = threat.value
                threat_counts[threat_key] = threat_counts.get(threat_key, 0) + 1

        # Component statistics
        component_stats = {}
        for pattern in patterns:
            comp = pattern.component_hash
            component_stats[comp] = component_stats.get(comp, 0) + 1

        return {
            "total_patterns": len(patterns),
            "patterns_by_severity": severity_counts,
            "patterns_by_threat_type": threat_counts,
            "component_statistics": component_stats,
        }


__all__ = [
    "AnalyticsConfig",
    "ErrorPattern",
    "PatternSeverity",
    "SecurePatternAnalytics",
    "ThreatType",
]
