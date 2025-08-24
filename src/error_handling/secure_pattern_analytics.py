"""
Secure error pattern analytics system.

This module provides comprehensive error pattern analysis while preventing
exposure of internal system architecture details and sensitive information.

CRITICAL: Analyzes error patterns for security threats and system issues
without revealing system internals, file paths, or sensitive configurations.
"""

import asyncio
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from src.core.logging import get_logger
from src.error_handling.secure_context_manager import (
    SecurityContext,
    UserRole,
)
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)


class PatternSeverity(Enum):
    """Severity levels for error patterns."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats detected in patterns."""

    BRUTE_FORCE = "brute_force"
    CREDENTIAL_STUFFING = "credential_stuffing"
    ACCOUNT_ENUMERATION = "account_enumeration"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    API_ABUSE = "api_abuse"
    DOS_ATTACK = "dos_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SYSTEM_RECONNAISSANCE = "system_reconnaissance"
    CONFIGURATION_LEAK = "configuration_leak"


@dataclass
class ErrorPattern:
    """Represents a detected error pattern."""

    pattern_id: str
    pattern_type: str
    severity: PatternSeverity
    frequency: int
    first_seen: datetime
    last_seen: datetime

    # Pattern characteristics (sanitized)
    error_signature: str  # Hash of sanitized error
    component_hash: str  # Hash of component name
    common_context: dict[str, Any]

    # Threat analysis
    threat_types: list[ThreatType] = field(default_factory=list)
    risk_score: float = 0.0

    # Metadata (no sensitive info)
    affected_users: int = 0
    affected_ips: int = 0
    geographic_distribution: dict[str, int] = field(default_factory=dict)
    time_distribution: dict[str, int] = field(default_factory=dict)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    auto_mitigation: str | None = None


@dataclass
class AnalyticsConfig:
    """Configuration for pattern analytics."""

    # Pattern detection settings
    min_occurrences: int = 5
    analysis_window_hours: int = 24
    pattern_similarity_threshold: float = 0.8

    # Threat detection settings
    enable_threat_detection: bool = True
    auto_threat_scoring: bool = True
    threat_score_threshold: float = 7.0

    # Data retention
    pattern_retention_days: int = 90
    raw_data_retention_days: int = 7

    # Performance settings
    max_patterns_tracked: int = 10000
    analysis_batch_size: int = 1000
    background_analysis_interval: int = 300  # 5 minutes

    # Security settings
    sanitization_level: SensitivityLevel = SensitivityLevel.HIGH
    anonymize_user_data: bool = True
    anonymize_ip_data: bool = True


class SecurePatternAnalytics:
    """
    Secure error pattern analytics with privacy protection.

    Features:
    - Pattern detection without exposing system internals
    - Threat intelligence and security analytics
    - Privacy-preserving user and IP analytics
    - Geographic and temporal pattern analysis
    - Automated threat scoring and risk assessment
    - Actionable security recommendations
    - Integration with monitoring and alerting systems
    """

    def __init__(self, config: AnalyticsConfig = None):
        self.config = config or AnalyticsConfig()
        self.logger = get_logger(self.__class__.__module__)
        self.sanitizer = get_security_sanitizer()

        # Pattern storage
        self._error_events: deque = deque(maxlen=100000)  # Recent error events
        self._detected_patterns: dict[str, ErrorPattern] = {}
        self._pattern_signatures: dict[str, set[str]] = defaultdict(set)  # signature -> pattern_ids

        # Analytics data structures
        self._component_stats: dict[str, dict[str, Any]] = defaultdict(dict)
        self._user_patterns: dict[str, list[str]] = defaultdict(list)  # anonymized
        self._ip_patterns: dict[str, list[str]] = defaultdict(list)  # anonymized
        self._temporal_patterns: dict[str, list[datetime]] = defaultdict(list)

        # Threat intelligence
        self._threat_signatures: dict[str, list[ThreatType]] = {}
        self._known_attack_patterns: set[str] = set()

        # Background analysis task
        self._analysis_task: asyncio.Task | None = None
        self._start_background_analysis()

        # Initialize threat detection patterns
        self._initialize_threat_patterns()

    def _initialize_threat_patterns(self) -> None:
        """Initialize known threat patterns for detection."""

        # Brute force patterns
        self._threat_signatures["brute_force_auth"] = [ThreatType.BRUTE_FORCE]
        self._threat_signatures["repeated_login_failures"] = [ThreatType.BRUTE_FORCE]
        self._threat_signatures["password_spray"] = [ThreatType.CREDENTIAL_STUFFING]

        # Injection attack patterns
        self._threat_signatures["sql_syntax_error"] = [ThreatType.SQL_INJECTION]
        self._threat_signatures["script_injection"] = [ThreatType.XSS_ATTACK]

        # API abuse patterns
        self._threat_signatures["rate_limit_exceeded"] = [ThreatType.API_ABUSE]
        self._threat_signatures["quota_exhausted"] = [ThreatType.API_ABUSE]

        # System reconnaissance
        self._threat_signatures["path_traversal"] = [ThreatType.SYSTEM_RECONNAISSANCE]
        self._threat_signatures["directory_enumeration"] = [ThreatType.SYSTEM_RECONNAISSANCE]

        # DoS patterns
        self._threat_signatures["resource_exhaustion"] = [ThreatType.DOS_ATTACK]
        self._threat_signatures["connection_flooding"] = [ThreatType.DOS_ATTACK]

    def record_error_event(
        self,
        error: Exception,
        component: str,
        security_context: SecurityContext | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an error event for pattern analysis.

        Args:
            error: Exception that occurred
            component: Component where error occurred
            security_context: Security context
            additional_context: Additional context information
        """

        # Create sanitized error event
        sanitized_event = self._create_sanitized_event(
            error, component, security_context, additional_context
        )

        # Add to event queue
        self._error_events.append(sanitized_event)

        # Update component statistics
        self._update_component_stats(sanitized_event)

        # Update user/IP patterns (anonymized)
        self._update_entity_patterns(sanitized_event)

        # Update temporal patterns
        self._update_temporal_patterns(sanitized_event)

        # Trigger immediate analysis for critical events
        if self._is_critical_event(sanitized_event):
            asyncio.create_task(self._analyze_immediate_threats(sanitized_event))

    async def analyze_patterns(self) -> list[ErrorPattern]:
        """
        Analyze error events for patterns.

        Returns:
            List of detected patterns
        """

        self.logger.info("Starting pattern analysis")

        # Group events by similarity
        pattern_groups = self._group_similar_events()

        # Detect new patterns
        new_patterns = []
        for group_signature, events in pattern_groups.items():
            if len(events) >= self.config.min_occurrences:
                pattern = self._analyze_pattern_group(group_signature, events)
                if pattern:
                    new_patterns.append(pattern)

        # Update existing patterns
        self._update_existing_patterns(new_patterns)

        # Perform threat analysis
        await self._analyze_threats()

        # Generate recommendations
        self._generate_recommendations()

        # Cleanup old patterns
        self._cleanup_old_patterns()

        self.logger.info(f"Pattern analysis complete. {len(new_patterns)} new patterns detected")

        return list(self._detected_patterns.values())

    def _create_sanitized_event(
        self,
        error: Exception,
        component: str,
        security_context: SecurityContext | None,
        additional_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create sanitized error event for analysis."""

        # Sanitize error message
        sanitized_message = self.sanitizer.sanitize_error_message(
            str(error), self.config.sanitization_level
        )

        # Create error signature (hash of sanitized content)
        error_signature = self._create_error_signature(error, sanitized_message)

        # Hash component name to prevent architecture exposure
        component_hash = self._hash_identifier(component)

        # Sanitize and anonymize context
        sanitized_context = self._sanitize_context(additional_context or {})

        event = {
            "timestamp": datetime.now(timezone.utc),
            "error_type": type(error).__name__,
            "error_signature": error_signature,
            "sanitized_message": sanitized_message,
            "component_hash": component_hash,
            "context": sanitized_context,
        }

        # Add anonymized security context
        if security_context:
            event.update(self._anonymize_security_context(security_context))

        return event

    def _create_error_signature(self, error: Exception, sanitized_message: str) -> str:
        """Create unique signature for error pattern matching."""

        # Combine error type and sanitized message
        signature_data = f"{type(error).__name__}:{sanitized_message}"

        # Create hash
        hash_obj = hashlib.sha256(signature_data.encode("utf-8"))
        return hash_obj.hexdigest()[:16]

    def _hash_identifier(self, identifier: str) -> str:
        """Create hash of identifier for privacy."""
        hash_obj = hashlib.sha256(identifier.encode("utf-8"))
        return f"H_{hash_obj.hexdigest()[:12]}"

    def _sanitize_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Sanitize context while preserving analytical value."""

        # Use sanitizer for basic sanitization
        sanitized = self.sanitizer.sanitize_context(context, self.config.sanitization_level)

        # Additional analytics-specific sanitization
        analytics_safe_context = {}

        for key, value in sanitized.items():
            # Keep certain keys that are useful for analysis
            safe_keys = [
                "user_agent_category",
                "request_method",
                "status_code",
                "response_time",
                "retry_count",
                "error_category",
                "geographic_region",
                "time_of_day",
                "day_of_week",
            ]

            if key in safe_keys:
                analytics_safe_context[key] = value
            elif isinstance(value, (int, float, bool)):
                # Keep numeric and boolean values (less sensitive)
                analytics_safe_context[key] = value
            elif isinstance(value, str) and len(value) < 50:
                # Keep short string values that might be categories
                analytics_safe_context[f"category_{self._hash_identifier(key)}"] = (
                    self._hash_identifier(value)
                )

        return analytics_safe_context

    def _anonymize_security_context(self, security_context: SecurityContext) -> dict[str, Any]:
        """Anonymize security context for pattern analysis."""

        anonymized = {}

        # Anonymize user ID
        if security_context.user_id and not self.config.anonymize_user_data:
            anonymized["user_hash"] = self._hash_identifier(security_context.user_id)
        elif security_context.user_id:
            anonymized["has_user"] = True

        # Anonymize IP address
        if security_context.client_ip and not self.config.anonymize_ip_data:
            anonymized["ip_hash"] = self._hash_identifier(security_context.client_ip)
            # Keep geographic region if available
            anonymized["ip_region"] = self._get_ip_region(security_context.client_ip)
        elif security_context.client_ip:
            anonymized["has_ip"] = True
            anonymized["ip_region"] = self._get_ip_region(security_context.client_ip)

        # Keep non-sensitive context
        anonymized["user_role"] = security_context.user_role.value
        anonymized["is_authenticated"] = security_context.is_authenticated
        anonymized["is_internal"] = security_context.is_internal_user

        return anonymized

    def _get_ip_region(self, ip_address: str) -> str:
        """Get geographic region from IP (simplified implementation)."""

        # This is a placeholder - in production, you'd use a GeoIP service
        # but only return region/country, not specific location

        if ip_address.startswith("192.168.") or ip_address.startswith("10."):
            return "internal"
        elif ip_address.startswith("127."):
            return "localhost"
        else:
            # In production, use GeoIP lookup returning only region
            return "external"

    def _group_similar_events(self) -> dict[str, list[dict[str, Any]]]:
        """Group similar error events for pattern detection."""

        pattern_groups = defaultdict(list)

        # Analyze events within the configured window
        cutoff_time = datetime.now(timezone.utc) - timedelta(
            hours=self.config.analysis_window_hours
        )

        for event in self._error_events:
            if event["timestamp"] < cutoff_time:
                continue

            # Group by error signature
            signature = event["error_signature"]
            pattern_groups[signature].append(event)

        # Filter groups by minimum occurrences
        return {
            sig: events
            for sig, events in pattern_groups.items()
            if len(events) >= self.config.min_occurrences
        }

    def _analyze_pattern_group(
        self, signature: str, events: list[dict[str, Any]]
    ) -> ErrorPattern | None:
        """Analyze a group of similar events to detect patterns."""

        if not events:
            return None

        # Sort events by timestamp
        events.sort(key=lambda e: e["timestamp"])

        first_event = events[0]
        last_event = events[-1]

        # Calculate frequency and timing
        time_span = (last_event["timestamp"] - first_event["timestamp"]).total_seconds()
        frequency = len(events)

        # Analyze common context
        common_context = self._extract_common_context(events)

        # Determine severity
        severity = self._calculate_pattern_severity(events, time_span)

        # Create pattern ID
        pattern_id = f"PAT_{signature}_{int(first_event['timestamp'].timestamp())}"

        # Count unique entities
        unique_users = len(set(e.get("user_hash") for e in events if e.get("user_hash")))
        unique_ips = len(set(e.get("ip_hash") for e in events if e.get("ip_hash")))

        # Analyze geographic distribution
        geo_dist = self._analyze_geographic_distribution(events)

        # Analyze time distribution
        time_dist = self._analyze_time_distribution(events)

        pattern = ErrorPattern(
            pattern_id=pattern_id,
            pattern_type=first_event["error_type"],
            severity=severity,
            frequency=frequency,
            first_seen=first_event["timestamp"],
            last_seen=last_event["timestamp"],
            error_signature=signature,
            component_hash=first_event["component_hash"],
            common_context=common_context,
            affected_users=unique_users,
            affected_ips=unique_ips,
            geographic_distribution=geo_dist,
            time_distribution=time_dist,
        )

        # Detect threats
        pattern.threat_types = self._detect_threats_in_pattern(pattern, events)
        pattern.risk_score = self._calculate_risk_score(pattern, events)

        return pattern

    def _extract_common_context(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract common context elements from events."""

        if not events:
            return {}

        common_context = {}

        # Find common keys across all events
        common_keys = set(events[0].get("context", {}).keys())
        for event in events[1:]:
            common_keys &= set(event.get("context", {}).keys())

        # For each common key, find common values
        for key in common_keys:
            values = [event["context"][key] for event in events if key in event["context"]]

            # If all values are the same, include it
            unique_values = set(str(v) for v in values)
            if len(unique_values) == 1:
                common_context[key] = values[0]
            elif len(unique_values) <= 3:  # Include if there are few unique values
                common_context[f"{key}_variants"] = list(unique_values)

        return common_context

    def _calculate_pattern_severity(
        self, events: list[dict[str, Any]], time_span: float
    ) -> PatternSeverity:
        """Calculate severity of error pattern."""

        frequency = len(events)
        rate = frequency / max(time_span / 3600, 1)  # Errors per hour

        # Count unique entities affected
        unique_users = len(set(e.get("user_hash") for e in events if e.get("user_hash")))
        unique_ips = len(set(e.get("ip_hash") for e in events if e.get("ip_hash")))

        # Severity criteria
        if rate > 100 or unique_users > 50 or unique_ips > 20:
            return PatternSeverity.CRITICAL
        elif rate > 20 or unique_users > 10 or unique_ips > 5:
            return PatternSeverity.HIGH
        elif rate > 5 or unique_users > 2 or unique_ips > 2:
            return PatternSeverity.MEDIUM
        else:
            return PatternSeverity.LOW

    def _analyze_geographic_distribution(self, events: list[dict[str, Any]]) -> dict[str, int]:
        """Analyze geographic distribution of events."""

        geo_dist = defaultdict(int)

        for event in events:
            region = event.get("ip_region", "unknown")
            geo_dist[region] += 1

        return dict(geo_dist)

    def _analyze_time_distribution(self, events: list[dict[str, Any]]) -> dict[str, int]:
        """Analyze temporal distribution of events."""

        time_dist = defaultdict(int)

        for event in events:
            timestamp = event["timestamp"]
            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            time_dist[f"hour_{hour}"] += 1
            time_dist[f"day_{day_of_week}"] += 1

        return dict(time_dist)

    def _detect_threats_in_pattern(
        self, pattern: ErrorPattern, events: list[dict[str, Any]]
    ) -> list[ThreatType]:
        """Detect security threats in error pattern."""

        if not self.config.enable_threat_detection:
            return []

        detected_threats = []

        # Check against known threat signatures
        for threat_sig, threat_types in self._threat_signatures.items():
            if any(threat_sig.lower() in event["sanitized_message"].lower() for event in events):
                detected_threats.extend(threat_types)

        # Behavioral threat detection

        # Brute force detection
        if (
            pattern.frequency > 10
            and pattern.affected_users < pattern.frequency / 5
            and "authentication" in pattern.common_context.get("error_category", "").lower()
        ):
            detected_threats.append(ThreatType.BRUTE_FORCE)

        # Credential stuffing detection
        if pattern.affected_users > pattern.affected_ips * 3 and pattern.frequency > 50:
            detected_threats.append(ThreatType.CREDENTIAL_STUFFING)

        # Account enumeration detection
        if pattern.affected_users > 20 and pattern.frequency / pattern.affected_users < 2:
            detected_threats.append(ThreatType.ACCOUNT_ENUMERATION)

        # API abuse detection
        if (
            pattern.frequency > 100
            and (pattern.last_seen - pattern.first_seen).total_seconds() < 3600
        ):
            detected_threats.append(ThreatType.API_ABUSE)

        # DoS attack detection
        if pattern.frequency > 1000 or (pattern.frequency > 100 and pattern.affected_ips < 5):
            detected_threats.append(ThreatType.DOS_ATTACK)

        return list(set(detected_threats))

    def _calculate_risk_score(self, pattern: ErrorPattern, events: list[dict[str, Any]]) -> float:
        """Calculate risk score for pattern."""

        if not self.config.auto_threat_scoring:
            return 0.0

        score = 0.0

        # Base score from frequency
        score += min(pattern.frequency / 10, 3.0)

        # Severity multiplier
        severity_multipliers = {
            PatternSeverity.LOW: 1.0,
            PatternSeverity.MEDIUM: 2.0,
            PatternSeverity.HIGH: 4.0,
            PatternSeverity.CRITICAL: 6.0,
        }
        score *= severity_multipliers[pattern.severity]

        # Threat type scoring
        threat_scores = {
            ThreatType.BRUTE_FORCE: 2.0,
            ThreatType.CREDENTIAL_STUFFING: 3.0,
            ThreatType.SQL_INJECTION: 4.0,
            ThreatType.XSS_ATTACK: 3.0,
            ThreatType.DOS_ATTACK: 3.5,
            ThreatType.DATA_EXFILTRATION: 5.0,
            ThreatType.PRIVILEGE_ESCALATION: 4.5,
        }

        for threat_type in pattern.threat_types:
            score += threat_scores.get(threat_type, 1.0)

        # Entity spread factor
        if pattern.affected_users > 0:
            score += min(pattern.affected_users / 10, 2.0)
        if pattern.affected_ips > 0:
            score += min(pattern.affected_ips / 5, 2.0)

        # Time concentration factor
        time_span_hours = (pattern.last_seen - pattern.first_seen).total_seconds() / 3600
        if time_span_hours < 1:
            score += 1.5  # Events concentrated in time
        elif time_span_hours < 6:
            score += 0.5

        return min(score, 10.0)  # Cap at 10

    def _update_existing_patterns(self, new_patterns: list[ErrorPattern]) -> None:
        """Update existing patterns with new data."""

        for pattern in new_patterns:
            if pattern.pattern_id in self._detected_patterns:
                # Update existing pattern
                existing = self._detected_patterns[pattern.pattern_id]
                existing.frequency += pattern.frequency
                existing.last_seen = max(existing.last_seen, pattern.last_seen)
                existing.affected_users = max(existing.affected_users, pattern.affected_users)
                existing.affected_ips = max(existing.affected_ips, pattern.affected_ips)

                # Update threat types and risk score
                existing.threat_types = list(set(existing.threat_types + pattern.threat_types))
                existing.risk_score = max(existing.risk_score, pattern.risk_score)
            else:
                # Add new pattern
                self._detected_patterns[pattern.pattern_id] = pattern

    async def _analyze_threats(self) -> None:
        """Perform comprehensive threat analysis."""

        for pattern in self._detected_patterns.values():
            if pattern.risk_score > self.config.threat_score_threshold:
                self.logger.warning(
                    f"High-risk pattern detected: {pattern.pattern_id}",
                    risk_score=pattern.risk_score,
                    threat_types=[t.value for t in pattern.threat_types],
                    frequency=pattern.frequency,
                    severity=pattern.severity.value,
                )

    def _generate_recommendations(self) -> None:
        """Generate actionable recommendations for patterns."""

        for pattern in self._detected_patterns.values():
            recommendations = []

            # Threat-specific recommendations
            if ThreatType.BRUTE_FORCE in pattern.threat_types:
                recommendations.extend(
                    [
                        "Implement progressive delays for failed authentication attempts",
                        "Enable account lockout after multiple failures",
                        "Monitor and alert on authentication patterns",
                    ]
                )

            if ThreatType.API_ABUSE in pattern.threat_types:
                recommendations.extend(
                    [
                        "Implement rate limiting per IP address",
                        "Add API key authentication requirements",
                        "Monitor API usage patterns",
                    ]
                )

            if ThreatType.DOS_ATTACK in pattern.threat_types:
                recommendations.extend(
                    [
                        "Implement DDoS protection measures",
                        "Add rate limiting and traffic shaping",
                        "Consider IP-based blocking for repeat offenders",
                    ]
                )

            # Severity-based recommendations
            if pattern.severity == PatternSeverity.CRITICAL:
                recommendations.extend(
                    [
                        "Immediate investigation required",
                        "Consider temporary service restrictions",
                        "Alert security team",
                    ]
                )
            elif pattern.severity == PatternSeverity.HIGH:
                recommendations.extend(
                    [
                        "Escalate for security review",
                        "Implement additional monitoring",
                    ]
                )

            pattern.recommendations = recommendations

    def _cleanup_old_patterns(self) -> None:
        """Clean up old patterns and data."""

        cutoff_time = datetime.now(timezone.utc) - timedelta(
            days=self.config.pattern_retention_days
        )

        # Remove old patterns
        old_patterns = [
            pid
            for pid, pattern in self._detected_patterns.items()
            if pattern.last_seen < cutoff_time
        ]

        for pid in old_patterns:
            del self._detected_patterns[pid]

        # Trim event queue
        raw_cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.config.raw_data_retention_days
        )
        while self._error_events and self._error_events[0]["timestamp"] < raw_cutoff:
            self._error_events.popleft()

    def _update_component_stats(self, event: dict[str, Any]) -> None:
        """Update component statistics."""

        component_hash = event["component_hash"]
        stats = self._component_stats[component_hash]

        stats["error_count"] = stats.get("error_count", 0) + 1
        stats["last_error"] = event["timestamp"]

        error_type = event["error_type"]
        error_types = stats.get("error_types", {})
        error_types[error_type] = error_types.get(error_type, 0) + 1
        stats["error_types"] = error_types

    def _update_entity_patterns(self, event: dict[str, Any]) -> None:
        """Update entity (user/IP) patterns."""

        user_hash = event.get("user_hash")
        if user_hash:
            self._user_patterns[user_hash].append(event["error_signature"])

        ip_hash = event.get("ip_hash")
        if ip_hash:
            self._ip_patterns[ip_hash].append(event["error_signature"])

    def _update_temporal_patterns(self, event: dict[str, Any]) -> None:
        """Update temporal patterns."""

        signature = event["error_signature"]
        self._temporal_patterns[signature].append(event["timestamp"])

    def _is_critical_event(self, event: dict[str, Any]) -> bool:
        """Check if event requires immediate analysis."""

        critical_keywords = [
            "security",
            "breach",
            "attack",
            "unauthorized",
            "injection",
            "overflow",
            "exploit",
            "malware",
            "intrusion",
        ]

        message = event["sanitized_message"].lower()
        return any(keyword in message for keyword in critical_keywords)

    async def _analyze_immediate_threats(self, event: dict[str, Any]) -> None:
        """Analyze immediate threats from critical events."""

        self.logger.warning(
            f"Critical security event detected: {event['error_type']}",
            component_hash=event["component_hash"],
            error_signature=event["error_signature"],
        )

    def _start_background_analysis(self) -> None:
        """Start background pattern analysis task."""

        async def analysis_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.background_analysis_interval)
                    await self.analyze_patterns()
                except Exception as e:
                    self.logger.error(f"Error in background analysis: {e}")

        self._analysis_task = asyncio.create_task(analysis_loop())

    async def get_patterns_summary(
        self,
        security_context: SecurityContext | None = None,
        severity_filter: PatternSeverity | None = None,
    ) -> dict[str, Any]:
        """Get summary of detected patterns."""

        patterns = list(self._detected_patterns.values())

        # Filter by severity if requested
        if severity_filter:
            patterns = [p for p in patterns if p.severity == severity_filter]

        # Filter based on user role
        if security_context and security_context.user_role in [UserRole.GUEST, UserRole.USER]:
            # Provide minimal information for low-privilege users
            return {
                "total_patterns": len(patterns),
                "critical_patterns": len(
                    [p for p in patterns if p.severity == PatternSeverity.CRITICAL]
                ),
                "high_risk_patterns": len([p for p in patterns if p.risk_score > 7.0]),
            }

        # Detailed summary for authorized users
        summary = {
            "total_patterns": len(patterns),
            "patterns_by_severity": {
                severity.value: len([p for p in patterns if p.severity == severity])
                for severity in PatternSeverity
            },
            "patterns_by_threat_type": {},
            "high_risk_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "severity": p.severity.value,
                    "risk_score": p.risk_score,
                    "threat_types": [t.value for t in p.threat_types],
                    "frequency": p.frequency,
                    "recommendations": p.recommendations,
                }
                for p in patterns
                if p.risk_score > 7.0
            ],
            "component_statistics": self._get_component_statistics(),
            "temporal_analysis": self._get_temporal_analysis(),
        }

        # Count patterns by threat type
        threat_counts = defaultdict(int)
        for pattern in patterns:
            for threat_type in pattern.threat_types:
                threat_counts[threat_type.value] += 1
        summary["patterns_by_threat_type"] = dict(threat_counts)

        return summary

    def _get_component_statistics(self) -> dict[str, Any]:
        """Get component statistics."""

        return {
            "components_with_errors": len(self._component_stats),
            "top_error_components": sorted(
                [
                    {"component_hash": comp_hash, "error_count": stats["error_count"]}
                    for comp_hash, stats in self._component_stats.items()
                ],
                key=lambda x: x["error_count"],
                reverse=True,
            )[:10],
        }

    def _get_temporal_analysis(self) -> dict[str, Any]:
        """Get temporal analysis of patterns."""

        now = datetime.now(timezone.utc)

        # Count patterns by time period
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)

        for events in self._temporal_patterns.values():
            for timestamp in events:
                if isinstance(timestamp, datetime) and (now - timestamp).days < 7:
                    hour_counts[timestamp.hour] += 1
                    day_counts[timestamp.weekday()] += 1

        return {
            "patterns_by_hour": dict(hour_counts),
            "patterns_by_day_of_week": dict(day_counts),
        }

    def get_system_health(self) -> dict[str, Any]:
        """Get analytics system health information."""

        return {
            "error_events_tracked": len(self._error_events),
            "patterns_detected": len(self._detected_patterns),
            "components_monitored": len(self._component_stats),
            "background_analysis_running": self._analysis_task and not self._analysis_task.done(),
            "config": {
                "min_occurrences": self.config.min_occurrences,
                "analysis_window_hours": self.config.analysis_window_hours,
                "threat_detection_enabled": self.config.enable_threat_detection,
            },
        }


# Global analytics instance
_global_analytics = None


def get_pattern_analytics(config: AnalyticsConfig = None) -> SecurePatternAnalytics:
    """Get global pattern analytics instance."""
    global _global_analytics
    if _global_analytics is None:
        _global_analytics = SecurePatternAnalytics(config)
    return _global_analytics


def record_error_for_analysis(
    error: Exception,
    component: str,
    security_context: SecurityContext | None = None,
    additional_context: dict[str, Any] | None = None,
) -> None:
    """Convenience function for recording errors for analysis."""
    analytics = get_pattern_analytics()
    analytics.record_error_event(error, component, security_context, additional_context)
