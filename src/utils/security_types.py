"""
Common security types and enums to eliminate duplication.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# Consolidated enums from various security modules
class SeverityLevel(Enum):
    """Generic severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LogCategory(Enum):
    """Log categories."""

    GENERAL = "general"
    SECURITY = "security"
    TRADING = "trading"
    SYSTEM = "system"


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    AUDIT = "audit"


class InformationLevel(Enum):
    """Information levels for security context."""

    MINIMAL = "minimal"
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"
    SENSITIVE = "sensitive"
    INTERNAL = "internal"
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class ThreatType(Enum):
    """Security threat types."""

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


class ReportType(Enum):
    """Report types."""

    ERROR_SUMMARY = "error_summary"
    SECURITY_INCIDENT = "security_incident"
    PATTERN_ANALYSIS = "pattern_analysis"
    SYSTEM_HEALTH = "system_health"


class ReportingChannel(Enum):
    """Reporting channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DATABASE = "database"


class UserRole(Enum):
    """User roles for access control."""

    GUEST = "guest"
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SYSTEM = "system"


# InformationLevel already defined above


# Common dataclasses
@dataclass
class SecurityContext:
    """Common security context."""

    user_role: UserRole = UserRole.GUEST
    information_level: InformationLevel = InformationLevel.INTERNAL
    user_id: str | None = None
    client_ip: str | None = None
    is_authenticated: bool = False
    component: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_admin_access(self) -> bool:
        """Check if user has admin access."""
        return self.user_role in (UserRole.ADMIN, UserRole.SYSTEM)


@dataclass
class ErrorAlert:
    """Generic error alert."""

    alert_id: str
    severity: SeverityLevel
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecureLogEntry:
    """Secure log entry."""

    timestamp: datetime
    level: LogLevel
    message: str
    category: LogCategory = LogCategory.GENERAL
    sanitized: bool = True
    context: dict[str, Any] = field(default_factory=dict)
    component: str = "unknown"
    information_level: InformationLevel = InformationLevel.BASIC
    threat_detected: bool = False
    security_classification: str = "INTERNAL"
    user_id: str | None = None
    client_ip: str | None = None
    error_type: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    sanitize: bool = True
    category: LogCategory = LogCategory.GENERAL
    max_message_length: int = 1000


@dataclass
class ReportingConfig:
    """Reporting configuration."""

    enabled: bool = True
    include_stack_traces: bool = True
    sanitize_data: bool = True
    max_report_size: int = 10000


@dataclass
class ReportingRule:
    """Reporting rule configuration."""

    name: str
    condition: str
    channel: ReportingChannel
    enabled: bool = True


@dataclass
class ReportingMetrics:
    """Reporting metrics."""

    alerts_sent: int = 0
    reports_generated: int = 0
    errors_processed: int = 0


__all__ = [
    # Enums
    "SeverityLevel",
    "LogCategory",
    "LogLevel",
    "ThreatType",
    "ReportType",
    "ReportingChannel",
    "UserRole",
    "InformationLevel",
    # Dataclasses
    "SecurityContext",
    "ErrorAlert",
    "SecureLogEntry",
    "LoggingConfig",
    "ReportingConfig",
    "ReportingRule",
    "ReportingMetrics",
]
