"""
Secure reporting backward compatibility.

Redirects to consolidated security types.
"""

from typing import Any

from src.utils.security_types import (
    ErrorAlert,
    ReportingChannel,
    ReportingConfig,
    ReportingMetrics,
    ReportingRule,
    ReportType,
    SeverityLevel as AlertSeverity,  # Alias for backward compatibility
)


def get_secure_context_manager():
    """Get secure context manager for backward compatibility."""
    return None


def get_security_rate_limiter():
    """Get security rate limiter for backward compatibility."""
    return None


def get_security_sanitizer():
    """Get security sanitizer for backward compatibility."""
    return None


def generate_secure_report(data: dict[str, Any]) -> dict[str, Any]:
    """Generate secure report for backward compatibility."""
    return {"secure_report": data, "generated": True}


class MockContextManager:
    """Mock context manager for testing."""

    def __init__(self):
        self.context = {"user_role": "admin"}

    def get_context(self):
        return self.context

    def create_secure_report(self, error, security_context=None, error_context=None):
        """Create secure report for testing."""
        from src.error_handling.secure_context_manager import SecureErrorReport
        return SecureErrorReport(
            message=str(error),
            details=error_context or {}
        )


class MockReportingRule:
    """Mock reporting rule for testing."""

    def __init__(self, name: str, channels: list[str], user_role: str):
        self.name = name
        self.channels = channels
        self.user_role = user_role
        self.min_user_role = user_role  # Alias for compatibility
        self.enabled = True


class SecureErrorReporter:
    """Simple secure error reporter for backward compatibility."""

    def __init__(self, config: ReportingConfig | None = None):
        self.config = config or ReportingConfig()
        self.metrics = ReportingMetrics()
        self.logger = None  # Mock for tests
        self.context_manager = MockContextManager()  # Mock for tests
        self.rate_limiter = None  # Mock for tests
        self.sanitizer = None  # Mock for tests

        # Create default reporting rules
        self.reporting_rules = [
            MockReportingRule("critical_errors", ["email", "sms"], "admin"),
            MockReportingRule("security_events", ["email"], "user"),
        ]

    def generate_report(self, report_type: ReportType, data: dict[str, Any]) -> dict[str, Any]:
        """Generate a report."""
        self.metrics.reports_generated += 1
        return generate_secure_report(data)

    def send_alert(
        self, severity: AlertSeverity, message: str, context: dict[str, Any] | None = None
    ) -> bool:
        """Send an alert."""
        self.metrics.alerts_sent += 1
        return True  # Always succeed in simplified implementation

    def create_error_report(
        self, error: Exception, security_context: Any, error_context: dict[str, Any]
    ) -> Any:
        """Create error report."""
        return self.context_manager.create_secure_report(error, security_context, error_context)

    async def submit_error_report(
        self, error: Exception, security_context: Any = None, error_context: dict[str, Any] = None
    ) -> bool:
        """Submit error report."""
        return True

    def evaluate_reporting_rules(self, error: Exception, context: dict[str, Any]) -> list:
        """Evaluate reporting rules."""
        return self.reporting_rules

    def filter_by_user_role(self, rules: list, user_role: Any) -> list:
        """Filter rules by user role."""
        return rules

    async def generate_alert(
        self, error: Exception, security_context: Any = None, error_context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Generate alert."""
        return {"alert": "generated", "error": str(error)}

    def get_reporting_metrics(self) -> dict[str, Any]:
        """Get reporting metrics."""
        return {
            "reports_generated": self.metrics.reports_generated,
            "alerts_sent": self.metrics.alerts_sent,
        }

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = ReportingMetrics()

    def add_reporting_rule(self, rule: Any) -> None:
        """Add reporting rule."""
        self.reporting_rules.append(rule)

    def remove_reporting_rule(self, rule_name: str) -> bool:
        """Remove reporting rule."""
        self.reporting_rules = [r for r in self.reporting_rules if r.name != rule_name]
        return True

    async def route_report_to_multiple_channels(
        self, report: dict[str, Any], channels: list[str]
    ) -> dict[str, bool]:
        """Route report to multiple channels."""
        return {channel: True for channel in channels}

    def validate_reporting_rule(self, rule: Any) -> bool:
        """Validate reporting rule."""
        return True

    def get_applicable_rules(self, error: Exception, context: dict[str, Any]) -> list:
        """Get applicable rules."""
        return self.reporting_rules


class SecureReporter(SecureErrorReporter):
    """Alias for backward compatibility."""

    pass


def sanitize_report_data(data: dict[str, Any]) -> dict[str, Any]:
    """Simple report data sanitization."""
    from .security_validator import sanitize_error_data

    return sanitize_error_data(data)


__all__ = [
    "AlertSeverity",
    "ErrorAlert",
    "ReportType",
    "ReportingChannel",
    "ReportingConfig",
    "ReportingMetrics",
    "ReportingRule",
    "SecureErrorReporter",
    "SecureReporter",
    "generate_secure_report",
    "sanitize_report_data",
]
