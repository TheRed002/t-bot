"""
Secure error reporting system with role-based access control.

This module provides a comprehensive secure error reporting system that ensures
appropriate information disclosure based on user roles and security clearance levels.

CRITICAL: Prevents unauthorized access to sensitive error information while
maintaining system observability for authorized personnel.
"""

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from src.core.logging import get_logger
from src.error_handling.secure_context_manager import (
    InformationLevel,
    SecureErrorReport,
    SecurityContext,
    UserRole,
    get_secure_context_manager,
)
from src.error_handling.security_rate_limiter import (
    get_security_rate_limiter,
)
from src.error_handling.security_sanitizer import (
    get_security_sanitizer,
)


class ReportingChannel(Enum):
    """Channels for error reporting."""

    LOG = "log"  # Standard logging
    DATABASE = "database"  # Database storage
    ALERT = "alert"  # Alert system
    EMAIL = "email"  # Email notifications
    WEBHOOK = "webhook"  # Webhook notifications
    METRICS = "metrics"  # Metrics system
    AUDIT = "audit"  # Audit trail


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ReportingRule:
    """Rule for routing error reports to appropriate channels."""

    name: str
    condition: str  # Expression to evaluate
    channels: list[ReportingChannel]
    min_user_role: UserRole
    alert_severity: AlertSeverity
    rate_limit: int | None = None  # Max reports per hour
    enabled: bool = True


@dataclass
class ErrorAlert:
    """Error alert for notification systems."""

    alert_id: str
    error_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    component: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportingMetrics:
    """Metrics for error reporting system."""

    total_reports: int = 0
    reports_by_role: dict[str, int] = field(default_factory=dict)
    reports_by_channel: dict[str, int] = field(default_factory=dict)
    blocked_reports: int = 0
    alert_counts: dict[str, int] = field(default_factory=dict)
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SecureErrorReporter:
    """
    Secure error reporting system with comprehensive access control.

    Features:
    - Role-based information disclosure
    - Multiple reporting channels
    - Rate limiting and abuse prevention
    - Alert generation and routing
    - Audit trail maintenance
    - Metrics collection
    - Secure storage and retrieval
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.context_manager = get_secure_context_manager()
        self.rate_limiter = get_security_rate_limiter()
        self.sanitizer = get_security_sanitizer()

        # Error storage (in-memory for demo, should use persistent storage)
        self._error_reports: dict[str, SecureErrorReport] = {}
        self._error_alerts: dict[str, ErrorAlert] = {}

        # Access control
        self._user_sessions: dict[str, SecurityContext] = {}
        self._access_logs: list[dict[str, Any]] = []

        # Reporting configuration
        self._reporting_rules: list[ReportingRule] = []
        self._metrics = ReportingMetrics()

        # Initialize default reporting rules
        self._init_default_rules()

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._start_background_tasks()

    def _init_default_rules(self) -> None:
        """Initialize default reporting rules."""
        self._reporting_rules = [
            # Critical errors - immediate alerts for security team
            ReportingRule(
                name="critical_security_alerts",
                condition="error_category in ['authentication', 'authorization'] and alert_severity == 'critical'",
                channels=[ReportingChannel.ALERT, ReportingChannel.EMAIL, ReportingChannel.AUDIT],
                min_user_role=UserRole.SECURITY,
                alert_severity=AlertSeverity.CRITICAL,
                rate_limit=10,  # Max 10 per hour
            ),
            # Trading system errors - alerts for admins
            ReportingRule(
                name="trading_system_errors",
                condition="component in ['exchange', 'trading', 'risk'] and alert_severity in ['error', 'critical']",
                channels=[
                    ReportingChannel.ALERT,
                    ReportingChannel.DATABASE,
                    ReportingChannel.METRICS,
                ],
                min_user_role=UserRole.ADMIN,
                alert_severity=AlertSeverity.ERROR,
                rate_limit=20,
            ),
            # Database errors - alerts for developers and admins
            ReportingRule(
                name="database_errors",
                condition="error_category == 'database' and alert_severity in ['error', 'critical']",
                channels=[ReportingChannel.LOG, ReportingChannel.DATABASE, ReportingChannel.ALERT],
                min_user_role=UserRole.DEVELOPER,
                alert_severity=AlertSeverity.WARNING,
                rate_limit=30,
            ),
            # Network errors - standard logging
            ReportingRule(
                name="network_errors",
                condition="error_category == 'network'",
                channels=[ReportingChannel.LOG, ReportingChannel.METRICS],
                min_user_role=UserRole.USER,
                alert_severity=AlertSeverity.INFO,
                rate_limit=100,
            ),
            # General errors - standard reporting
            ReportingRule(
                name="general_errors",
                condition="True",  # Catch-all
                channels=[ReportingChannel.LOG, ReportingChannel.DATABASE],
                min_user_role=UserRole.USER,
                alert_severity=AlertSeverity.INFO,
            ),
        ]

    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._background_maintenance())
            # Add done callback to handle any exceptions
            self._cleanup_task.add_done_callback(self._background_task_done_callback)

    def _background_task_done_callback(self, task: asyncio.Task) -> None:
        """Handle background task completion."""
        if task.exception():
            self.logger.error(f"Background maintenance task failed: {task.exception()}")
        # Reset task reference so it can be restarted if needed
        if self._cleanup_task is task:
            self._cleanup_task = None

    async def _background_maintenance(self) -> None:
        """Background maintenance tasks."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Clean up old reports
                await self._cleanup_old_reports()

                # Clean up old access logs
                self._cleanup_access_logs()

                # Update metrics
                self._update_metrics()

            except Exception as e:
                self.logger.error(f"Error in background maintenance: {e}")
                await asyncio.sleep(300)

    async def report_error(
        self,
        error: Exception,
        security_context: SecurityContext,
        original_context: dict[str, Any] | None = None,
        channels: list[ReportingChannel] | None = None,
    ) -> SecureErrorReport:
        """
        Report an error through the secure reporting system.

        Args:
            error: Exception to report
            security_context: Security context with user information
            original_context: Original error context
            channels: Specific channels to report to (optional)

        Returns:
            Secure error report
        """
        # Check rate limits
        rate_check = await self.rate_limiter.check_rate_limit(
            component=security_context.component or "error_reporter",
            operation="error_report",
            client_ip=security_context.client_ip,
        )

        if not rate_check.allowed:
            self.logger.warning(
                "Error reporting rate limited",
                user_id=security_context.user_id,
                component=security_context.component,
                reason=rate_check.reason,
            )
            self._metrics.blocked_reports += 1
            raise Exception("Error reporting rate limited")

        # Create secure error report
        secure_report = self.context_manager.create_secure_report(
            error, security_context, original_context
        )

        # Store the report
        self._error_reports[secure_report.error_id] = secure_report

        # Log access
        self._log_access(security_context, "report_error", secure_report.error_id)

        # Determine reporting channels
        if channels is None:
            channels = self._determine_channels(secure_report, security_context)

        # Route to appropriate channels
        await self._route_to_channels(secure_report, security_context, channels)

        # Update metrics
        self._metrics.total_reports += 1
        role_key = security_context.user_role.value
        self._metrics.reports_by_role[role_key] = self._metrics.reports_by_role.get(role_key, 0) + 1

        self.logger.info(
            "Error reported successfully",
            error_id=secure_report.error_id,
            user_role=security_context.user_role.value,
            channels=[c.value for c in channels],
            information_level=secure_report.information_level.value,
        )

        return secure_report

    async def get_error_report(
        self, error_id: str, security_context: SecurityContext
    ) -> SecureErrorReport | None:
        """
        Retrieve an error report with access control.

        Args:
            error_id: Error report ID
            security_context: Security context for access control

        Returns:
            Secure error report if authorized, None otherwise
        """
        # Check if report exists
        if error_id not in self._error_reports:
            return None

        original_report = self._error_reports[error_id]

        # Check access permissions
        if not self._check_report_access(original_report, security_context):
            self._log_access(security_context, "get_error_report_denied", error_id)
            self.logger.warning(
                "Unauthorized access to error report",
                error_id=error_id,
                user_id=security_context.user_id,
                user_role=security_context.user_role.value,
            )
            return None

        # Create filtered report based on user role
        filtered_report = self._filter_report_for_user(original_report, security_context)

        # Log access
        self._log_access(security_context, "get_error_report", error_id)

        return filtered_report

    async def search_error_reports(
        self,
        security_context: SecurityContext,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SecureErrorReport]:
        """
        Search error reports with access control and filtering.

        Args:
            security_context: Security context for access control
            filters: Search filters
            limit: Maximum number of results
            offset: Search offset

        Returns:
            List of filtered error reports
        """
        # Rate limit search operations
        rate_check = await self.rate_limiter.check_rate_limit(
            component="error_search", operation="search", client_ip=security_context.client_ip
        )

        if not rate_check.allowed:
            self.logger.warning("Error search rate limited", user_id=security_context.user_id)
            return []

        # Get accessible reports
        accessible_reports = []
        for report in self._error_reports.values():
            if self._check_report_access(report, security_context):
                # Apply filters
                if self._matches_filters(report, filters):
                    filtered_report = self._filter_report_for_user(report, security_context)
                    accessible_reports.append(filtered_report)

        # Sort by timestamp (newest first)
        accessible_reports.sort(key=lambda r: r.timestamp, reverse=True)

        # Apply pagination
        paginated_reports = accessible_reports[offset : offset + limit]

        # Log access
        self._log_access(
            security_context, "search_error_reports", f"found_{len(paginated_reports)}"
        )

        return paginated_reports

    async def get_error_statistics(
        self, security_context: SecurityContext, time_range: timedelta | None = None
    ) -> dict[str, Any]:
        """
        Get error statistics with role-based filtering.

        Args:
            security_context: Security context for access control
            time_range: Time range for statistics (default: last 24 hours)

        Returns:
            Error statistics dictionary
        """
        if time_range is None:
            time_range = timedelta(hours=24)

        cutoff_time = datetime.now(timezone.utc) - time_range

        # Filter reports by time and access
        accessible_reports = [
            report
            for report in self._error_reports.values()
            if (
                report.timestamp > cutoff_time
                and self._check_report_access(report, security_context)
            )
        ]

        # Generate statistics based on user role
        stats = {
            "total_errors": len(accessible_reports),
            "time_range_hours": time_range.total_seconds() / 3600,
        }

        if security_context.user_role in [UserRole.ADMIN, UserRole.DEVELOPER, UserRole.SECURITY]:
            # Detailed statistics for authorized users
            stats.update(
                {
                    "errors_by_component": self._get_component_stats(accessible_reports),
                    "errors_by_severity": self._get_severity_stats(accessible_reports),
                    "errors_over_time": self._get_time_series_stats(accessible_reports, time_range),
                    "top_error_types": self._get_error_type_stats(accessible_reports),
                }
            )

        if security_context.user_role in [UserRole.SECURITY, UserRole.SYSTEM]:
            # Security-specific statistics
            stats.update(
                {
                    "security_events": self._get_security_stats(accessible_reports),
                    "access_patterns": self._get_access_pattern_stats(),
                    "reporting_metrics": asdict(self._metrics),
                }
            )

        # Log access
        self._log_access(security_context, "get_error_statistics", str(len(accessible_reports)))

        return stats

    def _determine_channels(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> list[ReportingChannel]:
        """Determine appropriate reporting channels based on rules."""
        channels = set()

        # Evaluate reporting rules
        for rule in self._reporting_rules:
            if not rule.enabled:
                continue

            # Check if user role meets minimum requirement
            if not self._role_meets_minimum(security_context.user_role, rule.min_user_role):
                continue

            # Evaluate rule condition
            if self._evaluate_rule_condition(rule, report, security_context):
                channels.update(rule.channels)

                # Check rate limits for this rule
                if rule.rate_limit and not self._check_rule_rate_limit(rule, report):
                    self.logger.warning(f"Rule {rule.name} rate limited", error_id=report.error_id)
                    continue

        return list(channels)

    async def _route_to_channels(
        self,
        report: SecureErrorReport,
        security_context: SecurityContext,
        channels: list[ReportingChannel],
    ) -> None:
        """Route error report to specified channels."""
        for channel in channels:
            try:
                await self._send_to_channel(report, security_context, channel)

                # Update metrics
                channel_key = channel.value
                self._metrics.reports_by_channel[channel_key] = (
                    self._metrics.reports_by_channel.get(channel_key, 0) + 1
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to send report to channel {channel.value}",
                    error_id=report.error_id,
                    error=str(e),
                )

    async def _send_to_channel(
        self,
        report: SecureErrorReport,
        security_context: SecurityContext,
        channel: ReportingChannel,
    ) -> None:
        """Send report to a specific channel."""
        if channel == ReportingChannel.LOG:
            self._send_to_log(report, security_context)

        elif channel == ReportingChannel.DATABASE:
            await self._send_to_database(report, security_context)

        elif channel == ReportingChannel.ALERT:
            await self._send_to_alert_system(report, security_context)

        elif channel == ReportingChannel.EMAIL:
            await self._send_email_notification(report, security_context)

        elif channel == ReportingChannel.WEBHOOK:
            await self._send_webhook_notification(report, security_context)

        elif channel == ReportingChannel.METRICS:
            await self._send_to_metrics_system(report, security_context)

        elif channel == ReportingChannel.AUDIT:
            await self._send_to_audit_trail(report, security_context)

    def _send_to_log(self, report: SecureErrorReport, security_context: SecurityContext) -> None:
        """Send report to logging system."""
        log_level = "ERROR"
        if "critical" in report.technical_message.lower():
            log_level = "CRITICAL"
        elif "warning" in report.technical_message.lower():
            log_level = "WARNING"

        self.logger.log(
            getattr(self.logger, log_level.lower(), self.logger.error),
            report.technical_message,
            error_id=report.error_id,
            component=report.component,
            operation=report.operation,
            user_id=security_context.user_id,
            information_level=report.information_level.value,
        )

    async def _send_to_database(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> None:
        """Send report to database storage."""
        # TODO: Implement database storage
        # This would store the report in a secure database with proper indexing
        self.logger.info(f"Storing error report {report.error_id} in database")

    async def _send_to_alert_system(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> None:
        """Send report to alert system."""
        alert_severity = self._determine_alert_severity(report)

        alert = ErrorAlert(
            alert_id=f"ALERT_{report.error_id}",
            error_id=report.error_id,
            severity=alert_severity,
            title=f"Error in {report.component or 'System'}",
            message=report.technical_message,
            timestamp=report.timestamp,
            component=report.component,
            user_id=security_context.user_id,
            metadata={
                "information_level": report.information_level.value,
                "user_role": security_context.user_role.value,
            },
        )

        # Store alert
        self._error_alerts[alert.alert_id] = alert

        # Update metrics
        severity_key = alert_severity.value
        self._metrics.alert_counts[severity_key] = (
            self._metrics.alert_counts.get(severity_key, 0) + 1
        )

        self.logger.warning(
            f"Alert generated: {alert.title}",
            alert_id=alert.alert_id,
            severity=alert_severity.value,
        )

    async def _send_email_notification(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> None:
        """Send email notification."""
        # TODO: Implement email notification
        self.logger.info(f"Sending email notification for error {report.error_id}")

    async def _send_webhook_notification(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> None:
        """Send webhook notification."""
        # TODO: Implement webhook notification
        self.logger.info(f"Sending webhook notification for error {report.error_id}")

    async def _send_to_metrics_system(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> None:
        """Send metrics to monitoring system."""
        # TODO: Implement metrics system integration
        self.logger.info(f"Recording metrics for error {report.error_id}")

    async def _send_to_audit_trail(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> None:
        """Send to audit trail."""
        # TODO: Store in secure audit system
        # audit_entry = {
        #     "timestamp": report.timestamp.isoformat(),
        #     "error_id": report.error_id,
        #     "component": report.component,
        #     "operation": report.operation,
        #     "user_id": security_context.user_id,
        #     "user_role": security_context.user_role.value,
        #     "client_ip": security_context.client_ip,
        #     "information_level": report.information_level.value,
        #     "action": "error_reported",
        # }
        self.logger.info(f"Audit trail entry created for error {report.error_id}")

    def _check_report_access(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> bool:
        """Check if user has access to view report."""
        # Admin and security roles can see all reports
        if security_context.user_role in [UserRole.ADMIN, UserRole.SECURITY, UserRole.SYSTEM]:
            return True

        # Users can only see their own reports (if user_id matches)
        if security_context.user_role == UserRole.USER:
            return (
                security_context.user_id
                and report.context.get("user_id") == security_context.user_id
            )

        # Developers can see reports from their components
        if security_context.user_role == UserRole.DEVELOPER:
            return True  # For now, allow developers to see all reports

        # Guests have no access
        return False

    def _filter_report_for_user(
        self, report: SecureErrorReport, security_context: SecurityContext
    ) -> SecureErrorReport:
        """Filter error report based on user's access level."""
        user_info_level = self.context_manager.role_info_levels.get(
            security_context.user_role, InformationLevel.MINIMAL
        )

        # Create filtered copy
        filtered_report = SecureErrorReport(
            error_id=report.error_id,
            timestamp=report.timestamp,
            user_message=report.user_message,
            technical_message=report.technical_message,
            error_code=report.error_code,
            component=report.component,
            operation=report.operation,
            context=report.context.copy(),
            debug_info=None,  # Will be set based on access level
            information_level=user_info_level,
            sanitized=True,
        )

        # Filter technical message based on user level
        if user_info_level == InformationLevel.MINIMAL:
            filtered_report.technical_message = report.user_message
            filtered_report.context = {"timestamp": report.timestamp.isoformat()}

        elif user_info_level == InformationLevel.BASIC:
            # Keep basic info but sanitize
            filtered_report.technical_message = self.sanitizer.sanitize_error_message(
                report.technical_message
            )

        elif user_info_level in [InformationLevel.FULL, InformationLevel.DEBUG]:
            # Full access - include debug info if available
            filtered_report.debug_info = report.debug_info

        return filtered_report

    def _matches_filters(self, report: SecureErrorReport, filters: dict[str, Any] | None) -> bool:
        """Check if report matches search filters."""
        if not filters:
            return True

        for key, value in filters.items():
            if key == "component" and report.component != value:
                return False
            elif key == "error_code" and report.error_code != value:
                return False
            elif key == "from_date":
                from_date = datetime.fromisoformat(value) if isinstance(value, str) else value
                if report.timestamp < from_date:
                    return False
            elif key == "to_date":
                to_date = datetime.fromisoformat(value) if isinstance(value, str) else value
                if report.timestamp > to_date:
                    return False

        return True

    def _evaluate_rule_condition(
        self, rule: ReportingRule, report: SecureErrorReport, security_context: SecurityContext
    ) -> bool:
        """Evaluate rule condition against report."""
        try:
            # Create evaluation context
            eval_context = {
                "error_category": self._categorize_error_from_report(report),
                "component": report.component,
                "error_code": report.error_code,
                "information_level": report.information_level.value,
                "user_role": security_context.user_role.value,
                "alert_severity": self._determine_alert_severity(report).value,
                "timestamp": report.timestamp,
            }

            # Safely evaluate condition
            return eval(rule.condition, {"__builtins__": {}}, eval_context)
        except Exception as e:
            self.logger.error(f"Error evaluating rule condition: {e}")
            return False

    def _role_meets_minimum(self, user_role: UserRole, min_role: UserRole) -> bool:
        """Check if user role meets minimum requirement."""
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.ADMIN: 2,
            UserRole.DEVELOPER: 3,
            UserRole.SECURITY: 4,
            UserRole.SYSTEM: 5,
        }

        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(min_role, 0)

    def _determine_alert_severity(self, report: SecureErrorReport) -> AlertSeverity:
        """Determine alert severity from report."""
        message_lower = report.technical_message.lower()

        if any(keyword in message_lower for keyword in ["critical", "fatal", "security"]):
            return AlertSeverity.CRITICAL
        elif any(keyword in message_lower for keyword in ["error", "failed", "exception"]):
            return AlertSeverity.ERROR
        elif any(keyword in message_lower for keyword in ["warning", "warn"]):
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _categorize_error_from_report(self, report: SecureErrorReport) -> str:
        """Categorize error from report information."""
        message_lower = report.technical_message.lower()

        if any(keyword in message_lower for keyword in ["auth", "permission", "forbidden"]):
            return "authentication"
        elif any(keyword in message_lower for keyword in ["validation", "invalid"]):
            return "validation"
        elif any(keyword in message_lower for keyword in ["network", "connection", "timeout"]):
            return "network"
        elif any(keyword in message_lower for keyword in ["database", "sql"]):
            return "database"
        elif any(keyword in message_lower for keyword in ["exchange", "trading"]):
            return "exchange"
        else:
            return "unknown"

    def _log_access(self, security_context: SecurityContext, action: str, resource: str) -> None:
        """Log access attempt for audit trail."""
        access_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": security_context.user_id,
            "user_role": security_context.user_role.value,
            "client_ip": security_context.client_ip,
            "action": action,
            "resource": resource,
            "session_id": security_context.session_id,
        }

        self._access_logs.append(access_log)

    def _cleanup_access_logs(self) -> None:
        """Clean up old access logs."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        self._access_logs = [
            log for log in self._access_logs if datetime.fromisoformat(log["timestamp"]) > cutoff
        ]

    async def _cleanup_old_reports(self) -> None:
        """Clean up old error reports."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)  # Keep reports for 90 days

        expired_reports = [
            error_id
            for error_id, report in self._error_reports.items()
            if report.timestamp < cutoff
        ]

        for error_id in expired_reports:
            del self._error_reports[error_id]

        if expired_reports:
            self.logger.info(f"Cleaned up {len(expired_reports)} old error reports")

    def _update_metrics(self) -> None:
        """Update reporting metrics."""
        # Reset metrics daily
        now = datetime.now(timezone.utc)
        if (now - self._metrics.last_reset).days >= 1:
            self._metrics = ReportingMetrics()

    def _get_component_stats(self, reports: list[SecureErrorReport]) -> dict[str, int]:
        """Get error statistics by component."""
        stats = {}
        for report in reports:
            component = report.component or "unknown"
            stats[component] = stats.get(component, 0) + 1
        return stats

    def _get_severity_stats(self, reports: list[SecureErrorReport]) -> dict[str, int]:
        """Get error statistics by severity."""
        stats = {}
        for report in reports:
            severity = self._determine_alert_severity(report).value
            stats[severity] = stats.get(severity, 0) + 1
        return stats

    def _get_time_series_stats(
        self, reports: list[SecureErrorReport], time_range: timedelta
    ) -> list[dict[str, Any]]:
        """Get time series error statistics."""
        # Group reports by hour
        hourly_stats = {}
        for report in reports:
            hour = report.timestamp.replace(minute=0, second=0, microsecond=0)
            hour_key = hour.isoformat()
            hourly_stats[hour_key] = hourly_stats.get(hour_key, 0) + 1

        # Convert to list format
        return [{"timestamp": k, "count": v} for k, v in sorted(hourly_stats.items())]

    def _get_error_type_stats(self, reports: list[SecureErrorReport]) -> dict[str, int]:
        """Get statistics by error type."""
        stats = {}
        for report in reports:
            # Extract error type from technical message
            error_type = (
                report.technical_message.split(":")[0]
                if ":" in report.technical_message
                else "Unknown"
            )
            stats[error_type] = stats.get(error_type, 0) + 1
        return stats

    def _get_security_stats(self, reports: list[SecureErrorReport]) -> dict[str, Any]:
        """Get security-specific statistics."""
        return {
            "authentication_errors": len(
                [r for r in reports if "auth" in r.technical_message.lower()]
            ),
            "authorization_errors": len(
                [r for r in reports if "permission" in r.technical_message.lower()]
            ),
            "security_alerts": len(
                [r for r in reports if self._determine_alert_severity(r) == AlertSeverity.CRITICAL]
            ),
        }

    def _get_access_pattern_stats(self) -> dict[str, Any]:
        """Get access pattern statistics."""
        return {
            "total_access_attempts": len(self._access_logs),
            "unique_users": len(
                set(log.get("user_id") for log in self._access_logs if log.get("user_id"))
            ),
            "access_by_role": self._group_access_by_role(),
        }

    def _group_access_by_role(self) -> dict[str, int]:
        """Group access logs by user role."""
        stats = {}
        for log in self._access_logs:
            role = log.get("user_role", "unknown")
            stats[role] = stats.get(role, 0) + 1
        return stats

    def _check_rule_rate_limit(self, rule: ReportingRule, report: SecureErrorReport) -> bool:
        """Check rate limit for specific reporting rule."""
        # TODO: Implement rule-specific rate limiting
        return True

    def get_system_health(self) -> dict[str, Any]:
        """Get system health information."""
        return {
            "error_reports_count": len(self._error_reports),
            "error_alerts_count": len(self._error_alerts),
            "access_logs_count": len(self._access_logs),
            "reporting_rules_count": len(self._reporting_rules),
            "metrics": asdict(self._metrics),
            "background_tasks_running": not (self._cleanup_task and self._cleanup_task.done()),
        }


# Global secure error reporter instance
_global_reporter = None


def get_secure_error_reporter() -> SecureErrorReporter:
    """Get global secure error reporter instance."""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = SecureErrorReporter()
    return _global_reporter


async def secure_report_error(
    error: Exception,
    security_context: SecurityContext,
    original_context: dict[str, Any] | None = None,
    channels: list[ReportingChannel] | None = None,
) -> SecureErrorReport:
    """Convenience function for secure error reporting."""
    reporter = get_secure_error_reporter()
    return await reporter.report_error(error, security_context, original_context, channels)
