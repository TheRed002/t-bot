"""
Comprehensive alerting system for T-Bot Trading System.

This module implements P-031: Alerting System with:
- Alert rules configuration and management
- Multiple notification channels (email, Slack, webhook)
- Alert severity levels and escalation policies
- Alert suppression and deduplication
- Integration with Prometheus AlertManager

Key Features:
- Real-time alert processing
- Smart alert grouping and deduplication
- Multi-channel notification routing
- Escalation policies based on severity
- Alert suppression during maintenance
- Alert history and analytics
"""

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

# Use service injection for error handling - no direct global dependencies
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from src.core.base import BaseComponent
from src.core.types import AlertSeverity
from src.core.event_constants import AlertEvents
from src.monitoring.config import (
    ALERT_BACKGROUND_TASK_TIMEOUT,
    ALERT_ESCALATION_CHECK_TIMEOUT,
    ALERT_HISTORY_MAX_SIZE,
    ALERT_NOTIFICATION_TIMEOUT,
    ALERT_PROCESSING_CHECK_INTERVAL,
    ALERT_RETRY_DELAY,
    ALERT_RETRY_MAX_ATTEMPTS,
    DEFAULT_ESCALATION_MAX,
    DEFAULT_WEBHOOK_TIMEOUT,
    DISCORD_SEVERITY_COLORS,
    DURATION_PARSE_MINIMUM_MINUTES,
    EMAIL_RETRY_BASE_DELAY,
    EMAIL_RETRY_MAX_ATTEMPTS,
    HTTP_CONNECTOR_LIMIT,
    HTTP_CONNECTOR_LIMIT_PER_HOST,
    HTTP_OK,
    HTTP_SERVER_ERROR_THRESHOLD,
    HTTP_SESSION_TIMEOUT,
    SECONDS_PER_MINUTE,
    SESSION_CLEANUP_TIMEOUT,
    SESSION_CLOSE_TIMEOUT,
    SEVERITY_COLORS,
    WEBHOOK_RETRY_BASE_DELAY,
    WEBHOOK_RETRY_MAX_ATTEMPTS,
)

# Import utils decorators and error handling
from src.utils.decorators import logged, monitored, retry

if TYPE_CHECKING:
    from src.error_handling.context import ErrorContext
else:
    # Use runtime imports to avoid circular dependencies
    try:
        from src.error_handling.context import ErrorContext
    except ImportError:
        class ErrorContext:
            def __init__(self, component: str, operation: str, details: Union[dict, None] = None, error: Union[Exception, None] = None):
                self.component = component
                self.operation = operation
                self.details = details
                self.error = error


from src.utils.monitoring_helpers import (
    create_error_context,
    handle_error_with_fallback,
)

# Try to import email modules, handle gracefully if missing
try:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

    # Mock email classes
    class MockMimeText:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class MockMimeMultipart:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._headers: Dict[str, str] = {}

        def __setitem__(self, key: str, value: str) -> None:
            self._headers[key] = value

        def attach(self, *args: Any) -> None:
            pass

    class MockSMTP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def starttls(self) -> None:
            pass

        def login(self, *args: Any) -> None:
            pass

        def send_message(self, *args: Any) -> None:
            pass

        def quit(self) -> None:
            pass

    _mock_smtplib = type("MockSMTPLib", (), {"SMTP": MockSMTP})()
    smtplib = _mock_smtplib  # type: ignore[assignment]
    MIMEText = MockMimeText  # type: ignore[assignment,misc]
    MIMEMultipart = MockMimeMultipart  # type: ignore[assignment,misc]

# Try to import yaml
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]


import aiohttp

from src.core.exceptions import MonitoringError, ServiceError


class AlertStatus(Enum):
    """Alert status states."""

    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


class NotificationChannel(Enum):
    """Notification channel types."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DISCORD = "discord"


@dataclass
class AlertRule:
    """Definition of an alert rule."""

    name: str
    description: str
    severity: AlertSeverity
    query: str  # PromQL query
    threshold: float
    operator: str  # >, <, >=, <=, ==, !=
    duration: str  # Duration string like "5m", "1h"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_delay: Union[str, None] = None  # Escalation delay like "15m"
    enabled: bool = True

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.notification_channels:
            # Default channels based on severity
            if self.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                self.notification_channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK]
            else:
                self.notification_channels = [NotificationChannel.SLACK]


@dataclass
class Alert:
    """Alert instance with compatibility for database model."""

    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: datetime
    ends_at: Union[datetime, None] = None
    fingerprint: str = ""
    notification_sent: bool = False
    acknowledgment_by: Union[str, None] = None
    acknowledgment_at: Union[datetime, None] = None
    escalated: bool = False
    escalation_count: int = 0

    def __post_init__(self):
        """Generate fingerprint for deduplication."""
        if not self.fingerprint:
            # Create fingerprint from rule name and labels
            fingerprint_data = f"{self.rule_name}:{sorted(self.labels.items())}"
            self.fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()

    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == AlertStatus.FIRING

    def to_db_model_dict(self) -> Dict[str, Any]:
        """Convert to database model format."""
        return {
            "alert_type": self.rule_name,
            "severity": self.severity.value.upper(),
            "title": self.rule_name,
            "message": self.message,
            "status": self.status.value.upper() if self.status != AlertStatus.FIRING else "ACTIVE",
            "acknowledged_by": self.acknowledgment_by,
            "acknowledged_at": self.acknowledgment_at,
            "resolved_at": self.ends_at,
            "context": {
                "labels": self.labels,
                "annotations": self.annotations,
                "fingerprint": self.fingerprint,
                "escalated": self.escalated,
                "escalation_count": self.escalation_count,
            },
        }

    @classmethod
    def from_db_model(cls, db_alert: Dict[str, Any]) -> "Alert":
        """Create Alert from database model."""
        context = db_alert.get("context", {})
        status_map = {
            "ACTIVE": AlertStatus.FIRING,
            "ACKNOWLEDGED": AlertStatus.ACKNOWLEDGED,
            "RESOLVED": AlertStatus.RESOLVED,
            "SUPPRESSED": AlertStatus.SUPPRESSED,
        }

        return cls(
            rule_name=db_alert["alert_type"],
            severity=AlertSeverity(db_alert["severity"].lower()),
            status=status_map.get(db_alert["status"], AlertStatus.FIRING),
            message=db_alert["message"],
            labels=context.get("labels", {}),
            annotations=context.get("annotations", {}),
            starts_at=db_alert["created_at"],
            ends_at=db_alert.get("resolved_at"),
            fingerprint=context.get("fingerprint", ""),
            acknowledgment_by=db_alert.get("acknowledged_by"),
            acknowledgment_at=db_alert.get("acknowledged_at"),
            escalated=context.get("escalated", False),
            escalation_count=context.get("escalation_count", 0),
        )

    @property
    def duration(self) -> timedelta:
        """Get alert duration."""
        end_time = self.ends_at or datetime.now(timezone.utc)
        return end_time - self.starts_at


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""

    # Email configuration
    email_smtp_server: str = "localhost"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = "tbot-alerts@example.com"
    email_to: List[str] = field(default_factory=list)
    email_use_tls: bool = True

    # Slack configuration
    slack_webhook_url: str = ""
    slack_channel: str = "#tbot-alerts"
    slack_username: str = "T-Bot Alerts"
    slack_icon_emoji: str = ":robot_face:"

    # Discord configuration
    discord_webhook_url: str = ""

    # Webhook configuration
    webhook_urls: List[str] = field(default_factory=list)
    webhook_timeout: int = DEFAULT_WEBHOOK_TIMEOUT

    # SMS configuration
    sms_provider: str = ""
    sms_api_key: str = ""
    sms_phone_numbers: List[str] = field(default_factory=list)


@dataclass
class EscalationPolicy:
    """Alert escalation policy."""

    name: str
    description: str
    severity_levels: List[AlertSeverity]
    escalation_rules: List[Dict[str, Any]]  # [{"delay": "15m", "channels": ["email"]}, ...]
    max_escalations: int = DEFAULT_ESCALATION_MAX
    enabled: bool = True


class AlertManager(BaseComponent):
    """
    Central alert manager for the T-Bot trading system.

    Manages alert rules, processes incoming alerts, handles notifications,
    and implements escalation policies.
    """

    def __init__(self, config: NotificationConfig, error_handler=None):
        """
        Initialize alert manager.

        Args:
            config: Notification configuration
            error_handler: Error handler instance for pattern analytics
        """
        super().__init__(name="AlertManager")  # Initialize BaseComponent
        self.config = config
        # Use injected error handler for proper service layer separation
        self._error_handler = error_handler
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=ALERT_HISTORY_MAX_SIZE)
        self._suppression_rules: List[Dict[str, Any]] = []
        self._escalation_policies: Dict[str, EscalationPolicy] = {}
        self._running = False
        self._background_task: Union[asyncio.Task, None] = None
        self._notification_queue: asyncio.Queue = asyncio.Queue()
        # Initialize HTTP session manager for connection pooling
        try:
            from src.utils.monitoring_helpers import HTTPSessionManager

            self._http_session_manager = HTTPSessionManager()
        except ImportError:
            # Fallback to basic session management
            self._http_session_manager: Union[HTTPSessionManager, None] = None

        # Metrics for alerting system
        self._alerts_fired = 0
        self._alerts_resolved = 0
        self._notifications_sent = 0
        self._escalations_triggered = 0

        self.logger.debug("AlertManager initialized")

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        self._rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name} (severity: {rule.severity.value})")

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule.

        Args:
            rule_name: Name of rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        if rule_name in self._rules:
            del self._rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False

    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """
        Add an escalation policy.

        Args:
            policy: Escalation policy to add
        """
        self._escalation_policies[policy.name] = policy
        self.logger.info(f"Added escalation policy: {policy.name}")

    def add_suppression_rule(self, rule: Dict[str, Any]) -> None:
        """
        Add an alert suppression rule.

        Args:
            rule: Suppression rule (e.g., {"labels": {"environment": "test"}, "duration": "1h"})
        """
        self._suppression_rules.append(rule)
        self.logger.info(f"Added suppression rule: {rule}")

    @retry(max_attempts=ALERT_RETRY_MAX_ATTEMPTS, delay=ALERT_RETRY_DELAY)
    @logged(level="info")
    @monitored()
    async def fire_alert(self, alert: Alert) -> None:
        """
        Fire a new alert.

        Args:
            alert: Alert to fire
        """
        try:
            # Check if alert should be suppressed
            if self._is_suppressed(alert):
                alert.status = AlertStatus.SUPPRESSED
                self.logger.debug(f"Alert suppressed: {alert.rule_name}")
                return

            # Check for existing alert with same fingerprint
            existing_alert = self._active_alerts.get(alert.fingerprint)
            if existing_alert:
                # Update existing alert
                existing_alert.ends_at = None  # Reset end time
                self.logger.debug(f"Updated existing alert: {alert.rule_name}")
                return

            # Add new alert
            self._active_alerts[alert.fingerprint] = alert
            self._alert_history.append(alert)  # Automatically drops oldest if full
            self._alerts_fired += 1

            # Queue notification
            await self._notification_queue.put(("fire", alert))

            self.logger.warning(
                f"Alert fired: {alert.rule_name} (severity: {alert.severity.value}) - "
                f"{alert.message}"
            )

        except Exception as e:
            # Record error pattern for analysis
            if hasattr(self, "_error_handler") and self._error_handler:
                error_context = await create_error_context(
                    "AlertManager",
                    "fire_alert",
                    e,
                    details={
                        "alert_name": alert.rule_name,
                        "severity": alert.severity.value,
                    },
                )
                await handle_error_with_fallback(e, self._error_handler, error_context)

            # Use consistent error propagation with analytics
            from src.core.exceptions import ComponentError

            raise ComponentError(
                f"Failed to fire alert: {e}",
                component="AlertManager",
                operation="fire_alert",
                context={
                    "alert_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "fingerprint": alert.fingerprint,
                },
            ) from e

    async def resolve_alert(self, fingerprint: str) -> None:
        """
        Resolve an active alert.

        Args:
            fingerprint: Alert fingerprint
        """
        try:
            alert = self._active_alerts.get(fingerprint)
            if not alert:
                return

            alert.status = AlertStatus.RESOLVED
            alert.ends_at = datetime.now(timezone.utc)

            # Remove from active alerts
            del self._active_alerts[fingerprint]
            self._alerts_resolved += 1

            # Queue resolution notification
            await self._notification_queue.put(("resolve", alert))

            self.logger.info(f"Alert resolved: {alert.rule_name}")

        except Exception as e:
            if self._error_handler:
                error_context = ErrorContext(
                    component="AlertManager",
                    operation="resolve_alert",
                    details={
                        "fingerprint": fingerprint,
                    },
                    error=e
                )
                # Use proper async error handling
                try:
                    if hasattr(self._error_handler, "handle_error"):
                        await self._error_handler.handle_error(e, error_context)
                    elif hasattr(self._error_handler, "handle_error_sync"):
                        # Don't await sync methods
                        self._error_handler.handle_error_sync(
                            e,
                            error_context.component or "AlertManager",
                            error_context.operation or "resolve_alert"
                        )
                    else:
                        self.logger.error("No suitable error handler method available")
                except Exception as handler_error:
                    self.logger.error(f"Error handler call failed: {handler_error}")
                    # Critical alert system errors should be escalated
                    if isinstance(e, (ConnectionError, OSError, MemoryError)):
                        self.logger.critical(f"Critical alert system error in resolve_alert: {e}")
                        raise  # Re-raise critical errors
            # Use consistent error propagation with analytics
            from src.core.exceptions import ComponentError

            # Re-raise critical system errors with proper context
            if isinstance(e, (MemoryError, OSError)):
                raise ComponentError(
                    f"Critical system error resolving alert: {e}",
                    component="AlertManager",
                    operation="resolve_alert",
                    context={"fingerprint": fingerprint, "critical": True},
                ) from e
            else:
                # Log non-critical errors but don't re-raise to maintain alert system stability
                self.logger.error(f"Error resolving alert: {e}")

    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            fingerprint: Alert fingerprint
            acknowledged_by: User who acknowledged the alert

        Returns:
            True if acknowledgment was successful
        """
        try:
            alert = self._active_alerts.get(fingerprint)
            if not alert:
                return False

            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledgment_by = acknowledged_by
            alert.acknowledgment_at = datetime.now(timezone.utc)

            self.logger.info(f"Alert acknowledged by {acknowledged_by}: {alert.rule_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False

    def get_active_alerts(self, severity: Union[AlertSeverity, None] = None) -> List[Alert]:
        """
        Get active alerts, optionally filtered by severity.

        Args:
            severity: Optional severity filter

        Returns:
            List of active alerts
        """
        alerts = list(self._active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda x: x.starts_at, reverse=True)

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """
        Get alert history.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        # Convert deque to list and get last 'limit' items
        alerts = list(self._alert_history)[-limit:]
        return sorted(alerts, key=lambda x: x.starts_at, reverse=True)

    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get alerting system statistics.

        Returns:
            Dictionary with alerting statistics
        """
        active_by_severity = {}
        for severity in AlertSeverity:
            active_by_severity[severity.value] = len(
                [a for a in self._active_alerts.values() if a.severity == severity]
            )

        return {
            "active_alerts": len(self._active_alerts),
            "active_by_severity": active_by_severity,
            "total_fired": self._alerts_fired,
            "total_resolved": self._alerts_resolved,
            "notifications_sent": self._notifications_sent,
            "escalations_triggered": self._escalations_triggered,
            "rules_count": len(self._rules),
            "suppression_rules_count": len(self._suppression_rules),
        }

    async def start(self) -> None:
        """Start alert manager background tasks."""
        if self._running:
            self.logger.warning("Alert manager already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._processing_loop())
        self.logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop alert manager background tasks."""
        self._running = False

        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await asyncio.wait_for(self._background_task, timeout=ALERT_BACKGROUND_TASK_TIMEOUT)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self.logger.warning("Background task did not terminate gracefully")
            except Exception as e:
                self.logger.error(f"Error stopping background task: {e}")
            finally:
                self._background_task = None

        # Clear notification queue to prevent memory leaks
        while not self._notification_queue.empty():
            try:
                self._notification_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.logger.info("Alert manager stopped")

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        await self.stop()

        # Close HTTP sessions
        if hasattr(self, "_http_session_manager") and self._http_session_manager:
            try:
                await self._http_session_manager.close_all()
            except Exception as e:
                self.logger.warning(f"Error closing HTTP session manager: {e}")

        # Clear all data structures to prevent memory leaks
        self._active_alerts.clear()
        self._alert_history.clear()
        self._rules.clear()
        self._suppression_rules.clear()
        self._escalation_policies.clear()

        self.logger.info("Alert manager cleanup completed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    def _is_suppressed(self, alert: Alert) -> bool:
        """
        Check if an alert should be suppressed.

        Args:
            alert: Alert to check

        Returns:
            True if alert should be suppressed
        """
        for rule in self._suppression_rules:
            # Check label matching
            if "labels" in rule:
                rule_labels = rule["labels"]
                if all(alert.labels.get(k) == v for k, v in rule_labels.items()):
                    # Check if suppression is still active
                    # Time-based suppression rules can be implemented here
                    return True

        return False

    async def _processing_loop(self) -> None:
        """Background loop for processing notifications and escalations."""
        processing_tasks = set()

        while self._running:
            try:
                # Clean up completed tasks to prevent memory leaks
                processing_tasks = {task for task in processing_tasks if not task.done()}

                # Process notification queue with timeout protection
                try:
                    action, alert = await asyncio.wait_for(
                        self._notification_queue.get(),
                        timeout=ALERT_PROCESSING_CHECK_INTERVAL / 10.0,
                    )

                    # Process notifications concurrently to avoid blocking
                    if action == "fire":
                        task = asyncio.create_task(
                            asyncio.wait_for(
                                self._send_notifications(alert), timeout=ALERT_NOTIFICATION_TIMEOUT
                            )
                        )
                        processing_tasks.add(task)
                        # Don't await here to process multiple notifications concurrently
                    elif action == "resolve":
                        task = asyncio.create_task(
                            asyncio.wait_for(
                                self._send_resolution_notifications(alert),
                                timeout=ALERT_NOTIFICATION_TIMEOUT,
                            )
                        )
                        processing_tasks.add(task)

                except asyncio.TimeoutError:
                    pass

                # Check for escalations with timeout protection
                try:
                    await asyncio.wait_for(
                        self._check_escalations(), timeout=ALERT_ESCALATION_CHECK_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    self.logger.error("Escalation check timed out")
                except Exception as e:
                    self.logger.error(f"Error checking escalations: {e}")

                try:
                    await asyncio.sleep(ALERT_PROCESSING_CHECK_INTERVAL)
                except asyncio.CancelledError:
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    break

        # Cancel remaining processing tasks on shutdown
        for task in processing_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete or timeout
        if processing_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*processing_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some notification tasks did not complete during shutdown")

    async def _send_notifications(self, alert: Alert) -> None:
        """
        Send notifications for a fired alert.

        Args:
            alert: Alert to send notifications for
        """
        try:
            rule = self._rules.get(alert.rule_name)
            if not rule:
                return

            for channel in rule.notification_channels:
                try:
                    if channel == NotificationChannel.EMAIL:
                        await self._send_email_notification(alert)
                    elif channel == NotificationChannel.SLACK:
                        await self._send_slack_notification(alert)
                    elif channel == NotificationChannel.WEBHOOK:
                        await self._send_webhook_notification(alert)
                    elif channel == NotificationChannel.DISCORD:
                        await self._send_discord_notification(alert)

                    self._notifications_sent += 1

                except Exception as e:
                    # Track notification failures in error handling system
                    if self._error_handler:
                        error_context = ErrorContext(
                            component="AlertManager",
                            operation="send_notification",
                            details={
                                "channel": channel.value,
                                "alert_name": alert.rule_name,
                            },
                            error=e
                        )
                        # Properly handle error handling without incorrect await
                        try:
                            if hasattr(self._error_handler, "handle_error"):
                                await self._error_handler.handle_error(e, error_context)
                            elif hasattr(self._error_handler, "handle_error_sync"):
                                # Use sync version if async not available - don't await
                                self._error_handler.handle_error_sync(
                                    e,
                                    error_context.component or "AlertManager",
                                    error_context.operation or "send_notification"
                                )
                            else:
                                self.logger.error("No suitable error handler method available")
                        except Exception as handler_error:
                            self.logger.error(f"Error handler failed: {handler_error}")
                    self.logger.error(f"Failed to send {channel.value} notification: {e}")

            alert.notification_sent = True

        except Exception as e:
            self.logger.error(f"Error sending notifications for alert: {e}")

    async def _send_resolution_notifications(self, alert: Alert) -> None:
        """
        Send resolution notifications.

        Args:
            alert: Resolved alert
        """
        try:
            rule = self._rules.get(alert.rule_name)
            if not rule:
                return

            # Send resolution notifications to same channels as the original alert
            for channel in rule.notification_channels:
                try:
                    if channel == NotificationChannel.SLACK:
                        await self._send_slack_resolution(alert)
                    elif channel == NotificationChannel.EMAIL:
                        await self._send_email_resolution(alert)
                    elif channel == NotificationChannel.WEBHOOK:
                        await self._send_webhook_resolution(alert)
                    elif channel == NotificationChannel.DISCORD:
                        await self._send_discord_resolution(alert)

                except Exception as e:
                    self.logger.error(f"Failed to send {channel.value} resolution: {e}")

        except Exception as e:
            self.logger.error(f"Error sending resolution notifications: {e}")

    async def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification using async executor to avoid blocking."""
        if not self.config.email_to:
            return

        subject = f"[T-Bot Alert] {alert.severity.value.upper()}: {alert.rule_name}"

        body = f"""
Alert: {alert.rule_name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value}
Message: {alert.message}
Started: {alert.starts_at.strftime("%Y-%m-%d %H:%M:%S UTC")}

Labels:
{chr(10).join(f"  {k}: {v}" for k, v in alert.labels.items())}

Annotations:
{chr(10).join(f"  {k}: {v}" for k, v in alert.annotations.items())}

Fingerprint: {alert.fingerprint}
        """

        # Run email sending with retry logic
        loop = asyncio.get_event_loop()
        max_retries = EMAIL_RETRY_MAX_ATTEMPTS
        retry_delay = EMAIL_RETRY_BASE_DELAY

        for attempt in range(max_retries):
            try:
                await loop.run_in_executor(None, self._send_email_sync, subject, body)
                self.logger.debug(f"Email notification sent for alert: {alert.rule_name}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Failed to send email notification "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay * (2**attempt))  # Exponential backoff
                else:
                    self.logger.error(
                        f"Failed to send email notification after {max_retries} attempts: {e}"
                    )
                    raise

    def _send_email_sync(self, subject: str, body: str) -> None:
        """Synchronous email sending helper."""
        server = None
        try:
            msg = MIMEMultipart()
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            try:
                if self.config.email_use_tls:
                    server.starttls()

                if self.config.email_username and self.config.email_password:
                    try:
                        server.login(self.config.email_username, self.config.email_password)
                    except Exception as e:
                        # Sanitize error message to avoid logging credentials
                        self.logger.error(
                            "SMTP authentication failed for user "
                            f"'{self.config.email_username[:3]}***': {e}"
                        )
                        raise

                server.send_message(msg)
            except Exception as e:
                # Ensure server is closed on any error during operations
                self.logger.error(f"Failed to send email: {e}")
                if server:
                    try:
                        server.quit()
                    except Exception as close_error:
                        self.logger.warning(
                            f"Failed to close SMTP server after error: {close_error}"
                        )
                    server = None
                raise
        finally:
            if server:
                try:
                    server.quit()
                except Exception as close_error:
                    self.logger.warning(f"Failed to properly close SMTP server: {close_error}")

    async def _send_email_resolution(self, alert: Alert) -> None:
        """Send email resolution notification using async executor."""
        if not self.config.email_to:
            return

        subject = f"[T-Bot Alert] RESOLVED: {alert.rule_name}"
        duration = alert.duration

        body = f"""
Alert RESOLVED: {alert.rule_name}
Duration: {duration}
Resolved: {alert.ends_at.strftime("%Y-%m-%d %H:%M:%S UTC") if alert.ends_at else "N/A"}

Original Alert:
Severity: {alert.severity.value.upper()}
Message: {alert.message}
Started: {alert.starts_at.strftime("%Y-%m-%d %H:%M:%S UTC")}

Fingerprint: {alert.fingerprint}
        """

        # Run email sending in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._send_email_sync, subject, body)
            self.logger.debug(f"Email resolution sent for alert: {alert.rule_name}")
        except Exception as e:
            self.logger.error(f"Failed to send email resolution: {e}")

    async def _send_slack_notification(self, alert: Alert) -> None:
        """Send Slack notification."""
        if not self.config.slack_webhook_url:
            return

        # Choose color based on severity
        color_map = {
            AlertSeverity.CRITICAL: SEVERITY_COLORS["critical"],
            AlertSeverity.HIGH: SEVERITY_COLORS["high"],
            AlertSeverity.MEDIUM: SEVERITY_COLORS["medium"],
            AlertSeverity.LOW: SEVERITY_COLORS["low"],
            AlertSeverity.INFO: SEVERITY_COLORS["info"],
        }

        payload = {
            "username": self.config.slack_username,
            "icon_emoji": self.config.slack_icon_emoji,
            "channel": self.config.slack_channel,
            "attachments": [
                {
                    "color": color_map.get(alert.severity, SEVERITY_COLORS["default"]),
                    "title": f"🚨 {alert.severity.value.upper()}: {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Status", "value": alert.status.value, "short": True},
                        {
                            "title": "Started",
                            "value": alert.starts_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True,
                        },
                        {
                            "title": "Labels",
                            "value": "\n".join(f"{k}: {v}" for k, v in alert.labels.items()),
                            "short": False,
                        },
                    ],
                    "footer": "T-Bot Alert System",
                    "ts": int(alert.starts_at.timestamp()),
                }
            ],
        }

        session = None
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_SESSION_TIMEOUT),
                connector=aiohttp.TCPConnector(
                    limit=HTTP_CONNECTOR_LIMIT, limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST
                ),
            )
            try:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                ) as response:
                    if response.status == HTTP_OK:
                        self.logger.debug(f"Slack notification sent for alert: {alert.rule_name}")
                    else:
                        self.logger.error(
                            f"Slack notification failed with status {response.status}"
                        )
            finally:
                # Ensure proper connection cleanup with timeout
                if session and not session.closed:
                    try:
                        await asyncio.wait_for(session.close(), timeout=SESSION_CLOSE_TIMEOUT)
                    except asyncio.TimeoutError:
                        self.logger.warning("Session close timed out for Slack notification")
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            # Enhanced cleanup for WebSocket-related errors
            if session and not session.closed:
                try:
                    await asyncio.wait_for(session.close(), timeout=SESSION_CLEANUP_TIMEOUT)
                except (asyncio.TimeoutError, Exception) as cleanup_error:
                    self.logger.warning(f"Session cleanup failed: {cleanup_error}")

    async def _send_slack_resolution(self, alert: Alert) -> None:
        """Send Slack resolution notification."""
        if not self.config.slack_webhook_url:
            return

        duration = alert.duration

        payload = {
            "username": self.config.slack_username,
            "icon_emoji": ":white_check_mark:",
            "channel": self.config.slack_channel,
            "attachments": [
                {
                    "color": SEVERITY_COLORS["success"],
                    "title": f"✅ RESOLVED: {alert.rule_name}",
                    "text": f"Alert resolved after {duration}",
                    "fields": [
                        {"title": "Duration", "value": str(duration), "short": True},
                        {
                            "title": "Resolved",
                            "value": (
                                alert.ends_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                                if alert.ends_at
                                else "N/A"
                            ),
                            "short": True,
                        },
                    ],
                    "footer": "T-Bot Alert System",
                    "ts": int(alert.ends_at.timestamp()) if alert.ends_at else int(time.time()),
                }
            ],
        }

        session = None
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_SESSION_TIMEOUT),
                connector=aiohttp.TCPConnector(
                    limit=HTTP_CONNECTOR_LIMIT, limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST
                ),
            )
            try:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                ) as response:
                    if response.status == HTTP_OK:
                        self.logger.debug(f"Slack resolution sent for alert: {alert.rule_name}")
                    else:
                        self.logger.error(f"Slack resolution failed with status {response.status}")
            finally:
                # Ensure proper connection cleanup with timeout
                if session and not session.closed:
                    try:
                        await asyncio.wait_for(session.close(), timeout=SESSION_CLOSE_TIMEOUT)
                    except asyncio.TimeoutError:
                        self.logger.warning("Session close timed out for Slack resolution")
        except Exception as e:
            self.logger.error(f"Failed to send Slack resolution: {e}")
            # Enhanced cleanup for WebSocket-related errors
            if session and not session.closed:
                try:
                    await asyncio.wait_for(session.close(), timeout=SESSION_CLEANUP_TIMEOUT)
                except (asyncio.TimeoutError, Exception) as cleanup_error:
                    self.logger.warning(f"Slack resolution session cleanup failed: {cleanup_error}")

    async def _send_webhook_notification(self, alert: Alert) -> None:
        """Send webhook notification."""
        if not self.config.webhook_urls:
            return

        payload = {
            "event": AlertEvents.FIRED,
            "alert": {
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "starts_at": alert.starts_at.isoformat(),
                "fingerprint": alert.fingerprint,
            },
        }

        for webhook_url in self.config.webhook_urls:
            max_retries = WEBHOOK_RETRY_MAX_ATTEMPTS
            retry_delay = WEBHOOK_RETRY_BASE_DELAY

            for attempt in range(max_retries):
                session = None
                try:
                    # Use session manager for connection pooling if available
                    if self._http_session_manager:
                        session = await self._http_session_manager.get_session(
                            "webhooks", timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout)
                        )
                        async with session.post(
                            webhook_url,
                            json=payload,
                        ) as response:
                            if response.status == HTTP_OK:
                                self.logger.debug(
                                    f"Webhook notification sent for alert: {alert.rule_name}"
                                )
                                break
                            else:
                                self.logger.error(
                                    f"Webhook notification failed with status {response.status}"
                                )
                                if (
                                    response.status >= HTTP_SERVER_ERROR_THRESHOLD
                                    and attempt < max_retries - 1
                                ):
                                    # Retry on server errors
                                    await asyncio.sleep(retry_delay * (2**attempt))
                                    continue
                                break
                    else:
                        # Fallback to direct session creation
                        session = aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout),
                            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5),
                        )
                        async with session.post(
                            webhook_url,
                            json=payload,
                        ) as response:
                            if response.status == HTTP_OK:
                                self.logger.debug(
                                    f"Webhook notification sent for alert: {alert.rule_name}"
                                )
                                break
                            else:
                                self.logger.error(
                                    f"Webhook notification failed with status {response.status}"
                                )
                                if (
                                    response.status >= HTTP_SERVER_ERROR_THRESHOLD
                                    and attempt < max_retries - 1
                                ):
                                    # Retry on server errors
                                    await asyncio.sleep(retry_delay * (2**attempt))
                                    continue
                                break

                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"Failed to send webhook notification to {webhook_url} (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        await asyncio.sleep(retry_delay * (2**attempt))
                    else:
                        self.logger.error(
                            f"Failed to send webhook notification to {webhook_url} after {max_retries} attempts: {e}"
                        )
                finally:
                    # Close session if we created it directly
                    if session and not self._http_session_manager and not session.closed:
                        await session.close()

    async def _send_webhook_resolution(self, alert: Alert) -> None:
        """Send webhook resolution notification."""
        if not self.config.webhook_urls:
            return

        payload = {
            "event": AlertEvents.RESOLVED,
            "alert": {
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "starts_at": alert.starts_at.isoformat(),
                "ends_at": alert.ends_at.isoformat() if alert.ends_at else None,
                "duration": str(alert.duration),
                "fingerprint": alert.fingerprint,
            },
        }

        for webhook_url in self.config.webhook_urls:
            session = None
            try:
                session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout),
                    connector=aiohttp.TCPConnector(limit=10, limit_per_host=5),
                )
                try:
                    async with session.post(
                        webhook_url,
                        json=payload,
                    ) as response:
                        if response.status == HTTP_OK:
                            self.logger.debug(
                                f"Webhook resolution sent for alert: {alert.rule_name}"
                            )
                        else:
                            self.logger.error(
                                f"Webhook resolution failed with status {response.status}"
                            )
                finally:
                    # Ensure proper connection cleanup with timeout
                    if session and not session.closed:
                        try:
                            await asyncio.wait_for(session.close(), timeout=SESSION_CLOSE_TIMEOUT)
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                f"Session close timed out for webhook {webhook_url}"
                            )
            except Exception as e:
                self.logger.error(f"Failed to send webhook resolution to {webhook_url}: {e}")
                # Enhanced cleanup for WebSocket-related errors
                if session and not session.closed:
                    try:
                        await asyncio.wait_for(session.close(), timeout=SESSION_CLEANUP_TIMEOUT)
                    except (asyncio.TimeoutError, Exception) as cleanup_error:
                        self.logger.warning(
                            f"Webhook session cleanup failed for {webhook_url}: {cleanup_error}"
                        )

    async def _send_discord_notification(self, alert: Alert) -> None:
        """Send Discord notification."""
        if not self.config.discord_webhook_url:
            return

        # Choose color based on severity
        color_map = {
            AlertSeverity.CRITICAL: DISCORD_SEVERITY_COLORS["critical"],
            AlertSeverity.HIGH: DISCORD_SEVERITY_COLORS["high"],
            AlertSeverity.MEDIUM: DISCORD_SEVERITY_COLORS["medium"],
            AlertSeverity.LOW: DISCORD_SEVERITY_COLORS["low"],
            AlertSeverity.INFO: DISCORD_SEVERITY_COLORS["info"],
        }

        payload = {
            "embeds": [
                {
                    "title": f"🚨 {alert.severity.value.upper()}: {alert.rule_name}",
                    "description": alert.message,
                    "color": color_map.get(alert.severity, DISCORD_SEVERITY_COLORS["default"]),
                    "fields": [
                        {"name": "Status", "value": alert.status.value, "inline": True},
                        {
                            "name": "Started",
                            "value": alert.starts_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "inline": True,
                        },
                        {
                            "name": "Labels",
                            "value": "\n".join(f"{k}: {v}" for k, v in alert.labels.items())
                            or "None",
                            "inline": False,
                        },
                    ],
                    "footer": {"text": "T-Bot Alert System"},
                    "timestamp": alert.starts_at.isoformat(),
                }
            ]
        }

        session = None
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_SESSION_TIMEOUT),
                connector=aiohttp.TCPConnector(
                    limit=HTTP_CONNECTOR_LIMIT, limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST
                ),
            )
            try:
                async with session.post(
                    self.config.discord_webhook_url,
                    json=payload,
                ) as response:
                    if response.status in [HTTP_OK, 204]:
                        self.logger.debug(f"Discord notification sent for alert: {alert.rule_name}")
                    else:
                        self.logger.error(
                            f"Discord notification failed with status {response.status}"
                        )
            finally:
                if not session.closed:
                    await asyncio.wait_for(session.close(), timeout=SESSION_CLOSE_TIMEOUT)
        except Exception as e:
            self.logger.error(f"Failed to send Discord notification: {e}")
            if session and not session.closed:
                await session.close()

    async def _send_discord_resolution(self, alert: Alert) -> None:
        """Send Discord resolution notification."""
        if not self.config.discord_webhook_url:
            return

        duration = alert.duration

        payload = {
            "embeds": [
                {
                    "title": f"✅ RESOLVED: {alert.rule_name}",
                    "description": f"Alert resolved after {duration}",
                    "color": DISCORD_SEVERITY_COLORS["success"],
                    "fields": [
                        {"name": "Duration", "value": str(duration), "inline": True},
                        {
                            "name": "Resolved",
                            "value": (
                                alert.ends_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                                if alert.ends_at
                                else "N/A"
                            ),
                            "inline": True,
                        },
                    ],
                    "footer": {"text": "T-Bot Alert System"},
                    "timestamp": (
                        alert.ends_at.isoformat()
                        if alert.ends_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                }
            ]
        }

        session = None
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_SESSION_TIMEOUT),
                connector=aiohttp.TCPConnector(
                    limit=HTTP_CONNECTOR_LIMIT, limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST
                ),
            )
            try:
                async with session.post(
                    self.config.discord_webhook_url,
                    json=payload,
                ) as response:
                    if response.status in [HTTP_OK, 204]:
                        self.logger.debug(f"Discord resolution sent for alert: {alert.rule_name}")
                    else:
                        self.logger.error(
                            f"Discord resolution failed with status {response.status}"
                        )
            finally:
                if not session.closed:
                    await asyncio.wait_for(session.close(), timeout=SESSION_CLOSE_TIMEOUT)
        except Exception as e:
            self.logger.error(f"Failed to send Discord resolution: {e}")
            if session and not session.closed:
                await session.close()

    async def _check_escalations(self) -> None:
        """Check for alerts that need escalation."""
        current_time = datetime.now(timezone.utc)

        for alert in self._active_alerts.values():
            if alert.status != AlertStatus.FIRING or alert.escalated:
                continue

            rule = self._rules.get(alert.rule_name)
            if not rule or not rule.escalation_delay:
                continue

            # Parse escalation delay using robust duration parsing
            escalation_minutes = self._parse_duration_minutes(rule.escalation_delay)
            escalation_time = alert.starts_at + timedelta(minutes=escalation_minutes)

            if current_time >= escalation_time:
                await self._escalate_alert(alert)

    def _parse_duration_minutes(self, duration: str) -> int:
        """
        Parse duration string to minutes with proper financial time handling.

        Args:
            duration: Duration string like "5m", "1h", "30s", "1d"

        Returns:
            Duration in minutes

        Raises:
            ValueError: If duration format is invalid
        """
        if not duration or not isinstance(duration, str):
            raise ValueError("Duration must be a non-empty string")

        duration = duration.strip().lower()

        try:
            if duration.endswith("s"):
                seconds = int(duration[:-1])
                if seconds <= 0:
                    raise ValueError(f"Duration must be positive: {duration}")
                return max(DURATION_PARSE_MINIMUM_MINUTES, seconds // SECONDS_PER_MINUTE)
            elif duration.endswith("m"):
                minutes = int(duration[:-1])
                if minutes <= 0:
                    raise ValueError(f"Duration must be positive: {duration}")
                return max(DURATION_PARSE_MINIMUM_MINUTES, minutes)
            elif duration.endswith("h"):
                hours = int(duration[:-1])
                if hours <= 0:
                    raise ValueError(f"Duration must be positive: {duration}")
                return max(DURATION_PARSE_MINIMUM_MINUTES, hours * 60)
            elif duration.endswith("d"):
                days = int(duration[:-1])
                if days <= 0:
                    raise ValueError(f"Duration must be positive: {duration}")
                return max(1, days * 24 * 60)  # Convert to minutes (24h trading day)
            elif duration.endswith("w"):
                weeks = int(duration[:-1])
                if weeks <= 0:
                    raise ValueError(f"Duration must be positive: {duration}")
                return max(1, weeks * 7 * 24 * 60)  # Convert to minutes
            else:
                # Try parsing as raw number (assume minutes)
                minutes = int(duration)
                if minutes <= 0:
                    raise ValueError(f"Duration must be positive: {duration}")
                return max(1, minutes)
        except (ValueError, IndexError, TypeError) as e:
            if isinstance(e, ValueError) and "must be positive" in str(e):
                raise  # Re-raise our custom validation error
            raise ValueError(
                f"Invalid duration format: '{duration}'. Use format like '5m', '1h', '30s', '1d'. "
                f"Original error: {e}"
            ) from e

    async def _escalate_alert(self, alert: Alert) -> None:
        """
        Escalate an alert.

        Args:
            alert: Alert to escalate
        """
        try:
            alert.escalated = True
            alert.escalation_count += 1
            self._escalations_triggered += 1

            # Create escalated alert message
            escalated_alert = Alert(
                rule_name=f"ESCALATED: {alert.rule_name}",
                severity=AlertSeverity.CRITICAL,  # Escalated alerts are always critical
                status=AlertStatus.FIRING,
                message=f"ESCALATED ALERT: {alert.message} (escalation #{alert.escalation_count})",
                labels=alert.labels.copy(),
                annotations=alert.annotations.copy(),
                starts_at=datetime.now(timezone.utc),
            )

            # Add escalation information
            escalated_alert.labels["escalated"] = "true"
            escalated_alert.labels["original_severity"] = alert.severity.value
            escalated_alert.annotations["escalation_count"] = str(alert.escalation_count)

            # Send escalation notifications
            await self._send_notifications(escalated_alert)

            self.logger.warning(
                f"Alert escalated: {alert.rule_name} (escalation #{alert.escalation_count})"
            )

        except Exception as e:
            self.logger.error(f"Error escalating alert: {e}")


def load_alert_rules_from_file(file_path: str) -> List[AlertRule]:
    """
    Load alert rules from a YAML file.

    Args:
        file_path: Path to YAML file containing alert rules

    Returns:
        List of AlertRule objects

    Raises:
        MonitoringError: If file loading fails
    """
    if not YAML_AVAILABLE:
        raise MonitoringError("YAML library is not available. Please install PyYAML.")

    try:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        rules = []
        for rule_data in data.get("rules", []):
            # Convert string severity to enum
            severity = AlertSeverity(rule_data["severity"])

            # Convert string channels to enum list
            channels = [
                NotificationChannel(ch) for ch in rule_data.get("notification_channels", [])
            ]

            rule = AlertRule(
                name=rule_data["name"],
                description=rule_data["description"],
                severity=severity,
                query=rule_data["query"],
                threshold=float(rule_data["threshold"]),
                operator=rule_data["operator"],
                duration=rule_data["duration"],
                labels=rule_data.get("labels", {}),
                annotations=rule_data.get("annotations", {}),
                notification_channels=channels,
                escalation_delay=rule_data.get("escalation_delay"),
                enabled=rule_data.get("enabled", True),
            )
            rules.append(rule)

        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(rules)} alert rules from {file_path}")
        return rules
    except Exception as e:
        raise MonitoringError(f"Failed to load alert rules from {file_path}: {e}") from e


# HTTPSessionManager and related functions moved to src.utils.monitoring_helpers


# Global alert manager instance
_global_alert_manager: Optional["AlertManager"] = None


def get_alert_manager() -> Optional["AlertManager"]:
    """
    Get alert manager instance using factory pattern.

    Returns:
        AlertManager instance from factory or None if creation fails
    """
    try:
        from src.monitoring.dependency_injection import get_monitoring_container

        container = get_monitoring_container()
        return container.resolve(AlertManager)
    except (ServiceError, MonitoringError, ImportError, KeyError, ValueError) as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to resolve alert manager from DI container: {e}")
        # Fallback to global instance for backward compatibility
        return _global_alert_manager


def set_global_alert_manager(alert_manager: AlertManager) -> None:
    """
    Set the global alert manager instance.

    Note: This is for backward compatibility. Prefer using dependency injection.

    Args:
        alert_manager: AlertManager instance
    """
    global _global_alert_manager
    _global_alert_manager = alert_manager
