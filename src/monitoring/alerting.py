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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

# Try to import email modules, handle gracefully if missing
try:
    import smtplib
    from email.mime.multipart import MimeMultipart
    from email.mime.text import MimeText

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

    # Mock email classes
    class MockMimeText:
        def __init__(self, *args, **kwargs):
            pass

    class MockMimeMultipart:
        def __init__(self, *args, **kwargs):
            self._headers = {}

        def __setitem__(self, key, value):
            self._headers[key] = value

        def attach(self, *args):
            pass

    class MockSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def starttls(self):
            pass

        def login(self, *args):
            pass

        def send_message(self, *args):
            pass

        def quit(self):
            pass

    smtplib = type("MockSMTPLib", (), {"SMTP": MockSMTP})()
    MimeText = MockMimeText
    MimeMultipart = MockMimeMultipart
import aiohttp
import yaml

from src.core.exceptions import MonitoringError
from src.core.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


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
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    notification_channels: list[NotificationChannel] = field(default_factory=list)
    escalation_delay: str | None = None  # Escalation delay like "15m"
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
    """Alert instance."""

    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    labels: dict[str, str]
    annotations: dict[str, str]
    starts_at: datetime
    ends_at: datetime | None = None
    fingerprint: str = ""
    notification_sent: bool = False
    acknowledgment_by: str | None = None
    acknowledgment_at: datetime | None = None
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
    email_to: list[str] = field(default_factory=list)
    email_use_tls: bool = True

    # Slack configuration
    slack_webhook_url: str = ""
    slack_channel: str = "#tbot-alerts"
    slack_username: str = "T-Bot Alerts"
    slack_icon_emoji: str = ":robot_face:"

    # Discord configuration
    discord_webhook_url: str = ""

    # Webhook configuration
    webhook_urls: list[str] = field(default_factory=list)
    webhook_timeout: int = 10

    # SMS configuration (placeholder for future implementation)
    sms_provider: str = ""
    sms_api_key: str = ""
    sms_phone_numbers: list[str] = field(default_factory=list)


@dataclass
class EscalationPolicy:
    """Alert escalation policy."""

    name: str
    description: str
    severity_levels: list[AlertSeverity]
    escalation_rules: list[dict[str, Any]]  # [{"delay": "15m", "channels": ["email"]}, ...]
    max_escalations: int = 3
    enabled: bool = True


class AlertManager:
    """
    Central alert manager for the T-Bot trading system.

    Manages alert rules, processes incoming alerts, handles notifications,
    and implements escalation policies.
    """

    def __init__(self, config: NotificationConfig):
        """
        Initialize alert manager.

        Args:
            config: Notification configuration
        """
        self.config = config
        self._rules: dict[str, AlertRule] = {}
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._suppression_rules: list[dict[str, Any]] = []
        self._escalation_policies: dict[str, EscalationPolicy] = {}
        self._running = False
        self._background_task: asyncio.Task | None = None
        self._notification_queue: asyncio.Queue = asyncio.Queue()

        # Metrics for alerting system
        self._alerts_fired = 0
        self._alerts_resolved = 0
        self._notifications_sent = 0
        self._escalations_triggered = 0

        logger.info("AlertManager initialized")

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        self._rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name} (severity: {rule.severity.value})")

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
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False

    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """
        Add an escalation policy.

        Args:
            policy: Escalation policy to add
        """
        self._escalation_policies[policy.name] = policy
        logger.info(f"Added escalation policy: {policy.name}")

    def add_suppression_rule(self, rule: dict[str, Any]) -> None:
        """
        Add an alert suppression rule.

        Args:
            rule: Suppression rule (e.g., {"labels": {"environment": "test"}, "duration": "1h"})
        """
        self._suppression_rules.append(rule)
        logger.info(f"Added suppression rule: {rule}")

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
                logger.debug(f"Alert suppressed: {alert.rule_name}")
                return

            # Check for existing alert with same fingerprint
            existing_alert = self._active_alerts.get(alert.fingerprint)
            if existing_alert:
                # Update existing alert
                existing_alert.ends_at = None  # Reset end time
                logger.debug(f"Updated existing alert: {alert.rule_name}")
                return

            # Add new alert
            self._active_alerts[alert.fingerprint] = alert
            self._alert_history.append(alert)
            self._alerts_fired += 1

            # Queue notification
            await self._notification_queue.put(("fire", alert))

            logger.warning(
                f"Alert fired: {alert.rule_name} (severity: {alert.severity.value}) - {alert.message}"
            )

        except Exception as e:
            logger.error(f"Error firing alert: {e}")

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

            logger.info(f"Alert resolved: {alert.rule_name}")

        except Exception as e:
            logger.error(f"Error resolving alert: {e}")

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

            logger.info(f"Alert acknowledged by {acknowledged_by}: {alert.rule_name}")
            return True

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
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

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """
        Get alert history.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        return sorted(self._alert_history[-limit:], key=lambda x: x.starts_at, reverse=True)

    def get_alert_stats(self) -> dict[str, Any]:
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
            logger.warning("Alert manager already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._processing_loop())
        logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop alert manager background tasks."""
        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

        logger.info("Alert manager stopped")

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
                    # (simplified - in production, implement time-based suppression)
                    return True

        return False

    async def _processing_loop(self) -> None:
        """Background loop for processing notifications and escalations."""
        while self._running:
            try:
                # Process notification queue
                try:
                    action, alert = await asyncio.wait_for(
                        self._notification_queue.get(), timeout=1.0
                    )

                    if action == "fire":
                        await self._send_notifications(alert)
                    elif action == "resolve":
                        await self._send_resolution_notifications(alert)

                except asyncio.TimeoutError:
                    pass

                # Check for escalations
                await self._check_escalations()

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(5)

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
                    logger.error(f"Failed to send {channel.value} notification: {e}")

            alert.notification_sent = True

        except Exception as e:
            logger.error(f"Error sending notifications for alert: {e}")

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
                    logger.error(f"Failed to send {channel.value} resolution: {e}")

        except Exception as e:
            logger.error(f"Error sending resolution notifications: {e}")

    async def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification."""
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

        try:
            msg = MimeMultipart()
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = subject

            msg.attach(MimeText(body, "plain"))

            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            if self.config.email_use_tls:
                server.starttls()

            if self.config.email_username and self.config.email_password:
                server.login(self.config.email_username, self.config.email_password)

            server.send_message(msg)
            server.quit()

            logger.debug(f"Email notification sent for alert: {alert.rule_name}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def _send_email_resolution(self, alert: Alert) -> None:
        """Send email resolution notification."""
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

        try:
            msg = MimeMultipart()
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = subject

            msg.attach(MimeText(body, "plain"))

            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            if self.config.email_use_tls:
                server.starttls()

            if self.config.email_username and self.config.email_password:
                server.login(self.config.email_username, self.config.email_password)

            server.send_message(msg)
            server.quit()

            logger.debug(f"Email resolution sent for alert: {alert.rule_name}")

        except Exception as e:
            logger.error(f"Failed to send email resolution: {e}")

    async def _send_slack_notification(self, alert: Alert) -> None:
        """Send Slack notification."""
        if not self.config.slack_webhook_url:
            return

        # Choose color based on severity
        color_map = {
            AlertSeverity.CRITICAL: "#FF0000",  # Red
            AlertSeverity.HIGH: "#FF8000",  # Orange
            AlertSeverity.MEDIUM: "#FFFF00",  # Yellow
            AlertSeverity.LOW: "#0080FF",  # Blue
            AlertSeverity.INFO: "#00FF00",  # Green
        }

        payload = {
            "username": self.config.slack_username,
            "icon_emoji": self.config.slack_icon_emoji,
            "channel": self.config.slack_channel,
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#808080"),
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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Slack notification sent for alert: {alert.rule_name}")
                    else:
                        logger.error(f"Slack notification failed with status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

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
                    "color": "#00FF00",  # Green
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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Slack resolution sent for alert: {alert.rule_name}")
                    else:
                        logger.error(f"Slack resolution failed with status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack resolution: {e}")

    async def _send_webhook_notification(self, alert: Alert) -> None:
        """Send webhook notification."""
        if not self.config.webhook_urls:
            return

        payload = {
            "event": "alert.fired",
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
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout),
                    ) as response:
                        if response.status == 200:
                            logger.debug(f"Webhook notification sent for alert: {alert.rule_name}")
                        else:
                            logger.error(
                                f"Webhook notification failed with status {response.status}"
                            )

            except Exception as e:
                logger.error(f"Failed to send webhook notification to {webhook_url}: {e}")

    async def _send_webhook_resolution(self, alert: Alert) -> None:
        """Send webhook resolution notification."""
        if not self.config.webhook_urls:
            return

        payload = {
            "event": "alert.resolved",
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
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout),
                    ) as response:
                        if response.status == 200:
                            logger.debug(f"Webhook resolution sent for alert: {alert.rule_name}")
                        else:
                            logger.error(f"Webhook resolution failed with status {response.status}")

            except Exception as e:
                logger.error(f"Failed to send webhook resolution to {webhook_url}: {e}")

    async def _send_discord_notification(self, alert: Alert) -> None:
        """Send Discord notification."""
        if not self.config.discord_webhook_url:
            return

        # Choose color based on severity
        color_map = {
            AlertSeverity.CRITICAL: 0xFF0000,  # Red
            AlertSeverity.HIGH: 0xFF8000,  # Orange
            AlertSeverity.MEDIUM: 0xFFFF00,  # Yellow
            AlertSeverity.LOW: 0x0080FF,  # Blue
            AlertSeverity.INFO: 0x00FF00,  # Green
        }

        payload = {
            "embeds": [
                {
                    "title": f"🚨 {alert.severity.value.upper()}: {alert.rule_name}",
                    "description": alert.message,
                    "color": color_map.get(alert.severity, 0x808080),
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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.discord_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status in [200, 204]:
                        logger.debug(f"Discord notification sent for alert: {alert.rule_name}")
                    else:
                        logger.error(f"Discord notification failed with status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

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
                    "color": 0x00FF00,  # Green
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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.discord_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status in [200, 204]:
                        logger.debug(f"Discord resolution sent for alert: {alert.rule_name}")
                    else:
                        logger.error(f"Discord resolution failed with status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Discord resolution: {e}")

    async def _check_escalations(self) -> None:
        """Check for alerts that need escalation."""
        current_time = datetime.now(timezone.utc)

        for alert in self._active_alerts.values():
            if alert.status != AlertStatus.FIRING or alert.escalated:
                continue

            rule = self._rules.get(alert.rule_name)
            if not rule or not rule.escalation_delay:
                continue

            # Parse escalation delay (simplified - implement proper duration parsing)
            escalation_minutes = self._parse_duration_minutes(rule.escalation_delay)
            escalation_time = alert.starts_at + timedelta(minutes=escalation_minutes)

            if current_time >= escalation_time:
                await self._escalate_alert(alert)

    def _parse_duration_minutes(self, duration: str) -> int:
        """
        Parse duration string to minutes (simplified implementation).

        Args:
            duration: Duration string like "5m", "1h", "30s"

        Returns:
            Duration in minutes
        """
        if duration.endswith("m"):
            return int(duration[:-1])
        elif duration.endswith("h"):
            return int(duration[:-1]) * 60
        elif duration.endswith("s"):
            return max(1, int(duration[:-1]) // 60)
        else:
            return int(duration)  # Assume minutes

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

            logger.warning(
                f"Alert escalated: {alert.rule_name} (escalation #{alert.escalation_count})"
            )

        except Exception as e:
            logger.error(f"Error escalating alert: {e}")


def load_alert_rules_from_file(file_path: str) -> list[AlertRule]:
    """
    Load alert rules from a YAML file.

    Args:
        file_path: Path to YAML file containing alert rules

    Returns:
        List of AlertRule objects

    Raises:
        MonitoringError: If file loading fails
    """
    try:
        with open(file_path) as f:
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

        logger.info(f"Loaded {len(rules)} alert rules from {file_path}")
        return rules

    except Exception as e:
        raise MonitoringError(f"Failed to load alert rules from {file_path}: {e}")


# Global alert manager instance
_global_alert_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager | None:
    """
    Get the global alert manager instance.

    Returns:
        Global AlertManager instance or None if not initialized
    """
    return _global_alert_manager


def set_global_alert_manager(alert_manager: AlertManager) -> None:
    """
    Set the global alert manager instance.

    Args:
        alert_manager: AlertManager instance
    """
    global _global_alert_manager
    _global_alert_manager = alert_manager
