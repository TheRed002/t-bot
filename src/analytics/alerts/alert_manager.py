"""
Analytics Alert Management System.

This module provides comprehensive alert management for analytics events,
including threshold monitoring, alert routing, and notification services.
"""

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.analytics.types import (
    AlertSeverity,
    AnalyticsAlert,
    AnalyticsConfiguration,
)
from src.base import BaseComponent
from src.monitoring.metrics import get_metrics_collector
from src.utils.datetime_utils import get_current_utc_timestamp


class AlertRule:
    """Alert rule configuration."""

    def __init__(
        self,
        rule_id: str,
        metric_name: str,
        condition: str,
        threshold_value: float,
        severity: AlertSeverity,
        description: str,
        cooldown_minutes: int = 30,
        enabled: bool = True,
    ):
        self.rule_id = rule_id
        self.metric_name = metric_name
        self.condition = condition  # 'greater_than', 'less_than', 'equals', 'not_equals'
        self.threshold_value = threshold_value
        self.severity = severity
        self.description = description
        self.cooldown_minutes = cooldown_minutes
        self.enabled = enabled
        self.last_triggered: datetime | None = None


class AlertManager(BaseComponent):
    """
    Comprehensive alert management system for analytics.

    Provides:
    - Configurable alert rules and thresholds
    - Real-time metric monitoring and alerting
    - Alert routing and notification management
    - Alert history and analytics
    - Escalation and acknowledgment workflows
    """

    def __init__(self, config: AnalyticsConfiguration):
        """
        Initialize alert manager.

        Args:
            config: Analytics configuration
        """
        super().__init__()
        self.config = config
        self.metrics_collector = get_metrics_collector()

        # Alert rules and state
        self._alert_rules: dict[str, AlertRule] = {}
        self._active_alerts: dict[str, AnalyticsAlert] = {}
        self._alert_history: deque = deque(maxlen=10000)
        self._metric_values: dict[str, float] = {}

        # Notification handlers
        self._notification_handlers: dict[AlertSeverity, list[Callable]] = defaultdict(list)

        # Escalation configuration
        self._escalation_rules: dict[AlertSeverity, dict[str, Any]] = {}

        # Background tasks
        self._monitoring_tasks: set[asyncio.Task] = set()
        self._running = False

        # Initialize default alert rules
        self._initialize_default_rules()

        self.logger.info("AlertManager initialized")

    async def start(self) -> None:
        """Start alert monitoring and management tasks."""
        if self._running:
            self.logger.warning("Alert manager already running")
            return

        self._running = True

        # Start monitoring tasks
        tasks = [
            self._metric_monitoring_loop(),
            self._alert_maintenance_loop(),
            self._escalation_loop(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._monitoring_tasks.add(task)
            task.add_done_callback(self._monitoring_tasks.discard)

        self.logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop alert monitoring and management tasks."""
        self._running = False

        # Cancel all tasks
        for task in self._monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

        self.logger.info("Alert manager stopped")

    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        Add or update an alert rule.

        Args:
            rule: Alert rule configuration
        """
        self._alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.rule_id}")

    def remove_alert_rule(self, rule_id: str) -> None:
        """
        Remove an alert rule.

        Args:
            rule_id: Rule identifier to remove
        """
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")

    def update_metric_value(self, metric_name: str, value: float) -> None:
        """
        Update metric value for monitoring.

        Args:
            metric_name: Name of metric
            value: Current metric value
        """
        self._metric_values[metric_name] = value

        # Trigger immediate evaluation for this metric
        asyncio.create_task(self._evaluate_metric_rules(metric_name, value))

    async def generate_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        metric_name: str,
        current_value: float | None = None,
        threshold_value: float | None = None,
        affected_entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnalyticsAlert:
        """
        Generate a new analytics alert.

        Args:
            alert_id: Unique alert identifier
            severity: Alert severity level
            title: Alert title
            message: Alert message
            metric_name: Associated metric name
            current_value: Current metric value
            threshold_value: Threshold that was breached
            affected_entities: List of affected entities
            metadata: Additional metadata

        Returns:
            Generated alert
        """
        now = get_current_utc_timestamp()

        alert = AnalyticsAlert(
            id=alert_id,
            timestamp=now,
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            current_value=Decimal(str(current_value)) if current_value is not None else None,
            threshold_value=Decimal(str(threshold_value)) if threshold_value is not None else None,
            affected_entities=affected_entities or [],
            metadata=metadata or {},
        )

        # Store alert
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)

        # Send notifications
        await self._send_notifications(alert)

        # Update metrics
        self.metrics_collector.increment_counter(
            "analytics_alerts_generated", labels={"severity": severity.value, "metric": metric_name}
        )

        self.logger.warning(f"Generated alert: {title} - {message}")

        return alert

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an active alert.

        Args:
            alert_id: Alert identifier
            acknowledged_by: User who acknowledged the alert

        Returns:
            True if alert was acknowledged successfully
        """
        if alert_id not in self._active_alerts:
            return False

        alert = self._active_alerts[alert_id]
        alert.acknowledged = True
        alert.metadata["acknowledged_by"] = acknowledged_by
        alert.metadata["acknowledged_at"] = get_current_utc_timestamp().isoformat()

        self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True

    async def resolve_alert(
        self, alert_id: str, resolved_by: str, resolution_note: str | None = None
    ) -> bool:
        """
        Resolve an active alert.

        Args:
            alert_id: Alert identifier
            resolved_by: User who resolved the alert
            resolution_note: Optional resolution note

        Returns:
            True if alert was resolved successfully
        """
        if alert_id not in self._active_alerts:
            return False

        alert = self._active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_timestamp = get_current_utc_timestamp()
        alert.metadata["resolved_by"] = resolved_by

        if resolution_note:
            alert.metadata["resolution_note"] = resolution_note

        # Remove from active alerts
        del self._active_alerts[alert_id]

        self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        return True

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[AnalyticsAlert]:
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

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_history(
        self,
        limit: int = 100,
        severity: AlertSeverity | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[AnalyticsAlert]:
        """
        Get alert history with optional filters.

        Args:
            limit: Maximum number of alerts to return
            severity: Optional severity filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of historical alerts
        """
        alerts = list(self._alert_history)

        # Apply filters
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]

        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]

        # Sort by timestamp (newest first) and limit
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]

    def get_alert_statistics(self, period_hours: int = 24) -> dict[str, Any]:
        """
        Get alert statistics for specified period.

        Args:
            period_hours: Period in hours to analyze

        Returns:
            Alert statistics
        """
        cutoff_time = get_current_utc_timestamp() - timedelta(hours=period_hours)
        recent_alerts = [a for a in self._alert_history if a.timestamp >= cutoff_time]

        # Count by severity
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1

        # Count by metric
        metric_counts = defaultdict(int)
        for alert in recent_alerts:
            metric_counts[alert.metric_name] += 1

        # Calculate resolution rate
        resolved_alerts = [a for a in recent_alerts if a.resolved]
        resolution_rate = (len(resolved_alerts) / len(recent_alerts) * 100) if recent_alerts else 0

        # Average time to resolution
        resolution_times = []
        for alert in resolved_alerts:
            if alert.resolved_timestamp:
                resolution_time = (alert.resolved_timestamp - alert.timestamp).total_seconds()
                resolution_times.append(resolution_time)

        avg_resolution_time = (
            sum(resolution_times) / len(resolution_times) if resolution_times else 0
        )

        return {
            "period_hours": period_hours,
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self._active_alerts),
            "severity_breakdown": dict(severity_counts),
            "metric_breakdown": dict(metric_counts),
            "resolution_rate_percent": resolution_rate,
            "avg_resolution_time_minutes": avg_resolution_time / 60,
            "top_metrics": sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        }

    def add_notification_handler(
        self, severity: AlertSeverity, handler: Callable[[AnalyticsAlert], None]
    ) -> None:
        """
        Add notification handler for alerts of specified severity.

        Args:
            severity: Alert severity to handle
            handler: Notification handler function
        """
        self._notification_handlers[severity].append(handler)
        self.logger.info(f"Added notification handler for {severity.value} alerts")

    def configure_escalation(
        self, severity: AlertSeverity, escalation_config: dict[str, Any]
    ) -> None:
        """
        Configure alert escalation rules.

        Args:
            severity: Alert severity
            escalation_config: Escalation configuration
        """
        self._escalation_rules[severity] = escalation_config
        self.logger.info(f"Configured escalation for {severity.value} alerts")

    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            # Portfolio risk alerts
            AlertRule(
                rule_id="portfolio_var_breach",
                metric_name="portfolio_var_95",
                condition="greater_than",
                threshold_value=0.02,  # 2% daily VaR
                severity=AlertSeverity.HIGH,
                description="Portfolio VaR 95% exceeds 2% daily limit",
            ),
            AlertRule(
                rule_id="max_drawdown_breach",
                metric_name="portfolio_max_drawdown",
                condition="greater_than",
                threshold_value=0.10,  # 10% max drawdown
                severity=AlertSeverity.CRITICAL,
                description="Portfolio maximum drawdown exceeds 10%",
            ),
            AlertRule(
                rule_id="position_concentration",
                metric_name="max_position_weight",
                condition="greater_than",
                threshold_value=0.15,  # 15% single position limit
                severity=AlertSeverity.MEDIUM,
                description="Single position concentration exceeds 15%",
            ),
            # Performance alerts
            AlertRule(
                rule_id="low_sharpe_ratio",
                metric_name="portfolio_sharpe_ratio",
                condition="less_than",
                threshold_value=0.5,
                severity=AlertSeverity.LOW,
                description="Portfolio Sharpe ratio below 0.5",
            ),
            AlertRule(
                rule_id="high_volatility",
                metric_name="portfolio_volatility",
                condition="greater_than",
                threshold_value=0.25,  # 25% annual volatility
                severity=AlertSeverity.MEDIUM,
                description="Portfolio volatility exceeds 25% annually",
            ),
            # Operational alerts
            AlertRule(
                rule_id="low_order_fill_rate",
                metric_name="order_fill_rate",
                condition="less_than",
                threshold_value=0.95,  # 95% fill rate
                severity=AlertSeverity.MEDIUM,
                description="Order fill rate below 95%",
            ),
            AlertRule(
                rule_id="high_execution_time",
                metric_name="avg_execution_time_ms",
                condition="greater_than",
                threshold_value=5000,  # 5 seconds
                severity=AlertSeverity.HIGH,
                description="Average order execution time exceeds 5 seconds",
            ),
            AlertRule(
                rule_id="high_error_rate",
                metric_name="system_error_rate",
                condition="greater_than",
                threshold_value=0.05,  # 5% error rate
                severity=AlertSeverity.HIGH,
                description="System error rate exceeds 5%",
            ),
            # System resource alerts
            AlertRule(
                rule_id="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition="greater_than",
                threshold_value=85.0,
                severity=AlertSeverity.HIGH,
                description="CPU usage exceeds 85%",
            ),
            AlertRule(
                rule_id="high_memory_usage",
                metric_name="memory_usage_percent",
                condition="greater_than",
                threshold_value=90.0,
                severity=AlertSeverity.CRITICAL,
                description="Memory usage exceeds 90%",
            ),
        ]

        for rule in default_rules:
            self._alert_rules[rule.rule_id] = rule

    async def _metric_monitoring_loop(self) -> None:
        """Background loop for metric monitoring."""
        while self._running:
            try:
                # Evaluate all metric values against rules
                for metric_name, value in self._metric_values.items():
                    await self._evaluate_metric_rules(metric_name, value)

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metric monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _alert_maintenance_loop(self) -> None:
        """Background loop for alert maintenance."""
        while self._running:
            try:
                await self._cleanup_resolved_alerts()
                await self._update_alert_metrics()

                await asyncio.sleep(300)  # Run every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert maintenance loop: {e}")
                await asyncio.sleep(300)

    async def _escalation_loop(self) -> None:
        """Background loop for alert escalation."""
        while self._running:
            try:
                await self._process_escalations()
                await asyncio.sleep(600)  # Check every 10 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in escalation loop: {e}")
                await asyncio.sleep(600)

    async def _evaluate_metric_rules(self, metric_name: str, value: float) -> None:
        """Evaluate metric value against applicable rules."""
        try:
            applicable_rules = [
                rule
                for rule in self._alert_rules.values()
                if rule.metric_name == metric_name and rule.enabled
            ]

            for rule in applicable_rules:
                should_trigger = False

                # Check condition
                if rule.condition == "greater_than" and value > rule.threshold_value:
                    should_trigger = True
                elif rule.condition == "less_than" and value < rule.threshold_value:
                    should_trigger = True
                elif rule.condition == "equals" and abs(value - rule.threshold_value) < 0.001:
                    should_trigger = True
                elif rule.condition == "not_equals" and abs(value - rule.threshold_value) >= 0.001:
                    should_trigger = True

                if should_trigger:
                    await self._trigger_rule(rule, metric_name, value)

        except Exception as e:
            self.logger.error(f"Error evaluating metric rules for {metric_name}: {e}")

    async def _trigger_rule(self, rule: AlertRule, metric_name: str, value: float) -> None:
        """Trigger an alert rule."""
        try:
            now = get_current_utc_timestamp()

            # Check cooldown period
            if rule.last_triggered and now - rule.last_triggered < timedelta(
                minutes=rule.cooldown_minutes
            ):
                return

            # Check if alert already active
            if rule.rule_id in self._active_alerts:
                return

            # Generate alert
            await self.generate_alert(
                alert_id=rule.rule_id,
                severity=rule.severity,
                title=f"Alert: {rule.description}",
                message=f"Metric {metric_name} value {value:.4f} triggered rule: {rule.description}",
                metric_name=metric_name,
                current_value=value,
                threshold_value=rule.threshold_value,
            )

            rule.last_triggered = now

        except Exception as e:
            self.logger.error(f"Error triggering rule {rule.rule_id}: {e}")

    async def _send_notifications(self, alert: AnalyticsAlert) -> None:
        """Send notifications for alert."""
        try:
            handlers = self._notification_handlers.get(alert.severity, [])

            for handler in handlers:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, handler, alert)
                except Exception as e:
                    self.logger.error(f"Error in notification handler: {e}")

        except Exception as e:
            self.logger.error(f"Error sending notifications for alert {alert.id}: {e}")

    async def _cleanup_resolved_alerts(self) -> None:
        """Clean up old resolved alerts from active list."""
        try:
            cutoff_time = get_current_utc_timestamp() - timedelta(hours=24)

            # Remove old resolved alerts from active list
            expired_alerts = [
                alert_id
                for alert_id, alert in self._active_alerts.items()
                if alert.resolved
                and alert.resolved_timestamp
                and alert.resolved_timestamp < cutoff_time
            ]

            for alert_id in expired_alerts:
                del self._active_alerts[alert_id]

        except Exception as e:
            self.logger.error(f"Error cleaning up resolved alerts: {e}")

    async def _update_alert_metrics(self) -> None:
        """Update alert-related metrics."""
        try:
            # Count active alerts by severity
            severity_counts = defaultdict(int)
            for alert in self._active_alerts.values():
                severity_counts[alert.severity.value] += 1

            # Update metrics
            self.metrics_collector.set_gauge(
                "analytics_active_alerts_total", len(self._active_alerts)
            )

            for severity, count in severity_counts.items():
                self.metrics_collector.set_gauge(
                    "analytics_active_alerts_by_severity", count, labels={"severity": severity}
                )

        except Exception as e:
            self.logger.error(f"Error updating alert metrics: {e}")

    async def _process_escalations(self) -> None:
        """Process alert escalations based on configured rules."""
        try:
            now = get_current_utc_timestamp()

            for alert in self._active_alerts.values():
                if alert.acknowledged:
                    continue

                escalation_config = self._escalation_rules.get(alert.severity)
                if not escalation_config:
                    continue

                # Check escalation time
                escalation_time = escalation_config.get("escalation_time_minutes", 30)
                if (now - alert.timestamp).total_seconds() / 60 >= escalation_time:
                    await self._escalate_alert(alert, escalation_config)

        except Exception as e:
            self.logger.error(f"Error processing escalations: {e}")

    async def _escalate_alert(
        self, alert: AnalyticsAlert, escalation_config: dict[str, Any]
    ) -> None:
        """Escalate an alert according to configuration."""
        try:
            # Mark as escalated
            alert.metadata["escalated"] = True
            alert.metadata["escalated_at"] = get_current_utc_timestamp().isoformat()

            # Send escalation notifications
            escalation_handlers = escalation_config.get("handlers", [])
            for handler in escalation_handlers:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, handler, alert)
                except Exception as e:
                    self.logger.error(f"Error in escalation handler: {e}")

            self.logger.warning(f"Escalated alert: {alert.id}")

        except Exception as e:
            self.logger.error(f"Error escalating alert {alert.id}: {e}")
