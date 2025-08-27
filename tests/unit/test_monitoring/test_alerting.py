"""
Unit tests for alerting system.

Tests the comprehensive alerting infrastructure including:
- Alert rule management
- Alert firing and resolution
- Notification channels
- Escalation policies
- Alert deduplication
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
import json

from src.monitoring.alerting import (
    AlertManager,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationConfig,
    EscalationPolicy,
    get_alert_manager,
    set_global_alert_manager,
    load_alert_rules_from_file
)
from src.core.exceptions import MonitoringError


class TestAlert:
    """Test Alert data structure."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test alert message",
            labels={"service": "test"},
            annotations={"description": "Test description"},
            starts_at=datetime.now(timezone.utc)
        )
        
        assert alert.rule_name == "test_rule"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.status == AlertStatus.FIRING
        assert alert.message == "Test alert message"
        assert alert.labels == {"service": "test"}
        assert alert.fingerprint != ""  # Should auto-generate fingerprint
    
    def test_alert_fingerprint_generation(self):
        """Test alert fingerprint generation for deduplication."""
        alert1 = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test message",
            labels={"service": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        alert2 = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Different message",  # Different message but same rule/labels
            labels={"service": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        # Same rule name and labels should produce same fingerprint
        assert alert1.fingerprint == alert2.fingerprint
    
    def test_alert_is_active(self):
        """Test alert active status check."""
        firing_alert = Alert(
            rule_name="test",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test",
            labels={},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        resolved_alert = Alert(
            rule_name="test",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.RESOLVED,
            message="Test",
            labels={},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        assert firing_alert.is_active is True
        assert resolved_alert.is_active is False
    
    def test_alert_duration(self):
        """Test alert duration calculation."""
        start_time = datetime.now(timezone.utc)
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test",
            labels={},
            annotations={},
            starts_at=start_time,
            ends_at=start_time + timedelta(minutes=5)
        )
        
        duration = alert.duration
        assert duration.total_seconds() == 300  # 5 minutes


class TestAlertRule:
    """Test AlertRule data structure."""
    
    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            name="high_cpu_usage",
            description="Alert when CPU usage is high",
            severity=AlertSeverity.MEDIUM,
            query="cpu_usage > 80",
            threshold=80.0,
            operator=">",
            duration="5m"
        )
        
        assert rule.name == "high_cpu_usage"
        assert rule.severity == AlertSeverity.MEDIUM
        assert rule.threshold == 80.0
        assert rule.enabled is True
    
    def test_alert_rule_default_channels(self):
        """Test alert rule default notification channels."""
        critical_rule = AlertRule(
            name="critical_test",
            description="Critical test",
            severity=AlertSeverity.CRITICAL,
            query="test > 0",
            threshold=0.0,
            operator=">",
            duration="1m"
        )
        
        medium_rule = AlertRule(
            name="medium_test",
            description="Medium test",
            severity=AlertSeverity.MEDIUM,
            query="test > 0",
            threshold=0.0,
            operator=">",
            duration="1m"
        )
        
        # Critical/High should have email + slack by default
        assert NotificationChannel.EMAIL in critical_rule.notification_channels
        assert NotificationChannel.SLACK in critical_rule.notification_channels
        
        # Medium/Low should have slack only by default
        assert NotificationChannel.SLACK in medium_rule.notification_channels
        assert NotificationChannel.EMAIL not in medium_rule.notification_channels


class TestNotificationConfig:
    """Test NotificationConfig."""
    
    def test_notification_config_creation(self):
        """Test creating notification configuration."""
        config = NotificationConfig(
            email_smtp_server="smtp.gmail.com",
            email_smtp_port=587,
            email_username="test@example.com",
            email_password="password",
            email_from="alerts@example.com",
            email_to=["admin@example.com"],
            slack_webhook_url="https://hooks.slack.com/services/test",
            webhook_urls=["https://webhook.example.com"]
        )
        
        assert config.email_smtp_server == "smtp.gmail.com"
        assert config.email_to == ["admin@example.com"]
        assert config.slack_webhook_url == "https://hooks.slack.com/services/test"
        assert config.webhook_urls == ["https://webhook.example.com"]


class TestAlertManager:
    """Test AlertManager functionality."""
    
    @pytest.fixture
    def notification_config(self):
        """Create test notification configuration."""
        return NotificationConfig(
            email_from="test@example.com",
            email_to=["admin@example.com"],
            slack_webhook_url="https://hooks.slack.com/test",
            webhook_urls=["https://webhook.example.com"]
        )
    
    @pytest.fixture
    def alert_manager(self, notification_config):
        """Create test alert manager."""
        return AlertManager(notification_config)
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert alert_manager.config is not None
        assert alert_manager._rules == {}
        assert alert_manager._active_alerts == {}
        assert len(alert_manager._alert_history) == 0
        assert alert_manager._running is False
    
    def test_add_rule(self, alert_manager):
        """Test adding alert rules."""
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            severity=AlertSeverity.HIGH,
            query="test > 0",
            threshold=0.0,
            operator=">",
            duration="1m"
        )
        
        alert_manager.add_rule(rule)
        assert "test_rule" in alert_manager._rules
        assert alert_manager._rules["test_rule"] == rule
    
    def test_remove_rule(self, alert_manager):
        """Test removing alert rules."""
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            severity=AlertSeverity.HIGH,
            query="test > 0",
            threshold=0.0,
            operator=">",
            duration="1m"
        )
        
        alert_manager.add_rule(rule)
        assert alert_manager.remove_rule("test_rule") is True
        assert "test_rule" not in alert_manager._rules
        assert alert_manager.remove_rule("nonexistent") is False
    
    def test_add_escalation_policy(self, alert_manager):
        """Test adding escalation policies."""
        policy = EscalationPolicy(
            name="test_policy",
            description="Test escalation",
            severity_levels=[AlertSeverity.CRITICAL],
            escalation_rules=[{"delay": "15m", "channels": ["email"]}]
        )
        
        alert_manager.add_escalation_policy(policy)
        assert "test_policy" in alert_manager._escalation_policies
    
    def test_add_suppression_rule(self, alert_manager):
        """Test adding suppression rules."""
        rule = {"labels": {"environment": "test"}, "duration": "1h"}
        
        alert_manager.add_suppression_rule(rule)
        assert rule in alert_manager._suppression_rules
    
    @pytest.mark.asyncio
    async def test_fire_alert(self, alert_manager):
        """Test firing an alert."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test alert",
            labels={"service": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        # Mock the notification queue to avoid actual notifications
        alert_manager._notification_queue = AsyncMock()
        
        await alert_manager.fire_alert(alert)
        
        assert alert.fingerprint in alert_manager._active_alerts
        assert alert in alert_manager._alert_history
        assert alert_manager._alerts_fired == 1
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test alert",
            labels={"service": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        # Fire the alert first
        alert_manager._active_alerts[alert.fingerprint] = alert
        
        await alert_manager.resolve_alert(alert.fingerprint)
        
        assert alert.fingerprint not in alert_manager._active_alerts
        assert alert.status == AlertStatus.RESOLVED
        assert alert.ends_at is not None
        assert alert_manager._alerts_resolved == 1
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging an alert."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test alert",
            labels={"service": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        alert_manager._active_alerts[alert.fingerprint] = alert
        
        success = await alert_manager.acknowledge_alert(alert.fingerprint, "admin")
        
        assert success is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledgment_by == "admin"
        assert alert.acknowledgment_at is not None
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        alert1 = Alert(
            rule_name="test1",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test 1",
            labels={},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        alert2 = Alert(
            rule_name="test2",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Test 2",
            labels={},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        alert_manager._active_alerts[alert1.fingerprint] = alert1
        alert_manager._active_alerts[alert2.fingerprint] = alert2
        
        # Get all active alerts
        all_alerts = alert_manager.get_active_alerts()
        assert len(all_alerts) == 2
        
        # Get alerts by severity
        critical_alerts = alert_manager.get_active_alerts(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_get_alert_stats(self, alert_manager):
        """Test getting alert statistics."""
        # Add some test data
        alert_manager._alerts_fired = 10
        alert_manager._alerts_resolved = 8
        alert_manager._notifications_sent = 15
        
        stats = alert_manager.get_alert_stats()
        
        assert stats["total_fired"] == 10
        assert stats["total_resolved"] == 8
        assert stats["notifications_sent"] == 15
        assert "active_alerts" in stats
        assert "active_by_severity" in stats
    
    @pytest.mark.asyncio
    async def test_start_stop(self, alert_manager):
        """Test starting and stopping alert manager."""
        await alert_manager.start()
        assert alert_manager._running is True
        assert alert_manager._background_task is not None
        
        await alert_manager.stop()
        assert alert_manager._running is False
        assert alert_manager._background_task is None


class TestNotifications:
    """Test notification functionality."""
    
    @pytest.fixture
    def alert_manager_with_mock_queue(self):
        """Create alert manager with mocked notification queue."""
        config = NotificationConfig(
            email_from="test@example.com",
            email_to=["admin@example.com"],
            slack_webhook_url="https://hooks.slack.com/test",
            webhook_urls=["https://webhook.example.com/test"]
        )
        manager = AlertManager(config)
        manager._notification_queue = AsyncMock()
        return manager
    
    @pytest.mark.asyncio
    @patch('src.monitoring.alerting.smtplib.SMTP')
    async def test_send_email_notification(self, mock_smtp, alert_manager_with_mock_queue):
        """Test sending email notifications."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Critical test alert",
            labels={"service": "test"},
            annotations={"description": "Test description"},
            starts_at=datetime.now(timezone.utc)
        )
        
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        await alert_manager_with_mock_queue._send_email_notification(alert)
        
        # Verify SMTP interactions
        mock_smtp.assert_called_once()
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_send_slack_notification(self, mock_post, alert_manager_with_mock_queue):
        """Test sending Slack notifications."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="High priority alert",
            labels={"service": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await alert_manager_with_mock_queue._send_slack_notification(alert)
        
        # Verify Slack webhook was called
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_send_webhook_notification(self, mock_post, alert_manager_with_mock_queue):
        """Test sending webhook notifications."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.FIRING,
            message="Medium priority alert",
            labels={"service": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await alert_manager_with_mock_queue._send_webhook_notification(alert)
        
        # Verify webhook was called
        mock_post.assert_called_once()


class TestEscalation:
    """Test alert escalation functionality."""
    
    @pytest.fixture
    def alert_manager_with_escalation(self):
        """Create alert manager with escalation setup."""
        config = NotificationConfig()
        manager = AlertManager(config)
        
        # Add test rule with escalation
        rule = AlertRule(
            name="escalation_test",
            description="Test escalation",
            severity=AlertSeverity.HIGH,
            query="test > 0",
            threshold=0.0,
            operator=">",
            duration="1m",
            escalation_delay="5m"
        )
        manager.add_rule(rule)
        
        return manager
    
    @pytest.mark.asyncio
    async def test_escalation_trigger(self, alert_manager_with_escalation):
        """Test alert escalation triggering."""
        # Create alert that should escalate
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        alert = Alert(
            rule_name="escalation_test",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test escalation",
            labels={},
            annotations={},
            starts_at=old_time
        )
        
        alert_manager_with_escalation._active_alerts[alert.fingerprint] = alert
        alert_manager_with_escalation._notification_queue = AsyncMock()
        
        # Check escalations
        await alert_manager_with_escalation._check_escalations()
        
        assert alert.escalated is True
        assert alert.escalation_count == 1


class TestSuppression:
    """Test alert suppression functionality."""
    
    @pytest.fixture
    def alert_manager_with_suppression(self):
        """Create alert manager with suppression rules."""
        config = NotificationConfig()
        manager = AlertManager(config)
        
        # Add suppression rule
        manager.add_suppression_rule({
            "labels": {"environment": "test"},
            "duration": "1h"
        })
        
        return manager
    
    @pytest.mark.asyncio
    async def test_alert_suppression(self, alert_manager_with_suppression):
        """Test alert suppression."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Test alert",
            labels={"environment": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        await alert_manager_with_suppression.fire_alert(alert)
        
        assert alert.status == AlertStatus.SUPPRESSED
        assert alert.fingerprint not in alert_manager_with_suppression._active_alerts


class TestGlobalFunctions:
    """Test global utility functions."""
    
    def test_global_alert_manager(self):
        """Test global alert manager functions."""
        config = NotificationConfig()
        manager = AlertManager(config)
        
        set_global_alert_manager(manager)
        retrieved_manager = get_alert_manager()
        
        assert retrieved_manager is manager
    
    @patch('builtins.open')
    @patch('src.monitoring.alerting.yaml.safe_load')
    def test_load_alert_rules_from_file(self, mock_yaml_load, mock_open):
        """Test loading alert rules from YAML file."""
        mock_yaml_data = {
            'rules': [
                {
                    'name': 'test_rule',
                    'description': 'Test rule',
                    'severity': 'high',
                    'query': 'test > 0',
                    'threshold': 0.0,
                    'operator': '>',
                    'duration': '1m',
                    'notification_channels': ['email', 'slack']
                }
            ]
        }
        
        mock_yaml_load.return_value = mock_yaml_data
        mock_open.return_value.__enter__.return_value = Mock()
        
        rules = load_alert_rules_from_file('test.yml')
        
        assert len(rules) == 1
        assert rules[0].name == 'test_rule'
        assert rules[0].severity == AlertSeverity.HIGH
        assert NotificationChannel.EMAIL in rules[0].notification_channels
        assert NotificationChannel.SLACK in rules[0].notification_channels


class TestAlertingIntegration:
    """Integration tests for alerting system."""
    
    @pytest.mark.asyncio
    async def test_full_alert_lifecycle(self):
        """Test complete alert lifecycle."""
        config = NotificationConfig(
            email_from="test@example.com",
            email_to=["admin@example.com"]
        )
        manager = AlertManager(config)
        
        # Add alert rule
        rule = AlertRule(
            name="integration_test",
            description="Integration test rule",
            severity=AlertSeverity.HIGH,
            query="test > 0",
            threshold=0.0,
            operator=">",
            duration="1m"
        )
        manager.add_rule(rule)
        
        # Create and fire alert
        alert = Alert(
            rule_name="integration_test",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            message="Integration test alert",
            labels={"service": "test"},
            annotations={},
            starts_at=datetime.now(timezone.utc)
        )
        
        # Mock notification queue
        manager._notification_queue = AsyncMock()
        
        # Fire alert
        await manager.fire_alert(alert)
        assert len(manager.get_active_alerts()) == 1
        
        # Acknowledge alert
        success = await manager.acknowledge_alert(alert.fingerprint, "admin")
        assert success is True
        
        # Resolve alert
        await manager.resolve_alert(alert.fingerprint)
        assert len(manager.get_active_alerts()) == 0
        
        # Check statistics
        stats = manager.get_alert_stats()
        assert stats["total_fired"] == 1
        assert stats["total_resolved"] == 1