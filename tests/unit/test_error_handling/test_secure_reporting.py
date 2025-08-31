"""
Tests for the secure error reporting system.

This module tests the secure error reporting capabilities including role-based
access control, multiple reporting channels, and alert generation.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.error_handling.secure_context_manager import (
    InformationLevel,
    SecureErrorReport,
    SecurityContext,
    UserRole,
)
from src.error_handling.secure_reporting import (
    AlertSeverity,
    ErrorAlert,
    ReportingChannel,
    ReportingMetrics,
    ReportingRule,
    SecureErrorReporter,
)


class TestReportingChannel:
    """Test reporting channel enum."""

    def test_reporting_channel_values(self):
        """Test reporting channel enum values."""
        assert ReportingChannel.LOG.value == "log"
        assert ReportingChannel.DATABASE.value == "database"
        assert ReportingChannel.ALERT.value == "alert"
        assert ReportingChannel.EMAIL.value == "email"
        assert ReportingChannel.WEBHOOK.value == "webhook"
        assert ReportingChannel.METRICS.value == "metrics"
        assert ReportingChannel.AUDIT.value == "audit"


class TestReportingRule:
    """Test reporting rule dataclass."""

    def test_reporting_rule_creation(self):
        """Test reporting rule creation."""
        rule = ReportingRule(
            name="test_rule",
            condition="error_type == 'DatabaseError'",
            channels=[ReportingChannel.LOG, ReportingChannel.ALERT],
            min_user_role=UserRole.ADMIN,
            alert_severity=AlertSeverity.HIGH
        )
        
        assert rule.name == "test_rule"
        assert rule.condition == "error_type == 'DatabaseError'"
        assert ReportingChannel.LOG in rule.channels
        assert ReportingChannel.ALERT in rule.channels
        assert rule.min_user_role == UserRole.ADMIN
        assert rule.alert_severity == AlertSeverity.HIGH
        assert rule.enabled is True
        assert rule.rate_limit is None

    def test_reporting_rule_with_rate_limit(self):
        """Test reporting rule with rate limiting."""
        rule = ReportingRule(
            name="rate_limited_rule",
            condition="severity == 'low'",
            channels=[ReportingChannel.LOG],
            min_user_role=UserRole.USER,
            alert_severity=AlertSeverity.LOW,
            rate_limit=10,
            enabled=False
        )
        
        assert rule.rate_limit == 10
        assert rule.enabled is False


class TestErrorAlert:
    """Test error alert dataclass."""

    def test_error_alert_creation(self):
        """Test error alert creation."""
        alert = ErrorAlert(
            alert_id="ALERT001",
            error_id="ERR001",
            severity=AlertSeverity.CRITICAL,
            title="Critical Database Error",
            message="Database connection failed",
            timestamp=datetime.now(timezone.utc),
            component="database_service",
            user_id="user123"
        )
        
        assert alert.alert_id == "ALERT001"
        assert alert.error_id == "ERR001"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.title == "Critical Database Error"
        assert alert.message == "Database connection failed"
        assert alert.component == "database_service"
        assert alert.user_id == "user123"
        assert isinstance(alert.metadata, dict)

    def test_error_alert_with_metadata(self):
        """Test error alert with metadata."""
        metadata = {"retry_count": 3, "last_error": "Connection timeout"}
        
        alert = ErrorAlert(
            alert_id="ALERT002",
            error_id="ERR002",
            severity=AlertSeverity.HIGH,
            title="Network Error",
            message="Network timeout occurred",
            timestamp=datetime.now(timezone.utc),
            metadata=metadata
        )
        
        assert alert.metadata == metadata


class TestReportingMetrics:
    """Test reporting metrics dataclass."""

    def test_reporting_metrics_creation(self):
        """Test reporting metrics creation with defaults."""
        metrics = ReportingMetrics()
        
        assert metrics.total_reports == 0
        assert isinstance(metrics.reports_by_role, dict)
        assert isinstance(metrics.reports_by_channel, dict)
        assert metrics.blocked_reports == 0
        assert isinstance(metrics.alert_counts, dict)
        assert isinstance(metrics.last_reset, datetime)

    def test_reporting_metrics_with_data(self):
        """Test reporting metrics with custom data."""
        reports_by_role = {"admin": 10, "user": 5}
        reports_by_channel = {"log": 12, "alert": 3}
        alert_counts = {"high": 2, "low": 1}
        
        metrics = ReportingMetrics(
            total_reports=15,
            reports_by_role=reports_by_role,
            reports_by_channel=reports_by_channel,
            blocked_reports=2,
            alert_counts=alert_counts
        )
        
        assert metrics.total_reports == 15
        assert metrics.reports_by_role == reports_by_role
        assert metrics.reports_by_channel == reports_by_channel
        assert metrics.blocked_reports == 2
        assert metrics.alert_counts == alert_counts


class TestSecureErrorReporter:
    """Test secure error reporter implementation."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with patch('src.error_handling.secure_reporting.get_secure_context_manager') as mock_context_mgr, \
             patch('src.error_handling.secure_reporting.get_security_rate_limiter') as mock_rate_limiter, \
             patch('src.error_handling.secure_reporting.get_security_sanitizer') as mock_sanitizer:
            
            mock_context_mgr.return_value = MagicMock()
            mock_rate_limiter.return_value = MagicMock()
            mock_sanitizer.return_value = MagicMock()
            
            yield {
                'context_mgr': mock_context_mgr,
                'rate_limiter': mock_rate_limiter,
                'sanitizer': mock_sanitizer
            }

    @pytest.fixture
    def reporter(self, mock_dependencies):
        """Create reporter instance with mocked dependencies."""
        return SecureErrorReporter()

    def test_reporter_initialization(self, reporter):
        """Test reporter initialization."""
        assert reporter is not None
        assert hasattr(reporter, 'logger')
        assert hasattr(reporter, 'context_manager')
        assert hasattr(reporter, 'rate_limiter')
        assert hasattr(reporter, 'sanitizer')

    def test_default_reporting_rules(self, reporter):
        """Test that default reporting rules are configured."""
        assert hasattr(reporter, 'reporting_rules')
        assert len(reporter.reporting_rules) > 0
        
        # Check that rules have required attributes
        for rule in reporter.reporting_rules:
            assert hasattr(rule, 'name')
            assert hasattr(rule, 'channels')
            assert hasattr(rule, 'min_user_role')

    def test_create_error_report(self, reporter):
        """Test error report creation."""
        error = Exception("Test error")
        security_context = SecurityContext(user_role=UserRole.USER)
        error_context = {"component": "test_service"}
        
        with patch.object(reporter.context_manager, 'create_secure_report') as mock_create:
            mock_report = SecureErrorReport(
                error_id="TEST001",
                timestamp=datetime.now(timezone.utc),
                user_message="Test message",
                technical_message="Test technical message"
            )
            mock_create.return_value = mock_report
            
            report = reporter.create_error_report(error, security_context, error_context)
            
            assert report == mock_report
            mock_create.assert_called_once_with(error, security_context, error_context)

    @pytest.mark.asyncio
    async def test_submit_error_report_basic(self, reporter):
        """Test basic error report submission."""
        report = SecureErrorReport(
            error_id="TEST001",
            timestamp=datetime.now(timezone.utc),
            user_message="Test message",
            technical_message="Test technical message"
        )
        security_context = SecurityContext(user_role=UserRole.USER)
        
        # Mock rate limiter to allow request
        reporter.rate_limiter.check_rate_limit = AsyncMock(return_value=MagicMock(allowed=True))
        
        with patch.object(reporter, '_route_report') as mock_route, \
             patch.object(reporter, '_update_metrics') as mock_metrics:
            
            mock_route.return_value = True
            
            result = await reporter.submit_error_report(report, security_context)
            
            assert result is True
            mock_route.assert_called_once()
            mock_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_error_report_with_rate_limiting(self, reporter):
        """Test error report submission with rate limiting."""
        report = SecureErrorReport(
            error_id="TEST001",
            timestamp=datetime.now(timezone.utc),
            user_message="Test message",
            technical_message="Test technical message"
        )
        security_context = SecurityContext(user_role=UserRole.USER)
        
        # Mock rate limiter to deny request
        reporter.rate_limiter.check_rate_limit = AsyncMock(return_value=MagicMock(allowed=False))
        
        with patch.object(reporter, '_update_metrics') as mock_metrics:
            result = await reporter.submit_error_report(report, security_context)
            
            assert result is False
            # Should still update metrics for blocked reports
            mock_metrics.assert_called()

    def test_evaluate_reporting_rules(self, reporter):
        """Test evaluation of reporting rules."""
        report = SecureErrorReport(
            error_id="TEST001",
            timestamp=datetime.now(timezone.utc),
            user_message="Database error",
            technical_message="Database connection failed",
            component="database_service"
        )
        security_context = SecurityContext(user_role=UserRole.ADMIN)
        
        # Create a test rule
        test_rule = ReportingRule(
            name="database_rule",
            condition="'database' in user_message.lower()",
            channels=[ReportingChannel.LOG, ReportingChannel.ALERT],
            min_user_role=UserRole.USER,
            alert_severity=AlertSeverity.HIGH
        )
        
        # Test rule evaluation
        matches = reporter._evaluate_rule_condition(test_rule.condition, report, security_context)
        assert matches is True

    def test_filter_by_user_role(self, reporter):
        """Test filtering of reporting rules by user role."""
        rules = [
            ReportingRule(
                name="guest_rule",
                condition="True",
                channels=[ReportingChannel.LOG],
                min_user_role=UserRole.GUEST,
                alert_severity=AlertSeverity.LOW
            ),
            ReportingRule(
                name="admin_rule", 
                condition="True",
                channels=[ReportingChannel.ALERT],
                min_user_role=UserRole.ADMIN,
                alert_severity=AlertSeverity.HIGH
            )
        ]
        
        # Test with user role
        user_context = SecurityContext(user_role=UserRole.USER)
        filtered = reporter._filter_rules_by_role(rules, user_context)
        
        # User should see guest rule but not admin rule
        rule_names = [rule.name for rule in filtered]
        assert "guest_rule" in rule_names
        assert "admin_rule" not in rule_names
        
        # Test with admin role
        admin_context = SecurityContext(user_role=UserRole.ADMIN)
        filtered = reporter._filter_rules_by_role(rules, admin_context)
        
        # Admin should see both rules
        rule_names = [rule.name for rule in filtered]
        assert "guest_rule" in rule_names
        assert "admin_rule" in rule_names

    @pytest.mark.asyncio
    async def test_generate_alert(self, reporter):
        """Test alert generation from error report."""
        report = SecureErrorReport(
            error_id="TEST001",
            timestamp=datetime.now(timezone.utc),
            user_message="Critical error",
            technical_message="Critical system error occurred",
            component="critical_service"
        )
        
        rule = ReportingRule(
            name="critical_rule",
            condition="True",
            channels=[ReportingChannel.ALERT],
            min_user_role=UserRole.USER,
            alert_severity=AlertSeverity.CRITICAL
        )
        
        alert = await reporter._generate_alert(report, rule)
        
        assert alert.error_id == report.error_id
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.component == report.component
        assert len(alert.alert_id) > 0
        assert isinstance(alert.timestamp, datetime)

    def test_get_reporting_metrics(self, reporter):
        """Test retrieval of reporting metrics."""
        # Initialize some test metrics
        reporter.metrics.total_reports = 10
        reporter.metrics.reports_by_role["admin"] = 5
        reporter.metrics.reports_by_channel["log"] = 8
        
        metrics = reporter.get_reporting_metrics()
        
        assert metrics.total_reports == 10
        assert metrics.reports_by_role["admin"] == 5
        assert metrics.reports_by_channel["log"] == 8

    def test_reset_metrics(self, reporter):
        """Test metrics reset functionality."""
        # Set some metrics
        reporter.metrics.total_reports = 10
        reporter.metrics.blocked_reports = 2
        reporter.metrics.reports_by_role["admin"] = 5
        
        reporter.reset_metrics()
        
        assert reporter.metrics.total_reports == 0
        assert reporter.metrics.blocked_reports == 0
        assert len(reporter.metrics.reports_by_role) == 0
        assert isinstance(reporter.metrics.last_reset, datetime)

    def test_add_reporting_rule(self, reporter):
        """Test adding custom reporting rules."""
        initial_count = len(reporter.reporting_rules)
        
        new_rule = ReportingRule(
            name="custom_rule",
            condition="component == 'test_service'",
            channels=[ReportingChannel.EMAIL],
            min_user_role=UserRole.DEVELOPER,
            alert_severity=AlertSeverity.MEDIUM
        )
        
        reporter.add_reporting_rule(new_rule)
        
        assert len(reporter.reporting_rules) == initial_count + 1
        assert new_rule in reporter.reporting_rules

    def test_remove_reporting_rule(self, reporter):
        """Test removing reporting rules."""
        # Add a test rule first
        test_rule = ReportingRule(
            name="removable_rule",
            condition="True",
            channels=[ReportingChannel.LOG],
            min_user_role=UserRole.USER,
            alert_severity=AlertSeverity.LOW
        )
        
        reporter.add_reporting_rule(test_rule)
        initial_count = len(reporter.reporting_rules)
        
        # Remove the rule
        removed = reporter.remove_reporting_rule("removable_rule")
        
        assert removed is True
        assert len(reporter.reporting_rules) == initial_count - 1
        
        # Try to remove non-existent rule
        removed = reporter.remove_reporting_rule("non_existent_rule")
        assert removed is False

    @pytest.mark.asyncio
    async def test_route_report_to_multiple_channels(self, reporter):
        """Test routing report to multiple channels."""
        report = SecureErrorReport(
            error_id="TEST001",
            timestamp=datetime.now(timezone.utc),
            user_message="Test message",
            technical_message="Test technical message"
        )
        
        rule = ReportingRule(
            name="multi_channel_rule",
            condition="True",
            channels=[ReportingChannel.LOG, ReportingChannel.DATABASE, ReportingChannel.ALERT],
            min_user_role=UserRole.USER,
            alert_severity=AlertSeverity.MEDIUM
        )
        
        with patch.object(reporter, '_send_to_log') as mock_log, \
             patch.object(reporter, '_send_to_database') as mock_db, \
             patch.object(reporter, '_send_to_alert') as mock_alert:
            
            mock_log.return_value = True
            mock_db.return_value = True  
            mock_alert.return_value = True
            
            success = await reporter._route_to_channels(report, rule)
            
            assert success is True
            mock_log.assert_called_once()
            mock_db.assert_called_once() 
            mock_alert.assert_called_once()

    def test_validate_reporting_rule(self, reporter):
        """Test reporting rule validation."""
        # Valid rule
        valid_rule = ReportingRule(
            name="valid_rule",
            condition="error_id is not None",
            channels=[ReportingChannel.LOG],
            min_user_role=UserRole.USER,
            alert_severity=AlertSeverity.LOW
        )
        
        assert reporter._validate_reporting_rule(valid_rule) is True
        
        # Invalid rule - empty name
        invalid_rule = ReportingRule(
            name="",
            condition="True",
            channels=[ReportingChannel.LOG],
            min_user_role=UserRole.USER,
            alert_severity=AlertSeverity.LOW
        )
        
        assert reporter._validate_reporting_rule(invalid_rule) is False

    def test_get_applicable_rules(self, reporter):
        """Test getting applicable rules for a report and context."""
        report = SecureErrorReport(
            error_id="TEST001",
            timestamp=datetime.now(timezone.utc),
            user_message="Database connection failed",
            technical_message="Database connection timed out",
            component="database_service"
        )
        security_context = SecurityContext(user_role=UserRole.ADMIN)
        
        # Should return rules that match the context
        applicable_rules = reporter._get_applicable_rules(report, security_context)
        
        assert isinstance(applicable_rules, list)
        # All rules should have min_user_role <= ADMIN
        for rule in applicable_rules:
            admin_level = UserRole.ADMIN
            rule_level = rule.min_user_role
            # Admin should be able to see rules for admin and below
            assert rule_level.value in [UserRole.GUEST.value, UserRole.USER.value, UserRole.ADMIN.value]

    @pytest.mark.asyncio
    async def test_error_report_full_workflow(self, reporter):
        """Test complete error reporting workflow."""
        error = Exception("Critical database failure")
        security_context = SecurityContext(
            user_role=UserRole.ADMIN,
            user_id="admin123",
            component="database_service"
        )
        error_context = {"query": "SELECT * FROM users", "timeout": 30}
        
        with patch.object(reporter, 'create_error_report') as mock_create, \
             patch.object(reporter, 'submit_error_report') as mock_submit:
            
            mock_report = SecureErrorReport(
                error_id="TEST001",
                timestamp=datetime.now(timezone.utc),
                user_message="Service temporarily unavailable",
                technical_message="Database connection timeout"
            )
            mock_create.return_value = mock_report
            mock_submit.return_value = True
            
            # Create and submit report
            report = reporter.create_error_report(error, security_context, error_context)
            success = await reporter.submit_error_report(report, security_context)
            
            assert report == mock_report
            assert success is True
            mock_create.assert_called_once_with(error, security_context, error_context)
            mock_submit.assert_called_once_with(mock_report, security_context)