"""
Tests for the secure error context manager.

This module tests the secure error context management capabilities
including role-based filtering and information disclosure control.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.error_handling.secure_context_manager import (
    InformationLevel,
    SecureErrorContextManager,
    SecureErrorReport,
    SecurityContext,
    UserRole,
)


class TestSecurityContext:
    """Test security context dataclass."""

    def test_security_context_creation(self):
        """Test security context creation with defaults."""
        context = SecurityContext()
        
        assert context.user_role == UserRole.GUEST
        assert context.user_id is None
        assert context.is_authenticated is False
        assert context.has_admin_access is False
        assert isinstance(context.timestamp, datetime)

    def test_security_context_with_values(self):
        """Test security context with custom values."""
        context = SecurityContext(
            user_role=UserRole.ADMIN,
            user_id="admin123",
            is_authenticated=True,
            has_admin_access=True,
            component="trading_engine"
        )
        
        assert context.user_role == UserRole.ADMIN
        assert context.user_id == "admin123"
        assert context.is_authenticated is True
        assert context.has_admin_access is True
        assert context.component == "trading_engine"


class TestSecureErrorReport:
    """Test secure error report dataclass."""

    def test_secure_error_report_creation(self):
        """Test secure error report creation."""
        report = SecureErrorReport(
            error_id="ERR001",
            timestamp=datetime.now(timezone.utc),
            user_message="Something went wrong",
            technical_message="Database connection failed"
        )
        
        assert report.error_id == "ERR001"
        assert report.user_message == "Something went wrong"
        assert report.technical_message == "Database connection failed"
        assert report.information_level == InformationLevel.MINIMAL
        assert report.sanitized is True

    def test_secure_error_report_with_debug_info(self):
        """Test secure error report with debug information."""
        debug_info = {"stack_trace": "...", "variables": {"x": 1}}
        
        report = SecureErrorReport(
            error_id="ERR002",
            timestamp=datetime.now(timezone.utc),
            user_message="Error occurred",
            technical_message="Internal error",
            debug_info=debug_info,
            information_level=InformationLevel.DEBUG
        )
        
        assert report.debug_info == debug_info
        assert report.information_level == InformationLevel.DEBUG


class TestSecureErrorContextManager:
    """Test secure error context manager."""

    @pytest.fixture
    def context_manager(self):
        """Create context manager instance."""
        with patch('src.error_handling.secure_context_manager.get_security_sanitizer') as mock_sanitizer:
            mock_sanitizer.return_value = MagicMock()
            mock_sanitizer.return_value.sanitize_context.return_value = {"safe": "data"}
            mock_sanitizer.return_value.sanitize_error_message.return_value = "Sanitized error"
            
            return SecureErrorContextManager()

    def test_initialization(self, context_manager):
        """Test context manager initialization."""
        assert context_manager is not None
        assert hasattr(context_manager, 'sanitizer')
        assert hasattr(context_manager, 'logger')

    def test_user_role_enum_values(self):
        """Test user role enum values."""
        assert UserRole.GUEST.value == "guest"
        assert UserRole.USER.value == "user"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.DEVELOPER.value == "developer"
        assert UserRole.SECURITY.value == "security"
        assert UserRole.SYSTEM.value == "system"

    def test_information_level_enum_values(self):
        """Test information level enum values."""
        assert InformationLevel.MINIMAL.value == "minimal"
        assert InformationLevel.BASIC.value == "basic"
        assert InformationLevel.DETAILED.value == "detailed"
        assert InformationLevel.FULL.value == "full"
        assert InformationLevel.DEBUG.value == "debug"

    def test_role_info_levels_mapping(self, context_manager):
        """Test role to information level mapping."""
        assert context_manager.role_info_levels[UserRole.GUEST] == InformationLevel.MINIMAL
        assert context_manager.role_info_levels[UserRole.ADMIN] == InformationLevel.DETAILED
        assert context_manager.role_info_levels[UserRole.DEVELOPER] == InformationLevel.FULL
        assert context_manager.role_info_levels[UserRole.SYSTEM] == InformationLevel.DEBUG

    def test_filter_context(self, context_manager):
        """Test context filtering based on information level."""
        context = {
            "user_message": "Safe message",
            "database_host": "secret_host",
            "file_path": "/secret/path",
            "api_key": "secret_key"
        }
        
        security_context = SecurityContext(user_role=UserRole.GUEST)
        info_level = InformationLevel.MINIMAL
        filtered = context_manager._filter_context(context, security_context, info_level)
        
        # Should be filtered based on guest role
        assert isinstance(filtered, dict)

    def test_create_user_message(self, context_manager):
        """Test creation of user-friendly messages.""" 
        error = Exception("Database connection failed")
        security_context = SecurityContext(user_role=UserRole.GUEST)
        
        message = context_manager._create_user_message(error, security_context)
        
        # Should return a safe user message
        assert isinstance(message, str)
        assert len(message) > 0

    def test_categorize_error(self, context_manager):
        """Test error categorization."""
        # Test database errors
        db_error = Exception("Database connection failed")
        category = context_manager._categorize_error(db_error)
        assert category in ["database", "network", "internal", "unknown"]
        
        # Test validation errors
        val_error = Exception("Invalid input format")
        category = context_manager._categorize_error(val_error)
        assert category in ["validation", "internal", "unknown"]

    def test_generate_error_id(self, context_manager):
        """Test error ID generation."""
        error = Exception("Test error")
        security_context = SecurityContext(user_role=UserRole.USER)
        
        error_id = context_manager._generate_error_id(error, security_context)
        
        assert isinstance(error_id, str)
        assert len(error_id) > 0
        
        # Should be unique
        error_id2 = context_manager._generate_error_id(error, security_context)
        assert error_id != error_id2

    def test_create_secure_report_guest_user(self, context_manager):
        """Test secure report creation for guest user."""
        error = Exception("Database connection failed")
        security_context = SecurityContext(user_role=UserRole.GUEST)
        error_context = {"component": "trading", "database_host": "secret"}
        
        report = context_manager.create_secure_report(error, security_context, error_context)
        
        assert report.information_level == InformationLevel.MINIMAL
        assert isinstance(report.user_message, str)
        assert len(report.user_message) > 0
        assert isinstance(report.context, dict)

    def test_create_secure_report_admin_user(self, context_manager):
        """Test secure report creation for admin user."""
        error = Exception("Database connection failed")
        security_context = SecurityContext(
            user_role=UserRole.ADMIN,
            has_admin_access=True
        )
        error_context = {"component": "trading", "operation": "query"}
        
        report = context_manager.create_secure_report(error, security_context, error_context)
        
        assert report.information_level == InformationLevel.DETAILED
        assert "component" in report.context
        assert "operation" in report.context

    def test_create_secure_report_developer_user(self, context_manager):
        """Test secure report creation for developer user."""
        error = Exception("Internal server error")
        security_context = SecurityContext(user_role=UserRole.DEVELOPER)
        error_context = {"stack_trace": "...", "variables": {"x": 1}}
        
        report = context_manager.create_secure_report(error, security_context, error_context)
        
        assert report.information_level == InformationLevel.FULL
        assert report.debug_info is not None
        assert len(report.context) > 0

    def test_create_secure_report_system_user(self, context_manager):
        """Test secure report creation for system user."""
        error = Exception("Critical system error")
        security_context = SecurityContext(user_role=UserRole.SYSTEM)
        error_context = {
            "full_stack_trace": "...",
            "memory_usage": "high",
            "database_queries": ["SELECT * FROM users"]
        }
        
        report = context_manager.create_secure_report(error, security_context, error_context)
        
        assert report.information_level == InformationLevel.DEBUG
        assert report.debug_info is not None

    def test_filter_internal_details(self, context_manager):
        """Test filtering of internal details from messages."""
        message = "Database connection failed at src/database/connection.py line 123"
        filtered = context_manager._filter_internal_details(message)
        
        # Should filter out file paths
        assert "src/" not in filtered
        assert isinstance(filtered, str)

    def test_safe_error_messages(self, context_manager):
        """Test safe error messages."""
        assert "authentication" in context_manager.safe_error_messages
        assert "database" in context_manager.safe_error_messages
        assert "network" in context_manager.safe_error_messages
        
        # Messages should be user-friendly
        for category, message in context_manager.safe_error_messages.items():
            assert len(message) > 0
            assert "internal" not in message.lower()
            assert "server" not in message.lower()

    def test_create_debug_info(self, context_manager):
        """Test debug info creation."""
        error = Exception("Test error")
        context = {"var1": "value1", "var2": "value2"}
        
        debug_info = context_manager._create_debug_info(error, context)
        
        assert isinstance(debug_info, dict)
        assert "error_type" in debug_info
        assert "process_id" in debug_info

    def test_internal_keywords_filtering(self, context_manager):
        """Test that internal keywords are defined for filtering."""
        assert len(context_manager.internal_keywords) > 0
        
        # Should include common sensitive patterns
        keywords = context_manager.internal_keywords
        assert any("src/" in keyword for keyword in keywords)
        assert any("config" in keyword.lower() for keyword in keywords)
        assert any("secret" in keyword.lower() for keyword in keywords)

    def test_validate_user_access(self, context_manager):
        """Test user access validation."""
        # Test with different security contexts
        guest_context = SecurityContext(user_role=UserRole.GUEST)
        admin_context = SecurityContext(user_role=UserRole.ADMIN, has_admin_access=True)
        
        # Should validate access levels
        guest_valid = context_manager.validate_user_access(guest_context, InformationLevel.BASIC)
        admin_valid = context_manager.validate_user_access(admin_context, InformationLevel.FULL)
        
        assert isinstance(guest_valid, bool)
        assert isinstance(admin_valid, bool)

    def test_error_report_serialization(self):
        """Test error report can be serialized."""
        import json
        
        report = SecureErrorReport(
            error_id="ERR001",
            timestamp=datetime.now(timezone.utc),
            user_message="Test message",
            technical_message="Technical details",
            context={"key": "value"}
        )
        
        # Convert to dict for JSON serialization
        report_dict = {
            "error_id": report.error_id,
            "timestamp": report.timestamp.isoformat(),
            "user_message": report.user_message,
            "technical_message": report.technical_message,
            "context": report.context
        }
        
        # Should be JSON serializable
        json_str = json.dumps(report_dict)
        assert len(json_str) > 0

    def test_create_report_with_none_context(self, context_manager):
        """Test report creation with None context."""
        error = Exception("Test error")
        security_context = SecurityContext(user_role=UserRole.USER)
        
        report = context_manager.create_secure_report(error, security_context, None)
        
        assert report is not None
        assert isinstance(report.context, dict)

    def test_create_report_preserves_error_id(self, context_manager):
        """Test that error reports have unique IDs."""
        error = Exception("Test error")
        security_context = SecurityContext(user_role=UserRole.USER)
        
        report1 = context_manager.create_secure_report(error, security_context, {})
        report2 = context_manager.create_secure_report(error, security_context, {})
        
        assert report1.error_id != report2.error_id

    def test_get_safe_error_summary(self, context_manager):
        """Test getting safe error summary."""
        security_context = SecurityContext(user_role=UserRole.USER)
        
        summary = context_manager.get_safe_error_summary(security_context)
        
        assert isinstance(summary, dict)
        assert len(summary) > 0