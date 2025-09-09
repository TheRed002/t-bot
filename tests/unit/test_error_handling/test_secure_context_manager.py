"""
Tests for the secure error context manager.

This module tests the secure error context management capabilities
including role-based filtering and information disclosure control.
"""

from datetime import datetime, timezone

import pytest

from src.error_handling.secure_context_manager import (
    InformationLevel,
    SecureErrorContextManager,
    SecureErrorReport,
    SecurityContext,
    UserRole,
    create_secure_context,
    secure_context,
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
            component="trading_engine",
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
            message="Something went wrong",
            details={"error_code": "ERR001", "technical": "Database connection failed"}
        )

        assert report.message == "Something went wrong"
        assert report.details["error_code"] == "ERR001"
        assert report.details["technical"] == "Database connection failed"
        assert isinstance(report.timestamp, datetime)

    def test_secure_error_report_with_debug_info(self):
        """Test secure error report with debug information."""
        debug_info = {"stack_trace": "...", "variables": {"x": 1}}

        report = SecureErrorReport(
            message="Error occurred",
            details={"debug_info": debug_info, "technical": "Internal error"}
        )

        assert report.details["debug_info"] == debug_info
        assert report.message == "Error occurred"


class TestSecureErrorContextManager:
    """Test secure error context manager."""

    @pytest.fixture
    def context_manager(self):
        """Create context manager instance."""
        return SecureErrorContextManager()

    def test_initialization(self, context_manager):
        """Test context manager initialization."""
        assert context_manager is not None
        assert hasattr(context_manager, "security_context")

    def test_user_role_enum_values(self):
        """Test user role enum values."""
        assert UserRole.GUEST.value == "guest"
        assert UserRole.USER.value == "user"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.SYSTEM.value == "system"

    def test_information_level_enum_values(self):
        """Test information level enum values."""
        assert InformationLevel.PUBLIC.value == "public"
        assert InformationLevel.INTERNAL.value == "internal"
        assert InformationLevel.CONFIDENTIAL.value == "confidential"
        assert InformationLevel.SECRET.value == "secret"

    def test_create_secure_report(self, context_manager):
        """Test creating secure reports."""
        error = ValueError("Test error")
        report = context_manager.create_secure_report(error, context="test")
        
        assert isinstance(report, SecureErrorReport)
        assert report.message == "Test error"
        assert "context" in report.details

    @pytest.mark.asyncio
    async def test_secure_context_manager(self):
        """Test secure context manager."""
        async with secure_context("test_operation", user="test") as context:
            assert "user" in context
            assert context["user"] == "test"

    def test_create_secure_context(self):
        """Test creating secure context dict."""
        context = create_secure_context(user="test", operation="login")
        assert context["user"] == "test"
        assert context["operation"] == "login"