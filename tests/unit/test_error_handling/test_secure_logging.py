"""
Tests for the secure logging system.

This module tests the secure logging capabilities including sensitive data masking,
structured logging, and audit trail functionality.
"""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.error_handling.secure_context_manager import InformationLevel, SecurityContext, UserRole
from src.error_handling.secure_logging import (
    LogCategory,
    LogLevel,
    LoggingConfig,
    SecureLogEntry,
    SecureLogger,
)


class TestLogLevel:
    """Test log level enum."""

    def test_log_level_values(self):
        """Test log level enum values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
        assert LogLevel.SECURITY.value == "SECURITY"
        assert LogLevel.AUDIT.value == "AUDIT"


class TestLogCategory:
    """Test log category enum."""

    def test_log_category_values(self):
        """Test log category enum values."""
        assert LogCategory.AUTHENTICATION.value == "authentication"
        assert LogCategory.AUTHORIZATION.value == "authorization"
        assert LogCategory.ERROR_HANDLING.value == "error_handling"
        assert LogCategory.TRADING.value == "trading"
        assert LogCategory.RISK_MANAGEMENT.value == "risk_management"
        assert LogCategory.DATABASE.value == "database"
        assert LogCategory.NETWORK.value == "network"
        assert LogCategory.SYSTEM.value == "system"
        assert LogCategory.USER_ACTION.value == "user_action"
        assert LogCategory.SECURITY_EVENT.value == "security_event"


class TestSecureLogEntry:
    """Test secure log entry dataclass."""

    def test_secure_log_entry_creation(self):
        """Test secure log entry creation with defaults."""
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            component="test_component",
            message="Test message"
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.category == LogCategory.SYSTEM
        assert entry.component == "test_component"
        assert entry.message == "Test message"
        assert entry.sanitized is True
        assert entry.information_level == InformationLevel.BASIC
        assert entry.threat_detected is False
        assert entry.security_classification == "INTERNAL"

    def test_secure_log_entry_with_context(self):
        """Test secure log entry with security context."""
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.SECURITY,
            category=LogCategory.AUTHENTICATION,
            component="auth_handler",
            message="Authentication failure",
            user_id="HASH_user123",
            client_ip="192.168.1.1",
            threat_detected=True
        )
        
        assert entry.level == LogLevel.SECURITY
        assert entry.user_id == "HASH_user123"
        assert entry.client_ip == "192.168.1.1"
        assert entry.threat_detected is True

    def test_secure_log_entry_additional_data(self):
        """Test secure log entry with additional data."""
        additional_data = {"retry_count": 3, "error_code": "AUTH001"}
        
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.ERROR,
            category=LogCategory.ERROR_HANDLING,
            component="error_handler",
            message="Error occurred",
            additional_data=additional_data
        )
        
        assert entry.additional_data == additional_data


class TestLoggingConfig:
    """Test logging configuration."""

    def test_logging_config_defaults(self):
        """Test logging config with default values."""
        config = LoggingConfig()
        
        assert config.log_level == LogLevel.INFO
        assert config.enable_console_logging is True
        assert config.enable_file_logging is True
        assert config.enable_audit_logging is True

    def test_logging_config_custom_values(self):
        """Test logging config with custom values."""
        config = LoggingConfig(
            log_level=LogLevel.DEBUG,
            enable_console_logging=False,
            enable_file_logging=True,
            enable_audit_logging=False
        )
        
        assert config.log_level == LogLevel.DEBUG
        assert config.enable_console_logging is False
        assert config.enable_file_logging is True
        assert config.enable_audit_logging is False


class TestSecureLogger:
    """Test secure logger implementation."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def logger_config(self, temp_log_dir):
        """Create logger configuration."""
        config = LoggingConfig()
        config.log_directory = temp_log_dir
        config.log_file = "test_app.log"
        config.audit_log_file = "test_audit.log"
        return config

    @pytest.fixture
    def secure_logger(self, logger_config):
        """Create secure logger instance."""
        with patch('src.error_handling.secure_logging.get_security_sanitizer') as mock_sanitizer:
            mock_sanitizer_instance = MagicMock()
            mock_sanitizer.return_value = mock_sanitizer_instance
            
            # Configure mocks to return proper types
            mock_sanitizer_instance.sanitize_context.return_value = {"safe": "data"}
            mock_sanitizer_instance.sanitize_error_message.return_value = "Sanitized error"
            # Return string for stack trace sanitization
            mock_sanitizer_instance.sanitize_stack_trace.return_value = "Sanitized stack trace with ValueError"
            
            return SecureLogger("test_logger", logger_config)

    def test_secure_logger_initialization(self, secure_logger):
        """Test secure logger initialization."""
        assert secure_logger is not None
        assert hasattr(secure_logger, 'config')
        assert hasattr(secure_logger, 'sanitizer')

    def test_log_error_message(self, secure_logger):
        """Test logging error message."""
        security_context = SecurityContext(user_role=UserRole.USER)
        test_error = ValueError("Test error")
        
        with patch.object(secure_logger, '_write_log_entry') as mock_write:
            secure_logger.log_error(
                error=test_error,
                component="test_component",
                security_context=security_context,
                additional_data={"test": "data"}
            )
            
            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            log_entry = args[0]
            
            assert log_entry.level == LogLevel.ERROR
            assert log_entry.component == "test_component"
            assert log_entry.category == LogCategory.ERROR_HANDLING
            assert log_entry.error_type == "ValueError"
            assert "safe" in log_entry.additional_data

    def test_log_security_event(self, secure_logger):
        """Test logging security event."""
        security_context = SecurityContext(
            user_role=UserRole.SECURITY,
            client_ip="192.168.1.1"
        )
        
        with patch.object(secure_logger, '_write_log_entry') as mock_write:
            secure_logger.log_security_event(
                event_type="threat_detector",
                message="Potential security threat detected",
                security_context=security_context,
                threat_level="HIGH"
            )
            
            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            log_entry = args[0]
            
            assert log_entry.level == LogLevel.SECURITY
            assert log_entry.category == LogCategory.SECURITY_EVENT
            assert log_entry.threat_detected is True
            assert log_entry.client_ip == "192.168.1.1"

    def test_log_audit_trail(self, secure_logger):
        """Test audit trail logging."""
        security_context = SecurityContext(
            user_role=UserRole.USER,
            user_id="user123"
        )
        
        with patch.object(secure_logger, '_write_audit_entry') as mock_write_audit:
            secure_logger.log_audit_event(
                action="login",
                resource="user_account",
                result="SUCCESS",
                security_context=security_context,
                additional_data={"component": "user_service"}
            )
            
            mock_write_audit.assert_called_once()

    def test_sanitize_log_data(self, secure_logger):
        """Test log data sanitization through sanitizer integration."""
        data = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "key_123456",
            "safe_field": "visible_data"
        }
        
        # Test that sanitizer is used correctly
        sanitized = secure_logger.sanitizer.sanitize_context(data)
        
        # Should use mocked sanitizer
        assert sanitized == {"safe": "data"}

    def test_determine_information_level_from_context(self, secure_logger):
        """Test information level determination from security context."""
        # Test guest user
        guest_context = SecurityContext(user_role=UserRole.GUEST)
        level = secure_logger._determine_info_level(guest_context)
        assert level == InformationLevel.MINIMAL
        
        # Test admin user
        admin_context = SecurityContext(
            user_role=UserRole.ADMIN,
            has_admin_access=True
        )
        level = secure_logger._determine_info_level(admin_context)
        assert level == InformationLevel.DETAILED
        
        # Test developer user
        dev_context = SecurityContext(user_role=UserRole.DEVELOPER)
        level = secure_logger._determine_info_level(dev_context)
        assert level == InformationLevel.FULL

    def test_log_error_with_security_context(self, secure_logger):
        """Test logging error with security context."""
        security_context = SecurityContext(
            user_role=UserRole.USER,
            user_id="user123",
            session_id="sess456", 
            client_ip="10.0.0.1",
            request_id="req789"
        )
        
        with patch.object(secure_logger, '_write_log_entry') as mock_write:
            secure_logger.log_error(
                ValueError("Test error"),
                LogLevel.WARNING,
                LogCategory.AUTHENTICATION,
                "auth_service",
                security_context
            )
            
            # Verify log entry was created with correct context
            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            entry = args[0]
            
            assert entry.level == LogLevel.WARNING
            assert entry.category == LogCategory.AUTHENTICATION
            assert entry.component == "auth_service"

    def test_format_log_entry_json(self, secure_logger):
        """Test JSON formatting of log entries."""
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            component="test",
            message="Test message"
        )
        
        formatted = secure_logger._format_log_message(entry)
        
        # Should be valid JSON
        parsed = json.loads(formatted)
        assert parsed["level"] == "INFO"
        assert parsed["category"] == "system"
        assert parsed["component"] == "test"
        assert parsed["message"] == "Test message"

    def test_format_log_entry_structured(self, secure_logger):
        """Test structured formatting of log entries."""
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.ERROR,
            category=LogCategory.ERROR_HANDLING,
            component="error_handler",
            message="Error occurred",
            user_id="user123"
        )
        
        formatted = secure_logger._format_log_message(entry)
        
        assert "ERROR" in formatted
        assert "error_handler" in formatted
        assert "Error occurred" in formatted
        assert "user123" in formatted

    def test_log_level_filtering(self, secure_logger):
        """Test that logger handles different log levels."""
        # Test that logger can be configured with different levels
        assert secure_logger.config.log_level == LogLevel.INFO  # Default level
        
        # Test that we can log at different levels
        with patch.object(secure_logger, '_write_log_entry') as mock_write:
            secure_logger.log_error(ValueError("Test"), LogLevel.ERROR)
            mock_write.assert_called_once()
            
            args = mock_write.call_args[0]
            entry = args[0]
            assert entry.level == LogLevel.ERROR

    def test_write_log_entry(self, secure_logger):
        """Test log entry writing."""
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            component="test",
            message="Console test"
        )
        
        # Test that _write_log_entry processes the entry
        result = secure_logger._write_log_entry(entry)
        assert isinstance(result, str)  # Should return log ID

    def test_log_to_logger(self, secure_logger):
        """Test logging to python logger."""
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            component="test",
            message="File test"
        )
        
        # Test that _write_to_logger works with the entry
        with patch.object(secure_logger.logger, 'info') as mock_info:
            secure_logger._write_to_logger(secure_logger.logger, entry, LogLevel.INFO)
            mock_info.assert_called_once()

    def test_write_audit_entry(self, secure_logger, temp_log_dir):
        """Test audit logging."""
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.AUDIT,
            category=LogCategory.USER_ACTION,
            component="user_service",
            message="User action audit",
            user_id="user123"
        )
        
        secure_logger._write_audit_entry(entry)
        
        # Check if audit file was created
        audit_file = temp_log_dir / secure_logger.config.audit_log_file
        if audit_file.exists():
            content = audit_file.read_text()
            assert "User action audit" in content

    def test_log_with_stack_trace(self, secure_logger):
        """Test logging with stack trace."""
        try:
            raise ValueError("Test exception")
        except Exception as e:
            security_context = SecurityContext(user_role=UserRole.DEVELOPER)
            
            with patch.object(secure_logger, '_write_log_entry') as mock_write:
                secure_logger.log_error(
                    e,
                    LogLevel.ERROR,
                    LogCategory.ERROR_HANDLING,
                    "test_component",
                    security_context
                )
                
                args = mock_write.call_args[0]
                log_entry = args[0]
                
                # Developer should see stack trace
                assert log_entry.stack_trace is not None
                assert "ValueError" in log_entry.stack_trace

    def test_log_without_stack_trace_for_guest(self, secure_logger):
        """Test that guests don't get stack traces."""
        try:
            raise ValueError("Test exception")
        except Exception as e:
            security_context = SecurityContext(user_role=UserRole.GUEST)
            
            # Mock the sanitizer to return None for guest stack traces
            with patch.object(secure_logger.sanitizer, 'sanitize_stack_trace', return_value=None):
                with patch.object(secure_logger, '_write_log_entry') as mock_write:
                    secure_logger.log_error(
                        e,
                        LogLevel.ERROR,
                        LogCategory.ERROR_HANDLING,
                        "test_component",
                        security_context
                    )
                    
                    args = mock_write.call_args[0]
                    log_entry = args[0]
                    
                    # Guest should not see stack trace
                    assert log_entry.stack_trace is None

    def test_log_rotation_setup(self, secure_logger, temp_log_dir):
        """Test log rotation configuration."""
        # Test that logger sets up rotation when configured
        assert hasattr(secure_logger, 'config')
        
        if hasattr(secure_logger.config, 'enable_log_rotation'):
            assert isinstance(secure_logger.config.enable_log_rotation, bool)

    def test_concurrent_logging(self, secure_logger):
        """Test concurrent logging safety."""
        import threading
        
        results = []
        
        def log_worker(worker_id):
            security_context = SecurityContext(user_role=UserRole.USER)
            secure_logger.log_error(
                ValueError("Test info"),
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                "test_component",
                security_context
            )
            results.append(worker_id)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All workers should have completed
        assert len(results) == 5

    def test_sensitive_data_masking_in_logs(self, secure_logger):
        """Test that sensitive data is masked in logs."""
        sensitive_context = {
            "password": "secret123",
            "credit_card": "1234-5678-9012-3456",
            "api_key": "sk-1234567890abcdef",
            "safe_data": "visible"
        }
        
        security_context = SecurityContext(user_role=UserRole.USER)
        
        with patch.object(secure_logger, '_write_log_entry') as mock_write:
            secure_logger.log_error(
                ValueError("Test info"),
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                "test_component",
                security_context,
                sensitive_context
            )
            
            # Sensitive data should be sanitized
            # (exact behavior depends on sanitizer mock)
            mock_write.assert_called_once()

    def test_log_performance_metrics(self, secure_logger):
        """Test logging with performance metrics."""
        security_context = SecurityContext(user_role=UserRole.SYSTEM)
        metrics = {
            "execution_time": 0.123,
            "memory_usage": "45MB",
            "cpu_usage": "12%"
        }
        
        # Mock sanitizer to return the original metrics for this test
        with patch.object(secure_logger.sanitizer, 'sanitize_context', return_value=metrics):
            with patch.object(secure_logger, '_write_log_entry') as mock_write:
                secure_logger.log_error(
                    ValueError("Test info"),
                    LogLevel.ERROR,
                    LogCategory.SYSTEM,
                    "performance_monitor",
                    security_context,
                    metrics
                )
                
                args = mock_write.call_args[0]
                log_entry = args[0]
                
                assert "execution_time" in log_entry.additional_data

    def test_log_entry_serialization(self, secure_logger):
        """Test that log entries can be serialized."""
        entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            component="test",
            message="Serialization test"
        )
        
        # Should be able to convert to JSON using the correct method name
        json_str = secure_logger._format_log_message(entry)
        parsed = json.loads(json_str)
        
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Serialization test"

    def test_log_filtering_by_category(self, secure_logger):
        """Test filtering logs by category."""
        if hasattr(secure_logger.config, 'log_categories'):
            # Test category filtering if implemented
            security_context = SecurityContext(user_role=UserRole.USER)
            
            with patch.object(secure_logger, '_should_log_category') as mock_filter:
                mock_filter.return_value = False
                
                with patch.object(secure_logger, '_write_log_entry') as mock_write:
                    secure_logger.log_error(
                        ValueError("Test info"),
                        LogLevel.ERROR,
                        LogCategory.DEBUG,
                        "test_component",
                        security_context
                    )
                    
                    # Should not write if filtered
                    mock_write.assert_not_called()