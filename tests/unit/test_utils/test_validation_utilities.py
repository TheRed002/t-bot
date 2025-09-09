"""
Tests for Validation Utilities.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.utils.validation_utilities import (
    ValidationResult,
    ErrorType,
    RecoveryStatus,
    AuditEventType,
    StateValidationError,
    ValidationWarning,
    ValidationResultData,
    AuditEntry,
    classify_error_type,
    create_audit_entry
)


class TestEnumerations:
    """Test enumeration classes."""

    def test_validation_result_enum(self):
        """Test ValidationResult enum values."""
        assert ValidationResult.PASSED.value == "passed"
        assert ValidationResult.WARNING.value == "warning"
        assert ValidationResult.FAILED.value == "failed"

    def test_error_type_enum(self):
        """Test ErrorType enum values."""
        assert ErrorType.DATABASE_CONNECTION.value == "database_connection"
        assert ErrorType.DATABASE_INTEGRITY.value == "database_integrity"
        assert ErrorType.DATABASE_TIMEOUT.value == "database_timeout"
        assert ErrorType.REDIS_CONNECTION.value == "redis_connection"
        assert ErrorType.REDIS_TIMEOUT.value == "redis_timeout"
        assert ErrorType.DATA_CORRUPTION.value == "data_corruption"
        assert ErrorType.DISK_SPACE.value == "disk_space"
        assert ErrorType.PERMISSION.value == "permission"
        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.CONCURRENCY.value == "concurrency"
        assert ErrorType.UNKNOWN.value == "unknown"

    def test_recovery_status_enum(self):
        """Test RecoveryStatus enum values."""
        assert RecoveryStatus.PENDING.value == "pending"
        assert RecoveryStatus.IN_PROGRESS.value == "in_progress"
        assert RecoveryStatus.COMPLETED.value == "completed"
        assert RecoveryStatus.FAILED.value == "failed"
        assert RecoveryStatus.PARTIAL.value == "partial"

    def test_audit_event_type_enum(self):
        """Test AuditEventType enum values."""
        assert AuditEventType.STATE_CREATED.value == "state_created"
        assert AuditEventType.STATE_UPDATED.value == "state_updated"
        assert AuditEventType.STATE_DELETED.value == "state_deleted"
        assert AuditEventType.STATE_RECOVERED.value == "state_recovered"
        assert AuditEventType.STATE_ROLLBACK.value == "state_rollback"
        assert AuditEventType.VALIDATION_FAILED.value == "validation_failed"
        assert AuditEventType.CORRUPTION_DETECTED.value == "corruption_detected"
        assert AuditEventType.RECOVERY_INITIATED.value == "recovery_initiated"
        assert AuditEventType.SNAPSHOT_CREATED.value == "snapshot_created"
        assert AuditEventType.SNAPSHOT_RESTORED.value == "snapshot_restored"


class TestStateValidationError:
    """Test StateValidationError dataclass."""

    def test_state_validation_error_defaults(self):
        """Test StateValidationError with default values."""
        error = StateValidationError()
        
        assert error.error_id is not None
        assert isinstance(error.error_id, str)
        assert isinstance(error.timestamp, datetime)
        assert error.rule_name == ""
        assert error.field_name == ""
        assert error.error_message == ""
        assert error.severity == "error"
        assert error.state_type == ""
        assert error.state_id == ""
        assert error.actual_value is None
        assert error.expected_value is None
        assert error.rule_config == {}
        assert error.error_code == ""

    def test_state_validation_error_with_values(self):
        """Test StateValidationError with custom values."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rule_config = {"max_length": 100, "required": True}
        
        error = StateValidationError(
            error_id="test-error-id",
            timestamp=custom_time,
            rule_name="length_validation",
            field_name="username",
            error_message="Username too long",
            severity="warning",
            state_type="user_state",
            state_id="user123",
            actual_value="very_long_username",
            expected_value="short_name",
            rule_config=rule_config,
            error_code="VAL001"
        )
        
        assert error.error_id == "test-error-id"
        assert error.timestamp == custom_time
        assert error.rule_name == "length_validation"
        assert error.field_name == "username"
        assert error.error_message == "Username too long"
        assert error.severity == "warning"
        assert error.state_type == "user_state"
        assert error.state_id == "user123"
        assert error.actual_value == "very_long_username"
        assert error.expected_value == "short_name"
        assert error.rule_config == rule_config
        assert error.error_code == "VAL001"

    def test_state_validation_error_uuid_generation(self):
        """Test unique ID generation."""
        error1 = StateValidationError()
        error2 = StateValidationError()
        
        assert error1.error_id != error2.error_id
        assert len(error1.error_id) > 0
        assert len(error2.error_id) > 0


class TestValidationWarning:
    """Test ValidationWarning dataclass."""

    def test_validation_warning_defaults(self):
        """Test ValidationWarning with default values."""
        warning = ValidationWarning()
        
        assert warning.warning_id is not None
        assert isinstance(warning.warning_id, str)
        assert isinstance(warning.timestamp, datetime)
        assert warning.rule_name == ""
        assert warning.field_name == ""
        assert warning.warning_message == ""
        assert warning.severity == "warning"
        assert warning.state_type == ""
        assert warning.state_id == ""
        assert warning.actual_value is None
        assert warning.recommended_value is None

    def test_validation_warning_with_values(self):
        """Test ValidationWarning with custom values."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        warning = ValidationWarning(
            warning_id="test-warning-id",
            timestamp=custom_time,
            rule_name="performance_check",
            field_name="response_time",
            warning_message="Response time exceeds recommended threshold",
            severity="info",
            state_type="api_state",
            state_id="api123",
            actual_value=1500,
            recommended_value=1000
        )
        
        assert warning.warning_id == "test-warning-id"
        assert warning.timestamp == custom_time
        assert warning.rule_name == "performance_check"
        assert warning.field_name == "response_time"
        assert warning.warning_message == "Response time exceeds recommended threshold"
        assert warning.severity == "info"
        assert warning.state_type == "api_state"
        assert warning.state_id == "api123"
        assert warning.actual_value == 1500
        assert warning.recommended_value == 1000

    def test_validation_warning_uuid_generation(self):
        """Test unique ID generation."""
        warning1 = ValidationWarning()
        warning2 = ValidationWarning()
        
        assert warning1.warning_id != warning2.warning_id


class TestValidationResultData:
    """Test ValidationResultData dataclass."""

    def test_validation_result_data_defaults(self):
        """Test ValidationResultData with default values."""
        result = ValidationResultData()
        
        assert result.result_id is not None
        assert isinstance(result.result_id, str)
        assert isinstance(result.timestamp, datetime)
        assert result.is_valid is True
        assert result.overall_result == ValidationResult.PASSED
        assert result.errors == []
        assert result.warnings == []
        assert result.validation_duration_ms == 0.0
        assert result.rules_executed == 0
        assert result.rules_passed == 0
        assert result.rules_failed == 0
        assert result.state_type == ""
        assert result.state_id == ""
        assert result.validation_level == "standard"

    def test_validation_result_data_with_values(self):
        """Test ValidationResultData with custom values."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        error = StateValidationError(rule_name="test_rule")
        warning = ValidationWarning(rule_name="test_warning")
        
        result = ValidationResultData(
            result_id="test-result-id",
            timestamp=custom_time,
            is_valid=False,
            overall_result=ValidationResult.FAILED,
            errors=[error],
            warnings=[warning],
            validation_duration_ms=150.5,
            rules_executed=10,
            rules_passed=8,
            rules_failed=2,
            state_type="trading_state",
            state_id="trade123",
            validation_level="strict"
        )
        
        assert result.result_id == "test-result-id"
        assert result.timestamp == custom_time
        assert result.is_valid is False
        assert result.overall_result == ValidationResult.FAILED
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert result.validation_duration_ms == 150.5
        assert result.rules_executed == 10
        assert result.rules_passed == 8
        assert result.rules_failed == 2
        assert result.state_type == "trading_state"
        assert result.state_id == "trade123"
        assert result.validation_level == "strict"

    def test_validation_result_data_list_operations(self):
        """Test list operations with errors and warnings."""
        result = ValidationResultData()
        
        error1 = StateValidationError(rule_name="rule1")
        error2 = StateValidationError(rule_name="rule2")
        warning1 = ValidationWarning(rule_name="warning1")
        
        result.errors.extend([error1, error2])
        result.warnings.append(warning1)
        
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.errors[0].rule_name == "rule1"
        assert result.errors[1].rule_name == "rule2"
        assert result.warnings[0].rule_name == "warning1"


class TestAuditEntry:
    """Test AuditEntry dataclass."""

    def test_audit_entry_defaults(self):
        """Test AuditEntry with default values."""
        entry = AuditEntry()
        
        assert entry.audit_id is not None
        assert isinstance(entry.audit_id, str)
        assert isinstance(entry.timestamp, datetime)
        assert entry.event_type == AuditEventType.STATE_UPDATED
        assert entry.state_type == ""
        assert entry.state_id == ""
        assert entry.old_value is None
        assert entry.new_value is None
        assert entry.changed_fields == set()
        assert entry.user_id is None
        assert entry.session_id is None
        assert entry.source_component == ""
        assert entry.correlation_id is None
        assert entry.reason == ""
        assert entry.version == 1
        assert entry.checksum_before == ""
        assert entry.checksum_after == ""
        assert entry.validation_status == "unknown"
        assert entry.validation_errors == []

    def test_audit_entry_with_values(self):
        """Test AuditEntry with custom values."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        old_value = {"name": "old_name", "value": 100}
        new_value = {"name": "new_name", "value": 200}
        changed_fields = {"name", "value"}
        validation_errors = ["Error 1", "Error 2"]
        
        entry = AuditEntry(
            audit_id="test-audit-id",
            timestamp=custom_time,
            event_type=AuditEventType.STATE_CREATED,
            state_type="portfolio",
            state_id="port123",
            old_value=old_value,
            new_value=new_value,
            changed_fields=changed_fields,
            user_id="user456",
            session_id="session789",
            source_component="trading_service",
            correlation_id="corr123",
            reason="User requested update",
            version=2,
            checksum_before="abc123",
            checksum_after="def456",
            validation_status="passed",
            validation_errors=validation_errors
        )
        
        assert entry.audit_id == "test-audit-id"
        assert entry.timestamp == custom_time
        assert entry.event_type == AuditEventType.STATE_CREATED
        assert entry.state_type == "portfolio"
        assert entry.state_id == "port123"
        assert entry.old_value == old_value
        assert entry.new_value == new_value
        assert entry.changed_fields == changed_fields
        assert entry.user_id == "user456"
        assert entry.session_id == "session789"
        assert entry.source_component == "trading_service"
        assert entry.correlation_id == "corr123"
        assert entry.reason == "User requested update"
        assert entry.version == 2
        assert entry.checksum_before == "abc123"
        assert entry.checksum_after == "def456"
        assert entry.validation_status == "passed"
        assert entry.validation_errors == validation_errors

    def test_audit_entry_changed_fields_set_operations(self):
        """Test set operations with changed_fields."""
        entry = AuditEntry()
        
        entry.changed_fields.add("field1")
        entry.changed_fields.add("field2")
        entry.changed_fields.add("field1")  # Duplicate, should not add
        
        assert len(entry.changed_fields) == 2
        assert "field1" in entry.changed_fields
        assert "field2" in entry.changed_fields


class TestClassifyErrorType:
    """Test error type classification function."""

    def test_classify_connection_error(self):
        """Test classification of connection errors."""
        error = ConnectionError("Database connection failed")
        result = classify_error_type(error)
        assert result == ErrorType.DATABASE_CONNECTION

    def test_classify_redis_connection_error(self):
        """Test classification of Redis connection errors."""
        error = ConnectionError("Redis connection timeout")
        result = classify_error_type(error)
        assert result == ErrorType.REDIS_CONNECTION

    def test_classify_os_error(self):
        """Test classification of OS errors."""
        error = OSError("Network is unreachable")
        result = classify_error_type(error)
        assert result == ErrorType.DATABASE_CONNECTION

    def test_classify_integrity_error(self):
        """Test classification of integrity errors."""
        # Mock SQLAlchemy IntegrityError
        from sqlalchemy.exc import IntegrityError
        error = IntegrityError("Foreign key constraint failed", None, None)
        result = classify_error_type(error)
        assert result == ErrorType.DATABASE_INTEGRITY

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        # Test asyncio timeout
        error = asyncio.TimeoutError("Operation timed out")
        result = classify_error_type(error)
        assert result == ErrorType.DATABASE_TIMEOUT
        
        # Test SQL timeout using actual SQLAlchemy TimeoutError
        from sqlalchemy.exc import TimeoutError as SQLTimeoutError
        error = SQLTimeoutError("Query timeout", None, None)
        result = classify_error_type(error)
        assert result == ErrorType.DATABASE_TIMEOUT

    def test_classify_operational_error_disk_space(self):
        """Test classification of operational error - disk space."""
        from sqlalchemy.exc import OperationalError
        error = OperationalError("No space left on disk", None, None)
        result = classify_error_type(error)
        assert result == ErrorType.DISK_SPACE

    def test_classify_operational_error_permission(self):
        """Test classification of operational error - permission."""
        from sqlalchemy.exc import OperationalError
        error = OperationalError("Access denied to database", None, None)
        result = classify_error_type(error)
        assert result == ErrorType.PERMISSION

    def test_classify_operational_error_generic(self):
        """Test classification of generic operational error."""
        from sqlalchemy.exc import OperationalError
        error = OperationalError("Database operation failed", None, None)
        result = classify_error_type(error)
        assert result == ErrorType.DATABASE_CONNECTION

    def test_classify_value_error(self):
        """Test classification of value errors."""
        error = ValueError("Invalid input value")
        result = classify_error_type(error)
        assert result == ErrorType.VALIDATION

    def test_classify_type_error(self):
        """Test classification of type errors."""
        error = TypeError("Expected string, got int")
        result = classify_error_type(error)
        assert result == ErrorType.VALIDATION

    def test_classify_json_decode_error(self):
        """Test classification of JSON decode errors."""
        error = json.JSONDecodeError("Invalid JSON", "doc", 0)
        result = classify_error_type(error)
        assert result == ErrorType.VALIDATION

    def test_classify_concurrency_error(self):
        """Test classification of concurrency errors."""
        error = Exception("Concurrent access detected")
        result = classify_error_type(error)
        assert result == ErrorType.CONCURRENCY

    def test_classify_lock_error(self):
        """Test classification of lock errors."""
        error = Exception("Database lock timeout")
        result = classify_error_type(error)
        assert result == ErrorType.CONCURRENCY

    def test_classify_corruption_error(self):
        """Test classification of corruption errors."""
        error = Exception("Data corruption detected in table")
        result = classify_error_type(error)
        assert result == ErrorType.DATA_CORRUPTION

    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        error = Exception("Some random error message")
        result = classify_error_type(error)
        assert result == ErrorType.UNKNOWN

    def test_classify_custom_exception(self):
        """Test classification of custom exception types."""
        class CustomError(Exception):
            pass
            
        error = CustomError("Custom error occurred")
        result = classify_error_type(error)
        assert result == ErrorType.UNKNOWN


class TestCreateAuditEntry:
    """Test audit entry creation function."""

    @patch('src.utils.checksum_utilities.calculate_state_checksum')
    def test_create_audit_entry_basic(self, mock_checksum):
        """Test basic audit entry creation."""
        mock_checksum.side_effect = lambda x: f"checksum_{hash(str(x))}" if x else ""
        
        entry = create_audit_entry(
            event_type=AuditEventType.STATE_CREATED,
            state_type="user",
            state_id="user123"
        )
        
        assert entry.event_type == AuditEventType.STATE_CREATED
        assert entry.state_type == "user"
        assert entry.state_id == "user123"
        assert entry.old_value is None
        assert entry.new_value is None
        assert entry.changed_fields == set()
        assert entry.source_component == ""
        assert entry.reason == ""
        assert entry.user_id is None
        assert isinstance(entry.audit_id, str)
        assert isinstance(entry.timestamp, datetime)

    @patch('src.utils.checksum_utilities.calculate_state_checksum')
    def test_create_audit_entry_with_values(self, mock_checksum):
        """Test audit entry creation with old and new values."""
        mock_checksum.side_effect = lambda x: f"checksum_{hash(str(x))}" if x else ""
        
        old_value = {"name": "John", "age": 30}
        new_value = {"name": "John", "age": 31, "city": "NYC"}
        
        entry = create_audit_entry(
            event_type=AuditEventType.STATE_UPDATED,
            state_type="user",
            state_id="user123",
            old_value=old_value,
            new_value=new_value,
            source_component="user_service",
            reason="Age update",
            user_id="admin"
        )
        
        assert entry.event_type == AuditEventType.STATE_UPDATED
        assert entry.old_value == old_value
        assert entry.new_value == new_value
        assert "age" in entry.changed_fields  # Modified
        assert "city" in entry.changed_fields  # Added
        assert entry.source_component == "user_service"
        assert entry.reason == "Age update"
        assert entry.user_id == "admin"
        # Check that checksums were calculated using the mocked function
        expected_checksum_before = f"checksum_{hash(str(old_value))}"
        expected_checksum_after = f"checksum_{hash(str(new_value))}"
        assert entry.checksum_before == expected_checksum_before
        assert entry.checksum_after == expected_checksum_after

    @patch('src.utils.checksum_utilities.calculate_state_checksum')
    def test_create_audit_entry_field_deletion(self, mock_checksum):
        """Test audit entry creation with field deletions."""
        mock_checksum.side_effect = lambda x: f"checksum_{hash(str(x))}" if x else ""
        
        old_value = {"name": "John", "age": 30, "temp_field": "remove_me"}
        new_value = {"name": "John", "age": 31}
        
        entry = create_audit_entry(
            event_type=AuditEventType.STATE_UPDATED,
            state_type="user",
            state_id="user123",
            old_value=old_value,
            new_value=new_value
        )
        
        assert "age" in entry.changed_fields  # Modified
        assert "temp_field" in entry.changed_fields  # Deleted

    @patch('src.utils.checksum_utilities.calculate_state_checksum')
    def test_create_audit_entry_only_new_value(self, mock_checksum):
        """Test audit entry creation with only new value."""
        mock_checksum.side_effect = lambda x: f"checksum_{hash(str(x))}" if x else ""
        
        new_value = {"name": "John", "age": 30}
        
        entry = create_audit_entry(
            event_type=AuditEventType.STATE_CREATED,
            state_type="user",
            state_id="user123",
            new_value=new_value
        )
        
        assert entry.changed_fields == {"name", "age"}
        assert entry.checksum_before == ""
        expected_checksum_after = f"checksum_{hash(str(new_value))}"
        assert entry.checksum_after == expected_checksum_after

    @patch('src.utils.checksum_utilities.calculate_state_checksum')
    def test_create_audit_entry_only_old_value(self, mock_checksum):
        """Test audit entry creation with only old value."""
        mock_checksum.side_effect = lambda x: f"checksum_{hash(str(x))}" if x else ""
        
        old_value = {"name": "John", "age": 30}
        
        entry = create_audit_entry(
            event_type=AuditEventType.STATE_DELETED,
            state_type="user",
            state_id="user123",
            old_value=old_value
        )
        
        assert entry.changed_fields == {"name", "age"}
        expected_checksum_before = f"checksum_{hash(str(old_value))}"
        assert entry.checksum_before == expected_checksum_before
        assert entry.checksum_after == ""

    @patch('src.utils.checksum_utilities.calculate_state_checksum')
    def test_create_audit_entry_no_changes(self, mock_checksum):
        """Test audit entry creation with no value changes."""
        mock_checksum.side_effect = lambda x: f"checksum_{hash(str(x))}" if x else ""
        
        entry = create_audit_entry(
            event_type=AuditEventType.SNAPSHOT_CREATED,
            state_type="system",
            state_id="sys123"
        )
        
        assert entry.changed_fields == set()
        assert entry.checksum_before == ""
        assert entry.checksum_after == ""

    @patch('src.utils.checksum_utilities.calculate_state_checksum')
    def test_create_audit_entry_checksum_calls(self, mock_checksum):
        """Test that checksum calculation is called correctly."""
        mock_checksum.return_value = "test_checksum"
        
        old_value = {"key": "old"}
        new_value = {"key": "new"}
        
        entry = create_audit_entry(
            event_type=AuditEventType.STATE_UPDATED,
            state_type="test",
            state_id="test123",
            old_value=old_value,
            new_value=new_value
        )
        
        # Should be called once for old_value and once for new_value
        assert mock_checksum.call_count == 2
        mock_checksum.assert_any_call(old_value)
        mock_checksum.assert_any_call(new_value)


class TestValidationUtilitiesEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_state_values(self):
        """Test handling of empty state values."""
        error = StateValidationError(
            actual_value={},
            expected_value=[]
        )
        
        assert error.actual_value == {}
        assert error.expected_value == []

    def test_none_state_values(self):
        """Test handling of None state values."""
        warning = ValidationWarning(
            actual_value=None,
            recommended_value=None
        )
        
        assert warning.actual_value is None
        assert warning.recommended_value is None

    def test_complex_state_values(self):
        """Test handling of complex nested state values."""
        complex_value = {
            "nested": {
                "dict": {"key": "value"},
                "list": [1, 2, 3],
                "tuple": (4, 5, 6)
            },
            "simple": "string"
        }
        
        result = ValidationResultData()
        error = StateValidationError(actual_value=complex_value)
        result.errors.append(error)
        
        assert result.errors[0].actual_value == complex_value

    @patch('src.utils.checksum_utilities.calculate_state_checksum')
    def test_create_audit_entry_identical_values(self, mock_checksum):
        """Test audit entry creation with identical old and new values."""
        mock_checksum.return_value = "same_checksum"
        
        same_value = {"key": "unchanged"}
        
        entry = create_audit_entry(
            event_type=AuditEventType.STATE_UPDATED,
            state_type="test",
            state_id="test123",
            old_value=same_value,
            new_value=same_value
        )
        
        # No fields should be marked as changed
        assert entry.changed_fields == set()

    def test_classify_error_with_none_message(self):
        """Test error classification with None message."""
        class ErrorWithNoneStr(Exception):
            def __str__(self):
                return None
                
        error = ErrorWithNoneStr()
        result = classify_error_type(error)
        # Should handle None gracefully and return UNKNOWN
        assert result == ErrorType.UNKNOWN