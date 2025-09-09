"""
Centralized validation utilities for state management.

This module provides common validation data structures, enums, and utilities
used across the state management system to eliminate duplication.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ValidationResult(Enum):
    """Standard validation result enumeration."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ErrorType(Enum):
    """Standard error type classification."""

    DATABASE_CONNECTION = "database_connection"
    DATABASE_INTEGRITY = "database_integrity"
    DATABASE_TIMEOUT = "database_timeout"
    REDIS_CONNECTION = "redis_connection"
    REDIS_TIMEOUT = "redis_timeout"
    DATA_CORRUPTION = "data_corruption"
    DISK_SPACE = "disk_space"
    PERMISSION = "permission"
    VALIDATION = "validation"
    CONCURRENCY = "concurrency"
    UNKNOWN = "unknown"


class RecoveryStatus(Enum):
    """Standard recovery operation status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class AuditEventType(Enum):
    """Standard audit event types."""

    STATE_CREATED = "state_created"
    STATE_UPDATED = "state_updated"
    STATE_DELETED = "state_deleted"
    STATE_RECOVERED = "state_recovered"
    STATE_ROLLBACK = "state_rollback"
    VALIDATION_FAILED = "validation_failed"
    CORRUPTION_DETECTED = "corruption_detected"
    RECOVERY_INITIATED = "recovery_initiated"
    SNAPSHOT_CREATED = "snapshot_created"
    SNAPSHOT_RESTORED = "snapshot_restored"


@dataclass
class StateValidationError:
    """Standard state validation error structure."""

    error_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Error details
    rule_name: str = ""
    field_name: str = ""
    error_message: str = ""
    severity: str = "error"  # error, warning, info

    # Context
    state_type: str = ""
    state_id: str = ""
    actual_value: Any = None
    expected_value: Any = None

    # Metadata
    rule_config: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""


@dataclass
class ValidationWarning:
    """Standard validation warning structure."""

    warning_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Warning details
    rule_name: str = ""
    field_name: str = ""
    warning_message: str = ""
    severity: str = "warning"

    # Context
    state_type: str = ""
    state_id: str = ""
    actual_value: Any = None
    recommended_value: Any = None


@dataclass
class ValidationResultData:
    """Standard validation result data structure."""

    result_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Results
    is_valid: bool = True
    overall_result: ValidationResult = ValidationResult.PASSED
    errors: list[StateValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    # Metrics
    validation_duration_ms: float = 0.0
    rules_executed: int = 0
    rules_passed: int = 0
    rules_failed: int = 0

    # Context
    state_type: str = ""
    state_id: str = ""
    validation_level: str = "standard"


@dataclass
class AuditEntry:
    """Standard audit trail entry structure."""

    audit_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType = AuditEventType.STATE_UPDATED

    # State information
    state_type: str = ""
    state_id: str = ""

    # Change details
    old_value: dict[str, Any] | None = None
    new_value: dict[str, Any] | None = None
    changed_fields: set[str] = field(default_factory=set)

    # Context information
    user_id: str | None = None
    session_id: str | None = None
    source_component: str = ""
    correlation_id: str | None = None

    # Change metadata
    reason: str = ""
    version: int = 1
    checksum_before: str = ""
    checksum_after: str = ""

    # Validation
    validation_status: str = "unknown"
    validation_errors: list[str] = field(default_factory=list)


def classify_error_type(exception: Exception) -> ErrorType:
    """
    Classify exception into standard error type.

    Args:
        exception: Exception to classify

    Returns:
        Classified error type
    """
    import asyncio
    import json

    from sqlalchemy.exc import IntegrityError, OperationalError, TimeoutError as SQLTimeoutError

    if isinstance(exception, (ConnectionError, OSError)):
        if "redis" in str(exception).lower():
            return ErrorType.REDIS_CONNECTION
        return ErrorType.DATABASE_CONNECTION

    elif isinstance(exception, IntegrityError):
        return ErrorType.DATABASE_INTEGRITY

    elif isinstance(exception, (SQLTimeoutError, asyncio.TimeoutError)):
        return ErrorType.DATABASE_TIMEOUT

    elif isinstance(exception, OperationalError):
        error_msg = str(exception).lower()
        if "disk" in error_msg or "space" in error_msg:
            return ErrorType.DISK_SPACE
        elif "permission" in error_msg or "access" in error_msg:
            return ErrorType.PERMISSION
        return ErrorType.DATABASE_CONNECTION

    elif isinstance(exception, (ValueError, TypeError, json.JSONDecodeError)):
        return ErrorType.VALIDATION

    else:
        # Check for concurrency-related errors in the message
        try:
            error_msg = str(exception) if exception is not None else ""
            # Handle case where str() returns None
            if error_msg is None:
                error_msg = ""
        except Exception:
            error_msg = ""
            
        if not error_msg:
            return ErrorType.UNKNOWN
        error_msg = error_msg.lower()
        if "concurrent" in error_msg or "lock" in error_msg:
            return ErrorType.CONCURRENCY
        elif "corruption" in error_msg:
            return ErrorType.DATA_CORRUPTION

    return ErrorType.UNKNOWN


def create_audit_entry(
    event_type: AuditEventType,
    state_type: str,
    state_id: str,
    old_value: dict[str, Any] | None = None,
    new_value: dict[str, Any] | None = None,
    source_component: str = "",
    reason: str = "",
    user_id: str | None = None,
) -> AuditEntry:
    """
    Create a standard audit entry.

    Args:
        event_type: Type of audit event
        state_type: Type of state
        state_id: State identifier
        old_value: Previous state value
        new_value: New state value
        source_component: Component that made the change
        reason: Reason for the change
        user_id: User who made the change

    Returns:
        Configured audit entry
    """
    from src.utils.checksum_utilities import calculate_state_checksum

    # Detect changed fields
    changed_fields = set()
    if old_value and new_value:
        # Check for modified and new fields
        for key, value in new_value.items():
            if key not in old_value or old_value[key] != value:
                changed_fields.add(key)

        # Check for deleted fields
        for key in old_value:
            if key not in new_value:
                changed_fields.add(key)
    elif new_value:
        changed_fields = set(new_value.keys())
    elif old_value:
        changed_fields = set(old_value.keys())

    # Calculate checksums
    checksum_before = calculate_state_checksum(old_value) if old_value else ""
    checksum_after = calculate_state_checksum(new_value) if new_value else ""

    return AuditEntry(
        event_type=event_type,
        state_type=state_type,
        state_id=state_id,
        old_value=old_value,
        new_value=new_value,
        changed_fields=changed_fields,
        source_component=source_component,
        reason=reason,
        user_id=user_id,
        checksum_before=checksum_before,
        checksum_after=checksum_after,
    )
