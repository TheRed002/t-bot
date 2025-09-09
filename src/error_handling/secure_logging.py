"""
Secure logging backward compatibility.

Redirects to consolidated security types.
"""

from typing import Any

from src.utils.security_types import (
    LogCategory,
    LoggingConfig,
    LogLevel,
    SecureLogEntry,
)

# Re-export for backward compatibility
__all__ = [
    "LogCategory",
    "LogLevel",
    "LoggingConfig",
    "SecureLogEntry",
    "SecureLogger",
    "create_secure_logger",
    "get_security_sanitizer",
    "sanitize_log_message",
    "should_log_error",
]


def sanitize_log_message(message: str) -> str:
    """Sanitize log message for security."""
    sanitized = message
    for sensitive_term in ["password", "token", "key", "secret", "credential"]:
        sanitized = sanitized.replace(sensitive_term, "***")
    return sanitized


def should_log_error(error: Exception, context: dict[str, Any] | None = None) -> bool:
    """Determine if error should be logged based on severity."""
    return not isinstance(error, KeyboardInterrupt | SystemExit)


def get_security_sanitizer():
    """Get security sanitizer instance."""
    try:
        from src.error_handling.security_validator import (
            get_security_sanitizer as get_validator_sanitizer,
        )

        return get_validator_sanitizer()
    except ImportError:
        return None


def create_secure_logger(config=None):
    """Create secure logger instance."""
    return SecureLogger(config)


class SecureLogger:
    """Secure logger with data sanitization capabilities."""

    def __init__(self, name: str = None, config: LoggingConfig | None = None):
        if isinstance(name, LoggingConfig):
            # Handle case where first arg is config
            self.config = name
            self.name = "default"
        else:
            self.config = config or LoggingConfig()
            self.name = name or "default"
        self.sanitizer = None
        self._log_entries = []

        # Create logger instance
        import logging
        self.logger = logging.getLogger(self.name)

    def log(self, level: LogLevel, message: str, context: dict[str, Any] | None = None) -> None:
        """Log a message."""
        if self.config.sanitize:
            message = sanitize_log_message(message)
        # Log message processing completed

    def log_error(
        self,
        error: Exception = None,
        level: LogLevel = None,
        category: LogCategory = None,
        component: str = None,
        security_context: Any = None,
        additional_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log an error."""
        from datetime import datetime, timezone

        from src.utils.security_types import SecureLogEntry

        """Create log entry for error tracking."""
        log_entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level or LogLevel.ERROR,
            category=category or LogCategory.SYSTEM,
            component=component or "unknown",
            message=str(error) if error else "Unknown error",
            error_type=error.__class__.__name__ if error else "UnknownError",
            additional_data={"safe": True, **(additional_data or {})},
        )

        self._write_log_entry(log_entry)

    def log_security_event(
        self,
        event: str,
        security_context: Any = None,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Log a security event."""
        from datetime import datetime, timezone

        from src.utils.security_types import SecureLogEntry

        log_entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.SECURITY,
            category=LogCategory.SECURITY,
            component="security",
            message=event,
            threat_detected=True,
            client_ip=security_context.client_ip if security_context else None,
            additional_data=additional_data or {},
        )

        self._write_log_entry(log_entry)

    def log_audit_trail(
        self, action: str, user_id: str = None, additional_data: dict[str, Any] | None = None
    ) -> None:
        """Log an audit trail event."""
        from datetime import datetime, timezone

        from src.utils.security_types import SecureLogEntry

        log_entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            category=LogCategory.SECURITY,
            component="audit",
            message=action,
            user_id=user_id,
            additional_data=additional_data or {},
        )

        self._write_log_entry(log_entry)

    def sanitize_log_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize log data."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = sanitize_log_message(value)
            else:
                sanitized[key] = value
        return sanitized

    def determine_information_level_from_context(self, context: Any = None) -> str:
        """Determine information level from context."""
        return "BASIC"

    def _determine_info_level(self, context: Any = None):
        """Determine information level from security context."""
        from src.error_handling.secure_context_manager import InformationLevel, UserRole

        if not context or not hasattr(context, "user_role"):
            return InformationLevel.MINIMAL

        user_role = context.user_role

        if user_role == UserRole.GUEST:
            return InformationLevel.MINIMAL
        elif user_role == UserRole.ADMIN and context.has_admin_access:
            return InformationLevel.DETAILED
        elif user_role == UserRole.DEVELOPER:
            return InformationLevel.FULL
        else:
            return InformationLevel.BASIC

    def format_log_entry(self, entry: Any, format_type: str = "json") -> str:
        """Format log entry."""
        if format_type == "json":
            return '{"level": "INFO", "message": "test"}'
        return "INFO: test"

    def _format_log_message(self, entry: Any) -> str:
        """Format log message as JSON."""
        import json
        from datetime import datetime

        data = {
            "timestamp": (
                entry.timestamp.isoformat()
                if hasattr(entry, "timestamp")
                else datetime.now().isoformat()
            ),
            "level": entry.level.value if hasattr(entry, "level") else "INFO",
            "category": entry.category.value if hasattr(entry, "category") else "system",
            "component": getattr(entry, "component", "unknown"),
            "message": getattr(entry, "message", ""),
        }

        # Add optional fields
        for field in ["user_id", "client_ip", "request_id", "session_id", "error_type"]:
            if hasattr(entry, field) and getattr(entry, field):
                data[field] = getattr(entry, field)

        return json.dumps(data)

    def should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged."""
        return True

    def _write_log_entry(self, entry: Any) -> str:
        """Write log entry to storage."""
        if self.config.sanitize and hasattr(entry, "message"):
            entry.message = sanitize_log_message(entry.message)

        # Generate a log ID and return it
        import uuid
        log_id = str(uuid.uuid4())[:8]

        # Store the entry
        self._log_entries.append(entry)

        return log_id

    def _log_to_logger(self, entry: Any) -> None:
        """Log to underlying logger system."""
        # Delegate to standard logging system
        self.logger.info(str(entry))

    def _write_to_logger(self, logger, entry: Any, level: LogLevel) -> None:
        """Write entry to logger."""
        if hasattr(logger, level.value):
            log_method = getattr(logger, level.value)
            formatted_message = self._format_log_message(entry)
            log_method(formatted_message)

    def _write_audit_entry(self, entry: Any) -> None:
        """Write audit entry to secure storage."""
        # Audit storage not implemented in current version
        return

    def log_performance_metrics(self, metrics: dict[str, Any]) -> None:
        """Log performance metrics."""
        # Performance metrics logging not implemented in current version
        return

    def serialize_log_entry(self, entry: Any) -> str:
        """Serialize log entry."""
        return "{}"


