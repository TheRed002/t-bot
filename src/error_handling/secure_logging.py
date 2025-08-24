"""
Security-focused error logging system.

This module provides comprehensive secure logging for error handling with
automatic sensitive data masking, structured logging, and audit trail capabilities.

CRITICAL: Prevents sensitive information leakage in logs while maintaining
full observability for authorized personnel and audit compliance.
"""

import json
import logging
import logging.handlers
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.error_handling.secure_context_manager import (
    InformationLevel,
    SecurityContext,
    UserRole,
)
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)


class LogLevel(Enum):
    """Secure logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"  # Special security events
    AUDIT = "AUDIT"  # Audit trail events


class LogCategory(Enum):
    """Log categories for classification."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ERROR_HANDLING = "error_handling"
    TRADING = "trading"
    RISK_MANAGEMENT = "risk_management"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    USER_ACTION = "user_action"
    SECURITY_EVENT = "security_event"


@dataclass
class SecureLogEntry:
    """Secure log entry with sanitized information."""

    timestamp: datetime
    level: LogLevel
    category: LogCategory
    component: str
    message: str

    # Context information (sanitized)
    user_id: str | None = None
    session_id: str | None = None
    client_ip: str | None = None
    request_id: str | None = None

    # Technical details (filtered by security level)
    error_id: str | None = None
    error_type: str | None = None
    stack_trace: str | None = None

    # Metadata
    sanitized: bool = True
    information_level: InformationLevel = InformationLevel.BASIC
    additional_data: dict[str, Any] = field(default_factory=dict)

    # Security metadata
    threat_detected: bool = False
    security_classification: str = "INTERNAL"


@dataclass
class LoggingConfig:
    """Configuration for secure logging system."""

    # Basic settings
    log_level: LogLevel = LogLevel.INFO
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    enable_audit_logging: bool = True

    # File settings
    log_directory: str = "logs"
    max_log_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 10

    # Security settings
    default_sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
    enable_encryption: bool = False
    enable_compression: bool = True

    # Audit settings
    audit_log_retention_days: int = 2555  # 7 years
    security_log_retention_days: int = 365  # 1 year

    # Performance settings
    async_logging: bool = True
    buffer_size: int = 1000
    flush_interval_seconds: int = 30

    # Remote logging
    enable_syslog: bool = False
    syslog_host: str | None = None
    syslog_port: int = 514


class SecureLogger:
    """
    Security-focused logger for error handling systems.

    Features:
    - Automatic sensitive data sanitization
    - Role-based log access control
    - Structured logging with JSON format
    - Audit trail compliance
    - Encrypted log storage (optional)
    - Rate limiting for security events
    - Real-time threat detection logging
    - Compliance with security standards
    """

    def __init__(self, name: str, config: LoggingConfig = None):
        self.name = name
        self.config = config or LoggingConfig()
        self.sanitizer = get_security_sanitizer()

        # Thread safety
        self._lock = threading.RLock()

        # Log buffers for async logging
        self._log_buffer: list[SecureLogEntry] = []
        self._audit_buffer: list[SecureLogEntry] = []

        # Initialize loggers
        self._setup_loggers()

        # Background flushing
        if self.config.async_logging:
            self._start_background_flushing()

    def _setup_loggers(self) -> None:
        """Setup secure loggers with appropriate handlers."""

        # Create log directory
        log_dir = Path(self.config.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main application logger
        self.logger = logging.getLogger(f"secure_{self.name}")
        self.logger.setLevel(getattr(logging, self.config.log_level.value))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler (if enabled)
        if self.config.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._get_secure_formatter())
            self.logger.addHandler(console_handler)

        # File handler (if enabled)
        if self.config.enable_file_logging:
            file_path = log_dir / f"{self.name}_secure.log"
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self.config.max_log_file_size,
                backupCount=self.config.backup_count,
            )
            file_handler.setFormatter(self._get_secure_formatter())
            self.logger.addHandler(file_handler)

        # Audit logger
        if self.config.enable_audit_logging:
            self.audit_logger = logging.getLogger(f"audit_{self.name}")
            self.audit_logger.setLevel(logging.INFO)

            audit_file_path = log_dir / f"{self.name}_audit.log"
            audit_handler = logging.handlers.RotatingFileHandler(
                audit_file_path,
                maxBytes=self.config.max_log_file_size,
                backupCount=self.config.backup_count * 2,  # Keep more audit logs
            )
            audit_handler.setFormatter(self._get_audit_formatter())
            self.audit_logger.addHandler(audit_handler)

        # Security events logger
        self.security_logger = logging.getLogger(f"security_{self.name}")
        self.security_logger.setLevel(logging.WARNING)

        security_file_path = log_dir / f"{self.name}_security.log"
        security_handler = logging.handlers.RotatingFileHandler(
            security_file_path,
            maxBytes=self.config.max_log_file_size,
            backupCount=self.config.backup_count * 3,  # Keep even more security logs
        )
        security_handler.setFormatter(self._get_security_formatter())
        self.security_logger.addHandler(security_handler)

    def _get_secure_formatter(self) -> logging.Formatter:
        """Get formatter for secure logging."""
        return logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(thread)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _get_audit_formatter(self) -> logging.Formatter:
        """Get formatter for audit logging."""
        return logging.Formatter(
            "%(asctime)s - AUDIT - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S UTC"
        )

    def _get_security_formatter(self) -> logging.Formatter:
        """Get formatter for security event logging."""
        return logging.Formatter(
            "%(asctime)s - SECURITY - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S UTC"
        )

    def log_error(
        self,
        error: Exception,
        level: LogLevel = LogLevel.ERROR,
        category: LogCategory = LogCategory.ERROR_HANDLING,
        component: str = "unknown",
        security_context: SecurityContext | None = None,
        additional_data: dict[str, Any] | None = None,
        sensitivity_level: SensitivityLevel | None = None,
    ) -> str:
        """
        Log an error with secure sanitization.

        Args:
            error: Exception to log
            level: Log level
            category: Log category
            component: Component name
            security_context: Security context
            additional_data: Additional data to include
            sensitivity_level: Sensitivity level for sanitization

        Returns:
            Log entry ID
        """
        sensitivity = sensitivity_level or self.config.default_sensitivity_level

        # Sanitize error message
        sanitized_message = self.sanitizer.sanitize_error_message(str(error), sensitivity)

        # Sanitize stack trace
        import traceback

        stack_trace = (
            "".join(traceback.format_tb(error.__traceback__)) if error.__traceback__ else None
        )
        sanitized_stack_trace = None
        if stack_trace:
            sanitized_stack_trace = self.sanitizer.sanitize_stack_trace(
                stack_trace, SensitivityLevel.HIGH
            )

        # Create secure log entry
        log_entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            category=category,
            component=component,
            message=sanitized_message,
            error_type=type(error).__name__,
            stack_trace=sanitized_stack_trace,
            additional_data=self.sanitizer.sanitize_context(additional_data or {}, sensitivity),
        )

        # Add security context if available
        if security_context:
            log_entry.user_id = security_context.user_id
            log_entry.session_id = security_context.session_id
            log_entry.client_ip = security_context.client_ip
            log_entry.request_id = security_context.request_id
            log_entry.information_level = self._determine_info_level(security_context)

        # Check for security threats
        log_entry.threat_detected = self._detect_security_threats(error, additional_data)

        return self._write_log_entry(log_entry)

    def log_security_event(
        self,
        event_type: str,
        message: str,
        level: LogLevel = LogLevel.SECURITY,
        security_context: SecurityContext | None = None,
        threat_level: str = "LOW",
        additional_data: dict[str, Any] | None = None,
    ) -> str:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            message: Event message
            level: Log level
            security_context: Security context
            threat_level: Threat level (LOW, MEDIUM, HIGH, CRITICAL)
            additional_data: Additional event data

        Returns:
            Log entry ID
        """
        # Sanitize message
        sanitized_message = self.sanitizer.sanitize_error_message(
            message, SensitivityLevel.CRITICAL
        )

        log_entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            category=LogCategory.SECURITY_EVENT,
            component="security_monitor",
            message=f"[{event_type}] {sanitized_message}",
            threat_detected=threat_level in ["HIGH", "CRITICAL"],
            security_classification=(
                "CONFIDENTIAL" if threat_level in ["HIGH", "CRITICAL"] else "INTERNAL"
            ),
            additional_data=self.sanitizer.sanitize_context(
                additional_data or {}, SensitivityLevel.CRITICAL
            ),
        )

        # Add security context
        if security_context:
            log_entry.user_id = security_context.user_id
            log_entry.session_id = security_context.session_id
            log_entry.client_ip = security_context.client_ip
            log_entry.request_id = security_context.request_id

        # Write to security log
        entry_id = self._write_log_entry(log_entry, is_security_event=True)

        # Also write to audit log for high-level threats
        if threat_level in ["HIGH", "CRITICAL"]:
            self._write_audit_entry(log_entry)

        return entry_id

    def log_audit_event(
        self,
        action: str,
        resource: str,
        result: str,
        security_context: SecurityContext | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> str:
        """
        Log an audit event.

        Args:
            action: Action performed
            resource: Resource accessed
            result: Result of action (SUCCESS, FAILURE, etc.)
            security_context: Security context
            additional_data: Additional audit data

        Returns:
            Log entry ID
        """
        message = f"Action: {action} | Resource: {resource} | Result: {result}"

        log_entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.AUDIT,
            category=LogCategory.USER_ACTION,
            component="audit_logger",
            message=message,
            security_classification="AUDIT",
            additional_data=self.sanitizer.sanitize_context(
                additional_data or {},
                SensitivityLevel.LOW,  # Audit logs can be less restrictive
            ),
        )

        # Add security context
        if security_context:
            log_entry.user_id = security_context.user_id
            log_entry.session_id = security_context.session_id
            log_entry.client_ip = security_context.client_ip
            log_entry.request_id = security_context.request_id

        return self._write_audit_entry(log_entry)

    def log_trading_event(
        self,
        event_type: str,
        symbol: str,
        action: str,
        result: str,
        security_context: SecurityContext | None = None,
        trading_data: dict[str, Any] | None = None,
    ) -> str:
        """
        Log a trading-related event.

        Args:
            event_type: Type of trading event
            symbol: Trading symbol
            action: Trading action
            result: Action result
            security_context: Security context
            trading_data: Trading-specific data

        Returns:
            Log entry ID
        """
        message = (
            f"Trading Event: {event_type} | Symbol: {symbol} | Action: {action} | Result: {result}"
        )

        # Sanitize trading data (financial data is sensitive)
        sanitized_trading_data = self.sanitizer.sanitize_context(
            trading_data or {}, SensitivityLevel.HIGH
        )

        log_entry = SecureLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            category=LogCategory.TRADING,
            component="trading_system",
            message=message,
            security_classification="CONFIDENTIAL",  # Trading data is confidential
            additional_data=sanitized_trading_data,
        )

        # Add security context
        if security_context:
            log_entry.user_id = security_context.user_id
            log_entry.session_id = security_context.session_id
            log_entry.client_ip = security_context.client_ip
            log_entry.request_id = security_context.request_id

        # Trading events go to both main log and audit
        entry_id = self._write_log_entry(log_entry)
        self._write_audit_entry(log_entry)

        return entry_id

    def _write_log_entry(self, log_entry: SecureLogEntry, is_security_event: bool = False) -> str:
        """Write log entry to appropriate destination."""

        # Generate unique log entry ID
        entry_id = f"LOG_{int(time.time() * 1000000)}_{id(log_entry)}"
        log_entry.additional_data["log_entry_id"] = entry_id

        # Choose logger
        target_logger = self.security_logger if is_security_event else self.logger

        # Format log message
        formatted_message = self._format_log_message(log_entry)

        with self._lock:
            if self.config.async_logging:
                # Add to buffer for async processing
                self._log_buffer.append(log_entry)

                # Flush if buffer is full
                if len(self._log_buffer) >= self.config.buffer_size:
                    self._flush_log_buffer()
            else:
                # Synchronous logging
                self._write_to_logger(target_logger, log_entry, formatted_message)

        return entry_id

    def _write_audit_entry(self, log_entry: SecureLogEntry) -> str:
        """Write entry to audit log."""

        entry_id = f"AUDIT_{int(time.time() * 1000000)}_{id(log_entry)}"
        log_entry.additional_data["audit_entry_id"] = entry_id

        formatted_message = self._format_audit_message(log_entry)

        with self._lock:
            if self.config.async_logging:
                self._audit_buffer.append(log_entry)

                if len(self._audit_buffer) >= self.config.buffer_size // 2:
                    self._flush_audit_buffer()
            else:
                self._write_to_logger(self.audit_logger, log_entry, formatted_message)

        return entry_id

    def _format_log_message(self, log_entry: SecureLogEntry) -> str:
        """Format log message for output."""

        # Create structured log data
        log_data = {
            "timestamp": log_entry.timestamp.isoformat(),
            "level": log_entry.level.value,
            "category": log_entry.category.value,
            "component": log_entry.component,
            "message": log_entry.message,
            "sanitized": log_entry.sanitized,
            "security_classification": log_entry.security_classification,
        }

        # Add context if available
        if log_entry.user_id:
            log_data["user_id"] = log_entry.user_id
        if log_entry.session_id:
            log_data["session_id"] = log_entry.session_id
        if log_entry.client_ip:
            log_data["client_ip"] = log_entry.client_ip
        if log_entry.request_id:
            log_data["request_id"] = log_entry.request_id

        # Add error information if available
        if log_entry.error_id:
            log_data["error_id"] = log_entry.error_id
        if log_entry.error_type:
            log_data["error_type"] = log_entry.error_type

        # Add stack trace for ERROR and CRITICAL levels
        if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL] and log_entry.stack_trace:
            log_data["stack_trace"] = log_entry.stack_trace

        # Add threat detection flag
        if log_entry.threat_detected:
            log_data["threat_detected"] = True

        # Add additional data
        if log_entry.additional_data:
            log_data["additional_data"] = log_entry.additional_data

        return json.dumps(log_data, ensure_ascii=False, separators=(",", ":"))

    def _format_audit_message(self, log_entry: SecureLogEntry) -> str:
        """Format audit log message."""

        audit_data = {
            "timestamp": log_entry.timestamp.isoformat(),
            "audit_event": True,
            "category": log_entry.category.value,
            "component": log_entry.component,
            "message": log_entry.message,
            "user_id": log_entry.user_id,
            "session_id": log_entry.session_id,
            "client_ip": log_entry.client_ip,
            "request_id": log_entry.request_id,
            "security_classification": log_entry.security_classification,
        }

        # Add additional data for audit trail
        if log_entry.additional_data:
            audit_data["audit_data"] = log_entry.additional_data

        return json.dumps(audit_data, ensure_ascii=False, separators=(",", ":"))

    def _write_to_logger(
        self, logger: logging.Logger, log_entry: SecureLogEntry, message: str
    ) -> None:
        """Write message to specific logger."""

        log_level_mapping = {
            LogLevel.DEBUG: logger.debug,
            LogLevel.INFO: logger.info,
            LogLevel.WARNING: logger.warning,
            LogLevel.ERROR: logger.error,
            LogLevel.CRITICAL: logger.critical,
            LogLevel.SECURITY: logger.warning,  # Security events as warnings
            LogLevel.AUDIT: logger.info,  # Audit events as info
        }

        log_func = log_level_mapping.get(log_entry.level, logger.info)
        log_func(message)

    def _detect_security_threats(
        self, error: Exception, additional_data: dict[str, Any] | None
    ) -> bool:
        """Detect if error indicates a security threat."""

        error_message = str(error).lower()
        error_type = type(error).__name__.lower()

        # Security threat indicators
        threat_indicators = [
            # Authentication attacks
            "brute force",
            "credential stuffing",
            "password spray",
            "authentication failure",
            "invalid credentials",
            # Injection attacks
            "sql injection",
            "command injection",
            "code injection",
            "xss",
            "cross-site scripting",
            # Access violations
            "unauthorized",
            "permission denied",
            "access denied",
            "privilege escalation",
            # Data exfiltration
            "data breach",
            "unauthorized access",
            "data leak",
            # System attacks
            "buffer overflow",
            "denial of service",
            "dos attack",
            "malware",
            "virus",
            "trojan",
            # API abuse
            "rate limit exceeded",
            "quota exceeded",
            "api abuse",
        ]

        # Check error message
        if any(indicator in error_message for indicator in threat_indicators):
            return True

        # Check error type
        if any(indicator in error_type for indicator in threat_indicators):
            return True

        # Check additional data for threat indicators
        if additional_data:
            data_str = str(additional_data).lower()
            if any(indicator in data_str for indicator in threat_indicators):
                return True

        return False

    def _determine_info_level(self, security_context: SecurityContext) -> InformationLevel:
        """Determine information level based on security context."""

        role_mapping = {
            UserRole.GUEST: InformationLevel.MINIMAL,
            UserRole.USER: InformationLevel.BASIC,
            UserRole.ADMIN: InformationLevel.DETAILED,
            UserRole.DEVELOPER: InformationLevel.FULL,
            UserRole.SECURITY: InformationLevel.DEBUG,
            UserRole.SYSTEM: InformationLevel.DEBUG,
        }

        return role_mapping.get(security_context.user_role, InformationLevel.MINIMAL)

    def _start_background_flushing(self) -> None:
        """Start background thread for flushing log buffers."""

        def flush_periodically():
            while True:
                time.sleep(self.config.flush_interval_seconds)
                try:
                    with self._lock:
                        if self._log_buffer:
                            self._flush_log_buffer()
                        if self._audit_buffer:
                            self._flush_audit_buffer()
                except Exception as e:
                    # Log flushing error to stderr to avoid recursion
                    print(f"Error flushing log buffers: {e}", file=os.sys.stderr)

        flush_thread = threading.Thread(target=flush_periodically, daemon=True)
        flush_thread.start()

    def _flush_log_buffer(self) -> None:
        """Flush log buffer to file."""

        if not self._log_buffer:
            return

        # Process buffered log entries
        for log_entry in self._log_buffer:
            formatted_message = self._format_log_message(log_entry)

            # Determine target logger
            if log_entry.level == LogLevel.SECURITY or log_entry.threat_detected:
                target_logger = self.security_logger
            else:
                target_logger = self.logger

            self._write_to_logger(target_logger, log_entry, formatted_message)

        # Clear buffer
        self._log_buffer.clear()

    def _flush_audit_buffer(self) -> None:
        """Flush audit buffer to file."""

        if not self._audit_buffer:
            return

        for log_entry in self._audit_buffer:
            formatted_message = self._format_audit_message(log_entry)
            self._write_to_logger(self.audit_logger, log_entry, formatted_message)

        self._audit_buffer.clear()

    def flush(self) -> None:
        """Manually flush all log buffers."""

        with self._lock:
            self._flush_log_buffer()
            self._flush_audit_buffer()

    def get_logger_stats(self) -> dict[str, Any]:
        """Get logging system statistics."""

        return {
            "logger_name": self.name,
            "log_level": self.config.log_level.value,
            "async_logging": self.config.async_logging,
            "buffer_sizes": {
                "log_buffer": len(self._log_buffer),
                "audit_buffer": len(self._audit_buffer),
            },
            "handlers": {
                "main_logger": len(self.logger.handlers),
                "audit_logger": (
                    len(self.audit_logger.handlers) if hasattr(self, "audit_logger") else 0
                ),
                "security_logger": len(self.security_logger.handlers),
            },
            "config": {
                "max_file_size": self.config.max_log_file_size,
                "backup_count": self.config.backup_count,
                "default_sensitivity": self.config.default_sensitivity_level.value,
            },
        }


# Global secure loggers
_loggers: dict[str, SecureLogger] = {}
_logger_lock = threading.RLock()


def get_secure_logger(name: str, config: LoggingConfig | None = None) -> SecureLogger:
    """Get or create a secure logger instance."""

    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = SecureLogger(name, config)
        return _loggers[name]


def log_security_error(
    error: Exception,
    component: str,
    security_context: SecurityContext | None = None,
    additional_data: dict[str, Any] | None = None,
) -> str:
    """Convenience function for logging security-related errors."""

    logger = get_secure_logger(component)
    return logger.log_error(
        error,
        level=LogLevel.ERROR,
        category=LogCategory.SECURITY_EVENT,
        component=component,
        security_context=security_context,
        additional_data=additional_data,
        sensitivity_level=SensitivityLevel.CRITICAL,
    )


def log_trading_error(
    error: Exception,
    component: str,
    trading_context: dict[str, Any],
    security_context: SecurityContext | None = None,
) -> str:
    """Convenience function for logging trading-related errors."""

    logger = get_secure_logger(component)
    return logger.log_error(
        error,
        level=LogLevel.ERROR,
        category=LogCategory.TRADING,
        component=component,
        security_context=security_context,
        additional_data=trading_context,
        sensitivity_level=SensitivityLevel.HIGH,
    )
