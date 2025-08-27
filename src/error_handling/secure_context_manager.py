"""
Secure error context manager for safe error reporting.

This module provides secure error context management that filters internal system
details from error reports based on user roles and security clearance levels.

CRITICAL: Prevents exposure of internal architecture, database schemas, file paths,
configuration details, and other sensitive system information in error reports.
"""

import hashlib
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.logging import get_logger
from src.error_handling.security_sanitizer import (
    SensitivityLevel,
    get_security_sanitizer,
)


class UserRole(Enum):
    """User roles for access control."""

    GUEST = "guest"  # External users, minimal access
    USER = "user"  # Regular users
    ADMIN = "admin"  # System administrators
    DEVELOPER = "developer"  # Development team
    SECURITY = "security"  # Security team
    SYSTEM = "system"  # Internal system operations


class InformationLevel(Enum):
    """Information disclosure levels."""

    MINIMAL = "minimal"  # User-friendly messages only
    BASIC = "basic"  # Basic error information
    DETAILED = "detailed"  # Detailed error information
    FULL = "full"  # Complete error information
    DEBUG = "debug"  # Full debug information


@dataclass
class SecurityContext:
    """Security context for error reporting."""

    user_role: UserRole = UserRole.GUEST
    user_id: str | None = None
    session_id: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None
    request_id: str | None = None

    # Security flags
    is_authenticated: bool = False
    has_admin_access: bool = False
    is_internal_user: bool = False

    # Context metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: str | None = None
    operation: str | None = None


@dataclass
class SecureErrorReport:
    """Secure error report with filtered information."""

    error_id: str
    timestamp: datetime
    user_message: str  # Safe message for end users
    technical_message: str  # Technical details (filtered)
    error_code: str | None = None

    # Metadata
    component: str | None = None
    operation: str | None = None

    # Context information (filtered based on role)
    context: dict[str, Any] = field(default_factory=dict)

    # Debugging information (only for authorized users)
    debug_info: dict[str, Any] | None = None

    # Security metadata
    information_level: InformationLevel = InformationLevel.MINIMAL
    sanitized: bool = True


class SecureErrorContextManager:
    """
    Secure error context manager for financial trading systems.

    Provides role-based error information filtering to prevent exposure of:
    - Internal system architecture details
    - Database schema and connection information
    - File system structure and paths
    - Configuration parameters and secrets
    - Network topology and internal addresses
    - Business logic implementation details
    - Third-party service configurations
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.sanitizer = get_security_sanitizer()

        # Role-based information level mapping
        self.role_info_levels = {
            UserRole.GUEST: InformationLevel.MINIMAL,
            UserRole.USER: InformationLevel.BASIC,
            UserRole.ADMIN: InformationLevel.DETAILED,
            UserRole.DEVELOPER: InformationLevel.FULL,
            UserRole.SECURITY: InformationLevel.DEBUG,
            UserRole.SYSTEM: InformationLevel.DEBUG,
        }

        # Safe error messages for different categories
        self.safe_error_messages = {
            "authentication": "Authentication failed. Please check your credentials.",
            "authorization": "You don't have permission to perform this action.",
            "validation": "The provided data is invalid. Please check your input.",
            "network": "A network error occurred. Please try again later.",
            "database": "A temporary service error occurred. Please try again later.",
            "exchange": "Trading service temporarily unavailable. Please try again later.",
            "rate_limit": "Too many requests. Please wait before trying again.",
            "maintenance": "Service is currently under maintenance. Please try again later.",
            "internal": "An internal error occurred. Please contact support if this continues.",
            "unknown": "An unexpected error occurred. Please try again or contact support.",
        }

        # Internal system keywords to filter out
        self.internal_keywords = [
            # File system
            "src/",
            "lib/",
            "bin/",
            "var/",
            "tmp/",
            "opt/",
            "usr/",
            "C:\\",
            "D:\\",
            "Program Files",
            "AppData",
            "Documents",
            # Database
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "postgresql://",
            "mysql://",
            "mongodb://",
            "redis://",
            "connection_string",
            "database_url",
            "db_host",
            "db_port",
            # Configuration
            "config",
            "settings",
            "environment",
            "dotenv",
            ".env",
            "SECRET_KEY",
            "API_KEY",
            "JWT_SECRET",
            "PRIVATE_KEY",
            # Network
            "localhost",
            "127.0.0.1",
            "192.168.",
            "10.0.",
            "172.16.",
            "internal.",
            "staging.",
            "dev.",
            "test.",
            # Code structure
            "__init__.py",
            "__main__.py",
            "requirements.txt",
            "setup.py",
            "Dockerfile",
            "docker-compose",
            "kubernetes",
            "helm",
            # Business logic
            "profit",
            "loss",
            "position",
            "portfolio",
            "balance",
            "trading_strategy",
            "algorithm",
            "model",
            "prediction",
        ]

    def create_secure_report(
        self,
        error: Exception,
        security_context: SecurityContext,
        original_context: dict[str, Any] | None = None,
    ) -> SecureErrorReport:
        """
        Create a secure error report filtered based on user role.

        Args:
            error: The original exception
            security_context: Security context with user role information
            original_context: Original error context (will be filtered)

        Returns:
            Secure error report with appropriate information level
        """
        info_level = self.role_info_levels.get(security_context.user_role, InformationLevel.MINIMAL)

        # Generate error ID
        error_id = self._generate_error_id(error, security_context)

        # Create user-safe message
        user_message = self._create_user_message(error, security_context)

        # Create technical message (filtered)
        technical_message = self._create_technical_message(error, security_context, info_level)

        # Filter context information
        filtered_context = self._filter_context(
            original_context or {}, security_context, info_level
        )

        # Create debug information (if authorized)
        debug_info = None
        if info_level in [InformationLevel.DEBUG, InformationLevel.FULL]:
            debug_info = self._create_debug_info(error, original_context)

        # Determine error code
        error_code = self._get_error_code(error, security_context)

        return SecureErrorReport(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            user_message=user_message,
            technical_message=technical_message,
            error_code=error_code,
            component=security_context.component,
            operation=security_context.operation,
            context=filtered_context,
            debug_info=debug_info,
            information_level=info_level,
            sanitized=True,
        )

    def _generate_error_id(self, error: Exception, security_context: SecurityContext) -> str:
        """Generate a unique error ID for tracking."""
        # Create hash from error details and timestamp
        error_details = f"{type(error).__name__}:{error!s}:{security_context.timestamp}"
        hash_obj = hashlib.sha256(error_details.encode("utf-8"))
        return f"ERR_{hash_obj.hexdigest()[:12].upper()}"

    def _create_user_message(self, error: Exception, security_context: SecurityContext) -> str:
        """Create a user-friendly error message."""
        error_category = self._categorize_error(error)

        # Get safe message for category
        base_message = self.safe_error_messages.get(
            error_category, self.safe_error_messages["unknown"]
        )

        # Add context if appropriate
        if security_context.user_role in [UserRole.ADMIN, UserRole.DEVELOPER, UserRole.SECURITY]:
            if security_context.component:
                base_message += f" (Component: {security_context.component})"

        return base_message

    def _create_technical_message(
        self, error: Exception, security_context: SecurityContext, info_level: InformationLevel
    ) -> str:
        """Create technical error message filtered by information level."""
        original_message = str(error)

        if info_level == InformationLevel.MINIMAL:
            # Only show error type
            return f"Error: {type(error).__name__}"

        elif info_level == InformationLevel.BASIC:
            # Show sanitized error message
            sanitized_message = self.sanitizer.sanitize_error_message(
                original_message, SensitivityLevel.LOW
            )
            return f"{type(error).__name__}: {sanitized_message}"

        elif info_level == InformationLevel.DETAILED:
            # Show more details but still sanitized
            sanitized_message = self.sanitizer.sanitize_error_message(
                original_message, SensitivityLevel.MEDIUM
            )
            filtered_message = self._filter_internal_details(sanitized_message)
            return f"{type(error).__name__}: {filtered_message}"

        elif info_level == InformationLevel.FULL:
            # Show full sanitized message
            sanitized_message = self.sanitizer.sanitize_error_message(
                original_message, SensitivityLevel.HIGH
            )
            return f"{type(error).__name__}: {sanitized_message}"

        else:  # DEBUG
            # Show everything (for security team and system operations)
            return f"{type(error).__name__}: {original_message}"

    def _filter_context(
        self,
        original_context: dict[str, Any],
        security_context: SecurityContext,
        info_level: InformationLevel,
    ) -> dict[str, Any]:
        """Filter context information based on information level."""
        if info_level == InformationLevel.MINIMAL:
            # Only show basic operation info
            return {
                "operation": security_context.operation,
                "timestamp": security_context.timestamp.isoformat(),
            }

        elif info_level == InformationLevel.BASIC:
            # Show limited context
            allowed_keys = [
                "operation",
                "component",
                "user_id",
                "session_id",
                "timestamp",
                "error_code",
                "retry_count",
            ]
            filtered = {}
            for key in allowed_keys:
                if key in original_context:
                    value = original_context[key]
                    if isinstance(value, str):
                        value = self.sanitizer.sanitize_error_message(value, SensitivityLevel.LOW)
                    filtered[key] = value
            return filtered

        elif info_level == InformationLevel.DETAILED:
            # Show more context but filter sensitive data
            filtered = {}
            for key, value in original_context.items():
                # Skip sensitive keys
                if any(
                    keyword in key.lower()
                    for keyword in ["password", "secret", "key", "token", "credential"]
                ):
                    continue

                # Sanitize values
                if isinstance(value, str):
                    value = self.sanitizer.sanitize_error_message(value, SensitivityLevel.MEDIUM)
                    value = self._filter_internal_details(value)
                elif isinstance(value, dict):
                    value = self._filter_context(value, security_context, info_level)

                filtered[key] = value
            return filtered

        elif info_level == InformationLevel.FULL:
            # Show comprehensive context with sanitization
            return self.sanitizer.sanitize_context(original_context, SensitivityLevel.HIGH)

        else:  # DEBUG
            # Show everything for debugging (security team only)
            return original_context

    def _create_debug_info(
        self, error: Exception, original_context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Create debug information for authorized users."""
        debug_info = {
            "error_type": type(error).__name__,
            "error_module": error.__class__.__module__,
            "python_version": sys.version,
            "process_id": os.getpid(),
        }

        # Add stack trace (sanitized)
        import traceback

        stack_trace = "".join(traceback.format_tb(error.__traceback__))
        debug_info["stack_trace"] = self.sanitizer.sanitize_stack_trace(
            stack_trace, SensitivityLevel.HIGH
        )

        # Add system information (filtered)
        debug_info["system_info"] = {
            "platform": os.name,
            "working_directory": self._filter_path(os.getcwd()),
        }

        # Add original context (full)
        if original_context:
            debug_info["original_context"] = original_context

        return debug_info

    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for appropriate messaging."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()

        # Authentication/Authorization errors
        if any(keyword in error_type for keyword in ["auth", "permission", "forbidden"]):
            return "authentication" if "auth" in error_type else "authorization"
        if any(keyword in error_message for keyword in ["permission", "forbidden", "unauthorized"]):
            return "authorization"

        # Validation errors
        if any(keyword in error_type for keyword in ["validation", "value", "type"]):
            return "validation"
        if any(keyword in error_message for keyword in ["invalid", "required", "format"]):
            return "validation"

        # Network errors
        if any(keyword in error_type for keyword in ["connection", "network", "timeout"]):
            return "network"
        if any(keyword in error_message for keyword in ["connection", "timeout", "network"]):
            return "network"

        # Database errors
        if any(keyword in error_type for keyword in ["database", "sql", "operational"]):
            return "database"
        if any(keyword in error_message for keyword in ["database", "connection pool", "sql"]):
            return "database"

        # Exchange errors
        if any(keyword in error_type for keyword in ["exchange", "trading", "order"]):
            return "exchange"
        if any(keyword in error_message for keyword in ["exchange", "trading", "market"]):
            return "exchange"

        # Rate limiting
        if any(keyword in error_message for keyword in ["rate limit", "too many", "429"]):
            return "rate_limit"

        # Maintenance
        if any(keyword in error_message for keyword in ["maintenance", "unavailable", "503"]):
            return "maintenance"

        # System errors
        if any(keyword in error_type for keyword in ["system", "os", "memory"]):
            return "internal"

        return "unknown"

    def _get_error_code(self, error: Exception, security_context: SecurityContext) -> str | None:
        """Get standardized error code."""
        error_category = self._categorize_error(error)
        type(error).__name__

        # Generate standardized error code
        code_mapping = {
            "authentication": "AUTH_001",
            "authorization": "AUTH_002",
            "validation": "VAL_001",
            "network": "NET_001",
            "database": "DB_001",
            "exchange": "EXC_001",
            "rate_limit": "RATE_001",
            "maintenance": "MAINT_001",
            "internal": "SYS_001",
            "unknown": "ERR_001",
        }

        base_code = code_mapping.get(error_category, "ERR_001")

        # Add component suffix if available
        if security_context.component:
            component_suffix = security_context.component.upper()[:3]
            return f"{base_code}_{component_suffix}"

        return base_code

    def _filter_internal_details(self, message: str) -> str:
        """Filter out internal system details from message."""
        filtered_message = message

        # Replace internal keywords with generic terms
        replacements = {
            # File paths
            r"src/[^\s]*": "[SYSTEM_PATH]",
            r"lib/[^\s]*": "[SYSTEM_PATH]",
            r"C:\\[^\s]*": "[SYSTEM_PATH]",
            r"/[a-zA-Z/]*\.py": "[SYSTEM_FILE]",
            # Database details
            r"postgresql://[^\s]*": "[DATABASE_URL]",
            r"mysql://[^\s]*": "[DATABASE_URL]",
            r"mongodb://[^\s]*": "[DATABASE_URL]",
            r"redis://[^\s]*": "[DATABASE_URL]",
            # Network details
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b": "[IP_ADDRESS]",
            r"localhost:\d+": "[LOCAL_SERVICE]",
            # Configuration
            r"SECRET_KEY.*": "[CONFIG_SECRET]",
            r"API_KEY.*": "[CONFIG_SECRET]",
        }

        import re

        for pattern, replacement in replacements.items():
            filtered_message = re.sub(pattern, replacement, filtered_message, flags=re.IGNORECASE)

        return filtered_message

    def _filter_path(self, path: str) -> str:
        """Filter file system path for safe reporting."""
        if not path:
            return path

        # Convert to relative path from project root
        if "t-bot" in path:
            parts = path.split("t-bot")
            return f"[PROJECT_ROOT]{parts[-1]}" if len(parts) > 1 else "[PROJECT_ROOT]"

        # Generic path filtering
        import os

        return f"[WORK_DIR]/{os.path.basename(path)}"

    def validate_user_access(
        self, security_context: SecurityContext, requested_level: InformationLevel
    ) -> bool:
        """Validate if user has access to requested information level."""
        user_max_level = self.role_info_levels.get(
            security_context.user_role, InformationLevel.MINIMAL
        )

        level_hierarchy = {
            InformationLevel.MINIMAL: 0,
            InformationLevel.BASIC: 1,
            InformationLevel.DETAILED: 2,
            InformationLevel.FULL: 3,
            InformationLevel.DEBUG: 4,
        }

        return level_hierarchy[requested_level] <= level_hierarchy[user_max_level]

    def get_safe_error_summary(self, security_context: SecurityContext) -> dict[str, Any]:
        """Get safe error handling summary for user's role."""
        info_level = self.role_info_levels.get(security_context.user_role, InformationLevel.MINIMAL)

        summary = {
            "user_role": security_context.user_role.value,
            "information_level": info_level.value,
            "available_error_categories": list(self.safe_error_messages.keys()),
        }

        if info_level in [InformationLevel.FULL, InformationLevel.DEBUG]:
            summary["sanitization_rules"] = self.sanitizer.get_sanitization_stats()

        return summary


# Global secure context manager instance
_global_context_manager = None


def get_secure_context_manager() -> SecureErrorContextManager:
    """Get global secure error context manager."""
    global _global_context_manager
    if _global_context_manager is None:
        _global_context_manager = SecureErrorContextManager()
    return _global_context_manager


def create_security_context(
    user_role: UserRole = UserRole.GUEST,
    user_id: str | None = None,
    client_ip: str | None = None,
    **kwargs,
) -> SecurityContext:
    """Convenience function for creating security context."""
    return SecurityContext(user_role=user_role, user_id=user_id, client_ip=client_ip, **kwargs)


def create_secure_error_report(
    error: Exception,
    security_context: SecurityContext,
    original_context: dict[str, Any] | None = None,
) -> SecureErrorReport:
    """Convenience function for creating secure error reports."""
    manager = get_secure_context_manager()
    return manager.create_secure_report(error, security_context, original_context)
