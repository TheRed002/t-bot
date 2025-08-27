"""
Security-focused data sanitizer for error handling.

This module provides comprehensive data sanitization for error messages, stack traces,
and context information to prevent exposure of sensitive information in trading systems.

CRITICAL: This module prevents exposure of API keys, passwords, tokens, private keys,
wallet addresses, personal information, and internal system architecture details.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.logging import get_logger


class SensitivityLevel(Enum):
    """Security sensitivity levels for error sanitization."""

    LOW = "low"  # General error messages
    MEDIUM = "medium"  # Internal system errors
    HIGH = "high"  # Authentication/authorization errors
    CRITICAL = "critical"  # Financial/trading errors


@dataclass
class SanitizationRule:
    """Rule for sanitizing sensitive data patterns."""

    name: str
    pattern: str
    replacement: str | Callable[[re.Match[str]], str]
    flags: int = re.IGNORECASE | re.MULTILINE
    description: str = ""
    enabled: bool = True


@dataclass
class SanitizationConfig:
    """Configuration for data sanitization."""

    # Masking characters
    mask_char: str = "*"
    mask_length: int = 8

    # Hash settings for sensitive values
    hash_sensitive_values: bool = True
    hash_algorithm: str = "sha256"
    hash_prefix: str = "HASH_"

    # Path sanitization
    sanitize_file_paths: bool = True
    show_relative_paths_only: bool = True

    # Network sanitization
    sanitize_network_info: bool = True
    mask_ip_addresses: bool = True
    mask_ports: bool = True

    # Custom patterns to sanitize
    custom_patterns: list[SanitizationRule] = field(default_factory=list)


class SecurityDataSanitizer:
    """
    Comprehensive data sanitizer for secure error handling in financial systems.

    This sanitizer prevents exposure of:
    - API keys, tokens, and secrets
    - Passwords and authentication credentials
    - Private keys and wallet addresses
    - Personal identifiable information (PII)
    - Internal system architecture details
    - Database connection strings
    - File system paths
    - Network topology information
    """

    def __init__(self, config: SanitizationConfig | None = None):
        self.config = config or SanitizationConfig()
        self.logger = get_logger(self.__class__.__module__)

        # Initialize sanitization rules
        self._init_sanitization_rules()

    def _init_sanitization_rules(self) -> None:
        """Initialize comprehensive sanitization rules."""

        # Core sensitive data patterns
        self.rules = [
            # Stripe-style API Keys (sk_test_, sk_live_, pk_test_, pk_live_, etc.)
            SanitizationRule(
                name="stripe_api_keys",
                pattern=r"(sk_test_|sk_live_|pk_test_|pk_live_|rk_test_|rk_live_)([a-zA-Z0-9]{1,})",
                replacement=lambda m: f"{m.group(1)}{self._mask_value(m.group(2))}",
                description="Stripe-style API keys",
            ),
            # Generic API Keys and Tokens
            SanitizationRule(
                name="api_keys",
                pattern=r"(?i)(api[_\-]?key|access[_\-]?key|secret[_\-]?key|auth[_\-]?key)['\"\s]*[:=]['\"\s]*([a-zA-Z0-9\-_+/=]{8,})",
                replacement=lambda m: f"{m.group(1)}={self._mask_value(m.group(2))}",
                description="Generic API keys and access tokens",
            ),
            # Generic secrets (like 'secret is abc123xyz')
            SanitizationRule(
                name="generic_secrets",
                pattern=r"(?i)\b(secret|password|key)\s+is\s+([a-zA-Z0-9\-_+/=]{6,})",
                replacement=lambda m: f"{m.group(1)} is {self._mask_value(m.group(2))}",
                description="Generic secret values in text",
            ),
            # Standalone API keys (when they appear alone without context)
            SanitizationRule(
                name="standalone_api_keys",
                pattern=r"^(sk_test_|sk_live_|pk_test_|pk_live_|rk_test_|rk_live_)([a-zA-Z0-9]{1,})$",
                replacement=lambda m: f"{m.group(1)}{self._mask_value(m.group(2))}",
                description="Standalone API keys without context",
            ),
            # JWT Tokens
            SanitizationRule(
                name="jwt_tokens",
                pattern=r"(?i)(bearer\s+|token['\"\s]*[:=]['\"\s]*)(ey[a-zA-Z0-9\-_=]+\.[a-zA-Z0-9\-_=]+\.?[a-zA-Z0-9\-_=]*)",
                replacement=lambda m: f"{m.group(1)}{self._mask_value(m.group(2))}",
                description="JWT tokens",
            ),
            # Passwords
            SanitizationRule(
                name="passwords",
                pattern=r"(?i)(password|pwd|passwd|pass)['\"\s]*[:=]['\"\s]*([^\s'\";,}]{6,})",
                replacement=lambda m: f"{m.group(1)}={self._mask_value(m.group(2))}",
                description="Passwords with assignment",
            ),
            # Password in text (like 'with password secret123')
            SanitizationRule(
                name="password_in_text",
                pattern=r"(?i)\b(password|pwd|passwd|pass)\s+([a-zA-Z0-9\-_+/=@#$%^&*!]{6,})",
                replacement=lambda m: f"{m.group(1)} {self._mask_value(m.group(2))}",
                description="Passwords in natural text",
            ),
            # Database Connection Strings
            SanitizationRule(
                name="db_connections",
                pattern=r"(?i)(postgresql|mysql|mongodb|redis)://([^:]+):([^@]+)@([^/]+)",
                replacement=lambda m: (
                    f"{m.group(1)}://{self._mask_value(m.group(2))}:"
                    f"{self._mask_value(m.group(3))}@{self._sanitize_host(m.group(4))}"
                ),
                description="Database connection strings",
            ),
            # Private Keys
            SanitizationRule(
                name="private_keys",
                pattern=(
                    r"(-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----.*?"
                    r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)"
                ),
                replacement=lambda m: (
                    "-----BEGIN PRIVATE KEY-----[REDACTED]-----END PRIVATE KEY-----"
                ),
                flags=re.IGNORECASE | re.DOTALL,
                description="Private keys",
            ),
            # Cryptocurrency Addresses (Bitcoin, Ethereum)
            SanitizationRule(
                name="crypto_addresses",
                pattern=r"\b([13][a-km-zA-HJ-NP-Z1-9]{25,34}|0x[a-fA-F0-9]{40}|bc1[a-z0-9]{39,59})\b",
                replacement=lambda m: self._mask_crypto_address(m.group(1)),
                description="Cryptocurrency addresses",
            ),
            # Credit Card Numbers
            SanitizationRule(
                name="credit_cards",
                pattern=r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                replacement=lambda m: self._mask_credit_card(m.group(0)),
                description="Credit card numbers",
            ),
            # SSN and Tax IDs
            SanitizationRule(
                name="ssn_tax_id",
                pattern=r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
                replacement=lambda m: "***-**-****",
                description="Social Security Numbers",
            ),
            # Email Addresses (partial masking)
            SanitizationRule(
                name="email_addresses",
                pattern=r"\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
                replacement=lambda m: f"{self._mask_email_user(m.group(1))}@{m.group(2)}",
                description="Email addresses",
            ),
            # Phone Numbers
            SanitizationRule(
                name="phone_numbers",
                pattern=r"(\+?1?[-.\s]?)?\(?([2-9]\d{2})\)?[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})",
                replacement=lambda m: "+1 (***) ***-****",
                description="Phone numbers",
            ),
            # IP Addresses
            SanitizationRule(
                name="ip_addresses",
                pattern=r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                replacement=lambda m: self._mask_ip_address(m.group(0)),
                description="IP addresses",
            ),
            # File System Paths (Windows and Unix)
            SanitizationRule(
                name="file_paths",
                pattern=r"(?:[A-Za-z]:\\|/)(?:[^\\/:*?\"<>|\r\n]+[\\\/])*[^\\/:*?\"<>|\r\n]*",
                replacement=lambda m: self._sanitize_file_path(m.group(0)),
                description="File system paths",
            ),
            # URLs with credentials
            SanitizationRule(
                name="urls_with_auth",
                pattern=r"(https?://)([^:]+):([^@]+)@([^/]+)",
                replacement=lambda m: (
                    f"{m.group(1)}{self._mask_value(m.group(2))}:"
                    f"{self._mask_value(m.group(3))}@{self._sanitize_host(m.group(4))}"
                ),
                description="URLs with authentication",
            ),
            # Exchange API Secrets
            SanitizationRule(
                name="exchange_secrets",
                pattern=r"(?i)(binance|coinbase|kraken|okx|huobi)[_\-]?(api[_\-]?secret|secret[_\-]?key)['\"\s]*[:=]['\"\s]*([a-zA-Z0-9+/=]{20,})",
                replacement=lambda m: f"{m.group(1)}_{m.group(2)}={self._mask_value(m.group(3))}",
                description="Exchange API secrets",
            ),
            # Trading Keys and Signatures
            SanitizationRule(
                name="trading_signatures",
                pattern=r"(?i)(signature|sign|hmac)['\"\s]*[:=]['\"\s]*([a-fA-F0-9]{32,})",
                replacement=lambda m: f"{m.group(1)}={self._mask_value(m.group(2))}",
                description="Trading signatures",
            ),
        ]

        # Add custom patterns from config
        if self.config.custom_patterns:
            self.rules.extend(self.config.custom_patterns)

    def sanitize_error_message(
        self, message: str, sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
    ) -> str:
        """
        Sanitize error message based on sensitivity level.

        Args:
            message: Error message to sanitize
            sensitivity_level: Security sensitivity level

        Returns:
            Sanitized error message
        """
        if not message:
            return message

        sanitized = message

        # Apply sanitization rules based on sensitivity level
        rules_to_apply = self._get_rules_for_level(sensitivity_level)

        for rule in rules_to_apply:
            if not rule.enabled:
                continue

            try:
                if callable(rule.replacement):
                    sanitized = re.sub(rule.pattern, rule.replacement, sanitized, flags=rule.flags)
                else:
                    sanitized = re.sub(rule.pattern, rule.replacement, sanitized, flags=rule.flags)
            except Exception as e:
                self.logger.warning(f"Error applying sanitization rule {rule.name}: {e}")

        return sanitized

    def sanitize_stack_trace(
        self, stack_trace: str, sensitivity_level: SensitivityLevel = SensitivityLevel.HIGH
    ) -> str:
        """
        Sanitize stack trace to remove sensitive information.

        Args:
            stack_trace: Stack trace to sanitize
            sensitivity_level: Security sensitivity level

        Returns:
            Sanitized stack trace
        """
        if not stack_trace:
            return stack_trace

        sanitized = stack_trace

        # First apply general sanitization
        sanitized = self.sanitize_error_message(sanitized, sensitivity_level)

        # Additional stack trace specific sanitization
        if self.config.sanitize_file_paths:
            # Sanitize file paths in stack trace
            sanitized = self._sanitize_stack_trace_paths(sanitized)

        # Remove sensitive variable values from stack trace
        sanitized = self._sanitize_variable_values(sanitized)

        return sanitized

    def sanitize_context(
        self, context: dict[str, Any], sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
    ) -> dict[str, Any]:
        """
        Sanitize error context dictionary.

        Args:
            context: Context dictionary to sanitize
            sensitivity_level: Security sensitivity level

        Returns:
            Sanitized context dictionary
        """
        if not context:
            return context

        sanitized_context = {}

        for key, value in context.items():
            # Sanitize key
            sanitized_key = self._sanitize_context_key(key, sensitivity_level)

            # Sanitize value with context awareness
            sanitized_value = self._sanitize_context_value_with_key(key, value, sensitivity_level)

            sanitized_context[sanitized_key] = sanitized_value

        return sanitized_context

    def _get_rules_for_level(self, level: SensitivityLevel) -> list[SanitizationRule]:
        """Get sanitization rules appropriate for sensitivity level."""

        if level == SensitivityLevel.LOW:
            # Only basic sanitization for low sensitivity
            return [
                rule
                for rule in self.rules
                if rule.name
                in [
                    "stripe_api_keys",
                    "standalone_api_keys",
                    "api_keys",
                    "generic_secrets",
                    "jwt_tokens",
                    "passwords",
                    "password_in_text",
                    "private_keys",
                ]
            ]
        elif level == SensitivityLevel.MEDIUM:
            # Standard sanitization - include all rules
            return self.rules
        elif level == SensitivityLevel.HIGH:
            # Comprehensive sanitization except internal paths
            return self.rules
        else:  # CRITICAL
            # Maximum sanitization including all rules
            return self.rules

    def _mask_value(self, value: str) -> str:
        """Mask a sensitive value."""
        if not value:
            return value

        if self.config.hash_sensitive_values:
            # Create hash of the value for debugging while maintaining security
            hash_obj = hashlib.new(self.config.hash_algorithm)
            hash_obj.update(value.encode("utf-8"))
            hash_digest = hash_obj.hexdigest()[:8]  # First 8 chars of hash
            return f"{self.config.hash_prefix}{hash_digest}"
        else:
            # Simple masking
            if len(value) <= 4:
                return self.config.mask_char * len(value)
            else:
                # Show first 2 and last 2 characters
                return f"{value[:2]}{self.config.mask_char * self.config.mask_length}{value[-2:]}"

    def _mask_crypto_address(self, address: str) -> str:
        """Mask cryptocurrency address while preserving format recognition."""
        if len(address) <= 8:
            return self.config.mask_char * len(address)
        return f"{address[:4]}{self.config.mask_char * 8}{address[-4:]}"

    def _mask_credit_card(self, card_number: str) -> str:
        """Mask credit card number."""
        digits_only = re.sub(r"\D", "", card_number)
        if len(digits_only) >= 4:
            return f"****-****-****-{digits_only[-4:]}"
        return "****-****-****-****"

    def _mask_email_user(self, user_part: str) -> str:
        """Mask email user part."""
        if len(user_part) <= 2:
            return self.config.mask_char * len(user_part)
        elif len(user_part) <= 4:
            return f"{user_part[0]}{self.config.mask_char * (len(user_part) - 1)}"
        else:
            return f"{user_part[0]}{self.config.mask_char * 3}{user_part[-1]}"

    def _mask_ip_address(self, ip: str) -> str:
        """Mask IP address."""
        if not self.config.mask_ip_addresses:
            return ip
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.*.* "
        return "*.*.*.* "

    def _sanitize_host(self, host: str) -> str:
        """Sanitize hostname/domain."""
        if "." in host:
            parts = host.split(".")
            if len(parts) >= 2:
                # Keep domain, mask subdomain
                return f"*.{'.'.join(parts[-2:])}"
        return self._mask_value(host)

    def _sanitize_file_path(self, path: str) -> str:
        """Sanitize file system path."""
        if not self.config.sanitize_file_paths:
            return path

        if self.config.show_relative_paths_only:
            # Convert to relative path format
            path_parts = path.replace("\\", "/").split("/")
            if len(path_parts) > 2:
                return f".../{'/'.join(path_parts[-2:])}"
            else:
                return "/".join(path_parts)
        else:
            # Mask directory structure
            path_parts = path.replace("\\", "/").split("/")
            if len(path_parts) > 1:
                masked_parts = ["***" for _ in path_parts[:-1]]
                masked_parts.append(path_parts[-1])  # Keep filename
                return "/".join(masked_parts)
            return path

    def _sanitize_stack_trace_paths(self, stack_trace: str) -> str:
        """Sanitize file paths in stack trace."""
        # Match common stack trace path patterns
        patterns = [
            r'File "([^"]+)"',  # Python style
            r"File '([^']+)'",  # Alternative Python style
            r"at ([^(]+)\(",  # JavaScript/Node.js style
            r"in ([^:]+):",  # General format
        ]

        sanitized = stack_trace
        for pattern in patterns:

            def replace_path(match):
                original_path = match.group(1)
                sanitized_path = self._sanitize_file_path(original_path)
                return match.group(0).replace(original_path, sanitized_path)

            sanitized = re.sub(pattern, replace_path, sanitized)

        return sanitized

    def _sanitize_variable_values(self, stack_trace: str) -> str:
        """Sanitize variable values in stack trace."""
        # Pattern for variable assignments in stack traces
        patterns = [
            r"(\w+)\s*=\s*'([^']*)'",  # var = 'value'
            r'(\w+)\s*=\s*"([^"]*)"',  # var = "value"
            r"(\w+)\s*=\s*([^,\s]+)",  # var = value
        ]

        sanitized = stack_trace
        for pattern in patterns:

            def replace_var(match):
                var_name = match.group(1)
                var_value = match.group(2) if len(match.groups()) > 1 else ""

                # Check if variable name suggests sensitive data
                sensitive_vars = [
                    "password",
                    "secret",
                    "key",
                    "token",
                    "api",
                    "auth",
                    "credential",
                    "private",
                    "wallet",
                    "seed",
                    "phrase",
                ]

                if any(sensitive in var_name.lower() for sensitive in sensitive_vars):
                    return f"{var_name} = {self._mask_value(var_value)}"
                elif len(var_value) > 20:  # Potentially sensitive long values
                    return f"{var_name} = {self._mask_value(var_value)}"

                return match.group(0)  # Return unchanged

            sanitized = re.sub(pattern, replace_var, sanitized)

        return sanitized

    def _sanitize_context_key(self, key: str, sensitivity_level: SensitivityLevel) -> str:
        """Sanitize context dictionary key."""
        # Generally, keys are safe to show as they indicate data structure
        # We preserve keys to maintain context structure readability
        # Only in extreme cases might we want to sanitize key names themselves
        return key

    def _sanitize_context_value_with_key(
        self, key: str, value: Any, sensitivity_level: SensitivityLevel
    ) -> Any:
        """Sanitize context dictionary value with awareness of the key name."""
        # Check if key suggests the value is sensitive
        sensitive_key_patterns = [
            "password",
            "pwd",
            "passwd",
            "pass",
            "secret",
            "key",
            "token",
            "auth",
            "credential",
            "private",
            "wallet",
            "signature",
            "seed",
            "phrase",
            "api",
        ]

        is_sensitive_key = any(pattern in key.lower() for pattern in sensitive_key_patterns)

        if isinstance(value, str):
            if is_sensitive_key:
                # For sensitive keys, always mask the value
                return self._mask_value(value)
            else:
                # For non-sensitive keys, apply normal sanitization
                return self.sanitize_error_message(value, sensitivity_level)
        elif isinstance(value, dict):
            return self.sanitize_context(value, sensitivity_level)
        elif isinstance(value, list | tuple):
            return type(value)(
                self._sanitize_context_value_with_key(f"{key}[{i}]", item, sensitivity_level)
                for i, item in enumerate(value)
            )
        else:
            # For non-string values, convert to string and sanitize if needed
            str_value = str(value)
            if is_sensitive_key or len(str_value) > 20:  # Sanitize if sensitive key or long string
                sanitized_str = self.sanitize_error_message(str_value, sensitivity_level)
                if sanitized_str != str_value:
                    return sanitized_str
            return value

    def _sanitize_context_value(self, value: Any, sensitivity_level: SensitivityLevel) -> Any:
        """Sanitize context dictionary value."""
        if isinstance(value, str):
            return self.sanitize_error_message(value, sensitivity_level)
        elif isinstance(value, dict):
            return self.sanitize_context(value, sensitivity_level)
        elif isinstance(value, list | tuple):
            return type(value)(
                self._sanitize_context_value(item, sensitivity_level) for item in value
            )
        else:
            # For non-string values, convert to string and sanitize if needed
            str_value = str(value)
            if len(str_value) > 20:  # Only sanitize long string representations
                sanitized_str = self.sanitize_error_message(str_value, sensitivity_level)
                if sanitized_str != str_value:
                    return sanitized_str
            return value

    def add_custom_rule(self, rule: SanitizationRule) -> None:
        """Add a custom sanitization rule."""
        self.rules.append(rule)
        self.logger.info(f"Added custom sanitization rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a sanitization rule by name."""
        original_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        removed = len(self.rules) < original_count

        if removed:
            self.logger.info(f"Removed sanitization rule: {rule_name}")
        else:
            self.logger.warning(f"Sanitization rule not found: {rule_name}")

        return removed

    def get_sanitization_stats(self) -> dict[str, Any]:
        """Get statistics about sanitization rules."""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules if r.enabled]),
            "rule_names": [r.name for r in self.rules],
            "config": {
                "hash_sensitive_values": self.config.hash_sensitive_values,
                "sanitize_file_paths": self.config.sanitize_file_paths,
                "mask_ip_addresses": self.config.mask_ip_addresses,
            },
        }


# Singleton instance for global use
_global_sanitizer = None


def get_security_sanitizer(config: SanitizationConfig | None = None) -> SecurityDataSanitizer:
    """Get global security sanitizer instance."""
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = SecurityDataSanitizer(config)
    return _global_sanitizer


def sanitize_for_logging(
    message: str, sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM
) -> str:
    """Convenience function for sanitizing messages for logging."""
    sanitizer = get_security_sanitizer()
    return sanitizer.sanitize_error_message(message, sensitivity_level)


def sanitize_stack_trace_for_logging(
    stack_trace: str, sensitivity_level: SensitivityLevel = SensitivityLevel.HIGH
) -> str:
    """Convenience function for sanitizing stack traces for logging."""
    sanitizer = get_security_sanitizer()
    return sanitizer.sanitize_stack_trace(stack_trace, sensitivity_level)
