
"""
Shared error categorization utilities for error handling system.

This module provides common keyword matching patterns and categorization logic
to eliminate duplication across error handling modules.
"""

# Error severity keywords
CRITICAL_KEYWORDS = ["critical", "fatal", "security"]
ERROR_KEYWORDS = ["error", "failed", "exception"]
WARNING_KEYWORDS = ["warning", "warn"]

# Error category keywords
AUTH_KEYWORDS = ["auth", "permission", "forbidden"]
AUTH_MESSAGE_KEYWORDS = ["permission", "forbidden", "unauthorized"]
VALIDATION_KEYWORDS = ["validation", "value", "type"]
VALIDATION_MESSAGE_KEYWORDS = ["invalid", "required", "format"]
NETWORK_KEYWORDS = ["connection", "network", "timeout"]
DATABASE_KEYWORDS = ["database", "sql", "operational"]
DATABASE_MESSAGE_KEYWORDS = ["database", "connection pool", "sql"]
EXCHANGE_KEYWORDS = ["exchange", "trading", "order"]
EXCHANGE_MESSAGE_KEYWORDS = ["exchange", "trading", "market"]
RATE_LIMIT_KEYWORDS = ["rate limit", "too many", "429", "api throttled", "throttled"]
MAINTENANCE_KEYWORDS = ["maintenance", "unavailable", "503"]
SYSTEM_KEYWORDS = ["system", "os", "memory"]

# Financial component keywords
FINANCIAL_COMPONENTS = ["trading", "exchange", "wallet", "payment", "order", "position"]

# Data type keywords for fallback values
LIST_KEYWORDS = ["list", "array", "items", "elements"]
DICT_KEYWORDS = ["dict", "map", "mapping", "config"]
SET_KEYWORDS = ["set", "unique"]
TUPLE_KEYWORDS = ["tuple", "pair"]
STRING_KEYWORDS = ["str", "string", "text"]
INT_KEYWORDS = ["int", "count", "num", "id"]
BOOL_KEYWORDS = ["bool", "is_", "has_", "can_"]

# Sensitive key patterns for sanitization
SENSITIVE_KEY_PATTERNS = [
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

# Authentication detection keywords
AUTH_TOKEN_KEYWORDS = ["token", "bearer", "jwt", "authorization"]

# Rate limiting keywords
RATE_LIMIT_DETECTION_KEYWORDS = [
    "rate limit",
    "rate-limit",
    "ratelimit",
    "429",
    "too many requests",
    "throttle",
    "throttled",
    "api throttled",
]

# Data validation keywords
DATA_VALIDATION_KEYWORDS = ["data", "format", "structure", "schema", "model"]


def contains_keywords(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the given keywords (case-insensitive)."""
    if not text or not keywords:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def categorize_by_keywords(text: str, keyword_map: dict[str, list[str]]) -> str | None:
    """Categorize text based on keyword matches. Returns first matching category or None."""
    if not text:
        return None

    text_lower = text.lower()
    for category, keywords in keyword_map.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return None


def get_error_category_keywords() -> dict[str, list[str]]:
    """Get mapping of error categories to their keyword lists."""
    return {
        "authentication": AUTH_KEYWORDS,
        "validation": VALIDATION_KEYWORDS,
        "network": NETWORK_KEYWORDS,
        "database": DATABASE_KEYWORDS,
        "exchange": EXCHANGE_KEYWORDS,
        "system": SYSTEM_KEYWORDS,
    }


def get_fallback_type_keywords() -> dict[str, list[str]]:
    """Get mapping of fallback data types to their keyword lists."""
    return {
        "list": LIST_KEYWORDS,
        "dict": DICT_KEYWORDS,
        "set": SET_KEYWORDS,
        "tuple": TUPLE_KEYWORDS,
        "str": STRING_KEYWORDS,
        "int": INT_KEYWORDS,
        "bool": BOOL_KEYWORDS,
    }


def is_financial_component(component_name: str) -> bool:
    """Check if component is financial-related."""
    return contains_keywords(component_name, FINANCIAL_COMPONENTS)


def is_sensitive_key(key_name: str) -> bool:
    """Check if key name indicates sensitive data."""
    return contains_keywords(key_name, SENSITIVE_KEY_PATTERNS)


def categorize_error_from_type_and_message(error_type: str, error_message: str) -> str:
    """
    Categorize error based on both type and message.

    This replaces the duplicated _categorize_error logic found in multiple modules.
    """
    error_type_lower = error_type.lower()
    error_message_lower = error_message.lower()

    # Authentication/Authorization errors
    if contains_keywords(error_type_lower, AUTH_KEYWORDS):
        return "authentication" if "auth" in error_type_lower else "authorization"
    if contains_keywords(error_message_lower, AUTH_MESSAGE_KEYWORDS):
        return "authorization"

    # Validation errors
    if contains_keywords(error_type_lower, VALIDATION_KEYWORDS):
        return "validation"
    if contains_keywords(error_message_lower, VALIDATION_MESSAGE_KEYWORDS):
        return "validation"

    # Database errors (check before network to catch "database connection" before "connection")
    if contains_keywords(error_type_lower, DATABASE_KEYWORDS):
        return "database"
    if contains_keywords(error_message_lower, DATABASE_MESSAGE_KEYWORDS):
        return "database"

    # Network errors
    if contains_keywords(error_type_lower, NETWORK_KEYWORDS):
        return "network"
    if contains_keywords(error_message_lower, NETWORK_KEYWORDS):
        return "network"

    # Exchange errors
    if contains_keywords(error_type_lower, EXCHANGE_KEYWORDS):
        return "exchange"
    if contains_keywords(error_message_lower, EXCHANGE_MESSAGE_KEYWORDS):
        return "exchange"

    # Rate limiting
    if contains_keywords(error_message_lower, RATE_LIMIT_KEYWORDS):
        return "rate_limit"

    # Maintenance
    if contains_keywords(error_message_lower, MAINTENANCE_KEYWORDS):
        return "maintenance"

    # System errors
    if contains_keywords(error_type_lower, SYSTEM_KEYWORDS):
        return "internal"

    return "unknown"


def categorize_error_from_message(message: str) -> str:
    """
    Categorize error based on message content only.

    This replaces the duplicated categorization logic in secure_reporting.py.
    """
    if contains_keywords(message, AUTH_KEYWORDS):
        return "authentication"
    elif contains_keywords(message, ["validation", "invalid"]):
        return "validation"
    elif contains_keywords(message, ["database", "sql"]):
        return "database"
    elif contains_keywords(message, NETWORK_KEYWORDS):
        return "network"
    elif contains_keywords(message, ["exchange", "trading"]):
        return "exchange"
    else:
        return "unknown"


def determine_alert_severity_from_message(message: str) -> str:
    """
    Determine alert severity based on message content.

    This replaces the duplicated severity determination logic.
    Returns: 'critical', 'high', 'medium', or 'info'
    """
    if contains_keywords(message, CRITICAL_KEYWORDS):
        return "critical"
    elif contains_keywords(message, ERROR_KEYWORDS):
        return "high"
    elif contains_keywords(message, WARNING_KEYWORDS):
        return "medium"
    else:
        return "info"


def is_retryable_error(error_message: str) -> bool:
    """Determine if an error is retryable based on common patterns."""
    non_retryable_keywords = [
        "authentication",
        "authorization",
        "permission",
        "forbidden",
        "invalid",
        "malformed",
        "validation",
        "syntax",
        "format",
    ]
    return not contains_keywords(error_message, non_retryable_keywords)


def detect_rate_limiting(error_message: str) -> bool:
    """Detect if error indicates rate limiting."""
    return contains_keywords(error_message, RATE_LIMIT_DETECTION_KEYWORDS)


def detect_auth_token_error(error_message: str) -> bool:
    """Detect if error is related to authentication tokens."""
    return contains_keywords(error_message, AUTH_TOKEN_KEYWORDS)


def detect_data_validation_error(error_message: str) -> bool:
    """Detect if error is related to data validation."""
    return contains_keywords(error_message, DATA_VALIDATION_KEYWORDS)
