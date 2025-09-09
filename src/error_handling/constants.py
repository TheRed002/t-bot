"""
Constants for error handling module.

Centralized constants for error handling, recovery scenarios,
and security configurations.
"""

from decimal import Decimal

# Validation constants
STRING_TRUNCATION_LIMIT = 100  # characters
DEFAULT_TIMEOUT = 60.0  # seconds
DEFAULT_COMPONENT_TIMEOUT = 60.0  # seconds

# State monitoring constants
DEFAULT_STATE_VALIDATION_FREQUENCY = 60  # seconds
MAX_VALIDATION_HISTORY = 1000  # number of validation results to keep
DEFAULT_MAX_DAILY_LOSS = 1000  # default risk limit

# Recovery scenario constants
DEFAULT_MAINTENANCE_CHECK_INTERVAL = 300  # seconds (5 minutes)
DEFAULT_NETWORK_MAX_OFFLINE_DURATION = 300  # seconds (5 minutes)
DEFAULT_CACHE_EXPIRY = 3600  # seconds (1 hour)
DEFAULT_DATA_FEED_MAX_STALENESS = 60  # seconds

# Network error constants
NETWORK_ERROR_MAX_ATTEMPTS = 5
API_RATE_LIMIT_MAX_ATTEMPTS = 3

# Rate limiting constants
DEFAULT_REQUESTS_PER_SECOND = 10
DEFAULT_REQUESTS_PER_MINUTE = 600
DEFAULT_REQUESTS_PER_HOUR = 3600
DEFAULT_BURST_ALLOWANCE = 5

# Security constants
SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "key",
    "token",
    "auth",
    "credential",
    "api_key",
    "private_key",
    "session_id",
}

# Decimal constants for precision
TOLERANCE_DECIMAL = Decimal("0.00000001")
BALANCE_TOLERANCE = Decimal("0.00000001")
POSITION_TOLERANCE = Decimal("0.00000001")

# Severity thresholds
CRITICAL_THRESHOLD = Decimal("1.0")
HIGH_THRESHOLD = Decimal("0.1")
MEDIUM_THRESHOLD = Decimal("0.01")

# Recovery configuration defaults
PARTIAL_FILL_MIN_PERCENTAGE = Decimal("0.1")
CONSERVATIVE_STOP_ADJUSTMENT = Decimal("0.8")  # Tighten by 20%
CONSERVATIVE_DATA_STOP_ADJUSTMENT = Decimal("0.95")  # Tighten by 5%

# Order adjustment factors
PRICE_ADJUSTMENT_DOWN = Decimal("0.999")  # 0.1% down
PRICE_ADJUSTMENT_UP = Decimal("1.001")  # 0.1% up
QUANTITY_ADJUSTMENT = Decimal("0.9")  # 10% reduction
