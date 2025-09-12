"""
Common constants for state management operations.

This module provides shared constants and default values to eliminate
duplication across state components.
"""

from decimal import Decimal

# Cache configuration
DEFAULT_CACHE_TTL = 300  # 5 minutes
DEFAULT_CACHE_TIMEOUT = 2.0  # 2 seconds
STATE_CACHE_PREFIX = "state"
CHECKPOINT_CACHE_PREFIX = "checkpoint"
METADATA_CACHE_PREFIX = "meta"

# Checkpoint configuration
DEFAULT_COMPRESSION_THRESHOLD = 1024  # 1KB
DEFAULT_MAX_CHECKPOINTS = 50
DEFAULT_CHECKPOINT_INTERVAL = 1800  # 30 minutes
DEFAULT_CLEANUP_INTERVAL = 3600  # 1 hour
CHECKPOINT_FILE_EXTENSION = ".checkpoint"
CHECKPOINT_METADATA_EXTENSION = ".meta"

# Validation constants
MIN_CASH_RATIO = Decimal("0.1")  # 10% minimum cash
MAX_VAR_LIMIT = Decimal("0.02")  # 2% maximum VaR
MAX_RISK_PER_TRADE = Decimal("0.05")  # 5% maximum risk per trade
MAX_CAPITAL_ALLOCATION = Decimal("1000000")  # $1M default maximum

# State field requirements
BOT_STATE_REQUIRED_FIELDS = {"bot_id", "status", "created_at", "updated_at"}

POSITION_STATE_REQUIRED_FIELDS = {"symbol", "quantity", "side", "entry_price"}

ORDER_STATE_REQUIRED_FIELDS = {"order_id", "symbol", "quantity", "side", "type"}

RISK_STATE_REQUIRED_FIELDS = {"max_position_size", "max_drawdown", "var_limit"}

# Decimal fields that need special validation
BOT_STATE_DECIMAL_FIELDS = {
    "current_capital",
    "allocated_capital",
    "unrealized_pnl",
    "realized_pnl",
}

POSITION_STATE_DECIMAL_FIELDS = {"quantity", "entry_price", "current_price", "unrealized_pnl"}

ORDER_STATE_DECIMAL_FIELDS = {"quantity", "price", "filled_quantity", "remaining_quantity"}

RISK_STATE_DECIMAL_FIELDS = {"max_position_size", "max_drawdown", "var_limit", "current_var"}

# Positive value fields
POSITIVE_VALUE_FIELDS = {
    "current_capital",
    "allocated_capital",
    "entry_price",
    "current_price",
    "price",
    "quantity",
    "max_position_size",
}

# Non-negative value fields (can be zero)
NON_NEGATIVE_VALUE_FIELDS = {
    "filled_quantity",
    "remaining_quantity",
    "realized_pnl",
    "unrealized_pnl",
    "current_var",
}

# State transition rules
BOT_STATUS_TRANSITIONS = {
    "CREATED": {"INITIALIZING", "ERROR"},
    "INITIALIZING": {"RUNNING", "ERROR", "STOPPED"},
    "RUNNING": {"PAUSED", "STOPPING", "ERROR"},
    "PAUSED": {"RUNNING", "STOPPING", "ERROR"},
    "STOPPING": {"STOPPED", "ERROR"},
    "STOPPED": {"INITIALIZING", "ERROR"},
    "ERROR": {"INITIALIZING", "STOPPED"},
}

ORDER_STATUS_TRANSITIONS = {
    "CREATED": {"PENDING", "CANCELLED"},
    "PENDING": {"PARTIALLY_FILLED", "FILLED", "CANCELLED", "REJECTED"},
    "PARTIALLY_FILLED": {"FILLED", "CANCELLED"},
    "FILLED": set(),  # Terminal state
    "CANCELLED": set(),  # Terminal state
    "REJECTED": set(),  # Terminal state
}

POSITION_STATUS_TRANSITIONS = {
    "OPENED": {"MODIFIED", "CLOSING", "CLOSED"},
    "MODIFIED": {"CLOSING", "CLOSED"},
    "CLOSING": {"CLOSED"},
    "CLOSED": set(),  # Terminal state
}

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 0.5
DEFAULT_RETRY_BACKOFF = 2.0

# Trade lifecycle configuration
DEFAULT_TRADE_STALENESS_THRESHOLD = 3600  # 1 hour

# Performance monitoring thresholds
MEMORY_WARNING_THRESHOLD = 1000  # MB
MEMORY_CRITICAL_THRESHOLD = 2000  # MB
OPERATION_TIMEOUT_WARNING = 1000  # ms
OPERATION_TIMEOUT_CRITICAL = 5000  # ms

# Compression ratios
MIN_COMPRESSION_RATIO = 0.7  # Only compress if >= 30% reduction
EXCELLENT_COMPRESSION_RATIO = 0.5  # 50% or better compression

# State synchronization
SYNC_BATCH_SIZE = 100
SYNC_TIMEOUT = 10.0  # seconds
SYNC_RETRY_LIMIT = 3
CONFLICT_RESOLUTION_TIMEOUT = 5.0  # seconds

# Health check configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 10  # seconds
HEALTH_CHECK_FAILURE_THRESHOLD = 3

# Metric collection
METRICS_RETENTION_HOURS = 24
METRICS_COLLECTION_INTERVAL = 60  # seconds
ALERT_COOLDOWN_MINUTES = 15

# File system
MAX_LOG_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_CHECKPOINT_FILE_SIZE = 50 * 1024 * 1024  # 50MB
BACKUP_RETENTION_DAYS = 30

# Pattern matching
BOT_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
SYMBOL_PATTERN = r"^[A-Z0-9]+([\/\-][A-Z0-9]+)*$"
ORDER_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"

# Error handling
MAX_ERROR_DETAILS_LENGTH = 1000
MAX_STACK_TRACE_LENGTH = 5000
ERROR_CONTEXT_TTL = 3600  # 1 hour

# Database configuration
DB_QUERY_TIMEOUT = 30.0  # seconds
DB_CONNECTION_TIMEOUT = 10.0  # seconds
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 0

# Redis configuration
REDIS_KEY_EXPIRE_TIME = 3600  # 1 hour
REDIS_LOCK_TIMEOUT = 10.0  # seconds
REDIS_PIPELINE_SIZE = 1000

# State size limits
MAX_STATE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_METADATA_SIZE_BYTES = 1024 * 1024  # 1MB
MAX_CHANGE_LOG_ENTRIES = 1000

# Validation error types
VALIDATION_ERROR_TYPES = {
    "MISSING_REQUIRED_FIELD",
    "INVALID_TYPE",
    "INVALID_FORMAT",
    "OUT_OF_RANGE",
    "BUSINESS_RULE_VIOLATION",
    "CONSTRAINT_VIOLATION",
}

# State event types
STATE_EVENT_TYPES = {
    "STATE_CREATED",
    "STATE_UPDATED",
    "STATE_DELETED",
    "STATE_VALIDATED",
    "STATE_SYNCHRONIZED",
    "STATE_PERSISTED",
    "STATE_ERROR",
}

# Priority levels
PRIORITY_LEVELS = {"LOW": 1, "NORMAL": 2, "HIGH": 3, "CRITICAL": 4}

# Resource limits
MAX_CONCURRENT_OPERATIONS = 100
MAX_MEMORY_USAGE_MB = 2048
MAX_CPU_USAGE_PERCENT = 80.0

# State types registry
STATE_TYPES = {
    "bot_state",
    "position_state",
    "order_state",
    "risk_state",
    "portfolio_state",
    "execution_state",
    "strategy_state",
}

# Default timeouts for different operations
OPERATION_TIMEOUTS = {
    "validate": 5.0,
    "persist": 10.0,
    "synchronize": 15.0,
    "checkpoint": 30.0,
    "restore": 60.0,
}

# Alert severities mapped to numeric levels
ALERT_SEVERITY_LEVELS = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# Date format strings
ISO_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
SHORT_DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Default configuration values
DEFAULT_CONFIG = {
    "cache_ttl": DEFAULT_CACHE_TTL,
    "cache_timeout": DEFAULT_CACHE_TIMEOUT,
    "compression_threshold": DEFAULT_COMPRESSION_THRESHOLD,
    "max_checkpoints": DEFAULT_MAX_CHECKPOINTS,
    "checkpoint_interval": DEFAULT_CHECKPOINT_INTERVAL,
    "cleanup_interval": DEFAULT_CLEANUP_INTERVAL,
    "max_retries": DEFAULT_MAX_RETRIES,
    "retry_delay": DEFAULT_RETRY_DELAY,
    "health_check_interval": HEALTH_CHECK_INTERVAL,
    "metrics_collection_interval": METRICS_COLLECTION_INTERVAL,
    "sync_timeout": SYNC_TIMEOUT,
    "db_query_timeout": DB_QUERY_TIMEOUT,
    "redis_key_expire_time": REDIS_KEY_EXPIRE_TIME,
}
