"""
Data Module Configuration Constants

This module contains production-ready configuration constants for the data module,
eliminating hardcoded magic numbers and improving maintainability.
"""

# Cache Configuration Constants
DEFAULT_L1_CACHE_MAX_SIZE = 1000
DEFAULT_L1_CACHE_TTL_SECONDS = 300  # 5 minutes
DEFAULT_L2_CACHE_TTL_SECONDS = 3600  # 1 hour

# Data Request Limits
DEFAULT_DATA_LIMIT = 100
MAX_DATA_LIMIT = 10000
MIN_DATA_LIMIT = 1

# Cache TTL Limits
MIN_CACHE_TTL_SECONDS = 1
MAX_CACHE_TTL_SECONDS = 86400  # 24 hours

# String Length Limits
MIN_SYMBOL_LENGTH = 1
MAX_SYMBOL_LENGTH = 20
MIN_EXCHANGE_LENGTH = 1
MAX_EXCHANGE_LENGTH = 50

# Lookback Period Limits
MIN_LOOKBACK_PERIOD = 1
MAX_LOOKBACK_PERIOD = 1000

# Performance Constants
HIGH_THROUGHPUT_THRESHOLD = 10000  # messages/second

# Connection Timeouts
EXCHANGE_CONNECTION_TIMEOUT_SECONDS = 30.0
STREAM_TASK_CLEANUP_TIMEOUT_SECONDS = 10.0

# Error Recovery Constants
STREAM_ERROR_DELAY_SECONDS = 5

# Default Data Intervals
DEFAULT_DATA_INTERVAL = "1m"

# Quality Score Constants
DEFAULT_QUALITY_SCORE = "1.0"

# Default Data Source
DEFAULT_DATA_SOURCE = "exchange"

# Default Exchange
DEFAULT_EXCHANGE = "binance"

# Retry Configuration
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0

# Validation Constants
DEFAULT_VALIDATION_STATUS = "valid"

# Data Processing Constants
BATCH_SIZE_DEFAULT = 100
BULK_OPERATION_SIZE = 1000

# Redis Configuration Defaults
REDIS_DEFAULT_HOST = "localhost"
REDIS_DEFAULT_PORT = 6379
REDIS_DEFAULT_DB = 0

# InfluxDB Configuration Defaults
INFLUXDB_DEFAULT_URL = "http://localhost:8086"
INFLUXDB_DEFAULT_ORG = "trading-bot"
INFLUXDB_DEFAULT_BUCKET = "market-data"
