"""
Constants for the web interface module.

This module centralizes all magic numbers and configuration constants
used throughout the web interface to improve maintainability.
"""

# Time constants (in seconds)
DEFAULT_JWT_EXPIRE_MINUTES = 30
DEFAULT_SESSION_TIMEOUT_MINUTES = 60
DEFAULT_PING_TIMEOUT = 30
DEFAULT_WEBSOCKET_TIMEOUT = 300
DEFAULT_SHUTDOWN_TIMEOUT = 30
DEFAULT_CONNECTION_TIMEOUT = 10
DEFAULT_POOL_TIMEOUT = 30
DEFAULT_IDLE_TIMEOUT = 300  # 5 minutes
DEFAULT_CONNECTION_LIFETIME = 3600  # 1 hour
DEFAULT_HEALTH_CHECK_INTERVAL = 30
RATE_LIMIT_WINDOW_SIZE = 60
RATE_LIMIT_CLEANUP_INTERVAL = 300  # 5 minutes
DEFAULT_RETRY_AFTER = 60  # 1 minute
HEALTH_CHECK_INTERVAL = 60  # 1 minute

# Rate limiting constants
ANONYMOUS_RATE_LIMIT = 60
AUTHENTICATED_RATE_LIMIT = 300
TRADER_RATE_LIMIT = 600
ADMIN_RATE_LIMIT = 300
BOT_ACTION_RATE_LIMIT = 30
TRADING_RATE_LIMIT = 30

# Network constants
DEFAULT_PORT = 8000
LOCAL_HOST = "localhost"
LOCAL_PORT_WEB = 3000
LOCAL_PORT_API = 8000

# Cache constants
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 60

# Connection limits
MAX_CONNECTIONS_PER_IP = 10

# Security constants
MAX_LOGIN_ATTEMPTS = 5
ACCOUNT_LOCK_DURATION_MINUTES = 30
HSTS_MAX_AGE = 31536000  # 1 year
SANITIZED_STRING_MAX_LENGTH = 30

# Mock data constants (for development/testing)
MOCK_BTC_PRICE = "45000.00"
MOCK_ETH_PRICE = "3000.00"
MOCK_DEFAULT_PRICE = "500.00"
MOCK_PORTFOLIO_VALUE = "10000.00"
MOCK_AVAILABLE_BALANCE = "5000.00"
MOCK_DATA_LIMIT = 30

# API versioning
SUPPORTED_API_VERSIONS = ["v1", "v1.1", "v2"]

# Development URLs
DEV_ORIGINS = ["http://localhost:3000", "http://testserver", "http://localhost:8000"]

# File and data limits
MAX_QUERY_DAYS = 365
MIN_QUERY_DAYS = 1
DEFAULT_QUERY_DAYS = 30
LOOKBACK_PERIOD_DAYS = 30
