"""
Configuration constants for the monitoring module.

This module contains all configuration constants used across the monitoring
system to ensure consistent behavior and easy maintenance.
"""

from typing import Final

# Alert Manager Configuration
ALERT_HISTORY_MAX_SIZE: Final[int] = 10000
ALERT_PROCESSING_CHECK_INTERVAL: Final[int] = 10  # seconds
ALERT_NOTIFICATION_TIMEOUT: Final[int] = 30  # seconds
ALERT_ESCALATION_CHECK_TIMEOUT: Final[int] = 15  # seconds
ALERT_BACKGROUND_TASK_TIMEOUT: Final[int] = 5  # seconds

# Retry Configuration
ALERT_RETRY_MAX_ATTEMPTS: Final[int] = 3
ALERT_RETRY_DELAY: Final[float] = 1.5  # seconds
EMAIL_RETRY_MAX_ATTEMPTS: Final[int] = 3
EMAIL_RETRY_BASE_DELAY: Final[float] = 1.0  # seconds
WEBHOOK_RETRY_MAX_ATTEMPTS: Final[int] = 3
WEBHOOK_RETRY_BASE_DELAY: Final[float] = 1.0  # seconds

# Network Configuration
HTTP_SESSION_TIMEOUT: Final[int] = 10  # seconds
HTTP_CONNECTOR_LIMIT: Final[int] = 10
HTTP_CONNECTOR_LIMIT_PER_HOST: Final[int] = 5
HTTP_SERVER_ERROR_THRESHOLD: Final[int] = 500
SESSION_CLOSE_TIMEOUT: Final[float] = 5.0  # seconds
SESSION_CLEANUP_TIMEOUT: Final[float] = 2.0  # seconds

# Notification Colors
SEVERITY_COLORS = {
    "critical": "#FF0000",  # Red
    "high": "#FF8000",  # Orange
    "medium": "#FFFF00",  # Yellow
    "low": "#0080FF",  # Blue
    "info": "#00FF00",  # Green
    "default": "#808080",  # Gray
    "success": "#00FF00",  # Green
}

# Discord Color Codes (hex values for embeds)
DISCORD_SEVERITY_COLORS = {
    "critical": 0xFF0000,  # Red
    "high": 0xFF8000,  # Orange
    "medium": 0xFFFF00,  # Yellow
    "low": 0x0080FF,  # Blue
    "info": 0x00FF00,  # Green
    "default": 0x808080,  # Gray
    "success": 0x00FF00,  # Green
}

# Duration Parsing Configuration
DURATION_PARSE_MINIMUM_MINUTES: Final[int] = 1
HOURS_PER_DAY: Final[int] = 24  # For financial markets
DAYS_PER_WEEK: Final[int] = 7
MINUTES_PER_HOUR: Final[int] = 60
SECONDS_PER_MINUTE: Final[int] = 60

# Metrics Configuration
METRICS_FALLBACK_STORAGE_LIMIT: Final[int] = 1000
METRICS_DEFAULT_PROMETHEUS_PORT: Final[int] = 8001
SYSTEM_METRICS_COLLECTION_INTERVAL: Final[int] = 60  # seconds

# Performance Configuration
CPU_HIGH_THRESHOLD: Final[float] = 80.0  # percent
MEMORY_HIGH_THRESHOLD: Final[float] = 85.0  # percent
DISK_HIGH_THRESHOLD: Final[float] = 90.0  # percent

# Financial Precision Configuration
DECIMAL_PRECISION: Final[int] = 8
MAX_FINANCIAL_VALUE: Final[float] = 1e12
MIN_FINANCIAL_VALUE: Final[float] = 1e-8

# Default Notification Configuration
DEFAULT_EMAIL_SMTP_PORT: Final[int] = 587
DEFAULT_WEBHOOK_TIMEOUT: Final[int] = 10  # seconds
DEFAULT_ESCALATION_MAX: Final[int] = 3

# Service Health Configuration
HEALTH_CHECK_TIMEOUT: Final[float] = 5.0  # seconds
SERVICE_START_TIMEOUT: Final[float] = 10.0  # seconds
SERVICE_STOP_TIMEOUT: Final[float] = 10.0  # seconds

# WebSocket Configuration
WEBSOCKET_CLOSE_TIMEOUT: Final[float] = 2.0  # seconds
WEBSOCKET_CONNECTION_TIMEOUT: Final[float] = 0.1  # seconds
WEBSOCKET_RESPONSE_SIMULATION: Final[float] = 0.01  # seconds
WEBSOCKET_SEND_SIMULATION: Final[float] = 0.001  # seconds

# Performance Profiler Configuration
PERFORMANCE_GRACEFUL_SHUTDOWN_TIMEOUT: Final[float] = 30.0  # seconds
PERFORMANCE_FORCE_SHUTDOWN_TIMEOUT: Final[float] = 10.0  # seconds
PERFORMANCE_ALERT_TASK_TIMEOUT: Final[float] = 5.0  # seconds
PERFORMANCE_ERROR_RECOVERY_SLEEP: Final[float] = 5.0  # seconds

# Dashboard Configuration
DASHBOARD_HTTP_TIMEOUT: Final[int] = 30  # seconds
DASHBOARD_CONNECTOR_LIMIT: Final[int] = 10
DASHBOARD_CONNECTOR_LIMIT_PER_HOST: Final[int] = 5
DASHBOARD_RETRY_SLEEP_BASE: Final[float] = 0.5  # seconds

# Default Service URLs
DEFAULT_GRAFANA_URL: Final[str] = "http://localhost:3000"

# HTTP Response Status Codes
HTTP_OK: Final[int] = 200
HTTP_NO_CONTENT: Final[int] = 204
