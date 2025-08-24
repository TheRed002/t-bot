"""
Database package for the trading bot framework.

This module provides database models, connection management, and utilities
for PostgreSQL, Redis, and InfluxDB integration.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

from .connection import get_async_session, get_sync_session

# Expose wrapper with a consistent name
from .influxdb_client import InfluxDBClientWrapper as InfluxDBClient
from .models import *  # noqa: F403
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as SQLTimeoutError
from .redis_client import RedisClient

__all__ = [
    # Models
    "Base",
    # Bot models
    "Bot",
    "BotLog",
    # Audit models
    "CapitalAuditLog",
    "ExecutionAuditLog",
    # Clients
    "InfluxDBClient",
    # Market data models
    "MarketDataRecord",
    # Trading models
    "Order",
    "OrderFill",
    "PerformanceAuditLog",
    "Position",
    "RedisClient",
    "RiskAuditLog",
    "Signal",
    "Strategy",
    "Trade",
    # Database exceptions
    "IntegrityError",
    "OperationalError",
    "SQLAlchemyError",
    "SQLTimeoutError",
    # Connection management
    "get_async_session",
    "get_sync_session",
]
