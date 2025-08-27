"""
Database package for the trading bot framework.

This module provides database models, connection management, and utilities
for PostgreSQL, Redis, and InfluxDB integration.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

from sqlalchemy.exc import (
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLTimeoutError,
)

from .connection import get_async_session, get_sync_session

# Expose wrapper with a consistent name
from .influxdb_client import InfluxDBClientWrapper as InfluxDBClient
from .manager import DatabaseManager

# Explicit imports from models to avoid star import issues
from .models import (
    Base,
    Bot,
    BotLog,
    CapitalAuditLog,
    ExecutionAuditLog,
    MarketDataRecord,
    Order,
    OrderFill,
    PerformanceAuditLog,
    Position,
    RiskAuditLog,
    Signal,
    Strategy,
    Trade,
)
from .redis_client import RedisClient
from .service import DatabaseService

__all__ = [
    # Models and classes
    "Base",
    "Bot",
    "BotLog",
    "CapitalAuditLog",
    "DatabaseManager",
    "DatabaseService",
    "ExecutionAuditLog",
    "InfluxDBClient",
    "IntegrityError",
    "MarketDataRecord",
    "OperationalError",
    "Order",
    "OrderFill",
    "PerformanceAuditLog",
    "Position",
    "RedisClient",
    "RiskAuditLog",
    "SQLAlchemyError",
    "SQLTimeoutError",
    "Signal",
    "Strategy",
    "Trade",
    # Functions
    "get_async_session",
    "get_sync_session",
]
