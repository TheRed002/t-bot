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
from .models import (
    Alert,
    AuditLog,
    BalanceSnapshot,
    Base,
    BotInstance,
    MLModel,
    PerformanceMetrics,
    Position,
    StrategyConfig,
    Trade,
    User,
)
from .redis_client import RedisClient

__all__ = [
    # Models
    "Base",
    "User",
    "BotInstance",
    "Trade",
    "Position",
    "BalanceSnapshot",
    "StrategyConfig",
    "MLModel",
    "PerformanceMetrics",
    "Alert",
    "AuditLog",
    # Connection management
    "get_sync_session",
    "get_async_session",
    # Clients
    "RedisClient",
    "InfluxDBClient",
]
