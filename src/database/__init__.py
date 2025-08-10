"""
Database package for the trading bot framework.

This module provides database models, connection management, and utilities
for PostgreSQL, Redis, and InfluxDB integration.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

from .models import Base, User, BotInstance, Trade, Position, BalanceSnapshot
from .models import StrategyConfig, MLModel, PerformanceMetrics, Alert, AuditLog
from .connection import get_sync_session, get_async_session
from .redis_client import RedisClient
from .influxdb_client import InfluxDBClient

__all__ = [
    # Models
    'Base', 'User', 'BotInstance', 'Trade', 'Position', 'BalanceSnapshot',
    'StrategyConfig', 'MLModel', 'PerformanceMetrics', 'Alert', 'AuditLog',

    # Connection management
    'get_sync_session', 'get_async_session',

    # Clients
    'RedisClient', 'InfluxDBClient',
]
