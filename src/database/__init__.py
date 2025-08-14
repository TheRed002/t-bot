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
    CapitalAllocationDB,
    CurrencyExposureDB,
    DataPipelineRecord,
    DataQualityRecord,
    ExchangeAllocationDB,
    FeatureRecord,
    FundFlowDB,
    MarketDataRecord,
    MLModel,
    PerformanceMetrics,
    Position,
    StrategyConfig,
    Trade,
    User,
)
from .redis_client import RedisClient

__all__ = [
    "Alert",
    "AuditLog",
    "BalanceSnapshot",
    # Models
    "Base",
    "BotInstance",
    "CapitalAllocationDB",
    "CurrencyExposureDB",
    "DataPipelineRecord",
    "DataQualityRecord",
    "ExchangeAllocationDB",
    "FeatureRecord",
    "FundFlowDB",
    "InfluxDBClient",
    "MarketDataRecord",
    "MLModel",
    "PerformanceMetrics",
    "Position",
    # Clients
    "RedisClient",
    "StrategyConfig",
    "Trade",
    "User",
    "get_async_session",
    # Connection management
    "get_sync_session",
]
