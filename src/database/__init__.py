"""
Database package for the trading bot framework.

This module provides database models, connection management, and utilities
for PostgreSQL, Redis, and InfluxDB integration.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

# Use core exceptions instead of SQLAlchemy directly
from src.core.exceptions import DatabaseConnectionError, DatabaseError, DatabaseQueryError

from .connection import DatabaseConnectionManager, get_async_session, get_sync_session
from .di_registration import (
    configure_database_dependencies,
    get_database_manager,
    get_database_service,
    get_uow_factory,
    register_database_services,
)

# Expose wrapper with a consistent name
from .influxdb_client import InfluxDBClientWrapper as InfluxDBClient
from .manager import DatabaseManager


# Lazy import of models to prevent circular import issues
def _import_models():
    """Import models only when needed to prevent circular imports."""
    try:
        from .models import (
            Base,
            Bot,
            BotInstance,
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
            User,
        )

        return {
            "Base": Base,
            "Bot": Bot,
            "BotInstance": BotInstance,
            "BotLog": BotLog,
            "CapitalAuditLog": CapitalAuditLog,
            "ExecutionAuditLog": ExecutionAuditLog,
            "MarketDataRecord": MarketDataRecord,
            "Order": Order,
            "OrderFill": OrderFill,
            "PerformanceAuditLog": PerformanceAuditLog,
            "Position": Position,
            "RiskAuditLog": RiskAuditLog,
            "Signal": Signal,
            "Strategy": Strategy,
            "Trade": Trade,
            "User": User,
        }
    except (ImportError, Exception) as e:
        # Models not available or import error - this is acceptable for basic functionality
        # Log the error but don't fail the entire module
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"Database models import failed: {e}. Database functionality may be limited."
        )
        return {}


# Import models on module load
_models = _import_models()

# Make models available at package level
Base = _models.get("Base")
Bot = _models.get("Bot")
BotInstance = _models.get("BotInstance")
BotLog = _models.get("BotLog")
CapitalAuditLog = _models.get("CapitalAuditLog")
ExecutionAuditLog = _models.get("ExecutionAuditLog")
MarketDataRecord = _models.get("MarketDataRecord")
Order = _models.get("Order")
OrderFill = _models.get("OrderFill")
PerformanceAuditLog = _models.get("PerformanceAuditLog")
Position = _models.get("Position")
RiskAuditLog = _models.get("RiskAuditLog")
Signal = _models.get("Signal")
Strategy = _models.get("Strategy")
Trade = _models.get("Trade")
User = _models.get("User")

from .redis_client import RedisClient
from .service import DatabaseService

__all__ = [
    # Models and classes
    "Base",
    "Bot",
    "BotInstance",
    "BotLog",
    "CapitalAuditLog",
    "DatabaseConnectionError",
    "DatabaseConnectionManager",
    # Core exceptions (properly integrated)
    "DatabaseError",
    "DatabaseManager",
    "DatabaseQueryError",
    "DatabaseService",
    "ExecutionAuditLog",
    "InfluxDBClient",
    "MarketDataRecord",
    "Order",
    "OrderFill",
    "PerformanceAuditLog",
    "Position",
    "RedisClient",
    "RiskAuditLog",
    "Signal",
    "Strategy",
    "Trade",
    "User",
    "configure_database_dependencies",
    # Functions
    "get_async_session",
    "get_database_manager",
    "get_database_service",
    "get_sync_session",
    "get_uow_factory",
    # Dependency injection
    "register_database_services",
]
