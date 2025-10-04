"""
Mock Service Infrastructure for Integration Testing

Provides mock implementations of database services for environments
where Docker is not available (like WSL or CI environments).
"""

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import Column, String, create_engine, JSON, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)


class MockRedisClient:
    """Mock Redis client that simulates Redis behavior in memory."""

    def __init__(self):
        self.storage: Dict[str, Any] = {}
        self.ttl_storage: Dict[str, float] = {}
        self.pubsub_handlers: Dict[str, List] = defaultdict(list)
        self.connected = False
        self._lock = asyncio.Lock()

    async def connect(self):
        """Simulate connection to Redis."""
        async with self._lock:
            self.connected = True
            logger.debug("MockRedisClient connected")

    async def disconnect(self):
        """Simulate disconnection from Redis."""
        async with self._lock:
            self.connected = False
            self.storage.clear()
            self.ttl_storage.clear()
            logger.debug("MockRedisClient disconnected")

    async def ping(self) -> bool:
        """Check if Redis is connected."""
        return self.connected

    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set a key-value pair with optional expiration."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        # Convert value to string for Redis compatibility
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif not isinstance(value, str):
            value = str(value)

        self.storage[key] = value

        if ex:
            # Store expiration time
            self.ttl_storage[key] = asyncio.get_event_loop().time() + ex

        return True

    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        # Check if key has expired
        if key in self.ttl_storage:
            if asyncio.get_event_loop().time() > self.ttl_storage[key]:
                del self.storage[key]
                del self.ttl_storage[key]
                return None

        return self.storage.get(key)

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        deleted = 0
        for key in keys:
            if key in self.storage:
                del self.storage[key]
                deleted += 1
            if key in self.ttl_storage:
                del self.ttl_storage[key]

        return deleted

    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        return sum(1 for key in keys if key in self.storage)

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        if key in self.storage:
            self.ttl_storage[key] = asyncio.get_event_loop().time() + seconds
            return True
        return False

    async def ttl(self, key: str) -> int:
        """Get TTL for a key."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        if key not in self.ttl_storage:
            return -1 if key in self.storage else -2

        remaining = self.ttl_storage[key] - asyncio.get_event_loop().time()
        return max(0, int(remaining))

    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        """Set key with expiration."""
        return await self.set(key, value, ex=seconds)

    async def hset(self, name: str, key: Optional[str] = None, value: Optional[Any] = None, mapping: Optional[Dict] = None) -> int:
        """Set hash field."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        if name not in self.storage:
            self.storage[name] = {}

        if not isinstance(self.storage[name], dict):
            raise TypeError("Wrong type for hash operation")

        added = 0
        if mapping:
            for k, v in mapping.items():
                if k not in self.storage[name]:
                    added += 1
                self.storage[name][k] = str(v) if not isinstance(v, str) else v
        elif key is not None:
            if key not in self.storage[name]:
                added += 1
            self.storage[name][key] = str(value) if not isinstance(value, str) else value

        return added

    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        if name in self.storage and isinstance(self.storage[name], dict):
            return self.storage[name].get(key)
        return None

    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        if name in self.storage and isinstance(self.storage[name], dict):
            return self.storage[name].copy()
        return {}

    async def lpush(self, name: str, *values) -> int:
        """Push values to list."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        if name not in self.storage:
            self.storage[name] = []

        if not isinstance(self.storage[name], list):
            raise TypeError("Wrong type for list operation")

        for value in reversed(values):
            self.storage[name].insert(0, str(value) if not isinstance(value, str) else value)

        return len(self.storage[name])

    async def lrange(self, name: str, start: int, stop: int) -> List[str]:
        """Get range of list elements."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        if name not in self.storage:
            return []

        if not isinstance(self.storage[name], list):
            raise TypeError("Wrong type for list operation")

        lst = self.storage[name]
        if stop == -1:
            return lst[start:]
        return lst[start:stop+1]

    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        if isinstance(message, (dict, list)):
            message = json.dumps(message)
        elif not isinstance(message, str):
            message = str(message)

        handlers = self.pubsub_handlers.get(channel, [])
        for handler in handlers:
            await handler(channel, message)

        return len(handlers)

    async def subscribe(self, channel: str, handler) -> None:
        """Subscribe to channel."""
        if not self.connected:
            raise ConnectionError("Redis client not connected")

        self.pubsub_handlers[channel].append(handler)

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from channel."""
        if channel in self.pubsub_handlers:
            del self.pubsub_handlers[channel]

    async def flushdb(self) -> bool:
        """Clear all data."""
        self.storage.clear()
        self.ttl_storage.clear()
        return True


class MockInfluxDBClient:
    """Mock InfluxDB client that simulates time-series database behavior."""

    def __init__(self):
        self.data: List[Dict] = []
        self.connected = False
        self.bucket = "test-bucket"
        self.org = "test-org"

    async def initialize(self):
        """Initialize connection."""
        self.connected = True
        logger.debug("MockInfluxDBClient initialized")

    async def cleanup(self):
        """Cleanup connection."""
        self.connected = False
        self.data.clear()
        logger.debug("MockInfluxDBClient cleaned up")

    async def health_check(self) -> bool:
        """Check if InfluxDB is healthy."""
        return self.connected

    async def write_point(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """Write a data point."""
        if not self.connected:
            raise ConnectionError("InfluxDB client not connected")

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        point = {
            "measurement": measurement,
            "tags": tags.copy(),
            "fields": fields.copy(),
            "timestamp": timestamp
        }

        self.data.append(point)
        logger.debug(f"Written point to {measurement}: {fields}")

    async def query_range(
        self,
        measurement: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """Query data points in time range."""
        if not self.connected:
            raise ConnectionError("InfluxDB client not connected")

        if end_time is None:
            end_time = datetime.now(timezone.utc)

        results = []
        for point in self.data:
            if point["measurement"] != measurement:
                continue

            if point["timestamp"] < start_time or point["timestamp"] > end_time:
                continue

            if tags:
                match = all(
                    point["tags"].get(k) == v
                    for k, v in tags.items()
                )
                if not match:
                    continue

            results.append(point)

        return results

    async def delete_series(
        self,
        measurement: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Delete data points."""
        if not self.connected:
            raise ConnectionError("InfluxDB client not connected")

        original_count = len(self.data)

        self.data = [
            point for point in self.data
            if not (
                point["measurement"] == measurement
                and start_time <= point["timestamp"] <= end_time
                and (not tags or all(point["tags"].get(k) == v for k, v in tags.items()))
            )
        ]

        deleted = original_count - len(self.data)
        logger.debug(f"Deleted {deleted} points from {measurement}")
        return deleted > 0


class MockDatabaseService:
    """Mock database service using SQLite for testing."""

    def __init__(self, config=None):
        self.config = config
        self.engine = None
        self.session_maker = None
        self._started = False

    async def start(self):
        """Start the mock database service."""
        if self._started:
            return

        # Create SQLite in-memory database with UUID and JSONB support
        from sqlalchemy import event, String, JSON
        from sqlalchemy.dialects.postgresql import UUID, JSONB

        # Create engine with SQLite
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # Add UUID support for SQLite
        @event.listens_for(self.engine.sync_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            # Enable foreign keys
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        # Create session maker
        self.session_maker = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create tables with type adaptation for SQLite
        from src.database.models.base import Base
        from sqlalchemy import MetaData, Table, Column

        # Create a new metadata object with SQLite-compatible types
        metadata_sqlite = MetaData()

        # Copy tables with modified types for SQLite
        for table_name, table in Base.metadata.tables.items():
            # Create columns with adapted types
            columns = []
            for column in table.columns:
                # Determine the new type for SQLite
                new_type = column.type
                # Check for UUID types (both PostgreSQL UUID and generic UUID)
                if (isinstance(column.type, UUID) or
                    type(column.type).__name__ == "UUID" or
                    str(type(column.type)).find("UUID") != -1):
                    new_type = String(36)
                elif type(column.type).__name__ == "JSONB":
                    new_type = JSON()
                elif type(column.type).__name__ == "ARRAY":
                    new_type = JSON()

                # Handle server defaults and onupdate for SQLite compatibility
                new_default = None
                new_server_default = None
                new_onupdate = None

                # Convert PostgreSQL func.now() to SQLite compatible defaults
                if column.server_default is not None:
                    # Check if it's a func.now() call for timestamp columns
                    if hasattr(column.server_default, 'arg') and str(column.server_default.arg).strip() == 'now()':
                        # SQLite uses datetime('now') for timestamps
                        from sqlalchemy import text
                        new_server_default = text("(datetime('now'))")
                    else:
                        new_server_default = column.server_default

                if column.onupdate is not None:
                    # Check if it's a func.now() call
                    if hasattr(column.onupdate, 'arg') and str(column.onupdate.arg).strip() == 'now()':
                        from sqlalchemy import text
                        new_onupdate = text("(datetime('now'))")
                    else:
                        new_onupdate = column.onupdate

                # Use regular default if available
                if column.default is not None:
                    new_default = column.default

                # Create a new column with the adapted type and defaults
                new_column = Column(
                    column.name,
                    new_type,
                    primary_key=column.primary_key,
                    nullable=column.nullable,
                    default=new_default,
                    server_default=new_server_default,
                    onupdate=new_onupdate,
                    unique=column.unique,
                    index=column.index,
                    autoincrement=column.autoincrement if hasattr(column, 'autoincrement') else False
                )
                columns.append(new_column)

            # Create new table with modified columns
            new_table = Table(table_name, metadata_sqlite, *columns)

            # Copy constraints (except foreign keys which might reference UUID columns)
            for constraint in list(table.constraints):
                if constraint.__class__.__name__ not in ['ForeignKeyConstraint']:
                    try:
                        new_table.append_constraint(constraint.copy())
                    except:
                        pass  # Skip constraints that can't be copied

        async with self.engine.begin() as conn:
            await conn.run_sync(metadata_sqlite.create_all)

        self._started = True
        logger.info("Mock database service started with SQLite")

    async def stop(self):
        """Stop the mock database service."""
        if not self._started:
            return

        if self.engine:
            await self.engine.dispose()

        self._started = False
        logger.info("Mock database service stopped")

    def get_session(self) -> AsyncSession:
        """Get a database session."""
        if not self.session_maker:
            raise RuntimeError("Database service not started")
        return self.session_maker()

    async def get_health_status(self):
        """Get health status."""
        from enum import Enum

        class HealthStatus(Enum):
            healthy = "healthy"
            unhealthy = "unhealthy"

        return HealthStatus.healthy if self._started else HealthStatus.unhealthy


def create_mock_services(config=None):
    """Create all mock services for testing."""
    return {
        "database": MockDatabaseService(config),
        "redis": MockRedisClient(),
        "influxdb": MockInfluxDBClient(),
    }


async def setup_mock_infrastructure(config=None):
    """Setup complete mock infrastructure for integration testing."""
    services = create_mock_services(config)

    # Start all services
    await services["database"].start()
    await services["redis"].connect()
    await services["influxdb"].initialize()

    logger.info("Mock infrastructure setup complete")
    return services


async def teardown_mock_infrastructure(services):
    """Teardown mock infrastructure."""
    if "database" in services:
        await services["database"].stop()

    if "redis" in services:
        await services["redis"].disconnect()

    if "influxdb" in services:
        await services["influxdb"].cleanup()

    logger.info("Mock infrastructure teardown complete")