# Database Module

The database module provides database models, connection management, and utilities for PostgreSQL, Redis, and InfluxDB integration in the trading bot framework.

## Overview

This module serves as the data persistence layer providing:
- **Models**: SQLAlchemy models for all trading data, user information, and system metrics
- **Connection Management**: Async database connection management with health monitoring
- **Query Utilities**: Common database operations with type safety and bulk operations
- **Redis Client**: Async Redis client for real-time state management and caching
- **InfluxDB Client**: Time series data storage for market data and performance metrics
- **Migrations**: Database schema migrations and version control

## Files

### `models.py`
Database models for persistent storage using SQLAlchemy.

#### Models
- `User(Base)`: User model for authentication and account management
  - `id: str` (UUID): Primary key
  - `username: str`: Unique username (3+ chars)
  - `email: str`: Unique email address (5+ chars)
  - `password_hash: str`: Hashed password
  - `is_active: bool`: User active status
  - `is_verified: bool`: Email verification status
  - `created_at: datetime`: Account creation timestamp
  - `updated_at: datetime`: Last update timestamp
  - **Relationships**: bot_instances, balance_snapshots, alerts, audit_logs

- `BotInstance(Base)`: Bot instance model for managing individual trading bots
  - `id: str` (UUID): Primary key
  - `name: str`: Bot instance name
  - `user_id: str`: Foreign key to User
  - `strategy_type: str`: Strategy type (static, dynamic, etc.)
  - `exchange: str`: Exchange name
  - `status: str`: Bot status (stopped, running, etc.)
  - `config: Dict[str, Any]`: Bot configuration (JSONB)
  - `created_at: datetime`: Creation timestamp
  - `updated_at: datetime`: Last update timestamp
  - `last_active: Optional[datetime]`: Last activity timestamp
  - **Relationships**: user, trades, positions, performance_metrics

- `Trade(Base)`: Trade execution record
  - `id: str` (UUID): Primary key
  - `bot_id: str`: Foreign key to BotInstance
  - `symbol: str`: Trading symbol
  - `side: str`: Order side (buy/sell)
  - `order_type: str`: Order type
  - `quantity: Decimal`: Trade quantity
  - `price: Decimal`: Order price
  - `executed_price: Decimal`: Actual execution price
  - `fee: Decimal`: Trading fee
  - `pnl: Optional[Decimal]`: Profit/Loss
  - `status: str`: Trade status
  - `timestamp: datetime`: Trade timestamp
  - **Relationships**: bot_instance

- `Position(Base)`: Position tracking model
  - `id: str` (UUID): Primary key
  - `bot_id: str`: Foreign key to BotInstance
  - `symbol: str`: Trading symbol
  - `side: str`: Position side
  - `quantity: Decimal`: Position size
  - `entry_price: Decimal`: Entry price
  - `current_price: Decimal`: Current market price
  - `unrealized_pnl: Decimal`: Unrealized profit/loss
  - `realized_pnl: Decimal`: Realized profit/loss
  - `created_at: datetime`: Position open timestamp
  - `updated_at: datetime`: Last update timestamp
  - `closed_at: Optional[datetime]`: Position close timestamp
  - **Relationships**: bot_instance

- `BalanceSnapshot(Base)`: Balance tracking for accounts
  - `id: str` (UUID): Primary key
  - `user_id: str`: Foreign key to User
  - `exchange: str`: Exchange name
  - `currency: str`: Currency code
  - `total_balance: Decimal`: Total balance
  - `available_balance: Decimal`: Available balance
  - `locked_balance: Decimal`: Locked balance
  - `timestamp: datetime`: Snapshot timestamp
  - **Relationships**: user

- `StrategyConfig(Base)`: Strategy configuration storage
  - `id: str` (UUID): Primary key
  - `name: str`: Strategy name
  - `strategy_type: str`: Strategy type
  - `config_data: Dict[str, Any]`: Configuration data (JSONB)
  - `is_active: bool`: Active status
  - `created_at: datetime`: Creation timestamp
  - `updated_at: datetime`: Last update timestamp

- `MLModel(Base)`: Machine learning model metadata
  - `id: str` (UUID): Primary key
  - `name: str`: Model name
  - `model_type: str`: Model type
  - `version: str`: Model version
  - `file_path: str`: Model file path
  - `metadata: Dict[str, Any]`: Model metadata (JSONB)
  - `performance_metrics: Dict[str, Any]`: Performance metrics (JSONB)
  - `is_active: bool`: Active status
  - `created_at: datetime`: Creation timestamp
  - `updated_at: datetime`: Last update timestamp

- `PerformanceMetrics(Base)`: Bot performance metrics
  - `id: str` (UUID): Primary key
  - `bot_id: str`: Foreign key to BotInstance
  - `metric_date: datetime`: Metric date
  - `total_trades: int`: Total number of trades
  - `winning_trades: int`: Number of winning trades
  - `losing_trades: int`: Number of losing trades
  - `total_pnl: Decimal`: Total profit/loss
  - `win_rate: float`: Win rate percentage
  - `sharpe_ratio: Optional[float]`: Sharpe ratio
  - `max_drawdown: Optional[float]`: Maximum drawdown
  - `created_at: datetime`: Creation timestamp
  - **Relationships**: bot_instance

- `Alert(Base)`: System alerts and notifications
  - `id: str` (UUID): Primary key
  - `user_id: str`: Foreign key to User
  - `alert_type: str`: Alert type
  - `severity: str`: Alert severity (low, medium, high, critical)
  - `title: str`: Alert title
  - `message: str`: Alert message
  - `metadata: Dict[str, Any]`: Additional metadata (JSONB)
  - `is_read: bool`: Read status
  - `timestamp: datetime`: Alert timestamp
  - **Relationships**: user

- `AuditLog(Base)`: System audit logging
  - `id: str` (UUID): Primary key
  - `user_id: str`: Foreign key to User
  - `action: str`: Action performed
  - `resource: str`: Resource affected
  - `details: Dict[str, Any]`: Action details (JSONB)
  - `ip_address: Optional[str]`: Client IP address
  - `user_agent: Optional[str]`: Client user agent
  - `timestamp: datetime`: Action timestamp
  - **Relationships**: user

### `connection.py`
Database connection management with health monitoring.

#### Classes
- `DatabaseConnectionManager`: Manages database connections with health monitoring
  - `__init__(self, config: Config)`
  - `initialize(self) -> None`: Initialize all database connections
  - `get_async_session(self) -> AsyncSession`: Get async database session
  - `get_sync_session(self) -> Session`: Get sync database session
  - `get_redis_client(self) -> redis.Redis`: Get Redis client
  - `get_influxdb_client(self) -> InfluxDBClient`: Get InfluxDB client
  - `close(self) -> None`: Close all database connections
  - `is_healthy(self) -> bool`: Check if all database connections are healthy

#### Functions
- `initialize_database(config: Config) -> None`: Initialize global database connections
- `close_database() -> None`: Close global database connections
- `get_async_session() -> AsyncGenerator[AsyncSession, None]`: Async context manager for database sessions
- `get_sync_session() -> Session`: Get sync database session
- `get_redis_client() -> redis.Redis`: Get Redis client
- `get_influxdb_client() -> InfluxDBClient`: Get InfluxDB client
- `is_database_healthy() -> bool`: Check if database connections are healthy
- `execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> Any`: Execute database query with parameters
- `health_check() -> Dict[str, bool]`: Perform comprehensive health check on all databases
- `debug_connection_info() -> Dict[str, Any]`: Debug function to get connection information

### `queries.py`
Database query utilities with common CRUD operations.

#### Classes
- `DatabaseQueries`: Database query utilities with common CRUD operations
  - `__init__(self, session: AsyncSession)`

#### Generic CRUD Operations
  - `create(self, model_instance: T) -> T`: Create a new record
  - `get_by_id(self, model_class: type[T], record_id: str) -> Optional[T]`: Get record by ID
  - `get_all(self, model_class: type[T], limit: Optional[int] = None, offset: int = 0) -> List[T]`: Get all records with pagination
  - `update(self, model_instance: T) -> T`: Update existing record
  - `delete(self, model_instance: T) -> bool`: Delete record
  - `bulk_create(self, model_instances: List[T]) -> List[T]`: Create multiple records in bulk
  - `bulk_update(self, model_class: type[T], updates: List[Dict[str, Any]], id_field: str = "id") -> int`: Update multiple records in bulk

#### User-Specific Queries
  - `get_user_by_username(self, username: str) -> Optional[User]`: Get user by username
  - `get_user_by_email(self, email: str) -> Optional[User]`: Get user by email
  - `get_active_users(self) -> List[User]`: Get all active users

#### Bot Instance Queries
  - `get_bot_instances_by_user(self, user_id: str) -> List[BotInstance]`: Get all bot instances for user
  - `get_bot_instance_by_name(self, user_id: str, name: str) -> Optional[BotInstance]`: Get bot instance by name
  - `get_running_bots(self) -> List[BotInstance]`: Get all running bot instances

#### Trade Queries
  - `get_trades_by_bot(self, bot_id: str, limit: Optional[int] = None, offset: int = 0) -> List[Trade]`: Get trades for specific bot
  - `get_trades_by_symbol(self, symbol: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Trade]`: Get trades for symbol within time range
  - `get_trades_by_date_range(self, start_time: datetime, end_time: datetime) -> List[Trade]`: Get all trades within date range

#### Position Queries
  - `get_positions_by_bot(self, bot_id: str) -> List[Position]`: Get all positions for specific bot
  - `get_open_positions(self) -> List[Position]`: Get all open positions

#### Balance Queries
  - `get_latest_balance_snapshot(self, user_id: str, exchange: str, currency: str) -> Optional[BalanceSnapshot]`: Get latest balance snapshot

#### Performance Metrics Queries
  - `get_performance_metrics_by_bot(self, bot_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[PerformanceMetrics]`: Get performance metrics for bot

#### Alert Queries
  - `get_unread_alerts_by_user(self, user_id: str) -> List[Alert]`: Get unread alerts for user
  - `get_alerts_by_severity(self, severity: str, limit: Optional[int] = None) -> List[Alert]`: Get alerts by severity level

#### Audit Log Queries
  - `get_audit_logs_by_user(self, user_id: str, limit: Optional[int] = None) -> List[AuditLog]`: Get audit logs for user

#### Aggregation Queries
  - `get_total_pnl_by_bot(self, bot_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Decimal`: Get total P&L for bot
  - `get_trade_count_by_bot(self, bot_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> int`: Get trade count for bot
  - `get_win_rate_by_bot(self, bot_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> float`: Get win rate for bot

#### Data Export Utilities
  - `export_trades_to_csv_data(self, bot_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]`: Export trades to CSV format data

#### Health Check
  - `health_check(self) -> bool`: Check database health with simple query

### `redis_client.py`
Redis client for real-time state management and caching.

#### Classes
- `RedisClient`: Async Redis client with utilities for trading bot data
  - `__init__(self, redis_url: str)`
  - `connect(self) -> None`: Connect to Redis with proper configuration
  - `disconnect(self) -> None`: Disconnect from Redis
  - `set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "trading_bot") -> bool`: Set key-value pair with optional TTL
  - `get(self, key: str, namespace: str = "trading_bot") -> Optional[Any]`: Get value by key
  - `delete(self, key: str, namespace: str = "trading_bot") -> bool`: Delete key
  - `exists(self, key: str, namespace: str = "trading_bot") -> bool`: Check if key exists
  - `set_multiple(self, data: Dict[str, Any], ttl: Optional[int] = None, namespace: str = "trading_bot") -> bool`: Set multiple key-value pairs
  - `get_multiple(self, keys: List[str], namespace: str = "trading_bot") -> Dict[str, Any]`: Get multiple values by keys
  - `delete_multiple(self, keys: List[str], namespace: str = "trading_bot") -> int`: Delete multiple keys
  - `get_keys_by_pattern(self, pattern: str, namespace: str = "trading_bot") -> List[str]`: Get keys matching pattern
  - `clear_namespace(self, namespace: str = "trading_bot") -> int`: Clear all keys in namespace
  - `set_hash(self, name: str, mapping: Dict[str, Any], namespace: str = "trading_bot") -> bool`: Set hash fields
  - `get_hash(self, name: str, namespace: str = "trading_bot") -> Dict[str, Any]`: Get hash fields
  - `get_hash_field(self, name: str, field: str, namespace: str = "trading_bot") -> Optional[Any]`: Get specific hash field
  - `set_list(self, name: str, values: List[Any], namespace: str = "trading_bot") -> bool`: Set list values
  - `get_list(self, name: str, start: int = 0, end: int = -1, namespace: str = "trading_bot") -> List[Any]`: Get list values
  - `append_to_list(self, name: str, value: Any, namespace: str = "trading_bot") -> int`: Append value to list
  - `set_expiry(self, key: str, ttl: int, namespace: str = "trading_bot") -> bool`: Set expiry for existing key
  - `get_ttl(self, key: str, namespace: str = "trading_bot") -> int`: Get time to live for key
  - `health_check(self) -> bool`: Check Redis connection health

### `influxdb_client.py`
InfluxDB client for time series data storage.

#### Classes
- `InfluxDBClientWrapper`: InfluxDB client wrapper with trading-specific utilities
  - `__init__(self, url: str, token: str, org: str, bucket: str)`
  - `connect(self) -> None`: Connect to InfluxDB
  - `disconnect(self) -> None`: Disconnect from InfluxDB
  - `write_point(self, point: Point) -> None`: Write single point to InfluxDB
  - `write_points(self, points: List[Point]) -> None`: Write multiple points to InfluxDB
  - `query_data(self, query: str) -> List[Dict[str, Any]]`: Execute query and return results
  - `write_market_data(self, symbol: str, price: float, volume: float, timestamp: Optional[datetime] = None) -> None`: Write market data point
  - `write_trade_data(self, trade_data: Dict[str, Any]) -> None`: Write trade execution data
  - `write_performance_metrics(self, bot_id: str, metrics: Dict[str, Any], timestamp: Optional[datetime] = None) -> None`: Write performance metrics
  - `write_system_metrics(self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None) -> None`: Write system monitoring metrics
  - `get_market_data(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]`: Get market data for symbol
  - `get_trade_history(self, bot_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]`: Get trade history for bot
  - `get_performance_metrics(self, bot_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]`: Get performance metrics for bot
  - `get_system_metrics(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]`: Get system metrics
  - `delete_data(self, start_time: datetime, end_time: datetime, measurement: Optional[str] = None) -> None`: Delete data within time range
  - `health_check(self) -> bool`: Check InfluxDB connection health

### `migrations/`
Database schema migrations and version control.

#### Files
- `env.py`: Alembic environment configuration
  - `get_url() -> str`: Get database URL for migrations
  - `run_migrations_offline() -> None`: Run migrations in offline mode
  - `run_migrations_online() -> None`: Run migrations in online mode

- `script.py.mako`: Template for migration scripts
  - `upgrade() -> None`: Apply migration changes
  - `downgrade() -> None`: Revert migration changes

- `versions/001_initial_schema.py`: Initial database schema
  - `upgrade() -> None`: Create initial tables and indexes
  - `downgrade() -> None`: Drop all tables

## Dependencies

- SQLAlchemy for ORM and database operations
- Alembic for database migrations
- asyncpg for async PostgreSQL support
- redis for Redis operations
- influxdb-client for InfluxDB operations
- pydantic for data validation
- src.core for core types and configuration

## Usage Examples

```python
# Database Connection
from src.database.connection import initialize_database, get_async_session
from src.core.config import Config

config = Config()
await initialize_database(config)

# Using Database Queries
from src.database.queries import DatabaseQueries
from src.database.models import User, Trade

async with get_async_session() as session:
    db = DatabaseQueries(session)
    
    # Create user
    user = User(username="trader1", email="trader@example.com")
    created_user = await db.create(user)
    
    # Get trades
    trades = await db.get_trades_by_bot("bot_id", limit=10)
    
    # Get performance metrics
    total_pnl = await db.get_total_pnl_by_bot("bot_id")

# Using Redis Client
from src.database.redis_client import RedisClient

redis_client = RedisClient("redis://localhost:6379")
await redis_client.connect()

# Store data
await redis_client.set("price:BTCUSDT", 50000, ttl=60)
price = await redis_client.get("price:BTCUSDT")

# Store complex data
await redis_client.set_hash("portfolio:user1", {
    "total_value": 10000,
    "positions": 5,
    "cash": 2000
})

# Using InfluxDB Client
from src.database.influxdb_client import InfluxDBClientWrapper

influx_client = InfluxDBClientWrapper(
    "http://localhost:8086",
    "token",
    "org",
    "bucket"
)
influx_client.connect()

# Write market data
influx_client.write_market_data("BTCUSDT", 50000.0, 1.5)

# Query data
market_data = influx_client.get_market_data(
    "BTCUSDT",
    start_time,
    end_time
)
```

## Integration with Core Framework

This module integrates tightly with:
- **P-001 Core Framework**: Uses core types, exceptions, config, and logging
- **P-007A Utils Framework**: Uses decorators for performance monitoring
- **P-002A Error Handling**: Uses error handling for connection management

All database operations follow the common patterns defined in the specifications and use the standardized exception hierarchy from the core module.
