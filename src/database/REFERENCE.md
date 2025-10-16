# DATABASE Module Reference

## INTEGRATION
**Dependencies**: core, error_handling, risk_management, utils, web_interface
**Used By**: None
**Provides**: BotService, DatabaseConnectionManager, DatabaseManager, DatabaseService, MLService, MarketDataService, TradingService
**Patterns**: Async Operations, Circuit Breaker, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Security**:
- Authentication
- Authentication
**Performance**:
- Parallel execution
- Retry mechanisms
**Architecture**:
- InfluxDBClientWrapper inherits from base architecture
- DatabaseManager inherits from base architecture
- DatabaseQueries inherits from base architecture

## MODULE OVERVIEW
**Files**: 59 Python files
**Classes**: 138
**Functions**: 55

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `DatabaseConnectionManager` ✅

**Purpose**: Manages database connections with health monitoring and reconnection
**Status**: Complete

**Implemented Methods:**
- `set_test_schema(self, schema: str) -> None` - Line 63
- `async initialize(self) -> None` - Line 131
- `async get_async_session(self) -> AsyncGenerator[AsyncSession, None]` - Line 326
- `get_sync_session(self) -> Session` - Line 365
- `async get_redis_client(self) -> redis.Redis` - Line 385
- `get_influxdb_client(self) -> InfluxDBClient` - Line 391
- `async close(self) -> None` - Line 397
- `is_healthy(self) -> bool` - Line 491
- `async_session_maker(self)` - Line 496
- `sync_session_maker(self)` - Line 501
- `async get_connection(self)` - Line 506
- `async get_pool_status(self) -> dict[str, int]` - Line 514

### Implementation: `InfluxDBClientWrapper` ✅

**Inherits**: BaseComponent
**Purpose**: InfluxDB client wrapper with trading-specific utilities
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 49
- `async disconnect(self) -> None` - Line 63
- `async write_point(self, point: Point) -> None` - Line 129
- `async write_points(self, points: list[Point]) -> None` - Line 157
- `async write_market_data(self, symbol: str, data: dict[str, Any], timestamp: datetime | None = None) -> None` - Line 186
- `async write_market_data_batch(self, data_list: list[dict[str, Any]], timestamp: datetime | None = None) -> None` - Line 204
- `async write_trade(self, trade_data: dict[str, Any], timestamp: datetime | None = None) -> None` - Line 243
- `async write_performance_metrics(self, bot_id: str, metrics: dict[str, Any], timestamp: datetime | None = None) -> None` - Line 266
- `async write_system_metrics(self, metrics: dict[str, Any], timestamp: datetime | None = None) -> None` - Line 288
- `async write_risk_metrics(self, bot_id: str, risk_data: dict[str, Any], timestamp: datetime | None = None) -> None` - Line 306
- `async query_market_data(self, symbol: str, start_time: datetime, end_time: datetime, limit: int = 1000) -> list[dict[str, Any]]` - Line 326
- `async query_trades(self, bot_id: str, start_time: datetime, end_time: datetime, limit: int = 1000) -> list[dict[str, Any]]` - Line 367
- `async query_performance_metrics(self, bot_id: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]` - Line 408
- `async get_daily_pnl(self, bot_id: str, date: datetime) -> dict[str, Decimal]` - Line 472
- `async get_win_rate(self, bot_id: str, start_time: datetime, end_time: datetime) -> Decimal` - Line 522
- `async health_check(self) -> bool` - Line 597

### Implementation: `DatabaseServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for database service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async start(self) -> None` - Line 21
- `async stop(self) -> None` - Line 26
- `async create_entity(self, entity: T) -> T` - Line 31
- `async get_entity_by_id(self, model_class: type[T], entity_id: K) -> T | None` - Line 36
- `async update_entity(self, entity: T) -> T` - Line 41
- `async delete_entity(self, model_class: type[T], entity_id: K) -> bool` - Line 46
- `async list_entities(self, ...) -> list[T]` - Line 51
- `async count_entities(self, model_class: type[T] | None = None, filters: dict[str, Any] | None = None) -> int` - Line 65
- `async bulk_create(self, entities: list[T]) -> list[T]` - Line 72
- `async get_health_status(self) -> HealthStatus` - Line 77
- `get_performance_metrics(self) -> dict[str, Any]` - Line 82

### Implementation: `TradingDataServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for trading-specific data operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_trades_by_bot(self, ...) -> list[Any]` - Line 91
- `async get_positions_by_bot(self, bot_id: str) -> list[Any]` - Line 103
- `async calculate_total_pnl(self, ...) -> Decimal` - Line 108

### Implementation: `BotMetricsServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot metrics operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]` - Line 122
- `async store_bot_metrics(self, metrics_record: dict[str, Any]) -> bool` - Line 127
- `async get_active_bots(self) -> list[dict[str, Any]]` - Line 132
- `async archive_bot_record(self, bot_id: str) -> bool` - Line 137

### Implementation: `HealthAnalyticsServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for health analytics operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async store_bot_health_analysis(self, health_analysis: dict[str, Any]) -> bool` - Line 146
- `async get_bot_health_analyses(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]` - Line 151
- `async get_recent_health_analyses(self, hours: int = 1) -> list[dict[str, Any]]` - Line 156

### Implementation: `ResourceManagementServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for resource management operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async store_resource_allocation(self, allocation_record: dict[str, Any]) -> bool` - Line 165
- `async store_resource_usage(self, usage_record: dict[str, Any]) -> bool` - Line 170
- `async store_resource_reservation(self, reservation: dict[str, Any]) -> bool` - Line 175
- `async update_resource_allocation_status(self, bot_id: str, status: str) -> bool` - Line 180

### Implementation: `RepositoryFactoryInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for repository factory operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `create_repository(self, repository_class: type[T], session: Any) -> T` - Line 189
- `register_repository(self, name: str, repository_class: type[T]) -> None` - Line 194
- `is_repository_registered(self, name: str) -> bool` - Line 199

### Implementation: `MLServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for ML service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_model_performance_summary(self, model_name: str, days: int = 30) -> dict[str, Any]` - Line 208
- `async validate_model_deployment(self, model_name: str, version: int) -> bool` - Line 215
- `async get_model_recommendations(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]` - Line 220

### Implementation: `ConnectionManagerInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for database connection management operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 229
- `async close(self) -> None` - Line 234
- `async get_async_session(self)` - Line 239
- `get_sync_session(self)` - Line 244
- `async get_redis_client(self)` - Line 249
- `get_influxdb_client(self)` - Line 254
- `is_healthy(self) -> bool` - Line 259

### Implementation: `UnitOfWorkFactoryInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for Unit of Work factory operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `create(self) -> Any` - Line 268
- `create_async(self) -> Any` - Line 273
- `configure_dependencies(self, dependency_injector: Any) -> None` - Line 278

### Implementation: `CapitalServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for capital management operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_available_capital(self, account_id: str) -> Decimal` - Line 287
- `async allocate_capital(self, account_id: str, amount: Decimal, purpose: str) -> bool` - Line 292
- `async release_capital(self, account_id: str, amount: Decimal, purpose: str) -> bool` - Line 297

### Implementation: `UserServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for user management operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async authenticate_user(self, email: str, password: str) -> dict[str, Any] | None` - Line 306
- `async get_user_permissions(self, user_id: str) -> list[str]` - Line 311

### Implementation: `MarketDataServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for market data operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_latest_price(self, symbol: str) -> Decimal | None` - Line 320
- `async get_historical_data(self, ...) -> list[dict[str, Any]]` - Line 325

### Implementation: `AuditServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for audit trail operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async log_action(self, action_data: dict[str, Any]) -> bool` - Line 336
- `async get_audit_trail(self, entity_type: str, entity_id: str, limit: int = 100) -> list[dict[str, Any]]` - Line 341

### Implementation: `DatabaseManager` ✅

**Inherits**: BaseComponent
**Purpose**: Database operations coordinator that enforces service layer pattern
**Status**: Complete

**Implemented Methods:**
- `async get_historical_data(self, ...) -> list[dict[str, Any]]` - Line 42
- `async save_trade(self, trade_data: dict[str, Any]) -> dict[str, Any]` - Line 70
- `async get_positions(self, strategy_id: str | None = None, symbol: str | None = None) -> list[dict[str, Any]]` - Line 91
- `async close(self)` - Line 115

### Implementation: `AnalyticsPortfolioMetrics` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Portfolio metrics storage
**Status**: Complete

### Implementation: `AnalyticsPositionMetrics` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Position metrics storage
**Status**: Complete

### Implementation: `AnalyticsRiskMetrics` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Risk metrics storage
**Status**: Complete

### Implementation: `AnalyticsStrategyMetrics` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Strategy performance metrics storage
**Status**: Complete

### Implementation: `AnalyticsOperationalMetrics` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Operational metrics storage
**Status**: Complete

### Implementation: `CapitalAuditLog` ✅

**Inherits**: Base
**Purpose**: Audit log model for capital management operations
**Status**: Complete

### Implementation: `ExecutionAuditLog` ✅

**Inherits**: Base
**Purpose**: Audit log model for execution operations
**Status**: Complete

### Implementation: `RiskAuditLog` ✅

**Inherits**: Base
**Purpose**: Audit log model for risk management decisions and violations
**Status**: Complete

### Implementation: `PerformanceAuditLog` ✅

**Inherits**: Base
**Purpose**: Audit log model for performance tracking and analysis
**Status**: Complete

### Implementation: `BacktestRun` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Backtest run configuration and metadata storage
**Status**: Complete

### Implementation: `BacktestResult` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Comprehensive backtest results and performance metrics
**Status**: Complete

### Implementation: `BacktestTrade` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Individual trade records from backtest simulations
**Status**: Complete

### Implementation: `TimestampMixin` ✅

**Purpose**: Mixin for automatic timestamp management
**Status**: Complete

**Implemented Methods:**
- `created_at(self)` - Line 22
- `updated_at(self)` - Line 26

### Implementation: `AuditMixin` ✅

**Inherits**: TimestampMixin
**Purpose**: Mixin for audit fields
**Status**: Complete

**Implemented Methods:**
- `created_by(self)` - Line 36
- `updated_by(self)` - Line 40
- `version(self)` - Line 44

### Implementation: `MetadataMixin` ✅

**Purpose**: Mixin for metadata storage
**Status**: Complete

**Implemented Methods:**
- `metadata_json(self)` - Line 52
- `get_metadata(self, key: str, default: Any = None) -> Any` - Line 55
- `set_metadata(self, key: str, value: Any) -> None` - Line 61
- `update_metadata(self, data: dict[str, Any]) -> None` - Line 67

### Implementation: `SoftDeleteMixin` ✅

**Purpose**: Mixin for soft delete functionality
**Status**: Complete

**Implemented Methods:**
- `deleted_at(self)` - Line 78
- `deleted_by(self)` - Line 82
- `is_deleted(self) -> bool` - Line 86
- `soft_delete(self, deleted_by: str | None = None) -> None` - Line 90
- `restore(self) -> None` - Line 96

### Implementation: `Bot` ✅

**Inherits**: Base, AuditMixin, MetadataMixin, SoftDeleteMixin
**Purpose**: Bot model
**Status**: Complete

**Implemented Methods:**
- `is_running(self) -> bool` - Line 124
- `win_rate(self) -> Decimal` - Line 128
- `average_pnl(self) -> Decimal` - Line 136

### Implementation: `Strategy` ✅

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: Strategy model
**Status**: Complete

**Implemented Methods:**
- `is_active(self) -> bool` - Line 239
- `signal_success_rate(self) -> Decimal` - Line 243

### Implementation: `Signal` ✅

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: Trading signal model
**Status**: Complete

**Implemented Methods:**
- `is_executed(self) -> bool` - Line 320
- `is_successful(self) -> bool` - Line 324

### Implementation: `BotLog` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Bot activity log model for error handling and audit trail
**Status**: Complete

**Implemented Methods:**

### Implementation: `BotInstance` ✅

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: Bot instance model for managing individual trading bots
**Status**: Complete

**Implemented Methods:**
- `is_running(self) -> bool` - Line 129
- `is_stopped(self) -> bool` - Line 133

### Implementation: `CapitalAllocationDB` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Capital allocation tracking model
**Status**: Complete

**Implemented Methods:**

### Implementation: `FundFlowDB` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Fund flow tracking model
**Status**: Complete

**Implemented Methods:**

### Implementation: `CurrencyExposureDB` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Currency exposure tracking model
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeAllocationDB` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange allocation tracking model
**Status**: Complete

**Implemented Methods:**

### Implementation: `FeatureRecord` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Feature record model for ML feature storage
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataQualityRecord` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Data quality tracking model
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataPipelineRecord` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Data pipeline execution tracking model
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeConfiguration` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange configuration model
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeTradingPair` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange-specific trading pair information
**Status**: Complete

**Implemented Methods:**
- `round_price(self, price: Decimal) -> Decimal` - Line 193
- `round_quantity(self, quantity: Decimal) -> Decimal` - Line 197
- `validate_order(self, price: Decimal, quantity: Decimal) -> bool` - Line 201

### Implementation: `ExchangeConnectionStatus` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange connection status tracking
**Status**: Complete

**Implemented Methods:**
- `calculate_success_rate(self) -> Decimal` - Line 285

### Implementation: `ExchangeRateLimit` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange rate limit tracking and enforcement
**Status**: Complete

**Implemented Methods:**
- `is_exceeded(self) -> bool` - Line 357
- `remaining_requests(self) -> int` - Line 362
- `usage_percentage(self) -> Decimal` - Line 367

### Implementation: `MarketDataRecord` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Market data record model
**Status**: Complete

**Implemented Methods:**
- `price_change(self) -> Decimal` - Line 108
- `price_change_percent(self) -> Decimal` - Line 115

### Implementation: `MLPrediction` ✅

**Inherits**: Base
**Purpose**: Model for storing ML prediction results
**Status**: Complete

**Implemented Methods:**

### Implementation: `MLModelMetadata` ✅

**Inherits**: Base
**Purpose**: Model for storing ML model metadata and versioning information
**Status**: Complete

**Implemented Methods:**

### Implementation: `MLTrainingJob` ✅

**Inherits**: Base
**Purpose**: Model for tracking ML model training jobs
**Status**: Complete

**Implemented Methods:**

### Implementation: `OptimizationRun` ✅

**Inherits**: Base
**Purpose**: Model for optimization run metadata
**Status**: Complete

### Implementation: `OptimizationResult` ✅

**Inherits**: Base
**Purpose**: Model for storing final optimization results
**Status**: Complete

### Implementation: `ParameterSet` ✅

**Inherits**: Base
**Purpose**: Model for storing individual parameter sets evaluated during optimization
**Status**: Complete

### Implementation: `OptimizationObjectiveDB` ✅

**Inherits**: Base
**Purpose**: Model for storing individual optimization objectives
**Status**: Complete

### Implementation: `RiskConfiguration` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Risk configuration storage for strategies and bots
**Status**: Complete

### Implementation: `CircuitBreakerConfig` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Circuit breaker configuration storage
**Status**: Complete

### Implementation: `CircuitBreakerEvent` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Circuit breaker trigger event storage
**Status**: Complete

### Implementation: `RiskViolation` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Risk violation event storage
**Status**: Complete

### Implementation: `StateSnapshot` ✅

**Inherits**: Base, AuditMixin, MetadataMixin, SoftDeleteMixin
**Purpose**: State snapshot table for point-in-time state captures
**Status**: Complete

**Implemented Methods:**
- `compression_ratio(self) -> Decimal` - Line 136
- `get_state_by_type(self, state_type: str) -> dict[str, Any] | None` - Line 142
- `get_state_count(self) -> int` - Line 148

### Implementation: `StateCheckpoint` ✅

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: State checkpoint table for incremental state saves
**Status**: Complete

**Implemented Methods:**
- `get_changed_state_ids(self) -> set[str]` - Line 271
- `has_state_type(self, state_type: str) -> bool` - Line 277

### Implementation: `StateHistory` ✅

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: State history table for detailed audit trail
**Status**: Complete

**Implemented Methods:**
- `size_change_bytes(self) -> int` - Line 407
- `get_changed_field_names(self) -> set[str]` - Line 413

### Implementation: `StateMetadata` ✅

**Inherits**: Base, AuditMixin
**Purpose**: State metadata table for state information and indexing
**Status**: Complete

**Implemented Methods:**
- `get_tag(self, key: str, default: Any = None) -> Any` - Line 519
- `set_tag(self, key: str, value: Any) -> None` - Line 525
- `increment_access_count(self) -> None` - Line 533
- `is_in_storage_layer(self, layer: str) -> bool` - Line 538

### Implementation: `StateBackup` ✅

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: State backup table for backup operations tracking
**Status**: Complete

**Implemented Methods:**
- `is_expired(self) -> bool` - Line 676
- `backup_success_rate(self) -> Decimal` - Line 682
- `get_included_state_types(self) -> set[str]` - Line 691
- `mark_verified(self, checksum: str) -> bool` - Line 701

### Implementation: `Alert` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Alert model for system notifications
**Status**: Complete

**Implemented Methods:**

### Implementation: `AlertRule` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Alert rule configuration model
**Status**: Complete

**Implemented Methods:**

### Implementation: `EscalationPolicy` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Escalation policy for alert management
**Status**: Complete

**Implemented Methods:**

### Implementation: `AuditLog` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: General audit log model
**Status**: Complete

**Implemented Methods:**

### Implementation: `PerformanceMetrics` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Performance metrics model
**Status**: Complete

**Implemented Methods:**

### Implementation: `BalanceSnapshot` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Balance snapshot model for tracking account balances over time
**Status**: Complete

**Implemented Methods:**

### Implementation: `ResourceAllocation` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Resource allocation model for tracking bot resource allocations
**Status**: Complete

**Implemented Methods:**

### Implementation: `ResourceUsage` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Resource usage model for tracking bot resource usage
**Status**: Complete

**Implemented Methods:**

### Implementation: `Order` ✅

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: Order model
**Status**: Complete

**Implemented Methods:**
- `is_filled(self) -> bool` - Line 108
- `is_active(self) -> bool` - Line 115
- `remaining_quantity(self) -> Decimal` - Line 120

### Implementation: `Position` ✅

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: Position model
**Status**: Complete

**Implemented Methods:**
- `is_open(self) -> bool` - Line 213
- `value(self) -> Decimal` - Line 218
- `calculate_pnl(self, current_price: Decimal | None = None) -> Decimal` - Line 224

### Implementation: `OrderFill` ✅

**Inherits**: Base, TimestampMixin
**Purpose**: Order fill/execution model
**Status**: Complete

**Implemented Methods:**
- `value(self) -> Decimal` - Line 286
- `net_value(self) -> Decimal` - Line 293

### Implementation: `Trade` ✅

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: Completed trade model
**Status**: Complete

**Implemented Methods:**
- `is_profitable(self) -> bool` - Line 374
- `return_percentage(self) -> Decimal` - Line 381

### Implementation: `User` ✅

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: User model for authentication and account management
**Status**: Complete

**Implemented Methods:**
- `full_name(self) -> str` - Line 95
- `is_authenticated(self) -> bool` - Line 102

### Implementation: `DatabaseQueries` ✅

**Inherits**: BaseComponent
**Purpose**: Database query utilities with common CRUD operations
**Status**: Complete

**Implemented Methods:**
- `async create(self, model_instance: T) -> T` - Line 138
- `async get_by_id(self, model_class: type[T], record_id: str) -> T | None` - Line 230
- `async get_all(self, model_class: type[T], limit: int | None = None, offset: int = 0) -> list[T]` - Line 283
- `async update(self, model_instance: T) -> T` - Line 298
- `async delete(self, model_instance: T) -> bool` - Line 336
- `async bulk_create(self, model_instances: list[T]) -> list[T]` - Line 373
- `async bulk_update(self, model_class: type[T], updates: list[dict[str, Any]], id_field: str = 'id') -> int` - Line 412
- `async get_user_by_username(self, username: str) -> User | None` - Line 461
- `async get_user_by_email(self, email: str) -> User | None` - Line 471
- `async get_active_users(self) -> list[User]` - Line 481
- `async get_bot_instances_by_user(self, user_id: str) -> list[BotInstance]` - Line 491
- `async get_bot_instance_by_name(self, user_id: str, name: str) -> BotInstance | None` - Line 505
- `async get_running_bots(self) -> list[BotInstance]` - Line 519
- `async get_trades_by_bot(self, bot_id: str, limit: int | None = None, offset: int = 0) -> list[Trade]` - Line 535
- `async get_trades_by_symbol(self, ...) -> list[Trade]` - Line 550
- `async get_trades_by_date_range(self, start_time: datetime, end_time: datetime) -> list[Trade]` - Line 570
- `async get_positions_by_bot(self, bot_id: str) -> list[Position]` - Line 586
- `async get_open_positions(self) -> list[Position]` - Line 600
- `async get_latest_balance_snapshot(self, user_id: str, exchange: str, currency: str) -> BalanceSnapshot | None` - Line 616
- `async get_performance_metrics_by_bot(self, ...) -> list[PerformanceMetrics]` - Line 639
- `async get_unread_alerts_by_user(self, user_id: str) -> list[Alert]` - Line 659
- `async get_alerts_by_severity(self, severity: str, limit: int | None = None) -> list[Alert]` - Line 673
- `async get_audit_logs_by_user(self, user_id: str, limit: int | None = None) -> list[AuditLog]` - Line 688
- `async get_total_pnl_by_bot(self, ...) -> Decimal` - Line 709
- `async get_trade_count_by_bot(self, ...) -> int` - Line 730
- `async get_win_rate_by_bot(self, ...) -> tuple[int, int]` - Line 749
- `async export_trades_to_csv_data(self, ...) -> list[dict[str, Any]]` - Line 786
- `async health_check(self) -> bool` - Line 829
- `async get_capital_allocations_by_strategy(self, strategy_id: str, limit: int | None = None, offset: int = 0) -> list[CapitalAllocationDB]` - Line 841
- `async get_capital_allocations_by_exchange(self, exchange: str, limit: int | None = None, offset: int = 0) -> list[CapitalAllocationDB]` - Line 862
- `async get_fund_flows_by_reason(self, ...) -> list[FundFlowDB]` - Line 883
- `async get_fund_flows_by_currency(self, ...) -> list[FundFlowDB]` - Line 911
- `async get_currency_exposure_by_currency(self, currency: str) -> CurrencyExposureDB | None` - Line 939
- `async get_exchange_allocation_by_exchange(self, exchange: str) -> ExchangeAllocationDB | None` - Line 950
- `async get_total_capital_allocated(self, start_time: datetime | None = None, end_time: datetime | None = None) -> Decimal` - Line 965
- `async get_total_fund_flows(self, start_time: datetime | None = None, end_time: datetime | None = None) -> Decimal` - Line 985
- `async bulk_create_capital_allocations(self, allocations: list[CapitalAllocationDB]) -> list[CapitalAllocationDB]` - Line 1005
- `async bulk_update_capital_allocations(self, updates: list[dict[str, Any]]) -> int` - Line 1011
- `async bulk_create_fund_flows(self, flows: list[FundFlowDB]) -> list[FundFlowDB]` - Line 1015
- `async create_market_data_record(self, market_data: MarketDataRecord) -> MarketDataRecord` - Line 1019
- `async bulk_create_market_data_records(self, records: list[MarketDataRecord]) -> list[MarketDataRecord]` - Line 1023
- `async get_market_data_records(self, ...) -> list[MarketDataRecord]` - Line 1029
- `async get_market_data_by_quality(self, ...) -> list[MarketDataRecord]` - Line 1060
- `async delete_old_market_data(self, cutoff_date: datetime) -> int` - Line 1086
- `async create_feature_record(self, feature: FeatureRecord) -> FeatureRecord` - Line 1108
- `async bulk_create_feature_records(self, features: list[FeatureRecord]) -> list[FeatureRecord]` - Line 1112
- `async get_feature_records(self, ...) -> list[FeatureRecord]` - Line 1118
- `async create_data_quality_record(self, quality_record: DataQualityRecord) -> DataQualityRecord` - Line 1148
- `async get_data_quality_records(self, ...) -> list[DataQualityRecord]` - Line 1154
- `async create_data_pipeline_record(self, pipeline_record: DataPipelineRecord) -> DataPipelineRecord` - Line 1183
- `async update_data_pipeline_status(self, ...) -> bool` - Line 1189
- `async get_data_pipeline_records(self, ...) -> list[DataPipelineRecord]` - Line 1220

### Implementation: `RedisClient` ✅

**Inherits**: BaseComponent, CacheClientInterface
**Purpose**: Async Redis client with utilities for trading bot data
**Status**: Complete

**Implemented Methods:**
- `client(self) -> redis.Redis | None` - Line 105
- `async connect(self) -> None` - Line 112
- `async disconnect(self) -> None` - Line 250
- `async set(self, ...) -> bool` - Line 285
- `async get(self, key: str, namespace: str = 'trading_bot') -> Any | None` - Line 319
- `async delete(self, key: str, namespace: str = 'trading_bot') -> bool` - Line 345
- `async exists(self, key: str, namespace: str = 'trading_bot') -> bool` - Line 359
- `async expire(self, key: str, ttl: int, namespace: str = 'trading_bot') -> bool` - Line 373
- `async ttl(self, key: str, namespace: str = 'trading_bot') -> int` - Line 387
- `async hset(self, key: str, field: str, value: Any, namespace: str = 'trading_bot') -> bool` - Line 402
- `async hget(self, key: str, field: str, namespace: str = 'trading_bot') -> Any | None` - Line 423
- `async hgetall(self, key: str, namespace: str = 'trading_bot') -> dict[str, Any]` - Line 445
- `async hdel(self, key: str, field: str, namespace: str = 'trading_bot') -> bool` - Line 468
- `async lpush(self, key: str, value: Any, namespace: str = 'trading_bot') -> int` - Line 483
- `async rpush(self, key: str, value: Any, namespace: str = 'trading_bot') -> int` - Line 504
- `async lrange(self, key: str, start: int = 0, end: int = Any, namespace: str = 'trading_bot') -> list[Any]` - Line 525
- `async sadd(self, key: str, value: Any, namespace: str = 'trading_bot') -> bool` - Line 551
- `async smembers(self, key: str, namespace: str = 'trading_bot') -> list[Any]` - Line 572
- `async store_market_data(self, symbol: str, data: dict[str, Any], ttl: int = 300) -> bool` - Line 596
- `async get_market_data(self, symbol: str) -> dict[str, Any] | None` - Line 601
- `async store_position(self, bot_id: str, position: dict[str, Any]) -> bool` - Line 606
- `async get_position(self, bot_id: str) -> dict[str, Any] | None` - Line 610
- `async store_balance(self, user_id: str, exchange: str, balance: dict[str, Any]) -> bool` - Line 614
- `async get_balance(self, user_id: str, exchange: str) -> dict[str, Any] | None` - Line 619
- `async store_cache(self, key: str, value: Any, ttl: int = 3600) -> bool` - Line 624
- `async get_cache(self, key: str) -> Any | None` - Line 628
- `async ping(self) -> bool` - Line 633
- `async info(self) -> dict[str, Any]` - Line 645
- `async health_check(self) -> bool` - Line 657

### Implementation: `CapitalAuditLogRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for CapitalAuditLog entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_strategy(self, strategy_id: str) -> list[CapitalAuditLog]` - Line 29
- `async get_by_exchange(self, exchange: str) -> list[CapitalAuditLog]` - Line 33
- `async get_by_date_range(self, start_date: datetime, end_date: datetime) -> list[CapitalAuditLog]` - Line 37

### Implementation: `ExecutionAuditLogRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for ExecutionAuditLog entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_order(self, order_id: str) -> list[ExecutionAuditLog]` - Line 59
- `async get_by_execution_status(self, status: str) -> list[ExecutionAuditLog]` - Line 63
- `async get_failed_executions(self) -> list[ExecutionAuditLog]` - Line 67

### Implementation: `RiskAuditLogRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for RiskAuditLog entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_risk_type(self, risk_type: str) -> list[RiskAuditLog]` - Line 85
- `async get_high_severity_risks(self) -> list[RiskAuditLog]` - Line 89
- `async get_critical_risks(self) -> list[RiskAuditLog]` - Line 93

### Implementation: `PerformanceAuditLogRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for PerformanceAuditLog entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_strategy(self, strategy_id: str) -> list[PerformanceAuditLog]` - Line 111
- `async get_by_metric_type(self, metric_type: str) -> list[PerformanceAuditLog]` - Line 115

### Implementation: `DatabaseRepository` ✅

**Inherits**: CoreBaseRepository
**Purpose**: Database repository implementation using core BaseRepository
**Status**: Complete

**Implemented Methods:**
- `async get_by(self, **kwargs)` - Line 160
- `async get(self, entity_id: Any)` - Line 172
- `async get_all(self, ...) -> list` - Line 176
- `async exists(self, entity_id: Any) -> bool` - Line 198
- `async soft_delete(self, entity_id: Any, deleted_by: str | None = None) -> bool` - Line 207
- `async begin(self)` - Line 220
- `async commit(self)` - Line 224
- `async rollback(self)` - Line 228
- `async refresh(self, entity)` - Line 232

### Implementation: `RepositoryInterface` ✅

**Purpose**: Interface for repository pattern
**Status**: Complete

### Implementation: `BotRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for Bot entities
**Status**: Complete

**Implemented Methods:**
- `async get_active_bots(self) -> list[Bot]` - Line 25
- `async get_running_bots(self) -> list[Bot]` - Line 29
- `async get_bot_by_name(self, name: str) -> Bot | None` - Line 33
- `async start_bot(self, bot_id: str) -> bool` - Line 37
- `async stop_bot(self, bot_id: str) -> bool` - Line 46
- `async pause_bot(self, bot_id: str) -> bool` - Line 55
- `async update_bot_status(self, bot_id: str, status: str) -> bool` - Line 64
- `async update_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> bool` - Line 68
- `async get_bot_performance(self, bot_id: str) -> dict[str, Any]` - Line 72

### Implementation: `StrategyRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for Strategy entities
**Status**: Complete

**Implemented Methods:**
- `async get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]` - Line 107
- `async get_strategies_by_bot(self, bot_id: str) -> list[Strategy]` - Line 115
- `async get_strategy_by_name(self, bot_id: str, name: str) -> Strategy | None` - Line 119
- `async activate_strategy(self, strategy_id: str) -> bool` - Line 123
- `async deactivate_strategy(self, strategy_id: str) -> bool` - Line 132
- `async update_strategy_params(self, strategy_id: str, params: dict[str, Any]) -> bool` - Line 141
- `async update_strategy_metrics(self, strategy_id: str, metrics: dict[str, Any]) -> bool` - Line 150

### Implementation: `SignalRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for Signal entities
**Status**: Complete

**Implemented Methods:**
- `async get_unexecuted_signals(self, strategy_id: str | None = None) -> list[Signal]` - Line 163
- `async get_signals_by_strategy(self, strategy_id: str, limit: int = 100) -> list[Signal]` - Line 171
- `async get_recent_signals(self, hours: int = 24, strategy_id: str | None = None) -> list[Signal]` - Line 177
- `async mark_signal_executed(self, signal_id: str, order_id: str, execution_time: Decimal) -> bool` - Line 184
- `async update_signal_outcome(self, signal_id: str, outcome: str, pnl: Decimal | None = None) -> bool` - Line 197
- `async get_signal_statistics(self, strategy_id: str, since: datetime | None = None) -> dict[str, Any]` - Line 213

### Implementation: `BotLogRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for BotLog entities
**Status**: Complete

**Implemented Methods:**
- `async get_logs_by_bot(self, bot_id: str, level: str | None = None, limit: int = 100) -> list[BotLog]` - Line 261
- `async get_error_logs(self, bot_id: str | None = None, hours: int = 24) -> list[BotLog]` - Line 271
- `async log_event(self, ...) -> BotLog` - Line 284
- `async cleanup_old_logs(self, days: int = 30) -> int` - Line 299

### Implementation: `BotInstanceRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot instance operations
**Status**: Complete

**Implemented Methods:**
- `async get_by_bot_id(self, bot_id: str) -> BotInstance | None` - Line 37
- `async get_active_bots(self) -> list[BotInstance]` - Line 50
- `async get_bots_by_status(self, status: str) -> list[BotInstance]` - Line 69
- `async get_bots_by_strategy(self, strategy_name: str) -> list[BotInstance]` - Line 86
- `async get_bots_by_exchange(self, exchange: str) -> list[BotInstance]` - Line 103
- `async update_bot_status(self, ...) -> BotInstance | None` - Line 120
- `async update_heartbeat(self, bot_id: str) -> bool` - Line 153
- `async get_stale_bots(self, heartbeat_timeout_seconds: int = 300) -> list[BotInstance]` - Line 170
- `async get_error_bots(self, min_error_count: int = 1) -> list[BotInstance]` - Line 199
- `async reset_bot_errors(self, bot_id: str) -> bool` - Line 224
- `async get_bot_statistics(self) -> dict[str, Any]` - Line 243
- `async cleanup_old_bots(self, days: int = 30, status: str = 'stopped') -> int` - Line 300
- `async get_bots_by_symbol(self, symbol: str) -> list[BotInstance]` - Line 319
- `async get_profitable_bots(self, min_pnl: Decimal = Any) -> list[BotInstance]` - Line 334
- `async update_bot_metrics(self, ...) -> BotInstance | None` - Line 356

### Implementation: `BotMetricsRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot metrics and performance data operations
**Status**: Complete

**Implemented Methods:**
- `async store_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> None` - Line 40
- `async get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]` - Line 68
- `async get_latest_metrics(self, bot_id: str) -> dict[str, Any] | None` - Line 108
- `async get_bot_health_checks(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]` - Line 121
- `async store_bot_health_analysis(self, bot_id: str, analysis: dict[str, Any]) -> None` - Line 162
- `async get_bot_health_analyses(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]` - Line 189
- `async get_recent_health_analyses(self, hours: int = 1) -> list[dict[str, Any]]` - Line 227
- `async archive_bot_record(self, bot_id: str) -> None` - Line 263
- `async get_active_bots(self) -> list[str]` - Line 296
- `async get_bot_performance_summary(self, bot_id: str, hours: int = 24) -> dict[str, Any]` - Line 325

### Implementation: `BotResourcesRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot resource management operations
**Status**: Complete

**Implemented Methods:**
- `async store_resource_allocation(self, allocation: dict[str, Any]) -> None` - Line 40
- `async update_resource_allocation_status(self, bot_id: str, status: str) -> None` - Line 69
- `async store_resource_usage(self, usage: dict[str, Any]) -> None` - Line 98
- `async store_resource_reservation(self, reservation: dict[str, Any]) -> None` - Line 126
- `async update_resource_reservation_status(self, reservation_id: str, status: str) -> None` - Line 160
- `async store_resource_usage_history(self, usage_entry: dict[str, Any]) -> None` - Line 193
- `async store_optimization_suggestion(self, suggestion: dict[str, Any]) -> None` - Line 202
- `async get_resource_allocations(self, bot_id: str) -> list[dict[str, Any]]` - Line 237
- `async get_resource_usage_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]` - Line 276
- `async cleanup_expired_reservations(self) -> int` - Line 322

### Implementation: `CapitalAllocationRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for CapitalAllocationDB entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_strategy(self, strategy_id: str) -> list[CapitalAllocationDB]` - Line 33
- `async get_by_exchange(self, exchange: str) -> list[CapitalAllocationDB]` - Line 37
- `async find_by_strategy_exchange(self, strategy_id: str, exchange: str) -> CapitalAllocationDB | None` - Line 41
- `async get_total_allocated_by_strategy(self, strategy_id: str) -> Decimal` - Line 52
- `async get_available_capital_by_exchange(self, exchange: str) -> Decimal` - Line 57

### Implementation: `FundFlowRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for FundFlowDB entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_from_strategy(self, strategy_id: str) -> list[FundFlowDB]` - Line 76
- `async get_by_to_strategy(self, strategy_id: str) -> list[FundFlowDB]` - Line 82
- `async get_by_exchange_flow(self, from_exchange: str, to_exchange: str) -> list[FundFlowDB]` - Line 88
- `async get_by_reason(self, reason: str) -> list[FundFlowDB]` - Line 93
- `async get_by_currency(self, currency: str) -> list[FundFlowDB]` - Line 97

### Implementation: `CurrencyExposureRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for CurrencyExposureDB entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_currency(self, currency: str) -> CurrencyExposureDB | None` - Line 115
- `async get_hedging_required(self) -> list[CurrencyExposureDB]` - Line 119
- `async get_total_exposure(self) -> Decimal` - Line 123

### Implementation: `ExchangeAllocationRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for ExchangeAllocationDB entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_exchange(self, exchange: str) -> ExchangeAllocationDB | None` - Line 142
- `async get_total_allocated(self) -> Decimal` - Line 146
- `async get_total_available(self) -> Decimal` - Line 151
- `async get_underutilized_exchanges(self, threshold: Decimal = Any) -> list[ExchangeAllocationDB]` - Line 157

### Implementation: `CapitalAuditLogRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for capital audit log entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_operation_id(self, operation_id: str) -> CapitalAuditLog | None` - Line 183
- `async get_by_strategy(self, strategy_id: str) -> list[CapitalAuditLog]` - Line 187
- `async get_by_exchange(self, exchange: str) -> list[CapitalAuditLog]` - Line 191
- `async get_failed_operations(self, limit: int = 100) -> list[CapitalAuditLog]` - Line 195

### Implementation: `FeatureRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for FeatureRecord entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_symbol(self, symbol: str) -> list[FeatureRecord]` - Line 30
- `async get_by_feature_type(self, feature_type: str) -> list[FeatureRecord]` - Line 34
- `async get_by_symbol_and_type(self, symbol: str, feature_type: str) -> list[FeatureRecord]` - Line 40
- `async get_latest_feature(self, symbol: str, feature_type: str, feature_name: str) -> FeatureRecord | None` - Line 47
- `async get_features_by_date_range(self, symbol: str, start_date: datetime, end_date: datetime) -> list[FeatureRecord]` - Line 58

### Implementation: `DataQualityRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for DataQualityRecord entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_symbol(self, symbol: str) -> list[DataQualityRecord]` - Line 85
- `async get_by_data_source(self, data_source: str) -> list[DataQualityRecord]` - Line 89
- `async get_poor_quality_records(self, threshold: Decimal = Any) -> list[DataQualityRecord]` - Line 95
- `async get_latest_quality_check(self, symbol: str, data_source: str) -> DataQualityRecord | None` - Line 102
- `async get_quality_trend(self, symbol: str, days: int = 30) -> list[DataQualityRecord]` - Line 113

### Implementation: `DataPipelineRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for DataPipelineRecord entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_pipeline_name(self, pipeline_name: str) -> list[DataPipelineRecord]` - Line 142
- `async get_by_status(self, status: str) -> list[DataPipelineRecord]` - Line 148
- `async get_running_pipelines(self) -> list[DataPipelineRecord]` - Line 152
- `async get_failed_pipelines(self) -> list[DataPipelineRecord]` - Line 156
- `async get_latest_execution(self, pipeline_name: str) -> DataPipelineRecord | None` - Line 160
- `async get_pipeline_performance(self, pipeline_name: str, days: int = 30) -> list[DataPipelineRecord]` - Line 167

### Implementation: `MarketDataRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for MarketDataRecord entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_symbol(self, symbol: str) -> list[MarketDataRecord]` - Line 28
- `async get_by_exchange(self, exchange: str) -> list[MarketDataRecord]` - Line 34
- `async get_by_symbol_and_exchange(self, symbol: str, exchange: str) -> list[MarketDataRecord]` - Line 40
- `async get_latest_price(self, symbol: str, exchange: str) -> MarketDataRecord | None` - Line 48
- `async get_ohlc_data(self, symbol: str, exchange: str, start_time: datetime, end_time: datetime) -> list[MarketDataRecord]` - Line 55
- `async get_recent_data(self, symbol: str, exchange: str, hours: int = 24) -> list[MarketDataRecord]` - Line 68
- `async get_by_data_source(self, data_source: str) -> list[MarketDataRecord]` - Line 80
- `async get_poor_quality_data(self, threshold: Decimal = Any) -> list[MarketDataRecord]` - Line 86
- `async get_invalid_data(self) -> list[MarketDataRecord]` - Line 93
- `async cleanup_old_data(self, days: int = 90) -> int` - Line 98
- `async save_ticker(self, exchange: str, symbol: str, data: dict[str, Any]) -> None` - Line 104
- `async get_volume_leaders(self, exchange: str | None = None, limit: int = 10) -> list[MarketDataRecord]` - Line 120
- `async get_price_changes(self, symbol: str, exchange: str, hours: int = 24) -> tuple[Decimal | None, Decimal | None]` - Line 132

### Implementation: `MLPredictionRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for ML predictions
**Status**: Complete

**Implemented Methods:**
- `async get_by_model_and_symbol(self, model_name: str, symbol: str, limit: int = 100) -> list[MLPrediction]` - Line 37
- `async get_recent_predictions(self, ...) -> list[MLPrediction]` - Line 64
- `async get_prediction_accuracy(self, model_name: str, symbol: str | None = None, days: int = 30) -> dict[str, Any]` - Line 100
- `async update_with_actual(self, prediction_id: int, actual_value: Decimal) -> MLPrediction | None` - Line 152

### Implementation: `MLModelMetadataRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for ML model metadata
**Status**: Complete

**Implemented Methods:**
- `async get_latest_model(self, model_name: str, model_type: str) -> MLModelMetadata | None` - Line 193
- `async get_active_models(self) -> list[MLModelMetadata]` - Line 218
- `async get_by_version(self, model_name: str, version: int) -> MLModelMetadata | None` - Line 232
- `async deactivate_old_versions(self, model_name: str, keep_versions: int = 3) -> int` - Line 253

### Implementation: `MLTrainingJobRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for ML training jobs
**Status**: Complete

**Implemented Methods:**
- `async get_running_jobs(self) -> list[MLTrainingJob]` - Line 314
- `async get_job_by_model(self, model_name: str, status: str | None = None) -> list[MLTrainingJob]` - Line 328
- `async update_job_status(self, ...) -> MLTrainingJob | None` - Line 351
- `async get_successful_jobs(self, days: int = 30, limit: int = 100) -> list[MLTrainingJob]` - Line 384

### Implementation: `MLRepository` ✅

**Purpose**: Unified ML repository providing data access to all ML-related repositories
**Status**: Complete

**Implemented Methods:**

### Implementation: `RiskMetricsRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for risk metrics data access
**Status**: Complete

**Implemented Methods:**
- `async get_historical_returns(self, symbol: str, days: int) -> list[Decimal]` - Line 34
- `async get_price_history(self, symbol: str, days: int) -> list[Decimal]` - Line 39
- `async get_portfolio_positions(self) -> list[Position]` - Line 44
- `async save_risk_metrics(self, metrics: RiskMetrics) -> None` - Line 52
- `async get_correlation_data(self, symbols: list[str], days: int) -> dict[str, list[Decimal]]` - Line 61

### Implementation: `PortfolioRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for portfolio data access
**Status**: Complete

**Implemented Methods:**
- `async get_current_positions(self) -> list[Position]` - Line 79
- `async get_portfolio_value(self) -> Decimal` - Line 87
- `async get_position_history(self, symbol: str, days: int) -> list[Position]` - Line 95
- `async update_portfolio_limits(self, limits: dict[str, Any]) -> None` - Line 103

### Implementation: `RiskMetricsRepositoryImpl` ✅

**Inherits**: RiskMetricsRepositoryInterface
**Purpose**: Implementation of risk metrics repository interface
**Status**: Complete

**Implemented Methods:**
- `async get_historical_returns(self, symbol: str, days: int) -> list[Decimal]` - Line 117
- `async get_price_history(self, symbol: str, days: int) -> list[Decimal]` - Line 121
- `async get_portfolio_positions(self) -> list[Position]` - Line 125
- `async save_risk_metrics(self, metrics: RiskMetrics) -> None` - Line 129
- `async get_correlation_data(self, symbols: list[str], days: int) -> dict[str, list[Decimal]]` - Line 133

### Implementation: `PortfolioRepositoryImpl` ✅

**Inherits**: PortfolioRepositoryInterface
**Purpose**: Implementation of portfolio repository interface
**Status**: Complete

**Implemented Methods:**
- `async get_current_positions(self) -> list[Position]` - Line 145
- `async get_portfolio_value(self) -> Decimal` - Line 149
- `async get_position_history(self, symbol: str, days: int) -> list[Position]` - Line 153
- `async update_portfolio_limits(self, limits: dict[str, Any]) -> None` - Line 157

### Implementation: `DatabaseServiceRepository` ✅

**Inherits**: BaseRepository[T, K]
**Purpose**: Database repository implementation using the repository pattern
**Status**: Complete

**Implemented Methods:**

### Implementation: `StateSnapshotRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateSnapshot entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_name_prefix(self, name_prefix: str) -> list[StateSnapshot]` - Line 30
- `async get_by_snapshot_type(self, snapshot_type: str) -> list[StateSnapshot]` - Line 36
- `async get_latest_snapshot(self, name_prefix: str, snapshot_type: str | None = None) -> StateSnapshot | None` - Line 40
- `async get_by_schema_version(self, schema_version: str) -> list[StateSnapshot]` - Line 51
- `async cleanup_old_snapshots(self, name_prefix: str, keep_count: int = 10) -> int` - Line 57

### Implementation: `StateCheckpointRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateCheckpoint entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_name_prefix(self, name_prefix: str) -> list[StateCheckpoint]` - Line 84
- `async get_by_checkpoint_type(self, checkpoint_type: str) -> list[StateCheckpoint]` - Line 90
- `async get_latest_checkpoint(self, name_prefix: str) -> StateCheckpoint | None` - Line 96
- `async get_by_status(self, status: str) -> list[StateCheckpoint]` - Line 103

### Implementation: `StateHistoryRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateHistory entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_state(self, state_type: str, state_id: str) -> list[StateHistory]` - Line 121
- `async get_by_operation(self, operation: str) -> list[StateHistory]` - Line 128
- `async get_recent_changes(self, state_type: str, state_id: str, hours: int = 24) -> list[StateHistory]` - Line 132
- `async get_by_component(self, source_component: str) -> list[StateHistory]` - Line 145

### Implementation: `StateMetadataRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateMetadata entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_state(self, state_type: str, state_id: str) -> StateMetadata | None` - Line 166
- `async get_by_state_type(self, state_type: str) -> list[StateMetadata]` - Line 170
- `async get_critical_states(self, state_type: str | None = None) -> list[StateMetadata]` - Line 174
- `async get_hot_states(self, state_type: str | None = None) -> list[StateMetadata]` - Line 181

### Implementation: `StateBackupRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateBackup entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_name_prefix(self, name_prefix: str) -> list[StateBackup]` - Line 202
- `async get_by_backup_type(self, backup_type: str) -> list[StateBackup]` - Line 208
- `async get_latest_backup(self, name_prefix: str) -> StateBackup | None` - Line 212
- `async get_verified_backups(self, name_prefix: str) -> list[StateBackup]` - Line 219
- `async cleanup_old_backups(self, name_prefix: str, keep_days: int = 30) -> int` - Line 226

### Implementation: `AlertRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for Alert entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_user(self, user_id: str) -> list[Alert]` - Line 27
- `async get_unread_alerts(self, user_id: str) -> list[Alert]` - Line 31
- `async get_by_severity(self, severity: str) -> list[Alert]` - Line 37
- `async get_critical_alerts(self) -> list[Alert]` - Line 41
- `async get_by_type(self, alert_type: str) -> list[Alert]` - Line 45
- `async mark_as_read(self, alert_id: str) -> bool` - Line 51
- `async mark_all_read(self, user_id: str) -> int` - Line 55

### Implementation: `AuditLogRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for AuditLog entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_user(self, user_id: str) -> list[AuditLog]` - Line 75
- `async get_by_action(self, action: str) -> list[AuditLog]` - Line 79
- `async get_by_resource_type(self, resource_type: str) -> list[AuditLog]` - Line 83
- `async get_by_resource(self, resource_type: str, resource_id: str) -> list[AuditLog]` - Line 89
- `async get_recent_logs(self, hours: int = 24) -> list[AuditLog]` - Line 94

### Implementation: `PerformanceMetricsRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for PerformanceMetrics entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_bot(self, bot_id: str) -> list[PerformanceMetrics]` - Line 113
- `async get_latest_metrics(self, bot_id: str) -> PerformanceMetrics | None` - Line 117
- `async get_metrics_by_date_range(self, bot_id: str, start_date: datetime, end_date: datetime) -> list[PerformanceMetrics]` - Line 122
- `async get_top_performing_bots(self, limit: int = 10) -> list[PerformanceMetrics]` - Line 131

### Implementation: `BalanceSnapshotRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for BalanceSnapshot entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_user(self, user_id: str) -> list[BalanceSnapshot]` - Line 152
- `async get_by_exchange(self, exchange: str) -> list[BalanceSnapshot]` - Line 156
- `async get_by_currency(self, currency: str) -> list[BalanceSnapshot]` - Line 160
- `async get_latest_snapshot(self, user_id: str, exchange: str, currency: str) -> BalanceSnapshot | None` - Line 164
- `async get_balance_history(self, user_id: str, exchange: str, currency: str, days: int = 30) -> list[BalanceSnapshot]` - Line 175

### Implementation: `OrderRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for Order entities
**Status**: Complete

**Implemented Methods:**
- `async get_active_orders(self, bot_id: str | None = None, symbol: str | None = None) -> list[Order]` - Line 26
- `async get_by_exchange_id(self, exchange: str, exchange_order_id: str) -> Order | None` - Line 39
- `async update_order_status(self, order_id: str, status: str) -> bool` - Line 43
- `async get_orders_by_position(self, position_id: str) -> list[Order]` - Line 47
- `async get_recent_orders(self, hours: int = 24, bot_id: str | None = None) -> list[Order]` - Line 51

### Implementation: `PositionRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for Position entities
**Status**: Complete

**Implemented Methods:**
- `async get_open_positions(self, bot_id: str | None = None, symbol: str | None = None) -> list[Position]` - Line 70
- `async get_position_by_symbol(self, bot_id: str, symbol: str, side: str) -> Position | None` - Line 83
- `async update_position_status(self, position_id: str, status: str, **fields) -> bool` - Line 87
- `async update_position_fields(self, position_id: str, **fields) -> bool` - Line 93

### Implementation: `TradeRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for Trade entities
**Status**: Complete

**Implemented Methods:**
- `async get_profitable_trades(self, bot_id: str | None = None) -> list[Trade]` - Line 110
- `async get_trades_by_symbol(self, symbol: str, bot_id: str | None = None) -> list[Trade]` - Line 119
- `async get_trades_by_bot_and_date(self, bot_id: str, since: datetime | None = None) -> list[Trade]` - Line 126
- `async create_from_position(self, position: Position, exit_order: Order) -> Trade` - Line 140

### Implementation: `OrderFillRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for OrderFill entities
**Status**: Complete

**Implemented Methods:**
- `async get_fills_by_order(self, order_id: str) -> list[OrderFill]` - Line 188
- `async get_total_filled(self, order_id: str) -> dict[str, Decimal]` - Line 192
- `async create_fill(self, ...) -> OrderFill` - Line 223

### Implementation: `UserRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for User entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_username(self, username: str) -> User | None` - Line 21
- `async get_by_email(self, email: str) -> User | None` - Line 25
- `async get_active_users(self) -> list[User]` - Line 29
- `async get_verified_users(self) -> list[User]` - Line 33
- `async get_admin_users(self) -> list[User]` - Line 37
- `async activate_user(self, user_id: str) -> bool` - Line 41
- `async deactivate_user(self, user_id: str) -> bool` - Line 45
- `async verify_user(self, user_id: str) -> bool` - Line 49

### Implementation: `RepositoryUtils` ✅

**Inherits**: Generic[T]
**Purpose**: Common utilities for repository operations
**Status**: Complete

**Implemented Methods:**
- `async update_entity_status(repository, ...) -> bool` - Line 22
- `async update_entity_fields(repository: Any, entity_id: str, entity_name: str, **fields: Any) -> bool` - Line 55
- `async get_entities_by_field(repository, ...) -> list[T]` - Line 86
- `async get_entities_by_multiple_fields(repository: Any, filters: dict[str, Any], order_by: str = '-created_at') -> list[T]` - Line 109
- `async get_recent_entities(repository, ...) -> list[T]` - Line 130
- `async mark_entity_field(repository, ...) -> bool` - Line 164
- `async bulk_mark_entities(repository, ...) -> int` - Line 194
- `async get_total_by_field_aggregation(session, ...) -> Decimal` - Line 225
- `async get_latest_entity_by_field(repository: Any, field_name: str, field_value: Any) -> T | None` - Line 267
- `async cleanup_old_entities(session, ...) -> int` - Line 291
- `async execute_time_based_query(session, ...) -> list[T]` - Line 336
- `async execute_date_range_query(session, ...) -> list[T]` - Line 409

### Implementation: `RepositoryFactory` ✅

**Inherits**: RepositoryFactoryInterface
**Purpose**: Factory for creating repository instances using dependency injection
**Status**: Complete

**Implemented Methods:**
- `create_repository(self, repository_class: type[Any], session: Any) -> Any` - Line 31
- `register_repository(self, name: str, repository_class: type[R]) -> None` - Line 67
- `is_repository_registered(self, name: str) -> bool` - Line 78
- `get_registered_repository(self, name: str) -> type[R] | None` - Line 90
- `configure_dependencies(self, dependency_injector) -> None` - Line 102
- `list_registered_repositories(self) -> list[str]` - Line 112
- `clear_registrations(self) -> None` - Line 121

### Implementation: `DatabaseSeeder` ✅

**Purpose**: Handles database seeding for development environment
**Status**: Complete

**Implemented Methods:**
- `async seed_users(self, session: AsyncSession) -> list[User]` - Line 200
- `async seed_bot_instances(self, session: AsyncSession, users: list[User]) -> list[BotInstance]` - Line 241
- `async seed_strategies(self, session: AsyncSession, bots: list[BotInstance]) -> list[Strategy]` - Line 296
- `async seed_exchange_credentials(self, session: AsyncSession, users: list[User]) -> list[dict[str, Any]]` - Line 348
- `async seed_sample_trades(self, session: AsyncSession, bots: list[BotInstance]) -> None` - Line 386
- `async seed_all(self) -> None` - Line 444

### Implementation: `DatabaseService` ✅

**Inherits**: BaseService, DatabaseServiceInterface
**Purpose**: Simple database service implementing service layer pattern
**Status**: Complete

**Implemented Methods:**
- `config_service(self) -> Any` - Line 94
- `validation_service(self) -> Any` - Line 99
- `async start(self) -> None` - Line 103
- `async stop(self) -> None` - Line 132
- `async get_health_status(self) -> HealthStatus` - Line 163
- `async health_check(self) -> HealthCheckResult` - Line 189
- `async create_entity(self, entity: T, processing_mode: str = 'stream') -> T` - Line 246
- `async get_entity_by_id(self, model_class: type[T], entity_id: K) -> T | None` - Line 315
- `async update_entity(self, entity: T) -> T` - Line 350
- `async delete_entity(self, model_class: type[T], entity_id: K) -> bool` - Line 379
- `async list_entities(self, ...) -> list[T]` - Line 412
- `async count_entities(self, model_class: type[T] | None = None, filters: dict[str, Any] | None = None) -> int` - Line 526
- `async bulk_create(self, entities: list[T]) -> list[T]` - Line 566
- `async transaction(self) -> AsyncGenerator[AsyncSession, None]` - Line 600
- `async get_session(self) -> AsyncGenerator[AsyncSession, None]` - Line 617
- `async get_health_status(self) -> HealthStatus` - Line 629
- `get_performance_metrics(self) -> dict[str, Any]` - Line 642
- `async execute_query(self, query: str, params: dict[str, Any] | None = None) -> Any` - Line 649
- `async get_connection_pool_status(self) -> dict[str, Any]` - Line 663

### Implementation: `BotService` ✅

**Inherits**: BaseService, BotMetricsServiceInterface
**Purpose**: Service layer for bot operations with business logic
**Status**: Complete

**Implemented Methods:**
- `async get_active_bots(self) -> list[dict[str, Any]]` - Line 35
- `async archive_bot_record(self, bot_id: str) -> bool` - Line 72
- `async get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]` - Line 112
- `async store_bot_metrics(self, metrics_record: dict[str, Any]) -> bool` - Line 157

### Implementation: `MarketDataService` ✅

**Inherits**: BaseService, MarketDataServiceInterface
**Purpose**: Service layer for market data operations with business logic
**Status**: Complete

**Implemented Methods:**
- `async get_latest_price(self, symbol: str) -> Decimal | None` - Line 24
- `async get_historical_data(self, ...) -> list[dict[str, Any]]` - Line 47

### Implementation: `MLService` ✅

**Inherits**: BaseService, MLServiceInterface
**Purpose**: Service layer for ML operations with business logic
**Status**: Complete

**Implemented Methods:**
- `async get_model_performance_summary(self, model_name: str, days: int = 30) -> dict[str, Any]` - Line 33
- `async validate_model_deployment(self, model_name: str, version: int) -> bool` - Line 81
- `async get_model_recommendations(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]` - Line 122

### Implementation: `ServiceRegistry` ✅

**Purpose**: Registry for managing service instances and dependencies
**Status**: Complete

**Implemented Methods:**
- `register_service(self, name: str, service_instance: Any) -> None` - Line 21
- `register_factory(self, name: str, factory: Callable[[], Any]) -> None` - Line 32
- `get_service(self, name: str) -> Any` - Line 43
- `has_service(self, name: str) -> bool` - Line 70
- `clear_services(self) -> None` - Line 82
- `list_services(self) -> list[str]` - Line 88

### Implementation: `TradingService` ✅

**Inherits**: BaseService, TradingDataServiceInterface
**Purpose**: Service layer for trading operations with business logic
**Status**: Complete

**Implemented Methods:**
- `async cancel_order(self, order_id: str, reason: str = 'User requested') -> bool` - Line 39
- `async close_position(self, position_id: str, close_price: Decimal) -> bool` - Line 70
- `async get_trades_by_bot(self, ...) -> list[Trade]` - Line 105
- `async get_positions_by_bot(self, bot_id: str) -> list[Position]` - Line 132
- `async calculate_total_pnl(self, ...) -> Decimal` - Line 142
- `async update_position_price(self, position_id: str, current_price: Decimal) -> bool` - Line 191
- `async create_trade(self, trade_data: dict) -> dict` - Line 222
- `async get_positions(self, strategy_id: str | None = None, symbol: str | None = None) -> list[dict]` - Line 268
- `async get_trade_statistics(self, bot_id: str, since: datetime | None = None) -> dict[str, Any]` - Line 309
- `async get_total_exposure(self, bot_id: str) -> dict[str, Decimal]` - Line 358
- `async get_order_fill_summary(self, order_id: str) -> dict[str, Decimal]` - Line 393

### Implementation: `UnitOfWork` ✅

**Purpose**: Unit of Work pattern for managing database transactions with service layer integration
**Status**: Complete

**Implemented Methods:**
- `commit(self, processing_mode: str = 'stream')` - Line 241
- `rollback(self)` - Line 292
- `close(self)` - Line 301
- `refresh(self, entity)` - Line 327
- `flush(self)` - Line 332
- `savepoint(self)` - Line 338

### Implementation: `AsyncUnitOfWork` ✅

**Purpose**: Async Unit of Work pattern for managing database transactions with service layer pattern
**Status**: Complete

**Implemented Methods:**
- `async commit(self, processing_mode: str = 'stream')` - Line 572
- `async rollback(self)` - Line 589
- `async close(self)` - Line 603
- `async refresh(self, entity)` - Line 618
- `async flush(self)` - Line 623
- `async savepoint(self)` - Line 629

### Implementation: `UnitOfWorkFactory` ✅

**Inherits**: UnitOfWorkFactoryInterface
**Purpose**: Factory for creating Unit of Work instances
**Status**: Complete

**Implemented Methods:**
- `create(self) -> UnitOfWork` - Line 712
- `create_async(self) -> AsyncUnitOfWork` - Line 716
- `transaction(self)` - Line 725
- `async async_transaction(self)` - Line 732
- `configure_dependencies(self, dependency_injector) -> None` - Line 738

### Implementation: `UnitOfWorkExample` ✅

**Purpose**: Example demonstrating Unit of Work usage patterns
**Status**: Complete

**Implemented Methods:**
- `async example_transaction(self, entity_data: dict)` - Line 753
- `async example_multi_service_operation(self, entity_id: str)` - Line 763

## COMPLETE API REFERENCE

### File: connection.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import DataSourceError`
- `from src.core.logging import PerformanceMonitor`
- `from src.core.logging import get_logger`
- `from src.error_handling.decorators import with_circuit_breaker`

#### Class: `DatabaseConnectionManager`

**Purpose**: Manages database connections with health monitoring and reconnection

```python
class DatabaseConnectionManager:
    def __init__(self, config: Config) -> None  # Line 52
    def set_test_schema(self, schema: str) -> None  # Line 63
    def _setup_test_schema_listeners(self) -> None  # Line 80
    async def initialize(self) -> None  # Line 131
    def _start_health_monitoring(self) -> None  # Line 143
    async def _health_check_loop(self) -> None  # Line 157
    async def _setup_postgresql(self) -> None  # Line 199
    async def _setup_redis(self) -> None  # Line 292
    async def _setup_influxdb(self) -> None  # Line 312
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]  # Line 326
    def get_sync_session(self) -> Session  # Line 365
    async def get_redis_client(self) -> redis.Redis  # Line 385
    def get_influxdb_client(self) -> InfluxDBClient  # Line 391
    async def close(self) -> None  # Line 397
    async def _stop_health_monitoring(self, health_task: Task[None] | None) -> None  # Line 418
    async def _prepare_close_tasks(self, connections: dict[str, Any]) -> list[Any]  # Line 427
    def _create_redis_close_task(self, redis_client) -> Any  # Line 450
    def _close_influxdb_task(self, influxdb_client) -> Any  # Line 465
    async def _execute_close_tasks(self, close_tasks: list[Any]) -> None  # Line 474
    def _clear_connection_references(self) -> None  # Line 483
    def is_healthy(self) -> bool  # Line 491
    def async_session_maker(self)  # Line 496
    def sync_session_maker(self)  # Line 501
    async def get_connection(self)  # Line 506
    async def get_pool_status(self) -> dict[str, int]  # Line 514
```

#### Functions:

```python
async def initialize_database(config: Config) -> None  # Line 560
async def init_database(config: Config) -> None  # Line 567
async def close_database() -> None  # Line 572
async def get_async_session() -> AsyncGenerator[AsyncSession, None]  # Line 583
async def get_db_session() -> AsyncGenerator[AsyncSession, None]  # Line 593
def get_sync_session() -> Session  # Line 599
async def get_redis_client() -> redis.Redis  # Line 607
def get_influxdb_client() -> InfluxDBClient  # Line 615
def is_database_healthy() -> bool  # Line 623
def set_connection_manager(manager: DatabaseConnectionManager) -> None  # Line 630
async def execute_query(query: str, params: dict[str, Any] | None = None) -> Any  # Line 645
async def health_check() -> dict[str, bool]  # Line 652
async def debug_connection_info() -> dict[str, Any]  # Line 683
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`
- `from src.database.connection import DatabaseConnectionManager`

#### Functions:

```python
def register_database_services(injector: DependencyInjector) -> None  # Line 42
def _register_connection_manager(injector: DependencyInjector) -> None  # Line 61
def _register_repository_factory(injector: DependencyInjector) -> None  # Line 88
def _register_session_factories(injector: DependencyInjector) -> None  # Line 104
def _register_async_session_factory(injector: DependencyInjector) -> None  # Line 112
def _register_sync_session_factory(injector: DependencyInjector) -> None  # Line 124
def _register_database_service_factory(injector: DependencyInjector) -> None  # Line 136
def _create_database_service_with_deps(injector: DependencyInjector) -> 'DatabaseService'  # Line 150
def _register_database_interfaces(injector: DependencyInjector) -> None  # Line 171
def configure_database_dependencies(injector: DependencyInjector | None = None) -> DependencyInjector  # Line 367
def get_database_service(injector: DependencyInjector) -> 'DatabaseService'  # Line 394
def get_database_manager(injector: DependencyInjector) -> 'DatabaseManager'  # Line 399
def get_uow_factory(injector: DependencyInjector) -> 'UnitOfWorkFactory'  # Line 404
def _register_database_service(injector: DependencyInjector) -> None  # Line 410
def _register_interface_implementations(injector: DependencyInjector) -> None  # Line 415
def _register_database_manager(injector: DependencyInjector) -> None  # Line 420
def _register_unit_of_work_factory(injector: DependencyInjector) -> None  # Line 425
def _register_unit_of_work_instances(injector: DependencyInjector) -> None  # Line 430
def _register_specialized_services(injector: DependencyInjector) -> None  # Line 435
def _register_repository_instances(injector: DependencyInjector) -> None  # Line 440
```

### File: influxdb_client.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import DataError`
- `from src.error_handling.decorators import with_circuit_breaker`
- `from src.error_handling.decorators import with_retry`

#### Class: `InfluxDBClientWrapper`

**Inherits**: BaseComponent
**Purpose**: InfluxDB client wrapper with trading-specific utilities

```python
class InfluxDBClientWrapper(BaseComponent):
    def __init__(self, url: str, token: str, org: str, bucket: str, config: Config | None = None)  # Line 35
    async def connect(self) -> None  # Line 49
    async def disconnect(self) -> None  # Line 63
    def _decimal_to_float(self, value: Any) -> float  # Line 80
    def _create_point(self, ...) -> Point  # Line 95
    async def write_point(self, point: Point) -> None  # Line 129
    async def write_points(self, points: list[Point]) -> None  # Line 157
    async def write_market_data(self, symbol: str, data: dict[str, Any], timestamp: datetime | None = None) -> None  # Line 186
    async def write_market_data_batch(self, data_list: list[dict[str, Any]], timestamp: datetime | None = None) -> None  # Line 204
    async def write_trade(self, trade_data: dict[str, Any], timestamp: datetime | None = None) -> None  # Line 243
    async def write_performance_metrics(self, bot_id: str, metrics: dict[str, Any], timestamp: datetime | None = None) -> None  # Line 266
    async def write_system_metrics(self, metrics: dict[str, Any], timestamp: datetime | None = None) -> None  # Line 288
    async def write_risk_metrics(self, bot_id: str, risk_data: dict[str, Any], timestamp: datetime | None = None) -> None  # Line 306
    async def query_market_data(self, symbol: str, start_time: datetime, end_time: datetime, limit: int = 1000) -> list[dict[str, Any]]  # Line 326
    async def query_trades(self, bot_id: str, start_time: datetime, end_time: datetime, limit: int = 1000) -> list[dict[str, Any]]  # Line 367
    async def query_performance_metrics(self, bot_id: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]  # Line 408
    def _parse_query_result(self, result) -> list[dict[str, Any]]  # Line 448
    async def get_daily_pnl(self, bot_id: str, date: datetime) -> dict[str, Decimal]  # Line 472
    async def get_win_rate(self, bot_id: str, start_time: datetime, end_time: datetime) -> Decimal  # Line 522
    async def health_check(self) -> bool  # Line 597
```

### File: interfaces.py

**Key Imports:**
- `from src.core.base.interfaces import HealthStatus`

#### Class: `DatabaseServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for database service operations

```python
class DatabaseServiceInterface(ABC):
    async def start(self) -> None  # Line 21
    async def stop(self) -> None  # Line 26
    async def create_entity(self, entity: T) -> T  # Line 31
    async def get_entity_by_id(self, model_class: type[T], entity_id: K) -> T | None  # Line 36
    async def update_entity(self, entity: T) -> T  # Line 41
    async def delete_entity(self, model_class: type[T], entity_id: K) -> bool  # Line 46
    async def list_entities(self, ...) -> list[T]  # Line 51
    async def count_entities(self, model_class: type[T] | None = None, filters: dict[str, Any] | None = None) -> int  # Line 65
    async def bulk_create(self, entities: list[T]) -> list[T]  # Line 72
    async def get_health_status(self) -> HealthStatus  # Line 77
    def get_performance_metrics(self) -> dict[str, Any]  # Line 82
```

#### Class: `TradingDataServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for trading-specific data operations

```python
class TradingDataServiceInterface(ABC):
    async def get_trades_by_bot(self, ...) -> list[Any]  # Line 91
    async def get_positions_by_bot(self, bot_id: str) -> list[Any]  # Line 103
    async def calculate_total_pnl(self, ...) -> Decimal  # Line 108
```

#### Class: `BotMetricsServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for bot metrics operations

```python
class BotMetricsServiceInterface(ABC):
    async def get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]  # Line 122
    async def store_bot_metrics(self, metrics_record: dict[str, Any]) -> bool  # Line 127
    async def get_active_bots(self) -> list[dict[str, Any]]  # Line 132
    async def archive_bot_record(self, bot_id: str) -> bool  # Line 137
```

#### Class: `HealthAnalyticsServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for health analytics operations

```python
class HealthAnalyticsServiceInterface(ABC):
    async def store_bot_health_analysis(self, health_analysis: dict[str, Any]) -> bool  # Line 146
    async def get_bot_health_analyses(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]  # Line 151
    async def get_recent_health_analyses(self, hours: int = 1) -> list[dict[str, Any]]  # Line 156
```

#### Class: `ResourceManagementServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for resource management operations

```python
class ResourceManagementServiceInterface(ABC):
    async def store_resource_allocation(self, allocation_record: dict[str, Any]) -> bool  # Line 165
    async def store_resource_usage(self, usage_record: dict[str, Any]) -> bool  # Line 170
    async def store_resource_reservation(self, reservation: dict[str, Any]) -> bool  # Line 175
    async def update_resource_allocation_status(self, bot_id: str, status: str) -> bool  # Line 180
```

#### Class: `RepositoryFactoryInterface`

**Inherits**: ABC
**Purpose**: Interface for repository factory operations

```python
class RepositoryFactoryInterface(ABC):
    def create_repository(self, repository_class: type[T], session: Any) -> T  # Line 189
    def register_repository(self, name: str, repository_class: type[T]) -> None  # Line 194
    def is_repository_registered(self, name: str) -> bool  # Line 199
```

#### Class: `MLServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for ML service operations

```python
class MLServiceInterface(ABC):
    async def get_model_performance_summary(self, model_name: str, days: int = 30) -> dict[str, Any]  # Line 208
    async def validate_model_deployment(self, model_name: str, version: int) -> bool  # Line 215
    async def get_model_recommendations(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]  # Line 220
```

#### Class: `ConnectionManagerInterface`

**Inherits**: ABC
**Purpose**: Interface for database connection management operations

```python
class ConnectionManagerInterface(ABC):
    async def initialize(self) -> None  # Line 229
    async def close(self) -> None  # Line 234
    async def get_async_session(self)  # Line 239
    def get_sync_session(self)  # Line 244
    async def get_redis_client(self)  # Line 249
    def get_influxdb_client(self)  # Line 254
    def is_healthy(self) -> bool  # Line 259
```

#### Class: `UnitOfWorkFactoryInterface`

**Inherits**: ABC
**Purpose**: Interface for Unit of Work factory operations

```python
class UnitOfWorkFactoryInterface(ABC):
    def create(self) -> Any  # Line 268
    def create_async(self) -> Any  # Line 273
    def configure_dependencies(self, dependency_injector: Any) -> None  # Line 278
```

#### Class: `CapitalServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for capital management operations

```python
class CapitalServiceInterface(ABC):
    async def get_available_capital(self, account_id: str) -> Decimal  # Line 287
    async def allocate_capital(self, account_id: str, amount: Decimal, purpose: str) -> bool  # Line 292
    async def release_capital(self, account_id: str, amount: Decimal, purpose: str) -> bool  # Line 297
```

#### Class: `UserServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for user management operations

```python
class UserServiceInterface(ABC):
    async def authenticate_user(self, email: str, password: str) -> dict[str, Any] | None  # Line 306
    async def get_user_permissions(self, user_id: str) -> list[str]  # Line 311
```

#### Class: `MarketDataServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for market data operations

```python
class MarketDataServiceInterface(ABC):
    async def get_latest_price(self, symbol: str) -> Decimal | None  # Line 320
    async def get_historical_data(self, ...) -> list[dict[str, Any]]  # Line 325
```

#### Class: `AuditServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for audit trail operations

```python
class AuditServiceInterface(ABC):
    async def log_action(self, action_data: dict[str, Any]) -> bool  # Line 336
    async def get_audit_trail(self, entity_type: str, entity_id: str, limit: int = 100) -> list[dict[str, Any]]  # Line 341
```

### File: manager.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`

#### Class: `DatabaseManager`

**Inherits**: BaseComponent
**Purpose**: Database operations coordinator that enforces service layer pattern

```python
class DatabaseManager(BaseComponent):
    def __init__(self, trading_service = None, market_data_service = None) -> None  # Line 28
    async def get_historical_data(self, ...) -> list[dict[str, Any]]  # Line 42
    async def save_trade(self, trade_data: dict[str, Any]) -> dict[str, Any]  # Line 70
    async def get_positions(self, strategy_id: str | None = None, symbol: str | None = None) -> list[dict[str, Any]]  # Line 91
    async def close(self)  # Line 115
```

### File: env.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.logging import get_logger`
- `from src.database.models import Base`

#### Functions:

```python
def get_url() -> str  # Line 42
def run_migrations_offline() -> None  # Line 64
def run_migrations_online() -> None  # Line 87
```

### File: 001_initial_schema.py

#### Functions:

```python
def upgrade() -> None  # Line 20
def downgrade() -> None  # Line 398
```

### File: 002_data_models.py

#### Functions:

```python
def upgrade() -> None  # Line 20
def downgrade() -> None  # Line 219
```

### File: 003_backtesting_models.py

#### Functions:

```python
def upgrade() -> None  # Line 20
def downgrade() -> None  # Line 461
```

### File: 004_optimization_models.py

#### Functions:

```python
def upgrade()  # Line 20
def downgrade()  # Line 298
```

### File: 249c01341fc2_add_default_status_to_position.py

#### Functions:

```python
def upgrade() -> None  # Line 20
def downgrade() -> None  # Line 28
```

### File: 9449369979fc_merge_heads.py

#### Functions:

```python
def upgrade() -> None  # Line 20
def downgrade() -> None  # Line 24
```

### File: 9dbcb25ea329_initial_migration.py

#### Functions:

```python
def upgrade() -> None  # Line 16
def downgrade() -> None  # Line 20
```

### File: analytics.py

**Key Imports:**
- `from src.database.models.base import Base`
- `from src.database.models.base import TimestampMixin`

#### Class: `AnalyticsPortfolioMetrics`

**Inherits**: Base, TimestampMixin
**Purpose**: Portfolio metrics storage

```python
class AnalyticsPortfolioMetrics(Base, TimestampMixin):
```

#### Class: `AnalyticsPositionMetrics`

**Inherits**: Base, TimestampMixin
**Purpose**: Position metrics storage

```python
class AnalyticsPositionMetrics(Base, TimestampMixin):
```

#### Class: `AnalyticsRiskMetrics`

**Inherits**: Base, TimestampMixin
**Purpose**: Risk metrics storage

```python
class AnalyticsRiskMetrics(Base, TimestampMixin):
```

#### Class: `AnalyticsStrategyMetrics`

**Inherits**: Base, TimestampMixin
**Purpose**: Strategy performance metrics storage

```python
class AnalyticsStrategyMetrics(Base, TimestampMixin):
```

#### Class: `AnalyticsOperationalMetrics`

**Inherits**: Base, TimestampMixin
**Purpose**: Operational metrics storage

```python
class AnalyticsOperationalMetrics(Base, TimestampMixin):
```

### File: audit.py

#### Class: `CapitalAuditLog`

**Inherits**: Base
**Purpose**: Audit log model for capital management operations

```python
class CapitalAuditLog(Base):
```

#### Class: `ExecutionAuditLog`

**Inherits**: Base
**Purpose**: Audit log model for execution operations

```python
class ExecutionAuditLog(Base):
```

#### Class: `RiskAuditLog`

**Inherits**: Base
**Purpose**: Audit log model for risk management decisions and violations

```python
class RiskAuditLog(Base):
```

#### Class: `PerformanceAuditLog`

**Inherits**: Base
**Purpose**: Audit log model for performance tracking and analysis

```python
class PerformanceAuditLog(Base):
```

### File: backtesting.py

**Key Imports:**
- `from src.database.models.base import Base`
- `from src.database.models.base import TimestampMixin`

#### Class: `BacktestRun`

**Inherits**: Base, TimestampMixin
**Purpose**: Backtest run configuration and metadata storage

```python
class BacktestRun(Base, TimestampMixin):
```

#### Class: `BacktestResult`

**Inherits**: Base, TimestampMixin
**Purpose**: Comprehensive backtest results and performance metrics

```python
class BacktestResult(Base, TimestampMixin):
```

#### Class: `BacktestTrade`

**Inherits**: Base, TimestampMixin
**Purpose**: Individual trade records from backtest simulations

```python
class BacktestTrade(Base, TimestampMixin):
```

### File: base.py

#### Class: `TimestampMixin`

**Purpose**: Mixin for automatic timestamp management

```python
class TimestampMixin:
    def created_at(self)  # Line 22
    def updated_at(self)  # Line 26
```

#### Class: `AuditMixin`

**Inherits**: TimestampMixin
**Purpose**: Mixin for audit fields

```python
class AuditMixin(TimestampMixin):
    def created_by(self)  # Line 36
    def updated_by(self)  # Line 40
    def version(self)  # Line 44
```

#### Class: `MetadataMixin`

**Purpose**: Mixin for metadata storage

```python
class MetadataMixin:
    def metadata_json(self)  # Line 52
    def get_metadata(self, key: str, default: Any = None) -> Any  # Line 55
    def set_metadata(self, key: str, value: Any) -> None  # Line 61
    def update_metadata(self, data: dict[str, Any]) -> None  # Line 67
```

#### Class: `SoftDeleteMixin`

**Purpose**: Mixin for soft delete functionality

```python
class SoftDeleteMixin:
    def deleted_at(self)  # Line 78
    def deleted_by(self)  # Line 82
    def is_deleted(self) -> bool  # Line 86
    def soft_delete(self, deleted_by: str | None = None) -> None  # Line 90
    def restore(self) -> None  # Line 96
```

### File: bot.py

**Key Imports:**
- `from src.database.models.base import AuditMixin`
- `from src.database.models.base import Base`
- `from src.database.models.base import MetadataMixin`
- `from src.database.models.base import SoftDeleteMixin`
- `from src.database.models.base import TimestampMixin`

#### Class: `Bot`

**Inherits**: Base, AuditMixin, MetadataMixin, SoftDeleteMixin
**Purpose**: Bot model

```python
class Bot(Base, AuditMixin, MetadataMixin, SoftDeleteMixin):
    def __repr__(self)  # Line 120
    def is_running(self) -> bool  # Line 124
    def win_rate(self) -> Decimal  # Line 128
    def average_pnl(self) -> Decimal  # Line 136
```

#### Class: `Strategy`

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: Strategy model

```python
class Strategy(Base, AuditMixin, MetadataMixin):
    def __repr__(self)  # Line 236
    def is_active(self) -> bool  # Line 239
    def signal_success_rate(self) -> Decimal  # Line 243
```

#### Class: `Signal`

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: Trading signal model

```python
class Signal(Base, TimestampMixin, MetadataMixin):
    def __repr__(self)  # Line 317
    def is_executed(self) -> bool  # Line 320
    def is_successful(self) -> bool  # Line 324
```

#### Class: `BotLog`

**Inherits**: Base, TimestampMixin
**Purpose**: Bot activity log model for error handling and audit trail

```python
class BotLog(Base, TimestampMixin):
    def __repr__(self)  # Line 395
```

### File: bot_instance.py

#### Class: `BotInstance`

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: Bot instance model for managing individual trading bots

```python
class BotInstance(Base, TimestampMixin, MetadataMixin):
    def __repr__(self)  # Line 126
    def is_running(self) -> bool  # Line 129
    def is_stopped(self) -> bool  # Line 133
```

### File: capital.py

#### Class: `CapitalAllocationDB`

**Inherits**: Base, TimestampMixin
**Purpose**: Capital allocation tracking model

```python
class CapitalAllocationDB(Base, TimestampMixin):
    def __repr__(self)  # Line 117
```

#### Class: `FundFlowDB`

**Inherits**: Base, TimestampMixin
**Purpose**: Fund flow tracking model

```python
class FundFlowDB(Base, TimestampMixin):
    def __repr__(self)  # Line 197
```

#### Class: `CurrencyExposureDB`

**Inherits**: Base, TimestampMixin
**Purpose**: Currency exposure tracking model

```python
class CurrencyExposureDB(Base, TimestampMixin):
    def __repr__(self)  # Line 250
```

#### Class: `ExchangeAllocationDB`

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange allocation tracking model

```python
class ExchangeAllocationDB(Base, TimestampMixin):
    def __repr__(self)  # Line 306
```

### File: data.py

#### Class: `FeatureRecord`

**Inherits**: Base, TimestampMixin
**Purpose**: Feature record model for ML feature storage

```python
class FeatureRecord(Base, TimestampMixin):
    def __init__(self, **kwargs)  # Line 105
    def __repr__(self)  # Line 116
```

#### Class: `DataQualityRecord`

**Inherits**: Base, TimestampMixin
**Purpose**: Data quality tracking model

```python
class DataQualityRecord(Base, TimestampMixin):
    def __init__(self, **kwargs)  # Line 233
    def __repr__(self)  # Line 244
```

#### Class: `DataPipelineRecord`

**Inherits**: Base, TimestampMixin
**Purpose**: Data pipeline execution tracking model

```python
class DataPipelineRecord(Base, TimestampMixin):
    def __init__(self, **kwargs)  # Line 330
    def __repr__(self)  # Line 341
```

### File: exchange.py

#### Class: `ExchangeConfiguration`

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange configuration model

```python
class ExchangeConfiguration(Base, TimestampMixin):
    def __repr__(self)  # Line 102
```

#### Class: `ExchangeTradingPair`

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange-specific trading pair information

```python
class ExchangeTradingPair(Base, TimestampMixin):
    def __repr__(self)  # Line 190
    def round_price(self, price: Decimal) -> Decimal  # Line 193
    def round_quantity(self, quantity: Decimal) -> Decimal  # Line 197
    def validate_order(self, price: Decimal, quantity: Decimal) -> bool  # Line 201
```

#### Class: `ExchangeConnectionStatus`

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange connection status tracking

```python
class ExchangeConnectionStatus(Base, TimestampMixin):
    def __repr__(self)  # Line 282
    def calculate_success_rate(self) -> Decimal  # Line 285
```

#### Class: `ExchangeRateLimit`

**Inherits**: Base, TimestampMixin
**Purpose**: Exchange rate limit tracking and enforcement

```python
class ExchangeRateLimit(Base, TimestampMixin):
    def __repr__(self)  # Line 353
    def is_exceeded(self) -> bool  # Line 357
    def remaining_requests(self) -> int  # Line 362
    def usage_percentage(self) -> Decimal  # Line 367
```

### File: market_data.py

**Key Imports:**
- `from src.database.models.base import Base`
- `from src.database.models.base import TimestampMixin`

#### Class: `MarketDataRecord`

**Inherits**: Base, TimestampMixin
**Purpose**: Market data record model

```python
class MarketDataRecord(Base, TimestampMixin):
    def __repr__(self)  # Line 103
    def price_change(self) -> Decimal  # Line 108
    def price_change_percent(self) -> Decimal  # Line 115
```

### File: ml.py

#### Class: `MLPrediction`

**Inherits**: Base
**Purpose**: Model for storing ML prediction results

```python
class MLPrediction(Base):
    def __repr__(self) -> str  # Line 120
```

#### Class: `MLModelMetadata`

**Inherits**: Base
**Purpose**: Model for storing ML model metadata and versioning information

```python
class MLModelMetadata(Base):
    def __repr__(self) -> str  # Line 239
```

#### Class: `MLTrainingJob`

**Inherits**: Base
**Purpose**: Model for tracking ML model training jobs

```python
class MLTrainingJob(Base):
    def __repr__(self) -> str  # Line 370
```

### File: optimization.py

#### Class: `OptimizationRun`

**Inherits**: Base
**Purpose**: Model for optimization run metadata

```python
class OptimizationRun(Base):
```

#### Class: `OptimizationResult`

**Inherits**: Base
**Purpose**: Model for storing final optimization results

```python
class OptimizationResult(Base):
```

#### Class: `ParameterSet`

**Inherits**: Base
**Purpose**: Model for storing individual parameter sets evaluated during optimization

```python
class ParameterSet(Base):
```

#### Class: `OptimizationObjectiveDB`

**Inherits**: Base
**Purpose**: Model for storing individual optimization objectives

```python
class OptimizationObjectiveDB(Base):
```

### File: risk.py

**Key Imports:**
- `from src.core.types import AlertSeverity`
- `from src.core.types import CircuitBreakerStatus`
- `from src.core.types import CircuitBreakerType`
- `from src.core.types import PositionSizeMethod`

#### Class: `RiskConfiguration`

**Inherits**: Base, TimestampMixin
**Purpose**: Risk configuration storage for strategies and bots

```python
class RiskConfiguration(Base, TimestampMixin):
```

#### Class: `CircuitBreakerConfig`

**Inherits**: Base, TimestampMixin
**Purpose**: Circuit breaker configuration storage

```python
class CircuitBreakerConfig(Base, TimestampMixin):
```

#### Class: `CircuitBreakerEvent`

**Inherits**: Base, TimestampMixin
**Purpose**: Circuit breaker trigger event storage

```python
class CircuitBreakerEvent(Base, TimestampMixin):
```

#### Class: `RiskViolation`

**Inherits**: Base, TimestampMixin
**Purpose**: Risk violation event storage

```python
class RiskViolation(Base, TimestampMixin):
```

### File: state.py

#### Class: `StateSnapshot`

**Inherits**: Base, AuditMixin, MetadataMixin, SoftDeleteMixin
**Purpose**: State snapshot table for point-in-time state captures

```python
class StateSnapshot(Base, AuditMixin, MetadataMixin, SoftDeleteMixin):
    def compression_ratio(self) -> Decimal  # Line 136
    def get_state_by_type(self, state_type: str) -> dict[str, Any] | None  # Line 142
    def get_state_count(self) -> int  # Line 148
```

#### Class: `StateCheckpoint`

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: State checkpoint table for incremental state saves

```python
class StateCheckpoint(Base, AuditMixin, MetadataMixin):
    def get_changed_state_ids(self) -> set[str]  # Line 271
    def has_state_type(self, state_type: str) -> bool  # Line 277
```

#### Class: `StateHistory`

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: State history table for detailed audit trail

```python
class StateHistory(Base, TimestampMixin, MetadataMixin):
    def size_change_bytes(self) -> int  # Line 407
    def get_changed_field_names(self) -> set[str]  # Line 413
```

#### Class: `StateMetadata`

**Inherits**: Base, AuditMixin
**Purpose**: State metadata table for state information and indexing

```python
class StateMetadata(Base, AuditMixin):
    def get_tag(self, key: str, default: Any = None) -> Any  # Line 519
    def set_tag(self, key: str, value: Any) -> None  # Line 525
    def increment_access_count(self) -> None  # Line 533
    def is_in_storage_layer(self, layer: str) -> bool  # Line 538
```

#### Class: `StateBackup`

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: State backup table for backup operations tracking

```python
class StateBackup(Base, AuditMixin, MetadataMixin):
    def is_expired(self) -> bool  # Line 676
    def backup_success_rate(self) -> Decimal  # Line 682
    def get_included_state_types(self) -> set[str]  # Line 691
    def mark_verified(self, checksum: str) -> bool  # Line 701
```

#### Functions:

```python
def create_state_indexes()  # Line 710
```

### File: system.py

#### Class: `Alert`

**Inherits**: Base, TimestampMixin
**Purpose**: Alert model for system notifications

```python
class Alert(Base, TimestampMixin):
    def __repr__(self)  # Line 87
```

#### Class: `AlertRule`

**Inherits**: Base, TimestampMixin
**Purpose**: Alert rule configuration model

```python
class AlertRule(Base, TimestampMixin):
    def __repr__(self)  # Line 149
```

#### Class: `EscalationPolicy`

**Inherits**: Base, TimestampMixin
**Purpose**: Escalation policy for alert management

```python
class EscalationPolicy(Base, TimestampMixin):
    def __repr__(self)  # Line 194
```

#### Class: `AuditLog`

**Inherits**: Base, TimestampMixin
**Purpose**: General audit log model

```python
class AuditLog(Base, TimestampMixin):
    def __repr__(self)  # Line 236
```

#### Class: `PerformanceMetrics`

**Inherits**: Base, TimestampMixin
**Purpose**: Performance metrics model

```python
class PerformanceMetrics(Base, TimestampMixin):
    def __repr__(self)  # Line 297
```

#### Class: `BalanceSnapshot`

**Inherits**: Base, TimestampMixin
**Purpose**: Balance snapshot model for tracking account balances over time

```python
class BalanceSnapshot(Base, TimestampMixin):
    def __repr__(self)  # Line 365
```

#### Class: `ResourceAllocation`

**Inherits**: Base, TimestampMixin
**Purpose**: Resource allocation model for tracking bot resource allocations

```python
class ResourceAllocation(Base, TimestampMixin):
    def __repr__(self)  # Line 398
```

#### Class: `ResourceUsage`

**Inherits**: Base, TimestampMixin
**Purpose**: Resource usage model for tracking bot resource usage

```python
class ResourceUsage(Base, TimestampMixin):
    def __repr__(self)  # Line 434
```

### File: trading.py

**Key Imports:**
- `from src.database.models.base import AuditMixin`
- `from src.database.models.base import Base`
- `from src.database.models.base import MetadataMixin`
- `from src.database.models.base import TimestampMixin`

#### Class: `Order`

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: Order model

```python
class Order(Base, AuditMixin, MetadataMixin):
    def __repr__(self)  # Line 104
    def is_filled(self) -> bool  # Line 108
    def is_active(self) -> bool  # Line 115
    def remaining_quantity(self) -> Decimal  # Line 120
```

#### Class: `Position`

**Inherits**: Base, AuditMixin, MetadataMixin
**Purpose**: Position model

```python
class Position(Base, AuditMixin, MetadataMixin):
    def __repr__(self)  # Line 209
    def is_open(self) -> bool  # Line 213
    def value(self) -> Decimal  # Line 218
    def calculate_pnl(self, current_price: Decimal | None = None) -> Decimal  # Line 224
```

#### Class: `OrderFill`

**Inherits**: Base, TimestampMixin
**Purpose**: Order fill/execution model

```python
class OrderFill(Base, TimestampMixin):
    def __repr__(self)  # Line 282
    def value(self) -> Decimal  # Line 286
    def net_value(self) -> Decimal  # Line 293
```

#### Class: `Trade`

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: Completed trade model

```python
class Trade(Base, TimestampMixin, MetadataMixin):
    def __repr__(self)  # Line 370
    def is_profitable(self) -> bool  # Line 374
    def return_percentage(self) -> Decimal  # Line 381
```

### File: user.py

#### Class: `User`

**Inherits**: Base, TimestampMixin, MetadataMixin
**Purpose**: User model for authentication and account management

```python
class User(Base, TimestampMixin, MetadataMixin):
    def __repr__(self)  # Line 91
    def full_name(self) -> str  # Line 95
    def is_authenticated(self) -> bool  # Line 102
```

### File: queries.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import DatabaseError`
- `from src.core.exceptions import DataError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import PerformanceMonitor`

#### Class: `DatabaseQueries`

**Inherits**: BaseComponent
**Purpose**: Database query utilities with common CRUD operations

```python
class DatabaseQueries(BaseComponent):
    def __init__(self, session: AsyncSession, config: dict[str, Any] | None = None)  # Line 94
    async def _acquire_session(self)  # Line 125
    async def create(self, model_instance: T) -> T  # Line 138
    async def get_by_id(self, model_class: type[T], record_id: str) -> T | None  # Line 230
    async def get_all(self, model_class: type[T], limit: int | None = None, offset: int = 0) -> list[T]  # Line 283
    async def update(self, model_instance: T) -> T  # Line 298
    async def delete(self, model_instance: T) -> bool  # Line 336
    async def bulk_create(self, model_instances: list[T]) -> list[T]  # Line 373
    async def bulk_update(self, model_class: type[T], updates: list[dict[str, Any]], id_field: str = 'id') -> int  # Line 412
    async def get_user_by_username(self, username: str) -> User | None  # Line 461
    async def get_user_by_email(self, email: str) -> User | None  # Line 471
    async def get_active_users(self) -> list[User]  # Line 481
    async def get_bot_instances_by_user(self, user_id: str) -> list[BotInstance]  # Line 491
    async def get_bot_instance_by_name(self, user_id: str, name: str) -> BotInstance | None  # Line 505
    async def get_running_bots(self) -> list[BotInstance]  # Line 519
    async def get_trades_by_bot(self, bot_id: str, limit: int | None = None, offset: int = 0) -> list[Trade]  # Line 535
    async def get_trades_by_symbol(self, ...) -> list[Trade]  # Line 550
    async def get_trades_by_date_range(self, start_time: datetime, end_time: datetime) -> list[Trade]  # Line 570
    async def get_positions_by_bot(self, bot_id: str) -> list[Position]  # Line 586
    async def get_open_positions(self) -> list[Position]  # Line 600
    async def get_latest_balance_snapshot(self, user_id: str, exchange: str, currency: str) -> BalanceSnapshot | None  # Line 616
    async def get_performance_metrics_by_bot(self, ...) -> list[PerformanceMetrics]  # Line 639
    async def get_unread_alerts_by_user(self, user_id: str) -> list[Alert]  # Line 659
    async def get_alerts_by_severity(self, severity: str, limit: int | None = None) -> list[Alert]  # Line 673
    async def get_audit_logs_by_user(self, user_id: str, limit: int | None = None) -> list[AuditLog]  # Line 688
    async def get_total_pnl_by_bot(self, ...) -> Decimal  # Line 709
    async def get_trade_count_by_bot(self, ...) -> int  # Line 730
    async def get_win_rate_by_bot(self, ...) -> tuple[int, int]  # Line 749
    async def export_trades_to_csv_data(self, ...) -> list[dict[str, Any]]  # Line 786
    async def health_check(self) -> bool  # Line 829
    async def get_capital_allocations_by_strategy(self, strategy_id: str, limit: int | None = None, offset: int = 0) -> list[CapitalAllocationDB]  # Line 841
    async def get_capital_allocations_by_exchange(self, exchange: str, limit: int | None = None, offset: int = 0) -> list[CapitalAllocationDB]  # Line 862
    async def get_fund_flows_by_reason(self, ...) -> list[FundFlowDB]  # Line 883
    async def get_fund_flows_by_currency(self, ...) -> list[FundFlowDB]  # Line 911
    async def get_currency_exposure_by_currency(self, currency: str) -> CurrencyExposureDB | None  # Line 939
    async def get_exchange_allocation_by_exchange(self, exchange: str) -> ExchangeAllocationDB | None  # Line 950
    async def get_total_capital_allocated(self, start_time: datetime | None = None, end_time: datetime | None = None) -> Decimal  # Line 965
    async def get_total_fund_flows(self, start_time: datetime | None = None, end_time: datetime | None = None) -> Decimal  # Line 985
    async def bulk_create_capital_allocations(self, allocations: list[CapitalAllocationDB]) -> list[CapitalAllocationDB]  # Line 1005
    async def bulk_update_capital_allocations(self, updates: list[dict[str, Any]]) -> int  # Line 1011
    async def bulk_create_fund_flows(self, flows: list[FundFlowDB]) -> list[FundFlowDB]  # Line 1015
    async def create_market_data_record(self, market_data: MarketDataRecord) -> MarketDataRecord  # Line 1019
    async def bulk_create_market_data_records(self, records: list[MarketDataRecord]) -> list[MarketDataRecord]  # Line 1023
    async def get_market_data_records(self, ...) -> list[MarketDataRecord]  # Line 1029
    async def get_market_data_by_quality(self, ...) -> list[MarketDataRecord]  # Line 1060
    async def delete_old_market_data(self, cutoff_date: datetime) -> int  # Line 1086
    async def create_feature_record(self, feature: FeatureRecord) -> FeatureRecord  # Line 1108
    async def bulk_create_feature_records(self, features: list[FeatureRecord]) -> list[FeatureRecord]  # Line 1112
    async def get_feature_records(self, ...) -> list[FeatureRecord]  # Line 1118
    async def create_data_quality_record(self, quality_record: DataQualityRecord) -> DataQualityRecord  # Line 1148
    async def get_data_quality_records(self, ...) -> list[DataQualityRecord]  # Line 1154
    async def create_data_pipeline_record(self, pipeline_record: DataPipelineRecord) -> DataPipelineRecord  # Line 1183
    async def update_data_pipeline_status(self, ...) -> bool  # Line 1189
    async def get_data_pipeline_records(self, ...) -> list[DataPipelineRecord]  # Line 1220
```

#### Functions:

```python
def get_query_config(config: dict[str, Any] | None = None) -> dict[str, int]  # Line 53
```

### File: redis_client.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.base.interfaces import CacheClientInterface`
- `from src.core.exceptions import DataError`
- `from src.error_handling.decorators import with_circuit_breaker`
- `from src.error_handling.decorators import with_retry`

#### Class: `RedisClient`

**Inherits**: BaseComponent, CacheClientInterface
**Purpose**: Async Redis client with utilities for trading bot data

```python
class RedisClient(BaseComponent, CacheClientInterface):
    def __init__(self, config_or_url: Any | str) -> None  # Line 66
    def client(self) -> redis.Redis | None  # Line 105
    async def connect(self) -> None  # Line 112
    def _start_heartbeat_monitoring(self) -> None  # Line 163
    async def _heartbeat_monitor(self) -> None  # Line 168
    async def _ensure_connected(self) -> None  # Line 183
    async def _cleanup_connection(self) -> None  # Line 206
    async def _with_backpressure(self, operation)  # Line 216
    async def _maybe_autoclose(self) -> None  # Line 233
    async def disconnect(self) -> None  # Line 250
    def _get_namespaced_key(self, key: str, namespace: str = 'trading_bot') -> str  # Line 280
    async def set(self, ...) -> bool  # Line 285
    async def get(self, key: str, namespace: str = 'trading_bot') -> Any | None  # Line 319
    async def delete(self, key: str, namespace: str = 'trading_bot') -> bool  # Line 345
    async def exists(self, key: str, namespace: str = 'trading_bot') -> bool  # Line 359
    async def expire(self, key: str, ttl: int, namespace: str = 'trading_bot') -> bool  # Line 373
    async def ttl(self, key: str, namespace: str = 'trading_bot') -> int  # Line 387
    async def hset(self, key: str, field: str, value: Any, namespace: str = 'trading_bot') -> bool  # Line 402
    async def hget(self, key: str, field: str, namespace: str = 'trading_bot') -> Any | None  # Line 423
    async def hgetall(self, key: str, namespace: str = 'trading_bot') -> dict[str, Any]  # Line 445
    async def hdel(self, key: str, field: str, namespace: str = 'trading_bot') -> bool  # Line 468
    async def lpush(self, key: str, value: Any, namespace: str = 'trading_bot') -> int  # Line 483
    async def rpush(self, key: str, value: Any, namespace: str = 'trading_bot') -> int  # Line 504
    async def lrange(self, key: str, start: int = 0, end: int = Any, namespace: str = 'trading_bot') -> list[Any]  # Line 525
    async def sadd(self, key: str, value: Any, namespace: str = 'trading_bot') -> bool  # Line 551
    async def smembers(self, key: str, namespace: str = 'trading_bot') -> list[Any]  # Line 572
    async def store_market_data(self, symbol: str, data: dict[str, Any], ttl: int = 300) -> bool  # Line 596
    async def get_market_data(self, symbol: str) -> dict[str, Any] | None  # Line 601
    async def store_position(self, bot_id: str, position: dict[str, Any]) -> bool  # Line 606
    async def get_position(self, bot_id: str) -> dict[str, Any] | None  # Line 610
    async def store_balance(self, user_id: str, exchange: str, balance: dict[str, Any]) -> bool  # Line 614
    async def get_balance(self, user_id: str, exchange: str) -> dict[str, Any] | None  # Line 619
    async def store_cache(self, key: str, value: Any, ttl: int = 3600) -> bool  # Line 624
    async def get_cache(self, key: str) -> Any | None  # Line 628
    async def ping(self) -> bool  # Line 633
    async def info(self) -> dict[str, Any]  # Line 645
    async def health_check(self) -> bool  # Line 657
```

#### Functions:

```python
def substitute_env_vars(url: str) -> str  # Line 33
```

### File: audit.py

**Key Imports:**
- `from src.database.models.audit import CapitalAuditLog`
- `from src.database.models.audit import ExecutionAuditLog`
- `from src.database.models.audit import PerformanceAuditLog`
- `from src.database.models.audit import RiskAuditLog`
- `from src.database.repository.base import DatabaseRepository`

#### Class: `CapitalAuditLogRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for CapitalAuditLog entities

```python
class CapitalAuditLogRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 19
    async def get_by_strategy(self, strategy_id: str) -> list[CapitalAuditLog]  # Line 29
    async def get_by_exchange(self, exchange: str) -> list[CapitalAuditLog]  # Line 33
    async def get_by_date_range(self, start_date: datetime, end_date: datetime) -> list[CapitalAuditLog]  # Line 37
```

#### Class: `ExecutionAuditLogRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for ExecutionAuditLog entities

```python
class ExecutionAuditLogRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 49
    async def get_by_order(self, order_id: str) -> list[ExecutionAuditLog]  # Line 59
    async def get_by_execution_status(self, status: str) -> list[ExecutionAuditLog]  # Line 63
    async def get_failed_executions(self) -> list[ExecutionAuditLog]  # Line 67
```

#### Class: `RiskAuditLogRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for RiskAuditLog entities

```python
class RiskAuditLogRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 75
    async def get_by_risk_type(self, risk_type: str) -> list[RiskAuditLog]  # Line 85
    async def get_high_severity_risks(self) -> list[RiskAuditLog]  # Line 89
    async def get_critical_risks(self) -> list[RiskAuditLog]  # Line 93
```

#### Class: `PerformanceAuditLogRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for PerformanceAuditLog entities

```python
class PerformanceAuditLogRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 101
    async def get_by_strategy(self, strategy_id: str) -> list[PerformanceAuditLog]  # Line 111
    async def get_by_metric_type(self, metric_type: str) -> list[PerformanceAuditLog]  # Line 115
```

### File: base.py

**Key Imports:**
- `from src.core.base.repository import BaseRepository`
- `from src.core.exceptions import RepositoryError`
- `from src.core.logging import get_logger`
- `from src.core.types.base import ConfigDict`

#### Class: `DatabaseRepository`

**Inherits**: CoreBaseRepository
**Purpose**: Database repository implementation using core BaseRepository

```python
class DatabaseRepository(CoreBaseRepository):
    def __init__(self, ...) -> None  # Line 20
    async def _create_entity(self, entity) -> Any  # Line 52
    async def _get_entity_by_id(self, entity_id: Any) -> Any | None  # Line 65
    async def _update_entity(self, entity: Any) -> Any | None  # Line 74
    async def _delete_entity(self, entity_id) -> bool  # Line 91
    async def _list_entities(self, ...) -> list  # Line 104
    async def _count_entities(self, filters: dict[str, Any] | None) -> int  # Line 139
    async def get_by(self, **kwargs)  # Line 160
    async def get(self, entity_id: Any)  # Line 172
    async def get_all(self, ...) -> list  # Line 176
    async def exists(self, entity_id: Any) -> bool  # Line 198
    async def soft_delete(self, entity_id: Any, deleted_by: str | None = None) -> bool  # Line 207
    async def begin(self)  # Line 220
    async def commit(self)  # Line 224
    async def rollback(self)  # Line 228
    async def refresh(self, entity)  # Line 232
    def _apply_update_transforms(self, entity) -> None  # Line 236
    def _apply_consistent_filters(self, stmt, filters: dict[str, Any])  # Line 248
    def _apply_single_filter(self, stmt, column, value)  # Line 256
    def _apply_complex_filter(self, stmt, column, value_dict: dict[str, Any])  # Line 265
    def _has_valid_between_filter(self, value_dict: dict[str, Any]) -> bool  # Line 281
    def _validate_entity_at_boundary(self, entity: Any, operation: str) -> None  # Line 289
```

#### Class: `RepositoryInterface`

**Purpose**: Interface for repository pattern

```python
class RepositoryInterface:
```

### File: bot.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.database.models.bot import Bot`
- `from src.database.models.bot import BotLog`
- `from src.database.models.bot import Signal`
- `from src.database.models.bot import Strategy`

#### Class: `BotRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for Bot entities

```python
class BotRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 20
    async def get_active_bots(self) -> list[Bot]  # Line 25
    async def get_running_bots(self) -> list[Bot]  # Line 29
    async def get_bot_by_name(self, name: str) -> Bot | None  # Line 33
    async def start_bot(self, bot_id: str) -> bool  # Line 37
    async def stop_bot(self, bot_id: str) -> bool  # Line 46
    async def pause_bot(self, bot_id: str) -> bool  # Line 55
    async def update_bot_status(self, bot_id: str, status: str) -> bool  # Line 64
    async def update_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> bool  # Line 68
    async def get_bot_performance(self, bot_id: str) -> dict[str, Any]  # Line 72
```

#### Class: `StrategyRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for Strategy entities

```python
class StrategyRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 98
    async def get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]  # Line 107
    async def get_strategies_by_bot(self, bot_id: str) -> list[Strategy]  # Line 115
    async def get_strategy_by_name(self, bot_id: str, name: str) -> Strategy | None  # Line 119
    async def activate_strategy(self, strategy_id: str) -> bool  # Line 123
    async def deactivate_strategy(self, strategy_id: str) -> bool  # Line 132
    async def update_strategy_params(self, strategy_id: str, params: dict[str, Any]) -> bool  # Line 141
    async def update_strategy_metrics(self, strategy_id: str, metrics: dict[str, Any]) -> bool  # Line 150
```

#### Class: `SignalRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for Signal entities

```python
class SignalRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 158
    async def get_unexecuted_signals(self, strategy_id: str | None = None) -> list[Signal]  # Line 163
    async def get_signals_by_strategy(self, strategy_id: str, limit: int = 100) -> list[Signal]  # Line 171
    async def get_recent_signals(self, hours: int = 24, strategy_id: str | None = None) -> list[Signal]  # Line 177
    async def mark_signal_executed(self, signal_id: str, order_id: str, execution_time: Decimal) -> bool  # Line 184
    async def update_signal_outcome(self, signal_id: str, outcome: str, pnl: Decimal | None = None) -> bool  # Line 197
    async def get_signal_statistics(self, strategy_id: str, since: datetime | None = None) -> dict[str, Any]  # Line 213
```

#### Class: `BotLogRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for BotLog entities

```python
class BotLogRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 256
    async def get_logs_by_bot(self, bot_id: str, level: str | None = None, limit: int = 100) -> list[BotLog]  # Line 261
    async def get_error_logs(self, bot_id: str | None = None, hours: int = 24) -> list[BotLog]  # Line 271
    async def log_event(self, ...) -> BotLog  # Line 284
    async def cleanup_old_logs(self, days: int = 30) -> int  # Line 299
```

### File: bot_instance.py

**Key Imports:**
- `from src.database.models.bot_instance import BotInstance`
- `from src.database.repository.base import DatabaseRepository`

#### Class: `BotInstanceRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot instance operations

```python
class BotInstanceRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 22
    async def get_by_bot_id(self, bot_id: str) -> BotInstance | None  # Line 37
    async def get_active_bots(self) -> list[BotInstance]  # Line 50
    async def get_bots_by_status(self, status: str) -> list[BotInstance]  # Line 69
    async def get_bots_by_strategy(self, strategy_name: str) -> list[BotInstance]  # Line 86
    async def get_bots_by_exchange(self, exchange: str) -> list[BotInstance]  # Line 103
    async def update_bot_status(self, ...) -> BotInstance | None  # Line 120
    async def update_heartbeat(self, bot_id: str) -> bool  # Line 153
    async def get_stale_bots(self, heartbeat_timeout_seconds: int = 300) -> list[BotInstance]  # Line 170
    async def get_error_bots(self, min_error_count: int = 1) -> list[BotInstance]  # Line 199
    async def reset_bot_errors(self, bot_id: str) -> bool  # Line 224
    async def get_bot_statistics(self) -> dict[str, Any]  # Line 243
    async def cleanup_old_bots(self, days: int = 30, status: str = 'stopped') -> int  # Line 300
    async def get_bots_by_symbol(self, symbol: str) -> list[BotInstance]  # Line 319
    async def get_profitable_bots(self, min_pnl: Decimal = Any) -> list[BotInstance]  # Line 334
    async def update_bot_metrics(self, ...) -> BotInstance | None  # Line 356
```

### File: bot_metrics.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.database.models.system import PerformanceMetrics`
- `from src.database.repository.base import DatabaseRepository`

#### Class: `BotMetricsRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot metrics and performance data operations

```python
class BotMetricsRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 25
    async def store_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> None  # Line 40
    async def get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]  # Line 68
    async def get_latest_metrics(self, bot_id: str) -> dict[str, Any] | None  # Line 108
    async def get_bot_health_checks(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]  # Line 121
    async def store_bot_health_analysis(self, bot_id: str, analysis: dict[str, Any]) -> None  # Line 162
    async def get_bot_health_analyses(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]  # Line 189
    async def get_recent_health_analyses(self, hours: int = 1) -> list[dict[str, Any]]  # Line 227
    async def archive_bot_record(self, bot_id: str) -> None  # Line 263
    async def get_active_bots(self) -> list[str]  # Line 296
    async def get_bot_performance_summary(self, bot_id: str, hours: int = 24) -> dict[str, Any]  # Line 325
```

### File: bot_resources.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.database.models.system import ResourceAllocation`
- `from src.database.models.system import ResourceUsage`
- `from src.database.repository.base import DatabaseRepository`

#### Class: `BotResourcesRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot resource management operations

```python
class BotResourcesRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 25
    async def store_resource_allocation(self, allocation: dict[str, Any]) -> None  # Line 40
    async def update_resource_allocation_status(self, bot_id: str, status: str) -> None  # Line 69
    async def store_resource_usage(self, usage: dict[str, Any]) -> None  # Line 98
    async def store_resource_reservation(self, reservation: dict[str, Any]) -> None  # Line 126
    async def update_resource_reservation_status(self, reservation_id: str, status: str) -> None  # Line 160
    async def store_resource_usage_history(self, usage_entry: dict[str, Any]) -> None  # Line 193
    async def store_optimization_suggestion(self, suggestion: dict[str, Any]) -> None  # Line 202
    async def get_resource_allocations(self, bot_id: str) -> list[dict[str, Any]]  # Line 237
    async def get_resource_usage_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]  # Line 276
    async def cleanup_expired_reservations(self) -> int  # Line 322
```

### File: capital.py

**Key Imports:**
- `from src.core.exceptions import DatabaseError`
- `from src.database.models.audit import CapitalAuditLog`
- `from src.database.models.capital import CapitalAllocationDB`
- `from src.database.models.capital import CurrencyExposureDB`
- `from src.database.models.capital import ExchangeAllocationDB`

#### Class: `CapitalAllocationRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for CapitalAllocationDB entities

```python
class CapitalAllocationRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 23
    async def get_by_strategy(self, strategy_id: str) -> list[CapitalAllocationDB]  # Line 33
    async def get_by_exchange(self, exchange: str) -> list[CapitalAllocationDB]  # Line 37
    async def find_by_strategy_exchange(self, strategy_id: str, exchange: str) -> CapitalAllocationDB | None  # Line 41
    async def get_total_allocated_by_strategy(self, strategy_id: str) -> Decimal  # Line 52
    async def get_available_capital_by_exchange(self, exchange: str) -> Decimal  # Line 57
```

#### Class: `FundFlowRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for FundFlowDB entities

```python
class FundFlowRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 66
    async def get_by_from_strategy(self, strategy_id: str) -> list[FundFlowDB]  # Line 76
    async def get_by_to_strategy(self, strategy_id: str) -> list[FundFlowDB]  # Line 82
    async def get_by_exchange_flow(self, from_exchange: str, to_exchange: str) -> list[FundFlowDB]  # Line 88
    async def get_by_reason(self, reason: str) -> list[FundFlowDB]  # Line 93
    async def get_by_currency(self, currency: str) -> list[FundFlowDB]  # Line 97
```

#### Class: `CurrencyExposureRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for CurrencyExposureDB entities

```python
class CurrencyExposureRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 105
    async def get_by_currency(self, currency: str) -> CurrencyExposureDB | None  # Line 115
    async def get_hedging_required(self) -> list[CurrencyExposureDB]  # Line 119
    async def get_total_exposure(self) -> Decimal  # Line 123
```

#### Class: `ExchangeAllocationRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for ExchangeAllocationDB entities

```python
class ExchangeAllocationRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 132
    async def get_by_exchange(self, exchange: str) -> ExchangeAllocationDB | None  # Line 142
    async def get_total_allocated(self) -> Decimal  # Line 146
    async def get_total_available(self) -> Decimal  # Line 151
    async def get_underutilized_exchanges(self, threshold: Decimal = Any) -> list[ExchangeAllocationDB]  # Line 157
```

#### Class: `CapitalAuditLogRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for capital audit log entities

```python
class CapitalAuditLogRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 173
    async def get_by_operation_id(self, operation_id: str) -> CapitalAuditLog | None  # Line 183
    async def get_by_strategy(self, strategy_id: str) -> list[CapitalAuditLog]  # Line 187
    async def get_by_exchange(self, exchange: str) -> list[CapitalAuditLog]  # Line 191
    async def get_failed_operations(self, limit: int = 100) -> list[CapitalAuditLog]  # Line 195
```

### File: data.py

**Key Imports:**
- `from src.database.models.data import DataPipelineRecord`
- `from src.database.models.data import DataQualityRecord`
- `from src.database.models.data import FeatureRecord`
- `from src.database.repository.base import DatabaseRepository`

#### Class: `FeatureRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for FeatureRecord entities

```python
class FeatureRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 19
    async def get_by_symbol(self, symbol: str) -> list[FeatureRecord]  # Line 30
    async def get_by_feature_type(self, feature_type: str) -> list[FeatureRecord]  # Line 34
    async def get_by_symbol_and_type(self, symbol: str, feature_type: str) -> list[FeatureRecord]  # Line 40
    async def get_latest_feature(self, symbol: str, feature_type: str, feature_name: str) -> FeatureRecord | None  # Line 47
    async def get_features_by_date_range(self, symbol: str, start_date: datetime, end_date: datetime) -> list[FeatureRecord]  # Line 58
```

#### Class: `DataQualityRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for DataQualityRecord entities

```python
class DataQualityRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 74
    async def get_by_symbol(self, symbol: str) -> list[DataQualityRecord]  # Line 85
    async def get_by_data_source(self, data_source: str) -> list[DataQualityRecord]  # Line 89
    async def get_poor_quality_records(self, threshold: Decimal = Any) -> list[DataQualityRecord]  # Line 95
    async def get_latest_quality_check(self, symbol: str, data_source: str) -> DataQualityRecord | None  # Line 102
    async def get_quality_trend(self, symbol: str, days: int = 30) -> list[DataQualityRecord]  # Line 113
```

#### Class: `DataPipelineRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for DataPipelineRecord entities

```python
class DataPipelineRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 131
    async def get_by_pipeline_name(self, pipeline_name: str) -> list[DataPipelineRecord]  # Line 142
    async def get_by_status(self, status: str) -> list[DataPipelineRecord]  # Line 148
    async def get_running_pipelines(self) -> list[DataPipelineRecord]  # Line 152
    async def get_failed_pipelines(self) -> list[DataPipelineRecord]  # Line 156
    async def get_latest_execution(self, pipeline_name: str) -> DataPipelineRecord | None  # Line 160
    async def get_pipeline_performance(self, pipeline_name: str, days: int = 30) -> list[DataPipelineRecord]  # Line 167
```

### File: market_data.py

**Key Imports:**
- `from src.database.models.market_data import MarketDataRecord`
- `from src.database.repository.base import DatabaseRepository`
- `from src.database.repository.utils import RepositoryUtils`

#### Class: `MarketDataRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for MarketDataRecord entities

```python
class MarketDataRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 17
    async def get_by_symbol(self, symbol: str) -> list[MarketDataRecord]  # Line 28
    async def get_by_exchange(self, exchange: str) -> list[MarketDataRecord]  # Line 34
    async def get_by_symbol_and_exchange(self, symbol: str, exchange: str) -> list[MarketDataRecord]  # Line 40
    async def get_latest_price(self, symbol: str, exchange: str) -> MarketDataRecord | None  # Line 48
    async def get_ohlc_data(self, symbol: str, exchange: str, start_time: datetime, end_time: datetime) -> list[MarketDataRecord]  # Line 55
    async def get_recent_data(self, symbol: str, exchange: str, hours: int = 24) -> list[MarketDataRecord]  # Line 68
    async def get_by_data_source(self, data_source: str) -> list[MarketDataRecord]  # Line 80
    async def get_poor_quality_data(self, threshold: Decimal = Any) -> list[MarketDataRecord]  # Line 86
    async def get_invalid_data(self) -> list[MarketDataRecord]  # Line 93
    async def cleanup_old_data(self, days: int = 90) -> int  # Line 98
    async def save_ticker(self, exchange: str, symbol: str, data: dict[str, Any]) -> None  # Line 104
    async def get_volume_leaders(self, exchange: str | None = None, limit: int = 10) -> list[MarketDataRecord]  # Line 120
    async def get_price_changes(self, symbol: str, exchange: str, hours: int = 24) -> tuple[Decimal | None, Decimal | None]  # Line 132
```

### File: ml.py

**Key Imports:**
- `from src.database.models.ml import MLModelMetadata`
- `from src.database.models.ml import MLPrediction`
- `from src.database.models.ml import MLTrainingJob`
- `from src.database.repository.base import DatabaseRepository`

#### Class: `MLPredictionRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for ML predictions

```python
class MLPredictionRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 22
    async def get_by_model_and_symbol(self, model_name: str, symbol: str, limit: int = 100) -> list[MLPrediction]  # Line 37
    async def get_recent_predictions(self, ...) -> list[MLPrediction]  # Line 64
    async def get_prediction_accuracy(self, model_name: str, symbol: str | None = None, days: int = 30) -> dict[str, Any]  # Line 100
    async def update_with_actual(self, prediction_id: int, actual_value: Decimal) -> MLPrediction | None  # Line 152
```

#### Class: `MLModelMetadataRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for ML model metadata

```python
class MLModelMetadataRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 178
    async def get_latest_model(self, model_name: str, model_type: str) -> MLModelMetadata | None  # Line 193
    async def get_active_models(self) -> list[MLModelMetadata]  # Line 218
    async def get_by_version(self, model_name: str, version: int) -> MLModelMetadata | None  # Line 232
    async def deactivate_old_versions(self, model_name: str, keep_versions: int = 3) -> int  # Line 253
```

#### Class: `MLTrainingJobRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for ML training jobs

```python
class MLTrainingJobRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 299
    async def get_running_jobs(self) -> list[MLTrainingJob]  # Line 314
    async def get_job_by_model(self, model_name: str, status: str | None = None) -> list[MLTrainingJob]  # Line 328
    async def update_job_status(self, ...) -> MLTrainingJob | None  # Line 351
    async def get_successful_jobs(self, days: int = 30, limit: int = 100) -> list[MLTrainingJob]  # Line 384
```

#### Class: `MLRepository`

**Purpose**: Unified ML repository providing data access to all ML-related repositories

```python
class MLRepository:
    def __init__(self, session: AsyncSession)  # Line 421
```

### File: risk.py

**Key Imports:**
- `from src.core.exceptions import RepositoryError`
- `from src.core.types import Position`
- `from src.core.types import RiskMetrics`
- `from src.database.models.risk import RiskConfiguration`
- `from src.database.repository.base import DatabaseRepository`

#### Class: `RiskMetricsRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for risk metrics data access

```python
class RiskMetricsRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 25
    async def get_historical_returns(self, symbol: str, days: int) -> list[Decimal]  # Line 34
    async def get_price_history(self, symbol: str, days: int) -> list[Decimal]  # Line 39
    async def get_portfolio_positions(self) -> list[Position]  # Line 44
    async def save_risk_metrics(self, metrics: RiskMetrics) -> None  # Line 52
    async def get_correlation_data(self, symbols: list[str], days: int) -> dict[str, list[Decimal]]  # Line 61
```

#### Class: `PortfolioRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for portfolio data access

```python
class PortfolioRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 70
    async def get_current_positions(self) -> list[Position]  # Line 79
    async def get_portfolio_value(self) -> Decimal  # Line 87
    async def get_position_history(self, symbol: str, days: int) -> list[Position]  # Line 95
    async def update_portfolio_limits(self, limits: dict[str, Any]) -> None  # Line 103
```

#### Class: `RiskMetricsRepositoryImpl`

**Inherits**: RiskMetricsRepositoryInterface
**Purpose**: Implementation of risk metrics repository interface

```python
class RiskMetricsRepositoryImpl(RiskMetricsRepositoryInterface):
    def __init__(self, repository: RiskMetricsRepository)  # Line 113
    async def get_historical_returns(self, symbol: str, days: int) -> list[Decimal]  # Line 117
    async def get_price_history(self, symbol: str, days: int) -> list[Decimal]  # Line 121
    async def get_portfolio_positions(self) -> list[Position]  # Line 125
    async def save_risk_metrics(self, metrics: RiskMetrics) -> None  # Line 129
    async def get_correlation_data(self, symbols: list[str], days: int) -> dict[str, list[Decimal]]  # Line 133
```

#### Class: `PortfolioRepositoryImpl`

**Inherits**: PortfolioRepositoryInterface
**Purpose**: Implementation of portfolio repository interface

```python
class PortfolioRepositoryImpl(PortfolioRepositoryInterface):
    def __init__(self, repository: PortfolioRepository)  # Line 141
    async def get_current_positions(self) -> list[Position]  # Line 145
    async def get_portfolio_value(self) -> Decimal  # Line 149
    async def get_position_history(self, symbol: str, days: int) -> list[Position]  # Line 153
    async def update_portfolio_limits(self, limits: dict[str, Any]) -> None  # Line 157
```

### File: service_repository.py

**Key Imports:**
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.repository import BaseRepository`
- `from src.core.exceptions import DatabaseConnectionError`
- `from src.core.exceptions import DatabaseQueryError`
- `from src.core.logging import get_logger`

#### Class: `DatabaseServiceRepository`

**Inherits**: BaseRepository[T, K]
**Purpose**: Database repository implementation using the repository pattern

```python
class DatabaseServiceRepository(BaseRepository[T, K]):
    def __init__(self, ...)  # Line 34
    async def _create_entity(self, entity: T) -> T  # Line 63
    async def _get_entity_by_id(self, entity_id: K) -> T | None  # Line 67
    async def _update_entity(self, entity: T) -> T | None  # Line 71
    async def _delete_entity(self, entity_id: K) -> bool  # Line 75
    async def _list_entities(self, ...) -> list[T]  # Line 79
    async def _count_entities(self, filters: dict[str, Any] | None) -> int  # Line 98
    async def _bulk_create_entities(self, entities: list[T]) -> list[T]  # Line 102
    async def _test_connection(self, connection: Any) -> bool  # Line 106
    async def _repository_health_check(self) -> Any  # Line 118
```

### File: state.py

**Key Imports:**
- `from src.database.models.state import StateBackup`
- `from src.database.models.state import StateCheckpoint`
- `from src.database.models.state import StateHistory`
- `from src.database.models.state import StateMetadata`
- `from src.database.models.state import StateSnapshot`

#### Class: `StateSnapshotRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateSnapshot entities

```python
class StateSnapshotRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 20
    async def get_by_name_prefix(self, name_prefix: str) -> list[StateSnapshot]  # Line 30
    async def get_by_snapshot_type(self, snapshot_type: str) -> list[StateSnapshot]  # Line 36
    async def get_latest_snapshot(self, name_prefix: str, snapshot_type: str | None = None) -> StateSnapshot | None  # Line 40
    async def get_by_schema_version(self, schema_version: str) -> list[StateSnapshot]  # Line 51
    async def cleanup_old_snapshots(self, name_prefix: str, keep_count: int = 10) -> int  # Line 57
```

#### Class: `StateCheckpointRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateCheckpoint entities

```python
class StateCheckpointRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 74
    async def get_by_name_prefix(self, name_prefix: str) -> list[StateCheckpoint]  # Line 84
    async def get_by_checkpoint_type(self, checkpoint_type: str) -> list[StateCheckpoint]  # Line 90
    async def get_latest_checkpoint(self, name_prefix: str) -> StateCheckpoint | None  # Line 96
    async def get_by_status(self, status: str) -> list[StateCheckpoint]  # Line 103
```

#### Class: `StateHistoryRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateHistory entities

```python
class StateHistoryRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 111
    async def get_by_state(self, state_type: str, state_id: str) -> list[StateHistory]  # Line 121
    async def get_by_operation(self, operation: str) -> list[StateHistory]  # Line 128
    async def get_recent_changes(self, state_type: str, state_id: str, hours: int = 24) -> list[StateHistory]  # Line 132
    async def get_by_component(self, source_component: str) -> list[StateHistory]  # Line 145
```

#### Class: `StateMetadataRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateMetadata entities

```python
class StateMetadataRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 156
    async def get_by_state(self, state_type: str, state_id: str) -> StateMetadata | None  # Line 166
    async def get_by_state_type(self, state_type: str) -> list[StateMetadata]  # Line 170
    async def get_critical_states(self, state_type: str | None = None) -> list[StateMetadata]  # Line 174
    async def get_hot_states(self, state_type: str | None = None) -> list[StateMetadata]  # Line 181
```

#### Class: `StateBackupRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for StateBackup entities

```python
class StateBackupRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 192
    async def get_by_name_prefix(self, name_prefix: str) -> list[StateBackup]  # Line 202
    async def get_by_backup_type(self, backup_type: str) -> list[StateBackup]  # Line 208
    async def get_latest_backup(self, name_prefix: str) -> StateBackup | None  # Line 212
    async def get_verified_backups(self, name_prefix: str) -> list[StateBackup]  # Line 219
    async def cleanup_old_backups(self, name_prefix: str, keep_days: int = 30) -> int  # Line 226
```

### File: system.py

**Key Imports:**
- `from src.database.models.system import Alert`
- `from src.database.models.system import AuditLog`
- `from src.database.models.system import BalanceSnapshot`
- `from src.database.models.system import PerformanceMetrics`
- `from src.database.repository.base import DatabaseRepository`

#### Class: `AlertRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for Alert entities

```python
class AlertRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 20
    async def get_by_user(self, user_id: str) -> list[Alert]  # Line 27
    async def get_unread_alerts(self, user_id: str) -> list[Alert]  # Line 31
    async def get_by_severity(self, severity: str) -> list[Alert]  # Line 37
    async def get_critical_alerts(self) -> list[Alert]  # Line 41
    async def get_by_type(self, alert_type: str) -> list[Alert]  # Line 45
    async def mark_as_read(self, alert_id: str) -> bool  # Line 51
    async def mark_all_read(self, user_id: str) -> int  # Line 55
```

#### Class: `AuditLogRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for AuditLog entities

```python
class AuditLogRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 64
    async def get_by_user(self, user_id: str) -> list[AuditLog]  # Line 75
    async def get_by_action(self, action: str) -> list[AuditLog]  # Line 79
    async def get_by_resource_type(self, resource_type: str) -> list[AuditLog]  # Line 83
    async def get_by_resource(self, resource_type: str, resource_id: str) -> list[AuditLog]  # Line 89
    async def get_recent_logs(self, hours: int = 24) -> list[AuditLog]  # Line 94
```

#### Class: `PerformanceMetricsRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for PerformanceMetrics entities

```python
class PerformanceMetricsRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 102
    async def get_by_bot(self, bot_id: str) -> list[PerformanceMetrics]  # Line 113
    async def get_latest_metrics(self, bot_id: str) -> PerformanceMetrics | None  # Line 117
    async def get_metrics_by_date_range(self, bot_id: str, start_date: datetime, end_date: datetime) -> list[PerformanceMetrics]  # Line 122
    async def get_top_performing_bots(self, limit: int = 10) -> list[PerformanceMetrics]  # Line 131
```

#### Class: `BalanceSnapshotRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for BalanceSnapshot entities

```python
class BalanceSnapshotRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 141
    async def get_by_user(self, user_id: str) -> list[BalanceSnapshot]  # Line 152
    async def get_by_exchange(self, exchange: str) -> list[BalanceSnapshot]  # Line 156
    async def get_by_currency(self, currency: str) -> list[BalanceSnapshot]  # Line 160
    async def get_latest_snapshot(self, user_id: str, exchange: str, currency: str) -> BalanceSnapshot | None  # Line 164
    async def get_balance_history(self, user_id: str, exchange: str, currency: str, days: int = 30) -> list[BalanceSnapshot]  # Line 175
```

### File: trading.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.database.models.trading import Order`
- `from src.database.models.trading import OrderFill`
- `from src.database.models.trading import Position`
- `from src.database.models.trading import Trade`

#### Class: `OrderRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for Order entities

```python
class OrderRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 20
    async def get_active_orders(self, bot_id: str | None = None, symbol: str | None = None) -> list[Order]  # Line 26
    async def get_by_exchange_id(self, exchange: str, exchange_order_id: str) -> Order | None  # Line 39
    async def update_order_status(self, order_id: str, status: str) -> bool  # Line 43
    async def get_orders_by_position(self, position_id: str) -> list[Order]  # Line 47
    async def get_recent_orders(self, hours: int = 24, bot_id: str | None = None) -> list[Order]  # Line 51
```

#### Class: `PositionRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for Position entities

```python
class PositionRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 60
    async def get_open_positions(self, bot_id: str | None = None, symbol: str | None = None) -> list[Position]  # Line 70
    async def get_position_by_symbol(self, bot_id: str, symbol: str, side: str) -> Position | None  # Line 83
    async def update_position_status(self, position_id: str, status: str, **fields) -> bool  # Line 87
    async def update_position_fields(self, position_id: str, **fields) -> bool  # Line 93
```

#### Class: `TradeRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for Trade entities

```python
class TradeRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 104
    async def get_profitable_trades(self, bot_id: str | None = None) -> list[Trade]  # Line 110
    async def get_trades_by_symbol(self, symbol: str, bot_id: str | None = None) -> list[Trade]  # Line 119
    async def get_trades_by_bot_and_date(self, bot_id: str, since: datetime | None = None) -> list[Trade]  # Line 126
    async def create_from_position(self, position: Position, exit_order: Order) -> Trade  # Line 140
```

#### Class: `OrderFillRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for OrderFill entities

```python
class OrderFillRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 178
    async def get_fills_by_order(self, order_id: str) -> list[OrderFill]  # Line 188
    async def get_total_filled(self, order_id: str) -> dict[str, Decimal]  # Line 192
    async def create_fill(self, ...) -> OrderFill  # Line 223
```

### File: user.py

**Key Imports:**
- `from src.database.models.user import User`
- `from src.database.repository.base import DatabaseRepository`
- `from src.database.repository.utils import RepositoryUtils`

#### Class: `UserRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for User entities

```python
class UserRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 14
    async def get_by_username(self, username: str) -> User | None  # Line 21
    async def get_by_email(self, email: str) -> User | None  # Line 25
    async def get_active_users(self) -> list[User]  # Line 29
    async def get_verified_users(self) -> list[User]  # Line 33
    async def get_admin_users(self) -> list[User]  # Line 37
    async def activate_user(self, user_id: str) -> bool  # Line 41
    async def deactivate_user(self, user_id: str) -> bool  # Line 45
    async def verify_user(self, user_id: str) -> bool  # Line 49
```

### File: utils.py

**Key Imports:**
- `from src.core.exceptions import RepositoryError`
- `from src.core.logging import get_logger`

#### Class: `RepositoryUtils`

**Inherits**: Generic[T]
**Purpose**: Common utilities for repository operations

```python
class RepositoryUtils(Generic[T]):
    async def update_entity_status(repository, ...) -> bool  # Line 22
    async def update_entity_fields(repository: Any, entity_id: str, entity_name: str, **fields: Any) -> bool  # Line 55
    async def get_entities_by_field(repository, ...) -> list[T]  # Line 86
    async def get_entities_by_multiple_fields(repository: Any, filters: dict[str, Any], order_by: str = '-created_at') -> list[T]  # Line 109
    async def get_recent_entities(repository, ...) -> list[T]  # Line 130
    async def mark_entity_field(repository, ...) -> bool  # Line 164
    async def bulk_mark_entities(repository, ...) -> int  # Line 194
    async def get_total_by_field_aggregation(session, ...) -> Decimal  # Line 225
    async def get_latest_entity_by_field(repository: Any, field_name: str, field_value: Any) -> T | None  # Line 267
    async def cleanup_old_entities(session, ...) -> int  # Line 291
    async def execute_time_based_query(session, ...) -> list[T]  # Line 336
    async def execute_date_range_query(session, ...) -> list[T]  # Line 409
```

### File: repository_factory.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.database.interfaces import RepositoryFactoryInterface`

#### Class: `RepositoryFactory`

**Inherits**: RepositoryFactoryInterface
**Purpose**: Factory for creating repository instances using dependency injection

```python
class RepositoryFactory(RepositoryFactoryInterface):
    def __init__(self, dependency_injector: Any | None = None) -> None  # Line 20
    def create_repository(self, repository_class: type[Any], session: Any) -> Any  # Line 31
    def register_repository(self, name: str, repository_class: type[R]) -> None  # Line 67
    def is_repository_registered(self, name: str) -> bool  # Line 78
    def get_registered_repository(self, name: str) -> type[R] | None  # Line 90
    def configure_dependencies(self, dependency_injector) -> None  # Line 102
    def list_registered_repositories(self) -> list[str]  # Line 112
    def clear_registrations(self) -> None  # Line 121
```

### File: seed_data.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.logging import get_logger`
- `from src.core.types import StrategyStatus`
- `from src.core.types import StrategyType`
- `from src.database.connection import get_db_session`

#### Class: `DatabaseSeeder`

**Purpose**: Handles database seeding for development environment

```python
class DatabaseSeeder:
    def __init__(self, config: Config)  # Line 41
    def _load_seed_data(self) -> dict[str, Any]  # Line 65
    async def seed_users(self, session: AsyncSession) -> list[User]  # Line 200
    async def seed_bot_instances(self, session: AsyncSession, users: list[User]) -> list[BotInstance]  # Line 241
    async def seed_strategies(self, session: AsyncSession, bots: list[BotInstance]) -> list[Strategy]  # Line 296
    async def seed_exchange_credentials(self, session: AsyncSession, users: list[User]) -> list[dict[str, Any]]  # Line 348
    async def seed_sample_trades(self, session: AsyncSession, bots: list[BotInstance]) -> None  # Line 386
    async def seed_all(self) -> None  # Line 444
```

#### Functions:

```python
async def run_seed(config: Config | None = None) -> None  # Line 498
def main()  # Line 512
```

### File: service.py

**Key Imports:**
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.service import BaseService`
- `from src.database.interfaces import DatabaseServiceInterface`
- `from src.core.exceptions import ComponentError`

#### Class: `DatabaseService`

**Inherits**: BaseService, DatabaseServiceInterface
**Purpose**: Simple database service implementing service layer pattern

```python
class DatabaseService(BaseService, DatabaseServiceInterface):
    def __init__(self, ...)  # Line 56
    def config_service(self) -> Any  # Line 94
    def validation_service(self) -> Any  # Line 99
    async def start(self) -> None  # Line 103
    async def stop(self) -> None  # Line 132
    async def get_health_status(self) -> HealthStatus  # Line 163
    async def health_check(self) -> HealthCheckResult  # Line 189
    async def create_entity(self, entity: T, processing_mode: str = 'stream') -> T  # Line 246
    async def get_entity_by_id(self, model_class: type[T], entity_id: K) -> T | None  # Line 315
    async def update_entity(self, entity: T) -> T  # Line 350
    async def delete_entity(self, model_class: type[T], entity_id: K) -> bool  # Line 379
    async def list_entities(self, ...) -> list[T]  # Line 412
    async def count_entities(self, model_class: type[T] | None = None, filters: dict[str, Any] | None = None) -> int  # Line 526
    async def bulk_create(self, entities: list[T]) -> list[T]  # Line 566
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]  # Line 600
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]  # Line 617
    async def get_health_status(self) -> HealthStatus  # Line 629
    def get_performance_metrics(self) -> dict[str, Any]  # Line 642
    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> Any  # Line 649
    async def get_connection_pool_status(self) -> dict[str, Any]  # Line 663
    async def _invalidate_cache_pattern(self, pattern: str) -> None  # Line 684
    def _transform_entity_data(self, entity: T, processing_mode: str) -> T  # Line 698
    def _validate_filter_boundary(self, filters: dict[str, Any], entity_name: str) -> None  # Line 718
    def _apply_consistent_filters(self, query: Any, model_class: type[T], filters: dict[str, Any]) -> Any  # Line 763
    def _propagate_database_error(self, error: Exception, operation: str, entity_name: str) -> None  # Line 782
```

### File: bot_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.database.interfaces import BotMetricsServiceInterface`

#### Class: `BotService`

**Inherits**: BaseService, BotMetricsServiceInterface
**Purpose**: Service layer for bot operations with business logic

```python
class BotService(BaseService, BotMetricsServiceInterface):
    def __init__(self, bot_repo: BotRepository)  # Line 17
    async def get_active_bots(self) -> list[dict[str, Any]]  # Line 35
    async def archive_bot_record(self, bot_id: str) -> bool  # Line 72
    async def get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]  # Line 112
    async def store_bot_metrics(self, metrics_record: dict[str, Any]) -> bool  # Line 157
    def _assess_bot_health(self, bot) -> bool  # Line 195
    def _calculate_uptime_hours(self, bot) -> float  # Line 200
    def _can_archive_bot(self, bot) -> bool  # Line 210
```

### File: market_data_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`
- `from src.database.interfaces import MarketDataServiceInterface`
- `from src.database.repository.market_data import MarketDataRepository`

#### Class: `MarketDataService`

**Inherits**: BaseService, MarketDataServiceInterface
**Purpose**: Service layer for market data operations with business logic

```python
class MarketDataService(BaseService, MarketDataServiceInterface):
    def __init__(self, market_data_repo: MarketDataRepository)  # Line 19
    async def get_latest_price(self, symbol: str) -> Decimal | None  # Line 24
    async def get_historical_data(self, ...) -> list[dict[str, Any]]  # Line 47
```

### File: ml_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.database.interfaces import MLServiceInterface`

#### Class: `MLService`

**Inherits**: BaseService, MLServiceInterface
**Purpose**: Service layer for ML operations with business logic

```python
class MLService(BaseService, MLServiceInterface):
    def __init__(self, ...)  # Line 21
    async def get_model_performance_summary(self, model_name: str, days: int = 30) -> dict[str, Any]  # Line 33
    async def validate_model_deployment(self, model_name: str, version: int) -> bool  # Line 81
    async def get_model_recommendations(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]  # Line 122
```

### File: service_registry.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `ServiceRegistry`

**Purpose**: Registry for managing service instances and dependencies

```python
class ServiceRegistry:
    def __init__(self) -> None  # Line 16
    def register_service(self, name: str, service_instance: Any) -> None  # Line 21
    def register_factory(self, name: str, factory: Callable[[], Any]) -> None  # Line 32
    def get_service(self, name: str) -> Any  # Line 43
    def has_service(self, name: str) -> bool  # Line 70
    def clear_services(self) -> None  # Line 82
    def list_services(self) -> list[str]  # Line 88
```

### File: trading_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.database.interfaces import TradingDataServiceInterface`

#### Class: `TradingService`

**Inherits**: BaseService, TradingDataServiceInterface
**Purpose**: Service layer for trading operations with business logic

```python
class TradingService(BaseService, TradingDataServiceInterface):
    def __init__(self, ...)  # Line 25
    async def cancel_order(self, order_id: str, reason: str = 'User requested') -> bool  # Line 39
    async def close_position(self, position_id: str, close_price: Decimal) -> bool  # Line 70
    async def get_trades_by_bot(self, ...) -> list[Trade]  # Line 105
    async def get_positions_by_bot(self, bot_id: str) -> list[Position]  # Line 132
    async def calculate_total_pnl(self, ...) -> Decimal  # Line 142
    def _can_cancel_order(self, order: Order) -> bool  # Line 164
    def _calculate_realized_pnl(self, position: Position, close_price: Decimal) -> Decimal  # Line 169
    def _calculate_unrealized_pnl(self, position: Position, current_price: Decimal) -> Decimal  # Line 180
    async def update_position_price(self, position_id: str, current_price: Decimal) -> bool  # Line 191
    async def _log_order_cancellation(self, order_id: str, reason: str) -> None  # Line 218
    async def create_trade(self, trade_data: dict) -> dict  # Line 222
    async def get_positions(self, strategy_id: str | None = None, symbol: str | None = None) -> list[dict]  # Line 268
    async def get_trade_statistics(self, bot_id: str, since: datetime | None = None) -> dict[str, Any]  # Line 309
    async def get_total_exposure(self, bot_id: str) -> dict[str, Decimal]  # Line 358
    async def get_order_fill_summary(self, order_id: str) -> dict[str, Decimal]  # Line 393
```

### File: uow.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import DatabaseError`
- `from src.core.exceptions import DatabaseQueryError`
- `from src.core.logging import get_logger`
- `from src.database.interfaces import UnitOfWorkFactoryInterface`

#### Class: `UnitOfWork`

**Purpose**: Unit of Work pattern for managing database transactions with service layer integration

```python
class UnitOfWork:
    def __init__(self, ...)  # Line 43
    def __enter__(self)  # Line 70
    def _create_services_via_di(self)  # Line 84
    def _create_service(self, service_name: str)  # Line 102
    def _create_services_direct(self)  # Line 106
    def _create_internal_repositories(self)  # Line 124
    def _hide_repositories(self) -> None  # Line 138
    def __getattr__(self, name)  # Line 181
    def __exit__(self, exc_type, exc_val, exc_tb)  # Line 226
    def commit(self, processing_mode: str = 'stream')  # Line 241
    def rollback(self)  # Line 292
    def close(self)  # Line 301
    def refresh(self, entity)  # Line 327
    def flush(self)  # Line 332
    def savepoint(self)  # Line 338
    def _propagate_uow_error(self, error: Exception, operation: str, processing_mode: str = 'stream') -> None  # Line 369
```

#### Class: `AsyncUnitOfWork`

**Purpose**: Async Unit of Work pattern for managing database transactions with service layer pattern

```python
class AsyncUnitOfWork:
    def __init__(self, async_session_factory, dependency_injector = None)  # Line 408
    async def __aenter__(self)  # Line 430
    async def _create_services_via_di(self)  # Line 444
    def _create_service(self, service_name: str)  # Line 462
    async def _create_services_direct(self)  # Line 466
    async def _create_internal_repositories(self)  # Line 484
    def _hide_repositories(self) -> None  # Line 498
    def __getattr__(self, name)  # Line 502
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 511
    async def commit(self, processing_mode: str = 'stream')  # Line 572
    async def rollback(self)  # Line 589
    async def close(self)  # Line 603
    async def refresh(self, entity)  # Line 618
    async def flush(self)  # Line 623
    async def savepoint(self)  # Line 629
    async def _propagate_async_uow_error(self, error: Exception, operation: str, processing_mode: str = 'stream') -> None  # Line 659
```

#### Class: `UnitOfWorkFactory`

**Inherits**: UnitOfWorkFactoryInterface
**Purpose**: Factory for creating Unit of Work instances

```python
class UnitOfWorkFactory(UnitOfWorkFactoryInterface):
    def __init__(self, ...)  # Line 693
    def create(self) -> UnitOfWork  # Line 712
    def create_async(self) -> AsyncUnitOfWork  # Line 716
    def transaction(self)  # Line 725
    async def async_transaction(self)  # Line 732
    def configure_dependencies(self, dependency_injector) -> None  # Line 738
```

#### Class: `UnitOfWorkExample`

**Purpose**: Example demonstrating Unit of Work usage patterns

```python
class UnitOfWorkExample:
    def __init__(self, uow_factory: UnitOfWorkFactory)  # Line 748
    async def example_transaction(self, entity_data: dict)  # Line 753
    async def example_multi_service_operation(self, entity_id: str)  # Line 763
```

---
**Generated**: Complete reference for database module
**Total Classes**: 138
**Total Functions**: 55