# STATE Module Reference

## INTEGRATION
**Dependencies**: core, database, error_handling, monitoring, utils
**Used By**: strategies
**Provides**: CheckpointManager, EnvironmentAwareStateManager, MockDatabaseService, QualityController, QualityService, StateBusinessService, StateController, StateManager, StateMonitoringService, StatePersistenceService, StateRecoveryManager, StateService, StateSyncManager, StateSynchronizationService, StateValidationService, TradeLifecycleManager, TradeLifecycleService
**Patterns**: Async Operations, Circuit Breaker, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Performance**:
- Parallel execution
- Parallel execution
- Parallel execution
**Architecture**:
- CheckpointManager inherits from base architecture
- StateController inherits from base architecture
- StateMonitoringService inherits from base architecture

## MODULE OVERVIEW
**Files**: 29 Python files
**Classes**: 96
**Functions**: 12

## COMPLETE API REFERENCE

## PROTOCOLS & INTERFACES

### Protocol: `DatabaseServiceProtocol`

**Purpose**: Protocol defining the interface for database services

**Required Methods:**
- `async start(self) -> None`
- `async stop(self) -> None`
- `async create_entity(self, entity: Any) -> Any`
- `async get_entity_by_id(self, model_class: type, entity_id: Any) -> Any | None`
- `async health_check(self) -> Any`
- `get_metrics(self) -> dict[str, Any]`

### Protocol: `StateControllerProtocol`

**Purpose**: Protocol for state controllers that coordinate service operations

**Required Methods:**
- `async get_state(self, state_type: 'StateType', state_id: str, include_metadata: bool = False) -> dict[str, Any] | None`
- `async set_state(self, ...) -> bool`
- `async delete_state(self, ...) -> bool`

### Protocol: `QualityServiceProtocol`

**Purpose**: Protocol defining the quality service interface

**Required Methods:**
- `async validate_pre_trade(self, ...) -> 'PreTradeValidation'`
- `async analyze_post_trade(self, ...) -> 'PostTradeAnalysis'`
- `async validate_state_consistency(self, state: Any) -> bool`
- `async validate_portfolio_balance(self, portfolio_state: Any) -> bool`
- `async validate_position_consistency(self, position: Any, related_orders: list) -> bool`

### Protocol: `StateBusinessServiceProtocol`

**Purpose**: Protocol defining the state business service interface

**Required Methods:**
- `async validate_state_change(self, ...) -> dict[str, Any]`
- `async process_state_update(self, ...) -> 'StateChange'`
- `async calculate_state_metadata(self, ...) -> 'StateMetadata'`
- `async validate_business_rules(self, state_type: 'StateType', state_data: dict[str, Any], operation: str) -> list[str]`

### Protocol: `StatePersistenceServiceProtocol`

**Purpose**: Protocol defining the state persistence service interface

**Required Methods:**
- `async save_state(self, ...) -> bool`
- `async load_state(self, state_type: 'StateType', state_id: str) -> dict[str, Any] | None`
- `async delete_state(self, state_type: 'StateType', state_id: str) -> bool`
- `async list_states(self, state_type: 'StateType', limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]`
- `async save_snapshot(self, snapshot: 'RuntimeStateSnapshot') -> bool`
- `async load_snapshot(self, snapshot_id: str) -> 'RuntimeStateSnapshot | None'`

### Protocol: `StateSynchronizationServiceProtocol`

**Purpose**: Protocol defining the state synchronization service interface

**Required Methods:**
- `async synchronize_state_change(self, state_change: 'StateChange') -> bool`
- `async broadcast_state_change(self, ...) -> None`
- `async resolve_conflicts(self, ...) -> 'StateChange'`

### Protocol: `StateValidationServiceProtocol`

**Purpose**: Protocol defining the state validation service interface

**Required Methods:**
- `async validate_state_data(self, ...) -> dict[str, Any]`
- `async validate_state_transition(self, ...) -> bool`
- `async validate_business_rules(self, ...) -> list[str]`
- `matches_criteria(self, state: dict[str, Any], criteria: dict[str, Any]) -> bool`

### Protocol: `TradeLifecycleServiceProtocol`

**Purpose**: Protocol defining the trade lifecycle service interface

**Required Methods:**
- `async create_trade_context(self, ...) -> TradeContext`
- `async validate_trade_transition(self, current_state: TradeLifecycleState, new_state: TradeLifecycleState) -> bool`
- `async calculate_trade_performance(self, context: TradeContext) -> dict[str, Any]`
- `async create_history_record(self, context: TradeContext) -> TradeHistoryRecord`

### Protocol: `DatabaseServiceProtocol`

**Purpose**: Protocol for database service interactions

**Required Methods:**
- `initialized(self) -> bool`
- `async start(self) -> None`
- `async health_check(self) -> HealthCheckResult`
- `async create_redis_client(self, config: dict[str, Any]) -> Any`
- `async create_influxdb_client(self, config: dict[str, Any]) -> Any`

### Protocol: `RedisClientProtocol`

**Purpose**: Protocol for Redis client operations

**Required Methods:**
- `async connect(self) -> None`
- `async get(self, key: str) -> str | None`
- `async setex(self, key: str, ttl: int, value: str) -> None`
- `async delete(self, *keys: str) -> int`
- `async keys(self, pattern: str) -> list[str]`
- `async ping(self) -> bool`

### Protocol: `InfluxDBClientProtocol`

**Purpose**: Protocol for InfluxDB client operations

**Required Methods:**
- `connect(self) -> None`
- `write_point(self, point: Any) -> None`
- `ping(self) -> bool`

## IMPLEMENTATIONS

### Implementation: `CheckpointMetadata` ✅

**Purpose**: Metadata for checkpoint management
**Status**: Complete

### Implementation: `RecoveryPlan` ✅

**Purpose**: Recovery plan for restoring bot state
**Status**: Complete

### Implementation: `CheckpointManager` ✅

**Inherits**: BaseComponent
**Purpose**: Advanced checkpoint management system for bot state persistence
**Status**: Complete

**Implemented Methods:**
- `async create_checkpoint(self, ...) -> str` - Line 247
- `async create_compressed_checkpoint(self, data: dict[str, Any]) -> str` - Line 383
- `async create_checkpoint_with_integrity(self, data: dict[str, Any]) -> str` - Line 397
- `async cleanup_old_checkpoints(self) -> None` - Line 409
- `async verify_checkpoint_integrity(self, checkpoint_id: str) -> bool` - Line 459
- `async create_checkpoint_from_dict(self, ...) -> str` - Line 481
- `async restore_checkpoint(self, checkpoint_id: str) -> tuple[str, BotState]` - Line 532
- `async restore_from_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None` - Line 607
- `async create_recovery_plan(self, bot_id: str, target_time: datetime | None = None) -> RecoveryPlan` - Line 643
- `async execute_recovery_plan(self, plan: RecoveryPlan) -> bool` - Line 706
- `async schedule_checkpoint(self, bot_id: str, interval_minutes: int | None = None) -> None` - Line 751
- `async list_checkpoints(self, bot_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]` - Line 769
- `async get_checkpoint_stats(self) -> dict[str, Any]` - Line 803
- `checkpoints(self) -> dict[str, Any]` - Line 1051
- `error_handler(self) -> ErrorHandler` - Line 1057

### Implementation: `StateController` ✅

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: State controller that coordinates state management operations
**Status**: Complete

**Implemented Methods:**
- `async get_state(self, state_type: 'StateType', state_id: str, include_metadata: bool = False) -> dict[str, Any] | None` - Line 89
- `async set_state(self, ...) -> bool` - Line 131
- `async delete_state(self, ...) -> bool` - Line 217
- `async cleanup(self) -> None` - Line 354

### Implementation: `StateDataTransformer` ✅

**Purpose**: Handles consistent data transformation for state module
**Status**: Complete

**Implemented Methods:**
- `transform_state_change_to_event_data(state_type, ...) -> dict[str, Any]` - Line 23
- `transform_state_snapshot_to_event_data(snapshot_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 55
- `transform_error_to_event_data(error, ...) -> dict[str, Any]` - Line 82
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 104
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'state_management') -> dict[str, Any]` - Line 126
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 162
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]` - Line 212
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 237
- `apply_cross_module_validation(cls, ...) -> dict[str, Any]` - Line 301

### Implementation: `StateIsolationLevel` ✅

**Inherits**: Enum
**Purpose**: State isolation levels for different environments
**Status**: Complete

### Implementation: `StatePersistenceMode` ✅

**Inherits**: Enum
**Purpose**: State persistence modes
**Status**: Complete

### Implementation: `StateValidationLevel` ✅

**Inherits**: Enum
**Purpose**: State validation levels
**Status**: Complete

### Implementation: `EnvironmentAwareStateConfiguration` ✅

**Purpose**: Environment-specific state configuration
**Status**: Complete

**Implemented Methods:**
- `get_sandbox_state_config() -> dict[str, Any]` - Line 54
- `get_live_state_config() -> dict[str, Any]` - Line 82

### Implementation: `EnvironmentAwareStateManager` ✅

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware state management functionality
**Status**: Complete

**Implemented Methods:**
- `get_environment_state_config(self, exchange: str) -> dict[str, Any]` - Line 164
- `async set_environment_aware_state(self, ...) -> bool` - Line 177
- `async get_environment_aware_state(self, key: str, state_type: StateType, exchange: str, default: Any = None) -> Any` - Line 239
- `async delete_environment_aware_state(self, key: str, exchange: str) -> bool` - Line 288
- `async create_environment_state_checkpoint(self, exchange: str) -> dict[str, Any]` - Line 322
- `async rollback_to_checkpoint(self, checkpoint_id: str, exchange: str, confirm_rollback: bool = False) -> bool` - Line 369
- `async validate_environment_state_consistency(self, exchange: str) -> dict[str, Any]` - Line 423
- `async migrate_state_between_environments(self, ...) -> dict[str, Any]` - Line 473
- `get_environment_state_metrics(self, exchange: str) -> dict[str, Any]` - Line 631

### Implementation: `StateErrorRecovery` ✅

**Inherits**: BaseErrorRecovery
**Purpose**: State-specific error recovery system extending base recovery functionality
**Status**: Complete

**Implemented Methods:**
- `async create_recovery_checkpoint(self, operation: str, state_data: dict[str, Any] | None = None, **context) -> str` - Line 48
- `async handle_error(self, ...) -> StateErrorContext` - Line 100
- `async rollback_to_checkpoint(self, checkpoint_id: str, session: AsyncSession | None = None) -> bool` - Line 218
- `get_error_statistics(self) -> dict[str, Any]` - Line 382
- `cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int` - Line 399

### Implementation: `DatabaseServiceWrapper` ✅

**Purpose**: Wrapper to add compatibility for database service with StateService
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 59
- `async stop(self) -> None` - Line 64
- `initialized(self) -> bool` - Line 70

### Implementation: `MockDatabaseService` ✅

**Purpose**: Mock database service implementation for testing
**Status**: Complete

**Implemented Methods:**
- `initialized(self) -> bool` - Line 93
- `async start(self) -> None` - Line 97
- `async stop(self) -> None` - Line 102
- `async create_entity(self, entity: Any) -> Any` - Line 108
- `async get_entity_by_id(self, model_class: type, entity_id: Any) -> Any | None` - Line 116
- `async health_check(self) -> dict[str, Any]` - Line 126
- `get_metrics(self) -> dict[str, Any]` - Line 134

### Implementation: `StateServiceFactory` ✅

**Inherits**: StateServiceFactoryInterface
**Purpose**: Factory for creating and configuring StateService instances
**Status**: Complete

**Implemented Methods:**
- `async create_state_service(self, ...) -> 'StateService'` - Line 167
- `async create_state_service_for_testing(self, config: Config | None = None, mock_database: bool = False) -> 'StateService'` - Line 220

### Implementation: `StateServiceRegistry` ✅

**Purpose**: Registry for managing StateService instances across the application
**Status**: Complete

**Implemented Methods:**
- `async get_instance(cls, ...) -> 'StateService'` - Line 272
- `async register_instance(cls, name: str, instance: 'StateService') -> None` - Line 311
- `async remove_instance(cls, name: str) -> None` - Line 323
- `async cleanup_all(cls) -> None` - Line 336
- `list_instances(cls) -> list[str]` - Line 344
- `async get_health_status(cls) -> dict[str, dict]` - Line 349

### Implementation: `StateBusinessServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for state business services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async validate_state_change(self, ...) -> dict[str, Any]` - Line 44
- `async process_state_update(self, ...) -> 'StateChange'` - Line 56
- `async calculate_state_metadata(self, ...) -> 'StateMetadata'` - Line 68

### Implementation: `StatePersistenceServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for state persistence services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async save_state(self, ...) -> bool` - Line 83
- `async load_state(self, state_type: 'StateType', state_id: str) -> dict[str, Any] | None` - Line 94
- `async delete_state(self, state_type: 'StateType', state_id: str) -> bool` - Line 99
- `async save_snapshot(self, snapshot: 'RuntimeStateSnapshot') -> bool` - Line 104
- `async load_snapshot(self, snapshot_id: str) -> 'RuntimeStateSnapshot | None'` - Line 109

### Implementation: `StateValidationServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for state validation services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async validate_state_data(self, ...) -> dict[str, Any]` - Line 118
- `async validate_state_transition(self, ...) -> bool` - Line 128
- `async validate_business_rules(self, ...) -> list[str]` - Line 138
- `matches_criteria(self, state: dict[str, Any], criteria: dict[str, Any]) -> bool` - Line 148

### Implementation: `StateSynchronizationServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for state synchronization services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async synchronize_state_change(self, state_change: 'StateChange') -> bool` - Line 161
- `async broadcast_state_change(self, ...) -> None` - Line 166
- `async resolve_conflicts(self, ...) -> 'StateChange'` - Line 177

### Implementation: `CheckpointServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for checkpoint services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_checkpoint(self, bot_id: str, state_data: dict[str, Any], checkpoint_type: str = 'manual') -> str` - Line 191
- `async restore_checkpoint(self, checkpoint_id: str) -> tuple[str, dict[str, Any]] | None` - Line 201
- `async list_checkpoints(self, bot_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]` - Line 206
- `async delete_checkpoint(self, checkpoint_id: str) -> bool` - Line 213

### Implementation: `StateEventServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for state event services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async emit_state_event(self, event_type: str, event_data: dict[str, Any]) -> None` - Line 222
- `subscribe_to_events(self, event_type: str, callback: Any) -> None` - Line 227
- `unsubscribe_from_events(self, event_type: str, callback: Any) -> None` - Line 232

### Implementation: `StateServiceFactoryInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for state service factories
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_state_service(self, ...) -> 'StateService'` - Line 241
- `async create_state_service_for_testing(self, config: Union['Config', None] = None, mock_database: bool = False) -> 'StateService'` - Line 251

### Implementation: `MetricsStorageInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for metrics storage operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async store_validation_metrics(self, validation_data: dict[str, Any]) -> bool` - Line 264
- `async store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool` - Line 269
- `async get_historical_metrics(self, metric_type: str, start_time: Any, end_time: Any) -> list[dict[str, Any]]` - Line 274

### Implementation: `MetricType` ✅

**Inherits**: Enum
**Purpose**: Metric type enumeration for state metrics
**Status**: Complete

### Implementation: `HealthCheck` ✅

**Purpose**: Health check definition
**Status**: Complete

### Implementation: `Metric` ✅

**Purpose**: Metric data point
**Status**: Complete

### Implementation: `Alert` ✅

**Purpose**: Alert notification
**Status**: Complete

### Implementation: `PerformanceReport` ✅

**Purpose**: Performance analysis report
**Status**: Complete

### Implementation: `StateMonitoringService` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive monitoring service for state management system with central integration
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 251
- `async cleanup(self) -> None` - Line 272
- `register_health_check(self, ...) -> str` - Line 339
- `async get_health_status(self) -> dict[str, Any]` - Line 376
- `record_metric(self, ...) -> None` - Line 443
- `record_operation_time(self, operation_name: str, duration_ms: float) -> None` - Line 493
- `get_metrics(self) -> dict[str, Any]` - Line 513
- `async get_filtered_metrics(self, ...) -> dict[str, list[Metric]]` - Line 517
- `set_alert_threshold(self, ...) -> None` - Line 566
- `register_alert_handler(self, handler: Callable[[Alert], None]) -> None` - Line 596
- `async get_active_alerts(self) -> list[Alert]` - Line 600
- `async acknowledge_alert(self, alert_id: str, acknowledged_by: str = '') -> bool` - Line 604
- `async generate_performance_report(self, start_time: datetime | None = None, end_time: datetime | None = None) -> PerformanceReport` - Line 621

### Implementation: `StateMetricsAdapter` ✅

**Inherits**: BaseComponent
**Purpose**: Adapter to bridge state monitoring with central metrics collection
**Status**: Complete

**Implemented Methods:**
- `record_state_metric(self, metric: Metric) -> None` - Line 57
- `record_operation_time(self, operation: str, duration_ms: float) -> None` - Line 112
- `record_health_check(self, check_name: str, status: HealthStatus) -> None` - Line 118

### Implementation: `StateAlertAdapter` ✅

**Inherits**: BaseComponent
**Purpose**: Adapter to integrate state alerts with central alerting system
**Status**: Complete

**Implemented Methods:**
- `alert_manager(self)` - Line 158
- `async send_alert(self, alert: Alert) -> None` - Line 164

### Implementation: `QualityLevel` ✅

**Inherits**: Enum
**Purpose**: Quality assessment levels
**Status**: Complete

### Implementation: `QualityTrend` ✅

**Purpose**: Quality trend analysis
**Status**: Complete

### Implementation: `InfluxDBMetricsStorage` ✅

**Inherits**: MetricsStorageInterface
**Purpose**: InfluxDB implementation of MetricsStorage interface
**Status**: Complete

**Implemented Methods:**
- `async close(self) -> None` - Line 123
- `async store_validation_metrics(self, validation_data: dict[str, Any]) -> bool` - Line 159
- `async store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool` - Line 195
- `async get_historical_metrics(self, metric_type: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]` - Line 242

### Implementation: `NullMetricsStorage` ✅

**Inherits**: MetricsStorageInterface
**Purpose**: Null implementation of MetricsStorage for testing or when metrics storage is disabled
**Status**: Complete

**Implemented Methods:**
- `async store_validation_metrics(self, validation_data: dict[str, Any]) -> bool` - Line 263
- `async store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool` - Line 267
- `async get_historical_metrics(self, metric_type: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]` - Line 271

### Implementation: `QualityController` ✅

**Inherits**: BaseComponent
**Purpose**: Quality control controller that coordinates quality management operations
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 401
- `async validate_pre_trade(self, ...) -> PreTradeValidation` - Line 422
- `async analyze_post_trade(self, ...) -> PostTradeAnalysis` - Line 486
- `async get_quality_summary(self, bot_id: str | None = None, hours: int = 24) -> dict[str, Any]` - Line 546
- `async get_quality_trend_analysis(self, metric: str, days: int = 7) -> QualityTrend` - Line 584
- `get_quality_metrics(self) -> dict[str, Any]` - Line 947
- `async get_summary_statistics(self, hours: int = 24, bot_id: str | None = None) -> dict[str, Any]` - Line 968
- `async validate_state_consistency(self, state: Any) -> bool` - Line 1042
- `async validate_portfolio_balance(self, portfolio_state: Any) -> bool` - Line 1058
- `async validate_position_consistency(self, position: Any, related_orders: list) -> bool` - Line 1074
- `async run_integrity_checks(self, state: Any) -> dict[str, Any]` - Line 1091
- `async suggest_corrections(self, state: Any) -> list[dict[str, Any]]` - Line 1128
- `async cleanup(self) -> None` - Line 1181
- `add_validation_rule(self, name: str, rule: Callable[Ellipsis, Any]) -> None` - Line 1194

### Implementation: `RecoveryStatus` ✅

**Inherits**: Enum
**Purpose**: Recovery operation status
**Status**: Complete

### Implementation: `AuditEntry` ✅

**Purpose**: Audit trail entry for state changes
**Status**: Complete

### Implementation: `RecoveryPoint` ✅

**Purpose**: Point-in-time recovery information
**Status**: Complete

### Implementation: `RecoveryOperation` ✅

**Purpose**: Recovery operation tracking
**Status**: Complete

### Implementation: `CorruptionReport` ✅

**Purpose**: State corruption detection report
**Status**: Complete

### Implementation: `StateRecoveryManager` ✅

**Inherits**: BaseComponent
**Purpose**: Enterprise-grade state recovery and audit trail manager
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 215
- `async cleanup(self) -> None` - Line 242
- `async record_state_change(self, ...) -> str` - Line 309
- `async get_audit_trail(self, ...) -> list[AuditEntry]` - Line 373
- `async create_recovery_point(self, description: str = '') -> str` - Line 432
- `async list_recovery_points(self, ...) -> list[RecoveryPoint]` - Line 480
- `async recover_to_point(self, ...) -> str` - Line 521
- `async get_recovery_status(self, operation_id: str) -> RecoveryOperation | None` - Line 585
- `async detect_corruption(self, state_type: str | None = None, state_id: str | None = None) -> list[CorruptionReport]` - Line 591
- `async repair_corruption(self, report_id: str, repair_method: str = 'auto') -> bool` - Line 644

### Implementation: `QualityService` ✅

**Inherits**: BaseService
**Purpose**: Quality service implementing core quality control business logic
**Status**: Complete

**Implemented Methods:**
- `async validate_pre_trade(self, ...) -> 'PreTradeValidation'` - Line 82
- `async analyze_post_trade(self, ...) -> 'PostTradeAnalysis'` - Line 112
- `async validate_state_consistency(self, state: Any) -> bool` - Line 148
- `async validate_portfolio_balance(self, portfolio_state: Any) -> bool` - Line 182
- `async validate_position_consistency(self, position: Any, related_orders: list) -> bool` - Line 219

### Implementation: `StateBusinessService` ✅

**Inherits**: BaseService
**Purpose**: State business service implementing core state management business logic
**Status**: Complete

**Implemented Methods:**
- `async validate_state_change(self, ...) -> dict[str, Any]` - Line 89
- `async process_state_update(self, ...) -> 'StateChange'` - Line 156
- `async calculate_state_metadata(self, ...) -> 'StateMetadata'` - Line 231
- `async validate_business_rules(self, state_type: 'StateType', state_data: dict[str, Any], operation: str) -> list[str]` - Line 269

### Implementation: `StatePersistenceService` ✅

**Inherits**: BaseService
**Purpose**: State persistence service providing database-agnostic state storage
**Status**: Complete

**Implemented Methods:**
- `database_service(self)` - Line 106
- `async start(self) -> None` - Line 110
- `async stop(self) -> None` - Line 125
- `async save_state(self, ...) -> bool` - Line 158
- `async load_state(self, state_type: 'StateType', state_id: str) -> dict[str, Any] | None` - Line 234
- `async delete_state(self, state_type: 'StateType', state_id: str) -> bool` - Line 280
- `async list_states(self, state_type: 'StateType', limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]` - Line 321
- `async save_snapshot(self, snapshot: 'RuntimeStateSnapshot') -> bool` - Line 388
- `async load_snapshot(self, snapshot_id: str) -> 'RuntimeStateSnapshot | None'` - Line 433
- `async queue_save_operation(self, ...) -> None` - Line 470
- `async queue_delete_operation(self, state_type: 'StateType', state_id: str) -> None` - Line 490
- `is_available(self) -> bool` - Line 569

### Implementation: `StateSynchronizationService` ✅

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: State synchronization service providing distributed state consistency
**Status**: Complete

**Implemented Methods:**
- `async synchronize_state_change(self, state_change: 'StateChange') -> bool` - Line 95
- `async broadcast_state_change(self, ...) -> None` - Line 202
- `async resolve_conflicts(self, ...) -> 'StateChange'` - Line 260
- `subscribe_to_state_changes(self, state_type: StateType, callback: Callable) -> None` - Line 301
- `unsubscribe_from_state_changes(self, state_type: StateType, callback: Callable) -> None` - Line 315
- `get_synchronization_metrics(self) -> dict[str, Any]` - Line 327
- `async cleanup_expired_locks(self) -> None` - Line 563

### Implementation: `StateValidationService` ✅

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: State validation service providing comprehensive validation capabilities
**Status**: Complete

**Implemented Methods:**
- `async validate_state_data(self, ...) -> dict[str, Any]` - Line 103
- `async validate_state_transition(self, ...) -> bool` - Line 200
- `async validate_business_rules(self, ...) -> list[str]` - Line 248
- `matches_criteria(self, state: dict[str, Any], criteria: dict[str, Any]) -> bool` - Line 294
- `get_validation_metrics(self) -> dict[str, Any]` - Line 335

### Implementation: `TradeLifecycleState` ✅

**Inherits**: Enum
**Purpose**: Trade lifecycle state enumeration
**Status**: Complete

### Implementation: `TradeContext` ✅

**Purpose**: Complete context for a trade throughout its lifecycle
**Status**: Complete

### Implementation: `TradeHistoryRecord` ✅

**Purpose**: Historical trade record for analysis
**Status**: Complete

### Implementation: `TradeLifecycleService` ✅

**Inherits**: BaseService
**Purpose**: Trade lifecycle service implementing core trade lifecycle business logic
**Status**: Complete

**Implemented Methods:**
- `async create_trade_context(self, ...) -> TradeContext` - Line 168
- `async validate_trade_transition(self, current_state: TradeLifecycleState, new_state: TradeLifecycleState) -> bool` - Line 238
- `async calculate_trade_performance(self, context: TradeContext) -> dict[str, Any]` - Line 266
- `async create_history_record(self, context: TradeContext) -> TradeHistoryRecord` - Line 328
- `async apply_business_rules(self, context: TradeContext) -> list[str]` - Line 390

### Implementation: `StateManager` ✅

**Inherits**: BaseComponent
**Purpose**: Backward compatibility wrapper for StateService
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 41
- `async shutdown(self) -> None` - Line 60
- `async save_bot_state(self, bot_id: str, state: dict[str, Any], create_snapshot: bool = False) -> str` - Line 65
- `async load_bot_state(self, bot_id: str) -> BotState | None` - Line 95
- `async create_checkpoint(self, bot_id: str, checkpoint_data: dict[str, Any] | None = None) -> str` - Line 148
- `async restore_from_checkpoint(self, bot_id: str, checkpoint_id: str) -> bool` - Line 158
- `async get_state_metrics(self, bot_id: str | None = None, hours: int = 24) -> dict[str, Any]` - Line 166

### Implementation: `StatePersistence` ✅

**Inherits**: BaseComponent
**Purpose**: Handles state persistence operations for the StateService
**Status**: Complete

**Implemented Methods:**
- `database_service(self)` - Line 55
- `async initialize(self) -> None` - Line 59
- `async cleanup(self) -> None` - Line 86
- `async load_state(self, state_type: 'StateType', state_id: str) -> dict[str, Any] | None` - Line 119
- `async save_state(self, ...) -> bool` - Line 142
- `async delete_state(self, state_type: 'StateType', state_id: str) -> bool` - Line 175
- `async queue_state_save(self, ...) -> None` - Line 198
- `async queue_state_delete(self, state_type: 'StateType', state_id: str) -> None` - Line 215
- `async get_states_by_type(self, ...) -> list[dict[str, Any]]` - Line 224
- `async search_states(self, ...) -> list[dict[str, Any]]` - Line 257
- `async save_snapshot(self, snapshot: 'RuntimeStateSnapshot') -> bool` - Line 305
- `async load_snapshot(self, snapshot_id: str) -> 'RuntimeStateSnapshot | None'` - Line 327
- `async load_all_states_to_cache(self) -> None` - Line 349

### Implementation: `StateOperation` ✅

**Inherits**: Enum
**Purpose**: State operation enumeration
**Status**: Complete

### Implementation: `StateChange` ✅

**Purpose**: Represents a state change for audit and synchronization
**Status**: Complete

### Implementation: `RuntimeStateSnapshot` ✅

**Purpose**: Runtime state snapshot data structure for in-memory operations
**Status**: Complete

### Implementation: `StateValidationResult` ✅

**Purpose**: Result of state validation operation
**Status**: Complete

### Implementation: `StateMetrics` ✅

**Purpose**: State management performance and health metrics
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict[str, int | float | str | None]` - Line 199

### Implementation: `StateService` ✅

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: Comprehensive state management service providing enterprise-grade
state handling with synchronizatio
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 399
- `async cleanup(self) -> None` - Line 449
- `async get_state(self, state_type: StateType, state_id: str, include_metadata: bool = False) -> dict[str, Any] | None` - Line 530
- `async set_state(self, ...) -> bool` - Line 689
- `async delete_state(self, ...) -> bool` - Line 813
- `async get_states_by_type(self, ...) -> list[dict[str, Any]]` - Line 887
- `async search_states(self, ...) -> list[dict[str, Any]]` - Line 931
- `async create_snapshot(self, description: str = '', state_types: list[StateType] | None = None) -> str` - Line 969
- `async restore_snapshot(self, snapshot_id: str) -> bool` - Line 1014
- `subscribe(self, ...) -> None` - Line 1057
- `unsubscribe(self, state_type: StateType, callback: Callable) -> None` - Line 1075
- `get_metrics(self) -> dict[str, int | float | str]` - Line 1090
- `async get_state_metrics(self) -> StateMetrics` - Line 1108
- `async get_health_status(self) -> dict[str, Any]` - Line 1116
- `error_handler(self) -> ErrorHandler` - Line 1686

### Implementation: `SyncEventType` ✅

**Inherits**: Enum
**Purpose**: Sync event types for backward compatibility
**Status**: Complete

### Implementation: `StateSyncManager` ✅

**Inherits**: BaseComponent
**Purpose**: Backward compatibility wrapper for StateSynchronizer
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 66
- `async shutdown(self) -> None` - Line 71
- `async sync_state(self, ...) -> bool` - Line 76
- `async force_sync(self, *args) -> dict[str, Any]` - Line 127
- `async get_sync_status(self, *args) -> dict[str, Any]` - Line 160
- `async get_sync_metrics(self, hours: int = 24) -> dict[str, Any]` - Line 190
- `async subscribe_to_events(self, event_type: str, callback: Any) -> str` - Line 209
- `async register_conflict_resolver(self, state_type: str, resolver: Any) -> None` - Line 219
- `event_subscribers(self)` - Line 234
- `custom_resolvers(self)` - Line 239

### Implementation: `StateSynchronizer` ✅

**Inherits**: BaseComponent
**Purpose**: Handles state synchronization across components and services
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 63
- `async cleanup(self) -> None` - Line 87
- `async queue_state_sync(self, state_change: 'StateChange') -> None` - Line 127
- `async sync_pending_changes(self) -> bool` - Line 159
- `async force_sync(self) -> bool` - Line 368
- `async get_sync_status(self) -> dict[str, Any]` - Line 389
- `async sync_with_remotes(self, remotes: list[str]) -> bool` - Line 408
- `async check_consistency(self, state_data: dict) -> bool` - Line 424
- `async synchronize_state(self, state_change) -> bool` - Line 430
- `async rollback_sync(self, state_change) -> bool` - Line 436
- `async synchronize_with_priority(self, state_change) -> bool` - Line 442
- `async send_heartbeat(self, heartbeat_data: dict) -> bool` - Line 448
- `async cleanup_stale_data(self, max_age_hours: int) -> int` - Line 454

### Implementation: `ValidationLevel` ✅

**Inherits**: Enum
**Purpose**: Validation level enumeration
**Status**: Complete

### Implementation: `ValidationRule` ✅

**Inherits**: Enum
**Purpose**: Validation rule types
**Status**: Complete

### Implementation: `ValidationRuleConfig` ✅

**Purpose**: Configuration for a validation rule
**Status**: Complete

### Implementation: `StateValidationError` ✅

**Purpose**: Individual validation error
**Status**: Complete

### Implementation: `ValidationWarning` ✅

**Purpose**: Individual validation warning
**Status**: Complete

### Implementation: `ValidationResult` ✅

**Purpose**: Complete validation result
**Status**: Complete

### Implementation: `ValidationMetrics` ✅

**Purpose**: Validation performance metrics
**Status**: Complete

### Implementation: `StateValidator` ✅

**Inherits**: BaseComponent
**Purpose**: State validation controller that delegates to StateValidationService
**Status**: Complete

**Implemented Methods:**
- `async validate_state(self, ...) -> ValidationResult` - Line 214
- `async validate_state_transition(self, ...) -> bool` - Line 303
- `async validate_cross_state_consistency(self, ...) -> ValidationResult` - Line 332
- `add_validation_rule(self, state_type: 'StateType', rule_config: ValidationRuleConfig) -> None` - Line 384
- `remove_validation_rule(self, state_type: 'StateType', field_name: str, rule_type: ValidationRule) -> bool` - Line 394
- `update_validation_level(self, level: ValidationLevel) -> None` - Line 407
- `get_metrics(self) -> dict[str, int | float | str]` - Line 417
- `async get_validation_metrics(self) -> ValidationMetrics` - Line 432
- `get_validation_rules(self, state_type: 'StateType') -> list[ValidationRuleConfig]` - Line 441

### Implementation: `TradeEvent` ✅

**Inherits**: str, Enum
**Purpose**: Trade event enumeration
**Status**: Complete

### Implementation: `PerformanceAttribution` ✅

**Purpose**: Performance attribution analysis for trades
**Status**: Complete

### Implementation: `TradeLifecycleManager` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive trade lifecycle management system
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 174
- `async cleanup(self) -> None` - Line 206
- `async start_trade_lifecycle(self, bot_id: str, strategy_name: str, order_request: OrderRequest) -> str` - Line 249
- `async transition_trade_state(self, ...) -> bool` - Line 306
- `async update_trade_execution(self, trade_id: str, execution_result: ExecutionResult) -> None` - Line 384
- `async update_trade_event(self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]) -> None` - Line 456
- `async calculate_trade_performance(self, trade_id: str) -> dict[str, Any]` - Line 523
- `async get_trade_history(self, ...) -> list[dict[str, Any]]` - Line 605
- `async get_performance_attribution(self, bot_id: str, period_days: int = 30) -> dict[str, Any]` - Line 673
- `async create_trade_state(self, trade: Any) -> None` - Line 1046
- `async validate_trade_state(self, trade: Any) -> bool` - Line 1082
- `async calculate_trade_pnl(self, trade: Any) -> Decimal` - Line 1115
- `async assess_trade_risk(self, trade: Any) -> str` - Line 1144
- `async close_trade(self, trade_id: str, final_pnl: Decimal) -> None` - Line 1173
- `async update_trade_state(self, trade_id: str, trade_data: Any) -> None` - Line 1198

### Implementation: `ValidationResult` ✅

**Inherits**: Enum
**Purpose**: Validation result enumeration
**Status**: Complete

### Implementation: `ValidationCheck` ✅

**Purpose**: Individual validation check result
**Status**: Complete

### Implementation: `PreTradeValidation` ✅

**Purpose**: Pre-trade validation results
**Status**: Complete

### Implementation: `PostTradeAnalysis` ✅

**Purpose**: Post-trade analysis results
**Status**: Complete

### Implementation: `MigrationType` ✅

**Inherits**: Enum
**Purpose**: Types of migrations
**Status**: Complete

### Implementation: `MigrationStatus` ✅

**Inherits**: Enum
**Purpose**: Migration execution status
**Status**: Complete

### Implementation: `StateVersion` ✅

**Purpose**: State version information
**Status**: Complete

**Implemented Methods:**
- `is_compatible_with(self, other: 'StateVersion') -> bool` - Line 93

### Implementation: `MigrationRecord` ✅

**Purpose**: Record of a migration operation
**Status**: Complete

### Implementation: `StateMigration` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for state migrations
**Status**: Abstract Base Class

**Implemented Methods:**
- `from_version(self) -> StateVersion` - Line 149
- `to_version(self) -> StateVersion` - Line 155
- `migration_type(self) -> MigrationType` - Line 161
- `depends_on(self) -> list[str]` - Line 166
- `affected_state_types(self) -> list[str]` - Line 171
- `async migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]` - Line 176
- `async rollback(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]` - Line 190
- `async validate_pre_migration(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> bool` - Line 205
- `async validate_post_migration(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> bool` - Line 211

### Implementation: `StateVersioningSystem` ✅

**Purpose**: State versioning and migration management system
**Status**: Complete

**Implemented Methods:**
- `register_migration(self, migration: StateMigration) -> None` - Line 272
- `register_version_schema(self, version: str, schema: dict[str, Any]) -> None` - Line 282
- `set_version_compatibility(self, version: str, compatible_versions: list[str]) -> None` - Line 287
- `get_migration_path(self, from_version: StateVersion, to_version: StateVersion) -> list[str]` - Line 291
- `async migrate_state(self, ...) -> dict[str, Any]` - Line 336
- `async batch_migrate_states(self, states: list[dict[str, Any]], target_version: str | None = None) -> list[dict[str, Any]]` - Line 465
- `is_version_compatible(self, version1: str, version2: str) -> bool` - Line 516
- `get_schema_for_version(self, version: str) -> dict[str, Any]` - Line 531
- `async validate_state_schema(self, state_data: dict[str, Any], state_type: str, version: str) -> bool` - Line 535
- `async get_version_statistics(self) -> dict[str, Any]` - Line 557
- `get_migration_history(self, limit: int = 100) -> list[dict[str, Any]]` - Line 626

### Implementation: `AddTimestampMigration` ✅

**Inherits**: StateMigration
**Purpose**: Example migration to add timestamp field
**Status**: Complete

**Implemented Methods:**
- `from_version(self) -> StateVersion` - Line 666
- `to_version(self) -> StateVersion` - Line 670
- `migration_type(self) -> MigrationType` - Line 674
- `async migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]` - Line 677
- `async rollback(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]` - Line 690

### Implementation: `RenameFieldMigration` ✅

**Inherits**: StateMigration
**Purpose**: Example migration to rename a field
**Status**: Complete

**Implemented Methods:**
- `from_version(self) -> StateVersion` - Line 714
- `to_version(self) -> StateVersion` - Line 718
- `migration_type(self) -> MigrationType` - Line 722
- `affected_state_types(self) -> list[str]` - Line 726
- `async migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]` - Line 729
- `async rollback(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]` - Line 740

## COMPLETE API REFERENCE

### File: checkpoint_manager.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
- `from src.core.exceptions import ErrorSeverity`
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.exceptions import StateCorruptionError`

#### Class: `CheckpointMetadata`

**Purpose**: Metadata for checkpoint management

```python
class CheckpointMetadata:
```

#### Class: `RecoveryPlan`

**Purpose**: Recovery plan for restoring bot state

```python
class RecoveryPlan:
```

#### Class: `CheckpointManager`

**Inherits**: BaseComponent
**Purpose**: Advanced checkpoint management system for bot state persistence

```python
class CheckpointManager(BaseComponent):
    def __init__(self, config: Config, checkpoint_service = None)  # Line 99
    async def _do_start(self) -> None  # Line 158
    async def _do_stop(self) -> None  # Line 191
    async def create_checkpoint(self, ...) -> str  # Line 247
    async def create_compressed_checkpoint(self, data: dict[str, Any]) -> str  # Line 383
    async def create_checkpoint_with_integrity(self, data: dict[str, Any]) -> str  # Line 397
    async def cleanup_old_checkpoints(self) -> None  # Line 409
    async def verify_checkpoint_integrity(self, checkpoint_id: str) -> bool  # Line 459
    async def create_checkpoint_from_dict(self, ...) -> str  # Line 481
    async def restore_checkpoint(self, checkpoint_id: str) -> tuple[str, BotState]  # Line 532
    async def restore_from_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None  # Line 607
    async def create_recovery_plan(self, bot_id: str, target_time: datetime | None = None) -> RecoveryPlan  # Line 643
    async def execute_recovery_plan(self, plan: RecoveryPlan) -> bool  # Line 706
    async def schedule_checkpoint(self, bot_id: str, interval_minutes: int | None = None) -> None  # Line 751
    async def list_checkpoints(self, bot_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]  # Line 769
    async def get_checkpoint_stats(self) -> dict[str, Any]  # Line 803
    async def _get_last_checkpoint(self, bot_id: str) -> CheckpointMetadata | None  # Line 837
    async def _find_best_checkpoint(self, bot_id: str, target_time: datetime | None = None) -> CheckpointMetadata | None  # Line 846
    async def _validate_checkpoint(self, checkpoint_id: str, metadata: CheckpointMetadata) -> dict[str, Any]  # Line 869
    async def _cleanup_old_checkpoints(self, bot_id: str) -> None  # Line 921
    async def _load_existing_checkpoints(self) -> None  # Line 954
    def _update_performance_stats(self, operation: str, duration_ms: float, original_size: int, final_size: int) -> None  # Line 968
    async def _scheduler_loop(self) -> None  # Line 1002
    async def _cleanup_loop(self) -> None  # Line 1034
    def checkpoints(self) -> dict[str, Any]  # Line 1051
    def error_handler(self) -> ErrorHandler  # Line 1057
    def __contains__(self, checkpoint_id: str) -> bool  # Line 1064
    def __getitem__(self, checkpoint_id: str) -> CheckpointMetadata  # Line 1068
```

### File: consistency.py

**Key Imports:**
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.logging import get_logger`

#### Functions:

```python
def validate_state_data(data_type: str, data: Any) -> dict[str, Any]  # Line 17
def raise_state_error(message: str, context: dict[str, Any] | None = None) -> None  # Line 61
async def process_state_change(change: Any, processor: Callable) -> Any  # Line 79
async def emit_state_event(event_type: str, event_data: dict[str, Any]) -> None  # Line 103
```

### File: controller.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.config.main import Config`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.exceptions import ValidationError`

#### Class: `StateController`

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: State controller that coordinates state management operations

```python
class StateController(BaseService, ErrorPropagationMixin):
    def __init__(self, ...)  # Line 45
    def _resolve_service(self, service_name: str, factory_func = None)  # Line 79
    async def get_state(self, state_type: 'StateType', state_id: str, include_metadata: bool = False) -> dict[str, Any] | None  # Line 89
    async def set_state(self, ...) -> bool  # Line 131
    async def delete_state(self, ...) -> bool  # Line 217
    async def _coordinate_validation(self, ...) -> None  # Line 283
    async def _coordinate_persistence(self, ...) -> None  # Line 319
    async def _coordinate_synchronization(self, state_change: 'StateChange') -> None  # Line 336
    def _get_transaction_lock(self, transaction_key: str) -> asyncio.Lock  # Line 348
    async def cleanup(self) -> None  # Line 354
    def _extract_config_dict(self, config: Config) -> dict[str, Any]  # Line 363
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import StateType`
- `from src.utils.decimal_utils import to_decimal`
- `from src.utils.messaging_patterns import MessagePattern`

#### Class: `StateDataTransformer`

**Purpose**: Handles consistent data transformation for state module

```python
class StateDataTransformer:
    def transform_state_change_to_event_data(state_type, ...) -> dict[str, Any]  # Line 23
    def transform_state_snapshot_to_event_data(snapshot_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 55
    def transform_error_to_event_data(error, ...) -> dict[str, Any]  # Line 82
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 104
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'state_management') -> dict[str, Any]  # Line 126
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 162
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]  # Line 212
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 237
    def apply_cross_module_validation(cls, ...) -> dict[str, Any]  # Line 301
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyContainer`
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.exceptions import DependencyError`
- `from src.core.exceptions import ServiceError`

#### Functions:

```python
def register_state_services(container: DependencyContainer) -> None  # Line 40
async def create_state_service_with_dependencies(...) -> 'StateService'  # Line 202
```

### File: environment_integration.py

**Key Imports:**
- `from src.core.exceptions import StateError`
- `from src.core.integration.environment_aware_service import EnvironmentAwareServiceMixin`
- `from src.core.integration.environment_aware_service import EnvironmentContext`
- `from src.core.logging import get_logger`
- `from src.core.types import StateType`

#### Class: `StateIsolationLevel`

**Inherits**: Enum
**Purpose**: State isolation levels for different environments

```python
class StateIsolationLevel(Enum):
```

#### Class: `StatePersistenceMode`

**Inherits**: Enum
**Purpose**: State persistence modes

```python
class StatePersistenceMode(Enum):
```

#### Class: `StateValidationLevel`

**Inherits**: Enum
**Purpose**: State validation levels

```python
class StateValidationLevel(Enum):
```

#### Class: `EnvironmentAwareStateConfiguration`

**Purpose**: Environment-specific state configuration

```python
class EnvironmentAwareStateConfiguration:
    def get_sandbox_state_config() -> dict[str, Any]  # Line 54
    def get_live_state_config() -> dict[str, Any]  # Line 82
```

#### Class: `EnvironmentAwareStateManager`

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware state management functionality

```python
class EnvironmentAwareStateManager(EnvironmentAwareServiceMixin):
    def __init__(self, *args: Any, **kwargs: Any) -> None  # Line 118
    async def _update_service_environment(self, context: EnvironmentContext) -> None  # Line 126
    def get_environment_state_config(self, exchange: str) -> dict[str, Any]  # Line 164
    async def set_environment_aware_state(self, ...) -> bool  # Line 177
    async def get_environment_aware_state(self, key: str, state_type: StateType, exchange: str, default: Any = None) -> Any  # Line 239
    async def delete_environment_aware_state(self, key: str, exchange: str) -> bool  # Line 288
    async def create_environment_state_checkpoint(self, exchange: str) -> dict[str, Any]  # Line 322
    async def rollback_to_checkpoint(self, checkpoint_id: str, exchange: str, confirm_rollback: bool = False) -> bool  # Line 369
    async def validate_environment_state_consistency(self, exchange: str) -> dict[str, Any]  # Line 423
    async def migrate_state_between_environments(self, ...) -> dict[str, Any]  # Line 473
    async def _validate_state_operation(self, operation: str, key: str, value: Any, exchange: str) -> bool  # Line 555
    async def _apply_environment_namespacing(self, key: str, exchange: str, state_config: dict[str, Any]) -> str  # Line 559
    async def _serialize_state_value(self, value: Any, state_config: dict[str, Any]) -> str  # Line 573
    async def _deserialize_state_value(self, serialized_value: str, state_config: dict[str, Any]) -> Any  # Line 577
    async def _get_next_state_version(self, key: str, exchange: str) -> int  # Line 581
    async def _encrypt_state_entry(self, state_entry: dict[str, Any], exchange: str) -> dict[str, Any]  # Line 585
    async def _decrypt_state_entry(self, state_entry: dict[str, Any], exchange: str) -> dict[str, Any]  # Line 591
    async def _store_state_entry(self, state_entry: dict[str, Any], exchange: str, state_config: dict[str, Any]) -> bool  # Line 598
    async def _retrieve_state_entry(self, key: str, exchange: str, state_config: dict[str, Any]) -> dict[str, Any] | None  # Line 608
    async def _delete_state_entry(self, key: str, exchange: str, state_config: dict[str, Any]) -> bool  # Line 614
    async def _setup_environment_state_components(self, exchange: str, state_config: dict[str, Any]) -> None  # Line 623
    def get_environment_state_metrics(self, exchange: str) -> dict[str, Any]  # Line 631
    async def _add_to_state_history(self, state_entry: dict[str, Any], exchange: str) -> None  # Line 668
    async def _update_state_metrics(self, exchange: str, start_time: datetime, success: bool, operation: str) -> None  # Line 674
    async def _should_create_checkpoint(self, exchange: str) -> bool  # Line 689
    async def _create_state_checkpoint(self, exchange: str) -> None  # Line 701
    async def _get_current_state_snapshot(self, exchange: str) -> dict[str, Any]  # Line 709
    async def _compress_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]  # Line 712
    async def _encrypt_checkpoint(self, checkpoint: dict[str, Any], exchange: str) -> dict[str, Any]  # Line 715
    async def _decrypt_checkpoint(self, checkpoint: dict[str, Any], exchange: str) -> dict[str, Any]  # Line 718
    async def _decompress_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]  # Line 721
    async def _store_checkpoint(self, checkpoint: dict[str, Any], exchange: str) -> None  # Line 724
    async def _retrieve_checkpoint(self, checkpoint_id: str, exchange: str) -> dict[str, Any] | None  # Line 729
    async def _validate_state_integrity(self, state_entry: dict[str, Any], exchange: str) -> bool  # Line 736
    async def _handle_corrupted_state(self, key: str, exchange: str) -> None  # Line 739
    async def _log_state_deletion(self, key: str, exchange: str) -> None  # Line 742
    async def _log_state_rollback(self, checkpoint_id: str, exchange: str, backup_id: str) -> None  # Line 745
    async def _restore_state_from_snapshot(self, snapshot: dict[str, Any], exchange: str) -> bool  # Line 748
    async def _perform_basic_consistency_checks(self, exchange: str) -> dict[str, Any]  # Line 752
    async def _perform_production_consistency_checks(self, exchange: str, state_config: dict[str, Any]) -> dict[str, Any]  # Line 755
    async def _get_all_state_keys(self, exchange: str) -> list[str]  # Line 760
```

### File: error_recovery.py

**Key Imports:**
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.logging import get_logger`
- `from src.utils.error_recovery_utilities import BaseErrorRecovery`
- `from src.utils.error_recovery_utilities import ErrorContext`
- `from src.utils.error_recovery_utilities import RecoveryCheckpoint`

#### Class: `StateErrorRecovery`

**Inherits**: BaseErrorRecovery
**Purpose**: State-specific error recovery system extending base recovery functionality

```python
class StateErrorRecovery(BaseErrorRecovery):
    def __init__(self, logger = None)  # Line 41
    async def create_recovery_checkpoint(self, operation: str, state_data: dict[str, Any] | None = None, **context) -> str  # Line 48
    async def handle_error(self, ...) -> StateErrorContext  # Line 100
    async def _attempt_recovery(self, ...) -> bool  # Line 179
    async def rollback_to_checkpoint(self, checkpoint_id: str, session: AsyncSession | None = None) -> bool  # Line 218
    async def _handle_database_connection_error(self, ...) -> bool  # Line 269
    async def _handle_database_integrity_error(self, ...) -> bool  # Line 284
    async def _handle_database_timeout_error(self, ...) -> bool  # Line 293
    async def _handle_redis_connection_error(self, ...) -> bool  # Line 304
    async def _handle_redis_timeout_error(self, ...) -> bool  # Line 315
    async def _handle_data_corruption_error(self, ...) -> bool  # Line 321
    async def _handle_disk_space_error(self, ...) -> bool  # Line 333
    async def _handle_permission_error(self, ...) -> bool  # Line 341
    async def _handle_validation_error(self, ...) -> bool  # Line 349
    async def _handle_concurrency_error(self, ...) -> bool  # Line 358
    async def _handle_unknown_error(self, ...) -> bool  # Line 372
    def get_error_statistics(self) -> dict[str, Any]  # Line 382
    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int  # Line 399
```

#### Functions:

```python
def with_error_recovery(...)  # Line 416
```

### File: factory.py

**Key Imports:**
- `from src.core.config.main import Config`
- `from src.core.dependency_injection import DependencyInjector`

#### Class: `DatabaseServiceWrapper`

**Purpose**: Wrapper to add compatibility for database service with StateService

```python
class DatabaseServiceWrapper:
    def __init__(self, database_service: Any)  # Line 54
    async def start(self) -> None  # Line 59
    async def stop(self) -> None  # Line 64
    def initialized(self) -> bool  # Line 70
    def __getattr__(self, name: str) -> Any  # Line 74
```

#### Class: `MockDatabaseService`

**Purpose**: Mock database service implementation for testing

```python
class MockDatabaseService:
    def __init__(self)  # Line 86
    def initialized(self) -> bool  # Line 93
    async def start(self) -> None  # Line 97
    async def stop(self) -> None  # Line 102
    async def create_entity(self, entity: Any) -> Any  # Line 108
    async def get_entity_by_id(self, model_class: type, entity_id: Any) -> Any | None  # Line 116
    async def health_check(self) -> dict[str, Any]  # Line 126
    def get_metrics(self) -> dict[str, Any]  # Line 134
```

#### Class: `StateServiceFactory`

**Inherits**: StateServiceFactoryInterface
**Purpose**: Factory for creating and configuring StateService instances

```python
class StateServiceFactory(StateServiceFactoryInterface):
    def __init__(self, injector: DependencyInjector | None = None)  # Line 147
    async def create_state_service(self, ...) -> 'StateService'  # Line 167
    async def create_state_service_for_testing(self, config: Config | None = None, mock_database: bool = False) -> 'StateService'  # Line 220
    def _create_test_config(self) -> Config  # Line 247
    def _create_mock_database_service(self) -> DatabaseServiceInterface  # Line 253
```

#### Class: `StateServiceRegistry`

**Purpose**: Registry for managing StateService instances across the application

```python
class StateServiceRegistry:
    async def get_instance(cls, ...) -> 'StateService'  # Line 272
    async def register_instance(cls, name: str, instance: 'StateService') -> None  # Line 311
    async def remove_instance(cls, name: str) -> None  # Line 323
    async def cleanup_all(cls) -> None  # Line 336
    def list_instances(cls) -> list[str]  # Line 344
    async def get_health_status(cls) -> dict[str, dict]  # Line 349
```

#### Functions:

```python
async def create_default_state_service(config: Config, injector: DependencyInjector | None = None) -> 'StateService'  # Line 364
async def get_state_service(name: str = 'default', injector: DependencyInjector | None = None) -> 'StateService'  # Line 378
async def create_test_state_service(injector: DependencyInjector | None = None) -> 'StateService'  # Line 391
```

### File: interfaces.py

#### Class: `StateBusinessServiceInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for state business services

```python
class StateBusinessServiceInterface(ABC):
    async def validate_state_change(self, ...) -> dict[str, Any]  # Line 44
    async def process_state_update(self, ...) -> 'StateChange'  # Line 56
    async def calculate_state_metadata(self, ...) -> 'StateMetadata'  # Line 68
```

#### Class: `StatePersistenceServiceInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for state persistence services

```python
class StatePersistenceServiceInterface(ABC):
    async def save_state(self, ...) -> bool  # Line 83
    async def load_state(self, state_type: 'StateType', state_id: str) -> dict[str, Any] | None  # Line 94
    async def delete_state(self, state_type: 'StateType', state_id: str) -> bool  # Line 99
    async def save_snapshot(self, snapshot: 'RuntimeStateSnapshot') -> bool  # Line 104
    async def load_snapshot(self, snapshot_id: str) -> 'RuntimeStateSnapshot | None'  # Line 109
```

#### Class: `StateValidationServiceInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for state validation services

```python
class StateValidationServiceInterface(ABC):
    async def validate_state_data(self, ...) -> dict[str, Any]  # Line 118
    async def validate_state_transition(self, ...) -> bool  # Line 128
    async def validate_business_rules(self, ...) -> list[str]  # Line 138
    def matches_criteria(self, state: dict[str, Any], criteria: dict[str, Any]) -> bool  # Line 148
```

#### Class: `StateSynchronizationServiceInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for state synchronization services

```python
class StateSynchronizationServiceInterface(ABC):
    async def synchronize_state_change(self, state_change: 'StateChange') -> bool  # Line 161
    async def broadcast_state_change(self, ...) -> None  # Line 166
    async def resolve_conflicts(self, ...) -> 'StateChange'  # Line 177
```

#### Class: `CheckpointServiceInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for checkpoint services

```python
class CheckpointServiceInterface(ABC):
    async def create_checkpoint(self, bot_id: str, state_data: dict[str, Any], checkpoint_type: str = 'manual') -> str  # Line 191
    async def restore_checkpoint(self, checkpoint_id: str) -> tuple[str, dict[str, Any]] | None  # Line 201
    async def list_checkpoints(self, bot_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]  # Line 206
    async def delete_checkpoint(self, checkpoint_id: str) -> bool  # Line 213
```

#### Class: `StateEventServiceInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for state event services

```python
class StateEventServiceInterface(ABC):
    async def emit_state_event(self, event_type: str, event_data: dict[str, Any]) -> None  # Line 222
    def subscribe_to_events(self, event_type: str, callback: Any) -> None  # Line 227
    def unsubscribe_from_events(self, event_type: str, callback: Any) -> None  # Line 232
```

#### Class: `StateServiceFactoryInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for state service factories

```python
class StateServiceFactoryInterface(ABC):
    async def create_state_service(self, ...) -> 'StateService'  # Line 241
    async def create_state_service_for_testing(self, config: Union['Config', None] = None, mock_database: bool = False) -> 'StateService'  # Line 251
```

#### Class: `MetricsStorageInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for metrics storage operations

```python
class MetricsStorageInterface(ABC):
    async def store_validation_metrics(self, validation_data: dict[str, Any]) -> bool  # Line 264
    async def store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool  # Line 269
    async def get_historical_metrics(self, metric_type: str, start_time: Any, end_time: Any) -> list[dict[str, Any]]  # Line 274
```

### File: monitoring.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.types import AlertSeverity`

#### Class: `MetricType`

**Inherits**: Enum
**Purpose**: Metric type enumeration for state metrics

```python
class MetricType(Enum):
```

#### Class: `HealthCheck`

**Purpose**: Health check definition

```python
class HealthCheck:
```

#### Class: `Metric`

**Purpose**: Metric data point

```python
class Metric:
```

#### Class: `Alert`

**Purpose**: Alert notification

```python
class Alert:
```

#### Class: `PerformanceReport`

**Purpose**: Performance analysis report

```python
class PerformanceReport:
```

#### Class: `StateMonitoringService`

**Inherits**: BaseComponent
**Purpose**: Comprehensive monitoring service for state management system with central integration

```python
class StateMonitoringService(BaseComponent):
    def __init__(self, state_service: Any, metrics_collector: Any = None)  # Line 187
    async def initialize(self) -> None  # Line 251
    async def cleanup(self) -> None  # Line 272
    def register_health_check(self, ...) -> str  # Line 339
    async def get_health_status(self) -> dict[str, Any]  # Line 376
    def record_metric(self, ...) -> None  # Line 443
    def record_operation_time(self, operation_name: str, duration_ms: float) -> None  # Line 493
    def get_metrics(self) -> dict[str, Any]  # Line 513
    async def get_filtered_metrics(self, ...) -> dict[str, list[Metric]]  # Line 517
    def set_alert_threshold(self, ...) -> None  # Line 566
    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None  # Line 596
    async def get_active_alerts(self) -> list[Alert]  # Line 600
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = '') -> bool  # Line 604
    async def generate_performance_report(self, start_time: datetime | None = None, end_time: datetime | None = None) -> PerformanceReport  # Line 621
    def _initialize_builtin_health_checks(self) -> None  # Line 671
    def _initialize_builtin_metrics(self) -> None  # Line 711
    async def _check_state_service_connectivity(self) -> dict[str, Any]  # Line 731
    async def _check_database_connectivity(self) -> dict[str, Any]  # Line 755
    async def _check_cache_connectivity(self) -> dict[str, Any]  # Line 786
    async def _check_memory_usage(self) -> dict[str, Any]  # Line 817
    async def _check_error_rate(self) -> dict[str, Any]  # Line 838
    async def _run_all_health_checks(self) -> None  # Line 861
    async def _run_health_check(self, check: HealthCheck) -> None  # Line 878
    def _update_metric_aggregates(self, name: str, value: float) -> None  # Line 952
    async def _check_metric_alerts(self, metric_name: str, value: float) -> None  # Line 971
    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool  # Line 1014
    async def _create_alert(self, ...) -> None  # Line 1025
    async def _get_key_metrics(self) -> dict[str, Any]  # Line 1076
    async def _calculate_uptime(self, start_time: datetime, end_time: datetime) -> float  # Line 1116
    async def _calculate_performance_metrics(self, report: PerformanceReport, start_time: datetime, end_time: datetime) -> None  # Line 1137
    async def _calculate_resource_metrics(self, report: PerformanceReport, start_time: datetime, end_time: datetime) -> None  # Line 1166
    async def _calculate_state_metrics(self, report: PerformanceReport, start_time: datetime, end_time: datetime) -> None  # Line 1182
    async def _calculate_alert_metrics(self, report: PerformanceReport, start_time: datetime, end_time: datetime) -> None  # Line 1203
    async def _health_check_loop(self) -> None  # Line 1230
    async def _metrics_collection_loop(self) -> None  # Line 1241
    async def _alert_processing_loop(self) -> None  # Line 1257
    async def _cleanup_loop(self) -> None  # Line 1273
    async def _collect_system_metrics(self) -> None  # Line 1286
    async def _collect_state_service_metrics(self) -> None  # Line 1303
    async def _auto_resolve_alerts(self) -> None  # Line 1321
    async def _cleanup_old_alerts(self) -> None  # Line 1356
    async def _cleanup_old_metrics(self) -> None  # Line 1372
    async def _heartbeat_loop(self) -> None  # Line 1408
    async def _send_heartbeat(self) -> None  # Line 1422
    def _record_to_central_monitoring(self, metric: Metric) -> None  # Line 1452
    async def _send_alert_to_central(self, alert: Alert) -> None  # Line 1489
    def _setup_central_integration(self) -> None  # Line 1541
    async def _forward_alert_handler(self, alert: Alert) -> None  # Line 1546
    async def _forward_alert_to_central(self, alert: Alert) -> None  # Line 1553
```

### File: monitoring_integration.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import StateConsistencyError`
- `from src.error_handling import ErrorContext`
- `from src.monitoring import MetricsCollector`
- `from src.monitoring.alerting import Alert`

#### Class: `StateMetricsAdapter`

**Inherits**: BaseComponent
**Purpose**: Adapter to bridge state monitoring with central metrics collection

```python
class StateMetricsAdapter(BaseComponent):
    def __init__(self, metrics_collector: MetricsCollector | None = None)  # Line 41
    def record_state_metric(self, metric: Metric) -> None  # Line 57
    def record_operation_time(self, operation: str, duration_ms: float) -> None  # Line 112
    def record_health_check(self, check_name: str, status: HealthStatus) -> None  # Line 118
```

#### Class: `StateAlertAdapter`

**Inherits**: BaseComponent
**Purpose**: Adapter to integrate state alerts with central alerting system

```python
class StateAlertAdapter(BaseComponent):
    def __init__(self)  # Line 139
    def alert_manager(self)  # Line 158
    async def send_alert(self, alert: Alert) -> None  # Line 164
```

#### Functions:

```python
def create_integrated_monitoring_service(state_service: Any, metrics_collector: MetricsCollector | None = None) -> StateMonitoringService  # Line 265
```

### File: quality_controller.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import ExecutionResult`

#### Class: `QualityLevel`

**Inherits**: Enum
**Purpose**: Quality assessment levels

```python
class QualityLevel(Enum):
```

#### Class: `QualityTrend`

**Purpose**: Quality trend analysis

```python
class QualityTrend:
```

#### Class: `InfluxDBMetricsStorage`

**Inherits**: MetricsStorageInterface
**Purpose**: InfluxDB implementation of MetricsStorage interface

```python
class InfluxDBMetricsStorage(MetricsStorageInterface):
    def __init__(self, config: Config | None = None)  # Line 108
    async def close(self) -> None  # Line 123
    async def store_validation_metrics(self, validation_data: dict[str, Any]) -> bool  # Line 159
    async def store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool  # Line 195
    async def get_historical_metrics(self, metric_type: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]  # Line 242
```

#### Class: `NullMetricsStorage`

**Inherits**: MetricsStorageInterface
**Purpose**: Null implementation of MetricsStorage for testing or when metrics storage is disabled

```python
class NullMetricsStorage(MetricsStorageInterface):
    async def store_validation_metrics(self, validation_data: dict[str, Any]) -> bool  # Line 263
    async def store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool  # Line 267
    async def get_historical_metrics(self, metric_type: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]  # Line 271
```

#### Class: `QualityController`

**Inherits**: BaseComponent
**Purpose**: Quality control controller that coordinates quality management operations

```python
class QualityController(BaseComponent):
    def __init__(self, ...)  # Line 292
    async def initialize(self) -> None  # Line 401
    async def validate_pre_trade(self, ...) -> PreTradeValidation  # Line 422
    async def analyze_post_trade(self, ...) -> PostTradeAnalysis  # Line 486
    async def get_quality_summary(self, bot_id: str | None = None, hours: int = 24) -> dict[str, Any]  # Line 546
    async def get_quality_trend_analysis(self, metric: str, days: int = 7) -> QualityTrend  # Line 584
    def _summarize_validations(self, validations: list[PreTradeValidation]) -> dict[str, Any]  # Line 675
    def _summarize_analyses(self, analyses: list[PostTradeAnalysis]) -> dict[str, Any]  # Line 697
    async def _get_quality_trends(self, hours: int) -> list[dict[str, Any]]  # Line 721
    async def _get_quality_alerts(self) -> list[dict[str, Any]]  # Line 739
    async def _get_improvement_recommendations(self) -> list[str]  # Line 773
    def _update_quality_metrics(self, operation: str, data: Any) -> None  # Line 804
    def _check_trend_alerts(self, metric: str, trend: QualityTrend) -> tuple[bool, str]  # Line 835
    async def _load_benchmarks(self) -> None  # Line 862
    async def _log_validation_metrics(self, validation: PreTradeValidation) -> None  # Line 874
    async def _log_analysis_metrics(self, analysis: PostTradeAnalysis) -> None  # Line 898
    async def _quality_monitoring_loop(self) -> None  # Line 925
    async def _trend_analysis_loop(self) -> None  # Line 936
    def get_quality_metrics(self) -> dict[str, Any]  # Line 947
    async def get_summary_statistics(self, hours: int = 24, bot_id: str | None = None) -> dict[str, Any]  # Line 968
    def _calculate_avg_validation_time(self) -> Decimal  # Line 1025
    def _calculate_avg_analysis_time(self) -> Decimal  # Line 1034
    async def validate_state_consistency(self, state: Any) -> bool  # Line 1042
    async def validate_portfolio_balance(self, portfolio_state: Any) -> bool  # Line 1058
    async def validate_position_consistency(self, position: Any, related_orders: list) -> bool  # Line 1074
    async def run_integrity_checks(self, state: Any) -> dict[str, Any]  # Line 1091
    async def suggest_corrections(self, state: Any) -> list[dict[str, Any]]  # Line 1128
    async def cleanup(self) -> None  # Line 1181
    def add_validation_rule(self, name: str, rule: Callable[Ellipsis, Any]) -> None  # Line 1194
```

### File: recovery.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import ErrorSeverity`
- `from src.core.exceptions import StateConsistencyError`
- `from src.error_handling import ErrorContext`
- `from src.error_handling import with_circuit_breaker`

#### Class: `RecoveryStatus`

**Inherits**: Enum
**Purpose**: Recovery operation status

```python
class RecoveryStatus(Enum):
```

#### Class: `AuditEntry`

**Purpose**: Audit trail entry for state changes

```python
class AuditEntry:
```

#### Class: `RecoveryPoint`

**Purpose**: Point-in-time recovery information

```python
class RecoveryPoint:
```

#### Class: `RecoveryOperation`

**Purpose**: Recovery operation tracking

```python
class RecoveryOperation:
```

#### Class: `CorruptionReport`

**Purpose**: State corruption detection report

```python
class CorruptionReport:
```

#### Class: `StateRecoveryManager`

**Inherits**: BaseComponent
**Purpose**: Enterprise-grade state recovery and audit trail manager

```python
class StateRecoveryManager(BaseComponent):
    def __init__(self, state_service: Any)  # Line 170
    async def initialize(self) -> None  # Line 215
    async def cleanup(self) -> None  # Line 242
    async def record_state_change(self, ...) -> str  # Line 309
    async def get_audit_trail(self, ...) -> list[AuditEntry]  # Line 373
    async def create_recovery_point(self, description: str = '') -> str  # Line 432
    async def list_recovery_points(self, ...) -> list[RecoveryPoint]  # Line 480
    async def recover_to_point(self, ...) -> str  # Line 521
    async def get_recovery_status(self, operation_id: str) -> RecoveryOperation | None  # Line 585
    async def detect_corruption(self, state_type: str | None = None, state_id: str | None = None) -> list[CorruptionReport]  # Line 591
    async def repair_corruption(self, report_id: str, repair_method: str = 'auto') -> bool  # Line 644
    def _detect_changed_fields(self, old_value: dict[str, Any] | None, new_value: dict[str, Any] | None) -> set[str]  # Line 693
    async def _capture_state_snapshot(self, recovery_point: RecoveryPoint) -> None  # Line 718
    def _calculate_consistency_hash(self, recovery_point: RecoveryPoint) -> str  # Line 750
    async def _validate_recovery_point(self, recovery_point: RecoveryPoint) -> bool  # Line 765
    async def _execute_recovery(self, ...) -> None  # Line 786
    async def _check_state_corruption(self, state_type: str, state_id: str) -> CorruptionReport | None  # Line 869
    async def _get_states_by_type(self, state_type: str) -> list[tuple[str, str]]  # Line 908
    async def _get_all_states(self) -> list[tuple[str, str]]  # Line 914
    async def _auto_repair_corruption(self, report: CorruptionReport) -> bool  # Line 920
    async def _rollback_repair_corruption(self, report: CorruptionReport) -> bool  # Line 925
    async def _audit_cleanup_loop(self) -> None  # Line 932
    async def _auto_recovery_loop(self) -> None  # Line 950
    async def _corruption_monitor_loop(self) -> None  # Line 963
```

### File: quality_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.types import ExecutionResult`
- `from src.core.types import MarketData`
- `from src.core.types import OrderRequest`

#### Class: `QualityService`

**Inherits**: BaseService
**Purpose**: Quality service implementing core quality control business logic

```python
class QualityService(BaseService):
    def __init__(self, config: Any = None)  # Line 51
    async def validate_pre_trade(self, ...) -> 'PreTradeValidation'  # Line 82
    async def analyze_post_trade(self, ...) -> 'PostTradeAnalysis'  # Line 112
    async def validate_state_consistency(self, state: Any) -> bool  # Line 148
    async def validate_portfolio_balance(self, portfolio_state: Any) -> bool  # Line 182
    async def validate_position_consistency(self, position: Any, related_orders: list) -> bool  # Line 219
```

### File: state_business_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.exceptions import ValidationError`

#### Class: `StateBusinessService`

**Inherits**: BaseService
**Purpose**: State business service implementing core state management business logic

```python
class StateBusinessService(BaseService):
    def __init__(self, config: Any = None)  # Line 67
    async def validate_state_change(self, ...) -> dict[str, Any]  # Line 89
    async def process_state_update(self, ...) -> 'StateChange'  # Line 156
    async def calculate_state_metadata(self, ...) -> 'StateMetadata'  # Line 231
    async def validate_business_rules(self, state_type: 'StateType', state_data: dict[str, Any], operation: str) -> list[str]  # Line 269
    async def _validate_state_transition(self, ...) -> list[str]  # Line 318
    async def _validate_critical_state_constraints(self, state_type: 'StateType', state_data: dict[str, Any]) -> list[str]  # Line 342
    async def _validate_critical_priority_constraints(self, state_type: 'StateType', state_data: dict[str, Any]) -> list[str]  # Line 377
    async def _apply_business_transformations(self, state_change: 'StateChange') -> 'StateChange'  # Line 400
    async def _apply_financial_transformations(self, state_data: dict[str, Any]) -> None  # Line 429
    def _generate_state_tags(self, state_type: 'StateType', state_data: dict[str, Any]) -> dict[str, str]  # Line 461
    def _extract_status_field(self, state_data: dict[str, Any], state_type: 'StateType') -> str | None  # Line 480
    async def _is_valid_transition(self, state_type: 'StateType', current_status: str, new_status: str) -> bool  # Line 501
    async def _validate_bot_state_rules(self, state_data: dict[str, Any], operation: str) -> list[str]  # Line 532
    async def _validate_position_state_rules(self, state_data: dict[str, Any], operation: str) -> list[str]  # Line 560
    async def _validate_risk_state_rules(self, state_data: dict[str, Any], operation: str) -> list[str]  # Line 593
    async def _validate_order_state_rules(self, state_data: dict[str, Any], operation: str) -> list[str]  # Line 632
    async def _validate_general_business_rules(self, state_data: dict[str, Any], operation: str) -> list[str]  # Line 661
    async def _transform_bot_state(self, state_change: 'StateChange') -> None  # Line 692
    async def _transform_position_state(self, state_change: 'StateChange') -> None  # Line 709
```

### File: state_persistence_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import DatabaseError`
- `from src.core.exceptions import StateConsistencyError`

#### Class: `StatePersistenceService`

**Inherits**: BaseService
**Purpose**: State persistence service providing database-agnostic state storage

```python
class StatePersistenceService(BaseService):
    def __init__(self, database_service: Any = None)  # Line 59
    def _get_database_service(self)  # Line 90
    def database_service(self)  # Line 106
    async def start(self) -> None  # Line 110
    async def stop(self) -> None  # Line 125
    async def save_state(self, ...) -> bool  # Line 158
    async def load_state(self, state_type: 'StateType', state_id: str) -> dict[str, Any] | None  # Line 234
    async def delete_state(self, state_type: 'StateType', state_id: str) -> bool  # Line 280
    async def list_states(self, state_type: 'StateType', limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]  # Line 321
    async def save_snapshot(self, snapshot: 'RuntimeStateSnapshot') -> bool  # Line 388
    async def load_snapshot(self, snapshot_id: str) -> 'RuntimeStateSnapshot | None'  # Line 433
    async def queue_save_operation(self, ...) -> None  # Line 470
    async def queue_delete_operation(self, state_type: 'StateType', state_id: str) -> None  # Line 490
    async def _process_operations(self) -> None  # Line 504
    async def _handle_persistence_event(self, event_data: dict[str, Any]) -> None  # Line 528
    async def _flush_events(self) -> None  # Line 555
    def is_available(self) -> bool  # Line 569
```

### File: state_synchronization_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import StateConsistencyError`
- `from src.utils.messaging_patterns import ErrorPropagationMixin`
- `from src.core.types import StateType`

#### Class: `StateSynchronizationService`

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: State synchronization service providing distributed state consistency

```python
class StateSynchronizationService(BaseService, ErrorPropagationMixin):
    def __init__(self, event_service: Any = None)  # Line 53
    async def synchronize_state_change(self, state_change: 'StateChange') -> bool  # Line 95
    async def broadcast_state_change(self, ...) -> None  # Line 202
    async def resolve_conflicts(self, ...) -> 'StateChange'  # Line 260
    def subscribe_to_state_changes(self, state_type: StateType, callback: Callable) -> None  # Line 301
    def unsubscribe_from_state_changes(self, state_type: StateType, callback: Callable) -> None  # Line 315
    def get_synchronization_metrics(self) -> dict[str, Any]  # Line 327
    async def _detect_conflicts(self, state_change: 'StateChange') -> bool  # Line 344
    async def _resolve_conflict(self, state_change: 'StateChange') -> 'StateChange | None'  # Line 368
    async def _perform_synchronization(self, state_change: 'StateChange') -> bool  # Line 389
    async def _validate_change(self, state_change: 'StateChange') -> None  # Line 407
    async def _apply_change(self, state_change: 'StateChange') -> None  # Line 416
    async def _confirm_sync(self, state_change: 'StateChange') -> None  # Line 425
    async def _notify_subscribers(self, state_change: 'StateChange') -> None  # Line 434
    async def _send_to_subscribers(self, subscription_key: str, event_data: dict[str, Any]) -> None  # Line 458
    async def _apply_conflict_resolution_strategy(self, ...) -> 'StateChange'  # Line 477
    async def _create_conflict_audit_record(self, ...) -> None  # Line 536
    async def cleanup_expired_locks(self) -> None  # Line 563
```

### File: state_validation_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.utils.messaging_patterns import ErrorPropagationMixin`

#### Class: `StateValidationService`

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: State validation service providing comprehensive validation capabilities

```python
class StateValidationService(BaseService, ErrorPropagationMixin):
    def __init__(self, validation_service: ValidationService | None = None)  # Line 63
    async def validate_state_data(self, ...) -> dict[str, Any]  # Line 103
    async def validate_state_transition(self, ...) -> bool  # Line 200
    async def validate_business_rules(self, ...) -> list[str]  # Line 248
    def matches_criteria(self, state: dict[str, Any], criteria: dict[str, Any]) -> bool  # Line 294
    def get_validation_metrics(self) -> dict[str, Any]  # Line 335
    async def _validate_basic_data_structure(self, state_type: 'StateType', state_data: dict[str, Any]) -> list[str]  # Line 348
    async def _validate_state_type_specific(self, state_type: 'StateType', state_data: dict[str, Any]) -> list[str]  # Line 377
    async def _validate_strict_requirements(self, state_type: 'StateType', state_data: dict[str, Any]) -> list[str]  # Line 398
    def _get_required_fields(self, state_type: 'StateType') -> list[str]  # Line 431
    async def _validate_field_types(self, state_type: 'StateType', state_data: dict[str, Any]) -> list[str]  # Line 445
    async def _validate_bot_state_structure(self, state_data: dict[str, Any]) -> list[str]  # Line 492
    async def _validate_position_state_structure(self, state_data: dict[str, Any]) -> list[str]  # Line 511
    async def _validate_order_state_structure(self, state_data: dict[str, Any]) -> list[str]  # Line 533
    async def _validate_risk_state_structure(self, state_data: dict[str, Any]) -> list[str]  # Line 562
    async def _validate_bot_business_rules(self, state_data: dict[str, Any]) -> list[str]  # Line 582
    async def _validate_position_business_rules(self, state_data: dict[str, Any]) -> list[str]  # Line 604
    async def _validate_order_business_rules(self, state_data: dict[str, Any]) -> list[str]  # Line 621
    async def _validate_risk_business_rules(self, state_data: dict[str, Any]) -> list[str]  # Line 633
    async def _validate_general_business_rules(self, state_data: dict[str, Any]) -> list[str]  # Line 653
    def _extract_status_field(self, state_data: dict[str, Any], state_type: 'StateType') -> str | None  # Line 676
    def _get_valid_transitions(self, state_type: 'StateType') -> dict[str, set[str]]  # Line 696
    async def _get_cached_validation(self, state_type: 'StateType', state_data: dict[str, Any], validation_level: str) -> dict[str, Any] | None  # Line 720
    async def _cache_validation_result(self, ...) -> None  # Line 738
    def _generate_cache_key(self, state_type: 'StateType', state_data: dict[str, Any], validation_level: str) -> str  # Line 756
```

### File: trade_lifecycle_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import OrderSide`
- `from src.core.types import OrderType`

#### Class: `TradeLifecycleState`

**Inherits**: Enum
**Purpose**: Trade lifecycle state enumeration

```python
class TradeLifecycleState(Enum):
```

#### Class: `TradeContext`

**Purpose**: Complete context for a trade throughout its lifecycle

```python
class TradeContext:
```

#### Class: `TradeHistoryRecord`

**Purpose**: Historical trade record for analysis

```python
class TradeHistoryRecord:
```

#### Class: `TradeLifecycleService`

**Inherits**: BaseService
**Purpose**: Trade lifecycle service implementing core trade lifecycle business logic

```python
class TradeLifecycleService(BaseService):
    def __init__(self, config: Any = None)  # Line 128
    async def create_trade_context(self, ...) -> TradeContext  # Line 168
    async def validate_trade_transition(self, current_state: TradeLifecycleState, new_state: TradeLifecycleState) -> bool  # Line 238
    async def calculate_trade_performance(self, context: TradeContext) -> dict[str, Any]  # Line 266
    async def create_history_record(self, context: TradeContext) -> TradeHistoryRecord  # Line 328
    async def apply_business_rules(self, context: TradeContext) -> list[str]  # Line 390
```

### File: state_manager.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
- `from src.core.types import BotState`

#### Class: `StateManager`

**Inherits**: BaseComponent
**Purpose**: Backward compatibility wrapper for StateService

```python
class StateManager(BaseComponent):
    def __init__(self, config: Config)  # Line 34
    async def initialize(self) -> None  # Line 41
    async def shutdown(self) -> None  # Line 60
    async def save_bot_state(self, bot_id: str, state: dict[str, Any], create_snapshot: bool = False) -> str  # Line 65
    async def load_bot_state(self, bot_id: str) -> BotState | None  # Line 95
    async def create_checkpoint(self, bot_id: str, checkpoint_data: dict[str, Any] | None = None) -> str  # Line 148
    async def restore_from_checkpoint(self, bot_id: str, checkpoint_id: str) -> bool  # Line 158
    async def get_state_metrics(self, bot_id: str | None = None, hours: int = 24) -> dict[str, Any]  # Line 166
    def __getattr__(self, name: str) -> Any  # Line 221
```

#### Functions:

```python
def get_cache_manager()  # Line 19
```

### File: state_persistence.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import StateConsistencyError`

#### Class: `StatePersistence`

**Inherits**: BaseComponent
**Purpose**: Handles state persistence operations for the StateService

```python
class StatePersistence(BaseComponent):
    def __init__(self, state_service: 'StateService')  # Line 31
    def database_service(self)  # Line 55
    async def initialize(self) -> None  # Line 59
    async def cleanup(self) -> None  # Line 86
    async def load_state(self, state_type: 'StateType', state_id: str) -> dict[str, Any] | None  # Line 119
    async def save_state(self, ...) -> bool  # Line 142
    async def delete_state(self, state_type: 'StateType', state_id: str) -> bool  # Line 175
    async def queue_state_save(self, ...) -> None  # Line 198
    async def queue_state_delete(self, state_type: 'StateType', state_id: str) -> None  # Line 215
    async def get_states_by_type(self, ...) -> list[dict[str, Any]]  # Line 224
    async def search_states(self, ...) -> list[dict[str, Any]]  # Line 257
    async def save_snapshot(self, snapshot: 'RuntimeStateSnapshot') -> bool  # Line 305
    async def load_snapshot(self, snapshot_id: str) -> 'RuntimeStateSnapshot | None'  # Line 327
    async def load_all_states_to_cache(self) -> None  # Line 349
    async def _persistence_loop(self) -> None  # Line 370
    async def _process_save_batch(self, batch: list[dict[str, Any]]) -> None  # Line 450
    async def _process_delete_batch(self, batch: list[dict[str, Any]]) -> None  # Line 488
    async def _flush_queues(self) -> None  # Line 521
    def _matches_criteria(self, state_data: dict[str, Any], criteria: dict[str, Any]) -> bool  # Line 549
    def _is_service_available(self) -> bool  # Line 556
    def _is_database_available(self) -> bool  # Line 573
```

### File: state_service.py

**Key Imports:**
- `from src.core.base.events import BaseEventEmitter`
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.service import BaseService`
- `from src.core.config.main import Config`
- `from src.core.exceptions import DependencyError`

#### Class: `StateOperation`

**Inherits**: Enum
**Purpose**: State operation enumeration

```python
class StateOperation(Enum):
```

#### Class: `StateChange`

**Purpose**: Represents a state change for audit and synchronization

```python
class StateChange:
```

#### Class: `RuntimeStateSnapshot`

**Purpose**: Runtime state snapshot data structure for in-memory operations

```python
class RuntimeStateSnapshot:
```

#### Class: `StateValidationResult`

**Purpose**: Result of state validation operation

```python
class StateValidationResult:
```

#### Class: `StateMetrics`

**Purpose**: State management performance and health metrics

```python
class StateMetrics:
    def to_dict(self) -> dict[str, int | float | str | None]  # Line 199
```

#### Class: `StateService`

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: Comprehensive state management service providing enterprise-grade
state handling with synchronizatio

```python
class StateService(BaseService, ErrorPropagationMixin):
    def __init__(self, ...)  # Line 237
    async def initialize(self) -> None  # Line 399
    async def cleanup(self) -> None  # Line 449
    async def get_state(self, state_type: StateType, state_id: str, include_metadata: bool = False) -> dict[str, Any] | None  # Line 530
    async def set_state(self, ...) -> bool  # Line 689
    async def delete_state(self, ...) -> bool  # Line 813
    async def get_states_by_type(self, ...) -> list[dict[str, Any]]  # Line 887
    async def search_states(self, ...) -> list[dict[str, Any]]  # Line 931
    async def create_snapshot(self, description: str = '', state_types: list[StateType] | None = None) -> str  # Line 969
    async def restore_snapshot(self, snapshot_id: str) -> bool  # Line 1014
    def subscribe(self, ...) -> None  # Line 1057
    def unsubscribe(self, state_type: StateType, callback: Callable) -> None  # Line 1075
    def get_metrics(self) -> dict[str, int | float | str]  # Line 1090
    async def get_state_metrics(self) -> StateMetrics  # Line 1108
    async def get_health_status(self) -> dict[str, Any]  # Line 1116
    async def _check_rate_limit(self, cache_key: str) -> bool  # Line 1171
    def _get_state_lock(self, state_key: str) -> asyncio.Lock  # Line 1199
    def _detect_changed_fields(self, old_state: dict[str, Any] | None, new_state: dict[str, Any]) -> set[str]  # Line 1222
    def _get_next_version(self, cache_key: str) -> int  # Line 1243
    def _update_hit_rate(self, hit: bool) -> float  # Line 1248
    def _update_operation_metrics(self, operation_time_ms: float, success: bool) -> None  # Line 1260
    def _calculate_memory_usage(self) -> float  # Line 1282
    def _matches_criteria(self, state: dict[str, Any], criteria: dict[str, Any]) -> bool  # Line 1305
    async def _load_metadata_through_service(self, cache_key: str, state_type: StateType, state_id: str) -> StateMetadata | None  # Line 1322
    async def _load_existing_states(self) -> None  # Line 1347
    async def _broadcast_state_change(self, ...) -> None  # Line 1357
    async def _notify_legacy_subscribers(self, ...) -> None  # Line 1468
    async def _synchronization_loop(self) -> None  # Line 1490
    async def _cleanup_loop(self) -> None  # Line 1503
    async def _metrics_loop(self) -> None  # Line 1524
    async def _backup_loop(self) -> None  # Line 1548
    async def _initialize_service_components(self) -> None  # Line 1578
    async def _initialize_state_components(self) -> None  # Line 1642
    def error_handler(self) -> ErrorHandler  # Line 1686
    async def _store_state_through_services(self, ...) -> None  # Line 1702
    async def _coordinate_post_storage_activities(self, ...) -> None  # Line 1731
    def _resolve_service_dependency(self, service_name: str, fallback_factory)  # Line 1769
    def _create_business_service_fallback(self)  # Line 1779
    def _create_persistence_service_fallback(self)  # Line 1783
    def _create_validation_service_fallback(self)  # Line 1787
    def _create_synchronization_service_fallback(self)  # Line 1791
    def _extract_config_dict(self, config: Config) -> dict[str, Any]  # Line 1797
```

### File: state_sync_manager.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`

#### Class: `SyncEventType`

**Inherits**: Enum
**Purpose**: Sync event types for backward compatibility

```python
class SyncEventType(Enum):
```

#### Class: `StateSyncManager`

**Inherits**: BaseComponent
**Purpose**: Backward compatibility wrapper for StateSynchronizer

```python
class StateSyncManager(BaseComponent):
    def __init__(self, ...)  # Line 38
    async def initialize(self) -> None  # Line 66
    async def shutdown(self) -> None  # Line 71
    async def sync_state(self, ...) -> bool  # Line 76
    async def force_sync(self, *args) -> dict[str, Any]  # Line 127
    async def get_sync_status(self, *args) -> dict[str, Any]  # Line 160
    async def get_sync_metrics(self, hours: int = 24) -> dict[str, Any]  # Line 190
    async def subscribe_to_events(self, event_type: str, callback: Any) -> str  # Line 209
    async def register_conflict_resolver(self, state_type: str, resolver: Any) -> None  # Line 219
    async def _load_from_primary_storage(self, entity_type: str, entity_id: str) -> dict[str, Any]  # Line 223
    async def _check_consistency(self, entity_type: str, entity_id: str) -> dict[str, Any]  # Line 228
    def event_subscribers(self)  # Line 234
    def custom_resolvers(self)  # Line 239
    def __getattr__(self, name: str) -> Any  # Line 243
```

### File: state_synchronizer.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import StateConsistencyError`

#### Class: `StateSynchronizer`

**Inherits**: BaseComponent
**Purpose**: Handles state synchronization across components and services

```python
class StateSynchronizer(BaseComponent):
    def __init__(self, state_service: 'StateService')  # Line 30
    async def initialize(self) -> None  # Line 63
    async def cleanup(self) -> None  # Line 87
    async def queue_state_sync(self, state_change: 'StateChange') -> None  # Line 127
    async def sync_pending_changes(self) -> bool  # Line 159
    async def _execute_sync_operation(self) -> bool  # Line 190
    async def _sync_state_change_with_error_handling(self, change) -> bool  # Line 292
    async def _sync_legacy_changes(self) -> bool  # Line 300
    async def force_sync(self) -> bool  # Line 368
    async def get_sync_status(self) -> dict[str, Any]  # Line 389
    async def sync_with_remotes(self, remotes: list[str]) -> bool  # Line 408
    async def check_consistency(self, state_data: dict) -> bool  # Line 424
    async def synchronize_state(self, state_change) -> bool  # Line 430
    async def rollback_sync(self, state_change) -> bool  # Line 436
    async def synchronize_with_priority(self, state_change) -> bool  # Line 442
    async def send_heartbeat(self, heartbeat_data: dict) -> bool  # Line 448
    async def cleanup_stale_data(self, max_age_hours: int) -> int  # Line 454
    async def _sync_state_change(self, change: 'StateChange') -> bool  # Line 462
    async def _broadcast_change(self, change: 'StateChange') -> None  # Line 495
    async def _update_dependent_states(self, change: 'StateChange') -> None  # Line 510
    async def _detect_conflicts(self, change: 'StateChange') -> bool  # Line 516
    async def _resolve_conflicts(self, change: 'StateChange') -> None  # Line 530
    async def _sync_loop(self) -> None  # Line 542
```

### File: state_validator.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import BotStatus`
- `from src.core.types import OrderSide`
- `from src.core.types import OrderType`

#### Class: `ValidationLevel`

**Inherits**: Enum
**Purpose**: Validation level enumeration

```python
class ValidationLevel(Enum):
```

#### Class: `ValidationRule`

**Inherits**: Enum
**Purpose**: Validation rule types

```python
class ValidationRule(Enum):
```

#### Class: `ValidationRuleConfig`

**Purpose**: Configuration for a validation rule

```python
class ValidationRuleConfig:
```

#### Class: `StateValidationError`

**Purpose**: Individual validation error

```python
class StateValidationError:
```

#### Class: `ValidationWarning`

**Purpose**: Individual validation warning

```python
class ValidationWarning:
```

#### Class: `ValidationResult`

**Purpose**: Complete validation result

```python
class ValidationResult:
```

#### Class: `ValidationMetrics`

**Purpose**: Validation performance metrics

```python
class ValidationMetrics:
```

#### Class: `StateValidator`

**Inherits**: BaseComponent
**Purpose**: State validation controller that delegates to StateValidationService

```python
class StateValidator(BaseComponent):
    def __init__(self, state_service: 'StateService')  # Line 133
    async def _do_start(self) -> None  # Line 169
    async def _do_stop(self) -> None  # Line 191
    async def validate_state(self, ...) -> ValidationResult  # Line 214
    async def validate_state_transition(self, ...) -> bool  # Line 303
    async def validate_cross_state_consistency(self, ...) -> ValidationResult  # Line 332
    def add_validation_rule(self, state_type: 'StateType', rule_config: ValidationRuleConfig) -> None  # Line 384
    def remove_validation_rule(self, state_type: 'StateType', field_name: str, rule_type: ValidationRule) -> bool  # Line 394
    def update_validation_level(self, level: ValidationLevel) -> None  # Line 407
    def get_metrics(self) -> dict[str, int | float | str]  # Line 417
    async def get_validation_metrics(self) -> ValidationMetrics  # Line 432
    def get_validation_rules(self, state_type: 'StateType') -> list[ValidationRuleConfig]  # Line 441
    def _initialize_builtin_rules(self) -> None  # Line 447
    def _add_bot_state_rules(self) -> None  # Line 474
    def _add_position_state_rules(self) -> None  # Line 519
    def _add_order_state_rules(self) -> None  # Line 575
    def _add_portfolio_state_rules(self) -> None  # Line 631
    def _add_risk_state_rules(self) -> None  # Line 670
    def _add_strategy_state_rules(self) -> None  # Line 709
    def _add_market_state_rules(self) -> None  # Line 735
    def _add_trade_state_rules(self) -> None  # Line 779
    def _initialize_transition_rules(self) -> None  # Line 805
    async def _apply_validation_rule(self, ...) -> dict[str, Any]  # Line 838
    def _validate_required_field(self, data: dict[str, Any], field_name: str | None = None) -> bool  # Line 864
    def _validate_string_field(self, data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 876
    def _validate_decimal_field(self, data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 888
    def _validate_positive_value(self, data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 916
    def _validate_non_negative_value(self, data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 938
    def _validate_list_field(self, data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 960
    def _validate_dict_field(self, data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 972
    def _validate_bot_id_format(self, bot_id: str) -> dict[str, Any]  # Line 986
    def _validate_bot_status(self, status: Any) -> dict[str, Any]  # Line 1002
    def _validate_order_side(self, side: Any) -> dict[str, Any]  # Line 1024
    def _validate_order_type(self, order_type: Any) -> dict[str, Any]  # Line 1046
    def _validate_symbol_format(self, symbol: str) -> dict[str, Any]  # Line 1068
    def _validate_capital_allocation(self, data: dict[str, Any]) -> dict[str, Any]  # Line 1073
    def _validate_order_price_logic(self, data: dict[str, Any]) -> dict[str, Any]  # Line 1078
    def _validate_cash_balance(self, data: dict[str, Any]) -> dict[str, Any]  # Line 1083
    def _validate_var_limits(self, data: dict[str, Any]) -> dict[str, Any]  # Line 1088
    def _validate_strategy_params(self, data: dict[str, Any]) -> dict[str, Any]  # Line 1093
    def _validate_trade_execution(self, data: dict[str, Any]) -> dict[str, Any]  # Line 1098
    async def _validate_business_transition_rules(self, ...) -> bool  # Line 1105
    async def _validate_bot_transition_rules(self, current_state: dict[str, Any], new_state: dict[str, Any]) -> bool  # Line 1123
    async def _validate_order_transition_rules(self, current_state: dict[str, Any], new_state: dict[str, Any]) -> bool  # Line 1143
    async def _validate_risk_transition_rules(self, current_state: dict[str, Any], new_state: dict[str, Any]) -> bool  # Line 1163
    def _extract_status_field(self, state_data: dict[str, Any], state_type: 'StateType') -> str | None  # Line 1183
    def _generate_cache_key(self, ...) -> str  # Line 1204
    def _get_cached_result(self, cache_key: str) -> ValidationResult | None  # Line 1214
    def _cache_result(self, cache_key: str, result: ValidationResult) -> None  # Line 1228
    def _update_hit_rate(self, hit: bool) -> float  # Line 1240
    def _update_validation_metrics(self, result: ValidationResult) -> None  # Line 1251
    def _get_consistency_rule(self, rule_name: str) -> Callable | None  # Line 1264
    async def _check_portfolio_position_consistency(self, portfolio_state: dict[str, Any], related_states: list[dict[str, Any]]) -> dict[str, Any]  # Line 1274
    async def _check_order_position_consistency(self, order_state: dict[str, Any], related_states: list[dict[str, Any]]) -> dict[str, Any]  # Line 1309
    async def _check_risk_exposure_consistency(self, risk_state: dict[str, Any], related_states: list[dict[str, Any]]) -> dict[str, Any]  # Line 1316
    async def _load_custom_rules(self) -> None  # Line 1323
    async def _cache_cleanup_loop(self) -> None  # Line 1333
```

### File: trade_lifecycle_manager.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.caching import CacheKeys`
- `from src.core.caching import cache_invalidate`
- `from src.core.caching import cached`
- `from src.core.caching import get_cache_manager`

#### Class: `TradeEvent`

**Inherits**: str, Enum
**Purpose**: Trade event enumeration

```python
class TradeEvent(str, Enum):
```

#### Class: `PerformanceAttribution`

**Purpose**: Performance attribution analysis for trades

```python
class PerformanceAttribution:
```

#### Class: `TradeLifecycleManager`

**Inherits**: BaseComponent
**Purpose**: Comprehensive trade lifecycle management system

```python
class TradeLifecycleManager(BaseComponent):
    def __init__(self, ...)  # Line 99
    async def initialize(self) -> None  # Line 174
    async def cleanup(self) -> None  # Line 206
    async def start_trade_lifecycle(self, bot_id: str, strategy_name: str, order_request: OrderRequest) -> str  # Line 249
    async def transition_trade_state(self, ...) -> bool  # Line 306
    async def update_trade_execution(self, trade_id: str, execution_result: ExecutionResult) -> None  # Line 384
    async def update_trade_event(self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]) -> None  # Line 456
    async def calculate_trade_performance(self, trade_id: str) -> dict[str, Any]  # Line 523
    async def get_trade_history(self, ...) -> list[dict[str, Any]]  # Line 605
    async def get_performance_attribution(self, bot_id: str, period_days: int = 30) -> dict[str, Any]  # Line 673
    async def _cache_trade_context(self, trade_context: TradeContext) -> None  # Line 735
    async def _log_trade_event(self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]) -> None  # Line 809
    def _state_to_event(self, state: TradeLifecycleState) -> TradeEvent  # Line 854
    async def _finalize_trade(self, trade_id: str) -> None  # Line 869
    def _update_performance_metrics(self, trade_context: TradeContext) -> None  # Line 939
    def _history_record_to_performance(self, record: TradeHistoryRecord) -> dict[str, Any]  # Line 964
    async def _load_active_trades(self) -> None  # Line 980
    async def _monitoring_loop(self) -> None  # Line 1026
    async def create_trade_state(self, trade: Any) -> None  # Line 1046
    async def validate_trade_state(self, trade: Any) -> bool  # Line 1082
    async def calculate_trade_pnl(self, trade: Any) -> Decimal  # Line 1115
    async def assess_trade_risk(self, trade: Any) -> str  # Line 1144
    async def close_trade(self, trade_id: str, final_pnl: Decimal) -> None  # Line 1173
    async def update_trade_state(self, trade_id: str, trade_data: Any) -> None  # Line 1198
    def _get_state_ttl(self) -> int  # Line 1233
    def _get_staleness_threshold(self) -> int  # Line 1243
```

### File: types.py

**Key Imports:**
- `from src.core.types import ExecutionResult`
- `from src.core.types import OrderRequest`

#### Class: `ValidationResult`

**Inherits**: Enum
**Purpose**: Validation result enumeration

```python
class ValidationResult(Enum):
```

#### Class: `ValidationCheck`

**Purpose**: Individual validation check result

```python
class ValidationCheck:
```

#### Class: `PreTradeValidation`

**Purpose**: Pre-trade validation results

```python
class PreTradeValidation:
```

#### Class: `PostTradeAnalysis`

**Purpose**: Post-trade analysis results

```python
class PostTradeAnalysis:
```

### File: versioning.py

**Key Imports:**
- `from src.core.exceptions import StateConsistencyError`
- `from src.core.logging import get_logger`

#### Class: `MigrationType`

**Inherits**: Enum
**Purpose**: Types of migrations

```python
class MigrationType(Enum):
```

#### Class: `MigrationStatus`

**Inherits**: Enum
**Purpose**: Migration execution status

```python
class MigrationStatus(Enum):
```

#### Class: `StateVersion`

**Purpose**: State version information

```python
class StateVersion:
    def __post_init__(self) -> None  # Line 54
    def __str__(self) -> str  # Line 73
    def __lt__(self, other: 'StateVersion') -> bool  # Line 76
    def __le__(self, other: 'StateVersion') -> bool  # Line 79
    def __gt__(self, other: 'StateVersion') -> bool  # Line 82
    def __ge__(self, other: 'StateVersion') -> bool  # Line 85
    def __eq__(self, other: object) -> bool  # Line 88
    def is_compatible_with(self, other: 'StateVersion') -> bool  # Line 93
```

#### Class: `MigrationRecord`

**Purpose**: Record of a migration operation

```python
class MigrationRecord:
```

#### Class: `StateMigration`

**Inherits**: ABC
**Purpose**: Abstract base class for state migrations

```python
class StateMigration(ABC):
    def __init__(self, migration_id: str, name: str, description: str = '')  # Line 140
    def from_version(self) -> StateVersion  # Line 149
    def to_version(self) -> StateVersion  # Line 155
    def migration_type(self) -> MigrationType  # Line 161
    def depends_on(self) -> list[str]  # Line 166
    def affected_state_types(self) -> list[str]  # Line 171
    async def migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]  # Line 176
    async def rollback(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]  # Line 190
    async def validate_pre_migration(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> bool  # Line 205
    async def validate_post_migration(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> bool  # Line 211
```

#### Class: `StateVersioningSystem`

**Purpose**: State versioning and migration management system

```python
class StateVersioningSystem:
    def __init__(self, current_version: str = '1.0.0', metadata_service = None)  # Line 236
    def register_migration(self, migration: StateMigration) -> None  # Line 272
    def register_version_schema(self, version: str, schema: dict[str, Any]) -> None  # Line 282
    def set_version_compatibility(self, version: str, compatible_versions: list[str]) -> None  # Line 287
    def get_migration_path(self, from_version: StateVersion, to_version: StateVersion) -> list[str]  # Line 291
    async def migrate_state(self, ...) -> dict[str, Any]  # Line 336
    async def batch_migrate_states(self, states: list[dict[str, Any]], target_version: str | None = None) -> list[dict[str, Any]]  # Line 465
    def is_version_compatible(self, version1: str, version2: str) -> bool  # Line 516
    def get_schema_for_version(self, version: str) -> dict[str, Any]  # Line 531
    async def validate_state_schema(self, state_data: dict[str, Any], state_type: str, version: str) -> bool  # Line 535
    async def get_version_statistics(self) -> dict[str, Any]  # Line 557
    def get_migration_history(self, limit: int = 100) -> list[dict[str, Any]]  # Line 626
```

#### Class: `AddTimestampMigration`

**Inherits**: StateMigration
**Purpose**: Example migration to add timestamp field

```python
class AddTimestampMigration(StateMigration):
    def __init__(self)  # Line 658
    def from_version(self) -> StateVersion  # Line 666
    def to_version(self) -> StateVersion  # Line 670
    def migration_type(self) -> MigrationType  # Line 674
    async def migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]  # Line 677
    async def rollback(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]  # Line 690
```

#### Class: `RenameFieldMigration`

**Inherits**: StateMigration
**Purpose**: Example migration to rename a field

```python
class RenameFieldMigration(StateMigration):
    def __init__(self, old_field: str, new_field: str, affected_types: list[str])  # Line 703
    def from_version(self) -> StateVersion  # Line 714
    def to_version(self) -> StateVersion  # Line 718
    def migration_type(self) -> MigrationType  # Line 722
    def affected_state_types(self) -> list[str]  # Line 726
    async def migrate(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]  # Line 729
    async def rollback(self, state_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]  # Line 740
```

---
**Generated**: Complete reference for state module
**Total Classes**: 96
**Total Functions**: 12