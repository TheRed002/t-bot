# CORE Module Reference

## INTEGRATION
**Dependencies**: analytics, backtesting, bot_management, capital_management, data, database, error_handling, exchanges, execution, ml, monitoring, risk_management, state, strategies, utils, web_interface
**Used By**: error_handling
**Provides**: AsyncContextManager, BaseService, CacheManager, ConfigService, EnvironmentAwareService, HealthCheckManager, HighPerformanceMemoryManager, ResourceManager, ServiceManager, TaskManager, TransactionalService, WebSocketManager
**Patterns**: Async Operations, Component Architecture, Dependency Injection, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Security**:
- Credential management
- Credential management
- Credential management
**Performance**:
- Parallel execution
- Parallel execution
- Parallel execution
**Architecture**:
- BaseEventEmitter inherits from base architecture
- BaseFactory inherits from base architecture
- HealthCheckManager inherits from base architecture

## MODULE OVERVIEW
**Files**: 62 Python files
**Classes**: 396
**Functions**: 102

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `BaseComponent` ✅

**Inherits**: Lifecycle, HealthCheckable, Injectable, Loggable, Monitorable, Configurable
**Purpose**: Enhanced base component with complete lifecycle management
**Status**: Complete

**Implemented Methods:**
- `name(self) -> str` - Line 122
- `component_name(self) -> str` - Line 127
- `logger(self) -> Any` - Line 132
- `correlation_id(self) -> str` - Line 137
- `is_running(self) -> bool` - Line 142
- `is_starting(self) -> bool` - Line 147
- `is_stopping(self) -> bool` - Line 152
- `uptime(self) -> float` - Line 157
- `async start(self) -> None` - Line 165
- `async stop(self) -> None` - Line 211
- `async restart(self) -> None` - Line 254
- `async health_check(self) -> HealthCheckResult` - Line 273
- `async ready_check(self) -> HealthCheckResult` - Line 335
- `async live_check(self) -> HealthCheckResult` - Line 351
- `configure(self, config: ConfigDict) -> None` - Line 384
- `get_config(self) -> ConfigDict` - Line 409
- `validate_config(self, config: ConfigDict) -> bool` - Line 413
- `configure_dependencies(self, container: Any) -> None` - Line 426
- `async initialize(self) -> None` - Line 439
- `async cleanup(self) -> None` - Line 443
- `get_dependencies(self) -> list[str]` - Line 447
- `add_dependency(self, dependency_name: str) -> None` - Line 451
- `remove_dependency(self, dependency_name: str) -> None` - Line 455
- `get_metrics(self) -> dict[str, Any]` - Line 487
- `reset_metrics(self) -> None` - Line 501
- `async lifecycle_context(self)` - Line 515

### Implementation: `EventPriority` ✅

**Inherits**: Enum
**Purpose**: Event priority levels
**Status**: Complete

### Implementation: `EventMetadata` ✅

**Purpose**: Metadata for event tracking
**Status**: Complete

### Implementation: `EventContext` ✅

**Purpose**: Context for event processing
**Status**: Complete

### Implementation: `EventHandler` ✅

**Purpose**: Wrapper for event handler functions
**Status**: Complete

**Implemented Methods:**

### Implementation: `BaseEventEmitter` ✅

**Inherits**: BaseComponent, EventEmitter
**Purpose**: Base event emitter implementing the observer pattern
**Status**: Complete

**Implemented Methods:**
- `event_metrics(self) -> dict[str, Any]` - Line 214
- `on(self, ...) -> EventHandler` - Line 224
- `off(self, event: str, callback: Callable | EventHandler | None = None) -> None` - Line 271
- `once(self, ...) -> EventHandler` - Line 315
- `on_pattern(self, ...) -> EventHandler` - Line 357
- `on_global(self, ...) -> EventHandler` - Line 399
- `remove_all_listeners(self, event: str | None = None) -> None` - Line 434
- `emit(self, ...) -> None` - Line 470
- `async emit_async(self, ...) -> None` - Line 510
- `get_event_history(self, event_type: str | None = None, limit: int | None = None) -> list[EventContext]` - Line 748
- `get_handler_info(self) -> dict[str, Any]` - Line 773
- `get_events_summary(self) -> dict[str, Any]` - Line 816
- `configure_processing(self, ...) -> None` - Line 878
- `get_metrics(self) -> dict[str, Any]` - Line 916
- `reset_metrics(self) -> None` - Line 922

### Implementation: `DependencyInjectionMixin` ✅

**Purpose**: Simple mixin for dependency injection to avoid circular imports
**Status**: Complete

**Implemented Methods:**
- `get_injector(self)` - Line 36

### Implementation: `CreatorFunction` ✅

**Inherits**: Protocol, Generic[T]
**Purpose**: Protocol for creator functions
**Status**: Complete

**Implemented Methods:**

### Implementation: `BaseFactory` ✅

**Inherits**: BaseComponent, FactoryComponent, DependencyInjectionMixin, Generic[T]
**Purpose**: Base factory implementing the factory pattern
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, container: Any) -> None` - Line 139
- `product_type(self) -> type[T]` - Line 272
- `creation_metrics(self) -> dict[str, Any]` - Line 277
- `register(self, ...) -> None` - Line 282
- `register_interface(self, ...) -> None` - Line 335
- `unregister(self, name: str) -> None` - Line 386
- `update_creator_config(self, name: str, config: dict[str, Any]) -> None` - Line 425
- `create(self, name: str, *args: Any, **kwargs: Any) -> T` - Line 450
- `create_batch(self, requests: list[dict[str, Any]]) -> list[T]` - Line 557
- `list_registered(self) -> list[str]` - Line 725
- `is_registered(self, name: str) -> bool` - Line 734
- `get_creator_info(self, name: str) -> dict[str, Any] | None` - Line 746
- `get_all_creator_info(self) -> dict[str, dict[str, Any]]` - Line 771
- `get_singleton(self, name: str) -> T | None` - Line 780
- `clear_singletons(self) -> None` - Line 792
- `reset_singleton(self, name: str) -> None` - Line 814
- `get_metrics(self) -> dict[str, Any]` - Line 911
- `reset_metrics(self) -> None` - Line 923
- `configure_validation(self, validate_creators: bool = True, validate_products: bool = True) -> None` - Line 936

### Implementation: `HealthCheckType` ✅

**Inherits**: Enum
**Purpose**: Types of health checks
**Status**: Complete

### Implementation: `ComponentHealthInfo` ✅

**Purpose**: Health information for a registered component
**Status**: Complete

**Implemented Methods:**

### Implementation: `HealthCheckManager` ✅

**Inherits**: BaseComponent
**Purpose**: Centralized health check manager for all system components
**Status**: Complete

**Implemented Methods:**
- `health_metrics(self) -> dict[str, Any]` - Line 151
- `register_component(self, ...) -> None` - Line 159
- `unregister_component(self, name: str) -> None` - Line 210
- `enable_component_monitoring(self, name: str) -> None` - Line 245
- `disable_component_monitoring(self, name: str) -> None` - Line 261
- `async check_component_health(self, ...) -> HealthCheckResult` - Line 280
- `async check_all_components(self, ...) -> dict[str, HealthCheckResult]` - Line 387
- `async get_overall_health(self) -> HealthCheckResult` - Line 482
- `add_alert_callback(self, callback: Callable[[str, HealthCheckResult], Awaitable[None]]) -> None` - Line 551
- `remove_alert_callback(self, callback: Callable[[str, HealthCheckResult], Awaitable[None]]) -> None` - Line 568
- `clear_cache(self, component_name: str | None = None) -> None` - Line 685
- `get_component_info(self, component_name: str) -> dict[str, Any] | None` - Line 712
- `get_all_component_info(self) -> dict[str, dict[str, Any]]` - Line 747
- `list_components(self, enabled_only: bool = False) -> list[str]` - Line 755
- `get_metrics(self) -> dict[str, Any]` - Line 797
- `reset_metrics(self) -> None` - Line 803
- `configure_checks(self, ...) -> None` - Line 826
- `configure_alerts(self, ...) -> None` - Line 866

### Implementation: `HealthStatus` ✅

**Inherits**: Enum
**Purpose**: Health status enumeration for components
**Status**: Complete

### Implementation: `HealthCheckResult` ✅

**Purpose**: Result of a health check operation
**Status**: Complete

**Implemented Methods:**
- `healthy(self) -> bool` - Line 49
- `component(self) -> str` - Line 54
- `to_dict(self) -> dict[str, Any]` - Line 58

### Implementation: `Lifecycle` ✅

**Inherits**: Protocol
**Purpose**: Protocol for components with lifecycle management
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 71
- `async stop(self) -> None` - Line 75
- `async restart(self) -> None` - Line 79
- `is_running(self) -> bool` - Line 84

### Implementation: `HealthCheckable` ✅

**Inherits**: Protocol
**Purpose**: Protocol for components that support health checks
**Status**: Complete

**Implemented Methods:**
- `async health_check(self) -> HealthCheckResult` - Line 92
- `async ready_check(self) -> HealthCheckResult` - Line 96
- `async live_check(self) -> HealthCheckResult` - Line 100

### Implementation: `Injectable` ✅

**Inherits**: Protocol
**Purpose**: Protocol for dependency injection support
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, container: Any) -> None` - Line 108
- `get_dependencies(self) -> list[str]` - Line 112

### Implementation: `Loggable` ✅

**Inherits**: Protocol
**Purpose**: Protocol for components with structured logging
**Status**: Complete

**Implemented Methods:**
- `logger(self) -> Any` - Line 121
- `correlation_id(self) -> str | None` - Line 126

### Implementation: `Monitorable` ✅

**Inherits**: Protocol
**Purpose**: Protocol for components with metrics and monitoring
**Status**: Complete

**Implemented Methods:**
- `get_metrics(self) -> dict[str, int | float | str]` - Line 134
- `reset_metrics(self) -> None` - Line 138

### Implementation: `Configurable` ✅

**Inherits**: Protocol
**Purpose**: Protocol for components with configuration support
**Status**: Complete

**Implemented Methods:**
- `configure(self, config: ConfigDict) -> None` - Line 146
- `get_config(self) -> ConfigDict` - Line 150
- `validate_config(self, config: ConfigDict) -> bool` - Line 154

### Implementation: `Repository` ✅

**Inherits**: Protocol
**Purpose**: Protocol for repository pattern implementation
**Status**: Complete

**Implemented Methods:**
- `async create(self, entity: Any) -> Any` - Line 162
- `async get_by_id(self, entity_id: Any) -> Any | None` - Line 166
- `async update(self, entity: Any) -> Any` - Line 170
- `async delete(self, entity_id: Any) -> bool` - Line 174
- `async list(self, ...) -> list[Any]` - Line 178
- `async count(self, filters: dict[str, Any] | None = None) -> int` - Line 187

### Implementation: `Factory` ✅

**Inherits**: Protocol
**Purpose**: Protocol for factory pattern implementation
**Status**: Complete

**Implemented Methods:**
- `register(self, name: str, creator_func: Any) -> None` - Line 195
- `unregister(self, name: str) -> None` - Line 199
- `create(self, name: str, *args: Any, **kwargs: Any) -> Any` - Line 203
- `list_registered(self) -> list[str]` - Line 207

### Implementation: `EventEmitter` ✅

**Inherits**: Protocol
**Purpose**: Protocol for event emission and subscription
**Status**: Complete

**Implemented Methods:**
- `emit(self, event: str, data: Any = None) -> None` - Line 215
- `on(self, event: str, callback: Any) -> Any` - Line 219
- `off(self, event: str, callback: Any | None = None) -> None` - Line 223
- `once(self, event: str, callback: Any) -> Any` - Line 227
- `remove_all_listeners(self, event: str | None = None) -> None` - Line 231

### Implementation: `DIContainer` ✅

**Inherits**: Protocol
**Purpose**: Protocol for dependency injection container
**Status**: Complete

**Implemented Methods:**
- `register(self, interface: type, implementation: type | Any, singleton: bool = False) -> None` - Line 239
- `resolve(self, interface: type) -> Any` - Line 248
- `is_registered(self, interface: type) -> bool` - Line 252
- `register_factory(self, name: str, factory_func: Any, singleton: bool = False) -> None` - Line 256

### Implementation: `AsyncContextManager` ✅

**Inherits**: Protocol
**Purpose**: Protocol for async context managers
**Status**: Complete

**Implemented Methods:**

### Implementation: `ServiceComponent` 🔧

**Inherits**: Protocol
**Purpose**: Combined protocol for service layer components
**Status**: Abstract Base Class

**Implemented Methods:**
- `async start(self) -> None` - Line 283
- `async stop(self) -> None` - Line 284
- `async restart(self) -> None` - Line 285
- `is_running(self) -> bool` - Line 287
- `async health_check(self) -> HealthCheckResult` - Line 290
- `async ready_check(self) -> HealthCheckResult` - Line 291
- `async live_check(self) -> HealthCheckResult` - Line 292
- `configure_dependencies(self, container: Any) -> None` - Line 295
- `get_dependencies(self) -> list[str]` - Line 296
- `logger(self) -> Any` - Line 300
- `correlation_id(self) -> str | None` - Line 302
- `get_metrics(self) -> dict[str, int | float | str]` - Line 305
- `reset_metrics(self) -> None` - Line 306
- `configure(self, config: ConfigDict) -> None` - Line 309
- `get_config(self) -> ConfigDict` - Line 310
- `validate_config(self, config: ConfigDict) -> bool` - Line 311

### Implementation: `RepositoryComponent` 🔧

**Inherits**: Protocol
**Purpose**: Combined protocol for repository layer components
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create(self, entity: Any) -> Any` - Line 318
- `async get_by_id(self, entity_id: Any) -> Any | None` - Line 319
- `async update(self, entity: Any) -> Any` - Line 320
- `async delete(self, entity_id: Any) -> bool` - Line 321
- `async list(self, ...) -> list[Any]` - Line 322
- `async count(self, filters: dict[str, Any] | None = None) -> int` - Line 328
- `async health_check(self) -> HealthCheckResult` - Line 331
- `async ready_check(self) -> HealthCheckResult` - Line 332
- `async live_check(self) -> HealthCheckResult` - Line 333
- `configure_dependencies(self, container: Any) -> None` - Line 336
- `get_dependencies(self) -> builtins.list[str]` - Line 337
- `logger(self) -> Any` - Line 341
- `correlation_id(self) -> str | None` - Line 343

### Implementation: `FactoryComponent` 🔧

**Inherits**: Protocol
**Purpose**: Combined protocol for factory components
**Status**: Abstract Base Class

**Implemented Methods:**
- `register(self, name: str, creator_func: Any) -> None` - Line 350
- `unregister(self, name: str) -> None` - Line 351
- `create(self, name: str, *args: Any, **kwargs: Any) -> Any` - Line 352
- `list_registered(self) -> list[str]` - Line 353
- `configure_dependencies(self, container: Any) -> None` - Line 356
- `get_dependencies(self) -> list[str]` - Line 357
- `logger(self) -> Any` - Line 361
- `correlation_id(self) -> str | None` - Line 363

### Implementation: `WebServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Base interface for web service implementations
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 370
- `async cleanup(self) -> None` - Line 374

### Implementation: `TradingServiceInterface` ✅

**Inherits**: WebServiceInterface
**Purpose**: Interface for trading operations
**Status**: Complete

**Implemented Methods:**
- `async place_order(self, ...) -> str` - Line 383
- `async cancel_order(self, order_id: str) -> bool` - Line 395
- `async get_positions(self) -> list[Any]` - Line 400

### Implementation: `BotManagementServiceInterface` ✅

**Inherits**: WebServiceInterface
**Purpose**: Interface for bot management operations
**Status**: Complete

**Implemented Methods:**
- `async create_bot(self, config: Any) -> str` - Line 409
- `async start_bot(self, bot_id: str) -> bool` - Line 414
- `async stop_bot(self, bot_id: str) -> bool` - Line 419
- `async get_bot_status(self, bot_id: str) -> dict[str, Any]` - Line 424
- `async list_bots(self) -> list[dict[str, Any]]` - Line 429
- `async get_all_bots_status(self) -> dict[str, Any]` - Line 434
- `async delete_bot(self, bot_id: str, force: bool = False) -> bool` - Line 439

### Implementation: `MarketDataServiceInterface` ✅

**Inherits**: WebServiceInterface
**Purpose**: Interface for market data operations
**Status**: Complete

**Implemented Methods:**
- `async get_ticker(self, symbol: str) -> Any` - Line 448
- `async subscribe_to_ticker(self, symbol: str, callback: Any) -> None` - Line 453
- `async unsubscribe_from_ticker(self, symbol: str) -> None` - Line 458

### Implementation: `PortfolioServiceInterface` ✅

**Inherits**: WebServiceInterface
**Purpose**: Interface for portfolio operations
**Status**: Complete

**Implemented Methods:**
- `async get_balance(self) -> dict[str, Any]` - Line 467
- `async get_portfolio_summary(self) -> dict[str, Any]` - Line 472
- `async get_pnl_report(self, start_date: Any, end_date: Any) -> dict[str, Any]` - Line 477

### Implementation: `RiskServiceInterface` ✅

**Inherits**: WebServiceInterface
**Purpose**: Interface for risk management operations
**Status**: Complete

**Implemented Methods:**
- `async validate_order(self, symbol: str, side: str, amount: Any, price: Any | None = None) -> dict[str, Any]` - Line 486
- `async get_risk_metrics(self) -> dict[str, Any]` - Line 493
- `async update_risk_limits(self, limits: dict[str, Any]) -> bool` - Line 498

### Implementation: `StrategyServiceInterface` ✅

**Inherits**: WebServiceInterface
**Purpose**: Interface for strategy operations
**Status**: Complete

**Implemented Methods:**
- `async list_strategies(self) -> list[dict[str, Any]]` - Line 507
- `async get_strategy_config(self, strategy_name: str) -> dict[str, Any]` - Line 512
- `async validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool` - Line 517

### Implementation: `CacheClientInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for cache client implementations (Redis, etc
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 526
- `async disconnect(self) -> None` - Line 530
- `async ping(self) -> bool` - Line 534
- `async get(self, key: str, namespace: str = 'cache') -> Any | None` - Line 538
- `async set(self, key: str, value: Any, ttl: int | None = None, namespace: str = 'cache') -> bool` - Line 542
- `async delete(self, key: str, namespace: str = 'cache') -> bool` - Line 546
- `async exists(self, key: str, namespace: str = 'cache') -> bool` - Line 550
- `async expire(self, key: str, ttl: int, namespace: str = 'cache') -> bool` - Line 554
- `async info(self) -> dict[str, Any]` - Line 558
- `client(self) -> Any` - Line 567

### Implementation: `DatabaseServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for database service implementations
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 576
- `async stop(self) -> None` - Line 580
- `async health_check(self) -> HealthCheckResult` - Line 584
- `get_performance_metrics(self) -> dict[str, Any]` - Line 588
- `async execute_query(self, query: str, params: dict[str, Any] | None = None) -> Any` - Line 592
- `async get_connection_pool_status(self) -> dict[str, Any]` - Line 596

### Implementation: `BaseRepository` 🔧

**Inherits**: BaseComponent, RepositoryComponent, Generic[T, K]
**Purpose**: Base repository implementing the repository pattern
**Status**: Abstract Base Class

**Implemented Methods:**
- `entity_type(self) -> type[T]` - Line 136
- `key_type(self) -> type[K]` - Line 141
- `query_metrics(self) -> dict[str, Any]` - Line 146
- `set_connection_pool(self, connection_pool: Any) -> None` - Line 151
- `get_connection(self) -> AbstractAsyncContextManager[Any]` - Line 164
- `set_transaction_manager(self, transaction_manager: Any) -> None` - Line 182
- `async create(self, entity: T) -> T` - Line 196
- `async get_by_id(self, entity_id: K) -> T | None` - Line 244
- `async update(self, entity: T) -> T` - Line 291
- `async delete(self, entity_id: K) -> bool` - Line 346
- `async list(self, ...) -> list[T]` - Line 396
- `async count(self, filters: dict[str, Any] | None = None) -> int` - Line 465
- `async bulk_create(self, entities: builtins.list[T]) -> builtins.list[T]` - Line 511
- `async bulk_update(self, entities: builtins.list[T]) -> builtins.list[T]` - Line 562
- `async execute_in_transaction(self, operation_func: Callable[Ellipsis, Any], *args, **kwargs) -> Any` - Line 614
- `configure_cache(self, enabled: bool = True, ttl: int = 300) -> None` - Line 647
- `get_metrics(self) -> dict[str, Any]` - Line 809
- `reset_metrics(self) -> None` - Line 815

### Implementation: `BaseService` ✅

**Inherits**: BaseComponent, ServiceComponent
**Purpose**: Base service class implementing the service layer pattern
**Status**: Complete

**Implemented Methods:**
- `service_metrics(self) -> dict[str, Any]` - Line 106
- `async execute_with_monitoring(self, operation_name: str, operation_func: Any, *args, **kwargs) -> Any` - Line 119
- `configure_dependencies(self, dependency_injector: Any) -> None` - Line 392
- `resolve_dependency(self, dependency_name: str) -> Any` - Line 408
- `validate_config(self, config: ConfigDict) -> bool` - Line 511
- `get_metrics(self) -> dict[str, Any]` - Line 532
- `reset_metrics(self) -> None` - Line 538
- `get_operation_history(self, limit: int | None = None) -> list[dict[str, Any]]` - Line 553
- `configure_circuit_breaker(self, enabled: bool = True, threshold: int = 5, timeout: int = 60) -> None` - Line 568
- `configure_retry(self, ...) -> None` - Line 594
- `reset_circuit_breaker(self) -> None` - Line 624

### Implementation: `TransactionalService` ✅

**Inherits**: BaseService
**Purpose**: Base service with transaction management support
**Status**: Complete

**Implemented Methods:**
- `set_transaction_manager(self, transaction_manager: Any) -> None` - Line 715
- `async execute_in_transaction(self, operation_name: str, operation_func: Any, *args, **kwargs) -> Any` - Line 723

### Implementation: `CacheKeys` ✅

**Purpose**: Centralized cache key management for consistent naming patterns
**Status**: Complete

**Implemented Methods:**
- `state_snapshot(cls, bot_id: str) -> str` - Line 33
- `trade_lifecycle(cls, trade_id: str) -> str` - Line 38
- `state_checkpoint(cls, checkpoint_id: str) -> str` - Line 43
- `risk_metrics(cls, bot_id: str, timeframe: str = '1h') -> str` - Line 49
- `position_limits(cls, bot_id: str, symbol: str) -> str` - Line 54
- `correlation_matrix(cls, timeframe: str = '1h') -> str` - Line 59
- `var_calculation(cls, portfolio_id: str, confidence: str = '95') -> str` - Line 64
- `market_price(cls, symbol: str, exchange: str = 'all') -> str` - Line 70
- `order_book(cls, symbol: str, exchange: str, depth: int = 20) -> str` - Line 75
- `technical_indicator(cls, symbol: str, indicator: str, period: int) -> str` - Line 80
- `ohlcv_data(cls, symbol: str, timeframe: str, exchange: str = 'all') -> str` - Line 85
- `active_orders(cls, bot_id: str, symbol: str = 'all') -> str` - Line 91
- `order_history(cls, bot_id: str, page: int = 1) -> str` - Line 96
- `execution_state(cls, algorithm: str, bot_id: str) -> str` - Line 101
- `order_lock(cls, symbol: str, bot_id: str) -> str` - Line 106
- `strategy_signals(cls, strategy_id: str, symbol: str) -> str` - Line 112
- `strategy_params(cls, strategy_id: str) -> str` - Line 117
- `backtest_results(cls, strategy_id: str, config_hash: str) -> str` - Line 122
- `strategy_performance(cls, strategy_id: str, timeframe: str = '1d') -> str` - Line 127
- `bot_config(cls, bot_id: str) -> str` - Line 133
- `bot_status(cls, bot_id: str) -> str` - Line 138
- `resource_allocation(cls, bot_id: str) -> str` - Line 143
- `bot_session(cls, bot_id: str, session_id: str) -> str` - Line 148
- `api_response(cls, endpoint: str, user_id: str = 'anonymous', **params: Any) -> str` - Line 154
- `user_session(cls, user_id: str, session_id: str) -> str` - Line 160
- `auth_token(cls, user_id: str, token_hash: str) -> str` - Line 165
- `cache_stats(cls, namespace: str) -> str` - Line 171
- `performance_metrics(cls, component: str, timeframe: str = '5m') -> str` - Line 176
- `time_window_key(cls, base_key: str, window_minutes: int = 5) -> str` - Line 182
- `daily_key(cls, base_key: str) -> str` - Line 189
- `hourly_key(cls, base_key: str) -> str` - Line 195

### Implementation: `DependencyInjectionMixin` ✅

**Purpose**: Simple mixin for dependency injection
**Status**: Complete

**Implemented Methods:**
- `get_injector(self)` - Line 23

### Implementation: `ConnectionManagerMixin` ✅

**Purpose**: Simple connection manager mixin
**Status**: Complete

**Implemented Methods:**

### Implementation: `ResourceCleanupMixin` ✅

**Purpose**: Simple resource cleanup mixin
**Status**: Complete

**Implemented Methods:**
- `async cleanup_resources(self)` - Line 44

### Implementation: `LoggingHelperMixin` ✅

**Purpose**: Simple logging helper mixin
**Status**: Complete

**Implemented Methods:**

### Implementation: `CacheManager` ✅

**Inherits**: BaseComponent, DependencyInjectionMixin, ConnectionManagerMixin, ResourceCleanupMixin, LoggingHelperMixin
**Purpose**: Advanced cache manager with:
- Distributed locking for critical operations
- Cache warming strategie
**Status**: Complete

**Implemented Methods:**
- `async get(self, ...) -> Any` - Line 197
- `async set(self, ...) -> bool` - Line 255
- `async delete(self, key: str, namespace: str = 'cache') -> bool` - Line 287
- `async exists(self, key: str, namespace: str = 'cache') -> bool` - Line 308
- `async expire(self, key: str, ttl: int, namespace: str = 'cache') -> bool` - Line 324
- `async get_many(self, keys: list[str], namespace: str = 'cache') -> dict[str, Any]` - Line 342
- `async set_many(self, ...) -> bool` - Line 386
- `async acquire_lock(self, resource: str, timeout: int | None = None, namespace: str = 'locks') -> str | None` - Line 435
- `async release_lock(self, resource: str, lock_value: str, namespace: str = 'locks') -> bool` - Line 463
- `async with_lock(self, resource: str, func: Callable, *args, **kwargs)` - Line 492
- `async warm_cache(self, ...)` - Line 521
- `async invalidate_pattern(self, pattern: str, namespace: str = 'cache')` - Line 573
- `async health_check(self) -> Any` - Line 595
- `async cleanup(self) -> None` - Line 633
- `async shutdown(self) -> None` - Line 677
- `get_dependencies(self) -> list[str]` - Line 706

### Implementation: `CacheStats` ✅

**Purpose**: Cache statistics for monitoring with memory tracking
**Status**: Complete

**Implemented Methods:**
- `hit_rate(self) -> float` - Line 34
- `miss_rate(self) -> float` - Line 40

### Implementation: `CacheMetrics` ✅

**Inherits**: BaseComponent
**Purpose**: Cache metrics collector and reporter with memory accounting
**Status**: Complete

**Implemented Methods:**
- `record_hit(self, namespace: str, response_time: float = 0.0)` - Line 66
- `record_miss(self, namespace: str, response_time: float = 0.0)` - Line 124
- `record_set(self, namespace: str, response_time: float = 0.0, memory_bytes: int = 0)` - Line 137
- `record_delete(self, namespace: str, response_time: float = 0.0, memory_freed: int = 0)` - Line 153
- `record_error(self, namespace: str, error_type: str = 'unknown')` - Line 165
- `get_stats(self, namespace: str | None = None) -> dict[str, Any]` - Line 174
- `get_recent_operations(self, namespace: str, limit: int = 100) -> list[dict[str, Any]]` - Line 233
- `get_performance_summary(self, time_window_minutes: int = 5) -> dict[str, Any]` - Line 238
- `shutdown(self)` - Line 274
- `reset_stats(self, namespace: str | None = None)` - Line 290
- `export_metrics_for_monitoring(self) -> dict[str, Any]` - Line 300

### Implementation: `CacheHealthStatus` ✅

**Inherits**: Enum
**Purpose**: Cache health status enumeration
**Status**: Complete

### Implementation: `CacheAlert` ✅

**Purpose**: Cache alert definition
**Status**: Complete

### Implementation: `CacheHealthReport` ✅

**Purpose**: Comprehensive cache health report
**Status**: Complete

### Implementation: `CacheMonitor` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive cache monitoring and health management
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self) -> None` - Line 121
- `async stop_monitoring(self) -> None` - Line 130
- `async get_health_report(self) -> CacheHealthReport` - Line 254
- `async acknowledge_alert(self, alert_id: str) -> bool` - Line 439
- `async clear_acknowledged_alerts(self) -> int` - Line 447
- `async get_performance_trends(self, hours: int = 24) -> dict[str, Any]` - Line 465

### Implementation: `WarmingStrategy` ✅

**Inherits**: Enum
**Purpose**: Cache warming strategy types
**Status**: Complete

### Implementation: `WarmingPriority` ✅

**Inherits**: Enum
**Purpose**: Warming priority levels
**Status**: Complete

### Implementation: `WarmingTask` ✅

**Purpose**: Represents a cache warming task
**Status**: Complete

### Implementation: `CacheWarmer` ✅

**Inherits**: BaseComponent
**Purpose**: Intelligent cache warming system for trading data
**Status**: Complete

**Implemented Methods:**
- `async start_warming(self) -> None` - Line 126
- `async stop_warming(self) -> None` - Line 148
- `register_warming_task(self, task: WarmingTask) -> None` - Line 264
- `register_market_data_warming(self, symbols: list[str], exchange: str = 'all') -> None` - Line 272
- `register_bot_state_warming(self, bot_ids: list[str]) -> None` - Line 307
- `register_risk_metrics_warming(self, bot_ids: list[str] | None = None, timeframes: list[str] | None = None) -> None` - Line 339
- `register_strategy_performance_warming(self, strategy_ids: list[str]) -> None` - Line 361
- `async get_warming_status(self) -> dict[str, Any]` - Line 683
- `async warm_critical_data_now(self) -> dict[str, Any]` - Line 709

### Implementation: `CacheLevel` ✅

**Inherits**: Enum
**Purpose**: Cache levels in the hierarchy
**Status**: Complete

### Implementation: `CacheStrategy` ✅

**Inherits**: Enum
**Purpose**: Cache management strategies
**Status**: Complete

### Implementation: `DataCategory` ✅

**Inherits**: Enum
**Purpose**: Categories of data for cache optimization
**Status**: Complete

### Implementation: `CachePolicy` ✅

**Purpose**: Caching policy for different data categories
**Status**: Complete

### Implementation: `CacheEntry` ✅

**Purpose**: Enhanced cache entry with metadata
**Status**: Complete

**Implemented Methods:**
- `is_expired(self) -> bool` - Line 122
- `touch(self) -> None` - Line 128

### Implementation: `CacheStats` ✅

**Purpose**: Comprehensive cache statistics
**Status**: Complete

### Implementation: `CacheInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for cache implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get(self, key: str) -> Any | None` - Line 155
- `async set(self, key: str, value: Any, ttl: int | None = None) -> bool` - Line 160
- `async delete(self, key: str) -> bool` - Line 165
- `async clear(self) -> None` - Line 170
- `async get_stats(self) -> CacheStats` - Line 175

### Implementation: `L1CPUCache` ✅

**Inherits**: CacheInterface
**Purpose**: L1 CPU Cache - Ultra-fast cache optimized for CPU cache efficiency
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> Any | None` - Line 199
- `async set(self, key: str, value: Any, ttl: int | None = None) -> bool` - Line 245
- `async delete(self, key: str) -> bool` - Line 308
- `async clear(self) -> None` - Line 324
- `async get_stats(self) -> CacheStats` - Line 332

### Implementation: `L2MemoryCache` ✅

**Inherits**: CacheInterface
**Purpose**: L2 Memory Cache - Application-level memory cache with advanced features
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> Any | None` - Line 387
- `async set(self, ...) -> bool` - Line 409
- `async delete(self, key: str) -> bool` - Line 440
- `async clear(self) -> None` - Line 460
- `async get_stats(self) -> CacheStats` - Line 513

### Implementation: `L3RedisCache` ✅

**Inherits**: CacheInterface
**Purpose**: L3 Redis Cache - Distributed cache for sharing data across instances
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> Any | None` - Line 536
- `async set(self, key: str, value: Any, ttl: int | None = None) -> bool` - Line 559
- `async delete(self, key: str) -> bool` - Line 578
- `async clear(self) -> None` - Line 596
- `async get_stats(self) -> CacheStats` - Line 658

### Implementation: `UnifiedCacheLayer` ✅

**Inherits**: BaseComponent
**Purpose**: Unified caching layer that coordinates all cache levels for optimal performance
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 767
- `async get(self, ...) -> Any | None` - Line 823
- `async set(self, ...) -> bool` - Line 877
- `async delete(self, key: str, category: DataCategory = DataCategory.TRADING_DATA) -> bool` - Line 922
- `async invalidate_pattern(self, pattern: str, category: DataCategory) -> int` - Line 948
- `async warm_cache(self, keys: list[str], category: DataCategory, loader: Callable) -> int` - Line 969
- `async get_comprehensive_stats(self) -> dict[str, Any]` - Line 1162
- `async cleanup(self) -> None` - Line 1184

### Implementation: `BaseConfig` ✅

**Inherits**: BaseSettings
**Purpose**: Base configuration class with common patterns
**Status**: Complete

**Implemented Methods:**
- `run_validators(self) -> None` - Line 29
- `add_validator(self, validator: Callable) -> None` - Line 34

### Implementation: `BotManagementConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for bot management system
**Status**: Complete

**Implemented Methods:**
- `get_resource_limits(self) -> dict` - Line 208
- `get_alert_thresholds(self) -> dict` - Line 212
- `get_coordination_config(self) -> dict` - Line 216
- `get_lifecycle_config(self) -> dict` - Line 225
- `get_monitoring_config(self) -> dict` - Line 234
- `get_connection_timeouts(self) -> dict` - Line 243
- `get_operational_delays(self) -> dict` - Line 247
- `get_circuit_breaker_configs(self) -> dict` - Line 251
- `serialize_decimal(self, value)` - Line 258

### Implementation: `CapitalManagementConfig` ✅

**Inherits**: BaseModel
**Purpose**: Capital management configuration settings
**Status**: Complete

**Implemented Methods:**
- `get_available_capital(self) -> Decimal` - Line 186
- `get_emergency_reserve(self) -> Decimal` - Line 191
- `get_max_allocation_for_strategy(self) -> Decimal` - Line 195
- `get_min_allocation_for_strategy(self) -> Decimal` - Line 200
- `model_dump(self, **kwargs: Any) -> dict[str, Any]` - Line 205

### Implementation: `DatabaseConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Database configuration for PostgreSQL, Redis, and InfluxDB
**Status**: Complete

**Implemented Methods:**
- `validate_ports(cls, v: int) -> int` - Line 71
- `validate_pool_size(cls, v: int) -> int` - Line 79
- `validate_redis_db(cls, v: int) -> int` - Line 87
- `postgresql_url(self) -> str` - Line 94
- `redis_url(self) -> str` - Line 108

### Implementation: `TradingEnvironment` ✅

**Inherits**: Enum
**Purpose**: Trading environment types
**Status**: Complete

### Implementation: `ExchangeEnvironment` ✅

**Inherits**: Enum
**Purpose**: Exchange-specific environment types
**Status**: Complete

### Implementation: `EnvironmentConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Configuration for trading environment switching
**Status**: Complete

**Implemented Methods:**
- `validate_global_environment(cls, v)` - Line 173
- `validate_exchange_environment(cls, v)` - Line 184
- `get_exchange_environment(self, exchange_name: str) -> ExchangeEnvironment` - Line 195
- `get_exchange_endpoints(self, exchange_name: str) -> dict[str, str]` - Line 216
- `get_exchange_credentials(self, exchange_name: str) -> dict[str, Any]` - Line 266
- `is_production_environment(self, exchange_name: str) -> bool` - Line 323
- `validate_production_credentials(self, exchange_name: str) -> bool` - Line 328
- `get_environment_summary(self) -> dict[str, Any]` - Line 353

### Implementation: `ExchangeConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Exchange-specific configuration
**Status**: Complete

**Implemented Methods:**
- `validate_default_exchange(cls, v: str) -> str` - Line 142
- `validate_enabled_exchanges(cls, v: list[str]) -> list[str]` - Line 151
- `get_exchange_credentials(self, exchange: str) -> dict[str, Any]` - Line 159
- `is_exchange_configured(self, exchange: str) -> bool` - Line 192
- `get_websocket_config(self, exchange: str) -> dict[str, Any]` - Line 200
- `get_connection_pool_config(self) -> dict[str, Any]` - Line 227
- `get_rate_limit_config(self) -> dict[str, Any]` - Line 240

### Implementation: `ExecutionConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Execution-specific configuration
**Status**: Complete

**Implemented Methods:**
- `validate_decimal_fields(cls, v)` - Line 78
- `get(self, key: str, default: Any = None) -> Any` - Line 86
- `get_routing_config(self) -> dict[str, Any]` - Line 90
- `get_order_size_limits(self) -> dict[str, Decimal | None]` - Line 94
- `get_performance_settings(self) -> dict[str, int]` - Line 98

### Implementation: `Config` ✅

**Purpose**: Main configuration aggregator that maintains backward compatibility
**Status**: Complete

**Implemented Methods:**
- `load_from_file(self, config_file: str) -> None` - Line 73
- `save_to_file(self, config_file: str) -> None` - Line 134
- `validate(self) -> None` - Line 173
- `db_url(self) -> str` - Line 187
- `redis_url(self) -> str` - Line 192
- `postgresql_host(self) -> str` - Line 197
- `postgresql_port(self) -> int` - Line 202
- `postgresql_database(self) -> str` - Line 207
- `postgresql_username(self) -> str` - Line 212
- `postgresql_password(self) -> str | None` - Line 217
- `redis_host(self) -> str` - Line 222
- `redis_port(self) -> int` - Line 227
- `binance_api_key(self) -> str` - Line 232
- `binance_api_secret(self) -> str` - Line 237
- `max_position_size(self) -> Any` - Line 242
- `risk_per_trade(self) -> float` - Line 247
- `get_exchange_config(self, exchange: str) -> dict[str, Any]` - Line 251
- `get_environment_exchange_config(self, exchange: str) -> dict[str, Any]` - Line 264
- `get_strategy_config(self, strategy_type: str) -> dict[str, Any]` - Line 295
- `get_risk_config(self) -> dict[str, Any]` - Line 299
- `to_dict(self) -> dict[str, Any]` - Line 303
- `switch_environment(self, environment: str, exchange: str = None) -> bool` - Line 332
- `validate_environment_switch(self, environment: str, exchange: str = None) -> dict[str, Any]` - Line 373
- `get_current_environment_status(self) -> dict[str, Any]` - Line 438
- `is_production_mode(self, exchange: str = None) -> bool` - Line 454

### Implementation: `RiskConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Risk management configuration
**Status**: Complete

**Implemented Methods:**
- `validate_sizing_method(cls, v: str) -> str` - Line 236
- `get_position_size_params(self) -> dict` - Line 249
- `is_risk_exceeded(self, current_loss: Decimal) -> bool` - Line 275

### Implementation: `SandboxEnvironment` ✅

**Inherits**: str, Enum
**Purpose**: Sandbox environment types
**Status**: Complete

### Implementation: `SandboxExchangeConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Sandbox-specific exchange configuration
**Status**: Complete

**Implemented Methods:**
- `validate_environment(cls, v: SandboxEnvironment) -> SandboxEnvironment` - Line 126
- `get_sandbox_credentials(self, exchange: str) -> dict[str, Any]` - Line 135
- `get_mock_balances(self) -> dict[str, str]` - Line 170
- `is_sandbox_environment(self) -> bool` - Line 178
- `get_environment_config(self) -> dict[str, Any]` - Line 182

### Implementation: `SecurityConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Security configuration for JWT, authentication, and other security settings
**Status**: Complete

**Implemented Methods:**
- `get_jwt_config(self) -> dict` - Line 121
- `get_session_config(self) -> dict` - Line 130
- `get_cors_config(self) -> dict` - Line 136
- `get_rate_limit_config(self) -> dict` - Line 145

### Implementation: `ConfigProvider` ✅

**Inherits**: Protocol
**Purpose**: Protocol for configuration providers
**Status**: Complete

**Implemented Methods:**
- `async load_config(self) -> ConfigDict` - Line 75
- `async save_config(self, config: ConfigDict) -> None` - Line 79
- `async watch_changes(self, callback: ConfigCallback) -> None` - Line 83

### Implementation: `ConfigChangeEvent` ✅

**Inherits**: BaseValidatedModel
**Purpose**: Configuration change event
**Status**: Complete

### Implementation: `ConfigCache` ✅

**Purpose**: Thread-safe configuration cache with TTL support
**Status**: Complete

**Implemented Methods:**
- `get(self, key: str, default: Any = None) -> Any` - Line 110
- `set(self, key: str, value: Any, ttl: int | None = None) -> None` - Line 125
- `invalidate(self, key: str) -> None` - Line 135
- `clear(self) -> None` - Line 144
- `get_stats(self) -> dict[str, Any]` - Line 151

### Implementation: `FileConfigProvider` ✅

**Purpose**: File-based configuration provider
**Status**: Complete

**Implemented Methods:**
- `async load_config(self) -> ConfigDict` - Line 174
- `async save_config(self, config: ConfigDict) -> None` - Line 222
- `async watch_changes(self, callback: ConfigCallback) -> None` - Line 254

### Implementation: `EnvironmentConfigProvider` ✅

**Purpose**: Environment-based configuration provider
**Status**: Complete

**Implemented Methods:**
- `async load_config(self) -> ConfigDict` - Line 273
- `async save_config(self, config: ConfigDict) -> None` - Line 298
- `async watch_changes(self, callback: ConfigCallback) -> None` - Line 306

### Implementation: `ConfigValidator` ✅

**Purpose**: Configuration validation service
**Status**: Complete

**Implemented Methods:**
- `async validate_database_config(self, config: dict) -> DatabaseConfig` - Line 321
- `async validate_exchange_config(self, config: dict) -> ExchangeConfig` - Line 333
- `async validate_risk_config(self, config: dict) -> RiskConfig` - Line 345
- `async validate_strategy_config(self, config: dict) -> StrategyConfig` - Line 357
- `register_validator(self, config_section: str, validator: Callable[[dict], Any]) -> None` - Line 369
- `async validate_custom_config(self, section: str, config: dict) -> Any` - Line 374

### Implementation: `ConfigService` ✅

**Purpose**: Main configuration service with dependency injection support
**Status**: Complete

**Implemented Methods:**
- `async initialize(self, config_file: str | Path | None = None, watch_changes: bool = False) -> None` - Line 455
- `async shutdown(self) -> None` - Line 489
- `add_change_listener(self, callback: ConfigCallback) -> None` - Line 624
- `remove_change_listener(self, callback: ConfigCallback) -> None` - Line 628
- `get_database_config(self) -> DatabaseConfig` - Line 635
- `get_exchange_config(self, exchange: str | None = None) -> ExchangeConfig | dict[str, Any]` - Line 648
- `get_risk_config(self) -> RiskConfig` - Line 671
- `get_strategy_config(self, strategy_type: str | None = None) -> StrategyConfig | dict[str, Any]` - Line 684
- `get_app_config(self) -> dict[str, Any]` - Line 709
- `get_config_value(self, key: str, default: Any = None) -> Any` - Line 730
- `get_cache_stats(self) -> dict[str, Any]` - Line 779
- `invalidate_cache(self, key: str | None = None) -> None` - Line 783
- `get_loaded_config(self) -> dict[str, Any] | None` - Line 790
- `get_config_dict(self) -> dict[str, Any]` - Line 796
- `get_config(self) -> dict[str, Any]` - Line 805

### Implementation: `StateManagementConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for state management system
**Status**: Complete

**Implemented Methods:**
- `get_checkpoint_config(self) -> dict` - Line 144
- `get_validation_config(self) -> dict` - Line 148
- `get_recovery_config(self) -> dict` - Line 157
- `get_performance_config(self) -> dict` - Line 165
- `get_monitoring_config(self) -> dict` - Line 176
- `get_sync_config(self) -> dict` - Line 185

### Implementation: `StrategyConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Strategy-specific configuration
**Status**: Complete

**Implemented Methods:**
- `validate_timeframe(cls, v: str) -> str` - Line 116
- `validate_combination_method(cls, v: str) -> str` - Line 125
- `get_strategy_params(self, strategy_type: str) -> dict[str, Any]` - Line 132

### Implementation: `BaseConfig` ✅

**Inherits**: BaseSettings
**Purpose**: Base configuration class with common patterns
**Status**: Complete

### Implementation: `DatabaseConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Database configuration for PostgreSQL, Redis, and InfluxDB
**Status**: Complete

**Implemented Methods:**
- `validate_ports(cls, v: int) -> int` - Line 86
- `validate_pool_size(cls, v)` - Line 94
- `postgresql_url(self) -> str` - Line 101
- `redis_url(self) -> str` - Line 109

### Implementation: `SecurityConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Security configuration for authentication and encryption
**Status**: Complete

**Implemented Methods:**
- `validate_jwt_expire(cls, v)` - Line 138
- `validate_key_length(cls, v)` - Line 218

### Implementation: `ErrorHandlingConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Error handling configuration for P-002A framework
**Status**: Complete

**Implemented Methods:**
- `validate_positive_integers(cls, v)` - Line 281
- `validate_positive_floats(cls, v)` - Line 291

### Implementation: `ExchangeConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Exchange configuration for API credentials and rate limits
**Status**: Complete

**Implemented Methods:**
- `validate_api_credentials(cls, v: str) -> str` - Line 358
- `default_exchange(self) -> str` - Line 387
- `testnet_mode(self) -> bool` - Line 392
- `rate_limit_per_second(self) -> int` - Line 404
- `get_exchange_credentials(self, exchange: str) -> dict[str, Any]` - Line 409
- `get_websocket_config(self, exchange: str) -> dict[str, Any]` - Line 433

### Implementation: `RiskConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Risk management configuration for P-008 framework
**Status**: Complete

**Implemented Methods:**
- `validate_percentage_fields(cls, v)` - Line 539
- `validate_positive_integers(cls, v)` - Line 554

### Implementation: `CapitalManagementConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Capital management configuration for P-010A framework
**Status**: Complete

**Implemented Methods:**
- `validate_percentage_fields(cls, v)` - Line 681
- `validate_positive_integers(cls, v)` - Line 689
- `validate_positive_decimals(cls, v: Decimal) -> Decimal` - Line 703

### Implementation: `StrategyManagementConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Strategy management configuration for P-011 framework
**Status**: Complete

**Implemented Methods:**
- `validate_positive_integers(cls, v)` - Line 760
- `validate_percentage_fields(cls, v)` - Line 777

### Implementation: `MLConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Machine Learning configuration for P-014 framework
**Status**: Complete

**Implemented Methods:**
- `validate_percentage_fields(cls, v)` - Line 910
- `validate_positive_integers(cls, v)` - Line 931
- `validate_positive_float(cls, v)` - Line 939

### Implementation: `BotManagementConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Bot management configuration for bot lifecycle and coordination
**Status**: Complete

**Implemented Methods:**
- `validate_decimal_positive(cls, v: Decimal) -> Decimal` - Line 1030
- `validate_positive_integers(cls, v: int) -> int` - Line 1052

### Implementation: `ExecutionConfig` ✅

**Inherits**: BaseConfig
**Purpose**: Execution engine configuration for order processing and algorithms
**Status**: Complete

### Implementation: `Config` ✅

**Inherits**: BaseConfig
**Purpose**: Master configuration class for the entire application
**Status**: Complete

**Implemented Methods:**
- `validate_environment(cls, v: str) -> str` - Line 1142
- `generate_schema(self) -> None` - Line 1149
- `from_yaml(cls, yaml_path: str | Path) -> 'Config'` - Line 1158
- `from_yaml_with_env_override(cls, yaml_path: str | Path) -> 'Config'` - Line 1187
- `to_yaml(self, yaml_path: str | Path) -> None` - Line 1222
- `get_database_url(self) -> str` - Line 1237
- `get_async_database_url(self) -> str` - Line 1251
- `get_redis_url(self) -> str` - Line 1265
- `is_production(self) -> bool` - Line 1282
- `is_development(self) -> bool` - Line 1286
- `validate_yaml_config(self, yaml_path: str | Path) -> bool` - Line 1290

### Implementation: `CoreDataTransformer` ✅

**Purpose**: Handles consistent data transformation for core module events and messaging
**Status**: Complete

**Implemented Methods:**
- `transform_event_to_standard_format(event_type, ...) -> dict[str, Any]` - Line 29
- `transform_for_pub_sub_pattern(event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 79
- `transform_for_request_reply_pattern(request_type, ...) -> dict[str, Any]` - Line 109
- `align_processing_paradigm(data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 143
- `validate_boundary_fields(data: dict[str, Any]) -> dict[str, Any]` - Line 188
- `apply_cross_module_consistency(cls, data: dict[str, Any], target_module: str, source_module: str = 'core') -> dict[str, Any]` - Line 287

### Implementation: `DependencyContainer` ✅

**Purpose**: Container for managing dependencies
**Status**: Complete

**Implemented Methods:**
- `register(self, name: str, service: Any | Callable, singleton: bool = False) -> None` - Line 30
- `register_class(self, name: str, cls: type[T], *args, **kwargs) -> None` - Line 60
- `get(self, name: str) -> Any` - Line 124
- `has(self, name: str) -> bool` - Line 175
- `resolve(self, name: str) -> Any` - Line 179
- `get_optional(self, name: str) -> Any | None` - Line 194
- `register_factory(self, name: str, factory: Callable, singleton: bool = False) -> None` - Line 209
- `register_singleton(self, name: str, service: Any) -> None` - Line 220
- `register_service(self, name: str, service: Any, singleton: bool = False) -> None` - Line 230
- `has_service(self, name: str) -> bool` - Line 241
- `is_registered(self, name: str) -> bool` - Line 253
- `get_container(self) -> 'DependencyContainer'` - Line 265
- `clear(self) -> None` - Line 278

### Implementation: `DependencyInjector` ✅

**Purpose**: Dependency injector for automatic dependency resolution
**Status**: Complete

**Implemented Methods:**
- `register(self, name: str | None = None, singleton: bool = False)` - Line 314
- `inject(self, func: Callable) -> Callable` - Line 337
- `resolve(self, name: str) -> Any` - Line 397
- `get_optional(self, name: str) -> Any | None` - Line 422
- `register_service(self, name: str, service: Any, singleton: bool = False) -> None` - Line 437
- `register_factory(self, name: str, factory: Callable, singleton: bool = False) -> None` - Line 460
- `register_interface(self, interface: type, implementation_factory: Callable, singleton: bool = False) -> None` - Line 485
- `register_singleton(self, name: str, service: Any) -> None` - Line 517
- `register_transient(self, name: str, service_class: type, *args, **kwargs) -> None` - Line 530
- `has_service(self, name: str) -> bool` - Line 566
- `is_registered(self, name: str) -> bool` - Line 570
- `clear(self) -> None` - Line 574
- `get_instance(cls) -> DependencyInjector` - Line 579
- `reset_instance(cls) -> None` - Line 586
- `get_container(self) -> DependencyContainer` - Line 596
- `configure_service_dependencies(self, service_instance: Any) -> None` - Line 600

### Implementation: `ServiceLocator` ✅

**Purpose**: Service locator for easy access to services
**Status**: Complete

**Implemented Methods:**

### Implementation: `DependencyLevel` ✅

**Inherits**: IntEnum
**Purpose**: Dependency levels for ordering service registration
**Status**: Complete

### Implementation: `DependencyRegistrar` ✅

**Purpose**: Manages ordered dependency registration to prevent circular dependencies
**Status**: Complete

**Implemented Methods:**
- `register_at_level(self, level: DependencyLevel, registration_func: Callable) -> None` - Line 79
- `add_lazy_configuration(self, config_func: Callable) -> None` - Line 91
- `register_all(self) -> None` - Line 100

### Implementation: `AlertEvents` ✅

**Purpose**: Alert-related event names
**Status**: Complete

### Implementation: `OrderEvents` ✅

**Purpose**: Order-related event names
**Status**: Complete

### Implementation: `TradeEvents` ✅

**Purpose**: Trade-related event names
**Status**: Complete

### Implementation: `PositionEvents` ✅

**Purpose**: Position-related event names
**Status**: Complete

### Implementation: `RiskEvents` ✅

**Purpose**: Risk management event names
**Status**: Complete

### Implementation: `MetricEvents` ✅

**Purpose**: Metric-related event names
**Status**: Complete

### Implementation: `SystemEvents` ✅

**Purpose**: System-level event names
**Status**: Complete

### Implementation: `MarketDataEvents` ✅

**Purpose**: Market data event names
**Status**: Complete

### Implementation: `StrategyEvents` ✅

**Purpose**: Strategy-related event names
**Status**: Complete

### Implementation: `CapitalEvents` ✅

**Purpose**: Capital management event names
**Status**: Complete

### Implementation: `ExchangeEvents` ✅

**Purpose**: Exchange connection event names
**Status**: Complete

### Implementation: `MLEvents` ✅

**Purpose**: Machine learning event names
**Status**: Complete

### Implementation: `TrainingEvents` ✅

**Purpose**: Model training event names
**Status**: Complete

### Implementation: `InferenceEvents` ✅

**Purpose**: Model inference event names
**Status**: Complete

### Implementation: `FeatureEvents` ✅

**Purpose**: Feature engineering event names
**Status**: Complete

### Implementation: `ModelValidationEvents` ✅

**Purpose**: Model validation event names
**Status**: Complete

### Implementation: `StateEvents` ✅

**Purpose**: State management event names
**Status**: Complete

### Implementation: `BacktestEvents` ✅

**Purpose**: Backtesting event names
**Status**: Complete

### Implementation: `DataEvents` ✅

**Purpose**: Data processing event names
**Status**: Complete

### Implementation: `OptimizationEvents` ✅

**Purpose**: Optimization-related event names
**Status**: Complete

### Implementation: `BotEvents` ✅

**Purpose**: Bot management event names
**Status**: Complete

### Implementation: `BotEventType` ✅

**Inherits**: Enum
**Purpose**: Bot event types for coordination and monitoring
**Status**: Complete

### Implementation: `BotEvent` ✅

**Purpose**: Bot event data structure
**Status**: Complete

**Implemented Methods:**

### Implementation: `EventHandler` ✅

**Purpose**: Base class for event handlers
**Status**: Complete

**Implemented Methods:**
- `async handle(self, event: BotEvent) -> None` - Line 82

### Implementation: `EventPublisher` ✅

**Purpose**: Event publisher for bot management coordination
**Status**: Complete

**Implemented Methods:**
- `subscribe(self, event_type: BotEventType, handler: EventHandler) -> None` - Line 97
- `subscribe_all(self, handler: EventHandler) -> None` - Line 106
- `unsubscribe(self, event_type: BotEventType, handler: EventHandler) -> None` - Line 111
- `async publish(self, event: BotEvent, processing_mode: str = 'stream') -> None` - Line 118
- `get_recent_events(self, ...) -> list[BotEvent]` - Line 196

### Implementation: `AnalyticsEventHandler` ✅

**Inherits**: EventHandler
**Purpose**: Event handler for analytics integration
**Status**: Complete

**Implemented Methods:**
- `async handle(self, event: BotEvent) -> None` - Line 401

### Implementation: `RiskMonitoringEventHandler` ✅

**Inherits**: EventHandler
**Purpose**: Event handler for risk monitoring
**Status**: Complete

**Implemented Methods:**
- `async handle(self, event: BotEvent) -> None` - Line 502

### Implementation: `ErrorCategory` ✅

**Inherits**: Enum
**Purpose**: Error categorization for automated handling
**Status**: Complete

### Implementation: `ErrorSeverity` ✅

**Inherits**: Enum
**Purpose**: Error severity levels
**Status**: Complete

### Implementation: `TradingBotError` ✅

**Inherits**: Exception
**Purpose**: Base exception for all trading bot errors
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict[str, Any]` - Line 188

### Implementation: `ExchangeError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all exchange-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeConnectionError` ✅

**Inherits**: ExchangeError
**Purpose**: Network connection failures to exchange APIs
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeRateLimitError` ✅

**Inherits**: ExchangeError
**Purpose**: Rate limit violations from exchange APIs
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeInsufficientFundsError` ✅

**Inherits**: ExchangeError
**Purpose**: Insufficient balance for order execution
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeOrderError` ✅

**Inherits**: ExchangeError
**Purpose**: General order-related exchange errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeAuthenticationError` ✅

**Inherits**: ExchangeError
**Purpose**: Exchange authentication and authorization failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `InvalidOrderError` ✅

**Inherits**: ExchangeOrderError
**Purpose**: Invalid order parameters
**Status**: Complete

**Implemented Methods:**

### Implementation: `RiskManagementError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all risk management violations and errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `PositionLimitError` ✅

**Inherits**: RiskManagementError
**Purpose**: Position size or count limit violations
**Status**: Complete

**Implemented Methods:**

### Implementation: `DrawdownLimitError` ✅

**Inherits**: RiskManagementError
**Purpose**: Maximum drawdown limit violations
**Status**: Complete

**Implemented Methods:**

### Implementation: `RiskCalculationError` ✅

**Inherits**: RiskManagementError
**Purpose**: Risk metric calculation failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `CapitalAllocationError` ✅

**Inherits**: RiskManagementError
**Purpose**: Capital allocation rule violations
**Status**: Complete

**Implemented Methods:**

### Implementation: `AllocationError` ✅

**Inherits**: RiskManagementError
**Purpose**: Portfolio allocation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `CircuitBreakerTriggeredError` ✅

**Inherits**: RiskManagementError
**Purpose**: Circuit breaker activation
**Status**: Complete

**Implemented Methods:**

### Implementation: `EmergencyStopError` ✅

**Inherits**: RiskManagementError
**Purpose**: Emergency stop system failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all data-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataValidationError` ✅

**Inherits**: DataError
**Purpose**: Data validation and schema compliance failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataSourceError` ✅

**Inherits**: DataError
**Purpose**: External data source connectivity and reliability issues
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataProcessingError` ✅

**Inherits**: DataError
**Purpose**: Data transformation and processing pipeline failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataCorruptionError` ✅

**Inherits**: DataError
**Purpose**: Data integrity and corruption detection
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataQualityError` ✅

**Inherits**: DataError
**Purpose**: Data quality issues affecting trading decisions
**Status**: Complete

**Implemented Methods:**

### Implementation: `ModelError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all ML model-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ModelLoadError` ✅

**Inherits**: ModelError
**Purpose**: Model loading and initialization failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `ModelInferenceError` ✅

**Inherits**: ModelError
**Purpose**: Model prediction and inference failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `ModelDriftError` ✅

**Inherits**: ModelError
**Purpose**: Model performance drift detection
**Status**: Complete

**Implemented Methods:**

### Implementation: `ModelTrainingError` ✅

**Inherits**: ModelError
**Purpose**: Model training and optimization failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `ModelValidationError` ✅

**Inherits**: ModelError
**Purpose**: Model validation and testing failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `ValidationError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all input and schema validation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ConfigurationError` ✅

**Inherits**: ValidationError
**Purpose**: Configuration file and parameter validation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `SchemaValidationError` ✅

**Inherits**: ValidationError
**Purpose**: Data schema and structure validation failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `InputValidationError` ✅

**Inherits**: ValidationError
**Purpose**: Function and API input parameter validation failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `BusinessRuleValidationError` ✅

**Inherits**: ValidationError
**Purpose**: Business logic and rule validation failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExecutionError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all order execution and trading errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `OrderRejectionError` ✅

**Inherits**: ExecutionError
**Purpose**: Order rejected by exchange
**Status**: Complete

**Implemented Methods:**

### Implementation: `SlippageError` ✅

**Inherits**: ExecutionError
**Purpose**: Excessive slippage during order execution
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExecutionTimeoutError` ✅

**Inherits**: ExecutionError
**Purpose**: Order execution timeout errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExecutionPartialFillError` ✅

**Inherits**: ExecutionError
**Purpose**: Partial order fill handling errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `NetworkError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all network and communication errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ConnectionError` ✅

**Inherits**: NetworkError
**Purpose**: Network connection establishment failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `TimeoutError` ✅

**Inherits**: NetworkError
**Purpose**: Network operation timeout errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `WebSocketError` ✅

**Inherits**: NetworkError
**Purpose**: WebSocket connection and messaging errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `StateConsistencyError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all state management and consistency errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `StateError` ✅

**Inherits**: StateConsistencyError
**Purpose**: General state management errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `StateCorruptionError` ✅

**Inherits**: StateConsistencyError
**Purpose**: State data corruption detected
**Status**: Complete

**Implemented Methods:**

### Implementation: `StateLockError` ✅

**Inherits**: StateConsistencyError
**Purpose**: State lock acquisition failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `SynchronizationError` ✅

**Inherits**: StateConsistencyError
**Purpose**: Real-time synchronization errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ConflictError` ✅

**Inherits**: StateConsistencyError
**Purpose**: State conflict errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `SecurityError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all security-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `AuthenticationError` ✅

**Inherits**: SecurityError
**Purpose**: Authentication failures and credential issues
**Status**: Complete

**Implemented Methods:**

### Implementation: `AuthorizationError` ✅

**Inherits**: SecurityError
**Purpose**: Authorization and permission failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `EncryptionError` ✅

**Inherits**: SecurityError
**Purpose**: Encryption and decryption failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `TokenValidationError` ✅

**Inherits**: SecurityError
**Purpose**: Token validation and parsing failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `StrategyError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all strategy-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `StrategyConfigurationError` ✅

**Inherits**: StrategyError
**Purpose**: Strategy configuration errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `SignalGenerationError` ✅

**Inherits**: StrategyError
**Purpose**: Signal generation failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `ArbitrageError` ✅

**Inherits**: StrategyError
**Purpose**: Arbitrage strategy errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestError` ✅

**Inherits**: TradingBotError
**Purpose**: Backtesting operation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestConfigurationError` ✅

**Inherits**: BacktestError
**Purpose**: Backtesting configuration errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestDataError` ✅

**Inherits**: BacktestError
**Purpose**: Backtesting data-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestExecutionError` ✅

**Inherits**: BacktestError
**Purpose**: Backtesting execution errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestServiceError` ✅

**Inherits**: BacktestError
**Purpose**: Backtesting service unavailability errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestValidationError` ✅

**Inherits**: BacktestError
**Purpose**: Backtesting validation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestResultError` ✅

**Inherits**: BacktestError
**Purpose**: Backtesting result processing errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestMetricsError` ✅

**Inherits**: BacktestError
**Purpose**: Backtesting metrics calculation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestStrategyError` ✅

**Inherits**: BacktestError
**Purpose**: Backtesting strategy-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `DatabaseError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all database-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `DatabaseConnectionError` ✅

**Inherits**: DatabaseError
**Purpose**: Database connection failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `DatabaseQueryError` ✅

**Inherits**: DatabaseError
**Purpose**: Database query failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `CircuitBreakerOpenError` ✅

**Inherits**: TradingBotError
**Purpose**: Circuit breaker is open due to too many failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `MaxRetriesExceededError` ✅

**Inherits**: TradingBotError
**Purpose**: Maximum retry attempts exceeded
**Status**: Complete

**Implemented Methods:**

### Implementation: `ErrorCodeRegistry` ✅

**Purpose**: Registry for all error codes in the system
**Status**: Complete

**Implemented Methods:**
- `validate_code(cls, error_code: str) -> bool` - Line 1879

### Implementation: `ExchangeErrorMapper` ✅

**Purpose**: Maps exchange-specific errors to standardized exceptions
**Status**: Complete

**Implemented Methods:**
- `map_error(cls, exchange: str, error_data: dict[str, Any]) -> TradingBotError` - Line 1943
- `map_binance_error(cls, error_data: dict[str, Any]) -> TradingBotError` - Line 1966
- `map_coinbase_error(cls, error_data: dict[str, Any]) -> TradingBotError` - Line 2011
- `map_okx_error(cls, error_data: dict[str, Any]) -> TradingBotError` - Line 2037

### Implementation: `ComponentError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all component-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ServiceError` ✅

**Inherits**: ComponentError
**Purpose**: Service layer errors for BaseService implementations
**Status**: Complete

**Implemented Methods:**

### Implementation: `RepositoryError` ✅

**Inherits**: ComponentError
**Purpose**: Repository layer errors for BaseRepository implementations
**Status**: Complete

**Implemented Methods:**

### Implementation: `FactoryError` ✅

**Inherits**: ComponentError
**Purpose**: Factory pattern errors for BaseFactory implementations
**Status**: Complete

**Implemented Methods:**

### Implementation: `DependencyError` ✅

**Inherits**: ComponentError
**Purpose**: Dependency injection and resolution errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `HealthCheckError` ✅

**Inherits**: ComponentError
**Purpose**: Health check system errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `CircuitBreakerError` ✅

**Inherits**: ComponentError
**Purpose**: Circuit breaker pattern errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `EventError` ✅

**Inherits**: ComponentError
**Purpose**: Event system errors for BaseEventEmitter
**Status**: Complete

**Implemented Methods:**

### Implementation: `EntityNotFoundError` ✅

**Inherits**: DatabaseError
**Purpose**: Entity not found in repository
**Status**: Complete

**Implemented Methods:**

### Implementation: `CreationError` ✅

**Inherits**: FactoryError
**Purpose**: Factory creation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `RegistrationError` ✅

**Inherits**: FactoryError
**Purpose**: Factory registration errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `EventHandlerError` ✅

**Inherits**: EventError
**Purpose**: Event handler execution errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `MonitoringError` ✅

**Inherits**: ComponentError
**Purpose**: Monitoring and metrics collection errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `AnalyticsError` ✅

**Inherits**: ComponentError
**Purpose**: Analytics calculation and processing errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `OptimizationError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for all optimization-related errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ParameterValidationError` ✅

**Inherits**: OptimizationError
**Purpose**: Parameter space validation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `OptimizationTimeoutError` ✅

**Inherits**: OptimizationError
**Purpose**: Optimization process timeout errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ConvergenceError` ✅

**Inherits**: OptimizationError
**Purpose**: Optimization convergence failures
**Status**: Complete

**Implemented Methods:**

### Implementation: `OverfittingError` ✅

**Inherits**: OptimizationError
**Purpose**: Overfitting detection errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `GeneticAlgorithmError` ✅

**Inherits**: OptimizationError
**Purpose**: Genetic algorithm optimization errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `HyperparameterOptimizationError` ✅

**Inherits**: OptimizationError
**Purpose**: Hyperparameter optimization errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `PerformanceError` ✅

**Inherits**: TradingBotError
**Purpose**: Base class for performance optimization errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `CacheError` ✅

**Inherits**: PerformanceError
**Purpose**: Cache operation errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `MemoryOptimizationError` ✅

**Inherits**: PerformanceError
**Purpose**: Memory optimization errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `DatabaseOptimizationError` ✅

**Inherits**: PerformanceError
**Purpose**: Database performance optimization errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ConnectionPoolError` ✅

**Inherits**: PerformanceError
**Purpose**: Connection pool optimization errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `ProfilingError` ✅

**Inherits**: PerformanceError
**Purpose**: Performance profiling errors
**Status**: Complete

**Implemented Methods:**

### Implementation: `EnvironmentMode` ✅

**Inherits**: Enum
**Purpose**: Environment operation modes
**Status**: Complete

### Implementation: `EnvironmentContext` ✅

**Inherits**: BaseModel
**Purpose**: Context information for environment-aware operations
**Status**: Complete

### Implementation: `EnvironmentAwareServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for environment-aware services
**Status**: Complete

**Implemented Methods:**
- `async switch_environment(self, environment: str | ExchangeEnvironment, exchange: str | None = None) -> bool` - Line 48
- `async validate_environment_operation(self, operation: str, context: EnvironmentContext) -> bool` - Line 56
- `get_environment_context(self, exchange: str) -> EnvironmentContext` - Line 64

### Implementation: `EnvironmentAwareServiceMixin` ✅

**Purpose**: Mixin class providing environment-aware functionality to services
**Status**: Complete

**Implemented Methods:**
- `register_environment_switch_callback(self, callback: Callable) -> None` - Line 83
- `unregister_environment_switch_callback(self, callback: Callable) -> None` - Line 88
- `get_environment_context(self, exchange: str) -> EnvironmentContext` - Line 108
- `async switch_environment(self, environment: str | ExchangeEnvironment, exchange: str | None = None) -> bool` - Line 138
- `async validate_environment_operation(self, ...) -> bool` - Line 234
- `get_environment_specific_config(self, exchange: str, config_key: str, default: Any = None) -> Any` - Line 282
- `is_environment_ready(self, exchange: str) -> bool` - Line 297
- `async get_environment_health_status(self) -> dict[str, Any]` - Line 311

### Implementation: `EnvironmentAwareService` ✅

**Inherits**: BaseService, EnvironmentAwareServiceMixin
**Purpose**: Base class for services that need environment awareness
**Status**: Complete

**Implemented Methods:**
- `async get_service_health(self) -> dict[str, Any]` - Line 381

### Implementation: `EnvironmentIntegrationOrchestrator` ✅

**Inherits**: EnvironmentAwareService
**Purpose**: Orchestrates environment-aware integration across all T-Bot services
**Status**: Complete

**Implemented Methods:**
- `register_service(self, ...) -> None` - Line 77
- `async switch_global_environment(self, ...) -> dict[str, Any]` - Line 115
- `async switch_exchange_environment(self, ...) -> dict[str, Any]` - Line 224
- `async get_integrated_health_status(self) -> dict[str, Any]` - Line 349
- `async get_environment_status_summary(self) -> dict[str, Any]` - Line 443
- `async validate_environment_consistency(self) -> dict[str, Any]` - Line 463

### Implementation: `CorrelationContext` ✅

**Purpose**: Context manager for correlation ID tracking
**Status**: Complete

**Implemented Methods:**
- `set_correlation_id(self, correlation_id: str) -> None` - Line 43
- `get_correlation_id(self) -> str | None` - Line 47
- `generate_correlation_id(self) -> str` - Line 51
- `correlation_context(self, correlation_id: str | None = None)` - Line 56

### Implementation: `SecureLogger` ✅

**Purpose**: Logger wrapper that prevents sensitive data from being logged
**Status**: Complete

**Implemented Methods:**
- `info(self, message: str, **kwargs) -> None` - Line 370
- `warning(self, message: str, **kwargs) -> None` - Line 375
- `error(self, message: str, **kwargs) -> None` - Line 380
- `critical(self, message: str, **kwargs) -> None` - Line 385
- `debug(self, message: str, **kwargs) -> None` - Line 390

### Implementation: `PerformanceMonitor` ✅

**Purpose**: Performance monitoring utility for tracking operation metrics
**Status**: Complete

**Implemented Methods:**

### Implementation: `CacheOptimizedList` ✅

**Purpose**: Cache-optimized list implementation
**Status**: Complete

**Implemented Methods:**
- `clear(self)` - Line 49
- `append(self, item)` - Line 53

### Implementation: `DependencyInjectionMixin` ✅

**Purpose**: Simple mixin for dependency injection to avoid circular imports
**Status**: Complete

**Implemented Methods:**
- `get_injector(self)` - Line 77

### Implementation: `MemoryStats` ✅

**Purpose**: Memory usage statistics
**Status**: Complete

**Implemented Methods:**
- `memory_pressure(self) -> float` - Line 99

### Implementation: `ObjectPool` ✅

**Inherits**: Generic[T]
**Purpose**: High-performance object pool for frequently used objects
**Status**: Complete

**Implemented Methods:**
- `borrow(self) -> T` - Line 158
- `return_object(self, obj: T)` - Line 174
- `get_stats(self) -> dict[str, Any]` - Line 200
- `clear(self)` - Line 216

### Implementation: `MemoryLeakDetector` ✅

**Purpose**: Detect and track memory leaks
**Status**: Complete

**Implemented Methods:**
- `async start(self)` - Line 236
- `stop(self)` - Line 322
- `get_leak_report(self) -> dict[str, Any]` - Line 326

### Implementation: `MemoryMappedCache` ✅

**Purpose**: Memory-mapped cache for large datasets
**Status**: Complete

**Implemented Methods:**
- `write_data(self, offset: int, data: bytes) -> bool` - Line 390
- `read_data(self, offset: int, length: int) -> bytes | None` - Line 406
- `close(self)` - Line 420

### Implementation: `HighPerformanceMemoryManager` ✅

**Inherits**: DependencyInjectionMixin
**Purpose**: Comprehensive memory management system
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self)` - Line 556
- `get_pool(self, pool_name: str) -> ObjectPool | None` - Line 662
- `borrow_object(self, pool_name: str)` - Line 666
- `return_object(self, pool_name: str, obj)` - Line 673
- `track_object(self, obj)` - Line 679
- `get_memory_stats(self) -> MemoryStats` - Line 683
- `get_performance_report(self) -> dict[str, Any]` - Line 687
- `async stop_monitoring(self)` - Line 750
- `cleanup(self)` - Line 775
- `get_dependencies(self) -> list[str]` - Line 792

### Implementation: `MemoryCategory` ✅

**Inherits**: Enum
**Purpose**: Categories of memory usage for optimization
**Status**: Complete

### Implementation: `GCStrategy` ✅

**Inherits**: Enum
**Purpose**: Garbage collection strategies
**Status**: Complete

### Implementation: `MemoryStats` ✅

**Purpose**: Memory usage statistics
**Status**: Complete

### Implementation: `ObjectPoolStats` ✅

**Purpose**: Statistics for object pools
**Status**: Complete

### Implementation: `ObjectPool` ✅

**Purpose**: High-performance object pool for frequently allocated objects
**Status**: Complete

**Implemented Methods:**
- `async acquire(self) -> Any` - Line 134
- `async release(self, obj: Any) -> None` - Line 160
- `get_stats(self) -> ObjectPoolStats` - Line 176
- `clear(self) -> None` - Line 180

### Implementation: `MemoryProfiler` ✅

**Purpose**: Advanced memory profiler for detecting leaks and optimization opportunities
**Status**: Complete

**Implemented Methods:**
- `start_profiling(self) -> None` - Line 200
- `stop_profiling(self) -> None` - Line 207
- `take_snapshot(self, label: str | None = None) -> None` - Line 214
- `detect_leaks(self) -> list[dict[str, Any]]` - Line 227
- `get_top_allocations(self, limit: int = 10) -> list[dict[str, Any]]` - Line 252

### Implementation: `GarbageCollectionOptimizer` ✅

**Purpose**: Garbage collection optimizer for trading workloads
**Status**: Complete

**Implemented Methods:**
- `set_strategy(self, strategy: GCStrategy) -> None` - Line 289
- `disable_gc_during_trading(self) -> None` - Line 311
- `enable_gc_after_trading(self) -> None` - Line 319
- `force_collection(self) -> dict[str, Any]` - Line 332
- `get_gc_stats(self) -> dict[str, Any]` - Line 371

### Implementation: `MemoryOptimizer` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive memory optimizer for the T-Bot trading system
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 454
- `async acquire_pooled_object(self, pool_name: str) -> Any` - Line 646
- `async release_pooled_object(self, pool_name: str, obj: Any) -> None` - Line 653
- `track_category_usage(self, category: MemoryCategory, size_mb: float) -> None` - Line 671
- `add_alert_callback(self, callback: Callable) -> None` - Line 675
- `remove_alert_callback(self, callback: Callable) -> None` - Line 679
- `async optimize_for_trading_operation(self) -> None` - Line 684
- `async cleanup_after_trading_operation(self) -> None` - Line 704
- `async get_memory_report(self) -> dict[str, Any]` - Line 718
- `async force_memory_optimization(self) -> dict[str, Any]` - Line 770
- `async cleanup(self) -> None` - Line 821

### Implementation: `TradingMemoryContext` ✅

**Purpose**: Context manager for trading operations with memory optimization
**Status**: Complete

**Implemented Methods:**

### Implementation: `MetricType` ✅

**Inherits**: Enum
**Purpose**: Types of performance metrics
**Status**: Complete

### Implementation: `OperationType` ✅

**Inherits**: Enum
**Purpose**: Types of operations being monitored
**Status**: Complete

### Implementation: `AlertLevel` ✅

**Inherits**: Enum
**Purpose**: Alert severity levels
**Status**: Complete

### Implementation: `PerformanceMetric` ✅

**Purpose**: Individual performance metric data point
**Status**: Complete

### Implementation: `LatencyStats` ✅

**Purpose**: Latency statistics for an operation type
**Status**: Complete

### Implementation: `ThroughputStats` ✅

**Purpose**: Throughput statistics for an operation type
**Status**: Complete

### Implementation: `ResourceUsageStats` ✅

**Purpose**: System resource usage statistics
**Status**: Complete

### Implementation: `PerformanceAlert` ✅

**Purpose**: Performance alert definition
**Status**: Complete

### Implementation: `PerformanceThresholds` ✅

**Purpose**: Performance thresholds for alerting
**Status**: Complete

### Implementation: `LatencyTracker` ✅

**Purpose**: High-precision latency tracking for trading operations
**Status**: Complete

**Implemented Methods:**
- `async record_latency(self, latency_ms: float, metadata: dict[str, Any] | None = None) -> None` - Line 179
- `get_stats(self) -> LatencyStats` - Line 215
- `reset_stats(self) -> None` - Line 219

### Implementation: `ThroughputTracker` ✅

**Purpose**: Throughput tracking for operations per second
**Status**: Complete

**Implemented Methods:**
- `async record_operation(self) -> None` - Line 233
- `get_stats(self) -> ThroughputStats` - Line 258

### Implementation: `PrometheusMetricsCollector` ✅

**Purpose**: Prometheus metrics collector for external monitoring
**Status**: Complete

**Implemented Methods:**
- `record_latency(self, ...) -> None` - Line 319
- `increment_operation(self, operation_type: str, status: str, labels: dict[str, str] | None = None) -> None` - Line 328
- `update_resource_usage(self, cpu_percent: float, memory_bytes: float) -> None` - Line 339
- `get_metrics(self) -> str` - Line 344

### Implementation: `PerformanceMonitor` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive performance monitoring system for T-Bot trading operations
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 400
- `async record_operation_start(self, operation_type: OperationType, metadata: dict[str, Any] | None = None) -> str` - Line 447
- `async record_operation_end(self, ...) -> float` - Line 479
- `async record_simple_latency(self, ...) -> None` - Line 528
- `add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None` - Line 853
- `remove_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None` - Line 857
- `async get_performance_summary(self) -> dict[str, Any]` - Line 862
- `get_prometheus_metrics(self) -> str` - Line 915
- `async reset_statistics(self) -> None` - Line 919
- `async cleanup(self) -> None` - Line 941

### Implementation: `OperationTracker` ✅

**Purpose**: Context manager for automatic operation tracking
**Status**: Complete

**Implemented Methods:**

### Implementation: `PerformanceOptimizer` ✅

**Inherits**: BaseComponent
**Purpose**: Integrated performance optimizer that coordinates all optimization components
to achieve optimal tra
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 92
- `optimize_trading_operation(self, operation: 'TradingOperation')` - Line 536
- `async get_performance_report(self) -> dict[str, Any]` - Line 547
- `async force_optimization(self) -> dict[str, Any]` - Line 650
- `async cleanup(self) -> None` - Line 703
- `is_initialized(self) -> bool` - Line 792
- `get_component_status(self) -> dict[str, bool]` - Line 796

### Implementation: `TradingOperation` ✅

**Inherits**: Enum
**Purpose**: Specific trading operations that require optimization
**Status**: Complete

### Implementation: `OptimizationLevel` ✅

**Inherits**: Enum
**Purpose**: Levels of optimization to apply
**Status**: Complete

### Implementation: `OperationProfile` ✅

**Purpose**: Detailed profile of a trading operation
**Status**: Complete

### Implementation: `TradingBenchmark` ✅

**Purpose**: Benchmark results for trading operations
**Status**: Complete

### Implementation: `TradingProfiler` ✅

**Purpose**: High-precision profiler for individual trading operations
**Status**: Complete

**Implemented Methods:**
- `start_profiling(self, enable_memory_tracing: bool = False) -> None` - Line 125
- `stop_profiling(self) -> dict[str, Any]` - Line 140
- `get_performance_summary(self) -> dict[str, Any]` - Line 269

### Implementation: `TradingOperationOptimizer` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive optimizer for critical trading operations
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 348
- `async profile_operation(self, operation: TradingOperation, func: Callable, *args, **kwargs) -> tuple[Any, dict[str, Any]]` - Line 399
- `optimize_function(self, operation: TradingOperation)` - Line 476
- `async get_optimization_report(self) -> dict[str, Any]` - Line 751
- `async force_optimization_analysis(self) -> dict[str, Any]` - Line 800
- `async cleanup(self) -> None` - Line 824

### Implementation: `TradingOperationContext` ✅

**Purpose**: Context manager for automatic trading operation profiling
**Status**: Complete

**Implemented Methods:**

### Implementation: `ResourceType` ✅

**Inherits**: Enum
**Purpose**: Types of resources being managed
**Status**: Complete

### Implementation: `ResourceState` ✅

**Inherits**: Enum
**Purpose**: Resource lifecycle states
**Status**: Complete

### Implementation: `ResourceInfo` ✅

**Purpose**: Information about a managed resource
**Status**: Complete

**Implemented Methods:**
- `touch(self) -> None` - Line 66

### Implementation: `ResourceMonitor` ✅

**Purpose**: Monitors resource usage and detects leaks
**Status**: Complete

**Implemented Methods:**
- `get_memory_usage(self) -> dict[str, Any]` - Line 81
- `get_connection_stats(self) -> dict[str, Any]` - Line 94
- `get_gc_stats(self) -> dict[str, Any]` - Line 120

### Implementation: `ResourceManager` ✅

**Purpose**: Centralized resource lifecycle manager
**Status**: Complete

**Implemented Methods:**
- `configure(self, ...) -> None` - Line 190
- `async start(self)` - Line 222
- `async stop(self)` - Line 236
- `register_resource(self, ...) -> str` - Line 264
- `async unregister_resource(self, resource_id: str)` - Line 302
- `touch_resource(self, resource_id: str)` - Line 348
- `async cleanup_all_resources(self)` - Line 357
- `get_resource_stats(self) -> dict[str, Any]` - Line 566

### Implementation: `ServiceManager` ✅

**Purpose**: Centralized service manager for dependency resolution and lifecycle management
**Status**: Complete

**Implemented Methods:**
- `register_service(self, ...) -> None` - Line 40
- `async start_all_services(self) -> None` - Line 255
- `async stop_all_services(self) -> None` - Line 302
- `get_service(self, service_name: str) -> Any` - Line 357
- `is_service_running(self, service_name: str) -> bool` - Line 378
- `get_running_services(self) -> list[str]` - Line 382
- `async restart_service(self, service_name: str) -> None` - Line 386
- `async health_check_all(self) -> dict[str, Any]` - Line 400

### Implementation: `TaskState` ✅

**Inherits**: Enum
**Purpose**: Task lifecycle states
**Status**: Complete

### Implementation: `TaskPriority` ✅

**Inherits**: Enum
**Purpose**: Task priority levels
**Status**: Complete

### Implementation: `TaskInfo` ✅

**Purpose**: Information about a managed task
**Status**: Complete

### Implementation: `TaskManager` ✅

**Purpose**: Comprehensive task lifecycle manager
**Status**: Complete

**Implemented Methods:**
- `async start(self)` - Line 126
- `async stop(self)` - Line 145
- `async create_task(self, ...) -> str` - Line 182
- `async cancel_task(self, task_id: str) -> bool` - Line 232
- `get_task_stats(self) -> dict[str, Any]` - Line 550
- `get_task_info(self, task_id: str) -> dict[str, Any] | None` - Line 577

### Implementation: `AlertSeverity` ✅

**Inherits**: Enum
**Purpose**: Alert severity levels for monitoring and alerting system
**Status**: Complete

### Implementation: `TradingMode` ✅

**Inherits**: Enum
**Purpose**: Trading mode enumeration for different execution environments
**Status**: Complete

**Implemented Methods:**
- `is_real_money(self) -> bool` - Line 61
- `allows_testing(self) -> bool` - Line 65
- `from_string(cls, value: str) -> 'TradingMode'` - Line 70

### Implementation: `ExchangeType` ✅

**Inherits**: Enum
**Purpose**: Exchange types for API integration and rate limiting coordination
**Status**: Complete

**Implemented Methods:**
- `get_rate_limit(self) -> int` - Line 117
- `supports_websocket(self) -> bool` - Line 128
- `get_base_url(self) -> str` - Line 132

### Implementation: `MarketType` ✅

**Inherits**: Enum
**Purpose**: Market types for different trading venues and instruments
**Status**: Complete

**Implemented Methods:**
- `requires_margin(self) -> bool` - Line 164
- `has_expiration(self) -> bool` - Line 168
- `supports_leverage(self) -> bool` - Line 172

### Implementation: `RequestType` ✅

**Inherits**: Enum
**Purpose**: Request types for API coordination and rate limiting
**Status**: Complete

**Implemented Methods:**
- `get_priority(self) -> int` - Line 207
- `is_modifying_operation(self) -> bool` - Line 223

### Implementation: `ConnectionType` ✅

**Inherits**: Enum
**Purpose**: WebSocket connection types for different data streams
**Status**: Complete

**Implemented Methods:**
- `is_public_stream(self) -> bool` - Line 258
- `requires_authentication(self) -> bool` - Line 270
- `get_update_frequency(self) -> str` - Line 274

### Implementation: `ValidationLevel` ✅

**Inherits**: Enum
**Purpose**: Data validation severity levels used across the system
**Status**: Complete

**Implemented Methods:**
- `should_halt_system(self) -> bool` - Line 315
- `requires_immediate_attention(self) -> bool` - Line 319
- `get_numeric_value(self) -> int` - Line 323

### Implementation: `ValidationResult` ✅

**Inherits**: Enum
**Purpose**: Data validation result enumeration with enhanced functionality
**Status**: Complete

**Implemented Methods:**
- `is_success(self) -> bool` - Line 359
- `is_failure(self) -> bool` - Line 363
- `should_proceed(self) -> bool` - Line 367
- `get_severity(self) -> ValidationLevel` - Line 371

### Implementation: `BaseValidatedModel` ✅

**Inherits**: BaseModel
**Purpose**: Enhanced base model with comprehensive validation and utilities
**Status**: Complete

**Implemented Methods:**
- `mark_updated(self) -> None` - Line 402
- `to_dict(self) -> dict[str, Any]` - Line 406
- `to_json(self) -> str` - Line 410
- `from_dict(cls, data: dict[str, Any]) -> 'BaseValidatedModel'` - Line 415
- `from_json(cls, json_str: str) -> 'BaseValidatedModel'` - Line 420
- `add_metadata(self, key: str, value: Any) -> None` - Line 424
- `get_metadata(self, key: str, default: Any = None) -> Any` - Line 429
- `has_metadata(self, key: str) -> bool` - Line 433
- `model_dump_json(self, **kwargs) -> str` - Line 449

### Implementation: `FinancialBaseModel` ✅

**Inherits**: BaseValidatedModel
**Purpose**: Base model for financial data with Decimal precision handling
**Status**: Complete

**Implemented Methods:**
- `convert_financial_floats(cls, v: Any, info) -> Any` - Line 476
- `to_dict_with_decimals(self) -> dict[str, Any]` - Line 497
- `validate_financial_precision(self) -> bool` - Line 522
- `model_dump_json(self, **kwargs) -> str` - Line 544

### Implementation: `BotStatus` ✅

**Inherits**: Enum
**Purpose**: Bot operational status
**Status**: Complete

### Implementation: `BotType` ✅

**Inherits**: Enum
**Purpose**: Bot type classification
**Status**: Complete

### Implementation: `BotPriority` ✅

**Inherits**: Enum
**Purpose**: Bot execution priority
**Status**: Complete

### Implementation: `ResourceType` ✅

**Inherits**: Enum
**Purpose**: System resource types
**Status**: Complete

### Implementation: `BotConfiguration` ✅

**Inherits**: BaseModel
**Purpose**: Bot configuration parameters
**Status**: Complete

**Implemented Methods:**
- `bot_name(self) -> str` - Line 120

### Implementation: `BotMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Bot performance and resource metrics
**Status**: Complete

### Implementation: `BotState` ✅

**Inherits**: BaseModel
**Purpose**: Bot runtime state
**Status**: Complete

### Implementation: `ResourceAllocation` ✅

**Inherits**: BaseModel
**Purpose**: Resource allocation for bots
**Status**: Complete

### Implementation: `CapitalFundFlow` ✅

**Inherits**: BaseModel
**Purpose**: Extended fund flow for capital management operations
**Status**: Complete

### Implementation: `CapitalCurrencyExposure` ✅

**Inherits**: BaseModel
**Purpose**: Extended currency exposure for capital management
**Status**: Complete

### Implementation: `CapitalExchangeAllocation` ✅

**Inherits**: BaseModel
**Purpose**: Extended exchange allocation for capital management
**Status**: Complete

### Implementation: `ExtendedCapitalProtection` ✅

**Inherits**: BaseModel
**Purpose**: Extended capital protection with additional fields
**Status**: Complete

### Implementation: `ExtendedWithdrawalRule` ✅

**Inherits**: BaseModel
**Purpose**: Extended withdrawal rule for fund flow manager
**Status**: Complete

### Implementation: `QualityLevel` ✅

**Inherits**: Enum
**Purpose**: Data quality level classification
**Status**: Complete

### Implementation: `DriftType` ✅

**Inherits**: Enum
**Purpose**: Data drift type classification
**Status**: Complete

### Implementation: `IngestionMode` ✅

**Inherits**: Enum
**Purpose**: Data ingestion mode
**Status**: Complete

### Implementation: `PipelineStatus` ✅

**Inherits**: Enum
**Purpose**: Data pipeline status
**Status**: Complete

### Implementation: `ProcessingStep` ✅

**Inherits**: Enum
**Purpose**: Data processing pipeline steps
**Status**: Complete

### Implementation: `StorageMode` ✅

**Inherits**: Enum
**Purpose**: Data storage mode
**Status**: Complete

### Implementation: `ErrorPattern` ✅

**Purpose**: Common error patterns in data processing
**Status**: Complete

### Implementation: `MLMarketData` ✅

**Inherits**: BaseModel
**Purpose**: Market data structure for ML processing
**Status**: Complete

### Implementation: `PredictionResult` ✅

**Inherits**: BaseModel
**Purpose**: ML prediction result structure
**Status**: Complete

### Implementation: `FeatureSet` ✅

**Inherits**: BaseModel
**Purpose**: Feature set for ML models
**Status**: Complete

### Implementation: `ExecutionAlgorithm` ✅

**Inherits**: Enum
**Purpose**: Execution algorithm types
**Status**: Complete

### Implementation: `ExecutionStatus` ✅

**Inherits**: Enum
**Purpose**: Execution status
**Status**: Complete

### Implementation: `SlippageType` ✅

**Inherits**: Enum
**Purpose**: Slippage classification
**Status**: Complete

### Implementation: `ExecutionInstruction` ✅

**Inherits**: BaseModel
**Purpose**: Execution instruction for order placement
**Status**: Complete

### Implementation: `ExecutionResult` ✅

**Inherits**: BaseModel
**Purpose**: Result of execution algorithm
**Status**: Complete

**Implemented Methods:**
- `fill_percentage(self) -> Decimal` - Line 127
- `is_complete(self) -> bool` - Line 136

### Implementation: `SlippageMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Slippage analysis metrics
**Status**: Complete

### Implementation: `ExchangeStatus` ✅

**Inherits**: Enum
**Purpose**: Exchange operational status
**Status**: Complete

### Implementation: `MarketData` ✅

**Inherits**: BaseModel
**Purpose**: Market data snapshot
**Status**: Complete

**Implemented Methods:**
- `price(self) -> Decimal` - Line 41
- `high_price(self) -> Decimal` - Line 46
- `low_price(self) -> Decimal` - Line 51
- `open_price(self) -> Decimal` - Line 56
- `close_price(self) -> Decimal` - Line 61
- `bid(self) -> Decimal | None` - Line 66
- `ask(self) -> Decimal | None` - Line 71

### Implementation: `Ticker` ✅

**Inherits**: BaseModel
**Purpose**: Market ticker information
**Status**: Complete

**Implemented Methods:**
- `spread(self) -> Decimal` - Line 97
- `spread_percent(self) -> Decimal` - Line 102

### Implementation: `OrderBookLevel` ✅

**Inherits**: BaseModel
**Purpose**: Single level in order book
**Status**: Complete

### Implementation: `OrderBook` ✅

**Inherits**: BaseModel
**Purpose**: Order book snapshot
**Status**: Complete

**Implemented Methods:**
- `best_bid(self) -> OrderBookLevel | None` - Line 130
- `best_ask(self) -> OrderBookLevel | None` - Line 135
- `spread(self) -> Decimal | None` - Line 140
- `get_depth(self, side: str, levels: int = 5) -> Decimal` - Line 146

### Implementation: `Trade` ✅

**Inherits**: BaseModel
**Purpose**: Represents a trade executed on an exchange
**Status**: Complete

### Implementation: `ExchangeGeneralInfo` ✅

**Inherits**: BaseModel
**Purpose**: General exchange information and capabilities
**Status**: Complete

### Implementation: `ExchangeInfo` ✅

**Inherits**: BaseModel
**Purpose**: Exchange trading rules and information for a specific symbol
**Status**: Complete

**Implemented Methods:**
- `round_price(self, price: Decimal) -> Decimal` - Line 197
- `round_quantity(self, quantity: Decimal) -> Decimal` - Line 201
- `validate_order(self, price: Decimal, quantity: Decimal) -> bool` - Line 205

### Implementation: `RiskLevel` ✅

**Inherits**: Enum
**Purpose**: Risk level classification
**Status**: Complete

### Implementation: `PositionSizeMethod` ✅

**Inherits**: Enum
**Purpose**: Position sizing methodology
**Status**: Complete

### Implementation: `CircuitBreakerStatus` ✅

**Inherits**: Enum
**Purpose**: Circuit breaker status
**Status**: Complete

### Implementation: `CircuitBreakerType` ✅

**Inherits**: Enum
**Purpose**: Circuit breaker trigger type
**Status**: Complete

### Implementation: `EmergencyAction` ✅

**Inherits**: Enum
**Purpose**: Emergency action types
**Status**: Complete

### Implementation: `AllocationStrategy` ✅

**Inherits**: Enum
**Purpose**: Capital allocation strategy
**Status**: Complete

### Implementation: `RiskMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Risk metrics for positions and strategies
**Status**: Complete

### Implementation: `PositionLimits` ✅

**Inherits**: BaseModel
**Purpose**: Position size and risk limits
**Status**: Complete

### Implementation: `RiskLimits` ✅

**Inherits**: BaseModel
**Purpose**: Risk limits configuration
**Status**: Complete

### Implementation: `RiskAlert` ✅

**Inherits**: BaseModel
**Purpose**: Risk alert notification
**Status**: Complete

### Implementation: `CircuitBreakerEvent` ✅

**Inherits**: BaseModel
**Purpose**: Circuit breaker trigger event
**Status**: Complete

### Implementation: `CapitalAllocation` ✅

**Inherits**: BaseModel
**Purpose**: Capital allocation for strategies and positions
**Status**: Complete

### Implementation: `FundFlow` ✅

**Inherits**: BaseModel
**Purpose**: Fund flow tracking for deposits and withdrawals
**Status**: Complete

### Implementation: `CapitalMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Overall capital and portfolio metrics
**Status**: Complete

### Implementation: `CurrencyExposure` ✅

**Inherits**: BaseModel
**Purpose**: Currency exposure tracking
**Status**: Complete

### Implementation: `ExchangeAllocation` ✅

**Inherits**: BaseModel
**Purpose**: Capital allocation across exchanges
**Status**: Complete

### Implementation: `WithdrawalRule` ✅

**Inherits**: BaseModel
**Purpose**: Automated withdrawal rules
**Status**: Complete

### Implementation: `CapitalProtection` ✅

**Inherits**: BaseModel
**Purpose**: Capital protection settings
**Status**: Complete

### Implementation: `PortfolioState` ✅

**Inherits**: BaseModel
**Purpose**: Complete portfolio state representation
**Status**: Complete

### Implementation: `PortfolioMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Unified portfolio metrics model for cross-module consistency
**Status**: Complete

### Implementation: `StrategyType` ✅

**Inherits**: Enum
**Purpose**: Strategy type enumeration
**Status**: Complete

### Implementation: `StrategyStatus` ✅

**Inherits**: Enum
**Purpose**: Strategy operational status
**Status**: Complete

### Implementation: `MarketRegime` ✅

**Inherits**: Enum
**Purpose**: Market regime classification
**Status**: Complete

### Implementation: `NewsSentiment` ✅

**Inherits**: Enum
**Purpose**: News sentiment classification
**Status**: Complete

### Implementation: `SocialSentiment` ✅

**Inherits**: Enum
**Purpose**: Social media sentiment classification
**Status**: Complete

### Implementation: `StrategyConfig` ✅

**Inherits**: BaseModel
**Purpose**: Strategy configuration parameters
**Status**: Complete

### Implementation: `StrategyMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Strategy performance metrics
**Status**: Complete

**Implemented Methods:**
- `update_win_rate(self) -> None` - Line 144
- `calculate_profit_factor(self) -> Decimal` - Line 149

### Implementation: `RegimeChangeEvent` ✅

**Inherits**: BaseModel
**Purpose**: Market regime change event
**Status**: Complete

### Implementation: `SignalDirection` ✅

**Inherits**: Enum
**Purpose**: Signal direction for trading decisions
**Status**: Complete

### Implementation: `OrderSide` ✅

**Inherits**: Enum
**Purpose**: Order side for buy/sell operations
**Status**: Complete

### Implementation: `PositionSide` ✅

**Inherits**: Enum
**Purpose**: Position side for long/short positions
**Status**: Complete

### Implementation: `PositionStatus` ✅

**Inherits**: Enum
**Purpose**: Position status
**Status**: Complete

### Implementation: `OrderType` ✅

**Inherits**: Enum
**Purpose**: Order type for different execution strategies
**Status**: Complete

### Implementation: `OrderStatus` ✅

**Inherits**: Enum
**Purpose**: Order status in exchange systems
**Status**: Complete

### Implementation: `TimeInForce` ✅

**Inherits**: Enum
**Purpose**: Time in force for order execution
**Status**: Complete

### Implementation: `TradeState` ✅

**Inherits**: Enum
**Purpose**: Trade lifecycle states
**Status**: Complete

### Implementation: `Signal` ✅

**Inherits**: BaseModel
**Purpose**: Trading signal with direction and metadata - consistent validation patterns
**Status**: Complete

**Implemented Methods:**
- `validate_strength(cls, v: Decimal) -> Decimal` - Line 117
- `validate_symbol(cls, v: str) -> str` - Line 137
- `validate_timestamp(cls, v: datetime) -> datetime` - Line 189

### Implementation: `OrderRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request to create an order
**Status**: Complete

**Implemented Methods:**
- `validate_symbol(cls, v: str) -> str` - Line 219
- `validate_quantity(cls, v: Decimal) -> Decimal` - Line 227
- `validate_price(cls, v: Decimal | None) -> Decimal | None` - Line 256
- `validate_quote_quantity(cls, v: Decimal | None) -> Decimal | None` - Line 283

### Implementation: `OrderResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response from order creation
**Status**: Complete

**Implemented Methods:**
- `id(self) -> str` - Line 328
- `remaining_quantity(self) -> Decimal` - Line 333

### Implementation: `Order` ✅

**Inherits**: BaseModel
**Purpose**: Complete order information
**Status**: Complete

**Implemented Methods:**
- `is_filled(self) -> bool` - Line 365
- `is_active(self) -> bool` - Line 369

### Implementation: `Position` ✅

**Inherits**: BaseModel
**Purpose**: Trading position information
**Status**: Complete

**Implemented Methods:**
- `is_open(self) -> bool` - Line 391
- `calculate_pnl(self, current_price: Decimal) -> Decimal` - Line 395

### Implementation: `Trade` ✅

**Inherits**: BaseModel
**Purpose**: Executed trade information
**Status**: Complete

### Implementation: `Balance` ✅

**Inherits**: BaseModel
**Purpose**: Account balance information
**Status**: Complete

**Implemented Methods:**
- `free(self) -> Decimal` - Line 431

### Implementation: `ArbitrageOpportunity` ✅

**Inherits**: BaseModel
**Purpose**: Arbitrage opportunity data structure
**Status**: Complete

**Implemented Methods:**
- `validate_prices(cls, v: Decimal) -> Decimal` - Line 453
- `validate_quantity(cls, v: Decimal) -> Decimal` - Line 477
- `is_expired(self) -> bool` - Line 500
- `calculate_profit(self) -> Decimal` - Line 506

### Implementation: `ValidatorInterface` 🔧

**Inherits**: ABC
**Purpose**: Base interface for all validators
**Status**: Abstract Base Class

**Implemented Methods:**
- `validate(self, data: Any, **kwargs) -> bool` - Line 19

### Implementation: `CompositeValidator` ✅

**Inherits**: ValidatorInterface
**Purpose**: Composite validator that chains multiple validators
**Status**: Complete

**Implemented Methods:**
- `validate(self, data: Any, **kwargs) -> bool` - Line 48

### Implementation: `ValidatorRegistry` ✅

**Purpose**: Central registry for all validators
**Status**: Complete

**Implemented Methods:**
- `register_validator(self, name: str, validator: ValidatorInterface) -> None` - Line 70
- `register_validator_class(self, name: str, validator_class: type[ValidatorInterface]) -> None` - Line 81
- `register_rule(self, ...) -> None` - Line 94
- `get_validator(self, name: str) -> ValidatorInterface` - Line 117
- `validate(self, data_type: str, data: Any, validator_name: str | None = None, **kwargs) -> bool` - Line 142
- `create_composite_validator(self, validator_names: list[str]) -> CompositeValidator` - Line 178
- `clear(self) -> None` - Line 191

### Implementation: `RangeValidator` ✅

**Inherits**: ValidatorInterface
**Purpose**: Validator for numeric ranges
**Status**: Complete

**Implemented Methods:**
- `validate(self, data: Any, **kwargs) -> bool` - Line 213

### Implementation: `LengthValidator` ✅

**Inherits**: ValidatorInterface
**Purpose**: Validator for string/collection length
**Status**: Complete

**Implemented Methods:**
- `validate(self, data: Any, **kwargs) -> bool` - Line 241

### Implementation: `PatternValidator` ✅

**Inherits**: ValidatorInterface
**Purpose**: Validator for regex patterns
**Status**: Complete

**Implemented Methods:**
- `validate(self, data: Any, **kwargs) -> bool` - Line 271

### Implementation: `TypeValidator` ✅

**Inherits**: ValidatorInterface
**Purpose**: Validator for type checking
**Status**: Complete

**Implemented Methods:**
- `validate(self, data: Any, **kwargs) -> bool` - Line 294

### Implementation: `WebSocketState` ✅

**Inherits**: Enum
**Purpose**: WebSocket connection states
**Status**: Complete

### Implementation: `WebSocketManager` ✅

**Purpose**: Async WebSocket connection manager with proper resource cleanup
**Status**: Complete

**Implemented Methods:**
- `async connection(self) -> AsyncGenerator['WebSocketManager', None]` - Line 109
- `async send_message(self, message: dict) -> None` - Line 378
- `set_message_callback(self, callback: Callable[[dict], None]) -> None` - Line 426
- `set_error_callback(self, callback: Callable[[Exception], None]) -> None` - Line 430
- `set_disconnect_callback(self, callback: Callable[[], None]) -> None` - Line 434
- `is_connected(self) -> bool` - Line 439
- `get_stats(self) -> dict[str, Any]` - Line 445

### Implementation: `StreamType` ✅

**Inherits**: Enum
**Purpose**: WebSocket stream types for different data subscriptions
**Status**: Complete

### Implementation: `MessagePriority` ✅

**Inherits**: Enum
**Purpose**: Message priority levels for WebSocket message handling
**Status**: Complete

**Implemented Methods:**

## COMPLETE API REFERENCE

### File: component.py

**Key Imports:**
- `from src.core.base.interfaces import Configurable`
- `from src.core.base.interfaces import HealthCheckable`
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.interfaces import Injectable`

#### Class: `BaseComponent`

**Inherits**: Lifecycle, HealthCheckable, Injectable, Loggable, Monitorable, Configurable
**Purpose**: Enhanced base component with complete lifecycle management

```python
class BaseComponent(Lifecycle, HealthCheckable, Injectable, Loggable, Monitorable, Configurable):
    def __init__(self, ...)  # Line 70
    def name(self) -> str  # Line 122
    def component_name(self) -> str  # Line 127
    def logger(self) -> Any  # Line 132
    def correlation_id(self) -> str  # Line 137
    def is_running(self) -> bool  # Line 142
    def is_starting(self) -> bool  # Line 147
    def is_stopping(self) -> bool  # Line 152
    def uptime(self) -> float  # Line 157
    async def start(self) -> None  # Line 165
    async def stop(self) -> None  # Line 211
    async def restart(self) -> None  # Line 254
    async def health_check(self) -> HealthCheckResult  # Line 273
    async def ready_check(self) -> HealthCheckResult  # Line 335
    async def live_check(self) -> HealthCheckResult  # Line 351
    def configure(self, config: ConfigDict) -> None  # Line 384
    def get_config(self) -> ConfigDict  # Line 409
    def validate_config(self, config: ConfigDict) -> bool  # Line 413
    def configure_dependencies(self, container: Any) -> None  # Line 426
    async def initialize(self) -> None  # Line 439
    async def cleanup(self) -> None  # Line 443
    def get_dependencies(self) -> list[str]  # Line 447
    def add_dependency(self, dependency_name: str) -> None  # Line 451
    def remove_dependency(self, dependency_name: str) -> None  # Line 455
    def _auto_resolve_dependencies(self) -> None  # Line 459
    def get_metrics(self) -> dict[str, Any]  # Line 487
    def reset_metrics(self) -> None  # Line 501
    async def lifecycle_context(self)  # Line 515
    async def _do_start(self) -> None  # Line 524
    async def _do_stop(self) -> None  # Line 528
    async def _health_check_internal(self) -> HealthStatus  # Line 532
    async def _readiness_check_internal(self) -> HealthCheckResult  # Line 536
    def _on_config_changed(self, old_config: ConfigDict, new_config: ConfigDict) -> None  # Line 543
    def __repr__(self) -> str  # Line 548
```

### File: events.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.base.interfaces import EventEmitter`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.exceptions import EventError`
- `from src.core.exceptions import EventHandlerError`

#### Class: `EventPriority`

**Inherits**: Enum
**Purpose**: Event priority levels

```python
class EventPriority(Enum):
```

#### Class: `EventMetadata`

**Purpose**: Metadata for event tracking

```python
class EventMetadata:
```

#### Class: `EventContext`

**Purpose**: Context for event processing

```python
class EventContext:
```

#### Class: `EventHandler`

**Purpose**: Wrapper for event handler functions

```python
class EventHandler:
    def __init__(self, ...)  # Line 70
    async def __call__(self, event_context: EventContext) -> bool  # Line 90
```

#### Class: `BaseEventEmitter`

**Inherits**: BaseComponent, EventEmitter
**Purpose**: Base event emitter implementing the observer pattern

```python
class BaseEventEmitter(BaseComponent, EventEmitter):
    def __init__(self, ...)  # Line 153
    def event_metrics(self) -> dict[str, Any]  # Line 214
    def on(self, ...) -> EventHandler  # Line 224
    def off(self, event: str, callback: Callable | EventHandler | None = None) -> None  # Line 271
    def once(self, ...) -> EventHandler  # Line 315
    def on_pattern(self, ...) -> EventHandler  # Line 357
    def on_global(self, ...) -> EventHandler  # Line 399
    def remove_all_listeners(self, event: str | None = None) -> None  # Line 434
    def emit(self, ...) -> None  # Line 470
    async def emit_async(self, ...) -> None  # Line 510
    async def _emit_sync(self, ...) -> None  # Line 537
    def _collect_handlers(self, event: str) -> list[EventHandler]  # Line 631
    async def _execute_handler(self, handler: EventHandler, event_context: EventContext) -> bool  # Line 651
    def _remove_handler(self, target_handler: EventHandler) -> None  # Line 686
    def _add_to_history(self, event_context: EventContext) -> None  # Line 704
    def _record_event_metrics(self, event: str, start_time: datetime, handlers_count: int, success_count: int) -> None  # Line 712
    def get_event_history(self, event_type: str | None = None, limit: int | None = None) -> list[EventContext]  # Line 748
    def get_handler_info(self) -> dict[str, Any]  # Line 773
    def get_events_summary(self) -> dict[str, Any]  # Line 816
    async def _health_check_internal(self) -> HealthStatus  # Line 843
    def configure_processing(self, ...) -> None  # Line 878
    def get_metrics(self) -> dict[str, Any]  # Line 916
    def reset_metrics(self) -> None  # Line 922
    async def _do_stop(self) -> None  # Line 945
    def _transform_event_data(self, data: Any) -> dict[str, Any] | None  # Line 973
    def _validate_cross_module_boundary(self, data: Any, source: str, event: str) -> None  # Line 1027
```

### File: factory.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.base.interfaces import FactoryComponent`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.exceptions import CreationError`
- `from src.core.exceptions import RegistrationError`

#### Class: `DependencyInjectionMixin`

**Purpose**: Simple mixin for dependency injection to avoid circular imports

```python
class DependencyInjectionMixin:
    def __init__(self)  # Line 32
    def get_injector(self)  # Line 36
```

#### Class: `CreatorFunction`

**Inherits**: Protocol, Generic[T]
**Purpose**: Protocol for creator functions

```python
class CreatorFunction(Protocol, Generic[T]):
    def __call__(self, *args: Any, **kwargs: Any) -> T  # Line 51
```

#### Class: `BaseFactory`

**Inherits**: BaseComponent, FactoryComponent, DependencyInjectionMixin, Generic[T]
**Purpose**: Base factory implementing the factory pattern

```python
class BaseFactory(BaseComponent, FactoryComponent, DependencyInjectionMixin, Generic[T]):
    def __init__(self, ...)  # Line 86
    def configure_dependencies(self, container: Any) -> None  # Line 139
    def _inject_dependencies_into_kwargs(self, target_callable: Callable, kwargs: dict[str, Any]) -> dict[str, Any]  # Line 162
    def _resolve_dependency(self, type_name: str, param_name: str) -> Any  # Line 224
    def product_type(self) -> type[T]  # Line 272
    def creation_metrics(self) -> dict[str, Any]  # Line 277
    def register(self, ...) -> None  # Line 282
    def register_interface(self, ...) -> None  # Line 335
    def unregister(self, name: str) -> None  # Line 386
    def update_creator_config(self, name: str, config: dict[str, Any]) -> None  # Line 425
    def create(self, name: str, *args: Any, **kwargs: Any) -> T  # Line 450
    def create_batch(self, requests: list[dict[str, Any]]) -> list[T]  # Line 557
    def _execute_creator(self, ...) -> T  # Line 611
    def _inject_dependencies(self, creator_name: str, kwargs: dict[str, Any]) -> dict[str, Any]  # Line 641
    def _validate_creator(self, name: str, creator: type[T] | CreatorFunction[T] | Callable[Ellipsis, T]) -> None  # Line 657
    def _validate_product(self, creator_name: str, instance: Any) -> None  # Line 707
    def list_registered(self) -> list[str]  # Line 725
    def is_registered(self, name: str) -> bool  # Line 734
    def get_creator_info(self, name: str) -> dict[str, Any] | None  # Line 746
    def get_all_creator_info(self) -> dict[str, dict[str, Any]]  # Line 771
    def get_singleton(self, name: str) -> T | None  # Line 780
    def clear_singletons(self) -> None  # Line 792
    def reset_singleton(self, name: str) -> None  # Line 814
    async def _health_check_internal(self) -> HealthStatus  # Line 844
    def _record_creation_success(self, creator_name: str, execution_time: float) -> None  # Line 885
    def _record_creation_failure(self, creator_name: str, execution_time: float, error: Exception) -> None  # Line 903
    def get_metrics(self) -> dict[str, Any]  # Line 911
    def reset_metrics(self) -> None  # Line 923
    def configure_validation(self, validate_creators: bool = True, validate_products: bool = True) -> None  # Line 936
    async def _do_stop(self) -> None  # Line 959
```

### File: health.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.base.interfaces import HealthCheckable`
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.exceptions import HealthCheckError`

#### Class: `HealthCheckType`

**Inherits**: Enum
**Purpose**: Types of health checks

```python
class HealthCheckType(Enum):
```

#### Class: `ComponentHealthInfo`

**Purpose**: Health information for a registered component

```python
class ComponentHealthInfo:
    def __init__(self, ...)  # Line 35
```

#### Class: `HealthCheckManager`

**Inherits**: BaseComponent
**Purpose**: Centralized health check manager for all system components

```python
class HealthCheckManager(BaseComponent):
    def __init__(self, ...)  # Line 96
    def health_metrics(self) -> dict[str, Any]  # Line 151
    def register_component(self, ...) -> None  # Line 159
    def unregister_component(self, name: str) -> None  # Line 210
    def enable_component_monitoring(self, name: str) -> None  # Line 245
    def disable_component_monitoring(self, name: str) -> None  # Line 261
    async def check_component_health(self, ...) -> HealthCheckResult  # Line 280
    async def check_all_components(self, ...) -> dict[str, HealthCheckResult]  # Line 387
    async def get_overall_health(self) -> HealthCheckResult  # Line 482
    def add_alert_callback(self, callback: Callable[[str, HealthCheckResult], Awaitable[None]]) -> None  # Line 551
    def remove_alert_callback(self, callback: Callable[[str, HealthCheckResult], Awaitable[None]]) -> None  # Line 568
    async def _handle_check_result(self, component_name: str, result: HealthCheckResult) -> None  # Line 575
    def _start_component_monitoring(self, component_name: str) -> None  # Line 616
    def _get_cached_result(self, component_name: str, check_type: HealthCheckType) -> HealthCheckResult | None  # Line 654
    def _cache_result(self, ...) -> None  # Line 677
    def clear_cache(self, component_name: str | None = None) -> None  # Line 685
    def get_component_info(self, component_name: str) -> dict[str, Any] | None  # Line 712
    def get_all_component_info(self) -> dict[str, dict[str, Any]]  # Line 747
    def list_components(self, enabled_only: bool = False) -> list[str]  # Line 755
    def _record_check_metrics(self, health_info: ComponentHealthInfo, execution_time: float, success: bool) -> None  # Line 770
    def get_metrics(self) -> dict[str, Any]  # Line 797
    def reset_metrics(self) -> None  # Line 803
    def configure_checks(self, ...) -> None  # Line 826
    def configure_alerts(self, ...) -> None  # Line 866
    async def _do_start(self) -> None  # Line 889
    async def _do_stop(self) -> None  # Line 903
    async def _health_check_internal(self) -> HealthStatus  # Line 928
```

### File: interfaces.py

#### Class: `HealthStatus`

**Inherits**: Enum
**Purpose**: Health status enumeration for components

```python
class HealthStatus(Enum):
```

#### Class: `HealthCheckResult`

**Purpose**: Result of a health check operation

```python
class HealthCheckResult:
    def __init__(self, ...)  # Line 36
    def healthy(self) -> bool  # Line 49
    def component(self) -> str  # Line 54
    def to_dict(self) -> dict[str, Any]  # Line 58
```

#### Class: `Lifecycle`

**Inherits**: Protocol
**Purpose**: Protocol for components with lifecycle management

```python
class Lifecycle(Protocol):
    async def start(self) -> None  # Line 71
    async def stop(self) -> None  # Line 75
    async def restart(self) -> None  # Line 79
    def is_running(self) -> bool  # Line 84
```

#### Class: `HealthCheckable`

**Inherits**: Protocol
**Purpose**: Protocol for components that support health checks

```python
class HealthCheckable(Protocol):
    async def health_check(self) -> HealthCheckResult  # Line 92
    async def ready_check(self) -> HealthCheckResult  # Line 96
    async def live_check(self) -> HealthCheckResult  # Line 100
```

#### Class: `Injectable`

**Inherits**: Protocol
**Purpose**: Protocol for dependency injection support

```python
class Injectable(Protocol):
    def configure_dependencies(self, container: Any) -> None  # Line 108
    def get_dependencies(self) -> list[str]  # Line 112
```

#### Class: `Loggable`

**Inherits**: Protocol
**Purpose**: Protocol for components with structured logging

```python
class Loggable(Protocol):
    def logger(self) -> Any  # Line 121
    def correlation_id(self) -> str | None  # Line 126
```

#### Class: `Monitorable`

**Inherits**: Protocol
**Purpose**: Protocol for components with metrics and monitoring

```python
class Monitorable(Protocol):
    def get_metrics(self) -> dict[str, int | float | str]  # Line 134
    def reset_metrics(self) -> None  # Line 138
```

#### Class: `Configurable`

**Inherits**: Protocol
**Purpose**: Protocol for components with configuration support

```python
class Configurable(Protocol):
    def configure(self, config: ConfigDict) -> None  # Line 146
    def get_config(self) -> ConfigDict  # Line 150
    def validate_config(self, config: ConfigDict) -> bool  # Line 154
```

#### Class: `Repository`

**Inherits**: Protocol
**Purpose**: Protocol for repository pattern implementation

```python
class Repository(Protocol):
    async def create(self, entity: Any) -> Any  # Line 162
    async def get_by_id(self, entity_id: Any) -> Any | None  # Line 166
    async def update(self, entity: Any) -> Any  # Line 170
    async def delete(self, entity_id: Any) -> bool  # Line 174
    async def list(self, ...) -> list[Any]  # Line 178
    async def count(self, filters: dict[str, Any] | None = None) -> int  # Line 187
```

#### Class: `Factory`

**Inherits**: Protocol
**Purpose**: Protocol for factory pattern implementation

```python
class Factory(Protocol):
    def register(self, name: str, creator_func: Any) -> None  # Line 195
    def unregister(self, name: str) -> None  # Line 199
    def create(self, name: str, *args: Any, **kwargs: Any) -> Any  # Line 203
    def list_registered(self) -> list[str]  # Line 207
```

#### Class: `EventEmitter`

**Inherits**: Protocol
**Purpose**: Protocol for event emission and subscription

```python
class EventEmitter(Protocol):
    def emit(self, event: str, data: Any = None) -> None  # Line 215
    def on(self, event: str, callback: Any) -> Any  # Line 219
    def off(self, event: str, callback: Any | None = None) -> None  # Line 223
    def once(self, event: str, callback: Any) -> Any  # Line 227
    def remove_all_listeners(self, event: str | None = None) -> None  # Line 231
```

#### Class: `DIContainer`

**Inherits**: Protocol
**Purpose**: Protocol for dependency injection container

```python
class DIContainer(Protocol):
    def register(self, interface: type, implementation: type | Any, singleton: bool = False) -> None  # Line 239
    def resolve(self, interface: type) -> Any  # Line 248
    def is_registered(self, interface: type) -> bool  # Line 252
    def register_factory(self, name: str, factory_func: Any, singleton: bool = False) -> None  # Line 256
```

#### Class: `AsyncContextManager`

**Inherits**: Protocol
**Purpose**: Protocol for async context managers

```python
class AsyncContextManager(Protocol):
    async def __aenter__(self) -> Any  # Line 269
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None  # Line 273
```

#### Class: `ServiceComponent`

**Inherits**: Protocol
**Purpose**: Combined protocol for service layer components

```python
class ServiceComponent(Protocol):
    async def start(self) -> None  # Line 283
    async def stop(self) -> None  # Line 284
    async def restart(self) -> None  # Line 285
    def is_running(self) -> bool  # Line 287
    async def health_check(self) -> HealthCheckResult  # Line 290
    async def ready_check(self) -> HealthCheckResult  # Line 291
    async def live_check(self) -> HealthCheckResult  # Line 292
    def configure_dependencies(self, container: Any) -> None  # Line 295
    def get_dependencies(self) -> list[str]  # Line 296
    def logger(self) -> Any  # Line 300
    def correlation_id(self) -> str | None  # Line 302
    def get_metrics(self) -> dict[str, int | float | str]  # Line 305
    def reset_metrics(self) -> None  # Line 306
    def configure(self, config: ConfigDict) -> None  # Line 309
    def get_config(self) -> ConfigDict  # Line 310
    def validate_config(self, config: ConfigDict) -> bool  # Line 311
```

#### Class: `RepositoryComponent`

**Inherits**: Protocol
**Purpose**: Combined protocol for repository layer components

```python
class RepositoryComponent(Protocol):
    async def create(self, entity: Any) -> Any  # Line 318
    async def get_by_id(self, entity_id: Any) -> Any | None  # Line 319
    async def update(self, entity: Any) -> Any  # Line 320
    async def delete(self, entity_id: Any) -> bool  # Line 321
    async def list(self, ...) -> list[Any]  # Line 322
    async def count(self, filters: dict[str, Any] | None = None) -> int  # Line 328
    async def health_check(self) -> HealthCheckResult  # Line 331
    async def ready_check(self) -> HealthCheckResult  # Line 332
    async def live_check(self) -> HealthCheckResult  # Line 333
    def configure_dependencies(self, container: Any) -> None  # Line 336
    def get_dependencies(self) -> builtins.list[str]  # Line 337
    def logger(self) -> Any  # Line 341
    def correlation_id(self) -> str | None  # Line 343
```

#### Class: `FactoryComponent`

**Inherits**: Protocol
**Purpose**: Combined protocol for factory components

```python
class FactoryComponent(Protocol):
    def register(self, name: str, creator_func: Any) -> None  # Line 350
    def unregister(self, name: str) -> None  # Line 351
    def create(self, name: str, *args: Any, **kwargs: Any) -> Any  # Line 352
    def list_registered(self) -> list[str]  # Line 353
    def configure_dependencies(self, container: Any) -> None  # Line 356
    def get_dependencies(self) -> list[str]  # Line 357
    def logger(self) -> Any  # Line 361
    def correlation_id(self) -> str | None  # Line 363
```

#### Class: `WebServiceInterface`

**Inherits**: Protocol
**Purpose**: Base interface for web service implementations

```python
class WebServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 370
    async def cleanup(self) -> None  # Line 374
```

#### Class: `TradingServiceInterface`

**Inherits**: WebServiceInterface
**Purpose**: Interface for trading operations

```python
class TradingServiceInterface(WebServiceInterface):
    async def place_order(self, ...) -> str  # Line 383
    async def cancel_order(self, order_id: str) -> bool  # Line 395
    async def get_positions(self) -> list[Any]  # Line 400
```

#### Class: `BotManagementServiceInterface`

**Inherits**: WebServiceInterface
**Purpose**: Interface for bot management operations

```python
class BotManagementServiceInterface(WebServiceInterface):
    async def create_bot(self, config: Any) -> str  # Line 409
    async def start_bot(self, bot_id: str) -> bool  # Line 414
    async def stop_bot(self, bot_id: str) -> bool  # Line 419
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]  # Line 424
    async def list_bots(self) -> list[dict[str, Any]]  # Line 429
    async def get_all_bots_status(self) -> dict[str, Any]  # Line 434
    async def delete_bot(self, bot_id: str, force: bool = False) -> bool  # Line 439
```

#### Class: `MarketDataServiceInterface`

**Inherits**: WebServiceInterface
**Purpose**: Interface for market data operations

```python
class MarketDataServiceInterface(WebServiceInterface):
    async def get_ticker(self, symbol: str) -> Any  # Line 448
    async def subscribe_to_ticker(self, symbol: str, callback: Any) -> None  # Line 453
    async def unsubscribe_from_ticker(self, symbol: str) -> None  # Line 458
```

#### Class: `PortfolioServiceInterface`

**Inherits**: WebServiceInterface
**Purpose**: Interface for portfolio operations

```python
class PortfolioServiceInterface(WebServiceInterface):
    async def get_balance(self) -> dict[str, Any]  # Line 467
    async def get_portfolio_summary(self) -> dict[str, Any]  # Line 472
    async def get_pnl_report(self, start_date: Any, end_date: Any) -> dict[str, Any]  # Line 477
```

#### Class: `RiskServiceInterface`

**Inherits**: WebServiceInterface
**Purpose**: Interface for risk management operations

```python
class RiskServiceInterface(WebServiceInterface):
    async def validate_order(self, symbol: str, side: str, amount: Any, price: Any | None = None) -> dict[str, Any]  # Line 486
    async def get_risk_metrics(self) -> dict[str, Any]  # Line 493
    async def update_risk_limits(self, limits: dict[str, Any]) -> bool  # Line 498
```

#### Class: `StrategyServiceInterface`

**Inherits**: WebServiceInterface
**Purpose**: Interface for strategy operations

```python
class StrategyServiceInterface(WebServiceInterface):
    async def list_strategies(self) -> list[dict[str, Any]]  # Line 507
    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]  # Line 512
    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool  # Line 517
```

#### Class: `CacheClientInterface`

**Inherits**: Protocol
**Purpose**: Interface for cache client implementations (Redis, etc

```python
class CacheClientInterface(Protocol):
    async def connect(self) -> None  # Line 526
    async def disconnect(self) -> None  # Line 530
    async def ping(self) -> bool  # Line 534
    async def get(self, key: str, namespace: str = 'cache') -> Any | None  # Line 538
    async def set(self, key: str, value: Any, ttl: int | None = None, namespace: str = 'cache') -> bool  # Line 542
    async def delete(self, key: str, namespace: str = 'cache') -> bool  # Line 546
    async def exists(self, key: str, namespace: str = 'cache') -> bool  # Line 550
    async def expire(self, key: str, ttl: int, namespace: str = 'cache') -> bool  # Line 554
    async def info(self) -> dict[str, Any]  # Line 558
    def _get_namespaced_key(self, key: str, namespace: str) -> str  # Line 562
    def client(self) -> Any  # Line 567
```

#### Class: `DatabaseServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for database service implementations

```python
class DatabaseServiceInterface(Protocol):
    async def start(self) -> None  # Line 576
    async def stop(self) -> None  # Line 580
    async def health_check(self) -> HealthCheckResult  # Line 584
    def get_performance_metrics(self) -> dict[str, Any]  # Line 588
    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> Any  # Line 592
    async def get_connection_pool_status(self) -> dict[str, Any]  # Line 596
```

### File: repository.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.interfaces import RepositoryComponent`
- `from src.core.exceptions import DatabaseConnectionError`
- `from src.core.exceptions import DataValidationError`

#### Class: `BaseRepository`

**Inherits**: BaseComponent, RepositoryComponent, Generic[T, K]
**Purpose**: Base repository implementing the repository pattern

```python
class BaseRepository(BaseComponent, RepositoryComponent, Generic[T, K]):
    def __init__(self, ...)  # Line 74
    def entity_type(self) -> type[T]  # Line 136
    def key_type(self) -> type[K]  # Line 141
    def query_metrics(self) -> dict[str, Any]  # Line 146
    def set_connection_pool(self, connection_pool: Any) -> None  # Line 151
    def get_connection(self) -> AbstractAsyncContextManager[Any]  # Line 164
    def set_transaction_manager(self, transaction_manager: Any) -> None  # Line 182
    async def create(self, entity: T) -> T  # Line 196
    async def get_by_id(self, entity_id: K) -> T | None  # Line 244
    async def update(self, entity: T) -> T  # Line 291
    async def delete(self, entity_id: K) -> bool  # Line 346
    async def list(self, ...) -> list[T]  # Line 396
    async def count(self, filters: dict[str, Any] | None = None) -> int  # Line 465
    async def bulk_create(self, entities: builtins.list[T]) -> builtins.list[T]  # Line 511
    async def bulk_update(self, entities: builtins.list[T]) -> builtins.list[T]  # Line 562
    async def execute_in_transaction(self, operation_func: Callable[Ellipsis, Any], *args, **kwargs) -> Any  # Line 614
    def configure_cache(self, enabled: bool = True, ttl: int = 300) -> None  # Line 647
    def _get_from_cache(self, key: str) -> Any | None  # Line 672
    def _set_cache(self, key: str, value: Any) -> None  # Line 689
    def _invalidate_cache(self, key: str) -> None  # Line 697
    def _invalidate_cache_pattern(self, pattern: str) -> None  # Line 702
    def _clear_cache(self) -> None  # Line 708
    async def _health_check_internal(self) -> HealthStatus  # Line 714
    async def _execute_with_monitoring(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any  # Line 757
    def _record_query_metrics(self, operation: str, start_time: datetime) -> None  # Line 796
    def get_metrics(self) -> dict[str, Any]  # Line 809
    def reset_metrics(self) -> None  # Line 815
    def _create_list_cache_key(self, ...) -> str  # Line 829
    def _validate_entity(self, entity: T) -> None  # Line 848
    def _extract_entity_id(self, entity: T) -> K | None  # Line 864
    async def _create_entity(self, entity: T) -> T  # Line 871
    async def _get_entity_by_id(self, entity_id: K) -> T | None  # Line 876
    async def _update_entity(self, entity: T) -> T | None  # Line 881
    async def _delete_entity(self, entity_id: K) -> bool  # Line 886
    async def _list_entities(self, ...) -> builtins.list[T]  # Line 891
    async def _count_entities(self, filters: dict[str, Any] | None) -> int  # Line 903
    async def _bulk_create_entities(self, entities: builtins.list[T]) -> builtins.list[T]  # Line 908
    async def _bulk_update_entities(self, entities: builtins.list[T]) -> builtins.list[T]  # Line 916
    async def _test_connection(self, connection: Any) -> bool  # Line 925
    async def _repository_health_check(self) -> HealthStatus  # Line 934
    def _validate_entity_data(self, entity: T) -> None  # Line 938
```

### File: service.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.interfaces import ServiceComponent`
- `from src.core.exceptions import DependencyError`
- `from src.core.exceptions import ServiceError`

#### Class: `BaseService`

**Inherits**: BaseComponent, ServiceComponent
**Purpose**: Base service class implementing the service layer pattern

```python
class BaseService(BaseComponent, ServiceComponent):
    def __init__(self, ...)  # Line 58
    def service_metrics(self) -> dict[str, Any]  # Line 106
    async def execute_with_monitoring(self, operation_name: str, operation_func: Any, *args, **kwargs) -> Any  # Line 119
    async def _execute_with_retry(self, operation_func: Any, operation_name: str, *args, **kwargs) -> Any  # Line 223
    def _check_circuit_breaker(self) -> bool  # Line 286
    def _record_operation_success(self, operation_name: str, execution_time: float) -> None  # Line 313
    def _record_operation_failure(self, operation_name: str, execution_time: float, error: Exception) -> None  # Line 345
    def _add_to_history(self, record: dict[str, Any]) -> None  # Line 383
    def configure_dependencies(self, dependency_injector: Any) -> None  # Line 392
    def resolve_dependency(self, dependency_name: str) -> Any  # Line 408
    async def _health_check_internal(self) -> HealthStatus  # Line 449
    async def _service_health_check(self) -> HealthStatus  # Line 506
    def validate_config(self, config: ConfigDict) -> bool  # Line 511
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 527
    def get_metrics(self) -> dict[str, Any]  # Line 532
    def reset_metrics(self) -> None  # Line 538
    def get_operation_history(self, limit: int | None = None) -> list[dict[str, Any]]  # Line 553
    def configure_circuit_breaker(self, enabled: bool = True, threshold: int = 5, timeout: int = 60) -> None  # Line 568
    def configure_retry(self, ...) -> None  # Line 594
    def reset_circuit_breaker(self) -> None  # Line 624
    def _propagate_service_error_consistently(self, error: Exception, operation: str, execution_time: float) -> None  # Line 635
```

#### Class: `TransactionalService`

**Inherits**: BaseService
**Purpose**: Base service with transaction management support

```python
class TransactionalService(BaseService):
    def __init__(self, *args: Any, **kwargs: Any) -> None  # Line 710
    def set_transaction_manager(self, transaction_manager: Any) -> None  # Line 715
    async def execute_in_transaction(self, operation_name: str, operation_func: Any, *args, **kwargs) -> Any  # Line 723
```

### File: cache_decorators.py

#### Functions:

```python
def _create_cache_key(*args, **kwargs) -> str  # Line 13
def cached(...)  # Line 24
def _create_async_cached_wrapper(func: Callable, config: dict) -> Callable  # Line 65
def _create_sync_cached_wrapper(func: Callable, config: dict) -> Callable  # Line 86
def _generate_cache_key(func: Callable, key_generator: Callable | None, *args, **kwargs) -> str  # Line 97
async def _try_get_from_cache(cache_manager, cache_key: str, namespace: str)  # Line 106
async def _execute_and_cache_async(func: Callable, cache_manager, cache_key: str, config: dict, *args, **kwargs)  # Line 116
async def _execute_with_lock(...)  # Line 149
async def _store_result_in_cache(...)  # Line 161
async def _safe_invalidate_cache(cache_manager, cache_key: str, namespace: str)  # Line 173
def _make_func_async(func: Callable) -> Callable  # Line 184
def _run_in_event_loop(async_func: Callable, *args, **kwargs)  # Line 193
def cache_invalidate(...)  # Line 205
def _create_async_invalidation_wrapper(func: Callable, config: dict) -> Callable  # Line 230
def _create_sync_invalidation_wrapper(func: Callable, config: dict) -> Callable  # Line 246
async def _perform_cache_invalidation(config: dict)  # Line 257
async def _invalidate_specific_keys(cache_manager, keys, namespace: str)  # Line 275
async def _invalidate_by_patterns(cache_manager, patterns, namespace: str)  # Line 282
def cache_warm(...)  # Line 289
def _create_async_warming_wrapper(func: Callable, config: dict) -> Callable  # Line 318
def _create_sync_warming_wrapper(func: Callable, config: dict) -> Callable  # Line 334
async def _perform_cache_warming(warming_data, config: dict)  # Line 345
def _build_warming_functions(warming_data: dict, warming_keys) -> dict  # Line 361
def cache_market_data(symbol_arg_name: str = 'symbol', ttl: int = 5)  # Line 374
def cache_risk_metrics(bot_id_arg_name: str = 'bot_id', ttl: int = 60)  # Line 385
def cache_strategy_signals(strategy_id_arg_name: str = 'strategy_id', ttl: int = 300)  # Line 396
def cache_bot_status(bot_id_arg_name: str = 'bot_id', ttl: int = 30)  # Line 413
```

### File: cache_keys.py

#### Class: `CacheKeys`

**Purpose**: Centralized cache key management for consistent naming patterns

```python
class CacheKeys:
    def _build_key(cls, namespace: str, *parts: Any) -> str  # Line 26
    def state_snapshot(cls, bot_id: str) -> str  # Line 33
    def trade_lifecycle(cls, trade_id: str) -> str  # Line 38
    def state_checkpoint(cls, checkpoint_id: str) -> str  # Line 43
    def risk_metrics(cls, bot_id: str, timeframe: str = '1h') -> str  # Line 49
    def position_limits(cls, bot_id: str, symbol: str) -> str  # Line 54
    def correlation_matrix(cls, timeframe: str = '1h') -> str  # Line 59
    def var_calculation(cls, portfolio_id: str, confidence: str = '95') -> str  # Line 64
    def market_price(cls, symbol: str, exchange: str = 'all') -> str  # Line 70
    def order_book(cls, symbol: str, exchange: str, depth: int = 20) -> str  # Line 75
    def technical_indicator(cls, symbol: str, indicator: str, period: int) -> str  # Line 80
    def ohlcv_data(cls, symbol: str, timeframe: str, exchange: str = 'all') -> str  # Line 85
    def active_orders(cls, bot_id: str, symbol: str = 'all') -> str  # Line 91
    def order_history(cls, bot_id: str, page: int = 1) -> str  # Line 96
    def execution_state(cls, algorithm: str, bot_id: str) -> str  # Line 101
    def order_lock(cls, symbol: str, bot_id: str) -> str  # Line 106
    def strategy_signals(cls, strategy_id: str, symbol: str) -> str  # Line 112
    def strategy_params(cls, strategy_id: str) -> str  # Line 117
    def backtest_results(cls, strategy_id: str, config_hash: str) -> str  # Line 122
    def strategy_performance(cls, strategy_id: str, timeframe: str = '1d') -> str  # Line 127
    def bot_config(cls, bot_id: str) -> str  # Line 133
    def bot_status(cls, bot_id: str) -> str  # Line 138
    def resource_allocation(cls, bot_id: str) -> str  # Line 143
    def bot_session(cls, bot_id: str, session_id: str) -> str  # Line 148
    def api_response(cls, endpoint: str, user_id: str = 'anonymous', **params: Any) -> str  # Line 154
    def user_session(cls, user_id: str, session_id: str) -> str  # Line 160
    def auth_token(cls, user_id: str, token_hash: str) -> str  # Line 165
    def cache_stats(cls, namespace: str) -> str  # Line 171
    def performance_metrics(cls, component: str, timeframe: str = '5m') -> str  # Line 176
    def time_window_key(cls, base_key: str, window_minutes: int = 5) -> str  # Line 182
    def daily_key(cls, base_key: str) -> str  # Line 189
    def hourly_key(cls, base_key: str) -> str  # Line 195
```

### File: cache_manager.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.base.interfaces import CacheClientInterface`
- `from src.core.exceptions import CacheError`

#### Class: `DependencyInjectionMixin`

**Purpose**: Simple mixin for dependency injection

```python
class DependencyInjectionMixin:
    def __init__(self)  # Line 20
    def get_injector(self)  # Line 23
```

#### Class: `ConnectionManagerMixin`

**Purpose**: Simple connection manager mixin

```python
class ConnectionManagerMixin:
    def __init__(self)  # Line 33
```

#### Class: `ResourceCleanupMixin`

**Purpose**: Simple resource cleanup mixin

```python
class ResourceCleanupMixin:
    def __init__(self)  # Line 41
    async def cleanup_resources(self)  # Line 44
```

#### Class: `LoggingHelperMixin`

**Purpose**: Simple logging helper mixin

```python
class LoggingHelperMixin:
    def __init__(self)  # Line 52
```

#### Class: `CacheManager`

**Inherits**: BaseComponent, DependencyInjectionMixin, ConnectionManagerMixin, ResourceCleanupMixin, LoggingHelperMixin
**Purpose**: Advanced cache manager with:
- Distributed locking for critical operations
- Cache warming strategie

```python
class CacheManager(BaseComponent, DependencyInjectionMixin, ConnectionManagerMixin, ResourceCleanupMixin, LoggingHelperMixin):
    def __init__(self, ...)  # Line 75
    async def _ensure_client(self)  # Line 101
    async def _is_client_healthy(self) -> bool  # Line 125
    async def _reconnect_with_retries(self)  # Line 137
    def _get_ttl(self, data_type: str = 'default') -> int  # Line 189
    def _hash_key(self, key: str) -> str  # Line 193
    async def get(self, ...) -> Any  # Line 197
    async def set(self, ...) -> bool  # Line 255
    async def delete(self, key: str, namespace: str = 'cache') -> bool  # Line 287
    async def exists(self, key: str, namespace: str = 'cache') -> bool  # Line 308
    async def expire(self, key: str, ttl: int, namespace: str = 'cache') -> bool  # Line 324
    async def get_many(self, keys: list[str], namespace: str = 'cache') -> dict[str, Any]  # Line 342
    async def set_many(self, ...) -> bool  # Line 386
    async def acquire_lock(self, resource: str, timeout: int | None = None, namespace: str = 'locks') -> str | None  # Line 435
    async def release_lock(self, resource: str, lock_value: str, namespace: str = 'locks') -> bool  # Line 463
    async def with_lock(self, resource: str, func: Callable, *args, **kwargs)  # Line 492
    async def warm_cache(self, ...)  # Line 521
    async def _warm_single_async(self, key: str, func: Callable)  # Line 552
    async def _warm_single_sync(self, key: str, func: Callable)  # Line 562
    async def invalidate_pattern(self, pattern: str, namespace: str = 'cache')  # Line 573
    async def health_check(self) -> Any  # Line 595
    async def cleanup(self) -> None  # Line 633
    async def shutdown(self) -> None  # Line 677
    def get_dependencies(self) -> list[str]  # Line 706
    def __del__(self)  # Line 710
```

#### Functions:

```python
def get_cache_manager(redis_client: CacheClientInterface | None = None, config: Any | None = None) -> CacheManager  # Line 720
def create_cache_manager_factory(config: Any | None = None) -> Callable[[], CacheManager]  # Line 730
```

### File: cache_metrics.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`

#### Class: `CacheStats`

**Purpose**: Cache statistics for monitoring with memory tracking

```python
class CacheStats:
    def hit_rate(self) -> float  # Line 34
    def miss_rate(self) -> float  # Line 40
```

#### Class: `CacheMetrics`

**Inherits**: BaseComponent
**Purpose**: Cache metrics collector and reporter with memory accounting

```python
class CacheMetrics(BaseComponent):
    def __init__(self) -> None  # Line 48
    def record_hit(self, namespace: str, response_time: float = 0.0)  # Line 66
    def _record_operation(self, namespace: str, op_type: str, response_time: float = 0.0, **kwargs)  # Line 79
    def _cleanup_if_needed(self)  # Line 91
    def _perform_cleanup(self)  # Line 98
    def record_miss(self, namespace: str, response_time: float = 0.0)  # Line 124
    def record_set(self, namespace: str, response_time: float = 0.0, memory_bytes: int = 0)  # Line 137
    def record_delete(self, namespace: str, response_time: float = 0.0, memory_freed: int = 0)  # Line 153
    def record_error(self, namespace: str, error_type: str = 'unknown')  # Line 165
    def get_stats(self, namespace: str | None = None) -> dict[str, Any]  # Line 174
    def get_recent_operations(self, namespace: str, limit: int = 100) -> list[dict[str, Any]]  # Line 233
    def get_performance_summary(self, time_window_minutes: int = 5) -> dict[str, Any]  # Line 238
    def shutdown(self)  # Line 274
    def reset_stats(self, namespace: str | None = None)  # Line 290
    def export_metrics_for_monitoring(self) -> dict[str, Any]  # Line 300
```

#### Functions:

```python
def get_cache_metrics() -> CacheMetrics  # Line 331
```

### File: cache_monitoring.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`

#### Class: `CacheHealthStatus`

**Inherits**: Enum
**Purpose**: Cache health status enumeration

```python
class CacheHealthStatus(Enum):
```

#### Class: `CacheAlert`

**Purpose**: Cache alert definition

```python
class CacheAlert:
```

#### Class: `CacheHealthReport`

**Purpose**: Comprehensive cache health report

```python
class CacheHealthReport:
```

#### Class: `CacheMonitor`

**Inherits**: BaseComponent
**Purpose**: Comprehensive cache monitoring and health management

```python
class CacheMonitor(BaseComponent):
    def __init__(self, config: Any | None = None)  # Line 90
    async def start_monitoring(self) -> None  # Line 121
    async def stop_monitoring(self) -> None  # Line 130
    async def _monitoring_loop(self) -> None  # Line 141
    async def _perform_health_check(self) -> None  # Line 151
    def _update_performance_history(self, health_data: dict[str, Any]) -> None  # Line 167
    async def _check_alerts(self, health_data: dict[str, Any]) -> None  # Line 184
    async def _create_alert(self, ...) -> None  # Line 222
    async def get_health_report(self) -> CacheHealthReport  # Line 254
    def _calculate_overall_health(self, health_data: dict[str, Any], cache_stats: dict[str, Any]) -> CacheHealthStatus  # Line 312
    def _assess_performance_status(self, cache_stats: dict[str, Any]) -> CacheHealthStatus  # Line 335
    def _assess_memory_status(self, health_data: dict[str, Any]) -> CacheHealthStatus  # Line 350
    def _calculate_memory_usage_percent(self, health_data: dict[str, Any]) -> float  # Line 361
    def _calculate_ops_per_second(self) -> float  # Line 376
    def _generate_recommendations(self, health_data: dict[str, Any], cache_stats: dict[str, Any]) -> list[str]  # Line 393
    async def acknowledge_alert(self, alert_id: str) -> bool  # Line 439
    async def clear_acknowledged_alerts(self) -> int  # Line 447
    async def get_performance_trends(self, hours: int = 24) -> dict[str, Any]  # Line 465
    def _analyze_memory_trend(self, history: list[dict[str, Any]]) -> dict[str, Any]  # Line 492
    def _analyze_operation_trend(self, history: list[dict[str, Any]]) -> dict[str, Any]  # Line 502
```

#### Functions:

```python
def get_cache_monitor(config: Any | None = None) -> CacheMonitor  # Line 543
```

### File: cache_warming.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import CacheError`

#### Class: `WarmingStrategy`

**Inherits**: Enum
**Purpose**: Cache warming strategy types

```python
class WarmingStrategy(Enum):
```

#### Class: `WarmingPriority`

**Inherits**: Enum
**Purpose**: Warming priority levels

```python
class WarmingPriority(Enum):
```

#### Class: `WarmingTask`

**Purpose**: Represents a cache warming task

```python
class WarmingTask:
```

#### Class: `CacheWarmer`

**Inherits**: BaseComponent
**Purpose**: Intelligent cache warming system for trading data

```python
class CacheWarmer(BaseComponent):
    def __init__(self, config: Any | None = None)  # Line 88
    async def start_warming(self) -> None  # Line 126
    async def stop_warming(self) -> None  # Line 148
    async def _queue_worker(self, worker_name: str) -> None  # Line 212
    def register_warming_task(self, task: WarmingTask) -> None  # Line 264
    def register_market_data_warming(self, symbols: list[str], exchange: str = 'all') -> None  # Line 272
    def register_bot_state_warming(self, bot_ids: list[str]) -> None  # Line 307
    def register_risk_metrics_warming(self, bot_ids: list[str] | None = None, timeframes: list[str] | None = None) -> None  # Line 339
    def register_strategy_performance_warming(self, strategy_ids: list[str]) -> None  # Line 361
    async def _run_immediate_warming(self) -> None  # Line 378
    async def _queue_warming_batch(self, tasks: list[WarmingTask]) -> None  # Line 405
    async def _warming_scheduler(self) -> None  # Line 412
    async def _should_run_market_hours_task(self, task: WarmingTask, current_time: datetime) -> bool  # Line 475
    async def _should_run_scheduled_task(self, task: WarmingTask, current_time: datetime) -> bool  # Line 502
    async def _should_run_progressive_task(self, task: WarmingTask, current_time: datetime) -> bool  # Line 515
    async def _execute_warming_batch(self, tasks: list[WarmingTask]) -> None  # Line 530
    async def _execute_single_warming_task(self, task: WarmingTask) -> bool  # Line 555
    async def _warm_latest_price(self, symbol: str, exchange: str = 'all') -> dict[str, Any] | None  # Line 625
    async def _warm_order_book(self, symbol: str, exchange: str, depth: int = 20) -> dict[str, Any] | None  # Line 637
    async def _warm_bot_status(self, bot_id: str) -> dict[str, Any] | None  # Line 649
    async def _warm_bot_config(self, bot_id: str) -> dict[str, Any] | None  # Line 658
    async def _warm_risk_metrics(self, bot_id: str, timeframe: str) -> dict[str, Any] | None  # Line 666
    async def _warm_strategy_performance(self, strategy_id: str) -> dict[str, Any] | None  # Line 675
    async def get_warming_status(self) -> dict[str, Any]  # Line 683
    async def warm_critical_data_now(self) -> dict[str, Any]  # Line 709
```

#### Functions:

```python
def get_cache_warmer(config: Any | None = None) -> CacheWarmer  # Line 738
```

### File: unified_cache_layer.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import CacheError`
- `from src.core.logging import get_logger`

#### Class: `CacheLevel`

**Inherits**: Enum
**Purpose**: Cache levels in the hierarchy

```python
class CacheLevel(Enum):
```

#### Class: `CacheStrategy`

**Inherits**: Enum
**Purpose**: Cache management strategies

```python
class CacheStrategy(Enum):
```

#### Class: `DataCategory`

**Inherits**: Enum
**Purpose**: Categories of data for cache optimization

```python
class DataCategory(Enum):
```

#### Class: `CachePolicy`

**Purpose**: Caching policy for different data categories

```python
class CachePolicy:
```

#### Class: `CacheEntry`

**Purpose**: Enhanced cache entry with metadata

```python
class CacheEntry:
    def is_expired(self) -> bool  # Line 122
    def touch(self) -> None  # Line 128
```

#### Class: `CacheStats`

**Purpose**: Comprehensive cache statistics

```python
class CacheStats:
```

#### Class: `CacheInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for cache implementations

```python
class CacheInterface(ABC):
    async def get(self, key: str) -> Any | None  # Line 155
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool  # Line 160
    async def delete(self, key: str) -> bool  # Line 165
    async def clear(self) -> None  # Line 170
    async def get_stats(self) -> CacheStats  # Line 175
```

#### Class: `L1CPUCache`

**Inherits**: CacheInterface
**Purpose**: L1 CPU Cache - Ultra-fast cache optimized for CPU cache efficiency

```python
class L1CPUCache(CacheInterface):
    def __init__(self, max_size: int = 1000)  # Line 188
    async def get(self, key: str) -> Any | None  # Line 199
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool  # Line 245
    def _promote_to_hot_cache(self, key: str, value: Any) -> None  # Line 272
    def _evict_lru(self) -> None  # Line 279
    def _estimate_size(self, value: Any) -> int  # Line 285
    def _update_access_time(self, start_time: float) -> None  # Line 294
    def _update_stats(self) -> None  # Line 299
    async def delete(self, key: str) -> bool  # Line 308
    async def clear(self) -> None  # Line 324
    async def get_stats(self) -> CacheStats  # Line 332
```

#### Class: `L2MemoryCache`

**Inherits**: CacheInterface
**Purpose**: L2 Memory Cache - Application-level memory cache with advanced features

```python
class L2MemoryCache(CacheInterface):
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 100)  # Line 349
    def _get_category_cache_size(self, category: DataCategory) -> int  # Line 363
    def _get_category_ttl(self, category: DataCategory) -> int  # Line 375
    async def get(self, key: str) -> Any | None  # Line 387
    async def set(self, ...) -> bool  # Line 409
    async def delete(self, key: str) -> bool  # Line 440
    async def clear(self) -> None  # Line 460
    async def _evict_by_memory(self) -> None  # Line 467
    def _estimate_size(self, value: Any) -> int  # Line 480
    def _update_access_time(self, start_time: float) -> None  # Line 500
    def _update_stats(self) -> None  # Line 505
    async def get_stats(self) -> CacheStats  # Line 513
```

#### Class: `L3RedisCache`

**Inherits**: CacheInterface
**Purpose**: L3 Redis Cache - Distributed cache for sharing data across instances

```python
class L3RedisCache(CacheInterface):
    def __init__(self, redis_client: redis.Redis)  # Line 530
    async def get(self, key: str) -> Any | None  # Line 536
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool  # Line 559
    async def delete(self, key: str) -> bool  # Line 578
    async def clear(self) -> None  # Line 596
    def _serialize(self, value: Any) -> bytes  # Line 605
    def _deserialize(self, data: bytes) -> Any  # Line 631
    def _update_access_time(self, start_time: float) -> None  # Line 653
    async def get_stats(self) -> CacheStats  # Line 658
```

#### Class: `UnifiedCacheLayer`

**Inherits**: BaseComponent
**Purpose**: Unified caching layer that coordinates all cache levels for optimal performance

```python
class UnifiedCacheLayer(BaseComponent):
    def __init__(self, config: 'Config')  # Line 675
    def _define_cache_policies(self) -> dict[DataCategory, CachePolicy]  # Line 710
    async def initialize(self) -> None  # Line 767
    async def _initialize_cache_levels(self) -> None  # Line 787
    async def _start_background_tasks(self) -> None  # Line 808
    async def get(self, ...) -> Any | None  # Line 823
    async def set(self, ...) -> bool  # Line 877
    async def delete(self, key: str, category: DataCategory = DataCategory.TRADING_DATA) -> bool  # Line 922
    async def invalidate_pattern(self, pattern: str, category: DataCategory) -> int  # Line 948
    async def warm_cache(self, keys: list[str], category: DataCategory, loader: Callable) -> int  # Line 969
    def _get_cache_by_level(self, level: CacheLevel) -> CacheInterface | None  # Line 996
    async def _promote_to_higher_levels(self, key: str, value: Any, hit_level: CacheLevel, policy: CachePolicy) -> None  # Line 1008
    async def _warm_startup_caches(self) -> None  # Line 1032
    async def _cache_warming_loop(self) -> None  # Line 1045
    async def _process_warming_request(self, request: dict[str, Any]) -> None  # Line 1062
    async def _predictive_warming(self) -> None  # Line 1070
    def _track_warming_pattern(self, key: str, category: DataCategory) -> None  # Line 1076
    async def _queue_write_behind(self, key: str, value: Any, category: DataCategory) -> None  # Line 1085
    async def _publish_invalidation(self, key: str, category: DataCategory) -> None  # Line 1090
    def _update_global_response_time(self, start_time: float) -> None  # Line 1098
    async def _statistics_collection_loop(self) -> None  # Line 1105
    async def _collect_statistics(self) -> None  # Line 1117
    async def _cache_maintenance_loop(self) -> None  # Line 1139
    async def _perform_maintenance(self) -> None  # Line 1151
    async def get_comprehensive_stats(self) -> dict[str, Any]  # Line 1162
    async def cleanup(self) -> None  # Line 1184
```

#### Functions:

```python
def time_execution(func)  # Line 47
```

### File: base.py

#### Class: `BaseConfig`

**Inherits**: BaseSettings
**Purpose**: Base configuration class with common patterns

```python
class BaseConfig(BaseSettings):
    def __init__(self, **kwargs: Any) -> None  # Line 24
    def run_validators(self) -> None  # Line 29
    def add_validator(self, validator: Callable) -> None  # Line 34
```

### File: bot_management.py

#### Class: `BotManagementConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for bot management system

```python
class BotManagementConfig(BaseModel):
    def get_resource_limits(self) -> dict  # Line 208
    def get_alert_thresholds(self) -> dict  # Line 212
    def get_coordination_config(self) -> dict  # Line 216
    def get_lifecycle_config(self) -> dict  # Line 225
    def get_monitoring_config(self) -> dict  # Line 234
    def get_connection_timeouts(self) -> dict  # Line 243
    def get_operational_delays(self) -> dict  # Line 247
    def get_circuit_breaker_configs(self) -> dict  # Line 251
    def serialize_decimal(self, value)  # Line 258
```

### File: capital.py

#### Class: `CapitalManagementConfig`

**Inherits**: BaseModel
**Purpose**: Capital management configuration settings

```python
class CapitalManagementConfig(BaseModel):
    def get_available_capital(self) -> Decimal  # Line 186
    def get_emergency_reserve(self) -> Decimal  # Line 191
    def get_max_allocation_for_strategy(self) -> Decimal  # Line 195
    def get_min_allocation_for_strategy(self) -> Decimal  # Line 200
    def model_dump(self, **kwargs: Any) -> dict[str, Any]  # Line 205
```

### File: database.py

#### Class: `DatabaseConfig`

**Inherits**: BaseConfig
**Purpose**: Database configuration for PostgreSQL, Redis, and InfluxDB

```python
class DatabaseConfig(BaseConfig):
    def validate_ports(cls, v: int) -> int  # Line 71
    def validate_pool_size(cls, v: int) -> int  # Line 79
    def validate_redis_db(cls, v: int) -> int  # Line 87
    def postgresql_url(self) -> str  # Line 94
    def redis_url(self) -> str  # Line 108
```

### File: environment.py

#### Class: `TradingEnvironment`

**Inherits**: Enum
**Purpose**: Trading environment types

```python
class TradingEnvironment(Enum):
```

#### Class: `ExchangeEnvironment`

**Inherits**: Enum
**Purpose**: Exchange-specific environment types

```python
class ExchangeEnvironment(Enum):
```

#### Class: `EnvironmentConfig`

**Inherits**: BaseConfig
**Purpose**: Configuration for trading environment switching

```python
class EnvironmentConfig(BaseConfig):
    def validate_global_environment(cls, v)  # Line 173
    def validate_exchange_environment(cls, v)  # Line 184
    def get_exchange_environment(self, exchange_name: str) -> ExchangeEnvironment  # Line 195
    def get_exchange_endpoints(self, exchange_name: str) -> dict[str, str]  # Line 216
    def get_exchange_credentials(self, exchange_name: str) -> dict[str, Any]  # Line 266
    def is_production_environment(self, exchange_name: str) -> bool  # Line 323
    def validate_production_credentials(self, exchange_name: str) -> bool  # Line 328
    def get_environment_summary(self) -> dict[str, Any]  # Line 353
```

### File: exchange.py

#### Class: `ExchangeConfig`

**Inherits**: BaseConfig
**Purpose**: Exchange-specific configuration

```python
class ExchangeConfig(BaseConfig):
    def validate_default_exchange(cls, v: str) -> str  # Line 142
    def validate_enabled_exchanges(cls, v: list[str]) -> list[str]  # Line 151
    def get_exchange_credentials(self, exchange: str) -> dict[str, Any]  # Line 159
    def is_exchange_configured(self, exchange: str) -> bool  # Line 192
    def get_websocket_config(self, exchange: str) -> dict[str, Any]  # Line 200
    def get_connection_pool_config(self) -> dict[str, Any]  # Line 227
    def get_rate_limit_config(self) -> dict[str, Any]  # Line 240
```

### File: execution.py

#### Class: `ExecutionConfig`

**Inherits**: BaseConfig
**Purpose**: Execution-specific configuration

```python
class ExecutionConfig(BaseConfig):
    def validate_decimal_fields(cls, v)  # Line 78
    def get(self, key: str, default: Any = None) -> Any  # Line 86
    def get_routing_config(self) -> dict[str, Any]  # Line 90
    def get_order_size_limits(self) -> dict[str, Decimal | None]  # Line 94
    def get_performance_settings(self) -> dict[str, int]  # Line 98
```

### File: main.py

#### Class: `Config`

**Purpose**: Main configuration aggregator that maintains backward compatibility

```python
class Config:
    def __init__(self, config_file: str | None = None, env_file: str | None = '.env')  # Line 32
    def load_from_file(self, config_file: str) -> None  # Line 73
    def _validate_config_file(self, config_path: Path) -> None  # Line 81
    def _parse_config_file(self, config_path: Path) -> dict[str, Any]  # Line 89
    def _apply_config_data(self, config_data: dict[str, Any]) -> None  # Line 102
    def _update_config_section(self, config_data: dict[str, Any], section_name: str, config_obj: Any) -> None  # Line 122
    def save_to_file(self, config_file: str) -> None  # Line 134
    def _build_config_data(self) -> dict[str, Any]  # Line 140
    def _write_config_file(self, config_path: Path, config_data: dict[str, Any]) -> None  # Line 163
    def validate(self) -> None  # Line 173
    def db_url(self) -> str  # Line 187
    def redis_url(self) -> str  # Line 192
    def postgresql_host(self) -> str  # Line 197
    def postgresql_port(self) -> int  # Line 202
    def postgresql_database(self) -> str  # Line 207
    def postgresql_username(self) -> str  # Line 212
    def postgresql_password(self) -> str | None  # Line 217
    def redis_host(self) -> str  # Line 222
    def redis_port(self) -> int  # Line 227
    def binance_api_key(self) -> str  # Line 232
    def binance_api_secret(self) -> str  # Line 237
    def max_position_size(self) -> Any  # Line 242
    def risk_per_trade(self) -> float  # Line 247
    def get_exchange_config(self, exchange: str) -> dict[str, Any]  # Line 251
    def get_environment_exchange_config(self, exchange: str) -> dict[str, Any]  # Line 264
    def get_strategy_config(self, strategy_type: str) -> dict[str, Any]  # Line 295
    def get_risk_config(self) -> dict[str, Any]  # Line 299
    def to_dict(self) -> dict[str, Any]  # Line 303
    def _json_serializer(self, obj)  # Line 316
    def switch_environment(self, environment: str, exchange: str = None) -> bool  # Line 332
    def validate_environment_switch(self, environment: str, exchange: str = None) -> dict[str, Any]  # Line 373
    def get_current_environment_status(self) -> dict[str, Any]  # Line 438
    def is_production_mode(self, exchange: str = None) -> bool  # Line 454
```

#### Functions:

```python
def get_config(config_file: str | None = None, reload: bool = False) -> Config  # Line 478
```

### File: migration.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Functions:

```python
async def setup_config_service(...) -> ConfigService  # Line 32
def migrate_legacy_config_usage() -> None  # Line 94
def validate_migration_status() -> dict[str, bool]  # Line 158
async def test_migration() -> None  # Line 195
def create_service_with_config_injection()  # Line 259
def create_service_with_fallback_injection()  # Line 282
```

### File: risk.py

#### Class: `RiskConfig`

**Inherits**: BaseConfig
**Purpose**: Risk management configuration

```python
class RiskConfig(BaseConfig):
    def validate_sizing_method(cls, v: str) -> str  # Line 236
    def get_position_size_params(self) -> dict  # Line 249
    def is_risk_exceeded(self, current_loss: Decimal) -> bool  # Line 275
```

### File: sandbox.py

#### Class: `SandboxEnvironment`

**Inherits**: str, Enum
**Purpose**: Sandbox environment types

```python
class SandboxEnvironment(str, Enum):
```

#### Class: `SandboxExchangeConfig`

**Inherits**: BaseConfig
**Purpose**: Sandbox-specific exchange configuration

```python
class SandboxExchangeConfig(BaseConfig):
    def validate_environment(cls, v: SandboxEnvironment) -> SandboxEnvironment  # Line 126
    def get_sandbox_credentials(self, exchange: str) -> dict[str, Any]  # Line 135
    def get_mock_balances(self) -> dict[str, str]  # Line 170
    def is_sandbox_environment(self) -> bool  # Line 178
    def get_environment_config(self) -> dict[str, Any]  # Line 182
```

### File: security.py

#### Class: `SecurityConfig`

**Inherits**: BaseConfig
**Purpose**: Security configuration for JWT, authentication, and other security settings

```python
class SecurityConfig(BaseConfig):
    def get_jwt_config(self) -> dict  # Line 121
    def get_session_config(self) -> dict  # Line 130
    def get_cors_config(self) -> dict  # Line 136
    def get_rate_limit_config(self) -> dict  # Line 145
```

### File: service.py

#### Class: `ConfigProvider`

**Inherits**: Protocol
**Purpose**: Protocol for configuration providers

```python
class ConfigProvider(Protocol):
    async def load_config(self) -> ConfigDict  # Line 75
    async def save_config(self, config: ConfigDict) -> None  # Line 79
    async def watch_changes(self, callback: ConfigCallback) -> None  # Line 83
```

#### Class: `ConfigChangeEvent`

**Inherits**: BaseValidatedModel
**Purpose**: Configuration change event

```python
class ConfigChangeEvent(BaseValidatedModel):
```

#### Class: `ConfigCache`

**Purpose**: Thread-safe configuration cache with TTL support

```python
class ConfigCache:
    def __init__(self, default_ttl: int = 300)  # Line 101
    def get(self, key: str, default: Any = None) -> Any  # Line 110
    def set(self, key: str, value: Any, ttl: int | None = None) -> None  # Line 125
    def invalidate(self, key: str) -> None  # Line 135
    def clear(self) -> None  # Line 144
    def get_stats(self) -> dict[str, Any]  # Line 151
```

#### Class: `FileConfigProvider`

**Purpose**: File-based configuration provider

```python
class FileConfigProvider:
    def __init__(self, config_file: Path, watch_changes: bool = False)  # Line 165
    async def load_config(self) -> ConfigDict  # Line 174
    async def save_config(self, config: ConfigDict) -> None  # Line 222
    async def watch_changes(self, callback: ConfigCallback) -> None  # Line 254
```

#### Class: `EnvironmentConfigProvider`

**Purpose**: Environment-based configuration provider

```python
class EnvironmentConfigProvider:
    def __init__(self, prefix: str = 'TBOT_')  # Line 267
    async def load_config(self) -> ConfigDict  # Line 273
    def _set_nested_value(self, config: dict[str, Any], key: str, value: str) -> dict[str, Any]  # Line 287
    async def save_config(self, config: ConfigDict) -> None  # Line 298
    async def watch_changes(self, callback: ConfigCallback) -> None  # Line 306
```

#### Class: `ConfigValidator`

**Purpose**: Configuration validation service

```python
class ConfigValidator:
    def __init__(self) -> None  # Line 315
    async def validate_database_config(self, config: dict) -> DatabaseConfig  # Line 321
    async def validate_exchange_config(self, config: dict) -> ExchangeConfig  # Line 333
    async def validate_risk_config(self, config: dict) -> RiskConfig  # Line 345
    async def validate_strategy_config(self, config: dict) -> StrategyConfig  # Line 357
    def register_validator(self, config_section: str, validator: Callable[[dict], Any]) -> None  # Line 369
    async def validate_custom_config(self, section: str, config: dict) -> Any  # Line 374
```

#### Class: `ConfigService`

**Purpose**: Main configuration service with dependency injection support

```python
class ConfigService:
    def __init__(self, ...)  # Line 424
    async def initialize(self, config_file: str | Path | None = None, watch_changes: bool = False) -> None  # Line 455
    async def shutdown(self) -> None  # Line 489
    async def _load_configuration(self) -> None  # Line 504
    async def _load_from_all_providers(self) -> dict[str, Any]  # Line 521
    async def _create_config_instance(self, merged_config: dict[str, Any]) -> None  # Line 536
    async def _update_domain_configs(self, merged_config: dict[str, Any]) -> None  # Line 546
    def _update_app_configs(self, merged_config: dict[str, Any]) -> None  # Line 560
    def _deep_merge(self, base: dict, override: dict) -> dict  # Line 570
    async def _hot_reload_loop(self) -> None  # Line 580
    async def _notify_changes(self, old_config: Config, new_config: Config) -> None  # Line 603
    def add_change_listener(self, callback: ConfigCallback) -> None  # Line 624
    def remove_change_listener(self, callback: ConfigCallback) -> None  # Line 628
    def get_database_config(self) -> DatabaseConfig  # Line 635
    def get_exchange_config(self, exchange: str | None = None) -> ExchangeConfig | dict[str, Any]  # Line 648
    def get_risk_config(self) -> RiskConfig  # Line 671
    def get_strategy_config(self, strategy_type: str | None = None) -> StrategyConfig | dict[str, Any]  # Line 684
    def get_app_config(self) -> dict[str, Any]  # Line 709
    def get_config_value(self, key: str, default: Any = None) -> Any  # Line 730
    def _get_nested_value(self, config: dict, key: str, default: Any = None) -> Any  # Line 746
    def _ensure_initialized(self) -> None  # Line 759
    async def __aenter__(self) -> 'ConfigService'  # Line 769
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None  # Line 774
    def get_cache_stats(self) -> dict[str, Any]  # Line 779
    def invalidate_cache(self, key: str | None = None) -> None  # Line 783
    def get_loaded_config(self) -> dict[str, Any] | None  # Line 790
    def get_config_dict(self) -> dict[str, Any]  # Line 796
    def get_config(self) -> dict[str, Any]  # Line 805
```

#### Functions:

```python
async def get_config_service(config_file: str | Path | None = None, reload: bool = False) -> ConfigService  # Line 818
async def shutdown_config_service() -> None  # Line 864
def register_config_service_in_container(config_file: str | None = None, enable_hot_reload: bool = False) -> None  # Line 873
```

### File: state_management.py

#### Class: `StateManagementConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for state management system

```python
class StateManagementConfig(BaseModel):
    def get_checkpoint_config(self) -> dict  # Line 144
    def get_validation_config(self) -> dict  # Line 148
    def get_recovery_config(self) -> dict  # Line 157
    def get_performance_config(self) -> dict  # Line 165
    def get_monitoring_config(self) -> dict  # Line 176
    def get_sync_config(self) -> dict  # Line 185
```

### File: strategy.py

#### Class: `StrategyConfig`

**Inherits**: BaseConfig
**Purpose**: Strategy-specific configuration

```python
class StrategyConfig(BaseConfig):
    def validate_timeframe(cls, v: str) -> str  # Line 116
    def validate_combination_method(cls, v: str) -> str  # Line 125
    def get_strategy_params(self, strategy_type: str) -> dict[str, Any]  # Line 132
```

### File: validator.py

#### Functions:

```python
def check_env_pollution() -> Dict[str, str]  # Line 7
def validate_environment(strict: bool = False) -> None  # Line 28
def validate_credentials(config_dict: Dict[str, str], source: str = 'config') -> None  # Line 62
```

### File: config.py

#### Class: `BaseConfig`

**Inherits**: BaseSettings
**Purpose**: Base configuration class with common patterns

```python
class BaseConfig(BaseSettings):
```

#### Class: `DatabaseConfig`

**Inherits**: BaseConfig
**Purpose**: Database configuration for PostgreSQL, Redis, and InfluxDB

```python
class DatabaseConfig(BaseConfig):
    def validate_ports(cls, v: int) -> int  # Line 86
    def validate_pool_size(cls, v)  # Line 94
    def postgresql_url(self) -> str  # Line 101
    def redis_url(self) -> str  # Line 109
```

#### Class: `SecurityConfig`

**Inherits**: BaseConfig
**Purpose**: Security configuration for authentication and encryption

```python
class SecurityConfig(BaseConfig):
    def validate_jwt_expire(cls, v)  # Line 138
    def _validate_key_basic(cls, v: str) -> None  # Line 145
    def _validate_key_character_types(cls, v: str) -> None  # Line 153
    def _validate_key_patterns(cls, v: str) -> None  # Line 168
    def _calculate_shannon_entropy(cls, s: str) -> float  # Line 179
    def _validate_key_entropy(cls, v: str) -> None  # Line 202
    def validate_key_length(cls, v)  # Line 218
```

#### Class: `ErrorHandlingConfig`

**Inherits**: BaseConfig
**Purpose**: Error handling configuration for P-002A framework

```python
class ErrorHandlingConfig(BaseConfig):
    def validate_positive_integers(cls, v)  # Line 281
    def validate_positive_floats(cls, v)  # Line 291
```

#### Class: `ExchangeConfig`

**Inherits**: BaseConfig
**Purpose**: Exchange configuration for API credentials and rate limits

```python
class ExchangeConfig(BaseConfig):
    def validate_api_credentials(cls, v: str) -> str  # Line 358
    def default_exchange(self) -> str  # Line 387
    def testnet_mode(self) -> bool  # Line 392
    def rate_limit_per_second(self) -> int  # Line 404
    def get_exchange_credentials(self, exchange: str) -> dict[str, Any]  # Line 409
    def get_websocket_config(self, exchange: str) -> dict[str, Any]  # Line 433
```

#### Class: `RiskConfig`

**Inherits**: BaseConfig
**Purpose**: Risk management configuration for P-008 framework

```python
class RiskConfig(BaseConfig):
    def validate_percentage_fields(cls, v)  # Line 539
    def validate_positive_integers(cls, v)  # Line 554
```

#### Class: `CapitalManagementConfig`

**Inherits**: BaseConfig
**Purpose**: Capital management configuration for P-010A framework

```python
class CapitalManagementConfig(BaseConfig):
    def validate_percentage_fields(cls, v)  # Line 681
    def validate_positive_integers(cls, v)  # Line 689
    def validate_positive_decimals(cls, v: Decimal) -> Decimal  # Line 703
```

#### Class: `StrategyManagementConfig`

**Inherits**: BaseConfig
**Purpose**: Strategy management configuration for P-011 framework

```python
class StrategyManagementConfig(BaseConfig):
    def validate_positive_integers(cls, v)  # Line 760
    def validate_percentage_fields(cls, v)  # Line 777
```

#### Class: `MLConfig`

**Inherits**: BaseConfig
**Purpose**: Machine Learning configuration for P-014 framework

```python
class MLConfig(BaseConfig):
    def validate_percentage_fields(cls, v)  # Line 910
    def validate_positive_integers(cls, v)  # Line 931
    def validate_positive_float(cls, v)  # Line 939
```

#### Class: `BotManagementConfig`

**Inherits**: BaseConfig
**Purpose**: Bot management configuration for bot lifecycle and coordination

```python
class BotManagementConfig(BaseConfig):
    def validate_decimal_positive(cls, v: Decimal) -> Decimal  # Line 1030
    def validate_positive_integers(cls, v: int) -> int  # Line 1052
```

#### Class: `ExecutionConfig`

**Inherits**: BaseConfig
**Purpose**: Execution engine configuration for order processing and algorithms

```python
class ExecutionConfig(BaseConfig):
```

#### Class: `Config`

**Inherits**: BaseConfig
**Purpose**: Master configuration class for the entire application

```python
class Config(BaseConfig):
    def validate_environment(cls, v: str) -> str  # Line 1142
    def generate_schema(self) -> None  # Line 1149
    def from_yaml(cls, yaml_path: str | Path) -> 'Config'  # Line 1158
    def from_yaml_with_env_override(cls, yaml_path: str | Path) -> 'Config'  # Line 1187
    def to_yaml(self, yaml_path: str | Path) -> None  # Line 1222
    def get_database_url(self) -> str  # Line 1237
    def get_async_database_url(self) -> str  # Line 1251
    def get_redis_url(self) -> str  # Line 1265
    def is_production(self) -> bool  # Line 1282
    def is_development(self) -> bool  # Line 1286
    def validate_yaml_config(self, yaml_path: str | Path) -> bool  # Line 1290
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `CoreDataTransformer`

**Purpose**: Handles consistent data transformation for core module events and messaging

```python
class CoreDataTransformer:
    def transform_event_to_standard_format(event_type, ...) -> dict[str, Any]  # Line 29
    def transform_for_pub_sub_pattern(event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 79
    def transform_for_request_reply_pattern(request_type, ...) -> dict[str, Any]  # Line 109
    def align_processing_paradigm(data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 143
    def validate_boundary_fields(data: dict[str, Any]) -> dict[str, Any]  # Line 188
    def _apply_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 245
    def apply_cross_module_consistency(cls, data: dict[str, Any], target_module: str, source_module: str = 'core') -> dict[str, Any]  # Line 287
    def _apply_boundary_validation(data: dict[str, Any], source_module: str, target_module: str) -> None  # Line 337
```

### File: dependency_injection.py

**Key Imports:**
- `from src.core.exceptions import ComponentError`
- `from src.core.exceptions import DependencyError`
- `from src.core.logging import get_logger`

#### Class: `DependencyContainer`

**Purpose**: Container for managing dependencies

```python
class DependencyContainer:
    def __init__(self) -> None  # Line 22
    def register(self, name: str, service: Any | Callable, singleton: bool = False) -> None  # Line 30
    def register_class(self, name: str, cls: type[T], *args, **kwargs) -> None  # Line 60
    def get(self, name: str) -> Any  # Line 124
    def has(self, name: str) -> bool  # Line 175
    def resolve(self, name: str) -> Any  # Line 179
    def get_optional(self, name: str) -> Any | None  # Line 194
    def register_factory(self, name: str, factory: Callable, singleton: bool = False) -> None  # Line 209
    def register_singleton(self, name: str, service: Any) -> None  # Line 220
    def register_service(self, name: str, service: Any, singleton: bool = False) -> None  # Line 230
    def has_service(self, name: str) -> bool  # Line 241
    def is_registered(self, name: str) -> bool  # Line 253
    def get_container(self) -> 'DependencyContainer'  # Line 265
    def __contains__(self, name: str) -> bool  # Line 274
    def clear(self) -> None  # Line 278
```

#### Class: `DependencyInjector`

**Purpose**: Dependency injector for automatic dependency resolution

```python
class DependencyInjector:
    def __new__(cls) -> DependencyInjector  # Line 296
    def __init__(self)  # Line 303
    def register(self, name: str | None = None, singleton: bool = False)  # Line 314
    def inject(self, func: Callable) -> Callable  # Line 337
    def resolve(self, name: str) -> Any  # Line 397
    def get_optional(self, name: str) -> Any | None  # Line 422
    def register_service(self, name: str, service: Any, singleton: bool = False) -> None  # Line 437
    def register_factory(self, name: str, factory: Callable, singleton: bool = False) -> None  # Line 460
    def register_interface(self, interface: type, implementation_factory: Callable, singleton: bool = False) -> None  # Line 485
    def register_singleton(self, name: str, service: Any) -> None  # Line 517
    def register_transient(self, name: str, service_class: type, *args, **kwargs) -> None  # Line 530
    def has_service(self, name: str) -> bool  # Line 566
    def is_registered(self, name: str) -> bool  # Line 570
    def clear(self) -> None  # Line 574
    def get_instance(cls) -> DependencyInjector  # Line 579
    def reset_instance(cls) -> None  # Line 586
    def get_container(self) -> DependencyContainer  # Line 596
    def configure_service_dependencies(self, service_instance: Any) -> None  # Line 600
```

#### Class: `ServiceLocator`

**Purpose**: Service locator for easy access to services

```python
class ServiceLocator:
    def __init__(self, injector: DependencyInjector)  # Line 644
    def __getattr__(self, name: str) -> Any  # Line 647
```

#### Functions:

```python
def injectable(name: str | None = None, singleton: bool = False)  # Line 627
def inject(func: Callable) -> Callable  # Line 632
def get_container() -> DependencyContainer  # Line 666
def get_global_injector() -> DependencyInjector  # Line 671
```

### File: dependency_ordering.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.dependency_injection import DependencyInjector`

#### Class: `DependencyLevel`

**Inherits**: IntEnum
**Purpose**: Dependency levels for ordering service registration

```python
class DependencyLevel(IntEnum):
```

#### Class: `DependencyRegistrar`

**Purpose**: Manages ordered dependency registration to prevent circular dependencies

```python
class DependencyRegistrar:
    def __init__(self, injector: DependencyInjector)  # Line 67
    def register_at_level(self, level: DependencyLevel, registration_func: Callable) -> None  # Line 79
    def add_lazy_configuration(self, config_func: Callable) -> None  # Line 91
    def register_all(self) -> None  # Line 100
```

#### Functions:

```python
def create_ordered_registrar(injector: Optional[DependencyInjector] = None) -> DependencyRegistrar  # Line 130
def _register_core_dependencies(registrar: DependencyRegistrar) -> None  # Line 151
def register_module_at_level(...) -> None  # Line 212
def register_lazy_configuration(...) -> None  # Line 228
```

### File: di_master_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.dependency_ordering import DependencyRegistrar`
- `from src.core.dependency_ordering import DependencyLevel`
- `from src.core.dependency_ordering import create_ordered_registrar`
- `from src.core.dependency_ordering import register_module_at_level`

#### Functions:

```python
def register_all_services(injector: Optional[DependencyInjector] = None, config: Optional[Config] = None) -> DependencyInjector  # Line 21
def _register_all_modules(registrar: DependencyRegistrar, config: Config) -> None  # Line 55
def get_master_injector() -> DependencyInjector  # Line 312
```

### File: event_constants.py

#### Class: `AlertEvents`

**Purpose**: Alert-related event names

```python
class AlertEvents:
```

#### Class: `OrderEvents`

**Purpose**: Order-related event names

```python
class OrderEvents:
```

#### Class: `TradeEvents`

**Purpose**: Trade-related event names

```python
class TradeEvents:
```

#### Class: `PositionEvents`

**Purpose**: Position-related event names

```python
class PositionEvents:
```

#### Class: `RiskEvents`

**Purpose**: Risk management event names

```python
class RiskEvents:
```

#### Class: `MetricEvents`

**Purpose**: Metric-related event names

```python
class MetricEvents:
```

#### Class: `SystemEvents`

**Purpose**: System-level event names

```python
class SystemEvents:
```

#### Class: `MarketDataEvents`

**Purpose**: Market data event names

```python
class MarketDataEvents:
```

#### Class: `StrategyEvents`

**Purpose**: Strategy-related event names

```python
class StrategyEvents:
```

#### Class: `CapitalEvents`

**Purpose**: Capital management event names

```python
class CapitalEvents:
```

#### Class: `ExchangeEvents`

**Purpose**: Exchange connection event names

```python
class ExchangeEvents:
```

#### Class: `MLEvents`

**Purpose**: Machine learning event names

```python
class MLEvents:
```

#### Class: `TrainingEvents`

**Purpose**: Model training event names

```python
class TrainingEvents:
```

#### Class: `InferenceEvents`

**Purpose**: Model inference event names

```python
class InferenceEvents:
```

#### Class: `FeatureEvents`

**Purpose**: Feature engineering event names

```python
class FeatureEvents:
```

#### Class: `ModelValidationEvents`

**Purpose**: Model validation event names

```python
class ModelValidationEvents:
```

#### Class: `StateEvents`

**Purpose**: State management event names

```python
class StateEvents:
```

#### Class: `BacktestEvents`

**Purpose**: Backtesting event names

```python
class BacktestEvents:
```

#### Class: `DataEvents`

**Purpose**: Data processing event names

```python
class DataEvents:
```

#### Class: `OptimizationEvents`

**Purpose**: Optimization-related event names

```python
class OptimizationEvents:
```

#### Class: `BotEvents`

**Purpose**: Bot management event names

```python
class BotEvents:
```

### File: events.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `BotEventType`

**Inherits**: Enum
**Purpose**: Bot event types for coordination and monitoring

```python
class BotEventType(Enum):
```

#### Class: `BotEvent`

**Purpose**: Bot event data structure

```python
class BotEvent:
    def __post_init__(self)  # Line 65
```

#### Class: `EventHandler`

**Purpose**: Base class for event handlers

```python
class EventHandler:
    def __init__(self, handler_name: str)  # Line 78
    async def handle(self, event: BotEvent) -> None  # Line 82
```

#### Class: `EventPublisher`

**Purpose**: Event publisher for bot management coordination

```python
class EventPublisher:
    def __init__(self)  # Line 90
    def subscribe(self, event_type: BotEventType, handler: EventHandler) -> None  # Line 97
    def subscribe_all(self, handler: EventHandler) -> None  # Line 106
    def unsubscribe(self, event_type: BotEventType, handler: EventHandler) -> None  # Line 111
    async def publish(self, event: BotEvent, processing_mode: str = 'stream') -> None  # Line 118
    async def _safe_handle(self, handler: EventHandler, event: BotEvent) -> None  # Line 185
    def get_recent_events(self, ...) -> list[BotEvent]  # Line 196
    def _transform_event_data_consistent(self, event: BotEvent, processing_mode: str) -> BotEvent  # Line 218
    def _collect_handlers_by_priority(self, event_type: BotEventType) -> list[EventHandler]  # Line 241
    async def _process_handlers_batch(self, handlers: list[EventHandler], event: BotEvent) -> None  # Line 257
    async def _process_handlers_stream(self, handlers: list[EventHandler], event: BotEvent) -> None  # Line 301
    def _propagate_event_error_consistently(self, error: Exception, operation: str, event_type: str) -> None  # Line 338
```

#### Class: `AnalyticsEventHandler`

**Inherits**: EventHandler
**Purpose**: Event handler for analytics integration

```python
class AnalyticsEventHandler(EventHandler):
    def __init__(self, analytics_service = None)  # Line 397
    async def handle(self, event: BotEvent) -> None  # Line 401
```

#### Class: `RiskMonitoringEventHandler`

**Inherits**: EventHandler
**Purpose**: Event handler for risk monitoring

```python
class RiskMonitoringEventHandler(EventHandler):
    def __init__(self, risk_service = None)  # Line 498
    async def handle(self, event: BotEvent) -> None  # Line 502
```

#### Functions:

```python
def get_event_publisher() -> EventPublisher  # Line 590
def setup_bot_management_events(analytics_service = None, risk_service = None) -> EventPublisher  # Line 598
```

### File: exceptions.py

#### Class: `ErrorCategory`

**Inherits**: Enum
**Purpose**: Error categorization for automated handling

```python
class ErrorCategory(Enum):
```

#### Class: `ErrorSeverity`

**Inherits**: Enum
**Purpose**: Error severity levels

```python
class ErrorSeverity(Enum):
```

#### Class: `TradingBotError`

**Inherits**: Exception
**Purpose**: Base exception for all trading bot errors

```python
class TradingBotError(Exception):
    def __init__(self, ...) -> None  # Line 86
    def _sanitize_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]  # Line 119
    def _log_error(self) -> None  # Line 156
    def to_dict(self) -> dict[str, Any]  # Line 188
    def __str__(self) -> str  # Line 204
    def __repr__(self) -> str  # Line 228
```

#### Class: `ExchangeError`

**Inherits**: TradingBotError
**Purpose**: Base class for all exchange-related errors

```python
class ExchangeError(TradingBotError):
    def __init__(self, ...) -> None  # Line 261
```

#### Class: `ExchangeConnectionError`

**Inherits**: ExchangeError
**Purpose**: Network connection failures to exchange APIs

```python
class ExchangeConnectionError(ExchangeError):
    def __init__(self, message: str, exchange: str | None = None, **kwargs: Any) -> None  # Line 298
```

#### Class: `ExchangeRateLimitError`

**Inherits**: ExchangeError
**Purpose**: Rate limit violations from exchange APIs

```python
class ExchangeRateLimitError(ExchangeError):
    def __init__(self, ...) -> None  # Line 316
```

#### Class: `ExchangeInsufficientFundsError`

**Inherits**: ExchangeError
**Purpose**: Insufficient balance for order execution

```python
class ExchangeInsufficientFundsError(ExchangeError):
    def __init__(self, ...) -> None  # Line 352
```

#### Class: `ExchangeOrderError`

**Inherits**: ExchangeError
**Purpose**: General order-related exchange errors

```python
class ExchangeOrderError(ExchangeError):
    def __init__(self, ...) -> None  # Line 389
```

#### Class: `ExchangeAuthenticationError`

**Inherits**: ExchangeError
**Purpose**: Exchange authentication and authorization failures

```python
class ExchangeAuthenticationError(ExchangeError):
    def __init__(self, ...) -> None  # Line 420
```

#### Class: `InvalidOrderError`

**Inherits**: ExchangeOrderError
**Purpose**: Invalid order parameters

```python
class InvalidOrderError(ExchangeOrderError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 441
```

#### Class: `RiskManagementError`

**Inherits**: TradingBotError
**Purpose**: Base class for all risk management violations and errors

```python
class RiskManagementError(TradingBotError):
    def __init__(self, ...) -> None  # Line 461
```

#### Class: `PositionLimitError`

**Inherits**: RiskManagementError
**Purpose**: Position size or count limit violations

```python
class PositionLimitError(RiskManagementError):
    def __init__(self, ...) -> None  # Line 500
```

#### Class: `DrawdownLimitError`

**Inherits**: RiskManagementError
**Purpose**: Maximum drawdown limit violations

```python
class DrawdownLimitError(RiskManagementError):
    def __init__(self, ...) -> None  # Line 534
```

#### Class: `RiskCalculationError`

**Inherits**: RiskManagementError
**Purpose**: Risk metric calculation failures

```python
class RiskCalculationError(RiskManagementError):
    def __init__(self, ...) -> None  # Line 566
```

#### Class: `CapitalAllocationError`

**Inherits**: RiskManagementError
**Purpose**: Capital allocation rule violations

```python
class CapitalAllocationError(RiskManagementError):
    def __init__(self, ...) -> None  # Line 595
```

#### Class: `AllocationError`

**Inherits**: RiskManagementError
**Purpose**: Portfolio allocation errors

```python
class AllocationError(RiskManagementError):
    def __init__(self, ...) -> None  # Line 627
```

#### Class: `CircuitBreakerTriggeredError`

**Inherits**: RiskManagementError
**Purpose**: Circuit breaker activation

```python
class CircuitBreakerTriggeredError(RiskManagementError):
    def __init__(self, ...) -> None  # Line 661
```

#### Class: `EmergencyStopError`

**Inherits**: RiskManagementError
**Purpose**: Emergency stop system failures

```python
class EmergencyStopError(RiskManagementError):
    def __init__(self, ...) -> None  # Line 695
```

#### Class: `DataError`

**Inherits**: TradingBotError
**Purpose**: Base class for all data-related errors

```python
class DataError(TradingBotError):
    def __init__(self, ...) -> None  # Line 728
```

#### Class: `DataValidationError`

**Inherits**: DataError
**Purpose**: Data validation and schema compliance failures

```python
class DataValidationError(DataError):
    def __init__(self, ...) -> None  # Line 763
```

#### Class: `DataSourceError`

**Inherits**: DataError
**Purpose**: External data source connectivity and reliability issues

```python
class DataSourceError(DataError):
    def __init__(self, ...) -> None  # Line 795
```

#### Class: `DataProcessingError`

**Inherits**: DataError
**Purpose**: Data transformation and processing pipeline failures

```python
class DataProcessingError(DataError):
    def __init__(self, ...) -> None  # Line 831
```

#### Class: `DataCorruptionError`

**Inherits**: DataError
**Purpose**: Data integrity and corruption detection

```python
class DataCorruptionError(DataError):
    def __init__(self, ...) -> None  # Line 856
```

#### Class: `DataQualityError`

**Inherits**: DataError
**Purpose**: Data quality issues affecting trading decisions

```python
class DataQualityError(DataError):
    def __init__(self, ...) -> None  # Line 893
```

#### Class: `ModelError`

**Inherits**: TradingBotError
**Purpose**: Base class for all ML model-related errors

```python
class ModelError(TradingBotError):
    def __init__(self, ...) -> None  # Line 930
```

#### Class: `ModelLoadError`

**Inherits**: ModelError
**Purpose**: Model loading and initialization failures

```python
class ModelLoadError(ModelError):
    def __init__(self, ...) -> None  # Line 959
```

#### Class: `ModelInferenceError`

**Inherits**: ModelError
**Purpose**: Model prediction and inference failures

```python
class ModelInferenceError(ModelError):
    def __init__(self, ...) -> None  # Line 984
```

#### Class: `ModelDriftError`

**Inherits**: ModelError
**Purpose**: Model performance drift detection

```python
class ModelDriftError(ModelError):
    def __init__(self, ...) -> None  # Line 1016
```

#### Class: `ModelTrainingError`

**Inherits**: ModelError
**Purpose**: Model training and optimization failures

```python
class ModelTrainingError(ModelError):
    def __init__(self, ...) -> None  # Line 1051
```

#### Class: `ModelValidationError`

**Inherits**: ModelError
**Purpose**: Model validation and testing failures

```python
class ModelValidationError(ModelError):
    def __init__(self, ...) -> None  # Line 1076
```

#### Class: `ValidationError`

**Inherits**: TradingBotError
**Purpose**: Base class for all input and schema validation errors

```python
class ValidationError(TradingBotError):
    def __init__(self, ...) -> None  # Line 1113
```

#### Class: `ConfigurationError`

**Inherits**: ValidationError
**Purpose**: Configuration file and parameter validation errors

```python
class ConfigurationError(ValidationError):
    def __init__(self, ...) -> None  # Line 1149
```

#### Class: `SchemaValidationError`

**Inherits**: ValidationError
**Purpose**: Data schema and structure validation failures

```python
class SchemaValidationError(ValidationError):
    def __init__(self, ...) -> None  # Line 1174
```

#### Class: `InputValidationError`

**Inherits**: ValidationError
**Purpose**: Function and API input parameter validation failures

```python
class InputValidationError(ValidationError):
    def __init__(self, ...) -> None  # Line 1205
```

#### Class: `BusinessRuleValidationError`

**Inherits**: ValidationError
**Purpose**: Business logic and rule validation failures

```python
class BusinessRuleValidationError(ValidationError):
    def __init__(self, ...) -> None  # Line 1238
```

#### Class: `ExecutionError`

**Inherits**: TradingBotError
**Purpose**: Base class for all order execution and trading errors

```python
class ExecutionError(TradingBotError):
    def __init__(self, ...) -> None  # Line 1268
```

#### Class: `OrderRejectionError`

**Inherits**: ExecutionError
**Purpose**: Order rejected by exchange

```python
class OrderRejectionError(ExecutionError):
    def __init__(self, message: str, rejection_reason: str | None = None, **kwargs: Any) -> None  # Line 1294
```

#### Class: `SlippageError`

**Inherits**: ExecutionError
**Purpose**: Excessive slippage during order execution

```python
class SlippageError(ExecutionError):
    def __init__(self, ...) -> None  # Line 1312
```

#### Class: `ExecutionTimeoutError`

**Inherits**: ExecutionError
**Purpose**: Order execution timeout errors

```python
class ExecutionTimeoutError(ExecutionError):
    def __init__(self, ...) -> None  # Line 1345
```

#### Class: `ExecutionPartialFillError`

**Inherits**: ExecutionError
**Purpose**: Partial order fill handling errors

```python
class ExecutionPartialFillError(ExecutionError):
    def __init__(self, ...) -> None  # Line 1370
```

#### Class: `NetworkError`

**Inherits**: TradingBotError
**Purpose**: Base class for all network and communication errors

```python
class NetworkError(TradingBotError):
    def __init__(self, ...) -> None  # Line 1407
```

#### Class: `ConnectionError`

**Inherits**: NetworkError
**Purpose**: Network connection establishment failures

```python
class ConnectionError(NetworkError):
    def __init__(self, message: str, connection_type: str | None = None, **kwargs: Any) -> None  # Line 1436
```

#### Class: `TimeoutError`

**Inherits**: NetworkError
**Purpose**: Network operation timeout errors

```python
class TimeoutError(NetworkError):
    def __init__(self, ...) -> None  # Line 1454
```

#### Class: `WebSocketError`

**Inherits**: NetworkError
**Purpose**: WebSocket connection and messaging errors

```python
class WebSocketError(NetworkError):
    def __init__(self, ...) -> None  # Line 1478
```

#### Class: `StateConsistencyError`

**Inherits**: TradingBotError
**Purpose**: Base class for all state management and consistency errors

```python
class StateConsistencyError(TradingBotError):
    def __init__(self, ...) -> None  # Line 1514
```

#### Class: `StateError`

**Inherits**: StateConsistencyError
**Purpose**: General state management errors

```python
class StateError(StateConsistencyError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1537
```

#### Class: `StateCorruptionError`

**Inherits**: StateConsistencyError
**Purpose**: State data corruption detected

```python
class StateCorruptionError(StateConsistencyError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1545
```

#### Class: `StateLockError`

**Inherits**: StateConsistencyError
**Purpose**: State lock acquisition failures

```python
class StateLockError(StateConsistencyError):
    def __init__(self, message: str, lock_name: str | None = None, **kwargs: Any) -> None  # Line 1556
```

#### Class: `SynchronizationError`

**Inherits**: StateConsistencyError
**Purpose**: Real-time synchronization errors

```python
class SynchronizationError(StateConsistencyError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1571
```

#### Class: `ConflictError`

**Inherits**: StateConsistencyError
**Purpose**: State conflict errors

```python
class ConflictError(StateConsistencyError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1580
```

#### Class: `SecurityError`

**Inherits**: TradingBotError
**Purpose**: Base class for all security-related errors

```python
class SecurityError(TradingBotError):
    def __init__(self, ...) -> None  # Line 1597
```

#### Class: `AuthenticationError`

**Inherits**: SecurityError
**Purpose**: Authentication failures and credential issues

```python
class AuthenticationError(SecurityError):
    def __init__(self, message: str, auth_method: str | None = None, **kwargs: Any) -> None  # Line 1624
```

#### Class: `AuthorizationError`

**Inherits**: SecurityError
**Purpose**: Authorization and permission failures

```python
class AuthorizationError(SecurityError):
    def __init__(self, message: str, required_permission: str | None = None, **kwargs: Any) -> None  # Line 1638
```

#### Class: `EncryptionError`

**Inherits**: SecurityError
**Purpose**: Encryption and decryption failures

```python
class EncryptionError(SecurityError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1652
```

#### Class: `TokenValidationError`

**Inherits**: SecurityError
**Purpose**: Token validation and parsing failures

```python
class TokenValidationError(SecurityError):
    def __init__(self, message: str, token_type: str | None = None, **kwargs: Any) -> None  # Line 1661
```

#### Class: `StrategyError`

**Inherits**: TradingBotError
**Purpose**: Base class for all strategy-related errors

```python
class StrategyError(TradingBotError):
    def __init__(self, message: str, strategy_id: str | None = None, **kwargs: Any) -> None  # Line 1679
```

#### Class: `StrategyConfigurationError`

**Inherits**: StrategyError
**Purpose**: Strategy configuration errors

```python
class StrategyConfigurationError(StrategyError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1694
```

#### Class: `SignalGenerationError`

**Inherits**: StrategyError
**Purpose**: Signal generation failures

```python
class SignalGenerationError(StrategyError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1703
```

#### Class: `ArbitrageError`

**Inherits**: StrategyError
**Purpose**: Arbitrage strategy errors

```python
class ArbitrageError(StrategyError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1711
```

#### Class: `BacktestError`

**Inherits**: TradingBotError
**Purpose**: Backtesting operation errors

```python
class BacktestError(TradingBotError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1719
```

#### Class: `BacktestConfigurationError`

**Inherits**: BacktestError
**Purpose**: Backtesting configuration errors

```python
class BacktestConfigurationError(BacktestError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1729
```

#### Class: `BacktestDataError`

**Inherits**: BacktestError
**Purpose**: Backtesting data-related errors

```python
class BacktestDataError(BacktestError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1738
```

#### Class: `BacktestExecutionError`

**Inherits**: BacktestError
**Purpose**: Backtesting execution errors

```python
class BacktestExecutionError(BacktestError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1747
```

#### Class: `BacktestServiceError`

**Inherits**: BacktestError
**Purpose**: Backtesting service unavailability errors

```python
class BacktestServiceError(BacktestError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1756
```

#### Class: `BacktestValidationError`

**Inherits**: BacktestError
**Purpose**: Backtesting validation errors

```python
class BacktestValidationError(BacktestError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1765
```

#### Class: `BacktestResultError`

**Inherits**: BacktestError
**Purpose**: Backtesting result processing errors

```python
class BacktestResultError(BacktestError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1774
```

#### Class: `BacktestMetricsError`

**Inherits**: BacktestError
**Purpose**: Backtesting metrics calculation errors

```python
class BacktestMetricsError(BacktestError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1782
```

#### Class: `BacktestStrategyError`

**Inherits**: BacktestError
**Purpose**: Backtesting strategy-related errors

```python
class BacktestStrategyError(BacktestError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1790
```

#### Class: `DatabaseError`

**Inherits**: TradingBotError
**Purpose**: Base class for all database-related errors

```python
class DatabaseError(TradingBotError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1803
```

#### Class: `DatabaseConnectionError`

**Inherits**: DatabaseError
**Purpose**: Database connection failures

```python
class DatabaseConnectionError(DatabaseError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1813
```

#### Class: `DatabaseQueryError`

**Inherits**: DatabaseError
**Purpose**: Database query failures

```python
class DatabaseQueryError(DatabaseError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1823
```

#### Class: `CircuitBreakerOpenError`

**Inherits**: TradingBotError
**Purpose**: Circuit breaker is open due to too many failures

```python
class CircuitBreakerOpenError(TradingBotError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1836
```

#### Class: `MaxRetriesExceededError`

**Inherits**: TradingBotError
**Purpose**: Maximum retry attempts exceeded

```python
class MaxRetriesExceededError(TradingBotError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 1847
```

#### Class: `ErrorCodeRegistry`

**Purpose**: Registry for all error codes in the system

```python
class ErrorCodeRegistry:
    def validate_code(cls, error_code: str) -> bool  # Line 1879
```

#### Class: `ExchangeErrorMapper`

**Purpose**: Maps exchange-specific errors to standardized exceptions

```python
class ExchangeErrorMapper:
    def map_error(cls, exchange: str, error_data: dict[str, Any]) -> TradingBotError  # Line 1943
    def map_binance_error(cls, error_data: dict[str, Any]) -> TradingBotError  # Line 1966
    def _map_binance(cls, error_data: dict[str, Any], exchange: str) -> TradingBotError  # Line 1971
    def map_coinbase_error(cls, error_data: dict[str, Any]) -> TradingBotError  # Line 2011
    def _map_coinbase(cls, error_data: dict[str, Any], exchange: str) -> TradingBotError  # Line 2016
    def map_okx_error(cls, error_data: dict[str, Any]) -> TradingBotError  # Line 2037
    def _map_okx(cls, error_data: dict[str, Any], exchange: str) -> TradingBotError  # Line 2042
    def _map_generic(cls, error_data: dict[str, Any], exchange: str) -> TradingBotError  # Line 2072
    def _extract_retry_after(error_data: dict[str, Any]) -> int | None  # Line 2105
```

#### Class: `ComponentError`

**Inherits**: TradingBotError
**Purpose**: Base class for all component-related errors

```python
class ComponentError(TradingBotError):
    def __init__(self, ...) -> None  # Line 2149
```

#### Class: `ServiceError`

**Inherits**: ComponentError
**Purpose**: Service layer errors for BaseService implementations

```python
class ServiceError(ComponentError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2177
```

#### Class: `RepositoryError`

**Inherits**: ComponentError
**Purpose**: Repository layer errors for BaseRepository implementations

```python
class RepositoryError(ComponentError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2185
```

#### Class: `FactoryError`

**Inherits**: ComponentError
**Purpose**: Factory pattern errors for BaseFactory implementations

```python
class FactoryError(ComponentError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2193
```

#### Class: `DependencyError`

**Inherits**: ComponentError
**Purpose**: Dependency injection and resolution errors

```python
class DependencyError(ComponentError):
    def __init__(self, message: str, dependency_name: str | None = None, **kwargs: Any) -> None  # Line 2201
```

#### Class: `HealthCheckError`

**Inherits**: ComponentError
**Purpose**: Health check system errors

```python
class HealthCheckError(ComponentError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2214
```

#### Class: `CircuitBreakerError`

**Inherits**: ComponentError
**Purpose**: Circuit breaker pattern errors

```python
class CircuitBreakerError(ComponentError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2222
```

#### Class: `EventError`

**Inherits**: ComponentError
**Purpose**: Event system errors for BaseEventEmitter

```python
class EventError(ComponentError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2230
```

#### Class: `EntityNotFoundError`

**Inherits**: DatabaseError
**Purpose**: Entity not found in repository

```python
class EntityNotFoundError(DatabaseError):
    def __init__(self, ...) -> None  # Line 2238
```

#### Class: `CreationError`

**Inherits**: FactoryError
**Purpose**: Factory creation errors

```python
class CreationError(FactoryError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2258
```

#### Class: `RegistrationError`

**Inherits**: FactoryError
**Purpose**: Factory registration errors

```python
class RegistrationError(FactoryError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2266
```

#### Class: `EventHandlerError`

**Inherits**: EventError
**Purpose**: Event handler execution errors

```python
class EventHandlerError(EventError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2274
```

#### Class: `MonitoringError`

**Inherits**: ComponentError
**Purpose**: Monitoring and metrics collection errors

```python
class MonitoringError(ComponentError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2282
```

#### Class: `AnalyticsError`

**Inherits**: ComponentError
**Purpose**: Analytics calculation and processing errors

```python
class AnalyticsError(ComponentError):
    def __init__(self, message: str, **kwargs: Any) -> None  # Line 2291
```

#### Class: `OptimizationError`

**Inherits**: TradingBotError
**Purpose**: Base class for all optimization-related errors

```python
class OptimizationError(TradingBotError):
    def __init__(self, ...) -> None  # Line 2310
```

#### Class: `ParameterValidationError`

**Inherits**: OptimizationError
**Purpose**: Parameter space validation errors

```python
class ParameterValidationError(OptimizationError):
    def __init__(self, ...) -> None  # Line 2340
```

#### Class: `OptimizationTimeoutError`

**Inherits**: OptimizationError
**Purpose**: Optimization process timeout errors

```python
class OptimizationTimeoutError(OptimizationError):
    def __init__(self, ...) -> None  # Line 2369
```

#### Class: `ConvergenceError`

**Inherits**: OptimizationError
**Purpose**: Optimization convergence failures

```python
class ConvergenceError(OptimizationError):
    def __init__(self, ...) -> None  # Line 2398
```

#### Class: `OverfittingError`

**Inherits**: OptimizationError
**Purpose**: Overfitting detection errors

```python
class OverfittingError(OptimizationError):
    def __init__(self, ...) -> None  # Line 2428
```

#### Class: `GeneticAlgorithmError`

**Inherits**: OptimizationError
**Purpose**: Genetic algorithm optimization errors

```python
class GeneticAlgorithmError(OptimizationError):
    def __init__(self, ...) -> None  # Line 2459
```

#### Class: `HyperparameterOptimizationError`

**Inherits**: OptimizationError
**Purpose**: Hyperparameter optimization errors

```python
class HyperparameterOptimizationError(OptimizationError):
    def __init__(self, ...) -> None  # Line 2487
```

#### Class: `PerformanceError`

**Inherits**: TradingBotError
**Purpose**: Base class for performance optimization errors

```python
class PerformanceError(TradingBotError):
    def __init__(self, ...) -> None  # Line 2518
```

#### Class: `CacheError`

**Inherits**: PerformanceError
**Purpose**: Cache operation errors

```python
class CacheError(PerformanceError):
    def __init__(self, ...) -> None  # Line 2550
```

#### Class: `MemoryOptimizationError`

**Inherits**: PerformanceError
**Purpose**: Memory optimization errors

```python
class MemoryOptimizationError(PerformanceError):
    def __init__(self, ...) -> None  # Line 2572
```

#### Class: `DatabaseOptimizationError`

**Inherits**: PerformanceError
**Purpose**: Database performance optimization errors

```python
class DatabaseOptimizationError(PerformanceError):
    def __init__(self, ...) -> None  # Line 2603
```

#### Class: `ConnectionPoolError`

**Inherits**: PerformanceError
**Purpose**: Connection pool optimization errors

```python
class ConnectionPoolError(PerformanceError):
    def __init__(self, ...) -> None  # Line 2632
```

#### Class: `ProfilingError`

**Inherits**: PerformanceError
**Purpose**: Performance profiling errors

```python
class ProfilingError(PerformanceError):
    def __init__(self, ...) -> None  # Line 2662
```

#### Functions:

```python
def create_error_from_dict(error_dict: dict[str, Any]) -> TradingBotError  # Line 2692
def is_retryable_error(error: Exception) -> bool  # Line 2718
def get_retry_delay(error: Exception) -> int | None  # Line 2732
```

### File: environment_aware_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.config.environment import ExchangeEnvironment`
- `from src.core.logging import get_logger`

#### Class: `EnvironmentMode`

**Inherits**: Enum
**Purpose**: Environment operation modes

```python
class EnvironmentMode(Enum):
```

#### Class: `EnvironmentContext`

**Inherits**: BaseModel
**Purpose**: Context information for environment-aware operations

```python
class EnvironmentContext(BaseModel):
```

#### Class: `EnvironmentAwareServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for environment-aware services

```python
class EnvironmentAwareServiceInterface(Protocol):
    async def switch_environment(self, environment: str | ExchangeEnvironment, exchange: str | None = None) -> bool  # Line 48
    async def validate_environment_operation(self, operation: str, context: EnvironmentContext) -> bool  # Line 56
    def get_environment_context(self, exchange: str) -> EnvironmentContext  # Line 64
```

#### Class: `EnvironmentAwareServiceMixin`

**Purpose**: Mixin class providing environment-aware functionality to services

```python
class EnvironmentAwareServiceMixin:
    def __init__(self, *args, **kwargs)  # Line 77
    def register_environment_switch_callback(self, callback: Callable) -> None  # Line 83
    def unregister_environment_switch_callback(self, callback: Callable) -> None  # Line 88
    async def _notify_environment_switch(self, old_env: EnvironmentContext, new_env: EnvironmentContext) -> None  # Line 93
    def get_environment_context(self, exchange: str) -> EnvironmentContext  # Line 108
    async def switch_environment(self, environment: str | ExchangeEnvironment, exchange: str | None = None) -> bool  # Line 138
    async def _validate_environment_switch(self, ...) -> bool  # Line 206
    async def _update_service_environment(self, context: EnvironmentContext) -> None  # Line 229
    async def validate_environment_operation(self, ...) -> bool  # Line 234
    async def _validate_production_operation(self, operation: str, context: EnvironmentContext) -> bool  # Line 273
    def get_environment_specific_config(self, exchange: str, config_key: str, default: Any = None) -> Any  # Line 282
    def is_environment_ready(self, exchange: str) -> bool  # Line 297
    async def get_environment_health_status(self) -> dict[str, Any]  # Line 311
```

#### Class: `EnvironmentAwareService`

**Inherits**: BaseService, EnvironmentAwareServiceMixin
**Purpose**: Base class for services that need environment awareness

```python
class EnvironmentAwareService(BaseService, EnvironmentAwareServiceMixin):
    def __init__(self, ...)  # Line 336
    def _initialize_environment_contexts(self) -> None  # Line 348
    async def _do_start(self) -> None  # Line 364
    async def _do_stop(self) -> None  # Line 373
    async def get_service_health(self) -> dict[str, Any]  # Line 381
```

### File: environment_orchestrator.py

**Key Imports:**
- `from src.core.config.environment import ExchangeEnvironment`
- `from src.core.config.environment import TradingEnvironment`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.integration.environment_aware_service import EnvironmentAwareService`

#### Class: `EnvironmentIntegrationOrchestrator`

**Inherits**: EnvironmentAwareService
**Purpose**: Orchestrates environment-aware integration across all T-Bot services

```python
class EnvironmentIntegrationOrchestrator(EnvironmentAwareService):
    def __init__(self, config: dict[str, Any] | None = None, correlation_id: str | None = None)  # Line 33
    async def _do_start(self) -> None  # Line 54
    def register_service(self, ...) -> None  # Line 77
    async def switch_global_environment(self, ...) -> dict[str, Any]  # Line 115
    async def switch_exchange_environment(self, ...) -> dict[str, Any]  # Line 224
    async def get_integrated_health_status(self) -> dict[str, Any]  # Line 349
    async def get_environment_status_summary(self) -> dict[str, Any]  # Line 443
    async def validate_environment_consistency(self) -> dict[str, Any]  # Line 463
    async def _validate_global_environment_switch(self, target_environment: TradingEnvironment) -> dict[str, Any]  # Line 519
    async def _validate_exchange_environment_switch(self, exchange: str, target_environment: ExchangeEnvironment) -> dict[str, Any]  # Line 556
    def _map_global_to_exchange_environments(self, global_env: TradingEnvironment) -> dict[str, ExchangeEnvironment]  # Line 579
    def _get_service_switch_order(self) -> list[str]  # Line 603
    async def _post_switch_validation(self, exchange: str, target_environment: ExchangeEnvironment) -> None  # Line 628
    async def _validate_cross_service_consistency(self, validation_results: dict[str, Any]) -> None  # Line 645
    async def _on_service_environment_switch(self, old_env: EnvironmentContext, new_env: EnvironmentContext) -> None  # Line 665
```

### File: logging.py

#### Class: `CorrelationContext`

**Purpose**: Context manager for correlation ID tracking

```python
class CorrelationContext:
    def __init__(self)  # Line 40
    def set_correlation_id(self, correlation_id: str) -> None  # Line 43
    def get_correlation_id(self) -> str | None  # Line 47
    def generate_correlation_id(self) -> str  # Line 51
    def correlation_context(self, correlation_id: str | None = None)  # Line 56
```

#### Class: `SecureLogger`

**Purpose**: Logger wrapper that prevents sensitive data from being logged

```python
class SecureLogger:
    def __init__(self, logger: structlog.BoundLogger)  # Line 325
    def _sanitize_data(self, data: dict[str, Any]) -> dict[str, Any]  # Line 356
    def info(self, message: str, **kwargs) -> None  # Line 370
    def warning(self, message: str, **kwargs) -> None  # Line 375
    def error(self, message: str, **kwargs) -> None  # Line 380
    def critical(self, message: str, **kwargs) -> None  # Line 385
    def debug(self, message: str, **kwargs) -> None  # Line 390
```

#### Class: `PerformanceMonitor`

**Purpose**: Performance monitoring utility for tracking operation metrics

```python
class PerformanceMonitor:
    def __init__(self, operation_name: str)  # Line 414
    def __enter__(self)  # Line 419
    def __exit__(self, exc_type, exc_val, exc_tb)  # Line 429
```

#### Functions:

```python
def _add_correlation_id(logger: Any, method_name: str, event_dict: dict[str, Any] | None) -> dict[str, Any]  # Line 68
def _safe_unicode_decoder(logger: Any, method_name: str, event_dict: dict[str, Any] | None) -> dict[str, Any]  # Line 78
def setup_logging(...) -> None  # Line 101
def get_logger(name: str) -> structlog.BoundLogger  # Line 198
def log_performance(func: Callable) -> Callable  # Line 210
def log_async_performance(func: Callable) -> Callable  # Line 262
def get_secure_logger(name: str) -> SecureLogger  # Line 396
def _cleanup_old_logs(log_dir: Path, log_name: str, retention_days: int) -> None  # Line 452
def setup_production_logging(log_dir: str = 'logs', app_name: str = 'trading-bot') -> None  # Line 487
def setup_development_logging() -> None  # Line 505
```

### File: memory_manager.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `CacheOptimizedList`

**Purpose**: Cache-optimized list implementation

```python
class CacheOptimizedList:
    def __init__(self)  # Line 45
    def clear(self)  # Line 49
    def append(self, item)  # Line 53
    def __len__(self)  # Line 57
    def __getitem__(self, key)  # Line 61
    def __setitem__(self, key, value)  # Line 65
```

#### Class: `DependencyInjectionMixin`

**Purpose**: Simple mixin for dependency injection to avoid circular imports

```python
class DependencyInjectionMixin:
    def __init__(self)  # Line 73
    def get_injector(self)  # Line 77
```

#### Class: `MemoryStats`

**Purpose**: Memory usage statistics

```python
class MemoryStats:
    def memory_pressure(self) -> float  # Line 99
```

#### Class: `ObjectPool`

**Inherits**: Generic[T]
**Purpose**: High-performance object pool for frequently used objects

```python
class ObjectPool(Generic[T]):
    def __init__(self, ...)  # Line 109
    def _populate_pool(self, initial_size: int | None = None)  # Line 147
    def borrow(self) -> T  # Line 158
    def return_object(self, obj: T)  # Line 174
    def get_stats(self) -> dict[str, Any]  # Line 200
    def clear(self)  # Line 216
```

#### Class: `MemoryLeakDetector`

**Purpose**: Detect and track memory leaks

```python
class MemoryLeakDetector:
    def __init__(self, check_interval: int = 300)  # Line 226
    async def start(self)  # Line 236
    async def _take_snapshot(self)  # Line 248
    async def _analyze_leaks(self)  # Line 276
    async def _log_top_growers(self, current: dict, previous: dict)  # Line 301
    def stop(self)  # Line 322
    def get_leak_report(self) -> dict[str, Any]  # Line 326
```

#### Class: `MemoryMappedCache`

**Purpose**: Memory-mapped cache for large datasets

```python
class MemoryMappedCache:
    def __init__(self, file_path: str, max_size: int = Any)  # Line 352
    def _open_mmap(self)  # Line 367
    def write_data(self, offset: int, data: bytes) -> bool  # Line 390
    def read_data(self, offset: int, length: int) -> bytes | None  # Line 406
    def close(self)  # Line 420
```

#### Class: `HighPerformanceMemoryManager`

**Inherits**: DependencyInjectionMixin
**Purpose**: Comprehensive memory management system

```python
class HighPerformanceMemoryManager(DependencyInjectionMixin):
    def __init__(self, config: 'Config | None' = None)  # Line 442
    def _resolve_config(self, config: 'Config | None') -> 'Config | None'  # Line 477
    def _initialize_pools(self)  # Line 508
    def _optimize_gc(self)  # Line 534
    async def start_monitoring(self)  # Line 556
    async def _monitoring_loop(self)  # Line 568
    def _get_memory_usage(self) -> MemoryStats  # Line 598
    async def _emergency_cleanup(self)  # Line 627
    async def _perform_gc(self)  # Line 650
    def get_pool(self, pool_name: str) -> ObjectPool | None  # Line 662
    def borrow_object(self, pool_name: str)  # Line 666
    def return_object(self, pool_name: str, obj)  # Line 673
    def track_object(self, obj)  # Line 679
    def get_memory_stats(self) -> MemoryStats  # Line 683
    def get_performance_report(self) -> dict[str, Any]  # Line 687
    def _generate_recommendations(self, stats: MemoryStats) -> list[str]  # Line 726
    async def stop_monitoring(self)  # Line 750
    def cleanup(self)  # Line 775
    def get_dependencies(self) -> list[str]  # Line 792
```

#### Functions:

```python
def initialize_memory_manager(config: 'Config') -> HighPerformanceMemoryManager  # Line 801
def get_memory_manager() -> HighPerformanceMemoryManager | None  # Line 808
def create_memory_manager_factory(config: 'Config | None' = None) -> Callable[[], HighPerformanceMemoryManager]  # Line 813
def borrow_dict() -> dict  # Line 825
def return_dict(obj: dict)  # Line 832
def borrow_list() -> list  # Line 838
def return_list(obj: list)  # Line 845
```

### File: memory_optimizer.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import PerformanceError`
- `from src.core.logging import get_logger`

#### Class: `MemoryCategory`

**Inherits**: Enum
**Purpose**: Categories of memory usage for optimization

```python
class MemoryCategory(Enum):
```

#### Class: `GCStrategy`

**Inherits**: Enum
**Purpose**: Garbage collection strategies

```python
class GCStrategy(Enum):
```

#### Class: `MemoryStats`

**Purpose**: Memory usage statistics

```python
class MemoryStats:
```

#### Class: `ObjectPoolStats`

**Purpose**: Statistics for object pools

```python
class ObjectPoolStats:
```

#### Class: `ObjectPool`

**Purpose**: High-performance object pool for frequently allocated objects

```python
class ObjectPool:
    def __init__(self, ...)  # Line 115
    async def acquire(self) -> Any  # Line 134
    async def release(self, obj: Any) -> None  # Line 160
    def _object_finalizer(self, weak_ref: weakref.ref) -> None  # Line 172
    def get_stats(self) -> ObjectPoolStats  # Line 176
    def clear(self) -> None  # Line 180
```

#### Class: `MemoryProfiler`

**Purpose**: Advanced memory profiler for detecting leaks and optimization opportunities

```python
class MemoryProfiler:
    def __init__(self)  # Line 194
    def start_profiling(self) -> None  # Line 200
    def stop_profiling(self) -> None  # Line 207
    def take_snapshot(self, label: str | None = None) -> None  # Line 214
    def detect_leaks(self) -> list[dict[str, Any]]  # Line 227
    def get_top_allocations(self, limit: int = 10) -> list[dict[str, Any]]  # Line 252
```

#### Class: `GarbageCollectionOptimizer`

**Purpose**: Garbage collection optimizer for trading workloads

```python
class GarbageCollectionOptimizer:
    def __init__(self)  # Line 281
    def set_strategy(self, strategy: GCStrategy) -> None  # Line 289
    def _apply_strategy(self) -> None  # Line 294
    def disable_gc_during_trading(self) -> None  # Line 311
    def enable_gc_after_trading(self) -> None  # Line 319
    def force_collection(self) -> dict[str, Any]  # Line 332
    def get_gc_stats(self) -> dict[str, Any]  # Line 371
```

#### Class: `MemoryOptimizer`

**Inherits**: BaseComponent
**Purpose**: Comprehensive memory optimizer for the T-Bot trading system

```python
class MemoryOptimizer(BaseComponent):
    def __init__(self, config: 'Config')  # Line 403
    def _initialize_object_pools(self) -> None  # Line 435
    async def initialize(self) -> None  # Line 454
    async def _start_monitoring(self) -> None  # Line 478
    async def _monitoring_loop(self) -> None  # Line 484
    async def _collect_memory_stats(self) -> MemoryStats  # Line 508
    async def _check_memory_alerts(self) -> None  # Line 551
    async def _check_memory_leaks(self) -> None  # Line 597
    async def _auto_optimize(self) -> None  # Line 620
    async def acquire_pooled_object(self, pool_name: str) -> Any  # Line 646
    async def release_pooled_object(self, pool_name: str, obj: Any) -> None  # Line 653
    async def _optimize_object_pools(self) -> None  # Line 660
    def track_category_usage(self, category: MemoryCategory, size_mb: float) -> None  # Line 671
    def add_alert_callback(self, callback: Callable) -> None  # Line 675
    def remove_alert_callback(self, callback: Callable) -> None  # Line 679
    async def optimize_for_trading_operation(self) -> None  # Line 684
    async def cleanup_after_trading_operation(self) -> None  # Line 704
    async def get_memory_report(self) -> dict[str, Any]  # Line 718
    def _get_current_alert_level(self, stats: MemoryStats) -> str  # Line 761
    async def force_memory_optimization(self) -> dict[str, Any]  # Line 770
    async def cleanup(self) -> None  # Line 821
```

#### Class: `TradingMemoryContext`

**Purpose**: Context manager for trading operations with memory optimization

```python
class TradingMemoryContext:
    def __init__(self, memory_optimizer: MemoryOptimizer)  # Line 883
    async def __aenter__(self)  # Line 886
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 890
```

#### Functions:

```python
def time_execution(func)  # Line 44
def memory_optimized_trading_operation(memory_optimizer: MemoryOptimizer)  # Line 895
```

### File: performance_monitor.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import PerformanceError`
- `from src.core.logging import get_logger`

#### Class: `MetricType`

**Inherits**: Enum
**Purpose**: Types of performance metrics

```python
class MetricType(Enum):
```

#### Class: `OperationType`

**Inherits**: Enum
**Purpose**: Types of operations being monitored

```python
class OperationType(Enum):
```

#### Class: `AlertLevel`

**Inherits**: Enum
**Purpose**: Alert severity levels

```python
class AlertLevel(Enum):
```

#### Class: `PerformanceMetric`

**Purpose**: Individual performance metric data point

```python
class PerformanceMetric:
```

#### Class: `LatencyStats`

**Purpose**: Latency statistics for an operation type

```python
class LatencyStats:
```

#### Class: `ThroughputStats`

**Purpose**: Throughput statistics for an operation type

```python
class ThroughputStats:
```

#### Class: `ResourceUsageStats`

**Purpose**: System resource usage statistics

```python
class ResourceUsageStats:
```

#### Class: `PerformanceAlert`

**Purpose**: Performance alert definition

```python
class PerformanceAlert:
```

#### Class: `PerformanceThresholds`

**Purpose**: Performance thresholds for alerting

```python
class PerformanceThresholds:
```

#### Class: `LatencyTracker`

**Purpose**: High-precision latency tracking for trading operations

```python
class LatencyTracker:
    def __init__(self, operation_type: OperationType)  # Line 174
    async def record_latency(self, latency_ms: float, metadata: dict[str, Any] | None = None) -> None  # Line 179
    def _percentile(self, sorted_list: list[float], percentile: float) -> float  # Line 202
    def get_stats(self) -> LatencyStats  # Line 215
    def reset_stats(self) -> None  # Line 219
```

#### Class: `ThroughputTracker`

**Purpose**: Throughput tracking for operations per second

```python
class ThroughputTracker:
    def __init__(self, operation_type: OperationType)  # Line 227
    async def record_operation(self) -> None  # Line 233
    def _calculate_throughput(self) -> None  # Line 245
    def get_stats(self) -> ThroughputStats  # Line 258
```

#### Class: `PrometheusMetricsCollector`

**Purpose**: Prometheus metrics collector for external monitoring

```python
class PrometheusMetricsCollector:
    def __init__(self)  # Line 266
    def record_latency(self, ...) -> None  # Line 319
    def increment_operation(self, operation_type: str, status: str, labels: dict[str, str] | None = None) -> None  # Line 328
    def update_resource_usage(self, cpu_percent: float, memory_bytes: float) -> None  # Line 339
    def get_metrics(self) -> str  # Line 344
```

#### Class: `PerformanceMonitor`

**Inherits**: BaseComponent
**Purpose**: Comprehensive performance monitoring system for T-Bot trading operations

```python
class PerformanceMonitor(BaseComponent):
    def __init__(self, config: 'Config')  # Line 357
    async def initialize(self) -> None  # Line 400
    async def _start_monitoring(self) -> None  # Line 417
    async def _monitoring_loop(self) -> None  # Line 423
    async def record_operation_start(self, operation_type: OperationType, metadata: dict[str, Any] | None = None) -> str  # Line 447
    async def record_operation_end(self, ...) -> float  # Line 479
    async def record_simple_latency(self, ...) -> None  # Line 528
    async def _collect_resource_usage(self) -> None  # Line 549
    async def _check_performance_alerts(self) -> None  # Line 593
    async def _check_latency_alerts(self, alerts_to_add: list, alerts_to_remove: list) -> None  # Line 606
    def _create_latency_alert_if_needed(self, ...)  # Line 625
    def _check_resource_usage_alerts(self, alerts_to_add: list) -> None  # Line 662
    def _create_resource_alert(self, ...)  # Line 691
    def _check_error_rate_alerts(self, alerts_to_add: list) -> None  # Line 722
    def _create_error_rate_alert(self, ...)  # Line 742
    def _get_latency_thresholds(self, operation_type: OperationType) -> tuple[float, float]  # Line 773
    async def _process_alerts(self, alerts_to_add: list[PerformanceAlert], alerts_to_remove: list[str]) -> None  # Line 795
    async def _update_prometheus_metrics(self) -> None  # Line 836
    async def _detect_performance_regressions(self) -> None  # Line 842
    async def _load_baseline_metrics(self) -> None  # Line 848
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None  # Line 853
    def remove_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None  # Line 857
    async def get_performance_summary(self) -> dict[str, Any]  # Line 862
    def get_prometheus_metrics(self) -> str  # Line 915
    async def reset_statistics(self) -> None  # Line 919
    async def cleanup(self) -> None  # Line 941
```

#### Class: `OperationTracker`

**Purpose**: Context manager for automatic operation tracking

```python
class OperationTracker:
    def __init__(self, ...)  # Line 965
    async def __aenter__(self)  # Line 976
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 982
```

#### Functions:

```python
def time_execution(func)  # Line 31
def track_performance(operation_type: OperationType, monitor: PerformanceMonitor | None = None)  # Line 989
```

### File: performance_optimizer.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.base.interfaces import DatabaseServiceInterface`
- `from src.core.exceptions import PerformanceError`
- `from src.core.logging import get_logger`

#### Class: `PerformanceOptimizer`

**Inherits**: BaseComponent
**Purpose**: Integrated performance optimizer that coordinates all optimization components
to achieve optimal tra

```python
class PerformanceOptimizer(BaseComponent):
    def __init__(self, config: 'Config')  # Line 57
    async def initialize(self) -> None  # Line 92
    async def _initialize_performance_monitor(self) -> None  # Line 135
    async def _initialize_memory_optimizer(self) -> None  # Line 142
    async def _initialize_cache_layer(self) -> None  # Line 149
    async def _initialize_database_service(self) -> None  # Line 156
    async def _initialize_connection_pools(self) -> None  # Line 170
    async def _initialize_trading_profiler(self) -> None  # Line 178
    async def _start_coordination_tasks(self) -> None  # Line 186
    async def _setup_alert_handlers(self) -> None  # Line 196
    async def _coordination_loop(self) -> None  # Line 204
    async def _reporting_loop(self) -> None  # Line 237
    async def _collect_comprehensive_metrics(self) -> dict[str, Any]  # Line 272
    async def _analyze_optimization_opportunities(self, metrics: dict[str, Any]) -> list[dict[str, Any]]  # Line 316
    async def _execute_optimizations(self, opportunities: list[dict[str, Any]]) -> None  # Line 404
    async def _handle_performance_alert(self, alert) -> None  # Line 453
    async def _handle_memory_alert(self, alert_level: str, message: str, stats) -> None  # Line 485
    async def _run_initial_assessment(self) -> None  # Line 505
    def optimize_trading_operation(self, operation: 'TradingOperation')  # Line 536
    async def get_performance_report(self) -> dict[str, Any]  # Line 547
    async def force_optimization(self) -> dict[str, Any]  # Line 650
    async def cleanup(self) -> None  # Line 703
    async def _cleanup_background_tasks(self) -> None  # Line 723
    async def _cancel_task_safely(self, task, task_name: str) -> None  # Line 734
    async def _cleanup_optimization_components(self) -> None  # Line 744
    async def _cleanup_single_component(self, component, component_name: str) -> None  # Line 765
    async def _cleanup_database_service(self) -> None  # Line 773
    def _clear_history_data(self) -> None  # Line 781
    def is_initialized(self) -> bool  # Line 792
    def get_component_status(self) -> dict[str, bool]  # Line 796
```

### File: trading_profiler.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import PerformanceError`
- `from src.core.logging import get_logger`

#### Class: `TradingOperation`

**Inherits**: Enum
**Purpose**: Specific trading operations that require optimization

```python
class TradingOperation(Enum):
```

#### Class: `OptimizationLevel`

**Inherits**: Enum
**Purpose**: Levels of optimization to apply

```python
class OptimizationLevel(Enum):
```

#### Class: `OperationProfile`

**Purpose**: Detailed profile of a trading operation

```python
class OperationProfile:
```

#### Class: `TradingBenchmark`

**Purpose**: Benchmark results for trading operations

```python
class TradingBenchmark:
```

#### Class: `TradingProfiler`

**Purpose**: High-precision profiler for individual trading operations

```python
class TradingProfiler:
    def __init__(self, operation: TradingOperation)  # Line 117
    def start_profiling(self, enable_memory_tracing: bool = False) -> None  # Line 125
    def stop_profiling(self) -> dict[str, Any]  # Line 140
    def _analyze_cpu_profile(self) -> dict[str, Any]  # Line 172
    def _parse_hot_functions(self, stats: pstats.Stats) -> list[dict[str, Any]]  # Line 195
    def _identify_bottlenecks(self, stats: pstats.Stats) -> list[dict[str, Any]]  # Line 218
    def _generate_optimization_suggestion(self, function_name: str, call_count: int, cumulative_time: float) -> str  # Line 246
    def get_performance_summary(self) -> dict[str, Any]  # Line 269
```

#### Class: `TradingOperationOptimizer`

**Inherits**: BaseComponent
**Purpose**: Comprehensive optimizer for critical trading operations

```python
class TradingOperationOptimizer(BaseComponent):
    def __init__(self, config: 'Config', performance_monitor: 'PerformanceMonitor')  # Line 301
    async def initialize(self) -> None  # Line 348
    async def _start_background_optimization(self) -> None  # Line 370
    async def _optimization_loop(self) -> None  # Line 376
    async def profile_operation(self, operation: TradingOperation, func: Callable, *args, **kwargs) -> tuple[Any, dict[str, Any]]  # Line 399
    def optimize_function(self, operation: TradingOperation)  # Line 476
    async def _queue_optimization_recommendation(self, ...) -> None  # Line 543
    def _generate_specific_optimizations(self, operation: TradingOperation, profiling_data: dict[str, Any]) -> list[str]  # Line 570
    async def _analyze_performance_trends(self) -> None  # Line 631
    async def _generate_optimization_recommendations(self) -> None  # Line 655
    async def _update_benchmarks(self) -> None  # Line 690
    async def _log_optimization_summary(self) -> None  # Line 719
    async def _run_initial_benchmarks(self) -> None  # Line 745
    async def get_optimization_report(self) -> dict[str, Any]  # Line 751
    async def force_optimization_analysis(self) -> dict[str, Any]  # Line 800
    async def cleanup(self) -> None  # Line 824
```

#### Class: `TradingOperationContext`

**Purpose**: Context manager for automatic trading operation profiling

```python
class TradingOperationContext:
    def __init__(self, optimizer: TradingOperationOptimizer, operation: TradingOperation)  # Line 917
    async def __aenter__(self)  # Line 923
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 932
```

### File: resource_manager.py

#### Class: `ResourceType`

**Inherits**: Enum
**Purpose**: Types of resources being managed

```python
class ResourceType(Enum):
```

#### Class: `ResourceState`

**Inherits**: Enum
**Purpose**: Resource lifecycle states

```python
class ResourceState(Enum):
```

#### Class: `ResourceInfo`

**Purpose**: Information about a managed resource

```python
class ResourceInfo:
    def touch(self) -> None  # Line 66
```

#### Class: `ResourceMonitor`

**Purpose**: Monitors resource usage and detects leaks

```python
class ResourceMonitor:
    def __init__(self) -> None  # Line 75
    def get_memory_usage(self) -> dict[str, Any]  # Line 81
    def get_connection_stats(self) -> dict[str, Any]  # Line 94
    def get_gc_stats(self) -> dict[str, Any]  # Line 120
```

#### Class: `ResourceManager`

**Purpose**: Centralized resource lifecycle manager

```python
class ResourceManager:
    def __init__(self) -> None  # Line 156
    def configure(self, ...) -> None  # Line 190
    async def start(self)  # Line 222
    async def stop(self)  # Line 236
    def register_resource(self, ...) -> str  # Line 264
    async def unregister_resource(self, resource_id: str)  # Line 302
    def touch_resource(self, resource_id: str)  # Line 348
    async def cleanup_all_resources(self)  # Line 357
    async def _cleanup_loop(self)  # Line 398
    async def _monitoring_loop(self)  # Line 428
    async def _perform_cleanup(self)  # Line 461
    async def _cleanup_idle_resource(self, resource_id: str)  # Line 511
    def _detect_leaks(self)  # Line 519
    def _log_resource_stats(self) -> None  # Line 545
    def get_resource_stats(self) -> dict[str, Any]  # Line 566
```

#### Functions:

```python
def get_resource_manager() -> ResourceManager  # Line 591
async def initialize_resource_manager()  # Line 599
async def shutdown_resource_manager()  # Line 606
```

### File: service_manager.py

**Key Imports:**
- `from src.core.base.interfaces import DIContainer`
- `from src.core.exceptions import DependencyError`
- `from src.core.exceptions import ServiceError`

#### Class: `ServiceManager`

**Purpose**: Centralized service manager for dependency resolution and lifecycle management

```python
class ServiceManager:
    def __init__(self, injector: DIContainer) -> None  # Line 30
    def register_service(self, ...) -> None  # Line 40
    def _create_service(self, service_name: str) -> Any  # Line 87
    def _resolve_dependencies(self, service_name: str, dependencies: list[str]) -> dict[str, Any]  # Line 119
    def _instantiate_service(self, config: dict[str, Any], resolved_deps: dict[str, Any]) -> Any  # Line 146
    def _build_constructor_args(self, ...) -> dict[str, Any]  # Line 159
    def _get_parameter_mapping(self) -> dict[str, str | None]  # Line 189
    def _should_skip_parameter(self, param: str) -> bool  # Line 206
    def _configure_service_instance(self, service_instance: Any) -> None  # Line 211
    def _calculate_startup_order(self) -> list[str]  # Line 222
    async def start_all_services(self) -> None  # Line 255
    async def _start_service(self, service_name: str) -> None  # Line 278
    async def stop_all_services(self) -> None  # Line 302
    async def _stop_service(self, service_name: str) -> None  # Line 329
    def get_service(self, service_name: str) -> Any  # Line 357
    def is_service_running(self, service_name: str) -> bool  # Line 378
    def get_running_services(self) -> list[str]  # Line 382
    async def restart_service(self, service_name: str) -> None  # Line 386
    async def health_check_all(self) -> dict[str, Any]  # Line 400
    async def _check_service_health(self, service_name: str) -> dict[str, Any]  # Line 461
    def _normalize_health_status(self, status_result: Any) -> dict[str, Any]  # Line 487
    def _aggregate_health_status(self, service_statuses: dict[str, dict[str, Any]]) -> str  # Line 518
```

#### Functions:

```python
def get_service_manager(injector_instance: Any = None) -> ServiceManager  # Line 552
def _register_core_infrastructure_factories(injector: Any, service_manager: ServiceManager) -> None  # Line 573
def register_core_services(config: Any) -> None  # Line 592
def register_business_services(config: Any) -> None  # Line 761
def register_application_services(config: Any) -> None  # Line 829
async def initialize_all_services(config: Any) -> ServiceManager  # Line 871
async def shutdown_all_services() -> None  # Line 906
```

### File: task_manager.py

#### Class: `TaskState`

**Inherits**: Enum
**Purpose**: Task lifecycle states

```python
class TaskState(Enum):
```

#### Class: `TaskPriority`

**Inherits**: Enum
**Purpose**: Task priority levels

```python
class TaskPriority(Enum):
```

#### Class: `TaskInfo`

**Purpose**: Information about a managed task

```python
class TaskInfo:
```

#### Class: `TaskManager`

**Purpose**: Comprehensive task lifecycle manager

```python
class TaskManager:
    def __init__(self) -> None  # Line 85
    async def start(self)  # Line 126
    async def stop(self)  # Line 145
    async def create_task(self, ...) -> str  # Line 182
    async def cancel_task(self, task_id: str) -> bool  # Line 232
    async def _cancel_all_tasks(self)  # Line 259
    def _cleanup_task_resource(self, task_id: str)  # Line 293
    async def _worker_loop(self, worker_name: str)  # Line 298
    async def _execute_task(self, task_id: str, coro: Coroutine, worker_name: str)  # Line 351
    async def _cleanup_loop(self)  # Line 442
    async def _monitor_loop(self)  # Line 467
    async def _cleanup_completed_tasks(self)  # Line 492
    def _log_task_stats(self) -> None  # Line 528
    def get_task_stats(self) -> dict[str, Any]  # Line 550
    def get_task_info(self, task_id: str) -> dict[str, Any] | None  # Line 577
```

#### Functions:

```python
def get_task_manager() -> TaskManager  # Line 602
async def initialize_task_manager()  # Line 610
async def shutdown_task_manager()  # Line 617
```

### File: base.py

#### Class: `AlertSeverity`

**Inherits**: Enum
**Purpose**: Alert severity levels for monitoring and alerting system

```python
class AlertSeverity(Enum):
```

#### Class: `TradingMode`

**Inherits**: Enum
**Purpose**: Trading mode enumeration for different execution environments

```python
class TradingMode(Enum):
    def is_real_money(self) -> bool  # Line 61
    def allows_testing(self) -> bool  # Line 65
    def from_string(cls, value: str) -> 'TradingMode'  # Line 70
```

#### Class: `ExchangeType`

**Inherits**: Enum
**Purpose**: Exchange types for API integration and rate limiting coordination

```python
class ExchangeType(Enum):
    def get_rate_limit(self) -> int  # Line 117
    def supports_websocket(self) -> bool  # Line 128
    def get_base_url(self) -> str  # Line 132
```

#### Class: `MarketType`

**Inherits**: Enum
**Purpose**: Market types for different trading venues and instruments

```python
class MarketType(Enum):
    def requires_margin(self) -> bool  # Line 164
    def has_expiration(self) -> bool  # Line 168
    def supports_leverage(self) -> bool  # Line 172
```

#### Class: `RequestType`

**Inherits**: Enum
**Purpose**: Request types for API coordination and rate limiting

```python
class RequestType(Enum):
    def get_priority(self) -> int  # Line 207
    def is_modifying_operation(self) -> bool  # Line 223
```

#### Class: `ConnectionType`

**Inherits**: Enum
**Purpose**: WebSocket connection types for different data streams

```python
class ConnectionType(Enum):
    def is_public_stream(self) -> bool  # Line 258
    def requires_authentication(self) -> bool  # Line 270
    def get_update_frequency(self) -> str  # Line 274
```

#### Class: `ValidationLevel`

**Inherits**: Enum
**Purpose**: Data validation severity levels used across the system

```python
class ValidationLevel(Enum):
    def should_halt_system(self) -> bool  # Line 315
    def requires_immediate_attention(self) -> bool  # Line 319
    def get_numeric_value(self) -> int  # Line 323
    def __lt__(self, other: 'ValidationLevel') -> bool  # Line 334
```

#### Class: `ValidationResult`

**Inherits**: Enum
**Purpose**: Data validation result enumeration with enhanced functionality

```python
class ValidationResult(Enum):
    def is_success(self) -> bool  # Line 359
    def is_failure(self) -> bool  # Line 363
    def should_proceed(self) -> bool  # Line 367
    def get_severity(self) -> ValidationLevel  # Line 371
```

#### Class: `BaseValidatedModel`

**Inherits**: BaseModel
**Purpose**: Enhanced base model with comprehensive validation and utilities

```python
class BaseValidatedModel(BaseModel):
    def mark_updated(self) -> None  # Line 402
    def to_dict(self) -> dict[str, Any]  # Line 406
    def to_json(self) -> str  # Line 410
    def from_dict(cls, data: dict[str, Any]) -> 'BaseValidatedModel'  # Line 415
    def from_json(cls, json_str: str) -> 'BaseValidatedModel'  # Line 420
    def add_metadata(self, key: str, value: Any) -> None  # Line 424
    def get_metadata(self, key: str, default: Any = None) -> Any  # Line 429
    def has_metadata(self, key: str) -> bool  # Line 433
    def __get_pydantic_json_schema__(cls, schema_generator: Any, handler: Any) -> dict[str, Any]  # Line 444
    def model_dump_json(self, **kwargs) -> str  # Line 449
```

#### Class: `FinancialBaseModel`

**Inherits**: BaseValidatedModel
**Purpose**: Base model for financial data with Decimal precision handling

```python
class FinancialBaseModel(BaseValidatedModel):
    def convert_financial_floats(cls, v: Any, info) -> Any  # Line 476
    def to_dict_with_decimals(self) -> dict[str, Any]  # Line 497
    def validate_financial_precision(self) -> bool  # Line 522
    def model_dump_json(self, **kwargs) -> str  # Line 544
```

### File: bot.py

#### Class: `BotStatus`

**Inherits**: Enum
**Purpose**: Bot operational status

```python
class BotStatus(Enum):
```

#### Class: `BotType`

**Inherits**: Enum
**Purpose**: Bot type classification

```python
class BotType(Enum):
```

#### Class: `BotPriority`

**Inherits**: Enum
**Purpose**: Bot execution priority

```python
class BotPriority(Enum):
```

#### Class: `ResourceType`

**Inherits**: Enum
**Purpose**: System resource types

```python
class ResourceType(Enum):
```

#### Class: `BotConfiguration`

**Inherits**: BaseModel
**Purpose**: Bot configuration parameters

```python
class BotConfiguration(BaseModel):
    def bot_name(self) -> str  # Line 120
```

#### Class: `BotMetrics`

**Inherits**: BaseModel
**Purpose**: Bot performance and resource metrics

```python
class BotMetrics(BaseModel):
```

#### Class: `BotState`

**Inherits**: BaseModel
**Purpose**: Bot runtime state

```python
class BotState(BaseModel):
```

#### Class: `ResourceAllocation`

**Inherits**: BaseModel
**Purpose**: Resource allocation for bots

```python
class ResourceAllocation(BaseModel):
```

### File: capital.py

#### Class: `CapitalFundFlow`

**Inherits**: BaseModel
**Purpose**: Extended fund flow for capital management operations

```python
class CapitalFundFlow(BaseModel):
```

#### Class: `CapitalCurrencyExposure`

**Inherits**: BaseModel
**Purpose**: Extended currency exposure for capital management

```python
class CapitalCurrencyExposure(BaseModel):
```

#### Class: `CapitalExchangeAllocation`

**Inherits**: BaseModel
**Purpose**: Extended exchange allocation for capital management

```python
class CapitalExchangeAllocation(BaseModel):
```

#### Class: `ExtendedCapitalProtection`

**Inherits**: BaseModel
**Purpose**: Extended capital protection with additional fields

```python
class ExtendedCapitalProtection(BaseModel):
```

#### Class: `ExtendedWithdrawalRule`

**Inherits**: BaseModel
**Purpose**: Extended withdrawal rule for fund flow manager

```python
class ExtendedWithdrawalRule(BaseModel):
```

### File: data.py

#### Class: `QualityLevel`

**Inherits**: Enum
**Purpose**: Data quality level classification

```python
class QualityLevel(Enum):
```

#### Class: `DriftType`

**Inherits**: Enum
**Purpose**: Data drift type classification

```python
class DriftType(Enum):
```

#### Class: `IngestionMode`

**Inherits**: Enum
**Purpose**: Data ingestion mode

```python
class IngestionMode(Enum):
```

#### Class: `PipelineStatus`

**Inherits**: Enum
**Purpose**: Data pipeline status

```python
class PipelineStatus(Enum):
```

#### Class: `ProcessingStep`

**Inherits**: Enum
**Purpose**: Data processing pipeline steps

```python
class ProcessingStep(Enum):
```

#### Class: `StorageMode`

**Inherits**: Enum
**Purpose**: Data storage mode

```python
class StorageMode(Enum):
```

#### Class: `ErrorPattern`

**Purpose**: Common error patterns in data processing

```python
class ErrorPattern:
```

#### Class: `MLMarketData`

**Inherits**: BaseModel
**Purpose**: Market data structure for ML processing

```python
class MLMarketData(BaseModel):
```

#### Class: `PredictionResult`

**Inherits**: BaseModel
**Purpose**: ML prediction result structure

```python
class PredictionResult(BaseModel):
```

#### Class: `FeatureSet`

**Inherits**: BaseModel
**Purpose**: Feature set for ML models

```python
class FeatureSet(BaseModel):
```

### File: execution.py

#### Class: `ExecutionAlgorithm`

**Inherits**: Enum
**Purpose**: Execution algorithm types

```python
class ExecutionAlgorithm(Enum):
```

#### Class: `ExecutionStatus`

**Inherits**: Enum
**Purpose**: Execution status

```python
class ExecutionStatus(Enum):
```

#### Class: `SlippageType`

**Inherits**: Enum
**Purpose**: Slippage classification

```python
class SlippageType(Enum):
```

#### Class: `ExecutionInstruction`

**Inherits**: BaseModel
**Purpose**: Execution instruction for order placement

```python
class ExecutionInstruction(BaseModel):
```

#### Class: `ExecutionResult`

**Inherits**: BaseModel
**Purpose**: Result of execution algorithm

```python
class ExecutionResult(BaseModel):
    def fill_percentage(self) -> Decimal  # Line 127
    def is_complete(self) -> bool  # Line 136
```

#### Class: `SlippageMetrics`

**Inherits**: BaseModel
**Purpose**: Slippage analysis metrics

```python
class SlippageMetrics(BaseModel):
```

### File: market.py

#### Class: `ExchangeStatus`

**Inherits**: Enum
**Purpose**: Exchange operational status

```python
class ExchangeStatus(Enum):
```

#### Class: `MarketData`

**Inherits**: BaseModel
**Purpose**: Market data snapshot

```python
class MarketData(BaseModel):
    def price(self) -> Decimal  # Line 41
    def high_price(self) -> Decimal  # Line 46
    def low_price(self) -> Decimal  # Line 51
    def open_price(self) -> Decimal  # Line 56
    def close_price(self) -> Decimal  # Line 61
    def bid(self) -> Decimal | None  # Line 66
    def ask(self) -> Decimal | None  # Line 71
```

#### Class: `Ticker`

**Inherits**: BaseModel
**Purpose**: Market ticker information

```python
class Ticker(BaseModel):
    def spread(self) -> Decimal  # Line 97
    def spread_percent(self) -> Decimal  # Line 102
```

#### Class: `OrderBookLevel`

**Inherits**: BaseModel
**Purpose**: Single level in order book

```python
class OrderBookLevel(BaseModel):
```

#### Class: `OrderBook`

**Inherits**: BaseModel
**Purpose**: Order book snapshot

```python
class OrderBook(BaseModel):
    def best_bid(self) -> OrderBookLevel | None  # Line 130
    def best_ask(self) -> OrderBookLevel | None  # Line 135
    def spread(self) -> Decimal | None  # Line 140
    def get_depth(self, side: str, levels: int = 5) -> Decimal  # Line 146
```

#### Class: `Trade`

**Inherits**: BaseModel
**Purpose**: Represents a trade executed on an exchange

```python
class Trade(BaseModel):
```

#### Class: `ExchangeGeneralInfo`

**Inherits**: BaseModel
**Purpose**: General exchange information and capabilities

```python
class ExchangeGeneralInfo(BaseModel):
```

#### Class: `ExchangeInfo`

**Inherits**: BaseModel
**Purpose**: Exchange trading rules and information for a specific symbol

```python
class ExchangeInfo(BaseModel):
    def round_price(self, price: Decimal) -> Decimal  # Line 197
    def round_quantity(self, quantity: Decimal) -> Decimal  # Line 201
    def validate_order(self, price: Decimal, quantity: Decimal) -> bool  # Line 205
```

### File: risk.py

#### Class: `RiskLevel`

**Inherits**: Enum
**Purpose**: Risk level classification

```python
class RiskLevel(Enum):
```

#### Class: `PositionSizeMethod`

**Inherits**: Enum
**Purpose**: Position sizing methodology

```python
class PositionSizeMethod(Enum):
```

#### Class: `CircuitBreakerStatus`

**Inherits**: Enum
**Purpose**: Circuit breaker status

```python
class CircuitBreakerStatus(Enum):
```

#### Class: `CircuitBreakerType`

**Inherits**: Enum
**Purpose**: Circuit breaker trigger type

```python
class CircuitBreakerType(Enum):
```

#### Class: `EmergencyAction`

**Inherits**: Enum
**Purpose**: Emergency action types

```python
class EmergencyAction(Enum):
```

#### Class: `AllocationStrategy`

**Inherits**: Enum
**Purpose**: Capital allocation strategy

```python
class AllocationStrategy(Enum):
```

#### Class: `RiskMetrics`

**Inherits**: BaseModel
**Purpose**: Risk metrics for positions and strategies

```python
class RiskMetrics(BaseModel):
```

#### Class: `PositionLimits`

**Inherits**: BaseModel
**Purpose**: Position size and risk limits

```python
class PositionLimits(BaseModel):
```

#### Class: `RiskLimits`

**Inherits**: BaseModel
**Purpose**: Risk limits configuration

```python
class RiskLimits(BaseModel):
```

#### Class: `RiskAlert`

**Inherits**: BaseModel
**Purpose**: Risk alert notification

```python
class RiskAlert(BaseModel):
```

#### Class: `CircuitBreakerEvent`

**Inherits**: BaseModel
**Purpose**: Circuit breaker trigger event

```python
class CircuitBreakerEvent(BaseModel):
```

#### Class: `CapitalAllocation`

**Inherits**: BaseModel
**Purpose**: Capital allocation for strategies and positions

```python
class CapitalAllocation(BaseModel):
```

#### Class: `FundFlow`

**Inherits**: BaseModel
**Purpose**: Fund flow tracking for deposits and withdrawals

```python
class FundFlow(BaseModel):
```

#### Class: `CapitalMetrics`

**Inherits**: BaseModel
**Purpose**: Overall capital and portfolio metrics

```python
class CapitalMetrics(BaseModel):
```

#### Class: `CurrencyExposure`

**Inherits**: BaseModel
**Purpose**: Currency exposure tracking

```python
class CurrencyExposure(BaseModel):
```

#### Class: `ExchangeAllocation`

**Inherits**: BaseModel
**Purpose**: Capital allocation across exchanges

```python
class ExchangeAllocation(BaseModel):
```

#### Class: `WithdrawalRule`

**Inherits**: BaseModel
**Purpose**: Automated withdrawal rules

```python
class WithdrawalRule(BaseModel):
```

#### Class: `CapitalProtection`

**Inherits**: BaseModel
**Purpose**: Capital protection settings

```python
class CapitalProtection(BaseModel):
```

#### Class: `PortfolioState`

**Inherits**: BaseModel
**Purpose**: Complete portfolio state representation

```python
class PortfolioState(BaseModel):
```

#### Class: `PortfolioMetrics`

**Inherits**: BaseModel
**Purpose**: Unified portfolio metrics model for cross-module consistency

```python
class PortfolioMetrics(BaseModel):
```

### File: strategy.py

#### Class: `StrategyType`

**Inherits**: Enum
**Purpose**: Strategy type enumeration

```python
class StrategyType(Enum):
```

#### Class: `StrategyStatus`

**Inherits**: Enum
**Purpose**: Strategy operational status

```python
class StrategyStatus(Enum):
```

#### Class: `MarketRegime`

**Inherits**: Enum
**Purpose**: Market regime classification

```python
class MarketRegime(Enum):
```

#### Class: `NewsSentiment`

**Inherits**: Enum
**Purpose**: News sentiment classification

```python
class NewsSentiment(Enum):
```

#### Class: `SocialSentiment`

**Inherits**: Enum
**Purpose**: Social media sentiment classification

```python
class SocialSentiment(Enum):
```

#### Class: `StrategyConfig`

**Inherits**: BaseModel
**Purpose**: Strategy configuration parameters

```python
class StrategyConfig(BaseModel):
```

#### Class: `StrategyMetrics`

**Inherits**: BaseModel
**Purpose**: Strategy performance metrics

```python
class StrategyMetrics(BaseModel):
    def update_win_rate(self) -> None  # Line 144
    def calculate_profit_factor(self) -> Decimal  # Line 149
```

#### Class: `RegimeChangeEvent`

**Inherits**: BaseModel
**Purpose**: Market regime change event

```python
class RegimeChangeEvent(BaseModel):
```

### File: trading.py

#### Class: `SignalDirection`

**Inherits**: Enum
**Purpose**: Signal direction for trading decisions

```python
class SignalDirection(Enum):
```

#### Class: `OrderSide`

**Inherits**: Enum
**Purpose**: Order side for buy/sell operations

```python
class OrderSide(Enum):
```

#### Class: `PositionSide`

**Inherits**: Enum
**Purpose**: Position side for long/short positions

```python
class PositionSide(Enum):
```

#### Class: `PositionStatus`

**Inherits**: Enum
**Purpose**: Position status

```python
class PositionStatus(Enum):
```

#### Class: `OrderType`

**Inherits**: Enum
**Purpose**: Order type for different execution strategies

```python
class OrderType(Enum):
```

#### Class: `OrderStatus`

**Inherits**: Enum
**Purpose**: Order status in exchange systems

```python
class OrderStatus(Enum):
```

#### Class: `TimeInForce`

**Inherits**: Enum
**Purpose**: Time in force for order execution

```python
class TimeInForce(Enum):
```

#### Class: `TradeState`

**Inherits**: Enum
**Purpose**: Trade lifecycle states

```python
class TradeState(Enum):
```

#### Class: `Signal`

**Inherits**: BaseModel
**Purpose**: Trading signal with direction and metadata - consistent validation patterns

```python
class Signal(BaseModel):
    def validate_strength(cls, v: Decimal) -> Decimal  # Line 117
    def validate_symbol(cls, v: str) -> str  # Line 137
    def validate_timestamp(cls, v: datetime) -> datetime  # Line 189
```

#### Class: `OrderRequest`

**Inherits**: BaseModel
**Purpose**: Request to create an order

```python
class OrderRequest(BaseModel):
    def validate_symbol(cls, v: str) -> str  # Line 219
    def validate_quantity(cls, v: Decimal) -> Decimal  # Line 227
    def validate_price(cls, v: Decimal | None) -> Decimal | None  # Line 256
    def validate_quote_quantity(cls, v: Decimal | None) -> Decimal | None  # Line 283
```

#### Class: `OrderResponse`

**Inherits**: BaseModel
**Purpose**: Response from order creation

```python
class OrderResponse(BaseModel):
    def id(self) -> str  # Line 328
    def remaining_quantity(self) -> Decimal  # Line 333
```

#### Class: `Order`

**Inherits**: BaseModel
**Purpose**: Complete order information

```python
class Order(BaseModel):
    def is_filled(self) -> bool  # Line 365
    def is_active(self) -> bool  # Line 369
```

#### Class: `Position`

**Inherits**: BaseModel
**Purpose**: Trading position information

```python
class Position(BaseModel):
    def is_open(self) -> bool  # Line 391
    def calculate_pnl(self, current_price: Decimal) -> Decimal  # Line 395
```

#### Class: `Trade`

**Inherits**: BaseModel
**Purpose**: Executed trade information

```python
class Trade(BaseModel):
```

#### Class: `Balance`

**Inherits**: BaseModel
**Purpose**: Account balance information

```python
class Balance(BaseModel):
    def free(self) -> Decimal  # Line 431
```

#### Class: `ArbitrageOpportunity`

**Inherits**: BaseModel
**Purpose**: Arbitrage opportunity data structure

```python
class ArbitrageOpportunity(BaseModel):
    def validate_prices(cls, v: Decimal) -> Decimal  # Line 453
    def validate_quantity(cls, v: Decimal) -> Decimal  # Line 477
    def is_expired(self) -> bool  # Line 500
    def calculate_profit(self) -> Decimal  # Line 506
```

### File: validator_registry.py

**Key Imports:**
- `from src.core.dependency_injection import injectable`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `ValidatorInterface`

**Inherits**: ABC
**Purpose**: Base interface for all validators

```python
class ValidatorInterface(ABC):
    def validate(self, data: Any, **kwargs) -> bool  # Line 19
```

#### Class: `CompositeValidator`

**Inherits**: ValidatorInterface
**Purpose**: Composite validator that chains multiple validators

```python
class CompositeValidator(ValidatorInterface):
    def __init__(self, validators: list[ValidatorInterface])  # Line 39
    def validate(self, data: Any, **kwargs) -> bool  # Line 48
```

#### Class: `ValidatorRegistry`

**Purpose**: Central registry for all validators

```python
class ValidatorRegistry:
    def __init__(self) -> None  # Line 63
    def register_validator(self, name: str, validator: ValidatorInterface) -> None  # Line 70
    def register_validator_class(self, name: str, validator_class: type[ValidatorInterface]) -> None  # Line 81
    def register_rule(self, ...) -> None  # Line 94
    def get_validator(self, name: str) -> ValidatorInterface  # Line 117
    def validate(self, data_type: str, data: Any, validator_name: str | None = None, **kwargs) -> bool  # Line 142
    def create_composite_validator(self, validator_names: list[str]) -> CompositeValidator  # Line 178
    def clear(self) -> None  # Line 191
```

#### Class: `RangeValidator`

**Inherits**: ValidatorInterface
**Purpose**: Validator for numeric ranges

```python
class RangeValidator(ValidatorInterface):
    def __init__(self, min_value: Decimal | None = None, max_value: Decimal | None = None)  # Line 202
    def validate(self, data: Any, **kwargs) -> bool  # Line 213
```

#### Class: `LengthValidator`

**Inherits**: ValidatorInterface
**Purpose**: Validator for string/collection length

```python
class LengthValidator(ValidatorInterface):
    def __init__(self, min_length: int | None = None, max_length: int | None = None)  # Line 230
    def validate(self, data: Any, **kwargs) -> bool  # Line 241
```

#### Class: `PatternValidator`

**Inherits**: ValidatorInterface
**Purpose**: Validator for regex patterns

```python
class PatternValidator(ValidatorInterface):
    def __init__(self, pattern: str)  # Line 260
    def validate(self, data: Any, **kwargs) -> bool  # Line 271
```

#### Class: `TypeValidator`

**Inherits**: ValidatorInterface
**Purpose**: Validator for type checking

```python
class TypeValidator(ValidatorInterface):
    def __init__(self, expected_type: type)  # Line 285
    def validate(self, data: Any, **kwargs) -> bool  # Line 294
```

#### Functions:

```python
def register_validator(name: str, validator: ValidatorInterface) -> None  # Line 342
def register_validator_class(name: str, validator_class: type[ValidatorInterface]) -> None  # Line 347
def register_rule(data_type: str, rule: Callable, error_message: str | None = None) -> None  # Line 352
def validate(data_type: str, data: Any, **kwargs) -> bool  # Line 357
def get_validator(name: str) -> ValidatorInterface  # Line 362
```

### File: websocket_manager.py

**Key Imports:**
- `from src.core.exceptions import WebSocketError`
- `from src.core.resource_manager import ResourceManager`
- `from src.core.resource_manager import ResourceType`

#### Class: `WebSocketState`

**Inherits**: Enum
**Purpose**: WebSocket connection states

```python
class WebSocketState(Enum):
```

#### Class: `WebSocketManager`

**Purpose**: Async WebSocket connection manager with proper resource cleanup

```python
class WebSocketManager:
    def __init__(self, ...)  # Line 56
    async def connection(self) -> AsyncGenerator['WebSocketManager', None]  # Line 109
    async def _connect(self)  # Line 127
    async def _disconnect(self)  # Line 172
    async def _cleanup_connection(self)  # Line 227
    async def _heartbeat_loop(self)  # Line 246
    async def _message_handler_loop(self)  # Line 277
    async def _message_queue_processor(self)  # Line 342
    async def send_message(self, message: dict) -> None  # Line 378
    def set_message_callback(self, callback: Callable[[dict], None]) -> None  # Line 426
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None  # Line 430
    def set_disconnect_callback(self, callback: Callable[[], None]) -> None  # Line 434
    def is_connected(self) -> bool  # Line 439
    def get_stats(self) -> dict[str, Any]  # Line 445
```

#### Functions:

```python
def create_websocket_manager(url: str, resource_manager: ResourceManager | None = None, **kwargs) -> WebSocketManager  # Line 465
```

### File: websocket_types.py

#### Class: `StreamType`

**Inherits**: Enum
**Purpose**: WebSocket stream types for different data subscriptions

```python
class StreamType(Enum):
```

#### Class: `MessagePriority`

**Inherits**: Enum
**Purpose**: Message priority levels for WebSocket message handling

```python
class MessagePriority(Enum):
    def __lt__(self, other)  # Line 29
    def __le__(self, other)  # Line 35
    def __gt__(self, other)  # Line 41
    def __ge__(self, other)  # Line 47
```

---
**Generated**: Complete reference for core module
**Total Classes**: 396
**Total Functions**: 102