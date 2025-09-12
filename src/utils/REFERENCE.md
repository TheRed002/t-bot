# UTILS Module Reference

## INTEGRATION
**Dependencies**: core, error_handling, monitoring, state, web_interface
**Used By**: analytics
**Provides**: AsyncTaskManager, AuthenticatedWebSocketManager, BaseUtilityService, BaseWebSocketManager, BotStatusWebSocketManager, CacheManager, ExchangeWebSocketReconnectionManager, GPUManager, HTTPSessionManager, MarketDataWebSocketManager, MultiStreamWebSocketManager, PriceHistoryManager, ResourceManager, ValidationService, WebSocketConnectionManager, WebSocketHeartbeatManager, WebSocketStreamManager, WebSocketSubscriptionManager
**Patterns**: Async Operations, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Security**:
- Credential management
- Authentication
- Authentication
**Performance**:
- Parallel execution
- Parallel execution
- Retry mechanisms
**Architecture**:
- BaseUtilityService inherits from base architecture
- ValidationService inherits from base architecture

## MODULE OVERVIEW
**Files**: 69 Python files
**Classes**: 177
**Functions**: 358

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `FeeCalculator` ✅

**Purpose**: Common fee calculation utilities for arbitrage strategies
**Status**: Complete

**Implemented Methods:**
- `calculate_cross_exchange_fees(buy_price: Decimal, sell_price: Decimal) -> Decimal` - Line 35
- `calculate_triangular_fees(rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal` - Line 100

### Implementation: `OpportunityAnalyzer` ✅

**Purpose**: Common opportunity analysis utilities for arbitrage strategies
**Status**: Complete

**Implemented Methods:**
- `calculate_priority(profit_percentage: Decimal, arbitrage_type: str) -> Decimal` - Line 192
- `prioritize_opportunities(signals: list[Any], max_opportunities: int = 10) -> list[Any]` - Line 264

### Implementation: `SpreadAnalyzer` ✅

**Purpose**: Common spread analysis utilities for arbitrage strategies
**Status**: Complete

**Implemented Methods:**
- `calculate_spread_percentage(buy_price: Decimal, sell_price: Decimal) -> Decimal` - Line 313
- `calculate_net_profit(gross_spread: Decimal, fees: Decimal, base_price: Decimal) -> tuple[Decimal, Decimal]` - Line 338

### Implementation: `PositionSizingCalculator` ✅

**Purpose**: Position sizing utilities for arbitrage strategies
**Status**: Complete

**Implemented Methods:**
- `calculate_arbitrage_position_size(total_capital, ...) -> Decimal` - Line 376

### Implementation: `MarketDataValidator` ✅

**Purpose**: Market data validation utilities for arbitrage strategies
**Status**: Complete

**Implemented Methods:**
- `validate_price_data(price_data: dict[str, Any]) -> bool` - Line 476
- `check_arbitrage_thresholds(profit_percentage, ...) -> bool` - Line 519

### Implementation: `ServiceHealthChecker` ✅

**Purpose**: Helper for standardized service health checks
**Status**: Complete

**Implemented Methods:**
- `async check_service_health(service: Any, service_name: str, default_healthy: bool = True) -> dict[str, Any]` - Line 489

### Implementation: `CacheLevel` ✅

**Inherits**: Enum
**Purpose**: Cache level enumeration
**Status**: Complete

### Implementation: `CacheStrategy` ✅

**Inherits**: Enum
**Purpose**: Cache strategy enumeration
**Status**: Complete

### Implementation: `CacheMode` ✅

**Inherits**: Enum
**Purpose**: Cache operation mode
**Status**: Complete

### Implementation: `CacheEntry` ✅

**Purpose**: Cache entry with metadata
**Status**: Complete

**Implemented Methods:**
- `is_expired(self) -> bool` - Line 68
- `update_access(self) -> None` - Line 75

### Implementation: `CacheStats` ✅

**Purpose**: Cache statistics
**Status**: Complete

**Implemented Methods:**
- `calculate_hit_rate(self) -> float` - Line 95
- `calculate_memory_usage_mb(self) -> float` - Line 103

### Implementation: `CacheSerializationUtils` ✅

**Purpose**: Shared serialization utilities for cache implementations
**Status**: Complete

**Implemented Methods:**
- `serialize_json(value: Any) -> str` - Line 113
- `deserialize_json(data: str) -> Any` - Line 133
- `serialize_pickle(value: Any) -> bytes` - Line 153
- `deserialize_pickle(data: bytes) -> Any` - Line 173
- `calculate_size_bytes(value: Any) -> int` - Line 193

### Implementation: `CacheKeyUtils` ✅

**Purpose**: Shared cache key utilities
**Status**: Complete

**Implemented Methods:**
- `generate_key(prefix: str, *args: Any) -> str` - Line 217
- `validate_key(key: str) -> bool` - Line 239

### Implementation: `CacheLRUUtils` ✅

**Purpose**: Shared LRU cache utilities
**Status**: Complete

**Implemented Methods:**
- `update_lru_order(cache: OrderedDict, key: str) -> None` - Line 268
- `evict_lru(cache: OrderedDict, stats: CacheStats) -> str | None` - Line 280

### Implementation: `CacheValidationUtils` ✅

**Purpose**: Shared cache validation utilities
**Status**: Complete

**Implemented Methods:**
- `validate_ttl(ttl: int | None) -> bool` - Line 310
- `validate_cache_size(max_size: int) -> bool` - Line 326
- `validate_cache_config(config: dict[str, Any]) -> list[str]` - Line 339

### Implementation: `FinancialCalculator` ✅

**Inherits**: CalculatorInterface
**Purpose**: Class for all financial calculations
**Status**: Complete

**Implemented Methods:**
- `sharpe_ratio(returns, ...) -> Decimal` - Line 32
- `sortino_ratio(returns, ...) -> Decimal` - Line 81
- `calmar_ratio(returns: tuple[Decimal, Ellipsis], periods_per_year: int = 252) -> Decimal` - Line 94
- `moving_average(prices: tuple[Decimal, Ellipsis], period: int, ma_type: str = 'simple') -> Decimal` - Line 162
- `max_drawdown(prices: list[Decimal] | NDArray[np.float64]) -> tuple[Decimal, int, int]` - Line 222
- `kelly_criterion(win_probability, ...) -> Decimal` - Line 270
- `position_size_volatility_adjusted(account_balance, ...) -> Decimal` - Line 317
- `calculate_returns(prices: list[Decimal] | NDArray[np.float64], method: str = 'simple') -> list[Decimal]` - Line 362
- `risk_reward_ratio(entry_price: Decimal, stop_loss: Decimal, take_profit: Decimal) -> Decimal` - Line 410
- `expected_value(win_probability: Decimal, avg_win: Decimal, avg_loss: Decimal) -> Decimal` - Line 438
- `profit_factor(wins: list[Decimal], losses: list[Decimal]) -> Decimal` - Line 457
- `calculate_compound_return(principal: Decimal, rate: Decimal, periods: int) -> Decimal` - Line 481
- `calculate_sharpe_ratio(returns: list[Decimal], risk_free_rate: Decimal) -> Decimal` - Line 493

### Implementation: `CapitalErrorContext` ✅

**Purpose**: Context manager for capital management operations with error handling
**Status**: Complete

**Implemented Methods:**

### Implementation: `ResourceManager` ✅

**Purpose**: Unified resource manager for capital management services
**Status**: Complete

**Implemented Methods:**
- `limit_list_size(self, ...) -> list[Any]` - Line 39
- `clean_time_based_data(self, ...) -> dict[str, list[tuple[datetime, Any]]]` - Line 74
- `clean_fund_flows(self, fund_flows: list[Any], max_age_days: int = 30) -> list[Any]` - Line 121
- `clean_performance_data(self, performance_data: dict[str, dict[str, Any]], max_age_days: int = 30) -> dict[str, dict[str, Any]]` - Line 153
- `cleanup_allocations_data(self, allocations_data: list[dict[str, Any]]) -> None` - Line 198
- `should_trigger_cleanup(self, current_size: int, data_type: str = 'default') -> bool` - Line 220
- `get_memory_usage_info(self) -> dict[str, Any]` - Line 238

### Implementation: `DependencyInjectionMixin` ✅

**Purpose**: Mixin providing common dependency injection patterns
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, container: Any) -> None` - Line 40
- `get_dependencies(self) -> list[str]` - Line 62

### Implementation: `ConnectionManagerMixin` ✅

**Purpose**: Mixin providing common connection management patterns
**Status**: Complete

**Implemented Methods:**

### Implementation: `LifecycleManagerMixin` ✅

**Purpose**: Mixin providing common lifecycle management patterns
**Status**: Complete

**Implemented Methods:**
- `uptime(self) -> float | None` - Line 336

### Implementation: `HealthCheckMixin` ✅

**Purpose**: Mixin providing common health check patterns
**Status**: Complete

**Implemented Methods:**
- `async basic_health_check(self) -> HealthStatus` - Line 348

### Implementation: `ResourceCleanupMixin` ✅

**Purpose**: Mixin providing common resource cleanup patterns
**Status**: Complete

**Implemented Methods:**
- `register_cleanup_callback(self, callback: Callable) -> None` - Line 385

### Implementation: `LoggingHelperMixin` ✅

**Purpose**: Mixin providing structured logging helpers
**Status**: Complete

**Implemented Methods:**

### Implementation: `BaseUtilityMixin` ✅

**Inherits**: DependencyInjectionMixin, ConnectionManagerMixin, LifecycleManagerMixin, HealthCheckMixin, ResourceCleanupMixin, LoggingHelperMixin
**Purpose**: Combined utility mixin with all common patterns
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 537
- `async shutdown(self) -> None` - Line 541

### Implementation: `DataFlowConsistencyValidator` ✅

**Purpose**: Validates data flow consistency between risk_management and utils modules
**Status**: Complete

**Implemented Methods:**
- `validate_all(self) -> dict[str, Any]` - Line 28

### Implementation: `DataFlowValidator` ✅

**Purpose**: Validates data flow consistency across module boundaries
**Status**: Complete

**Implemented Methods:**
- `validate_message_pattern_consistency(data: dict[str, Any]) -> None` - Line 38
- `validate_boundary_crossing_metadata(data: dict[str, Any]) -> None` - Line 89
- `validate_complete_data_flow(cls, ...) -> None` - Line 123

### Implementation: `DataFlowIntegrityError` ✅

**Inherits**: Exception
**Purpose**: Raised when data flow integrity is compromised
**Status**: Complete

### Implementation: `PrecisionTracker` ✅

**Purpose**: Tracks precision loss across data transformations
**Status**: Complete

**Implemented Methods:**
- `track_conversion(self, ...) -> None` - Line 190
- `get_summary(self) -> dict[str, Any]` - Line 236
- `track_operation(self, operation: str, input_precision: int, output_precision: int) -> None` - Line 254
- `get_precision_stats(self) -> dict[str, Any]` - Line 272

### Implementation: `DataFlowValidator` ✅

**Purpose**: Comprehensive data flow validation system
**Status**: Complete

**Implemented Methods:**
- `validate_data_flow(self, data: dict[str, Any], context: str = 'unknown') -> dict[str, Any]` - Line 348
- `add_validation_rule(self, field_pattern: str, rule: dict[str, Any]) -> None` - Line 618
- `validate_data_integrity(self, data: Any) -> bool` - Line 623
- `get_validation_report(self) -> dict[str, Any]` - Line 638

### Implementation: `IntegrityPreservingConverter` ✅

**Purpose**: Converter that preserves data integrity across module boundaries
**Status**: Complete

**Implemented Methods:**
- `safe_convert_for_metrics(self, ...) -> float` - Line 667
- `batch_convert_with_integrity(self, ...) -> dict[str, float]` - Line 717

### Implementation: `DecimalEncoder` ✅

**Purpose**: JSON encoder that handles Decimal values
**Status**: Complete

**Implemented Methods:**
- `encode(obj: Any) -> Any` - Line 530

### Implementation: `FloatDeprecationWarning` ✅

**Purpose**: Context manager to detect and warn about float usage in financial code
**Status**: Complete

**Implemented Methods:**
- `warn_float_usage(context: str) -> None` - Line 568

### Implementation: `ExceptionCategory` ✅

**Purpose**: Classification of exceptions for intelligent retry behavior
**Status**: Complete

**Implemented Methods:**
- `should_retry(cls, exception: Exception) -> bool` - Line 90
- `get_retry_delay(cls, exception: Exception, attempt: int, base_delay: float) -> float` - Line 130

### Implementation: `UnifiedDecorator` ✅

**Purpose**: Single configurable decorator replacing multiple decorators
**Status**: Complete

**Implemented Methods:**
- `clear_cache(cls) -> None` - Line 326
- `enhance(cls, ...) -> Callable[[F], F]` - Line 599

### Implementation: `RecoveryStrategy` ✅

**Inherits**: Enum
**Purpose**: Recovery strategy options
**Status**: Complete

### Implementation: `ErrorContext` ✅

**Purpose**: Context information for error handling
**Status**: Complete

### Implementation: `RecoveryCheckpoint` ✅

**Purpose**: Recovery checkpoint for rollback operations
**Status**: Complete

### Implementation: `BaseErrorRecovery` ✅

**Purpose**: Base error recovery system providing common recovery patterns
**Status**: Complete

**Implemented Methods:**
- `classify_error(self, exception: Exception) -> ErrorType` - Line 132
- `async create_recovery_checkpoint(self, operation: str, state_data: dict[str, Any] | None = None, **context) -> str` - Line 138
- `async handle_error(self, ...) -> ErrorContext` - Line 164
- `async rollback_to_checkpoint(self, checkpoint_id: str) -> bool` - Line 229
- `get_error_statistics(self) -> dict[str, Any]` - Line 398

### Implementation: `SymbolConversionUtils` ✅

**Purpose**: Utilities for converting symbols between different exchange formats
**Status**: Complete

**Implemented Methods:**
- `normalize_symbol(symbol: str, target_exchange: str) -> str` - Line 65
- `to_binance_format(symbol: str) -> str` - Line 86
- `to_coinbase_format(symbol: str) -> str` - Line 96
- `to_okx_format(symbol: str) -> str` - Line 107
- `to_standard_format(symbol: str) -> str` - Line 118
- `get_base_quote(symbol: str) -> tuple[str, str]` - Line 143

### Implementation: `OrderConversionUtils` ✅

**Purpose**: Utilities for converting order parameters between exchanges
**Status**: Complete

**Implemented Methods:**
- `convert_order_to_exchange_format(order, ...) -> dict[str, Any]` - Line 173

### Implementation: `ResponseConversionUtils` ✅

**Purpose**: Utilities for converting exchange responses to unified formats
**Status**: Complete

**Implemented Methods:**
- `create_unified_order_response(exchange_response: dict[str, Any], exchange: str, original_symbol: str = None) -> OrderResponse` - Line 349

### Implementation: `MarketDataConversionUtils` ✅

**Purpose**: Utilities for converting market data between formats
**Status**: Complete

**Implemented Methods:**
- `create_unified_ticker(exchange_data: dict[str, Any], exchange: str, symbol: str) -> Ticker` - Line 518
- `create_unified_order_book(exchange_data: dict[str, Any], symbol: str, exchange: str) -> OrderBook` - Line 558

### Implementation: `ExchangeConversionUtils` ✅

**Purpose**: Utilities for converting between unified and exchange-specific formats
**Status**: Complete

**Implemented Methods:**
- `create_order_response(exchange_data, ...) -> OrderResponse` - Line 606
- `get_binance_field_mapping() -> dict[str, str]` - Line 698
- `get_binance_status_mapping() -> dict[str, OrderStatus]` - Line 717
- `get_binance_type_mapping() -> dict[str, OrderType]` - Line 729
- `get_coinbase_field_mapping() -> dict[str, str]` - Line 741
- `get_coinbase_status_mapping() -> dict[str, OrderStatus]` - Line 757
- `get_coinbase_type_mapping() -> dict[str, OrderType]` - Line 773
- `get_okx_field_mapping() -> dict[str, str]` - Line 783
- `get_okx_status_mapping() -> dict[str, OrderStatus]` - Line 799
- `get_okx_type_mapping() -> dict[str, OrderType]` - Line 811
- `convert_binance_order_to_response(result: dict[str, Any]) -> OrderResponse` - Line 823
- `convert_coinbase_order_to_response(result: dict[str, Any]) -> OrderResponse` - Line 838
- `convert_okx_order_to_response(result: dict[str, Any]) -> OrderResponse` - Line 849

### Implementation: `ExchangeErrorHandler` ✅

**Purpose**: Common error handling utilities for exchanges
**Status**: Complete

**Implemented Methods:**
- `async handle_exchange_error(self, ...) -> None` - Line 56
- `async handle_api_error(self, error: Exception, operation: str, context: dict[str, Any] | None = None) -> None` - Line 105

### Implementation: `ErrorMappingUtils` ✅

**Purpose**: Utilities for mapping exchange-specific errors to unified exceptions
**Status**: Complete

**Implemented Methods:**
- `map_exchange_error(error_data, ...) -> Exception` - Line 170

### Implementation: `RetryableOperationHandler` ✅

**Purpose**: Handler for retryable exchange operations
**Status**: Complete

**Implemented Methods:**
- `async execute_with_retry(self, operation_func, operation_name: str, *args, **kwargs) -> Any` - Line 284
- `async execute_with_aggressive_retry(self, operation_func, operation_name: str, *args, **kwargs) -> Any` - Line 313

### Implementation: `OperationTimeoutHandler` ✅

**Purpose**: Handler for operation timeouts
**Status**: Complete

**Implemented Methods:**
- `async execute_with_timeout(operation_func, ...) -> Any` - Line 354

### Implementation: `ExchangeCircuitBreaker` ✅

**Purpose**: Simple circuit breaker for exchange operations
**Status**: Complete

**Implemented Methods:**
- `async call(self, func, *args, **kwargs)` - Line 406
- `get_state(self) -> dict[str, Any]` - Line 474

### Implementation: `OrderManagementUtils` ✅

**Purpose**: Shared utilities for order management across exchanges
**Status**: Complete

**Implemented Methods:**
- `validate_order_structure(self, order: OrderRequest) -> None` - Line 31
- `track_order(order_response, ...) -> None` - Line 53
- `update_order_status(order_id: str, status: OrderStatus, pending_orders: dict[str, dict[str, Any]]) -> None` - Line 83

### Implementation: `OrderConversionUtils` ✅

**Purpose**: Utilities for converting orders between exchange formats
**Status**: Complete

**Implemented Methods:**
- `create_base_order_response(order_id, ...) -> OrderResponse` - Line 102
- `standardize_symbol_format(symbol: str, target_format: str = 'dash') -> str` - Line 147

### Implementation: `OrderStatusUtils` ✅

**Purpose**: Utilities for handling order status conversions
**Status**: Complete

**Implemented Methods:**
- `convert_status(status: str, exchange: str) -> OrderStatus` - Line 199
- `is_terminal_status(status: OrderStatus) -> bool` - Line 251

### Implementation: `OrderTypeUtils` ✅

**Purpose**: Utilities for order type conversions
**Status**: Complete

**Implemented Methods:**
- `convert_to_exchange_format(order_type: OrderType, exchange: str) -> str` - Line 274
- `convert_from_exchange_format(exchange_type: str, exchange: str) -> OrderType` - Line 318

### Implementation: `AssetPrecisionUtils` ✅

**Purpose**: Utilities for asset precision calculations
**Status**: Complete

**Implemented Methods:**
- `get_asset_precision(symbol: str, precision_type: str = 'quantity') -> int` - Line 348

### Implementation: `FeeCalculationUtils` ✅

**Purpose**: Utilities for fee calculations
**Status**: Complete

**Implemented Methods:**
- `calculate_fee(order_value: Decimal, exchange: str, symbol: str, is_maker: bool = False) -> Decimal` - Line 401
- `get_fee_rates(exchange: str) -> dict[str, Decimal]` - Line 437

### Implementation: `ExchangeValidationUtils` ✅

**Purpose**: Common validation utilities for exchanges
**Status**: Complete

**Implemented Methods:**
- `validate_exchange_specific_order(order: OrderRequest, exchange: str) -> None` - Line 46

### Implementation: `SymbolValidationUtils` ✅

**Purpose**: Utilities for validating trading symbols
**Status**: Complete

**Implemented Methods:**
- `is_valid_symbol_format(symbol: str, exchange: str) -> bool` - Line 141
- `is_valid_binance_symbol(symbol: str) -> bool` - Line 162
- `is_valid_coinbase_symbol(symbol: str) -> bool` - Line 188
- `is_valid_okx_symbol(symbol: str) -> bool` - Line 214
- `get_supported_symbols(exchange: str) -> set[str]` - Line 240

### Implementation: `PrecisionValidationUtils` ✅

**Purpose**: Utilities for validating precision requirements
**Status**: Complete

**Implemented Methods:**
- `validate_precision(value: Decimal, precision: int, value_type: str = 'value') -> None` - Line 272
- `validate_order_precision(order, ...) -> None` - Line 300
- `round_to_exchange_precision(value: Decimal, precision: int, rounding_mode = ROUND_HALF_UP) -> Decimal` - Line 336

### Implementation: `RiskValidationUtils` ✅

**Purpose**: Utilities for risk-based validation
**Status**: Complete

**Implemented Methods:**
- `validate_order_size_limits(order, ...) -> None` - Line 361
- `validate_price_bounds(order, ...) -> None` - Line 398
- `validate_stop_price_logic(order: OrderRequest) -> None` - Line 426

### Implementation: `GPUManager` ✅

**Purpose**: Manages GPU resources and provides utilities for GPU acceleration
**Status**: Complete

**Implemented Methods:**
- `is_available(self) -> bool` - Line 188
- `get_memory_info(self, device_id: int = 0) -> dict[str, float]` - Line 192
- `clear_cache(self) -> None` - Line 217
- `to_gpu(self, data: Any, dtype: str | None = None) -> Any` - Line 234
- `to_cpu(self, data: Any) -> Any` - Line 296
- `accelerate_computation(self, func: Any, *args: Any, **kwargs: Any) -> Any` - Line 320

### Implementation: `ValidationServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Interface for validation services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async validate_order(self, order_data: dict[str, Any], context: 'ValidationContext | None' = None) -> 'ValidationResult'` - Line 18
- `async validate_risk_parameters(self, risk_data: dict[str, Any], context: 'ValidationContext | None' = None) -> 'ValidationResult'` - Line 25
- `async validate_strategy_config(self, strategy_data: dict[str, Any], context: 'ValidationContext | None' = None) -> 'ValidationResult'` - Line 32
- `async validate_market_data(self, market_data: dict[str, Any], context: 'ValidationContext | None' = None) -> 'ValidationResult'` - Line 39
- `async validate_batch(self, ...) -> dict[str, 'ValidationResult']` - Line 46

### Implementation: `GPUInterface` 🔧

**Inherits**: Protocol
**Purpose**: Interface for GPU management services
**Status**: Abstract Base Class

**Implemented Methods:**
- `is_available(self) -> bool` - Line 58
- `get_memory_info(self) -> dict[str, Any]` - Line 63

### Implementation: `PrecisionInterface` 🔧

**Inherits**: Protocol
**Purpose**: Interface for precision tracking services
**Status**: Abstract Base Class

**Implemented Methods:**
- `track_operation(self, operation: str, input_precision: int, output_precision: int) -> None` - Line 73
- `get_precision_stats(self) -> dict[str, Any]` - Line 78

### Implementation: `DataFlowInterface` 🔧

**Inherits**: Protocol
**Purpose**: Interface for data flow validation services
**Status**: Abstract Base Class

**Implemented Methods:**
- `validate_data_integrity(self, data: Any) -> bool` - Line 88
- `get_validation_report(self) -> dict[str, Any]` - Line 93

### Implementation: `CalculatorInterface` 🔧

**Inherits**: Protocol
**Purpose**: Interface for financial calculation services
**Status**: Abstract Base Class

**Implemented Methods:**
- `calculate_compound_return(self, principal: Decimal, rate: Decimal, periods: int) -> Decimal` - Line 103
- `calculate_sharpe_ratio(self, returns: list[Decimal], risk_free_rate: Decimal) -> Decimal` - Line 108

### Implementation: `BaseUtilityService` 🔧

**Inherits**: BaseService
**Purpose**: Base class for utility services that inherits from core BaseService
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 122
- `async shutdown(self) -> None` - Line 127

### Implementation: `MessagePattern` ✅

**Inherits**: Enum
**Purpose**: Standardized message patterns
**Status**: Complete

### Implementation: `MessageType` ✅

**Inherits**: Enum
**Purpose**: Standard message types
**Status**: Complete

### Implementation: `StandardMessage` ✅

**Purpose**: Standardized message format across all communication patterns
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict[str, Any]` - Line 65
- `from_dict(cls, data: dict[str, Any]) -> 'StandardMessage'` - Line 79

### Implementation: `MessageHandler` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for consistent message handling
**Status**: Abstract Base Class

**Implemented Methods:**
- `async handle(self, message: StandardMessage) -> StandardMessage | None` - Line 96

### Implementation: `ErrorPropagationMixin` ✅

**Purpose**: Mixin for consistent error propagation patterns across all modules with enhanced boundary validation
**Status**: Complete

**Implemented Methods:**
- `propagate_validation_error(self, error: Exception, context: str) -> None` - Line 104
- `propagate_database_error(self, error: Exception, context: str) -> None` - Line 133
- `propagate_service_error(self, error: Exception, context: str) -> None` - Line 156
- `propagate_monitoring_error(self, error: Exception, context: str) -> None` - Line 193

### Implementation: `BoundaryValidator` ✅

**Purpose**: Validator for module boundary consistency
**Status**: Complete

**Implemented Methods:**
- `validate_database_entity(entity_dict: dict[str, Any], operation: str) -> None` - Line 233
- `validate_database_to_error_boundary(data: dict[str, Any]) -> None` - Line 257
- `validate_monitoring_to_error_boundary(data: dict[str, Any]) -> None` - Line 299
- `validate_error_to_monitoring_boundary(data: dict[str, Any]) -> None` - Line 389
- `validate_web_interface_to_error_boundary(data: dict[str, Any]) -> None` - Line 480
- `validate_risk_to_state_boundary(data: dict[str, Any]) -> None` - Line 550

### Implementation: `ProcessingParadigmAligner` ✅

**Purpose**: Aligns processing paradigms between batch and stream processing
**Status**: Complete

**Implemented Methods:**
- `create_batch_from_stream(stream_items: list[dict[str, Any]]) -> dict[str, Any]` - Line 618
- `create_stream_from_batch(batch_data: dict[str, Any]) -> list[dict[str, Any]]` - Line 629
- `align_processing_modes(source_mode: str, target_mode: str, data: dict[str, Any]) -> dict[str, Any]` - Line 645

### Implementation: `MessagingCoordinator` ✅

**Purpose**: Coordinates messaging patterns to prevent conflicts between pub/sub and req/reply
**Status**: Complete

**Implemented Methods:**
- `register_handler(self, pattern: MessagePattern, handler: MessageHandler) -> None` - Line 727
- `async publish(self, ...) -> None` - Line 736
- `async request(self, ...) -> Any` - Line 769
- `async reply(self, original_message: StandardMessage, response_data: Any) -> None` - Line 814
- `async stream_start(self, ...) -> None` - Line 833
- `async batch_process(self, ...) -> None` - Line 860

### Implementation: `DataTransformationHandler` ✅

**Inherits**: MessageHandler
**Purpose**: Handler for consistent data transformation patterns
**Status**: Complete

**Implemented Methods:**
- `async handle(self, message: StandardMessage) -> StandardMessage | None` - Line 1012

### Implementation: `TTLCache` ✅

**Inherits**: Generic[T]
**Purpose**: Time-To-Live cache implementation
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> T | None` - Line 36
- `async set(self, key: str, value: T) -> None` - Line 56
- `async delete(self, key: str) -> bool` - Line 73
- `async clear(self) -> int` - Line 89
- `async size(self) -> int` - Line 101
- `async cleanup_expired(self) -> int` - Line 105

### Implementation: `ModelCache` ✅

**Inherits**: TTLCache[Any]
**Purpose**: Specialized cache for ML models
**Status**: Complete

**Implemented Methods:**
- `async get_model(self, model_id: str) -> Any | None` - Line 137
- `async cache_model(self, model_id: str, model: Any) -> None` - Line 141
- `async remove_model(self, model_id: str) -> bool` - Line 145

### Implementation: `PredictionCache` ✅

**Inherits**: TTLCache[dict[str, Any]]
**Purpose**: Specialized cache for ML predictions
**Status**: Complete

**Implemented Methods:**
- `async get_prediction(self, cache_key: str) -> dict[str, Any] | None` - Line 163
- `async cache_prediction(self, cache_key: str, prediction: dict[str, Any]) -> None` - Line 167

### Implementation: `FeatureCache` ✅

**Inherits**: TTLCache[dict[str, Any]]
**Purpose**: Specialized cache for feature sets
**Status**: Complete

**Implemented Methods:**
- `async get_features(self, cache_key: str) -> dict[str, Any] | None` - Line 185
- `async cache_features(self, cache_key: str, features: dict[str, Any]) -> None` - Line 189

### Implementation: `CacheManager` ✅

**Purpose**: Centralized cache manager for ML operations
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 305
- `async stop(self) -> None` - Line 311
- `async clear_all(self) -> dict[str, int]` - Line 321
- `async get_cache_stats(self) -> dict[str, Any]` - Line 337

### Implementation: `BaseMLConfig` ✅

**Inherits**: BaseModel
**Purpose**: Base configuration for ML components
**Status**: Complete

### Implementation: `MLModelConfig` ✅

**Inherits**: BaseMLConfig
**Purpose**: Configuration for ML models
**Status**: Complete

### Implementation: `MLServiceConfig` ✅

**Inherits**: BaseMLConfig
**Purpose**: Configuration for ML service
**Status**: Complete

### Implementation: `ModelManagerConfig` ✅

**Inherits**: BaseMLConfig
**Purpose**: Configuration for model manager service
**Status**: Complete

### Implementation: `PredictorConfig` ✅

**Inherits**: MLModelConfig
**Purpose**: Configuration for predictor models
**Status**: Complete

### Implementation: `ClassifierConfig` ✅

**Inherits**: MLModelConfig
**Purpose**: Configuration for classifier models
**Status**: Complete

### Implementation: `MLCacheConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for ML caching
**Status**: Complete

### Implementation: `FinancialPrecisionWarning` ✅

**Inherits**: UserWarning
**Purpose**: Warning raised when precision loss is detected in financial calculations
**Status**: Complete

### Implementation: `HTTPSessionManager` ✅

**Purpose**: Shared HTTP session manager for monitoring components
**Status**: Complete

**Implemented Methods:**
- `async get_session(self, key: str = 'default', **session_kwargs) -> aiohttp.ClientSession` - Line 54
- `async close_all(self) -> None` - Line 78

### Implementation: `AsyncTaskManager` ✅

**Purpose**: Manages async task lifecycle with proper cleanup
**Status**: Complete

**Implemented Methods:**
- `create_task(self, coro, name: str | None = None) -> asyncio.Task` - Line 337
- `async shutdown(self, timeout: float = 5.0) -> None` - Line 354
- `shutdown_requested(self) -> bool` - Line 388

### Implementation: `MetricValueProcessor` ✅

**Purpose**: Processes and validates metric values consistently
**Status**: Complete

**Implemented Methods:**
- `process_financial_value(value, ...) -> float` - Line 493
- `process_latency_value(value: float, metric_name: str) -> float` - Line 510

### Implementation: `SystemMetricsCollector` ✅

**Purpose**: Shared system metrics collection to eliminate duplication
**Status**: Complete

**Implemented Methods:**
- `async collect_system_metrics() -> dict[str, Any]` - Line 526

### Implementation: `PipelineStage` ✅

**Inherits**: Enum
**Purpose**: Common pipeline stage enumeration
**Status**: Complete

### Implementation: `ProcessingMode` ✅

**Inherits**: Enum
**Purpose**: Data processing mode enumeration
**Status**: Complete

### Implementation: `DataQuality` ✅

**Inherits**: Enum
**Purpose**: Data quality levels
**Status**: Complete

### Implementation: `PipelineAction` ✅

**Inherits**: Enum
**Purpose**: Pipeline action enumeration
**Status**: Complete

### Implementation: `PipelineMetrics` ✅

**Purpose**: Common pipeline processing metrics
**Status**: Complete

**Implemented Methods:**
- `calculate_success_rate(self) -> float` - Line 84
- `calculate_failure_rate(self) -> float` - Line 90
- `update_processing_time(self, processing_time_ms: float) -> None` - Line 96
- `calculate_throughput(self, time_window_seconds: float = 1.0) -> float` - Line 107

### Implementation: `PipelineRecord` ✅

**Purpose**: Common pipeline record structure
**Status**: Complete

**Implemented Methods:**
- `add_error(self, error: str, stage: PipelineStage | None = None) -> None` - Line 134
- `add_warning(self, warning: str, stage: PipelineStage | None = None) -> None` - Line 140
- `has_errors(self) -> bool` - Line 146
- `has_warnings(self) -> bool` - Line 150

### Implementation: `PipelineStageProcessor` ✅

**Inherits**: Protocol
**Purpose**: Protocol for pipeline stage processors
**Status**: Complete

**Implemented Methods:**
- `async process(self, record: PipelineRecord) -> PipelineRecord` - Line 158
- `get_stage_name(self) -> str` - Line 162

### Implementation: `PipelineUtils` ✅

**Purpose**: Shared pipeline utility functions
**Status**: Complete

**Implemented Methods:**
- `validate_pipeline_config(config: dict[str, Any]) -> list[str]` - Line 171
- `calculate_data_quality_score(record: PipelineRecord, error_weight: float = 0.5, warning_weight: float = 0.2) -> float` - Line 210
- `determine_pipeline_action(quality_score, ...) -> PipelineAction` - Line 236
- `create_processing_summary(metrics: PipelineMetrics, duration_seconds: float) -> dict[str, Any]` - Line 276
- `log_pipeline_summary(pipeline_name: str, summary: dict[str, Any], logger_instance: Any = None) -> None` - Line 307

### Implementation: `PositionSizingAlgorithm` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for position sizing algorithms
**Status**: Abstract Base Class

**Implemented Methods:**
- `calculate_size(self, ...) -> Decimal` - Line 26
- `validate_inputs(self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal) -> bool` - Line 32

### Implementation: `FixedPercentageAlgorithm` ✅

**Inherits**: PositionSizingAlgorithm
**Purpose**: Fixed percentage position sizing algorithm
**Status**: Complete

**Implemented Methods:**
- `calculate_size(self, ...) -> Decimal` - Line 54

### Implementation: `KellyCriterionAlgorithm` ✅

**Inherits**: PositionSizingAlgorithm
**Purpose**: Kelly Criterion position sizing algorithm
**Status**: Complete

**Implemented Methods:**
- `calculate_size(self, ...) -> Decimal` - Line 78

### Implementation: `VolatilityAdjustedAlgorithm` ✅

**Inherits**: PositionSizingAlgorithm
**Purpose**: Volatility-adjusted position sizing algorithm
**Status**: Complete

**Implemented Methods:**
- `calculate_size(self, ...) -> Decimal` - Line 131

### Implementation: `ConfidenceWeightedAlgorithm` ✅

**Inherits**: PositionSizingAlgorithm
**Purpose**: Confidence-weighted position sizing for ML signals
**Status**: Complete

**Implemented Methods:**
- `calculate_size(self, ...) -> Decimal` - Line 180

### Implementation: `ATRBasedAlgorithm` ✅

**Inherits**: PositionSizingAlgorithm
**Purpose**: Average True Range based position sizing
**Status**: Complete

**Implemented Methods:**
- `calculate_size(self, ...) -> Decimal` - Line 213

### Implementation: `RiskEventType` ✅

**Inherits**: Enum
**Purpose**: Standardized risk event types
**Status**: Complete

### Implementation: `RiskEventSeverity` ✅

**Inherits**: Enum
**Purpose**: Risk event severity levels
**Status**: Complete

### Implementation: `RiskEvent` ✅

**Purpose**: Standardized risk event structure
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict[str, Any]` - Line 74

### Implementation: `RiskObserver` 🔧

**Inherits**: ABC
**Purpose**: Abstract base for risk event observers
**Status**: Abstract Base Class

**Implemented Methods:**
- `async handle_event(self, event: RiskEvent) -> None` - Line 89
- `get_observer_id(self) -> str` - Line 99

### Implementation: `LoggingRiskObserver` ✅

**Inherits**: RiskObserver
**Purpose**: Observer that logs risk events
**Status**: Complete

**Implemented Methods:**
- `async handle_event(self, event: RiskEvent) -> None` - Line 117
- `get_observer_id(self) -> str` - Line 134

### Implementation: `AlertingRiskObserver` ✅

**Inherits**: RiskObserver
**Purpose**: Observer that creates and stores risk alerts
**Status**: Complete

**Implemented Methods:**
- `async handle_event(self, event: RiskEvent) -> None` - Line 159
- `get_observer_id(self) -> str` - Line 192
- `get_recent_alerts(self, limit: int = 50) -> list[RiskAlert]` - Line 196

### Implementation: `CircuitBreakerObserver` ✅

**Inherits**: RiskObserver
**Purpose**: Observer that implements circuit breaker functionality
**Status**: Complete

**Implemented Methods:**
- `async handle_event(self, event: RiskEvent) -> None` - Line 233
- `get_observer_id(self) -> str` - Line 286
- `get_status(self) -> dict[str, Any]` - Line 290

### Implementation: `UnifiedRiskMonitor` ✅

**Purpose**: Centralized risk monitor that unifies all monitoring patterns
**Status**: Complete

**Implemented Methods:**
- `add_observer(self, observer: RiskObserver) -> None` - Line 336
- `remove_observer(self, observer_id: str) -> None` - Line 347
- `async notify_observers(self, event: RiskEvent) -> None` - Line 358
- `async monitor_metrics(self, metrics: RiskMetrics) -> None` - Line 371
- `async monitor_portfolio(self, portfolio_data: dict[str, Any]) -> None` - Line 394
- `set_threshold(self, key: str, value: Decimal) -> None` - Line 542
- `get_thresholds(self) -> dict[str, Decimal]` - Line 547
- `get_observer_status(self) -> dict[str, Any]` - Line 551
- `async trigger_emergency_stop(self, reason: str) -> None` - Line 561

### Implementation: `UnifiedRiskValidator` ✅

**Purpose**: Centralized risk validator that eliminates validation code duplication
**Status**: Complete

**Implemented Methods:**
- `validate_signal(self, ...) -> tuple[bool, str]` - Line 49
- `validate_order(self, ...) -> tuple[bool, str]` - Line 99
- `validate_position(self, position: Position, portfolio_value: Decimal) -> tuple[bool, str]` - Line 156
- `validate_portfolio(self, portfolio_data: dict[str, Any]) -> tuple[bool, str]` - Line 200
- `validate_risk_metrics(self, var_1d: Decimal, current_drawdown: Decimal, portfolio_value: Decimal) -> tuple[bool, str]` - Line 242
- `update_limits(self, new_limits: RiskLimits) -> None` - Line 399

### Implementation: `SeverityLevel` ✅

**Inherits**: Enum
**Purpose**: Generic severity levels
**Status**: Complete

### Implementation: `LogCategory` ✅

**Inherits**: Enum
**Purpose**: Log categories
**Status**: Complete

### Implementation: `LogLevel` ✅

**Inherits**: Enum
**Purpose**: Log levels
**Status**: Complete

### Implementation: `InformationLevel` ✅

**Inherits**: Enum
**Purpose**: Information levels for security context
**Status**: Complete

### Implementation: `ThreatType` ✅

**Inherits**: Enum
**Purpose**: Security threat types
**Status**: Complete

### Implementation: `ReportType` ✅

**Inherits**: Enum
**Purpose**: Report types
**Status**: Complete

### Implementation: `ReportingChannel` ✅

**Inherits**: Enum
**Purpose**: Reporting channels
**Status**: Complete

### Implementation: `UserRole` ✅

**Inherits**: Enum
**Purpose**: User roles for access control
**Status**: Complete

### Implementation: `SecurityContext` ✅

**Purpose**: Common security context
**Status**: Complete

**Implemented Methods:**
- `has_admin_access(self) -> bool` - Line 117

### Implementation: `ErrorAlert` ✅

**Purpose**: Generic error alert
**Status**: Complete

### Implementation: `SecureLogEntry` ✅

**Purpose**: Secure log entry
**Status**: Complete

### Implementation: `LoggingConfig` ✅

**Purpose**: Logging configuration
**Status**: Complete

### Implementation: `ReportingConfig` ✅

**Purpose**: Reporting configuration
**Status**: Complete

### Implementation: `ReportingRule` ✅

**Purpose**: Reporting rule configuration
**Status**: Complete

### Implementation: `ReportingMetrics` ✅

**Purpose**: Reporting metrics
**Status**: Complete

### Implementation: `HashGenerator` ✅

**Purpose**: Utility class for generating consistent hashes across the system
**Status**: Complete

**Implemented Methods:**
- `generate_backtest_hash(request: Any) -> str` - Line 153
- `generate_data_hash(data: dict[str, Any]) -> str` - Line 174

### Implementation: `StateOperationLock` ✅

**Purpose**: Centralized lock manager for state operations to eliminate duplication
**Status**: Complete

**Implemented Methods:**
- `get_lock(self, key: str) -> asyncio.Lock` - Line 178
- `async with_lock(self, key: str, operation: callable, *args, **kwargs)` - Line 203

### Implementation: `StateCache` ✅

**Purpose**: Centralized cache manager for state operations to eliminate duplication
**Status**: Complete

**Implemented Methods:**
- `get(self, key: str) -> Any | None` - Line 237
- `set(self, key: str, value: Any) -> None` - Line 251
- `pop(self, key: str, default: Any = None) -> Any` - Line 263
- `clear(self) -> None` - Line 270
- `size(self) -> int` - Line 274
- `keys(self) -> list[str]` - Line 278
- `get_stats(self) -> dict[str, int | float]` - Line 308

### Implementation: `StrategyCommons` ✅

**Purpose**: Comprehensive utility class providing common strategy operations
**Status**: Complete

**Implemented Methods:**
- `update_market_data(self, data: MarketData) -> None` - Line 65
- `get_technical_analysis(self, indicator_type: str, period: int = 14, **kwargs) -> float | None` - Line 78
- `check_volume_confirmation(self, current_volume: float, lookback_period: int = 20, min_ratio: float = 1.5) -> bool` - Line 123
- `get_volume_profile(self, periods: int = 20) -> dict[str, float]` - Line 147
- `validate_signal_comprehensive(self, ...) -> bool` - Line 165
- `calculate_position_size_with_risk(self, ...) -> Decimal` - Line 204
- `check_stop_loss_conditions(self, ...) -> bool` - Line 244
- `calculate_take_profit_level(self, ...) -> Decimal` - Line 304
- `get_market_condition_summary(self) -> dict[str, Any]` - Line 352
- `analyze_trend_strength(self, lookback_period: int = 20) -> dict[str, float]` - Line 411
- `cleanup(self) -> None` - Line 470

### Implementation: `PriceHistoryManager` ✅

**Purpose**: Manages price history data for strategy calculations
**Status**: Complete

**Implemented Methods:**
- `update_history(self, data: MarketData) -> None` - Line 47
- `get_recent_prices(self, periods: int) -> list[Decimal]` - Line 77
- `get_recent_volumes(self, periods: int) -> list[Decimal]` - Line 96
- `has_sufficient_data(self, required_periods: int) -> bool` - Line 115
- `clear_history(self) -> None` - Line 127

### Implementation: `TechnicalIndicators` ✅

**Purpose**: Common technical indicator calculations used across strategies
**Status**: Complete

**Implemented Methods:**
- `calculate_sma(prices: list[Decimal], period: int) -> Decimal | None` - Line 144
- `calculate_rsi(prices: list[Decimal], period: int = 14) -> Decimal | None` - Line 168
- `calculate_zscore(prices: list[Decimal], period: int) -> Decimal | None` - Line 220
- `calculate_atr(highs, ...) -> Decimal | None` - Line 257
- `calculate_volatility(prices: list[Decimal], periods: int = 20) -> Decimal` - Line 299

### Implementation: `VolumeAnalysis` ✅

**Purpose**: Common volume analysis functions used across strategies
**Status**: Complete

**Implemented Methods:**
- `check_volume_confirmation(current_volume, ...) -> bool` - Line 349
- `calculate_volume_profile(volume_history: list[Decimal], periods: int = 20) -> dict[str, Decimal]` - Line 390

### Implementation: `StrategySignalValidator` ✅

**Purpose**: Common signal validation patterns used across strategies
**Status**: Complete

**Implemented Methods:**
- `validate_signal_metadata(signal_metadata: dict[str, Any], required_fields: list[str]) -> bool` - Line 450
- `validate_price_range(price: Decimal, min_price: Decimal = ZERO, max_price: Decimal = Any) -> bool` - Line 475
- `check_signal_freshness(signal_timestamp: datetime, max_age_seconds: int = 300) -> bool` - Line 498

### Implementation: `BaseRecordValidator` 🔧

**Inherits**: ABC
**Purpose**: Base class for validators that process both single records and batches
**Status**: Abstract Base Class

**Implemented Methods:**
- `validate(self, data: Any) -> bool` - Line 14
- `get_errors(self) -> list[str]` - Line 54
- `reset(self) -> None` - Line 58
- `async health_check(self) -> dict[str, Any]` - Line 62

### Implementation: `ValidationFramework` ✅

**Purpose**: Centralized validation framework to eliminate duplication
**Status**: Complete

**Implemented Methods:**
- `validate_order(order: dict[str, Any]) -> bool` - Line 16
- `validate_strategy_params(params: dict[str, Any]) -> bool` - Line 114
- `validate_price(price: Any, max_price: Decimal = Any) -> Decimal` - Line 146
- `validate_quantity(quantity: Any, min_qty: Decimal = Any) -> Decimal` - Line 191
- `validate_symbol(symbol: str) -> str` - Line 235
- `validate_exchange_credentials(credentials: dict[str, Any]) -> bool` - Line 261
- `validate_risk_params(params: dict[str, Any]) -> bool` - Line 289
- `validate_risk_parameters(params: dict[str, Any]) -> bool` - Line 325
- `validate_timeframe(timeframe: str) -> str` - Line 370
- `validate_batch(validations: list[tuple[str, Callable[[Any], Any], Any]]) -> dict[str, Any]` - Line 421

### Implementation: `MarketDataValidationUtils` ✅

**Purpose**: Centralized market data validation utilities
**Status**: Complete

**Implemented Methods:**
- `validate_symbol_format(symbol: str) -> bool` - Line 25
- `validate_price_value(price, ...) -> Decimal` - Line 57
- `validate_volume_value(volume, ...) -> Decimal` - Line 96
- `validate_timestamp_value(timestamp, ...) -> datetime` - Line 130
- `validate_decimal_precision(value: int | float | str | Decimal, field_name: str, max_decimal_places: int = 8) -> bool` - Line 172
- `validate_bid_ask_spread(bid: Decimal, ask: Decimal) -> bool` - Line 207
- `validate_price_consistency(data: MarketData) -> bool` - Line 235
- `create_validation_issue(field, ...) -> ValidationIssue` - Line 277

### Implementation: `MarketDataValidator` ✅

**Purpose**: Consolidated market data validator that replaces multiple duplicate implementations
**Status**: Complete

**Implemented Methods:**
- `validate_market_data_record(self, data: MarketData) -> bool` - Line 351
- `validate_market_data_batch(self, data_list: list[MarketData]) -> list[MarketData]` - Line 438
- `get_validation_errors(self) -> list[str]` - Line 473
- `reset(self) -> None` - Line 477

### Implementation: `ValidationType` ✅

**Inherits**: Enum
**Purpose**: Types of validation operations
**Status**: Complete

### Implementation: `ValidationContext` ✅

**Inherits**: BaseModel
**Purpose**: Context information for validation operations
**Status**: Complete

**Implemented Methods:**
- `get_context_hash(self) -> str` - Line 110

### Implementation: `ValidationDetail` ✅

**Inherits**: BaseModel
**Purpose**: Detailed validation information
**Status**: Complete

### Implementation: `ValidationResult` ✅

**Inherits**: BaseModel
**Purpose**: Comprehensive validation result
**Status**: Complete

**Implemented Methods:**
- `add_error(self, ...) -> None` - Line 145
- `add_warning(self, ...) -> None` - Line 169
- `get_error_summary(self) -> str` - Line 189
- `get_critical_errors(self) -> list[ValidationDetail]` - Line 197
- `has_critical_errors(self) -> bool` - Line 201

### Implementation: `ValidationRule` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for validation rules
**Status**: Abstract Base Class

**Implemented Methods:**
- `async validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult` - Line 214
- `get_cache_key(self, value: Any, context: ValidationContext | None = None) -> str` - Line 220

### Implementation: `NumericValidationRule` ✅

**Inherits**: ValidationRule
**Purpose**: Validation rule for numeric values
**Status**: Complete

**Implemented Methods:**
- `async validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult` - Line 244

### Implementation: `StringValidationRule` ✅

**Inherits**: ValidationRule
**Purpose**: Validation rule for string values
**Status**: Complete

**Implemented Methods:**
- `async validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult` - Line 354

### Implementation: `ValidationCache` ✅

**Purpose**: Thread-safe validation cache with TTL support
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> ValidationResult | None` - Line 446
- `async set(self, key: str, result: ValidationResult, ttl: int | None = None) -> None` - Line 461
- `get_stats(self) -> dict[str, Any]` - Line 511

### Implementation: `ValidatorRegistry` ✅

**Purpose**: Registry for validation rules
**Status**: Complete

**Implemented Methods:**
- `register(self, rule: ValidationRule) -> None` - Line 531
- `get(self, name: str) -> ValidationRule | None` - Line 536
- `list_rules(self) -> list[str]` - Line 540

### Implementation: `ValidationService` ✅

**Inherits**: BaseService
**Purpose**: Comprehensive validation service for the T-Bot trading system
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 721
- `async shutdown(self) -> None` - Line 725
- `async validate_order(self, order_data: dict[str, Any], context: ValidationContext | None = None) -> ValidationResult` - Line 877
- `async validate_risk_parameters(self, risk_data: dict[str, Any], context: ValidationContext | None = None) -> ValidationResult` - Line 952
- `async validate_strategy_config(self, strategy_data: dict[str, Any], context: ValidationContext | None = None) -> ValidationResult` - Line 1044
- `async validate_market_data(self, market_data: dict[str, Any], context: ValidationContext | None = None) -> ValidationResult` - Line 1212
- `async validate_batch(self, ...) -> dict[str, ValidationResult]` - Line 1276
- `validate_price(self, price: Any) -> bool` - Line 1439
- `validate_quantity(self, quantity: Any) -> bool` - Line 1478
- `validate_symbol(self, symbol: str) -> bool` - Line 1517
- `register_custom_rule(self, rule: ValidationRule) -> None` - Line 1552
- `validate_decimal(self, value: Any) -> Any` - Line 1557
- `get_validation_stats(self) -> dict[str, Any]` - Line 1582

### Implementation: `ValidationCategory` ✅

**Inherits**: Enum
**Purpose**: Validation category types
**Status**: Complete

### Implementation: `ValidationSeverity` ✅

**Inherits**: Enum
**Purpose**: Validation severity levels
**Status**: Complete

### Implementation: `QualityDimension` ✅

**Inherits**: Enum
**Purpose**: Data quality dimensions
**Status**: Complete

### Implementation: `ValidationIssue` ✅

**Purpose**: Standardized validation issue record
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict[str, Any]` - Line 64

### Implementation: `QualityScore` ✅

**Purpose**: Data quality score breakdown
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict[str, float]` - Line 93

### Implementation: `ValidationResult` ✅

**Inherits**: Enum
**Purpose**: Standard validation result enumeration
**Status**: Complete

### Implementation: `ErrorType` ✅

**Inherits**: Enum
**Purpose**: Standard error type classification
**Status**: Complete

### Implementation: `RecoveryStatus` ✅

**Inherits**: Enum
**Purpose**: Standard recovery operation status
**Status**: Complete

### Implementation: `AuditEventType` ✅

**Inherits**: Enum
**Purpose**: Standard audit event types
**Status**: Complete

### Implementation: `StateValidationError` ✅

**Purpose**: Standard state validation error structure
**Status**: Complete

### Implementation: `ValidationWarning` ✅

**Purpose**: Standard validation warning structure
**Status**: Complete

### Implementation: `ValidationResultData` ✅

**Purpose**: Standard validation result data structure
**Status**: Complete

### Implementation: `AuditEntry` ✅

**Purpose**: Standard audit trail entry structure
**Status**: Complete

### Implementation: `ValidationFramework` ✅

**Purpose**: Centralized validation framework that provides simplified interfaces
to the comprehensive validation
**Status**: Complete

**Implemented Methods:**
- `validate_order(order: dict[str, Any]) -> bool` - Line 464
- `validate_strategy_params(params: dict[str, Any]) -> bool` - Line 489
- `validate_price(price: Any, max_price: Decimal = Any) -> Decimal` - Line 512
- `validate_quantity(quantity: Any, min_qty: Decimal = Any) -> Decimal` - Line 534
- `validate_symbol(symbol: str) -> str` - Line 556
- `validate_exchange_credentials(credentials: dict[str, Any]) -> bool` - Line 577
- `validate_risk_params(params: dict[str, Any]) -> bool` - Line 597
- `validate_risk_parameters(params: dict[str, Any]) -> bool` - Line 618
- `validate_timeframe(timeframe: str) -> str` - Line 634
- `validate_batch(validations: list[tuple[str, Callable[[Any], Any], Any]]) -> dict[str, Any]` - Line 654

### Implementation: `WebSocketConnectionManager` ✅

**Purpose**: Base WebSocket connection manager with common functionality
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 95
- `async disconnect(self) -> None` - Line 151
- `async send_message(self, message: dict[str, Any]) -> bool` - Line 184
- `add_callback(self, channel: str, callback: Callable) -> None` - Line 214
- `remove_callback(self, channel: str, callback: Callable) -> None` - Line 226
- `async subscribe_channel(self, channel: str, subscription_data: dict[str, Any]) -> bool` - Line 242
- `async unsubscribe_channel(self, channel: str, unsubscription_data: dict[str, Any]) -> bool` - Line 263
- `get_connection_stats(self) -> dict[str, Any]` - Line 489

### Implementation: `AuthenticatedWebSocketManager` ✅

**Inherits**: WebSocketConnectionManager
**Purpose**: WebSocket manager with authentication support
**Status**: Complete

**Implemented Methods:**
- `async authenticate(self) -> bool` - Line 535

### Implementation: `MultiStreamWebSocketManager` ✅

**Purpose**: Manager for multiple WebSocket connections
**Status**: Complete

**Implemented Methods:**
- `add_connection(self, name: str, connection: WebSocketConnectionManager) -> None` - Line 561
- `async connect_all(self) -> dict[str, bool]` - Line 571
- `async disconnect_all(self) -> None` - Line 616
- `get_connection(self, name: str) -> WebSocketConnectionManager | None` - Line 641
- `get_all_stats(self) -> dict[str, dict[str, Any]]` - Line 653

### Implementation: `WebSocketMessageBuffer` ✅

**Purpose**: Buffer for WebSocket messages during connection issues
**Status**: Complete

**Implemented Methods:**
- `is_full(self) -> bool` - Line 674
- `add_message(self, message: dict[str, Any]) -> None` - Line 678
- `get_messages(self, count: int | None = None) -> list[dict[str, Any]]` - Line 690
- `get_message_count(self) -> int` - Line 709

### Implementation: `WebSocketHeartbeatManager` ✅

**Purpose**: Manager for WebSocket heartbeat/ping functionality
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 737
- `async stop(self) -> None` - Line 745

### Implementation: `WebSocketSubscriptionManager` ✅

**Purpose**: Manager for WebSocket subscriptions and callbacks
**Status**: Complete

**Implemented Methods:**
- `add_subscription(self, stream_name: str, callback: Callable) -> None` - Line 793
- `remove_subscription(self, stream_name: str) -> None` - Line 804
- `get_subscriptions(self) -> list[str]` - Line 814
- `async handle_message(self, message: dict[str, Any]) -> None` - Line 818

### Implementation: `WebSocketStreamManager` ✅

**Purpose**: Manager for WebSocket streams
**Status**: Complete

**Implemented Methods:**
- `add_stream(self, stream_id: str, stream_config: dict[str, Any], handler: Callable) -> bool` - Line 856
- `remove_stream(self, stream_id: str) -> bool` - Line 875
- `get_stream_count(self) -> int` - Line 891
- `is_at_capacity(self) -> bool` - Line 895
- `async handle_stream_message(self, stream_id: str, message: dict[str, Any]) -> None` - Line 899

### Implementation: `BaseWebSocketManager` ✅

**Purpose**: Base WebSocket connection manager with common functionality
**Status**: Complete

**Implemented Methods:**
- `async connect(self, websocket: WebSocket, user_id: str) -> bool` - Line 35
- `disconnect(self, user_id: str)` - Line 58
- `subscribe(self, user_id: str, subscription_key: str)` - Line 79
- `unsubscribe(self, user_id: str, subscription_key: str)` - Line 107
- `async send_personal_message(self, message: dict[str, Any], user_id: str) -> bool` - Line 131
- `async broadcast_to_subscription(self, message: dict[str, Any], subscription_key: str)` - Line 162
- `async broadcast_to_all(self, message: dict[str, Any])` - Line 198
- `get_connection_count(self) -> int` - Line 233
- `get_subscription_count(self, subscription_key: str) -> int` - Line 237
- `get_user_subscriptions(self, user_id: str) -> set[str]` - Line 241
- `is_user_subscribed(self, user_id: str, subscription_key: str) -> bool` - Line 245

### Implementation: `MarketDataWebSocketManager` ✅

**Inherits**: BaseWebSocketManager
**Purpose**: WebSocket manager specifically for market data streaming
**Status**: Complete

**Implemented Methods:**
- `subscribe_to_symbol(self, user_id: str, symbol: str)` - Line 256
- `unsubscribe_from_symbol(self, user_id: str, symbol: str)` - Line 260
- `async broadcast_market_data(self, symbol: str, data: dict[str, Any])` - Line 264

### Implementation: `BotStatusWebSocketManager` ✅

**Inherits**: BaseWebSocketManager
**Purpose**: WebSocket manager specifically for bot status updates
**Status**: Complete

**Implemented Methods:**
- `subscribe_to_bot(self, user_id: str, bot_id: str)` - Line 292
- `unsubscribe_from_bot(self, user_id: str, bot_id: str)` - Line 296
- `async broadcast_bot_status(self, bot_id: str, status_data: dict[str, Any])` - Line 300

### Implementation: `ExchangeWebSocketReconnectionManager` ✅

**Purpose**: Common reconnection logic for exchange WebSocket connections
**Status**: Complete

**Implemented Methods:**
- `reset_reconnect_attempts(self) -> None` - Line 407
- `should_attempt_reconnect(self) -> bool` - Line 411
- `calculate_reconnect_delay(self) -> float` - Line 415
- `async schedule_reconnect(self, reconnect_callback) -> None` - Line 422
- `cancel_reconnect(self) -> None` - Line 456

## COMPLETE API REFERENCE

### File: arbitrage_helpers.py

**Key Imports:**
- `from src.core.exceptions import ArbitrageError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.constants import GLOBAL_FEE_STRUCTURE`
- `from src.utils.constants import PRECISION_LEVELS`

#### Class: `FeeCalculator`

**Purpose**: Common fee calculation utilities for arbitrage strategies

```python
class FeeCalculator:
    def calculate_cross_exchange_fees(buy_price: Decimal, sell_price: Decimal) -> Decimal  # Line 35
    def calculate_triangular_fees(rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal  # Line 100
```

#### Class: `OpportunityAnalyzer`

**Purpose**: Common opportunity analysis utilities for arbitrage strategies

```python
class OpportunityAnalyzer:
    def calculate_priority(profit_percentage: Decimal, arbitrage_type: str) -> Decimal  # Line 192
    def prioritize_opportunities(signals: list[Any], max_opportunities: int = 10) -> list[Any]  # Line 264
```

#### Class: `SpreadAnalyzer`

**Purpose**: Common spread analysis utilities for arbitrage strategies

```python
class SpreadAnalyzer:
    def calculate_spread_percentage(buy_price: Decimal, sell_price: Decimal) -> Decimal  # Line 313
    def calculate_net_profit(gross_spread: Decimal, fees: Decimal, base_price: Decimal) -> tuple[Decimal, Decimal]  # Line 338
```

#### Class: `PositionSizingCalculator`

**Purpose**: Position sizing utilities for arbitrage strategies

```python
class PositionSizingCalculator:
    def calculate_arbitrage_position_size(total_capital, ...) -> Decimal  # Line 376
```

#### Class: `MarketDataValidator`

**Purpose**: Market data validation utilities for arbitrage strategies

```python
class MarketDataValidator:
    def validate_price_data(price_data: dict[str, Any]) -> bool  # Line 476
    def check_arbitrage_thresholds(profit_percentage, ...) -> bool  # Line 519
```

### File: attribution_structures.py

#### Functions:

```python
def create_empty_attribution_structure() -> dict[str, Any]  # Line 11
def create_empty_service_attribution_structure() -> dict[str, Any]  # Line 31
def create_symbol_attribution_summary(...) -> dict[str, Any]  # Line 47
def create_attribution_summary(...) -> dict[str, Any]  # Line 73
```

### File: bot_service_helpers.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `ServiceHealthChecker`

**Purpose**: Helper for standardized service health checks

```python
class ServiceHealthChecker:
    async def check_service_health(service: Any, service_name: str, default_healthy: bool = True) -> dict[str, Any]  # Line 489
```

#### Functions:

```python
def create_fallback_decorator(name: str, func_type: str = 'sync') -> Callable  # Line 25
def safe_import_decorators() -> dict[str, Callable]  # Line 72
def safe_import_error_handling() -> dict[str, Any]  # Line 104
def safe_import_monitoring() -> dict[str, Any]  # Line 180
def resolve_service_dependencies(...) -> None  # Line 216
async def safe_close_connection(connection: Any, connection_name: str = 'connection', timeout: float = 5.0) -> None  # Line 246
async def safe_close_connections(...) -> None  # Line 267
def create_resource_usage_entry(...) -> dict[str, Any]  # Line 301
def safe_record_metric(...) -> None  # Line 326
async def execute_with_timeout_and_cleanup(...) -> Any  # Line 361
def create_bot_state_data(...) -> dict[str, Any]  # Line 404
def create_bot_metrics_data(bot_id: str, additional_fields: dict[str, Any] | None = None) -> dict[str, Any]  # Line 446
def batch_process_async(...) -> Callable  # Line 539
```

### File: cache_utilities.py

**Key Imports:**
- `from src.core.exceptions import CacheError`
- `from src.core.logging import get_logger`

#### Class: `CacheLevel`

**Inherits**: Enum
**Purpose**: Cache level enumeration

```python
class CacheLevel(Enum):
```

#### Class: `CacheStrategy`

**Inherits**: Enum
**Purpose**: Cache strategy enumeration

```python
class CacheStrategy(Enum):
```

#### Class: `CacheMode`

**Inherits**: Enum
**Purpose**: Cache operation mode

```python
class CacheMode(Enum):
```

#### Class: `CacheEntry`

**Purpose**: Cache entry with metadata

```python
class CacheEntry:
    def __post_init__(self)  # Line 63
    def is_expired(self) -> bool  # Line 68
    def update_access(self) -> None  # Line 75
```

#### Class: `CacheStats`

**Purpose**: Cache statistics

```python
class CacheStats:
    def calculate_hit_rate(self) -> float  # Line 95
    def calculate_memory_usage_mb(self) -> float  # Line 103
```

#### Class: `CacheSerializationUtils`

**Purpose**: Shared serialization utilities for cache implementations

```python
class CacheSerializationUtils:
    def serialize_json(value: Any) -> str  # Line 113
    def deserialize_json(data: str) -> Any  # Line 133
    def serialize_pickle(value: Any) -> bytes  # Line 153
    def deserialize_pickle(data: bytes) -> Any  # Line 173
    def calculate_size_bytes(value: Any) -> int  # Line 193
```

#### Class: `CacheKeyUtils`

**Purpose**: Shared cache key utilities

```python
class CacheKeyUtils:
    def generate_key(prefix: str, *args: Any) -> str  # Line 217
    def validate_key(key: str) -> bool  # Line 239
```

#### Class: `CacheLRUUtils`

**Purpose**: Shared LRU cache utilities

```python
class CacheLRUUtils:
    def update_lru_order(cache: OrderedDict, key: str) -> None  # Line 268
    def evict_lru(cache: OrderedDict, stats: CacheStats) -> str | None  # Line 280
```

#### Class: `CacheValidationUtils`

**Purpose**: Shared cache validation utilities

```python
class CacheValidationUtils:
    def validate_ttl(ttl: int | None) -> bool  # Line 310
    def validate_cache_size(max_size: int) -> bool  # Line 326
    def validate_cache_config(config: dict[str, Any]) -> list[str]  # Line 339
```

### File: financial.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decimal_utils import ZERO`
- `from src.utils.decimal_utils import safe_divide`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `FinancialCalculator`

**Inherits**: CalculatorInterface
**Purpose**: Class for all financial calculations

```python
class FinancialCalculator(CalculatorInterface):
    def sharpe_ratio(returns, ...) -> Decimal  # Line 32
    def sortino_ratio(returns, ...) -> Decimal  # Line 81
    def calmar_ratio(returns: tuple[Decimal, Ellipsis], periods_per_year: int = 252) -> Decimal  # Line 94
    def moving_average(prices: tuple[Decimal, Ellipsis], period: int, ma_type: str = 'simple') -> Decimal  # Line 162
    def max_drawdown(prices: list[Decimal] | NDArray[np.float64]) -> tuple[Decimal, int, int]  # Line 222
    def kelly_criterion(win_probability, ...) -> Decimal  # Line 270
    def position_size_volatility_adjusted(account_balance, ...) -> Decimal  # Line 317
    def calculate_returns(prices: list[Decimal] | NDArray[np.float64], method: str = 'simple') -> list[Decimal]  # Line 362
    def risk_reward_ratio(entry_price: Decimal, stop_loss: Decimal, take_profit: Decimal) -> Decimal  # Line 410
    def expected_value(win_probability: Decimal, avg_win: Decimal, avg_loss: Decimal) -> Decimal  # Line 438
    def profit_factor(wins: list[Decimal], losses: list[Decimal]) -> Decimal  # Line 457
    def calculate_compound_return(principal: Decimal, rate: Decimal, periods: int) -> Decimal  # Line 481
    def calculate_sharpe_ratio(returns: list[Decimal], risk_free_rate: Decimal) -> Decimal  # Line 493
```

#### Functions:

```python
def get_financial_calculator(calculator: FinancialCalculator | None = None) -> FinancialCalculator  # Line 505
```

### File: capital_config.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Functions:

```python
def get_capital_config_defaults() -> dict[str, Any]  # Line 16
def load_capital_config(config_service: Any = None, config_key: str = 'capital_management') -> dict[str, Any]  # Line 58
def resolve_config_service(dependency_resolver: Any = None) -> Any | None  # Line 96
def extract_decimal_config(config: dict[str, Any], key: str, default: Decimal) -> Decimal  # Line 120
def extract_percentage_config(config: dict[str, Any], key: str, default: float) -> Decimal  # Line 144
def get_supported_currencies(config: dict[str, Any]) -> list[str]  # Line 173
def validate_config_values(config: dict[str, Any]) -> dict[str, Any]  # Line 195
```

### File: capital_errors.py

**Key Imports:**
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.formatters import format_currency`

#### Class: `CapitalErrorContext`

**Purpose**: Context manager for capital management operations with error handling

```python
class CapitalErrorContext:
    def __init__(self, operation: str, component: str, **context)  # Line 256
    async def __aenter__(self)  # Line 262
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 266
```

#### Functions:

```python
def handle_service_error(...) -> ServiceError | None  # Line 18
def handle_repository_error(...) -> Any  # Line 65
def log_allocation_operation(...) -> None  # Line 106
def log_fund_flow_operation(...) -> None  # Line 147
def create_operation_context(...) -> dict[str, Any]  # Line 188
def wrap_service_operation(operation_name: str, component: str)  # Line 223
```

### File: capital_resources.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `ResourceManager`

**Purpose**: Unified resource manager for capital management services

```python
class ResourceManager:
    def __init__(self, max_history_size: int = 1000)  # Line 24
    def limit_list_size(self, ...) -> list[Any]  # Line 39
    def clean_time_based_data(self, ...) -> dict[str, list[tuple[datetime, Any]]]  # Line 74
    def clean_fund_flows(self, fund_flows: list[Any], max_age_days: int = 30) -> list[Any]  # Line 121
    def clean_performance_data(self, performance_data: dict[str, dict[str, Any]], max_age_days: int = 30) -> dict[str, dict[str, Any]]  # Line 153
    def cleanup_allocations_data(self, allocations_data: list[dict[str, Any]]) -> None  # Line 198
    def should_trigger_cleanup(self, current_size: int, data_type: str = 'default') -> bool  # Line 220
    def get_memory_usage_info(self) -> dict[str, Any]  # Line 238
```

#### Functions:

```python
def get_resource_manager() -> ResourceManager  # Line 276
def cleanup_large_dict(data_dict: dict[Any, Any], max_size: int = 1000, keep_recent_keys: bool = True) -> dict[Any, Any]  # Line 286
def safe_clear_references(*objects) -> None  # Line 317
```

### File: capital_validation.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.utils.decimal_utils import safe_decimal_conversion`

#### Functions:

```python
def validate_capital_amount(...) -> None  # Line 17
def validate_strategy_id(strategy_id: str, component: str = 'CapitalManagement') -> None  # Line 53
def validate_exchange_name(exchange: str, component: str = 'CapitalManagement') -> None  # Line 72
def validate_currency_code(currency: str, component: str = 'CapitalManagement') -> None  # Line 91
def validate_percentage(...) -> None  # Line 118
def validate_allocation_request(...) -> None  # Line 158
def validate_withdrawal_request(...) -> None  # Line 178
def validate_supported_currencies(...) -> None  # Line 203
def safe_decimal_conversion(value: Any, default: Decimal = Any) -> Decimal  # Line 231
```

### File: checksum_utilities.py

#### Functions:

```python
def calculate_state_checksum(data: dict[str, Any]) -> str  # Line 14
def verify_state_integrity(data: dict[str, Any], expected_checksum: str) -> bool  # Line 41
def calculate_metadata_hash(metadata: dict[str, Any]) -> str  # Line 62
def batch_calculate_checksums(data_items: list[dict[str, Any]]) -> list[str]  # Line 78
```

### File: config_conversion.py

#### Functions:

```python
def convert_config_to_dict(config: Any) -> dict[str, Any]  # Line 11
```

### File: core_utilities.py

**Key Imports:**
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.exceptions import ComponentError`
- `from src.core.exceptions import ConnectionError`
- `from src.core.exceptions import DependencyError`
- `from src.core.logging import get_logger`

#### Class: `DependencyInjectionMixin`

**Purpose**: Mixin providing common dependency injection patterns

```python
class DependencyInjectionMixin:
    def __init__(self, *args, **kwargs) -> None  # Line 35
    def configure_dependencies(self, container: Any) -> None  # Line 40
    def get_dependencies(self) -> list[str]  # Line 62
    def _resolve_dependency(self, type_name: str, param_name: str) -> Any  # Line 71
    def _inject_dependencies_into_kwargs(self, target_callable: Callable, kwargs: dict[str, Any]) -> dict[str, Any]  # Line 105
```

#### Class: `ConnectionManagerMixin`

**Purpose**: Mixin providing common connection management patterns

```python
class ConnectionManagerMixin:
    def __init__(self, *args, **kwargs) -> None  # Line 173
    async def _ensure_connection(self, client_attr: str, reconnect_method: str) -> None  # Line 181
    async def _is_connection_healthy(self, client: Any) -> bool  # Line 200
    async def _reconnect_with_backoff(self, connect_method: str, test_method: str = 'ping') -> None  # Line 212
```

#### Class: `LifecycleManagerMixin`

**Purpose**: Mixin providing common lifecycle management patterns

```python
class LifecycleManagerMixin:
    def __init__(self, *args, **kwargs) -> None  # Line 255
    async def _start_lifecycle(self) -> None  # Line 263
    async def _stop_lifecycle(self) -> None  # Line 275
    def _create_background_task(self, coro, name: str | None = None) -> asyncio.Task  # Line 291
    async def _cancel_background_tasks(self) -> None  # Line 312
    def uptime(self) -> float | None  # Line 336
```

#### Class: `HealthCheckMixin`

**Purpose**: Mixin providing common health check patterns

```python
class HealthCheckMixin:
    async def basic_health_check(self) -> HealthStatus  # Line 348
```

#### Class: `ResourceCleanupMixin`

**Purpose**: Mixin providing common resource cleanup patterns

```python
class ResourceCleanupMixin:
    def __init__(self, *args, **kwargs) -> None  # Line 381
    def register_cleanup_callback(self, callback: Callable) -> None  # Line 385
    async def _cleanup_resources(self) -> None  # Line 394
```

#### Class: `LoggingHelperMixin`

**Purpose**: Mixin providing structured logging helpers

```python
class LoggingHelperMixin:
    def _log_operation_start(self, operation: str, **context) -> datetime  # Line 413
    def _log_operation_success(self, operation: str, start_time: datetime, **context) -> None  # Line 436
    def _log_operation_error(self, operation: str, start_time: datetime, error: Exception, **context) -> None  # Line 456
```

#### Class: `BaseUtilityMixin`

**Inherits**: DependencyInjectionMixin, ConnectionManagerMixin, LifecycleManagerMixin, HealthCheckMixin, ResourceCleanupMixin, LoggingHelperMixin
**Purpose**: Combined utility mixin with all common patterns

```python
class BaseUtilityMixin(DependencyInjectionMixin, ConnectionManagerMixin, LifecycleManagerMixin, HealthCheckMixin, ResourceCleanupMixin, LoggingHelperMixin):
    def __init__(self, *args, **kwargs) -> None  # Line 534
    async def initialize(self) -> None  # Line 537
    async def shutdown(self) -> None  # Line 541
```

#### Functions:

```python
async def managed_connection(connection_manager: Any, client_attr: str, reconnect_method: str) -> AsyncGenerator[Any, None]  # Line 484
async def operation_logging(logger_mixin: Any, operation: str, **context: Any) -> AsyncGenerator[None, None]  # Line 504
```

### File: data_flow_consistency_validator.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `DataFlowConsistencyValidator`

**Purpose**: Validates data flow consistency between risk_management and utils modules

```python
class DataFlowConsistencyValidator:
    def __init__(self)  # Line 23
    def validate_all(self) -> dict[str, Any]  # Line 28
    def _validate_data_transformations(self) -> dict[str, Any]  # Line 71
    def _validate_message_patterns(self) -> dict[str, Any]  # Line 123
    def _validate_processing_alignment(self) -> dict[str, Any]  # Line 168
    def _validate_boundary_consistency(self) -> dict[str, Any]  # Line 206
    def _validate_error_propagation(self) -> dict[str, Any]  # Line 247
    def _validate_financial_types(self) -> dict[str, Any]  # Line 297
```

#### Functions:

```python
def validate_data_flow_consistency() -> dict[str, Any]  # Line 341
def log_consistency_results(results: dict[str, Any]) -> None  # Line 352
```

### File: data_flow_integrity.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.monitoring.financial_precision import FinancialPrecisionWarning`
- `from src.monitoring.financial_precision import safe_decimal_to_float`

#### Class: `DataFlowValidator`

**Purpose**: Validates data flow consistency across module boundaries

```python
class DataFlowValidator:
    def validate_message_pattern_consistency(data: dict[str, Any]) -> None  # Line 38
    def validate_boundary_crossing_metadata(data: dict[str, Any]) -> None  # Line 89
    def validate_complete_data_flow(cls, ...) -> None  # Line 123
```

#### Class: `DataFlowIntegrityError`

**Inherits**: Exception
**Purpose**: Raised when data flow integrity is compromised

```python
class DataFlowIntegrityError(Exception):
```

#### Class: `PrecisionTracker`

**Purpose**: Tracks precision loss across data transformations

```python
class PrecisionTracker:
    def __init__(self) -> None  # Line 185
    def track_conversion(self, ...) -> None  # Line 190
    def _calculate_relative_error(self, original: Decimal, converted: float) -> float  # Line 223
    def get_summary(self) -> dict[str, Any]  # Line 236
    def _get_avg_relative_error(self) -> float  # Line 245
    def track_operation(self, operation: str, input_precision: int, output_precision: int) -> None  # Line 254
    def get_precision_stats(self) -> dict[str, Any]  # Line 272
```

#### Class: `DataFlowValidator`

**Purpose**: Comprehensive data flow validation system

```python
class DataFlowValidator:
    def __init__(self) -> None  # Line 310
    def _setup_default_rules(self) -> None  # Line 314
    def validate_data_flow(self, data: dict[str, Any], context: str = 'unknown') -> dict[str, Any]  # Line 348
    def _validate_field(self, field_name: str, value: Any, context: str) -> Any  # Line 375
    def _validate_null_handling_service(self, value: Any, allow_null: bool = False, field_name: str = 'value') -> Any  # Line 454
    def _validate_type_conversion_service(self, ...) -> Any  # Line 482
    def _validate_financial_range_service(self, ...) -> Decimal  # Line 560
    def _get_rule_for_field(self, field_name: str) -> dict[str, Any] | None  # Line 603
    def add_validation_rule(self, field_pattern: str, rule: dict[str, Any]) -> None  # Line 618
    def validate_data_integrity(self, data: Any) -> bool  # Line 623
    def get_validation_report(self) -> dict[str, Any]  # Line 638
```

#### Class: `IntegrityPreservingConverter`

**Purpose**: Converter that preserves data integrity across module boundaries

```python
class IntegrityPreservingConverter:
    def __init__(self, ...)  # Line 655
    def safe_convert_for_metrics(self, ...) -> float  # Line 667
    def batch_convert_with_integrity(self, ...) -> dict[str, float]  # Line 717
```

#### Functions:

```python
def get_precision_tracker(tracker: PrecisionTracker | None = None) -> PrecisionTracker  # Line 277
def get_data_flow_validator(validator: DataFlowValidator | None = None) -> DataFlowValidator  # Line 746
def get_integrity_converter(converter: IntegrityPreservingConverter | None = None) -> IntegrityPreservingConverter  # Line 768
def validate_cross_module_data(...) -> dict[str, Any]  # Line 795
def fix_precision_cascade(data: dict[str, Any] | None = None, target_formats: dict[str, str] | None = None) -> dict[str, Any]  # Line 852
```

### File: data_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.utils.decimal_utils import ZERO`
- `from src.utils.decimal_utils import to_decimal`

#### Functions:

```python
def dict_to_dataframe(data: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame  # Line 20
def normalize_array(arr: list[float] | NDArray[np.float64]) -> NDArray[np.float64]  # Line 51
def convert_currency(amount: Decimal, from_currency: str, to_currency: str, exchange_rate: Decimal) -> Decimal  # Line 80
def normalize_price(price: Decimal | int, symbol: str, precision: int | None = None) -> Decimal  # Line 137
def flatten_dict(d: dict[str, Any], parent_key: str = '', sep: str = '.') -> dict[str, Any]  # Line 191
def unflatten_dict(d: dict[str, Any], sep: str = '.') -> dict[str, Any]  # Line 213
def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]  # Line 236
def filter_none_values(d: dict[str, Any]) -> dict[str, Any]  # Line 259
def chunk_list(lst: list[Any], chunk_size: int) -> list[list[Any]]  # Line 272
```

### File: datetime_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`

#### Functions:

```python
def get_current_utc_timestamp() -> datetime  # Line 11
def to_timestamp(dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str  # Line 16
def parse_timeframe(timeframe: str) -> int  # Line 32
def get_trading_session(dt: datetime, exchange: str = 'binance') -> str  # Line 61
def is_market_open(dt: datetime, exchange: str = 'binance') -> bool  # Line 102
def convert_timezone(dt: datetime, target_tz: str) -> datetime  # Line 125
def parse_datetime(dt_str: str, format_str: str | None = None) -> datetime  # Line 159
def get_redis_key_ttl(key: str, default_ttl: int = 3600) -> int  # Line 199
```

### File: decimal_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `DecimalEncoder`

**Purpose**: JSON encoder that handles Decimal values

```python
class DecimalEncoder:
    def encode(obj: Any) -> Any  # Line 530
```

#### Class: `FloatDeprecationWarning`

**Purpose**: Context manager to detect and warn about float usage in financial code

```python
class FloatDeprecationWarning:
    def __enter__(self) -> 'FloatDeprecationWarning'  # Line 556
    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any | None) -> None  # Line 561
    def warn_float_usage(context: str) -> None  # Line 568
```

#### Functions:

```python
def to_decimal(value: str | int | float | Decimal, context: Context | None = None) -> Decimal  # Line 70
def decimal_to_str(value: Decimal, precision: int | None = None) -> str  # Line 141
def round_price(price: Decimal, tick_size: Decimal) -> Decimal  # Line 161
def round_quantity(quantity: Decimal, lot_size: Decimal) -> Decimal  # Line 178
def round_to_precision(value: Any, precision: int) -> Decimal  # Line 196
def calculate_percentage(value: Decimal, percentage: Decimal) -> Decimal  # Line 215
def calculate_basis_points(value: Decimal, bps: Decimal) -> Decimal  # Line 229
def safe_divide(numerator: Decimal, denominator: Decimal, default: Decimal = ZERO) -> Decimal  # Line 243
def validate_positive(value: Decimal, name: str = 'value') -> None  # Line 262
def validate_non_negative(value: Decimal, name: str = 'value') -> None  # Line 277
def validate_percentage(value: Decimal, name: str = 'percentage') -> None  # Line 292
def compare_decimals(a: Decimal, b: Decimal, tolerance: Decimal = SATOSHI) -> int  # Line 307
def safe_decimal_conversion(value: Any, precision: int = 8, default: Decimal = ZERO) -> Decimal  # Line 325
def format_decimal(value: Decimal, decimals: int = 8, thousands_sep: bool = True) -> str  # Line 370
def sum_decimals(values: list[Decimal]) -> Decimal  # Line 419
def avg_decimals(values: list[Decimal]) -> Decimal  # Line 432
def min_decimal(*values: Decimal) -> Decimal  # Line 448
def max_decimal(*values: Decimal) -> Decimal  # Line 461
def clamp_decimal(value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal  # Line 474
def decimal_to_float(value: Decimal) -> float  # Line 489
def float_to_decimal(value: float) -> Decimal  # Line 513
```

### File: decorators.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `ExceptionCategory`

**Purpose**: Classification of exceptions for intelligent retry behavior

```python
class ExceptionCategory:
    def should_retry(cls, exception: Exception) -> bool  # Line 90
    def get_retry_delay(cls, exception: Exception, attempt: int, base_delay: float) -> float  # Line 130
```

#### Class: `UnifiedDecorator`

**Purpose**: Single configurable decorator replacing multiple decorators

```python
class UnifiedDecorator:
    def _get_cache_key(cls, ...) -> str  # Line 187
    def _cache_result(cls, ...) -> None  # Line 198
    def _get_cached_result(cls, ...) -> Any | None  # Line 218
    def _cleanup_cache_if_needed(cls) -> None  # Line 240
    def _force_cache_cleanup(cls) -> None  # Line 294
    def clear_cache(cls) -> None  # Line 326
    async def _with_retry(cls, ...) -> Any  # Line 334
    def _record_metrics(cls, func: Callable[Ellipsis, Any], result: Any, execution_time: float) -> None  # Line 410
    def _bind_function_arguments(cls, ...) -> Any  # Line 418
    def _validate_numeric_param(cls, param_name: str, value: Any) -> None  # Line 433
    def _validate_symbol_param(cls, param_name: str, value: Any) -> None  # Line 442
    def _validate_single_param(cls, param_name: str, value: Any) -> None  # Line 448
    def _validate_args(cls, ...) -> None  # Line 465
    def _create_enhancement_config(cls, ...) -> dict[str, Any]  # Line 484
    def _prepare_execution_context(cls, func: Callable[Ellipsis, Any], config: dict[str, Any]) -> dict[str, Any]  # Line 514
    def _handle_validation(cls, ...) -> None  # Line 523
    def _handle_pre_execution_logging(cls, ...) -> None  # Line 541
    def _handle_cache_check(cls, ...) -> Any  # Line 555
    def _handle_post_execution(cls, ...) -> None  # Line 573
    def enhance(cls, ...) -> Callable[[F], F]  # Line 599
    async def _execute_async_enhanced(cls, ...) -> Any  # Line 665
    def _execute_sync_enhanced(cls, ...) -> Any  # Line 722
    def _execute_sync_with_retry(cls, ...) -> Any  # Line 748
    async def _handle_execution_error_async(cls, ...) -> Any  # Line 822
    def _handle_execution_error_sync(cls, ...) -> Any  # Line 871
    async def _execute_with_retry(cls, ...) -> Any  # Line 915
```

#### Functions:

```python
def retry(max_attempts: int = 3, delay: float = 1.0, base_delay: float | None = None) -> Callable[[F], F]  # Line 940
def cached(ttl: int = 300) -> Callable[[F], F]  # Line 949
def validated() -> Callable[[F], F]  # Line 954
def logged(level: str = 'info') -> Callable[[F], F]  # Line 959
def monitored() -> Callable[[F], F]  # Line 964
def timeout(seconds: float) -> Callable[[F], F]  # Line 969
def _make_hybrid_decorator(...) -> Callable[Ellipsis, Any]  # Line 975
def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60) -> Callable[[F], F]  # Line 1024
def robust() -> Callable[[F], F]  # Line 1029
```

### File: error_categorization.py

#### Functions:

```python
def contains_keywords(text: str, keywords: list[str]) -> bool  # Line 78
def categorize_by_keywords(text: str, keyword_map: dict[str, list[str]]) -> str | None  # Line 86
def get_error_category_keywords() -> dict[str, list[str]]  # Line 98
def get_fallback_type_keywords() -> dict[str, list[str]]  # Line 110
def is_financial_component(component_name: str) -> bool  # Line 123
def is_sensitive_key(key_name: str) -> bool  # Line 128
def categorize_error_from_type_and_message(error_type: str, error_message: str) -> str  # Line 133
def categorize_error_from_message(message: str) -> str  # Line 187
def determine_alert_severity_from_message(message: str) -> str  # Line 207
def is_retryable_error(error_message: str) -> bool  # Line 224
def detect_rate_limiting(error_message: str) -> bool  # Line 240
def detect_auth_token_error(error_message: str) -> bool  # Line 245
def detect_data_validation_error(error_message: str) -> bool  # Line 250
```

### File: error_handling_utils.py

**Key Imports:**
- `from src.error_handling.security_validator import SensitivityLevel`

#### Functions:

```python
def get_or_create_sanitizer(sanitizer = None)  # Line 10
def sanitize_error_with_level(error: Exception, level: SensitivityLevel, sanitizer = None) -> str  # Line 27
def create_recovery_response(...) -> dict[str, Any]  # Line 43
def extract_field_from_error(error: Exception, context: dict[str, Any] | None = None) -> str | None  # Line 74
def extract_retry_after_from_error(error: Exception, context: dict[str, Any] | None = None)  # Line 107
```

### File: error_recovery_utilities.py

#### Class: `RecoveryStrategy`

**Inherits**: Enum
**Purpose**: Recovery strategy options

```python
class RecoveryStrategy(Enum):
```

#### Class: `ErrorContext`

**Purpose**: Context information for error handling

```python
class ErrorContext:
```

#### Class: `RecoveryCheckpoint`

**Purpose**: Recovery checkpoint for rollback operations

```python
class RecoveryCheckpoint:
```

#### Class: `BaseErrorRecovery`

**Purpose**: Base error recovery system providing common recovery patterns

```python
class BaseErrorRecovery:
    def __init__(self, component_name: str = '')  # Line 97
    def classify_error(self, exception: Exception) -> ErrorType  # Line 132
    async def create_recovery_checkpoint(self, operation: str, state_data: dict[str, Any] | None = None, **context) -> str  # Line 138
    async def handle_error(self, ...) -> ErrorContext  # Line 164
    async def rollback_to_checkpoint(self, checkpoint_id: str) -> bool  # Line 229
    async def _attempt_recovery(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 261
    async def _handle_database_connection_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 292
    async def _handle_database_integrity_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 307
    async def _handle_database_timeout_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 315
    async def _handle_redis_connection_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 326
    async def _handle_redis_timeout_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 336
    async def _handle_data_corruption_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 342
    async def _handle_disk_space_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 353
    async def _handle_permission_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 360
    async def _handle_validation_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 367
    async def _handle_concurrency_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 375
    async def _handle_unknown_error(self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None) -> bool  # Line 389
    def get_error_statistics(self) -> dict[str, Any]  # Line 398
```

### File: exchange_conversion_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.types import OrderBook`
- `from src.core.types import OrderBookLevel`
- `from src.core.types import OrderRequest`
- `from src.core.types import OrderResponse`

#### Class: `SymbolConversionUtils`

**Purpose**: Utilities for converting symbols between different exchange formats

```python
class SymbolConversionUtils:
    def normalize_symbol(symbol: str, target_exchange: str) -> str  # Line 65
    def to_binance_format(symbol: str) -> str  # Line 86
    def to_coinbase_format(symbol: str) -> str  # Line 96
    def to_okx_format(symbol: str) -> str  # Line 107
    def to_standard_format(symbol: str) -> str  # Line 118
    def _parse_concatenated_symbol(symbol: str, separator: str) -> str  # Line 129
    def get_base_quote(symbol: str) -> tuple[str, str]  # Line 143
```

#### Class: `OrderConversionUtils`

**Purpose**: Utilities for converting order parameters between exchanges

```python
class OrderConversionUtils:
    def convert_order_to_exchange_format(order, ...) -> dict[str, Any]  # Line 173
    def _to_binance_order(order: OrderRequest, precision_config: dict[str, int] | None = None) -> dict[str, Any]  # Line 198
    def _to_coinbase_order(order: OrderRequest, precision_config: dict[str, int] | None = None) -> dict[str, Any]  # Line 230
    def _to_okx_order(order: OrderRequest, precision_config: dict[str, int] | None = None) -> dict[str, Any]  # Line 262
    def _to_generic_order(order: OrderRequest, precision_config: dict[str, int] | None = None) -> dict[str, Any]  # Line 292
    def _get_binance_order_type(order_type: OrderType) -> str  # Line 323
    def _get_okx_order_type(order_type: OrderType) -> str  # Line 334
```

#### Class: `ResponseConversionUtils`

**Purpose**: Utilities for converting exchange responses to unified formats

```python
class ResponseConversionUtils:
    def create_unified_order_response(exchange_response: dict[str, Any], exchange: str, original_symbol: str = None) -> OrderResponse  # Line 349
    def _from_binance_response(response: dict[str, Any]) -> OrderResponse  # Line 375
    def _from_coinbase_response(response: dict[str, Any], original_symbol: str = None) -> OrderResponse  # Line 393
    def _from_okx_response(response: dict[str, Any]) -> OrderResponse  # Line 426
    def _from_generic_response(response: dict[str, Any]) -> OrderResponse  # Line 444
    def _parse_binance_order_type(binance_type: str) -> OrderType  # Line 460
    def _parse_coinbase_order_type(order_config: dict[str, Any]) -> OrderType  # Line 471
    def _parse_okx_order_type(okx_type: str) -> OrderType  # Line 482
    def _parse_timestamp(timestamp_str: str | int | None, format: str = 'milliseconds') -> datetime  # Line 492
```

#### Class: `MarketDataConversionUtils`

**Purpose**: Utilities for converting market data between formats

```python
class MarketDataConversionUtils:
    def create_unified_ticker(exchange_data: dict[str, Any], exchange: str, symbol: str) -> Ticker  # Line 518
    def create_unified_order_book(exchange_data: dict[str, Any], symbol: str, exchange: str) -> OrderBook  # Line 558
```

#### Class: `ExchangeConversionUtils`

**Purpose**: Utilities for converting between unified and exchange-specific formats

```python
class ExchangeConversionUtils:
    def create_order_response(exchange_data, ...) -> OrderResponse  # Line 606
    def get_binance_field_mapping() -> dict[str, str]  # Line 698
    def get_binance_status_mapping() -> dict[str, OrderStatus]  # Line 717
    def get_binance_type_mapping() -> dict[str, OrderType]  # Line 729
    def get_coinbase_field_mapping() -> dict[str, str]  # Line 741
    def get_coinbase_status_mapping() -> dict[str, OrderStatus]  # Line 757
    def get_coinbase_type_mapping() -> dict[str, OrderType]  # Line 773
    def get_okx_field_mapping() -> dict[str, str]  # Line 783
    def get_okx_status_mapping() -> dict[str, OrderStatus]  # Line 799
    def get_okx_type_mapping() -> dict[str, OrderType]  # Line 811
    def convert_binance_order_to_response(result: dict[str, Any]) -> OrderResponse  # Line 823
    def convert_coinbase_order_to_response(result: dict[str, Any]) -> OrderResponse  # Line 838
    def convert_okx_order_to_response(result: dict[str, Any]) -> OrderResponse  # Line 849
```

### File: exchange_error_utils.py

**Key Imports:**
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import ExchangeError`
- `from src.core.exceptions import ExchangeInsufficientFundsError`
- `from src.core.exceptions import ExchangeRateLimitError`
- `from src.core.exceptions import ExecutionError`

#### Class: `ExchangeErrorHandler`

**Purpose**: Common error handling utilities for exchanges

```python
class ExchangeErrorHandler:
    def __init__(self, exchange_name: str, config, error_handler: ErrorHandler | None = None)  # Line 33
    async def handle_exchange_error(self, ...) -> None  # Line 56
    async def handle_api_error(self, error: Exception, operation: str, context: dict[str, Any] | None = None) -> None  # Line 105
```

#### Class: `ErrorMappingUtils`

**Purpose**: Utilities for mapping exchange-specific errors to unified exceptions

```python
class ErrorMappingUtils:
    def map_exchange_error(error_data, ...) -> Exception  # Line 170
    def _map_binance_error(error_data: dict[str, Any], message: str, code: str) -> Exception  # Line 224
    def _map_coinbase_error(error_data: dict[str, Any], message: str, code: str) -> Exception  # Line 241
    def _map_okx_error(error_data: dict[str, Any], message: str, code: str) -> Exception  # Line 253
```

#### Class: `RetryableOperationHandler`

**Purpose**: Handler for retryable exchange operations

```python
class RetryableOperationHandler:
    def __init__(self, exchange_name: str, logger = None)  # Line 272
    async def execute_with_retry(self, operation_func, operation_name: str, *args, **kwargs) -> Any  # Line 284
    async def execute_with_aggressive_retry(self, operation_func, operation_name: str, *args, **kwargs) -> Any  # Line 313
```

#### Class: `OperationTimeoutHandler`

**Purpose**: Handler for operation timeouts

```python
class OperationTimeoutHandler:
    async def execute_with_timeout(operation_func, ...) -> Any  # Line 354
```

#### Class: `ExchangeCircuitBreaker`

**Purpose**: Simple circuit breaker for exchange operations

```python
class ExchangeCircuitBreaker:
    def __init__(self, ...)  # Line 382
    async def call(self, func, *args, **kwargs)  # Line 406
    def _should_attempt_reset(self) -> bool  # Line 449
    def _record_success(self) -> None  # Line 458
    def _record_failure(self) -> None  # Line 464
    def get_state(self) -> dict[str, Any]  # Line 474
```

### File: exchange_order_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import OrderRequest`
- `from src.core.types import OrderResponse`
- `from src.core.types import OrderSide`

#### Class: `OrderManagementUtils`

**Purpose**: Shared utilities for order management across exchanges

```python
class OrderManagementUtils:
    def validate_order_structure(self, order: OrderRequest) -> None  # Line 31
    def track_order(order_response, ...) -> None  # Line 53
    def update_order_status(order_id: str, status: OrderStatus, pending_orders: dict[str, dict[str, Any]]) -> None  # Line 83
```

#### Class: `OrderConversionUtils`

**Purpose**: Utilities for converting orders between exchange formats

```python
class OrderConversionUtils:
    def create_base_order_response(order_id, ...) -> OrderResponse  # Line 102
    def standardize_symbol_format(symbol: str, target_format: str = 'dash') -> str  # Line 147
```

#### Class: `OrderStatusUtils`

**Purpose**: Utilities for handling order status conversions

```python
class OrderStatusUtils:
    def convert_status(status: str, exchange: str) -> OrderStatus  # Line 199
    def is_terminal_status(status: OrderStatus) -> bool  # Line 251
```

#### Class: `OrderTypeUtils`

**Purpose**: Utilities for order type conversions

```python
class OrderTypeUtils:
    def convert_to_exchange_format(order_type: OrderType, exchange: str) -> str  # Line 274
    def convert_from_exchange_format(exchange_type: str, exchange: str) -> OrderType  # Line 318
```

#### Class: `AssetPrecisionUtils`

**Purpose**: Utilities for asset precision calculations

```python
class AssetPrecisionUtils:
    def get_asset_precision(symbol: str, precision_type: str = 'quantity') -> int  # Line 348
```

#### Class: `FeeCalculationUtils`

**Purpose**: Utilities for fee calculations

```python
class FeeCalculationUtils:
    def calculate_fee(order_value: Decimal, exchange: str, symbol: str, is_maker: bool = False) -> Decimal  # Line 401
    def get_fee_rates(exchange: str) -> dict[str, Decimal]  # Line 437
```

#### Functions:

```python
def get_order_management_utils() -> OrderManagementUtils  # Line 453
```

### File: exchange_validation_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import OrderRequest`
- `from src.core.types import OrderSide`
- `from src.core.types import OrderType`

#### Class: `ExchangeValidationUtils`

**Purpose**: Common validation utilities for exchanges

```python
class ExchangeValidationUtils:
    def validate_exchange_specific_order(order: OrderRequest, exchange: str) -> None  # Line 46
    def _validate_binance_order(order: OrderRequest) -> None  # Line 67
    def _validate_coinbase_order(order: OrderRequest) -> None  # Line 80
    def _validate_okx_order(order: OrderRequest) -> None  # Line 87
```

#### Class: `SymbolValidationUtils`

**Purpose**: Utilities for validating trading symbols

```python
class SymbolValidationUtils:
    def is_valid_symbol_format(symbol: str, exchange: str) -> bool  # Line 141
    def is_valid_binance_symbol(symbol: str) -> bool  # Line 162
    def is_valid_coinbase_symbol(symbol: str) -> bool  # Line 188
    def is_valid_okx_symbol(symbol: str) -> bool  # Line 214
    def get_supported_symbols(exchange: str) -> set[str]  # Line 240
```

#### Class: `PrecisionValidationUtils`

**Purpose**: Utilities for validating precision requirements

```python
class PrecisionValidationUtils:
    def validate_precision(value: Decimal, precision: int, value_type: str = 'value') -> None  # Line 272
    def validate_order_precision(order, ...) -> None  # Line 300
    def round_to_exchange_precision(value: Decimal, precision: int, rounding_mode = ROUND_HALF_UP) -> Decimal  # Line 336
```

#### Class: `RiskValidationUtils`

**Purpose**: Utilities for risk-based validation

```python
class RiskValidationUtils:
    def validate_order_size_limits(order, ...) -> None  # Line 361
    def validate_price_bounds(order, ...) -> None  # Line 398
    def validate_stop_price_logic(order: OrderRequest) -> None  # Line 426
```

#### Functions:

```python
def get_exchange_validation_utils() -> ExchangeValidationUtils  # Line 461
```

### File: execution_utils.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import Order`
- `from src.core.types import OrderRequest`
- `from src.utils.decimal_utils import ZERO`
- `from src.utils.decimal_utils import safe_divide`

#### Functions:

```python
def calculate_order_value(...) -> Decimal  # Line 16
def calculate_price_deviation_bps(order_price: Decimal, market_price: Decimal) -> Decimal  # Line 49
def is_order_within_price_bounds(order: Order, market_data: MarketData, max_deviation_percent: Decimal = Any) -> bool  # Line 67
def calculate_trade_risk_ratio(order_value: Decimal, account_value: Decimal) -> Decimal  # Line 95
def extract_order_details(order: Order) -> dict[str, Any]  # Line 112
def convert_order_to_request(order: Order) -> OrderRequest  # Line 134
def validate_order_basic(order: Order) -> list[str]  # Line 155
def calculate_slippage_bps(executed_price: Decimal, expected_price: Decimal) -> Decimal  # Line 179
```

### File: file_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Functions:

```python
def safe_read_file(file_path: str, encoding: str = 'utf-8') -> str  # Line 16
def safe_write_file(file_path: str, content: str, encoding: str = 'utf-8') -> None  # Line 49
def ensure_directory_exists(directory_path: str) -> None  # Line 90
def load_config_file(file_path: str) -> dict[str, Any]  # Line 110
def save_config_file(file_path: str, config: dict[str, Any]) -> None  # Line 145
def delete_file(file_path: str) -> None  # Line 174
def get_file_size(file_path: str) -> int  # Line 197
def list_files(directory: str, pattern: str = '*') -> list[str]  # Line 221
```

### File: formatters.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Functions:

```python
def format_currency(amount: Decimal | int, currency: str = 'USD', precision: int | None = None) -> str  # Line 43
def format_percentage(value: Decimal | int, precision: int = 2) -> str  # Line 99
def format_pnl(pnl: Decimal | int, currency: str = 'USD') -> tuple[str, str]  # Line 140
def format_quantity(quantity: Decimal | int, symbol: str) -> str  # Line 180
def format_price(price: Decimal | int, symbol: str) -> str  # Line 226
def format_api_response(data: Any, success: bool = True, message: str | None = None) -> dict[str, Any]  # Line 277
def format_error_response(error: Exception, error_code: str | None = None) -> dict[str, Any]  # Line 303
def format_success_response(data: Any, message: str = 'Operation completed successfully') -> dict[str, Any]  # Line 335
def format_paginated_response(data: list[Any], page: int, page_size: int, total: int) -> dict[str, Any]  # Line 351
def format_log_entry(level: str, message: str, **kwargs: Any) -> dict[str, Any]  # Line 390
def format_correlation_id(correlation_id: str) -> str  # Line 412
def format_structured_log(level: str, message: str, correlation_id: str | None = None, **kwargs: Any) -> str  # Line 433
def format_performance_log(function_name: str, execution_time_ms: float, success: bool, **kwargs: Any) -> dict[str, Any]  # Line 456
def format_ohlcv_data(ohlcv_data: list[dict[str, Any]]) -> list[dict[str, Any]]  # Line 486
def format_indicator_data(...) -> dict[str, Any]  # Line 541
def format_chart_data(...) -> dict[str, Any]  # Line 587
def format_performance_report(performance_data: dict[str, Any]) -> dict[str, Any]  # Line 622
def format_risk_report(risk_data: dict[str, Any]) -> dict[str, Any]  # Line 650
def format_trade_report(trades: list[dict[str, Any]]) -> dict[str, Any]  # Line 676
def format_csv_data(data: list[dict[str, Any]] | None = None, headers: list[str] | None = None) -> str  # Line 746
def format_excel_data(data: list[dict[str, Any]], sheet_name: str = 'Data') -> bytes  # Line 786
def format_json_data(data: Any, pretty: bool = True) -> str  # Line 818
def export_to_file(data: Any, file_path: str, format_type: str = 'json') -> None  # Line 841
```

### File: gpu_utils.py

**Key Imports:**
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`

#### Class: `GPUManager`

**Purpose**: Manages GPU resources and provides utilities for GPU acceleration

```python
class GPUManager:
    def __init__(self) -> None  # Line 66
    def _initialize_gpu_settings(self) -> None  # Line 73
    def _check_gpu_availability(self) -> bool  # Line 108
    def _get_device_info(self) -> dict[str, Any]  # Line 136
    def _log_gpu_status(self) -> None  # Line 174
    def is_available(self) -> bool  # Line 188
    def get_memory_info(self, device_id: int = 0) -> dict[str, float]  # Line 192
    def clear_cache(self) -> None  # Line 217
    def to_gpu(self, data: Any, dtype: str | None = None) -> Any  # Line 234
    def _convert_to_gpu(self, data: Any, dtype: str | None = None) -> Any  # Line 245
    def _convert_numpy_to_gpu(self, data: NDArray[np.float64], dtype: str | None = None) -> Any  # Line 261
    def _convert_dataframe_to_gpu(self, data: pd.DataFrame) -> Any  # Line 272
    def _is_tensor_like(self, data: Any) -> bool  # Line 280
    def _convert_tensor_to_gpu(self, data: Any) -> Any  # Line 290
    def to_cpu(self, data: Any) -> Any  # Line 296
    def accelerate_computation(self, func: Any, *args: Any, **kwargs: Any) -> Any  # Line 320
```

#### Functions:

```python
def get_optimal_batch_size(...) -> int  # Line 359
def _get_gpu_manager_service(gpu_manager: GPUManager | None = None) -> GPUManager  # Line 389
def parallel_apply(...) -> pd.DataFrame  # Line 419
def gpu_accelerated_correlation(data: NDArray[np.float64], gpu_manager: 'GPUManager | None' = None) -> NDArray[np.float64]  # Line 444
def gpu_accelerated_rolling_window(...) -> NDArray[np.float64]  # Line 489
def setup_gpu_logging(gpu_manager: GPUManager | None = None) -> Callable[[], None] | None  # Line 542
```

### File: interfaces.py

**Key Imports:**
- `from src.core.base import BaseService`

#### Class: `ValidationServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for validation services

```python
class ValidationServiceInterface(Protocol):
    async def validate_order(self, order_data: dict[str, Any], context: 'ValidationContext | None' = None) -> 'ValidationResult'  # Line 18
    async def validate_risk_parameters(self, risk_data: dict[str, Any], context: 'ValidationContext | None' = None) -> 'ValidationResult'  # Line 25
    async def validate_strategy_config(self, strategy_data: dict[str, Any], context: 'ValidationContext | None' = None) -> 'ValidationResult'  # Line 32
    async def validate_market_data(self, market_data: dict[str, Any], context: 'ValidationContext | None' = None) -> 'ValidationResult'  # Line 39
    async def validate_batch(self, ...) -> dict[str, 'ValidationResult']  # Line 46
```

#### Class: `GPUInterface`

**Inherits**: Protocol
**Purpose**: Interface for GPU management services

```python
class GPUInterface(Protocol):
    def is_available(self) -> bool  # Line 58
    def get_memory_info(self) -> dict[str, Any]  # Line 63
```

#### Class: `PrecisionInterface`

**Inherits**: Protocol
**Purpose**: Interface for precision tracking services

```python
class PrecisionInterface(Protocol):
    def track_operation(self, operation: str, input_precision: int, output_precision: int) -> None  # Line 73
    def get_precision_stats(self) -> dict[str, Any]  # Line 78
```

#### Class: `DataFlowInterface`

**Inherits**: Protocol
**Purpose**: Interface for data flow validation services

```python
class DataFlowInterface(Protocol):
    def validate_data_integrity(self, data: Any) -> bool  # Line 88
    def get_validation_report(self) -> dict[str, Any]  # Line 93
```

#### Class: `CalculatorInterface`

**Inherits**: Protocol
**Purpose**: Interface for financial calculation services

```python
class CalculatorInterface(Protocol):
    def calculate_compound_return(self, principal: Decimal, rate: Decimal, periods: int) -> Decimal  # Line 103
    def calculate_sharpe_ratio(self, returns: list[Decimal], risk_free_rate: Decimal) -> Decimal  # Line 108
```

#### Class: `BaseUtilityService`

**Inherits**: BaseService
**Purpose**: Base class for utility services that inherits from core BaseService

```python
class BaseUtilityService(BaseService):
    def __init__(self, name: str | None = None, config: dict[str, Any] | None = None)  # Line 116
    async def initialize(self) -> None  # Line 122
    async def shutdown(self) -> None  # Line 127
```

### File: math_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.utils.decimal_utils import ZERO`
- `from src.utils.decimal_utils import safe_divide`
- `from src.utils.decimal_utils import to_decimal`

#### Functions:

```python
def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> Decimal  # Line 9
def calculate_sharpe_ratio(returns: list[Decimal], risk_free_rate: Decimal = Any, frequency: str = 'daily') -> Decimal  # Line 33
def calculate_max_drawdown(equity_curve: list[Decimal]) -> tuple[Decimal, int, int]  # Line 91
def calculate_var(returns: list[Decimal], confidence_level: Decimal = Any) -> Decimal  # Line 142
def calculate_volatility(returns: list[Decimal], window: int | None = None) -> Decimal  # Line 182
def calculate_correlation(series1: list[Decimal], series2: list[Decimal]) -> Decimal  # Line 227
def calculate_beta(asset_returns: list[Decimal], market_returns: list[Decimal]) -> Decimal  # Line 288
def calculate_sortino_ratio(...) -> Decimal  # Line 333
def safe_min(*args: Decimal) -> Decimal  # Line 388
def safe_max(*args: Decimal) -> Decimal  # Line 420
def safe_percentage(value: Decimal, total: Decimal, default: Decimal = ZERO) -> Decimal  # Line 452
```

### File: messaging_patterns.py

**Key Imports:**
- `from src.core.base.events import BaseEventEmitter`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `MessagePattern`

**Inherits**: Enum
**Purpose**: Standardized message patterns

```python
class MessagePattern(Enum):
```

#### Class: `MessageType`

**Inherits**: Enum
**Purpose**: Standard message types

```python
class MessageType(Enum):
```

#### Class: `StandardMessage`

**Purpose**: Standardized message format across all communication patterns

```python
class StandardMessage:
    def __init__(self, ...)  # Line 46
    def to_dict(self) -> dict[str, Any]  # Line 65
    def from_dict(cls, data: dict[str, Any]) -> 'StandardMessage'  # Line 79
```

#### Class: `MessageHandler`

**Inherits**: ABC
**Purpose**: Abstract base class for consistent message handling

```python
class MessageHandler(ABC):
    async def handle(self, message: StandardMessage) -> StandardMessage | None  # Line 96
```

#### Class: `ErrorPropagationMixin`

**Purpose**: Mixin for consistent error propagation patterns across all modules with enhanced boundary validation

```python
class ErrorPropagationMixin:
    def propagate_validation_error(self, error: Exception, context: str) -> None  # Line 104
    def propagate_database_error(self, error: Exception, context: str) -> None  # Line 133
    def propagate_service_error(self, error: Exception, context: str) -> None  # Line 156
    def propagate_monitoring_error(self, error: Exception, context: str) -> None  # Line 193
```

#### Class: `BoundaryValidator`

**Purpose**: Validator for module boundary consistency

```python
class BoundaryValidator:
    def validate_database_entity(entity_dict: dict[str, Any], operation: str) -> None  # Line 233
    def validate_database_to_error_boundary(data: dict[str, Any]) -> None  # Line 257
    def validate_monitoring_to_error_boundary(data: dict[str, Any]) -> None  # Line 299
    def validate_error_to_monitoring_boundary(data: dict[str, Any]) -> None  # Line 389
    def validate_web_interface_to_error_boundary(data: dict[str, Any]) -> None  # Line 480
    def validate_risk_to_state_boundary(data: dict[str, Any]) -> None  # Line 550
```

#### Class: `ProcessingParadigmAligner`

**Purpose**: Aligns processing paradigms between batch and stream processing

```python
class ProcessingParadigmAligner:
    def create_batch_from_stream(stream_items: list[dict[str, Any]]) -> dict[str, Any]  # Line 618
    def create_stream_from_batch(batch_data: dict[str, Any]) -> list[dict[str, Any]]  # Line 629
    def align_processing_modes(source_mode: str, target_mode: str, data: dict[str, Any]) -> dict[str, Any]  # Line 645
```

#### Class: `MessagingCoordinator`

**Purpose**: Coordinates messaging patterns to prevent conflicts between pub/sub and req/reply

```python
class MessagingCoordinator:
    def __init__(self, ...)  # Line 715
    def register_handler(self, pattern: MessagePattern, handler: MessageHandler) -> None  # Line 727
    async def publish(self, ...) -> None  # Line 736
    async def request(self, ...) -> Any  # Line 769
    async def reply(self, original_message: StandardMessage, response_data: Any) -> None  # Line 814
    async def stream_start(self, ...) -> None  # Line 833
    async def batch_process(self, ...) -> None  # Line 860
    def _apply_data_transformation(self, data: Any) -> Any  # Line 890
    async def _route_message(self, message: StandardMessage) -> None  # Line 944
```

#### Class: `DataTransformationHandler`

**Inherits**: MessageHandler
**Purpose**: Handler for consistent data transformation patterns

```python
class DataTransformationHandler(MessageHandler):
    def __init__(self, transform_func: Callable[[Any], Any] | None = None)  # Line 1009
    async def handle(self, message: StandardMessage) -> StandardMessage | None  # Line 1012
    async def _transform_financial_data(self, data: dict[str, Any]) -> None  # Line 1029
```

#### Functions:

```python
def get_messaging_coordinator(coordinator: MessagingCoordinator | None = None) -> MessagingCoordinator  # Line 1072
```

### File: ml_cache.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `TTLCache`

**Inherits**: Generic[T]
**Purpose**: Time-To-Live cache implementation

```python
class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000)  # Line 23
    async def get(self, key: str) -> T | None  # Line 36
    async def set(self, key: str, value: T) -> None  # Line 56
    async def delete(self, key: str) -> bool  # Line 73
    async def clear(self) -> int  # Line 89
    async def size(self) -> int  # Line 101
    async def cleanup_expired(self) -> int  # Line 105
```

#### Class: `ModelCache`

**Inherits**: TTLCache[Any]
**Purpose**: Specialized cache for ML models

```python
class ModelCache(TTLCache[Any]):
    def __init__(self, ttl_hours: int = 24, max_models: int = 100)  # Line 127
    async def get_model(self, model_id: str) -> Any | None  # Line 137
    async def cache_model(self, model_id: str, model: Any) -> None  # Line 141
    async def remove_model(self, model_id: str) -> bool  # Line 145
```

#### Class: `PredictionCache`

**Inherits**: TTLCache[dict[str, Any]]
**Purpose**: Specialized cache for ML predictions

```python
class PredictionCache(TTLCache[dict[str, Any]]):
    def __init__(self, ttl_minutes: int = 5, max_predictions: int = 10000)  # Line 153
    async def get_prediction(self, cache_key: str) -> dict[str, Any] | None  # Line 163
    async def cache_prediction(self, cache_key: str, prediction: dict[str, Any]) -> None  # Line 167
```

#### Class: `FeatureCache`

**Inherits**: TTLCache[dict[str, Any]]
**Purpose**: Specialized cache for feature sets

```python
class FeatureCache(TTLCache[dict[str, Any]]):
    def __init__(self, ttl_hours: int = 4, max_feature_sets: int = 1000)  # Line 175
    async def get_features(self, cache_key: str) -> dict[str, Any] | None  # Line 185
    async def cache_features(self, cache_key: str, features: dict[str, Any]) -> None  # Line 189
```

#### Class: `CacheManager`

**Purpose**: Centralized cache manager for ML operations

```python
class CacheManager:
    def __init__(self, config: dict[str, Any] | None = None)  # Line 276
    async def start(self) -> None  # Line 305
    async def stop(self) -> None  # Line 311
    async def clear_all(self) -> dict[str, int]  # Line 321
    async def get_cache_stats(self) -> dict[str, Any]  # Line 337
    async def _background_cleanup(self) -> None  # Line 353
```

#### Functions:

```python
def generate_cache_key(*args: Any) -> str  # Line 194
def generate_model_cache_key(model_id: str, version: str | None = None) -> str  # Line 224
def generate_prediction_cache_key(model_id: str, features_hash: str, return_probabilities: bool = False) -> str  # Line 240
def generate_feature_cache_key(symbol: str, feature_types: list[str], data_hash: str) -> str  # Line 257
def get_cache_manager() -> CacheManager  # Line 382
async def init_cache_manager(config: dict[str, Any] | None = None) -> CacheManager  # Line 390
```

### File: ml_config.py

**Key Imports:**
- `from src.utils.constants import ML_MODEL_CONSTANTS`

#### Class: `BaseMLConfig`

**Inherits**: BaseModel
**Purpose**: Base configuration for ML components

```python
class BaseMLConfig(BaseModel):
```

#### Class: `MLModelConfig`

**Inherits**: BaseMLConfig
**Purpose**: Configuration for ML models

```python
class MLModelConfig(BaseMLConfig):
```

#### Class: `MLServiceConfig`

**Inherits**: BaseMLConfig
**Purpose**: Configuration for ML service

```python
class MLServiceConfig(BaseMLConfig):
```

#### Class: `ModelManagerConfig`

**Inherits**: BaseMLConfig
**Purpose**: Configuration for model manager service

```python
class ModelManagerConfig(BaseMLConfig):
```

#### Class: `PredictorConfig`

**Inherits**: MLModelConfig
**Purpose**: Configuration for predictor models

```python
class PredictorConfig(MLModelConfig):
```

#### Class: `ClassifierConfig`

**Inherits**: MLModelConfig
**Purpose**: Configuration for classifier models

```python
class ClassifierConfig(MLModelConfig):
```

#### Class: `MLCacheConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for ML caching

```python
class MLCacheConfig(BaseModel):
```

#### Functions:

```python
def create_ml_service_config(config_dict: dict | None = None) -> MLServiceConfig  # Line 149
def create_model_manager_config(config_dict: dict | None = None) -> ModelManagerConfig  # Line 155
def create_predictor_config(config_dict: dict | None = None) -> PredictorConfig  # Line 161
def create_classifier_config(config_dict: dict | None = None) -> ClassifierConfig  # Line 167
def create_cache_config(config_dict: dict | None = None) -> MLCacheConfig  # Line 173
```

### File: ml_data_transforms.py

**Key Imports:**
- `from src.utils.decimal_utils import to_decimal`
- `from src.core.logging import get_logger`

#### Functions:

```python
def transform_market_data_to_decimal(data: dict[str, Any] | pd.DataFrame) -> dict[str, Any] | pd.DataFrame  # Line 18
def convert_pydantic_to_dict_with_decimals(data: Any) -> dict[str, Any]  # Line 57
def prepare_dataframe_from_market_data(market_data: dict[str, Any] | Any, apply_decimal_transform: bool = True) -> pd.DataFrame  # Line 74
def align_training_data_lengths(X: pd.DataFrame, y: pd.Series, model_name: str = 'Unknown') -> tuple[pd.DataFrame, pd.Series]  # Line 106
def create_returns_series(prices: pd.Series, horizon: int = 1, return_type: str = 'simple') -> pd.Series  # Line 128
def batch_transform_requests_to_aligned_format(requests: list[Any]) -> list[dict[str, Any]]  # Line 172
```

### File: ml_metrics.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.utils.decimal_utils import to_decimal`

#### Functions:

```python
def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]  # Line 27
def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]  # Line 82
def calculate_volatility_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]  # Line 112
def calculate_volatility_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.2) -> float  # Line 144
def calculate_directional_volatility_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float  # Line 168
def calculate_volatility_regime_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float  # Line 199
def calculate_trading_metrics(y_true: np.ndarray, y_pred: np.ndarray, transaction_cost: float = 0.001) -> dict[str, Any]  # Line 236
def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float  # Line 333
```

### File: ml_validation.py

**Key Imports:**
- `from src.core.exceptions import DataValidationError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Functions:

```python
def validate_features(X: pd.DataFrame, model_name: str = 'Unknown') -> pd.DataFrame  # Line 19
def validate_targets(y: pd.Series, model_name: str = 'Unknown') -> pd.Series  # Line 90
def validate_training_data(X: pd.DataFrame, y: pd.Series, model_name: str = 'Unknown') -> tuple[pd.DataFrame, pd.Series]  # Line 145
def validate_market_data(data: pd.DataFrame, required_columns: list[str] | None = None) -> pd.DataFrame  # Line 178
def validate_prediction_data(X: pd.DataFrame, model_name: str = 'Unknown') -> pd.DataFrame  # Line 223
def validate_direction_threshold(threshold: float, model_name: str = 'Unknown') -> float  # Line 243
def validate_prediction_horizon(horizon: int, model_name: str = 'Unknown') -> int  # Line 266
def validate_algorithm_choice(algorithm: str, allowed_algorithms: list[str], model_name: str = 'Unknown') -> str  # Line 292
def validate_class_weights(class_weights: Any, model_name: str = 'Unknown') -> Any  # Line 315
def check_data_quality(...) -> dict[str, Any]  # Line 346
def validate_model_hyperparameters(params: dict[str, Any], model_type: str, model_name: str = 'Unknown') -> dict[str, Any]  # Line 407
```

### File: monitoring_helpers.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.error_handling.context import ErrorContext`

#### Class: `FinancialPrecisionWarning`

**Inherits**: UserWarning
**Purpose**: Warning raised when precision loss is detected in financial calculations

```python
class FinancialPrecisionWarning(UserWarning):
```

#### Class: `HTTPSessionManager`

**Purpose**: Shared HTTP session manager for monitoring components

```python
class HTTPSessionManager:
    def __init__(self) -> None  # Line 50
    async def get_session(self, key: str = 'default', **session_kwargs) -> aiohttp.ClientSession  # Line 54
    async def close_all(self) -> None  # Line 78
    async def _safe_session_close(self, key: str, session) -> None  # Line 105
```

#### Class: `AsyncTaskManager`

**Purpose**: Manages async task lifecycle with proper cleanup

```python
class AsyncTaskManager:
    def __init__(self, component_name: str)  # Line 332
    def create_task(self, coro, name: str | None = None) -> asyncio.Task  # Line 337
    def _task_done_callback(self, task: asyncio.Task) -> None  # Line 344
    async def shutdown(self, timeout: float = 5.0) -> None  # Line 354
    def shutdown_requested(self) -> bool  # Line 388
```

#### Class: `MetricValueProcessor`

**Purpose**: Processes and validates metric values consistently

```python
class MetricValueProcessor:
    def process_financial_value(value, ...) -> float  # Line 493
    def process_latency_value(value: float, metric_name: str) -> float  # Line 510
```

#### Class: `SystemMetricsCollector`

**Purpose**: Shared system metrics collection to eliminate duplication

```python
class SystemMetricsCollector:
    async def collect_system_metrics() -> dict[str, Any]  # Line 526
```

#### Functions:

```python
async def get_http_session(...) -> aiohttp.ClientSession  # Line 123
async def cleanup_http_sessions(session_manager: HTTPSessionManager | None = None)  # Line 150
def generate_correlation_id() -> str  # Line 163
def generate_fingerprint(data: dict[str, Any]) -> str  # Line 168
async def create_error_context(...) -> ErrorContext  # Line 174
def validate_cross_module_data_consistency(source_module: str, target_module: str, data: dict[str, Any]) -> dict[str, Any]  # Line 206
async def handle_error_with_fallback(error: Exception, error_handler: Any | None, context: ErrorContext) -> bool  # Line 257
def validate_monitoring_parameter(...) -> None  # Line 281
async def http_request_with_retry(...)  # Line 394
def safe_duration_parse(duration_str: str) -> int  # Line 434
def log_unusual_values(value: float, threshold: float, metric_name: str, unit: str = '')  # Line 482
def safe_decimal_to_float(...) -> float  # Line 657
def validate_financial_range(...) -> None  # Line 724
def convert_financial_batch(...) -> dict[str, float]  # Line 763
```

### File: network_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Functions:

```python
async def check_connection(host: str, port: int, timeout: float = 5.0) -> bool  # Line 14
async def measure_latency(host: str, port: int, timeout: float = 5.0) -> float  # Line 51
async def ping_host(host: str, count: int = 3, port: int = 80) -> dict[str, Any]  # Line 100
async def check_multiple_hosts(hosts: list[tuple[str, int]], timeout: float = 5.0) -> dict[str, bool]  # Line 141
def parse_url(url: str) -> dict[str, Any]  # Line 206
async def wait_for_service(host: str, port: int, max_wait: float = 30.0, check_interval: float = 1.0) -> bool  # Line 253
```

### File: pipeline_utilities.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `PipelineStage`

**Inherits**: Enum
**Purpose**: Common pipeline stage enumeration

```python
class PipelineStage(Enum):
```

#### Class: `ProcessingMode`

**Inherits**: Enum
**Purpose**: Data processing mode enumeration

```python
class ProcessingMode(Enum):
```

#### Class: `DataQuality`

**Inherits**: Enum
**Purpose**: Data quality levels

```python
class DataQuality(Enum):
```

#### Class: `PipelineAction`

**Inherits**: Enum
**Purpose**: Pipeline action enumeration

```python
class PipelineAction(Enum):
```

#### Class: `PipelineMetrics`

**Purpose**: Common pipeline processing metrics

```python
class PipelineMetrics:
    def calculate_success_rate(self) -> float  # Line 84
    def calculate_failure_rate(self) -> float  # Line 90
    def update_processing_time(self, processing_time_ms: float) -> None  # Line 96
    def calculate_throughput(self, time_window_seconds: float = 1.0) -> float  # Line 107
```

#### Class: `PipelineRecord`

**Purpose**: Common pipeline record structure

```python
class PipelineRecord:
    def add_error(self, error: str, stage: PipelineStage | None = None) -> None  # Line 134
    def add_warning(self, warning: str, stage: PipelineStage | None = None) -> None  # Line 140
    def has_errors(self) -> bool  # Line 146
    def has_warnings(self) -> bool  # Line 150
```

#### Class: `PipelineStageProcessor`

**Inherits**: Protocol
**Purpose**: Protocol for pipeline stage processors

```python
class PipelineStageProcessor(Protocol):
    async def process(self, record: PipelineRecord) -> PipelineRecord  # Line 158
    def get_stage_name(self) -> str  # Line 162
```

#### Class: `PipelineUtils`

**Purpose**: Shared pipeline utility functions

```python
class PipelineUtils:
    def validate_pipeline_config(config: dict[str, Any]) -> list[str]  # Line 171
    def calculate_data_quality_score(record: PipelineRecord, error_weight: float = 0.5, warning_weight: float = 0.2) -> float  # Line 210
    def determine_pipeline_action(quality_score, ...) -> PipelineAction  # Line 236
    def create_processing_summary(metrics: PipelineMetrics, duration_seconds: float) -> dict[str, Any]  # Line 276
    def log_pipeline_summary(pipeline_name: str, summary: dict[str, Any], logger_instance: Any = None) -> None  # Line 307
```

### File: position_sizing.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types.risk import PositionSizeMethod`
- `from src.core.types.trading import Signal`
- `from src.utils.decimal_utils import ONE`
- `from src.utils.decimal_utils import ZERO`

#### Class: `PositionSizingAlgorithm`

**Inherits**: ABC
**Purpose**: Abstract base class for position sizing algorithms

```python
class PositionSizingAlgorithm(ABC):
    def calculate_size(self, ...) -> Decimal  # Line 26
    def validate_inputs(self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal) -> bool  # Line 32
```

#### Class: `FixedPercentageAlgorithm`

**Inherits**: PositionSizingAlgorithm
**Purpose**: Fixed percentage position sizing algorithm

```python
class FixedPercentageAlgorithm(PositionSizingAlgorithm):
    def calculate_size(self, ...) -> Decimal  # Line 54
```

#### Class: `KellyCriterionAlgorithm`

**Inherits**: PositionSizingAlgorithm
**Purpose**: Kelly Criterion position sizing algorithm

```python
class KellyCriterionAlgorithm(PositionSizingAlgorithm):
    def __init__(self, kelly_fraction: Decimal = Any)  # Line 74
    def calculate_size(self, ...) -> Decimal  # Line 78
    def _fallback_to_fixed(self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal) -> Decimal  # Line 120
```

#### Class: `VolatilityAdjustedAlgorithm`

**Inherits**: PositionSizingAlgorithm
**Purpose**: Volatility-adjusted position sizing algorithm

```python
class VolatilityAdjustedAlgorithm(PositionSizingAlgorithm):
    def calculate_size(self, ...) -> Decimal  # Line 131
    def _fallback_to_fixed(self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal) -> Decimal  # Line 169
```

#### Class: `ConfidenceWeightedAlgorithm`

**Inherits**: PositionSizingAlgorithm
**Purpose**: Confidence-weighted position sizing for ML signals

```python
class ConfidenceWeightedAlgorithm(PositionSizingAlgorithm):
    def calculate_size(self, ...) -> Decimal  # Line 180
```

#### Class: `ATRBasedAlgorithm`

**Inherits**: PositionSizingAlgorithm
**Purpose**: Average True Range based position sizing

```python
class ATRBasedAlgorithm(PositionSizingAlgorithm):
    def calculate_size(self, ...) -> Decimal  # Line 213
    def _fallback_to_fixed(self, signal: Signal, portfolio_value: Decimal, risk_per_trade: Decimal) -> Decimal  # Line 251
```

#### Functions:

```python
def get_signal_confidence(signal: Signal) -> Decimal  # Line 262
def validate_position_size(...) -> tuple[bool, Decimal]  # Line 286
def calculate_position_size(...) -> Decimal  # Line 327
def update_position_history(...) -> None  # Line 385
def calculate_position_metrics(symbol: str, position_history: dict[str, list[Decimal]]) -> dict[str, Any]  # Line 413
```

### File: risk_calculations.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types.risk import RiskLevel`
- `from src.utils.decimal_utils import ONE`
- `from src.utils.decimal_utils import ZERO`
- `from src.utils.decimal_utils import safe_divide`

#### Functions:

```python
def calculate_var(returns: list[Decimal], confidence_level: Decimal = Any, time_horizon: int = 1) -> Decimal  # Line 21
def calculate_expected_shortfall(returns: list[Decimal], confidence_level: Decimal = Any) -> Decimal  # Line 58
def calculate_sharpe_ratio(returns: list[Decimal], risk_free_rate: Decimal = Any) -> Decimal | None  # Line 94
def calculate_max_drawdown(values: list[Decimal]) -> tuple[Decimal, int, int]  # Line 132
def calculate_current_drawdown(current_value: Decimal, historical_values: list[Decimal]) -> Decimal  # Line 165
def calculate_portfolio_value(positions, market_data) -> Decimal  # Line 192
def calculate_sortino_ratio(...) -> Decimal  # Line 222
def calculate_calmar_ratio(returns: list[Decimal], period_years: Decimal = ONE) -> Decimal  # Line 261
def determine_risk_level(...) -> RiskLevel  # Line 292
def update_returns_history(...) -> list[Decimal]  # Line 357
def validate_risk_inputs(...) -> bool  # Line 399
```

### File: risk_monitoring.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types.risk import RiskAlert`
- `from src.core.types.risk import RiskLevel`
- `from src.core.types.risk import RiskMetrics`
- `from src.utils.decimal_utils import ZERO`

#### Class: `RiskEventType`

**Inherits**: Enum
**Purpose**: Standardized risk event types

```python
class RiskEventType(Enum):
```

#### Class: `RiskEventSeverity`

**Inherits**: Enum
**Purpose**: Risk event severity levels

```python
class RiskEventSeverity(Enum):
```

#### Class: `RiskEvent`

**Purpose**: Standardized risk event structure

```python
class RiskEvent:
    def __init__(self, ...)  # Line 50
    def to_dict(self) -> dict[str, Any]  # Line 74
```

#### Class: `RiskObserver`

**Inherits**: ABC
**Purpose**: Abstract base for risk event observers

```python
class RiskObserver(ABC):
    async def handle_event(self, event: RiskEvent) -> None  # Line 89
    def get_observer_id(self) -> str  # Line 99
```

#### Class: `LoggingRiskObserver`

**Inherits**: RiskObserver
**Purpose**: Observer that logs risk events

```python
class LoggingRiskObserver(RiskObserver):
    def __init__(self, observer_id: str = 'logging_observer') -> None  # Line 107
    async def handle_event(self, event: RiskEvent) -> None  # Line 117
    def get_observer_id(self) -> str  # Line 134
```

#### Class: `AlertingRiskObserver`

**Inherits**: RiskObserver
**Purpose**: Observer that creates and stores risk alerts

```python
class AlertingRiskObserver(RiskObserver):
    def __init__(self, ...)  # Line 142
    async def handle_event(self, event: RiskEvent) -> None  # Line 159
    def get_observer_id(self) -> str  # Line 192
    def get_recent_alerts(self, limit: int = 50) -> list[RiskAlert]  # Line 196
```

#### Class: `CircuitBreakerObserver`

**Inherits**: RiskObserver
**Purpose**: Observer that implements circuit breaker functionality

```python
class CircuitBreakerObserver(RiskObserver):
    def __init__(self, ...)  # Line 204
    async def handle_event(self, event: RiskEvent) -> None  # Line 233
    def _should_reset_circuit(self) -> bool  # Line 251
    async def _trigger_circuit_breaker(self, event: RiskEvent) -> None  # Line 259
    async def _reset_circuit_breaker(self) -> None  # Line 280
    def get_observer_id(self) -> str  # Line 286
    def get_status(self) -> dict[str, Any]  # Line 290
```

#### Class: `UnifiedRiskMonitor`

**Purpose**: Centralized risk monitor that unifies all monitoring patterns

```python
class UnifiedRiskMonitor:
    def __init__(self) -> None  # Line 307
    def _set_default_thresholds(self) -> None  # Line 324
    def add_observer(self, observer: RiskObserver) -> None  # Line 336
    def remove_observer(self, observer_id: str) -> None  # Line 347
    async def notify_observers(self, event: RiskEvent) -> None  # Line 358
    async def monitor_metrics(self, metrics: RiskMetrics) -> None  # Line 371
    async def monitor_portfolio(self, portfolio_data: dict[str, Any]) -> None  # Line 394
    async def _check_drawdown(self, metrics: RiskMetrics) -> None  # Line 411
    async def _check_var(self, metrics: RiskMetrics) -> None  # Line 425
    async def _check_correlation(self, metrics: RiskMetrics) -> None  # Line 441
    async def _check_risk_level_change(self, metrics: RiskMetrics) -> None  # Line 456
    async def _check_volatility_spike(self, metrics: RiskMetrics) -> None  # Line 476
    async def _check_position_limits(self, portfolio_data: dict[str, Any]) -> None  # Line 500
    async def _check_daily_loss(self, portfolio_data: dict[str, Any]) -> None  # Line 512
    async def _check_liquidity(self, portfolio_data: dict[str, Any]) -> None  # Line 528
    def set_threshold(self, key: str, value: Decimal) -> None  # Line 542
    def get_thresholds(self) -> dict[str, Decimal]  # Line 547
    def get_observer_status(self) -> dict[str, Any]  # Line 551
    async def trigger_emergency_stop(self, reason: str) -> None  # Line 561
```

#### Functions:

```python
def get_unified_risk_monitor() -> UnifiedRiskMonitor  # Line 576
```

### File: risk_validation.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types.risk import RiskLevel`
- `from src.core.types.risk import RiskLimits`
- `from src.core.types.trading import OrderRequest`
- `from src.core.types.trading import Position`

#### Class: `UnifiedRiskValidator`

**Purpose**: Centralized risk validator that eliminates validation code duplication

```python
class UnifiedRiskValidator:
    def __init__(self, risk_limits: RiskLimits | None = None)  # Line 26
    def _get_default_limits(self) -> RiskLimits  # Line 36
    def validate_signal(self, ...) -> tuple[bool, str]  # Line 49
    def validate_order(self, ...) -> tuple[bool, str]  # Line 99
    def validate_position(self, position: Position, portfolio_value: Decimal) -> tuple[bool, str]  # Line 156
    def validate_portfolio(self, portfolio_data: dict[str, Any]) -> tuple[bool, str]  # Line 200
    def validate_risk_metrics(self, var_1d: Decimal, current_drawdown: Decimal, portfolio_value: Decimal) -> tuple[bool, str]  # Line 242
    def _validate_signal_structure(self, signal: Signal) -> bool  # Line 294
    def _validate_order_structure(self, order: OrderRequest) -> bool  # Line 310
    def _validate_position_structure(self, position: Position) -> bool  # Line 330
    def _get_signal_confidence(self, signal: Signal) -> Decimal  # Line 346
    def _get_min_confidence_for_risk_level(self, risk_level: RiskLevel | None) -> Decimal  # Line 359
    def _validate_signal_direction(self, signal: Signal) -> bool  # Line 370
    def _validate_stop_loss_requirement(self, signal: Signal) -> bool  # Line 378
    def _calculate_order_value(self, order: OrderRequest) -> Decimal  # Line 387
    def update_limits(self, new_limits: RiskLimits) -> None  # Line 399
```

#### Functions:

```python
def validate_financial_inputs(**kwargs) -> tuple[bool, str]  # Line 410
def check_position_limits(...) -> tuple[bool, str]  # Line 454
def validate_correlation_risk(...) -> tuple[bool, str]  # Line 485
```

### File: security_types.py

#### Class: `SeverityLevel`

**Inherits**: Enum
**Purpose**: Generic severity levels

```python
class SeverityLevel(Enum):
```

#### Class: `LogCategory`

**Inherits**: Enum
**Purpose**: Log categories

```python
class LogCategory(Enum):
```

#### Class: `LogLevel`

**Inherits**: Enum
**Purpose**: Log levels

```python
class LogLevel(Enum):
```

#### Class: `InformationLevel`

**Inherits**: Enum
**Purpose**: Information levels for security context

```python
class InformationLevel(Enum):
```

#### Class: `ThreatType`

**Inherits**: Enum
**Purpose**: Security threat types

```python
class ThreatType(Enum):
```

#### Class: `ReportType`

**Inherits**: Enum
**Purpose**: Report types

```python
class ReportType(Enum):
```

#### Class: `ReportingChannel`

**Inherits**: Enum
**Purpose**: Reporting channels

```python
class ReportingChannel(Enum):
```

#### Class: `UserRole`

**Inherits**: Enum
**Purpose**: User roles for access control

```python
class UserRole(Enum):
```

#### Class: `SecurityContext`

**Purpose**: Common security context

```python
class SecurityContext:
    def has_admin_access(self) -> bool  # Line 117
```

#### Class: `ErrorAlert`

**Purpose**: Generic error alert

```python
class ErrorAlert:
```

#### Class: `SecureLogEntry`

**Purpose**: Secure log entry

```python
class SecureLogEntry:
```

#### Class: `LoggingConfig`

**Purpose**: Logging configuration

```python
class LoggingConfig:
```

#### Class: `ReportingConfig`

**Purpose**: Reporting configuration

```python
class ReportingConfig:
```

#### Class: `ReportingRule`

**Purpose**: Reporting rule configuration

```python
class ReportingRule:
```

#### Class: `ReportingMetrics`

**Purpose**: Reporting metrics

```python
class ReportingMetrics:
```

### File: serialization_utilities.py

#### Class: `HashGenerator`

**Purpose**: Utility class for generating consistent hashes across the system

```python
class HashGenerator:
    def generate_backtest_hash(request: Any) -> str  # Line 153
    def generate_data_hash(data: dict[str, Any]) -> str  # Line 174
```

#### Functions:

```python
def serialize_state_data(data: dict[str, Any], compress: bool = False, compression_threshold: int = 1024) -> bytes  # Line 14
def deserialize_state_data(data: bytes, is_compressed: bool = False) -> dict[str, Any]  # Line 51
def calculate_compression_ratio(original_size: int, compressed_size: int) -> float  # Line 84
def should_compress_data(data_size: int, threshold: int = 1024) -> bool  # Line 100
def serialize_with_metadata(data: dict[str, Any], metadata: dict[str, Any], compress: bool = False) -> bytes  # Line 114
def deserialize_with_metadata(data: bytes, is_compressed: bool = False) -> tuple[dict[str, Any], dict[str, Any]]  # Line 132
```

### File: service_registry.py

**Key Imports:**
- `from src.core.dependency_injection import injector`
- `from src.core.exceptions import ServiceError`

#### Functions:

```python
def register_util_services() -> None  # Line 25
```

### File: state_utils.py

**Key Imports:**
- `from src.core.exceptions import StateError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `StateOperationLock`

**Purpose**: Centralized lock manager for state operations to eliminate duplication

```python
class StateOperationLock:
    def __init__(self)  # Line 174
    def get_lock(self, key: str) -> asyncio.Lock  # Line 178
    def _cleanup_unused_locks(self) -> None  # Line 189
    async def with_lock(self, key: str, operation: callable, *args, **kwargs)  # Line 203
```

#### Class: `StateCache`

**Purpose**: Centralized cache manager for state operations to eliminate duplication

```python
class StateCache:
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300)  # Line 232
    def get(self, key: str) -> Any | None  # Line 237
    def set(self, key: str, value: Any) -> None  # Line 251
    def pop(self, key: str, default: Any = None) -> Any  # Line 263
    def clear(self) -> None  # Line 270
    def size(self) -> int  # Line 274
    def keys(self) -> list[str]  # Line 278
    def _cleanup_expired(self) -> int  # Line 282
    def _remove_oldest(self) -> None  # Line 296
    def get_stats(self) -> dict[str, int | float]  # Line 308
```

#### Functions:

```python
def create_state_metadata(...) -> 'StateMetadata'  # Line 30
def handle_state_error(...) -> None  # Line 98
def log_state_operation(...) -> None  # Line 141
def get_state_lock(key: str) -> asyncio.Lock  # Line 217
async def with_state_lock(key: str, operation: callable, *args, **kwargs)  # Line 222
def get_validation_cache() -> StateCache  # Line 334
def get_metadata_cache() -> StateCache  # Line 339
def get_general_state_cache() -> StateCache  # Line 344
def calculate_state_checksum(data: dict[str, Any]) -> str  # Line 350
def serialize_state_data(data: dict[str, Any], compress: bool = False, compression_threshold: int = 1024) -> bytes  # Line 355
def deserialize_state_data(data: bytes, is_compressed: bool = False) -> dict[str, Any]  # Line 362
async def validate_required_fields(data: dict[str, Any], required_fields: list[str]) -> list[str]  # Line 367
async def validate_decimal_fields(data: dict[str, Any], decimal_fields: list[str]) -> list[str]  # Line 385
async def validate_positive_values(data: dict[str, Any], positive_fields: list[str]) -> list[str]  # Line 420
async def create_state_metadata(...) -> dict[str, Any]  # Line 462
def format_cache_key(state_type: str, state_id: str, prefix: str = 'state') -> str  # Line 513
async def store_in_redis_with_timeout(redis_client, key: str, value: str, ttl: int, timeout: float = 2.0) -> bool  # Line 528
async def get_from_redis_with_timeout(redis_client, key: str, timeout: float = 2.0) -> str | None  # Line 555
def detect_state_changes(old_state: dict[str, Any] | None, new_state: dict[str, Any]) -> set[str]  # Line 577
def calculate_memory_usage(data_structures: list[Any]) -> float  # Line 606
def create_state_change_record(...) -> dict[str, Any]  # Line 629
def ensure_directory_exists(directory_path: str | Path) -> None  # Line 672
def validate_state_transition_rules(...) -> bool  # Line 689
def update_moving_average(current_average: float, new_value: float, count: int) -> float  # Line 718
def format_state_for_logging(state_data: dict[str, Any], max_length: int = 200) -> str  # Line 735
```

### File: state_validation_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotStatus`
- `from src.core.types import OrderSide`
- `from src.core.types import OrderType`

#### Functions:

```python
async def validate_required_fields_with_details(data: dict[str, Any], required_fields: list[str]) -> dict[str, Any]  # Line 20
def validate_string_field_with_details(data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 75
async def validate_decimal_field_with_details(data: dict[str, Any], field_name: str, max_places: int = 8) -> dict[str, Any]  # Line 97
def validate_positive_value_with_details(data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 178
def validate_non_negative_value_with_details(data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 214
def validate_list_field_with_details(data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 252
def validate_dict_field_with_details(data: dict[str, Any], field_name: str) -> dict[str, Any]  # Line 276
def validate_bot_id_format(bot_id: str) -> dict[str, Any]  # Line 300
def validate_bot_status(status: Any) -> dict[str, Any]  # Line 328
def validate_order_side(side: Any) -> dict[str, Any]  # Line 367
def validate_order_type(order_type: Any) -> dict[str, Any]  # Line 406
def validate_symbol_format(symbol: str) -> dict[str, Any]  # Line 445
def validate_capital_allocation(data: dict[str, Any], max_allocation: Decimal | None = None) -> dict[str, Any]  # Line 474
def validate_order_price_logic(data: dict[str, Any]) -> dict[str, Any]  # Line 535
def validate_cash_balance(data: dict[str, Any], min_cash_ratio: Decimal | None = None) -> dict[str, Any]  # Line 605
def validate_var_limits(data: dict[str, Any]) -> dict[str, Any]  # Line 675
def validate_trade_execution(data: dict[str, Any]) -> dict[str, Any]  # Line 724
def validate_strategy_params(data: dict[str, Any]) -> dict[str, Any]  # Line 797
```

### File: strategy_commons.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.utils.arbitrage_helpers import FeeCalculator`

#### Class: `StrategyCommons`

**Purpose**: Comprehensive utility class providing common strategy operations

```python
class StrategyCommons:
    def __init__(self, strategy_name: str, config: dict[str, Any] | None = None)  # Line 35
    def update_market_data(self, data: MarketData) -> None  # Line 65
    def get_technical_analysis(self, indicator_type: str, period: int = 14, **kwargs) -> float | None  # Line 78
    def check_volume_confirmation(self, current_volume: float, lookback_period: int = 20, min_ratio: float = 1.5) -> bool  # Line 123
    def get_volume_profile(self, periods: int = 20) -> dict[str, float]  # Line 147
    def validate_signal_comprehensive(self, ...) -> bool  # Line 165
    def calculate_position_size_with_risk(self, ...) -> Decimal  # Line 204
    def check_stop_loss_conditions(self, ...) -> bool  # Line 244
    def calculate_take_profit_level(self, ...) -> Decimal  # Line 304
    def get_market_condition_summary(self) -> dict[str, Any]  # Line 352
    def analyze_trend_strength(self, lookback_period: int = 20) -> dict[str, float]  # Line 411
    def cleanup(self) -> None  # Line 470
```

### File: strategy_helpers.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import MarketData`
- `from src.utils.decimal_utils import ZERO`
- `from src.utils.decimal_utils import safe_divide`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `PriceHistoryManager`

**Purpose**: Manages price history data for strategy calculations

```python
class PriceHistoryManager:
    def __init__(self, max_length: int = 200)  # Line 30
    def update_history(self, data: MarketData) -> None  # Line 47
    def get_recent_prices(self, periods: int) -> list[Decimal]  # Line 77
    def get_recent_volumes(self, periods: int) -> list[Decimal]  # Line 96
    def has_sufficient_data(self, required_periods: int) -> bool  # Line 115
    def clear_history(self) -> None  # Line 127
```

#### Class: `TechnicalIndicators`

**Purpose**: Common technical indicator calculations used across strategies

```python
class TechnicalIndicators:
    def calculate_sma(prices: list[Decimal], period: int) -> Decimal | None  # Line 144
    def calculate_rsi(prices: list[Decimal], period: int = 14) -> Decimal | None  # Line 168
    def calculate_zscore(prices: list[Decimal], period: int) -> Decimal | None  # Line 220
    def calculate_atr(highs, ...) -> Decimal | None  # Line 257
    def calculate_volatility(prices: list[Decimal], periods: int = 20) -> Decimal  # Line 299
```

#### Class: `VolumeAnalysis`

**Purpose**: Common volume analysis functions used across strategies

```python
class VolumeAnalysis:
    def check_volume_confirmation(current_volume, ...) -> bool  # Line 349
    def calculate_volume_profile(volume_history: list[Decimal], periods: int = 20) -> dict[str, Decimal]  # Line 390
```

#### Class: `StrategySignalValidator`

**Purpose**: Common signal validation patterns used across strategies

```python
class StrategySignalValidator:
    def validate_signal_metadata(signal_metadata: dict[str, Any], required_fields: list[str]) -> bool  # Line 450
    def validate_price_range(price: Decimal, min_price: Decimal = ZERO, max_price: Decimal = Any) -> bool  # Line 475
    def check_signal_freshness(signal_timestamp: datetime, max_age_seconds: int = 300) -> bool  # Line 498
```

### File: string_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`

#### Functions:

```python
def normalize_symbol(symbol: str) -> str  # Line 9
def parse_trading_pair(pair: str) -> tuple[str, str]  # Line 37
def generate_hash(data: str) -> str  # Line 76
def validate_email(email: str) -> bool  # Line 89
def extract_numbers(text: str) -> list[float]  # Line 103
def camel_to_snake(name: str) -> str  # Line 118
def snake_to_camel(name: str) -> str  # Line 133
def truncate(text: str, max_length: int, suffix: str = '...') -> str  # Line 147
```

### File: synthetic_data_generator.py

#### Functions:

```python
def generate_synthetic_ohlcv_data(...) -> pd.DataFrame  # Line 15
def generate_timeframe_mapping() -> dict[str, str]  # Line 93
def validate_ohlcv_data(data: pd.DataFrame) -> tuple[bool, list[str]]  # Line 110
```

### File: technical_indicators.py

**Key Imports:**
- `from src.utils.decimal_utils import to_decimal`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Functions:

```python
def _check_talib()  # Line 45
def calculate_sma_vectorized(prices: np.ndarray, period: int) -> np.ndarray  # Line 55
def calculate_ema_vectorized(prices: np.ndarray, period: int) -> np.ndarray  # Line 72
def calculate_rsi_vectorized(prices: np.ndarray, period: int = 14) -> np.ndarray  # Line 89
def calculate_bollinger_bands_vectorized(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # Line 124
def calculate_macd_vectorized(...) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # Line 153
def calculate_sma_talib(prices: np.ndarray, period: int) -> Decimal | None  # Line 168
def calculate_ema_talib(prices: np.ndarray, period: int) -> Decimal | None  # Line 187
def calculate_rsi_talib(prices: np.ndarray, period: int = 14) -> Decimal | None  # Line 206
def calculate_macd_talib(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, Decimal] | None  # Line 224
def calculate_bollinger_bands_talib(prices: np.ndarray, period: int = 20, std_dev: Decimal = Any) -> dict[str, Decimal] | None  # Line 252
def calculate_atr_talib(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Decimal | None  # Line 294
```

### File: timezone_utils.py

#### Functions:

```python
def ensure_timezone_aware(dt: datetime | None, default_tz: timezone = timezone.utc) -> datetime  # Line 12
def ensure_utc_timezone(dt: datetime | None) -> datetime  # Line 40
```

### File: base_validator.py

#### Class: `BaseRecordValidator`

**Inherits**: ABC
**Purpose**: Base class for validators that process both single records and batches

```python
class BaseRecordValidator(ABC):
    def __init__(self) -> None  # Line 10
    def validate(self, data: Any) -> bool  # Line 14
    def _validate_record(self, record: dict, index: int | None = None) -> bool  # Line 41
    def get_errors(self) -> list[str]  # Line 54
    def reset(self) -> None  # Line 58
    async def health_check(self) -> dict[str, Any]  # Line 62
    def _add_error(self, message: str, index: int | None = None) -> None  # Line 69
```

### File: core.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`

#### Class: `ValidationFramework`

**Purpose**: Centralized validation framework to eliminate duplication

```python
class ValidationFramework:
    def validate_order(order: dict[str, Any]) -> bool  # Line 16
    def _validate_common_params(params: dict[str, Any]) -> None  # Line 72
    def _validate_mean_reversion_params(params: dict[str, Any]) -> None  # Line 80
    def _validate_momentum_params(params: dict[str, Any]) -> None  # Line 93
    def _validate_market_making_params(params: dict[str, Any]) -> None  # Line 104
    def validate_strategy_params(params: dict[str, Any]) -> bool  # Line 114
    def validate_price(price: Any, max_price: Decimal = Any) -> Decimal  # Line 146
    def validate_quantity(quantity: Any, min_qty: Decimal = Any) -> Decimal  # Line 191
    def validate_symbol(symbol: str) -> str  # Line 235
    def validate_exchange_credentials(credentials: dict[str, Any]) -> bool  # Line 261
    def validate_risk_params(params: dict[str, Any]) -> bool  # Line 289
    def validate_risk_parameters(params: dict[str, Any]) -> bool  # Line 325
    def validate_timeframe(timeframe: str) -> str  # Line 370
    def validate_batch(validations: list[tuple[str, Callable[[Any], Any], Any]]) -> dict[str, Any]  # Line 421
```

### File: market_data_validation.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`
- `from src.utils.validation.validation_types import ValidationCategory`
- `from src.utils.validation.validation_types import ValidationIssue`
- `from src.utils.validation.validation_types import ValidationLevel`

#### Class: `MarketDataValidationUtils`

**Purpose**: Centralized market data validation utilities

```python
class MarketDataValidationUtils:
    def validate_symbol_format(symbol: str) -> bool  # Line 25
    def validate_price_value(price, ...) -> Decimal  # Line 57
    def validate_volume_value(volume, ...) -> Decimal  # Line 96
    def validate_timestamp_value(timestamp, ...) -> datetime  # Line 130
    def validate_decimal_precision(value: int | float | str | Decimal, field_name: str, max_decimal_places: int = 8) -> bool  # Line 172
    def validate_bid_ask_spread(bid: Decimal, ask: Decimal) -> bool  # Line 207
    def validate_price_consistency(data: MarketData) -> bool  # Line 235
    def create_validation_issue(field, ...) -> ValidationIssue  # Line 277
```

#### Class: `MarketDataValidator`

**Purpose**: Consolidated market data validator that replaces multiple duplicate implementations

```python
class MarketDataValidator:
    def __init__(self, ...)  # Line 323
    def validate_market_data_record(self, data: MarketData) -> bool  # Line 351
    def validate_market_data_batch(self, data_list: list[MarketData]) -> list[MarketData]  # Line 438
    def _validate_required_fields(self, data: MarketData) -> None  # Line 460
    def get_validation_errors(self) -> list[str]  # Line 473
    def reset(self) -> None  # Line 477
```

### File: service.py

**Key Imports:**
- `from src.core import BaseService`
- `from src.core import HealthStatus`
- `from src.core.exceptions import ErrorCategory`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `ValidationType`

**Inherits**: Enum
**Purpose**: Types of validation operations

```python
class ValidationType(Enum):
```

#### Class: `ValidationContext`

**Inherits**: BaseModel
**Purpose**: Context information for validation operations

```python
class ValidationContext(BaseModel):
    def get_context_hash(self) -> str  # Line 110
```

#### Class: `ValidationDetail`

**Inherits**: BaseModel
**Purpose**: Detailed validation information

```python
class ValidationDetail(BaseModel):
```

#### Class: `ValidationResult`

**Inherits**: BaseModel
**Purpose**: Comprehensive validation result

```python
class ValidationResult(BaseModel):
    def add_error(self, ...) -> None  # Line 145
    def add_warning(self, ...) -> None  # Line 169
    def get_error_summary(self) -> str  # Line 189
    def get_critical_errors(self) -> list[ValidationDetail]  # Line 197
    def has_critical_errors(self) -> bool  # Line 201
```

#### Class: `ValidationRule`

**Inherits**: ABC
**Purpose**: Abstract base class for validation rules

```python
class ValidationRule(ABC):
    def __init__(self, name: str, description: str)  # Line 209
    async def validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult  # Line 214
    def get_cache_key(self, value: Any, context: ValidationContext | None = None) -> str  # Line 220
```

#### Class: `NumericValidationRule`

**Inherits**: ValidationRule
**Purpose**: Validation rule for numeric values

```python
class NumericValidationRule(ValidationRule):
    def __init__(self, ...)  # Line 230
    async def validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult  # Line 244
```

#### Class: `StringValidationRule`

**Inherits**: ValidationRule
**Purpose**: Validation rule for string values

```python
class StringValidationRule(ValidationRule):
    def __init__(self, ...)  # Line 338
    async def validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult  # Line 354
```

#### Class: `ValidationCache`

**Purpose**: Thread-safe validation cache with TTL support

```python
class ValidationCache:
    def __init__(self, default_ttl: int = 300, max_size: int = 10000)  # Line 439
    async def get(self, key: str) -> ValidationResult | None  # Line 446
    async def set(self, key: str, result: ValidationResult, ttl: int | None = None) -> None  # Line 461
    async def _cleanup_cache(self) -> None  # Line 473
    def get_stats(self) -> dict[str, Any]  # Line 511
```

#### Class: `ValidatorRegistry`

**Purpose**: Registry for validation rules

```python
class ValidatorRegistry:
    def __init__(self) -> None  # Line 526
    def register(self, rule: ValidationRule) -> None  # Line 531
    def get(self, name: str) -> ValidationRule | None  # Line 536
    def list_rules(self) -> list[str]  # Line 540
    def _register_default_rules(self) -> None  # Line 544
```

#### Class: `ValidationService`

**Inherits**: BaseService
**Purpose**: Comprehensive validation service for the T-Bot trading system

```python
class ValidationService(BaseService):
    def __init__(self, ...)  # Line 639
    async def _do_start(self) -> None  # Line 679
    async def _do_stop(self) -> None  # Line 686
    async def _service_health_check(self) -> HealthStatus  # Line 693
    async def initialize(self) -> None  # Line 721
    async def shutdown(self) -> None  # Line 725
    def _ensure_initialized(self) -> None  # Line 729
    async def _validate_with_rule(self, rule_name: str, value: Any, context: ValidationContext | None = None) -> ValidationResult  # Line 738
    def _validate_required_order_fields(self, order_data: dict[str, Any], result: ValidationResult) -> None  # Line 796
    def _setup_field_validations(self, order_data: dict[str, Any]) -> list[tuple[str, str, Any]]  # Line 812
    def _validate_price_requirement(self, order_data: dict[str, Any], result: ValidationResult) -> None  # Line 828
    async def _execute_field_validations(self, ...) -> None  # Line 841
    def _store_normalized_value(self, field_result: ValidationResult, result: ValidationResult, field_name: str) -> None  # Line 865
    async def validate_order(self, order_data: dict[str, Any], context: ValidationContext | None = None) -> ValidationResult  # Line 877
    async def _validate_order_business_logic(self, ...) -> None  # Line 921
    async def validate_risk_parameters(self, risk_data: dict[str, Any], context: ValidationContext | None = None) -> ValidationResult  # Line 952
    async def _validate_risk_business_logic(self, ...) -> None  # Line 975
    async def validate_strategy_config(self, strategy_data: dict[str, Any], context: ValidationContext | None = None) -> ValidationResult  # Line 1044
    async def _validate_strategy_business_logic(self, ...) -> None  # Line 1067
    async def _validate_common_strategy_params(self, strategy_data: dict[str, Any], result: ValidationResult) -> None  # Line 1099
    async def _validate_mean_reversion_params(self, strategy_data: dict[str, Any], result: ValidationResult) -> None  # Line 1115
    async def _validate_momentum_params(self, strategy_data: dict[str, Any], result: ValidationResult) -> None  # Line 1151
    async def _validate_market_making_params(self, strategy_data: dict[str, Any], result: ValidationResult) -> None  # Line 1177
    async def validate_market_data(self, market_data: dict[str, Any], context: ValidationContext | None = None) -> ValidationResult  # Line 1212
    async def validate_batch(self, ...) -> dict[str, ValidationResult]  # Line 1276
    def validate_price(self, price: Any) -> bool  # Line 1439
    def validate_quantity(self, quantity: Any) -> bool  # Line 1478
    def validate_symbol(self, symbol: str) -> bool  # Line 1517
    def register_custom_rule(self, rule: ValidationRule) -> None  # Line 1552
    def validate_decimal(self, value: Any) -> Any  # Line 1557
    def get_validation_stats(self) -> dict[str, Any]  # Line 1582
    async def __aenter__(self) -> 'ValidationService'  # Line 1596
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None  # Line 1601
```

#### Functions:

```python
async def get_validation_service(validation_service: ValidationService | None = None) -> ValidationService  # Line 1609
async def shutdown_validation_service(validation_service: ValidationService | None = None) -> None  # Line 1638
```

### File: validation_types.py

**Key Imports:**
- `from src.core.types import ValidationLevel`

#### Class: `ValidationCategory`

**Inherits**: Enum
**Purpose**: Validation category types

```python
class ValidationCategory(Enum):
```

#### Class: `ValidationSeverity`

**Inherits**: Enum
**Purpose**: Validation severity levels

```python
class ValidationSeverity(Enum):
```

#### Class: `QualityDimension`

**Inherits**: Enum
**Purpose**: Data quality dimensions

```python
class QualityDimension(Enum):
```

#### Class: `ValidationIssue`

**Purpose**: Standardized validation issue record

```python
class ValidationIssue:
    def to_dict(self) -> dict[str, Any]  # Line 64
```

#### Class: `QualityScore`

**Purpose**: Data quality score breakdown

```python
class QualityScore:
    def to_dict(self) -> dict[str, float]  # Line 93
```

#### Functions:

```python
def _get_utc_now() -> datetime  # Line 45
```

### File: validation_utilities.py

#### Class: `ValidationResult`

**Inherits**: Enum
**Purpose**: Standard validation result enumeration

```python
class ValidationResult(Enum):
```

#### Class: `ErrorType`

**Inherits**: Enum
**Purpose**: Standard error type classification

```python
class ErrorType(Enum):
```

#### Class: `RecoveryStatus`

**Inherits**: Enum
**Purpose**: Standard recovery operation status

```python
class RecoveryStatus(Enum):
```

#### Class: `AuditEventType`

**Inherits**: Enum
**Purpose**: Standard audit event types

```python
class AuditEventType(Enum):
```

#### Class: `StateValidationError`

**Purpose**: Standard state validation error structure

```python
class StateValidationError:
```

#### Class: `ValidationWarning`

**Purpose**: Standard validation warning structure

```python
class ValidationWarning:
```

#### Class: `ValidationResultData`

**Purpose**: Standard validation result data structure

```python
class ValidationResultData:
```

#### Class: `AuditEntry`

**Purpose**: Standard audit trail entry structure

```python
class AuditEntry:
```

#### Functions:

```python
def classify_error_type(exception: Exception) -> ErrorType  # Line 167
def create_audit_entry(...) -> AuditEntry  # Line 225
```

### File: validators.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.validation.core import ValidationFramework`

#### Class: `ValidationFramework`

**Purpose**: Centralized validation framework that provides simplified interfaces
to the comprehensive validation

```python
class ValidationFramework:
    def validate_order(order: dict[str, Any]) -> bool  # Line 464
    def validate_strategy_params(params: dict[str, Any]) -> bool  # Line 489
    def validate_price(price: Any, max_price: Decimal = Any) -> Decimal  # Line 512
    def validate_quantity(quantity: Any, min_qty: Decimal = Any) -> Decimal  # Line 534
    def validate_symbol(symbol: str) -> str  # Line 556
    def validate_exchange_credentials(credentials: dict[str, Any]) -> bool  # Line 577
    def validate_risk_params(params: dict[str, Any]) -> bool  # Line 597
    def validate_risk_parameters(params: dict[str, Any]) -> bool  # Line 618
    def validate_timeframe(timeframe: str) -> str  # Line 634
    def validate_batch(validations: list[tuple[str, Callable[[Any], Any], Any]]) -> dict[str, Any]  # Line 654
```

#### Functions:

```python
def validate_decimal_precision(value: Decimal | float | str, places: int = 8) -> bool  # Line 31
def validate_ttl(ttl: int | float | None, max_ttl: int = 86400) -> int  # Line 74
def validate_precision_range(precision: int, min_precision: int = 0, max_precision: int = 28) -> int  # Line 110
def validate_financial_range(...) -> Decimal  # Line 139
def validate_null_handling(value: Any, allow_null: bool = False, field_name: str = 'value') -> Any  # Line 188
def validate_type_conversion(value: Any, target_type: type, field_name: str = 'value', strict: bool = True) -> Any  # Line 226
def validate_market_conditions(...) -> dict[str, Decimal]  # Line 346
def _validate_symbol(data: dict[str, Any]) -> None  # Line 382
def _validate_numeric_field(data: dict[str, Any], field: str) -> None  # Line 392
def _validate_bid_ask_relationship(data: dict[str, Any]) -> None  # Line 408
def _validate_timestamp(data: dict[str, Any]) -> None  # Line 431
def validate_market_data(data: dict[str, Any]) -> bool  # Line 438
```

### File: web_interface_utils.py

**Key Imports:**
- `from src.core.exceptions import EntityNotFoundError`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Functions:

```python
def handle_api_error(...) -> HTTPException  # Line 25
def safe_format_currency(amount: Decimal, currency: str = 'USD') -> str  # Line 107
def safe_format_percentage(percentage: Decimal) -> str  # Line 128
def safe_get_api_facade()  # Line 148
def create_error_response(message: str, status_code: int = 500) -> HTTPException  # Line 167
def handle_not_found(item_type: str, item_id: str) -> HTTPException  # Line 181
def extract_error_details(error: Exception) -> dict[str, Any]  # Line 197
```

### File: websocket_connection_utils.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.utils.decorators import retry`

#### Class: `WebSocketConnectionManager`

**Purpose**: Base WebSocket connection manager with common functionality

```python
class WebSocketConnectionManager:
    def __init__(self, ...)  # Line 29
    async def connect(self) -> bool  # Line 95
    async def disconnect(self) -> None  # Line 151
    async def send_message(self, message: dict[str, Any]) -> bool  # Line 184
    def add_callback(self, channel: str, callback: Callable) -> None  # Line 214
    def remove_callback(self, channel: str, callback: Callable) -> None  # Line 226
    async def subscribe_channel(self, channel: str, subscription_data: dict[str, Any]) -> bool  # Line 242
    async def unsubscribe_channel(self, channel: str, unsubscription_data: dict[str, Any]) -> bool  # Line 263
    async def _listen_messages(self) -> None  # Line 286
    async def _process_message(self, data: dict[str, Any]) -> None  # Line 326
    async def _health_monitor(self) -> None  # Line 365
    async def _schedule_reconnect(self) -> None  # Line 391
    async def _reconnect(self) -> None  # Line 398
    async def _resubscribe_channels(self) -> None  # Line 434
    async def _cancel_tasks(self) -> None  # Line 468
    def get_connection_stats(self) -> dict[str, Any]  # Line 489
```

#### Class: `AuthenticatedWebSocketManager`

**Inherits**: WebSocketConnectionManager
**Purpose**: WebSocket manager with authentication support

```python
class AuthenticatedWebSocketManager(WebSocketConnectionManager):
    def __init__(self, ...)  # Line 510
    async def authenticate(self) -> bool  # Line 535
```

#### Class: `MultiStreamWebSocketManager`

**Purpose**: Manager for multiple WebSocket connections

```python
class MultiStreamWebSocketManager:
    def __init__(self, exchange_name: str) -> None  # Line 550
    def add_connection(self, name: str, connection: WebSocketConnectionManager) -> None  # Line 561
    async def connect_all(self) -> dict[str, bool]  # Line 571
    async def disconnect_all(self) -> None  # Line 616
    def get_connection(self, name: str) -> WebSocketConnectionManager | None  # Line 641
    def get_all_stats(self) -> dict[str, dict[str, Any]]  # Line 653
```

#### Class: `WebSocketMessageBuffer`

**Purpose**: Buffer for WebSocket messages during connection issues

```python
class WebSocketMessageBuffer:
    def __init__(self, max_size: int = 1000) -> None  # Line 663
    def is_full(self) -> bool  # Line 674
    def add_message(self, message: dict[str, Any]) -> None  # Line 678
    def get_messages(self, count: int | None = None) -> list[dict[str, Any]]  # Line 690
    def get_message_count(self) -> int  # Line 709
```

#### Class: `WebSocketHeartbeatManager`

**Purpose**: Manager for WebSocket heartbeat/ping functionality

```python
class WebSocketHeartbeatManager:
    def __init__(self, ...)  # Line 717
    async def start(self) -> None  # Line 737
    async def stop(self) -> None  # Line 745
    async def _heartbeat_loop(self) -> None  # Line 756
```

#### Class: `WebSocketSubscriptionManager`

**Purpose**: Manager for WebSocket subscriptions and callbacks

```python
class WebSocketSubscriptionManager:
    def __init__(self) -> None  # Line 788
    def add_subscription(self, stream_name: str, callback: Callable) -> None  # Line 793
    def remove_subscription(self, stream_name: str) -> None  # Line 804
    def get_subscriptions(self) -> list[str]  # Line 814
    async def handle_message(self, message: dict[str, Any]) -> None  # Line 818
```

#### Class: `WebSocketStreamManager`

**Purpose**: Manager for WebSocket streams

```python
class WebSocketStreamManager:
    def __init__(self, max_streams: int = 50) -> None  # Line 845
    def add_stream(self, stream_id: str, stream_config: dict[str, Any], handler: Callable) -> bool  # Line 856
    def remove_stream(self, stream_id: str) -> bool  # Line 875
    def get_stream_count(self) -> int  # Line 891
    def is_at_capacity(self) -> bool  # Line 895
    async def handle_stream_message(self, stream_id: str, message: dict[str, Any]) -> None  # Line 899
```

### File: websocket_manager_utils.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `BaseWebSocketManager`

**Purpose**: Base WebSocket connection manager with common functionality

```python
class BaseWebSocketManager:
    def __init__(self, manager_name: str = 'WebSocket')  # Line 29
    async def connect(self, websocket: WebSocket, user_id: str) -> bool  # Line 35
    def disconnect(self, user_id: str)  # Line 58
    def subscribe(self, user_id: str, subscription_key: str)  # Line 79
    def unsubscribe(self, user_id: str, subscription_key: str)  # Line 107
    async def send_personal_message(self, message: dict[str, Any], user_id: str) -> bool  # Line 131
    async def broadcast_to_subscription(self, message: dict[str, Any], subscription_key: str)  # Line 162
    async def broadcast_to_all(self, message: dict[str, Any])  # Line 198
    def get_connection_count(self) -> int  # Line 233
    def get_subscription_count(self, subscription_key: str) -> int  # Line 237
    def get_user_subscriptions(self, user_id: str) -> set[str]  # Line 241
    def is_user_subscribed(self, user_id: str, subscription_key: str) -> bool  # Line 245
```

#### Class: `MarketDataWebSocketManager`

**Inherits**: BaseWebSocketManager
**Purpose**: WebSocket manager specifically for market data streaming

```python
class MarketDataWebSocketManager(BaseWebSocketManager):
    def __init__(self)  # Line 253
    def subscribe_to_symbol(self, user_id: str, symbol: str)  # Line 256
    def unsubscribe_from_symbol(self, user_id: str, symbol: str)  # Line 260
    async def broadcast_market_data(self, symbol: str, data: dict[str, Any])  # Line 264
```

#### Class: `BotStatusWebSocketManager`

**Inherits**: BaseWebSocketManager
**Purpose**: WebSocket manager specifically for bot status updates

```python
class BotStatusWebSocketManager(BaseWebSocketManager):
    def __init__(self)  # Line 289
    def subscribe_to_bot(self, user_id: str, bot_id: str)  # Line 292
    def unsubscribe_from_bot(self, user_id: str, bot_id: str)  # Line 296
    async def broadcast_bot_status(self, bot_id: str, status_data: dict[str, Any])  # Line 300
```

#### Class: `ExchangeWebSocketReconnectionManager`

**Purpose**: Common reconnection logic for exchange WebSocket connections

```python
class ExchangeWebSocketReconnectionManager:
    def __init__(self, exchange_name: str, max_reconnect_attempts: int = 5)  # Line 401
    def reset_reconnect_attempts(self) -> None  # Line 407
    def should_attempt_reconnect(self) -> bool  # Line 411
    def calculate_reconnect_delay(self) -> float  # Line 415
    async def schedule_reconnect(self, reconnect_callback) -> None  # Line 422
    async def _reconnect_with_delay(self, reconnect_callback) -> None  # Line 441
    def cancel_reconnect(self) -> None  # Line 456
```

#### Functions:

```python
async def authenticate_websocket(websocket: WebSocket, token: str | None = None) -> str | None  # Line 338
async def handle_websocket_disconnect(websocket: WebSocket, manager: BaseWebSocketManager, user_id: str)  # Line 378
```

---
**Generated**: Complete reference for utils module
**Total Classes**: 177
**Total Functions**: 358