# ERROR_HANDLING Module Reference

## INTEGRATION
**Dependencies**: core, exchanges, execution, strategies, utils
**Used By**: strategies
**Provides**: ConnectionManager, ErrorHandlingService, MockContextManager, SecureErrorContextManager
**Patterns**: Async Operations, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Security**:
- Authentication
**Performance**:
- Parallel execution
- Parallel execution
- Retry mechanisms
**Architecture**:
- GlobalErrorHandler inherits from base architecture
- ErrorPatternAnalytics inherits from base architecture
- RetryRecovery inherits from base architecture

## MODULE OVERVIEW
**Files**: 29 Python files
**Classes**: 78
**Functions**: 59

## COMPLETE API REFERENCE

## PROTOCOLS & INTERFACES

### Protocol: `ErrorHandlerProtocol`

**Purpose**: Protocol for error handlers

**Required Methods:**
- `can_handle(self, error: Exception) -> bool`
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> Any`
- `async process(self, error: Exception, context: dict[str, Any] | None = None) -> Any`

## IMPLEMENTATIONS

### Implementation: `ErrorHandlerBase` 🔧

**Inherits**: ABC
**Purpose**: Base class for all error handlers using Chain of Responsibility pattern
**Status**: Abstract Base Class

**Implemented Methods:**
- `can_handle(self, error: Exception) -> bool` - Line 30
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> Any` - Line 43
- `async process(self, error: Exception, context: dict[str, Any] | None = None) -> Any` - Line 56
- `set_next(self, handler: 'ErrorHandlerBase') -> 'ErrorHandlerBase'` - Line 77

### Implementation: `ConnectionState` ✅

**Inherits**: Enum
**Purpose**: Connection state enumeration
**Status**: Complete

### Implementation: `ConnectionHealth` ✅

**Inherits**: Enum
**Purpose**: Connection health status
**Status**: Complete

### Implementation: `ConnectionInfo` ✅

**Purpose**: Basic connection information
**Status**: Complete

### Implementation: `ConnectionManager` ✅

**Purpose**: Simple connection manager with reconnection support
**Status**: Complete

**Implemented Methods:**
- `async establish_connection(self, ...) -> bool` - Line 63
- `async close_connection(self, connection_id: str) -> bool` - Line 111
- `async reconnect_connection(self, connection_id: str) -> bool` - Line 136
- `get_connection_status(self, connection_id: str) -> dict[str, Any] | None` - Line 157
- `is_connection_healthy(self, connection_id: str) -> bool` - Line 172
- `async cleanup_resources(self) -> None` - Line 181

### Implementation: `ErrorContext` ✅

**Purpose**: Comprehensive error context for tracking, recovery, and analysis
**Status**: Complete

**Implemented Methods:**
- `error_type(self) -> str` - Line 96
- `error_message(self) -> str` - Line 101
- `from_exception(cls, ...) -> 'ErrorContext'` - Line 106
- `function(self) -> str | None` - Line 169
- `function(self, value: str | None)` - Line 174
- `to_dict(self, ...) -> dict[str, Any]` - Line 178
- `to_legacy_dict(self, ...) -> dict[str, Any]` - Line 261
- `add_detail(self, key: str, value: Any) -> None` - Line 302
- `add_metadata(self, key: str, value: Any) -> None` - Line 306
- `increment_recovery_attempts(self) -> None` - Line 310
- `can_retry_recovery(self) -> bool` - Line 314
- `is_critical(self) -> bool` - Line 318
- `is_high_severity(self) -> bool` - Line 322
- `requires_escalation(self) -> bool` - Line 326
- `from_decorator_context(cls, ...) -> 'ErrorContext'` - Line 331

### Implementation: `ErrorContextFactory` ✅

**Inherits**: BaseFactory[ErrorContext]
**Purpose**: Factory for creating standardized error contexts with DI support
**Status**: Complete

**Implemented Methods:**
- `create_context_dict(self, error: Exception, **kwargs) -> dict[str, Any]` - Line 374
- `create(self, error: Exception, **kwargs) -> dict[str, Any]` - Line 437
- `create_context(self, context_type: str = 'standard', error: Exception | None = None, **kwargs) -> ErrorContext` - Line 450
- `create_from_frame(self, error: Exception, frame: Any | None = None, **kwargs) -> dict[str, Any]` - Line 469
- `create_minimal(self, error: Exception) -> dict[str, Any]` - Line 513
- `enrich_context(self, base_context: dict[str, Any], **additional) -> dict[str, Any]` - Line 529

### Implementation: `ErrorDataTransformer` ✅

**Purpose**: Handles consistent data transformation for error_handling module
**Status**: Complete

**Implemented Methods:**
- `transform_error_to_event_data(error, ...) -> dict[str, Any]` - Line 21
- `transform_context_to_event_data(context: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 55
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 89
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'error_handling') -> dict[str, Any]` - Line 112
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 148
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]` - Line 194
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 240
- `apply_cross_module_validation(cls, ...) -> dict[str, Any]` - Line 259

### Implementation: `FallbackStrategy` ✅

**Inherits**: Enum
**Purpose**: Fallback strategies for error handling
**Status**: Complete

### Implementation: `RetryConfig` ✅

**Purpose**: Simple retry configuration
**Status**: Complete

### Implementation: `CircuitBreakerConfig` ✅

**Purpose**: Simple circuit breaker configuration
**Status**: Complete

### Implementation: `FallbackConfig` ✅

**Purpose**: Simple fallback configuration
**Status**: Complete

### Implementation: `CircuitBreaker` ✅

**Purpose**: Circuit breaker pattern implementation for preventing cascading failures
**Status**: Complete

**Implemented Methods:**
- `call(self, func: Callable[Ellipsis, Any], *args: Any, **kwargs: Any) -> Any` - Line 86
- `should_transition_to_half_open(self) -> bool` - Line 115
- `open(self) -> None` - Line 122
- `is_open(self) -> bool` - Line 128
- `reset(self) -> None` - Line 132
- `threshold(self) -> int` - Line 140

### Implementation: `ErrorPatternCache` ✅

**Purpose**: Optimized error pattern storage with size limits and TTL
**Status**: Complete

**Implemented Methods:**
- `add_pattern(self, pattern: dict[str, Any] | Any) -> None` - Line 166
- `get_pattern(self, pattern_id: str) -> dict[str, Any] | None` - Line 182
- `get_all_patterns(self) -> dict[str, dict[str, Any]]` - Line 200
- `cleanup_expired(self) -> None` - Line 234
- `size(self) -> int` - Line 238
- `get_last_cleanup(self) -> datetime` - Line 242

### Implementation: `ErrorHandler` ✅

**Purpose**: Comprehensive error handling and recovery system with optimized memory management
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector) -> None` - Line 301
- `classify_error(self, error: Exception) -> ErrorSeverity` - Line 336
- `validate_module_boundary_input(self, data: dict[str, Any], source_module: str) -> dict[str, Any]` - Line 367
- `create_error_context(self, error: Exception, component: str, operation: str, **kwargs) -> 'ErrorContext'` - Line 434
- `async handle_error(self, ...) -> bool` - Line 584
- `handle_error_sync(self, ...) -> bool` - Line 740
- `handle_error_batch(self, ...) -> list[bool]` - Line 839
- `get_retry_policy(self, error_type: str) -> dict[str, Any]` - Line 1024
- `get_circuit_breaker_status(self) -> dict[str, str]` - Line 1028
- `get_error_patterns(self) -> dict[str, dict[str, Any]]` - Line 1032
- `get_memory_usage_stats(self) -> dict[str, Any]` - Line 1036
- `async cleanup_resources(self) -> None` - Line 1045
- `validate_data_flow_consistency(self, data: dict[str, Any]) -> dict[str, Any]` - Line 1112
- `async shutdown(self) -> None` - Line 1161

### Implementation: `ErrorHandlerFactory` ✅

**Inherits**: BaseFactory[ErrorHandlerProtocol]
**Purpose**: Factory to create handlers without direct imports
**Status**: Complete

**Implemented Methods:**
- `register(cls, ...) -> None` - Line 51
- `create(cls, error_type: str, next_handler: ErrorHandlerBase | None = None, **kwargs) -> ErrorHandlerProtocol` - Line 84
- `list_handlers(cls) -> list[str]` - Line 135
- `set_dependency_container(cls, container: Any) -> None` - Line 140
- `clear(cls) -> None` - Line 194

### Implementation: `ErrorHandlerChain` ✅

**Inherits**: BaseFactory[ErrorHandlerProtocol]
**Purpose**: Manages a chain of error handlers
**Status**: Complete

**Implemented Methods:**
- `build_chain(self, handler_types: list[str]) -> None` - Line 227
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> Any` - Line 248
- `add_handler(self, handler_type: str) -> None` - Line 264
- `create_default_chain(cls, dependency_container: Any | None = None) -> 'ErrorHandlerChain'` - Line 280

### Implementation: `GlobalErrorHandler` ✅

**Inherits**: BaseService
**Purpose**: Global error handler that provides consistent error handling across the application
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector) -> None` - Line 107
- `register_database_handler(self)` - Line 126
- `register_error_callback(self, callback: Callable[[Exception, dict], None])` - Line 142
- `register_critical_callback(self, callback: Callable[[Exception, dict], None])` - Line 151
- `register_recovery_strategy(self, error_type: type, strategy: Callable)` - Line 160
- `async handle_error(self, ...) -> dict[str, Any]` - Line 170
- `handle_error_sync(self, ...) -> dict[str, Any]` - Line 283
- `handle_exception_hook(self, exc_type, exc_value, exc_traceback)` - Line 356
- `install_global_handler(self)` - Line 413
- `error_handler_decorator(self, severity: str = 'error', reraise: bool = True, default_return: Any = None)` - Line 417
- `get_statistics(self) -> dict[str, Any]` - Line 483
- `reset_statistics(self)` - Line 499

### Implementation: `UniversalErrorHandler` ✅

**Purpose**: Simple universal error handler stub for backward compatibility
**Status**: Complete

**Implemented Methods:**
- `handle_error(self, error: Exception, context: dict | None = None) -> bool` - Line 39

### Implementation: `HandlerPool` ✅

**Purpose**: Singleton pool for managing UniversalErrorHandler instances
**Status**: Complete

**Implemented Methods:**
- `get_handler(self, ...) -> UniversalErrorHandler` - Line 141
- `get_stats(self) -> dict[str, Any]` - Line 197
- `shutdown(self)` - Line 217
- `async async_shutdown(self)` - Line 279
- `clear(self)` - Line 289

### Implementation: `AuthenticationErrorHandler` ✅

**Inherits**: ErrorHandlerBase
**Purpose**: Secure handler for authentication and authorization errors
**Status**: Complete

**Implemented Methods:**
- `can_handle(self, error: Exception) -> bool` - Line 86
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]` - Line 147
- `get_security_stats(self) -> dict[str, Any]` - Line 692
- `reset_entity_failures(self, entity: str) -> bool` - Line 724

### Implementation: `DatabaseErrorHandler` ✅

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for database-related errors with secure sanitization
**Status**: Complete

**Implemented Methods:**
- `can_handle(self, error: Exception) -> bool` - Line 21
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]` - Line 60

### Implementation: `NetworkErrorHandler` ✅

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for network-related errors
**Status**: Complete

**Implemented Methods:**
- `can_handle(self, error: Exception) -> bool` - Line 42
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]` - Line 68

### Implementation: `RateLimitErrorHandler` ✅

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for rate limit errors with secure sanitization
**Status**: Complete

**Implemented Methods:**
- `can_handle(self, error: Exception) -> bool` - Line 136
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]` - Line 141

### Implementation: `ValidationErrorHandler` ✅

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for validation errors with secure sanitization
**Status**: Complete

**Implemented Methods:**
- `can_handle(self, error: Exception) -> bool` - Line 24
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]` - Line 50

### Implementation: `DataValidationErrorHandler` ✅

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for data validation errors with secure sanitization
**Status**: Complete

**Implemented Methods:**
- `can_handle(self, error: Exception) -> bool` - Line 91
- `async handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]` - Line 96

### Implementation: `ErrorHandlingServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for error handling service layer
**Status**: Complete

**Implemented Methods:**
- `async handle_error(self, ...) -> dict[str, Any]` - Line 18
- `async handle_global_error(self, ...) -> dict[str, Any]` - Line 29
- `async validate_state_consistency(self, component: str = 'all') -> dict[str, Any]` - Line 35
- `async get_error_patterns(self) -> dict[str, Any]` - Line 39
- `async health_check(self) -> HealthCheckResult` - Line 43

### Implementation: `ErrorPatternAnalyticsInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for error pattern analytics service
**Status**: Complete

**Implemented Methods:**
- `add_error_event(self, error_context: dict[str, Any]) -> None` - Line 52
- `async add_batch_error_events(self, error_contexts: list[dict[str, Any]]) -> None` - Line 56
- `get_pattern_summary(self) -> dict[str, Any]` - Line 60
- `get_correlation_summary(self) -> dict[str, Any]` - Line 64
- `get_trend_summary(self) -> dict[str, Any]` - Line 68
- `async cleanup(self) -> None` - Line 72

### Implementation: `ErrorHandlerInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for error handler components
**Status**: Complete

**Implemented Methods:**
- `async handle_error(self, error: Exception, context: Any, recovery_strategy: Any | None = None) -> bool` - Line 81
- `classify_error(self, error: Exception) -> Any` - Line 90
- `create_error_context(self, error: Exception, component: str, operation: str, **kwargs) -> Any` - Line 94
- `async cleanup_resources(self) -> None` - Line 100
- `async shutdown(self) -> None` - Line 104

### Implementation: `GlobalErrorHandlerInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for global error handler
**Status**: Complete

**Implemented Methods:**
- `async handle_error(self, ...) -> dict[str, Any]` - Line 113
- `get_statistics(self) -> dict[str, Any]` - Line 122

### Implementation: `ErrorHandlingServicePort` 🔧

**Inherits**: ABC
**Purpose**: Port interface for error handling service (hexagonal architecture)
**Status**: Abstract Base Class

**Implemented Methods:**
- `async process_error(self, ...) -> dict[str, Any]` - Line 131
- `async analyze_error_patterns(self) -> dict[str, Any]` - Line 142
- `async validate_system_state(self, component: str = 'all') -> dict[str, Any]` - Line 147

### Implementation: `ErrorHandlingRepositoryPort` 🔧

**Inherits**: ABC
**Purpose**: Repository port for error handling data persistence
**Status**: Abstract Base Class

**Implemented Methods:**
- `async store_error_event(self, error_data: dict[str, Any]) -> str` - Line 156
- `async retrieve_error_patterns(self, component: str | None = None, hours: int = 24) -> list[dict[str, Any]]` - Line 161
- `async update_error_statistics(self, stats: dict[str, Any]) -> None` - Line 168

### Implementation: `ErrorTrend` ✅

**Purpose**: Error trend information
**Status**: Complete

### Implementation: `ErrorPatternAnalytics` ✅

**Inherits**: BaseService
**Purpose**: Simple error pattern analyzer
**Status**: Complete

**Implemented Methods:**
- `add_error_event(self, error_context: dict[str, Any]) -> None` - Line 95
- `get_pattern_summary(self) -> dict[str, Any]` - Line 146
- `get_recent_errors(self, hours: int = 1) -> list[dict[str, Any]]` - Line 164
- `get_correlation_summary(self) -> dict[str, Any]` - Line 170
- `get_trend_summary(self) -> dict[str, Any]` - Line 189
- `get_error_patterns(self) -> list[dict[str, Any]]` - Line 227
- `async add_batch_error_events(self, error_contexts: list[dict[str, Any]]) -> None` - Line 245
- `async cleanup(self) -> None` - Line 287

### Implementation: `PropagationMethod` ✅

**Inherits**: Enum
**Purpose**: Error propagation methods aligned with core patterns
**Status**: Complete

### Implementation: `ProcessingStage` ✅

**Inherits**: Enum
**Purpose**: Processing stages for error flow tracking
**Status**: Complete

### Implementation: `RecoveryStrategy` ✅

**Inherits**: Protocol
**Purpose**: Protocol for all recovery strategies with proper type annotations
**Status**: Complete

**Implemented Methods:**
- `can_recover(self, error: Exception, context: dict[str, Any]) -> bool` - Line 25
- `async recover(self, error: Exception, context: dict[str, Any]) -> Any` - Line 29
- `max_attempts(self) -> int` - Line 34

### Implementation: `RetryRecovery` ✅

**Inherits**: BaseComponent
**Purpose**: Retry recovery strategy with exponential backoff
**Status**: Complete

**Implemented Methods:**
- `can_recover(self, error: Exception, context: dict[str, Any]) -> bool` - Line 74
- `async recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]` - Line 85
- `max_attempts(self) -> int` - Line 136

### Implementation: `CircuitBreakerRecovery` ✅

**Inherits**: BaseComponent
**Purpose**: Circuit breaker recovery strategy
**Status**: Complete

**Implemented Methods:**
- `can_recover(self, error: Exception, context: dict[str, Any]) -> bool` - Line 179
- `async recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]` - Line 184
- `max_attempts(self) -> int` - Line 267

### Implementation: `FallbackRecovery` ✅

**Inherits**: BaseComponent
**Purpose**: Fallback to alternative implementation
**Status**: Complete

**Implemented Methods:**
- `can_recover(self, error: Exception, context: dict[str, Any]) -> bool` - Line 295
- `async recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]` - Line 299
- `max_attempts(self) -> int` - Line 427

### Implementation: `RecoveryDataServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for recovery data operations abstracted from database details
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_recovery_context(self, scenario: str) -> dict[str, Any]` - Line 42
- `async execute_position_recovery(self, recovery_data: dict[str, Any]) -> bool` - Line 43
- `async execute_order_recovery(self, recovery_data: dict[str, Any]) -> bool` - Line 44
- `async log_recovery_action(self, action: str, details: dict[str, Any]) -> None` - Line 45

### Implementation: `RiskServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for risk management service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 51
- `async update_position(self, symbol: str, quantity: Any, side: str, exchange: str) -> None` - Line 52
- `async update_stop_loss(self, symbol: str, stop_loss_price: Any, exchange: str) -> None` - Line 55

### Implementation: `CacheServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for cache service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 61
- `async get(self, key: str) -> Any` - Line 62
- `async set(self, key: str, value: Any, expiry: int | None = None) -> None` - Line 63

### Implementation: `StateServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for state service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 69
- `async create_checkpoint(self, component_name: str, state_data: dict[str, Any]) -> str` - Line 70
- `async get_latest_checkpoint(self, component_name: str) -> dict[str, Any] | None` - Line 71
- `async restore_checkpoint(self, checkpoint_id: str, component_name: str) -> None` - Line 72

### Implementation: `BotServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for bot management service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 78
- `async pause_bot(self, component: str) -> None` - Line 79
- `async resume_bot(self, component: str) -> None` - Line 80

### Implementation: `RecoveryScenario` ✅

**Inherits**: BaseComponent
**Purpose**: Base class for recovery scenarios with service injection
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector) -> None` - Line 127
- `async execute_recovery(self, context: Any) -> bool` - Line 153

### Implementation: `PartialFillRecovery` ✅

**Inherits**: RecoveryScenario
**Purpose**: Handle partially filled orders with intelligent recovery
**Status**: Complete

**Implemented Methods:**
- `async execute_recovery(self, context: dict[str, Any]) -> bool` - Line 202

### Implementation: `NetworkDisconnectionRecovery` ✅

**Inherits**: RecoveryScenario
**Purpose**: Handle network disconnection with automatic reconnection
**Status**: Complete

**Implemented Methods:**
- `async execute_recovery(self, context: dict[str, Any]) -> bool` - Line 474

### Implementation: `ExchangeMaintenanceRecovery` ✅

**Inherits**: RecoveryScenario
**Purpose**: Handle exchange maintenance with graceful degradation
**Status**: Complete

**Implemented Methods:**
- `async execute_recovery(self, context: dict[str, Any]) -> bool` - Line 937

### Implementation: `DataFeedInterruptionRecovery` ✅

**Inherits**: RecoveryScenario
**Purpose**: Handle data feed interruptions with fallback sources
**Status**: Complete

**Implemented Methods:**
- `async execute_recovery(self, context: dict[str, Any]) -> bool` - Line 1025

### Implementation: `OrderRejectionRecovery` ✅

**Inherits**: RecoveryScenario
**Purpose**: Handle order rejections with intelligent retry
**Status**: Complete

**Implemented Methods:**
- `async execute_recovery(self, context: dict[str, Any]) -> bool` - Line 1130

### Implementation: `APIRateLimitRecovery` ✅

**Inherits**: RecoveryScenario
**Purpose**: Handle API rate limit violations with automatic throttling
**Status**: Complete

**Implemented Methods:**
- `async execute_recovery(self, context: dict[str, Any]) -> bool` - Line 1249

### Implementation: `SecureErrorReport` ✅

**Purpose**: Secure error report with filtered information
**Status**: Complete

**Implemented Methods:**

### Implementation: `SecureErrorContextManager` ✅

**Purpose**: Secure error context manager with role-based filtering
**Status**: Complete

**Implemented Methods:**
- `create_secure_report(self, error: Exception, **context) -> SecureErrorReport` - Line 47

### Implementation: `SecureLogger` ✅

**Purpose**: Secure logger with data sanitization capabilities
**Status**: Complete

**Implemented Methods:**
- `log(self, level: LogLevel, message: str, context: dict[str, Any] | None = None) -> None` - Line 78
- `log_error(self, ...) -> None` - Line 84
- `log_security_event(self, ...) -> None` - Line 112
- `log_audit_trail(self, ...) -> None` - Line 136
- `sanitize_log_data(self, data: dict[str, Any]) -> dict[str, Any]` - Line 156
- `determine_information_level_from_context(self, context: Any = None) -> str` - Line 166
- `format_log_entry(self, entry: Any, format_type: str = 'json') -> str` - Line 188
- `should_log(self, level: LogLevel) -> bool` - Line 218
- `log_performance_metrics(self, metrics: dict[str, Any]) -> None` - Line 253
- `serialize_log_entry(self, entry: Any) -> str` - Line 258

### Implementation: `ErrorPattern` ✅

**Purpose**: Enhanced error pattern with security analytics
**Status**: Complete

### Implementation: `AnalyticsConfig` ✅

**Purpose**: Analytics configuration for backward compatibility
**Status**: Complete

### Implementation: `SecurePatternAnalytics` ✅

**Purpose**: Simplified pattern analytics for backward compatibility
**Status**: Complete

**Implemented Methods:**
- `record_error_event(self, ...) -> None` - Line 82
- `async analyze_patterns(self) -> list[ErrorPattern]` - Line 116
- `async get_patterns_summary(self, severity_filter: PatternSeverity | None = None) -> dict[str, Any]` - Line 237

### Implementation: `MockContextManager` ✅

**Purpose**: Mock context manager for testing
**Status**: Complete

**Implemented Methods:**
- `get_context(self)` - Line 46
- `create_secure_report(self, error, security_context = None, error_context = None)` - Line 49

### Implementation: `MockReportingRule` ✅

**Purpose**: Mock reporting rule for testing
**Status**: Complete

**Implemented Methods:**

### Implementation: `SecureErrorReporter` ✅

**Purpose**: Simple secure error reporter for backward compatibility
**Status**: Complete

**Implemented Methods:**
- `generate_report(self, report_type: ReportType, data: dict[str, Any]) -> dict[str, Any]` - Line 86
- `send_alert(self, ...) -> bool` - Line 91
- `create_error_report(self, error: Exception, security_context: Any, error_context: dict[str, Any]) -> Any` - Line 98
- `async submit_error_report(self, ...) -> bool` - Line 104
- `evaluate_reporting_rules(self, error: Exception, context: dict[str, Any]) -> list` - Line 110
- `filter_by_user_role(self, rules: list, user_role: Any) -> list` - Line 114
- `async generate_alert(self, ...) -> dict[str, Any]` - Line 118
- `get_reporting_metrics(self) -> dict[str, Any]` - Line 124
- `reset_metrics(self) -> None` - Line 131
- `add_reporting_rule(self, rule: Any) -> None` - Line 135
- `remove_reporting_rule(self, rule_name: str) -> bool` - Line 139
- `async route_report_to_multiple_channels(self, report: dict[str, Any], channels: list[str]) -> dict[str, bool]` - Line 144
- `validate_reporting_rule(self, rule: Any) -> bool` - Line 150
- `get_applicable_rules(self, error: Exception, context: dict[str, Any]) -> list` - Line 154

### Implementation: `SecureReporter` ✅

**Inherits**: SecureErrorReporter
**Purpose**: Alias for backward compatibility
**Status**: Complete

### Implementation: `RateLimitConfig` ✅

**Purpose**: Rate limit configuration
**Status**: Complete

### Implementation: `SecurityThreat` ✅

**Purpose**: Security threat stub for backward compatibility
**Status**: Complete

### Implementation: `RateLimitResult` ✅

**Purpose**: Rate limit check result
**Status**: Complete

### Implementation: `SecuritySanitizer` ✅

**Purpose**: Simple security sanitizer
**Status**: Complete

**Implemented Methods:**
- `sanitize_context(self, context: dict[str, Any], sensitivity_level = None) -> dict[str, Any]` - Line 112
- `sanitize_error_message(self, message: str, sensitivity_level = None) -> str` - Line 116
- `sanitize_stack_trace(self, stack_trace: str, sensitivity_level = None) -> str` - Line 120
- `validate_context(self, context: dict[str, Any]) -> bool` - Line 124

### Implementation: `SecurityRateLimiter` ✅

**Purpose**: Simple rate limiter stub
**Status**: Complete

**Implemented Methods:**
- `is_allowed(self, key: str) -> bool` - Line 135
- `increment(self, key: str) -> None` - Line 139
- `async check_rate_limit(self, component: str, operation: str, context: dict[str, Any] | None = None) -> 'RateLimitResult'` - Line 144

### Implementation: `SensitivityLevel` ✅

**Inherits**: Enum
**Status**: Complete

### Implementation: `ErrorPattern` ✅

**Purpose**: Simple error pattern for backward compatibility
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict` - Line 176

### Implementation: `StateMonitorInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for state monitoring service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async validate_state_consistency(self, component: str = 'all') -> dict[str, Any]` - Line 33
- `async reconcile_state(self, component: str, discrepancies: list[dict[str, Any]]) -> bool` - Line 34
- `async start_monitoring(self) -> None` - Line 37
- `get_state_summary(self) -> dict[str, Any]` - Line 38

### Implementation: `ErrorHandlingService` ✅

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: Service layer for error handling operations
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 89
- `configure_dependencies(self, injector) -> None` - Line 128
- `async handle_error(self, ...) -> dict[str, Any]` - Line 150
- `async handle_global_error(self, ...) -> dict[str, Any]` - Line 309
- `async validate_state_consistency(self, component: str = 'all') -> dict[str, Any]` - Line 340
- `async reconcile_state_discrepancies(self, component: str, discrepancies: list[dict[str, Any]]) -> bool` - Line 372
- `async get_error_patterns(self) -> dict[str, Any]` - Line 405
- `async get_state_monitoring_status(self) -> dict[str, Any]` - Line 433
- `async get_error_handler_metrics(self) -> dict[str, Any]` - Line 458
- `async start_error_monitoring(self) -> None` - Line 542
- `async stop_error_monitoring(self) -> None` - Line 554
- `async handle_batch_errors(self, errors: list[tuple[Exception, str, str, dict[str, Any]]] | None) -> list[dict[str, Any]]` - Line 564
- `async start_monitoring(self) -> None` - Line 632
- `async cleanup_resources(self) -> None` - Line 656
- `async shutdown(self) -> None` - Line 699
- `async health_check(self) -> HealthCheckResult` - Line 725

### Implementation: `StateDataServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for state data operations abstracted from database details
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_balance_state(self) -> dict[str, Any]` - Line 39
- `async get_position_state(self) -> dict[str, Any]` - Line 40
- `async get_order_state(self) -> dict[str, Any]` - Line 41
- `async get_risk_state(self) -> dict[str, Any]` - Line 42
- `async reconcile_balance_discrepancies(self, discrepancies: list[dict[str, Any]]) -> bool` - Line 43
- `async reconcile_position_discrepancies(self, discrepancies: list[dict[str, Any]]) -> bool` - Line 44
- `async reconcile_order_discrepancies(self, discrepancies: list[dict[str, Any]]) -> bool` - Line 45
- `async reconcile_risk_discrepancies(self, discrepancies: list[dict[str, Any]]) -> bool` - Line 46
- `async add_missing_stop_losses(self) -> int` - Line 47
- `async get_order_details(self, order_id: str) -> dict[str, Any] | None` - Line 48

### Implementation: `RiskServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for risk management service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 54
- `async cleanup(self) -> None` - Line 55
- `async get_current_risk_metrics(self) -> dict[str, Any]` - Line 56
- `async get_current_positions(self) -> dict[str, dict[str, Any]]` - Line 57
- `async update_position(self, symbol: str, quantity: Decimal, side: str, exchange: str) -> None` - Line 58
- `async reduce_position(self, symbol: str, amount: Decimal) -> None` - Line 61
- `async reduce_portfolio_exposure(self, amount: Decimal) -> None` - Line 62
- `async adjust_leverage(self, reduction_factor: Decimal) -> None` - Line 63
- `async activate_emergency_shutdown(self, reason: str) -> None` - Line 64
- `async halt_trading(self, reason: str) -> None` - Line 65
- `async reduce_correlation_risk(self, excess: Decimal) -> None` - Line 66
- `async send_alert(self, level: str, message: str) -> None` - Line 67

### Implementation: `ExecutionServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for execution service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 73
- `async cleanup(self) -> None` - Line 74
- `async cancel_orders_by_symbol(self, symbol: str) -> None` - Line 75
- `async cancel_all_orders(self) -> None` - Line 76
- `async update_order_status(self, ...) -> None` - Line 77

### Implementation: `ExchangeServiceInterface` 🔧

**Inherits**: Protocol
**Purpose**: Protocol for exchange service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_account_balance(self) -> dict[str, dict[str, Any]]` - Line 85
- `async get_positions(self) -> list[Any]` - Line 86
- `async get_order(self, order_id: str) -> Any | None` - Line 87

### Implementation: `StateValidationResult` ✅

**Purpose**: Result of state validation check
**Status**: Complete

### Implementation: `StateMonitor` ✅

**Purpose**: Monitors and validates state consistency across system components
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector) -> None` - Line 155
- `async validate_state_consistency(self, component: str = 'all') -> StateValidationResult` - Line 208
- `async reconcile_state(self, component: str, discrepancies: list[dict[str, Any]]) -> bool` - Line 808
- `async start_monitoring(self) -> None` - Line 1267
- `get_state_summary(self) -> dict[str, Any]` - Line 1353
- `get_state_history(self, hours: int = 24) -> list[StateValidationResult]` - Line 1381
- `async reconcile_state(self, component: str, discrepancies: list[dict[str, Any]]) -> bool` - Line 1387

## COMPLETE API REFERENCE

### File: base.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `ErrorHandlerBase`

**Inherits**: ABC
**Purpose**: Base class for all error handlers using Chain of Responsibility pattern

```python
class ErrorHandlerBase(ABC):
    def __init__(self, next_handler: Optional['ErrorHandlerBase'] = None)  # Line 19
    def can_handle(self, error: Exception) -> bool  # Line 30
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> Any  # Line 43
    async def process(self, error: Exception, context: dict[str, Any] | None = None) -> Any  # Line 56
    def set_next(self, handler: 'ErrorHandlerBase') -> 'ErrorHandlerBase'  # Line 77
```

### File: connection_manager.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.logging import get_logger`

#### Class: `ConnectionState`

**Inherits**: Enum
**Purpose**: Connection state enumeration

```python
class ConnectionState(Enum):
```

#### Class: `ConnectionHealth`

**Inherits**: Enum
**Purpose**: Connection health status

```python
class ConnectionHealth(Enum):
```

#### Class: `ConnectionInfo`

**Purpose**: Basic connection information

```python
class ConnectionInfo:
```

#### Class: `ConnectionManager`

**Purpose**: Simple connection manager with reconnection support

```python
class ConnectionManager:
    def __init__(self, config: Config) -> None  # Line 49
    async def establish_connection(self, ...) -> bool  # Line 63
    async def close_connection(self, connection_id: str) -> bool  # Line 111
    async def reconnect_connection(self, connection_id: str) -> bool  # Line 136
    def get_connection_status(self, connection_id: str) -> dict[str, Any] | None  # Line 157
    def is_connection_healthy(self, connection_id: str) -> bool  # Line 172
    async def cleanup_resources(self) -> None  # Line 181
    async def __aenter__(self)  # Line 199
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 203
```

### File: context.py

**Key Imports:**
- `from src.core.base.factory import BaseFactory`
- `from src.core.exceptions import ErrorCategory`
- `from src.core.exceptions import ErrorSeverity`
- `from src.error_handling.security_sanitizer import SensitivityLevel`
- `from src.error_handling.security_sanitizer import get_security_sanitizer`

#### Class: `ErrorContext`

**Purpose**: Comprehensive error context for tracking, recovery, and analysis

```python
class ErrorContext:
    def __post_init__(self)  # Line 70
    def error_type(self) -> str  # Line 96
    def error_message(self) -> str  # Line 101
    def from_exception(cls, ...) -> 'ErrorContext'  # Line 106
    def function(self) -> str | None  # Line 169
    def function(self, value: str | None)  # Line 174
    def to_dict(self, ...) -> dict[str, Any]  # Line 178
    def to_legacy_dict(self, ...) -> dict[str, Any]  # Line 261
    def add_detail(self, key: str, value: Any) -> None  # Line 302
    def add_metadata(self, key: str, value: Any) -> None  # Line 306
    def increment_recovery_attempts(self) -> None  # Line 310
    def can_retry_recovery(self) -> bool  # Line 314
    def is_critical(self) -> bool  # Line 318
    def is_high_severity(self) -> bool  # Line 322
    def requires_escalation(self) -> bool  # Line 326
    def from_decorator_context(cls, ...) -> 'ErrorContext'  # Line 331
```

#### Class: `ErrorContextFactory`

**Inherits**: BaseFactory[ErrorContext]
**Purpose**: Factory for creating standardized error contexts with DI support

```python
class ErrorContextFactory(BaseFactory[ErrorContext]):
    def __init__(self, dependency_container: Any | None = None)  # Line 356
    def create_context_dict(self, error: Exception, **kwargs) -> dict[str, Any]  # Line 374
    def create(self, error: Exception, **kwargs) -> dict[str, Any]  # Line 437
    def create_context(self, context_type: str = 'standard', error: Exception | None = None, **kwargs) -> ErrorContext  # Line 450
    def create_from_frame(self, error: Exception, frame: Any | None = None, **kwargs) -> dict[str, Any]  # Line 469
    def create_minimal(self, error: Exception) -> dict[str, Any]  # Line 513
    def enrich_context(self, base_context: dict[str, Any], **additional) -> dict[str, Any]  # Line 529
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `ErrorDataTransformer`

**Purpose**: Handles consistent data transformation for error_handling module

```python
class ErrorDataTransformer:
    def transform_error_to_event_data(error, ...) -> dict[str, Any]  # Line 21
    def transform_context_to_event_data(context: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 55
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 89
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'error_handling') -> dict[str, Any]  # Line 112
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 148
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]  # Line 194
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 240
    def apply_cross_module_validation(cls, ...) -> dict[str, Any]  # Line 259
```

### File: decorators.py

**Key Imports:**
- `from src.core.exceptions import DatabaseConnectionError`
- `from src.core.exceptions import NetworkError`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`

#### Class: `FallbackStrategy`

**Inherits**: Enum
**Purpose**: Fallback strategies for error handling

```python
class FallbackStrategy(Enum):
```

#### Class: `RetryConfig`

**Purpose**: Simple retry configuration

```python
class RetryConfig:
```

#### Class: `CircuitBreakerConfig`

**Purpose**: Simple circuit breaker configuration

```python
class CircuitBreakerConfig:
```

#### Class: `FallbackConfig`

**Purpose**: Simple fallback configuration

```python
class FallbackConfig:
```

#### Functions:

```python
def error_handler(...)  # Line 70
def _should_circuit_break(func_name: str, config: CircuitBreakerConfig) -> bool  # Line 183
def _should_retry(error: Exception, config: RetryConfig | None, exceptions: tuple | None = None) -> bool  # Line 189
def _calculate_delay(attempt: int, config: RetryConfig | None) -> float  # Line 204
def _handle_fallback(config: FallbackConfig | None) -> Any  # Line 217
def with_retry(...)  # Line 235
def with_circuit_breaker(...)  # Line 266
def with_fallback(...)  # Line 290
def with_error_context(**context_kwargs)  # Line 317
def get_active_handler_count() -> int  # Line 364
def shutdown_all_error_handlers() -> None  # Line 369
def retry_with_backoff(...)  # Line 382
```

### File: di_registration.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.logging import get_logger`

#### Functions:

```python
def register_error_handling_services(injector, config: Config | None = None) -> None  # Line 17
def _configure_service_dependencies(injector) -> None  # Line 342
def get_error_handling_service(injector) -> Any  # Line 384
def _setup_global_error_handler(injector) -> None  # Line 397
def _register_error_handlers(factory) -> None  # Line 412
def configure_error_handling_di(injector, config: Config | None = None) -> None  # Line 443
```

### File: error_handler.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import DataError`
- `from src.core.exceptions import ErrorSeverity`
- `from src.core.exceptions import ExchangeError`
- `from src.core.exceptions import ExecutionError`

#### Class: `CircuitBreaker`

**Purpose**: Circuit breaker pattern implementation for preventing cascading failures

```python
class CircuitBreaker:
    def __init__(self, ...) -> None  # Line 74
    def call(self, func: Callable[Ellipsis, Any], *args: Any, **kwargs: Any) -> Any  # Line 86
    def should_transition_to_half_open(self) -> bool  # Line 115
    def open(self) -> None  # Line 122
    def is_open(self) -> bool  # Line 128
    def reset(self) -> None  # Line 132
    def threshold(self) -> int  # Line 140
```

#### Class: `ErrorPatternCache`

**Purpose**: Optimized error pattern storage with size limits and TTL

```python
class ErrorPatternCache:
    def __init__(self, ...) -> None  # Line 155
    def add_pattern(self, pattern: dict[str, Any] | Any) -> None  # Line 166
    def get_pattern(self, pattern_id: str) -> dict[str, Any] | None  # Line 182
    def get_all_patterns(self) -> dict[str, dict[str, Any]]  # Line 200
    def _should_cleanup(self) -> bool  # Line 205
    def _cleanup_expired(self) -> None  # Line 214
    def cleanup_expired(self) -> None  # Line 234
    def size(self) -> int  # Line 238
    def get_last_cleanup(self) -> datetime  # Line 242
```

#### Class: `ErrorHandler`

**Purpose**: Comprehensive error handling and recovery system with optimized memory management

```python
class ErrorHandler:
    def __init__(self, config: Config, sanitizer = None, rate_limiter = None) -> None  # Line 250
    def configure_dependencies(self, injector) -> None  # Line 301
    def _initialize_circuit_breakers(self) -> None  # Line 326
    def classify_error(self, error: Exception) -> ErrorSeverity  # Line 336
    def validate_module_boundary_input(self, data: dict[str, Any], source_module: str) -> dict[str, Any]  # Line 367
    def create_error_context(self, error: Exception, component: str, operation: str, **kwargs) -> 'ErrorContext'  # Line 434
    def _get_stack_trace(self) -> str  # Line 533
    def _validate_cross_module_boundary(self, data: dict[str, Any], source: str, operation: str) -> None  # Line 537
    async def handle_error(self, ...) -> bool  # Line 584
    def handle_error_sync(self, ...) -> bool  # Line 740
    def handle_error_batch(self, ...) -> list[bool]  # Line 839
    def _update_error_patterns(self, context: 'ErrorContext') -> None  # Line 916
    def _calculate_frequency(self, pattern: dict[str, Any]) -> Decimal  # Line 950
    def _log_performance_metrics(self) -> None  # Line 967
    def _get_circuit_breaker_key(self, context: 'ErrorContext') -> str | None  # Line 984
    def _raise_error(self, error: Exception) -> None  # Line 997
    async def _escalate_error(self, context: 'ErrorContext') -> None  # Line 1001
    def get_retry_policy(self, error_type: str) -> dict[str, Any]  # Line 1024
    def get_circuit_breaker_status(self) -> dict[str, str]  # Line 1028
    def get_error_patterns(self) -> dict[str, dict[str, Any]]  # Line 1032
    def get_memory_usage_stats(self) -> dict[str, Any]  # Line 1036
    async def cleanup_resources(self) -> None  # Line 1045
    def _get_sensitivity_level(self, severity: ErrorSeverity, component: str) -> SensitivityLevel  # Line 1096
    def validate_data_flow_consistency(self, data: dict[str, Any]) -> dict[str, Any]  # Line 1112
    async def shutdown(self) -> None  # Line 1161
```

#### Functions:

```python
def ensure_timezone_aware(dt: datetime) -> datetime  # Line 145
def create_error_handler_factory(config: Config | None = None, dependency_container: Any | None = None)  # Line 1177
def register_error_handler_with_di(injector, config: Config | None = None) -> None  # Line 1214
def error_handler_decorator(...) -> Callable[Ellipsis, Any]  # Line 1225
```

### File: factory.py

**Key Imports:**
- `from src.core.base.factory import BaseFactory`
- `from src.core.exceptions import CreationError`
- `from src.error_handling.base import ErrorHandlerBase`

#### Class: `ErrorHandlerFactory`

**Inherits**: BaseFactory[ErrorHandlerProtocol]
**Purpose**: Factory to create handlers without direct imports

```python
class ErrorHandlerFactory(BaseFactory[ErrorHandlerProtocol]):
    def __init__(self, dependency_container: Any | None = None)  # Line 42
    def register(cls, ...) -> None  # Line 51
    def create(cls, error_type: str, next_handler: ErrorHandlerBase | None = None, **kwargs) -> ErrorHandlerProtocol  # Line 84
    def list_handlers(cls) -> list[str]  # Line 135
    def set_dependency_container(cls, container: Any) -> None  # Line 140
    def _inject_common_dependencies(cls, config: dict[str, Any], handler_class = None) -> None  # Line 145
    def clear(cls) -> None  # Line 194
```

#### Class: `ErrorHandlerChain`

**Inherits**: BaseFactory[ErrorHandlerProtocol]
**Purpose**: Manages a chain of error handlers

```python
class ErrorHandlerChain(BaseFactory[ErrorHandlerProtocol]):
    def __init__(self, handlers: list[str] | None = None, dependency_container: Any | None = None)  # Line 209
    def build_chain(self, handler_types: list[str]) -> None  # Line 227
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> Any  # Line 248
    def add_handler(self, handler_type: str) -> None  # Line 264
    def create_default_chain(cls, dependency_container: Any | None = None) -> 'ErrorHandlerChain'  # Line 280
```

### File: global_handler.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.logging import get_logger`

#### Class: `GlobalErrorHandler`

**Inherits**: BaseService
**Purpose**: Global error handler that provides consistent error handling across the application

```python
class GlobalErrorHandler(BaseService):
    def __init__(self, ...)  # Line 31
    async def _do_start(self) -> None  # Line 65
    def _setup_error_handlers(self)  # Line 72
    def configure_dependencies(self, injector) -> None  # Line 107
    def register_database_handler(self)  # Line 126
    def register_error_callback(self, callback: Callable[[Exception, dict], None])  # Line 142
    def register_critical_callback(self, callback: Callable[[Exception, dict], None])  # Line 151
    def register_recovery_strategy(self, error_type: type, strategy: Callable)  # Line 160
    async def handle_error(self, ...) -> dict[str, Any]  # Line 170
    def handle_error_sync(self, ...) -> dict[str, Any]  # Line 283
    def handle_exception_hook(self, exc_type, exc_value, exc_traceback)  # Line 356
    def _log_error_handler_exception(self, task: asyncio.Task) -> None  # Line 399
    def install_global_handler(self)  # Line 413
    def error_handler_decorator(self, severity: str = 'error', reraise: bool = True, default_return: Any = None)  # Line 417
    def get_statistics(self) -> dict[str, Any]  # Line 483
    def reset_statistics(self)  # Line 499
```

#### Functions:

```python
def create_global_error_handler_factory(config: Any | None = None)  # Line 506
def register_global_error_handler_with_di(injector, config: Any | None = None) -> None  # Line 515
```

### File: handler_pool.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.error_handling.decorators import CircuitBreakerConfig`
- `from src.error_handling.decorators import FallbackConfig`
- `from src.error_handling.decorators import RetryConfig`

#### Class: `UniversalErrorHandler`

**Purpose**: Simple universal error handler stub for backward compatibility

```python
class UniversalErrorHandler:
    def __init__(self, ...)  # Line 25
    def handle_error(self, error: Exception, context: dict | None = None) -> bool  # Line 39
```

#### Class: `HandlerPool`

**Purpose**: Singleton pool for managing UniversalErrorHandler instances

```python
class HandlerPool:
    def __new__(cls) -> 'HandlerPool'  # Line 56
    def __init__(self)  # Line 65
    def _create_config_key(self, ...) -> str  # Line 77
    def get_handler(self, ...) -> UniversalErrorHandler  # Line 141
    def get_stats(self) -> dict[str, Any]  # Line 197
    def shutdown(self)  # Line 217
    def _is_event_loop_running(self) -> bool  # Line 255
    def _sync_shutdown_handlers(self)  # Line 265
    async def _async_shutdown_impl(self)  # Line 272
    async def async_shutdown(self)  # Line 279
    def clear(self)  # Line 289
```

#### Functions:

```python
def get_pooled_handler(...) -> UniversalErrorHandler  # Line 299
def get_pool_stats() -> dict[str, Any]  # Line 333
def shutdown_handler_pool()  # Line 339
async def async_shutdown_handler_pool()  # Line 345
```

### File: authentication.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.error_handling.base import ErrorHandlerBase`
- `from src.error_handling.security_rate_limiter import SecurityThreat`
- `from src.error_handling.security_rate_limiter import get_security_rate_limiter`
- `from src.error_handling.security_sanitizer import SensitivityLevel`

#### Class: `AuthenticationErrorHandler`

**Inherits**: ErrorHandlerBase
**Purpose**: Secure handler for authentication and authorization errors

```python
class AuthenticationErrorHandler(ErrorHandlerBase):
    def __init__(self, next_handler: ErrorHandlerBase | None = None)  # Line 47
    def can_handle(self, error: Exception) -> bool  # Line 86
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]  # Line 147
    def _extract_client_ip(self, context: dict[str, Any]) -> str | None  # Line 222
    def _extract_user_id(self, context: dict[str, Any]) -> str | None  # Line 235
    def _extract_session_id(self, context: dict[str, Any]) -> str | None  # Line 247
    def _extract_user_agent(self, context: dict[str, Any]) -> str | None  # Line 258
    def _sanitize_auth_context(self, context: dict[str, Any]) -> dict[str, Any]  # Line 270
    def _categorize_auth_error(self, error: Exception) -> str  # Line 286
    def _is_entity_blocked(self, entity: str | None) -> bool  # Line 364
    def _record_auth_failure(self, ...) -> None  # Line 377
    def _analyze_threat_patterns(self, client_ip: str | None, user_id: str | None) -> SecurityThreat  # Line 404
    def _calculate_progressive_delay(self, client_ip: str | None, user_id: str | None) -> float  # Line 446
    def _check_and_apply_blocks(self, client_ip: str | None, user_id: str | None) -> None  # Line 468
    def _create_blocked_response(self, error_category: str) -> dict[str, Any]  # Line 497
    def _create_rate_limited_response(self, rate_check) -> dict[str, Any]  # Line 509
    def _create_secure_auth_response(self, ...) -> dict[str, Any]  # Line 521
    def _get_retry_recommendations(self, error_category: str) -> dict[str, Any]  # Line 564
    def _get_security_headers(self) -> dict[str, str]  # Line 602
    async def _log_security_event(self, ...) -> None  # Line 613
    def _cleanup_old_attempts(self, cutoff: datetime) -> None  # Line 642
    def _is_valid_ip(self, ip: str) -> bool  # Line 654
    def _hash_identifier(self, identifier: str) -> str  # Line 664
    def _sanitize_user_agent(self, user_agent: str) -> str  # Line 669
    def get_security_stats(self) -> dict[str, Any]  # Line 692
    def reset_entity_failures(self, entity: str) -> bool  # Line 724
```

### File: database.py

**Key Imports:**
- `from src.core.exceptions import DatabaseConnectionError`
- `from src.core.exceptions import DatabaseError`
- `from src.core.exceptions import DatabaseQueryError`
- `from src.error_handling.base import ErrorHandlerBase`
- `from src.error_handling.security_validator import SensitivityLevel`

#### Class: `DatabaseErrorHandler`

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for database-related errors with secure sanitization

```python
class DatabaseErrorHandler(ErrorHandlerBase):
    def __init__(self, next_handler = None, sanitizer = None)  # Line 17
    def can_handle(self, error: Exception) -> bool  # Line 21
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]  # Line 60
```

### File: network.py

**Key Imports:**
- `from src.error_handling.base import ErrorHandlerBase`
- `from src.error_handling.security_validator import SensitivityLevel`
- `from src.utils.error_categorization import detect_rate_limiting`
- `from src.utils.error_handling_utils import create_recovery_response`
- `from src.utils.error_handling_utils import extract_retry_after_from_error`

#### Class: `NetworkErrorHandler`

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for network-related errors

```python
class NetworkErrorHandler(ErrorHandlerBase):
    def __init__(self, ...) -> None  # Line 21
    def can_handle(self, error: Exception) -> bool  # Line 42
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]  # Line 68
    def _calculate_retry_delay(self, retry_count: int) -> Decimal  # Line 119
```

#### Class: `RateLimitErrorHandler`

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for rate limit errors with secure sanitization

```python
class RateLimitErrorHandler(ErrorHandlerBase):
    def __init__(self, next_handler: ErrorHandlerBase | None = None, sanitizer = None) -> None  # Line 132
    def can_handle(self, error: Exception) -> bool  # Line 136
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]  # Line 141
```

### File: validation.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.error_handling.base import ErrorHandlerBase`
- `from src.error_handling.security_validator import SensitivityLevel`
- `from src.utils.error_categorization import detect_data_validation_error`
- `from src.utils.error_handling_utils import create_recovery_response`

#### Class: `ValidationErrorHandler`

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for validation errors with secure sanitization

```python
class ValidationErrorHandler(ErrorHandlerBase):
    def __init__(self, next_handler = None, sanitizer = None)  # Line 20
    def can_handle(self, error: Exception) -> bool  # Line 24
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]  # Line 50
```

#### Class: `DataValidationErrorHandler`

**Inherits**: ErrorHandlerBase
**Purpose**: Handler for data validation errors with secure sanitization

```python
class DataValidationErrorHandler(ErrorHandlerBase):
    def __init__(self, next_handler = None, sanitizer = None)  # Line 87
    def can_handle(self, error: Exception) -> bool  # Line 91
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]  # Line 96
```

### File: interfaces.py

**Key Imports:**
- `from src.core.base.interfaces import HealthCheckResult`

#### Class: `ErrorHandlingServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for error handling service layer

```python
class ErrorHandlingServiceInterface(Protocol):
    async def handle_error(self, ...) -> dict[str, Any]  # Line 18
    async def handle_global_error(self, ...) -> dict[str, Any]  # Line 29
    async def validate_state_consistency(self, component: str = 'all') -> dict[str, Any]  # Line 35
    async def get_error_patterns(self) -> dict[str, Any]  # Line 39
    async def health_check(self) -> HealthCheckResult  # Line 43
```

#### Class: `ErrorPatternAnalyticsInterface`

**Inherits**: Protocol
**Purpose**: Protocol for error pattern analytics service

```python
class ErrorPatternAnalyticsInterface(Protocol):
    def add_error_event(self, error_context: dict[str, Any]) -> None  # Line 52
    async def add_batch_error_events(self, error_contexts: list[dict[str, Any]]) -> None  # Line 56
    def get_pattern_summary(self) -> dict[str, Any]  # Line 60
    def get_correlation_summary(self) -> dict[str, Any]  # Line 64
    def get_trend_summary(self) -> dict[str, Any]  # Line 68
    async def cleanup(self) -> None  # Line 72
```

#### Class: `ErrorHandlerInterface`

**Inherits**: Protocol
**Purpose**: Protocol for error handler components

```python
class ErrorHandlerInterface(Protocol):
    async def handle_error(self, error: Exception, context: Any, recovery_strategy: Any | None = None) -> bool  # Line 81
    def classify_error(self, error: Exception) -> Any  # Line 90
    def create_error_context(self, error: Exception, component: str, operation: str, **kwargs) -> Any  # Line 94
    async def cleanup_resources(self) -> None  # Line 100
    async def shutdown(self) -> None  # Line 104
```

#### Class: `GlobalErrorHandlerInterface`

**Inherits**: Protocol
**Purpose**: Protocol for global error handler

```python
class GlobalErrorHandlerInterface(Protocol):
    async def handle_error(self, ...) -> dict[str, Any]  # Line 113
    def get_statistics(self) -> dict[str, Any]  # Line 122
```

#### Class: `ErrorHandlingServicePort`

**Inherits**: ABC
**Purpose**: Port interface for error handling service (hexagonal architecture)

```python
class ErrorHandlingServicePort(ABC):
    async def process_error(self, ...) -> dict[str, Any]  # Line 131
    async def analyze_error_patterns(self) -> dict[str, Any]  # Line 142
    async def validate_system_state(self, component: str = 'all') -> dict[str, Any]  # Line 147
```

#### Class: `ErrorHandlingRepositoryPort`

**Inherits**: ABC
**Purpose**: Repository port for error handling data persistence

```python
class ErrorHandlingRepositoryPort(ABC):
    async def store_error_event(self, error_data: dict[str, Any]) -> str  # Line 156
    async def retrieve_error_patterns(self, component: str | None = None, hours: int = 24) -> list[dict[str, Any]]  # Line 161
    async def update_error_statistics(self, stats: dict[str, Any]) -> None  # Line 168
```

### File: pattern_analytics.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.config import Config`

#### Class: `ErrorTrend`

**Purpose**: Error trend information

```python
class ErrorTrend:
```

#### Class: `ErrorPatternAnalytics`

**Inherits**: BaseService
**Purpose**: Simple error pattern analyzer

```python
class ErrorPatternAnalytics(BaseService):
    def __init__(self, config: Config)  # Line 30
    def add_error_event(self, error_context: dict[str, Any]) -> None  # Line 95
    def _check_patterns(self) -> None  # Line 118
    def get_pattern_summary(self) -> dict[str, Any]  # Line 146
    def get_recent_errors(self, hours: int = 1) -> list[dict[str, Any]]  # Line 164
    def get_correlation_summary(self) -> dict[str, Any]  # Line 170
    def get_trend_summary(self) -> dict[str, Any]  # Line 189
    def get_error_patterns(self) -> list[dict[str, Any]]  # Line 227
    async def add_batch_error_events(self, error_contexts: list[dict[str, Any]]) -> None  # Line 245
    def _transform_error_event_data(self, error_context: dict[str, Any]) -> dict[str, Any]  # Line 262
    async def cleanup(self) -> None  # Line 287
```

### File: propagation_utils.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `PropagationMethod`

**Inherits**: Enum
**Purpose**: Error propagation methods aligned with core patterns

```python
class PropagationMethod(Enum):
```

#### Class: `ProcessingStage`

**Inherits**: Enum
**Purpose**: Processing stages for error flow tracking

```python
class ProcessingStage(Enum):
```

#### Functions:

```python
def create_propagation_metadata(...) -> dict[str, Any]  # Line 39
def validate_propagation_data(data: dict[str, Any], source_module: str, target_module: str) -> dict[str, Any]  # Line 78
def transform_error_for_module(error_data: dict[str, Any], target_module: str, processing_mode: str = 'stream') -> dict[str, Any]  # Line 152
def _transform_for_core(data: dict[str, Any]) -> dict[str, Any]  # Line 197
def _transform_for_database(data: dict[str, Any]) -> dict[str, Any]  # Line 223
def _transform_for_monitoring(data: dict[str, Any]) -> dict[str, Any]  # Line 244
def _transform_for_exchanges(data: dict[str, Any]) -> dict[str, Any]  # Line 264
def get_propagation_chain(error_data: dict[str, Any]) -> list[dict[str, Any]]  # Line 286
def add_propagation_step(...) -> dict[str, Any]  # Line 299
```

### File: recovery.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.error_handling.security_rate_limiter import get_security_rate_limiter`
- `from src.error_handling.security_rate_limiter import record_recovery_failure`
- `from src.error_handling.security_sanitizer import SensitivityLevel`
- `from src.error_handling.security_sanitizer import get_security_sanitizer`

#### Class: `RecoveryStrategy`

**Inherits**: Protocol
**Purpose**: Protocol for all recovery strategies with proper type annotations

```python
class RecoveryStrategy(Protocol):
    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool  # Line 25
    async def recover(self, error: Exception, context: dict[str, Any]) -> Any  # Line 29
    def max_attempts(self) -> int  # Line 34
```

#### Class: `RetryRecovery`

**Inherits**: BaseComponent
**Purpose**: Retry recovery strategy with exponential backoff

```python
class RetryRecovery(BaseComponent):
    def __init__(self, ...) -> None  # Line 42
    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool  # Line 74
    async def recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]  # Line 85
    def max_attempts(self) -> int  # Line 136
```

#### Class: `CircuitBreakerRecovery`

**Inherits**: BaseComponent
**Purpose**: Circuit breaker recovery strategy

```python
class CircuitBreakerRecovery(BaseComponent):
    def __init__(self, ...) -> None  # Line 144
    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool  # Line 179
    async def recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]  # Line 184
    def max_attempts(self) -> int  # Line 267
```

#### Class: `FallbackRecovery`

**Inherits**: BaseComponent
**Purpose**: Fallback to alternative implementation

```python
class FallbackRecovery(BaseComponent):
    def __init__(self, fallback_function: Callable[Ellipsis, Any], max_attempts: int = 1) -> None  # Line 275
    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool  # Line 295
    async def recover(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]  # Line 299
    def max_attempts(self) -> int  # Line 427
```

### File: recovery_scenarios.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.utils.decorators import retry`
- `from src.utils.decorators import time_execution`

#### Class: `RecoveryDataServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for recovery data operations abstracted from database details

```python
class RecoveryDataServiceInterface(Protocol):
    async def get_recovery_context(self, scenario: str) -> dict[str, Any]  # Line 42
    async def execute_position_recovery(self, recovery_data: dict[str, Any]) -> bool  # Line 43
    async def execute_order_recovery(self, recovery_data: dict[str, Any]) -> bool  # Line 44
    async def log_recovery_action(self, action: str, details: dict[str, Any]) -> None  # Line 45
```

#### Class: `RiskServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk management service operations

```python
class RiskServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 51
    async def update_position(self, symbol: str, quantity: Any, side: str, exchange: str) -> None  # Line 52
    async def update_stop_loss(self, symbol: str, stop_loss_price: Any, exchange: str) -> None  # Line 55
```

#### Class: `CacheServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for cache service operations

```python
class CacheServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 61
    async def get(self, key: str) -> Any  # Line 62
    async def set(self, key: str, value: Any, expiry: int | None = None) -> None  # Line 63
```

#### Class: `StateServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for state service operations

```python
class StateServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 69
    async def create_checkpoint(self, component_name: str, state_data: dict[str, Any]) -> str  # Line 70
    async def get_latest_checkpoint(self, component_name: str) -> dict[str, Any] | None  # Line 71
    async def restore_checkpoint(self, checkpoint_id: str, component_name: str) -> None  # Line 72
```

#### Class: `BotServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for bot management service operations

```python
class BotServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 78
    async def pause_bot(self, component: str) -> None  # Line 79
    async def resume_bot(self, component: str) -> None  # Line 80
```

#### Class: `RecoveryScenario`

**Inherits**: BaseComponent
**Purpose**: Base class for recovery scenarios with service injection

```python
class RecoveryScenario(BaseComponent):
    def __init__(self, ...)  # Line 86
    def configure_dependencies(self, injector) -> None  # Line 127
    async def execute_recovery(self, context: Any) -> bool  # Line 153
```

#### Class: `PartialFillRecovery`

**Inherits**: RecoveryScenario
**Purpose**: Handle partially filled orders with intelligent recovery

```python
class PartialFillRecovery(RecoveryScenario):
    def __init__(self, ...)  # Line 178
    async def execute_recovery(self, context: dict[str, Any]) -> bool  # Line 202
    async def _cancel_order(self, order_id: str) -> None  # Line 233
    async def _log_partial_fill(self, order: dict[str, Any], filled_quantity: Decimal) -> None  # Line 255
    async def _reevaluate_signal(self, signal: dict[str, Any]) -> None  # Line 266
    async def _update_position(self, order: dict[str, Any], filled_quantity: Decimal) -> None  # Line 320
    async def _adjust_stop_loss(self, order: dict[str, Any], filled_quantity: Decimal) -> None  # Line 370
```

#### Class: `NetworkDisconnectionRecovery`

**Inherits**: RecoveryScenario
**Purpose**: Handle network disconnection with automatic reconnection

```python
class NetworkDisconnectionRecovery(RecoveryScenario):
    def __init__(self, ...)  # Line 453
    async def execute_recovery(self, context: dict[str, Any]) -> bool  # Line 474
    async def _switch_to_offline_mode(self, component: str) -> None  # Line 507
    async def _persist_pending_operations(self, component: str) -> None  # Line 543
    async def _try_reconnect(self, component: str) -> bool  # Line 587
    async def _reconcile_positions(self, component: str) -> None  # Line 636
    async def _reconcile_orders(self, component: str) -> None  # Line 663
    async def _verify_balances(self, component: str) -> None  # Line 695
    async def _switch_to_online_mode(self, component: str) -> None  # Line 734
    async def _enter_safe_mode(self, component: str) -> None  # Line 781
    async def _get_cached_positions(self, component: str) -> dict[str, Any]  # Line 820
    async def _fetch_exchange_positions(self, component: str) -> dict[str, Any]  # Line 825
    def _compare_positions(self, cached: dict, exchange: dict) -> list[dict]  # Line 830
    async def _resolve_discrepancies(self, component: str, discrepancies: list) -> None  # Line 846
    async def _get_cached_orders(self, component: str) -> list[dict]  # Line 852
    async def _fetch_exchange_orders(self, component: str) -> list[dict]  # Line 856
    async def _handle_missing_orders(self, component: str, orders: list) -> None  # Line 860
    async def _handle_unknown_orders(self, component: str, orders: list) -> None  # Line 866
    async def _get_cached_balances(self, component: str) -> dict[str, Decimal]  # Line 872
    async def _fetch_exchange_balances(self, component: str) -> dict[str, Decimal]  # Line 876
    async def _handle_balance_discrepancies(self, component: str, discrepancies: dict) -> None  # Line 880
    async def _cancel_all_pending_orders(self, component: str) -> None  # Line 886
    async def _close_all_positions(self, component: str) -> None  # Line 891
    async def _disable_trading(self, component: str) -> None  # Line 896
    async def _set_safe_mode_flag(self, component: str, enabled: bool) -> None  # Line 901
    async def _send_critical_alert(self, component: str, message: str) -> None  # Line 906
    async def _emergency_shutdown(self, component: str) -> None  # Line 911
```

#### Class: `ExchangeMaintenanceRecovery`

**Inherits**: RecoveryScenario
**Purpose**: Handle exchange maintenance with graceful degradation

```python
class ExchangeMaintenanceRecovery(RecoveryScenario):
    def __init__(self, ...)  # Line 920
    async def execute_recovery(self, context: dict[str, Any]) -> bool  # Line 937
    async def _detect_maintenance_schedule(self, exchange: str) -> None  # Line 958
    async def _redistribute_capital(self, exchange: str) -> None  # Line 969
    async def _pause_new_orders(self, exchange: str) -> None  # Line 986
```

#### Class: `DataFeedInterruptionRecovery`

**Inherits**: RecoveryScenario
**Purpose**: Handle data feed interruptions with fallback sources

```python
class DataFeedInterruptionRecovery(RecoveryScenario):
    def __init__(self, ...)  # Line 1006
    async def execute_recovery(self, context: dict[str, Any]) -> bool  # Line 1025
    async def _check_data_staleness(self, data_source: str) -> bool  # Line 1044
    async def _switch_to_fallback_source(self, data_source: str) -> None  # Line 1058
    async def _enable_conservative_trading(self, data_source: str) -> None  # Line 1079
```

#### Class: `OrderRejectionRecovery`

**Inherits**: RecoveryScenario
**Purpose**: Handle order rejections with intelligent retry

```python
class OrderRejectionRecovery(RecoveryScenario):
    def __init__(self, ...)  # Line 1112
    async def execute_recovery(self, context: dict[str, Any]) -> bool  # Line 1130
    async def _analyze_rejection_reason(self, order: dict[str, Any], rejection_reason: str) -> None  # Line 1149
    async def _adjust_order_parameters(self, order: dict[str, Any], rejection_reason: str) -> None  # Line 1191
```

#### Class: `APIRateLimitRecovery`

**Inherits**: RecoveryScenario
**Purpose**: Handle API rate limit violations with automatic throttling

```python
class APIRateLimitRecovery(RecoveryScenario):
    def __init__(self, ...)  # Line 1232
    async def execute_recovery(self, context: dict[str, Any]) -> bool  # Line 1249
```

#### Functions:

```python
def create_partial_fill_recovery_factory(config: Config | None = None)  # Line 1280
def create_network_disconnection_recovery_factory(config: Config | None = None)  # Line 1330
```

### File: secure_context_manager.py

**Key Imports:**
- `from src.utils.security_types import InformationLevel`
- `from src.utils.security_types import SecurityContext`
- `from src.utils.security_types import UserRole`

#### Class: `SecureErrorReport`

**Purpose**: Secure error report with filtered information

```python
class SecureErrorReport:
    def __init__(self, message: str = None, details: dict[str, Any] | None = None, **kwargs)  # Line 21
```

#### Class: `SecureErrorContextManager`

**Purpose**: Secure error context manager with role-based filtering

```python
class SecureErrorContextManager:
    def __init__(self, security_context: SecurityContext | None = None)  # Line 44
    def create_secure_report(self, error: Exception, **context) -> SecureErrorReport  # Line 47
```

#### Functions:

```python
async def secure_context(operation: str, **context)  # Line 55
def create_secure_context(**kwargs) -> dict[str, Any]  # Line 67
```

### File: secure_logging.py

**Key Imports:**
- `from src.utils.security_types import LogCategory`
- `from src.utils.security_types import LoggingConfig`
- `from src.utils.security_types import LogLevel`
- `from src.utils.security_types import SecureLogEntry`

#### Class: `SecureLogger`

**Purpose**: Secure logger with data sanitization capabilities

```python
class SecureLogger:
    def __init__(self, name: str = None, config: LoggingConfig | None = None)  # Line 63
    def log(self, level: LogLevel, message: str, context: dict[str, Any] | None = None) -> None  # Line 78
    def log_error(self, ...) -> None  # Line 84
    def log_security_event(self, ...) -> None  # Line 112
    def log_audit_trail(self, ...) -> None  # Line 136
    def sanitize_log_data(self, data: dict[str, Any]) -> dict[str, Any]  # Line 156
    def determine_information_level_from_context(self, context: Any = None) -> str  # Line 166
    def _determine_info_level(self, context: Any = None)  # Line 170
    def format_log_entry(self, entry: Any, format_type: str = 'json') -> str  # Line 188
    def _format_log_message(self, entry: Any) -> str  # Line 194
    def should_log(self, level: LogLevel) -> bool  # Line 218
    def _write_log_entry(self, entry: Any) -> str  # Line 222
    def _log_to_logger(self, entry: Any) -> None  # Line 236
    def _write_to_logger(self, logger, entry: Any, level: LogLevel) -> None  # Line 241
    def _write_audit_entry(self, entry: Any) -> None  # Line 248
    def log_performance_metrics(self, metrics: dict[str, Any]) -> None  # Line 253
    def serialize_log_entry(self, entry: Any) -> str  # Line 258
```

#### Functions:

```python
def sanitize_log_message(message: str) -> str  # Line 30
def should_log_error(error: Exception, context: dict[str, Any] | None = None) -> bool  # Line 38
def get_security_sanitizer()  # Line 43
def create_secure_logger(config = None)  # Line 55
```

### File: secure_pattern_analytics.py

**Key Imports:**
- `from src.utils.security_types import SeverityLevel`
- `from src.utils.security_types import ThreatType`

#### Class: `ErrorPattern`

**Purpose**: Enhanced error pattern with security analytics

```python
class ErrorPattern:
```

#### Class: `AnalyticsConfig`

**Purpose**: Analytics configuration for backward compatibility

```python
class AnalyticsConfig:
```

#### Class: `SecurePatternAnalytics`

**Purpose**: Simplified pattern analytics for backward compatibility

```python
class SecurePatternAnalytics:
    def __init__(self, config: AnalyticsConfig | None = None)  # Line 67
    def _start_background_analysis(self) -> None  # Line 77
    def record_error_event(self, ...) -> None  # Line 82
    def _create_sanitized_event(self, ...) -> dict[str, Any]  # Line 91
    def _create_error_signature(self, error: Exception, context: dict[str, Any]) -> str  # Line 106
    async def analyze_patterns(self) -> list[ErrorPattern]  # Line 116
    def _determine_severity(self, frequency: int) -> PatternSeverity  # Line 166
    def _detect_threats_in_pattern(self, pattern: ErrorPattern, events: list[dict[str, Any]]) -> list[ThreatType]  # Line 177
    def _calculate_risk_score(self, pattern: ErrorPattern, events: list[dict[str, Any]]) -> Decimal  # Line 200
    def _generate_recommendations(self) -> None  # Line 232
    async def get_patterns_summary(self, severity_filter: PatternSeverity | None = None) -> dict[str, Any]  # Line 237
```

### File: secure_reporting.py

**Key Imports:**
- `from src.utils.security_types import ErrorAlert`
- `from src.utils.security_types import ReportingChannel`
- `from src.utils.security_types import ReportingConfig`
- `from src.utils.security_types import ReportingMetrics`
- `from src.utils.security_types import ReportingRule`

#### Class: `MockContextManager`

**Purpose**: Mock context manager for testing

```python
class MockContextManager:
    def __init__(self)  # Line 43
    def get_context(self)  # Line 46
    def create_secure_report(self, error, security_context = None, error_context = None)  # Line 49
```

#### Class: `MockReportingRule`

**Purpose**: Mock reporting rule for testing

```python
class MockReportingRule:
    def __init__(self, name: str, channels: list[str], user_role: str)  # Line 61
```

#### Class: `SecureErrorReporter`

**Purpose**: Simple secure error reporter for backward compatibility

```python
class SecureErrorReporter:
    def __init__(self, config: ReportingConfig | None = None)  # Line 72
    def generate_report(self, report_type: ReportType, data: dict[str, Any]) -> dict[str, Any]  # Line 86
    def send_alert(self, ...) -> bool  # Line 91
    def create_error_report(self, error: Exception, security_context: Any, error_context: dict[str, Any]) -> Any  # Line 98
    async def submit_error_report(self, ...) -> bool  # Line 104
    def evaluate_reporting_rules(self, error: Exception, context: dict[str, Any]) -> list  # Line 110
    def filter_by_user_role(self, rules: list, user_role: Any) -> list  # Line 114
    async def generate_alert(self, ...) -> dict[str, Any]  # Line 118
    def get_reporting_metrics(self) -> dict[str, Any]  # Line 124
    def reset_metrics(self) -> None  # Line 131
    def add_reporting_rule(self, rule: Any) -> None  # Line 135
    def remove_reporting_rule(self, rule_name: str) -> bool  # Line 139
    async def route_report_to_multiple_channels(self, report: dict[str, Any], channels: list[str]) -> dict[str, bool]  # Line 144
    def validate_reporting_rule(self, rule: Any) -> bool  # Line 150
    def get_applicable_rules(self, error: Exception, context: dict[str, Any]) -> list  # Line 154
```

#### Class: `SecureReporter`

**Inherits**: SecureErrorReporter
**Purpose**: Alias for backward compatibility

```python
class SecureReporter(SecureErrorReporter):
```

#### Functions:

```python
def get_secure_context_manager()  # Line 20
def get_security_rate_limiter()  # Line 25
def get_security_sanitizer()  # Line 30
def generate_secure_report(data: dict[str, Any]) -> dict[str, Any]  # Line 35
def sanitize_report_data(data: dict[str, Any]) -> dict[str, Any]  # Line 165
```

### File: security_rate_limiter.py

#### Class: `RateLimitConfig`

**Purpose**: Rate limit configuration

```python
class RateLimitConfig:
```

#### Class: `SecurityThreat`

**Purpose**: Security threat stub for backward compatibility

```python
class SecurityThreat:
```

#### Functions:

```python
def record_recovery_failure(component: str, operation: str, error_severity: str, **kwargs) -> None  # Line 31
```

### File: security_validator.py

#### Class: `RateLimitResult`

**Purpose**: Rate limit check result

```python
class RateLimitResult:
```

#### Class: `SecuritySanitizer`

**Purpose**: Simple security sanitizer

```python
class SecuritySanitizer:
    def sanitize_context(self, context: dict[str, Any], sensitivity_level = None) -> dict[str, Any]  # Line 112
    def sanitize_error_message(self, message: str, sensitivity_level = None) -> str  # Line 116
    def sanitize_stack_trace(self, stack_trace: str, sensitivity_level = None) -> str  # Line 120
    def validate_context(self, context: dict[str, Any]) -> bool  # Line 124
```

#### Class: `SecurityRateLimiter`

**Purpose**: Simple rate limiter stub

```python
class SecurityRateLimiter:
    def __init__(self)  # Line 132
    def is_allowed(self, key: str) -> bool  # Line 135
    def increment(self, key: str) -> None  # Line 139
    async def check_rate_limit(self, component: str, operation: str, context: dict[str, Any] | None = None) -> 'RateLimitResult'  # Line 144
```

#### Class: `SensitivityLevel`

**Inherits**: Enum

```python
class SensitivityLevel(Enum):
```

#### Class: `ErrorPattern`

**Purpose**: Simple error pattern for backward compatibility

```python
class ErrorPattern:
    def __init__(self, pattern_id: str, pattern_type: str = 'frequency', **kwargs)  # Line 166
    def to_dict(self) -> dict  # Line 176
```

#### Functions:

```python
def sanitize_error_data(data: dict[str, Any]) -> dict[str, Any]  # Line 25
def sanitize_string_value(value: str) -> str  # Line 62
def validate_error_context(context: dict[str, Any]) -> bool  # Line 82
def get_security_sanitizer() -> 'SecuritySanitizer'  # Line 101
def get_security_rate_limiter() -> 'SecurityRateLimiter'  # Line 151
```

### File: service.py

**Key Imports:**
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.service import BaseService`
- `from src.core.config import Config`
- `from src.core.exceptions import DataValidationError`
- `from src.core.exceptions import ServiceError`

#### Class: `StateMonitorInterface`

**Inherits**: Protocol
**Purpose**: Protocol for state monitoring service

```python
class StateMonitorInterface(Protocol):
    async def validate_state_consistency(self, component: str = 'all') -> dict[str, Any]  # Line 33
    async def reconcile_state(self, component: str, discrepancies: list[dict[str, Any]]) -> bool  # Line 34
    async def start_monitoring(self) -> None  # Line 37
    def get_state_summary(self) -> dict[str, Any]  # Line 38
```

#### Class: `ErrorHandlingService`

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: Service layer for error handling operations

```python
class ErrorHandlingService(BaseService, ErrorPropagationMixin):
    def __init__(self, ...) -> None  # Line 49
    async def initialize(self) -> None  # Line 89
    def configure_dependencies(self, injector) -> None  # Line 128
    async def handle_error(self, ...) -> dict[str, Any]  # Line 150
    async def _handle_error_impl(self, ...) -> dict[str, Any]  # Line 190
    async def _notify_monitoring_of_error_handling(self, error_response: dict[str, Any]) -> None  # Line 274
    async def handle_global_error(self, ...) -> dict[str, Any]  # Line 309
    async def _handle_global_error_impl(self, ...) -> dict[str, Any]  # Line 327
    async def validate_state_consistency(self, component: str = 'all') -> dict[str, Any]  # Line 340
    async def reconcile_state_discrepancies(self, component: str, discrepancies: list[dict[str, Any]]) -> bool  # Line 372
    async def get_error_patterns(self) -> dict[str, Any]  # Line 405
    async def get_state_monitoring_status(self) -> dict[str, Any]  # Line 433
    async def get_error_handler_metrics(self) -> dict[str, Any]  # Line 458
    def _transform_error_context(self, context: dict[str, Any], component: str) -> dict[str, Any]  # Line 493
    def _validate_data_module_boundary(self, error: Exception, component: str, operation: str, context: dict[str, Any]) -> None  # Line 523
    async def start_error_monitoring(self) -> None  # Line 542
    async def stop_error_monitoring(self) -> None  # Line 554
    async def handle_batch_errors(self, errors: list[tuple[Exception, str, str, dict[str, Any]]] | None) -> list[dict[str, Any]]  # Line 564
    async def _handle_batch_errors_impl(self, errors: list[tuple[Exception, str, str, dict[str, Any]]] | None) -> list[dict[str, Any]]  # Line 575
    async def start_monitoring(self) -> None  # Line 632
    async def cleanup_resources(self) -> None  # Line 656
    async def shutdown(self) -> None  # Line 699
    async def _ensure_initialized(self) -> None  # Line 720
    async def health_check(self) -> HealthCheckResult  # Line 725
```

#### Functions:

```python
def create_error_handling_service(...) -> 'ErrorHandlingService'  # Line 766
```

### File: state_monitor.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decimal_utils import to_decimal`
- `from src.utils.decorators import retry`

#### Class: `StateDataServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for state data operations abstracted from database details

```python
class StateDataServiceInterface(Protocol):
    async def get_balance_state(self) -> dict[str, Any]  # Line 39
    async def get_position_state(self) -> dict[str, Any]  # Line 40
    async def get_order_state(self) -> dict[str, Any]  # Line 41
    async def get_risk_state(self) -> dict[str, Any]  # Line 42
    async def reconcile_balance_discrepancies(self, discrepancies: list[dict[str, Any]]) -> bool  # Line 43
    async def reconcile_position_discrepancies(self, discrepancies: list[dict[str, Any]]) -> bool  # Line 44
    async def reconcile_order_discrepancies(self, discrepancies: list[dict[str, Any]]) -> bool  # Line 45
    async def reconcile_risk_discrepancies(self, discrepancies: list[dict[str, Any]]) -> bool  # Line 46
    async def add_missing_stop_losses(self) -> int  # Line 47
    async def get_order_details(self, order_id: str) -> dict[str, Any] | None  # Line 48
```

#### Class: `RiskServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk management service operations

```python
class RiskServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 54
    async def cleanup(self) -> None  # Line 55
    async def get_current_risk_metrics(self) -> dict[str, Any]  # Line 56
    async def get_current_positions(self) -> dict[str, dict[str, Any]]  # Line 57
    async def update_position(self, symbol: str, quantity: Decimal, side: str, exchange: str) -> None  # Line 58
    async def reduce_position(self, symbol: str, amount: Decimal) -> None  # Line 61
    async def reduce_portfolio_exposure(self, amount: Decimal) -> None  # Line 62
    async def adjust_leverage(self, reduction_factor: Decimal) -> None  # Line 63
    async def activate_emergency_shutdown(self, reason: str) -> None  # Line 64
    async def halt_trading(self, reason: str) -> None  # Line 65
    async def reduce_correlation_risk(self, excess: Decimal) -> None  # Line 66
    async def send_alert(self, level: str, message: str) -> None  # Line 67
```

#### Class: `ExecutionServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for execution service operations

```python
class ExecutionServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 73
    async def cleanup(self) -> None  # Line 74
    async def cancel_orders_by_symbol(self, symbol: str) -> None  # Line 75
    async def cancel_all_orders(self) -> None  # Line 76
    async def update_order_status(self, ...) -> None  # Line 77
```

#### Class: `ExchangeServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for exchange service operations

```python
class ExchangeServiceInterface(Protocol):
    async def get_account_balance(self) -> dict[str, dict[str, Any]]  # Line 85
    async def get_positions(self) -> list[Any]  # Line 86
    async def get_order(self, order_id: str) -> Any | None  # Line 87
```

#### Class: `StateValidationResult`

**Purpose**: Result of state validation check

```python
class StateValidationResult:
```

#### Class: `StateMonitor`

**Purpose**: Monitors and validates state consistency across system components

```python
class StateMonitor:
    def __init__(self, ...) -> None  # Line 104
    def configure_dependencies(self, injector) -> None  # Line 155
    def _validate_service_availability(self) -> None  # Line 172
    def _safe_to_decimal(self, value: Any, field_name: str = 'value') -> Decimal  # Line 190
    async def validate_state_consistency(self, component: str = 'all') -> StateValidationResult  # Line 208
    async def _perform_consistency_check(self, check_name: str) -> dict[str, Any]  # Line 278
    async def _check_portfolio_balance_sync(self) -> dict[str, Any]  # Line 293
    async def _check_position_quantity_sync(self) -> dict[str, Any]  # Line 424
    async def _check_order_status_sync(self) -> dict[str, Any]  # Line 582
    async def _check_risk_limit_compliance(self) -> dict[str, Any]  # Line 616
    async def reconcile_state(self, component: str, discrepancies: list[dict[str, Any]]) -> bool  # Line 808
    async def _reconcile_portfolio_balances(self, discrepancies: list[dict[str, Any]]) -> bool  # Line 860
    async def _reconcile_position_quantities(self, discrepancies: list[dict[str, Any]]) -> bool  # Line 935
    async def _reconcile_order_statuses(self, discrepancies: list[dict[str, Any]]) -> bool  # Line 1022
    async def _reconcile_risk_limits(self, discrepancies: list[dict[str, Any]]) -> bool  # Line 1128
    async def start_monitoring(self) -> None  # Line 1267
    def get_state_summary(self) -> dict[str, Any]  # Line 1353
    def get_state_history(self, hours: int = 24) -> list[StateValidationResult]  # Line 1381
    async def reconcile_state(self, component: str, discrepancies: list[dict[str, Any]]) -> bool  # Line 1387
```

#### Functions:

```python
def create_state_monitor_factory(...)  # Line 1421
def register_state_monitor_with_di(injector, config: Config | None = None) -> None  # Line 1440
```

---
**Generated**: Complete reference for error_handling module
**Total Classes**: 78
**Total Functions**: 59