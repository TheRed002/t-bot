# MONITORING Module Reference

## INTEGRATION
**Dependencies**: core, error_handling, utils
**Used By**: strategies
**Provides**: AlertManager, DefaultAlertService, DefaultDashboardService, DefaultMetricsService, DefaultPerformanceService, GrafanaDashboardManager, MonitoringService, WebSocketManager
**Patterns**: Async Operations, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Decimal precision arithmetic
**Security**:
- Credential management
- Credential management
**Performance**:
- Parallel execution
- Retry mechanisms
- Caching
**Architecture**:
- AlertManager inherits from base architecture
- MonitoringDataTransformer inherits from base architecture
- MetricsCollector inherits from base architecture

## MODULE OVERVIEW
**Files**: 16 Python files
**Classes**: 60
**Functions**: 65

## COMPLETE API REFERENCE

## PROTOCOLS & INTERFACES

### Protocol: `MetricsCollectorProtocol`

**Purpose**: Protocol for metrics collection

**Required Methods:**
- `increment_counter(self, ...) -> None`
- `set_gauge(self, ...) -> None`
- `observe_histogram(self, ...) -> None`

### Protocol: `AlertManagerProtocol`

**Purpose**: Protocol for alert management

**Required Methods:**
- `async fire_alert(self, alert: Alert) -> None`
- `async resolve_alert(self, fingerprint: str) -> None`

### Protocol: `PerformanceProfilerProtocol`

**Purpose**: Protocol for performance profiling

**Required Methods:**
- `start_operation(self, name: str, metadata: dict[str, Any] | None = None) -> Any`
- `end_operation(self, operation_id: str) -> None`
- `record_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> None`

## IMPLEMENTATIONS

### Implementation: `AlertStatus` ✅

**Inherits**: Enum
**Purpose**: Alert status states
**Status**: Complete

### Implementation: `NotificationChannel` ✅

**Inherits**: Enum
**Purpose**: Notification channel types
**Status**: Complete

### Implementation: `AlertRule` ✅

**Purpose**: Definition of an alert rule
**Status**: Complete

**Implemented Methods:**

### Implementation: `Alert` ✅

**Purpose**: Alert instance with compatibility for database model
**Status**: Complete

**Implemented Methods:**
- `is_active(self) -> bool` - Line 226
- `to_db_model_dict(self) -> dict[str, Any]` - Line 230
- `from_db_model(cls, db_alert: dict[str, Any]) -> 'Alert'` - Line 251
- `duration(self) -> timedelta` - Line 278

### Implementation: `NotificationConfig` ✅

**Purpose**: Configuration for notification channels
**Status**: Complete

### Implementation: `EscalationPolicy` ✅

**Purpose**: Alert escalation policy
**Status**: Complete

### Implementation: `AlertManager` ✅

**Inherits**: BaseComponent
**Purpose**: Central alert manager for the T-Bot trading system
**Status**: Complete

**Implemented Methods:**
- `add_rule(self, rule: AlertRule) -> None` - Line 373
- `remove_rule(self, rule_name: str) -> bool` - Line 383
- `add_escalation_policy(self, policy: EscalationPolicy) -> None` - Line 399
- `add_suppression_rule(self, rule: dict[str, Any]) -> None` - Line 409
- `async fire_alert(self, alert: Alert) -> None` - Line 422
- `async resolve_alert(self, fingerprint: str) -> None` - Line 485
- `async acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool` - Line 553
- `get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]` - Line 580
- `get_alert_history(self, limit: int = 100) -> list[Alert]` - Line 595
- `get_alert_stats(self) -> dict[str, Any]` - Line 609
- `async start(self) -> None` - Line 633
- `async stop(self) -> None` - Line 643
- `async cleanup(self) -> None` - Line 667

### Implementation: `ErrorContext` ✅

**Purpose**: Local error context to avoid circular dependencies
**Status**: Complete

**Implemented Methods:**

### Implementation: `Panel` ✅

**Purpose**: Grafana panel configuration
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict[str, Any]` - Line 103

### Implementation: `Dashboard` ✅

**Purpose**: Grafana dashboard configuration
**Status**: Complete

**Implemented Methods:**
- `to_dict(self) -> dict[str, Any]` - Line 130

### Implementation: `DashboardBuilder` ✅

**Purpose**: Builder for creating Grafana dashboards for T-Bot
**Status**: Complete

**Implemented Methods:**
- `get_next_panel_id(self) -> int` - Line 161
- `create_trading_overview_dashboard(self) -> Dashboard` - Line 167
- `create_system_performance_dashboard(self) -> Dashboard` - Line 346
- `create_risk_management_dashboard(self) -> Dashboard` - Line 483
- `create_alerts_dashboard(self) -> Dashboard` - Line 631

### Implementation: `GrafanaDashboardManager` ✅

**Purpose**: Manager for Grafana dashboard operations
**Status**: Complete

**Implemented Methods:**
- `async deploy_all_dashboards(self) -> dict[str, bool]` - Line 732
- `async deploy_dashboard(self, dashboard: Dashboard) -> bool` - Line 763
- `export_dashboards_to_files(self, output_dir: str) -> None` - Line 844

### Implementation: `MonitoringDataTransformer` ✅

**Inherits**: BaseService
**Purpose**: Handles consistent data transformation for monitoring module following analytics patterns
**Status**: Complete

**Implemented Methods:**
- `transform_alert_to_event_data(cls, alert_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 24
- `transform_metric_to_event_data(cls, metric_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 57
- `transform_performance_to_event_data(cls, performance_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 87
- `transform_error_to_event_data(cls, ...) -> dict[str, Any]` - Line 121
- `validate_financial_precision(cls, data: dict[str, Any]) -> dict[str, Any]` - Line 144
- `ensure_boundary_fields(cls, data: dict[str, Any], source: str = 'monitoring') -> dict[str, Any]` - Line 165
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 201
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]` - Line 249
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 275
- `apply_cross_module_validation(cls, ...) -> dict[str, Any]` - Line 339

### Implementation: `ServiceBinding` ✅

**Purpose**: Represents a service binding in the DI container
**Status**: Complete

### Implementation: `DIContainer` ✅

**Purpose**: Dependency injection container for monitoring services
**Status**: Complete

**Implemented Methods:**
- `register(self, ...) -> None` - Line 128
- `resolve(self, interface: type[T] | str) -> T` - Line 173
- `clear(self) -> None` - Line 283

### Implementation: `MonitoringServiceFactory` ✅

**Purpose**: Factory for creating monitoring services with dependency injection
**Status**: Complete

**Implemented Methods:**
- `create_monitoring_service(self) -> 'MonitoringService'` - Line 42
- `create_metrics_service(self) -> MetricsServiceInterface` - Line 51
- `create_alert_service(self) -> AlertServiceInterface` - Line 55
- `create_performance_service(self) -> PerformanceServiceInterface` - Line 59
- `create_dashboard_service(self) -> DashboardServiceInterface` - Line 63

### Implementation: `FinancialPrecisionWarning` ✅

**Inherits**: UserWarning
**Purpose**: Warning raised when precision loss is detected in financial calculations
**Status**: Complete

### Implementation: `MonitoringServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for monitoring service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async start_monitoring(self) -> None` - Line 25
- `async stop_monitoring(self) -> None` - Line 30
- `async get_health_status(self) -> dict[str, Any]` - Line 35

### Implementation: `AlertServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for alert service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_alert(self, request: 'AlertRequest') -> str` - Line 44
- `async resolve_alert(self, fingerprint: str) -> bool` - Line 49
- `async acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool` - Line 54
- `get_active_alerts(self, severity: AlertSeverity | None = None) -> list['Alert']` - Line 59
- `get_alert_stats(self) -> dict[str, Any]` - Line 64
- `add_rule(self, rule: 'AlertRule') -> None` - Line 69
- `add_escalation_policy(self, policy: 'EscalationPolicy') -> None` - Line 74

### Implementation: `MetricsServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for metrics service operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `record_counter(self, request: 'MetricRequest') -> None` - Line 83
- `record_gauge(self, request: 'MetricRequest') -> None` - Line 88
- `record_histogram(self, request: 'MetricRequest') -> None` - Line 93
- `export_metrics(self) -> str` - Line 98

### Implementation: `PerformanceServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for performance monitoring service
**Status**: Abstract Base Class

**Implemented Methods:**
- `get_performance_summary(self) -> dict[str, Any]` - Line 107
- `record_order_execution(self, ...) -> None` - Line 112
- `record_market_data_processing(self, ...) -> None` - Line 125
- `get_latency_stats(self, metric_name: str) -> Optional['LatencyStats']` - Line 136
- `get_system_resource_stats(self) -> Optional['SystemResourceStats']` - Line 141

### Implementation: `DashboardServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for dashboard management service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async deploy_dashboard(self, dashboard: 'Dashboard') -> bool` - Line 150
- `async deploy_all_dashboards(self) -> dict[str, bool]` - Line 155
- `export_dashboards_to_files(self, output_dir: str) -> None` - Line 160
- `create_trading_overview_dashboard(self) -> 'Dashboard'` - Line 165
- `create_system_performance_dashboard(self) -> 'Dashboard'` - Line 170

### Implementation: `MetricType` ✅

**Inherits**: Enum
**Purpose**: Metric types for different components and Prometheus metrics
**Status**: Complete

### Implementation: `MetricDefinition` ✅

**Purpose**: Definition for a custom metric
**Status**: Complete

### Implementation: `MetricsCollector` ✅

**Inherits**: BaseComponent
**Purpose**: Central metrics collector for the T-Bot trading system
**Status**: Complete

**Implemented Methods:**
- `register_metric(self, definition: MetricDefinition) -> None` - Line 394
- `get_metric(self, name: str, namespace: str = 'tbot') -> Any | None` - Line 477
- `get_all_metrics(self) -> dict[str, Any]` - Line 491
- `increment_counter(self, ...) -> None` - Line 501
- `set_gauge(self, ...) -> None` - Line 617
- `observe_histogram(self, ...) -> None` - Line 670
- `time_operation(self, name: str, labels: dict[str, str] | None = None, namespace: str = 'tbot')` - Line 723
- `async start_collection(self) -> None` - Line 765
- `async stop_collection(self) -> None` - Line 775
- `async cleanup(self) -> None` - Line 794
- `async start(self) -> None` - Line 841
- `async stop(self) -> None` - Line 845
- `export_metrics(self) -> bytes` - Line 869
- `get_metrics_content_type(self) -> str` - Line 878

### Implementation: `TradingMetrics` ✅

**Inherits**: BaseComponent
**Purpose**: Trading-specific metrics collection
**Status**: Complete

**Implemented Methods:**
- `record_order(self, ...) -> None` - Line 1240
- `record_trade(self, ...) -> None` - Line 1316
- `update_portfolio_metrics(self, ...) -> None` - Line 1410
- `record_strategy_signal(self, strategy: str, signal_type: str, symbol: str) -> None` - Line 1485
- `record_pnl(self, ...) -> None` - Line 1497
- `record_latency(self, operation: str, exchange: str, latency_ms: float, **kwargs) -> None` - Line 1524
- `record_order_latency(self, exchange: str, latency: float, order_type: str | None = None) -> None` - Line 1539

### Implementation: `SystemMetrics` ✅

**Inherits**: BaseComponent
**Purpose**: System-level metrics collection
**Status**: Complete

**Implemented Methods:**
- `record_cpu_usage(self, cpu_percent: float) -> None` - Line 1621
- `record_memory_usage(self, used_mb: float, total_mb: float) -> None` - Line 1630
- `record_network_io(self, bytes_sent: float, bytes_received: float) -> None` - Line 1646
- `record_disk_usage(self, mount_point: str, usage_percent: float) -> None` - Line 1659
- `async collect_and_record_system_metrics(self) -> None` - Line 1670

### Implementation: `ExchangeMetrics` ✅

**Inherits**: BaseComponent
**Purpose**: Exchange-specific metrics collection
**Status**: Complete

**Implemented Methods:**
- `record_api_request(self, exchange: str, endpoint: str, status: str, response_time: float) -> None` - Line 1837
- `update_rate_limits(self, exchange: str, limit_type: str, remaining: int) -> None` - Line 1857
- `record_connection(self, success: bool, exchange: str | None = None) -> None` - Line 1869
- `record_health_check(self, success: bool, duration: float | None = None, exchange: str | None = None) -> None` - Line 1885
- `record_rate_limit_violation(self, endpoint: str, exchange: str | None = None) -> None` - Line 1909
- `record_rate_limit_check(self, endpoint: str, weight: int = 1, exchange: str | None = None) -> None` - Line 1921
- `record_order(self, order_type = None, side = None, success = None, **kwargs) -> None` - Line 1936
- `record_order_latency(self, exchange: str, latency: float, order_type: str | None = None) -> None` - Line 1999
- `async record_websocket_connection(self, exchange: str, connected: bool, error_type: str | None = None) -> None` - Line 2024
- `async record_websocket_message(self, ...) -> None` - Line 2049
- `async record_websocket_heartbeat(self, exchange: str, latency_seconds: float) -> None` - Line 2093
- `async record_websocket_reconnection(self, exchange: str, reason: str) -> None` - Line 2124

### Implementation: `RiskMetrics` ✅

**Inherits**: BaseComponent
**Purpose**: Risk management metrics collection with financial validation
**Status**: Complete

**Implemented Methods:**
- `record_var(self, confidence_level: float, timeframe: str, var_value: float) -> None` - Line 2162
- `record_drawdown(self, timeframe: str, drawdown_pct: float) -> None` - Line 2198
- `record_sharpe_ratio(self, timeframe: str, sharpe_ratio: float) -> None` - Line 2227
- `record_position_size(self, exchange: str, symbol: str, size_usd: float) -> None` - Line 2253

### Implementation: `PerformanceCategory` ✅

**Inherits**: Enum
**Purpose**: Performance monitoring categories
**Status**: Complete

### Implementation: `PerformanceMetric` ✅

**Purpose**: Individual performance metric
**Status**: Complete

### Implementation: `LatencyStats` ✅

**Purpose**: Latency statistics with percentiles
**Status**: Complete

**Implemented Methods:**
- `from_values(cls, values: list[float]) -> LatencyStats` - Line 162

### Implementation: `ThroughputStats` ✅

**Purpose**: Throughput statistics
**Status**: Complete

### Implementation: `SystemResourceStats` ✅

**Purpose**: System resource utilization statistics
**Status**: Complete

### Implementation: `GCStats` ✅

**Purpose**: Garbage collection statistics
**Status**: Complete

### Implementation: `PerformanceProfiler` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive performance profiler for high-frequency trading systems
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 477
- `async start_async(self) -> None` - Line 486
- `async stop_async(self) -> None` - Line 503
- `async stop(self) -> None` - Line 571
- `async cleanup(self) -> None` - Line 577
- `profile_function(self, ...)` - Line 724
- `async profile_async_function(self, ...)` - Line 768
- `record_order_execution(self, ...) -> None` - Line 811
- `record_market_data_processing(self, ...) -> None` - Line 932
- `async record_websocket_latency(self, exchange: str, message_type: str, latency_ms: Decimal) -> None` - Line 976
- `record_database_query(self, database: str, operation: str, table: str, query_time_ms: Decimal) -> None` - Line 1032
- `record_strategy_performance(self, ...) -> None` - Line 1080
- `get_latency_stats(self, metric_name: str) -> LatencyStats | None` - Line 1130
- `get_throughput_stats(self, metric_name: str) -> ThroughputStats | None` - Line 1152
- `get_system_resource_stats(self) -> SystemResourceStats | None` - Line 1192
- `get_gc_stats(self) -> GCStats | None` - Line 1213
- `get_performance_summary(self) -> dict[str, Any]` - Line 1232
- `get_metrics(self) -> dict[str, Any]` - Line 1282
- `reset_metrics(self) -> None` - Line 1286
- `clear_metrics(self) -> None` - Line 1297

### Implementation: `PerformanceMetrics` ✅

**Purpose**: Performance metrics data structure
**Status**: Complete

**Implemented Methods:**
- `add_metric(self, metric: PerformanceMetric) -> None` - Line 1899
- `get_metrics_by_category(self, category: PerformanceCategory) -> list[PerformanceMetric]` - Line 1903

### Implementation: `QueryMetrics` ✅

**Purpose**: Database query performance metrics
**Status**: Complete

**Implemented Methods:**
- `record_query(self, query: str, execution_time: float) -> None` - Line 1923
- `get_average_query_time(self) -> float` - Line 1927

### Implementation: `CacheMetrics` ✅

**Purpose**: Cache performance metrics
**Status**: Complete

**Implemented Methods:**
- `record_hit(self) -> None` - Line 1948
- `record_miss(self) -> None` - Line 1952
- `get_hit_rate(self) -> float` - Line 1956

### Implementation: `QueryOptimizer` ✅

**Purpose**: Query optimization utilities
**Status**: Complete

**Implemented Methods:**
- `analyze_query(self, query: str) -> dict[str, Any]` - Line 1971
- `optimize_query(self, query: str) -> str` - Line 1975
- `cache_query_plan(self, query: str, plan: dict[str, Any]) -> None` - Line 1979
- `get_cached_plan(self, query: str) -> dict[str, Any] | None` - Line 1983

### Implementation: `CacheOptimizer` ✅

**Purpose**: Cache optimization utilities
**Status**: Complete

**Implemented Methods:**
- `analyze_cache_performance(self) -> dict[str, Any]` - Line 1994
- `optimize_ttl(self, key: str) -> int` - Line 1998

### Implementation: `AlertRequest` ✅

**Purpose**: Request to create an alert
**Status**: Complete

### Implementation: `MetricRequest` ✅

**Purpose**: Request to record a metric
**Status**: Complete

### Implementation: `DefaultAlertService` ✅

**Inherits**: BaseService, AlertServiceInterface, ErrorPropagationMixin
**Purpose**: Default implementation of AlertService
**Status**: Complete

**Implemented Methods:**
- `async create_alert(self, request: AlertRequest) -> str` - Line 88
- `async resolve_alert(self, fingerprint: str) -> bool` - Line 166
- `async acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool` - Line 171
- `get_active_alerts(self, severity: AlertSeverity | None = None) -> list['Alert']` - Line 175
- `get_alert_stats(self) -> dict[str, Any]` - Line 179
- `add_rule(self, rule) -> None` - Line 183
- `add_escalation_policy(self, policy) -> None` - Line 187
- `async handle_error_event_from_error_handling(self, error_data: dict[str, Any]) -> str` - Line 191
- `async handle_batch_error_events_from_error_handling(self, error_events: list[dict[str, Any]]) -> list[str]` - Line 236

### Implementation: `DefaultMetricsService` ✅

**Inherits**: BaseService, MetricsServiceInterface, ErrorPropagationMixin
**Purpose**: Default implementation of MetricsService
**Status**: Complete

**Implemented Methods:**
- `record_counter(self, request: MetricRequest) -> None` - Line 321
- `record_gauge(self, request: MetricRequest) -> None` - Line 416
- `record_histogram(self, request: MetricRequest) -> None` - Line 465
- `export_metrics(self) -> str` - Line 514
- `record_error_pattern_metric(self, error_data: dict[str, Any]) -> None` - Line 518

### Implementation: `DefaultPerformanceService` ✅

**Inherits**: BaseService, PerformanceServiceInterface, ErrorPropagationMixin
**Purpose**: Default implementation of PerformanceService
**Status**: Complete

**Implemented Methods:**
- `get_performance_summary(self) -> dict[str, Any]` - Line 550
- `record_order_execution(self, ...) -> None` - Line 554
- `record_market_data_processing(self, ...) -> None` - Line 679
- `get_latency_stats(self, metric_name: str)` - Line 691
- `get_system_resource_stats(self)` - Line 695

### Implementation: `DefaultDashboardService` ✅

**Inherits**: BaseService, DashboardServiceInterface
**Purpose**: Default implementation of DashboardService
**Status**: Complete

**Implemented Methods:**
- `async deploy_dashboard(self, dashboard: 'Dashboard') -> bool` - Line 711
- `async deploy_all_dashboards(self) -> dict[str, bool]` - Line 715
- `export_dashboards_to_files(self, output_dir: str) -> None` - Line 719
- `create_trading_overview_dashboard(self) -> Dashboard` - Line 723
- `create_system_performance_dashboard(self) -> Dashboard` - Line 727

### Implementation: `MonitoringService` ✅

**Inherits**: BaseService
**Purpose**: Composite service for all monitoring operations
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self) -> None` - Line 773
- `async stop_monitoring(self) -> None` - Line 789
- `async get_health_status(self) -> dict[str, Any]` - Line 805
- `async health_check(self) -> dict[str, Any]` - Line 809

### Implementation: `OpenTelemetryConfig` ✅

**Purpose**: Configuration for OpenTelemetry setup
**Status**: Complete

**Implemented Methods:**

### Implementation: `TradingTracer` ✅

**Purpose**: Custom tracer for trading operations with financial context
**Status**: Complete

**Implemented Methods:**
- `trace_order_execution(self, ...) -> Any` - Line 268
- `trace_strategy_execution(self, ...) -> Any` - Line 306
- `async trace_risk_calculation(self, check_type: str, portfolio_value: Decimal) -> Any` - Line 331
- `trace_risk_check(self, ...) -> Any` - Line 373
- `trace_market_data_processing(self, ...) -> Any` - Line 398
- `add_trading_event(self, span: Any, event_type: str, attributes: dict[str, Any] | None = None) -> None` - Line 422
- `start_span(self, operation_name: str, attributes: dict[str, Any] | None = None) -> Any` - Line 438
- `cleanup(self) -> None` - Line 471

### Implementation: `MockStatus` ✅

**Status**: Complete

**Implemented Methods:**

### Implementation: `MockStatusCode` ✅

**Status**: Complete

### Implementation: `MockTrace` ✅

**Status**: Complete

**Implemented Methods:**
- `status(self, status_code: str, description: str | None = None) -> MockStatus` - Line 28
- `status_code(self) -> MockStatusCode` - Line 32

### Implementation: `WebSocketState` ✅

**Inherits**: Enum
**Purpose**: WebSocket connection states
**Status**: Complete

### Implementation: `WebSocketConfig` ✅

**Purpose**: WebSocket connection configuration
**Status**: Complete

### Implementation: `WebSocketMetrics` ✅

**Purpose**: WebSocket connection metrics
**Status**: Complete

### Implementation: `WebSocketManager` ✅

**Inherits**: BaseComponent
**Purpose**: Managed WebSocket connection with automatic reconnection and monitoring
**Status**: Complete

**Implemented Methods:**
- `state(self) -> WebSocketState` - Line 106
- `metrics(self) -> WebSocketMetrics` - Line 111
- `is_connected(self) -> bool` - Line 116
- `async connect(self, timeout: float | None = None) -> None` - Line 120
- `async disconnect(self, timeout: float = 5.0) -> None` - Line 150
- `async send_message(self, message: Any, timeout: float = 5.0) -> None` - Line 202
- `async connection_context(self) -> AsyncIterator['WebSocketManager']` - Line 236
- `async wait_connected(self, timeout: float | None = None) -> None` - Line 248

## COMPLETE API REFERENCE

### File: alerting.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.event_constants import AlertEvents`
- `from src.core.types import AlertSeverity`
- `from src.monitoring.config import ALERT_BACKGROUND_TASK_TIMEOUT`
- `from src.monitoring.config import ALERT_ESCALATION_CHECK_TIMEOUT`

#### Class: `AlertStatus`

**Inherits**: Enum
**Purpose**: Alert status states

```python
class AlertStatus(Enum):
```

#### Class: `NotificationChannel`

**Inherits**: Enum
**Purpose**: Notification channel types

```python
class NotificationChannel(Enum):
```

#### Class: `AlertRule`

**Purpose**: Definition of an alert rule

```python
class AlertRule:
    def __post_init__(self)  # Line 189
```

#### Class: `Alert`

**Purpose**: Alert instance with compatibility for database model

```python
class Alert:
    def __post_init__(self)  # Line 218
    def is_active(self) -> bool  # Line 226
    def to_db_model_dict(self) -> dict[str, Any]  # Line 230
    def from_db_model(cls, db_alert: dict[str, Any]) -> 'Alert'  # Line 251
    def duration(self) -> timedelta  # Line 278
```

#### Class: `NotificationConfig`

**Purpose**: Configuration for notification channels

```python
class NotificationConfig:
```

#### Class: `EscalationPolicy`

**Purpose**: Alert escalation policy

```python
class EscalationPolicy:
```

#### Class: `AlertManager`

**Inherits**: BaseComponent
**Purpose**: Central alert manager for the T-Bot trading system

```python
class AlertManager(BaseComponent):
    def __init__(self, config: NotificationConfig, error_handler = None)  # Line 336
    def add_rule(self, rule: AlertRule) -> None  # Line 373
    def remove_rule(self, rule_name: str) -> bool  # Line 383
    def add_escalation_policy(self, policy: EscalationPolicy) -> None  # Line 399
    def add_suppression_rule(self, rule: dict[str, Any]) -> None  # Line 409
    async def fire_alert(self, alert: Alert) -> None  # Line 422
    async def resolve_alert(self, fingerprint: str) -> None  # Line 485
    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool  # Line 553
    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]  # Line 580
    def get_alert_history(self, limit: int = 100) -> list[Alert]  # Line 595
    def get_alert_stats(self) -> dict[str, Any]  # Line 609
    async def start(self) -> None  # Line 633
    async def stop(self) -> None  # Line 643
    async def cleanup(self) -> None  # Line 667
    async def __aenter__(self)  # Line 687
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 692
    def _is_suppressed(self, alert: Alert) -> bool  # Line 696
    async def _processing_loop(self) -> None  # Line 717
    async def _send_notifications(self, alert: Alert) -> None  # Line 792
    async def _send_resolution_notifications(self, alert: Alert) -> None  # Line 851
    async def _send_email_notification(self, alert: Alert) -> None  # Line 881
    def _send_email_sync(self, subject: str, body: str) -> None  # Line 927
    async def _send_email_resolution(self, alert: Alert) -> None  # Line 974
    async def _send_slack_notification(self, alert: Alert) -> None  # Line 1003
    async def _send_slack_resolution(self, alert: Alert) -> None  # Line 1080
    async def _send_webhook_notification(self, alert: Alert) -> None  # Line 1147
    async def _send_webhook_resolution(self, alert: Alert) -> None  # Line 1243
    async def _send_discord_notification(self, alert: Alert) -> None  # Line 1304
    async def _send_discord_resolution(self, alert: Alert) -> None  # Line 1371
    async def _check_escalations(self) -> None  # Line 1433
    def _parse_duration_minutes(self, duration: str) -> int  # Line 1452
    async def _escalate_alert(self, alert: Alert) -> None  # Line 1510
```

#### Functions:

```python
def load_alert_rules_from_file(file_path: str) -> list[AlertRule]  # Line 1549
def get_alert_manager() -> Optional['AlertManager']  # Line 1609
def set_global_alert_manager(alert_manager: AlertManager) -> None  # Line 1630
```

### File: dashboards.py

#### Class: `ErrorContext`

**Purpose**: Local error context to avoid circular dependencies

```python
class ErrorContext:
    def __init__(self, ...)  # Line 58
```

#### Class: `Panel`

**Purpose**: Grafana panel configuration

```python
class Panel:
    def to_dict(self) -> dict[str, Any]  # Line 103
```

#### Class: `Dashboard`

**Purpose**: Grafana dashboard configuration

```python
class Dashboard:
    def to_dict(self) -> dict[str, Any]  # Line 130
```

#### Class: `DashboardBuilder`

**Purpose**: Builder for creating Grafana dashboards for T-Bot

```python
class DashboardBuilder:
    def __init__(self)  # Line 157
    def get_next_panel_id(self) -> int  # Line 161
    def create_trading_overview_dashboard(self) -> Dashboard  # Line 167
    def create_system_performance_dashboard(self) -> Dashboard  # Line 346
    def create_risk_management_dashboard(self) -> Dashboard  # Line 483
    def create_alerts_dashboard(self) -> Dashboard  # Line 631
```

#### Class: `GrafanaDashboardManager`

**Purpose**: Manager for Grafana dashboard operations

```python
class GrafanaDashboardManager:
    def __init__(self, grafana_url: str, api_key: str, error_handler = None)  # Line 709
    async def deploy_all_dashboards(self) -> dict[str, bool]  # Line 732
    async def deploy_dashboard(self, dashboard: Dashboard) -> bool  # Line 763
    def export_dashboards_to_files(self, output_dir: str) -> None  # Line 844
```

#### Functions:

```python
def get_logger(name: str)  # Line 25
def with_retry(max_attempts: int = 3, backoff_factor = None)  # Line 32
async def create_http_session(timeout: int = 30, connector_limit: int = 10, connector_limit_per_host: int = 5)  # Line 67
async def safe_session_close(session)  # Line 79
def create_default_dashboards() -> list[Dashboard]  # Line 874
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import AlertSeverity`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `MonitoringDataTransformer`

**Inherits**: BaseService
**Purpose**: Handles consistent data transformation for monitoring module following analytics patterns

```python
class MonitoringDataTransformer(BaseService):
    def transform_alert_to_event_data(cls, alert_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 24
    def transform_metric_to_event_data(cls, metric_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 57
    def transform_performance_to_event_data(cls, performance_data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 87
    def transform_error_to_event_data(cls, ...) -> dict[str, Any]  # Line 121
    def validate_financial_precision(cls, data: dict[str, Any]) -> dict[str, Any]  # Line 144
    def ensure_boundary_fields(cls, data: dict[str, Any], source: str = 'monitoring') -> dict[str, Any]  # Line 165
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 201
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]  # Line 249
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 275
    def apply_cross_module_validation(cls, ...) -> dict[str, Any]  # Line 339
```

### File: dependency_injection.py

**Key Imports:**
- `from src.core import get_logger`
- `from src.core.exceptions import MonitoringError`
- `from src.core.exceptions import ServiceError`

#### Class: `ServiceBinding`

**Purpose**: Represents a service binding in the DI container

```python
class ServiceBinding:
```

#### Class: `DIContainer`

**Purpose**: Dependency injection container for monitoring services

```python
class DIContainer:
    def __init__(self)  # Line 123
    def register(self, ...) -> None  # Line 128
    def resolve(self, interface: type[T] | str) -> T  # Line 173
    def _get_param_name(self, cls: type, param_type: type) -> str  # Line 274
    def clear(self) -> None  # Line 283
```

#### Functions:

```python
def create_factory(cls: type[T], *deps: type) -> Callable[[], T]  # Line 36
def get_monitoring_container() -> DIContainer  # Line 293
def setup_monitoring_dependencies() -> None  # Line 298
def create_metrics_collector() -> MetricsCollector  # Line 440
def create_alert_manager() -> AlertManager  # Line 448
def create_performance_profiler() -> PerformanceProfiler  # Line 461
def create_monitoring_service()  # Line 490
def create_alert_service()  # Line 525
def create_metrics_service()  # Line 541
def create_performance_service()  # Line 557
def create_dashboard_service()  # Line 573
def create_dashboard_manager() -> GrafanaDashboardManager  # Line 592
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`
- `from src.monitoring.alerting import AlertManager`
- `from src.monitoring.alerting import NotificationConfig`
- `from src.monitoring.interfaces import AlertServiceInterface`

#### Functions:

```python
def register_monitoring_services(injector: DependencyInjector) -> None  # Line 26
```

### File: factory.py

**Key Imports:**
- `from src.monitoring.interfaces import AlertServiceInterface`
- `from src.monitoring.interfaces import DashboardServiceInterface`
- `from src.monitoring.interfaces import MetricsServiceInterface`
- `from src.monitoring.interfaces import PerformanceServiceInterface`

#### Class: `MonitoringServiceFactory`

**Purpose**: Factory for creating monitoring services with dependency injection

```python
class MonitoringServiceFactory:
    def __init__(self, injector = None)  # Line 24
    def create_monitoring_service(self) -> 'MonitoringService'  # Line 42
    def create_metrics_service(self) -> MetricsServiceInterface  # Line 51
    def create_alert_service(self) -> AlertServiceInterface  # Line 55
    def create_performance_service(self) -> PerformanceServiceInterface  # Line 59
    def create_dashboard_service(self) -> DashboardServiceInterface  # Line 63
```

#### Functions:

```python
def create_default_monitoring_service(injector = None) -> 'MonitoringService'  # Line 68
```

### File: financial_precision.py

**Key Imports:**
- `from src.core import get_logger`
- `from src.core.exceptions import ValidationError`

#### Class: `FinancialPrecisionWarning`

**Inherits**: UserWarning
**Purpose**: Warning raised when precision loss is detected in financial calculations

```python
class FinancialPrecisionWarning(UserWarning):
```

#### Functions:

```python
def safe_decimal_to_float(...) -> float  # Line 44
def convert_financial_batch(...) -> dict[str, float]  # Line 111
def validate_financial_range(...) -> None  # Line 145
def detect_precision_requirements(value: Decimal, metric_name: str) -> tuple[int, bool]  # Line 172
def get_recommended_precision(metric_name: str) -> int  # Line 230
```

### File: financial_validation.py

**Key Imports:**
- `from src.monitoring.financial_precision import _FINANCIAL_DECIMAL_CONTEXT`

#### Functions:

```python
def validate_price(price: float | int | Decimal, symbol: str = 'UNKNOWN') -> Decimal  # Line 33
def validate_quantity(quantity: float | int | Decimal, symbol: str = 'UNKNOWN') -> Decimal  # Line 65
def validate_pnl_usd(pnl_usd: float | int | Decimal, context: str = '') -> Decimal  # Line 97
def validate_volume_usd(volume_usd: float | int | Decimal, context: str = '') -> Decimal  # Line 123
def validate_slippage_bps(slippage_bps: float | int | Decimal, context: str = '') -> Decimal  # Line 152
def validate_execution_time(execution_time: float | int | Decimal, context: str = '') -> Decimal  # Line 181
def validate_var(...) -> Decimal  # Line 210
def validate_drawdown_percent(drawdown_pct: float | int | Decimal, context: str = '') -> Decimal  # Line 246
def validate_sharpe_ratio(sharpe_ratio: float | int | Decimal, context: str = '') -> Decimal  # Line 278
def validate_portfolio_value(value_usd: float | int | Decimal, context: str = '') -> Decimal  # Line 309
def validate_timeframe(timeframe: str) -> str  # Line 338
def calculate_pnl_percentage(pnl_usd: float | Decimal, portfolio_value: float | Decimal) -> Decimal  # Line 378
def validate_position_size_usd(size_usd: float | int | Decimal, context: str = '') -> Decimal  # Line 411
```

### File: interfaces.py

**Key Imports:**
- `from src.core.types import AlertSeverity`

#### Class: `MonitoringServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for monitoring service operations

```python
class MonitoringServiceInterface(ABC):
    async def start_monitoring(self) -> None  # Line 25
    async def stop_monitoring(self) -> None  # Line 30
    async def get_health_status(self) -> dict[str, Any]  # Line 35
```

#### Class: `AlertServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for alert service operations

```python
class AlertServiceInterface(ABC):
    async def create_alert(self, request: 'AlertRequest') -> str  # Line 44
    async def resolve_alert(self, fingerprint: str) -> bool  # Line 49
    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool  # Line 54
    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list['Alert']  # Line 59
    def get_alert_stats(self) -> dict[str, Any]  # Line 64
    def add_rule(self, rule: 'AlertRule') -> None  # Line 69
    def add_escalation_policy(self, policy: 'EscalationPolicy') -> None  # Line 74
```

#### Class: `MetricsServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for metrics service operations

```python
class MetricsServiceInterface(ABC):
    def record_counter(self, request: 'MetricRequest') -> None  # Line 83
    def record_gauge(self, request: 'MetricRequest') -> None  # Line 88
    def record_histogram(self, request: 'MetricRequest') -> None  # Line 93
    def export_metrics(self) -> str  # Line 98
```

#### Class: `PerformanceServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for performance monitoring service

```python
class PerformanceServiceInterface(ABC):
    def get_performance_summary(self) -> dict[str, Any]  # Line 107
    def record_order_execution(self, ...) -> None  # Line 112
    def record_market_data_processing(self, ...) -> None  # Line 125
    def get_latency_stats(self, metric_name: str) -> Optional['LatencyStats']  # Line 136
    def get_system_resource_stats(self) -> Optional['SystemResourceStats']  # Line 141
```

#### Class: `DashboardServiceInterface`

**Inherits**: ABC
**Purpose**: Interface for dashboard management service

```python
class DashboardServiceInterface(ABC):
    async def deploy_dashboard(self, dashboard: 'Dashboard') -> bool  # Line 150
    async def deploy_all_dashboards(self) -> dict[str, bool]  # Line 155
    def export_dashboards_to_files(self, output_dir: str) -> None  # Line 160
    def create_trading_overview_dashboard(self) -> 'Dashboard'  # Line 165
    def create_system_performance_dashboard(self) -> 'Dashboard'  # Line 170
```

### File: metrics.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import MonitoringError`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`
- `from src.core.types import OrderStatus`

#### Class: `MetricType`

**Inherits**: Enum
**Purpose**: Metric types for different components and Prometheus metrics

```python
class MetricType(Enum):
```

#### Class: `MetricDefinition`

**Purpose**: Definition for a custom metric

```python
class MetricDefinition:
```

#### Class: `MetricsCollector`

**Inherits**: BaseComponent
**Purpose**: Central metrics collector for the T-Bot trading system

```python
class MetricsCollector(BaseComponent):
    def __init__(self, ...)  # Line 341
    def register_metric(self, definition: MetricDefinition) -> None  # Line 394
    def get_metric(self, name: str, namespace: str = 'tbot') -> Any | None  # Line 477
    def get_all_metrics(self) -> dict[str, Any]  # Line 491
    def increment_counter(self, ...) -> None  # Line 501
    def set_gauge(self, ...) -> None  # Line 617
    def observe_histogram(self, ...) -> None  # Line 670
    def time_operation(self, name: str, labels: dict[str, str] | None = None, namespace: str = 'tbot')  # Line 723
    async def start_collection(self) -> None  # Line 765
    async def stop_collection(self) -> None  # Line 775
    async def cleanup(self) -> None  # Line 794
    async def __aenter__(self)  # Line 818
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 823
    async def _collection_loop(self) -> None  # Line 827
    async def start(self) -> None  # Line 841
    async def stop(self) -> None  # Line 845
    async def _collect_system_metrics(self) -> None  # Line 849
    def export_metrics(self) -> bytes  # Line 869
    def get_metrics_content_type(self) -> str  # Line 878
    def _register_error_handling_metrics(self) -> None  # Line 882
    def _register_system_monitoring_metrics(self) -> None  # Line 935
    def _register_analytics_metrics(self) -> None  # Line 983
```

#### Class: `TradingMetrics`

**Inherits**: BaseComponent
**Purpose**: Trading-specific metrics collection

```python
class TradingMetrics(BaseComponent):
    def __init__(self, collector: MetricsCollector)  # Line 1147
    def _initialize_metrics(self) -> None  # Line 1158
    def record_order(self, ...) -> None  # Line 1240
    def record_trade(self, ...) -> None  # Line 1316
    def update_portfolio_metrics(self, ...) -> None  # Line 1410
    def record_strategy_signal(self, strategy: str, signal_type: str, symbol: str) -> None  # Line 1485
    def record_pnl(self, ...) -> None  # Line 1497
    def record_latency(self, operation: str, exchange: str, latency_ms: float, **kwargs) -> None  # Line 1524
    def record_order_latency(self, exchange: str, latency: float, order_type: str | None = None) -> None  # Line 1539
```

#### Class: `SystemMetrics`

**Inherits**: BaseComponent
**Purpose**: System-level metrics collection

```python
class SystemMetrics(BaseComponent):
    def __init__(self, collector: MetricsCollector)  # Line 1570
    def _initialize_metrics(self) -> None  # Line 1581
    def record_cpu_usage(self, cpu_percent: float) -> None  # Line 1621
    def record_memory_usage(self, used_mb: float, total_mb: float) -> None  # Line 1630
    def record_network_io(self, bytes_sent: float, bytes_received: float) -> None  # Line 1646
    def record_disk_usage(self, mount_point: str, usage_percent: float) -> None  # Line 1659
    async def collect_and_record_system_metrics(self) -> None  # Line 1670
```

#### Class: `ExchangeMetrics`

**Inherits**: BaseComponent
**Purpose**: Exchange-specific metrics collection

```python
class ExchangeMetrics(BaseComponent):
    def __init__(self, collector: MetricsCollector)  # Line 1743
    def _initialize_metrics(self) -> None  # Line 1754
    def record_api_request(self, exchange: str, endpoint: str, status: str, response_time: float) -> None  # Line 1837
    def update_rate_limits(self, exchange: str, limit_type: str, remaining: int) -> None  # Line 1857
    def record_connection(self, success: bool, exchange: str | None = None) -> None  # Line 1869
    def record_health_check(self, success: bool, duration: float | None = None, exchange: str | None = None) -> None  # Line 1885
    def record_rate_limit_violation(self, endpoint: str, exchange: str | None = None) -> None  # Line 1909
    def record_rate_limit_check(self, endpoint: str, weight: int = 1, exchange: str | None = None) -> None  # Line 1921
    def record_order(self, order_type = None, side = None, success = None, **kwargs) -> None  # Line 1936
    def _record_basic_order_metrics(self, exchange: str, status, order_type, symbol: str) -> None  # Line 1989
    def record_order_latency(self, exchange: str, latency: float, order_type: str | None = None) -> None  # Line 1999
    async def record_websocket_connection(self, exchange: str, connected: bool, error_type: str | None = None) -> None  # Line 2024
    async def record_websocket_message(self, ...) -> None  # Line 2049
    async def record_websocket_heartbeat(self, exchange: str, latency_seconds: float) -> None  # Line 2093
    async def record_websocket_reconnection(self, exchange: str, reason: str) -> None  # Line 2124
    async def _safe_increment_counter(self, metric_name: str, labels: dict) -> None  # Line 2135
    async def _safe_observe_histogram(self, metric_name: str, value: float, labels: dict) -> None  # Line 2140
```

#### Class: `RiskMetrics`

**Inherits**: BaseComponent
**Purpose**: Risk management metrics collection with financial validation

```python
class RiskMetrics(BaseComponent):
    def __init__(self, collector: MetricsCollector)  # Line 2151
    def record_var(self, confidence_level: float, timeframe: str, var_value: float) -> None  # Line 2162
    def record_drawdown(self, timeframe: str, drawdown_pct: float) -> None  # Line 2198
    def record_sharpe_ratio(self, timeframe: str, sharpe_ratio: float) -> None  # Line 2227
    def record_position_size(self, exchange: str, symbol: str, size_usd: float) -> None  # Line 2253
    def _initialize_metrics(self) -> None  # Line 2278
```

#### Functions:

```python
def validate_null_handling(value: Any, allow_null: bool = False, field_name: str = 'value') -> Any  # Line 220
def validate_type_conversion(value: Any, target_type: type, field_name: str = 'value', strict: bool = True) -> Any  # Line 251
def get_metrics_collector() -> MetricsCollector  # Line 2327
def set_metrics_collector(collector: MetricsCollector) -> None  # Line 2355
def setup_prometheus_server(...) -> None  # Line 2368
```

### File: performance.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.types import OrderType`
- `from src.monitoring.alerting import Alert`
- `from src.monitoring.alerting import AlertManager`
- `from src.monitoring.alerting import AlertSeverity`

#### Class: `PerformanceCategory`

**Inherits**: Enum
**Purpose**: Performance monitoring categories

```python
class PerformanceCategory(Enum):
```

#### Class: `PerformanceMetric`

**Purpose**: Individual performance metric

```python
class PerformanceMetric:
```

#### Class: `LatencyStats`

**Purpose**: Latency statistics with percentiles

```python
class LatencyStats:
    def from_values(cls, values: list[float]) -> LatencyStats  # Line 162
```

#### Class: `ThroughputStats`

**Purpose**: Throughput statistics

```python
class ThroughputStats:
```

#### Class: `SystemResourceStats`

**Purpose**: System resource utilization statistics

```python
class SystemResourceStats:
```

#### Class: `GCStats`

**Purpose**: Garbage collection statistics

```python
class GCStats:
```

#### Class: `PerformanceProfiler`

**Inherits**: BaseComponent
**Purpose**: Comprehensive performance profiler for high-frequency trading systems

```python
class PerformanceProfiler(BaseComponent):
    def __init__(self, ...)  # Line 269
    def _register_metrics(self) -> None  # Line 359
    def _safe_lock(self)  # Line 468
    async def start(self) -> None  # Line 477
    async def start_async(self) -> None  # Line 486
    async def stop_async(self) -> None  # Line 503
    async def stop(self) -> None  # Line 571
    async def cleanup(self) -> None  # Line 577
    async def __aenter__(self)  # Line 600
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 605
    async def _monitoring_loop(self) -> None  # Line 611
    def profile_function(self, ...)  # Line 724
    async def profile_async_function(self, ...)  # Line 768
    def record_order_execution(self, ...) -> None  # Line 811
    def record_market_data_processing(self, ...) -> None  # Line 932
    async def record_websocket_latency(self, exchange: str, message_type: str, latency_ms: Decimal) -> None  # Line 976
    def record_database_query(self, database: str, operation: str, table: str, query_time_ms: Decimal) -> None  # Line 1032
    def record_strategy_performance(self, ...) -> None  # Line 1080
    def get_latency_stats(self, metric_name: str) -> LatencyStats | None  # Line 1130
    def get_throughput_stats(self, metric_name: str) -> ThroughputStats | None  # Line 1152
    def get_system_resource_stats(self) -> SystemResourceStats | None  # Line 1192
    def get_gc_stats(self) -> GCStats | None  # Line 1213
    def get_performance_summary(self) -> dict[str, Any]  # Line 1232
    def get_metrics(self) -> dict[str, Any]  # Line 1282
    def reset_metrics(self) -> None  # Line 1286
    def clear_metrics(self) -> None  # Line 1297
    def _get_memory_usage(self) -> int  # Line 1301
    def _calculate_throughput(self, metric_name: str) -> float  # Line 1311
    def _calculate_websocket_health(self, exchange: str) -> float  # Line 1350
    async def _collect_system_resources(self) -> None  # Line 1390
    async def _collect_gc_stats(self) -> None  # Line 1489
    async def _check_performance_thresholds(self) -> None  # Line 1544
    async def _detect_anomalies(self) -> None  # Line 1573
    async def _update_performance_baselines(self) -> None  # Line 1610
    async def _send_performance_alert(self, title: str, message: str, severity: AlertSeverity, labels: dict[str, str]) -> None  # Line 1670
    def _create_managed_alert_task(self, title: str, message: str, severity, labels: dict[str, str]) -> None  # Line 1693
    def _handle_managed_alert_task_completion(self, task: asyncio.Task) -> None  # Line 1714
    def _handle_alert_task_completion(self, task: asyncio.Task) -> None  # Line 1727
```

#### Class: `PerformanceMetrics`

**Purpose**: Performance metrics data structure

```python
class PerformanceMetrics:
    def __post_init__(self)  # Line 1895
    def add_metric(self, metric: PerformanceMetric) -> None  # Line 1899
    def get_metrics_by_category(self, category: PerformanceCategory) -> list[PerformanceMetric]  # Line 1903
```

#### Class: `QueryMetrics`

**Purpose**: Database query performance metrics

```python
class QueryMetrics:
    def __post_init__(self)  # Line 1919
    def record_query(self, query: str, execution_time: float) -> None  # Line 1923
    def get_average_query_time(self) -> float  # Line 1927
```

#### Class: `CacheMetrics`

**Purpose**: Cache performance metrics

```python
class CacheMetrics:
    def __post_init__(self)  # Line 1943
    def record_hit(self) -> None  # Line 1948
    def record_miss(self) -> None  # Line 1952
    def get_hit_rate(self) -> float  # Line 1956
```

#### Class: `QueryOptimizer`

**Purpose**: Query optimization utilities

```python
class QueryOptimizer:
    def __init__(self)  # Line 1967
    def analyze_query(self, query: str) -> dict[str, Any]  # Line 1971
    def optimize_query(self, query: str) -> str  # Line 1975
    def cache_query_plan(self, query: str, plan: dict[str, Any]) -> None  # Line 1979
    def get_cached_plan(self, query: str) -> dict[str, Any] | None  # Line 1983
```

#### Class: `CacheOptimizer`

**Purpose**: Cache optimization utilities

```python
class CacheOptimizer:
    def __init__(self)  # Line 1991
    def analyze_cache_performance(self) -> dict[str, Any]  # Line 1994
    def optimize_ttl(self, key: str) -> int  # Line 1998
```

#### Functions:

```python
def format_timestamp(dt: datetime | None) -> str  # Line 104
def profile_async(...)  # Line 1738
def profile_sync(...)  # Line 1759
def get_performance_profiler() -> PerformanceProfiler | None  # Line 1784
def set_global_profiler(profiler: PerformanceProfiler) -> None  # Line 1800
def initialize_performance_monitoring(config: dict[str, Any] | None = None, **kwargs) -> PerformanceProfiler  # Line 1810
```

### File: services.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ComponentError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import AlertSeverity`
- `from src.core.types import OrderType`

#### Class: `AlertRequest`

**Purpose**: Request to create an alert

```python
class AlertRequest:
```

#### Class: `MetricRequest`

**Purpose**: Request to record a metric

```python
class MetricRequest:
```

#### Class: `DefaultAlertService`

**Inherits**: BaseService, AlertServiceInterface, ErrorPropagationMixin
**Purpose**: Default implementation of AlertService

```python
class DefaultAlertService(BaseService, AlertServiceInterface, ErrorPropagationMixin):
    def __init__(self, alert_manager: 'AlertManager')  # Line 81
    async def create_alert(self, request: AlertRequest) -> str  # Line 88
    async def resolve_alert(self, fingerprint: str) -> bool  # Line 166
    async def acknowledge_alert(self, fingerprint: str, acknowledged_by: str) -> bool  # Line 171
    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list['Alert']  # Line 175
    def get_alert_stats(self) -> dict[str, Any]  # Line 179
    def add_rule(self, rule) -> None  # Line 183
    def add_escalation_policy(self, policy) -> None  # Line 187
    async def handle_error_event_from_error_handling(self, error_data: dict[str, Any]) -> str  # Line 191
    async def handle_batch_error_events_from_error_handling(self, error_events: list[dict[str, Any]]) -> list[str]  # Line 236
    def _transform_error_event_data(self, error_data: dict[str, Any]) -> dict[str, Any]  # Line 263
    def _transform_alert_request_data(self, request: AlertRequest) -> AlertRequest  # Line 282
```

#### Class: `DefaultMetricsService`

**Inherits**: BaseService, MetricsServiceInterface, ErrorPropagationMixin
**Purpose**: Default implementation of MetricsService

```python
class DefaultMetricsService(BaseService, MetricsServiceInterface, ErrorPropagationMixin):
    def __init__(self, metrics_collector: 'MetricsCollector')  # Line 315
    def record_counter(self, request: MetricRequest) -> None  # Line 321
    def record_gauge(self, request: MetricRequest) -> None  # Line 416
    def record_histogram(self, request: MetricRequest) -> None  # Line 465
    def export_metrics(self) -> str  # Line 514
    def record_error_pattern_metric(self, error_data: dict[str, Any]) -> None  # Line 518
```

#### Class: `DefaultPerformanceService`

**Inherits**: BaseService, PerformanceServiceInterface, ErrorPropagationMixin
**Purpose**: Default implementation of PerformanceService

```python
class DefaultPerformanceService(BaseService, PerformanceServiceInterface, ErrorPropagationMixin):
    def __init__(self, performance_profiler: 'PerformanceProfiler')  # Line 544
    def get_performance_summary(self) -> dict[str, Any]  # Line 550
    def record_order_execution(self, ...) -> None  # Line 554
    def record_market_data_processing(self, ...) -> None  # Line 679
    def get_latency_stats(self, metric_name: str)  # Line 691
    def get_system_resource_stats(self)  # Line 695
```

#### Class: `DefaultDashboardService`

**Inherits**: BaseService, DashboardServiceInterface
**Purpose**: Default implementation of DashboardService

```python
class DefaultDashboardService(BaseService, DashboardServiceInterface):
    def __init__(self, dashboard_manager: 'GrafanaDashboardManager')  # Line 704
    async def deploy_dashboard(self, dashboard: 'Dashboard') -> bool  # Line 711
    async def deploy_all_dashboards(self) -> dict[str, bool]  # Line 715
    def export_dashboards_to_files(self, output_dir: str) -> None  # Line 719
    def create_trading_overview_dashboard(self) -> Dashboard  # Line 723
    def create_system_performance_dashboard(self) -> Dashboard  # Line 727
```

#### Class: `MonitoringService`

**Inherits**: BaseService
**Purpose**: Composite service for all monitoring operations

```python
class MonitoringService(BaseService):
    def __init__(self, ...)  # Line 735
    async def start_monitoring(self) -> None  # Line 773
    async def stop_monitoring(self) -> None  # Line 789
    async def get_health_status(self) -> dict[str, Any]  # Line 805
    async def health_check(self) -> dict[str, Any]  # Line 809
```

### File: telemetry.py

**Key Imports:**
- `from src.core.exceptions import MonitoringError`

#### Class: `OpenTelemetryConfig`

**Purpose**: Configuration for OpenTelemetry setup

```python
class OpenTelemetryConfig:
    def __post_init__(self) -> None  # Line 240
```

#### Class: `TradingTracer`

**Purpose**: Custom tracer for trading operations with financial context

```python
class TradingTracer:
    def __init__(self, tracer: Any) -> None  # Line 254
    def trace_order_execution(self, ...) -> Any  # Line 268
    def trace_strategy_execution(self, ...) -> Any  # Line 306
    async def trace_risk_calculation(self, check_type: str, portfolio_value: Decimal) -> Any  # Line 331
    def trace_risk_check(self, ...) -> Any  # Line 373
    def trace_market_data_processing(self, ...) -> Any  # Line 398
    def add_trading_event(self, span: Any, event_type: str, attributes: dict[str, Any] | None = None) -> None  # Line 422
    def start_span(self, operation_name: str, attributes: dict[str, Any] | None = None) -> Any  # Line 438
    def cleanup(self) -> None  # Line 471
```

#### Functions:

```python
def get_error_handler_fallback()  # Line 128
def get_monitoring_logger(name: str)  # Line 190
def setup_telemetry(config: OpenTelemetryConfig) -> TradingTracer  # Line 546
def _setup_auto_instrumentation(config: OpenTelemetryConfig) -> None  # Line 692
def get_tracer(name: str = 'tbot-trading')  # Line 770
def instrument_fastapi(app: Any, config: OpenTelemetryConfig) -> None  # Line 792
def trace_async_function(operation_name: str, attributes: dict[str, Any] | None = None) -> Callable  # Line 839
def trace_function(operation_name: str, attributes: dict[str, Any] | None = None) -> Callable  # Line 880
async def trace_async_context(operation_name: str, attributes: dict[str, Any] | None = None) -> Any  # Line 922
def get_trading_tracer() -> TradingTracer | None  # Line 948
def set_global_trading_tracer(tracer: TradingTracer) -> None  # Line 958
```

### File: trace_wrapper.py

#### Class: `MockStatus`


```python
class MockStatus:
    def __init__(self, status_code: str, description: str | None = None) -> None  # Line 13
```

#### Class: `MockStatusCode`


```python
class MockStatusCode:
```

#### Class: `MockTrace`


```python
class MockTrace:
    def __init__(self) -> None  # Line 25
    def status(self, status_code: str, description: str | None = None) -> MockStatus  # Line 28
    def status_code(self) -> MockStatusCode  # Line 32
```

### File: websocket_helpers.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import MonitoringError`

#### Class: `WebSocketState`

**Inherits**: Enum
**Purpose**: WebSocket connection states

```python
class WebSocketState(Enum):
```

#### Class: `WebSocketConfig`

**Purpose**: WebSocket connection configuration

```python
class WebSocketConfig:
```

#### Class: `WebSocketMetrics`

**Purpose**: WebSocket connection metrics

```python
class WebSocketMetrics:
```

#### Class: `WebSocketManager`

**Inherits**: BaseComponent
**Purpose**: Managed WebSocket connection with automatic reconnection and monitoring

```python
class WebSocketManager(BaseComponent):
    def __init__(self, ...)  # Line 69
    def state(self) -> WebSocketState  # Line 106
    def metrics(self) -> WebSocketMetrics  # Line 111
    def is_connected(self) -> bool  # Line 116
    async def connect(self, timeout: float | None = None) -> None  # Line 120
    async def disconnect(self, timeout: float = 5.0) -> None  # Line 150
    async def send_message(self, message: Any, timeout: float = 5.0) -> None  # Line 202
    async def connection_context(self) -> AsyncIterator['WebSocketManager']  # Line 236
    async def wait_connected(self, timeout: float | None = None) -> None  # Line 248
    async def _establish_connection(self) -> None  # Line 263
    async def _simulate_connection(self) -> None  # Line 295
    async def _heartbeat_loop(self) -> None  # Line 300
    async def _send_heartbeat(self) -> None  # Line 333
    async def _wait_heartbeat_response(self) -> None  # Line 338
    async def _handle_connection_error(self, error_msg: str) -> None  # Line 343
    async def _attempt_reconnection(self, reason: str) -> None  # Line 359
    async def _set_error_state(self, error_msg: str) -> None  # Line 386
    async def _process_message_queue(self) -> None  # Line 394
    async def _message_processing_loop(self) -> None  # Line 415
```

#### Functions:

```python
async def managed_websocket(...) -> AsyncIterator[WebSocketManager]  # Line 436
async def with_websocket_timeout(coro, timeout: float, error_msg: str = 'WebSocket operation timed out')  # Line 460
def create_websocket_config(url: str, **kwargs) -> WebSocketConfig  # Line 470
```

---
**Generated**: Complete reference for monitoring module
**Total Classes**: 60
**Total Functions**: 65