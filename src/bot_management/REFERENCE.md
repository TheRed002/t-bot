# BOT_MANAGEMENT Module Reference

## INTEGRATION
**Dependencies**: capital_management, core, database, error_handling, exchanges, execution, monitoring, risk_management, state, strategies, utils
**Used By**: None
**Provides**: BotCoordinationService, BotInstanceService, BotLifecycleService, BotManagementController, BotMonitoringService, BotResourceService, BotService, IBotCoordinationService, IBotInstanceService, IBotLifecycleService, IBotMonitoringService, IResourceManagementService, ResourceManager
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
- BotMonitor inherits from base architecture
- BotManagementController inherits from base architecture
- BotCoordinationService inherits from base architecture

## MODULE OVERVIEW
**Files**: 18 Python files
**Classes**: 23
**Functions**: 12

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `BotCoordinator` ✅

**Purpose**: Inter-bot communication and coordination manager
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 120
- `async stop(self) -> None` - Line 142
- `async register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None` - Line 182
- `async unregister_bot(self, bot_id: str) -> None` - Line 224
- `async report_position_change(self, ...) -> dict[str, Any]` - Line 260
- `async share_signal(self, ...) -> int` - Line 323
- `async get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]` - Line 408
- `async check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]` - Line 450
- `async get_coordination_summary(self) -> dict[str, Any]` - Line 583
- `async update_bot_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None` - Line 950
- `async remove_bot_position(self, bot_id: str, symbol: str) -> None` - Line 959
- `async check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]` - Line 965
- `async coordinate_bot_actions(self, action_data: dict[str, Any]) -> dict[str, Any]` - Line 990
- `async analyze_bot_interactions(self) -> dict[str, Any]` - Line 998
- `async optimize_coordination(self) -> dict[str, Any]` - Line 1008
- `async emergency_coordination(self, emergency_type: str, action: str) -> None` - Line 1012

### Implementation: `BotInstance` ✅

**Purpose**: Bot Instance Entity
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 112
- `async stop(self) -> None` - Line 134
- `async pause(self) -> None` - Line 155
- `async resume(self) -> None` - Line 176
- `get_bot_state(self) -> BotState` - Line 198
- `get_bot_metrics(self) -> BotMetrics` - Line 207
- `get_bot_config(self) -> BotConfiguration` - Line 218
- `async get_bot_summary(self) -> dict[str, Any]` - Line 227
- `async execute_trade(self, order_request: OrderRequest, execution_params: dict) -> Any` - Line 245
- `async update_position(self, symbol: str, position_data: dict) -> None` - Line 280
- `async close_position(self, symbol: str, reason: str) -> bool` - Line 296
- `async get_heartbeat(self) -> dict[str, Any]` - Line 320
- `async restart(self, reason: str) -> None` - Line 336
- `async queue_websocket_message(self, message: dict) -> bool` - Line 357
- `set_metrics_collector(self, metrics_collector) -> None` - Line 379

### Implementation: `BotInstance` ✅

**Purpose**: Individual bot instance that runs a specific trading strategy
**Status**: Complete

**Implemented Methods:**
- `set_metrics_collector(self, metrics_collector: MetricsCollector) -> None` - Line 317
- `async start(self) -> None` - Line 331
- `async stop(self) -> None` - Line 382
- `async pause(self) -> None` - Line 460
- `async resume(self) -> None` - Line 482
- `get_bot_state(self) -> BotState` - Line 1369
- `get_bot_metrics(self) -> BotMetrics` - Line 1374
- `get_bot_config(self) -> BotConfiguration` - Line 1379
- `async get_bot_summary(self) -> dict[str, Any]` - Line 1384
- `async execute_trade(self, order_request: OrderRequest, execution_params: dict) -> Any` - Line 1417
- `async update_position(self, symbol: str, position_data: dict) -> None` - Line 1528
- `async close_position(self, symbol: str, reason: str) -> bool` - Line 1534
- `async get_heartbeat(self) -> dict[str, Any]` - Line 1602
- `async restart(self, reason: str) -> None` - Line 1659
- `async queue_websocket_message(self, message: dict) -> bool` - Line 1925

### Implementation: `BotLifecycle` ✅

**Purpose**: Comprehensive bot lifecycle management system
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 211
- `async stop(self) -> None` - Line 232
- `async create_bot_from_template(self, ...) -> BotConfiguration` - Line 262
- `async deploy_bot(self, ...) -> bool` - Line 336
- `async terminate_bot(self, ...) -> bool` - Line 414
- `async restart_bot(self, bot_id: str, orchestrator, reason: str = 'manual_restart') -> bool` - Line 493
- `async get_lifecycle_summary(self) -> dict[str, Any]` - Line 564
- `async get_bot_lifecycle_details(self, bot_id: str) -> dict[str, Any] | None` - Line 631

### Implementation: `BotMonitor` ✅

**Inherits**: BaseService
**Purpose**: Comprehensive bot health and performance monitoring system using service layer
**Status**: Complete

**Implemented Methods:**
- `async register_bot(self, bot_id: str) -> None` - Line 356
- `async unregister_bot(self, bot_id: str) -> None` - Line 410
- `async update_bot_metrics(self, bot_id: str, metrics: BotMetrics) -> None` - Line 433
- `async check_bot_health(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]` - Line 477
- `async get_monitoring_summary(self) -> dict[str, Any]` - Line 650
- `async get_bot_health_details(self, bot_id: str) -> dict[str, Any] | None` - Line 742
- `async get_bot_health_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]` - Line 1777
- `async get_alert_history(self, bot_id: str | None = None, hours: int = 24) -> list[dict[str, Any]]` - Line 1828
- `async get_resource_usage_summary(self, bot_id: str) -> dict[str, Any]` - Line 2040
- `async compare_bot_performance(self) -> dict[str, Any]` - Line 2151

### Implementation: `BotManagementController` ✅

**Inherits**: BaseService
**Purpose**: Controller for bot management operations
**Status**: Complete

**Implemented Methods:**
- `bot_instance_service(self) -> IBotInstanceService` - Line 53
- `bot_coordination_service(self) -> IBotCoordinationService` - Line 58
- `bot_lifecycle_service(self) -> IBotLifecycleService` - Line 63
- `bot_monitoring_service(self) -> IBotMonitoringService` - Line 68
- `resource_management_service(self) -> IResourceManagementService` - Line 73
- `async create_bot(self, ...) -> dict[str, Any]` - Line 77
- `async start_bot(self, bot_id: str) -> dict[str, Any]` - Line 124
- `async stop_bot(self, bot_id: str) -> dict[str, Any]` - Line 154
- `async terminate_bot(self, bot_id: str, reason: str = 'user_request') -> dict[str, Any]` - Line 184
- `async get_bot_status(self, bot_id: str) -> dict[str, Any]` - Line 215
- `async execute_bot_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> dict[str, Any]` - Line 245
- `async get_system_overview(self) -> dict[str, Any]` - Line 277
- `async pause_bot(self, bot_id: str) -> dict[str, Any]` - Line 298
- `async resume_bot(self, bot_id: str) -> dict[str, Any]` - Line 320
- `async get_bot_state(self, bot_id: str) -> dict[str, Any]` - Line 342
- `async get_bot_metrics(self, bot_id: str) -> dict[str, Any]` - Line 364
- `async allocate_resources(self, bot_id: str, resources: dict[str, Any]) -> dict[str, Any]` - Line 386
- `async deallocate_resources(self, bot_id: str) -> dict[str, Any]` - Line 418
- `async list_bots(self) -> dict[str, Any]` - Line 440
- `async delete_bot(self, bot_id: str) -> dict[str, Any]` - Line 474

### Implementation: `BotCoordinationService` ✅

**Inherits**: BaseService, IBotCoordinationService
**Purpose**: Service for coordinating bot operations and interactions
**Status**: Complete

**Implemented Methods:**
- `registered_bots(self) -> dict[str, BotConfiguration]` - Line 41
- `async register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None` - Line 45
- `async unregister_bot(self, bot_id: str) -> None` - Line 70
- `async check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]` - Line 96
- `async share_signal(self, ...) -> int` - Line 132
- `async get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]` - Line 186
- `async check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]` - Line 218
- `async cleanup_expired_signals(self) -> int` - Line 274

### Implementation: `BotManagementDataTransformer` ✅

**Purpose**: Handles consistent data transformation for bot_management module
**Status**: Complete

**Implemented Methods:**
- `transform_bot_event_to_event_data(bot_id, ...) -> dict[str, Any]` - Line 24
- `transform_bot_metrics_to_event_data(bot_metrics: BotMetrics, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 67
- `transform_error_to_event_data(error, ...) -> dict[str, Any]` - Line 102
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 136
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'bot_management') -> dict[str, Any]` - Line 158
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 194
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]` - Line 253
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 287
- `apply_cross_module_validation(cls, ...) -> dict[str, Any]` - Line 346

### Implementation: `BotManagementFactory` ✅

**Purpose**: Simplified factory for creating bot management components
**Status**: Complete

**Implemented Methods:**
- `create_bot_service(self) -> 'BotService'` - Line 30
- `create_bot_coordinator(self) -> 'BotCoordinator'` - Line 65
- `create_resource_manager(self) -> 'ResourceManager'` - Line 79

### Implementation: `BotInstanceService` ✅

**Inherits**: BaseService, IBotInstanceService
**Purpose**: Service for managing bot instances
**Status**: Complete

**Implemented Methods:**
- `async create_bot_instance(self, bot_config: BotConfiguration) -> str` - Line 51
- `async start_bot(self, bot_id: str) -> bool` - Line 101
- `async stop_bot(self, bot_id: str) -> bool` - Line 125
- `async pause_bot(self, bot_id: str) -> bool` - Line 149
- `async resume_bot(self, bot_id: str) -> bool` - Line 173
- `async get_bot_state(self, bot_id: str) -> BotState` - Line 197
- `async execute_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> Any` - Line 218
- `async update_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None` - Line 249
- `async close_position(self, bot_id: str, symbol: str, reason: str) -> bool` - Line 273
- `async remove_bot_instance(self, bot_id: str) -> bool` - Line 299
- `get_active_bot_ids(self) -> list[str]` - Line 330
- `get_bot_count(self) -> int` - Line 337

### Implementation: `IBotCoordinationService` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot coordination services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None` - Line 26
- `async unregister_bot(self, bot_id: str) -> None` - Line 30
- `async check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]` - Line 34
- `async share_signal(self, ...) -> int` - Line 38
- `async get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]` - Line 50
- `async check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]` - Line 54

### Implementation: `IBotLifecycleService` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot lifecycle management services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_bot_from_template(self, ...) -> BotConfiguration` - Line 64
- `async deploy_bot(self, bot_config: BotConfiguration, strategy: str = 'immediate') -> bool` - Line 78
- `async terminate_bot(self, bot_id: str, reason: str = 'user_request') -> bool` - Line 84
- `async restart_bot(self, bot_id: str, reason: str = 'restart_request') -> bool` - Line 88
- `async get_lifecycle_status(self, bot_id: str) -> dict[str, Any]` - Line 92
- `async rollback_deployment(self, bot_id: str, target_version: str) -> bool` - Line 96

### Implementation: `IBotMonitoringService` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot monitoring services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_bot_health(self, bot_id: str) -> dict[str, Any]` - Line 104
- `async get_bot_metrics(self, bot_id: str) -> BotMetrics` - Line 108
- `async get_system_health(self) -> dict[str, Any]` - Line 112
- `async get_performance_summary(self) -> dict[str, Any]` - Line 116
- `async check_alert_conditions(self) -> list[dict[str, Any]]` - Line 120

### Implementation: `IResourceManagementService` 🔧

**Inherits**: ABC
**Purpose**: Interface for resource management services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async request_resources(self, ...) -> bool` - Line 128
- `async release_resources(self, bot_id: str) -> bool` - Line 137
- `async verify_resources(self, bot_id: str) -> bool` - Line 141
- `async get_resource_summary(self) -> dict[str, Any]` - Line 145
- `async check_resource_availability(self, resource_type: str, amount: Decimal) -> bool` - Line 149
- `async update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool` - Line 155

### Implementation: `IBotInstanceService` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot instance management services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_bot_instance(self, bot_config: BotConfiguration) -> str` - Line 165
- `async start_bot(self, bot_id: str) -> bool` - Line 169
- `async stop_bot(self, bot_id: str) -> bool` - Line 173
- `async pause_bot(self, bot_id: str) -> bool` - Line 177
- `async resume_bot(self, bot_id: str) -> bool` - Line 181
- `async get_bot_state(self, bot_id: str) -> BotState` - Line 185
- `async execute_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> Any` - Line 189
- `async update_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None` - Line 198
- `async close_position(self, bot_id: str, symbol: str, reason: str) -> bool` - Line 204

### Implementation: `BotLifecycleService` ✅

**Inherits**: BaseService, IBotLifecycleService
**Purpose**: Service for managing bot lifecycles
**Status**: Complete

**Implemented Methods:**
- `async create_bot_from_template(self, ...) -> BotConfiguration` - Line 87
- `async deploy_bot(self, bot_config: BotConfiguration, strategy: str = 'immediate') -> bool` - Line 161
- `async terminate_bot(self, bot_id: str, reason: str = 'user_request') -> bool` - Line 212
- `async restart_bot(self, bot_id: str, reason: str = 'restart_request') -> bool` - Line 257
- `async get_lifecycle_status(self, bot_id: str) -> dict[str, Any]` - Line 306
- `async rollback_deployment(self, bot_id: str, target_version: str) -> bool` - Line 337

### Implementation: `BotMonitoringService` ✅

**Inherits**: BaseService, IBotMonitoringService
**Purpose**: Service for monitoring bot health and performance
**Status**: Complete

**Implemented Methods:**
- `async get_bot_health(self, bot_id: str) -> dict[str, Any]` - Line 43
- `async get_bot_metrics(self, bot_id: str) -> BotMetrics` - Line 70
- `async get_system_health(self) -> dict[str, Any]` - Line 98
- `async get_performance_summary(self) -> dict[str, Any]` - Line 146
- `async check_alert_conditions(self) -> list[dict[str, Any]]` - Line 201

### Implementation: `BotRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_name(self, name: str) -> Bot | None` - Line 47
- `async get_active_bots(self) -> list[Bot]` - Line 58
- `async update_status(self, bot_id: str, status: BotStatus) -> Bot` - Line 73
- `async create_bot_configuration(self, bot_config: Any) -> bool` - Line 101
- `async get_bot_configuration(self, bot_id: str) -> dict[str, Any] | None` - Line 121
- `async update_bot_configuration(self, bot_config: Any) -> bool` - Line 141
- `async delete_bot_configuration(self, bot_id: str) -> bool` - Line 161
- `async list_bot_configurations(self) -> list[dict[str, Any]]` - Line 181
- `async store_bot_metrics(self, metrics: dict[str, Any]) -> bool` - Line 199
- `async get_bot_metrics(self, bot_id: str) -> list[dict[str, Any]]` - Line 219
- `async health_check(self) -> bool` - Line 239

### Implementation: `BotInstanceRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot instance entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_bot_id(self, bot_id: str) -> list[BotInstance]` - Line 265
- `async get_active_instance(self, bot_id: str) -> BotInstance | None` - Line 280
- `async get_active_instances(self) -> list[BotInstance]` - Line 296
- `async update_metrics(self, instance_id: str, metrics: BotMetrics) -> BotInstance` - Line 311
- `async get_performance_stats(self, bot_id: str) -> dict[str, Any]` - Line 345

### Implementation: `BotMetricsRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot metrics
**Status**: Complete

**Implemented Methods:**
- `async save_metrics(self, metrics: BotMetrics) -> None` - Line 389
- `async get_latest_metrics(self, bot_id: str) -> BotMetrics | None` - Line 428

### Implementation: `ResourceManager` ✅

**Inherits**: BaseComponent
**Purpose**: Central resource manager for bot instances
**Status**: Complete

**Implemented Methods:**
- `set_metrics_collector(self, metrics_collector: 'MetricsCollector') -> None` - Line 168
- `async start(self) -> None` - Line 182
- `async stop(self) -> None` - Line 213
- `async request_resources(self, ...) -> bool` - Line 250
- `async release_resources(self, bot_id: str) -> bool` - Line 346
- `async verify_resources(self, bot_id: str) -> bool` - Line 389
- `async update_resource_usage(self, bot_id: str, usage_data: dict[str, float]) -> None` - Line 418
- `async get_resource_usage(self, bot_id: str) -> dict[str, Any] | None` - Line 432
- `async get_resource_summary(self) -> dict[str, Any]` - Line 488
- `async get_bot_resource_usage(self, bot_id: str) -> dict[str, Any] | None` - Line 547
- `async get_bot_allocations(self) -> dict[str, dict[str, Decimal]]` - Line 583
- `async update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool` - Line 604
- `async check_single_resource_availability(self, resource_type: ResourceType, amount: Decimal) -> bool` - Line 1498
- `async allocate_api_limits(self, bot_id: str, requests_per_minute: int) -> bool` - Line 1532
- `async allocate_database_connections(self, bot_id: str, connections: int) -> bool` - Line 1561
- `async detect_resource_conflicts(self) -> list[dict[str, Any]]` - Line 1594
- `async emergency_reallocate(self, bot_id: str, capital_amount: Decimal) -> bool` - Line 1632
- `async get_optimization_suggestions(self) -> list[dict[str, Any]]` - Line 1678
- `async get_resource_alerts(self) -> list[str]` - Line 1763
- `async reserve_resources(self, ...) -> str | None` - Line 1804
- `async allocate_resources(self, bot_id: str, resource_request: dict[str, Any]) -> bool` - Line 1950
- `async deallocate_resources(self, bot_id: str) -> bool` - Line 1959
- `async get_system_resource_usage(self) -> dict[str, Any]` - Line 1969
- `async check_resource_availability(self, resource_request_or_type, amount: Any = None) -> bool` - Line 1989
- `async get_allocated_resources(self, bot_id: str) -> dict[str, Any] | None` - Line 2049
- `async optimize_resource_allocation(self) -> dict[str, Any]` - Line 2053
- `async check_resource_alerts(self) -> list[dict[str, Any]]` - Line 2066
- `async collect_resource_metrics(self) -> dict[str, Any]` - Line 2114
- `async allocate_resources_with_priority(self, bot_id: str, resource_request: dict[str, Any], priority: Any) -> bool` - Line 2134
- `async commit_resource_reservation(self, reservation_id: str) -> bool` - Line 2153
- `async health_check(self) -> dict[str, Any]` - Line 2171

### Implementation: `BotResourceService` ✅

**Inherits**: BaseService, IResourceManagementService
**Purpose**: Service for managing bot resources with dependency injection
**Status**: Complete

**Implemented Methods:**
- `async request_resources(self, ...) -> bool` - Line 53
- `async release_resources(self, bot_id: str) -> bool` - Line 105
- `async verify_resources(self, bot_id: str) -> bool` - Line 133
- `async get_resource_summary(self) -> dict[str, Any]` - Line 161
- `async check_resource_availability(self, resource_type: str, amount: Decimal) -> bool` - Line 186
- `async update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool` - Line 216

### Implementation: `BotService` ✅

**Inherits**: BaseService
**Purpose**: Comprehensive bot management service
**Status**: Complete

**Implemented Methods:**
- `async execute_with_monitoring(self, operation_name: str, operation_func: Any, *args, **kwargs) -> Any` - Line 255
- `async create_bot(self, bot_config: BotConfiguration) -> str` - Line 336
- `async start_bot(self, bot_id: str) -> bool` - Line 585
- `async stop_bot(self, bot_id: str) -> bool` - Line 838
- `async delete_bot(self, bot_id: str, force: bool = False) -> bool` - Line 960
- `async get_bot_status(self, bot_id: str) -> dict[str, Any]` - Line 1100
- `async get_all_bots_status(self) -> dict[str, Any]` - Line 1175
- `async update_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> bool` - Line 1225
- `async start_all_bots(self, priority_filter: BotPriority | None = None) -> dict[str, bool]` - Line 1515
- `async stop_all_bots(self) -> dict[str, bool]` - Line 1564
- `async perform_health_check(self, bot_id: str) -> dict[str, Any]` - Line 1595

## COMPLETE API REFERENCE

### File: bot_coordinator.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.data_transformer import CoreDataTransformer`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotConfiguration`

#### Class: `BotCoordinator`

**Purpose**: Inter-bot communication and coordination manager

```python
class BotCoordinator:
    def __init__(self, config: Config)  # Line 65
    async def start(self) -> None  # Line 120
    async def stop(self) -> None  # Line 142
    async def register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None  # Line 182
    async def unregister_bot(self, bot_id: str) -> None  # Line 224
    async def report_position_change(self, ...) -> dict[str, Any]  # Line 260
    async def share_signal(self, ...) -> int  # Line 323
    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]  # Line 408
    async def check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]  # Line 450
    def _check_symbol_exposure_limits(self, ...) -> None  # Line 504
    def _check_position_concentration(self, risk_assessment: dict[str, Any], bot_id: str, symbol: str) -> None  # Line 522
    def _check_conflicting_positions(self, ...) -> None  # Line 537
    def _generate_risk_recommendations(self, risk_assessment: dict[str, Any]) -> None  # Line 572
    async def get_coordination_summary(self) -> dict[str, Any]  # Line 583
    async def _coordination_loop(self) -> None  # Line 622
    async def _signal_distribution_loop(self) -> None  # Line 662
    async def _update_symbol_exposure(self, ...) -> None  # Line 695
    async def _analyze_position_change(self, ...) -> dict[str, Any]  # Line 727
    async def _detect_arbitrage_opportunities(self) -> None  # Line 771
    async def _detect_position_conflicts(self) -> None  # Line 803
    async def _analyze_symbol_conflicts(self, symbol: str) -> bool  # Line 835
    async def _cleanup_expired_signals(self) -> None  # Line 864
    async def _update_exchange_metrics(self) -> None  # Line 877
    async def _analyze_signal_correlations(self) -> None  # Line 890
    async def _process_priority_signals(self) -> None  # Line 915
    async def _update_signal_statistics(self) -> None  # Line 932
    async def update_bot_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None  # Line 950
    async def remove_bot_position(self, bot_id: str, symbol: str) -> None  # Line 959
    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]  # Line 965
    async def coordinate_bot_actions(self, action_data: dict[str, Any]) -> dict[str, Any]  # Line 990
    async def analyze_bot_interactions(self) -> dict[str, Any]  # Line 998
    async def optimize_coordination(self) -> dict[str, Any]  # Line 1008
    async def emergency_coordination(self, emergency_type: str, action: str) -> None  # Line 1012
```

### File: bot_entity.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotConfiguration`
- `from src.core.types import BotMetrics`
- `from src.core.types import BotState`

#### Class: `BotInstance`

**Purpose**: Bot Instance Entity

```python
class BotInstance:
    def __init__(self, ...)  # Line 36
    def _validate_configuration(self) -> None  # Line 70
    def _create_default_metrics(self) -> BotMetrics  # Line 92
    async def start(self) -> None  # Line 112
    async def stop(self) -> None  # Line 134
    async def pause(self) -> None  # Line 155
    async def resume(self) -> None  # Line 176
    def get_bot_state(self) -> BotState  # Line 198
    def get_bot_metrics(self) -> BotMetrics  # Line 207
    def get_bot_config(self) -> BotConfiguration  # Line 218
    async def get_bot_summary(self) -> dict[str, Any]  # Line 227
    async def execute_trade(self, order_request: OrderRequest, execution_params: dict) -> Any  # Line 245
    async def update_position(self, symbol: str, position_data: dict) -> None  # Line 280
    async def close_position(self, symbol: str, reason: str) -> bool  # Line 296
    async def get_heartbeat(self) -> dict[str, Any]  # Line 320
    async def restart(self, reason: str) -> None  # Line 336
    async def queue_websocket_message(self, message: dict) -> bool  # Line 357
    def set_metrics_collector(self, metrics_collector) -> None  # Line 379
```

### File: bot_instance.py

**Key Imports:**
- `from src.capital_management.service import CapitalService`
- `from src.core.config import Config`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import StrategyError`

#### Class: `BotInstance`

**Purpose**: Individual bot instance that runs a specific trading strategy

```python
class BotInstance:
    def _convert_to_strategy_type(strategy_id: str) -> StrategyType  # Line 135
    def __init__(self, ...)  # Line 153
    def set_metrics_collector(self, metrics_collector: MetricsCollector) -> None  # Line 317
    async def start(self) -> None  # Line 331
    async def stop(self) -> None  # Line 382
    async def pause(self) -> None  # Line 460
    async def resume(self) -> None  # Line 482
    async def _validate_configuration(self) -> None  # Line 503
    async def _initialize_components(self) -> None  # Line 533
    async def _allocate_resources(self) -> None  # Line 555
    async def _initialize_strategy(self) -> None  # Line 590
    async def _start_strategy_execution(self) -> None  # Line 631
    async def _strategy_execution_loop(self) -> None  # Line 639
    async def _get_current_market_data(self, symbol: str) -> MarketData | None  # Line 710
    async def _process_trading_signal(self, signal) -> None  # Line 744
    async def _check_position_limits(self) -> bool  # Line 784
    async def _create_order_request_from_signal(self, signal) -> OrderRequest | None  # Line 796
    async def _validate_order_request(self, order_request: OrderRequest, symbol: str) -> bool  # Line 829
    async def _execute_order_request(self, order_request: OrderRequest, signal) -> Any | None  # Line 873
    async def _record_execution_metrics(self, start_time: datetime) -> None  # Line 928
    async def _start_monitoring(self) -> None  # Line 957
    async def _start_websocket_monitoring(self) -> None  # Line 961
    async def _heartbeat_loop(self) -> None  # Line 971
    async def _check_daily_limits(self) -> None  # Line 993
    async def _update_strategy_state(self) -> None  # Line 1017
    async def _track_execution(self, execution_result, order_request = None) -> None  # Line 1044
    async def _get_order_from_execution(self, execution_result, order_request = None)  # Line 1070
    def _update_basic_execution_metrics(self, execution_result) -> None  # Line 1080
    async def _update_position_tracking(self, order) -> None  # Line 1086
    async def _process_execution_pnl(self, execution_result, order) -> None  # Line 1099
    async def _record_pnl_metrics(self, trade_pnl: Decimal) -> None  # Line 1119
    async def _notify_strategy_of_execution(self, execution_result, order) -> None  # Line 1135
    async def _calculate_portfolio_value(self) -> Decimal  # Line 1175
    async def _update_performance_metrics(self) -> None  # Line 1181
    async def _check_resource_usage(self) -> None  # Line 1207
    async def _create_state_checkpoint(self) -> None  # Line 1216
    async def _close_open_positions(self) -> None  # Line 1232
    async def _cancel_pending_orders(self) -> None  # Line 1247
    async def _release_resources(self) -> None  # Line 1263
    async def _release_capital_resources(self) -> None  # Line 1276
    async def _close_websocket_connections(self) -> None  # Line 1289
    async def _close_individual_connections(self, websocket_connections: list) -> None  # Line 1314
    async def _cleanup_remaining_connections(self, websocket_connections: list) -> None  # Line 1344
    def get_bot_state(self) -> BotState  # Line 1369
    def get_bot_metrics(self) -> BotMetrics  # Line 1374
    def get_bot_config(self) -> BotConfiguration  # Line 1379
    async def get_bot_summary(self) -> dict[str, Any]  # Line 1384
    async def execute_trade(self, order_request: OrderRequest, execution_params: dict) -> Any  # Line 1417
    async def update_position(self, symbol: str, position_data: dict) -> None  # Line 1528
    async def close_position(self, symbol: str, reason: str) -> bool  # Line 1534
    async def get_heartbeat(self) -> dict[str, Any]  # Line 1602
    async def _trading_loop(self) -> None  # Line 1617
    async def _calculate_performance_metrics(self) -> None  # Line 1629
    async def _check_risk_limits(self, order_request: OrderRequest) -> bool  # Line 1649
    async def restart(self, reason: str) -> None  # Line 1659
    async def _websocket_heartbeat_loop(self) -> None  # Line 1670
    async def _websocket_timeout_monitor_loop(self) -> None  # Line 1731
    async def _check_and_handle_websocket_timeouts(self) -> None  # Line 1754
    async def _handle_websocket_timeout(self, exchange_name: str) -> None  # Line 1776
    async def _websocket_message_processor_loop(self) -> None  # Line 1811
    async def _process_websocket_message_batch(self, messages: list) -> None  # Line 1841
    async def _process_single_websocket_message(self, message: dict) -> None  # Line 1854
    async def _handle_market_data_message(self, message: dict) -> None  # Line 1878
    async def _handle_order_update_message(self, message: dict) -> None  # Line 1895
    async def _handle_account_update_message(self, message: dict) -> None  # Line 1906
    async def _handle_pong_message(self, message: dict) -> None  # Line 1920
    async def queue_websocket_message(self, message: dict) -> bool  # Line 1925
    async def _close_existing_websocket_connection(self, exchange_name: str) -> None  # Line 1952
    async def _attempt_websocket_reconnection(self, exchange_name: str) -> bool  # Line 1972
    async def _collect_message_batch(self) -> list  # Line 2039
    async def _process_remaining_messages_on_shutdown(self) -> None  # Line 2066
    async def _clear_message_queue(self) -> None  # Line 2111
    async def _circuit_breaker_reset_loop(self) -> None  # Line 2140
```

### File: bot_lifecycle.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotConfiguration`

#### Class: `BotLifecycle`

**Purpose**: Comprehensive bot lifecycle management system

```python
class BotLifecycle:
    def __init__(self, config: Config)  # Line 59
    def _initialize_bot_templates(self) -> None  # Line 109
    def _initialize_deployment_strategies(self) -> None  # Line 199
    async def start(self) -> None  # Line 211
    async def stop(self) -> None  # Line 232
    async def create_bot_from_template(self, ...) -> BotConfiguration  # Line 262
    async def deploy_bot(self, ...) -> bool  # Line 336
    async def terminate_bot(self, ...) -> bool  # Line 414
    async def restart_bot(self, bot_id: str, orchestrator, reason: str = 'manual_restart') -> bool  # Line 493
    async def get_lifecycle_summary(self) -> dict[str, Any]  # Line 564
    async def get_bot_lifecycle_details(self, bot_id: str) -> dict[str, Any] | None  # Line 631
    async def _initialize_bot_lifecycle(self, bot_id: str, template_name: str, deployment_strategy: str) -> None  # Line 681
    async def _record_lifecycle_event(self, bot_id: str, event_type: str, event_data: dict[str, Any]) -> None  # Line 696
    async def _deploy_immediate(self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]) -> bool  # Line 716
    async def _deploy_staged(self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]) -> bool  # Line 734
    async def _deploy_blue_green(self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]) -> bool  # Line 755
    async def _deploy_canary(self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]) -> bool  # Line 763
    async def _deploy_rolling(self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]) -> bool  # Line 771
    async def _graceful_termination(self, bot_id: str, orchestrator) -> bool  # Line 779
    async def _immediate_termination(self, bot_id: str, orchestrator) -> bool  # Line 806
    async def _lifecycle_loop(self) -> None  # Line 821
    async def _cleanup_old_events(self) -> None  # Line 845
    async def _monitor_lifecycle_health(self) -> None  # Line 860
    async def _update_lifecycle_statistics(self) -> None  # Line 877
```

### File: bot_monitor.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.data_transformer import CoreDataTransformer`
- `from src.core.exceptions import NetworkError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`

#### Class: `BotMonitor`

**Inherits**: BaseService
**Purpose**: Comprehensive bot health and performance monitoring system using service layer

```python
class BotMonitor(BaseService):
    def __init__(self)  # Line 115
    def _load_configuration(self) -> dict[str, Any]  # Line 220
    async def _do_start(self) -> None  # Line 246
    async def _do_stop(self) -> None  # Line 322
    async def register_bot(self, bot_id: str) -> None  # Line 356
    async def unregister_bot(self, bot_id: str) -> None  # Line 410
    async def update_bot_metrics(self, bot_id: str, metrics: BotMetrics) -> None  # Line 433
    async def check_bot_health(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]  # Line 477
    async def _update_monitoring_stats(self, bot_id: str) -> datetime  # Line 508
    async def _initialize_health_results(self, bot_id: str, current_time: datetime) -> dict[str, Any]  # Line 529
    async def _perform_health_checks(self, bot_id: str, bot_status: BotStatus, health_results: dict[str, Any]) -> None  # Line 543
    async def _calculate_and_update_health_score(self, bot_id: str, bot_status: BotStatus, health_results: dict[str, Any]) -> None  # Line 571
    async def _calculate_fallback_health_score(self, health_results: dict[str, Any]) -> float  # Line 598
    async def _determine_overall_health_status(self, health_results: dict[str, Any]) -> None  # Line 609
    async def _finalize_health_results(self, bot_id: str, health_results: dict[str, Any], current_time: datetime) -> None  # Line 618
    async def get_monitoring_summary(self) -> dict[str, Any]  # Line 650
    async def get_bot_health_details(self, bot_id: str) -> dict[str, Any] | None  # Line 742
    async def _health_check_loop(self) -> None  # Line 804
    async def _metrics_collection_loop(self) -> None  # Line 850
    async def _update_bot_health_status(self, bot_id: str, metrics: BotMetrics) -> None  # Line 898
    async def _check_performance_issues(self, metrics: BotMetrics) -> list[str]  # Line 917
    async def _check_cpu_usage_issues(self, metrics: BotMetrics, issues: list[str]) -> None  # Line 935
    async def _check_memory_usage_issues(self, metrics: BotMetrics, issues: list[str]) -> None  # Line 942
    async def _check_error_rate_issues(self, metrics: BotMetrics, issues: list[str]) -> None  # Line 949
    async def _check_win_rate_issues(self, metrics: BotMetrics, issues: list[str]) -> None  # Line 958
    async def _store_metrics(self, bot_id: str, metrics: BotMetrics) -> None  # Line 965
    async def _check_performance_anomalies(self, bot_id: str, metrics: BotMetrics) -> None  # Line 1007
    async def _update_performance_baseline(self, bot_id: str, metrics: BotMetrics) -> None  # Line 1069
    async def _check_bot_status(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]  # Line 1104
    async def _check_bot_heartbeat(self, bot_id: str) -> dict[str, Any]  # Line 1136
    async def _check_resource_usage(self, bot_id: str) -> dict[str, Any]  # Line 1170
    async def _check_performance_health(self, bot_id: str) -> dict[str, Any]  # Line 1234
    async def _check_error_rate(self, bot_id: str) -> dict[str, Any]  # Line 1240
    async def _check_risk_health(self, bot_id: str) -> dict[str, Any]  # Line 1246
    async def _generate_health_alerts(self, bot_id: str, health_results: dict[str, Any]) -> None  # Line 1283
    async def _generate_anomaly_alerts(self, bot_id: str, anomalies: list[dict[str, Any]]) -> None  # Line 1320
    async def _cleanup_old_alerts(self) -> None  # Line 1346
    async def _process_alert_escalations(self) -> None  # Line 1372
    async def _update_monitoring_statistics(self) -> None  # Line 1392
    async def _collect_system_metrics(self) -> None  # Line 1399
    async def _update_all_baselines(self) -> None  # Line 1470
    async def _collect_risk_metrics(self, bot_id: str) -> None  # Line 1482
    async def _check_alert_conditions(self, bot_id: str) -> None  # Line 1552
    async def _generate_alert(self, bot_id: str, alert_type: str, severity: str, message: str) -> None  # Line 1611
    async def _establish_performance_baseline(self, bot_id: str) -> None  # Line 1673
    async def _detect_anomalies(self, bot_id: str, metrics: BotMetrics) -> list[dict[str, Any]]  # Line 1711
    async def get_bot_health_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]  # Line 1777
    async def get_alert_history(self, bot_id: str | None = None, hours: int = 24) -> list[dict[str, Any]]  # Line 1828
    async def _calculate_health_score(self, bot_id: str, bot_status: BotStatus, metrics: BotMetrics) -> float  # Line 1881
    async def _monitoring_loop(self) -> None  # Line 1893
    async def _cleanup_old_metrics(self) -> None  # Line 1925
    async def _export_metrics_to_influxdb(self, bot_id: str, metrics: BotMetrics) -> None  # Line 1944
    async def _detect_performance_degradation(self, bot_id: str, metrics: BotMetrics) -> dict[str, Any]  # Line 1965
    async def get_resource_usage_summary(self, bot_id: str) -> dict[str, Any]  # Line 2040
    async def _generate_predictive_alerts(self, bot_id: str) -> list[dict[str, Any]]  # Line 2090
    async def compare_bot_performance(self) -> dict[str, Any]  # Line 2151
```

### File: controller.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import EntityNotFoundError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `BotManagementController`

**Inherits**: BaseService
**Purpose**: Controller for bot management operations

```python
class BotManagementController(BaseService):
    def __init__(self, ...)  # Line 35
    def bot_instance_service(self) -> IBotInstanceService  # Line 53
    def bot_coordination_service(self) -> IBotCoordinationService  # Line 58
    def bot_lifecycle_service(self) -> IBotLifecycleService  # Line 63
    def bot_monitoring_service(self) -> IBotMonitoringService  # Line 68
    def resource_management_service(self) -> IResourceManagementService  # Line 73
    async def create_bot(self, ...) -> dict[str, Any]  # Line 77
    async def start_bot(self, bot_id: str) -> dict[str, Any]  # Line 124
    async def stop_bot(self, bot_id: str) -> dict[str, Any]  # Line 154
    async def terminate_bot(self, bot_id: str, reason: str = 'user_request') -> dict[str, Any]  # Line 184
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]  # Line 215
    async def execute_bot_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> dict[str, Any]  # Line 245
    async def get_system_overview(self) -> dict[str, Any]  # Line 277
    async def pause_bot(self, bot_id: str) -> dict[str, Any]  # Line 298
    async def resume_bot(self, bot_id: str) -> dict[str, Any]  # Line 320
    async def get_bot_state(self, bot_id: str) -> dict[str, Any]  # Line 342
    async def get_bot_metrics(self, bot_id: str) -> dict[str, Any]  # Line 364
    async def allocate_resources(self, bot_id: str, resources: dict[str, Any]) -> dict[str, Any]  # Line 386
    async def deallocate_resources(self, bot_id: str) -> dict[str, Any]  # Line 418
    async def list_bots(self) -> dict[str, Any]  # Line 440
    async def delete_bot(self, bot_id: str) -> dict[str, Any]  # Line 474
```

### File: coordination_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotConfiguration`

#### Class: `BotCoordinationService`

**Inherits**: BaseService, IBotCoordinationService
**Purpose**: Service for coordinating bot operations and interactions

```python
class BotCoordinationService(BaseService, IBotCoordinationService):
    def __init__(self, name: str = 'BotCoordinationService', config: dict[str, Any] = None)  # Line 27
    def registered_bots(self) -> dict[str, BotConfiguration]  # Line 41
    async def register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None  # Line 45
    async def unregister_bot(self, bot_id: str) -> None  # Line 70
    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]  # Line 96
    async def share_signal(self, ...) -> int  # Line 132
    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]  # Line 186
    async def check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]  # Line 218
    async def cleanup_expired_signals(self) -> int  # Line 274
    def _detect_position_conflict(self, bot1: dict[str, Any], bot2: dict[str, Any], symbol: str) -> dict[str, Any] | None  # Line 304
    def _calculate_total_exposure(self, symbol: str) -> Decimal  # Line 328
    def _check_symbol_exposure_limits(self, risk_assessment: dict[str, Any], symbol: str, quantity: Decimal) -> None  # Line 340
    def _check_position_concentration(self, risk_assessment: dict[str, Any], bot_id: str, symbol: str) -> None  # Line 347
    def _check_conflicting_positions(self, ...) -> None  # Line 354
    def _generate_risk_recommendations(self, risk_assessment: dict[str, Any]) -> None  # Line 366
    def _get_current_timestamp(self)  # Line 371
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import BotConfiguration`
- `from src.core.types import BotMetrics`
- `from src.core.types import BotState`
- `from src.core.types import BotStatus`

#### Class: `BotManagementDataTransformer`

**Purpose**: Handles consistent data transformation for bot_management module

```python
class BotManagementDataTransformer:
    def transform_bot_event_to_event_data(bot_id, ...) -> dict[str, Any]  # Line 24
    def transform_bot_metrics_to_event_data(bot_metrics: BotMetrics, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 67
    def transform_error_to_event_data(error, ...) -> dict[str, Any]  # Line 102
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 136
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'bot_management') -> dict[str, Any]  # Line 158
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 194
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]  # Line 253
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 287
    def apply_cross_module_validation(cls, ...) -> dict[str, Any]  # Line 346
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def register_bot_management_services(injector: DependencyInjector) -> None  # Line 14
def configure_bot_management_dependencies(injector: DependencyInjector | None = None) -> DependencyInjector  # Line 61
def get_bot_service(injector: DependencyInjector) -> 'BotService'  # Line 83
def get_bot_instance_service(injector: DependencyInjector) -> 'IBotInstanceService'  # Line 88
def get_bot_lifecycle_service(injector: DependencyInjector) -> 'IBotLifecycleService'  # Line 94
def get_bot_coordination_service(injector: DependencyInjector) -> 'IBotCoordinationService'  # Line 100
def get_bot_monitoring_service(injector: DependencyInjector) -> 'IBotMonitoringService'  # Line 106
def get_bot_resource_service(injector: DependencyInjector) -> 'IResourceManagementService'  # Line 112
def get_bot_management_controller(injector: DependencyInjector) -> 'BotManagementController'  # Line 118
def initialize_bot_management_services(injector: DependencyInjector) -> dict[str, Any]  # Line 124
```

### File: factory.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.exceptions import DependencyError`
- `from src.core.logging import get_logger`

#### Class: `BotManagementFactory`

**Purpose**: Simplified factory for creating bot management components

```python
class BotManagementFactory:
    def __init__(self, injector: DependencyInjector = None)  # Line 25
    def create_bot_service(self) -> 'BotService'  # Line 30
    def create_bot_coordinator(self) -> 'BotCoordinator'  # Line 65
    def create_resource_manager(self) -> 'ResourceManager'  # Line 79
```

#### Functions:

```python
def register_bot_management_services(injector: DependencyInjector) -> None  # Line 95
def create_bot_management_factory(injector: DependencyInjector = None) -> BotManagementFactory  # Line 110
```

### File: instance_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.config import Config`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `BotInstanceService`

**Inherits**: BaseService, IBotInstanceService
**Purpose**: Service for managing bot instances

```python
class BotInstanceService(BaseService, IBotInstanceService):
    def __init__(self, ...)  # Line 35
    async def create_bot_instance(self, bot_config: BotConfiguration) -> str  # Line 51
    async def start_bot(self, bot_id: str) -> bool  # Line 101
    async def stop_bot(self, bot_id: str) -> bool  # Line 125
    async def pause_bot(self, bot_id: str) -> bool  # Line 149
    async def resume_bot(self, bot_id: str) -> bool  # Line 173
    async def get_bot_state(self, bot_id: str) -> BotState  # Line 197
    async def execute_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> Any  # Line 218
    async def update_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None  # Line 249
    async def close_position(self, bot_id: str, symbol: str, reason: str) -> bool  # Line 273
    async def remove_bot_instance(self, bot_id: str) -> bool  # Line 299
    def get_active_bot_ids(self) -> list[str]  # Line 330
    def get_bot_count(self) -> int  # Line 337
    async def _validate_exchange_connectivity(self, bot_config: BotConfiguration) -> None  # Line 341
```

### File: interfaces.py

**Key Imports:**
- `from src.core.types import BotConfiguration`
- `from src.core.types import BotMetrics`
- `from src.core.types import BotPriority`
- `from src.core.types import BotState`
- `from src.core.types import OrderRequest`

#### Class: `IBotCoordinationService`

**Inherits**: ABC
**Purpose**: Interface for bot coordination services

```python
class IBotCoordinationService(ABC):
    async def register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None  # Line 26
    async def unregister_bot(self, bot_id: str) -> None  # Line 30
    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]  # Line 34
    async def share_signal(self, ...) -> int  # Line 38
    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]  # Line 50
    async def check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]  # Line 54
```

#### Class: `IBotLifecycleService`

**Inherits**: ABC
**Purpose**: Interface for bot lifecycle management services

```python
class IBotLifecycleService(ABC):
    async def create_bot_from_template(self, ...) -> BotConfiguration  # Line 64
    async def deploy_bot(self, bot_config: BotConfiguration, strategy: str = 'immediate') -> bool  # Line 78
    async def terminate_bot(self, bot_id: str, reason: str = 'user_request') -> bool  # Line 84
    async def restart_bot(self, bot_id: str, reason: str = 'restart_request') -> bool  # Line 88
    async def get_lifecycle_status(self, bot_id: str) -> dict[str, Any]  # Line 92
    async def rollback_deployment(self, bot_id: str, target_version: str) -> bool  # Line 96
```

#### Class: `IBotMonitoringService`

**Inherits**: ABC
**Purpose**: Interface for bot monitoring services

```python
class IBotMonitoringService(ABC):
    async def get_bot_health(self, bot_id: str) -> dict[str, Any]  # Line 104
    async def get_bot_metrics(self, bot_id: str) -> BotMetrics  # Line 108
    async def get_system_health(self) -> dict[str, Any]  # Line 112
    async def get_performance_summary(self) -> dict[str, Any]  # Line 116
    async def check_alert_conditions(self) -> list[dict[str, Any]]  # Line 120
```

#### Class: `IResourceManagementService`

**Inherits**: ABC
**Purpose**: Interface for resource management services

```python
class IResourceManagementService(ABC):
    async def request_resources(self, ...) -> bool  # Line 128
    async def release_resources(self, bot_id: str) -> bool  # Line 137
    async def verify_resources(self, bot_id: str) -> bool  # Line 141
    async def get_resource_summary(self) -> dict[str, Any]  # Line 145
    async def check_resource_availability(self, resource_type: str, amount: Decimal) -> bool  # Line 149
    async def update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool  # Line 155
```

#### Class: `IBotInstanceService`

**Inherits**: ABC
**Purpose**: Interface for bot instance management services

```python
class IBotInstanceService(ABC):
    async def create_bot_instance(self, bot_config: BotConfiguration) -> str  # Line 165
    async def start_bot(self, bot_id: str) -> bool  # Line 169
    async def stop_bot(self, bot_id: str) -> bool  # Line 173
    async def pause_bot(self, bot_id: str) -> bool  # Line 177
    async def resume_bot(self, bot_id: str) -> bool  # Line 181
    async def get_bot_state(self, bot_id: str) -> BotState  # Line 185
    async def execute_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> Any  # Line 189
    async def update_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None  # Line 198
    async def close_position(self, bot_id: str, symbol: str, reason: str) -> bool  # Line 204
```

### File: lifecycle_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotConfiguration`

#### Class: `BotLifecycleService`

**Inherits**: BaseService, IBotLifecycleService
**Purpose**: Service for managing bot lifecycles

```python
class BotLifecycleService(BaseService, IBotLifecycleService):
    def __init__(self, name: str = 'BotLifecycleService', config: dict[str, Any] = None)  # Line 27
    def _initialize_bot_templates(self) -> None  # Line 37
    def _initialize_deployment_strategies(self) -> None  # Line 77
    async def create_bot_from_template(self, ...) -> BotConfiguration  # Line 87
    async def deploy_bot(self, bot_config: BotConfiguration, strategy: str = 'immediate') -> bool  # Line 161
    async def terminate_bot(self, bot_id: str, reason: str = 'user_request') -> bool  # Line 212
    async def restart_bot(self, bot_id: str, reason: str = 'restart_request') -> bool  # Line 257
    async def get_lifecycle_status(self, bot_id: str) -> dict[str, Any]  # Line 306
    async def rollback_deployment(self, bot_id: str, target_version: str) -> bool  # Line 337
    async def _record_lifecycle_event(self, bot_id: str, event_type: str, data: dict[str, Any]) -> None  # Line 383
    def _merge_configs(self, default_config: dict[str, Any], custom_config: dict[str, Any]) -> dict[str, Any]  # Line 403
    def _determine_lifecycle_status(self, latest_events: dict[str, Any]) -> str  # Line 409
    async def _deploy_immediate(self, bot_config: BotConfiguration) -> bool  # Line 424
    async def _deploy_staged(self, bot_config: BotConfiguration) -> bool  # Line 429
    async def _deploy_blue_green(self, bot_config: BotConfiguration) -> bool  # Line 434
    async def _deploy_canary(self, bot_config: BotConfiguration) -> bool  # Line 439
    async def _deploy_rolling(self, bot_config: BotConfiguration) -> bool  # Line 444
    async def _graceful_termination(self, bot_id: str) -> bool  # Line 449
```

### File: monitoring_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotMetrics`

#### Class: `BotMonitoringService`

**Inherits**: BaseService, IBotMonitoringService
**Purpose**: Service for monitoring bot health and performance

```python
class BotMonitoringService(BaseService, IBotMonitoringService):
    def __init__(self, ...)  # Line 29
    async def get_bot_health(self, bot_id: str) -> dict[str, Any]  # Line 43
    async def get_bot_metrics(self, bot_id: str) -> BotMetrics  # Line 70
    async def get_system_health(self) -> dict[str, Any]  # Line 98
    async def get_performance_summary(self) -> dict[str, Any]  # Line 146
    async def check_alert_conditions(self) -> list[dict[str, Any]]  # Line 201
    async def _perform_health_check(self, bot_id: str) -> dict[str, Any]  # Line 229
    async def _collect_bot_metrics(self, bot_id: str) -> BotMetrics  # Line 265
    async def _check_bot_alerts(self, bot_id: str) -> list[dict[str, Any]]  # Line 304
    async def _check_system_alerts(self) -> list[dict[str, Any]]  # Line 348
    async def _check_system_issues(self) -> list[str]  # Line 369
    def _get_current_timestamp(self)  # Line 383
```

### File: repository.py

**Key Imports:**
- `from src.core.exceptions import DatabaseError`
- `from src.core.exceptions import EntityNotFoundError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotMetrics`
- `from src.core.types import BotStatus`

#### Class: `BotRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot entities

```python
class BotRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession | Any)  # Line 30
    async def get_by_name(self, name: str) -> Bot | None  # Line 47
    async def get_active_bots(self) -> list[Bot]  # Line 58
    async def update_status(self, bot_id: str, status: BotStatus) -> Bot  # Line 73
    async def create_bot_configuration(self, bot_config: Any) -> bool  # Line 101
    async def get_bot_configuration(self, bot_id: str) -> dict[str, Any] | None  # Line 121
    async def update_bot_configuration(self, bot_config: Any) -> bool  # Line 141
    async def delete_bot_configuration(self, bot_id: str) -> bool  # Line 161
    async def list_bot_configurations(self) -> list[dict[str, Any]]  # Line 181
    async def store_bot_metrics(self, metrics: dict[str, Any]) -> bool  # Line 199
    async def get_bot_metrics(self, bot_id: str) -> list[dict[str, Any]]  # Line 219
    async def health_check(self) -> bool  # Line 239
```

#### Class: `BotInstanceRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot instance entities

```python
class BotInstanceRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 259
    async def get_by_bot_id(self, bot_id: str) -> list[BotInstance]  # Line 265
    async def get_active_instance(self, bot_id: str) -> BotInstance | None  # Line 280
    async def get_active_instances(self) -> list[BotInstance]  # Line 296
    async def update_metrics(self, instance_id: str, metrics: BotMetrics) -> BotInstance  # Line 311
    async def get_performance_stats(self, bot_id: str) -> dict[str, Any]  # Line 345
```

#### Class: `BotMetricsRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot metrics

```python
class BotMetricsRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 382
    async def save_metrics(self, metrics: BotMetrics) -> None  # Line 389
    async def get_latest_metrics(self, bot_id: str) -> BotMetrics | None  # Line 428
```

### File: resource_manager.py

**Key Imports:**
- `from src.capital_management.service import CapitalService`
- `from src.core.base.component import BaseComponent`
- `from src.core.config import Config`
- `from src.core.data_transformer import CoreDataTransformer`
- `from src.core.exceptions import DatabaseConnectionError`

#### Class: `ResourceManager`

**Inherits**: BaseComponent
**Purpose**: Central resource manager for bot instances

```python
class ResourceManager(BaseComponent):
    def __init__(self, config: Config, capital_service: CapitalService | None = None)  # Line 70
    def _initialize_resource_limits(self) -> None  # Line 125
    def set_metrics_collector(self, metrics_collector: 'MetricsCollector') -> None  # Line 168
    async def start(self) -> None  # Line 182
    async def stop(self) -> None  # Line 213
    async def request_resources(self, ...) -> bool  # Line 250
    async def release_resources(self, bot_id: str) -> bool  # Line 346
    async def verify_resources(self, bot_id: str) -> bool  # Line 389
    async def update_resource_usage(self, bot_id: str, usage_data: dict[str, float]) -> None  # Line 418
    async def get_resource_usage(self, bot_id: str) -> dict[str, Any] | None  # Line 432
    async def _update_resource_usage_by_type(self, bot_id: str, resource_type: ResourceType, used_amount: Decimal) -> None  # Line 444
    async def get_resource_summary(self) -> dict[str, Any]  # Line 488
    async def get_bot_resource_usage(self, bot_id: str) -> dict[str, Any] | None  # Line 547
    async def get_bot_allocations(self) -> dict[str, dict[str, Decimal]]  # Line 583
    async def update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool  # Line 604
    async def _calculate_resource_requirements(self, bot_id: str, capital_amount: Decimal, priority: BotPriority) -> dict[ResourceType, Decimal]  # Line 719
    async def _check_resource_availability(self, requirements: dict[ResourceType, Decimal]) -> dict[str, Any]  # Line 766
    async def _allocate_resources(self, bot_id: str, requirements: dict[ResourceType, Decimal]) -> dict[ResourceType, ResourceAllocation]  # Line 803
    async def _reallocate_for_high_priority(self, ...) -> bool  # Line 874
    async def _release_resource_allocation(self, allocation: ResourceAllocation) -> None  # Line 944
    async def _release_specific_resource_types(self, allocation: ResourceAllocation) -> list  # Line 966
    async def _collect_websocket_connections(self, allocation: ResourceAllocation) -> list  # Line 1009
    async def _handle_resource_release_error(self, error: Exception, allocation: ResourceAllocation) -> None  # Line 1024
    async def _cleanup_resource_connections(self, db_connection, websocket_connections: list) -> None  # Line 1044
    async def _perform_connection_cleanup(self, db_connection, websocket_connections: list) -> None  # Line 1058
    async def _close_database_connection(self, db_connection) -> None  # Line 1068
    async def _close_websocket_connections(self, websocket_connections: list) -> None  # Line 1079
    async def _verify_resource_allocation(self, allocation: ResourceAllocation) -> bool  # Line 1097
    async def _basic_allocation_validation(self, allocation: ResourceAllocation) -> bool  # Line 1120
    async def _verify_specific_resource_type(self, allocation: ResourceAllocation) -> bool  # Line 1139
    async def _verify_capital_allocation(self, allocation: ResourceAllocation) -> bool  # Line 1151
    async def _verify_websocket_connections(self, allocation: ResourceAllocation) -> bool  # Line 1173
    async def _handle_verification_error(self, error: Exception, allocation: ResourceAllocation) -> None  # Line 1190
    async def _cleanup_verification_connections(self, db_connection, websocket_connections: list) -> None  # Line 1210
    async def _cleanup_verification_websockets(self, websocket_connections: list) -> None  # Line 1229
    async def _monitoring_loop(self) -> None  # Line 1247
    async def _update_resource_usage_tracking(self) -> None  # Line 1284
    async def _check_resource_violations(self) -> None  # Line 1365
    async def _optimize_resource_allocations(self) -> None  # Line 1392
    async def _cleanup_expired_allocations(self) -> None  # Line 1428
    async def _release_all_resources(self) -> None  # Line 1443
    async def _cleanup_failed_allocation(self, bot_id: str) -> None  # Line 1476
    async def check_single_resource_availability(self, resource_type: ResourceType, amount: Decimal) -> bool  # Line 1498
    async def allocate_api_limits(self, bot_id: str, requests_per_minute: int) -> bool  # Line 1532
    async def allocate_database_connections(self, bot_id: str, connections: int) -> bool  # Line 1561
    async def detect_resource_conflicts(self) -> list[dict[str, Any]]  # Line 1594
    async def emergency_reallocate(self, bot_id: str, capital_amount: Decimal) -> bool  # Line 1632
    async def get_optimization_suggestions(self) -> list[dict[str, Any]]  # Line 1678
    async def _resource_monitoring_loop(self) -> None  # Line 1725
    async def _cleanup_stale_allocations(self) -> int  # Line 1730
    async def get_resource_alerts(self) -> list[str]  # Line 1763
    async def reserve_resources(self, ...) -> str | None  # Line 1804
    async def _cleanup_expired_reservations(self) -> int  # Line 1908
    async def allocate_resources(self, bot_id: str, resource_request: dict[str, Any]) -> bool  # Line 1950
    async def deallocate_resources(self, bot_id: str) -> bool  # Line 1959
    async def get_system_resource_usage(self) -> dict[str, Any]  # Line 1969
    async def check_resource_availability(self, resource_request_or_type, amount: Any = None) -> bool  # Line 1989
    async def get_allocated_resources(self, bot_id: str) -> dict[str, Any] | None  # Line 2049
    async def optimize_resource_allocation(self) -> dict[str, Any]  # Line 2053
    async def check_resource_alerts(self) -> list[dict[str, Any]]  # Line 2066
    async def _cleanup_inactive_bot_resources(self) -> None  # Line 2093
    async def collect_resource_metrics(self) -> dict[str, Any]  # Line 2114
    async def allocate_resources_with_priority(self, bot_id: str, resource_request: dict[str, Any], priority: Any) -> bool  # Line 2134
    async def commit_resource_reservation(self, reservation_id: str) -> bool  # Line 2153
    async def health_check(self) -> dict[str, Any]  # Line 2171
```

### File: resource_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotPriority`

#### Class: `BotResourceService`

**Inherits**: BaseService, IResourceManagementService
**Purpose**: Service for managing bot resources with dependency injection

```python
class BotResourceService(BaseService, IResourceManagementService):
    def __init__(self, ...)  # Line 30
    def _load_resource_limits(self) -> dict[str, Any]  # Line 42
    async def request_resources(self, ...) -> bool  # Line 53
    async def release_resources(self, bot_id: str) -> bool  # Line 105
    async def verify_resources(self, bot_id: str) -> bool  # Line 133
    async def get_resource_summary(self) -> dict[str, Any]  # Line 161
    async def check_resource_availability(self, resource_type: str, amount: Decimal) -> bool  # Line 186
    async def update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool  # Line 216
    def _calculate_utilization(self, summary: dict[str, Any]) -> dict[str, float]  # Line 253
    def _get_current_timestamp(self)  # Line 272
    def _get_current_timestamp_iso(self) -> str  # Line 277
```

### File: service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.events import BotEvent`
- `from src.core.events import BotEventType`
- `from src.core.events import get_event_publisher`
- `from src.core.exceptions import ServiceError`

#### Class: `BotService`

**Inherits**: BaseService
**Purpose**: Comprehensive bot management service

```python
class BotService(BaseService):
    def __init__(self, ...)  # Line 84
    async def _do_start(self) -> None  # Line 178
    async def execute_with_monitoring(self, operation_name: str, operation_func: Any, *args, **kwargs) -> Any  # Line 255
    async def _do_stop(self) -> None  # Line 322
    async def create_bot(self, bot_config: BotConfiguration) -> str  # Line 336
    async def _create_bot_impl(self, bot_config: BotConfiguration) -> str  # Line 352
    async def start_bot(self, bot_id: str) -> bool  # Line 585
    async def _start_bot_impl(self, bot_id: str) -> bool  # Line 600
    async def stop_bot(self, bot_id: str) -> bool  # Line 838
    async def _stop_bot_impl(self, bot_id: str) -> bool  # Line 850
    async def delete_bot(self, bot_id: str, force: bool = False) -> bool  # Line 960
    async def _delete_bot_impl(self, bot_id: str, force: bool = False) -> bool  # Line 975
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]  # Line 1100
    async def _get_bot_status_impl(self, bot_id: str) -> dict[str, Any]  # Line 1114
    async def get_all_bots_status(self) -> dict[str, Any]  # Line 1175
    async def _get_all_bots_status_impl(self) -> dict[str, Any]  # Line 1186
    async def update_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> bool  # Line 1225
    async def _update_bot_metrics_impl(self, bot_id: str, metrics: dict[str, Any]) -> bool  # Line 1240
    async def _handle_risk_action(self, bot_id: str, risk_metrics: dict[str, Any]) -> None  # Line 1367
    async def _handle_high_risk_bot(self, bot_id: str, risk_metrics: Any) -> None  # Line 1412
    def _setup_event_handlers(self) -> None  # Line 1458
    def _calculate_bot_rate_requirements(self, bot_config: BotConfiguration) -> int  # Line 1474
    async def start_all_bots(self, priority_filter: BotPriority | None = None) -> dict[str, bool]  # Line 1515
    async def _start_all_bots_impl(self, priority_filter: BotPriority | None = None) -> dict[str, bool]  # Line 1529
    async def stop_all_bots(self) -> dict[str, bool]  # Line 1564
    async def _stop_all_bots_impl(self) -> dict[str, bool]  # Line 1573
    async def perform_health_check(self, bot_id: str) -> dict[str, Any]  # Line 1595
    async def _perform_health_check_impl(self, bot_id: str) -> dict[str, Any]  # Line 1609
    async def _validate_bot_configuration(self, bot_config: BotConfiguration) -> None  # Line 1688
    async def _validate_exchange_configuration(self, bot_config: BotConfiguration) -> None  # Line 1724
    async def _stop_all_active_bots(self) -> None  # Line 1865
    async def _load_existing_bot_states(self) -> None  # Line 1880
    async def _service_health_check(self) -> Any  # Line 1971
```

---
**Generated**: Complete reference for bot_management module
**Total Classes**: 23
**Total Functions**: 12