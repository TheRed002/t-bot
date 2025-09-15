# BOT_MANAGEMENT Module Reference

## INTEGRATION
**Dependencies**: capital_management, core, database, error_handling, exchanges, execution, monitoring, risk_management, state, strategies, utils
**Used By**: None
**Provides**: BotCoordinationService, BotHealthService, BotInstanceService, BotLifecycleService, BotManagementController, BotMonitoringService, BotResourceService, BotService, IBotCoordinationService, IBotInstanceService, IBotLifecycleService, IBotMonitoringService, IResourceManagementService, ResourceManagementService, ResourceManager
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
**Functions**: 8

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `BotCoordinator` ✅

**Purpose**: Inter-bot communication and coordination manager
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 118
- `async stop(self) -> None` - Line 140
- `async register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None` - Line 180
- `async unregister_bot(self, bot_id: str) -> None` - Line 222
- `async report_position_change(self, ...) -> dict[str, Any]` - Line 258
- `async share_signal(self, ...) -> int` - Line 321
- `async get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]` - Line 391
- `async check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]` - Line 433
- `async get_coordination_summary(self) -> dict[str, Any]` - Line 551
- `async update_bot_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None` - Line 918
- `async remove_bot_position(self, bot_id: str, symbol: str) -> None` - Line 927
- `async check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]` - Line 933
- `async coordinate_bot_actions(self, action_data: dict[str, Any]) -> dict[str, Any]` - Line 958
- `async analyze_bot_interactions(self) -> dict[str, Any]` - Line 966
- `async optimize_coordination(self) -> dict[str, Any]` - Line 976
- `async emergency_coordination(self, emergency_type: str, action: str) -> None` - Line 980

### Implementation: `BotInstance` ✅

**Purpose**: Individual bot instance that runs a specific trading strategy
**Status**: Complete

**Implemented Methods:**
- `set_metrics_collector(self, metrics_collector: MetricsCollector) -> None` - Line 271
- `async start(self) -> None` - Line 285
- `async stop(self) -> None` - Line 336
- `async pause(self) -> None` - Line 403
- `async resume(self) -> None` - Line 425
- `get_bot_state(self) -> BotState` - Line 1264
- `get_bot_metrics(self) -> BotMetrics` - Line 1269
- `get_bot_config(self) -> BotConfiguration` - Line 1274
- `async get_bot_summary(self) -> dict[str, Any]` - Line 1279
- `async execute_trade(self, order_request: OrderRequest, execution_params: dict) -> Any` - Line 1312
- `async update_position(self, symbol: str, position_data: dict) -> None` - Line 1429
- `async close_position(self, symbol: str, reason: str) -> bool` - Line 1435
- `async get_heartbeat(self) -> dict[str, Any]` - Line 1509
- `async restart(self, reason: str) -> None` - Line 1575
- `async queue_websocket_message(self, message: dict) -> bool` - Line 1818

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
- `async register_bot(self, bot_id: str) -> None` - Line 274
- `async unregister_bot(self, bot_id: str) -> None` - Line 328
- `async update_bot_metrics(self, bot_id: str, metrics: BotMetrics) -> None` - Line 351
- `async check_bot_health(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]` - Line 394
- `async get_monitoring_summary(self) -> dict[str, Any]` - Line 567
- `async get_bot_health_details(self, bot_id: str) -> dict[str, Any] | None` - Line 644
- `async get_bot_health_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]` - Line 1738
- `async get_alert_history(self, bot_id: str | None = None, hours: int = 24) -> list[dict[str, Any]]` - Line 1789
- `async get_resource_usage_summary(self, bot_id: str) -> dict[str, Any]` - Line 2070
- `async compare_bot_performance(self) -> dict[str, Any]` - Line 2177

### Implementation: `BotManagementController` ✅

**Inherits**: BaseService
**Purpose**: Controller for bot management operations
**Status**: Complete

**Implemented Methods:**
- `bot_instance_service(self)` - Line 68
- `bot_coordination_service(self)` - Line 73
- `bot_lifecycle_service(self)` - Line 78
- `bot_monitoring_service(self)` - Line 83
- `resource_management_service(self)` - Line 88
- `async create_bot(self, ...) -> dict[str, Any]` - Line 93
- `async start_bot(self, bot_id: str) -> dict[str, Any]` - Line 178
- `async stop_bot(self, bot_id: str) -> dict[str, Any]` - Line 213
- `async terminate_bot(self, bot_id: str, reason: str = 'user_request') -> dict[str, Any]` - Line 251
- `async get_bot_status(self, bot_id: str) -> dict[str, Any]` - Line 286
- `async execute_bot_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> dict[str, Any]` - Line 322
- `async get_system_overview(self) -> dict[str, Any]` - Line 370
- `async pause_bot(self, bot_id: str) -> dict[str, Any]` - Line 397
- `async resume_bot(self, bot_id: str) -> dict[str, Any]` - Line 403
- `async delete_bot(self, bot_id: str) -> dict[str, Any]` - Line 409
- `async get_bot_state(self, bot_id: str) -> dict[str, Any]` - Line 417
- `async get_bot_metrics(self, bot_id: str) -> dict[str, Any]` - Line 422
- `async list_bots(self) -> dict[str, Any]` - Line 426
- `async get_system_status(self) -> dict[str, Any]` - Line 430
- `async allocate_resources(self, bot_id: str, resources: dict[str, Any]) -> dict[str, Any]` - Line 434
- `async deallocate_resources(self, bot_id: str) -> dict[str, Any]` - Line 439

### Implementation: `BotCoordinationService` ✅

**Inherits**: BaseService, IBotCoordinationService
**Purpose**: Service for coordinating bot operations and interactions
**Status**: Complete

**Implemented Methods:**
- `registered_bots(self) -> dict[str, BotConfiguration]` - Line 111
- `async register_bot(self, bot_id: str, bot_config: BotConfiguration) -> bool` - Line 116
- `async unregister_bot(self, bot_id: str) -> bool` - Line 167
- `async check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]` - Line 206
- `async share_signal(self, ...) -> int` - Line 259
- `async get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]` - Line 328
- `async check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]` - Line 375
- `async cleanup_expired_signals(self) -> int` - Line 533
- `async coordinate_bots(self, coordination_data: dict[str, Any]) -> dict[str, Any]` - Line 559
- `async get_coordination_status(self) -> dict[str, Any]` - Line 563
- `async send_signal_to_bot(self, bot_id: str, signal: dict[str, Any]) -> bool` - Line 570
- `async broadcast_signal(self, signal_data: dict[str, Any]) -> int` - Line 578
- `async emergency_stop_all(self, reason: str) -> dict[str, Any]` - Line 586
- `async get_coordination_metrics(self) -> dict[str, Any]` - Line 602
- `async coordinate_bot_health_checks(self) -> dict[str, Any]` - Line 610
- `async resolve_coordination_conflicts(self, conflicts: list[dict[str, Any]]) -> dict[str, Any]` - Line 628
- `async execute_synchronized_action(self, action_data: dict[str, Any]) -> dict[str, Any]` - Line 636
- `async coordinate_resource_allocation(self, resource_request: dict[str, Any]) -> dict[str, Any]` - Line 645
- `async coordinate_based_on_health(self) -> dict[str, Any]` - Line 674

### Implementation: `BotManagementFactory` ✅

**Inherits**: BaseComponent
**Purpose**: Factory for creating and configuring BotManagement service instances
**Status**: Complete

**Implemented Methods:**
- `create_bot_instance_service(self) -> 'IBotInstanceService'` - Line 192
- `create_bot_lifecycle_service(self) -> 'IBotLifecycleService'` - Line 196
- `create_bot_coordination_service(self) -> 'IBotCoordinationService'` - Line 200
- `create_bot_resource_service(self) -> 'IResourceManagementService'` - Line 204
- `create_bot_monitoring_service(self) -> 'IBotMonitoringService'` - Line 208
- `create_capital_service(self) -> 'CapitalService'` - Line 212

### Implementation: `BotHealthService` ✅

**Inherits**: BaseService
**Purpose**: Advanced bot health monitoring and analysis service
**Status**: Complete

**Implemented Methods:**
- `async analyze_bot_health(self, bot_id: str) -> dict[str, Any]` - Line 129
- `async get_bot_health_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]` - Line 605
- `async compare_bot_health(self) -> dict[str, Any]` - Line 644

### Implementation: `BotInstanceService` ✅

**Inherits**: BaseService, IBotInstanceService
**Purpose**: Service for managing bot instances
**Status**: Complete

**Implemented Methods:**
- `async create_bot_instance(self, bot_config: BotConfiguration) -> str` - Line 95
- `async start_bot(self, bot_id: str) -> bool` - Line 151
- `async stop_bot(self, bot_id: str) -> bool` - Line 176
- `async pause_bot(self, bot_id: str) -> bool` - Line 201
- `async resume_bot(self, bot_id: str) -> bool` - Line 226
- `async get_bot_state(self, bot_id: str) -> BotState` - Line 251
- `async execute_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> Any` - Line 276
- `async update_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None` - Line 302
- `async close_position(self, bot_id: str, symbol: str, reason: str) -> bool` - Line 325
- `async remove_bot_instance(self, bot_id: str) -> bool` - Line 348
- `get_active_bot_ids(self) -> list[str]` - Line 377
- `get_bot_count(self) -> int` - Line 381

### Implementation: `IBotCoordinationService` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot coordination services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None` - Line 26
- `async unregister_bot(self, bot_id: str) -> None` - Line 31
- `async check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]` - Line 36
- `async share_signal(self, ...) -> int` - Line 41
- `async get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]` - Line 48
- `async check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]` - Line 53

### Implementation: `IBotLifecycleService` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot lifecycle management services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_bot_from_template(self, ...) -> BotConfiguration` - Line 64
- `async deploy_bot(self, bot_config: BotConfiguration, strategy: str = 'immediate') -> bool` - Line 75
- `async terminate_bot(self, bot_id: str, reason: str = 'user_request') -> bool` - Line 80
- `async restart_bot(self, bot_id: str, reason: str = 'restart_request') -> bool` - Line 85
- `async get_lifecycle_status(self, bot_id: str) -> dict[str, Any]` - Line 90
- `async rollback_deployment(self, bot_id: str, target_version: str) -> bool` - Line 95

### Implementation: `IBotMonitoringService` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot monitoring services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async get_bot_health(self, bot_id: str) -> dict[str, Any]` - Line 104
- `async get_bot_metrics(self, bot_id: str) -> BotMetrics` - Line 109
- `async get_system_health(self) -> dict[str, Any]` - Line 114
- `async get_performance_summary(self) -> dict[str, Any]` - Line 119
- `async check_alert_conditions(self) -> list[dict[str, Any]]` - Line 124

### Implementation: `IResourceManagementService` 🔧

**Inherits**: ABC
**Purpose**: Interface for resource management services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async request_resources(self, ...) -> bool` - Line 133
- `async release_resources(self, bot_id: str) -> bool` - Line 140
- `async verify_resources(self, bot_id: str) -> bool` - Line 145
- `async get_resource_summary(self) -> dict[str, Any]` - Line 150
- `async check_resource_availability(self, resource_type: str, amount: Decimal) -> bool` - Line 155
- `async update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool` - Line 160

### Implementation: `IBotInstanceService` 🔧

**Inherits**: ABC
**Purpose**: Interface for bot instance management services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_bot_instance(self, bot_config: BotConfiguration) -> str` - Line 169
- `async start_bot(self, bot_id: str) -> bool` - Line 174
- `async stop_bot(self, bot_id: str) -> bool` - Line 179
- `async pause_bot(self, bot_id: str) -> bool` - Line 184
- `async resume_bot(self, bot_id: str) -> bool` - Line 189
- `async get_bot_state(self, bot_id: str) -> BotState` - Line 194
- `async execute_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> Any` - Line 199
- `async update_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None` - Line 206
- `async close_position(self, bot_id: str, symbol: str, reason: str) -> bool` - Line 213

### Implementation: `BotLifecycleService` ✅

**Inherits**: BaseService, IBotLifecycleService
**Purpose**: Service for managing bot lifecycles
**Status**: Complete

**Implemented Methods:**
- `async create_bot_from_template(self, ...) -> BotConfiguration` - Line 152
- `async deploy_bot(self, bot_config: BotConfiguration, strategy: str = 'immediate') -> bool` - Line 244
- `async terminate_bot(self, bot_id: str, reason: str = 'user_request') -> bool` - Line 317
- `async restart_bot(self, bot_id: str, reason: str = 'restart_request') -> bool` - Line 373
- `async get_lifecycle_status(self, bot_id: str) -> dict[str, Any]` - Line 423
- `async rollback_deployment(self, bot_id: str, target_version: str) -> bool` - Line 460

### Implementation: `BotMonitoringService` ✅

**Inherits**: BaseService, IBotMonitoringService
**Purpose**: Service for monitoring bot health and performance
**Status**: Complete

**Implemented Methods:**
- `async get_bot_health(self, bot_id: str) -> dict[str, Any]` - Line 72
- `async get_bot_metrics(self, bot_id: str) -> BotMetrics` - Line 133
- `async get_system_health(self) -> dict[str, Any]` - Line 173
- `async get_performance_summary(self) -> dict[str, Any]` - Line 242
- `async check_alert_conditions(self) -> list[dict[str, Any]]` - Line 300

### Implementation: `BotRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_name(self, name: str) -> Bot | None` - Line 46
- `async get_active_bots(self) -> list[Bot]` - Line 59
- `async update_status(self, bot_id: str, status: BotStatus) -> Bot` - Line 74
- `async create_bot_configuration(self, bot_config: Any) -> bool` - Line 93
- `async get_bot_configuration(self, bot_id: str) -> dict[str, Any] | None` - Line 104
- `async update_bot_configuration(self, bot_config: Any) -> bool` - Line 114
- `async delete_bot_configuration(self, bot_id: str) -> bool` - Line 125
- `async list_bot_configurations(self) -> list[dict[str, Any]]` - Line 136
- `async store_bot_metrics(self, metrics: dict[str, Any]) -> bool` - Line 146
- `async get_bot_metrics(self, bot_id: str) -> list[dict[str, Any]]` - Line 157
- `async health_check(self) -> bool` - Line 167

### Implementation: `BotInstanceRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot instance entities
**Status**: Complete

**Implemented Methods:**
- `async get_by_bot_id(self, bot_id: str) -> list[BotInstance]` - Line 186
- `async get_active_instance(self, bot_id: str) -> BotInstance | None` - Line 201
- `async get_active_instances(self) -> list[BotInstance]` - Line 222
- `async update_metrics(self, instance_id: str, metrics: BotMetrics) -> BotInstance` - Line 237
- `async get_performance_stats(self, bot_id: str) -> dict[str, Any]` - Line 264

### Implementation: `BotMetricsRepository` ✅

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot metrics
**Status**: Complete

**Implemented Methods:**
- `async save_metrics(self, metrics: BotMetrics) -> None` - Line 308
- `async get_latest_metrics(self, bot_id: str) -> BotMetrics | None` - Line 338

### Implementation: `ResourceManager` ✅

**Inherits**: BaseComponent
**Purpose**: Central resource manager for bot instances
**Status**: Complete

**Implemented Methods:**
- `set_metrics_collector(self, metrics_collector: 'MetricsCollector') -> None` - Line 144
- `async start(self) -> None` - Line 158
- `async stop(self) -> None` - Line 189
- `async request_resources(self, ...) -> bool` - Line 226
- `async release_resources(self, bot_id: str) -> bool` - Line 322
- `async verify_resources(self, bot_id: str) -> bool` - Line 365
- `async update_resource_usage(self, bot_id: str, usage_data: dict[str, float]) -> None` - Line 394
- `async get_resource_usage(self, bot_id: str) -> dict[str, Any] | None` - Line 408
- `async get_resource_summary(self) -> dict[str, Any]` - Line 464
- `async get_bot_resource_usage(self, bot_id: str) -> dict[str, Any] | None` - Line 519
- `async get_bot_allocations(self) -> dict[str, dict[str, Decimal]]` - Line 555
- `async update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool` - Line 576
- `async check_resource_availability(self, resource_type: ResourceType, amount: Decimal) -> bool` - Line 1361
- `async allocate_api_limits(self, bot_id: str, requests_per_minute: int) -> bool` - Line 1395
- `async allocate_database_connections(self, bot_id: str, connections: int) -> bool` - Line 1424
- `async detect_resource_conflicts(self) -> list[dict[str, Any]]` - Line 1457
- `async emergency_reallocate(self, bot_id: str, capital_amount: Decimal) -> bool` - Line 1495
- `async get_optimization_suggestions(self) -> list[dict[str, Any]]` - Line 1541
- `async get_resource_alerts(self) -> list[str]` - Line 1626
- `async reserve_resources(self, ...) -> str | None` - Line 1667
- `async allocate_resources(self, bot_id: str, resource_request: dict[str, Any]) -> bool` - Line 1802
- `async deallocate_resources(self, bot_id: str) -> bool` - Line 1811
- `async get_resource_usage(self) -> dict[str, Any]` - Line 1821
- `async check_resource_availability(self, resource_request_or_type, amount: Any = None) -> bool` - Line 1841
- `async get_allocated_resources(self, bot_id: str) -> dict[str, Any] | None` - Line 1898
- `async optimize_resource_allocation(self) -> dict[str, Any]` - Line 1902
- `async check_resource_alerts(self) -> list[dict[str, Any]]` - Line 1915
- `async collect_resource_metrics(self) -> dict[str, Any]` - Line 1956
- `async allocate_resources_with_priority(self, bot_id: str, resource_request: dict[str, Any], priority: Any) -> bool` - Line 1976
- `async commit_resource_reservation(self, reservation_id: str) -> bool` - Line 1996
- `async health_check(self) -> dict[str, Any]` - Line 2014

### Implementation: `BotResourceService` ✅

**Inherits**: BaseService, IResourceManagementService
**Purpose**: Service for managing bot resources with dependency injection
**Status**: Complete

**Implemented Methods:**
- `async request_resources(self, ...) -> bool` - Line 79
- `async release_resources(self, bot_id: str) -> bool` - Line 127
- `async verify_resources(self, bot_id: str) -> bool` - Line 156
- `async get_resource_summary(self) -> dict[str, Any]` - Line 183
- `async check_resource_availability(self, resource_type: str, amount: Decimal) -> bool` - Line 218
- `async update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool` - Line 244

### Implementation: `ResourceManagementService` ✅

**Inherits**: BaseService
**Purpose**: Comprehensive resource management service using service layer pattern
**Status**: Complete

**Implemented Methods:**
- `async request_resources(self, ...) -> bool` - Line 150
- `async release_resources(self, bot_id: str) -> bool` - Line 236
- `async verify_resources(self, bot_id: str) -> bool` - Line 287
- `async update_resource_usage(self, bot_id: str, usage_data: dict[str, float]) -> bool` - Line 322
- `async reserve_resources(self, ...) -> str | None` - Line 366
- `async get_resource_summary(self) -> dict[str, Any]` - Line 450
- `async get_optimization_suggestions(self) -> list[dict[str, Any]]` - Line 518
- `async detect_resource_conflicts(self) -> list[dict[str, Any]]` - Line 561

### Implementation: `BotService` ✅

**Inherits**: BaseService
**Purpose**: Comprehensive bot management service
**Status**: Complete

**Implemented Methods:**
- `async execute_with_monitoring(self, operation_name: str, operation_func: Any, *args, **kwargs) -> Any` - Line 259
- `async create_bot(self, bot_config: BotConfiguration) -> str` - Line 330
- `async start_bot(self, bot_id: str) -> bool` - Line 536
- `async stop_bot(self, bot_id: str) -> bool` - Line 752
- `async delete_bot(self, bot_id: str, force: bool = False) -> bool` - Line 870
- `async get_bot_status(self, bot_id: str) -> dict[str, Any]` - Line 974
- `async get_all_bots_status(self) -> dict[str, Any]` - Line 1044
- `async update_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> bool` - Line 1094
- `async start_all_bots(self, priority_filter: BotPriority | None = None) -> dict[str, bool]` - Line 1360
- `async stop_all_bots(self) -> dict[str, bool]` - Line 1409
- `async perform_health_check(self, bot_id: str) -> dict[str, Any]` - Line 1440

## COMPLETE API REFERENCE

### File: bot_coordinator.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotConfiguration`
- `from src.core.types import OrderSide`

#### Class: `BotCoordinator`

**Purpose**: Inter-bot communication and coordination manager

```python
class BotCoordinator:
    def __init__(self, config: Config)  # Line 63
    async def start(self) -> None  # Line 118
    async def stop(self) -> None  # Line 140
    async def register_bot(self, bot_id: str, bot_config: BotConfiguration) -> None  # Line 180
    async def unregister_bot(self, bot_id: str) -> None  # Line 222
    async def report_position_change(self, ...) -> dict[str, Any]  # Line 258
    async def share_signal(self, ...) -> int  # Line 321
    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]  # Line 391
    async def check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]  # Line 433
    def _check_symbol_exposure_limits(self, ...) -> None  # Line 472
    def _check_position_concentration(self, risk_assessment: dict[str, Any], bot_id: str, symbol: str) -> None  # Line 490
    def _check_conflicting_positions(self, ...) -> None  # Line 505
    def _generate_risk_recommendations(self, risk_assessment: dict[str, Any]) -> None  # Line 540
    async def get_coordination_summary(self) -> dict[str, Any]  # Line 551
    async def _coordination_loop(self) -> None  # Line 590
    async def _signal_distribution_loop(self) -> None  # Line 630
    async def _update_symbol_exposure(self, ...) -> None  # Line 663
    async def _analyze_position_change(self, ...) -> dict[str, Any]  # Line 695
    async def _detect_arbitrage_opportunities(self) -> None  # Line 739
    async def _detect_position_conflicts(self) -> None  # Line 769
    async def _analyze_symbol_conflicts(self, symbol: str) -> bool  # Line 801
    async def _cleanup_expired_signals(self) -> None  # Line 830
    async def _update_exchange_metrics(self) -> None  # Line 843
    async def _analyze_signal_correlations(self) -> None  # Line 856
    async def _process_priority_signals(self) -> None  # Line 882
    async def _update_signal_statistics(self) -> None  # Line 899
    async def update_bot_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None  # Line 918
    async def remove_bot_position(self, bot_id: str, symbol: str) -> None  # Line 927
    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]  # Line 933
    async def coordinate_bot_actions(self, action_data: dict[str, Any]) -> dict[str, Any]  # Line 958
    async def analyze_bot_interactions(self) -> dict[str, Any]  # Line 966
    async def optimize_coordination(self) -> dict[str, Any]  # Line 976
    async def emergency_coordination(self, emergency_type: str, action: str) -> None  # Line 980
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
    def _convert_to_strategy_type(strategy_id: str) -> StrategyType  # Line 102
    def __init__(self, ...)  # Line 120
    def set_metrics_collector(self, metrics_collector: MetricsCollector) -> None  # Line 271
    async def start(self) -> None  # Line 285
    async def stop(self) -> None  # Line 336
    async def pause(self) -> None  # Line 403
    async def resume(self) -> None  # Line 425
    async def _validate_configuration(self) -> None  # Line 444
    async def _initialize_components(self) -> None  # Line 474
    async def _allocate_resources(self) -> None  # Line 496
    async def _initialize_strategy(self) -> None  # Line 530
    async def _start_strategy_execution(self) -> None  # Line 566
    async def _strategy_execution_loop(self) -> None  # Line 574
    async def _process_trading_signal(self, signal) -> None  # Line 641
    async def _check_position_limits(self) -> bool  # Line 681
    async def _create_order_request_from_signal(self, signal) -> OrderRequest | None  # Line 693
    async def _validate_order_request(self, order_request: OrderRequest, symbol: str) -> bool  # Line 726
    async def _execute_order_request(self, order_request: OrderRequest, signal) -> Any | None  # Line 770
    async def _record_execution_metrics(self, start_time: datetime) -> None  # Line 831
    async def _start_monitoring(self) -> None  # Line 860
    async def _start_websocket_monitoring(self) -> None  # Line 864
    async def _heartbeat_loop(self) -> None  # Line 873
    async def _check_daily_limits(self) -> None  # Line 895
    async def _update_strategy_state(self) -> None  # Line 919
    async def _track_execution(self, execution_result, order_request = None) -> None  # Line 942
    async def _get_order_from_execution(self, execution_result, order_request = None)  # Line 968
    def _update_basic_execution_metrics(self, execution_result) -> None  # Line 978
    async def _update_position_tracking(self, order) -> None  # Line 984
    async def _process_execution_pnl(self, execution_result, order) -> None  # Line 997
    async def _record_pnl_metrics(self, trade_pnl: Decimal) -> None  # Line 1017
    async def _notify_strategy_of_execution(self, execution_result, order) -> None  # Line 1033
    async def _calculate_portfolio_value(self) -> Decimal  # Line 1088
    async def _update_performance_metrics(self) -> None  # Line 1094
    async def _check_resource_usage(self) -> None  # Line 1116
    async def _create_state_checkpoint(self) -> None  # Line 1125
    async def _close_open_positions(self) -> None  # Line 1141
    async def _cancel_pending_orders(self) -> None  # Line 1156
    async def _release_resources(self) -> None  # Line 1172
    async def _release_capital_resources(self) -> None  # Line 1182
    async def _close_websocket_connections(self) -> None  # Line 1198
    async def _close_individual_connections(self, websocket_connections: list) -> None  # Line 1211
    async def _cleanup_remaining_connections(self, websocket_connections: list) -> None  # Line 1239
    def get_bot_state(self) -> BotState  # Line 1264
    def get_bot_metrics(self) -> BotMetrics  # Line 1269
    def get_bot_config(self) -> BotConfiguration  # Line 1274
    async def get_bot_summary(self) -> dict[str, Any]  # Line 1279
    async def execute_trade(self, order_request: OrderRequest, execution_params: dict) -> Any  # Line 1312
    async def update_position(self, symbol: str, position_data: dict) -> None  # Line 1429
    async def close_position(self, symbol: str, reason: str) -> bool  # Line 1435
    async def get_heartbeat(self) -> dict[str, Any]  # Line 1509
    async def _trading_loop(self) -> None  # Line 1524
    async def _calculate_performance_metrics(self) -> None  # Line 1545
    async def _check_risk_limits(self, order_request: OrderRequest) -> bool  # Line 1565
    async def restart(self, reason: str) -> None  # Line 1575
    async def _websocket_heartbeat_loop(self) -> None  # Line 1586
    async def _websocket_timeout_monitor_loop(self) -> None  # Line 1625
    async def _check_and_handle_websocket_timeouts(self) -> None  # Line 1648
    async def _handle_websocket_timeout(self, exchange_name: str) -> None  # Line 1668
    async def _websocket_message_processor_loop(self) -> None  # Line 1703
    async def _process_websocket_message_batch(self, messages: list) -> None  # Line 1733
    async def _process_single_websocket_message(self, message: dict) -> None  # Line 1746
    async def _handle_market_data_message(self, message: dict) -> None  # Line 1770
    async def _handle_order_update_message(self, message: dict) -> None  # Line 1788
    async def _handle_account_update_message(self, message: dict) -> None  # Line 1799
    async def _handle_pong_message(self, message: dict) -> None  # Line 1813
    async def queue_websocket_message(self, message: dict) -> bool  # Line 1818
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
- `from src.core.exceptions import NetworkError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import BotMetrics`

#### Class: `BotMonitor`

**Inherits**: BaseService
**Purpose**: Comprehensive bot health and performance monitoring system using service layer

```python
class BotMonitor(BaseService):
    def __init__(self)  # Line 83
    async def _do_start(self) -> None  # Line 189
    async def _do_stop(self) -> None  # Line 240
    async def register_bot(self, bot_id: str) -> None  # Line 274
    async def unregister_bot(self, bot_id: str) -> None  # Line 328
    async def update_bot_metrics(self, bot_id: str, metrics: BotMetrics) -> None  # Line 351
    async def check_bot_health(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]  # Line 394
    async def _update_monitoring_stats(self, bot_id: str) -> datetime  # Line 425
    async def _initialize_health_results(self, bot_id: str, current_time: datetime) -> dict[str, Any]  # Line 446
    async def _perform_health_checks(self, bot_id: str, bot_status: BotStatus, health_results: dict[str, Any]) -> None  # Line 460
    async def _calculate_and_update_health_score(self, bot_id: str, bot_status: BotStatus, health_results: dict[str, Any]) -> None  # Line 488
    async def _calculate_fallback_health_score(self, health_results: dict[str, Any]) -> float  # Line 515
    async def _determine_overall_health_status(self, health_results: dict[str, Any]) -> None  # Line 526
    async def _finalize_health_results(self, bot_id: str, health_results: dict[str, Any], current_time: datetime) -> None  # Line 535
    async def get_monitoring_summary(self) -> dict[str, Any]  # Line 567
    async def get_bot_health_details(self, bot_id: str) -> dict[str, Any] | None  # Line 644
    async def _monitoring_loop(self) -> None  # Line 706
    async def _health_check_loop(self) -> None  # Line 751
    async def _metrics_collection_loop(self) -> None  # Line 797
    async def _update_bot_health_status(self, bot_id: str, metrics: BotMetrics) -> None  # Line 845
    async def _check_performance_issues(self, metrics: BotMetrics) -> list[str]  # Line 864
    async def _check_cpu_usage_issues(self, metrics: BotMetrics, issues: list[str]) -> None  # Line 882
    async def _check_memory_usage_issues(self, metrics: BotMetrics, issues: list[str]) -> None  # Line 889
    async def _check_error_rate_issues(self, metrics: BotMetrics, issues: list[str]) -> None  # Line 896
    async def _check_win_rate_issues(self, metrics: BotMetrics, issues: list[str]) -> None  # Line 905
    async def _store_metrics(self, bot_id: str, metrics: BotMetrics) -> None  # Line 912
    async def _check_performance_anomalies(self, bot_id: str, metrics: BotMetrics) -> None  # Line 963
    async def _update_performance_baseline(self, bot_id: str, metrics: BotMetrics) -> None  # Line 1025
    async def _check_bot_status(self, bot_id: str, bot_status: BotStatus) -> dict[str, Any]  # Line 1060
    async def _check_bot_heartbeat(self, bot_id: str) -> dict[str, Any]  # Line 1092
    async def _check_resource_usage(self, bot_id: str) -> dict[str, Any]  # Line 1126
    async def _check_performance_health(self, bot_id: str) -> dict[str, Any]  # Line 1190
    async def _check_error_rate(self, bot_id: str) -> dict[str, Any]  # Line 1196
    async def _check_risk_health(self, bot_id: str) -> dict[str, Any]  # Line 1202
    async def _generate_health_alerts(self, bot_id: str, health_results: dict[str, Any]) -> None  # Line 1239
    async def _generate_anomaly_alerts(self, bot_id: str, anomalies: list[dict[str, Any]]) -> None  # Line 1276
    async def _cleanup_old_alerts(self) -> None  # Line 1302
    async def _process_alert_escalations(self) -> None  # Line 1328
    async def _update_monitoring_statistics(self) -> None  # Line 1348
    async def _collect_system_metrics(self) -> None  # Line 1355
    async def _update_all_baselines(self) -> None  # Line 1438
    async def _collect_risk_metrics(self, bot_id: str) -> None  # Line 1445
    async def _check_alert_conditions(self, bot_id: str) -> None  # Line 1515
    async def _generate_alert(self, bot_id: str, alert_type: str, severity: str, message: str) -> None  # Line 1574
    async def _establish_performance_baseline(self, bot_id: str) -> None  # Line 1636
    async def _detect_anomalies(self, bot_id: str, metrics: BotMetrics) -> list[dict[str, Any]]  # Line 1674
    async def get_bot_health_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]  # Line 1738
    async def get_alert_history(self, bot_id: str | None = None, hours: int = 24) -> list[dict[str, Any]]  # Line 1789
    async def _calculate_health_score(self, bot_id: str, bot_status: BotStatus, metrics: BotMetrics) -> float  # Line 1841
    async def _calculate_status_score(self, bot_status: BotStatus) -> float  # Line 1868
    async def _calculate_resource_score(self, metrics: BotMetrics) -> float  # Line 1882
    async def _calculate_error_rate_score(self, metrics: BotMetrics) -> float  # Line 1902
    async def _calculate_win_rate_score(self, metrics: BotMetrics) -> float  # Line 1915
    async def _monitoring_loop(self) -> None  # Line 1925
    async def _cleanup_old_metrics(self) -> None  # Line 1957
    async def _export_metrics_to_influxdb(self, bot_id: str, metrics: BotMetrics) -> None  # Line 1976
    async def _detect_performance_degradation(self, bot_id: str, metrics: BotMetrics) -> dict[str, Any]  # Line 1997
    async def get_resource_usage_summary(self, bot_id: str) -> dict[str, Any]  # Line 2070
    async def _generate_predictive_alerts(self, bot_id: str) -> list[dict[str, Any]]  # Line 2120
    async def compare_bot_performance(self) -> dict[str, Any]  # Line 2177
```

### File: controller.py

**Key Imports:**
- Note: Interface imports are currently implemented within service files rather than separate interface files

#### Class: `BotManagementController`

**Inherits**: BaseService
**Purpose**: Controller for bot management operations

```python
class BotManagementController(BaseService):
    def __init__(self, ...)  # Line 33
    def bot_instance_service(self)  # Line 68
    def bot_coordination_service(self)  # Line 73
    def bot_lifecycle_service(self)  # Line 78
    def bot_monitoring_service(self)  # Line 83
    def resource_management_service(self)  # Line 88
    async def create_bot(self, ...) -> dict[str, Any]  # Line 93
    async def start_bot(self, bot_id: str) -> dict[str, Any]  # Line 178
    async def stop_bot(self, bot_id: str) -> dict[str, Any]  # Line 213
    async def terminate_bot(self, bot_id: str, reason: str = 'user_request') -> dict[str, Any]  # Line 251
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]  # Line 286
    async def execute_bot_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> dict[str, Any]  # Line 322
    async def get_system_overview(self) -> dict[str, Any]  # Line 370
    async def pause_bot(self, bot_id: str) -> dict[str, Any]  # Line 397
    async def resume_bot(self, bot_id: str) -> dict[str, Any]  # Line 403
    async def delete_bot(self, bot_id: str) -> dict[str, Any]  # Line 409
    async def get_bot_state(self, bot_id: str) -> dict[str, Any]  # Line 417
    async def get_bot_metrics(self, bot_id: str) -> dict[str, Any]  # Line 422
    async def list_bots(self) -> dict[str, Any]  # Line 426
    async def get_system_status(self) -> dict[str, Any]  # Line 430
    async def allocate_resources(self, bot_id: str, resources: dict[str, Any]) -> dict[str, Any]  # Line 434
    async def deallocate_resources(self, bot_id: str) -> dict[str, Any]  # Line 439
    async def _cleanup_failed_creation(self, bot_name: str, bot_id: str | None) -> None  # Line 444
    def _get_current_timestamp_iso(self) -> str  # Line 453
```

### File: coordination_service.py

**Key Imports:**
- Note: Interfaces are implemented within the service class itself
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `BotCoordinationService`

**Inherits**: BaseService, IBotCoordinationService
**Purpose**: Service for coordinating bot operations and interactions

```python
class BotCoordinationService(BaseService, IBotCoordinationService):
    def __init__(self, ...)  # Line 38
    def registered_bots(self) -> dict[str, BotConfiguration]  # Line 111
    async def register_bot(self, bot_id: str, bot_config: BotConfiguration) -> bool  # Line 116
    async def unregister_bot(self, bot_id: str) -> bool  # Line 167
    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]  # Line 206
    async def share_signal(self, ...) -> int  # Line 259
    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]  # Line 328
    async def check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]  # Line 375
    async def _update_symbol_exposure(self, symbol: str, position: dict[str, Any], remove: bool = False) -> None  # Line 426
    async def _check_symbol_exposure_limits(self, ...) -> None  # Line 453
    async def _check_position_concentration(self, risk_assessment: dict[str, Any], bot_id: str, symbol: str) -> None  # Line 470
    async def _check_conflicting_positions(self, ...) -> None  # Line 486
    def _generate_risk_recommendations(self, risk_assessment: dict[str, Any]) -> None  # Line 522
    async def cleanup_expired_signals(self) -> int  # Line 533
    async def _coordination_loop(self) -> None  # Line 555
    async def coordinate_bots(self, coordination_data: dict[str, Any]) -> dict[str, Any]  # Line 559
    async def get_coordination_status(self) -> dict[str, Any]  # Line 563
    async def send_signal_to_bot(self, bot_id: str, signal: dict[str, Any]) -> bool  # Line 570
    async def broadcast_signal(self, signal_data: dict[str, Any]) -> int  # Line 578
    async def emergency_stop_all(self, reason: str) -> dict[str, Any]  # Line 586
    async def get_coordination_metrics(self) -> dict[str, Any]  # Line 602
    async def coordinate_bot_health_checks(self) -> dict[str, Any]  # Line 610
    async def _process_signal_queue(self) -> None  # Line 617
    async def resolve_coordination_conflicts(self, conflicts: list[dict[str, Any]]) -> dict[str, Any]  # Line 628
    async def execute_synchronized_action(self, action_data: dict[str, Any]) -> dict[str, Any]  # Line 636
    async def coordinate_resource_allocation(self, resource_request: dict[str, Any]) -> dict[str, Any]  # Line 645
    async def _check_bot_health(self, bot_id: str) -> bool  # Line 653
    async def _cleanup_old_coordination_data(self) -> int  # Line 657
    async def coordinate_based_on_health(self) -> dict[str, Any]  # Line 674
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def register_bot_management_services(injector: DependencyInjector) -> None  # Line 20
def configure_bot_management_dependencies(injector: DependencyInjector | None = None) -> DependencyInjector  # Line 185
def get_bot_instance_service(injector: DependencyInjector) -> 'IBotInstanceService'  # Line 206
def get_bot_lifecycle_service(injector: DependencyInjector) -> 'IBotLifecycleService'  # Line 212
def get_bot_coordination_service(injector: DependencyInjector) -> 'IBotCoordinationService'  # Line 218
```

### File: factory.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.exceptions import DependencyError`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`

#### Class: `BotManagementFactory`

**Inherits**: BaseComponent
**Purpose**: Factory for creating and configuring BotManagement service instances

```python
class BotManagementFactory(BaseComponent):
    def __init__(self, injector: DependencyInjector)  # Line 36
    def _register_factories(self) -> None  # Line 51
    def _create_bot_instance_service(self) -> 'IBotInstanceService'  # Line 81
    def _create_bot_lifecycle_service(self) -> 'IBotLifecycleService'  # Line 123
    def _create_bot_coordination_service(self) -> 'IBotCoordinationService'  # Line 136
    def _create_bot_resource_service(self) -> 'IResourceManagementService'  # Line 156
    def _create_bot_monitoring_service(self) -> 'IBotMonitoringService'  # Line 176
    def create_bot_instance_service(self) -> 'IBotInstanceService'  # Line 192
    def create_bot_lifecycle_service(self) -> 'IBotLifecycleService'  # Line 196
    def create_bot_coordination_service(self) -> 'IBotCoordinationService'  # Line 200
    def create_bot_resource_service(self) -> 'IResourceManagementService'  # Line 204
    def create_bot_monitoring_service(self) -> 'IBotMonitoringService'  # Line 208
    def create_capital_service(self) -> 'CapitalService'  # Line 212
```

#### Functions:

```python
def create_bot_management_factory(injector: DependencyInjector | None = None) -> BotManagementFactory  # Line 220
def get_bot_instance_service(injector: DependencyInjector | None = None) -> 'IBotInstanceService'  # Line 242
def get_bot_coordination_service(injector: DependencyInjector | None = None) -> 'IBotCoordinationService'  # Line 256
```

### File: service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`

#### Class: `BotHealthService`

**Inherits**: BaseService
**Purpose**: Advanced bot health monitoring and analysis service

```python
class BotHealthService(BaseService):
    def __init__(self)  # Line 29
    async def _do_start(self) -> None  # Line 72
    async def _do_stop(self) -> None  # Line 90
    async def _load_configuration(self) -> None  # Line 107
    async def analyze_bot_health(self, bot_id: str) -> dict[str, Any]  # Line 129
    async def _analyze_bot_health_impl(self, bot_id: str) -> dict[str, Any]  # Line 143
    async def _analyze_performance_health(self, bot_id: str, metrics: list[dict[str, Any]]) -> dict[str, Any]  # Line 234
    async def _analyze_stability_health(self, bot_id: str, health_checks: list[dict[str, Any]]) -> dict[str, Any]  # Line 273
    async def _analyze_resource_health(self, bot_id: str, metrics: list[dict[str, Any]]) -> dict[str, Any]  # Line 312
    async def _analyze_error_health(self, bot_id: str, metrics: list[dict[str, Any]]) -> dict[str, Any]  # Line 345
    async def _analyze_connectivity_health(self, bot_id: str, health_checks: list[dict[str, Any]]) -> dict[str, Any]  # Line 373
    async def _generate_health_recommendations(self, bot_id: str, component_scores: dict[str, Any]) -> list[dict[str, Any]]  # Line 421
    async def _get_component_recommendation(self, component_name: str, component_data: dict[str, Any]) -> dict[str, Any] | None  # Line 447
    async def _get_issue_recommendation(self, issue: str) -> dict[str, Any] | None  # Line 501
    async def _analyze_health_trends(self, bot_id: str, current_health_score: float) -> dict[str, Any]  # Line 541
    async def _store_health_analysis(self, health_analysis: dict[str, Any]) -> None  # Line 587
    async def _update_health_history(self, bot_id: str, health_analysis: dict[str, Any]) -> None  # Line 594
    async def get_bot_health_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]  # Line 605
    async def _get_bot_health_history_impl(self, bot_id: str, hours: int) -> list[dict[str, Any]]  # Line 611
    async def compare_bot_health(self) -> dict[str, Any]  # Line 644
    async def _compare_bot_health_impl(self) -> dict[str, Any]  # Line 650
    async def _generate_system_recommendations(self, latest_analyses: dict[str, dict[str, Any]]) -> list[dict[str, Any]]  # Line 707
    def _identify_performance_issues(self, total_pnl: float, avg_win_rate: float, avg_trades_per_day: float) -> list[str]  # Line 762
    def _identify_stability_issues(self, uptime_percentage: float, consecutive_failures: int) -> list[str]  # Line 779
    def _identify_resource_issues(self, avg_cpu_usage: float, avg_memory_usage: float) -> list[str]  # Line 793
    def _identify_error_issues(self, error_rate: float, total_errors: int) -> list[str]  # Line 805
    def _identify_connectivity_issues(self, heartbeat_issues: int, connectivity_issues: int) -> list[str]  # Line 817
    def _calculate_trend_slope(self, values: list[float]) -> float  # Line 831
    def _calculate_variance(self, values: list[float]) -> float  # Line 846
    async def _health_analysis_loop(self) -> None  # Line 856
    async def _service_health_check(self) -> Any  # Line 885
```

### File: instance_service.py

**Key Imports:**
- Note: Interfaces are implemented within the service class itself
- `from src.capital_management.service import CapitalService`
- `from src.core.base.service import BaseService`
- `from src.core.config import Config`
- `from src.core.exceptions import ServiceError`

#### Class: `BotInstanceService`

**Inherits**: BaseService, IBotInstanceService
**Purpose**: Service for managing bot instances

```python
class BotInstanceService(BaseService, IBotInstanceService):
    def __init__(self, ...)  # Line 42
    async def create_bot_instance(self, bot_config: BotConfiguration) -> str  # Line 95
    async def start_bot(self, bot_id: str) -> bool  # Line 151
    async def stop_bot(self, bot_id: str) -> bool  # Line 176
    async def pause_bot(self, bot_id: str) -> bool  # Line 201
    async def resume_bot(self, bot_id: str) -> bool  # Line 226
    async def get_bot_state(self, bot_id: str) -> BotState  # Line 251
    async def execute_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> Any  # Line 276
    async def update_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None  # Line 302
    async def close_position(self, bot_id: str, symbol: str, reason: str) -> bool  # Line 325
    async def remove_bot_instance(self, bot_id: str) -> bool  # Line 348
    def get_active_bot_ids(self) -> list[str]  # Line 377
    def get_bot_count(self) -> int  # Line 381
    async def _validate_exchange_connectivity(self, bot_config: BotConfiguration) -> None  # Line 385
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
    async def unregister_bot(self, bot_id: str) -> None  # Line 31
    async def check_position_conflicts(self, symbol: str) -> list[dict[str, Any]]  # Line 36
    async def share_signal(self, ...) -> int  # Line 41
    async def get_shared_signals(self, bot_id: str) -> list[dict[str, Any]]  # Line 48
    async def check_cross_bot_risk(self, bot_id: str, symbol: str, side: OrderSide, quantity: Decimal) -> dict[str, Any]  # Line 53
```

#### Class: `IBotLifecycleService`

**Inherits**: ABC
**Purpose**: Interface for bot lifecycle management services

```python
class IBotLifecycleService(ABC):
    async def create_bot_from_template(self, ...) -> BotConfiguration  # Line 64
    async def deploy_bot(self, bot_config: BotConfiguration, strategy: str = 'immediate') -> bool  # Line 75
    async def terminate_bot(self, bot_id: str, reason: str = 'user_request') -> bool  # Line 80
    async def restart_bot(self, bot_id: str, reason: str = 'restart_request') -> bool  # Line 85
    async def get_lifecycle_status(self, bot_id: str) -> dict[str, Any]  # Line 90
    async def rollback_deployment(self, bot_id: str, target_version: str) -> bool  # Line 95
```

#### Class: `IBotMonitoringService`

**Inherits**: ABC
**Purpose**: Interface for bot monitoring services

```python
class IBotMonitoringService(ABC):
    async def get_bot_health(self, bot_id: str) -> dict[str, Any]  # Line 104
    async def get_bot_metrics(self, bot_id: str) -> BotMetrics  # Line 109
    async def get_system_health(self) -> dict[str, Any]  # Line 114
    async def get_performance_summary(self) -> dict[str, Any]  # Line 119
    async def check_alert_conditions(self) -> list[dict[str, Any]]  # Line 124
```

#### Class: `IResourceManagementService`

**Inherits**: ABC
**Purpose**: Interface for resource management services

```python
class IResourceManagementService(ABC):
    async def request_resources(self, ...) -> bool  # Line 133
    async def release_resources(self, bot_id: str) -> bool  # Line 140
    async def verify_resources(self, bot_id: str) -> bool  # Line 145
    async def get_resource_summary(self) -> dict[str, Any]  # Line 150
    async def check_resource_availability(self, resource_type: str, amount: Decimal) -> bool  # Line 155
    async def update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool  # Line 160
```

#### Class: `IBotInstanceService`

**Inherits**: ABC
**Purpose**: Interface for bot instance management services

```python
class IBotInstanceService(ABC):
    async def create_bot_instance(self, bot_config: BotConfiguration) -> str  # Line 169
    async def start_bot(self, bot_id: str) -> bool  # Line 174
    async def stop_bot(self, bot_id: str) -> bool  # Line 179
    async def pause_bot(self, bot_id: str) -> bool  # Line 184
    async def resume_bot(self, bot_id: str) -> bool  # Line 189
    async def get_bot_state(self, bot_id: str) -> BotState  # Line 194
    async def execute_trade(self, bot_id: str, order_request: OrderRequest, execution_params: dict[str, Any]) -> Any  # Line 199
    async def update_position(self, bot_id: str, symbol: str, position_data: dict[str, Any]) -> None  # Line 206
    async def close_position(self, bot_id: str, symbol: str, reason: str) -> bool  # Line 213
```

### File: lifecycle_service.py

**Key Imports:**
- Note: Interfaces are implemented within the service class itself
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `BotLifecycleService`

**Inherits**: BaseService, IBotLifecycleService
**Purpose**: Service for managing bot lifecycles

```python
class BotLifecycleService(BaseService, IBotLifecycleService):
    def __init__(self, name: str = 'BotLifecycleService', config: dict[str, Any] | None = None)  # Line 41
    def _initialize_bot_templates(self) -> None  # Line 84
    def _initialize_deployment_strategies(self) -> None  # Line 139
    async def create_bot_from_template(self, ...) -> BotConfiguration  # Line 152
    async def deploy_bot(self, bot_config: BotConfiguration, strategy: str = 'immediate') -> bool  # Line 244
    async def terminate_bot(self, bot_id: str, reason: str = 'user_request') -> bool  # Line 317
    async def restart_bot(self, bot_id: str, reason: str = 'restart_request') -> bool  # Line 373
    async def get_lifecycle_status(self, bot_id: str) -> dict[str, Any]  # Line 423
    async def rollback_deployment(self, bot_id: str, target_version: str) -> bool  # Line 460
    async def _record_lifecycle_event(self, bot_id: str, event_type: str, data: dict[str, Any]) -> None  # Line 508
    async def _cleanup_old_events(self) -> None  # Line 534
    async def _deploy_immediate(self, bot_config: BotConfiguration) -> bool  # Line 557
    async def _deploy_staged(self, bot_config: BotConfiguration) -> bool  # Line 566
    async def _deploy_blue_green(self, bot_config: BotConfiguration) -> bool  # Line 575
    async def _deploy_canary(self, bot_config: BotConfiguration) -> bool  # Line 584
    async def _deploy_rolling(self, bot_config: BotConfiguration) -> bool  # Line 593
```

### File: monitoring_service.py

**Key Imports:**
- Note: Interfaces are implemented within the service class itself
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotMetrics`

#### Class: `BotMonitoringService`

**Inherits**: BaseService, IBotMonitoringService
**Purpose**: Service for monitoring bot health and performance

```python
class BotMonitoringService(BaseService, IBotMonitoringService):
    def __init__(self, ...)  # Line 36
    async def get_bot_health(self, bot_id: str) -> dict[str, Any]  # Line 72
    async def get_bot_metrics(self, bot_id: str) -> BotMetrics  # Line 133
    async def get_system_health(self) -> dict[str, Any]  # Line 173
    async def get_performance_summary(self) -> dict[str, Any]  # Line 242
    async def check_alert_conditions(self) -> list[dict[str, Any]]  # Line 300
    async def _check_exchange_health(self, exchange_name: str) -> dict[str, Any]  # Line 333
    async def _check_performance_health(self, bot_id: str) -> dict[str, Any]  # Line 394
    async def _get_bot_configuration(self, bot_id: str) -> dict[str, Any] | None  # Line 436
    async def _get_active_bot_ids(self) -> list[str]  # Line 446
    async def _check_bot_health_alerts(self) -> list[dict[str, Any]]  # Line 456
    async def _check_exchange_alerts(self) -> list[dict[str, Any]]  # Line 489
    async def _check_performance_alerts(self) -> list[dict[str, Any]]  # Line 523
    async def _check_resource_alerts(self) -> list[dict[str, Any]]  # Line 549
    async def _check_database_health(self) -> dict[str, Any]  # Line 570
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
    async def get_by_name(self, name: str) -> Bot | None  # Line 46
    async def get_active_bots(self) -> list[Bot]  # Line 59
    async def update_status(self, bot_id: str, status: BotStatus) -> Bot  # Line 74
    async def create_bot_configuration(self, bot_config: Any) -> bool  # Line 93
    async def get_bot_configuration(self, bot_id: str) -> dict[str, Any] | None  # Line 104
    async def update_bot_configuration(self, bot_config: Any) -> bool  # Line 114
    async def delete_bot_configuration(self, bot_id: str) -> bool  # Line 125
    async def list_bot_configurations(self) -> list[dict[str, Any]]  # Line 136
    async def store_bot_metrics(self, metrics: dict[str, Any]) -> bool  # Line 146
    async def get_bot_metrics(self, bot_id: str) -> list[dict[str, Any]]  # Line 157
    async def health_check(self) -> bool  # Line 167
```

#### Class: `BotInstanceRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot instance entities

```python
class BotInstanceRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 180
    async def get_by_bot_id(self, bot_id: str) -> list[BotInstance]  # Line 186
    async def get_active_instance(self, bot_id: str) -> BotInstance | None  # Line 201
    async def get_active_instances(self) -> list[BotInstance]  # Line 222
    async def update_metrics(self, instance_id: str, metrics: BotMetrics) -> BotInstance  # Line 237
    async def get_performance_stats(self, bot_id: str) -> dict[str, Any]  # Line 264
```

#### Class: `BotMetricsRepository`

**Inherits**: DatabaseRepository
**Purpose**: Repository for bot metrics

```python
class BotMetricsRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession)  # Line 301
    async def save_metrics(self, metrics: BotMetrics) -> None  # Line 308
    async def get_latest_metrics(self, bot_id: str) -> BotMetrics | None  # Line 338
```

### File: resource_manager.py

**Key Imports:**
- `from src.capital_management.service import CapitalService`
- `from src.core.base.component import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import DatabaseConnectionError`
- `from src.core.exceptions import ExecutionError`

#### Class: `ResourceManager`

**Inherits**: BaseComponent
**Purpose**: Central resource manager for bot instances

```python
class ResourceManager(BaseComponent):
    def __init__(self, config: Config, capital_service: CapitalService | None = None)  # Line 69
    def _initialize_resource_limits(self) -> None  # Line 119
    def set_metrics_collector(self, metrics_collector: 'MetricsCollector') -> None  # Line 144
    async def start(self) -> None  # Line 158
    async def stop(self) -> None  # Line 189
    async def request_resources(self, ...) -> bool  # Line 226
    async def release_resources(self, bot_id: str) -> bool  # Line 322
    async def verify_resources(self, bot_id: str) -> bool  # Line 365
    async def update_resource_usage(self, bot_id: str, usage_data: dict[str, float]) -> None  # Line 394
    async def get_resource_usage(self, bot_id: str) -> dict[str, Any] | None  # Line 408
    async def _update_resource_usage_by_type(self, bot_id: str, resource_type: ResourceType, used_amount: Decimal) -> None  # Line 420
    async def get_resource_summary(self) -> dict[str, Any]  # Line 464
    async def get_bot_resource_usage(self, bot_id: str) -> dict[str, Any] | None  # Line 519
    async def get_bot_allocations(self) -> dict[str, dict[str, Decimal]]  # Line 555
    async def update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool  # Line 576
    async def _calculate_resource_requirements(self, bot_id: str, capital_amount: Decimal, priority: BotPriority) -> dict[ResourceType, Decimal]  # Line 651
    async def _check_resource_availability(self, requirements: dict[ResourceType, Decimal]) -> dict[str, Any]  # Line 698
    async def _allocate_resources(self, bot_id: str, requirements: dict[ResourceType, Decimal]) -> dict[ResourceType, ResourceAllocation]  # Line 735
    async def _reallocate_for_high_priority(self, ...) -> bool  # Line 783
    async def _release_resource_allocation(self, allocation: ResourceAllocation) -> None  # Line 850
    async def _release_specific_resource_types(self, allocation: ResourceAllocation) -> list  # Line 872
    async def _collect_websocket_connections(self, allocation: ResourceAllocation) -> list  # Line 895
    async def _handle_resource_release_error(self, error: Exception, allocation: ResourceAllocation) -> None  # Line 910
    async def _cleanup_resource_connections(self, db_connection, websocket_connections: list) -> None  # Line 930
    async def _close_database_connection(self, db_connection) -> None  # Line 942
    async def _close_websocket_connections(self, websocket_connections: list) -> None  # Line 953
    async def _verify_resource_allocation(self, allocation: ResourceAllocation) -> bool  # Line 971
    async def _basic_allocation_validation(self, allocation: ResourceAllocation) -> bool  # Line 994
    async def _verify_specific_resource_type(self, allocation: ResourceAllocation) -> bool  # Line 1010
    async def _verify_capital_allocation(self, allocation: ResourceAllocation) -> bool  # Line 1022
    async def _verify_websocket_connections(self, allocation: ResourceAllocation) -> bool  # Line 1036
    async def _handle_verification_error(self, error: Exception, allocation: ResourceAllocation) -> None  # Line 1053
    async def _cleanup_verification_connections(self, db_connection, websocket_connections: list) -> None  # Line 1073
    async def _cleanup_verification_websockets(self, websocket_connections: list) -> None  # Line 1092
    async def _monitoring_loop(self) -> None  # Line 1110
    async def _update_resource_usage_tracking(self) -> None  # Line 1147
    async def _check_resource_violations(self) -> None  # Line 1228
    async def _optimize_resource_allocations(self) -> None  # Line 1255
    async def _cleanup_expired_allocations(self) -> None  # Line 1291
    async def _release_all_resources(self) -> None  # Line 1306
    async def _cleanup_failed_allocation(self, bot_id: str) -> None  # Line 1339
    async def check_resource_availability(self, resource_type: ResourceType, amount: Decimal) -> bool  # Line 1361
    async def allocate_api_limits(self, bot_id: str, requests_per_minute: int) -> bool  # Line 1395
    async def allocate_database_connections(self, bot_id: str, connections: int) -> bool  # Line 1424
    async def detect_resource_conflicts(self) -> list[dict[str, Any]]  # Line 1457
    async def emergency_reallocate(self, bot_id: str, capital_amount: Decimal) -> bool  # Line 1495
    async def get_optimization_suggestions(self) -> list[dict[str, Any]]  # Line 1541
    async def _resource_monitoring_loop(self) -> None  # Line 1588
    async def _cleanup_stale_allocations(self) -> int  # Line 1593
    async def get_resource_alerts(self) -> list[str]  # Line 1626
    async def reserve_resources(self, ...) -> str | None  # Line 1667
    async def _cleanup_expired_reservations(self) -> int  # Line 1760
    async def allocate_resources(self, bot_id: str, resource_request: dict[str, Any]) -> bool  # Line 1802
    async def deallocate_resources(self, bot_id: str) -> bool  # Line 1811
    async def get_resource_usage(self) -> dict[str, Any]  # Line 1821
    async def check_resource_availability(self, resource_request_or_type, amount: Any = None) -> bool  # Line 1841
    async def get_allocated_resources(self, bot_id: str) -> dict[str, Any] | None  # Line 1898
    async def optimize_resource_allocation(self) -> dict[str, Any]  # Line 1902
    async def check_resource_alerts(self) -> list[dict[str, Any]]  # Line 1915
    async def _cleanup_inactive_bot_resources(self) -> None  # Line 1936
    async def collect_resource_metrics(self) -> dict[str, Any]  # Line 1956
    async def allocate_resources_with_priority(self, bot_id: str, resource_request: dict[str, Any], priority: Any) -> bool  # Line 1976
    async def commit_resource_reservation(self, reservation_id: str) -> bool  # Line 1996
    async def health_check(self) -> dict[str, Any]  # Line 2014
```

### File: resource_service.py

**Key Imports:**
- Note: Interfaces are implemented within the service class itself
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `BotResourceService`

**Inherits**: BaseService, IResourceManagementService
**Purpose**: Service for managing bot resources with dependency injection

```python
class BotResourceService(BaseService, IResourceManagementService):
    def __init__(self, ...)  # Line 29
    def _load_resource_limits(self) -> dict[str, Any]  # Line 63
    async def request_resources(self, ...) -> bool  # Line 79
    async def release_resources(self, bot_id: str) -> bool  # Line 127
    async def verify_resources(self, bot_id: str) -> bool  # Line 156
    async def get_resource_summary(self) -> dict[str, Any]  # Line 183
    async def check_resource_availability(self, resource_type: str, amount: Decimal) -> bool  # Line 218
    async def update_capital_allocation(self, bot_id: str, new_amount: Decimal) -> bool  # Line 244
    def _get_current_timestamp(self)  # Line 287
    def _get_current_timestamp_iso(self) -> str  # Line 293
```

### File: service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.types import BotPriority`

#### Class: `ResourceManagementService`

**Inherits**: BaseService
**Purpose**: Comprehensive resource management service using service layer pattern

```python
class ResourceManagementService(BaseService):
    def __init__(self)  # Line 30
    async def _do_start(self) -> None  # Line 63
    async def _do_stop(self) -> None  # Line 85
    async def _load_configuration(self) -> None  # Line 105
    async def _initialize_resource_limits(self) -> None  # Line 118
    async def request_resources(self, ...) -> bool  # Line 150
    async def _request_resources_impl(self, bot_id: str, capital_amount: Decimal, priority: BotPriority) -> bool  # Line 168
    async def release_resources(self, bot_id: str) -> bool  # Line 236
    async def _release_resources_impl(self, bot_id: str) -> bool  # Line 250
    async def verify_resources(self, bot_id: str) -> bool  # Line 287
    async def _verify_resources_impl(self, bot_id: str) -> bool  # Line 301
    async def update_resource_usage(self, bot_id: str, usage_data: dict[str, float]) -> bool  # Line 322
    async def _update_resource_usage_impl(self, bot_id: str, usage_data: dict[str, float]) -> bool  # Line 337
    async def reserve_resources(self, ...) -> str | None  # Line 366
    async def _reserve_resources_impl(self, bot_id: str, amount: Decimal, priority: BotPriority, duration_minutes: int) -> str | None  # Line 390
    async def get_resource_summary(self) -> dict[str, Any]  # Line 450
    async def _get_resource_summary_impl(self) -> dict[str, Any]  # Line 456
    async def get_optimization_suggestions(self) -> list[dict[str, Any]]  # Line 518
    async def _get_optimization_suggestions_impl(self) -> list[dict[str, Any]]  # Line 524
    async def detect_resource_conflicts(self) -> list[dict[str, Any]]  # Line 561
    async def _detect_resource_conflicts_impl(self) -> list[dict[str, Any]]  # Line 567
    async def _calculate_resource_requirements(self, bot_id: str, capital_amount: Decimal, priority: BotPriority) -> dict[str, Decimal]  # Line 595
    async def _check_resource_availability(self, requirements: dict[str, Decimal]) -> dict[str, Any]  # Line 641
    async def _allocate_resources(self, bot_id: str, requirements: dict[str, Decimal], priority: BotPriority) -> dict[str, Any]  # Line 682
    async def _reallocate_for_high_priority(self, bot_id: str, requirements: dict[str, Decimal], priority: BotPriority) -> bool  # Line 711
    async def _release_resource_allocation(self, bot_id: str, resource_type: str, allocation: dict[str, Any]) -> None  # Line 736
    async def _verify_resource_allocation(self, bot_id: str, resource_type: str, allocation: dict[str, Any]) -> bool  # Line 754
    async def _check_resource_violations(self, bot_id: str, usage_data: dict[str, float]) -> list[dict[str, Any]]  # Line 788
    async def _release_all_resources(self) -> None  # Line 820
    async def _cleanup_expired_reservations(self) -> int  # Line 832
    async def _monitoring_loop(self) -> None  # Line 871
    async def _update_resource_usage_tracking(self) -> None  # Line 902
    async def _check_all_resource_violations(self) -> None  # Line 928
    async def _optimize_resource_allocations(self) -> None  # Line 954
    async def _service_health_check(self) -> Any  # Line 969
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
    def __init__(self, ...)  # Line 83
    async def _do_start(self) -> None  # Line 200
    async def execute_with_monitoring(self, operation_name: str, operation_func: Any, *args, **kwargs) -> Any  # Line 259
    async def _do_stop(self) -> None  # Line 316
    async def create_bot(self, bot_config: BotConfiguration) -> str  # Line 330
    async def _create_bot_impl(self, bot_config: BotConfiguration) -> str  # Line 346
    async def start_bot(self, bot_id: str) -> bool  # Line 536
    async def _start_bot_impl(self, bot_id: str) -> bool  # Line 551
    async def stop_bot(self, bot_id: str) -> bool  # Line 752
    async def _stop_bot_impl(self, bot_id: str) -> bool  # Line 764
    async def delete_bot(self, bot_id: str, force: bool = False) -> bool  # Line 870
    async def _delete_bot_impl(self, bot_id: str, force: bool = False) -> bool  # Line 885
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]  # Line 974
    async def _get_bot_status_impl(self, bot_id: str) -> dict[str, Any]  # Line 988
    async def get_all_bots_status(self) -> dict[str, Any]  # Line 1044
    async def _get_all_bots_status_impl(self) -> dict[str, Any]  # Line 1055
    async def update_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> bool  # Line 1094
    async def _update_bot_metrics_impl(self, bot_id: str, metrics: dict[str, Any]) -> bool  # Line 1109
    async def _handle_risk_action(self, bot_id: str, risk_metrics: dict[str, Any]) -> None  # Line 1213
    async def _handle_high_risk_bot(self, bot_id: str, risk_metrics: Any) -> None  # Line 1259
    def _setup_event_handlers(self) -> None  # Line 1302
    def _calculate_bot_rate_requirements(self, bot_config: BotConfiguration) -> int  # Line 1319
    async def start_all_bots(self, priority_filter: BotPriority | None = None) -> dict[str, bool]  # Line 1360
    async def _start_all_bots_impl(self, priority_filter: BotPriority | None = None) -> dict[str, bool]  # Line 1374
    async def stop_all_bots(self) -> dict[str, bool]  # Line 1409
    async def _stop_all_bots_impl(self) -> dict[str, bool]  # Line 1418
    async def perform_health_check(self, bot_id: str) -> dict[str, Any]  # Line 1440
    async def _perform_health_check_impl(self, bot_id: str) -> dict[str, Any]  # Line 1454
    async def _validate_bot_configuration(self, bot_config: BotConfiguration) -> None  # Line 1533
    async def _validate_exchange_configuration(self, bot_config: BotConfiguration) -> None  # Line 1569
    async def _stop_all_active_bots(self) -> None  # Line 1684
    async def _load_existing_bot_states(self) -> None  # Line 1699
    async def _service_health_check(self) -> Any  # Line 1790
```

---
**Generated**: Complete reference for bot_management module
**Total Classes**: 23
**Total Functions**: 8