# CAPITAL_MANAGEMENT Module Reference

## INTEGRATION
**Dependencies**: core, database, error_handling, risk_management, state, utils
**Used By**: strategies
**Provides**: AbstractCapitalService, AbstractCurrencyManagementService, AbstractExchangeDistributionService, AbstractFundFlowManagementService, CapitalService, CurrencyManager, FundFlowManager, MinimalValidationService
**Patterns**: Async Operations, Circuit Breaker, Component Architecture

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Performance**:
- Parallel execution
- Parallel execution
**Architecture**:
- CapitalAllocator inherits from base architecture

## MODULE OVERVIEW
**Files**: 11 Python files
**Classes**: 25
**Functions**: 13

## COMPLETE API REFERENCE

## PROTOCOLS & INTERFACES

### Protocol: `CapitalRepositoryProtocol`

**Purpose**: Protocol for capital allocation repository operations

**Required Methods:**
- `async create(self, allocation_data: dict[str, Any]) -> Any`
- `async update(self, allocation_data: dict[str, Any]) -> Any`
- `async delete(self, allocation_id: str) -> bool`
- `async get_by_strategy_exchange(self, strategy_id: str, exchange: str) -> Any | None`
- `async get_by_strategy(self, strategy_id: str) -> list[Any]`
- `async get_all(self, limit: int | None = None) -> list[Any]`

### Protocol: `AuditRepositoryProtocol`

**Purpose**: Protocol for audit log repository operations

**Required Methods:**
- `async create(self, audit_data: dict[str, Any]) -> Any`

### Protocol: `CapitalServiceProtocol`

**Purpose**: Protocol for capital management service operations

**Required Methods:**
- `async allocate_capital(self, ...) -> CapitalAllocation`
- `async release_capital(self, ...) -> bool`
- `async update_utilization(self, ...) -> bool`
- `async get_capital_metrics(self) -> CapitalMetrics`
- `async get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]`
- `async get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]`

### Protocol: `CapitalAllocatorProtocol`

**Purpose**: Protocol for capital allocator operations

**Required Methods:**
- `async allocate_capital(self, strategy_id: str, exchange: str, requested_amount: Decimal, **kwargs) -> CapitalAllocation`
- `async release_capital(self, strategy_id: str, exchange: str, release_amount: Decimal, **kwargs) -> bool`
- `async get_capital_metrics(self) -> CapitalMetrics`

### Protocol: `CurrencyManagementServiceProtocol`

**Purpose**: Protocol for currency management service operations

**Required Methods:**
- `async update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CapitalCurrencyExposure]`
- `async calculate_hedging_requirements(self) -> dict[str, Decimal]`
- `async execute_currency_conversion(self, from_currency: str, to_currency: str, amount: Decimal, exchange: str) -> CapitalFundFlow`
- `async get_currency_risk_metrics(self) -> dict[str, dict[str, float]]`

### Protocol: `ExchangeDistributionServiceProtocol`

**Purpose**: Protocol for exchange distribution service operations

**Required Methods:**
- `async distribute_capital(self, total_amount: Decimal) -> dict[str, CapitalExchangeAllocation]`
- `async rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]`
- `async get_exchange_allocation(self, exchange: str) -> CapitalExchangeAllocation | None`
- `async update_exchange_utilization(self, exchange: str, utilized_amount: Decimal) -> None`

### Protocol: `FundFlowManagementServiceProtocol`

**Purpose**: Protocol for fund flow management service operations

**Required Methods:**
- `async process_deposit(self, ...) -> CapitalFundFlow`
- `async process_withdrawal(self, ...) -> CapitalFundFlow`
- `async process_strategy_reallocation(self, ...) -> CapitalFundFlow`
- `async get_flow_history(self, days: int = 30) -> list[CapitalFundFlow]`

## IMPLEMENTATIONS

### Implementation: `CapitalAllocator` ✅

**Inherits**: BaseComponent
**Purpose**: Dynamic capital allocation framework using enterprise CapitalService
**Status**: Complete

**Implemented Methods:**
- `async allocate_capital(self, ...) -> CapitalAllocation` - Line 165
- `async release_capital(self, ...) -> bool` - Line 239
- `async rebalance_allocations(self, authorized_by: str | None = None) -> dict[str, CapitalAllocation]` - Line 298
- `async update_utilization(self, ...) -> None` - Line 351
- `async get_capital_metrics(self) -> CapitalMetrics` - Line 398
- `async get_emergency_reserve(self) -> Decimal` - Line 578
- `async get_allocation_summary(self) -> dict[str, Any]` - Line 588
- `async reserve_capital_for_trade(self, ...) -> CapitalAllocation | None` - Line 630
- `async release_capital_from_trade(self, ...) -> bool` - Line 706
- `async get_trade_capital_efficiency(self, trade_id: str, strategy_id: str, exchange: str, realized_pnl: Decimal) -> dict[str, Any]` - Line 781

### Implementation: `CurrencyManager` 🔧

**Inherits**: AbstractCurrencyManagementService, TransactionalService
**Purpose**: Multi-currency capital management system
**Status**: Abstract Base Class

**Implemented Methods:**
- `async update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CurrencyExposure]` - Line 164
- `async calculate_hedging_requirements(self) -> dict[str, Decimal]` - Line 196
- `async execute_currency_conversion(self, from_currency: str, to_currency: str, amount: Decimal, exchange: str) -> FundFlow` - Line 231
- `async optimize_currency_allocation(self, target_allocations: dict[str, Decimal]) -> dict[str, Decimal]` - Line 314
- `async get_currency_risk_metrics(self) -> dict[str, dict[str, float]]` - Line 355
- `async get_total_base_value(self) -> Decimal` - Line 558
- `async get_currency_exposure(self, currency: str) -> CurrencyExposure | None` - Line 565
- `async get_exchange_rate(self, from_currency: str, to_currency: str) -> Decimal | None` - Line 569
- `async update_hedge_position(self, currency: str, hedge_amount: Decimal) -> None` - Line 584
- `async get_hedging_summary(self) -> dict[str, Any]` - Line 594
- `async cleanup_resources(self) -> None` - Line 623

### Implementation: `CapitalDataTransformer` ✅

**Purpose**: Handles consistent data transformation for capital_management module
**Status**: Complete

**Implemented Methods:**
- `transform_allocation_to_event_data(allocation: CapitalAllocation, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 25
- `transform_metrics_to_event_data(metrics: CapitalMetrics, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 68
- `transform_error_to_event_data(error, ...) -> dict[str, Any]` - Line 119
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 160
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'capital_management') -> dict[str, Any]` - Line 174
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 189
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]` - Line 220
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 252
- `transform_allocation_data_for_state(cls, allocation_data: dict[str, Any], operation: str = 'unknown') -> dict[str, Any]` - Line 310
- `validate_boundary_fields(data: dict[str, Any]) -> dict[str, Any]` - Line 338
- `apply_cross_module_consistency(cls, ...) -> dict[str, Any]` - Line 395
- `propagate_capital_error(cls, ...) -> dict[str, Any]` - Line 488

### Implementation: `ExchangeDistributor` 🔧

**Inherits**: AbstractExchangeDistributionService, TransactionalService
**Purpose**: Multi-exchange capital distribution manager
**Status**: Abstract Base Class

**Implemented Methods:**
- `async distribute_capital(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]` - Line 174
- `async rebalance_exchanges(self) -> dict[str, ExchangeAllocation]` - Line 219
- `async get_exchange_allocation(self, exchange: str) -> ExchangeAllocation | None` - Line 257
- `async update_exchange_utilization(self, exchange: str, utilized_amount: Decimal) -> None` - Line 270
- `async calculate_optimal_distribution(self, total_capital: Decimal) -> dict[str, Decimal]` - Line 300
- `supported_exchanges(self) -> list[str]` - Line 845
- `total_capital(self) -> Decimal` - Line 850
- `async get_exchange_metrics(self) -> dict[str, dict[str, float]]` - Line 856
- `async get_distribution_summary(self) -> dict[str, Any]` - Line 877
- `async handle_failed_exchange(self, exchange_name: str) -> None` - Line 1029
- `async get_available_exchanges(self) -> list[str]` - Line 1075
- `async emergency_redistribute(self, from_exchange: str) -> list[Any]` - Line 1089
- `get_allocation_history(self, exchange: str) -> list[dict]` - Line 1220
- `async check_exchange_health(self) -> dict[str, bool]` - Line 1240
- `async check_individual_exchange_health(self, exchange: str) -> bool` - Line 1256
- `async cleanup_resources(self) -> None` - Line 1275

### Implementation: `CapitalServiceFactory` ✅

**Inherits**: BaseFactory['CapitalService']
**Purpose**: Factory for creating CapitalService instances
**Status**: Complete

**Implemented Methods:**
- `create(self, name: str = '', *args, **kwargs) -> 'CapitalService'` - Line 30

### Implementation: `CapitalAllocatorFactory` ✅

**Inherits**: BaseFactory['CapitalAllocator']
**Purpose**: Factory for creating CapitalAllocator instances
**Status**: Complete

**Implemented Methods:**
- `create(self, name: str = '', *args, **kwargs) -> 'CapitalAllocator'` - Line 49

### Implementation: `CurrencyManagerFactory` ✅

**Inherits**: BaseFactory['CurrencyManager']
**Purpose**: Factory for creating CurrencyManager instances
**Status**: Complete

**Implemented Methods:**
- `create(self, name: str = '', *args, **kwargs) -> 'CurrencyManager'` - Line 68

### Implementation: `ExchangeDistributorFactory` ✅

**Inherits**: BaseFactory['ExchangeDistributor']
**Purpose**: Factory for creating ExchangeDistributor instances
**Status**: Complete

**Implemented Methods:**
- `create(self, name: str = '', *args, **kwargs) -> 'ExchangeDistributor'` - Line 87

### Implementation: `FundFlowManagerFactory` ✅

**Inherits**: BaseFactory['FundFlowManager']
**Purpose**: Factory for creating FundFlowManager instances
**Status**: Complete

**Implemented Methods:**
- `create(self, name: str = '', *args, **kwargs) -> 'FundFlowManager'` - Line 106

### Implementation: `CapitalManagementFactory` ✅

**Purpose**: Simple factory for all capital management services
**Status**: Complete

**Implemented Methods:**
- `dependency_container(self) -> Any` - Line 132
- `correlation_id(self) -> str | None` - Line 137
- `capital_service_factory(self) -> 'CapitalServiceFactory'` - Line 142
- `capital_allocator_factory(self) -> 'CapitalAllocatorFactory'` - Line 147
- `currency_manager_factory(self) -> 'CurrencyManagerFactory'` - Line 152
- `exchange_distributor_factory(self) -> 'ExchangeDistributorFactory'` - Line 157
- `fund_flow_manager_factory(self) -> 'FundFlowManagerFactory'` - Line 162
- `create_capital_service(self, **kwargs) -> 'CapitalService'` - Line 166
- `create_capital_allocator(self, **kwargs) -> 'CapitalAllocator'` - Line 205
- `create_currency_manager(self, **kwargs) -> 'CurrencyManager'` - Line 270
- `create_exchange_distributor(self, **kwargs) -> 'ExchangeDistributor'` - Line 316
- `create_fund_flow_manager(self, **kwargs) -> 'FundFlowManager'` - Line 362
- `register_factories(self, container: Any) -> None` - Line 415

### Implementation: `FundFlowManager` 🔧

**Inherits**: AbstractFundFlowManagementService, TransactionalService
**Purpose**: Deposit/withdrawal management system
**Status**: Abstract Base Class

**Implemented Methods:**
- `async process_deposit(self, ...) -> FundFlow` - Line 331
- `async process_withdrawal(self, ...) -> FundFlow` - Line 404
- `async process_strategy_reallocation(self, ...) -> FundFlow` - Line 494
- `async process_auto_compound(self) -> FundFlow | None` - Line 569
- `async update_performance(self, strategy_id: str, performance_metrics: dict[str, float]) -> None` - Line 625
- `async get_flow_history(self, days: int = 30) -> list[FundFlow]` - Line 655
- `async get_flow_summary(self, days: int = 30) -> dict[str, Any]` - Line 677
- `async update_total_capital(self, total_capital: Decimal) -> None` - Line 1019
- `async get_total_capital(self) -> Decimal` - Line 1033
- `async get_capital_protection_status(self) -> dict[str, Any]` - Line 1037
- `async get_performance_summary(self) -> dict[str, Any]` - Line 1065
- `async cleanup_resources(self) -> None` - Line 1157
- `set_capital_allocator(self, capital_allocator: Any) -> None` - Line 1239

### Implementation: `AbstractCapitalService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for capital service implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async allocate_capital(self, ...) -> CapitalAllocation` - Line 205
- `async release_capital(self, ...) -> bool` - Line 219
- `async get_capital_metrics(self) -> CapitalMetrics` - Line 230
- `async get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]` - Line 235

### Implementation: `AbstractCurrencyManagementService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for currency management service implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CapitalCurrencyExposure]` - Line 244
- `async calculate_hedging_requirements(self) -> dict[str, Decimal]` - Line 251

### Implementation: `AbstractExchangeDistributionService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for exchange distribution service implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async distribute_capital(self, total_amount: Decimal) -> dict[str, CapitalExchangeAllocation]` - Line 260
- `async rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]` - Line 267

### Implementation: `AbstractFundFlowManagementService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for fund flow management service implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async process_deposit(self, ...) -> CapitalFundFlow` - Line 276
- `async process_withdrawal(self, ...) -> CapitalFundFlow` - Line 286

### Implementation: `CapitalRepository` ✅

**Inherits**: CapitalRepositoryProtocol
**Purpose**: Service-layer adapter that implements CapitalRepositoryProtocol
**Status**: Complete

**Implemented Methods:**
- `async create(self, allocation_data: dict[str, Any]) -> Any` - Line 33
- `async update(self, allocation_data: dict[str, Any]) -> Any` - Line 56
- `async delete(self, allocation_id: str) -> bool` - Line 85
- `async get_by_strategy_exchange(self, strategy_id: str, exchange: str) -> Any | None` - Line 96
- `async get_by_strategy(self, strategy_id: str) -> list[Any]` - Line 106
- `async get_all(self, limit: int | None = None) -> list[Any]` - Line 116

### Implementation: `AuditRepository` ✅

**Inherits**: AuditRepositoryProtocol
**Purpose**: Service-layer adapter for audit operations with proper infrastructure abstraction
**Status**: Complete

**Implemented Methods:**
- `async create(self, audit_data: dict[str, Any]) -> Any` - Line 141

### Implementation: `CapitalService` 🔧

**Inherits**: AbstractCapitalService, TransactionalService, ErrorPropagationMixin
**Purpose**: Simple capital management service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async allocate_capital(self, ...) -> CapitalAllocation` - Line 90
- `async release_capital(self, ...) -> bool` - Line 225
- `async update_utilization(self, ...) -> bool` - Line 310
- `async get_capital_metrics(self) -> CapitalMetrics` - Line 385
- `async get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]` - Line 459
- `async get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]` - Line 472

## COMPLETE API REFERENCE

### File: capital_allocator.py

**Key Imports:**
- `from src.capital_management.constants import DEFAULT_EMERGENCY_RESERVE_PCT`
- `from src.capital_management.constants import DEFAULT_PERFORMANCE_WINDOW_DAYS`
- `from src.capital_management.constants import FINANCIAL_DECIMAL_PRECISION`
- `from src.capital_management.constants import HIGH_PORTFOLIO_EXPOSURE_THRESHOLD`
- `from src.capital_management.constants import LOW_ALLOCATION_RATIO_THRESHOLD`

#### Class: `CapitalAllocator`

**Inherits**: BaseComponent
**Purpose**: Dynamic capital allocation framework using enterprise CapitalService

```python
class CapitalAllocator(BaseComponent):
    def __init__(self, ...) -> None  # Line 108
    async def allocate_capital(self, ...) -> CapitalAllocation  # Line 165
    async def release_capital(self, ...) -> bool  # Line 239
    async def rebalance_allocations(self, authorized_by: str | None = None) -> dict[str, CapitalAllocation]  # Line 298
    async def update_utilization(self, ...) -> None  # Line 351
    async def get_capital_metrics(self) -> CapitalMetrics  # Line 398
    async def _assess_allocation_risk(self, strategy_id: str, exchange: str, amount: Decimal) -> dict[str, Any]  # Line 425
    async def _assess_standard_risk_interface(self, risk_assessment: dict[str, Any]) -> None  # Line 445
    async def _check_standard_risk_level(self, risk_assessment: dict[str, Any]) -> None  # Line 457
    async def _check_portfolio_exposure(self, risk_assessment: dict[str, Any]) -> None  # Line 471
    async def _check_legacy_risk_level(self, risk_assessment: dict[str, Any]) -> None  # Line 486
    def _handle_unknown_risk_service(self, risk_assessment: dict[str, Any]) -> None  # Line 506
    def _handle_risk_assessment_error(self, error: Exception, risk_assessment: dict[str, Any]) -> None  # Line 513
    async def _should_rebalance(self, current_metrics: CapitalMetrics) -> bool  # Line 535
    async def _calculate_performance_metrics(self) -> dict[str, dict[str, Decimal]]  # Line 570
    async def get_emergency_reserve(self) -> Decimal  # Line 578
    async def get_allocation_summary(self) -> dict[str, Any]  # Line 588
    async def reserve_capital_for_trade(self, ...) -> CapitalAllocation | None  # Line 630
    async def release_capital_from_trade(self, ...) -> bool  # Line 706
    async def get_trade_capital_efficiency(self, trade_id: str, strategy_id: str, exchange: str, realized_pnl: Decimal) -> dict[str, Any]  # Line 781
    async def _get_allocation(self, strategy_id: str, exchange: str) -> CapitalAllocation | None  # Line 857
```

### File: currency_manager.py

**Key Imports:**
- `from src.capital_management.constants import DEFAULT_BASE_CURRENCY`
- `from src.capital_management.constants import DEFAULT_CONNECTION_TIMEOUT`
- `from src.capital_management.constants import DEFAULT_CONVERSION_FEE_RATE`
- `from src.capital_management.constants import DEFAULT_HEDGE_RATIO`
- `from src.capital_management.constants import DEFAULT_HEDGING_THRESHOLD`

#### Class: `CurrencyManager`

**Inherits**: AbstractCurrencyManagementService, TransactionalService
**Purpose**: Multi-currency capital management system

```python
class CurrencyManager(AbstractCurrencyManagementService, TransactionalService):
    def __init__(self, ...) -> None  # Line 75
    async def _do_start(self) -> None  # Line 113
    async def _do_stop(self) -> None  # Line 140
    async def _load_configuration(self) -> None  # Line 150
    async def update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CurrencyExposure]  # Line 164
    async def calculate_hedging_requirements(self) -> dict[str, Decimal]  # Line 196
    async def execute_currency_conversion(self, from_currency: str, to_currency: str, amount: Decimal, exchange: str) -> FundFlow  # Line 231
    async def optimize_currency_allocation(self, target_allocations: dict[str, Decimal]) -> dict[str, Decimal]  # Line 314
    async def get_currency_risk_metrics(self) -> dict[str, dict[str, float]]  # Line 355
    def _initialize_currencies(self) -> None  # Line 408
    async def _update_exchange_rates(self) -> None  # Line 425
    async def _optimize_conversions(self, required_changes: dict[str, Decimal]) -> dict[str, Decimal]  # Line 517
    async def get_total_base_value(self) -> Decimal  # Line 558
    async def get_currency_exposure(self, currency: str) -> CurrencyExposure | None  # Line 565
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Decimal | None  # Line 569
    async def update_hedge_position(self, currency: str, hedge_amount: Decimal) -> None  # Line 584
    async def get_hedging_summary(self) -> dict[str, Any]  # Line 594
    async def _cleanup_rate_history(self) -> None  # Line 610
    async def cleanup_resources(self) -> None  # Line 623
    async def _calculate_currency_volatility_via_risk_service(self, currency: str, rate_history: list[tuple[datetime, Decimal]]) -> Decimal  # Line 663
    async def _calculate_currency_var_via_risk_service(self, exposure_amount: Decimal, rate_history: list[tuple[datetime, Decimal]]) -> Decimal  # Line 684
    def _fallback_risk_calculation(self, ...) -> tuple[Decimal, Decimal]  # Line 714
    async def _validate_currencies(self, balances: dict[str, dict[str, Decimal]]) -> None  # Line 754
    def _calculate_total_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, Decimal]  # Line 761
    def _calculate_base_equivalents(self, total_exposures: dict[str, Decimal]) -> tuple[dict[str, Decimal], Decimal]  # Line 773
    def _create_currency_exposures(self, ...) -> dict[str, CurrencyExposure]  # Line 792
    def _log_exposure_update(self, exposures: dict[str, CurrencyExposure], total_base_value: Decimal) -> None  # Line 833
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import CapitalAllocation`
- `from src.core.types import CapitalMetrics`
- `from src.core.types import StateType`

#### Class: `CapitalDataTransformer`

**Purpose**: Handles consistent data transformation for capital_management module

```python
class CapitalDataTransformer:
    def transform_allocation_to_event_data(allocation: CapitalAllocation, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 25
    def transform_metrics_to_event_data(metrics: CapitalMetrics, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 68
    def transform_error_to_event_data(error, ...) -> dict[str, Any]  # Line 119
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 160
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'capital_management') -> dict[str, Any]  # Line 174
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 189
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]  # Line 220
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 252
    def transform_allocation_data_for_state(cls, allocation_data: dict[str, Any], operation: str = 'unknown') -> dict[str, Any]  # Line 310
    def validate_boundary_fields(data: dict[str, Any]) -> dict[str, Any]  # Line 338
    def apply_cross_module_consistency(cls, ...) -> dict[str, Any]  # Line 395
    def propagate_capital_error(cls, ...) -> dict[str, Any]  # Line 488
    def _apply_boundary_validation(data: dict[str, Any], source_module: str, target_module: str) -> None  # Line 519
```

### File: di_registration.py

#### Functions:

```python
def register_capital_management_services(container: Any) -> None  # Line 29
def _register_fallback_services(container: Any) -> None  # Line 237
def _setup_cross_dependencies(container: Any) -> None  # Line 257
def _has_service(container: Any, service_name: str) -> bool  # Line 269
def _register_capital_repositories(container: Any) -> None  # Line 278
def _create_repository_with_fallback(container: Any, repo_class_name: str, db_repo_class_name: str, fallback_creator)  # Line 307
def _create_minimal_capital_repository()  # Line 342
def _create_minimal_audit_repository()  # Line 367
```

### File: exchange_distributor.py

**Key Imports:**
- `from src.capital_management.constants import CONNECTIVITY_PENALTY`
- `from src.capital_management.constants import DEFAULT_EXCHANGE`
- `from src.capital_management.constants import DEFAULT_EXCHANGE_SCORE`
- `from src.capital_management.constants import DEFAULT_EXCHANGE_TIMEOUT_SECONDS`
- `from src.capital_management.constants import DEFAULT_FEE_QUERY_TIMEOUT_SECONDS`

#### Class: `ExchangeDistributor`

**Inherits**: AbstractExchangeDistributionService, TransactionalService
**Purpose**: Multi-exchange capital distribution manager

```python
class ExchangeDistributor(AbstractExchangeDistributionService, TransactionalService):
    def __init__(self, ...) -> None  # Line 72
    async def _do_start(self) -> None  # Line 108
    async def _do_stop(self) -> None  # Line 132
    async def _load_configuration(self) -> None  # Line 142
    def _validate_config(self) -> None  # Line 155
    async def distribute_capital(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]  # Line 174
    async def rebalance_exchanges(self) -> dict[str, ExchangeAllocation]  # Line 219
    async def get_exchange_allocation(self, exchange: str) -> ExchangeAllocation | None  # Line 257
    async def update_exchange_utilization(self, exchange: str, utilized_amount: Decimal) -> None  # Line 270
    async def calculate_optimal_distribution(self, total_capital: Decimal) -> dict[str, Decimal]  # Line 300
    async def _initialize_exchange_allocations(self) -> None  # Line 369
    async def _update_exchange_metrics(self) -> None  # Line 384
    async def _calculate_liquidity_score(self, exchange_name: str) -> Decimal  # Line 427
    async def _calculate_fee_efficiency(self, exchange_name: str) -> Decimal  # Line 519
    async def _calculate_reliability_score(self, exchange_name: str) -> Decimal  # Line 572
    async def _update_slippage_data(self, exchange_name: str) -> None  # Line 637
    async def _dynamic_distribution(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]  # Line 709
    async def _weighted_distribution(self, total_amount: Decimal, weights: dict[str, Decimal]) -> dict[str, ExchangeAllocation]  # Line 733
    async def _apply_minimum_balances(self, allocations: dict[str, ExchangeAllocation]) -> dict[str, ExchangeAllocation]  # Line 756
    async def _calculate_optimal_distribution(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]  # Line 777
    async def _apply_rebalancing_limits(self, new_allocations: dict[str, ExchangeAllocation]) -> dict[str, ExchangeAllocation]  # Line 805
    def supported_exchanges(self) -> list[str]  # Line 845
    def total_capital(self) -> Decimal  # Line 850
    async def get_exchange_metrics(self) -> dict[str, dict[str, float]]  # Line 856
    async def get_distribution_summary(self) -> dict[str, Any]  # Line 877
    def _validate_distribution_constraints(self, distribution: list[Any], total_capital: Decimal) -> None  # Line 906
    def _should_rebalance(self) -> bool  # Line 932
    def _calculate_distribution_efficiency(self, allocations: list[Any]) -> float  # Line 959
    def _apply_exchange_weights(self, weights: dict[str, float], total_capital: Decimal) -> list[Any]  # Line 997
    async def handle_failed_exchange(self, exchange_name: str) -> None  # Line 1029
    async def get_available_exchanges(self) -> list[str]  # Line 1075
    async def emergency_redistribute(self, from_exchange: str) -> list[Any]  # Line 1089
    async def _calculate_equal_distribution(self, total_capital: Decimal) -> list[Any]  # Line 1113
    async def _calculate_performance_based_distribution(self, total_capital: Decimal) -> list[Any]  # Line 1137
    def _add_to_allocation_history(self, allocation: Any) -> None  # Line 1186
    def get_allocation_history(self, exchange: str) -> list[dict]  # Line 1220
    async def check_exchange_health(self) -> dict[str, bool]  # Line 1240
    async def check_individual_exchange_health(self, exchange: str) -> bool  # Line 1256
    async def cleanup_resources(self) -> None  # Line 1275
```

### File: factory.py

**Key Imports:**
- `from src.core.base.factory import BaseFactory`
- `from src.core.exceptions import CreationError`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`

#### Class: `CapitalServiceFactory`

**Inherits**: BaseFactory['CapitalService']
**Purpose**: Factory for creating CapitalService instances

```python
class CapitalServiceFactory(BaseFactory['CapitalService']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 24
    def create(self, name: str = '', *args, **kwargs) -> 'CapitalService'  # Line 30
```

#### Class: `CapitalAllocatorFactory`

**Inherits**: BaseFactory['CapitalAllocator']
**Purpose**: Factory for creating CapitalAllocator instances

```python
class CapitalAllocatorFactory(BaseFactory['CapitalAllocator']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 43
    def create(self, name: str = '', *args, **kwargs) -> 'CapitalAllocator'  # Line 49
```

#### Class: `CurrencyManagerFactory`

**Inherits**: BaseFactory['CurrencyManager']
**Purpose**: Factory for creating CurrencyManager instances

```python
class CurrencyManagerFactory(BaseFactory['CurrencyManager']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 62
    def create(self, name: str = '', *args, **kwargs) -> 'CurrencyManager'  # Line 68
```

#### Class: `ExchangeDistributorFactory`

**Inherits**: BaseFactory['ExchangeDistributor']
**Purpose**: Factory for creating ExchangeDistributor instances

```python
class ExchangeDistributorFactory(BaseFactory['ExchangeDistributor']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 81
    def create(self, name: str = '', *args, **kwargs) -> 'ExchangeDistributor'  # Line 87
```

#### Class: `FundFlowManagerFactory`

**Inherits**: BaseFactory['FundFlowManager']
**Purpose**: Factory for creating FundFlowManager instances

```python
class FundFlowManagerFactory(BaseFactory['FundFlowManager']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 100
    def create(self, name: str = '', *args, **kwargs) -> 'FundFlowManager'  # Line 106
```

#### Class: `CapitalManagementFactory`

**Purpose**: Simple factory for all capital management services

```python
class CapitalManagementFactory:
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 119
    def dependency_container(self) -> Any  # Line 132
    def correlation_id(self) -> str | None  # Line 137
    def capital_service_factory(self) -> 'CapitalServiceFactory'  # Line 142
    def capital_allocator_factory(self) -> 'CapitalAllocatorFactory'  # Line 147
    def currency_manager_factory(self) -> 'CurrencyManagerFactory'  # Line 152
    def exchange_distributor_factory(self) -> 'ExchangeDistributorFactory'  # Line 157
    def fund_flow_manager_factory(self) -> 'FundFlowManagerFactory'  # Line 162
    def create_capital_service(self, **kwargs) -> 'CapitalService'  # Line 166
    def create_capital_allocator(self, **kwargs) -> 'CapitalAllocator'  # Line 205
    def create_currency_manager(self, **kwargs) -> 'CurrencyManager'  # Line 270
    def create_exchange_distributor(self, **kwargs) -> 'ExchangeDistributor'  # Line 316
    def create_fund_flow_manager(self, **kwargs) -> 'FundFlowManager'  # Line 362
    def register_factories(self, container: Any) -> None  # Line 415
```

#### Functions:

```python
def create_capital_service(**kwargs) -> 'CapitalService'  # Line 451
def create_capital_allocator(**kwargs) -> 'CapitalAllocator'  # Line 457
def create_currency_manager(**kwargs) -> 'CurrencyManager'  # Line 463
def create_exchange_distributor(**kwargs) -> 'ExchangeDistributor'  # Line 469
def create_fund_flow_manager(**kwargs) -> 'FundFlowManager'  # Line 475
```

### File: fund_flow_manager.py

**Key Imports:**
- `from src.capital_management.constants import COMPOUND_FREQUENCY_DAYS`
- `from src.capital_management.constants import DEFAULT_BASE_CURRENCY`
- `from src.capital_management.constants import DEFAULT_CACHE_OPERATION_TIMEOUT`
- `from src.capital_management.constants import DEFAULT_CACHE_TTL_SECONDS`
- `from src.capital_management.constants import DEFAULT_CLEANUP_TIMEOUT`

#### Class: `FundFlowManager`

**Inherits**: AbstractFundFlowManagementService, TransactionalService
**Purpose**: Deposit/withdrawal management system

```python
class FundFlowManager(AbstractFundFlowManagementService, TransactionalService):
    def __init__(self, ...) -> None  # Line 88
    async def _do_start(self) -> None  # Line 138
    async def _load_configuration(self) -> None  # Line 183
    async def _do_stop(self) -> None  # Line 189
    def _initialize_capital_protection(self) -> None  # Line 199
    async def _cache_fund_flows(self, flows: list[FundFlow]) -> None  # Line 230
    async def _get_cached_fund_flows(self) -> list[FundFlow] | None  # Line 270
    async def _store_fund_flow_time_series(self, flow: FundFlow) -> None  # Line 290
    async def process_deposit(self, ...) -> FundFlow  # Line 331
    async def process_withdrawal(self, ...) -> FundFlow  # Line 404
    async def process_strategy_reallocation(self, ...) -> FundFlow  # Line 494
    async def process_auto_compound(self) -> FundFlow | None  # Line 569
    async def update_performance(self, strategy_id: str, performance_metrics: dict[str, float]) -> None  # Line 625
    async def get_flow_history(self, days: int = 30) -> list[FundFlow]  # Line 655
    async def get_flow_summary(self, days: int = 30) -> dict[str, Any]  # Line 677
    def _initialize_withdrawal_rules(self) -> None  # Line 738
    async def _validate_withdrawal_rules(self, amount: Decimal, currency: str) -> None  # Line 767
    async def _check_withdrawal_cooldown(self) -> None  # Line 832
    async def _calculate_minimum_capital_required(self) -> Decimal  # Line 867
    async def _check_performance_threshold(self, threshold: Decimal) -> bool  # Line 901
    async def _get_daily_reallocation_amount(self) -> Decimal  # Line 942
    def _should_compound(self) -> bool  # Line 958
    async def _calculate_compound_amount(self) -> Decimal  # Line 979
    def _calculate_compound_schedule(self) -> dict[str, Any]  # Line 1002
    async def update_total_capital(self, total_capital: Decimal) -> None  # Line 1019
    async def get_total_capital(self) -> Decimal  # Line 1033
    async def get_capital_protection_status(self) -> dict[str, Any]  # Line 1037
    async def get_performance_summary(self) -> dict[str, Any]  # Line 1065
    def _validate_config(self) -> None  # Line 1127
    async def cleanup_resources(self) -> None  # Line 1157
    def set_capital_allocator(self, capital_allocator: Any) -> None  # Line 1239
```

### File: interfaces.py

**Key Imports:**
- `from src.capital_management.constants import DEFAULT_BASE_CURRENCY`
- `from src.capital_management.constants import DEFAULT_EXCHANGE`
- `from src.core.types import CapitalAllocation`
- `from src.core.types import CapitalMetrics`
- `from src.core.types.capital import CapitalCurrencyExposure`

#### Class: `AbstractCapitalService`

**Inherits**: ABC
**Purpose**: Abstract base class for capital service implementations

```python
class AbstractCapitalService(ABC):
    async def allocate_capital(self, ...) -> CapitalAllocation  # Line 205
    async def release_capital(self, ...) -> bool  # Line 219
    async def get_capital_metrics(self) -> CapitalMetrics  # Line 230
    async def get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]  # Line 235
```

#### Class: `AbstractCurrencyManagementService`

**Inherits**: ABC
**Purpose**: Abstract base class for currency management service implementations

```python
class AbstractCurrencyManagementService(ABC):
    async def update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CapitalCurrencyExposure]  # Line 244
    async def calculate_hedging_requirements(self) -> dict[str, Decimal]  # Line 251
```

#### Class: `AbstractExchangeDistributionService`

**Inherits**: ABC
**Purpose**: Abstract base class for exchange distribution service implementations

```python
class AbstractExchangeDistributionService(ABC):
    async def distribute_capital(self, total_amount: Decimal) -> dict[str, CapitalExchangeAllocation]  # Line 260
    async def rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]  # Line 267
```

#### Class: `AbstractFundFlowManagementService`

**Inherits**: ABC
**Purpose**: Abstract base class for fund flow management service implementations

```python
class AbstractFundFlowManagementService(ABC):
    async def process_deposit(self, ...) -> CapitalFundFlow  # Line 276
    async def process_withdrawal(self, ...) -> CapitalFundFlow  # Line 286
```

### File: service.py

**Key Imports:**
- `from src.capital_management.constants import DEFAULT_TOTAL_CAPITAL`
- `from src.capital_management.constants import EMERGENCY_RESERVE_PCT`
- `from src.capital_management.constants import FINANCIAL_DECIMAL_PRECISION`
- `from src.capital_management.constants import MAX_ALLOCATION_PCT`
- `from src.capital_management.constants import PERCENTAGE_MULTIPLIER`

#### Class: `CapitalService`

**Inherits**: AbstractCapitalService, TransactionalService, ErrorPropagationMixin
**Purpose**: Simple capital management service

```python
class CapitalService(AbstractCapitalService, TransactionalService, ErrorPropagationMixin):
    def __init__(self, ...) -> None  # Line 48
    async def _do_start(self) -> None  # Line 80
    async def _do_stop(self) -> None  # Line 84
    async def allocate_capital(self, ...) -> CapitalAllocation  # Line 90
    async def release_capital(self, ...) -> bool  # Line 225
    async def update_utilization(self, ...) -> bool  # Line 310
    async def get_capital_metrics(self) -> CapitalMetrics  # Line 385
    async def get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]  # Line 459
    async def get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]  # Line 472
    async def _get_available_capital(self) -> Decimal  # Line 486
```

---
**Generated**: Complete reference for capital_management module
**Total Classes**: 25
**Total Functions**: 13