# CAPITAL_MANAGEMENT Module Reference

## INTEGRATION
**Dependencies**: core, database, error_handling, exchanges, risk_management, state, utils
**Used By**: None
**Provides**: AbstractCapitalService, AbstractCurrencyManagementService, AbstractExchangeDistributionService, AbstractFundFlowManagementService, CapitalService, CurrencyManager, FundFlowManager
**Patterns**: Async Operations, Circuit Breaker, Component Architecture

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Performance**:
- Caching
**Architecture**:
- CapitalAllocator inherits from base architecture

## MODULE OVERVIEW
**Files**: 11 Python files
**Classes**: 31
**Functions**: 2

## COMPLETE API REFERENCE

## PROTOCOLS & INTERFACES

### Protocol: `ExchangeDataProtocol`

**Purpose**: Protocol for exchange data operations

**Required Methods:**
- `async fetch_tickers(self) -> dict[str, Any]`
- `async fetch_order_book(self, symbol: str, limit: int = 50) -> dict[str, Any]`
- `async fetch_status(self) -> dict[str, Any]`

### Protocol: `TimeSeriesServiceProtocol`

**Purpose**: Protocol for time series data storage

**Required Methods:**
- `async write_point(self, measurement: str, tags: dict[str, str], fields: dict[str, Any]) -> None`

### Protocol: `CacheServiceProtocol`

**Purpose**: Protocol for caching operations

**Required Methods:**
- `async get(self, key: str) -> Any`
- `async set(self, key: str, value: Any, ttl: int) -> None`

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

### Protocol: `ExchangeDataServiceProtocol`

**Purpose**: Protocol for exchange data service operations

**Required Methods:**
- `async get_tickers(self) -> dict[str, Any]`
- `async get_order_book(self, exchange: str, symbol: str, limit: int = 50) -> dict[str, Any]`
- `async get_status(self, exchange: str) -> dict[str, Any]`
- `async get_fees(self, exchange: str) -> dict[str, Any]`

### Protocol: `CapitalServiceProtocol`

**Purpose**: Protocol for capital management service operations

**Required Methods:**
- `async allocate_capital(self, ...) -> CapitalAllocation`
- `async release_capital(self, ...) -> bool`
- `async update_utilization(self, ...) -> bool`
- `async get_capital_metrics(self) -> CapitalMetrics`
- `async get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]`
- `async get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]`

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
- `async process_deposit(self, amount: Decimal, currency: str = 'USDT', exchange: str = 'binance') -> CapitalFundFlow`
- `async process_withdrawal(self, ...) -> CapitalFundFlow`
- `async process_strategy_reallocation(self, ...) -> CapitalFundFlow`
- `async get_flow_history(self, days: int = 30) -> list[CapitalFundFlow]`

## IMPLEMENTATIONS

### Implementation: `CapitalAllocator` ✅

**Inherits**: BaseComponent
**Purpose**: Dynamic capital allocation framework using enterprise CapitalService
**Status**: Complete

**Implemented Methods:**
- `async allocate_capital(self, ...) -> CapitalAllocation` - Line 159
- `async release_capital(self, ...) -> bool` - Line 233
- `async rebalance_allocations(self, authorized_by: str | None = None) -> dict[str, CapitalAllocation]` - Line 283
- `async update_utilization(self, ...) -> None` - Line 336
- `async get_capital_metrics(self) -> CapitalMetrics` - Line 383
- `async get_emergency_reserve(self) -> Decimal` - Line 542
- `async get_allocation_summary(self) -> dict[str, Any]` - Line 551
- `async reserve_capital_for_trade(self, ...) -> CapitalAllocation | None` - Line 591
- `async release_capital_from_trade(self, ...) -> bool` - Line 665
- `async get_trade_capital_efficiency(self, trade_id: str, strategy_id: str, exchange: str, realized_pnl: Decimal) -> dict[str, Any]` - Line 738

### Implementation: `CurrencyManager` 🔧

**Inherits**: AbstractCurrencyManagementService, TransactionalService
**Purpose**: Multi-currency capital management system
**Status**: Abstract Base Class

**Implemented Methods:**
- `async update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CurrencyExposure]` - Line 183
- `async calculate_hedging_requirements(self) -> dict[str, Decimal]` - Line 278
- `async execute_currency_conversion(self, from_currency: str, to_currency: str, amount: Decimal, exchange: str) -> FundFlow` - Line 315
- `async optimize_currency_allocation(self, target_allocations: dict[str, Decimal]) -> dict[str, Decimal]` - Line 406
- `async get_currency_risk_metrics(self) -> dict[str, dict[str, float]]` - Line 450
- `async get_total_base_value(self) -> Decimal` - Line 658
- `async get_currency_exposure(self, currency: str) -> CurrencyExposure | None` - Line 665
- `async get_exchange_rate(self, from_currency: str, to_currency: str) -> Decimal | None` - Line 669
- `async update_hedge_position(self, currency: str, hedge_amount: Decimal) -> None` - Line 686
- `async get_hedging_summary(self) -> dict[str, Any]` - Line 696
- `async cleanup_resources(self) -> None` - Line 724

### Implementation: `CapitalDataTransformer` ✅

**Purpose**: Handles consistent data transformation for capital_management module
**Status**: Complete

**Implemented Methods:**
- `transform_allocation_to_event_data(allocation: CapitalAllocation, metadata: dict[str, Any] = None) -> dict[str, Any]` - Line 19
- `transform_metrics_to_event_data(metrics: CapitalMetrics, metadata: dict[str, Any] = None) -> dict[str, Any]` - Line 52
- `transform_error_to_event_data(error, ...) -> dict[str, Any]` - Line 94
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 119
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'capital_management') -> dict[str, Any]` - Line 170
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] = None) -> dict[str, Any]` - Line 206
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str = None) -> dict[str, Any]` - Line 251
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 276
- `transform_allocation_data_for_state(cls, allocation_data: dict[str, Any], operation: str = 'unknown') -> dict[str, Any]` - Line 336

### Implementation: `ExchangeDistributor` 🔧

**Inherits**: AbstractExchangeDistributionService, TransactionalService
**Purpose**: Multi-exchange capital distribution manager
**Status**: Abstract Base Class

**Implemented Methods:**
- `async distribute_capital(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]` - Line 169
- `async rebalance_exchanges(self) -> dict[str, ExchangeAllocation]` - Line 221
- `async get_exchange_allocation(self, exchange: str) -> ExchangeAllocation | None` - Line 260
- `async update_exchange_utilization(self, exchange: str, utilized_amount: Decimal) -> None` - Line 273
- `async calculate_optimal_distribution(self, total_capital: Decimal) -> dict[str, Decimal]` - Line 305
- `supported_exchanges(self) -> list[str]` - Line 751
- `total_capital(self) -> Decimal` - Line 756
- `async get_exchange_metrics(self) -> dict[str, dict[str, float]]` - Line 762
- `async get_distribution_summary(self) -> dict[str, Any]` - Line 778
- `async cleanup_resources(self) -> None` - Line 803

### Implementation: `CapitalServiceFactory` ✅

**Inherits**: BaseFactory['CapitalService']
**Purpose**: Factory for creating CapitalService instances
**Status**: Complete

**Implemented Methods:**

### Implementation: `CapitalAllocatorFactory` ✅

**Inherits**: BaseFactory['CapitalAllocator']
**Purpose**: Factory for creating CapitalAllocator instances
**Status**: Complete

**Implemented Methods:**

### Implementation: `CurrencyManagerFactory` ✅

**Inherits**: BaseFactory['CurrencyManager']
**Purpose**: Factory for creating CurrencyManager instances
**Status**: Complete

**Implemented Methods:**

### Implementation: `ExchangeDistributorFactory` ✅

**Inherits**: BaseFactory['ExchangeDistributor']
**Purpose**: Factory for creating ExchangeDistributor instances
**Status**: Complete

**Implemented Methods:**

### Implementation: `FundFlowManagerFactory` ✅

**Inherits**: BaseFactory['FundFlowManager']
**Purpose**: Factory for creating FundFlowManager instances
**Status**: Complete

**Implemented Methods:**

### Implementation: `CapitalManagementFactory` ✅

**Purpose**: Composite factory for all capital management services
**Status**: Complete

**Implemented Methods:**
- `create_capital_service(self, **kwargs) -> 'CapitalService'` - Line 519
- `create_capital_allocator(self, **kwargs) -> 'CapitalAllocator'` - Line 523
- `create_currency_manager(self, **kwargs) -> 'CurrencyManager'` - Line 527
- `create_exchange_distributor(self, **kwargs) -> 'ExchangeDistributor'` - Line 531
- `create_fund_flow_manager(self, **kwargs) -> 'FundFlowManager'` - Line 535
- `register_factories(self, container: Any) -> None` - Line 539

### Implementation: `FundFlowManager` 🔧

**Inherits**: AbstractFundFlowManagementService, TransactionalService
**Purpose**: Deposit/withdrawal management system
**Status**: Abstract Base Class

**Implemented Methods:**
- `async process_deposit(self, amount: Decimal, currency: str = 'USDT', exchange: str = 'binance') -> FundFlow` - Line 301
- `async process_withdrawal(self, ...) -> FundFlow` - Line 369
- `async process_strategy_reallocation(self, ...) -> FundFlow` - Line 459
- `async process_auto_compound(self) -> FundFlow | None` - Line 536
- `async update_performance(self, strategy_id: str, performance_metrics: dict[str, float]) -> None` - Line 592
- `async get_flow_history(self, days: int = 30) -> list[FundFlow]` - Line 619
- `async get_flow_summary(self, days: int = 30) -> dict[str, Any]` - Line 641
- `async update_total_capital(self, total_capital: Decimal) -> None` - Line 962
- `async get_total_capital(self) -> Decimal` - Line 976
- `async get_capital_protection_status(self) -> dict[str, Any]` - Line 980
- `async get_performance_summary(self) -> dict[str, Any]` - Line 1008
- `set_capital_allocator(self, capital_allocator: Any) -> None` - Line 1070
- `async cleanup_resources(self) -> None` - Line 1110

### Implementation: `AbstractCapitalService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for capital service implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async allocate_capital(self, ...) -> CapitalAllocation` - Line 213
- `async release_capital(self, ...) -> bool` - Line 226
- `async get_capital_metrics(self) -> CapitalMetrics` - Line 238
- `async get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]` - Line 243

### Implementation: `AbstractCurrencyManagementService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for currency management service implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CapitalCurrencyExposure]` - Line 252
- `async calculate_hedging_requirements(self) -> dict[str, Decimal]` - Line 259

### Implementation: `AbstractExchangeDistributionService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for exchange distribution service implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async distribute_capital(self, total_amount: Decimal) -> dict[str, CapitalExchangeAllocation]` - Line 268
- `async rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]` - Line 275

### Implementation: `AbstractFundFlowManagementService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for fund flow management service implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async process_deposit(self, amount: Decimal, currency: str = 'USDT', exchange: str = 'binance') -> CapitalFundFlow` - Line 284
- `async process_withdrawal(self, ...) -> CapitalFundFlow` - Line 291

### Implementation: `CapitalRepository` ✅

**Inherits**: CapitalRepositoryProtocol
**Purpose**: Service-layer adapter that implements CapitalRepositoryProtocol
**Status**: Complete

**Implemented Methods:**
- `async create(self, allocation_data: dict[str, Any]) -> Any` - Line 36
- `async update(self, allocation_data: dict[str, Any]) -> Any` - Line 55
- `async delete(self, allocation_id: str) -> bool` - Line 87
- `async get_by_strategy_exchange(self, strategy_id: str, exchange: str) -> Any | None` - Line 96
- `async get_by_strategy(self, strategy_id: str) -> list[Any]` - Line 104
- `async get_all(self, limit: int | None = None) -> list[Any]` - Line 111

### Implementation: `AuditRepository` ✅

**Inherits**: AuditRepositoryProtocol
**Purpose**: Service-layer adapter for audit operations with proper infrastructure abstraction
**Status**: Complete

**Implemented Methods:**
- `async create(self, audit_data: dict[str, Any]) -> Any` - Line 132

### Implementation: `SimpleEventPattern` ✅

**Purpose**: Simple event pattern placeholder for testing compatibility
**Status**: Complete

**Implemented Methods:**
- `async emit_consistent(self, event_type, data)` - Line 120

### Implementation: `SimpleValidationPattern` ✅

**Purpose**: Simple validation pattern placeholder for testing compatibility
**Status**: Complete

**Implemented Methods:**
- `validate_consistent(self, data)` - Line 128

### Implementation: `SimpleProcessingPattern` ✅

**Purpose**: Simple processing pattern placeholder for testing compatibility
**Status**: Complete

**Implemented Methods:**
- `async process_item_consistent(self, item, *args, **kwargs)` - Line 136

### Implementation: `CapitalService` 🔧

**Inherits**: AbstractCapitalService, TransactionalService
**Purpose**: Enterprise-grade capital management service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async allocate_capital(self, ...) -> CapitalAllocation` - Line 330
- `async release_capital(self, ...) -> bool` - Line 674
- `async update_utilization(self, ...) -> bool` - Line 925
- `async get_capital_metrics(self) -> CapitalMetrics` - Line 1044
- `async get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]` - Line 1063
- `async get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]` - Line 1079
- `get_performance_metrics(self) -> dict[str, Any]` - Line 1616
- `reset_metrics(self) -> None` - Line 1626
- `async restore_capital_state(self) -> bool` - Line 1813
- `async cleanup_resources(self) -> None` - Line 1875

## COMPLETE API REFERENCE

### File: capital_allocator.py

**Key Imports:**
- `from src.capital_management.constants import DEFAULT_PERFORMANCE_WINDOW_DAYS`
- `from src.capital_management.constants import MAX_DAILY_REALLOCATION_PCT`
- `from src.capital_management.constants import LOW_ALLOCATION_RATIO_THRESHOLD`
- `from src.capital_management.constants import LOW_UTILIZATION_THRESHOLD`
- `from src.core.base.component import BaseComponent`

#### Class: `CapitalAllocator`

**Inherits**: BaseComponent
**Purpose**: Dynamic capital allocation framework using enterprise CapitalService

```python
class CapitalAllocator(BaseComponent):
    def __init__(self, ...) -> None  # Line 106
    async def allocate_capital(self, ...) -> CapitalAllocation  # Line 159
    async def release_capital(self, ...) -> bool  # Line 233
    async def rebalance_allocations(self, authorized_by: str | None = None) -> dict[str, CapitalAllocation]  # Line 283
    async def update_utilization(self, ...) -> None  # Line 336
    async def get_capital_metrics(self) -> CapitalMetrics  # Line 383
    async def _assess_allocation_risk(self, strategy_id: str, exchange: str, amount: Decimal) -> dict[str, Any]  # Line 410
    async def _should_rebalance(self, current_metrics: CapitalMetrics) -> bool  # Line 499
    async def _calculate_performance_metrics(self) -> dict[str, dict[str, Decimal]]  # Line 534
    async def get_emergency_reserve(self) -> Decimal  # Line 542
    async def get_allocation_summary(self) -> dict[str, Any]  # Line 551
    async def reserve_capital_for_trade(self, ...) -> CapitalAllocation | None  # Line 591
    async def release_capital_from_trade(self, ...) -> bool  # Line 665
    async def get_trade_capital_efficiency(self, trade_id: str, strategy_id: str, exchange: str, realized_pnl: Decimal) -> dict[str, Any]  # Line 738
    async def _get_allocation(self, strategy_id: str, exchange: str) -> CapitalAllocation | None  # Line 808
```

### File: currency_manager.py

**Key Imports:**
- `from src.capital_management.constants import DEFAULT_HEDGING_THRESHOLD`
- `from src.capital_management.constants import DEFAULT_HEDGE_RATIO`
- `from src.capital_management.constants import MIN_HEDGE_AMOUNT`
- `from src.capital_management.constants import MIN_CHANGE_THRESHOLD`
- `from src.capital_management.constants import DEFAULT_CONVERSION_FEE_RATE`

#### Class: `CurrencyManager`

**Inherits**: AbstractCurrencyManagementService, TransactionalService
**Purpose**: Multi-currency capital management system

```python
class CurrencyManager(AbstractCurrencyManagementService, TransactionalService):
    def __init__(self, ...) -> None  # Line 85
    async def _do_start(self) -> None  # Line 128
    async def _do_stop(self) -> None  # Line 158
    async def _load_configuration(self) -> None  # Line 168
    async def update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CurrencyExposure]  # Line 183
    async def calculate_hedging_requirements(self) -> dict[str, Decimal]  # Line 278
    async def execute_currency_conversion(self, from_currency: str, to_currency: str, amount: Decimal, exchange: str) -> FundFlow  # Line 315
    async def optimize_currency_allocation(self, target_allocations: dict[str, Decimal]) -> dict[str, Decimal]  # Line 406
    async def get_currency_risk_metrics(self) -> dict[str, dict[str, float]]  # Line 450
    def _initialize_currencies(self) -> None  # Line 512
    async def _update_exchange_rates(self) -> None  # Line 529
    async def _optimize_conversions(self, required_changes: dict[str, Decimal]) -> dict[str, Decimal]  # Line 609
    async def get_total_base_value(self) -> Decimal  # Line 658
    async def get_currency_exposure(self, currency: str) -> CurrencyExposure | None  # Line 665
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Decimal | None  # Line 669
    async def update_hedge_position(self, currency: str, hedge_amount: Decimal) -> None  # Line 686
    async def get_hedging_summary(self) -> dict[str, Any]  # Line 696
    async def _cleanup_rate_history(self) -> None  # Line 712
    async def cleanup_resources(self) -> None  # Line 724
    async def _calculate_currency_volatility_via_risk_service(self, currency: str, rate_history: list[tuple[datetime, Decimal]]) -> Decimal  # Line 750
    async def _calculate_currency_var_via_risk_service(self, exposure_amount: Decimal, volatility: Decimal) -> Decimal  # Line 768
    def _fallback_risk_calculation(self, ...) -> tuple[Decimal, Decimal]  # Line 780
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.types import CapitalAllocation`
- `from src.core.types import CapitalMetrics`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `CapitalDataTransformer`

**Purpose**: Handles consistent data transformation for capital_management module

```python
class CapitalDataTransformer:
    def transform_allocation_to_event_data(allocation: CapitalAllocation, metadata: dict[str, Any] = None) -> dict[str, Any]  # Line 19
    def transform_metrics_to_event_data(metrics: CapitalMetrics, metadata: dict[str, Any] = None) -> dict[str, Any]  # Line 52
    def transform_error_to_event_data(error, ...) -> dict[str, Any]  # Line 94
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 119
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'capital_management') -> dict[str, Any]  # Line 170
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] = None) -> dict[str, Any]  # Line 206
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str = None) -> dict[str, Any]  # Line 251
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 276
    def transform_allocation_data_for_state(cls, allocation_data: dict[str, Any], operation: str = 'unknown') -> dict[str, Any]  # Line 336
```

### File: di_registration.py

#### Functions:

```python
def register_capital_management_services(container: Any) -> None  # Line 11
def _register_capital_repositories(container: Any) -> None  # Line 131
```

### File: exchange_distributor.py

**Key Imports:**
- `from src.capital_management.constants import DEFAULT_EXCHANGE_SCORE`
- `from src.capital_management.constants import DEFAULT_MAX_SLIPPAGE_HISTORY`
- `from src.capital_management.constants import EXCHANGE_LIQUIDITY_SCORES`
- `from src.capital_management.constants import EXCHANGE_FEE_EFFICIENCIES`
- `from src.capital_management.constants import RELIABILITY_BONUS_PER_SERVICE`

#### Class: `ExchangeDistributor`

**Inherits**: AbstractExchangeDistributionService, TransactionalService
**Purpose**: Multi-exchange capital distribution manager

```python
class ExchangeDistributor(AbstractExchangeDistributionService, TransactionalService):
    def __init__(self, ...) -> None  # Line 81
    async def _do_start(self) -> None  # Line 121
    async def _do_stop(self) -> None  # Line 153
    async def _load_configuration(self) -> None  # Line 163
    async def distribute_capital(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]  # Line 169
    async def rebalance_exchanges(self) -> dict[str, ExchangeAllocation]  # Line 221
    async def get_exchange_allocation(self, exchange: str) -> ExchangeAllocation | None  # Line 260
    async def update_exchange_utilization(self, exchange: str, utilized_amount: Decimal) -> None  # Line 273
    async def calculate_optimal_distribution(self, total_capital: Decimal) -> dict[str, Decimal]  # Line 305
    async def _initialize_exchange_allocations(self) -> None  # Line 362
    async def _update_exchange_metrics(self) -> None  # Line 377
    async def _calculate_liquidity_score(self, exchange_name: str) -> float  # Line 402
    async def _calculate_fee_efficiency(self, exchange_name: str) -> float  # Line 471
    async def _calculate_reliability_score(self, exchange_name: str) -> float  # Line 513
    async def _update_slippage_data(self, exchange_name: str) -> None  # Line 573
    async def _dynamic_distribution(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]  # Line 619
    async def _weighted_distribution(self, total_amount: Decimal, weights: dict[str, float]) -> dict[str, ExchangeAllocation]  # Line 641
    async def _apply_minimum_balances(self, allocations: dict[str, ExchangeAllocation]) -> dict[str, ExchangeAllocation]  # Line 664
    async def _calculate_optimal_distribution(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]  # Line 685
    async def _apply_rebalancing_limits(self, new_allocations: dict[str, ExchangeAllocation]) -> dict[str, ExchangeAllocation]  # Line 711
    def supported_exchanges(self) -> list[str]  # Line 751
    def total_capital(self) -> Decimal  # Line 756
    async def get_exchange_metrics(self) -> dict[str, dict[str, float]]  # Line 762
    async def get_distribution_summary(self) -> dict[str, Any]  # Line 778
    async def cleanup_resources(self) -> None  # Line 803
```

### File: factory.py

**Key Imports:**
- `from src.core.base.factory import BaseFactory`
- `from src.core.exceptions import CreationError`

#### Class: `CapitalServiceFactory`

**Inherits**: BaseFactory['CapitalService']
**Purpose**: Factory for creating CapitalService instances

```python
class CapitalServiceFactory(BaseFactory['CapitalService']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 24
    def _create_capital_service(self, ...) -> 'CapitalService'  # Line 51
```

#### Class: `CapitalAllocatorFactory`

**Inherits**: BaseFactory['CapitalAllocator']
**Purpose**: Factory for creating CapitalAllocator instances

```python
class CapitalAllocatorFactory(BaseFactory['CapitalAllocator']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 130
    def _create_capital_allocator(self, ...) -> 'CapitalAllocator'  # Line 157
```

#### Class: `CurrencyManagerFactory`

**Inherits**: BaseFactory['CurrencyManager']
**Purpose**: Factory for creating CurrencyManager instances

```python
class CurrencyManagerFactory(BaseFactory['CurrencyManager']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 241
    def _create_currency_manager(self, ...) -> 'CurrencyManager'  # Line 268
```

#### Class: `ExchangeDistributorFactory`

**Inherits**: BaseFactory['ExchangeDistributor']
**Purpose**: Factory for creating ExchangeDistributor instances

```python
class ExchangeDistributorFactory(BaseFactory['ExchangeDistributor']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 329
    def _create_exchange_distributor(self, ...) -> 'ExchangeDistributor'  # Line 356
```

#### Class: `FundFlowManagerFactory`

**Inherits**: BaseFactory['FundFlowManager']
**Purpose**: Factory for creating FundFlowManager instances

```python
class FundFlowManagerFactory(BaseFactory['FundFlowManager']):
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 405
    def _create_fund_flow_manager(self, ...) -> 'FundFlowManager'  # Line 432
```

#### Class: `CapitalManagementFactory`

**Purpose**: Composite factory for all capital management services

```python
class CapitalManagementFactory:
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 495
    def create_capital_service(self, **kwargs) -> 'CapitalService'  # Line 519
    def create_capital_allocator(self, **kwargs) -> 'CapitalAllocator'  # Line 523
    def create_currency_manager(self, **kwargs) -> 'CurrencyManager'  # Line 527
    def create_exchange_distributor(self, **kwargs) -> 'ExchangeDistributor'  # Line 531
    def create_fund_flow_manager(self, **kwargs) -> 'FundFlowManager'  # Line 535
    def register_factories(self, container: Any) -> None  # Line 539
```

### File: fund_flow_manager.py

**Key Imports:**
- `from src.capital_management.constants import DEFAULT_MAX_FLOW_HISTORY`
- `from src.capital_management.constants import DEFAULT_PROFIT_THRESHOLD`
- `from src.capital_management.constants import MIN_DEPOSIT_AMOUNT`
- `from src.capital_management.constants import MIN_WITHDRAWAL_AMOUNT`
- `from src.capital_management.constants import MAX_WITHDRAWAL_PCT`

#### Class: `FundFlowManager`

**Inherits**: AbstractFundFlowManagementService, TransactionalService
**Purpose**: Deposit/withdrawal management system

```python
class FundFlowManager(AbstractFundFlowManagementService, TransactionalService):
    def __init__(self, ...) -> None  # Line 99
    async def _do_start(self) -> None  # Line 156
    async def _load_configuration(self) -> None  # Line 197
    async def _do_stop(self) -> None  # Line 203
    def _initialize_capital_protection(self) -> None  # Line 213
    async def _cache_fund_flows(self, flows: list[FundFlow]) -> None  # Line 241
    async def _get_cached_fund_flows(self) -> list[FundFlow] | None  # Line 258
    async def _store_fund_flow_time_series(self, flow: FundFlow) -> None  # Line 278
    async def process_deposit(self, amount: Decimal, currency: str = 'USDT', exchange: str = 'binance') -> FundFlow  # Line 301
    async def process_withdrawal(self, ...) -> FundFlow  # Line 369
    async def process_strategy_reallocation(self, ...) -> FundFlow  # Line 459
    async def process_auto_compound(self) -> FundFlow | None  # Line 536
    async def update_performance(self, strategy_id: str, performance_metrics: dict[str, float]) -> None  # Line 592
    async def get_flow_history(self, days: int = 30) -> list[FundFlow]  # Line 619
    async def get_flow_summary(self, days: int = 30) -> dict[str, Any]  # Line 641
    def _initialize_withdrawal_rules(self) -> None  # Line 702
    async def _validate_withdrawal_rules(self, amount: Decimal, currency: str) -> None  # Line 731
    async def _check_withdrawal_cooldown(self) -> None  # Line 791
    async def _calculate_minimum_capital_required(self) -> Decimal  # Line 823
    async def _check_performance_threshold(self, threshold: float) -> bool  # Line 849
    async def _get_daily_reallocation_amount(self) -> Decimal  # Line 885
    def _should_compound(self) -> bool  # Line 901
    async def _calculate_compound_amount(self) -> Decimal  # Line 922
    def _calculate_compound_schedule(self) -> dict[str, Any]  # Line 945
    async def update_total_capital(self, total_capital: Decimal) -> None  # Line 962
    async def get_total_capital(self) -> Decimal  # Line 976
    async def get_capital_protection_status(self) -> dict[str, Any]  # Line 980
    async def get_performance_summary(self) -> dict[str, Any]  # Line 1008
    def set_capital_allocator(self, capital_allocator: Any) -> None  # Line 1070
    def _validate_config(self) -> None  # Line 1080
    async def cleanup_resources(self) -> None  # Line 1110
```

### File: interfaces.py

**Key Imports:**
- `from src.core.types import CapitalAllocation`
- `from src.core.types import CapitalMetrics`
- `from src.core.types.capital import CapitalCurrencyExposure`
- `from src.core.types.capital import CapitalExchangeAllocation`
- `from src.core.types.capital import CapitalFundFlow`

#### Class: `AbstractCapitalService`

**Inherits**: ABC
**Purpose**: Abstract base class for capital service implementations

```python
class AbstractCapitalService(ABC):
    async def allocate_capital(self, ...) -> CapitalAllocation  # Line 213
    async def release_capital(self, ...) -> bool  # Line 226
    async def get_capital_metrics(self) -> CapitalMetrics  # Line 238
    async def get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]  # Line 243
```

#### Class: `AbstractCurrencyManagementService`

**Inherits**: ABC
**Purpose**: Abstract base class for currency management service implementations

```python
class AbstractCurrencyManagementService(ABC):
    async def update_currency_exposures(self, balances: dict[str, dict[str, Decimal]]) -> dict[str, CapitalCurrencyExposure]  # Line 252
    async def calculate_hedging_requirements(self) -> dict[str, Decimal]  # Line 259
```

#### Class: `AbstractExchangeDistributionService`

**Inherits**: ABC
**Purpose**: Abstract base class for exchange distribution service implementations

```python
class AbstractExchangeDistributionService(ABC):
    async def distribute_capital(self, total_amount: Decimal) -> dict[str, CapitalExchangeAllocation]  # Line 268
    async def rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]  # Line 275
```

#### Class: `AbstractFundFlowManagementService`

**Inherits**: ABC
**Purpose**: Abstract base class for fund flow management service implementations

```python
class AbstractFundFlowManagementService(ABC):
    async def process_deposit(self, amount: Decimal, currency: str = 'USDT', exchange: str = 'binance') -> CapitalFundFlow  # Line 284
    async def process_withdrawal(self, ...) -> CapitalFundFlow  # Line 291
```

### File: service.py

**Key Imports:**
- `from src.capital_management.constants import EMERGENCY_RESERVE_PCT`
- `from src.capital_management.constants import MAX_ALLOCATION_PCT`
- `from src.capital_management.constants import MAX_DAILY_REALLOCATION_PCT`
- `from src.capital_management.constants import DEFAULT_CACHE_TTL_SECONDS`
- `from src.capital_management.constants import CIRCUIT_BREAKER_FAILURE_THRESHOLD`

#### Class: `SimpleEventPattern`

**Purpose**: Simple event pattern placeholder for testing compatibility

```python
class SimpleEventPattern:
    async def emit_consistent(self, event_type, data)  # Line 120
```

#### Class: `SimpleValidationPattern`

**Purpose**: Simple validation pattern placeholder for testing compatibility

```python
class SimpleValidationPattern:
    def validate_consistent(self, data)  # Line 128
```

#### Class: `SimpleProcessingPattern`

**Purpose**: Simple processing pattern placeholder for testing compatibility

```python
class SimpleProcessingPattern:
    async def process_item_consistent(self, item, *args, **kwargs)  # Line 136
```

#### Class: `CapitalService`

**Inherits**: AbstractCapitalService, TransactionalService
**Purpose**: Enterprise-grade capital management service

```python
class CapitalService(AbstractCapitalService, TransactionalService):
    def __init__(self, ...) -> None  # Line 158
    async def _do_start(self) -> None  # Line 234
    async def _do_stop(self) -> None  # Line 285
    async def _initialize_capital_state(self) -> None  # Line 295
    async def allocate_capital(self, ...) -> CapitalAllocation  # Line 330
    async def _allocate_capital_impl(self, ...) -> CapitalAllocation  # Line 368
    async def release_capital(self, ...) -> bool  # Line 674
    async def _release_capital_impl(self, ...) -> bool  # Line 709
    async def update_utilization(self, ...) -> bool  # Line 925
    async def _update_utilization_impl(self, ...) -> bool  # Line 953
    async def get_capital_metrics(self) -> CapitalMetrics  # Line 1044
    async def get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]  # Line 1063
    async def get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]  # Line 1079
    async def _get_capital_metrics_impl(self) -> CapitalMetrics  # Line 1095
    async def _get_allocations_by_strategy_impl(self, strategy_id: str) -> list[CapitalAllocation]  # Line 1142
    async def _get_all_allocations_impl(self, limit: int | None = None) -> list[CapitalAllocation]  # Line 1174
    async def _validate_allocation_request_consistent(self, strategy_id: str, exchange: str, amount: Decimal) -> None  # Line 1214
    def _validate_allocation_request_fallback(self, strategy_id: str, exchange: str, amount: Decimal) -> None  # Line 1249
    async def _validate_allocation_limits_consistent(self, strategy_id: str, exchange: str, amount: Decimal) -> None  # Line 1259
    def _validate_allocation_limits_fallback(self, strategy_id: str, exchange: str, amount: Decimal) -> None  # Line 1267
    async def _get_available_capital(self) -> Decimal  # Line 1289
    async def _get_existing_allocation(self, strategy_id: str, exchange: str) -> Any | None  # Line 1316
    async def _create_allocation(self, allocation_data: dict[str, Any]) -> Any  # Line 1329
    async def _update_allocation(self, allocation_data: dict[str, Any]) -> Any  # Line 1342
    async def _validate_allocation_business_rules(self, allocation_data: dict[str, Any], is_update: bool = False) -> None  # Line 1355
    async def _delete_allocation(self, allocation_id: str) -> bool  # Line 1390
    async def _get_all_allocations(self, limit: int | None = None) -> list[Any]  # Line 1402
    async def _create_audit_log_record(self, audit_log_data: dict[str, Any]) -> None  # Line 1413
    async def _restore_allocations_in_transaction(self, allocations_data: list[dict[str, Any]]) -> None  # Line 1425
    async def _calculate_allocation_efficiency(self, allocations: list[Any]) -> float  # Line 1482
    async def _create_audit_log(self, ...) -> None  # Line 1508
    def _update_allocation_metrics(self, start_time: datetime) -> None  # Line 1560
    async def _service_health_check(self) -> HealthStatus  # Line 1578
    def get_performance_metrics(self) -> dict[str, Any]  # Line 1616
    def reset_metrics(self) -> None  # Line 1626
    def _load_configuration(self) -> None  # Line 1643
    def _safe_decimal_conversion(self, value: Any) -> Decimal  # Line 1654
    def _safe_get_value(self, obj: Any, key: str, default: Any = None) -> Any  # Line 1658
    def _propagate_capital_error_consistently(self, error: Exception, operation: str, operation_id: str) -> None  # Line 1667
    async def _save_capital_state_snapshot(self, reason: str = 'allocation_change') -> None  # Line 1721
    async def _build_consistent_state_data(self, allocations: list[Any], reason: str) -> dict[str, Any]  # Line 1779
    async def restore_capital_state(self) -> bool  # Line 1813
    async def cleanup_resources(self) -> None  # Line 1875
```

---
**Generated**: Complete reference for capital_management module
**Total Classes**: 31
**Total Functions**: 2