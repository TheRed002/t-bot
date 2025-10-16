# BACKTESTING Module Reference

## INTEGRATION
**Dependencies**: core, data, database, execution, risk_management, strategies, utils
**Used By**: None
**Provides**: BacktestController, BacktestEngine, BacktestService, DataReplayManager
**Patterns**: Async Operations, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Decimal precision arithmetic
**Performance**:
- Parallel execution
- Parallel execution
**Architecture**:
- MonteCarloAnalyzer inherits from base architecture
- WalkForwardAnalyzer inherits from base architecture
- PerformanceAttributor inherits from base architecture

## MODULE OVERVIEW
**Files**: 14 Python files
**Classes**: 31
**Functions**: 6

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `MonteCarloAnalyzer` ✅

**Inherits**: BaseComponent
**Purpose**: Monte Carlo simulation for backtesting robustness analysis
**Status**: Complete

**Implemented Methods:**
- `async analyze_trades(self, trades: list[dict[str, Any]], initial_capital: Decimal) -> dict[str, Any]` - Line 74
- `async run_analysis(self, ...) -> dict[str, Any]` - Line 142
- `async analyze_returns(self, daily_returns: list[float], num_days: int = 252) -> dict[str, Any]` - Line 197

### Implementation: `WalkForwardAnalyzer` ✅

**Inherits**: BaseComponent
**Purpose**: Walk-Forward Analysis for robust parameter optimization
**Status**: Complete

**Implemented Methods:**
- `async analyze(self, ...) -> dict[str, Any]` - Line 377
- `async run_analysis(self, ...) -> dict[str, Any]` - Line 443

### Implementation: `PerformanceAttributor` ✅

**Inherits**: BaseComponent
**Purpose**: Analyzes and attributes performance to various factors
**Status**: Complete

**Implemented Methods:**
- `attribute_returns(self, ...) -> dict[str, Any]` - Line 44
- `calculate_rolling_attribution(self, trades: list[dict[str, Any]], window_days: int = 30) -> pd.DataFrame` - Line 395
- `generate_attribution_report(self, attribution_results: dict[str, Any]) -> str` - Line 441
- `async analyze(self, simulation_result: dict[str, Any], market_data: dict[str, pd.DataFrame]) -> dict[str, Any]` - Line 502

### Implementation: `BacktestController` ✅

**Inherits**: BaseComponent
**Purpose**: Controller for backtesting operations
**Status**: Complete

**Implemented Methods:**
- `async run_backtest(self, request_data: dict[str, Any]) -> dict[str, Any]` - Line 55
- `async get_active_backtests(self) -> dict[str, Any]` - Line 84
- `async cancel_backtest(self, backtest_id: str) -> dict[str, Any]` - Line 94
- `async clear_cache(self, pattern: str = '*') -> dict[str, Any]` - Line 116
- `async get_cache_stats(self) -> dict[str, Any]` - Line 125
- `async health_check(self) -> HealthCheckResult` - Line 135
- `async get_backtest_result(self, result_id: str) -> dict[str, Any]` - Line 166
- `async list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> dict[str, Any]` - Line 191
- `async delete_backtest_result(self, result_id: str) -> dict[str, Any]` - Line 213
- `async cleanup(self) -> None` - Line 235

### Implementation: `ReplayMode` ✅

**Inherits**: Enum
**Purpose**: Data replay modes
**Status**: Complete

### Implementation: `DataReplayManager` ✅

**Inherits**: BaseComponent
**Purpose**: Manages historical data replay for backtesting
**Status**: Complete

**Implemented Methods:**
- `async load_data(self, ...) -> None` - Line 123
- `async start_replay(self, ...) -> None` - Line 268
- `get_current_data(self, symbol: str) -> pd.Series | None` - Line 502
- `get_historical_data(self, symbol: str, lookback: int) -> pd.DataFrame | None` - Line 514
- `subscribe(self, callback: Callable) -> None` - Line 524
- `unsubscribe(self, callback: Callable) -> None` - Line 528
- `reset(self) -> None` - Line 533
- `get_statistics(self) -> dict[str, Any]` - Line 541
- `async cleanup(self) -> None` - Line 556

### Implementation: `BacktestDataTransformer` ✅

**Purpose**: Handles consistent data transformation for backtesting module
**Status**: Complete

**Implemented Methods:**
- `transform_signal_to_event_data(signal: Signal, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 23
- `transform_backtest_result_to_event_data(result: 'BacktestResult', metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 56
- `transform_market_data_to_event_data(market_data: MarketData, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 96
- `transform_trade_to_event_data(trade: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 134
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 171
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'backtesting') -> dict[str, Any]` - Line 201
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]` - Line 228
- `transform_for_batch_processing(cls, ...) -> dict[str, Any]` - Line 289
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'batch') -> dict[str, Any]` - Line 361
- `apply_cross_module_validation(cls, ...) -> dict[str, Any]` - Line 440

### Implementation: `BacktestConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for backtesting engine
**Status**: Complete

**Implemented Methods:**
- `validate_dates(cls, v: datetime, info) -> datetime` - Line 86
- `validate_rates(cls, v: Decimal) -> Decimal` - Line 92

### Implementation: `BacktestResult` ✅

**Inherits**: BaseModel
**Purpose**: Results from a backtest run
**Status**: Complete

### Implementation: `BacktestEngine` ✅

**Purpose**: Main backtesting engine for strategy evaluation
**Status**: Complete

**Implemented Methods:**
- `async run(self) -> BacktestResult` - Line 205

### Implementation: `BacktestFactory` ✅

**Inherits**: BaseComponent
**Purpose**: Factory for creating backtesting components with dependency injection
**Status**: Complete

**Implemented Methods:**
- `get_injector(self) -> DependencyInjector` - Line 56
- `create_controller(self) -> 'BacktestController'` - Line 64
- `create_service(self, config: Any = None) -> 'BacktestService'` - Line 89
- `create_repository(self) -> 'BacktestRepository'` - Line 147
- `create_engine(self, config: Any, strategy: Any, **kwargs) -> 'BacktestEngine'` - Line 172
- `create_simulator(self, config: Any) -> 'TradeSimulator'` - Line 220
- `create_metrics_calculator(self, risk_free_rate: float | None = None) -> 'MetricsCalculator'` - Line 253
- `create_analyzer(self, analyzer_type: str, config: dict[str, Any] | None = None) -> Any` - Line 277
- `wire_dependencies(self) -> None` - Line 338
- `get_interface(self) -> 'BacktestServiceInterface'` - Line 350

### Implementation: `DataServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Data service interface for backtesting dependencies
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 19
- `async store_market_data(self, ...) -> bool` - Line 24
- `async get_market_data(self, request: 'DataRequest') -> 'list[MarketDataRecord]'` - Line 34
- `async get_recent_data(self, symbol: str, limit: int = 100, exchange: str = 'binance') -> 'list[MarketData]'` - Line 39
- `async cleanup(self) -> None` - Line 46

### Implementation: `BacktestServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for BacktestService
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 55
- `async run_backtest(self, request: 'BacktestRequest') -> 'BacktestResult'` - Line 59
- `async run_backtest_from_dict(self, request_data: dict[str, Any]) -> 'BacktestResult'` - Line 63
- `async serialize_result(self, result: 'BacktestResult') -> dict[str, Any]` - Line 67
- `async get_active_backtests(self) -> dict[str, dict[str, Any]]` - Line 71
- `async cancel_backtest(self, backtest_id: str) -> bool` - Line 75
- `async clear_cache(self, pattern: str = '*') -> int` - Line 79
- `async get_cache_stats(self) -> dict[str, Any]` - Line 83
- `async health_check(self) -> HealthCheckResult` - Line 87
- `async get_backtest_result(self, result_id: str) -> dict[str, Any] | None` - Line 91
- `async list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> list[dict[str, Any]]` - Line 95
- `async delete_backtest_result(self, result_id: str) -> bool` - Line 101
- `async cleanup(self) -> None` - Line 105

### Implementation: `MetricsCalculatorInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for MetricsCalculator
**Status**: Complete

**Implemented Methods:**
- `calculate_all(self, ...) -> dict[str, Any]` - Line 113

### Implementation: `BacktestAnalyzerInterface` 🔧

**Inherits**: ABC
**Purpose**: Base interface for backtest analyzers
**Status**: Abstract Base Class

**Implemented Methods:**
- `async run_analysis(self, **kwargs) -> dict[str, Any]` - Line 128

### Implementation: `BacktestEngineFactoryInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for BacktestEngine factory
**Status**: Complete

**Implemented Methods:**

### Implementation: `ComponentFactoryInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for component factories
**Status**: Complete

**Implemented Methods:**

### Implementation: `BacktestControllerInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for BacktestController
**Status**: Complete

**Implemented Methods:**
- `async run_backtest(self, request_data: dict[str, Any]) -> dict[str, Any]` - Line 153
- `async get_active_backtests(self) -> dict[str, Any]` - Line 157
- `async cancel_backtest(self, backtest_id: str) -> dict[str, Any]` - Line 161
- `async health_check(self) -> dict[str, Any]` - Line 165
- `async get_backtest_result(self, result_id: str) -> dict[str, Any]` - Line 169
- `async list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> dict[str, Any]` - Line 173
- `async delete_backtest_result(self, result_id: str) -> dict[str, Any]` - Line 179

### Implementation: `BacktestRepositoryInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for BacktestRepository
**Status**: Complete

**Implemented Methods:**
- `async save_backtest_result(self, result_data: dict[str, Any], request_data: dict[str, Any]) -> str` - Line 188
- `async get_backtest_result(self, result_id: str) -> dict[str, Any] | None` - Line 194
- `async list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> list[dict[str, Any]]` - Line 198
- `async delete_backtest_result(self, result_id: str) -> bool` - Line 204

### Implementation: `BacktestFactoryInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for BacktestFactory
**Status**: Complete

**Implemented Methods:**
- `create_controller(self) -> Any` - Line 213
- `create_service(self, config: Any) -> Any` - Line 217
- `create_repository(self) -> Any` - Line 221
- `create_engine(self, config: Any, strategy: Any, **kwargs) -> Any` - Line 225

### Implementation: `TradeSimulatorInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for TradeSimulator
**Status**: Complete

**Implemented Methods:**
- `async execute_order(self, order_request: Any, market_data: Any, **kwargs) -> dict[str, Any]` - Line 234
- `async get_simulation_results(self) -> dict[str, Any]` - Line 238

### Implementation: `CacheServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for caching service to decouple from Redis
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 247
- `async get(self, key: str) -> Any` - Line 251
- `async set(self, key: str, value: Any, ttl: int) -> None` - Line 255
- `async delete(self, key: str) -> None` - Line 259
- `async clear_pattern(self, pattern: str) -> int` - Line 263
- `async get_stats(self) -> dict[str, Any]` - Line 267
- `async cleanup(self) -> None` - Line 271

### Implementation: `BacktestMetrics` ✅

**Inherits**: BaseComponent
**Purpose**: Container for all backtest metrics
**Status**: Complete

**Implemented Methods:**
- `add(self, name: str, value: Any) -> None` - Line 37
- `get(self, name: str, default: Any = None) -> Any` - Line 41
- `to_dict(self) -> dict[str, Any]` - Line 45

### Implementation: `MetricsCalculator` ✅

**Purpose**: Calculator for comprehensive backtest metrics
**Status**: Complete

**Implemented Methods:**
- `calculate_all(self, ...) -> dict[str, Any]` - Line 79
- `calculate_rolling_metrics(self, equity_curve: list[dict[str, Any]], window: int = 30) -> pd.DataFrame` - Line 367

### Implementation: `BacktestRepository` ✅

**Inherits**: BaseComponent
**Purpose**: Repository for backtesting data operations
**Status**: Complete

**Implemented Methods:**
- `async save_backtest_result(self, result_data: dict[str, Any], request_data: dict[str, Any]) -> str` - Line 77
- `async get_backtest_result(self, result_id: str) -> dict[str, Any] | None` - Line 191
- `async list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> list[dict[str, Any]]` - Line 258
- `async delete_backtest_result(self, result_id: str) -> bool` - Line 306
- `async save_trade_history(self, backtest_id: str, trades: list[dict[str, Any]]) -> None` - Line 342
- `async get_trade_history(self, backtest_id: str) -> list[dict[str, Any]]` - Line 378
- `async cleanup_old_results(self, days_old: int = 30) -> int` - Line 417

### Implementation: `BacktestRequest` ✅

**Inherits**: BaseModel
**Purpose**: Comprehensive backtest request model with validation
**Status**: Complete

**Implemented Methods:**
- `validate_dates(cls, v: datetime, info) -> datetime` - Line 96
- `validate_symbols(cls, v: list[str]) -> list[str]` - Line 102

### Implementation: `BacktestCacheEntry` ✅

**Inherits**: BaseModel
**Purpose**: Cache entry for backtest results
**Status**: Complete

**Implemented Methods:**
- `is_expired(self) -> bool` - Line 116

### Implementation: `BacktestService` ✅

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: Comprehensive BacktestService integrating all service layers
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 249
- `async run_backtest(self, request: BacktestRequest) -> 'BacktestResult'` - Line 318
- `async run_backtest_from_dict(self, request_data: dict[str, Any]) -> 'BacktestResult'` - Line 322
- `async serialize_result(self, result: 'BacktestResult') -> dict[str, Any]` - Line 332
- `async get_active_backtests(self) -> dict[str, dict[str, Any]]` - Line 871
- `async cancel_backtest(self, backtest_id: str) -> bool` - Line 881
- `async clear_cache(self, pattern: str = '*') -> int` - Line 895
- `async get_cache_stats(self) -> dict[str, Any]` - Line 923
- `async get_backtest_result(self, result_id: str) -> dict[str, Any] | None` - Line 941
- `async list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> list[dict[str, Any]]` - Line 954
- `async delete_backtest_result(self, result_id: str) -> bool` - Line 969
- `async health_check(self) -> HealthCheckResult` - Line 982
- `async cleanup(self) -> None` - Line 1034

### Implementation: `SimulationConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for trade simulation
**Status**: Complete

### Implementation: `SimulatedOrder` ✅

**Inherits**: BaseModel
**Purpose**: Extended order representation for simulation tracking
**Status**: Complete

### Implementation: `TradeSimulator` ✅

**Purpose**: Realistic trade execution simulator
**Status**: Complete

**Implemented Methods:**
- `async execute_order(self, ...) -> dict[str, Any]` - Line 121
- `async check_pending_orders(self, market_data: dict[str, pd.Series]) -> list[dict[str, Any]]` - Line 594
- `calculate_execution_costs(self, trades: list[dict[str, Any]]) -> dict[str, Decimal]` - Line 625
- `get_execution_statistics(self) -> dict[str, Any]` - Line 661
- `cleanup(self) -> None` - Line 687
- `async get_simulation_results(self) -> dict[str, Any]` - Line 700

## COMPLETE API REFERENCE

### File: analysis.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import TradingBotError`
- `from src.utils.config_conversion import convert_config_to_dict`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `MonteCarloAnalyzer`

**Inherits**: BaseComponent
**Purpose**: Monte Carlo simulation for backtesting robustness analysis

```python
class MonteCarloAnalyzer(BaseComponent):
    def __init__(self, ...)  # Line 37
    async def analyze_trades(self, trades: list[dict[str, Any]], initial_capital: Decimal) -> dict[str, Any]  # Line 74
    async def run_analysis(self, ...) -> dict[str, Any]  # Line 142
    async def analyze_returns(self, daily_returns: list[float], num_days: int = 252) -> dict[str, Any]  # Line 197
    def _calculate_simulation_metrics(self, returns: list[Decimal], initial_capital: Decimal) -> dict[str, Any]  # Line 264
    def _analyze_simulation_results(self, results: list[dict[str, Any]]) -> dict[str, Any]  # Line 277
```

#### Class: `WalkForwardAnalyzer`

**Inherits**: BaseComponent
**Purpose**: Walk-Forward Analysis for robust parameter optimization

```python
class WalkForwardAnalyzer(BaseComponent):
    def __init__(self, ...)  # Line 342
    async def analyze(self, ...) -> dict[str, Any]  # Line 377
    async def run_analysis(self, ...) -> dict[str, Any]  # Line 443
    def _generate_windows(self, data: pd.DataFrame) -> list[tuple[datetime, datetime, datetime, datetime]]  # Line 481
    async def _optimize_parameters(self, ...) -> dict[str, Any]  # Line 506
    async def _test_parameters(self, strategy_class: type, parameters: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]  # Line 560
    def _analyze_results(self, window_results: list[dict[str, Any]], optimization_metric: str) -> dict[str, Any]  # Line 593
    def _analyze_parameter_stability(self, window_results: list[dict[str, Any]]) -> dict[str, Any]  # Line 635
```

### File: attribution.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.utils.attribution_structures import create_empty_attribution_structure`
- `from src.utils.config_conversion import convert_config_to_dict`
- `from src.utils.decimal_utils import to_decimal`
- `from src.utils.decorators import time_execution`

#### Class: `PerformanceAttributor`

**Inherits**: BaseComponent
**Purpose**: Analyzes and attributes performance to various factors

```python
class PerformanceAttributor(BaseComponent):
    def __init__(self, config: Any = None) -> None  # Line 35
    def attribute_returns(self, ...) -> dict[str, Any]  # Line 44
    def _empty_attribution(self) -> dict[str, Any]  # Line 104
    def _group_trades_by_symbol(self, trades: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]  # Line 108
    def _attribute_by_symbol(self, symbol_trades: dict[str, list[dict[str, Any]]]) -> dict[str, Any]  # Line 122
    def _timing_vs_selection(self, trades: list[dict[str, Any]], market_returns: dict[str, pd.Series]) -> dict[str, Any]  # Line 184
    def _factor_decomposition(self, trades: list[dict[str, Any]], market_returns: dict[str, pd.Series]) -> dict[str, Any]  # Line 260
    def _cost_analysis(self, trades: list[dict[str, Any]]) -> dict[str, Any]  # Line 322
    def _calculate_summary(self, ...) -> dict[str, Any]  # Line 366
    def calculate_rolling_attribution(self, trades: list[dict[str, Any]], window_days: int = 30) -> pd.DataFrame  # Line 395
    def generate_attribution_report(self, attribution_results: dict[str, Any]) -> str  # Line 441
    async def analyze(self, simulation_result: dict[str, Any], market_data: dict[str, pd.DataFrame]) -> dict[str, Any]  # Line 502
```

### File: controller.py

**Key Imports:**
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.component import BaseComponent`
- `from src.core.event_constants import BacktestEvents`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`

#### Class: `BacktestController`

**Inherits**: BaseComponent
**Purpose**: Controller for backtesting operations

```python
class BacktestController(BaseComponent):
    def __init__(self, backtest_service: 'BacktestServiceInterface')  # Line 35
    def _get_local_logger(self)  # Line 46
    async def run_backtest(self, request_data: dict[str, Any]) -> dict[str, Any]  # Line 55
    async def get_active_backtests(self) -> dict[str, Any]  # Line 84
    async def cancel_backtest(self, backtest_id: str) -> dict[str, Any]  # Line 94
    async def clear_cache(self, pattern: str = '*') -> dict[str, Any]  # Line 116
    async def get_cache_stats(self) -> dict[str, Any]  # Line 125
    async def health_check(self) -> HealthCheckResult  # Line 135
    async def get_backtest_result(self, result_id: str) -> dict[str, Any]  # Line 166
    async def list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> dict[str, Any]  # Line 191
    async def delete_backtest_result(self, result_id: str) -> dict[str, Any]  # Line 213
    async def cleanup(self) -> None  # Line 235
```

### File: data_replay.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import DataError`
- `from src.utils.backtesting_decorators import data_loading_operation`
- `from src.utils.config_conversion import convert_config_to_dict`

#### Class: `ReplayMode`

**Inherits**: Enum
**Purpose**: Data replay modes

```python
class ReplayMode(Enum):
```

#### Class: `DataReplayManager`

**Inherits**: BaseComponent
**Purpose**: Manages historical data replay for backtesting

```python
class DataReplayManager(BaseComponent):
    def __init__(self, config: Config | None = None, cache_size: int = DEFAULT_CACHE_SIZE) -> None  # Line 56
    async def __aenter__(self) -> 'DataReplayManager'  # Line 103
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None  # Line 109
    async def load_data(self, ...) -> None  # Line 123
    async def _load_from_csv(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame  # Line 183
    async def _generate_synthetic_data(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> pd.DataFrame  # Line 205
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame  # Line 221
    async def start_replay(self, ...) -> None  # Line 268
    async def _replay_sequential(self) -> None  # Line 299
    async def _replay_random_walk(self) -> None  # Line 327
    async def _replay_bootstrap(self) -> None  # Line 365
    async def _replay_shuffle(self) -> None  # Line 394
    async def _notify_subscribers(self, timestamp: datetime, data: dict[str, pd.Series]) -> None  # Line 435
    async def _execute_callback_with_timeout(self, callback: Callable, timestamp: datetime, data: dict[str, pd.Series]) -> None  # Line 488
    def get_current_data(self, symbol: str) -> pd.Series | None  # Line 502
    def get_historical_data(self, symbol: str, lookback: int) -> pd.DataFrame | None  # Line 514
    def subscribe(self, callback: Callable) -> None  # Line 524
    def unsubscribe(self, callback: Callable) -> None  # Line 528
    def reset(self) -> None  # Line 533
    def get_statistics(self) -> dict[str, Any]  # Line 541
    async def cleanup(self) -> None  # Line 556
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import Signal`
- `from src.utils.decimal_utils import to_decimal`
- `from src.utils.messaging_patterns import MessagePattern`

#### Class: `BacktestDataTransformer`

**Purpose**: Handles consistent data transformation for backtesting module

```python
class BacktestDataTransformer:
    def transform_signal_to_event_data(signal: Signal, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 23
    def transform_backtest_result_to_event_data(result: 'BacktestResult', metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 56
    def transform_market_data_to_event_data(market_data: MarketData, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 96
    def transform_trade_to_event_data(trade: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 134
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 171
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'backtesting') -> dict[str, Any]  # Line 201
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]  # Line 228
    def transform_for_batch_processing(cls, ...) -> dict[str, Any]  # Line 289
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'batch') -> dict[str, Any]  # Line 361
    def apply_cross_module_validation(cls, ...) -> dict[str, Any]  # Line 440
    def _apply_core_boundary_validation(cls, data: dict[str, Any], source_module: str, target_module: str) -> dict[str, Any]  # Line 543
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def register_backtesting_services(injector: DependencyInjector) -> None  # Line 22
def configure_backtesting_dependencies(injector: DependencyInjector | None = None) -> DependencyInjector  # Line 155
def get_backtest_service(injector: DependencyInjector) -> 'BacktestService'  # Line 176
```

### File: engine.py

**Key Imports:**
- `from src.backtesting.utils import convert_market_records_to_dataframe`
- `from src.core.exceptions import BacktestServiceError`
- `from src.core.exceptions import BacktestValidationError`
- `from src.core.exceptions import DataValidationError`
- `from src.core.exceptions import ServiceError`

#### Class: `BacktestConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for backtesting engine

```python
class BacktestConfig(BaseModel):
    def validate_dates(cls, v: datetime, info) -> datetime  # Line 86
    def validate_rates(cls, v: Decimal) -> Decimal  # Line 92
```

#### Class: `BacktestResult`

**Inherits**: BaseModel
**Purpose**: Results from a backtest run

```python
class BacktestResult(BaseModel):
```

#### Class: `BacktestEngine`

**Purpose**: Main backtesting engine for strategy evaluation

```python
class BacktestEngine:
    def __init__(self, ...)  # Line 156
    async def run(self) -> BacktestResult  # Line 205
    async def _load_historical_data(self) -> None  # Line 265
    async def _load_from_data_service(self, symbol: str) -> pd.DataFrame  # Line 281
    async def _load_default_data(self, symbol: str) -> pd.DataFrame  # Line 309
    async def _initialize_strategy(self) -> None  # Line 324
    async def _run_simulation(self) -> None  # Line 374
    def _get_current_market_data(self, timestamp: datetime) -> dict[str, pd.Series]  # Line 408
    async def _generate_signals(self, market_data: dict[str, pd.Series]) -> dict[str, Signal]  # Line 416
    async def _execute_trades(self, signals: dict[str, Signal], market_data: dict[str, pd.Series]) -> None  # Line 459
    async def _execute_with_engine(self, symbol: str, signal: Signal, market_data_row: pd.Series) -> None  # Line 490
    async def _open_position(self, symbol: str, price: Decimal, signal: SignalDirection) -> None  # Line 667
    async def _close_position(self, symbol: str, price: Decimal) -> None  # Line 739
    def _update_positions(self, market_data: dict[str, pd.Series]) -> None  # Line 786
    def _record_equity(self) -> None  # Line 805
    async def _check_risk_limits(self) -> None  # Line 815
    async def _close_all_positions(self) -> None  # Line 832
    async def _calculate_results(self) -> BacktestResult  # Line 840
```

### File: factory.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.exceptions import FactoryError`
- `from src.core.logging import get_logger`
- `from src.utils.initialization_helpers import log_factory_initialization`

#### Class: `BacktestFactory`

**Inherits**: BaseComponent
**Purpose**: Factory for creating backtesting components with dependency injection

```python
class BacktestFactory(BaseComponent):
    def __init__(self, ...)  # Line 40
    def get_injector(self) -> DependencyInjector  # Line 56
    def create_controller(self) -> 'BacktestController'  # Line 64
    def create_service(self, config: Any = None) -> 'BacktestService'  # Line 89
    def create_repository(self) -> 'BacktestRepository'  # Line 147
    def create_engine(self, config: Any, strategy: Any, **kwargs) -> 'BacktestEngine'  # Line 172
    def create_simulator(self, config: Any) -> 'TradeSimulator'  # Line 220
    def create_metrics_calculator(self, risk_free_rate: float | None = None) -> 'MetricsCalculator'  # Line 253
    def create_analyzer(self, analyzer_type: str, config: dict[str, Any] | None = None) -> Any  # Line 277
    def wire_dependencies(self) -> None  # Line 338
    def get_interface(self) -> 'BacktestServiceInterface'  # Line 350
```

### File: interfaces.py

**Key Imports:**
- `from src.core.base.interfaces import HealthCheckResult`

#### Class: `DataServiceInterface`

**Inherits**: ABC
**Purpose**: Data service interface for backtesting dependencies

```python
class DataServiceInterface(ABC):
    async def initialize(self) -> None  # Line 19
    async def store_market_data(self, ...) -> bool  # Line 24
    async def get_market_data(self, request: 'DataRequest') -> 'list[MarketDataRecord]'  # Line 34
    async def get_recent_data(self, symbol: str, limit: int = 100, exchange: str = 'binance') -> 'list[MarketData]'  # Line 39
    async def cleanup(self) -> None  # Line 46
```

#### Class: `BacktestServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for BacktestService

```python
class BacktestServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 55
    async def run_backtest(self, request: 'BacktestRequest') -> 'BacktestResult'  # Line 59
    async def run_backtest_from_dict(self, request_data: dict[str, Any]) -> 'BacktestResult'  # Line 63
    async def serialize_result(self, result: 'BacktestResult') -> dict[str, Any]  # Line 67
    async def get_active_backtests(self) -> dict[str, dict[str, Any]]  # Line 71
    async def cancel_backtest(self, backtest_id: str) -> bool  # Line 75
    async def clear_cache(self, pattern: str = '*') -> int  # Line 79
    async def get_cache_stats(self) -> dict[str, Any]  # Line 83
    async def health_check(self) -> HealthCheckResult  # Line 87
    async def get_backtest_result(self, result_id: str) -> dict[str, Any] | None  # Line 91
    async def list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> list[dict[str, Any]]  # Line 95
    async def delete_backtest_result(self, result_id: str) -> bool  # Line 101
    async def cleanup(self) -> None  # Line 105
```

#### Class: `MetricsCalculatorInterface`

**Inherits**: Protocol
**Purpose**: Interface for MetricsCalculator

```python
class MetricsCalculatorInterface(Protocol):
    def calculate_all(self, ...) -> dict[str, Any]  # Line 113
```

#### Class: `BacktestAnalyzerInterface`

**Inherits**: ABC
**Purpose**: Base interface for backtest analyzers

```python
class BacktestAnalyzerInterface(ABC):
    async def run_analysis(self, **kwargs) -> dict[str, Any]  # Line 128
```

#### Class: `BacktestEngineFactoryInterface`

**Inherits**: Protocol
**Purpose**: Interface for BacktestEngine factory

```python
class BacktestEngineFactoryInterface(Protocol):
    def __call__(self, config: Any, strategy: Any, **kwargs) -> Any  # Line 136
```

#### Class: `ComponentFactoryInterface`

**Inherits**: Protocol
**Purpose**: Interface for component factories

```python
class ComponentFactoryInterface(Protocol):
    def __call__(self) -> Any  # Line 144
```

#### Class: `BacktestControllerInterface`

**Inherits**: Protocol
**Purpose**: Interface for BacktestController

```python
class BacktestControllerInterface(Protocol):
    async def run_backtest(self, request_data: dict[str, Any]) -> dict[str, Any]  # Line 153
    async def get_active_backtests(self) -> dict[str, Any]  # Line 157
    async def cancel_backtest(self, backtest_id: str) -> dict[str, Any]  # Line 161
    async def health_check(self) -> dict[str, Any]  # Line 165
    async def get_backtest_result(self, result_id: str) -> dict[str, Any]  # Line 169
    async def list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> dict[str, Any]  # Line 173
    async def delete_backtest_result(self, result_id: str) -> dict[str, Any]  # Line 179
```

#### Class: `BacktestRepositoryInterface`

**Inherits**: Protocol
**Purpose**: Interface for BacktestRepository

```python
class BacktestRepositoryInterface(Protocol):
    async def save_backtest_result(self, result_data: dict[str, Any], request_data: dict[str, Any]) -> str  # Line 188
    async def get_backtest_result(self, result_id: str) -> dict[str, Any] | None  # Line 194
    async def list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> list[dict[str, Any]]  # Line 198
    async def delete_backtest_result(self, result_id: str) -> bool  # Line 204
```

#### Class: `BacktestFactoryInterface`

**Inherits**: Protocol
**Purpose**: Interface for BacktestFactory

```python
class BacktestFactoryInterface(Protocol):
    def create_controller(self) -> Any  # Line 213
    def create_service(self, config: Any) -> Any  # Line 217
    def create_repository(self) -> Any  # Line 221
    def create_engine(self, config: Any, strategy: Any, **kwargs) -> Any  # Line 225
```

#### Class: `TradeSimulatorInterface`

**Inherits**: Protocol
**Purpose**: Interface for TradeSimulator

```python
class TradeSimulatorInterface(Protocol):
    async def execute_order(self, order_request: Any, market_data: Any, **kwargs) -> dict[str, Any]  # Line 234
    async def get_simulation_results(self) -> dict[str, Any]  # Line 238
```

#### Class: `CacheServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for caching service to decouple from Redis

```python
class CacheServiceInterface(Protocol):
    async def initialize(self) -> None  # Line 247
    async def get(self, key: str) -> Any  # Line 251
    async def set(self, key: str, value: Any, ttl: int) -> None  # Line 255
    async def delete(self, key: str) -> None  # Line 259
    async def clear_pattern(self, pattern: str) -> int  # Line 263
    async def get_stats(self) -> dict[str, Any]  # Line 267
    async def cleanup(self) -> None  # Line 271
```

### File: metrics.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.utils.decimal_utils import to_decimal`
- `from src.utils.decorators import time_execution`
- `from src.utils.financial_calculations import calculate_max_drawdown`
- `from src.utils.financial_calculations import calculate_sharpe_ratio`

#### Class: `BacktestMetrics`

**Inherits**: BaseComponent
**Purpose**: Container for all backtest metrics

```python
class BacktestMetrics(BaseComponent):
    def __init__(self) -> None  # Line 32
    def add(self, name: str, value: Any) -> None  # Line 37
    def get(self, name: str, default: Any = None) -> Any  # Line 41
    def to_dict(self) -> dict[str, Any]  # Line 45
```

#### Class: `MetricsCalculator`

**Purpose**: Calculator for comprehensive backtest metrics

```python
class MetricsCalculator:
    def __init__(self, risk_free_rate: float | None = None) -> None  # Line 63
    def calculate_all(self, ...) -> dict[str, Any]  # Line 79
    def _calculate_return_metrics(self, equity_curve: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]  # Line 123
    def _calculate_risk_adjusted_metrics(self, daily_returns: list[float]) -> dict[str, Any]  # Line 153
    def _calculate_drawdown_metrics(self, equity_curve: list[dict[str, Any]]) -> dict[str, Any]  # Line 175
    def _calculate_trade_statistics(self, trades: list[dict[str, Any]]) -> dict[str, Any]  # Line 227
    def _calculate_risk_metrics(self, daily_returns: list[float], initial_capital: float) -> dict[str, Any]  # Line 316
    def calculate_rolling_metrics(self, equity_curve: list[dict[str, Any]], window: int = 30) -> pd.DataFrame  # Line 367
```

### File: repository.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.event_constants import BacktestEvents`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `BacktestRepository`

**Inherits**: BaseComponent
**Purpose**: Repository for backtesting data operations

```python
class BacktestRepository(BaseComponent):
    def __init__(self, db_manager: DatabaseServiceInterface)  # Line 34
    def _serialize_jsonb_field(self, data: Any) -> Any  # Line 45
    async def save_backtest_result(self, result_data: dict[str, Any], request_data: dict[str, Any]) -> str  # Line 77
    async def get_backtest_result(self, result_id: str) -> dict[str, Any] | None  # Line 191
    async def list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> list[dict[str, Any]]  # Line 258
    async def delete_backtest_result(self, result_id: str) -> bool  # Line 306
    async def save_trade_history(self, backtest_id: str, trades: list[dict[str, Any]]) -> None  # Line 342
    async def get_trade_history(self, backtest_id: str) -> list[dict[str, Any]]  # Line 378
    async def cleanup_old_results(self, days_old: int = 30) -> int  # Line 417
```

### File: service.py

**Key Imports:**
- `from src.backtesting.utils import convert_market_records_to_dataframe`
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.service import BaseService`
- `from src.core.config import Config`

#### Class: `BacktestRequest`

**Inherits**: BaseModel
**Purpose**: Comprehensive backtest request model with validation

```python
class BacktestRequest(BaseModel):
    def validate_dates(cls, v: datetime, info) -> datetime  # Line 96
    def validate_symbols(cls, v: list[str]) -> list[str]  # Line 102
```

#### Class: `BacktestCacheEntry`

**Inherits**: BaseModel
**Purpose**: Cache entry for backtest results

```python
class BacktestCacheEntry(BaseModel):
    def is_expired(self) -> bool  # Line 116
```

#### Class: `BacktestService`

**Inherits**: BaseService, ErrorPropagationMixin
**Purpose**: Comprehensive BacktestService integrating all service layers

```python
class BacktestService(BaseService, ErrorPropagationMixin):
    def __init__(self, config: Config, **services)  # Line 133
    async def initialize(self) -> None  # Line 249
    async def _initialize_impl(self) -> None  # Line 253
    async def _initialize_services(self) -> None  # Line 273
    async def _initialize_backtesting_components(self) -> None  # Line 296
    async def run_backtest(self, request: BacktestRequest) -> 'BacktestResult'  # Line 318
    async def run_backtest_from_dict(self, request_data: dict[str, Any]) -> 'BacktestResult'  # Line 322
    async def serialize_result(self, result: 'BacktestResult') -> dict[str, Any]  # Line 332
    async def _run_backtest_impl(self, request: BacktestRequest) -> 'BacktestResult'  # Line 365
    async def _execute_backtest_pipeline(self, backtest_id: str, request: BacktestRequest) -> 'BacktestResult'  # Line 410
    async def _prepare_market_data(self, request: BacktestRequest) -> dict[str, pd.DataFrame]  # Line 445
    async def _initialize_strategy(self, strategy_config: dict[str, Any]) -> Any  # Line 480
    async def _setup_risk_management(self, risk_config: dict[str, Any]) -> Any  # Line 517
    async def _run_core_simulation(self, ...) -> dict[str, Any]  # Line 530
    async def _run_advanced_analysis(self, ...) -> dict[str, Any]  # Line 593
    async def _consolidate_results(self, ...) -> 'BacktestResult'  # Line 642
    async def _update_backtest_stage(self, backtest_id: str, stage: str, progress: int) -> None  # Line 764
    def _generate_request_hash(self, request: BacktestRequest) -> str  # Line 773
    async def _get_cached_result(self, request: BacktestRequest) -> 'BacktestResult | None'  # Line 801
    async def _cache_result(self, request: BacktestRequest, result: 'BacktestResult') -> None  # Line 823
    async def _persist_result(self, request: BacktestRequest, result: 'BacktestResult') -> str | None  # Line 851
    async def get_active_backtests(self) -> dict[str, dict[str, Any]]  # Line 871
    async def _get_active_backtests_impl(self) -> dict[str, dict[str, Any]]  # Line 877
    async def cancel_backtest(self, backtest_id: str) -> bool  # Line 881
    async def _cancel_backtest_impl(self, backtest_id: str) -> bool  # Line 887
    async def clear_cache(self, pattern: str = '*') -> int  # Line 895
    async def _clear_cache_impl(self, pattern: str = '*') -> int  # Line 899
    async def get_cache_stats(self) -> dict[str, Any]  # Line 923
    async def get_backtest_result(self, result_id: str) -> dict[str, Any] | None  # Line 941
    async def list_backtest_results(self, limit: int = 50, offset: int = 0, strategy_type: str | None = None) -> list[dict[str, Any]]  # Line 954
    async def delete_backtest_result(self, result_id: str) -> bool  # Line 969
    async def health_check(self) -> HealthCheckResult  # Line 982
    async def cleanup(self) -> None  # Line 1034
    async def _safe_async_cleanup(self, service_name: str, cleanup_coro) -> None  # Line 1109
    def _propagate_error_with_boundary_validation(self, error: Exception, context: str, source_module: str, target_module: str) -> None  # Line 1117
```

### File: simulator.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import ExecutionAlgorithm`
- `from src.core.types import OrderRequest`
- `from src.core.types import OrderSide`

#### Class: `SimulationConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for trade simulation

```python
class SimulationConfig(BaseModel):
```

#### Class: `SimulatedOrder`

**Inherits**: BaseModel
**Purpose**: Extended order representation for simulation tracking

```python
class SimulatedOrder(BaseModel):
```

#### Class: `TradeSimulator`

**Purpose**: Realistic trade execution simulator

```python
class TradeSimulator:
    def __init__(self, config: SimulationConfig, slippage_model: Any = None)  # Line 104
    async def execute_order(self, ...) -> dict[str, Any]  # Line 121
    async def _simulate_latency(self) -> None  # Line 184
    def _should_reject_order(self) -> bool  # Line 190
    def _create_rejection_result(self, order: SimulatedOrder, reason: str) -> dict[str, Any]  # Line 194
    async def _execute_market_order(self, ...) -> dict[str, Any]  # Line 206
    async def _execute_limit_order(self, ...) -> dict[str, Any]  # Line 255
    async def _execute_stop_order(self, order: SimulatedOrder, market_data: pd.Series) -> dict[str, Any]  # Line 321
    async def _execute_with_algorithm(self, ...) -> dict[str, Any]  # Line 382
    async def _execute_twap(self, order: SimulatedOrder, market_data: pd.Series, params: dict[str, Any]) -> dict[str, Any]  # Line 405
    async def _execute_vwap(self, order: SimulatedOrder, market_data: pd.Series, params: dict[str, Any]) -> dict[str, Any]  # Line 453
    async def _execute_iceberg(self, order: SimulatedOrder, market_data: pd.Series, params: dict[str, Any]) -> dict[str, Any]  # Line 489
    def _calculate_market_impact(self, order_size: Decimal, volume: Decimal) -> Decimal  # Line 560
    async def check_pending_orders(self, market_data: dict[str, pd.Series]) -> list[dict[str, Any]]  # Line 594
    def calculate_execution_costs(self, trades: list[dict[str, Any]]) -> dict[str, Decimal]  # Line 625
    def get_execution_statistics(self) -> dict[str, Any]  # Line 661
    def cleanup(self) -> None  # Line 687
    async def get_simulation_results(self) -> dict[str, Any]  # Line 700
    def _calculate_daily_returns(self, equity_curve: list[dict[str, Any]]) -> list[float]  # Line 707
```

### File: utils.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`

#### Functions:

```python
def convert_market_records_to_dataframe(records: list[Any]) -> pd.DataFrame  # Line 14
def get_backtest_engine_factory(injector: DependencyInjector) -> Any  # Line 69
def create_component_with_factory(injector: DependencyInjector, component_name: str) -> Any  # Line 82
```

---
**Generated**: Complete reference for backtesting module
**Total Classes**: 31
**Total Functions**: 6