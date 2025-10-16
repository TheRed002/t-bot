# RISK_MANAGEMENT Module Reference

## INTEGRATION
**Dependencies**: core, database, error_handling, monitoring, state, utils
**Used By**: None
**Provides**: AbstractRiskService, AdaptiveRiskManager, BaseRiskManager, CircuitBreakerManager, EnvironmentAwareRiskManager, PortfolioLimitsService, PositionSizingService, RiskManagementController, RiskManager, RiskMetricsService, RiskMonitoringService, RiskService, RiskValidationService
**Patterns**: Async Operations, Circuit Breaker, Component Architecture, Dependency Injection, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Performance**:
- Parallel execution
- Caching
**Architecture**:
- AdaptiveRiskManager inherits from base architecture
- BaseRiskManager inherits from base architecture
- RiskManagementController inherits from base architecture

## MODULE OVERVIEW
**Files**: 26 Python files
**Classes**: 54
**Functions**: 13

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `AdaptiveRiskManager` ✅

**Inherits**: BaseComponent
**Purpose**: Adaptive risk management system that adjusts parameters based on market regimes
**Status**: Complete

**Implemented Methods:**
- `async calculate_adaptive_position_size(self, signal: Signal, current_regime: MarketRegime, portfolio_value: Decimal) -> Decimal` - Line 131
- `async calculate_adaptive_stop_loss(self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal` - Line 197
- `async calculate_adaptive_take_profit(self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal` - Line 284
- `async calculate_adaptive_portfolio_limits(self, current_regime: MarketRegime, base_limits: dict[str, Any]) -> dict[str, Any]` - Line 355
- `async run_stress_test(self, portfolio_positions: list[Position], scenario_name: str = 'market_crash') -> dict[str, Any]` - Line 417
- `get_adaptive_parameters(self, regime: MarketRegime) -> dict[str, Any]` - Line 599
- `get_stress_test_scenarios(self) -> list[str]` - Line 621
- `update_regime_detector(self, new_detector: 'MarketRegimeDetector') -> None` - Line 630

### Implementation: `BaseRiskManager` 🔧

**Inherits**: BaseComponent, ABC
**Purpose**: Abstract base class for risk management implementations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async calculate_position_size(self, signal: Signal, portfolio_value: Decimal) -> Decimal` - Line 89
- `async validate_signal(self, signal: Signal) -> bool` - Line 108
- `async validate_order(self, order: OrderRequest, portfolio_value: Decimal) -> bool` - Line 125
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 144
- `async check_portfolio_limits(self, new_position: Position) -> bool` - Line 164
- `async should_exit_position(self, position: Position, market_data: MarketData) -> bool` - Line 181
- `async update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None` - Line 196
- `async get_risk_summary(self) -> dict[str, Any]` - Line 222
- `async emergency_stop(self, reason: str) -> None` - Line 243
- `async validate_risk_parameters(self) -> bool` - Line 265
- `async cleanup(self) -> None` - Line 414

### Implementation: `BaseCircuitBreaker` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for circuit breakers
**Status**: Abstract Base Class

**Implemented Methods:**
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 115
- `async get_threshold_value(self) -> Decimal` - Line 128
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 138
- `async evaluate(self, data: dict[str, Any]) -> bool` - Line 151
- `get_status(self) -> dict[str, Any]` - Line 267
- `reset(self) -> None` - Line 277

### Implementation: `DailyLossLimitBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for daily loss limit monitoring
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 311
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 315
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 327

### Implementation: `DrawdownLimitBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for portfolio drawdown monitoring
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 350
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 354
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 369

### Implementation: `VolatilitySpikeBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for volatility spike detection
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 405
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 409
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 464

### Implementation: `ModelConfidenceBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for model confidence degradation
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 489
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 493
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 499

### Implementation: `SystemErrorRateBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for system error rate monitoring
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 529
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 533
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 593

### Implementation: `CorrelationSpikeBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for portfolio correlation spike detection
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 639
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 643
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 672
- `get_correlation_metrics(self) -> dict[str, Any] | None` - Line 746
- `async get_position_limits(self) -> dict[str, Any]` - Line 762
- `async cleanup_old_data(self, cutoff_time) -> None` - Line 773
- `reset(self) -> None` - Line 777

### Implementation: `CircuitBreakerManager` ✅

**Purpose**: Manager for all circuit breakers in the system
**Status**: Complete

**Implemented Methods:**
- `async evaluate_all(self, data: dict[str, Any]) -> dict[str, bool]` - Line 831
- `async get_status(self) -> dict[str, Any]` - Line 924
- `reset_all(self) -> None` - Line 933
- `get_triggered_breakers(self) -> list[str]` - Line 940
- `is_trading_allowed(self) -> bool` - Line 950
- `async cleanup_resources(self) -> None` - Line 979

### Implementation: `RiskManagementController` ✅

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Controller for risk management operations
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, ...) -> Decimal` - Line 70
- `async validate_signal(self, signal: Signal) -> bool` - Line 138
- `async validate_order(self, order: OrderRequest) -> bool` - Line 171
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 204
- `async validate_portfolio_limits(self, new_position: Position) -> bool` - Line 264
- `async start_monitoring(self, interval: int = 60) -> None` - Line 295
- `async stop_monitoring(self) -> None` - Line 314
- `async get_risk_summary(self) -> dict[str, Any]` - Line 328

### Implementation: `RiskCalculator` ✅

**Inherits**: BaseComponent
**Purpose**: Centralized risk calculator with caching
**Status**: Complete

**Implemented Methods:**
- `calculate_var(self, ...) -> Decimal` - Line 48
- `calculate_expected_shortfall(self, returns: list[Decimal], confidence_level: Decimal = Any) -> Decimal` - Line 67
- `calculate_sharpe_ratio(self, returns: list[Decimal], risk_free_rate: Decimal = Any) -> Decimal` - Line 82
- `calculate_sortino_ratio(self, ...) -> Decimal` - Line 105
- `calculate_max_drawdown(self, values: list[Decimal]) -> tuple[Decimal, int, int]` - Line 132
- `calculate_calmar_ratio(self, returns: list[Decimal], period_years: Decimal = Any) -> Decimal` - Line 149
- `async calculate_portfolio_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 164
- `clear_cache(self) -> None` - Line 304
- `update_history(self, symbol: str, price: Decimal, return_value: float | None = None) -> None` - Line 309

### Implementation: `RiskMonitor` ✅

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Legacy risk monitor that delegates to centralized utilities
**Status**: Complete

**Implemented Methods:**
- `add_observer(self, observer) -> None` - Line 45
- `remove_observer(self, observer) -> None` - Line 50
- `async monitor_metrics(self, metrics: RiskMetrics) -> None` - Line 58
- `async monitor_portfolio(self, portfolio_data: dict[str, Any]) -> None` - Line 85
- `set_threshold(self, key: str, value) -> None` - Line 108
- `get_thresholds(self) -> dict[str, Any]` - Line 120
- `async start_monitoring(self, interval: int = 60) -> None` - Line 124
- `async stop_monitoring(self) -> None` - Line 138
- `get_alerts(self) -> list[RiskAlert]` - Line 195

### Implementation: `PositionSizer` ✅

**Purpose**: Centralized position sizer using strategy pattern
**Status**: Complete

**Implemented Methods:**
- `calculate_position_size(self, ...) -> Decimal` - Line 49
- `set_limits(self, max_size: Decimal | None = None, min_size: Decimal | None = None) -> None` - Line 131
- `update_price_history(self, symbol: str, price: Decimal) -> None` - Line 145
- `get_position_metrics(self, symbol: str) -> dict` - Line 155

### Implementation: `RiskValidator` ✅

**Inherits**: BaseComponent, ValidatorInterface, ErrorPropagationMixin
**Purpose**: Risk validation using centralized ValidationFramework
**Status**: Complete

**Implemented Methods:**
- `validate(self, data: Any, **kwargs) -> bool` - Line 66
- `validate_order(self, ...) -> bool` - Line 92
- `validate_signal(self, signal: Signal, current_risk_level: RiskLevel | None = None) -> bool` - Line 147
- `validate_position(self, position: Position, portfolio_value: Decimal) -> bool` - Line 180
- `validate_portfolio(self, portfolio_data: dict[str, Any], **kwargs) -> bool` - Line 204
- `update_limits(self, new_limits: RiskLimits) -> None` - Line 226

### Implementation: `RiskValidationUtility` ✅

**Purpose**: Utility for risk validation across the system
**Status**: Complete

**Implemented Methods:**
- `validate_order(self, order: OrderRequest, **kwargs) -> bool` - Line 256
- `validate_signal(self, signal: Signal, **kwargs) -> bool` - Line 260
- `validate_portfolio(self, portfolio_data: dict, **kwargs) -> bool` - Line 264
- `set_risk_limits(self, limits: RiskLimits) -> None` - Line 268

### Implementation: `CorrelationLevel` ✅

**Inherits**: Enum
**Purpose**: Correlation level classification
**Status**: Complete

### Implementation: `CorrelationThresholds` ✅

**Purpose**: Configuration for correlation-based risk thresholds
**Status**: Complete

### Implementation: `CorrelationMetrics` ✅

**Purpose**: Correlation metrics for portfolio analysis
**Status**: Complete

### Implementation: `CorrelationMonitor` ✅

**Inherits**: BaseComponent
**Purpose**: Real-time correlation monitoring system for portfolio positions
**Status**: Complete

**Implemented Methods:**
- `async update_price_data(self, market_data: MarketData) -> None` - Line 103
- `async calculate_pairwise_correlation(self, symbol1: str, symbol2: str, min_periods: int | None = None) -> Decimal | None` - Line 130
- `async calculate_portfolio_correlation(self, positions: list[Position]) -> CorrelationMetrics` - Line 228
- `async get_position_limits_for_correlation(self, correlation_metrics: CorrelationMetrics) -> dict[str, Any]` - Line 364
- `async cleanup_old_data(self, cutoff_time: datetime) -> None` - Line 401
- `async cleanup_resources(self) -> None` - Line 484
- `get_status(self) -> dict[str, Any]` - Line 508

### Implementation: `RiskDataTransformer` ✅

**Purpose**: Handles consistent data transformation for risk_management module
**Status**: Complete

**Implemented Methods:**
- `transform_signal_to_event_data(signal: Signal, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 22
- `transform_position_to_event_data(position: Position, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 51
- `transform_risk_metrics_to_event_data(risk_metrics: RiskMetrics, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 82
- `transform_error_to_event_data(error, ...) -> dict[str, Any]` - Line 117
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 155
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'risk_management') -> dict[str, Any]` - Line 178
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 203
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]` - Line 261
- `transform_for_batch_processing(cls, ...) -> dict[str, Any]` - Line 294
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 349
- `apply_cross_module_validation(cls, ...) -> dict[str, Any]` - Line 380

### Implementation: `EmergencyState` ✅

**Inherits**: Enum
**Purpose**: Emergency state enumeration
**Status**: Complete

### Implementation: `EmergencyAction` ✅

**Inherits**: Enum
**Purpose**: Emergency action types
**Status**: Complete

### Implementation: `EmergencyEvent` ✅

**Inherits**: BaseModel
**Purpose**: Emergency event record
**Status**: Complete

### Implementation: `EmergencyControls` ✅

**Inherits**: BaseComponent
**Purpose**: Emergency trading controls system
**Status**: Complete

**Implemented Methods:**
- `register_exchange(self, exchange_name: str, exchange: 'ExchangeServiceInterface') -> None` - Line 134
- `async activate_emergency_stop(self, reason: str, trigger_type: CircuitBreakerType) -> None` - Line 140
- `async validate_order_during_emergency(self, order: OrderRequest) -> bool` - Line 361
- `async deactivate_emergency_stop(self, reason: str = 'Manual deactivation') -> None` - Line 470
- `async activate_manual_override(self, user_id: str, reason: str) -> None` - Line 563
- `async deactivate_manual_override(self, user_id: str) -> None` - Line 578
- `async deactivate_circuit_breaker(self, reason: str = 'Manual circuit breaker deactivation') -> None` - Line 595
- `async activate_circuit_breaker(self, event) -> None` - Line 626
- `get_status(self) -> dict[str, Any]` - Line 696
- `is_trading_allowed(self) -> bool` - Line 713
- `get_emergency_events(self, limit: int = 10) -> list[EmergencyEvent]` - Line 717
- `is_circuit_breaker_active(self) -> bool` - Line 721
- `get_active_trigger(self) -> str | None` - Line 733
- `async cleanup_resources(self) -> None` - Line 745

### Implementation: `EnvironmentAwareRiskConfiguration` ✅

**Purpose**: Environment-specific risk configuration
**Status**: Complete

**Implemented Methods:**
- `get_sandbox_risk_config() -> dict[str, Any]` - Line 28
- `get_live_risk_config() -> dict[str, Any]` - Line 48

### Implementation: `EnvironmentAwareRiskManager` ✅

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware risk management functionality
**Status**: Complete

**Implemented Methods:**
- `get_environment_risk_config(self, exchange: str) -> dict[str, Any]` - Line 103
- `calculate_environment_aware_position_size(self, ...) -> Decimal` - Line 116
- `async validate_environment_order(self, ...) -> bool` - Line 190
- `async update_environment_risk_state(self, ...) -> None` - Line 339
- `async reset_environment_risk_state(self, exchange: str) -> None` - Line 384
- `get_environment_risk_metrics(self, exchange: str) -> dict[str, Any]` - Line 399

### Implementation: `RiskManagementFactory` ✅

**Inherits**: RiskManagementFactoryInterface
**Purpose**: Factory for creating risk management components
**Status**: Complete

**Implemented Methods:**
- `create_risk_service(self, correlation_id: str | None = None) -> RiskServiceInterface` - Line 66
- `create_legacy_risk_manager(self) -> RiskManager` - Line 98
- `create_legacy_position_sizer(self) -> PositionSizer` - Line 133
- `create_legacy_risk_calculator(self) -> RiskCalculator` - Line 153
- `create_risk_management_controller(self, correlation_id: str | None = None) -> RiskManagementController` - Line 173
- `get_recommended_component(self) -> RiskService | RiskManager` - Line 223
- `validate_dependencies(self) -> dict[str, bool]` - Line 256
- `async start_services(self) -> None` - Line 290
- `async stop_services(self) -> None` - Line 300
- `get_migration_guide(self) -> dict[str, str]` - Line 310

### Implementation: `CacheServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for cache service implementations
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> Any` - Line 27
- `async set(self, key: str, value: Any, ttl: int | None = None) -> None` - Line 31
- `async delete(self, key: str) -> None` - Line 35
- `async clear(self) -> None` - Line 39
- `async close(self) -> None` - Line 43

### Implementation: `ExchangeServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for exchange service implementations to avoid direct coupling
**Status**: Complete

**Implemented Methods:**
- `async cancel_all_orders(self, symbol: str | None = None) -> int` - Line 52
- `async close_all_positions(self) -> int` - Line 56
- `async get_account_balance(self) -> Decimal` - Line 60

### Implementation: `RiskServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk management service implementations
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, ...) -> Decimal` - Line 69
- `async validate_signal(self, signal: Signal) -> bool` - Line 79
- `async validate_order(self, order: OrderRequest) -> bool` - Line 83
- `async calculate_risk_metrics(self, ...) -> RiskMetrics` - Line 87
- `async should_exit_position(self, position: Position, market_data: MarketData) -> bool` - Line 93
- `get_current_risk_level(self) -> RiskLevel` - Line 97
- `is_emergency_stop_active(self) -> bool` - Line 101
- `async get_risk_summary(self) -> dict[str, Any]` - Line 105
- `async get_portfolio_metrics(self) -> Any` - Line 109
- `async validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]` - Line 113
- `async get_current_risk_limits(self) -> dict[str, Any]` - Line 117

### Implementation: `PositionSizingServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for position sizing service implementations
**Status**: Complete

**Implemented Methods:**
- `async calculate_size(self, ...) -> Decimal` - Line 126
- `async validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool` - Line 136

### Implementation: `RiskMetricsServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk metrics service implementations
**Status**: Complete

**Implemented Methods:**
- `async calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 145
- `async get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal` - Line 151

### Implementation: `RiskValidationServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk validation service implementations
**Status**: Complete

**Implemented Methods:**
- `async validate_signal(self, signal: Signal) -> bool` - Line 162
- `async validate_order(self, order: OrderRequest) -> bool` - Line 166
- `async validate_portfolio_limits(self, new_position: Position) -> bool` - Line 170

### Implementation: `PortfolioLimitsServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for portfolio limits service implementations
**Status**: Complete

**Implemented Methods:**
- `async check_portfolio_limits(self, new_position: Position) -> bool` - Line 179
- `async update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None` - Line 183
- `async update_return_history(self, symbol: str, price: Decimal) -> None` - Line 189
- `async get_portfolio_summary(self) -> dict[str, Any]` - Line 193

### Implementation: `RiskMonitoringServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk monitoring service implementations
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self, interval: int = 60) -> None` - Line 202
- `async stop_monitoring(self) -> None` - Line 206
- `async check_emergency_conditions(self, metrics: RiskMetrics) -> bool` - Line 210
- `async get_risk_summary(self) -> dict[str, Any]` - Line 214

### Implementation: `AbstractRiskService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for risk services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async start(self) -> None` - Line 223
- `async stop(self) -> None` - Line 228
- `async calculate_position_size(self, ...) -> Decimal` - Line 233
- `async validate_signal(self, signal: Signal) -> bool` - Line 244
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 249

### Implementation: `RiskMetricsRepositoryInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk metrics data access
**Status**: Complete

**Implemented Methods:**
- `async get_historical_returns(self, symbol: str, days: int) -> list[Decimal]` - Line 260
- `async get_price_history(self, symbol: str, days: int) -> list[Decimal]` - Line 264
- `async get_portfolio_positions(self) -> list[Position]` - Line 268
- `async save_risk_metrics(self, metrics: RiskMetrics) -> None` - Line 272
- `async get_correlation_data(self, symbols: list[str], days: int) -> dict[str, list[Decimal]]` - Line 276

### Implementation: `PortfolioRepositoryInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for portfolio data access
**Status**: Complete

**Implemented Methods:**
- `async get_current_positions(self) -> list[Position]` - Line 285
- `async get_portfolio_value(self) -> Decimal` - Line 289
- `async get_position_history(self, symbol: str, days: int) -> list[Position]` - Line 293
- `async update_portfolio_limits(self, limits: dict[str, Any]) -> None` - Line 297

### Implementation: `RiskManagementFactoryInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for risk management service factories
**Status**: Abstract Base Class

**Implemented Methods:**
- `create_risk_service(self, correlation_id: str | None = None) -> 'RiskServiceInterface'` - Line 306
- `create_risk_management_controller(self, correlation_id: str | None = None) -> Any` - Line 311
- `validate_dependencies(self) -> dict[str, bool]` - Line 316

### Implementation: `PortfolioLimits` ✅

**Inherits**: BaseComponent
**Purpose**: Portfolio limits enforcer for risk management
**Status**: Complete

**Implemented Methods:**
- `async check_portfolio_limits(self, new_position: Position) -> bool` - Line 91
- `async update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None` - Line 451
- `async update_return_history(self, symbol: str, price: Decimal) -> None` - Line 471
- `async get_portfolio_summary(self) -> dict[str, Any]` - Line 510

### Implementation: `PositionSizer` ✅

**Inherits**: BaseComponent
**Purpose**: Position sizing calculator with multiple algorithms
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, ...) -> Decimal` - Line 57
- `async update_price_history(self, symbol: str, price: Decimal) -> None` - Line 212
- `async get_position_size_summary(self, signal: Signal, portfolio_value: Decimal) -> dict[str, Any]` - Line 271
- `async validate_position_size(self, position_size: Decimal, portfolio_value: Decimal) -> bool` - Line 305
- `get_signal_confidence(self, signal: Signal) -> Decimal` - Line 349

### Implementation: `MarketRegimeDetector` ✅

**Inherits**: BaseComponent
**Purpose**: Market regime detection and classification system
**Status**: Complete

**Implemented Methods:**
- `async detect_volatility_regime(self, symbol: str, price_data: list[float]) -> MarketRegime` - Line 72
- `async detect_trend_regime(self, symbol: str, price_data: list[float]) -> MarketRegime` - Line 119
- `async detect_correlation_regime(self, symbols: list[str], price_data_dict: dict[str, list[float]]) -> MarketRegime` - Line 174
- `async detect_comprehensive_regime(self, market_data: list[MarketData]) -> MarketRegime` - Line 241
- `get_regime_history(self, limit: int = 10) -> list[RegimeChangeEvent]` - Line 422
- `get_current_regime(self) -> MarketRegime` - Line 434
- `get_regime_statistics(self) -> dict[str, Any]` - Line 443

### Implementation: `RiskManager` ✅

**Inherits**: BaseRiskManager
**Purpose**: Legacy Risk Manager implementation for backward compatibility
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> Decimal` - Line 150
- `async validate_signal(self, signal: Signal) -> bool` - Line 226
- `async validate_order(self, order: OrderRequest) -> bool` - Line 291
- `async calculate_risk_metrics(self, ...) -> RiskMetrics` - Line 355
- `update_positions(self, positions: list[Position]) -> None` - Line 405
- `check_risk_limits(self) -> tuple[bool, str]` - Line 439
- `get_position_limits(self) -> PositionLimits` - Line 483
- `async emergency_stop(self, reason: str) -> None` - Line 492
- `calculate_leverage(self) -> Decimal` - Line 549
- `async check_portfolio_limits(self, new_position: Position) -> bool` - Line 602
- `async should_exit_position(self, position: Position, market_data: MarketData) -> bool` - Line 672
- `async get_comprehensive_risk_summary(self) -> dict[str, Any]` - Line 751

### Implementation: `RiskCalculator` ✅

**Inherits**: BaseComponent
**Purpose**: Risk metrics calculator for portfolio risk assessment
**Status**: Complete

**Implemented Methods:**
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 64
- `async update_position_returns(self, symbol: str, price: Decimal) -> None` - Line 185
- `async get_risk_summary(self) -> dict[str, Any]` - Line 211

### Implementation: `RiskConfiguration` ✅

**Inherits**: BaseModel
**Purpose**: Risk management configuration model
**Status**: Complete

### Implementation: `RiskAlert` ✅

**Inherits**: BaseModel
**Purpose**: Risk alert model
**Status**: Complete

### Implementation: `RiskService` ✅

**Inherits**: BaseService
**Purpose**: Enterprise Risk Management Service
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, ...) -> Decimal` - Line 390
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 893
- `async get_portfolio_metrics(self) -> PortfolioMetrics` - Line 1452
- `async validate_signal(self, signal: Signal) -> bool` - Line 1472
- `async validate_order(self, order: OrderRequest) -> bool` - Line 1564
- `async trigger_emergency_stop(self, reason: str) -> None` - Line 1841
- `async reset_emergency_stop(self, reason: str) -> None` - Line 1918
- `async update_price_history(self, symbol: str, price: Decimal) -> None` - Line 1971
- `async get_risk_alerts(self, limit: int | None = None) -> list[RiskAlert]` - Line 2416
- `async acknowledge_risk_alert(self, alert_id: str) -> bool` - Line 2431
- `get_current_risk_level(self) -> RiskLevel` - Line 2450
- `is_emergency_stop_active(self) -> bool` - Line 2454
- `async save_risk_metrics(self, risk_metrics: RiskMetrics) -> None` - Line 2458
- `is_healthy(self) -> bool` - Line 2467
- `async get_risk_summary(self) -> dict[str, Any]` - Line 2497
- `async validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]` - Line 2518
- `async get_current_risk_limits(self) -> dict[str, Any]` - Line 2566
- `reset_metrics(self) -> None` - Line 2692
- `async should_exit_position(self, position: Position, market_data: MarketData) -> bool` - Line 2709
- `async check_emergency_conditions(self, risk_metrics: RiskMetrics) -> bool` - Line 2838
- `async check_portfolio_limits(self, new_position: Position) -> bool` - Line 2902
- `async update_portfolio_state(self, positions: list[Position], available_capital: Decimal) -> None` - Line 2950
- `async risk_monitoring_context(self, operation: str) -> Any` - Line 3006

### Implementation: `PortfolioLimitsService` ✅

**Inherits**: BaseService, PortfolioLimitsServiceInterface
**Purpose**: Service layer for portfolio limits enforcement
**Status**: Complete

**Implemented Methods:**
- `async check_portfolio_limits(self, new_position: Position) -> bool` - Line 37
- `async update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None` - Line 69
- `async update_return_history(self, symbol: str, price: Decimal) -> None` - Line 110
- `async get_portfolio_summary(self) -> dict[str, Any]` - Line 144
- `async start(self) -> None` - Line 161
- `async stop(self) -> None` - Line 165
- `async health_check(self) -> bool` - Line 169

### Implementation: `PositionSizingService` ✅

**Inherits**: BaseService
**Purpose**: Service for calculating position sizes using various methods
**Status**: Complete

**Implemented Methods:**
- `async calculate_size(self, ...) -> Decimal` - Line 62
- `async validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool` - Line 174

### Implementation: `RiskMetricsService` ✅

**Inherits**: BaseService
**Purpose**: Service for calculating comprehensive risk metrics
**Status**: Complete

**Implemented Methods:**
- `async calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 71
- `async get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal` - Line 252

### Implementation: `RiskAlert` ✅

**Purpose**: Risk alert model
**Status**: Complete

**Implemented Methods:**

### Implementation: `RiskMonitoringService` ✅

**Inherits**: BaseService
**Purpose**: Service for real-time risk monitoring and alerting
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self, interval: int = 60) -> None` - Line 103
- `async stop_monitoring(self) -> None` - Line 119
- `async check_emergency_conditions(self, metrics: RiskMetrics) -> bool` - Line 146
- `async monitor_metrics(self, metrics: RiskMetrics) -> None` - Line 178
- `async get_active_alerts(self, limit: int | None = None) -> list[RiskAlert]` - Line 219
- `async acknowledge_alert(self, alert_id: str) -> bool` - Line 244
- `async set_threshold(self, threshold_name: str, value) -> None` - Line 262
- `async get_risk_summary(self) -> dict[str, Any]` - Line 517
- `async publish_risk_event(self, event_type: str, event_data: dict) -> None` - Line 671

### Implementation: `RiskValidationService` ✅

**Inherits**: BaseService
**Purpose**: Service for validating trading signals and orders against risk constraints
**Status**: Complete

**Implemented Methods:**
- `async validate_signal(self, signal: Signal) -> bool` - Line 76
- `async validate_order(self, order: OrderRequest) -> bool` - Line 182
- `async validate_portfolio_limits(self, new_position: Position) -> bool` - Line 240

## COMPLETE API REFERENCE

### File: adaptive_risk.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketRegime`
- `from src.core.types import Position`

#### Class: `AdaptiveRiskManager`

**Inherits**: BaseComponent
**Purpose**: Adaptive risk management system that adjusts parameters based on market regimes

```python
class AdaptiveRiskManager(BaseComponent):
    def __init__(self, config: dict[str, Any], regime_detector: 'MarketRegimeDetector')  # Line 37
    async def calculate_adaptive_position_size(self, signal: Signal, current_regime: MarketRegime, portfolio_value: Decimal) -> Decimal  # Line 131
    async def calculate_adaptive_stop_loss(self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal  # Line 197
    async def calculate_adaptive_take_profit(self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal  # Line 284
    async def calculate_adaptive_portfolio_limits(self, current_regime: MarketRegime, base_limits: dict[str, Any]) -> dict[str, Any]  # Line 355
    async def run_stress_test(self, portfolio_positions: list[Position], scenario_name: str = 'market_crash') -> dict[str, Any]  # Line 417
    async def _get_correlation_regime(self) -> str | None  # Line 513
    async def _calculate_momentum_adjustment(self, symbol: str) -> Decimal  # Line 542
    def get_adaptive_parameters(self, regime: MarketRegime) -> dict[str, Any]  # Line 599
    def get_stress_test_scenarios(self) -> list[str]  # Line 621
    def update_regime_detector(self, new_detector: 'MarketRegimeDetector') -> None  # Line 630
```

### File: base.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`

#### Class: `BaseRiskManager`

**Inherits**: BaseComponent, ABC
**Purpose**: Abstract base class for risk management implementations

```python
class BaseRiskManager(BaseComponent, ABC):
    def __init__(self, ...)  # Line 53
    async def calculate_position_size(self, signal: Signal, portfolio_value: Decimal) -> Decimal  # Line 89
    async def validate_signal(self, signal: Signal) -> bool  # Line 108
    async def validate_order(self, order: OrderRequest, portfolio_value: Decimal) -> bool  # Line 125
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 144
    async def check_portfolio_limits(self, new_position: Position) -> bool  # Line 164
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool  # Line 181
    async def update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None  # Line 196
    async def get_risk_summary(self) -> dict[str, Any]  # Line 222
    async def emergency_stop(self, reason: str) -> None  # Line 243
    async def validate_risk_parameters(self) -> bool  # Line 265
    def _calculate_portfolio_exposure(self, positions: list[Position]) -> Decimal  # Line 302
    def _check_drawdown_limit(self, current_drawdown: Decimal) -> bool  # Line 328
    def _check_daily_loss_limit(self, daily_pnl: Decimal) -> bool  # Line 341
    async def _log_risk_violation(self, violation_type: str, details: dict[str, Any]) -> None  # Line 358
    def _determine_violation_severity(self, violation_type: str, details: dict[str, Any]) -> str  # Line 390
    async def cleanup(self) -> None  # Line 414
```

### File: circuit_breakers.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import CircuitBreakerTriggeredError`
- `from src.core.logging import get_logger`
- `from src.core.types import CircuitBreakerEvent`
- `from src.core.types import CircuitBreakerStatus`

#### Class: `BaseCircuitBreaker`

**Inherits**: ABC
**Purpose**: Abstract base class for circuit breakers

```python
class BaseCircuitBreaker(ABC):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 79
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 115
    async def get_threshold_value(self) -> Decimal  # Line 128
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 138
    async def evaluate(self, data: dict[str, Any]) -> bool  # Line 151
    async def _trigger_circuit_breaker(self, data: dict[str, Any]) -> None  # Line 200
    async def _close_circuit_breaker(self) -> None  # Line 256
    def get_status(self) -> dict[str, Any]  # Line 267
    def reset(self) -> None  # Line 277
```

#### Class: `DailyLossLimitBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for daily loss limit monitoring

```python
class DailyLossLimitBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 298
    async def get_threshold_value(self) -> Decimal  # Line 311
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 315
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 327
```

#### Class: `DrawdownLimitBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for portfolio drawdown monitoring

```python
class DrawdownLimitBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 345
    async def get_threshold_value(self) -> Decimal  # Line 350
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 354
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 369
```

#### Class: `VolatilitySpikeBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for volatility spike detection

```python
class VolatilitySpikeBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 387
    async def get_threshold_value(self) -> Decimal  # Line 405
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 409
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 464
```

#### Class: `ModelConfidenceBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for model confidence degradation

```python
class ModelConfidenceBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 482
    async def get_threshold_value(self) -> Decimal  # Line 489
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 493
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 499
```

#### Class: `SystemErrorRateBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for system error rate monitoring

```python
class SystemErrorRateBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 517
    async def get_threshold_value(self) -> Decimal  # Line 529
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 533
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 593
```

#### Class: `CorrelationSpikeBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for portfolio correlation spike detection

```python
class CorrelationSpikeBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 612
    async def get_threshold_value(self) -> Decimal  # Line 639
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 643
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 672
    def get_correlation_metrics(self) -> dict[str, Any] | None  # Line 746
    async def get_position_limits(self) -> dict[str, Any]  # Line 762
    async def cleanup_old_data(self, cutoff_time) -> None  # Line 773
    def reset(self) -> None  # Line 777
```

#### Class: `CircuitBreakerManager`

**Purpose**: Manager for all circuit breakers in the system

```python
class CircuitBreakerManager:
    def __init__(self, ...)  # Line 793
    async def evaluate_all(self, data: dict[str, Any]) -> dict[str, bool]  # Line 831
    async def get_status(self) -> dict[str, Any]  # Line 924
    def reset_all(self) -> None  # Line 933
    def get_triggered_breakers(self) -> list[str]  # Line 940
    def is_trading_allowed(self) -> bool  # Line 950
    async def _evaluation_context(self, breaker_name: str) -> AsyncIterator[None]  # Line 955
    async def cleanup_resources(self) -> None  # Line 979
```

### File: controller.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.types import MarketData`
- `from src.core.types import OrderRequest`
- `from src.core.types import Position`
- `from src.core.types import RiskMetrics`

#### Class: `RiskManagementController`

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Controller for risk management operations

```python
class RiskManagementController(BaseComponent, ErrorPropagationMixin):
    def __init__(self, ...)  # Line 37
    async def calculate_position_size(self, ...) -> Decimal  # Line 70
    async def validate_signal(self, signal: Signal) -> bool  # Line 138
    async def validate_order(self, order: OrderRequest) -> bool  # Line 171
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 204
    async def validate_portfolio_limits(self, new_position: Position) -> bool  # Line 264
    async def start_monitoring(self, interval: int = 60) -> None  # Line 295
    async def stop_monitoring(self) -> None  # Line 314
    async def get_risk_summary(self) -> dict[str, Any]  # Line 328
```

### File: calculator.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.types import Position`
- `from src.core.types.market import MarketData`
- `from src.core.types.risk import RiskLevel`
- `from src.core.types.risk import RiskMetrics`

#### Class: `RiskCalculator`

**Inherits**: BaseComponent
**Purpose**: Centralized risk calculator with caching

```python
class RiskCalculator(BaseComponent):
    def __init__(self)  # Line 38
    def calculate_var(self, ...) -> Decimal  # Line 48
    def calculate_expected_shortfall(self, returns: list[Decimal], confidence_level: Decimal = Any) -> Decimal  # Line 67
    def calculate_sharpe_ratio(self, returns: list[Decimal], risk_free_rate: Decimal = Any) -> Decimal  # Line 82
    def calculate_sortino_ratio(self, ...) -> Decimal  # Line 105
    def calculate_max_drawdown(self, values: list[Decimal]) -> tuple[Decimal, int, int]  # Line 132
    def calculate_calmar_ratio(self, returns: list[Decimal], period_years: Decimal = Any) -> Decimal  # Line 149
    async def calculate_portfolio_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 164
    def _calculate_current_drawdown(self) -> Decimal  # Line 231
    def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal  # Line 239
    def _determine_risk_level(self, var: Decimal, max_dd: Decimal, sharpe: Decimal) -> RiskLevel  # Line 289
    def clear_cache(self) -> None  # Line 304
    def update_history(self, symbol: str, price: Decimal, return_value: float | None = None) -> None  # Line 309
```

#### Functions:

```python
def get_risk_calculator() -> RiskCalculator  # Line 394
```

### File: monitor.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.types.risk import RiskAlert`
- `from src.core.types.risk import RiskMetrics`
- `from src.utils.decimal_utils import format_decimal`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `RiskMonitor`

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Legacy risk monitor that delegates to centralized utilities

```python
class RiskMonitor(BaseComponent, ErrorPropagationMixin):
    def __init__(self, messaging_coordinator: MessagingCoordinator | None = None) -> None  # Line 29
    def add_observer(self, observer) -> None  # Line 45
    def remove_observer(self, observer) -> None  # Line 50
    async def monitor_metrics(self, metrics: RiskMetrics) -> None  # Line 58
    async def monitor_portfolio(self, portfolio_data: dict[str, Any]) -> None  # Line 85
    def set_threshold(self, key: str, value) -> None  # Line 108
    def get_thresholds(self) -> dict[str, Any]  # Line 120
    async def start_monitoring(self, interval: int = 60) -> None  # Line 124
    async def stop_monitoring(self) -> None  # Line 138
    async def _monitoring_loop(self, interval: int) -> None  # Line 167
    async def _cleanup_resources(self) -> None  # Line 177
    def get_alerts(self) -> list[RiskAlert]  # Line 195
```

### File: position_sizer.py

**Key Imports:**
- `from src.core.dependency_injection import injectable`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import PositionSizeMethod`

#### Class: `PositionSizer`

**Purpose**: Centralized position sizer using strategy pattern

```python
class PositionSizer:
    def __init__(self) -> None  # Line 37
    def calculate_position_size(self, ...) -> Decimal  # Line 49
    def _apply_limits(self, position_size: Decimal, portfolio_value: Decimal) -> Decimal  # Line 108
    def set_limits(self, max_size: Decimal | None = None, min_size: Decimal | None = None) -> None  # Line 131
    def update_price_history(self, symbol: str, price: Decimal) -> None  # Line 145
    def get_position_metrics(self, symbol: str) -> dict  # Line 155
```

### File: validator.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.dependency_injection import injectable`
- `from src.core.exceptions import PositionLimitError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `RiskValidator`

**Inherits**: BaseComponent, ValidatorInterface, ErrorPropagationMixin
**Purpose**: Risk validation using centralized ValidationFramework

```python
class RiskValidator(BaseComponent, ValidatorInterface, ErrorPropagationMixin):
    def __init__(self, ...)  # Line 31
    def _default_limits(self) -> RiskLimits  # Line 53
    def validate(self, data: Any, **kwargs) -> bool  # Line 66
    def validate_order(self, ...) -> bool  # Line 92
    def validate_signal(self, signal: Signal, current_risk_level: RiskLevel | None = None) -> bool  # Line 147
    def validate_position(self, position: Position, portfolio_value: Decimal) -> bool  # Line 180
    def validate_portfolio(self, portfolio_data: dict[str, Any], **kwargs) -> bool  # Line 204
    def update_limits(self, new_limits: RiskLimits) -> None  # Line 226
```

#### Class: `RiskValidationUtility`

**Purpose**: Utility for risk validation across the system

```python
class RiskValidationUtility:
    def __init__(self)  # Line 248
    def validate_order(self, order: OrderRequest, **kwargs) -> bool  # Line 256
    def validate_signal(self, signal: Signal, **kwargs) -> bool  # Line 260
    def validate_portfolio(self, portfolio_data: dict, **kwargs) -> bool  # Line 264
    def set_risk_limits(self, limits: RiskLimits) -> None  # Line 268
```

### File: correlation_monitor.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`

#### Class: `CorrelationLevel`

**Inherits**: Enum
**Purpose**: Correlation level classification

```python
class CorrelationLevel(Enum):
```

#### Class: `CorrelationThresholds`

**Purpose**: Configuration for correlation-based risk thresholds

```python
class CorrelationThresholds:
```

#### Class: `CorrelationMetrics`

**Purpose**: Correlation metrics for portfolio analysis

```python
class CorrelationMetrics:
```

#### Class: `CorrelationMonitor`

**Inherits**: BaseComponent
**Purpose**: Real-time correlation monitoring system for portfolio positions

```python
class CorrelationMonitor(BaseComponent):
    def __init__(self, config: Config, thresholds: CorrelationThresholds | None = None)  # Line 68
    async def update_price_data(self, market_data: MarketData) -> None  # Line 103
    async def calculate_pairwise_correlation(self, symbol1: str, symbol2: str, min_periods: int | None = None) -> Decimal | None  # Line 130
    async def calculate_portfolio_correlation(self, positions: list[Position]) -> CorrelationMetrics  # Line 228
    async def get_position_limits_for_correlation(self, correlation_metrics: CorrelationMetrics) -> dict[str, Any]  # Line 364
    async def cleanup_old_data(self, cutoff_time: datetime) -> None  # Line 401
    def _periodic_cache_cleanup(self) -> None  # Line 457
    async def cleanup_resources(self) -> None  # Line 484
    def get_status(self) -> dict[str, Any]  # Line 508
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import Position`
- `from src.core.types import RiskMetrics`
- `from src.core.types import Signal`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `RiskDataTransformer`

**Purpose**: Handles consistent data transformation for risk_management module

```python
class RiskDataTransformer:
    def transform_signal_to_event_data(signal: Signal, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 22
    def transform_position_to_event_data(position: Position, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 51
    def transform_risk_metrics_to_event_data(risk_metrics: RiskMetrics, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 82
    def transform_error_to_event_data(error, ...) -> dict[str, Any]  # Line 117
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 155
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'risk_management') -> dict[str, Any]  # Line 178
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 203
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]  # Line 261
    def transform_for_batch_processing(cls, ...) -> dict[str, Any]  # Line 294
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 349
    def apply_cross_module_validation(cls, ...) -> dict[str, Any]  # Line 380
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`
- `from src.database.repository.risk import PortfolioRepository`
- `from src.database.repository.risk import PortfolioRepositoryImpl`
- `from src.database.repository.risk import RiskMetricsRepository`

#### Functions:

```python
def register_risk_management_services(injector: DependencyInjector) -> None  # Line 41
def configure_risk_management_dependencies(injector: DependencyInjector | None = None) -> DependencyInjector  # Line 273
def get_risk_service(injector: DependencyInjector) -> 'RiskService'  # Line 296
def get_position_sizing_service(injector: DependencyInjector) -> 'PositionSizingService'  # Line 301
def get_risk_validation_service(injector: DependencyInjector) -> 'RiskValidationService'  # Line 306
def get_risk_metrics_service(injector: DependencyInjector) -> 'RiskMetricsService'  # Line 311
def get_risk_monitoring_service(injector: DependencyInjector) -> 'RiskMonitoringService'  # Line 316
def get_risk_management_factory(injector: DependencyInjector) -> 'RiskManagementFactoryInterface'  # Line 321
```

### File: emergency_controls.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import EmergencyStopError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import CircuitBreakerType`

#### Class: `EmergencyState`

**Inherits**: Enum
**Purpose**: Emergency state enumeration

```python
class EmergencyState(Enum):
```

#### Class: `EmergencyAction`

**Inherits**: Enum
**Purpose**: Emergency action types

```python
class EmergencyAction(Enum):
```

#### Class: `EmergencyEvent`

**Inherits**: BaseModel
**Purpose**: Emergency event record

```python
class EmergencyEvent(BaseModel):
```

#### Class: `EmergencyControls`

**Inherits**: BaseComponent
**Purpose**: Emergency trading controls system

```python
class EmergencyControls(BaseComponent):
    def __init__(self, ...)  # Line 89
    def register_exchange(self, exchange_name: str, exchange: 'ExchangeServiceInterface') -> None  # Line 134
    async def activate_emergency_stop(self, reason: str, trigger_type: CircuitBreakerType) -> None  # Line 140
    async def _execute_emergency_procedures(self) -> None  # Line 185
    async def _cancel_all_pending_orders(self) -> None  # Line 217
    async def _close_all_positions(self) -> None  # Line 273
    async def _block_new_orders(self) -> None  # Line 347
    async def _switch_to_safe_mode(self) -> None  # Line 353
    async def validate_order_during_emergency(self, order: OrderRequest) -> bool  # Line 361
    async def _validate_recovery_order(self, order: OrderRequest) -> bool  # Line 394
    async def _get_portfolio_value(self) -> Decimal  # Line 446
    async def deactivate_emergency_stop(self, reason: str = 'Manual deactivation') -> None  # Line 470
    async def _recovery_validation_timer(self) -> None  # Line 520
    async def _validate_recovery_completion(self) -> bool  # Line 532
    async def activate_manual_override(self, user_id: str, reason: str) -> None  # Line 563
    async def deactivate_manual_override(self, user_id: str) -> None  # Line 578
    async def deactivate_circuit_breaker(self, reason: str = 'Manual circuit breaker deactivation') -> None  # Line 595
    async def activate_circuit_breaker(self, event) -> None  # Line 626
    def get_status(self) -> dict[str, Any]  # Line 696
    def is_trading_allowed(self) -> bool  # Line 713
    def get_emergency_events(self, limit: int = 10) -> list[EmergencyEvent]  # Line 717
    def is_circuit_breaker_active(self) -> bool  # Line 721
    def get_active_trigger(self) -> str | None  # Line 733
    async def cleanup_resources(self) -> None  # Line 745
```

### File: environment_integration.py

**Key Imports:**
- `from src.core.exceptions import ConfigurationError`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.integration.environment_aware_service import EnvironmentAwareServiceMixin`
- `from src.core.integration.environment_aware_service import EnvironmentContext`
- `from src.core.logging import get_logger`

#### Class: `EnvironmentAwareRiskConfiguration`

**Purpose**: Environment-specific risk configuration

```python
class EnvironmentAwareRiskConfiguration:
    def get_sandbox_risk_config() -> dict[str, Any]  # Line 28
    def get_live_risk_config() -> dict[str, Any]  # Line 48
```

#### Class: `EnvironmentAwareRiskManager`

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware risk management functionality

```python
class EnvironmentAwareRiskManager(EnvironmentAwareServiceMixin):
    def __init__(self, *args, **kwargs)  # Line 75
    async def _update_service_environment(self, context: EnvironmentContext) -> None  # Line 80
    def get_environment_risk_config(self, exchange: str) -> dict[str, Any]  # Line 103
    def calculate_environment_aware_position_size(self, ...) -> Decimal  # Line 116
    async def validate_environment_order(self, ...) -> bool  # Line 190
    async def _validate_production_order(self, order_request: OrderRequest, exchange: str, risk_config: dict[str, Any]) -> bool  # Line 223
    async def _validate_sandbox_order(self, order_request: OrderRequest, exchange: str, risk_config: dict[str, Any]) -> bool  # Line 255
    async def _validate_common_risk_rules(self, ...) -> bool  # Line 272
    def _calculate_volatility_adjustment(self, market_data: Any, risk_config: dict[str, Any]) -> Decimal  # Line 300
    async def _detect_suspicious_order_pattern(self, order_request: OrderRequest, exchange: str) -> bool  # Line 326
    async def update_environment_risk_state(self, ...) -> None  # Line 339
    async def _notify_circuit_breaker_triggered(self, exchange: str, loss_amount: Decimal) -> None  # Line 371
    async def reset_environment_risk_state(self, exchange: str) -> None  # Line 384
    def get_environment_risk_metrics(self, exchange: str) -> dict[str, Any]  # Line 399
```

### File: factory.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.exceptions import DependencyError`
- `from src.core.logging import get_logger`

#### Class: `RiskManagementFactory`

**Inherits**: RiskManagementFactoryInterface
**Purpose**: Factory for creating risk management components

```python
class RiskManagementFactory(RiskManagementFactoryInterface):
    def __init__(self, injector: DependencyInjector | None = None)  # Line 42
    def create_risk_service(self, correlation_id: str | None = None) -> RiskServiceInterface  # Line 66
    def create_legacy_risk_manager(self) -> RiskManager  # Line 98
    def create_legacy_position_sizer(self) -> PositionSizer  # Line 133
    def create_legacy_risk_calculator(self) -> RiskCalculator  # Line 153
    def create_risk_management_controller(self, correlation_id: str | None = None) -> RiskManagementController  # Line 173
    def get_recommended_component(self) -> RiskService | RiskManager  # Line 223
    def validate_dependencies(self) -> dict[str, bool]  # Line 256
    async def start_services(self) -> None  # Line 290
    async def stop_services(self) -> None  # Line 300
    def get_migration_guide(self) -> dict[str, str]  # Line 310
```

#### Functions:

```python
def get_risk_factory(injector: DependencyInjector | None = None) -> RiskManagementFactory  # Line 350
def create_risk_service(injector: DependencyInjector, correlation_id: str | None = None) -> RiskServiceInterface  # Line 374
def create_recommended_risk_component(injector: DependencyInjector) -> RiskService | RiskManager  # Line 392
def create_risk_management_controller(injector: DependencyInjector, correlation_id: str | None = None) -> RiskManagementController  # Line 406
```

### File: interfaces.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import OrderRequest`
- `from src.core.types import Position`
- `from src.core.types import PositionSizeMethod`
- `from src.core.types import RiskLevel`

#### Class: `CacheServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for cache service implementations

```python
class CacheServiceInterface(Protocol):
    async def get(self, key: str) -> Any  # Line 27
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None  # Line 31
    async def delete(self, key: str) -> None  # Line 35
    async def clear(self) -> None  # Line 39
    async def close(self) -> None  # Line 43
```

#### Class: `ExchangeServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for exchange service implementations to avoid direct coupling

```python
class ExchangeServiceInterface(Protocol):
    async def cancel_all_orders(self, symbol: str | None = None) -> int  # Line 52
    async def close_all_positions(self) -> int  # Line 56
    async def get_account_balance(self) -> Decimal  # Line 60
```

#### Class: `RiskServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk management service implementations

```python
class RiskServiceInterface(Protocol):
    async def calculate_position_size(self, ...) -> Decimal  # Line 69
    async def validate_signal(self, signal: Signal) -> bool  # Line 79
    async def validate_order(self, order: OrderRequest) -> bool  # Line 83
    async def calculate_risk_metrics(self, ...) -> RiskMetrics  # Line 87
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool  # Line 93
    def get_current_risk_level(self) -> RiskLevel  # Line 97
    def is_emergency_stop_active(self) -> bool  # Line 101
    async def get_risk_summary(self) -> dict[str, Any]  # Line 105
    async def get_portfolio_metrics(self) -> Any  # Line 109
    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]  # Line 113
    async def get_current_risk_limits(self) -> dict[str, Any]  # Line 117
```

#### Class: `PositionSizingServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for position sizing service implementations

```python
class PositionSizingServiceInterface(Protocol):
    async def calculate_size(self, ...) -> Decimal  # Line 126
    async def validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool  # Line 136
```

#### Class: `RiskMetricsServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk metrics service implementations

```python
class RiskMetricsServiceInterface(Protocol):
    async def calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 145
    async def get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal  # Line 151
```

#### Class: `RiskValidationServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk validation service implementations

```python
class RiskValidationServiceInterface(Protocol):
    async def validate_signal(self, signal: Signal) -> bool  # Line 162
    async def validate_order(self, order: OrderRequest) -> bool  # Line 166
    async def validate_portfolio_limits(self, new_position: Position) -> bool  # Line 170
```

#### Class: `PortfolioLimitsServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for portfolio limits service implementations

```python
class PortfolioLimitsServiceInterface(Protocol):
    async def check_portfolio_limits(self, new_position: Position) -> bool  # Line 179
    async def update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None  # Line 183
    async def update_return_history(self, symbol: str, price: Decimal) -> None  # Line 189
    async def get_portfolio_summary(self) -> dict[str, Any]  # Line 193
```

#### Class: `RiskMonitoringServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk monitoring service implementations

```python
class RiskMonitoringServiceInterface(Protocol):
    async def start_monitoring(self, interval: int = 60) -> None  # Line 202
    async def stop_monitoring(self) -> None  # Line 206
    async def check_emergency_conditions(self, metrics: RiskMetrics) -> bool  # Line 210
    async def get_risk_summary(self) -> dict[str, Any]  # Line 214
```

#### Class: `AbstractRiskService`

**Inherits**: ABC
**Purpose**: Abstract base class for risk services

```python
class AbstractRiskService(ABC):
    async def start(self) -> None  # Line 223
    async def stop(self) -> None  # Line 228
    async def calculate_position_size(self, ...) -> Decimal  # Line 233
    async def validate_signal(self, signal: Signal) -> bool  # Line 244
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 249
```

#### Class: `RiskMetricsRepositoryInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk metrics data access

```python
class RiskMetricsRepositoryInterface(Protocol):
    async def get_historical_returns(self, symbol: str, days: int) -> list[Decimal]  # Line 260
    async def get_price_history(self, symbol: str, days: int) -> list[Decimal]  # Line 264
    async def get_portfolio_positions(self) -> list[Position]  # Line 268
    async def save_risk_metrics(self, metrics: RiskMetrics) -> None  # Line 272
    async def get_correlation_data(self, symbols: list[str], days: int) -> dict[str, list[Decimal]]  # Line 276
```

#### Class: `PortfolioRepositoryInterface`

**Inherits**: Protocol
**Purpose**: Protocol for portfolio data access

```python
class PortfolioRepositoryInterface(Protocol):
    async def get_current_positions(self) -> list[Position]  # Line 285
    async def get_portfolio_value(self) -> Decimal  # Line 289
    async def get_position_history(self, symbol: str, days: int) -> list[Position]  # Line 293
    async def update_portfolio_limits(self, limits: dict[str, Any]) -> None  # Line 297
```

#### Class: `RiskManagementFactoryInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for risk management service factories

```python
class RiskManagementFactoryInterface(ABC):
    def create_risk_service(self, correlation_id: str | None = None) -> 'RiskServiceInterface'  # Line 306
    def create_risk_management_controller(self, correlation_id: str | None = None) -> Any  # Line 311
    def validate_dependencies(self) -> dict[str, bool]  # Line 316
```

### File: portfolio_limits.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import PositionLimitError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import Position`

#### Class: `PortfolioLimits`

**Inherits**: BaseComponent
**Purpose**: Portfolio limits enforcer for risk management

```python
class PortfolioLimits(BaseComponent):
    def __init__(self, config: Config)  # Line 48
    async def check_portfolio_limits(self, new_position: Position) -> bool  # Line 91
    def _check_total_positions_limit(self, new_position: Position) -> bool  # Line 150
    def _check_positions_per_symbol_limit(self, new_position: Position) -> bool  # Line 177
    def _check_portfolio_exposure_limit(self, new_position: Position) -> bool  # Line 205
    def _check_sector_exposure_limit(self, new_position: Position) -> bool  # Line 274
    def _check_correlation_exposure_limit(self, new_position: Position) -> bool  # Line 334
    def _check_leverage_limit(self, new_position: Position) -> bool  # Line 394
    def _get_correlation(self, symbol1: str, symbol2: str) -> Decimal  # Line 414
    async def update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None  # Line 451
    async def update_return_history(self, symbol: str, price: Decimal) -> None  # Line 471
    async def get_portfolio_summary(self) -> dict[str, Any]  # Line 510
    async def _log_risk_violation(self, violation_type: str, details: dict[str, Any]) -> None  # Line 573
    def _log_risk_violation_sync(self, violation_type: str, details: dict[str, Any]) -> None  # Line 586
```

### File: position_sizing.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import PositionSizeMethod`

#### Class: `PositionSizer`

**Inherits**: BaseComponent
**Purpose**: Position sizing calculator with multiple algorithms

```python
class PositionSizer(BaseComponent):
    def __init__(self, config: Config)  # Line 37
    async def calculate_position_size(self, ...) -> Decimal  # Line 57
    async def update_price_history(self, symbol: str, price: Decimal) -> None  # Line 212
    async def get_position_size_summary(self, signal: Signal, portfolio_value: Decimal) -> dict[str, Any]  # Line 271
    async def validate_position_size(self, position_size: Decimal, portfolio_value: Decimal) -> bool  # Line 305
    def get_signal_confidence(self, signal: Signal) -> Decimal  # Line 349
    def _get_kelly_parameters(self, symbol: str) -> dict[str, float]  # Line 361
    def _get_volatility_parameters(self, symbol: str) -> dict[str, float]  # Line 397
```

### File: regime_detection.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.types import MarketData`
- `from src.core.types import MarketRegime`
- `from src.core.types import RegimeChangeEvent`

#### Class: `MarketRegimeDetector`

**Inherits**: BaseComponent
**Purpose**: Market regime detection and classification system

```python
class MarketRegimeDetector(BaseComponent):
    def __init__(self, config: dict[str, Any])  # Line 36
    async def detect_volatility_regime(self, symbol: str, price_data: list[float]) -> MarketRegime  # Line 72
    async def detect_trend_regime(self, symbol: str, price_data: list[float]) -> MarketRegime  # Line 119
    async def detect_correlation_regime(self, symbols: list[str], price_data_dict: dict[str, list[float]]) -> MarketRegime  # Line 174
    async def detect_comprehensive_regime(self, market_data: list[MarketData]) -> MarketRegime  # Line 241
    def _combine_regimes(self, ...) -> MarketRegime  # Line 293
    def _check_regime_change(self, new_regime: MarketRegime) -> None  # Line 351
    def _calculate_change_confidence(self, new_regime: MarketRegime) -> float  # Line 393
    def get_regime_history(self, limit: int = 10) -> list[RegimeChangeEvent]  # Line 422
    def get_current_regime(self) -> MarketRegime  # Line 434
    def get_regime_statistics(self) -> dict[str, Any]  # Line 443
```

### File: risk_manager.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.config import get_config`
- `from src.core.dependency_injection import injectable`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.logging import get_logger`

#### Class: `RiskManager`

**Inherits**: BaseRiskManager
**Purpose**: Legacy Risk Manager implementation for backward compatibility

```python
class RiskManager(BaseRiskManager):
    def __init__(self, ...)  # Line 79
    async def calculate_position_size(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> Decimal  # Line 150
    async def validate_signal(self, signal: Signal) -> bool  # Line 226
    async def validate_order(self, order: OrderRequest) -> bool  # Line 291
    async def calculate_risk_metrics(self, ...) -> RiskMetrics  # Line 355
    def update_positions(self, positions: list[Position]) -> None  # Line 405
    def check_risk_limits(self) -> tuple[bool, str]  # Line 439
    def get_position_limits(self) -> PositionLimits  # Line 483
    async def emergency_stop(self, reason: str) -> None  # Line 492
    def _update_risk_level(self, metrics: RiskMetrics) -> None  # Line 522
    def calculate_leverage(self) -> Decimal  # Line 549
    def _calculate_signal_score(self, signal: Signal) -> Decimal  # Line 581
    def _apply_portfolio_constraints(self, size: Decimal, symbol: str) -> Decimal  # Line 589
    def _check_capital_availability(self, required: Decimal, available: Decimal) -> bool  # Line 597
    async def check_portfolio_limits(self, new_position: Position) -> bool  # Line 602
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool  # Line 672
    async def get_comprehensive_risk_summary(self) -> dict[str, Any]  # Line 751
```

### File: risk_metrics.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`

#### Class: `RiskCalculator`

**Inherits**: BaseComponent
**Purpose**: Risk metrics calculator for portfolio risk assessment

```python
class RiskCalculator(BaseComponent):
    def __init__(self, config: Config)  # Line 42
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 64
    async def _create_empty_risk_metrics(self) -> RiskMetrics  # Line 143
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None  # Line 164
    async def update_position_returns(self, symbol: str, price: Decimal) -> None  # Line 185
    async def get_risk_summary(self) -> dict[str, Any]  # Line 211
```

### File: service.py

**Key Imports:**
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.service import BaseService`
- `from src.core.caching.cache_decorators import cache_risk_metrics`
- `from src.core.caching.cache_decorators import cached`
- `from src.core.exceptions import RiskManagementError`

#### Class: `RiskConfiguration`

**Inherits**: BaseModel
**Purpose**: Risk management configuration model

```python
class RiskConfiguration(BaseModel):
```

#### Class: `RiskAlert`

**Inherits**: BaseModel
**Purpose**: Risk alert model

```python
class RiskAlert(BaseModel):
```

#### Class: `RiskService`

**Inherits**: BaseService
**Purpose**: Enterprise Risk Management Service

```python
class RiskService(BaseService):
    def __init__(self, ...)  # Line 176
    async def _do_start(self) -> None  # Line 308
    async def _do_stop(self) -> None  # Line 327
    async def calculate_position_size(self, ...) -> Decimal  # Line 390
    async def _calculate_position_size_impl(self, ...) -> Decimal  # Line 422
    async def _fixed_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 485
    async def _fixed_percentage_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 502
    def _validate_kelly_data(self, winning_returns: list, losing_returns: list, returns_decimal: list) -> bool  # Line 516
    def _calculate_kelly_statistics(self, winning_returns: list, losing_returns: list, returns_decimal: list) -> tuple[Decimal, Decimal, Decimal]  # Line 542
    async def _kelly_criterion_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 574
    async def _volatility_adjusted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 646
    async def _strength_weighted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 712
    def _validate_position_sizing_inputs(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> None  # Line 740
    def _apply_position_size_limits(self, position_size: Decimal, available_capital: Decimal) -> Decimal  # Line 818
    async def _apply_portfolio_constraints(self, position_size: Decimal, symbol: str, available_capital: Decimal) -> Decimal  # Line 849
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 893
    async def _calculate_risk_metrics_impl(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 913
    def _create_empty_risk_metrics(self) -> RiskMetrics  # Line 1066
    async def _calculate_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal  # Line 1087
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None  # Line 1109
    async def _calculate_var(self, days: int, portfolio_value: Decimal) -> Decimal  # Line 1128
    async def _calculate_expected_shortfall(self, portfolio_value: Decimal, var: Decimal) -> Decimal  # Line 1210
    async def _calculate_max_drawdown(self) -> Decimal  # Line 1270
    async def _calculate_current_drawdown(self, portfolio_value: Decimal) -> Decimal  # Line 1297
    async def _calculate_sharpe_ratio(self) -> Decimal  # Line 1312
    async def _calculate_sortino_ratio(self) -> Decimal  # Line 1340
    async def _calculate_portfolio_beta(self, positions: list[Position]) -> Decimal | None  # Line 1388
    async def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal  # Line 1394
    def _determine_risk_level(self, ...) -> RiskLevel  # Line 1407
    async def get_portfolio_metrics(self) -> PortfolioMetrics  # Line 1452
    async def validate_signal(self, signal: Signal) -> bool  # Line 1472
    async def _validate_signal_impl(self, signal: Signal) -> bool  # Line 1486
    async def validate_order(self, order: OrderRequest) -> bool  # Line 1564
    async def _validate_order_impl(self, order: OrderRequest) -> bool  # Line 1578
    async def _risk_monitoring_loop(self) -> None  # Line 1631
    async def _perform_risk_check(self) -> None  # Line 1648
    async def _check_risk_alerts(self, risk_metrics: RiskMetrics) -> None  # Line 1679
    async def _check_emergency_stop_conditions(self, risk_metrics: RiskMetrics) -> None  # Line 1816
    async def trigger_emergency_stop(self, reason: str) -> None  # Line 1841
    async def reset_emergency_stop(self, reason: str) -> None  # Line 1918
    async def update_price_history(self, symbol: str, price: Decimal) -> None  # Line 1971
    async def _get_all_positions(self) -> list[Position]  # Line 2041
    async def _get_positions_for_symbol(self, symbol: str) -> list[Position]  # Line 2071
    async def _get_current_market_data(self) -> list[MarketData]  # Line 2090
    async def _load_risk_state(self) -> None  # Line 2098
    async def _save_risk_state(self) -> None  # Line 2137
    async def _save_risk_metrics(self, risk_metrics: RiskMetrics) -> None  # Line 2208
    async def _save_emergency_state(self, reason: str) -> None  # Line 2281
    async def _verify_dependencies(self) -> bool  # Line 2317
    async def _service_health_check(self) -> Any  # Line 2352
    async def _emit_state_event(self, event_type: str, event_data: dict) -> None  # Line 2382
    def _emit_validation_error(self, error: Exception, context: dict[str, Any]) -> None  # Line 2391
    async def get_risk_alerts(self, limit: int | None = None) -> list[RiskAlert]  # Line 2416
    async def acknowledge_risk_alert(self, alert_id: str) -> bool  # Line 2431
    def get_current_risk_level(self) -> RiskLevel  # Line 2450
    def is_emergency_stop_active(self) -> bool  # Line 2454
    async def save_risk_metrics(self, risk_metrics: RiskMetrics) -> None  # Line 2458
    def is_healthy(self) -> bool  # Line 2467
    async def get_risk_summary(self) -> dict[str, Any]  # Line 2497
    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]  # Line 2518
    async def get_current_risk_limits(self) -> dict[str, Any]  # Line 2566
    async def _cleanup_resources(self) -> None  # Line 2587
    async def _cleanup_stale_symbols(self) -> None  # Line 2666
    def reset_metrics(self) -> None  # Line 2692
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool  # Line 2709
    async def _should_exit_position_impl(self, position: Position, market_data: MarketData) -> bool  # Line 2724
    async def check_emergency_conditions(self, risk_metrics: RiskMetrics) -> bool  # Line 2838
    async def check_portfolio_limits(self, new_position: Position) -> bool  # Line 2902
    async def update_portfolio_state(self, positions: list[Position], available_capital: Decimal) -> None  # Line 2950
    async def risk_monitoring_context(self, operation: str) -> Any  # Line 3006
    async def _cleanup_connection_resources(self) -> None  # Line 3098
```

### File: portfolio_limits_service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.config import Config`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import Position`

#### Class: `PortfolioLimitsService`

**Inherits**: BaseService, PortfolioLimitsServiceInterface
**Purpose**: Service layer for portfolio limits enforcement

```python
class PortfolioLimitsService(BaseService, PortfolioLimitsServiceInterface):
    def __init__(self, config: Config)  # Line 24
    async def check_portfolio_limits(self, new_position: Position) -> bool  # Line 37
    async def update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None  # Line 69
    async def update_return_history(self, symbol: str, price: Decimal) -> None  # Line 110
    async def get_portfolio_summary(self) -> dict[str, Any]  # Line 144
    async def start(self) -> None  # Line 161
    async def stop(self) -> None  # Line 165
    async def health_check(self) -> bool  # Line 169
```

### File: position_sizing_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import PositionSizeMethod`
- `from src.core.types import Signal`

#### Class: `PositionSizingService`

**Inherits**: BaseService
**Purpose**: Service for calculating position sizes using various methods

```python
class PositionSizingService(BaseService):
    def __init__(self, ...)  # Line 36
    async def calculate_size(self, ...) -> Decimal  # Line 62
    async def validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool  # Line 174
    def _validate_inputs(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> None  # Line 222
    async def _fixed_percentage_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 238
    async def _kelly_criterion_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 248
    async def _volatility_adjusted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 281
    async def _confidence_weighted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 309
    def _apply_limits(self, position_size: Decimal, available_capital: Decimal) -> Decimal  # Line 321
    def _calculate_kelly_fraction(self, returns: list[Decimal]) -> Decimal  # Line 333
    def _calculate_volatility(self, prices: list[Decimal]) -> Decimal  # Line 362
    async def _get_historical_returns(self, symbol: str) -> list[Decimal]  # Line 371
    async def _get_price_history(self, symbol: str) -> list[Decimal]  # Line 381
```

### File: risk_metrics_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import RiskLevel`

#### Class: `RiskMetricsService`

**Inherits**: BaseService
**Purpose**: Service for calculating comprehensive risk metrics

```python
class RiskMetricsService(BaseService):
    def __init__(self, ...)  # Line 45
    async def calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 71
    async def get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal  # Line 252
    async def _create_empty_metrics(self) -> RiskMetrics  # Line 268
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None  # Line 288
    async def _get_portfolio_history(self) -> list[Decimal]  # Line 320
    async def _calculate_returns_from_history(self, history: list[Decimal]) -> list[Decimal]  # Line 333
    async def _calculate_var(self, days: int, portfolio_value: Decimal, history: list[Decimal]) -> Decimal  # Line 354
    async def _calculate_expected_shortfall(self, portfolio_value: Decimal, history: list[Decimal]) -> Decimal  # Line 386
    async def _calculate_max_drawdown(self, history: list[Decimal]) -> Decimal  # Line 418
    async def _calculate_current_drawdown(self, portfolio_value: Decimal, history: list[Decimal]) -> Decimal  # Line 430
    async def _calculate_sharpe_ratio(self, history: list[Decimal]) -> Decimal | None  # Line 444
    async def _calculate_total_exposure(self, positions: list[Position], market_data: list[MarketData]) -> Decimal  # Line 473
    async def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal  # Line 488
    async def _calculate_portfolio_beta(self, positions: list[Position]) -> Decimal | None  # Line 502
    async def _determine_risk_level(self, ...) -> RiskLevel  # Line 507
    async def _store_metrics(self, metrics: RiskMetrics) -> None  # Line 549
```

### File: risk_monitoring_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.types import RiskLevel`
- `from src.core.types import RiskMetrics`
- `from src.risk_management.interfaces import PortfolioRepositoryInterface`
- `from src.utils.decimal_utils import format_decimal`

#### Class: `RiskAlert`

**Purpose**: Risk alert model

```python
class RiskAlert:
    def __init__(self, ...)  # Line 36
```

#### Class: `RiskMonitoringService`

**Inherits**: BaseService
**Purpose**: Service for real-time risk monitoring and alerting

```python
class RiskMonitoringService(BaseService):
    def __init__(self, ...)  # Line 57
    async def start_monitoring(self, interval: int = 60) -> None  # Line 103
    async def stop_monitoring(self) -> None  # Line 119
    async def check_emergency_conditions(self, metrics: RiskMetrics) -> bool  # Line 146
    async def monitor_metrics(self, metrics: RiskMetrics) -> None  # Line 178
    async def get_active_alerts(self, limit: int | None = None) -> list[RiskAlert]  # Line 219
    async def acknowledge_alert(self, alert_id: str) -> bool  # Line 244
    async def set_threshold(self, threshold_name: str, value) -> None  # Line 262
    async def _monitoring_loop(self, interval: int) -> None  # Line 276
    async def _check_var_thresholds(self, metrics: RiskMetrics) -> None  # Line 308
    async def _check_drawdown_thresholds(self, metrics: RiskMetrics) -> None  # Line 336
    async def _check_sharpe_ratio(self, metrics: RiskMetrics) -> None  # Line 364
    async def _check_risk_level_changes(self, metrics: RiskMetrics) -> None  # Line 382
    async def _check_portfolio_concentration(self, metrics: RiskMetrics) -> None  # Line 414
    async def _create_alert(self, alert_type: str, severity: str, message: str, details: dict[str, Any]) -> None  # Line 424
    async def _trigger_emergency_stop(self, conditions: list[str]) -> None  # Line 445
    async def _get_latest_metrics(self) -> RiskMetrics | None  # Line 485
    async def _get_previous_risk_level(self) -> RiskLevel | None  # Line 504
    async def get_risk_summary(self) -> dict[str, Any]  # Line 517
    def _setup_message_handlers(self) -> None  # Line 566
    async def _handle_threshold_breach(self, event_data: dict) -> None  # Line 610
    async def _handle_emergency_condition(self, event_data: dict) -> None  # Line 632
    async def _handle_risk_level_change(self, event_data: dict) -> None  # Line 646
    async def publish_risk_event(self, event_type: str, event_data: dict) -> None  # Line 671
    async def _monitoring_context(self) -> AsyncIterator[None]  # Line 724
    async def _cleanup_monitoring_resources(self) -> None  # Line 746
```

### File: risk_validation_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import OrderRequest`
- `from src.core.types import Position`
- `from src.core.types import RiskLevel`

#### Class: `RiskValidationService`

**Inherits**: BaseService
**Purpose**: Service for validating trading signals and orders against risk constraints

```python
class RiskValidationService(BaseService):
    def __init__(self, ...)  # Line 26
    async def validate_signal(self, signal: Signal) -> bool  # Line 76
    async def validate_order(self, order: OrderRequest) -> bool  # Line 182
    async def validate_portfolio_limits(self, new_position: Position) -> bool  # Line 240
    def _validate_signal_structure(self, signal: Signal) -> bool  # Line 314
    def _validate_order_structure(self, order: OrderRequest) -> bool  # Line 340
    async def _validate_order_size_limits(self, order: OrderRequest) -> bool  # Line 360
    async def _validate_portfolio_exposure(self, order: OrderRequest) -> bool  # Line 394
    async def _validate_position_exposure(self, position: Position) -> bool  # Line 430
    async def _check_symbol_position_limits(self, symbol: str) -> bool  # Line 458
    async def _get_current_risk_level(self) -> RiskLevel  # Line 481
    async def _is_emergency_stop_active(self) -> bool  # Line 495
    async def _get_current_positions(self) -> list[Position]  # Line 509
    async def _get_positions_for_symbol(self, symbol: str) -> list[Position]  # Line 532
    async def _get_portfolio_value(self) -> Decimal  # Line 542
    async def _get_current_exposure(self) -> Decimal  # Line 552
    async def _get_current_price(self, symbol: str) -> Decimal  # Line 561
    def _get_max_total_positions(self) -> int  # Line 570
    def _get_max_positions_per_symbol(self) -> int  # Line 576
    def _get_max_position_size_pct(self)  # Line 582
    def _get_max_portfolio_exposure_pct(self)  # Line 590
```

---
**Generated**: Complete reference for risk_management module
**Total Classes**: 54
**Total Functions**: 13