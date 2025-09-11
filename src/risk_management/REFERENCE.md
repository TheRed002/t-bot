# RISK_MANAGEMENT Module Reference

## INTEGRATION
**Dependencies**: core, database, error_handling, monitoring, state, utils
**Used By**: None
**Provides**: AbstractRiskService, AdaptiveRiskManager, BaseRiskManager, CircuitBreakerManager, EnvironmentAwareRiskManager, PositionSizingService, RiskManagementController, RiskManager, RiskMetricsService, RiskMonitoringService, RiskService, RiskValidationService
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
**Files**: 25 Python files
**Classes**: 51
**Functions**: 13

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `AdaptiveRiskManager` ✅

**Inherits**: BaseComponent
**Purpose**: Adaptive risk management system that adjusts parameters based on market regimes
**Status**: Complete

**Implemented Methods:**
- `async calculate_adaptive_position_size(self, signal: Signal, current_regime: MarketRegime, portfolio_value: Decimal) -> Decimal` - Line 128
- `async calculate_adaptive_stop_loss(self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal` - Line 194
- `async calculate_adaptive_take_profit(self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal` - Line 281
- `async calculate_adaptive_portfolio_limits(self, current_regime: MarketRegime, base_limits: dict[str, Any]) -> dict[str, Any]` - Line 352
- `async run_stress_test(self, portfolio_positions: list[Position], scenario_name: str = 'market_crash') -> dict[str, Any]` - Line 414
- `get_adaptive_parameters(self, regime: MarketRegime) -> dict[str, Any]` - Line 596
- `get_stress_test_scenarios(self) -> list[str]` - Line 618
- `update_regime_detector(self, new_detector: 'MarketRegimeDetector') -> None` - Line 627

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
- `async cleanup(self) -> None` - Line 415

### Implementation: `BaseCircuitBreaker` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for circuit breakers
**Status**: Abstract Base Class

**Implemented Methods:**
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 114
- `async get_threshold_value(self) -> Decimal` - Line 127
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 137
- `async evaluate(self, data: dict[str, Any]) -> bool` - Line 150
- `get_status(self) -> dict[str, Any]` - Line 266
- `reset(self) -> None` - Line 276

### Implementation: `DailyLossLimitBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for daily loss limit monitoring
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 310
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 314
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 326

### Implementation: `DrawdownLimitBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for portfolio drawdown monitoring
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 349
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 353
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 368

### Implementation: `VolatilitySpikeBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for volatility spike detection
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 404
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 408
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 463

### Implementation: `ModelConfidenceBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for model confidence degradation
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 488
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 492
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 498

### Implementation: `SystemErrorRateBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for system error rate monitoring
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 528
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 532
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 592

### Implementation: `CorrelationSpikeBreaker` ✅

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for portfolio correlation spike detection
**Status**: Complete

**Implemented Methods:**
- `async get_threshold_value(self) -> Decimal` - Line 636
- `async get_current_value(self, data: dict[str, Any]) -> Decimal` - Line 640
- `async check_condition(self, data: dict[str, Any]) -> bool` - Line 669
- `get_correlation_metrics(self) -> Optional[dict[str, Any]]` - Line 743
- `async get_position_limits(self) -> dict[str, Any]` - Line 759
- `async cleanup_old_data(self, cutoff_time) -> None` - Line 770
- `reset(self) -> None` - Line 774

### Implementation: `CircuitBreakerManager` ✅

**Purpose**: Manager for all circuit breakers in the system
**Status**: Complete

**Implemented Methods:**
- `async evaluate_all(self, data: dict[str, Any]) -> dict[str, bool]` - Line 828
- `async get_status(self) -> dict[str, Any]` - Line 922
- `reset_all(self) -> None` - Line 931
- `get_triggered_breakers(self) -> list[str]` - Line 938
- `is_trading_allowed(self) -> bool` - Line 948
- `async cleanup_resources(self) -> None` - Line 975

### Implementation: `RiskManagementController` ✅

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Controller for risk management operations
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, ...) -> Decimal` - Line 66
- `async validate_signal(self, signal: Signal) -> bool` - Line 128
- `async validate_order(self, order: OrderRequest) -> bool` - Line 161
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 192
- `async validate_portfolio_limits(self, new_position: Position) -> bool` - Line 250
- `async start_monitoring(self, interval: int = 60) -> None` - Line 281
- `async stop_monitoring(self) -> None` - Line 300
- `async get_risk_summary(self) -> dict[str, Any]` - Line 314

### Implementation: `RiskCalculator` ✅

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Centralized risk calculator with caching
**Status**: Complete

**Implemented Methods:**
- `calculate_var(self, ...) -> Decimal` - Line 50
- `calculate_expected_shortfall(self, returns: list[Decimal], confidence_level: Decimal = Any) -> Decimal` - Line 69
- `calculate_sharpe_ratio(self, returns: list[Decimal], risk_free_rate: Decimal = Any) -> Decimal` - Line 84
- `calculate_sortino_ratio(self, ...) -> Decimal` - Line 100
- `calculate_max_drawdown(self, values: list[Decimal]) -> tuple[Decimal, int, int]` - Line 119
- `calculate_calmar_ratio(self, returns: list[Decimal], period_years: Decimal = Any) -> Decimal` - Line 131
- `async calculate_portfolio_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 146
- `clear_cache(self) -> None` - Line 286
- `update_history(self, symbol: str, price: Decimal, return_value: Optional[float] = None) -> None` - Line 291

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
- `transform_signal_to_event_data(signal: Signal, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]` - Line 23
- `transform_position_to_event_data(position: Position, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]` - Line 49
- `transform_risk_metrics_to_event_data(risk_metrics: RiskMetrics, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]` - Line 79
- `transform_error_to_event_data(error, ...) -> dict[str, Any]` - Line 109
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 136
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'risk_management') -> dict[str, Any]` - Line 168
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]` - Line 202
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: Optional[str] = None) -> dict[str, Any]` - Line 250
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 278
- `apply_cross_module_validation(cls, ...) -> dict[str, Any]` - Line 332

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
- `get_active_trigger(self) -> Optional[str]` - Line 733
- `async cleanup_resources(self) -> None` - Line 745

### Implementation: `EnvironmentAwareRiskConfiguration` ✅

**Purpose**: Environment-specific risk configuration
**Status**: Complete

**Implemented Methods:**
- `get_sandbox_risk_config() -> dict[str, Any]` - Line 27
- `get_live_risk_config() -> dict[str, Any]` - Line 47

### Implementation: `EnvironmentAwareRiskManager` ✅

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware risk management functionality
**Status**: Complete

**Implemented Methods:**
- `get_environment_risk_config(self, exchange: str) -> dict[str, Any]` - Line 102
- `calculate_environment_aware_position_size(self, ...) -> Decimal` - Line 115
- `async validate_environment_order(self, ...) -> bool` - Line 183
- `async update_environment_risk_state(self, ...) -> None` - Line 335
- `async reset_environment_risk_state(self, exchange: str) -> None` - Line 384
- `get_environment_risk_metrics(self, exchange: str) -> dict[str, Any]` - Line 397

### Implementation: `RiskManagementFactory` ✅

**Inherits**: RiskManagementFactoryInterface
**Purpose**: Factory for creating risk management components
**Status**: Complete

**Implemented Methods:**
- `create_risk_service(self, correlation_id: str | None = None) -> RiskServiceInterface` - Line 69
- `create_legacy_risk_manager(self) -> RiskManager` - Line 101
- `create_legacy_position_sizer(self) -> PositionSizer` - Line 128
- `create_legacy_risk_calculator(self) -> RiskCalculator` - Line 153
- `create_risk_management_controller(self, correlation_id: str | None = None) -> RiskManagementController` - Line 178
- `get_recommended_component(self) -> RiskService | RiskManager` - Line 226
- `validate_dependencies(self) -> dict[str, bool]` - Line 259
- `async start_services(self) -> None` - Line 293
- `async stop_services(self) -> None` - Line 303
- `get_migration_guide(self) -> dict[str, str]` - Line 313

### Implementation: `CacheServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for cache service implementations
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> Any` - Line 25
- `async set(self, key: str, value: Any, ttl: int | None = None) -> None` - Line 29
- `async delete(self, key: str) -> None` - Line 33
- `async clear(self) -> None` - Line 37
- `async close(self) -> None` - Line 41

### Implementation: `ExchangeServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for exchange service implementations to avoid direct coupling
**Status**: Complete

**Implemented Methods:**
- `async cancel_all_orders(self, symbol: str | None = None) -> int` - Line 49
- `async close_all_positions(self) -> int` - Line 53
- `async get_account_balance(self) -> Decimal` - Line 57

### Implementation: `RiskServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk management service implementations
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, ...) -> Decimal` - Line 65
- `async validate_signal(self, signal: Signal) -> bool` - Line 75
- `async validate_order(self, order: OrderRequest) -> bool` - Line 79
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 83
- `async should_exit_position(self, position: Position, market_data: MarketData) -> bool` - Line 89
- `get_current_risk_level(self) -> RiskLevel` - Line 93
- `is_emergency_stop_active(self) -> bool` - Line 97
- `async get_risk_summary(self) -> dict[str, Any]` - Line 101

### Implementation: `PositionSizingServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for position sizing service implementations
**Status**: Complete

**Implemented Methods:**
- `async calculate_size(self, ...) -> Decimal` - Line 109
- `async validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool` - Line 119

### Implementation: `RiskMetricsServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk metrics service implementations
**Status**: Complete

**Implemented Methods:**
- `async calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 127
- `async get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal` - Line 133

### Implementation: `RiskValidationServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk validation service implementations
**Status**: Complete

**Implemented Methods:**
- `async validate_signal(self, signal: Signal) -> bool` - Line 143
- `async validate_order(self, order: OrderRequest) -> bool` - Line 147
- `async validate_portfolio_limits(self, new_position: Position) -> bool` - Line 151

### Implementation: `RiskMonitoringServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for risk monitoring service implementations
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self, interval: int = 60) -> None` - Line 159
- `async stop_monitoring(self) -> None` - Line 163
- `async check_emergency_conditions(self, metrics: RiskMetrics) -> bool` - Line 167
- `async get_risk_summary(self) -> dict[str, Any]` - Line 171

### Implementation: `AbstractRiskService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for risk services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async start(self) -> None` - Line 180
- `async stop(self) -> None` - Line 185
- `async calculate_position_size(self, ...) -> Decimal` - Line 190
- `async validate_signal(self, signal: Signal) -> bool` - Line 201
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 206

### Implementation: `RiskManagementFactoryInterface` 🔧

**Inherits**: ABC
**Purpose**: Abstract interface for risk management service factories
**Status**: Abstract Base Class

**Implemented Methods:**
- `create_risk_service(self, correlation_id: str | None = None) -> 'RiskServiceInterface'` - Line 217
- `create_risk_management_controller(self, correlation_id: str | None = None) -> Any` - Line 222
- `validate_dependencies(self) -> dict[str, bool]` - Line 227

### Implementation: `PortfolioLimits` ✅

**Inherits**: BaseComponent
**Purpose**: Portfolio limits enforcer for risk management
**Status**: Complete

**Implemented Methods:**
- `async check_portfolio_limits(self, new_position: Position) -> bool` - Line 91
- `async update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None` - Line 449
- `async update_return_history(self, symbol: str, price: Decimal) -> None` - Line 469
- `async get_portfolio_summary(self) -> dict[str, Any]` - Line 508

### Implementation: `PositionSizer` ✅

**Inherits**: BaseComponent
**Purpose**: Position sizing calculator with multiple algorithms
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, ...) -> Decimal` - Line 68
- `async update_price_history(self, symbol: str, price: Decimal) -> None` - Line 132
- `async get_position_size_summary(self, signal: Signal, portfolio_value: Decimal) -> dict[str, Any]` - Line 172
- `async validate_position_size(self, position_size: Decimal, portfolio_value: Decimal) -> bool` - Line 206
- `get_signal_confidence(self, signal: Signal) -> Decimal` - Line 226

### Implementation: `MarketRegimeDetector` ✅

**Inherits**: BaseComponent
**Purpose**: Market regime detection and classification system
**Status**: Complete

**Implemented Methods:**
- `async detect_volatility_regime(self, symbol: str, price_data: list[float]) -> MarketRegime` - Line 71
- `async detect_trend_regime(self, symbol: str, price_data: list[float]) -> MarketRegime` - Line 118
- `async detect_correlation_regime(self, symbols: list[str], price_data_dict: dict[str, list[float]]) -> MarketRegime` - Line 173
- `async detect_comprehensive_regime(self, market_data: list[MarketData]) -> MarketRegime` - Line 240
- `get_regime_history(self, limit: int = 10) -> list[RegimeChangeEvent]` - Line 421
- `get_current_regime(self) -> MarketRegime` - Line 433
- `get_regime_statistics(self) -> dict[str, Any]` - Line 442

### Implementation: `RiskManager` ✅

**Inherits**: BaseRiskManager
**Purpose**: Legacy Risk Manager implementation for backward compatibility
**Status**: Complete

**Implemented Methods:**
- `async calculate_position_size(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> Decimal` - Line 149
- `async validate_signal(self, signal: Signal) -> bool` - Line 220
- `async validate_order(self, order: OrderRequest) -> bool` - Line 285
- `async calculate_risk_metrics(self, ...) -> RiskMetrics` - Line 350
- `update_positions(self, positions: list[Position]) -> None` - Line 400
- `check_risk_limits(self) -> tuple[bool, str]` - Line 434
- `get_position_limits(self) -> PositionLimits` - Line 479
- `async emergency_stop(self, reason: str) -> None` - Line 488
- `calculate_leverage(self) -> Decimal` - Line 545
- `async check_portfolio_limits(self, new_position: Position) -> bool` - Line 596
- `async should_exit_position(self, position: Position, market_data: MarketData) -> bool` - Line 667
- `async get_comprehensive_risk_summary(self) -> dict[str, Any]` - Line 746

### Implementation: `RiskCalculator` ✅

**Inherits**: BaseComponent
**Purpose**: Risk metrics calculator for portfolio risk assessment
**Status**: Complete

**Implemented Methods:**
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 75
- `async update_position_returns(self, symbol: str, price: Decimal) -> None` - Line 191
- `async get_risk_summary(self) -> dict[str, Any]` - Line 217

### Implementation: `RiskConfiguration` ✅

**Inherits**: BaseModel
**Purpose**: Risk management configuration model
**Status**: Complete

### Implementation: `PortfolioMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Portfolio metrics model for caching
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
- `async calculate_position_size(self, ...) -> Decimal` - Line 400
- `async calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 864
- `async get_portfolio_metrics(self) -> PortfolioMetrics` - Line 1345
- `async validate_signal(self, signal: Signal) -> bool` - Line 1365
- `async validate_order(self, order: OrderRequest) -> bool` - Line 1446
- `async trigger_emergency_stop(self, reason: str) -> None` - Line 1723
- `async reset_emergency_stop(self, reason: str) -> None` - Line 1799
- `async update_price_history(self, symbol: str, price: Decimal) -> None` - Line 1851
- `async get_risk_alerts(self, limit: int | None = None) -> list[RiskAlert]` - Line 2243
- `async acknowledge_risk_alert(self, alert_id: str) -> bool` - Line 2258
- `get_current_risk_level(self) -> RiskLevel` - Line 2277
- `is_emergency_stop_active(self) -> bool` - Line 2281
- `async get_risk_summary(self) -> dict[str, Any]` - Line 2285
- `reset_metrics(self) -> None` - Line 2411
- `async should_exit_position(self, position: Position, market_data: MarketData) -> bool` - Line 2428
- `async risk_monitoring_context(self, operation: str) -> Any` - Line 2556

### Implementation: `PositionSizingService` ✅

**Inherits**: BaseService
**Purpose**: Service for calculating position sizes using various methods
**Status**: Complete

**Implemented Methods:**
- `async calculate_size(self, ...) -> Decimal` - Line 55
- `async validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool` - Line 179

### Implementation: `RiskMetricsService` ✅

**Inherits**: BaseService
**Purpose**: Service for calculating comprehensive risk metrics
**Status**: Complete

**Implemented Methods:**
- `async calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics` - Line 64
- `async get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal` - Line 245

### Implementation: `RiskAlert` ✅

**Purpose**: Risk alert model
**Status**: Complete

**Implemented Methods:**

### Implementation: `RiskMonitoringService` ✅

**Inherits**: BaseService
**Purpose**: Service for real-time risk monitoring and alerting
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self, interval: int = 60) -> None` - Line 102
- `async stop_monitoring(self) -> None` - Line 118
- `async check_emergency_conditions(self, metrics: RiskMetrics) -> bool` - Line 145
- `async monitor_metrics(self, metrics: RiskMetrics) -> None` - Line 177
- `async get_active_alerts(self, limit: int | None = None) -> list[RiskAlert]` - Line 218
- `async acknowledge_alert(self, alert_id: str) -> bool` - Line 243
- `async set_threshold(self, threshold_name: str, value) -> None` - Line 261
- `async get_risk_summary(self) -> dict[str, Any]` - Line 509
- `async publish_risk_event(self, event_type: str, event_data: dict) -> None` - Line 659

### Implementation: `RiskValidationService` ✅

**Inherits**: BaseService
**Purpose**: Service for validating trading signals and orders against risk constraints
**Status**: Complete

**Implemented Methods:**
- `async validate_signal(self, signal: Signal) -> bool` - Line 75
- `async validate_order(self, order: OrderRequest) -> bool` - Line 183
- `async validate_portfolio_limits(self, new_position: Position) -> bool` - Line 241

## COMPLETE API REFERENCE

### File: adaptive_risk.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketRegime`
- `from src.core.types import Position`

#### Class: `AdaptiveRiskManager`

**Inherits**: BaseComponent
**Purpose**: Adaptive risk management system that adjusts parameters based on market regimes

```python
class AdaptiveRiskManager(BaseComponent):
    def __init__(self, config: dict[str, Any], regime_detector: 'MarketRegimeDetector')  # Line 34
    async def calculate_adaptive_position_size(self, signal: Signal, current_regime: MarketRegime, portfolio_value: Decimal) -> Decimal  # Line 128
    async def calculate_adaptive_stop_loss(self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal  # Line 194
    async def calculate_adaptive_take_profit(self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal  # Line 281
    async def calculate_adaptive_portfolio_limits(self, current_regime: MarketRegime, base_limits: dict[str, Any]) -> dict[str, Any]  # Line 352
    async def run_stress_test(self, portfolio_positions: list[Position], scenario_name: str = 'market_crash') -> dict[str, Any]  # Line 414
    async def _get_correlation_regime(self) -> Optional[str]  # Line 510
    async def _calculate_momentum_adjustment(self, symbol: str) -> Decimal  # Line 539
    def get_adaptive_parameters(self, regime: MarketRegime) -> dict[str, Any]  # Line 596
    def get_stress_test_scenarios(self) -> list[str]  # Line 618
    def update_regime_detector(self, new_detector: 'MarketRegimeDetector') -> None  # Line 627
```

### File: base.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
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
    def _determine_violation_severity(self, violation_type: str, details: dict[str, Any]) -> str  # Line 391
    async def cleanup(self) -> None  # Line 415
```

### File: circuit_breakers.py

**Key Imports:**
- `from src.core.config.main import Config`
- `from src.core.exceptions import CircuitBreakerTriggeredError`
- `from src.core.logging import get_logger`
- `from src.core.types import CircuitBreakerEvent`
- `from src.core.types import CircuitBreakerStatus`

#### Class: `BaseCircuitBreaker`

**Inherits**: ABC
**Purpose**: Abstract base class for circuit breakers

```python
class BaseCircuitBreaker(ABC):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 80
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 114
    async def get_threshold_value(self) -> Decimal  # Line 127
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 137
    async def evaluate(self, data: dict[str, Any]) -> bool  # Line 150
    async def _trigger_circuit_breaker(self, data: dict[str, Any]) -> None  # Line 199
    async def _close_circuit_breaker(self) -> None  # Line 255
    def get_status(self) -> dict[str, Any]  # Line 266
    def reset(self) -> None  # Line 276
```

#### Class: `DailyLossLimitBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for daily loss limit monitoring

```python
class DailyLossLimitBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 297
    async def get_threshold_value(self) -> Decimal  # Line 310
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 314
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 326
```

#### Class: `DrawdownLimitBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for portfolio drawdown monitoring

```python
class DrawdownLimitBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 344
    async def get_threshold_value(self) -> Decimal  # Line 349
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 353
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 368
```

#### Class: `VolatilitySpikeBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for volatility spike detection

```python
class VolatilitySpikeBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 386
    async def get_threshold_value(self) -> Decimal  # Line 404
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 408
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 463
```

#### Class: `ModelConfidenceBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for model confidence degradation

```python
class ModelConfidenceBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 481
    async def get_threshold_value(self) -> Decimal  # Line 488
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 492
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 498
```

#### Class: `SystemErrorRateBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for system error rate monitoring

```python
class SystemErrorRateBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 516
    async def get_threshold_value(self) -> Decimal  # Line 528
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 532
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 592
```

#### Class: `CorrelationSpikeBreaker`

**Inherits**: BaseCircuitBreaker
**Purpose**: Circuit breaker for portfolio correlation spike detection

```python
class CorrelationSpikeBreaker(BaseCircuitBreaker):
    def __init__(self, config: Config, risk_manager: 'BaseRiskManager')  # Line 611
    async def get_threshold_value(self) -> Decimal  # Line 636
    async def get_current_value(self, data: dict[str, Any]) -> Decimal  # Line 640
    async def check_condition(self, data: dict[str, Any]) -> bool  # Line 669
    def get_correlation_metrics(self) -> Optional[dict[str, Any]]  # Line 743
    async def get_position_limits(self) -> dict[str, Any]  # Line 759
    async def cleanup_old_data(self, cutoff_time) -> None  # Line 770
    def reset(self) -> None  # Line 774
```

#### Class: `CircuitBreakerManager`

**Purpose**: Manager for all circuit breakers in the system

```python
class CircuitBreakerManager:
    def __init__(self, ...)  # Line 790
    async def evaluate_all(self, data: dict[str, Any]) -> dict[str, bool]  # Line 828
    async def get_status(self) -> dict[str, Any]  # Line 922
    def reset_all(self) -> None  # Line 931
    def get_triggered_breakers(self) -> list[str]  # Line 938
    def is_trading_allowed(self) -> bool  # Line 948
    async def _evaluation_context(self, breaker_name: str) -> AsyncIterator[None]  # Line 953
    async def cleanup_resources(self) -> None  # Line 975
```

### File: controller.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.types import MarketData`
- `from src.core.types import OrderRequest`
- `from src.core.types import Position`
- `from src.core.types import RiskMetrics`

#### Class: `RiskManagementController`

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Controller for risk management operations

```python
class RiskManagementController(BaseComponent, ErrorPropagationMixin):
    def __init__(self, ...)  # Line 36
    async def calculate_position_size(self, ...) -> Decimal  # Line 66
    async def validate_signal(self, signal: Signal) -> bool  # Line 128
    async def validate_order(self, order: OrderRequest) -> bool  # Line 161
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 192
    async def validate_portfolio_limits(self, new_position: Position) -> bool  # Line 250
    async def start_monitoring(self, interval: int = 60) -> None  # Line 281
    async def stop_monitoring(self) -> None  # Line 300
    async def get_risk_summary(self) -> dict[str, Any]  # Line 314
```

### File: calculator.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.types import Position`
- `from src.core.types.market import MarketData`
- `from src.core.types.risk import RiskLevel`
- `from src.core.types.risk import RiskMetrics`

#### Class: `RiskCalculator`

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Centralized risk calculator with caching

```python
class RiskCalculator(BaseComponent, ErrorPropagationMixin):
    def __init__(self)  # Line 40
    def calculate_var(self, ...) -> Decimal  # Line 50
    def calculate_expected_shortfall(self, returns: list[Decimal], confidence_level: Decimal = Any) -> Decimal  # Line 69
    def calculate_sharpe_ratio(self, returns: list[Decimal], risk_free_rate: Decimal = Any) -> Decimal  # Line 84
    def calculate_sortino_ratio(self, ...) -> Decimal  # Line 100
    def calculate_max_drawdown(self, values: list[Decimal]) -> tuple[Decimal, int, int]  # Line 119
    def calculate_calmar_ratio(self, returns: list[Decimal], period_years: Decimal = Any) -> Decimal  # Line 131
    async def calculate_portfolio_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 146
    def _calculate_current_drawdown(self) -> Decimal  # Line 213
    def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal  # Line 221
    def _determine_risk_level(self, var: Decimal, max_dd: Decimal, sharpe: Decimal) -> RiskLevel  # Line 271
    def clear_cache(self) -> None  # Line 286
    def update_history(self, symbol: str, price: Decimal, return_value: Optional[float] = None) -> None  # Line 291
```

#### Functions:

```python
def get_risk_calculator() -> RiskCalculator  # Line 376
```

### File: monitor.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.types.risk import RiskAlert`
- `from src.core.types.risk import RiskMetrics`
- `from src.utils.decorators import UnifiedDecorator`
- `from src.utils.decimal_utils import format_decimal`

#### Class: `RiskMonitor`

**Inherits**: BaseComponent, ErrorPropagationMixin
**Purpose**: Legacy risk monitor that delegates to centralized utilities

```python
class RiskMonitor(BaseComponent, ErrorPropagationMixin):
    def __init__(self, messaging_coordinator: Optional[MessagingCoordinator] = None) -> None  # Line 29
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
- `from src.core.base.component import BaseComponent`
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
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
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
- `from src.core.exceptions import DataValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import Position`
- `from src.core.types import RiskMetrics`
- `from src.core.types import Signal`

#### Class: `RiskDataTransformer`

**Purpose**: Handles consistent data transformation for risk_management module

```python
class RiskDataTransformer:
    def transform_signal_to_event_data(signal: Signal, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]  # Line 23
    def transform_position_to_event_data(position: Position, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]  # Line 49
    def transform_risk_metrics_to_event_data(risk_metrics: RiskMetrics, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]  # Line 79
    def transform_error_to_event_data(error, ...) -> dict[str, Any]  # Line 109
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 136
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'risk_management') -> dict[str, Any]  # Line 168
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]  # Line 202
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: Optional[str] = None) -> dict[str, Any]  # Line 250
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 278
    def apply_cross_module_validation(cls, ...) -> dict[str, Any]  # Line 332
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def register_risk_management_services(injector: DependencyInjector) -> None  # Line 30
def configure_risk_management_dependencies(injector: DependencyInjector | None = None) -> DependencyInjector  # Line 180
def get_risk_service(injector: DependencyInjector) -> 'RiskService'  # Line 203
def get_position_sizing_service(injector: DependencyInjector) -> 'PositionSizingService'  # Line 208
def get_risk_validation_service(injector: DependencyInjector) -> 'RiskValidationService'  # Line 213
def get_risk_metrics_service(injector: DependencyInjector) -> 'RiskMetricsService'  # Line 218
def get_risk_monitoring_service(injector: DependencyInjector) -> 'RiskMonitoringService'  # Line 223
def get_risk_management_factory(injector: DependencyInjector) -> 'RiskManagementFactoryInterface'  # Line 228
```

### File: emergency_controls.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
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
    def get_active_trigger(self) -> Optional[str]  # Line 733
    async def cleanup_resources(self) -> None  # Line 745
```

### File: environment_integration.py

**Key Imports:**
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ConfigurationError`
- `from src.core.integration.environment_aware_service import EnvironmentAwareServiceMixin`
- `from src.core.integration.environment_aware_service import EnvironmentContext`
- `from src.core.logging import get_logger`

#### Class: `EnvironmentAwareRiskConfiguration`

**Purpose**: Environment-specific risk configuration

```python
class EnvironmentAwareRiskConfiguration:
    def get_sandbox_risk_config() -> dict[str, Any]  # Line 27
    def get_live_risk_config() -> dict[str, Any]  # Line 47
```

#### Class: `EnvironmentAwareRiskManager`

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware risk management functionality

```python
class EnvironmentAwareRiskManager(EnvironmentAwareServiceMixin):
    def __init__(self, *args, **kwargs)  # Line 74
    async def _update_service_environment(self, context: EnvironmentContext) -> None  # Line 79
    def get_environment_risk_config(self, exchange: str) -> dict[str, Any]  # Line 102
    def calculate_environment_aware_position_size(self, ...) -> Decimal  # Line 115
    async def validate_environment_order(self, ...) -> bool  # Line 183
    async def _validate_production_order(self, order_request: OrderRequest, exchange: str, risk_config: dict[str, Any]) -> bool  # Line 216
    async def _validate_sandbox_order(self, order_request: OrderRequest, exchange: str, risk_config: dict[str, Any]) -> bool  # Line 246
    async def _validate_common_risk_rules(self, ...) -> bool  # Line 266
    def _calculate_volatility_adjustment(self, market_data: Any, risk_config: dict[str, Any]) -> Decimal  # Line 294
    async def _detect_suspicious_order_pattern(self, order_request: OrderRequest, exchange: str) -> bool  # Line 320
    async def update_environment_risk_state(self, ...) -> None  # Line 335
    async def _notify_circuit_breaker_triggered(self, exchange: str, loss_amount: Decimal) -> None  # Line 371
    async def reset_environment_risk_state(self, exchange: str) -> None  # Line 384
    def get_environment_risk_metrics(self, exchange: str) -> dict[str, Any]  # Line 397
```

### File: factory.py

**Key Imports:**
- `from src.core.config.main import Config`
- `from src.core.config.main import get_config`
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.dependency_injection import get_container`
- `from src.core.exceptions import DependencyError`

#### Class: `RiskManagementFactory`

**Inherits**: RiskManagementFactoryInterface
**Purpose**: Factory for creating risk management components

```python
class RiskManagementFactory(RiskManagementFactoryInterface):
    def __init__(self, injector: DependencyInjector | None = None)  # Line 45
    def create_risk_service(self, correlation_id: str | None = None) -> RiskServiceInterface  # Line 69
    def create_legacy_risk_manager(self) -> RiskManager  # Line 101
    def create_legacy_position_sizer(self) -> PositionSizer  # Line 128
    def create_legacy_risk_calculator(self) -> RiskCalculator  # Line 153
    def create_risk_management_controller(self, correlation_id: str | None = None) -> RiskManagementController  # Line 178
    def get_recommended_component(self) -> RiskService | RiskManager  # Line 226
    def validate_dependencies(self) -> dict[str, bool]  # Line 259
    async def start_services(self) -> None  # Line 293
    async def stop_services(self) -> None  # Line 303
    def get_migration_guide(self) -> dict[str, str]  # Line 313
```

#### Functions:

```python
def get_risk_factory(injector: DependencyInjector | None = None) -> RiskManagementFactory  # Line 353
def create_risk_service(injector: DependencyInjector, correlation_id: str | None = None) -> RiskServiceInterface  # Line 377
def create_recommended_risk_component(injector: DependencyInjector) -> RiskService | RiskManager  # Line 395
def create_risk_management_controller(injector: DependencyInjector, correlation_id: str | None = None) -> RiskManagementController  # Line 409
```

### File: interfaces.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import OrderRequest`
- `from src.core.types import Position`
- `from src.core.types import RiskLevel`
- `from src.core.types import RiskMetrics`

#### Class: `CacheServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for cache service implementations

```python
class CacheServiceInterface(Protocol):
    async def get(self, key: str) -> Any  # Line 25
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None  # Line 29
    async def delete(self, key: str) -> None  # Line 33
    async def clear(self) -> None  # Line 37
    async def close(self) -> None  # Line 41
```

#### Class: `ExchangeServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for exchange service implementations to avoid direct coupling

```python
class ExchangeServiceInterface(Protocol):
    async def cancel_all_orders(self, symbol: str | None = None) -> int  # Line 49
    async def close_all_positions(self) -> int  # Line 53
    async def get_account_balance(self) -> Decimal  # Line 57
```

#### Class: `RiskServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk management service implementations

```python
class RiskServiceInterface(Protocol):
    async def calculate_position_size(self, ...) -> Decimal  # Line 65
    async def validate_signal(self, signal: Signal) -> bool  # Line 75
    async def validate_order(self, order: OrderRequest) -> bool  # Line 79
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 83
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool  # Line 89
    def get_current_risk_level(self) -> RiskLevel  # Line 93
    def is_emergency_stop_active(self) -> bool  # Line 97
    async def get_risk_summary(self) -> dict[str, Any]  # Line 101
```

#### Class: `PositionSizingServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for position sizing service implementations

```python
class PositionSizingServiceInterface(Protocol):
    async def calculate_size(self, ...) -> Decimal  # Line 109
    async def validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool  # Line 119
```

#### Class: `RiskMetricsServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk metrics service implementations

```python
class RiskMetricsServiceInterface(Protocol):
    async def calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 127
    async def get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal  # Line 133
```

#### Class: `RiskValidationServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk validation service implementations

```python
class RiskValidationServiceInterface(Protocol):
    async def validate_signal(self, signal: Signal) -> bool  # Line 143
    async def validate_order(self, order: OrderRequest) -> bool  # Line 147
    async def validate_portfolio_limits(self, new_position: Position) -> bool  # Line 151
```

#### Class: `RiskMonitoringServiceInterface`

**Inherits**: Protocol
**Purpose**: Protocol for risk monitoring service implementations

```python
class RiskMonitoringServiceInterface(Protocol):
    async def start_monitoring(self, interval: int = 60) -> None  # Line 159
    async def stop_monitoring(self) -> None  # Line 163
    async def check_emergency_conditions(self, metrics: RiskMetrics) -> bool  # Line 167
    async def get_risk_summary(self) -> dict[str, Any]  # Line 171
```

#### Class: `AbstractRiskService`

**Inherits**: ABC
**Purpose**: Abstract base class for risk services

```python
class AbstractRiskService(ABC):
    async def start(self) -> None  # Line 180
    async def stop(self) -> None  # Line 185
    async def calculate_position_size(self, ...) -> Decimal  # Line 190
    async def validate_signal(self, signal: Signal) -> bool  # Line 201
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 206
```

#### Class: `RiskManagementFactoryInterface`

**Inherits**: ABC
**Purpose**: Abstract interface for risk management service factories

```python
class RiskManagementFactoryInterface(ABC):
    def create_risk_service(self, correlation_id: str | None = None) -> 'RiskServiceInterface'  # Line 217
    def create_risk_management_controller(self, correlation_id: str | None = None) -> Any  # Line 222
    def validate_dependencies(self) -> dict[str, bool]  # Line 227
```

### File: portfolio_limits.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
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
    def _check_leverage_limit(self, new_position: Position) -> bool  # Line 392
    def _get_correlation(self, symbol1: str, symbol2: str) -> Decimal  # Line 412
    async def update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None  # Line 449
    async def update_return_history(self, symbol: str, price: Decimal) -> None  # Line 469
    async def get_portfolio_summary(self) -> dict[str, Any]  # Line 508
    async def _log_risk_violation(self, violation_type: str, details: dict[str, Any]) -> None  # Line 564
    def _log_risk_violation_sync(self, violation_type: str, details: dict[str, Any]) -> None  # Line 577
```

### File: position_sizing.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import PositionSizeMethod`

#### Class: `PositionSizer`

**Inherits**: BaseComponent
**Purpose**: Position sizing calculator with multiple algorithms

```python
class PositionSizer(BaseComponent):
    def __init__(self, config: Config, database_service: 'DatabaseService | None' = None)  # Line 39
    async def calculate_position_size(self, ...) -> Decimal  # Line 68
    async def update_price_history(self, symbol: str, price: Decimal) -> None  # Line 132
    async def get_position_size_summary(self, signal: Signal, portfolio_value: Decimal) -> dict[str, Any]  # Line 172
    async def validate_position_size(self, position_size: Decimal, portfolio_value: Decimal) -> bool  # Line 206
    def get_signal_confidence(self, signal: Signal) -> Decimal  # Line 226
```

### File: regime_detection.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.types import MarketData`
- `from src.core.types import MarketRegime`
- `from src.core.types import RegimeChangeEvent`

#### Class: `MarketRegimeDetector`

**Inherits**: BaseComponent
**Purpose**: Market regime detection and classification system

```python
class MarketRegimeDetector(BaseComponent):
    def __init__(self, config: dict[str, Any])  # Line 35
    async def detect_volatility_regime(self, symbol: str, price_data: list[float]) -> MarketRegime  # Line 71
    async def detect_trend_regime(self, symbol: str, price_data: list[float]) -> MarketRegime  # Line 118
    async def detect_correlation_regime(self, symbols: list[str], price_data_dict: dict[str, list[float]]) -> MarketRegime  # Line 173
    async def detect_comprehensive_regime(self, market_data: list[MarketData]) -> MarketRegime  # Line 240
    def _combine_regimes(self, ...) -> MarketRegime  # Line 292
    def _check_regime_change(self, new_regime: MarketRegime) -> None  # Line 350
    def _calculate_change_confidence(self, new_regime: MarketRegime) -> float  # Line 392
    def get_regime_history(self, limit: int = 10) -> list[RegimeChangeEvent]  # Line 421
    def get_current_regime(self) -> MarketRegime  # Line 433
    def get_regime_statistics(self) -> dict[str, Any]  # Line 442
```

### File: risk_manager.py

**Key Imports:**
- `from src.core.config.main import Config`
- `from src.core.config.main import get_config`
- `from src.core.dependency_injection import injectable`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.logging import get_logger`

#### Class: `RiskManager`

**Inherits**: BaseRiskManager
**Purpose**: Legacy Risk Manager implementation for backward compatibility

```python
class RiskManager(BaseRiskManager):
    def __init__(self, ...)  # Line 79
    async def calculate_position_size(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> Decimal  # Line 149
    async def validate_signal(self, signal: Signal) -> bool  # Line 220
    async def validate_order(self, order: OrderRequest) -> bool  # Line 285
    async def calculate_risk_metrics(self, ...) -> RiskMetrics  # Line 350
    def update_positions(self, positions: list[Position]) -> None  # Line 400
    def check_risk_limits(self) -> tuple[bool, str]  # Line 434
    def get_position_limits(self) -> PositionLimits  # Line 479
    async def emergency_stop(self, reason: str) -> None  # Line 488
    def _update_risk_level(self, metrics: RiskMetrics) -> None  # Line 518
    def calculate_leverage(self) -> Decimal  # Line 545
    def _calculate_signal_score(self, signal: Signal) -> Decimal  # Line 575
    def _apply_portfolio_constraints(self, size: Decimal, symbol: str) -> Decimal  # Line 583
    def _check_capital_availability(self, required: Decimal, available: Decimal) -> bool  # Line 591
    async def check_portfolio_limits(self, new_position: Position) -> bool  # Line 596
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool  # Line 667
    async def get_comprehensive_risk_summary(self) -> dict[str, Any]  # Line 746
```

### File: risk_metrics.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.config.main import Config`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`

#### Class: `RiskCalculator`

**Inherits**: BaseComponent
**Purpose**: Risk metrics calculator for portfolio risk assessment

```python
class RiskCalculator(BaseComponent):
    def __init__(self, config: Config, database_service: 'DatabaseService | None' = None)  # Line 44
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 75
    async def _create_empty_risk_metrics(self) -> RiskMetrics  # Line 149
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None  # Line 170
    async def update_position_returns(self, symbol: str, price: Decimal) -> None  # Line 191
    async def get_risk_summary(self) -> dict[str, Any]  # Line 217
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

#### Class: `PortfolioMetrics`

**Inherits**: BaseModel
**Purpose**: Portfolio metrics model for caching

```python
class PortfolioMetrics(BaseModel):
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
    def __init__(self, ...)  # Line 199
    async def _do_start(self) -> None  # Line 318
    async def _do_stop(self) -> None  # Line 337
    async def calculate_position_size(self, ...) -> Decimal  # Line 400
    async def _calculate_position_size_impl(self, ...) -> Decimal  # Line 432
    async def _fixed_percentage_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 488
    def _validate_kelly_data(self, winning_returns: list, losing_returns: list, returns_decimal: list) -> bool  # Line 502
    def _calculate_kelly_statistics(self, winning_returns: list, losing_returns: list, returns_decimal: list) -> tuple[Decimal, Decimal, Decimal]  # Line 528
    async def _kelly_criterion_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 560
    async def _volatility_adjusted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 632
    async def _strength_weighted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 694
    def _validate_position_sizing_inputs(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> None  # Line 716
    def _apply_position_size_limits(self, position_size: Decimal, available_capital: Decimal) -> Decimal  # Line 789
    async def _apply_portfolio_constraints(self, position_size: Decimal, symbol: str, available_capital: Decimal) -> Decimal  # Line 820
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 864
    async def _calculate_risk_metrics_impl(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 884
    def _create_empty_risk_metrics(self) -> RiskMetrics  # Line 1041
    async def _calculate_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal  # Line 1060
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None  # Line 1082
    async def _calculate_var(self, days: int, portfolio_value: Decimal) -> Decimal  # Line 1101
    async def _calculate_expected_shortfall(self, portfolio_value: Decimal) -> Decimal  # Line 1180
    async def _calculate_max_drawdown(self) -> Decimal  # Line 1213
    async def _calculate_current_drawdown(self, portfolio_value: Decimal) -> Decimal  # Line 1240
    async def _calculate_sharpe_ratio(self) -> Decimal | None  # Line 1255
    async def _calculate_portfolio_beta(self, positions: list[Position]) -> Decimal | None  # Line 1281
    async def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal  # Line 1287
    def _determine_risk_level(self, ...) -> RiskLevel  # Line 1300
    async def get_portfolio_metrics(self) -> PortfolioMetrics  # Line 1345
    async def validate_signal(self, signal: Signal) -> bool  # Line 1365
    async def _validate_signal_impl(self, signal: Signal) -> bool  # Line 1379
    async def validate_order(self, order: OrderRequest) -> bool  # Line 1446
    async def _validate_order_impl(self, order: OrderRequest) -> bool  # Line 1460
    async def _risk_monitoring_loop(self) -> None  # Line 1513
    async def _perform_risk_check(self) -> None  # Line 1530
    async def _check_risk_alerts(self, risk_metrics: RiskMetrics) -> None  # Line 1561
    async def _check_emergency_stop_conditions(self, risk_metrics: RiskMetrics) -> None  # Line 1698
    async def trigger_emergency_stop(self, reason: str) -> None  # Line 1723
    async def reset_emergency_stop(self, reason: str) -> None  # Line 1799
    async def update_price_history(self, symbol: str, price: Decimal) -> None  # Line 1851
    async def _get_all_positions(self) -> list[Position]  # Line 1921
    async def _get_positions_for_symbol(self, symbol: str) -> list[Position]  # Line 1947
    async def _get_current_market_data(self) -> list[MarketData]  # Line 1966
    async def _load_risk_state(self) -> None  # Line 1974
    async def _save_risk_state(self) -> None  # Line 2013
    async def _save_risk_metrics(self, risk_metrics: RiskMetrics) -> None  # Line 2074
    async def _save_emergency_state(self, reason: str) -> None  # Line 2127
    async def _verify_dependencies(self) -> bool  # Line 2158
    async def _service_health_check(self) -> Any  # Line 2184
    async def _emit_state_event(self, event_type: str, event_data: dict) -> None  # Line 2209
    def _emit_validation_error(self, error: Exception, context: dict[str, Any]) -> None  # Line 2218
    async def get_risk_alerts(self, limit: int | None = None) -> list[RiskAlert]  # Line 2243
    async def acknowledge_risk_alert(self, alert_id: str) -> bool  # Line 2258
    def get_current_risk_level(self) -> RiskLevel  # Line 2277
    def is_emergency_stop_active(self) -> bool  # Line 2281
    async def get_risk_summary(self) -> dict[str, Any]  # Line 2285
    async def _cleanup_resources(self) -> None  # Line 2306
    async def _cleanup_stale_symbols(self) -> None  # Line 2385
    def reset_metrics(self) -> None  # Line 2411
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool  # Line 2428
    async def _should_exit_position_impl(self, position: Position, market_data: MarketData) -> bool  # Line 2443
    async def risk_monitoring_context(self, operation: str) -> Any  # Line 2556
    async def _cleanup_connection_resources(self) -> None  # Line 2648
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
    def __init__(self, ...)  # Line 29
    async def calculate_size(self, ...) -> Decimal  # Line 55
    async def validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool  # Line 179
    def _validate_inputs(self, signal: Signal, available_capital: Decimal, current_price: Decimal) -> None  # Line 227
    async def _fixed_percentage_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 243
    async def _kelly_criterion_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 253
    async def _volatility_adjusted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 286
    async def _confidence_weighted_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal  # Line 314
    def _apply_limits(self, position_size: Decimal, available_capital: Decimal) -> Decimal  # Line 326
    def _calculate_kelly_fraction(self, returns: list[Decimal]) -> Decimal  # Line 338
    def _calculate_volatility(self, prices: list[Decimal]) -> Decimal  # Line 367
    async def _get_historical_returns(self, symbol: str) -> list[Decimal]  # Line 376
    async def _get_price_history(self, symbol: str) -> list[Decimal]  # Line 386
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
    def __init__(self, ...)  # Line 38
    async def calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics  # Line 64
    async def get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal  # Line 245
    async def _create_empty_metrics(self) -> RiskMetrics  # Line 261
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None  # Line 281
    async def _get_portfolio_history(self) -> list[Decimal]  # Line 313
    async def _calculate_returns_from_history(self, history: list[Decimal]) -> list[Decimal]  # Line 326
    async def _calculate_var(self, days: int, portfolio_value: Decimal, history: list[Decimal]) -> Decimal  # Line 347
    async def _calculate_expected_shortfall(self, portfolio_value: Decimal, history: list[Decimal]) -> Decimal  # Line 377
    async def _calculate_max_drawdown(self, history: list[Decimal]) -> Decimal  # Line 407
    async def _calculate_current_drawdown(self, portfolio_value: Decimal, history: list[Decimal]) -> Decimal  # Line 419
    async def _calculate_sharpe_ratio(self, history: list[Decimal]) -> Decimal | None  # Line 433
    async def _calculate_total_exposure(self, positions: list[Position], market_data: list[MarketData]) -> Decimal  # Line 460
    async def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal  # Line 475
    async def _calculate_portfolio_beta(self, positions: list[Position]) -> Decimal | None  # Line 489
    async def _determine_risk_level(self, ...) -> RiskLevel  # Line 494
    async def _store_metrics(self, metrics: RiskMetrics) -> None  # Line 536
```

### File: risk_monitoring_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.types import RiskLevel`
- `from src.core.types import RiskMetrics`
- `from src.utils.decimal_utils import to_decimal`
- `from src.utils.messaging_patterns import BoundaryValidator`

#### Class: `RiskAlert`

**Purpose**: Risk alert model

```python
class RiskAlert:
    def __init__(self, ...)  # Line 35
```

#### Class: `RiskMonitoringService`

**Inherits**: BaseService
**Purpose**: Service for real-time risk monitoring and alerting

```python
class RiskMonitoringService(BaseService):
    def __init__(self, ...)  # Line 56
    async def start_monitoring(self, interval: int = 60) -> None  # Line 102
    async def stop_monitoring(self) -> None  # Line 118
    async def check_emergency_conditions(self, metrics: RiskMetrics) -> bool  # Line 145
    async def monitor_metrics(self, metrics: RiskMetrics) -> None  # Line 177
    async def get_active_alerts(self, limit: int | None = None) -> list[RiskAlert]  # Line 218
    async def acknowledge_alert(self, alert_id: str) -> bool  # Line 243
    async def set_threshold(self, threshold_name: str, value) -> None  # Line 261
    async def _monitoring_loop(self, interval: int) -> None  # Line 275
    async def _check_var_thresholds(self, metrics: RiskMetrics) -> None  # Line 307
    async def _check_drawdown_thresholds(self, metrics: RiskMetrics) -> None  # Line 329
    async def _check_sharpe_ratio(self, metrics: RiskMetrics) -> None  # Line 357
    async def _check_risk_level_changes(self, metrics: RiskMetrics) -> None  # Line 375
    async def _check_portfolio_concentration(self, metrics: RiskMetrics) -> None  # Line 406
    async def _create_alert(self, alert_type: str, severity: str, message: str, details: dict[str, Any]) -> None  # Line 416
    async def _trigger_emergency_stop(self, conditions: list[str]) -> None  # Line 437
    async def _get_latest_metrics(self) -> RiskMetrics | None  # Line 477
    async def _get_previous_risk_level(self) -> RiskLevel | None  # Line 496
    async def get_risk_summary(self) -> dict[str, Any]  # Line 509
    def _setup_message_handlers(self) -> None  # Line 558
    async def _handle_threshold_breach(self, event_data: dict) -> None  # Line 599
    async def _handle_emergency_condition(self, event_data: dict) -> None  # Line 620
    async def _handle_risk_level_change(self, event_data: dict) -> None  # Line 634
    async def publish_risk_event(self, event_type: str, event_data: dict) -> None  # Line 659
    async def _monitoring_context(self) -> AsyncIterator[None]  # Line 712
    async def _cleanup_monitoring_resources(self) -> None  # Line 734
```

### File: risk_validation_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.types import OrderRequest`
- `from src.core.types import Position`
- `from src.core.types import RiskLevel`
- `from src.core.types import Signal`

#### Class: `RiskValidationService`

**Inherits**: BaseService
**Purpose**: Service for validating trading signals and orders against risk constraints

```python
class RiskValidationService(BaseService):
    def __init__(self, ...)  # Line 25
    async def validate_signal(self, signal: Signal) -> bool  # Line 75
    async def validate_order(self, order: OrderRequest) -> bool  # Line 183
    async def validate_portfolio_limits(self, new_position: Position) -> bool  # Line 241
    def _validate_signal_structure(self, signal: Signal) -> bool  # Line 315
    def _validate_order_structure(self, order: OrderRequest) -> bool  # Line 339
    async def _validate_order_size_limits(self, order: OrderRequest) -> bool  # Line 359
    async def _validate_portfolio_exposure(self, order: OrderRequest) -> bool  # Line 393
    async def _validate_position_exposure(self, position: Position) -> bool  # Line 429
    async def _check_symbol_position_limits(self, symbol: str) -> bool  # Line 457
    async def _get_current_risk_level(self) -> RiskLevel  # Line 480
    async def _is_emergency_stop_active(self) -> bool  # Line 494
    async def _get_current_positions(self) -> list[Position]  # Line 508
    async def _get_positions_for_symbol(self, symbol: str) -> list[Position]  # Line 531
    async def _get_portfolio_value(self) -> Decimal  # Line 541
    async def _get_current_exposure(self) -> Decimal  # Line 551
    async def _get_current_price(self, symbol: str) -> Decimal  # Line 560
    def _get_max_total_positions(self) -> int  # Line 569
    def _get_max_positions_per_symbol(self) -> int  # Line 575
    def _get_max_position_size_pct(self)  # Line 581
    def _get_max_portfolio_exposure_pct(self)  # Line 589
```

---
**Generated**: Complete reference for risk_management module
**Total Classes**: 51
**Total Functions**: 13