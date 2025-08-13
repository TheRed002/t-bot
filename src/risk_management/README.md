# Risk Management Module

The Risk Management Module provides portfolio and trade risk controls above the exchange layer. It includes position sizing, portfolio limits, circuit breakers, emergency controls, risk metrics, regime detection, and adaptive risk. All components use normalized core types and integrate with exchanges only via `BaseExchange` in emergency procedures.

## What this module does
- Position sizing: fixed %, Kelly, volatility-adjusted, confidence-weighted
- Portfolio limits: total positions, per-symbol limits, exposure, sector, correlation, leverage
- Circuit breakers: daily loss, drawdown, volatility spike, model confidence, system error rate
- Emergency controls: cancel open orders, close positions, block new orders, safe mode, recovery
- Risk metrics: VaR (1d/5d), Expected Shortfall, max/current drawdown, Sharpe, RiskLevel
- Regime detection: volatility/trend/correlation regimes and change tracking
- Adaptive risk: regime-driven parameter and sizing adjustments; stress testing

## What this module does NOT do
- Exchange-specific REST/WebSocket logic
- Strategy signal generation or execution routing
- Data persistence or ETL pipelines

---

## Submodules and summaries
- `base.py`: Abstract base for risk managers; shared validation/state helpers
- `risk_manager.py`: Concrete orchestrator combining sizing, limits, metrics
- `position_sizing.py`: Sizing algorithms and summaries
- `portfolio_limits.py`: Portfolio-level limit checks, correlation/sector exposure
- `risk_metrics.py`: Risk metrics computation and risk level assessment
- `circuit_breakers.py`: Breakers + manager for evaluation, status, and resets
- `emergency_controls.py`: Cancel orders, close positions, block orders, recovery
- `regime_detection.py`: Volatility/trend/correlation regime detection
- `adaptive_risk.py`: Regime-driven adjustments and stress testing

---

## File reference: classes and functions

### base.py
- `class BaseRiskManager`
  - `__init__(config: Config) -> None`
  - `async calculate_position_size(signal: Signal, portfolio_value: Decimal) -> Decimal`
  - `async validate_signal(signal: Signal) -> bool`
  - `async validate_order(order: OrderRequest, portfolio_value: Decimal) -> bool`
  - `async calculate_risk_metrics(positions: list[Position], market_data: list[MarketData]) -> RiskMetrics`
  - `async check_portfolio_limits(new_position: Position) -> bool`
  - `async should_exit_position(position: Position, market_data: MarketData) -> bool`
  - `async update_portfolio_state(positions: list[Position], portfolio_value: Decimal) -> None`
  - `async get_risk_summary() -> dict[str, Any]`
  - `async emergency_stop(reason: str) -> None`
  - `async validate_risk_parameters() -> bool`
  - Helpers: `_calculate_portfolio_exposure(positions) -> Decimal`, `_check_drawdown_limit(current_drawdown) -> bool`, `_check_daily_loss_limit(daily_pnl) -> bool`, `async _log_risk_violation(violation_type, details) -> None`

### risk_manager.py
- `class RiskManager(BaseRiskManager)`
  - `async calculate_position_size(signal: Signal, portfolio_value: Decimal) -> Decimal`
  - `async validate_signal(signal: Signal) -> bool`
  - `async validate_order(order: OrderRequest, portfolio_value: Decimal) -> bool`
  - `async calculate_risk_metrics(positions: list[Position], market_data: list[MarketData]) -> RiskMetrics`
  - `async check_portfolio_limits(new_position: Position) -> bool`
  - `async should_exit_position(position: Position, market_data: MarketData) -> bool`
  - `async update_portfolio_state(positions: list[Position], portfolio_value: Decimal) -> None`
  - `async get_comprehensive_risk_summary() -> dict[str, Any]`
  - `async validate_risk_parameters() -> bool`

### position_sizing.py
- `class PositionSizer`
  - `async calculate_position_size(signal: Signal, portfolio_value: Decimal, method: PositionSizeMethod | None = None) -> Decimal`
  - `async validate_position_size(position_size: Decimal, portfolio_value: Decimal) -> bool`
  - `async update_price_history(symbol: str, price: float) -> None`
  - `async get_position_size_summary(signal: Signal, portfolio_value: Decimal) -> dict[str, Any]`
  - Internals: `async _fixed_percentage_sizing(...) -> Decimal`, `async _kelly_criterion_sizing(...) -> Decimal`, `async _volatility_adjusted_sizing(...) -> Decimal`, `async _confidence_weighted_sizing(...) -> Decimal`

### portfolio_limits.py
- `class PortfolioLimits`
  - `async check_portfolio_limits(new_position: Position) -> bool`
  - `async update_portfolio_state(positions: list[Position], portfolio_value: Decimal) -> None`
  - `async update_return_history(symbol: str, price: float) -> None`
  - `async get_portfolio_summary() -> dict[str, Any]`
  - Internals: `async _check_total_positions_limit(...) -> bool`, `async _check_positions_per_symbol_limit(...) -> bool`, `async _check_portfolio_exposure_limit(...) -> bool`, `async _check_sector_exposure_limit(...) -> bool`, `async _check_correlation_exposure_limit(...) -> bool`, `async _check_leverage_limit(...) -> bool`, `_get_correlation(symbol1: str, symbol2: str) -> float`, `async _log_risk_violation(...) -> None`

### risk_metrics.py
- `class RiskCalculator`
  - `async calculate_risk_metrics(positions: list[Position], market_data: list[MarketData]) -> RiskMetrics`
  - `async get_risk_summary() -> dict[str, Any]`
  - Internals: `async _create_empty_risk_metrics() -> RiskMetrics`, `async _calculate_portfolio_value(...) -> Decimal`, `async _update_portfolio_history(portfolio_value: Decimal) -> None`, `async _calculate_var(days: int, portfolio_value: Decimal) -> Decimal`, `async _calculate_expected_shortfall(portfolio_value: Decimal) -> Decimal`, `async _calculate_max_drawdown() -> Decimal`, `async _calculate_current_drawdown(portfolio_value: Decimal) -> Decimal`, `async _calculate_sharpe_ratio() -> Decimal | None`, `async _determine_risk_level(var_1d: Decimal, current_drawdown: Decimal, sharpe_ratio: Decimal | None) -> RiskLevel`, `async update_position_returns(symbol: str, price: float) -> None`

### circuit_breakers.py
- `class BaseCircuitBreaker`
  - `async check_condition(data: dict[str, Any]) -> bool`
  - `async get_threshold_value() -> Decimal`
  - `async get_current_value(data: dict[str, Any]) -> Decimal`
  - `async evaluate(data: dict[str, Any]) -> bool`
  - `get_status() -> dict[str, Any]`
  - `reset() -> None`
- Concrete breakers:
  - `DailyLossLimitBreaker`, `DrawdownLimitBreaker`, `VolatilitySpikeBreaker`, `ModelConfidenceBreaker`, `SystemErrorRateBreaker`
- `class CircuitBreakerManager`
  - `async evaluate_all(data: dict[str, Any]) -> dict[str, bool]`
  - `async get_status() -> dict[str, Any]`
  - `reset_all() -> None`, `get_triggered_breakers() -> list[str]`, `is_trading_allowed() -> bool`
- Notes: Uses `src.core.types.CircuitBreakerStatus`; alias `CircuitBreakerState = CircuitBreakerStatus` for backward compatibility.

### emergency_controls.py
- `class EmergencyControls`
  - `register_exchange(exchange_name: str, exchange: BaseExchange) -> None`
  - `async activate_emergency_stop(reason: str, trigger_type: CircuitBreakerType) -> None`
  - `async deactivate_emergency_stop(reason: str = "Manual deactivation") -> None`
  - `async validate_order_during_emergency(order: OrderRequest) -> bool`
  - `get_status() -> dict[str, Any]`, `is_trading_allowed() -> bool`, `get_emergency_events(limit: int = 10) -> list[EmergencyEvent]`
  - Internals: `async _execute_emergency_procedures() -> None`, `async _cancel_all_pending_orders() -> None` (uses `get_pending_orders()` or `get_open_orders()`), `async _close_all_positions() -> None` (uses `get_positions()` if provided), `async _block_new_orders() -> None`, `async _switch_to_safe_mode() -> None`, `async _validate_recovery_order(order: OrderRequest) -> bool`, `async _get_portfolio_value() -> Decimal`, `async _recovery_validation_timer() -> None`, `async _validate_recovery_completion() -> bool`, `async activate_manual_override(user_id: str, reason: str) -> None`, `async deactivate_manual_override(user_id: str) -> None`

### regime_detection.py
- `class MarketRegimeDetector`
  - `async detect_volatility_regime(symbol: str, price_data: list[float]) -> MarketRegime`
  - `async detect_trend_regime(symbol: str, price_data: list[float]) -> MarketRegime`
  - `async detect_correlation_regime(symbols: list[str], price_data_dict: dict[str, list[float]]) -> MarketRegime`
  - `async detect_comprehensive_regime(market_data: list[MarketData]) -> MarketRegime`
  - Helpers: `_combine_regimes(...) -> MarketRegime`, `async _check_regime_change(new_regime: MarketRegime) -> None`, `_calculate_change_confidence(new_regime: MarketRegime) -> float`, `get_regime_history(limit: int = 10) -> list[RegimeChangeEvent]`, `get_current_regime() -> MarketRegime`, `get_regime_statistics() -> dict[str, Any]`

### adaptive_risk.py
- `class AdaptiveRiskManager`
  - `async calculate_adaptive_position_size(signal: Signal, current_regime: MarketRegime, portfolio_value: Decimal) -> Decimal`
  - `async calculate_adaptive_stop_loss(signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal`
  - `async calculate_adaptive_take_profit(signal: Signal, current_regime: MarketRegime, entry_price: Decimal) -> Decimal`
  - `async calculate_adaptive_portfolio_limits(current_regime: MarketRegime, base_limits: dict[str, Any]) -> dict[str, Any]`
  - `async run_stress_test(portfolio_positions: list[Position], scenario_name: str = "market_crash") -> dict[str, Any]`
  - Helpers: `async _get_correlation_regime() -> MarketRegime | None`, `async _calculate_momentum_adjustment(symbol: str) -> float`, `get_adaptive_parameters(regime: MarketRegime) -> dict[str, Any]`, `get_stress_test_scenarios() -> list[str]`, `update_regime_detector(new_detector: MarketRegimeDetector) -> None`

---

## Import relationships
### Modules that should import from Risk Management
- `src.strategies` (sizing, limits, overlays), `src/main.py` (or application entrypoint) for orchestration, `tests` for unit/integration validation
- Optional: `src.capital_management` (reverse integration notes in `position_sizing.py`)

### Local module dependencies (only local modules)
- `src.core` (types, config, exceptions, logging)
- `src.utils` (decorators, helpers)
- `src.error_handling` (ErrorHandler and context creation)
- `src.exchanges` (only via `BaseExchange` in `emergency_controls.py`)

---

## Consistency and notes
- Breaker state uses `src.core.types.CircuitBreakerStatus` to avoid enum duplication; alias maintained for tests.
- Emergency procedures rely on optional `BaseExchange.get_open_orders()` and `BaseExchange.get_positions()`; concrete exchanges implement safe defaults for spot.
- Pydantic validations in `src.core.types` are intentionally relaxed for tests; this module enforces business validations (quantities, exposure, etc.).
- All significant public methods are async and decorated with `time_execution` for observability.

---

## Quick usage
- Instantiate `RiskManager(config)`; call `update_portfolio_state(...)`; use `calculate_position_size(...)`, `validate_order(...)`, and `calculate_risk_metrics(...)` in execution pipeline.
- For emergency flows, create `EmergencyControls(config, risk_manager, breaker_manager)` and register exchanges via `register_exchange(name, exchange)`.
