"""Centralized event constants for the T-Bot trading system.

This module provides a single source of truth for all event names used
across the system to ensure consistency between publishers and subscribers.
"""


class AlertEvents:
    """Alert-related event names."""

    CREATED = "alert.created"
    FIRED = "alert.fired"
    RESOLVED = "alert.resolved"
    ACKNOWLEDGED = "alert.acknowledged"
    ESCALATED = "alert.escalated"
    SUPPRESSED = "alert.suppressed"


class OrderEvents:
    """Order-related event names."""

    CREATED = "order.created"
    FILLED = "order.filled"
    PARTIALLY_FILLED = "order.partially_filled"
    CANCELLED = "order.cancelled"
    REJECTED = "order.rejected"
    EXPIRED = "order.expired"


class TradeEvents:
    """Trade-related event names."""

    EXECUTED = "trade.executed"
    SETTLED = "trade.settled"
    FAILED = "trade.failed"


class PositionEvents:
    """Position-related event names."""

    OPENED = "position.opened"
    CLOSED = "position.closed"
    MODIFIED = "position.modified"
    LIQUIDATED = "position.liquidated"


class RiskEvents:
    """Risk management event names."""

    LIMIT_EXCEEDED = "risk.limit_exceeded"
    MARGIN_CALL = "risk.margin_call"
    CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker_triggered"
    EXPOSURE_WARNING = "risk.exposure_warning"
    VALIDATION_ERROR = "risk.validation_error"
    THRESHOLD_BREACH = "risk.threshold_breach"  # Added for consistency with risk_management
    EMERGENCY_CONDITION = "risk.emergency_condition"  # Added for consistency with risk_management
    RISK_LEVEL_CHANGE = "risk.risk_level_change"  # Added for consistency with risk_management


class MetricEvents:
    """Metric-related event names."""

    RECORDED = "metric.recorded"
    EXPORTED = "metric.exported"
    AGGREGATED = "metric.aggregated"


class SystemEvents:
    """System-level event names."""

    STARTUP = "system.startup"
    SHUTDOWN = "system.shutdown"
    HEALTH_CHECK_FAILED = "system.health_check_failed"
    COMPONENT_ERROR = "system.component_error"
    MAINTENANCE_MODE = "system.maintenance_mode"


class MarketDataEvents:
    """Market data event names."""

    PRICE_UPDATE = "market.price_update"
    ORDER_BOOK_UPDATE = "market.order_book_update"
    TRADE_UPDATE = "market.trade_update"
    TICKER_UPDATE = "market.ticker_update"


class StrategyEvents:
    """Strategy-related event names."""

    SIGNAL_GENERATED = "strategy.signal_generated"
    ENTRY_TRIGGERED = "strategy.entry_triggered"
    EXIT_TRIGGERED = "strategy.exit_triggered"
    REBALANCE_REQUIRED = "strategy.rebalance_required"
    OPTIMIZATION_STARTED = "strategy.optimization_started"
    OPTIMIZATION_COMPLETED = "strategy.optimization_completed"
    OPTIMIZATION_FAILED = "strategy.optimization_failed"
    PARAMETERS_UPDATED = "strategy.parameters_updated"


class CapitalEvents:
    """Capital management event names."""

    ALLOCATED = "capital.allocated"
    RELEASED = "capital.released"
    REBALANCED = "capital.rebalanced"
    UTILIZATION_UPDATED = "capital.utilization_updated"
    RESERVE_BREACHED = "capital.reserve_breached"
    LIMIT_EXCEEDED = "capital.limit_exceeded"


class ExchangeEvents:
    """Exchange connection event names."""

    CONNECTED = "exchange.connected"
    DISCONNECTED = "exchange.disconnected"
    RECONNECTING = "exchange.reconnecting"
    RATE_LIMITED = "exchange.rate_limited"
    API_ERROR = "exchange.api_error"


class MLEvents:
    """Machine learning event names."""

    MODEL_REGISTERED = "ml.model_registered"
    MODEL_LOADED = "ml.model_loaded"
    MODEL_PROMOTED = "ml.model_promoted"
    MODEL_DEACTIVATED = "ml.model_deactivated"
    MODEL_DELETED = "ml.model_deleted"


class TrainingEvents:
    """Model training event names."""

    STARTED = "training.started"
    COMPLETED = "training.completed"
    FAILED = "training.failed"
    PROGRESS_UPDATED = "training.progress_updated"
    EARLY_STOPPED = "training.early_stopped"


class InferenceEvents:
    """Model inference event names."""

    PREDICTION_REQUESTED = "inference.prediction_requested"
    PREDICTION_COMPLETED = "inference.prediction_completed"
    PREDICTION_FAILED = "inference.prediction_failed"
    BATCH_STARTED = "inference.batch_started"
    BATCH_COMPLETED = "inference.batch_completed"


class FeatureEvents:
    """Feature engineering event names."""

    FEATURES_COMPUTED = "features.computed"
    FEATURES_SELECTED = "features.selected"
    FEATURES_CACHED = "features.cached"
    FEATURE_DRIFT_DETECTED = "features.drift_detected"


class ModelValidationEvents:
    """Model validation event names."""

    VALIDATION_STARTED = "validation.started"
    VALIDATION_COMPLETED = "validation.completed"
    VALIDATION_FAILED = "validation.failed"
    DRIFT_DETECTED = "validation.drift_detected"
    PERFORMANCE_DEGRADED = "validation.performance_degraded"


class StateEvents:
    """State management event names."""

    CHANGED = "state.changed"
    VALIDATED = "state.validated"
    PERSISTED = "state.persisted"
    SYNCHRONIZED = "state.synchronized"
    CREATED = "state.created"
    UPDATED = "state.updated"
    DELETED = "state.deleted"
    RESTORED = "state.restored"


class BacktestEvents:
    """Backtesting event names."""

    STARTED = "backtest.started"
    COMPLETED = "backtest.completed"
    FAILED = "backtest.failed"
    CANCELLED = "backtest.cancelled"
    PROGRESS_UPDATED = "backtest.progress_updated"
    RESULT_GENERATED = "backtest.result_generated"


class DataEvents:
    """Data processing event names."""

    STORED = "data.stored"
    RETRIEVED = "data.retrieved"
    VALIDATED = "data.validated"
    VALIDATION_FAILED = "data.validation_failed"
    CACHE_HIT = "data.cache_hit"
    CACHE_MISS = "data.cache_miss"
    PIPELINE_STARTED = "data.pipeline_started"
    PIPELINE_COMPLETED = "data.pipeline_completed"
    PIPELINE_FAILED = "data.pipeline_failed"
    QUALITY_ALERT = "data.quality_alert"
    PERFORMANCE_ALERT = "data.performance_alert"


class OptimizationEvents:
    """Optimization-related event names."""

    STARTED = "optimization.started"
    COMPLETED = "optimization.completed"
    FAILED = "optimization.failed"
    CANCELLED = "optimization.cancelled"
    PROGRESS_UPDATED = "optimization.progress_updated"
    PARAMETER_SET_EVALUATED = "optimization.parameter_set_evaluated"
    CONVERGENCE_ACHIEVED = "optimization.convergence_achieved"
    OVERFITTING_DETECTED = "optimization.overfitting_detected"
    RESULT_SAVED = "optimization.result_saved"
    BACKTEST_REQUESTED = "optimization.backtest_requested"
    BACKTEST_COMPLETED = "optimization.backtest_completed"
    BACKTEST_FAILED = "optimization.backtest_failed"


class BotEvents:
    """Bot management event names."""

    # Lifecycle events
    CREATED = "bot.created"
    STARTED = "bot.started"
    STOPPED = "bot.stopped"
    PAUSED = "bot.paused"
    RESUMED = "bot.resumed"
    ERROR = "bot.error"
    DELETED = "bot.deleted"

    # Status events
    STARTING = "bot.starting"
    STOPPING = "bot.stopping"
    INITIALIZING = "bot.initializing"

    # Performance events
    METRICS_UPDATE = "bot.metrics_update"
    PERFORMANCE_ALERT = "bot.performance_alert"

    # Risk events
    RISK_ALERT = "bot.risk_alert"
    RISK_LIMIT_EXCEEDED = "bot.risk_limit_exceeded"
    EMERGENCY_STOP = "bot.emergency_stop"

    # Trading events
    TRADE_EXECUTED = "bot.trade_executed"
    ORDER_PLACED = "bot.order_placed"
    ORDER_FILLED = "bot.order_filled"
    POSITION_UPDATE = "bot.position_update"
