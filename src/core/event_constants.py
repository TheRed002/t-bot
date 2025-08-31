"""Centralized event constants for the T-Bot trading system.

This module provides a single source of truth for all event names used
across the system to ensure consistency between publishers and subscribers.
"""


class AlertEvents:
    """Alert-related event names."""

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
