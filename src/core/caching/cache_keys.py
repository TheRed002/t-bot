"""Cache key management for consistent Redis key naming."""

from datetime import datetime
from typing import Any


class CacheKeys:
    """Centralized cache key management for consistent naming patterns."""

    # Namespace separators
    SEPARATOR = ":"

    # Main namespaces
    STATE = "state"
    RISK = "risk"
    MARKET_DATA = "market"
    ORDERS = "orders"
    STRATEGY = "strategy"
    BOT = "bot"
    API = "api"
    SESSION = "session"
    LOCKS = "locks"
    METRICS = "metrics"

    @classmethod
    def _build_key(cls, namespace: str, *parts: Any) -> str:
        """Build a cache key with proper formatting."""
        key_parts = [str(part).replace(":", "_") for part in [namespace, *list(parts)]]
        return cls.SEPARATOR.join(key_parts)

    # State Management Keys
    @classmethod
    def state_snapshot(cls, bot_id: str) -> str:
        """Cache key for bot state snapshots."""
        return cls._build_key(cls.STATE, "snapshot", bot_id)

    @classmethod
    def trade_lifecycle(cls, trade_id: str) -> str:
        """Cache key for trade lifecycle states."""
        return cls._build_key(cls.STATE, "trade", trade_id)

    @classmethod
    def state_checkpoint(cls, checkpoint_id: str) -> str:
        """Cache key for state checkpoints."""
        return cls._build_key(cls.STATE, "checkpoint", checkpoint_id)

    # Risk Management Keys
    @classmethod
    def risk_metrics(cls, bot_id: str, timeframe: str = "1h") -> str:
        """Cache key for risk metrics."""
        return cls._build_key(cls.RISK, "metrics", bot_id, timeframe)

    @classmethod
    def position_limits(cls, bot_id: str, symbol: str) -> str:
        """Cache key for position limits."""
        return cls._build_key(cls.RISK, "limits", bot_id, symbol)

    @classmethod
    def correlation_matrix(cls, timeframe: str = "1h") -> str:
        """Cache key for correlation matrices."""
        return cls._build_key(cls.RISK, "correlation", timeframe)

    @classmethod
    def var_calculation(cls, portfolio_id: str, confidence: str = "95") -> str:
        """Cache key for VaR calculations."""
        return cls._build_key(cls.RISK, "var", portfolio_id, confidence)

    # Market Data Keys
    @classmethod
    def market_price(cls, symbol: str, exchange: str = "all") -> str:
        """Cache key for latest market prices."""
        return cls._build_key(cls.MARKET_DATA, "price", exchange, symbol)

    @classmethod
    def order_book(cls, symbol: str, exchange: str, depth: int = 20) -> str:
        """Cache key for order book snapshots."""
        return cls._build_key(cls.MARKET_DATA, "orderbook", exchange, symbol, depth)

    @classmethod
    def technical_indicator(cls, symbol: str, indicator: str, period: int) -> str:
        """Cache key for technical indicators."""
        return cls._build_key(cls.MARKET_DATA, "indicator", symbol, indicator, period)

    @classmethod
    def ohlcv_data(cls, symbol: str, timeframe: str, exchange: str = "all") -> str:
        """Cache key for OHLCV data."""
        return cls._build_key(cls.MARKET_DATA, "ohlcv", exchange, symbol, timeframe)

    # Order Management Keys
    @classmethod
    def active_orders(cls, bot_id: str, symbol: str = "all") -> str:
        """Cache key for active orders."""
        return cls._build_key(cls.ORDERS, "active", bot_id, symbol)

    @classmethod
    def order_history(cls, bot_id: str, page: int = 1) -> str:
        """Cache key for order history with pagination."""
        return cls._build_key(cls.ORDERS, "history", bot_id, page)

    @classmethod
    def execution_state(cls, algorithm: str, bot_id: str) -> str:
        """Cache key for execution algorithm state."""
        return cls._build_key(cls.ORDERS, "execution", algorithm, bot_id)

    @classmethod
    def order_lock(cls, symbol: str, bot_id: str) -> str:
        """Cache key for distributed order locks."""
        return cls._build_key(cls.LOCKS, "order", symbol, bot_id)

    # Strategy Keys
    @classmethod
    def strategy_signals(cls, strategy_id: str, symbol: str) -> str:
        """Cache key for strategy signals."""
        return cls._build_key(cls.STRATEGY, "signals", strategy_id, symbol)

    @classmethod
    def strategy_params(cls, strategy_id: str) -> str:
        """Cache key for strategy parameters."""
        return cls._build_key(cls.STRATEGY, "params", strategy_id)

    @classmethod
    def backtest_results(cls, strategy_id: str, config_hash: str) -> str:
        """Cache key for backtesting results."""
        return cls._build_key(cls.STRATEGY, "backtest", strategy_id, config_hash)

    @classmethod
    def strategy_performance(cls, strategy_id: str, timeframe: str = "1d") -> str:
        """Cache key for strategy performance metrics."""
        return cls._build_key(cls.STRATEGY, "performance", strategy_id, timeframe)

    # Bot Management Keys
    @classmethod
    def bot_config(cls, bot_id: str) -> str:
        """Cache key for bot configurations."""
        return cls._build_key(cls.BOT, "config", bot_id)

    @classmethod
    def bot_status(cls, bot_id: str) -> str:
        """Cache key for bot status and health."""
        return cls._build_key(cls.BOT, "status", bot_id)

    @classmethod
    def resource_allocation(cls, bot_id: str) -> str:
        """Cache key for resource allocations."""
        return cls._build_key(cls.BOT, "resources", bot_id)

    @classmethod
    def bot_session(cls, bot_id: str, session_id: str) -> str:
        """Cache key for bot session management."""
        return cls._build_key(cls.BOT, "session", bot_id, session_id)

    # API Response Keys
    @classmethod
    def api_response(cls, endpoint: str, user_id: str = "anonymous", **params) -> str:
        """Cache key for API responses."""
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        return cls._build_key(cls.API, "response", endpoint, user_id, param_str)

    @classmethod
    def user_session(cls, user_id: str, session_id: str) -> str:
        """Cache key for user sessions."""
        return cls._build_key(cls.SESSION, "user", user_id, session_id)

    @classmethod
    def auth_token(cls, user_id: str, token_hash: str) -> str:
        """Cache key for authentication tokens."""
        return cls._build_key(cls.SESSION, "auth", user_id, token_hash)

    # Metrics and Monitoring Keys
    @classmethod
    def cache_stats(cls, namespace: str) -> str:
        """Cache key for cache hit/miss statistics."""
        return cls._build_key(cls.METRICS, "cache_stats", namespace)

    @classmethod
    def performance_metrics(cls, component: str, timeframe: str = "5m") -> str:
        """Cache key for performance metrics."""
        return cls._build_key(cls.METRICS, "performance", component, timeframe)

    # Time-based keys with auto-expiry patterns
    @classmethod
    def time_window_key(cls, base_key: str, window_minutes: int = 5) -> str:
        """Create time-windowed key that auto-expires."""
        timestamp = datetime.utcnow().timestamp()
        window_timestamp = int(timestamp // (window_minutes * 60)) * (window_minutes * 60)
        return f"{base_key}:window:{window_timestamp}"

    @classmethod
    def daily_key(cls, base_key: str) -> str:
        """Create daily key that resets each day."""
        date_str = datetime.utcnow().strftime("%Y%m%d")
        return f"{base_key}:daily:{date_str}"

    @classmethod
    def hourly_key(cls, base_key: str) -> str:
        """Create hourly key that resets each hour."""
        hour_str = datetime.utcnow().strftime("%Y%m%d%H")
        return f"{base_key}:hourly:{hour_str}"
