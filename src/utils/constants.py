"""
System-Wide Constants and Enumerations

This module defines system-wide constants and enumerations used throughout the
trading bot system to ensure consistency and maintainability.

Key Constants:
- Trading Constants: market hours, settlement times, precision levels
- API Constants: endpoints, rate limits, timeouts
- Financial Constants: fee structures, minimum amounts, maximum amounts
- Configuration Constants: default values, limits, thresholds
- Error Constants: error codes, message templates
- Market Constants: symbol mappings, exchange specifications

Dependencies:
- P-001: Core types, exceptions, logging
"""


# =============================================================================
# Trading Constants
# =============================================================================

# Market hours for different exchanges (UTC)
MARKET_HOURS = {
    "binance": {
        "open": "00:00",
        "close": "23:59",
        "timezone": "UTC",
        "description": "24/7 crypto trading",
    },
    "okx": {
        "open": "00:00",
        "close": "23:59",
        "timezone": "UTC",
        "description": "24/7 crypto trading",
    },
    "coinbase": {
        "open": "00:00",
        "close": "23:59",
        "timezone": "UTC",
        "description": "24/7 crypto trading",
    },
    "nyse": {
        "open": "13:30",  # 9:30 AM EST
        "close": "20:00",  # 4:00 PM EST
        "timezone": "America/New_York",
        "description": "Traditional market hours",
    },
}

# Settlement times for different asset types
SETTLEMENT_TIMES = {
    "crypto": "T+0",  # Immediate settlement
    "forex": "T+2",  # 2 business days
    "stocks": "T+2",  # 2 business days
    "options": "T+1",  # 1 business day
    "futures": "T+0",  # Daily settlement
}

# Precision levels for different asset types
PRECISION_LEVELS = {
    "BTC": 8,  # Bitcoin precision
    "ETH": 6,  # Ethereum precision
    "USDT": 2,  # Tether precision
    "USD": 2,  # US Dollar precision
    "JPY": 0,  # Japanese Yen (no decimals)
    "default": 4,  # Default precision
    "fee": 6,  # Fee calculation precision
    "price": 8,  # Price calculation precision
    "position": 8,  # Position size precision
}

# Trading session definitions
TRADING_SESSIONS = {
    "asian": {
        "start": "00:00",
        "end": "08:00",
        "timezone": "UTC",
        "description": "Asian trading session",
    },
    "european": {
        "start": "08:00",
        "end": "16:00",
        "timezone": "UTC",
        "description": "European trading session",
    },
    "american": {
        "start": "13:30",
        "end": "20:00",
        "timezone": "UTC",
        "description": "American trading session",
    },
}


# =============================================================================
# API Constants
# =============================================================================

# API endpoints for different exchanges
API_ENDPOINTS = {
    "binance": {
        "base_url": "https://api.binance.com",
        "testnet_url": "https://testnet.binance.vision",
        "ws_url": "wss://stream.binance.com:9443/ws",
        "ws_testnet_url": "wss://testnet-dex.binance.org/ws",
    },
    "okx": {
        "base_url": "https://www.okx.com",
        "sandbox_url": "https://www.okx.com",
        "ws_url": "wss://ws.okx.com:8443/ws/v5/public",
        "ws_private_url": "wss://ws.okx.com:8443/ws/v5/private",
    },
    "coinbase": {
        "base_url": "https://api.pro.coinbase.com",
        "sandbox_url": "https://api-public.sandbox.pro.coinbase.com",
        "ws_url": "wss://ws-feed.pro.coinbase.com",
    },
}

# Rate limits for different exchanges (requests per time window)
RATE_LIMITS = {
    "binance": {
        "requests_per_minute": 1200,
        "orders_per_10_seconds": 50,
        "orders_per_24_hours": 160000,
        "weight_per_minute": 1200,
    },
    "okx": {
        "requests_per_2_seconds": 60,
        "orders_per_2_seconds": 600,
        "historical_data_per_2_seconds": 20,
    },
    "coinbase": {"requests_per_minute": 600, "orders_per_second": 15, "points_per_minute": 8000},
}

# API timeout settings
TIMEOUTS = {
    "default": 30,  # Default timeout in seconds
    "short": 10,  # Short timeout for quick operations
    "long": 60,  # Long timeout for complex operations
    "websocket": 5,  # WebSocket connection timeout
    "order_timeout": 15,  # Order placement timeout
    "data_timeout": 20,  # Data retrieval timeout
}

# HTTP status codes
HTTP_STATUS_CODES = {
    "OK": 200,
    "CREATED": 201,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "RATE_LIMIT": 429,
    "INTERNAL_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503,
}


# =============================================================================
# Financial Constants
# =============================================================================

# Fee structures for different exchanges
FEE_STRUCTURES = {
    "binance": {
        "maker_fee": 0.001,  # 0.1% maker fee
        "taker_fee": 0.001,  # 0.1% taker fee
        "min_fee": 0.00001,  # Minimum fee in BTC
        "fee_currency": "BNB",  # Fee discount currency
    },
    "okx": {
        "maker_fee": 0.0008,  # 0.08% maker fee
        "taker_fee": 0.001,  # 0.1% taker fee
        "min_fee": 0.00001,  # Minimum fee
        "fee_currency": "OKB",  # Fee discount currency
    },
    "coinbase": {
        "maker_fee": 0.004,  # 0.4% maker fee
        "taker_fee": 0.006,  # 0.6% taker fee
        "min_fee": 0.01,  # Minimum fee in USD
        "fee_currency": "USD",  # Fee currency
    },
}

# Global fee structure for arbitrage calculations
GLOBAL_FEE_STRUCTURE = {
    "maker_fee": 0.001,  # Default maker fee
    "taker_fee": 0.001,  # Default taker fee
    "min_fee": 0.00001,  # Default minimum fee
    "fee_currency": "USDT",  # Default fee currency
}

# Minimum amounts for different operations
MINIMUM_AMOUNTS = {
    "BTC": {
        "min_order_size": 0.00001,  # Minimum BTC order size
        "min_notional": 10.0,  # Minimum order value in USDT
        "min_increment": 0.00000001,  # Minimum price increment
    },
    "ETH": {"min_order_size": 0.001, "min_notional": 10.0, "min_increment": 0.000001},
    "USDT": {"min_order_size": 10.0, "min_notional": 10.0, "min_increment": 0.01},
}

# Global minimum amounts for general operations
GLOBAL_MINIMUM_AMOUNTS = {
    "position": 0.001,  # Minimum position size for arbitrage
    "order": 0.00001,  # Minimum order size
    "notional": 10.0,  # Minimum notional value
}

# Maximum amounts for safety limits
MAXIMUM_AMOUNTS = {
    "max_order_size": 1000.0,  # Maximum order size in base currency
    "max_notional": 1000000.0,  # Maximum order value in quote currency
    "max_leverage": 125.0,  # Maximum leverage for futures
    "max_position_size": 0.5,  # Maximum position size as % of portfolio
    "max_daily_loss": 0.05,  # Maximum daily loss as % of portfolio
    "max_drawdown": 0.15,  # Maximum drawdown as % of portfolio
}

# Slippage tolerance levels
SLIPPAGE_TOLERANCE = {
    "low": 0.001,  # 0.1% slippage tolerance
    "medium": 0.005,  # 0.5% slippage tolerance
    "high": 0.01,  # 1.0% slippage tolerance
    "max": 0.05,  # 5.0% maximum slippage tolerance
}


# =============================================================================
# Configuration Constants
# =============================================================================

# Default configuration values
DEFAULT_VALUES = {
    "position_size": 0.02,  # Default position size (2% of portfolio)
    "stop_loss": 0.02,  # Default stop loss (2%)
    "take_profit": 0.04,  # Default take profit (4%)
    "max_positions": 10,  # Default maximum positions
    "confidence_threshold": 0.6,  # Default confidence threshold
    "risk_free_rate": 0.02,  # Default risk-free rate (2%)
    "volatility_window": 20,  # Default volatility calculation window
    "correlation_threshold": 0.7,  # Default correlation threshold
    "rebalance_frequency": 24,  # Default rebalance frequency (hours)
    "data_retention_days": 365,  # Default data retention period
}

# System limits
LIMITS = {
    "max_concurrent_connections": 100,  # Maximum concurrent connections
    "max_memory_usage_mb": 2048,  # Maximum memory usage in MB
    "max_cpu_usage_percent": 80,  # Maximum CPU usage percentage
    "max_disk_usage_gb": 100,  # Maximum disk usage in GB
    "max_log_file_size_mb": 100,  # Maximum log file size in MB
    "max_backup_files": 10,  # Maximum number of backup files
    "max_retry_attempts": 3,  # Maximum retry attempts
    "max_circuit_breaker_failures": 5,  # Maximum failures before circuit breaker
    "max_cache_size_mb": 512,  # Maximum cache size in MB
    "max_websocket_connections": 50,  # Maximum WebSocket connections
}

# Thresholds for different operations
THRESHOLDS = {
    "performance": {
        "max_execution_time_ms": 1000,  # Maximum function execution time
        "max_api_latency_ms": 500,  # Maximum API latency
        "max_database_query_time_ms": 100,  # Maximum database query time
        "min_throughput_requests_per_second": 10,  # Minimum throughput
    },
    "risk": {
        "max_daily_loss_pct": 0.05,  # Maximum daily loss (5%)
        "max_weekly_loss_pct": 0.10,  # Maximum weekly loss (10%)
        "max_monthly_loss_pct": 0.15,  # Maximum monthly loss (15%)
        "min_sharpe_ratio": 0.5,  # Minimum Sharpe ratio
        "max_var_confidence": 0.95,  # Maximum VaR confidence level
    },
    "data_quality": {
        "min_data_points": 100,  # Minimum data points for analysis
        "max_missing_data_pct": 0.05,  # Maximum missing data percentage
        "min_price_change_pct": 0.0001,  # Minimum price change for validation
        "max_price_change_pct": 0.5,  # Maximum price change for validation
    },
}


# =============================================================================
# Error Constants
# =============================================================================

# Error codes for different types of errors
ERROR_CODES = {
    # System errors (1000-1999)
    "SYSTEM_ERROR": 1000,
    "CONFIGURATION_ERROR": 1001,
    "DATABASE_ERROR": 1002,
    "NETWORK_ERROR": 1003,
    "TIMEOUT_ERROR": 1004,
    "MEMORY_ERROR": 1005,
    # Exchange errors (2000-2999)
    "EXCHANGE_ERROR": 2000,
    "EXCHANGE_CONNECTION_ERROR": 2001,
    "EXCHANGE_RATE_LIMIT_ERROR": 2002,
    "EXCHANGE_INSUFFICIENT_FUNDS": 2003,
    "EXCHANGE_INVALID_ORDER": 2004,
    "EXCHANGE_ORDER_REJECTED": 2005,
    "EXCHANGE_SYMBOL_NOT_FOUND": 2006,
    # Trading errors (3000-3999)
    "TRADING_ERROR": 3000,
    "INSUFFICIENT_BALANCE": 3001,
    "INVALID_ORDER_PARAMETERS": 3002,
    "ORDER_EXECUTION_FAILED": 3003,
    "POSITION_LIMIT_EXCEEDED": 3004,
    "RISK_LIMIT_VIOLATION": 3005,
    "SLIPPAGE_TOO_HIGH": 3006,
    # Data errors (4000-4999)
    "DATA_ERROR": 4000,
    "DATA_VALIDATION_ERROR": 4001,
    "DATA_SOURCE_ERROR": 4002,
    "DATA_NOT_AVAILABLE": 4003,
    "DATA_FORMAT_ERROR": 4004,
    "DATA_TIMEOUT_ERROR": 4005,
    # Strategy errors (5000-5999)
    "STRATEGY_ERROR": 5000,
    "STRATEGY_CONFIGURATION_ERROR": 5001,
    "STRATEGY_EXECUTION_ERROR": 5002,
    "SIGNAL_GENERATION_ERROR": 5003,
    "SIGNAL_VALIDATION_ERROR": 5004,
    # Risk management errors (6000-6999)
    "RISK_MANAGEMENT_ERROR": 6000,
    "POSITION_LIMIT_ERROR": 6001,
    "DRAWDOWN_LIMIT_ERROR": 6002,
    "VAR_LIMIT_ERROR": 6003,
    "CORRELATION_LIMIT_ERROR": 6004,
    # Validation errors (7000-7999)
    "VALIDATION_ERROR": 7000,
    "INPUT_VALIDATION_ERROR": 7001,
    "OUTPUT_VALIDATION_ERROR": 7002,
    "TYPE_VALIDATION_ERROR": 7003,
    "RANGE_VALIDATION_ERROR": 7004,
}

# Error message templates
ERROR_MESSAGES = {
    "SYSTEM_ERROR": "System error occurred: {details}",
    "CONFIGURATION_ERROR": "Configuration error: {details}",
    "DATABASE_ERROR": "Database error: {details}",
    "NETWORK_ERROR": "Network error: {details}",
    "TIMEOUT_ERROR": "Operation timed out: {details}",
    "EXCHANGE_ERROR": "Exchange error: {details}",
    "EXCHANGE_CONNECTION_ERROR": "Failed to connect to exchange: {details}",
    "EXCHANGE_RATE_LIMIT_ERROR": "Rate limit exceeded: {details}",
    "INSUFFICIENT_BALANCE": "Insufficient balance for order: {details}",
    "INVALID_ORDER_PARAMETERS": "Invalid order parameters: {details}",
    "ORDER_EXECUTION_FAILED": "Order execution failed: {details}",
    "POSITION_LIMIT_EXCEEDED": "Position limit exceeded: {details}",
    "RISK_LIMIT_VIOLATION": "Risk limit violation: {details}",
    "DATA_ERROR": "Data error: {details}",
    "STRATEGY_ERROR": "Strategy error: {details}",
    "VALIDATION_ERROR": "Validation error: {details}",
}

# Error severity levels
ERROR_SEVERITY = {
    "CRITICAL": 1,  # System failure, immediate action required
    "HIGH": 2,  # Major issue, action required soon
    "MEDIUM": 3,  # Minor issue, monitor closely
    "LOW": 4,  # Informational, no immediate action
    "DEBUG": 5,  # Debug information only
}

# Error recovery strategies
ERROR_RECOVERY_STRATEGIES = {
    "RETRY": "retry",
    "CIRCUIT_BREAKER": "circuit_breaker",
    "FALLBACK": "fallback",
    "DEGRADED_MODE": "degraded_mode",
    "MANUAL_INTERVENTION": "manual_intervention",
    "SYSTEM_RESTART": "system_restart",
}


# =============================================================================
# Market Constants
# =============================================================================

# Symbol mappings for different exchanges
SYMBOL_MAPPINGS = {
    "binance": {
        "BTCUSDT": "BTC/USDT",
        "ETHUSDT": "ETH/USDT",
        "BNBUSDT": "BNB/USDT",
        "ADAUSDT": "ADA/USDT",
        "DOTUSDT": "DOT/USDT",
    },
    "okx": {
        "BTC-USDT": "BTC/USDT",
        "ETH-USDT": "ETH/USDT",
        "BNB-USDT": "BNB/USDT",
        "ADA-USDT": "ADA/USDT",
        "DOT-USDT": "DOT/USDT",
    },
    "coinbase": {
        "BTC-USD": "BTC/USD",
        "ETH-USD": "ETH/USD",
        "BTC-USDT": "BTC/USDT",
        "ETH-USDT": "ETH/USDT",
    },
}

# Exchange specifications
EXCHANGE_SPECIFICATIONS = {
    "binance": {
        "name": "Binance",
        "type": "crypto",
        "supported_markets": ["spot", "futures", "options"],
        "supported_order_types": ["market", "limit", "stop_loss", "take_profit", "oco"],
        "supports_margin_trading": True,
        "supports_short_selling": True,
        "max_leverage": 125,
        "min_order_size": 0.00001,
        "fee_structure": "maker_taker",
    },
    "okx": {
        "name": "OKX",
        "type": "crypto",
        "supported_markets": ["spot", "futures", "options"],
        "supported_order_types": ["market", "limit", "stop_loss", "take_profit"],
        "supports_margin_trading": True,
        "supports_short_selling": True,
        "max_leverage": 125,
        "min_order_size": 0.00001,
        "fee_structure": "maker_taker",
    },
    "coinbase": {
        "name": "Coinbase Pro",
        "type": "crypto",
        "supported_markets": ["spot"],
        "supported_order_types": ["market", "limit", "stop_loss"],
        "supports_margin_trading": False,
        "supports_short_selling": False,
        "max_leverage": 1,
        "min_order_size": 10.0,
        "fee_structure": "maker_taker",
    },
}

# Trading pairs with their specifications
TRADING_PAIRS = {
    "BTCUSDT": {
        "base_currency": "BTC",
        "quote_currency": "USDT",
        "min_order_size": 0.00001,
        "price_precision": 8,
        "quantity_precision": 6,
        "min_notional": 10.0,
        "status": "TRADING",
    },
    "ETHUSDT": {
        "base_currency": "ETH",
        "quote_currency": "USDT",
        "min_order_size": 0.001,
        "price_precision": 6,
        "quantity_precision": 5,
        "min_notional": 10.0,
        "status": "TRADING",
    },
    "BNBUSDT": {
        "base_currency": "BNB",
        "quote_currency": "USDT",
        "min_order_size": 0.01,
        "price_precision": 4,
        "quantity_precision": 3,
        "min_notional": 10.0,
        "status": "TRADING",
    },
}

# Market data intervals
MARKET_DATA_INTERVALS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}

# Order book depth levels
ORDER_BOOK_DEPTHS = {"minimal": 5, "standard": 10, "detailed": 20, "comprehensive": 50, "full": 100}

# Time in force options
TIME_IN_FORCE_OPTIONS = {
    "GTC": "Good Till Canceled",
    "IOC": "Immediate or Cancel",
    "FOK": "Fill or Kill",
    "DAY": "Day Order",
    "GTT": "Good Till Time",
}

# Order status values
ORDER_STATUS_VALUES = {
    "PENDING": "pending",
    "FILLED": "filled",
    "PARTIALLY_FILLED": "partial",
    "CANCELED": "canceled",
    "REJECTED": "rejected",
    "EXPIRED": "expired",
}

# Position side values
POSITION_SIDE_VALUES = {"LONG": "long", "SHORT": "short", "BOTH": "both"}

# Signal direction values
SIGNAL_DIRECTION_VALUES = {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}

# Risk level values
RISK_LEVEL_VALUES = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high", "CRITICAL": "critical"}
