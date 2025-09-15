"""
Capital Management Constants.

This module defines constants used throughout the capital_management module
to avoid magic numbers and provide centralized configuration.
"""

from decimal import Decimal

# Default scoring values
DEFAULT_EXCHANGE_SCORE = Decimal("0.5")
DEFAULT_LIQUIDITY_SCORE = Decimal("0.5")
DEFAULT_FEE_EFFICIENCY = Decimal("0.5")
DEFAULT_RELIABILITY_SCORE = Decimal("0.5")

# Risk thresholds
DEFAULT_HEDGING_THRESHOLD = Decimal("0.1")
DEFAULT_HEDGE_RATIO = Decimal("0.5")
MIN_HEDGE_AMOUNT = Decimal("0.01")
MIN_CHANGE_THRESHOLD = Decimal("0.01")

# Performance thresholds
LOW_ALLOCATION_RATIO_THRESHOLD = Decimal("0.3")
LOW_UTILIZATION_THRESHOLD = Decimal("0.5")

# Fee calculations
DEFAULT_CONVERSION_FEE_RATE = Decimal("0.001")  # 0.1%

# History and caching limits
DEFAULT_MAX_FLOW_HISTORY = 1000
DEFAULT_MAX_SLIPPAGE_HISTORY = 100
DEFAULT_MAX_RATE_HISTORY = 100
DEFAULT_CACHE_TTL_SECONDS = 300  # 5 minutes
RATE_CALCULATION_LOOKBACK_DAYS = 30

# Capital protection thresholds
EMERGENCY_RESERVE_PCT = Decimal("0.1")  # 10%
MAX_ALLOCATION_PCT = Decimal("0.2")  # 20%
MAX_DAILY_REALLOCATION_PCT = Decimal("0.1")  # 10%
MAX_DAILY_LOSS_PCT = Decimal("0.05")  # 5%
MAX_WITHDRAWAL_PCT = Decimal("0.2")  # 20%
PROFIT_LOCK_PCT = Decimal("0.5")  # 50%

# Minimum amounts
MIN_DEPOSIT_AMOUNT = Decimal("100")
MIN_WITHDRAWAL_AMOUNT = Decimal("100")
MIN_EXCHANGE_BALANCE = Decimal("1000")
DEFAULT_PROFIT_THRESHOLD = Decimal("1000")

# Service configurations
DEFAULT_FUND_FLOW_COOLDOWN_MINUTES = 60
DEFAULT_PERFORMANCE_WINDOW_DAYS = 30
COMPOUND_FREQUENCY_DAYS = 30

# Reliability scoring
RELIABILITY_BONUS_PER_SERVICE = Decimal("0.075")
CONNECTIVITY_PENALTY = Decimal("0.1")
STATUS_CHECK_BONUS = Decimal("0.2")

# Exchange variance simulation
SLIPPAGE_VARIANCE_RANGE = Decimal("0.0002")

# Circuit breaker settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60

# Error thresholds
HIGH_ERROR_RATE_THRESHOLD = Decimal("0.1")  # 10%
CRITICAL_ERROR_RATE_THRESHOLD = Decimal("0.2")  # 20%

# Risk assessment thresholds
HIGH_PORTFOLIO_EXPOSURE_THRESHOLD = Decimal("0.8")  # 80%

# Performance calculation constants
PERFORMANCE_MULTIPLIER_MAX = Decimal("2.0")
NEUTRAL_EFFICIENCY = Decimal("0.5")
DEGRADED_CAPITAL_RATIO = Decimal("0.5")
AVAILABLE_CAPITAL_RATIO = Decimal("0.8")
EMERGENCY_CAPITAL_RATIO = Decimal("0.3")

# Allocation limits
MAX_ALLOCATION_EFFICIENCY_COUNT = 100
MAX_PERFORMANCE_METRICS_COUNT = 10000

# Default amounts and limits
DEFAULT_TOTAL_CAPITAL = Decimal("100000")
DEFAULT_MAX_QUEUE_SIZE = 1000
DEFAULT_BASE_CURRENCY = "USDT"
DEFAULT_EXCHANGE = "binance"

# Connection and timeout settings
DEFAULT_CONNECTION_TIMEOUT = 8.0
DEFAULT_CACHE_OPERATION_TIMEOUT = 5.0
DEFAULT_TIME_SERIES_TIMEOUT = 3.0
DEFAULT_CLEANUP_TIMEOUT = 10.0
MAX_RATE_HISTORY_PER_SYMBOL = 50
MAX_ITEMS_PER_CLEANUP_KEY = 50
RATE_HISTORY_MAX_AGE_DAYS = 7

# Decimal precision settings
FINANCIAL_DECIMAL_PRECISION = 28

# Processing limits
MAX_CONCURRENT_OPERATIONS = 10
EMERGENCY_THRESHOLD_PCT = Decimal("0.02")  # 2%

# Configuration defaults
DEFAULT_MAX_WEEKLY_LOSS_PCT = Decimal("0.10")  # 10%
DEFAULT_MAX_MONTHLY_LOSS_PCT = Decimal("0.20")  # 20%
DEFAULT_EMERGENCY_RESERVE_PCT = Decimal("0.1")  # 10%

# Default supported currencies
DEFAULT_SUPPORTED_CURRENCIES = ["USDT", "BTC", "ETH"]

# Distribution configuration
DEFAULT_MAX_ALLOCATION_PCT = Decimal("0.3")  # 30% max allocation per exchange
DEFAULT_REBALANCE_THRESHOLD = Decimal("0.05")  # 5% threshold for rebalancing
DEFAULT_MIN_REBALANCE_INTERVAL_HOURS = 1  # Minimum 1 hour between rebalances
DEFAULT_LIQUIDITY_WEIGHT = Decimal("0.4")
DEFAULT_FEE_WEIGHT = Decimal("0.3")
DEFAULT_RELIABILITY_WEIGHT = Decimal("0.3")
DEFAULT_EXCHANGE_TIMEOUT_SECONDS = 15.0  # 15 second timeout per exchange
DEFAULT_OPERATION_TIMEOUT_SECONDS = 3.0  # 3 second timeout for operations
DEFAULT_FEE_QUERY_TIMEOUT_SECONDS = 3.0  # 3 second timeout for fee queries
DEFAULT_MAX_SLIPPAGE_FACTOR = Decimal("200")  # Used in fee efficiency calculation
DEFAULT_VaR_CONFIDENCE_MULTIPLIER = Decimal("1.645")  # 95% confidence level

# Percentage calculation multiplier
PERCENTAGE_MULTIPLIER = Decimal("100")

# Default exchange scores (fallback values)
EXCHANGE_LIQUIDITY_SCORES = {
    "binance": Decimal("0.9"),  # High liquidity
}

EXCHANGE_FEE_EFFICIENCIES = {
    "binance": Decimal("0.8"),  # ~0.1% fees
    "okx": Decimal("0.7"),
    "coinbase": Decimal("0.6"),  # ~0.2% fees
}

EXCHANGE_RELIABILITY_SCORES = {
    "binance": Decimal("0.95"),  # Highly reliable
    "okx": Decimal("0.05"),  # Established
    "coinbase": Decimal("0.1"),  # Well-established
}

EXCHANGE_SLIPPAGE_DEFAULTS = {
    "binance": Decimal("0.0005"),  # 0.05% - high liquidity
    "okx": Decimal("0.001"),  # 0.1% - medium liquidity
}

# Missing exchange scores for constants
EXCHANGE_LIQUIDITY_SCORES.update(
    {
        "okx": Decimal("0.7"),  # Medium liquidity
        "coinbase": Decimal("0.8"),  # Good liquidity
    }
)

EXCHANGE_FEE_EFFICIENCIES.update(
    {
        "okx": Decimal("0.7"),  # ~0.15% fees
    }
)
