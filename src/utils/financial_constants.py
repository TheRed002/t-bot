"""
Financial Constants for Trading Systems.

Centralized constants used across financial calculations to ensure consistency
and avoid magic numbers throughout the codebase.
"""

# Trading calendar constants
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_HOURS_PER_DAY = 24  # Crypto markets
TRADING_MINUTES_PER_HOUR = 60

# Risk-free rates (annual)
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual
US_TREASURY_3M_RATE = 0.015  # 1.5% annual (example)
US_TREASURY_10Y_RATE = 0.025  # 2.5% annual (example)

# Statistical constants
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95% confidence
BOOTSTRAP_SAMPLES = 1000
VAR_PERCENTILE = 5  # For 95% VaR (5th percentile)
CVAR_PERCENTILE = 5  # For 95% CVaR (5th percentile)

# Performance benchmarks
SHARPE_RATIO_EXCELLENT = 2.0
SHARPE_RATIO_GOOD = 1.0
SHARPE_RATIO_POOR = 0.5

# Drawdown thresholds
MAX_ACCEPTABLE_DRAWDOWN = 0.20  # 20%
WARNING_DRAWDOWN = 0.10  # 10%

# Position sizing defaults
DEFAULT_POSITION_SIZE_PCT = 0.02  # 2% per position
MAX_POSITION_SIZE_PCT = 0.10  # 10% maximum per position
MAX_PORTFOLIO_EXPOSURE = 1.0  # 100% of capital

# Commission and slippage defaults
DEFAULT_COMMISSION_RATE = 0.001  # 0.1%
DEFAULT_SLIPPAGE_RATE = 0.0005  # 0.05%
CRYPTO_MAKER_FEE = 0.0001  # 0.01% (typical maker fee)
CRYPTO_TAKER_FEE = 0.0001  # 0.01% (typical taker fee)

# Market impact constants
DEFAULT_MARKET_IMPACT_FACTOR = 0.0001  # 0.01%
LIQUIDITY_FACTOR = 0.1  # 10% of volume available

# Time constants for analysis
SECONDS_PER_DAY = 86400
MILLISECONDS_PER_SECOND = 1000
MICROSECONDS_PER_SECOND = 1000000

# Monte Carlo simulation defaults
DEFAULT_MONTE_CARLO_RUNS = 1000
MAX_MONTE_CARLO_RUNS = 10000
MIN_MONTE_CARLO_RUNS = 100
