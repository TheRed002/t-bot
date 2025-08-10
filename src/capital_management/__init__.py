"""
Capital Management System Implementation (P-010A)

This module implements comprehensive capital allocation and fund management system
for the trading bot framework. It provides dynamic capital allocation across
strategies and exchanges with sophisticated risk management and protection mechanisms.

Key Components:
- CapitalAllocator: Dynamic capital allocation framework
- ExchangeDistributor: Multi-exchange capital distribution
- CurrencyManager: Multi-currency capital management
- FundFlowManager: Deposit/withdrawal management
- PositionSizer: Uses existing risk management position sizing

Dependencies:
- P-008+ (risk management)
- P-003+ (multi-exchange support)
- P-002 (portfolio tracking)
- P-001 (types, config)
- P-002A (error handling)
- P-007A (utils)

Author: Trading Bot Framework
Version: 1.0.0
"""

from .capital_allocator import CapitalAllocator
from .exchange_distributor import ExchangeDistributor
from .currency_manager import CurrencyManager
from .fund_flow_manager import FundFlowManager

__all__ = [
    'CapitalAllocator',
    'ExchangeDistributor',
    'CurrencyManager',
    'FundFlowManager'
]

__version__ = "1.0.0"
__author__ = "Trading Bot Framework"
