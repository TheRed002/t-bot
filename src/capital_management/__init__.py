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

# Export capital-specific types for convenience
# Import DI registration function (no circular dependencies)
from src.capital_management.di_registration import register_capital_management_services

# Import service interfaces first (no circular dependencies)
from src.capital_management.interfaces import (
    AbstractCapitalService,
    AbstractCurrencyManagementService,
    AbstractExchangeDistributionService,
    AbstractFundFlowManagementService,
    CapitalServiceProtocol,
    CurrencyManagementServiceProtocol,
    ExchangeDistributionServiceProtocol,
    FundFlowManagementServiceProtocol,
)
from src.core.types.capital import (
    CapitalCurrencyExposure,
    CapitalExchangeAllocation,
    CapitalFundFlow,
    ExtendedCapitalProtection,
    ExtendedWithdrawalRule,
)


# Use lazy imports for concrete implementations to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    imports = {
        "CapitalAllocator": "src.capital_management.capital_allocator",
        "CurrencyManager": "src.capital_management.currency_manager",
        "ExchangeDistributor": "src.capital_management.exchange_distributor",
        "FundFlowManager": "src.capital_management.fund_flow_manager",
        "CapitalService": "src.capital_management.service",
        "CapitalAllocatorFactory": "src.capital_management.factory",
        "CapitalManagementFactory": "src.capital_management.factory",
        "CapitalServiceFactory": "src.capital_management.factory",
        "CurrencyManagerFactory": "src.capital_management.factory",
        "ExchangeDistributorFactory": "src.capital_management.factory",
        "FundFlowManagerFactory": "src.capital_management.factory",
        "CapitalRepository": "src.capital_management.repository",
        "AuditRepository": "src.capital_management.repository",
    }

    if name in imports:
        module = __import__(imports[name], fromlist=[name])
        return getattr(module, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "AbstractCapitalService",
    "AbstractCurrencyManagementService",
    "AbstractExchangeDistributionService",
    "AbstractFundFlowManagementService",
    "AuditRepository",
    "CapitalAllocator",
    "CapitalAllocatorFactory",
    "CapitalCurrencyExposure",
    "CapitalExchangeAllocation",
    "CapitalFundFlow",
    "CapitalManagementFactory",
    "CapitalRepository",
    "CapitalService",
    "CapitalServiceFactory",
    "CapitalServiceProtocol",
    "CurrencyManagementServiceProtocol",
    "CurrencyManager",
    "CurrencyManagerFactory",
    "ExchangeDistributionServiceProtocol",
    "ExchangeDistributor",
    "ExchangeDistributorFactory",
    "ExtendedCapitalProtection",
    "ExtendedWithdrawalRule",
    "FundFlowManagementServiceProtocol",
    "FundFlowManager",
    "FundFlowManagerFactory",
    "register_capital_management_services",
]

__version__ = "1.0.0"
__author__ = "Trading Bot Framework"
