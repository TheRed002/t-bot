"""
Risk Management Framework for P-008 and P-009.

This module provides comprehensive risk management capabilities including:
- Position sizing algorithms
- Portfolio risk monitoring
- Risk limit enforcement
- Real-time risk metrics calculation
- Circuit breakers and emergency controls

CRITICAL: This module integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

from .base import BaseRiskManager
from .circuit_breakers import BaseCircuitBreaker, CircuitBreakerManager
from .emergency_controls import EmergencyControls
from .portfolio_limits import PortfolioLimits
from .position_sizing import PositionSizer
from .risk_manager import RiskManager
from .risk_metrics import RiskCalculator

__all__ = [
    "BaseCircuitBreaker",
    "BaseRiskManager",
    "CircuitBreakerManager",
    "EmergencyControls",
    "PortfolioLimits",
    "PositionSizer",
    "RiskCalculator",
    "RiskManager",
]

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Framework"
__description__ = "Risk Management Framework with Circuit Breakers (P-008, P-009)"
