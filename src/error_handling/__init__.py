"""
Error handling module for the trading bot framework.

This module provides comprehensive error handling, recovery, and resilience
framework for production-ready error management.

CRITICAL: This module integrates with P-001 core framework and P-002 database
layer and will be used by all subsequent prompts for robust error handling.
"""

from .connection_manager import ConnectionManager, ConnectionState
from .error_handler import ErrorContext, ErrorHandler, ErrorSeverity
from .pattern_analytics import ErrorPattern, ErrorPatternAnalytics
from .recovery_scenarios import (
    APIRateLimitRecovery,
    DataFeedInterruptionRecovery,
    ExchangeMaintenanceRecovery,
    NetworkDisconnectionRecovery,
    OrderRejectionRecovery,
    PartialFillRecovery,
)
from .state_monitor import StateMonitor
from src.core.exceptions import StateConsistencyError

__all__ = [
    # Core error handling
    "ErrorHandler",
    "ErrorSeverity",
    "ErrorContext",
    # Recovery scenarios
    "PartialFillRecovery",
    "NetworkDisconnectionRecovery",
    "ExchangeMaintenanceRecovery",
    "DataFeedInterruptionRecovery",
    "OrderRejectionRecovery",
    "APIRateLimitRecovery",
    # Connection management
    "ConnectionManager",
    "ConnectionState",
    # State monitoring
    "StateMonitor",
    "StateConsistencyError",
    # Pattern analytics
    "ErrorPatternAnalytics",
    "ErrorPattern",
]
