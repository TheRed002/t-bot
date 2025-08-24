"""
Error handling module for the trading bot framework.

This module provides comprehensive error handling, recovery, and resilience
framework for production-ready error management.

CRITICAL: This module integrates with P-001 core framework and P-002 database
layer and will be used by all subsequent prompts for robust error handling.
"""

from .connection_manager import ConnectionManager, ConnectionState
from .context import ErrorSeverity
from .decorators import (
    FallbackStrategy,
    with_circuit_breaker,
    with_error_context,
    with_fallback,
    with_retry,
)
from .error_handler import ErrorContext, ErrorHandler
from .global_handler import get_global_error_handler
from .pattern_analytics import ErrorPatternAnalytics
from .recovery_scenarios import (
    APIRateLimitRecovery,
    DataFeedInterruptionRecovery,
    ExchangeMaintenanceRecovery,
    NetworkDisconnectionRecovery,
    OrderRejectionRecovery,
    PartialFillRecovery,
    RecoveryScenario,
)
from .state_monitor import StateMonitor

__all__ = [
    "APIRateLimitRecovery",
    "ConnectionManager",
    "ConnectionState",
    "DataFeedInterruptionRecovery",
    "ErrorContext",
    "ErrorHandler",
    "ErrorPatternAnalytics",
    "ErrorSeverity",
    "ExchangeMaintenanceRecovery",
    "FallbackStrategy",
    "NetworkDisconnectionRecovery",
    "OrderRejectionRecovery",
    "PartialFillRecovery",
    "RecoveryScenario",
    "StateMonitor",
    "get_global_error_handler",
    "with_circuit_breaker",
    "with_error_context",
    "with_fallback",
    "with_retry",
]
