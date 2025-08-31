"""
Error handling module for the trading bot framework.

This module provides comprehensive error handling, recovery, and resilience
framework for production-ready error management.

CRITICAL: This module integrates with P-001 core framework and P-002 database
layer and will be used by all subsequent prompts for robust error handling.
"""

from src.core.exceptions import ErrorSeverity

from .connection_manager import ConnectionManager, ConnectionState
from .decorators import (
    FallbackStrategy,
    get_active_handler_count,
    shutdown_all_error_handlers,
    with_circuit_breaker,
    with_error_context,
    with_fallback,
    with_retry,
)
from .di_registration import configure_error_handling_di, register_error_handling_services
from .context import ErrorContext
from .error_handler import ErrorHandler
from .global_handler import GlobalErrorHandler
from .interfaces import (
    ErrorHandlingServiceInterface,
    ErrorPatternAnalyticsInterface,
    ErrorHandlerInterface,
    GlobalErrorHandlerInterface,
    ErrorHandlingServicePort,
)
from .pattern_analytics import ErrorPatternAnalytics
from .service_adapter import ErrorHandlingServiceAdapter, create_error_handling_service_adapter
from .recovery_scenarios import (
    APIRateLimitRecovery,
    DataFeedInterruptionRecovery,
    ExchangeMaintenanceRecovery,
    NetworkDisconnectionRecovery,
    OrderRejectionRecovery,
    PartialFillRecovery,
    RecoveryScenario,
)
from .service import ErrorHandlingService
from .state_monitor import StateMonitor

# Global error handler instance
_global_error_handler = None


def get_global_error_handler() -> GlobalErrorHandler | None:
    """
    Get the global error handler instance.

    Returns:
        The global error handler instance or None if not initialized
    """
    return _global_error_handler


def set_global_error_handler(handler: GlobalErrorHandler) -> None:
    """
    Set the global error handler instance.

    Args:
        handler: The global error handler to set
    """
    global _global_error_handler
    _global_error_handler = handler


__all__ = [
    "APIRateLimitRecovery",
    "ConnectionManager",
    "ConnectionState",
    "DataFeedInterruptionRecovery",
    "ErrorContext",
    "ErrorHandler",
    "ErrorHandlerInterface",
    "ErrorHandlingService",
    "ErrorHandlingServiceAdapter",
    "ErrorHandlingServiceInterface",
    "ErrorHandlingServicePort",
    "ErrorPatternAnalytics",
    "ErrorPatternAnalyticsInterface",
    "ErrorSeverity",
    "ExchangeMaintenanceRecovery",
    "FallbackStrategy",
    "GlobalErrorHandler",
    "GlobalErrorHandlerInterface",
    "NetworkDisconnectionRecovery",
    "OrderRejectionRecovery",
    "PartialFillRecovery",
    "RecoveryScenario",
    "StateMonitor",
    "configure_error_handling_di",
    "create_error_handling_service_adapter",
    "get_active_handler_count",
    "get_global_error_handler",
    "register_error_handling_services",
    "set_global_error_handler",
    "shutdown_all_error_handlers",
    "with_circuit_breaker",
    "with_error_context",
    "with_fallback",
    "with_retry",
]
