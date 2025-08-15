"""
Exception hierarchy for the trading bot framework.

This module defines all custom exceptions used throughout the system.
CRITICAL: Never create duplicate exceptions. All prompts must import and use these exact classes.

USAGE RULE: ALL prompts must import these exceptions:
from src.core.exceptions import (
    TradingBotError, ExchangeError, RiskManagementError,
    ValidationError, ExecutionError, ModelError, DataError,
    StateConsistencyError, StateError, SynchronizationError, ConflictError,
    SecurityError
)
"""

from datetime import datetime, timezone
from typing import Any


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""

    def __init__(
        self, message: str, error_code: str | None = None, details: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)

    def __str__(self) -> str:
        """Return formatted error message with code and details."""
        error_str = f"{self.message}"
        if self.error_code:
            error_str += f" (Code: {self.error_code})"
        if self.details:
            error_str += f" - Details: {self.details}"
        return error_str


# Exchange-related exceptions
class ExchangeError(TradingBotError):
    """Exchange API and connection errors."""

    pass


class ExchangeConnectionError(ExchangeError):
    """Exchange connection failures."""

    pass


class ExchangeRateLimitError(ExchangeError):
    """Exchange rate limit violations."""

    pass


class ExchangeInsufficientFundsError(ExchangeError):
    """Insufficient funds for order."""

    pass


class ExchangeOrderError(ExchangeError):
    """General order-related exchange errors."""

    pass


class ExchangeAuthenticationError(ExchangeError):
    """Exchange authentication failures."""

    pass


# Risk management exceptions
class RiskManagementError(TradingBotError):
    """Risk management violations."""

    pass


class PositionLimitError(RiskManagementError):
    """Position size limit violations."""

    pass


class DrawdownLimitError(RiskManagementError):
    """Drawdown limit violations."""

    pass


class RiskCalculationError(RiskManagementError):
    """Risk calculation failures."""

    pass


class CapitalAllocationError(RiskManagementError):
    """Capital allocation violations."""

    pass


class CircuitBreakerTriggeredError(RiskManagementError):
    """Circuit breaker triggered, trading halted."""

    pass


class EmergencyStopError(RiskManagementError):
    """Emergency stop activation failed."""

    pass


# Data-related exceptions
class DataError(TradingBotError):
    """Data quality and processing errors."""

    pass


class DataValidationError(DataError):
    """Data validation failures."""

    pass


class DataSourceError(DataError):
    """External data source failures."""

    pass


class DataProcessingError(DataError):
    """Data processing and transformation errors."""

    pass


class DataCorruptionError(DataError):
    """Data corruption detected."""

    pass


# ML model exceptions
class ModelError(TradingBotError):
    """Machine learning model errors."""

    pass


class ModelLoadError(ModelError):
    """Model loading failures."""

    pass


class ModelInferenceError(ModelError):
    """Model inference failures."""

    pass


class ModelDriftError(ModelError):
    """Model drift detection."""

    pass


class ModelTrainingError(ModelError):
    """Model training failures."""

    pass


class ModelValidationError(ModelError):
    """Model validation failures."""

    pass


# Validation exceptions
class ValidationError(TradingBotError):
    """Input and schema validation errors."""

    pass


class ConfigurationError(ValidationError):
    """Configuration validation errors."""

    pass


class SchemaValidationError(ValidationError):
    """Schema validation failures."""

    pass


class InputValidationError(ValidationError):
    """Input parameter validation failures."""

    pass


# Execution exceptions
class ExecutionError(TradingBotError):
    """Order execution errors."""

    pass


class OrderRejectionError(ExecutionError):
    """Order rejection by exchange."""

    pass


class SlippageError(ExecutionError):
    """Excessive slippage detected."""

    pass


class ExecutionTimeoutError(ExecutionError):
    """Order execution timeout."""

    pass


class ExecutionPartialFillError(ExecutionError):
    """Partial order fill errors."""

    pass


# State consistency exceptions
class StateConsistencyError(TradingBotError):
    """State synchronization problems."""

    pass


class StateError(StateConsistencyError):
    """General state management errors."""

    pass


class StateCorruptionError(StateConsistencyError):
    """State data corruption detected."""

    pass


class StateSyncError(StateConsistencyError):
    """State synchronization failures."""

    pass


class StateLockError(StateConsistencyError):
    """State lock acquisition failures."""

    pass


class SynchronizationError(StateConsistencyError):
    """Real-time synchronization errors."""

    pass


class ConflictError(StateConsistencyError):
    """State conflict errors."""

    pass


# Security exceptions
class SecurityError(TradingBotError):
    """Security and authentication errors."""

    pass


class AuthenticationError(SecurityError):
    """Authentication failures."""

    pass


class AuthorizationError(SecurityError):
    """Authorization failures."""

    pass


class EncryptionError(SecurityError):
    """Encryption/decryption failures."""

    pass


class TokenValidationError(SecurityError):
    """Token validation failures."""

    pass


# Strategy exceptions
class StrategyError(TradingBotError):
    """Strategy execution errors."""

    pass


class StrategyConfigurationError(StrategyError):
    """Strategy configuration errors."""

    pass


class StrategyExecutionError(StrategyError):
    """Strategy execution failures."""

    pass


class SignalGenerationError(StrategyError):
    """Signal generation failures."""

    pass


# Arbitrage exceptions (P-013A)
class ArbitrageError(StrategyError):
    """Arbitrage strategy specific errors."""

    pass


class ArbitrageOpportunityError(ArbitrageError):
    """Arbitrage opportunity detection errors."""

    pass


class ArbitrageExecutionError(ArbitrageError):
    """Arbitrage execution failures."""

    pass


class ArbitrageTimingError(ArbitrageError):
    """Arbitrage timing constraint violations."""

    pass


# Database exceptions
class DatabaseError(TradingBotError):
    """Database operation errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection failures."""

    pass


class DatabaseQueryError(DatabaseError):
    """Database query failures."""

    pass


class DatabaseTransactionError(DatabaseError):
    """Database transaction failures."""

    pass


# Network and communication exceptions
class NetworkError(TradingBotError):
    """Network communication errors."""

    pass


class TimeoutError(NetworkError):
    """Network timeout errors."""

    pass


class ConnectionError(NetworkError):
    """Connection establishment failures."""

    pass


# Capital management exceptions
class CapitalManagementError(TradingBotError):
    """Capital management and allocation errors."""

    pass


class InsufficientCapitalError(CapitalManagementError):
    """Insufficient capital for allocation."""

    pass


class WithdrawalError(CapitalManagementError):
    """Withdrawal rule violations."""

    pass


class CurrencyError(CapitalManagementError):
    """Currency conversion and exposure errors."""

    pass


# Backtesting exceptions
class BacktestError(TradingBotError):
    """Backtesting execution errors."""

    pass


class BacktestDataError(BacktestError):
    """Backtesting data issues."""

    pass


class BacktestConfigurationError(BacktestError):
    """Backtesting configuration errors."""

    pass


# Optimization exceptions
class OptimizationError(TradingBotError):
    """Parameter and strategy optimization errors."""

    pass


class OptimizationConvergenceError(OptimizationError):
    """Optimization convergence failures."""

    pass


class OptimizationConstraintError(OptimizationError):
    """Optimization constraint violations."""

    pass


# Circuit breaker and retry exceptions
class CircuitBreakerOpen(TradingBotError):
    """Circuit breaker is open due to too many failures."""

    pass


class MaxRetriesExceeded(TradingBotError):
    """Maximum retry attempts exceeded."""

    pass


# Simulation and backtesting exceptions
class SimulationError(TradingBotError):
    """Simulation execution errors."""

    pass


# REVERSE INTEGRATION: Future prompts may add specific sub-exceptions but must
# extend these base classes, never replace them.
