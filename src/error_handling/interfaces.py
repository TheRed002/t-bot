"""
Service layer interfaces for error handling module.

These interfaces define the contracts between different layers of the error handling system,
ensuring loose coupling and proper service layer architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from src.core.base.interfaces import HealthCheckResult


@runtime_checkable 
class ErrorHandlingServiceInterface(Protocol):
    """Protocol for error handling service layer."""

    async def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: dict[str, Any] | None = None,
        recovery_strategy: Any | None = None,
    ) -> dict[str, Any]:
        """Handle an error with recovery strategy."""
        ...

    async def handle_global_error(
        self, error: Exception, context: dict[str, Any] | None = None, severity: str = "error"
    ) -> dict[str, Any]:
        """Handle error through global error handler."""
        ...

    async def validate_state_consistency(self, component: str = "all") -> dict[str, Any]:
        """Validate state consistency for specified component."""
        ...

    async def get_error_patterns(self) -> dict[str, Any]:
        """Get current error patterns and analytics."""
        ...

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on error handling service."""
        ...


@runtime_checkable
class ErrorPatternAnalyticsInterface(Protocol):
    """Protocol for error pattern analytics service."""

    def add_error_event(self, error_context: dict[str, Any]) -> None:
        """Add an error event to analytics."""
        ...

    async def add_batch_error_events(self, error_contexts: list[dict[str, Any]]) -> None:
        """Add multiple error events in batch."""
        ...

    def get_pattern_summary(self) -> dict[str, Any]:
        """Get summary of detected patterns."""
        ...

    def get_correlation_summary(self) -> dict[str, Any]:
        """Get summary of error correlations."""
        ...

    def get_trend_summary(self) -> dict[str, Any]:
        """Get summary of error trends."""
        ...


@runtime_checkable
class ErrorHandlerInterface(Protocol):
    """Protocol for error handler components."""

    async def handle_error(
        self,
        error: Exception,
        context: Any,
        recovery_strategy: Any | None = None,
    ) -> bool:
        """Handle error with recovery strategy."""
        ...

    def classify_error(self, error: Exception) -> Any:
        """Classify error severity."""
        ...

    def create_error_context(
        self, error: Exception, component: str, operation: str, **kwargs
    ) -> Any:
        """Create error context for tracking."""
        ...


@runtime_checkable
class GlobalErrorHandlerInterface(Protocol):
    """Protocol for global error handler."""

    async def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "error",
    ) -> dict[str, Any]:
        """Handle error globally."""
        ...

    def get_statistics(self) -> dict[str, Any]:
        """Get error handling statistics."""
        ...


class ErrorHandlingServicePort(ABC):
    """Port interface for error handling service (hexagonal architecture)."""

    @abstractmethod
    async def process_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process an error through the service layer."""
        pass

    @abstractmethod
    async def analyze_error_patterns(self) -> dict[str, Any]:
        """Analyze current error patterns."""
        pass

    @abstractmethod
    async def validate_system_state(self, component: str = "all") -> dict[str, Any]:
        """Validate system state consistency."""
        pass


class ErrorHandlingRepositoryPort(ABC):
    """Repository port for error handling data persistence."""

    @abstractmethod
    async def store_error_event(self, error_data: dict[str, Any]) -> str:
        """Store error event data."""
        pass

    @abstractmethod
    async def retrieve_error_patterns(
        self, component: str | None = None, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Retrieve error patterns for analysis."""
        pass

    @abstractmethod
    async def update_error_statistics(self, stats: dict[str, Any]) -> None:
        """Update error statistics."""
        pass


__all__ = [
    "ErrorHandlingServiceInterface",
    "ErrorPatternAnalyticsInterface", 
    "ErrorHandlerInterface",
    "GlobalErrorHandlerInterface",
    "ErrorHandlingServicePort",
    "ErrorHandlingRepositoryPort",
]