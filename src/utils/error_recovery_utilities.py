"""
Centralized error recovery utilities for state management.

This module provides common error recovery patterns, data structures, and utilities
used across the state management system to eliminate duplication.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from .validation_utilities import ErrorType


class RecoveryStrategy(Enum):
    """Recovery strategy options."""

    RETRY = "retry"
    ROLLBACK = "rollback"
    FALLBACK = "fallback"
    SKIP = "skip"
    MANUAL = "manual"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    error_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation: str = ""
    component: str = ""

    # Error details
    error_type: ErrorType = ErrorType.UNKNOWN
    error_message: str = ""
    exception: Exception | None = None

    # Operation context
    session_id: str = ""
    transaction_id: str = ""
    correlation_id: str = ""

    # Retry information
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Recovery state
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    recovery_data: dict[str, Any] = field(default_factory=dict)
    fallback_executed: bool = False

    # Monitoring
    resolved: bool = False
    resolution_time: datetime | None = None
    resolution_method: str = ""


@dataclass
class RecoveryCheckpoint:
    """Recovery checkpoint for rollback operations."""

    checkpoint_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # State snapshot
    state_before: dict[str, Any] = field(default_factory=dict)
    metadata_before: dict[str, Any] = field(default_factory=dict)

    # Database state
    database_state: dict[str, Any] = field(default_factory=dict)
    cache_state: dict[str, Any] = field(default_factory=dict)

    # Transaction info
    transaction_id: str = ""
    savepoint_name: str = ""

    # Recovery info
    can_rollback: bool = True
    rollback_instructions: list[str] = field(default_factory=list)


class BaseErrorRecovery:
    """
    Base error recovery system providing common recovery patterns.

    This class provides standardized error handling and recovery mechanisms
    that can be used across the state management system.
    """

    def __init__(self, component_name: str = ""):
        """Initialize error recovery system."""
        self.component_name = component_name

        # Error tracking
        self._active_errors: dict[str, ErrorContext] = {}
        self._recovery_checkpoints: dict[str, RecoveryCheckpoint] = {}
        self._error_history: list[ErrorContext] = []

        # Configuration
        self.max_history_size = 1000
        self.default_retry_count = 3
        self.default_retry_delay = 1.0
        self.exponential_backoff = True

        # Error handlers
        self._error_handlers: dict[ErrorType, Callable] = {
            ErrorType.DATABASE_CONNECTION: self._handle_database_connection_error,
            ErrorType.DATABASE_INTEGRITY: self._handle_database_integrity_error,
            ErrorType.DATABASE_TIMEOUT: self._handle_database_timeout_error,
            ErrorType.REDIS_CONNECTION: self._handle_redis_connection_error,
            ErrorType.REDIS_TIMEOUT: self._handle_redis_timeout_error,
            ErrorType.DATA_CORRUPTION: self._handle_data_corruption_error,
            ErrorType.DISK_SPACE: self._handle_disk_space_error,
            ErrorType.PERMISSION: self._handle_permission_error,
            ErrorType.VALIDATION: self._handle_validation_error,
            ErrorType.CONCURRENCY: self._handle_concurrency_error,
            ErrorType.UNKNOWN: self._handle_unknown_error,
        }

        # Metrics
        self.error_counts = {error_type: 0 for error_type in ErrorType}
        self.recovery_success_count = 0
        self.recovery_failure_count = 0

    def classify_error(self, exception: Exception) -> ErrorType:
        """Classify exception into error type."""
        from .validation_utilities import classify_error_type

        return classify_error_type(exception)

    async def create_recovery_checkpoint(
        self, operation: str, state_data: dict[str, Any] | None = None, **context
    ) -> str:
        """
        Create a recovery checkpoint before risky operations.

        Args:
            operation: Operation being performed
            state_data: Current state data
            **context: Additional context

        Returns:
            Checkpoint ID
        """
        checkpoint = RecoveryCheckpoint(
            timestamp=datetime.now(timezone.utc),
            state_before=state_data.copy() if state_data else {},
            transaction_id=context.get("transaction_id", ""),
            savepoint_name=context.get("savepoint_name", ""),
        )

        # Store checkpoint
        self._recovery_checkpoints[checkpoint.checkpoint_id] = checkpoint

        return checkpoint.checkpoint_id

    async def handle_error(
        self, exception: Exception, operation: str, checkpoint_id: str | None = None, **context
    ) -> ErrorContext:
        """
        Handle an error with automatic recovery attempt.

        Args:
            exception: Exception that occurred
            operation: Operation that failed
            checkpoint_id: Associated checkpoint ID
            **context: Additional context

        Returns:
            Error context with resolution info
        """
        # Create error context
        error_context = ErrorContext(
            timestamp=datetime.now(timezone.utc),
            operation=operation,
            component=self.component_name,
            error_type=self.classify_error(exception),
            error_message=str(exception),
            exception=exception,
            session_id=context.get("session_id", ""),
            transaction_id=context.get("transaction_id", ""),
            correlation_id=context.get("correlation_id", ""),
            retry_count=context.get("retry_count", 0),
            max_retries=context.get("max_retries", self.default_retry_count),
            retry_delay=context.get("retry_delay", self.default_retry_delay),
        )

        # Update error counts
        self.error_counts[error_context.error_type] += 1

        # Store active error
        self._active_errors[error_context.error_id] = error_context

        # Get recovery checkpoint if provided
        checkpoint = None
        if checkpoint_id:
            checkpoint = self._recovery_checkpoints.get(checkpoint_id)
            if checkpoint:
                error_context.recovery_data = checkpoint.state_before.copy()

        # Attempt recovery
        recovery_success = await self._attempt_recovery(error_context, checkpoint)

        if recovery_success:
            self.recovery_success_count += 1
            error_context.resolved = True
            error_context.resolution_time = datetime.now(timezone.utc)
        else:
            self.recovery_failure_count += 1

        # Move to history
        self._error_history.append(error_context)
        if error_context.error_id in self._active_errors:
            del self._active_errors[error_context.error_id]

        # Cleanup old history
        if len(self._error_history) > self.max_history_size:
            self._error_history = self._error_history[-self.max_history_size :]

        return error_context

    async def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback to a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint to rollback to

        Returns:
            True if rollback successful
        """
        try:
            checkpoint = self._recovery_checkpoints.get(checkpoint_id)
            if not checkpoint:
                return False

            if not checkpoint.can_rollback:
                return False

            # Execute rollback instructions
            for instruction in checkpoint.rollback_instructions:
                try:
                    # Implementation would execute specific rollback commands
                    pass
                except Exception:
                    # Log warning but continue
                    pass

            return True

        except Exception:
            return False

    async def _attempt_recovery(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """
        Attempt to recover from error using appropriate strategy.

        Args:
            error_context: Error context information
            checkpoint: Recovery checkpoint if available

        Returns:
            True if recovery successful
        """
        try:
            # Get error handler
            handler = self._error_handlers.get(error_context.error_type, self._handle_unknown_error)

            # Attempt recovery
            recovery_result = await handler(error_context, checkpoint)

            if recovery_result:
                error_context.resolution_method = f"auto_recovery_{error_context.error_type.value}"
                return True
            else:
                return False

        except Exception:
            return False

    # Error type handlers

    async def _handle_database_connection_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle database connection errors."""
        if error_context.retry_count < error_context.max_retries:
            # Wait with exponential backoff
            delay = (
                error_context.retry_delay * (2**error_context.retry_count)
                if self.exponential_backoff
                else error_context.retry_delay
            )
            await asyncio.sleep(delay)
            return True
        return False

    async def _handle_database_integrity_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle database integrity errors."""
        if checkpoint:
            return await self.rollback_to_checkpoint(checkpoint.checkpoint_id)
        return False

    async def _handle_database_timeout_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle database timeout errors."""
        if error_context.retry_count < error_context.max_retries:
            # Increase delay for timeouts
            delay = error_context.retry_delay * 2 * (error_context.retry_count + 1)
            await asyncio.sleep(delay)
            return True
        return False

    async def _handle_redis_connection_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle Redis connection errors."""
        if error_context.retry_count < error_context.max_retries:
            delay = error_context.retry_delay * (error_context.retry_count + 1)
            await asyncio.sleep(delay)
            return True
        return False

    async def _handle_redis_timeout_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle Redis timeout errors."""
        return await self._handle_redis_connection_error(error_context, checkpoint)

    async def _handle_data_corruption_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle data corruption errors."""
        if checkpoint:
            rollback_success = await self.rollback_to_checkpoint(checkpoint.checkpoint_id)
            if rollback_success:
                error_context.recovery_strategy = RecoveryStrategy.MANUAL
                return True
        return False

    async def _handle_disk_space_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle disk space errors."""
        error_context.recovery_strategy = RecoveryStrategy.MANUAL
        return False

    async def _handle_permission_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle permission errors."""
        error_context.recovery_strategy = RecoveryStrategy.MANUAL
        return False

    async def _handle_validation_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle validation errors."""
        if checkpoint:
            return await self.rollback_to_checkpoint(checkpoint.checkpoint_id)
        return False

    async def _handle_concurrency_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle concurrency errors."""
        if error_context.retry_count < error_context.max_retries:
            # Add random jitter to reduce collision probability
            import random

            jitter = random.uniform(0.1, 1.0)
            delay = error_context.retry_delay * (error_context.retry_count + 1) + jitter
            await asyncio.sleep(delay)
            return True
        return False

    async def _handle_unknown_error(
        self, error_context: ErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle unknown errors."""
        if error_context.retry_count < min(2, error_context.max_retries):
            await asyncio.sleep(error_context.retry_delay)
            return True
        return False

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error handling statistics."""
        return {
            "error_counts_by_type": {
                error_type.value: count for error_type, count in self.error_counts.items()
            },
            "total_errors": sum(self.error_counts.values()),
            "recovery_success_rate": (
                self.recovery_success_count
                / max(1, self.recovery_success_count + self.recovery_failure_count)
                * 100
            ),
            "active_errors": len(self._active_errors),
            "active_checkpoints": len(self._recovery_checkpoints),
            "error_history_size": len(self._error_history),
        }
