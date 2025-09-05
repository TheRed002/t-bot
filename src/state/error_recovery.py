"""
Error handling and recovery mechanisms for state persistence operations.

This module provides comprehensive error recovery, rollback capabilities,
and data integrity validation for all state persistence operations.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.exc import IntegrityError, OperationalError, TimeoutError as SQLTimeoutError

from src.core.exceptions import StateError
from src.utils.error_recovery_utilities import (
    BaseErrorRecovery,
    ErrorContext,
    RecoveryCheckpoint,
    RecoveryStrategy,
)
from src.utils.validation_utilities import ErrorType, classify_error_type


# Note: ErrorType and RecoveryStrategy moved to src.utils.validation_utilities and error_recovery_utilities


# Note: StateErrorContext and RecoveryCheckpoint moved to src.utils.error_recovery_utilities
# Use ErrorContext and RecoveryCheckpoint from centralized utilities

# Type aliases for backward compatibility
StateErrorContext = ErrorContext


class StateErrorRecovery(BaseErrorRecovery):
    """
    State-specific error recovery system extending base recovery functionality.

    Provides state persistence operations with comprehensive error recovery,
    building upon the centralized error recovery utilities.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize state-specific error recovery system."""
        super().__init__(component_name="StateErrorRecovery")
        self.logger = logger or logging.getLogger(__name__)

    # Note: classify_error() method now inherited from BaseErrorRecovery

    async def create_recovery_checkpoint(
        self,
        operation: str,
        state_type: str,
        state_id: str,
        current_state: dict[str, Any] | None = None,
        session: AsyncSession | None = None,
        **context,
    ) -> str:
        """
        Create a recovery checkpoint before risky operations.

        Args:
            operation: Operation being performed
            state_type: Type of state
            state_id: State identifier
            current_state: Current state data
            session: Database session
            **context: Additional context

        Returns:
            Checkpoint ID
        """
        checkpoint = RecoveryCheckpoint(
            timestamp=datetime.now(timezone.utc),
            state_before=current_state.copy() if current_state else {},
            transaction_id=context.get("transaction_id", ""),
            savepoint_name=context.get("savepoint_name", ""),
        )

        # Capture database state if session provided
        if session:
            try:
                # Could capture current transaction state, locks, etc.
                checkpoint.database_state = {
                    "transaction_active": session.in_transaction(),
                }
            except Exception as e:
                self.logger.warning(f"Failed to capture database state: {e}")
            finally:
                # Ensure session is properly managed - session should be closed by caller
                pass

        # Store checkpoint
        self._recovery_checkpoints[checkpoint.checkpoint_id] = checkpoint

        self.logger.debug(
            f"Created recovery checkpoint: {checkpoint.checkpoint_id}",
            extra={
                "operation": operation,
                "state_type": state_type,
                "state_id": state_id,
            },
        )

        return checkpoint.checkpoint_id

    async def handle_error(
        self,
        exception: Exception,
        operation: str,
        state_type: str = "",
        state_id: str = "",
        checkpoint_id: str | None = None,
        **context,
    ) -> StateErrorContext:
        """
        Handle an error with automatic recovery attempt.

        Args:
            exception: Exception that occurred
            operation: Operation that failed
            state_type: State type involved
            state_id: State ID involved
            checkpoint_id: Associated checkpoint ID
            **context: Additional context

        Returns:
            Error context with resolution info
        """
        # Create error context
        error_context = StateErrorContext(
            timestamp=datetime.now(timezone.utc),
            operation=operation,
            error_type=self.classify_error(exception),
            error_message=str(exception),
            exception=exception,
            session_id=context.get("session_id", ""),
            transaction_id=context.get("transaction_id", ""),
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
                error_context.rollback_data = checkpoint.state_before.copy()

        self.logger.error(
            f"Error occurred in {operation}: {exception}",
            extra={
                "error_id": error_context.error_id,
                "error_type": error_context.error_type.value,
                "state_type": state_type,
                "state_id": state_id,
                "checkpoint_id": checkpoint_id,
            },
            exc_info=True,
        )

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

    async def _attempt_recovery(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
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
                self.logger.info(
                    f"Successfully recovered from error: {error_context.error_id}",
                    extra={"recovery_method": error_context.resolution_method},
                )
                return True
            else:
                self.logger.warning(f"Failed to recover from error: {error_context.error_id}")
                return False

        except Exception as recovery_error:
            self.logger.error(
                f"Recovery attempt failed: {recovery_error}",
                extra={"original_error_id": error_context.error_id},
                exc_info=True,
            )
            return False

    async def rollback_to_checkpoint(
        self, checkpoint_id: str, session: AsyncSession | None = None
    ) -> bool:
        """
        Rollback to a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint to rollback to
            session: Database session for rollback

        Returns:
            True if rollback successful
        """
        try:
            checkpoint = self._recovery_checkpoints.get(checkpoint_id)
            if not checkpoint:
                self.logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False

            if not checkpoint.can_rollback:
                self.logger.warning(f"Checkpoint cannot be rolled back: {checkpoint_id}")
                return False

            # Rollback database transaction
            if session and checkpoint.savepoint_name:
                try:
                    await session.rollback()
                    self.logger.info(
                        f"Rolled back database transaction: {checkpoint.savepoint_name}"
                    )
                except Exception as e:
                    self.logger.error(f"Database rollback failed: {e}")
                    return False

            # Execute rollback instructions
            for instruction in checkpoint.rollback_instructions:
                try:
                    # Could implement specific rollback commands
                    self.logger.debug(f"Executing rollback instruction: {instruction}")
                except Exception as e:
                    self.logger.warning(f"Rollback instruction failed: {instruction}: {e}")

            self.logger.info(f"Successfully rolled back to checkpoint: {checkpoint_id}")
            return True

        except Exception as e:
            self.logger.error(f"Rollback to checkpoint failed: {e}", exc_info=True)
            return False

    # Error type handlers

    async def _handle_database_connection_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
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
            return True  # Indicate retry should be attempted
        return False

    async def _handle_database_integrity_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle database integrity errors."""
        # For integrity errors, we usually need to rollback
        if checkpoint:
            return await self.rollback_to_checkpoint(checkpoint.checkpoint_id)
        return False

    async def _handle_database_timeout_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle database timeout errors."""
        if error_context.retry_count < error_context.max_retries:
            # Increase delay for timeouts
            delay = error_context.retry_delay * 2 * (error_context.retry_count + 1)
            await asyncio.sleep(delay)
            return True
        return False

    async def _handle_redis_connection_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle Redis connection errors."""
        # Redis errors are often transient
        if error_context.retry_count < error_context.max_retries:
            delay = error_context.retry_delay * (error_context.retry_count + 1)
            await asyncio.sleep(delay)
            return True
        return False

    async def _handle_redis_timeout_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle Redis timeout errors."""
        return await self._handle_redis_connection_error(error_context, checkpoint)

    async def _handle_data_corruption_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle data corruption errors."""
        # Data corruption usually requires rollback and manual intervention
        if checkpoint:
            rollback_success = await self.rollback_to_checkpoint(checkpoint.checkpoint_id)
            if rollback_success:
                error_context.recovery_strategy = RecoveryStrategy.MANUAL
                return True
        return False

    async def _handle_disk_space_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle disk space errors."""
        # Disk space errors require manual intervention
        error_context.recovery_strategy = RecoveryStrategy.MANUAL
        return False

    async def _handle_permission_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle permission errors."""
        # Permission errors usually require manual fixing
        error_context.recovery_strategy = RecoveryStrategy.MANUAL
        return False

    async def _handle_validation_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle validation errors."""
        # Validation errors usually indicate data issues
        if checkpoint:
            return await self.rollback_to_checkpoint(checkpoint.checkpoint_id)
        return False

    async def _handle_concurrency_error(
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
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
        self, error_context: StateErrorContext, checkpoint: RecoveryCheckpoint | None = None
    ) -> bool:
        """Handle unknown errors."""
        # For unknown errors, try a limited number of retries
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

    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """Clean up old recovery checkpoints."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        old_checkpoints = [
            checkpoint_id
            for checkpoint_id, checkpoint in self._recovery_checkpoints.items()
            if checkpoint.timestamp < cutoff_time
        ]

        for checkpoint_id in old_checkpoints:
            del self._recovery_checkpoints[checkpoint_id]

        return len(old_checkpoints)


# Decorator for automatic error recovery
def with_error_recovery(
    operation_name: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    create_checkpoint: bool = True,
):
    """
    Decorator to add automatic error recovery to functions.

    Args:
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retries
        retry_delay: Initial retry delay
        create_checkpoint: Whether to create recovery checkpoint
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get recovery system from first argument (usually self)
            recovery_system = getattr(args[0], "_error_recovery", None) if args else None
            if not recovery_system:
                # No recovery system available, execute normally
                return await func(*args, **kwargs)

            checkpoint_id = None
            retry_count = 0

            while retry_count <= max_retries:
                try:
                    # Create checkpoint before first attempt
                    if create_checkpoint and retry_count == 0:
                        checkpoint_id = await recovery_system.create_recovery_checkpoint(
                            operation=operation_name,
                            state_type=kwargs.get("state_type", ""),
                            state_id=kwargs.get("state_id", ""),
                            current_state=kwargs.get("state_data", {}),
                        )

                    # Execute function
                    return await func(*args, **kwargs)

                except Exception as e:
                    # Handle error through recovery system
                    error_context = await recovery_system.handle_error(
                        exception=e,
                        operation=operation_name,
                        state_type=kwargs.get("state_type", ""),
                        state_id=kwargs.get("state_id", ""),
                        checkpoint_id=checkpoint_id,
                        retry_count=retry_count,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                    )

                    # Check if we should retry
                    if error_context.resolved and retry_count < max_retries:
                        retry_count += 1
                        continue
                    else:
                        # No more retries or unrecoverable error
                        raise e

            # Should never reach here
            raise StateError("Maximum retries exceeded")

        return wrapper

    return decorator
