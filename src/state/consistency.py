"""
State-Core Module Consistency Layer

This module provides utilities to ensure consistent data flow patterns
between state and core modules, addressing identified inconsistencies.
"""

import asyncio
from typing import Any, Callable, Protocol, TypeVar
from datetime import datetime, timezone

from src.core.base.events import BaseEventEmitter, EventPriority
from src.core.exceptions import StateConsistencyError, ValidationError
from src.core.validator_registry import ValidatorRegistry
from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DataTransformationProtocol(Protocol):
    """Protocol for consistent data transformation across modules."""

    def transform(self, data: Any) -> Any:
        """Transform data to consistent format."""
        ...

    def validate(self, data: Any) -> bool:
        """Validate data consistency."""
        ...


class ConsistentEventPattern:
    """
    Provides consistent event-driven patterns for state and core modules.

    Ensures all modules use the same pub/sub patterns instead of mixed
    queue/event approaches.
    """

    def __init__(self, name: str):
        self.name = name
        self.event_emitter = BaseEventEmitter(name=f"{name}_EventPattern")
        self._subscribers: dict[str, list[Callable]] = {}

    async def emit_consistent(
        self, event_type: str, data: Any, priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """
        Emit event using consistent pattern.

        Args:
            event_type: Type of event (e.g., 'state.changed', 'data.validated')
            data: Event data
            priority: Event priority
        """
        await self.event_emitter.emit_async(event_type, data, priority=priority, source=self.name)

    def subscribe_consistent(self, event_pattern: str, callback: Callable[[Any], None]) -> None:
        """
        Subscribe using consistent pattern.

        Args:
            event_pattern: Event pattern to match
            callback: Callback function
        """
        self.event_emitter.on_pattern(event_pattern, callback)


class ConsistentValidationPattern:
    """
    Provides consistent validation patterns across state and core modules.

    Ensures all modules use the same validation interfaces and error types.
    """

    def __init__(self):
        self.validator_registry = ValidatorRegistry()

    async def validate_consistent(
        self, data_type: str, data: Any, strict: bool = False
    ) -> dict[str, Any]:
        """
        Validate data using consistent patterns.

        Args:
            data_type: Type of data being validated
            data: Data to validate
            strict: Whether to use strict validation

        Returns:
            Validation result with consistent structure
        """
        try:
            # Use centralized validation
            is_valid = self.validator_registry.validate(data_type, data)

            return {
                "is_valid": is_valid,
                "data_type": data_type,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "errors": [],
                "warnings": [],
            }

        except ValidationError as e:
            return {
                "is_valid": False,
                "data_type": data_type,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "errors": [str(e)],
                "warnings": [],
            }


class ConsistentProcessingPattern:
    """
    Provides consistent processing patterns (batch vs stream) alignment.

    Ensures modules use consistent async/await patterns and processing approaches.
    """

    def __init__(self, name: str):
        self.name = name
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._running = False

    async def start_consistent_processing(self) -> None:
        """Start consistent async processing."""
        if not self._running:
            self._running = True
            self._processing_task = asyncio.create_task(self._process_loop())

    async def stop_consistent_processing(self) -> None:
        """Stop consistent async processing."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

    async def process_item_consistent(self, item: Any, processor: Callable[[Any], Any]) -> Any:
        """
        Process item using consistent async patterns.

        Args:
            item: Item to process
            processor: Processing function

        Returns:
            Processed result
        """
        try:
            if asyncio.iscoroutinefunction(processor):
                return await processor(item)
            else:
                return processor(item)
        except Exception as e:
            raise StateConsistencyError(f"Processing failed for {self.name}: {e}") from e

    async def _process_batch(self, batch: list[Any]) -> None:
        """Process a batch of items (override in subclasses)."""
        # Default implementation - process items sequentially with timeout
        for item in batch:
            try:
                await asyncio.wait_for(
                    self._process_single_item(item),
                    timeout=5.0  # Timeout per item
                )
            except asyncio.TimeoutError:
                logger.warning(f"Item processing timeout in {self.name}")
            except Exception as e:
                logger.error(f"Item processing error in {self.name}: {e}")
                
    async def _process_single_item(self, item: Any) -> None:
        """Process a single item (override in subclasses)."""
        # Default no-op implementation
        pass

    async def _process_loop(self) -> None:
        """Consistent processing loop."""
        while self._running:
            try:
                # Process items from queue with timeout
                try:
                    item = await asyncio.wait_for(self._batch_queue.get(), timeout=1.0)
                    self._batch_queue.task_done()
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                # Log error but continue processing
                logger.error(f"Processing error in {self.name}: {e}")
                await asyncio.sleep(0.1)


class ConsistentErrorPattern:
    """
    Provides consistent error propagation patterns across modules.

    Ensures all modules use the same exception hierarchy and error handling.
    """

    @staticmethod
    def raise_consistent_error(
        error_type: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        """
        Raise error using consistent exception hierarchy.

        Args:
            error_type: Type of error ('validation', 'state', 'sync')
            message: Error message
            context: Additional error context
        """
        if error_type == "validation":
            raise ValidationError(message, context=context or {})
        elif error_type == "state":
            raise StateConsistencyError(message, context=context or {})
        elif error_type == "sync":
            # Use StateConsistencyError for sync issues to maintain consistency
            raise StateConsistencyError(f"Synchronization error: {message}", context=context or {})
        elif error_type == "service":
            from src.core.exceptions import ServiceError

            raise ServiceError(message, context=context or {})
        else:
            from src.core.exceptions import TradingBotError

            raise TradingBotError(message, context=context or {})


# Global consistency instances for shared use
_event_pattern = ConsistentEventPattern("GlobalState")
_validation_pattern = ConsistentValidationPattern()
_processing_pattern = ConsistentProcessingPattern("GlobalState")
_error_pattern = ConsistentErrorPattern()


# Convenience functions for consistent patterns
async def emit_state_event(event_type: str, data: Any) -> None:
    """Emit state event using consistent pattern."""
    await _event_pattern.emit_consistent(f"state.{event_type}", data)


async def validate_state_data(data_type: str, data: Any) -> dict[str, Any]:
    """Validate state data using consistent pattern."""
    return await _validation_pattern.validate_consistent(data_type, data)


async def process_state_change(change: Any, processor: Callable) -> Any:
    """Process state change using consistent pattern."""
    return await _processing_pattern.process_item_consistent(change, processor)


def raise_state_error(message: str, context: dict[str, Any] | None = None) -> None:
    """Raise state error using consistent pattern."""
    _error_pattern.raise_consistent_error("state", message, context)
