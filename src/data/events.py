"""
Data Events - Simple Event Patterns for Data Module

This module provides basic event-driven patterns for the data module.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Import centralized event constants to ensure consistency
from src.core.event_constants import DataEvents


class DataEventType(Enum):
    """Types of data events - aligned with core event constants."""
    DATA_STORED = DataEvents.STORED
    DATA_RETRIEVED = DataEvents.RETRIEVED
    DATA_VALIDATED = DataEvents.VALIDATED
    DATA_VALIDATION_FAILED = DataEvents.VALIDATION_FAILED
    CACHE_HIT = DataEvents.CACHE_HIT
    CACHE_MISS = DataEvents.CACHE_MISS
    PIPELINE_STARTED = DataEvents.PIPELINE_STARTED
    PIPELINE_COMPLETED = DataEvents.PIPELINE_COMPLETED
    PIPELINE_FAILED = DataEvents.PIPELINE_FAILED
    QUALITY_ALERT = DataEvents.QUALITY_ALERT
    PERFORMANCE_ALERT = DataEvents.PERFORMANCE_ALERT


@dataclass
class DataEvent:
    """Data event structure aligned with database service messaging patterns."""
    event_type: DataEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Add messaging pattern metadata for consistency with database service
    message_pattern: str = "pub_sub"  # Default to pub/sub for events
    processing_mode: str = "stream"  # Align with database service default
    boundary_crossed: bool = True  # Data events cross module boundaries


class DataEventPublisher:
    """
    Simple mixin class to add basic event publishing capabilities.
    """

    def __init__(self, *args, **kwargs):
        """Initialize event publisher."""
        super().__init__(*args, **kwargs)

    async def _publish_data_event(
        self,
        event_type: DataEventType,
        source: str = None,
        data: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """
        Publish a data event using consistent pub/sub pattern aligned with error_handling module.
        
        Args:
            event_type: Type of event
            source: Source component name
            data: Event data
            metadata: Additional metadata
        """
        try:
            # Use CoreDataTransformer for consistent data transformation patterns
            from src.core.data_transformer import CoreDataTransformer

            transformed_data = CoreDataTransformer.transform_event_to_standard_format(
                event_type=event_type.value,
                data=data or {},
                metadata=metadata,
                source=source or self.__class__.__name__
            )

            # Apply boundary validation for data to error_handling module communication
            from src.utils.messaging_patterns import BoundaryValidator
            boundary_data = {
                "component": source or self.__class__.__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_mode": "stream",  # Consistent with error_handling
                "data_format": "core_event_data_v1",  # Use consistent format
                "message_pattern": "pub_sub",  # Consistent messaging pattern
                "boundary_crossed": True,
                **transformed_data
            }
            BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)

            event = DataEvent(
                event_type=event_type,
                source=source or self.__class__.__name__,
                data=transformed_data.get("data", {}),
                metadata=transformed_data.get("metadata", {}),
                message_pattern="pub_sub",  # Consistent pub/sub pattern
                processing_mode="stream",  # Align with error_handling
                boundary_crossed=True,  # Data events cross boundaries
            )

            # Use consistent messaging pattern for event publishing
            if hasattr(self, "logger"):
                self.logger.debug(
                    f"Event published via {event.message_pattern}: {event_type.value} from {event.source}",
                    extra={
                        "message_pattern": event.message_pattern,
                        "processing_mode": event.processing_mode,
                        "boundary_crossed": event.boundary_crossed,
                        "data_format": boundary_data["data_format"]
                    }
                )

        except Exception as e:
            # Use consistent error propagation patterns
            from src.utils.messaging_patterns import ErrorPropagationMixin
            mixin = ErrorPropagationMixin()
            try:
                mixin.propagate_monitoring_error(e, f"data_event_publishing.{event_type.value}")
            except Exception:
                # Fallback to regular logging if propagation fails
                if hasattr(self, "logger"):
                    self.logger.warning(f"Failed to publish event {event_type.value}: {e}")



class DataEventSubscriber:
    """
    Base class for data event subscribers.
    """

    def __init__(self, *args, **kwargs):
        """Initialize event subscriber."""
        super().__init__(*args, **kwargs)

    async def handle_data_event(self, event: DataEvent) -> None:
        """
        Handle a data event. Override in subclasses.
        
        Args:
            event: Data event to handle
        """
        raise NotImplementedError("Subclasses must implement handle_data_event")
