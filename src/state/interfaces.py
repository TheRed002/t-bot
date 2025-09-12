"""
State Management Interface Definitions.

This module defines the interfaces and protocols for state management services,
providing clear contracts and avoiding tight coupling between components.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, Union

if TYPE_CHECKING:
    from src.core.config.main import Config
    from src.database.models.state import StateMetadata

    from .state_service import RuntimeStateSnapshot, StateChange, StateService, StateType


class StateControllerProtocol(Protocol):
    """Protocol for state controllers that coordinate service operations."""

    async def get_state(
        self, state_type: "StateType", state_id: str, include_metadata: bool = False
    ) -> dict[str, Any] | None: ...

    async def set_state(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        source_component: str = "",
        validate: bool = True,
        reason: str = "",
    ) -> bool: ...

    async def delete_state(
        self, state_type: "StateType", state_id: str, source_component: str = "", reason: str = ""
    ) -> bool: ...


class StateBusinessServiceInterface(ABC):
    """Abstract interface for state business services."""

    @abstractmethod
    async def validate_state_change(
        self,
        state_type: "StateType",
        state_id: str,
        current_state: dict[str, Any] | None,
        new_state: dict[str, Any],
        priority: str,
    ) -> dict[str, Any]:
        """Validate a state change against business rules."""
        pass

    @abstractmethod
    async def process_state_update(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
        reason: str,
    ) -> "StateChange":
        """Process a state update and create change record."""
        pass

    @abstractmethod
    async def calculate_state_metadata(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
    ) -> "StateMetadata":
        """Calculate metadata for state data."""
        pass


class StatePersistenceServiceInterface(ABC):
    """Abstract interface for state persistence services."""

    @abstractmethod
    async def save_state(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        metadata: "StateMetadata",
    ) -> bool:
        """Save state data to persistent storage."""
        pass

    @abstractmethod
    async def load_state(self, state_type: "StateType", state_id: str) -> dict[str, Any] | None:
        """Load state data from persistent storage."""
        pass

    @abstractmethod
    async def delete_state(self, state_type: "StateType", state_id: str) -> bool:
        """Delete state data from persistent storage."""
        pass

    @abstractmethod
    async def save_snapshot(self, snapshot: "RuntimeStateSnapshot") -> bool:
        """Save a state snapshot."""
        pass

    @abstractmethod
    async def load_snapshot(self, snapshot_id: str) -> "RuntimeStateSnapshot | None":
        """Load a state snapshot."""
        pass


class StateValidationServiceInterface(ABC):
    """Abstract interface for state validation services."""

    @abstractmethod
    async def validate_state_data(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
        validation_level: str = "normal",
    ) -> dict[str, Any]:
        """Validate state data against all applicable rules."""
        pass

    @abstractmethod
    async def validate_state_transition(
        self,
        state_type: "StateType",
        current_state: dict[str, Any],
        new_state: dict[str, Any],
    ) -> bool:
        """Validate a state transition."""
        pass

    @abstractmethod
    async def validate_business_rules(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
        operation: str = "update",
    ) -> list[str]:
        """Validate business rules for state data."""
        pass

    @abstractmethod
    def matches_criteria(
        self,
        state: dict[str, Any],
        criteria: dict[str, Any]
    ) -> bool:
        """Check if state matches search criteria."""
        pass


class StateSynchronizationServiceInterface(ABC):
    """Abstract interface for state synchronization services."""

    @abstractmethod
    async def synchronize_state_change(self, state_change: "StateChange") -> bool:
        """Synchronize a state change across the system."""
        pass

    @abstractmethod
    async def broadcast_state_change(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any] | None,
        change_info: dict[str, Any],
    ) -> None:
        """Broadcast state change to interested parties."""
        pass

    @abstractmethod
    async def resolve_conflicts(
        self,
        state_type: "StateType",
        state_id: str,
        conflicting_changes: list["StateChange"],
    ) -> "StateChange":
        """Resolve conflicts between multiple state changes."""
        pass


class CheckpointServiceInterface(ABC):
    """Abstract interface for checkpoint services."""

    @abstractmethod
    async def create_checkpoint(
        self,
        bot_id: str,
        state_data: dict[str, Any],
        checkpoint_type: str = "manual",
    ) -> str:
        """Create a checkpoint for state data."""
        pass

    @abstractmethod
    async def restore_checkpoint(self, checkpoint_id: str) -> tuple[str, dict[str, Any]] | None:
        """Restore state from a checkpoint."""
        pass

    @abstractmethod
    async def list_checkpoints(
        self, bot_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List available checkpoints."""
        pass

    @abstractmethod
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        pass


class StateEventServiceInterface(ABC):
    """Abstract interface for state event services."""

    @abstractmethod
    async def emit_state_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        """Emit a state change event."""
        pass

    @abstractmethod
    def subscribe_to_events(self, event_type: str, callback: Any) -> None:
        """Subscribe to state change events."""
        pass

    @abstractmethod
    def unsubscribe_from_events(self, event_type: str, callback: Any) -> None:
        """Unsubscribe from state change events."""
        pass


class StateServiceFactoryInterface(ABC):
    """Abstract interface for state service factories."""

    @abstractmethod
    async def create_state_service(
        self,
        config: "Config",
        database_service: Any | None = None,
        auto_start: bool = True,
    ) -> "StateService":
        """Create a StateService instance with dependency injection."""
        pass

    @abstractmethod
    async def create_state_service_for_testing(
        self,
        config: Union["Config", None] = None,
        mock_database: bool = False,
    ) -> "StateService":
        """Create a StateService instance for testing."""
        pass


class MetricsStorageInterface(ABC):
    """Abstract interface for metrics storage operations."""

    @abstractmethod
    async def store_validation_metrics(self, validation_data: dict[str, Any]) -> bool:
        """Store validation metrics."""
        pass

    @abstractmethod
    async def store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool:
        """Store analysis metrics."""
        pass

    @abstractmethod
    async def get_historical_metrics(
        self, metric_type: str, start_time: Any, end_time: Any
    ) -> list[dict[str, Any]]:
        """Retrieve historical metrics."""
        pass
