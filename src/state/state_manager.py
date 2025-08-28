"""
StateManager wrapper for backward compatibility with existing tests.

This module provides a StateManager class that wraps the StateService
to maintain compatibility with existing tests while the codebase
transitions to the new StateService architecture.
"""

from typing import Any

from src.core.config.main import Config
from src.core.types import BotState

from .state_service import StatePriority, StateService, StateType


def get_cache_manager():
    """Get cache manager instance (stub for backward compatibility)."""
    from src.core.caching.cache_manager import CacheManager

    return CacheManager.get_instance()


class StateManager:
    """
    Backward compatibility wrapper for StateService.

    This class provides the StateManager interface expected by existing tests
    while delegating all operations to the new StateService implementation.
    """

    def __init__(self, config: Config):
        """Initialize StateManager with StateService delegation."""
        self.config = config
        # Initialize state service without database service (will use default)
        self.state_service = StateService(config, None)

    async def initialize(self) -> None:
        """Initialize the state manager."""
        # Initialize state service
        await self.state_service.initialize()

    async def shutdown(self) -> None:
        """Shutdown the state manager."""
        await self.state_service.cleanup()

    async def save_bot_state(
        self, bot_id: str, state: dict[str, Any], create_snapshot: bool = False
    ) -> str:
        """Save bot state and optionally create a snapshot."""
        # Save the state
        success = await self.state_service.set_state(
            StateType.BOT_STATE, bot_id, state, priority=StatePriority.HIGH
        )

        if not success:
            raise RuntimeError(f"Failed to save state for bot {bot_id}")

        # Create snapshot if requested
        if create_snapshot:
            snapshot_id = await self.state_service.create_snapshot(bot_id)
            return snapshot_id

        # Return a version ID (could be timestamp or state hash)
        import hashlib
        import json

        state_json = json.dumps(state, sort_keys=True, default=str)
        version_id = hashlib.sha256(state_json.encode()).hexdigest()[:12]
        return version_id

    async def load_bot_state(self, bot_id: str) -> BotState | None:
        """Load bot state."""
        result = await self.state_service.get_state(StateType.BOT_STATE, bot_id)

        if result is None:
            return None

        # If result is already a BotState object, return it
        if isinstance(result, BotState):
            return result

        # If result is a dict, try to reconstruct BotState
        if isinstance(result, dict):
            # Handle case where result might have 'data' key
            data = result.get("data", result)

            # Ensure bot_id is in the data
            if "bot_id" not in data and bot_id:
                data["bot_id"] = bot_id

            # Try to create BotState from dict
            try:
                return BotState(**data)
            except Exception:
                # If BotState construction fails, return the dict wrapped in a simple object
                class StateWrapper:
                    def __init__(self, data):
                        self.__dict__.update(data)

                return StateWrapper(data)

        # For any other type, try to extract data
        if hasattr(result, "data"):
            return self.load_bot_state(bot_id)  # Recursive call with extracted data

        return result

    async def create_checkpoint(
        self, bot_id: str, checkpoint_data: dict[str, Any] | None = None
    ) -> str:
        """Create a checkpoint."""
        checkpoint_id = await self.state_service.create_snapshot(bot_id)
        return checkpoint_id

    async def restore_from_checkpoint(self, bot_id: str, checkpoint_id: str) -> bool:
        """Restore from checkpoint."""
        # StateService.restore_snapshot only expects snapshot_id parameter
        return await self.state_service.restore_snapshot(checkpoint_id)

    async def get_state_metrics(self, bot_id: str | None = None, hours: int = 24) -> dict[str, Any]:
        """Get state metrics."""
        metrics = await self.state_service.get_metrics()

        # Base result with bot_id and period_hours
        result = {
            "bot_id": bot_id,
            "period_hours": hours,
        }

        # Convert metrics to expected format
        if isinstance(metrics, dict):
            result.update(
                {
                    "total_states": metrics.get("total_states", 0),
                    "cache_hit_rate": metrics.get("cache_hit_rate", 0.0),
                    "operations_per_second": metrics.get("operations_per_second", 0.0),
                    "error_rate": metrics.get("error_rate", 0.0),
                    "state_updates": metrics.get(
                        "total_operations", 0
                    ),  # Map total_operations to state_updates
                }
            )
        elif hasattr(metrics, "__dict__"):
            # Handle case where metrics is an object
            result.update(
                {
                    "total_operations": getattr(metrics, "total_operations", 0),
                    "state_updates": getattr(metrics, "total_operations", 0),  # Add state_updates
                    "last_successful_sync": getattr(metrics, "last_successful_sync", None),
                    "storage_usage_mb": getattr(metrics, "storage_usage_mb", 0.0),
                    "cache_hit_rate": getattr(metrics, "cache_hit_rate", 0.0),
                    "error_rate": getattr(metrics, "error_rate", 0.0),
                    "active_states_count": getattr(metrics, "active_states_count", 0),
                }
            )
        else:
            # Return empty metrics if unexpected type
            result.update(
                {
                    "total_operations": 0,
                    "state_updates": 0,
                    "last_successful_sync": None,
                    "storage_usage_mb": 0.0,
                    "cache_hit_rate": 0.0,
                    "error_rate": 0.0,
                    "active_states_count": 0,
                }
            )

        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to state_service."""
        return getattr(self.state_service, name)
