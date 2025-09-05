"""
StateManager wrapper for backward compatibility with existing tests.

This module provides a StateManager class that wraps the StateService
to maintain compatibility with existing tests while the codebase
transitions to the new StateService architecture.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from src.core.base.component import BaseComponent
from src.core.config.main import Config
from src.core.types import BotState

if TYPE_CHECKING:
    from .state_service import StatePriority, StateService, StateType


def get_cache_manager():
    """Get cache manager instance (stub for backward compatibility)."""
    from src.core.caching.cache_manager import CacheManager

    return CacheManager.get_instance()


class StateManager(BaseComponent):
    """
    Backward compatibility wrapper for StateService.

    This class provides the StateManager interface expected by existing tests
    while delegating all operations to the new StateService implementation.
    """

    def __init__(self, config: Config):
        """Initialize StateManager with StateService delegation."""
        super().__init__()
        self.config = config
        # State service will be created using factory in initialize()
        self.state_service: Optional[StateService] = None

    async def initialize(self) -> None:
        """Initialize the state manager using factory pattern with dependency injection."""
        # Use dependency injection container to get factory
        from src.core.dependency_injection import get_container
        from src.core.exceptions import DependencyError, ServiceError
        from .factory import StateServiceFactory
        
        container = get_container()
        try:
            # Use dependency injection to get factory
            factory = container.get("StateServiceFactory")
        except (DependencyError, ServiceError):
            # Fallback to direct factory creation
            factory = StateServiceFactory()
        
        self.state_service = await factory.create_state_service(
            config=self.config, 
            auto_start=True
        )

    async def shutdown(self) -> None:
        """Shutdown the state manager."""
        if self.state_service:
            await self.state_service.cleanup()

    async def save_bot_state(
        self, bot_id: str, state: dict[str, Any], create_snapshot: bool = False
    ) -> str:
        """Save bot state and optionally create a snapshot."""
        if not self.state_service:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")
            
        # Save the state - late import to avoid circular dependency
        from .state_service import StatePriority, StateType
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
        if not self.state_service:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")
            
        # Late import to avoid circular dependency
        from .state_service import StateType
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
            except Exception as e:
                self.logger.warning(f"BotState construction failed: {e}")
                # If BotState construction fails, return None for consistency
                return None

        # For any other type, try to convert or return None
        if hasattr(result, "data"):
            return await self.load_bot_state(bot_id)  # Recursive call with extracted data

        # If we can't convert to BotState, return None
        if isinstance(result, BotState):
            return result
        
        return None

    async def create_checkpoint(
        self, bot_id: str, checkpoint_data: dict[str, Any] | None = None
    ) -> str:
        """Create a checkpoint."""
        if not self.state_service:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")
            
        checkpoint_id = await self.state_service.create_snapshot(bot_id)
        return checkpoint_id

    async def restore_from_checkpoint(self, bot_id: str, checkpoint_id: str) -> bool:
        """Restore from checkpoint."""
        if not self.state_service:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")
            
        # StateService.restore_snapshot only expects snapshot_id parameter
        return await self.state_service.restore_snapshot(checkpoint_id)

    async def get_state_metrics(self, bot_id: str | None = None, hours: int = 24) -> dict[str, Any]:
        """Get state metrics."""
        if not self.state_service:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")
            
        metrics = self.state_service.get_metrics()

        # Base result with bot_id and period_hours
        result: dict[str, Any] = {
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
        """
        Delegate unknown attributes to state_service.
        
        Args:
            name: Attribute name to retrieve
            
        Returns:
            The requested attribute from the state service
            
        Raises:
            AttributeError: If the attribute doesn't exist on the state service
        """
        if self.state_service is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' and no state service is available")
        return getattr(self.state_service, name)
