"""
State Controller - Coordinates state management operations.

This controller follows proper service layer patterns by:
- Delegating business logic to services
- Handling HTTP/API concerns only
- Coordinating between multiple services
- Managing transaction boundaries
- Providing error handling and logging
"""

import asyncio

# Import types from service layer, not from state_service directly
from typing import TYPE_CHECKING, Any

from src.core.base.service import BaseService
from src.core.config.main import Config
from src.core.exceptions import ServiceError, StateConsistencyError, ValidationError
from src.utils.messaging_patterns import ErrorPropagationMixin

from .services import (
    StateBusinessServiceProtocol,
    StatePersistenceServiceProtocol,
    StateSynchronizationServiceProtocol,
    StateValidationServiceProtocol,
)

if TYPE_CHECKING:
    from .state_service import StateChange
    from src.core.types import StatePriority, StateType


class StateController(BaseService, ErrorPropagationMixin):
    """
    State controller that coordinates state management operations.

    This controller follows proper service layer architecture by:
    - Only handling coordination and transaction boundaries
    - Delegating all business logic to appropriate services
    - Managing error handling and logging
    - Providing API interface without containing business logic
    """

    def __init__(
        self,
        config: Config,
        business_service: StateBusinessServiceProtocol | None = None,
        persistence_service: StatePersistenceServiceProtocol | None = None,
        validation_service: StateValidationServiceProtocol | None = None,
        synchronization_service: StateSynchronizationServiceProtocol | None = None,
    ):
        """
        Initialize state controller with service dependencies.

        Args:
            config: Application configuration
            business_service: Service for business logic operations
            persistence_service: Service for persistence operations
            validation_service: Service for validation operations
            synchronization_service: Service for synchronization operations
        """
        # Convert Config to dict for BaseService
        config_dict = self._extract_config_dict(config)
        super().__init__(name="StateController", config=config_dict)

        # Service layer dependencies - use dependency injection if not provided
        self._business_service = business_service or self._resolve_service("StateBusinessService")
        self._persistence_service = persistence_service or self._resolve_service("StatePersistenceService")
        self._validation_service = validation_service or self._resolve_service("StateValidationService")
        self._synchronization_service = synchronization_service or self._resolve_service("StateSynchronizationService")

        # Controller state (not business state)
        self._transaction_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        self.logger.info("StateController initialized")

    def _resolve_service(self, service_name: str, factory_func=None):
        """Resolve service from DI container or return None if not available."""
        try:
            from src.core.dependency_injection import get_container
            container = get_container()
            return container.get(service_name)
        except Exception:
            # Return None if service not available (services are optional)
            return None

    async def get_state(
        self, state_type: "StateType", state_id: str, include_metadata: bool = False
    ) -> dict[str, Any] | None:
        """
        Get state by coordinating with appropriate services.

        Args:
            state_type: Type of state to retrieve
            state_id: Unique state identifier
            include_metadata: Whether to include state metadata

        Returns:
            State data or None if not found

        Raises:
            ServiceError: If service operations fail
        """
        try:
            # Check if persistence service is available
            if not self._persistence_service:
                self.logger.warning("Persistence service not available")
                return None

            # Delegate to persistence service
            state_data = await self._persistence_service.load_state(state_type, state_id)

            if not state_data:
                return None

            if include_metadata and self._business_service:
                # Get metadata through business service
                metadata = await self._business_service.calculate_state_metadata(
                    state_type, state_id, state_data, "StateController"
                )
                return {"data": state_data, "metadata": metadata}

            return state_data

        except Exception as e:
            self.logger.error(f"Failed to get state {state_type.value}:{state_id}: {e}")
            raise ServiceError(f"State retrieval failed: {e}") from e

    async def set_state(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        source_component: str = "",
        validate: bool = True,
        priority: "StatePriority" = None,
        reason: str = "",
    ) -> bool:
        """
        Set state by coordinating services with transaction-like guarantees.

        Args:
            state_type: Type of state
            state_id: Unique state identifier
            state_data: State data to store
            source_component: Component making the change
            validate: Whether to validate the state
            priority: Operation priority
            reason: Reason for the change

        Returns:
            True if successful

        Raises:
            ValidationError: If validation fails
            StateConsistencyError: If state update fails
        """
        # Handle default priority
        if priority is None:
            from .state_service import StatePriority as _StatePriority

            priority = _StatePriority.MEDIUM

        transaction_key = f"{state_type.value}:{state_id}"

        async with self._get_transaction_lock(transaction_key):
            try:
                # Get current state for validation and rollback
                current_state = await self.get_state(state_type, state_id)

                # Coordinate validation through service layer
                if validate and self._validation_service:
                    await self._coordinate_validation(
                        state_type, state_id, current_state, state_data, priority.value
                    )

                # Process state update through business service
                state_change = None
                metadata = None
                if self._business_service:
                    state_change = await self._business_service.process_state_update(
                        state_type, state_id, state_data, source_component, reason
                    )
                    metadata = await self._business_service.calculate_state_metadata(
                        state_type, state_id, state_data, source_component
                    )

                # Coordinate persistence through service layer
                if self._persistence_service and metadata:
                    await self._coordinate_persistence(state_type, state_id, state_data, metadata)

                # Coordinate synchronization through service layer
                if self._synchronization_service and state_change:
                    await self._coordinate_synchronization(state_change)

                self.logger.info(
                    f"State updated successfully: {transaction_key}",
                    extra={
                        "state_type": state_type.value,
                        "state_id": state_id,
                        "source": source_component,
                        "priority": priority.value,
                    },
                )

                return True

            except ValidationError:
                # Re-raise validation errors without wrapping
                raise
            except Exception as e:
                self.logger.error(f"State update failed for {transaction_key}: {e}")
                raise StateConsistencyError(f"State update failed: {e}") from e

    async def delete_state(
        self, state_type: "StateType", state_id: str, source_component: str = "", reason: str = ""
    ) -> bool:
        """
        Delete state by coordinating with appropriate services.

        Args:
            state_type: Type of state
            state_id: Unique state identifier
            source_component: Component making the change
            reason: Reason for deletion

        Returns:
            True if successful

        Raises:
            ServiceError: If deletion fails
        """
        transaction_key = f"{state_type.value}:{state_id}"

        async with self._get_transaction_lock(transaction_key):
            try:
                # Get current state for change record
                current_state = await self.get_state(state_type, state_id)
                if not current_state:
                    return True  # Already deleted

                # Create state change through business service
                from .state_service import (
                    StateChange as _StateChange,
                    StateOperation as _StateOperation,
                    StatePriority as _StatePriority,
                )

                state_change = _StateChange(
                    state_id=state_id,
                    state_type=state_type,
                    operation=_StateOperation.DELETE,
                    priority=_StatePriority.HIGH,
                    old_value=current_state,
                    new_value=None,
                    source_component=source_component,
                    reason=reason,
                )

                # Coordinate deletion through persistence service
                success = await self._persistence_service.delete_state(state_type, state_id)
                if not success:
                    raise ServiceError("Persistence layer deletion failed")

                # Coordinate synchronization of deletion
                await self._coordinate_synchronization(state_change)

                self.logger.info(
                    f"State deleted: {transaction_key}",
                    extra={"source": source_component, "reason": reason},
                )

                return True

            except Exception as e:
                self.logger.error(f"State deletion failed for {transaction_key}: {e}")
                raise ServiceError(f"State deletion failed: {e}") from e

    # Private coordination methods

    async def _coordinate_validation(
        self,
        state_type: "StateType",
        state_id: str,
        current_state: dict[str, Any] | None,
        new_state: dict[str, Any],
        priority: str,
    ) -> None:
        """Coordinate validation through validation service."""
        if not self._validation_service:
            self.logger.warning("Validation service not available, skipping validation")
            return

        # Validate state data
        validation_result = await self._validation_service.validate_state_data(
            state_type, new_state
        )
        if not validation_result["is_valid"]:
            raise ValidationError(f"State validation failed: {validation_result['errors']}")

        # Validate state transition if current state exists
        if current_state:
            transition_valid = await self._validation_service.validate_state_transition(
                state_type, current_state, new_state
            )
            if not transition_valid:
                raise ValidationError("Invalid state transition")

        # Delegate business rule validation to validation service
        if self._validation_service and hasattr(self._validation_service, "validate_business_rules"):
            business_violations = await self._validation_service.validate_business_rules(
                state_type, new_state, "update"
            )
            if business_violations:
                raise ValidationError(f"Business rule violations: {business_violations}")

    async def _coordinate_persistence(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        metadata: Any,
    ) -> None:
        """Coordinate persistence through persistence service."""
        if not self._persistence_service:
            raise StateConsistencyError("Persistence service not available")

        success = await self._persistence_service.save_state(
            state_type, state_id, state_data, metadata
        )
        if not success:
            raise StateConsistencyError("Failed to persist state")

    async def _coordinate_synchronization(self, state_change: "StateChange") -> None:
        """Coordinate synchronization through synchronization service."""
        if not self._synchronization_service:
            self.logger.debug("Synchronization service not available, skipping synchronization")
            return

        try:
            await self._synchronization_service.synchronize_state_change(state_change)
        except Exception as e:
            # Log synchronization failures but don't fail the main operation
            self.logger.warning(f"State synchronization failed: {e}")

    def _get_transaction_lock(self, transaction_key: str) -> asyncio.Lock:
        """Get or create a lock for transaction coordination."""
        if transaction_key not in self._transaction_locks:
            self._transaction_locks[transaction_key] = asyncio.Lock()
        return self._transaction_locks[transaction_key]

    async def cleanup(self) -> None:
        """Cleanup controller resources."""
        try:
            # Clear transaction locks
            self._transaction_locks.clear()
            self.logger.info("StateController cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during StateController cleanup: {e}")

    def _extract_config_dict(self, config: Config) -> dict[str, Any]:
        """Extract config as dictionary for BaseService."""
        if not config:
            return {}

        # Try to get config as dict
        if hasattr(config, "dict") and callable(config.dict):
            # Pydantic model
            return config.dict()
        elif hasattr(config, "__dict__"):
            return getattr(config, "__dict__", {})
        else:
            # Fallback
            return {}
