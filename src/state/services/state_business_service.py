"""
State Business Service - Core business logic for state management.

This service contains all business rules and logic for state operations,
decoupled from infrastructure and presentation concerns.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from src.core.base.service import BaseService
from src.core.exceptions import BusinessRuleValidationError, StateError, ValidationError

if TYPE_CHECKING:
    from ..state_service import StateChange, StateMetadata, StateType


class StateBusinessServiceProtocol(Protocol):
    """Protocol defining the state business service interface."""

    async def validate_state_change(
        self,
        state_type: "StateType",
        state_id: str,
        current_state: dict[str, Any] | None,
        new_state: dict[str, Any],
        priority: str,
    ) -> dict[str, Any]: ...

    async def process_state_update(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
        reason: str,
    ) -> "StateChange": ...

    async def calculate_state_metadata(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
    ) -> "StateMetadata": ...

    async def validate_business_rules(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
        operation: str,
    ) -> list[str]: ...


class StateBusinessService(BaseService):
    """
    State business service implementing core state management business logic.

    This service handles all business rules, validations, and state processing
    logic independent of infrastructure concerns.
    """

    def __init__(self):
        """Initialize the state business service."""
        super().__init__(name="StateBusinessService")

        # Business rule configurations
        self.max_state_versions = 10
        self.state_retention_days = 90
        self.critical_state_types = {"bot_state", "risk_state", "position_state"}

        self.logger.info("StateBusinessService initialized")

    async def validate_state_change(
        self,
        state_type: "StateType",
        state_id: str,
        current_state: dict[str, Any] | None,
        new_state: dict[str, Any],
        priority: str,
    ) -> dict[str, Any]:
        """
        Validate a state change against business rules.

        Args:
            state_type: Type of state being changed
            state_id: State identifier
            current_state: Current state data
            new_state: Proposed new state
            priority: Change priority

        Returns:
            Validation result with any issues found

        Raises:
            BusinessRuleViolationError: If business rules are violated
        """
        try:
            issues = []
            warnings = []

            # Check state transition validity
            if current_state:
                transition_issues = await self._validate_state_transition(
                    state_type, current_state, new_state
                )
                issues.extend(transition_issues)

            # Validate business rules for the state type
            business_rule_issues = await self.validate_business_rules(
                state_type, new_state, "update"
            )
            issues.extend(business_rule_issues)

            # Check critical state constraints
            if state_type.value in self.critical_state_types:
                critical_issues = await self._validate_critical_state_constraints(
                    state_type, new_state
                )
                issues.extend(critical_issues)

            # Priority-specific validations
            if priority == "critical":
                priority_issues = await self._validate_critical_priority_constraints(
                    state_type, new_state
                )
                issues.extend(priority_issues)

            # Return validation result
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"State change validation failed: {e}")
            raise BusinessRuleValidationError(f"Validation failed: {e}") from e

    async def process_state_update(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
        reason: str,
    ) -> "StateChange":
        """
        Process a state update and create a state change record.

        Args:
            state_type: Type of state
            state_id: State identifier
            state_data: New state data
            source_component: Component making the change
            reason: Reason for the change

        Returns:
            StateChange record

        Raises:
            StateError: If processing fails
        """
        try:
            from ..state_service import StateChange, StateOperation, StatePriority

            # Determine operation type based on context
            operation = StateOperation.UPDATE  # Default to update

            # Map priority string to enum
            priority_mapping = {
                "critical": StatePriority.CRITICAL,
                "high": StatePriority.HIGH,
                "medium": StatePriority.MEDIUM,
                "low": StatePriority.LOW,
            }

            # Determine priority based on state type and context
            if state_type.value in self.critical_state_types:
                priority = StatePriority.HIGH
            else:
                priority = StatePriority.MEDIUM

            # Create state change record
            change_id = str(uuid4())
            timestamp = datetime.now(timezone.utc)

            state_change = StateChange(
                change_id=change_id,
                state_id=state_id,
                state_type=state_type,
                operation=operation,
                priority=priority,
                new_value=state_data.copy(),
                timestamp=timestamp,
                source_component=source_component,
                reason=reason,
            )

            # Apply business logic transformations
            state_change = await self._apply_business_transformations(state_change)

            self.logger.debug(
                f"Processed state update for {state_type.value}:{state_id}",
                extra={
                    "change_id": change_id,
                    "source": source_component,
                    "priority": priority.value,
                },
            )

            return state_change

        except Exception as e:
            self.logger.error(f"State update processing failed: {e}")
            raise StateError(f"Failed to process state update: {e}") from e

    async def calculate_state_metadata(
        self,
        state_type: "StateType",
        state_id: str,
        state_data: dict[str, Any],
        source_component: str,
    ) -> "StateMetadata":
        """
        Calculate metadata for state data.

        Args:
            state_type: Type of state
            state_id: State identifier
            state_data: State data
            source_component: Component creating the state

        Returns:
            StateMetadata record
        """
        try:
            import hashlib
            import json

            from ..state_service import StateMetadata

            # Calculate checksum
            data_str = json.dumps(state_data, sort_keys=True, default=str)
            checksum = hashlib.sha256(data_str.encode()).hexdigest()

            # Calculate size
            size_bytes = len(data_str.encode())

            # Create metadata
            metadata = StateMetadata(
                state_id=state_id,
                state_type=state_type,
                version=1,  # Will be updated by persistence layer
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                checksum=checksum,
                size_bytes=size_bytes,
                source_component=source_component,
                tags=self._generate_state_tags(state_type, state_data),
            )

            return metadata

        except Exception as e:
            self.logger.error(f"Metadata calculation failed: {e}")
            raise StateError(f"Failed to calculate state metadata: {e}") from e

    async def validate_business_rules(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
        operation: str,
    ) -> list[str]:
        """
        Validate business rules for state data.

        Args:
            state_type: Type of state
            state_data: State data to validate
            operation: Operation being performed

        Returns:
            List of validation issues
        """
        try:
            issues = []

            # Apply type-specific business rules
            if state_type.value == "bot_state":
                bot_issues = await self._validate_bot_state_rules(state_data, operation)
                issues.extend(bot_issues)

            elif state_type.value == "position_state":
                position_issues = await self._validate_position_state_rules(state_data, operation)
                issues.extend(position_issues)

            elif state_type.value == "risk_state":
                risk_issues = await self._validate_risk_state_rules(state_data, operation)
                issues.extend(risk_issues)

            elif state_type.value == "order_state":
                order_issues = await self._validate_order_state_rules(state_data, operation)
                issues.extend(order_issues)

            # Apply general business rules
            general_issues = await self._validate_general_business_rules(state_data, operation)
            issues.extend(general_issues)

            return issues

        except Exception as e:
            self.logger.error(f"Business rule validation failed: {e}")
            return [f"Business rule validation error: {e}"]

    # Private helper methods

    async def _validate_state_transition(
        self,
        state_type: "StateType",
        current_state: dict[str, Any],
        new_state: dict[str, Any],
    ) -> list[str]:
        """Validate state transition logic."""
        issues = []

        try:
            # Extract status/state fields for transition validation
            current_status = self._extract_status_field(current_state, state_type)
            new_status = self._extract_status_field(new_state, state_type)

            if current_status and new_status and current_status != new_status:
                # Validate transition based on state type
                if not await self._is_valid_transition(state_type, current_status, new_status):
                    issues.append(f"Invalid state transition from {current_status} to {new_status}")

            return issues

        except Exception as e:
            return [f"State transition validation error: {e}"]

    async def _validate_critical_state_constraints(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
    ) -> list[str]:
        """Validate constraints for critical state types."""
        issues = []

        try:
            if state_type.value == "bot_state":
                # Critical bot state constraints
                if "bot_id" not in state_data:
                    issues.append("bot_id is required for bot state")

                if "status" not in state_data:
                    issues.append("status is required for bot state")

                # Validate bot is not in invalid state
                status = state_data.get("status")
                if status == "error" and not state_data.get("error_message"):
                    issues.append("error_message required when status is error")

            elif state_type.value == "risk_state":
                # Critical risk state constraints
                exposure = state_data.get("exposure", 0)
                max_exposure = state_data.get("max_exposure", float("inf"))

                if exposure > max_exposure:
                    issues.append(f"Exposure {exposure} exceeds maximum {max_exposure}")

            return issues

        except Exception as e:
            return [f"Critical state constraint validation error: {e}"]

    async def _validate_critical_priority_constraints(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
    ) -> list[str]:
        """Validate constraints for critical priority operations."""
        issues = []

        try:
            # Critical priority operations must have audit trail
            if not state_data.get("audit_trail"):
                issues.append("Audit trail required for critical priority operations")

            # Critical operations must have approval for certain state types
            if state_type.value in ["risk_state", "bot_state"]:
                if not state_data.get("approved_by"):
                    issues.append("Approval required for critical risk/bot state changes")

            return issues

        except Exception as e:
            return [f"Critical priority constraint validation error: {e}"]

    async def _apply_business_transformations(self, state_change: "StateChange") -> "StateChange":
        """Apply business logic transformations to state change."""
        try:
            # Add audit information
            if state_change.new_value:
                state_change.new_value["_audit"] = {
                    "change_id": state_change.change_id,
                    "timestamp": state_change.timestamp.isoformat(),
                    "source": state_change.source_component,
                    "reason": state_change.reason,
                }

            # Apply state-type specific transformations
            if state_change.state_type.value == "bot_state":
                await self._transform_bot_state(state_change)
            elif state_change.state_type.value == "position_state":
                await self._transform_position_state(state_change)

            return state_change

        except Exception as e:
            self.logger.warning(f"Business transformation failed: {e}")
            return state_change

    def _generate_state_tags(
        self, state_type: "StateType", state_data: dict[str, Any]
    ) -> dict[str, str]:
        """Generate metadata tags for state."""
        tags = {
            "type": state_type.value,
            "version": "1.0",
        }

        # Add type-specific tags
        if state_type.value == "bot_state":
            tags["bot_id"] = state_data.get("bot_id", "unknown")
            tags["strategy"] = state_data.get("strategy_name", "unknown")
        elif state_type.value == "position_state":
            tags["symbol"] = state_data.get("symbol", "unknown")
            tags["side"] = state_data.get("side", "unknown")

        return tags

    def _extract_status_field(
        self, state_data: dict[str, Any], state_type: "StateType"
    ) -> str | None:
        """Extract status field from state data based on type."""
        status_fields = {
            "bot_state": "status",
            "order_state": "status",
            "position_state": "state",
            "strategy_state": "status",
        }

        field_name = status_fields.get(state_type.value)
        if field_name and field_name in state_data:
            value = state_data[field_name]
            # Handle enum values
            if hasattr(value, "value"):
                return value.value
            return str(value)

        return None

    async def _is_valid_transition(
        self, state_type: "StateType", current_status: str, new_status: str
    ) -> bool:
        """Check if state transition is valid."""
        # Define valid transitions for each state type
        valid_transitions = {
            "bot_state": {
                "initializing": {"running", "error", "stopped"},
                "running": {"paused", "stopping", "error"},
                "paused": {"running", "stopping", "error"},
                "stopping": {"stopped", "error"},
                "stopped": {"initializing"},
                "error": {"initializing", "stopped"},
            },
            "order_state": {
                "pending": {"open", "cancelled", "rejected"},
                "open": {"partially_filled", "filled", "cancelled"},
                "partially_filled": {"filled", "cancelled"},
                "filled": set(),
                "cancelled": set(),
                "rejected": set(),
            },
        }

        transitions = valid_transitions.get(state_type.value, {})
        allowed_states = transitions.get(current_status, set())

        return new_status in allowed_states

    # Business rule validation methods

    async def _validate_bot_state_rules(
        self, state_data: dict[str, Any], operation: str
    ) -> list[str]:
        """Validate bot state specific business rules."""
        issues = []

        try:
            # Bot must have valid configuration
            if "config" in state_data:
                config = state_data["config"]
                if not isinstance(config, dict):
                    issues.append("Bot config must be a dictionary")
                elif not config.get("exchange"):
                    issues.append("Bot config must specify exchange")

            # Bot capital allocation rules
            if "capital_allocation" in state_data:
                allocation = state_data["capital_allocation"]
                if allocation <= 0:
                    issues.append("Capital allocation must be positive")
                elif allocation > 1000000:  # $1M limit
                    issues.append("Capital allocation exceeds maximum limit")

            return issues

        except Exception as e:
            return [f"Bot state rule validation error: {e}"]

    async def _validate_position_state_rules(
        self, state_data: dict[str, Any], operation: str
    ) -> list[str]:
        """Validate position state specific business rules."""
        issues = []

        try:
            # Position size limits
            quantity = state_data.get("quantity", 0)
            if quantity <= 0:
                issues.append("Position quantity must be positive")

            # Price validation
            entry_price = state_data.get("entry_price", 0)
            if entry_price <= 0:
                issues.append("Entry price must be positive")

            # Risk limits
            if "stop_loss" in state_data and "entry_price" in state_data:
                stop_loss = state_data["stop_loss"]
                entry_price = state_data["entry_price"]
                side = state_data.get("side", "buy")

                if side.lower() == "buy" and stop_loss >= entry_price:
                    issues.append("Stop loss must be below entry price for long positions")
                elif side.lower() == "sell" and stop_loss <= entry_price:
                    issues.append("Stop loss must be above entry price for short positions")

            return issues

        except Exception as e:
            return [f"Position state rule validation error: {e}"]

    async def _validate_risk_state_rules(
        self, state_data: dict[str, Any], operation: str
    ) -> list[str]:
        """Validate risk state specific business rules."""
        issues = []

        try:
            # VaR limits
            var = state_data.get("var", 0)
            max_var = state_data.get("max_var", 0.05)  # 5% default
            if var > max_var:
                issues.append(f"VaR {var} exceeds maximum {max_var}")

            # Exposure limits
            exposure = state_data.get("exposure", 0)
            max_exposure = state_data.get("max_exposure", float("inf"))
            if exposure > max_exposure:
                issues.append(f"Exposure {exposure} exceeds maximum {max_exposure}")

            # Concentration limits
            positions = state_data.get("positions", [])
            if positions and len(positions) > 0:
                # Check for over-concentration in single asset
                total_value = sum(p.get("value", 0) for p in positions)
                if total_value > 0:
                    for position in positions:
                        position_value = position.get("value", 0)
                        concentration = position_value / total_value
                        if concentration > 0.25:  # 25% max per position
                            symbol = position.get("symbol", "unknown")
                            issues.append(f"Over-concentration in {symbol}: {concentration:.1%}")

            return issues

        except Exception as e:
            return [f"Risk state rule validation error: {e}"]

    async def _validate_order_state_rules(
        self, state_data: dict[str, Any], operation: str
    ) -> list[str]:
        """Validate order state specific business rules."""
        issues = []

        try:
            # Order quantity validation
            quantity = state_data.get("quantity", 0)
            if quantity <= 0:
                issues.append("Order quantity must be positive")

            # Price validation for limit orders
            order_type = state_data.get("type", "market")
            if order_type.lower() in ["limit", "stop_limit"]:
                price = state_data.get("price")
                if not price or price <= 0:
                    issues.append(f"{order_type} orders must have a positive price")

            # Market order price validation (shouldn't have price)
            elif order_type.lower() == "market":
                if "price" in state_data and state_data["price"] is not None:
                    issues.append("Market orders should not specify a price")

            return issues

        except Exception as e:
            return [f"Order state rule validation error: {e}"]

    async def _validate_general_business_rules(
        self, state_data: dict[str, Any], operation: str
    ) -> list[str]:
        """Validate general business rules applicable to all states."""
        issues = []

        try:
            # Required audit fields for certain operations
            if operation in ["create", "update", "delete"]:
                if not state_data.get("timestamp"):
                    state_data["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Data consistency checks
            if "created_at" in state_data and "updated_at" in state_data:
                created_at = state_data["created_at"]
                updated_at = state_data["updated_at"]

                # Parse timestamps if they're strings
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

                if updated_at < created_at:
                    issues.append("updated_at cannot be before created_at")

            return issues

        except Exception as e:
            return [f"General business rule validation error: {e}"]

    async def _transform_bot_state(self, state_change: "StateChange") -> None:
        """Apply bot state specific transformations."""
        try:
            if state_change.new_value:
                # Ensure bot has unique instance ID
                if "instance_id" not in state_change.new_value:
                    state_change.new_value["instance_id"] = str(uuid4())

                # Set last_heartbeat for running bots
                if state_change.new_value.get("status") == "running":
                    state_change.new_value["last_heartbeat"] = datetime.now(
                        timezone.utc
                    ).isoformat()

        except Exception as e:
            self.logger.warning(f"Bot state transformation failed: {e}")

    async def _transform_position_state(self, state_change: "StateChange") -> None:
        """Apply position state specific transformations."""
        try:
            if state_change.new_value:
                # Calculate unrealized P&L if current price is available
                if (
                    "current_price" in state_change.new_value
                    and "entry_price" in state_change.new_value
                ):
                    current_price = float(state_change.new_value["current_price"])
                    entry_price = float(state_change.new_value["entry_price"])
                    quantity = float(state_change.new_value.get("quantity", 0))
                    side = state_change.new_value.get("side", "buy")

                    if side.lower() == "buy":
                        unrealized_pnl = (current_price - entry_price) * quantity
                    else:
                        unrealized_pnl = (entry_price - current_price) * quantity

                    state_change.new_value["unrealized_pnl"] = unrealized_pnl

        except Exception as e:
            self.logger.warning(f"Position state transformation failed: {e}")
