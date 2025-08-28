"""
State Validation Service - Provides state validation capabilities.

This service handles all validation logic for state data, providing
a clean interface for validation operations without exposing
implementation details.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from src.core.base.service import BaseService
from src.core.validator_registry import ValidatorRegistry, validate

if TYPE_CHECKING:
    from ..state_service import StateType


class StateValidationServiceProtocol(Protocol):
    """Protocol defining the state validation service interface."""

    async def validate_state_data(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
        validation_level: str = "normal",
    ) -> dict[str, Any]: ...

    async def validate_state_transition(
        self,
        state_type: "StateType",
        current_state: dict[str, Any],
        new_state: dict[str, Any],
    ) -> bool: ...

    async def validate_business_rules(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
    ) -> list[str]: ...


class StateValidationService(BaseService):
    """
    State validation service providing comprehensive validation capabilities.

    This service handles all validation logic for state data, including
    business rules, data integrity, and state transition validation.
    """

    def __init__(self):
        """Initialize the state validation service."""
        super().__init__(name="StateValidationService")

        # Use centralized validator registry for consistency
        self.validator_registry = ValidatorRegistry()
        
        # Validation configuration
        self.strict_validation = True
        self.enable_business_rules = True
        self.cache_validation_results = True

        # Validation cache
        self._validation_cache: dict[str, tuple[dict[str, Any], datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes

        # Validation metrics
        self._validation_count = 0
        self._validation_failures = 0
        self._cache_hits = 0

        self.logger.info("StateValidationService initialized")

    async def validate_state_data(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
        validation_level: str = "normal",
    ) -> dict[str, Any]:
        """
        Validate state data against all applicable rules.

        Args:
            state_type: Type of state to validate
            state_data: State data to validate
            validation_level: Validation strictness level

        Returns:
            Validation result dictionary
        """
        try:
            self._validation_count += 1
            start_time = datetime.now(timezone.utc)

            # Check cache for recent validation result
            if self.cache_validation_results:
                cached_result = await self._get_cached_validation(
                    state_type, state_data, validation_level
                )
                if cached_result:
                    self._cache_hits += 1
                    return cached_result

            # Initialize validation result
            result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "validation_level": validation_level,
                "state_type": state_type.value,
                "validated_at": start_time.isoformat(),
            }

            # Perform basic data validation
            basic_errors = await self._validate_basic_data_structure(state_type, state_data)
            result["errors"].extend(basic_errors)

            # Perform type-specific validation
            type_errors = await self._validate_state_type_specific(state_type, state_data)
            result["errors"].extend(type_errors)

            # Perform business rule validation if enabled
            if self.enable_business_rules:
                business_errors = await self.validate_business_rules(state_type, state_data)
                result["errors"].extend(business_errors)

            # Perform strict validation if required
            if validation_level == "strict" or self.strict_validation:
                strict_errors = await self._validate_strict_requirements(state_type, state_data)
                result["errors"].extend(strict_errors)

            # Update validation result
            result["is_valid"] = len(result["errors"]) == 0

            # Calculate validation time
            end_time = datetime.now(timezone.utc)
            result["validation_time_ms"] = (end_time - start_time).total_seconds() * 1000

            # Cache result if enabled
            if self.cache_validation_results:
                await self._cache_validation_result(
                    state_type, state_data, validation_level, result
                )

            # Update failure count if validation failed
            if not result["is_valid"]:
                self._validation_failures += 1

            return result

        except Exception as e:
            self.logger.error(f"State validation failed: {e}")
            self._validation_failures += 1
            return {
                "is_valid": False,
                "errors": [f"Validation system error: {e}"],
                "warnings": [],
                "validation_level": validation_level,
                "state_type": state_type.value,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "validation_time_ms": 0.0,
            }

    async def validate_state_transition(
        self,
        state_type: "StateType",
        current_state: dict[str, Any],
        new_state: dict[str, Any],
    ) -> bool:
        """
        Validate a state transition.

        Args:
            state_type: Type of state
            current_state: Current state data
            new_state: Proposed new state data

        Returns:
            True if transition is valid
        """
        try:
            # Extract status fields from states
            current_status = self._extract_status_field(current_state, state_type)
            new_status = self._extract_status_field(new_state, state_type)

            # If no status fields, allow transition
            if not current_status or not new_status:
                return True

            # Get valid transitions for this state type
            valid_transitions = self._get_valid_transitions(state_type)

            # Check if transition is allowed
            allowed_states = valid_transitions.get(current_status, set())
            is_valid = new_status in allowed_states or current_status == new_status

            if not is_valid:
                self.logger.warning(
                    f"Invalid state transition: {current_status} -> {new_status}",
                    extra={
                        "state_type": state_type.value,
                        "allowed_transitions": list(allowed_states),
                    },
                )

            return is_valid

        except Exception as e:
            self.logger.error(f"State transition validation failed: {e}")
            return False

    async def validate_business_rules(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
    ) -> list[str]:
        """
        Validate business rules for state data.

        Args:
            state_type: Type of state
            state_data: State data to validate

        Returns:
            List of business rule violations
        """
        try:
            violations = []

            # Apply state-type specific business rules
            if state_type.value == "bot_state":
                bot_violations = await self._validate_bot_business_rules(state_data)
                violations.extend(bot_violations)

            elif state_type.value == "position_state":
                position_violations = await self._validate_position_business_rules(state_data)
                violations.extend(position_violations)

            elif state_type.value == "order_state":
                order_violations = await self._validate_order_business_rules(state_data)
                violations.extend(order_violations)

            elif state_type.value == "risk_state":
                risk_violations = await self._validate_risk_business_rules(state_data)
                violations.extend(risk_violations)

            # Apply general business rules
            general_violations = await self._validate_general_business_rules(state_data)
            violations.extend(general_violations)

            return violations

        except Exception as e:
            self.logger.error(f"Business rule validation failed: {e}")
            return [f"Business rule validation error: {e}"]

    def get_validation_metrics(self) -> dict[str, Any]:
        """Get validation service metrics."""
        return {
            "total_validations": self._validation_count,
            "validation_failures": self._validation_failures,
            "success_rate": (self._validation_count - self._validation_failures)
            / max(self._validation_count, 1),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(self._validation_count, 1),
        }

    # Private helper methods

    async def _validate_basic_data_structure(
        self, state_type: "StateType", state_data: dict[str, Any]
    ) -> list[str]:
        """Validate basic data structure requirements."""
        errors = []

        try:
            # Check if data is a dictionary
            if not isinstance(state_data, dict):
                errors.append("State data must be a dictionary")
                return errors

            # Check for required base fields
            required_fields = self._get_required_fields(state_type)
            for field in required_fields:
                if field not in state_data:
                    errors.append(f"Required field '{field}' is missing")
                elif state_data[field] is None:
                    errors.append(f"Required field '{field}' cannot be null")

            # Validate field types
            field_type_errors = await self._validate_field_types(state_type, state_data)
            errors.extend(field_type_errors)

            return errors

        except Exception as e:
            return [f"Basic validation error: {e}"]

    async def _validate_state_type_specific(
        self, state_type: "StateType", state_data: dict[str, Any]
    ) -> list[str]:
        """Validate state-type specific requirements."""
        errors = []

        try:
            if state_type.value == "bot_state":
                errors.extend(await self._validate_bot_state_structure(state_data))
            elif state_type.value == "position_state":
                errors.extend(await self._validate_position_state_structure(state_data))
            elif state_type.value == "order_state":
                errors.extend(await self._validate_order_state_structure(state_data))
            elif state_type.value == "risk_state":
                errors.extend(await self._validate_risk_state_structure(state_data))

            return errors

        except Exception as e:
            return [f"Type-specific validation error: {e}"]

    async def _validate_strict_requirements(
        self, state_type: "StateType", state_data: dict[str, Any]
    ) -> list[str]:
        """Validate strict requirements for state data."""
        errors = []

        try:
            # Strict validation requires audit fields
            if "created_at" not in state_data:
                errors.append("created_at field required in strict mode")

            if "updated_at" not in state_data:
                errors.append("updated_at field required in strict mode")

            # Validate timestamp consistency
            if "created_at" in state_data and "updated_at" in state_data:
                created_at = state_data["created_at"]
                updated_at = state_data["updated_at"]

                # Convert strings to datetime if necessary
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

                if updated_at < created_at:
                    errors.append("updated_at cannot be before created_at")

            return errors

        except Exception as e:
            return [f"Strict validation error: {e}"]

    def _get_required_fields(self, state_type: "StateType") -> list[str]:
        """Get required fields for a state type."""
        field_map = {
            "bot_state": ["bot_id", "status"],
            "position_state": ["symbol", "quantity", "entry_price"],
            "order_state": ["order_id", "symbol", "side", "type"],
            "risk_state": ["exposure"],
            "portfolio_state": ["total_value"],
            "strategy_state": ["strategy_id"],
            "market_state": ["symbol", "price"],
            "trade_state": ["trade_id", "symbol"],
        }
        return field_map.get(state_type.value, [])

    async def _validate_field_types(
        self, state_type: "StateType", state_data: dict[str, Any]
    ) -> list[str]:
        """Validate field types for state data."""
        errors = []

        try:
            # Define expected field types for each state type
            type_specs = {
                "bot_state": {
                    "bot_id": str,
                    "status": str,
                    "capital_allocation": (int, float),
                },
                "position_state": {
                    "symbol": str,
                    "quantity": (int, float),
                    "entry_price": (int, float),
                    "side": str,
                },
                "order_state": {
                    "order_id": str,
                    "symbol": str,
                    "side": str,
                    "type": str,
                    "quantity": (int, float),
                    "price": (int, float, type(None)),
                },
            }

            expected_types = type_specs.get(state_type.value, {})

            for field, expected_type in expected_types.items():
                if field in state_data:
                    value = state_data[field]
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"Field '{field}' must be of type {expected_type.__name__ if hasattr(expected_type, '__name__') else expected_type}"
                        )

            return errors

        except Exception as e:
            return [f"Field type validation error: {e}"]

    # State-specific validation methods

    async def _validate_bot_state_structure(self, state_data: dict[str, Any]) -> list[str]:
        """Validate bot state structure."""
        errors = []

        if "bot_id" in state_data:
            bot_id = state_data["bot_id"]
            if not bot_id or not isinstance(bot_id, str):
                errors.append("bot_id must be a non-empty string")
            elif len(bot_id) > 100:
                errors.append("bot_id cannot exceed 100 characters")

        if "status" in state_data:
            status = state_data["status"]
            valid_statuses = {"initializing", "running", "paused", "stopped", "error"}
            if isinstance(status, str) and status not in valid_statuses:
                errors.append(f"Invalid bot status: {status}")

        return errors

    async def _validate_position_state_structure(self, state_data: dict[str, Any]) -> list[str]:
        """Validate position state structure."""
        errors = []

        if "quantity" in state_data:
            quantity = state_data["quantity"]
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                errors.append("quantity must be a positive number")

        if "entry_price" in state_data:
            entry_price = state_data["entry_price"]
            if not isinstance(entry_price, (int, float)) or entry_price <= 0:
                errors.append("entry_price must be a positive number")

        if "side" in state_data:
            side = state_data["side"]
            valid_sides = {"buy", "sell", "long", "short"}
            if isinstance(side, str) and side.lower() not in valid_sides:
                errors.append(f"Invalid position side: {side}")

        return errors

    async def _validate_order_state_structure(self, state_data: dict[str, Any]) -> list[str]:
        """Validate order state structure."""
        errors = []

        if "type" in state_data:
            order_type = state_data["type"]
            valid_types = {"market", "limit", "stop", "stop_limit"}
            if isinstance(order_type, str) and order_type.lower() not in valid_types:
                errors.append(f"Invalid order type: {order_type}")

        if "quantity" in state_data:
            quantity = state_data["quantity"]
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                errors.append("order quantity must be a positive number")

        # Validate price requirements based on order type
        order_type = state_data.get("type", "").lower()
        price = state_data.get("price")

        if order_type in ["limit", "stop", "stop_limit"]:
            if price is None:
                errors.append(f"{order_type} orders require a price")
            elif not isinstance(price, (int, float)) or price <= 0:
                errors.append("order price must be a positive number")
        elif order_type == "market" and price is not None:
            errors.append("market orders should not specify a price")

        return errors

    async def _validate_risk_state_structure(self, state_data: dict[str, Any]) -> list[str]:
        """Validate risk state structure."""
        errors = []

        if "exposure" in state_data:
            exposure = state_data["exposure"]
            if not isinstance(exposure, (int, float)) or exposure < 0:
                errors.append("exposure must be a non-negative number")

        if "var" in state_data:
            var = state_data["var"]
            if not isinstance(var, (int, float)):
                errors.append("VaR must be a number")
            elif var < 0:
                errors.append("VaR cannot be negative")

        return errors

    # Business rule validation methods

    async def _validate_bot_business_rules(self, state_data: dict[str, Any]) -> list[str]:
        """Validate bot-specific business rules."""
        violations = []

        # Capital allocation limits
        if "capital_allocation" in state_data:
            allocation = state_data["capital_allocation"]
            if isinstance(allocation, (int, float)):
                if allocation <= 0:
                    violations.append("Capital allocation must be positive")
                elif allocation > 1000000:  # $1M limit
                    violations.append("Capital allocation exceeds maximum limit of $1M")

        # Configuration validation
        if "config" in state_data:
            config = state_data["config"]
            if isinstance(config, dict):
                if not config.get("exchange"):
                    violations.append("Bot configuration must specify exchange")

        return violations

    async def _validate_position_business_rules(self, state_data: dict[str, Any]) -> list[str]:
        """Validate position-specific business rules."""
        violations = []

        # Stop loss validation
        entry_price = state_data.get("entry_price")
        stop_loss = state_data.get("stop_loss")
        side = state_data.get("side", "").lower()

        if entry_price and stop_loss and side:
            if side in ["buy", "long"] and stop_loss >= entry_price:
                violations.append("Stop loss must be below entry price for long positions")
            elif side in ["sell", "short"] and stop_loss <= entry_price:
                violations.append("Stop loss must be above entry price for short positions")

        return violations

    async def _validate_order_business_rules(self, state_data: dict[str, Any]) -> list[str]:
        """Validate order-specific business rules."""
        violations = []

        # Order size limits
        quantity = state_data.get("quantity")
        if isinstance(quantity, (int, float)):
            if quantity > 1000000:  # Large order limit
                violations.append("Order quantity exceeds maximum limit")

        return violations

    async def _validate_risk_business_rules(self, state_data: dict[str, Any]) -> list[str]:
        """Validate risk-specific business rules."""
        violations = []

        # VaR limits
        var = state_data.get("var")
        max_var = state_data.get("max_var", 0.05)  # 5% default
        if isinstance(var, (int, float)) and isinstance(max_var, (int, float)):
            if var > max_var:
                violations.append(f"VaR {var:.3f} exceeds maximum {max_var:.3f}")

        # Exposure limits
        exposure = state_data.get("exposure")
        max_exposure = state_data.get("max_exposure")
        if isinstance(exposure, (int, float)) and isinstance(max_exposure, (int, float)):
            if exposure > max_exposure:
                violations.append(f"Exposure {exposure} exceeds maximum {max_exposure}")

        return violations

    async def _validate_general_business_rules(self, state_data: dict[str, Any]) -> list[str]:
        """Validate general business rules."""
        violations = []

        # Timestamp consistency
        if "created_at" in state_data and "updated_at" in state_data:
            try:
                created_at = state_data["created_at"]
                updated_at = state_data["updated_at"]

                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

                if updated_at < created_at:
                    violations.append("updated_at cannot be before created_at")

            except (ValueError, TypeError):
                violations.append("Invalid timestamp format")

        return violations

    def _extract_status_field(
        self, state_data: dict[str, Any], state_type: "StateType"
    ) -> str | None:
        """Extract status field from state data."""
        status_fields = {
            "bot_state": "status",
            "order_state": "status",
            "position_state": "state",
            "strategy_state": "status",
        }

        field_name = status_fields.get(state_type.value)
        if field_name and field_name in state_data:
            value = state_data[field_name]
            if hasattr(value, "value"):
                return value.value
            return str(value)

        return None

    def _get_valid_transitions(self, state_type: "StateType") -> dict[str, set[str]]:
        """Get valid state transitions for a state type."""
        transition_map = {
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
        return transition_map.get(state_type.value, {})

    # Caching methods

    async def _get_cached_validation(
        self, state_type: "StateType", state_data: dict[str, Any], validation_level: str
    ) -> dict[str, Any] | None:
        """Get cached validation result if available and not expired."""
        cache_key = self._generate_cache_key(state_type, state_data, validation_level)

        if cache_key in self._validation_cache:
            result, cached_time = self._validation_cache[cache_key]

            # Check if cache entry is expired
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_ttl_seconds:
                return result
            else:
                # Remove expired entry
                del self._validation_cache[cache_key]

        return None

    async def _cache_validation_result(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
        validation_level: str,
        result: dict[str, Any],
    ) -> None:
        """Cache validation result."""
        cache_key = self._generate_cache_key(state_type, state_data, validation_level)
        self._validation_cache[cache_key] = (result, datetime.now(timezone.utc))

        # Cleanup cache if it gets too large
        if len(self._validation_cache) > 1000:
            # Remove oldest 50% of entries
            sorted_items = sorted(self._validation_cache.items(), key=lambda x: x[1][1])
            keep_count = len(sorted_items) // 2
            self._validation_cache = dict(sorted_items[-keep_count:])

    def _generate_cache_key(
        self, state_type: "StateType", state_data: dict[str, Any], validation_level: str
    ) -> str:
        """Generate cache key for validation result."""
        import hashlib
        import json

        data_hash = hashlib.md5(
            json.dumps(state_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return f"{state_type.value}:{validation_level}:{data_hash}"
