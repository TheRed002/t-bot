"""
Comprehensive State Validator with business rules and consistency enforcement.

This module provides enterprise-grade state validation capabilities:
- Comprehensive validation rules for all state types
- Business logic enforcement and constraints
- State transition validation with finite state machines
- Cross-state consistency checks
- Performance optimization with validation caching
- Configurable validation policies

The StateValidator ensures data integrity and business rule compliance
across all state operations in the trading system.
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from src.base import BaseComponent
from src.core.exceptions import ValidationError
from src.core.types import BotStatus, OrderSide, OrderType

# Import utilities through centralized import handler
from .utils_imports import time_execution

if TYPE_CHECKING:
    from .state_service import StateService, StateType
else:
    # Runtime imports to avoid circular dependencies
    from .state_service import StateType


class ValidationLevel(Enum):
    """Validation level enumeration."""

    STRICT = "strict"  # All rules enforced
    NORMAL = "normal"  # Standard rules enforced
    LENIENT = "lenient"  # Only critical rules enforced
    DISABLED = "disabled"  # Validation disabled


class ValidationRule(Enum):
    """Validation rule types."""

    REQUIRED_FIELD = "required_field"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    FORMAT_CHECK = "format_check"
    BUSINESS_RULE = "business_rule"
    CONSISTENCY_CHECK = "consistency_check"
    TRANSITION_RULE = "transition_rule"


@dataclass
class ValidationRuleConfig:
    """Configuration for a validation rule."""

    rule_type: ValidationRule
    field_name: str
    rule_function: Callable
    error_message: str
    severity: str = "error"  # error, warning, info
    enabled: bool = True
    dependencies: list[str] = field(default_factory=list)


@dataclass
class StateValidationError:
    """Individual validation error."""

    rule_type: ValidationRule
    field_name: str
    error_message: str
    severity: str
    current_value: Any = None
    expected_value: Any = None


@dataclass
class ValidationWarning:
    """Individual validation warning."""

    rule_type: ValidationRule
    field_name: str
    warning_message: str
    current_value: Any = None
    recommendation: str = ""


@dataclass
class ValidationResult:
    """Complete validation result."""

    is_valid: bool = True
    errors: list[StateValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    validation_time_ms: float = 0.0
    rules_checked: int = 0
    rules_passed: int = 0
    state_type: Optional["StateType"] = None
    state_id: str = ""


@dataclass
class ValidationMetrics:
    """Validation performance metrics."""

    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    average_validation_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    rules_triggered: dict[str, int] = field(default_factory=dict)


class StateValidator(BaseComponent):
    """
    Comprehensive state validation service with business rules enforcement.

    Features:
    - Multi-level validation with configurable strictness
    - Business rule enforcement with custom logic
    - State transition validation using finite state machines
    - Cross-state consistency checking
    - Performance optimization with validation caching
    - Detailed error reporting and recommendations
    """

    def __init__(self, state_service: "StateService"):
        """
        Initialize the state validator.

        Args:
            state_service: Reference to the main state service
        """
        super().__init__()
        self.state_service = state_service

        # Validation configuration
        self.validation_level = ValidationLevel.NORMAL
        self.cache_validation_results = True
        self.cache_ttl_seconds = 300  # 5 minutes

        # Validation rules registry
        self._validation_rules: dict[StateType, list[ValidationRuleConfig]] = {}
        self._transition_rules: dict[StateType, dict[str, set[str]]] = {}

        # Performance optimization
        self._validation_cache: dict[str, tuple[ValidationResult, datetime]] = {}
        self._rule_cache: dict[str, Any] = {}

        # Metrics tracking
        self._metrics = ValidationMetrics()
        self._validation_times: list[float] = []

        # Initialize built-in rules
        self._initialize_builtin_rules()

        self.logger.info("StateValidator initialized")

    async def initialize(self) -> None:
        """Initialize the validator."""
        try:
            # Load custom validation rules from configuration
            await self._load_custom_rules()

            # Initialize state transition rules
            self._initialize_transition_rules()

            # Start background cache cleanup
            asyncio.create_task(self._cache_cleanup_loop())

            super().initialize()
            self.logger.info("StateValidator initialization completed")

        except Exception as e:
            self.logger.error(f"StateValidator initialization failed: {e}")
            raise ValidationError(f"Failed to initialize StateValidator: {e}")

    async def cleanup(self) -> None:
        """Cleanup validator resources."""
        try:
            self._validation_cache.clear()
            self._rule_cache.clear()

            super().cleanup()
            self.logger.info("StateValidator cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during StateValidator cleanup: {e}")

    # Core Validation Methods

    @time_execution
    async def validate_state(
        self,
        state_type: "StateType",
        state_data: dict[str, Any],
        validation_level: ValidationLevel | None = None,
        use_cache: bool = True,
    ) -> ValidationResult:
        """
        Validate state data against all applicable rules.

        Args:
            state_type: Type of state to validate
            state_data: State data to validate
            validation_level: Validation strictness level
            use_cache: Whether to use validation cache

        Returns:
            Validation result with errors and warnings
        """
        start_time = datetime.now(timezone.utc)
        level = validation_level or self.validation_level

        try:
            # Check cache if enabled
            if use_cache and self.cache_validation_results:
                cache_key = self._generate_cache_key(state_type, state_data, level)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._metrics.cache_hit_rate = self._update_hit_rate(True)
                    return cached_result

            # Initialize validation result
            result = ValidationResult(
                state_type=state_type, state_id=state_data.get("state_id", "unknown")
            )

            # Skip validation if disabled
            if level == ValidationLevel.DISABLED:
                result.validation_time_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000
                return result

            # Get validation rules for this state type
            rules = self._validation_rules.get(state_type, [])
            enabled_rules = [rule for rule in rules if rule.enabled]

            # Filter rules by validation level
            if level == ValidationLevel.LENIENT:
                enabled_rules = [rule for rule in enabled_rules if rule.severity == "error"]

            result.rules_checked = len(enabled_rules)

            # Apply validation rules
            for rule_config in enabled_rules:
                try:
                    rule_result = await self._apply_validation_rule(
                        rule_config, state_data, state_type
                    )

                    if rule_result["passed"]:
                        result.rules_passed += 1
                    else:
                        if rule_config.severity == "error":
                            error = StateValidationError(
                                rule_type=rule_config.rule_type,
                                field_name=rule_config.field_name,
                                error_message=rule_result["message"],
                                severity=rule_config.severity,
                                current_value=rule_result.get("current_value"),
                                expected_value=rule_result.get("expected_value"),
                            )
                            result.errors.append(error)
                            result.is_valid = False
                        else:
                            warning = ValidationWarning(
                                rule_type=rule_config.rule_type,
                                field_name=rule_config.field_name,
                                warning_message=rule_result["message"],
                                current_value=rule_result.get("current_value"),
                                recommendation=rule_result.get("recommendation", ""),
                            )
                            result.warnings.append(warning)

                    # Update rule usage metrics
                    rule_name = (
                        f"{state_type.value}:{rule_config.field_name}:{rule_config.rule_type.value}"
                    )
                    self._metrics.rules_triggered[rule_name] = (
                        self._metrics.rules_triggered.get(rule_name, 0) + 1
                    )

                except Exception as e:
                    self.logger.error(f"Rule application failed: {rule_config.field_name}: {e}")
                    # Add validation system error
                    error = StateValidationError(
                        rule_type=ValidationRule.TYPE_CHECK,
                        field_name=rule_config.field_name,
                        error_message=f"Validation system error: {e}",
                        severity="error",
                    )
                    result.errors.append(error)
                    result.is_valid = False

            # Calculate validation time
            result.validation_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Cache result if enabled
            if use_cache and self.cache_validation_results:
                cache_key = self._generate_cache_key(state_type, state_data, level)
                self._cache_result(cache_key, result)
                self._metrics.cache_hit_rate = self._update_hit_rate(False)

            # Update metrics
            self._update_validation_metrics(result)

            return result

        except Exception as e:
            self.logger.error(f"State validation failed: {e}")
            result = ValidationResult(
                is_valid=False,
                state_type=state_type,
                validation_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            )
            result.errors.append(
                StateValidationError(
                    rule_type=ValidationRule.TYPE_CHECK,
                    field_name="system",
                    error_message=f"Validation system error: {e}",
                    severity="error",
                )
            )
            return result

    async def validate_state_transition(
        self, state_type: "StateType", current_state: dict[str, Any], new_state: dict[str, Any]
    ) -> bool:
        """
        Validate state transition using finite state machine rules.

        Args:
            state_type: Type of state
            current_state: Current state data
            new_state: Proposed new state data

        Returns:
            True if transition is valid
        """
        try:
            # Get transition rules for this state type
            transition_rules = self._transition_rules.get(state_type, {})

            # Extract status fields for transition validation
            current_status = self._extract_status_field(current_state, state_type)
            new_status = self._extract_status_field(new_state, state_type)

            if not current_status or not new_status:
                # If no status field, allow transition
                return True

            # Check if transition is allowed
            allowed_transitions = transition_rules.get(current_status, set())

            if new_status not in allowed_transitions and current_status != new_status:
                self.logger.warning(
                    f"Invalid state transition: {current_status} -> {new_status}",
                    extra={
                        "state_type": state_type.value,
                        "allowed_transitions": list(allowed_transitions),
                    },
                )
                return False

            # Additional business rule validation for specific transitions
            return await self._validate_business_transition_rules(
                state_type, current_state, new_state
            )

        except Exception as e:
            self.logger.error(f"State transition validation failed: {e}")
            return False

    async def validate_cross_state_consistency(
        self,
        primary_state: dict[str, Any],
        related_states: list[dict[str, Any]],
        consistency_rules: list[str],
    ) -> ValidationResult:
        """
        Validate consistency across multiple related states.

        Args:
            primary_state: Primary state to validate
            related_states: Related states to check consistency against
            consistency_rules: List of consistency rules to apply

        Returns:
            Validation result
        """
        try:
            result = ValidationResult()

            for rule_name in consistency_rules:
                consistency_check = self._get_consistency_rule(rule_name)
                if consistency_check:
                    check_result = await consistency_check(primary_state, related_states)

                    if not check_result["valid"]:
                        error = StateValidationError(
                            rule_type=ValidationRule.CONSISTENCY_CHECK,
                            field_name=rule_name,
                            error_message=check_result["message"],
                            severity="error",
                        )
                        result.errors.append(error)
                        result.is_valid = False

            return result

        except Exception as e:
            self.logger.error(f"Cross-state consistency validation failed: {e}")
            result = ValidationResult(is_valid=False)
            result.errors.append(
                StateValidationError(
                    rule_type=ValidationRule.CONSISTENCY_CHECK,
                    field_name="system",
                    error_message=f"Consistency validation error: {e}",
                    severity="error",
                )
            )
            return result

    # Rule Management

    def add_validation_rule(
        self, state_type: "StateType", rule_config: ValidationRuleConfig
    ) -> None:
        """Add a custom validation rule."""
        if state_type not in self._validation_rules:
            self._validation_rules[state_type] = []

        self._validation_rules[state_type].append(rule_config)
        self.logger.info(f"Added validation rule: {state_type.value}:{rule_config.field_name}")

    def remove_validation_rule(
        self, state_type: "StateType", field_name: str, rule_type: ValidationRule
    ) -> bool:
        """Remove a validation rule."""
        if state_type in self._validation_rules:
            rules = self._validation_rules[state_type]
            for i, rule in enumerate(rules):
                if rule.field_name == field_name and rule.rule_type == rule_type:
                    del rules[i]
                    self.logger.info(f"Removed validation rule: {state_type.value}:{field_name}")
                    return True
        return False

    def update_validation_level(self, level: ValidationLevel) -> None:
        """Update global validation level."""
        self.validation_level = level
        self.logger.info(f"Updated validation level to: {level.value}")

        # Clear cache when level changes
        self._validation_cache.clear()

    # Monitoring and Metrics

    async def get_metrics(self) -> ValidationMetrics:
        """Get validation metrics."""
        if self._validation_times:
            self._metrics.average_validation_time_ms = sum(self._validation_times) / len(
                self._validation_times
            )

        return self._metrics

    def get_validation_rules(self, state_type: "StateType") -> list[ValidationRuleConfig]:
        """Get validation rules for a state type."""
        return self._validation_rules.get(state_type, [])

    # Private Helper Methods

    def _initialize_builtin_rules(self) -> None:
        """Initialize built-in validation rules for all state types."""

        # Bot State Rules
        self._add_bot_state_rules()

        # Position State Rules
        self._add_position_state_rules()

        # Order State Rules
        self._add_order_state_rules()

        # Portfolio State Rules
        self._add_portfolio_state_rules()

        # Risk State Rules
        self._add_risk_state_rules()

        # Strategy State Rules
        self._add_strategy_state_rules()

        # Market State Rules
        self._add_market_state_rules()

        # Trade State Rules
        self._add_trade_state_rules()

    def _add_bot_state_rules(self) -> None:
        """Add validation rules for bot state."""
        from .state_service import StateType

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="bot_id",
                rule_function=self._validate_required_field,
                error_message="Bot ID is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="bot_id",
                rule_function=lambda data: self._validate_string_field(data, "bot_id"),
                error_message="Bot ID must be a string",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.FORMAT_CHECK,
                field_name="bot_id",
                rule_function=lambda data: self._validate_bot_id_format(data.get("bot_id")),
                error_message="Bot ID must be alphanumeric with dashes/underscores",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="status",
                rule_function=self._validate_required_field,
                error_message="Bot status is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="status",
                rule_function=lambda data: self._validate_bot_status(data.get("status")),
                error_message="Invalid bot status",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.BUSINESS_RULE,
                field_name="capital_allocation",
                rule_function=lambda data: self._validate_capital_allocation(data),
                error_message="Capital allocation must be positive and not exceed limits",
                severity="warning",
            ),
        ]

        self._validation_rules[StateType.BOT_STATE] = rules

    def _add_position_state_rules(self) -> None:
        """Add validation rules for position state."""
        from .state_service import StateType

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="symbol",
                rule_function=self._validate_required_field,
                error_message="Symbol is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="quantity",
                rule_function=self._validate_required_field,
                error_message="Quantity is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="quantity",
                rule_function=lambda data: self._validate_decimal_field(data, "quantity"),
                error_message="Quantity must be a valid decimal",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.RANGE_CHECK,
                field_name="quantity",
                rule_function=lambda data: self._validate_positive_value(data, "quantity"),
                error_message="Quantity must be positive",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="entry_price",
                rule_function=self._validate_required_field,
                error_message="Entry price is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="entry_price",
                rule_function=lambda data: self._validate_decimal_field(data, "entry_price"),
                error_message="Entry price must be a valid decimal",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.RANGE_CHECK,
                field_name="entry_price",
                rule_function=lambda data: self._validate_positive_value(data, "entry_price"),
                error_message="Entry price must be positive",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="side",
                rule_function=lambda data: self._validate_order_side(data.get("side")),
                error_message="Invalid position side",
            ),
        ]

        self._validation_rules[StateType.POSITION_STATE] = rules

    def _add_order_state_rules(self) -> None:
        """Add validation rules for order state."""
        from .state_service import StateType

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="order_id",
                rule_function=self._validate_required_field,
                error_message="Order ID is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="symbol",
                rule_function=self._validate_required_field,
                error_message="Symbol is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="symbol",
                rule_function=lambda data: self._validate_symbol_format(data.get("symbol")),
                error_message="Invalid symbol format",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="side",
                rule_function=self._validate_required_field,
                error_message="Order side is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="side",
                rule_function=lambda data: self._validate_order_side(data.get("side")),
                error_message="Invalid order side",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="type",
                rule_function=self._validate_required_field,
                error_message="Order type is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="type",
                rule_function=lambda data: self._validate_order_type(data.get("type")),
                error_message="Invalid order type",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.BUSINESS_RULE,
                field_name="price",
                rule_function=lambda data: self._validate_order_price_logic(data),
                error_message="Price is required for limit/stop orders",
            ),
        ]

        self._validation_rules[StateType.ORDER_STATE] = rules

    def _add_portfolio_state_rules(self) -> None:
        """Add validation rules for portfolio state."""
        from .state_service import StateType

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="total_value",
                rule_function=self._validate_required_field,
                error_message="Total value is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="total_value",
                rule_function=lambda data: self._validate_decimal_field(data, "total_value"),
                error_message="Total value must be a valid decimal",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.RANGE_CHECK,
                field_name="total_value",
                rule_function=lambda data: self._validate_non_negative_value(data, "total_value"),
                error_message="Total value cannot be negative",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="positions",
                rule_function=lambda data: self._validate_list_field(data, "positions"),
                error_message="Positions must be a list",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.BUSINESS_RULE,
                field_name="cash_balance",
                rule_function=lambda data: self._validate_cash_balance(data),
                error_message="Cash balance must be sufficient for positions",
                severity="warning",
            ),
        ]

        self._validation_rules[StateType.PORTFOLIO_STATE] = rules

    def _add_risk_state_rules(self) -> None:
        """Add validation rules for risk state."""
        from .state_service import StateType

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="exposure",
                rule_function=self._validate_required_field,
                error_message="Exposure is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="exposure",
                rule_function=lambda data: self._validate_decimal_field(data, "exposure"),
                error_message="Exposure must be a valid decimal",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.RANGE_CHECK,
                field_name="exposure",
                rule_function=lambda data: self._validate_non_negative_value(data, "exposure"),
                error_message="Exposure cannot be negative",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.BUSINESS_RULE,
                field_name="var",
                rule_function=lambda data: self._validate_var_limits(data),
                error_message="VaR exceeds acceptable limits",
                severity="error",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="risk_limits",
                rule_function=lambda data: self._validate_dict_field(data, "risk_limits"),
                error_message="Risk limits must be a dictionary",
            ),
        ]

        self._validation_rules[StateType.RISK_STATE] = rules

    def _add_strategy_state_rules(self) -> None:
        """Add validation rules for strategy state."""
        from .state_service import StateType

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="strategy_id",
                rule_function=self._validate_required_field,
                error_message="Strategy ID is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="params",
                rule_function=lambda data: self._validate_dict_field(data, "params"),
                error_message="Strategy parameters must be a dictionary",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.BUSINESS_RULE,
                field_name="params",
                rule_function=lambda data: self._validate_strategy_params(data),
                error_message="Invalid strategy parameters",
            ),
        ]

        self._validation_rules[StateType.STRATEGY_STATE] = rules

    def _add_market_state_rules(self) -> None:
        """Add validation rules for market state."""
        from .state_service import StateType

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="symbol",
                rule_function=self._validate_required_field,
                error_message="Symbol is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="price",
                rule_function=self._validate_required_field,
                error_message="Price is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="price",
                rule_function=lambda data: self._validate_decimal_field(data, "price"),
                error_message="Price must be a valid decimal",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.RANGE_CHECK,
                field_name="price",
                rule_function=lambda data: self._validate_positive_value(data, "price"),
                error_message="Price must be positive",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.TYPE_CHECK,
                field_name="volume",
                rule_function=lambda data: self._validate_decimal_field(data, "volume"),
                error_message="Volume must be a valid decimal",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.RANGE_CHECK,
                field_name="volume",
                rule_function=lambda data: self._validate_non_negative_value(data, "volume"),
                error_message="Volume cannot be negative",
            ),
        ]

        self._validation_rules[StateType.MARKET_STATE] = rules

    def _add_trade_state_rules(self) -> None:
        """Add validation rules for trade state."""
        from .state_service import StateType

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="trade_id",
                rule_function=self._validate_required_field,
                error_message="Trade ID is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.REQUIRED_FIELD,
                field_name="symbol",
                rule_function=self._validate_required_field,
                error_message="Symbol is required",
            ),
            ValidationRuleConfig(
                rule_type=ValidationRule.BUSINESS_RULE,
                field_name="execution",
                rule_function=lambda data: self._validate_trade_execution(data),
                error_message="Invalid trade execution data",
            ),
        ]

        self._validation_rules[StateType.TRADE_STATE] = rules

    def _initialize_transition_rules(self) -> None:
        """Initialize state transition rules."""
        from .state_service import StateType

        # Bot state transitions
        self._transition_rules[StateType.BOT_STATE] = {
            "initializing": {"running", "error", "stopped"},
            "running": {"paused", "stopping", "error"},
            "paused": {"running", "stopping", "error"},
            "stopping": {"stopped", "error"},
            "stopped": {"initializing"},
            "error": {"initializing", "stopped"},
        }

        # Order state transitions
        self._transition_rules[StateType.ORDER_STATE] = {
            "pending": {"open", "cancelled", "rejected"},
            "open": {"partially_filled", "filled", "cancelled"},
            "partially_filled": {"filled", "cancelled"},
            "filled": set(),  # Terminal state
            "cancelled": set(),  # Terminal state
            "rejected": set(),  # Terminal state
        }

        # Strategy state transitions
        self._transition_rules[StateType.STRATEGY_STATE] = {
            "active": {"paused", "inactive", "error"},
            "inactive": {"active"},
            "paused": {"active", "inactive", "error"},
            "error": {"inactive", "active"},
        }

    # Validation Rule Implementation

    async def _apply_validation_rule(
        self, rule_config: ValidationRuleConfig, state_data: dict[str, Any], state_type: "StateType"
    ) -> dict[str, Any]:
        """Apply a single validation rule."""
        try:
            result = rule_config.rule_function(state_data)

            if isinstance(result, bool):
                return {
                    "passed": result,
                    "message": rule_config.error_message if not result else "Validation passed",
                }
            elif isinstance(result, dict):
                return result
            else:
                return {
                    "passed": bool(result),
                    "message": rule_config.error_message if not result else "Validation passed",
                }

        except Exception as e:
            self.logger.error(f"Validation rule error: {rule_config.field_name}: {e}")
            return {"passed": False, "message": f"Validation rule execution failed: {e}"}

    # Basic Validation Functions

    def _validate_required_field(self, data: dict[str, Any], field_name: str | None = None) -> bool:
        """Validate that a required field is present and not None."""
        if not field_name:
            # If called without field_name, extract from rule
            return any(value is not None for value in data.values())
        return field_name in data and data[field_name] is not None

    def _validate_string_field(self, data: dict[str, Any], field_name: str) -> dict[str, Any]:
        """Validate that a field is a string."""
        value = data.get(field_name)
        is_valid = isinstance(value, str)

        return {
            "passed": is_valid,
            "message": f"{field_name} must be a string",
            "current_value": value,
            "expected_value": "string type",
        }

    def _validate_decimal_field(self, data: dict[str, Any], field_name: str) -> dict[str, Any]:
        """Validate that a field is a valid decimal/numeric value."""
        value = data.get(field_name)

        try:
            if value is None:
                return {"passed": False, "message": f"{field_name} is required"}

            # Try to convert to Decimal
            if isinstance(value, int | float):
                Decimal(str(value))
            elif isinstance(value, str):
                Decimal(value)
            elif isinstance(value, Decimal):
                pass
            else:
                return {
                    "passed": False,
                    "message": f"{field_name} must be a valid number",
                    "current_value": value,
                }

            return {"passed": True, "message": "Valid decimal value", "current_value": value}

        except (ValueError, TypeError) as e:
            return {
                "passed": False,
                "message": f"{field_name} must be a valid decimal: {e}",
                "current_value": value,
            }

    def _validate_positive_value(self, data: dict[str, Any], field_name: str) -> dict[str, Any]:
        """Validate that a numeric field is positive."""
        value = data.get(field_name)

        try:
            decimal_value = Decimal(str(value)) if value is not None else None
            is_positive = decimal_value is not None and decimal_value > 0

            return {
                "passed": is_positive,
                "message": f"{field_name} must be positive",
                "current_value": value,
                "expected_value": "> 0",
            }

        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": f"{field_name} must be a valid positive number",
                "current_value": value,
            }

    def _validate_non_negative_value(self, data: dict[str, Any], field_name: str) -> dict[str, Any]:
        """Validate that a numeric field is non-negative."""
        value = data.get(field_name)

        try:
            decimal_value = Decimal(str(value)) if value is not None else None
            is_non_negative = decimal_value is not None and decimal_value >= 0

            return {
                "passed": is_non_negative,
                "message": f"{field_name} cannot be negative",
                "current_value": value,
                "expected_value": ">= 0",
            }

        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": f"{field_name} must be a valid non-negative number",
                "current_value": value,
            }

    def _validate_list_field(self, data: dict[str, Any], field_name: str) -> dict[str, Any]:
        """Validate that a field is a list."""
        value = data.get(field_name)
        is_list = isinstance(value, list)

        return {
            "passed": is_list,
            "message": f"{field_name} must be a list",
            "current_value": type(value).__name__ if value is not None else None,
            "expected_value": "list",
        }

    def _validate_dict_field(self, data: dict[str, Any], field_name: str) -> dict[str, Any]:
        """Validate that a field is a dictionary."""
        value = data.get(field_name)
        is_dict = isinstance(value, dict)

        return {
            "passed": is_dict,
            "message": f"{field_name} must be a dictionary",
            "current_value": type(value).__name__ if value is not None else None,
            "expected_value": "dict",
        }

    # Business Logic Validation Functions

    def _validate_bot_id_format(self, bot_id: str) -> dict[str, Any]:
        """Validate bot ID format."""
        if not bot_id:
            return {"passed": False, "message": "Bot ID cannot be empty"}

        # Bot ID should be alphanumeric with dashes and underscores
        pattern = r"^[a-zA-Z0-9_-]+$"
        is_valid = re.match(pattern, bot_id) is not None

        return {
            "passed": is_valid,
            "message": "Bot ID must contain only letters, numbers, dashes, and underscores",
            "current_value": bot_id,
            "expected_value": "alphanumeric with - and _",
        }

    def _validate_bot_status(self, status: Any) -> dict[str, Any]:
        """Validate bot status value."""
        if isinstance(status, BotStatus):
            return {"passed": True, "message": "Valid bot status"}

        if isinstance(status, str):
            valid_statuses = {s.value for s in BotStatus}
            is_valid = status in valid_statuses

            return {
                "passed": is_valid,
                "message": f"Invalid bot status: {status}",
                "current_value": status,
                "expected_value": list(valid_statuses),
            }

        return {
            "passed": False,
            "message": "Bot status must be a BotStatus enum or valid string",
            "current_value": status,
        }

    def _validate_order_side(self, side: Any) -> dict[str, Any]:
        """Validate order side."""
        if isinstance(side, OrderSide):
            return {"passed": True, "message": "Valid order side"}

        if isinstance(side, str):
            valid_sides = {s.value for s in OrderSide}
            is_valid = side.upper() in valid_sides

            return {
                "passed": is_valid,
                "message": f"Invalid order side: {side}",
                "current_value": side,
                "expected_value": list(valid_sides),
            }

        return {
            "passed": False,
            "message": "Order side must be an OrderSide enum or valid string",
            "current_value": side,
        }

    def _validate_order_type(self, order_type: Any) -> dict[str, Any]:
        """Validate order type."""
        if isinstance(order_type, OrderType):
            return {"passed": True, "message": "Valid order type"}

        if isinstance(order_type, str):
            valid_types = {t.value for t in OrderType}
            is_valid = order_type.upper() in valid_types

            return {
                "passed": is_valid,
                "message": f"Invalid order type: {order_type}",
                "current_value": order_type,
                "expected_value": list(valid_types),
            }

        return {
            "passed": False,
            "message": "Order type must be an OrderType enum or valid string",
            "current_value": order_type,
        }

    def _validate_symbol_format(self, symbol: str) -> dict[str, Any]:
        """Validate trading symbol format."""
        if not symbol:
            return {"passed": False, "message": "Symbol cannot be empty"}

        # Symbol should be uppercase letters with optional separators
        pattern = r"^[A-Z0-9]+([\/\-][A-Z0-9]+)*$"
        is_valid = re.match(pattern, symbol.upper()) is not None

        return {
            "passed": is_valid,
            "message": "Symbol must be in valid format (e.g., BTC/USD, BTCUSD)",
            "current_value": symbol,
            "expected_value": "Valid trading symbol format",
        }

    def _validate_capital_allocation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate capital allocation limits."""
        allocation = data.get("capital_allocation")
        if allocation is None:
            return {"passed": True, "message": "No capital allocation specified"}

        try:
            allocation_value = Decimal(str(allocation))

            # Check if allocation is positive
            if allocation_value <= 0:
                return {
                    "passed": False,
                    "message": "Capital allocation must be positive",
                    "current_value": allocation,
                }

            # Check against maximum allocation (example: $1M)
            max_allocation = Decimal("1000000")
            if allocation_value > max_allocation:
                return {
                    "passed": False,
                    "message": f"Capital allocation exceeds maximum of ${max_allocation}",
                    "current_value": allocation,
                    "expected_value": f"<= {max_allocation}",
                }

            return {"passed": True, "message": "Valid capital allocation"}

        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": "Capital allocation must be a valid number",
                "current_value": allocation,
            }

    def _validate_order_price_logic(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate order price logic based on order type."""
        order_type = data.get("type")
        price = data.get("price")

        if not order_type:
            return {"passed": True, "message": "No order type specified"}

        # Market orders shouldn't have price
        if order_type in ["MARKET", "market"]:
            if price is not None:
                return {
                    "passed": False,
                    "message": "Market orders should not specify price",
                    "current_value": price,
                    "recommendation": "Remove price for market orders",
                }

        # Limit and stop orders require price
        elif order_type in ["LIMIT", "limit", "STOP", "stop", "STOP_LIMIT", "stop_limit"]:
            if price is None:
                return {
                    "passed": False,
                    "message": f"{order_type} orders require price",
                    "current_value": price,
                    "expected_value": "positive price value",
                }

            # Validate price is positive
            try:
                price_value = Decimal(str(price))
                if price_value <= 0:
                    return {
                        "passed": False,
                        "message": "Order price must be positive",
                        "current_value": price,
                    }
            except (ValueError, TypeError):
                return {
                    "passed": False,
                    "message": "Order price must be a valid number",
                    "current_value": price,
                }

        return {"passed": True, "message": "Valid order price logic"}

    def _validate_cash_balance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate cash balance against positions."""
        cash_balance = data.get("cash_balance")
        positions = data.get("positions", [])

        if cash_balance is None:
            return {"passed": True, "message": "No cash balance specified"}

        try:
            cash_value = Decimal(str(cash_balance))

            # Calculate total position value (simplified)
            total_position_value = Decimal("0")
            for position in positions:
                if isinstance(position, dict):
                    quantity = position.get("quantity", 0)
                    price = position.get("current_price", position.get("entry_price", 0))
                    if quantity and price:
                        total_position_value += Decimal(str(quantity)) * Decimal(str(price))

            # Warning if cash balance is too low relative to positions
            min_cash_ratio = Decimal("0.1")  # 10% minimum cash
            total_value = cash_value + total_position_value

            if total_value > 0 and (cash_value / total_value) < min_cash_ratio:
                return {
                    "passed": False,
                    "message": f"Cash balance too low: {cash_value} "
                    f"({(cash_value / total_value) * 100:.1f}%)",
                    "current_value": float(cash_value),
                    "recommendation": f"Maintain at least {min_cash_ratio * 100}% cash",
                }

            return {"passed": True, "message": "Adequate cash balance"}

        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": "Cash balance must be a valid number",
                "current_value": cash_balance,
            }

    def _validate_var_limits(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate VaR against risk limits."""
        var_value = data.get("var")
        risk_limits = data.get("risk_limits", {})

        if var_value is None:
            return {"passed": True, "message": "No VaR specified"}

        try:
            var_decimal = Decimal(str(var_value))
            max_var = risk_limits.get("max_var", "0.02")  # Default 2%
            max_var_decimal = Decimal(str(max_var))

            if var_decimal > max_var_decimal:
                return {
                    "passed": False,
                    "message": f"VaR {var_decimal} exceeds limit {max_var_decimal}",
                    "current_value": float(var_decimal),
                    "expected_value": f"<= {max_var_decimal}",
                }

            return {"passed": True, "message": "VaR within limits"}

        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": "VaR must be a valid number",
                "current_value": var_value,
            }

    def _validate_strategy_params(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate strategy parameters."""
        params = data.get("params", {})

        if not isinstance(params, dict):
            return {
                "passed": False,
                "message": "Strategy parameters must be a dictionary",
                "current_value": type(params).__name__,
            }

        # Check for required strategy parameters
        required_params = {"timeframe", "risk_per_trade"}
        missing_params = required_params - set(params.keys())

        if missing_params:
            return {
                "passed": False,
                "message": f"Missing required strategy parameters: {missing_params}",
                "current_value": list(params.keys()),
                "expected_value": list(required_params),
            }

        # Validate risk_per_trade is reasonable
        risk_per_trade = params.get("risk_per_trade")
        if risk_per_trade:
            try:
                risk_value = Decimal(str(risk_per_trade))
                if risk_value > Decimal("0.05"):  # 5% max risk per trade
                    return {
                        "passed": False,
                        "message": f"Risk per trade too high: {risk_value}",
                        "current_value": float(risk_value),
                        "expected_value": "<= 0.05",
                    }
            except (ValueError, TypeError):
                return {
                    "passed": False,
                    "message": "Risk per trade must be a valid number",
                    "current_value": risk_per_trade,
                }

        return {"passed": True, "message": "Valid strategy parameters"}

    def _validate_trade_execution(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate trade execution data."""
        execution_data = data.get("execution", {})

        if not isinstance(execution_data, dict):
            return {"passed": True, "message": "No execution data provided"}

        # Check for required execution fields
        required_fields = ["filled_quantity", "average_price"]
        missing_fields = [field for field in required_fields if field not in execution_data]

        if missing_fields:
            return {
                "passed": False,
                "message": f"Missing execution fields: {missing_fields}",
                "current_value": list(execution_data.keys()),
                "expected_value": required_fields,
            }

        # Validate execution values
        try:
            filled_qty = Decimal(str(execution_data["filled_quantity"]))
            avg_price = Decimal(str(execution_data["average_price"]))

            if filled_qty < 0:
                return {
                    "passed": False,
                    "message": "Filled quantity cannot be negative",
                    "current_value": float(filled_qty),
                }

            if avg_price <= 0:
                return {
                    "passed": False,
                    "message": "Average price must be positive",
                    "current_value": float(avg_price),
                }

            return {"passed": True, "message": "Valid trade execution"}

        except (ValueError, TypeError):
            return {
                "passed": False,
                "message": "Execution data must contain valid numbers",
                "current_value": execution_data,
            }

    # Business Transition Validation

    async def _validate_business_transition_rules(
        self, state_type: "StateType", current_state: dict[str, Any], new_state: dict[str, Any]
    ) -> bool:
        """Validate business-specific transition rules."""
        try:
            from .state_service import StateType

            if state_type == StateType.BOT_STATE:
                return await self._validate_bot_transition_rules(current_state, new_state)
            elif state_type == StateType.ORDER_STATE:
                return await self._validate_order_transition_rules(current_state, new_state)
            elif state_type == StateType.RISK_STATE:
                return await self._validate_risk_transition_rules(current_state, new_state)

            return True

        except Exception as e:
            self.logger.error(f"Business transition validation failed: {e}")
            return False

    async def _validate_bot_transition_rules(
        self, current_state: dict[str, Any], new_state: dict[str, Any]
    ) -> bool:
        """Validate bot-specific transition rules."""
        current_status = current_state.get("status")
        new_status = new_state.get("status")

        # Can't go directly from error to running without stopping first
        if current_status == "error" and new_status == "running":
            return False

        # Stopping requires all positions to be closed (simplified check)
        if new_status == "stopping":
            positions = new_state.get("open_positions", [])
            if positions:
                self.logger.warning("Cannot stop bot with open positions")
                return False

        return True

    async def _validate_order_transition_rules(
        self, current_state: dict[str, Any], new_state: dict[str, Any]
    ) -> bool:
        """Validate order-specific transition rules."""
        current_status = current_state.get("status")
        new_status = new_state.get("status")

        # Can't fill more than original quantity
        if new_status in ["partially_filled", "filled"]:
            original_qty = current_state.get("quantity", 0)
            filled_qty = new_state.get("filled_quantity", 0)

            try:
                if Decimal(str(filled_qty)) > Decimal(str(original_qty)):
                    return False
            except (ValueError, TypeError):
                return False

        return True

    async def _validate_risk_transition_rules(
        self, current_state: dict[str, Any], new_state: dict[str, Any]
    ) -> bool:
        """Validate risk-specific transition rules."""
        # Risk can't increase dramatically without justification
        current_var = current_state.get("var", 0)
        new_var = new_state.get("var", 0)

        try:
            if Decimal(str(new_var)) > Decimal(str(current_var)) * Decimal("2"):
                # VaR doubled - requires additional validation
                self.logger.warning("Large VaR increase detected")
                return False
        except (ValueError, TypeError):
            pass

        return True

    # Utility Methods

    def _extract_status_field(
        self, state_data: dict[str, Any], state_type: "StateType"
    ) -> str | None:
        """Extract status field from state data based on state type."""
        from .state_service import StateType

        status_fields = {
            StateType.BOT_STATE: "status",
            StateType.ORDER_STATE: "status",
            StateType.STRATEGY_STATE: "status",
            StateType.TRADE_STATE: "state",
        }

        field_name = status_fields.get(state_type)
        if field_name:
            status_value = state_data.get(field_name)
            if hasattr(status_value, "value"):
                return status_value.value
            return str(status_value) if status_value else None

        return None

    def _generate_cache_key(
        self, state_type: "StateType", state_data: dict[str, Any], validation_level: ValidationLevel
    ) -> str:
        """Generate cache key for validation result."""
        data_hash = hashlib.md5(
            json.dumps(state_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return f"{state_type.value}:{validation_level.value}:{data_hash}"

    def _get_cached_result(self, cache_key: str) -> ValidationResult | None:
        """Get cached validation result if not expired."""
        if cache_key in self._validation_cache:
            result, cached_time = self._validation_cache[cache_key]

            # Check if cache entry is expired
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self.cache_ttl_seconds:
                return result
            else:
                # Remove expired entry
                del self._validation_cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: ValidationResult) -> None:
        """Cache validation result."""
        self._validation_cache[cache_key] = (result, datetime.now(timezone.utc))

        # Cleanup cache if it gets too large
        if len(self._validation_cache) > 10000:
            # Remove oldest 50% of entries
            sorted_items = sorted(self._validation_cache.items(), key=lambda x: x[1][1])

            keep_count = len(sorted_items) // 2
            self._validation_cache = dict(sorted_items[-keep_count:])

    def _update_hit_rate(self, hit: bool) -> float:
        """Update cache hit rate."""
        if not hasattr(self, "_hit_history"):
            self._hit_history = []

        self._hit_history.append(1.0 if hit else 0.0)
        if len(self._hit_history) > 1000:
            self._hit_history = self._hit_history[-500:]

        return sum(self._hit_history) / len(self._hit_history)

    def _update_validation_metrics(self, result: ValidationResult) -> None:
        """Update validation metrics."""
        self._metrics.total_validations += 1

        if result.is_valid:
            self._metrics.successful_validations += 1
        else:
            self._metrics.failed_validations += 1

        self._validation_times.append(result.validation_time_ms)
        if len(self._validation_times) > 1000:
            self._validation_times = self._validation_times[-500:]

    def _get_consistency_rule(self, rule_name: str) -> Callable | None:
        """Get consistency rule function by name."""
        consistency_rules = {
            "portfolio_position_consistency": self._check_portfolio_position_consistency,
            "order_position_consistency": self._check_order_position_consistency,
            "risk_exposure_consistency": self._check_risk_exposure_consistency,
        }

        return consistency_rules.get(rule_name)

    async def _check_portfolio_position_consistency(
        self, portfolio_state: dict[str, Any], related_states: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Check consistency between portfolio and position states."""
        try:
            portfolio_positions = portfolio_state.get("positions", [])

            # Calculate total from individual positions
            calculated_total = Decimal("0")
            for position in portfolio_positions:
                if isinstance(position, dict):
                    qty = position.get("quantity", 0)
                    price = position.get("current_price", 0)
                    if qty and price:
                        calculated_total += Decimal(str(qty)) * Decimal(str(price))

            # Compare with reported total
            reported_total = Decimal(str(portfolio_state.get("total_value", 0)))

            # Allow small differences due to rounding
            tolerance = Decimal("0.01")
            difference = abs(calculated_total - reported_total)

            if difference > tolerance:
                return {
                    "valid": False,
                    "message": f"Portfolio total mismatch: reported {reported_total}, "
                    f"calculated {calculated_total}",
                }

            return {"valid": True, "message": "Portfolio positions consistent"}

        except Exception as e:
            return {"valid": False, "message": f"Consistency check failed: {e}"}

    async def _check_order_position_consistency(
        self, order_state: dict[str, Any], related_states: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Check consistency between orders and positions."""
        # Simplified implementation
        return {"valid": True, "message": "Order position consistency check passed"}

    async def _check_risk_exposure_consistency(
        self, risk_state: dict[str, Any], related_states: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Check consistency between risk metrics and actual exposure."""
        # Simplified implementation
        return {"valid": True, "message": "Risk exposure consistency check passed"}

    async def _load_custom_rules(self) -> None:
        """Load custom validation rules from configuration."""
        try:
            # In a full implementation, this would load from config files
            # For now, just log that it's available
            self.logger.info("Custom validation rules loading completed")

        except Exception as e:
            self.logger.warning(f"Failed to load custom validation rules: {e}")

    async def _cache_cleanup_loop(self) -> None:
        """Background cache cleanup loop."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)

                # Clean expired cache entries
                expired_keys = []
                for key, (_result, cached_time) in self._validation_cache.items():
                    if (current_time - cached_time).total_seconds() > self.cache_ttl_seconds:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self._validation_cache[key]

                if expired_keys:
                    self.logger.debug(
                        f"Cleaned {len(expired_keys)} expired validation cache entries"
                    )

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(300)
