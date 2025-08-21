"""Validation Service for the T-Bot trading system.

This module provides a comprehensive validation service that eliminates validation
logic duplication across 20+ modules and provides a centralized, consistent
validation framework.

Key Features:
- Centralized validation logic for all data types
- Type-safe validation with comprehensive error messages
- Async/sync validation support
- Batch validation capabilities
- Custom validation rule registry
- Performance monitoring and caching
- Detailed validation reporting
- Context-aware validation (trading mode, exchange-specific)

Architecture:
- ValidationService: Main service interface for all validation operations
- ValidationRule: Abstract base for validation rules
- ValidationResult: Comprehensive validation result with context
- ValidationContext: Context information for validation (exchange, mode, etc.)
- ValidatorRegistry: Registry for custom validation rules
- ValidationCache: Caching layer for expensive validations

Usage Example:
    ```python
    # In service constructors - dependency injection
    def __init__(self, validation_service: ValidationService):
        self.validation_service = validation_service


    # Single validation
    result = await self.validation_service.validate_order(order_data)

    # Batch validation
    results = await self.validation_service.validate_batch(
        [("order", order_data), ("risk_params", risk_data)]
    )

    # Context-aware validation
    result = await self.validation_service.validate_order(
        order_data, context=ValidationContext(exchange="binance", mode="live")
    )
    ```

TODO: Update the following modules to use ValidationService:
- All strategy modules (remove duplicate validation logic)
- All risk management modules (centralize risk validation)
- All exchange modules (consolidate exchange validation)
- All execution modules (unified order validation)
- All data processing modules (centralize data validation)
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import Field

from src.core.exceptions import (
    ErrorCategory,
    ValidationError as TradingValidationError,
)
from src.core.types.base import (
    BaseValidatedModel,
    ValidationLevel,
)

from .core import ValidationFramework

# Epsilon for float comparisons to handle floating-point precision issues
EPSILON = 1e-10


class ValidationType(Enum):
    """Types of validation operations."""

    ORDER = "order"
    POSITION = "position"
    BALANCE = "balance"
    STRATEGY = "strategy"
    RISK = "risk"
    EXCHANGE = "exchange"
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    CONFIGURATION = "configuration"
    API_REQUEST = "api_request"
    WEBHOOK = "webhook"
    TRADE_DATA = "trade_data"


class ValidationContext(BaseValidatedModel):
    """Context information for validation operations."""

    exchange: str | None = Field(None, description="Exchange name for context-specific validation")
    trading_mode: str | None = Field(None, description="Trading mode (live, paper, backtest)")
    strategy_type: str | None = Field(
        None, description="Strategy type for strategy-specific validation"
    )
    user_id: str | None = Field(None, description="User ID for user-specific rules")
    session_id: str | None = Field(None, description="Session ID for tracking")
    request_id: str | None = Field(None, description="Request ID for tracing")
    additional_context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context data"
    )

    def get_context_hash(self) -> str:
        """Generate a hash for caching purposes."""
        context_str = f"{self.exchange}:{self.trading_mode}:{self.strategy_type}:{self.user_id}"
        return hashlib.md5(context_str.encode()).hexdigest()


class ValidationDetail(BaseValidatedModel):
    """Detailed validation information."""

    field: str = Field(..., description="Field that was validated")
    validation_type: str = Field(..., description="Type of validation performed")
    expected: Any | None = Field(None, description="Expected value or condition")
    actual: Any | None = Field(None, description="Actual value found")
    message: str = Field(..., description="Validation message")
    severity: ValidationLevel = Field(
        ValidationLevel.MEDIUM, description="Severity of the validation issue"
    )
    suggestion: str | None = Field(None, description="Suggested fix")


class ValidationResult(BaseValidatedModel):
    """Comprehensive validation result."""

    is_valid: bool = Field(..., description="Whether validation passed")
    validation_type: ValidationType = Field(..., description="Type of validation performed")
    value: Any = Field(None, description="The validated value")
    normalized_value: Any | None = Field(None, description="Normalized/processed value")
    errors: list[ValidationDetail] = Field(default_factory=list, description="Validation errors")
    warnings: list[ValidationDetail] = Field(
        default_factory=list, description="Validation warnings"
    )
    context: ValidationContext | None = Field(None, description="Validation context")
    execution_time_ms: float = Field(0.0, description="Validation execution time in milliseconds")
    cache_hit: bool = Field(False, description="Whether result was from cache")

    def add_error(
        self,
        field: str,
        message: str,
        validation_type: str = "general",
        expected: Any = None,
        actual: Any = None,
        severity: ValidationLevel = ValidationLevel.HIGH,
        suggestion: str | None = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(
            ValidationDetail(
                field=field,
                validation_type=validation_type,
                expected=expected,
                actual=actual,
                message=message,
                severity=severity,
                suggestion=suggestion,
            )
        )
        self.is_valid = False

    def add_warning(
        self,
        field: str,
        message: str,
        validation_type: str = "general",
        suggestion: str | None = None,
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(
            ValidationDetail(
                field=field,
                validation_type=validation_type,
                message=message,
                severity=ValidationLevel.LOW,
                suggestion=suggestion,
            )
        )

    def get_error_summary(self) -> str:
        """Get a summary of all errors."""
        if not self.errors:
            return "No errors"

        error_messages = [f"{error.field}: {error.message}" for error in self.errors]
        return "; ".join(error_messages)

    def get_critical_errors(self) -> list[ValidationDetail]:
        """Get only critical validation errors."""
        return [error for error in self.errors if error.severity == ValidationLevel.CRITICAL]

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return len(self.get_critical_errors()) > 0


class ValidationRule(ABC):
    """Abstract base class for validation rules."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def validate(
        self, value: Any, context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate a value and return detailed results."""
        pass

    def get_cache_key(self, value: Any, context: ValidationContext | None = None) -> str:
        """Generate cache key for this validation."""
        value_hash = hashlib.md5(str(value).encode()).hexdigest()
        context_hash = context.get_context_hash() if context else "no_context"
        return f"{self.name}:{value_hash}:{context_hash}"


class NumericValidationRule(ValidationRule):
    """Validation rule for numeric values."""

    def __init__(
        self,
        name: str,
        min_value: float | None = None,
        max_value: float | None = None,
        allow_zero: bool = True,
        decimal_places: int | None = None,
    ):
        super().__init__(name, f"Numeric validation with range {min_value} to {max_value}")
        self.min_value = min_value
        self.max_value = max_value
        self.allow_zero = allow_zero
        self.decimal_places = decimal_places

    async def validate(
        self, value: Any, context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate numeric value."""
        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.ORDER,  # Default, can be overridden
            value=value,
            context=context,
        )

        # Check if value is numeric
        try:
            if isinstance(value, str):
                numeric_value = Decimal(value)
            elif isinstance(value, int | float):
                numeric_value = Decimal(str(value))
            elif isinstance(value, Decimal):
                numeric_value = value
            else:
                result.add_error(
                    "value",
                    f"Value must be numeric, got {type(value).__name__}",
                    "type_check",
                    expected="numeric",
                    actual=str(type(value)),
                    suggestion="Provide a valid number",
                )
                return result
        except (InvalidOperation, ValueError) as e:
            result.add_error(
                "value",
                f"Invalid numeric value: {e!s}",
                "format_check",
                suggestion="Provide a valid decimal number",
            )
            return result

        # Store normalized value
        result.normalized_value = numeric_value

        # Check zero
        if not self.allow_zero and abs(float(numeric_value)) < EPSILON:
            result.add_error(
                "value",
                "Zero values are not allowed",
                "zero_check",
                expected="non-zero",
                actual="0",
            )

        # Check minimum value
        if self.min_value is not None and numeric_value < Decimal(str(self.min_value)):
            result.add_error(
                "value",
                f"Value {numeric_value} is below minimum {self.min_value}",
                "range_check",
                expected=f">= {self.min_value}",
                actual=str(numeric_value),
                suggestion=f"Use a value >= {self.min_value}",
            )

        # Check maximum value
        if self.max_value is not None and numeric_value > Decimal(str(self.max_value)):
            result.add_error(
                "value",
                f"Value {numeric_value} exceeds maximum {self.max_value}",
                "range_check",
                expected=f"<= {self.max_value}",
                actual=str(numeric_value),
                suggestion=f"Use a value <= {self.max_value}",
            )

        # Check decimal places
        if self.decimal_places is not None:
            decimal_parts = str(numeric_value).split(".")
            if len(decimal_parts) > 1 and len(decimal_parts[1]) > self.decimal_places:
                decimal_count = len(decimal_parts[1])
                result.add_warning(
                    "value",
                    f"Value has {decimal_count} decimal places, maximum is {self.decimal_places}",
                    "precision_check",
                    suggestion=f"Round to {self.decimal_places} decimal places",
                )

        return result


class StringValidationRule(ValidationRule):
    """Validation rule for string values."""

    def __init__(
        self,
        name: str,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        allowed_values: list[str] | None = None,
        case_sensitive: bool = True,
    ):
        super().__init__(name, f"String validation with length {min_length}-{max_length}")
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.allowed_values = allowed_values
        self.case_sensitive = case_sensitive

    async def validate(
        self, value: Any, context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate string value."""
        result = ValidationResult(
            is_valid=True, validation_type=ValidationType.ORDER, value=value, context=context
        )

        # Check if value is string
        if not isinstance(value, str):
            result.add_error(
                "value",
                f"Value must be a string, got {type(value).__name__}",
                "type_check",
                expected="string",
                actual=str(type(value)),
            )
            return result

        # Normalize value
        normalized_value = value if self.case_sensitive else value.upper()
        result.normalized_value = normalized_value

        # Check length
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(
                "value",
                f"String length {len(value)} is below minimum {self.min_length}",
                "length_check",
                expected=f">= {self.min_length} characters",
                actual=f"{len(value)} characters",
            )

        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(
                "value",
                f"String length {len(value)} exceeds maximum {self.max_length}",
                "length_check",
                expected=f"<= {self.max_length} characters",
                actual=f"{len(value)} characters",
            )

        # Check pattern
        if self.pattern:
            import re

            if not re.match(self.pattern, value):
                result.add_error(
                    "value",
                    f"Value '{value}' does not match required pattern",
                    "pattern_check",
                    expected=f"Pattern: {self.pattern}",
                    actual=value,
                )

        # Check allowed values
        if self.allowed_values:
            check_values = (
                [v.upper() for v in self.allowed_values]
                if not self.case_sensitive
                else self.allowed_values
            )
            check_value = normalized_value

            if check_value not in check_values:
                result.add_error(
                    "value",
                    f"Value '{value}' is not in allowed values",
                    "enum_check",
                    expected=f"One of: {self.allowed_values}",
                    actual=value,
                )

        return result


class ValidationCache:
    """Thread-safe validation cache with TTL support."""

    def __init__(self, default_ttl: int = 300, max_size: int = 10000):
        self._cache: dict[str, tuple[ValidationResult, float]] = {}
        self._access_count: dict[str, int] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self.logger = logging.getLogger(f"{__name__}.ValidationCache")

    async def get(self, key: str) -> ValidationResult | None:
        """Get cached validation result."""
        if key in self._cache:
            result, expiry = self._cache[key]
            if time.time() < expiry:
                self._access_count[key] = self._access_count.get(key, 0) + 1
                result.cache_hit = True
                return result
            else:
                # Expired
                del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]
        return None

    async def set(self, key: str, result: ValidationResult, ttl: int | None = None) -> None:
        """Set cached validation result."""
        ttl = ttl or self._default_ttl
        expiry = time.time() + ttl

        # Clean cache if too large
        if len(self._cache) >= self._max_size:
            await self._cleanup_cache()

        self._cache[key] = (result, expiry)
        self.logger.debug(f"Cached validation result for key '{key[:50]}...' with TTL {ttl}s")

    async def _cleanup_cache(self) -> None:
        """Clean up expired entries and least accessed entries."""
        current_time = time.time()

        # Remove expired entries
        expired_keys = [k for k, (_, expiry) in self._cache.items() if current_time >= expiry]
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_count:
                del self._access_count[key]

        # If still too large, remove least accessed entries
        if len(self._cache) >= self._max_size:
            # Sort by access count and remove least accessed
            sorted_by_access = sorted(self._access_count.items(), key=lambda x: x[1])
            to_remove = (
                len(self._cache) - self._max_size + 100
            )  # Remove extra to avoid frequent cleanup

            for key, _ in sorted_by_access[:to_remove]:
                if key in self._cache:
                    del self._cache[key]
                del self._access_count[key]

        self.logger.debug(f"Cache cleanup completed. Current size: {len(self._cache)}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self._cache),
            "total_accesses": sum(self._access_count.values()),
            "most_accessed": (
                max(self._access_count.items(), key=lambda x: x[1]) if self._access_count else None
            ),
            "cache_size_bytes": sum(len(str(result)) for result, _ in self._cache.values()),
        }


class ValidatorRegistry:
    """Registry for validation rules."""

    def __init__(self):
        self._rules: dict[str, ValidationRule] = {}
        self.logger = logging.getLogger(f"{__name__}.ValidatorRegistry")
        self._register_default_rules()

    def register(self, rule: ValidationRule) -> None:
        """Register a validation rule."""
        self._rules[rule.name] = rule
        self.logger.debug(f"Registered validation rule: {rule.name}")

    def get(self, name: str) -> ValidationRule | None:
        """Get a validation rule by name."""
        return self._rules.get(name)

    def list_rules(self) -> list[str]:
        """List all registered rule names."""
        return list(self._rules.keys())

    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        # Price validation
        self.register(
            NumericValidationRule(
                "price",
                min_value=0.00000001,
                max_value=1000000.0,
                allow_zero=False,
                decimal_places=8,
            )
        )

        # Quantity validation
        self.register(
            NumericValidationRule(
                "quantity", min_value=0.00000001, allow_zero=False, decimal_places=8
            )
        )

        # Symbol validation
        self.register(
            StringValidationRule(
                "symbol",
                min_length=3,
                max_length=20,
                pattern=r"^[A-Z]+(/|_|-)?[A-Z]+$",
                case_sensitive=False,
            )
        )

        # Order side validation
        self.register(
            StringValidationRule("order_side", allowed_values=["BUY", "SELL"], case_sensitive=False)
        )

        # Order type validation
        self.register(
            StringValidationRule(
                "order_type",
                allowed_values=["MARKET", "LIMIT", "STOP_LOSS", "TAKE_PROFIT", "STOP_LIMIT"],
                case_sensitive=False,
            )
        )

        # Risk percentage validation
        self.register(
            NumericValidationRule(
                "risk_percentage",
                min_value=0.001,
                max_value=0.1,  # 10% max
                allow_zero=False,
            )
        )


class ValidationService:
    """Comprehensive validation service for the T-Bot trading system.

    This service provides centralized validation functionality that eliminates
    duplicate validation logic across the codebase and ensures consistent
    validation behavior throughout the application.

    Features:
    - Centralized validation rules
    - Context-aware validation
    - Async/sync validation support
    - Batch validation capabilities
    - Performance monitoring and caching
    - Detailed validation reporting
    - Custom validation rule registry

    Example Usage:
        ```python
        # Initialize service (usually done in main application)
        validation_service = ValidationService()
        await validation_service.initialize()

        # Inject into services
        order_service = OrderService(validation_service=validation_service)

        # Single validation
        result = await validation_service.validate_order(order_data)
        if not result.is_valid:
            raise ValidationError(result.get_error_summary())

        # Batch validation
        results = await validation_service.validate_batch(
            [("order", order_data), ("risk", risk_params)]
        )
        ```
    """

    def __init__(
        self, cache_ttl: int = 300, enable_cache: bool = True, max_cache_size: int = 10000
    ):
        """Initialize ValidationService.

        Args:
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Enable validation result caching
            max_cache_size: Maximum cache size
        """
        self.cache = ValidationCache(cache_ttl, max_cache_size) if enable_cache else None
        self.registry = ValidatorRegistry()
        self.enable_cache = enable_cache

        # Use ValidationFramework as single source of truth
        self.framework = ValidationFramework()

        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.ValidationService")

    async def initialize(self) -> None:
        """Initialize the validation service."""
        if self._initialized:
            self.logger.warning("ValidationService already initialized")
            return

        self.logger.info("ValidationService initialized successfully")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the validation service."""
        self._initialized = False
        self.logger.info("ValidationService shutdown completed")

    def _ensure_initialized(self) -> None:
        """Ensure the service is initialized."""
        if not self._initialized:
            raise TradingValidationError(
                "ValidationService not initialized. Call initialize() first.",
                error_code="VALIDATION_SERVICE_NOT_INITIALIZED",
                category=ErrorCategory.VALIDATION,
            )

    async def _validate_with_rule(
        self, rule_name: str, value: Any, context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate using a specific rule."""
        start_time = time.time()

        # Check cache first
        if self.cache:
            cache_key = (
                f"{rule_name}:{value!s}:{context.get_context_hash() if context else 'no_context'}"
            )
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result

        # Get rule
        rule = self.registry.get(rule_name)
        if not rule:
            result = ValidationResult(
                is_valid=False, validation_type=ValidationType.ORDER, value=value, context=context
            )
            result.add_error(
                "rule",
                f"Validation rule '{rule_name}' not found",
                "rule_lookup",
                suggestion="Check available rules or register custom rule",
            )
            return result

        # Execute validation
        result = await rule.validate(value, context)
        result.execution_time_ms = (time.time() - start_time) * 1000

        # Cache result
        if self.cache and result.is_valid:  # Only cache successful validations
            await self.cache.set(cache_key, result)

        return result

    # Order Validation
    async def validate_order(
        self, order_data: dict[str, Any], context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate trading order data.

        Args:
            order_data: Order data to validate
            context: Validation context

        Returns:
            Comprehensive validation result
        """
        self._ensure_initialized()
        start_time = time.time()

        result = ValidationResult(
            is_valid=True, validation_type=ValidationType.ORDER, value=order_data, context=context
        )

        # Check required fields
        required_fields = ["symbol", "side", "type", "quantity"]
        for field in required_fields:
            if field not in order_data:
                result.add_error(
                    field,
                    f"Required field '{field}' is missing",
                    "required_field",
                    expected="present",
                    actual="missing",
                    severity=ValidationLevel.CRITICAL,
                )

        if not result.is_valid:
            return result  # Early return for missing required fields

        # Validate individual fields
        field_validations = [
            ("symbol", "symbol", order_data.get("symbol")),
            ("side", "order_side", order_data.get("side")),
            ("type", "order_type", order_data.get("type")),
            ("quantity", "quantity", order_data.get("quantity")),
        ]

        # Add price validation for limit orders
        order_type = order_data.get("type", "").upper()
        if order_type in ["LIMIT", "STOP_LIMIT"]:
            if "price" not in order_data:
                result.add_error(
                    "price",
                    f"Price is required for {order_type} orders",
                    "conditional_required",
                    severity=ValidationLevel.CRITICAL,
                )
            else:
                field_validations.append(("price", "price", order_data.get("price")))

        # Execute field validations
        for field_name, rule_name, value in field_validations:
            if value is not None:
                field_result = await self._validate_with_rule(rule_name, value, context)
                if not field_result.is_valid:
                    for error in field_result.errors:
                        result.add_error(
                            field_name,
                            error.message,
                            error.validation_type,
                            error.expected,
                            error.actual,
                            error.severity,
                            error.suggestion,
                        )
                else:
                    # Store normalized value
                    if field_result.normalized_value is not None:
                        if not hasattr(result, "normalized_data"):
                            result.normalized_data = {}
                        result.normalized_data[field_name] = field_result.normalized_value

        # Business logic validations
        if result.is_valid:
            await self._validate_order_business_logic(order_data, result, context)

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    async def _validate_order_business_logic(
        self,
        order_data: dict[str, Any],
        result: ValidationResult,
        context: ValidationContext | None = None,
    ) -> None:
        """Validate order business logic."""
        # Validate minimum notional value if context provides exchange info
        if context and context.exchange:
            # This would typically fetch exchange-specific limits
            # For now, we'll use generic minimums
            price = order_data.get("price", 0)
            quantity = order_data.get("quantity", 0)

            if price and quantity:
                notional = float(price) * float(quantity)
                min_notional = 10.0  # This would come from exchange info

                if notional < min_notional:
                    result.add_error(
                        "notional",
                        f"Order notional value {notional} below minimum {min_notional}",
                        "business_rule",
                        expected=f">= {min_notional}",
                        actual=str(notional),
                        suggestion="Increase quantity or price to meet minimum notional",
                    )

    # Risk Validation
    async def validate_risk_parameters(
        self, risk_data: dict[str, Any], context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate risk management parameters."""
        self._ensure_initialized()
        start_time = time.time()

        result = ValidationResult(
            is_valid=True, validation_type=ValidationType.RISK, value=risk_data, context=context
        )

        # Use legacy framework for complex validation
        try:
            self.framework.validate_risk_parameters(risk_data)
        except ValueError as e:
            error_msg = str(e)
            field_name = error_msg.split(":")[0] if ":" in error_msg else "unknown"
            result.add_error(field_name, error_msg, "business_rule", severity=ValidationLevel.HIGH)

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    # Strategy Validation
    async def validate_strategy_config(
        self, strategy_data: dict[str, Any], context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate strategy configuration."""
        self._ensure_initialized()
        start_time = time.time()

        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.STRATEGY,
            value=strategy_data,
            context=context,
        )

        # Use legacy framework
        try:
            self.framework.validate_strategy_params(strategy_data)
        except ValueError as e:
            error_msg = str(e)
            field_name = error_msg.split(" ")[0] if " " in error_msg else "unknown"
            result.add_error(field_name, error_msg, "strategy_rule", severity=ValidationLevel.HIGH)

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    # Market Data Validation
    async def validate_market_data(
        self, market_data: dict[str, Any], context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate market data."""
        self._ensure_initialized()
        start_time = time.time()

        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.MARKET_DATA,
            value=market_data,
            context=context,
        )

        # Check required fields for OHLCV data
        required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        for field in required_fields:
            if field not in market_data:
                result.add_error(
                    field,
                    f"Required field '{field}' is missing from market data",
                    "required_field",
                    severity=ValidationLevel.CRITICAL,
                )

        if result.is_valid:
            # Validate OHLC relationships
            try:
                open_price = float(market_data["open"])
                high_price = float(market_data["high"])
                low_price = float(market_data["low"])
                close_price = float(market_data["close"])

                if high_price < max(open_price, close_price):
                    result.add_error(
                        "high",
                        "High price cannot be less than open or close price",
                        "ohlc_relationship",
                        severity=ValidationLevel.HIGH,
                    )

                if low_price > min(open_price, close_price):
                    result.add_error(
                        "low",
                        "Low price cannot be greater than open or close price",
                        "ohlc_relationship",
                        severity=ValidationLevel.HIGH,
                    )

            except (ValueError, KeyError) as e:
                result.add_error(
                    "prices",
                    f"Invalid price data: {e!s}",
                    "data_format",
                    severity=ValidationLevel.HIGH,
                )

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    # Batch Validation
    async def validate_batch(
        self, validations: list[tuple[str, Any]], context: ValidationContext | None = None
    ) -> dict[str, ValidationResult]:
        """Validate multiple items in batch.

        Args:
            validations: List of (validation_type, data) tuples
            context: Validation context

        Returns:
            Dictionary of validation results by validation name
        """
        self._ensure_initialized()
        start_time = time.time()

        results = {}
        validation_methods = {
            "order": self.validate_order,
            "risk": self.validate_risk_parameters,
            "strategy": self.validate_strategy_config,
            "market_data": self.validate_market_data,
        }

        # Execute validations concurrently
        tasks = []
        for validation_name, data in validations:
            validation_type = validation_name.lower()
            if validation_type in validation_methods:
                task = validation_methods[validation_type](data, context)
                tasks.append((validation_name, task))
            else:
                # Create error result for unknown validation type
                error_result = ValidationResult(
                    is_valid=False,
                    validation_type=ValidationType.ORDER,  # Default
                    value=data,
                    context=context,
                )
                error_result.add_error(
                    "validation_type",
                    f"Unknown validation type: {validation_type}",
                    "type_check",
                    severity=ValidationLevel.CRITICAL,
                )
                results[validation_name] = error_result

        # Wait for all validations to complete
        if tasks:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )

            for (validation_name, _), task_result in zip(tasks, completed_tasks, strict=False):
                if isinstance(task_result, Exception):
                    error_result = ValidationResult(
                        is_valid=False,
                        validation_type=ValidationType.ORDER,
                        value=None,
                        context=context,
                    )
                    error_result.add_error(
                        "execution",
                        f"Validation failed with exception: {task_result!s}",
                        "execution_error",
                        severity=ValidationLevel.CRITICAL,
                    )
                    results[validation_name] = error_result
                else:
                    results[validation_name] = task_result

        total_time = (time.time() - start_time) * 1000
        self.logger.debug(
            f"Batch validation of {len(validations)} items completed in {total_time:.2f}ms"
        )

        return results

    # Backward Compatibility Methods
    def validate_price(self, price: Any) -> bool:
        """Backward compatibility for price validation."""
        try:
            return self.framework.validate_price(price)
        except ValueError:
            return False

    def validate_quantity(self, quantity: Any) -> bool:
        """Backward compatibility for quantity validation."""
        try:
            return self.framework.validate_quantity(quantity)
        except ValueError:
            return False

    def validate_symbol(self, symbol: str) -> bool:
        """Backward compatibility for symbol validation."""
        try:
            return self.framework.validate_symbol(symbol)
        except ValueError:
            return False

    # Service Management
    def register_custom_rule(self, rule: ValidationRule) -> None:
        """Register a custom validation rule."""
        self.registry.register(rule)

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation service statistics."""
        stats = {
            "initialized": self._initialized,
            "registered_rules": len(self.registry.list_rules()),
            "cache_enabled": self.enable_cache,
        }

        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()

        return stats

    # Context manager support
    async def __aenter__(self) -> "ValidationService":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()


# Singleton instance for backward compatibility
_validation_service_instance: ValidationService | None = None


async def get_validation_service(reload: bool = False) -> ValidationService:
    """Get or create the global ValidationService instance.

    Args:
        reload: Force reload of validation service

    Returns:
        Global ValidationService instance
    """
    global _validation_service_instance

    if _validation_service_instance is None or reload:
        _validation_service_instance = ValidationService()
        await _validation_service_instance.initialize()

    return _validation_service_instance


async def shutdown_validation_service() -> None:
    """Shutdown the global validation service."""
    global _validation_service_instance

    if _validation_service_instance:
        await _validation_service_instance.shutdown()
        _validation_service_instance = None
