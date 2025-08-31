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

Modules integrated with ValidationService:
- Core validation framework provides centralized validation logic
- Strategy modules utilize centralized validation through dependency injection
- Risk management modules use unified validation patterns
- Exchange modules leverage consistent validation interfaces
- Execution modules implement standardized order validation
- Data processing modules use centralized data validation patterns
"""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.core import BaseService, HealthStatus
from src.core.exceptions import (
    ErrorCategory,
    ServiceError,
    ValidationError as TradingValidationError,
)
from src.core.logging import get_logger
from src.core.types import ValidationLevel
from pydantic import ConfigDict
from src.utils.validation.core import ValidationFramework

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


class ValidationContext(BaseModel):
    """Context information for validation operations."""

    exchange: str | None = Field(None, description="Exchange name for context-specific validation")
    trading_mode: str | None = Field(None, description="Trading mode (live, paper, backtest)")
    strategy_type: str | None = Field(None, description="Strategy type for strategy-specific validation")
    user_id: str | None = Field(None, description="User ID for user-specific rules")
    session_id: str | None = Field(None, description="Session ID for tracking")
    request_id: str | None = Field(None, description="Request ID for tracing")
    additional_context: dict[str, Any] = Field(default_factory=dict, description="Additional context data")

    def get_context_hash(self) -> str:
        """Generate a hash for caching purposes."""
        context_str = f"{self.exchange}:{self.trading_mode}:{self.strategy_type}:{self.user_id}"
        return hashlib.md5(context_str.encode()).hexdigest()


class ValidationDetail(BaseModel):
    """Detailed validation information."""

    field: str = Field(..., description="Field that was validated")
    validation_type: str = Field(..., description="Type of validation performed")
    expected: Any | None = Field(None, description="Expected value or condition")
    actual: Any | None = Field(None, description="Actual value found")
    message: str = Field(..., description="Validation message")
    severity: ValidationLevel = Field(ValidationLevel.MEDIUM, description="Severity of the validation issue")
    suggestion: str | None = Field(None, description="Suggested fix")


class ValidationResult(BaseModel):
    """Comprehensive validation result."""

    is_valid: bool = Field(..., description="Whether validation passed")
    validation_type: ValidationType = Field(..., description="Type of validation performed")
    value: Any = Field(None, description="The validated value")
    normalized_value: Any | None = Field(None, description="Normalized/processed value")
    errors: list[ValidationDetail] = Field(default_factory=list, description="Validation errors")
    warnings: list[ValidationDetail] = Field(default_factory=list, description="Validation warnings")
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
                expected=None,
                actual=None,
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
    async def validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult:
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

    async def validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult:
        """Validate numeric value."""
        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.ORDER,  # Default, can be overridden
            value=value,
            normalized_value=None,
            context=context,
            execution_time_ms=0.0,
            cache_hit=False,
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
        if not self.allow_zero and abs(numeric_value) < Decimal(str(EPSILON)):
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

    async def validate(self, value: Any, context: ValidationContext | None = None) -> ValidationResult:
        """Validate string value."""
        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.ORDER,
            value=value,
            normalized_value=None,
            context=context,
            execution_time_ms=0.0,
            cache_hit=False,
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
            check_values = [v.upper() for v in self.allowed_values] if not self.case_sensitive else self.allowed_values
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
        self.logger = get_logger(f"{__name__}.ValidationCache")

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

        try:
            # Remove expired entries
            expired_keys = [k for k, (_, expiry) in self._cache.items() if current_time >= expiry]
            for key in expired_keys:
                try:
                    self._cache.pop(key, None)  # Use pop to handle concurrent modifications
                    self._access_count.pop(key, None)
                except KeyError as e:
                    # Key might have been removed by another thread
                    self.logger.debug(f"Key removed during cache cleanup: {e}")
                    pass

            # If still too large, remove least accessed entries
            if len(self._cache) >= self._max_size:
                # Sort by access count and remove least accessed
                sorted_by_access = sorted(self._access_count.items(), key=lambda x: x[1])
                to_remove = len(self._cache) - self._max_size + 100  # Remove extra to avoid frequent cleanup

                for key, _ in sorted_by_access[:to_remove]:
                    try:
                        self._cache.pop(key, None)
                        self._access_count.pop(key, None)
                    except KeyError as e:
                        # Key might have been removed by another thread
                        self.logger.debug(f"Key removed during cache cleanup: {e}")
                        pass

            self.logger.debug(f"Cache cleanup completed. Current size: {len(self._cache)}")
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
            # Don't raise - cache cleanup failures shouldn't break validation

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self._cache),
            "total_accesses": sum(self._access_count.values()),
            "most_accessed": (max(self._access_count.items(), key=lambda x: x[1]) if self._access_count else None),
            "cache_size_bytes": sum(len(str(result)) for result, _ in self._cache.values()),
        }


class ValidatorRegistry:
    """Registry for validation rules."""

    def __init__(self) -> None:
        self._rules: dict[str, ValidationRule] = {}
        self.logger = get_logger(f"{__name__}.ValidatorRegistry")
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
        self.register(NumericValidationRule("quantity", min_value=0.00000001, allow_zero=False, decimal_places=8))

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
        self.register(StringValidationRule("order_side", allowed_values=["BUY", "SELL"], case_sensitive=False))

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


class ValidationService(BaseService):
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
        # Initialize service using dependency injection (usually done in main application)
        from src.core.dependency_injection import injector
        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        # Inject into services using DI container
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
        self,
        name: str | None = None,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
        cache_ttl: int = 300,
        enable_cache: bool = True,
        max_cache_size: int = 10000,
        validation_framework: ValidationFramework | None = None,
    ):
        """Initialize ValidationService with dependency injection.

        Args:
            name: Service name for identification
            config: Service configuration
            correlation_id: Request correlation ID
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Enable validation result caching
            max_cache_size: Maximum cache size
            validation_framework: Injected ValidationFramework instance
        """
        # Call BaseService constructor with proper parameters
        super().__init__(
            name=name or "ValidationService",
            config=config,
            correlation_id=correlation_id,
        )

        self.cache = ValidationCache(cache_ttl, max_cache_size) if enable_cache else None
        self.registry = ValidatorRegistry()
        self.enable_cache = enable_cache

        # Use dependency injection for ValidationFramework - no fallback to avoid circular dependencies
        if validation_framework is None:
            raise ServiceError(
                "ValidationFramework must be provided via dependency injection",
                error_code="SERV_000"
            )
        self.framework = validation_framework

    async def _do_start(self) -> None:
        """Override BaseService start method."""
        await super()._do_start()

        # Perform custom initialization
        self.logger.info("ValidationService started successfully")

    async def _do_stop(self) -> None:
        """Override BaseService stop method."""
        # Perform custom shutdown
        self.logger.info("ValidationService shutdown completed")

        await super()._do_stop()

    async def _service_health_check(self) -> HealthStatus:
        """Override BaseService health check method."""
        if not self.is_running:
            return HealthStatus.UNHEALTHY

        # Check cache health if enabled
        if self.cache:
            # Check cache is responsive
            try:
                test_key = "_health_check_test"
                test_result = ValidationResult(
                    is_valid=True,
                    validation_type=ValidationType.CONFIGURATION,
                    value="health_check",
                    normalized_value=None,
                    context=None,
                    execution_time_ms=0.0,
                    cache_hit=False,
                )
                await self.cache.set(test_key, test_result, ttl=1)
                await self.cache.get(test_key)
            except Exception as e:
                self.logger.error(f"Cache health check failed: {e}")
                return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    # Keep legacy methods for backward compatibility
    async def initialize(self) -> None:
        """Initialize the validation service (legacy method)."""
        await self.start()

    async def shutdown(self) -> None:
        """Shutdown the validation service (legacy method)."""
        await self.stop()

    def _ensure_initialized(self) -> None:
        """Ensure the service is initialized."""
        if not self.is_running:
            raise TradingValidationError(
                "ValidationService not initialized. Call initialize() first.",
                error_code="SERV_000",
                category=ErrorCategory.VALIDATION,
            )

    async def _validate_with_rule(
        self, rule_name: str, value: Any, context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate using a specific rule."""
        start_time = time.time()

        # Check cache first with timeout protection
        if self.cache:
            cache_key = f"{rule_name}:{value!s}:{context.get_context_hash() if context else 'no_context'}"
            try:
                # Add timeout to cache operations to prevent hanging
                cached_result = await asyncio.wait_for(self.cache.get(cache_key), timeout=1.0)
                if cached_result:
                    return cached_result
            except asyncio.TimeoutError:
                self.logger.warning(f"Cache get timed out for key: {cache_key[:50]}...")
            except Exception as e:
                self.logger.error(f"Cache get failed for key {cache_key[:50]}...: {e}")
                # Continue without cache on error

        # Get rule
        rule = self.registry.get(rule_name)
        if not rule:
            result = ValidationResult(
                is_valid=False,
                validation_type=ValidationType.ORDER,
                value=value,
                normalized_value=None,
                context=context,
                execution_time_ms=0.0,
                cache_hit=False,
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

        # Cache result with timeout protection
        if self.cache and result.is_valid:  # Only cache successful validations
            try:
                await asyncio.wait_for(self.cache.set(cache_key, result), timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.warning(f"Cache set timed out for key: {cache_key[:50]}...")
            except Exception as e:
                self.logger.error(f"Cache set failed for key {cache_key[:50]}...: {e}")
                # Continue without caching on error

        return result

    def _validate_required_order_fields(self, order_data: dict[str, Any], result: ValidationResult) -> None:
        """Validate required order fields."""
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

    def _setup_field_validations(self, order_data: dict[str, Any]) -> list[tuple[str, str, Any]]:
        """Setup field validations based on order type."""
        field_validations = [
            ("symbol", "symbol", order_data.get("symbol")),
            ("side", "order_side", order_data.get("side")),
            ("type", "order_type", order_data.get("type")),
            ("quantity", "quantity", order_data.get("quantity")),
        ]

        # Add price validation for limit orders
        order_type = order_data.get("type", "").upper()
        if order_type in ["LIMIT", "STOP_LIMIT"]:
            field_validations.append(("price", "price", order_data.get("price")))

        return field_validations

    def _validate_price_requirement(self, order_data: dict[str, Any], result: ValidationResult) -> None:
        """Validate price requirement for limit orders."""
        order_type = order_data.get("type", "").upper()
        if order_type in ["LIMIT", "STOP_LIMIT"] and "price" not in order_data:
            result.add_error(
                "price",
                f"Price is required for {order_type} orders",
                "conditional_required",
                severity=ValidationLevel.CRITICAL,
            )

    async def _execute_field_validations(
        self,
        field_validations: list[tuple[str, str, Any]],
        result: ValidationResult,
        context: ValidationContext | None = None,
    ) -> None:
        """Execute individual field validations."""
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
                    self._store_normalized_value(field_result, result, field_name)

    def _store_normalized_value(
        self, field_result: ValidationResult, result: ValidationResult, field_name: str
    ) -> None:
        """Store normalized value from field validation."""
        if field_result.normalized_value is not None:
            if result.normalized_value is None:
                result.normalized_value = {}
            if not isinstance(result.normalized_value, dict):
                result.normalized_value = {}
            result.normalized_value[field_name] = field_result.normalized_value

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
            is_valid=True,
            validation_type=ValidationType.ORDER,
            value=order_data,
            normalized_value=None,
            context=context,
            execution_time_ms=0.0,
            cache_hit=False,
        )

        # Check required fields
        self._validate_required_order_fields(order_data, result)
        if not result.is_valid:
            return result  # Early return for missing required fields

        # Validate price requirement for limit orders
        self._validate_price_requirement(order_data, result)

        # Setup and execute field validations
        field_validations = self._setup_field_validations(order_data)
        await self._execute_field_validations(field_validations, result, context)

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
                price_decimal = Decimal(str(price))
                quantity_decimal = Decimal(str(quantity))
                notional = price_decimal * quantity_decimal
                min_notional = Decimal("10.0")  # This would come from exchange info

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
            is_valid=True,
            validation_type=ValidationType.RISK,
            value=risk_data,
            normalized_value=None,
            context=context,
            execution_time_ms=0.0,
            cache_hit=False,
        )

        # Direct business logic validation - no external service calls needed
        await self._validate_risk_business_logic(risk_data, result, context)

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    async def _validate_risk_business_logic(
        self, risk_data: dict[str, Any], result: ValidationResult, context: ValidationContext | None = None
    ) -> None:
        """Validate risk parameters using service layer business logic."""
        # Risk per trade validation
        if "risk_per_trade" in risk_data:
            risk_per_trade = risk_data["risk_per_trade"]
            if risk_per_trade > 0.1:  # 10% max
                result.add_error(
                    "risk_per_trade",
                    "Risk per trade must be at most 0.1 (10%)",
                    "business_rule",
                    expected="<= 0.1",
                    actual=str(risk_per_trade),
                    severity=ValidationLevel.HIGH,
                )
            elif risk_per_trade <= 0:
                result.add_error(
                    "risk_per_trade",
                    "Risk per trade must be positive",
                    "business_rule",
                    expected="> 0",
                    actual=str(risk_per_trade),
                    severity=ValidationLevel.HIGH,
                )

        # Stop loss validation
        if "stop_loss" in risk_data:
            stop_loss = risk_data["stop_loss"]
            if stop_loss <= 0 or stop_loss >= 1:
                result.add_error(
                    "stop_loss",
                    "Stop loss must be between 0 and 1",
                    "business_rule",
                    expected="0 < stop_loss < 1",
                    actual=str(stop_loss),
                    severity=ValidationLevel.HIGH,
                )

        # Take profit validation
        if "take_profit" in risk_data:
            take_profit = risk_data["take_profit"]
            if take_profit <= 0:
                result.add_error(
                    "take_profit",
                    "Take profit must be positive",
                    "business_rule",
                    expected="> 0",
                    actual=str(take_profit),
                    severity=ValidationLevel.HIGH,
                )

        # Max position size validation
        if "max_position_size" in risk_data:
            max_position_size = risk_data["max_position_size"]
            if max_position_size <= 0:
                result.add_error(
                    "max_position_size",
                    "Max position size must be positive",
                    "business_rule",
                    expected="> 0",
                    actual=str(max_position_size),
                    severity=ValidationLevel.HIGH,
                )

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
            normalized_value=None,
            context=context,
            execution_time_ms=0.0,
            cache_hit=False,
        )

        # Direct business logic validation - no external service calls needed
        await self._validate_strategy_business_logic(strategy_data, result, context)

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    async def _validate_strategy_business_logic(
        self, strategy_data: dict[str, Any], result: ValidationResult, context: ValidationContext | None = None
    ) -> None:
        """Validate strategy configuration using service layer business logic."""
        # Require strategy type
        if "strategy_type" not in strategy_data:
            result.add_error(
                "strategy_type",
                "strategy_type is required",
                "required_field",
                expected="present",
                actual="missing",
                severity=ValidationLevel.CRITICAL,
            )
            return

        strategy_type = strategy_data["strategy_type"]

        # Validate common parameters
        await self._validate_common_strategy_params(strategy_data, result)

        # Strategy-specific validations
        if strategy_type == "MEAN_REVERSION":
            await self._validate_mean_reversion_params(strategy_data, result)
        elif strategy_type == "MOMENTUM":
            await self._validate_momentum_params(strategy_data, result)
        elif strategy_type == "market_making":
            await self._validate_market_making_params(strategy_data, result)

    async def _validate_common_strategy_params(self, strategy_data: dict[str, Any], result: ValidationResult) -> None:
        """Validate common strategy parameters."""
        if "timeframe" in strategy_data:
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if strategy_data["timeframe"] not in valid_timeframes:
                result.add_error(
                    "timeframe",
                    f"Invalid timeframe. Must be one of {valid_timeframes}",
                    "enum_check",
                    expected=f"One of: {valid_timeframes}",
                    actual=strategy_data["timeframe"],
                    severity=ValidationLevel.HIGH,
                )

    async def _validate_mean_reversion_params(self, strategy_data: dict[str, Any], result: ValidationResult) -> None:
        """Validate mean reversion strategy parameters."""
        required = ["window_size", "num_std", "entry_threshold"]
        for field in required:
            if field not in strategy_data:
                result.add_error(
                    field,
                    f"{field} is required for MEAN_REVERSION strategy",
                    "required_field",
                    expected="present",
                    actual="missing",
                    severity=ValidationLevel.CRITICAL,
                )

        if "window_size" in strategy_data and strategy_data["window_size"] < 2:
            result.add_error(
                "window_size",
                "window_size must be at least 2",
                "range_check",
                expected=">= 2",
                actual=str(strategy_data["window_size"]),
                severity=ValidationLevel.HIGH,
            )

        if "num_std" in strategy_data and strategy_data["num_std"] <= 0:
            result.add_error(
                "num_std",
                "num_std must be positive",
                "range_check",
                expected="> 0",
                actual=str(strategy_data["num_std"]),
                severity=ValidationLevel.HIGH,
            )

    async def _validate_momentum_params(self, strategy_data: dict[str, Any], result: ValidationResult) -> None:
        """Validate momentum strategy parameters."""
        required = ["lookback_period", "momentum_threshold"]
        for field in required:
            if field not in strategy_data:
                result.add_error(
                    field,
                    f"{field} is required for MOMENTUM strategy",
                    "required_field",
                    expected="present",
                    actual="missing",
                    severity=ValidationLevel.CRITICAL,
                )

        if "lookback_period" in strategy_data and strategy_data["lookback_period"] < 1:
            result.add_error(
                "lookback_period",
                "lookback_period must be at least 1",
                "range_check",
                expected=">= 1",
                actual=str(strategy_data["lookback_period"]),
                severity=ValidationLevel.HIGH,
            )

    async def _validate_market_making_params(self, strategy_data: dict[str, Any], result: ValidationResult) -> None:
        """Validate market making strategy parameters."""
        if "bid_spread" in strategy_data and strategy_data["bid_spread"] < 0:
            result.add_error(
                "bid_spread",
                "bid_spread must be non-negative",
                "range_check",
                expected=">= 0",
                actual=str(strategy_data["bid_spread"]),
                severity=ValidationLevel.HIGH,
            )

        if "ask_spread" in strategy_data and strategy_data["ask_spread"] < 0:
            result.add_error(
                "ask_spread",
                "ask_spread must be non-negative",
                "range_check",
                expected=">= 0",
                actual=str(strategy_data["ask_spread"]),
                severity=ValidationLevel.HIGH,
            )

        if "order_size" in strategy_data and strategy_data["order_size"] <= 0:
            result.add_error(
                "order_size",
                "order_size must be positive",
                "range_check",
                expected="> 0",
                actual=str(strategy_data["order_size"]),
                severity=ValidationLevel.HIGH,
            )

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
            normalized_value=None,
            context=context,
            execution_time_ms=0.0,
            cache_hit=False,
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
                open_price = Decimal(str(market_data["open"]))
                high_price = Decimal(str(market_data["high"]))
                low_price = Decimal(str(market_data["low"]))
                close_price = Decimal(str(market_data["close"]))

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

        # Use event-driven batch processing for consistency with core patterns
        # Execute validations as a stream of events rather than concurrent tasks
        validation_events = []
        for validation_name, data in validations:
            validation_type = validation_name.lower()
            if validation_type in validation_methods:
                # Create validation event for stream processing
                validation_events.append((validation_name, validation_type, data))
            else:
                # Create error result for unknown validation type
                error_result = ValidationResult(
                    is_valid=False,
                    validation_type=ValidationType.ORDER,  # Default
                    value=data,
                    normalized_value=None,
                    context=context,
                    execution_time_ms=0.0,
                    cache_hit=False,
                )
                error_result.add_error(
                    "validation_type",
                    f"Unknown validation type: {validation_type}",
                    "type_check",
                    severity=ValidationLevel.CRITICAL,
                )
                results[validation_name] = error_result

        # Process validation events with concurrency control
        if validation_events:
            # Limit concurrent validations to prevent resource exhaustion
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent validations

            async def run_validation(validation_name, validation_type, data):
                async with semaphore:
                    try:
                        return await asyncio.wait_for(
                            validation_methods[validation_type](data, context),
                            timeout=30.0,  # Individual validation timeout
                        )
                    except asyncio.TimeoutError:
                        error_result = ValidationResult(
                            is_valid=False,
                            validation_type=ValidationType.ORDER,
                            value=data,
                            normalized_value=None,
                            context=context,
                            execution_time_ms=0.0,
                            cache_hit=False,
                        )
                        error_result.add_error(
                            "timeout",
                            "Validation timed out after 30 seconds",
                            "timeout_error",
                            severity=ValidationLevel.CRITICAL,
                        )
                        return error_result
                    except Exception as e:
                        error_result = ValidationResult(
                            is_valid=False,
                            validation_type=ValidationType.ORDER,
                            value=data,
                            normalized_value=None,
                            context=context,
                            execution_time_ms=0.0,
                            cache_hit=False,
                        )
                        error_result.add_error(
                            "execution",
                            f"Validation failed with exception: {e!s}",
                            "execution_error",
                            severity=ValidationLevel.CRITICAL,
                        )
                        return error_result

            # Execute all validations concurrently
            tasks = [
                run_validation(validation_name, validation_type, data)
                for validation_name, validation_type, data in validation_events
            ]

            try:
                # Wait for all validations to complete with overall timeout
                completed_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=60.0  # Overall batch timeout
                )

                # Process results
                for (validation_name, _, _), result in zip(validation_events, completed_results, strict=False):
                    if isinstance(result, Exception):
                        error_result = ValidationResult(
                            is_valid=False,
                            validation_type=ValidationType.ORDER,
                            value=None,
                            normalized_value=None,
                            context=context,
                            execution_time_ms=0.0,
                            cache_hit=False,
                        )
                        error_result.add_error(
                            "batch_execution",
                            f"Batch validation failed: {result!s}",
                            "batch_execution_error",
                            severity=ValidationLevel.CRITICAL,
                        )
                        results[validation_name] = error_result
                    elif isinstance(result, ValidationResult):
                        results[validation_name] = result

            except asyncio.TimeoutError:
                self.logger.error("Batch validation timed out after 60 seconds")
                for validation_name, _, _ in validation_events:
                    if validation_name not in results:
                        error_result = ValidationResult(
                            is_valid=False,
                            validation_type=ValidationType.ORDER,
                            value=None,
                            normalized_value=None,
                            context=context,
                            execution_time_ms=0.0,
                            cache_hit=False,
                        )
                        error_result.add_error(
                            "batch_timeout",
                            "Batch validation timed out",
                            "timeout_error",
                            severity=ValidationLevel.CRITICAL,
                        )
                        results[validation_name] = error_result

        total_time = (time.time() - start_time) * 1000
        self.logger.debug(f"Batch validation of {len(validations)} items completed in {total_time:.2f}ms")

        return results

    # Backward Compatibility Methods
    def validate_price(self, price: Any) -> bool:
        """Backward compatibility for price validation."""
        try:
            # Use validation rules from registry
            rule = self.registry.get("price")
            if rule:
                try:
                    # Check if we're already in an async context
                    loop = asyncio.get_running_loop()
                    # We're in an async context - should use async version
                    self.logger.warning("validate_price called from async context - use async version")
                    return False
                except RuntimeError:
                    # No running loop, safe to use run_until_complete
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(rule.validate(price))
                    finally:
                        loop.close()
                return result.is_valid
            else:
                # Fallback validation
                from decimal import InvalidOperation

                from src.utils.decimal_utils import ZERO, to_decimal

                try:
                    price_decimal = to_decimal(price)
                    if not price_decimal.is_finite():
                        return False
                    return price_decimal > ZERO and price_decimal <= Decimal("1000000")
                except (TypeError, ValueError, InvalidOperation):
                    return False
        except Exception as e:
            self.logger.error(f"Price validation failed: {e}")
            return False

    def validate_quantity(self, quantity: Any) -> bool:
        """Backward compatibility for quantity validation."""
        try:
            # Use validation rules from registry
            rule = self.registry.get("quantity")
            if rule:
                try:
                    # Check if we're already in an async context
                    loop = asyncio.get_running_loop()
                    # We're in an async context - should use async version
                    self.logger.warning("validate_quantity called from async context - use async version")
                    return False
                except RuntimeError:
                    # No running loop, safe to use run_until_complete
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(rule.validate(quantity))
                    finally:
                        loop.close()
                return result.is_valid
            else:
                # Fallback validation
                from decimal import InvalidOperation

                from src.utils.decimal_utils import ZERO, to_decimal

                try:
                    qty_decimal = to_decimal(quantity)
                    if not qty_decimal.is_finite():
                        return False
                    return qty_decimal > ZERO
                except (TypeError, ValueError, InvalidOperation):
                    return False
        except Exception as e:
            self.logger.error(f"Quantity validation failed: {e}")
            return False

    def validate_symbol(self, symbol: str) -> bool:
        """Backward compatibility for symbol validation."""
        try:
            # Use validation rules from registry
            rule = self.registry.get("symbol")
            if rule:
                try:
                    # Check if we're already in an async context
                    loop = asyncio.get_running_loop()
                    # We're in an async context - should use async version
                    self.logger.warning("validate_symbol called from async context - use async version")
                    return False
                except RuntimeError:
                    # No running loop, safe to use run_until_complete
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(rule.validate(symbol))
                    finally:
                        loop.close()
                return result.is_valid
            else:
                # Fallback validation
                import re

                if not symbol or not isinstance(symbol, str):
                    return False
                symbol_norm = symbol.upper().strip()
                return bool(re.match(r"^[A-Z]+(/|_|-)?[A-Z]+$", symbol_norm))
        except Exception as e:
            self.logger.error(f"Symbol validation failed: {e}")
            return False

    # Service Management
    def register_custom_rule(self, rule: ValidationRule) -> None:
        """Register a custom validation rule."""
        self.registry.register(rule)

    # Simple validation utility methods
    def validate_decimal(self, value: Any) -> Any:
        """
        Validate and convert value to Decimal.

        Args:
            value: Value to validate

        Returns:
            Decimal value

        Raises:
            ValidationError: If value cannot be converted to Decimal
        """
        from decimal import Decimal, InvalidOperation

        from src.core.exceptions import ValidationError

        if isinstance(value, Decimal):
            return value

        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid decimal value: {value}") from e

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation service statistics."""
        stats: dict[str, Any] = {
            "initialized": self.is_running,
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

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()


# ValidationService registration is handled by service_registry.py


async def get_validation_service(reload: bool = False) -> ValidationService:
    """Get or create the global ValidationService instance from DI container.

    This factory function uses proper service locator pattern with dependency injection.

    Args:
        reload: Force reload of validation service

    Returns:
        Global ValidationService instance
    """
    from src.core.dependency_injection import injector

    if reload:
        # Clear and re-register the service
        injector.get_container().clear()
        if not TYPE_CHECKING:
            from src.utils.service_registry import register_util_services

            register_util_services()

    try:
        # Use interface-based resolution for better decoupling
        service = injector.resolve("ValidationServiceInterface")
    except Exception as e:
        # Fallback with proper error handling
        import logging

        logger = get_logger(__name__)
        logger.debug(f"Failed to resolve ValidationServiceInterface, registering util services: {e}")
        if not TYPE_CHECKING:
            from src.utils.service_registry import register_util_services

            register_util_services()
        service = injector.resolve("ValidationServiceInterface")

    # Initialize if not already running
    if not service.is_running:
        await service.initialize()

    return service


async def shutdown_validation_service() -> None:
    """Shutdown the global validation service using service locator pattern."""
    import logging

    from src.core.dependency_injection import injector

    logger = logging.getLogger(__name__)
    try:
        # Use interface-based resolution
        service = injector.resolve("ValidationServiceInterface")
        if service.is_running:
            await service.shutdown()
    except Exception as e:
        # Service not found or already shut down
        logger.debug(f"ValidationService shutdown failed or service not found: {e}")
