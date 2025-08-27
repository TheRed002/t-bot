"""
Strategy Validation Framework - Comprehensive validation for strategies and signals.

This module provides robust validation for:
- Strategy configurations
- Signal generation and quality
- Risk parameters
- Market conditions
- Performance thresholds
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.core.types import (
    MarketData,
    Signal,
    SignalDirection,
    StrategyConfig,
    StrategyType,
)
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class ValidationResult(BaseModel):
    """Result of a validation operation."""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(default_factory=list, description="Validation error messages")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional validation metadata"
    )

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)
        if not other.is_valid:
            self.is_valid = False


class BaseValidator(ABC):
    """Base class for all validators."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """
        Initialize validator.

        Args:
            name: Validator name
            config: Validator configuration
        """
        self.name = name
        self.config = config or {}
        self._logger = logger.getChild(name)

    @abstractmethod
    async def validate(
        self, target: Any, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Validate a target object.

        Args:
            target: Object to validate
            context: Additional context for validation

        Returns:
            ValidationResult with validation outcome
        """
        pass


class SignalValidator(BaseValidator):
    """Validator for trading signals."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize signal validator."""
        super().__init__("SignalValidator", config)

        # Validation thresholds
        self.min_confidence = config.get("min_confidence", 0.1) if config else 0.1
        self.max_signal_age_seconds = config.get("max_signal_age_seconds", 60) if config else 60
        self.required_fields = ["symbol", "direction", "confidence", "timestamp", "strategy_name"]

    async def validate(
        self, signal: Signal, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Validate a trading signal.

        Args:
            signal: Signal to validate
            context: Additional context (market_data, etc.)

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        try:
            # Basic field validation
            await self._validate_required_fields(signal, result)

            # Confidence validation
            await self._validate_confidence(signal, result)

            # Timestamp validation
            await self._validate_timestamp(signal, result)

            # Direction validation
            await self._validate_direction(signal, result)

            # Symbol validation
            await self._validate_symbol(signal, result)

            # Context-based validation
            if context:
                await self._validate_with_context(signal, context, result)

        except Exception as e:
            result.add_error(f"Signal validation failed: {e}")
            self._logger.error("Signal validation error", error=str(e))

        return result

    async def _validate_required_fields(self, signal: Signal, result: ValidationResult) -> None:
        """Validate required signal fields."""
        for field in self.required_fields:
            if not hasattr(signal, field) or getattr(signal, field) is None:
                result.add_error(f"Missing required field: {field}")

    async def _validate_confidence(self, signal: Signal, result: ValidationResult) -> None:
        """Validate signal confidence."""
        if signal.confidence < self.min_confidence:
            result.add_error(
                f"Signal confidence {signal.confidence} below minimum {self.min_confidence}"
            )

        if signal.confidence > 1.0:
            result.add_error(f"Signal confidence {signal.confidence} above maximum 1.0")

        if signal.confidence < 0.5:
            result.add_warning(f"Low signal confidence: {signal.confidence}")

    async def _validate_timestamp(self, signal: Signal, result: ValidationResult) -> None:
        """Validate signal timestamp."""
        now = datetime.now(timezone.utc)
        age_seconds = (now - signal.timestamp).total_seconds()

        if age_seconds > self.max_signal_age_seconds:
            result.add_error(f"Signal too old: {age_seconds}s > {self.max_signal_age_seconds}s")

        if signal.timestamp > now:
            result.add_error("Signal timestamp is in the future")

    async def _validate_direction(self, signal: Signal, result: ValidationResult) -> None:
        """Validate signal direction."""
        valid_directions = [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
        if signal.direction not in valid_directions:
            result.add_error(f"Invalid signal direction: {signal.direction}")

    async def _validate_symbol(self, signal: Signal, result: ValidationResult) -> None:
        """Validate signal symbol."""
        if not signal.symbol or len(signal.symbol) < 2:
            result.add_error("Invalid symbol format")

        # Check for common symbol formats
        if not signal.symbol.replace("/", "").replace("-", "").isalnum():
            result.add_warning(f"Unusual symbol format: {signal.symbol}")

    async def _validate_with_context(
        self, signal: Signal, context: dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate signal with market context."""
        market_data = context.get("market_data")
        if market_data and isinstance(market_data, MarketData):
            # Validate signal matches market data
            if signal.symbol != market_data.symbol:
                result.add_error("Signal symbol doesn't match market data")

            # Check if signal price is reasonable relative to market
            if hasattr(signal, "target_price") and signal.target_price:
                price_diff = abs(float(signal.target_price - market_data.price)) / float(
                    market_data.price
                )
                if price_diff > 0.1:  # 10% difference
                    result.add_warning(
                        f"Signal target price deviates significantly from market: {price_diff:.2%}"
                    )


class StrategyConfigValidator(BaseValidator):
    """Validator for strategy configurations."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize strategy config validator."""
        super().__init__("StrategyConfigValidator", config)

        # Validation rules
        self.required_parameters = {
            StrategyType.MOMENTUM: ["lookback_period", "momentum_threshold"],
            StrategyType.MEAN_REVERSION: ["mean_period", "deviation_threshold"],
            StrategyType.ARBITRAGE: ["spread_threshold", "max_position_size"],
            StrategyType.MARKET_MAKING: ["spread_size", "inventory_limit"],
        }

    async def validate(
        self, config: StrategyConfig, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Validate strategy configuration.

        Args:
            config: Strategy configuration to validate
            context: Additional context

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        try:
            # Basic validation
            await self._validate_basic_config(config, result)

            # Strategy-specific validation
            await self._validate_strategy_specific(config, result)

            # Risk parameter validation
            await self._validate_risk_parameters(config, result)

            # Exchange compatibility
            await self._validate_exchange_compatibility(config, result)

        except Exception as e:
            result.add_error(f"Configuration validation failed: {e}")
            self._logger.error("Config validation error", error=str(e))

        return result

    async def _validate_basic_config(
        self, config: StrategyConfig, result: ValidationResult
    ) -> None:
        """Validate basic configuration fields."""
        if not config.name or len(config.name) < 3:
            result.add_error("Strategy name must be at least 3 characters")

        if not config.strategy_type:
            result.add_error("Strategy type is required")

        if not config.parameters:
            result.add_error("Strategy parameters are required")

        if config.max_position_size and config.max_position_size <= 0:
            result.add_error("Max position size must be positive")

    async def _validate_strategy_specific(
        self, config: StrategyConfig, result: ValidationResult
    ) -> None:
        """Validate strategy-specific parameters."""
        strategy_type = config.strategy_type
        required_params = self.required_parameters.get(strategy_type, [])

        for param in required_params:
            if param not in config.parameters:
                result.add_error(f"Missing required parameter for {strategy_type}: {param}")

        # Validate parameter values
        await self._validate_parameter_values(config, result)

    async def _validate_parameter_values(
        self, config: StrategyConfig, result: ValidationResult
    ) -> None:
        """Validate parameter value ranges."""
        params = config.parameters

        # Common validations
        if "lookback_period" in params:
            if not isinstance(params["lookback_period"], int) or params["lookback_period"] < 1:
                result.add_error("Lookback period must be a positive integer")

        if "threshold" in params:
            threshold = params["threshold"]
            if isinstance(threshold, int | float) and (threshold < 0 or threshold > 1):
                result.add_warning("Threshold values typically range from 0 to 1")

        # Strategy-specific validations
        if config.strategy_type == StrategyType.MOMENTUM:
            if "momentum_threshold" in params:
                if params["momentum_threshold"] <= 0:
                    result.add_error("Momentum threshold must be positive")

        elif config.strategy_type == StrategyType.MEAN_REVERSION:
            if "deviation_threshold" in params:
                if params["deviation_threshold"] <= 0:
                    result.add_error("Deviation threshold must be positive")

    async def _validate_risk_parameters(
        self, config: StrategyConfig, result: ValidationResult
    ) -> None:
        """Validate risk management parameters."""
        if hasattr(config, "risk_parameters") and config.risk_parameters:
            risk_params = config.risk_parameters

            if "max_drawdown" in risk_params:
                max_dd = risk_params["max_drawdown"]
                if max_dd <= 0 or max_dd >= 1:
                    result.add_error("Max drawdown must be between 0 and 1")

            if "position_size_pct" in risk_params:
                pos_size = risk_params["position_size_pct"]
                if pos_size <= 0 or pos_size > 1:
                    result.add_error("Position size percentage must be between 0 and 1")

    async def _validate_exchange_compatibility(
        self, config: StrategyConfig, result: ValidationResult
    ) -> None:
        """Validate exchange compatibility."""
        if hasattr(config, "exchange_type") and config.exchange_type:
            # Check if strategy type is compatible with exchange
            if config.strategy_type == StrategyType.ARBITRAGE:
                # Arbitrage strategies may need multiple exchanges
                if not hasattr(config, "secondary_exchange"):
                    result.add_warning("Arbitrage strategies typically require multiple exchanges")


class MarketConditionValidator(BaseValidator):
    """Validator for market conditions and trading environment."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize market condition validator."""
        super().__init__("MarketConditionValidator", config)

        self.min_volume_threshold = config.get("min_volume", 1000) if config else 1000
        self.max_spread_pct = config.get("max_spread_pct", 0.01) if config else 0.01

    async def validate(
        self, market_data: MarketData, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Validate market conditions for trading.

        Args:
            market_data: Current market data
            context: Additional context

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        try:
            # Volume validation
            await self._validate_volume(market_data, result)

            # Spread validation
            await self._validate_spread(market_data, result)

            # Price validation
            await self._validate_price_data(market_data, result)

            # Market hours validation
            await self._validate_market_hours(market_data, result)

        except Exception as e:
            result.add_error(f"Market condition validation failed: {e}")
            self._logger.error("Market validation error", error=str(e))

        return result

    async def _validate_volume(self, market_data: MarketData, result: ValidationResult) -> None:
        """Validate trading volume."""
        if market_data.volume < self.min_volume_threshold:
            result.add_warning(f"Low volume: {market_data.volume} < {self.min_volume_threshold}")

    async def _validate_spread(self, market_data: MarketData, result: ValidationResult) -> None:
        """Validate bid-ask spread."""
        if market_data.bid and market_data.ask:
            spread_pct = float((market_data.ask - market_data.bid) / market_data.price)
            if spread_pct > self.max_spread_pct:
                result.add_warning(f"Wide spread: {spread_pct:.4f} > {self.max_spread_pct:.4f}")

    async def _validate_price_data(self, market_data: MarketData, result: ValidationResult) -> None:
        """Validate price data integrity."""
        if market_data.price <= 0:
            result.add_error("Invalid price: must be positive")

        if market_data.high and market_data.low:
            if market_data.low > market_data.high:
                result.add_error("Invalid OHLC data: low > high")

            if market_data.price < market_data.low or market_data.price > market_data.high:
                result.add_warning("Current price outside OHLC range")

    async def _validate_market_hours(
        self, market_data: MarketData, result: ValidationResult
    ) -> None:
        """Validate market trading hours."""
        # This would be extended with actual market hours logic
        current_hour = datetime.now(timezone.utc).hour

        # Basic check for major market hours (approximate)
        if current_hour < 6 or current_hour > 22:  # Outside major trading hours
            result.add_warning("Trading outside major market hours")


class CompositeValidator(BaseValidator):
    """Composite validator that runs multiple validators."""

    def __init__(self, validators: list[BaseValidator], name: str = "CompositeValidator"):
        """
        Initialize composite validator.

        Args:
            validators: List of validators to run
            name: Validator name
        """
        super().__init__(name)
        self.validators = validators

    async def validate(
        self, target: Any, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Run all validators and combine results.

        Args:
            target: Object to validate
            context: Additional context

        Returns:
            Combined ValidationResult
        """
        combined_result = ValidationResult(is_valid=True)

        # Run all validators concurrently
        tasks = [validator.validate(target, context) for validator in self.validators]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                combined_result.add_error(f"Validator {self.validators[i].name} failed: {result}")
            elif isinstance(result, ValidationResult):
                combined_result.merge(result)

        return combined_result


class ValidationFramework:
    """
    Main validation framework for strategies.

    Provides centralized validation for all strategy-related operations.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize validation framework.

        Args:
            config: Framework configuration
        """
        self.config = config or {}
        self._logger = logger

        # Initialize validators
        self.signal_validator = SignalValidator(self.config.get("signal_validation"))
        self.config_validator = StrategyConfigValidator(self.config.get("config_validation"))
        self.market_validator = MarketConditionValidator(self.config.get("market_validation"))

        # Create composite validators for common use cases
        self.trading_validator = CompositeValidator(
            [self.signal_validator, self.market_validator], "TradingValidator"
        )

    @time_execution
    async def validate_signal(
        self, signal: Signal, market_data: MarketData | None = None
    ) -> ValidationResult:
        """
        Validate a trading signal with market context.

        Args:
            signal: Signal to validate
            market_data: Optional market data for context

        Returns:
            ValidationResult
        """
        context = {}
        if market_data:
            context["market_data"] = market_data

        return await self.signal_validator.validate(signal, context)

    async def validate_strategy_config(self, config: StrategyConfig) -> ValidationResult:
        """
        Validate strategy configuration.

        Args:
            config: Configuration to validate

        Returns:
            ValidationResult
        """
        return await self.config_validator.validate(config)

    async def validate_market_conditions(self, market_data: MarketData) -> ValidationResult:
        """
        Validate market conditions for trading.

        Args:
            market_data: Market data to validate

        Returns:
            ValidationResult
        """
        return await self.market_validator.validate(market_data)

    async def validate_for_trading(
        self, signal: Signal, market_data: MarketData
    ) -> ValidationResult:
        """
        Comprehensive validation for trading operations.

        Args:
            signal: Trading signal
            market_data: Current market data

        Returns:
            Combined ValidationResult
        """
        context = {"market_data": market_data}
        return await self.trading_validator.validate(signal, context)

    async def batch_validate_signals(
        self, signals: list[Signal], market_data: MarketData | None = None
    ) -> list[tuple[Signal, ValidationResult]]:
        """
        Validate multiple signals in batch.

        Args:
            signals: List of signals to validate
            market_data: Optional market data for context

        Returns:
            List of (signal, validation_result) tuples
        """
        context = {}
        if market_data:
            context["market_data"] = market_data

        tasks = [self.signal_validator.validate(signal, context) for signal in signals]

        results = await asyncio.gather(*tasks)
        return list(zip(signals, results, strict=False))

    def add_custom_validator(
        self, validator: BaseValidator, validator_type: str = "custom"
    ) -> None:
        """
        Add a custom validator to the framework.

        Args:
            validator: Custom validator instance
            validator_type: Type of validator for organization
        """
        if not hasattr(self, f"_{validator_type}_validators"):
            setattr(self, f"_{validator_type}_validators", [])

        validators_list = getattr(self, f"_{validator_type}_validators")
        validators_list.append(validator)

        self._logger.info(
            "Custom validator added", validator_name=validator.name, validator_type=validator_type
        )

    def get_validation_stats(self) -> dict[str, Any]:
        """
        Get validation framework statistics.

        Returns:
            Dictionary with validation statistics
        """
        return {
            "framework_version": "1.0.0",
            "validators_registered": {
                "signal": 1,
                "config": 1,
                "market": 1,
                "composite": 1,
            },
            "config": self.config,
        }
