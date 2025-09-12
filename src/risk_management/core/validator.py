"""Risk validation using centralized utilities to eliminate duplication."""

from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.dependency_injection import injectable
from src.core.exceptions import PositionLimitError, ValidationError
from src.core.logging import get_logger
from src.core.types import OrderRequest, Position, RiskLevel, RiskLimits, Signal
from src.core.validator_registry import ValidatorInterface, register_validator
from src.utils.decimal_utils import format_decimal
from src.utils.decorators import UnifiedDecorator as dec
from src.utils.messaging_patterns import (
    BoundaryValidator,
    ErrorPropagationMixin,
    MessagingCoordinator,
)
from src.utils.risk_validation import UnifiedRiskValidator
from src.utils.validation.core import ValidationFramework


class RiskValidator(BaseComponent, ValidatorInterface, ErrorPropagationMixin):
    """
    Risk validation using centralized ValidationFramework.

    This eliminates duplication of risk validation logic and ensures consistent
    error propagation and data flow patterns.
    """

    def __init__(
        self,
        risk_limits: RiskLimits | None = None,
        messaging_coordinator: MessagingCoordinator | None = None,
    ):
        """
        Initialize risk validator using centralized utilities.

        Args:
            risk_limits: Risk limits configuration
            messaging_coordinator: Messaging coordinator for consistent data flow
        """
        super().__init__()  # Initialize BaseComponent
        self.base_validator = ValidationFramework()
        self.risk_limits = risk_limits or self._default_limits()
        # Use the logger from BaseComponent
        self.logger.bind(component="risk_validator")

        # Use centralized risk validator
        self.unified_validator = UnifiedRiskValidator(self.risk_limits)
        self._messaging_coordinator = messaging_coordinator or MessagingCoordinator("RiskValidator")

    def _default_limits(self) -> RiskLimits:
        """Get default risk limits."""
        return RiskLimits(
            max_position_size=Decimal("10000"),
            max_portfolio_risk=Decimal("0.1"),
            max_correlation=Decimal("0.7"),
            max_leverage=Decimal("3.0"),
            max_drawdown=Decimal("0.2"),
            max_daily_loss=Decimal("0.05"),
            max_positions=10,
            min_liquidity_ratio=Decimal("0.2"),
        )

    def validate(self, data: Any, **kwargs) -> bool:
        """
        Validate data against risk limits.

        Args:
            data: Data to validate (Order, Signal, Position, etc.)
            **kwargs: Additional validation parameters

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(data, OrderRequest):
            return self.validate_order(data, **kwargs)
        elif isinstance(data, Signal):
            return self.validate_signal(data, **kwargs)
        elif isinstance(data, Position):
            return self.validate_position(data, **kwargs)
        elif isinstance(data, dict):
            return self.validate_portfolio(data, **kwargs)
        else:
            raise ValidationError(f"Unknown data type for risk validation: {type(data)}")

    @dec.enhance(log=True, monitor=True)
    def validate_order(
        self,
        order: OrderRequest,
        portfolio_value: Decimal,
        current_positions: list[Position] | None = None,
    ) -> bool:
        """
        Validate order using centralized validator.

        Args:
            order: Order to validate
            portfolio_value: Current portfolio value
            current_positions: Current open positions

        Returns:
            True if order passes validation

        Raises:
            ValidationError: If order validation fails
            PositionLimitError: If position limits exceeded
        """
        try:
            # Validate at boundary
            order_data = {
                "symbol": order.symbol,
                "side": order.side,
                "type": getattr(order, "type", None) or getattr(order, "order_type", None),
                "quantity": format_decimal(order.quantity),
                "price": format_decimal(order.price) if order.price else None,
                "portfolio_value": str(portfolio_value),
            }
            BoundaryValidator.validate_database_entity(order_data, "validate_order")

            # Use base validator for order fields
            self.base_validator.validate_order(order_data)

            # Use centralized validator
            is_valid, error_message = self.unified_validator.validate_order(
                order, portfolio_value, current_positions
            )

            if not is_valid:
                # Determine appropriate exception type based on error
                if "position" in error_message.lower() and "limit" in error_message.lower():
                    raise PositionLimitError(error_message)
                else:
                    raise ValidationError(error_message)

            return True

        except Exception as e:
            self.propagate_validation_error(e, "validate_order")
            raise

    @dec.enhance(log=True)
    def validate_signal(self, signal: Signal, current_risk_level: RiskLevel | None = None) -> bool:
        """
        Validate trading signal using centralized validator.

        Args:
            signal: Signal to validate
            current_risk_level: Current portfolio risk level

        Returns:
            True if signal passes validation

        Raises:
            ValidationError: If signal validation fails
        """
        try:
            # Validate at boundary
            signal_data = signal.model_dump()
            BoundaryValidator.validate_database_entity(signal_data, "validate_signal")

            # Use centralized validator
            is_valid, error_message = self.unified_validator.validate_signal(
                signal, current_risk_level, emergency_stop_active=False
            )

            if not is_valid:
                raise ValidationError(error_message)

            return True

        except Exception as e:
            self.propagate_validation_error(e, "validate_signal")
            raise

    def validate_position(self, position: Position, portfolio_value: Decimal) -> bool:
        """
        Validate position using centralized validator.

        Args:
            position: Position to validate
            portfolio_value: Current portfolio value

        Returns:
            True if position is valid

        Raises:
            ValidationError: If position validation fails
        """
        # Use centralized validator
        is_valid, error_message = self.unified_validator.validate_position(
            position, portfolio_value
        )

        if not is_valid:
            raise ValidationError(error_message)

        return True

    def validate_portfolio(self, portfolio_data: dict[str, Any], **kwargs) -> bool:
        """
        Validate portfolio using centralized validator.

        Args:
            portfolio_data: Portfolio data including positions and metrics
            **kwargs: Additional validation parameters

        Returns:
            True if portfolio is valid

        Raises:
            ValidationError: If portfolio validation fails
        """
        # Use centralized validator
        is_valid, error_message = self.unified_validator.validate_portfolio(portfolio_data)

        if not is_valid:
            raise ValidationError(error_message)

        return True

    def update_limits(self, new_limits: RiskLimits) -> None:
        """
        Update risk limits in both local and centralized validators.

        Args:
            new_limits: New risk limits configuration
        """
        self.risk_limits = new_limits
        self.unified_validator.update_limits(new_limits)
        self.logger.info("Risk limits updated")


@injectable(singleton=True)
class RiskValidationUtility:
    """
    Utility for risk validation across the system.

    Provides centralized access to risk validation.
    Note: Renamed from RiskValidationService to avoid conflict with
    the proper service layer implementation.
    """

    def __init__(self):
        """Initialize risk validation utility."""
        self.validator = RiskValidator()
        self.logger = get_logger(__name__)

        # Register with global validator registry
        register_validator("risk", self.validator)

    def validate_order(self, order: OrderRequest, **kwargs) -> bool:
        """Validate order."""
        return self.validator.validate_order(order, **kwargs)

    def validate_signal(self, signal: Signal, **kwargs) -> bool:
        """Validate signal."""
        return self.validator.validate_signal(signal, **kwargs)

    def validate_portfolio(self, portfolio_data: dict, **kwargs) -> bool:
        """Validate portfolio."""
        return self.validator.validate_portfolio(portfolio_data, **kwargs)

    def set_risk_limits(self, limits: RiskLimits) -> None:
        """Update risk limits."""
        self.validator.update_limits(limits)
