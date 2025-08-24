"""Risk validation using centralized ValidationFramework."""

import logging
from decimal import Decimal
from typing import Any

from src.core.base.component import BaseComponent
from src.core.dependency_injection import injectable
from src.core.exceptions import PositionLimitError, ValidationError
from src.core.types.risk import RiskLevel, RiskLimits
from src.core.types.trading import OrderRequest, Position, Signal, SignalDirection
from src.core.validator_registry import ValidatorInterface, register_validator
from src.utils.decorators import UnifiedDecorator as dec
from src.utils.validation.core import ValidationFramework

# Module level logger
logger = logging.getLogger(__name__)


class RiskValidator(BaseComponent, ValidatorInterface):
    """
    Risk validation using centralized ValidationFramework.

    This eliminates duplication of risk validation logic.
    """

    def __init__(self, risk_limits: RiskLimits | None = None):
        """
        Initialize risk validator.

        Args:
            risk_limits: Risk limits configuration
        """
        super().__init__()  # Initialize BaseComponent
        self.base_validator = ValidationFramework()
        self.risk_limits = risk_limits or self._default_limits()
        self._logger = logger

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
        Validate order against risk limits.

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
        # Use base validator for order fields
        self.base_validator.validate_order(
            {
                "symbol": order.symbol,
                "side": order.side,
                "type": getattr(order, "type", None) or getattr(order, "order_type", None),
                "quantity": float(order.quantity),
                "price": float(order.price) if order.price else None,
            }
        )

        # Calculate order value
        order_value = order.quantity * (order.price or Decimal("1"))

        # Check position size limit
        if order_value > self.risk_limits.max_position_size:
            raise PositionLimitError(
                f"Order value {order_value} exceeds max position size "
                f"{self.risk_limits.max_position_size}"
            )

        # Check portfolio risk limit
        portfolio_risk = order_value / portfolio_value
        if portfolio_risk > self.risk_limits.max_portfolio_risk:
            raise ValidationError(
                f"Order risk {portfolio_risk:.2%} exceeds max portfolio risk "
                f"{self.risk_limits.max_portfolio_risk:.2%}"
            )

        # Check max positions limit
        if current_positions:
            if len(current_positions) >= self.risk_limits.max_positions:
                # Check if this is adding to existing position
                existing_position = next(
                    (p for p in current_positions if p.symbol == order.symbol), None
                )
                if not existing_position:
                    raise PositionLimitError(
                        f"Maximum number of positions ({self.risk_limits.max_positions}) reached"
                    )

        # Check leverage
        if order.leverage and order.leverage > self.risk_limits.max_leverage:
            raise ValidationError(
                f"Order leverage {order.leverage} exceeds max leverage "
                f"{self.risk_limits.max_leverage}"
            )

        return True

    @dec.enhance(log=True)
    def validate_signal(self, signal: Signal, current_risk_level: RiskLevel | None = None) -> bool:
        """
        Validate trading signal against risk constraints.

        Args:
            signal: Signal to validate
            current_risk_level: Current portfolio risk level

        Returns:
            True if signal passes validation

        Raises:
            ValidationError: If signal validation fails
        """
        # Check signal fields
        if not signal.symbol:
            raise ValidationError("Signal must have a symbol")

        # Handle both old and new signal formats
        action = getattr(signal, "action", None)
        if not action and hasattr(signal, "direction"):
            # Convert direction to action
            if signal.direction == SignalDirection.BUY:
                action = "BUY"
            elif signal.direction == SignalDirection.SELL:
                action = "SELL"
            else:
                action = str(signal.direction)

        if action and action not in ["BUY", "SELL", "CLOSE"]:
            raise ValidationError(f"Invalid signal action: {action}")

        # Check confidence threshold (handle both confidence and strength fields)
        confidence = getattr(signal, "confidence", None) or getattr(signal, "strength", None)

        min_confidence = 0.3  # Minimum confidence for high risk
        if current_risk_level == RiskLevel.HIGH:
            min_confidence = 0.6
        elif current_risk_level == RiskLevel.CRITICAL:
            min_confidence = 0.8

        if confidence and confidence < min_confidence:
            raise ValidationError(
                f"Signal confidence {confidence:.2f} below minimum "
                f"{min_confidence:.2f} for risk level {current_risk_level}"
            )

        # Check stop loss requirement
        if action in ["BUY", "SELL"] and not getattr(signal, "stop_loss", None):
            raise ValidationError("Signal must include stop loss")

        return True

    def validate_position(self, position: Position, portfolio_value: Decimal) -> bool:
        """
        Validate position against risk limits.

        Args:
            position: Position to validate
            portfolio_value: Current portfolio value

        Returns:
            True if position is valid

        Raises:
            ValidationError: If position validation fails
        """
        # Calculate position value
        position_value = position.quantity * position.current_price

        # Check position concentration
        concentration = position_value / portfolio_value
        max_concentration = Decimal("0.3")  # Max 30% in single position

        if concentration > max_concentration:
            raise ValidationError(
                f"Position concentration {concentration:.2%} exceeds "
                f"maximum {max_concentration:.2%}"
            )

        # Check unrealized loss
        if position.unrealized_pnl < 0:
            loss_percentage = abs(position.unrealized_pnl) / position_value
            if loss_percentage > Decimal("0.1"):  # 10% loss threshold
                self._logger.warning(
                    f"Position {position.symbol} has {loss_percentage:.2%} unrealized loss"
                )

        return True

    def validate_portfolio(self, portfolio_data: dict[str, Any], **kwargs) -> bool:
        """
        Validate entire portfolio against risk limits.

        Args:
            portfolio_data: Portfolio data including positions and metrics
            **kwargs: Additional validation parameters

        Returns:
            True if portfolio is valid

        Raises:
            ValidationError: If portfolio validation fails
        """
        # Extract portfolio metrics
        total_value = Decimal(str(portfolio_data.get("total_value", 0)))
        positions = portfolio_data.get("positions", [])
        current_drawdown = Decimal(str(portfolio_data.get("drawdown", 0)))
        daily_pnl = Decimal(str(portfolio_data.get("daily_pnl", 0)))

        # Check drawdown limit
        if current_drawdown > self.risk_limits.max_drawdown:
            raise ValidationError(
                f"Portfolio drawdown {current_drawdown:.2%} exceeds "
                f"maximum {self.risk_limits.max_drawdown:.2%}"
            )

        # Check daily loss limit
        if daily_pnl < 0:
            daily_loss = abs(daily_pnl) / total_value
            if daily_loss > self.risk_limits.max_daily_loss:
                raise ValidationError(
                    f"Daily loss {daily_loss:.2%} exceeds "
                    f"maximum {self.risk_limits.max_daily_loss:.2%}"
                )

        # Check position count
        if len(positions) > self.risk_limits.max_positions:
            raise ValidationError(
                f"Position count {len(positions)} exceeds maximum {self.risk_limits.max_positions}"
            )

        # Check correlation risk
        correlation = Decimal(str(portfolio_data.get("correlation", 0)))
        if correlation > self.risk_limits.max_correlation:
            self._logger.warning(
                f"Portfolio correlation {correlation:.2f} exceeds "
                f"recommended maximum {self.risk_limits.max_correlation:.2f}"
            )

        return True

    def update_limits(self, new_limits: RiskLimits) -> None:
        """
        Update risk limits.

        Args:
            new_limits: New risk limits configuration
        """
        self.risk_limits = new_limits
        self._logger.info("Risk limits updated")


@injectable(singleton=True)
class RiskValidationService:
    """
    Service for risk validation across the system.

    Provides centralized access to risk validation.
    """

    def __init__(self):
        """Initialize risk validation service."""
        self.validator = RiskValidator()
        self._logger = logger

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
