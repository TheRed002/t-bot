"""
Risk Validation Service Implementation.

This service handles all risk validation logic through dependency injection,
following proper service layer patterns. Now uses centralized validation utilities
to eliminate code duplication.
"""

from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.base.service import BaseService
from src.core.exceptions import ValidationError
from src.core.types import OrderRequest, Position, RiskLevel, Signal
from src.risk_management.interfaces import PortfolioRepositoryInterface
from src.utils.decimal_utils import ZERO, format_decimal, to_decimal
from src.utils.risk_validation import UnifiedRiskValidator

if TYPE_CHECKING:
    from src.state import StateService


class RiskValidationService(BaseService):
    """Service for validating trading signals and orders against risk constraints."""

    def __init__(
        self,
        portfolio_repository: PortfolioRepositoryInterface,
        state_service: "StateService",
        config=None,
        correlation_id: str | None = None,
    ):
        """
        Initialize risk validation service.

        Args:
            portfolio_repository: Repository for portfolio data access
            state_service: State service for state management
            config: Application configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="RiskValidationService",
            config=config.__dict__ if config else {},
            correlation_id=correlation_id,
        )

        self.portfolio_repository = portfolio_repository
        self.state_service = state_service
        self.config = config

        # Initialize centralized validator with default limits if no config
        risk_limits = None
        if config and hasattr(config, "risk"):
            from src.core.types.risk import RiskLimits

            risk_limits = RiskLimits(
                max_position_size=to_decimal(
                    str(getattr(config.risk, "max_position_size", "10000"))
                ),
                max_portfolio_risk=to_decimal(
                    str(getattr(config.risk, "max_portfolio_risk", "0.1"))
                ),
                max_correlation=to_decimal(str(getattr(config.risk, "max_correlation", "0.7"))),
                max_leverage=to_decimal(str(getattr(config.risk, "max_leverage", "3.0"))),
                max_drawdown=to_decimal(str(getattr(config.risk, "max_drawdown", "0.2"))),
                max_daily_loss=to_decimal(str(getattr(config.risk, "max_daily_loss", "0.05"))),
                max_positions=getattr(config.risk, "max_positions", 10),
                min_liquidity_ratio=to_decimal(
                    str(getattr(config.risk, "min_liquidity_ratio", "0.2"))
                ),
            )

        self.validator = UnifiedRiskValidator(risk_limits)

    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate trading signal against risk constraints using centralized validator.

        Args:
            signal: Trading signal to validate

        Returns:
            True if signal passes validation
        """
        try:
            # Get required data
            try:
                current_risk_level = await self._get_current_risk_level()
                emergency_stop_active = await self._is_emergency_stop_active()

            except Exception as e:
                self.logger.error(f"Error getting risk state: {e}")
                return False

            # Use centralized validator with consistent error handling matching monitoring patterns
            try:
                is_valid, error_message = self.validator.validate_signal(
                    signal, current_risk_level, emergency_stop_active
                )

                if not is_valid:
                    # Check validation error and propagate consistently
                    validation_error = ValidationError(
                        f"Signal validation failed: {error_message}",
                        field_name="signal",
                        field_value=signal.symbol if hasattr(signal, "symbol") else str(signal),
                        validation_rule="centralized_validator",
                    )
                    self.logger.error(f"Signal validation error: {validation_error}")
                    return False

            except Exception as e:
                # Check if it's a validation error and propagate accordingly
                if hasattr(e, "__class__") and (
                    "ValidationError" in e.__class__.__name__
                    or "DataValidationError" in e.__class__.__name__
                ):
                    self.logger.error(f"Validation error: {e}")
                else:
                    self.logger.error(f"Service error in validation: {e}")
                # Continue with fallback validation
                self.logger.warning(
                    "Falling back to local validation due to centralized validator error"
                )

            # Additional local validations with consistent patterns
            if not self._validate_signal_structure(signal):
                validation_error = ValidationError("Invalid signal structure")
                self.logger.error(f"Signal structure validation error: {validation_error}")
                return False

            if hasattr(signal, "strength") and signal.strength < 0.3:
                self.logger.warning(
                    "Signal strength too low",
                    symbol=signal.symbol,
                    strength=signal.strength,
                    min_required=0.3,
                )
                return False

            # Check current risk level with consistent error handling
            if current_risk_level == RiskLevel.CRITICAL:
                self.logger.warning(
                    "Risk level critical - rejecting signal",
                    symbol=signal.symbol,
                    risk_level=current_risk_level.value,
                )
                return False

            # Check emergency stop status
            if emergency_stop_active:
                self.logger.warning(
                    "Emergency stop active - rejecting signal",
                    symbol=signal.symbol,
                )
                return False

            # Check position limits for symbol
            if not await self._check_symbol_position_limits(signal.symbol):
                return False

            self.logger.info(
                "Signal validation passed",
                symbol=signal.symbol,
                direction=signal.direction.value
                if hasattr(signal.direction, "value")
                else str(signal.direction),
                strength=getattr(signal, "strength", "unknown"),
            )

            return True

        except Exception as e:
            # Use consistent error propagation for unexpected errors
            from src.utils.messaging_patterns import ErrorPropagationMixin

            error_handler = ErrorPropagationMixin()
            error_handler.propagate_service_error(e, "signal_validation")
            return False

    async def validate_order(self, order: OrderRequest) -> bool:
        """
        Validate order against risk constraints using centralized validator.

        Args:
            order: Order request to validate

        Returns:
            True if order passes validation
        """
        try:
            # Get required data for validation
            portfolio_value = await self._get_portfolio_value()
            current_positions = await self._get_current_positions()

            # Use centralized validator first
            try:
                is_valid, error_message = self.validator.validate_order(
                    order, portfolio_value, current_positions
                )

                if not is_valid:
                    self.logger.warning(f"Order validation failed: {error_message}")
                    return False
            except Exception as e:
                self.logger.error(f"Centralized validator error: {e}")
                # Fall back to local validation - error logged

            # Additional local validations
            if not self._validate_order_structure(order):
                return False

            if await self._is_emergency_stop_active():
                self.logger.warning(
                    "Emergency stop active - rejecting order",
                    symbol=order.symbol,
                )
                return False

            if not await self._validate_order_size_limits(order):
                return False

            if not await self._validate_portfolio_exposure(order):
                return False

            self.logger.info(
                "Order validation passed",
                symbol=order.symbol,
                side=order.side.value,
                quantity=format_decimal(order.quantity),
            )

            return True

        except Exception as e:
            self.logger.error(f"Order validation error: {e}")
            return False

    async def validate_portfolio_limits(self, new_position: Position) -> bool:
        """
        Validate that adding a position won't violate portfolio limits using centralized utilities.

        Args:
            new_position: Position to be added

        Returns:
            True if position addition is allowed
        """
        try:
            from src.utils.risk_validation import check_position_limits

            # Get current portfolio data
            current_positions = await self._get_current_positions()
            portfolio_value = await self._get_portfolio_value()

            # Use centralized position limit validation
            try:
                is_valid, error_message = check_position_limits(
                    new_position_count=len(current_positions) + 1,
                    max_positions=self._get_max_total_positions(),
                    symbol=new_position.symbol,
                    symbol_positions=len(await self._get_positions_for_symbol(new_position.symbol)),
                    max_per_symbol=self._get_max_positions_per_symbol(),
                )

                if not is_valid:
                    self.logger.warning(f"Position limits check failed: {error_message}")
                    return False
            except Exception as e:
                self.logger.error(f"Centralized position limits error: {e}")
                # Fall back to local validation - error logged

            # Use centralized validator for position validation
            try:
                is_valid, error_message = self.validator.validate_position(
                    new_position, portfolio_value
                )
                if not is_valid:
                    self.logger.warning(f"Position validation failed: {error_message}")
                    return False
            except Exception as e:
                self.logger.error(f"Centralized position validator error: {e}")
                # Fall back to local validation - error logged

            # Additional local checks
            if len(current_positions) >= self._get_max_total_positions():
                self.logger.warning(
                    "Total position limit reached",
                    current_positions=len(current_positions),
                    max_positions=self._get_max_total_positions(),
                )
                return False

            symbol_positions = await self._get_positions_for_symbol(new_position.symbol)
            if len(symbol_positions) >= self._get_max_positions_per_symbol():
                self.logger.warning(
                    "Symbol position limit reached",
                    symbol=new_position.symbol,
                    current_positions=len(symbol_positions),
                    max_positions=self._get_max_positions_per_symbol(),
                )
                return False

            if not await self._validate_position_exposure(new_position):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Portfolio limit validation error: {e}")
            return False

    def _validate_signal_structure(self, signal: Signal) -> bool:
        """Validate basic signal structure."""
        if not signal:
            self.logger.error("Signal is None")
            return False

        if not hasattr(signal, "symbol") or not signal.symbol:
            self.logger.error("Signal missing symbol")
            return False

        if not hasattr(signal, "direction") or not signal.direction:
            self.logger.error("Signal missing direction")
            return False

        if not hasattr(signal, "strength") or not isinstance(
            signal.strength, (int, float, Decimal)
        ):
            self.logger.error("Signal missing or invalid strength")
            return False

        if not (0 < signal.strength <= 1):
            self.logger.error(f"Signal strength out of range: {signal.strength}")
            return False

        return True

    def _validate_order_structure(self, order: OrderRequest) -> bool:
        """Validate basic order structure."""
        if not order:
            self.logger.error("Order is None")
            return False

        if not hasattr(order, "symbol") or not order.symbol:
            self.logger.error("Order missing symbol")
            return False

        if not hasattr(order, "side") or not order.side:
            self.logger.error("Order missing side")
            return False

        if not hasattr(order, "quantity") or order.quantity <= ZERO:
            self.logger.error("Order missing or invalid quantity")
            return False

        return True

    async def _validate_order_size_limits(self, order: OrderRequest) -> bool:
        """Validate order size against limits."""
        try:
            # Get portfolio metrics
            portfolio_value = await self._get_portfolio_value()
            if portfolio_value <= ZERO:
                return True  # No portfolio value to limit against

            # Calculate order value
            order_price = getattr(order, "price", None) or await self._get_current_price(
                order.symbol
            )
            if not order_price:
                self.logger.warning(f"Cannot determine price for order validation: {order.symbol}")
                return True  # Allow order if price cannot be determined

            order_value = order.quantity * order_price

            max_order_size = portfolio_value * self._get_max_position_size_pct()
            if order_value > max_order_size:
                self.logger.warning(
                    "Order size exceeds limit",
                    symbol=order.symbol,
                    order_value=format_decimal(order_value),
                    max_size=format_decimal(max_order_size),
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Order size validation error: {e}")
            return True  # Allow order if validation fails

    async def _validate_portfolio_exposure(self, order: OrderRequest) -> bool:
        """Validate order against portfolio exposure limits."""
        try:
            # Get current portfolio metrics
            current_exposure = await self._get_current_exposure()
            portfolio_value = await self._get_portfolio_value()

            if portfolio_value <= ZERO:
                return True

            # Calculate additional exposure from order
            order_price = getattr(order, "price", None) or await self._get_current_price(
                order.symbol
            )
            if not order_price:
                return True

            additional_exposure = order.quantity * order_price
            potential_exposure = current_exposure + additional_exposure

            max_exposure = portfolio_value * self._get_max_portfolio_exposure_pct()
            if potential_exposure > max_exposure:
                self.logger.warning(
                    "Order would exceed portfolio exposure limit",
                    current_exposure=format_decimal(current_exposure),
                    additional_exposure=format_decimal(additional_exposure),
                    max_exposure=format_decimal(max_exposure),
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Portfolio exposure validation error: {e}")
            return True

    async def _validate_position_exposure(self, position: Position) -> bool:
        """Validate position exposure."""
        try:
            position_value = position.quantity * position.current_price
            portfolio_value = await self._get_portfolio_value()

            if portfolio_value <= ZERO:
                return True

            # Check position size against limits
            position_pct = position_value / portfolio_value
            max_position_pct = self._get_max_position_size_pct()

            if position_pct > max_position_pct:
                self.logger.warning(
                    "Position would exceed size limit",
                    symbol=position.symbol,
                    position_pct=format_decimal(position_pct),
                    max_pct=format_decimal(max_position_pct),
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Position exposure validation error: {e}")
            return True

    async def _check_symbol_position_limits(self, symbol: str) -> bool:
        """Check position limits for specific symbol."""
        try:
            symbol_positions = await self._get_positions_for_symbol(symbol)
            max_positions = self._get_max_positions_per_symbol()

            if len(symbol_positions) >= max_positions:
                self.logger.warning(
                    "Max positions per symbol reached",
                    symbol=symbol,
                    current_positions=len(symbol_positions),
                    max_allowed=max_positions,
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Symbol position limit check error: {e}")
            return True

    # Helper methods for accessing data through services

    async def _get_current_risk_level(self) -> RiskLevel:
        """Get current risk level from state service."""
        try:
            # Use state service to get current risk level
            from src.core.types import StateType

            risk_state = await self.state_service.get_state(StateType.RISK_STATE, "current_level")
            if risk_state and "risk_level" in risk_state:
                return RiskLevel(risk_state["risk_level"])
            return RiskLevel.LOW
        except Exception as e:
            self.logger.error(f"Error getting risk level: {e}")
            return RiskLevel.LOW

    async def _is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is active."""
        try:
            # Use state service to check emergency stop
            from src.core.types import StateType

            emergency_state = await self.state_service.get_state(
                StateType.RISK_STATE, "emergency_stop"
            )
            return bool(emergency_state and emergency_state.get("active", False))
        except Exception as e:
            self.logger.error(f"Error checking emergency stop: {e}")
            return False

    async def _get_current_positions(self) -> list[Position]:
        """Get current positions from state service."""
        try:
            # Get positions from state service instead of direct database access
            from src.core.types import StateType

            positions_state = await self.state_service.get_state(
                StateType.PORTFOLIO_STATE, "positions"
            )
            if not positions_state:
                return []

            # Convert state data to Position objects if needed
            positions = []
            for pos_data in positions_state.get("open_positions", []):
                if isinstance(pos_data, Position):
                    positions.append(pos_data)

            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    async def _get_positions_for_symbol(self, symbol: str) -> list[Position]:
        """Get positions for specific symbol from state service."""
        try:
            positions = await self._get_current_positions()
            symbol_positions = [pos for pos in positions if pos.symbol == symbol]
            return symbol_positions
        except Exception as e:
            self.logger.error(f"Error getting positions for {symbol}: {e}")
            return []

    async def _get_portfolio_value(self) -> Decimal:
        """Get current portfolio value."""
        try:
            # This would typically calculate portfolio value from positions
            # For now, return a default value
            return ZERO  # Portfolio value calculation not implemented
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return ZERO

    async def _get_current_exposure(self) -> Decimal:
        """Get current portfolio exposure."""
        try:
            # This would typically calculate exposure from positions
            return ZERO  # Exposure calculation not implemented
        except Exception as e:
            self.logger.error(f"Error getting current exposure: {e}")
            return ZERO

    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol."""
        try:
            # This would get price from market data service
            return None  # Price retrieval not implemented
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def _get_max_total_positions(self) -> int:
        """Get maximum total positions from config."""
        if self.config and hasattr(self.config, "risk"):
            return getattr(self.config.risk, "max_total_positions", 10)
        return 10

    def _get_max_positions_per_symbol(self) -> int:
        """Get maximum positions per symbol from config."""
        if self.config and hasattr(self.config, "risk"):
            return getattr(self.config.risk, "max_positions_per_symbol", 3)
        return 3

    def _get_max_position_size_pct(self):
        """Get maximum position size percentage from config."""
        from src.utils.decimal_utils import to_decimal

        if self.config and hasattr(self.config, "risk"):
            return to_decimal(getattr(self.config.risk, "max_position_size_pct", "0.25"))
        return to_decimal("0.25")

    def _get_max_portfolio_exposure_pct(self):
        """Get maximum portfolio exposure percentage from config."""
        from src.utils.decimal_utils import to_decimal

        if self.config and hasattr(self.config, "risk"):
            return to_decimal(getattr(self.config.risk, "max_portfolio_exposure_pct", "0.80"))
        return to_decimal("0.80")
