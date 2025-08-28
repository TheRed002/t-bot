"""
Unified Risk Manager Implementation - DEPRECATED.

DEPRECATED: This module is deprecated in favor of the new RiskService.
The new RiskService (src/risk_management/service.py) provides all risk management
functionality with enterprise-grade features including:
- DatabaseService integration (no direct DB access)
- StateService integration for state management
- Comprehensive caching layer
- Enhanced error handling with circuit breakers
- Real-time risk monitoring

This manager is maintained for backward compatibility only.
New implementations should use RiskService directly.
"""

from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.config.main import Config, get_config
from src.core.dependency_injection import injectable
from src.core.exceptions import RiskManagementError
from src.core.logging import get_logger

# Import from core types
from src.core.types import (
    MarketData,
    OrderRequest,
    Position,
    PositionLimits,
    PositionSizeMethod,
    RiskLevel,
    RiskMetrics,
    Signal,
)
from src.core.types.trading import OrderType
from src.error_handling import ErrorHandler

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from src.database.service import DatabaseService
    from src.state import StateService
from src.utils.decimal_utils import (
    ZERO,
    format_decimal,
    safe_divide,
    to_decimal,
)

# Import decorators
from src.utils.decorators import time_execution

# Import risk management components
from .base import BaseRiskManager
from .portfolio_limits import PortfolioLimits
from .position_sizing import PositionSizer
from .risk_metrics import RiskCalculator
from .service import RiskService

logger = get_logger(__name__)


@injectable(singleton=True)
class RiskManager(BaseRiskManager):
    """
    DEPRECATED: Unified Risk Manager implementation.

    This class is deprecated in favor of RiskService which provides:
    - Enterprise-grade service architecture
    - DatabaseService integration (no direct DB access)
    - StateService integration
    - Comprehensive caching
    - Enhanced error handling
    - Real-time monitoring

    This class now acts as a wrapper around RiskService for backward compatibility.
    New code should use RiskService directly.

    Components (DEPRECATED):
    - Position sizing strategies -> Use RiskService.calculate_position_size()
    - Portfolio limits enforcement -> Use RiskService.validate_order()
    - Risk metrics calculation -> Use RiskService.calculate_risk_metrics()
    - Emergency controls -> Use RiskService.trigger_emergency_stop()
    - Error handling integration -> Built into RiskService
    """

    def __init__(
        self,
        config: Config | None = None,
        database_service: "DatabaseService | None" = None,
        state_service: "StateService | None" = None,
    ):
        """
        Initialize DEPRECATED risk manager with all components.

        DEPRECATED: Use RiskService directly instead of this wrapper.

        Args:
            config: Application configuration (uses global config if None)
            database_service: Database service for data access (required for new functionality)
            state_service: State service for state management (required for new functionality)
        """
        # Use provided config or get default config
        if config is None:
            config = get_config()

        super().__init__(config)

        # DEPRECATED: Legacy components for backward compatibility
        # New implementations should use RiskService directly
        self.position_sizer = PositionSizer(config)
        self.portfolio_limits = PortfolioLimits(config)
        self.risk_calculator = RiskCalculator(config)

        # Initialize error handler for proper error management
        self.error_handler = ErrorHandler(config)

        # NEW: Initialize RiskService if services are provided
        self.risk_service: RiskService | None = None
        if database_service and state_service:
            try:
                self.risk_service = RiskService(
                    database_service=database_service, state_service=state_service, config=config
                )
                logger.info("RiskManager initialized with RiskService integration")
            except Exception as e:
                logger.warning(f"Failed to initialize RiskService: {e}")
        else:
            logger.warning(
                "RiskManager initialized in DEPRECATED mode - "
                "DatabaseService and StateService not provided. "
                "Consider migrating to RiskService directly."
            )

        # Initialize position limits
        self.position_limits = PositionLimits(
            max_position_size=to_decimal(config.risk.max_position_size),
            max_positions=config.risk.max_total_positions,
            max_leverage=to_decimal(config.risk.max_leverage),
            min_position_size=to_decimal(
                config.risk.max_position_size * Decimal("0.01")
            ),  # 1% of max
        )

        # Initialize tracking
        self.active_positions: dict[str, list[Position]] = {}
        self.total_exposure = ZERO
        self.current_risk_level = RiskLevel.LOW

        logger.warning(
            "DEPRECATED Risk Manager initialized - consider migrating to RiskService",
            position_sizing_method=config.risk.position_sizing_method,
            max_position_size=format_decimal(self.position_limits.max_position_size),
            max_positions=self.position_limits.max_positions,
            risk_service_available=self.risk_service is not None,
        )

    @time_execution
    async def calculate_position_size(
        self, signal: Signal, available_capital: Decimal, current_price: Decimal
    ) -> Decimal:
        """
        Calculate position size based on signal and risk parameters.

        DEPRECATED: Use RiskService.calculate_position_size() directly.

        Args:
            signal: Trading signal
            available_capital: Available capital for position
            current_price: Current market price

        Returns:
            Position size in base currency
        """
        try:
            # NEW: Delegate to RiskService if available
            if self.risk_service is not None:
                logger.debug("Delegating position size calculation to RiskService")
                return await self.risk_service.calculate_position_size(
                    signal=signal,
                    available_capital=available_capital,
                    current_price=current_price,
                )

            # DEPRECATED: Fallback to legacy implementation
            logger.warning("Using DEPRECATED position sizing - migrate to RiskService")

            # Calculate base position size
            position_size = await self.position_sizer.calculate_position_size(
                signal=signal,
                portfolio_value=available_capital,  # Using available capital as portfolio proxy
                method=PositionSizeMethod(self.config.risk.position_sizing_method),
            )

            # Apply portfolio limits (manual implementation since PortfolioLimits is a Pydantic model)
            # Limit by max position size
            if position_size > self.position_limits.max_position_size:
                position_size = self.position_limits.max_position_size
                logger.info(
                    "Position size limited by max position size",
                    original_size=format_decimal(position_size),
                    max_size=format_decimal(self.position_limits.max_position_size),
                )

            # Limit by min position size
            if position_size < self.position_limits.min_position_size:
                position_size = self.position_limits.min_position_size
                logger.info(
                    "Position size increased to meet min position size",
                    original_size=format_decimal(position_size),
                    min_size=format_decimal(self.position_limits.min_position_size),
                )

            logger.info(
                "Position size calculated",
                symbol=signal.symbol,
                signal_strength=signal.strength,
                position_size=format_decimal(position_size),
                available_capital=format_decimal(available_capital),
            )

            return position_size

        except Exception as e:
            logger.error(
                "Position sizing failed",
                symbol=signal.symbol,
                error=str(e),
            )
            raise RiskManagementError(f"Position sizing failed: {e}") from e

    @time_execution
    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate trading signal against risk constraints.

        DEPRECATED: Use RiskService.validate_signal() directly.

        Args:
            signal: Signal to validate

        Returns:
            True if signal passes risk validation
        """
        try:
            # NEW: Delegate to RiskService if available
            if self.risk_service is not None:
                logger.debug("Delegating signal validation to RiskService")
                return await self.risk_service.validate_signal(signal)

            # DEPRECATED: Fallback to legacy implementation
            logger.warning("Using DEPRECATED signal validation - migrate to RiskService")
            # Check confidence threshold
            if signal.strength < 0.3:  # Use strength instead of confidence
                logger.warning(
                    "Signal strength too low",
                    symbol=signal.symbol,
                    strength=signal.strength,
                    min_required=0.3,
                )
                return False

            # Check current risk level
            if self.current_risk_level == RiskLevel.CRITICAL:
                logger.warning(
                    "Risk level critical - rejecting signal",
                    symbol=signal.symbol,
                    risk_level=self.current_risk_level.value,
                )
                return False

            # Check position limits
            symbol_positions = self.active_positions.get(signal.symbol, [])
            if len(symbol_positions) >= self.position_limits.max_positions_per_symbol:
                logger.warning(
                    "Max positions per symbol reached",
                    symbol=signal.symbol,
                    current_positions=len(symbol_positions),
                    max_allowed=self.position_limits.max_positions_per_symbol,
                )
                return False

            logger.debug(
                "Signal validated",
                symbol=signal.symbol,
                direction=signal.direction.value,
                confidence=signal.confidence,
            )

            return True

        except Exception as e:
            logger.error(
                "Signal validation failed",
                symbol=signal.symbol,
                error=str(e),
            )
            return False

    @time_execution
    async def validate_order(self, order: OrderRequest) -> bool:
        """
        Validate order against risk constraints.

        DEPRECATED: Use RiskService.validate_order() directly.

        Args:
            order: Order to validate

        Returns:
            True if order passes risk validation
        """
        try:
            # NEW: Delegate to RiskService if available
            if self.risk_service is not None:
                logger.debug("Delegating order validation to RiskService")
                return await self.risk_service.validate_order(order)

            # DEPRECATED: Fallback to legacy implementation
            logger.warning("Using DEPRECATED order validation - migrate to RiskService")
            # Check order size limits
            if order.quantity > self.position_limits.max_position_size:
                logger.warning(
                    "Order size exceeds limit",
                    symbol=order.symbol,
                    order_size=format_decimal(order.quantity),
                    max_size=format_decimal(self.position_limits.max_position_size),
                )
                return False

            # Check portfolio exposure
            potential_exposure = self.total_exposure + (
                order.quantity * order.price if hasattr(order, "price") else order.quantity
            )
            if potential_exposure > self.position_limits.max_portfolio_exposure:
                logger.warning(
                    "Order would exceed portfolio exposure limit",
                    current_exposure=format_decimal(self.total_exposure),
                    additional_exposure=format_decimal(order.quantity),
                    max_exposure=format_decimal(self.position_limits.max_portfolio_exposure),
                )
                return False

            logger.debug(
                "Order validated",
                symbol=order.symbol,
                side=order.side.value,
                quantity=format_decimal(order.quantity),
            )

            return True

        except Exception as e:
            logger.error(
                "Order validation failed",
                symbol=order.symbol,
                error=str(e),
            )
            return False

    @time_execution
    async def calculate_risk_metrics(
        self,
        positions: list[Position],
        market_data: list[MarketData],
        returns: list[float] | None = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for positions.

        DEPRECATED: Use RiskService.calculate_risk_metrics() directly.

        Args:
            positions: Current positions
            market_data: Current market data
            returns: Historical returns (optional)

        Returns:
            Calculated risk metrics
        """
        try:
            # NEW: Delegate to RiskService if available
            if self.risk_service is not None:
                logger.debug("Delegating risk metrics calculation to RiskService")
                return await self.risk_service.calculate_risk_metrics(positions, market_data)

            # DEPRECATED: Fallback to legacy implementation
            logger.warning("Using DEPRECATED risk metrics calculation - migrate to RiskService")

            metrics = await self.risk_calculator.calculate_risk_metrics(
                positions=positions,
                market_data=market_data,
            )

            # Update risk level based on metrics
            self._update_risk_level(metrics)

            logger.info(
                "Risk metrics calculated",
                total_exposure=format_decimal(metrics.total_exposure),
                var_95=format_decimal(metrics.var_95) if metrics.var_95 else None,
                sharpe_ratio=metrics.sharpe_ratio,
                current_risk_level=self.current_risk_level.value,
            )

            return metrics

        except Exception as e:
            logger.error(
                "Risk metrics calculation failed",
                error=str(e),
            )
            # Return default metrics on error
            return RiskMetrics()

    def update_positions(self, positions: list[Position]) -> None:
        """
        Update tracked positions and exposure.

        Args:
            positions: Current positions
        """
        try:
            # Group positions by symbol
            self.active_positions.clear()
            self.total_exposure = ZERO

            for position in positions:
                if position.symbol not in self.active_positions:
                    self.active_positions[position.symbol] = []
                self.active_positions[position.symbol].append(position)

                # Update total exposure
                position_value = position.quantity * position.current_price
                self.total_exposure += position_value

            logger.debug(
                "Positions updated",
                total_positions=len(positions),
                unique_symbols=len(self.active_positions),
                total_exposure=format_decimal(self.total_exposure),
            )

        except Exception as e:
            logger.error(
                "Position update failed",
                error=str(e),
            )

    def check_risk_limits(self) -> tuple[bool, str]:
        """
        Check if current risk levels are within limits.

        Returns:
            Tuple of (within_limits, message)
        """
        try:
            # Check total exposure
            if self.total_exposure > self.position_limits.max_portfolio_exposure:
                message = (
                    f"Portfolio exposure {format_decimal(self.total_exposure)} "
                    f"exceeds limit {format_decimal(self.position_limits.max_portfolio_exposure)}"
                )
                logger.warning(message)
                return False, message

            # Check position count
            total_positions = sum(len(pos) for pos in self.active_positions.values())
            if total_positions > self.position_limits.max_total_positions:
                message = (
                    f"Total positions {total_positions} "
                    f"exceeds limit {self.position_limits.max_total_positions}"
                )
                logger.warning(message)
                return False, message

            # Check risk level
            if self.current_risk_level == RiskLevel.CRITICAL:
                message = "Risk level is CRITICAL"
                logger.warning(message)
                return False, message

            return True, "Risk within limits"

        except Exception as e:
            logger.error("Risk limit check failed", error=str(e))
            return False, f"Risk limit check failed: {e}"

    def get_position_limits(self) -> PositionLimits:
        """
        Get current position limits.

        Returns:
            Position limits configuration
        """
        return self.position_limits

    async def emergency_stop(self, reason: str) -> None:
        """
        Execute emergency stop of all trading.

        Args:
            reason: Reason for emergency stop
        """
        try:
            logger.critical(
                "EMERGENCY STOP TRIGGERED",
                reason=reason,
                active_positions=len(self.active_positions),
                total_exposure=format_decimal(self.total_exposure),
            )

            # Set risk level to critical
            self.current_risk_level = RiskLevel.CRITICAL

            # Clear all positions tracking (actual closing handled elsewhere)
            self.active_positions.clear()
            self.total_exposure = ZERO

            logger.info("Emergency stop completed")

        except Exception as e:
            logger.error(
                "Emergency stop failed",
                reason=reason,
                error=str(e),
            )

    def _update_risk_level(self, metrics: RiskMetrics) -> None:
        """
        Update current risk level based on metrics.

        Args:
            metrics: Current risk metrics
        """
        try:
            # Determine risk level based on metrics
            if metrics.max_drawdown and metrics.max_drawdown > Decimal("0.3"):
                self.current_risk_level = RiskLevel.CRITICAL
            elif metrics.max_drawdown and metrics.max_drawdown > Decimal("0.2"):
                self.current_risk_level = RiskLevel.HIGH
            elif metrics.max_drawdown and metrics.max_drawdown > Decimal("0.1"):
                self.current_risk_level = RiskLevel.MEDIUM
            else:
                self.current_risk_level = RiskLevel.LOW

            logger.debug(
                "Risk level updated",
                new_level=self.current_risk_level.value,
                max_drawdown=format_decimal(metrics.max_drawdown) if metrics.max_drawdown else None,
            )

        except Exception as e:
            logger.error("Risk level update failed", error=str(e))

    def calculate_leverage(self) -> Decimal:
        """
        Calculate current portfolio leverage.

        Returns:
            Current leverage ratio
        """
        try:
            if not self.config.risk.available_capital or self.config.risk.available_capital == ZERO:
                return ZERO

            leverage = safe_divide(
                self.total_exposure,
                to_decimal(self.config.risk.available_capital),
            )

            logger.debug(
                "Leverage calculated",
                total_exposure=format_decimal(self.total_exposure),
                available_capital=format_decimal(to_decimal(self.config.risk.available_capital)),
                leverage=format_decimal(leverage),
            )

            return leverage

        except Exception as e:
            logger.error("Leverage calculation failed", error=str(e))
            return ZERO

    # Additional helper methods for compatibility
    def _calculate_signal_score(self, signal: Signal) -> float:
        """Calculate signal score for position sizing."""
        return float(signal.confidence)

    def _apply_portfolio_constraints(self, size: Decimal, symbol: str) -> Decimal:
        """Apply portfolio-level constraints to position size."""
        return self.portfolio_limits.apply_limits(
            position_size=size,
            symbol=symbol,
            current_positions=self.active_positions,
        )

    def _check_capital_availability(self, required: Decimal, available: Decimal) -> bool:
        """Check if sufficient capital is available."""
        return available >= required

    @time_execution
    async def check_portfolio_limits(self, new_position: Position) -> bool:
        """
        Check if adding a new position would violate portfolio limits.

        DEPRECATED: Use RiskService.validate_order() directly.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if position addition is allowed
        """
        try:
            # NEW: Delegate to RiskService if available
            if self.risk_service is not None:
                logger.debug("Delegating portfolio limit check to RiskService")
                # Convert position to order for validation
                order = OrderRequest(
                    symbol=new_position.symbol,
                    side=new_position.side,
                    order_type=OrderType.MARKET,  # Default type
                    quantity=new_position.quantity,
                )
                return await self.risk_service.validate_order(order)

            # DEPRECATED: Fallback to legacy implementation
            logger.warning("Using DEPRECATED portfolio limit check - migrate to RiskService")

            # Check if adding position would exceed portfolio limits
            position_value = new_position.quantity * new_position.current_price
            potential_exposure = self.total_exposure + position_value

            if potential_exposure > self.position_limits.max_portfolio_exposure:
                logger.warning(
                    "Adding position would exceed portfolio exposure limit",
                    current_exposure=format_decimal(self.total_exposure),
                    position_value=format_decimal(position_value),
                    max_exposure=format_decimal(self.position_limits.max_portfolio_exposure),
                )
                return False

            # Check position count limits
            total_positions = sum(len(pos) for pos in self.active_positions.values())
            if total_positions >= self.position_limits.max_total_positions:
                logger.warning(
                    "Adding position would exceed total position limit",
                    current_positions=total_positions,
                    max_positions=self.position_limits.max_total_positions,
                )
                return False

            # Check symbol-specific limits
            symbol_positions = self.active_positions.get(new_position.symbol, [])
            if len(symbol_positions) >= self.position_limits.max_positions_per_symbol:
                logger.warning(
                    "Adding position would exceed symbol position limit",
                    symbol=new_position.symbol,
                    current_positions=len(symbol_positions),
                    max_positions=self.position_limits.max_positions_per_symbol,
                )
                return False

            return True

        except Exception as e:
            logger.error("Portfolio limit check failed", error=str(e))
            return False

    @time_execution
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool:
        """
        Determine if a position should be closed based on risk criteria.

        DEPRECATED: Use RiskService.should_exit_position() directly.

        Args:
            position: Position to evaluate
            market_data: Current market data for the position

        Returns:
            bool: True if position should be closed
        """
        try:
            # NEW: Delegate to RiskService if available
            if self.risk_service is not None:
                logger.debug("Delegating exit decision to RiskService")
                return await self.risk_service.should_exit_position(position, market_data)

            # DEPRECATED: Fallback to legacy implementation
            logger.warning("Using DEPRECATED exit decision logic - migrate to RiskService")

            # Check stop-loss conditions
            if hasattr(position, "stop_loss") and position.stop_loss:
                if position.side.value == "BUY" and market_data.price <= position.stop_loss:
                    logger.info(
                        "Stop-loss triggered for long position",
                        symbol=position.symbol,
                        current_price=format_decimal(market_data.price),
                        stop_loss=format_decimal(position.stop_loss),
                    )
                    return True
                elif position.side.value == "SELL" and market_data.price >= position.stop_loss:
                    logger.info(
                        "Stop-loss triggered for short position",
                        symbol=position.symbol,
                        current_price=format_decimal(market_data.price),
                        stop_loss=format_decimal(position.stop_loss),
                    )
                    return True

            # Check if position loss exceeds risk limits
            if position.unrealized_pnl:
                position_loss_pct = safe_divide(
                    abs(position.unrealized_pnl), position.quantity * position.entry_price
                )

                max_position_loss = to_decimal(
                    self.config.risk.max_position_loss_pct
                    if hasattr(self.config.risk, "max_position_loss_pct")
                    else 0.1
                )

                if position.unrealized_pnl < ZERO and position_loss_pct > max_position_loss:
                    logger.info(
                        "Position loss limit exceeded",
                        symbol=position.symbol,
                        loss_pct=format_decimal(position_loss_pct),
                        max_loss=format_decimal(max_position_loss),
                    )
                    return True

            # Check portfolio-level risk
            if self.current_risk_level == RiskLevel.CRITICAL:
                logger.info(
                    "Critical risk level - closing position",
                    symbol=position.symbol,
                    risk_level=self.current_risk_level.value,
                )
                return True

            return False

        except Exception as e:
            logger.error(
                "Position exit evaluation failed",
                symbol=position.symbol,
                error=str(e),
            )
            # Conservative approach: don't exit on evaluation errors
            return False
