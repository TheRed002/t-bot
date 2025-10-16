"""
Legacy Risk Manager Implementation.

This module provides backward compatibility for existing risk management implementations.
For production deployments, use the RiskService (src/risk_management/service.py) which offers:
- Enterprise-grade service architecture
- Comprehensive database integration
- Advanced state management
- Real-time monitoring capabilities
- Enhanced error handling

This implementation remains stable for existing integrations.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.core.config import Config, get_config
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

# Note: Removed ErrorHandler import to avoid dependency issues

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
    Legacy Risk Manager implementation for backward compatibility.

    This implementation provides stable risk management functionality including:
    - Position sizing strategies with multiple methods
    - Portfolio limits enforcement and monitoring
    - Comprehensive risk metrics calculation
    - Emergency controls and circuit breakers
    - Integrated error handling and recovery

    For new implementations, consider using RiskService for enhanced features.
    This class maintains API compatibility for existing integrations.
    """

    def __init__(
        self,
        config: Config | None = None,
        database_service: "DatabaseService | None" = None,
        state_service: "StateService | None" = None,
        risk_service: RiskService | None = None,
    ):
        """
        Initialize risk manager with all components.

        Args:
            config: Application configuration (uses global config if None)
            database_service: Database service for data access
            state_service: State service for state management
            risk_service: Optional injected RiskService for enhanced functionality
        """
        # Use provided config or get default config
        if config is None:
            config = get_config()

        super().__init__(config)

        # Initialize core risk management components
        self.position_sizer = PositionSizer(config)
        self.portfolio_limits = PortfolioLimits(config)
        self.risk_calculator = RiskCalculator(config)

        # Initialize RiskService integration if available
        if risk_service is not None:
            # Use injected RiskService for enhanced functionality
            self.risk_service = risk_service
            logger.info("RiskManager initialized with injected RiskService")
        elif database_service and state_service:
            # Create RiskService for enhanced features
            try:
                # RiskService uses repositories, not database_service directly
                self.risk_service = RiskService(
                    state_service=state_service, config=config
                )
                logger.info("RiskManager initialized with RiskService integration")
            except Exception as e:
                logger.error(f"Failed to initialize RiskService: {e}")
                self.risk_service = None
        else:
            self.risk_service = None
            logger.info("RiskManager initialized in standalone mode")

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

        logger.info(
            "Risk Manager initialized successfully",
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

        Args:
            signal: Trading signal
            available_capital: Available capital for position
            current_price: Current market price

        Returns:
            Position size in base currency
        """
        try:
            # Validate signal first
            is_valid = await self.validate_signal(signal)
            if not is_valid:
                raise RiskManagementError(f"Signal validation failed for {signal.symbol}")

            # Use RiskService if available for enhanced functionality
            if self.risk_service is not None:
                return await self.risk_service.calculate_position_size(
                    signal=signal,
                    available_capital=available_capital,
                    current_price=current_price,
                )

            # Fallback to direct implementation

            # Calculate base position size
            position_size = await self.position_sizer.calculate_position_size(
                signal=signal,
                portfolio_value=available_capital,  # Using available capital as portfolio proxy
                method=PositionSizeMethod(self.config.risk.position_sizing_method),
            )

            # Apply portfolio limits
            # (manual implementation since PortfolioLimits is a Pydantic model)
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

        Args:
            signal: Signal to validate

        Returns:
            True if signal passes risk validation
        """
        try:
            # Use RiskService if available for enhanced validation
            if self.risk_service is not None:
                return await self.risk_service.validate_signal(signal)

            # Direct implementation fallback
            # Check confidence threshold using configurable minimum signal strength
            min_signal_strength = getattr(self.config.risk, "min_signal_strength", 0.3)
            if signal.strength < min_signal_strength:
                logger.warning(
                    "Signal strength too low",
                    symbol=signal.symbol,
                    strength=signal.strength,
                    min_required=min_signal_strength,
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
            if len(symbol_positions) >= self.config.risk.max_positions_per_symbol:
                logger.warning(
                    "Max positions per symbol reached",
                    symbol=signal.symbol,
                    current_positions=len(symbol_positions),
                    max_allowed=self.config.risk.max_positions_per_symbol,
                )
                return False

            logger.info(
                "Signal validated",
                symbol=signal.symbol,
                direction=signal.direction.value,
                strength=float(signal.strength),
            )

            return True

        except Exception as e:
            logger.error(
                "Signal validation failed",
                symbol=signal.symbol,
                error=str(e),
            )
            raise

    @time_execution
    async def validate_order(self, order: OrderRequest) -> bool:
        """
        Validate order against risk constraints.

        Args:
            order: Order to validate

        Returns:
            True if order passes risk validation
        """
        try:
            # Use RiskService if available for enhanced validation
            if self.risk_service is not None:
                return await self.risk_service.validate_order(order)

            # Direct implementation fallback
            # Check order size limits
            if order.quantity > self.position_limits.max_position_size:
                logger.warning(
                    "Order size exceeds limit",
                    symbol=order.symbol,
                    order_size=format_decimal(order.quantity),
                    max_size=format_decimal(self.position_limits.max_position_size),
                )
                return False

            order_value = (
                order.quantity * order.price
                if hasattr(order, "price") and order.price is not None
                else order.quantity
            )
            potential_exposure = self.total_exposure + order_value
            max_portfolio_exposure = (
                to_decimal(getattr(self.config.risk, "available_capital", ZERO))
                if getattr(self.config.risk, "available_capital", None)
                else to_decimal(100000)
            )
            if potential_exposure > max_portfolio_exposure:
                logger.warning(
                    "Order would exceed portfolio exposure limit",
                    current_exposure=format_decimal(self.total_exposure),
                    additional_exposure=format_decimal(order.quantity),
                    max_exposure=format_decimal(max_portfolio_exposure),
                )
                return False

            logger.info(
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
            raise

    @time_execution
    async def calculate_risk_metrics(
        self,
        positions: list[Position],
        market_data: list[MarketData],
        returns: list[float] | None = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for positions.

        Args:
            positions: Current positions
            market_data: Current market data
            returns: Historical returns (optional)

        Returns:
            Calculated risk metrics
        """
        try:
            # Use RiskService if available for enhanced metrics
            if self.risk_service is not None:
                return await self.risk_service.calculate_risk_metrics(positions, market_data)

            # Direct implementation fallback

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
            raise RiskManagementError(f"Risk metrics calculation failed: {e}") from e

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

            logger.info(
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
            max_portfolio_exposure = (
                to_decimal(getattr(self.config.risk, "available_capital", ZERO))
                if getattr(self.config.risk, "available_capital", None)
                else to_decimal(100000)
            )
            if self.total_exposure > max_portfolio_exposure:
                message = (
                    f"Portfolio exposure {format_decimal(self.total_exposure)} "
                    f"exceeds limit {format_decimal(max_portfolio_exposure)}"
                )
                logger.warning(message)
                return False, message

            # Check position count
            total_positions = sum(len(pos) for pos in self.active_positions.values())
            if total_positions > self.position_limits.max_positions:
                message = (
                    f"Total positions {total_positions} "
                    f"exceeds limit {self.position_limits.max_positions}"
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

            logger.info(
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
            # Check if available_capital field exists in config
            available_capital = getattr(self.config.risk, "available_capital", None)
            if not available_capital or available_capital == ZERO:
                return ZERO

            leverage = safe_divide(
                self.total_exposure,
                to_decimal(available_capital),
            )

            logger.info(
                "Leverage calculated",
                total_exposure=format_decimal(self.total_exposure),
                available_capital=format_decimal(to_decimal(available_capital)),
                leverage=format_decimal(leverage),
            )

            return leverage

        except Exception as e:
            logger.error("Leverage calculation failed", error=str(e))
            return ZERO

    # Additional helper methods for compatibility
    def _calculate_signal_score(self, signal: Signal) -> Decimal:
        """Calculate signal score for position sizing."""
        return (
            signal.strength
            if isinstance(signal.strength, Decimal)
            else Decimal(str(signal.strength))
        )

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

        Args:
            new_position: Position to be added

        Returns:
            bool: True if position addition is allowed
        """
        try:
            # Use RiskService if available for enhanced validation
            if self.risk_service is not None:
                # Convert position to order for validation
                order = OrderRequest(
                    symbol=new_position.symbol,
                    side=new_position.side,
                    order_type=OrderType.MARKET,
                    quantity=new_position.quantity,
                )
                return await self.risk_service.validate_order(order)

            # Direct implementation fallback

            # Check if adding position would exceed portfolio limits
            position_value = new_position.quantity * new_position.current_price
            potential_exposure = self.total_exposure + position_value

            max_portfolio_exposure = (
                to_decimal(getattr(self.config.risk, "available_capital", ZERO))
                if getattr(self.config.risk, "available_capital", None)
                else to_decimal(100000)
            )
            if potential_exposure > max_portfolio_exposure:
                logger.warning(
                    "Adding position would exceed portfolio exposure limit",
                    current_exposure=format_decimal(self.total_exposure),
                    position_value=format_decimal(position_value),
                    max_exposure=format_decimal(max_portfolio_exposure),
                )
                return False

            # Check position count limits
            total_positions = sum(len(pos) for pos in self.active_positions.values())
            if total_positions >= self.position_limits.max_positions:
                logger.warning(
                    "Adding position would exceed total position limit",
                    current_positions=total_positions,
                    max_positions=self.position_limits.max_positions,
                )
                return False

            # Check symbol-specific limits
            symbol_positions = self.active_positions.get(new_position.symbol, [])
            if len(symbol_positions) >= self.config.risk.max_positions_per_symbol:
                logger.warning(
                    "Adding position would exceed symbol position limit",
                    symbol=new_position.symbol,
                    current_positions=len(symbol_positions),
                    max_positions=self.config.risk.max_positions_per_symbol,
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

        Args:
            position: Position to evaluate
            market_data: Current market data for the position

        Returns:
            bool: True if position should be closed
        """
        try:
            # Use RiskService if available for enhanced exit logic
            if self.risk_service is not None:
                return await self.risk_service.should_exit_position(position, market_data)

            # Direct implementation fallback

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

    @time_execution
    async def get_comprehensive_risk_summary(self) -> dict[str, Any]:
        """
        Get comprehensive risk summary including all components.

        Returns:
            dict: Comprehensive risk summary
        """
        try:
            # Use RiskService if available for enhanced summary
            if self.risk_service is not None:
                return await self.risk_service.get_comprehensive_summary()

            # Direct implementation fallback

            # Aggregate summaries from all components
            summary: dict[str, Any] = {
                "risk_level": self.current_risk_level.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_exposure": format_decimal(self.total_exposure),
                "active_positions_count": sum(
                    len(positions) for positions in self.active_positions.values()
                ),
                "unique_symbols": len(self.active_positions),
            }

            # Add portfolio limits information
            summary["portfolio_limits"] = {
                "max_position_size": format_decimal(self.position_limits.max_position_size),
                "max_positions": self.position_limits.max_positions,
                "max_leverage": format_decimal(self.position_limits.max_leverage),
                "min_position_size": format_decimal(self.position_limits.min_position_size),
                # Add optional fields if they exist
                "max_position_value": (
                    format_decimal(self.position_limits.max_position_value)
                    if self.position_limits.max_position_value
                    else None
                ),
                "max_daily_loss": (
                    format_decimal(self.position_limits.max_daily_loss)
                    if self.position_limits.max_daily_loss
                    else None
                ),
                "concentration_limit": self.position_limits.concentration_limit,
            }

            try:
                if hasattr(self, "risk_calculator") and self.risk_calculator:
                    if hasattr(self.risk_calculator, "get_risk_summary"):
                        risk_calc_summary = await self.risk_calculator.get_risk_summary()
                    else:
                        risk_calc_summary = {}
                    summary["risk_calculator"] = risk_calc_summary
            except Exception as e:
                logger.warning(f"Could not get risk calculator summary: {e}")
                summary["risk_calculator"] = {"error": "Risk calculator summary unavailable"}

            # Add position sizer information
            summary["position_sizer_methods"] = {
                "current_method": self.config.risk.position_sizing_method,
                "available_methods": ["fixed", "kelly", "volatility_adjusted", "risk_parity"],
            }

            # Add risk configuration
            summary["risk_config"] = {
                "max_position_size_pct": self.config.risk.max_position_size,
                "max_total_positions": self.config.risk.max_total_positions,
                "max_positions_per_symbol": self.config.risk.max_positions_per_symbol,
                "position_sizing_method": self.config.risk.position_sizing_method,
            }

            # Add current leverage
            current_leverage = self.calculate_leverage()
            summary["current_leverage"] = format_decimal(current_leverage)

            logger.info(
                "Comprehensive risk summary generated",
                risk_level=summary["risk_level"],
                total_exposure=summary["total_exposure"],
                active_positions=summary["active_positions_count"],
            )

            return summary

        except Exception as e:
            logger.error("Comprehensive risk summary generation failed", error=str(e))
            # Return minimal summary on error
            return {
                "error": "Risk summary generation failed",
                "risk_level": "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": str(e),
            }
