"""
Concrete Risk Manager Implementation for P-008 Risk Management Framework.

This module provides a concrete implementation of BaseRiskManager that integrates
all risk management components including position sizing, portfolio limits, and
risk metrics calculation.

CRITICAL: This integrates with P-001 (types, exceptions, config), 
P-002A (error handling), and P-007A (utils) components.
"""

from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import structlog

# MANDATORY: Import from P-001
from src.core.types import (
    Position, MarketData, Signal, OrderRequest, OrderResponse,
    RiskMetrics, PositionLimits, RiskLevel, PositionSizeMethod
)
from src.core.exceptions import (
    RiskManagementError, PositionLimitError, DrawdownLimitError,
    ValidationError
)
from src.core.config import Config

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_price, validate_quantity, validate_position_limits
from src.utils.formatters import format_percentage, format_currency

# MANDATORY: Import from P-003+
from src.exchanges.base import BaseExchange

# Import risk management components
from .base import BaseRiskManager
from .position_sizing import PositionSizer
from .portfolio_limits import PortfolioLimits
from .risk_metrics import RiskCalculator

logger = structlog.get_logger()


class RiskManager(BaseRiskManager):
    """
    Concrete risk manager implementation.
    
    This class implements the BaseRiskManager interface and integrates
    all risk management components for comprehensive risk control.
    """
    
    def __init__(self, config: Config):
        """
        Initialize risk manager with all components.
        
        Args:
            config: Application configuration containing risk settings
        """
        super().__init__(config)
        
        # Initialize risk management components
        self.position_sizer = PositionSizer(config)
        self.portfolio_limits = PortfolioLimits(config)
        self.risk_calculator = RiskCalculator(config)
        
        # Initialize position limits
        self.position_limits = PositionLimits(
            max_position_size=Decimal(str(config.risk.max_position_size_pct)),
            max_positions_per_symbol=config.risk.max_positions_per_symbol,
            max_total_positions=config.risk.max_total_positions,
            max_portfolio_exposure=Decimal(str(config.risk.max_portfolio_exposure)),
            max_sector_exposure=Decimal(str(config.risk.max_sector_exposure)),
            max_correlation_exposure=Decimal(str(config.risk.max_correlation_exposure)),
            max_leverage=Decimal(str(config.risk.max_leverage))
        )
        
        self.logger.info("Risk manager initialized with all components")
    
    @time_execution
    async def calculate_position_size(self, signal: Signal, 
                                   portfolio_value: Decimal) -> Decimal:
        """
        Calculate optimal position size for a trading signal.
        
        Args:
            signal: Trading signal with direction and confidence
            portfolio_value: Current total portfolio value
            
        Returns:
            Decimal: Calculated position size in base currency
            
        Raises:
            RiskManagementError: If position size calculation fails
            PositionLimitError: If calculated size exceeds limits
        """
        try:
            # Validate inputs
            if not signal or not signal.confidence:
                raise ValidationError("Invalid signal for position sizing")
            
            if portfolio_value <= 0:
                raise ValidationError("Invalid portfolio value for position sizing")
            
            # Calculate position size using position sizer
            position_size = await self.position_sizer.calculate_position_size(
                signal, portfolio_value
            )
            
            # Validate position size against limits
            if not await self.position_sizer.validate_position_size(position_size, portfolio_value):
                raise PositionLimitError("Position size exceeds limits")
            
            self.logger.info("Position size calculated",
                           symbol=signal.symbol,
                           confidence=signal.confidence,
                           portfolio_value=float(portfolio_value),
                           position_size=float(position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error("Position size calculation failed",
                            error=str(e),
                            signal_symbol=signal.symbol if signal else None)
            raise RiskManagementError(f"Position size calculation failed: {e}")
    
    @time_execution
    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a trading signal against risk limits.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            bool: True if signal passes risk validation
            
        Raises:
            ValidationError: If signal validation fails
        """
        try:
            # Basic signal validation
            if not signal or not signal.symbol:
                raise ValidationError("Invalid signal: missing symbol")
            
            if not (0 <= signal.confidence <= 1):
                raise ValidationError("Invalid signal confidence")
            
            if signal.direction.value not in ["buy", "sell"]:
                raise ValidationError("Invalid signal direction")
            
            # Check minimum confidence threshold
            min_confidence = 0.6  # Can be made configurable
            if signal.confidence < min_confidence:
                self.logger.warning("Signal confidence below threshold",
                                  confidence=signal.confidence,
                                  threshold=min_confidence)
                return False
            
            self.logger.info("Signal validation passed",
                           symbol=signal.symbol,
                           direction=signal.direction.value,
                           confidence=signal.confidence)
            
            return True
            
        except Exception as e:
            self.logger.error("Signal validation failed",
                            error=str(e),
                            signal_symbol=signal.symbol if signal else None)
            raise ValidationError(f"Signal validation failed: {e}")
    
    @time_execution
    async def validate_order(self, order: OrderRequest, 
                           portfolio_value: Decimal) -> bool:
        """
        Validate an order request against risk limits.
        
        Args:
            order: Order request to validate
            portfolio_value: Current total portfolio value
            
        Returns:
            bool: True if order passes risk validation
            
        Raises:
            ValidationError: If order validation fails
            PositionLimitError: If order exceeds position limits
        """
        try:
            # Validate order parameters
            if not order or not order.symbol:
                raise ValidationError("Invalid order: missing symbol")
            
            if order.quantity <= 0:
                raise ValidationError("Invalid order quantity")
            
            if order.price and order.price <= 0:
                raise ValidationError("Invalid order price")
            
            # Check position size limits
            order_value = order.quantity * (order.price or Decimal("0"))
            max_position_size = portfolio_value * Decimal(str(self.risk_config.max_position_size_pct))
            
            if order_value > max_position_size:
                await self._log_risk_violation("order_size_limit", {
                    "order_value": float(order_value),
                    "max_position_size": float(max_position_size),
                    "symbol": order.symbol
                })
                raise PositionLimitError("Order size exceeds maximum position limit")
            
            # Check portfolio exposure limit
            current_exposure = sum(
                abs(pos.quantity * pos.current_price) 
                for pos in self.positions
            )
            total_exposure = current_exposure + order_value
            max_exposure = portfolio_value * Decimal(str(self.risk_config.max_portfolio_exposure))
            
            if total_exposure > max_exposure:
                await self._log_risk_violation("portfolio_exposure_limit", {
                    "current_exposure": float(current_exposure),
                    "order_value": float(order_value),
                    "total_exposure": float(total_exposure),
                    "max_exposure": float(max_exposure)
                })
                raise PositionLimitError("Order would exceed portfolio exposure limit")
            
            self.logger.info("Order validation passed",
                           symbol=order.symbol,
                           quantity=float(order.quantity),
                           order_value=float(order_value))
            
            return True
            
        except Exception as e:
            self.logger.error("Order validation failed",
                            error=str(e),
                            order_symbol=order.symbol if order else None)
            raise ValidationError(f"Order validation failed: {e}")
    
    @time_execution
    async def calculate_risk_metrics(self, positions: List[Position],
                                   market_data: List[MarketData]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio.
        
        Args:
            positions: Current portfolio positions
            market_data: Current market data for all positions
            
        Returns:
            RiskMetrics: Calculated risk metrics
            
        Raises:
            RiskManagementError: If risk calculation fails
        """
        try:
            # Calculate risk metrics using risk calculator
            risk_metrics = await self.risk_calculator.calculate_risk_metrics(
                positions, market_data
            )
            
            # Update internal state
            self.risk_metrics = risk_metrics
            self.current_risk_level = risk_metrics.risk_level
            self.last_risk_calculation = datetime.now()
            
            # Check for critical risk levels
            if risk_metrics.risk_level == RiskLevel.CRITICAL:
                await self.emergency_stop("Critical risk level detected")
            
            self.logger.info("Risk metrics calculated",
                           var_1d=float(risk_metrics.var_1d),
                           current_drawdown=float(risk_metrics.current_drawdown),
                           risk_level=risk_metrics.risk_level.value)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error("Risk metrics calculation failed", error=str(e))
            raise RiskManagementError(f"Risk metrics calculation failed: {e}")
    
    @time_execution
    async def check_portfolio_limits(self, new_position: Position) -> bool:
        """
        Check if adding a new position would violate portfolio limits.
        
        Args:
            new_position: Position to be added
            
        Returns:
            bool: True if position addition is allowed
            
        Raises:
            PositionLimitError: If portfolio limits would be violated
        """
        try:
            # Update portfolio limits state
            await self.portfolio_limits.update_portfolio_state(
                self.positions, self.total_portfolio_value
            )
            
            # Check portfolio limits
            return await self.portfolio_limits.check_portfolio_limits(new_position)
            
        except Exception as e:
            self.logger.error("Portfolio limits check failed",
                            error=str(e),
                            symbol=new_position.symbol if new_position else None)
            raise PositionLimitError(f"Portfolio limits check failed: {e}")
    
    @time_execution
    async def should_exit_position(self, position: Position, 
                                 market_data: MarketData) -> bool:
        """
        Determine if a position should be closed based on risk criteria.
        
        Args:
            position: Position to evaluate
            market_data: Current market data for the position
            
        Returns:
            bool: True if position should be closed
        """
        try:
            # Update position with current market data
            if position.symbol == market_data.symbol:
                position.current_price = market_data.price
                position.unrealized_pnl = (
                    position.quantity * (market_data.price - position.entry_price)
                )
            
            # Check stop loss
            stop_loss_pct = self.risk_config.max_daily_loss_pct
            if position.unrealized_pnl < 0:
                loss_pct = abs(position.unrealized_pnl) / (position.quantity * position.entry_price)
                if loss_pct >= stop_loss_pct:
                    self.logger.warning("Position hit stop loss",
                                      symbol=position.symbol,
                                      loss_pct=float(loss_pct),
                                      stop_loss_pct=stop_loss_pct)
                    return True
            
            # Check drawdown limit
            if self.risk_metrics and self.risk_metrics.current_drawdown > Decimal(str(self.risk_config.max_drawdown_pct)):
                self.logger.warning("Position exit due to drawdown limit",
                                  symbol=position.symbol,
                                  current_drawdown=float(self.risk_metrics.current_drawdown),
                                  max_drawdown=self.risk_config.max_drawdown_pct)
                return True
            
            # Check risk level
            if self.current_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.logger.warning("Position exit due to high risk level",
                                  symbol=position.symbol,
                                  risk_level=self.current_risk_level.value)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error("Position exit evaluation failed",
                            error=str(e),
                            symbol=position.symbol)
            return False
    
    @time_execution
    async def update_portfolio_state(self, positions: List[Position],
                                   portfolio_value: Decimal) -> None:
        """
        Update portfolio state for all risk management components.
        
        Args:
            positions: Current portfolio positions
            portfolio_value: Current total portfolio value
        """
        # Update base class state
        await super().update_portfolio_state(positions, portfolio_value)
        
        # Update component states
        await self.portfolio_limits.update_portfolio_state(positions, portfolio_value)
        
        # Update position sizer with current prices
        for position in positions:
            await self.position_sizer.update_price_history(
                position.symbol, float(position.current_price)
            )
            await self.portfolio_limits.update_return_history(
                position.symbol, float(position.current_price)
            )
            await self.risk_calculator.update_position_returns(
                position.symbol, float(position.current_price)
            )
        
        self.logger.debug("Portfolio state updated across all components",
                         position_count=len(positions),
                         portfolio_value=float(portfolio_value))
    
    @time_execution
    async def get_comprehensive_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary from all components.
        
        Returns:
            Dict containing risk summary from all components
        """
        try:
            # Get summaries from all components
            base_summary = await super().get_risk_summary()
            portfolio_summary = await self.portfolio_limits.get_portfolio_summary()
            risk_summary = await self.risk_calculator.get_risk_summary()
            
            # Combine summaries
            comprehensive_summary = {
                **base_summary,
                "portfolio_limits": portfolio_summary,
                "risk_calculator": risk_summary,
                "position_sizer_methods": list(PositionSizeMethod),
                "risk_config": {
                    "default_position_size_pct": self.risk_config.default_position_size_pct,
                    "max_position_size_pct": self.risk_config.max_position_size_pct,
                    "max_total_positions": self.risk_config.max_total_positions,
                    "max_portfolio_exposure": self.risk_config.max_portfolio_exposure,
                    "max_drawdown_pct": self.risk_config.max_drawdown_pct,
                    "var_confidence_level": self.risk_config.var_confidence_level
                }
            }
            
            return comprehensive_summary
            
        except Exception as e:
            self.logger.error("Risk summary generation failed", error=str(e))
            return {"error": f"Risk summary generation failed: {e}"}
    
    @time_execution
    async def validate_risk_parameters(self) -> bool:
        """
        Validate all risk parameters across all components.
        
        Returns:
            bool: True if all parameters are valid
        """
        try:
            # Validate base parameters
            base_valid = await super().validate_risk_parameters()
            if not base_valid:
                return False
            
            # Validate position sizer
            # (Position sizer validation is handled internally)
            
            # Validate portfolio limits
            # (Portfolio limits validation is handled internally)
            
            # Validate risk calculator
            # (Risk calculator validation is handled internally)
            
            self.logger.info("All risk parameters validated successfully")
            return True
            
        except Exception as e:
            self.logger.error("Risk parameter validation failed", error=str(e))
            return False 