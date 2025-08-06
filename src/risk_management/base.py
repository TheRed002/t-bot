"""
Base Risk Manager for P-008 Risk Management Framework.

This module provides the abstract base class for all risk management implementations.
It defines the core interface that all risk managers must implement.

CRITICAL: This integrates with P-001 (types, exceptions, config), 
P-002A (error handling), and P-007A (utils) components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio

# MANDATORY: Import from P-001
from src.core.types import (
    Position, MarketData, Signal, OrderRequest,
    RiskMetrics, PositionLimits, RiskLevel, PositionSizeMethod
)
from src.core.exceptions import (
    RiskManagementError, PositionLimitError, DrawdownLimitError,
    ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_price, validate_quantity, validate_position_limits
from src.utils.formatters import format_percentage, format_currency

# MANDATORY: Import from P-003+
from src.exchanges.base import BaseExchange

logger = get_logger(__name__)


class BaseRiskManager(ABC):
    """
    Abstract base class for risk management implementations.
    
    This class defines the core interface that all risk managers must implement.
    It provides position sizing, portfolio monitoring, and risk limit enforcement.
    
    CRITICAL: All implementations must follow the exact interface defined here.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the risk manager with configuration.
        
        Args:
            config: Application configuration containing risk settings
        """
        self.config = config
        self.risk_config = config.risk
        self.error_handler = ErrorHandler(config)
        self.logger = logger.bind(component="risk_manager")
        
        # Risk state tracking
        self.current_risk_level = RiskLevel.LOW
        self.last_risk_calculation = datetime.now()
        self.risk_metrics: Optional[RiskMetrics] = None
        self.position_limits: Optional[PositionLimits] = None
        
        # Portfolio state
        self.positions: List[Position] = []
        self.total_portfolio_value = Decimal("0")
        self.current_drawdown = Decimal("0")
        self.max_drawdown = Decimal("0")
        
        self.logger.info("Risk manager initialized", 
                        risk_config=dict(self.risk_config))
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    # Standard methods that can be overridden
    @time_execution
    async def update_portfolio_state(self, positions: List[Position],
                                   portfolio_value: Decimal) -> None:
        """
        Update internal portfolio state for risk calculations.
        
        Args:
            positions: Current portfolio positions
            portfolio_value: Current total portfolio value
        """
        self.positions = positions
        self.total_portfolio_value = portfolio_value
        
        # Calculate current drawdown
        if self.risk_metrics:
            self.current_drawdown = self.risk_metrics.current_drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        self.logger.debug("Portfolio state updated", 
                         position_count=len(positions),
                         portfolio_value=float(portfolio_value))
    
    @time_execution
    async def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary for monitoring and reporting.
        
        Returns:
            Dict containing current risk state and metrics
        """
        summary = {
            "risk_level": self.current_risk_level.value,
            "total_positions": len(self.positions),
            "portfolio_value": float(self.total_portfolio_value),
            "current_drawdown": float(self.current_drawdown),
            "max_drawdown": float(self.max_drawdown),
            "last_calculation": self.last_risk_calculation.isoformat(),
            "position_limits": self.position_limits.model_dump() if self.position_limits else None,
            "risk_metrics": self.risk_metrics.model_dump() if self.risk_metrics else None
        }
        
        return summary
    
    @time_execution
    async def emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop due to risk violation.
        
        Args:
            reason: Reason for emergency stop
        """
        self.current_risk_level = RiskLevel.CRITICAL
        self.logger.critical("Emergency stop triggered", reason=reason)
        
        # Create error context and handle error
        error_context = self.error_handler.create_error_context(
            RiskManagementError(f"Emergency stop: {reason}"),
            component="risk_manager",
            operation="emergency_stop",
            severity="critical"
        )
        await self.error_handler.handle_error(
            RiskManagementError(f"Emergency stop: {reason}"),
            error_context
        )
    
    @time_execution
    async def validate_risk_parameters(self) -> bool:
        """
        Validate all risk parameters are within acceptable ranges.
        
        Returns:
            bool: True if all parameters are valid
        """
        try:
            # Validate position size parameters
            if not (0 < self.risk_config.default_position_size_pct <= 1):
                raise ValidationError("Invalid default position size percentage")
            
            if not (0 < self.risk_config.max_position_size_pct <= 1):
                raise ValidationError("Invalid max position size percentage")
            
            # Validate portfolio limits
            if not (0 < self.risk_config.max_portfolio_exposure <= 1):
                raise ValidationError("Invalid max portfolio exposure")
            
            if not (0 < self.risk_config.max_drawdown_pct <= 1):
                raise ValidationError("Invalid max drawdown percentage")
            
            # Validate Kelly Criterion parameters
            if not (0 < self.risk_config.kelly_max_fraction <= 1):
                raise ValidationError("Invalid Kelly max fraction")
            
            self.logger.info("Risk parameters validated successfully")
            return True
            
        except Exception as e:
            self.logger.error("Risk parameter validation failed", error=str(e))
            raise ValidationError(f"Risk parameter validation failed: {e}")
    
    def _calculate_portfolio_exposure(self, positions: List[Position]) -> Decimal:
        """
        Calculate total portfolio exposure as percentage of portfolio value.
        
        Args:
            positions: List of current positions
            
        Returns:
            Decimal: Portfolio exposure as decimal (0.0 to 1.0)
        """
        if not self.total_portfolio_value or self.total_portfolio_value == 0:
            return Decimal("0")
        
        total_exposure = sum(
            abs(pos.quantity * pos.current_price) 
            for pos in positions
        )
        
        return total_exposure / self.total_portfolio_value
    
    def _check_drawdown_limit(self, current_drawdown: Decimal) -> bool:
        """
        Check if current drawdown exceeds maximum allowed.
        
        Args:
            current_drawdown: Current drawdown as decimal
            
        Returns:
            bool: True if drawdown is within limits
        """
        max_drawdown = Decimal(str(self.risk_config.max_drawdown_pct))
        return current_drawdown <= max_drawdown
    
    def _check_daily_loss_limit(self, daily_pnl: Decimal) -> bool:
        """
        Check if daily loss exceeds maximum allowed.
        
        Args:
            daily_pnl: Daily P&L as decimal (negative for losses)
            
        Returns:
            bool: True if daily loss is within limits
        """
        if daily_pnl >= 0:
            return True  # No loss to check
        
        max_daily_loss = self.total_portfolio_value * Decimal(str(self.risk_config.max_daily_loss_pct))
        return abs(daily_pnl) <= max_daily_loss
    
    async def _log_risk_violation(self, violation_type: str, details: Dict[str, Any]) -> None:
        """
        Log risk violation for monitoring and alerting.
        
        Args:
            violation_type: Type of risk violation
            details: Additional violation details
        """
        self.logger.warning("Risk violation detected",
                           violation_type=violation_type,
                           details=details,
                           risk_level=self.current_risk_level.value)
        
        # TODO: Remove in production - Debug logging
        self.logger.debug("Risk violation details", 
                         violation_type=violation_type,
                         details=details) 