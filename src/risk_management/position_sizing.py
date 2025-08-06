"""
Position Sizing Module for P-008 Risk Management Framework.

This module implements various position sizing algorithms including:
- Fixed percentage sizing
- Kelly Criterion optimal sizing
- Volatility-adjusted sizing using ATR
- Confidence-weighted sizing for ML strategies

CRITICAL: This integrates with P-001 (types, exceptions, config), 
P-002A (error handling), and P-007A (utils) components.
"""

from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np

# MANDATORY: Import from P-001
from src.core.types import (
    Signal, Position, MarketData, PositionSizeMethod, RiskLevel
)
from src.core.exceptions import (
    RiskManagementError, PositionLimitError, ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_price, validate_quantity, validate_position_limits
from src.utils.formatters import format_percentage, format_currency

logger = get_logger(__name__)


class PositionSizer:
    """
    Position sizing calculator with multiple algorithms.
    
    This class implements various position sizing methods to optimize
    risk-adjusted returns while respecting portfolio limits.
    """
    
    def __init__(self, config: Config):
        """
        Initialize position sizer with configuration.
        
        Args:
            config: Application configuration containing risk settings
        """
        self.config = config
        self.risk_config = config.risk
        self.error_handler = ErrorHandler(config)
        self.logger = logger.bind(component="position_sizer")
        
        # Historical data for calculations
        self.price_history: Dict[str, List[float]] = {}
        self.return_history: Dict[str, List[float]] = {}
        
        self.logger.info("Position sizer initialized")
    
    @time_execution
    async def calculate_position_size(self, signal: Signal, 
                                   portfolio_value: Decimal,
                                   method: PositionSizeMethod = None) -> Decimal:
        """
        Calculate position size using specified method.
        
        Args:
            signal: Trading signal with direction and confidence
            portfolio_value: Current total portfolio value
            method: Position sizing method to use (defaults to config setting)
            
        Returns:
            Decimal: Calculated position size in base currency
            
        Raises:
            RiskManagementError: If position size calculation fails
            PositionLimitError: If calculated size exceeds limits
        """
        try:
            # Use default method if not specified
            if method is None:
                method = PositionSizeMethod(self.risk_config.default_position_size_method)
            
            # Validate inputs
            if not signal or not signal.confidence:
                raise ValidationError("Invalid signal for position sizing")
            
            if portfolio_value <= 0:
                raise ValidationError("Invalid portfolio value for position sizing")
            
            # Calculate position size based on method
            if method == PositionSizeMethod.FIXED_PCT:
                position_size = await self._fixed_percentage_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.KELLY_CRITERION:
                position_size = await self._kelly_criterion_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                position_size = await self._volatility_adjusted_sizing(signal, portfolio_value)
            elif method == PositionSizeMethod.CONFIDENCE_WEIGHTED:
                position_size = await self._confidence_weighted_sizing(signal, portfolio_value)
            else:
                raise ValidationError(f"Unsupported position sizing method: {method}")
            
            # Apply maximum position size limit
            max_position_size = portfolio_value * Decimal(str(self.risk_config.max_position_size_pct))
            if position_size > max_position_size:
                self.logger.warning("Position size exceeds maximum limit, capping",
                                  calculated_size=float(position_size),
                                  max_size=float(max_position_size))
                position_size = max_position_size
            
            # Validate minimum position size
            min_position_size = portfolio_value * Decimal("0.001")  # 0.1% minimum
            if position_size < min_position_size:
                self.logger.warning("Position size below minimum, skipping",
                                  calculated_size=float(position_size),
                                  min_size=float(min_position_size))
                return Decimal("0")
            
            self.logger.info("Position size calculated",
                           method=method.value,
                           signal_symbol=signal.symbol,
                           signal_confidence=signal.confidence,
                           portfolio_value=float(portfolio_value),
                           position_size=float(position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error("Position size calculation failed",
                            error=str(e),
                            signal_symbol=signal.symbol if signal else None)
            raise RiskManagementError(f"Position size calculation failed: {e}")
    
    @time_execution
    async def _fixed_percentage_sizing(self, signal: Signal, 
                                     portfolio_value: Decimal) -> Decimal:
        """
        Calculate position size using fixed percentage method.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Decimal: Position size as fixed percentage of portfolio
        """
        # Base position size as percentage of portfolio
        base_size = portfolio_value * Decimal(str(self.risk_config.default_position_size_pct))
        
        # Adjust for signal confidence
        confidence_multiplier = Decimal(str(signal.confidence))
        position_size = base_size * confidence_multiplier
        
        self.logger.debug("Fixed percentage sizing",
                         base_size=float(base_size),
                         confidence_multiplier=float(confidence_multiplier),
                         final_size=float(position_size))
        
        return position_size
    
    @time_execution
    async def _kelly_criterion_sizing(self, signal: Signal, 
                                    portfolio_value: Decimal) -> Decimal:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Decimal: Position size using Kelly Criterion
        """
        try:
            # Get historical returns for the symbol
            symbol = signal.symbol
            returns = self.return_history.get(symbol, [])
            
            if len(returns) < self.risk_config.kelly_lookback_days:
                self.logger.warning("Insufficient data for Kelly Criterion, using fixed percentage",
                                  symbol=symbol,
                                  available_data=len(returns),
                                  required_data=self.risk_config.kelly_lookback_days)
                return await self._fixed_percentage_sizing(signal, portfolio_value)
            
            # Calculate Kelly Criterion parameters
            returns_array = np.array(returns[-self.risk_config.kelly_lookback_days:])
            mean_return = np.mean(returns_array)
            variance = np.var(returns_array)
            
            if variance == 0:
                self.logger.warning("Zero variance in returns, using fixed percentage")
                return await self._fixed_percentage_sizing(signal, portfolio_value)
            
            # Kelly fraction = (mean_return) / variance
            kelly_fraction = mean_return / variance
            
            # Apply maximum Kelly fraction limit
            max_kelly = self.risk_config.kelly_max_fraction
            kelly_fraction = min(kelly_fraction, max_kelly)
            
            # Apply confidence adjustment
            kelly_fraction *= signal.confidence
            
            # Calculate position size
            position_size = portfolio_value * Decimal(str(kelly_fraction))
            
            self.logger.debug("Kelly Criterion sizing",
                             mean_return=float(mean_return),
                             variance=float(variance),
                             kelly_fraction=float(kelly_fraction),
                             position_size=float(position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error("Kelly Criterion calculation failed", error=str(e))
            # Fallback to fixed percentage
            return await self._fixed_percentage_sizing(signal, portfolio_value)
    
    @time_execution
    async def _volatility_adjusted_sizing(self, signal: Signal, 
                                        portfolio_value: Decimal) -> Decimal:
        """
        Calculate position size using volatility-adjusted method.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Decimal: Position size adjusted for volatility
        """
        try:
            symbol = signal.symbol
            prices = self.price_history.get(symbol, [])
            
            if len(prices) < self.risk_config.volatility_window:
                self.logger.warning("Insufficient data for volatility adjustment, using fixed percentage",
                                  symbol=symbol,
                                  available_data=len(prices),
                                  required_data=self.risk_config.volatility_window)
                return await self._fixed_percentage_sizing(signal, portfolio_value)
            
            # Calculate volatility (standard deviation of returns)
            prices_array = np.array(prices[-self.risk_config.volatility_window:])
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility = np.std(returns)
            
            # Calculate volatility adjustment factor
            target_volatility = self.risk_config.volatility_target
            volatility_adjustment = target_volatility / max(volatility, 0.001)  # Avoid division by zero
            
            # Cap volatility adjustment to reasonable bounds
            volatility_adjustment = max(0.1, min(volatility_adjustment, 5.0))
            
            # Base position size
            base_size = portfolio_value * Decimal(str(self.risk_config.default_position_size_pct))
            
            # Apply volatility adjustment and confidence
            position_size = base_size * Decimal(str(volatility_adjustment)) * Decimal(str(signal.confidence))
            
            self.logger.debug("Volatility-adjusted sizing",
                             volatility=float(volatility),
                             target_volatility=float(target_volatility),
                             adjustment=float(volatility_adjustment),
                             position_size=float(position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error("Volatility adjustment calculation failed", error=str(e))
            # Fallback to fixed percentage
            return await self._fixed_percentage_sizing(signal, portfolio_value)
    
    @time_execution
    async def _confidence_weighted_sizing(self, signal: Signal, 
                                        portfolio_value: Decimal) -> Decimal:
        """
        Calculate position size using confidence-weighted method for ML strategies.
        
        Args:
            signal: Trading signal with confidence from ML model
            portfolio_value: Current portfolio value
            
        Returns:
            Decimal: Position size weighted by ML confidence
        """
        # Base position size
        base_size = portfolio_value * Decimal(str(self.risk_config.default_position_size_pct))
        
        # Apply confidence weighting with non-linear scaling
        # Higher confidence gets proportionally larger position
        confidence = signal.confidence
        confidence_weight = confidence ** 2  # Square for non-linear scaling
        
        position_size = base_size * Decimal(str(confidence_weight))
        
        self.logger.debug("Confidence-weighted sizing",
                         confidence=float(confidence),
                         confidence_weight=float(confidence_weight),
                         position_size=float(position_size))
        
        return position_size
    
    @time_execution
    async def update_price_history(self, symbol: str, price: float) -> None:
        """
        Update price history for volatility calculations.
        
        Args:
            symbol: Trading symbol
            price: Current price
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only recent history to manage memory
        max_history = max(self.risk_config.volatility_window * 2, 100)
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        
        # Calculate and store returns
        if len(self.price_history[symbol]) > 1:
            if symbol not in self.return_history:
                self.return_history[symbol] = []
            
            prev_price = self.price_history[symbol][-2]
            if prev_price > 0:
                return_rate = (price - prev_price) / prev_price
                self.return_history[symbol].append(return_rate)
                
                # Keep only recent returns
                if len(self.return_history[symbol]) > max_history:
                    self.return_history[symbol] = self.return_history[symbol][-max_history:]
    
    @time_execution
    async def get_position_size_summary(self, signal: Signal, 
                                      portfolio_value: Decimal) -> Dict[str, Any]:
        """
        Get comprehensive position size summary for all methods.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Dict containing position sizes for all methods
        """
        summary = {}
        
        for method in PositionSizeMethod:
            try:
                size = await self.calculate_position_size(signal, portfolio_value, method)
                summary[method.value] = {
                    "position_size": float(size),
                    "portfolio_percentage": float(size / portfolio_value) if portfolio_value > 0 else 0
                }
            except Exception as e:
                summary[method.value] = {
                    "error": str(e),
                    "position_size": 0,
                    "portfolio_percentage": 0
                }
        
        return summary
    
    @time_execution
    async def validate_position_size(self, position_size: Decimal, 
                                  portfolio_value: Decimal) -> bool:
        """
        Validate calculated position size against limits.
        
        Args:
            position_size: Calculated position size
            portfolio_value: Current portfolio value
            
        Returns:
            bool: True if position size is valid
        """
        try:
            # Check minimum position size (0.1% of portfolio)
            min_size = portfolio_value * Decimal("0.001")
            if position_size < min_size:
                self.logger.warning("Position size below minimum",
                                  position_size=float(position_size),
                                  min_size=float(min_size))
                return False
            
            # Check maximum position size
            max_size = portfolio_value * Decimal(str(self.risk_config.max_position_size_pct))
            if position_size > max_size:
                self.logger.warning("Position size exceeds maximum",
                                  position_size=float(position_size),
                                  max_size=float(max_size))
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Position size validation failed", error=str(e))
            return False 