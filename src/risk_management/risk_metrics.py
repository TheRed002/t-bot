"""
Risk Metrics Calculator for P-008 Risk Management Framework.

This module calculates comprehensive risk metrics including:
- Value at Risk (VaR)
- Expected Shortfall (Conditional VaR)
- Maximum Drawdown
- Sharpe Ratio
- Current Drawdown
- Risk Level Assessment

CRITICAL: This integrates with P-001 (types, exceptions, config), 
P-002A (error handling), and P-007A (utils) components.
"""

from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import structlog

# MANDATORY: Import from P-001
from src.core.types import (
    Position, MarketData, RiskMetrics, RiskLevel
)
from src.core.exceptions import (
    RiskManagementError, ValidationError
)
from src.core.config import Config

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_price, validate_quantity, validate_position_limits
from src.utils.formatters import format_percentage, format_currency

logger = structlog.get_logger()


class RiskCalculator:
    """
    Risk metrics calculator for portfolio risk assessment.
    
    This class calculates comprehensive risk metrics to assess
    portfolio risk and determine appropriate risk levels.
    """
    
    def __init__(self, config: Config):
        """
        Initialize risk calculator with configuration.
        
        Args:
            config: Application configuration containing risk settings
        """
        self.config = config
        self.risk_config = config.risk
        self.error_handler = ErrorHandler(config)
        self.logger = logger.bind(component="risk_calculator")
        
        # Historical data for calculations
        self.portfolio_values: List[float] = []
        self.portfolio_returns: List[float] = []
        self.position_returns: Dict[str, List[float]] = {}
        self.position_prices: Dict[str, List[float]] = {}
        
        self.logger.info("Risk calculator initialized")
    
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
            # Validate inputs
            if not positions:
                return await self._create_empty_risk_metrics()
            
            if len(positions) != len(market_data):
                raise ValidationError("Position and market data count mismatch")
            
            # Calculate portfolio value and returns
            portfolio_value = await self._calculate_portfolio_value(positions, market_data)
            await self._update_portfolio_history(portfolio_value)
            
            # Calculate risk metrics
            var_1d = await self._calculate_var(1, portfolio_value)
            var_5d = await self._calculate_var(5, portfolio_value)
            expected_shortfall = await self._calculate_expected_shortfall(portfolio_value)
            max_drawdown = await self._calculate_max_drawdown()
            current_drawdown = await self._calculate_current_drawdown(portfolio_value)
            sharpe_ratio = await self._calculate_sharpe_ratio()
            
            # Determine risk level
            risk_level = await self._determine_risk_level(
                var_1d, current_drawdown, sharpe_ratio
            )
            
            # Create risk metrics object
            risk_metrics = RiskMetrics(
                var_1d=var_1d,
                var_5d=var_5d,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                current_drawdown=current_drawdown,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
            self.logger.info("Risk metrics calculated",
                           var_1d=float(var_1d),
                           var_5d=float(var_5d),
                           current_drawdown=float(current_drawdown),
                           risk_level=risk_level.value)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error("Risk metrics calculation failed", error=str(e))
            raise RiskManagementError(f"Risk metrics calculation failed: {e}")
    
    @time_execution
    async def _create_empty_risk_metrics(self) -> RiskMetrics:
        """
        Create empty risk metrics for portfolios with no positions.
        
        Returns:
            RiskMetrics: Empty risk metrics
        """
        return RiskMetrics(
            var_1d=Decimal("0"),
            var_5d=Decimal("0"),
            expected_shortfall=Decimal("0"),
            max_drawdown=Decimal("0"),
            sharpe_ratio=None,
            current_drawdown=Decimal("0"),
            risk_level=RiskLevel.LOW,
            timestamp=datetime.now()
        )
    
    @time_execution
    async def _calculate_portfolio_value(self, positions: List[Position],
                                       market_data: List[MarketData]) -> Decimal:
        """
        Calculate current portfolio value.
        
        Args:
            positions: Current portfolio positions
            market_data: Current market data for positions
            
        Returns:
            Decimal: Current portfolio value
        """
        portfolio_value = Decimal("0")
        
        for position, market in zip(positions, market_data):
            if position.symbol == market.symbol:
                # Update position with current price
                position.current_price = market.price
                position.unrealized_pnl = (
                    position.quantity * (market.price - position.entry_price)
                )
                
                # Add position value to portfolio
                position_value = position.quantity * market.price
                portfolio_value += position_value
        
        return portfolio_value
    
    @time_execution
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None:
        """
        Update portfolio value history for risk calculations.
        
        Args:
            portfolio_value: Current portfolio value
        """
        self.portfolio_values.append(float(portfolio_value))
        
        # Calculate portfolio return if we have previous value
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                return_rate = (float(portfolio_value) - prev_value) / prev_value
                self.portfolio_returns.append(return_rate)
        
        # Keep only recent history
        max_history = max(self.risk_config.var_calculation_window, 252)
        if len(self.portfolio_values) > max_history:
            self.portfolio_values = self.portfolio_values[-max_history:]
        
        if len(self.portfolio_returns) > max_history:
            self.portfolio_returns = self.portfolio_returns[-max_history:]
    
    @time_execution
    async def _calculate_var(self, days: int, portfolio_value: Decimal) -> Decimal:
        """
        Calculate Value at Risk for specified time horizon.
        
        Args:
            days: Time horizon in days
            portfolio_value: Current portfolio value
            
        Returns:
            Decimal: VaR value
        """
        if len(self.portfolio_returns) < 30:
            # Insufficient data, use conservative estimate
            return portfolio_value * Decimal("0.02")  # 2% VaR
        
        # Calculate daily volatility
        returns_array = np.array(self.portfolio_returns)
        daily_volatility = np.std(returns_array)
        
        # Calculate VaR using normal distribution assumption
        # VaR = portfolio_value * volatility * sqrt(days) * z_score
        confidence_level = self.risk_config.var_confidence_level
        
        # Z-score for confidence level (95% = 1.645, 99% = 2.326)
        if confidence_level == 0.95:
            z_score = 1.645
        elif confidence_level == 0.99:
            z_score = 2.326
        else:
            # Interpolate for other confidence levels
            z_score = 1.645 + (confidence_level - 0.95) * (2.326 - 1.645) / 0.04
        
        var_value = portfolio_value * Decimal(str(daily_volatility)) * Decimal(str(np.sqrt(days))) * Decimal(str(z_score))
        
        return var_value
    
    @time_execution
    async def _calculate_expected_shortfall(self, portfolio_value: Decimal) -> Decimal:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Decimal: Expected shortfall value
        """
        if len(self.portfolio_returns) < 30:
            # Insufficient data, use conservative estimate
            return portfolio_value * Decimal("0.025")  # 2.5% ES
        
        # Calculate expected shortfall as average of worst returns
        returns_array = np.array(self.portfolio_returns)
        confidence_level = self.risk_config.var_confidence_level
        
        # Find threshold for worst (1-confidence_level) returns
        threshold = np.percentile(returns_array, (1 - confidence_level) * 100)
        
        # Calculate average of returns below threshold
        worst_returns = returns_array[returns_array <= threshold]
        
        if len(worst_returns) == 0:
            return portfolio_value * Decimal("0.02")  # Conservative fallback
        
        expected_shortfall = portfolio_value * Decimal(str(abs(np.mean(worst_returns))))
        
        return expected_shortfall
    
    @time_execution
    async def _calculate_max_drawdown(self) -> Decimal:
        """
        Calculate maximum historical drawdown.
        
        Returns:
            Decimal: Maximum drawdown value
        """
        if len(self.portfolio_values) < 2:
            return Decimal("0")
        
        # Calculate running maximum and drawdown
        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdowns = (running_max - self.portfolio_values) / running_max
        
        max_drawdown = np.max(drawdowns)
        
        return Decimal(str(max_drawdown))
    
    @time_execution
    async def _calculate_current_drawdown(self, portfolio_value: Decimal) -> Decimal:
        """
        Calculate current drawdown from peak.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Decimal: Current drawdown value
        """
        if len(self.portfolio_values) < 2:
            return Decimal("0")
        
        # Find peak value
        peak_value = max(self.portfolio_values)
        
        if peak_value <= 0:
            return Decimal("0")
        
        # Calculate current drawdown
        current_drawdown = (peak_value - float(portfolio_value)) / peak_value
        
        return Decimal(str(max(0, current_drawdown)))
    
    @time_execution
    async def _calculate_sharpe_ratio(self) -> Optional[Decimal]:
        """
        Calculate Sharpe ratio for the portfolio.
        
        Returns:
            Optional[Decimal]: Sharpe ratio or None if insufficient data
        """
        if len(self.portfolio_returns) < 30:
            return None  # Insufficient data
        
        returns_array = np.array(self.portfolio_returns)
        
        # Calculate annualized metrics
        mean_return = np.mean(returns_array) * 252  # Annualize daily returns
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualize daily volatility
        
        if volatility == 0:
            return None
        
        # Risk-free rate (assume 0% for crypto)
        risk_free_rate = 0.0
        
        # Sharpe ratio = (return - risk_free_rate) / volatility
        sharpe_ratio = (mean_return - risk_free_rate) / volatility
        
        return Decimal(str(sharpe_ratio))
    
    @time_execution
    async def _determine_risk_level(self, var_1d: Decimal, 
                                  current_drawdown: Decimal,
                                  sharpe_ratio: Optional[Decimal]) -> RiskLevel:
        """
        Determine risk level based on current metrics.
        
        Args:
            var_1d: 1-day Value at Risk
            current_drawdown: Current drawdown
            sharpe_ratio: Sharpe ratio
            
        Returns:
            RiskLevel: Determined risk level
        """
        # Risk level thresholds
        var_threshold_high = Decimal("0.05")  # 5% VaR
        var_threshold_critical = Decimal("0.10")  # 10% VaR
        drawdown_threshold_high = Decimal("0.10")  # 10% drawdown
        drawdown_threshold_critical = Decimal("0.20")  # 20% drawdown
        sharpe_threshold_low = Decimal("-1.0")  # Negative Sharpe ratio
        
        # Check for critical risk
        if (var_1d > var_threshold_critical or 
            current_drawdown > drawdown_threshold_critical):
            return RiskLevel.CRITICAL
        
        # Check for high risk
        if (var_1d > var_threshold_high or 
            current_drawdown > drawdown_threshold_high or
            (sharpe_ratio and sharpe_ratio < sharpe_threshold_low)):
            return RiskLevel.HIGH
        
        # Check for medium risk
        if (var_1d > Decimal("0.02") or  # 2% VaR
            current_drawdown > Decimal("0.05") or  # 5% drawdown
            (sharpe_ratio and sharpe_ratio < Decimal("0.5"))):
            return RiskLevel.MEDIUM
        
        # Default to low risk
        return RiskLevel.LOW
    
    @time_execution
    async def update_position_returns(self, symbol: str, price: float) -> None:
        """
        Update position return history for individual position risk.
        
        Args:
            symbol: Trading symbol
            price: Current price
        """
        if symbol not in self.position_returns:
            self.position_returns[symbol] = []
            self.position_prices[symbol] = []
        
        # Store the current price
        self.position_prices[symbol].append(price)
        
        # Calculate return if we have previous price
        if len(self.position_prices[symbol]) > 1:
            prev_price = self.position_prices[symbol][-2]
            if prev_price > 0:
                return_rate = (price - prev_price) / prev_price
                self.position_returns[symbol].append(return_rate)
        
        # Keep only recent history
        max_history = 252  # One year of trading days
        if len(self.position_returns[symbol]) > max_history:
            self.position_returns[symbol] = self.position_returns[symbol][-max_history:]
        if len(self.position_prices[symbol]) > max_history:
            self.position_prices[symbol] = self.position_prices[symbol][-max_history:]
    
    @time_execution
    async def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary.
        
        Returns:
            Dict containing current risk state and metrics
        """
        if not self.portfolio_values:
            return {"error": "No portfolio data available"}
        
        current_value = self.portfolio_values[-1] if self.portfolio_values else 0
        peak_value = max(self.portfolio_values) if self.portfolio_values else 0
        
        summary = {
            "current_portfolio_value": current_value,
            "peak_portfolio_value": peak_value,
            "total_return": (current_value - self.portfolio_values[0]) / self.portfolio_values[0] if len(self.portfolio_values) > 1 else 0,
            "data_points": len(self.portfolio_values),
            "return_data_points": len(self.portfolio_returns),
            "position_symbols": list(self.position_returns.keys())
        }
        
        return summary 