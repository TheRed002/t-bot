"""Mathematical and statistical utilities for the T-Bot trading system."""

from typing import List, Optional, Tuple
import numpy as np

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class MathUtils:
    """All mathematical and statistical operations."""
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """
        Calculate percentage change between two values.
        
        Args:
            old_value: Original value
            new_value: New value
            
        Returns:
            Percentage change as a float (e.g., 0.05 for 5% increase)
            
        Raises:
            ValidationError: If old_value is zero
        """
        if old_value == 0:
            raise ValidationError("Cannot calculate percentage change with zero old value")
        
        percentage_change = (new_value - old_value) / old_value
        return float(percentage_change)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.02,
        frequency: str = "daily"
    ) -> float:
        """
        Calculate the Sharpe ratio for a series of returns.
        
        Args:
            returns: List of return values (as decimals, e.g., 0.05 for 5%)
            risk_free_rate: Annual risk-free rate (default 2%)
            frequency: Data frequency ("daily", "weekly", "monthly", "yearly")
            
        Returns:
            Sharpe ratio as a float
            
        Raises:
            ValidationError: If returns list is empty or contains invalid values
        """
        if not returns:
            raise ValidationError("Returns list cannot be empty")
        
        if len(returns) < 2:
            raise ValidationError("Need at least 2 returns to calculate Sharpe ratio")
        
        # Validate frequency
        valid_frequencies = {"daily": 252, "weekly": 52, "monthly": 12, "yearly": 1}
        if frequency not in valid_frequencies:
            raise ValidationError(
                f"Invalid frequency: {frequency}. Must be one of {list(valid_frequencies.keys())}"
            )
        
        # Convert to numpy array for calculations
        returns_array = np.array(returns)
        
        # Calculate annualization factor
        periods_per_year = valid_frequencies[frequency]
        
        # Calculate mean return (annualized)
        mean_return = np.mean(returns_array) * periods_per_year
        
        # Calculate standard deviation (annualized)
        std_return = np.std(returns_array, ddof=1) * np.sqrt(periods_per_year)
        
        # Avoid division by zero
        if std_return == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        return float(sharpe_ratio)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
        """
        Calculate the maximum drawdown from an equity curve.
        
        Args:
            equity_curve: List of equity values over time
            
        Returns:
            Tuple of (max_drawdown, start_index, end_index)
            
        Raises:
            ValidationError: If equity curve is empty or contains invalid values
        """
        if not equity_curve:
            raise ValidationError("Equity curve cannot be empty")
        
        if len(equity_curve) < 2:
            raise ValidationError("Need at least 2 points to calculate drawdown")
        
        # Convert to numpy array
        equity = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown_idx = np.argmin(drawdown)
        max_drawdown = drawdown[max_drawdown_idx]
        
        # Find the peak before the maximum drawdown
        peak_idx = np.argmax(equity[:max_drawdown_idx + 1])
        
        return float(max_drawdown), int(peak_idx), int(max_drawdown_idx)
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for a series of returns.
        
        Args:
            returns: List of return values
            confidence_level: Confidence level for VaR calculation (default 95%)
            
        Returns:
            VaR as a float (negative value represents loss)
            
        Raises:
            ValidationError: If returns list is empty or confidence level is invalid
        """
        if not returns:
            raise ValidationError("Returns list cannot be empty")
        
        if not 0 < confidence_level < 1:
            raise ValidationError("Confidence level must be between 0 and 1")
        
        # Convert to numpy array
        returns_array = np.array(returns)
        
        # Calculate VaR using historical simulation
        # For 95% confidence, we want the 5th percentile (worst 5% of returns)
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns_array, var_percentile)
        
        return float(var)
    
    @staticmethod
    def calculate_volatility(returns: List[float], window: Optional[int] = None) -> float:
        """
        Calculate volatility (standard deviation) of returns.
        
        Args:
            returns: List of return values
            window: Rolling window size (None for full series)
            
        Returns:
            Volatility as a float
            
        Raises:
            ValidationError: If returns list is empty or window is invalid
        """
        if not returns:
            raise ValidationError("Returns list cannot be empty")
        
        if window is not None and (window <= 0 or window > len(returns)):
            raise ValidationError(f"Invalid window size: {window}")
        
        # Convert to numpy array
        returns_array = np.array(returns)
        
        if window is None:
            # Calculate volatility for entire series
            volatility = np.std(returns_array, ddof=1)
        else:
            # Calculate rolling volatility
            if len(returns_array) < window:
                raise ValidationError(f"Not enough data for window size {window}")
            
            # Use the last window elements
            recent_returns = returns_array[-window:]
            volatility = np.std(recent_returns, ddof=1)
        
        return float(volatility)
    
    @staticmethod
    def calculate_correlation(series1: List[float], series2: List[float]) -> float:
        """
        Calculate correlation coefficient between two series.
        
        Args:
            series1: First series of values
            series2: Second series of values
            
        Returns:
            Correlation coefficient as a float
            
        Raises:
            ValidationError: If series are empty or have different lengths
        """
        if not series1 or not series2:
            raise ValidationError("Both series must not be empty")
        
        if len(series1) != len(series2):
            raise ValidationError("Series must have the same length")
        
        if len(series1) < 2:
            raise ValidationError("Need at least 2 points to calculate correlation")
        
        # Convert to numpy arrays
        arr1 = np.array(series1)
        arr2 = np.array(series2)
        
        # Remove any NaN values from both arrays
        mask = ~(np.isnan(arr1) | np.isnan(arr2))
        if np.sum(mask) < 2:
            raise ValidationError("Not enough valid data points after removing NaN values")
        
        arr1_clean = arr1[mask]
        arr2_clean = arr2[mask]
        
        # Calculate correlation
        correlation = np.corrcoef(arr1_clean, arr2_clean)[0, 1]
        
        # Handle NaN values
        if np.isnan(correlation):
            return 0.0
        
        return float(correlation)
    
    @staticmethod
    def calculate_beta(
        asset_returns: List[float],
        market_returns: List[float]
    ) -> float:
        """
        Calculate beta coefficient for an asset relative to market.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            
        Returns:
            Beta coefficient
            
        Raises:
            ValidationError: If series are invalid
        """
        if not asset_returns or not market_returns:
            raise ValidationError("Return series cannot be empty")
        
        if len(asset_returns) != len(market_returns):
            raise ValidationError("Return series must have the same length")
        
        # Convert to numpy arrays
        asset = np.array(asset_returns)
        market = np.array(market_returns)
        
        # Calculate covariance and variance
        covariance = np.cov(asset, market)[0, 1]
        market_variance = np.var(market, ddof=1)
        
        if market_variance == 0:
            return 0.0
        
        beta = covariance / market_variance
        return float(beta)
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            
        Returns:
            Sortino ratio
        """
        returns_array = np.array(returns)
        if len(returns_array) < 2:
            return 0.0
        
        period_rf_rate = risk_free_rate / periods_per_year
        excess_returns = returns_array - period_rf_rate
        
        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        mean_excess = np.mean(excess_returns)
        return (mean_excess / downside_std) * np.sqrt(periods_per_year)