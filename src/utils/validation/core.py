"""Centralized validation framework for the T-Bot trading system."""

from typing import Any, Dict, List, Tuple, Callable, Optional
from decimal import Decimal
import re


class ValidationFramework:
    """Centralized validation framework to eliminate duplication."""
    
    @staticmethod
    def validate_order(order: Dict[str, Any]) -> bool:
        """
        Single source of truth for order validation.
        
        Args:
            order: Order dictionary to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        validators = [
            ('price', lambda x: x > 0, "Price must be positive"),
            ('quantity', lambda x: x > 0, "Quantity must be positive"),
            ('symbol', lambda x: bool(x) and isinstance(x, str), "Symbol required and must be string"),
            ('side', lambda x: x in ['BUY', 'SELL'], "Side must be BUY or SELL"),
            ('type', lambda x: x in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'], "Invalid order type")
        ]
        
        for field, validator_func, error_msg in validators:
            if field in order:
                if not validator_func(order[field]):
                    raise ValueError(f"{field}: {error_msg}")
            elif field in ['price', 'quantity', 'symbol', 'side', 'type']:
                # Required fields
                raise ValueError(f"{field} is required")
        
        return True
    
    @staticmethod
    def validate_strategy_params(params: Dict[str, Any]) -> bool:
        """
        Single source for strategy parameter validation.
        
        Args:
            params: Strategy parameters to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if 'strategy_type' not in params:
            raise ValueError("strategy_type is required")
        
        strategy_type = params['strategy_type']
        
        # Common validations
        if 'timeframe' in params:
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if params['timeframe'] not in valid_timeframes:
                raise ValueError(f"Invalid timeframe. Must be one of {valid_timeframes}")
        
        # Strategy-specific validations
        if strategy_type == 'MEAN_REVERSION':
            required = ['window_size', 'num_std', 'entry_threshold']
            for field in required:
                if field not in params:
                    raise ValueError(f"{field} is required for MEAN_REVERSION strategy")
            
            if params['window_size'] < 2:
                raise ValueError("window_size must be at least 2")
            if params['num_std'] <= 0:
                raise ValueError("num_std must be positive")
                
        elif strategy_type == 'MOMENTUM':
            required = ['lookback_period', 'momentum_threshold']
            for field in required:
                if field not in params:
                    raise ValueError(f"{field} is required for MOMENTUM strategy")
            
            if params['lookback_period'] < 1:
                raise ValueError("lookback_period must be at least 1")
        
        return True
    
    @staticmethod
    def validate_price(price: Any, max_price: float = 1_000_000) -> float:
        """
        Validate and normalize price.
        
        Args:
            price: Price to validate
            max_price: Maximum allowed price
            
        Returns:
            Normalized price
            
        Raises:
            ValueError: If price is invalid
        """
        try:
            price_float = float(price)
        except (TypeError, ValueError):
            raise ValueError(f"Price must be numeric, got {type(price)}")
        
        if price_float <= 0:
            raise ValueError("Price must be positive")
        if price_float > max_price:
            raise ValueError(f"Price {price_float} exceeds maximum {max_price}")
        
        # Round to 8 decimal places (crypto precision)
        return round(price_float, 8)
    
    @staticmethod
    def validate_quantity(quantity: Any, min_qty: float = 0.00000001) -> float:
        """
        Validate and normalize quantity.
        
        Args:
            quantity: Quantity to validate
            min_qty: Minimum allowed quantity
            
        Returns:
            Normalized quantity
            
        Raises:
            ValueError: If quantity is invalid
        """
        try:
            qty_float = float(quantity)
        except (TypeError, ValueError):
            raise ValueError(f"Quantity must be numeric, got {type(quantity)}")
        
        if qty_float <= 0:
            raise ValueError("Quantity must be positive")
        if qty_float < min_qty:
            raise ValueError(f"Quantity {qty_float} below minimum {min_qty}")
        
        return qty_float
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Validate and normalize trading symbol.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            Normalized symbol
            
        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Normalize to uppercase
        symbol = symbol.upper().strip()
        
        # Check format (e.g., BTC/USDT or BTCUSDT)
        if not re.match(r'^[A-Z]+(/|_|-)?[A-Z]+$', symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        return symbol
    
    @staticmethod
    def validate_exchange_credentials(credentials: Dict[str, Any]) -> bool:
        """
        Validate exchange API credentials.
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        required_fields = ['api_key', 'api_secret']
        
        for field in required_fields:
            if field not in credentials:
                raise ValueError(f"{field} is required")
            if not credentials[field] or not isinstance(credentials[field], str):
                raise ValueError(f"{field} must be a non-empty string")
        
        # Check for test/production mode
        if 'testnet' in credentials and not isinstance(credentials['testnet'], bool):
            raise ValueError("testnet must be a boolean")
        
        return True
    
    @staticmethod
    def validate_risk_parameters(params: Dict[str, Any]) -> bool:
        """
        Validate risk management parameters.
        
        Args:
            params: Risk parameters to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        validators = [
            ('max_position_size', lambda x: 0 < x <= 1, "Max position size must be between 0 and 1"),
            ('stop_loss_pct', lambda x: 0 < x < 1, "Stop loss percentage must be between 0 and 1"),
            ('take_profit_pct', lambda x: 0 < x < 10, "Take profit percentage must be between 0 and 10"),
            ('max_drawdown', lambda x: 0 < x < 1, "Max drawdown must be between 0 and 1"),
            ('risk_per_trade', lambda x: 0 < x <= 0.1, "Risk per trade must be between 0 and 0.1 (10%)")
        ]
        
        for field, validator_func, error_msg in validators:
            if field in params:
                if not validator_func(params[field]):
                    raise ValueError(f"{field}: {error_msg}")
        
        return True
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """
        Validate and normalize timeframe.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Normalized timeframe
            
        Raises:
            ValueError: If timeframe is invalid
        """
        valid_timeframes = {
            '1m': '1m', '1min': '1m', '1minute': '1m',
            '5m': '5m', '5min': '5m', '5minutes': '5m',
            '15m': '15m', '15min': '15m', '15minutes': '15m',
            '30m': '30m', '30min': '30m', '30minutes': '30m',
            '1h': '1h', '1hr': '1h', '1hour': '1h', '60m': '1h',
            '4h': '4h', '4hr': '4h', '4hours': '4h', '240m': '4h',
            '1d': '1d', '1day': '1d', 'daily': '1d',
            '1w': '1w', '1week': '1w', 'weekly': '1w'
        }
        
        timeframe_lower = timeframe.lower().strip()
        
        if timeframe_lower not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid options: {list(set(valid_timeframes.values()))}")
        
        return valid_timeframes[timeframe_lower]
    
    @staticmethod
    def validate_batch(validations: List[Tuple[str, Callable, Any]]) -> Dict[str, Any]:
        """
        Run multiple validations and collect results.
        
        Args:
            validations: List of (name, validator_func, data) tuples
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        for name, validator_func, data in validations:
            try:
                result = validator_func(data)
                results[name] = {'status': 'success', 'result': result}
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}
        
        return results