"""Tests for the ValidationFramework."""

import pytest
from decimal import Decimal

from src.core.exceptions import ValidationError
from src.utils.validation import (
    validator,
    validate_order,
    validate_price,
    validate_quantity,
    validate_symbol,
    validate_timeframe,
    validate_strategy_params,
    validate_risk_parameters,
    validate_exchange_credentials
)


class TestValidationFramework:
    """Test ValidationFramework methods."""
    
    def test_validate_order_valid(self):
        """Test valid order validation."""
        order = {
            'price': 100.0,
            'quantity': 1.0,
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'type': 'LIMIT'
        }
        
        assert validator.validate_order(order) is True
    
    def test_validate_order_missing_required(self):
        """Test order validation with missing required fields."""
        order = {
            'price': 100.0,
            'quantity': 1.0
            # Missing symbol, side, type
        }
        
        with pytest.raises(ValidationError, match="symbol is required"):
            validator.validate_order(order)
    
    def test_validate_order_invalid_price(self):
        """Test order validation with invalid price."""
        order = {
            'price': -100.0,  # Negative price
            'quantity': 1.0,
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'type': 'LIMIT'
        }
        
        with pytest.raises(ValidationError, match="Price must be positive"):
            validator.validate_order(order)
    
    def test_validate_order_invalid_side(self):
        """Test order validation with invalid side."""
        order = {
            'price': 100.0,
            'quantity': 1.0,
            'symbol': 'BTC/USDT',
            'side': 'INVALID',  # Invalid side
            'type': 'LIMIT'
        }
        
        with pytest.raises(ValidationError, match="Side must be BUY or SELL"):
            validator.validate_order(order)
    
    def test_validate_price(self):
        """Test price validation."""
        # Valid prices
        assert validate_price(100.0) == 100.0
        assert validate_price("100.123456789") == 100.12345679  # Rounded to 8 decimals
        assert validate_price(Decimal("50000")) == 50000.0
        
        # Invalid prices
        with pytest.raises(ValidationError, match="Price must be positive"):
            validate_price(0)
        
        with pytest.raises(ValidationError, match="Price must be positive"):
            validate_price(-100)
        
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_price(2_000_000)  # Over default max
        
        with pytest.raises(ValidationError, match="Price must be numeric"):
            validate_price("not_a_number")
    
    def test_validate_quantity(self):
        """Test quantity validation."""
        # Valid quantities
        assert validate_quantity(1.0) == 1.0
        assert validate_quantity("0.001") == 0.001
        assert validate_quantity(1000) == 1000.0
        
        # Invalid quantities
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            validate_quantity(0)
        
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            validate_quantity(-1)
        
        with pytest.raises(ValidationError, match="below minimum"):
            validate_quantity(0.000000001, min_qty=0.001)
    
    def test_validate_symbol(self):
        """Test symbol validation."""
        # Valid symbols
        assert validate_symbol("btc/usdt") == "BTC/USDT"  # Normalized
        assert validate_symbol("BTCUSDT") == "BTCUSDT"
        assert validate_symbol("ETH-USD") == "ETH-USD"
        assert validate_symbol("BTC_USDT") == "BTC_USDT"
        
        # Invalid symbols
        with pytest.raises(ValidationError, match="non-empty string"):
            validate_symbol("")
        
        with pytest.raises(ValidationError, match="non-empty string"):
            validate_symbol(None)
        
        with pytest.raises(ValidationError, match="Invalid symbol format"):
            validate_symbol("123/456")  # Numbers only
        
        with pytest.raises(ValidationError, match="Invalid symbol format"):
            validate_symbol("BTC/")  # Incomplete
    
    def test_validate_timeframe(self):
        """Test timeframe validation."""
        # Valid timeframes with normalization
        assert validate_timeframe("1m") == "1m"
        assert validate_timeframe("1min") == "1m"
        assert validate_timeframe("1minute") == "1m"
        assert validate_timeframe("1h") == "1h"
        assert validate_timeframe("1hour") == "1h"
        assert validate_timeframe("60m") == "1h"
        assert validate_timeframe("daily") == "1d"
        assert validate_timeframe("weekly") == "1w"
        
        # Invalid timeframes
        with pytest.raises(ValidationError, match="Invalid timeframe"):
            validate_timeframe("2h")  # Not in valid list
        
        with pytest.raises(ValidationError, match="Invalid timeframe"):
            validate_timeframe("invalid")
    
    def test_validate_strategy_params(self):
        """Test strategy parameter validation."""
        # Valid mean reversion params
        params = {
            'strategy_type': 'MEAN_REVERSION',
            'window_size': 20,
            'num_std': 2.0,
            'entry_threshold': 0.8,
            'timeframe': '1h'
        }
        assert validate_strategy_params(params) is True
        
        # Missing required field
        params = {
            'strategy_type': 'MEAN_REVERSION',
            'window_size': 20
            # Missing num_std, entry_threshold
        }
        with pytest.raises(ValidationError, match="num_std is required"):
            validate_strategy_params(params)
        
        # Invalid window size
        params = {
            'strategy_type': 'MEAN_REVERSION',
            'window_size': 1,  # Too small
            'num_std': 2.0,
            'entry_threshold': 0.8
        }
        with pytest.raises(ValidationError, match="window_size must be at least 2"):
            validate_strategy_params(params)
        
        # Valid momentum params
        params = {
            'strategy_type': 'MOMENTUM',
            'lookback_period': 14,
            'momentum_threshold': 0.05
        }
        assert validate_strategy_params(params) is True
    
    def test_validate_risk_parameters(self):
        """Test risk parameter validation."""
        # Valid params
        params = {
            'max_position_size': 0.1,  # 10%
            'stop_loss_pct': 0.02,  # 2%
            'take_profit_pct': 0.05,  # 5%
            'max_drawdown': 0.2,  # 20%
            'risk_per_trade': 0.02  # 2%
        }
        assert validate_risk_parameters(params) is True
        
        # Invalid position size
        params = {'max_position_size': 1.5}  # 150% - too high
        with pytest.raises(ValidationError, match="Max position size must be between"):
            validate_risk_parameters(params)
        
        # Invalid stop loss
        params = {'stop_loss_pct': 0.6}  # 60% - too high
        with pytest.raises(ValidationError, match="Stop loss percentage must be between"):
            validate_risk_parameters(params)
        
        # Invalid risk per trade
        params = {'risk_per_trade': 0.15}  # 15% - too high
        with pytest.raises(ValidationError, match="Risk per trade must be between"):
            validate_risk_parameters(params)
    
    def test_validate_exchange_credentials(self):
        """Test exchange credential validation."""
        # Valid credentials
        creds = {
            'api_key': 'test_api_key_123',
            'api_secret': 'test_api_secret_456',
            'testnet': True
        }
        assert validate_exchange_credentials(creds) is True
        
        # Missing required field
        creds = {
            'api_key': 'test_api_key_123'
            # Missing api_secret
        }
        with pytest.raises(ValidationError, match="api_secret is required"):
            validate_exchange_credentials(creds)
        
        # Empty api_key
        creds = {
            'api_key': '',
            'api_secret': 'test_api_secret_456'
        }
        with pytest.raises(ValidationError, match="api_key must be a non-empty string"):
            validate_exchange_credentials(creds)
        
        # Invalid testnet type
        creds = {
            'api_key': 'test_api_key_123',
            'api_secret': 'test_api_secret_456',
            'testnet': 'yes'  # Should be boolean
        }
        with pytest.raises(ValidationError, match="testnet must be a boolean"):
            validate_exchange_credentials(creds)
    
    def test_validate_batch(self):
        """Test batch validation."""
        validations = [
            ('price', validator.validate_price, 100.0),
            ('quantity', validator.validate_quantity, 1.0),
            ('symbol', validator.validate_symbol, 'BTC/USDT'),
            ('invalid_price', validator.validate_price, -100)  # This will fail
        ]
        
        results = validator.validate_batch(validations)
        
        # Check successful validations
        assert results['price']['status'] == 'success'
        assert results['price']['result'] == 100.0
        
        assert results['quantity']['status'] == 'success'
        assert results['quantity']['result'] == 1.0
        
        assert results['symbol']['status'] == 'success'
        assert results['symbol']['result'] == 'BTC/USDT'
        
        # Check failed validation
        assert results['invalid_price']['status'] == 'error'
        assert 'Price must be positive' in results['invalid_price']['error']