"""Exchange error mapping and standardization."""

from typing import Dict, Optional, Any
from datetime import datetime

from src.core.logging import get_logger

logger = get_logger(__name__)


class ExchangeError(Exception):
    """Base exchange error with common attributes."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        response: Optional[Dict] = None,
        exchange: Optional[str] = None
    ):
        """
        Initialize exchange error.
        
        Args:
            message: Error message
            code: Error code
            response: Raw response from exchange
            exchange: Exchange name
        """
        super().__init__(message)
        self.code = code
        self.response = response
        self.exchange = exchange
        self.timestamp = datetime.utcnow()


class RateLimitError(ExchangeError):
    """Rate limit exceeded error."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            **kwargs: Additional arguments for ExchangeError
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class OrderError(ExchangeError):
    """Order-related errors."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize order error.
        
        Args:
            message: Error message
            order_id: Related order ID
            **kwargs: Additional arguments for ExchangeError
        """
        super().__init__(message, **kwargs)
        self.order_id = order_id


class InsufficientBalanceError(OrderError):
    """Insufficient balance for order."""
    pass


class InvalidOrderError(OrderError):
    """Invalid order parameters."""
    pass


class AuthenticationError(ExchangeError):
    """Authentication failures."""
    pass


class NetworkError(ExchangeError):
    """Network-related errors."""
    pass


class DataError(ExchangeError):
    """Data parsing or validation errors."""
    pass


class ErrorMapper:
    """
    Maps exchange-specific errors to common errors.
    
    This eliminates duplication of error handling logic across exchanges.
    """
    
    # Binance error code mappings
    BINANCE_ERRORS = {
        -1000: ('UNKNOWN', ExchangeError),
        -1001: ('DISCONNECTED', NetworkError),
        -1002: ('UNAUTHORIZED', AuthenticationError),
        -1003: ('TOO_MANY_REQUESTS', RateLimitError),
        -1006: ('UNEXPECTED_RESP', DataError),
        -1007: ('TIMEOUT', NetworkError),
        -1021: ('INVALID_TIMESTAMP', AuthenticationError),
        -1022: ('INVALID_SIGNATURE', AuthenticationError),
        -2010: ('NEW_ORDER_REJECTED', InvalidOrderError),
        -2011: ('CANCEL_REJECTED', OrderError),
        -2013: ('NO_SUCH_ORDER', OrderError),
        -2014: ('BAD_API_KEY_FMT', AuthenticationError),
        -2015: ('REJECTED_API_KEY', AuthenticationError),
        -2018: ('BALANCE_NOT_SUFFICIENT', InsufficientBalanceError),
    }
    
    # Coinbase error mappings
    COINBASE_ERRORS = {
        'authentication_error': AuthenticationError,
        'invalid_request': InvalidOrderError,
        'rate_limit': RateLimitError,
        'insufficient_funds': InsufficientBalanceError,
        'not_found': OrderError,
        'validation_error': InvalidOrderError,
    }
    
    # OKX error code mappings
    OKX_ERRORS = {
        '1': ('Operation failed', ExchangeError),
        '2': ('Bulk operation partially succeeded', ExchangeError),
        '50000': ('Service temporarily unavailable', NetworkError),
        '50001': ('Signature authentication failed', AuthenticationError),
        '50002': ('Too many requests', RateLimitError),
        '50004': ('Endpoint request timeout', NetworkError),
        '50005': ('Invalid API key', AuthenticationError),
        '50008': ('Invalid passphrase', AuthenticationError),
        '50011': ('Invalid request', InvalidOrderError),
        '50013': ('Invalid sign', AuthenticationError),
        '51000': ('Parameter validation error', InvalidOrderError),
        '51001': ('Instrument ID does not exist', InvalidOrderError),
        '51008': ('Order amount exceeds the limit', InvalidOrderError),
        '51009': ('Order placement function is blocked', OrderError),
        '51020': ('Insufficient balance', InsufficientBalanceError),
        '51400': ('Cancellation failed', OrderError),
        '51401': ('Order does not exist', OrderError),
    }
    
    @classmethod
    def map_error(
        cls,
        exchange: str,
        error_data: Dict[str, Any]
    ) -> ExchangeError:
        """
        Map exchange-specific error to common error.
        
        Args:
            exchange: Exchange name
            error_data: Error data from exchange
            
        Returns:
            Standardized exchange error
        """
        exchange_lower = exchange.lower()
        
        if exchange_lower == 'binance':
            return cls._map_binance(error_data, exchange)
        elif exchange_lower == 'coinbase':
            return cls._map_coinbase(error_data, exchange)
        elif exchange_lower == 'okx':
            return cls._map_okx(error_data, exchange)
        else:
            return cls._map_generic(error_data, exchange)
    
    @classmethod
    def _map_binance(
        cls,
        error_data: Dict[str, Any],
        exchange: str
    ) -> ExchangeError:
        """Map Binance error."""
        code = error_data.get('code')
        msg = error_data.get('msg', 'Unknown error')
        
        if code in cls.BINANCE_ERRORS:
            error_info = cls.BINANCE_ERRORS[code]
            if isinstance(error_info, tuple):
                msg_prefix, error_class = error_info
                full_msg = f"{msg_prefix}: {msg}"
            else:
                error_class = error_info
                full_msg = msg
            
            # Special handling for rate limit
            if error_class == RateLimitError:
                # Try to extract retry-after from headers or message
                retry_after = cls._extract_retry_after(error_data)
                return RateLimitError(
                    full_msg,
                    retry_after=retry_after,
                    code=str(code),
                    response=error_data,
                    exchange=exchange
                )
            
            return error_class(
                full_msg,
                code=str(code),
                response=error_data,
                exchange=exchange
            )
        
        return ExchangeError(msg, code=str(code), response=error_data, exchange=exchange)
    
    @classmethod
    def _map_coinbase(
        cls,
        error_data: Dict[str, Any],
        exchange: str
    ) -> ExchangeError:
        """Map Coinbase error."""
        error_type = error_data.get('type', '').lower()
        message = error_data.get('message', 'Unknown error')
        
        error_class = cls.COINBASE_ERRORS.get(error_type, ExchangeError)
        
        # Special handling for rate limit
        if error_class == RateLimitError:
            retry_after = cls._extract_retry_after(error_data)
            return RateLimitError(
                message,
                retry_after=retry_after,
                code=error_type,
                response=error_data,
                exchange=exchange
            )
        
        return error_class(
            message,
            code=error_type,
            response=error_data,
            exchange=exchange
        )
    
    @classmethod
    def _map_okx(
        cls,
        error_data: Dict[str, Any],
        exchange: str
    ) -> ExchangeError:
        """Map OKX error."""
        code = str(error_data.get('code', ''))
        msg = error_data.get('msg', 'Unknown error')
        
        if code in cls.OKX_ERRORS:
            error_info = cls.OKX_ERRORS[code]
            if isinstance(error_info, tuple):
                msg_prefix, error_class = error_info
                full_msg = f"{msg_prefix}: {msg}"
            else:
                error_class = error_info
                full_msg = msg
            
            # Special handling for rate limit
            if error_class == RateLimitError:
                retry_after = cls._extract_retry_after(error_data)
                return RateLimitError(
                    full_msg,
                    retry_after=retry_after,
                    code=code,
                    response=error_data,
                    exchange=exchange
                )
            
            return error_class(
                full_msg,
                code=code,
                response=error_data,
                exchange=exchange
            )
        
        return ExchangeError(msg, code=code, response=error_data, exchange=exchange)
    
    @classmethod
    def _map_generic(
        cls,
        error_data: Dict[str, Any],
        exchange: str
    ) -> ExchangeError:
        """Map generic/unknown exchange error."""
        message = (
            error_data.get('message') or
            error_data.get('msg') or
            error_data.get('error') or
            str(error_data)
        )
        
        code = (
            error_data.get('code') or
            error_data.get('error_code') or
            'UNKNOWN'
        )
        
        # Try to detect error type from message
        message_lower = message.lower()
        
        if 'rate limit' in message_lower or '429' in message_lower:
            return RateLimitError(
                message,
                retry_after=cls._extract_retry_after(error_data),
                code=str(code),
                response=error_data,
                exchange=exchange
            )
        elif 'unauthorized' in message_lower or 'authentication' in message_lower:
            return AuthenticationError(message, code=str(code), response=error_data, exchange=exchange)
        elif 'insufficient' in message_lower or 'balance' in message_lower:
            return InsufficientBalanceError(message, code=str(code), response=error_data, exchange=exchange)
        elif 'order' in message_lower:
            return OrderError(message, code=str(code), response=error_data, exchange=exchange)
        elif 'network' in message_lower or 'timeout' in message_lower:
            return NetworkError(message, code=str(code), response=error_data, exchange=exchange)
        
        return ExchangeError(message, code=str(code), response=error_data, exchange=exchange)
    
    @staticmethod
    def _extract_retry_after(error_data: Dict[str, Any]) -> Optional[int]:
        """
        Try to extract retry-after value from error data.
        
        Args:
            error_data: Error data
            
        Returns:
            Retry after in seconds or None
        """
        # Check common fields
        retry_after = (
            error_data.get('retry_after') or
            error_data.get('retryAfter') or
            error_data.get('Retry-After')
        )
        
        if retry_after:
            try:
                return int(retry_after)
            except (TypeError, ValueError):
                pass
        
        # Try to extract from message
        import re
        message = str(error_data)
        match = re.search(r'(\d+)\s*seconds?', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None