"""DateTime utilities for the T-Bot trading system."""

from datetime import datetime, timezone
from typing import Optional

import pytz

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class DateTimeUtils:
    """All datetime operations."""
    
    @staticmethod
    def to_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Convert datetime to timestamp string.
        
        Args:
            dt: Datetime to convert
            format_str: Format string
            
        Returns:
            Timestamp string
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_timeframe(timeframe: str) -> int:
        """
        Parse timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "5m", "1d")
            
        Returns:
            Number of seconds
            
        Raises:
            ValidationError: If timeframe format is invalid
        """
        if not timeframe:
            raise ValidationError("Timeframe cannot be empty")
        
        # Parse number and unit
        import re
        match = re.match(r'^(\d+)([smhd])$', timeframe.lower())
        if not match:
            raise ValidationError(f"Invalid timeframe format: {timeframe}")
        
        number = int(match.group(1))
        unit = match.group(2)
        
        multipliers = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400
        }
        
        return number * multipliers[unit]
    
    @staticmethod
    def get_trading_session(dt: datetime, exchange: str = "binance") -> str:
        """
        Determine the trading session for a given datetime and exchange.
        
        Args:
            dt: Datetime to check
            exchange: Exchange name (default "binance")
            
        Returns:
            Trading session as string
        """
        # Convert to UTC if not already
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Get hour in UTC
        hour = dt.hour
        
        # Crypto exchanges are typically 24/7, but we can define sessions
        if exchange.lower() in ["binance", "okx", "coinbase"]:
            # Crypto markets are 24/7, but we can define peak hours
            if 0 <= hour < 6:
                return "low_activity"
            elif 6 <= hour < 14:
                return "asian_session"
            elif 14 <= hour < 22:
                return "european_session"
            else:
                return "american_session"
        else:
            # Traditional market hours (example for NYSE)
            if 9 <= hour < 16:
                return "regular"
            elif 4 <= hour < 9:
                return "pre_market"
            elif 16 <= hour < 20:
                return "post_market"
            else:
                return "closed"
    
    @staticmethod
    def is_market_open(dt: datetime, exchange: str = "binance") -> bool:
        """
        Check if the market is open for a given datetime and exchange.
        
        Args:
            dt: Datetime to check
            exchange: Exchange name (default "binance")
            
        Returns:
            True if market is open, False otherwise
        """
        if exchange.lower() in ["binance", "okx", "coinbase"]:
            # Crypto markets are always open
            return True
        else:
            # For traditional markets, check if it's a weekday and during market hours
            if dt.weekday() >= 5:  # Saturday or Sunday
                return False
            
            hour = dt.hour
            return 9 <= hour < 16  # 9 AM to 4 PM
    
    @staticmethod
    def convert_timezone(dt: datetime, target_tz: str) -> datetime:
        """
        Convert datetime to target timezone.
        
        Args:
            dt: Datetime to convert
            target_tz: Target timezone string (e.g., "UTC", "America/New_York")
            
        Returns:
            Datetime in target timezone
            
        Raises:
            ValidationError: If timezone is invalid
        """
        # Validate timezone string format
        if not isinstance(target_tz, str) or not target_tz.strip():
            raise ValidationError("Timezone must be a non-empty string")
        
        try:
            # Ensure datetime has timezone info
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Convert to target timezone
            target_timezone = pytz.timezone(target_tz)
            converted_dt = dt.astimezone(target_timezone)
            
            return converted_dt
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValidationError(f"Invalid timezone: {target_tz}")
        except Exception as e:
            raise ValidationError(f"Error converting timezone: {str(e)}")
    
    @staticmethod
    def parse_datetime(dt_str: str, format_str: Optional[str] = None) -> datetime:
        """
        Parse datetime string to datetime object.
        
        Args:
            dt_str: Datetime string to parse
            format_str: Format string (if None, will try common formats)
            
        Returns:
            Parsed datetime object
            
        Raises:
            ValidationError: If datetime string cannot be parsed
        """
        if format_str:
            try:
                return datetime.strptime(dt_str, format_str)
            except ValueError as e:
                raise ValidationError(
                    f"Cannot parse datetime '{dt_str}' with format '{format_str}': {str(e)}"
                )
        
        # Try common formats
        common_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%H:%M:%S",
        ]
        
        for fmt in common_formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        
        raise ValidationError(f"Cannot parse datetime string '{dt_str}' with any known format")