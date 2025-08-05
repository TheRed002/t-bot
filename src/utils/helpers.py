"""
Helper Functions for Common Operations

This module provides utility functions for mathematical calculations, date/time handling,
data conversion, file operations, network utilities, and string processing that are used
across all components of the trading bot system.

Key Functions:
- Mathematical Utilities: statistical calculations, financial metrics
- Date/Time Utilities: timezone handling, trading session detection
- Data Conversion: unit conversions, currency conversions
- File Operations: safe file I/O, configuration loading
- Network Utilities: connection testing, latency measurement
- String Utilities: parsing, formatting, sanitization

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
"""

import math
import statistics
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta, timezone
import pytz
import asyncio
import aiohttp
import socket
import os
import json
import yaml
from pathlib import Path
import re
import hashlib

# Import from P-001 core components
from src.core.exceptions import ValidationError, DataError
from src.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Mathematical Utilities
# =============================================================================

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: List of return values (as decimals, e.g., 0.05 for 5%)
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Sharpe ratio as a float
        
    Raises:
        ValidationError: If returns list is empty or contains invalid values
    """
    if not returns:
        raise ValidationError("Returns list cannot be empty")
    
    if len(returns) < 2:
        raise ValidationError("Need at least 2 returns to calculate Sharpe ratio")
    
    # Convert to numpy array for calculations
    returns_array = np.array(returns)
    
    # Calculate mean return (annualized)
    mean_return = np.mean(returns_array) * 252  # Assuming daily returns
    
    # Calculate standard deviation (annualized)
    std_return = np.std(returns_array, ddof=1) * np.sqrt(252)
    
    # Avoid division by zero
    if std_return == 0:
        return 0.0
    
    # Calculate Sharpe ratio
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    
    return float(sharpe_ratio)


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
    var_percentile = (1 - confidence_level) * 100
    var = np.percentile(returns_array, [var_percentile])[0]
    
    return float(var)


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
    
    # Calculate correlation
    correlation = np.corrcoef(arr1, arr2)[0, 1]
    
    # Handle NaN values
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


# =============================================================================
# Date/Time Utilities
# =============================================================================

def get_trading_session(dt: datetime, exchange: str = "binance") -> str:
    """
    Determine the trading session for a given datetime and exchange.
    
    Args:
        dt: Datetime to check
        exchange: Exchange name (default "binance")
        
    Returns:
        Trading session as string ("pre_market", "regular", "post_market", "closed")
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
    try:
        # Ensure datetime has timezone info
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to target timezone
        target_timezone = pytz.timezone(target_tz)
        converted_dt = dt.astimezone(target_timezone)
        
        return converted_dt
    except Exception as e:
        raise ValidationError(f"Invalid timezone '{target_tz}': {str(e)}")


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
            raise ValidationError(f"Cannot parse datetime '{dt_str}' with format '{format_str}': {str(e)}")
    
    # Try common formats
    common_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%H:%M:%S"
    ]
    
    for fmt in common_formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    
    raise ValidationError(f"Cannot parse datetime string '{dt_str}' with any known format")


# =============================================================================
# Data Conversion Utilities
# =============================================================================

def convert_currency(amount: float, from_currency: str, to_currency: str, 
                    exchange_rate: float) -> float:
    """
    Convert amount from one currency to another.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency
        to_currency: Target currency
        exchange_rate: Exchange rate (from_currency/to_currency)
        
    Returns:
        Converted amount
        
    Raises:
        ValidationError: If amount is negative or exchange rate is invalid
    """
    if amount < 0:
        raise ValidationError("Amount cannot be negative")
    
    if exchange_rate <= 0:
        raise ValidationError("Exchange rate must be positive")
    
    converted_amount = amount * exchange_rate
    
    # Round to appropriate precision based on currency
    if to_currency.upper() in ["BTC", "ETH"]:
        precision = 8
    elif to_currency.upper() in ["USDT", "USDC", "USD"]:
        precision = 2
    else:
        precision = 4
    
    return round(converted_amount, precision)


def normalize_price(price: float, symbol: str) -> Decimal:
    """
    Normalize price to appropriate precision for a given symbol.
    
    Args:
        price: Price to normalize
        symbol: Trading symbol
        
    Returns:
        Normalized price as Decimal
        
    Raises:
        ValidationError: If price is invalid
    """
    if price <= 0:
        raise ValidationError(f"Price must be positive for {symbol}, got {price}")
    
    # Determine precision based on symbol
    if "BTC" in symbol.upper():
        precision = 8
    elif "ETH" in symbol.upper():
        precision = 6
    elif "USDT" in symbol.upper() or "USD" in symbol.upper():
        precision = 2
    else:
        precision = 4
    
    # Convert to Decimal and round
    decimal_price = Decimal(str(price))
    normalized_price = decimal_price.quantize(Decimal(f"0.{'0' * (precision - 1)}1"), 
                                           rounding=ROUND_HALF_UP)
    
    return normalized_price


def round_to_precision(value: float, precision: int) -> float:
    """
    Round value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    if precision < 0:
        raise ValidationError("Precision must be non-negative")
    
    factor = 10 ** precision
    return round(value * factor) / factor


# =============================================================================
# File Operations
# =============================================================================

def safe_read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Safely read a file with error handling.
    
    Args:
        file_path: Path to file to read
        encoding: File encoding (default "utf-8")
        
    Returns:
        File contents as string
        
    Raises:
        ValidationError: If file cannot be read
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        
        logger.debug(f"Successfully read file: {file_path}")
        return content
        
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        raise ValidationError(f"Cannot read file '{file_path}': {str(e)}")


def safe_write_file(file_path: str, content: str, encoding: str = "utf-8") -> None:
    """
    Safely write content to a file with error handling.
    
    Args:
        file_path: Path to file to write
        content: Content to write
        encoding: File encoding (default "utf-8")
        
    Raises:
        ValidationError: If file cannot be written
    """
    try:
        path = Path(file_path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content atomically using temporary file
        temp_path = path.with_suffix(path.suffix + '.tmp')
        
        with open(temp_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        # Atomic move
        temp_path.replace(path)
        
        logger.debug(f"Successfully wrote file: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {str(e)}")
        raise ValidationError(f"Cannot write file '{file_path}': {str(e)}")


def load_config_file(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValidationError: If file cannot be loaded or parsed
    """
    try:
        content = safe_read_file(file_path)
        path = Path(file_path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(content)
        elif path.suffix.lower() == '.json':
            config = json.loads(content)
        else:
            raise ValidationError(f"Unsupported file format: {path.suffix}")
        
        if not isinstance(config, dict):
            raise ValidationError("Configuration must be a dictionary")
        
        logger.debug(f"Successfully loaded config file: {file_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config file {file_path}: {str(e)}")
        raise ValidationError(f"Cannot load config file '{file_path}': {str(e)}")


# =============================================================================
# Network Utilities
# =============================================================================

async def test_connection(host: str, port: int, timeout: float = 5.0) -> bool:
    """
    Test network connection to a host and port.
    
    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception as e:
        logger.debug(f"Connection test failed for {host}:{port}: {str(e)}")
        return False


async def measure_latency(host: str, port: int, timeout: float = 5.0) -> float:
    """
    Measure network latency to a host and port.
    
    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds
        
    Returns:
        Latency in milliseconds
        
    Raises:
        ValidationError: If connection fails
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        
        end_time = asyncio.get_event_loop().time()
        latency_ms = (end_time - start_time) * 1000
        
        writer.close()
        await writer.wait_closed()
        
        return latency_ms
        
    except Exception as e:
        logger.error(f"Failed to measure latency for {host}:{port}: {str(e)}")
        raise ValidationError(f"Cannot measure latency to {host}:{port}: {str(e)}")


async def ping_host(host: str, count: int = 3) -> Dict[str, Any]:
    """
    Ping a host and return statistics.
    
    Args:
        host: Hostname or IP address
        count: Number of ping attempts
        
    Returns:
        Dictionary with ping statistics
    """
    try:
        latencies = []
        
        for i in range(count):
            try:
                latency = await measure_latency(host, 80)  # HTTP port
                latencies.append(latency)
                await asyncio.sleep(0.1)  # Small delay between pings
            except Exception as e:
                logger.warning(f"Ping attempt {i+1} failed for {host}: {str(e)}")
        
        if not latencies:
            return {
                "host": host,
                "success": False,
                "error": "All ping attempts failed"
            }
        
        return {
            "host": host,
            "success": True,
            "count": len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "packet_loss_pct": ((count - len(latencies)) / count) * 100
        }
        
    except Exception as e:
        logger.error(f"Ping failed for {host}: {str(e)}")
        return {
            "host": host,
            "success": False,
            "error": str(e)
        }


# =============================================================================
# String Utilities
# =============================================================================

def sanitize_symbol(symbol: str) -> str:
    """
    Sanitize trading symbol string.
    
    Args:
        symbol: Raw symbol string
        
    Returns:
        Sanitized symbol string
        
    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol:
        raise ValidationError("Symbol cannot be empty")
    
    # Remove whitespace and convert to uppercase
    sanitized = symbol.strip().upper()
    
    # Remove invalid characters (keep only alphanumeric and common separators)
    sanitized = re.sub(r'[^A-Z0-9/_-]', '', sanitized)
    
    if not sanitized:
        raise ValidationError("Symbol contains no valid characters")
    
    return sanitized


def parse_trading_pair(pair: str) -> Tuple[str, str]:
    """
    Parse trading pair string into base and quote currencies.
    
    Args:
        pair: Trading pair string (e.g., "BTCUSDT", "ETH/BTC")
        
    Returns:
        Tuple of (base_currency, quote_currency)
        
    Raises:
        ValidationError: If pair format is invalid
    """
    if not pair:
        raise ValidationError("Trading pair cannot be empty")
    
    # Remove common separators and convert to uppercase
    pair = pair.upper().replace('/', '').replace('-', '').replace('_', '')
    
    # Common quote currencies
    quote_currencies = ['USDT', 'USDC', 'USD', 'BTC', 'ETH', 'BNB', 'ADA', 'DOT']
    
    for quote in quote_currencies:
        if pair.endswith(quote):
            base = pair[:-len(quote)]
            if base:
                return base, quote
    
    # If no common quote currency found, try to split at common lengths
    if len(pair) >= 6:
        # Try splitting at position 3 (common for crypto pairs)
        base = pair[:3]
        quote = pair[3:]
        if base and quote:
            return base, quote
    
    raise ValidationError(f"Cannot parse trading pair: {pair}")


def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime as timestamp string.
    
    Args:
        dt: Datetime to format
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.strftime(format_str)


def generate_hash(data: str) -> str:
    """
    Generate SHA-256 hash of data string.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from a text string.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of extracted numbers
    """
    pattern = r'-?\d*\.?\d+'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches] 