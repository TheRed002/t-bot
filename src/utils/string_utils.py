"""String utilities for the T-Bot trading system."""

import hashlib
import re

from src.core.exceptions import ValidationError


def normalize_symbol(symbol: str) -> str:
    """
    Normalize and sanitize trading symbol string.

    Args:
        symbol: Raw symbol string

    Returns:
        Normalized symbol string

    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol:
        raise ValidationError("Symbol cannot be empty")

    # Remove whitespace and convert to uppercase
    sanitized = symbol.strip().upper()

    # Remove invalid characters (keep only alphanumeric and common separators)
    sanitized = re.sub(r"[^A-Z0-9/_-]", "", sanitized)

    if not sanitized:
        raise ValidationError("Symbol contains no valid characters")

    return sanitized


# Note: format_price is available in formatters module
# from src.utils.formatters import format_price


def parse_trading_pair(pair: str) -> tuple[str, str]:
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
    pair = pair.upper().replace("/", "").replace("-", "").replace("_", "")

    # Common quote currencies
    quote_currencies = ["USDT", "USDC", "USD", "BTC", "ETH", "BNB", "ADA", "DOT"]

    for quote in quote_currencies:
        if pair.endswith(quote):
            base = pair[: -len(quote)]
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


def generate_hash(data: str) -> str:
    """
    Generate SHA-256 hash of data string.

    Args:
        data: Data to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def extract_numbers(text: str) -> list[float]:
    """
    Extract all numbers from a text string.

    Args:
        text: Text to extract numbers from

    Returns:
        List of extracted numbers
    """
    pattern = r"-?\d*\.?\d+"
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]


def camel_to_snake(name: str) -> str:
    """
    Convert camelCase to snake_case.

    Args:
        name: CamelCase string

    Returns:
        snake_case string
    """
    # Insert underscore before uppercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str) -> str:
    """
    Convert snake_case to camelCase.

    Args:
        name: snake_case string

    Returns:
        camelCase string
    """
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to specified length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return suffix[:max_length]

    return text[: max_length - len(suffix)] + suffix


# Note: format_percentage is available in formatters module
# from src.utils.formatters import format_percentage
