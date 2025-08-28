"""Data conversion and manipulation utilities for the T-Bot trading system."""

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.exceptions import ValidationError
from src.utils.decimal_utils import ZERO, to_decimal


def dict_to_dataframe(data: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert dictionary or list of dictionaries to DataFrame.

    Args:
        data: Dictionary or list of dictionaries

    Returns:
        Pandas DataFrame

    Raises:
        ValidationError: If data format is invalid
    """
    try:
        if isinstance(data, dict):
            # Single dictionary - convert to DataFrame with one row
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            if not data:
                raise ValidationError("Cannot create DataFrame from empty list")
            if not all(isinstance(item, dict) for item in data):
                raise ValidationError("All items in list must be dictionaries")
            df = pd.DataFrame(data)
        else:
            raise ValidationError(f"Invalid data type: {type(data)}")

        return df
    except Exception as e:
        raise ValidationError(f"Failed to convert to DataFrame: {e!s}") from e


def normalize_array(arr: list[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize array to [0, 1] range.

    Args:
        arr: Array to normalize

    Returns:
        Normalized numpy array

    Raises:
        ValidationError: If array is empty or invalid
    """
    if not arr or (isinstance(arr, np.ndarray) and arr.size == 0):
        raise ValidationError("Cannot normalize empty array")

    arr_np = np.array(arr) if not isinstance(arr, np.ndarray) else arr

    min_val = np.min(arr_np)
    max_val = np.max(arr_np)

    if min_val == max_val:
        # All values are the same
        return np.ones_like(arr_np) * 0.5

    normalized = (arr_np - min_val) / (max_val - min_val)
    return normalized


def convert_currency(
    amount: Decimal, from_currency: str, to_currency: str, exchange_rate: Decimal
) -> Decimal:
    """
    Convert amount from one currency to another.

    Args:
        amount: Amount to convert as Decimal
        from_currency: Source currency
        to_currency: Target currency
        exchange_rate: Exchange rate as Decimal (from_currency/to_currency)

    Returns:
        Converted amount as Decimal

    Raises:
        ValidationError: If amount is negative or exchange rate is invalid
    """
    amount_decimal = to_decimal(amount) if not isinstance(amount, Decimal) else amount
    rate_decimal = (
        to_decimal(exchange_rate) if not isinstance(exchange_rate, Decimal) else exchange_rate
    )

    if amount_decimal < ZERO:
        raise ValidationError("Amount cannot be negative")

    if rate_decimal <= ZERO:
        raise ValidationError("Exchange rate must be positive")

    converted_amount = amount_decimal * rate_decimal

    # Round to appropriate precision based on currency
    if to_currency.upper() in ["BTC", "ETH"]:
        precision = 8
    elif to_currency.upper() in ["USDT", "USDC", "USD"]:
        precision = 2
    else:
        precision = 4

    # Use Decimal quantize for precise rounding
    quantizer = Decimal(10) ** -precision
    return converted_amount.quantize(quantizer, rounding=ROUND_HALF_UP)


def normalize_price(price: float | Decimal, symbol: str) -> Decimal:
    """
    Normalize price to appropriate precision for a given symbol.

    Args:
        price: Price to normalize (accepts float or Decimal)
        symbol: Trading symbol

    Returns:
        Normalized price as Decimal

    Raises:
        ValidationError: If price is invalid
    """
    # Convert to Decimal for all operations
    decimal_price = to_decimal(price)

    if decimal_price <= ZERO:
        raise ValidationError(f"Price must be positive for {symbol}, got {decimal_price}")

    # Determine precision based on symbol
    if "BTC" in symbol.upper():
        precision = 8
    elif "ETH" in symbol.upper():
        precision = 6
    elif "USDT" in symbol.upper() or "USD" in symbol.upper():
        precision = 2
    else:
        precision = 4

    # Round to appropriate precision
    normalized_price = decimal_price.quantize(
        Decimal(f"0.{'0' * (precision - 1)}1"), rounding=ROUND_HALF_UP
    )

    return normalized_price


def round_to_precision(value: Decimal, precision: int) -> Decimal:
    """
    Round value to specified precision using Decimal for financial accuracy.

    Args:
        value: Value to round as Decimal
        precision: Number of decimal places

    Returns:
        Rounded Decimal value

    Raises:
        ValidationError: If precision is negative
    """
    if precision < 0:
        raise ValidationError("Precision must be non-negative")

    decimal_value = to_decimal(value) if not isinstance(value, Decimal) else value

    # Use Decimal's quantize method for precise rounding
    quantizer = Decimal(f"0.{'0' * (precision - 1)}1") if precision > 0 else Decimal("1")
    return decimal_value.quantize(quantizer, rounding=ROUND_HALF_UP)


def round_to_precision_decimal(value: Decimal, precision: int) -> Decimal:
    """
    Round Decimal value to specified precision.

    Args:
        value: Decimal value to round
        precision: Number of decimal places

    Returns:
        Rounded Decimal value

    Raises:
        ValidationError: If precision is negative
    """
    if precision < 0:
        raise ValidationError("Precision must be non-negative")

    # Use Decimal's quantize method for precise rounding
    factor = Decimal(f"0.{'0' * (precision - 1)}1") if precision > 0 else Decimal("1")
    return value.quantize(factor, rounding=ROUND_HALF_UP)


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for keys

    Returns:
        Flattened dictionary
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """
    Unflatten dictionary with dot notation keys.

    Args:
        d: Flattened dictionary
        sep: Separator used in keys

    Returns:
        Nested dictionary
    """
    result: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge multiple dictionaries.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    if not dicts:
        return {}

    result: dict[str, Any] = {}
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
    return result


def filter_none_values(d: dict[str, Any]) -> dict[str, Any]:
    """
    Remove None values from dictionary.

    Args:
        d: Dictionary to filter

    Returns:
        Dictionary without None values
    """
    return {k: v for k, v in d.items() if v is not None}


def chunk_list(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    """
    Split list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Raises:
        ValidationError: If chunk_size is invalid
    """
    if chunk_size <= 0:
        raise ValidationError("Chunk size must be positive")

    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
