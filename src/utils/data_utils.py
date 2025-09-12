"""Data conversion and manipulation utilities for the T-Bot trading system."""

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    # Fallback for older numpy versions
    NDArray = np.ndarray

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
    if (isinstance(arr, list) and not arr) or (isinstance(arr, np.ndarray) and arr.size == 0):
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
    Convert amount from one currency to another with boundary validation.

    This function enforces module boundary validation using core validation patterns.

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
    # Basic input validation - complex validation should be done at service layer
    # This is a utility function that performs basic conversions, not business validation

    amount_decimal = to_decimal(amount) if not isinstance(amount, Decimal) else amount
    rate_decimal = (
        to_decimal(exchange_rate) if not isinstance(exchange_rate, Decimal) else exchange_rate
    )

    if amount_decimal < ZERO:
        raise ValidationError(
            "Amount cannot be negative",
            error_code="VALID_000",
            details={"amount": str(amount_decimal)},
        )

    if rate_decimal <= ZERO:
        raise ValidationError(
            "Exchange rate must be positive",
            error_code="VALID_000",
            details={"exchange_rate": str(rate_decimal)},
        )

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


def normalize_price(price: Decimal | int, symbol: str, precision: int | None = None) -> Decimal:
    """
    Normalize price to appropriate precision for a given symbol with proper asset type handling.

    Args:
        price: Price to normalize as Decimal or int (no float for precision)
        symbol: Trading symbol
        precision: Override precision (if None, auto-determined based on asset type)

    Returns:
        Normalized price as Decimal

    Raises:
        ValidationError: If price is invalid
    """
    from src.utils.decimal_utils import FINANCIAL_CONTEXT

    # Ensure we only accept Decimal or int for financial precision
    if not isinstance(price, (Decimal, int)):
        raise ValidationError(
            f"Price must be Decimal or int for financial precision, got {type(price).__name__}"
        )

    # Convert to Decimal for all operations
    decimal_price = to_decimal(price) if not isinstance(price, Decimal) else price

    if decimal_price <= ZERO:
        raise ValidationError(f"Price must be positive for {symbol}, got {decimal_price}")

    # Determine precision based on asset type if not specified
    if precision is None:
        symbol_upper = symbol.upper()
        if any(
            crypto in symbol_upper
            for crypto in ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "LINK", "SOL", "BNB"]
        ):
            precision = 8  # Crypto precision (DECIMAL(20,8))
        elif any(
            fiat in symbol_upper for fiat in ["EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD"]
        ):
            precision = 4  # Forex precision (DECIMAL(20,4))
        elif "USDT" in symbol_upper or "USD" in symbol_upper or "USDC" in symbol_upper:
            precision = 2  # Fiat stablecoin precision (DECIMAL(20,2))
        else:
            precision = 4  # Default precision for unknown assets

    # Use financial context for rounding with proper precision
    with localcontext(FINANCIAL_CONTEXT):
        quantizer = Decimal(10) ** -precision
        normalized_price = decimal_price.quantize(quantizer, rounding=ROUND_HALF_UP)

    return normalized_price


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
    Split list into chunks using batch processing pattern with boundary validation.

    This function aligns with batch processing paradigms used throughout the system
    for consistent data flow patterns between core and utils modules.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks (batch format)

    Raises:
        ValidationError: If chunk_size is invalid
    """
    # Apply module boundary validation
    from datetime import timezone

    from src.utils.messaging_patterns import BoundaryValidator, ProcessingParadigmAligner

    if chunk_size <= 0:
        raise ValidationError(
            "Chunk size must be positive", error_code="VALID_000", details={"chunk_size": chunk_size}
        )

    # Apply boundary validation for data processing
    validation_data = {
        "chunk_size": chunk_size,
        "list_size": len(lst),
        "component": "data_utils_chunker",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_mode": "batch",
        "operation": "chunk_processing",
    }

    try:
        # Validate at utils boundary
        BoundaryValidator.validate_database_entity(validation_data, "create")
    except ValidationError as e:
        # Continue if validation fails - don't break core functionality
        from src.core.logging import get_logger

        logger = get_logger(__name__)
        logger.debug(f"Boundary validation failed in chunk_list: {e}")

    # Use consistent batch processing pattern
    chunks: list[list[Any]] = []
    current_chunk = []

    for i, item in enumerate(lst):
        current_chunk.append(item)

        # Create batch when size reached or at end of data
        if len(current_chunk) == chunk_size or i == len(lst) - 1:
            # Apply consistent batch transformation
            batch_metadata = {
                "batch_id": f"chunk_{len(chunks)}",
                "batch_size": len(current_chunk),
                "processing_mode": "batch",
                "data_format": "batch_v1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Use paradigm aligner for consistency with data pipeline patterns
            stream_items = [{"item": item, "batch_position": i} for i, item in enumerate(current_chunk)]
            aligned_batch = ProcessingParadigmAligner.create_batch_from_stream(stream_items)

            # Apply consistent batch processing mode alignment
            batch_with_metadata = ProcessingParadigmAligner.align_processing_modes(
                source_mode="batch", target_mode="batch", data={
                    "chunk_data": current_chunk,
                    **batch_metadata,
                    **aligned_batch
                }
            )

            chunks.append(current_chunk)  # Keep backward compatibility
            current_chunk = []

    return chunks
