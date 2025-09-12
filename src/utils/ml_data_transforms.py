"""
ML Data Transformation Utilities.

This module provides common data transformation functions to eliminate
duplicate transformation code across the ML module.
"""

from typing import Any

import pandas as pd

from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal

logger = get_logger(__name__)


def transform_market_data_to_decimal(data: dict[str, Any] | pd.DataFrame) -> dict[str, Any] | pd.DataFrame:
    """
    Transform market data fields to Decimal for financial precision.
    
    Args:
        data: Market data as dict or DataFrame
        
    Returns:
        Data with financial fields converted to Decimal
    """
    # Define financial field names that need Decimal conversion
    financial_fields = {
        "price", "close", "open", "high", "low",
        "open_price", "high_price", "low_price", "close_price",
        "volume", "bid", "ask"
    }

    if isinstance(data, dict):
        transformed_data = {}
        for key, value in data.items():
            if key in financial_fields and value is not None:
                transformed_data[key] = to_decimal(value)
            else:
                transformed_data[key] = value
        return transformed_data

    elif isinstance(data, pd.DataFrame):
        data_copy = data.copy()
        for col in data_copy.columns:
            if col in financial_fields:
                data_copy[col] = data_copy[col].apply(
                    lambda x: to_decimal(x) if pd.notna(x) else x
                )
        return data_copy

    else:
        return data


def convert_pydantic_to_dict_with_decimals(data: Any) -> dict[str, Any]:
    """
    Convert pydantic model to dict with financial fields as Decimals.
    
    Args:
        data: Pydantic model instance
        
    Returns:
        Dictionary with financial fields converted to Decimal
    """
    if not hasattr(data, "model_dump"):
        return data

    data_dict = data.model_dump()
    return transform_market_data_to_decimal(data_dict)


def prepare_dataframe_from_market_data(market_data: dict[str, Any] | Any, apply_decimal_transform: bool = True) -> pd.DataFrame:
    """
    Convert market data to DataFrame with consistent transformations.
    
    Args:
        market_data: Market data as dict, DataFrame, or pydantic model
        apply_decimal_transform: Whether to apply decimal transformations
        
    Returns:
        DataFrame ready for ML processing
    """
    if isinstance(market_data, dict):
        if apply_decimal_transform:
            transformed_data = transform_market_data_to_decimal(market_data)
        else:
            transformed_data = market_data
        return pd.DataFrame([transformed_data]) if not isinstance(transformed_data, pd.DataFrame) else transformed_data

    elif hasattr(market_data, "model_dump"):
        data_dict = convert_pydantic_to_dict_with_decimals(market_data)
        return pd.DataFrame([data_dict])

    elif isinstance(market_data, pd.DataFrame):
        if apply_decimal_transform:
            return transform_market_data_to_decimal(market_data)
        return market_data

    else:
        # Fallback - try to convert to DataFrame
        return pd.DataFrame([market_data])


def align_training_data_lengths(X: pd.DataFrame, y: pd.Series, model_name: str = "Unknown") -> tuple[pd.DataFrame, pd.Series]:
    """
    Align feature and target data to same length after cleaning.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        model_name: Model name for logging
        
    Returns:
        Aligned feature and target data
    """
    if len(X) != len(y):
        min_len = min(len(X), len(y))
        X_aligned = X.iloc[:min_len]
        y_aligned = y.iloc[:min_len]
        logger.warning(f"{model_name}: Aligned data to {min_len} samples after cleaning")
        return X_aligned, y_aligned

    return X, y


def create_returns_series(prices: pd.Series, horizon: int = 1, return_type: str = "simple") -> pd.Series:
    """
    Create returns series from prices with Decimal precision.
    
    Args:
        prices: Price series
        horizon: Forward looking periods
        return_type: Type of return ('simple', 'log')
        
    Returns:
        Returns series
    """
    from decimal import Decimal

    import numpy as np

    future_prices = prices.shift(-horizon)
    returns = pd.Series(index=prices.index, dtype=float)

    for i in range(len(prices)):
        if (
            pd.notna(prices.iloc[i])
            and pd.notna(future_prices.iloc[i])
            and prices.iloc[i] != 0
        ):
            price_decimal = Decimal(str(prices.iloc[i]))
            future_price_decimal = Decimal(str(future_prices.iloc[i]))

            if return_type == "simple":
                return_decimal = (future_price_decimal / price_decimal) - Decimal("1")
                returns.iloc[i] = float(return_decimal)
            elif return_type == "log":
                ratio = future_price_decimal / price_decimal
                returns.iloc[i] = float(np.log(float(ratio)))
            else:
                raise ValueError(f"Unknown return type: {return_type}")
        else:
            returns.iloc[i] = float("nan")

    # Remove last N values that don't have future data
    returns = returns.iloc[:-horizon]

    return returns


def batch_transform_requests_to_aligned_format(requests: list[Any]) -> list[dict[str, Any]]:
    """
    Transform batch requests to aligned format for consistent processing.
    
    Args:
        requests: List of request objects
        
    Returns:
        List of aligned request dictionaries
    """
    aligned_requests = []

    for request in requests:
        # Apply consistent data transformation aligned with data module
        market_data = request.market_data
        if hasattr(request.market_data, "model_dump"):
            market_data = request.market_data.model_dump()
        elif isinstance(request.market_data, dict):
            market_data = transform_market_data_to_decimal(request.market_data)

        request_dict = {
            "request_id": request.request_id,
            "symbol": request.symbol,
            "model_id": getattr(request, "model_id", None),
            "model_name": getattr(request, "model_name", None),
            "market_data": market_data,
        }
        aligned_requests.append(request_dict)

    return aligned_requests
