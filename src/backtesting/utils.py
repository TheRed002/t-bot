"""
Utility functions for backtesting module.

Common functions used across backtesting components.
"""

from typing import Any

import pandas as pd


def convert_market_records_to_dataframe(records: list[Any]) -> pd.DataFrame:
    """
    Convert market data records to pandas DataFrame.

    Args:
        records: List of market data records with price fields

    Returns:
        DataFrame with OHLCV data indexed by timestamp
    """
    if not records:
        return pd.DataFrame()

    # Convert to DataFrame
    df_data = []
    for record in records:
        df_data.append(
            {
                "timestamp": getattr(record, "data_timestamp", getattr(record, "timestamp", None)),
                "open": record.open_price or record.close_price,
                "high": record.high_price or record.close_price,
                "low": record.low_price or record.close_price,
                "close": record.close_price,
                "volume": record.volume or 0,
            }
        )

    df = pd.DataFrame(df_data)
    if not df.empty:
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()

    return df
