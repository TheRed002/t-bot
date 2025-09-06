"""
Utility functions for backtesting module.

Common functions used across backtesting components.
"""

from typing import Any

import pandas as pd

from src.core.dependency_injection import DependencyInjector


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
    try:
        for record in records:
            if not hasattr(record, "close_price"):
                raise AttributeError(f"Record missing required close_price attribute: {type(record)}")

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
    except AttributeError as e:
        from src.core.logging import get_logger

        logger = get_logger(__name__)
        logger.error(f"Failed to convert market records: {e}")
        raise
    except Exception as e:
        from src.core.logging import get_logger

        logger = get_logger(__name__)
        logger.error(f"Unexpected error converting market records: {e}")
        raise


def get_backtest_engine_factory(injector: DependencyInjector) -> Any:
    """
    Service locator pattern for BacktestEngine factory.

    Args:
        injector: Dependency injector instance

    Returns:
        BacktestEngine factory function
    """
    return injector.resolve("BacktestEngineFactory")


def create_component_with_factory(injector: DependencyInjector, component_name: str) -> Any:
    """
    Service locator pattern for creating components via factories.

    Args:
        injector: Dependency injector instance
        component_name: Name of component to create

    Returns:
        Created component instance
    """
    factory_name = f"{component_name}Factory" if not component_name.endswith("Factory") else component_name
    return injector.resolve(factory_name)
