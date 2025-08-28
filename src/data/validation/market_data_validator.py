"""
Market Data Validator Implementation

Concrete implementation of DataValidatorInterface for market data validation.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.base.component import BaseComponent
from src.core.types import MarketData
from src.data.interfaces import DataValidatorInterface
from src.utils.validators import validate_decimal_precision, validate_market_data


class MarketDataValidator(BaseComponent, DataValidatorInterface):
    """Market data validator implementation."""

    def __init__(self):
        """Initialize market data validator."""
        super().__init__()
        self._validation_errors: list[str] = []

    async def validate_market_data(self, data_list: list[MarketData]) -> list[MarketData]:
        """Validate market data with comprehensive checks."""
        self._validation_errors.clear()
        valid_data = []

        for i, data in enumerate(data_list):
            try:
                # Basic validation
                if not validate_market_data(data.model_dump()):
                    self._validation_errors.append(f"Invalid market data structure for {data.symbol} at index {i}")
                    continue

                # Decimal precision validation for financial data
                if data.price and not validate_decimal_precision(float(data.price), places=8):
                    self._validation_errors.append(f"Invalid price precision for {data.symbol} at index {i}")
                    continue

                # Volume validation
                if data.volume and data.volume < 0:
                    self._validation_errors.append(f"Invalid negative volume for {data.symbol} at index {i}")
                    continue

                # Timestamp validation
                if data.timestamp:
                    now = datetime.now(timezone.utc)
                    if data.timestamp > now + timedelta(minutes=5):
                        self._validation_errors.append(f"Future timestamp for {data.symbol} at index {i}")
                        continue

                # Price consistency checks
                if (data.high_price and data.low_price and 
                    data.high_price < data.low_price):
                    self._validation_errors.append(f"High price less than low price for {data.symbol} at index {i}")
                    continue

                if (data.price and data.high_price and 
                    data.price > data.high_price):
                    self._validation_errors.append(f"Price higher than high price for {data.symbol} at index {i}")
                    continue

                if (data.price and data.low_price and 
                    data.price < data.low_price):
                    self._validation_errors.append(f"Price lower than low price for {data.symbol} at index {i}")
                    continue

                # Bid/Ask validation
                if (data.bid and data.ask and data.bid > data.ask):
                    self._validation_errors.append(f"Bid higher than ask for {data.symbol} at index {i}")
                    continue

                valid_data.append(data)

            except Exception as e:
                self._validation_errors.append(f"Validation error for {data.symbol} at index {i}: {e}")
                continue

        if self._validation_errors:
            self.logger.warning(f"Validation found {len(self._validation_errors)} errors in {len(data_list)} records")

        return valid_data

    def get_validation_errors(self) -> list[str]:
        """Get validation errors from last validation."""
        return self._validation_errors.copy()

    async def health_check(self) -> dict[str, Any]:
        """Perform validator health check."""
        return {
            "status": "healthy",
            "component": "market_data_validator",
            "last_error_count": len(self._validation_errors),
        }