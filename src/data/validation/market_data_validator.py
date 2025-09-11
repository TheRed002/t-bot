"""
Market Data Validator Implementation

Refactored to use consolidated validation utilities from src.utils.validation.
"""

from typing import Any

from src.core import BaseComponent, MarketData
from src.data.interfaces import DataValidatorInterface, ServiceDataValidatorInterface
from src.utils.validation.market_data_validation import MarketDataValidator as ConsolidatedValidator


class MarketDataValidator(BaseComponent, DataValidatorInterface, ServiceDataValidatorInterface):
    """
    Market data validator implementation that uses consolidated validation utilities.

    This class serves as a bridge between the data module interfaces and the
    consolidated validation utilities, eliminating code duplication.
    """

    def __init__(self):
        """Initialize market data validator."""
        super().__init__()
        # Use consolidated validator with default settings
        self._validator = ConsolidatedValidator(
            enable_precision_validation=True,
            enable_consistency_validation=True,
            enable_timestamp_validation=True,
        )

    async def validate_market_data(self, data_list: list[MarketData]) -> list[MarketData]:
        """Validate market data with comprehensive checks using consolidated validator."""
        try:
            # Use consolidated validator for batch validation
            valid_data = self._validator.validate_market_data_batch(data_list)

            if len(valid_data) < len(data_list):
                invalid_count = len(data_list) - len(valid_data)
                self.logger.warning(
                    f"Validation found {invalid_count} invalid records out of "
                    f"{len(data_list)} total"
                )

            return valid_data

        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            return []

    def get_validation_errors(self) -> list[str]:
        """Get validation errors from last validation."""
        return self._validator.get_validation_errors()

    def validate(self, data) -> bool:
        """Validate single data item."""
        try:
            if isinstance(data, dict):
                # Convert dict to MarketData for validation
                from src.core.types import MarketData as MD

                market_data = MD(**data)
                return self._validator.validate_market_data_record(market_data)
            elif isinstance(data, MarketData):
                return self._validator.validate_market_data_record(data)
            elif hasattr(data, "model_dump"):
                # Handle Pydantic models
                market_data = MarketData(**data.model_dump())
                return self._validator.validate_market_data_record(market_data)
            else:
                return False
        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            return False

    def get_errors(self) -> list[str]:
        """Get validation errors from last validation."""
        return self._validator.get_validation_errors()

    def reset(self) -> None:
        """Reset validator state."""
        self._validator.reset()

    async def health_check(self) -> dict[str, Any]:
        """Perform validator health check."""
        errors = self._validator.get_validation_errors()
        error_count = len(errors)
        status = "healthy" if error_count == 0 else "degraded"

        return {
            "status": status,
            "component": "market_data_validator",
            "last_error_count": error_count,
            "message": f"Market data validator: {status}",
            "validator_type": "consolidated",
        }
