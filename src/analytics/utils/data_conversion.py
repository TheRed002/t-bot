"""Data conversion utilities for analytics module."""

import json
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Any

from src.core.base import BaseComponent


class DataConverter(BaseComponent):
    """Centralized data conversion utilities for analytics."""

    def __init__(self):
        super().__init__()

        # Set decimal precision context for financial calculations
        getcontext().prec = 28
        getcontext().rounding = ROUND_HALF_UP

    def convert_decimals_to_float(
        self, data: dict[str, Any], exclude_keys: set[str] | None = None
    ) -> dict[str, Any]:
        """Convert Decimal values to strings in a dictionary for precision preservation.

        Args:
            data: Dictionary containing potential Decimal values
            exclude_keys: Keys to exclude from conversion

        Returns:
            Dictionary with Decimal values converted to strings
        """
        if exclude_keys is None:
            exclude_keys = set()

        # Use unified conversion method with exclusion handling
        converted_data = self._convert_decimals(data, conversion_type="string")

        # Apply exclusions by restoring original values
        if exclude_keys:
            for key in exclude_keys:
                if key in data:
                    converted_data[key] = data[key]

        return converted_data

    def prepare_for_json_export(self, data: Any, remove_metadata: bool = False) -> dict[str, Any]:
        """Prepare data for JSON export with common transformations.

        Args:
            data: Data to prepare (dict, Pydantic model, or other)
            remove_metadata: Whether to remove metadata fields

        Returns:
            Dictionary ready for JSON serialization
        """
        # Convert Pydantic models to dict
        if hasattr(data, "dict") and callable(data.dict):
            prepared_data = data.dict()
        elif isinstance(data, dict):
            prepared_data = data.copy()
        else:
            prepared_data = {"data": data}

        # Remove metadata if requested
        if remove_metadata:
            prepared_data.pop("metadata", None)
            prepared_data.pop("_metadata", None)

        # Convert Decimals to strings for precision
        return self.convert_decimals_to_float(prepared_data)

    def json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for common types."""
        if isinstance(obj, Decimal):
            # Use string representation to maintain precision
            return str(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def safe_json_dumps(self, data: Any, **kwargs) -> str:
        """Safely serialize data to JSON using custom serializer."""
        return json.dumps(data, default=self.json_serializer, **kwargs)

    def convert_decimals_for_json(
        self, data: dict[str, Any], use_float: bool = True
    ) -> dict[str, Any]:
        """Convert Decimal values for JSON serialization.

        Args:
            data: Dictionary containing potential Decimal values
            use_float: If True, convert to float; if False, convert to string for precision

        Returns:
            Dictionary with converted Decimal values
        """
        # Delegate to existing method with appropriate conversion type
        if use_float:
            return self._convert_decimals(data, conversion_type="float")
        else:
            return self.convert_decimals_to_float(data)

    def _convert_decimals(
        self, data: dict[str, Any], conversion_type: str = "string"
    ) -> dict[str, Any]:
        """Unified decimal conversion method.

        Args:
            data: Dictionary containing potential Decimal values
            conversion_type: "string" for precision, "float" for API compatibility

        Returns:
            Dictionary with converted values
        """
        converted_data = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                if conversion_type == "float":
                    converted_data[key] = float(value)
                else:
                    converted_data[key] = str(value)
            elif isinstance(value, dict):
                converted_data[key] = self._convert_decimals(value, conversion_type)
            elif isinstance(value, list):
                converted_data[key] = self._convert_list_decimals_unified(value, conversion_type)
            else:
                converted_data[key] = value
        return converted_data

    def _convert_list_decimals_unified(
        self, data_list: list[Any], conversion_type: str
    ) -> list[Any]:
        """Unified list decimal conversion method."""
        converted_list: list[Any] = []
        for item in data_list:
            if isinstance(item, Decimal):
                if conversion_type == "float":
                    converted_list.append(float(item))
                else:
                    converted_list.append(str(item))
            elif isinstance(item, dict):
                converted_list.append(self._convert_decimals(item, conversion_type))
            elif isinstance(item, list):
                converted_list.append(self._convert_list_decimals_unified(item, conversion_type))
            else:
                converted_list.append(item)
        return converted_list
