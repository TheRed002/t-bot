"""
Unit tests for data_utils module.

Tests data utility functions for conversion and processing.
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.utils.data_utils import (
    dict_to_dataframe,
    normalize_array, 
    convert_currency,
    normalize_price,
    flatten_dict,
    unflatten_dict,
    merge_dicts,
    filter_none_values,
    chunk_list,
)
from src.core.exceptions import ValidationError


class TestDictToDataframe:
    """Test dict_to_dataframe function."""

    def test_single_dict_to_dataframe(self):
        """Test converting single dictionary to DataFrame."""
        data = {"price": 100.0, "volume": 1000, "symbol": "BTC"}
        result = dict_to_dataframe(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "price" in result.columns
        assert "volume" in result.columns
        assert "symbol" in result.columns
        assert result.iloc[0]["price"] == 100.0

    def test_list_of_dicts_to_dataframe(self):
        """Test converting list of dictionaries to DataFrame."""
        data = [
            {"price": 100.0, "volume": 1000, "symbol": "BTC"},
            {"price": 200.0, "volume": 2000, "symbol": "ETH"}
        ]
        result = dict_to_dataframe(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result.iloc[0]["price"] == 100.0
        assert result.iloc[1]["price"] == 200.0

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValidationError."""
        with pytest.raises(ValidationError, match="Cannot create DataFrame from empty list"):
            dict_to_dataframe([])

    def test_invalid_data_type_raises_error(self):
        """Test that invalid data type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid data type"):
            dict_to_dataframe("invalid")

    def test_list_with_non_dicts_raises_error(self):
        """Test that list with non-dictionaries raises ValidationError."""
        with pytest.raises(ValidationError, match="All items in list must be dictionaries"):
            dict_to_dataframe([{"valid": "dict"}, "invalid_item"])


class TestNormalizeArray:
    """Test normalize_array function."""

    def test_normalize_list(self):
        """Test normalizing list of numbers."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_array(arr)
        
        assert isinstance(result, np.ndarray)
        assert np.isclose(result.min(), 0.0)
        assert np.isclose(result.max(), 1.0)
        assert len(result) == len(arr)

    def test_normalize_numpy_array(self):
        """Test normalizing numpy array."""
        arr = np.array([10, 20, 30, 40, 50])
        result = normalize_array(arr)
        
        assert isinstance(result, np.ndarray)
        assert np.isclose(result.min(), 0.0)
        assert np.isclose(result.max(), 1.0)

    def test_normalize_identical_values(self):
        """Test normalizing array with identical values."""
        arr = [5.0, 5.0, 5.0, 5.0]
        result = normalize_array(arr)
        
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, 0.5)

    def test_normalize_empty_array_raises_error(self):
        """Test that empty array raises ValidationError."""
        with pytest.raises(ValidationError, match="Cannot normalize empty array"):
            normalize_array([])

    def test_normalize_empty_numpy_array_raises_error(self):
        """Test that empty numpy array raises ValidationError."""
        with pytest.raises(ValidationError, match="Cannot normalize empty array"):
            normalize_array(np.array([]))


class TestConvertCurrency:
    """Test convert_currency function."""

    @patch('src.utils.data_flow_integrity.validate_cross_module_data')
    def test_convert_currency_basic(self, mock_validate):
        """Test basic currency conversion."""
        mock_validate.return_value = {
            "amount": Decimal("100"),
            "from_currency": "USD", 
            "to_currency": "EUR",
            "exchange_rate": Decimal("0.85")
        }
        
        result = convert_currency(
            Decimal("100"), "USD", "EUR", Decimal("0.85")
        )
        
        assert isinstance(result, Decimal)
        assert result == Decimal("85.00")  # 100 * 0.85, rounded to 2 decimal places for EUR

    @patch('src.utils.data_flow_integrity.validate_cross_module_data')
    def test_convert_to_crypto_precision(self, mock_validate):
        """Test conversion to crypto with 8 decimal precision."""
        mock_validate.return_value = {
            "amount": Decimal("1000"),
            "from_currency": "USD",
            "to_currency": "BTC", 
            "exchange_rate": Decimal("0.00002")
        }
        
        result = convert_currency(
            Decimal("1000"), "USD", "BTC", Decimal("0.00002")
        )
        
        assert isinstance(result, Decimal)
        # Should have 8 decimal places for BTC
        assert str(result).count('.') == 1
        decimal_places = len(str(result).split('.')[1])
        assert decimal_places <= 8

    @patch('src.utils.data_flow_integrity.validate_cross_module_data')
    def test_convert_negative_amount_raises_error(self, mock_validate):
        """Test that negative amount raises ValidationError."""
        mock_validate.return_value = {
            "amount": Decimal("-100"),
            "from_currency": "USD",
            "to_currency": "EUR",
            "exchange_rate": Decimal("0.85")
        }
        
        with pytest.raises(ValidationError, match="Amount cannot be negative"):
            convert_currency(Decimal("-100"), "USD", "EUR", Decimal("0.85"))

    @patch('src.utils.data_flow_integrity.validate_cross_module_data')  
    def test_convert_zero_rate_raises_error(self, mock_validate):
        """Test that zero exchange rate raises ValidationError."""
        mock_validate.return_value = {
            "amount": Decimal("100"),
            "from_currency": "USD",
            "to_currency": "EUR", 
            "exchange_rate": Decimal("0")
        }
        
        with pytest.raises(ValidationError, match="Exchange rate must be positive"):
            convert_currency(Decimal("100"), "USD", "EUR", Decimal("0"))


class TestNormalizePrice:
    """Test normalize_price function."""

    def test_normalize_btc_price(self):
        """Test normalizing BTC price with 8 decimal precision."""
        price = Decimal("45123.12345678")
        result = normalize_price(price, "BTCUSD")
        
        assert isinstance(result, Decimal)
        # Should maintain 8 decimal places
        decimal_places = len(str(result).split('.')[1])
        assert decimal_places <= 8

    def test_normalize_usd_price(self):
        """Test normalizing USD price with 2 decimal precision.""" 
        price = Decimal("123.456789")
        result = normalize_price(price, "EURUSD")
        
        assert isinstance(result, Decimal)
        # Should be rounded to 2 decimal places for USD  
        assert abs(result - Decimal("123.46")) < Decimal("0.01")

    def test_normalize_zero_price_raises_error(self):
        """Test that zero price raises ValidationError."""
        with pytest.raises(ValidationError, match="Price must be positive"):
            normalize_price(Decimal("0"), "BTCUSD")

    def test_normalize_negative_price_raises_error(self):
        """Test that negative price raises ValidationError."""
        with pytest.raises(ValidationError, match="Price must be positive"):
            normalize_price(Decimal("-100"), "BTCUSD")

    def test_normalize_float_raises_error(self):
        """Test that float input raises ValidationError for precision."""
        with pytest.raises(ValidationError, match="Price must be Decimal or int for financial precision"):
            normalize_price(123.45, "BTCUSD")

    def test_normalize_with_custom_precision(self):
        """Test normalizing with custom precision."""
        price = Decimal("123.123456")
        result = normalize_price(price, "CUSTOMSYMBOL", precision=4)
        
        assert isinstance(result, Decimal)
        decimal_places = len(str(result).split('.')[1])
        assert decimal_places == 4


class TestFlattenDict:
    """Test flatten_dict function."""

    def test_flatten_simple_dict(self):
        """Test flattening simple nested dictionary."""
        nested = {
            "level1": {
                "level2": {
                    "value": 42
                }
            },
            "simple": "test"
        }
        
        result = flatten_dict(nested)
        assert result["level1.level2.value"] == 42
        assert result["simple"] == "test"

    def test_flatten_with_custom_separator(self):
        """Test flattening with custom separator.""" 
        nested = {"a": {"b": {"c": 1}}}
        result = flatten_dict(nested, sep="_")
        assert "a_b_c" in result
        assert result["a_b_c"] == 1

    def test_flatten_empty_dict(self):
        """Test flattening empty dictionary."""
        result = flatten_dict({})
        assert result == {}

    def test_flatten_already_flat(self):
        """Test flattening already flat dictionary."""
        flat = {"key1": "value1", "key2": "value2"}
        result = flatten_dict(flat)
        assert result == flat


class TestUnflattenDict:
    """Test unflatten_dict function."""

    def test_unflatten_simple(self):
        """Test unflattening simple flat dictionary."""
        flat = {"a.b.c": 1, "a.b.d": 2, "e": 3}
        result = unflatten_dict(flat)
        
        assert result["a"]["b"]["c"] == 1
        assert result["a"]["b"]["d"] == 2  
        assert result["e"] == 3

    def test_unflatten_with_custom_separator(self):
        """Test unflattening with custom separator."""
        flat = {"a_b_c": 1}
        result = unflatten_dict(flat, sep="_")
        assert result["a"]["b"]["c"] == 1

    def test_unflatten_empty_dict(self):
        """Test unflattening empty dictionary."""
        result = unflatten_dict({})
        assert result == {}

    def test_unflatten_no_nesting(self):
        """Test unflattening dictionary with no nested keys."""
        flat = {"key1": "value1", "key2": "value2"}
        result = unflatten_dict(flat)
        assert result == flat


class TestMergeDicts:
    """Test merge_dicts function."""

    def test_merge_simple_dicts(self):
        """Test merging simple dictionaries."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        result = merge_dicts(dict1, dict2)
        
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_nested_dicts(self):
        """Test deep merging nested dictionaries."""
        dict1 = {"nested": {"a": 1, "b": 2}}
        dict2 = {"nested": {"c": 3, "d": 4}}
        result = merge_dicts(dict1, dict2)
        
        assert result["nested"]["a"] == 1
        assert result["nested"]["c"] == 3

    def test_merge_overlapping_keys(self):
        """Test merging with overlapping keys."""
        dict1 = {"key": "value1"}
        dict2 = {"key": "value2"}
        result = merge_dicts(dict1, dict2)
        
        # Second dict should override first
        assert result["key"] == "value2"

    def test_merge_no_dicts(self):
        """Test merging with no dictionaries."""
        result = merge_dicts()
        assert result == {}

    def test_merge_single_dict(self):
        """Test merging single dictionary."""
        single = {"a": 1}
        result = merge_dicts(single)
        assert result == single


class TestFilterNoneValues:
    """Test filter_none_values function."""

    def test_filter_with_none_values(self):
        """Test filtering dictionary with None values."""
        data = {"a": 1, "b": None, "c": 3, "d": None}
        result = filter_none_values(data)
        
        assert result == {"a": 1, "c": 3}
        assert "b" not in result
        assert "d" not in result

    def test_filter_no_none_values(self):
        """Test filtering dictionary without None values."""
        data = {"a": 1, "b": 2, "c": 3}
        result = filter_none_values(data)
        assert result == data

    def test_filter_all_none_values(self):
        """Test filtering dictionary with all None values."""
        data = {"a": None, "b": None}
        result = filter_none_values(data)
        assert result == {}

    def test_filter_empty_dict(self):
        """Test filtering empty dictionary."""
        result = filter_none_values({})
        assert result == {}

    def test_filter_preserves_falsy_values(self):
        """Test that falsy non-None values are preserved."""
        data = {"a": 0, "b": "", "c": [], "d": None, "e": False}
        result = filter_none_values(data)
        
        assert result["a"] == 0
        assert result["b"] == ""
        assert result["c"] == []
        assert result["e"] is False
        assert "d" not in result


class TestChunkList:
    """Test chunk_list function."""

    def test_chunk_basic_list(self):
        """Test chunking basic list."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        result = chunk_list(data, 3)
        
        assert len(result) == 3  # 3 chunks
        assert result[0] == [1, 2, 3]
        assert result[1] == [4, 5, 6] 
        assert result[2] == [7, 8]  # Last chunk has remaining items

    def test_chunk_exact_division(self):
        """Test chunking with exact division."""
        data = [1, 2, 3, 4, 5, 6]
        result = chunk_list(data, 2)
        
        assert len(result) == 3
        assert result[0] == [1, 2]
        assert result[1] == [3, 4]
        assert result[2] == [5, 6]

    def test_chunk_larger_than_list(self):
        """Test chunk size larger than list."""
        data = [1, 2, 3]
        result = chunk_list(data, 10)
        
        assert len(result) == 1
        assert result[0] == [1, 2, 3]

    def test_chunk_empty_list(self):
        """Test chunking empty list."""
        result = chunk_list([], 5)
        assert result == []

    def test_chunk_invalid_size_raises_error(self):
        """Test that invalid chunk size raises ValidationError.""" 
        with pytest.raises(ValidationError, match="Chunk size must be positive"):
            chunk_list([1, 2, 3], 0)

        with pytest.raises(ValidationError, match="Chunk size must be positive"):
            chunk_list([1, 2, 3], -1)

    def test_chunk_single_item_chunks(self):
        """Test chunking with size 1."""
        data = [1, 2, 3]
        result = chunk_list(data, 1)
        
        assert len(result) == 3
        assert result[0] == [1]
        assert result[1] == [2] 
        assert result[2] == [3]


class TestDataUtilsIntegration:
    """Test integration between different data utility functions."""

    def test_flatten_unflatten_roundtrip(self):
        """Test that flatten/unflatten are inverse operations."""
        original = {
            "trading": {
                "symbols": ["BTC", "ETH"], 
                "config": {
                    "max_risk": 0.02
                }
            },
            "simple": "value"
        }
        
        flattened = flatten_dict(original)
        unflattened = unflatten_dict(flattened)
        
        assert unflattened == original

    def test_merge_filter_workflow(self):
        """Test merging and filtering workflow."""
        dict1 = {"a": 1, "b": None, "c": {"nested": 2}}
        dict2 = {"b": 3, "d": None, "c": {"other": 4}}
        
        merged = merge_dicts(dict1, dict2)
        filtered = filter_none_values(merged)
        
        assert "d" not in filtered  # None value filtered out
        assert filtered["b"] == 3   # Merged value
        assert "nested" in str(filtered)  # Nested preserved

    def test_dataframe_conversion_workflow(self):
        """Test DataFrame conversion with processed data."""
        raw_data = [
            {"price": 100, "volume": None, "symbol": "BTC"},
            {"price": 200, "volume": 2000, "symbol": "ETH"}
        ]
        
        # Filter None values from each dict
        cleaned_data = [filter_none_values(d) for d in raw_data]
        df = dict_to_dataframe(cleaned_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # First row should not have volume column or it should be NaN
        assert "symbol" in df.columns