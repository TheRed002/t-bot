"""Tests for analytics utils modules to increase coverage."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch
import json

from src.analytics.utils.data_conversion import DataConverter
from src.analytics.utils.validation import ValidationHelper
from src.core.types import Position, Trade
from src.core.exceptions import ValidationError


class TestDataConversionUtils:
    """Test data conversion utilities."""

    @pytest.fixture
    def converter(self):
        """Create DataConverter instance."""
        return DataConverter()

    def test_convert_decimals_to_float(self, converter):
        """Test convert_decimals_to_float method."""
        data = {
            'price': Decimal('100.50'),
            'volume': Decimal('10.25'),
            'text': 'hello'
        }
        
        result = converter.convert_decimals_to_float(data)
        
        assert isinstance(result, dict)
        assert result['price'] == '100.50'  # Converted to string for precision
        assert result['volume'] == '10.25'
        assert result['text'] == 'hello'

    def test_prepare_for_json_export(self, converter):
        """Test prepare_for_json_export method."""
        data = {
            'price': Decimal('50000'),
            'symbol': 'BTC/USDT',
            'quantity': Decimal('1.5')
        }
        
        result = converter.prepare_for_json_export(data)
        
        assert isinstance(result, dict)
        assert 'data_format' in result
        assert 'module' in result
        assert result['module'] == 'analytics'

    def test_json_serializer(self, converter):
        """Test json_serializer method."""
        # Test Decimal serialization
        assert converter.json_serializer(Decimal('100.50')) == '100.50'
        
        # Test datetime serialization
        dt = datetime(2024, 1, 1, 12, 0, 0)
        assert converter.json_serializer(dt) == dt.isoformat()
        
        # Test unsupported type
        with pytest.raises(TypeError):
            converter.json_serializer(set())

    def test_safe_json_dumps(self, converter):
        """Test safe_json_dumps method."""
        data = {
            'price': Decimal('50000'),
            'timestamp': datetime(2024, 1, 1),
            'symbol': 'BTC/USDT'
        }
        
        result = converter.safe_json_dumps(data)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed['price'] == '50000'
        assert parsed['symbol'] == 'BTC/USDT'

    def test_convert_decimals_for_json(self, converter):
        """Test convert_decimals_for_json method."""
        data = {
            'price': Decimal('100.50'),
            'nested': {
                'value': Decimal('200.75')
            },
            'list': [Decimal('1.1'), Decimal('2.2')]
        }
        
        # Test with use_float=True
        result_float = converter.convert_decimals_for_json(data, use_float=True)
        assert result_float['price'] == 100.5
        assert result_float['nested']['value'] == 200.75
        
        # Test with use_float=False (string conversion)
        result_string = converter.convert_decimals_for_json(data, use_float=False)
        assert result_string['price'] == '100.50'
        assert result_string['nested']['value'] == '200.75'

    def test_convert_decimals_with_exclusions(self, converter):
        """Test decimal conversion with exclusions."""
        data = {
            'price': Decimal('100.50'),
            'keep_as_decimal': Decimal('999.99'),
            'volume': Decimal('10')
        }
        
        result = converter.convert_decimals_for_json(
            data, 
            use_float=False,
            exclude_keys={'keep_as_decimal'}
        )
        
        assert result['price'] == '100.50'
        assert result['keep_as_decimal'] == Decimal('999.99')  # Excluded from conversion
        assert result['volume'] == '10'


class TestValidationUtils:
    """Test validation utilities."""

    @pytest.fixture
    def validator(self):
        """Create ValidationHelper instance."""
        return ValidationHelper()

    def test_validate_export_format(self, validator):
        """Test validate_export_format method."""
        supported_formats = ['json', 'csv', 'excel']
        
        # Test valid format
        result = validator.validate_export_format('JSON', supported_formats)
        assert result == 'json'
        
        # Test invalid format
        with pytest.raises(ValidationError):
            validator.validate_export_format('xml', supported_formats)

    def test_validate_date_range(self, validator):
        """Test validate_date_range method."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        # Test valid range - returns None if valid
        validator.validate_date_range(start, end)
        
        # Test invalid range (end before start)
        with pytest.raises(ValidationError):
            validator.validate_date_range(end, start)
        
        # Test with max_range_days
        with pytest.raises(ValidationError):
            validator.validate_date_range(start, end, max_range_days=10)

    def test_validate_numeric_range(self, validator):
        """Test validate_numeric_range method."""
        # Test valid range
        validator.validate_numeric_range(50, min_value=0, max_value=100)
        
        # Test below minimum
        with pytest.raises(ValidationError):
            validator.validate_numeric_range(5, min_value=10)
        
        # Test above maximum
        with pytest.raises(ValidationError):
            validator.validate_numeric_range(150, max_value=100)

    def test_validate_required_fields(self, validator):
        """Test validate_required_fields method."""
        data = {
            'symbol': 'BTC/USDT',
            'price': Decimal('50000'),
            'quantity': Decimal('1')
        }
        
        # Test with all required fields present
        validator.validate_required_fields(data, ['symbol', 'price'])
        
        # Test with missing required field
        with pytest.raises(ValidationError):
            validator.validate_required_fields(data, ['symbol', 'price', 'missing_field'])

    def test_validate_list_not_empty(self, validator):
        """Test validate_list_not_empty method."""
        # Test with non-empty list
        validator.validate_list_not_empty([1, 2, 3])
        
        # Test with empty list
        with pytest.raises(ValidationError):
            validator.validate_list_not_empty([])


class TestUtilsEdgeCases:
    """Test edge cases in utils modules."""

    @pytest.fixture
    def converter(self):
        """Create DataConverter instance."""
        return DataConverter()

    @pytest.fixture
    def validator(self):
        """Create ValidationHelper instance."""
        return ValidationHelper()

    def test_handle_none_values(self, converter):
        """Test handling of None values."""
        data = {'value': None}
        result = converter.prepare_for_json_export(data)
        assert isinstance(result, dict)

    def test_handle_empty_collections(self, converter):
        """Test handling of empty collections."""
        data = {'items': [], 'mapping': {}}
        result = converter.convert_decimals_for_json(data)
        assert result['items'] == []
        assert result['mapping'] == {}

    def test_handle_nested_structures(self, converter):
        """Test handling of deeply nested structures."""
        data = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': Decimal('123.456')
                    }
                }
            }
        }
        result = converter.convert_decimals_for_json(data, use_float=False)
        assert result['level1']['level2']['level3']['value'] == '123.456'

    def test_handle_mixed_types(self, converter):
        """Test handling of mixed data types."""
        data = {
            'decimal': Decimal('100.50'),
            'float': 50.25,
            'int': 100,
            'string': 'hello',
            'bool': True,
            'none': None
        }
        result = converter.convert_decimals_for_json(data)
        assert isinstance(result, dict)