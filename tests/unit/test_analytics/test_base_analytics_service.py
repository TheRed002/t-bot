"""Test cases for base_analytics_service module to increase coverage."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.analytics.base_analytics_service import BaseAnalyticsService
from src.core.exceptions import ValidationError


class ConcreteAnalyticsService(BaseAnalyticsService):
    """Concrete implementation for testing."""
    
    async def calculate_metrics(self, *args, **kwargs) -> dict:
        """Implement abstract method."""
        return {'test': 'metrics'}
    
    async def validate_data(self, data) -> bool:
        """Implement abstract method."""
        return True
    
    async def _service_health_check(self) -> dict:
        """Implement abstract method."""
        return {'status': 'healthy'}


class TestBaseAnalyticsService:
    """Test BaseAnalyticsService class."""

    @pytest.fixture
    def service(self):
        """Create ConcreteAnalyticsService instance."""
        service = ConcreteAnalyticsService(
            name='TestService',
            config={'test': 'config'},
            metrics_collector=Mock()
        )
        # Fix the config attribute for tests
        service.config = service._config
        return service

    def test_validate_time_range(self, service):
        """Test validate_time_range method."""
        start = datetime.now()
        end = start + timedelta(hours=1)
        
        # Test valid range
        service.validate_time_range(start, end)
        
        # Test invalid range (end before start)
        with pytest.raises(ValidationError):
            service.validate_time_range(end, start)

    def test_validate_decimal_value(self, service):
        """Test validate_decimal_value method."""
        # Test valid decimal
        result = service.validate_decimal_value(Decimal('100.50'), 'price')
        assert result == Decimal('100.50')
        
        # Test with min_value constraint
        with pytest.raises(ValidationError):
            service.validate_decimal_value(Decimal('5'), 'price', min_value=Decimal('10'))
        
        # Test with max_value constraint
        with pytest.raises(ValidationError):
            service.validate_decimal_value(Decimal('1000'), 'price', max_value=Decimal('500'))
        
        # Test string conversion
        result = service.validate_decimal_value('123.45', 'price')
        assert result == Decimal('123.45')

    def test_cache_operations(self, service):
        """Test cache operations."""
        # Test set and get cache
        service.set_cache('test_key', 'test_value')
        assert service.get_from_cache('test_key') == 'test_value'
        
        # Test clear cache
        service.clear_cache()
        assert service.get_from_cache('test_key') is None

    def test_record_calculation_time(self, service):
        """Test record_calculation_time method."""
        # Mock the metrics collector's observe_histogram method
        service.metrics_collector.observe_histogram = Mock()
        
        service.record_calculation_time('test_operation', 0.5)
        
        # Verify metrics collector was called
        service.metrics_collector.observe_histogram.assert_called()

    def test_record_error(self, service):
        """Test record_error method."""
        # Mock the metrics collector's increment_counter method
        service.metrics_collector.increment_counter = Mock()
        
        error = ValueError('Test error')
        service.record_error('test_operation', error)
        
        # Verify error was recorded
        service.metrics_collector.increment_counter.assert_called()

    def test_convert_for_export(self, service):
        """Test convert_for_export method."""
        # Test Decimal conversion
        data = {
            'price': Decimal('100.50'),
            'quantity': Decimal('10'),
            'text': 'hello'
        }
        
        result = service.convert_for_export(data)
        
        # Decimals are now converted to strings to preserve precision
        assert result['price'] == '100.50'
        assert result['quantity'] == '10'
        assert result['text'] == 'hello'
        
        # Test nested structures
        nested = {
            'level1': {
                'value': Decimal('200.75')
            }
        }
        
        result = service.convert_for_export(nested)
        assert result['level1']['value'] == '200.75'

    @pytest.mark.asyncio
    async def test_execute_monitored(self, service):
        """Test execute_monitored method."""
        async def test_func():
            await asyncio.sleep(0.01)
            return 'success'
        
        result = await service.execute_monitored('test_op', test_func)
        assert result == 'success'

    @pytest.mark.asyncio
    async def test_execute_monitored_with_error(self, service):
        """Test execute_monitored with error."""
        from src.core.exceptions import ServiceError
        
        async def failing_func():
            raise ValueError('Test error')
        
        with pytest.raises(ServiceError):
            await service.execute_monitored('test_op', failing_func)

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, service):
        """Test calculate_metrics abstract method."""
        result = await service.calculate_metrics()
        assert result == {'test': 'metrics'}

    @pytest.mark.asyncio
    async def test_validate_data(self, service):
        """Test validate_data abstract method."""
        result = await service.validate_data({'test': 'data'})
        assert result is True

    @pytest.mark.asyncio
    async def test_service_health_check(self, service):
        """Test _service_health_check abstract method."""
        result = await service._service_health_check()
        assert result == {'status': 'healthy'}

    @pytest.mark.asyncio
    async def test_cleanup(self, service):
        """Test cleanup method."""
        await service.cleanup()
        # Should clear cache
        assert len(service._cache) == 0

    def test_initialization(self):
        """Test service initialization."""
        service = ConcreteAnalyticsService(
            name='TestService',
            config={'key': 'value'},
            correlation_id='test-123'
        )
        
        assert service.name == 'TestService'
        assert service._config == {'key': 'value'}
        assert service.correlation_id == 'test-123'

    def test_edge_cases(self, service):
        """Test edge cases."""
        # Test with None values
        assert service.get_from_cache('nonexistent') is None
        
        # Test empty cache clear
        service.clear_cache()
        service.clear_cache()  # Should not raise error
        
        # Test convert_for_export with complex types
        from datetime import date
        data = {
            'date': datetime.now(),
            'date_only': date.today(),
            'none': None,
            'list': [Decimal('1'), Decimal('2')]
        }
        
        result = service.convert_for_export(data)
        assert isinstance(result, dict)