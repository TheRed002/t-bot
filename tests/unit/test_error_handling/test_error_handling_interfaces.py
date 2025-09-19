"""
Tests for error handling interfaces.

Testing interface definitions and protocols.
"""

import pytest
from typing import runtime_checkable, Protocol

from src.core.base.interfaces import HealthCheckResult
from src.error_handling.interfaces import (
    ErrorHandlerInterface,
    ErrorHandlingRepositoryPort,
    ErrorHandlingServiceInterface,
    ErrorHandlingServicePort,
    ErrorPatternAnalyticsInterface,
    GlobalErrorHandlerInterface,
)


class TestErrorHandlingServiceInterface:
    """Test error handling service interface."""

    def test_interface_is_protocol(self):
        """Test that ErrorHandlingServiceInterface is a protocol."""
        # Check if it's marked as runtime_checkable
        assert hasattr(ErrorHandlingServiceInterface, '__instancecheck__')

    def test_interface_methods_defined(self):
        """Test that all expected methods are defined."""
        # These are abstract methods in the protocol
        expected_methods = [
            'handle_error',
            'handle_global_error', 
            'validate_state_consistency',
            'get_error_patterns',
            'health_check'
        ]
        
        for method_name in expected_methods:
            assert hasattr(ErrorHandlingServiceInterface, method_name)

    def test_runtime_checkable(self):
        """Test that interface is runtime checkable."""
        assert hasattr(ErrorHandlingServiceInterface, '__instancecheck__')


class TestErrorPatternAnalyticsInterface:
    """Test error pattern analytics interface."""

    def test_interface_is_protocol(self):
        """Test that ErrorPatternAnalyticsInterface is a protocol."""
        # Check if it's marked as runtime_checkable
        assert hasattr(ErrorPatternAnalyticsInterface, '__instancecheck__')

    def test_interface_methods_defined(self):
        """Test that all expected methods are defined."""
        expected_methods = [
            'add_error_event',
            'add_batch_error_events',
            'get_pattern_summary',
            'get_correlation_summary',
            'get_trend_summary',
            'cleanup'
        ]
        
        for method_name in expected_methods:
            assert hasattr(ErrorPatternAnalyticsInterface, method_name)


class TestErrorHandlerInterface:
    """Test error handler interface."""

    def test_interface_is_protocol(self):
        """Test that ErrorHandlerInterface is a protocol."""
        # Check if it's marked as runtime_checkable
        assert hasattr(ErrorHandlerInterface, '__instancecheck__')

    def test_interface_methods_defined(self):
        """Test that all expected methods are defined."""
        expected_methods = [
            'handle_error',
            'classify_error',
            'create_error_context',
            'cleanup_resources',
            'shutdown'
        ]
        
        for method_name in expected_methods:
            assert hasattr(ErrorHandlerInterface, method_name)


class TestGlobalErrorHandlerInterface:
    """Test global error handler interface."""

    def test_interface_is_protocol(self):
        """Test that GlobalErrorHandlerInterface is a protocol."""
        # Check if it's marked as runtime_checkable
        assert hasattr(GlobalErrorHandlerInterface, '__instancecheck__')

    def test_interface_methods_defined(self):
        """Test that all expected methods are defined."""
        expected_methods = [
            'handle_error',
            'get_statistics'
        ]
        
        for method_name in expected_methods:
            assert hasattr(GlobalErrorHandlerInterface, method_name)


class TestErrorHandlingServicePort:
    """Test error handling service port (hexagonal architecture)."""

    def test_is_abstract_base_class(self):
        """Test that ErrorHandlingServicePort is an ABC."""
        from abc import ABC
        assert issubclass(ErrorHandlingServicePort, ABC)

    def test_abstract_methods_defined(self):
        """Test that all abstract methods are defined."""
        abstract_methods = ErrorHandlingServicePort.__abstractmethods__
        
        expected_methods = {
            'process_error',
            'analyze_error_patterns',
            'validate_system_state'
        }
        
        assert abstract_methods == expected_methods

    def test_cannot_instantiate_directly(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ErrorHandlingServicePort()


class TestErrorHandlingRepositoryPort:
    """Test error handling repository port."""

    def test_is_abstract_base_class(self):
        """Test that ErrorHandlingRepositoryPort is an ABC."""
        from abc import ABC
        assert issubclass(ErrorHandlingRepositoryPort, ABC)

    def test_abstract_methods_defined(self):
        """Test that all abstract methods are defined."""
        abstract_methods = ErrorHandlingRepositoryPort.__abstractmethods__
        
        expected_methods = {
            'store_error_event',
            'retrieve_error_patterns', 
            'update_error_statistics'
        }
        
        assert abstract_methods == expected_methods

    def test_cannot_instantiate_directly(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ErrorHandlingRepositoryPort()


class TestInterfaceExports:
    """Test interface module exports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.error_handling.interfaces import __all__
        
        expected_exports = [
            "ErrorHandlerInterface",
            "ErrorHandlingRepositoryPort",
            "ErrorHandlingServiceInterface", 
            "ErrorHandlingServicePort",
            "ErrorPatternAnalyticsInterface",
            "GlobalErrorHandlerInterface",
        ]
        
        for export in expected_exports:
            assert export in __all__


# Mock implementations for testing interface compliance
class MockErrorHandlingService:
    """Mock implementation of ErrorHandlingServiceInterface."""
    
    async def handle_error(self, error, component, operation, context=None, recovery_strategy=None):
        return {"status": "handled"}
    
    async def handle_global_error(self, error, context=None, severity="error"):
        return {"status": "handled_globally"}
    
    async def validate_state_consistency(self, component="all"):
        return {"status": "valid"}
    
    async def get_error_patterns(self):
        return {"patterns": []}
    
    async def health_check(self):
        return HealthCheckResult(healthy=True, message="OK")


class MockErrorPatternAnalytics:
    """Mock implementation of ErrorPatternAnalyticsInterface."""
    
    def add_error_event(self, error_context):
        pass
    
    async def add_batch_error_events(self, error_contexts):
        pass
    
    def get_pattern_summary(self):
        return {"patterns": []}
    
    def get_correlation_summary(self):
        return {"correlations": []}
    
    def get_trend_summary(self):
        return {"trends": []}
    
    async def cleanup(self):
        pass


class MockErrorHandler:
    """Mock implementation of ErrorHandlerInterface."""
    
    async def handle_error(self, error, context, recovery_strategy=None):
        return True
    
    def classify_error(self, error):
        return "medium"
    
    def create_error_context(self, error, component, operation, **kwargs):
        return {"error": str(error)}
    
    async def cleanup_resources(self):
        pass
    
    async def shutdown(self):
        pass


class MockGlobalErrorHandler:
    """Mock implementation of GlobalErrorHandlerInterface."""
    
    async def handle_error(self, error, context=None, severity="error"):
        return {"handled": True}
    
    def get_statistics(self):
        return {"total_errors": 0}


class TestInterfaceCompliance:
    """Test interface compliance with mock implementations."""

    def test_error_handling_service_interface_compliance(self):
        """Test that mock service implements the interface correctly."""
        mock_service = MockErrorHandlingService()
        
        assert isinstance(mock_service, ErrorHandlingServiceInterface)

    def test_error_pattern_analytics_interface_compliance(self):
        """Test that mock analytics implements the interface correctly."""
        mock_analytics = MockErrorPatternAnalytics()
        
        assert isinstance(mock_analytics, ErrorPatternAnalyticsInterface)

    def test_error_handler_interface_compliance(self):
        """Test that mock handler implements the interface correctly."""
        mock_handler = MockErrorHandler()
        
        assert isinstance(mock_handler, ErrorHandlerInterface)

    def test_global_error_handler_interface_compliance(self):
        """Test that mock global handler implements the interface correctly."""
        mock_global_handler = MockGlobalErrorHandler()
        
        assert isinstance(mock_global_handler, GlobalErrorHandlerInterface)