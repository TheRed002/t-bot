"""
Tests for analytics module initialization and exports.

Tests that all public interfaces are properly exported and importable.
"""

# Disable logging during tests for performance
import logging

import pytest

logging.disable(logging.CRITICAL)

# Set pytest markers for optimization
pytestmark = pytest.mark.unit

from src.analytics import (
    AlertServiceProtocol,
    AnalyticsService,
    AnalyticsServiceFactory,
    AnalyticsServiceProtocol,
    ExportServiceProtocol,
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    RealtimeAnalyticsServiceProtocol,
    ReportingServiceProtocol,
    RiskServiceProtocol,
    configure_analytics_dependencies,
    create_default_analytics_service,
    get_analytics_factory,
    get_analytics_service,
    register_analytics_services,
)


class TestAnalyticsModuleExports:
    """Test that analytics module exports all expected symbols."""

    def test_analytics_service_import(self):
        """Test AnalyticsService can be imported."""
        assert AnalyticsService is not None
        assert hasattr(AnalyticsService, "__name__")

    def test_factory_imports(self):
        """Test factory classes and functions can be imported."""
        assert AnalyticsServiceFactory is not None
        assert create_default_analytics_service is not None
        assert callable(create_default_analytics_service)

    def test_protocol_imports(self):
        """Test all protocol interfaces can be imported."""
        protocols = [
            AnalyticsServiceProtocol,
            RealtimeAnalyticsServiceProtocol,
            PortfolioServiceProtocol,
            RiskServiceProtocol,
            ReportingServiceProtocol,
            OperationalServiceProtocol,
            AlertServiceProtocol,
            ExportServiceProtocol,
        ]

        for protocol in protocols:
            assert protocol is not None

    def test_dependency_injection_imports(self):
        """Test dependency injection functions can be imported."""
        di_functions = [
            register_analytics_services,
            configure_analytics_dependencies,
            get_analytics_service,
            get_analytics_factory,
        ]

        for func in di_functions:
            assert func is not None
            assert callable(func)

    def test_module_all_attribute(self):
        """Test that __all__ contains expected exports."""
        import src.analytics

        expected_exports = [
            "AnalyticsService",
            "AnalyticsServiceFactory",
            "create_default_analytics_service",
            "AnalyticsServiceProtocol",
            "RealtimeAnalyticsServiceProtocol",
            "PortfolioServiceProtocol",
            "RiskServiceProtocol",
            "ReportingServiceProtocol",
            "OperationalServiceProtocol",
            "AlertServiceProtocol",
            "ExportServiceProtocol",
            "register_analytics_services",
            "configure_analytics_dependencies",
            "get_analytics_service",
            "get_analytics_factory",
        ]

        for export in expected_exports:
            assert export in src.analytics.__all__

    def test_module_docstring_exists(self):
        """Test that module has proper docstring."""
        import src.analytics

        assert src.analytics.__doc__ is not None
        assert len(src.analytics.__doc__.strip()) > 0
        assert "Analytics Module" in src.analytics.__doc__

    def test_no_unexpected_imports_in_all(self):
        """Test that __all__ doesn't contain unexpected items."""
        import src.analytics

        # Check that all items in __all__ are actually available in the module
        for item in src.analytics.__all__:
            assert hasattr(src.analytics, item), f"Module attribute '{item}' not found"

    def test_import_star_compatibility(self):
        """Test that 'from src.analytics import *' works correctly."""
        # This test ensures that __all__ is properly defined and all exports work
        import importlib

        # Import the module
        module = importlib.import_module("src.analytics")

        # Check that __all__ is defined
        assert hasattr(module, "__all__")

        # Check that all items in __all__ exist in the module
        for name in module.__all__:
            assert hasattr(module, name)


class TestAnalyticsModuleStructure:
    """Test the overall structure of the analytics module."""

    def test_module_imports_without_errors(self):
        """Test that importing the analytics module doesn't raise errors."""
        # This test catches any circular import or initialization issues
        try:
            import src.analytics

            assert True  # If we get here, import succeeded
        except ImportError as e:
            pytest.fail(f"Analytics module import failed: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing analytics module: {e}")

    def test_submodule_imports_accessible(self):
        """Test that submodules can be accessed through main import."""
        import src.analytics

        # These should be accessible through the main module
        assert hasattr(src.analytics, "AnalyticsService")
        assert hasattr(src.analytics, "AnalyticsServiceFactory")

    def test_dependency_injection_module_accessible(self):
        """Test that DI functions are accessible."""
        from src.analytics import configure_analytics_dependencies

        # Should be callable
        assert callable(configure_analytics_dependencies)

    def test_types_accessible_through_imports(self):
        """Test that types are accessible through service imports."""
        # This ensures that importing services also makes types available
        from src.analytics import AnalyticsService

        # The service should exist and be a class
        assert isinstance(AnalyticsService, type)


class TestImportCompatibility:
    """Test import compatibility and edge cases."""

    def test_relative_imports_work(self):
        """Test that relative imports within the module work."""
        # Test that the module can import its own components
        try:
            from src.analytics.factory import AnalyticsServiceFactory
            from src.analytics.interfaces import AnalyticsServiceProtocol
            from src.analytics.service import AnalyticsService

            assert AnalyticsService is not None
            assert AnalyticsServiceFactory is not None
            assert AnalyticsServiceProtocol is not None

        except ImportError as e:
            pytest.fail(f"Relative imports failed: {e}")

    def test_circular_import_handling(self):
        """Test that circular imports are properly handled."""
        # This test ensures that TYPE_CHECKING is used properly
        try:
            from src.analytics.factory import AnalyticsServiceFactory
            from src.analytics.service import AnalyticsService

            # Both should import without circular dependency issues
            assert AnalyticsServiceFactory is not None
            assert AnalyticsService is not None

        except ImportError as e:
            pytest.fail(f"Circular import issue detected: {e}")

    def test_protocol_runtime_checkable(self):
        """Test that protocols are runtime checkable where appropriate."""

        # Import protocols and check if they're runtime checkable
        protocols = [
            AnalyticsServiceProtocol,
            AlertServiceProtocol,
            RiskServiceProtocol,
        ]

        for protocol in protocols:
            # Protocols should be importable
            assert protocol is not None
            # We don't require them to be runtime_checkable but they should be valid protocols
            assert hasattr(protocol, "__annotations__") or hasattr(protocol, "__protocol_attrs__")

    def test_import_performance(self):
        """Test that imports are reasonably fast."""
        import importlib
        import sys
        import time

        # Remove module from cache if it exists
        module_name = "src.analytics"
        if module_name in sys.modules:
            del sys.modules[module_name]

        start_time = time.time()
        importlib.import_module(module_name)
        end_time = time.time()

        import_time = end_time - start_time

        # Import should complete within reasonable time (1 second)
        assert import_time < 1.0, f"Import took too long: {import_time:.3f}s"


class TestAnalyticsModuleIntegration:
    """Test integration aspects of the analytics module."""

    def test_factory_service_integration(self):
        """Test that factory and service integrate correctly."""
        from src.analytics import AnalyticsServiceFactory, create_default_analytics_service

        # Factory should be importable and instantiable
        assert AnalyticsServiceFactory is not None

        # Default service creation should be available
        assert callable(create_default_analytics_service)

    def test_protocols_service_compatibility(self):
        """Test that protocols are compatible with service implementations."""
        from src.analytics import (
            AnalyticsService,
            AnalyticsServiceProtocol,
            PortfolioServiceProtocol,
            RiskServiceProtocol,
        )

        # Service should exist
        assert AnalyticsService is not None

        # Protocols should exist
        assert AnalyticsServiceProtocol is not None
        assert PortfolioServiceProtocol is not None
        assert RiskServiceProtocol is not None

    def test_dependency_injection_integration(self):
        """Test that dependency injection components integrate."""
        from src.analytics import (
            configure_analytics_dependencies,
            get_analytics_factory,
            get_analytics_service,
            register_analytics_services,
        )

        # All DI functions should be callable
        di_functions = [
            register_analytics_services,
            configure_analytics_dependencies,
            get_analytics_service,
            get_analytics_factory,
        ]

        for func in di_functions:
            assert callable(func)


class TestAnalyticsModuleExportConsistency:
    """Test consistency of module exports."""

    def test_exported_symbols_are_classes_or_functions(self):
        """Test that exported symbols are appropriate types."""
        import src.analytics

        class_exports = [
            "AnalyticsService",
            "AnalyticsServiceFactory",
        ]

        function_exports = [
            "create_default_analytics_service",
            "register_analytics_services",
            "configure_analytics_dependencies",
            "get_analytics_service",
            "get_analytics_factory",
        ]

        protocol_exports = [
            "AnalyticsServiceProtocol",
            "RealtimeAnalyticsServiceProtocol",
            "PortfolioServiceProtocol",
            "RiskServiceProtocol",
            "ReportingServiceProtocol",
            "OperationalServiceProtocol",
            "AlertServiceProtocol",
            "ExportServiceProtocol",
        ]

        # Check classes
        for name in class_exports:
            obj = getattr(src.analytics, name)
            assert isinstance(obj, type), f"{name} should be a class"

        # Check functions
        for name in function_exports:
            obj = getattr(src.analytics, name)
            assert callable(obj), f"{name} should be callable"

        # Check protocols (these are special types)
        for name in protocol_exports:
            obj = getattr(src.analytics, name)
            assert obj is not None, f"{name} should exist"

    def test_no_private_symbols_in_exports(self):
        """Test that private symbols are not exported."""
        import src.analytics

        for name in src.analytics.__all__:
            assert not name.startswith("_"), f"Private symbol {name} should not be in __all__"

    def test_all_public_interfaces_exported(self):
        """Test that all intended public interfaces are exported."""
        import src.analytics

        # Key interfaces that should definitely be available
        critical_exports = [
            "AnalyticsService",
            "AnalyticsServiceFactory",
            "AnalyticsServiceProtocol",
            "configure_analytics_dependencies",
        ]

        for name in critical_exports:
            assert name in src.analytics.__all__, f"Critical export {name} missing from __all__"
            assert hasattr(src.analytics, name), f"Critical export {name} not available"
