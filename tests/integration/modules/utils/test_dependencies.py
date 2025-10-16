"""
Integration tests for utils module dependency injection patterns.

These tests validate that:
1. Utils services are properly registered with the DI container
2. Service dependencies are correctly resolved
3. Circular dependencies are avoided
4. Service lifetimes are managed correctly
5. Factory patterns work as expected
"""

from unittest.mock import Mock

import pytest

from src.core.dependency_injection import injector
from src.utils.calculations.financial import FinancialCalculator
from src.utils.data_flow_integrity import (
    DataFlowValidator,
    IntegrityPreservingConverter,
    PrecisionTracker,
)
from src.utils.gpu_utils import GPUManager
from src.utils.interfaces import (
    CalculatorInterface,
    DataFlowInterface,
    GPUInterface,
    PrecisionInterface,
    ValidationServiceInterface,
)
from src.utils.service_registry import register_util_services
from src.utils.validation.core import ValidationFramework
from src.utils.validation.service import ValidationService


class TestUtilsDependencyInjection:
    """Test dependency injection patterns for utils module."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """
        Setup clean DI container for each test with proper cleanup.

        This fixture ensures:
        1. Clean DI container before each test
        2. Proper state reset after each test
        3. No resource leaks between tests
        """
        # Setup: Clear and reset before test
        injector.get_container().clear()

        # Reset the registration flag to allow re-registration
        import src.utils.service_registry as registry_module

        registry_module._services_registered = False

        yield

        # Teardown: Clean up after test
        try:
            injector.get_container().clear()
        except Exception:
            pass  # Ignore cleanup errors to prevent cascade failures

        # Reset the registration flag
        try:
            registry_module._services_registered = False
        except Exception:
            pass

    def test_service_registry_registration(self):
        """Test that register_util_services properly registers all services."""
        # Register services
        register_util_services()

        # Verify all services are registered
        expected_registrations = [
            # ValidationService registrations
            ("ValidationService", ValidationService),
            ("ValidationServiceInterface", ValidationServiceInterface),
            ("ValidationFramework", ValidationFramework),
            # GPU utilities
            ("GPUManager", GPUManager),
            ("GPUInterface", GPUInterface),
            # Data flow utilities
            ("PrecisionTracker", PrecisionTracker),
            ("PrecisionInterface", PrecisionInterface),
            ("DataFlowValidator", DataFlowValidator),
            ("DataFlowInterface", DataFlowInterface),
            ("IntegrityPreservingConverter", IntegrityPreservingConverter),
            # Financial calculations
            ("FinancialCalculator", FinancialCalculator),
            ("CalculatorInterface", CalculatorInterface),
        ]

        for service_name, expected_type in expected_registrations:
            try:
                service = injector.resolve(service_name)
                # For interface registrations, check that the resolved service
                # implements the interface
                if service_name.endswith("Interface"):
                    assert isinstance(service, expected_type), (
                        f"{service_name} does not implement {expected_type}"
                    )
                else:
                    assert isinstance(service, expected_type), (
                        f"{service_name} is not instance of {expected_type}"
                    )
            except Exception as e:
                pytest.fail(f"Failed to resolve {service_name}: {e}")

    def test_validation_service_dependency_injection(self):
        """Test ValidationService dependency injection pattern."""
        register_util_services()

        # Resolve ValidationService
        validation_service = injector.resolve("ValidationService")

        # Verify it has required dependencies injected
        assert hasattr(validation_service, "framework")
        assert validation_service.framework is not None
        assert isinstance(validation_service.framework, ValidationFramework)

        # Verify it implements the interface
        assert isinstance(validation_service, ValidationServiceInterface)

    def test_singleton_pattern_enforcement(self):
        """Test that singleton services return the same instance."""
        register_util_services()

        # Services that should be singletons
        singleton_services = [
            "ValidationService",
            "ValidationFramework",
            "GPUManager",
            "PrecisionTracker",
            "DataFlowValidator",
            "FinancialCalculator",
        ]

        for service_name in singleton_services:
            instance1 = injector.resolve(service_name)
            instance2 = injector.resolve(service_name)

            assert instance1 is instance2, (
                f"{service_name} should be singleton but returned different instances"
            )

    def test_interface_to_implementation_mapping(self):
        """Test that interfaces resolve to correct implementations."""
        register_util_services()

        interface_mappings = [
            ("ValidationServiceInterface", ValidationService),
            ("GPUInterface", GPUManager),
            ("PrecisionInterface", PrecisionTracker),
            ("DataFlowInterface", DataFlowValidator),
            ("CalculatorInterface", FinancialCalculator),
        ]

        for interface_name, expected_impl in interface_mappings:
            resolved_service = injector.resolve(interface_name)
            assert isinstance(resolved_service, expected_impl), (
                f"{interface_name} should resolve to {expected_impl}"
            )

    def test_factory_pattern_execution(self):
        """Test that factory functions execute correctly."""
        register_util_services()

        # Test factory-created services can be resolved
        factory_services = [
            "ValidationService",
            "GPUManager",
            "PrecisionTracker",
            "DataFlowValidator",
            "FinancialCalculator",
        ]

        for service_name in factory_services:
            try:
                service = injector.resolve(service_name)
                assert service is not None
                # Verify the service is properly initialized
                assert hasattr(service, "__class__")
            except Exception as e:
                pytest.fail(f"Factory for {service_name} failed: {e}")

    def test_complex_dependency_chain(self):
        """Test complex dependency chain resolution."""
        register_util_services()

        # IntegrityPreservingConverter depends on PrecisionTracker
        converter = injector.resolve("IntegrityPreservingConverter")

        # Verify dependency was properly injected (attribute is named 'tracker')
        assert hasattr(converter, "tracker")
        assert converter.tracker is not None

        # Verify it's the same instance as the singleton
        precision_tracker = injector.resolve("PrecisionTracker")
        assert converter.tracker is precision_tracker

    def test_no_circular_dependencies(self):
        """Test that no circular dependencies exist."""
        register_util_services()

        # Try to resolve all services - this will fail if circular deps exist
        all_services = [
            "ValidationService",
            "ValidationServiceInterface",
            "ValidationFramework",
            "GPUManager",
            "GPUInterface",
            "PrecisionTracker",
            "PrecisionInterface",
            "DataFlowValidator",
            "DataFlowInterface",
            "IntegrityPreservingConverter",
            "FinancialCalculator",
            "CalculatorInterface",
        ]

        resolved_services = {}
        for service_name in all_services:
            try:
                service = injector.resolve(service_name)
                resolved_services[service_name] = service
            except Exception as e:
                pytest.fail(f"Circular dependency detected for {service_name}: {e}")

        # Verify all services were resolved
        assert len(resolved_services) == len(all_services)

    def test_idempotent_registration(self):
        """Test that multiple registrations are idempotent."""
        # Register services multiple times
        register_util_services()
        register_util_services()
        register_util_services()

        # Should not cause issues
        validation_service1 = injector.resolve("ValidationService")
        validation_service2 = injector.resolve("ValidationService")

        # Should still be singleton
        assert validation_service1 is validation_service2

    def test_service_initialization_order(self):
        """Test that services can be initialized in any order."""
        register_util_services()

        # Try resolving services in different orders
        orders = [
            ["ValidationService", "GPUManager", "PrecisionTracker"],
            ["PrecisionTracker", "ValidationService", "GPUManager"],
            ["GPUManager", "PrecisionTracker", "ValidationService"],
        ]

        for order in orders:
            # Clear and re-register for each test
            injector.get_container().clear()

            # Reset registration flag before re-registering
            import src.utils.service_registry as registry_module

            registry_module._services_registered = False
            register_util_services()

            services = []
            for service_name in order:
                service = injector.resolve(service_name)
                services.append(service)

            # Verify all services are valid instances
            for service in services:
                assert service is not None

    def test_dependency_error_handling(self):
        """Test proper error handling for missing dependencies."""
        # Don't register services

        with pytest.raises(Exception):  # Should raise DependencyError or similar
            injector.resolve("ValidationService")

    def test_service_interface_contracts(self):
        """Test that resolved services satisfy their interface contracts."""
        register_util_services()

        # Test ValidationServiceInterface contract
        validation_service = injector.resolve("ValidationServiceInterface")
        required_methods = [
            "validate_order",
            "validate_risk_parameters",
            "validate_strategy_config",
            "validate_market_data",
            "validate_batch",
        ]

        for method_name in required_methods:
            assert hasattr(validation_service, method_name)
            assert callable(getattr(validation_service, method_name))

    def test_service_configuration_injection(self):
        """Test that services can be configured through DI."""
        register_util_services()

        validation_service = injector.resolve("ValidationService")

        # Verify service has proper configuration capabilities
        assert hasattr(validation_service, "name")
        assert hasattr(validation_service, "_config")  # BaseComponent stores config as private

        # Test that the service was properly initialized with dependencies
        assert validation_service.framework is not None

    def test_factory_function_isolation(self):
        """Test that factory functions are properly isolated."""
        register_util_services()

        # Create multiple instances through factory (if not singleton)
        # This tests that factory state doesn't leak between calls

        # ValidationFramework is singleton, so test with a non-singleton if available
        # For now, test that factory functions don't interfere with each other

        framework1 = injector.resolve("ValidationFramework")
        service1 = injector.resolve("ValidationService")

        # Both should be valid and independent
        assert framework1 is not None
        assert service1 is not None
        assert service1.framework is framework1  # Proper dependency injection

    def test_service_cleanup_and_reregistration(self):
        """Test service cleanup and reregistration scenarios."""
        # Initial registration
        register_util_services()
        service1 = injector.resolve("ValidationService")

        # Clear container
        injector.get_container().clear()

        # Reset registration flag before re-registering
        import src.utils.service_registry as registry_module

        registry_module._services_registered = False

        # Re-register
        register_util_services()
        service2 = injector.resolve("ValidationService")

        # Should be different instances after cleanup
        assert service1 is not service2

        # But both should be valid
        assert service1 is not None
        assert service2 is not None


class TestUtilsDependencyInjectionEdgeCases:
    """Test edge cases for utils dependency injection."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """
        Setup clean DI container for each test with proper cleanup.

        This fixture ensures:
        1. Clean DI container before each test
        2. Proper state reset after each test
        3. No resource leaks between tests
        """
        # Setup: Clear and reset before test
        injector.get_container().clear()

        # Reset the registration flag to allow re-registration
        import src.utils.service_registry as registry_module

        registry_module._services_registered = False

        yield

        # Teardown: Clean up after test
        try:
            injector.get_container().clear()
        except Exception:
            pass  # Ignore cleanup errors to prevent cascade failures

        # Reset the registration flag
        try:
            registry_module._services_registered = False
        except Exception:
            pass

    def test_partial_registration_handling(self):
        """Test handling of partial service registration."""
        # Manually register only some services to test error handling
        from src.utils.validation.core import ValidationFramework

        # Use register_factory instead of register_class (which doesn't exist)
        injector.register_factory(
            "ValidationFramework", lambda: ValidationFramework(), singleton=True
        )

        # Try to resolve ValidationService without full registration
        with pytest.raises(Exception):
            injector.resolve("ValidationService")

    def test_service_replacement_scenarios(self):
        """Test service replacement scenarios."""
        register_util_services()

        original_service = injector.resolve("ValidationFramework")

        # Clear container and re-register to test replacement
        injector.get_container().clear()

        # Re-register with mock for testing (use lambda to avoid mock being called as factory)
        mock_framework = Mock(spec=ValidationFramework)
        injector.register_factory("ValidationFramework", lambda: mock_framework, singleton=True)

        replaced_service = injector.resolve("ValidationFramework")
        assert replaced_service is mock_framework
        assert replaced_service is not original_service

    def test_concurrent_resolution(self):
        """Test concurrent service resolution for thread safety."""
        import threading

        register_util_services()

        resolved_services = []
        errors = []

        def resolve_service():
            try:
                service = injector.resolve("ValidationService")
                resolved_services.append(service)
            except Exception as e:
                errors.append(e)

        # Create multiple threads resolving services concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=resolve_service)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should not have errors
        assert len(errors) == 0, f"Concurrent resolution errors: {errors}"

        # All should resolve to same singleton instance
        assert len(resolved_services) == 10
        first_service = resolved_services[0]
        for service in resolved_services[1:]:
            assert service is first_service
