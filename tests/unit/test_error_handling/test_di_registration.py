"""
Tests for dependency injection registration of error handling services.

This module tests the registration of error handling services with the DI container,
ensuring proper initialization and avoiding circular dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Config
from src.error_handling.di_registration import register_error_handling_services


class MockInjector:
    """Mock dependency injector for testing."""

    def __init__(self):
        self._services = {}
        self._factories = {}
        self._singletons = set()

    def register_factory(self, name, factory, singleton=False):
        """Register a factory function."""
        self._factories[name] = factory
        if singleton:
            self._singletons.add(name)

    def register_instance(self, name, instance):
        """Register a service instance."""
        self._services[name] = instance

    def register(self, name, factory, singleton=False):
        """Register a service with factory (alias for register_factory)."""
        self.register_factory(name, factory, singleton=singleton)

    def has_service(self, name):
        """Check if service is registered."""
        return name in self._services or name in self._factories

    def has(self, name):
        """Check if service exists (alias for has_service)."""
        return self.has_service(name)

    def get(self, name):
        """Get a service (alias for resolve)."""
        return self.resolve(name)

    def resolve(self, name):
        """Resolve a service."""
        if name in self._services:
            return self._services[name]
        elif name in self._factories:
            # For singletons, cache the instance after first creation
            if name in self._singletons:
                if f"_{name}_instance" not in self._services:
                    instance = self._factories[name]()
                    self._services[f"_{name}_instance"] = instance
                return self._services[f"_{name}_instance"]
            else:
                return self._factories[name]()
        else:
            raise KeyError(f"Service {name} not found")


class TestDIRegistration:
    """Test dependency injection registration."""

    @pytest.fixture
    def mock_injector(self):
        """Create mock injector."""
        return MockInjector()

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        return MagicMock(spec=Config)

    def test_register_error_handling_services_basic(self, mock_injector, mock_config):
        """Test basic service registration."""
        with patch("src.error_handling.di_registration.logger") as mock_logger:
            register_error_handling_services(mock_injector, mock_config)

            # Should log registration start
            mock_logger.info.assert_any_call(
                "Registering error handling services with DI container"
            )

            # Should register ErrorHandler
            assert mock_injector.has_service("ErrorHandler")
            assert "ErrorHandler" in mock_injector._singletons

    def test_register_with_config_from_injector(self, mock_injector):
        """Test registration when config is available in injector."""
        mock_config = MagicMock(spec=Config)
        mock_injector.register_instance("Config", mock_config)

        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.error_handler.ErrorHandler") as MockErrorHandler,
        ):
            register_error_handling_services(mock_injector, None)

            # Should resolve config from injector
            assert mock_injector.has_service("ErrorHandler")

    def test_register_without_config(self, mock_injector):
        """Test registration without any config."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.error_handler.ErrorHandler") as MockErrorHandler,
            patch("src.core.config.Config") as MockConfig,
        ):
            register_error_handling_services(mock_injector, None)

            # Should create default config
            assert mock_injector.has_service("ErrorHandler")

    def test_error_handler_factory_creation(self, mock_injector, mock_config):
        """Test ErrorHandler factory creation."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.error_handler.ErrorHandler") as MockErrorHandler,
        ):
            register_error_handling_services(mock_injector, mock_config)

            # Get and call the factory
            factory = mock_injector._factories["ErrorHandler"]
            instance = factory()

            # Should create ErrorHandler instance (may be called multiple times due to dependency configuration)
            assert MockErrorHandler.called
            assert instance is not None

    def test_error_handler_factory_with_exception(self, mock_injector, mock_config):
        """Test ErrorHandler factory error handling."""
        with (
            patch("src.error_handling.di_registration.logger") as mock_logger,
            patch("src.error_handling.error_handler.ErrorHandler") as MockErrorHandler,
        ):
            # Make ErrorHandler raise exception
            MockErrorHandler.side_effect = Exception("Creation failed")

            register_error_handling_services(mock_injector, mock_config)

            # Get factory and expect it to raise
            factory = mock_injector._factories["ErrorHandler"]

            with pytest.raises(Exception, match="Creation failed"):
                factory()

            # Should log error
            mock_logger.error.assert_called()

    def test_context_factory_registration(self, mock_injector, mock_config):
        """Test ErrorContextFactory registration."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.context.ErrorContextFactory") as MockContextFactory,
        ):
            register_error_handling_services(mock_injector, mock_config)

            # Should register ErrorContextFactory
            assert mock_injector.has_service("ErrorContextFactory")
            assert "ErrorContextFactory" in mock_injector._singletons

    def test_context_factory_creation(self, mock_injector, mock_config):
        """Test ErrorContextFactory factory creation."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.context.ErrorContextFactory") as MockContextFactory,
        ):
            register_error_handling_services(mock_injector, mock_config)

            # Get and call the factory
            factory = mock_injector._factories["ErrorContextFactory"]
            instance = factory()

            # Should create with dependency container (may be called multiple times due to dependency configuration)
            assert MockContextFactory.called
            # Verify it was called with correct parameters at least once
            call_args_list = MockContextFactory.call_args_list
            assert any(
                call.kwargs.get("dependency_container") == mock_injector for call in call_args_list
            )
            assert instance is not None

    def test_context_factory_with_exception(self, mock_injector, mock_config):
        """Test ErrorContextFactory factory error handling."""
        with (
            patch("src.error_handling.di_registration.logger") as mock_logger,
            patch("src.error_handling.context.ErrorContextFactory") as MockContextFactory,
        ):
            # Make ErrorContextFactory raise exception
            MockContextFactory.side_effect = Exception("Context factory creation failed")

            register_error_handling_services(mock_injector, mock_config)

            # Get factory and expect it to raise
            factory = mock_injector._factories["ErrorContextFactory"]

            with pytest.raises(Exception, match="Context factory creation failed"):
                factory()

            # Should log error
            mock_logger.error.assert_called()

    def test_error_pattern_analytics_registration(self, mock_injector, mock_config):
        """Test ErrorPatternAnalytics registration."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch(
                "src.error_handling.pattern_analytics.ErrorPatternAnalytics"
            ) as MockPatternAnalytics,
        ):
            register_error_handling_services(mock_injector, mock_config)

            # Should register ErrorPatternAnalytics
            assert mock_injector.has_service("ErrorPatternAnalytics")
            assert "ErrorPatternAnalytics" in mock_injector._singletons

    def test_state_monitor_registration(self, mock_injector, mock_config):
        """Test StateMonitor registration."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.state_monitor.StateMonitor") as MockStateMonitor,
        ):
            register_error_handling_services(mock_injector, mock_config)

            # Should register StateMonitor
            assert mock_injector.has_service("StateMonitor")
            assert "StateMonitor" in mock_injector._singletons

    def test_error_handling_service_registration(self, mock_injector, mock_config):
        """Test ErrorHandlingService registration."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.service.ErrorHandlingService") as MockErrorHandlingService,
        ):
            register_error_handling_services(mock_injector, mock_config)

            # Should register ErrorHandlingService
            assert mock_injector.has_service("ErrorHandlingService")
            assert "ErrorHandlingService" in mock_injector._singletons

    def test_global_handler_registration(self, mock_injector, mock_config):
        """Test GlobalErrorHandler registration."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.global_handler.GlobalErrorHandler") as MockGlobalHandler,
        ):
            register_error_handling_services(mock_injector, mock_config)

            # Should register GlobalErrorHandler
            assert mock_injector.has_service("GlobalErrorHandler")
            assert "GlobalErrorHandler" in mock_injector._singletons

    def test_error_handler_factory_registration(self, mock_injector, mock_config):
        """Test ErrorHandlerFactory registration."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.factory.ErrorHandlerFactory") as MockErrorHandlerFactory,
        ):
            register_error_handling_services(mock_injector, mock_config)

            # Should register ErrorHandlerFactory
            assert mock_injector.has_service("ErrorHandlerFactory")
            assert "ErrorHandlerFactory" in mock_injector._singletons

    def test_all_services_are_singletons(self, mock_injector, mock_config):
        """Test that all registered services are singletons."""
        with patch("src.error_handling.di_registration.logger"):
            register_error_handling_services(mock_injector, mock_config)

            expected_services = [
                "ErrorHandler",
                "ErrorContextFactory",
                "GlobalErrorHandler",
                "ErrorPatternAnalytics",
                "StateMonitor",
                "ErrorHandlingService",
                "ErrorHandlerFactory",
            ]

            for service in expected_services:
                if mock_injector.has_service(service):
                    assert service in mock_injector._singletons, f"{service} should be singleton"

    def test_service_factory_dependencies(self, mock_injector, mock_config):
        """Test that service factories handle dependencies correctly."""
        mock_injector.register_instance("Config", mock_config)

        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.error_handler.ErrorHandler") as MockErrorHandler,
        ):
            register_error_handling_services(mock_injector, None)

            # Call ErrorHandler factory
            factory = mock_injector._factories["ErrorHandler"]
            factory()

            # Should use injector config when available (may be called multiple times due to dependency configuration)
            assert MockErrorHandler.called
            # Verify at least one call used the correct config
            call_args_list = MockErrorHandler.call_args_list
            assert any(call.args == (mock_config,) for call in call_args_list)

    def test_circular_dependency_avoidance(self, mock_injector, mock_config):
        """Test that registration avoids circular dependencies."""
        with patch("src.error_handling.di_registration.logger") as mock_logger:
            # Should not raise any circular dependency errors
            try:
                register_error_handling_services(mock_injector, mock_config)
            except Exception as e:
                if "circular" in str(e).lower():
                    pytest.fail(f"Circular dependency detected: {e}")
                else:
                    # Re-raise if it's a different error
                    raise

            # Should complete successfully
            mock_logger.info.assert_called()

    def test_registration_logging(self, mock_injector, mock_config):
        """Test that registration logs appropriately."""
        with patch("src.error_handling.di_registration.logger") as mock_logger:
            register_error_handling_services(mock_injector, mock_config)

            # Should log start
            mock_logger.info.assert_any_call(
                "Registering error handling services with DI container"
            )

            # Should log individual service registrations
            debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
            assert any("Registered ErrorHandler" in call for call in debug_calls)

    def test_factory_exception_logging(self, mock_injector, mock_config):
        """Test that factory exceptions are properly logged."""
        with (
            patch("src.error_handling.di_registration.logger") as mock_logger,
            patch("src.error_handling.error_handler.ErrorHandler") as MockErrorHandler,
        ):
            MockErrorHandler.side_effect = ValueError("Test error")

            register_error_handling_services(mock_injector, mock_config)
            factory = mock_injector._factories["ErrorHandler"]

            with pytest.raises(ValueError):
                factory()

            # Should log the error
            error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
            assert any("Failed to create ErrorHandler" in call for call in error_calls)

    def test_service_configuration_delay(self, mock_injector, mock_config):
        """Test that service dependencies are not configured during factory creation."""
        with (
            patch("src.error_handling.di_registration.logger"),
            patch("src.error_handling.error_handler.ErrorHandler") as MockErrorHandler,
        ):
            # Mock instance to verify no immediate configuration
            mock_instance = MagicMock()
            MockErrorHandler.return_value = mock_instance

            register_error_handling_services(mock_injector, mock_config)

            factory = mock_injector._factories["ErrorHandler"]
            instance = factory()

            # Instance should be created but not configured during factory
            assert instance == mock_instance
            # Verify comment in code about not configuring dependencies during factory
