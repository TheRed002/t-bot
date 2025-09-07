"""
Comprehensive tests for analytics factory module.

Tests the factory pattern implementation and dependency injection
for creating analytics services.
"""

# Disable logging during tests for performance
import logging
from unittest.mock import Mock, patch

import pytest

logging.disable(logging.CRITICAL)

# Set pytest markers for optimization
pytestmark = pytest.mark.unit

from src.analytics.factory import AnalyticsServiceFactory, create_default_analytics_service
from src.analytics.interfaces import (
    AlertServiceProtocol,
    ExportServiceProtocol,
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    ReportingServiceProtocol,
    RiskServiceProtocol,
)
from src.analytics.types import AnalyticsConfiguration, AnalyticsFrequency


class TestAnalyticsServiceFactory:
    """Test AnalyticsServiceFactory class."""

    @pytest.fixture
    def mock_injector(self):
        """Mock dependency injector (fresh for each test)."""
        injector = Mock()
        injector.resolve = Mock()
        return injector

    @pytest.fixture
    def factory_with_mock_injector(self, mock_injector):
        """Factory with mock injector."""
        return AnalyticsServiceFactory(injector=mock_injector)

    def test_factory_init_with_injector(self, mock_injector):
        """Test factory initialization with provided injector."""
        factory = AnalyticsServiceFactory(injector=mock_injector)
        assert factory._injector == mock_injector

    def test_factory_init_without_injector(self):
        """Test factory initialization without injector raises error."""
        # Factory now requires injector to be provided
        with pytest.raises(Exception):  # Should raise ComponentError
            factory = AnalyticsServiceFactory()

    def test_create_analytics_service(self, factory_with_mock_injector, mock_injector):
        """Test creating analytics service."""
        mock_service = Mock()
        mock_injector.resolve.return_value = mock_service

        result = factory_with_mock_injector.create_analytics_service()

        mock_injector.resolve.assert_called_once_with("AnalyticsService")
        assert result == mock_service

    def test_create_analytics_service_with_config(self, factory_with_mock_injector, mock_injector):
        """Test creating analytics service with configuration."""
        mock_service = Mock()
        mock_injector.resolve.return_value = mock_service

        config = AnalyticsConfiguration(
            reporting_frequency=AnalyticsFrequency.WEEKLY, currency="EUR"
        )

        result = factory_with_mock_injector.create_analytics_service(config)

        mock_injector.resolve.assert_called_once_with("AnalyticsService")
        assert result == mock_service

    def test_create_portfolio_service(self, factory_with_mock_injector, mock_injector):
        """Test creating portfolio service."""
        mock_service = Mock(spec=PortfolioServiceProtocol)
        mock_injector.resolve.return_value = mock_service

        result = factory_with_mock_injector.create_portfolio_service()

        mock_injector.resolve.assert_called_once_with("PortfolioServiceProtocol")
        assert result == mock_service

    def test_create_risk_service(self, factory_with_mock_injector, mock_injector):
        """Test creating risk service."""
        mock_service = Mock(spec=RiskServiceProtocol)
        mock_injector.resolve.return_value = mock_service

        result = factory_with_mock_injector.create_risk_service()

        mock_injector.resolve.assert_called_once_with("RiskServiceProtocol")
        assert result == mock_service

    def test_create_reporting_service(self, factory_with_mock_injector, mock_injector):
        """Test creating reporting service."""
        mock_service = Mock(spec=ReportingServiceProtocol)
        mock_injector.resolve.return_value = mock_service

        result = factory_with_mock_injector.create_reporting_service()

        mock_injector.resolve.assert_called_once_with("ReportingServiceProtocol")
        assert result == mock_service

    def test_create_operational_service(self, factory_with_mock_injector, mock_injector):
        """Test creating operational service."""
        mock_service = Mock(spec=OperationalServiceProtocol)
        mock_injector.resolve.return_value = mock_service

        result = factory_with_mock_injector.create_operational_service()

        mock_injector.resolve.assert_called_once_with("OperationalServiceProtocol")
        assert result == mock_service

    def test_create_alert_service(self, factory_with_mock_injector, mock_injector):
        """Test creating alert service."""
        mock_service = Mock(spec=AlertServiceProtocol)
        mock_injector.resolve.return_value = mock_service

        result = factory_with_mock_injector.create_alert_service()

        mock_injector.resolve.assert_called_once_with("AlertServiceProtocol")
        assert result == mock_service

    def test_create_export_service(self, factory_with_mock_injector, mock_injector):
        """Test creating export service."""
        mock_service = Mock(spec=ExportServiceProtocol)
        mock_injector.resolve.return_value = mock_service

        result = factory_with_mock_injector.create_export_service()

        mock_injector.resolve.assert_called_once_with("ExportServiceProtocol")
        assert result == mock_service

    def test_factory_reuses_injector_across_calls(self, factory_with_mock_injector, mock_injector):
        """Test factory reuses the same injector for multiple service creations."""
        mock_portfolio_service = Mock(spec=PortfolioServiceProtocol)
        mock_risk_service = Mock(spec=RiskServiceProtocol)

        mock_injector.resolve.side_effect = [mock_portfolio_service, mock_risk_service]

        portfolio_result = factory_with_mock_injector.create_portfolio_service()
        risk_result = factory_with_mock_injector.create_risk_service()

        assert mock_injector.resolve.call_count == 2
        assert portfolio_result == mock_portfolio_service
        assert risk_result == mock_risk_service

    def test_injector_resolve_error_propagation(self, factory_with_mock_injector, mock_injector):
        """Test that injector resolution errors are properly propagated."""
        mock_injector.resolve.side_effect = Exception("Dependency injection failed")

        with pytest.raises(Exception, match="Dependency injection failed"):
            factory_with_mock_injector.create_analytics_service()

    def test_factory_type_checking(self, factory_with_mock_injector):
        """Test that factory properly handles type checking scenarios."""
        # This test ensures the TYPE_CHECKING import works correctly
        from src.analytics.factory import TYPE_CHECKING

        assert isinstance(TYPE_CHECKING, bool)


class TestCreateDefaultAnalyticsService:
    """Test create_default_analytics_service function."""

    @patch("src.analytics.factory.AnalyticsServiceFactory")
    @patch("src.analytics.di_registration.configure_analytics_dependencies")
    def test_create_default_service_without_params(self, mock_configure, mock_factory_class):
        """Test creating default service without parameters."""
        mock_injector = Mock()
        mock_configure.return_value = mock_injector
        mock_factory_instance = Mock()
        mock_service = Mock()
        mock_factory_class.return_value = mock_factory_instance
        mock_factory_instance.create_analytics_service.return_value = mock_service

        result = create_default_analytics_service()

        mock_configure.assert_called_once()
        mock_factory_class.assert_called_once_with(injector=mock_injector)
        mock_factory_instance.create_analytics_service.assert_called_once_with(None)
        assert result == mock_service

    @patch("src.analytics.factory.AnalyticsServiceFactory")
    @patch("src.analytics.di_registration.configure_analytics_dependencies")
    def test_create_default_service_with_config(self, mock_configure, mock_factory_class):
        """Test creating default service with configuration."""
        mock_injector = Mock()
        mock_configure.return_value = mock_injector
        mock_factory_instance = Mock()
        mock_service = Mock()
        mock_factory_class.return_value = mock_factory_instance
        mock_factory_instance.create_analytics_service.return_value = mock_service

        config = AnalyticsConfiguration(currency="GBP")

        result = create_default_analytics_service(config=config)

        mock_configure.assert_called_once()
        mock_factory_class.assert_called_once_with(injector=mock_injector)
        mock_factory_instance.create_analytics_service.assert_called_once_with(config)
        assert result == mock_service

    @patch("src.analytics.factory.AnalyticsServiceFactory")
    def test_create_default_service_with_injector(self, mock_factory_class):
        """Test creating default service with custom injector."""
        mock_factory_instance = Mock()
        mock_service = Mock()
        mock_factory_class.return_value = mock_factory_instance
        mock_factory_instance.create_analytics_service.return_value = mock_service

        custom_injector = Mock()

        result = create_default_analytics_service(injector=custom_injector)

        mock_factory_class.assert_called_once_with(injector=custom_injector)
        mock_factory_instance.create_analytics_service.assert_called_once_with(None)
        assert result == mock_service

    @patch("src.analytics.factory.AnalyticsServiceFactory")
    def test_create_default_service_with_all_params(self, mock_factory_class):
        """Test creating default service with all parameters."""
        mock_factory_instance = Mock()
        mock_service = Mock()
        mock_factory_class.return_value = mock_factory_instance
        mock_factory_instance.create_analytics_service.return_value = mock_service

        config = AnalyticsConfiguration(
            risk_free_rate="0.035", reporting_frequency=AnalyticsFrequency.MONTHLY
        )
        custom_injector = Mock()

        result = create_default_analytics_service(config=config, injector=custom_injector)

        mock_factory_class.assert_called_once_with(injector=custom_injector)
        mock_factory_instance.create_analytics_service.assert_called_once_with(config)
        assert result == mock_service

    @patch("src.analytics.factory.AnalyticsServiceFactory")
    def test_create_default_service_factory_error_propagation(self, mock_factory_class):
        """Test that factory creation errors are properly propagated."""
        mock_factory_class.side_effect = Exception("Factory creation failed")

        with pytest.raises(Exception, match="Factory creation failed"):
            create_default_analytics_service()

    @patch("src.analytics.factory.AnalyticsServiceFactory")
    def test_create_default_service_creation_error_propagation(self, mock_factory_class):
        """Test that service creation errors are properly propagated."""
        mock_factory_instance = Mock()
        mock_factory_class.return_value = mock_factory_instance
        mock_factory_instance.create_analytics_service.side_effect = Exception(
            "Service creation failed"
        )

        with pytest.raises(Exception, match="Service creation failed"):
            create_default_analytics_service()


class TestFactoryIntegration:
    """Integration tests for factory functionality."""

    def test_factory_service_creation_consistency(self):
        """Test that factory creates services consistently."""
        # This is a basic integration test that ensures the factory
        # can be instantiated and doesn't fail on basic operations
        # without full dependency resolution

        mock_injector = Mock()
        factory = AnalyticsServiceFactory(injector=mock_injector)

        # Ensure factory has the correct injector
        assert factory._injector == mock_injector

        # Ensure all service creation methods exist
        assert hasattr(factory, "create_analytics_service")
        assert hasattr(factory, "create_portfolio_service")
        assert hasattr(factory, "create_risk_service")
        assert hasattr(factory, "create_reporting_service")
        assert hasattr(factory, "create_operational_service")
        assert hasattr(factory, "create_alert_service")
        assert hasattr(factory, "create_export_service")

    def test_factory_method_return_types(self):
        """Test that factory methods have correct return type hints."""
        # This test ensures proper typing is maintained
        from typing import get_type_hints

        factory = AnalyticsServiceFactory(injector=Mock())

        # Check that methods have appropriate return type hints
        portfolio_hints = get_type_hints(factory.create_portfolio_service)
        assert "return" in portfolio_hints

        risk_hints = get_type_hints(factory.create_risk_service)
        assert "return" in risk_hints

        reporting_hints = get_type_hints(factory.create_reporting_service)
        assert "return" in reporting_hints

    def test_default_injector_configuration(self):
        """Test that factory validates injector requirement."""
        # Factory now requires injector - test that it validates this
        with pytest.raises(Exception):  # Should raise ComponentError
            factory = AnalyticsServiceFactory()

        # Test that providing injector works
        mock_injector = Mock()
        factory = AnalyticsServiceFactory(injector=mock_injector)
        assert factory._injector == mock_injector


class TestFactoryErrorHandling:
    """Test error handling in factory operations."""

    def test_injector_none_handling(self):
        """Test handling when injector is None."""
        # Factory now requires injector - should raise error when None
        with pytest.raises(Exception):  # Should raise ComponentError
            factory = AnalyticsServiceFactory(injector=None)

    def test_service_resolution_failure_handling(self):
        """Test handling of service resolution failures."""
        mock_injector = Mock()
        mock_injector.resolve.side_effect = KeyError("Service not found")

        factory = AnalyticsServiceFactory(injector=mock_injector)

        with pytest.raises(KeyError, match="Service not found"):
            factory.create_portfolio_service()

    def test_injector_method_missing(self):
        """Test handling when injector doesn't have resolve method."""
        invalid_injector = object()  # Object without resolve method

        factory = AnalyticsServiceFactory(injector=invalid_injector)

        with pytest.raises(AttributeError):
            factory.create_analytics_service()

    def test_configuration_validation_error_propagation(self):
        """Test that configuration validation errors are propagated."""
        mock_injector = Mock()
        factory = AnalyticsServiceFactory(injector=mock_injector)

        # Create invalid configuration that might cause validation errors
        # This test ensures any validation errors during service creation are propagated
        invalid_config = object()  # Invalid config object

        # The factory should pass the config to the service creation,
        # any validation errors should bubble up
        mock_injector.resolve.return_value = Mock()

        # Should not raise an error at factory level (config is passed through)
        result = factory.create_analytics_service(invalid_config)
        assert result is not None
