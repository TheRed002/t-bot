"""
Simple tests for DataService via DataServiceFactory.

These tests verify the modern data service functionality
without external dependencies. This replaces the deprecated
DataIntegrationService tests to fix deprecation warnings.
"""

import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.dependency_injection import DependencyInjector
from src.data.factory import DataServiceFactory, create_default_data_service
from src.data.interfaces import DataServiceInterface


class TestDataServiceFactory:
    """Test DataServiceFactory functionality."""

    def test_factory_initialization_with_injector(self):
        """Test factory can be initialized with injector."""
        mock_injector = MagicMock(spec=DependencyInjector)
        factory = DataServiceFactory(injector=mock_injector)
        assert factory is not None
        assert factory._injector is mock_injector

    def test_factory_initialization_without_injector_raises_error(self):
        """Test factory raises error without injector."""
        import pytest

        from src.core.exceptions import ComponentError

        with pytest.raises(ComponentError, match="Injector must be provided to factory"):
            DataServiceFactory()

    def test_create_data_service(self):
        """Test creating a data service through factory."""
        mock_injector = MagicMock(spec=DependencyInjector)
        factory = DataServiceFactory(injector=mock_injector)

        # Mock the injector to return a mock service
        mock_service = AsyncMock(spec=DataServiceInterface)
        mock_injector.resolve.return_value = mock_service

        service = factory.create_data_service()

        assert service is not None
        assert service == mock_service
        mock_injector.resolve.assert_called_once_with("DataServiceInterface")

    def test_create_minimal_data_service(self):
        """Test creating a minimal data service."""
        mock_injector = MagicMock(spec=DependencyInjector)
        factory = DataServiceFactory(injector=mock_injector)

        # Mock the injector to return a mock service
        mock_service = AsyncMock(spec=DataServiceInterface)
        mock_injector.resolve.return_value = mock_service

        service = factory.create_minimal_data_service()

        assert service is not None
        assert service == mock_service
        mock_injector.resolve.assert_called_once_with("DataServiceInterface")

    def test_factory_with_custom_injector(self):
        """Test factory with custom dependency injector."""
        custom_injector = MagicMock(spec=DependencyInjector)
        factory = DataServiceFactory(injector=custom_injector)

        assert factory._injector is custom_injector

    def test_create_default_data_service(self):
        """Test default data service creation function."""
        mock_config = MagicMock()
        mock_service = AsyncMock(spec=DataServiceInterface)

        with patch("src.data.di_registration.configure_data_dependencies") as mock_configure:
            mock_injector = MagicMock(spec=DependencyInjector)
            mock_injector.resolve.return_value = mock_service
            mock_injector.register_factory = MagicMock()
            mock_configure.return_value = mock_injector

            service = create_default_data_service(config=mock_config)

            assert service is not None
            assert service == mock_service
            mock_configure.assert_called_once()
            mock_injector.register_factory.assert_called_once_with("Config", unittest.mock.ANY, singleton=True)
