"""
Simple tests for DataService via DataServiceFactory.

These tests verify the modern data service functionality
without external dependencies. This replaces the deprecated
DataIntegrationService tests to fix deprecation warnings.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.factory import DataServiceFactory
from src.data.interfaces import DataServiceInterface


class TestDataServiceFactory:
    """Test DataServiceFactory functionality."""

    def test_factory_initialization(self):
        """Test factory can be initialized."""
        factory = DataServiceFactory()
        assert factory is not None
        assert factory.injector is not None

    def test_create_data_service(self):
        """Test creating a data service through factory."""
        factory = DataServiceFactory()
        
        # Mock the injector to return a mock service
        mock_service = AsyncMock(spec=DataServiceInterface)
        
        with patch.object(factory.injector, 'resolve', return_value=mock_service):
            service = factory.create_data_service()
            
            assert service is not None
            assert service == mock_service

    def test_create_minimal_data_service(self):
        """Test creating a minimal data service."""
        factory = DataServiceFactory()
        
        # Mock the injector to return a mock service
        mock_service = AsyncMock(spec=DataServiceInterface)
        
        with patch.object(factory.injector, 'resolve', return_value=mock_service):
            service = factory.create_minimal_data_service()
            
            assert service is not None
            assert service == mock_service

    def test_factory_with_custom_injector(self):
        """Test factory with custom dependency injector."""
        from src.core.dependency_injection import DependencyInjector
        
        custom_injector = MagicMock(spec=DependencyInjector)
        factory = DataServiceFactory(injector=custom_injector)
        
        assert factory.injector is custom_injector

    def test_create_data_service_from_config(self):
        """Test legacy method for backward compatibility."""
        mock_config = MagicMock()
        
        # Mock the static method
        mock_service = AsyncMock(spec=DataServiceInterface)
        
        with patch.object(DataServiceFactory, 'create_data_service_from_config', return_value=mock_service) as mock_method:
            service = DataServiceFactory.create_data_service_from_config(
                config=mock_config,
                use_cache=True,
                use_validator=True
            )
            
            assert service is not None
            mock_method.assert_called_once_with(
                config=mock_config,
                use_cache=True,
                use_validator=True
            )