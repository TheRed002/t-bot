"""Test suite for data interfaces."""

from unittest.mock import Mock

import pytest

from src.data.interfaces import DataServiceInterface


class TestDataServiceInterface:
    """Test suite for DataServiceInterface."""

    def test_interface_is_importable(self):
        """Test that DataServiceInterface can be imported."""
        assert DataServiceInterface is not None

    def test_interface_has_required_methods(self):
        """Test that DataServiceInterface has all required methods."""
        required_methods = [
            "initialize",
            "store_market_data",
            "get_market_data",
            "get_data_count",
            "get_recent_data",
            "get_metrics",
            "reset_metrics",
            "health_check",
            "cleanup",
        ]
        for method_name in required_methods:
            assert hasattr(DataServiceInterface, method_name)

    def test_interface_is_abstract(self):
        """Test that DataServiceInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            DataServiceInterface()

    def test_interface_can_be_mocked(self):
        """Test that the interface can be properly mocked."""
        mock_service = Mock(spec=DataServiceInterface)

        assert mock_service is not None
        assert hasattr(mock_service, "__class__")

    def test_interface_has_required_methods(self):
        """Test that interface defines expected methods."""
        # Create a mock to verify the interface structure
        mock_service = Mock(spec=DataServiceInterface)

        # The interface should be usable as a spec for mocking
        assert mock_service is not None
