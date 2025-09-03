"""
Test suite for monitoring dependency injection module.

Tests cover dependency registration and resolution for monitoring components.
"""

from unittest.mock import Mock, patch

import pytest

from src.monitoring.dependency_injection import (
    setup_monitoring_dependencies,
    get_monitoring_container,
    DIContainer,
)


class TestDependencyRegistration:
    """Test dependency registration functionality."""
    
    def test_setup_monitoring_dependencies(self):
        """Test setup monitoring dependencies."""
        # Should not raise any exceptions
        setup_monitoring_dependencies()

    def test_get_container(self):
        """Test get container function."""
        container = get_monitoring_container()
        assert isinstance(container, DIContainer)

    def test_di_container_creation(self):
        """Test DIContainer creation."""
        container = DIContainer()
        assert container is not None

    def test_di_container_register_singleton(self):
        """Test DIContainer register singleton."""
        container = DIContainer()
        
        # Register a simple singleton
        container.register(str, factory=lambda: "test_value", singleton=True)
        
        # Should be able to resolve it
        result = container.resolve(str)
        assert result == "test_value"

    def test_di_container_register_factory(self):
        """Test DIContainer register factory."""
        container = DIContainer()
        
        # Register a factory (transient)
        container.register(list, factory=lambda: [1, 2, 3], singleton=False)
        
        # Should create new instances each time
        result1 = container.resolve(list)
        result2 = container.resolve(list)
        assert result1 == [1, 2, 3]
        assert result2 == [1, 2, 3]
        assert result1 is not result2  # Different instances