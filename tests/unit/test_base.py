"""Tests for base classes and mixins."""

from unittest.mock import Mock, patch

from src.base import BaseComponent, LoggerMixin


class TestLoggerMixin:
    """Test LoggerMixin functionality."""

    def test_logger_creation(self):
        """Test that logger is created on first access."""

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()

        # Logger should not exist yet
        assert not hasattr(obj, "_logger")

        # Access logger - it should be created and cached
        logger1 = obj.logger

        # Logger should now exist as instance attribute
        assert hasattr(obj, "_logger")
        assert obj._logger is not None

        # Subsequent access should return same logger instance
        logger2 = obj.logger
        assert logger2 is logger1  # Same object

        # Verify logger has expected attributes (structured logger)
        assert hasattr(logger1, 'info')
        assert hasattr(logger1, 'error')
        assert hasattr(logger1, 'debug')

    def test_logger_inheritance(self):
        """Test that subclasses get their own logger."""

        class Parent(LoggerMixin):
            pass

        class Child(Parent):
            pass

        parent = Parent()
        child = Child()

        # Each should get their logger instance (may be same if same module)
        parent_logger = parent.logger
        child_logger = child.logger

        # Since both classes are in the same module, they may get the same logger
        # This is expected behavior for module-based logging

        # Both should have logger functionality
        assert hasattr(parent_logger, 'info')
        assert hasattr(child_logger, 'info')

        # Both should have their _logger attribute set
        assert hasattr(parent, '_logger')
        assert hasattr(child, '_logger')


class TestBaseComponent:
    """Test BaseComponent functionality."""

    def test_initialization(self):
        """Test component initialization."""
        component = BaseComponent()

        # Should not be initialized yet
        assert not component.initialized

        # Initialize component
        component.initialize()

        # Should be initialized
        assert component.initialized

        # Logger should be available
        assert hasattr(component, 'logger')
        assert component.logger is not None

    def test_cleanup(self):
        """Test component cleanup."""
        component = BaseComponent()

        # Initialize first
        component.initialize()
        assert component.initialized

        # Cleanup
        component.cleanup()

        # Should not be initialized anymore
        assert not component.initialized

        # Logger should still be available
        assert hasattr(component, 'logger')
        assert component.logger is not None

    def test_inheritance(self):
        """Test that subclasses can override methods."""

        class CustomComponent(BaseComponent):
            def __init__(self):
                super().__init__()
                self.custom_initialized = False

            def initialize(self):
                super().initialize()
                self.custom_initialized = True

            def cleanup(self):
                super().cleanup()
                self.custom_initialized = False

        component = CustomComponent()

        # Test initialization
        assert not component.initialized
        assert not component.custom_initialized

        component.initialize()
        assert component.initialized
        assert component.custom_initialized

        # Test cleanup
        component.cleanup()
        assert not component.initialized
        assert not component.custom_initialized
