"""Tests for base classes and mixins."""

import pytest
from unittest.mock import Mock, patch

from src.base import LoggerMixin, BaseComponent


class TestLoggerMixin:
    """Test LoggerMixin functionality."""
    
    def test_logger_creation(self):
        """Test that logger is created on first access."""
        
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        
        # Logger should not exist yet
        assert not hasattr(obj, '_logger')
        
        # Access logger
        with patch('src.base.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            logger = obj.logger
            
            # Logger should be created
            assert logger == mock_logger
            mock_get_logger.assert_called_once_with(TestClass.__module__)
            
            # Subsequent access should return same logger
            logger2 = obj.logger
            assert logger2 == mock_logger
            assert mock_get_logger.call_count == 1  # Not called again
    
    def test_logger_inheritance(self):
        """Test that subclasses get their own logger."""
        
        class Parent(LoggerMixin):
            pass
        
        class Child(Parent):
            pass
        
        with patch('src.base.get_logger') as mock_get_logger:
            parent = Parent()
            child = Child()
            
            # Each should get logger for their own module
            parent_logger = parent.logger
            child_logger = child.logger
            
            calls = mock_get_logger.call_args_list
            assert len(calls) == 2
            assert Parent.__module__ in str(calls[0])
            assert Child.__module__ in str(calls[1])


class TestBaseComponent:
    """Test BaseComponent functionality."""
    
    def test_initialization(self):
        """Test component initialization."""
        component = BaseComponent()
        
        # Should not be initialized yet
        assert not component.initialized
        
        # Initialize
        with patch.object(component, 'logger') as mock_logger:
            component.initialize()
            
            # Should be initialized
            assert component.initialized
            mock_logger.debug.assert_called_once()
    
    def test_cleanup(self):
        """Test component cleanup."""
        component = BaseComponent()
        component.initialize()
        
        assert component.initialized
        
        # Cleanup
        with patch.object(component, 'logger') as mock_logger:
            component.cleanup()
            
            # Should not be initialized
            assert not component.initialized
            mock_logger.debug.assert_called_once()
    
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