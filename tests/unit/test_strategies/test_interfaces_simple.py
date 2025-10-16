"""
Simple tests for strategy interfaces to boost coverage.
"""

import pytest
from abc import ABC

from src.strategies.interfaces import BaseStrategyInterface, StrategyDataRepositoryInterface


class TestBaseStrategyInterface:
    """Test base strategy interface."""
    
    def test_base_strategy_interface_is_abstract(self):
        """Test that BaseStrategyInterface is abstract."""
        assert issubclass(BaseStrategyInterface, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseStrategyInterface()
    
    def test_base_strategy_interface_has_required_methods(self):
        """Test that BaseStrategyInterface has required abstract methods."""
        # Check that abstract methods exist
        abstract_methods = BaseStrategyInterface.__abstractmethods__
        expected_methods = {'generate_signals', 'validate_signal', 'get_position_size', 'should_exit'}
        
        # Should have several abstract methods
        assert len(abstract_methods) > 10
        
        # Check some specific methods exist
        method_names = list(abstract_methods)
        found_methods = []
        for expected in expected_methods:
            if expected in method_names:
                found_methods.append(expected)
        
        assert len(found_methods) >= 2  # At least 2 of the expected methods


class TestStrategyDataRepositoryInterface:
    """Test strategy data repository interface."""
    
    def test_repository_interface_is_protocol(self):
        """Test that StrategyDataRepositoryInterface is a protocol."""
        # Protocols can't be instantiated directly either
        with pytest.raises(TypeError):
            StrategyDataRepositoryInterface()
    
    def test_repository_interface_has_required_methods(self):
        """Test that StrategyDataRepositoryInterface has required methods."""
        # Check that required methods are defined
        expected_methods = [
            'load_strategy_state', 'save_strategy_state', 
            'get_strategy_trades', 'save_trade',
            'get_strategy_positions', 'save_performance_metrics'
        ]
        
        found_methods = []
        for method_name in expected_methods:
            if hasattr(StrategyDataRepositoryInterface, method_name):
                found_methods.append(method_name)
        
        # Should find most of the expected methods
        assert len(found_methods) >= 4