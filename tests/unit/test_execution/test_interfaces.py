"""Unit tests for execution interfaces."""

import pytest
from abc import ABC
from src.execution.interfaces import ExecutionAlgorithmInterface


class TestExecutionAlgorithmInterface:
    """Test cases for ExecutionAlgorithmInterface."""

    def test_interface_is_abstract(self):
        """Test that ExecutionAlgorithmInterface is an abstract base class."""
        assert issubclass(ExecutionAlgorithmInterface, ABC)

    def test_interface_cannot_be_instantiated(self):
        """Test that ExecutionAlgorithmInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExecutionAlgorithmInterface()

    def test_interface_has_required_methods(self):
        """Test that interface defines required methods."""
        # Check that required methods exist
        assert hasattr(ExecutionAlgorithmInterface, 'execute')
        assert hasattr(ExecutionAlgorithmInterface, 'cancel')
        assert hasattr(ExecutionAlgorithmInterface, 'get_status')

    def test_concrete_implementation_requirements(self):
        """Test that concrete implementations must implement abstract methods."""
        
        class IncompleteAlgorithm(ExecutionAlgorithmInterface):
            """Incomplete implementation for testing."""
            pass
        
        # Should not be able to instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            IncompleteAlgorithm()

    def test_valid_concrete_implementation(self):
        """Test that a complete concrete implementation can be instantiated."""
        
        class CompleteAlgorithm(ExecutionAlgorithmInterface):
            """Complete implementation for testing."""
            
            async def execute(self, instruction, exchange_factory, risk_manager):
                """Mock execute implementation."""
                return {"status": "completed"}
                
            async def cancel_execution(self, execution_id):
                """Mock cancel_execution implementation."""
                return True
                
            async def cancel(self, execution_id):
                """Mock cancel implementation."""
                return {"status": "cancelled"}
                
            async def get_status(self, execution_id):
                """Mock get_status implementation."""
                return {"status": "pending"}
                
            def get_algorithm_type(self):
                """Mock get_algorithm_type implementation."""
                from src.core.types import ExecutionAlgorithm
                return ExecutionAlgorithm.TWAP
        
        # Should be able to instantiate
        algorithm = CompleteAlgorithm()
        assert isinstance(algorithm, ExecutionAlgorithmInterface)
        assert isinstance(algorithm, CompleteAlgorithm)

    def test_interface_method_signatures(self):
        """Test interface method signatures are correctly defined."""
        import inspect
        
        # Get method signatures
        execute_sig = inspect.signature(ExecutionAlgorithmInterface.execute)
        cancel_sig = inspect.signature(ExecutionAlgorithmInterface.cancel)
        status_sig = inspect.signature(ExecutionAlgorithmInterface.get_status)
        
        # Check execute method signature
        execute_params = list(execute_sig.parameters.keys())
        assert 'self' in execute_params
        assert 'instruction' in execute_params
        assert 'exchange_factory' in execute_params
        assert 'risk_manager' in execute_params
        
        # Check cancel method signature  
        cancel_params = list(cancel_sig.parameters.keys())
        assert 'self' in cancel_params
        assert 'execution_id' in cancel_params
        
        # Check get_status method signature
        status_params = list(status_sig.parameters.keys())
        assert 'self' in status_params
        assert 'execution_id' in status_params