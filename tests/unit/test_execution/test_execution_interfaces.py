"""Optimized unit tests for execution interfaces."""

import logging
from abc import ABC

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

from src.execution.interfaces import ExecutionAlgorithmInterface

# Cache common test configurations
COMMON_ATTRS = {
    "mock_status": "completed",
    "cancelled_status": "cancelled",
    "pending_status": "pending",
    "execution_id": "test_execution_123"
}


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
        assert hasattr(ExecutionAlgorithmInterface, "execute")
        assert hasattr(ExecutionAlgorithmInterface, "cancel")
        assert hasattr(ExecutionAlgorithmInterface, "get_status")

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
                return {"status": COMMON_ATTRS["mock_status"]}

            async def cancel_execution(self, execution_id):
                """Mock cancel_execution implementation."""
                return True

            async def cancel(self, execution_id):
                """Mock cancel implementation."""
                return {"status": COMMON_ATTRS["cancelled_status"]}

            async def get_status(self, execution_id):
                """Mock get_status implementation."""
                return {"status": COMMON_ATTRS["pending_status"]}

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

        # Get method signatures (cached for performance)
        execute_sig = inspect.signature(ExecutionAlgorithmInterface.execute)
        cancel_sig = inspect.signature(ExecutionAlgorithmInterface.cancel)
        status_sig = inspect.signature(ExecutionAlgorithmInterface.get_status)

        # Pre-defined expected parameters for faster comparison
        expected_execute_params = {"self", "instruction", "exchange_factory", "risk_manager"}
        expected_cancel_params = {"self", "execution_id"}
        expected_status_params = {"self", "execution_id"}

        # Check method signatures using set operations for better performance
        execute_params = set(execute_sig.parameters.keys())
        cancel_params = set(cancel_sig.parameters.keys())
        status_params = set(status_sig.parameters.keys())

        assert expected_execute_params.issubset(execute_params)
        assert expected_cancel_params.issubset(cancel_params)
        assert expected_status_params.issubset(status_params)
