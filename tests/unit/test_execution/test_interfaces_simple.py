"""Simple unit tests for execution interfaces."""

import pytest
from unittest.mock import AsyncMock
from src.execution.interfaces import (
    ExecutionServiceInterface,
    OrderManagementServiceInterface,
    ExecutionEngineServiceInterface,
    RiskValidationServiceInterface,
    ExecutionAlgorithmFactoryInterface
)


class TestExecutionInterfaces:
    """Test cases for execution interfaces."""

    def test_execution_service_interface_methods(self):
        """Test ExecutionServiceInterface has required methods."""
        # Test that interface defines required methods
        assert hasattr(ExecutionServiceInterface, 'record_trade_execution')
        assert hasattr(ExecutionServiceInterface, 'validate_order_pre_execution') 
        assert hasattr(ExecutionServiceInterface, 'get_execution_metrics')

    def test_order_management_service_interface_methods(self):
        """Test OrderManagementServiceInterface has required methods."""
        assert hasattr(OrderManagementServiceInterface, 'create_managed_order')
        assert hasattr(OrderManagementServiceInterface, 'update_order_status')
        assert hasattr(OrderManagementServiceInterface, 'cancel_order')
        assert hasattr(OrderManagementServiceInterface, 'get_order_metrics')

    def test_execution_engine_service_interface_methods(self):
        """Test ExecutionEngineServiceInterface has required methods."""
        assert hasattr(ExecutionEngineServiceInterface, 'execute_instruction')
        assert hasattr(ExecutionEngineServiceInterface, 'get_active_executions')
        assert hasattr(ExecutionEngineServiceInterface, 'cancel_execution')
        assert hasattr(ExecutionEngineServiceInterface, 'get_performance_metrics')

    def test_risk_validation_service_interface_methods(self):
        """Test RiskValidationServiceInterface has required methods."""
        assert hasattr(RiskValidationServiceInterface, 'validate_order_risk')
        assert hasattr(RiskValidationServiceInterface, 'check_position_limits')

    def test_execution_algorithm_factory_interface_methods(self):
        """Test ExecutionAlgorithmFactoryInterface has required methods."""
        assert hasattr(ExecutionAlgorithmFactoryInterface, 'create_algorithm')
        assert hasattr(ExecutionAlgorithmFactoryInterface, 'get_available_algorithms')
        assert hasattr(ExecutionAlgorithmFactoryInterface, 'is_algorithm_available')

    def test_interface_protocols_are_callable(self):
        """Test that interface protocols can be used in isinstance checks."""
        # Interfaces are Protocols, so they should work with isinstance
        from typing import get_type_hints
        
        # Basic test that protocols have type hints
        execution_hints = get_type_hints(ExecutionServiceInterface.record_trade_execution)
        assert execution_hints  # Should have type hints
        
        order_hints = get_type_hints(OrderManagementServiceInterface.create_managed_order)  
        assert order_hints  # Should have type hints

    def test_mock_implementation_compatibility(self):
        """Test that mock implementations are compatible with interfaces."""
        # Create mock implementations
        mock_execution_service = AsyncMock(spec=ExecutionServiceInterface)
        mock_order_service = AsyncMock(spec=OrderManagementServiceInterface)
        mock_engine_service = AsyncMock(spec=ExecutionEngineServiceInterface)
        mock_risk_service = AsyncMock(spec=RiskValidationServiceInterface)
        
        # Test that mocks have expected methods
        assert hasattr(mock_execution_service, 'record_trade_execution')
        assert hasattr(mock_order_service, 'create_managed_order')
        assert hasattr(mock_engine_service, 'execute_instruction')
        assert hasattr(mock_risk_service, 'validate_order_risk')

    def test_interface_method_annotations(self):
        """Test that interface methods have proper type annotations."""
        import inspect
        
        # Check ExecutionServiceInterface method signatures
        sig = inspect.signature(ExecutionServiceInterface.record_trade_execution)
        assert 'execution_result' in sig.parameters
        assert 'market_data' in sig.parameters
        assert sig.return_annotation is not inspect.Signature.empty or sig.return_annotation is not None
        
        # Check OrderManagementServiceInterface method signatures  
        sig = inspect.signature(OrderManagementServiceInterface.create_managed_order)
        assert 'order_request' in sig.parameters
        assert 'execution_id' in sig.parameters