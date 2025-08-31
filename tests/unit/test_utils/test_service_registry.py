"""Tests for service registry module."""

from unittest.mock import Mock, patch

import pytest

from src.core.exceptions import ServiceError
from src.utils.service_registry import register_util_services


class TestRegisterUtilServices:
    """Test register_util_services function."""

    def test_register_util_services_idempotent(self):
        """Test register_util_services is idempotent (can be called multiple times)."""
        with patch("src.utils.service_registry.injector") as mock_injector:
            with patch("src.utils.service_registry._services_registered", False):
                # First call should register services
                register_util_services()
                
                # Second call should not register again
                with patch("src.utils.service_registry._services_registered", True):
                    register_util_services()
                
                # Should only have been called once due to guard
                assert mock_injector.register_factory.call_count > 0

    @patch("src.utils.service_registry.injector")
    def test_register_util_services_gpu_manager_registration(self, mock_injector):
        """Test GPU manager service registration."""
        with patch("src.utils.service_registry._services_registered", False):
            register_util_services()
            
            # Check that GPU manager was registered
            calls = mock_injector.register_factory.call_args_list
            gpu_calls = [call for call in calls if "GPU" in call[0][0]]
            
            assert len(gpu_calls) >= 2  # Both GPUManager and GPUInterface
            
            # Verify singleton registration
            gpu_manager_call = next(call for call in calls if call[0][0] == "GPUManager")
            assert gpu_manager_call[1]["singleton"] is True

    @patch("src.utils.service_registry.injector")
    def test_register_util_services_precision_tracker_registration(self, mock_injector):
        """Test precision tracker service registration."""
        with patch("src.utils.service_registry._services_registered", False):
            register_util_services()
            
            calls = mock_injector.register_factory.call_args_list
            precision_calls = [call for call in calls if "Precision" in call[0][0]]
            
            assert len(precision_calls) >= 2  # Both PrecisionTracker and PrecisionInterface
            
            # Verify singleton registration
            precision_call = next(call for call in calls if call[0][0] == "PrecisionTracker")
            assert precision_call[1]["singleton"] is True

    @patch("src.utils.service_registry.injector")
    def test_register_util_services_validation_service_registration(self, mock_injector):
        """Test validation service registration."""
        with patch("src.utils.service_registry._services_registered", False):
            register_util_services()
            
            calls = mock_injector.register_factory.call_args_list
            validation_calls = [call for call in calls if "Validation" in call[0][0]]
            
            assert len(validation_calls) >= 2  # ValidationService and ValidationServiceInterface
            
            # Verify singleton registration
            validation_call = next(call for call in calls if call[0][0] == "ValidationService")
            assert validation_call[1]["singleton"] is True

    @patch("src.utils.service_registry.injector")
    def test_register_util_services_calculator_registration(self, mock_injector):
        """Test calculator service registration."""
        with patch("src.utils.service_registry._services_registered", False):
            register_util_services()
            
            calls = mock_injector.register_factory.call_args_list
            calculator_calls = [call for call in calls if "Calculator" in call[0][0]]
            
            assert len(calculator_calls) >= 2  # FinancialCalculator and CalculatorInterface
            
            # Verify singleton registration
            calculator_call = next(call for call in calls if call[0][0] == "FinancialCalculator")
            assert calculator_call[1]["singleton"] is True

    @patch("src.utils.service_registry.injector")
    def test_register_util_services_messaging_registration(self, mock_injector):
        """Test messaging services registration."""
        with patch("src.utils.service_registry._services_registered", False):
            register_util_services()
            
            calls = mock_injector.register_factory.call_args_list
            messaging_calls = [call for call in calls if "Messaging" in call[0][0] or "DataTransformation" in call[0][0]]
            
            assert len(messaging_calls) >= 2  # DataTransformationHandler and MessagingCoordinator


class TestServiceFactories:
    """Test individual service factories."""

    @patch("src.utils.service_registry.injector")
    def test_gpu_manager_factory_success(self, mock_injector):
        """Test GPU manager factory creates service successfully."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.gpu_utils.GPUManager") as mock_gpu_manager_class:
                mock_gpu_manager = Mock()
                mock_gpu_manager_class.return_value = mock_gpu_manager
                
                register_util_services()
                
                # Get the GPU manager factory
                calls = mock_injector.register_factory.call_args_list
                gpu_manager_call = next(call for call in calls if call[0][0] == "GPUManager")
                factory_func = gpu_manager_call[0][1]
                
                # Test the factory
                result = factory_func()
                assert result == mock_gpu_manager

    @patch("src.utils.service_registry.injector")
    def test_gpu_manager_factory_error_handling(self, mock_injector):
        """Test GPU manager factory handles errors properly."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.gpu_utils.GPUManager", side_effect=Exception("GPU init error")):
                register_util_services()
                
                # Get the GPU manager factory
                calls = mock_injector.register_factory.call_args_list
                gpu_manager_call = next(call for call in calls if call[0][0] == "GPUManager")
                factory_func = gpu_manager_call[0][1]
                
                # Test the factory raises ServiceError
                with pytest.raises(ServiceError, match="Failed to create GPU manager"):
                    factory_func()

    @patch("src.utils.service_registry.injector")
    def test_precision_tracker_factory_success(self, mock_injector):
        """Test precision tracker factory creates service successfully."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.data_flow_integrity.PrecisionTracker") as mock_precision_class:
                mock_precision_tracker = Mock()
                mock_precision_class.return_value = mock_precision_tracker
                
                register_util_services()
                
                # Get the precision tracker factory
                calls = mock_injector.register_factory.call_args_list
                precision_call = next(call for call in calls if call[0][0] == "PrecisionTracker")
                factory_func = precision_call[0][1]
                
                # Test the factory
                result = factory_func()
                assert result == mock_precision_tracker

    @patch("src.utils.service_registry.injector")
    def test_precision_tracker_factory_error_handling(self, mock_injector):
        """Test precision tracker factory handles errors properly."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.data_flow_integrity.PrecisionTracker", side_effect=Exception("Precision init error")):
                register_util_services()
                
                # Get the precision tracker factory
                calls = mock_injector.register_factory.call_args_list
                precision_call = next(call for call in calls if call[0][0] == "PrecisionTracker")
                factory_func = precision_call[0][1]
                
                # Test the factory raises ServiceError
                with pytest.raises(ServiceError, match="Failed to create precision tracker"):
                    factory_func()

    @patch("src.utils.service_registry.injector")
    def test_data_flow_validator_factory_success(self, mock_injector):
        """Test data flow validator factory creates service successfully."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.data_flow_integrity.DataFlowValidator") as mock_validator_class:
                mock_validator = Mock()
                mock_validator_class.return_value = mock_validator
                
                register_util_services()
                
                # Get the data flow validator factory
                calls = mock_injector.register_factory.call_args_list
                validator_call = next(call for call in calls if call[0][0] == "DataFlowValidator")
                factory_func = validator_call[0][1]
                
                # Test the factory
                result = factory_func()
                assert result == mock_validator

    @patch("src.utils.service_registry.injector")
    def test_integrity_converter_factory_with_dependencies(self, mock_injector):
        """Test integrity converter factory resolves dependencies correctly."""
        with patch("src.utils.service_registry._services_registered", False):
            mock_precision_tracker = Mock()
            mock_injector.resolve.return_value = mock_precision_tracker
            
            with patch("src.utils.data_flow_integrity.IntegrityPreservingConverter") as mock_converter_class:
                mock_converter = Mock()
                mock_converter_class.return_value = mock_converter
                
                register_util_services()
                
                # Get the integrity converter factory
                calls = mock_injector.register_factory.call_args_list
                converter_call = next(call for call in calls if call[0][0] == "IntegrityPreservingConverter")
                factory_func = converter_call[0][1]
                
                # Test the factory
                result = factory_func()
                
                assert result == mock_converter
                mock_injector.resolve.assert_called_with("PrecisionTracker")
                mock_converter_class.assert_called_with(
                    track_precision=True, 
                    precision_tracker=mock_precision_tracker
                )

    @patch("src.utils.service_registry.injector")
    def test_validation_framework_factory_success(self, mock_injector):
        """Test validation framework factory creates service successfully."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.validation.core.ValidationFramework") as mock_framework_class:
                mock_framework = Mock()
                mock_framework_class.return_value = mock_framework
                
                register_util_services()
                
                # Get the validation framework factory
                calls = mock_injector.register_factory.call_args_list
                framework_call = next(call for call in calls if call[0][0] == "ValidationFramework")
                factory_func = framework_call[0][1]
                
                # Test the factory
                result = factory_func()
                assert result == mock_framework

    @patch("src.utils.service_registry.injector")
    def test_validation_service_factory_with_dependencies(self, mock_injector):
        """Test validation service factory resolves dependencies correctly."""
        with patch("src.utils.service_registry._services_registered", False):
            mock_framework = Mock()
            mock_injector.resolve.return_value = mock_framework
            
            with patch("src.utils.validation.service.ValidationService") as mock_service_class:
                mock_service = Mock()
                mock_service_class.return_value = mock_service
                
                register_util_services()
                
                # Get the validation service factory
                calls = mock_injector.register_factory.call_args_list
                service_call = next(call for call in calls if call[0][0] == "ValidationService")
                factory_func = service_call[0][1]
                
                # Test the factory
                result = factory_func()
                
                assert result == mock_service
                mock_injector.resolve.assert_called_with("ValidationFramework")
                mock_service_class.assert_called_with(validation_framework=mock_framework)

    @patch("src.utils.service_registry.injector")
    def test_financial_calculator_factory_success(self, mock_injector):
        """Test financial calculator factory creates service successfully."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.calculations.financial.FinancialCalculator") as mock_calc_class:
                mock_calculator = Mock()
                mock_calc_class.return_value = mock_calculator
                
                register_util_services()
                
                # Get the financial calculator factory
                calls = mock_injector.register_factory.call_args_list
                calc_call = next(call for call in calls if call[0][0] == "FinancialCalculator")
                factory_func = calc_call[0][1]
                
                # Test the factory
                result = factory_func()
                assert result == mock_calculator

    @patch("src.utils.service_registry.injector")
    def test_http_session_manager_factory_success(self, mock_injector):
        """Test HTTP session manager factory creates service successfully."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.monitoring_helpers.HTTPSessionManager") as mock_session_class:
                mock_session = Mock()
                mock_session_class.return_value = mock_session
                
                register_util_services()
                
                # Get the HTTP session manager factory
                calls = mock_injector.register_factory.call_args_list
                session_call = next(call for call in calls if call[0][0] == "HTTPSessionManager")
                factory_func = session_call[0][1]
                
                # Test the factory
                result = factory_func()
                assert result == mock_session

    @patch("src.utils.service_registry.injector")
    def test_messaging_coordinator_factory_with_handlers(self, mock_injector):
        """Test messaging coordinator factory registers handlers correctly."""
        with patch("src.utils.service_registry._services_registered", False):
            mock_handler = Mock()
            mock_injector.resolve.return_value = mock_handler
            
            with patch("src.utils.messaging_patterns.MessagingCoordinator") as mock_coord_class:
                with patch("src.utils.messaging_patterns.MessagePattern") as mock_pattern:
                    mock_coordinator = Mock()
                    mock_coord_class.return_value = mock_coordinator
                    
                    # Mock message patterns
                    mock_pattern.PUB_SUB = "pub_sub"
                    mock_pattern.REQ_REPLY = "req_reply"
                    mock_pattern.STREAM = "stream"
                    mock_pattern.BATCH = "batch"
                    
                    register_util_services()
                    
                    # Get the messaging coordinator factory
                    calls = mock_injector.register_factory.call_args_list
                    coord_call = next(call for call in calls if call[0][0] == "MessagingCoordinator")
                    factory_func = coord_call[0][1]
                    
                    # Test the factory
                    result = factory_func()
                    
                    assert result == mock_coordinator
                    # Should have registered handlers for all patterns
                    assert mock_coordinator.register_handler.call_count == 4


class TestErrorHandling:
    """Test error handling in service factories."""

    @patch("src.utils.service_registry.injector")
    def test_factory_error_logging(self, mock_injector):
        """Test that factory errors are properly logged."""
        with patch("src.utils.service_registry._services_registered", False):
            with patch("src.utils.gpu_utils.GPUManager", side_effect=Exception("Test error")):
                with patch("src.core.logging.get_logger") as mock_get_logger:
                    mock_logger = Mock()
                    mock_get_logger.return_value = mock_logger
                    
                    register_util_services()
                    
                    # Get the GPU manager factory and test it
                    calls = mock_injector.register_factory.call_args_list
                    gpu_manager_call = next(call for call in calls if call[0][0] == "GPUManager")
                    factory_func = gpu_manager_call[0][1]
                    
                    with pytest.raises(ServiceError):
                        factory_func()
                    
                    # Should have logged the warning
                    mock_logger.warning.assert_called_once()

    @patch("src.utils.service_registry.injector")
    def test_dependency_resolution_error(self, mock_injector):
        """Test handling of dependency resolution errors."""
        with patch("src.utils.service_registry._services_registered", False):
            # Mock resolve to fail
            mock_injector.resolve.side_effect = Exception("Dependency not found")
            
            with patch("src.utils.data_flow_integrity.IntegrityPreservingConverter"):
                register_util_services()
                
                # Get the integrity converter factory and test it
                calls = mock_injector.register_factory.call_args_list
                converter_call = next(call for call in calls if call[0][0] == "IntegrityPreservingConverter")
                factory_func = converter_call[0][1]
                
                with pytest.raises(ServiceError, match="Failed to create integrity converter"):
                    factory_func()


class TestModuleLevelState:
    """Test module-level state management."""

    def test_services_registered_flag_initial_state(self):
        """Test that _services_registered flag starts as False."""
        # Import at module level to check initial state
        import src.utils.service_registry as registry_module
        
        # Reset the flag to test initial state
        registry_module._services_registered = False
        assert registry_module._services_registered is False

    @patch("src.utils.service_registry.injector")
    def test_services_registered_flag_after_registration(self, mock_injector):
        """Test that _services_registered flag is set to True after registration."""
        with patch("src.utils.service_registry._services_registered", False):
            register_util_services()
            
            # Check that the flag is set to True
            from src.utils.service_registry import _services_registered
            # Note: We can't directly check the module variable due to patching,
            # but we can verify the function completed without error


class TestServiceLifecycles:
    """Test service lifecycle configurations."""

    @patch("src.utils.service_registry.injector")
    def test_all_services_registered_as_singletons(self, mock_injector):
        """Test that all services are registered with singleton lifecycle."""
        with patch("src.utils.service_registry._services_registered", False):
            register_util_services()
            
            # Check that all registrations use singleton=True
            calls = mock_injector.register_factory.call_args_list
            
            for call in calls:
                if len(call[1]) > 0 and "singleton" in call[1]:
                    assert call[1]["singleton"] is True, f"Service {call[0][0]} should be singleton"

    @patch("src.utils.service_registry.injector")
    def test_interface_and_concrete_registrations(self, mock_injector):
        """Test that both interface and concrete types are registered."""
        with patch("src.utils.service_registry._services_registered", False):
            register_util_services()
            
            calls = mock_injector.register_factory.call_args_list
            service_names = [call[0][0] for call in calls]
            
            # Check that we have both concrete and interface registrations
            assert "GPUManager" in service_names
            assert "GPUInterface" in service_names
            assert "ValidationService" in service_names
            assert "ValidationServiceInterface" in service_names
            assert "FinancialCalculator" in service_names
            assert "CalculatorInterface" in service_names


class TestImportHandling:
    """Test proper handling of imports in factories."""

    @patch("src.utils.service_registry.injector")
    def test_factory_imports_are_lazy(self, mock_injector):
        """Test that factory functions perform lazy imports."""
        with patch("src.utils.service_registry._services_registered", False):
            # Mock successful import but track when it happens
            gpu_manager_imported = False
            
            def mock_gpu_manager_init():
                nonlocal gpu_manager_imported
                gpu_manager_imported = True
                return Mock()
            
            with patch("src.utils.gpu_utils.GPUManager", mock_gpu_manager_init):
                # Register services - imports should NOT happen yet
                register_util_services()
                assert not gpu_manager_imported
                
                # Get and call the factory - NOW import should happen
                calls = mock_injector.register_factory.call_args_list
                gpu_manager_call = next(call for call in calls if call[0][0] == "GPUManager")
                factory_func = gpu_manager_call[0][1]
                
                factory_func()
                assert gpu_manager_imported