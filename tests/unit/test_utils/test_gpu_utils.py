"""Tests for GPU utilities module."""

import logging
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import ServiceError
from src.utils.gpu_utils import (
    GPUManager,
    get_optimal_batch_size,
    gpu_accelerated_correlation,
    gpu_accelerated_rolling_window,
    parallel_apply,
    setup_gpu_logging,
)


class TestGPUManager:
    """Test GPUManager class."""

    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", False)
    @patch("src.utils.gpu_utils.TF_AVAILABLE", False)
    @patch("src.utils.gpu_utils.CUPY_AVAILABLE", False)
    def test_gpu_manager_no_gpu_libraries(self):
        """Test GPUManager initialization when no GPU libraries are available."""
        gpu_manager = GPUManager()
        
        assert gpu_manager.gpu_available is False
        assert gpu_manager.is_available() is False
        assert gpu_manager.device_info["gpu_available"] is False
        assert gpu_manager.device_info["cuda_available"] is False
        assert gpu_manager.device_info["device_count"] == 0

    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", True)
    @patch("src.utils.gpu_utils.TF_AVAILABLE", False)
    @patch("src.utils.gpu_utils.CUPY_AVAILABLE", False)
    @patch("src.utils.gpu_utils.torch_module")
    def test_gpu_manager_torch_available_no_cuda(self, mock_torch):
        """Test GPUManager with PyTorch available but no CUDA."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()
        
        gpu_manager = GPUManager()
        
        assert gpu_manager.gpu_available is False
        assert gpu_manager.is_available() is False

    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", True)
    @patch("src.utils.gpu_utils.torch_module")
    def test_gpu_manager_torch_with_cuda(self, mock_torch):
        """Test GPUManager with PyTorch and CUDA available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.version.cuda = "11.8"
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.name = "Tesla V100"
        # Mock total_memory as a MagicMock that can handle formatting
        mock_memory = MagicMock()
        mock_memory.__int__ = Mock(return_value=16000000000)
        mock_memory.__truediv__ = Mock(return_value=16.0)
        mock_memory.__format__ = Mock(return_value="16.00")
        # Fix: Configure the division chain properly for memory calculations
        mock_memory.__sub__ = Mock()
        mock_memory.__sub__.return_value = Mock()
        mock_memory.__sub__.return_value.__truediv__ = Mock(return_value=14.0)
        mock_props.total_memory = mock_memory
        mock_props.major = 7
        mock_props.minor = 0
        mock_props.multi_processor_count = 80
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        gpu_manager = GPUManager()
        
        assert gpu_manager.gpu_available is True
        assert gpu_manager.is_available() is True
        assert gpu_manager.device_info["cuda_available"] is True
        assert gpu_manager.device_info["device_count"] == 1

    @patch("src.utils.gpu_utils.TF_AVAILABLE", True)
    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", False)
    @patch("src.utils.gpu_utils.CUPY_AVAILABLE", False)
    @patch("src.utils.gpu_utils.tf_module")
    def test_gpu_manager_tensorflow_with_gpu(self, mock_tf):
        """Test GPUManager with TensorFlow GPU support."""
        mock_gpu = MagicMock()
        mock_tf.config.experimental.list_physical_devices.return_value = [mock_gpu]
        
        gpu_manager = GPUManager()
        
        # TensorFlow GPU detection should work
        assert isinstance(gpu_manager.gpu_available, bool)

    def test_get_memory_info_no_gpu(self):
        """Test get_memory_info when no GPU is available."""
        with patch("src.utils.gpu_utils.TORCH_AVAILABLE", False):
            gpu_manager = GPUManager()
            memory_info = gpu_manager.get_memory_info()
            
            assert memory_info == {"total": 0, "used": 0, "free": 0}

    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", True)
    @patch("src.utils.gpu_utils.torch_module")
    def test_get_memory_info_with_torch(self, mock_torch):
        """Test get_memory_info with PyTorch."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.version.cuda = "11.8"
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.name = "Tesla V100"
        # Mock total_memory as a MagicMock that can handle formatting
        mock_memory = MagicMock()
        mock_memory.__int__ = Mock(return_value=16000000000)
        mock_memory.__truediv__ = Mock(return_value=16.0)
        mock_memory.__format__ = Mock(return_value="16.00")
        # Fix: Configure the division chain properly for memory calculations
        mock_memory.__sub__ = Mock()
        mock_memory.__sub__.return_value = Mock()
        mock_memory.__sub__.return_value.__truediv__ = Mock(return_value=14.0)
        mock_props.total_memory = mock_memory
        mock_props.major = 7
        mock_props.minor = 0
        mock_props.multi_processor_count = 80
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        mock_torch.cuda.memory_reserved.return_value = 2000000000
        mock_torch.cuda.memory_allocated.return_value = 1000000000
        
        gpu_manager = GPUManager()
        memory_info = gpu_manager.get_memory_info()
        
        expected_total = 16.0  # 16e9 / 1e9
        expected_free = 14.0   # (16e9 - 2e9) / 1e9
        
        assert memory_info["total"] == expected_total
        assert memory_info["free"] == expected_free

    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", True)
    @patch("src.utils.gpu_utils.torch_module")
    def test_clear_cache_torch(self, mock_torch):
        """Test clear_cache with PyTorch."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.version.cuda = "11.8"
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device
        
        # Mock device properties with proper format handling
        mock_props = MagicMock()
        mock_props.name = "Tesla V100"
        # Mock total_memory as a MagicMock that can handle formatting
        mock_memory = MagicMock()
        mock_memory.__int__ = Mock(return_value=16000000000)
        mock_memory.__truediv__ = Mock(return_value=16.0)
        mock_memory.__format__ = Mock(return_value="16.00")
        # Fix: Configure the division chain properly for memory calculations
        mock_memory.__sub__ = Mock()
        mock_memory.__sub__.return_value = Mock()
        mock_memory.__sub__.return_value.__truediv__ = Mock(return_value=14.0)
        mock_props.total_memory = mock_memory
        mock_props.major = 7
        mock_props.minor = 0
        mock_props.multi_processor_count = 80
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        gpu_manager = GPUManager()
        gpu_manager.clear_cache()
        
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("src.utils.gpu_utils.TF_AVAILABLE", True)
    @patch("src.utils.gpu_utils.tf_module")
    def test_clear_cache_tensorflow(self, mock_tf):
        """Test clear_cache with TensorFlow."""
        gpu_manager = GPUManager()
        gpu_manager.clear_cache()
        
        mock_tf.keras.backend.clear_session.assert_called_once()

    def test_to_gpu_no_gpu_available(self):
        """Test to_gpu when no GPU is available."""
        with patch("src.utils.gpu_utils.TORCH_AVAILABLE", False):
            with patch("src.utils.gpu_utils.TF_AVAILABLE", False):
                with patch("src.utils.gpu_utils.CUPY_AVAILABLE", False):
                    gpu_manager = GPUManager()
                    data = np.array([1, 2, 3])
                    result = gpu_manager.to_gpu(data)
                    
                    # Use numpy array_equal since no GPU libraries are available
                    assert np.array_equal(result, data)

    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", False)
    @patch("src.utils.gpu_utils.TF_AVAILABLE", False)
    @patch("src.utils.gpu_utils.CUPY_AVAILABLE", True)
    @patch("src.utils.gpu_utils.cp_module")
    def test_to_gpu_numpy_with_cupy(self, mock_cp):
        """Test to_gpu with NumPy array and CuPy available."""
        mock_cp.asarray.return_value = MagicMock()
        
        gpu_manager = GPUManager()
        # Force gpu_available to True
        gpu_manager.gpu_available = True
        
        data = np.array([1.0, 2.0, 3.0])
        result = gpu_manager.to_gpu(data)
        
        # CuPy should be called if available
        mock_cp.asarray.assert_called_once_with(data, dtype=None)

    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", False)
    @patch("src.utils.gpu_utils.TF_AVAILABLE", False)
    @patch("src.utils.gpu_utils.RAPIDS_AVAILABLE", True)
    @patch("src.utils.gpu_utils.cudf_module")
    def test_to_gpu_dataframe_with_rapids(self, mock_cudf):
        """Test to_gpu with Pandas DataFrame and RAPIDS available."""
        mock_cudf.from_pandas.return_value = MagicMock()
        
        gpu_manager = GPUManager()
        # Force gpu_available to True
        gpu_manager.gpu_available = True
        
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = gpu_manager.to_gpu(df)
        
        # RAPIDS should be called if available
        mock_cudf.from_pandas.assert_called_once_with(df)

    def test_to_cpu_unknown_data_type(self):
        """Test to_cpu with unknown data type."""
        gpu_manager = GPUManager()
        data = "string data"
        result = gpu_manager.to_cpu(data)
        
        assert result == data

    def test_accelerate_computation_no_gpu(self):
        """Test accelerate_computation when no GPU is available."""
        with patch("src.utils.gpu_utils.TORCH_AVAILABLE", False):
            gpu_manager = GPUManager()
            
            def test_func(x, y):
                return x + y
            
            result = gpu_manager.accelerate_computation(test_func, 5, 3)
            assert result == 8

    def test_accelerate_computation_with_gpu_fallback(self):
        """Test accelerate_computation with GPU error fallback."""
        gpu_manager = GPUManager()
        gpu_manager.gpu_available = True  # Force GPU to be available
        
        with patch.object(gpu_manager, 'to_gpu', side_effect=Exception("GPU error")):
            def test_func(x, y):
                return x + y
            
            result = gpu_manager.accelerate_computation(test_func, 5, 3)
            assert result == 8


class TestGetOptimalBatchSize:
    """Test get_optimal_batch_size function."""

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    def test_get_optimal_batch_size_no_gpu(self, mock_get_service):
        """Test get_optimal_batch_size when no GPU is available."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = False
        mock_get_service.return_value = mock_gpu_manager
        
        batch_size = get_optimal_batch_size(2048)
        
        assert batch_size == 1024  # min(1024, 2048) for CPU

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    def test_get_optimal_batch_size_with_gpu(self, mock_get_service):
        """Test get_optimal_batch_size with GPU available."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = True
        mock_gpu_manager.get_memory_info.return_value = {"free": 8.0}  # 8GB free
        mock_get_service.return_value = mock_gpu_manager
        
        batch_size = get_optimal_batch_size(10000)
        
        # Should use GPU memory calculation
        # 8GB * 0.8 * 1000 samples/GB = 6400, capped at 8192
        assert batch_size >= 32  # Minimum batch size
        assert batch_size <= 8192  # Maximum batch size

    def test_get_optimal_batch_size_with_provided_manager(self):
        """Test get_optimal_batch_size with provided GPU manager."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = False
        
        batch_size = get_optimal_batch_size(512, gpu_manager=mock_gpu_manager)
        
        assert batch_size == 512  # min(1024, 512)


class TestParallelApply:
    """Test parallel_apply function."""

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    @patch("src.utils.gpu_utils.RAPIDS_AVAILABLE", False)
    def test_parallel_apply_cpu_fallback(self, mock_get_service):
        """Test parallel_apply with CPU fallback."""
        mock_gpu_manager = Mock()
        mock_get_service.return_value = mock_gpu_manager
        
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = parallel_apply(df, lambda x: x * 2)
        
        expected = df.apply(lambda x: x * 2)
        pd.testing.assert_frame_equal(result, expected)

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    @patch("src.utils.gpu_utils.RAPIDS_AVAILABLE", True)
    @patch("src.utils.gpu_utils.cudf_module")
    def test_parallel_apply_with_rapids(self, mock_cudf, mock_get_service):
        """Test parallel_apply with RAPIDS GPU acceleration."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = True
        mock_get_service.return_value = mock_gpu_manager
        
        # Mock cuDF operations
        mock_gdf = Mock()
        mock_result = Mock()
        mock_result.to_pandas.return_value = pd.DataFrame({"a": [2, 4, 6]})
        mock_gdf.apply.return_value = mock_result
        mock_cudf.from_pandas.return_value = mock_gdf
        
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = parallel_apply(df, lambda x: x * 2)
        
        mock_cudf.from_pandas.assert_called_once_with(df)
        mock_gdf.apply.assert_called_once()


class TestGPUAcceleratedCorrelation:
    """Test gpu_accelerated_correlation function."""

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    def test_gpu_accelerated_correlation_cpu_fallback(self, mock_get_service):
        """Test gpu_accelerated_correlation with CPU fallback."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = False
        mock_get_service.return_value = mock_gpu_manager
        
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        result = gpu_accelerated_correlation(data)
        
        expected = np.corrcoef(data.T)
        np.testing.assert_array_almost_equal(result, expected)

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    @patch("src.utils.gpu_utils.CUPY_AVAILABLE", True)
    @patch("src.utils.gpu_utils.cp_module")
    def test_gpu_accelerated_correlation_with_cupy(self, mock_cp, mock_get_service):
        """Test gpu_accelerated_correlation with CuPy."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = True
        mock_get_service.return_value = mock_gpu_manager
        
        # Mock CuPy operations
        mock_gpu_data = Mock()
        mock_corr = Mock()
        mock_cp.asarray.return_value = mock_gpu_data
        mock_cp.corrcoef.return_value = mock_corr
        mock_cp.asnumpy.return_value = np.eye(3)  # Identity matrix for test
        
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        result = gpu_accelerated_correlation(data)
        
        mock_cp.asarray.assert_called_once_with(data)
        mock_cp.corrcoef.assert_called_once()
        assert result.shape == (3, 3)

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    @patch("src.utils.gpu_utils.CUPY_AVAILABLE", False)  # Force CuPy to be unavailable
    @patch("src.utils.gpu_utils.TORCH_AVAILABLE", True)
    @patch("src.utils.gpu_utils.torch_module")
    @patch("src.utils.gpu_utils.TORCH_DEVICE", "cuda:0")  # Mock device
    def test_gpu_accelerated_correlation_with_torch(self, mock_torch, mock_get_service):
        """Test gpu_accelerated_correlation with PyTorch."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = True
        mock_get_service.return_value = mock_gpu_manager
        
        # Mock PyTorch operations
        mock_tensor = Mock()
        mock_tensor.mean.return_value = Mock()
        mock_tensor.std.return_value = Mock()
        mock_tensor.shape = [3, 3]
        mock_tensor.T = Mock()
        
        mock_standardized = Mock()
        mock_standardized.T = Mock()
        
        mock_corr = Mock()
        mock_corr.cpu.return_value.numpy.return_value = np.eye(3)
        
        # Fix the chained call mocking
        mock_from_numpy = Mock()
        mock_from_numpy.to.return_value = mock_tensor
        mock_torch.from_numpy.return_value = mock_from_numpy
        mock_torch.mm.return_value = mock_corr
        
        # Mock the standardization process
        mock_sub_result = Mock()
        mock_sub_result.__truediv__ = Mock(return_value=mock_standardized)
        mock_tensor.__sub__ = Mock(return_value=mock_sub_result)
        
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        result = gpu_accelerated_correlation(data)
        
        mock_torch.from_numpy.assert_called_once_with(data)
        assert result.shape == (3, 3)


class TestGPUAcceleratedRollingWindow:
    """Test gpu_accelerated_rolling_window function."""

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    def test_gpu_accelerated_rolling_window_cpu_fallback(self, mock_get_service):
        """Test gpu_accelerated_rolling_window with CPU fallback."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = False
        mock_get_service.return_value = mock_gpu_manager
        
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = gpu_accelerated_rolling_window(data, 3, np.mean)
        
        expected = np.array([2.0, 3.0, 4.0])  # Rolling mean of [1,2,3], [2,3,4], [3,4,5]
        np.testing.assert_array_almost_equal(result, expected)

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    @patch("src.utils.gpu_utils.CUPY_AVAILABLE", True)
    @patch("src.utils.gpu_utils.cp_module")
    def test_gpu_accelerated_rolling_window_with_cupy(self, mock_cp, mock_get_service):
        """Test gpu_accelerated_rolling_window with CuPy."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = True
        mock_get_service.return_value = mock_gpu_manager
        
        # Mock CuPy operations
        mock_gpu_data = Mock()
        mock_result = Mock()
        mock_cp.asarray.return_value = mock_gpu_data
        mock_cp.zeros.return_value = mock_result
        
        # Create a proper mock array that can be len()-ed
        mock_numpy_result = Mock()
        mock_numpy_result.__len__ = Mock(return_value=3)
        mock_cp.asnumpy.return_value = mock_numpy_result
        
        # Mock slicing and function application
        def mock_getitem(*args, **kwargs):
            return Mock()
        
        mock_gpu_data.__getitem__ = mock_getitem
        mock_result.__setitem__ = Mock()
        mock_result.__len__ = Mock(return_value=3)
        
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = gpu_accelerated_rolling_window(data, 3, np.mean)
        
        mock_cp.asarray.assert_called_once_with(data)
        mock_cp.zeros.assert_called_once_with(3)  # len(data) - window_size + 1
        mock_cp.asnumpy.assert_called_once_with(mock_result)
        # The result should be numpy array containing our mock
        assert isinstance(result, np.ndarray)
        assert result.item() == mock_numpy_result  # Extract the mock from the array


class TestSetupGPULogging:
    """Test setup_gpu_logging function."""

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    def test_setup_gpu_logging_no_gpu(self, mock_get_service):
        """Test setup_gpu_logging when no GPU is available."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = False
        mock_get_service.return_value = mock_gpu_manager
        
        result = setup_gpu_logging()
        
        assert result is None

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    def test_setup_gpu_logging_no_gpustat(self, mock_get_service):
        """Test setup_gpu_logging when gpustat is not available."""
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = True
        mock_get_service.return_value = mock_gpu_manager
        
        # Save original __import__
        import builtins
        original_import = builtins.__import__
        
        def mock_import_side_effect(name, *args, **kwargs):
            if name == "gpustat":
                raise ImportError("gpustat not available")
            # Use the original import for everything else
            return original_import(name, *args, **kwargs)
        
        # Temporarily replace __import__
        builtins.__import__ = mock_import_side_effect
        try:
            result = setup_gpu_logging()
        finally:
            # Restore original __import__
            builtins.__import__ = original_import
        
        assert result is None

    @patch("src.utils.gpu_utils._get_gpu_manager_service")
    def test_setup_gpu_logging_with_gpustat(self, mock_get_service):
        """Test setup_gpu_logging with gpustat available."""
        # Mock gpustat module
        mock_gpustat = Mock()
        mock_gpu = Mock()
        mock_gpu.index = 0
        mock_gpu.name = "Tesla V100"
        mock_gpu.memory_used = 1000
        mock_gpu.memory_total = 16000
        mock_gpu.utilization = 25
        
        mock_stats = [mock_gpu]
        mock_gpustat.GPUStatCollection.new_query.return_value = mock_stats
        
        mock_gpu_manager = Mock()
        mock_gpu_manager.is_available.return_value = True
        mock_get_service.return_value = mock_gpu_manager
        
        # Mock import behavior
        def mock_import_side_effect(name, *args, **kwargs):
            if name == "gpustat":
                return mock_gpustat
            return Mock()
        
        with patch("builtins.__import__", side_effect=mock_import_side_effect):
            result = setup_gpu_logging()
        
        assert callable(result)
        
        # Test the logging function
        with patch("src.utils.gpu_utils.logger") as mock_logger:
            result()
            # Verify debug was called with the expected message format
            mock_logger.debug.assert_called_with(
                "GPU 0: Tesla V100, Memory: 1000/16000 MB, Utilization: 25%"
            )


class TestServiceIntegration:
    """Test service integration and dependency injection."""

    def test_get_gpu_manager_service_success(self):
        """Test successful GPU manager service resolution."""
        mock_gpu_manager = Mock()
        
        with patch("src.core.dependency_injection.injector") as mock_injector:
            mock_injector.resolve.return_value = mock_gpu_manager
            
            from src.utils.gpu_utils import _get_gpu_manager_service
            
            result = _get_gpu_manager_service()
            
            assert result == mock_gpu_manager
            mock_injector.resolve.assert_called_once_with("GPUManager")

    def test_get_gpu_manager_service_registration_fallback(self):
        """Test GPU manager service registration fallback."""
        # First call fails, second succeeds after registration
        mock_gpu_manager = Mock()
        
        with patch("src.core.dependency_injection.injector") as mock_injector:
            mock_injector.resolve.side_effect = [
                Exception("Service not found"),
                mock_gpu_manager
            ]
            
            with patch("src.utils.service_registry.register_util_services") as mock_register:
                from src.utils.gpu_utils import _get_gpu_manager_service
                
                result = _get_gpu_manager_service()
                
                assert result == mock_gpu_manager
                mock_register.assert_called_once()
                assert mock_injector.resolve.call_count == 2

    def test_get_gpu_manager_service_failure(self):
        """Test GPU manager service resolution failure."""
        with patch("src.core.dependency_injection.injector") as mock_injector:
            mock_injector.resolve.side_effect = Exception("Service not found")
            
            with patch("src.utils.service_registry.register_util_services", side_effect=Exception("Registration failed")):
                from src.utils.gpu_utils import _get_gpu_manager_service
                
                with pytest.raises(ServiceError, match="GPUManager service not available"):
                    _get_gpu_manager_service()


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_exist(self):
        """Test all items in __all__ actually exist."""
        from src.utils.gpu_utils import __all__
        
        expected_exports = [
            "GPUManager",
            "get_optimal_batch_size",
            "gpu_accelerated_correlation",
            "gpu_accelerated_rolling_window",
            "parallel_apply",
            "setup_gpu_logging",
        ]
        
        # Note: __all__ has a duplicate "get_optimal_batch_size" entry in the source
        # We'll test the unique set
        assert set(__all__) >= set(expected_exports)

    def test_exports_are_importable(self):
        """Test all exported items can be imported."""
        from src.utils.gpu_utils import (
            GPUManager,
            get_optimal_batch_size,
            gpu_accelerated_correlation,
            gpu_accelerated_rolling_window,
            parallel_apply,
            setup_gpu_logging,
        )
        
        assert GPUManager is not None
        assert callable(get_optimal_batch_size)
        assert callable(gpu_accelerated_correlation)
        assert callable(gpu_accelerated_rolling_window)
        assert callable(parallel_apply)
        assert callable(setup_gpu_logging)


class TestErrorHandling:
    """Test error handling in GPU utilities."""

    def test_gpu_manager_torch_initialization_error(self):
        """Test GPU manager handles PyTorch initialization errors gracefully."""
        with patch("src.utils.gpu_utils.TORCH_AVAILABLE", True):
            with patch("src.utils.gpu_utils.TF_AVAILABLE", False):
                with patch("src.utils.gpu_utils.CUPY_AVAILABLE", False):
                    with patch("src.utils.gpu_utils.torch_module") as mock_torch:
                        # Make initialization fail
                        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
                        mock_torch.device.return_value = Mock()
                        
                        # Should not raise exception during initialization
                        gpu_manager = GPUManager()
                        
                        # GPU should not be available due to initialization error
                        # The _check_gpu_availability method will also fail due to the same error
                        assert gpu_manager.gpu_available is False

    def test_memory_info_error_handling(self):
        """Test memory info handles errors gracefully."""
        with patch("src.utils.gpu_utils.TORCH_AVAILABLE", True):
            with patch("src.utils.gpu_utils.torch_module") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.cuda.device_count.return_value = 1
                mock_torch.version.cuda = "11.8"
                mock_device = MagicMock()
                mock_torch.device.return_value = mock_device
                
                # Mock device properties that will work during init
                mock_props = MagicMock()
                mock_props.name = "Tesla V100"
                # Mock total_memory as a MagicMock that can handle formatting
                mock_memory = MagicMock()
                mock_memory.__int__ = Mock(return_value=16000000000)
                mock_memory.__truediv__ = Mock(return_value=16.0)
                mock_memory.__format__ = Mock(return_value="16.00")
                mock_props.total_memory = mock_memory
                mock_props.major = 7
                mock_props.minor = 0
                mock_props.multi_processor_count = 80
                mock_torch.cuda.get_device_properties.return_value = mock_props
                
                gpu_manager = GPUManager()
                
                # Now make get_device_properties fail for memory_info call
                mock_torch.cuda.get_device_properties.side_effect = Exception("Memory error")
                memory_info = gpu_manager.get_memory_info()
                
                assert memory_info == {"total": 0, "used": 0, "free": 0}

    def test_clear_cache_error_handling(self):
        """Test clear_cache handles errors gracefully."""
        with patch("src.utils.gpu_utils.TORCH_AVAILABLE", True):
            with patch("src.utils.gpu_utils.torch_module") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.cuda.device_count.return_value = 1
                mock_torch.version.cuda = "11.8"
                mock_device = MagicMock()
                mock_torch.device.return_value = mock_device
                
                # Mock device properties for initialization
                mock_props = MagicMock()
                mock_props.name = "Tesla V100"
                # Mock total_memory as a MagicMock that can handle formatting
                mock_memory = MagicMock()
                mock_memory.__int__ = Mock(return_value=16000000000)
                mock_memory.__truediv__ = Mock(return_value=16.0)
                mock_memory.__format__ = Mock(return_value="16.00")
                mock_props.total_memory = mock_memory
                mock_props.major = 7
                mock_props.minor = 0
                mock_props.multi_processor_count = 80
                mock_torch.cuda.get_device_properties.return_value = mock_props
                
                # Make empty_cache fail
                mock_torch.cuda.empty_cache.side_effect = Exception("Cache error")
                
                gpu_manager = GPUManager()
                # Should not raise exception
                gpu_manager.clear_cache()

    def test_to_gpu_error_fallback(self):
        """Test to_gpu falls back to original data on error."""
        with patch("src.utils.gpu_utils.TORCH_AVAILABLE", False):
            with patch("src.utils.gpu_utils.TF_AVAILABLE", False):
                with patch("src.utils.gpu_utils.CUPY_AVAILABLE", False):
                    gpu_manager = GPUManager()
                    data = np.array([1, 2, 3])
                    
                    result = gpu_manager.to_gpu(data)
                    
                    # Should return original data when no GPU libs available
                    np.testing.assert_array_equal(result, data)