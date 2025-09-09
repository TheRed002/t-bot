"""GPU Utilities for T-Bot Trading System.

This module provides utilities for GPU acceleration throughout the codebase.
It automatically detects available GPUs and provides fallback to CPU when needed.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.exceptions import ServiceError
from src.core.logging import get_logger

# Try to import GPU libraries with graceful fallbacks
try:
    import torch

    TORCH_AVAILABLE = True
    TORCH_DEVICE = None  # Will be set later when needed
    torch_module = torch
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None
    torch_module = None  # type: ignore[assignment]

try:
    import tensorflow as tf

    TF_AVAILABLE = True
    tf_module = tf
except ImportError:
    TF_AVAILABLE = False
    tf_module = None

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    cp_module = cp
except ImportError:
    CUPY_AVAILABLE = False
    cp_module = None

try:
    import cudf
    import cuml

    RAPIDS_AVAILABLE = True
    cudf_module = cudf
    cuml_module = cuml
except ImportError:
    RAPIDS_AVAILABLE = False
    cudf_module = None
    cuml_module = None

# Configure logging
logger = get_logger(__name__)


class GPUManager:
    """Manages GPU resources and provides utilities for GPU acceleration."""

    def __init__(self) -> None:
        """Initialize GPU manager."""
        self._initialize_gpu_settings()
        self.gpu_available = self._check_gpu_availability()
        self.device_info = self._get_device_info()
        self._log_gpu_status()

    def _initialize_gpu_settings(self) -> None:
        """Initialize GPU settings for available libraries."""
        global TORCH_DEVICE

        # Initialize PyTorch settings
        if TORCH_AVAILABLE and torch_module:
            try:
                if torch_module.cuda.is_available():
                    TORCH_DEVICE = torch_module.device("cuda")
                    # Use modern PyTorch API instead of deprecated set_default_tensor_type
                    try:
                        torch_module.set_default_dtype(torch_module.float32)
                        torch_module.set_default_device("cuda")
                    except Exception as e:
                        logger.warning(f"Could not set default CUDA settings: {e}")
                        TORCH_DEVICE = torch_module.device("cpu")
                else:
                    TORCH_DEVICE = torch_module.device("cpu")
            except Exception as e:
                logger.warning(f"PyTorch initialization error: {e}")
                TORCH_DEVICE = torch_module.device("cpu") if torch_module else None

        # Initialize TensorFlow settings
        if TF_AVAILABLE and tf_module:
            try:
                gpus = tf_module.config.experimental.list_physical_devices("GPU")
                if gpus:
                    try:
                        for gpu in gpus:
                            tf_module.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        logger.warning(f"TensorFlow GPU configuration error: {e}")
            except Exception as e:
                logger.warning(f"TensorFlow initialization error: {e}")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        if TORCH_AVAILABLE and torch_module:
            try:
                if torch_module.cuda.is_available():
                    return True
            except Exception as e:
                logger.debug(f"PyTorch GPU check failed: {e}")
                # Continue checking other GPU libraries

        if TF_AVAILABLE and tf_module:
            try:
                if tf_module.config.list_physical_devices("GPU"):
                    return True
            except Exception as e:
                logger.debug(f"TensorFlow GPU check failed: {e}")
                # Continue checking other GPU libraries

        if CUPY_AVAILABLE and cp_module:
            try:
                cp_module.cuda.Device(0)
                return True
            except Exception as e:
                logger.debug(f"CuPy GPU check failed: {e}")
                # Continue checking other GPU libraries

        return False

    def _get_device_info(self) -> dict[str, Any]:
        """Get information about available devices."""
        info: dict[str, Any] = {
            "gpu_available": self.gpu_available,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "libraries": {
                "torch": TORCH_AVAILABLE,
                "tensorflow": TF_AVAILABLE,
                "cupy": CUPY_AVAILABLE,
                "rapids": RAPIDS_AVAILABLE,
            },
        }

        if TORCH_AVAILABLE and torch_module:
            try:
                if torch_module.cuda.is_available():
                    info["cuda_available"] = True
                    info["device_count"] = torch_module.cuda.device_count()
                    info["cuda_version"] = torch_module.version.cuda
                    for i in range(torch_module.cuda.device_count()):
                        device_props = torch_module.cuda.get_device_properties(i)
                        info["devices"].append(
                            {
                                "index": i,
                                "name": device_props.name,
                                "total_memory": device_props.total_memory,
                                "major": device_props.major,
                                "minor": device_props.minor,
                                "multi_processor_count": device_props.multi_processor_count,
                            }
                        )
            except Exception as e:
                logger.warning(f"Error getting PyTorch device info: {e}")

        return info

    def _log_gpu_status(self) -> None:
        """Log GPU status information."""
        if self.gpu_available:
            logger.info(
                f"GPU acceleration enabled: {self.device_info['device_count']} device(s) available"
            )
            for device in self.device_info["devices"]:
                memory_gb = device["total_memory"] / 1e9
                logger.info(f"  Device {device['index']}: {device['name']} ({memory_gb:.2f} GB)")
        else:
            logger.warning("GPU acceleration not available, using CPU")

        logger.debug(f"Available libraries: {self.device_info['libraries']}")

    def is_available(self) -> bool:
        """Check if GPU is available for factory pattern interface compliance."""
        return self.gpu_available

    def get_memory_info(self, device_id: int = 0) -> dict[str, float]:
        """Get GPU memory information."""
        if not self.gpu_available:
            return {"total": 0, "used": 0, "free": 0}

        if TORCH_AVAILABLE and torch_module:
            try:
                if torch_module.cuda.is_available():
                    torch_module.cuda.set_device(device_id)
                    total = torch_module.cuda.get_device_properties(device_id).total_memory
                    reserved = torch_module.cuda.memory_reserved(device_id)
                    allocated = torch_module.cuda.memory_allocated(device_id)
                    free = total - reserved

                    return {
                        "total": total / 1e9,  # Convert to GB
                        "reserved": reserved / 1e9,
                        "allocated": allocated / 1e9,
                        "free": free / 1e9,
                    }
            except Exception as e:
                logger.warning(f"Error getting GPU memory info: {e}")

        return {"total": 0, "used": 0, "free": 0}

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if TORCH_AVAILABLE and torch_module:
            try:
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                    logger.debug("Cleared PyTorch GPU cache")
            except Exception as e:
                logger.warning(f"Error clearing PyTorch cache: {e}")

        if TF_AVAILABLE and tf_module:
            try:
                tf_module.keras.backend.clear_session()
                logger.debug("Cleared TensorFlow session")
            except Exception as e:
                logger.warning(f"Error clearing TensorFlow session: {e}")

    def to_gpu(self, data: Any, dtype: str | None = None) -> Any:
        """Move data to GPU if available."""
        if not self.gpu_available:
            return data

        try:
            return self._convert_to_gpu(data, dtype)
        except Exception as e:
            logger.warning(f"Error moving data to GPU, using CPU: {e}")
            return data

    def _convert_to_gpu(self, data: Any, dtype: str | None = None) -> Any:
        """Helper method to convert data to GPU format."""
        # Handle NumPy arrays
        if isinstance(data, np.ndarray):
            return self._convert_numpy_to_gpu(data, dtype)

        # Handle Pandas DataFrames
        if isinstance(data, pd.DataFrame):
            return self._convert_dataframe_to_gpu(data)

        # Handle PyTorch tensors
        if self._is_tensor_like(data):
            return self._convert_tensor_to_gpu(data)

        return data

    def _convert_numpy_to_gpu(self, data: NDArray[np.float64], dtype: str | None = None) -> Any:
        """Convert NumPy array to GPU format."""
        if CUPY_AVAILABLE and cp_module:
            return cp_module.asarray(data, dtype=dtype)
        elif TORCH_AVAILABLE and torch_module and TORCH_DEVICE:
            tensor = torch_module.from_numpy(data)
            if dtype:
                tensor = tensor.to(getattr(torch_module, dtype))
            return tensor.to(TORCH_DEVICE)
        return data

    def _convert_dataframe_to_gpu(self, data: pd.DataFrame) -> Any:
        """Convert Pandas DataFrame to GPU format."""
        if RAPIDS_AVAILABLE and cudf_module:
            return cudf_module.from_pandas(data)
        elif TORCH_AVAILABLE and torch_module and TORCH_DEVICE:
            return torch_module.from_numpy(data.values).to(TORCH_DEVICE)
        return data

    def _is_tensor_like(self, data: Any) -> bool:
        """Check if data is a tensor-like object."""
        return bool(
            TORCH_AVAILABLE
            and torch_module
            and TORCH_DEVICE
            and hasattr(data, "to")
            and hasattr(data, "device")
        )

    def _convert_tensor_to_gpu(self, data: Any) -> Any:
        """Convert tensor to GPU."""
        if TORCH_AVAILABLE and torch_module and TORCH_DEVICE:
            return data.to(TORCH_DEVICE)
        return data

    def to_cpu(self, data: Any) -> Any:
        """Move data from GPU to CPU."""
        try:
            # Handle CuPy arrays
            if CUPY_AVAILABLE and cp_module and hasattr(cp_module, "ndarray"):
                if isinstance(data, cp_module.ndarray):
                    return cp_module.asnumpy(data)

            # Handle PyTorch tensors
            if TORCH_AVAILABLE and torch_module:
                if hasattr(data, "cpu") and hasattr(
                    data, "numpy"
                ):  # Check if it's a tensor-like object
                    return data.cpu().numpy()

            # Handle cuDF DataFrames
            if RAPIDS_AVAILABLE and cudf_module and hasattr(data, "to_pandas"):
                return data.to_pandas()

        except Exception as e:
            logger.warning(f"Error moving data to CPU: {e}")

        return data

    def accelerate_computation(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Decorator to automatically accelerate computations on GPU."""
        if not self.gpu_available:
            return func(*args, **kwargs)

        gpu_args = []
        gpu_kwargs = {}

        try:
            # Move inputs to GPU
            gpu_args = [self.to_gpu(arg) for arg in args]
            gpu_kwargs = {k: self.to_gpu(v) for k, v in kwargs.items()}

            # Run computation
            result = func(*gpu_args, **gpu_kwargs)
            # Move result back to CPU if needed
            return self.to_cpu(result)
        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            return func(*args, **kwargs)
        finally:
            # Clear GPU memory to prevent memory leaks
            if TORCH_AVAILABLE and torch_module and torch_module.cuda.is_available():
                try:
                    torch_module.cuda.empty_cache()
                except Exception as cleanup_e:
                    logger.debug(f"Failed to clear GPU cache: {cleanup_e}")
            # Explicitly delete GPU tensors/arrays
            try:
                del gpu_args
                del gpu_kwargs
            except Exception as e:
                # GPU memory cleanup can fail safely
                logger.debug(f"GPU memory cleanup failed: {e}")


# GPUManager registration is handled by service_registry.py


def get_optimal_batch_size(
    data_size: int, memory_limit_gb: float = 4.0, gpu_manager: GPUManager | None = None
) -> int:
    """Calculate optimal batch size based on available GPU memory."""
    if gpu_manager is None:
        # Use proper dependency injection instead of direct instantiation
        gpu_manager = _get_gpu_manager_service()

    if not gpu_manager.is_available():
        # CPU batch size
        return min(1024, data_size)

    memory_info = gpu_manager.get_memory_info()
    available_memory_gb = memory_info.get("free", 4.0)

    # Conservative estimate: use 80% of available memory
    usable_memory_gb = available_memory_gb * 0.8

    # Estimate batch size based on memory
    # Assuming each sample needs ~1MB (conservative)
    samples_per_gb = 1000
    max_batch_size = int(usable_memory_gb * samples_per_gb)

    # Apply reasonable limits
    batch_size = min(max_batch_size, data_size, 8192)
    batch_size = max(batch_size, 32)  # Minimum batch size

    return batch_size


def _get_gpu_manager_service(gpu_manager: GPUManager | None = None) -> GPUManager:
    """Get GPU manager using dependency injection pattern.

    Args:
        gpu_manager: Injected GPU manager (preferred from service layer)

    Returns:
        GPUManager instance

    Raises:
        ServiceError: If GPU manager cannot be obtained
    """
    if gpu_manager is not None:
        return gpu_manager

    # Fallback but warn about architectural violation
    logger.warning(
        "GPUManager not injected - creating default instance. "
        "Inject via dependency injection for better testability."
    )

    try:
        return GPUManager()
    except Exception as e:
        raise ServiceError(
            f"Failed to create GPUManager: {e}",
            error_code="SERV_000",
        ) from e


def parallel_apply(
    df: pd.DataFrame,
    func: Any,
    axis: int = 0,
    use_gpu: bool = True,
    gpu_manager: "GPUManager | None" = None,
) -> pd.DataFrame:
    """Apply function to DataFrame with GPU acceleration if available."""
    if gpu_manager is None:
        # Use proper dependency injection instead of direct instantiation
        gpu_manager = _get_gpu_manager_service()

    if use_gpu and gpu_manager.is_available() and RAPIDS_AVAILABLE and cudf_module:
        try:
            # Convert to cuDF and apply
            gdf = cudf_module.from_pandas(df)
            result = gdf.apply(func, axis=axis)
            return result.to_pandas()
        except Exception as e:
            logger.warning(f"GPU DataFrame operation failed, using CPU: {e}")

    # Fallback to pandas
    return df.apply(func, axis=axis)


def gpu_accelerated_correlation(
    data: NDArray[np.float64], gpu_manager: "GPUManager | None" = None
) -> NDArray[np.float64]:
    """Calculate correlation matrix with GPU acceleration."""
    if gpu_manager is None:
        # Use proper dependency injection instead of direct instantiation
        gpu_manager = _get_gpu_manager_service()

    gpu_data = None
    try:
        if gpu_manager.is_available():
            if CUPY_AVAILABLE and cp_module:
                gpu_data = cp_module.asarray(data)
                corr = cp_module.corrcoef(gpu_data.T)
                return np.asarray(cp_module.asnumpy(corr))
            elif TORCH_AVAILABLE and torch_module and TORCH_DEVICE:
                gpu_data = torch_module.from_numpy(data).to(TORCH_DEVICE)
                # Standardize the data
                mean = gpu_data.mean(dim=0)
                std = gpu_data.std(dim=0)
                standardized = (gpu_data - mean) / (std + 1e-8)
                # Calculate correlation
                corr = torch_module.mm(standardized.T, standardized) / (gpu_data.shape[0] - 1)
                return corr.cpu().numpy()
    except Exception as e:
        logger.warning(f"GPU correlation calculation failed, using CPU: {e}")
    finally:
        # Clean up GPU memory
        if gpu_data is not None:
            try:
                del gpu_data
            except Exception as e:
                # GPU memory cleanup can fail safely
                logger.debug(f"GPU memory cleanup failed: {e}")
        if TORCH_AVAILABLE and torch_module and torch_module.cuda.is_available():
            try:
                torch_module.cuda.empty_cache()
            except Exception as e:
                # GPU memory cleanup can fail safely
                logger.debug(f"GPU memory cleanup failed: {e}")

    # Fallback to NumPy
    return np.corrcoef(data.T)


def gpu_accelerated_rolling_window(
    data: NDArray[np.float64], window_size: int, func: Any, gpu_manager: GPUManager | None = None
) -> NDArray[np.float64]:
    """Apply rolling window function with GPU acceleration."""
    if gpu_manager is None:
        # Use proper dependency injection instead of direct instantiation
        gpu_manager = _get_gpu_manager_service()

    gpu_data = None
    result = None

    try:
        if gpu_manager.is_available() and CUPY_AVAILABLE and cp_module:
            gpu_data = cp_module.asarray(data)
            result = cp_module.zeros(len(data) - window_size + 1)

            for i in range(len(result)):
                window = gpu_data[i : i + window_size]
                result[i] = func(window)

            return np.asarray(cp_module.asnumpy(result))
    except Exception as e:
        logger.warning(f"GPU rolling window calculation failed, using CPU: {e}")
    finally:
        # Clean up GPU memory
        if gpu_data is not None:
            try:
                del gpu_data
            except Exception as e:
                # GPU memory cleanup can fail safely
                logger.debug(f"GPU memory cleanup failed: {e}")
        if result is not None:
            try:
                del result
            except Exception as e:
                # GPU memory cleanup can fail safely
                logger.debug(f"GPU memory cleanup failed: {e}")
        if CUPY_AVAILABLE and cp_module:
            try:
                cp_module.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                # GPU memory cleanup can fail safely
                logger.debug(f"GPU memory cleanup failed: {e}")

    # Fallback to NumPy
    result_np = np.zeros(len(data) - window_size + 1)
    for i in range(len(result_np)):
        window = data[i : i + window_size]
        result_np[i] = func(window)

    return result_np


def setup_gpu_logging(gpu_manager: GPUManager | None = None) -> Callable[[], None] | None:
    """Setup GPU usage monitoring and logging."""
    if gpu_manager is None:
        # Use proper dependency injection instead of direct instantiation
        gpu_manager = _get_gpu_manager_service()

    if not gpu_manager.is_available():
        return None

    try:
        import gpustat

        # Log GPU stats periodically
        def log_gpu_stats() -> None:
            try:
                stats = gpustat.GPUStatCollection.new_query()
                for gpu in stats:
                    logger.debug(
                        f"GPU {gpu.index}: {gpu.name}, "
                        f"Memory: {gpu.memory_used}/{gpu.memory_total} MB, "
                        f"Utilization: {gpu.utilization}%"
                    )
            except Exception as e:
                logger.error(f"Failed to log GPU stats: {e}")

        return log_gpu_stats
    except ImportError:
        logger.warning("GPU monitoring libraries not available")
        return None


# Export commonly used functions
__all__ = [
    "GPUManager",
    "get_optimal_batch_size",
    "get_optimal_batch_size",
    "gpu_accelerated_correlation",
    "gpu_accelerated_rolling_window",
    "parallel_apply",
    "setup_gpu_logging",
]
