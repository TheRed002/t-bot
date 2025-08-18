"""GPU Utilities for T-Bot Trading System.

This module provides utilities for GPU acceleration throughout the codebase.
It automatically detects available GPUs and provides fallback to CPU when needed.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

# Try to import GPU libraries
try:
    import torch

    TORCH_AVAILABLE = True
    # Set default device
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        TORCH_DEVICE = torch.device("cuda")
    else:
        TORCH_DEVICE = torch.device("cpu")
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None

try:
    import tensorflow as tf

    TF_AVAILABLE = True
    # Configure TensorFlow GPU settings
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.warning(f"TensorFlow GPU configuration error: {e}")
except ImportError:
    TF_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # Fallback to NumPy

try:
    import cudf
    import cuml

    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    cudf = pd  # Fallback to Pandas

# Configure logging
logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources and provides utilities for GPU acceleration."""

    def __init__(self):
        """Initialize GPU manager."""
        self.gpu_available = self._check_gpu_availability()
        self.device_info = self._get_device_info()
        self._log_gpu_status()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return True
        if TF_AVAILABLE and tf.config.list_physical_devices("GPU"):
            return True
        if CUPY_AVAILABLE:
            try:
                cp.cuda.Device(0)
                return True
            except:
                pass
        return False

    def _get_device_info(self) -> dict[str, Any]:
        """Get information about available devices."""
        info = {
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

        if TORCH_AVAILABLE and torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
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

        return info

    def _log_gpu_status(self):
        """Log GPU status information."""
        if self.gpu_available:
            logger.info(
                f"GPU acceleration enabled: {self.device_info['device_count']} device(s) available"
            )
            for device in self.device_info["devices"]:
                logger.info(
                    f"  Device {device['index']}: {device['name']} ({device['total_memory'] / 1e9:.2f} GB)"
                )
        else:
            logger.warning("GPU acceleration not available, using CPU")

        logger.debug(f"Available libraries: {self.device_info['libraries']}")

    def get_memory_info(self, device_id: int = 0) -> dict[str, float]:
        """Get GPU memory information."""
        if not self.gpu_available:
            return {"total": 0, "used": 0, "free": 0}

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            reserved = torch.cuda.memory_reserved(device_id)
            allocated = torch.cuda.memory_allocated(device_id)
            free = total - reserved

            return {
                "total": total / 1e9,  # Convert to GB
                "reserved": reserved / 1e9,
                "allocated": allocated / 1e9,
                "free": free / 1e9,
            }

        return {"total": 0, "used": 0, "free": 0}

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared PyTorch GPU cache")

        if TF_AVAILABLE:
            tf.keras.backend.clear_session()
            logger.debug("Cleared TensorFlow session")

    def to_gpu(self, data: Any, dtype: str | None = None) -> Any:
        """Move data to GPU if available."""
        if not self.gpu_available:
            return data

        # Handle NumPy arrays
        if isinstance(data, np.ndarray):
            if CUPY_AVAILABLE:
                return cp.asarray(data, dtype=dtype)
            elif TORCH_AVAILABLE:
                tensor = torch.from_numpy(data)
                if dtype:
                    tensor = tensor.to(getattr(torch, dtype))
                return tensor.to(TORCH_DEVICE)

        # Handle Pandas DataFrames
        if isinstance(data, pd.DataFrame):
            if RAPIDS_AVAILABLE:
                return cudf.from_pandas(data)
            elif TORCH_AVAILABLE:
                return torch.from_numpy(data.values).to(TORCH_DEVICE)

        # Handle PyTorch tensors
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            return data.to(TORCH_DEVICE)

        return data

    def to_cpu(self, data: Any) -> Any:
        """Move data from GPU to CPU."""
        # Handle CuPy arrays
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)

        # Handle PyTorch tensors
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            return data.cpu().numpy()

        # Handle cuDF DataFrames
        if RAPIDS_AVAILABLE and hasattr(data, "to_pandas"):
            return data.to_pandas()

        return data

    def accelerate_computation(self, func, *args, **kwargs):
        """Decorator to automatically accelerate computations on GPU."""
        if not self.gpu_available:
            return func(*args, **kwargs)

        # Move inputs to GPU
        gpu_args = [self.to_gpu(arg) for arg in args]
        gpu_kwargs = {k: self.to_gpu(v) for k, v in kwargs.items()}

        # Run computation
        try:
            result = func(*gpu_args, **gpu_kwargs)
            # Move result back to CPU if needed
            return self.to_cpu(result)
        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            return func(*args, **kwargs)


# Global GPU manager instance
gpu_manager = GPUManager()


def get_optimal_batch_size(data_size: int, memory_limit_gb: float = 4.0) -> int:
    """Calculate optimal batch size based on available GPU memory."""
    if not gpu_manager.gpu_available:
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


def parallel_apply(df: pd.DataFrame, func, axis: int = 0, use_gpu: bool = True) -> pd.DataFrame:
    """Apply function to DataFrame with GPU acceleration if available."""
    if use_gpu and gpu_manager.gpu_available and RAPIDS_AVAILABLE:
        # Convert to cuDF and apply
        gdf = cudf.from_pandas(df)
        result = gdf.apply(func, axis=axis)
        return result.to_pandas()

    # Fallback to pandas
    return df.apply(func, axis=axis)


def gpu_accelerated_correlation(data: np.ndarray) -> np.ndarray:
    """Calculate correlation matrix with GPU acceleration."""
    if gpu_manager.gpu_available:
        if CUPY_AVAILABLE:
            gpu_data = cp.asarray(data)
            corr = cp.corrcoef(gpu_data.T)
            return cp.asnumpy(corr)
        elif TORCH_AVAILABLE:
            gpu_data = torch.from_numpy(data).to(TORCH_DEVICE)
            # Standardize the data
            mean = gpu_data.mean(dim=0)
            std = gpu_data.std(dim=0)
            standardized = (gpu_data - mean) / (std + 1e-8)
            # Calculate correlation
            corr = torch.mm(standardized.T, standardized) / (gpu_data.shape[0] - 1)
            return corr.cpu().numpy()

    # Fallback to NumPy
    return np.corrcoef(data.T)


def gpu_accelerated_rolling_window(data: np.ndarray, window_size: int, func) -> np.ndarray:
    """Apply rolling window function with GPU acceleration."""
    if gpu_manager.gpu_available and CUPY_AVAILABLE:
        gpu_data = cp.asarray(data)
        result = cp.zeros(len(data) - window_size + 1)

        for i in range(len(result)):
            window = gpu_data[i : i + window_size]
            result[i] = func(window)

        return cp.asnumpy(result)

    # Fallback to NumPy
    result = np.zeros(len(data) - window_size + 1)
    for i in range(len(result)):
        window = data[i : i + window_size]
        result[i] = func(window)

    return result


def setup_gpu_logging():
    """Setup GPU usage monitoring and logging."""
    if not gpu_manager.gpu_available:
        return

    try:
        import gpustat
        import py3nvml

        # Log GPU stats periodically
        def log_gpu_stats():
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
    "gpu_accelerated_correlation",
    "gpu_accelerated_rolling_window",
    "gpu_manager",
    "parallel_apply",
    "setup_gpu_logging",
]
