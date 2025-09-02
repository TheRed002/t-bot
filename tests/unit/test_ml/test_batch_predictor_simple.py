"""
Unit tests for ML batch predictor service.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

import pandas as pd
import pytest

from src.ml.inference.batch_predictor import (
    BatchPredictorConfig,
    BatchPredictorService,
)


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        "batch_predictor": {
            "max_batch_size": 500,
            "batch_timeout_minutes": 10,
            "enable_result_persistence": True,
            "cleanup_interval_hours": 2,
            "max_concurrent_jobs": 3,
        }
    }


class TestBatchPredictorConfig:
    """Test batch predictor configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchPredictorConfig()
        
        assert config.max_batch_size == 1000
        assert config.batch_timeout_minutes == 30
        assert config.enable_result_persistence is True
        assert config.cleanup_interval_hours == 2
        assert config.max_concurrent_jobs == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BatchPredictorConfig(
            max_batch_size=2000,
            batch_timeout_minutes=60,
            enable_result_persistence=False,
        )
        
        assert config.max_batch_size == 2000
        assert config.batch_timeout_minutes == 60
        assert config.enable_result_persistence is False


class TestBatchPredictorService:
    """Test batch predictor service."""

    @pytest.fixture
    def service(self, sample_config):
        """Create service instance for tests."""
        return BatchPredictorService(config=sample_config)

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.name == "BatchPredictorService"
        assert isinstance(service._config, dict)  # Config is stored internally as dict
        assert hasattr(service, 'bp_config')
        assert isinstance(service.bp_config, BatchPredictorConfig)
        assert service.bp_config.max_batch_size == 500

    def test_initialization_no_config(self):
        """Test initialization without config."""
        service = BatchPredictorService()
        assert isinstance(service._config, dict)  # Config is stored internally as dict
        assert hasattr(service, 'bp_config')
        assert isinstance(service.bp_config, BatchPredictorConfig)
        # Should use defaults
        assert service.bp_config.max_batch_size == 1000

    @pytest.mark.asyncio
    async def test_start_service(self, service):
        """Test service startup."""
        # Mock the dependencies
        mock_container = Mock()
        mock_container.get = Mock(return_value=Mock())
        service.configure_dependencies(mock_container)
        
        await service.start()
        
        assert service.is_running
        
        # Cleanup
        await service.stop()

    @pytest.mark.asyncio
    async def test_stop_service(self, service):
        """Test service shutdown."""
        # Mock the dependencies
        mock_container = Mock()
        mock_container.get = Mock(return_value=Mock())
        service.configure_dependencies(mock_container)
        
        await service.start()
        await service.stop()
        
        assert not service.is_running

    @pytest.mark.asyncio
    async def test_submit_batch_prediction_basic(self, service):
        """Test basic batch prediction submission."""
        service._is_running = True
        
        input_data = pd.DataFrame({
            'price': [100.0, 101.0, 99.0],
            'volume': [1000, 1100, 900]
        })
        
        # Should not raise exception (may return None or job_id)
        result = await service.submit_batch_prediction(
            model_id="test_model",
            input_data=input_data,
        )

    @pytest.mark.asyncio
    async def test_submit_batch_prediction_service_not_running(self, service):
        """Test batch prediction when service not running."""
        input_data = pd.DataFrame({'price': [100.0]})
        
        result = await service.submit_batch_prediction("model", input_data)
        # Should handle gracefully

    def test_get_job_status_basic(self, service):
        """Test getting job status."""
        job_id = "test_job_123"
        
        # Should not raise exception
        result = service.get_job_status(job_id)

    def test_get_job_result_basic(self, service):
        """Test getting job result."""
        job_id = "test_job_123"
        
        # Should not raise exception
        result = service.get_job_result(job_id)

    def test_list_jobs_basic(self, service):
        """Test listing jobs."""
        # Should not raise exception
        result = service.list_jobs()
        assert isinstance(result, list)

    def test_cancel_job_basic(self, service):
        """Test cancelling job."""
        job_id = "test_job_123"
        
        # Should not raise exception
        result = service.cancel_job(job_id)
        assert isinstance(result, bool)

    def test_get_service_statistics(self, service):
        """Test getting service statistics."""
        stats = service.get_service_statistics()
        assert isinstance(stats, dict)

    def test_cleanup_old_jobs(self, service):
        """Test cleaning up old jobs."""
        # Should not raise exception
        result = service.cleanup_old_jobs()
        assert isinstance(result, int)


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def service(self, sample_config):
        """Create service for error testing."""
        return BatchPredictorService(config=sample_config)

    @pytest.mark.asyncio
    async def test_submit_empty_data(self, service):
        """Test submitting empty data."""
        service._is_running = True
        
        empty_data = pd.DataFrame()
        
        # Should handle gracefully
        result = await service.submit_batch_prediction("model", empty_data)

    @pytest.mark.asyncio
    async def test_submit_large_data(self, service):
        """Test submitting data larger than max batch size."""
        service._is_running = True
        
        # Create data larger than max batch size
        large_data = pd.DataFrame({
            'price': range(service.bp_config.max_batch_size + 1)
        })
        
        # Should handle gracefully (may reject or process)
        result = await service.submit_batch_prediction("model", large_data)

    def test_get_nonexistent_job_status(self, service):
        """Test getting status of non-existent job."""
        result = service.get_job_status("nonexistent_job")
        # Should not raise exception

    def test_get_nonexistent_job_result(self, service):
        """Test getting result of non-existent job."""
        result = service.get_job_result("nonexistent_job")
        # Should not raise exception

    def test_cancel_nonexistent_job(self, service):
        """Test cancelling non-existent job."""
        result = service.cancel_job("nonexistent_job")
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_submit_none_data(self, service):
        """Test submitting None data."""
        service._is_running = True
        
        # Should handle gracefully
        result = await service.submit_batch_prediction("model", None)

    @pytest.mark.asyncio
    async def test_submit_invalid_model_id(self, service):
        """Test submitting with invalid model ID."""
        service._is_running = True
        
        input_data = pd.DataFrame({'price': [100.0]})
        
        # Should handle gracefully
        result = await service.submit_batch_prediction("", input_data)
        result = await service.submit_batch_prediction(None, input_data)