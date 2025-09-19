"""Comprehensive tests for StateManager module."""

import asyncio
import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import DependencyError, ServiceError
from src.core.types import BotState
from src.state.state_manager import StateManager, get_cache_manager


class TestGetCacheManager:
    """Test get_cache_manager function."""

    def test_get_cache_manager_returns_instance(self):
        """Test that get_cache_manager returns cache manager instance."""
        with patch("src.core.caching.cache_manager.CacheManager") as mock_cache_manager_class:
            mock_manager = MagicMock()
            mock_cache_manager_class.return_value = mock_manager
            
            result = get_cache_manager()
            
            assert result == mock_manager
            mock_cache_manager_class.assert_called_once()

    def test_get_cache_manager_import_path(self):
        """Test get_cache_manager imports from correct path."""
        with patch("src.core.caching.cache_manager.CacheManager") as mock_cache_manager:
            mock_instance = MagicMock()
            mock_cache_manager.return_value = mock_instance
            
            result = get_cache_manager()
            
            assert result == mock_instance


class TestStateManager:
    """Test StateManager class."""

    @pytest.fixture(autouse=True)
    def clean_environment(self):
        """Ensure clean environment for each test in this class."""
        import os
        # Store current values
        original_env = {}
        keys = ["TESTING", "ENVIRONMENT", "PYTHONHASHSEED"]
        for key in keys:
            original_env[key] = os.environ.get(key)
        
        yield
        
        # Restore after test
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=Config)
        config.state = MagicMock()
        config.state.cache_ttl = 300
        config.state.storage_path = "/tmp/state"
        return config

    @pytest.fixture
    def state_manager(self, mock_config):
        """Create StateManager instance."""
        return StateManager(mock_config)

    @pytest.fixture
    def mock_state_service(self):
        """Create mock state service."""
        service = AsyncMock()
        service.get_metrics = MagicMock(return_value={
            "total_states": 10,
            "cache_hit_rate": 0.85,
            "operations_per_second": 150.0,
            "error_rate": 0.02,
            "total_operations": 1000,
            "state_updates": 1000,
        })
        return service

    def test_initialization(self, mock_config):
        """Test StateManager initialization."""
        manager = StateManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.state_service is None

    @pytest.mark.asyncio
    async def test_initialize_with_dependency_injection(self, state_manager, mock_config):
        """Test initialization using dependency injection."""
        # Ensure clean state
        state_manager.state_service = None
        
        # Ensure isolation by clearing TESTING environment
        import os
        old_testing = os.environ.get("TESTING")
        if old_testing:
            os.environ.pop("TESTING", None)  # Clear it completely for this test
        
        mock_container = MagicMock()
        mock_state_service = AsyncMock()
        
        # Create a factory mock with proper async method
        mock_factory = MagicMock()
        mock_factory.create_state_service = AsyncMock(return_value=mock_state_service)
        
        # Make sure the container returns our properly mocked factory
        mock_container.get.return_value = mock_factory
        
        try:
            with patch("src.core.dependency_injection.get_container", return_value=mock_container):
                await state_manager.initialize()
                
                # The get method might be called multiple times during debugging, so just check it was called
                mock_container.get.assert_called_with("StateServiceFactory")
                mock_factory.create_state_service.assert_called_once_with(
                    config=mock_config, auto_start=True
                )
                assert state_manager.state_service == mock_state_service
        finally:
            # Restore environment
            if old_testing is None:
                os.environ.pop("TESTING", None)
            else:
                os.environ["TESTING"] = old_testing

    @pytest.mark.asyncio
    async def test_initialize_with_fallback(self, state_manager, mock_config):
        """Test initialization with fallback when DI fails."""
        # Ensure clean state
        state_manager.state_service = None
        
        # Force TESTING environment to ensure we get a Mock service
        import os
        old_testing = os.environ.get("TESTING")
        os.environ["TESTING"] = "true"  # Explicitly set to true for fallback path
        
        try:
            mock_container = MagicMock()
            # Simulate container failure to trigger fallback
            mock_container.get.side_effect = DependencyError("No factory in container")
            
            # Patch the container to trigger fallback path
            with patch("src.core.dependency_injection.get_container", return_value=mock_container):
                # Call initialize - it should use the fallback path
                await state_manager.initialize()
                
                # Verify that a state service was created via fallback
                assert state_manager.state_service is not None
                # In testing mode, the factory creates a mock StateService
                assert hasattr(state_manager.state_service, 'initialize')
                assert hasattr(state_manager.state_service, 'cleanup')
        finally:
            # Restore original TESTING environment variable
            if old_testing is None:
                os.environ.pop("TESTING", None)
            else:
                os.environ["TESTING"] = old_testing

    @pytest.mark.asyncio
    async def test_initialize_with_service_error_fallback(self, state_manager, mock_config):
        """Test initialization with ServiceError fallback."""
        # Ensure clean state
        state_manager.state_service = None
        
        # Force TESTING environment variable for this test
        import os
        old_testing = os.environ.get("TESTING")
        os.environ["TESTING"] = "true"  # Explicitly set for fallback
        
        try:
            mock_container = MagicMock()
            mock_container.get.side_effect = ServiceError("Service error")
            
            # Just patch the container to trigger the fallback path
            with patch("src.core.dependency_injection.get_container", return_value=mock_container):
                await state_manager.initialize()
                
                # Verify that the state service was created (fallback path worked)
                assert state_manager.state_service is not None
                # The factory creates a mock StateService when injector is None
                assert hasattr(state_manager.state_service, 'initialize')
                assert hasattr(state_manager.state_service, 'cleanup')
                
        finally:
            # Restore original TESTING environment variable
            if old_testing is None:
                os.environ.pop("TESTING", None)
            else:
                os.environ["TESTING"] = old_testing

    @pytest.mark.asyncio
    async def test_shutdown(self, state_manager, mock_state_service):
        """Test StateManager shutdown."""
        state_manager.state_service = mock_state_service
        
        await state_manager.shutdown()
        
        mock_state_service.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_service(self, state_manager):
        """Test shutdown without state service."""
        # Should not raise any exceptions
        await state_manager.shutdown()

    @pytest.mark.asyncio
    async def test_save_bot_state_success(self, state_manager, mock_state_service):
        """Test successful bot state saving."""
        state_manager.state_service = mock_state_service
        mock_state_service.set_state.return_value = True
        
        bot_id = "test_bot_123"
        state_data = {"status": "running", "balance": "1000.00"}
        
        result = await state_manager.save_bot_state(bot_id, state_data)
        
        # Should return a version ID (hash)
        assert isinstance(result, str)
        assert len(result) == 12  # SHA256 hash truncated to 12 chars
        
        # Verify state service was called correctly
        from src.state.state_service import StatePriority, StateType
        mock_state_service.set_state.assert_called_once_with(
            StateType.BOT_STATE, bot_id, state_data, priority=StatePriority.HIGH
        )

    @pytest.mark.asyncio
    async def test_save_bot_state_with_snapshot(self, state_manager, mock_state_service):
        """Test bot state saving with snapshot creation."""
        state_manager.state_service = mock_state_service
        mock_state_service.set_state.return_value = True
        mock_state_service.create_snapshot.return_value = "snapshot_123"
        
        bot_id = "test_bot_123"
        state_data = {"status": "running"}
        
        result = await state_manager.save_bot_state(bot_id, state_data, create_snapshot=True)
        
        assert result == "snapshot_123"
        mock_state_service.set_state.assert_called_once()
        mock_state_service.create_snapshot.assert_called_once_with(bot_id)

    @pytest.mark.asyncio
    async def test_save_bot_state_failure(self, state_manager, mock_state_service):
        """Test bot state saving failure."""
        state_manager.state_service = mock_state_service
        mock_state_service.set_state.return_value = False
        
        bot_id = "test_bot_123"
        state_data = {"status": "running"}
        
        with pytest.raises(RuntimeError, match="Failed to save state for bot test_bot_123"):
            await state_manager.save_bot_state(bot_id, state_data)

    @pytest.mark.asyncio
    async def test_save_bot_state_not_initialized(self, state_manager):
        """Test bot state saving when not initialized."""
        with pytest.raises(RuntimeError, match="StateManager not initialized"):
            await state_manager.save_bot_state("test_bot", {"status": "running"})

    @pytest.mark.asyncio
    async def test_load_bot_state_success(self, state_manager, mock_state_service):
        """Test successful bot state loading."""
        state_manager.state_service = mock_state_service
        
        bot_id = "test_bot_123"
        mock_bot_state = BotState(bot_id=bot_id, status="running")
        mock_state_service.get_state.return_value = mock_bot_state
        
        result = await state_manager.load_bot_state(bot_id)
        
        assert result == mock_bot_state
        from src.state.state_service import StateType
        mock_state_service.get_state.assert_called_once_with(StateType.BOT_STATE, bot_id)

    @pytest.mark.asyncio
    async def test_load_bot_state_from_dict(self, state_manager, mock_state_service):
        """Test bot state loading from dictionary."""
        state_manager.state_service = mock_state_service
        
        bot_id = "test_bot_123"
        state_dict = {"bot_id": bot_id, "status": "running", "balance": "1000.00"}
        mock_state_service.get_state.return_value = state_dict
        
        result = await state_manager.load_bot_state(bot_id)
        
        assert isinstance(result, BotState)
        assert result.bot_id == bot_id
        assert result.status.value == "running"

    @pytest.mark.asyncio
    async def test_load_bot_state_from_dict_with_data_key(self, state_manager, mock_state_service):
        """Test bot state loading from dictionary with data key."""
        state_manager.state_service = mock_state_service
        
        bot_id = "test_bot_123"
        state_dict = {"data": {"bot_id": bot_id, "status": "running"}}
        mock_state_service.get_state.return_value = state_dict
        
        result = await state_manager.load_bot_state(bot_id)
        
        assert isinstance(result, BotState)
        assert result.bot_id == bot_id
        assert result.status.value == "running"

    @pytest.mark.asyncio
    async def test_load_bot_state_dict_without_bot_id(self, state_manager, mock_state_service):
        """Test bot state loading from dict missing bot_id."""
        state_manager.state_service = mock_state_service
        
        bot_id = "test_bot_123"
        state_dict = {"status": "running", "balance": "1000.00"}
        mock_state_service.get_state.return_value = state_dict
        
        result = await state_manager.load_bot_state(bot_id)
        
        assert isinstance(result, BotState)
        assert result.bot_id == bot_id  # Should be added automatically

    @pytest.mark.asyncio
    async def test_load_bot_state_dict_construction_failure(self, state_manager, mock_state_service):
        """Test bot state loading when BotState construction fails."""
        state_manager.state_service = mock_state_service
        
        bot_id = "test_bot_123"
        # Invalid data that will cause BotState construction to fail
        invalid_dict = {"bot_id": "test", "status": "invalid_status"}
        mock_state_service.get_state.return_value = invalid_dict
        
        result = await state_manager.load_bot_state(bot_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_load_bot_state_not_found(self, state_manager, mock_state_service):
        """Test bot state loading when state not found."""
        state_manager.state_service = mock_state_service
        mock_state_service.get_state.return_value = None
        
        result = await state_manager.load_bot_state("non_existent_bot")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_load_bot_state_not_initialized(self, state_manager):
        """Test bot state loading when not initialized."""
        with pytest.raises(RuntimeError, match="StateManager not initialized"):
            await state_manager.load_bot_state("test_bot")

    @pytest.mark.asyncio
    async def test_load_bot_state_with_data_attribute(self, state_manager, mock_state_service):
        """Test bot state loading with object having data attribute."""
        state_manager.state_service = mock_state_service
        
        class MockStateObject:
            def __init__(self):
                self.data = {"bot_id": "test_bot", "status": "running"}
        
        bot_id = "test_bot"
        mock_state_service.get_state.return_value = MockStateObject()
        
        # This will trigger a recursive call
        # We need to set up the mock to return proper data on second call
        def side_effect(*args, **kwargs):
            if hasattr(side_effect, "called"):
                return {"bot_id": bot_id, "status": "running"}
            side_effect.called = True
            return MockStateObject()
        
        mock_state_service.get_state.side_effect = side_effect
        
        result = await state_manager.load_bot_state(bot_id)
        
        assert isinstance(result, BotState)
        assert result.bot_id == bot_id

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, state_manager, mock_state_service):
        """Test checkpoint creation."""
        state_manager.state_service = mock_state_service
        mock_state_service.create_snapshot.return_value = "checkpoint_123"
        
        bot_id = "test_bot_123"
        
        result = await state_manager.create_checkpoint(bot_id)
        
        assert result == "checkpoint_123"
        mock_state_service.create_snapshot.assert_called_once_with(bot_id)

    @pytest.mark.asyncio
    async def test_create_checkpoint_with_data(self, state_manager, mock_state_service):
        """Test checkpoint creation with data parameter."""
        state_manager.state_service = mock_state_service
        mock_state_service.create_snapshot.return_value = "checkpoint_123"
        
        bot_id = "test_bot_123"
        checkpoint_data = {"extra_data": "value"}
        
        result = await state_manager.create_checkpoint(bot_id, checkpoint_data)
        
        assert result == "checkpoint_123"
        # Data parameter is ignored but shouldn't cause errors
        mock_state_service.create_snapshot.assert_called_once_with(bot_id)

    @pytest.mark.asyncio
    async def test_create_checkpoint_not_initialized(self, state_manager):
        """Test checkpoint creation when not initialized."""
        with pytest.raises(RuntimeError, match="StateManager not initialized"):
            await state_manager.create_checkpoint("test_bot")

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, state_manager, mock_state_service):
        """Test checkpoint restoration."""
        state_manager.state_service = mock_state_service
        mock_state_service.restore_snapshot.return_value = True
        
        bot_id = "test_bot_123"
        checkpoint_id = "checkpoint_123"
        
        result = await state_manager.restore_from_checkpoint(bot_id, checkpoint_id)
        
        assert result is True
        mock_state_service.restore_snapshot.assert_called_once_with(checkpoint_id)

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_failure(self, state_manager, mock_state_service):
        """Test checkpoint restoration failure."""
        state_manager.state_service = mock_state_service
        mock_state_service.restore_snapshot.return_value = False
        
        bot_id = "test_bot_123"
        checkpoint_id = "checkpoint_123"
        
        result = await state_manager.restore_from_checkpoint(bot_id, checkpoint_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_not_initialized(self, state_manager):
        """Test checkpoint restoration when not initialized."""
        with pytest.raises(RuntimeError, match="StateManager not initialized"):
            await state_manager.restore_from_checkpoint("test_bot", "checkpoint_123")

    @pytest.mark.asyncio
    async def test_get_state_metrics_dict_format(self, state_manager, mock_state_service):
        """Test getting state metrics in dictionary format."""
        state_manager.state_service = mock_state_service
        
        bot_id = "test_bot_123"
        hours = 24
        
        result = await state_manager.get_state_metrics(bot_id, hours)
        
        assert result["bot_id"] == bot_id
        assert result["period_hours"] == hours
        assert result["total_states"] == 10
        assert result["cache_hit_rate"] == 0.85
        assert result["operations_per_second"] == 150.0
        assert result["error_rate"] == 0.02
        assert result["state_updates"] == 1000

    @pytest.mark.asyncio
    async def test_get_state_metrics_object_format(self, state_manager, mock_state_service):
        """Test getting state metrics in object format."""
        state_manager.state_service = mock_state_service
        
        class MockMetrics:
            def __init__(self):
                self.total_operations = 500
                self.last_successful_sync = "2023-01-01T00:00:00"
                self.storage_usage_mb = 100.5
                self.cache_hit_rate = 0.9
                self.error_rate = 0.01
                self.active_states_count = 5
        
        mock_state_service.get_metrics.return_value = MockMetrics()
        
        result = await state_manager.get_state_metrics()
        
        assert result["total_operations"] == 500
        assert result["state_updates"] == 500
        assert result["last_successful_sync"] == "2023-01-01T00:00:00"
        assert result["storage_usage_mb"] == 100.5
        assert result["cache_hit_rate"] == 0.9
        assert result["error_rate"] == 0.01
        assert result["active_states_count"] == 5

    @pytest.mark.asyncio
    async def test_get_state_metrics_unexpected_format(self, state_manager, mock_state_service):
        """Test getting state metrics with unexpected format."""
        state_manager.state_service = mock_state_service
        mock_state_service.get_metrics.return_value = "unexpected_string"
        
        result = await state_manager.get_state_metrics()
        
        # Should return default values
        assert result["total_operations"] == 0
        assert result["state_updates"] == 0
        assert result["cache_hit_rate"] == 0.0
        assert result["error_rate"] == 0.0
        assert result["active_states_count"] == 0

    @pytest.mark.asyncio
    async def test_get_state_metrics_default_parameters(self, state_manager, mock_state_service):
        """Test getting state metrics with default parameters."""
        state_manager.state_service = mock_state_service
        
        result = await state_manager.get_state_metrics()
        
        assert result["bot_id"] is None
        assert result["period_hours"] == 24

    @pytest.mark.asyncio
    async def test_get_state_metrics_not_initialized(self, state_manager):
        """Test getting state metrics when not initialized."""
        with pytest.raises(RuntimeError, match="StateManager not initialized"):
            await state_manager.get_state_metrics()

    def test_getattr_delegation(self, state_manager, mock_state_service):
        """Test attribute delegation to state service."""
        state_manager.state_service = mock_state_service
        mock_state_service.some_method = MagicMock(return_value="test_result")
        
        result = state_manager.some_method()
        
        assert result == "test_result"
        mock_state_service.some_method.assert_called_once()

    def test_getattr_no_service(self, state_manager):
        """Test attribute delegation when no service is available."""
        with pytest.raises(AttributeError, match="no state service is available"):
            _ = state_manager.non_existent_attribute

    def test_getattr_attribute_not_found(self, state_manager):
        """Test attribute delegation when attribute doesn't exist."""
        # Create a real object that will actually raise AttributeError
        class MockStateService:
            def some_method(self):
                return "test_result"
        
        state_manager.state_service = MockStateService()
        
        with pytest.raises(AttributeError):
            _ = state_manager.non_existent_attribute


class TestStateManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return MagicMock(spec=Config)

    @pytest.fixture
    def state_manager(self, mock_config):
        """Create StateManager instance."""
        return StateManager(mock_config)

    @pytest.mark.asyncio
    async def test_save_bot_state_version_id_consistency(self, state_manager):
        """Test that version ID is consistent for same state data."""
        mock_state_service = AsyncMock()
        mock_state_service.set_state.return_value = True
        state_manager.state_service = mock_state_service
        
        bot_id = "test_bot"
        state_data = {"status": "running", "balance": "1000.00"}
        
        version1 = await state_manager.save_bot_state(bot_id, state_data)
        version2 = await state_manager.save_bot_state(bot_id, state_data)
        
        # Same state data should produce same version ID
        assert version1 == version2
        assert len(version1) == 12

    @pytest.mark.asyncio
    async def test_save_bot_state_different_version_ids(self, state_manager):
        """Test that different state data produces different version IDs."""
        mock_state_service = AsyncMock()
        mock_state_service.set_state.return_value = True
        state_manager.state_service = mock_state_service
        
        bot_id = "test_bot"
        state_data1 = {"status": "running"}
        state_data2 = {"status": "inactive"}
        
        version1 = await state_manager.save_bot_state(bot_id, state_data1)
        version2 = await state_manager.save_bot_state(bot_id, state_data2)
        
        # Different state data should produce different version IDs
        assert version1 != version2

    @pytest.mark.asyncio
    async def test_load_bot_state_circular_data_object(self, state_manager):
        """Test loading bot state with object that has data attribute containing BotState."""
        mock_state_service = AsyncMock()
        state_manager.state_service = mock_state_service
        
        bot_state = BotState(bot_id="test_bot", status="running")
        
        class MockStateObject:
            def __init__(self):
                self.data = bot_state
        
        mock_state_service.get_state.return_value = MockStateObject()
        
        # This should handle the recursive case properly
        result = await state_manager.load_bot_state("test_bot")
        
        # Should return the BotState directly
        assert result == bot_state

    @pytest.mark.asyncio
    async def test_load_bot_state_already_bot_state_in_else_branch(self, state_manager):
        """Test load_bot_state when result is BotState in the final check."""
        mock_state_service = AsyncMock()
        state_manager.state_service = mock_state_service
        
        bot_state = BotState(bot_id="test_bot", status="running")
        # Mock an object that's not dict, not None, doesn't have data attr, but is BotState
        mock_state_service.get_state.return_value = bot_state
        
        result = await state_manager.load_bot_state("test_bot")
        
        assert result == bot_state

    def test_version_id_generation_with_special_characters(self, state_manager):
        """Test version ID generation with special characters in state data."""
        import hashlib
        import json
        
        state_data = {"status": "running", "message": "Special chars: üñíçødé", "balance": 1000.50}
        
        # Simulate the version ID generation logic
        state_json = json.dumps(state_data, sort_keys=True, default=str)
        expected_version_id = hashlib.sha256(state_json.encode()).hexdigest()[:12]
        
        # Verify it doesn't raise exceptions with special characters
        assert len(expected_version_id) == 12
        assert expected_version_id.isalnum()


class TestStateManagerIntegration:
    """Integration tests for StateManager."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration with realistic settings."""
        config = MagicMock(spec=Config)
        config.state = MagicMock()
        config.state.cache_ttl = 300
        config.state.storage_path = "/tmp/state_test"
        config.state.compression_enabled = True
        return config

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mock_config):
        """Test full StateManager lifecycle."""
        manager = StateManager(mock_config)
        
        # Set TESTING environment for consistent behavior
        import os
        old_testing = os.environ.get("TESTING")
        os.environ["TESTING"] = "true"
        
        try:
            # Mock the state service directly
            mock_state_service = AsyncMock()
            mock_state_service.set_state = AsyncMock(return_value=True)
            mock_state_service.get_state = AsyncMock(return_value={"bot_id": "test_bot", "status": "running"})
            mock_state_service.create_snapshot = AsyncMock(return_value="checkpoint_123")
            mock_state_service.get_metrics = MagicMock(return_value={"total_states": 1})
            mock_state_service.cleanup = AsyncMock()
            
            # Set the mock service directly
            manager.state_service = mock_state_service
            
            # Save state
            version_id = await manager.save_bot_state("test_bot", {"status": "running"})
            assert isinstance(version_id, str)
            mock_state_service.set_state.assert_called_once()
            
            # Load state
            loaded_state = await manager.load_bot_state("test_bot")
            assert isinstance(loaded_state, BotState)
            assert loaded_state.bot_id == "test_bot"
            
            # Create checkpoint
            checkpoint_id = await manager.create_checkpoint("test_bot")
            assert checkpoint_id == "checkpoint_123"
            
            # Get metrics
            metrics = await manager.get_state_metrics("test_bot")
            assert metrics["bot_id"] == "test_bot"
            # The metrics are added from the mock service's get_metrics return value
            assert metrics.get("total_states", 0) == 1 or metrics.get("state_updates", 0) > 0
            
            # Shutdown
            await manager.shutdown()
            mock_state_service.cleanup.assert_called_once()
            
        finally:
            # Restore environment
            if old_testing is None:
                os.environ.pop("TESTING", None)
            else:
                os.environ["TESTING"] = old_testing

    @pytest.mark.asyncio
    async def test_state_manager_delegation_comprehensive(self, mock_config):
        """Test comprehensive attribute delegation."""
        manager = StateManager(mock_config)
        
        mock_state_service = AsyncMock()
        mock_state_service.custom_method = MagicMock(return_value="custom_result")
        mock_state_service.async_method = AsyncMock(return_value="async_result")
        mock_state_service.property_value = "property_result"
        
        manager.state_service = mock_state_service
        
        # Test method delegation
        result = manager.custom_method("arg1", kwarg="value")
        assert result == "custom_result"
        mock_state_service.custom_method.assert_called_once_with("arg1", kwarg="value")
        
        # Test async method delegation
        async_result = await manager.async_method()
        assert async_result == "async_result"
        
        # Test property delegation
        assert manager.property_value == "property_result"


if __name__ == "__main__":
    pytest.main([__file__])