"""
Unit tests for ML repository layer.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from src.ml.repository import IMLRepository, MLRepository


class TestMLRepository:
    """Test ML repository implementation."""

    @pytest.fixture
    def repository(self):
        """Create repository instance for tests."""
        repo = MLRepository()
        # Mock the data service dependency
        mock_data_service = AsyncMock()
        mock_data_service.store_ml_model_metadata = AsyncMock()
        mock_data_service.get_ml_model_by_id = AsyncMock()
        mock_data_service.get_ml_models_by_criteria = AsyncMock()
        mock_data_service.update_ml_model_metadata = AsyncMock()
        mock_data_service.delete_ml_model = AsyncMock()
        mock_data_service.store_ml_prediction = AsyncMock()
        mock_data_service.get_ml_predictions = AsyncMock()
        mock_data_service.store_ml_training_job = AsyncMock()
        mock_data_service.get_ml_training_job = AsyncMock()
        mock_data_service.update_ml_training_job = AsyncMock()
        repo.data_service = mock_data_service
        return repo

    def test_initialization(self, repository):
        """Test repository initialization."""
        assert repository.name == "MLRepository"
        assert len(repository._models) == 0
        assert len(repository._audit_entries) == 0

    @pytest.mark.asyncio
    async def test_store_model_metadata_success(self, repository):
        """Test successful model metadata storage."""
        metadata = {
            "model_id": "test_model_123",
            "name": "Test Model",
            "model_type": "classifier",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
        }

        repository.data_service.store_ml_model_metadata.return_value = {"id": "test_model_123", **metadata}
        
        model_id = await repository.store_model_metadata(metadata)

        assert model_id == "test_model_123"
        assert "test_model_123" in repository._models
        repository.data_service.store_ml_model_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_model_metadata_no_id(self, repository):
        """Test storing model metadata without ID - should work and generate ID."""
        metadata = {"name": "Test Model", "model_type": "classifier"}

        # Mock data service to return stored metadata with generated ID
        repository.data_service.store_ml_model_metadata.return_value = {"id": "generated_id_123", **metadata}
        
        model_id = await repository.store_model_metadata(metadata)
        
        # Should successfully generate and return an ID
        assert isinstance(model_id, str)
        assert len(model_id) > 0
        repository.data_service.store_ml_model_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_by_id_exists(self, repository):
        """Test getting existing model by ID."""
        metadata = {
            "model_id": "existing_model",
            "name": "Existing Model",
            "model_type": "regressor",
        }

        repository.data_service.store_ml_model_metadata.return_value = {"id": "existing_model", **metadata}
        repository.data_service.get_ml_model_by_id.return_value = metadata
        
        await repository.store_model_metadata(metadata)
        
        result = await repository.get_model_by_id("existing_model")

        assert result == metadata
        repository.data_service.get_ml_model_by_id.assert_called_once_with("existing_model")

    @pytest.mark.asyncio
    async def test_get_model_by_id_not_exists(self, repository):
        """Test getting non-existent model by ID."""
        repository.data_service.get_ml_model_by_id.return_value = None
        
        result = await repository.get_model_by_id("nonexistent")

        assert result is None
        repository.data_service.get_ml_model_by_id.assert_called_once_with("nonexistent")

    @pytest.mark.asyncio
    async def test_get_models_by_name_and_type(self, repository):
        """Test getting models by name and type."""
        # Store multiple models
        models = [
            {
                "model_id": "model_1",
                "name": "Test Model",
                "model_type": "classifier",
                "version": "1.0",
            },
            {
                "model_id": "model_2",
                "name": "Test Model",
                "model_type": "classifier",
                "version": "2.0",
            },
            {
                "model_id": "model_3",
                "name": "Other Model",
                "model_type": "regressor",
                "version": "1.0",
            },
        ]

        # Mock data service responses
        for i, model in enumerate(models):
            repository.data_service.store_ml_model_metadata.return_value = {"id": model["model_id"], **model}
            await repository.store_model_metadata(model)

        # Mock the query response
        expected_models = [models[0], models[1]]  # Only classifiers with "Test Model" name
        repository.data_service.get_ml_models_by_criteria.return_value = expected_models
        
        # Find models by name and type
        result = await repository.get_models_by_name_and_type("Test Model", "classifier")

        assert len(result) == 2
        assert all(m["name"] == "Test Model" for m in result)
        assert all(m["model_type"] == "classifier" for m in result)

    @pytest.mark.asyncio
    async def test_find_models_all_criteria(self, repository):
        """Test finding models with all criteria."""
        model = {
            "model_id": "findable_model",
            "name": "Findable Model",
            "model_type": "classifier",
            "version": "1.0",
            "stage": "production",
            "active": True,
        }

        repository.data_service.store_ml_model_metadata.return_value = {"id": "findable_model", **model}
        await repository.store_model_metadata(model)
        
        # Mock the find_models query
        repository.data_service.get_ml_models_by_criteria.return_value = [model]
        
        result = await repository.find_models(
            name="Findable Model",
            model_type="classifier",
            version="1.0",
            stage="production",
            active_only=True,
        )

        assert len(result) == 1
        assert result[0]["model_id"] == "findable_model"

    @pytest.mark.asyncio
    async def test_find_models_partial_criteria(self, repository):
        """Test finding models with partial criteria."""
        models = [
            {
                "model_id": "model_a",
                "name": "Model A",
                "model_type": "classifier",
                "version": "1.0",
                "active": True,
            },
            {
                "model_id": "model_b",
                "name": "Model B",
                "model_type": "classifier",
                "version": "1.0",
                "active": True,
            },
        ]

        for i, model in enumerate(models):
            repository.data_service.store_ml_model_metadata.return_value = {"id": model["model_id"], **model}
            await repository.store_model_metadata(model)

        # Mock the query to return both classifier models
        repository.data_service.get_ml_models_by_criteria.return_value = models
        
        # Find by type only
        result = await repository.find_models(model_type="classifier")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_find_models_no_criteria(self, repository):
        """Test finding models with no criteria."""
        model = {
            "model_id": "any_model",
            "name": "Any Model",
            "model_type": "regressor",
            "active": True,
        }

        repository.data_service.store_ml_model_metadata.return_value = {"id": "any_model", **model}
        await repository.store_model_metadata(model)
        
        # Mock the query to return the model
        repository.data_service.get_ml_models_by_criteria.return_value = [model]
        
        result = await repository.find_models()

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_get_all_models_no_filter(self, repository):
        """Test getting all models without filters."""
        models = [
            {"model_id": "model_1", "name": "Model 1", "model_type": "classifier", "active": True},
            {"model_id": "model_2", "name": "Model 2", "model_type": "regressor", "active": True},
        ]

        for i, model in enumerate(models):
            repository.data_service.store_ml_model_metadata.return_value = {"id": model["model_id"], **model}
            await repository.store_model_metadata(model)
        
        # Mock get_all_models to return all models
        repository.data_service.get_ml_models_by_criteria.return_value = models
        
        result = await repository.get_all_models()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_all_models_with_type_filter(self, repository):
        """Test getting all models with type filter."""
        models = [
            {
                "model_id": "classifier_1",
                "name": "Classifier",
                "model_type": "classifier",
                "active": True,
            },
            {
                "model_id": "regressor_1",
                "name": "Regressor",
                "model_type": "regressor",
                "active": True,
            },
        ]

        for i, model in enumerate(models):
            repository.data_service.store_ml_model_metadata.return_value = {"id": model["model_id"], **model}
            await repository.store_model_metadata(model)
        
        # Mock to return only classifier
        repository.data_service.get_ml_models_by_criteria.return_value = [models[0]]
        
        result = await repository.get_all_models(model_type="classifier")

        assert len(result) == 1
        assert result[0]["model_type"] == "classifier"

    @pytest.mark.asyncio
    async def test_get_all_models_active_only(self, repository):
        """Test getting only active models."""
        models = [
            {
                "model_id": "active_model",
                "name": "Active Model",
                "model_type": "classifier",
                "active": True,
            },
            {
                "model_id": "inactive_model",
                "name": "Inactive Model",
                "model_type": "classifier",
                "active": False,
            },
        ]

        for i, model in enumerate(models):
            repository.data_service.store_ml_model_metadata.return_value = {"id": model["model_id"], **model}
            await repository.store_model_metadata(model)
        
        # Mock to return only active model
        repository.data_service.get_ml_models_by_criteria.return_value = [models[0]]
        
        result = await repository.get_all_models(active_only=True)

        assert len(result) == 1
        assert result[0]["active"] is True

    @pytest.mark.asyncio
    async def test_update_model_metadata_success(self, repository):
        """Test successful model metadata update."""
        original_metadata = {
            "model_id": "update_model",
            "name": "Original Name",
            "model_type": "classifier",
            "version": "1.0",
        }

        repository.data_service.store_ml_model_metadata.return_value = {"id": "update_model", **original_metadata}
        repository.data_service.update_ml_model_metadata.return_value = True
        repository.data_service.get_ml_model_by_id.return_value = {
            "model_id": "update_model",
            "name": "Updated Name",
            "model_type": "classifier",
            "version": "1.1",
        }
        
        await repository.store_model_metadata(original_metadata)

        updated_metadata = {
            "model_id": "update_model",
            "name": "Updated Name",
            "model_type": "classifier",
            "version": "1.1",
        }

        success = await repository.update_model_metadata("update_model", updated_metadata)

        assert success is True

        # Verify update
        result = await repository.get_model_by_id("update_model")
        assert result["name"] == "Updated Name"
        assert result["version"] == "1.1"

    @pytest.mark.asyncio
    async def test_update_model_metadata_not_exists(self, repository):
        """Test updating non-existent model metadata."""
        repository.data_service.update_ml_model_metadata.return_value = False
        
        success = await repository.update_model_metadata("nonexistent", {"name": "Updated"})

        assert success is False

    @pytest.mark.asyncio
    async def test_delete_model_success(self, repository):
        """Test successful model deletion."""
        metadata = {"model_id": "delete_model", "name": "To Delete", "model_type": "classifier"}

        repository.data_service.store_ml_model_metadata.return_value = {"id": "delete_model", **metadata}
        repository.data_service.delete_ml_model.return_value = True
        
        await repository.store_model_metadata(metadata)
        assert "delete_model" in repository._models

        success = await repository.delete_model("delete_model")

        assert success is True
        assert "delete_model" not in repository._models

    @pytest.mark.asyncio
    async def test_delete_model_not_exists(self, repository):
        """Test deleting non-existent model."""
        repository.data_service.delete_ml_model.return_value = False
        
        success = await repository.delete_model("nonexistent")

        assert success is False

    @pytest.mark.asyncio
    async def test_store_audit_entry(self, repository):
        """Test storing audit entry."""
        entry = {
            "action": "model_created",
            "model_id": "audit_model",
            "timestamp": "2024-01-01T00:00:00Z",
            "user": "test_user",
        }

        success = await repository.store_audit_entry("model_lifecycle", entry)

        assert success is True
        assert len(repository._audit_entries) == 1

        stored_entry = repository._audit_entries[0]
        assert stored_entry["category"] == "model_lifecycle"
        assert stored_entry["entry"] == entry

    @pytest.mark.asyncio
    async def test_multiple_audit_entries(self, repository):
        """Test storing multiple audit entries."""
        entries = [
            {"action": "model_created", "model_id": "model_1", "timestamp": "2024-01-01T00:00:00Z"},
            {"action": "model_updated", "model_id": "model_1", "timestamp": "2024-01-01T01:00:00Z"},
        ]

        for entry in entries:
            await repository.store_audit_entry("model_lifecycle", entry)

        assert len(repository._audit_entries) == 2
        assert repository._audit_entries[0]["entry"]["action"] == "model_created"
        assert repository._audit_entries[1]["entry"]["action"] == "model_updated"


class TestIMLRepository:
    """Test ML repository interface."""

    def test_interface_methods(self):
        """Test that interface defines required methods."""
        # Check that all expected abstract methods are defined
        expected_methods = [
            "store_model_metadata",
            "get_model_by_id",
            "get_models_by_name_and_type",
            "find_models",
            "get_all_models",
            "update_model_metadata",
            "delete_model",
            "store_prediction",
            "get_predictions", 
            "store_training_job",
            "get_training_job",
            "update_training_progress",
        ]

        for method_name in expected_methods:
            assert hasattr(IMLRepository, method_name)
            method = getattr(IMLRepository, method_name)
            assert getattr(method, "__isabstractmethod__", False)

    def test_concrete_implementation(self):
        """Test that concrete class implements interface."""
        assert issubclass(MLRepository, IMLRepository)

        # Should be able to instantiate
        repo = MLRepository()
        assert isinstance(repo, IMLRepository)


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def repository(self):
        """Create repository for error testing."""
        repo = MLRepository()
        # Mock the data service dependency
        mock_data_service = AsyncMock()
        mock_data_service.store_ml_model_metadata = AsyncMock()
        mock_data_service.get_ml_model_by_id = AsyncMock()
        mock_data_service.update_ml_model_metadata = AsyncMock()
        mock_data_service.delete_ml_model = AsyncMock()
        repo.data_service = mock_data_service
        return repo

    @pytest.mark.asyncio
    async def test_store_empty_metadata(self, repository):
        """Test storing empty metadata."""
        # The repository's validation will catch empty dict and raise RepositoryError
        from src.core.exceptions import RepositoryError
        
        with pytest.raises(RepositoryError) as exc_info:
            await repository.store_model_metadata({})
            
        assert "Empty entity provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_store_none_metadata(self, repository):
        """Test storing None metadata."""
        # The repository's validation will catch None and raise RepositoryError  
        from src.core.exceptions import RepositoryError
        
        with pytest.raises(RepositoryError) as exc_info:
            await repository.store_model_metadata(None)
            
        assert "Empty entity provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_model_none_id(self, repository):
        """Test getting model with None ID."""
        repository.data_service.get_ml_model_by_id.return_value = None
        
        result = await repository.get_model_by_id(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_model_empty_id(self, repository):
        """Test getting model with empty ID."""
        repository.data_service.get_ml_model_by_id.return_value = None
        
        result = await repository.get_model_by_id("")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_with_none_metadata(self, repository):
        """Test updating with None metadata."""
        repository.data_service.update_ml_model_metadata.return_value = False
        
        # Should handle gracefully
        success = await repository.update_model_metadata("test", None)
        assert success is False

    @pytest.mark.asyncio
    async def test_delete_none_id(self, repository):
        """Test deleting with None ID."""
        repository.data_service.delete_ml_model.return_value = False
        
        success = await repository.delete_model(None)
        assert success is False

    @pytest.mark.asyncio
    async def test_store_audit_entry_none_values(self, repository):
        """Test storing audit entry with None values."""
        # Should handle gracefully
        success = await repository.store_audit_entry(None, None)
        assert success is True  # Current implementation accepts None
