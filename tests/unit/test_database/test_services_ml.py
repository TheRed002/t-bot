"""Tests for database ML service."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from src.core.exceptions import ServiceError, ValidationError
from src.database.services.ml_service import MLService


class TestMLService:
    """Test the MLService class."""

    @pytest.fixture
    def mock_prediction_repo(self):
        """Create a mock prediction repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_model_repo(self):
        """Create a mock model repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_training_repo(self):
        """Create a mock training repository."""
        return AsyncMock()

    @pytest.fixture
    def ml_service(self, mock_prediction_repo, mock_model_repo, mock_training_repo):
        """Create a MLService instance with mocked dependencies."""
        return MLService(
            prediction_repo=mock_prediction_repo,
            model_repo=mock_model_repo,
            training_repo=mock_training_repo
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock model object."""
        model = Mock()
        model.model_name = "test_model"
        model.model_type = "prediction"
        model.version = 1
        model.parameters = {"param1": "value1"}
        return model

    @pytest.fixture
    def mock_training_job(self):
        """Create a mock training job object."""
        job = Mock()
        job.model_name = "test_model"
        job.status = "completed"
        job.completed_at = datetime.now()
        return job

    @pytest.fixture
    def mock_prediction(self):
        """Create a mock prediction object."""
        prediction = Mock()
        prediction.model_name = "test_model"
        prediction.symbol = "BTC-USD"
        prediction.predicted_value = 50000.0
        prediction.confidence_score = 0.85
        prediction.timestamp = datetime.now()
        return prediction

    async def test_init(self, mock_prediction_repo, mock_model_repo, mock_training_repo):
        """Test MLService initialization."""
        service = MLService(
            prediction_repo=mock_prediction_repo,
            model_repo=mock_model_repo,
            training_repo=mock_training_repo
        )
        assert service.prediction_repo == mock_prediction_repo
        assert service.model_repo == mock_model_repo
        assert service.training_repo == mock_training_repo
        assert service.name == "MLService"

    async def test_get_model_performance_summary_success(
        self, ml_service, mock_prediction_repo, mock_model_repo, mock_training_repo, mock_model, mock_training_job
    ):
        """Test getting model performance summary successfully."""
        # Arrange
        accuracy_data = {"overall_accuracy": 0.85, "precision": 0.82}
        mock_prediction_repo.get_prediction_accuracy.return_value = accuracy_data
        mock_model_repo.get_latest_model.return_value = mock_model
        mock_training_repo.get_job_by_model.return_value = [mock_training_job]

        # Act
        result = await ml_service.get_model_performance_summary("test_model", days=30)

        # Assert
        assert result["model_name"] == "test_model"
        assert result["accuracy_metrics"] == accuracy_data
        assert result["latest_version"] == 1
        assert result["model_parameters"] == {"param1": "value1"}
        assert result["recent_training_jobs"] == 1
        assert result["last_training"] == mock_training_job.completed_at

        mock_prediction_repo.get_prediction_accuracy.assert_called_once_with("test_model", days=30)
        mock_model_repo.get_latest_model.assert_called_once_with("test_model", "prediction")
        mock_training_repo.get_job_by_model.assert_called_once_with("test_model", status="completed")

    async def test_get_model_performance_summary_empty_model_name(self, ml_service):
        """Test getting model performance summary with empty model name."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Model name is required"):
            await ml_service.get_model_performance_summary("")

    async def test_get_model_performance_summary_whitespace_model_name(self, ml_service):
        """Test getting model performance summary with whitespace model name."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Model name is required"):
            await ml_service.get_model_performance_summary("   ")

    async def test_get_model_performance_summary_invalid_days(self, ml_service):
        """Test getting model performance summary with invalid days."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Days must be positive"):
            await ml_service.get_model_performance_summary("test_model", days=0)

    async def test_get_model_performance_summary_negative_days(self, ml_service):
        """Test getting model performance summary with negative days."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Days must be positive"):
            await ml_service.get_model_performance_summary("test_model", days=-1)

    async def test_get_model_performance_summary_no_model(
        self, ml_service, mock_prediction_repo, mock_model_repo, mock_training_repo
    ):
        """Test getting model performance summary when no model exists."""
        # Arrange
        accuracy_data = {"overall_accuracy": 0.85}
        mock_prediction_repo.get_prediction_accuracy.return_value = accuracy_data
        mock_model_repo.get_latest_model.return_value = None
        mock_training_repo.get_job_by_model.return_value = []

        # Act
        result = await ml_service.get_model_performance_summary("test_model")

        # Assert
        assert result["model_name"] == "test_model"
        assert result["accuracy_metrics"] == accuracy_data
        assert result["latest_version"] is None
        assert result["model_parameters"] == {}
        assert result["recent_training_jobs"] == 0
        assert result["last_training"] is None

    async def test_get_model_performance_summary_repository_error(
        self, ml_service, mock_prediction_repo
    ):
        """Test getting model performance summary with repository error."""
        # Arrange
        mock_prediction_repo.get_prediction_accuracy.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Model performance summary failed"):
            await ml_service.get_model_performance_summary("test_model")

    async def test_validate_model_deployment_success(
        self, ml_service, mock_model_repo, mock_training_repo, mock_prediction_repo, mock_model, mock_training_job
    ):
        """Test validating model deployment successfully."""
        # Arrange
        mock_model_repo.get_by_version.return_value = mock_model
        mock_training_repo.get_job_by_model.return_value = [mock_training_job]
        mock_prediction_repo.get_prediction_accuracy.return_value = {"overall_accuracy": 0.85}

        # Act
        result = await ml_service.validate_model_deployment("test_model", 1)

        # Assert
        assert result is True
        mock_model_repo.get_by_version.assert_called_once_with("test_model", 1)
        mock_training_repo.get_job_by_model.assert_called_once_with("test_model", status="completed")
        mock_prediction_repo.get_prediction_accuracy.assert_called_once_with("test_model", days=7)

    async def test_validate_model_deployment_model_not_found(
        self, ml_service, mock_model_repo
    ):
        """Test validating model deployment when model not found."""
        # Arrange
        mock_model_repo.get_by_version.return_value = None

        # Act & Assert
        with pytest.raises(ValidationError, match="Model test_model version 1 not found"):
            await ml_service.validate_model_deployment("test_model", 1)

    async def test_validate_model_deployment_no_training_jobs(
        self, ml_service, mock_model_repo, mock_training_repo, mock_model
    ):
        """Test validating model deployment with no training jobs."""
        # Arrange
        mock_model_repo.get_by_version.return_value = mock_model
        mock_training_repo.get_job_by_model.return_value = []

        # Act
        result = await ml_service.validate_model_deployment("test_model", 1)

        # Assert
        assert result is False

    async def test_validate_model_deployment_low_accuracy(
        self, ml_service, mock_model_repo, mock_training_repo, mock_prediction_repo, mock_model, mock_training_job
    ):
        """Test validating model deployment with low accuracy."""
        # Arrange
        mock_model_repo.get_by_version.return_value = mock_model
        mock_training_repo.get_job_by_model.return_value = [mock_training_job]
        mock_prediction_repo.get_prediction_accuracy.return_value = {"overall_accuracy": 0.6}  # Below threshold

        # Act
        result = await ml_service.validate_model_deployment("test_model", 1)

        # Assert
        assert result is False

    async def test_validate_model_deployment_no_accuracy_data(
        self, ml_service, mock_model_repo, mock_training_repo, mock_prediction_repo, mock_model, mock_training_job
    ):
        """Test validating model deployment with no accuracy data."""
        # Arrange
        mock_model_repo.get_by_version.return_value = mock_model
        mock_training_repo.get_job_by_model.return_value = [mock_training_job]
        mock_prediction_repo.get_prediction_accuracy.return_value = None

        # Act
        result = await ml_service.validate_model_deployment("test_model", 1)

        # Assert
        assert result is True  # No accuracy data is not a blocker

    async def test_validate_model_deployment_repository_error(
        self, ml_service, mock_model_repo
    ):
        """Test validating model deployment with repository error."""
        # Arrange
        mock_model_repo.get_by_version.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Model validation failed"):
            await ml_service.validate_model_deployment("test_model", 1)

    async def test_get_model_recommendations_success(
        self, ml_service, mock_model_repo, mock_prediction_repo, mock_model, mock_prediction
    ):
        """Test getting model recommendations successfully."""
        # Arrange
        mock_model_repo.get_active_models.return_value = [mock_model]
        mock_prediction_repo.get_by_model_and_symbol.return_value = [mock_prediction]

        # Act
        result = await ml_service.get_model_recommendations("BTC-USD", limit=5)

        # Assert
        assert len(result) == 1
        recommendation = result[0]
        assert recommendation["model_name"] == "test_model"
        assert recommendation["model_type"] == "prediction"
        assert recommendation["latest_prediction"] == 50000.0
        assert recommendation["confidence"] == 0.85
        assert recommendation["symbol"] == "BTC-USD"

        mock_model_repo.get_active_models.assert_called_once()
        mock_prediction_repo.get_by_model_and_symbol.assert_called_once_with(
            "test_model", "BTC-USD", limit=10
        )

    async def test_get_model_recommendations_empty_symbol(self, ml_service):
        """Test getting model recommendations with empty symbol."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Symbol is required"):
            await ml_service.get_model_recommendations("")

    async def test_get_model_recommendations_whitespace_symbol(self, ml_service):
        """Test getting model recommendations with whitespace symbol."""
        # Act & Assert
        with pytest.raises(ValidationError, match="Symbol is required"):
            await ml_service.get_model_recommendations("   ")

    async def test_get_model_recommendations_no_active_models(
        self, ml_service, mock_model_repo
    ):
        """Test getting model recommendations with no active models."""
        # Arrange
        mock_model_repo.get_active_models.return_value = []

        # Act
        result = await ml_service.get_model_recommendations("BTC-USD")

        # Assert
        assert result == []

    async def test_get_model_recommendations_no_predictions(
        self, ml_service, mock_model_repo, mock_prediction_repo, mock_model
    ):
        """Test getting model recommendations with no predictions."""
        # Arrange
        mock_model_repo.get_active_models.return_value = [mock_model]
        mock_prediction_repo.get_by_model_and_symbol.return_value = []

        # Act
        result = await ml_service.get_model_recommendations("BTC-USD")

        # Assert
        assert result == []

    async def test_get_model_recommendations_multiple_models_sorted(
        self, ml_service, mock_model_repo, mock_prediction_repo
    ):
        """Test getting model recommendations with multiple models sorted by confidence."""
        # Arrange
        model1 = Mock()
        model1.model_name = "model1"
        model1.model_type = "prediction"
        
        model2 = Mock()
        model2.model_name = "model2"
        model2.model_type = "classification"

        prediction1 = Mock()
        prediction1.model_name = "model1"
        prediction1.predicted_value = 50000.0
        prediction1.confidence_score = 0.7
        prediction1.timestamp = datetime.now()

        prediction2 = Mock()
        prediction2.model_name = "model2"
        prediction2.predicted_value = 55000.0
        prediction2.confidence_score = 0.9
        prediction2.timestamp = datetime.now()

        mock_model_repo.get_active_models.return_value = [model1, model2]
        
        def mock_get_by_model_and_symbol(model_name, symbol, limit):
            if model_name == "model1":
                return [prediction1]
            elif model_name == "model2":
                return [prediction2]
            return []
        
        mock_prediction_repo.get_by_model_and_symbol.side_effect = mock_get_by_model_and_symbol

        # Act
        result = await ml_service.get_model_recommendations("BTC-USD", limit=5)

        # Assert
        assert len(result) == 2
        # Should be sorted by confidence (descending)
        assert result[0]["model_name"] == "model2"  # Higher confidence (0.9)
        assert result[0]["confidence"] == 0.9
        assert result[1]["model_name"] == "model1"  # Lower confidence (0.7)
        assert result[1]["confidence"] == 0.7

    async def test_get_model_recommendations_limit_applied(
        self, ml_service, mock_model_repo, mock_prediction_repo
    ):
        """Test getting model recommendations with limit applied."""
        # Arrange
        models = []
        for i in range(5):
            model = Mock()
            model.model_name = f"model{i}"
            model.model_type = "prediction"
            models.append(model)

        mock_model_repo.get_active_models.return_value = models
        
        def mock_get_by_model_and_symbol(model_name, symbol, limit):
            prediction = Mock()
            prediction.predicted_value = 50000.0
            prediction.confidence_score = 0.8
            prediction.timestamp = datetime.now()
            return [prediction]
        
        mock_prediction_repo.get_by_model_and_symbol.side_effect = mock_get_by_model_and_symbol

        # Act
        result = await ml_service.get_model_recommendations("BTC-USD", limit=3)

        # Assert
        assert len(result) == 3  # Limited to 3 despite 5 models

    async def test_get_model_recommendations_repository_error(
        self, ml_service, mock_model_repo
    ):
        """Test getting model recommendations with repository error."""
        # Arrange
        mock_model_repo.get_active_models.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Model recommendations failed"):
            await ml_service.get_model_recommendations("BTC-USD")


class TestMLServiceErrorHandling:
    """Test error handling in MLService."""

    @pytest.fixture
    def ml_service(self):
        """Create a MLService instance with mock dependencies."""
        return MLService(
            prediction_repo=AsyncMock(),
            model_repo=AsyncMock(),
            training_repo=AsyncMock()
        )

    async def test_get_model_performance_summary_prediction_repo_error(self, ml_service):
        """Test getting model performance summary with prediction repository error."""
        # Arrange
        ml_service.prediction_repo.get_prediction_accuracy.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError):
            await ml_service.get_model_performance_summary("test_model")

    async def test_get_model_performance_summary_model_repo_error(self, ml_service):
        """Test getting model performance summary with model repository error."""
        # Arrange
        ml_service.prediction_repo.get_prediction_accuracy.return_value = {}
        ml_service.model_repo.get_latest_model.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError):
            await ml_service.get_model_performance_summary("test_model")

    async def test_get_model_performance_summary_training_repo_error(self, ml_service):
        """Test getting model performance summary with training repository error."""
        # Arrange
        ml_service.prediction_repo.get_prediction_accuracy.return_value = {}
        ml_service.model_repo.get_latest_model.return_value = None
        ml_service.training_repo.get_job_by_model.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError):
            await ml_service.get_model_performance_summary("test_model")

    async def test_validate_model_deployment_model_repo_error(self, ml_service):
        """Test validating model deployment with model repository error."""
        # Arrange
        ml_service.model_repo.get_by_version.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError):
            await ml_service.validate_model_deployment("test_model", 1)

    async def test_validate_model_deployment_training_repo_error(self, ml_service):
        """Test validating model deployment with training repository error."""
        # Arrange
        mock_model = Mock()
        ml_service.model_repo.get_by_version.return_value = mock_model
        ml_service.training_repo.get_job_by_model.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError):
            await ml_service.validate_model_deployment("test_model", 1)

    async def test_validate_model_deployment_prediction_repo_error(self, ml_service):
        """Test validating model deployment with prediction repository error."""
        # Arrange
        mock_model = Mock()
        mock_job = Mock()
        ml_service.model_repo.get_by_version.return_value = mock_model
        ml_service.training_repo.get_job_by_model.return_value = [mock_job]
        ml_service.prediction_repo.get_prediction_accuracy.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError):
            await ml_service.validate_model_deployment("test_model", 1)

    async def test_get_model_recommendations_model_repo_error(self, ml_service):
        """Test getting model recommendations with model repository error."""
        # Arrange
        ml_service.model_repo.get_active_models.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError):
            await ml_service.get_model_recommendations("BTC-USD")

    async def test_get_model_recommendations_prediction_repo_error(self, ml_service):
        """Test getting model recommendations with prediction repository error."""
        # Arrange
        mock_model = Mock()
        mock_model.model_name = "test_model"
        ml_service.model_repo.get_active_models.return_value = [mock_model]
        ml_service.prediction_repo.get_by_model_and_symbol.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError):
            await ml_service.get_model_recommendations("BTC-USD")