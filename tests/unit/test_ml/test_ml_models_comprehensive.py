"""
Comprehensive unit tests for ML models and training components.

Tests model loading, inference, training, validation, and drift detection
for financial machine learning operations.
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, List, Any

from src.ml.models.base_model import BaseModel
from src.ml.models.price_predictor import PricePredictor
from src.ml.models.direction_classifier import DirectionClassifier
from src.ml.models.volatility_forecaster import VolatilityForecaster
from src.ml.models.regime_detector import RegimeDetector
from src.ml.training.trainer import TrainingService as ModelTrainer
from src.ml.validation.model_validator import ModelValidationService as ModelValidator
from src.ml.validation.drift_detector import DriftDetectionService as DriftDetector
from src.ml.inference import InferenceEngine
from src.ml.model_manager import ModelManagerService as ModelManager

from src.core.config import Config
from src.core.exceptions import ModelError, ModelLoadError, ModelInferenceError, ModelDriftError
from src.core.types import MarketData, Signal, SignalDirection


@pytest.fixture
def mock_config():
    """Mock configuration for ML tests."""
    config = Mock(spec=Config)
    config.ml = Mock()
    config.ml.model_path = "/tmp/models"
    config.ml.batch_size = 32
    config.ml.epochs = 10
    config.ml.learning_rate = 0.001
    config.ml.validation_split = 0.2
    config.ml.early_stopping_patience = 5
    config.ml.drift_threshold = 0.1
    config.ml.retrain_threshold = 0.2
    config.ml.max_inference_time_ms = 100
    return config


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    timestamps = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    return pd.DataFrame({
        'timestamp': timestamps,
        'symbol': ['BTC/USDT'] * 100,
        'open': np.random.uniform(50000, 52000, 100),
        'high': np.random.uniform(51000, 53000, 100),
        'low': np.random.uniform(49000, 51000, 100),
        'close': np.random.uniform(50000, 52000, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })


@pytest.fixture
def sample_features():
    """Sample feature matrix for testing."""
    np.random.seed(42)
    return np.random.rand(100, 20)  # 100 samples, 20 features


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    np.random.seed(42)
    return np.random.randint(0, 3, 100)  # 100 labels, 3 classes


class ConcreteTestModel(BaseModel):
    """Concrete test implementation of BaseModel."""
    
    def _get_model_type(self) -> str:
        return "test_model"
    
    def _create_model(self, **kwargs):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**kwargs)
    
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
    
    def _validate_targets(self, y: pd.Series) -> pd.Series:
        return y
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        from sklearn.metrics import mean_squared_error
        return {"mse": mean_squared_error(y_true, y_pred)}


class TestBaseModel:
    """Test BaseModel functionality."""

    def test_base_model_initialization(self, mock_config):
        """Test BaseModel initialization."""
        model = ConcreteTestModel(mock_config, "test_model")
        
        assert model.config == mock_config
        assert model.model_name == "test_model"
        assert model.version == "1.0.0"
        assert model.is_trained is False
        assert isinstance(model.metadata, dict)

    def test_base_model_abstract_enforcement(self, mock_config):
        """Test that BaseModel cannot be instantiated directly."""
        # BaseModel is abstract and should not be instantiable
        with pytest.raises(TypeError):
            BaseModel(mock_config, "test_model")

    def test_model_metadata_management(self, mock_config):
        """Test model metadata management."""
        model = ConcreteTestModel(mock_config, "test_model")
        
        # Test metadata is initialized
        assert isinstance(model.metadata, dict)
        assert "created_at" in model.metadata
        assert "updated_at" in model.metadata
        
        # Test updating metadata
        model.metadata["param1"] = "value1"
        model.metadata["param2"] = 42
        
        assert model.metadata["param1"] == "value1"
        assert model.metadata["param2"] == 42
        
        # Test getting model info
        info = model.get_model_info()
        assert info["model_name"] == "test_model"
        assert info["model_type"] == "test_model"
        assert info["is_trained"] is False

    def test_model_serialization_interface(self, mock_config, tmp_path):
        """Test model serialization interface."""
        model = ConcreteTestModel(mock_config, "test_model")
        
        # Test save/load functionality
        save_path = tmp_path / "test_model.pkl"
        
        # Should be able to save even when not trained
        saved_path = model.save(save_path)
        assert saved_path.exists()
        
        # Test loading
        loaded_model = ConcreteTestModel.load(saved_path, mock_config)
        assert loaded_model.model_name == "test_model"
        assert loaded_model.model_type == "test_model"
        assert loaded_model.is_trained is False


class TestPricePredictor:
    """Test PricePredictor model."""

    def test_price_predictor_initialization(self, mock_config):
        """Test PricePredictor initialization."""
        predictor = PricePredictor(mock_config, "price_predictor_v1")
        
        assert predictor.model_name == "price_predictor_v1"
        assert predictor.prediction_horizon is not None
        assert predictor.feature_columns is not None

    @patch('src.ml.models.price_predictor.RandomForestRegressor')
    def test_price_predictor_training(self, mock_rf, mock_config, sample_features, sample_labels):
        """Test PricePredictor training."""
        predictor = PricePredictor(mock_config, "test_predictor")
        
        # Mock the regressor
        mock_model = Mock()
        mock_rf.return_value = mock_model
        
        # Convert labels to continuous values for regression
        target_prices = sample_labels.astype(float) * 1000  # Convert to price-like values
        
        result = predictor.train(sample_features, target_prices)
        
        assert result is True
        assert predictor.is_trained is True
        mock_model.fit.assert_called_once()

    @patch('src.ml.models.price_predictor.RandomForestRegressor')
    def test_price_predictor_prediction(self, mock_rf, mock_config, sample_features):
        """Test PricePredictor prediction."""
        predictor = PricePredictor(mock_config, "test_predictor")
        
        # Mock the trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([50000.0, 51000.0, 49000.0])
        mock_rf.return_value = mock_model
        predictor.model = mock_model
        predictor.is_trained = True
        
        predictions = predictor.predict(sample_features[:3])
        
        assert len(predictions) == 3
        assert all(isinstance(pred, (float, np.floating)) for pred in predictions)
        mock_model.predict.assert_called_once()

    def test_price_predictor_prediction_untrained_model(self, mock_config, sample_features):
        """Test PricePredictor prediction with untrained model."""
        predictor = PricePredictor(mock_config, "test_predictor")
        
        with pytest.raises(ModelError, match="not trained"):
            predictor.predict(sample_features)

    @patch('src.ml.models.price_predictor.RandomForestRegressor')
    def test_price_predictor_validation(self, mock_rf, mock_config, sample_features):
        """Test PricePredictor validation."""
        predictor = PricePredictor(mock_config, "test_predictor")
        
        # Mock the trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([50000.0, 51000.0, 49000.0])
        mock_model.score.return_value = 0.85
        mock_rf.return_value = mock_model
        predictor.model = mock_model
        predictor.is_trained = True
        
        target_prices = np.array([50100.0, 50900.0, 49100.0])
        metrics = predictor.validate(sample_features[:3], target_prices)
        
        assert 'r2_score' in metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert metrics['r2_score'] == 0.85


class TestDirectionClassifier:
    """Test DirectionClassifier model."""

    def test_direction_classifier_initialization(self, mock_config):
        """Test DirectionClassifier initialization."""
        classifier = DirectionClassifier(mock_config, "direction_classifier_v1")
        
        assert classifier.model_name == "direction_classifier_v1"
        assert classifier.classes is not None

    @patch('src.ml.models.direction_classifier.RandomForestClassifier')
    def test_direction_classifier_training(self, mock_rf, mock_config, sample_features, sample_labels):
        """Test DirectionClassifier training."""
        classifier = DirectionClassifier(mock_config, "test_classifier")
        
        # Mock the classifier
        mock_model = Mock()
        mock_rf.return_value = mock_model
        
        result = classifier.train(sample_features, sample_labels)
        
        assert result is True
        assert classifier.is_trained is True
        mock_model.fit.assert_called_once()

    @patch('src.ml.models.direction_classifier.RandomForestClassifier')
    def test_direction_classifier_prediction(self, mock_rf, mock_config, sample_features):
        """Test DirectionClassifier prediction."""
        classifier = DirectionClassifier(mock_config, "test_classifier")
        
        # Mock the trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2])  # UP, DOWN, SIDEWAYS
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1], 
            [0.3, 0.3, 0.4]
        ])
        mock_rf.return_value = mock_model
        classifier.model = mock_model
        classifier.is_trained = True
        
        predictions = classifier.predict(sample_features[:3])
        
        assert len(predictions) == 3
        assert all(pred in [0, 1, 2] for pred in predictions)
        mock_model.predict.assert_called_once()

    @patch('src.ml.models.direction_classifier.RandomForestClassifier')
    def test_direction_classifier_prediction_with_probabilities(self, mock_rf, mock_config, sample_features):
        """Test DirectionClassifier prediction with probabilities."""
        classifier = DirectionClassifier(mock_config, "test_classifier")
        
        # Mock the trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1], 
            [0.3, 0.3, 0.4]
        ])
        mock_rf.return_value = mock_model
        classifier.model = mock_model
        classifier.is_trained = True
        
        predictions, probabilities = classifier.predict_with_confidence(sample_features[:3])
        
        assert len(predictions) == 3
        assert len(probabilities) == 3
        assert all(len(prob) == 3 for prob in probabilities)  # 3 classes
        assert all(np.sum(prob) == pytest.approx(1.0) for prob in probabilities)  # Probabilities sum to 1


class TestVolatilityForecaster:
    """Test VolatilityForecaster model."""

    def test_volatility_forecaster_initialization(self, mock_config):
        """Test VolatilityForecaster initialization."""
        forecaster = VolatilityForecaster(mock_config, "volatility_forecaster_v1")
        
        assert forecaster.model_name == "volatility_forecaster_v1"
        assert forecaster.lookback_period is not None

    @patch('src.ml.models.volatility_forecaster.LGBMRegressor')
    def test_volatility_forecaster_training(self, mock_lgbm, mock_config, sample_features):
        """Test VolatilityForecaster training."""
        forecaster = VolatilityForecaster(mock_config, "test_forecaster")
        
        # Mock the regressor
        mock_model = Mock()
        mock_lgbm.return_value = mock_model
        
        # Create volatility-like targets (always positive)
        volatility_targets = np.random.uniform(0.01, 0.1, len(sample_features))
        
        result = forecaster.train(sample_features, volatility_targets)
        
        assert result is True
        assert forecaster.is_trained is True
        mock_model.fit.assert_called_once()

    @patch('src.ml.models.volatility_forecaster.LGBMRegressor')
    def test_volatility_forecaster_prediction(self, mock_lgbm, mock_config, sample_features):
        """Test VolatilityForecaster prediction."""
        forecaster = VolatilityForecaster(mock_config, "test_forecaster")
        
        # Mock the trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.02, 0.035, 0.015])  # Volatility values
        mock_lgbm.return_value = mock_model
        forecaster.model = mock_model
        forecaster.is_trained = True
        
        predictions = forecaster.predict(sample_features[:3])
        
        assert len(predictions) == 3
        assert all(pred > 0 for pred in predictions)  # Volatility should be positive
        mock_model.predict.assert_called_once()


class TestRegimeDetector:
    """Test RegimeDetector model."""

    def test_regime_detector_initialization(self, mock_config):
        """Test RegimeDetector initialization."""
        detector = RegimeDetector(mock_config, "regime_detector_v1")
        
        assert detector.model_name == "regime_detector_v1"
        assert detector.regime_types is not None

    @patch('src.ml.models.regime_detector.GaussianMixture')
    def test_regime_detector_training(self, mock_gmm, mock_config, sample_features):
        """Test RegimeDetector training."""
        detector = RegimeDetector(mock_config, "test_detector")
        
        # Mock the GMM
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2, 0, 1])
        mock_gmm.return_value = mock_model
        
        # Regime detection is unsupervised, so no labels needed
        result = detector.train(sample_features, None)
        
        assert result is True
        assert detector.is_trained is True
        mock_model.fit.assert_called_once()

    @patch('src.ml.models.regime_detector.GaussianMixture')
    def test_regime_detector_prediction(self, mock_gmm, mock_config, sample_features):
        """Test RegimeDetector prediction."""
        detector = RegimeDetector(mock_config, "test_detector")
        
        # Mock the trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2])  # Bull, Bear, Sideways
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1], 
            [0.3, 0.2, 0.5]
        ])
        mock_gmm.return_value = mock_model
        detector.model = mock_model
        detector.is_trained = True
        
        regimes = detector.predict(sample_features[:3])
        
        assert len(regimes) == 3
        assert all(regime in [0, 1, 2] for regime in regimes)
        mock_model.predict.assert_called_once()

    def test_regime_detector_regime_interpretation(self, mock_config):
        """Test regime interpretation functionality."""
        detector = RegimeDetector(mock_config, "test_detector")
        
        regime_names = detector.get_regime_names()
        assert len(regime_names) > 0
        assert all(isinstance(name, str) for name in regime_names)
        
        # Test regime interpretation
        regime_id = 0
        regime_name = detector.interpret_regime(regime_id)
        assert isinstance(regime_name, str)
        assert len(regime_name) > 0


class TestModelTrainer:
    """Test ModelTrainer functionality."""

    def test_model_trainer_initialization(self, mock_config):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(mock_config)
        
        assert trainer.config == mock_config
        assert trainer.training_history == []

    def test_train_model_with_validation_split(self, mock_config, sample_features, sample_labels):
        """Test model training with validation split."""
        trainer = ModelTrainer(mock_config)
        model = Mock()
        model.train = Mock(return_value=True)
        model.validate = Mock(return_value={'accuracy': 0.85})
        
        result = trainer.train_model(
            model, 
            sample_features, 
            sample_labels,
            validation_split=0.2
        )
        
        assert result['success'] is True
        assert 'validation_metrics' in result
        assert result['validation_metrics']['accuracy'] == 0.85
        model.train.assert_called_once()

    def test_train_model_with_early_stopping(self, mock_config, sample_features, sample_labels):
        """Test model training with early stopping."""
        trainer = ModelTrainer(mock_config)
        model = Mock()
        
        # Simulate decreasing validation loss for first few epochs, then increasing
        validation_losses = [0.5, 0.45, 0.42, 0.44, 0.47, 0.50]
        model.train = Mock(return_value=True)
        model.validate = Mock(side_effect=[
            {'loss': loss} for loss in validation_losses
        ])
        
        result = trainer.train_model_with_early_stopping(
            model,
            sample_features,
            sample_labels,
            patience=3,
            max_epochs=10
        )
        
        assert result['success'] is True
        assert result['stopped_early'] is True
        assert result['best_epoch'] == 2  # Epoch with lowest loss

    def test_cross_validation(self, mock_config, sample_features, sample_labels):
        """Test cross-validation training."""
        trainer = ModelTrainer(mock_config)
        model_factory = Mock()
        
        # Mock model instances for each fold
        mock_models = []
        for i in range(5):  # 5-fold CV
            mock_model = Mock()
            mock_model.train = Mock(return_value=True)
            mock_model.validate = Mock(return_value={'accuracy': 0.8 + i * 0.02})
            mock_models.append(mock_model)
        
        model_factory.side_effect = mock_models
        
        cv_results = trainer.cross_validate(
            model_factory,
            sample_features,
            sample_labels,
            cv_folds=5
        )
        
        assert len(cv_results) == 5
        assert all(result['success'] for result in cv_results)
        assert all('validation_metrics' in result for result in cv_results)


class TestModelValidator:
    """Test ModelValidator functionality."""

    def test_model_validator_initialization(self, mock_config):
        """Test ModelValidator initialization."""
        validator = ModelValidator(mock_config)
        
        assert validator.config == mock_config
        assert validator.validation_thresholds is not None

    def test_validate_classification_model(self, mock_config, sample_features, sample_labels):
        """Test validation of classification model."""
        validator = ModelValidator(mock_config)
        
        model = Mock()
        model.predict = Mock(return_value=sample_labels)
        model.predict_proba = Mock(return_value=np.random.rand(len(sample_labels), 3))
        
        metrics = validator.validate_classification_model(
            model,
            sample_features,
            sample_labels
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics

    def test_validate_regression_model(self, mock_config, sample_features):
        """Test validation of regression model."""
        validator = ModelValidator(mock_config)
        
        # Create regression targets
        regression_targets = np.random.rand(len(sample_features)) * 1000
        predictions = regression_targets + np.random.normal(0, 50, len(regression_targets))
        
        model = Mock()
        model.predict = Mock(return_value=predictions)
        
        metrics = validator.validate_regression_model(
            model,
            sample_features,
            regression_targets
        )
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics

    def test_performance_threshold_validation(self, mock_config):
        """Test performance threshold validation."""
        validator = ModelValidator(mock_config)
        
        # Test metrics that meet threshold
        good_metrics = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77
        }
        
        assert validator.meets_performance_threshold(good_metrics) is True
        
        # Test metrics that don't meet threshold
        bad_metrics = {
            'accuracy': 0.60,
            'precision': 0.55,
            'recall': 0.50,
            'f1_score': 0.52
        }
        
        assert validator.meets_performance_threshold(bad_metrics) is False

    def test_model_stability_validation(self, mock_config, sample_features, sample_labels):
        """Test model stability validation across multiple runs."""
        validator = ModelValidator(mock_config)
        
        def mock_model_factory():
            model = Mock()
            # Add some variation to simulate real model training
            base_accuracy = 0.80
            variation = np.random.normal(0, 0.02)  # Small random variation
            model.validate = Mock(return_value={'accuracy': base_accuracy + variation})
            return model
        
        stability_metrics = validator.validate_model_stability(
            mock_model_factory,
            sample_features,
            sample_labels,
            num_runs=5
        )
        
        assert 'mean_performance' in stability_metrics
        assert 'std_performance' in stability_metrics
        assert 'min_performance' in stability_metrics
        assert 'max_performance' in stability_metrics
        assert stability_metrics['coefficient_of_variation'] is not None


class TestDriftDetector:
    """Test DriftDetector functionality."""

    def test_drift_detector_initialization(self, mock_config):
        """Test DriftDetector initialization."""
        detector = DriftDetector(mock_config)
        
        assert detector.config == mock_config
        assert detector.reference_data is None
        assert detector.drift_threshold == mock_config.ml.drift_threshold

    def test_set_reference_data(self, mock_config, sample_features):
        """Test setting reference data for drift detection."""
        detector = DriftDetector(mock_config)
        
        detector.set_reference_data(sample_features)
        
        assert detector.reference_data is not None
        assert detector.reference_stats is not None
        assert len(detector.reference_stats) > 0

    def test_detect_data_drift_no_drift(self, mock_config, sample_features):
        """Test drift detection when no drift is present."""
        detector = DriftDetector(mock_config)
        detector.set_reference_data(sample_features)
        
        # Use same data (no drift)
        drift_result = detector.detect_data_drift(sample_features)
        
        assert drift_result['drift_detected'] is False
        assert drift_result['drift_score'] < detector.drift_threshold

    def test_detect_data_drift_with_drift(self, mock_config, sample_features):
        """Test drift detection when drift is present."""
        detector = DriftDetector(mock_config)
        detector.set_reference_data(sample_features)
        
        # Create drifted data by adding bias
        drifted_data = sample_features + 2.0  # Significant shift
        
        drift_result = detector.detect_data_drift(drifted_data)
        
        assert drift_result['drift_detected'] is True
        assert drift_result['drift_score'] > detector.drift_threshold

    def test_detect_model_drift(self, mock_config, sample_features, sample_labels):
        """Test model performance drift detection."""
        detector = DriftDetector(mock_config)
        
        # Set baseline performance
        baseline_metrics = {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75}
        detector.set_baseline_performance(baseline_metrics)
        
        # Test with degraded performance
        current_metrics = {'accuracy': 0.70, 'precision': 0.68, 'recall': 0.65}
        
        drift_result = detector.detect_model_drift(current_metrics)
        
        assert drift_result['drift_detected'] is True
        assert drift_result['performance_degradation'] > 0

    def test_adaptive_threshold_adjustment(self, mock_config):
        """Test adaptive threshold adjustment based on historical drift."""
        detector = DriftDetector(mock_config)
        
        # Simulate historical drift scores
        historical_scores = [0.05, 0.08, 0.12, 0.15, 0.20]
        
        new_threshold = detector.adjust_threshold_adaptively(historical_scores)
        
        assert new_threshold != detector.drift_threshold
        assert new_threshold > 0


class TestInferenceEngine:
    """Test InferenceEngine functionality."""

    def test_inference_engine_initialization(self, mock_config):
        """Test InferenceEngine initialization."""
        engine = InferenceEngine(mock_config)
        
        assert engine.config == mock_config
        assert engine.models == {}
        assert engine.inference_cache == {}

    def test_register_model(self, mock_config):
        """Test model registration."""
        engine = InferenceEngine(mock_config)
        
        mock_model = Mock()
        mock_model.model_name = "test_model"
        mock_model.is_trained = True
        
        engine.register_model("test_model", mock_model)
        
        assert "test_model" in engine.models
        assert engine.models["test_model"] == mock_model

    def test_batch_inference(self, mock_config, sample_features):
        """Test batch inference across multiple models."""
        engine = InferenceEngine(mock_config)
        
        # Register multiple models
        for i in range(3):
            mock_model = Mock()
            mock_model.model_name = f"model_{i}"
            mock_model.is_trained = True
            mock_model.predict = Mock(return_value=np.random.rand(len(sample_features)))
            engine.register_model(f"model_{i}", mock_model)
        
        results = engine.batch_inference(sample_features, ["model_0", "model_1", "model_2"])
        
        assert len(results) == 3
        assert all(model_name in results for model_name in ["model_0", "model_1", "model_2"])

    def test_inference_caching(self, mock_config, sample_features):
        """Test inference result caching."""
        engine = InferenceEngine(mock_config)
        
        mock_model = Mock()
        mock_model.model_name = "cached_model"
        mock_model.is_trained = True
        mock_model.predict = Mock(return_value=np.array([1.0, 2.0, 3.0]))
        engine.register_model("cached_model", mock_model)
        
        # First inference (should call model)
        result1 = engine.inference_with_cache("cached_model", sample_features[:3])
        
        # Second inference with same data (should use cache)
        result2 = engine.inference_with_cache("cached_model", sample_features[:3])
        
        assert np.array_equal(result1, result2)
        # Model.predict should only be called once due to caching
        mock_model.predict.assert_called_once()

    def test_inference_timeout_handling(self, mock_config, sample_features):
        """Test inference timeout handling."""
        engine = InferenceEngine(mock_config)
        mock_config.ml.max_inference_time_ms = 50  # Very short timeout
        
        def slow_predict(data):
            import time
            time.sleep(0.1)  # Simulate slow inference
            return np.random.rand(len(data))
        
        mock_model = Mock()
        mock_model.model_name = "slow_model"
        mock_model.is_trained = True
        mock_model.predict = slow_predict
        engine.register_model("slow_model", mock_model)
        
        with pytest.raises(ModelInferenceError, match="timeout"):
            engine.inference_with_timeout("slow_model", sample_features)


class TestModelManager:
    """Test ModelManager functionality."""

    def test_model_manager_initialization(self, mock_config):
        """Test ModelManager initialization."""
        manager = ModelManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.models == {}
        assert manager.model_versions == {}

    def test_model_registration_and_versioning(self, mock_config):
        """Test model registration with versioning."""
        manager = ModelManager(mock_config)
        
        # Register first version
        model_v1 = Mock()
        model_v1.model_name = "test_model"
        model_v1.model_version = "v1.0"
        
        manager.register_model(model_v1)
        
        assert "test_model" in manager.models
        assert manager.get_current_version("test_model") == "v1.0"
        
        # Register newer version
        model_v2 = Mock()
        model_v2.model_name = "test_model"
        model_v2.model_version = "v2.0"
        
        manager.register_model(model_v2)
        
        # Should update to newer version
        assert manager.get_current_version("test_model") == "v2.0"

    def test_model_rollback(self, mock_config):
        """Test model rollback to previous version."""
        manager = ModelManager(mock_config)
        
        # Register multiple versions
        for i in range(3):
            model = Mock()
            model.model_name = "rollback_model"
            model.model_version = f"v{i+1}.0"
            manager.register_model(model)
        
        # Should be at v3.0
        assert manager.get_current_version("rollback_model") == "v3.0"
        
        # Rollback to v2.0
        success = manager.rollback_model("rollback_model", "v2.0")
        
        assert success is True
        assert manager.get_current_version("rollback_model") == "v2.0"

    def test_model_health_check(self, mock_config, sample_features):
        """Test model health check functionality."""
        manager = ModelManager(mock_config)
        
        # Register healthy model
        healthy_model = Mock()
        healthy_model.model_name = "healthy_model"
        healthy_model.is_trained = True
        healthy_model.predict = Mock(return_value=np.array([1.0, 2.0, 3.0]))
        manager.register_model(healthy_model)
        
        # Register unhealthy model
        unhealthy_model = Mock()
        unhealthy_model.model_name = "unhealthy_model"
        unhealthy_model.is_trained = True
        unhealthy_model.predict = Mock(side_effect=Exception("Model error"))
        manager.register_model(unhealthy_model)
        
        health_results = manager.check_model_health(sample_features[:3])
        
        assert "healthy_model" in health_results
        assert "unhealthy_model" in health_results
        assert health_results["healthy_model"]["healthy"] is True
        assert health_results["unhealthy_model"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_async_model_loading(self, mock_config):
        """Test asynchronous model loading."""
        manager = ModelManager(mock_config)
        
        async def mock_load_model(model_path):
            # Simulate async loading
            import asyncio
            await asyncio.sleep(0.01)
            mock_model = Mock()
            mock_model.model_name = "async_loaded_model"
            mock_model.is_trained = True
            return mock_model
        
        with patch.object(manager, '_load_model_async', side_effect=mock_load_model):
            model = await manager.load_model_async("/tmp/model.pkl")
            
            assert model.model_name == "async_loaded_model"
            assert model.is_trained is True

    def test_model_performance_tracking(self, mock_config):
        """Test model performance tracking over time."""
        manager = ModelManager(mock_config)
        
        model = Mock()
        model.model_name = "tracked_model"
        manager.register_model(model)
        
        # Record performance metrics over time
        performance_data = [
            {'timestamp': '2024-01-01', 'accuracy': 0.85, 'precision': 0.80},
            {'timestamp': '2024-01-02', 'accuracy': 0.83, 'precision': 0.78},
            {'timestamp': '2024-01-03', 'accuracy': 0.80, 'precision': 0.75},
        ]
        
        for perf in performance_data:
            manager.record_model_performance("tracked_model", perf)
        
        # Get performance history
        history = manager.get_performance_history("tracked_model")
        
        assert len(history) == 3
        assert all('accuracy' in entry for entry in history)
        
        # Check for performance degradation
        degradation_detected = manager.detect_performance_degradation("tracked_model")
        assert degradation_detected is True  # Accuracy decreased over time


class TestModelIntegration:
    """Test integration between ML components."""

    @pytest.mark.asyncio
    async def test_end_to_end_model_pipeline(self, mock_config, sample_market_data):
        """Test complete ML pipeline from training to inference."""
        # This test simulates a complete workflow:
        # 1. Train a model
        # 2. Validate it
        # 3. Deploy it
        # 4. Use it for inference
        # 5. Monitor for drift
        
        # Setup components
        trainer = ModelTrainer(mock_config)
        validator = ModelValidator(mock_config)
        drift_detector = DriftDetector(mock_config)
        inference_engine = InferenceEngine(mock_config)
        
        # Mock model
        mock_model = Mock()
        mock_model.model_name = "integration_test_model"
        mock_model.train = Mock(return_value=True)
        mock_model.predict = Mock(return_value=np.array([0, 1, 0, 1, 0]))
        mock_model.validate = Mock(return_value={
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77
        })
        
        # Prepare features and labels
        features = np.random.rand(100, 10)
        labels = np.random.randint(0, 2, 100)
        
        # 1. Train model
        training_result = trainer.train_model(mock_model, features, labels)
        assert training_result['success'] is True
        
        # 2. Validate model
        validation_metrics = validator.validate_classification_model(
            mock_model, features[:20], labels[:20]
        )
        assert validator.meets_performance_threshold(validation_metrics)
        
        # 3. Set up drift detection baseline
        drift_detector.set_reference_data(features)
        drift_detector.set_baseline_performance(validation_metrics)
        
        # 4. Deploy to inference engine
        inference_engine.register_model("integration_test_model", mock_model)
        
        # 5. Run inference
        new_data = np.random.rand(5, 10)
        predictions = inference_engine.inference_with_cache(
            "integration_test_model", new_data
        )
        assert len(predictions) == 5
        
        # 6. Check for drift
        drift_result = drift_detector.detect_data_drift(new_data)
        assert 'drift_detected' in drift_result
        
        # Pipeline completed successfully
        assert True  # All steps completed without errors