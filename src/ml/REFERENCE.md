# ML Module Reference

## INTEGRATION
**Dependencies**: base, core, data, utils
**Used By**: strategies
**Provides**: BatchPredictionService, BatchPredictorService, CrossValidationService, DriftDetectionService, FeatureEngineeringService, FeatureStoreService, HyperparameterOptimizationService, IBatchPredictionService, IDriftDetectionService, IFeatureEngineeringService, IInferenceService, IMLService, IModelManagerService, IModelRegistryService, IModelValidationService, ITrainingService, InferenceService, MLIntegrationService, MLService, MLValidationService, ModelCacheService, ModelManagerService, ModelRegistryService, ModelStorageManager, ModelTrainingService, ModelValidationService, TrainingService
**Patterns**: Async Operations, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Performance**:
- Parallel execution
- Parallel execution
- Parallel execution
**Architecture**:
- FeatureEngineeringService inherits from base architecture
- MLIntegrationService inherits from base architecture
- ModelManagerService inherits from base architecture

## MODULE OVERVIEW
**Files**: 29 Python files
**Classes**: 75
**Functions**: 2

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `MLDataTransformer` ✅

**Purpose**: Handles consistent data transformation for ML module operations aligned with core patterns
**Status**: Complete

**Implemented Methods:**
- `transform_ml_request_to_standard_format(request_type, ...) -> dict[str, Any]` - Line 26
- `transform_for_inference_pattern(model_id: str, input_data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 82
- `transform_for_training_pattern(training_data, ...) -> dict[str, Any]` - Line 113
- `align_with_core_processing_paradigm(data: dict[str, Any], target_mode: str | None = None) -> dict[str, Any]` - Line 151
- `validate_ml_boundary_fields(data: dict[str, Any]) -> dict[str, Any]` - Line 182
- `apply_cross_module_consistency_from_ml(cls, data: dict[str, Any], target_module: str) -> dict[str, Any]` - Line 262
- `handle_ml_error_propagation(error: Exception, context: dict[str, Any], target_module: str = 'core') -> dict[str, Any]` - Line 297
- `batch_transform_ml_data(batch_data, ...) -> dict[str, Any]` - Line 353
- `stream_to_batch_ml_data(stream_items: list[dict[str, Any]], batch_size: int = 100) -> list[dict[str, Any]]` - Line 397
- `validate_ml_to_utils_boundary(data: dict[str, Any]) -> dict[str, Any]` - Line 458
- `validate_utils_to_ml_boundary(data: dict[str, Any]) -> dict[str, Any]` - Line 552
- `create_ml_error_propagation_mixin(cls) -> 'MLErrorPropagationMixin'` - Line 621

### Implementation: `MLErrorPropagationMixin` ✅

**Purpose**: ML-specific error propagation mixin aligned with utils messaging patterns
**Status**: Complete

**Implemented Methods:**
- `propagate_ml_validation_error(self, error: Exception, context: str) -> None` - Line 629
- `propagate_ml_model_error(self, error: Exception, context: str) -> None` - Line 659
- `propagate_ml_training_error(self, error: Exception, context: str) -> None` - Line 692
- `propagate_ml_inference_error(self, error: Exception, context: str) -> None` - Line 727

### Implementation: `ModelFactory` ✅

**Inherits**: BaseFactory[BaseMLModel], IModelFactory
**Purpose**: Factory for creating ML model instances
**Status**: Complete

**Implemented Methods:**
- `set_container(self, container: Any) -> None` - Line 51
- `create_model(self, ...) -> BaseMLModel` - Line 139
- `get_available_models(self) -> list[str]` - Line 197
- `get_model_info(self, model_type: str) -> dict[str, Any] | None` - Line 206
- `register_custom_model(self, ...) -> None` - Line 218

### Implementation: `FeatureEngineeringConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for feature engineering service
**Status**: Complete

### Implementation: `FeatureRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request for feature computation
**Status**: Complete

### Implementation: `FeatureResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response from feature computation
**Status**: Complete

### Implementation: `FeatureEngineeringService` ✅

**Inherits**: BaseService
**Purpose**: Feature engineering service for creating, selecting, and transforming features
**Status**: Complete

**Implemented Methods:**
- `async compute_features(self, request: FeatureRequest) -> FeatureResponse` - Line 208
- `async select_features(self, ...) -> tuple[pd.DataFrame, list[str], dict[str, float]]` - Line 892
- `async get_feature_engineering_metrics(self) -> dict[str, Any]` - Line 1106
- `async clear_cache(self) -> dict[str, int]` - Line 1119

### Implementation: `BatchPredictorConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for batch predictor service
**Status**: Complete

### Implementation: `BatchPredictorService` ✅

**Inherits**: BaseService
**Purpose**: Simple batch prediction service for ML models
**Status**: Complete

**Implemented Methods:**
- `async predict_batch(self, ...) -> pd.DataFrame` - Line 72
- `async predict_multiple_symbols(self, model_name: str, data_dict: dict[str, pd.DataFrame], parallel: bool = True) -> dict[str, pd.DataFrame]` - Line 145
- `async submit_batch_prediction(self, model_id: str, input_data: pd.DataFrame, **kwargs) -> str | None` - Line 244
- `get_job_status(self, job_id: str) -> dict[str, Any] | None` - Line 260
- `get_job_result(self, job_id: str) -> pd.DataFrame | None` - Line 270
- `list_jobs(self) -> list[dict[str, Any]]` - Line 274

### Implementation: `InferenceConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for inference service
**Status**: Complete

### Implementation: `InferencePredictionRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request object for predictions
**Status**: Complete

### Implementation: `InferencePredictionResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response object for predictions
**Status**: Complete

### Implementation: `InferenceMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Inference service performance metrics
**Status**: Complete

### Implementation: `InferenceService` ✅

**Inherits**: BaseService
**Purpose**: Real-time inference service for ML models
**Status**: Complete

**Implemented Methods:**
- `async predict(self, ...) -> InferencePredictionResponse` - Line 186
- `async predict_async(self, ...) -> InferencePredictionResponse` - Line 323
- `async predict_batch(self, requests: list[InferencePredictionRequest]) -> list[InferencePredictionResponse]` - Line 347
- `async predict_with_features(self, ...) -> InferencePredictionResponse` - Line 506
- `async warm_up_models(self, model_ids: list[str]) -> dict[str, bool]` - Line 823
- `get_inference_metrics(self) -> dict[str, Any]` - Line 913
- `async clear_cache(self) -> dict[str, int]` - Line 932
- `reset_metrics(self) -> None` - Line 955
- `get_metrics(self) -> dict[str, Any]` - Line 1034

### Implementation: `ModelCacheConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for model cache service
**Status**: Complete

### Implementation: `ModelCacheService` ✅

**Inherits**: BaseService
**Purpose**: High-performance cache for ML models
**Status**: Complete

**Implemented Methods:**
- `start_cleanup_thread(self) -> None` - Line 122
- `stop_cleanup_thread(self) -> None` - Line 130
- `async cache_model(self, model_id: str, model: Any) -> bool` - Line 139
- `async get_model(self, model_id: str) -> Any | None` - Line 188
- `remove_model(self, model_id: str) -> bool` - Line 231
- `clear_cache(self) -> None` - Line 244
- `get_cached_models(self) -> dict[str, dict[str, Any]]` - Line 258
- `get_cache_stats(self) -> dict[str, Any]` - Line 281
- `clear_stats(self) -> None` - Line 299
- `health_check(self) -> dict[str, Any]` - Line 474
- `get_model_cache_metrics(self) -> dict[str, Any]` - Line 538
- `get_cache_size(self) -> int` - Line 552
- `get_cache_memory_usage(self) -> float` - Line 557
- `get_cache_statistics(self) -> dict[str, Any]` - Line 562
- `is_model_cached(self, model_id: str) -> bool` - Line 566
- `get_cached_model_ids(self) -> list[str]` - Line 571
- `async evict_model(self, model_id: str) -> bool` - Line 577

### Implementation: `MLIntegrationService` ✅

**Inherits**: BaseService
**Purpose**: Service for handling ML module integration with other modules
**Status**: Complete

**Implemented Methods:**
- `determine_target_processing_mode(self, target_module: str, operation_type: str) -> str` - Line 25
- `prepare_data_for_target_module(self, ...) -> dict[str, Any]` - Line 56
- `determine_integration_mode(self, target_module: str) -> str` - Line 98
- `validate_cross_module_compatibility(self, data: dict[str, Any], target_module: str) -> bool` - Line 118

### Implementation: `IFeatureEngineeringService` 🔧

**Inherits**: ABC
**Purpose**: Interface for feature engineering service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async compute_features(self, request: 'FeatureRequest') -> 'FeatureResponse'` - Line 29
- `async select_features(self, ...) -> tuple[pd.DataFrame, list[str], dict[str, float]]` - Line 34
- `async clear_cache(self) -> dict[str, int]` - Line 46

### Implementation: `IModelRegistryService` 🔧

**Inherits**: ABC
**Purpose**: Interface for model registry service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async register_model(self, request: 'ModelRegistrationRequest') -> str` - Line 55
- `async load_model(self, request: 'ModelLoadRequest') -> dict[str, Any]` - Line 60
- `async list_models(self, ...) -> list[dict[str, Any]]` - Line 65
- `async promote_model(self, model_id: str, stage: str, description: str = '') -> bool` - Line 75
- `async deactivate_model(self, model_id: str, reason: str = '') -> bool` - Line 80
- `async delete_model(self, model_id: str, remove_files: bool = True) -> bool` - Line 85
- `async get_model_metrics(self, model_id: str) -> dict[str, Any]` - Line 90

### Implementation: `IInferenceService` 🔧

**Inherits**: ABC
**Purpose**: Interface for inference service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async predict(self, ...) -> 'InferencePredictionResponse'` - Line 99
- `async predict_batch(self, requests: list['InferencePredictionRequest']) -> list['InferencePredictionResponse']` - Line 111
- `async predict_with_features(self, ...) -> 'InferencePredictionResponse'` - Line 118
- `async warm_up_models(self, model_ids: list[str]) -> dict[str, bool]` - Line 129
- `async clear_cache(self) -> dict[str, int]` - Line 134

### Implementation: `IModelValidationService` 🔧

**Inherits**: ABC
**Purpose**: Interface for model validation service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async validate_model_performance(self, model: Any, test_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, Any]` - Line 143
- `async validate_production_readiness(self, model: Any, validation_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, bool]` - Line 150

### Implementation: `IDriftDetectionService` 🔧

**Inherits**: ABC
**Purpose**: Interface for drift detection service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async detect_feature_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict[str, Any]` - Line 161
- `async detect_prediction_drift(self, ...) -> dict[str, Any]` - Line 168
- `async detect_performance_drift(self, ...) -> dict[str, Any]` - Line 175

### Implementation: `ITrainingService` 🔧

**Inherits**: ABC
**Purpose**: Interface for training service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async train_model(self, ...) -> dict[str, Any]` - Line 189
- `async save_artifacts(self, ...) -> dict[str, Any]` - Line 200

### Implementation: `IBatchPredictionService` 🔧

**Inherits**: ABC
**Purpose**: Interface for batch prediction service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async process_batch_predictions(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]` - Line 215

### Implementation: `IModelManagerService` 🔧

**Inherits**: ABC
**Purpose**: Interface for model manager service
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_and_train_model(self, ...) -> dict[str, Any]` - Line 226
- `async deploy_model(self, model_name: str, deployment_stage: str = 'production') -> dict[str, Any]` - Line 239
- `async monitor_model_performance(self, ...) -> dict[str, Any]` - Line 246
- `async retire_model(self, model_name: str, reason: str = 'replaced') -> dict[str, Any]` - Line 256
- `get_active_models(self) -> dict[str, Any]` - Line 261
- `async get_model_status(self, model_name: str) -> dict[str, Any] | None` - Line 266
- `async health_check(self) -> dict[str, Any]` - Line 271

### Implementation: `IModelFactory` 🔧

**Inherits**: ABC
**Purpose**: Interface for model factory service
**Status**: Abstract Base Class

**Implemented Methods:**
- `create_model(self, ...) -> Any` - Line 280
- `get_available_models(self) -> list[str]` - Line 292
- `register_custom_model(self, ...) -> None` - Line 297

### Implementation: `IMLService` 🔧

**Inherits**: ABC
**Purpose**: Interface for main ML service that coordinates all ML operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async process_pipeline(self, request: Any) -> Any` - Line 313
- `async train_model(self, request: Any) -> Any` - Line 318
- `async process_batch_pipeline(self, requests: list[Any]) -> list[Any]` - Line 323
- `async enhance_strategy_signals(self, ...) -> list` - Line 328
- `async list_available_models(self, model_type: str | None = None, stage: str | None = None) -> list[dict[str, Any]]` - Line 338
- `async promote_model(self, model_id: str, stage: str, description: str = '') -> bool` - Line 345
- `async get_model_info(self, model_id: str) -> dict[str, Any]` - Line 350
- `async clear_cache(self) -> dict[str, int]` - Line 355
- `get_ml_service_metrics(self) -> dict[str, Any]` - Line 360

### Implementation: `ModelManagerConfig` ✅

**Inherits**: PydanticBaseModel
**Purpose**: Configuration for model manager service
**Status**: Complete

### Implementation: `ModelManagerService` ✅

**Inherits**: BaseService, IModelManagerService
**Purpose**: Central manager for ML model lifecycle
**Status**: Complete

**Implemented Methods:**
- `async create_and_train_model(self, ...) -> dict[str, Any]` - Line 206
- `async deploy_model(self, model_name: str, deployment_stage: str = 'production') -> dict[str, Any]` - Line 323
- `async monitor_model_performance(self, ...) -> dict[str, Any]` - Line 423
- `async retire_model(self, model_name: str, reason: str = 'replaced') -> dict[str, Any]` - Line 578
- `get_active_models(self) -> dict[str, Any]` - Line 826
- `async get_model_status(self, model_name: str) -> dict[str, Any] | None` - Line 838
- `async health_check(self) -> dict[str, Any]` - Line 851
- `get_model_manager_metrics(self) -> dict[str, Any]` - Line 957

### Implementation: `BaseMLModelConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for ML models
**Status**: Complete

### Implementation: `BaseMLModel` 🔧

**Inherits**: BaseService, abc.ABC
**Purpose**: Abstract base class for all ML models in the trading system
**Status**: Abstract Base Class

**Implemented Methods:**
- `prepare_data(self, ...) -> tuple[pd.DataFrame, pd.Series | None]` - Line 157
- `train(self, ...) -> dict[str, float]` - Line 203
- `predict(self, X: pd.DataFrame, return_probabilities: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]` - Line 307
- `evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]` - Line 352
- `async save(self, filepath: str | Path) -> Path` - Line 391
- `load(cls, filepath: str | Path, config: ConfigDict | None = None) -> 'BaseMLModel'` - Line 451
- `get_feature_importance(self) -> pd.Series | None` - Line 510
- `get_model_info(self) -> dict[str, Any]` - Line 537
- `get_model_metrics(self) -> dict[str, Any]` - Line 601

### Implementation: `DirectionClassifier` ✅

**Inherits**: BaseMLModel
**Purpose**: Direction classification model for predicting price movement direction
**Status**: Complete

**Implemented Methods:**
- `fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]` - Line 178
- `predict(self, X: pd.DataFrame) -> np.ndarray` - Line 264
- `predict_proba(self, X: pd.DataFrame) -> np.ndarray` - Line 301
- `evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]` - Line 338
- `get_feature_importance(self) -> pd.Series | None` - Line 464
- `get_class_distribution(self) -> dict[str, int] | None` - Line 477
- `predict_direction_labels(self, X: pd.DataFrame) -> list[str]` - Line 490
- `get_prediction_confidence(self, X: pd.DataFrame) -> np.ndarray` - Line 508

### Implementation: `PricePredictor` ✅

**Inherits**: BaseMLModel
**Purpose**: Price prediction model for financial instruments
**Status**: Complete

**Implemented Methods:**
- `create_target_from_prices(self, prices: pd.Series, target_type: str = 'return', horizon: int | None = None) -> pd.Series` - Line 222
- `predict_price_sequence(self, X: pd.DataFrame, sequence_length: int = 10) -> np.ndarray` - Line 249
- `calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, transaction_cost: Decimal = Any) -> dict[str, Any]` - Line 281
- `get_feature_importance_analysis(self) -> dict[str, Any]` - Line 299

### Implementation: `RegimeDetector` ✅

**Inherits**: BaseMLModel
**Purpose**: Market regime detection model for identifying market conditions
**Status**: Complete

**Implemented Methods:**
- `fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]` - Line 151
- `predict(self, X: pd.DataFrame) -> np.ndarray` - Line 261
- `evaluate(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]` - Line 315
- `predict_regime_labels(self, X: pd.DataFrame) -> list[str]` - Line 518
- `get_regime_probabilities(self, X: pd.DataFrame) -> np.ndarray | None` - Line 539
- `get_regime_statistics(self) -> dict[str, Any] | None` - Line 574
- `get_feature_importance(self) -> pd.Series | None` - Line 587
- `get_regime_names(self) -> list[str]` - Line 600
- `interpret_regime(self, regime_id: int) -> str` - Line 604

### Implementation: `ModelStorageBackend` 🔧

**Inherits**: abc.ABC
**Purpose**: Abstract base class for model storage backends
**Status**: Abstract Base Class

**Implemented Methods:**
- `save(self, model_data: dict[str, Any], filepath: Path) -> None` - Line 19
- `load(self, filepath: Path) -> dict[str, Any]` - Line 24

### Implementation: `JoblibStorageBackend` ✅

**Inherits**: ModelStorageBackend
**Purpose**: Joblib-based storage backend for sklearn models
**Status**: Complete

**Implemented Methods:**
- `save(self, model_data: dict[str, Any], filepath: Path) -> None` - Line 32
- `load(self, filepath: Path) -> dict[str, Any]` - Line 47

### Implementation: `PickleStorageBackend` ✅

**Inherits**: ModelStorageBackend
**Purpose**: Pickle-based storage backend for general Python objects
**Status**: Complete

**Implemented Methods:**
- `save(self, model_data: dict[str, Any], filepath: Path) -> None` - Line 67
- `load(self, filepath: Path) -> dict[str, Any]` - Line 89

### Implementation: `ModelStorageManager` ✅

**Purpose**: Manager for model storage operations
**Status**: Complete

**Implemented Methods:**
- `save_model(self, model_data: dict[str, Any], filepath: str | Path) -> Path` - Line 135
- `load_model(self, filepath: str | Path) -> dict[str, Any]` - Line 150

### Implementation: `VolatilityForecaster` ✅

**Inherits**: BaseMLModel
**Purpose**: Volatility forecasting model for predicting future volatility
**Status**: Complete

**Implemented Methods:**
- `fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]` - Line 141
- `predict(self, X: pd.DataFrame) -> np.ndarray` - Line 260
- `evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]` - Line 308
- `get_feature_importance(self) -> pd.Series | None` - Line 614
- `get_volatility_stats(self) -> dict[str, float] | None` - Line 627
- `predict_volatility_regime(self, X: pd.DataFrame) -> list[str]` - Line 640

### Implementation: `ArtifactStoreConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for artifact store service
**Status**: Complete

### Implementation: `ArtifactStore` ✅

**Inherits**: BaseService
**Purpose**: Artifact store service for managing ML model artifacts
**Status**: Complete

**Implemented Methods:**
- `async store_artifact(self, ...) -> str` - Line 139
- `async retrieve_artifact(self, ...) -> Any` - Line 277
- `async list_artifacts(self, ...) -> pd.DataFrame` - Line 410
- `async delete_artifact(self, ...) -> bool` - Line 492
- `async cleanup_old_artifacts(self, days_to_keep: int | None = None) -> int` - Line 606
- `get_artifact_store_metrics(self) -> dict[str, Any]` - Line 1075

### Implementation: `ModelRegistryConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for model registry service
**Status**: Complete

### Implementation: `ModelMetadata` ✅

**Inherits**: BaseModel
**Purpose**: Model metadata structure
**Status**: Complete

### Implementation: `ModelRegistrationRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request for model registration
**Status**: Complete

### Implementation: `ModelLoadRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request for model loading
**Status**: Complete

### Implementation: `ModelRegistryService` ✅

**Inherits**: BaseService
**Purpose**: Model registry service for managing ML model versions and storage
**Status**: Complete

**Implemented Methods:**
- `async register_model(self, request: ModelRegistrationRequest) -> str` - Line 185
- `async load_model(self, request: ModelLoadRequest) -> dict[str, Any]` - Line 286
- `async list_models(self, ...) -> list[dict[str, Any]]` - Line 385
- `async promote_model(self, model_id: str, stage: str, description: str = '') -> bool` - Line 451
- `async deactivate_model(self, model_id: str, reason: str = '') -> bool` - Line 530
- `async delete_model(self, model_id: str, remove_files: bool = True) -> bool` - Line 597
- `async get_model_metrics(self, model_id: str) -> dict[str, Any]` - Line 675
- `get_model_registry_metrics(self) -> dict[str, Any]` - Line 1083
- `async clear_cache(self) -> dict[str, int]` - Line 1094

### Implementation: `IMLRepository` 🔧

**Inherits**: ABC
**Purpose**: Interface for ML data repository
**Status**: Abstract Base Class

**Implemented Methods:**
- `async store_model_metadata(self, metadata: dict[str, Any]) -> str` - Line 22
- `async get_model_by_id(self, model_id: str) -> dict[str, Any] | None` - Line 27
- `async get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]` - Line 32
- `async find_models(self, ...) -> list[dict[str, Any]]` - Line 37
- `async get_all_models(self, ...) -> list[dict[str, Any]]` - Line 49
- `async update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> bool` - Line 59
- `async delete_model(self, model_id: str) -> bool` - Line 64
- `async store_prediction(self, prediction_data: dict[str, Any]) -> str` - Line 69
- `async get_predictions(self, ...) -> list[dict[str, Any]]` - Line 74
- `async store_training_job(self, job_data: dict[str, Any]) -> str` - Line 86
- `async get_training_job(self, job_id: str) -> dict[str, Any] | None` - Line 91
- `async update_training_progress(self, job_id: str, progress: dict[str, Any]) -> bool` - Line 96

### Implementation: `MLRepository` ✅

**Inherits**: BaseRepository, IMLRepository
**Purpose**: ML repository implementation using actual database models
**Status**: Complete

**Implemented Methods:**
- `async store_model_metadata(self, metadata: dict[str, Any]) -> str` - Line 140
- `async get_model_by_id(self, model_id: str) -> dict[str, Any] | None` - Line 183
- `async get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]` - Line 205
- `async find_models(self, ...) -> list[dict[str, Any]]` - Line 232
- `async get_all_models(self, ...) -> list[dict[str, Any]]` - Line 282
- `async update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> bool` - Line 297
- `async delete_model(self, model_id: str) -> bool` - Line 324
- `async store_prediction(self, prediction_data: dict[str, Any]) -> str` - Line 350
- `async get_predictions(self, ...) -> list[dict[str, Any]]` - Line 372
- `async store_training_job(self, job_data: dict[str, Any]) -> str` - Line 413
- `async get_training_job(self, job_id: str) -> dict[str, Any] | None` - Line 431
- `async update_training_progress(self, job_id: str, progress: dict[str, Any]) -> bool` - Line 450
- `async store_audit_entry(self, category: str, entry: dict[str, Any]) -> bool` - Line 471

### Implementation: `MLServiceConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for ML service
**Status**: Complete

### Implementation: `MLPipelineRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request for ML pipeline processing
**Status**: Complete

### Implementation: `MLPipelineResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response from ML pipeline processing
**Status**: Complete

### Implementation: `MLTrainingRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request for ML model training
**Status**: Complete

### Implementation: `MLTrainingResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response from ML model training
**Status**: Complete

### Implementation: `MLService` ✅

**Inherits**: BaseService, IMLService
**Purpose**: Main ML service coordinating all machine learning operations
**Status**: Complete

**Implemented Methods:**
- `async process_pipeline(self, request: MLPipelineRequest) -> MLPipelineResponse` - Line 281
- `async train_model(self, request: MLTrainingRequest) -> MLTrainingResponse` - Line 649
- `async process_batch_pipeline(self, requests: list[MLPipelineRequest]) -> list[MLPipelineResponse]` - Line 1007
- `async clear_ml_cache(self) -> dict[str, int]` - Line 1139
- `async list_available_models(self, model_type: str | None = None, stage: str | None = None) -> list[dict[str, Any]]` - Line 1149
- `async promote_model(self, model_id: str, stage: str, description: str = '') -> bool` - Line 1160
- `async get_model_info(self, model_id: str) -> dict[str, Any]` - Line 1167
- `get_ml_service_metrics(self) -> dict[str, Any]` - Line 1257
- `async clear_cache(self) -> dict[str, int]` - Line 1276
- `async enhance_strategy_signals(self, ...) -> list` - Line 1282

### Implementation: `ModelValidationService` ✅

**Inherits**: BaseService, IModelValidationService
**Purpose**: Mock model validation service
**Status**: Complete

**Implemented Methods:**
- `async validate_model_performance(self, model: Any, test_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, Any]` - Line 33
- `async validate_production_readiness(self, model: Any, validation_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, bool]` - Line 54

### Implementation: `DriftDetectionService` ✅

**Inherits**: BaseService, IDriftDetectionService
**Purpose**: Mock drift detection service
**Status**: Complete

**Implemented Methods:**
- `async get_reference_data(self, data_type: str) -> Any` - Line 77
- `async set_reference_data(self, data: pd.DataFrame, data_type: str) -> None` - Line 81
- `async detect_feature_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict[str, Any]` - Line 85
- `async detect_prediction_drift(self, ...) -> dict[str, Any]` - Line 97
- `async detect_performance_drift(self, ...) -> dict[str, Any]` - Line 109

### Implementation: `TrainingService` ✅

**Inherits**: BaseService, ITrainingService
**Purpose**: Mock training service
**Status**: Complete

**Implemented Methods:**
- `async train_model(self, ...) -> dict[str, Any]` - Line 135
- `async save_artifacts(self, ...) -> dict[str, Any]` - Line 151

### Implementation: `BatchPredictionService` ✅

**Inherits**: BaseService, IBatchPredictionService
**Purpose**: Mock batch prediction service
**Status**: Complete

**Implemented Methods:**
- `async process_batch_predictions(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]` - Line 177

### Implementation: `FeatureStoreConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for feature store service
**Status**: Complete

### Implementation: `FeatureStoreMetadata` ✅

**Inherits**: BaseModel
**Purpose**: Feature store metadata structure
**Status**: Complete

### Implementation: `FeatureStoreRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request for feature store operations
**Status**: Complete

### Implementation: `FeatureStoreResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response from feature store operations
**Status**: Complete

### Implementation: `FeatureStoreService` ✅

**Inherits**: BaseService
**Purpose**: Feature store service for centralized ML feature management
**Status**: Complete

**Implemented Methods:**
- `async store_features(self, ...) -> FeatureStoreResponse` - Line 197
- `async retrieve_features(self, ...) -> FeatureStoreResponse` - Line 352
- `async list_feature_sets(self, ...) -> FeatureStoreResponse` - Line 492
- `async delete_features(self, ...) -> FeatureStoreResponse` - Line 579
- `get_feature_store_metrics(self) -> dict[str, Any]` - Line 1006
- `async clear_cache(self) -> dict[str, int]` - Line 1019

### Implementation: `TimeSeriesValidator` ✅

**Purpose**: Time series specific cross-validation strategies
**Status**: Complete

**Implemented Methods:**
- `purged_walk_forward_split(data, ...) -> Generator[tuple[np.ndarray, np.ndarray], None, None]` - Line 49
- `combinatorial_purged_cross_validation(data, ...) -> Generator[tuple[np.ndarray, np.ndarray], None, None]` - Line 90
- `walk_forward_split(data: pd.DataFrame, min_train_size: int, test_size: int, step_size: int = 1) -> Generator[tuple[np.ndarray, np.ndarray], None, None]` - Line 144
- `expanding_window_split(data: pd.DataFrame, min_train_size: int, test_size: int, step_size: int = 1) -> Generator[tuple[np.ndarray, np.ndarray], None, None]` - Line 169
- `sliding_window_split(data: pd.DataFrame, train_size: int, test_size: int, step_size: int = 1) -> Generator[tuple[np.ndarray, np.ndarray], None, None]` - Line 194

### Implementation: `CrossValidationService` ✅

**Inherits**: BaseService
**Purpose**: Cross-validation service for ML models
**Status**: Complete

**Implemented Methods:**
- `async validate_model(self, ...) -> dict[str, Any]` - Line 257
- `async time_series_validation(self, ...) -> dict[str, Any]` - Line 367
- `async nested_cross_validation(self, ...) -> dict[str, Any]` - Line 517
- `get_validation_history(self) -> list[dict[str, Any]]` - Line 1011
- `clear_history(self) -> None` - Line 1015

### Implementation: `HyperparameterOptimizationService` ✅

**Inherits**: BaseService
**Purpose**: Simple hyperparameter optimization service using Optuna
**Status**: Complete

**Implemented Methods:**
- `async optimize_model(self, ...) -> dict[str, Any]` - Line 41

### Implementation: `TrainingPipeline` ✅

**Purpose**: Training pipeline for managing data preparation and model training flow
**Status**: Complete

**Implemented Methods:**
- `fit(self, X: pd.DataFrame, y: pd.Series) -> 'TrainingPipeline'` - Line 43
- `transform(self, X: pd.DataFrame) -> pd.DataFrame` - Line 56
- `fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame` - Line 69

### Implementation: `ModelTrainingService` ✅

**Inherits**: BaseService
**Purpose**: Training orchestration service for ML models
**Status**: Complete

**Implemented Methods:**
- `async train_model(self, ...) -> dict[str, Any]` - Line 122
- `async batch_train_models(self, ...) -> list[dict[str, Any]]` - Line 273
- `get_training_history(self) -> list[dict[str, Any]]` - Line 593
- `get_best_model_by_metric(self, ...) -> dict[str, Any] | None` - Line 597
- `clear_history(self) -> None` - Line 664

### Implementation: `ABTestStatus` ✅

**Inherits**: Enum
**Purpose**: A/B Test status enumeration
**Status**: Complete

### Implementation: `ABTestVariant` ✅

**Purpose**: Represents a variant in an A/B test
**Status**: Complete

**Implemented Methods:**

### Implementation: `ABTest` ✅

**Purpose**: Represents an A/B test for ML model evaluation
**Status**: Complete

**Implemented Methods:**

### Implementation: `ABTestFramework` ✅

**Inherits**: BaseComponent
**Purpose**: A/B Testing Framework for ML Model Deployment
**Status**: Complete

**Implemented Methods:**
- `create_ab_test(self, ...) -> str` - Line 159
- `start_ab_test(self, test_id: str) -> bool` - Line 284
- `assign_variant(self, test_id: str, user_id: str) -> str` - Line 327
- `async record_result(self, ...) -> bool` - Line 380
- `analyze_ab_test(self, test_id: str) -> dict[str, Any]` - Line 448
- `async stop_ab_test(self, test_id: str, reason: str = 'manual_stop') -> bool` - Line 1149
- `get_active_tests(self) -> dict[str, dict[str, Any]]` - Line 1201
- `async get_test_results(self, test_id: str) -> dict[str, Any]` - Line 1226

### Implementation: `DriftDetectionService` ✅

**Inherits**: BaseService
**Purpose**: Drift detection system for monitoring data and model drift
**Status**: Complete

**Implemented Methods:**
- `async detect_feature_drift(self, ...) -> dict[str, Any]` - Line 122
- `async detect_prediction_drift(self, ...) -> dict[str, Any]` - Line 234
- `async detect_performance_drift(self, ...) -> dict[str, Any]` - Line 348
- `get_drift_history(self, ...) -> list[dict[str, Any]]` - Line 672
- `set_reference_data(self, reference_data: pd.DataFrame, data_type: str = 'features') -> None` - Line 720
- `get_reference_data(self, data_type: str = 'features') -> pd.DataFrame | None` - Line 754
- `clear_reference_data(self, data_type: str | None = None) -> None` - Line 766
- `async continuous_monitoring(self, ...) -> dict[str, Any]` - Line 805

### Implementation: `ModelValidationService` ✅

**Inherits**: BaseService
**Purpose**: Comprehensive model validation system
**Status**: Complete

**Implemented Methods:**
- `async validate_model_performance(self, ...) -> dict[str, Any]` - Line 136
- `async validate_model_stability(self, ...) -> dict[str, Any]` - Line 269
- `async validate_production_readiness(self, model: BaseMLModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]` - Line 401
- `async detect_overfitting(self, ...) -> dict[str, Any]` - Line 513
- `get_overfitting_alerts(self) -> list[dict[str, Any]]` - Line 1138
- `clear_overfitting_alerts(self) -> None` - Line 1142
- `get_validation_history(self) -> list[dict[str, Any]]` - Line 1436
- `get_benchmark_results(self) -> dict[str, Any]` - Line 1440
- `clear_validation_history(self) -> None` - Line 1444

### Implementation: `MLValidationService` ✅

**Inherits**: BaseService
**Purpose**: Service for ML-specific business logic validation
**Status**: Complete

**Implemented Methods:**
- `validate_ml_operation_type(self, ml_operation_type: str) -> bool` - Line 25
- `validate_ml_request_data(self, data: dict[str, Any]) -> dict[str, Any]` - Line 53
- `validate_model_parameters(self, model_type: str, parameters: dict[str, Any]) -> bool` - Line 87
- `validate_feature_data(self, feature_data: Any) -> bool` - Line 127

## COMPLETE API REFERENCE

### File: data_transformer.py

**Key Imports:**
- `from src.core.data_transformer import CoreDataTransformer`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `MLDataTransformer`

**Purpose**: Handles consistent data transformation for ML module operations aligned with core patterns

```python
class MLDataTransformer:
    def transform_ml_request_to_standard_format(request_type, ...) -> dict[str, Any]  # Line 26
    def transform_for_inference_pattern(model_id: str, input_data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 82
    def transform_for_training_pattern(training_data, ...) -> dict[str, Any]  # Line 113
    def align_with_core_processing_paradigm(data: dict[str, Any], target_mode: str | None = None) -> dict[str, Any]  # Line 151
    def validate_ml_boundary_fields(data: dict[str, Any]) -> dict[str, Any]  # Line 182
    def _apply_ml_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 213
    def apply_cross_module_consistency_from_ml(cls, data: dict[str, Any], target_module: str) -> dict[str, Any]  # Line 262
    def handle_ml_error_propagation(error: Exception, context: dict[str, Any], target_module: str = 'core') -> dict[str, Any]  # Line 297
    def batch_transform_ml_data(batch_data, ...) -> dict[str, Any]  # Line 353
    def stream_to_batch_ml_data(stream_items: list[dict[str, Any]], batch_size: int = 100) -> list[dict[str, Any]]  # Line 397
    def _apply_ml_financial_precision_to_item(item: dict[str, Any]) -> None  # Line 425
    def validate_ml_to_utils_boundary(data: dict[str, Any]) -> dict[str, Any]  # Line 458
    def validate_utils_to_ml_boundary(data: dict[str, Any]) -> dict[str, Any]  # Line 552
    def create_ml_error_propagation_mixin(cls) -> 'MLErrorPropagationMixin'  # Line 621
```

#### Class: `MLErrorPropagationMixin`

**Purpose**: ML-specific error propagation mixin aligned with utils messaging patterns

```python
class MLErrorPropagationMixin:
    def propagate_ml_validation_error(self, error: Exception, context: str) -> None  # Line 629
    def propagate_ml_model_error(self, error: Exception, context: str) -> None  # Line 659
    def propagate_ml_training_error(self, error: Exception, context: str) -> None  # Line 692
    def propagate_ml_inference_error(self, error: Exception, context: str) -> None  # Line 727
```

### File: di_registration.py

**Key Imports:**
- `from src.core.types.base import ConfigDict`
- `from src.ml.factory import ModelFactory`
- `from src.ml.feature_engineering import FeatureEngineeringService`
- `from src.ml.inference.inference_engine import InferenceService`
- `from src.ml.inference.model_cache import ModelCacheService`

#### Functions:

```python
def register_ml_services(container: 'DependencyContainer', config: ConfigDict) -> None  # Line 34
def get_ml_service_dependencies() -> dict[str, list[str]]  # Line 116
```

### File: factory.py

**Key Imports:**
- `from src.core.base.factory import BaseFactory`
- `from src.core.exceptions import CreationError`
- `from src.core.exceptions import RegistrationError`
- `from src.core.types.base import ConfigDict`
- `from src.ml.interfaces import IModelFactory`

#### Class: `ModelFactory`

**Inherits**: BaseFactory[BaseMLModel], IModelFactory
**Purpose**: Factory for creating ML model instances

```python
class ModelFactory(BaseFactory[BaseMLModel], IModelFactory):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 30
    def set_container(self, container: Any) -> None  # Line 51
    def _register_default_models(self) -> None  # Line 60
    def _create_direction_classifier(self, **kwargs: Any) -> DirectionClassifier  # Line 105
    def _create_price_predictor(self, **kwargs: Any) -> PricePredictor  # Line 110
    def _create_volatility_forecaster(self, **kwargs: Any) -> VolatilityForecaster  # Line 115
    def _create_regime_detector(self, **kwargs: Any) -> RegimeDetector  # Line 120
    def _get_injected_config(self, **kwargs: Any) -> Any  # Line 125
    def create_model(self, ...) -> BaseMLModel  # Line 139
    def get_available_models(self) -> list[str]  # Line 197
    def get_model_info(self, model_type: str) -> dict[str, Any] | None  # Line 206
    def register_custom_model(self, ...) -> None  # Line 218
```

### File: feature_engineering.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import DataError`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`

#### Class: `FeatureEngineeringConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for feature engineering service

```python
class FeatureEngineeringConfig(BaseModel):
```

#### Class: `FeatureRequest`

**Inherits**: BaseModel
**Purpose**: Request for feature computation

```python
class FeatureRequest(BaseModel):
```

#### Class: `FeatureResponse`

**Inherits**: BaseModel
**Purpose**: Response from feature computation

```python
class FeatureResponse(BaseModel):
```

#### Class: `FeatureEngineeringService`

**Inherits**: BaseService
**Purpose**: Feature engineering service for creating, selecting, and transforming features

```python
class FeatureEngineeringService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None  # Line 122
    async def _do_start(self) -> None  # Line 164
    async def _do_stop(self) -> None  # Line 199
    async def compute_features(self, request: FeatureRequest) -> FeatureResponse  # Line 208
    async def _compute_features_impl(self, request: FeatureRequest) -> FeatureResponse  # Line 227
    async def _compute_all_feature_types(self, market_data: pd.DataFrame, symbol: str, feature_types: list[str]) -> pd.DataFrame  # Line 471
    async def _compute_price_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame  # Line 528
    def _compute_price_features(self, data: pd.DataFrame) -> pd.DataFrame  # Line 533
    async def _compute_technical_features_async(self, market_data: pd.DataFrame, symbol: str) -> pd.DataFrame  # Line 593
    async def _compute_statistical_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame  # Line 638
    async def _compute_volume_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame  # Line 686
    def _compute_volume_features(self, data: pd.DataFrame) -> pd.DataFrame  # Line 693
    async def _compute_volatility_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame  # Line 731
    def _compute_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame  # Line 738
    async def _compute_momentum_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame  # Line 766
    def _compute_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame  # Line 773
    async def _compute_trend_features_async(self, market_data: pd.DataFrame) -> pd.DataFrame  # Line 821
    def _compute_trend_features(self, data: pd.DataFrame) -> pd.DataFrame  # Line 826
    async def select_features(self, ...) -> tuple[pd.DataFrame, list[str], dict[str, float]]  # Line 892
    async def _select_features_impl(self, ...) -> tuple[pd.DataFrame, list[str], dict[str, float]]  # Line 923
    async def _preprocess_features(self, features_df: pd.DataFrame, scaling_method: str = 'standard') -> tuple[pd.DataFrame, dict[str, Any]]  # Line 1005
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame  # Line 1066
    async def _service_health_check(self) -> 'HealthStatus'  # Line 1085
    async def get_feature_engineering_metrics(self) -> dict[str, Any]  # Line 1106
    async def clear_cache(self) -> dict[str, int]  # Line 1119
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 1145
```

### File: batch_predictor.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`

#### Class: `BatchPredictorConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for batch predictor service

```python
class BatchPredictorConfig(BaseModel):
```

#### Class: `BatchPredictorService`

**Inherits**: BaseService
**Purpose**: Simple batch prediction service for ML models

```python
class BatchPredictorService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None  # Line 34
    async def _do_start(self) -> None  # Line 58
    async def _do_stop(self) -> None  # Line 68
    async def predict_batch(self, ...) -> pd.DataFrame  # Line 72
    async def predict_multiple_symbols(self, model_name: str, data_dict: dict[str, pd.DataFrame], parallel: bool = True) -> dict[str, pd.DataFrame]  # Line 145
    async def _load_model(self, model_name: str)  # Line 172
    async def _save_predictions_to_db(self, predictions: pd.DataFrame, model_name: str, symbol: str) -> None  # Line 189
    async def _save_predictions_to_file(self, predictions: pd.DataFrame, output_file: str) -> None  # Line 214
    async def _service_health_check(self) -> 'HealthStatus'  # Line 232
    async def submit_batch_prediction(self, model_id: str, input_data: pd.DataFrame, **kwargs) -> str | None  # Line 244
    def get_job_status(self, job_id: str) -> dict[str, Any] | None  # Line 260
    def get_job_result(self, job_id: str) -> pd.DataFrame | None  # Line 270
    def list_jobs(self) -> list[dict[str, Any]]  # Line 274
```

### File: inference_engine.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.utils.constants import ML_MODEL_CONSTANTS`

#### Class: `InferenceConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for inference service

```python
class InferenceConfig(BaseModel):
```

#### Class: `InferencePredictionRequest`

**Inherits**: BaseModel
**Purpose**: Request object for predictions

```python
class InferencePredictionRequest(BaseModel):
```

#### Class: `InferencePredictionResponse`

**Inherits**: BaseModel
**Purpose**: Response object for predictions

```python
class InferencePredictionResponse(BaseModel):
```

#### Class: `InferenceMetrics`

**Inherits**: BaseModel
**Purpose**: Inference service performance metrics

```python
class InferenceMetrics(BaseModel):
```

#### Class: `InferenceService`

**Inherits**: BaseService
**Purpose**: Real-time inference service for ML models

```python
class InferenceService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None  # Line 100
    async def _do_start(self) -> None  # Line 150
    async def _do_stop(self) -> None  # Line 169
    async def predict(self, ...) -> InferencePredictionResponse  # Line 186
    async def _predict_impl(self, ...) -> InferencePredictionResponse  # Line 217
    async def predict_async(self, ...) -> InferencePredictionResponse  # Line 323
    async def predict_batch(self, requests: list[InferencePredictionRequest]) -> list[InferencePredictionResponse]  # Line 347
    async def _predict_batch_impl(self, requests: list[InferencePredictionRequest]) -> list[InferencePredictionResponse]  # Line 365
    async def _process_single_batch_request(self, model: Any, request: InferencePredictionRequest) -> InferencePredictionResponse  # Line 462
    async def predict_with_features(self, ...) -> InferencePredictionResponse  # Line 506
    async def _predict_with_features_impl(self, ...) -> InferencePredictionResponse  # Line 534
    async def _get_model(self, model_id: str, use_cache: bool = True) -> Any  # Line 592
    async def _make_prediction(self, model: Any, features: pd.DataFrame, return_probabilities: bool) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]  # Line 630
    def _predict_with_probabilities(self, model: Any, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]  # Line 657
    def _predict_without_probabilities(self, model: Any, features: pd.DataFrame) -> np.ndarray  # Line 672
    def _generate_prediction_cache_key(self, model_id: str, features: pd.DataFrame, return_probabilities: bool) -> str  # Line 680
    async def _get_cached_model(self, model_id: str) -> Any | None  # Line 688
    async def _cache_model(self, model_id: str, model: Any) -> None  # Line 703
    async def _get_cached_prediction(self, cache_key: str) -> InferencePredictionResponse | None  # Line 710
    async def _cache_prediction(self, cache_key: str, response: InferencePredictionResponse) -> None  # Line 723
    async def _clean_model_cache(self) -> None  # Line 732
    async def _clean_prediction_cache(self) -> None  # Line 747
    async def _batch_processor_loop(self) -> None  # Line 762
    async def _process_batch(self, batch: list[tuple[InferencePredictionRequest, asyncio.Future]]) -> None  # Line 800
    async def warm_up_models(self, model_ids: list[str]) -> dict[str, bool]  # Line 823
    async def _warm_up_models_impl(self, model_ids: list[str]) -> dict[str, bool]  # Line 843
    async def _warm_up_single_model(self, model_id: str) -> bool  # Line 874
    async def _service_health_check(self) -> Any  # Line 886
    def get_inference_metrics(self) -> dict[str, Any]  # Line 913
    async def clear_cache(self) -> dict[str, int]  # Line 932
    def reset_metrics(self) -> None  # Line 955
    def _preprocess_features(self, features: Any) -> np.ndarray  # Line 960
    def _postprocess_predictions(self, predictions: np.ndarray | None) -> list  # Line 982
    def _calculate_confidence_scores(self, probabilities: np.ndarray | None, predictions: np.ndarray | None = None) -> list | None  # Line 998
    def _update_metrics(self, processing_time: float, success: bool, cache_hit: bool) -> None  # Line 1013
    def get_metrics(self) -> dict[str, Any]  # Line 1034
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 1039
```

### File: model_cache.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.types.base import ConfigDict`
- `from src.utils.constants import ML_MODEL_CONSTANTS`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `ModelCacheConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for model cache service

```python
class ModelCacheConfig(BaseModel):
```

#### Class: `ModelCacheService`

**Inherits**: BaseService
**Purpose**: High-performance cache for ML models

```python
class ModelCacheService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None  # Line 51
    async def _do_start(self) -> None  # Line 101
    async def _do_stop(self) -> None  # Line 117
    def start_cleanup_thread(self) -> None  # Line 122
    def stop_cleanup_thread(self) -> None  # Line 130
    async def cache_model(self, model_id: str, model: Any) -> bool  # Line 139
    async def get_model(self, model_id: str) -> Any | None  # Line 188
    def remove_model(self, model_id: str) -> bool  # Line 231
    def clear_cache(self) -> None  # Line 244
    def get_cached_models(self) -> dict[str, dict[str, Any]]  # Line 258
    def get_cache_stats(self) -> dict[str, Any]  # Line 281
    def clear_stats(self) -> None  # Line 299
    def _make_space_for_model(self, required_memory_mb: float) -> None  # Line 308
    def _evict_lru_model(self, reason: str) -> None  # Line 323
    def _remove_model(self, model_id: str, reason: str = 'Unknown') -> bool  # Line 336
    def _estimate_model_memory(self, model: Any) -> float  # Line 364
    def _calculate_hit_rate(self) -> float  # Line 407
    def _cleanup_expired_models(self) -> int  # Line 412
    def _cleanup_loop(self) -> None  # Line 433
    def _check_memory_pressure(self) -> None  # Line 450
    def health_check(self) -> dict[str, Any]  # Line 474
    def __enter__(self) -> 'ModelCacheService'  # Line 503
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None  # Line 508
    async def _service_health_check(self) -> 'HealthStatus'  # Line 513
    def get_model_cache_metrics(self) -> dict[str, Any]  # Line 538
    def get_cache_size(self) -> int  # Line 552
    def get_cache_memory_usage(self) -> float  # Line 557
    def get_cache_statistics(self) -> dict[str, Any]  # Line 562
    def is_model_cached(self, model_id: str) -> bool  # Line 566
    def get_cached_model_ids(self) -> list[str]  # Line 571
    async def evict_model(self, model_id: str) -> bool  # Line 577
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 586
```

### File: integration_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.types.base import ConfigDict`
- `from src.ml.data_transformer import MLDataTransformer`

#### Class: `MLIntegrationService`

**Inherits**: BaseService
**Purpose**: Service for handling ML module integration with other modules

```python
class MLIntegrationService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 18
    def determine_target_processing_mode(self, target_module: str, operation_type: str) -> str  # Line 25
    def prepare_data_for_target_module(self, ...) -> dict[str, Any]  # Line 56
    def determine_integration_mode(self, target_module: str) -> str  # Line 98
    def validate_cross_module_compatibility(self, data: dict[str, Any], target_module: str) -> bool  # Line 118
```

### File: interfaces.py

#### Class: `IFeatureEngineeringService`

**Inherits**: ABC
**Purpose**: Interface for feature engineering service

```python
class IFeatureEngineeringService(ABC):
    async def compute_features(self, request: 'FeatureRequest') -> 'FeatureResponse'  # Line 29
    async def select_features(self, ...) -> tuple[pd.DataFrame, list[str], dict[str, float]]  # Line 34
    async def clear_cache(self) -> dict[str, int]  # Line 46
```

#### Class: `IModelRegistryService`

**Inherits**: ABC
**Purpose**: Interface for model registry service

```python
class IModelRegistryService(ABC):
    async def register_model(self, request: 'ModelRegistrationRequest') -> str  # Line 55
    async def load_model(self, request: 'ModelLoadRequest') -> dict[str, Any]  # Line 60
    async def list_models(self, ...) -> list[dict[str, Any]]  # Line 65
    async def promote_model(self, model_id: str, stage: str, description: str = '') -> bool  # Line 75
    async def deactivate_model(self, model_id: str, reason: str = '') -> bool  # Line 80
    async def delete_model(self, model_id: str, remove_files: bool = True) -> bool  # Line 85
    async def get_model_metrics(self, model_id: str) -> dict[str, Any]  # Line 90
```

#### Class: `IInferenceService`

**Inherits**: ABC
**Purpose**: Interface for inference service

```python
class IInferenceService(ABC):
    async def predict(self, ...) -> 'InferencePredictionResponse'  # Line 99
    async def predict_batch(self, requests: list['InferencePredictionRequest']) -> list['InferencePredictionResponse']  # Line 111
    async def predict_with_features(self, ...) -> 'InferencePredictionResponse'  # Line 118
    async def warm_up_models(self, model_ids: list[str]) -> dict[str, bool]  # Line 129
    async def clear_cache(self) -> dict[str, int]  # Line 134
```

#### Class: `IModelValidationService`

**Inherits**: ABC
**Purpose**: Interface for model validation service

```python
class IModelValidationService(ABC):
    async def validate_model_performance(self, model: Any, test_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, Any]  # Line 143
    async def validate_production_readiness(self, model: Any, validation_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, bool]  # Line 150
```

#### Class: `IDriftDetectionService`

**Inherits**: ABC
**Purpose**: Interface for drift detection service

```python
class IDriftDetectionService(ABC):
    async def detect_feature_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict[str, Any]  # Line 161
    async def detect_prediction_drift(self, ...) -> dict[str, Any]  # Line 168
    async def detect_performance_drift(self, ...) -> dict[str, Any]  # Line 175
```

#### Class: `ITrainingService`

**Inherits**: ABC
**Purpose**: Interface for training service

```python
class ITrainingService(ABC):
    async def train_model(self, ...) -> dict[str, Any]  # Line 189
    async def save_artifacts(self, ...) -> dict[str, Any]  # Line 200
```

#### Class: `IBatchPredictionService`

**Inherits**: ABC
**Purpose**: Interface for batch prediction service

```python
class IBatchPredictionService(ABC):
    async def process_batch_predictions(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]  # Line 215
```

#### Class: `IModelManagerService`

**Inherits**: ABC
**Purpose**: Interface for model manager service

```python
class IModelManagerService(ABC):
    async def create_and_train_model(self, ...) -> dict[str, Any]  # Line 226
    async def deploy_model(self, model_name: str, deployment_stage: str = 'production') -> dict[str, Any]  # Line 239
    async def monitor_model_performance(self, ...) -> dict[str, Any]  # Line 246
    async def retire_model(self, model_name: str, reason: str = 'replaced') -> dict[str, Any]  # Line 256
    def get_active_models(self) -> dict[str, Any]  # Line 261
    async def get_model_status(self, model_name: str) -> dict[str, Any] | None  # Line 266
    async def health_check(self) -> dict[str, Any]  # Line 271
```

#### Class: `IModelFactory`

**Inherits**: ABC
**Purpose**: Interface for model factory service

```python
class IModelFactory(ABC):
    def create_model(self, ...) -> Any  # Line 280
    def get_available_models(self) -> list[str]  # Line 292
    def register_custom_model(self, ...) -> None  # Line 297
```

#### Class: `IMLService`

**Inherits**: ABC
**Purpose**: Interface for main ML service that coordinates all ML operations

```python
class IMLService(ABC):
    async def process_pipeline(self, request: Any) -> Any  # Line 313
    async def train_model(self, request: Any) -> Any  # Line 318
    async def process_batch_pipeline(self, requests: list[Any]) -> list[Any]  # Line 323
    async def enhance_strategy_signals(self, ...) -> list  # Line 328
    async def list_available_models(self, model_type: str | None = None, stage: str | None = None) -> list[dict[str, Any]]  # Line 338
    async def promote_model(self, model_id: str, stage: str, description: str = '') -> bool  # Line 345
    async def get_model_info(self, model_id: str) -> dict[str, Any]  # Line 350
    async def clear_cache(self) -> dict[str, int]  # Line 355
    def get_ml_service_metrics(self) -> dict[str, Any]  # Line 360
```

### File: model_manager.py

**Key Imports:**
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.ml.interfaces import IModelManagerService`

#### Class: `ModelManagerConfig`

**Inherits**: PydanticBaseModel
**Purpose**: Configuration for model manager service

```python
class ModelManagerConfig(PydanticBaseModel):
```

#### Class: `ModelManagerService`

**Inherits**: BaseService, IModelManagerService
**Purpose**: Central manager for ML model lifecycle

```python
class ModelManagerService(BaseService, IModelManagerService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None  # Line 87
    async def _do_start(self) -> None  # Line 148
    async def _do_stop(self) -> None  # Line 197
    async def create_and_train_model(self, ...) -> dict[str, Any]  # Line 206
    async def deploy_model(self, model_name: str, deployment_stage: str = 'production') -> dict[str, Any]  # Line 323
    async def monitor_model_performance(self, ...) -> dict[str, Any]  # Line 423
    async def retire_model(self, model_name: str, reason: str = 'replaced') -> dict[str, Any]  # Line 578
    async def _prepare_and_train_model(self, ...) -> dict[str, Any]  # Line 632
    async def _validate_trained_model(self, model: Any, test_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, Any]  # Line 655
    async def _register_model(self, ...) -> dict[str, Any]  # Line 674
    async def _pre_deployment_validation(self, model: Any, model_info: dict[str, Any]) -> dict[str, Any]  # Line 716
    async def _start_model_monitoring(self, model_name: str, model: Any) -> None  # Line 771
    async def _stop_model_monitoring(self, model_name: str) -> None  # Line 786
    async def _generate_monitoring_alerts(self, ...) -> list[dict[str, Any]]  # Line 799
    def get_active_models(self) -> dict[str, Any]  # Line 826
    async def get_model_status(self, model_name: str) -> dict[str, Any] | None  # Line 838
    async def health_check(self) -> dict[str, Any]  # Line 851
    async def _create_model_instance(self, model_type: str, model_name: str, model_params: dict[str, Any]) -> Any  # Line 891
    async def _service_health_check(self) -> 'HealthStatus'  # Line 928
    def get_model_manager_metrics(self) -> dict[str, Any]  # Line 957
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 970
```

### File: base_model.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `BaseMLModelConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for ML models

```python
class BaseMLModelConfig(BaseModel):
```

#### Class: `BaseMLModel`

**Inherits**: BaseService, abc.ABC
**Purpose**: Abstract base class for all ML models in the trading system

```python
class BaseMLModel(BaseService, abc.ABC):
    def __init__(self, ...)  # Line 57
    async def _do_start(self) -> None  # Line 110
    async def _do_stop(self) -> None  # Line 129
    def _get_model_type(self) -> str  # Line 134
    def _create_model(self, **kwargs) -> Any  # Line 139
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame  # Line 143
    def _validate_targets(self, y: pd.Series) -> pd.Series  # Line 147
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]  # Line 152
    def prepare_data(self, ...) -> tuple[pd.DataFrame, pd.Series | None]  # Line 157
    def train(self, ...) -> dict[str, float]  # Line 203
    def predict(self, X: pd.DataFrame, return_probabilities: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]  # Line 307
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]  # Line 352
    async def save(self, filepath: str | Path) -> Path  # Line 391
    def load(cls, filepath: str | Path, config: ConfigDict | None = None) -> 'BaseMLModel'  # Line 451
    def get_feature_importance(self) -> pd.Series | None  # Line 510
    def get_model_info(self) -> dict[str, Any]  # Line 537
    def _calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str  # Line 567
    def __repr__(self) -> str  # Line 573
    async def _service_health_check(self) -> 'HealthStatus'  # Line 584
    def get_model_metrics(self) -> dict[str, Any]  # Line 601
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 616
```

### File: direction_classifier.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.ml.models.base_model import BaseMLModel`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `DirectionClassifier`

**Inherits**: BaseMLModel
**Purpose**: Direction classification model for predicting price movement direction

```python
class DirectionClassifier(BaseMLModel):
    def __init__(self, ...)  # Line 53
    def _get_model_type(self) -> str  # Line 104
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame  # Line 108
    def _validate_targets(self, y: pd.Series) -> pd.Series  # Line 113
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]  # Line 118
    def _create_model(self) -> Any  # Line 123
    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]  # Line 178
    def predict(self, X: pd.DataFrame) -> np.ndarray  # Line 264
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray  # Line 301
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]  # Line 338
    def _convert_to_direction_classes(self, price_data: pd.Series) -> np.ndarray  # Line 414
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float  # Line 446
    def get_feature_importance(self) -> pd.Series | None  # Line 464
    def get_class_distribution(self) -> dict[str, int] | None  # Line 477
    def predict_direction_labels(self, X: pd.DataFrame) -> list[str]  # Line 490
    def get_prediction_confidence(self, X: pd.DataFrame) -> np.ndarray  # Line 508
```

### File: price_predictor.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.ml.models.base_model import BaseMLModel`

#### Class: `PricePredictor`

**Inherits**: BaseMLModel
**Purpose**: Price prediction model for financial instruments

```python
class PricePredictor(BaseMLModel):
    def __init__(self, ...)  # Line 58
    def _get_model_type(self) -> str  # Line 102
    def _create_model(self, **kwargs: Any) -> Any  # Line 106
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame  # Line 207
    def _validate_targets(self, y: pd.Series) -> pd.Series  # Line 212
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]  # Line 217
    def create_target_from_prices(self, prices: pd.Series, target_type: str = 'return', horizon: int | None = None) -> pd.Series  # Line 222
    def predict_price_sequence(self, X: pd.DataFrame, sequence_length: int = 10) -> np.ndarray  # Line 249
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, transaction_cost: Decimal = Any) -> dict[str, Any]  # Line 281
    def get_feature_importance_analysis(self) -> dict[str, Any]  # Line 299
```

### File: regime_detector.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ValidationError`
- `from src.ml.models.base_model import BaseMLModel`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `RegimeDetector`

**Inherits**: BaseMLModel
**Purpose**: Market regime detection model for identifying market conditions

```python
class RegimeDetector(BaseMLModel):
    def __init__(self, ...)  # Line 47
    def _create_model(self) -> Any  # Line 103
    def _define_regime_names(self) -> list[str]  # Line 139
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]  # Line 151
    def predict(self, X: pd.DataFrame) -> np.ndarray  # Line 261
    def evaluate(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]  # Line 315
    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame  # Line 392
    def _calculate_regime_statistics(self, features: pd.DataFrame, regime_labels: np.ndarray) -> dict[str, Any]  # Line 494
    def predict_regime_labels(self, X: pd.DataFrame) -> list[str]  # Line 518
    def get_regime_probabilities(self, X: pd.DataFrame) -> np.ndarray | None  # Line 539
    def get_regime_statistics(self) -> dict[str, Any] | None  # Line 574
    def get_feature_importance(self) -> pd.Series | None  # Line 587
    def get_regime_names(self) -> list[str]  # Line 600
    def interpret_regime(self, regime_id: int) -> str  # Line 604
    def _get_model_type(self) -> str  # Line 610
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame  # Line 614
    def _validate_targets(self, y: pd.Series) -> pd.Series  # Line 620
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict  # Line 626
```

### File: storage.py

#### Class: `ModelStorageBackend`

**Inherits**: abc.ABC
**Purpose**: Abstract base class for model storage backends

```python
class ModelStorageBackend(abc.ABC):
    def save(self, model_data: dict[str, Any], filepath: Path) -> None  # Line 19
    def load(self, filepath: Path) -> dict[str, Any]  # Line 24
```

#### Class: `JoblibStorageBackend`

**Inherits**: ModelStorageBackend
**Purpose**: Joblib-based storage backend for sklearn models

```python
class JoblibStorageBackend(ModelStorageBackend):
    def save(self, model_data: dict[str, Any], filepath: Path) -> None  # Line 32
    def load(self, filepath: Path) -> dict[str, Any]  # Line 47
```

#### Class: `PickleStorageBackend`

**Inherits**: ModelStorageBackend
**Purpose**: Pickle-based storage backend for general Python objects

```python
class PickleStorageBackend(ModelStorageBackend):
    def save(self, model_data: dict[str, Any], filepath: Path) -> None  # Line 67
    def load(self, filepath: Path) -> dict[str, Any]  # Line 89
```

#### Class: `ModelStorageManager`

**Purpose**: Manager for model storage operations

```python
class ModelStorageManager:
    def __init__(self, backend: str = 'joblib')  # Line 116
    def save_model(self, model_data: dict[str, Any], filepath: str | Path) -> Path  # Line 135
    def load_model(self, filepath: str | Path) -> dict[str, Any]  # Line 150
```

### File: volatility_forecaster.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ValidationError`
- `from src.ml.models.base_model import BaseMLModel`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `VolatilityForecaster`

**Inherits**: BaseMLModel
**Purpose**: Volatility forecasting model for predicting future volatility

```python
class VolatilityForecaster(BaseMLModel):
    def __init__(self, ...)  # Line 49
    def _create_model(self) -> Any  # Line 104
    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]  # Line 141
    def predict(self, X: pd.DataFrame) -> np.ndarray  # Line 260
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]  # Line 308
    def _calculate_volatility_targets(self, price_data: pd.Series) -> np.ndarray  # Line 407
    def _calculate_realized_volatility(self, price_data: pd.Series) -> np.ndarray  # Line 423
    def _calculate_garch_volatility(self, price_data: pd.Series) -> np.ndarray  # Line 472
    def _calculate_intraday_volatility(self, price_data: pd.Series) -> np.ndarray  # Line 520
    def _calculate_volatility_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float  # Line 546
    def _calculate_directional_volatility_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float  # Line 561
    def _calculate_volatility_regime_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float  # Line 584
    def get_feature_importance(self) -> pd.Series | None  # Line 614
    def get_volatility_stats(self) -> dict[str, float] | None  # Line 627
    def predict_volatility_regime(self, X: pd.DataFrame) -> list[str]  # Line 640
    def _get_model_type(self) -> str  # Line 676
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame  # Line 680
    def _validate_targets(self, y: pd.Series) -> pd.Series  # Line 686
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict  # Line 692
```

### File: artifact_store.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `ArtifactStoreConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for artifact store service

```python
class ArtifactStoreConfig(BaseModel):
```

#### Class: `ArtifactStore`

**Inherits**: BaseService
**Purpose**: Artifact store service for managing ML model artifacts

```python
class ArtifactStore(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 60
    async def _do_start(self) -> None  # Line 97
    async def _do_stop(self) -> None  # Line 123
    async def store_artifact(self, ...) -> str  # Line 139
    async def _store_artifact_impl(self, ...) -> str  # Line 180
    async def retrieve_artifact(self, ...) -> Any  # Line 277
    async def _retrieve_artifact_impl(self, artifact_name: str, model_id: str, artifact_type: str, version: str | None) -> Any  # Line 308
    async def list_artifacts(self, ...) -> pd.DataFrame  # Line 410
    async def _list_artifacts_impl(self, model_id: str | None, artifact_type: str | None, version: str | None) -> pd.DataFrame  # Line 435
    async def delete_artifact(self, ...) -> bool  # Line 492
    async def _delete_artifact_impl(self, artifact_name: str, model_id: str, artifact_type: str, version: str | None) -> bool  # Line 523
    async def cleanup_old_artifacts(self, days_to_keep: int | None = None) -> int  # Line 606
    async def _cleanup_old_artifacts_impl(self, days_to_keep: int) -> int  # Line 622
    async def _save_artifact_data(self, artifact_data: Any, artifact_dir: Path, base_filename: str) -> Path  # Line 662
    def _save_json_data(self, data: dict, path: Path) -> None  # Line 716
    def _save_text_data(self, data: str, path: Path) -> None  # Line 726
    def _save_joblib_data(self, data: Any, path: Path) -> None  # Line 736
    async def _compress_artifact(self, artifact_path: Path) -> Path  # Line 753
    def _compress_file(self, input_path: Path, output_path: Path) -> None  # Line 766
    async def _calculate_file_hash(self, file_path: Path) -> str  # Line 780
    def _calculate_hash_sync(self, file_path: Path) -> str  # Line 785
    async def _save_artifact_metadata(self, artifact_dir: Path, base_filename: str, metadata: dict[str, Any]) -> None  # Line 798
    async def _load_json_file(self, file_path: Path) -> dict[str, Any]  # Line 806
    def _load_json_sync(self, file_path: Path) -> dict[str, Any]  # Line 811
    async def _find_metadata_file(self, metadata_files: list[Path], data_file: Path) -> Path | None  # Line 821
    async def _load_artifact_file(self, file_path: Path, compressed: bool = False) -> Any  # Line 828
    def _load_artifact_sync(self, file_path: Path, compressed: bool) -> Any  # Line 835
    def _load_by_extension(self, file_path: Path) -> Any  # Line 875
    async def _scan_artifact_directory(self, type_dir: Path, model_id: str | None, version: str | None) -> list[dict[str, Any]]  # Line 935
    async def _cleanup_type_directory(self, type_dir: Path, cutoff_date: datetime) -> int  # Line 991
    async def _log_audit_event(self, event_type: str, details: dict[str, Any]) -> None  # Line 1020
    async def _background_cleanup(self) -> None  # Line 1040
    async def _service_health_check(self) -> Any  # Line 1055
    def get_artifact_store_metrics(self) -> dict[str, Any]  # Line 1075
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 1095
```

### File: model_registry.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `ModelRegistryConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for model registry service

```python
class ModelRegistryConfig(BaseModel):
```

#### Class: `ModelMetadata`

**Inherits**: BaseModel
**Purpose**: Model metadata structure

```python
class ModelMetadata(BaseModel):
```

#### Class: `ModelRegistrationRequest`

**Inherits**: BaseModel
**Purpose**: Request for model registration

```python
class ModelRegistrationRequest(BaseModel):
```

#### Class: `ModelLoadRequest`

**Inherits**: BaseModel
**Purpose**: Request for model loading

```python
class ModelLoadRequest(BaseModel):
```

#### Class: `ModelRegistryService`

**Inherits**: BaseService
**Purpose**: Model registry service for managing ML model versions and storage

```python
class ModelRegistryService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 102
    async def _do_start(self) -> None  # Line 147
    async def _do_stop(self) -> None  # Line 168
    async def register_model(self, request: ModelRegistrationRequest) -> str  # Line 185
    async def _register_model_impl(self, request: ModelRegistrationRequest) -> str  # Line 205
    async def load_model(self, request: ModelLoadRequest) -> dict[str, Any]  # Line 286
    async def _load_model_impl(self, request: ModelLoadRequest) -> dict[str, Any]  # Line 305
    async def list_models(self, ...) -> list[dict[str, Any]]  # Line 385
    async def _list_models_impl(self, model_type: str | None, stage: str | None, active_only: bool) -> list[dict[str, Any]]  # Line 410
    async def promote_model(self, model_id: str, stage: str, description: str = '') -> bool  # Line 451
    async def _promote_model_impl(self, model_id: str, stage: str, description: str) -> bool  # Line 474
    async def deactivate_model(self, model_id: str, reason: str = '') -> bool  # Line 530
    async def _deactivate_model_impl(self, model_id: str, reason: str) -> bool  # Line 551
    async def delete_model(self, model_id: str, remove_files: bool = True) -> bool  # Line 597
    async def _delete_model_impl(self, model_id: str, remove_files: bool) -> bool  # Line 618
    async def get_model_metrics(self, model_id: str) -> dict[str, Any]  # Line 675
    async def _get_model_metrics_impl(self, model_id: str) -> dict[str, Any]  # Line 691
    async def _generate_model_id_and_version(self, name: str, model_type: str) -> tuple[str, str]  # Line 733
    async def _find_model_metadata(self, ...) -> ModelMetadata | None  # Line 754
    async def _get_model_metadata(self, model_id: str) -> ModelMetadata | None  # Line 802
    async def _store_model_metadata(self, metadata: ModelMetadata) -> None  # Line 825
    async def _update_model_metadata(self, metadata: ModelMetadata) -> None  # Line 830
    async def _save_model_to_file(self, model: Any, file_path: Path) -> None  # Line 835
    def _save_pickle_file(self, obj: Any, file_path: Path) -> None  # Line 846
    async def _load_model_from_file(self, file_path: Path) -> Any  # Line 853
    def _load_pickle_file(self, file_path: Path) -> Any  # Line 863
    def _load_json_file(self, file_path: Path) -> dict[str, Any]  # Line 870
    async def _save_registry_entry(self, metadata: ModelMetadata) -> None  # Line 875
    def _save_json_file(self, data: dict[str, Any], file_path: Path) -> None  # Line 896
    async def _load_existing_metadata(self) -> None  # Line 901
    async def _cache_model(self, model_id: str, metadata: ModelMetadata, model: Any) -> None  # Line 931
    async def _get_cached_model(self, model_id: str) -> Any | None  # Line 936
    async def _log_audit_event(self, event_type: str, model_id: str, details: dict[str, Any]) -> None  # Line 950
    async def _background_cleanup(self) -> None  # Line 977
    async def _clean_expired_cache(self) -> None  # Line 995
    async def _cleanup_old_versions(self) -> None  # Line 1024
    async def _service_health_check(self) -> Any  # Line 1059
    def get_model_registry_metrics(self) -> dict[str, Any]  # Line 1083
    async def clear_cache(self) -> dict[str, int]  # Line 1094
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 1114
```

### File: repository.py

**Key Imports:**
- `from src.core.base.repository import BaseRepository`
- `from src.core.exceptions import DatabaseError`
- `from src.core.exceptions import ServiceError`
- `from src.core.types.base import ConfigDict`
- `from src.utils.constants import ML_MODEL_CONSTANTS`

#### Class: `IMLRepository`

**Inherits**: ABC
**Purpose**: Interface for ML data repository

```python
class IMLRepository(ABC):
    async def store_model_metadata(self, metadata: dict[str, Any]) -> str  # Line 22
    async def get_model_by_id(self, model_id: str) -> dict[str, Any] | None  # Line 27
    async def get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]  # Line 32
    async def find_models(self, ...) -> list[dict[str, Any]]  # Line 37
    async def get_all_models(self, ...) -> list[dict[str, Any]]  # Line 49
    async def update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> bool  # Line 59
    async def delete_model(self, model_id: str) -> bool  # Line 64
    async def store_prediction(self, prediction_data: dict[str, Any]) -> str  # Line 69
    async def get_predictions(self, ...) -> list[dict[str, Any]]  # Line 74
    async def store_training_job(self, job_data: dict[str, Any]) -> str  # Line 86
    async def get_training_job(self, job_id: str) -> dict[str, Any] | None  # Line 91
    async def update_training_progress(self, job_id: str, progress: dict[str, Any]) -> bool  # Line 96
```

#### Class: `MLRepository`

**Inherits**: BaseRepository, IMLRepository
**Purpose**: ML repository implementation using actual database models

```python
class MLRepository(BaseRepository, IMLRepository):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 104
    async def _do_start(self) -> None  # Line 124
    async def store_model_metadata(self, metadata: dict[str, Any]) -> str  # Line 140
    async def get_model_by_id(self, model_id: str) -> dict[str, Any] | None  # Line 183
    async def get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]  # Line 205
    async def find_models(self, ...) -> list[dict[str, Any]]  # Line 232
    async def get_all_models(self, ...) -> list[dict[str, Any]]  # Line 282
    async def update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> bool  # Line 297
    async def delete_model(self, model_id: str) -> bool  # Line 324
    async def store_prediction(self, prediction_data: dict[str, Any]) -> str  # Line 350
    async def get_predictions(self, ...) -> list[dict[str, Any]]  # Line 372
    def _matches_prediction_criteria(self, prediction: dict[str, Any], criteria: dict[str, Any]) -> bool  # Line 404
    async def store_training_job(self, job_data: dict[str, Any]) -> str  # Line 413
    async def get_training_job(self, job_id: str) -> dict[str, Any] | None  # Line 431
    async def update_training_progress(self, job_id: str, progress: dict[str, Any]) -> bool  # Line 450
    async def store_audit_entry(self, category: str, entry: dict[str, Any]) -> bool  # Line 471
    async def _create_entity(self, entity: dict[str, Any]) -> str  # Line 478
    async def _get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None  # Line 482
    async def _update_entity(self, entity_id: str, entity: dict[str, Any]) -> bool  # Line 486
    async def _delete_entity(self, entity_id: str) -> bool  # Line 490
    async def _list_entities(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]  # Line 494
    async def _count_entities(self, filters: dict[str, Any] | None = None) -> int  # Line 510
```

### File: service.py

**Key Imports:**
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.service import BaseService`
- `from src.core.event_constants import InferenceEvents`
- `from src.core.event_constants import TrainingEvents`

#### Class: `MLServiceConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for ML service

```python
class MLServiceConfig(BaseModel):
```

#### Class: `MLPipelineRequest`

**Inherits**: BaseModel
**Purpose**: Request for ML pipeline processing

```python
class MLPipelineRequest(BaseModel):
```

#### Class: `MLPipelineResponse`

**Inherits**: BaseModel
**Purpose**: Response from ML pipeline processing

```python
class MLPipelineResponse(BaseModel):
```

#### Class: `MLTrainingRequest`

**Inherits**: BaseModel
**Purpose**: Request for ML model training

```python
class MLTrainingRequest(BaseModel):
```

#### Class: `MLTrainingResponse`

**Inherits**: BaseModel
**Purpose**: Response from ML model training

```python
class MLTrainingResponse(BaseModel):
```

#### Class: `MLService`

**Inherits**: BaseService, IMLService
**Purpose**: Main ML service coordinating all machine learning operations

```python
class MLService(BaseService, IMLService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None  # Line 151
    async def _do_start(self) -> None  # Line 205
    async def _do_stop(self) -> None  # Line 260
    async def process_pipeline(self, request: MLPipelineRequest) -> MLPipelineResponse  # Line 281
    async def _process_pipeline_impl(self, request: MLPipelineRequest) -> MLPipelineResponse  # Line 300
    async def train_model(self, request: MLTrainingRequest) -> MLTrainingResponse  # Line 649
    async def _train_model_impl(self, request: MLTrainingRequest) -> MLTrainingResponse  # Line 668
    async def _train_model_async(self, ...) -> tuple[Any, dict[str, Any], dict[str, Any]]  # Line 914
    def _train_model_sync(self, ...) -> tuple[Any, dict[str, Any], dict[str, Any]]  # Line 923
    def _calculate_metrics(self, y_true: Any, y_pred: Any, is_classification: bool) -> dict[str, Any]  # Line 977
    async def process_batch_pipeline(self, requests: list[MLPipelineRequest]) -> list[MLPipelineResponse]  # Line 1007
    async def _process_batch_pipeline_impl(self, requests: list[MLPipelineRequest]) -> list[MLPipelineResponse]  # Line 1025
    async def clear_ml_cache(self) -> dict[str, int]  # Line 1139
    async def list_available_models(self, model_type: str | None = None, stage: str | None = None) -> list[dict[str, Any]]  # Line 1149
    async def promote_model(self, model_id: str, stage: str, description: str = '') -> bool  # Line 1160
    async def get_model_info(self, model_id: str) -> dict[str, Any]  # Line 1167
    async def _service_health_check(self) -> HealthCheckResult  # Line 1175
    def get_ml_service_metrics(self) -> dict[str, Any]  # Line 1257
    async def clear_cache(self) -> dict[str, int]  # Line 1276
    async def enhance_strategy_signals(self, ...) -> list  # Line 1282
    async def _enhance_strategy_signals_impl(self, ...) -> list  # Line 1310
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 1425
```

### File: services.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.types.base import ConfigDict`
- `from src.ml.interfaces import IBatchPredictionService`
- `from src.ml.interfaces import IDriftDetectionService`
- `from src.ml.interfaces import IModelValidationService`

#### Class: `ModelValidationService`

**Inherits**: BaseService, IModelValidationService
**Purpose**: Mock model validation service

```python
class ModelValidationService(BaseService, IModelValidationService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 26
    async def validate_model_performance(self, model: Any, test_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, Any]  # Line 33
    async def validate_production_readiness(self, model: Any, validation_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, bool]  # Line 54
```

#### Class: `DriftDetectionService`

**Inherits**: BaseService, IDriftDetectionService
**Purpose**: Mock drift detection service

```python
class DriftDetectionService(BaseService, IDriftDetectionService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 69
    async def get_reference_data(self, data_type: str) -> Any  # Line 77
    async def set_reference_data(self, data: pd.DataFrame, data_type: str) -> None  # Line 81
    async def detect_feature_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict[str, Any]  # Line 85
    async def detect_prediction_drift(self, ...) -> dict[str, Any]  # Line 97
    async def detect_performance_drift(self, ...) -> dict[str, Any]  # Line 109
```

#### Class: `TrainingService`

**Inherits**: BaseService, ITrainingService
**Purpose**: Mock training service

```python
class TrainingService(BaseService, ITrainingService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 128
    async def train_model(self, ...) -> dict[str, Any]  # Line 135
    async def save_artifacts(self, ...) -> dict[str, Any]  # Line 151
```

#### Class: `BatchPredictionService`

**Inherits**: BaseService, IBatchPredictionService
**Purpose**: Mock batch prediction service

```python
class BatchPredictionService(BaseService, IBatchPredictionService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 170
    async def process_batch_predictions(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]  # Line 177
```

### File: feature_store.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.core.types.data import FeatureSet`

#### Class: `FeatureStoreConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for feature store service

```python
class FeatureStoreConfig(BaseModel):
```

#### Class: `FeatureStoreMetadata`

**Inherits**: BaseModel
**Purpose**: Feature store metadata structure

```python
class FeatureStoreMetadata(BaseModel):
```

#### Class: `FeatureStoreRequest`

**Inherits**: BaseModel
**Purpose**: Request for feature store operations

```python
class FeatureStoreRequest(BaseModel):
```

#### Class: `FeatureStoreResponse`

**Inherits**: BaseModel
**Purpose**: Response from feature store operations

```python
class FeatureStoreResponse(BaseModel):
```

#### Class: `FeatureStoreService`

**Inherits**: BaseService
**Purpose**: Feature store service for centralized ML feature management

```python
class FeatureStoreService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 106
    async def _do_start(self) -> None  # Line 149
    def _get_data_service(self)  # Line 176
    async def _do_stop(self) -> None  # Line 180
    async def store_features(self, ...) -> FeatureStoreResponse  # Line 197
    async def _store_features_impl(self, request: FeatureStoreRequest, version: str | None) -> FeatureStoreResponse  # Line 236
    async def retrieve_features(self, ...) -> FeatureStoreResponse  # Line 352
    async def _retrieve_features_impl(self, request: FeatureStoreRequest) -> FeatureStoreResponse  # Line 388
    async def list_feature_sets(self, ...) -> FeatureStoreResponse  # Line 492
    async def _list_feature_sets_impl(self, request: FeatureStoreRequest, include_expired: bool, limit: int | None) -> FeatureStoreResponse  # Line 522
    async def delete_features(self, ...) -> FeatureStoreResponse  # Line 579
    async def _delete_features_impl(self, request: FeatureStoreRequest, delete_all_versions: bool) -> FeatureStoreResponse  # Line 612
    async def _validate_feature_set(self, feature_set: FeatureSet) -> dict[str, Any]  # Line 665
    async def _compute_feature_statistics(self, feature_set: FeatureSet) -> dict[str, Any]  # Line 696
    def _compute_stats_sync(self, df: pd.DataFrame) -> dict[str, Any]  # Line 714
    async def _generate_version(self, symbol: str, feature_set_id: str) -> str  # Line 741
    async def _generate_data_hash(self, feature_set: FeatureSet) -> str  # Line 767
    async def _prepare_feature_data(self, feature_set: FeatureSet, compress: bool) -> dict[str, Any]  # Line 779
    async def _reconstruct_feature_set(self, feature_data: dict[str, Any], metadata: FeatureStoreMetadata) -> FeatureSet  # Line 818
    def _generate_cache_key(self, symbol: str, feature_set_id: str | None, version: str | None) -> str  # Line 837
    async def _cache_features(self, feature_set: FeatureSet, metadata: FeatureStoreMetadata) -> None  # Line 843
    async def _get_cached_features(self, cache_key: str) -> tuple[FeatureSet, FeatureStoreMetadata] | None  # Line 859
    async def _remove_from_cache(self, symbol: str, feature_set_id: str, version: str | None) -> None  # Line 876
    async def _cleanup_old_versions(self, symbol: str, feature_set_id: str) -> None  # Line 889
    async def _background_cleanup(self) -> None  # Line 932
    async def _clean_expired_cache(self) -> None  # Line 949
    async def _clean_expired_feature_sets(self) -> None  # Line 975
    async def _service_health_check(self) -> Any  # Line 987
    def get_feature_store_metrics(self) -> dict[str, Any]  # Line 1006
    async def clear_cache(self) -> dict[str, int]  # Line 1019
    def _validate_service_config(self, config: ConfigDict) -> bool  # Line 1039
```

### File: cross_validation.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.ml.models.base_model import BaseMLModel`

#### Class: `TimeSeriesValidator`

**Purpose**: Time series specific cross-validation strategies

```python
class TimeSeriesValidator:
    def purged_walk_forward_split(data, ...) -> Generator[tuple[np.ndarray, np.ndarray], None, None]  # Line 49
    def combinatorial_purged_cross_validation(data, ...) -> Generator[tuple[np.ndarray, np.ndarray], None, None]  # Line 90
    def walk_forward_split(data: pd.DataFrame, min_train_size: int, test_size: int, step_size: int = 1) -> Generator[tuple[np.ndarray, np.ndarray], None, None]  # Line 144
    def expanding_window_split(data: pd.DataFrame, min_train_size: int, test_size: int, step_size: int = 1) -> Generator[tuple[np.ndarray, np.ndarray], None, None]  # Line 169
    def sliding_window_split(data: pd.DataFrame, train_size: int, test_size: int, step_size: int = 1) -> Generator[tuple[np.ndarray, np.ndarray], None, None]  # Line 194
```

#### Class: `CrossValidationService`

**Inherits**: BaseService
**Purpose**: Cross-validation service for ML models

```python
class CrossValidationService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 230
    async def validate_model(self, ...) -> dict[str, Any]  # Line 257
    async def time_series_validation(self, ...) -> dict[str, Any]  # Line 367
    async def nested_cross_validation(self, ...) -> dict[str, Any]  # Line 517
    def _create_cv_splitter(self, cv_strategy: str, cv_folds: int, y: pd.Series, **kwargs)  # Line 665
    def _manual_cross_validation(self, ...) -> dict[str, np.ndarray]  # Line 676
    def _calculate_score(self, y_true, y_pred, scoring: str) -> float  # Line 718
    def _calculate_sharpe_ratio(self, y_true: np.ndarray, y_pred: np.ndarray, risk_free_rate: float = 0.02) -> Decimal  # Line 753
    def _calculate_information_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> Decimal  # Line 790
    def _calculate_calmar_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> Decimal  # Line 819
    def _calculate_max_drawdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> Decimal  # Line 844
    def _calculate_hit_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float  # Line 886
    def _calculate_profit_factor(self, y_true: np.ndarray, y_pred: np.ndarray) -> Decimal  # Line 905
    def _process_cv_results(self, ...) -> dict[str, Any]  # Line 930
    async def _do_start(self) -> None  # Line 991
    async def _do_stop(self) -> None  # Line 996
    async def _service_health_check(self) -> 'HealthStatus'  # Line 1001
    def get_validation_history(self) -> list[dict[str, Any]]  # Line 1011
    def clear_history(self) -> None  # Line 1015
```

### File: hyperparameter_optimization.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`

#### Class: `HyperparameterOptimizationService`

**Inherits**: BaseService
**Purpose**: Simple hyperparameter optimization service using Optuna

```python
class HyperparameterOptimizationService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None  # Line 24
    async def optimize_model(self, ...) -> dict[str, Any]  # Line 41
    def _create_objective_function(self, ...) -> Callable  # Line 121
    def _get_default_parameter_space(self, model_class: type) -> dict[str, Any]  # Line 193
    async def _do_start(self) -> None  # Line 212
    async def _do_stop(self) -> None  # Line 217
    async def _service_health_check(self) -> 'HealthStatus'  # Line 222
```

### File: trainer.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ModelError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.ml.models.base_model import BaseMLModel`

#### Class: `TrainingPipeline`

**Purpose**: Training pipeline for managing data preparation and model training flow

```python
class TrainingPipeline:
    def __init__(self, steps: list[tuple[str, Any]])  # Line 33
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TrainingPipeline'  # Line 43
    def transform(self, X: pd.DataFrame) -> pd.DataFrame  # Line 56
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame  # Line 69
```

#### Class: `ModelTrainingService`

**Inherits**: BaseService
**Purpose**: Training orchestration service for ML models

```python
class ModelTrainingService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 89
    async def train_model(self, ...) -> dict[str, Any]  # Line 122
    async def batch_train_models(self, ...) -> list[dict[str, Any]]  # Line 273
    async def _prepare_features(self, ...) -> pd.DataFrame  # Line 318
    def _align_data(self, features_df: pd.DataFrame, targets: pd.Series) -> tuple[pd.DataFrame, pd.Series]  # Line 350
    def _split_data(self, ...) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]  # Line 378
    async def _process_features(self, ...) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]  # Line 410
    def _train_model(self, ...) -> dict[str, float]  # Line 470
    def _evaluate_model(self, model: BaseMLModel, test_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, float]  # Line 488
    def _save_training_artifacts(self, ...) -> None  # Line 501
    def _register_trained_model(self, model: BaseMLModel, metrics: dict[str, float], symbol: str) -> str | None  # Line 570
    def get_training_history(self) -> list[dict[str, Any]]  # Line 593
    def get_best_model_by_metric(self, ...) -> dict[str, Any] | None  # Line 597
    async def _do_start(self) -> None  # Line 644
    async def _do_stop(self) -> None  # Line 649
    async def _service_health_check(self) -> 'HealthStatus'  # Line 654
    def clear_history(self) -> None  # Line 664
```

### File: ab_testing.py

**Key Imports:**
- `from src.base import BaseComponent`
- `from src.core.exceptions import ValidationError`
- `from src.utils.decimal_utils import to_decimal`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `ABTestStatus`

**Inherits**: Enum
**Purpose**: A/B Test status enumeration

```python
class ABTestStatus(Enum):
```

#### Class: `ABTestVariant`

**Purpose**: Represents a variant in an A/B test

```python
class ABTestVariant:
    def __init__(self, ...)  # Line 41
```

#### Class: `ABTest`

**Purpose**: Represents an A/B test for ML model evaluation

```python
class ABTest:
    def __init__(self, ...)  # Line 65
```

#### Class: `ABTestFramework`

**Inherits**: BaseComponent
**Purpose**: A/B Testing Framework for ML Model Deployment

```python
class ABTestFramework(BaseComponent):
    def __init__(self, ...)  # Line 121
    def create_ab_test(self, ...) -> str  # Line 159
    def start_ab_test(self, test_id: str) -> bool  # Line 284
    def assign_variant(self, test_id: str, user_id: str) -> str  # Line 327
    async def record_result(self, ...) -> bool  # Line 380
    def analyze_ab_test(self, test_id: str) -> dict[str, Any]  # Line 448
    def _validate_test_configuration(self, test: ABTest) -> None  # Line 554
    def _hash_based_assignment(self, test: ABTest, user_id: str) -> str  # Line 569
    def _random_assignment(self, test: ABTest) -> str  # Line 588
    def _update_trading_metrics(self, variant: ABTestVariant) -> None  # Line 601
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: Decimal = Any) -> float  # Line 630
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float  # Line 647
    def _calculate_profit_factor(self, returns: np.ndarray) -> float  # Line 664
    async def _check_early_stopping(self, test: ABTest) -> None  # Line 679
    def _check_risk_controls(self, test: ABTest, variant_id: str) -> None  # Line 707
    def _calculate_test_duration(self, test: ABTest) -> float  # Line 754
    def _calculate_statistical_power(self, test: ABTest, control: ABTestVariant, treatment: ABTestVariant) -> dict[str, Any]  # Line 762
    def _compare_variant_performance(self, test: ABTest, control: ABTestVariant, treatment: ABTestVariant) -> dict[str, Any]  # Line 828
    def _compare_trading_metrics(self, control: ABTestVariant, treatment: ABTestVariant) -> dict[str, Any]  # Line 863
    def _perform_significance_tests(self, test: ABTest, control: ABTestVariant, treatment: ABTestVariant) -> dict[str, Any]  # Line 911
    def _generate_test_recommendation(self, test: ABTest, analysis_result: dict[str, Any]) -> dict[str, Any]  # Line 983
    async def stop_ab_test(self, test_id: str, reason: str = 'manual_stop') -> bool  # Line 1149
    def get_active_tests(self) -> dict[str, dict[str, Any]]  # Line 1201
    async def get_test_results(self, test_id: str) -> dict[str, Any]  # Line 1226
```

### File: drift_detector.py

**Key Imports:**
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `DriftDetectionService`

**Inherits**: BaseService
**Purpose**: Drift detection system for monitoring data and model drift

```python
class DriftDetectionService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None) -> None  # Line 44
    async def _do_start(self) -> None  # Line 84
    async def _do_stop(self) -> None  # Line 96
    async def _service_health_check(self) -> HealthStatus  # Line 100
    async def detect_feature_drift(self, ...) -> dict[str, Any]  # Line 122
    async def _detect_feature_drift_impl(self, ...) -> dict[str, Any]  # Line 150
    async def detect_prediction_drift(self, ...) -> dict[str, Any]  # Line 234
    async def _detect_prediction_drift_impl(self, ...) -> dict[str, Any]  # Line 259
    async def detect_performance_drift(self, ...) -> dict[str, Any]  # Line 348
    async def _detect_performance_drift_impl(self, ...) -> dict[str, Any]  # Line 376
    def _detect_single_feature_drift(self, ...) -> dict[str, Any]  # Line 464
    def _detect_numerical_drift(self, reference_data: pd.Series, current_data: pd.Series, feature_name: str) -> dict[str, Any]  # Line 487
    def _detect_categorical_drift(self, reference_data: pd.Series, current_data: pd.Series, feature_name: str) -> dict[str, Any]  # Line 532
    def _calculate_distribution_stats(self, data: pd.Series) -> dict[str, float]  # Line 578
    def _calculate_js_distance(self, ref_data: pd.Series, curr_data: pd.Series) -> float  # Line 596
    def _calculate_js_divergence_categorical(self, ref_dist: pd.Series, curr_dist: pd.Series) -> float  # Line 626
    async def _store_drift_result(self, drift_result: dict[str, Any])  # Line 649
    def get_drift_history(self, ...) -> list[dict[str, Any]]  # Line 672
    def set_reference_data(self, reference_data: pd.DataFrame, data_type: str = 'features') -> None  # Line 720
    def get_reference_data(self, data_type: str = 'features') -> pd.DataFrame | None  # Line 754
    def clear_reference_data(self, data_type: str | None = None) -> None  # Line 766
    async def _trigger_drift_alert(self, drift_result: dict[str, Any])  # Line 780
    async def continuous_monitoring(self, ...) -> dict[str, Any]  # Line 805
    async def _continuous_monitoring_impl(self, ...) -> dict[str, Any]  # Line 836
    def _generate_drift_recommendations(self, monitoring_results: dict[str, Any]) -> list[str]  # Line 944
```

### File: model_validator.py

**Key Imports:**
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`
- `from src.ml.models.base_model import BaseMLModel`

#### Class: `ModelValidationService`

**Inherits**: BaseService
**Purpose**: Comprehensive model validation system

```python
class ModelValidationService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 54
    async def _do_start(self) -> None  # Line 98
    async def _do_stop(self) -> None  # Line 110
    async def _service_health_check(self) -> HealthStatus  # Line 114
    async def validate_model_performance(self, ...) -> dict[str, Any]  # Line 136
    async def _validate_model_performance_impl(self, ...) -> dict[str, Any]  # Line 167
    async def validate_model_stability(self, ...) -> dict[str, Any]  # Line 269
    async def _validate_model_stability_impl(self, ...) -> dict[str, Any]  # Line 297
    async def validate_production_readiness(self, model: BaseMLModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]  # Line 401
    async def _validate_production_readiness_impl(self, model: BaseMLModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]  # Line 426
    async def detect_overfitting(self, ...) -> dict[str, Any]  # Line 513
    async def _detect_overfitting_impl(self, ...) -> dict[str, Any]  # Line 557
    def _analyze_performance_gaps(self, ...) -> dict[str, Any]  # Line 655
    def _analyze_learning_curves(self, ...) -> dict[str, Any]  # Line 716
    async def _analyze_feature_importance_stability(self, ...) -> dict[str, Any]  # Line 769
    def _analyze_model_complexity(self, model: BaseMLModel, X_train: pd.DataFrame) -> dict[str, Any]  # Line 841
    def _analyze_cv_stability(self, model: BaseMLModel, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]  # Line 874
    def _estimate_model_parameters(self, model: BaseMLModel) -> int  # Line 917
    def _calculate_overfitting_risk(self, overfitting_indicators: dict[str, Any]) -> tuple[float, str]  # Line 960
    def _generate_overfitting_recommendations(self, overfitting_indicators: dict[str, Any], risk_level: str) -> list[str]  # Line 1027
    def _get_primary_risk_indicators(self, overfitting_indicators: dict[str, Any]) -> list[str]  # Line 1115
    def get_overfitting_alerts(self) -> list[dict[str, Any]]  # Line 1138
    def clear_overfitting_alerts(self) -> None  # Line 1142
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]  # Line 1147
    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]  # Line 1163
    def _test_statistical_significance(self, y_true: pd.Series, y_pred: np.ndarray, model_type: str) -> dict[str, Any]  # Line 1181
    def _analyze_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]  # Line 1221
    def _analyze_performance_trend(self, performance_over_time: list[dict]) -> dict[str, Any]  # Line 1259
    def _check_prediction_consistency(self, model: BaseMLModel, X_test: pd.DataFrame) -> dict[str, bool]  # Line 1294
    def _check_computational_efficiency(self, model: BaseMLModel, X_test: pd.DataFrame) -> dict[str, bool]  # Line 1312
    def _check_error_handling(self, model: BaseMLModel, X_test: pd.DataFrame) -> dict[str, bool]  # Line 1356
    def _check_data_quality_handling(self, model: BaseMLModel) -> dict[str, bool]  # Line 1410
    def get_validation_history(self) -> list[dict[str, Any]]  # Line 1436
    def get_benchmark_results(self) -> dict[str, Any]  # Line 1440
    def clear_validation_history(self) -> None  # Line 1444
```

### File: validation_service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.core.types.base import ConfigDict`

#### Class: `MLValidationService`

**Inherits**: BaseService
**Purpose**: Service for ML-specific business logic validation

```python
class MLValidationService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 18
    def validate_ml_operation_type(self, ml_operation_type: str) -> bool  # Line 25
    def validate_ml_request_data(self, data: dict[str, Any]) -> dict[str, Any]  # Line 53
    def validate_model_parameters(self, model_type: str, parameters: dict[str, Any]) -> bool  # Line 87
    def validate_feature_data(self, feature_data: Any) -> bool  # Line 127
```

---
**Generated**: Complete reference for ml module
**Total Classes**: 75
**Total Functions**: 2