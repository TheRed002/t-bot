# DATA Module Reference

## INTEGRATION
**Dependencies**: core, database, error_handling, monitoring, utils
**Used By**: None
**Provides**: DataController, DataMonitoringService, DataService, DataStorageManager, EnvironmentAwareDataManager, MLDataService, StreamingDataService
**Patterns**: Async Operations, Circuit Breaker, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Financial data handling
- Decimal precision arithmetic
- Database decimal columns
**Security**:
- Credential management
**Performance**:
- Caching
- Parallel execution
- Caching
**Architecture**:
- DataController inherits from base architecture
- ServiceRegistry inherits from base architecture
- L1MemoryCache inherits from base architecture

## MODULE OVERVIEW
**Files**: 40 Python files
**Classes**: 123
**Functions**: 20

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `CacheConfig` ✅

**Inherits**: BaseModel
**Purpose**: Cache configuration model
**Status**: Complete

### Implementation: `L1MemoryCache` ✅

**Inherits**: BaseComponent
**Purpose**: L1 Memory Cache - Fastest access, limited capacity
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> Any | None` - Line 81
- `async set(self, key: str, value: Any, ttl: int | None = None) -> bool` - Line 106
- `async delete(self, key: str) -> bool` - Line 145
- `async clear(self) -> None` - Line 154
- `async get_stats(self) -> CacheStats` - Line 220

### Implementation: `L2RedisCache` ✅

**Inherits**: BaseComponent
**Purpose**: L2 Redis Cache - Network-based, persistent, shared across instances
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 247
- `async get(self, key: str) -> Any | None` - Line 278
- `async set(self, key: str, value: Any, ttl: int | None = None) -> bool` - Line 303
- `async delete(self, key: str) -> bool` - Line 324
- `async clear(self) -> None` - Line 341
- `async batch_get(self, keys: list[str]) -> dict[str, Any]` - Line 356
- `async batch_set(self, data: dict[str, Any], ttl: int | None = None) -> bool` - Line 381
- `async get_stats(self) -> CacheStats` - Line 446
- `async cleanup(self) -> None` - Line 450

### Implementation: `DataCache` ✅

**Inherits**: BaseComponent
**Purpose**: Multi-level data cache orchestrator
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 545
- `async get(self, key: str, warm_lower_levels: bool = True) -> Any | None` - Line 569
- `async set(self, ...) -> bool` - Line 609
- `async delete(self, key: str, levels: list[CacheLevel] | None = None) -> bool` - Line 649
- `async batch_get(self, keys: list[str]) -> dict[str, Any]` - Line 682
- `async batch_set(self, data: dict[str, Any], ttl: int | None = None) -> bool` - Line 716
- `async clear(self, levels: list[CacheLevel] | None = None) -> None` - Line 741
- `async warm_cache(self, data_loader: Callable[[list[str]], dict[str, Any]], keys: list[str]) -> None` - Line 758
- `async get_stats(self) -> dict[str, CacheStats]` - Line 817
- `async health_check(self) -> HealthCheckResult` - Line 831
- `async cleanup(self) -> None` - Line 874

### Implementation: `RedisCache` ✅

**Inherits**: BaseComponent, DataCacheInterface
**Purpose**: Redis cache implementation
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 39
- `async get(self, key: str) -> Any | None` - Line 65
- `async set(self, key: str, value: Any, ttl: int | None = None) -> None` - Line 81
- `async delete(self, key: str) -> bool` - Line 98
- `async clear(self) -> None` - Line 111
- `async exists(self, key: str) -> bool` - Line 123
- `async health_check(self) -> dict[str, Any]` - Line 136
- `async cleanup(self) -> None` - Line 161

### Implementation: `DataController` ✅

**Inherits**: BaseComponent
**Purpose**: Data controller for handling data operations requests
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 50
- `async store_market_data_request(self, data: MarketData | list[MarketData], exchange: str, validate: bool = True) -> dict[str, Any]` - Line 63
- `async get_market_data_request(self, ...) -> dict[str, Any]` - Line 110
- `async get_data_count_request(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]` - Line 175
- `async get_recent_data_request(self, ...) -> dict[str, Any]` - Line 212
- `async get_health_status_request(self) -> dict[str, Any]` - Line 273
- `async cleanup(self) -> None` - Line 292

### Implementation: `DataQualityLevel` ✅

**Inherits**: Enum
**Purpose**: Data quality levels for different environments
**Status**: Complete

### Implementation: `DataStorageStrategy` ✅

**Inherits**: Enum
**Purpose**: Data storage strategies for different environments
**Status**: Complete

### Implementation: `EnvironmentAwareDataConfiguration` ✅

**Purpose**: Environment-specific data configuration
**Status**: Complete

**Implemented Methods:**
- `get_sandbox_data_config() -> dict[str, Any]` - Line 44
- `get_live_data_config() -> dict[str, Any]` - Line 70

### Implementation: `EnvironmentAwareDataManager` ✅

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware data management functionality
**Status**: Complete

**Implemented Methods:**
- `get_environment_data_config(self, exchange: str) -> dict[str, Any]` - Line 144
- `async validate_market_data_for_environment(self, market_data: MarketData, exchange: str) -> bool` - Line 157
- `async store_market_data_by_environment(self, market_data: MarketData, exchange: str) -> bool` - Line 299
- `async get_environment_aware_market_data(self, ...) -> list[MarketData]` - Line 377
- `get_environment_data_metrics(self, exchange: str) -> dict[str, Any]` - Line 589

### Implementation: `DataEventType` ✅

**Inherits**: Enum
**Purpose**: Types of data events - aligned with core event constants
**Status**: Complete

### Implementation: `DataEvent` ✅

**Purpose**: Data event structure aligned with database service messaging patterns
**Status**: Complete

### Implementation: `DataEventPublisher` ✅

**Purpose**: Simple mixin class to add basic event publishing capabilities
**Status**: Complete

**Implemented Methods:**

### Implementation: `DataEventSubscriber` ✅

**Purpose**: Base class for data event subscribers
**Status**: Complete

**Implemented Methods:**
- `async handle_data_event(self, event: DataEvent) -> None` - Line 139

### Implementation: `DataServiceFactory` ✅

**Purpose**: Factory for creating data services with proper dependency injection
**Status**: Complete

**Implemented Methods:**
- `create_data_service(self, use_cache: bool = True, use_validator: bool = True) -> DataServiceInterface` - Line 38
- `create_minimal_data_service(self) -> DataServiceInterface` - Line 55
- `create_testing_data_service(self, mock_storage = None, mock_cache = None, mock_validator = None) -> DataServiceInterface` - Line 64
- `create_data_storage(self) -> 'DataStorageInterface'` - Line 100
- `create_data_cache(self) -> 'DataCacheInterface'` - Line 104
- `create_data_validator(self) -> 'DataValidatorInterface'` - Line 108
- `create_market_data_source(self) -> 'MarketDataSource'` - Line 112
- `create_vectorized_processor(self) -> 'VectorizedProcessor'` - Line 116

### Implementation: `AlternativeFeatureType` ✅

**Inherits**: Enum
**Purpose**: Alternative feature type enumeration
**Status**: Complete

### Implementation: `SentimentStrength` ✅

**Inherits**: Enum
**Purpose**: Sentiment strength enumeration
**Status**: Complete

### Implementation: `AlternativeConfig` ✅

**Purpose**: Alternative feature calculation configuration
**Status**: Complete

### Implementation: `AlternativeResult` ✅

**Purpose**: Alternative feature calculation result
**Status**: Complete

### Implementation: `AlternativeFeatureCalculator` ✅

**Purpose**: Comprehensive alternative data feature calculator
**Status**: Complete

**Implemented Methods:**
- `set_data_sources(self, news_source = None, social_source = None, alt_data_source = None) -> None` - Line 164
- `async initialize(self) -> None` - Line 178
- `async calculate_news_sentiment(self, symbol: str, lookback_hours: int | None = None) -> AlternativeResult` - Line 194
- `async calculate_social_sentiment(self, symbol: str, lookback_hours: int | None = None) -> AlternativeResult` - Line 322
- `async calculate_economic_indicators(self, symbol: str, lookback_hours: int | None = None) -> AlternativeResult` - Line 482
- `async calculate_market_microstructure(self, symbol: str, lookback_hours: int | None = None) -> AlternativeResult` - Line 632
- `async calculate_batch_features(self, symbol: str, features: list[str]) -> dict[str, AlternativeResult]` - Line 677
- `async get_calculation_summary(self) -> dict[str, Any]` - Line 723

### Implementation: `FeatureType` ✅

**Inherits**: Enum
**Purpose**: Feature type enumeration
**Status**: Complete

### Implementation: `CalculationStatus` ✅

**Inherits**: Enum
**Purpose**: Feature calculation status
**Status**: Complete

### Implementation: `FeatureMetadata` ✅

**Purpose**: Feature metadata for tracking and versioning
**Status**: Complete

### Implementation: `FeatureValue` ✅

**Purpose**: Feature calculation result
**Status**: Complete

### Implementation: `FeatureRequest` ✅

**Inherits**: BaseModel
**Purpose**: Feature calculation request model
**Status**: Complete

**Implemented Methods:**
- `validate_feature_types(cls, v)` - Line 103

### Implementation: `FeatureCalculationPipeline` ✅

**Purpose**: Feature calculation pipeline for efficient batch processing
**Status**: Complete

**Implemented Methods:**
- `async calculate_batch(self, requests: list[FeatureRequest]) -> dict[str, list[FeatureValue]]` - Line 117

### Implementation: `FeatureStore` ✅

**Inherits**: BaseComponent
**Purpose**: Enterprise-grade FeatureStore for financial feature management
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 239
- `async register_feature(self, ...) -> None` - Line 365
- `async calculate_features(self, request: FeatureRequest) -> list[FeatureValue]` - Line 392
- `get_metrics(self) -> dict[str, Any]` - Line 948
- `async health_check(self) -> HealthCheckResult` - Line 970
- `async cleanup(self) -> None` - Line 995

### Implementation: `StatFeatureType` ✅

**Inherits**: Enum
**Purpose**: Statistical feature type enumeration
**Status**: Complete

### Implementation: `RegimeType` ✅

**Inherits**: Enum
**Purpose**: Market regime type enumeration
**Status**: Complete

### Implementation: `StatisticalConfig` ✅

**Purpose**: Statistical feature calculation configuration
**Status**: Complete

### Implementation: `StatisticalResult` ✅

**Purpose**: Statistical feature calculation result
**Status**: Complete

### Implementation: `StatisticalFeatures` ✅

**Inherits**: BaseComponent
**Purpose**: REFACTORED: Statistical feature calculator integrated with FeatureStore architecture
**Status**: Complete

**Implemented Methods:**
- `async add_market_data(self, data: MarketData) -> None` - Line 152
- `async calculate_rolling_stats(self, symbol: str, window: int | None = None, field: str = 'returns') -> StatisticalResult` - Line 222
- `async calculate_autocorrelation(self, symbol: str, max_lags: int | None = None, field: str = 'returns') -> StatisticalResult` - Line 305
- `async detect_regime(self, symbol: str, window: int | None = None, field: str = 'returns') -> StatisticalResult` - Line 391
- `async calculate_cross_correlation(self, symbol1: str, symbol2: str, max_lags: int = 20, field: str = 'returns') -> StatisticalResult` - Line 503
- `async detect_seasonality(self, symbol: str, field: str = 'returns') -> StatisticalResult` - Line 616
- `async calculate_batch_features(self, symbol: str, features: list[str]) -> dict[str, StatisticalResult]` - Line 718
- `async get_calculation_summary(self) -> dict[str, Any]` - Line 760

### Implementation: `IndicatorType` ✅

**Inherits**: Enum
**Purpose**: Technical indicator type enumeration
**Status**: Complete

### Implementation: `IndicatorConfig` ✅

**Purpose**: Technical indicator configuration
**Status**: Complete

### Implementation: `IndicatorResult` ✅

**Purpose**: Technical indicator calculation result
**Status**: Complete

### Implementation: `TechnicalIndicators` ✅

**Inherits**: BaseComponent
**Purpose**: REFACTORED: Technical indicator calculator integrated with DataService architecture
**Status**: Complete

**Implemented Methods:**
- `set_feature_store(self, feature_store)` - Line 155
- `set_data_service(self, data_service)` - Line 160
- `async calculate_indicators_batch(self, ...) -> dict[str, Any]` - Line 166
- `async sma(self, prices: list[Decimal], period: int = 20) -> Decimal | None` - Line 317
- `async ema(self, prices: list[Decimal], period: int = 20) -> Decimal | None` - Line 327
- `async rsi(self, prices: list[Decimal], period: int = 14) -> Decimal | None` - Line 337
- `async macd(self, prices: list[Decimal] | None) -> dict[str, Decimal]` - Line 347
- `async bollinger_bands(self, prices: list[Decimal], period: int = 20, std_dev: Decimal = Any) -> dict[str, Decimal] | None` - Line 357
- `async volume_sma(self, volumes: list[Decimal], period: int = 20) -> Decimal | None` - Line 369
- `get_calculation_stats(self) -> dict[str, Any]` - Line 379
- `async calculate_sma(self, symbol: str, period: int) -> Decimal | None` - Line 384
- `async calculate_ema(self, symbol: str, period: int) -> Decimal | None` - Line 404
- `async calculate_macd(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, Decimal] | None` - Line 425
- `async calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> dict[str, Decimal] | None` - Line 446
- `async calculate_rsi(self, symbol: str, period: int) -> Decimal | None` - Line 466
- `async calculate_momentum(self, symbol: str, period: int) -> Decimal | None` - Line 487
- `async calculate_volatility(self, symbol: str, period: int) -> Decimal | None` - Line 515
- `async calculate_volume_ratio(self, symbol: str, period: int) -> Decimal | None` - Line 542
- `async calculate_atr(self, symbol: str, period: int) -> Decimal | None` - Line 580
- `async calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: Decimal = Any) -> dict[str, Decimal] | None` - Line 623
- `async cleanup(self) -> None` - Line 664

### Implementation: `AlertCategory` ✅

**Inherits**: Enum
**Purpose**: Alert category types
**Status**: Complete

### Implementation: `MonitoringStatus` ✅

**Inherits**: Enum
**Purpose**: Monitoring component status
**Status**: Complete

### Implementation: `MetricValue` ✅

**Purpose**: Metric value with metadata
**Status**: Complete

### Implementation: `Alert` ✅

**Purpose**: Alert with comprehensive details
**Status**: Complete

### Implementation: `MonitoringConfig` ✅

**Inherits**: BaseModel
**Purpose**: Monitoring system configuration
**Status**: Complete

### Implementation: `ThresholdRule` ✅

**Inherits**: BaseModel
**Purpose**: Threshold-based alerting rule
**Status**: Complete

### Implementation: `DataMonitor` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive data monitoring and alerting system
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 180
- `register_component(self, name: str, health_check_func: Callable) -> None` - Line 281
- `unregister_component(self, name: str) -> None` - Line 286
- `async get_system_status(self) -> dict[str, Any]` - Line 664
- `async get_component_health(self, component: str | None = None) -> dict[str, HealthCheckResult]` - Line 720
- `async get_alerts(self, ...) -> list[Alert]` - Line 732
- `async acknowledge_alert(self, alert_id: str, acknowledged_by: str = 'system') -> bool` - Line 753
- `async resolve_alert(self, alert_id: str, resolved_by: str = 'system') -> bool` - Line 765
- `async add_threshold_rule(self, rule: ThresholdRule) -> None` - Line 775
- `async remove_threshold_rule(self, rule_name: str) -> bool` - Line 780
- `async health_check(self) -> HealthCheckResult` - Line 788
- `async cleanup(self) -> None` - Line 809

### Implementation: `MetricType` ✅

**Inherits**: Enum
**Purpose**: Metric type enumeration
**Status**: Complete

### Implementation: `SLAStatus` ✅

**Inherits**: Enum
**Purpose**: SLA compliance status
**Status**: Complete

### Implementation: `Alert` ✅

**Purpose**: Data infrastructure alert
**Status**: Complete

### Implementation: `Metric` ✅

**Purpose**: Performance metric
**Status**: Complete

### Implementation: `SLATarget` ✅

**Purpose**: SLA target definition
**Status**: Complete

### Implementation: `DataQualityMonitor` ✅

**Purpose**: Data quality monitoring component
**Status**: Complete

**Implemented Methods:**
- `async check_data_quality(self, metrics: dict[str, Any]) -> list[Alert]` - Line 116

### Implementation: `PerformanceMonitor` ✅

**Purpose**: Performance monitoring component
**Status**: Complete

**Implemented Methods:**
- `async check_performance(self, metrics: dict[str, Any]) -> list[Alert]` - Line 205

### Implementation: `SLAMonitor` ✅

**Purpose**: SLA compliance monitoring component
**Status**: Complete

**Implemented Methods:**
- `async check_sla_compliance(self, metrics: dict[str, Any]) -> list[Alert]` - Line 301

### Implementation: `DataMonitoringService` ✅

**Inherits**: BaseComponent, DataEventSubscriber
**Purpose**: Enterprise-grade Data Monitoring Service for comprehensive infrastructure monitoring
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 445
- `register_alert_handler(self, handler: Callable[[Alert], None]) -> None` - Line 759
- `get_active_alerts(self, severity: AlertSeverity = None) -> list[Alert]` - Line 763
- `resolve_alert(self, alert_id: str) -> bool` - Line 772
- `get_metrics_summary(self, hours: int = 1) -> dict[str, Any]` - Line 780
- `async health_check(self) -> HealthCheckResult` - Line 814
- `async cleanup(self) -> None` - Line 837

### Implementation: `DataValidationResult` ✅

**Purpose**: Data validation result
**Status**: Complete

### Implementation: `PipelineRecord` ✅

**Purpose**: Pipeline processing record
**Status**: Complete

### Implementation: `DataTransformation` ✅

**Purpose**: Data transformation utilities
**Status**: Complete

**Implemented Methods:**
- `async normalize_prices(data: MarketData) -> MarketData` - Line 96
- `async validate_ohlc_consistency(data: MarketData) -> bool` - Line 192
- `async detect_outliers(data: list[MarketData], symbol: str) -> list[bool]` - Line 205

### Implementation: `DataQualityChecker` ✅

**Purpose**: Comprehensive data quality assessment
**Status**: Complete

**Implemented Methods:**
- `async assess_data_quality(self, data: MarketData) -> DataValidationResult` - Line 250

### Implementation: `DataPipeline` ✅

**Inherits**: BaseComponent
**Purpose**: Data pipeline for financial data processing
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 449
- `async process_data(self, data: MarketData | list[MarketData], priority: int = 5) -> dict[str, Any]` - Line 489
- `get_metrics(self) -> dict[str, Any]` - Line 906
- `async health_check(self) -> dict[str, Any]` - Line 932
- `async cleanup(self) -> None` - Line 972

### Implementation: `IngestionConfig` ✅

**Purpose**: Data ingestion configuration
**Status**: Complete

### Implementation: `IngestionMetrics` ✅

**Purpose**: Pipeline ingestion metrics
**Status**: Complete

### Implementation: `DataIngestionPipeline` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive data ingestion pipeline for multi-source data collection
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 163
- `async start(self) -> None` - Line 248
- `async pause(self) -> None` - Line 888
- `async resume(self) -> None` - Line 911
- `async stop(self) -> None` - Line 934
- `register_callback(self, data_type: str, callback: Callable[[Any], None]) -> None` - Line 1033
- `get_status(self) -> dict[str, Any]` - Line 1064
- `async cleanup(self) -> None` - Line 1133

### Implementation: `ProcessingConfig` ✅

**Purpose**: Data processing configuration
**Status**: Complete

### Implementation: `ProcessingResult` ✅

**Purpose**: Data processing result
**Status**: Complete

### Implementation: `DataProcessor` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive data processing pipeline for multi-source data transformation
**Status**: Complete

**Implemented Methods:**
- `async process_market_data(self, data: MarketData, steps: list[ProcessingStep] | None = None) -> ProcessingResult` - Line 141
- `async process_batch(self, ...) -> list[ProcessingResult]` - Line 225
- `async get_aggregated_data(self, symbol: str, exchange: str | None = None) -> dict[str, Any]` - Line 599
- `get_processing_statistics(self) -> dict[str, Any]` - Line 627
- `async reset_windows(self) -> None` - Line 652
- `async cleanup(self) -> None` - Line 663

### Implementation: `StorageMetrics` ✅

**Purpose**: Storage operation metrics
**Status**: Complete

### Implementation: `DataStorageManager` ✅

**Inherits**: BaseComponent
**Purpose**: Data storage manager for pipeline data persistence
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 98
- `async store_market_data(self, data: MarketData) -> bool` - Line 123
- `async store_batch(self, data_list: list[MarketData]) -> int` - Line 262
- `async cleanup_old_data(self, days_to_keep: int = 30) -> int` - Line 315
- `get_storage_metrics(self) -> dict[str, Any]` - Line 340
- `async force_flush(self) -> bool` - Line 366
- `async cleanup(self) -> None` - Line 380

### Implementation: `PipelineValidationIssue` ✅

**Purpose**: Pipeline-specific validation issue
**Status**: Complete

### Implementation: `PipelineValidator` ✅

**Purpose**: Pipeline-specific data validator for data integrity and quality
**Status**: Complete

**Implemented Methods:**
- `async validate_pipeline_data(self, data: Any, data_type: str, pipeline_stage: str) -> tuple[bool, list[PipelineValidationIssue]]` - Line 74
- `get_validation_statistics(self) -> dict[str, Any]` - Line 214

### Implementation: `ValidationStage` ✅

**Inherits**: Enum
**Purpose**: Validation pipeline stage enumeration
**Status**: Complete

### Implementation: `ValidationPipelineConfig` ✅

**Inherits**: BaseModel
**Purpose**: Validation pipeline configuration
**Status**: Complete

### Implementation: `ValidationDisposition` ✅

**Inherits**: BaseModel
**Purpose**: Validation disposition result
**Status**: Complete

### Implementation: `ValidationPipelineResult` ✅

**Inherits**: BaseModel
**Purpose**: Comprehensive validation pipeline result
**Status**: Complete

### Implementation: `DataValidationPipeline` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive data validation pipeline orchestrator
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 161
- `async validate_batch(self, data: list[MarketData], symbols: list[str] | None = None) -> ValidationPipelineResult` - Line 190
- `async get_quarantined_data(self, symbol: str | None = None) -> dict[str, list[MarketData]]` - Line 518
- `async retry_quarantined_data(self, symbol: str) -> ValidationPipelineResult | None` - Line 524
- `async get_pipeline_status(self) -> dict[str, Any]` - Line 551
- `async health_check(self) -> dict[str, Any]` - Line 561
- `async cleanup(self) -> None` - Line 587

### Implementation: `CleaningStrategy` ✅

**Inherits**: Enum
**Purpose**: Data cleaning strategy enumeration
**Status**: Complete

### Implementation: `OutlierMethod` ✅

**Inherits**: Enum
**Purpose**: Outlier detection method enumeration
**Status**: Complete

### Implementation: `CleaningResult` ✅

**Purpose**: Data cleaning result record
**Status**: Complete

### Implementation: `DataCleaner` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive data cleaning system for market data preprocessing
**Status**: Complete

**Implemented Methods:**
- `async clean_market_data(self, data: MarketData) -> tuple[MarketData, CleaningResult]` - Line 116
- `async clean_signal_data(self, signals: list[Signal]) -> tuple[list[Signal], CleaningResult]` - Line 272
- `async get_cleaning_summary(self) -> dict[str, Any]` - Line 706

### Implementation: `QualityMetric` ✅

**Purpose**: Quality metric record
**Status**: Complete

### Implementation: `DriftAlert` ✅

**Purpose**: Data drift alert record
**Status**: Complete

### Implementation: `QualityMonitor` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive data quality monitoring system
**Status**: Complete

**Implemented Methods:**
- `async monitor_data_quality(self, data: MarketData) -> tuple[float, list[DriftAlert]]` - Line 112
- `async monitor_signal_quality(self, signals: list[Signal]) -> tuple[float, list[DriftAlert]]` - Line 174
- `async generate_quality_report(self, symbol: str | None = None) -> dict[str, Any]` - Line 229
- `async get_monitoring_summary(self) -> dict[str, Any]` - Line 687

### Implementation: `DataValidator` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive data validation system for market data quality assurance
**Status**: Complete

**Implemented Methods:**
- `async validate_market_data(self, data: MarketData) -> tuple[bool, list[ValidationIssue]]` - Line 99
- `async validate_signal(self, signal: Signal) -> tuple[bool, list[ValidationIssue]]` - Line 239
- `async validate_cross_source_consistency(self, primary_data: MarketData, secondary_data: MarketData) -> tuple[bool, list[ValidationIssue]]` - Line 330
- `async get_validation_summary(self) -> dict[str, Any]` - Line 505
- `async cleanup(self) -> None` - Line 553

### Implementation: `ServiceRegistry` ✅

**Inherits**: BaseComponent, Generic[ServiceType]
**Purpose**: Generic service registry for managing service instances and dependencies
**Status**: Complete

**Implemented Methods:**
- `register_service(self, name: str, service: ServiceType, metadata: dict[str, Any] | None = None) -> None` - Line 31
- `get_service(self, name: str) -> ServiceType | None` - Line 55
- `unregister_service(self, name: str) -> bool` - Line 71
- `list_services(self) -> dict[str, dict[str, Any]]` - Line 96
- `subscribe_to_event(self, event_name: str, handler: Callable[[dict[str, Any]], None]) -> None` - Line 105
- `async cleanup(self) -> None` - Line 133

### Implementation: `DataService` ✅

**Inherits**: BaseComponent
**Purpose**: Simplified DataService for core data management
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 97
- `async store_market_data(self, ...) -> bool` - Line 112
- `async get_market_data(self, request: DataRequest) -> list[MarketDataRecord]` - Line 335
- `async get_recent_data(self, ...) -> list[MarketData]` - Line 513
- `async get_data_count(self, symbol: str, exchange: str = DEFAULT_EXCHANGE) -> int` - Line 550
- `async get_volatility(self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None` - Line 565
- `async get_rsi(self, symbol: str, period: int = 14, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None` - Line 616
- `async get_sma(self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None` - Line 634
- `async get_ema(self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None` - Line 652
- `async get_macd(self, ...) -> dict[str, Decimal] | None` - Line 670
- `async get_bollinger_bands(self, ...) -> dict[str, Decimal] | None` - Line 697
- `async get_atr(self, symbol: str, period: int = 14, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None` - Line 722
- `async health_check(self) -> HealthCheckResult` - Line 745
- `async get_metrics(self)` - Line 834
- `async reset_metrics(self) -> None` - Line 844
- `async store_market_data_batch(self, market_data_list: list[MarketData]) -> bool` - Line 850
- `async aggregate_market_data(self, ...) -> list[MarketData]` - Line 897
- `async get_market_data_history(self, symbol: str, limit: int = 100, exchange: str = DEFAULT_EXCHANGE) -> list[MarketData]` - Line 976
- `async cleanup(self) -> None` - Line 1017

### Implementation: `MLDataService` ✅

**Inherits**: BaseService
**Purpose**: ML-specific data service providing storage and retrieval for ML artifacts
**Status**: Complete

**Implemented Methods:**
- `async store_model_metadata(self, metadata: dict[str, Any]) -> None` - Line 57
- `async update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> None` - Line 78
- `async get_model_by_id(self, model_id: str) -> dict[str, Any] | None` - Line 92
- `async get_all_models(self, ...) -> list[dict[str, Any]]` - Line 97
- `async get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]` - Line 118
- `async find_models(self, ...) -> list[dict[str, Any]]` - Line 127
- `async delete_model(self, model_id: str) -> None` - Line 154
- `async store_feature_set(self, ...) -> None` - Line 162
- `async get_feature_set(self, symbol: str, feature_set_id: str, version: str | None = None) -> dict[str, Any] | None` - Line 192
- `async update_feature_set_metadata(self, feature_set_id: str, metadata_updates: dict[str, Any]) -> None` - Line 203
- `async list_feature_sets(self, ...) -> list[dict[str, Any]]` - Line 215
- `async delete_feature_set(self, ...) -> int` - Line 239
- `async get_feature_set_versions(self, symbol: str, feature_set_id: str) -> list[str]` - Line 265
- `async store_artifact_info(self, artifact_metadata: dict[str, Any]) -> None` - Line 283
- `async get_artifact_info(self, ...) -> dict[str, Any] | None` - Line 299
- `async list_artifacts(self, ...) -> list[dict[str, Any]]` - Line 311
- `async delete_artifact_info(self, ...) -> None` - Line 334
- `async save_ml_predictions(self, prediction_data: dict[str, Any]) -> None` - Line 349
- `async store_audit_entry(self, service: str, audit_entry: dict[str, Any]) -> None` - Line 361
- `get_ml_data_metrics(self) -> dict[str, Any]` - Line 394

### Implementation: `DataSourceAdapter` ✅

**Inherits**: BaseComponent
**Purpose**: Adapter to standardize different data source interfaces
**Status**: Complete

**Implemented Methods:**
- `async fetch_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100, **kwargs: Any) -> list[dict[str, Any]]` - Line 61
- `async stream_market_data(self, symbol: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]` - Line 87
- `async connect(self) -> None` - Line 328
- `async disconnect(self) -> None` - Line 336
- `is_connected(self) -> bool` - Line 346

### Implementation: `DataType` ✅

**Inherits**: Enum
**Purpose**: Alternative data type enumeration
**Status**: Complete

### Implementation: `AlternativeDataPoint` ✅

**Purpose**: Alternative data point structure
**Status**: Complete

### Implementation: `EconomicIndicator` ✅

**Purpose**: Economic indicator data structure
**Status**: Complete

### Implementation: `AlternativeDataSource` ✅

**Inherits**: BaseComponent
**Purpose**: Alternative data source for economic and environmental indicators
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 143
- `async get_economic_indicators(self, indicators: list[str], days_back: int = 30) -> list[EconomicIndicator]` - Line 214
- `async get_weather_data(self, locations: list[str], days_back: int = 7) -> list[AlternativeDataPoint]` - Line 329
- `async get_satellite_data(self, regions: list[str], indicators: list[str], days_back: int = 30) -> list[AlternativeDataPoint]` - Line 446
- `async get_comprehensive_dataset(self, config: dict[str, Any]) -> dict[str, list[AlternativeDataPoint]]` - Line 553
- `async get_source_statistics(self) -> dict[str, Any]` - Line 614
- `async cleanup(self) -> None` - Line 634

### Implementation: `RateLimiter` ✅

**Purpose**: Simple rate limiter for API calls
**Status**: Complete

**Implemented Methods:**

### Implementation: `SimpleCache` ✅

**Inherits**: DataCacheInterface
**Purpose**: Simple in-memory cache implementation
**Status**: Complete

**Implemented Methods:**
- `async get(self, key: str) -> Any | None` - Line 52
- `async set(self, key: str, value: Any, ttl: int | None = None) -> None` - Line 68
- `async delete(self, key: str) -> bool` - Line 74
- `async clear(self) -> None` - Line 83
- `async exists(self, key: str) -> bool` - Line 88
- `async health_check(self) -> dict[str, Any]` - Line 92
- `async initialize(self) -> None` - Line 100
- `async cleanup(self) -> None` - Line 105

### Implementation: `BaseDataSource` ✅

**Inherits**: DataSourceInterface
**Purpose**: Base implementation with common functionality for data sources
**Status**: Complete

**Implemented Methods:**
- `async fetch_with_cache(self, cache_key: str, fetch_func: Callable[[], Awaitable[Any]], ttl: int = 60) -> Any` - Line 126
- `async connect(self) -> None` - Line 156
- `async disconnect(self) -> None` - Line 161
- `is_connected(self) -> bool` - Line 166
- `async fetch(self, symbol: str, timeframe: str, limit: int = 100, **kwargs) -> list[dict[str, Any]]` - Line 171
- `async stream(self, symbol: str, **kwargs) -> AsyncIterator[dict[str, Any]]` - Line 177

### Implementation: `DataStreamType` ✅

**Inherits**: Enum
**Purpose**: Data stream type enumeration
**Status**: Complete

### Implementation: `DataSubscription` ✅

**Purpose**: Data subscription configuration
**Status**: Complete

### Implementation: `MarketDataSource` ✅

**Inherits**: BaseComponent
**Purpose**: Market data source for real-time and historical data ingestion
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 132
- `async subscribe_to_ticker(self, exchange_name: str, symbol: str, callback: Callable[[Ticker], None]) -> str` - Line 214
- `async get_historical_data(self, ...) -> list[MarketData]` - Line 259
- `async unsubscribe(self, subscription_id: str) -> bool` - Line 425
- `async get_market_data_summary(self) -> dict[str, Any]` - Line 465
- `async cleanup(self) -> None` - Line 479
- `async get_error_analytics(self) -> dict[str, Any]` - Line 568

### Implementation: `NewsArticle` ✅

**Purpose**: News article data structure
**Status**: Complete

### Implementation: `NewsDataSource` ✅

**Inherits**: BaseComponent
**Purpose**: News data source for sentiment analysis and market impact assessment
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 109
- `async get_news_for_symbol(self, symbol: str, hours_back: int = 24, max_articles: int = 50) -> list[NewsArticle]` - Line 162
- `async get_market_sentiment(self, symbols: list[str]) -> dict[str, dict[str, float]]` - Line 426
- `async cleanup(self) -> None` - Line 477

### Implementation: `SocialPost` ✅

**Purpose**: Social media post data structure
**Status**: Complete

### Implementation: `SocialMetrics` ✅

**Purpose**: Aggregated social metrics for a symbol
**Status**: Complete

### Implementation: `SocialMediaDataSource` ✅

**Inherits**: BaseComponent
**Purpose**: Social media data source for sentiment analysis and trend detection
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 127
- `async get_social_sentiment(self, symbol: str, hours_back: int = 24, platforms: list[str] | None = None) -> SocialMetrics` - Line 188
- `async get_trending_symbols(self, limit: int = 10) -> list[dict[str, Any]]` - Line 439
- `async monitor_symbol_mentions(self, symbols: list[str], callback: Callable[[str, list[SocialPost]], None]) -> None` - Line 482
- `async get_platform_statistics(self) -> dict[str, Any]` - Line 530
- `async cleanup(self) -> None` - Line 545

### Implementation: `DatabaseStorage` ✅

**Inherits**: BaseComponent, DataStorageInterface
**Purpose**: Database storage implementation
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 35
- `async store_records(self, records: list[MarketDataRecord]) -> bool` - Line 41
- `async retrieve_records(self, request: DataRequest) -> list[MarketDataRecord]` - Line 60
- `async get_record_count(self, symbol: str, exchange: str) -> int` - Line 98
- `async health_check(self) -> dict[str, Any]` - Line 119
- `async cleanup(self) -> None` - Line 139

### Implementation: `StreamState` ✅

**Inherits**: Enum
**Purpose**: Streaming connection state
**Status**: Complete

### Implementation: `BufferStrategy` ✅

**Inherits**: Enum
**Purpose**: Buffer overflow strategy
**Status**: Complete

### Implementation: `StreamMetrics` ✅

**Purpose**: Streaming metrics
**Status**: Complete

### Implementation: `StreamConfig` ✅

**Inherits**: BaseModel
**Purpose**: Stream configuration model
**Status**: Complete

### Implementation: `StreamBuffer` ✅

**Purpose**: High-performance streaming data buffer
**Status**: Complete

**Implemented Methods:**
- `async put(self, item: Any) -> bool` - Line 124
- `async get(self, timeout: float | None = None) -> Any | None` - Line 143
- `async get_batch(self, max_size: int, timeout: float = 1.0) -> list[Any]` - Line 156
- `size(self) -> int` - Line 170
- `utilization(self) -> float` - Line 174
- `dropped_count(self) -> int` - Line 178
- `async clear(self) -> None` - Line 182

### Implementation: `WebSocketConnection` ✅

**Purpose**: WebSocket connection manager with automatic reconnection
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 202
- `async listen(self) -> AsyncGenerator[dict[str, Any], None]` - Line 288
- `async disconnect(self) -> None` - Line 324
- `is_connected(self) -> bool` - Line 391
- `get_uptime(self) -> timedelta` - Line 395

### Implementation: `StreamingDataService` ✅

**Inherits**: BaseComponent
**Purpose**: Enterprise-grade real-time market data streaming service
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 453
- `async add_stream(self, exchange: str, config: StreamConfig) -> bool` - Line 497
- `async start_stream(self, exchange: str) -> bool` - Line 513
- `async stop_stream(self, exchange: str) -> bool` - Line 551
- `async get_stream_status(self, exchange: str | None = None) -> dict[str, Any]` - Line 1020
- `get_metrics(self) -> dict[str, Any]` - Line 1044
- `async health_check(self) -> dict[str, Any]` - Line 1048
- `async cleanup(self) -> None` - Line 1069

### Implementation: `CacheLevel` ✅

**Inherits**: Enum
**Purpose**: Cache level enumeration
**Status**: Complete

### Implementation: `DataPipelineStage` ✅

**Inherits**: Enum
**Purpose**: Data pipeline stage enumeration
**Status**: Complete

### Implementation: `DataMetrics` ✅

**Purpose**: Data processing metrics
**Status**: Complete

### Implementation: `DataRequest` ✅

**Inherits**: BaseModel
**Purpose**: Data request model with validation
**Status**: Complete

**Implemented Methods:**
- `validate_time_range(cls, v: datetime | None, info: Any) -> datetime | None` - Line 81

### Implementation: `FeatureRequest` ✅

**Inherits**: BaseModel
**Purpose**: Feature calculation request model
**Status**: Complete

### Implementation: `DataValidationPipeline` ✅

**Inherits**: BaseComponent
**Purpose**: Centralized validation pipeline for data
**Status**: Complete

**Implemented Methods:**
- `add_validator(self, validator: DataValidatorInterface) -> 'DataValidationPipeline'` - Line 17
- `remove_validator(self, validator_type: type) -> bool` - Line 31
- `validate(self, data: Any) -> tuple[bool, list[str]]` - Line 48
- `validate_batch(self, data_list: list[Any]) -> list[tuple[bool, list[str]]]` - Line 106
- `clear(self) -> None` - Line 121
- `get_validator_count(self) -> int` - Line 126

### Implementation: `MarketDataValidationResult` ✅

**Inherits**: BaseModel
**Purpose**: Market data validation result
**Status**: Complete

### Implementation: `DataValidator` ✅

**Inherits**: BaseComponent
**Purpose**: Comprehensive data validator for financial data quality assurance
**Status**: Complete

**Implemented Methods:**
- `async validate_market_data(self, data: MarketData | list[MarketData], include_statistical: bool = True) -> MarketDataValidationResult | list[MarketDataValidationResult]` - Line 99
- `async add_custom_rule(self, rule_name: str, rule_config: dict[str, Any]) -> bool` - Line 183
- `async disable_rule(self, rule_name: str) -> bool` - Line 193
- `async enable_rule(self, rule_name: str) -> bool` - Line 198
- `async get_validation_stats(self) -> dict[str, Any]` - Line 203
- `async health_check(self) -> HealthCheckResult` - Line 218
- `async cleanup(self) -> None` - Line 233

### Implementation: `MarketDataValidator` ✅

**Inherits**: BaseComponent, DataValidatorInterface, ServiceDataValidatorInterface
**Purpose**: Market data validator implementation that uses consolidated validation utilities
**Status**: Complete

**Implemented Methods:**
- `async validate_market_data(self, data_list: list[MarketData]) -> list[MarketData]` - Line 32
- `get_validation_errors(self) -> list[str]` - Line 51
- `validate(self, data) -> bool` - Line 55
- `get_errors(self) -> list[str]` - Line 74
- `reset(self) -> None` - Line 78
- `async health_check(self) -> dict[str, Any]` - Line 82

### Implementation: `PriceValidator` ✅

**Inherits**: BaseRecordValidator
**Purpose**: Validator for price data using consolidated utilities
**Status**: Complete

**Implemented Methods:**

### Implementation: `VolumeValidator` ✅

**Inherits**: BaseRecordValidator
**Purpose**: Validator for volume data using consolidated utilities
**Status**: Complete

**Implemented Methods:**

### Implementation: `TimestampValidator` ✅

**Inherits**: BaseRecordValidator
**Purpose**: Validator for timestamp data using consolidated utilities
**Status**: Complete

**Implemented Methods:**
- `validate(self, data: Any) -> bool` - Line 101
- `reset(self) -> None` - Line 156

### Implementation: `SchemaValidator` ✅

**Inherits**: BaseRecordValidator
**Purpose**: Validator for data schema
**Status**: Complete

**Implemented Methods:**

### Implementation: `HighPerformanceDataBuffer` ✅

**Purpose**: High-performance circular buffer optimized for streaming market data
**Status**: Complete

**Implemented Methods:**
- `append_batch(self, data: np.ndarray) -> None` - Line 167
- `get_recent_vectorized(self, n: int) -> np.ndarray` - Line 188

### Implementation: `IndicatorCache` ✅

**Purpose**: Cache for calculated indicators to avoid recalculation
**Status**: Complete

**Implemented Methods:**
- `is_valid(self, key: str) -> bool` - Line 216
- `get(self, key: str) -> np.ndarray | None` - Line 220
- `set(self, key: str, value: np.ndarray) -> None` - Line 227

### Implementation: `VectorizedProcessor` ✅

**Purpose**: High-performance market data processor using vectorized operations
**Status**: Complete

**Implemented Methods:**
- `async process_market_data_batch(self, market_data: list[dict[str, Any]]) -> dict[str, np.ndarray]` - Line 309
- `calculate_real_time_indicators(self, current_price: Decimal) -> dict[str, Decimal]` - Line 449
- `get_performance_metrics(self) -> dict[str, Any]` - Line 520
- `cleanup(self) -> None` - Line 537

## COMPLETE API REFERENCE

### File: data_cache.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import Config`
- `from src.core import HealthCheckResult`
- `from src.core import HealthStatus`
- `from src.utils.cache_utilities import CacheEntry`

#### Class: `CacheConfig`

**Inherits**: BaseModel
**Purpose**: Cache configuration model

```python
class CacheConfig(BaseModel):
```

#### Class: `L1MemoryCache`

**Inherits**: BaseComponent
**Purpose**: L1 Memory Cache - Fastest access, limited capacity

```python
class L1MemoryCache(BaseComponent):
    def __init__(self, config: CacheConfig)  # Line 73
    async def get(self, key: str) -> Any | None  # Line 81
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool  # Line 106
    async def delete(self, key: str) -> bool  # Line 145
    async def clear(self) -> None  # Line 154
    async def _ensure_capacity(self, required_bytes: int) -> bool  # Line 160
    async def _evict_one_entry(self) -> bool  # Line 175
    async def _evict_entry(self, key: str) -> None  # Line 196
    def _calculate_size(self, value: Any) -> int  # Line 205
    async def get_stats(self) -> CacheStats  # Line 220
```

#### Class: `L2RedisCache`

**Inherits**: BaseComponent
**Purpose**: L2 Redis Cache - Network-based, persistent, shared across instances

```python
class L2RedisCache(BaseComponent):
    def __init__(self, config: CacheConfig, redis_config: dict[str, Any])  # Line 239
    async def initialize(self) -> None  # Line 247
    async def get(self, key: str) -> Any | None  # Line 278
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool  # Line 303
    async def delete(self, key: str) -> bool  # Line 324
    async def clear(self) -> None  # Line 341
    async def batch_get(self, keys: list[str]) -> dict[str, Any]  # Line 356
    async def batch_set(self, data: dict[str, Any], ttl: int | None = None) -> bool  # Line 381
    def _serialize(self, value: Any) -> bytes  # Line 404
    def _deserialize(self, data: bytes) -> Any  # Line 426
    async def get_stats(self) -> CacheStats  # Line 446
    async def cleanup(self) -> None  # Line 450
```

#### Class: `DataCache`

**Inherits**: BaseComponent
**Purpose**: Multi-level data cache orchestrator

```python
class DataCache(BaseComponent):
    def __init__(self, config: Config)  # Line 494
    def _setup_configurations(self) -> None  # Line 516
    async def initialize(self) -> None  # Line 545
    async def get(self, key: str, warm_lower_levels: bool = True) -> Any | None  # Line 569
    async def set(self, ...) -> bool  # Line 609
    async def delete(self, key: str, levels: list[CacheLevel] | None = None) -> bool  # Line 649
    async def batch_get(self, keys: list[str]) -> dict[str, Any]  # Line 682
    async def batch_set(self, data: dict[str, Any], ttl: int | None = None) -> bool  # Line 716
    async def clear(self, levels: list[CacheLevel] | None = None) -> None  # Line 741
    async def warm_cache(self, data_loader: Callable[[list[str]], dict[str, Any]], keys: list[str]) -> None  # Line 758
    async def _background_cache_warmer(self) -> None  # Line 774
    async def _process_prefetch_request(self, request: dict[str, Any]) -> None  # Line 795
    async def _perform_cache_maintenance(self) -> None  # Line 800
    async def get_stats(self) -> dict[str, CacheStats]  # Line 817
    async def health_check(self) -> HealthCheckResult  # Line 831
    async def cleanup(self) -> None  # Line 874
```

### File: redis_cache.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import Config`
- `from src.data.interfaces import DataCacheInterface`
- `from src.utils.cache_utilities import CacheSerializationUtils`

#### Class: `RedisCache`

**Inherits**: BaseComponent, DataCacheInterface
**Purpose**: Redis cache implementation

```python
class RedisCache(BaseComponent, DataCacheInterface):
    def __init__(self, config: Config)  # Line 20
    async def initialize(self) -> None  # Line 39
    async def get(self, key: str) -> Any | None  # Line 65
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None  # Line 81
    async def delete(self, key: str) -> bool  # Line 98
    async def clear(self) -> None  # Line 111
    async def exists(self, key: str) -> bool  # Line 123
    async def health_check(self) -> dict[str, Any]  # Line 136
    async def cleanup(self) -> None  # Line 161
```

### File: controller.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import Config`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`
- `from src.data.constants import DEFAULT_DATA_LIMIT`

#### Class: `DataController`

**Inherits**: BaseComponent
**Purpose**: Data controller for handling data operations requests

```python
class DataController(BaseComponent):
    def __init__(self, config: Config, data_service: DataServiceInterface)  # Line 34
    async def initialize(self) -> None  # Line 50
    async def store_market_data_request(self, data: MarketData | list[MarketData], exchange: str, validate: bool = True) -> dict[str, Any]  # Line 63
    async def get_market_data_request(self, ...) -> dict[str, Any]  # Line 110
    async def get_data_count_request(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]  # Line 175
    async def get_recent_data_request(self, ...) -> dict[str, Any]  # Line 212
    async def get_health_status_request(self) -> dict[str, Any]  # Line 273
    async def cleanup(self) -> None  # Line 292
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def _resolve_optional_dependency(injector: DependencyInjector, service_name: str, default_factory = None)  # Line 22
def _get_config(injector: DependencyInjector)  # Line 35
def register_data_services(injector: DependencyInjector) -> None  # Line 47
def configure_data_dependencies(injector: DependencyInjector | None = None) -> DependencyInjector  # Line 205
def get_data_service(injector: DependencyInjector) -> 'DataServiceInterface'  # Line 226
def get_data_storage(injector: DependencyInjector) -> 'DataStorageInterface'  # Line 231
def get_data_cache(injector: DependencyInjector) -> 'DataCacheInterface'  # Line 236
def get_data_validator(injector: DependencyInjector) -> 'DataValidatorInterface'  # Line 241
def get_market_data_source(injector: DependencyInjector) -> 'MarketDataSource'  # Line 246
def get_vectorized_processor(injector: DependencyInjector) -> 'VectorizedProcessor'  # Line 251
def get_service_data_validator(injector: DependencyInjector)  # Line 256
def get_data_service_factory(injector: DependencyInjector)  # Line 261
def get_data_pipeline_ingestion(injector: DependencyInjector)  # Line 266
def get_streaming_data_service(injector: DependencyInjector)  # Line 271
def get_data_service_registry(injector: DependencyInjector)  # Line 276
def get_ml_data_service(injector: DependencyInjector)  # Line 281
```

### File: environment_integration.py

**Key Imports:**
- `from src.core.integration.environment_aware_service import EnvironmentAwareServiceMixin`
- `from src.core.integration.environment_aware_service import EnvironmentContext`
- `from src.core.logging import get_logger`
- `from src.core.types import MarketData`

#### Class: `DataQualityLevel`

**Inherits**: Enum
**Purpose**: Data quality levels for different environments

```python
class DataQualityLevel(Enum):
```

#### Class: `DataStorageStrategy`

**Inherits**: Enum
**Purpose**: Data storage strategies for different environments

```python
class DataStorageStrategy(Enum):
```

#### Class: `EnvironmentAwareDataConfiguration`

**Purpose**: Environment-specific data configuration

```python
class EnvironmentAwareDataConfiguration:
    def get_sandbox_data_config() -> dict[str, Any]  # Line 44
    def get_live_data_config() -> dict[str, Any]  # Line 70
```

#### Class: `EnvironmentAwareDataManager`

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware data management functionality

```python
class EnvironmentAwareDataManager(EnvironmentAwareServiceMixin):
    def __init__(self, *args, **kwargs)  # Line 104
    async def _update_service_environment(self, context: EnvironmentContext) -> None  # Line 111
    def get_environment_data_config(self, exchange: str) -> dict[str, Any]  # Line 144
    async def validate_market_data_for_environment(self, market_data: MarketData, exchange: str) -> bool  # Line 157
    async def _validate_basic_data_quality(self, market_data: MarketData, exchange: str) -> bool  # Line 188
    async def _validate_sandbox_data(self, market_data: MarketData, exchange: str) -> bool  # Line 206
    async def _validate_production_data(self, market_data: MarketData, exchange: str, data_config: dict[str, Any]) -> bool  # Line 220
    async def _validate_ultra_strict_data(self, market_data: MarketData, exchange: str, data_config: dict[str, Any]) -> bool  # Line 257
    async def _validate_standard_data(self, market_data: MarketData, exchange: str) -> bool  # Line 282
    async def store_market_data_by_environment(self, market_data: MarketData, exchange: str) -> bool  # Line 299
    async def _store_in_memory(self, market_data: MarketData, exchange: str) -> bool  # Line 339
    async def _store_in_cache(self, market_data: MarketData, exchange: str, data_config: dict[str, Any]) -> bool  # Line 345
    async def _store_in_database(self, market_data: MarketData, exchange: str, data_config: dict[str, Any]) -> bool  # Line 359
    async def get_environment_aware_market_data(self, ...) -> list[MarketData]  # Line 377
    async def _get_production_market_data(self, symbol: str, exchange: str, timeframe: str | None, limit: int | None) -> list[MarketData]  # Line 404
    async def _get_sandbox_market_data(self, symbol: str, exchange: str, timeframe: str | None, limit: int | None) -> list[MarketData]  # Line 416
    async def _apply_environment_data_filters(self, data: list[MarketData], exchange: str, data_config: dict[str, Any]) -> list[MarketData]  # Line 439
    async def _setup_environment_data_components(self, exchange: str, data_config: dict[str, Any]) -> None  # Line 469
    async def _create_anomaly_detector(self, exchange: str) -> Any  # Line 489
    async def _is_price_anomalous(self, market_data: MarketData, exchange: str) -> bool  # Line 494
    async def _is_data_anomalous(self, market_data: MarketData, exchange: str) -> bool  # Line 499
    async def _is_price_unreasonable(self, market_data: MarketData, exchange: str) -> bool  # Line 508
    async def _validate_volume_data(self, market_data: MarketData, exchange: str) -> bool  # Line 514
    async def _cross_validate_market_data(self, market_data: MarketData, exchange: str) -> bool  # Line 522
    async def _validate_data_integrity(self, market_data: MarketData, exchange: str) -> bool  # Line 527
    async def _handle_anomalous_data(self, market_data: MarketData, exchange: str) -> bool  # Line 532
    async def _log_data_issue(self, message: str, exchange: str, severity: str) -> None  # Line 546
    async def _log_data_audit_event(self, market_data: MarketData, exchange: str, action: str) -> None  # Line 558
    async def _update_data_quality_metrics(self, exchange: str, data_points_count: int) -> None  # Line 572
    def get_environment_data_metrics(self, exchange: str) -> dict[str, Any]  # Line 589
```

### File: events.py

**Key Imports:**
- `from src.core.event_constants import DataEvents`

#### Class: `DataEventType`

**Inherits**: Enum
**Purpose**: Types of data events - aligned with core event constants

```python
class DataEventType(Enum):
```

#### Class: `DataEvent`

**Purpose**: Data event structure aligned with database service messaging patterns

```python
class DataEvent:
```

#### Class: `DataEventPublisher`

**Purpose**: Simple mixin class to add basic event publishing capabilities

```python
class DataEventPublisher:
    def __init__(self, *args, **kwargs)  # Line 51
    async def _publish_data_event(self, ...) -> None  # Line 55
```

#### Class: `DataEventSubscriber`

**Purpose**: Base class for data event subscribers

```python
class DataEventSubscriber:
    def __init__(self, *args, **kwargs)  # Line 135
    async def handle_data_event(self, event: DataEvent) -> None  # Line 139
```

### File: factory.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.data.interfaces import DataServiceInterface`

#### Class: `DataServiceFactory`

**Purpose**: Factory for creating data services with proper dependency injection

```python
class DataServiceFactory:
    def __init__(self, injector: DependencyInjector | None = None)  # Line 19
    def create_data_service(self, use_cache: bool = True, use_validator: bool = True) -> DataServiceInterface  # Line 38
    def create_minimal_data_service(self) -> DataServiceInterface  # Line 55
    def create_testing_data_service(self, mock_storage = None, mock_cache = None, mock_validator = None) -> DataServiceInterface  # Line 64
    def create_data_storage(self) -> 'DataStorageInterface'  # Line 100
    def create_data_cache(self) -> 'DataCacheInterface'  # Line 104
    def create_data_validator(self) -> 'DataValidatorInterface'  # Line 108
    def create_market_data_source(self) -> 'MarketDataSource'  # Line 112
    def create_vectorized_processor(self) -> 'VectorizedProcessor'  # Line 116
```

#### Functions:

```python
def create_default_data_service(config: 'Config | None' = None, injector: DependencyInjector | None = None) -> DataServiceInterface  # Line 121
```

### File: alternative_features.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import DataError`
- `from src.core.logging import get_logger`
- `from src.core.types import NewsSentiment`
- `from src.core.types import SocialSentiment`

#### Class: `AlternativeFeatureType`

**Inherits**: Enum
**Purpose**: Alternative feature type enumeration

```python
class AlternativeFeatureType(Enum):
```

#### Class: `SentimentStrength`

**Inherits**: Enum
**Purpose**: Sentiment strength enumeration

```python
class SentimentStrength(Enum):
```

#### Class: `AlternativeConfig`

**Purpose**: Alternative feature calculation configuration

```python
class AlternativeConfig:
```

#### Class: `AlternativeResult`

**Purpose**: Alternative feature calculation result

```python
class AlternativeResult:
```

#### Class: `AlternativeFeatureCalculator`

**Purpose**: Comprehensive alternative data feature calculator

```python
class AlternativeFeatureCalculator:
    def __init__(self, config: Config)  # Line 93
    def set_data_sources(self, news_source = None, social_source = None, alt_data_source = None) -> None  # Line 164
    async def initialize(self) -> None  # Line 178
    async def calculate_news_sentiment(self, symbol: str, lookback_hours: int | None = None) -> AlternativeResult  # Line 194
    async def calculate_social_sentiment(self, symbol: str, lookback_hours: int | None = None) -> AlternativeResult  # Line 322
    async def calculate_economic_indicators(self, symbol: str, lookback_hours: int | None = None) -> AlternativeResult  # Line 482
    def _calculate_sentiment_trend(self, sentiment_scores: list[float]) -> float  # Line 601
    def _determine_sentiment_strength(self, avg_sentiment: float, sentiment_std: float) -> SentimentStrength  # Line 613
    async def calculate_market_microstructure(self, symbol: str, lookback_hours: int | None = None) -> AlternativeResult  # Line 632
    async def calculate_batch_features(self, symbol: str, features: list[str]) -> dict[str, AlternativeResult]  # Line 677
    async def get_calculation_summary(self) -> dict[str, Any]  # Line 723
```

### File: feature_store.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import HealthCheckResult`
- `from src.core import HealthStatus`
- `from src.core.config import Config`
- `from src.core.exceptions import ValidationError`

#### Class: `FeatureType`

**Inherits**: Enum
**Purpose**: Feature type enumeration

```python
class FeatureType(Enum):
```

#### Class: `CalculationStatus`

**Inherits**: Enum
**Purpose**: Feature calculation status

```python
class CalculationStatus(Enum):
```

#### Class: `FeatureMetadata`

**Purpose**: Feature metadata for tracking and versioning

```python
class FeatureMetadata:
```

#### Class: `FeatureValue`

**Purpose**: Feature calculation result

```python
class FeatureValue:
```

#### Class: `FeatureRequest`

**Inherits**: BaseModel
**Purpose**: Feature calculation request model

```python
class FeatureRequest(BaseModel):
    def validate_feature_types(cls, v)  # Line 103
```

#### Class: `FeatureCalculationPipeline`

**Purpose**: Feature calculation pipeline for efficient batch processing

```python
class FeatureCalculationPipeline:
    def __init__(self, feature_store: 'FeatureStore')  # Line 112
    async def calculate_batch(self, requests: list[FeatureRequest]) -> dict[str, list[FeatureValue]]  # Line 117
    async def _calculate_symbol_batch(self, symbol: str, requests: list[FeatureRequest]) -> list[FeatureValue]  # Line 141
```

#### Class: `FeatureStore`

**Inherits**: BaseComponent
**Purpose**: Enterprise-grade FeatureStore for financial feature management

```python
class FeatureStore(BaseComponent):
    def __init__(self, config: Config, data_service = None)  # Line 187
    def _setup_configuration(self) -> None  # Line 223
    async def initialize(self) -> None  # Line 239
    async def _register_builtin_features(self) -> None  # Line 270
    async def _initialize_technical_indicators(self) -> None  # Line 332
    async def _initialize_statistical_features(self) -> None  # Line 343
    async def _initialize_alternative_features(self) -> None  # Line 354
    async def register_feature(self, ...) -> None  # Line 365
    async def calculate_features(self, request: FeatureRequest) -> list[FeatureValue]  # Line 392
    async def _get_cached_features(self, request: FeatureRequest) -> list[FeatureValue]  # Line 423
    def _is_cache_valid(self, cached_value: FeatureValue, feature_name: str) -> bool  # Line 440
    async def _calculate_features_batch(self, request: FeatureRequest) -> list[FeatureValue]  # Line 450
    async def _calculate_single_feature(self, ...) -> FeatureValue | None  # Line 489
    async def _get_market_data(self, symbol: str, lookback_period: int) -> list[MarketData]  # Line 568
    def _build_cache_key(self, symbol: str, feature_name: str, parameters: dict[str, Any]) -> str  # Line 616
    async def _cache_results(self, results: list[FeatureValue]) -> None  # Line 627
    async def _cleanup_old_cache_entries(self) -> None  # Line 643
    async def _cache_cleanup_loop(self) -> None  # Line 663
    async def _calculate_sma(self, market_data: list[MarketData], period: int = 20, **kwargs) -> Decimal | None  # Line 675
    async def _calculate_ema(self, market_data: list[MarketData], period: int = 20, **kwargs) -> Decimal | None  # Line 690
    async def _calculate_rsi(self, market_data: list[MarketData], period: int = 14, **kwargs) -> Decimal | None  # Line 712
    async def _calculate_macd(self, ...) -> dict[str, Decimal] | None  # Line 750
    async def _calculate_bollinger_bands(self, ...) -> dict[str, Decimal] | None  # Line 828
    async def _calculate_volatility(self, market_data: list[MarketData], window: int = 20, **kwargs) -> Decimal | None  # Line 869
    async def _calculate_momentum(self, market_data: list[MarketData], window: int = 10, **kwargs) -> Decimal | None  # Line 902
    async def _calculate_volume_trend(self, market_data: list[MarketData], window: int = 20, **kwargs) -> Decimal | None  # Line 924
    def get_metrics(self) -> dict[str, Any]  # Line 948
    async def health_check(self) -> HealthCheckResult  # Line 970
    async def cleanup(self) -> None  # Line 995
```

### File: statistical_features.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import DataError`
- `from src.core.types import MarketData`
- `from src.error_handling import ErrorHandler`

#### Class: `StatFeatureType`

**Inherits**: Enum
**Purpose**: Statistical feature type enumeration

```python
class StatFeatureType(Enum):
```

#### Class: `RegimeType`

**Inherits**: Enum
**Purpose**: Market regime type enumeration

```python
class RegimeType(Enum):
```

#### Class: `StatisticalConfig`

**Purpose**: Statistical feature calculation configuration

```python
class StatisticalConfig:
```

#### Class: `StatisticalResult`

**Purpose**: Statistical feature calculation result

```python
class StatisticalResult:
```

#### Class: `StatisticalFeatures`

**Inherits**: BaseComponent
**Purpose**: REFACTORED: Statistical feature calculator integrated with FeatureStore architecture

```python
class StatisticalFeatures(BaseComponent):
    def __init__(self, config: Config, feature_store = None)  # Line 107
    async def add_market_data(self, data: MarketData) -> None  # Line 152
    async def calculate_rolling_stats(self, symbol: str, window: int | None = None, field: str = 'returns') -> StatisticalResult  # Line 222
    async def calculate_autocorrelation(self, symbol: str, max_lags: int | None = None, field: str = 'returns') -> StatisticalResult  # Line 305
    async def detect_regime(self, symbol: str, window: int | None = None, field: str = 'returns') -> StatisticalResult  # Line 391
    async def calculate_cross_correlation(self, symbol1: str, symbol2: str, max_lags: int = 20, field: str = 'returns') -> StatisticalResult  # Line 503
    async def detect_seasonality(self, symbol: str, field: str = 'returns') -> StatisticalResult  # Line 616
    async def calculate_batch_features(self, symbol: str, features: list[str]) -> dict[str, StatisticalResult]  # Line 718
    async def get_calculation_summary(self) -> dict[str, Any]  # Line 760
```

### File: technical_indicators.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.types import MarketData`
- `from src.error_handling import ErrorHandler`
- `from src.utils.decorators import time_execution`

#### Class: `IndicatorType`

**Inherits**: Enum
**Purpose**: Technical indicator type enumeration

```python
class IndicatorType(Enum):
```

#### Class: `IndicatorConfig`

**Purpose**: Technical indicator configuration

```python
class IndicatorConfig:
```

#### Class: `IndicatorResult`

**Purpose**: Technical indicator calculation result

```python
class IndicatorResult:
```

#### Class: `TechnicalIndicators`

**Inherits**: BaseComponent
**Purpose**: REFACTORED: Technical indicator calculator integrated with DataService architecture

```python
class TechnicalIndicators(BaseComponent):
    def __init__(self, config: Config, feature_store = None, data_service = None)  # Line 103
    def set_feature_store(self, feature_store)  # Line 155
    def set_data_service(self, data_service)  # Line 160
    async def calculate_indicators_batch(self, ...) -> dict[str, Any]  # Line 166
    async def _calculate_sma(self, prices: np.ndarray, period: int) -> Decimal | None  # Line 286
    async def _calculate_ema(self, prices: np.ndarray, period: int) -> Decimal | None  # Line 290
    async def _calculate_rsi(self, prices: np.ndarray, period: int) -> Decimal | None  # Line 294
    async def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, Decimal] | None  # Line 298
    async def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: Decimal = Any) -> dict[str, Decimal] | None  # Line 304
    async def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Decimal | None  # Line 310
    async def sma(self, prices: list[Decimal], period: int = 20) -> Decimal | None  # Line 317
    async def ema(self, prices: list[Decimal], period: int = 20) -> Decimal | None  # Line 327
    async def rsi(self, prices: list[Decimal], period: int = 14) -> Decimal | None  # Line 337
    async def macd(self, prices: list[Decimal] | None) -> dict[str, Decimal]  # Line 347
    async def bollinger_bands(self, prices: list[Decimal], period: int = 20, std_dev: Decimal = Any) -> dict[str, Decimal] | None  # Line 357
    async def volume_sma(self, volumes: list[Decimal], period: int = 20) -> Decimal | None  # Line 369
    def get_calculation_stats(self) -> dict[str, Any]  # Line 379
    async def calculate_sma(self, symbol: str, period: int) -> Decimal | None  # Line 384
    async def calculate_ema(self, symbol: str, period: int) -> Decimal | None  # Line 404
    async def calculate_macd(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, Decimal] | None  # Line 425
    async def calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> dict[str, Decimal] | None  # Line 446
    async def calculate_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 466
    async def calculate_momentum(self, symbol: str, period: int) -> Decimal | None  # Line 487
    async def calculate_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 515
    async def calculate_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 542
    async def calculate_atr(self, symbol: str, period: int) -> Decimal | None  # Line 580
    async def calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: Decimal = Any) -> dict[str, Decimal] | None  # Line 623
    async def _get_price_data(self, symbol: str, limit: int) -> list[Any] | None  # Line 643
    async def cleanup(self) -> None  # Line 664
    def __str__(self) -> str  # Line 676
    def __repr__(self) -> str  # Line 687
```

### File: data_monitor.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import HealthCheckResult`
- `from src.core import HealthStatus`
- `from src.core.config import Config`
- `from src.core.types import AlertSeverity`

#### Class: `AlertCategory`

**Inherits**: Enum
**Purpose**: Alert category types

```python
class AlertCategory(Enum):
```

#### Class: `MonitoringStatus`

**Inherits**: Enum
**Purpose**: Monitoring component status

```python
class MonitoringStatus(Enum):
```

#### Class: `MetricValue`

**Purpose**: Metric value with metadata

```python
class MetricValue:
```

#### Class: `Alert`

**Purpose**: Alert with comprehensive details

```python
class Alert:
```

#### Class: `MonitoringConfig`

**Inherits**: BaseModel
**Purpose**: Monitoring system configuration

```python
class MonitoringConfig(BaseModel):
```

#### Class: `ThresholdRule`

**Inherits**: BaseModel
**Purpose**: Threshold-based alerting rule

```python
class ThresholdRule(BaseModel):
```

#### Class: `DataMonitor`

**Inherits**: BaseComponent
**Purpose**: Comprehensive data monitoring and alerting system

```python
class DataMonitor(BaseComponent):
    def __init__(self, config: Config)  # Line 144
    def _setup_configuration(self) -> None  # Line 168
    async def initialize(self) -> None  # Line 180
    async def _setup_default_threshold_rules(self) -> None  # Line 201
    async def _start_monitoring_tasks(self) -> None  # Line 261
    def register_component(self, name: str, health_check_func: Callable) -> None  # Line 281
    def unregister_component(self, name: str) -> None  # Line 286
    async def _health_check_loop(self) -> None  # Line 294
    async def _perform_health_checks(self) -> None  # Line 305
    def _determine_health_status(self, health_data: Any) -> MonitoringStatus  # Line 354
    async def _metric_collection_loop(self) -> None  # Line 370
    async def _collect_system_metrics(self) -> None  # Line 381
    async def _record_metric(self, name: str, value: float | int, tags: dict[str, str] | None = None) -> None  # Line 409
    async def _alert_evaluation_loop(self) -> None  # Line 428
    async def _evaluate_alerts(self) -> None  # Line 439
    async def _is_alert_on_cooldown(self, rule_name: str) -> bool  # Line 462
    async def _get_latest_metric(self, metric_name: str) -> MetricValue | None  # Line 470
    async def _evaluate_threshold(self, value: float | int, rule: ThresholdRule) -> bool  # Line 478
    async def _create_alert(self, rule: ThresholdRule, metric: MetricValue) -> None  # Line 502
    async def _generate_alert_suggestions(self, alert: Alert) -> list[str]  # Line 531
    async def _process_alert(self, alert: Alert) -> None  # Line 570
    async def _send_email_alert(self, alert: Alert) -> None  # Line 593
    async def _send_slack_alert(self, alert: Alert) -> None  # Line 598
    async def _send_webhook_alert(self, alert: Alert) -> None  # Line 603
    async def _cleanup_loop(self) -> None  # Line 608
    async def _cleanup_old_data(self) -> None  # Line 619
    async def get_system_status(self) -> dict[str, Any]  # Line 664
    async def get_component_health(self, component: str | None = None) -> dict[str, HealthCheckResult]  # Line 720
    async def get_alerts(self, ...) -> list[Alert]  # Line 732
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = 'system') -> bool  # Line 753
    async def resolve_alert(self, alert_id: str, resolved_by: str = 'system') -> bool  # Line 765
    async def add_threshold_rule(self, rule: ThresholdRule) -> None  # Line 775
    async def remove_threshold_rule(self, rule_name: str) -> bool  # Line 780
    async def health_check(self) -> HealthCheckResult  # Line 788
    async def cleanup(self) -> None  # Line 809
```

### File: data_monitoring_service.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import HealthCheckResult`
- `from src.core import HealthStatus`
- `from src.core.config import Config`
- `from src.core.types import AlertSeverity`

#### Class: `MetricType`

**Inherits**: Enum
**Purpose**: Metric type enumeration

```python
class MetricType(Enum):
```

#### Class: `SLAStatus`

**Inherits**: Enum
**Purpose**: SLA compliance status

```python
class SLAStatus(Enum):
```

#### Class: `Alert`

**Purpose**: Data infrastructure alert

```python
class Alert:
```

#### Class: `Metric`

**Purpose**: Performance metric

```python
class Metric:
```

#### Class: `SLATarget`

**Purpose**: SLA target definition

```python
class SLATarget:
```

#### Class: `DataQualityMonitor`

**Purpose**: Data quality monitoring component

```python
class DataQualityMonitor:
    def __init__(self, config: Config)  # Line 98
    async def check_data_quality(self, metrics: dict[str, Any]) -> list[Alert]  # Line 116
```

#### Class: `PerformanceMonitor`

**Purpose**: Performance monitoring component

```python
class PerformanceMonitor:
    def __init__(self, config: Config)  # Line 192
    async def check_performance(self, metrics: dict[str, Any]) -> list[Alert]  # Line 205
```

#### Class: `SLAMonitor`

**Purpose**: SLA compliance monitoring component

```python
class SLAMonitor:
    def __init__(self, config: Config)  # Line 272
    async def check_sla_compliance(self, metrics: dict[str, Any]) -> list[Alert]  # Line 301
```

#### Class: `DataMonitoringService`

**Inherits**: BaseComponent, DataEventSubscriber
**Purpose**: Enterprise-grade Data Monitoring Service for comprehensive infrastructure monitoring

```python
class DataMonitoringService(BaseComponent, DataEventSubscriber):
    def __init__(self, config: Config)  # Line 395
    def _setup_configuration(self) -> None  # Line 431
    async def initialize(self) -> None  # Line 445
    async def _setup_event_subscriptions(self) -> None  # Line 472
    async def _subscribe_to_data_event(self, event_type: DataEventType, handler: Callable[[DataEvent], None]) -> None  # Line 492
    async def _handle_data_stored(self, event: DataEvent) -> None  # Line 500
    async def _handle_data_retrieved(self, event: DataEvent) -> None  # Line 523
    async def _handle_validation_failed(self, event: DataEvent) -> None  # Line 532
    async def _handle_cache_hit(self, event: DataEvent) -> None  # Line 552
    async def _handle_cache_miss(self, event: DataEvent) -> None  # Line 561
    async def _monitoring_loop(self) -> None  # Line 585
    def _build_event_based_metrics(self) -> dict[str, Any]  # Line 614
    async def _run_monitoring_checks(self, metrics: dict[str, Any]) -> None  # Line 639
    def _flatten_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]  # Line 662
    async def _handle_alert(self, alert: Alert) -> None  # Line 680
    async def _alert_cleanup_loop(self) -> None  # Line 710
    async def _metrics_cleanup_loop(self) -> None  # Line 737
    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None  # Line 759
    def get_active_alerts(self, severity: AlertSeverity = None) -> list[Alert]  # Line 763
    def resolve_alert(self, alert_id: str) -> bool  # Line 772
    def get_metrics_summary(self, hours: int = 1) -> dict[str, Any]  # Line 780
    async def health_check(self) -> HealthCheckResult  # Line 814
    async def cleanup(self) -> None  # Line 837
```

### File: data_pipeline.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import get_logger`
- `from src.core.config import Config`
- `from src.core.exceptions import DataError`
- `from src.core.exceptions import DataValidationError`

#### Class: `DataValidationResult`

**Purpose**: Data validation result

```python
class DataValidationResult:
```

#### Class: `PipelineRecord`

**Purpose**: Pipeline processing record

```python
class PipelineRecord:
```

#### Class: `DataTransformation`

**Purpose**: Data transformation utilities

```python
class DataTransformation:
    async def normalize_prices(data: MarketData) -> MarketData  # Line 96
    async def validate_ohlc_consistency(data: MarketData) -> bool  # Line 192
    async def detect_outliers(data: list[MarketData], symbol: str) -> list[bool]  # Line 205
```

#### Class: `DataQualityChecker`

**Purpose**: Comprehensive data quality assessment

```python
class DataQualityChecker:
    def __init__(self, config: Config)  # Line 241
    async def assess_data_quality(self, data: MarketData) -> DataValidationResult  # Line 250
```

#### Class: `DataPipeline`

**Inherits**: BaseComponent
**Purpose**: Data pipeline for financial data processing

```python
class DataPipeline(BaseComponent):
    def __init__(self, ...)  # Line 371
    def _setup_configuration(self) -> None  # Line 420
    async def initialize(self) -> None  # Line 449
    async def _start_workers(self) -> None  # Line 476
    async def process_data(self, data: MarketData | list[MarketData], priority: int = 5) -> dict[str, Any]  # Line 489
    async def _apply_backpressure(self) -> None  # Line 596
    async def _stage_worker(self, stage: PipelineStage, worker_id: int) -> None  # Line 612
    async def _process_stage(self, stage: PipelineStage, record: PipelineRecord) -> None  # Line 650
    def _get_next_stage(self, current_stage: PipelineStage) -> PipelineStage | None  # Line 711
    async def _process_ingestion(self, record: PipelineRecord) -> None  # Line 734
    async def _process_validation(self, record: PipelineRecord) -> None  # Line 740
    async def _process_cleansing(self, record: PipelineRecord) -> None  # Line 752
    async def _process_transformation(self, record: PipelineRecord) -> None  # Line 762
    async def _process_enrichment(self, record: PipelineRecord) -> None  # Line 767
    async def _process_quality_check(self, record: PipelineRecord) -> None  # Line 773
    async def _process_storage(self, record: PipelineRecord) -> None  # Line 783
    async def _process_indexing(self, record: PipelineRecord) -> None  # Line 816
    async def _process_notification(self, record: PipelineRecord) -> None  # Line 822
    async def _handle_processing_failure(self, record: PipelineRecord) -> None  # Line 830
    async def _metrics_monitoring_loop(self) -> None  # Line 862
    async def _quality_monitoring_loop(self) -> None  # Line 883
    def get_metrics(self) -> dict[str, Any]  # Line 906
    async def health_check(self) -> dict[str, Any]  # Line 932
    async def cleanup(self) -> None  # Line 972
    def _validate_pipeline_data_boundary(self, record: PipelineRecord) -> None  # Line 1075
```

### File: ingestion.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import DataSourceError`
- `from src.core.types import IngestionMode`
- `from src.core.types import MarketData`

#### Class: `IngestionConfig`

**Purpose**: Data ingestion configuration

```python
class IngestionConfig:
```

#### Class: `IngestionMetrics`

**Purpose**: Pipeline ingestion metrics

```python
class IngestionMetrics:
```

#### Class: `DataIngestionPipeline`

**Inherits**: BaseComponent
**Purpose**: Comprehensive data ingestion pipeline for multi-source data collection

```python
class DataIngestionPipeline(BaseComponent):
    def __init__(self, config: Config, market_data_source: MarketDataSource | None = None)  # Line 87
    async def initialize(self) -> None  # Line 163
    async def start(self) -> None  # Line 248
    async def _start_real_time_ingestion(self) -> None  # Line 302
    async def _start_batch_ingestion(self) -> None  # Line 350
    async def _ingest_market_data_real_time(self, symbol: str) -> None  # Line 390
    async def _ingest_news_data_real_time(self) -> None  # Line 423
    async def _ingest_social_data_real_time(self) -> None  # Line 472
    async def _ingest_alternative_data_batch(self) -> None  # Line 520
    async def _ingest_historical_market_data(self) -> None  # Line 569
    async def _handle_market_data(self, ticker: Ticker, symbol: str) -> None  # Line 632
    def _add_to_buffer(self, source: str, data: Any) -> None  # Line 690
    async def _add_to_buffer_async(self, source: str, data: Any) -> None  # Line 735
    async def _process_buffers(self) -> None  # Line 772
    async def _collect_metrics(self) -> None  # Line 827
    async def pause(self) -> None  # Line 888
    async def resume(self) -> None  # Line 911
    async def stop(self) -> None  # Line 934
    def register_callback(self, data_type: str, callback: Callable[[Any], None]) -> None  # Line 1033
    def get_status(self) -> dict[str, Any]  # Line 1064
    async def cleanup(self) -> None  # Line 1133
```

### File: processing.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import DataError`
- `from src.core.exceptions import DataValidationError`
- `from src.core.types import MarketData`

#### Class: `ProcessingConfig`

**Purpose**: Data processing configuration

```python
class ProcessingConfig:
```

#### Class: `ProcessingResult`

**Purpose**: Data processing result

```python
class ProcessingResult:
```

#### Class: `DataProcessor`

**Inherits**: BaseComponent
**Purpose**: Comprehensive data processing pipeline for multi-source data transformation

```python
class DataProcessor(BaseComponent):
    def __init__(self, config: Config)  # Line 74
    async def process_market_data(self, data: MarketData, steps: list[ProcessingStep] | None = None) -> ProcessingResult  # Line 141
    async def process_batch(self, ...) -> list[ProcessingResult]  # Line 225
    async def _process_generic_data(self, data: Any, data_type: str, steps: list[ProcessingStep] | None = None) -> ProcessingResult  # Line 258
    async def _normalize_data(self, data: Any, data_type: str) -> Any  # Line 299
    def _normalize_price(self, price: Decimal) -> Decimal  # Line 331
    def _normalize_volume(self, volume: Decimal) -> Decimal  # Line 346
    def _normalize_timestamp(self, timestamp: datetime) -> datetime  # Line 360
    async def _enrich_data(self, data: Any, data_type: str) -> Any  # Line 373
    async def _aggregate_data(self, data: Any, data_type: str) -> Any  # Line 425
    def _calculate_aggregations(self, window: list[MarketData]) -> dict[str, float]  # Line 464
    async def _transform_data(self, data: Any, data_type: str) -> Any  # Line 500
    async def _validate_data(self, data: Any, data_type: str) -> Any  # Line 511
    async def _filter_data(self, data: Any, data_type: str) -> Any  # Line 588
    async def get_aggregated_data(self, symbol: str, exchange: str | None = None) -> dict[str, Any]  # Line 599
    def get_processing_statistics(self) -> dict[str, Any]  # Line 627
    async def reset_windows(self) -> None  # Line 652
    async def cleanup(self) -> None  # Line 663
```

### File: storage.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.types import MarketData`
- `from src.core.types import StorageMode`
- `from src.database.interfaces import DatabaseServiceInterface`

#### Class: `StorageMetrics`

**Purpose**: Storage operation metrics

```python
class StorageMetrics:
```

#### Class: `DataStorageManager`

**Inherits**: BaseComponent
**Purpose**: Data storage manager for pipeline data persistence

```python
class DataStorageManager(BaseComponent):
    def __init__(self, config: Config, database_service: DatabaseServiceInterface | None = None)  # Line 58
    async def initialize(self) -> None  # Line 98
    async def store_market_data(self, data: MarketData) -> bool  # Line 123
    async def _store_real_time(self, data: MarketData) -> bool  # Line 149
    async def _store_to_buffer(self, data: MarketData) -> bool  # Line 189
    async def _flush_buffer(self) -> bool  # Line 211
    async def store_batch(self, data_list: list[MarketData]) -> int  # Line 262
    async def cleanup_old_data(self, days_to_keep: int = 30) -> int  # Line 315
    def get_storage_metrics(self) -> dict[str, Any]  # Line 340
    async def force_flush(self) -> bool  # Line 366
    async def cleanup(self) -> None  # Line 380
    async def _store_batch_to_postgresql(self, data_list: list[MarketData]) -> bool  # Line 399
```

### File: validation.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.types import MarketData`
- `from src.core.types import ValidationLevel`
- `from src.data.quality.validation import ValidationIssue`
- `from src.error_handling import ErrorHandler`

#### Class: `PipelineValidationIssue`

**Purpose**: Pipeline-specific validation issue

```python
class PipelineValidationIssue:
```

#### Class: `PipelineValidator`

**Purpose**: Pipeline-specific data validator for data integrity and quality

```python
class PipelineValidator:
    def __init__(self, config: Config)  # Line 56
    async def validate_pipeline_data(self, data: Any, data_type: str, pipeline_stage: str) -> tuple[bool, list[PipelineValidationIssue]]  # Line 74
    async def _validate_market_data_pipeline(self, data: MarketData, pipeline_stage: str) -> list[PipelineValidationIssue]  # Line 125
    def _validate_ingestion_data(self, data: MarketData) -> list[ValidationIssue]  # Line 141
    def _validate_processing_data(self, data: MarketData) -> list[ValidationIssue]  # Line 174
    def _validate_storage_data(self, data: MarketData) -> list[ValidationIssue]  # Line 194
    def get_validation_statistics(self) -> dict[str, Any]  # Line 214
```

### File: validation_pipeline.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.types import MarketData`
- `from src.data.pipeline.data_pipeline import DataPipeline`
- `from src.data.validation.data_validator import DataValidator`

#### Class: `ValidationStage`

**Inherits**: Enum
**Purpose**: Validation pipeline stage enumeration

```python
class ValidationStage(Enum):
```

#### Class: `ValidationPipelineConfig`

**Inherits**: BaseModel
**Purpose**: Validation pipeline configuration

```python
class ValidationPipelineConfig(BaseModel):
```

#### Class: `ValidationDisposition`

**Inherits**: BaseModel
**Purpose**: Validation disposition result

```python
class ValidationDisposition(BaseModel):
```

#### Class: `ValidationPipelineResult`

**Inherits**: BaseModel
**Purpose**: Comprehensive validation pipeline result

```python
class ValidationPipelineResult(BaseModel):
```

#### Class: `DataValidationPipeline`

**Inherits**: BaseComponent
**Purpose**: Comprehensive data validation pipeline orchestrator

```python
class DataValidationPipeline(BaseComponent):
    def __init__(self, config: Config, validator: DataValidator | None = None)  # Line 122
    def _setup_configuration(self) -> None  # Line 149
    async def initialize(self) -> None  # Line 161
    async def validate_batch(self, data: list[MarketData], symbols: list[str] | None = None) -> ValidationPipelineResult  # Line 190
    async def _execute_validation_stages(self, pipeline_id: str, data: list[MarketData]) -> dict[str, ValidationDisposition]  # Line 278
    def _group_data_by_symbol(self, data: list[MarketData]) -> dict[str, list[MarketData]]  # Line 326
    async def _update_pipeline_stage(self, pipeline_id: str, stage: ValidationStage) -> None  # Line 335
    async def _determine_symbol_disposition(self, symbol: str, validation_results: list[MarketDataValidationResult]) -> ValidationDisposition  # Line 341
    async def _determine_record_disposition(self, validation_result: MarketDataValidationResult) -> ValidationDisposition  # Line 405
    def _determine_action(self, ...) -> ValidationAction  # Line 444
    def _calculate_pipeline_metrics(self, dispositions: dict[str, ValidationDisposition], data: list[MarketData]) -> ValidationMetrics  # Line 480
    def _update_session_metrics(self, pipeline_metrics: ValidationMetrics) -> None  # Line 508
    async def get_quarantined_data(self, symbol: str | None = None) -> dict[str, list[MarketData]]  # Line 518
    async def retry_quarantined_data(self, symbol: str) -> ValidationPipelineResult | None  # Line 524
    async def get_pipeline_status(self) -> dict[str, Any]  # Line 551
    async def health_check(self) -> dict[str, Any]  # Line 561
    async def cleanup(self) -> None  # Line 587
```

### File: cleaning.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import Config`
- `from src.core import MarketData`
- `from src.core import Signal`
- `from src.error_handling import ErrorHandler`

#### Class: `CleaningStrategy`

**Inherits**: Enum
**Purpose**: Data cleaning strategy enumeration

```python
class CleaningStrategy(Enum):
```

#### Class: `OutlierMethod`

**Inherits**: Enum
**Purpose**: Outlier detection method enumeration

```python
class OutlierMethod(Enum):
```

#### Class: `CleaningResult`

**Purpose**: Data cleaning result record

```python
class CleaningResult:
```

#### Class: `DataCleaner`

**Inherits**: BaseComponent
**Purpose**: Comprehensive data cleaning system for market data preprocessing

```python
class DataCleaner(BaseComponent):
    def __init__(self, config: Config, error_handler: ErrorHandler | None = None)  # Line 75
    async def clean_market_data(self, data: MarketData) -> tuple[MarketData, CleaningResult]  # Line 116
    async def clean_signal_data(self, signals: list[Signal]) -> tuple[list[Signal], CleaningResult]  # Line 272
    async def _handle_missing_data(self, data: MarketData) -> tuple[MarketData, int]  # Line 360
    async def _handle_outliers(self, data: MarketData) -> tuple[MarketData, int, int]  # Line 407
    async def _smooth_data(self, data: MarketData) -> MarketData  # Line 519
    async def _remove_duplicates(self, data: MarketData) -> tuple[MarketData, int]  # Line 566
    async def _normalize_data(self, data: MarketData) -> MarketData  # Line 584
    async def _impute_price(self, symbol: str) -> Decimal | None  # Line 610
    async def _impute_volume(self, symbol: str) -> Decimal | None  # Line 631
    async def _is_valid_signal(self, signal: Signal) -> bool  # Line 652
    async def _clean_confidence(self, confidence: float) -> float  # Line 665
    async def _is_duplicate_signal(self, signal: Signal, existing_signals: list[Signal]) -> bool  # Line 673
    def _create_data_hash(self, data: MarketData) -> str  # Line 685
    async def get_cleaning_summary(self) -> dict[str, Any]  # Line 706
```

### File: monitoring.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.types import DriftType`
- `from src.core.types import MarketData`
- `from src.core.types import QualityLevel`

#### Class: `QualityMetric`

**Purpose**: Quality metric record

```python
class QualityMetric:
```

#### Class: `DriftAlert`

**Purpose**: Data drift alert record

```python
class DriftAlert:
```

#### Class: `QualityMonitor`

**Inherits**: BaseComponent
**Purpose**: Comprehensive data quality monitoring system

```python
class QualityMonitor(BaseComponent):
    def __init__(self, config: Config)  # Line 70
    async def monitor_data_quality(self, data: MarketData) -> tuple[float, list[DriftAlert]]  # Line 112
    async def monitor_signal_quality(self, signals: list[Signal]) -> tuple[float, list[DriftAlert]]  # Line 174
    async def generate_quality_report(self, symbol: str | None = None) -> dict[str, Any]  # Line 229
    async def _update_distributions(self, data: MarketData) -> None  # Line 322
    async def _calculate_quality_score(self, data: MarketData) -> float  # Line 351
    async def _detect_drift(self, data: MarketData) -> list[DriftAlert]  # Line 427
    async def _detect_signal_drift(self, signals: list[Signal]) -> list[DriftAlert]  # Line 551
    async def _calculate_distribution_drift(self, recent: list[Decimal], historical: list[Decimal]) -> float  # Line 607
    async def _generate_recommendations(self, report: dict[str, Any]) -> list[str]  # Line 645
    async def get_monitoring_summary(self) -> dict[str, Any]  # Line 687
```

### File: validation.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import Config`
- `from src.core import MarketData`
- `from src.core import Signal`
- `from src.core import ValidationLevel`

#### Class: `DataValidator`

**Inherits**: BaseComponent
**Purpose**: Comprehensive data validation system for market data quality assurance

```python
class DataValidator(BaseComponent):
    def __init__(self, config)  # Line 42
    async def validate_market_data(self, data: MarketData) -> tuple[bool, list[ValidationIssue]]  # Line 99
    async def validate_signal(self, signal: Signal) -> tuple[bool, list[ValidationIssue]]  # Line 239
    async def validate_cross_source_consistency(self, primary_data: MarketData, secondary_data: MarketData) -> tuple[bool, list[ValidationIssue]]  # Line 330
    async def _detect_outliers(self, data: MarketData) -> list[ValidationIssue]  # Line 406
    async def _update_statistics(self, data: MarketData) -> None  # Line 461
    async def get_validation_summary(self) -> dict[str, Any]  # Line 505
    def _is_valid_symbol_format(self, symbol: str) -> bool  # Line 538
    async def cleanup(self) -> None  # Line 553
```

### File: registry.py

**Key Imports:**
- `from src.core.base import BaseComponent`

#### Class: `ServiceRegistry`

**Inherits**: BaseComponent, Generic[ServiceType]
**Purpose**: Generic service registry for managing service instances and dependencies

```python
class ServiceRegistry(BaseComponent, Generic[ServiceType]):
    def __init__(self)  # Line 24
    def register_service(self, name: str, service: ServiceType, metadata: dict[str, Any] | None = None) -> None  # Line 31
    def get_service(self, name: str) -> ServiceType | None  # Line 55
    def unregister_service(self, name: str) -> bool  # Line 71
    def list_services(self) -> dict[str, dict[str, Any]]  # Line 96
    def subscribe_to_event(self, event_name: str, handler: Callable[[dict[str, Any]], None]) -> None  # Line 105
    def _emit_event(self, event_name: str, event_data: dict[str, Any]) -> None  # Line 118
    async def cleanup(self) -> None  # Line 133
```

### File: data_service.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import Config`
- `from src.core import DatabaseError`
- `from src.core import DataError`
- `from src.core import DataValidationError`

#### Class: `DataService`

**Inherits**: BaseComponent
**Purpose**: Simplified DataService for core data management

```python
class DataService(BaseComponent):
    def __init__(self, ...)  # Line 58
    async def initialize(self) -> None  # Line 97
    async def store_market_data(self, ...) -> bool  # Line 112
    def _validate_market_data(self, data_list: list[MarketData], processing_mode: str = 'stream') -> list[MarketData]  # Line 196
    def _transform_to_db_records(self, ...) -> list[MarketDataRecord]  # Line 218
    def _apply_consistent_data_transformation(self, data: MarketData, processing_mode: str) -> MarketData  # Line 266
    def _align_processing_paradigm(self, processing_mode: str, data_count: int) -> str  # Line 297
    async def _store_to_database(self, records: list[MarketDataRecord], processing_mode: str = 'stream') -> None  # Line 310
    async def _update_l1_cache(self, data_list: list[MarketData]) -> None  # Line 324
    async def get_market_data(self, request: DataRequest) -> list[MarketDataRecord]  # Line 335
    def _get_from_l1_cache(self, request: DataRequest) -> list[MarketDataRecord] | None  # Line 386
    async def _get_from_l2_cache(self, request: DataRequest) -> list[MarketDataRecord] | None  # Line 405
    async def _get_from_database(self, request: DataRequest) -> list[MarketDataRecord]  # Line 423
    async def _cache_data(self, request: DataRequest, data: list[MarketDataRecord]) -> None  # Line 460
    def _build_cache_key(self, request: DataRequest) -> str  # Line 496
    async def get_recent_data(self, ...) -> list[MarketData]  # Line 513
    async def get_data_count(self, symbol: str, exchange: str = DEFAULT_EXCHANGE) -> int  # Line 550
    async def get_volatility(self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None  # Line 565
    def _get_technical_indicators(self)  # Line 606
    async def get_rsi(self, symbol: str, period: int = 14, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None  # Line 616
    async def get_sma(self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None  # Line 634
    async def get_ema(self, symbol: str, period: int = 20, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None  # Line 652
    async def get_macd(self, ...) -> dict[str, Decimal] | None  # Line 670
    async def get_bollinger_bands(self, ...) -> dict[str, Decimal] | None  # Line 697
    async def get_atr(self, symbol: str, period: int = 14, exchange: str = DEFAULT_EXCHANGE) -> Decimal | None  # Line 722
    async def health_check(self) -> HealthCheckResult  # Line 745
    def _validate_data_to_database_boundary(self, data_list: list[MarketData], processing_mode: str) -> None  # Line 793
    async def get_metrics(self)  # Line 834
    async def reset_metrics(self) -> None  # Line 844
    async def store_market_data_batch(self, market_data_list: list[MarketData]) -> bool  # Line 850
    async def aggregate_market_data(self, ...) -> list[MarketData]  # Line 897
    async def get_market_data_history(self, symbol: str, limit: int = 100, exchange: str = DEFAULT_EXCHANGE) -> list[MarketData]  # Line 976
    async def cleanup(self) -> None  # Line 1017
```

### File: ml_data_service.py

**Key Imports:**
- `from src.core import HealthStatus`
- `from src.core.base.service import BaseService`
- `from src.core.exceptions import DataError`
- `from src.core.types.base import ConfigDict`
- `from src.utils.decorators import UnifiedDecorator`

#### Class: `MLDataService`

**Inherits**: BaseService
**Purpose**: ML-specific data service providing storage and retrieval for ML artifacts

```python
class MLDataService(BaseService):
    def __init__(self, config: ConfigDict | None = None, correlation_id: str | None = None)  # Line 26
    async def _do_start(self) -> None  # Line 46
    async def _do_stop(self) -> None  # Line 51
    async def store_model_metadata(self, metadata: dict[str, Any]) -> None  # Line 57
    async def update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> None  # Line 78
    async def get_model_by_id(self, model_id: str) -> dict[str, Any] | None  # Line 92
    async def get_all_models(self, ...) -> list[dict[str, Any]]  # Line 97
    async def get_models_by_name_and_type(self, name: str, model_type: str) -> list[dict[str, Any]]  # Line 118
    async def find_models(self, ...) -> list[dict[str, Any]]  # Line 127
    async def delete_model(self, model_id: str) -> None  # Line 154
    async def store_feature_set(self, ...) -> None  # Line 162
    async def get_feature_set(self, symbol: str, feature_set_id: str, version: str | None = None) -> dict[str, Any] | None  # Line 192
    async def update_feature_set_metadata(self, feature_set_id: str, metadata_updates: dict[str, Any]) -> None  # Line 203
    async def list_feature_sets(self, ...) -> list[dict[str, Any]]  # Line 215
    async def delete_feature_set(self, ...) -> int  # Line 239
    async def get_feature_set_versions(self, symbol: str, feature_set_id: str) -> list[str]  # Line 265
    async def store_artifact_info(self, artifact_metadata: dict[str, Any]) -> None  # Line 283
    async def get_artifact_info(self, ...) -> dict[str, Any] | None  # Line 299
    async def list_artifacts(self, ...) -> list[dict[str, Any]]  # Line 311
    async def delete_artifact_info(self, ...) -> None  # Line 334
    async def save_ml_predictions(self, prediction_data: dict[str, Any]) -> None  # Line 349
    async def store_audit_entry(self, service: str, audit_entry: dict[str, Any]) -> None  # Line 361
    async def _service_health_check(self) -> HealthStatus  # Line 372
    def get_ml_data_metrics(self) -> dict[str, Any]  # Line 394
```

### File: adapter.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.exceptions import ConfigurationError`
- `from src.core.exceptions import DataError`
- `from src.data.interfaces import DataSourceInterface`

#### Class: `DataSourceAdapter`

**Inherits**: BaseComponent
**Purpose**: Adapter to standardize different data source interfaces

```python
class DataSourceAdapter(BaseComponent):
    def __init__(self, source_type: str, **config: Any) -> None  # Line 22
    def _create_source(self) -> DataSourceInterface  # Line 35
    async def fetch_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100, **kwargs: Any) -> list[dict[str, Any]]  # Line 61
    async def stream_market_data(self, symbol: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]  # Line 87
    def _adapt_fetch_params(self, symbol: str, timeframe: str, limit: int, **kwargs: Any) -> dict[str, Any]  # Line 108
    def _adapt_stream_params(self, symbol: str, **kwargs: Any) -> dict[str, Any]  # Line 153
    def _standardize_response(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]  # Line 164
    def _standardize_record(self, record: dict[str, Any]) -> dict[str, Any]  # Line 179
    def _symbol_to_binance(self, symbol: str) -> str  # Line 270
    def _symbol_to_coinbase_pair(self, symbol: str) -> str  # Line 275
    def _symbol_to_okx_inst(self, symbol: str) -> str  # Line 283
    def _timeframe_to_binance_interval(self, timeframe: str) -> str  # Line 288
    def _timeframe_to_coinbase_granularity(self, timeframe: str) -> int  # Line 307
    def _timeframe_to_okx_bar(self, timeframe: str) -> str  # Line 322
    async def connect(self) -> None  # Line 328
    async def disconnect(self) -> None  # Line 336
    def is_connected(self) -> bool  # Line 346
```

### File: alternative_data.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import get_logger`
- `from src.core.config import Config`
- `from src.core.exceptions import DataSourceError`
- `from src.core.logging import get_logger`

#### Class: `DataType`

**Inherits**: Enum
**Purpose**: Alternative data type enumeration

```python
class DataType(Enum):
```

#### Class: `AlternativeDataPoint`

**Purpose**: Alternative data point structure

```python
class AlternativeDataPoint:
```

#### Class: `EconomicIndicator`

**Purpose**: Economic indicator data structure

```python
class EconomicIndicator:
```

#### Class: `AlternativeDataSource`

**Inherits**: BaseComponent
**Purpose**: Alternative data source for economic and environmental indicators

```python
class AlternativeDataSource(BaseComponent):
    def __init__(self, config: Config)  # Line 89
    async def initialize(self) -> None  # Line 143
    async def _test_data_sources(self) -> None  # Line 175
    async def get_economic_indicators(self, indicators: list[str], days_back: int = 30) -> list[EconomicIndicator]  # Line 214
    async def _fetch_fred_indicator(self, indicator_id: str, start_date: str, end_date: str) -> list[EconomicIndicator]  # Line 267
    async def get_weather_data(self, locations: list[str], days_back: int = 7) -> list[AlternativeDataPoint]  # Line 329
    async def _fetch_weather_data(self, location: str, days_back: int) -> list[AlternativeDataPoint]  # Line 374
    async def get_satellite_data(self, regions: list[str], indicators: list[str], days_back: int = 30) -> list[AlternativeDataPoint]  # Line 446
    async def _fetch_satellite_data(self, region: str, indicator: str, days_back: int) -> list[AlternativeDataPoint]  # Line 495
    async def get_comprehensive_dataset(self, config: dict[str, Any]) -> dict[str, list[AlternativeDataPoint]]  # Line 553
    async def get_source_statistics(self) -> dict[str, Any]  # Line 614
    async def cleanup(self) -> None  # Line 634
```

### File: base.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.data.interfaces import DataCacheInterface`
- `from src.data.interfaces import DataSourceInterface`

#### Class: `RateLimiter`

**Purpose**: Simple rate limiter for API calls

```python
class RateLimiter:
    def __init__(self, calls_per_second: float = 10)  # Line 15
    async def __aenter__(self)  # Line 27
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 39
```

#### Class: `SimpleCache`

**Inherits**: DataCacheInterface
**Purpose**: Simple in-memory cache implementation

```python
class SimpleCache(DataCacheInterface):
    def __init__(self) -> None  # Line 47
    async def get(self, key: str) -> Any | None  # Line 52
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None  # Line 68
    async def delete(self, key: str) -> bool  # Line 74
    async def clear(self) -> None  # Line 83
    async def exists(self, key: str) -> bool  # Line 88
    async def health_check(self) -> dict[str, Any]  # Line 92
    async def initialize(self) -> None  # Line 100
    async def cleanup(self) -> None  # Line 105
```

#### Class: `BaseDataSource`

**Inherits**: DataSourceInterface
**Purpose**: Base implementation with common functionality for data sources

```python
class BaseDataSource(DataSourceInterface):
    def __init__(self, cache: DataCacheInterface | None = None, rate_limit: float = 10)  # Line 113
    async def fetch_with_cache(self, cache_key: str, fetch_func: Callable[[], Awaitable[Any]], ttl: int = 60) -> Any  # Line 126
    async def connect(self) -> None  # Line 156
    async def disconnect(self) -> None  # Line 161
    def is_connected(self) -> bool  # Line 166
    async def fetch(self, symbol: str, timeframe: str, limit: int = 100, **kwargs) -> list[dict[str, Any]]  # Line 171
    async def stream(self, symbol: str, **kwargs) -> AsyncIterator[dict[str, Any]]  # Line 177
```

### File: market_data.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import DataSourceError`
- `from src.core.types import MarketData`
- `from src.core.types import OrderBook`

#### Class: `DataStreamType`

**Inherits**: Enum
**Purpose**: Data stream type enumeration

```python
class DataStreamType(Enum):
```

#### Class: `DataSubscription`

**Purpose**: Data subscription configuration

```python
class DataSubscription:
```

#### Class: `MarketDataSource`

**Inherits**: BaseComponent
**Purpose**: Market data source for real-time and historical data ingestion

```python
class MarketDataSource(BaseComponent):
    def __init__(self, config: Config, exchange_factory = None)  # Line 87
    async def initialize(self) -> None  # Line 132
    async def subscribe_to_ticker(self, exchange_name: str, symbol: str, callback: Callable[[Ticker], None]) -> str  # Line 214
    async def get_historical_data(self, ...) -> list[MarketData]  # Line 259
    async def _ticker_stream(self, exchange_name: str) -> None  # Line 321
    async def unsubscribe(self, subscription_id: str) -> bool  # Line 425
    async def get_market_data_summary(self) -> dict[str, Any]  # Line 465
    async def cleanup(self) -> None  # Line 479
    async def get_error_analytics(self) -> dict[str, Any]  # Line 568
```

### File: news_data.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import get_logger`
- `from src.core.config import Config`
- `from src.core.exceptions import DataSourceError`
- `from src.core.logging import get_logger`

#### Class: `NewsArticle`

**Purpose**: News article data structure

```python
class NewsArticle:
```

#### Class: `NewsDataSource`

**Inherits**: BaseComponent
**Purpose**: News data source for sentiment analysis and market impact assessment

```python
class NewsDataSource(BaseComponent):
    def __init__(self, config: Config)  # Line 64
    async def initialize(self) -> None  # Line 109
    async def _test_connection(self) -> None  # Line 144
    async def get_news_for_symbol(self, symbol: str, hours_back: int = 24, max_articles: int = 50) -> list[NewsArticle]  # Line 162
    async def _fetch_articles(self, query: str, from_date: datetime, to_date: datetime, page_size: int = 50) -> list[NewsArticle]  # Line 222
    def _parse_article(self, article_data: dict[str, Any], query: str) -> NewsArticle | None  # Line 256
    def _build_search_queries(self, symbol: str) -> list[str]  # Line 301
    def _analyze_sentiment(self, text: str) -> tuple[NewsSentiment, float]  # Line 321
    def _extract_symbols(self, article_data: dict[str, Any], query: str) -> list[str]  # Line 350
    def _calculate_relevance(self, article_data: dict[str, Any], query: str) -> float  # Line 364
    def _deduplicate_articles(self, articles: list[NewsArticle]) -> list[NewsArticle]  # Line 392
    def _filter_and_score_articles(self, articles: list[NewsArticle], symbol: str) -> list[NewsArticle]  # Line 406
    async def get_market_sentiment(self, symbols: list[str]) -> dict[str, dict[str, float]]  # Line 426
    async def cleanup(self) -> None  # Line 477
```

### File: social_media.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import get_logger`
- `from src.core.config import Config`
- `from src.core.exceptions import DataSourceError`
- `from src.core.logging import get_logger`

#### Class: `SocialPost`

**Purpose**: Social media post data structure

```python
class SocialPost:
```

#### Class: `SocialMetrics`

**Purpose**: Aggregated social metrics for a symbol

```python
class SocialMetrics:
```

#### Class: `SocialMediaDataSource`

**Inherits**: BaseComponent
**Purpose**: Social media data source for sentiment analysis and trend detection

```python
class SocialMediaDataSource(BaseComponent):
    def __init__(self, config: Config)  # Line 84
    async def initialize(self) -> None  # Line 127
    async def _test_connections(self) -> None  # Line 159
    async def get_social_sentiment(self, symbol: str, hours_back: int = 24, platforms: list[str] | None = None) -> SocialMetrics  # Line 188
    async def _get_twitter_posts(self, symbol: str, hours_back: int) -> list[SocialPost]  # Line 238
    async def _get_reddit_posts(self, symbol: str, hours_back: int) -> list[SocialPost]  # Line 281
    def _analyze_social_sentiment(self, content: str) -> tuple[SocialSentiment, float]  # Line 324
    def _calculate_social_metrics(self, symbol: str, posts: list[SocialPost], hours_back: int) -> SocialMetrics  # Line 385
    async def get_trending_symbols(self, limit: int = 10) -> list[dict[str, Any]]  # Line 439
    async def monitor_symbol_mentions(self, symbols: list[str], callback: Callable[[str, list[SocialPost]], None]) -> None  # Line 482
    async def get_platform_statistics(self) -> dict[str, Any]  # Line 530
    async def cleanup(self) -> None  # Line 545
```

### File: database_storage.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import Config`
- `from src.core import DataError`
- `from src.core.exceptions import DatabaseError`
- `from src.data.interfaces import DataStorageInterface`

#### Class: `DatabaseStorage`

**Inherits**: BaseComponent, DataStorageInterface
**Purpose**: Database storage implementation

```python
class DatabaseStorage(BaseComponent, DataStorageInterface):
    def __init__(self, config: Config, database_service: 'DatabaseServiceInterface | None' = None)  # Line 22
    async def initialize(self) -> None  # Line 35
    async def store_records(self, records: list[MarketDataRecord]) -> bool  # Line 41
    async def retrieve_records(self, request: DataRequest) -> list[MarketDataRecord]  # Line 60
    async def get_record_count(self, symbol: str, exchange: str) -> int  # Line 98
    async def health_check(self) -> dict[str, Any]  # Line 119
    async def cleanup(self) -> None  # Line 139
```

### File: streaming_service.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import ConfigurationError`
- `from src.core.exceptions import ConnectionError`
- `from src.core.exceptions import DataError`

#### Class: `StreamState`

**Inherits**: Enum
**Purpose**: Streaming connection state

```python
class StreamState(Enum):
```

#### Class: `BufferStrategy`

**Inherits**: Enum
**Purpose**: Buffer overflow strategy

```python
class BufferStrategy(Enum):
```

#### Class: `StreamMetrics`

**Purpose**: Streaming metrics

```python
class StreamMetrics:
```

#### Class: `StreamConfig`

**Inherits**: BaseModel
**Purpose**: Stream configuration model

```python
class StreamConfig(BaseModel):
```

#### Class: `StreamBuffer`

**Purpose**: High-performance streaming data buffer

```python
class StreamBuffer:
    def __init__(self, config: StreamConfig)  # Line 117
    async def put(self, item: Any) -> bool  # Line 124
    async def get(self, timeout: float | None = None) -> Any | None  # Line 143
    async def get_batch(self, max_size: int, timeout: float = 1.0) -> list[Any]  # Line 156
    def size(self) -> int  # Line 170
    def utilization(self) -> float  # Line 174
    def dropped_count(self) -> int  # Line 178
    async def clear(self) -> None  # Line 182
```

#### Class: `WebSocketConnection`

**Purpose**: WebSocket connection manager with automatic reconnection

```python
class WebSocketConnection:
    def __init__(self, config: StreamConfig, message_handler: Callable)  # Line 192
    async def connect(self) -> bool  # Line 202
    async def _send_subscription(self) -> None  # Line 271
    async def listen(self) -> AsyncGenerator[dict[str, Any], None]  # Line 288
    async def disconnect(self) -> None  # Line 324
    async def _heartbeat_monitor(self) -> None  # Line 360
    def is_connected(self) -> bool  # Line 391
    def get_uptime(self) -> timedelta  # Line 395
```

#### Class: `StreamingDataService`

**Inherits**: BaseComponent
**Purpose**: Enterprise-grade real-time market data streaming service

```python
class StreamingDataService(BaseComponent):
    def __init__(self, ...)  # Line 415
    async def initialize(self) -> None  # Line 453
    async def _load_stream_configurations(self) -> None  # Line 474
    async def _setup_message_handlers(self) -> None  # Line 487
    async def add_stream(self, exchange: str, config: StreamConfig) -> bool  # Line 497
    async def start_stream(self, exchange: str) -> bool  # Line 513
    async def stop_stream(self, exchange: str) -> bool  # Line 551
    async def _stream_task(self, exchange: str) -> None  # Line 578
    async def _listen_with_timeout(self, connection: WebSocketConnection, shutdown_event: asyncio.Event) -> AsyncGenerator[dict[str, Any], None]  # Line 635
    async def _processor_task(self, exchange: str) -> None  # Line 646
    async def _process_message_batch(self, exchange: str, messages: list[dict[str, Any]]) -> None  # Line 688
    async def _is_duplicate(self, exchange: str, data: MarketData) -> bool  # Line 848
    async def _reconnect(self, exchange: str) -> None  # Line 872
    async def _handle_generic_message(self, message: dict[str, Any], exchange: str) -> MarketData | None  # Line 907
    async def _handle_binance_message(self, message: dict[str, Any], exchange: str) -> MarketData | None  # Line 943
    async def _handle_coinbase_message(self, message: dict[str, Any], exchange: str) -> MarketData | None  # Line 969
    async def _handle_okx_message(self, message: dict[str, Any], exchange: str) -> MarketData | None  # Line 994
    async def get_stream_status(self, exchange: str | None = None) -> dict[str, Any]  # Line 1020
    def get_metrics(self) -> dict[str, Any]  # Line 1044
    async def health_check(self) -> dict[str, Any]  # Line 1048
    async def cleanup(self) -> None  # Line 1069
```

### File: types.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.data.constants import MAX_CACHE_TTL_SECONDS`
- `from src.data.constants import MAX_DATA_LIMIT`
- `from src.data.constants import MAX_EXCHANGE_LENGTH`
- `from src.data.constants import MAX_LOOKBACK_PERIOD`

#### Class: `CacheLevel`

**Inherits**: Enum
**Purpose**: Cache level enumeration

```python
class CacheLevel(Enum):
```

#### Class: `DataPipelineStage`

**Inherits**: Enum
**Purpose**: Data pipeline stage enumeration

```python
class DataPipelineStage(Enum):
```

#### Class: `DataMetrics`

**Purpose**: Data processing metrics

```python
class DataMetrics:
```

#### Class: `DataRequest`

**Inherits**: BaseModel
**Purpose**: Data request model with validation

```python
class DataRequest(BaseModel):
    def validate_time_range(cls, v: datetime | None, info: Any) -> datetime | None  # Line 81
```

#### Class: `FeatureRequest`

**Inherits**: BaseModel
**Purpose**: Feature calculation request model

```python
class FeatureRequest(BaseModel):
```

### File: core.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.data.interfaces import DataValidatorInterface`

#### Class: `DataValidationPipeline`

**Inherits**: BaseComponent
**Purpose**: Centralized validation pipeline for data

```python
class DataValidationPipeline(BaseComponent):
    def __init__(self) -> None  # Line 12
    def add_validator(self, validator: DataValidatorInterface) -> 'DataValidationPipeline'  # Line 17
    def remove_validator(self, validator_type: type) -> bool  # Line 31
    def validate(self, data: Any) -> tuple[bool, list[str]]  # Line 48
    def validate_batch(self, data_list: list[Any]) -> list[tuple[bool, list[str]]]  # Line 106
    def clear(self) -> None  # Line 121
    def get_validator_count(self) -> int  # Line 126
```

### File: data_validator.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import Config`
- `from src.core import HealthCheckResult`
- `from src.core import HealthStatus`
- `from src.core import MarketData`

#### Class: `MarketDataValidationResult`

**Inherits**: BaseModel
**Purpose**: Market data validation result

```python
class MarketDataValidationResult(BaseModel):
```

#### Class: `DataValidator`

**Inherits**: BaseComponent
**Purpose**: Comprehensive data validator for financial data quality assurance

```python
class DataValidator(BaseComponent):
    def __init__(self, config: Config)  # Line 55
    def _setup_configuration(self) -> None  # Line 74
    async def validate_market_data(self, data: MarketData | list[MarketData], include_statistical: bool = True) -> MarketDataValidationResult | list[MarketDataValidationResult]  # Line 99
    async def _validate_single(self, data: MarketData) -> MarketDataValidationResult  # Line 122
    async def _validate_batch(self, data_list: list[MarketData]) -> list[MarketDataValidationResult]  # Line 147
    def _calculate_quality_score(self, data: MarketData, errors: list[str]) -> float  # Line 157
    async def add_custom_rule(self, rule_name: str, rule_config: dict[str, Any]) -> bool  # Line 183
    async def disable_rule(self, rule_name: str) -> bool  # Line 193
    async def enable_rule(self, rule_name: str) -> bool  # Line 198
    async def get_validation_stats(self) -> dict[str, Any]  # Line 203
    async def health_check(self) -> HealthCheckResult  # Line 218
    async def cleanup(self) -> None  # Line 233
```

#### Functions:

```python
def _get_utc_now()  # Line 27
```

### File: market_data_validator.py

**Key Imports:**
- `from src.core import BaseComponent`
- `from src.core import MarketData`
- `from src.data.interfaces import DataValidatorInterface`
- `from src.data.interfaces import ServiceDataValidatorInterface`
- `from src.utils.validation.market_data_validation import MarketDataValidator`

#### Class: `MarketDataValidator`

**Inherits**: BaseComponent, DataValidatorInterface, ServiceDataValidatorInterface
**Purpose**: Market data validator implementation that uses consolidated validation utilities

```python
class MarketDataValidator(BaseComponent, DataValidatorInterface, ServiceDataValidatorInterface):
    def __init__(self)  # Line 22
    async def validate_market_data(self, data_list: list[MarketData]) -> list[MarketData]  # Line 32
    def get_validation_errors(self) -> list[str]  # Line 51
    def validate(self, data) -> bool  # Line 55
    def get_errors(self) -> list[str]  # Line 74
    def reset(self) -> None  # Line 78
    async def health_check(self) -> dict[str, Any]  # Line 82
```

### File: validators.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.utils.validation.base_validator import BaseRecordValidator`
- `from src.utils.validation.market_data_validation import MarketDataValidationUtils`

#### Class: `PriceValidator`

**Inherits**: BaseRecordValidator
**Purpose**: Validator for price data using consolidated utilities

```python
class PriceValidator(BaseRecordValidator):
    def __init__(self, min_price: Decimal = Any, max_price: Decimal = Any)  # Line 19
    def _validate_record(self, record: dict, index: int | None = None) -> bool  # Line 31
```

#### Class: `VolumeValidator`

**Inherits**: BaseRecordValidator
**Purpose**: Validator for volume data using consolidated utilities

```python
class VolumeValidator(BaseRecordValidator):
    def __init__(self, min_volume: Decimal = Any)  # Line 51
    def _validate_record(self, record: dict, index: int | None = None) -> bool  # Line 61
```

#### Class: `TimestampValidator`

**Inherits**: BaseRecordValidator
**Purpose**: Validator for timestamp data using consolidated utilities

```python
class TimestampValidator(BaseRecordValidator):
    def __init__(self, ...)  # Line 81
    def validate(self, data: Any) -> bool  # Line 101
    def _validate_record(self, record: dict, index: int | None = None) -> bool  # Line 106
    def reset(self) -> None  # Line 156
```

#### Class: `SchemaValidator`

**Inherits**: BaseRecordValidator
**Purpose**: Validator for data schema

```python
class SchemaValidator(BaseRecordValidator):
    def __init__(self, ...)  # Line 165
    def _validate_record(self, record: dict, index: int | None = None) -> bool  # Line 177
```

### File: vectorized_processor.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import DataProcessingError`
- `from src.core.logging import get_logger`
- `from src.error_handling import FallbackStrategy`
- `from src.error_handling import with_fallback`

#### Class: `HighPerformanceDataBuffer`

**Purpose**: High-performance circular buffer optimized for streaming market data

```python
class HighPerformanceDataBuffer:
    def __init__(self, size: int = 100000, num_fields: int = 8) -> None  # Line 124
    def _setup_memory_map(self) -> None  # Line 151
    def append_batch(self, data: np.ndarray) -> None  # Line 167
    def get_recent_vectorized(self, n: int) -> np.ndarray  # Line 188
```

#### Class: `IndicatorCache`

**Purpose**: Cache for calculated indicators to avoid recalculation

```python
class IndicatorCache:
    def __post_init__(self) -> None  # Line 213
    def is_valid(self, key: str) -> bool  # Line 216
    def get(self, key: str) -> np.ndarray | None  # Line 220
    def set(self, key: str, value: np.ndarray) -> None  # Line 227
```

#### Class: `VectorizedProcessor`

**Purpose**: High-performance market data processor using vectorized operations

```python
class VectorizedProcessor:
    def __init__(self, ...) -> None  # Line 246
    async def process_market_data_batch(self, market_data: list[dict[str, Any]]) -> dict[str, np.ndarray]  # Line 309
    def _convert_to_numpy(self, market_data: list[dict[str, Any]]) -> np.ndarray  # Line 369
    async def _calculate_indicators_parallel(self, prices: np.ndarray, volumes: np.ndarray) -> dict[str, np.ndarray]  # Line 390
    def calculate_real_time_indicators(self, current_price: Decimal) -> dict[str, Decimal]  # Line 449
    def _update_metrics(self, batch_size: int, processing_time_us: float) -> None  # Line 496
    def get_performance_metrics(self) -> dict[str, Any]  # Line 520
    def cleanup(self) -> None  # Line 537
```

#### Functions:

```python
def calculate_volume_profile_vectorized(prices: np.ndarray, volumes: np.ndarray, num_bins: int = 50) -> tuple[np.ndarray, np.ndarray]  # Line 96
def benchmark_vectorized_vs_sequential(prices: np.ndarray, iterations: int = 1000) -> dict[str, float]  # Line 572
```

---
**Generated**: Complete reference for data module
**Total Classes**: 123
**Total Functions**: 20