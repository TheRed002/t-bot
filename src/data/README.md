# Data Module

The Data Module (`src/data`) provides end-to-end market and alternative data capabilities for the trading system. It covers data sources, ingestion pipelines, processing, feature engineering, quality monitoring, and validation. It integrates tightly with `src.core`, `src.utils`, `src.error_handling`, and `src.database`.

## What This Module Does
- Data sources for market, news, social media, and alternative data
- Ingestion pipelines for real-time and batch collection
- Processing: normalization, enrichment, aggregation, transformation, filtering
- Feature engineering: statistical, technical indicators, and alternative features
- Quality: ongoing monitoring, drift detection, quality scoring, reporting
- Validation: real-time validator for incoming data and pipeline-specific checks
- Storage: writing time-series data to InfluxDB via `src.database.influxdb_client`

## What This Module Does NOT Do
- Execute trading decisions or orders
- Manage exchange connections beyond data retrieval
- Persist SQL state (handled by `src.database`)
- Implement strategies, risk management, or capital management logic

---

## Submodules
- `features/`: Feature engineering
  - `alternative_features.py`: Alternative data derived features (news/social/economic/microstructure)
  - `statistical_features.py`: Rolling stats, autocorrelation, cross-correlation, regime, seasonality
  - `technical_indicators.py`: TA-Lib indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, etc.)
- `pipeline/`: Orchestration for data flow
  - `ingestion.py`: Real-time and batch ingestion, buffers, metrics, callbacks
  - `processing.py`: Normalization, enrichment, aggregation, transform, validate, filter
  - `storage.py`: Buffered and real-time writes to InfluxDB with metrics
  - `validation.py`: Pipeline-specific validations and statistics
- `quality/`: Quality management
  - `cleaning.py`: Missing data, outliers, smoothing, duplicates, normalization
  - `monitoring.py`: Drift detection, quality scoring, alerts, reporting
  - `validation.py`: Real-time data validation (schema, ranges, outliers, freshness, cross-source)
- `sources/`: Data providers
  - `market_data.py`: Market data streams and historical queries via exchanges
  - `news_data.py`: News API integration with basic NLP sentiment and relevance
  - `social_media.py`: Social sentiment metrics (Twitter/Reddit simulation)
  - `alternative_data.py`: Economic indicators, weather, satellite (simulated)

---

## File Reference (Key Classes/Functions)

### features/alternative_features.py
- `AlternativeFeatureCalculator(config: Config)`
  - `set_data_sources(news_source, social_source, alt_data_source) -> None`
  - `initialize() -> None`
  - `calculate_news_sentiment(symbol: str, lookback_hours: int | None) -> AlternativeResult`
  - `calculate_social_sentiment(symbol: str, lookback_hours: int | None) -> AlternativeResult`
  - `calculate_economic_indicators(symbol: str, lookback_hours: int | None) -> AlternativeResult`
  - `calculate_market_microstructure(symbol: str, lookback_hours: int | None) -> AlternativeResult`
  - `calculate_batch_features(symbol: str, features: list[str]) -> dict[str, AlternativeResult]`
  - `get_calculation_summary() -> dict[str, Any]`

### features/statistical_features.py
- `StatisticalFeatureCalculator(config: Config)`
  - `add_market_data(data: MarketData) -> None`
  - `calculate_rolling_stats(symbol: str, window: int | None, field: str) -> StatisticalResult`
  - `calculate_autocorrelation(symbol: str, max_lags: int | None, field: str) -> StatisticalResult`
  - `detect_regime(symbol: str, window: int | None, field: str) -> StatisticalResult`
  - `calculate_cross_correlation(symbol1: str, symbol2: str, max_lags: int, field: str) -> StatisticalResult`
  - `detect_seasonality(symbol: str, field: str) -> StatisticalResult`
  - `calculate_batch_features(symbol: str, features: list[str]) -> dict[str, StatisticalResult]`
  - `get_calculation_summary() -> dict[str, Any]`

### features/technical_indicators.py
- `TechnicalIndicatorCalculator(config: Config)`
  - `add_market_data(data: MarketData) -> None`
  - `calculate_sma(symbol: str, period: int | None, field: str) -> IndicatorResult`
  - `calculate_ema(symbol: str, period: int | None, field: str) -> IndicatorResult`
  - `calculate_rsi(symbol: str, period: int | None) -> IndicatorResult`
  - `calculate_macd(symbol: str, fast_period: int | None, slow_period: int | None, signal_period: int | None) -> IndicatorResult`
  - `calculate_bollinger_bands(symbol: str, period: int | None, std_dev: float) -> IndicatorResult`
  - `calculate_atr(symbol: str, period: int | None) -> IndicatorResult`
  - `calculate_batch_indicators(symbol: str, indicators: list[str]) -> dict[str, IndicatorResult]`
  - `get_calculation_summary() -> dict[str, Any]`

### pipeline/ingestion.py
- `DataIngestionPipeline(config: Config)`
  - `initialize() -> None`
  - `start() -> None`, `pause() -> None`, `resume() -> None`, `stop() -> None`
  - `register_callback(data_type: str, callback: Callable[[Any], None]) -> None`
  - `get_status() -> dict[str, Any]`

### pipeline/processing.py
- `DataProcessor(config: Config)`
  - `process_market_data(data: MarketData, steps: list[ProcessingStep] | None) -> ProcessingResult`
  - `process_batch(data_list: list[Any], data_type: str, steps: list[ProcessingStep] | None) -> list[ProcessingResult]`
  - `get_aggregated_data(symbol: str, exchange: str | None) -> dict[str, Any]`
  - `get_processing_statistics() -> dict[str, Any]`
  - `reset_windows() -> None`, `cleanup() -> None`

### pipeline/storage.py
- `DataStorageManager(config: Config)`
  - `store_market_data(data: MarketData) -> bool`
  - `store_batch(data_list: list[MarketData]) -> int`
  - `cleanup_old_data(days_to_keep: int) -> int`
  - `get_storage_metrics() -> dict[str, Any]`, `force_flush() -> bool`, `cleanup() -> None`

### pipeline/validation.py
- `PipelineValidator(config: Config)`
  - `validate_pipeline_data(data: Any, data_type: str, pipeline_stage: str) -> tuple[bool, list[PipelineValidationIssue]]`
  - `get_validation_statistics() -> dict[str, Any]`

### quality/cleaning.py
- `DataCleaner(config: Config | dict[str, Any])`
  - `clean_market_data(data: MarketData) -> tuple[MarketData, CleaningResult]`
  - `clean_signal_data(signals: list[Signal]) -> tuple[list[Signal], CleaningResult]`
  - `get_cleaning_summary() -> dict[str, Any]`

### quality/monitoring.py
- `QualityMonitor(config: Config)`
  - `monitor_data_quality(data: MarketData) -> tuple[float, list[DriftAlert]]`
  - `monitor_signal_quality(signals: list[Signal]) -> tuple[float, list[DriftAlert]]`
  - `generate_quality_report(symbol: str | None = None) -> dict[str, Any]`
  - `get_monitoring_summary() -> dict[str, Any]`

### quality/validation.py
- `DataValidator(config: Config)`
  - `validate_market_data(data: MarketData) -> tuple[bool, list[ValidationIssue]]`
  - `validate_signal(signal: Signal) -> tuple[bool, list[ValidationIssue]]`
  - `validate_cross_source_consistency(primary_data: MarketData, secondary_data: MarketData) -> tuple[bool, list[ValidationIssue]]`
  - `get_validation_summary() -> dict[str, Any]`

### sources/market_data.py
- `MarketDataSource(config: Config)`
  - `initialize() -> None`
  - `subscribe_to_ticker(exchange_name: str, symbol: str, callback: Callable[[Ticker], None]) -> str`
  - `get_historical_data(exchange_name: str, symbol: str, start_time: datetime, end_time: datetime, interval: str) -> list[MarketData]`
  - `get_market_data_summary() -> dict[str, Any]`, `unsubscribe(subscription_id: str) -> bool`, `cleanup() -> None`

### sources/news_data.py
- `NewsDataSource(config: Config)`
  - `initialize() -> None`, `get_news_for_symbol(symbol: str, hours_back: int, max_articles: int) -> list[NewsArticle]`
  - `get_market_sentiment(symbols: list[str]) -> dict[str, dict[str, float]]`, `cleanup() -> None`

### sources/social_media.py
- `SocialMediaDataSource(config: Config)`
  - `initialize() -> None`, `get_social_sentiment(symbol: str, hours_back: int, platforms: list[str] | None) -> SocialMetrics`
  - `get_trending_symbols(limit: int) -> list[dict[str, Any]]`, `monitor_symbol_mentions(symbols: list[str], callback: Callable) -> None`, `cleanup() -> None`

### sources/alternative_data.py
- `AlternativeDataSource(config: Config)`
  - `initialize() -> None`, `get_economic_indicators(indicators: list[str], days_back: int) -> list[EconomicIndicator]`
  - `get_weather_data(locations: list[str], days_back: int) -> list[AlternativeDataPoint]`
  - `get_satellite_data(regions: list[str], indicators: list[str], days_back: int) -> list[AlternativeDataPoint]`
  - `get_comprehensive_dataset(config: dict[str, Any]) -> dict[str, list[AlternativeDataPoint]]`, `get_source_statistics() -> dict[str, Any]`, `cleanup() -> None`

---

## Import Relationships
- Imports from: `src.core` (types, config, exceptions, logging), `src.utils` (decorators, validators, helpers), `src.error_handling`, `src.exchanges`, `src.database`
- Exported functionality is consumed by: `src.strategies`, `src.risk_management`, `src.capital_management`, `src.database` (storage), and integration tests

## Where This Module Should Be Imported
- `src.strategies` for feature engineering and data validation
- `src.risk_management` for risk metrics inputs and validation hooks
- `src.exchanges` for market data integration through `MarketDataSource`
- `src.database` for storage via `DataStorageManager`
- `src.core` is a dependency only; do not import this module from core

## Local Dependencies
- `src.core` for models/enums/config/logging
- `src.utils` for decorators, validation, formatting, helpers
- `src.error_handling` for `ErrorHandler`
- `src.database` for `InfluxDBClientWrapper` and DB sessions

---

## Notes
- Some data providers are simulated for testing (news, social, satellite). Replace stubs with real APIs in production.
- Quality/validation components are designed to be fast and async-friendly.
- Pipelines support hybrid modes and buffered storage.


