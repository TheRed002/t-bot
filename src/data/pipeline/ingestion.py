"""
Data Ingestion Pipeline

This module provides robust data ingestion capabilities:
- Real-time stream processing from multiple sources
- Batch data collection for historical data
- Data normalization across different sources
- Timestamp synchronization and alignment
- Error handling and retry logic

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import json

# Import from P-001 core components
from src.core.types import (
    MarketData, Ticker, OrderBook, Trade,
    IngestionMode, PipelineStatus
)
from src.core.exceptions import DataError, DataSourceError, ValidationError
from src.core.config import Config
from src.core.logging import get_logger

# Import from P-002 database components
from src.database.connection import get_async_session

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# Import from P-007A utilities
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_price, validate_quantity
from src.utils.formatters import format_currency

# Import data sources
from src.data.sources.market_data import MarketDataSource
from src.data.sources.news_data import NewsDataSource
from src.data.sources.social_media import SocialMediaDataSource
from src.data.sources.alternative_data import AlternativeDataSource

logger = get_logger(__name__)


# IngestionMode and PipelineStatus are now imported from core.types


@dataclass
class IngestionConfig:
    """Data ingestion configuration"""
    mode: IngestionMode
    sources: List[str]
    symbols: List[str]
    batch_size: int
    update_interval: int
    buffer_size: int
    error_threshold: int
    retry_attempts: int


@dataclass
class IngestionMetrics:
    """Pipeline ingestion metrics"""
    total_records_processed: int
    successful_ingestions: int
    failed_ingestions: int
    avg_processing_time: float
    records_per_second: float
    buffer_utilization: float
    error_rate: float
    last_update_time: datetime


class DataIngestionPipeline:
    """
    Comprehensive data ingestion pipeline for multi-source data collection.

    This class orchestrates data ingestion from multiple sources including
    market data, news, social media, and alternative data sources.
    """

    def __init__(self, config: Config):
        """
        Initialize data ingestion pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Pipeline configuration
        pipeline_config = getattr(config, 'data_pipeline', {})
        if hasattr(pipeline_config, 'get'):
            self.ingestion_config = IngestionConfig(
                mode=IngestionMode(pipeline_config.get('mode', 'real_time')),
                sources=pipeline_config.get('sources', ['market_data']),
                symbols=pipeline_config.get('symbols', ['BTC', 'ETH']),
                batch_size=pipeline_config.get('batch_size', 100),
                update_interval=pipeline_config.get('update_interval', 5),
                buffer_size=pipeline_config.get('buffer_size', 1000),
                error_threshold=pipeline_config.get('error_threshold', 10),
                retry_attempts=pipeline_config.get('retry_attempts', 3)
            )
        else:
            self.ingestion_config = IngestionConfig(
                mode=IngestionMode('real_time'),
                sources=['market_data'],
                symbols=['BTC', 'ETH'],
                batch_size=100,
                update_interval=5,
                buffer_size=1000,
                error_threshold=10,
                retry_attempts=3
            )

        # Data sources
        self.market_data_source: Optional[MarketDataSource] = None
        self.news_data_source: Optional[NewsDataSource] = None
        self.social_media_source: Optional[SocialMediaDataSource] = None
        self.alternative_data_source: Optional[AlternativeDataSource] = None

        # Pipeline state
        self.status = PipelineStatus.STOPPED
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.data_buffers: Dict[str, List[Any]] = {}

        # Metrics and monitoring
        self.metrics = IngestionMetrics(
            total_records_processed=0,
            successful_ingestions=0,
            failed_ingestions=0,
            avg_processing_time=0.0,
            records_per_second=0.0,
            buffer_utilization=0.0,
            error_rate=0.0,
            last_update_time=datetime.now(timezone.utc)
        )

        # Data callbacks
        self.data_callbacks: Dict[str, List[Callable]] = {
            'market_data': [],
            'news_data': [],
            'social_data': [],
            'alternative_data': []
        }

        logger.info(
            "DataIngestionPipeline initialized",
            config=self.ingestion_config)

    async def initialize(self) -> None:
        """Initialize data ingestion pipeline and sources."""
        try:
            self.status = PipelineStatus.STARTING

            # Initialize data sources based on configuration
            if 'market_data' in self.ingestion_config.sources:
                self.market_data_source = MarketDataSource(self.config)
                await self.market_data_source.initialize()
                logger.info("Market data source initialized")

            if 'news_data' in self.ingestion_config.sources:
                self.news_data_source = NewsDataSource(self.config)
                await self.news_data_source.initialize()
                logger.info("News data source initialized")

            if 'social_media' in self.ingestion_config.sources:
                self.social_media_source = SocialMediaDataSource(self.config)
                await self.social_media_source.initialize()
                logger.info("Social media source initialized")

            if 'alternative_data' in self.ingestion_config.sources:
                self.alternative_data_source = AlternativeDataSource(
                    self.config)
                await self.alternative_data_source.initialize()
                logger.info("Alternative data source initialized")

            # Initialize data buffers
            for source in self.ingestion_config.sources:
                self.data_buffers[source] = []

            logger.info("DataIngestionPipeline initialization completed")

        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(
                f"Failed to initialize DataIngestionPipeline: {
                    str(e)}")
            raise DataSourceError(f"Pipeline initialization failed: {str(e)}")

    @time_execution
    async def start(self) -> None:
        """Start the data ingestion pipeline."""
        try:
            if self.status == PipelineStatus.RUNNING:
                logger.warning("Pipeline is already running")
                return

            self.status = PipelineStatus.RUNNING

            # Start ingestion tasks based on mode
            if self.ingestion_config.mode in [
                    IngestionMode.REAL_TIME, IngestionMode.HYBRID]:
                await self._start_real_time_ingestion()

            if self.ingestion_config.mode in [
                    IngestionMode.BATCH, IngestionMode.HYBRID]:
                await self._start_batch_ingestion()

            # Start buffer processing task
            self.active_tasks['buffer_processor'] = asyncio.create_task(
                self._process_buffers()
            )

            # Start metrics collection task
            self.active_tasks['metrics_collector'] = asyncio.create_task(
                self._collect_metrics()
            )

            logger.info("DataIngestionPipeline started successfully")

        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(f"Failed to start DataIngestionPipeline: {str(e)}")
            raise DataSourceError(f"Pipeline start failed: {str(e)}")

    async def _start_real_time_ingestion(self) -> None:
        """Start real-time data ingestion tasks."""
        try:
            # Market data real-time ingestion
            if self.market_data_source:
                for symbol in self.ingestion_config.symbols:
                    task_name = f"market_data_{symbol}"
                    self.active_tasks[task_name] = asyncio.create_task(
                        self._ingest_market_data_real_time(symbol)
                    )
                logger.info("Real-time market data ingestion started")

            # News data real-time ingestion
            if self.news_data_source:
                self.active_tasks['news_data'] = asyncio.create_task(
                    self._ingest_news_data_real_time()
                )
                logger.info("Real-time news data ingestion started")

            # Social media real-time ingestion
            if self.social_media_source:
                self.active_tasks['social_media'] = asyncio.create_task(
                    self._ingest_social_data_real_time()
                )
                logger.info("Real-time social media ingestion started")

        except Exception as e:
            logger.error(f"Failed to start real-time ingestion: {str(e)}")
            raise

    async def _start_batch_ingestion(self) -> None:
        """Start batch data ingestion tasks."""
        try:
            # Alternative data batch ingestion
            if self.alternative_data_source:
                self.active_tasks['alternative_data'] = asyncio.create_task(
                    self._ingest_alternative_data_batch()
                )
                logger.info("Batch alternative data ingestion started")

            # Historical market data batch ingestion
            if self.market_data_source:
                self.active_tasks['historical_market_data'] = asyncio.create_task(
                    self._ingest_historical_market_data())
                logger.info("Batch historical market data ingestion started")

        except Exception as e:
            logger.error(f"Failed to start batch ingestion: {str(e)}")
            raise

    @retry(max_attempts=3, delay=1.0)
    async def _ingest_market_data_real_time(self, symbol: str) -> None:
        """Ingest real-time market data for a symbol."""
        try:
            # Subscribe to ticker updates
            subscription_id = await self.market_data_source.subscribe_to_ticker(
                exchange_name='binance',  # Primary exchange
                symbol=symbol,
                callback=lambda ticker: self._handle_market_data(
                    ticker, symbol)
            )

            logger.info(
                f"Started real-time market data ingestion for {symbol}")

            # Keep the task alive
            while self.status == PipelineStatus.RUNNING:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(
                f"Real-time market data ingestion failed for {symbol}: {str(e)}")
            self.metrics.failed_ingestions += 1
            raise

    async def _ingest_news_data_real_time(self) -> None:
        """Ingest real-time news data."""
        try:
            while self.status == PipelineStatus.RUNNING:
                for symbol in self.ingestion_config.symbols:
                    try:
                        # Get recent news for symbol
                        news_articles = await self.news_data_source.get_news_for_symbol(
                            symbol, hours_back=1
                        )

                        # Add to buffer
                        for article in news_articles:
                            self._add_to_buffer('news_data', article)

                        self.metrics.successful_ingestions += len(
                            news_articles)

                    except Exception as e:
                        logger.warning(
                            f"Failed to ingest news for {symbol}: {
                                str(e)}")
                        self.metrics.failed_ingestions += 1
                        continue

                # Wait before next collection
                # Convert to seconds
                await asyncio.sleep(self.ingestion_config.update_interval * 60)

        except Exception as e:
            logger.error(f"Real-time news data ingestion failed: {str(e)}")
            raise

    async def _ingest_social_data_real_time(self) -> None:
        """Ingest real-time social media data."""
        try:
            while self.status == PipelineStatus.RUNNING:
                for symbol in self.ingestion_config.symbols:
                    try:
                        # Get social sentiment for symbol
                        social_metrics = await self.social_media_source.get_social_sentiment(
                            symbol, hours_back=1
                        )

                        # Add to buffer
                        self._add_to_buffer('social_data', social_metrics)
                        self.metrics.successful_ingestions += 1

                    except Exception as e:
                        logger.warning(
                            f"Failed to ingest social data for {symbol}: {
                                str(e)}")
                        self.metrics.failed_ingestions += 1
                        continue

                # Wait before next collection
                await asyncio.sleep(self.ingestion_config.update_interval * 60)

        except Exception as e:
            logger.error(f"Real-time social data ingestion failed: {str(e)}")
            raise

    async def _ingest_alternative_data_batch(self) -> None:
        """Ingest alternative data in batch mode."""
        try:
            while self.status == PipelineStatus.RUNNING:
                try:
                    # Get economic indicators
                    indicators = ['GDP', 'UNRATE', 'FEDFUNDS']
                    economic_data = await self.alternative_data_source.get_economic_indicators(
                        indicators, days_back=7
                    )

                    # Add to buffer
                    for indicator in economic_data:
                        self._add_to_buffer('alternative_data', indicator)

                    # Get weather data
                    weather_data = await self.alternative_data_source.get_weather_data(
                        ['New York', 'London'], days_back=7
                    )

                    # Add to buffer
                    for weather_point in weather_data:
                        self._add_to_buffer('alternative_data', weather_point)

                    self.metrics.successful_ingestions += len(
                        economic_data) + len(weather_data)

                    # Wait before next batch (1 hour)
                    await asyncio.sleep(3600)

                except Exception as e:
                    logger.warning(
                        f"Failed to ingest alternative data batch: {
                            str(e)}")
                    self.metrics.failed_ingestions += 1
                    await asyncio.sleep(300)  # Wait 5 minutes on error

        except Exception as e:
            logger.error(f"Batch alternative data ingestion failed: {str(e)}")
            raise

    async def _ingest_historical_market_data(self) -> None:
        """Ingest historical market data in batch mode."""
        try:
            # Get historical data for the last 30 days
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30)

            for symbol in self.ingestion_config.symbols:
                try:
                    historical_data = await self.market_data_source.get_historical_data(
                        exchange_name='binance',
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        interval='1h'
                    )

                    # Add to buffer
                    for data_point in historical_data:
                        self._add_to_buffer('market_data', data_point)

                    self.metrics.successful_ingestions += len(historical_data)
                    logger.info(
                        f"Ingested {
                            len(historical_data)} historical data points for {symbol}")

                except Exception as e:
                    logger.warning(
                        f"Failed to ingest historical data for {symbol}: {
                            str(e)}")
                    self.metrics.failed_ingestions += 1
                    continue

        except Exception as e:
            logger.error(f"Historical market data ingestion failed: {str(e)}")
            raise

    def _handle_market_data(self, ticker: Ticker, symbol: str) -> None:
        """Handle incoming market data from real-time streams."""
        try:
            # Convert ticker to MarketData format
            market_data = MarketData(
                symbol=ticker.symbol,
                price=ticker.price,
                volume=ticker.volume,
                timestamp=ticker.timestamp,
                exchange=ticker.exchange,
                bid=getattr(ticker, 'bid', None),
                ask=getattr(ticker, 'ask', None),
                open_price=getattr(ticker, 'open_price', None),
                high_price=getattr(ticker, 'high_price', None),
                low_price=getattr(ticker, 'low_price', None)
            )

            # Add to buffer
            self._add_to_buffer('market_data', market_data)
            self.metrics.successful_ingestions += 1

            # Call registered callbacks
            for callback in self.data_callbacks['market_data']:
                try:
                    callback(market_data)
                except Exception as e:
                    logger.warning(f"Market data callback failed: {str(e)}")

        except Exception as e:
            logger.error(
                f"Failed to handle market data for {symbol}: {
                    str(e)}")
            self.metrics.failed_ingestions += 1

    def _add_to_buffer(self, source: str, data: Any) -> None:
        """Add data to the appropriate buffer."""
        try:
            if source not in self.data_buffers:
                self.data_buffers[source] = []

            self.data_buffers[source].append({
                'data': data,
                'timestamp': datetime.now(timezone.utc),
                'source': source
            })

            # Maintain buffer size
            if len(self.data_buffers[source]
                   ) > self.ingestion_config.buffer_size:
                self.data_buffers[source].pop(0)

        except Exception as e:
            logger.error(f"Failed to add data to buffer {source}: {str(e)}")

    async def _process_buffers(self) -> None:
        """Process data buffers and persist to storage."""
        try:
            while self.status == PipelineStatus.RUNNING:
                for source, buffer in self.data_buffers.items():
                    if len(buffer) >= self.ingestion_config.batch_size:
                        try:
                            # Extract batch
                            batch = buffer[:self.ingestion_config.batch_size]
                            self.data_buffers[source] = buffer[self.ingestion_config.batch_size:]

                            # Process batch
                            await self._process_data_batch(source, batch)

                        except Exception as e:
                            logger.error(
                                f"Failed to process buffer for {source}: {
                                    str(e)}")
                            continue

                # Wait before next processing cycle
                await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Buffer processing failed: {str(e)}")
            raise

    @time_execution
    async def _process_data_batch(
            self, source: str, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of data and store it."""
        try:
            # Import storage manager for database operations
            from .storage import DataStorageManager

            # Initialize storage manager if not already done
            if not hasattr(self, 'storage_manager'):
                self.storage_manager = DataStorageManager(self.config)

            # Process and store each item in the batch
            for item in batch:
                if item.get('type') == 'market_data' and item.get('data'):
                    # Store market data to database
                    await self.storage_manager.store_market_data(item['data'])
                # Add other data type handling as needed

            processed_count = len(batch)
            self.metrics.total_records_processed += processed_count

            logger.debug(
                f"Processed and stored {processed_count} records from {source}")

        except Exception as e:
            logger.error(
                f"Failed to process data batch for {source}: {
                    str(e)}")
            raise

    async def _collect_metrics(self) -> None:
        """Collect and update pipeline metrics."""
        try:
            last_time = datetime.now(timezone.utc)
            last_processed = self.metrics.total_records_processed

            while self.status == PipelineStatus.RUNNING:
                await asyncio.sleep(60)  # Update metrics every minute

                current_time = datetime.now(timezone.utc)
                current_processed = self.metrics.total_records_processed

                # Calculate rates
                time_diff = (current_time - last_time).total_seconds()
                records_diff = current_processed - last_processed

                if time_diff > 0:
                    self.metrics.records_per_second = records_diff / time_diff

                # Calculate buffer utilization
                total_buffer_items = sum(len(buffer)
                                         for buffer in self.data_buffers.values())
                max_buffer_size = len(
                    self.data_buffers) * self.ingestion_config.buffer_size

                if max_buffer_size > 0:
                    self.metrics.buffer_utilization = total_buffer_items / max_buffer_size

                # Calculate error rate
                total_operations = self.metrics.successful_ingestions + \
                    self.metrics.failed_ingestions
                if total_operations > 0:
                    self.metrics.error_rate = self.metrics.failed_ingestions / total_operations

                self.metrics.last_update_time = current_time

                # Update for next iteration
                last_time = current_time
                last_processed = current_processed

        except Exception as e:
            logger.error(f"Metrics collection failed: {str(e)}")

    def register_callback(self, data_type: str,
                          callback: Callable[[Any], None]) -> None:
        """
        Register a callback for specific data type.

        Args:
            data_type: Type of data ('market_data', 'news_data', etc.)
            callback: Callback function to handle data
        """
        if data_type in self.data_callbacks:
            self.data_callbacks[data_type].append(callback)
            logger.info(f"Registered callback for {data_type}")
        else:
            logger.warning(f"Unknown data type for callback: {data_type}")

    async def pause(self) -> None:
        """Pause the data ingestion pipeline."""
        if self.status == PipelineStatus.RUNNING:
            self.status = PipelineStatus.PAUSED
            logger.info("DataIngestionPipeline paused")

    async def resume(self) -> None:
        """Resume the data ingestion pipeline."""
        if self.status == PipelineStatus.PAUSED:
            self.status = PipelineStatus.RUNNING
            logger.info("DataIngestionPipeline resumed")

    async def stop(self) -> None:
        """Stop the data ingestion pipeline."""
        try:
            self.status = PipelineStatus.STOPPING

            # Cancel all active tasks
            for task_name, task in self.active_tasks.items():
                if not task.done():
                    task.cancel()
                    logger.info(f"Cancelled task: {task_name}")

            # Wait for tasks to complete
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

            # Cleanup data sources
            if self.market_data_source:
                await self.market_data_source.cleanup()
            if self.news_data_source:
                await self.news_data_source.cleanup()
            if self.social_media_source:
                await self.social_media_source.cleanup()
            if self.alternative_data_source:
                await self.alternative_data_source.cleanup()

            self.status = PipelineStatus.STOPPED
            logger.info("DataIngestionPipeline stopped successfully")

        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(f"Failed to stop DataIngestionPipeline: {str(e)}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            "status": self.status.value,
            "configuration": {
                "mode": self.ingestion_config.mode.value,
                "sources": self.ingestion_config.sources,
                "symbols": self.ingestion_config.symbols,
                "batch_size": self.ingestion_config.batch_size,
                "update_interval": self.ingestion_config.update_interval},
            "metrics": {
                "total_records_processed": self.metrics.total_records_processed,
                "successful_ingestions": self.metrics.successful_ingestions,
                "failed_ingestions": self.metrics.failed_ingestions,
                "records_per_second": self.metrics.records_per_second,
                "buffer_utilization": self.metrics.buffer_utilization,
                "error_rate": self.metrics.error_rate,
                "last_update_time": self.metrics.last_update_time.isoformat()},
            "buffer_sizes": {
                source: len(buffer) for source,
                buffer in self.data_buffers.items()},
            "active_tasks": list(
                self.active_tasks.keys())}
