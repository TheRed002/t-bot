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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import DataSourceError

# Import from P-001 core components
from src.core.types import IngestionMode, MarketData, PipelineStatus, Ticker
from src.data.sources.alternative_data import AlternativeDataSource

# Import data sources
from src.data.sources.market_data import MarketDataSource
from src.data.sources.news_data import NewsDataSource
from src.data.sources.social_media import SocialMediaDataSource
from src.error_handling.connection_manager import ConnectionManager

# Import from P-002 database components
# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.pattern_analytics import ErrorPatternAnalytics
from src.error_handling.recovery_scenarios import (
    DataFeedInterruptionRecovery,
    NetworkDisconnectionRecovery,
)

# Import from P-007A utilities
from src.utils.decorators import retry

# IngestionMode and PipelineStatus are now imported from core.types


@dataclass
class IngestionConfig:
    """Data ingestion configuration"""

    mode: IngestionMode
    sources: list[str]
    symbols: list[str]
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


class DataIngestionPipeline(BaseComponent):
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
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.recovery_scenario = DataFeedInterruptionRecovery(config)
        self.network_recovery = NetworkDisconnectionRecovery(config)
        self.connection_manager = ConnectionManager(config)
        self.pattern_analytics = ErrorPatternAnalytics(config)

        # Pipeline configuration
        pipeline_config = getattr(config, "data_pipeline", {})
        if hasattr(pipeline_config, "get"):
            self.ingestion_config = IngestionConfig(
                mode=IngestionMode(pipeline_config.get("mode", "real_time")),
                sources=pipeline_config.get("sources", ["market_data"]),
                symbols=pipeline_config.get("symbols", ["BTCUSDT"]),
                batch_size=pipeline_config.get("batch_size", 100),
                update_interval=pipeline_config.get("update_interval", 1),
                buffer_size=pipeline_config.get("buffer_size", 1000),
                error_threshold=pipeline_config.get("error_threshold", 10),
                retry_attempts=pipeline_config.get("retry_attempts", 3),
            )
        else:
            self.ingestion_config = IngestionConfig(
                mode=IngestionMode("real_time"),
                sources=["market_data"],
                symbols=["BTCUSDT"],
                batch_size=100,
                update_interval=1,
                buffer_size=1000,
                error_threshold=10,
                retry_attempts=3,
            )

        # Initialize data sources
        self.market_data_source = None
        self.news_data_source = None
        self.social_media_source = None
        self.alternative_data_source = None

        # Pipeline state
        self.status = PipelineStatus.STOPPED
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.data_buffers: dict[str, list[dict[str, Any]]] = {}
        self.data_callbacks: dict[str, list[Callable]] = {
            "market_data": [],
            "news_data": [],
            "social_media": [],
            "alternative_data": [],
        }

        # Metrics tracking
        self.metrics = IngestionMetrics(
            total_records_processed=0,
            successful_ingestions=0,
            failed_ingestions=0,
            avg_processing_time=0.0,
            records_per_second=0.0,
            buffer_utilization=0.0,
            error_rate=0.0,
            last_update_time=datetime.now(timezone.utc),
        )

        # Error tracking
        self.error_count = 0
        self.last_error_time = None

        self.logger.info("DataIngestionPipeline initialized", config=config)

    async def initialize(self) -> None:
        """Initialize data sources and connections."""
        try:
            # Initialize market data source
            if "market_data" in self.ingestion_config.sources:
                self.market_data_source = MarketDataSource(self.config)
                await self.market_data_source.initialize()

                # Establish connection with connection manager
                await self.connection_manager.establish_connection(
                    connection_id="market_data_source",
                    connection_type="data_source",
                    endpoint="market_data",
                    connection_options={
                        "retry_attempts": self.ingestion_config.retry_attempts,
                        "timeout": 30,
                    },
                )

            # Initialize news data source
            if "news_data" in self.ingestion_config.sources:
                self.news_data_source = NewsDataSource(self.config)
                await self.news_data_source.initialize()

                # Establish connection with connection manager
                await self.connection_manager.establish_connection(
                    connection_id="news_data_source",
                    connection_type="data_source",
                    endpoint="news_api",
                    connection_options={
                        "retry_attempts": self.ingestion_config.retry_attempts,
                        "timeout": 30,
                    },
                )

            # Initialize social media source
            if "social_media" in self.ingestion_config.sources:
                self.social_media_source = SocialMediaDataSource(self.config)
                await self.social_media_source.initialize()

                # Establish connection with connection manager
                await self.connection_manager.establish_connection(
                    connection_id="social_media_source",
                    connection_type="data_source",
                    endpoint="social_media",
                    connection_options={
                        "retry_attempts": self.ingestion_config.retry_attempts,
                        "timeout": 30,
                    },
                )

            # Initialize alternative data source
            if "alternative_data" in self.ingestion_config.sources:
                self.alternative_data_source = AlternativeDataSource(self.config)
                await self.alternative_data_source.initialize()

                # Establish connection with connection manager
                await self.connection_manager.establish_connection(
                    connection_id="alternative_data_source",
                    connection_type="data_source",
                    endpoint="alternative_data",
                    connection_options={
                        "retry_attempts": self.ingestion_config.retry_attempts,
                        "timeout": 30,
                    },
                )

            self.logger.info("DataIngestionPipeline initialized successfully")

        except Exception as e:
            # Use ErrorHandler for initialization errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="initialize",
                details={"stage": "initialization", "sources": self.ingestion_config.sources},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to initialize DataIngestionPipeline: {e!s}")
            raise DataSourceError(f"Pipeline initialization failed: {e!s}")

    async def start(self) -> None:
        """Start the data ingestion pipeline."""
        try:
            if self.status == PipelineStatus.RUNNING:
                self.logger.warning("Pipeline is already running")
                return

            # Check connection health before starting
            connection_status = await self.connection_manager.get_all_connection_status()
            unhealthy_connections = [
                conn_id
                for conn_id, status in connection_status.items()
                if not status.get("is_healthy", False)
            ]

            if unhealthy_connections:
                self.logger.warning(f"Unhealthy connections detected: {unhealthy_connections}")
                # Attempt to reconnect unhealthy connections
                for conn_id in unhealthy_connections:
                    await self.connection_manager.reconnect_connection(conn_id)

            self.status = PipelineStatus.RUNNING

            # Start ingestion tasks based on mode
            if self.ingestion_config.mode in [IngestionMode.REAL_TIME, IngestionMode.HYBRID]:
                await self._start_real_time_ingestion()

            if self.ingestion_config.mode in [IngestionMode.BATCH, IngestionMode.HYBRID]:
                await self._start_batch_ingestion()

            # Start buffer processing task
            self.active_tasks["buffer_processor"] = asyncio.create_task(self._process_buffers())

            # Start metrics collection task
            self.active_tasks["metrics_collector"] = asyncio.create_task(self._collect_metrics())

            self.logger.info("DataIngestionPipeline started successfully")

        except Exception as e:
            # Use ErrorHandler for start errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="start",
                details={"stage": "startup", "mode": self.ingestion_config.mode.value},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.status = PipelineStatus.ERROR
            self.logger.error(f"Failed to start DataIngestionPipeline: {e!s}")
            raise DataSourceError(f"Pipeline start failed: {e!s}")

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
                self.logger.info("Real-time market data ingestion started")

            # News data real-time ingestion
            if self.news_data_source:
                self.active_tasks["news_data"] = asyncio.create_task(
                    self._ingest_news_data_real_time()
                )
                self.logger.info("Real-time news data ingestion started")

            # Social media real-time ingestion
            if self.social_media_source:
                self.active_tasks["social_media"] = asyncio.create_task(
                    self._ingest_social_data_real_time()
                )
                self.logger.info("Real-time social media ingestion started")

        except Exception as e:
            # Use ErrorHandler for real-time ingestion start errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="start_real_time_ingestion",
                details={
                    "stage": "real_time_startup",
                    "sources": [
                        s
                        for s in ["market_data", "news_data", "social_media"]
                        if getattr(self, f"{s}_source")
                    ],
                },
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to start real-time ingestion: {e!s}")
            raise

    async def _start_batch_ingestion(self) -> None:
        """Start batch data ingestion tasks."""
        try:
            # Alternative data batch ingestion
            if self.alternative_data_source:
                self.active_tasks["alternative_data"] = asyncio.create_task(
                    self._ingest_alternative_data_batch()
                )
                self.logger.info("Batch alternative data ingestion started")

            # Historical market data batch ingestion
            if self.market_data_source:
                self.active_tasks["historical_market_data"] = asyncio.create_task(
                    self._ingest_historical_market_data()
                )
                self.logger.info("Batch historical market data ingestion started")

        except Exception as e:
            # Use ErrorHandler for batch ingestion start errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="start_batch_ingestion",
                details={
                    "stage": "batch_startup",
                    "sources": [
                        s
                        for s in ["alternative_data", "market_data"]
                        if getattr(self, f"{s}_source")
                    ],
                },
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to start batch ingestion: {e!s}")
            raise

    @retry(max_attempts=3, base_delay=1.0)
    async def _ingest_market_data_real_time(self, symbol: str) -> None:
        """Ingest real-time market data for a symbol."""
        try:
            # Subscribe to ticker updates
            await self.market_data_source.subscribe_to_ticker(
                exchange_name="binance",  # Primary exchange
                symbol=symbol,
                callback=lambda ticker: self._handle_market_data(ticker, symbol),
            )

            self.logger.info(f"Started real-time market data ingestion for {symbol}")

            # Keep the task alive
            while self.status == PipelineStatus.RUNNING:
                await asyncio.sleep(1)

        except Exception as e:
            # Use ErrorHandler for market data ingestion errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="ingest_market_data_real_time",
                symbol=symbol,
                details={"data_type": "market_data", "mode": "real_time", "exchange": "binance"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Real-time market data ingestion failed for {symbol}: {e!s}")
            self.metrics.failed_ingestions += 1
            raise

    async def _ingest_news_data_real_time(self) -> None:
        """Ingest real-time news data."""
        try:
            while self.status == PipelineStatus.RUNNING:
                try:
                    # Get recent news for tracked symbols
                    for symbol in self.ingestion_config.symbols:
                        news_articles = await self.news_data_source.get_news_for_symbol(
                            symbol=symbol, hours_back=1, max_articles=10
                        )

                        for article in news_articles:
                            self._add_to_buffer("news_data", article)

                        self.metrics.successful_ingestions += len(news_articles)

                    # Convert to minutes
                    await asyncio.sleep(self.ingestion_config.update_interval * 60)

                except Exception as e:
                    # Use ErrorHandler for news ingestion errors
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="DataIngestionPipeline",
                        operation="ingest_news_data_real_time",
                        details={"data_type": "news_data", "mode": "real_time"},
                    )

                    await self.error_handler.handle_error(e, error_context)
                    self.pattern_analytics.add_error_event(error_context.__dict__)

                    self.logger.warning(f"News data ingestion failed: {e!s}")
                    self.metrics.failed_ingestions += 1
                    await asyncio.sleep(60)  # Wait before retry

        except Exception as e:
            # Use ErrorHandler for fatal news ingestion errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="ingest_news_data_real_time",
                details={"data_type": "news_data", "mode": "real_time", "severity": "fatal"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Fatal error in news data ingestion: {e!s}")

    async def _ingest_social_data_real_time(self) -> None:
        """Ingest real-time social media data."""
        try:
            while self.status == PipelineStatus.RUNNING:
                try:
                    # Get social sentiment for tracked symbols
                    for symbol in self.ingestion_config.symbols:
                        social_metrics = await self.social_media_source.get_social_sentiment(
                            symbol=symbol, hours_back=1
                        )

                        if social_metrics:
                            self._add_to_buffer("social_media", social_metrics)
                            self.metrics.successful_ingestions += 1

                    # Convert to minutes
                    await asyncio.sleep(self.ingestion_config.update_interval * 60)

                except Exception as e:
                    # Use ErrorHandler for social data ingestion errors
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="DataIngestionPipeline",
                        operation="ingest_social_data_real_time",
                        details={"data_type": "social_media", "mode": "real_time"},
                    )

                    await self.error_handler.handle_error(e, error_context)
                    self.pattern_analytics.add_error_event(error_context.__dict__)

                    self.logger.warning(f"Social media data ingestion failed: {e!s}")
                    self.metrics.failed_ingestions += 1
                    await asyncio.sleep(60)  # Wait before retry

        except Exception as e:
            # Use ErrorHandler for fatal social data ingestion errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="ingest_social_data_real_time",
                details={"data_type": "social_media", "mode": "real_time", "severity": "fatal"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Fatal error in social media data ingestion: {e!s}")

    async def _ingest_alternative_data_batch(self) -> None:
        """Ingest alternative data in batch mode."""
        try:
            while self.status == PipelineStatus.RUNNING:
                try:
                    # Get economic indicators
                    economic_data = await self.alternative_data_source.get_economic_indicators(
                        indicators=["gdp", "inflation", "unemployment"], days_back=7
                    )

                    for data_point in economic_data:
                        self._add_to_buffer("alternative_data", data_point)

                    self.metrics.successful_ingestions += len(economic_data)
                    self.logger.info(f"Ingested {len(economic_data)} alternative data points")

                    # Run once per day
                    await asyncio.sleep(24 * 60 * 60)  # 24 hours

                except Exception as e:
                    # Use ErrorHandler for alternative data ingestion errors
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="DataIngestionPipeline",
                        operation="ingest_alternative_data_batch",
                        details={"data_type": "alternative_data", "mode": "batch"},
                    )

                    await self.error_handler.handle_error(e, error_context)
                    self.pattern_analytics.add_error_event(error_context.__dict__)

                    self.logger.warning(f"Alternative data ingestion failed: {e!s}")
                    self.metrics.failed_ingestions += 1
                    await asyncio.sleep(60 * 60)  # Wait 1 hour before retry

        except Exception as e:
            # Use ErrorHandler for fatal alternative data ingestion errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="ingest_alternative_data_batch",
                details={"data_type": "alternative_data", "mode": "batch", "severity": "fatal"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Fatal error in alternative data ingestion: {e!s}")

    async def _ingest_historical_market_data(self) -> None:
        """Ingest historical market data in batch mode."""
        try:
            # Get historical data for the last 30 days
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30)

            for symbol in self.ingestion_config.symbols:
                try:
                    historical_data = await self.market_data_source.get_historical_data(
                        exchange_name="binance",
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        interval="1h",
                    )

                    # Add to buffer
                    for data_point in historical_data:
                        self._add_to_buffer("market_data", data_point)

                    self.metrics.successful_ingestions += len(historical_data)
                    self.logger.info(
                        f"Ingested {len(historical_data)} historical data points for {symbol}"
                    )

                except Exception as e:
                    # Use ErrorHandler for historical data ingestion errors
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="DataIngestionPipeline",
                        operation="ingest_historical_market_data",
                        symbol=symbol,
                        details={
                            "data_type": "market_data",
                            "mode": "batch",
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                        },
                    )

                    await self.error_handler.handle_error(e, error_context)
                    self.pattern_analytics.add_error_event(error_context.__dict__)

                    self.logger.warning(f"Failed to ingest historical data for {symbol}: {e!s}")
                    self.metrics.failed_ingestions += 1
                    continue

        except Exception as e:
            # Use ErrorHandler for fatal historical data ingestion errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="ingest_historical_market_data",
                details={"data_type": "market_data", "mode": "batch", "severity": "fatal"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Historical market data ingestion failed: {e!s}")
            raise

    async def _handle_market_data(self, ticker: Ticker, symbol: str) -> None:
        """Handle incoming market data from real-time streams."""
        try:
            # Convert ticker to MarketData format mapping to core Ticker fields
            market_data = MarketData(
                symbol=ticker.symbol,
                price=ticker.last_price,
                volume=ticker.volume_24h,
                timestamp=ticker.timestamp,
                bid=ticker.bid,
                ask=ticker.ask,
                open_price=None,
                high_price=None,
                low_price=None,
            )

            # Add to buffer
            await self._add_to_buffer_async("market_data", market_data)
            self.metrics.successful_ingestions += 1

            # Call registered callbacks asynchronously
            for callback in self.data_callbacks["market_data"]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(market_data)
                    else:
                        callback(market_data)
                except Exception as e:
                    # Use ErrorHandler for callback errors
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="DataIngestionPipeline",
                        operation="handle_market_data_callback",
                        symbol=symbol,
                        details={"data_type": "market_data", "callback_type": "market_data"},
                    )

                    await self.error_handler.handle_error(e, error_context)
                    self.pattern_analytics.add_error_event(error_context.__dict__)

                    self.logger.warning(f"Market data callback failed: {e!s}")

        except Exception as e:
            # Use ErrorHandler for market data handling errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="handle_market_data",
                symbol=symbol,
                details={"data_type": "market_data", "stage": "data_handling"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to handle market data for {symbol}: {e!s}")
            self.metrics.failed_ingestions += 1

    def _add_to_buffer(self, source: str, data: Any) -> None:
        """Add data to the appropriate buffer."""
        try:
            if source not in self.data_buffers:
                self.data_buffers[source] = []

            self.data_buffers[source].append(
                {
                    "data": data,
                    "timestamp": datetime.now(timezone.utc),
                    "source": source,
                    "type": source,
                }
            )

            # Maintain buffer size
            if len(self.data_buffers[source]) > self.ingestion_config.buffer_size:
                self.data_buffers[source].pop(0)

        except Exception as e:
            # Use ErrorHandler for buffer operation errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="add_to_buffer",
                details={
                    "source": source,
                    "buffer_size": len(self.data_buffers.get(source, [])),
                    "max_buffer_size": self.ingestion_config.buffer_size,
                },
            )

            # Note: handle_error is async but this function is sync
            # TODO: Consider making this function async or using asyncio.create_task()
            # self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to add data to buffer {source}: {e!s}")

    async def _add_to_buffer_async(self, source: str, data: Any) -> None:
        """Add data to the appropriate buffer asynchronously."""
        try:
            if source not in self.data_buffers:
                self.data_buffers[source] = []

            self.data_buffers[source].append(
                {
                    "data": data,
                    "timestamp": datetime.now(timezone.utc),
                    "source": source,
                    "type": source,
                }
            )

            # Maintain buffer size
            if len(self.data_buffers[source]) > self.ingestion_config.buffer_size:
                self.data_buffers[source].pop(0)

        except Exception as e:
            # Use ErrorHandler for buffer operation errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="add_to_buffer_async",
                details={
                    "source": source,
                    "buffer_size": len(self.data_buffers.get(source, [])),
                    "max_buffer_size": self.ingestion_config.buffer_size,
                },
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to add data to buffer {source}: {e!s}")

    async def _process_buffers(self) -> None:
        """Process data buffers and persist to storage."""
        try:
            while self.status == PipelineStatus.RUNNING:
                for source, buffer in self.data_buffers.items():
                    if len(buffer) >= self.ingestion_config.batch_size:
                        try:
                            # Extract batch
                            batch = buffer[: self.ingestion_config.batch_size]
                            self.data_buffers[source] = buffer[self.ingestion_config.batch_size :]

                            # Process batch (placeholder for storage integration)
                            self.logger.debug(
                                f"Processing batch of {len(batch)} items from {source}"
                            )

                            # Update metrics
                            self.metrics.total_records_processed += len(batch)

                        except Exception as e:
                            # Use ErrorHandler for buffer processing errors
                            error_context = self.error_handler.create_error_context(
                                error=e,
                                component="DataIngestionPipeline",
                                operation="process_buffers",
                                details={
                                    "source": source,
                                    "batch_size": len(batch),
                                    "stage": "batch_processing",
                                },
                            )

                            await self.error_handler.handle_error(e, error_context)
                            self.pattern_analytics.add_error_event(error_context.__dict__)

                            self.logger.error(
                                f"Failed to process buffer batch from {source}: {e!s}"
                            )

                await asyncio.sleep(1)  # Check buffers every second

        except Exception as e:
            # Use ErrorHandler for fatal buffer processing errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="process_buffers",
                details={"stage": "buffer_processing", "severity": "fatal"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Fatal error in buffer processing: {e!s}")

    async def _collect_metrics(self) -> None:
        """Collect and update pipeline metrics."""
        try:
            while self.status == PipelineStatus.RUNNING:
                try:
                    # Calculate buffer utilization
                    total_buffer_items = sum(len(buffer) for buffer in self.data_buffers.values())
                    self.metrics.buffer_utilization = total_buffer_items / (
                        len(self.data_buffers) * self.ingestion_config.buffer_size
                    )

                    # Calculate error rate
                    total_operations = (
                        self.metrics.successful_ingestions + self.metrics.failed_ingestions
                    )
                    if total_operations > 0:
                        self.metrics.error_rate = self.metrics.failed_ingestions / total_operations

                    # Calculate records per second (rolling average)
                    current_time = datetime.now(timezone.utc)
                    time_diff = (current_time - self.metrics.last_update_time).total_seconds()
                    if time_diff > 0:
                        new_records = self.metrics.total_records_processed - getattr(
                            self, "_last_total_records", 0
                        )
                        self.metrics.records_per_second = new_records / time_diff
                        self._last_total_records = self.metrics.total_records_processed

                    self.metrics.last_update_time = current_time

                    await asyncio.sleep(60)  # Update metrics every minute

                except Exception as e:
                    # Use ErrorHandler for metrics collection errors
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="DataIngestionPipeline",
                        operation="collect_metrics",
                        details={"stage": "metrics_collection"},
                    )

                    await self.error_handler.handle_error(e, error_context)
                    self.pattern_analytics.add_error_event(error_context.__dict__)

                    self.logger.warning(f"Metrics collection failed: {e!s}")
                    await asyncio.sleep(60)  # Wait before retry

        except Exception as e:
            # Use ErrorHandler for fatal metrics collection errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="collect_metrics",
                details={"stage": "metrics_collection", "severity": "fatal"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Fatal error in metrics collection: {e!s}")

    async def pause(self) -> None:
        """Pause the data ingestion pipeline."""
        try:
            if self.status == PipelineStatus.RUNNING:
                self.status = PipelineStatus.PAUSED
                self.logger.info("DataIngestionPipeline paused")
            else:
                self.logger.warning("Pipeline is not running, cannot pause")

        except Exception as e:
            # Use ErrorHandler for pause errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="pause",
                details={"current_status": self.status.value},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to pause pipeline: {e!s}")

    async def resume(self) -> None:
        """Resume the data ingestion pipeline."""
        try:
            if self.status == PipelineStatus.PAUSED:
                self.status = PipelineStatus.RUNNING
                self.logger.info("DataIngestionPipeline resumed")
            else:
                self.logger.warning("Pipeline is not paused, cannot resume")

        except Exception as e:
            # Use ErrorHandler for resume errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="resume",
                details={"current_status": self.status.value},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to resume pipeline: {e!s}")

    async def stop(self) -> None:
        """Stop the data ingestion pipeline."""
        try:
            if self.status == PipelineStatus.RUNNING:
                self.status = PipelineStatus.STOPPED

                # Cancel all active tasks
                for _task_name, task in self.active_tasks.items():
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                # Close connections
                await self.connection_manager.close_connection("market_data_source")
                await self.connection_manager.close_connection("news_data_source")
                await self.connection_manager.close_connection("social_media_source")
                await self.connection_manager.close_connection("alternative_data_source")

                # Cleanup data sources
                if self.market_data_source:
                    await self.market_data_source.cleanup()
                if self.news_data_source:
                    await self.news_data_source.cleanup()
                if self.social_media_source:
                    await self.social_media_source.cleanup()
                if self.alternative_data_source:
                    await self.alternative_data_source.cleanup()

                self.logger.info("DataIngestionPipeline stopped successfully")

            else:
                self.logger.warning("Pipeline is not running, cannot stop")

        except Exception as e:
            # Use ErrorHandler for stop errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="stop",
                details={
                    "current_status": self.status.value,
                    "active_tasks": list(self.active_tasks.keys()),
                },
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to stop pipeline: {e!s}")

    def register_callback(self, data_type: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for data updates."""
        try:
            if data_type in self.data_callbacks:
                self.data_callbacks[data_type].append(callback)
                self.logger.info(f"Registered callback for {data_type}")
            else:
                self.logger.warning(f"Unknown data type: {data_type}")

        except Exception as e:
            # Use ErrorHandler for callback registration errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="register_callback",
                details={"data_type": data_type, "callback_type": type(callback).__name__},
            )

            # Note: handle_error is async but this function is sync
            # TODO: Consider making this function async or using asyncio.create_task()
            # self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to register callback for {data_type}: {e!s}")

    def get_status(self) -> dict[str, Any]:
        """Get pipeline status and metrics."""
        try:
            # Get connection status
            connection_status = self.connection_manager.get_all_connection_status()

            # Get error pattern summary
            pattern_summary = self.pattern_analytics.get_pattern_summary()
            correlation_summary = self.pattern_analytics.get_correlation_summary()
            trend_summary = self.pattern_analytics.get_trend_summary()

            # Get circuit breaker status
            circuit_breaker_status = self.error_handler.get_circuit_breaker_status()

            return {
                "status": self.status.value,
                "mode": self.ingestion_config.mode.value,
                "sources": self.ingestion_config.sources,
                "symbols": self.ingestion_config.symbols,
                "active_tasks": list(self.active_tasks.keys()),
                "buffer_sizes": {
                    source: len(buffer) for source, buffer in self.data_buffers.items()
                },
                "metrics": {
                    "total_records_processed": self.metrics.total_records_processed,
                    "successful_ingestions": self.metrics.successful_ingestions,
                    "failed_ingestions": self.metrics.failed_ingestions,
                    "avg_processing_time": self.metrics.avg_processing_time,
                    "records_per_second": self.metrics.records_per_second,
                    "buffer_utilization": self.metrics.buffer_utilization,
                    "error_rate": self.metrics.error_rate,
                    "last_update_time": self.metrics.last_update_time.isoformat(),
                },
                "connection_status": connection_status,
                "error_patterns": pattern_summary,
                "error_correlations": correlation_summary,
                "error_trends": trend_summary,
                "circuit_breaker_status": circuit_breaker_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            # Use ErrorHandler for status retrieval errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="get_status",
                details={"operation": "status_retrieval"},
            )

            # Note: handle_error is async but this function is sync
            # TODO: Consider making this function async or using asyncio.create_task()
            # self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to get pipeline status: {e!s}")
            return {
                "error": str(e),
                "error_id": error_context.error_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        try:
            # Stop pipeline if running
            if self.status == PipelineStatus.RUNNING:
                await self.stop()

            # Clear buffers
            self.data_buffers.clear()

            # Clear callbacks
            for data_type in self.data_callbacks:
                self.data_callbacks[data_type].clear()

            # Clear active tasks
            self.active_tasks.clear()

            self.logger.info("DataIngestionPipeline cleanup completed")

        except Exception as e:
            # Use ErrorHandler for cleanup errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="DataIngestionPipeline",
                operation="cleanup",
                details={"operation": "cleanup"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Error during pipeline cleanup: {e!s}")
