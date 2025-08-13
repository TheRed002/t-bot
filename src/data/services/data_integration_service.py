"""
Data Integration Service

This module provides comprehensive data integration services that properly use
the database module for all data operations, including market data, features,
quality metrics, and pipeline tracking.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models, queries, and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from src.core.config import Config
from src.core.exceptions import DataError, DataSourceError
from src.core.logging import get_logger
from src.core.types import MarketData, StorageMode

# Import from P-002 database components
from src.database.connection import get_async_session
from src.database.queries import DatabaseQueries
from src.database.models import (
    MarketDataRecord,
    FeatureRecord,
    DataQualityRecord,
    DataPipelineRecord,
)
from src.database.influxdb_client import InfluxDBClientWrapper

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution, retry, cache_result

logger = get_logger(__name__)


class DataIntegrationService:
    """
    Comprehensive data integration service that properly uses the database module.

    This service handles:
    - Market data storage and retrieval
    - Feature calculation and storage
    - Data quality monitoring
    - Pipeline execution tracking
    - Hybrid storage (InfluxDB + PostgreSQL)
    """

    def __init__(self, config: Config):
        """Initialize the data integration service."""
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Storage configuration
        storage_config = getattr(config, "data_storage", {})
        if hasattr(storage_config, "get"):
            self.storage_mode = StorageMode(
                storage_config.get("mode", "batch")
            )
            self.batch_size = storage_config.get("batch_size", 100)
            self.cleanup_interval = storage_config.get("cleanup_interval", 3600)
        else:
            self.storage_mode = StorageMode.BATCH
            self.batch_size = 100
            self.cleanup_interval = 3600

        # Initialize InfluxDB client for time series data
        influx_config = getattr(config, "influxdb", {})
        if hasattr(influx_config, "get"):
            self.influx_client = InfluxDBClientWrapper(
                url=influx_config.get("url", "http://localhost:8086"),
                token=influx_config.get("token", ""),
                org=influx_config.get("org", "trading-bot"),
                bucket=influx_config.get("bucket", "market-data"),
            )
            try:
                self.influx_client.connect()
                logger.info("InfluxDB client initialized successfully")
            except Exception as e:
                logger.warning(f"InfluxDB connection failed: {e}")
                self.influx_client = None
        else:
            self.influx_client = None

    @time_execution
    async def store_market_data(
        self,
        market_data: Union[MarketData, List[MarketData]],
        exchange: str = "unknown"
    ) -> bool:
        """
        Store market data using proper database integration.

        Args:
            market_data: Single MarketData object or list of MarketData objects
            exchange: Exchange name for the data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if isinstance(market_data, list):
                return await self._store_market_data_batch(market_data, exchange)
            else:
                return await self._store_single_market_data(market_data, exchange)

        except Exception as e:
            logger.error(f"Market data storage failed: {e}")
            return False

    async def _store_single_market_data(self, data: MarketData, exchange: str) -> bool:
        """Store a single market data record."""
        try:
            # Store to InfluxDB for time series analysis
            if self.influx_client:
                fields = {
                    "price": float(data.price) if data.price else 0.0,
                    "volume": float(data.volume) if data.volume else 0.0,
                }
                if data.bid is not None:
                    fields["bid"] = float(data.bid)
                if data.ask is not None:
                    fields["ask"] = float(data.ask)
                if data.high_price is not None:
                    fields["high"] = float(data.high_price)
                if data.low_price is not None:
                    fields["low"] = float(data.low_price)
                if data.open_price is not None:
                    fields["open"] = float(data.open_price)

                self.influx_client.write_market_data(
                    symbol=data.symbol, data=fields, timestamp=data.timestamp
                )

            # Store to PostgreSQL for persistent storage
            if self.storage_mode in [StorageMode.BATCH, StorageMode.BUFFER]:
                await self._store_market_data_to_postgresql([data], exchange)

            logger.info(f"Stored market data for {data.symbol}")
            return True

        except Exception as e:
            logger.error(f"Single market data storage failed: {e}")
            return False

    async def _store_market_data_batch(
        self,
        data_list: List[MarketData],
        exchange: str
    ) -> bool:
        """Store a batch of market data records."""
        try:
            # Store to InfluxDB for time series analysis
            if self.influx_client:
                await self.influx_client.write_market_data_batch(data_list)

            # Store to PostgreSQL for persistent storage
            if self.storage_mode in [StorageMode.BATCH, StorageMode.BUFFER]:
                await self._store_market_data_to_postgresql(data_list, exchange)

            logger.info(f"Stored batch of {len(data_list)} market data records")
            return True

        except Exception as e:
            logger.error(f"Batch market data storage failed: {e}")
            return False

    async def _store_market_data_to_postgresql(
        self,
        data_list: List[MarketData],
        exchange: str
    ) -> bool:
        """Store market data to PostgreSQL using proper database models."""
        try:
            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                # Convert MarketData to MarketDataRecord models
                records = []
                for data in data_list:
                    record = MarketDataRecord(
                        symbol=data.symbol,
                        exchange=exchange,
                        timestamp=data.timestamp or datetime.now(timezone.utc),
                        open_price=float(data.open_price) if data.open_price else None,
                        high_price=float(data.high_price) if data.high_price else None,
                        low_price=float(data.low_price) if data.low_price else None,
                        close_price=float(data.price) if data.price else None,
                        price=float(data.price) if data.price else None,
                        volume=float(data.volume) if data.volume else None,
                        bid=float(data.bid) if data.bid else None,
                        ask=float(data.ask) if data.ask else None,
                        data_source='exchange',
                        quality_score=1.0,  # Default quality score
                        validation_status='valid'
                    )
                    records.append(record)

                # Bulk create records
                if records:
                    await db_queries.bulk_create_market_data_records(records)
                    logger.info(f"Stored {len(records)} records to PostgreSQL")

                return True

        except Exception as e:
            logger.error(f"PostgreSQL storage failed: {e}")
            return False

    @time_execution
    async def store_feature(
        self,
        symbol: str,
        feature_type: str,
        feature_name: str,
        feature_value: float,
        calculation_timestamp: Optional[datetime] = None,
        confidence_score: Optional[float] = None,
        lookback_period: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        source_data_start: Optional[datetime] = None,
        source_data_end: Optional[datetime] = None
    ) -> bool:
        """
        Store a calculated feature using proper database integration.

        Args:
            symbol: Trading symbol
            feature_type: Type of feature (e.g., 'technical', 'statistical')
            feature_name: Name of the feature (e.g., 'sma_20', 'rsi_14')
            feature_value: Calculated feature value
            calculation_timestamp: When the feature was calculated
            confidence_score: Confidence in the feature value
            lookback_period: Period used for calculation
            parameters: Parameters used in calculation
            source_data_start: Start of source data period
            source_data_end: End of source data period

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                feature_record = FeatureRecord(
                    symbol=symbol,
                    feature_type=feature_type,
                    feature_name=feature_name,
                    calculation_timestamp=calculation_timestamp or datetime.now(timezone.utc),
                    feature_value=feature_value,
                    confidence_score=confidence_score,
                    lookback_period=lookback_period,
                    parameters=parameters,
                    calculation_method='standard',
                    source_data_start=source_data_start,
                    source_data_end=source_data_end
                )

                await db_queries.create_feature_record(feature_record)
                logger.info(f"Stored feature {feature_name} for {symbol}")
                return True

        except Exception as e:
            logger.error(f"Feature storage failed: {e}")
            return False

    @time_execution
    async def store_data_quality_metrics(
        self,
        symbol: str,
        data_source: str,
        completeness_score: float,
        accuracy_score: float,
        consistency_score: float,
        timeliness_score: float,
        overall_score: float,
        missing_data_count: int = 0,
        outlier_count: int = 0,
        duplicate_count: int = 0,
        validation_errors: Optional[List[str]] = None,
        check_type: str = "comprehensive",
        data_period_start: Optional[datetime] = None,
        data_period_end: Optional[datetime] = None
    ) -> bool:
        """
        Store data quality metrics using proper database integration.

        Args:
            symbol: Trading symbol
            data_source: Source of the data
            completeness_score: Data completeness score (0-1)
            accuracy_score: Data accuracy score (0-1)
            consistency_score: Data consistency score (0-1)
            timeliness_score: Data timeliness score (0-1)
            overall_score: Overall quality score (0-1)
            missing_data_count: Number of missing data points
            outlier_count: Number of outliers detected
            duplicate_count: Number of duplicates found
            validation_errors: List of validation errors
            check_type: Type of quality check
            data_period_start: Start of data period
            data_period_end: End of data period

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                quality_record = DataQualityRecord(
                    symbol=symbol,
                    data_source=data_source,
                    quality_check_timestamp=datetime.now(timezone.utc),
                    completeness_score=completeness_score,
                    accuracy_score=accuracy_score,
                    consistency_score=consistency_score,
                    timeliness_score=timeliness_score,
                    overall_score=overall_score,
                    missing_data_count=missing_data_count,
                    outlier_count=outlier_count,
                    duplicate_count=duplicate_count,
                    validation_errors=validation_errors,
                    check_type=check_type,
                    data_period_start=data_period_start,
                    data_period_end=data_period_end
                )

                await db_queries.create_data_quality_record(quality_record)
                logger.info(f"Stored quality metrics for {symbol} from {data_source}")
                return True

        except Exception as e:
            logger.error(f"Quality metrics storage failed: {e}")
            return False

    @time_execution
    async def track_pipeline_execution(
        self,
        pipeline_name: str,
        execution_id: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Start tracking a data pipeline execution.

        Args:
            pipeline_name: Name of the pipeline
            execution_id: Optional execution ID (auto-generated if not provided)
            configuration: Pipeline configuration
            dependencies: List of dependencies

        Returns:
            str: Execution ID for tracking
        """
        try:
            if not execution_id:
                execution_id = str(uuid.uuid4())

            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                pipeline_record = DataPipelineRecord(
                    pipeline_name=pipeline_name,
                    execution_id=execution_id,
                    execution_timestamp=datetime.now(timezone.utc),
                    status='running',
                    stage='started',
                    records_processed=0,
                    records_successful=0,
                    records_failed=0,
                    error_count=0,
                    configuration=configuration,
                    dependencies=dependencies
                )

                await db_queries.create_data_pipeline_record(pipeline_record)
                logger.info(f"Started tracking pipeline {pipeline_name} with ID {execution_id}")
                return execution_id

        except Exception as e:
            logger.error(f"Pipeline tracking start failed: {e}")
            return None

    async def update_pipeline_status(
        self,
        execution_id: str,
        status: str,
        stage: Optional[str] = None,
        records_processed: Optional[int] = None,
        records_successful: Optional[int] = None,
        records_failed: Optional[int] = None,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update pipeline execution status.

        Args:
            execution_id: Execution ID to update
            status: New status
            stage: New stage
            records_processed: Number of records processed
            records_successful: Number of successful records
            records_failed: Number of failed records
            processing_time_ms: Processing time in milliseconds
            error_message: Error message if any

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                # Update status
                success = await db_queries.update_data_pipeline_status(
                    execution_id=execution_id,
                    status=status,
                    stage=stage,
                    error_message=error_message
                )

                if success and any(v is not None for v in [records_processed, records_successful, records_failed, processing_time_ms]):
                    # Update additional metrics
                    await self._update_pipeline_metrics(
                        execution_id, records_processed, records_successful,
                        records_failed, processing_time_ms
                    )

                logger.info(f"Updated pipeline {execution_id} status to {status}")
                return True

        except Exception as e:
            logger.error(f"Pipeline status update failed: {e}")
            return False

    async def _update_pipeline_metrics(
        self,
        execution_id: str,
        records_processed: Optional[int],
        records_successful: Optional[int],
        records_failed: Optional[int],
        processing_time_ms: Optional[int]
    ) -> bool:
        """Update pipeline execution metrics."""
        try:
            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                # Get current record
                pipeline_records = await db_queries.get_data_pipeline_records(
                    execution_id=execution_id
                )

                if pipeline_records:
                    record = pipeline_records[0]

                    # Update metrics
                    if records_processed is not None:
                        record.records_processed = records_processed
                    if records_successful is not None:
                        record.records_successful = records_successful
                    if records_failed is not None:
                        record.records_failed = records_failed
                    if processing_time_ms is not None:
                        record.processing_time_ms = processing_time_ms

                    record.updated_at = datetime.now(timezone.utc)
                    await session.commit()

                return True

        except Exception as e:
            logger.error(f"Pipeline metrics update failed: {e}")
            return False

    @cache_result(ttl_seconds=300)  # Cache for 5 minutes
    async def get_market_data(
        self,
        symbol: str,
        exchange: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[MarketDataRecord]:
        """
        Retrieve market data using proper database integration.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            limit: Maximum number of records to return

        Returns:
            List[MarketDataRecord]: List of market data records
        """
        try:
            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                records = await db_queries.get_market_data_records(
                    symbol=symbol,
                    exchange=exchange,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )

                logger.info(f"Retrieved {len(records)} market data records for {symbol}")
                return records

        except Exception as e:
            logger.error(f"Market data retrieval failed: {e}")
            return []

    @cache_result(ttl_seconds=600)  # Cache for 10 minutes
    async def get_features(
        self,
        symbol: str,
        feature_type: Optional[str] = None,
        feature_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[FeatureRecord]:
        """
        Retrieve calculated features using proper database integration.

        Args:
            symbol: Trading symbol
            feature_type: Type of feature to filter by
            feature_name: Name of feature to filter by
            start_time: Start time for feature retrieval
            end_time: End time for feature retrieval

        Returns:
            List[FeatureRecord]: List of feature records
        """
        try:
            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                features = await db_queries.get_feature_records(
                    symbol=symbol,
                    feature_type=feature_type,
                    feature_name=feature_name,
                    start_time=start_time,
                    end_time=end_time
                )

                logger.info(f"Retrieved {len(features)} features for {symbol}")
                return features

        except Exception as e:
            logger.error(f"Feature retrieval failed: {e}")
            return []

    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old data using proper database integration.

        Args:
            days_to_keep: Number of days of data to keep

        Returns:
            int: Number of records cleaned up
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                # Clean up market data records
                deleted_count = await db_queries.delete_old_market_data(cutoff_date)

                logger.info(f"Cleaned up {deleted_count} old market data records")
                return deleted_count

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return 0

    async def get_data_quality_summary(
        self,
        symbol: Optional[str] = None,
        data_source: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get data quality summary using proper database integration.

        Args:
            symbol: Symbol to filter by
            data_source: Data source to filter by
            days: Number of days to look back

        Returns:
            Dict[str, Any]: Quality summary statistics
        """
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days)

            async with get_async_session() as session:
                db_queries = DatabaseQueries(session)

                quality_records = await db_queries.get_data_quality_records(
                    symbol=symbol,
                    data_source=data_source,
                    start_time=start_time
                )

                if not quality_records:
                    return {
                        "total_records": 0,
                        "average_overall_score": 0.0,
                        "quality_distribution": {},
                        "top_issues": []
                    }

                # Calculate summary statistics
                total_records = len(quality_records)
                avg_overall_score = sum(r.overall_score for r in quality_records) / total_records

                # Quality distribution
                quality_distribution = {
                    "excellent": len([r for r in quality_records if r.overall_score >= 0.9]),
                    "good": len([r for r in quality_records if 0.7 <= r.overall_score < 0.9]),
                    "fair": len([r for r in quality_records if 0.5 <= r.overall_score < 0.7]),
                    "poor": len([r for r in quality_records if r.overall_score < 0.5])
                }

                # Top issues
                total_missing = sum(r.missing_data_count for r in quality_records)
                total_outliers = sum(r.outlier_count for r in quality_records)
                total_duplicates = sum(r.duplicate_count for r in quality_records)

                top_issues = [
                    {"type": "missing_data", "count": total_missing},
                    {"type": "outliers", "count": total_outliers},
                    {"type": "duplicates", "count": total_duplicates}
                ]

                return {
                    "total_records": total_records,
                    "average_overall_score": round(avg_overall_score, 3),
                    "quality_distribution": quality_distribution,
                    "top_issues": top_issues
                }

        except Exception as e:
            logger.error(f"Quality summary retrieval failed: {e}")
            return {}

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        try:
            if self.influx_client:
                self.influx_client.disconnect()
            logger.info("DataIntegrationService cleanup completed")
        except Exception as e:
            logger.error(f"Error during DataIntegrationService cleanup: {e}")
