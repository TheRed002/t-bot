"""
Data Storage Manager

This module provides data storage management for the pipeline:
- Database storage operations
- Data persistence and retrieval
- Storage optimization and cleanup
- Storage health monitoring

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

# Import from P-001 core components
from src.core.types import MarketData, StorageMode
from src.core.exceptions import DataError, DatabaseError
from src.core.config import Config
from src.core.logging import get_logger

# Import from P-002 database components
from src.database.connection import get_async_session
from src.database.influxdb_client import InfluxDBClientWrapper

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution, retry

logger = get_logger(__name__)


# StorageMode is now imported from core.types


@dataclass
class StorageMetrics:
    """Storage operation metrics"""
    total_records_stored: int
    successful_stores: int
    failed_stores: int
    avg_storage_time: float
    storage_rate: float
    last_storage_time: datetime


class DataStorageManager:
    """
    Data storage manager for pipeline data persistence.
    
    This class handles storage operations for all data types processed
    through the pipeline, with support for different storage modes.
    """
    
    def __init__(self, config: Config):
        """Initialize data storage manager."""
        self.config = config
        self.error_handler = ErrorHandler(config)
        
        # Storage configuration
        storage_config = getattr(config, 'data_storage', {})
        if hasattr(storage_config, 'get'):
            self.storage_mode = StorageMode(storage_config.get('mode', 'batch'))
            self.batch_size = storage_config.get('batch_size', 100)
            self.buffer_threshold = storage_config.get('buffer_threshold', 50)
            self.cleanup_interval = storage_config.get('cleanup_interval', 3600)
        else:
            self.storage_mode = StorageMode('batch')
            self.batch_size = 100
            self.buffer_threshold = 50
            self.cleanup_interval = 3600
        
        # Initialize InfluxDB client for time series market data
        influx_config = getattr(config, 'influxdb', {})
        if hasattr(influx_config, 'get'):
            self.influx_client = InfluxDBClientWrapper(
                url=influx_config.get('url', 'http://localhost:8086'),
                token=influx_config.get('token', ''),
                org=influx_config.get('org', 'trading-bot'),
                bucket=influx_config.get('bucket', 'market-data')
            )
        else:
            # Default InfluxDB configuration
            self.influx_client = InfluxDBClientWrapper(
                url='http://localhost:8086',
                token='',
                org='trading-bot', 
                bucket='market-data'
            )
        
        # Storage buffers
        self.storage_buffer: List[Dict[str, Any]] = []
        
        # Storage metrics
        self.metrics = StorageMetrics(
            total_records_stored=0,
            successful_stores=0,
            failed_stores=0,
            avg_storage_time=0.0,
            storage_rate=0.0,
            last_storage_time=datetime.now(timezone.utc)
        )
        
        logger.info("DataStorageManager initialized")
    
    @time_execution
    @retry(max_attempts=3, delay=1.0)
    async def store_market_data(self, data: MarketData) -> bool:
        """
        Store market data to database.
        
        Args:
            data: Market data to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.storage_mode == StorageMode.REAL_TIME:
                return await self._store_real_time(data)
            elif self.storage_mode == StorageMode.BUFFER:
                return await self._store_to_buffer(data)
            else:  # BATCH mode
                return await self._store_to_buffer(data)
                
        except Exception as e:
            logger.error(f"Failed to store market data: {str(e)}")
            self.metrics.failed_stores += 1
            return False
    
    async def _store_real_time(self, data: MarketData) -> bool:
        """Store data immediately to InfluxDB."""
        try:
            # Write market data to InfluxDB
            await self.influx_client.write_market_data(
                data.symbol,
                data.exchange,
                data.price,
                data.volume,
                bid=data.bid,
                ask=data.ask,
                high=data.high_price,
                low=data.low_price,
                open_price=data.open_price,
                timestamp=data.timestamp
            )
            
            self.metrics.successful_stores += 1
            self.metrics.total_records_stored += 1
            self.metrics.last_storage_time = datetime.now(timezone.utc)
            
            return True
                
        except Exception as e:
            logger.error(f"Real-time storage failed: {str(e)}")
            self.metrics.failed_stores += 1
            return False
    
    async def _store_to_buffer(self, data: MarketData) -> bool:
        """Add data to storage buffer."""
        try:
            buffer_item = {
                'data': data,
                'timestamp': datetime.now(timezone.utc),
                'type': 'market_data'
            }
            
            self.storage_buffer.append(buffer_item)
            
            # Flush buffer if threshold reached
            if len(self.storage_buffer) >= self.buffer_threshold:
                await self._flush_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"Buffer storage failed: {str(e)}")
            return False
    
    @time_execution
    async def _flush_buffer(self) -> bool:
        """Flush storage buffer to InfluxDB."""
        try:
            if not self.storage_buffer:
                return True
            
            # Prepare market data points for InfluxDB
            market_data_points = []
            
            for item in self.storage_buffer:
                if item['type'] == 'market_data':
                    data = item['data']
                    market_data_points.append(data)
            
            # Bulk write to InfluxDB
            if market_data_points:
                await self.influx_client.write_market_data_batch(market_data_points)
                
                # Update metrics
                stored_count = len(market_data_points)
                self.metrics.successful_stores += stored_count
                self.metrics.total_records_stored += stored_count
                self.metrics.last_storage_time = datetime.now(timezone.utc)
                
                logger.info(f"Flushed {stored_count} records to database")
                return True
                
        except Exception as e:
            logger.error(f"Buffer flush failed: {str(e)}")
            self.metrics.failed_stores += len(self.storage_buffer)
            return False
    
    async def store_batch(self, data_list: List[MarketData]) -> int:
        """
        Store a batch of data records.
        
        Args:
            data_list: List of market data to store
            
        Returns:
            int: Number of successfully stored records
        """
        try:
            # Bulk write to InfluxDB
            await self.influx_client.write_market_data_batch(data_list)
            
            # Update metrics
            stored_count = len(data_list)
            self.metrics.successful_stores += stored_count
            self.metrics.total_records_stored += stored_count
            self.metrics.last_storage_time = datetime.now(timezone.utc)
            
            logger.info(f"Stored batch of {stored_count} records")
            return stored_count
                
        except Exception as e:
            logger.error(f"Batch storage failed: {str(e)}")
            self.metrics.failed_stores += len(data_list)
            return 0
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Cleanup old data from storage.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            int: Number of records cleaned up
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            async with get_async_session() as session:
                # Delete old market data records
                result = await session.execute(
                    f"DELETE FROM market_data_records WHERE timestamp < '{cutoff_date}'"
                )
                
                await session.commit()
                deleted_count = result.rowcount
                
                logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}")
            return 0
    
    def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage operation metrics."""
        success_rate = (
            self.metrics.successful_stores / 
            (self.metrics.successful_stores + self.metrics.failed_stores)
            if (self.metrics.successful_stores + self.metrics.failed_stores) > 0 else 0.0
        )
        
        return {
            "total_records_stored": self.metrics.total_records_stored,
            "successful_stores": self.metrics.successful_stores,
            "failed_stores": self.metrics.failed_stores,
            "success_rate": success_rate,
            "storage_rate": self.metrics.storage_rate,
            "avg_storage_time": self.metrics.avg_storage_time,
            "last_storage_time": self.metrics.last_storage_time,
            "buffer_size": len(self.storage_buffer),
            "storage_mode": self.storage_mode.value,
            "configuration": {
                "batch_size": self.batch_size,
                "buffer_threshold": self.buffer_threshold,
                "cleanup_interval": self.cleanup_interval
            }
        }
    
    async def force_flush(self) -> bool:
        """Force flush all buffered data."""
        try:
            if self.storage_buffer:
                return await self._flush_buffer()
            return True
            
        except Exception as e:
            logger.error(f"Force flush failed: {str(e)}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup storage manager resources."""
        try:
            # Flush any remaining buffered data
            await self.force_flush()
            logger.info("DataStorageManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during DataStorageManager cleanup: {str(e)}")
