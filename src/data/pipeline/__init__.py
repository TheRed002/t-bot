"""
Data Pipeline Module

This module provides comprehensive data ingestion and processing pipeline:
- Real-time stream processing and batch data collection
- Data normalization across multiple sources
- Timestamp synchronization and alignment
- Pipeline orchestration and management
- Data storage and retrieval

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

from .ingestion import DataIngestionPipeline
from .processing import DataProcessor
from .validation import PipelineValidator
from .storage import DataStorageManager

__all__ = [
    'DataIngestionPipeline',
    'DataProcessor',
    'PipelineValidator',
    'DataStorageManager'
]
