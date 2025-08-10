"""
Data Management System

This module provides comprehensive data management for the trading bot:
- Data sources integration (market data, news, social media, alternative data)
- Data pipeline (ingestion, processing, validation, storage)
- Data quality management (validation, cleaning, monitoring)

Components:
- Data Sources: Market data, news, social media, alternative data sources
- Data Pipeline: Real-time and batch data ingestion and processing
- Data Quality: Real-time validation, cleaning, and monitoring
- Statistical analysis and drift detection
- Cross-source consistency checks

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models and connections
- P-002A: Error handling framework
- P-003+: Exchange interfaces
- P-007A: Utility functions and decorators
"""

# Data Quality Management (P-000)
from .quality.validation import DataValidator
from .quality.cleaning import DataCleaner
from .quality.monitoring import QualityMonitor

# Data Sources (P-000A)
from .sources.market_data import MarketDataSource
from .sources.news_data import NewsDataSource
from .sources.social_media import SocialMediaDataSource
from .sources.alternative_data import AlternativeDataSource

# Data Pipeline (P-000A)
from .pipeline.ingestion import DataIngestionPipeline
from .pipeline.processing import DataProcessor
from .pipeline.validation import PipelineValidator
from .pipeline.storage import DataStorageManager

# Feature Engineering (P-015)
from .features.technical_indicators import TechnicalIndicatorCalculator
from .features.statistical_features import StatisticalFeatureCalculator
from .features.alternative_features import AlternativeFeatureCalculator

__all__ = [
    # Data Quality Management
    'DataValidator',
    'DataCleaner',
    'QualityMonitor',

    # Data Sources
    'MarketDataSource',
    'NewsDataSource',
    'SocialMediaDataSource',
    'AlternativeDataSource',

    # Data Pipeline
    'DataIngestionPipeline',
    'DataProcessor',
    'PipelineValidator',
    'DataStorageManager',

    # Feature Engineering
    'TechnicalIndicatorCalculator',
    'StatisticalFeatureCalculator',
    'AlternativeFeatureCalculator'
]
