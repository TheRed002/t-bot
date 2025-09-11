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

# Keep __init__ lightweight to avoid importing heavy dependencies at import time.
# Import only quality submodules and core services here; other submodules should be imported
# directly
# by consumers as needed to avoid circular or heavy dependency trees (e.g., exchanges).
# DI registration functions
from .di_registration import (
    configure_data_dependencies,
    get_data_cache,
    get_data_service,
    get_data_service_factory,
    get_data_storage,
    get_data_validator,
    get_service_data_validator,
    register_data_services,
)
from .factory import DataServiceFactory
from .interfaces import (
    DataCacheInterface,
    DataServiceInterface,
    DataStorageInterface,
    DataValidatorInterface,
)
from .quality.cleaning import DataCleaner
from .quality.monitoring import QualityMonitor
from .quality.validation import DataValidator
from .services import DataService

__all__ = [
    "DataCleaner",
    # Data Services
    "DataService",
    "DataServiceFactory",
    # Data Interfaces
    "DataServiceInterface",
    "DataStorageInterface",
    "DataCacheInterface",
    "DataValidatorInterface",
    # Data Quality Management
    "DataValidator",
    "QualityMonitor",
    # Dependency Injection
    "register_data_services",
    "configure_data_dependencies",
    "get_data_service",
    "get_data_service_factory",
    "get_data_storage",
    "get_data_cache",
    "get_data_validator",
    "get_service_data_validator",
    # Quality submodule exports only (other submodules must be imported directly)
]
