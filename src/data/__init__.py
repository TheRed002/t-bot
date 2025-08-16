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
# Import only quality submodules and core services here; other submodules should be imported directly
# by consumers as needed to avoid circular or heavy dependency trees (e.g., exchanges).
from .quality.cleaning import DataCleaner
from .quality.monitoring import QualityMonitor
from .quality.validation import DataValidator
from .services import DataIntegrationService

__all__ = [
    "DataCleaner",
    # Data Integration Service
    "DataIntegrationService",
    # Data Quality Management
    "DataValidator",
    "QualityMonitor",
    # Quality submodule exports only (other submodules must be imported directly)
]
