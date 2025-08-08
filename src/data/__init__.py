"""
Data Quality Management System

This module provides comprehensive data quality monitoring and validation for the trading bot.
It ensures high-quality data for ML models and trading strategies.

Components:
- Real-time validation for incoming data
- Data cleaning and preprocessing
- Quality monitoring and alerting
- Statistical outlier detection
- Cross-source consistency checks

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

from .quality.validation import DataValidator
from .quality.cleaning import DataCleaner
from .quality.monitoring import QualityMonitor

__all__ = [
    'DataValidator',
    'DataCleaner', 
    'QualityMonitor'
]
