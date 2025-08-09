"""
Feature Engineering Framework

This module provides comprehensive feature engineering capabilities for the trading bot:
- Technical indicators (50+ TA-Lib indicators)
- Statistical features (rolling statistics, autocorrelation)
- Alternative data features (news sentiment, social media, economic indicators)

Components:
- Technical Indicators: Price, momentum, volume, volatility indicators
- Statistical Features: Statistical analysis and regime detection
- Alternative Features: Multi-source alternative data processing

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- P-000A: Data pipeline integration
"""

from .technical_indicators import TechnicalIndicatorCalculator
from .statistical_features import StatisticalFeatureCalculator
from .alternative_features import AlternativeFeatureCalculator

__all__ = [
    'TechnicalIndicatorCalculator',
    'StatisticalFeatureCalculator', 
    'AlternativeFeatureCalculator'
]
