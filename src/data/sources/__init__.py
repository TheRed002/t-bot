"""
Data Sources Module

This module provides data source integrations for the trading bot:
- Market data from exchanges (OHLCV, order book, trades)
- News sentiment from NewsAPI
- Social media sentiment (Twitter/Reddit APIs)  
- Economic indicators (FRED API)
- Alternative data (weather, satellite data)

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-003+: Exchange interfaces
- P-007A: Utility functions and decorators
"""

from .market_data import MarketDataSource
from .news_data import NewsDataSource
from .social_media import SocialMediaDataSource
from .alternative_data import AlternativeDataSource

__all__ = [
    'MarketDataSource',
    'NewsDataSource', 
    'SocialMediaDataSource',
    'AlternativeDataSource'
]
