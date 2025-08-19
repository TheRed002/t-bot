"""
Configuration management for the T-Bot trading system.

This module provides backward compatibility while organizing configs
into domain-specific modules.
"""

from .base import BaseConfig
from .database import DatabaseConfig
from .exchange import ExchangeConfig
from .strategy import StrategyConfig
from .risk import RiskConfig
from .main import Config, get_config

# For backward compatibility, export Config as the default
__all__ = [
    'BaseConfig',
    'DatabaseConfig',
    'ExchangeConfig',
    'StrategyConfig',
    'RiskConfig',
    'Config',
    'get_config'
]