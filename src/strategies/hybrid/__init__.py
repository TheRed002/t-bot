"""
Hybrid Strategy Implementation Module

This module implements hybrid trading strategies that combine different approaches:
- Rule-based systems with AI predictions
- Ensemble methods with dynamic weight adjustment
- Intelligent fallback mechanisms for robust trading

All strategies in this module inherit from BaseStrategy and integrate with the
existing risk management, error handling, and logging systems.
"""

from .ensemble import EnsembleStrategy
from .fallback import FallbackStrategy
from .rule_based_ai import RuleBasedAIStrategy

__all__ = ["EnsembleStrategy", "FallbackStrategy", "RuleBasedAIStrategy"]
