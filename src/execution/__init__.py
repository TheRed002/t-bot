"""
Advanced Execution Engine for T-Bot Trading System (P-016).

This module provides sophisticated execution algorithms that minimize trading costs
and maximize execution quality for financial trading operations.

Components:
- ExecutionEngine: Main orchestrator for order execution
- OrderManager: Order lifecycle management
- Algorithms: TWAP, VWAP, Iceberg, Smart Router
- Slippage: Prediction models and transaction cost analysis

All classes integrate with existing exchange infrastructure and risk management.
"""

from .execution_engine import ExecutionEngine
from .order_manager import OrderManager

__all__ = ["ExecutionEngine", "OrderManager"]
