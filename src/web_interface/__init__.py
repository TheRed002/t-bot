"""
Web Interface for T-Bot Trading System.

This module provides the FastAPI-based web interface for the T-Bot trading system,
including authentication, REST APIs, WebSocket endpoints, and real-time data streaming.

The web interface serves as the primary user interface for:
- Bot management and monitoring
- Portfolio tracking and analysis
- Trading operations and order management
- Strategy configuration and deployment
- Risk management and monitoring
- ML model management and inference

CRITICAL: This integrates with P-001 (core types, exceptions, config),
P-002A (error handling), and all trading system components.
"""

from .app import create_app

__all__ = ["create_app"]
