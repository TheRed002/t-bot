"""
Web Interface for T-Bot Trading System - Refactored Architecture.

This module provides the FastAPI-based web interface for the T-Bot trading system,
including authentication, REST APIs, WebSocket endpoints, and real-time data streaming.

Refactored features:
- API facade pattern for service abstraction
- Comprehensive authentication layer with RBAC
- API versioning with backward compatibility
- Unified WebSocket manager for real-time data
- Service layer abstractions for decoupling

The web interface serves as the primary user interface for:
- Bot management and monitoring
- Portfolio tracking and analysis
- Trading operations and order management
- Strategy configuration and deployment
- Risk management and monitoring
- ML model management and inference
"""

# Import core refactored components
from .auth import get_auth_manager
from .facade import get_api_facade, get_service_registry
from .versioning import get_version_manager
from .websockets.unified_manager import get_unified_websocket_manager

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "get_api_facade",
    "get_auth_manager",
    "get_service_registry",
    "get_unified_websocket_manager",
    "get_version_manager",
]


# Lazy import the app to avoid monitoring issues during testing
def create_app(*args, **kwargs):
    """Lazy import and create app."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
