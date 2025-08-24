"""
API Facade package for T-Bot Trading System.

This package provides unified access to all system services through
a clean facade pattern, abstracting away the underlying implementations.
"""

from .api_facade import APIFacade, get_api_facade
from .service_registry import ServiceRegistry, get_service_registry

__all__ = ["APIFacade", "ServiceRegistry", "get_api_facade", "get_service_registry"]
