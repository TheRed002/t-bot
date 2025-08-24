"""
API Versioning system for T-Bot Trading System.

This package provides comprehensive API versioning support with
backward compatibility and feature deprecation management.
"""

from .decorators import deprecated, versioned_endpoint
from .middleware import VersioningMiddleware, VersionRoutingMiddleware
from .version_manager import APIVersion, VersionManager, get_version_manager

__all__ = [
    "APIVersion",
    "VersionManager",
    "VersionRoutingMiddleware",
    "VersioningMiddleware",
    "deprecated",
    "get_version_manager",
    "versioned_endpoint",
]
