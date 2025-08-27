"""
Core base classes and patterns for the T-Bot trading system.

This module provides the foundational architecture components that all other
modules inherit from. It implements essential patterns including:

- Service Layer Pattern (BaseService)
- Repository Pattern (BaseRepository)
- Factory Pattern (BaseFactory)
- Observer Pattern (BaseEventEmitter)
- Component Lifecycle Management
- Health Check Framework
- Dependency Injection Support
- Metrics and Monitoring Integration

All classes follow modern Python practices with:
- Full async/await support
- Comprehensive type hints
- Structured logging with correlation IDs
- Resource management and cleanup
- Error handling with context
- Performance monitoring
"""

from typing import TYPE_CHECKING

from .component import BaseComponent
from .events import BaseEventEmitter
from .factory import BaseFactory
from .health import HealthCheckManager
from .interfaces import (
    Configurable,
    HealthCheckable,
    HealthStatus,
    Injectable,
    Lifecycle,
    Loggable,
    Monitorable,
)
from .repository import BaseRepository
from .service import BaseService, TransactionalService

if TYPE_CHECKING:
    pass

__all__ = [
    # Core Classes
    "BaseComponent",
    "BaseEventEmitter",
    "BaseFactory",
    "BaseRepository",
    "BaseService",
    # Interfaces
    "Configurable",
    "HealthCheckManager",
    "HealthCheckable",
    "HealthStatus",
    "Injectable",
    "Lifecycle",
    "Loggable",
    "Monitorable",
    "TransactionalService",
]
