"""
CONSOLIDATED Bot Management System for T-Bot Trading Platform.

This module provides bot orchestration and resource management with
simplified, clear architecture following the service layer pattern.

Core Components:
- BotService: Core bot management business logic
- BotCoordinator: Bot coordination and communication
- BotMonitor: Health monitoring and performance tracking
- BotInstance: Individual bot implementation
- BotLifecycle: Bot lifecycle management
- ResourceManager: Resource allocation and management
"""

# Service layer exports
__all__ = [
    "BotCoordinator",
    "BotInstance",
    "BotLifecycle",
    "BotMonitor",
    "BotService",
    "ResourceManager",
    # Service layer classes
    "BotManagementController",
    "BotInstanceService",
    "BotLifecycleService",
    "BotCoordinationService",
    "BotMonitoringService",
    "BotResourceService",
    # Service interfaces
    "IBotInstanceService",
    "IBotLifecycleService",
    "IBotCoordinationService",
    "IBotMonitoringService",
    "IResourceManagementService",
]


# Simplified submodule list for tests
_submodules = [
    "bot_coordinator",
    "bot_instance",
    "bot_lifecycle",
    "bot_monitor",
    "service",
    "resource_manager",
    "factory",
    "repository",
    "di_registration",
]


def __getattr__(name: str):
    """Lazy import to prevent circular dependencies."""
    # Handle submodule imports for tests
    if name in _submodules:
        import importlib

        return importlib.import_module(f".{name}", __name__)

    # Handle class imports
    if name == "BotCoordinator":
        from .bot_coordinator import BotCoordinator

        return BotCoordinator
    elif name == "BotInstance":
        from .bot_entity import BotInstance

        return BotInstance
    elif name == "BotLifecycle":
        from .bot_lifecycle import BotLifecycle

        return BotLifecycle
    elif name == "BotMonitor":
        from .bot_monitor import BotMonitor

        return BotMonitor
    elif name == "BotService":
        from .service import BotService

        return BotService
    elif name == "ResourceManager":
        from .resource_manager import ResourceManager

        return ResourceManager
    # Service layer classes
    elif name == "BotManagementController":
        from .controller import BotManagementController

        return BotManagementController
    elif name == "BotInstanceService":
        from .instance_service import BotInstanceService

        return BotInstanceService
    elif name == "BotLifecycleService":
        from .lifecycle_service import BotLifecycleService

        return BotLifecycleService
    elif name == "BotCoordinationService":
        from .coordination_service import BotCoordinationService

        return BotCoordinationService
    elif name == "BotMonitoringService":
        from .monitoring_service import BotMonitoringService

        return BotMonitoringService
    elif name == "BotResourceService":
        from .resource_service import BotResourceService

        return BotResourceService
    # Service interfaces
    elif name in ["IBotInstanceService", "IBotLifecycleService", "IBotCoordinationService",
                  "IBotMonitoringService", "IResourceManagementService"]:
        from .interfaces import (
            IBotInstanceService,
            IBotLifecycleService,
            IBotCoordinationService,
            IBotMonitoringService,
            IResourceManagementService,
        )
        return locals()[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
