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

# Simplified exports
__all__ = [
    "BotCoordinator",
    "BotInstance",
    "BotLifecycle",
    "BotMonitor",
    "BotService",
    "ResourceManager",
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
        from .bot_instance import BotInstance

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
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
