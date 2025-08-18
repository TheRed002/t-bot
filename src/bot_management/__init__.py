"""
Bot Management System for T-Bot Trading Platform.

This module provides comprehensive bot orchestration and resource management
capabilities, enabling multiple bots to run different strategies simultaneously
with proper resource allocation and coordination.

CRITICAL: This module integrates with P-016 (execution engine), P-011 (strategies),
P-003+ (exchanges), and P-010A (capital management) components.

Components:
- BotInstance: Individual bot that runs a specific strategy
- BotOrchestrator: Central controller for all bots
- ResourceManager: Shared resource allocation and management
- BotCoordinator: Inter-bot communication and coordination
- BotMonitor: Health and performance tracking
- BotLifecycle: Bot lifecycle management
"""

from .bot_coordinator import BotCoordinator
from .bot_instance import BotInstance
from .bot_lifecycle import BotLifecycle
from .bot_monitor import BotMonitor
from .bot_orchestrator import BotOrchestrator
from .resource_manager import ResourceManager

__all__ = [
    "BotCoordinator",
    "BotInstance",
    "BotLifecycle",
    "BotMonitor",
    "BotOrchestrator",
    "ResourceManager",
]
