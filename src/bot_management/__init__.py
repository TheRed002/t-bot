"""
Bot Management System for T-Bot Trading Platform - CONSOLIDATED.

This module provides comprehensive bot orchestration and resource management
capabilities, enabling multiple bots to run different strategies simultaneously
with proper resource allocation and coordination.

CONSOLIDATED: Removed duplicate orchestrator implementations, using bot_coordinator.py
as the main orchestration component with proper service layer integration.

CRITICAL: This module integrates with execution, strategies, exchanges, and
capital management components through service layer dependencies.

Core Components:
- BotService: Core bot management business logic with proper DI
- BotCoordinator: Main orchestration and inter-bot coordination component
- BotMonitor: Advanced monitoring using service dependencies

Legacy Components (maintained for compatibility):
- BotInstance: Individual bot that runs a specific strategy
- BotLifecycle: Bot lifecycle management
- ResourceManager: Shared resource allocation and management
"""

# Core service layer components
from .bot_coordinator import BotCoordinator
from .bot_instance import BotInstance
from .bot_lifecycle import BotLifecycle
from .bot_monitor import BotMonitor
from .resource_manager import ResourceManager
from .service import BotService

__all__ = [
    # Core components
    "BotCoordinator",  # Main orchestration component
    # Legacy components (maintained for compatibility)
    "BotInstance",
    "BotLifecycle",
    "BotMonitor",  # Monitoring and health checks
    "BotService",  # Service layer business logic
    "ResourceManager",
]
