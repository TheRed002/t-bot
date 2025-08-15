"""
State Management module for the T-Bot trading system.

This module provides comprehensive state management capabilities including:
- P-018: State persistence and recovery system
- P-023: Trade lifecycle management  
- P-024: Quality controls and validation
- P-025: Real-time state synchronization

The state management system ensures:
- Consistent state across all bot components
- Crash recovery and state persistence
- Trade lifecycle tracking and quality metrics
- Real-time synchronization and conflict resolution

Components:
- StateManager: Core state management and persistence
- TradeLifecycleManager: Trade state transitions and tracking
- QualityController: Pre/post-trade validation and scoring
- StateSyncManager: Real-time synchronization and conflict resolution
- CheckpointManager: State checkpointing and recovery
"""

from .checkpoint_manager import CheckpointManager
from .quality_controller import QualityController
from .state_manager import StateManager
from .state_sync_manager import StateSyncManager
from .trade_lifecycle_manager import TradeLifecycleManager

__all__ = [
    "StateManager",
    "TradeLifecycleManager", 
    "QualityController",
    "StateSyncManager",
    "CheckpointManager",
]