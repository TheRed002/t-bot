"""
Enterprise-Grade State Management System for T-Bot Trading Platform.

This module provides a comprehensive, production-ready state management solution with:
- Enterprise-grade StateService architecture with dependency injection
- Database abstraction layer eliminating direct DB access
- Comprehensive state recovery and audit trail system
- Real-time monitoring and health checks
- Factory pattern for service composition
- Full compliance with trading system reliability requirements

Core Architecture:
- StateService: Central state coordination with DatabaseService integration
- StateRecoveryManager: Point-in-time recovery and audit trails
- StateMonitoringService: Health monitoring and performance metrics
- StateServiceFactory: Dependency injection and service composition
- StateServiceRegistry: Singleton management and lifecycle control

Key Features:
✅ Eliminates all direct database access violations
✅ Enterprise-grade recovery and audit capabilities
✅ Comprehensive health monitoring and alerting
✅ Dependency injection for testability
✅ Performance optimization and SLA compliance
✅ Trading-specific state handling (bots, positions, orders)
✅ Real-time state synchronization with conflict resolution
✅ Complete validation framework with business rules
"""

# Core state management services
# Legacy components (backward compatibility)
from .checkpoint_manager import CheckpointManager

# Enterprise components
from .factory import (
    StateServiceFactory,
    StateServiceRegistry,
    create_default_state_service,
    create_test_state_service,
    get_state_service,
)
from .monitoring import (
    Alert,
    AlertSeverity,
    HealthCheck,
    HealthStatus,
    Metric,
    MetricType,
    PerformanceReport,
    StateMonitoringService,
)
from .quality_controller import QualityController
from .recovery import (
    AuditEntry,
    AuditEventType,
    CorruptionReport,
    RecoveryOperation,
    RecoveryPoint,
    RecoveryStatus,
    StateRecoveryManager,
)

# Backward compatibility
from .state_manager import StateManager

# Core consolidated state management
from .state_service import StatePriority, StateService, StateType
from .state_sync_manager import StateSyncManager, SyncEventType
from .state_validator import StateValidator

# Trading-specific components
from .trade_lifecycle_manager import TradeEvent, TradeLifecycleManager

# Public API exports
__all__ = [
    "Alert",
    "AlertSeverity",
    "AuditEntry",
    "AuditEventType",
    "CheckpointManager",
    "CorruptionReport",
    "HealthCheck",
    "HealthStatus",
    "Metric",
    "MetricType",
    "PerformanceReport",
    "QualityController",
    "RecoveryOperation",
    "RecoveryPoint",
    "RecoveryStatus",
    "StateManager",
    "StateMonitoringService",
    "StatePriority",
    "StateRecoveryManager",
    "StateService",
    "StateServiceFactory",
    "StateServiceRegistry",
    "StateSyncManager",
    "StateType",
    "StateValidator",
    "SyncEventType",
    "TradeEvent",
    "TradeLifecycleManager",
    "create_default_state_service",
    "create_test_state_service",
    "get_state_service",
]

# Version information
__version__ = "2.0.0"
__author__ = "T-Bot Development Team"
__description__ = "Enterprise-grade state management system with full database service integration"
