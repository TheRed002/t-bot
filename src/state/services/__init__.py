"""
State Services Module - Service Layer Implementation.

This module provides service layer abstractions for state management,
separating business logic from components and infrastructure concerns.
"""

# Import shared types to avoid circular dependencies
from ..types import PostTradeAnalysis, PreTradeValidation, ValidationCheck, ValidationResult
from .quality_service import (
    QualityService,
    QualityServiceProtocol,
)
from .state_business_service import (
    StateBusinessService,
    StateBusinessServiceProtocol,
)
from .state_persistence_service import (
    StatePersistenceService,
    StatePersistenceServiceProtocol,
)
from .state_synchronization_service import (
    StateSynchronizationService,
    StateSynchronizationServiceProtocol,
)
from .state_validation_service import (
    StateValidationService,
    StateValidationServiceProtocol,
)
from .trade_lifecycle_service import (
    TradeLifecycleService,
    TradeLifecycleServiceProtocol,
)

__all__ = [
    # Services
    "QualityService",
    "QualityServiceProtocol",
    "StateBusinessService",
    "StateBusinessServiceProtocol",
    "StatePersistenceService",
    "StatePersistenceServiceProtocol",
    "StateSynchronizationService",
    "StateSynchronizationServiceProtocol",
    "StateValidationService",
    "StateValidationServiceProtocol",
    "TradeLifecycleService",
    "TradeLifecycleServiceProtocol",
    # Types
    "PostTradeAnalysis",
    "PreTradeValidation",
    "ValidationCheck",
    "ValidationResult",
]
