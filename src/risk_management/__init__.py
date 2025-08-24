"""
Risk Management Module - Enterprise-Grade Risk Management System.

This module provides comprehensive risk management functionality with both
legacy components (deprecated) and the new RiskService architecture.

## New Architecture (RECOMMENDED):
- RiskService: Enterprise-grade risk management service with full integration
- RiskManagementFactory: Factory for creating and managing risk components
- DatabaseService integration: No direct database access
- StateService integration: Centralized state management
- Real-time monitoring and alerting
- Comprehensive caching and circuit breakers

## Legacy Components (DEPRECATED):
- RiskManager: Legacy risk manager (now wrapper around RiskService)
- PositionSizer: Legacy position sizing (use RiskService.calculate_position_size())
- RiskCalculator: Legacy risk metrics (use RiskService.calculate_risk_metrics())

## Quick Start:

### Using New RiskService (Recommended):
```python
from src.risk_management import create_risk_service

# Create RiskService with dependencies
risk_service = create_risk_service(
    database_service=database_service, state_service=state_service, config=config
)

# Calculate position size
position_size = await risk_service.calculate_position_size(
    signal=signal, available_capital=capital, current_price=price
)

# Calculate risk metrics
risk_metrics = await risk_service.calculate_risk_metrics(positions, market_data)

# Get risk summary
risk_summary = await risk_service.get_risk_summary()
```

### Using Factory Pattern:
```python
from src.risk_management import get_risk_factory

factory = get_risk_factory(
    database_service=database_service, state_service=state_service, config=config
)

# Get recommended component (RiskService or RiskManager fallback)
risk_component = factory.get_recommended_component()
```

### Legacy Usage (Deprecated):
```python
from src.risk_management import RiskManager

# DEPRECATED - migrate to RiskService
risk_manager = RiskManager(config)
position_size = await risk_manager.calculate_position_size(signal, capital, price)
```

## Migration Guide:
- RiskManager.calculate_position_size() -> RiskService.calculate_position_size()
- RiskManager.validate_signal() -> RiskService.validate_signal()
- RiskManager.calculate_risk_metrics() -> RiskService.calculate_risk_metrics()
- PositionSizer -> Use RiskService.calculate_position_size()
- RiskCalculator -> Use RiskService.calculate_risk_metrics()

## Features:
- Multiple position sizing methods (Fixed %, Kelly Criterion, Volatility-Adjusted, Confidence-Weighted)
- Comprehensive risk metrics (VaR, Expected Shortfall, Drawdown, Sharpe Ratio)
- Real-time risk monitoring and alerting
- Emergency stop controls
- Portfolio risk aggregation
- State persistence and recovery
- Caching for performance
- Circuit breaker patterns
- Enhanced error handling

CRITICAL: This module integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

# New Architecture - RECOMMENDED
# Support modules
from .base import BaseRiskManager
from .factory import (
    RiskManagementFactory,
    create_recommended_risk_component,
    create_risk_service,
    get_risk_factory,
)
from .portfolio_limits import PortfolioLimits
from .position_sizing import PositionSizer

# Legacy Components - DEPRECATED (maintained for backward compatibility)
from .risk_manager import RiskManager
from .risk_metrics import RiskCalculator
from .service import RiskService

# Legacy circuit breakers and emergency controls (if they exist)
try:
    from .circuit_breakers import BaseCircuitBreaker, CircuitBreakerManager
    from .emergency_controls import EmergencyControls

    _has_legacy_controls = True
except ImportError:
    _has_legacy_controls = False
    BaseCircuitBreaker = None
    CircuitBreakerManager = None
    EmergencyControls = None

# Public API - prioritizes new architecture
__all__ = [
    # Support
    "BaseRiskManager",
    "PortfolioLimits",
    "PositionSizer",
    "RiskCalculator",
    "RiskManagementFactory",
    # Legacy Components (DEPRECATED)
    "RiskManager",
    # New Architecture (RECOMMENDED)
    "RiskService",
    "create_recommended_risk_component",
    "create_risk_service",
    "get_risk_factory",
]

# Add legacy controls if available
if _has_legacy_controls:
    __all__.extend(
        [
            "BaseCircuitBreaker",
            "CircuitBreakerManager",
            "EmergencyControls",
        ]
    )

# Version information
__version__ = "2.0.0"
__author__ = "Trading Bot Framework"
__description__ = "Enterprise Risk Management System with Service Architecture (P-008, P-009)"
