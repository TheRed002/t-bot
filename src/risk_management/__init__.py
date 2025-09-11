"""
Risk Management Module - Enterprise-Grade Risk Management System.

This module provides comprehensive risk management functionality including:

## Core Architecture:
- RiskService: Enterprise-grade risk management service with full integration
- RiskManagementFactory: Factory for creating and managing risk components
- DatabaseService integration: No direct database access
- StateService integration: Centralized state management
- Real-time monitoring and alerting
- Comprehensive caching and circuit breakers

## Available Components:
- RiskManager: Full-featured risk manager with service integration
- PositionSizer: Advanced position sizing with multiple strategies
- RiskCalculator: Comprehensive risk metrics calculation

## Quick Start:

### Using RiskManagementController:
```python
from src.risk_management.factory import create_risk_management_controller

controller = create_risk_management_controller(
    database_service=database_service, state_service=state_service, config=config
)

# Main operations
position_size = await controller.calculate_position_size(signal, capital, price)
is_valid = await controller.validate_signal(signal)
risk_metrics = await controller.calculate_risk_metrics(positions, market_data)
```

### Using RiskService directly:
```python
from src.risk_management.factory import create_risk_service

risk_service = create_risk_service(
    database_service=database_service, state_service=state_service, config=config
)

position_size = await risk_service.calculate_position_size(signal, capital, price)
risk_metrics = await risk_service.calculate_risk_metrics(positions, market_data)
```

## Features:
- Multiple position sizing methods (Fixed %, Kelly Criterion, Volatility-Adjusted,
  Confidence-Weighted)
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

# Core Architecture
from .base import BaseRiskManager
from .controller import RiskManagementController

# Dependency injection registration
from .di_registration import (
    configure_risk_management_dependencies,
    register_risk_management_services,
)
from .factory import (
    RiskManagementFactory,
    create_recommended_risk_component,
    create_risk_management_controller,
    create_risk_service,
    get_risk_factory,
)
from .interfaces import (
    AbstractRiskService,
    PositionSizingServiceInterface,
    RiskMetricsServiceInterface,
    RiskMonitoringServiceInterface,
    RiskServiceInterface,
    RiskValidationServiceInterface,
)
from .portfolio_limits import PortfolioLimits
from .position_sizing import PositionSizer

# Risk Management Components
from .risk_manager import RiskManager
from .risk_metrics import RiskCalculator
from .service import RiskService
from .services import (
    PositionSizingService,
    RiskMetricsService,
    RiskMonitoringService,
    RiskValidationService,
)

# Circuit breakers and emergency controls
try:
    from .circuit_breakers import BaseCircuitBreaker, CircuitBreakerManager
    from .emergency_controls import EmergencyControls

    _has_legacy_controls = True
except ImportError:
    _has_legacy_controls = False
    BaseCircuitBreaker = None  # type: ignore
    CircuitBreakerManager = None  # type: ignore
    EmergencyControls = None  # type: ignore

# Public API
__all__ = [
    "AbstractRiskService",
    # Core components
    "BaseRiskManager",
    "PortfolioLimits",
    "PositionSizer",
    "PositionSizingService",
    "PositionSizingServiceInterface",
    "RiskCalculator",
    # Risk Management Architecture
    "RiskManagementController",
    "RiskManagementFactory",
    "RiskService",
    "RiskServiceInterface",
    # Risk Management Components
    "RiskManager",
    "RiskMetricsService",
    "RiskMetricsServiceInterface",
    "RiskMonitoringService",
    "RiskMonitoringServiceInterface",
    "RiskValidationService",
    "RiskValidationServiceInterface",
    # Dependency injection
    "configure_risk_management_dependencies",
    "create_recommended_risk_component",
    "create_risk_management_controller",
    "create_risk_service",
    "get_risk_factory",
    "register_risk_management_services",
]

# Add circuit breakers if available
if _has_legacy_controls:
    __all__.extend(
        [
            "BaseCircuitBreaker",
            "CircuitBreakerManager",
            "EmergencyControls",
        ]
    )

# Version information
__version__ = "2.1.0"
__author__ = "Trading Bot Framework"
__description__ = (
    "Enterprise Risk Management System with Controller-Service-Repository Architecture"
)
