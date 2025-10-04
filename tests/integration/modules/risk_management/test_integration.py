"""
Integration tests for Risk Management Framework.

REFACTORED: This file now imports from test_integration_real.py which uses
REAL services instead of mocks. The old mock-based version has been moved to
test_integration.py.backup for reference.

Tests circuit breakers, position limits, emergency controls, portfolio risk metrics,
correlation monitoring, and adaptive risk management with REAL database persistence.
"""

# Import all test classes from the real implementation
from .test_integration_real import (
    TestRiskManagementRealIntegration,
    TestRiskManagementDatabasePersistence,
)

# For backward compatibility, create aliases
TestRiskManagementIntegration = TestRiskManagementRealIntegration

__all__ = [
    'TestRiskManagementRealIntegration',
    'TestRiskManagementDatabasePersistence',
    'TestRiskManagementIntegration',  # Alias for backward compatibility
]
