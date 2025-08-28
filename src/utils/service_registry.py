"""Service registry for utils module dependency injection.

This module centralizes the registration of all utility services in the
dependency injection container following the established patterns.
"""

from __future__ import annotations

from src.core.dependency_injection import injector


def register_util_services() -> None:
    """Register all utility services in the dependency injection container.

    This function should be called during application startup to ensure
    all utility services are properly registered with correct lifetimes.
    """
    # Register GPU utilities
    from .gpu_utils import GPUManager

    injector.register_service("GPUManager", GPUManager(), singleton=True)

    # Register data flow integrity services
    from .data_flow_integrity import (
        DataFlowValidator,
        IntegrityPreservingConverter,
        PrecisionTracker,
    )

    injector.register_service("PrecisionTracker", PrecisionTracker(), singleton=True)
    injector.register_service("DataFlowValidator", DataFlowValidator(), singleton=True)
    injector.register_service(
        "IntegrityPreservingConverter", IntegrityPreservingConverter(), singleton=True
    )

    # Register validation services
    from .validation.core import ValidationFramework
    from .validation.service import ValidationService

    injector.register_service("ValidationFramework", ValidationFramework(), singleton=True)

    # Factory for ValidationService to ensure proper initialization
    def validation_service_factory() -> ValidationService:
        return ValidationService()

    injector.register_factory("ValidationService", validation_service_factory, singleton=True)

    # Register financial calculator
    from .calculations.financial import FinancialCalculator

    injector.register_service("FinancialCalculator", FinancialCalculator(), singleton=True)

    # Register interface implementations
    # ValidationService implements ValidationServiceInterface
    def validation_service_interface_factory():
        return injector.resolve("ValidationService")

    injector.register_factory(
        "ValidationServiceInterface", validation_service_interface_factory, singleton=True
    )


# Services are registered lazily to avoid circular dependencies
# Call register_util_services() explicitly when needed
