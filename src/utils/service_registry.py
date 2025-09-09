"""Service registry for utils module dependency injection.

This module centralizes the registration of all utility services in the
dependency injection container following the established patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.dependency_injection import injector
from src.core.exceptions import ServiceError

if TYPE_CHECKING:
    from src.utils.interfaces import (
        DataFlowInterface,
        GPUInterface,
        PrecisionInterface,
        ValidationServiceInterface,
    )

_services_registered = False


def register_util_services() -> None:
    """Register all utility services in the dependency injection container.

    This function should be called during application startup to ensure
    all utility services are properly registered with correct lifetimes.
    Idempotent - safe to call multiple times.
    """
    global _services_registered

    # Guard against multiple registrations
    if _services_registered:
        return

    # Register GPU utilities as singleton using dependency injection
    def gpu_manager_factory() -> GPUInterface:
        """Create GPU manager using dependency injection."""
        try:
            from src.utils.gpu_utils import GPUManager

            return GPUManager()
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"GPU manager creation failed: {e}")
            raise ServiceError(f"Failed to create GPU manager: {e}") from e

    injector.register_factory("GPUManager", gpu_manager_factory, singleton=True)
    injector.register_factory("GPUInterface", gpu_manager_factory, singleton=True)

    # Register data flow integrity services using dependency injection pattern
    def precision_tracker_factory() -> PrecisionInterface:
        """Create precision tracker using dependency injection.

        This factory creates a standalone PrecisionTracker for service registration.
        No circular dependencies since this is a self-contained utility.
        """
        try:
            from src.utils.data_flow_integrity import PrecisionTracker

            return PrecisionTracker()
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"PrecisionTracker creation failed: {e}")
            raise ServiceError(f"Failed to create precision tracker: {e}") from e

    def data_flow_validator_factory() -> DataFlowInterface:
        """Create data flow validator using dependency injection."""
        try:
            from src.utils.data_flow_integrity import DataFlowValidator

            return DataFlowValidator()
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"DataFlowValidator creation failed: {e}")
            raise ServiceError(f"Failed to create data flow validator: {e}") from e

    def integrity_converter_factory():
        """Create integrity preserving converter using dependency injection."""
        try:
            from src.utils.data_flow_integrity import IntegrityPreservingConverter

            precision_tracker = injector.resolve("PrecisionTracker")
            return IntegrityPreservingConverter(
                track_precision=True, precision_tracker=precision_tracker
            )
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"IntegrityPreservingConverter creation failed: {e}")
            raise ServiceError(f"Failed to create integrity converter: {e}") from e

    # Register with both concrete and interface names
    injector.register_factory("PrecisionTracker", precision_tracker_factory, singleton=True)
    injector.register_factory("PrecisionInterface", precision_tracker_factory, singleton=True)
    injector.register_factory("DataFlowValidator", data_flow_validator_factory, singleton=True)
    injector.register_factory("DataFlowInterface", data_flow_validator_factory, singleton=True)
    injector.register_factory(
        "IntegrityPreservingConverter", integrity_converter_factory, singleton=True
    )

    # Register validation services using dependency injection pattern
    def validation_framework_factory():
        """Create validation framework using dependency injection."""
        try:
            from src.utils.validation.core import ValidationFramework

            return ValidationFramework()
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"ValidationFramework creation failed: {e}")
            raise ServiceError(f"Failed to create validation framework: {e}") from e

    injector.register_factory("ValidationFramework", validation_framework_factory, singleton=True)

    # Simplified ValidationService factory with proper dependency injection
    def validation_service_factory() -> ValidationServiceInterface:
        """Create validation service using dependency injection."""
        try:
            from src.utils.validation.service import ValidationService

            validation_framework = injector.resolve("ValidationFramework")
            return ValidationService(validation_framework=validation_framework)
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"ValidationService creation failed: {e}")
            raise ServiceError(f"Failed to create validation service: {e}") from e

    # Register as singleton since validation service manages internal state
    injector.register_factory("ValidationService", validation_service_factory, singleton=True)
    injector.register_factory(
        "ValidationServiceInterface", validation_service_factory, singleton=True
    )

    # Register financial calculator using dependency injection pattern
    from src.utils.interfaces import CalculatorInterface

    def financial_calculator_factory() -> CalculatorInterface:
        """Create financial calculator using dependency injection."""
        try:
            from src.utils.calculations.financial import FinancialCalculator

            return FinancialCalculator()
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"FinancialCalculator creation failed: {e}")
            raise ServiceError(f"Failed to create financial calculator: {e}") from e

    injector.register_factory("FinancialCalculator", financial_calculator_factory, singleton=True)
    injector.register_factory("CalculatorInterface", financial_calculator_factory, singleton=True)

    # Register HTTP session manager using dependency injection pattern
    def http_session_manager_factory():
        """Create HTTP session manager using dependency injection."""
        try:
            from src.utils.monitoring_helpers import HTTPSessionManager

            return HTTPSessionManager()
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"HTTPSessionManager creation failed: {e}")
            raise ServiceError(f"Failed to create HTTP session manager: {e}") from e

    injector.register_factory("HTTPSessionManager", http_session_manager_factory, singleton=True)

    # Register messaging services using dependency injection pattern
    def data_transformation_handler_factory():
        """Create data transformation handler using dependency injection."""
        try:
            from src.utils.messaging_patterns import DataTransformationHandler

            return DataTransformationHandler()
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"DataTransformationHandler creation failed: {e}")
            raise ServiceError(f"Failed to create data transformation handler: {e}") from e

    def messaging_coordinator_factory():
        """Create messaging coordinator using dependency injection."""
        try:
            # Inject event emitter for proper dependency injection
            from src.core.base.events import BaseEventEmitter
            from src.utils.messaging_patterns import MessagePattern, MessagingCoordinator

            event_emitter = BaseEventEmitter(name="MessagingCoordinator_EventEmitter")
            coordinator = MessagingCoordinator(event_emitter=event_emitter)

            # Register standard data transformation handler
            transform_handler = injector.resolve("DataTransformationHandler")
            coordinator.register_handler(MessagePattern.PUB_SUB, transform_handler)
            coordinator.register_handler(MessagePattern.REQ_REPLY, transform_handler)
            coordinator.register_handler(MessagePattern.STREAM, transform_handler)
            coordinator.register_handler(MessagePattern.BATCH, transform_handler)

            return coordinator
        except Exception as e:
            from src.core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"MessagingCoordinator creation failed: {e}")
            raise ServiceError(f"Failed to create messaging coordinator: {e}") from e

    injector.register_factory(
        "DataTransformationHandler", data_transformation_handler_factory, singleton=True
    )
    injector.register_factory("MessagingCoordinator", messaging_coordinator_factory, singleton=True)

    # Mark services as registered
    _services_registered = True


# Services are registered lazily to avoid circular dependencies
# Call register_util_services() explicitly when needed
