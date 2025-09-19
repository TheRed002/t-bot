"""
Shared initialization helpers for backtesting module.

This module contains common initialization patterns used across backtesting components
to eliminate duplication and ensure consistency.
"""

from typing import Any, Dict, Optional

from src.core.logging import get_logger


def log_component_initialization(
    component_name: str,
    logger_instance: Optional[Any] = None,
    **additional_info: Any
) -> None:
    """
    Log component initialization in a standardized format.

    Args:
        component_name: Name of the component being initialized
        logger_instance: Optional logger instance, creates new one if None
        **additional_info: Additional information to log
    """
    if logger_instance is None:
        logger_instance = get_logger(__name__)

    logger_instance.info(f"{component_name} initialized", **additional_info)


def log_service_initialization(
    service_name: str,
    available_services: Optional[Dict[str, Any]] = None,
    logger_instance: Optional[Any] = None,
    **additional_info: Any
) -> None:
    """
    Log service initialization with available services.

    Args:
        service_name: Name of the service being initialized
        available_services: Dictionary of available services
        logger_instance: Optional logger instance, creates new one if None
        **additional_info: Additional information to log
    """
    if logger_instance is None:
        logger_instance = get_logger(__name__)

    log_data = {}
    if available_services:
        log_data["available_services"] = available_services
    log_data.update(additional_info)

    logger_instance.info(f"{service_name} initialized with services", **log_data)


def log_factory_initialization(
    factory_name: str,
    injection_enabled: bool = False,
    logger_instance: Optional[Any] = None,
    **additional_info: Any
) -> None:
    """
    Log factory initialization with dependency injection status.

    Args:
        factory_name: Name of the factory being initialized
        injection_enabled: Whether dependency injection is enabled
        logger_instance: Optional logger instance, creates new one if None
        **additional_info: Additional information to log
    """
    if logger_instance is None:
        logger_instance = get_logger(__name__)

    injection_info = "with dependency injection" if injection_enabled else ""
    log_message = f"{factory_name} initialized {injection_info}".strip()

    logger_instance.info(log_message, **additional_info)


def log_configuration_initialization(
    component_name: str,
    config: Any,
    logger_instance: Optional[Any] = None,
    **additional_info: Any
) -> None:
    """
    Log component initialization with configuration details.

    Args:
        component_name: Name of the component being initialized
        config: Configuration object (should have model_dump or similar method)
        logger_instance: Optional logger instance, creates new one if None
        **additional_info: Additional information to log
    """
    if logger_instance is None:
        logger_instance = get_logger(__name__)

    log_data = {}

    # Try to extract config data
    if hasattr(config, 'model_dump'):
        log_data["config"] = config.model_dump()
    elif hasattr(config, 'dict'):
        log_data["config"] = config.dict()
    elif hasattr(config, '__dict__'):
        log_data["config"] = config.__dict__
    else:
        log_data["config"] = str(config)

    log_data.update(additional_info)

    logger_instance.info(f"{component_name} initialized", **log_data)


def create_initialization_summary(components: list[str]) -> Dict[str, Any]:
    """
    Create a summary of initialized components.

    Args:
        components: List of component names that were initialized

    Returns:
        Dictionary containing initialization summary
    """
    return {
        "initialized_components": components,
        "total_components": len(components),
        "status": "initialization_complete"
    }