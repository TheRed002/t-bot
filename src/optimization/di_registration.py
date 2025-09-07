"""
Dependency injection registration for optimization module.

This module handles registration of optimization components with the
dependency injection container following established patterns.
"""

from typing import Any

from src.core.dependency_injection import DependencyInjector
from src.optimization.factory import OptimizationFactory
from src.optimization.interfaces import (
    IOptimizationService,
)


def register_optimization_dependencies(injector: DependencyInjector) -> None:
    """
    Register optimization dependencies with the DI container.

    Args:
        injector: Dependency injector instance
    """
    # Register factory as singleton
    def factory_factory():
        return OptimizationFactory(injector)

    injector.register_factory("OptimizationFactory", factory_factory, singleton=True)

    # Register repository using factory
    def repository_factory():
        factory = injector.resolve("OptimizationFactory")
        return factory.create("repository")

    injector.register_factory("OptimizationRepository", repository_factory, singleton=False)

    # Register backtesting integration using factory
    def backtest_integration_factory():
        factory = injector.resolve("OptimizationFactory")
        return factory.create("backtest_integration")

    injector.register_factory("BacktestIntegration", backtest_integration_factory, singleton=False)

    # Register optimization service using factory
    def service_factory():
        factory = injector.resolve("OptimizationFactory")
        return factory.create("service")

    injector.register_factory("OptimizationService", service_factory, singleton=True)

    # Register controller using factory
    def controller_factory():
        factory = injector.resolve("OptimizationFactory")
        return factory.create("controller")

    injector.register_factory("OptimizationController", controller_factory, singleton=True)

    # Register interface implementations for proper service layer architecture
    injector.register_service(
        "IOptimizationService",
        lambda: injector.resolve("OptimizationService"),
        singleton=True
    )
    injector.register_service(
        "BacktestIntegrationProtocol",
        lambda: injector.resolve("BacktestIntegration"),
        singleton=False
    )
    injector.register_service(
        "OptimizationRepositoryProtocol",
        lambda: injector.resolve("OptimizationRepository"),
        singleton=False
    )


def configure_optimization_module(
    injector: DependencyInjector,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Configure optimization module with custom settings.

    Args:
        injector: Dependency injector instance
        config: Optional configuration overrides
    """
    # Register base dependencies
    register_optimization_dependencies(injector)

    # Apply any custom configuration
    if config:
        # Handle any module-specific configuration
        pass


def get_optimization_service(injector: DependencyInjector) -> IOptimizationService:
    """Get OptimizationService from DI container using service locator pattern."""
    return injector.resolve("OptimizationService")


def get_optimization_controller(injector: DependencyInjector):
    """Get OptimizationController from DI container using service locator pattern."""
    return injector.resolve("OptimizationController")


def get_optimization_repository(injector: DependencyInjector):
    """Get OptimizationRepository from DI container using service locator pattern."""
    return injector.resolve("OptimizationRepository")
