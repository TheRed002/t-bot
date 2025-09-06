"""
Capital Management Dependency Injection Registration
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def register_capital_management_services(container: Any) -> None:
    """
    Register capital management services with the DI container using factory pattern.

    Args:
        container: The dependency injection container
    """
    # Register repository adapters first
    _register_capital_repositories(container)

    # Create the main factory with dependency injection support
    def create_capital_management_factory():
        # Import at runtime to avoid circular imports
        from src.capital_management.factory import CapitalManagementFactory

        return CapitalManagementFactory(dependency_container=container)

    # Register the main factory
    container.register(
        "CapitalManagementFactory", create_capital_management_factory, singleton=True
    )

    # Register individual factories through the main factory
    def get_capital_management_factory():
        return container.get("CapitalManagementFactory")

    # Register service creation functions that use the factory
    def create_capital_service():
        factory = get_capital_management_factory()
        return factory.create_capital_service()

    def create_capital_allocator():
        factory = get_capital_management_factory()
        return factory.create_capital_allocator()

    def create_currency_manager():
        factory = get_capital_management_factory()
        return factory.create_currency_manager()

    def create_exchange_distributor():
        factory = get_capital_management_factory()
        return factory.create_exchange_distributor()

    def create_fund_flow_manager():
        factory = get_capital_management_factory()
        return factory.create_fund_flow_manager()

    # Register concrete services
    container.register("CapitalService", create_capital_service, singleton=True)
    container.register("CapitalAllocator", create_capital_allocator, singleton=True)
    container.register("CurrencyManager", create_currency_manager, singleton=True)
    container.register("ExchangeDistributor", create_exchange_distributor, singleton=True)
    container.register("FundFlowManager", create_fund_flow_manager, singleton=True)

    # Register interface mappings
    container.register(
        "AbstractCapitalService", lambda: container.get("CapitalService"), singleton=True
    )
    container.register(
        "CapitalServiceProtocol", lambda: container.get("CapitalService"), singleton=True
    )

    container.register(
        "AbstractCurrencyManagementService",
        lambda: container.get("CurrencyManager"),
        singleton=True,
    )
    container.register(
        "CurrencyManagementServiceProtocol",
        lambda: container.get("CurrencyManager"),
        singleton=True,
    )

    container.register(
        "AbstractExchangeDistributionService",
        lambda: container.get("ExchangeDistributor"),
        singleton=True,
    )
    container.register(
        "ExchangeDistributionServiceProtocol",
        lambda: container.get("ExchangeDistributor"),
        singleton=True,
    )

    container.register(
        "AbstractFundFlowManagementService",
        lambda: container.get("FundFlowManager"),
        singleton=True,
    )
    container.register(
        "FundFlowManagementServiceProtocol",
        lambda: container.get("FundFlowManager"),
        singleton=True,
    )

    # Register individual sub-factories for advanced usage
    def get_capital_service_factory():
        return get_capital_management_factory().capital_service_factory

    def get_capital_allocator_factory():
        return get_capital_management_factory().capital_allocator_factory

    def get_currency_manager_factory():
        return get_capital_management_factory().currency_manager_factory

    def get_exchange_distributor_factory():
        return get_capital_management_factory().exchange_distributor_factory

    def get_fund_flow_manager_factory():
        return get_capital_management_factory().fund_flow_manager_factory

    container.register("CapitalServiceFactory", get_capital_service_factory, singleton=True)
    container.register("CapitalAllocatorFactory", get_capital_allocator_factory, singleton=True)
    container.register("CurrencyManagerFactory", get_currency_manager_factory, singleton=True)
    container.register(
        "ExchangeDistributorFactory", get_exchange_distributor_factory, singleton=True
    )
    container.register("FundFlowManagerFactory", get_fund_flow_manager_factory, singleton=True)


def _register_capital_repositories(container: Any) -> None:
    """
    Register capital management repository adapters.

    Args:
        container: The dependency injection container
    """

    # Register CapitalRepository adapter
    def create_capital_repository():
        # Import at runtime to avoid circular imports
        from src.capital_management.repository import CapitalRepository
        from src.database.repository.capital import CapitalAllocationRepository

        # Get database session from container
        try:
            session_factory = container.get("AsyncSessionFactory")
            session = session_factory()
            capital_allocation_repo = CapitalAllocationRepository(session)
            return CapitalRepository(capital_allocation_repo)
        except Exception as e:
            # Fallback for testing or when database is not available
            import logging

            logging.getLogger(__name__).warning(f"Failed to create CapitalRepository: {e}")
            return None

    # Register AuditRepository adapter
    def create_audit_repository():
        # Import at runtime to avoid circular imports
        from src.capital_management.repository import AuditRepository
        from src.database.repository.audit import CapitalAuditLogRepository

        # Get database session from container
        try:
            session_factory = container.get("AsyncSessionFactory")
            session = session_factory()
            audit_repo = CapitalAuditLogRepository(session)
            return AuditRepository(audit_repo)
        except Exception as e:
            # Fallback for testing or when database is not available
            import logging

            logging.getLogger(__name__).warning(f"Failed to create AuditRepository: {e}")
            return None

    # Register the repository adapters
    container.register("CapitalRepository", create_capital_repository, singleton=True)
    container.register("AuditRepository", create_audit_repository, singleton=True)
