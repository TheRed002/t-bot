"""
Capital Management Dependency Injection Registration
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.capital_management.repository import AuditRepository, CapitalRepository
    from src.database.repository.audit import CapitalAuditLogRepository
    from src.database.repository.capital import CapitalAllocationRepository

# Module logger
logger = logging.getLogger(__name__)

# Import at module level for test patching compatibility
try:
    from src.capital_management.repository import AuditRepository, CapitalRepository
    from src.database.repository.audit import CapitalAuditLogRepository
    from src.database.repository.capital import CapitalAllocationRepository
except ImportError:
    # Fallback for testing or missing dependencies
    CapitalRepository = None  # type: ignore
    AuditRepository = None  # type: ignore
    CapitalAllocationRepository = None  # type: ignore
    CapitalAuditLogRepository = None  # type: ignore


def register_capital_management_services(container: Any) -> None:
    """
    Register capital management services with the DI container using factory pattern.

    Args:
        container: The dependency injection container
    """
    _register_capital_repositories(container)

    def create_capital_management_factory():
        from src.capital_management.factory import CapitalManagementFactory

        return CapitalManagementFactory(dependency_container=container)

    container.register(
        "CapitalManagementFactory", create_capital_management_factory, singleton=True
    )

    def create_capital_service_factory():
        from src.capital_management.factory import CapitalServiceFactory

        return CapitalServiceFactory(dependency_container=container)

    container.register("CapitalServiceFactory", create_capital_service_factory, singleton=True)

    def get_capital_management_factory():
        return container.get("CapitalManagementFactory")

    def create_capital_service():
        factory = get_capital_management_factory()
        capital_repository = None
        audit_repository = None

        try:
            capital_repository = container.get("CapitalRepository")
        except (KeyError, AttributeError, ImportError):
            pass
        try:
            audit_repository = container.get("AuditRepository")
        except (KeyError, AttributeError, ImportError):
            pass

        return factory.create_capital_service(
            capital_repository=capital_repository, audit_repository=audit_repository
        )

    def create_capital_allocator():
        factory = get_capital_management_factory()
        # Pass container to factory - it will resolve dependencies internally
        return factory.create_capital_allocator()

    def create_currency_manager():
        factory = get_capital_management_factory()
        # Pass container to factory - it will resolve dependencies internally
        return factory.create_currency_manager()

    def create_exchange_distributor():
        factory = get_capital_management_factory()
        # Pass container to factory - it will resolve dependencies internally
        return factory.create_exchange_distributor()

    def create_fund_flow_manager():
        factory = get_capital_management_factory()
        # Pass container to factory - it will resolve dependencies internally
        return factory.create_fund_flow_manager()

    container.register("CapitalService", create_capital_service, singleton=True)
    container.register("CapitalAllocator", create_capital_allocator, singleton=True)
    container.register("CurrencyManager", create_currency_manager, singleton=True)
    container.register("ExchangeDistributor", create_exchange_distributor, singleton=True)
    container.register("FundFlowManager", create_fund_flow_manager, singleton=True)

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

    # Register factory instances using service locator pattern
    get_capital_management_factory().register_factories(container)

    _register_fallback_services(container)
    _setup_cross_dependencies(container)


def _register_fallback_services(container: Any) -> None:
    """Register fallback services using service locator pattern."""
    if not _has_service(container, "ConfigService"):
        container.register("ConfigService", lambda: {"capital_config": {}}, singleton=True)

    if not _has_service(container, "AsyncSessionFactory"):
        container.register("AsyncSessionFactory", lambda: lambda: None, singleton=True)
    if not _has_service(container, "ValidationService") and not _has_service(
        container, "ValidationServiceInterface"
    ):

        class MinimalValidationService:
            def validate(self, data):
                return True

        minimal_validation = MinimalValidationService()
        container.register("ValidationService", lambda: minimal_validation, singleton=True)
        container.register("ValidationServiceInterface", lambda: minimal_validation, singleton=True)


def _setup_cross_dependencies(container: Any) -> None:
    """Set up cross-dependencies between services after registration."""
    try:
        fund_flow_manager = container.get("FundFlowManager")
        capital_allocator = container.get("CapitalAllocator")

        if hasattr(fund_flow_manager, "set_capital_allocator"):
            fund_flow_manager.set_capital_allocator(capital_allocator)
    except (KeyError, AttributeError, ImportError):
        pass


def _has_service(container: Any, service_name: str) -> bool:
    """Check if service is available in container."""
    try:
        container.get(service_name)
        return True
    except (KeyError, AttributeError, ImportError):
        return False


def _register_capital_repositories(container: Any) -> None:
    """
    Register capital management repository adapters.

    Args:
        container: The dependency injection container
    """

    # Register repositories using service locator pattern
    def create_capital_repository():
        return _create_repository_with_fallback(
            container,
            "CapitalRepository",
            "CapitalAllocationRepository",
            _create_minimal_capital_repository,
        )

    def create_audit_repository():
        return _create_repository_with_fallback(
            container,
            "AuditRepository",
            "CapitalAuditLogRepository",
            _create_minimal_audit_repository,
        )

    container.register("CapitalRepository", create_capital_repository, singleton=True)
    container.register("AuditRepository", create_audit_repository, singleton=True)


def _create_repository_with_fallback(
    container: Any, repo_class_name: str, db_repo_class_name: str, fallback_creator
):
    """Create repository with fallback using service locator pattern."""
    try:
        # Try to get AsyncSession directly (it's registered as a factory in DI)
        session = container.get("AsyncSession")
        if session:
            # AsyncSession is already a session instance from the factory
            if (
                repo_class_name == "CapitalRepository"
                and CapitalRepository
                and CapitalAllocationRepository
            ):
                db_repo = CapitalAllocationRepository(session)
                return CapitalRepository(db_repo)
            elif (
                repo_class_name == "AuditRepository"
                and AuditRepository
                and CapitalAuditLogRepository
            ):
                audit_db_repo = CapitalAuditLogRepository(session)
                return AuditRepository(audit_db_repo)
        else:
            logger.warning(
                f"AsyncSession not available for {repo_class_name}, using fallback"
            )
    except (KeyError, AttributeError, ImportError, Exception) as e:
        logger.warning(
            f"Failed to create {repo_class_name} with session factory: {e}, using fallback"
        )

    return fallback_creator()


def _create_minimal_capital_repository():
    """Create minimal capital repository for testing."""

    class MinimalCapitalRepository:
        async def create(self, allocation_data):
            return {"id": allocation_data.get("id", "test")}

        async def update(self, allocation_data):
            return {"id": allocation_data.get("id", "test")}

        async def delete(self, allocation_id):
            return True

        async def get_by_strategy_exchange(self, strategy_id, exchange):
            return None

        async def get_by_strategy(self, strategy_id):
            return []

        async def get_all(self, limit=None):
            return []

    return MinimalCapitalRepository()


def _create_minimal_audit_repository():
    """Create minimal audit repository for testing."""

    class MinimalAuditRepository:
        async def create(self, audit_data):
            return {"id": "test_audit"}

    return MinimalAuditRepository()
