"""
Workflow Integration Test Configuration.

Provides pytest fixtures for workflow integration tests including bot management,
exchange, and service fixtures.
"""

import pytest_asyncio


@pytest_asyncio.fixture
async def di_container():
    """
    Provide fully configured DI container with all services registered.

    Uses master DI registration to ensure all dependencies are properly configured
    in the correct order without circular dependency issues.
    """
    from tests.integration.conftest import cleanup_di_container, register_all_services_for_testing

    container = register_all_services_for_testing()
    yield container

    # Cleanup to prevent resource leaks
    await cleanup_di_container(container)


@pytest_asyncio.fixture
async def bot_instance_service(di_container):
    """Get real BotInstanceService from DI container."""
    from src.core.config import get_config

    # Create instance manually with required dependencies
    from src.bot_management.instance_service import BotInstanceService

    config = get_config()

    # Get required dependencies from container
    injector = di_container
    container = injector.get_container()

    # Get dependencies
    try:
        from src.capital_management.service import CapitalService

        capital_service = container.get("CapitalService")
        if callable(capital_service):
            capital_service = capital_service()
    except Exception:
        capital_service = None

    service = BotInstanceService(
        config=config, capital_service=capital_service, execution_service=None, risk_service=None
    )
    yield service


@pytest_asyncio.fixture
async def bot_coordination_service(di_container):
    """Get real BotCoordinationService from DI container."""
    from src.bot_management.coordination_service import BotCoordinationService

    service = BotCoordinationService()
    yield service


@pytest_asyncio.fixture
async def bot_lifecycle_service(di_container):
    """Get real BotLifecycleService from DI container."""
    from src.bot_management.lifecycle_service import BotLifecycleService

    service = BotLifecycleService()
    yield service


@pytest_asyncio.fixture
async def bot_monitoring_service(di_container):
    """Get real BotMonitoringService from DI container."""
    from src.bot_management.monitoring_service import BotMonitoringService

    service = BotMonitoringService()
    yield service


@pytest_asyncio.fixture
async def bot_resource_service(di_container):
    """Get real BotResourceService from DI container."""
    from src.core.config import get_config
    from src.bot_management.resource_manager import ResourceManager
    from src.bot_management.resource_service import BotResourceService

    config = get_config()

    # Get capital service from container if available
    injector = di_container
    container = injector.get_container()

    try:
        from src.capital_management.service import CapitalService
        capital_service = container.get("CapitalService")
        if callable(capital_service):
            capital_service = capital_service()
    except Exception:
        capital_service = None

    # Create resource manager
    resource_manager = ResourceManager(config=config, capital_service=capital_service)

    # Create resource service
    service = BotResourceService(resource_manager=resource_manager, name="BotResourceService", config=None)
    yield service


@pytest_asyncio.fixture
async def bot_management_controller(di_container):
    """Get real BotManagementController from DI container."""
    from src.bot_management.controller import BotManagementController

    # Get all required services
    from src.bot_management.coordination_service import BotCoordinationService
    from src.bot_management.instance_service import BotInstanceService
    from src.bot_management.lifecycle_service import BotLifecycleService
    from src.bot_management.monitoring_service import BotMonitoringService
    from src.bot_management.resource_manager import ResourceManager
    from src.bot_management.resource_service import BotResourceService
    from src.core.config import get_config

    config = get_config()

    # Get capital service from container if available
    injector = di_container
    container = injector.get_container()

    try:
        from src.capital_management.service import CapitalService
        capital_service = container.get("CapitalService")
        if callable(capital_service):
            capital_service = capital_service()
    except Exception:
        capital_service = None

    # Create resource manager
    resource_manager = ResourceManager(config=config, capital_service=capital_service)

    # Create service instances
    instance_service = BotInstanceService(config=config, capital_service=None)
    coordination_service = BotCoordinationService()
    lifecycle_service = BotLifecycleService()
    monitoring_service = BotMonitoringService()
    resource_service = BotResourceService(resource_manager=resource_manager, name="BotResourceService", config=None)

    controller = BotManagementController(
        bot_instance_service=instance_service,
        bot_coordination_service=coordination_service,
        bot_lifecycle_service=lifecycle_service,
        bot_monitoring_service=monitoring_service,
        resource_management_service=resource_service,
    )
    yield controller


@pytest_asyncio.fixture
async def mock_exchange():
    """Provide a mock exchange for testing."""
    from src.exchanges.mock_exchange import MockExchange

    exchange = MockExchange(name="mock_binance")
    await exchange.connect()
    yield exchange
    await exchange.disconnect()
