"""
Capital Management Module Integration Test Configuration.

Provides pytest fixtures for capital management integration tests with full DI support.
NO MOCKS - Uses real services from DI container with real database.
"""

from decimal import Decimal

import pytest
import pytest_asyncio

from src.core.types import AllocationStrategy


@pytest_asyncio.fixture
async def di_container():
    """
    Provide fully configured DI container with all services registered.

    Uses master DI registration to ensure all dependencies are properly configured
    in the correct order without circular dependency issues.
    """
    from tests.integration.conftest import cleanup_di_container, register_all_services_for_testing

    # Register all services
    container = register_all_services_for_testing()

    yield container

    # Cleanup to prevent resource leaks
    await cleanup_di_container(container)


@pytest.fixture
def test_capital_config():
    """Create test configuration for capital management."""
    from src.core.config import Config

    config = Config()
    config.capital_management.allocation_strategy = AllocationStrategy.EQUAL_WEIGHT
    config.capital_management.total_capital = Decimal("100000.00")
    config.capital_management.emergency_reserve_pct = Decimal("0.10")
    config.capital_management.max_allocation_pct = Decimal("0.40")
    config.capital_management.min_allocation_pct = Decimal("0.05")
    return config


@pytest_asyncio.fixture
async def capital_service(di_container):
    """Get real CapitalService from DI container."""
    container = di_container.get_container()
    service = container.get("CapitalService")
    await service.start()
    yield service
    await service.stop()


@pytest_asyncio.fixture
async def capital_allocator(di_container):
    """Get real CapitalAllocator from DI container."""
    container = di_container.get_container()
    allocator = container.get("CapitalAllocator")
    yield allocator


@pytest_asyncio.fixture
async def currency_manager(di_container):
    """Get real CurrencyManager from DI container."""
    container = di_container.get_container()
    manager = container.get("CurrencyManager")
    await manager.start()
    yield manager
    await manager.stop()


@pytest_asyncio.fixture
async def exchange_distributor(di_container):
    """Get real ExchangeDistributor from DI container."""
    container = di_container.get_container()
    distributor = container.get("ExchangeDistributor")
    await distributor.start()
    yield distributor
    await distributor.stop()


@pytest_asyncio.fixture
async def fund_flow_manager(di_container):
    """Get real FundFlowManager from DI container."""
    container = di_container.get_container()
    manager = container.get("FundFlowManager")
    await manager.start()
    yield manager
    await manager.stop()
