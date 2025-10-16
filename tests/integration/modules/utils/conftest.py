"""
Utils Module Integration Test Configuration.

Provides pytest fixtures for utils integration tests.
Utils tests manage their own DI container lifecycle per test to ensure isolation.
"""

import pytest
import pytest_asyncio

from src.error_handling.decorators import shutdown_all_error_handlers


@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Reset global state before and after each test to prevent cross-test pollution.

    This fixture runs for ALL utils tests automatically and ensures:
    1. Circuit breakers are reset
    2. Global error handler state is clean
    3. No state leakage between tests
    """
    # Reset before test
    try:
        shutdown_all_error_handlers()
    except Exception:
        pass

    yield

    # Reset after test
    try:
        shutdown_all_error_handlers()
    except Exception:
        pass


@pytest_asyncio.fixture
async def isolated_di_container():
    """
    Provide an isolated DI container for tests that need full service registration.

    This fixture is NOT autouse - tests must explicitly request it.
    Most utils tests manage their own minimal DI setup for better isolation.
    """
    from tests.integration.conftest import cleanup_di_container, register_all_services_for_testing

    container = register_all_services_for_testing()
    yield container

    # Proper async cleanup to prevent resource leaks
    await cleanup_di_container(container)
