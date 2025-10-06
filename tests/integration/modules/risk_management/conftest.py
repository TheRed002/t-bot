"""Pytest configuration for risk management integration tests."""

import pytest_asyncio

# Import and expose all fixtures from the fixtures module
from .fixtures.real_service_fixtures import (
    generate_bear_market_scenario,
    generate_bull_market_scenario,
    generate_high_volatility_scenario,
    generate_realistic_market_data_sequence,
    minimal_state_service,
    real_risk_factory,
    real_risk_service,
    sample_positions,
    sample_signal,
)

__all__ = [
    "generate_bear_market_scenario",
    "generate_bull_market_scenario",
    "generate_high_volatility_scenario",
    "generate_realistic_market_data_sequence",
    "minimal_state_service",
    "real_risk_factory",
    "real_risk_service",
    "sample_positions",
    "sample_signal",
]


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
