"""Pytest configuration for risk management integration tests."""

# Import and expose all fixtures from the fixtures module
from .fixtures.real_service_fixtures import (
    real_risk_service,
    real_risk_factory,
    minimal_state_service,
    generate_realistic_market_data_sequence,
    generate_bull_market_scenario,
    generate_bear_market_scenario,
    generate_high_volatility_scenario,
    sample_positions,
    sample_signal,
)

__all__ = [
    'real_risk_service',
    'real_risk_factory',
    'minimal_state_service',
    'generate_realistic_market_data_sequence',
    'generate_bull_market_scenario',
    'generate_bear_market_scenario',
    'generate_high_volatility_scenario',
    'sample_positions',
    'sample_signal',
]
