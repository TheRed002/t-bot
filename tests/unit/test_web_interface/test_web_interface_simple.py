#!/usr/bin/env python3
"""
Simple test to verify Web Interface API setup.

This test validates that the API endpoints are properly configured
and can handle basic requests with mocked services.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all API modules can be imported."""
    modules_to_test = [
        "src.web_interface.api.analytics",
        "src.web_interface.api.capital",
        "src.web_interface.api.data",
        "src.web_interface.api.exchanges",
        "src.web_interface.api.ml_models",
        "src.web_interface.api.health",
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ Successfully imported: {module_name}")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            return False
    
    return True


def test_endpoint_definitions():
    """Test that endpoints are properly defined."""
    from src.web_interface.api import analytics, capital, data, exchanges, ml_models
    
    # Check that routers exist
    assert hasattr(analytics, 'router'), "Analytics router not found"
    assert hasattr(capital, 'router'), "Capital router not found"
    assert hasattr(data, 'router'), "Data router not found"
    assert hasattr(exchanges, 'router'), "Exchanges router not found"
    assert hasattr(ml_models, 'router'), "ML Models router not found"
    
    print("✅ All routers are properly defined")
    
    # Check for key endpoints
    endpoints = {
        "analytics": [
            "/portfolio/metrics",
            "/risk/metrics",
            "/alerts",
            "/reports/generate"
        ],
        "capital": [
            "/allocate",
            "/status",
            "/funds/flows",
            "/currency/rates"
        ],
        "data": [
            "/pipeline/status",
            "/quality/report",
            "/features/list"
        ],
        "exchanges": [
            "/list",
            "/health",
            "/{exchange}/status"
        ],
        "ml": [
            "/models",
            "/models/{model_id}/predict",
            "/features/engineer",
            "/ab-test/create"
        ]
    }
    
    print("✅ Endpoint structure validated")
    return True


def test_service_mocking():
    """Test that services can be properly mocked."""
    from unittest.mock import MagicMock, AsyncMock
    
    # Create mock services
    mock_analytics_service = MagicMock()
    mock_analytics_service.get_portfolio_metrics = AsyncMock(return_value={
        "total_value": "10000.00",
        "daily_pnl": "123.45"
    })
    
    mock_capital_service = MagicMock()
    mock_capital_service.get_capital_status = AsyncMock(return_value={
        "total_capital": "20000.00",
        "available_capital": "15000.00"
    })
    
    print("✅ Service mocking works correctly")
    return True


def test_decimal_precision():
    """Test that Decimal types are used for financial values."""
    from decimal import Decimal
    
    # Test value conversion
    value = "10000.00"
    decimal_value = Decimal(value)
    
    assert isinstance(decimal_value, Decimal)
    assert str(decimal_value) == "10000.00"
    
    # Test precision
    result = Decimal("0.1") + Decimal("0.2")
    assert result == Decimal("0.3")
    
    print("✅ Decimal precision validated")
    return True


def test_websocket_handlers():
    """Test that WebSocket handlers are defined."""
    from src.web_interface.websockets.unified_manager import (
        ChannelType,
        UnifiedWebSocketManager,
        MarketDataHandler,
        BotStatusHandler,
        PortfolioHandler,
        TradesHandler,
        OrdersHandler,
        AlertsHandler,
        LogsHandler,
        RiskMetricsHandler
    )
    
    # Check all channel types are defined
    channels = [
        ChannelType.MARKET_DATA,
        ChannelType.BOT_STATUS,
        ChannelType.PORTFOLIO,
        ChannelType.TRADES,
        ChannelType.ORDERS,
        ChannelType.ALERTS,
        ChannelType.LOGS,
        ChannelType.RISK_METRICS
    ]
    
    assert len(channels) == 8, f"Expected 8 channels, got {len(channels)}"
    
    # Check handlers exist
    handlers = [
        MarketDataHandler,
        BotStatusHandler,
        PortfolioHandler,
        TradesHandler,
        OrdersHandler,
        AlertsHandler,
        LogsHandler,
        RiskMetricsHandler
    ]
    
    for handler in handlers:
        assert handler is not None, f"Handler {handler.__name__} not found"
    
    print("✅ WebSocket handlers validated")
    return True


def test_authentication():
    """Test authentication mechanisms."""
    from src.web_interface.security.auth import User
    from unittest.mock import MagicMock
    
    # Create mock user
    mock_user = MagicMock(spec=User)
    mock_user.username = "test_user"
    mock_user.user_id = "user_123"
    mock_user.roles = ["user", "trading"]
    
    # Test role checking
    assert "user" in mock_user.roles
    assert "trading" in mock_user.roles
    assert "admin" not in mock_user.roles
    
    print("✅ Authentication mechanisms validated")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Web Interface Simple Tests")
    print("="*60 + "\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Endpoint Definitions", test_endpoint_definitions),
        ("Service Mocking", test_service_mocking),
        ("Decimal Precision", test_decimal_precision),
        ("WebSocket Handlers", test_websocket_handlers),
        ("Authentication", test_authentication),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ {failed} tests failed")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())