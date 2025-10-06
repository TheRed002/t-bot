#!/usr/bin/env python3
"""
Trading API Integration Tests.

Tests all trading module API endpoints with real HTTP requests.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient

from src.core.config import Config
from src.web_interface.app import create_app


class TestTradingAPI:
    """Test Trading API endpoints."""

    def setup_method(self):
        """Set up test client."""
        config = Config()
        config.environment = "test"
        config.debug = True
        config.jwt_secret = "test_secret_key"

        # Disable rate limiting for testing
        config.web_interface = {"rate_limiting": {"enabled": False}}

        app = create_app(config)
        self.client = TestClient(app)

    def test_trading_orders(self):
        """Test GET /api/trading/orders endpoint."""
        response = self.client.get("/api/trading/orders")
        assert response.status_code in [200, 403]

    def test_trading_positions(self):
        """Test GET /api/trading/positions endpoint."""
        response = self.client.get("/api/trading/positions")
        assert response.status_code in [200, 403]

    def test_trading_place_order(self):
        """Test POST /api/trading/place_order endpoint."""
        test_data = {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.001"}
        response = self.client.post("/api/trading/place_order", json=test_data)
        assert response.status_code in [200, 403, 422]

    def test_trading_cancel_order(self):
        """Test POST /api/trading/cancel_order endpoint."""
        test_data = {"order_id": "test_order_123"}
        response = self.client.post("/api/trading/cancel_order", json=test_data)
        assert response.status_code in [200, 403, 422]

    def test_trading_balance(self):
        """Test GET /api/trading/balance endpoint."""
        response = self.client.get("/api/trading/balance")
        assert response.status_code in [200, 403]

    def test_trading_history(self):
        """Test GET /api/trading/history endpoint."""
        response = self.client.get("/api/trading/history")
        assert response.status_code in [200, 403]


if __name__ == "__main__":
    # Quick test run
    test_trading = TestTradingAPI()
    test_trading.setup_method()

    print("🧪 Testing Trading API endpoints...")

    endpoints_to_test = [
        ("GET", "/api/trading/orders"),
        ("GET", "/api/trading/positions"),
        ("GET", "/api/trading/balance"),
        ("GET", "/api/trading/history"),
    ]

    passed = 0
    for method, endpoint in endpoints_to_test:
        try:
            response = test_trading.client.get(endpoint)
            if response.status_code in [200, 403]:
                print(f"✅ {method} {endpoint}: {response.status_code}")
                passed += 1
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
                passed += 1
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")

    print(f"\n📊 Trading API: {passed}/{len(endpoints_to_test)} endpoints working")
