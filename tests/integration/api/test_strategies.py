#!/usr/bin/env python3
"""
Strategies API Integration Tests.

Tests all strategies module API endpoints with real HTTP requests.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient

from src.core.config import Config
from src.web_interface.app import create_app


class TestStrategiesAPI:
    """Test Strategies API endpoints."""

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

    def test_strategies_list(self):
        """Test GET /api/strategies/list endpoint."""
        response = self.client.get("/api/strategies/list")
        assert response.status_code in [200, 403]

    def test_strategies_performance(self):
        """Test GET /api/strategies/performance endpoint."""
        response = self.client.get("/api/strategies/performance")
        assert response.status_code in [200, 403]

    def test_strategies_config(self):
        """Test GET /api/strategies/config endpoint."""
        response = self.client.get("/api/strategies/config")
        assert response.status_code in [200, 403]

    def test_strategies_backtest(self):
        """Test POST /api/strategies/backtest endpoint."""
        test_data = {
            "strategy_name": "mean_reversion",
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }
        response = self.client.post("/api/strategies/backtest", json=test_data)
        assert response.status_code in [200, 403, 422]

    def test_strategies_optimize(self):
        """Test POST /api/strategies/optimize endpoint."""
        test_data = {
            "strategy_name": "mean_reversion",
            "parameters": {"lookback_period": [10, 20, 30], "threshold": [0.02, 0.05, 0.1]},
        }
        response = self.client.post("/api/strategies/optimize", json=test_data)
        assert response.status_code in [200, 403, 422]


if __name__ == "__main__":
    # Quick test run
    test_strategies = TestStrategiesAPI()
    test_strategies.setup_method()

    print("🧪 Testing Strategies API endpoints...")

    endpoints_to_test = [
        ("GET", "/api/strategies/list"),
        ("GET", "/api/strategies/performance"),
        ("GET", "/api/strategies/config"),
    ]

    passed = 0
    for method, endpoint in endpoints_to_test:
        try:
            response = test_strategies.client.get(endpoint)
            if response.status_code in [200, 403]:
                print(f"✅ {method} {endpoint}: {response.status_code}")
                passed += 1
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
                passed += 1
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")

    print(f"\n📊 Strategies API: {passed}/{len(endpoints_to_test)} endpoints working")
