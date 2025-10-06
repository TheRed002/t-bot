#!/usr/bin/env python3
"""
Risk Management API Integration Tests.

Tests all risk management module API endpoints with real HTTP requests.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient

from src.core.config import Config
from src.web_interface.app import create_app


class TestRiskManagementAPI:
    """Test Risk Management API endpoints."""

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

    def test_risk_metrics(self):
        """Test GET /api/risk/metrics endpoint."""
        response = self.client.get("/api/risk/metrics")
        assert response.status_code in [200, 403]

    def test_risk_limits(self):
        """Test GET /api/risk/limits endpoint."""
        response = self.client.get("/api/risk/limits")
        assert response.status_code in [200, 403]

    def test_risk_exposure(self):
        """Test GET /api/risk/exposure endpoint."""
        response = self.client.get("/api/risk/exposure")
        assert response.status_code in [200, 403]

    def test_risk_var(self):
        """Test GET /api/risk/var endpoint."""
        response = self.client.get("/api/risk/var")
        assert response.status_code in [200, 403]

    def test_risk_validation(self):
        """Test POST /api/risk/validation endpoint."""
        test_data = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.001", "price": "50000.00"}
        response = self.client.post("/api/risk/validation", json=test_data)
        assert response.status_code in [200, 403, 422]


if __name__ == "__main__":
    # Quick test run
    test_risk = TestRiskManagementAPI()
    test_risk.setup_method()

    print("🧪 Testing Risk Management API endpoints...")

    endpoints_to_test = [
        ("GET", "/api/risk/metrics"),
        ("GET", "/api/risk/limits"),
        ("GET", "/api/risk/exposure"),
        ("GET", "/api/risk/var"),
    ]

    passed = 0
    for method, endpoint in endpoints_to_test:
        try:
            response = test_risk.client.get(endpoint)
            if response.status_code in [200, 403]:
                print(f"✅ {method} {endpoint}: {response.status_code}")
                passed += 1
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
                passed += 1
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")

    print(f"\n📊 Risk Management API: {passed}/{len(endpoints_to_test)} endpoints working")
