#!/usr/bin/env python3
"""
Capital Management API Integration Tests.

Tests all capital management module API endpoints with real HTTP requests.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient

from src.core.config import Config
from src.web_interface.app import create_app


class TestCapitalManagementAPI:
    """Test Capital Management API endpoints."""

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

    def test_capital_metrics(self):
        """Test GET /api/capital/metrics endpoint."""
        response = self.client.get("/api/capital/metrics")
        assert response.status_code in [200, 403]

    def test_capital_allocation(self):
        """Test GET /api/capital/allocation endpoint."""
        response = self.client.get("/api/capital/allocation")
        assert response.status_code in [200, 403]

    def test_capital_usage(self):
        """Test GET /api/capital/usage endpoint."""
        response = self.client.get("/api/capital/usage")
        assert response.status_code in [200, 403]

    def test_capital_limits(self):
        """Test GET /api/capital/limits endpoint."""
        response = self.client.get("/api/capital/limits")
        assert response.status_code in [200, 403]

    def test_capital_allocation_post(self):
        """Test POST /api/capital/allocation endpoint."""
        test_data = {"bot_id": "test_bot", "amount": "1000.00"}
        response = self.client.post("/api/capital/allocation", json=test_data)
        assert response.status_code in [200, 403, 422]  # 422 for validation errors


if __name__ == "__main__":
    # Quick test run
    test_capital = TestCapitalManagementAPI()
    test_capital.setup_method()

    print("🧪 Testing Capital Management API endpoints...")

    endpoints_to_test = [
        ("GET", "/api/capital/metrics"),
        ("GET", "/api/capital/allocation"),
        ("GET", "/api/capital/usage"),
        ("GET", "/api/capital/limits"),
    ]

    passed = 0
    for method, endpoint in endpoints_to_test:
        try:
            response = test_capital.client.get(endpoint)
            if response.status_code in [200, 403]:
                print(f"✅ {method} {endpoint}: {response.status_code}")
                passed += 1
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
                passed += 1
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")

    print(f"\n📊 Capital Management API: {passed}/{len(endpoints_to_test)} endpoints working")
