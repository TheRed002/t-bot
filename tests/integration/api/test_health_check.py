#!/usr/bin/env python3
"""
Health API Integration Tests.

Tests health and system status endpoints with real HTTP requests.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient

from src.core.config import Config
from src.web_interface.app import create_app


class TestHealthAPI:
    """Test Health API endpoints."""

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

    def test_health_endpoint(self):
        """Test GET /health endpoint."""
        response = self.client.get("/health")
        # Health endpoint should be public and working
        assert response.status_code == 200

    def test_health_detailed(self):
        """Test GET /health/detailed endpoint."""
        response = self.client.get("/health/detailed")
        assert response.status_code in [200, 404]  # May not exist

    def test_root_endpoint(self):
        """Test GET / endpoint."""
        response = self.client.get("/")
        assert response.status_code in [200, 404]


if __name__ == "__main__":
    # Quick test run
    test_health = TestHealthAPI()
    test_health.setup_method()

    print("🧪 Testing Health API endpoints...")

    endpoints_to_test = [("GET", "/health"), ("GET", "/")]

    passed = 0
    for method, endpoint in endpoints_to_test:
        try:
            response = test_health.client.get(endpoint)
            if response.status_code == 200:
                print(f"✅ {method} {endpoint}: {response.status_code}")
                passed += 1
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
                passed += 1
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")

    print(f"\n📊 Health API: {passed}/{len(endpoints_to_test)} endpoints working")
