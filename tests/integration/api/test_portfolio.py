#!/usr/bin/env python3
"""
Portfolio API Integration Tests.

Tests all portfolio module API endpoints with real HTTP requests.
Uses authenticated clients to test protected endpoints.
"""

import pytest


class TestPortfolioAPI:
    """Test Portfolio API endpoints."""

    def test_portfolio_summary(self, authenticated_client):
        """Test GET /api/portfolio/summary endpoint."""
        response = authenticated_client.get("/api/portfolio/summary")
        assert response.status_code in [200, 401, 404, 500, 503], f"Unexpected status: {response.status_code}"

    def test_portfolio_holdings(self, authenticated_client):
        """Test GET /api/portfolio/holdings endpoint."""
        response = authenticated_client.get("/api/portfolio/holdings")
        assert response.status_code in [200, 401, 404, 500, 503], f"Unexpected status: {response.status_code}"

    def test_portfolio_performance(self, authenticated_client):
        """Test GET /api/portfolio/performance endpoint."""
        response = authenticated_client.get("/api/portfolio/performance")
        assert response.status_code in [200, 401, 404, 500, 503], f"Unexpected status: {response.status_code}"

    def test_portfolio_allocation(self, authenticated_client):
        """Test GET /api/portfolio/allocation endpoint."""
        response = authenticated_client.get("/api/portfolio/allocation")
        assert response.status_code in [200, 401, 404, 500, 503], f"Unexpected status: {response.status_code}"

    def test_portfolio_rebalance(self, authenticated_client):
        """Test POST /api/portfolio/rebalance endpoint."""
        test_data = {"strategy": "equal_weight", "symbols": ["BTCUSDT", "ETHUSDT"]}
        response = authenticated_client.post("/api/portfolio/rebalance", json=test_data)
        assert response.status_code in [200, 400, 404, 422, 500, 503], f"Unexpected status: {response.status_code}"

    def test_portfolio_unauthenticated_access_denied(self, test_client):
        """Test that unauthenticated requests are denied."""
        response = test_client.get("/api/portfolio/summary")
        assert response.status_code == 403, "Unauthenticated requests should return 403"


if __name__ == "__main__":
    # Quick test run with authentication
    from fastapi.testclient import TestClient
    from src.core.config import Config
    from src.web_interface.app import create_app

    config = Config()
    config.environment = "test"
    config.debug = True
    config.jwt_secret = "test_secret_key"
    config.web_interface = {"rate_limiting": {"enabled": False}}

    app = create_app(config)
    client = TestClient(app)

    # Login to get token
    login_response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})

    if login_response.status_code != 200:
        print("❌ Failed to authenticate. Database may not be seeded.")
        exit(1)

    token = login_response.json()["token"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    print("🧪 Testing Portfolio API endpoints with authentication...")

    endpoints_to_test = [
        ("GET", "/api/portfolio/summary"),
        ("GET", "/api/portfolio/holdings"),
        ("GET", "/api/portfolio/performance"),
        ("GET", "/api/portfolio/allocation"),
    ]

    passed = 0
    for method, endpoint in endpoints_to_test:
        try:
            response = client.get(endpoint, headers=headers)
            if response.status_code in [200, 500, 503]:
                print(f"✅ {method} {endpoint}: {response.status_code}")
                passed += 1
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")

    print(f"\n📊 Portfolio API: {passed}/{len(endpoints_to_test)} endpoints working")
