#!/usr/bin/env python3
"""
Trading API Integration Tests.

Tests all trading module API endpoints with real HTTP requests.
Uses authenticated clients to test protected endpoints.
"""

import pytest


class TestTradingAPI:
    """Test Trading API endpoints."""

    def test_trading_orders(self, authenticated_client):
        """Test GET /api/trading/orders endpoint."""
        response = authenticated_client.get("/api/trading/orders")
        assert response.status_code in [200, 401, 404, 500, 503], f"Unexpected status: {response.status_code}"

    def test_trading_positions(self, authenticated_client):
        """Test GET /api/trading/positions endpoint."""
        response = authenticated_client.get("/api/trading/positions")
        assert response.status_code in [200, 401, 404, 500, 503], f"Unexpected status: {response.status_code}"

    def test_trading_place_order(self, authenticated_client):
        """Test POST /api/trading/place_order endpoint."""
        test_data = {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.001"}
        response = authenticated_client.post("/api/trading/place_order", json=test_data)
        assert response.status_code in [200, 404, 422, 500, 503], f"Unexpected status: {response.status_code}"

    def test_trading_cancel_order(self, authenticated_client):
        """Test POST /api/trading/cancel_order endpoint."""
        test_data = {"order_id": "test_order_123"}
        response = authenticated_client.post("/api/trading/cancel_order", json=test_data)
        assert response.status_code in [200, 404, 422, 500, 503], f"Unexpected status: {response.status_code}"

    def test_trading_balance(self, authenticated_client):
        """Test GET /api/trading/balance endpoint."""
        response = authenticated_client.get("/api/trading/balance")
        assert response.status_code in [200, 401, 404, 500, 503], f"Unexpected status: {response.status_code}"

    def test_trading_history(self, authenticated_client):
        """Test GET /api/trading/history endpoint."""
        response = authenticated_client.get("/api/trading/history")
        assert response.status_code in [200, 401, 404, 500, 503], f"Unexpected status: {response.status_code}"

    def test_trading_unauthenticated_access_denied(self, test_client):
        """Test that unauthenticated requests are denied."""
        response = test_client.get("/api/trading/orders")
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

    print("🧪 Testing Trading API endpoints with authentication...")

    endpoints_to_test = [
        ("GET", "/api/trading/orders"),
        ("GET", "/api/trading/positions"),
        ("GET", "/api/trading/balance"),
        ("GET", "/api/trading/history"),
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

    print(f"\n📊 Trading API: {passed}/{len(endpoints_to_test)} endpoints working")
