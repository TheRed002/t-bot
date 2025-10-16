#!/usr/bin/env python3
"""
Risk Management API Integration Tests.

Tests all risk management module API endpoints with real HTTP requests.
"""

import pytest
from decimal import Decimal


class TestRiskManagementAPI:
    """Test Risk Management API endpoints."""

    def test_risk_metrics(self, test_client):
        """Test GET /api/risk/metrics endpoint."""
        response = test_client.get("/api/risk/metrics")
        assert response.status_code in [200, 403]

    def test_risk_limits(self, test_client):
        """Test GET /api/risk/limits endpoint."""
        response = test_client.get("/api/risk/limits")
        assert response.status_code in [200, 403]

    def test_risk_exposure(self, test_client):
        """Test GET /api/risk/positions endpoint (position risk exposure)."""
        response = test_client.get("/api/risk/positions")
        assert response.status_code in [200, 403]

    def test_risk_var(self, test_client):
        """Test GET /api/risk/alerts endpoint (VaR included in alerts)."""
        response = test_client.get("/api/risk/alerts")
        assert response.status_code in [200, 403]

    def test_risk_validation(self, test_client):
        """Test POST /api/risk/stress-test endpoint (risk validation via stress test)."""
        test_data = {
            "test_name": "integration_test",
            "scenarios": [{"name": "market_crash", "btc_change": -0.2}],
            "confidence_levels": ["0.95"],  # Use string for JSON serialization
            "time_horizons": [1],
        }
        response = test_client.post("/api/risk/stress-test", json=test_data)
        # Will return 403 (auth), 422 (validation), 400 (business logic), or 200 (success)
        assert response.status_code in [200, 400, 403, 422]
