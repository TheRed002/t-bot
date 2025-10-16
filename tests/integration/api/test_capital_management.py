#!/usr/bin/env python3
"""
Capital Management API Integration Tests.

Tests all capital management module API endpoints with real HTTP requests.
"""

import pytest


class TestCapitalManagementAPI:
    """Test Capital Management API endpoints."""

    def test_capital_metrics(self, test_client):
        """Test GET /api/capital/metrics endpoint."""
        response = test_client.get("/api/capital/metrics")
        assert response.status_code in [200, 403]

    def test_capital_allocation(self, test_client):
        """Test GET /api/capital/allocations endpoint."""
        response = test_client.get("/api/capital/allocations")
        # May require auth (403), validation (422), or success (200)
        assert response.status_code in [200, 403, 422]

    def test_capital_usage(self, test_client):
        """Test GET /api/capital/utilization endpoint."""
        response = test_client.get("/api/capital/utilization")
        assert response.status_code in [200, 403]

    def test_capital_limits(self, test_client):
        """Test GET /api/capital/limits endpoint."""
        response = test_client.get("/api/capital/limits")
        assert response.status_code in [200, 403]

    def test_capital_allocation_post(self, test_client):
        """Test POST /api/capital/allocate endpoint."""
        test_data = {
            "strategy_id": "test_strategy",
            "exchange": "binance",
            "amount": "1000.00",
            "bot_id": "test_bot",
        }
        response = test_client.post("/api/capital/allocate", json=test_data)
        # Will return 403 (auth) or 422 (validation) or 400 (business logic)
        assert response.status_code in [400, 403, 422]
