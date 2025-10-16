#!/usr/bin/env python3
"""
Bot Management API Integration Tests.

Tests all bot management module API endpoints with real HTTP requests.
"""

import pytest


class TestBotManagementAPI:
    """Test Bot Management API endpoints."""

    def test_bot_status(self, test_client):
        """Test GET /api/bot/status endpoint."""
        response = test_client.get("/api/bot/status")
        assert response.status_code in [200, 403]

    def test_bot_list(self, test_client):
        """Test GET /api/bot/list endpoint."""
        response = test_client.get("/api/bot/list")
        assert response.status_code in [200, 403]

    def test_bot_create(self, test_client):
        """Test POST /api/bot/create endpoint."""
        test_data = {
            "bot_name": "Test Bot",
            "bot_type": "manual",
            "strategy_name": "test_strategy",
            "exchanges": ["binance"],
            "symbols": ["BTCUSDT"],
            "allocated_capital": "1000.00",
            "risk_percentage": "0.02",
        }
        response = test_client.post("/api/bot/create", json=test_data)
        assert response.status_code in [200, 403, 422]

    def test_bot_start(self, test_client):
        """Test POST /api/bot/{bot_id}/start endpoint."""
        response = test_client.post("/api/bot/test_bot/start")
        # Will return 404 or 403 since bot doesn't exist or not authenticated
        assert response.status_code in [403, 404, 422]

    def test_bot_stop(self, test_client):
        """Test POST /api/bot/{bot_id}/stop endpoint."""
        response = test_client.post("/api/bot/test_bot/stop")
        # Will return 404 or 403 since bot doesn't exist or not authenticated
        assert response.status_code in [403, 404, 422]

    def test_bot_config(self, test_client):
        """Test GET /api/bot/config endpoint."""
        response = test_client.get("/api/bot/config")
        assert response.status_code in [200, 403]
