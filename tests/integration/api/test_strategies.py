#!/usr/bin/env python3
"""
Strategies API Integration Tests.

Tests all strategies module API endpoints with real HTTP requests.
"""


class TestStrategiesAPI:
    """Test Strategies API endpoints."""

    def test_strategies_list(self, test_client):
        """Test GET /api/strategies/ endpoint."""
        response = test_client.get("/api/strategies/")
        assert response.status_code in [200, 403]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    def test_strategies_performance(self, test_client):
        """Test GET /api/strategies/{strategy_name}/performance endpoint."""
        # First get list of strategies
        response = test_client.get("/api/strategies/")
        assert response.status_code in [200, 403]

        if response.status_code == 200:
            strategies = response.json()
            if strategies:
                strategy_name = strategies[0]["name"]
                response = test_client.get(f"/api/strategies/{strategy_name}/performance")
                assert response.status_code in [200, 403, 404]

    def test_strategies_config(self, test_client):
        """Test GET /api/strategies/{strategy_name} endpoint."""
        # First get list of strategies
        response = test_client.get("/api/strategies/")
        assert response.status_code in [200, 403]

        if response.status_code == 200:
            strategies = response.json()
            if strategies:
                strategy_name = strategies[0]["name"]
                response = test_client.get(f"/api/strategies/{strategy_name}")
                assert response.status_code in [200, 403, 404]

    def test_strategies_backtest(self, test_client):
        """Test POST /api/strategies/{strategy_name}/backtest endpoint."""
        # First get list of strategies
        response = test_client.get("/api/strategies/")
        assert response.status_code in [200, 403]

        if response.status_code == 200:
            strategies = response.json()
            if strategies:
                strategy_name = strategies[0]["name"]
                test_data = {
                    "symbol": "BTCUSDT",
                    "exchange": "binance",
                    "start_date": "2024-01-01T00:00:00",
                    "end_date": "2024-01-31T23:59:59",
                    "initial_capital": "10000.00",
                    "parameters": {}
                }
                response = test_client.post(f"/api/strategies/{strategy_name}/backtest", json=test_data)
                assert response.status_code in [200, 400, 403, 404, 422]

    def test_strategies_optimize(self, test_client):
        """Test GET /api/strategies/{strategy_name}/optimize endpoint."""
        # First get list of strategies
        response = test_client.get("/api/strategies/")
        assert response.status_code in [200, 403]

        if response.status_code == 200:
            strategies = response.json()
            if strategies:
                strategy_name = strategies[0]["name"]
                response = test_client.get(f"/api/strategies/{strategy_name}/optimize")
                assert response.status_code in [200, 403, 404]
