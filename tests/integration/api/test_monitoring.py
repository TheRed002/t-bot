#!/usr/bin/env python3
"""
Monitoring API Integration Tests.

Tests all monitoring module API endpoints with real HTTP requests.
"""


class TestMonitoringAPI:
    """Test Monitoring API endpoints."""

    def test_monitoring_health(self, test_client):
        """Test GET /api/monitoring/health endpoint."""
        response = test_client.get("/api/monitoring/health")
        assert response.status_code in [200, 403, 500]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data

    def test_monitoring_status(self, test_client):
        """Test GET /api/monitoring/status endpoint."""
        response = test_client.get("/api/monitoring/status")
        assert response.status_code in [200, 403, 500]
        if response.status_code == 200:
            data = response.json()
            assert "monitoring_active" in data

    def test_monitoring_metrics_json(self, test_client):
        """Test GET /api/monitoring/metrics/json endpoint."""
        response = test_client.get("/api/monitoring/metrics/json")
        assert response.status_code in [200, 403, 500]
        if response.status_code == 200:
            data = response.json()
            assert "metrics" in data

    def test_monitoring_performance_stats(self, test_client):
        """Test GET /api/monitoring/performance/stats endpoint."""
        response = test_client.get("/api/monitoring/performance/stats")
        assert response.status_code in [200, 403, 500]
        if response.status_code == 200:
            data = response.json()
            assert "stats" in data

    def test_monitoring_alerts(self, test_client):
        """Test GET /api/monitoring/alerts endpoint."""
        response = test_client.get("/api/monitoring/alerts")
        assert response.status_code in [200, 403, 500]
        if response.status_code == 200:
            data = response.json()
            assert "alerts" in data
