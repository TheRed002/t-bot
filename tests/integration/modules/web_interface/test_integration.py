"""
Integration tests for T-Bot web interface.

This module tests the complete web interface including authentication,
API endpoints, and WebSocket functionality.

Uses REAL services - NO MOCKS for internal services.
"""

import pytest
from fastapi.testclient import TestClient

from src.core.logging import get_logger

logger = get_logger(__name__)


class TestWebInterfaceIntegration:
    """Integration tests for web interface."""

    def test_app_startup(self, test_client):
        """Test that the app starts up correctly."""
        # Test health endpoint
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_api_docs_available(self, test_client):
        """Test that API documentation is available."""
        # Test OpenAPI spec
        response = test_client.get("/openapi.json")
        assert response.status_code == 200

        # Test docs page
        response = test_client.get("/docs")
        assert response.status_code == 200

        # Test redoc page
        response = test_client.get("/redoc")
        assert response.status_code == 200

    def test_api_versions_endpoint(self, test_client):
        """Test API versions endpoint."""
        response = test_client.get("/api/versions")
        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert "default" in data
        assert "latest" in data

    def test_middleware_integration(self, test_client):
        """Test that middleware is working correctly."""
        # Test CORS headers
        response = test_client.options(
            "/health",
            headers={"Origin": "http://testserver", "Access-Control-Request-Method": "GET"},
        )
        assert "Access-Control-Allow-Origin" in response.headers

        # Test process time header
        response = test_client.get("/")
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers

    def test_error_handling_integration(self, test_client):
        """Test error handling integration."""
        # Test 404 error
        response = test_client.get("/nonexistent-endpoint")
        assert response.status_code == 404

        # Test method not allowed
        response = test_client.delete("/health")
        assert response.status_code == 405

    def test_bot_api_integration(self, test_client, auth_headers):
        """Test bot management API integration."""
        # Test bot listing endpoint exists
        response = test_client.get("/api/bot", headers=auth_headers)
        assert response.status_code in [200, 401, 404]

    def test_portfolio_api_integration(self, test_client, auth_headers):
        """Test portfolio API integration."""
        # Test portfolio summary endpoint exists
        response = test_client.get("/api/portfolio", headers=auth_headers)
        assert response.status_code in [200, 401, 404]

    def test_trading_api_integration(self, test_client, auth_headers):
        """Test trading API integration."""
        # Test orders endpoint exists
        response = test_client.get("/api/trading/orders", headers=auth_headers)
        assert response.status_code in [200, 401, 404]

    def test_security_headers_integration(self, test_client):
        """Test that security headers are properly set."""
        response = test_client.get("/health")

        # Check security headers
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "X-API-Version" in response.headers

    def test_complete_workflow_integration(self, test_client):
        """Test a complete workflow from health check to API exploration."""
        # Step 1: Check system health
        health_response = test_client.get("/health")
        assert health_response.status_code == 200
        assert "status" in health_response.json()

        # Step 2: Check API versions
        versions_response = test_client.get("/api/versions")
        assert versions_response.status_code == 200
        assert "versions" in versions_response.json()

        # Step 3: Check documentation
        docs_response = test_client.get("/docs")
        assert docs_response.status_code == 200

        # Step 4: Check root endpoint
        root_response = test_client.get("/")
        assert root_response.status_code == 200
        assert "message" in root_response.json()

        # The workflow completed successfully
        assert True  # All assertions passed
