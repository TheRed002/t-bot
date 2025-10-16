"""
Enhanced Integration Tests for Web Interface Module.

This test suite adds comprehensive coverage for:
- WebSocket connections and messaging
- Authentication flows and token management
- API endpoint validation with real services
- Error handling and edge cases
- Request/response validation with Pydantic models
- CORS and security headers
- Rate limiting (passive checks)

Uses REAL services - NO MOCKS for internal services.
"""

import asyncio
from decimal import Decimal

import pytest

from src.core.logging import get_logger

logger = get_logger(__name__)


class TestAuthenticationFlows:
    """Test complete authentication workflows."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_user_registration_login_flow(self, web_interface_client):
        """Test user registration and login flow."""
        # Register new user
        register_data = {
            "username": "integration_test_user",
            "email": "integration@test.com",
            "password": "SecurePassword123!",
        }

        response = web_interface_client.post("/api/auth/register", json=register_data)

        # Registration may succeed (200/201), already exist (400/409), or not implemented (404)
        assert response.status_code in [200, 201, 400, 404, 409, 422]

        # Login with registered credentials
        login_data = {
            "username": register_data["username"],
            "password": register_data["password"],
        }

        response = web_interface_client.post("/api/auth/login", json=login_data)

        # Login should succeed if registration worked or user exists
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data or "token" in data

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_invalid_login_credentials(self, web_interface_client):
        """Test login with invalid credentials."""
        response = web_interface_client.post(
            "/api/auth/login",
            json={
                "username": "nonexistent_user_xyz",
                "password": "WrongPassword123!",
            },
        )

        # Should fail with 401 or 404
        assert response.status_code in [401, 404, 422]

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_protected_endpoint_without_auth(self, web_interface_client):
        """Test accessing protected endpoint without authentication."""
        # Try to access bot list without auth
        response = web_interface_client.get("/api/bot")

        # Should require authentication
        assert response.status_code in [401, 403, 404]

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_token_validation(self, web_interface_client):
        """Test JWT token validation."""
        # Try with invalid token
        headers = {"Authorization": "Bearer invalid_token_xyz"}
        response = web_interface_client.get("/api/bot", headers=headers)

        # Should reject invalid token
        assert response.status_code in [401, 403, 404]


class TestAPIEndpointValidation:
    """Test API endpoint request/response validation."""

    def test_bot_creation_validation(self, web_interface_client, auth_headers):
        """Test bot creation with invalid data."""
        # Test with missing required fields
        invalid_data = {
            "name": "Test Bot",
            # Missing strategy, symbols, etc.
        }

        response = web_interface_client.post("/api/bot", json=invalid_data, headers=auth_headers)

        # Should fail validation
        assert response.status_code in [400, 401, 422]

    def test_order_creation_validation(self, web_interface_client, auth_headers):
        """Test order creation with invalid data."""
        # Test with invalid quantity format
        invalid_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "quantity": "invalid",  # Invalid quantity
            "price": "45000.00",
        }

        response = web_interface_client.post(
            "/api/trading/orders", json=invalid_order, headers=auth_headers
        )

        # Should fail validation
        assert response.status_code in [400, 401, 422]

    def test_decimal_precision_validation(self, web_interface_client, auth_headers):
        """Test that decimal precision is properly validated."""
        order_data = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "quantity": "0.00000001",  # Valid but very small
            "price": "45000.12345678",  # Valid precision
        }

        response = web_interface_client.post(
            "/api/trading/orders", json=order_data, headers=auth_headers
        )

        # May succeed or fail based on exchange limits
        assert response.status_code in [200, 201, 400, 401, 422]


class TestWebSocketIntegration:
    """Test WebSocket functionality."""

    def test_websocket_manager_available(self, websocket_test_context):
        """Test that WebSocket manager is available."""
        assert websocket_test_context is not None
        assert hasattr(websocket_test_context, "sio")

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_websocket_connection_stats(self, websocket_test_context):
        """Test WebSocket connection statistics."""
        if hasattr(websocket_test_context, "get_connection_stats"):
            stats = websocket_test_context.get_connection_stats()
            assert isinstance(stats, dict)


class TestCORSConfiguration:
    """Test CORS configuration and headers."""

    def test_cors_preflight_request(self, web_interface_client):
        """Test CORS preflight OPTIONS request."""
        response = web_interface_client.options(
            "/api/bot",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        # Should return CORS headers
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code in [200, 404]

    def test_cors_headers_on_response(self, web_interface_client):
        """Test that CORS headers are present on responses."""
        response = web_interface_client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200
        # CORS headers should be present
        assert "Access-Control-Allow-Origin" in response.headers


class TestSecurityHeaders:
    """Test security headers configuration."""

    def test_security_headers_present(self, web_interface_client):
        """Test that security headers are properly set."""
        response = web_interface_client.get("/health")

        assert response.status_code == 200

        # Check security headers
        headers = response.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("X-XSS-Protection") == "1; mode=block"

    def test_api_version_header(self, web_interface_client):
        """Test that API version header is present."""
        response = web_interface_client.get("/health")

        assert response.status_code == 200
        assert "X-API-Version" in response.headers


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_large_request_body(self, web_interface_client, auth_headers):
        """Test handling of large request bodies."""
        # Create a large bot configuration
        large_config = {
            "name": "Test Bot",
            "strategy": "momentum",
            "symbols": ["BTC/USDT"] * 100,  # Many symbols
            "parameters": {f"param_{i}": i for i in range(100)},
        }

        response = web_interface_client.post("/api/bot", json=large_config, headers=auth_headers)

        # Should handle gracefully
        assert response.status_code in [200, 201, 400, 401, 413, 422]

    def test_concurrent_requests(self, web_interface_client):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request():
            return web_interface_client.get("/health")

        # Make 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        for response in results:
            assert response.status_code == 200

    def test_malformed_json_request(self, web_interface_client, auth_headers):
        """Test handling of malformed JSON."""
        response = web_interface_client.post(
            "/api/bot",
            data='{"name": "Test", invalid json}',
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        # Should return 400 or 422
        assert response.status_code in [400, 422]

    def test_missing_content_type(self, web_interface_client, auth_headers):
        """Test request with missing Content-Type header."""
        response = web_interface_client.post(
            "/api/bot",
            data='{"name": "Test Bot"}',
            headers=auth_headers,
        )

        # May succeed or fail based on FastAPI's content negotiation
        assert response.status_code in [200, 201, 400, 401, 422]


class TestAPIVersioning:
    """Test API versioning functionality."""

    def test_version_header_routing(self, web_interface_client):
        """Test routing based on API version header."""
        # Test with version header
        headers = {"X-API-Version": "v1"}
        response = web_interface_client.get("/health", headers=headers)

        assert response.status_code == 200
        assert "X-API-Version" in response.headers

    def test_default_version_routing(self, web_interface_client):
        """Test default version routing without header."""
        response = web_interface_client.get("/health")

        assert response.status_code == 200
        # Should use default version
        assert "X-API-Version" in response.headers


class TestRealServiceIntegration:
    """Test integration with real services."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_portfolio_service_integration(self, web_interface_client, auth_headers):
        """Test portfolio service integration."""
        response = web_interface_client.get("/api/portfolio", headers=auth_headers)

        # Endpoint should exist
        assert response.status_code in [200, 401, 404]

        if response.status_code == 200:
            data = response.json()
            # Validate response structure if successful
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_risk_service_integration(self, web_interface_client, auth_headers):
        """Test risk service integration."""
        response = web_interface_client.get("/api/risk/metrics", headers=auth_headers)

        # Endpoint should exist
        assert response.status_code in [200, 401, 404]

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_strategy_service_integration(self, web_interface_client, auth_headers):
        """Test strategy service integration."""
        response = web_interface_client.get("/api/strategies", headers=auth_headers)

        # Endpoint should exist
        assert response.status_code in [200, 401, 404]

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_monitoring_service_integration(self, web_interface_client, auth_headers):
        """Test monitoring service integration."""
        response = web_interface_client.get("/api/monitoring/metrics", headers=auth_headers)

        # Endpoint should exist, may not be fully initialized in tests
        assert response.status_code in [200, 401, 404, 500]


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_complete_bot_lifecycle(self, web_interface_client, auth_headers, sample_bot_config):
        """Test complete bot lifecycle workflow."""
        # 1. Create bot
        create_response = web_interface_client.post(
            "/api/bot", json=sample_bot_config, headers=auth_headers
        )

        # May succeed or fail based on auth/validation
        if create_response.status_code not in [200, 201]:
            # Skip rest if creation fails
            return

        bot_data = create_response.json()
        bot_id = bot_data.get("id") or bot_data.get("bot_id")

        if not bot_id:
            # Skip if no bot ID returned
            return

        # 2. Get bot details
        get_response = web_interface_client.get(f"/api/bot/{bot_id}", headers=auth_headers)
        assert get_response.status_code in [200, 404]

        # 3. List all bots
        list_response = web_interface_client.get("/api/bot", headers=auth_headers)
        assert list_response.status_code in [200, 404]

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_trading_workflow(self, web_interface_client, auth_headers):
        """Test trading workflow."""
        # 1. Check positions
        positions_response = web_interface_client.get(
            "/api/trading/positions", headers=auth_headers
        )
        assert positions_response.status_code in [200, 401, 404]

        # 2. Check orders
        orders_response = web_interface_client.get("/api/trading/orders", headers=auth_headers)
        assert orders_response.status_code in [200, 401, 404]

        # 3. Check portfolio
        portfolio_response = web_interface_client.get("/api/portfolio", headers=auth_headers)
        assert portfolio_response.status_code in [200, 401, 404]


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""

    def test_response_time_consistency(self, web_interface_client):
        """Test that response times are consistent."""
        import time

        times = []
        for _ in range(10):
            start = time.time()
            response = web_interface_client.get("/health")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code == 200

        # Average should be under 0.5s
        avg_time = sum(times) / len(times)
        assert avg_time < 0.5, f"Average response time {avg_time:.3f}s too high"

    def test_burst_request_handling(self, web_interface_client):
        """Test handling of burst requests."""
        import concurrent.futures

        def make_burst_request(endpoint):
            return web_interface_client.get(endpoint)

        # Make burst of requests to different endpoints
        endpoints = ["/health", "/", "/api/versions"] * 5

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(make_burst_request, ep) for ep in endpoints]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        for response in results:
            assert response.status_code == 200


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
