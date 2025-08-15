"""
Integration tests for T-Bot web interface.

This module tests the complete web interface including authentication,
API endpoints, and WebSocket functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.core.config import Config
from src.web_interface.app import create_app


class TestWebInterfaceIntegration:
    """Integration tests for web interface."""

    @pytest.fixture
    def integration_config(self):
        """Create integration test configuration."""
        config = Config()
        config.debug = True
        # Use existing security config
        config.security.secret_key = "integration_test_secret_key_32_chars_long_minimum"
        config.security.jwt_algorithm = "HS256"
        config.security.jwt_expire_minutes = 30
        return config

    @pytest.fixture
    def integration_app(self, integration_config):
        """Create integration test app."""
        # Mock dependencies
        mock_orchestrator = AsyncMock()
        mock_execution_engine = AsyncMock()
        mock_model_manager = AsyncMock()
        
        # Setup basic mock responses
        mock_orchestrator.get_orchestrator_status.return_value = {
            "orchestrator": {"is_running": True, "total_bots": 0},
            "global_metrics": {"total_pnl": 0, "total_trades": 0}
        }
        
        app = create_app(
            config=integration_config,
            bot_orchestrator_instance=mock_orchestrator,
            execution_engine_instance=mock_execution_engine,
            model_manager_instance=mock_model_manager
        )
        
        return app

    def test_app_startup(self, integration_app):
        """Test that the app starts up correctly."""
        with TestClient(integration_app) as client:
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "t-bot-api"

    def test_root_endpoint(self, integration_app):
        """Test root endpoint."""
        with TestClient(integration_app) as client:
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "T-Bot Trading System API"
            assert data["version"] == "1.0.0"
            assert data["docs"] == "/docs"

    def test_api_docs_available(self, integration_app):
        """Test that API documentation is available."""
        with TestClient(integration_app) as client:
            # Test OpenAPI spec
            response = client.get("/openapi.json")
            assert response.status_code == 200
            
            # Test docs page  
            response = client.get("/docs")
            assert response.status_code == 200

    def test_auth_flow_integration(self, integration_app):
        """Test complete authentication flow."""
        with TestClient(integration_app) as client:
            # Test login
            with patch('src.web_interface.api.auth.authenticate_user') as mock_auth:
                with patch('src.web_interface.api.auth.create_access_token') as mock_create_token:
                    from src.web_interface.security.auth import UserInDB, Token
                    
                    # Mock successful authentication
                    mock_user = UserInDB(
                        user_id="test_001",
                        username="testuser",
                        email="test@example.com",
                        hashed_password="hashed",
                        scopes=["read", "write"]
                    )
                    mock_auth.return_value = mock_user
                    
                    mock_token = Token(
                        access_token="test_token",
                        refresh_token="test_refresh",
                        expires_in=1800
                    )
                    mock_create_token.return_value = mock_token
                    
                    # Login
                    login_response = client.post("/auth/login", json={
                        "username": "testuser",
                        "password": "password"
                    })
                    
                    assert login_response.status_code == 200
                    login_data = login_response.json()
                    assert login_data["success"] is True
                    assert "token" in login_data

    def test_middleware_integration(self, integration_app):
        """Test that middleware is working correctly."""
        with TestClient(integration_app) as client:
            # Test CORS headers
            response = client.options("/health", headers={
                "Origin": "http://testserver",
                "Access-Control-Request-Method": "GET"
            })
            assert "Access-Control-Allow-Origin" in response.headers
            
            # Test rate limiting headers (should be added by middleware)
            # Use root endpoint instead of /health since /health is exempt from rate limiting
            response = client.get("/")
            assert response.status_code == 200
            # Rate limit headers should be present
            assert "X-RateLimit-Limit" in response.headers
            
            # Test process time header
            assert "X-Process-Time" in response.headers

    def test_error_handling_integration(self, integration_app):
        """Test error handling integration."""
        with TestClient(integration_app) as client:
            # Test 404 error
            response = client.get("/nonexistent-endpoint")
            assert response.status_code == 404
            
            # Test validation error
            response = client.post("/auth/login", json={
                "username": "",  # Invalid empty username
                "password": "test"
            })
            assert response.status_code == 422  # Validation error

    def test_bot_api_integration(self, integration_app):
        """Test bot management API integration."""
        with TestClient(integration_app) as client:
            with patch('src.web_interface.api.bot_management.get_current_user') as mock_get_user:
                with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                    from src.web_interface.security.auth import User
                    
                    # Mock authenticated user
                    mock_user = User(
                        user_id="test_001",
                        username="testuser",
                        email="test@example.com",
                        is_active=True,
                        scopes=["read"]
                    )
                    mock_get_user.return_value = mock_user
                    
                    # Mock bot list
                    mock_orchestrator.get_bot_list = AsyncMock(return_value=[])
                    
                    # Test bot listing
                    response = client.get("/api/bots/", headers={
                        "Authorization": "Bearer test_token"
                    })
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "bots" in data
                    assert "total" in data

    def test_portfolio_api_integration(self, integration_app):
        """Test portfolio API integration.""" 
        with TestClient(integration_app) as client:
            with patch('src.web_interface.api.portfolio.get_current_user') as mock_get_user:
                from src.web_interface.security.auth import User
                
                mock_user = User(
                    user_id="test_001",
                    username="testuser",
                    email="test@example.com",
                    is_active=True,
                    scopes=["read"]
                )
                mock_get_user.return_value = mock_user
                
                # Test portfolio summary
                response = client.get("/api/portfolio/summary", headers={
                    "Authorization": "Bearer test_token"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert "total_value" in data
                assert "daily_pnl" in data

    def test_trading_api_integration(self, integration_app):
        """Test trading API integration."""
        with TestClient(integration_app) as client:
            with patch('src.web_interface.api.trading.get_current_user') as mock_get_user:
                from src.web_interface.security.auth import User
                
                mock_user = User(
                    user_id="test_001",
                    username="testuser", 
                    email="test@example.com",
                    is_active=True,
                    scopes=["read"]
                )
                mock_get_user.return_value = mock_user
                
                # Test market data
                response = client.get("/api/trading/market-data/BTCUSDT", headers={
                    "Authorization": "Bearer test_token"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert "symbol" in data
                assert "price" in data

    def test_security_headers_integration(self, integration_app):
        """Test that security headers are properly set."""
        with TestClient(integration_app) as client:
            response = client.get("/health")
            
            # Check security headers
            assert response.headers.get("X-Content-Type-Options") == "nosniff"
            assert response.headers.get("X-Frame-Options") == "DENY"
            assert response.headers.get("X-XSS-Protection") == "1; mode=block"
            assert "X-API-Version" in response.headers

    def test_complete_workflow_integration(self, integration_app):
        """Test a complete workflow from authentication to API usage."""
        with TestClient(integration_app) as client:
            # Step 1: Check system health
            health_response = client.get("/health")
            assert health_response.status_code == 200
            
            # Step 2: Mock authentication
            with patch('src.web_interface.api.auth.authenticate_user') as mock_auth:
                with patch('src.web_interface.api.auth.create_access_token') as mock_create_token:
                    from src.web_interface.security.auth import UserInDB, Token
                    
                    mock_user = UserInDB(
                        user_id="test_001",
                        username="testuser",
                        email="test@example.com",
                        hashed_password="hashed",
                        scopes=["read", "write", "trade"]
                    )
                    mock_auth.return_value = mock_user
                    
                    mock_token = Token(
                        access_token="test_token",
                        refresh_token="test_refresh",
                        expires_in=1800
                    )
                    mock_create_token.return_value = mock_token
                    
                    # Login
                    login_response = client.post("/auth/login", json={
                        "username": "testuser",
                        "password": "password"
                    })
                    assert login_response.status_code == 200
                    token = login_response.json()["token"]["access_token"]
            
            # Step 3: Use authenticated endpoints
            headers = {"Authorization": f"Bearer {token}"}
            
            with patch('src.web_interface.api.bot_management.get_current_user') as mock_get_user:
                from src.web_interface.security.auth import User
                
                mock_user = User(
                    user_id="test_001",
                    username="testuser",
                    email="test@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"]
                )
                mock_get_user.return_value = mock_user
                
                # List bots
                with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                    mock_orchestrator.get_bot_list = AsyncMock(return_value=[])
                    
                    bots_response = client.get("/api/bots/", headers=headers)
                    assert bots_response.status_code == 200
                    
            # Step 4: Test portfolio access
            with patch('src.web_interface.api.portfolio.get_current_user') as mock_get_user:
                mock_get_user.return_value = mock_user
                
                portfolio_response = client.get("/api/portfolio/summary", headers=headers)
                assert portfolio_response.status_code == 200
                
            # The workflow completed successfully
            assert True  # All assertions passed