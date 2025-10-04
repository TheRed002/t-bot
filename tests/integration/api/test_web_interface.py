"""
Comprehensive API Integration Tests for Web Interface.

This module tests all new API endpoints to ensure:
1. Proper service layer integration
2. Correct data transformation
3. Authentication and authorization
4. Error handling
5. Financial precision (Decimal usage)
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.web_interface.app import create_app


@pytest.fixture
def app():
    """Create test app instance."""
    # create_app() requires a Config instance
    from src.core.config import Config
    config = Config()
    return create_app(config)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(app):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Get auth headers for testing."""
    return {
        "Authorization": "Bearer test_token",
        "X-User-ID": "test_user",
        "X-User-Role": "admin",
    }


@pytest.fixture
def mock_services():
    """Mock all service dependencies."""
    with patch("src.web_interface.dependencies.get_web_analytics_service") as mock_analytics, \
         patch("src.web_interface.dependencies.get_web_capital_service") as mock_capital, \
         patch("src.web_interface.dependencies.get_web_data_service") as mock_data, \
         patch("src.web_interface.dependencies.get_web_exchange_service") as mock_exchange:
        
        # Configure mock services
        mock_analytics.return_value = create_mock_analytics_service()
        mock_capital.return_value = create_mock_capital_service()
        mock_data.return_value = create_mock_data_service()
        mock_exchange.return_value = create_mock_exchange_service()
        
        yield {
            "analytics": mock_analytics,
            "capital": mock_capital,
            "data": mock_data,
            "exchange": mock_exchange,
        }


def create_mock_analytics_service():
    """Create mock analytics service."""
    service = AsyncMock()
    
    # Portfolio metrics
    service.get_portfolio_metrics.return_value = {
        "total_value": "100000.00",
        "total_pnl": "5000.00",
        "total_pnl_percentage": 5.0,
        "win_rate": 0.65,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.15,
        "positions_count": 10,
        "active_strategies": 3,
        "timestamp": datetime.utcnow(),
    }
    
    # Risk metrics
    service.get_risk_metrics.return_value = {
        "portfolio_var": {
            "var_95": "1000.00",
            "var_99": "1500.00",
        },
        "portfolio_volatility": 0.2,
        "portfolio_beta": 1.1,
        "correlation_risk": 0.3,
        "concentration_risk": 0.25,
        "leverage_ratio": 1.5,
        "margin_usage": 0.4,
        "timestamp": datetime.utcnow(),
    }
    
    # Active alerts
    service.get_active_alerts.return_value = [
        {
            "id": "alert_001",
            "severity": "medium",
            "title": "High volatility detected",
            "message": "Market volatility exceeds threshold",
            "metric_name": "volatility",
            "created_at": datetime.utcnow(),
            "acknowledged": False,
        }
    ]
    
    return service


def create_mock_capital_service():
    """Create mock capital service."""
    service = AsyncMock()
    
    # Capital allocation
    service.allocate_capital.return_value = {
        "allocation_id": "alloc_123",
        "strategy_id": "strategy_001",
        "exchange": "binance",
        "allocated_amount": "10000.00",
        "utilized_amount": "0.00",
        "available_amount": "10000.00",
        "utilization_ratio": 0.0,
        "created_at": datetime.utcnow(),
        "last_updated": datetime.utcnow(),
        "status": "active",
    }
    
    # Capital metrics
    service.get_capital_metrics.return_value = {
        "total_capital": "1000000.00",
        "allocated_capital": "500000.00",
        "utilized_capital": "300000.00",
        "available_capital": "500000.00",
        "allocation_ratio": 0.5,
        "utilization_ratio": 0.3,
        "currency": "USD",
        "last_updated": datetime.utcnow(),
    }
    
    # Release capital
    service.release_capital.return_value = True
    
    return service


def create_mock_data_service():
    """Create mock data service."""
    service = AsyncMock()
    
    # Pipeline status
    service.get_pipeline_status.return_value = [
        {
            "pipeline_id": "market_data_pipeline",
            "status": "running",
            "uptime_seconds": 86400,
            "messages_processed": 1000000,
            "error_count": 10,
            "latency_ms": 15.5,
            "throughput_per_second": 1250,
            "last_error": None,
            "last_updated": datetime.utcnow(),
        }
    ]
    
    # Data quality metrics
    service.get_data_quality_metrics.return_value = {
        "completeness": 0.99,
        "accuracy": 0.98,
        "consistency": 0.97,
        "timeliness": 0.99,
        "validity": 0.98,
        "uniqueness": 1.0,
        "total_records": 1000000,
        "valid_records": 980000,
        "invalid_records": 20000,
        "missing_fields": {"volume": 100},
        "timestamp": datetime.utcnow(),
    }
    
    # Features
    service.list_features.return_value = [
        {
            "feature_id": "rsi_14",
            "name": "RSI 14",
            "category": "technical",
            "description": "Relative Strength Index",
            "active": True,
        }
    ]
    
    return service


def create_mock_exchange_service():
    """Create mock exchange service."""
    service = AsyncMock()
    
    # Connections
    service.get_connections.return_value = [
        {
            "exchange": "binance",
            "status": "connected",
            "test_mode": False,
            "created_at": datetime.utcnow(),
        }
    ]
    
    # Exchange status
    service.get_exchange_status.return_value = {
        "exchange": "binance",
        "status": "connected",
        "connected": True,
        "uptime_seconds": 86400,
        "last_heartbeat": datetime.utcnow(),
        "active_connections": 2,
        "error_count": 5,
        "latency_ms": 25.5,
    }
    
    # Exchange health
    service.get_exchange_health.return_value = {
        "exchange": "binance",
        "health_score": 0.95,
        "status": "healthy",
        "checks": {
            "api_connection": True,
            "websocket_connection": True,
            "rate_limits": True,
        },
        "issues": [],
        "last_check": datetime.utcnow(),
    }
    
    return service


class TestAnalyticsAPI:
    """Test Analytics API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_portfolio_metrics(self, client, auth_headers, mock_services):
        """Test getting portfolio metrics."""
        response = client.get("/api/analytics/portfolio/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_value" in data
        assert "total_pnl" in data
        assert "win_rate" in data
        assert "sharpe_ratio" in data
        
        # Verify Decimal values are strings
        assert isinstance(data["total_value"], str)
        assert isinstance(data["total_pnl"], str)
        
        # Verify service was called
        mock_services["analytics"].return_value.get_portfolio_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_risk_metrics(self, client, auth_headers, mock_services):
        """Test getting risk metrics."""
        response = client.get("/api/analytics/risk/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "portfolio_var" in data
        assert "portfolio_volatility" in data
        assert "leverage_ratio" in data
    
    @pytest.mark.asyncio
    async def test_calculate_var(self, client, auth_headers, mock_services):
        """Test VaR calculation."""
        request_data = {
            "confidence_level": 0.95,
            "time_horizon": 1,
            "method": "historical",
        }
        
        mock_services["analytics"].return_value.calculate_var.return_value = {
            "var_95": "1000.00",
        }
        
        response = client.post(
            "/api/analytics/risk/var",
            json=request_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "var_results" in data
    
    @pytest.mark.asyncio
    async def test_get_active_alerts(self, client, auth_headers, mock_services):
        """Test getting active alerts."""
        response = client.get("/api/analytics/alerts/active", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "alerts" in data
        assert "count" in data
        assert len(data["alerts"]) > 0
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, client, auth_headers, mock_services):
        """Test acknowledging an alert."""
        mock_services["analytics"].return_value.acknowledge_alert.return_value = True
        
        request_data = {
            "acknowledged_by": "test_user",
            "notes": "Alert reviewed",
        }
        
        response = client.post(
            "/api/analytics/alerts/acknowledge/alert_001",
            json=request_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"


class TestCapitalManagementAPI:
    """Test Capital Management API endpoints."""
    
    @pytest.mark.asyncio
    async def test_allocate_capital(self, client, auth_headers, mock_services):
        """Test capital allocation."""
        request_data = {
            "strategy_id": "strategy_001",
            "exchange": "binance",
            "amount": "10000.00",
            "bot_id": "bot_001",
        }
        
        response = client.post(
            "/api/capital/allocate",
            json=request_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "allocation_id" in data
        assert "allocated_amount" in data
        assert data["allocated_amount"] == "10000.00"
    
    @pytest.mark.asyncio
    async def test_release_capital(self, client, auth_headers, mock_services):
        """Test capital release."""
        request_data = {
            "strategy_id": "strategy_001",
            "exchange": "binance",
            "amount": "5000.00",
            "reason": "Strategy stopped",
        }
        
        response = client.post(
            "/api/capital/release",
            json=request_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_get_capital_metrics(self, client, auth_headers, mock_services):
        """Test getting capital metrics."""
        response = client.get("/api/capital/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_capital" in data
        assert "allocated_capital" in data
        assert "utilization_ratio" in data
        
        # Verify Decimal precision
        assert isinstance(data["total_capital"], str)
        assert "." in data["total_capital"]
    
    @pytest.mark.asyncio
    async def test_invalid_amount_validation(self, client, auth_headers):
        """Test validation of invalid amounts."""
        request_data = {
            "strategy_id": "strategy_001",
            "exchange": "binance",
            "amount": "-1000.00",  # Negative amount
        }
        
        response = client.post(
            "/api/capital/allocate",
            json=request_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error


class TestDataManagementAPI:
    """Test Data Management API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_pipeline_status(self, client, auth_headers, mock_services):
        """Test getting pipeline status."""
        response = client.get("/api/data/pipeline/status", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "pipelines" in data
        assert len(data["pipelines"]) > 0
        
        pipeline = data["pipelines"][0]
        assert "pipeline_id" in pipeline
        assert "status" in pipeline
        assert "throughput_per_second" in pipeline
    
    @pytest.mark.asyncio
    async def test_get_data_quality_metrics(self, client, auth_headers, mock_services):
        """Test getting data quality metrics."""
        response = client.get("/api/data/quality/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "completeness" in data
        assert "accuracy" in data
        assert "validity" in data
        assert data["completeness"] >= 0.0 and data["completeness"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_list_features(self, client, auth_headers, mock_services):
        """Test listing features."""
        response = client.get("/api/data/features/list", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "features" in data
        assert "count" in data
        
        if data["count"] > 0:
            feature = data["features"][0]
            assert "feature_id" in feature
            assert "name" in feature
            assert "category" in feature


class TestExchangeManagementAPI:
    """Test Exchange Management API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_connections(self, client, auth_headers, mock_services):
        """Test getting exchange connections."""
        response = client.get("/api/exchanges/connections", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "connections" in data
        assert "count" in data
    
    @pytest.mark.asyncio
    async def test_get_exchange_status(self, client, auth_headers, mock_services):
        """Test getting exchange status."""
        response = client.get("/api/exchanges/binance/status", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "exchange" in data
        assert "status" in data
        assert "connected" in data
        assert "latency_ms" in data
    
    @pytest.mark.asyncio
    async def test_get_exchange_health(self, client, auth_headers, mock_services):
        """Test getting exchange health."""
        response = client.get("/api/exchanges/binance/health", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "health_score" in data
        assert "status" in data
        assert "checks" in data
        assert data["health_score"] >= 0.0 and data["health_score"] <= 1.0


class TestAuthorizationAndSecurity:
    """Test authorization and security aspects."""
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, client):
        """Test that endpoints require authentication."""
        # No auth headers
        response = client.get("/api/analytics/portfolio/metrics")
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_insufficient_permissions(self, client):
        """Test role-based access control."""
        # User role instead of admin
        headers = {
            "Authorization": "Bearer test_token",
            "X-User-ID": "test_user",
            "X-User-Role": "viewer",
        }
        
        # Try to allocate capital (requires admin/trader)
        request_data = {
            "strategy_id": "strategy_001",
            "exchange": "binance",
            "amount": "10000.00",
        }
        
        response = client.post(
            "/api/capital/allocate",
            json=request_data,
            headers=headers,
        )
        
        assert response.status_code == 403


class TestFinancialPrecision:
    """Test financial precision and Decimal usage."""
    
    @pytest.mark.asyncio
    async def test_decimal_precision_in_responses(self, client, auth_headers, mock_services):
        """Test that all financial values use proper decimal precision."""
        response = client.get("/api/capital/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that financial values are strings (for Decimal precision)
        assert isinstance(data["total_capital"], str)
        assert isinstance(data["allocated_capital"], str)
        
        # Verify decimal places
        assert "." in data["total_capital"]
        assert len(data["total_capital"].split(".")[1]) == 2  # Two decimal places
    
    @pytest.mark.asyncio
    async def test_decimal_validation_in_requests(self, client, auth_headers):
        """Test that invalid decimal values are rejected."""
        request_data = {
            "strategy_id": "strategy_001",
            "exchange": "binance",
            "amount": "not_a_number",  # Invalid decimal
        }
        
        response = client.post(
            "/api/capital/allocate",
            json=request_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Test error handling across APIs."""
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self, client, auth_headers, mock_services):
        """Test that service errors are handled properly."""
        # Make service throw an error
        mock_services["analytics"].return_value.get_portfolio_metrics.side_effect = Exception(
            "Service unavailable"
        )
        
        response = client.get("/api/analytics/portfolio/metrics", headers=auth_headers)
        
        assert response.status_code == 500
        assert "detail" in response.json()
    
    @pytest.mark.asyncio
    async def test_not_found_handling(self, client, auth_headers, mock_services):
        """Test 404 error handling."""
        mock_services["analytics"].return_value.get_strategy_metrics.return_value = None
        
        response = client.get("/api/analytics/strategy/nonexistent/metrics", headers=auth_headers)
        
        assert response.status_code == 404


class TestConcurrentRequests:
    """Test handling of concurrent requests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_api_calls(self, async_client, auth_headers, mock_services):
        """Test that APIs handle concurrent requests properly."""
        # Make 10 concurrent requests
        tasks = [
            async_client.get("/api/analytics/portfolio/metrics", headers=auth_headers)
            for _ in range(10)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_full_integration_flow(client, auth_headers, mock_services):
    """Test a complete integration flow across multiple APIs."""
    # 1. Check capital metrics
    response = client.get("/api/capital/metrics", headers=auth_headers)
    assert response.status_code == 200
    capital_data = response.json()
    
    # 2. Allocate capital
    allocation_request = {
        "strategy_id": "strategy_001",
        "exchange": "binance",
        "amount": "10000.00",
    }
    response = client.post("/api/capital/allocate", json=allocation_request, headers=auth_headers)
    assert response.status_code == 200
    
    # 3. Check exchange status
    response = client.get("/api/exchanges/binance/status", headers=auth_headers)
    assert response.status_code == 200
    
    # 4. Check data pipeline
    response = client.get("/api/data/pipeline/status", headers=auth_headers)
    assert response.status_code == 200
    
    # 5. Get portfolio metrics
    response = client.get("/api/analytics/portfolio/metrics", headers=auth_headers)
    assert response.status_code == 200
    
    # 6. Release capital
    release_request = {
        "strategy_id": "strategy_001",
        "exchange": "binance",
        "amount": "10000.00",
    }
    response = client.post("/api/capital/release", json=release_request, headers=auth_headers)
    assert response.status_code == 200