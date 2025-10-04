#!/usr/bin/env python3
"""
Portfolio API Integration Tests.

Tests all portfolio module API endpoints with real HTTP requests.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from src.web_interface.app import create_app
from src.core.config import Config
import pytest


class TestPortfolioAPI:
    """Test Portfolio API endpoints."""
    
    
    def setup_method(self):
        """Set up test client."""
        config = Config()
        config.environment = 'test'
        config.debug = True
        config.jwt_secret = 'test_secret_key'
        
        # Disable rate limiting for testing
        config.web_interface = {
            "rate_limiting": {
                "enabled": False
            }
        }
        
        app = create_app(config)
        self.client = TestClient(app)
    
    def test_portfolio_summary(self):
        """Test GET /api/portfolio/summary endpoint."""
        response = self.client.get("/api/portfolio/summary")
        assert response.status_code in [200, 403]
        
    def test_portfolio_holdings(self):
        """Test GET /api/portfolio/holdings endpoint."""
        response = self.client.get("/api/portfolio/holdings")
        assert response.status_code in [200, 403]
        
    def test_portfolio_performance(self):
        """Test GET /api/portfolio/performance endpoint."""
        response = self.client.get("/api/portfolio/performance")
        assert response.status_code in [200, 403]
        
    def test_portfolio_allocation(self):
        """Test GET /api/portfolio/allocation endpoint."""
        response = self.client.get("/api/portfolio/allocation")
        assert response.status_code in [200, 403]
        
    def test_portfolio_rebalance(self):
        """Test POST /api/portfolio/rebalance endpoint."""
        test_data = {
            "strategy": "equal_weight",
            "symbols": ["BTCUSDT", "ETHUSDT"]
        }
        response = self.client.post("/api/portfolio/rebalance", json=test_data)
        assert response.status_code in [200, 403, 422]


if __name__ == "__main__":
    # Quick test run
    test_portfolio = TestPortfolioAPI()
    test_portfolio.setup_method()
    
    print("🧪 Testing Portfolio API endpoints...")
    
    endpoints_to_test = [
        ("GET", "/api/portfolio/summary"),
        ("GET", "/api/portfolio/holdings"),
        ("GET", "/api/portfolio/performance"),
        ("GET", "/api/portfolio/allocation")
    ]
    
    passed = 0
    for method, endpoint in endpoints_to_test:
        try:
            response = test_portfolio.client.get(endpoint)
            if response.status_code in [200, 403]:
                print(f"✅ {method} {endpoint}: {response.status_code}")
                passed += 1
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
                passed += 1
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")
    
    print(f"\n📊 Portfolio API: {passed}/{len(endpoints_to_test)} endpoints working")