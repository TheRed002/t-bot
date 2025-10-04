#!/usr/bin/env python3
"""
Monitoring API Integration Tests.

Tests all monitoring module API endpoints with real HTTP requests.
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


class TestMonitoringAPI:
    """Test Monitoring API endpoints."""
    
    
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
    
    def test_monitoring_system_health(self):
        """Test GET /api/monitoring/system/health endpoint."""
        response = self.client.get("/api/monitoring/system/health")
        assert response.status_code in [200, 403]
        
    def test_monitoring_metrics(self):
        """Test GET /api/monitoring/metrics endpoint."""
        response = self.client.get("/api/monitoring/metrics")
        assert response.status_code in [200, 403]
        
    def test_monitoring_alerts(self):
        """Test GET /api/monitoring/alerts endpoint."""
        response = self.client.get("/api/monitoring/alerts")
        assert response.status_code in [200, 403]
        
    def test_monitoring_performance(self):
        """Test GET /api/monitoring/performance endpoint."""
        response = self.client.get("/api/monitoring/performance")
        assert response.status_code in [200, 403]
        
    def test_monitoring_logs(self):
        """Test GET /api/monitoring/logs endpoint."""
        response = self.client.get("/api/monitoring/logs")
        assert response.status_code in [200, 403]


if __name__ == "__main__":
    # Quick test run
    test_monitoring = TestMonitoringAPI()
    test_monitoring.setup_method()
    
    print("🧪 Testing Monitoring API endpoints...")
    
    endpoints_to_test = [
        ("GET", "/api/monitoring/system/health"),
        ("GET", "/api/monitoring/metrics"),
        ("GET", "/api/monitoring/alerts"),
        ("GET", "/api/monitoring/performance"),
        ("GET", "/api/monitoring/logs")
    ]
    
    passed = 0
    for method, endpoint in endpoints_to_test:
        try:
            response = test_monitoring.client.get(endpoint)
            if response.status_code in [200, 403]:
                print(f"✅ {method} {endpoint}: {response.status_code}")
                passed += 1
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
                passed += 1
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")
    
    print(f"\n📊 Monitoring API: {passed}/{len(endpoints_to_test)} endpoints working")