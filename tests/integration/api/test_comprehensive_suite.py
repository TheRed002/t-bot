#!/usr/bin/env python3
"""
Comprehensive API Test Suite for T-Bot Trading System.

This script tests all API endpoints with proper authentication,
creating detailed status reports and identifying working vs broken endpoints.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set mock mode for testing
os.environ["MOCK_MODE"] = "true"

from fastapi.testclient import TestClient
from src.core.config import Config
from src.web_interface.app import create_app
from src.web_interface.security.jwt_handler import JWTHandler


class APITester:
    """Comprehensive API testing utility."""
    
    def __init__(self):
        """Initialize the API tester."""
        self.config = Config()
        self.config.environment = "test"
        self.config.debug = True
        
        # Create app and client
        try:
            self.app = create_app(self.config)
            self.client = TestClient(self.app)
            self.jwt_handler = JWTHandler(self.config)
            print("✅ App and client initialized successfully")
        except Exception as e:
            print(f"💥 Failed to initialize app: {e}")
            raise
            
        # Test results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_endpoints": 0,
            "working": [],
            "auth_required": [],
            "not_implemented": [],
            "errors": [],
            "summary": {}
        }
        
        # Define all API endpoints to test
        self.endpoints = self._get_all_endpoints()
        
    def _get_all_endpoints(self) -> Dict[str, List[str]]:
        """Define all API endpoints organized by category."""
        return {
            "health": [
                "/health",
                "/",
            ],
            "auth": [
                "/api/auth/login",
                "/api/auth/refresh", 
                "/api/auth/logout",
                "/api/auth/users",
            ],
            "analytics": [
                "/api/analytics/portfolio/metrics",
                "/api/analytics/portfolio/composition",
                "/api/analytics/portfolio/correlation",
                "/api/analytics/risk/metrics",
                "/api/analytics/performance/attribution",
                "/api/analytics/reports",
                "/api/analytics/alerts",
            ],
            "bot_management": [
                "/api/bot/status",
                "/api/bot/list",
                "/api/bot/create",
                "/api/bot/start",
                "/api/bot/stop",
                "/api/bot/config",
                "/api/bot/logs",
            ],
            "portfolio": [
                "/api/portfolio/summary",
                "/api/portfolio/positions",
                "/api/portfolio/history",
                "/api/portfolio/performance",
                "/api/portfolio/allocation",
            ],
            "trading": [
                "/api/trading/orders",
                "/api/trading/orders/active", 
                "/api/trading/trades",
                "/api/trading/positions",
                "/api/trading/balance",
            ],
            "strategies": [
                "/api/strategies/list",
                "/api/strategies/create",
                "/api/strategies/status",
                "/api/strategies/performance",
                "/api/strategies/config",
            ],
            "risk": [
                "/api/risk/metrics",
                "/api/risk/limits",
                "/api/risk/violations",
                "/api/risk/stress-test",
            ],
            "monitoring": [
                "/api/monitoring/system",
                "/api/monitoring/metrics",
                "/api/monitoring/alerts",
                "/api/monitoring/health",
            ],
            "capital": [
                "/api/capital/balance",
                "/api/capital/allocation",
                "/api/capital/transfers",
                "/api/capital/history",
            ],
            "data": [
                "/api/data/market",
                "/api/data/symbols",
                "/api/data/historical",
                "/api/data/status",
            ],
            "exchanges": [
                "/api/exchanges/list",
                "/api/exchanges/status",
                "/api/exchanges/balance",
                "/api/exchanges/orders",
            ],
        }
    
    def create_test_token(self) -> str:
        """Create a test JWT token for authentication."""
        try:
            # Create test user payload
            test_user = {
                "user_id": "test_user_123",
                "username": "test_user", 
                "email": "test@example.com",
                "scopes": ["read", "trade", "admin"],
                "is_active": True
            }
            
            # Generate token with 1 hour expiry
            token = self.jwt_handler.create_access_token(
                user_id=test_user["user_id"],
                username=test_user["username"],
                scopes=test_user["scopes"]
            )
            print("✅ Created test JWT token")
            return token
            
        except Exception as e:
            print(f"⚠️  Failed to create JWT token: {e}")
            # Return a simple bearer token for basic testing
            return "test_bearer_token"
    
    def test_endpoint(self, endpoint: str, method: str = "GET", 
                     auth_token: Optional[str] = None) -> Tuple[int, str, Optional[Dict]]:
        """
        Test a single endpoint.
        
        Args:
            endpoint: The endpoint path to test
            method: HTTP method (GET, POST, etc.)
            auth_token: Optional authentication token
            
        Returns:
            Tuple of (status_code, status_description, response_data)
        """
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            
        try:
            if method == "GET":
                response = self.client.get(endpoint, headers=headers)
            elif method == "POST":
                # Use minimal test data for POST requests
                test_data = self._get_test_data_for_endpoint(endpoint)
                response = self.client.post(endpoint, json=test_data, headers=headers)
            else:
                return 405, "Method not supported in test", None
                
            status_code = response.status_code
            
            try:
                response_data = response.json() if response.content else None
            except:
                response_data = None
                
            if status_code == 200:
                return status_code, "✅ Working", response_data
            elif status_code in [401, 403]:
                return status_code, "🔐 Auth Required", response_data
            elif status_code == 404:
                return status_code, "❌ Not Implemented", response_data
            elif status_code == 422:
                return status_code, "📝 Validation Error", response_data
            elif status_code >= 500:
                return status_code, "💥 Server Error", response_data
            else:
                return status_code, f"⚠️  Other ({status_code})", response_data
                
        except Exception as e:
            return 0, f"💥 Exception: {str(e)[:100]}", None
    
    def _get_test_data_for_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """Get appropriate test data for POST endpoints."""
        if "login" in endpoint:
            return {"username": "test_user", "password": "test_password"}
        elif "create" in endpoint and "bot" in endpoint:
            return {"name": "test_bot", "strategy": "test_strategy"}
        elif "create" in endpoint and "user" in endpoint:
            return {"username": "new_user", "email": "new@example.com", "password": "password123"}
        else:
            return {"test": "data"}
    
    def test_all_endpoints(self) -> None:
        """Test all defined endpoints and generate comprehensive report."""
        print("🚀 Starting comprehensive API endpoint testing...")
        print("=" * 80)
        
        # Create test token
        auth_token = self.create_test_token()
        
        total_tested = 0
        
        for category, endpoints in self.endpoints.items():
            print(f"\n📂 Testing {category.upper()} endpoints:")
            print("-" * 40)
            
            for endpoint in endpoints:
                total_tested += 1
                
                # Test without auth first
                status_code, status_desc, response_data = self.test_endpoint(endpoint)
                
                # If auth required, test with token
                if status_code in [401, 403]:
                    auth_status_code, auth_status_desc, auth_response_data = self.test_endpoint(
                        endpoint, auth_token=auth_token
                    )
                    
                    if auth_status_code == 200:
                        status_code, status_desc, response_data = auth_status_code, auth_status_desc, auth_response_data
                        self.results["working"].append({
                            "endpoint": endpoint,
                            "category": category,
                            "status_code": status_code,
                            "auth_required": True,
                            "response_sample": str(response_data)[:200] if response_data else None
                        })
                    else:
                        self.results["auth_required"].append({
                            "endpoint": endpoint,
                            "category": category, 
                            "status_code": auth_status_code,
                            "status_desc": auth_status_desc,
                            "error": str(auth_response_data) if auth_response_data else None
                        })
                else:
                    # Categorize results
                    if status_code == 200:
                        self.results["working"].append({
                            "endpoint": endpoint,
                            "category": category,
                            "status_code": status_code,
                            "auth_required": False,
                            "response_sample": str(response_data)[:200] if response_data else None
                        })
                    elif status_code == 404:
                        self.results["not_implemented"].append({
                            "endpoint": endpoint,
                            "category": category,
                            "status_code": status_code
                        })
                    else:
                        self.results["errors"].append({
                            "endpoint": endpoint,
                            "category": category,
                            "status_code": status_code,
                            "error": status_desc,
                            "response": str(response_data) if response_data else None
                        })
                
                print(f"  {endpoint:<40} {status_desc}")
        
        self.results["total_endpoints"] = total_tested
        self._generate_summary()
        self._print_final_report()
        
    def _generate_summary(self) -> None:
        """Generate summary statistics."""
        total = self.results["total_endpoints"]
        working = len(self.results["working"])
        auth_required = len(self.results["auth_required"])
        not_implemented = len(self.results["not_implemented"])
        errors = len(self.results["errors"])
        
        self.results["summary"] = {
            "total_endpoints": total,
            "working_endpoints": working,
            "auth_required_endpoints": auth_required,
            "not_implemented_endpoints": not_implemented,
            "error_endpoints": errors,
            "success_rate": f"{(working / total * 100):.1f}%" if total > 0 else "0%",
            "implementation_rate": f"{((working + auth_required) / total * 100):.1f}%" if total > 0 else "0%"
        }
    
    def _print_final_report(self) -> None:
        """Print comprehensive final report."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE API TEST RESULTS")
        print("=" * 80)
        
        summary = self.results["summary"]
        print(f"📈 Total Endpoints Tested: {summary['total_endpoints']}")
        print(f"✅ Working Endpoints: {summary['working_endpoints']}")
        print(f"🔐 Auth Required: {summary['auth_required_endpoints']}")
        print(f"❌ Not Implemented: {summary['not_implemented_endpoints']}")
        print(f"💥 Error Endpoints: {summary['error_endpoints']}")
        print(f"🎯 Success Rate: {summary['success_rate']}")
        print(f"📋 Implementation Rate: {summary['implementation_rate']}")
        
        # Working endpoints details
        if self.results["working"]:
            print(f"\n✅ WORKING ENDPOINTS ({len(self.results['working'])})")
            print("-" * 50)
            for item in self.results["working"]:
                auth_note = " (Auth Required)" if item["auth_required"] else ""
                print(f"  {item['endpoint']}{auth_note}")
        
        # Auth required endpoints
        if self.results["auth_required"]:
            print(f"\n🔐 AUTH REQUIRED BUT ACCESSIBLE ({len(self.results['auth_required'])})")
            print("-" * 50)
            for item in self.results["auth_required"]:
                print(f"  {item['endpoint']} - {item['status_desc']}")
        
        # Not implemented endpoints
        if self.results["not_implemented"]:
            print(f"\n❌ NOT IMPLEMENTED ({len(self.results['not_implemented'])})")
            print("-" * 50)
            for item in self.results["not_implemented"]:
                print(f"  {item['endpoint']}")
        
        # Error endpoints
        if self.results["errors"]:
            print(f"\n💥 ERROR ENDPOINTS ({len(self.results['errors'])})")
            print("-" * 50)
            for item in self.results["errors"]:
                print(f"  {item['endpoint']} - {item['error']}")
        
        # Save results to file
        results_file = project_root / ".claude_experiments" / "api_test_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")


def main():
    """Main test execution."""
    try:
        tester = APITester()
        tester.test_all_endpoints()
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()