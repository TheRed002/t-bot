#!/usr/bin/env python3
"""
Quick API Status Check for T-Bot Trading System.

Fast endpoint testing to determine what's working vs broken.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set mock mode for testing
os.environ["MOCK_MODE"] = "true"

from fastapi.testclient import TestClient
from src.core.config import Config
from src.web_interface.app import create_app


def quick_api_test():
    """Quick test of key API endpoints."""
    print("🚀 Quick API Status Check")
    print("=" * 50)
    
    # Initialize app
    config = Config()
    config.environment = "test"
    app = create_app(config)
    client = TestClient(app)
    
    # Key endpoints to test
    endpoints = {
        "Health": "/health",
        "Root": "/",
        "Analytics Portfolio": "/api/analytics/portfolio/metrics",
        "Analytics Risk": "/api/analytics/risk/metrics",
        "Bot Status": "/api/bot/status",
        "Bot List": "/api/bot/list", 
        "Portfolio Summary": "/api/portfolio/summary",
        "Trading Orders": "/api/trading/orders",
        "Strategies List": "/api/strategies/list",
        "Risk Metrics": "/api/risk/metrics",
        "Monitoring Health": "/api/monitoring/health",
        "Capital Balance": "/api/capital/balance",
        "Data Market": "/api/data/market",
        "Exchanges List": "/api/exchanges/list",
        "Auth Login": "/api/auth/login"
    }
    
    # Test results
    working = []
    auth_required = []
    not_implemented = []
    errors = []
    
    for name, endpoint in endpoints.items():
        try:
            if endpoint == "/api/auth/login":
                # Test login with credentials
                response = client.post(endpoint, json={"username": "test", "password": "test"})
            else:
                response = client.get(endpoint)
                
            status = response.status_code
            
            if status == 200:
                working.append(f"✅ {name}")
                print(f"✅ {name:<20} WORKING (200)")
            elif status in [401, 403]:
                auth_required.append(f"🔐 {name}")
                print(f"🔐 {name:<20} AUTH REQUIRED ({status})")
            elif status == 404:
                not_implemented.append(f"❌ {name}")
                print(f"❌ {name:<20} NOT IMPLEMENTED (404)")
            elif status == 422:
                print(f"📝 {name:<20} VALIDATION ERROR (422)")
                if "login" in endpoint:
                    auth_required.append(f"🔐 {name}")
                else:
                    errors.append(f"📝 {name}")
            else:
                errors.append(f"⚠️  {name} ({status})")
                print(f"⚠️  {name:<20} OTHER ({status})")
                
        except Exception as e:
            errors.append(f"💥 {name}: {str(e)[:50]}")
            print(f"💥 {name:<20} ERROR: {str(e)[:50]}")
    
    # Summary
    total = len(endpoints)
    print(f"\n📊 SUMMARY")
    print("-" * 30)
    print(f"Total Endpoints: {total}")
    print(f"Working: {len(working)}")
    print(f"Auth Required: {len(auth_required)}")
    print(f"Not Implemented: {len(not_implemented)}")
    print(f"Errors: {len(errors)}")
    print(f"Success Rate: {len(working)/total*100:.1f}%")
    print(f"Implementation Rate: {(len(working)+len(auth_required))/total*100:.1f}%")
    
    return {
        "working": working,
        "auth_required": auth_required,
        "not_implemented": not_implemented,
        "errors": errors,
        "total": total
    }


if __name__ == "__main__":
    try:
        results = quick_api_test()
        print(f"\n🎯 Quick test completed successfully!")
    except Exception as e:
        print(f"💥 Quick test failed: {e}")
        import traceback
        traceback.print_exc()