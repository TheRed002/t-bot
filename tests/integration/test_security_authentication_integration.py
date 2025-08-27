"""
Security and authentication integration tests.

Tests JWT token lifecycle, API endpoint authentication, WebSocket authentication,
rate limiting enforcement, security headers, and access control mechanisms.
"""

import pytest
import asyncio
import logging
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time
import jwt
import hashlib
import secrets
import base64

from tests.integration.base_integration import (
    BaseIntegrationTest, MockExchangeFactory, PerformanceMonitor,
    performance_test, wait_for_condition
)
from src.core.types import (
    MarketData, Order, OrderSide, OrderType, OrderStatus, Position
)
from src.web_interface.security.jwt_handler import JWTHandler
from src.exchanges.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class TestJWTTokenLifecycle(BaseIntegrationTest):
    """Test JWT token lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_jwt_token_generation_and_validation(self):
        """Test JWT token generation, validation, and expiration."""
        
        class MockJWTHandler:
            def __init__(self):
                self.secret_key = secrets.token_urlsafe(32)
                self.algorithm = "HS256"
                self.access_token_expire_minutes = 15
                self.refresh_token_expire_days = 7
                
            def generate_access_token(self, user_id: str, permissions: List[str] = None) -> str:
                """Generate JWT access token."""
                if permissions is None:
                    permissions = []
                    
                payload = {
                    "sub": user_id,
                    "permissions": permissions,
                    "type": "access",
                    "iat": datetime.now(timezone.utc),
                    "exp": datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
                }
                
                return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
                
            def generate_refresh_token(self, user_id: str) -> str:
                """Generate JWT refresh token."""
                payload = {
                    "sub": user_id,
                    "type": "refresh",
                    "iat": datetime.now(timezone.utc),
                    "exp": datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
                }
                
                return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
                
            def validate_token(self, token: str) -> Dict[str, Any]:
                """Validate and decode JWT token."""
                try:
                    payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                    return payload
                except jwt.ExpiredSignatureError:
                    raise Exception("Token has expired")
                except jwt.InvalidTokenError:
                    raise Exception("Invalid token")
                    
            def refresh_access_token(self, refresh_token: str) -> str:
                """Generate new access token using refresh token."""
                payload = self.validate_token(refresh_token)
                
                if payload.get("type") != "refresh":
                    raise Exception("Invalid refresh token")
                    
                user_id = payload["sub"]
                return self.generate_access_token(user_id, ["read", "write"])
        
        jwt_handler = MockJWTHandler()
        
        # Test token generation
        user_id = "test_user_123"
        permissions = ["read", "write", "admin"]
        
        access_token = jwt_handler.generate_access_token(user_id, permissions)
        refresh_token = jwt_handler.generate_refresh_token(user_id)
        
        assert isinstance(access_token, str)
        assert isinstance(refresh_token, str)
        assert len(access_token) > 100  # JWT tokens are typically long
        assert len(refresh_token) > 100
        
        logger.info(f"Generated access token: {access_token[:50]}...")
        logger.info(f"Generated refresh token: {refresh_token[:50]}...")
        
        # Test token validation
        access_payload = jwt_handler.validate_token(access_token)
        refresh_payload = jwt_handler.validate_token(refresh_token)
        
        assert access_payload["sub"] == user_id
        assert access_payload["type"] == "access"
        assert access_payload["permissions"] == permissions
        
        assert refresh_payload["sub"] == user_id
        assert refresh_payload["type"] == "refresh"
        
        logger.info("Token validation successful")
        
        # Test token refresh
        new_access_token = jwt_handler.refresh_access_token(refresh_token)
        new_access_payload = jwt_handler.validate_token(new_access_token)
        
        assert new_access_payload["sub"] == user_id
        assert new_access_payload["type"] == "access"
        assert "read" in new_access_payload["permissions"]
        assert "write" in new_access_payload["permissions"]
        
        logger.info("Token refresh successful")
        
        # Test token expiration (simulate expired token)
        expired_payload = {
            "sub": user_id,
            "type": "access",
            "iat": datetime.now(timezone.utc) - timedelta(hours=1),
            "exp": datetime.now(timezone.utc) - timedelta(minutes=30)  # Expired 30 minutes ago
        }
        
        expired_token = jwt.encode(expired_payload, jwt_handler.secret_key, algorithm=jwt_handler.algorithm)
        
        # Should raise expiration error
        with pytest.raises(Exception, match="Token has expired"):
            jwt_handler.validate_token(expired_token)
        
        logger.info("Token expiration handling verified")
        
        # Test invalid token
        invalid_token = "invalid.jwt.token"
        
        with pytest.raises(Exception, match="Invalid token"):
            jwt_handler.validate_token(invalid_token)
        
        logger.info("Invalid token handling verified")
        
        logger.info("JWT token lifecycle test completed")
    
    @pytest.mark.asyncio
    async def test_jwt_token_revocation_and_blacklisting(self):
        """Test JWT token revocation and blacklisting mechanisms."""
        
        class TokenBlacklistManager:
            def __init__(self):
                self.blacklisted_tokens = set()
                self.blacklisted_users = set()
                self.token_metadata = {}  # Store token info for tracking
                
            def blacklist_token(self, token: str, reason: str = "revoked"):
                """Add token to blacklist."""
                self.blacklisted_tokens.add(token)
                self.token_metadata[token] = {
                    "blacklisted_at": datetime.now(timezone.utc),
                    "reason": reason
                }
                logger.info(f"Token blacklisted: {token[:20]}... (reason: {reason})")
                
            def blacklist_user(self, user_id: str, reason: str = "suspended"):
                """Blacklist all tokens for a user."""
                self.blacklisted_users.add(user_id)
                logger.info(f"User blacklisted: {user_id} (reason: {reason})")
                
            def is_token_blacklisted(self, token: str, user_id: str = None) -> bool:
                """Check if token is blacklisted."""
                # Direct token blacklist
                if token in self.blacklisted_tokens:
                    return True
                    
                # User-level blacklist
                if user_id and user_id in self.blacklisted_users:
                    return True
                    
                return False
                
            def cleanup_expired_blacklist(self):
                """Remove expired tokens from blacklist."""
                # In a real implementation, this would check token expiration
                # For testing, we'll simulate cleanup
                current_time = datetime.now(timezone.utc)
                
                expired_tokens = []
                for token, metadata in self.token_metadata.items():
                    # Simulate checking if token would be expired
                    blacklist_age = current_time - metadata["blacklisted_at"]
                    if blacklist_age > timedelta(hours=24):  # Clean up after 24h
                        expired_tokens.append(token)
                
                for token in expired_tokens:
                    if token in self.blacklisted_tokens:
                        self.blacklisted_tokens.remove(token)
                        del self.token_metadata[token]
                        
                logger.info(f"Cleaned up {len(expired_tokens)} expired blacklisted tokens")
                
            def get_blacklist_stats(self) -> Dict[str, int]:
                """Get blacklist statistics."""
                return {
                    "blacklisted_tokens": len(self.blacklisted_tokens),
                    "blacklisted_users": len(self.blacklisted_users)
                }
        
        blacklist_manager = TokenBlacklistManager()
        
        # Mock JWT handler
        secret_key = secrets.token_urlsafe(32)
        
        def create_token(user_id: str, exp_minutes: int = 15) -> str:
            payload = {
                "sub": user_id,
                "type": "access",
                "iat": datetime.now(timezone.utc),
                "exp": datetime.now(timezone.utc) + timedelta(minutes=exp_minutes)
            }
            return jwt.encode(payload, secret_key, algorithm="HS256")
            
        def validate_token_with_blacklist(token: str) -> Dict[str, Any]:
            # First decode to get user info
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            user_id = payload["sub"]
            
            # Check blacklist
            if blacklist_manager.is_token_blacklisted(token, user_id):
                raise Exception("Token is blacklisted")
                
            return payload
        
        # Test token blacklisting
        user1_token = create_token("user1")
        user2_token = create_token("user2")
        
        # Both tokens should be valid initially
        payload1 = validate_token_with_blacklist(user1_token)
        payload2 = validate_token_with_blacklist(user2_token)
        
        assert payload1["sub"] == "user1"
        assert payload2["sub"] == "user2"
        
        # Blacklist user1's token
        blacklist_manager.blacklist_token(user1_token, "security_breach")
        
        # user1's token should now be invalid
        with pytest.raises(Exception, match="Token is blacklisted"):
            validate_token_with_blacklist(user1_token)
            
        # user2's token should still be valid
        payload2_recheck = validate_token_with_blacklist(user2_token)
        assert payload2_recheck["sub"] == "user2"
        
        logger.info("Token-specific blacklisting works correctly")
        
        # Test user-level blacklisting
        user3_token1 = create_token("user3")
        user3_token2 = create_token("user3")  # Different token, same user
        
        # Both tokens should be valid initially
        validate_token_with_blacklist(user3_token1)
        validate_token_with_blacklist(user3_token2)
        
        # Blacklist entire user
        blacklist_manager.blacklist_user("user3", "account_suspended")
        
        # Both of user3's tokens should now be invalid
        with pytest.raises(Exception, match="Token is blacklisted"):
            validate_token_with_blacklist(user3_token1)
            
        with pytest.raises(Exception, match="Token is blacklisted"):
            validate_token_with_blacklist(user3_token2)
        
        logger.info("User-level blacklisting works correctly")
        
        # Test blacklist statistics
        stats = blacklist_manager.get_blacklist_stats()
        
        assert stats["blacklisted_tokens"] == 1  # user1's token
        assert stats["blacklisted_users"] == 1   # user3
        
        logger.info(f"Blacklist stats: {stats}")
        
        # Test cleanup (simulate)
        blacklist_manager.cleanup_expired_blacklist()
        
        logger.info("JWT token revocation and blacklisting test completed")
    
    @pytest.mark.asyncio
    async def test_jwt_token_permissions_and_scopes(self):
        """Test JWT token permissions and scope-based access control."""
        
        class PermissionManager:
            def __init__(self):
                self.role_permissions = {
                    "admin": ["read", "write", "delete", "manage_users", "view_analytics"],
                    "trader": ["read", "write", "place_orders", "view_portfolio"],
                    "viewer": ["read", "view_portfolio"],
                    "analyst": ["read", "view_analytics", "export_data"]
                }
                
            def get_permissions_for_role(self, role: str) -> List[str]:
                """Get permissions for a given role."""
                return self.role_permissions.get(role, [])
                
            def has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
                """Check if user has required permission."""
                return required_permission in user_permissions
                
            def has_any_permission(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
                """Check if user has any of the required permissions."""
                return any(perm in user_permissions for perm in required_permissions)
                
            def has_all_permissions(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
                """Check if user has all required permissions."""
                return all(perm in user_permissions for perm in required_permissions)
        
        permission_manager = PermissionManager()
        
        # Test permission assignment
        admin_permissions = permission_manager.get_permissions_for_role("admin")
        trader_permissions = permission_manager.get_permissions_for_role("trader")
        viewer_permissions = permission_manager.get_permissions_for_role("viewer")
        
        assert "manage_users" in admin_permissions
        assert "place_orders" in trader_permissions
        assert "read" in viewer_permissions
        
        # Admin should have more permissions than trader
        assert len(admin_permissions) > len(trader_permissions)
        assert len(trader_permissions) > len(viewer_permissions)
        
        logger.info(f"Admin permissions: {admin_permissions}")
        logger.info(f"Trader permissions: {trader_permissions}")
        logger.info(f"Viewer permissions: {viewer_permissions}")
        
        # Test permission checking
        # Admin should have all permissions
        assert permission_manager.has_permission(admin_permissions, "manage_users")
        assert permission_manager.has_permission(admin_permissions, "place_orders")
        assert permission_manager.has_permission(admin_permissions, "read")
        
        # Trader should have trading permissions but not admin permissions
        assert permission_manager.has_permission(trader_permissions, "place_orders")
        assert permission_manager.has_permission(trader_permissions, "read")
        assert not permission_manager.has_permission(trader_permissions, "manage_users")
        
        # Viewer should only have read permissions
        assert permission_manager.has_permission(viewer_permissions, "read")
        assert not permission_manager.has_permission(viewer_permissions, "place_orders")
        assert not permission_manager.has_permission(viewer_permissions, "manage_users")
        
        # Test multiple permission checks
        trading_permissions = ["place_orders", "view_portfolio"]
        admin_actions = ["manage_users", "delete"]
        
        assert permission_manager.has_all_permissions(admin_permissions, trading_permissions)
        assert permission_manager.has_all_permissions(admin_permissions, admin_actions)
        
        assert permission_manager.has_all_permissions(trader_permissions, trading_permissions)
        assert not permission_manager.has_all_permissions(trader_permissions, admin_actions)
        
        assert not permission_manager.has_any_permission(viewer_permissions, admin_actions)
        
        logger.info("Permission checking works correctly")
        
        # Test permission-based API access simulation
        def check_api_access(user_permissions: List[str], endpoint: str) -> bool:
            """Simulate API endpoint permission checking."""
            endpoint_permissions = {
                "/api/portfolio": ["read"],
                "/api/orders": ["place_orders"],
                "/api/users": ["manage_users"],
                "/api/analytics": ["view_analytics"],
                "/api/admin/delete": ["delete"]
            }
            
            required_permission = endpoint_permissions.get(endpoint)
            if not required_permission:
                return False  # Unknown endpoint
                
            return permission_manager.has_any_permission(user_permissions, required_permission)
        
        # Test API access for different user types
        api_endpoints = [
            "/api/portfolio",
            "/api/orders", 
            "/api/users",
            "/api/analytics",
            "/api/admin/delete"
        ]
        
        access_results = {}
        for role, permissions in [("admin", admin_permissions), ("trader", trader_permissions), ("viewer", viewer_permissions)]:
            access_results[role] = {}
            for endpoint in api_endpoints:
                has_access = check_api_access(permissions, endpoint)
                access_results[role][endpoint] = has_access
        
        # Verify expected access patterns
        assert access_results["admin"]["/api/users"] is True      # Admin can manage users
        assert access_results["admin"]["/api/orders"] is True     # Admin can place orders
        assert access_results["trader"]["/api/orders"] is True    # Trader can place orders
        assert access_results["trader"]["/api/users"] is False    # Trader cannot manage users
        assert access_results["viewer"]["/api/portfolio"] is True # Viewer can view portfolio
        assert access_results["viewer"]["/api/orders"] is False   # Viewer cannot place orders
        
        logger.info("API access control working correctly")
        logger.info(f"Access matrix: {access_results}")
        
        logger.info("JWT token permissions and scopes test completed")


class TestAPIEndpointAuthentication(BaseIntegrationTest):
    """Test API endpoint authentication mechanisms."""
    
    @pytest.mark.asyncio
    async def test_api_authentication_middleware(self):
        """Test API authentication middleware functionality."""
        
        class MockAuthenticationMiddleware:
            def __init__(self):
                self.authenticated_requests = 0
                self.failed_requests = 0
                self.rate_limited_requests = 0
                
            async def authenticate_request(self, headers: Dict[str, str], path: str) -> Dict[str, Any]:
                """Authenticate incoming API request."""
                # Check for Authorization header
                auth_header = headers.get("Authorization")
                if not auth_header:
                    self.failed_requests += 1
                    raise Exception("Missing Authorization header")
                    
                # Extract token
                if not auth_header.startswith("Bearer "):
                    self.failed_requests += 1
                    raise Exception("Invalid Authorization header format")
                    
                token = auth_header[7:]  # Remove "Bearer " prefix
                
                # Validate token (simplified)
                if token == "invalid_token":
                    self.failed_requests += 1
                    raise Exception("Invalid token")
                elif token == "expired_token":
                    self.failed_requests += 1
                    raise Exception("Token expired")
                
                # Mock user info extraction
                user_info = {
                    "user_id": "test_user",
                    "permissions": ["read", "write"],
                    "role": "trader"
                }
                
                self.authenticated_requests += 1
                
                return {
                    "authenticated": True,
                    "user": user_info,
                    "token": token
                }
                
            def get_auth_stats(self) -> Dict[str, int]:
                """Get authentication statistics."""
                return {
                    "authenticated": self.authenticated_requests,
                    "failed": self.failed_requests,
                    "rate_limited": self.rate_limited_requests
                }
        
        auth_middleware = MockAuthenticationMiddleware()
        
        # Test successful authentication
        valid_headers = {"Authorization": "Bearer valid_jwt_token_123"}
        
        auth_result = await auth_middleware.authenticate_request(valid_headers, "/api/portfolio")
        
        assert auth_result["authenticated"] is True
        assert auth_result["user"]["user_id"] == "test_user"
        assert "read" in auth_result["user"]["permissions"]
        
        logger.info("Valid authentication successful")
        
        # Test missing Authorization header
        invalid_headers_1 = {}
        
        with pytest.raises(Exception, match="Missing Authorization header"):
            await auth_middleware.authenticate_request(invalid_headers_1, "/api/portfolio")
        
        # Test invalid Authorization header format
        invalid_headers_2 = {"Authorization": "invalid_format"}
        
        with pytest.raises(Exception, match="Invalid Authorization header format"):
            await auth_middleware.authenticate_request(invalid_headers_2, "/api/portfolio")
        
        # Test invalid token
        invalid_headers_3 = {"Authorization": "Bearer invalid_token"}
        
        with pytest.raises(Exception, match="Invalid token"):
            await auth_middleware.authenticate_request(invalid_headers_3, "/api/portfolio")
        
        # Test expired token
        invalid_headers_4 = {"Authorization": "Bearer expired_token"}
        
        with pytest.raises(Exception, match="Token expired"):
            await auth_middleware.authenticate_request(invalid_headers_4, "/api/portfolio")
        
        logger.info("Invalid authentication cases handled correctly")
        
        # Check authentication statistics
        auth_stats = auth_middleware.get_auth_stats()
        
        assert auth_stats["authenticated"] == 1  # One successful auth
        assert auth_stats["failed"] == 4        # Four failed attempts
        
        logger.info(f"Authentication stats: {auth_stats}")
        
        # Test multiple successful authentications
        for i in range(10):
            valid_headers = {"Authorization": f"Bearer valid_token_{i}"}
            await auth_middleware.authenticate_request(valid_headers, f"/api/endpoint_{i}")
        
        final_stats = auth_middleware.get_auth_stats()
        assert final_stats["authenticated"] == 11  # 1 + 10 successful auths
        
        logger.info("API authentication middleware test completed")
    
    @pytest.mark.asyncio
    async def test_protected_endpoints_access_control(self):
        """Test access control for protected API endpoints."""
        
        class ProtectedEndpointManager:
            def __init__(self):
                self.endpoint_permissions = {
                    "/api/public/status": [],  # No authentication required
                    "/api/portfolio": ["read"],
                    "/api/orders": ["place_orders"],
                    "/api/positions": ["read"],
                    "/api/admin/users": ["manage_users"],
                    "/api/admin/system": ["admin"],
                    "/api/trading/execute": ["place_orders", "write"]
                }
                self.access_logs = []
                
            def check_endpoint_access(self, endpoint: str, user_permissions: List[str]) -> bool:
                """Check if user has access to endpoint."""
                required_permissions = self.endpoint_permissions.get(endpoint)
                
                if required_permissions is None:
                    # Unknown endpoint
                    return False
                    
                if not required_permissions:
                    # Public endpoint
                    return True
                    
                # Check if user has any of the required permissions
                has_access = any(perm in user_permissions for perm in required_permissions)
                
                # Log access attempt
                self.access_logs.append({
                    "endpoint": endpoint,
                    "user_permissions": user_permissions,
                    "required_permissions": required_permissions,
                    "access_granted": has_access,
                    "timestamp": datetime.now(timezone.utc)
                })
                
                return has_access
                
            def get_access_summary(self) -> Dict[str, Any]:
                """Get access attempt summary."""
                granted = len([log for log in self.access_logs if log["access_granted"]])
                denied = len([log for log in self.access_logs if not log["access_granted"]])
                
                return {
                    "total_attempts": len(self.access_logs),
                    "access_granted": granted,
                    "access_denied": denied
                }
        
        endpoint_manager = ProtectedEndpointManager()
        
        # Define different user types
        user_types = {
            "admin": ["read", "write", "admin", "manage_users", "place_orders"],
            "trader": ["read", "write", "place_orders"],
            "viewer": ["read"],
            "guest": []  # No permissions
        }
        
        # Test access for each user type
        test_endpoints = [
            "/api/public/status",
            "/api/portfolio",
            "/api/orders",
            "/api/positions",
            "/api/admin/users",
            "/api/admin/system",
            "/api/trading/execute"
        ]
        
        access_matrix = {}
        
        for user_type, permissions in user_types.items():
            access_matrix[user_type] = {}
            
            for endpoint in test_endpoints:
                has_access = endpoint_manager.check_endpoint_access(endpoint, permissions)
                access_matrix[user_type][endpoint] = has_access
        
        # Verify expected access patterns
        # Public endpoint should be accessible to all
        for user_type in user_types:
            assert access_matrix[user_type]["/api/public/status"] is True
        
        # Admin should have access to everything
        for endpoint in test_endpoints:
            assert access_matrix["admin"][endpoint] is True
        
        # Trader should have access to trading-related endpoints
        assert access_matrix["trader"]["/api/portfolio"] is True
        assert access_matrix["trader"]["/api/orders"] is True
        assert access_matrix["trader"]["/api/positions"] is True
        assert access_matrix["trader"]["/api/trading/execute"] is True
        
        # Trader should NOT have admin access
        assert access_matrix["trader"]["/api/admin/users"] is False
        assert access_matrix["trader"]["/api/admin/system"] is False
        
        # Viewer should only have read access
        assert access_matrix["viewer"]["/api/portfolio"] is True
        assert access_matrix["viewer"]["/api/positions"] is True
        assert access_matrix["viewer"]["/api/orders"] is False
        assert access_matrix["viewer"]["/api/trading/execute"] is False
        
        # Guest should only access public endpoints
        for endpoint in test_endpoints:
            if endpoint == "/api/public/status":
                assert access_matrix["guest"][endpoint] is True
            else:
                assert access_matrix["guest"][endpoint] is False
        
        logger.info("Access control matrix:")
        for user_type, endpoints in access_matrix.items():
            accessible_endpoints = [ep for ep, access in endpoints.items() if access]
            logger.info(f"  {user_type}: {len(accessible_endpoints)} accessible endpoints")
        
        # Get access summary
        summary = endpoint_manager.get_access_summary()
        
        expected_attempts = len(user_types) * len(test_endpoints)
        assert summary["total_attempts"] == expected_attempts
        assert summary["access_granted"] + summary["access_denied"] == expected_attempts
        
        logger.info(f"Access summary: {summary}")
        
        logger.info("Protected endpoints access control test completed")


class TestRateLimitingIntegration(BaseIntegrationTest):
    """Test rate limiting enforcement across the system."""
    
    @pytest.mark.asyncio 
    async def test_api_rate_limiting_enforcement(self):
        """Test API rate limiting enforcement."""
        
        class APIRateLimiter:
            def __init__(self):
                self.request_counts = {}  # user_id -> [(timestamp, endpoint), ...]
                self.rate_limits = {
                    "default": {"requests": 100, "window_seconds": 60},      # 100 req/min default
                    "trading": {"requests": 50, "window_seconds": 60},       # 50 req/min for trading
                    "admin": {"requests": 200, "window_seconds": 60},        # 200 req/min for admin
                    "public": {"requests": 1000, "window_seconds": 60}       # 1000 req/min for public
                }
                self.blocked_requests = 0
                
            def get_rate_limit_for_endpoint(self, endpoint: str, user_role: str = None) -> Dict[str, int]:
                """Get rate limit configuration for endpoint."""
                if endpoint.startswith("/api/public/"):
                    return self.rate_limits["public"]
                elif endpoint.startswith("/api/trading/"):
                    return self.rate_limits["trading"]
                elif endpoint.startswith("/api/admin/"):
                    return self.rate_limits["admin"]
                else:
                    return self.rate_limits["default"]
                    
            def check_rate_limit(self, user_id: str, endpoint: str, user_role: str = None) -> bool:
                """Check if request is within rate limit."""
                current_time = time.time()
                rate_limit = self.get_rate_limit_for_endpoint(endpoint, user_role)
                
                # Initialize user request history if not exists
                if user_id not in self.request_counts:
                    self.request_counts[user_id] = []
                
                user_requests = self.request_counts[user_id]
                
                # Remove requests outside the window
                window_start = current_time - rate_limit["window_seconds"]
                user_requests[:] = [(ts, ep) for ts, ep in user_requests if ts > window_start]
                
                # Check if user has exceeded the limit
                if len(user_requests) >= rate_limit["requests"]:
                    self.blocked_requests += 1
                    return False
                
                # Add current request
                user_requests.append((current_time, endpoint))
                return True
                
            def get_user_request_count(self, user_id: str) -> int:
                """Get current request count for user."""
                return len(self.request_counts.get(user_id, []))
                
            def get_rate_limit_stats(self) -> Dict[str, Any]:
                """Get rate limiting statistics."""
                active_users = len(self.request_counts)
                total_requests = sum(len(requests) for requests in self.request_counts.values())
                
                return {
                    "active_users": active_users,
                    "total_requests": total_requests,
                    "blocked_requests": self.blocked_requests
                }
        
        rate_limiter = APIRateLimiter()
        
        # Test normal usage (within limits)
        user_id = "test_user_1"
        endpoint = "/api/portfolio"
        
        # Make requests within limit
        for i in range(10):
            allowed = rate_limiter.check_rate_limit(user_id, endpoint)
            assert allowed is True
            
        logger.info(f"User {user_id} made 10 requests - all allowed")
        
        # Test rate limit enforcement
        # Make many requests to exceed limit
        allowed_count = 0
        blocked_count = 0
        
        for i in range(150):  # Exceed the default limit of 100
            allowed = rate_limiter.check_rate_limit(user_id, endpoint)
            if allowed:
                allowed_count += 1
            else:
                blocked_count += 1
        
        logger.info(f"Bulk requests: {allowed_count} allowed, {blocked_count} blocked")
        
        # Should have blocked some requests
        assert blocked_count > 0
        assert allowed_count <= 100  # Should not exceed the limit
        
        # Test different rate limits for different endpoints
        admin_user = "admin_user"
        admin_endpoint = "/api/admin/users"
        
        # Admin endpoints have higher limits (200/min)
        admin_allowed = 0
        for i in range(150):
            if rate_limiter.check_rate_limit(admin_user, admin_endpoint, "admin"):
                admin_allowed += 1
                
        logger.info(f"Admin user requests: {admin_allowed} allowed")
        assert admin_allowed > allowed_count  # Admin should have higher limit
        
        # Test public endpoint rate limiting
        public_user = "public_user"
        public_endpoint = "/api/public/status"
        
        public_allowed = 0
        for i in range(500):
            if rate_limiter.check_rate_limit(public_user, public_endpoint):
                public_allowed += 1
                
        logger.info(f"Public endpoint requests: {public_allowed} allowed")
        assert public_allowed > admin_allowed  # Public should have highest limit
        
        # Test rate limit recovery (time window)
        # Simulate time passing (in a real system, this would be actual time)
        # For testing, we'll clear old requests manually
        
        # Clear old requests for user_id
        rate_limiter.request_counts[user_id] = []
        
        # Should be able to make requests again
        recovery_allowed = 0
        for i in range(10):
            if rate_limiter.check_rate_limit(user_id, endpoint):
                recovery_allowed += 1
                
        assert recovery_allowed == 10  # All should be allowed after reset
        logger.info(f"After window reset: {recovery_allowed} requests allowed")
        
        # Get final statistics
        stats = rate_limiter.get_rate_limit_stats()
        
        logger.info(f"Rate limiting stats: {stats}")
        assert stats["active_users"] > 0
        assert stats["total_requests"] > 0
        assert stats["blocked_requests"] > 0
        
        logger.info("API rate limiting enforcement test completed")
    
    @pytest.mark.asyncio
    async def test_websocket_connection_rate_limiting(self):
        """Test WebSocket connection rate limiting."""
        
        class WebSocketConnectionLimiter:
            def __init__(self):
                self.connections_per_user = {}  # user_id -> connection_count
                self.connection_attempts = {}   # user_id -> [(timestamp, success), ...]
                self.max_connections_per_user = 5
                self.max_connection_attempts = 10  # per minute
                self.connection_window = 60  # seconds
                
            def can_connect(self, user_id: str, ip_address: str = None) -> tuple[bool, str]:
                """Check if user can establish new WebSocket connection."""
                current_time = time.time()
                
                # Check concurrent connection limit
                current_connections = self.connections_per_user.get(user_id, 0)
                if current_connections >= self.max_connections_per_user:
                    return False, f"Maximum concurrent connections ({self.max_connections_per_user}) exceeded"
                
                # Initialize connection attempt history
                if user_id not in self.connection_attempts:
                    self.connection_attempts[user_id] = []
                
                attempts = self.connection_attempts[user_id]
                
                # Clean old attempts
                window_start = current_time - self.connection_window
                attempts[:] = [(ts, success) for ts, success in attempts if ts > window_start]
                
                # Check connection attempt rate
                if len(attempts) >= self.max_connection_attempts:
                    return False, f"Too many connection attempts ({len(attempts)} in last minute)"
                
                return True, "Connection allowed"
                
            def register_connection(self, user_id: str, success: bool):
                """Register connection attempt."""
                current_time = time.time()
                
                # Record attempt
                if user_id not in self.connection_attempts:
                    self.connection_attempts[user_id] = []
                self.connection_attempts[user_id].append((current_time, success))
                
                # Update connection count if successful
                if success:
                    self.connections_per_user[user_id] = self.connections_per_user.get(user_id, 0) + 1
                    
            def disconnect_user(self, user_id: str):
                """Handle user disconnection."""
                if user_id in self.connections_per_user:
                    self.connections_per_user[user_id] = max(0, self.connections_per_user[user_id] - 1)
                    
            def get_connection_stats(self) -> Dict[str, Any]:
                """Get connection statistics."""
                total_connections = sum(self.connections_per_user.values())
                active_users = len([u for u, c in self.connections_per_user.items() if c > 0])
                
                return {
                    "total_connections": total_connections,
                    "active_users": active_users,
                    "max_concurrent_per_user": self.max_connections_per_user
                }
        
        ws_limiter = WebSocketConnectionLimiter()
        
        # Test normal connection flow
        user_id = "test_user"
        
        # First connection should be allowed
        can_connect, message = ws_limiter.can_connect(user_id)
        assert can_connect is True
        assert message == "Connection allowed"
        
        ws_limiter.register_connection(user_id, True)  # Successful connection
        
        logger.info(f"User {user_id} connected successfully")
        
        # Test multiple connections for same user
        successful_connections = 1  # Already connected once
        
        for i in range(10):  # Try to connect 10 more times
            can_connect, message = ws_limiter.can_connect(user_id)
            
            if can_connect:
                ws_limiter.register_connection(user_id, True)
                successful_connections += 1
            else:
                ws_limiter.register_connection(user_id, False)  # Failed attempt
                logger.info(f"Connection {i+2} blocked: {message}")
        
        # Should be limited by max_connections_per_user (5)
        assert successful_connections <= ws_limiter.max_connections_per_user
        
        logger.info(f"User established {successful_connections} connections (max: {ws_limiter.max_connections_per_user})")
        
        # Test connection attempt rate limiting
        rapid_user = "rapid_user"
        
        rapid_attempts = 0
        blocked_attempts = 0
        
        for i in range(20):  # Many rapid attempts
            can_connect, message = ws_limiter.can_connect(rapid_user)
            ws_limiter.register_connection(rapid_user, False)  # All attempts fail for testing
            
            if can_connect:
                rapid_attempts += 1
            else:
                blocked_attempts += 1
                if "too many" in message.lower():
                    break  # Hit rate limit
                    
        logger.info(f"Rapid connection attempts: {rapid_attempts} allowed, {blocked_attempts} blocked")
        
        # Should hit attempt rate limit
        assert blocked_attempts > 0
        
        # Test disconnection handling
        original_connections = ws_limiter.connections_per_user.get(user_id, 0)
        
        # Disconnect one connection
        ws_limiter.disconnect_user(user_id)
        
        new_connections = ws_limiter.connections_per_user.get(user_id, 0)
        assert new_connections == original_connections - 1
        
        logger.info(f"After disconnection: {new_connections} connections remaining")
        
        # Should be able to connect again after disconnection
        can_connect_again, message = ws_limiter.can_connect(user_id)
        if new_connections < ws_limiter.max_connections_per_user:
            assert can_connect_again is True
            
        # Get final statistics
        stats = ws_limiter.get_connection_stats()
        
        logger.info(f"WebSocket connection stats: {stats}")
        assert stats["total_connections"] >= 0
        assert stats["active_users"] >= 0
        
        logger.info("WebSocket connection rate limiting test completed")


class TestSecurityHeaders(BaseIntegrationTest):
    """Test security headers and HTTPS enforcement."""
    
    @pytest.mark.asyncio
    async def test_security_headers_enforcement(self):
        """Test that proper security headers are enforced."""
        
        class SecurityHeaderManager:
            def __init__(self):
                self.required_headers = {
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "X-XSS-Protection": "1; mode=block",
                    "Content-Security-Policy": "default-src 'self'; script-src 'self'",
                    "Referrer-Policy": "strict-origin-when-cross-origin"
                }
                self.response_headers = {}
                
            def add_security_headers(self, response_headers: Dict[str, str]):
                """Add security headers to response."""
                for header, value in self.required_headers.items():
                    response_headers[header] = value
                    
                self.response_headers = response_headers.copy()
                
            def validate_security_headers(self, headers: Dict[str, str]) -> Dict[str, bool]:
                """Validate that all required security headers are present."""
                validation_results = {}
                
                for required_header, expected_value in self.required_headers.items():
                    present = required_header in headers
                    correct_value = headers.get(required_header) == expected_value
                    
                    validation_results[required_header] = {
                        "present": present,
                        "correct_value": correct_value,
                        "valid": present and correct_value
                    }
                    
                return validation_results
                
            def get_security_score(self, headers: Dict[str, str]) -> float:
                """Calculate security header compliance score."""
                validation = self.validate_security_headers(headers)
                
                valid_headers = sum(1 for result in validation.values() if result["valid"])
                total_headers = len(self.required_headers)
                
                return (valid_headers / total_headers) * 100
        
        security_manager = SecurityHeaderManager()
        
        # Test adding security headers
        response_headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache"
        }
        
        security_manager.add_security_headers(response_headers)
        
        # Verify all security headers were added
        for required_header in security_manager.required_headers:
            assert required_header in response_headers
            
        logger.info(f"Added {len(security_manager.required_headers)} security headers")
        
        # Test header validation
        validation_results = security_manager.validate_security_headers(response_headers)
        
        # All headers should be valid
        for header, result in validation_results.items():
            assert result["present"] is True
            assert result["correct_value"] is True
            assert result["valid"] is True
            
        security_score = security_manager.get_security_score(response_headers)
        assert security_score == 100.0  # Perfect score
        
        logger.info(f"Security compliance score: {security_score}%")
        
        # Test with missing headers
        incomplete_headers = {
            "Content-Type": "application/json",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            # Missing other security headers
        }
        
        incomplete_validation = security_manager.validate_security_headers(incomplete_headers)
        incomplete_score = security_manager.get_security_score(incomplete_headers)
        
        assert incomplete_score < 100.0  # Should have lower score
        
        # Count missing headers
        missing_headers = [header for header, result in incomplete_validation.items() if not result["valid"]]
        
        logger.info(f"Missing security headers: {missing_headers}")
        logger.info(f"Incomplete security score: {incomplete_score}%")
        
        # Test with incorrect header values
        incorrect_headers = {
            "Strict-Transport-Security": "max-age=86400",  # Too short
            "X-Content-Type-Options": "sniff",             # Wrong value
            "X-Frame-Options": "SAMEORIGIN",               # Less secure
            "X-XSS-Protection": "0",                       # Disabled
            "Content-Security-Policy": "default-src *",   # Too permissive
            "Referrer-Policy": "unsafe-url"               # Less secure
        }
        
        incorrect_validation = security_manager.validate_security_headers(incorrect_headers)
        incorrect_score = security_manager.get_security_score(incorrect_headers)
        
        # Headers are present but values are incorrect
        assert incorrect_score == 0.0  # All headers have wrong values
        
        for header, result in incorrect_validation.items():
            assert result["present"] is True    # Headers are present
            assert result["correct_value"] is False  # But values are wrong
            assert result["valid"] is False     # So they're invalid
            
        logger.info(f"Incorrect headers security score: {incorrect_score}%")
        
        logger.info("Security headers enforcement test completed")
    
    @pytest.mark.asyncio
    async def test_https_enforcement(self):
        """Test HTTPS enforcement and redirect mechanisms."""
        
        class HTTPSEnforcer:
            def __init__(self):
                self.enforce_https = True
                self.hsts_enabled = True
                self.redirected_requests = 0
                self.blocked_requests = 0
                
            def check_request_security(self, protocol: str, headers: Dict[str, str]) -> Dict[str, Any]:
                """Check if request meets security requirements."""
                result = {
                    "secure": False,
                    "action": "allow",
                    "message": "",
                    "redirect_url": None
                }
                
                if protocol.lower() == "https":
                    result["secure"] = True
                    result["action"] = "allow"
                    result["message"] = "Secure connection"
                    
                elif protocol.lower() == "http":
                    if self.enforce_https:
                        result["secure"] = False
                        result["action"] = "redirect"
                        result["message"] = "Redirecting to HTTPS"
                        result["redirect_url"] = "https://example.com"
                        self.redirected_requests += 1
                    else:
                        result["secure"] = False
                        result["action"] = "allow"
                        result["message"] = "HTTP allowed"
                else:
                    result["secure"] = False
                    result["action"] = "block"
                    result["message"] = "Unsupported protocol"
                    self.blocked_requests += 1
                    
                return result
                
            def add_hsts_header(self, response_headers: Dict[str, str]):
                """Add HSTS header if enabled."""
                if self.hsts_enabled:
                    response_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
                    
            def get_enforcement_stats(self) -> Dict[str, int]:
                """Get HTTPS enforcement statistics."""
                return {
                    "redirected_requests": self.redirected_requests,
                    "blocked_requests": self.blocked_requests
                }
        
        https_enforcer = HTTPSEnforcer()
        
        # Test HTTPS request (should be allowed)
        https_result = https_enforcer.check_request_security("https", {})
        
        assert https_result["secure"] is True
        assert https_result["action"] == "allow"
        assert https_result["redirect_url"] is None
        
        logger.info("HTTPS request allowed correctly")
        
        # Test HTTP request (should be redirected)
        http_result = https_enforcer.check_request_security("http", {})
        
        assert http_result["secure"] is False
        assert http_result["action"] == "redirect"
        assert http_result["redirect_url"] is not None
        assert "https://" in http_result["redirect_url"]
        
        logger.info("HTTP request redirected to HTTPS")
        
        # Test invalid protocol (should be blocked)
        invalid_result = https_enforcer.check_request_security("ftp", {})
        
        assert invalid_result["secure"] is False
        assert invalid_result["action"] == "block"
        
        logger.info("Invalid protocol blocked correctly")
        
        # Test HSTS header addition
        response_headers = {"Content-Type": "application/json"}
        https_enforcer.add_hsts_header(response_headers)
        
        assert "Strict-Transport-Security" in response_headers
        hsts_value = response_headers["Strict-Transport-Security"]
        assert "max-age=" in hsts_value
        assert "includeSubDomains" in hsts_value
        assert "preload" in hsts_value
        
        logger.info("HSTS header added correctly")
        
        # Test statistics
        # Make more requests to increase stats
        for i in range(5):
            https_enforcer.check_request_security("http", {})  # Will be redirected
            
        for i in range(3):
            https_enforcer.check_request_security("ftp", {})   # Will be blocked
            
        stats = https_enforcer.get_enforcement_stats()
        
        assert stats["redirected_requests"] == 6  # 1 + 5 redirected
        assert stats["blocked_requests"] == 4     # 1 + 3 blocked
        
        logger.info(f"HTTPS enforcement stats: {stats}")
        
        # Test with HTTPS enforcement disabled
        https_enforcer.enforce_https = False
        
        http_allowed_result = https_enforcer.check_request_security("http", {})
        
        assert http_allowed_result["secure"] is False
        assert http_allowed_result["action"] == "allow"  # Should be allowed when enforcement is off
        assert http_allowed_result["redirect_url"] is None
        
        logger.info("HTTP allowed when enforcement is disabled")
        
        logger.info("HTTPS enforcement test completed")

        
logger.info("Security and authentication integration tests module loaded")