"""
API Integration Test Configuration and Fixtures.

Provides authentication fixtures and utilities for API integration tests.
"""

import pytest
from fastapi.testclient import TestClient

from src.core.config import Config
from src.web_interface.app import create_app


@pytest.fixture(scope="function")
def test_app():
    """Create test FastAPI application with test configuration."""
    from src.web_interface.security.auth import init_auth

    config = Config()
    config.environment = "test"
    config.debug = True

    # Set JWT secret for testing
    if hasattr(config, "security"):
        config.security.secret_key = "test_secret_key_for_integration_tests_minimum_32_chars_long"
    else:
        import os
        os.environ["SECRET_KEY"] = "test_secret_key_for_integration_tests_minimum_32_chars_long"

    # Use test Redis database (no external dependencies)
    if hasattr(config, "database"):
        config.database.redis_host = "localhost"
        config.database.redis_port = 6379
        config.database.redis_db = 15  # Separate test DB
        config.database.redis_password = None  # No password for local testing

    # Initialize authentication system BEFORE creating app
    init_auth(config)

    app = create_app(config)
    return app


@pytest.fixture(scope="function")
def test_client(test_app):
    """Create test client for API requests."""
    return TestClient(test_app)


@pytest.fixture(scope="function")
def admin_token(test_client):
    """
    Get authentication token for admin user.

    Uses seeded admin user credentials (admin/admin123).
    Returns the access token string.
    """
    response = test_client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"}
    )

    if response.status_code != 200:
        pytest.skip(f"Admin login failed with status {response.status_code}. Database may not be seeded.")

    data = response.json()
    assert data.get("success") is True, "Login should succeed"
    assert "token" in data, "Response should contain token"

    return data["token"]["access_token"]


@pytest.fixture(scope="function")
def trader_token(test_client):
    """
    Get authentication token for trader user.

    Uses seeded trader user credentials (trader1/trader123).
    Returns the access token string.
    """
    response = test_client.post(
        "/api/auth/login",
        json={"username": "trader1", "password": "trader123"}
    )

    if response.status_code != 200:
        pytest.skip(f"Trader login failed with status {response.status_code}. Database may not be seeded.")

    data = response.json()
    assert data.get("success") is True, "Login should succeed"
    assert "token" in data, "Response should contain token"

    return data["token"]["access_token"]


@pytest.fixture(scope="function")
def viewer_token(test_client):
    """
    Get authentication token for viewer user.

    Uses seeded viewer user credentials (viewer/viewer123).
    Returns the access token string.
    """
    response = test_client.post(
        "/api/auth/login",
        json={"username": "viewer", "password": "viewer123"}
    )

    if response.status_code != 200:
        pytest.skip(f"Viewer login failed with status {response.status_code}. Database may not be seeded.")

    data = response.json()
    assert data.get("success") is True, "Login should succeed"
    assert "token" in data, "Response should contain token"

    return data["token"]["access_token"]


@pytest.fixture(scope="function")
def authenticated_client(test_client, admin_token):
    """
    Authenticated test client with admin privileges.

    Returns a TestClient with Authorization header set using admin token.
    Use this fixture for endpoints requiring authentication.
    """
    test_client.headers["Authorization"] = f"Bearer {admin_token}"
    return test_client


@pytest.fixture(scope="function")
def trader_client(test_client, trader_token):
    """
    Authenticated test client with trader privileges.

    Returns a TestClient with Authorization header set using trader token.
    Use this fixture for testing trader-level permissions.
    """
    test_client.headers["Authorization"] = f"Bearer {trader_token}"
    return test_client


@pytest.fixture(scope="function")
def viewer_client(test_client, viewer_token):
    """
    Authenticated test client with viewer privileges.

    Returns a TestClient with Authorization header set using viewer token.
    Use this fixture for testing viewer-level permissions.
    """
    test_client.headers["Authorization"] = f"Bearer {viewer_token}"
    return test_client


def get_auth_headers(token: str) -> dict[str, str]:
    """
    Get authentication headers for manual requests.

    Args:
        token: JWT access token

    Returns:
        Dictionary with Authorization header
    """
    return {"Authorization": f"Bearer {token}"}
