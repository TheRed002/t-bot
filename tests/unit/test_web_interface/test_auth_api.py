"""
Unit tests for authentication API endpoints.

This module tests the authentication endpoints including login, logout,
token refresh, and user management functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import sys

from fastapi import status


class TestAuthAPI:
    """Test authentication API endpoints."""

    def test_login_success(self, test_client):
        """Test successful user login."""
        login_data = {"username": "admin", "password": "secret"}

        with patch("src.web_interface.api.auth.authenticate_user") as mock_auth:
            with patch("src.web_interface.api.auth.create_access_token") as mock_create_token:

                from src.web_interface.security.auth import Token, UserInDB

                # Mock successful authentication
                mock_user = UserInDB(
                    id="admin_001",
                    username="admin",
                    email="admin@test.com",
                    password_hash="hashed_password",
                    is_active=True,
                    is_verified=True,
                    scopes=["admin", "read", "write", "trade"],
                )
                mock_auth.return_value = mock_user

                # Mock token creation
                mock_token = Token(
                    access_token="test_access_token",
                    refresh_token="test_refresh_token",
                    expires_in=1800,
                )
                mock_create_token.return_value = mock_token

                response = test_client.post("/auth/login", json=login_data)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["message"] == "Login successful"
                assert data["user"]["username"] == "admin"
                assert data["token"]["access_token"] == "test_access_token"

    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials."""
        login_data = {"username": "invalid", "password": "wrong"}

        with patch("src.web_interface.api.auth.authenticate_user") as mock_auth:
            mock_auth.return_value = None  # Authentication failed

            response = test_client.post("/auth/login", json=login_data)

            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid username or password" in response.json()["detail"]

    def test_refresh_token_success(self, test_client):
        """Test successful token refresh."""
        refresh_data = {"refresh_token": "valid_refresh_token"}

        with patch("src.web_interface.api.auth.jwt_handler") as mock_jwt:
            mock_jwt.refresh_access_token.return_value = ("new_access_token", "new_refresh_token")

            response = test_client.post("/auth/refresh", json=refresh_data)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["token"]["access_token"] == "new_access_token"

    def test_refresh_token_invalid(self, test_client):
        """Test token refresh with invalid token."""
        refresh_data = {"refresh_token": "invalid_refresh_token"}

        with patch("src.web_interface.api.auth.jwt_handler") as mock_jwt:
            mock_jwt.refresh_access_token.side_effect = Exception("Invalid refresh token")

            response = test_client.post("/auth/refresh", json=refresh_data)

            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_user_info(self, test_client, auth_headers):
        """Test getting current user information."""
        with patch("src.web_interface.api.auth.get_current_user") as mock_get_user:
            from src.web_interface.security.auth import User

            mock_user = User(
                id="test_001",
                username="testuser",
                email="test@example.com",
                is_active=True,
                is_verified=True,
                scopes=["read", "write"],
            )
            mock_get_user.return_value = mock_user

            response = test_client.get("/auth/me", headers=auth_headers)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["username"] == "testuser"
            assert data["id"] == "test_001"

    def test_logout(self, test_client, auth_headers):
        """Test user logout."""
        with patch("src.web_interface.api.auth.get_current_user") as mock_get_user:
            from src.web_interface.security.auth import User

            mock_user = User(
                id="test_001",
                username="testuser",
                email="test@example.com",
                is_active=True,
                is_verified=True,
                scopes=["read"],
            )
            mock_get_user.return_value = mock_user

            response = test_client.post("/auth/logout", headers=auth_headers)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Logout successful"

    def test_change_password_success(self, test_client, auth_headers):
        """Test successful password change."""
        password_data = {"current_password": "old_password", "new_password": "new_password"}

        with patch("src.web_interface.api.auth.get_current_user") as mock_get_user:
            with patch("src.web_interface.security.auth.get_user", new_callable=AsyncMock) as mock_get_db_user:
                with patch("src.web_interface.api.auth.jwt_handler") as mock_jwt:
                    # Mock the import at the module level
                    
                    # Create a mock fake_users_db and add it to the auth module before import
                    mock_fake_db = MagicMock()
                    with patch.dict("sys.modules"):
                        import src.web_interface.security.auth
                        src.web_interface.security.auth.fake_users_db = mock_fake_db
                        
                        from src.web_interface.security.auth import User, UserInDB

                        # Mock current user
                        mock_user = User(
                            id="test_001",
                            username="testuser",
                            email="test@example.com",
                            is_active=True,
                            is_verified=True,
                            scopes=["read"],
                        )
                        mock_get_user.return_value = mock_user

                        # Mock database user (get_user is async)
                        mock_db_user = UserInDB(
                            id="test_001",
                            username="testuser",
                            email="test@example.com",
                            password_hash="old_hashed_password",
                            is_active=True,
                            is_verified=True,
                            scopes=["read"],
                        )
                        mock_get_db_user.return_value = mock_db_user

                        # Mock fake_users_db access
                        mock_fake_db.__getitem__ = lambda self, key: mock_db_user
                        mock_fake_db.__setitem__ = lambda self, key, value: None

                        # Mock JWT handler
                        mock_jwt.verify_password.return_value = True
                        mock_jwt.hash_password.return_value = "new_hashed_password"

                        response = test_client.put(
                            "/auth/me/password", json=password_data, headers=auth_headers
                        )

                        assert response.status_code == status.HTTP_200_OK
                        data = response.json()
                        assert data["success"] is True
                        assert data["message"] == "Password changed successfully"

    def test_change_password_wrong_current(self, test_client, auth_headers):
        """Test password change with wrong current password."""
        password_data = {"current_password": "wrong_password", "new_password": "new_password"}

        with patch("src.web_interface.api.auth.get_current_user") as mock_get_user:
            with patch("src.web_interface.security.auth.get_user", new_callable=AsyncMock) as mock_get_db_user:
                with patch("src.web_interface.api.auth.jwt_handler") as mock_jwt:
                    # Mock the import at the module level
                    
                    # Create a mock fake_users_db and add it to the auth module before import
                    mock_fake_db = MagicMock()
                    with patch.dict("sys.modules"):
                        import src.web_interface.security.auth
                        src.web_interface.security.auth.fake_users_db = mock_fake_db
                        
                        from src.web_interface.security.auth import User, UserInDB

                        # Mock current user
                        mock_user = User(
                            id="test_001",
                            username="testuser",
                            email="test@example.com",
                            is_active=True,
                            is_verified=True,
                            scopes=["read"],
                        )
                        mock_get_user.return_value = mock_user

                        # Mock database user (get_user is async)
                        mock_db_user = UserInDB(
                            id="test_001",
                            username="testuser",
                            email="test@example.com",
                            password_hash="old_hashed_password",
                            is_active=True,
                            is_verified=True,
                            scopes=["read"],
                        )
                        mock_get_db_user.return_value = mock_db_user

                        # Mock fake_users_db access
                        mock_fake_db.__getitem__ = lambda self, key: mock_db_user
                        mock_fake_db.__setitem__ = lambda self, key, value: None

                        # Mock JWT handler - wrong password
                        mock_jwt.verify_password.return_value = False

                        response = test_client.put(
                            "/auth/me/password", json=password_data, headers=auth_headers
                        )

                        assert response.status_code == status.HTTP_400_BAD_REQUEST
                        assert "Current password is incorrect" in response.json()["detail"]

    def test_create_user_admin_only(self, test_client, admin_auth_headers):
        """Test user creation (admin only)."""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123",
            "scopes": ["read", "write"],
        }

        with patch("src.web_interface.api.auth.get_admin_user") as mock_get_admin:
            with patch("src.web_interface.api.auth.create_user") as mock_create:
                from src.web_interface.security.auth import User, UserInDB

                # Mock admin user
                mock_admin = User(
                    id="admin_001",
                    username="admin",
                    email="admin@example.com",
                    is_active=True,
                    is_verified=True,
                    scopes=["admin"],
                )
                mock_get_admin.return_value = mock_admin

                # Mock user creation
                mock_created_user = UserInDB(
                    id="new_001",
                    username="newuser",
                    email="newuser@example.com",
                    password_hash="hashed",
                    is_active=True,
                    is_verified=True,
                    scopes=["read", "write"],
                )
                mock_create.return_value = mock_created_user

                response = test_client.post(
                    "/auth/users", json=user_data, headers=admin_auth_headers
                )

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["username"] == "newuser"
                assert data["scopes"] == ["read", "write"]

    def test_list_users_admin_only(self, test_client, admin_auth_headers):
        """Test listing users (admin only)."""
        with patch("src.web_interface.api.auth.get_admin_user") as mock_get_admin:
            # Mock the import at the module level
            
            # Create a mock fake_users_db and add it to the auth module before import
            mock_db = MagicMock()
            with patch.dict("sys.modules"):
                import src.web_interface.security.auth
                src.web_interface.security.auth.fake_users_db = mock_db
                        
                from src.web_interface.security.auth import User, UserInDB

                # Mock admin user
                mock_admin = User(
                    id="admin_001",
                    username="admin",
                    email="admin@example.com",
                    is_active=True,
                    is_verified=True,
                    scopes=["admin"],
                )
                mock_get_admin.return_value = mock_admin

                # Mock user database
                mock_db.values.return_value = [
                    UserInDB(
                        id="user_001",
                        username="user1",
                        email="user1@example.com",
                        password_hash="hashed",
                        scopes=["read"],
                    ),
                    UserInDB(
                        id="user_002",
                        username="user2",
                        email="user2@example.com",
                        password_hash="hashed",
                        scopes=["read", "write"],
                    ),
                ]

                response = test_client.get("/auth/users", headers=admin_auth_headers)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["total"] == 2
                assert len(data["users"]) == 2

    def test_auth_status(self, test_client):
        """Test authentication status endpoint."""
        with patch("src.web_interface.api.auth.get_auth_summary") as mock_summary:
            mock_summary.return_value = {
                "initialized": True,
                "total_users": 3,
                "active_users": 3,
                "available_scopes": ["read", "write", "trade", "admin"],
            }

            response = test_client.get("/auth/status")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["initialized"] is True
            assert data["total_users"] == 3
