"""
Unit tests for bot management API endpoints.

This module tests the bot management endpoints including bot creation,
lifecycle operations, and status monitoring.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import status


class TestBotManagementAPI:
    """Test bot management API endpoints."""

    def test_create_bot_success(self, test_client, auth_headers, sample_bot_config):
        """Test successful bot creation."""
        bot_data = {
            "bot_name": "Test Trading Bot",
            "bot_type": "trading",  # Fixed: use valid BotType enum value
            "strategy_name": "trend_following",
            "exchanges": ["binance"],
            "symbols": ["BTCUSDT"],
            "allocated_capital": "10000.0",  # Use string for Decimal
            "risk_percentage": "0.02",  # Use string for Decimal validation
            "priority": "normal",  # Fixed: use valid enum value
            "auto_start": False,
            "configuration": {},
        }

        with patch("src.web_interface.api.bot_management.get_trading_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"],
                )
                mock_get_user.return_value = mock_user

                # Mock web bot service
                mock_service = AsyncMock()
                mock_service.validate_bot_configuration = AsyncMock(return_value={"valid": True, "errors": []})
                mock_service.create_bot_configuration = AsyncMock(return_value={})
                mock_service.create_bot_through_service = AsyncMock(return_value="bot_123")
                mock_service.format_bot_response = AsyncMock(return_value={
                    "success": True,
                    "bot_id": "bot_123",
                    "bot_name": "Test Trading Bot",
                    "allocated_capital": "10000.0",
                    "risk_percentage": "0.02",
                    "auto_start": False
                })
                mock_get_bot_service.return_value = mock_service

                response = test_client.post("/api/bot/", json=bot_data, headers=auth_headers)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == "bot_123"
                assert data["bot_name"] == "Test Trading Bot"

    def test_create_bot_validation_error(self, test_client, auth_headers):
        """Test bot creation with validation errors."""
        invalid_bot_data = {
            "bot_name": "",  # Empty name
            "bot_type": "trading",  # Fixed: use valid BotType enum value
            "strategy_name": "trend_following",
            "exchanges": [],  # Empty exchanges
            "symbols": ["BTCUSDT"],
            "allocated_capital": "-1000.0",  # Negative capital (string for Decimal)
            "risk_percentage": "0.02",  # Use string for Decimal validation
        }

        with patch("src.web_interface.api.bot_management.get_trading_user") as mock_get_user:
            from src.web_interface.security.auth import User

            mock_user = User(
                user_id="trader_001",
                username="trader",
                email="trader@example.com",
                is_active=True,
                scopes=["read", "write", "trade"],
            )
            mock_get_user.return_value = mock_user

            response = test_client.post("/api/bot/", json=invalid_bot_data, headers=auth_headers)

            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_list_bots(self, test_client, auth_headers):
        """Test listing bots."""
        with patch("src.web_interface.api.bot_management.get_current_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock authenticated user
                mock_user = User(
                    user_id="user_001",
                    username="user",
                    email="user@example.com",
                    is_active=True,
                    scopes=["read"],
                )
                mock_get_user.return_value = mock_user

                # Mock bot list
                mock_bot_list = [
                    {
                        "bot_id": "bot_001",
                        "bot_name": "Trend Bot",
                        "status": "running",
                        "allocated_capital": 10000,
                        "metrics": {"total_pnl": 500.0, "total_trades": 25, "win_rate": 0.68},
                    },
                    {
                        "bot_id": "bot_002",
                        "bot_name": "Arbitrage Bot",
                        "status": "stopped",
                        "allocated_capital": 20000,
                        "metrics": {"total_pnl": -100.0, "total_trades": 15, "win_rate": 0.53},
                    },
                ]

                # Mock web bot service
                mock_service = AsyncMock()
                mock_service.get_formatted_bot_list = AsyncMock(return_value={
                    "bots": mock_bot_list,
                    "total": 2,
                    "status_counts": {"running": 1, "stopped": 1, "error": 0}
                })
                mock_get_bot_service.return_value = mock_service

                response = test_client.get("/api/bot/", headers=auth_headers)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["total"] == 2
                assert len(data["bots"]) == 2
                assert data["bots"][0]["bot_id"] == "bot_001"

    def test_get_bot_details(self, test_client, auth_headers):
        """Test getting bot details."""
        bot_id = "bot_001"

        with patch("src.web_interface.api.bot_management.get_current_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock authenticated user
                mock_user = User(
                    user_id="user_001",
                    username="user",
                    email="user@example.com",
                    is_active=True,
                    scopes=["read"],
                )
                mock_get_user.return_value = mock_user

                # Mock bot status
                mock_bot_status = {
                    "bot_id": bot_id,
                    "bot_name": "Test Bot",
                    "status": "running",
                    "uptime": "5 hours",
                    "metrics": {"total_trades": 10, "total_pnl": 250.0},
                }

                # Mock web bot service
                mock_service = AsyncMock()
                mock_service.get_bot_status_through_service = AsyncMock(return_value=mock_bot_status)
                mock_service.calculate_bot_metrics = AsyncMock(return_value={"enhanced_pnl": 275.0})
                mock_get_bot_service.return_value = mock_service

                response = test_client.get(f"/api/bot/{bot_id}", headers=auth_headers)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot"]["bot_id"] == bot_id

    def test_get_bot_not_found(self, test_client, auth_headers):
        """Test getting non-existent bot."""
        bot_id = "non_existent_bot"

        with patch("src.web_interface.api.bot_management.get_current_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User
                from src.core.exceptions import EntityNotFoundError

                mock_user = User(
                    user_id="user_001",
                    username="user",
                    email="user@example.com",
                    is_active=True,
                    scopes=["read"],
                )
                mock_get_user.return_value = mock_user

                # Mock web bot service to raise EntityNotFoundError
                mock_service = AsyncMock()
                mock_service.get_bot_status_through_service = AsyncMock(side_effect=EntityNotFoundError(f"Bot not found: {bot_id}"))
                mock_get_bot_service.return_value = mock_service

                response = test_client.get(f"/api/bot/{bot_id}", headers=auth_headers)

                assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_start_bot(self, test_client, auth_headers):
        """Test starting a bot."""
        bot_id = "bot_001"

        with patch("src.web_interface.api.bot_management.get_trading_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"],
                )
                mock_get_user.return_value = mock_user

                # Mock web bot service
                mock_service = AsyncMock()
                mock_service.start_bot_through_service = AsyncMock(return_value=True)
                mock_get_bot_service.return_value = mock_service

                response = test_client.post(f"/api/bot/{bot_id}/start", headers=auth_headers)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id

    def test_stop_bot(self, test_client, auth_headers):
        """Test stopping a bot."""
        bot_id = "bot_001"

        with patch("src.web_interface.api.bot_management.get_trading_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"],
                )
                mock_get_user.return_value = mock_user

                # Mock web bot service
                mock_service = AsyncMock()
                mock_service.stop_bot_through_service = AsyncMock(return_value=True)
                mock_get_bot_service.return_value = mock_service

                response = test_client.post(f"/api/bot/{bot_id}/stop", headers=auth_headers)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id

    def test_pause_bot(self, test_client, auth_headers):
        """Test pausing a bot."""
        bot_id = "bot_001"

        with patch("src.web_interface.api.bot_management.get_trading_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"],
                )
                mock_get_user.return_value = mock_user

                # Mock web bot service - pause returns False (not implemented)
                mock_service = AsyncMock()
                mock_get_bot_service.return_value = mock_service

                response = test_client.post(f"/api/bot/{bot_id}/pause", headers=auth_headers)

                # Pause functionality is not implemented in the API, expect 501
                assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

    def test_resume_bot(self, test_client, auth_headers):
        """Test resuming a bot."""
        bot_id = "bot_001"

        with patch("src.web_interface.api.bot_management.get_trading_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"],
                )
                mock_get_user.return_value = mock_user

                # Mock web bot service - resume returns False (not implemented)
                mock_service = AsyncMock()
                mock_get_bot_service.return_value = mock_service

                response = test_client.post(f"/api/bot/{bot_id}/resume", headers=auth_headers)

                # Resume functionality is not implemented in the API, expect 501
                assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

    def test_update_bot(self, test_client, auth_headers):
        """Test updating bot configuration."""
        bot_id = "bot_001"
        update_data = {
            "bot_name": "Updated Bot Name",
            "allocated_capital": "15000.0",  # Use string for Decimal
            "risk_percentage": "0.03",  # Use string for Decimal validation
        }

        with patch("src.web_interface.api.bot_management.get_trading_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User
                from unittest.mock import MagicMock

                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"],
                )
                mock_get_user.return_value = mock_user

                # Mock bot status with configuration
                mock_config = MagicMock()
                mock_config.bot_name = "Old Bot Name"
                mock_config.allocated_capital = Decimal("10000")
                mock_config.risk_percentage = 0.02
                mock_config.configuration = {}

                mock_bot_status = {
                    "state": {"configuration": mock_config},
                    "bot_id": bot_id
                }

                # Mock web bot service
                mock_service = AsyncMock()
                mock_service.get_bot_status_through_service = AsyncMock(return_value=mock_bot_status)
                mock_get_bot_service.return_value = mock_service

                response = test_client.put(
                    f"/api/bot/{bot_id}", json=update_data, headers=auth_headers
                )

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id
                assert "bot_name" in data["updated_fields"]

    def test_delete_bot_admin_only(self, test_client, admin_auth_headers):
        """Test deleting a bot (admin only)."""
        bot_id = "bot_001"

        with patch("src.web_interface.api.bot_management.get_admin_user") as mock_get_admin:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock admin user
                mock_admin = User(
                    user_id="admin_001",
                    username="admin",
                    email="admin@example.com",
                    is_active=True,
                    scopes=["admin"],
                )
                mock_get_admin.return_value = mock_admin

                # Mock web bot service
                mock_service = AsyncMock()
                mock_service.delete_bot_through_service = AsyncMock(return_value=True)
                mock_get_bot_service.return_value = mock_service

                response = test_client.delete(f"/api/bot/{bot_id}", headers=admin_auth_headers)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id

    def test_get_orchestrator_status(self, test_client, auth_headers):
        """Test getting orchestrator status."""
        with patch("src.web_interface.api.bot_management.get_current_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User

                # Mock authenticated user
                mock_user = User(
                    user_id="user_001",
                    username="user",
                    email="user@example.com",
                    is_active=True,
                    scopes=["read"],
                )
                mock_get_user.return_value = mock_user

                # Mock health check and bot list
                mock_health = {"status": "healthy"}
                mock_bots = [
                    {"status": "running"},
                    {"status": "running"},
                    {"status": "running"},
                    {"status": "stopped"},
                    {"status": "stopped"},
                ]

                # Mock web bot service
                mock_service = AsyncMock()
                mock_service.get_facade_health_check = MagicMock(return_value=mock_health)
                mock_service.list_bots_through_service = AsyncMock(return_value=mock_bots)
                mock_get_bot_service.return_value = mock_service

                response = test_client.get("/api/bot/orchestrator/status", headers=auth_headers)

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["status"]["service"]["is_running"] is True
                assert data["status"]["bots"]["total"] == 5

    def test_bot_orchestrator_not_available(self, test_client, auth_headers):
        """Test API behavior when bot service is not available."""
        with patch("src.web_interface.api.bot_management.get_trading_user") as mock_get_user:
            with patch(
                "src.web_interface.api.bot_management.get_web_bot_service_instance"
            ) as mock_get_bot_service:
                from src.web_interface.security.auth import User
                from src.core.exceptions import ServiceError

                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"],
                )
                mock_get_user.return_value = mock_user

                # Mock service error
                mock_get_bot_service.side_effect = ServiceError("Bot management service not available")

                bot_data = {
                    "bot_name": "Test Bot",
                    "bot_type": "trading",  # Fixed: use valid BotType enum value
                    "strategy_name": "trend_following",
                    "exchanges": ["binance"],
                    "symbols": ["BTCUSDT"],
                    "allocated_capital": "10000.0",  # Use string for Decimal
                    "risk_percentage": "0.02",  # Use string for Decimal validation
                }

                response = test_client.post("/api/bot/", json=bot_data, headers=auth_headers)

                assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
