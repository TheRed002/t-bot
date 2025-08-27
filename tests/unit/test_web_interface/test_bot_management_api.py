"""
Unit tests for bot management API endpoints.

This module tests the bot management endpoints including bot creation,
lifecycle operations, and status monitoring.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi import status
from decimal import Decimal
from datetime import datetime


class TestBotManagementAPI:
    """Test bot management API endpoints."""

    def test_create_bot_success(self, test_client, auth_headers, sample_bot_config):
        """Test successful bot creation."""
        bot_data = {
            "bot_name": "Test Trading Bot",
            "bot_type": "strategy",  # Fixed: use valid enum value
            "strategy_name": "trend_following",
            "exchanges": ["binance"],
            "symbols": ["BTCUSDT"],
            "allocated_capital": "10000.0",  # Use string for Decimal
            "risk_percentage": 0.02,
            "priority": "normal",  # Fixed: use valid enum value
            "auto_start": False,
            "configuration": {}
        }
        
        with patch('src.web_interface.api.bot_management.get_trading_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock bot orchestrator
                mock_orchestrator.create_bot = AsyncMock(return_value="bot_123")
                
                response = test_client.post("/api/bots/", 
                                          json=bot_data, 
                                          headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == "bot_123"
                assert data["bot_name"] == "Test Trading Bot"

    def test_create_bot_validation_error(self, test_client, auth_headers):
        """Test bot creation with validation errors."""
        invalid_bot_data = {
            "bot_name": "",  # Empty name
            "bot_type": "strategy",  # Fixed: use valid enum value
            "strategy_name": "trend_following",
            "exchanges": [],  # Empty exchanges
            "symbols": ["BTCUSDT"],
            "allocated_capital": "-1000.0",  # Negative capital (string for Decimal)
            "risk_percentage": 0.02
        }
        
        with patch('src.web_interface.api.bot_management.get_trading_user') as mock_get_user:
            from src.web_interface.security.auth import User
            
            mock_user = User(
                user_id="trader_001",
                username="trader",
                email="trader@example.com",
                is_active=True,
                scopes=["read", "write", "trade"]
            )
            mock_get_user.return_value = mock_user
            
            response = test_client.post("/api/bots/", 
                                      json=invalid_bot_data, 
                                      headers=auth_headers)
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_list_bots(self, test_client, auth_headers):
        """Test listing bots."""
        with patch('src.web_interface.api.bot_management.get_current_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="user_001",
                    username="user",
                    email="user@example.com",
                    is_active=True,
                    scopes=["read"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock bot list
                mock_bot_list = [
                    {
                        "bot_id": "bot_001",
                        "bot_name": "Trend Bot",
                        "status": "running",
                        "allocated_capital": Decimal("10000"),
                        "metrics": {
                            "total_pnl": 500.0,
                            "total_trades": 25,
                            "win_rate": 0.68
                        }
                    },
                    {
                        "bot_id": "bot_002",
                        "bot_name": "Arbitrage Bot",
                        "status": "stopped",
                        "allocated_capital": Decimal("20000"),
                        "metrics": {
                            "total_pnl": -100.0,
                            "total_trades": 15,
                            "win_rate": 0.53
                        }
                    }
                ]
                
                mock_orchestrator.get_bot_list = AsyncMock(return_value=mock_bot_list)
                
                response = test_client.get("/api/bots/", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["total"] == 2
                assert len(data["bots"]) == 2
                assert data["bots"][0]["bot_id"] == "bot_001"

    def test_get_bot_details(self, test_client, auth_headers):
        """Test getting bot details."""
        bot_id = "bot_001"
        
        with patch('src.web_interface.api.bot_management.get_current_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="user_001",
                    username="user",
                    email="user@example.com",
                    is_active=True,
                    scopes=["read"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock bot instance
                mock_bot_instance = AsyncMock()
                mock_bot_instance.get_bot_summary = AsyncMock(return_value={
                    "bot_id": bot_id,
                    "bot_name": "Test Bot",
                    "status": "running",
                    "uptime": "5 hours",
                    "metrics": {
                        "total_trades": 10,
                        "total_pnl": 250.0
                    }
                })
                
                mock_orchestrator.bot_instances = {bot_id: mock_bot_instance}
                
                response = test_client.get(f"/api/bots/{bot_id}", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot"]["bot_id"] == bot_id

    def test_get_bot_not_found(self, test_client, auth_headers):
        """Test getting non-existent bot."""
        bot_id = "non_existent_bot"
        
        with patch('src.web_interface.api.bot_management.get_current_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                mock_user = User(
                    user_id="user_001",
                    username="user",
                    email="user@example.com",
                    is_active=True,
                    scopes=["read"]
                )
                mock_get_user.return_value = mock_user
                
                mock_orchestrator.bot_instances = {}  # Empty bot instances
                
                response = test_client.get(f"/api/bots/{bot_id}", headers=auth_headers)
                
                assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_start_bot(self, test_client, auth_headers):
        """Test starting a bot."""
        bot_id = "bot_001"
        
        with patch('src.web_interface.api.bot_management.get_trading_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock successful bot start
                mock_orchestrator.start_bot = AsyncMock(return_value=True)
                
                response = test_client.post(f"/api/bots/{bot_id}/start", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id

    def test_stop_bot(self, test_client, auth_headers):
        """Test stopping a bot."""
        bot_id = "bot_001"
        
        with patch('src.web_interface.api.bot_management.get_trading_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock successful bot stop
                mock_orchestrator.stop_bot = AsyncMock(return_value=True)
                
                response = test_client.post(f"/api/bots/{bot_id}/stop", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id

    def test_pause_bot(self, test_client, auth_headers):
        """Test pausing a bot."""
        bot_id = "bot_001"
        
        with patch('src.web_interface.api.bot_management.get_trading_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock successful bot pause
                mock_orchestrator.pause_bot = AsyncMock(return_value=True)
                
                response = test_client.post(f"/api/bots/{bot_id}/pause", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id

    def test_resume_bot(self, test_client, auth_headers):
        """Test resuming a bot."""
        bot_id = "bot_001"
        
        with patch('src.web_interface.api.bot_management.get_trading_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock successful bot resume
                mock_orchestrator.resume_bot = AsyncMock(return_value=True)
                
                response = test_client.post(f"/api/bots/{bot_id}/resume", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id

    def test_update_bot(self, test_client, auth_headers):
        """Test updating bot configuration."""
        bot_id = "bot_001"
        update_data = {
            "bot_name": "Updated Bot Name",
            "allocated_capital": "15000.0",  # Use string for Decimal
            "risk_percentage": 0.03
        }
        
        with patch('src.web_interface.api.bot_management.get_trading_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock bot instance
                from unittest.mock import MagicMock
                mock_bot_instance = AsyncMock()
                mock_config = MagicMock()
                mock_config.bot_name = "Old Bot Name"
                mock_config.allocated_capital = Decimal("10000")
                mock_config.risk_percentage = 0.02
                
                mock_bot_instance.get_bot_config = MagicMock(return_value=mock_config)
                mock_bot_instance.update_configuration = AsyncMock(return_value=None)
                
                mock_orchestrator.bot_instances = {bot_id: mock_bot_instance}
                
                response = test_client.put(f"/api/bots/{bot_id}", 
                                         json=update_data, 
                                         headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id
                assert "bot_name" in data["updated_fields"]

    def test_delete_bot_admin_only(self, test_client, admin_auth_headers):
        """Test deleting a bot (admin only)."""
        bot_id = "bot_001"
        
        with patch('src.web_interface.api.bot_management.get_admin_user') as mock_get_admin:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock admin user
                mock_admin = User(
                    user_id="admin_001",
                    username="admin",
                    email="admin@example.com",
                    is_active=True,
                    scopes=["admin"]
                )
                mock_get_admin.return_value = mock_admin
                
                # Mock successful bot deletion
                mock_orchestrator.delete_bot = AsyncMock(return_value=True)
                
                response = test_client.delete(f"/api/bots/{bot_id}", headers=admin_auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["bot_id"] == bot_id

    def test_get_orchestrator_status(self, test_client, auth_headers):
        """Test getting orchestrator status."""
        with patch('src.web_interface.api.bot_management.get_current_user') as mock_get_user:
            with patch('src.web_interface.api.bot_management.bot_orchestrator') as mock_orchestrator:
                from src.web_interface.security.auth import User
                
                # Mock authenticated user
                mock_user = User(
                    user_id="user_001",
                    username="user",
                    email="user@example.com",
                    is_active=True,
                    scopes=["read"]
                )
                mock_get_user.return_value = mock_user
                
                # Mock orchestrator status
                mock_status = {
                    "orchestrator": {
                        "is_running": True,
                        "total_bots": 5,
                        "emergency_shutdown": False
                    },
                    "global_metrics": {
                        "total_pnl": 1250.0,
                        "total_trades": 150,
                        "running_bots": 3,
                        "stopped_bots": 2
                    }
                }
                
                mock_orchestrator.get_orchestrator_status = AsyncMock(return_value=mock_status)
                
                response = test_client.get("/api/bots/orchestrator/status", headers=auth_headers)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["success"] is True
                assert data["status"]["orchestrator"]["is_running"] is True
                assert data["status"]["global_metrics"]["total_pnl"] == 1250.0

    def test_bot_orchestrator_not_available(self, test_client, auth_headers):
        """Test API behavior when bot orchestrator is not available."""
        # First set the bot orchestrator to None
        from src.web_interface.api import bot_management
        original_orchestrator = bot_management.bot_orchestrator
        bot_management.set_bot_orchestrator(None)
        
        try:
            with patch('src.web_interface.api.bot_management.get_trading_user') as mock_get_user:
                from src.web_interface.security.auth import User
                
                mock_user = User(
                    user_id="trader_001",
                    username="trader",
                    email="trader@example.com",
                    is_active=True,
                    scopes=["read", "write", "trade"]
                )
                mock_get_user.return_value = mock_user
                
                bot_data = {
                    "bot_name": "Test Bot",
                    "bot_type": "strategy",  # Fixed: use valid enum value
                    "strategy_name": "trend_following",
                    "exchanges": ["binance"],
                    "symbols": ["BTCUSDT"],
                    "allocated_capital": "10000.0",  # Use string for Decimal
                    "risk_percentage": 0.02
                }
                
                response = test_client.post("/api/bots/", 
                                          json=bot_data, 
                                          headers=auth_headers)
                
                assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        finally:
            # Restore the original orchestrator
            bot_management.set_bot_orchestrator(original_orchestrator)