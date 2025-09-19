"""
Simple test for simplified BotService to verify basic functionality.
"""

import pytest
from unittest.mock import MagicMock

from src.bot_management.service import BotService
from src.core.exceptions import ServiceError


class TestSimplifiedBotService:
    """Test simplified bot service functionality."""

    def test_bot_service_init_with_required_dependencies(self):
        """Test service initialization with required dependencies."""
        # Create mock services
        exchange_service = MagicMock()
        capital_service = MagicMock()

        # Create service with required dependencies
        service = BotService(
            exchange_service=exchange_service,
            capital_service=capital_service
        )

        assert service._name == "BotService"
        assert service._exchange_service == exchange_service
        assert service._capital_service == capital_service

    def test_bot_service_init_missing_exchange_service(self):
        """Test service initialization fails without exchange service."""
        capital_service = MagicMock()

        with pytest.raises(ServiceError, match="ExchangeService is required"):
            BotService(
                exchange_service=None,
                capital_service=capital_service
            )

    def test_bot_service_init_missing_capital_service(self):
        """Test service initialization fails without capital service."""
        exchange_service = MagicMock()

        with pytest.raises(ServiceError, match="CapitalService is required"):
            BotService(
                exchange_service=exchange_service,
                capital_service=None
            )

    def test_bot_service_init_with_optional_dependencies(self):
        """Test service initialization with optional dependencies."""
        # Create mock services
        exchange_service = MagicMock()
        capital_service = MagicMock()
        execution_service = MagicMock()
        risk_service = MagicMock()

        # Create service with optional dependencies
        service = BotService(
            exchange_service=exchange_service,
            capital_service=capital_service,
            execution_service=execution_service,
            risk_service=risk_service
        )

        assert service._execution_service == execution_service
        assert service._risk_service == risk_service

    def test_bot_service_init_minimal_dependencies(self):
        """Test service initialization with minimal required dependencies."""
        # Create mock services
        exchange_service = MagicMock()
        capital_service = MagicMock()

        # Create service with minimal dependencies
        service = BotService(
            exchange_service=exchange_service,
            capital_service=capital_service
        )

        # Verify the required services are set
        assert service._exchange_service == exchange_service
        assert service._capital_service == capital_service
        # Optional services should be None or handled via DI
        assert hasattr(service, '_execution_service')
        assert hasattr(service, '_state_service')
        assert hasattr(service, '_risk_service')