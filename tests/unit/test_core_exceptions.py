"""
Unit tests for core exception hierarchy.

These tests verify the exception hierarchy and error handling patterns.
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from src.core.exceptions import (
    TradingBotError, ExchangeError, RiskManagementError,
    ValidationError, ExecutionError, ModelError, DataError,
    StateConsistencyError, SecurityError, ExchangeConnectionError,
    ExchangeRateLimitError, ExchangeInsufficientFundsError,
    PositionLimitError, DrawdownLimitError, ConfigurationError,
    SchemaValidationError, OrderRejectionError, DataValidationError,
    ModelLoadError, StateCorruptionError, AuthenticationError,
    AuthorizationError
)


class TestTradingBotError:
    """Test base exception functionality."""
    
    def test_base_exception(self):
        """Test TradingBotError base exception."""
        error = TradingBotError(
            "Test error message",
            error_code="TEST_ERROR",
            details={"test": "data"}
        )
        
        assert error.message == "Test error message"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"test": "data"}
        assert error.timestamp is not None
        assert isinstance(error.timestamp, datetime)
    
    def test_base_exception_str_representation(self):
        """Test string representation of base exception."""
        error = TradingBotError(
            "Test error message",
            error_code="TEST_ERROR",
            details={"test": "data"}
        )
        
        error_str = str(error)
        assert "Test error message" in error_str
        assert "TEST_ERROR" in error_str
        assert "test" in error_str
    
    def test_base_exception_without_optional_fields(self):
        """Test TradingBotError without optional fields."""
        error = TradingBotError("Simple error message")
        
        assert error.message == "Simple error message"
        assert error.error_code is None
        assert error.details == {}
        assert error.timestamp is not None


class TestExchangeExceptions:
    """Test exchange-related exceptions."""
    
    def test_exchange_error_inheritance(self):
        """Test ExchangeError inheritance."""
        error = ExchangeError("Exchange error")
        assert isinstance(error, TradingBotError)
        assert isinstance(error, ExchangeError)
    
    def test_exchange_connection_error(self):
        """Test ExchangeConnectionError."""
        error = ExchangeConnectionError("Connection failed")
        assert isinstance(error, ExchangeError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Connection failed"
    
    def test_exchange_rate_limit_error(self):
        """Test ExchangeRateLimitError."""
        error = ExchangeRateLimitError("Rate limit exceeded")
        assert isinstance(error, ExchangeError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Rate limit exceeded"
    
    def test_exchange_insufficient_funds_error(self):
        """Test ExchangeInsufficientFundsError."""
        error = ExchangeInsufficientFundsError("Insufficient funds")
        assert isinstance(error, ExchangeError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Insufficient funds"


class TestRiskManagementExceptions:
    """Test risk management exceptions."""
    
    def test_risk_management_error_inheritance(self):
        """Test RiskManagementError inheritance."""
        error = RiskManagementError("Risk management error")
        assert isinstance(error, TradingBotError)
        assert isinstance(error, RiskManagementError)
    
    def test_position_limit_error(self):
        """Test PositionLimitError."""
        error = PositionLimitError("Position limit exceeded")
        assert isinstance(error, RiskManagementError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Position limit exceeded"
    
    def test_drawdown_limit_error(self):
        """Test DrawdownLimitError."""
        error = DrawdownLimitError("Drawdown limit exceeded")
        assert isinstance(error, RiskManagementError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Drawdown limit exceeded"


class TestValidationExceptions:
    """Test validation exceptions."""
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inheritance."""
        error = ValidationError("Validation error")
        assert isinstance(error, TradingBotError)
        assert isinstance(error, ValidationError)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Configuration error")
        assert isinstance(error, ValidationError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Configuration error"
    
    def test_schema_validation_error(self):
        """Test SchemaValidationError."""
        error = SchemaValidationError("Schema validation failed")
        assert isinstance(error, ValidationError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Schema validation failed"


class TestExecutionExceptions:
    """Test execution exceptions."""
    
    def test_execution_error_inheritance(self):
        """Test ExecutionError inheritance."""
        error = ExecutionError("Execution error")
        assert isinstance(error, TradingBotError)
        assert isinstance(error, ExecutionError)
    
    def test_order_rejection_error(self):
        """Test OrderRejectionError."""
        error = OrderRejectionError("Order rejected")
        assert isinstance(error, ExecutionError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Order rejected"


class TestDataExceptions:
    """Test data-related exceptions."""
    
    def test_data_error_inheritance(self):
        """Test DataError inheritance."""
        error = DataError("Data error")
        assert isinstance(error, TradingBotError)
        assert isinstance(error, DataError)
    
    def test_data_validation_error(self):
        """Test DataValidationError."""
        error = DataValidationError("Data validation failed")
        assert isinstance(error, DataError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Data validation failed"


class TestModelExceptions:
    """Test ML model exceptions."""
    
    def test_model_error_inheritance(self):
        """Test ModelError inheritance."""
        error = ModelError("Model error")
        assert isinstance(error, TradingBotError)
        assert isinstance(error, ModelError)
    
    def test_model_load_error(self):
        """Test ModelLoadError."""
        error = ModelLoadError("Model loading failed")
        assert isinstance(error, ModelError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Model loading failed"


class TestStateConsistencyExceptions:
    """Test state consistency exceptions."""
    
    def test_state_consistency_error_inheritance(self):
        """Test StateConsistencyError inheritance."""
        error = StateConsistencyError("State consistency error")
        assert isinstance(error, TradingBotError)
        assert isinstance(error, StateConsistencyError)
    
    def test_state_corruption_error(self):
        """Test StateCorruptionError."""
        error = StateCorruptionError("State corruption detected")
        assert isinstance(error, StateConsistencyError)
        assert isinstance(error, TradingBotError)
        assert error.message == "State corruption detected"


class TestSecurityExceptions:
    """Test security exceptions."""
    
    def test_security_error_inheritance(self):
        """Test SecurityError inheritance."""
        error = SecurityError("Security error")
        assert isinstance(error, TradingBotError)
        assert isinstance(error, SecurityError)
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Authentication failed")
        assert isinstance(error, SecurityError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Authentication failed"
    
    def test_authorization_error(self):
        """Test AuthorizationError."""
        error = AuthorizationError("Authorization failed")
        assert isinstance(error, SecurityError)
        assert isinstance(error, TradingBotError)
        assert error.message == "Authorization failed"


class TestExceptionHierarchy:
    """Test exception hierarchy relationships."""
    
    def test_exception_hierarchy_inheritance(self):
        """Test that all exceptions properly inherit from TradingBotError."""
        exceptions = [
            ExchangeError("test"),
            RiskManagementError("test"),
            ValidationError("test"),
            ExecutionError("test"),
            ModelError("test"),
            DataError("test"),
            StateConsistencyError("test"),
            SecurityError("test")
        ]
        
        for exception in exceptions:
            assert isinstance(exception, TradingBotError)
    
    def test_specific_exception_inheritance(self):
        """Test that specific exceptions inherit from their parent categories."""
        # Exchange exceptions
        assert isinstance(ExchangeConnectionError("test"), ExchangeError)
        assert isinstance(ExchangeRateLimitError("test"), ExchangeError)
        assert isinstance(ExchangeInsufficientFundsError("test"), ExchangeError)
        
        # Risk management exceptions
        assert isinstance(PositionLimitError("test"), RiskManagementError)
        assert isinstance(DrawdownLimitError("test"), RiskManagementError)
        
        # Validation exceptions
        assert isinstance(ConfigurationError("test"), ValidationError)
        assert isinstance(SchemaValidationError("test"), ValidationError)
        
        # Execution exceptions
        assert isinstance(OrderRejectionError("test"), ExecutionError)
        
        # Data exceptions
        assert isinstance(DataValidationError("test"), DataError)
        
        # Model exceptions
        assert isinstance(ModelLoadError("test"), ModelError)
        
        # State consistency exceptions
        assert isinstance(StateCorruptionError("test"), StateConsistencyError)
        
        # Security exceptions
        assert isinstance(AuthenticationError("test"), SecurityError)
        assert isinstance(AuthorizationError("test"), SecurityError) 