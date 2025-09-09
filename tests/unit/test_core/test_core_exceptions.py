"""
Unit tests for core exception hierarchy.

These tests verify the exception hierarchy and error handling patterns.
"""

from datetime import datetime, timezone

from src.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DataError,
    DataValidationError,
    DrawdownLimitError,
    ExchangeConnectionError,
    ExchangeError,
    ExchangeInsufficientFundsError,
    ExchangeRateLimitError,
    ExecutionError,
    ModelError,
    ModelLoadError,
    OrderRejectionError,
    PositionLimitError,
    RiskManagementError,
    SchemaValidationError,
    SecurityError,
    StateConsistencyError,
    StateCorruptionError,
    TradingBotError,
    ValidationError,
)


class TestTradingBotError:
    """Test base exception functionality."""

    def test_base_exception(self):
        """Test TradingBotError base exception."""
        error = TradingBotError(
            "Test error message", error_code="TEST_ERROR", details={"test": "data"}
        )

        assert error.message == "Test error message"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"test": "data"}
        assert error.timestamp is not None
        assert isinstance(error.timestamp, datetime)

    def test_base_exception_str_representation(self):
        """Test string representation of base exception."""
        error = TradingBotError(
            "Test error message", error_code="TEST_ERROR", details={"test": "data"}
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
            SecurityError("test"),
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


class TestExceptionErrorCodes:
    """Test exception error codes and categorization."""

    def test_error_code_assignment(self):
        """Test that error codes are properly assigned."""
        error = TradingBotError("Test error", error_code="RISK_001", details={"risk_level": "high"})

        assert error.error_code == "RISK_001"
        assert error.details["risk_level"] == "high"
        assert error.timestamp is not None

    def test_error_code_formatting(self):
        """Test error code formatting standards."""
        valid_error_codes = ["EXCH_001", "RISK_002", "EXEC_003", "DATA_004", "AUTH_005", "CFG_006"]

        for code in valid_error_codes:
            error = TradingBotError("Test", error_code=code)
            assert error.error_code == code
            assert "_" in error.error_code
            assert error.error_code.split("_")[1].isdigit()

    def test_nested_exception_details(self):
        """Test nested exception details structure."""
        error = ExchangeRateLimitError(
            "Rate limit exceeded",
            error_code="EXCH_RATE_001",
            details={
                "exchange": "binance",
                "limit": 1200,
                "window": "1min",
                "retry_after": 60,
                "request_details": {
                    "endpoint": "/api/v3/order",
                    "method": "POST",
                    "headers": {"X-MBX-APIKEY": "***REDACTED***"},
                },
            },
        )

        assert error.details["exchange"] == "binance"
        assert error.details["limit"] == 1200
        assert error.details["request_details"]["endpoint"] == "/api/v3/order"
        assert "REDACTED" in error.details["request_details"]["headers"]["X-MBX-APIKEY"]


class TestExceptionFinancialContext:
    """Test exceptions in financial trading context."""

    def test_position_limit_with_financial_details(self):
        """Test position limit error with financial context."""
        error = PositionLimitError(
            "Position size exceeds limit",
            error_code="RISK_POS_001",
            details={
                "requested_size": "15000.00",
                "max_allowed": "10000.00",
                "current_positions": "5000.00",
                "available_margin": "8000.00",
                "symbol": "BTC/USDT",
                "account_balance": "50000.00",
            },
        )

        # Verify financial data is preserved accurately
        assert error.details["requested_size"] == "15000.00"
        assert error.details["max_allowed"] == "10000.00"
        assert float(error.details["requested_size"]) > float(error.details["max_allowed"])

    def test_insufficient_funds_calculation(self):
        """Test insufficient funds error with precise calculations."""
        from decimal import Decimal

        required = Decimal("1500.50")
        available = Decimal("1200.25")
        shortfall = required - available

        error = ExchangeInsufficientFundsError(
            f"Insufficient funds: need {required}, have {available}",
            error_code="EXCH_FUNDS_001",
            details={
                "required_amount": str(required),
                "available_amount": str(available),
                "shortfall": str(shortfall),
                "currency": "USDT",
                "precision": 8,
            },
        )

        assert error.details["shortfall"] == "300.25"
        assert Decimal(error.details["required_amount"]) == required
        assert Decimal(error.details["available_amount"]) == available

    def test_drawdown_limit_breach(self):
        """Test drawdown limit error with portfolio metrics."""
        error = DrawdownLimitError(
            "Maximum drawdown exceeded",
            error_code="RISK_DD_001",
            details={
                "current_drawdown": 0.08,  # 8%
                "max_allowed": 0.05,  # 5%
                "peak_value": "100000.00",
                "current_value": "92000.00",
                "start_date": "2024-01-01",
                "breach_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        assert error.details["current_drawdown"] == 0.08
        assert error.details["current_drawdown"] > error.details["max_allowed"]

        # Verify drawdown calculation
        peak = float(error.details["peak_value"])
        current = float(error.details["current_value"])
        calculated_dd = (peak - current) / peak
        assert abs(calculated_dd - error.details["current_drawdown"]) < 0.001


class TestExceptionChaining:
    """Test exception chaining and causation."""

    def test_exception_chain_preservation(self):
        """Test that exception chains are properly preserved."""
        try:
            # Simulate nested exception scenario
            try:
                raise ConnectionError("Network timeout")
            except ConnectionError as conn_err:
                raise ExchangeConnectionError(
                    "Failed to connect to exchange", error_code="EXCH_CONN_001"
                ) from conn_err
        except ExchangeConnectionError as exc:
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, ConnectionError)
            assert str(exc.__cause__) == "Network timeout"

    def test_multi_level_exception_chain(self):
        """Test multi-level exception chains."""
        try:
            try:
                try:
                    raise ValueError("Invalid price format")
                except ValueError as val_err:
                    raise DataValidationError(
                        "Price validation failed", error_code="DATA_VAL_001"
                    ) from val_err
            except DataValidationError as data_err:
                raise OrderRejectionError(
                    "Order rejected due to invalid data", error_code="EXEC_REJ_001"
                ) from data_err
        except OrderRejectionError as order_err:
            # Check the chain
            assert order_err.__cause__ is not None
            assert isinstance(order_err.__cause__, DataValidationError)
            assert order_err.__cause__.__cause__ is not None
            assert isinstance(order_err.__cause__.__cause__, ValueError)


class TestExceptionSerialization:
    """Test exception serialization for logging and monitoring."""

    def test_exception_to_dict(self):
        """Test exception conversion to dictionary."""
        error = RiskManagementError(
            "Risk threshold breached",
            error_code="RISK_001",
            details={"threshold": 0.05, "current_value": 0.08, "action_taken": "position_reduced"},
        )

        # Convert to dict (assuming method exists or can be added)
        error_dict = {
            "message": error.message,
            "error_code": error.error_code,
            "details": error.details,
            "timestamp": error.timestamp.isoformat(),
            "type": error.__class__.__name__,
        }

        assert error_dict["message"] == "Risk threshold breached"
        assert error_dict["error_code"] == "RISK_001"
        assert error_dict["type"] == "RiskManagementError"
        assert "threshold" in error_dict["details"]

    def test_exception_json_serialization(self):
        """Test exception JSON serialization."""
        import json

        error = ExecutionError(
            "Order execution failed",
            error_code="EXEC_001",
            details={
                "order_id": "12345",
                "quantity": "1.5",  # Store as string to preserve precision
                "price": "50000.12345678",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Prepare for JSON serialization
        error_data = {
            "message": error.message,
            "error_code": error.error_code,
            "details": error.details,
            "timestamp": error.timestamp.isoformat(),
            "type": error.__class__.__name__,
        }

        # Should be JSON serializable
        json_str = json.dumps(error_data)
        assert "Order execution failed" in json_str
        assert "50000.12345678" in json_str

        # Should be deserializable
        deserialized = json.loads(json_str)
        assert deserialized["message"] == error.message
        assert deserialized["error_code"] == error.error_code
