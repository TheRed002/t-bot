"""
Test suite for P-001 Core Framework Foundation.

This test file validates that all core framework components are working correctly
and can be imported and used as expected.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any

# Import core framework components
from src.core.types import (
    TradingMode, SignalDirection, OrderSide, OrderType,
    Signal, MarketData, OrderRequest, OrderResponse, Position
)

from src.core.config import Config, DatabaseConfig, SecurityConfig

from src.core.exceptions import (
    TradingBotError, ExchangeError, RiskManagementError,
    ValidationError, ExecutionError, ModelError, DataError,
    StateConsistencyError, SecurityError, ExchangeConnectionError,
    ExchangeRateLimitError, ExchangeInsufficientFundsError,
    PositionLimitError, DrawdownLimitError, ConfigurationError,
    SchemaValidationError
)

from src.core.logging import (
    get_logger, setup_logging, log_performance, log_async_performance,
    get_secure_logger, PerformanceMonitor, correlation_context
)

from src.main import Application


class TestCoreTypes:
    """Test core type definitions."""
    
    def test_trading_mode_enum(self):
        """Test TradingMode enum values."""
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.BACKTEST.value == "backtest"
    
    def test_signal_direction_enum(self):
        """Test SignalDirection enum values."""
        assert SignalDirection.BUY.value == "buy"
        assert SignalDirection.SELL.value == "sell"
        assert SignalDirection.HOLD.value == "hold"
    
    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_LOSS.value == "stop_loss"
        assert OrderType.TAKE_PROFIT.value == "take_profit"
    
    def test_signal_creation(self):
        """Test Signal model creation and validation."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            strategy_name="test_strategy",
            metadata={"test": "data"}
        )
        
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == 0.8
        assert signal.symbol == "BTC/USDT"
        assert signal.strategy_name == "test_strategy"
        assert signal.metadata["test"] == "data"
    
    def test_signal_confidence_validation(self):
        """Test Signal confidence validation."""
        # Valid confidence
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.5,
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            strategy_name="test_strategy"
        )
        assert signal.confidence == 0.5
        
        # Invalid confidence should raise validation error
        with pytest.raises(ValueError):
            Signal(
                direction=SignalDirection.BUY,
                confidence=1.5,  # Invalid: > 1.0
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                strategy_name="test_strategy"
            )
    
    def test_market_data_creation(self):
        """Test MarketData model creation."""
        market_data = MarketData(
            symbol="BTC/USDT",
            price=Decimal("50000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999.00"),
            ask=Decimal("50001.00"),
            open_price=Decimal("49900.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49800.00")
        )
        
        assert market_data.symbol == "BTC/USDT"
        assert market_data.price == Decimal("50000.00")
        assert market_data.volume == Decimal("100.5")
        assert market_data.bid == Decimal("49999.00")
        assert market_data.ask == Decimal("50001.00")
    
    def test_order_request_creation(self):
        """Test OrderRequest model creation."""
        order_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            time_in_force="GTC",
            client_order_id="test_order_123"
        )
        
        assert order_request.symbol == "BTC/USDT"
        assert order_request.side == OrderSide.BUY
        assert order_request.order_type == OrderType.LIMIT
        assert order_request.quantity == Decimal("1.0")
        assert order_request.price == Decimal("50000.00")
        assert order_request.client_order_id == "test_order_123"
    
    def test_position_creation(self):
        """Test Position model creation."""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("2.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00"),
            unrealized_pnl=Decimal("2000.00"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert position.symbol == "BTC/USDT"
        assert position.quantity == Decimal("2.0")
        assert position.entry_price == Decimal("50000.00")
        assert position.current_price == Decimal("51000.00")
        assert position.unrealized_pnl == Decimal("2000.00")
        assert position.side == OrderSide.BUY


class TestCoreExceptions:
    """Test exception hierarchy."""
    
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
    
    def test_exchange_exceptions(self):
        """Test exchange-related exceptions."""
        # Test inheritance
        connection_error = ExchangeConnectionError("Connection failed")
        assert isinstance(connection_error, ExchangeError)
        assert isinstance(connection_error, TradingBotError)
        
        rate_limit_error = ExchangeRateLimitError("Rate limit exceeded")
        assert isinstance(rate_limit_error, ExchangeError)
        
        insufficient_funds = ExchangeInsufficientFundsError("Insufficient funds")
        assert isinstance(insufficient_funds, ExchangeError)
    
    def test_risk_management_exceptions(self):
        """Test risk management exceptions."""
        position_limit = PositionLimitError("Position limit exceeded")
        assert isinstance(position_limit, RiskManagementError)
        
        drawdown_limit = DrawdownLimitError("Drawdown limit exceeded")
        assert isinstance(drawdown_limit, RiskManagementError)
    
    def test_validation_exceptions(self):
        """Test validation exceptions."""
        config_error = ConfigurationError("Configuration error")
        assert isinstance(config_error, ValidationError)
        
        schema_error = SchemaValidationError("Schema validation failed")
        assert isinstance(schema_error, ValidationError)


class TestCoreLogging:
    """Test logging system."""
    
    def test_logger_creation(self):
        """Test logger creation and basic functionality."""
        logger = get_logger(__name__)
        assert logger is not None
        
        # Test secure logger
        secure_logger = get_secure_logger(__name__)
        assert secure_logger is not None
    
    def test_correlation_context(self):
        """Test correlation context functionality."""
        correlation_id = correlation_context.generate_correlation_id()
        assert correlation_id is not None
        assert len(correlation_id) > 0
        
        # Test context manager
        with correlation_context.correlation_context("test_correlation_id"):
            assert correlation_context.get_correlation_id() == "test_correlation_id"
    
    def test_performance_decorator(self):
        """Test performance logging decorator."""
        @log_performance
        def test_function():
            return "test_result"
        
        result = test_function()
        assert result == "test_result"
    
    @pytest.mark.asyncio
    async def test_async_performance_decorator(self):
        """Test async performance logging decorator."""
        @log_async_performance
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "test_async_result"
        
        result = await test_async_function()
        assert result == "test_async_result"
    
    def test_performance_monitor(self):
        """Test performance monitor context manager."""
        with PerformanceMonitor("test_operation"):
            # Simulate some work
            pass


class TestCoreConfiguration:
    """Test configuration system."""
    
    def test_config_creation(self):
        """Test configuration creation with defaults."""
        # This will use environment variables or defaults
        config = Config()
        
        assert config.app_name == "trading-bot-suite"
        assert config.app_version == "2.0.0"
        assert config.environment in ["development", "staging", "production"]
    
    def test_database_config(self):
        """Test database configuration."""
        db_config = DatabaseConfig()
        
        # Test default values
        assert db_config.postgresql_port == 5432
        assert db_config.redis_port == 6379
        assert db_config.influxdb_port == 8086
    
    def test_security_config(self):
        """Test security configuration."""
        security_config = SecurityConfig()
        
        # Test default values
        assert security_config.jwt_algorithm == "HS256"
        assert security_config.jwt_expire_minutes == 30
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test environment validation
        with pytest.raises(ValueError):
            config = Config()
            config.environment = "invalid_environment"
            config.validate_environment(config.environment)


class TestMainApplication:
    """Test main application functionality."""
    
    @pytest.mark.asyncio
    async def test_application_creation(self):
        """Test application creation and basic functionality."""
        app = Application()
        assert app is not None
        assert app.config is None  # Not initialized yet
        assert app.components == {}
    
    def test_health_status(self):
        """Test health status functionality."""
        app = Application()
        health_status = app.get_health_status()
        
        assert "status" in health_status
        assert "components" in health_status
        assert health_status["status"] == "starting"


# Integration tests
class TestIntegration:
    """Integration tests for core framework."""
    
    def test_type_imports(self):
        """Test that all types can be imported correctly."""
        from src.core.types import (
            TradingMode, SignalDirection, OrderSide, OrderType,
            Signal, MarketData, OrderRequest, OrderResponse, Position
        )
        assert True  # If we get here, imports worked
    
    def test_config_imports(self):
        """Test that all config classes can be imported correctly."""
        from src.core.config import Config, DatabaseConfig, SecurityConfig
        assert True  # If we get here, imports worked
    
    def test_exception_imports(self):
        """Test that all exceptions can be imported correctly."""
        from src.core.exceptions import (
            TradingBotError, ExchangeError, RiskManagementError,
            ValidationError, ExecutionError, ModelError, DataError,
            StateConsistencyError, SecurityError
        )
        assert True  # If we get here, imports worked
    
    def test_logging_imports(self):
        """Test that all logging functions can be imported correctly."""
        from src.core.logging import (
            get_logger, setup_logging, log_performance, log_async_performance,
            get_secure_logger, PerformanceMonitor, correlation_context
        )
        assert True  # If we get here, imports worked


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 