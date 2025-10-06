"""
Integration tests for execution module proper dependency injection and usage.

This test validates that:
1. ExecutionService properly uses injected DatabaseService
2. ExecutionEngine properly uses injected ExecutionService
3. Service layer boundaries are respected
4. Error handling propagates correctly
5. Module interfaces are used correctly
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

# pytestmark = pytest.mark.skip("Execution module tests need comprehensive setup")  # ENABLED for real services testing
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionStatus,
    MarketData,
    OrderSide,
    OrderType,
)
from src.execution.di_registration import ExecutionModuleDIRegistration
from src.execution.execution_orchestration_service import ExecutionOrchestrationService
from src.execution.service import ExecutionService


class MockDependencyContainer:
    """Mock dependency container for testing."""

    def __init__(self):
        self._services = {}

    def register(self, key, factory, singleton=True, force=False):
        """Register a service."""
        self._services[key] = factory

    def get(self, key):
        """Get a service."""
        if key in self._services:
            factory = self._services[key]
            if callable(factory):
                return factory(self)
            return factory
        raise KeyError(f"Service {key} not found")

    def get_optional(self, key):
        """Get an optional service."""
        try:
            return self.get(key)
        except KeyError:
            return None

    def is_registered(self, key):
        """Check if service is registered."""
        return key in self._services

    def has_service(self, key):
        """Check if service exists (alias for is_registered for compatibility)."""
        return key in self._services

    def resolve(self, key):
        """Resolve a service (alias for get for compatibility)."""
        return self.get(key)


@pytest.fixture
def mock_database_service():
    """Create mock database service."""
    mock_db = AsyncMock()
    mock_db.create_entity = AsyncMock()
    mock_db.list_entities = AsyncMock(return_value=[])
    mock_db.health_check = AsyncMock(return_value="healthy")
    mock_db.is_running = True
    return mock_db


@pytest.fixture
def mock_risk_service():
    """Create mock risk service."""
    mock_risk = AsyncMock()
    mock_risk.validate_order = AsyncMock(return_value=True)
    mock_risk.validate_signal = AsyncMock(return_value=True)
    mock_risk.calculate_position_size = AsyncMock(return_value=Decimal("100"))
    mock_risk.calculate_risk_metrics = AsyncMock(return_value={"portfolio_var": 0.05})
    mock_risk.get_risk_summary = AsyncMock(return_value={"total_risk_level": "low"})
    mock_risk.is_running = True
    return mock_risk


@pytest.fixture
def mock_validation_service():
    """Create mock validation service."""
    mock_validation = AsyncMock()
    mock_validation.validate_quantity = AsyncMock()
    return mock_validation


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.execution = MagicMock()

    # Set up the execution config values
    config.execution.max_order_value = Decimal("100000")
    config.execution.default_slippage_tolerance = Decimal("0.001")

    # Configure the get method to return proper values
    def get_execution_value(key, default=None):
        values = {
            "max_order_value": Decimal("100000"),
            "default_slippage_tolerance": Decimal("0.001"),
            "default_daily_volume": "1000000",
        }
        return values.get(key, default)

    config.execution.get = get_execution_value
    return config


@pytest.fixture
def container_with_mocks(
    mock_database_service, mock_risk_service, mock_validation_service, mock_config
):
    """Create container with mock dependencies."""
    container = MockDependencyContainer()

    # Register mock services - ensure Config is properly registered
    container.register("DatabaseService", lambda c: mock_database_service)
    container.register("RiskService", lambda c: mock_risk_service)
    container.register("ValidationService", lambda c: mock_validation_service)
    container.register("Config", lambda c: mock_config)
    container.register("MetricsService", lambda c: None)
    container.register("AnalyticsService", lambda c: None)
    container.register("RedisClient", lambda c: None)
    container.register("StateService", lambda c: None)
    container.register("MetricsCollector", lambda c: None)
    container.register("ExchangeFactory", lambda c: None)
    container.register("TradeLifecycleManager", lambda c: None)

    return container


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_execution_service_dependency_injection(
    container_with_mocks, mock_database_service, mock_risk_service
):
    """Test that ExecutionService properly uses injected dependencies."""

    # Create ExecutionService with injected dependencies
    execution_service = ExecutionService(
        repository_service=mock_database_service,
        risk_service=mock_risk_service,
        validation_service=container_with_mocks.get("ValidationService"),
        correlation_id="test-correlation-id",
    )

    # Start the service
    await execution_service.start()

    # Verify dependencies are properly set
    assert execution_service.repository_service is mock_database_service
    assert execution_service.risk_service is mock_risk_service
    assert execution_service.validation_service is not None

    # Verify service doesn't bypass dependencies
    assert not hasattr(execution_service, "_direct_db_connection")
    assert not hasattr(execution_service, "_direct_risk_client")

    await execution_service.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_execution_di_registration(container_with_mocks, mock_config):
    """Test execution module DI registration creates proper service layer."""

    # Register execution module
    di_registration = ExecutionModuleDIRegistration(container_with_mocks, mock_config)
    di_registration.register_all()

    # Verify core services are registered
    assert container_with_mocks.is_registered("ExecutionService")
    assert container_with_mocks.is_registered(ExecutionOrchestrationService)

    # Verify services can be resolved
    execution_service = container_with_mocks.get("ExecutionService")
    assert execution_service is not None

    orchestration_service = container_with_mocks.get(ExecutionOrchestrationService)
    assert orchestration_service is not None

    # Verify dependencies are properly injected
    assert execution_service.repository_service is not None


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_service_layer_boundaries_respected():
    """Test that service layer boundaries are properly respected."""

    # Mock database service to track calls
    mock_db = AsyncMock()
    mock_db.create_entity = AsyncMock()
    mock_db.is_running = True

    # Create ExecutionService
    execution_service = ExecutionService(
        repository_service=mock_db, correlation_id="test-boundary-check"
    )

    await execution_service.start()

    # Create sample execution result
    execution_result = MagicMock()
    execution_result.execution_id = "test-execution-123"
    execution_result.original_order = MagicMock()
    execution_result.original_order.symbol = "BTCUSDT"
    execution_result.original_order.side = OrderSide.BUY
    execution_result.original_order.order_type = OrderType.MARKET
    execution_result.original_order.quantity = Decimal("1.0")
    execution_result.original_order.exchange = "binance"
    execution_result.total_filled_quantity = Decimal("1.0")
    execution_result.average_fill_price = Decimal("50000.0")
    execution_result.total_fees = Decimal("0.1")
    execution_result.status = ExecutionStatus.COMPLETED
    execution_result.algorithm = ExecutionAlgorithm.MARKET
    execution_result.execution_duration = 1.5

    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        open=Decimal("49000.0"),
        high=Decimal("51000.0"),
        low=Decimal("48000.0"),
        close=Decimal("50000.0"),
        volume=Decimal("1000.0"),
        exchange="binance",
        bid_price=Decimal("49999.0"),
        ask_price=Decimal("50001.0"),
    )

    # Record execution - should use database service, not direct database access
    await execution_service.record_trade_execution(
        execution_result=execution_result,
        market_data=market_data,
        bot_id="test-bot",
        strategy_name="test-strategy",
    )

    # Verify database service was called (proper abstraction)
    # Should call list_orders and create_order_record, not create_entity
    assert len(mock_db.method_calls) > 0, "Database service should have been called"
    assert any(
        "list_orders" in str(call) or "create_order_record" in str(call)
        for call in mock_db.method_calls
    ), f"Expected list_orders or create_order_record calls, got: {mock_db.method_calls}"

    await execution_service.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_error_handling_propagation():
    """Test that error handling properly propagates through service layers."""

    # Create database service that raises errors only on record operations
    failing_db = AsyncMock()
    failing_db.create_entity = AsyncMock(side_effect=Exception("Database connection failed"))
    failing_db.create_order_record = AsyncMock(side_effect=Exception("Database connection failed"))
    failing_db.list_orders = AsyncMock(return_value=[])  # Allow startup to succeed
    failing_db.is_running = True

    execution_service = ExecutionService(
        repository_service=failing_db, correlation_id="test-error-propagation"
    )

    await execution_service.start()

    # Create sample data
    execution_result = MagicMock()
    execution_result.execution_id = "test-execution-456"
    execution_result.original_order = MagicMock()
    execution_result.original_order.symbol = "ETHUSDT"
    execution_result.original_order.side = OrderSide.SELL
    execution_result.original_order.order_type = OrderType.LIMIT
    execution_result.original_order.quantity = Decimal("2.0")
    execution_result.original_order.price = Decimal("3000.0")
    execution_result.original_order.exchange = "coinbase"
    execution_result.total_filled_quantity = Decimal("2.0")
    execution_result.average_fill_price = Decimal("3000.0")
    execution_result.total_fees = Decimal("0.05")
    execution_result.status = ExecutionStatus.COMPLETED
    execution_result.algorithm = ExecutionAlgorithm.LIMIT
    execution_result.execution_duration = 2.1

    market_data = MarketData(
        symbol="ETHUSDT",
        timestamp=datetime.now(timezone.utc),
        open=Decimal("2950.0"),
        high=Decimal("3100.0"),
        low=Decimal("2900.0"),
        close=Decimal("3000.0"),
        volume=Decimal("500.0"),
        exchange="coinbase",
        bid_price=Decimal("2999.0"),
        ask_price=Decimal("3001.0"),
    )

    # Attempt execution - should handle error gracefully
    with pytest.raises(Exception):  # ServiceError or similar
        await execution_service.record_trade_execution(
            execution_result=execution_result,
            market_data=market_data,
            bot_id="test-bot",
            strategy_name="test-strategy",
        )

    await execution_service.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_orchestration_service_coordinates_properly(container_with_mocks, mock_config):
    """Test that ExecutionOrchestrationService properly coordinates service calls."""

    # Setup mocks
    mock_execution_service = AsyncMock()
    mock_order_service = AsyncMock()
    mock_engine_service = AsyncMock()

    # Create orchestration service
    orchestration_service = ExecutionOrchestrationService(
        execution_service=mock_execution_service,
        order_manager=mock_order_service,
        execution_engine=mock_engine_service,
        correlation_id="test-orchestration",
    )

    await orchestration_service.start()

    # Verify dependencies are set
    assert orchestration_service.execution_service is mock_execution_service
    assert orchestration_service.order_manager is mock_order_service
    assert orchestration_service.execution_engine is mock_engine_service

    await orchestration_service.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_no_direct_database_access():
    """Test that execution module doesn't bypass service layer for database access."""

    # Verify these classes don't import database models directly
    import inspect

    from src.execution.execution_engine import ExecutionEngine
    from src.execution.service import ExecutionService

    # Check ExecutionService source
    execution_service_source = inspect.getsource(ExecutionService)

    # Should use repository service, not direct model imports
    assert "repository_service" in execution_service_source.lower()
    # Should not have direct database connections
    assert "Session(" not in execution_service_source
    assert "session.execute(" not in execution_service_source

    # Check ExecutionEngine source
    execution_engine_source = inspect.getsource(ExecutionEngine)

    # Should use ExecutionService for data operations
    assert "execution_service" in execution_engine_source.lower()
    # Should not have direct database operations
    assert "Session(" not in execution_engine_source
    assert "session.query(" not in execution_engine_source


def test_proper_interface_usage():
    """Test that modules use interfaces correctly."""

    # Import BotInstance to verify it uses interfaces
    # Check the class signature uses interface
    import inspect

    from src.bot_management.bot_instance import BotInstance

    signature = inspect.signature(BotInstance.__init__)

    # Should accept ExecutionEngineServiceInterface, not concrete ExecutionEngine
    params = signature.parameters
    assert "execution_engine_service" in params

    # Check type hint (if available)
    param = params["execution_engine_service"]
    if param.annotation != inspect.Parameter.empty:
        # Should be interface type or union with interface
        assert "Interface" in str(param.annotation)


if __name__ == "__main__":
    asyncio.run(test_execution_service_dependency_injection())
    asyncio.run(test_execution_di_registration())
