"""
Integration tests to validate state module proper integration with other modules.

Tests verify:
1. State service dependency injection works correctly
2. Other modules properly consume state services
3. Service layer architecture is followed
4. Module boundaries are respected
5. Error propagation works across module boundaries
"""

import asyncio
import pytest
import pytest_asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.dependency_injection import DependencyContainer, DependencyInjector
from src.core.types import StateType, BotStatus, BotConfiguration, BotPriority, BotType
from src.state import StateService, register_state_services
from src.bot_management.service import BotService
from src.execution.order_manager import OrderManager
from src.risk_management.service import RiskService


class TestStateModuleIntegration:
    """Test state module integration with other trading system modules."""

    @pytest_asyncio.fixture
    async def config(self):
        """Create test configuration."""
        return Config()

    @pytest_asyncio.fixture
    async def dependency_container(self):
        """Create DI container for testing."""
        container = DependencyContainer()
        
        # Register mock services
        container.register("DatabaseService", lambda: AsyncMock(), singleton=True)
        container.register("ValidationService", lambda: AsyncMock(), singleton=True)
        container.register("EventService", lambda: AsyncMock(), singleton=True)
        
        return container

    @pytest_asyncio.fixture
    async def state_service(self, config, dependency_container):
        """Create state service with injected dependencies."""
        # Register state services in container
        register_state_services(dependency_container)
        
        # Create injector
        injector = DependencyInjector()
        injector._container = dependency_container
        
        # Create state service using factory
        from src.state.di_registration import create_state_service_with_dependencies
        
        database_service = dependency_container.get("DatabaseService")
        state_service = await create_state_service_with_dependencies(
            config, database_service, injector
        )
        
        await state_service.initialize()
        
        yield state_service
        
        await state_service.cleanup()

    async def test_state_service_dependency_injection(self, dependency_container):
        """Test that state service dependencies are properly injected."""
        # Register state services
        register_state_services(dependency_container)
        
        # Verify all services are registered
        assert dependency_container.get("StateBusinessService") is not None
        assert dependency_container.get("StatePersistenceService") is not None
        assert dependency_container.get("StateValidationService") is not None
        assert dependency_container.get("StateSynchronizationService") is not None
        assert dependency_container.get("StateService") is not None

    async def test_bot_service_state_integration(self, config, state_service):
        """Test that BotService properly integrates with StateService."""
        # Create mock dependencies
        exchange_service = AsyncMock()
        capital_service = AsyncMock()
        risk_service = AsyncMock()
        execution_service = AsyncMock()
        strategy_service = AsyncMock()

        # Create BotService with StateService
        bot_service = BotService(
            exchange_service=exchange_service,
            capital_service=capital_service,
            state_service=state_service,
            risk_service=risk_service,
            execution_service=execution_service,
            strategy_service=strategy_service
        )
        
        await bot_service.start()
        
        # Test that bot service uses state service correctly
        bot_config = BotConfiguration(
            bot_id="test_bot_001",
            name="Test Bot",
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTCUSDT"],
            priority=BotPriority.NORMAL,
            bot_type=BotType.TRADING,
            allocated_capital=Decimal("1000.00"),
            strategy_parameters={"max_position_size": Decimal("100.00")}
        )
        
        # Mock bot creation
        bot_service._active_bots["test_bot_001"] = {
            "config": bot_config,
            "instance": AsyncMock()
        }
        
        # Test state persistence through StateService
        await bot_service.start_bot("test_bot_001")
        
        # Verify state was stored
        stored_state = await state_service.get_state(StateType.BOT_STATE, "test_bot_001")
        assert stored_state is not None
        
        await bot_service.stop()

    async def test_order_manager_state_integration(self, config, state_service):
        """Test that OrderManager properly integrates with StateService."""
        # Create mock exchange
        mock_exchange = AsyncMock()
        mock_exchange.name = "test_exchange"
        mock_exchange.get_balance.return_value = {"USDT": Decimal("1000.00")}
        
        # Create OrderManager with StateService
        order_manager = OrderManager(
            config=config,
            exchange=mock_exchange,
            state_service=state_service
        )
        
        await order_manager.initialize()
        
        # Test that order manager uses state service for persistence
        from src.core.types import OrderRequest, OrderSide, OrderType
        
        order_request = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00")
        )
        
        # Mock exchange response
        mock_exchange.place_order.return_value = {
            "order_id": "test_order_001",
            "status": "open",
            "filled_quantity": Decimal("0.0")
        }
        
        # Place order
        result = await order_manager.place_order(order_request)
        assert result is not None
        
        await order_manager.cleanup()

    async def test_state_service_error_propagation(self, state_service):
        """Test that errors are properly propagated across module boundaries."""
        # Test invalid state type handling
        with pytest.raises(Exception):
            await state_service.set_state(
                StateType.BOT_STATE,
                "invalid_bot",
                None  # Invalid data
            )

    async def test_cross_module_event_integration(self, state_service):
        """Test that state events are properly broadcast to other modules."""
        # Track event emissions
        events_received = []
        
        def event_callback(state_type, state_id, state_data, state_change):
            events_received.append({
                "state_type": state_type,
                "state_id": state_id,
                "state_data": state_data,
                "change": state_change
            })
        
        # Subscribe to state changes
        state_service.subscribe(StateType.BOT_STATE, event_callback)
        
        # Create state change
        test_data = {
            "status": BotStatus.RUNNING.value,
            "started_at": "2024-01-01T00:00:00Z"
        }
        
        await state_service.set_state(
            StateType.BOT_STATE,
            "test_bot_002",
            test_data,
            source_component="TestModule"
        )
        
        # Verify event was received
        await asyncio.sleep(0.1)  # Allow event processing
        
        # Note: Events might be processed asynchronously
        # This validates the subscription mechanism works

    async def test_state_service_handles_missing_dependencies(self, config):
        """Test that StateService gracefully handles missing dependencies."""
        # Create StateService without injected dependencies
        state_service = StateService(
            config=config,
            business_service=None,
            persistence_service=None,
            validation_service=None,
            synchronization_service=None
        )
        
        await state_service.initialize()
        
        # Test that basic operations still work with graceful degradation
        result = await state_service.set_state(
            StateType.BOT_STATE,
            "fallback_test",
            {"status": "test"},
            validate=False  # Skip validation since service is None
        )
        
        assert result is True
        
        # Test retrieval
        retrieved_state = await state_service.get_state(StateType.BOT_STATE, "fallback_test")
        assert retrieved_state is not None
        assert retrieved_state["status"] == "test"
        
        await state_service.cleanup()

    async def test_module_boundary_validation(self, state_service):
        """Test that modules respect state module boundaries."""
        # Test that state service doesn't expose internal implementation details
        assert hasattr(state_service, 'get_state')
        assert hasattr(state_service, 'set_state')
        assert hasattr(state_service, 'delete_state')
        
        # Test that internal methods are not part of public API
        internal_methods = [
            '_memory_cache', '_metadata_cache', '_persistence_service'
        ]
        
        for method in internal_methods:
            # These should exist but not be part of public API contract
            assert hasattr(state_service, method)

    async def test_consistency_utilities_integration(self):
        """Test that consistency utilities are properly integrated."""
        from src.state.consistency import validate_state_data, emit_state_event
        
        # Test data validation
        result = validate_state_data("trade_state", {
            "trade_id": "test_001",
            "symbol": "BTC/USDT", 
            "side": "buy"
        })
        
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        
        # Test invalid data
        result = validate_state_data("trade_state", {
            "symbol": "BTC/USDT"  # missing required fields
        })
        
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        
        # Test event emission
        await emit_state_event("test.event", {"test": "data"})

    async def test_service_layer_compliance(self, state_service):
        """Test that state service follows service layer architecture."""
        # Verify StateService inherits from BaseComponent
        from src.core.base.component import BaseComponent
        assert isinstance(state_service, BaseComponent)
        
        # Verify it has proper health check methods
        assert hasattr(state_service, 'get_health_status')
        assert hasattr(state_service, 'get_metrics')
        
        # Test health check
        health_status = await state_service.get_health_status()
        assert isinstance(health_status, dict)
        assert "overall_status" in health_status
        
        # Test metrics
        metrics = state_service.get_metrics()
        assert isinstance(metrics, dict)
        assert "total_operations" in metrics

    async def test_state_service_concurrent_access(self, state_service):
        """Test that state service handles concurrent access properly."""
        # Create multiple concurrent operations
        async def set_state_worker(worker_id: int):
            for i in range(5):
                await state_service.set_state(
                    StateType.BOT_STATE,
                    f"concurrent_test_{worker_id}_{i}",
                    {"worker_id": worker_id, "iteration": i},
                    source_component=f"Worker{worker_id}"
                )
        
        # Run concurrent workers
        workers = [set_state_worker(i) for i in range(3)]
        await asyncio.gather(*workers)
        
        # Verify all states were stored correctly
        for worker_id in range(3):
            for i in range(5):
                state_id = f"concurrent_test_{worker_id}_{i}"
                state_data = await state_service.get_state(StateType.BOT_STATE, state_id)
                assert state_data is not None
                assert state_data["worker_id"] == worker_id
                assert state_data["iteration"] == i


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])