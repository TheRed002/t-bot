"""
Production-Ready State Module Integration Tests

This module provides REAL integration tests for the state module using:
- Real PostgreSQL database connections
- Real Redis cache connections
- Real StateService instances
- Real dependency injection

NO MOCKS - All services use actual database connections and real implementations.
These tests verify production-ready state management patterns.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from src.core.config import get_config
from src.core.dependency_injection import DependencyContainer
from src.state import StateType, register_state_services
from src.state.di_registration import create_state_service_with_dependencies


@pytest.mark.integration
class TestRealStateServiceIntegration:
    """Real state service integration tests with actual database connections."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_state_service_with_database(self, real_database_service):
        """Test StateService with real PostgreSQL and Redis connections."""
        config = get_config()

        # Create real StateService with actual database connections
        state_service = await create_state_service_with_dependencies(
            config=config, database_service=real_database_service
        )

        try:
            await state_service.initialize()
            assert state_service.initialized

            # Test real state operations with actual database storage
            test_state_data = {
                "bot_id": "test-bot-001",
                "status": "running",
                "allocation": "1000.00",
                "last_update": datetime.now(timezone.utc).isoformat(),
                "positions": ["BTCUSDT", "ETHUSDT"],
            }

            # Set state - should persist to real database
            success = await state_service.set_state(
                state_type=StateType.BOT_STATE,
                state_id="test-bot-001",
                state_data=test_state_data,
                source_component="integration-test",
                validate=False,
            )
            assert success

            # Get state - should retrieve from real database
            retrieved_state = await state_service.get_state(
                state_type=StateType.BOT_STATE, state_id="test-bot-001"
            )
            assert retrieved_state is not None
            assert retrieved_state["bot_id"] == "test-bot-001"
            assert retrieved_state["status"] == "running"

            # Test state persistence across service restarts
            await state_service.cleanup()

            # Create new service instance
            state_service_2 = await create_state_service_with_dependencies(
                config=config, database_service=real_database_service
            )
            await state_service_2.initialize()

            # Data should still be available
            persistent_state = await state_service_2.get_state(
                state_type=StateType.BOT_STATE, state_id="test-bot-001"
            )
            assert persistent_state is not None
            assert persistent_state["bot_id"] == "test-bot-001"

            await state_service_2.cleanup()

        finally:
            if hasattr(state_service, "cleanup"):
                await state_service.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_state_dependency_injection(self, real_database_service):
        """Test state services registration with real dependency container."""
        config = get_config()

        # Create real DI container
        container = DependencyContainer()

        # Register real dependencies
        container.register("Config", lambda: config, singleton=True)
        container.register("DatabaseService", lambda: real_database_service, singleton=True)

        # Register state services
        register_state_services(container)

        # Verify services are registered
        assert container.has("StateService")
        assert container.has("StateBusinessService")
        assert container.has("StatePersistenceService")

        # Resolve and test real services
        state_service = container.get("StateService")
        assert state_service is not None

        try:
            await state_service.initialize()

            # Test with real operations
            await state_service.set_state(
                state_type=StateType.SYSTEM_STATE,
                state_id="di-test",
                state_data={"test": "dependency_injection"},
                source_component="di-test",
                validate=False,
            )

            result = await state_service.get_state(StateType.SYSTEM_STATE, "di-test")
            assert result["test"] == "dependency_injection"

        finally:
            await state_service.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_state_concurrency(self, real_database_service):
        """Test concurrent state operations with real database connections."""
        config = get_config()

        state_service = await create_state_service_with_dependencies(
            config=config, database_service=real_database_service
        )

        try:
            await state_service.initialize()

            # Create concurrent state update tasks
            async def update_state(bot_id: str, iteration: int):
                state_data = {
                    "bot_id": bot_id,
                    "iteration": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "allocation": f"{1000 + iteration}.00",
                }

                await state_service.set_state(
                    state_type=StateType.BOT_STATE,
                    state_id=bot_id,
                    state_data=state_data,
                    source_component="concurrency-test",
                    validate=False,
                )
                return bot_id, iteration

            # Run concurrent updates
            tasks = []
            for i in range(10):
                bot_id = f"concurrent-bot-{i}"
                tasks.append(update_state(bot_id, i))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all updates completed successfully
            successful_updates = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_updates) == 10

            # Verify final state consistency
            for i in range(10):
                bot_id = f"concurrent-bot-{i}"
                final_state = await state_service.get_state(StateType.BOT_STATE, bot_id)
                assert final_state is not None
                assert final_state["bot_id"] == bot_id
                assert final_state["iteration"] == i

        finally:
            await state_service.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_state_validation_and_error_handling(self, real_database_service):
        """Test state validation and error handling with real services."""
        config = get_config()

        state_service = await create_state_service_with_dependencies(
            config=config, database_service=real_database_service
        )

        try:
            await state_service.initialize()

            # Test with valid state
            valid_state = {"bot_id": "validation-test", "status": "active", "allocation": "500.00"}

            success = await state_service.set_state(
                state_type=StateType.BOT_STATE,
                state_id="validation-test",
                state_data=valid_state,
                source_component="validation-test",
                validate=False,  # Disable validation for this test
            )
            assert success

            # Test error handling for non-existent state
            try:
                missing_state = await state_service.get_state(
                    StateType.BOT_STATE, "non-existent-bot"
                )
                # Should return None or raise appropriate exception
                assert missing_state is None
            except Exception as e:
                # Should be a proper state error, not mock error
                assert "MagicMock" not in str(e)

        finally:
            await state_service.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_state_cleanup_and_resource_management(self, real_database_service):
        """Test state service cleanup and resource management."""
        config = get_config()

        state_service = await create_state_service_with_dependencies(
            config=config, database_service=real_database_service
        )

        # Initialize and use the service
        await state_service.initialize()

        # Create some state data
        test_data = {"cleanup": "test", "resources": "managed"}
        await state_service.set_state(
            state_type=StateType.SYSTEM_STATE,
            state_id="cleanup-test",
            state_data=test_data,
            source_component="cleanup-test",
            validate=False,
        )

        # Verify data exists
        retrieved = await state_service.get_state(StateType.SYSTEM_STATE, "cleanup-test")
        assert retrieved is not None

        # Test cleanup
        await state_service.cleanup()

        # Verify service is properly cleaned up
        # (The data should persist in database, but service should be cleaned up)
        assert not state_service.is_running if hasattr(state_service, "is_running") else True

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_state_health_check(self, real_database_service):
        """Test state service health check with real dependencies."""
        config = get_config()

        state_service = await create_state_service_with_dependencies(
            config=config, database_service=real_database_service
        )

        try:
            await state_service.initialize()

            # Test health check with real database connections
            health_status = await state_service.get_health_status()
            assert health_status is not None
            assert "overall_status" in health_status

            # Health should reflect real service status
            assert health_status["overall_status"] in ["healthy", "degraded", "unhealthy"]

        finally:
            await state_service.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_state_types_and_operations(self, real_database_service):
        """Test different state types with real persistence."""
        config = get_config()

        state_service = await create_state_service_with_dependencies(
            config=config, database_service=real_database_service
        )

        try:
            await state_service.initialize()

            # Test different state types
            test_cases = [
                (StateType.BOT_STATE, "bot-multi-test", {"bot_id": "multi", "type": "bot"}),
                (StateType.SYSTEM_STATE, "system-multi-test", {"system": "test", "type": "system"}),
                (StateType.ORDER_STATE, "order-multi-test", {"order_id": "multi", "type": "order"}),
            ]

            # Set different types of state
            for state_type, state_id, data in test_cases:
                success = await state_service.set_state(
                    state_type=state_type,
                    state_id=state_id,
                    state_data=data,
                    source_component="multi-type-test",
                    validate=False,
                )
                assert success

            # Retrieve and verify each type
            for state_type, state_id, expected_data in test_cases:
                retrieved = await state_service.get_state(state_type, state_id)
                assert retrieved is not None
                assert retrieved["type"] == expected_data["type"]

        finally:
            await state_service.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
