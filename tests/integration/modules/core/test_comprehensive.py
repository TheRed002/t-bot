"""
Comprehensive Core Module Integration Tests.

This test suite validates the complete integration of the core module with
all other system modules, ensuring proper dependency injection, service
patterns, and module boundary respect.
"""

import asyncio
import pytest
import sys
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime, timezone

from src.core.base.service import BaseService, TransactionalService
from src.core.base.component import BaseComponent
from src.core.dependency_injection import (
    DependencyInjector,
    get_global_injector,
    injectable
)
from src.core.exceptions import (
    ServiceError, 
    ValidationError, 
    DependencyError,
    ComponentError
)
from src.core.types import ConfigDict
from src.core.base.interfaces import HealthStatus


class TestRealWorldIntegrationPatterns:
    """Test real-world integration patterns used throughout the system."""
    
    @pytest.mark.asyncio
    async def test_trading_workflow_integration(self):
        """Test a realistic trading workflow using core components."""
        injector = DependencyInjector()
        injector.clear()
        
        # Mock external services
        class MockMarketDataService(BaseService):
            def __init__(self):
                super().__init__("MarketDataService")
                
            async def get_ticker(self, symbol):
                return {
                    "symbol": symbol,
                    "price": Decimal("45000.00"),
                    "timestamp": datetime.now(timezone.utc)
                }
        
        class MockRiskService(BaseService):
            def __init__(self):
                super().__init__("RiskService")
                
            async def validate_order(self, order_request):
                # Basic risk validation
                return order_request.get("quantity", 0) <= 1.0
        
        class MockExecutionService(BaseService):
            def __init__(self):
                super().__init__("ExecutionService")
                self.market_data_service = None
                self.risk_service = None
                
            def configure_dependencies(self, dependency_injector):
                super().configure_dependencies(dependency_injector)
                self.market_data_service = self.resolve_dependency("MarketDataService")
                self.risk_service = self.resolve_dependency("RiskService")
                
            async def execute_order(self, order_request):
                return await self.execute_with_monitoring(
                    "execute_order",
                    self._execute_order_impl,
                    order_request
                )
                
            async def _execute_order_impl(self, order_request):
                # Get current market data
                ticker = await self.market_data_service.get_ticker(order_request["symbol"])
                
                # Validate with risk service
                risk_approved = await self.risk_service.validate_order(order_request)
                
                if not risk_approved:
                    raise ValidationError("Order rejected by risk management")
                    
                # Execute order (mock)
                return {
                    "order_id": "order_123",
                    "symbol": order_request["symbol"],
                    "quantity": order_request["quantity"],
                    "price": ticker["price"],
                    "status": "filled"
                }
        
        # Register services
        market_data = MockMarketDataService()
        risk_service = MockRiskService()
        execution_service = MockExecutionService()
        
        injector.register_service("MarketDataService", market_data, singleton=True)
        injector.register_service("RiskService", risk_service, singleton=True)
        injector.register_service("ExecutionService", execution_service, singleton=True)
        
        # Configure dependencies
        execution_service.configure_dependencies(injector)
        
        # Test workflow
        await execution_service.start()
        
        order_request = {
            "symbol": "BTC/USD",
            "quantity": 0.5,
            "type": "market"
        }
        
        result = await execution_service.execute_order(order_request)
        
        assert result["order_id"] == "order_123"
        assert result["symbol"] == "BTC/USD" 
        assert result["quantity"] == 0.5
        assert result["status"] == "filled"
        
        await execution_service.stop()
        injector.clear()
        
    @pytest.mark.asyncio
    async def test_bot_lifecycle_integration(self):
        """Test bot lifecycle management using core patterns."""
        injector = DependencyInjector()
        injector.clear()
        
        class MockBotRepository:
            def __init__(self):
                self.bots = {}
                
            async def create_bot(self, bot_data):
                bot_id = f"bot_{len(self.bots)}"
                self.bots[bot_id] = bot_data
                return bot_id
                
            async def get_bot(self, bot_id):
                return self.bots.get(bot_id)
        
        class MockBotService(BaseService):
            def __init__(self):
                super().__init__("BotService")
                self.repository = None
                
            def configure_dependencies(self, dependency_injector):
                super().configure_dependencies(dependency_injector)
                self.repository = self.resolve_dependency("BotRepository")
                
            async def create_bot(self, bot_config):
                return await self.execute_with_monitoring(
                    "create_bot",
                    self._create_bot_impl,
                    bot_config
                )
                
            async def _create_bot_impl(self, bot_config):
                # Validate configuration
                if not bot_config.get("name"):
                    raise ValidationError("Bot name is required")
                    
                # Create bot via repository
                bot_id = await self.repository.create_bot(bot_config)
                
                return {"bot_id": bot_id, "status": "created"}
        
        # Register services
        repository = MockBotRepository()
        bot_service = MockBotService()
        
        injector.register_service("BotRepository", repository, singleton=True)
        injector.register_service("BotService", bot_service, singleton=True)
        
        # Configure dependencies
        bot_service.configure_dependencies(injector)
        
        # Test bot creation workflow
        await bot_service.start()
        
        result = await bot_service.create_bot({
            "name": "TestBot",
            "strategy": "mean_reversion"
        })
        
        assert "bot_id" in result
        assert result["status"] == "created"
        
        await bot_service.stop()
        injector.clear()
        
    @pytest.mark.asyncio
    async def test_error_propagation_across_modules(self):
        """Test error propagation across module boundaries."""
        injector = DependencyInjector()
        injector.clear()
        
        class FailingService(BaseService):
            def __init__(self):
                super().__init__("FailingService")
                
            async def failing_operation(self):
                return await self.execute_with_monitoring(
                    "failing_operation",
                    self._failing_impl
                )
                
            async def _failing_impl(self):
                raise ValidationError("Service validation failed")
        
        class ConsumerService(BaseService):
            def __init__(self):
                super().__init__("ConsumerService")
                self.failing_service = None
                
            def configure_dependencies(self, dependency_injector):
                super().configure_dependencies(dependency_injector)
                self.failing_service = self.resolve_dependency("FailingService")
                
            async def use_failing_service(self):
                try:
                    return await self.failing_service.failing_operation()
                except ValidationError as e:
                    # Re-raise with additional context
                    raise ServiceError(f"Consumer service failed: {e}")
        
        # Register services
        failing_service = FailingService()
        consumer_service = ConsumerService()
        
        injector.register_service("FailingService", failing_service, singleton=True)
        injector.register_service("ConsumerService", consumer_service, singleton=True)
        
        # Configure dependencies
        consumer_service.configure_dependencies(injector)
        
        # Test error propagation
        await consumer_service.start()
        
        with pytest.raises(ServiceError, match="Consumer service failed"):
            await consumer_service.use_failing_service()
            
        await consumer_service.stop()
        injector.clear()
        
    @pytest.mark.asyncio
    async def test_transactional_service_patterns(self):
        """Test transactional service patterns."""
        injector = DependencyInjector()
        injector.clear()
        
        class MockTransactionManager:
            def __init__(self):
                self.in_transaction = False
                self.committed = False
                self.rolled_back = False

            def transaction(self):
                """Return self as the async context manager for transactions."""
                return self

            async def __aenter__(self):
                self.in_transaction = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.in_transaction = False
                if exc_type is None:
                    self.committed = True
                else:
                    self.rolled_back = True
        
        class TestTransactionalService(TransactionalService):
            def __init__(self):
                super().__init__(name="TestTransactionalService")
                
            async def transactional_operation(self, should_fail=False):
                return await self.execute_in_transaction(
                    "transactional_operation",
                    self._transactional_impl,
                    should_fail
                )
                
            async def _transactional_impl(self, should_fail):
                if should_fail:
                    raise ServiceError("Operation failed")
                return {"result": "success"}
        
        # Create service and mock transaction manager
        service = TestTransactionalService()
        tx_manager = MockTransactionManager()
        service.set_transaction_manager(tx_manager)
        
        # Test successful transaction
        result = await service.transactional_operation(should_fail=False)
        assert result["result"] == "success"
        
        injector.clear()
        
    def test_module_boundary_validation(self):
        """Test that module boundaries are properly respected."""
        # Test core types are used consistently
        from src.core.types import OrderRequest, OrderResponse, Signal
        
        # These should be the canonical types used across modules
        order = OrderRequest(
            symbol="BTC/USD",
            side="buy",
            order_type="market",
            quantity=1.0
        )
        assert order.symbol == "BTC/USD"
        
        # Test core exceptions are used consistently
        from src.core.exceptions import ServiceError, ValidationError
        
        error = ServiceError("Test error", details={"component": "test"})
        assert "Test error" in str(error)
        
    @pytest.mark.asyncio
    async def test_health_check_aggregation(self):
        """Test health check aggregation across services."""
        injector = DependencyInjector()
        injector.clear()
        
        class HealthyService(BaseService):
            def __init__(self, name):
                super().__init__(name)
                
            async def _service_health_check(self):
                return HealthStatus.HEALTHY
        
        class UnhealthyService(BaseService):
            def __init__(self, name):
                super().__init__(name)
                
            async def _service_health_check(self):
                return HealthStatus.UNHEALTHY
        
        # Create services
        healthy1 = HealthyService("HealthyService1")
        healthy2 = HealthyService("HealthyService2")
        unhealthy = UnhealthyService("UnhealthyService")
        
        # Test individual health checks
        await healthy1.start()
        await healthy2.start()
        await unhealthy.start()
        
        health1 = await healthy1.health_check()
        health2 = await healthy2.health_check()
        health3 = await unhealthy.health_check()
        
        assert health1.status == HealthStatus.HEALTHY
        assert health2.status == HealthStatus.HEALTHY
        assert health3.status == HealthStatus.UNHEALTHY
        
        await healthy1.stop()
        await healthy2.stop()
        await unhealthy.stop()
        
        injector.clear()
        
    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self):
        """Test metrics collection across services."""
        injector = DependencyInjector()
        injector.clear()
        
        class MetricsTestService(BaseService):
            def __init__(self):
                super().__init__("MetricsTestService")
                
            @pytest.mark.asyncio
            async def test_operation_success(self):
                return await self.execute_with_monitoring(
                    "test_operation",
                    self._success_impl
                )
                
            @pytest.mark.asyncio
            async def test_operation_failure(self):
                return await self.execute_with_monitoring(
                    "test_operation",
                    self._failure_impl
                )
                
            async def _success_impl(self):
                return {"result": "success"}
                
            async def _failure_impl(self):
                raise ServiceError("Test failure")
        
        service = MetricsTestService()
        await service.start()
        
        # Generate some metrics
        await service.test_operation_success()
        await service.test_operation_success()
        
        try:
            await service.test_operation_failure()
        except ServiceError:
            pass
            
        # Check metrics
        metrics = service.get_metrics()
        
        assert metrics["operations_count"] == 3
        assert metrics["operations_success"] == 2
        assert metrics["operations_error"] == 1
        assert metrics["average_response_time"] > 0
        
        await service.stop()
        injector.clear()
        
    def test_configuration_propagation(self):
        """Test configuration propagation through services."""
        config = {
            "service": {
                "timeout": 30,
                "retries": 3
            },
            "circuit_breaker": {
                "enabled": True,
                "threshold": 5
            }
        }
        
        class ConfiguredService(BaseService):
            def __init__(self, config):
                super().__init__("ConfiguredService", config)
                
            def get_timeout(self):
                return self._config.get("service", {}).get("timeout", 10)
                
            def get_retries(self):
                return self._config.get("service", {}).get("retries", 1)
        
        service = ConfiguredService(config)
        
        assert service.get_timeout() == 30
        assert service.get_retries() == 3
        
    @pytest.mark.asyncio
    async def test_concurrent_service_operations(self):
        """Test concurrent service operations."""
        injector = DependencyInjector()
        injector.clear()
        
        class ConcurrentService(BaseService):
            def __init__(self):
                super().__init__("ConcurrentService")
                self.operation_count = 0
                
            async def concurrent_operation(self, delay=0.1):
                return await self.execute_with_monitoring(
                    "concurrent_operation",
                    self._concurrent_impl,
                    delay
                )
                
            async def _concurrent_impl(self, delay):
                await asyncio.sleep(delay)
                self.operation_count += 1
                return {"operation": self.operation_count}
        
        service = ConcurrentService()
        await service.start()
        
        # Run multiple concurrent operations
        tasks = [
            service.concurrent_operation(0.05)
            for _ in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should complete
        assert len(results) == 10
        assert service.operation_count == 10
        
        # Check metrics reflect concurrent operations
        metrics = service.get_metrics()
        assert metrics["operations_count"] == 10
        assert metrics["operations_success"] == 10
        
        await service.stop()
        injector.clear()


class TestSystemIntegrationValidation:
    """Test system-wide integration validation."""
    
    @pytest.mark.asyncio
    async def test_full_system_startup_simulation(self):
        """Simulate full system startup with dependency resolution."""
        injector = DependencyInjector()
        injector.clear()
        
        # Define service hierarchy
        services = []
        
        class CoreService(BaseService):
            def __init__(self, name):
                super().__init__(name)
                services.append(f"{name}_started")
                
        class DatabaseService(CoreService):
            def __init__(self):
                super().__init__("DatabaseService")
                
        class CacheService(CoreService):
            def __init__(self):
                super().__init__("CacheService")
                self.database_service = None
                
            def configure_dependencies(self, dependency_injector):
                super().configure_dependencies(dependency_injector)
                self.database_service = self.resolve_dependency("DatabaseService")
        
        class BusinessService(CoreService):
            def __init__(self):
                super().__init__("BusinessService")
                self.database_service = None
                self.cache_service = None
                
            def configure_dependencies(self, dependency_injector):
                super().configure_dependencies(dependency_injector)
                self.database_service = self.resolve_dependency("DatabaseService")
                self.cache_service = self.resolve_dependency("CacheService")
        
        # Register services in dependency order
        from unittest.mock import MagicMock
        mock_connection_manager = MagicMock()
        db_service = DatabaseService()
        cache_service = CacheService()
        business_service = BusinessService()
        
        injector.register_service("DatabaseService", db_service, singleton=True)
        injector.register_service("CacheService", cache_service, singleton=True)
        injector.register_service("BusinessService", business_service, singleton=True)
        
        # Configure dependencies
        cache_service.configure_dependencies(injector)
        business_service.configure_dependencies(injector)
        
        # Verify dependencies are resolved
        assert cache_service.database_service is db_service
        assert business_service.database_service is db_service
        assert business_service.cache_service is cache_service
        
        # Verify startup order was maintained
        assert "DatabaseService_started" in services
        assert "CacheService_started" in services  
        assert "BusinessService_started" in services
        
        injector.clear()
        
    def test_memory_leak_prevention(self):
        """Test that DI container doesn't cause memory leaks."""
        import gc
        import weakref
        
        injector = DependencyInjector()
        injector.clear()
        
        # Create service and weak reference
        service = BaseService("TestService")
        weak_ref = weakref.ref(service)
        
        # Register service
        injector.register_service("TestService", service, singleton=True)
        
        # Delete local reference
        del service
        
        # Service should still be alive (held by container)
        assert weak_ref() is not None
        
        # Clear container
        injector.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Service should now be garbage collected
        assert weak_ref() is None
        
    @pytest.mark.asyncio
    async def test_error_recovery_patterns(self):
        """Test error recovery patterns in services."""
        injector = DependencyInjector()
        injector.clear()
        
        class RecoverableService(BaseService):
            def __init__(self):
                super().__init__("RecoverableService")
                self.failure_count = 0
                
            async def flaky_operation(self):
                return await self.execute_with_monitoring(
                    "flaky_operation",
                    self._flaky_impl
                )
                
            async def _flaky_impl(self):
                self.failure_count += 1
                if self.failure_count <= 2:
                    raise ServiceError("Temporary failure")
                return {"result": "success_after_retries"}
        
        service = RecoverableService()
        
        # Configure retry mechanism
        service.configure_retry(enabled=True, max_retries=3, delay=0.01, backoff=1.0)
        
        await service.start()
        
        # Should succeed after retries
        result = await service.flaky_operation()
        assert result["result"] == "success_after_retries"
        
        # Verify retry metrics
        metrics = service.get_metrics()
        assert metrics["operations_count"] == 1  # One logical operation
        assert metrics["operations_success"] == 1
        
        await service.stop()
        injector.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])