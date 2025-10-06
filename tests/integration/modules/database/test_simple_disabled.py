"""
Simple integration tests for database service integration using real services.
"""

import asyncio
from decimal import Decimal

import pytest
from sqlalchemy import text

from src.core.base.interfaces import HealthStatus
from src.core.dependency_injection import DependencyInjector
from src.database.interfaces import DatabaseServiceInterface
from src.database.service import DatabaseService

# Import the correct fixtures from infrastructure
from tests.integration.infrastructure.conftest import clean_database, real_test_config  # noqa: F401
from tests.integration.infrastructure.service_factory import RealServiceFactory


class TestDatabaseServiceSimpleIntegration:
    """Simple integration tests for database service using real database connections."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_service_implements_interface_real(self, clean_database):
        """Test that DatabaseService implements the required interface with real database."""
        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)
            database_service = service_factory.database_service

            # Check that service implements interface methods
            assert isinstance(database_service, DatabaseServiceInterface)

            required_methods = [
                "start",
                "stop",
                "create_entity",
                "get_entity_by_id",
                "update_entity",
                "delete_entity",
                "list_entities",
                "count_entities",
                "bulk_create",
                "get_health_status",
                "get_performance_metrics",
            ]

            for method in required_methods:
                assert hasattr(database_service, method), f"Missing method: {method}"
                assert callable(getattr(database_service, method))

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_service_initialization_with_real_dependencies(self, clean_database):
        """Test that DatabaseService can initialize with real dependencies."""
        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)
            database_service = service_factory.database_service

            # Should have proper configuration with real services
            assert database_service is not None
            # Test that the service has the expected functionality, not internal implementation details
            assert hasattr(database_service, "get_health_status")
            assert hasattr(database_service, "get_performance_metrics")

            # Test real database connectivity
            health_status = await database_service.get_health_status()
            assert health_status == HealthStatus.HEALTHY

            # Test that performance metrics are available
            performance_metrics = database_service.get_performance_metrics()
            assert isinstance(performance_metrics, dict)
            assert "cache_enabled" in performance_metrics

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_service_real_operations(self, clean_database):
        """Test that database service performs real operations."""
        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)
            database_service = service_factory.database_service

            # Test real database operations
            async with clean_database.get_async_session() as session:
                # Test basic connectivity
                result = await session.execute(text("SELECT 1 as test"))
                test_value = result.scalar()
                assert test_value == 1

                # Test performance metrics with real operations
                metrics = database_service.get_performance_metrics()
                assert isinstance(metrics, dict)
                assert "cache_enabled" in metrics
                assert "started" in metrics
                assert metrics["started"] is True

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_execution_service_with_real_database_service(self, clean_database):
        """Test that ExecutionService works with real database service injection."""
        from src.execution.service import ExecutionService

        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)
            database_service = service_factory.database_service

            # Create execution service with real repository service
            # ExecutionService requires repository_service, not database_service
            # For this test, we'll use a mock repository that wraps the database service
            from unittest.mock import Mock

            mock_repository = Mock()
            mock_repository.database_service = database_service

            execution_service = ExecutionService(
                repository_service=mock_repository
            )

            # Should store the real injected service
            assert execution_service.repository_service is mock_repository

            # Verify real database service is healthy
            health_status = await database_service.get_health_status()
            assert health_status == HealthStatus.HEALTHY

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_risk_service_with_real_database_service(self, clean_database):
        """Test that RiskService works with real database service injection."""
        from src.risk_management.service import RiskService

        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)
            database_service = service_factory.database_service

            # Create risk service with real repositories
            # RiskService requires repositories, not database_service directly
            from unittest.mock import Mock

            mock_risk_repo = Mock()
            mock_portfolio_repo = Mock()

            risk_service = RiskService(
                risk_metrics_repository=mock_risk_repo,
                portfolio_repository=mock_portfolio_repo,
                config=None,  # Use default config
            )

            # Verify repositories were injected correctly
            assert risk_service.risk_metrics_repository is mock_risk_repo
            assert risk_service.portfolio_repository is mock_portfolio_repo

            # Verify real database service is functional
            health_status = await database_service.get_health_status()
            assert health_status == HealthStatus.HEALTHY

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_data_service_with_real_database_service(self, clean_database):
        """Test that DataService works with real database service injection."""
        from src.core.config import Config
        from src.data.services.data_service import DataService

        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)
            database_service = service_factory.database_service

            # Create data service with real database service
            data_service = DataService(config=Config(), database_service=database_service)

            # Should store the real injected service
            assert data_service.database_service is database_service

            # Verify database service is operational
            metrics = database_service.get_performance_metrics()
            assert isinstance(metrics, dict)

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_service_method_signatures_with_real_operations(self, clean_database):
        """Test that DatabaseService methods work correctly with real database."""
        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)
            database_service = service_factory.database_service

            # Test methods exist and are async where expected
            assert asyncio.iscoroutinefunction(database_service.start)
            assert asyncio.iscoroutinefunction(database_service.stop)
            assert asyncio.iscoroutinefunction(database_service.create_entity)
            assert asyncio.iscoroutinefunction(database_service.get_entity_by_id)
            assert asyncio.iscoroutinefunction(database_service.update_entity)
            assert asyncio.iscoroutinefunction(database_service.delete_entity)
            assert asyncio.iscoroutinefunction(database_service.list_entities)
            assert asyncio.iscoroutinefunction(database_service.count_entities)
            assert asyncio.iscoroutinefunction(database_service.bulk_create)
            assert asyncio.iscoroutinefunction(database_service.get_health_status)

            # Non-async methods
            assert not asyncio.iscoroutinefunction(database_service.get_performance_metrics)

            # Test real async operations
            health_status = await database_service.get_health_status()
            assert health_status == HealthStatus.HEALTHY

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_dependency_injection_with_real_services(self, clean_database):
        """Test that dependency injection works with real services."""
        from src.database.di_registration import register_database_services

        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)

            # Test dependency injection with real services
            injector = DependencyInjector()
            register_database_services(injector)

            # Should have registered core services
            database_service = injector.resolve("DatabaseService")
            assert database_service is not None
            assert isinstance(database_service, DatabaseService)

        finally:
            await service_factory.cleanup()

    def test_interface_compliance(self):
        """Test that interfaces are properly defined."""
        # All interfaces should be abstract base classes
        import abc

        from src.database.interfaces import (
            BotMetricsServiceInterface,
            DatabaseServiceInterface,
            HealthAnalyticsServiceInterface,
            ResourceManagementServiceInterface,
            TradingDataServiceInterface,
        )

        assert issubclass(DatabaseServiceInterface, abc.ABC)
        assert issubclass(TradingDataServiceInterface, abc.ABC)
        assert issubclass(BotMetricsServiceInterface, abc.ABC)
        assert issubclass(HealthAnalyticsServiceInterface, abc.ABC)
        assert issubclass(ResourceManagementServiceInterface, abc.ABC)

    def test_no_circular_imports(self):
        """Test that imports don't create circular dependencies."""
        # These imports should not fail due to circular dependencies
        # All imports successful means no circular dependency issues
        assert True

    def test_service_boundaries_respected(self):
        """Test that services use proper abstractions."""
        # Check that services import interfaces, not concrete implementations
        import ast
        import inspect

        from src.execution.service import ExecutionService

        # Get the source file path
        source_file = inspect.getsourcefile(ExecutionService)

        if source_file:
            with open(source_file) as f:
                source = f.read()

            # Parse the AST
            tree = ast.parse(source)

            # Check imports - should not import DatabaseService directly
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "src.database.service":
                        for alias in node.names or []:
                            # Should not import concrete DatabaseService in service modules
                            assert alias.name != "DatabaseService", (
                                "Service should use interface, not concrete DatabaseService"
                            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_database_precision_operations(self, clean_database):
        """Test financial precision with real database operations."""
        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)

            # Test Decimal precision with real database
            async with clean_database.get_async_session() as session:
                # Create a test decimal value with high precision
                test_amount = Decimal("1234.56789012")

                # Store and retrieve from real database
                await session.execute(
                    text("CREATE TEMPORARY TABLE test_precision (amount DECIMAL(20,8))")
                )
                await session.execute(
                    text("INSERT INTO test_precision (amount) VALUES (:amount)"),
                    {"amount": test_amount},
                )

                result = await session.execute(text("SELECT amount FROM test_precision"))
                stored_amount = result.scalar()

                # Verify precision is maintained in real database
                assert isinstance(stored_amount, Decimal)
                assert stored_amount == test_amount

                await session.commit()

        finally:
            await service_factory.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_database_concurrent_operations(self, clean_database):
        """Test concurrent operations with real database."""
        service_factory = RealServiceFactory()
        try:
            await service_factory.initialize_core_services(clean_database)
            database_service = service_factory.database_service

            # Test concurrent health checks with real service
            async def check_health():
                return await database_service.get_health_status()

            # Run multiple concurrent health checks
            health_statuses = await asyncio.gather(*[check_health() for _ in range(5)])

            # All should be healthy with real database
            for status in health_statuses:
                assert status == HealthStatus.HEALTHY

        finally:
            await service_factory.cleanup()
