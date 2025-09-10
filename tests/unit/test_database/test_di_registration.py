"""
Unit tests for database dependency injection registration.

Tests the registration of database services with the dependency injector
and ensures proper configuration of service lifetimes and fallback handling.
"""

from unittest.mock import Mock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.core.dependency_injection import DependencyInjector
from src.database.di_registration import (
    configure_database_dependencies,
    get_database_manager,
    get_database_service,
    get_uow_factory,
    register_database_services,
)


class TestDatabaseDIRegistration:
    """Test database dependency injection registration."""

    @pytest.fixture
    def mock_injector(self):
        """Create mock dependency injector."""
        injector = Mock(spec=DependencyInjector)
        injector.register_factory = Mock()
        injector.register_service = Mock()
        injector.resolve = Mock()
        return injector

    @pytest.fixture
    def real_injector(self):
        """Create real dependency injector for integration tests."""
        return DependencyInjector()

    def test_register_database_services_basic_registration(self, mock_injector):
        """Test basic database service registration."""
        # Mock required services
        mock_config_service = Mock()
        mock_config = Mock()
        mock_config_service.get_config.return_value = mock_config
        mock_validation_service = Mock()

        mock_injector.resolve.side_effect = lambda service: {
            "ConfigService": mock_config_service,
            "ValidationService": mock_validation_service,
        }.get(service, Mock())

        # Register services
        register_database_services(mock_injector)

        # Verify registration calls were made
        assert mock_injector.register_factory.call_count >= 5  # Multiple factories registered
        assert mock_injector.register_service.call_count >= 5  # Multiple services registered

    def test_register_database_services_connection_manager(self, mock_injector):
        """Test DatabaseConnectionManager registration."""
        mock_config_service = Mock()
        mock_config = Mock()
        mock_config_service.get_config.return_value = mock_config
        mock_injector.resolve.return_value = mock_config_service

        with patch("src.database.di_registration.DatabaseConnectionManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            register_database_services(mock_injector)

            # Check that DatabaseConnectionManager factory was registered
            factory_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if "DatabaseConnectionManager" in str(call)
            ]
            assert len(factory_calls) >= 1

    def test_register_database_services_repository_factory(self, mock_injector):
        """Test RepositoryFactory registration."""
        mock_injector.resolve.return_value = Mock()

        with patch("src.database.di_registration.RepositoryFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory_class.return_value = mock_factory

            register_database_services(mock_injector)

            # Check that RepositoryFactory was registered
            factory_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if "RepositoryFactory" in str(call)
            ]
            assert len(factory_calls) >= 1

    def test_register_database_services_session_factories(self, mock_injector):
        """Test session factory registration."""
        mock_connection_manager = Mock()
        mock_connection_manager.async_session_maker = Mock()
        mock_connection_manager.sync_session_maker = Mock()
        mock_injector.resolve.return_value = mock_connection_manager

        register_database_services(mock_injector)

        # Check session factory registrations
        session_calls = [
            call
            for call in mock_injector.register_factory.call_args_list
            if "Session" in str(call)
        ]
        assert len(session_calls) >= 2  # AsyncSession and Session

    def test_register_database_services_database_service(self, mock_injector):
        """Test DatabaseService registration."""
        mock_config_service = Mock()
        mock_validation_service = Mock()

        mock_injector.resolve.side_effect = lambda service: {
            "ConfigService": mock_config_service,
            "ValidationService": mock_validation_service,
        }.get(service, Mock())

        with patch("src.database.di_registration.DatabaseService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            register_database_services(mock_injector)

            # Check DatabaseService factory registration
            service_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if "DatabaseService" in str(call)
            ]
            assert len(service_calls) >= 1

    def test_register_database_services_database_service_fallback(self, mock_injector):
        """Test DatabaseService fallback when dependencies fail."""
        # Make resolve fail for dependencies
        mock_injector.resolve.side_effect = Exception("Dependency resolution failed")

        with patch("src.database.di_registration.DatabaseService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            register_database_services(mock_injector)

            # Should still register (with fallback)
            service_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if "DatabaseService" in str(call)
            ]
            assert len(service_calls) >= 1

    def test_register_database_services_interface_mappings(self, mock_injector):
        """Test interface to implementation mappings."""
        mock_injector.resolve.return_value = Mock()

        register_database_services(mock_injector)

        # Check that interface services were registered
        expected_interfaces = [
            "DatabaseServiceInterface",
            "TradingDataServiceInterface",
            "BotMetricsServiceInterface",
            "HealthAnalyticsServiceInterface",
            "ResourceManagementServiceInterface",
        ]

        for interface in expected_interfaces:
            interface_calls = [
                call
                for call in mock_injector.register_service.call_args_list
                if interface in str(call)
            ]
            assert len(interface_calls) >= 1, f"Interface {interface} not registered"

    def test_register_database_services_database_manager(self, mock_injector):
        """Test DatabaseManager registration."""
        mock_database_service = Mock()
        mock_injector.resolve.side_effect = lambda service: {
            "DatabaseService": mock_database_service
        }.get(service, Mock())

        with patch("src.database.di_registration.DatabaseManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            register_database_services(mock_injector)

            # Check DatabaseManager registration
            manager_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if "DatabaseManager" in str(call)
            ]
            assert len(manager_calls) >= 1

    def test_register_database_services_database_manager_fallback(self, mock_injector):
        """Test DatabaseManager fallback when DatabaseService resolution fails."""
        mock_injector.resolve.side_effect = Exception("Service resolution failed")

        with patch("src.database.di_registration.DatabaseManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            register_database_services(mock_injector)

            # Should still register with fallback
            manager_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if "DatabaseManager" in str(call)
            ]
            assert len(manager_calls) >= 1

    def test_register_database_services_uow_factory(self, mock_injector):
        """Test UnitOfWorkFactory registration."""
        mock_connection_manager = Mock()
        mock_connection_manager.sync_engine = Mock()
        mock_connection_manager.async_engine = Mock()

        mock_injector.resolve.side_effect = lambda service: {
            "DatabaseConnectionManager": mock_connection_manager
        }.get(service, Mock())

        with (
            patch("src.database.di_registration.UnitOfWorkFactory") as mock_uow_factory_class,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
            patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_async_sessionmaker,
        ):
            mock_uow_factory = Mock()
            mock_uow_factory_class.return_value = mock_uow_factory

            register_database_services(mock_injector)

            # Check UnitOfWorkFactory registration
            uow_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if "UnitOfWorkFactory" in str(call)
            ]
            assert len(uow_calls) >= 1

    def test_register_database_services_uow_factory_fallback(self, mock_injector):
        """Test UnitOfWorkFactory fallback creation."""
        mock_injector.resolve.side_effect = Exception("Connection manager resolution failed")

        with (
            patch("src.database.di_registration.UnitOfWorkFactory") as mock_uow_factory_class,
            patch("sqlalchemy.create_engine") as mock_create_engine,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
        ):
            mock_uow_factory = Mock()
            mock_uow_factory_class.return_value = mock_uow_factory

            register_database_services(mock_injector)

            # Should still register with fallback
            uow_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if "UnitOfWorkFactory" in str(call)
            ]
            assert len(uow_calls) >= 1

    def test_register_database_services_uow_instances(self, mock_injector):
        """Test UnitOfWork instance registration."""
        mock_uow_factory = Mock()
        mock_uow = Mock()
        mock_async_uow = Mock()
        mock_uow_factory.create.return_value = mock_uow
        mock_uow_factory.create_async.return_value = mock_async_uow

        mock_injector.resolve.side_effect = lambda service: {
            "UnitOfWorkFactory": mock_uow_factory
        }.get(service, Mock())

        register_database_services(mock_injector)

        # Check UoW instance registrations
        uow_instance_calls = [
            call
            for call in mock_injector.register_factory.call_args_list
            if "UnitOfWork" in str(call) and "Factory" not in str(call)
        ]
        assert len(uow_instance_calls) >= 2  # UnitOfWork and AsyncUnitOfWork

    def test_register_database_services_repository_registration(self, mock_injector):
        """Test repository registration."""
        mock_repository_factory = Mock()
        mock_session = Mock()

        mock_injector.resolve.side_effect = lambda service: {
            "RepositoryFactory": mock_repository_factory,
            "AsyncSession": mock_session,
        }.get(service, Mock())

        register_database_services(mock_injector)

        # Check that trading repositories were registered
        expected_repos = [
            "OrderRepository",
            "PositionRepository",
            "TradeRepository",
            "OrderFillRepository",
        ]

        for repo_name in expected_repos:
            repo_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if repo_name in str(call)
            ]
            assert len(repo_calls) >= 1, f"Repository {repo_name} not registered"

    def test_configure_database_dependencies_with_injector(self, mock_injector):
        """Test configure_database_dependencies with existing injector."""
        result = configure_database_dependencies(mock_injector)

        assert result is mock_injector
        assert mock_injector.register_factory.call_count > 0

    def test_configure_database_dependencies_without_injector(self):
        """Test configure_database_dependencies uses global instance."""
        with patch("src.database.di_registration.DependencyInjector") as mock_injector_class:
            mock_injector = Mock()
            mock_injector_class.get_instance.return_value = mock_injector

            result = configure_database_dependencies()

            assert result is mock_injector
            mock_injector_class.get_instance.assert_called_once()

    def test_get_database_service(self, mock_injector):
        """Test get_database_service convenience function."""
        mock_service = Mock()
        mock_injector.resolve.return_value = mock_service

        result = get_database_service(mock_injector)

        assert result is mock_service
        mock_injector.resolve.assert_called_once_with("DatabaseService")

    def test_get_database_manager(self, mock_injector):
        """Test get_database_manager convenience function."""
        mock_manager = Mock()
        mock_injector.resolve.return_value = mock_manager

        result = get_database_manager(mock_injector)

        assert result is mock_manager
        mock_injector.resolve.assert_called_once_with("DatabaseManager")

    def test_get_uow_factory(self, mock_injector):
        """Test get_uow_factory convenience function."""
        mock_factory = Mock()
        mock_injector.resolve.return_value = mock_factory

        result = get_uow_factory(mock_injector)

        assert result is mock_factory
        mock_injector.resolve.assert_called_once_with("UnitOfWorkFactory")


class TestDatabaseDIRegistrationIntegration:
    """Integration tests with real dependency injector."""

    def test_register_database_services_integration(self):
        """Test registration with real dependency injector (integration)."""
        injector = DependencyInjector()

        # Mock required external dependencies
        with (
            patch("src.database.di_registration.DatabaseConnectionManager"),
            patch("src.database.di_registration.DatabaseService"),
            patch("src.database.di_registration.DatabaseManager"),
            patch("src.database.di_registration.RepositoryFactory"),
            patch("src.database.di_registration.UnitOfWorkFactory"),
        ):
            # Should not raise exceptions
            register_database_services(injector)

            # Verify some registrations exist
            assert injector.has_service("DatabaseConnectionManager")
            assert injector.has_service("RepositoryFactory")
            assert injector.has_service("DatabaseService")
            assert injector.has_service("DatabaseManager")

    def test_configure_database_dependencies_integration(self):
        """Test complete dependency configuration (integration)."""
        with (
            patch("src.database.di_registration.DatabaseConnectionManager"),
            patch("src.database.di_registration.DatabaseService"),
            patch("src.database.di_registration.DatabaseManager"),
            patch("src.database.di_registration.RepositoryFactory"),
            patch("src.database.di_registration.UnitOfWorkFactory"),
        ):
            injector = configure_database_dependencies()

            # Should be properly configured injector
            assert isinstance(injector, DependencyInjector)
            assert injector.has_service("DatabaseService")
            assert injector.has_service("DatabaseManager")
            assert injector.has_service("RepositoryFactory")


class TestDatabaseDIRegistrationErrorHandling:
    """Test error handling in dependency registration."""

    def test_register_database_services_partial_failure(self):
        """Test registration continues despite partial failures."""
        injector = Mock(spec=DependencyInjector)
        injector.register_factory = Mock()
        injector.register_service = Mock()

        # Make resolve fail for some services but not others
        resolve_count = 0

        def mock_resolve(service):
            nonlocal resolve_count
            resolve_count += 1
            if resolve_count <= 2:
                raise Exception(f"Resolution failed for {service}")
            return Mock()

        injector.resolve = Mock(side_effect=mock_resolve)

        # Should not raise exception despite some failures
        register_database_services(injector)

        # Should still register some services
        assert injector.register_factory.call_count > 0

    def test_repository_factory_creation_fallback(self):
        """Test repository factory fallback when DI fails."""
        injector = Mock(spec=DependencyInjector)
        injector.resolve.side_effect = Exception("Repository factory resolution failed")

        mock_session = Mock()

        # Import the repository factory function from the module
        from src.database.di_registration import register_database_services

        with patch("src.database.di_registration.RepositoryFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory_class.return_value = mock_factory

            # Mock repository class for testing
            mock_repo_class = Mock()
            mock_repo_instance = Mock()
            mock_repo_class.return_value = mock_repo_instance

            register_database_services(injector)

            # Get the registered repository factory function
            factory_calls = injector.register_factory.call_args_list
            repo_factory_call = next(
                (call for call in factory_calls if "OrderRepository" in str(call)), None
            )

            assert repo_factory_call is not None

    def test_uow_factory_complete_failure_fallback(self):
        """Test UoW factory fallback when all DI fails."""
        injector = Mock(spec=DependencyInjector)
        injector.resolve.side_effect = Exception("Complete DI failure")
        injector.register_factory = Mock()
        injector.register_service = Mock()

        with (
            patch("sqlalchemy.create_engine") as mock_create_engine,
            patch("sqlalchemy.orm.sessionmaker") as mock_sessionmaker,
            patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_create_async_engine,
            patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_async_sessionmaker,
        ):
            # Should use fallback engines
            register_database_services(injector)

            # Verify fallback was used
            assert injector.register_factory.call_count > 0


class TestDatabaseDIRegistrationServiceLifetimes:
    """Test service lifetime configuration."""

    def test_singleton_services_registered_as_singletons(self, monkeypatch):
        """Test that singleton services are registered with singleton=True."""
        from unittest.mock import Mock

        mock_injector = Mock()
        register_database_services(mock_injector)

        # Check singleton registrations
        singleton_services = [
            "DatabaseConnectionManager",
            "RepositoryFactory",
            "DatabaseService",
            "DatabaseManager",
            "UnitOfWorkFactory",
        ]

        for service in singleton_services:
            singleton_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if service in str(call) and "singleton=True" in str(call)
            ]
            assert len(singleton_calls) >= 1, f"Service {service} not registered as singleton"

    def test_transient_services_registered_as_transient(self, monkeypatch):
        """Test that transient services are registered with singleton=False."""
        from unittest.mock import Mock

        mock_injector = Mock()
        register_database_services(mock_injector)

        # Check transient registrations
        transient_services = ["AsyncSession", "Session", "UnitOfWork", "AsyncUnitOfWork"]

        for service in transient_services:
            transient_calls = [
                call
                for call in mock_injector.register_factory.call_args_list
                if service in str(call) and "singleton=False" in str(call)
            ]
            # Should have at least one transient registration
            # Note: repositories are also transient but checked separately
            if service in ["AsyncSession", "Session", "UnitOfWork", "AsyncUnitOfWork"]:
                assert len(transient_calls) >= 1, f"Service {service} not registered as transient"
