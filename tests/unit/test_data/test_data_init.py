"""Test suite for data module initialization."""

from unittest.mock import MagicMock, patch


class TestDataModuleImports:
    """Test suite for data module imports."""

    def test_imports_available(self):
        """Test that main data module imports are available."""
        # Mock the dependencies to avoid import errors
        with patch.dict(
            "sys.modules",
            {
                "src.data.di_registration": MagicMock(),
                "src.data.factory": MagicMock(),
                "src.data.interfaces": MagicMock(),
                "src.data.quality.cleaning": MagicMock(),
                "src.data.quality.monitoring": MagicMock(),
                "src.data.quality.validation": MagicMock(),
                "src.data.services": MagicMock(),
            },
        ):
            import src.data

            # Test that the module loads without error
            assert src.data is not None

    def test_all_exports_defined(self):
        """Test that __all__ is properly defined."""
        with patch.dict(
            "sys.modules",
            {
                "src.data.di_registration": MagicMock(),
                "src.data.factory": MagicMock(),
                "src.data.interfaces": MagicMock(),
                "src.data.quality.cleaning": MagicMock(),
                "src.data.quality.monitoring": MagicMock(),
                "src.data.quality.validation": MagicMock(),
                "src.data.services": MagicMock(),
            },
        ):
            import src.data

            # Test that __all__ is defined and contains expected items
            assert hasattr(src.data, "__all__")
            assert isinstance(src.data.__all__, list)
            assert len(src.data.__all__) > 0

            # Check for key components
            expected_exports = [
                "DataService",
                "DataServiceFactory",
                "DataServiceInterface",
                "DataValidator",
                "configure_data_dependencies",
            ]

            for export in expected_exports:
                assert export in src.data.__all__, f"{export} not found in __all__"

    def test_factory_import(self):
        """Test DataServiceFactory import."""
        mock_factory = MagicMock()
        mock_factory.DataServiceFactory = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "src.data.factory": mock_factory,
                "src.data.di_registration": MagicMock(),
                "src.data.interfaces": MagicMock(),
                "src.data.quality.cleaning": MagicMock(),
                "src.data.quality.monitoring": MagicMock(),
                "src.data.quality.validation": MagicMock(),
                "src.data.services": MagicMock(),
            },
        ):
            import src.data

            # Verify factory is accessible
            assert hasattr(src.data, "DataServiceFactory")

    def test_interfaces_import(self):
        """Test data interfaces import."""
        mock_interfaces = MagicMock()
        mock_interfaces.DataServiceInterface = MagicMock()
        mock_interfaces.DataStorageInterface = MagicMock()
        mock_interfaces.DataCacheInterface = MagicMock()
        mock_interfaces.DataValidatorInterface = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "src.data.interfaces": mock_interfaces,
                "src.data.di_registration": MagicMock(),
                "src.data.factory": MagicMock(),
                "src.data.quality.cleaning": MagicMock(),
                "src.data.quality.monitoring": MagicMock(),
                "src.data.quality.validation": MagicMock(),
                "src.data.services": MagicMock(),
            },
        ):
            import src.data

            # Verify interfaces are accessible
            expected_interfaces = [
                "DataServiceInterface",
                "DataStorageInterface",
                "DataCacheInterface",
                "DataValidatorInterface",
            ]

            for interface in expected_interfaces:
                assert hasattr(src.data, interface)

    def test_quality_components_import(self):
        """Test quality components import."""
        mock_cleaning = MagicMock()
        mock_cleaning.DataCleaner = MagicMock()
        mock_monitoring = MagicMock()
        mock_monitoring.QualityMonitor = MagicMock()
        mock_validation = MagicMock()
        mock_validation.DataValidator = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "src.data.quality.cleaning": mock_cleaning,
                "src.data.quality.monitoring": mock_monitoring,
                "src.data.quality.validation": mock_validation,
                "src.data.di_registration": MagicMock(),
                "src.data.factory": MagicMock(),
                "src.data.interfaces": MagicMock(),
                "src.data.services": MagicMock(),
            },
        ):
            import src.data

            # Verify quality components are accessible
            quality_components = ["DataCleaner", "QualityMonitor", "DataValidator"]

            for component in quality_components:
                assert hasattr(src.data, component)

    def test_services_import(self):
        """Test services import."""
        mock_services = MagicMock()
        mock_services.DataService = MagicMock()
        mock_services.MLDataService = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "src.data.services": mock_services,
                "src.data.di_registration": MagicMock(),
                "src.data.factory": MagicMock(),
                "src.data.interfaces": MagicMock(),
                "src.data.quality.cleaning": MagicMock(),
                "src.data.quality.monitoring": MagicMock(),
                "src.data.quality.validation": MagicMock(),
            },
        ):
            import src.data

            # Verify services are accessible
            expected_services = ["DataService"]

            for service in expected_services:
                assert hasattr(src.data, service)

    def test_dependency_injection_import(self):
        """Test dependency injection functions import."""
        mock_di = MagicMock()
        mock_di.configure_data_dependencies = MagicMock()
        mock_di.register_data_services = MagicMock()
        mock_di.get_data_service = MagicMock()
        mock_di.get_data_storage = MagicMock()
        mock_di.get_data_cache = MagicMock()
        mock_di.get_data_validator = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "src.data.di_registration": mock_di,
                "src.data.factory": MagicMock(),
                "src.data.interfaces": MagicMock(),
                "src.data.quality.cleaning": MagicMock(),
                "src.data.quality.monitoring": MagicMock(),
                "src.data.quality.validation": MagicMock(),
                "src.data.services": MagicMock(),
            },
        ):
            import src.data

            # Verify DI functions are accessible
            di_functions = [
                "configure_data_dependencies",
                "register_data_services",
                "get_data_service",
                "get_data_storage",
                "get_data_cache",
                "get_data_validator",
            ]

            for func in di_functions:
                assert hasattr(src.data, func)

    def test_module_docstring(self):
        """Test module has proper docstring."""
        with patch.dict(
            "sys.modules",
            {
                "src.data.di_registration": MagicMock(),
                "src.data.factory": MagicMock(),
                "src.data.interfaces": MagicMock(),
                "src.data.quality.cleaning": MagicMock(),
                "src.data.quality.monitoring": MagicMock(),
                "src.data.quality.validation": MagicMock(),
                "src.data.services": MagicMock(),
            },
        ):
            import src.data

            # Verify module has docstring
            assert src.data.__doc__ is not None
            assert len(src.data.__doc__.strip()) > 0
            assert "Data Management System" in src.data.__doc__
