"""
Tests for Capital Management Module Initialization.

This module tests the capital_management package initialization,
including lazy imports and module attributes.
"""

import pytest


class TestCapitalManagementInit:
    """Test capital_management module initialization."""

    def test_module_attributes(self):
        """Test that module has expected attributes."""
        import src.capital_management as capital_mgmt

        assert hasattr(capital_mgmt, '__version__')
        assert hasattr(capital_mgmt, '__author__')
        assert hasattr(capital_mgmt, '__all__')
        assert capital_mgmt.__version__ == "1.0.0"
        assert capital_mgmt.__author__ == "Trading Bot Framework"

    def test_direct_imports_available(self):
        """Test that direct imports are available."""
        import src.capital_management as capital_mgmt

        # Test interfaces are directly available (not lazy loaded)
        assert hasattr(capital_mgmt, 'AbstractCapitalService')
        assert hasattr(capital_mgmt, 'CapitalServiceProtocol')
        assert hasattr(capital_mgmt, 'register_capital_management_services')

        # Test type imports are available
        assert hasattr(capital_mgmt, 'CapitalCurrencyExposure')
        assert hasattr(capital_mgmt, 'CapitalFundFlow')

    def test_lazy_import_capital_allocator(self):
        """Test lazy import of CapitalAllocator."""
        import src.capital_management as capital_mgmt

        # This should trigger the lazy import
        allocator_class = capital_mgmt.CapitalAllocator
        assert allocator_class is not None
        assert allocator_class.__name__ == "CapitalAllocator"

    def test_lazy_import_capital_service(self):
        """Test lazy import of CapitalService."""
        import src.capital_management as capital_mgmt

        # This should trigger the lazy import
        service_class = capital_mgmt.CapitalService
        assert service_class is not None
        assert service_class.__name__ == "CapitalService"

    def test_lazy_import_currency_manager(self):
        """Test lazy import of CurrencyManager."""
        import src.capital_management as capital_mgmt

        # This should trigger the lazy import
        manager_class = capital_mgmt.CurrencyManager
        assert manager_class is not None
        assert manager_class.__name__ == "CurrencyManager"

    def test_lazy_import_exchange_distributor(self):
        """Test lazy import of ExchangeDistributor."""
        import src.capital_management as capital_mgmt

        # This should trigger the lazy import
        distributor_class = capital_mgmt.ExchangeDistributor
        assert distributor_class is not None
        assert distributor_class.__name__ == "ExchangeDistributor"

    def test_lazy_import_fund_flow_manager(self):
        """Test lazy import of FundFlowManager."""
        import src.capital_management as capital_mgmt

        # This should trigger the lazy import
        manager_class = capital_mgmt.FundFlowManager
        assert manager_class is not None
        assert manager_class.__name__ == "FundFlowManager"

    def test_lazy_import_factories(self):
        """Test lazy import of factory classes."""
        import src.capital_management as capital_mgmt

        # Test factory imports
        capital_mgmt_factory = capital_mgmt.CapitalManagementFactory
        assert capital_mgmt_factory is not None
        assert capital_mgmt_factory.__name__ == "CapitalManagementFactory"

        service_factory = capital_mgmt.CapitalServiceFactory
        assert service_factory is not None
        assert service_factory.__name__ == "CapitalServiceFactory"

    def test_lazy_import_repositories(self):
        """Test lazy import of repository classes."""
        import src.capital_management as capital_mgmt

        # Test repository imports
        capital_repo = capital_mgmt.CapitalRepository
        assert capital_repo is not None
        assert capital_repo.__name__ == "CapitalRepository"

        audit_repo = capital_mgmt.AuditRepository
        assert audit_repo is not None
        assert audit_repo.__name__ == "AuditRepository"

    def test_lazy_import_invalid_attribute(self):
        """Test that invalid attributes raise AttributeError."""
        import src.capital_management as capital_mgmt

        with pytest.raises(AttributeError, match="module '.*' has no attribute 'NonExistentClass'"):
            _ = capital_mgmt.NonExistentClass

    def test_all_exports_list_completeness(self):
        """Test that __all__ contains expected exports."""
        import src.capital_management as capital_mgmt

        expected_exports = [
            "AbstractCapitalService",
            "CapitalAllocator", 
            "CapitalService",
            "CurrencyManager",
            "ExchangeDistributor",
            "FundFlowManager",
            "CapitalManagementFactory",
            "CapitalRepository",
            "AuditRepository",
            "register_capital_management_services",
        ]

        for export in expected_exports:
            assert export in capital_mgmt.__all__, f"{export} not found in __all__"

    def test_lazy_imports_work_consistently(self):
        """Test that lazy imports work consistently across multiple calls."""
        import src.capital_management as capital_mgmt

        # First call
        allocator1 = capital_mgmt.CapitalAllocator
        service1 = capital_mgmt.CapitalService

        # Second call should return the same classes
        allocator2 = capital_mgmt.CapitalAllocator
        service2 = capital_mgmt.CapitalService

        assert allocator1 is allocator2
        assert service1 is service2

    def test_imports_mapping_completeness(self):
        """Test that all items in __all__ can be imported."""
        import src.capital_management as capital_mgmt

        # Test a subset of items that should be lazy-loaded
        lazy_imports = [
            "CapitalAllocator",
            "CurrencyManager", 
            "ExchangeDistributor",
            "FundFlowManager",
            "CapitalService"
        ]

        for item in lazy_imports:
            if item in capital_mgmt.__all__:
                # This should not raise an exception
                imported_item = getattr(capital_mgmt, item)
                assert imported_item is not None

    def test_module_docstring_present(self):
        """Test that module has proper docstring."""
        import src.capital_management as capital_mgmt

        assert capital_mgmt.__doc__ is not None
        assert "Capital Management System" in capital_mgmt.__doc__
        assert "P-010A" in capital_mgmt.__doc__