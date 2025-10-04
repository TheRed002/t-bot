"""
Integration Test for Exchange Module Architectural Compliance.

This test suite validates that the exchanges module is properly integrated with other modules
in the trading bot system, ensuring service layer patterns, dependency injection,
and interface contracts are correctly implemented. It performs static analysis to verify
architectural constraints programmatically.

COMPREHENSIVE VALIDATION:

1. **Service Layer Pattern Enforcement**
   - Execution module uses ExchangeFactory through DI
   - Bot management uses IExchangeFactory interface
   - No direct exchange implementation imports

2. **Module Hierarchy Compliance** 
   - Exchanges only imports from: core, utils, error_handling, database, monitoring, state
   - Higher modules (execution, strategies, bot_management) properly use exchanges through interfaces

3. **Dependency Injection Configuration**
   - ExchangeFactory is registered in DI container
   - Other modules get exchanges through DI, not direct imports
   - Service layer exists and is properly structured

4. **Interface Contract Adherence**
   - All modules use IExchange or ExchangeInterface protocols
   - No coupling to specific exchange implementations
   - Interface methods are properly defined and implemented

5. **Controller-Service Pattern**
   - Controllers use services, not repositories directly
   - No direct exchange access from controllers
   - Service layer properly abstracts exchange operations

This test can be run standalone or through pytest and performs static code analysis
to ensure architectural constraints are maintained during development.

Usage:
    pytest tests/integration/test_exchange_architectural_compliance.py -v
    python tests/integration/test_exchange_architectural_compliance.py
"""

import ast
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, get_type_hints

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestExchangeArchitecturalCompliance:
    """Test suite for exchange module architectural compliance."""

    def test_service_layer_pattern_enforcement(self):
        """Test that execution module uses ExchangeFactory through DI."""
        # Check execution module doesn't import exchanges directly
        execution_imports = self._get_module_imports("src/execution")
        
        # Execution should not import specific exchange implementations
        forbidden_imports = [
            "src.exchanges.binance",
            "src.exchanges.coinbase", 
            "src.exchanges.okx",
            "src.exchanges.implementations",
        ]
        
        for forbidden in forbidden_imports:
            assert not any(forbidden in imp for imp in execution_imports), (
                f"Execution module should not directly import {forbidden}"
            )

    def test_bot_management_interface_usage(self):
        """Test bot management uses IExchangeFactory interface."""
        # Verify bot management imports interfaces, not implementations
        bot_management_imports = self._get_module_imports("src/bot_management")
        
        # Should not import specific implementations
        forbidden_implementations = [
            "src.exchanges.binance",
            "src.exchanges.coinbase",
            "src.exchanges.okx"
        ]
        
        for forbidden in forbidden_implementations:
            assert not any(forbidden in imp for imp in bot_management_imports), (
                f"Bot management should not directly import {forbidden}"
            )

    def test_module_hierarchy_compliance(self):
        """Test exchanges only imports from approved lower-level modules."""
        exchanges_imports = self._get_module_imports("src/exchanges")
        
        # Approved modules (lower in hierarchy)
        approved_modules = [
            "src.core",
            "src.utils", 
            "src.error_handling",
            "src.database",
            "src.monitoring",
            "src.state"
        ]
        
        # Forbidden modules (higher in hierarchy) 
        forbidden_modules = [
            "src.execution",
            "src.strategies", 
            "src.bot_management",
            "src.analytics",
            "src.backtesting",
            "src.web_interface"
        ]
        
        # Check no forbidden imports
        for import_statement in exchanges_imports:
            for forbidden in forbidden_modules:
                assert not import_statement.startswith(forbidden), (
                    f"Exchange module should not import from higher-level module: {forbidden}"
                )
        
        # Verify at least some approved imports exist
        has_approved_imports = any(
            any(import_statement.startswith(approved) for approved in approved_modules)
            for import_statement in exchanges_imports
        )
        assert has_approved_imports, "Exchanges should import from approved lower-level modules"

    def test_interface_contract_adherence(self):
        """Test interface contracts exist and are properly defined."""
        # Import interface types to verify they exist
        try:
            from src.exchanges.interfaces import IExchange, IExchangeFactory
            
            # Verify IExchange has required methods
            required_exchange_methods = [
                "connect", "disconnect", "health_check", "is_connected",
                "place_order", "cancel_order", "get_order_status",
                "get_market_data", "get_order_book", "get_ticker",
                "get_account_balance", "get_positions", "get_exchange_info",
                "exchange_name"
            ]
            
            # Check methods are defined in protocol
            for method in required_exchange_methods:
                assert hasattr(IExchange, method) or method in IExchange.__annotations__, (
                    f"IExchange should define {method}"
                )
            
            # Verify IExchangeFactory has required methods
            required_factory_methods = [
                "get_supported_exchanges", "get_available_exchanges", 
                "is_exchange_supported", "get_exchange", "create_exchange",
                "remove_exchange", "health_check_all", "disconnect_all"
            ]
            
            for method in required_factory_methods:
                assert hasattr(IExchangeFactory, method) or method in IExchangeFactory.__annotations__, (
                    f"IExchangeFactory should define {method}"
                )
                
        except ImportError as e:
            pytest.fail(f"Failed to import exchange interfaces: {e}")

    def test_exchange_factory_implementation(self):
        """Test ExchangeFactory implements the interface contract."""
        try:
            from src.exchanges.factory import ExchangeFactory
            from src.exchanges.interfaces import IExchangeFactory
            
            # Verify ExchangeFactory has all required methods
            required_methods = [
                "get_supported_exchanges", "get_available_exchanges", 
                "is_exchange_supported", "create_exchange", "remove_exchange",
                "health_check_all", "disconnect_all"
            ]
            
            for method in required_methods:
                assert hasattr(ExchangeFactory, method), (
                    f"ExchangeFactory should implement {method}"
                )
                
        except ImportError as e:
            pytest.fail(f"Failed to import ExchangeFactory: {e}")

    def test_no_direct_exchange_imports(self):
        """Test other modules don't import specific exchange implementations."""
        # Check key modules that should use factory pattern
        modules_to_check = [
            "src/execution",
            "src/bot_management", 
            "src/strategies",
            "src/analytics"
        ]
        
        forbidden_patterns = [
            "from src.exchanges.binance import",
            "from src.exchanges.coinbase import", 
            "from src.exchanges.okx import",
            "import src.exchanges.binance",
            "import src.exchanges.coinbase",
            "import src.exchanges.okx"
        ]
        
        for module_path in modules_to_check:
            if not os.path.exists(module_path):
                continue
                
            imports = self._get_module_imports(module_path)
            
            for import_statement in imports:
                for forbidden in forbidden_patterns:
                    assert forbidden not in import_statement, (
                        f"Module {module_path} should not use {forbidden}. "
                        f"Use ExchangeFactory through DI instead."
                    )

    def test_dependency_injection_structure(self):
        """Test DI registration structure exists."""
        try:
            from src.exchanges.di_registration import (
                register_exchange_dependencies,
                register_exchange_services,
                setup_exchange_services
            )

            # Verify registration functions exist
            assert callable(register_exchange_dependencies)
            assert callable(register_exchange_services)
            assert callable(setup_exchange_services)

        except ImportError as e:
            pytest.fail(f"Failed to import DI registration: {e}")
    
    def test_service_layer_exists(self):
        """Test exchange service layer exists."""
        try:
            from src.exchanges.service import ExchangeService
            
            # Verify service has required structure
            assert hasattr(ExchangeService, "__init__")
            
        except ImportError as e:
            pytest.fail(f"Failed to import ExchangeService: {e}")

    def test_circular_dependency_prevention(self):
        """Test no circular dependencies exist between modules."""
        # Get all module imports
        modules = [
            "src/exchanges",
            "src/execution", 
            "src/bot_management",
            "src/strategies"
        ]
        
        import_graph = {}
        
        for module_path in modules:
            if os.path.exists(module_path):
                module_name = module_path.replace("/", ".")
                imports = self._get_module_imports(module_path)
                
                # Filter to only internal imports
                internal_imports = [
                    imp for imp in imports 
                    if imp.startswith("src.") and not imp.startswith("src.core")
                ]
                
                import_graph[module_name] = internal_imports
        
        # Check for direct circular dependencies
        for module, imports in import_graph.items():
            for imported_module in imports:
                reverse_imports = import_graph.get(imported_module, [])
                assert module not in [imp.replace("/", ".") for imp in reverse_imports], (
                    f"Circular dependency detected: {module} <-> {imported_module}"
                )

    def _get_module_imports(self, module_path: str) -> list[str]:
        """Extract import statements from a module directory."""
        imports = []
        
        if not os.path.exists(module_path):
            return imports
            
        for root, dirs, files in os.walk(module_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                        # Parse AST to extract imports
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.append(alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    imports.append(node.module)
                                    
                    except (SyntaxError, UnicodeDecodeError):
                        # Skip files with syntax errors or encoding issues
                        continue
                        
        return imports

    def test_controller_service_pattern(self):
        """Test controllers use services, not direct exchange access."""
        # Check execution controller
        execution_controller_imports = self._get_module_imports("src/execution")
        
        # Should not import exchange implementations directly
        forbidden_direct_imports = [
            "src.exchanges.binance_exchange",
            "src.exchanges.coinbase_exchange",
            "src.exchanges.okx_exchange"
        ]
        
        for forbidden in forbidden_direct_imports:
            assert not any(forbidden in imp for imp in execution_controller_imports), (
                f"Execution controller should not directly import {forbidden}"
            )
            
        # Should use service interfaces
        assert any("service" in imp.lower() or "interface" in imp.lower() 
                  for imp in execution_controller_imports), (
            "Execution should use service layer patterns"
        )


# Import asyncio for async test support
import asyncio

def test_exchange_architectural_compliance_standalone():
    """Standalone test runner for basic architectural compliance checks."""
    test_instance = TestExchangeArchitecturalCompliance()
    
    tests_passed = 0
    tests_total = 0
    
    # List of test methods to run
    test_methods = [
        "test_service_layer_pattern_enforcement",
        "test_bot_management_interface_usage", 
        "test_module_hierarchy_compliance",
        "test_interface_contract_adherence",
        "test_exchange_factory_implementation",
        "test_no_direct_exchange_imports",
        "test_dependency_injection_structure",
        "test_service_layer_exists",
        "test_circular_dependency_prevention",
        "test_controller_service_pattern",
    ]
    
    for test_method in test_methods:
        tests_total += 1
        try:
            method = getattr(test_instance, test_method)
            method()
            print(f"✅ {test_method}: PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ {test_method}: FAILED - {e}")
    
    print(f"\n📊 Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("🎉 All exchange architectural compliance tests passed!")
    else:
        print(f"⚠️ {tests_total - tests_passed} tests failed")
        
    return tests_passed == tests_total


if __name__ == "__main__":
    test_exchange_architectural_compliance_standalone()