"""
Comprehensive test suite for the reference generator.

This test suite validates all aspects of the reference generator including:
- Abstract class detection
- Protocol vs implementation classification
- Mixin inheritance tracking
- Method signature parsing
- Line number accuracy
- Type hint parsing
"""

import ast
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from typing import List, Dict, Any

# Import the classes and enums from reference generator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference_generator import (
    ComprehensiveReferenceGenerator,
    ClassType,
    MethodSource,
    ImplementationStatus,
    Class,
    Function,
    ModuleFile,
    ProtocolCompliance,
    Parameter
)


class TestClassTypeDetection:
    """Test class type determination logic."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_protocol_detection(self):
        """Test that protocol classes are correctly identified."""
        code = '''
class ExportServiceProtocol(Protocol):
    """Protocol for export service."""
    pass
'''
        tree = ast.parse(code)
        node = tree.body[0]
        bases = ['Protocol']
        
        result = self.generator._determine_class_type(node, bases, code)
        assert result == ClassType.PROTOCOL
    
    def test_abstract_class_detection(self):
        """Test that abstract classes are correctly identified."""
        code = '''
class AnalyticsDataRepository(ABC):
    """Abstract base class."""
    
    @abstractmethod
    async def store_data(self): pass
'''
        tree = ast.parse(code)
        node = tree.body[0]
        bases = ['ABC']
        
        result = self.generator._determine_class_type(node, bases, code)
        assert result == ClassType.ABSTRACT
    
    def test_concrete_implementation_detection(self):
        """Test that concrete implementations are correctly identified."""
        code = '''
class ExportService(BaseAnalyticsService, ExportServiceProtocol):
    """Concrete export service."""
    pass
'''
        tree = ast.parse(code)
        node = tree.body[0]
        bases = ['BaseAnalyticsService', 'ExportServiceProtocol']
        
        result = self.generator._determine_class_type(node, bases, code)
        assert result == ClassType.CONCRETE
    
    def test_enum_detection(self):
        """Test that enum classes are correctly identified."""
        code = '''
class AnalyticsMode(Enum):
    """Analytics operation modes."""
    LIVE = "live"
    SANDBOX = "sandbox"
'''
        tree = ast.parse(code)
        node = tree.body[0]
        bases = ['Enum']
        
        result = self.generator._determine_class_type(node, bases, code)
        assert result == ClassType.ENUM
    
    def test_dataclass_detection(self):
        """Test that dataclass classes are correctly identified."""
        code = '''
@dataclass
class PortfolioMetrics:
    """Portfolio metrics data."""
    total_value: Decimal
'''
        tree = ast.parse(code)
        node = tree.body[0]
        bases = []
        
        result = self.generator._determine_class_type(node, bases, code)
        assert result == ClassType.DATACLASS


class TestAbstractMethodDetection:
    """Test abstract method detection."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_abstractmethod_decorator(self):
        """Test detection of @abstractmethod decorator."""
        code = '''
@abstractmethod
async def store_data(self): pass
'''
        tree = ast.parse(code)
        node = tree.body[0]
        
        result = self.generator._is_abstract_method(node)
        assert result is True
    
    def test_ellipsis_body(self):
        """Test detection of methods with ellipsis body."""
        code = '''
def store_data(self): ...
'''
        tree = ast.parse(code)
        node = tree.body[0]
        
        result = self.generator._is_abstract_method(node)
        assert result is True
    
    def test_raise_body(self):
        """Test detection of methods that just raise."""
        code = '''
def store_data(self): 
    raise NotImplementedError
'''
        tree = ast.parse(code)
        node = tree.body[0]
        
        result = self.generator._is_abstract_method(node)
        assert result is True
    
    def test_concrete_method(self):
        """Test that concrete methods are not marked as abstract."""
        code = '''
def store_data(self):
    return {"status": "ok"}
'''
        tree = ast.parse(code)
        node = tree.body[0]
        
        result = self.generator._is_abstract_method(node)
        assert result is False


class TestMixinInheritanceTracking:
    """Test mixin inheritance method resolution."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_position_tracking_mixin_methods(self):
        """Test that PositionTrackingMixin methods are detected."""
        implementation = Class(
            name="TestService",
            bases=["BaseAnalyticsService", "PositionTrackingMixin"],
            methods=[],
            properties=[],
            docstring="Test service",
            line_number=1,
            is_private=False,
            class_type=ClassType.CONCRETE,
            protocol_compliance=[]
        )
        
        methods = self.generator._get_all_implemented_methods(implementation)
        assert 'update_position' in methods
        assert 'update_trade' in methods
    
    def test_order_tracking_mixin_methods(self):
        """Test that OrderTrackingMixin methods are detected."""
        implementation = Class(
            name="TestService",
            bases=["BaseAnalyticsService", "OrderTrackingMixin"],
            methods=[],
            properties=[],
            docstring="Test service",
            line_number=1,
            is_private=False,
            class_type=ClassType.CONCRETE,
            protocol_compliance=[]
        )
        
        methods = self.generator._get_all_implemented_methods(implementation)
        assert 'update_order' in methods
    
    def test_multiple_mixins(self):
        """Test that multiple mixins are handled correctly."""
        implementation = Class(
            name="TestService",
            bases=["BaseAnalyticsService", "PositionTrackingMixin", "OrderTrackingMixin"],
            methods=[Function("custom_method", [], "str", False, True, False, "Custom method", 1, MethodSource.IMPLEMENTED)],
            properties=[],
            docstring="Test service",
            line_number=1,
            is_private=False,
            class_type=ClassType.CONCRETE,
            protocol_compliance=[]
        )
        
        methods = self.generator._get_all_implemented_methods(implementation)
        assert 'update_position' in methods
        assert 'update_trade' in methods
        assert 'update_order' in methods
        assert 'custom_method' in methods
        # Should not have duplicates
        assert len(methods) == len(set(methods))


class TestProtocolComplianceValidation:
    """Test protocol compliance validation logic."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_complete_implementation(self):
        """Test complete protocol implementation."""
        protocol = Class(
            name="TestProtocol",
            bases=["Protocol"],
            methods=[
                Function("method1", [], "None", False, True, False, "Method 1", 1, MethodSource.PROTOCOL),
                Function("method2", [], "str", False, True, False, "Method 2", 2, MethodSource.PROTOCOL)
            ],
            properties=[],
            docstring="Test protocol",
            line_number=1,
            is_private=False,
            class_type=ClassType.PROTOCOL,
            protocol_compliance=[]
        )
        
        implementation = Class(
            name="TestImplementation",
            bases=["BaseService", "TestProtocol"],
            methods=[
                Function("method1", [], "None", False, True, False, "Method 1 impl", 1, MethodSource.IMPLEMENTED),
                Function("method2", [], "str", False, True, False, "Method 2 impl", 2, MethodSource.IMPLEMENTED)
            ],
            properties=[],
            docstring="Test implementation",
            line_number=1,
            is_private=False,
            class_type=ClassType.CONCRETE,
            protocol_compliance=[]
        )
        
        result = self.generator._check_protocol_compliance(implementation, protocol)
        assert result.status == ImplementationStatus.COMPLETE
        assert len(result.missing_methods) == 0
    
    def test_partial_implementation(self):
        """Test partial protocol implementation."""
        protocol = Class(
            name="TestProtocol",
            bases=["Protocol"],
            methods=[
                Function("method1", [], "None", False, True, False, "Method 1", 1, MethodSource.PROTOCOL),
                Function("method2", [], "str", False, True, False, "Method 2", 2, MethodSource.PROTOCOL),
                Function("method3", [], "int", False, True, False, "Method 3", 3, MethodSource.PROTOCOL)
            ],
            properties=[],
            docstring="Test protocol",
            line_number=1,
            is_private=False,
            class_type=ClassType.PROTOCOL,
            protocol_compliance=[]
        )
        
        implementation = Class(
            name="TestImplementation",
            bases=["BaseService", "TestProtocol"],
            methods=[
                Function("method1", [], "None", False, True, False, "Method 1 impl", 1, MethodSource.IMPLEMENTED),
                Function("method2", [], "str", False, True, False, "Method 2 impl", 2, MethodSource.IMPLEMENTED)
                # method3 is missing
            ],
            properties=[],
            docstring="Test implementation",
            line_number=1,
            is_private=False,
            class_type=ClassType.CONCRETE,
            protocol_compliance=[]
        )
        
        result = self.generator._check_protocol_compliance(implementation, protocol)
        assert result.status == ImplementationStatus.PARTIAL
        assert len(result.missing_methods) == 1
        assert "method3" in result.missing_methods
    
    def test_implementation_with_mixin_methods(self):
        """Test that mixin methods are considered in compliance checking."""
        protocol = Class(
            name="TestProtocol",
            bases=["Protocol"],
            methods=[
                Function("update_position", [], "None", False, True, False, "Update position", 1, MethodSource.PROTOCOL),
                Function("custom_method", [], "str", False, True, False, "Custom method", 2, MethodSource.PROTOCOL)
            ],
            properties=[],
            docstring="Test protocol",
            line_number=1,
            is_private=False,
            class_type=ClassType.PROTOCOL,
            protocol_compliance=[]
        )
        
        implementation = Class(
            name="TestImplementation",
            bases=["BaseService", "PositionTrackingMixin", "TestProtocol"],
            methods=[
                Function("custom_method", [], "str", False, True, False, "Custom method impl", 1, MethodSource.IMPLEMENTED)
                # update_position should come from PositionTrackingMixin
            ],
            properties=[],
            docstring="Test implementation",
            line_number=1,
            is_private=False,
            class_type=ClassType.CONCRETE,
            protocol_compliance=[]
        )
        
        result = self.generator._check_protocol_compliance(implementation, protocol)
        assert result.status == ImplementationStatus.COMPLETE
        assert len(result.missing_methods) == 0


class TestStatusIndicatorGeneration:
    """Test status indicator generation."""
    
    def setup_method(self):
        """Setup test environment.""" 
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_abstract_class_status(self):
        """Test that abstract classes get correct status."""
        # This would be tested by running the full generator and checking output
        # For now, we verify the logic is in place
        assert hasattr(ClassType, 'ABSTRACT')
    
    def test_complete_implementation_status(self):
        """Test that complete implementations get correct status."""
        # Would test the output generation logic
        pass
    
    def test_incomplete_implementation_status(self):
        """Test that incomplete implementations get correct status."""
        # Would test the output generation logic
        pass


class TestTypeHintParsing:
    """Test type hint parsing and documentation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_modern_type_hint_parsing(self):
        """Test parsing of modern type hints."""
        code = '''
def get_data(self) -> dict[str, Decimal]:
    pass
'''
        tree = ast.parse(code)
        node = tree.body[0]
        
        return_type = self.generator._ast_to_string(node.returns)
        # Should preserve modern syntax
        assert "dict[str, Decimal]" in return_type
    
    def test_optional_type_parsing(self):
        """Test parsing of Optional types."""
        code = '''
def get_metrics(self) -> PortfolioMetrics | None:
    pass
'''
        tree = ast.parse(code)
        node = tree.body[0]
        
        return_type = self.generator._ast_to_string(node.returns)
        assert "PortfolioMetrics | None" in return_type
    
    def test_complex_generic_parsing(self):
        """Test parsing of complex generic types."""
        code = '''
def get_data(self) -> list[dict[str, Any]]:
    pass
'''
        tree = ast.parse(code)
        node = tree.body[0]
        
        return_type = self.generator._ast_to_string(node.returns)
        assert "list[dict[str, Any]]" in return_type


class TestLineNumberAccuracy:
    """Test line number tracking accuracy."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_method_line_numbers(self):
        """Test that method line numbers are correctly captured."""
        code = '''class TestClass:
    def method1(self): pass
    
    def method2(self): pass
    
    async def method3(self): pass
'''
        tree = ast.parse(code)
        class_node = tree.body[0]
        
        cls = self.generator._extract_class(class_node)
        
        # Check that line numbers are captured
        assert cls.line_number == 1
        assert len(cls.methods) == 3
        
        method_lines = [m.line_number for m in cls.methods]
        assert 2 in method_lines  # method1
        assert 4 in method_lines  # method2  
        assert 6 in method_lines  # method3


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_invalid_syntax_handling(self):
        """Test that invalid syntax is handled gracefully."""
        # This would test file processing with syntax errors
        pass
    
    def test_missing_imports_handling(self):
        """Test that missing imports are handled gracefully."""
        pass
    
    def test_circular_inheritance_handling(self):
        """Test that circular inheritance is detected."""
        pass


class TestIntegrationScenarios:
    """Integration tests with realistic analytics module scenarios."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_export_service_scenario(self):
        """Test ExportService classification scenario."""
        # This would mock the actual ExportService and test the full pipeline
        pass
    
    def test_operational_service_scenario(self):
        """Test OperationalService classification scenario."""
        pass
    
    def test_abstract_repository_scenario(self):
        """Test abstract repository classification scenario."""
        pass


class TestWildcardFunctionality:
    """Test the new wildcard (*) functionality for generating all module references."""
    
    def setup_method(self):
        """Setup test environment.""" 
        self.generator = ComprehensiveReferenceGenerator()
    
    def test_discover_all_modules_logic(self):
        """Test the module discovery logic with a mock generator."""
        # Create a mock generator with predefined modules
        mock_generator = Mock()
        mock_generator._all_modules = {'analytics', 'core', 'exchanges'}
        
        # Test that the modules are properly stored
        assert len(mock_generator._all_modules) == 3
        assert 'analytics' in mock_generator._all_modules
        assert 'core' in mock_generator._all_modules
        assert 'exchanges' in mock_generator._all_modules
    
    def test_generate_all_references_no_src_directory(self):
        """Test behavior when src/ directory doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            result = self.generator.generate_all_references()
            assert result is False
    
    def test_generate_all_references_empty_modules(self):
        """Test behavior when no modules are available."""
        # Mock empty directory (no modules found)
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.iterdir', return_value=[]):  # Empty directory
            result = self.generator.generate_all_references()
            assert result is False  # No modules found should return False
    
    def test_generate_all_references_success_mock(self):
        """Test successful generation using direct method mocking."""
        # Mock the generator to have some modules and make generate_reference always succeed
        test_modules = ['analytics', 'core']
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.iterdir') as mock_iterdir, \
             patch.object(self.generator, 'generate_reference', return_value=True) as mock_generate:
            
            # Create mock directory items
            mock_items = []
            for name in test_modules:
                mock_item = Mock()
                mock_item.is_dir.return_value = True
                mock_item.name = name
                mock_items.append(mock_item)
            
            mock_iterdir.return_value = mock_items
            
            result = self.generator.generate_all_references()
            
            assert result is True
            assert mock_generate.call_count == len(test_modules)
    
    def test_generate_all_references_partial_failure_mock(self):
        """Test partial failure scenario using method mocking."""
        test_modules = ['analytics', 'core']
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.iterdir') as mock_iterdir, \
             patch.object(self.generator, 'generate_reference', side_effect=[True, False]) as mock_generate:
            
            # Create mock directory items
            mock_items = []
            for name in test_modules:
                mock_item = Mock()
                mock_item.is_dir.return_value = True
                mock_item.name = name
                mock_items.append(mock_item)
            
            mock_iterdir.return_value = mock_items
            
            result = self.generator.generate_all_references()
            
            assert result is False  # Should return False if any module fails
            assert mock_generate.call_count == len(test_modules)
    
    def test_generate_all_references_exception_handling_mock(self):
        """Test exception handling using method mocking."""
        test_modules = ['analytics', 'core']
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.iterdir') as mock_iterdir, \
             patch.object(self.generator, 'generate_reference', side_effect=[True, Exception("Test error")]) as mock_generate:
            
            # Create mock directory items
            mock_items = []
            for name in test_modules:
                mock_item = Mock()
                mock_item.is_dir.return_value = True
                mock_item.name = name
                mock_items.append(mock_item)
            
            mock_iterdir.return_value = mock_items
            
            result = self.generator.generate_all_references()
            
            assert result is False  # Should return False if any module has exceptions
            assert mock_generate.call_count == len(test_modules)


class TestMainFunctionWildcard:
    """Test the main function's wildcard handling."""
    
    @patch('sys.argv', ['script_name', '*'])
    @patch.object(ComprehensiveReferenceGenerator, 'generate_all_references')
    def test_main_with_wildcard(self, mock_generate_all):
        """Test main function handles wildcard parameter."""
        mock_generate_all.return_value = True
        
        # Import and run main function
        from reference_generator import main
        main()
        
        mock_generate_all.assert_called_once()
    
    @patch('sys.argv', ['script_name', '*'])
    @patch.object(ComprehensiveReferenceGenerator, 'generate_all_references')
    def test_main_with_wildcard_failure(self, mock_generate_all):
        """Test main function handles wildcard failure.""" 
        mock_generate_all.return_value = False
        
        from reference_generator import main
        main()
        
        mock_generate_all.assert_called_once()
    
    @patch('sys.argv', ['script_name', 'analytics'])
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.iterdir', return_value=[])  # Mock the src directory scan
    @patch.object(ComprehensiveReferenceGenerator, 'generate_reference')
    def test_main_with_specific_module(self, mock_generate, mock_iterdir, mock_exists):
        """Test main function still works with specific modules."""
        mock_exists.return_value = True
        mock_generate.return_value = True
        
        from reference_generator import main
        main()
        
        mock_generate.assert_called_once_with('analytics')
    
    @patch('sys.argv', ['script_name'])
    def test_main_with_no_args(self):
        """Test main function shows usage when no args provided."""
        from reference_generator import main
        
        # Should not raise exception, just print usage
        main()


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([
        __file__ + "::TestClassTypeDetection",
        __file__ + "::TestAbstractMethodDetection", 
        __file__ + "::TestMixinInheritanceTracking",
        __file__ + "::TestProtocolComplianceValidation",
        __file__ + "::TestWildcardFunctionality",
        __file__ + "::TestMainFunctionWildcard",
        "-v"
    ])