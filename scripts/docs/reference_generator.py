#!/usr/bin/env python3
"""
Comprehensive T-Bot Module Reference Generator

Extracts ALL classes, methods, functions, and signatures from every Python file
in a module to create complete LLM-friendly reference documents.

Usage:
    python scripts/docs/reference_generator.py analytics
    python scripts/docs/reference_generator.py core
    python scripts/docs/reference_generator.py "*"  # Generate for all modules
"""

import ast
import sys
import importlib.util
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum


class MethodSource(Enum):
    """Source of method implementation"""
    PROTOCOL = "protocol"          # Method defined in Protocol
    ABSTRACT = "abstract"          # Abstract method in ABC
    IMPLEMENTED = "implemented"    # Actually implemented method
    INHERITED = "inherited"        # Inherited from parent class
    MISSING = "missing"            # Required by protocol but not implemented


class ClassType(Enum):
    """Type of class"""
    PROTOCOL = "protocol"          # Protocol/Interface definition
    ABSTRACT = "abstract"          # Abstract base class
    CONCRETE = "concrete"          # Concrete implementation
    DATACLASS = "dataclass"        # Data class
    ENUM = "enum"                  # Enum class


class ImplementationStatus(Enum):
    """Implementation status for protocol compliance"""
    COMPLETE = "complete"          # All protocol methods implemented
    PARTIAL = "partial"            # Some protocol methods missing
    MISSING = "missing"            # No protocol methods implemented
    NOT_APPLICABLE = "n/a"         # Not implementing any protocols


@dataclass
class Parameter:
    name: str
    annotation: Optional[str]
    default: Optional[str]


@dataclass
class Function:
    name: str
    parameters: List[Parameter]
    return_type: Optional[str]
    is_async: bool
    is_method: bool
    is_private: bool
    docstring: Optional[str]
    line_number: int
    source: MethodSource
    protocol_origin: Optional[str] = None  # Which protocol requires this method


@dataclass
class ProtocolCompliance:
    """Protocol compliance information"""
    protocol_name: str
    required_methods: List[str]
    implemented_methods: List[str]
    missing_methods: List[str]
    status: ImplementationStatus


@dataclass
class Class:
    name: str
    bases: List[str]
    methods: List[Function]
    properties: List[str]
    docstring: Optional[str]
    line_number: int
    is_private: bool
    class_type: ClassType
    protocol_compliance: List[ProtocolCompliance] = None


@dataclass
class ModuleFile:
    filename: str
    classes: List[Class]
    functions: List[Function]
    imports: List[str]
    protocols: List[Class] = None        # Separate protocols from implementations
    implementations: List[Class] = None  # Concrete implementations


@dataclass
class BusinessContext:
    role: str
    layer: str
    responsibilities: List[str]
    critical_notes: Dict[str, str]
    usage_patterns: List[str]


@dataclass
class IntegrationInfo:
    provides: List[str]
    used_by: List[str]
    patterns: List[str]


@dataclass
class ModuleReference:
    name: str
    dependencies: List[str]
    files: List[ModuleFile]
    business_context: Optional[BusinessContext]
    integration_info: IntegrationInfo
    detected_patterns: Dict[str, List[str]]


class ComprehensiveReferenceGenerator:
    """Extract complete module signatures with business context for LLM consumption"""
    
    def __init__(self):
        self.src_path = Path("src")
        self._all_modules = self._discover_all_modules()
        self._module_imports_cache = {}
        self._protocol_registry = {}  # protocol_name -> Protocol class definition
        self._implementation_map = {}  # class_name -> list of protocols it claims to implement
        
    def generate_reference(self, module_name: str) -> bool:
        """Generate comprehensive REFERENCE.md for module"""
        src_path = Path("src")
        module_path = src_path / module_name
        
        if not module_path.exists():
            print(f"❌ Module not found: {module_path}")
            return False
            
        print(f"🔍 Generating comprehensive reference for: {module_name}")
        
        # Extract dependencies and integration info
        dependencies = self._extract_dependencies(module_path)
        print(f"   📦 Dependencies: {', '.join(dependencies) if dependencies else 'None'}")
        
        # Discover business context and integration patterns
        business_context = self._load_business_context(module_path)
        integration_info = self._discover_integration_info(module_name)
        detected_patterns = self._detect_code_patterns(module_path)
        
        print(f"   🔗 Used by: {', '.join(integration_info.used_by) if integration_info.used_by else 'None'}")
        print(f"   🎯 Patterns: {', '.join(detected_patterns.keys()) if detected_patterns else 'None'}")
        
        # Process all Python files
        files = []
        py_files = list(module_path.glob("**/*.py"))
        py_files = [f for f in py_files if f.name != "__init__.py"]
        
        for py_file in sorted(py_files):
            relative_path = py_file.relative_to(module_path)
            print(f"   🔍 Processing: {relative_path}")
            
            module_file = self._process_file(py_file)
            if module_file:
                files.append(module_file)
                print(f"      ✓ {len(module_file.classes)} classes, {len(module_file.functions)} functions")
        
        # Perform protocol compliance validation
        print(f"   🔍 Validating protocol compliance...")
        self._validate_protocol_compliance(files)
        
        # Create module reference
        module_ref = ModuleReference(
            name=module_name,
            dependencies=dependencies,
            files=files,
            business_context=business_context,
            integration_info=integration_info,
            detected_patterns=detected_patterns
        )
        
        # Generate and save reference document
        reference_content = self._generate_reference_doc(module_ref)
        
        output_path = module_path / "REFERENCE.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(reference_content)
            
        print(f"✅ Comprehensive reference saved: {output_path}")
        print(f"   📊 Total: {sum(len(f.classes) for f in files)} classes, {sum(len(f.functions) for f in files)} functions")
        
        return True
    
    def generate_all_references(self) -> bool:
        """Generate REFERENCE.md for all modules"""
        src_path = Path("src")
        if not src_path.exists():
            print("❌ src/ directory not found")
            return False
        
        # Discover all modules
        modules = []
        for item in src_path.iterdir():
            if item.is_dir() and not item.name.startswith('_') and not item.name.startswith('.'):
                modules.append(item.name)
        
        if not modules:
            print("❌ No modules found in src/")
            return False
        
        modules.sort()
        total_modules = len(modules)
        successful = 0
        failed = 0
        
        print(f"🚀 Generating REFERENCE.md for {total_modules} modules...")
        print("=" * 60)
        
        for i, module_name in enumerate(modules, 1):
            print(f"\n[{i}/{total_modules}] Processing module: {module_name}")
            try:
                if self.generate_reference(module_name):
                    successful += 1
                    print(f"✅ {module_name} completed successfully")
                else:
                    failed += 1
                    print(f"❌ {module_name} failed")
            except Exception as e:
                failed += 1
                print(f"❌ {module_name} failed with error: {e}")
        
        print("\n" + "=" * 60)
        print(f"🎯 Summary: {successful} successful, {failed} failed out of {total_modules} modules")
        
        if failed > 0:
            print(f"⚠️  {failed} modules had issues - check output above for details")
        
        return failed == 0
    
    def _extract_dependencies(self, module_path: Path) -> List[str]:
        """Extract module dependencies from imports"""
        dependencies = set()
        module_name = module_path.name
        
        for py_file in module_path.glob("**/*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith("src."):
                            parts = node.module.split(".")
                            if len(parts) >= 2 and parts[1] != module_name:
                                dependencies.add(parts[1])
                                
            except Exception:
                continue
                
        return sorted(dependencies)
    
    def _discover_all_modules(self) -> Set[str]:
        """Discover all available modules in src/"""
        modules = set()
        if self.src_path.exists():
            for item in self.src_path.iterdir():
                if item.is_dir() and not item.name.startswith('_') and not item.name.startswith('.'):
                    modules.add(item.name)
        return modules
    
    def _load_business_context(self, module_path: Path) -> Optional[BusinessContext]:
        """Load business context from MODULE_METADATA.py if it exists"""
        metadata_file = module_path / "MODULE_METADATA.py"
        if not metadata_file.exists():
            return None
        
        try:
            spec = importlib.util.spec_from_file_location("module_metadata", metadata_file)
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            context_data = getattr(module, 'BUSINESS_CONTEXT', None)
            if context_data:
                return BusinessContext(
                    role=context_data.get('role', ''),
                    layer=context_data.get('layer', ''),
                    responsibilities=context_data.get('responsibilities', []),
                    critical_notes=context_data.get('critical_notes', {}),
                    usage_patterns=context_data.get('usage_patterns', [])
                )
        except Exception as e:
            print(f"      ⚠️ Error loading metadata: {e}")
        
        return None
    
    def _discover_integration_info(self, target_module: str) -> IntegrationInfo:
        """Discover which modules use this module and what it provides"""
        used_by = []
        provides = []
        patterns = []
        
        # Scan all modules to find imports of target_module
        for module_name in self._all_modules:
            if module_name == target_module:
                continue
                
            module_path = self.src_path / module_name
            if not module_path.exists():
                continue
                
            # Check if this module imports target_module
            if self._module_imports_target(module_path, target_module):
                used_by.append(module_name)
        
        # Extract provided services by analyzing class names and exports
        target_path = self.src_path / target_module
        provides = self._extract_provided_services(target_path)
        
        # Detect integration patterns
        patterns = self._detect_integration_patterns(target_path)
        
        return IntegrationInfo(
            provides=provides,
            used_by=sorted(used_by),
            patterns=patterns
        )
    
    def _module_imports_target(self, module_path: Path, target_module: str) -> bool:
        """Check if module imports target_module"""
        if target_module in self._module_imports_cache:
            return module_path.name in self._module_imports_cache[target_module]
        
        importers = set()
        for py_file in module_path.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith(f"src.{target_module}"):
                            importers.add(module_path.name)
                            break
            except Exception:
                continue
        
        self._module_imports_cache[target_module] = importers
        return module_path.name in importers
    
    def _extract_provided_services(self, module_path: Path) -> List[str]:
        """Extract key services/interfaces this module provides"""
        services = set()
        
        for py_file in module_path.glob("**/*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Look for service-like classes
                        if (node.name.endswith('Service') or 
                            node.name.endswith('Manager') or 
                            node.name.endswith('Engine') or
                            node.name.endswith('Controller')):
                            services.add(node.name)
            except Exception:
                continue
                
        return sorted(services)
    
    def _detect_integration_patterns(self, module_path: Path) -> List[str]:
        """Detect common integration patterns in the module"""
        patterns = set()
        
        for py_file in module_path.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple pattern detection based on common decorators/patterns
                if '@inject' in content:
                    patterns.add("Dependency Injection")
                if 'BaseService' in content:
                    patterns.add("Service Layer")
                if 'BaseComponent' in content:
                    patterns.add("Component Architecture")
                if '@with_circuit_breaker' in content:
                    patterns.add("Circuit Breaker")
                if 'async def' in content:
                    patterns.add("Async Operations")
            except Exception:
                continue
                
        return sorted(patterns)
    
    def _detect_code_patterns(self, module_path: Path) -> Dict[str, List[str]]:
        """Detect financial, security, and performance patterns in code"""
        patterns = {
            "financial": [],
            "security": [],
            "performance": [],
            "architecture": []
        }
        
        for py_file in module_path.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Financial patterns
                if 'Decimal' in content:
                    patterns["financial"].append("Decimal precision arithmetic")
                if 'DECIMAL(' in content.upper():
                    patterns["financial"].append("Database decimal columns")
                if any(word in content.lower() for word in ['price', 'amount', 'balance', 'quantity']):
                    patterns["financial"].append("Financial data handling")
                
                # Security patterns
                if any(decorator in content for decorator in ['@validate', '@sanitize']):
                    patterns["security"].append("Input validation")
                if 'API_KEY' in content or 'SECRET' in content:
                    patterns["security"].append("Credential management")
                if '@require_auth' in content or 'authenticate' in content:
                    patterns["security"].append("Authentication")
                
                # Performance patterns
                if '@cache' in content or '@lru_cache' in content:
                    patterns["performance"].append("Caching")
                if 'asyncio.gather' in content:
                    patterns["performance"].append("Parallel execution")
                if '@retry' in content:
                    patterns["performance"].append("Retry mechanisms")
                    
                # Architecture patterns
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if any(base.id in ['BaseService', 'BaseComponent'] for base in node.bases if isinstance(base, ast.Name)):
                            patterns["architecture"].append(f"{node.name} inherits from base architecture")
            except Exception:
                continue
        
        # Remove empty lists
        return {k: v for k, v in patterns.items() if v}
    
    def _process_file(self, file_path: Path) -> Optional[ModuleFile]:
        """Process a single Python file with protocol analysis"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            classes = []
            functions = []
            imports = []
            
            # Process top-level nodes
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    cls = self._extract_class(node, content)
                    classes.append(cls)
                    
                    # Register protocols for later validation
                    if cls.class_type == ClassType.PROTOCOL:
                        self._protocol_registry[cls.name] = cls
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(self._extract_function(node, is_method=False))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.extend(self._extract_imports(node))
            
            # Separate protocols from implementations
            protocols = [cls for cls in classes if cls.class_type == ClassType.PROTOCOL]
            implementations = [cls for cls in classes if cls.class_type != ClassType.PROTOCOL]
            
            return ModuleFile(
                filename=file_path.name,
                classes=classes,
                functions=functions,
                imports=imports,
                protocols=protocols,
                implementations=implementations
            )
            
        except Exception as e:
            print(f"      ⚠️ Error processing {file_path.name}: {e}")
            return None
    
    def _extract_class(self, node: ast.ClassDef, file_content: str = "") -> Class:
        """Extract complete class information with protocol detection"""
        bases = [self._ast_to_string(base) for base in node.bases]
        methods = []
        properties = []
        
        # Determine class type
        class_type = self._determine_class_type(node, bases, file_content)
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Determine method source based on class type
                method_source = MethodSource.PROTOCOL if class_type == ClassType.PROTOCOL else MethodSource.IMPLEMENTED
                if self._is_abstract_method(item):
                    method_source = MethodSource.ABSTRACT
                
                method = self._extract_function(item, is_method=True, source=method_source)
                methods.append(method)
            elif isinstance(item, ast.FunctionDef) and any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in getattr(item, 'decorator_list', [])
            ):
                properties.append(item.name)
        
        return Class(
            name=node.name,
            bases=bases,
            methods=methods,
            properties=properties,
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
            is_private=node.name.startswith('_'),
            class_type=class_type,
            protocol_compliance=[]
        )
    
    def _extract_function(self, node, is_method: bool = False, source: MethodSource = MethodSource.IMPLEMENTED) -> Function:
        """Extract complete function information"""
        parameters = []
        
        # Extract all parameters
        for arg in node.args.args:
            param = Parameter(
                name=arg.arg,
                annotation=self._ast_to_string(arg.annotation) if arg.annotation else None,
                default=None
            )
            parameters.append(param)
        
        # Extract defaults
        defaults = node.args.defaults
        if defaults:
            # Defaults apply to the last N parameters
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                param_index = len(parameters) - num_defaults + i
                if param_index >= 0 and param_index < len(parameters):
                    parameters[param_index].default = self._ast_to_string(default)
        
        # Handle *args and **kwargs
        if node.args.vararg:
            parameters.append(Parameter(
                name=f"*{node.args.vararg.arg}",
                annotation=self._ast_to_string(node.args.vararg.annotation) if node.args.vararg.annotation else None,
                default=None
            ))
        
        if node.args.kwarg:
            parameters.append(Parameter(
                name=f"**{node.args.kwarg.arg}",
                annotation=self._ast_to_string(node.args.kwarg.annotation) if node.args.kwarg.annotation else None,
                default=None
            ))
        
        return Function(
            name=node.name,
            parameters=parameters,
            return_type=self._ast_to_string(node.returns) if node.returns else None,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            is_private=node.name.startswith('_'),
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
            source=source
        )
    
    def _determine_class_type(self, node: ast.ClassDef, bases: List[str], file_content: str) -> ClassType:
        """Determine the type of class (Protocol, Abstract, Concrete, etc.)"""
        class_name = node.name
        
        # Check if THIS class IS a Protocol (by name)
        if class_name.endswith('Protocol'):
            return ClassType.PROTOCOL
        
        # Check for @dataclass decorator first (higher priority)
        if hasattr(node, 'decorator_list'):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
                    return ClassType.DATACLASS
        
        # Check if it's an Enum
        if any('Enum' in base for base in bases):
            return ClassType.ENUM
        
        # Check if it's an abstract class (inherits from ABC or has abstract methods)
        is_abstract = (any('ABC' in base or 'Abstract' in base for base in bases) or
                      self._has_abstract_methods(node))
        if is_abstract:
            return ClassType.ABSTRACT
        
        # If it inherits from a Protocol, it's a concrete implementation
        if any(base.endswith('Protocol') for base in bases):
            return ClassType.CONCRETE
            
        # Default to concrete implementation
        return ClassType.CONCRETE
    
    def _is_abstract_method(self, node: ast.FunctionDef) -> bool:
        """Check if a method is abstract"""
        if hasattr(node, 'decorator_list'):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                    return True
        
        # Also check method body for common abstract patterns
        if len(node.body) == 1:
            if isinstance(node.body[0], ast.Expr):
                if isinstance(node.body[0].value, ast.Constant) and node.body[0].value.value == Ellipsis:
                    return True
            elif isinstance(node.body[0], ast.Raise):
                return True
                
        return False
    
    def _has_abstract_methods(self, node: ast.ClassDef) -> bool:
        """Check if class has any abstract methods"""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_abstract_method(item):
                    return True
        return False
    
    def _validate_protocol_compliance(self, files: List[ModuleFile]) -> None:
        """Validate that implementation classes properly implement their declared protocols"""
        # Build comprehensive protocol registry from all files
        all_protocols = {}
        for file_info in files:
            if file_info.protocols:
                for protocol in file_info.protocols:
                    all_protocols[protocol.name] = protocol
        
        compliance_issues = 0
        
        for file_info in files:
            if not file_info.implementations:
                continue
                
            for impl_class in file_info.implementations:
                protocol_compliance = []
                
                # Find protocols this class claims to implement
                protocol_bases = [base for base in impl_class.bases if base.endswith('Protocol')]
                
                for protocol_name in protocol_bases:
                    if protocol_name in all_protocols:
                        protocol = all_protocols[protocol_name]
                        compliance = self._check_protocol_compliance(impl_class, protocol)
                        protocol_compliance.append(compliance)
                        
                        if compliance.missing_methods:
                            compliance_issues += len(compliance.missing_methods)
                            print(f"      ⚠️ {impl_class.name} missing {len(compliance.missing_methods)} methods from {protocol_name}: {', '.join(compliance.missing_methods[:3])}{'...' if len(compliance.missing_methods) > 3 else ''}")
                    else:
                        # Protocol not found in current module - could be external
                        print(f"      ℹ️ {impl_class.name} implements external protocol: {protocol_name}")
                
                impl_class.protocol_compliance = protocol_compliance
        
        if compliance_issues > 0:
            print(f"   ⚠️ Found {compliance_issues} missing protocol methods across implementations")
    
    def _check_protocol_compliance(self, implementation: Class, protocol: Class) -> ProtocolCompliance:
        """Check if an implementation satisfies a protocol's requirements"""
        required_methods = [m.name for m in protocol.methods if not m.name.startswith('_')]
        
        # Get all implemented methods including those from mixins
        implemented_methods = self._get_all_implemented_methods(implementation)
        
        missing_methods = [m for m in required_methods if m not in implemented_methods]
        
        # Determine status
        if not missing_methods:
            status = ImplementationStatus.COMPLETE
        elif len(missing_methods) == len(required_methods):
            status = ImplementationStatus.MISSING
        else:
            status = ImplementationStatus.PARTIAL
        
        return ProtocolCompliance(
            protocol_name=protocol.name,
            required_methods=required_methods,
            implemented_methods=[m for m in implemented_methods if m in required_methods],
            missing_methods=missing_methods,
            status=status
        )
    
    def _get_all_implemented_methods(self, implementation: Class) -> list[str]:
        """Get all methods available to an implementation, including from mixins and generic patterns"""
        implemented_methods = [m.name for m in implementation.methods if not m.name.startswith('_')]
        
        # Check for common mixin patterns that provide specific methods
        for base in implementation.bases:
            if 'PositionTrackingMixin' in base:
                implemented_methods.extend(['update_position', 'update_trade'])
            elif 'OrderTrackingMixin' in base:
                implemented_methods.extend(['update_order'])
            elif 'ErrorHandlingMixin' in base:
                implemented_methods.extend(['handle_operation_error'])
        
        # Note: Removed generic method pattern recognition to ensure strict protocol compliance
        # All protocol methods must be explicitly implemented
        
        if 'get_operational_metrics' in implemented_methods:
            # If service implements get_operational_metrics, it's likely complete for operational purposes
            # The record_* methods might be optional for this specific implementation
            pass
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(implemented_methods))
    
    def _extract_imports(self, node) -> List[str]:
        """Extract import statements"""
        imports = []
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"from {module} import {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        return imports
    
    def _ast_to_string(self, node) -> str:
        """Convert AST node to string"""
        if node is None:
            return "None"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            # Handle subscripts with tuple slices (e.g., dict[str, Any])
            slice_str = self._ast_to_string(node.slice)
            # Remove parentheses from tuple slices in subscripts
            if isinstance(node.slice, ast.Tuple) and slice_str.startswith('(') and slice_str.endswith(')'):
                slice_str = slice_str[1:-1]
            return f"{self._ast_to_string(node.value)}[{slice_str}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.List):
            elements = [self._ast_to_string(elt) for elt in node.elts]
            return f"[{', '.join(elements)}]"
        elif isinstance(node, ast.Tuple):
            elements = [self._ast_to_string(elt) for elt in node.elts]
            return f"({', '.join(elements)})"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Handle union types (e.g., str | None)
            left = self._ast_to_string(node.left)
            right = self._ast_to_string(node.right)
            return f"{left} | {right}"
        else:
            return "Any"
    
    def _generate_reference_doc(self, module_ref: ModuleReference) -> str:
        """Generate comprehensive reference document with business context"""
        lines = []
        
        # Header
        lines.extend([
            f"# {module_ref.name.upper()} Module Reference",
            "",
        ])
        
        # Business Context Section
        if module_ref.business_context:
            ctx = module_ref.business_context
            lines.extend([
                "## BUSINESS CONTEXT",
                f"**Role**: {ctx.role}",
                f"**Layer**: {ctx.layer}",
                f"**Responsibilities**: {', '.join(ctx.responsibilities)}",
                ""
            ])
            
            if ctx.critical_notes:
                lines.append("**Critical Notes:**")
                for category, note in ctx.critical_notes.items():
                    lines.append(f"- **{category.title()}**: {note}")
                lines.append("")
                
            if ctx.usage_patterns:
                lines.append("**Usage Patterns:**")
                for pattern in ctx.usage_patterns:
                    lines.append(f"- {pattern}")
                lines.append("")
        
        # Integration Section
        lines.extend([
            "## INTEGRATION",
            f"**Dependencies**: {', '.join(module_ref.dependencies) if module_ref.dependencies else 'None'}",
            f"**Used By**: {', '.join(module_ref.integration_info.used_by) if module_ref.integration_info.used_by else 'None'}",
            f"**Provides**: {', '.join(module_ref.integration_info.provides) if module_ref.integration_info.provides else 'Core functionality'}",
        ])
        
        if module_ref.integration_info.patterns:
            lines.extend([
                f"**Patterns**: {', '.join(module_ref.integration_info.patterns)}",
            ])
        
        lines.append("")
        
        # Detected Patterns Section
        if module_ref.detected_patterns:
            lines.extend([
                "## DETECTED PATTERNS",
            ])
            
            for category, pattern_list in module_ref.detected_patterns.items():
                if pattern_list:
                    lines.extend([
                        f"**{category.title()}**:",
                    ])
                    for pattern in pattern_list[:3]:  # Limit to first 3
                        lines.append(f"- {pattern}")
            lines.append("")
        
        # Module Overview
        lines.extend([
            "## MODULE OVERVIEW", 
            f"**Files**: {len(module_ref.files)} Python files",
            f"**Classes**: {sum(len(f.classes) for f in module_ref.files)}",
            f"**Functions**: {sum(len(f.functions) for f in module_ref.files)}",
            "",
            "## COMPLETE API REFERENCE",
            ""
        ])
        
        # Separate Protocols and Implementations sections
        all_protocols = []
        all_implementations = []
        
        for file_info in module_ref.files:
            if file_info.protocols:
                all_protocols.extend(file_info.protocols)
            if file_info.implementations:
                all_implementations.extend(file_info.implementations)
        
        # PROTOCOLS & INTERFACES section
        if all_protocols:
            lines.extend([
                "## PROTOCOLS & INTERFACES",
                ""
            ])
            
            for protocol in all_protocols:
                lines.extend([
                    f"### Protocol: `{protocol.name}`",
                    ""
                ])
                
                if protocol.docstring:
                    lines.append(f"**Purpose**: {protocol.docstring.split('.')[0][:100]}")
                    lines.append("")
                
                if protocol.methods:
                    lines.append("**Required Methods:**")
                    for method in protocol.methods:
                        param_strs = []
                        for param in method.parameters:
                            param_str = param.name
                            if param.annotation:
                                param_str += f": {param.annotation}"
                            if param.default:
                                param_str += f" = {param.default}"
                            param_strs.append(param_str)
                        
                        params = ", ".join(param_strs) if len(", ".join(param_strs)) <= 80 else f"{method.parameters[0].name if method.parameters else ''}, ..."
                        async_prefix = "async " if method.is_async else ""
                        return_type = f" -> {method.return_type}" if method.return_type else ""
                        
                        lines.append(f"- `{async_prefix}{method.name}({params}){return_type}`")
                    lines.append("")
        
        # IMPLEMENTATIONS section  
        if all_implementations:
            lines.extend([
                "## IMPLEMENTATIONS",
                ""
            ])
            
            for impl in all_implementations:
                # Implementation status indicator
                status_indicator = "✅"
                status_text = "Complete"
                total_missing = 0
                
                # Check if this is an abstract class first
                if impl.class_type == ClassType.ABSTRACT:
                    status_indicator = "🔧"
                    status_text = "Abstract Base Class"
                elif impl.protocol_compliance:
                    incomplete_protocols = [p for p in impl.protocol_compliance if p.status != ImplementationStatus.COMPLETE]
                    if incomplete_protocols:
                        total_missing = sum(len(p.missing_methods) for p in incomplete_protocols)
                        if total_missing > 0:
                            status_indicator = "⚠️"
                            status_text = f"Incomplete ({total_missing} missing methods)"
                
                lines.extend([
                    f"### Implementation: `{impl.name}` {status_indicator}",
                    ""
                ])
                
                if impl.bases:
                    lines.append(f"**Inherits**: {', '.join(impl.bases)}")
                
                if impl.docstring:
                    lines.append(f"**Purpose**: {impl.docstring.split('.')[0][:100]}")
                
                lines.append(f"**Status**: {status_text}")
                lines.append("")
                
                # Protocol compliance details
                if impl.protocol_compliance:
                    for compliance in impl.protocol_compliance:
                        if compliance.status != ImplementationStatus.COMPLETE:
                            lines.append(f"**{compliance.protocol_name} Compliance**: {compliance.status.value.upper()}")
                            if compliance.missing_methods:
                                lines.append("**Missing Methods:**")
                                for missing in compliance.missing_methods:
                                    lines.append(f"- `{missing}()` - Required by {compliance.protocol_name}")
                            lines.append("")
                
                # Show actually implemented methods
                if impl.methods:
                    lines.append("**Implemented Methods:**")
                    for method in impl.methods:
                        if not method.is_private:  # Only show public methods
                            param_strs = []
                            for param in method.parameters:
                                param_str = param.name
                                if param.annotation:
                                    param_str += f": {param.annotation}"
                                if param.default:
                                    param_str += f" = {param.default}"
                                param_strs.append(param_str)
                            
                            params = ", ".join(param_strs) if len(", ".join(param_strs)) <= 80 else f"{method.parameters[0].name if method.parameters else ''}, ..."
                            async_prefix = "async " if method.is_async else ""
                            return_type = f" -> {method.return_type}" if method.return_type else ""
                            
                            lines.append(f"- `{async_prefix}{method.name}({params}){return_type}` - Line {method.line_number}")
                    lines.append("")
        
        # COMPLETE API REFERENCE (remaining classes and functions)
        lines.extend([
            "## COMPLETE API REFERENCE",
            ""
        ])
        
        for file_info in module_ref.files:
            remaining_classes = [cls for cls in file_info.classes 
                               if cls.class_type not in [ClassType.PROTOCOL, ClassType.CONCRETE] or
                               (cls.class_type == ClassType.CONCRETE and not cls.protocol_compliance)]
            
            if not remaining_classes and not file_info.functions:
                continue
                
            lines.extend([
                f"### File: {file_info.filename}",
                ""
            ])
            
            # Show imports if significant
            if file_info.imports:
                key_imports = [imp for imp in file_info.imports if 'src.' in imp][:5]
                if key_imports:
                    lines.append("**Key Imports:**")
                    for imp in key_imports:
                        lines.append(f"- `{imp}`")
                    lines.append("")
            
            # Remaining classes
            for cls in remaining_classes:
                lines.extend([
                    f"#### Class: `{cls.name}`",
                    ""
                ])
                
                if cls.bases:
                    lines.append(f"**Inherits**: {', '.join(cls.bases)}")
                
                if cls.docstring:
                    lines.append(f"**Purpose**: {cls.docstring.split('.')[0][:100]}")
                
                lines.extend([
                    "",
                    "```python",
                    f"class {cls.name}{'(' + ', '.join(cls.bases) + ')' if cls.bases else ''}:"
                ])
                
                # Show ALL methods (public and private)
                for method in cls.methods:
                    param_strs = []
                    for param in method.parameters:
                        param_str = param.name
                        if param.annotation:
                            param_str += f": {param.annotation}"
                        if param.default:
                            param_str += f" = {param.default}"
                        param_strs.append(param_str)
                    
                    params = ", ".join(param_strs) if len(", ".join(param_strs)) <= 80 else f"{method.parameters[0].name}, ..."
                    async_prefix = "async " if method.is_async else ""
                    return_type = f" -> {method.return_type}" if method.return_type else ""
                    
                    lines.append(f"    {async_prefix}def {method.name}({params}){return_type}  # Line {method.line_number}")
                
                lines.extend([
                    "```",
                    ""
                ])
            
            # Module-level functions
            if file_info.functions:
                lines.extend([
                    "#### Functions:",
                    "",
                    "```python"
                ])
                
                for func in file_info.functions:
                    param_strs = []
                    for param in func.parameters:
                        param_str = param.name
                        if param.annotation:
                            param_str += f": {param.annotation}"
                        if param.default:
                            param_str += f" = {param.default}"
                        param_strs.append(param_str)
                    
                    params = ", ".join(param_strs) if len(", ".join(param_strs)) <= 80 else "..."
                    async_prefix = "async " if func.is_async else ""
                    return_type = f" -> {func.return_type}" if func.return_type else ""
                    
                    lines.append(f"{async_prefix}def {func.name}({params}){return_type}  # Line {func.line_number}")
                
                lines.extend([
                    "```",
                    ""
                ])
        
        # Footer
        lines.extend([
            "---",
            f"**Generated**: Complete reference for {module_ref.name} module",
            f"**Total Classes**: {sum(len(f.classes) for f in module_ref.files)}",
            f"**Total Functions**: {sum(len(f.functions) for f in module_ref.files)}"
        ])
        
        return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/docs/reference_generator.py <module_name>")
        print("       python scripts/docs/reference_generator.py \"*\"  # Generate for all modules")
        print("Example: python scripts/docs/reference_generator.py analytics")
        return
    
    module_name = sys.argv[1]
    generator = ComprehensiveReferenceGenerator()
    
    # Handle wildcard for all modules
    if module_name == "*":
        print("🌟 Generating REFERENCE.md for ALL modules...")
        success = generator.generate_all_references()
        if success:
            print("🎯 Successfully generated references for all modules")
        else:
            print("❌ Some modules failed to generate references")
        return
    
    # Validate specific module exists
    src_path = Path("src")
    if not src_path.exists():
        print("❌ src/ directory not found")
        return
    
    module_path = src_path / module_name
    if not module_path.exists():
        print(f"❌ Module not found: {module_path}")
        available = [d.name for d in src_path.iterdir() if d.is_dir() and not d.name.startswith('_')]
        print("Available modules:", ", ".join(sorted(available)))
        print("Use \"*\" to generate for all modules")
        return
    
    # Generate reference for specific module
    success = generator.generate_reference(module_name)
    
    if success:
        print(f"🎯 Successfully generated comprehensive reference for {module_name}")
    else:
        print(f"❌ Failed to generate reference for {module_name}")


if __name__ == "__main__":
    main()