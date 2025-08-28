#!/usr/bin/env python3
"""
AUTO_FINAL.py - Production-Ready Module Fixer for Financial Application

Key Features:
- ENFORCES MINIMUM 70% TEST COVERAGE (90% for critical financial modules)
- Generates comprehensive tests including edge cases and performance benchmarks
- Smart parallel execution preventing correlated modules from running simultaneously  
- Hallucination prevention with explicit constraints in prompts
- CASCADE PROPAGATION: Breaking changes are cascaded to all dependent modules
- NO ENHANCED VERSIONS: Direct modification only, no _v2 or _enhanced suffixes
- Automatic retry on API 500 errors (3 attempts with 5s delay)
- Financial calculation verification and precision testing
- Performance benchmarks for time-critical operations
- Integration test generation for complete workflows

CRITICAL: This is a FINANCIAL APPLICATION - accuracy and test coverage are paramount!
"""

import subprocess
import sys
import os
import time
import json
import argparse
import hashlib
import signal
import threading
import shutil
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# DETAILED LOGGING SETUP
# ============================================================================


class DetailedLogger:
    """Enhanced logger for AUTO.py with structured logging to files"""

    def __init__(self, log_file: Path = Path("scripts/AUTO.logs")):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        self.logger = logging.getLogger("AUTO_DETAILED")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler with detailed format
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Initialize log session
        self.log_session_start()

    def log_session_start(self):
        """Log session start with system info"""
        self.logger.info("="*80)
        self.logger.info("AUTO.PY SESSION STARTED")
        self.logger.info(f"Session ID: {datetime.now().isoformat()}")
        self.logger.info(f"Working Directory: {Path.cwd()}")
        self.logger.info(f"Python Version: {sys.version}")
        self.logger.info("="*80)

    def log_module_start(self, module: str, total_steps: int):
        """Log module processing start"""
        self.logger.info(f"MODULE_START | {module} | steps={total_steps}")

    def log_step(self, module: str, step: int, total: int, action: str):
        """Log individual step"""
        progress = (step / total * 100) if total > 0 else 0
        self.logger.info(f"STEP | {module} | {step:03d}/{total:03d} | {progress:5.1f}% | {action}")

    def log_test_run(self, module: str, test_dir: str, timeout: int, result: Dict[str, Any]):
        """Log test execution details"""
        success = result.get("success", False)
        returncode = result.get("returncode", -1)
        failures_count = len(result.get("failures", []))
        slow_tests_count = len(result.get("slow_tests", []))

        self.logger.info(f"TEST_RUN | {module} | dir={test_dir} | timeout={timeout}s")
        self.logger.info(
            f"TEST_RESULT | {module} | success={success} | returncode={returncode} | failures={failures_count} | slow_tests={slow_tests_count}")

        # Log failures in detail
        for failure in result.get("failures", [])[:10]:  # Limit to 10
            self.logger.warning(f"TEST_FAILURE | {module} | {failure}")

    def log_claude_request(self, prompt_num: int, total_prompts: int, description: str,
                           prompt_length: int):
        """Log Claude API request details"""
        self.logger.info(f"CLAUDE_CALL | {prompt_num:03d}/{total_prompts} | {description}")
        self.logger.info(f"CLAUDE_REQUEST | len={prompt_length} | timeout=900s")
    
    def log_claude_response(self, response_time: float, response_length: int,
                            success: bool, error: str = ""):
        """Log Claude API response details"""
        if success:
            self.logger.info(
                f"CLAUDE_RESPONSE | time={response_time:.1f}s | len={response_length} | SUCCESS")
        else:
            self.logger.error(f"CLAUDE_RESPONSE | time={response_time:.1f}s | ERROR | {error}")
    
    def log_claude_call(self, prompt_num: int, total_prompts: int, description: str,
                        prompt_length: int, response_time: float, response_length: int,
                        success: bool, error: str = ""):
        """Legacy method for backward compatibility - logs both request and response"""
        self.log_claude_request(prompt_num, total_prompts, description, prompt_length)
        if response_time > 0 or error:  # Only log response if there's actual data
            self.log_claude_response(response_time, response_length, success, error)

    def log_progress_update(self, module: str, **kwargs):
        """Log progress updates"""
        updates = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"PROGRESS_UPDATE | {module} | {updates}")

    def log_cascade_check(self, module: str, dependent: str, success: bool):
        """Log cascade check results"""
        result = "SUCCESS" if success else "FAILED"
        self.logger.warning(f"CASCADE_CHECK | {module} -> {dependent} | {result}")

    def log_integration_check(self, module: str, dependency: str, prompt_type: str):
        """Log integration checks"""
        self.logger.info(f"INTEGRATION_CHECK | {module} -> {dependency} | {prompt_type}")

    def log_error(self, module: str, error: str, traceback_str: str = None):
        """Log errors with traceback"""
        self.logger.error(f"MODULE_ERROR | {module} | {error}")
        if traceback_str:
            self.logger.error(f"TRACEBACK | {module} | {traceback_str}")

    def log_completion(self, module: str, status: str, iterations: int, prompts_completed: List[str]):
        """Log module completion"""
        prompts_str = ",".join(prompts_completed)
        self.logger.info(
            f"MODULE_COMPLETE | {module} | status={status} | iterations={iterations} | prompts=[{prompts_str}]")

    def log_final_summary(self, completed: Set[str], failed: Set[str], success_rate: float):
        """Log final execution summary"""
        self.logger.info("="*80)
        self.logger.info("FINAL SUMMARY")
        self.logger.info(
            f"COMPLETED_MODULES | count={len(completed)} | modules={','.join(sorted(completed))}")
        self.logger.info(
            f"FAILED_MODULES | count={len(failed)} | modules={','.join(sorted(failed))}")
        self.logger.info(f"SUCCESS_RATE | {success_rate:.1f}%")
        self.logger.info("="*80)


# Global logger instance
detailed_logger = None

# ============================================================================
# CONFIGURATION
# ============================================================================

# Module hierarchy from lowest to highest dependencies (based on actual codebase analysis)
MODULE_HIERARCHY = [
    # Foundation layer - no dependencies
    "core",
    # Base utilities layer
    "utils", "error_handling",
    # Data & Infrastructure layer
    "database", "monitoring",
    # State & Data layer
    "state", "data",
    # Exchange & Risk layer
    "exchanges", "risk_management",
    # Execution layer
    "execution",
    # ML & Strategy layer
    "ml", "strategies",
    # Advanced features layer
    "analytics", "backtesting", "optimization",
    # Management layer
    "capital_management", "bot_management",
    # Interface layer
    "web_interface"
]

# Module dependencies - Based on ACTUAL imports analysis from codebase
MODULE_DEPENDENCIES = {
    # Foundation layer
    "core": [],  # No dependencies - foundation module

    # Base utilities layer
    "utils": ["core"],
    "error_handling": ["core"],

    # Data & Infrastructure layer
    "database": ["core", "error_handling"],
    "monitoring": ["core", "error_handling"],

    # State & Data layer
    "state": ["core", "database", "error_handling", "utils"],
    "data": ["core", "database", "utils", "error_handling"],

    # Exchange & Risk layer
    "exchanges": ["core", "error_handling", "utils"],
    "risk_management": ["core", "error_handling", "monitoring", "state", "utils"],

    # Execution layer
    "execution": ["core", "exchanges", "risk_management", "error_handling"],

    # ML & Strategy layer
    "ml": ["core", "data", "utils"],
    "strategies": ["core", "data", "risk_management", "utils"],

    # Advanced features layer
    "analytics": ["core", "database", "monitoring"],
    "backtesting": ["core", "strategies", "data", "execution"],
    "optimization": ["core", "strategies", "backtesting"],

    # Management layer (fixed circular dependency)
    "capital_management": ["core", "risk_management", "state"],
    "bot_management": ["core", "state", "capital_management", "strategies"],

    # Interface layer - depends on most modules
    "web_interface": ["core", "error_handling", "utils", "bot_management", "execution", "strategies", "risk_management"],
}

# Reverse dependencies - who depends on each module
DEPENDENT_MODULES = defaultdict(list)
for module, deps in MODULE_DEPENDENCIES.items():
    for dep in deps:
        DEPENDENT_MODULES[dep].append(module)

# Test mapping - verified against actual test directories
TEST_DIRECTORIES = {
    "core": "tests/unit/test_core",
    "utils": "tests/unit/test_utils",
    "error_handling": "tests/unit/test_error_handling",
    "database": "tests/unit/test_database",
    "state": "tests/unit/test_state",
    "monitoring": "tests/unit/test_monitoring",
    "exchanges": "tests/unit/test_exchanges",  # Now consistent with module name
    "risk_management": "tests/unit/test_risk_management",
    "execution": "tests/unit/test_execution",
    "data": "tests/unit/test_data",
    "ml": "tests/unit/test_ml",
    "analytics": "tests/unit/test_analytics",
    "optimization": "tests/unit/test_optimization",
    "strategies": "tests/unit/test_strategies",
    "backtesting": "tests/unit/test_backtesting",
    "capital_management": "tests/unit/test_capital_management",
    "bot_management": "tests/unit/test_bot_management",
    "web_interface": "tests/unit/test_web_interface"
}

# Configuration
# MAX_ITERATIONS_PER_MODULE:
#   > 0: Maximum number of iterations before giving up on a module
#   = 0: Infinite iterations until all tests pass (recommended for comprehensive fixes)
MAX_ITERATIONS_PER_MODULE = 0
MAX_PARALLEL_MODULES = 2  # Reduced for safety
API_RATE_LIMIT_PER_MINUTE = 5
API_RATE_LIMIT_PER_HOUR = 100
TEST_TIMEOUT_PER_MODULE = 300
TEST_TIMEOUT_PER_TEST = 60
CASCADE_CHECK_ENABLED = True
HALLUCINATION_CHECK_ENABLED = True
DRY_RUN_MODE = False  # Set via command line argument

# CRITICAL: Test coverage requirements for financial application
MIN_TEST_COVERAGE = 70  # Minimum acceptable coverage
CRITICAL_MODULE_COVERAGE = 90  # For critical financial modules
FINANCIAL_CRITICAL_MODULES = [
    "risk_management", "execution", "capital_management",
    "strategies", "exchanges", "data"
]

# Modules to skip
SKIP_MODULES = []

# ============================================================================
# HALLUCINATION DETECTOR
# ============================================================================


class HallucinationDetector:
    """Prevents Claude from creating non-existent functions/classes"""

    def __init__(self):
        self.known_functions = set()
        self.known_classes = set()
        self.known_imports = set()
        self._scan_codebase()

    def _scan_codebase(self):
        """Scan codebase to know what exists"""
        import re

        for py_file in Path("src").rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                    # Find function definitions
                    functions = re.findall(r'def\s+(\w+)\s*\(', content)
                    self.known_functions.update(functions)

                    # Find class definitions
                    classes = re.findall(r'class\s+(\w+)\s*[:\(]', content)
                    self.known_classes.update(classes)

                    # Find imports
                    imports = re.findall(r'from\s+([\w\.]+)\s+import', content)
                    self.known_imports.update(imports)
                    imports = re.findall(r'import\s+([\w\.]+)', content)
                    self.known_imports.update(imports)
            except:
                pass

    def get_constraints(self) -> str:
        """Get constraints to add to prompts"""
        return """
CRITICAL ANTI-HALLUCINATION RULES:
1. DO NOT create new functions that don't already exist
2. DO NOT create new classes that don't already exist  
3. DO NOT import modules that aren't in requirements.txt or src/
4. DO NOT add functionality that wasn't there before
5. ONLY fix the specific issue mentioned
6. If a function/class doesn't exist, DO NOT create it - report error instead
7. Use ONLY existing exception classes from src.core.exceptions
8. Use ONLY existing types from src.core.types
9. DO NOT write lengthy reports or documentation - just fix the code
10. Focus on fixing issues, not explaining them
"""

# ============================================================================
# CASCADE PROTECTOR
# ============================================================================


class CascadeProtector:
    """Prevents cascading failures across modules"""

    def __init__(self, test_runner):
        self.test_runner = test_runner
        self.module_test_results = {}
        self.cascade_detected = False

    def check_dependents(self, fixed_module: str) -> List[Tuple[str, bool]]:
        """Check if fixing a module broke its dependents"""
        results = []
        dependents = DEPENDENT_MODULES.get(fixed_module, [])

        for dependent in dependents:
            # Only check if dependent was previously working
            if dependent in self.module_test_results:
                prev_result = self.module_test_results[dependent]
                if prev_result.get("success"):
                    # Re-run tests
                    print(f"🔍 Checking if {fixed_module} broke {dependent}...")
                    new_result = self.test_runner.run_tests(dependent, timeout=60)

                    if not new_result.get("success"):
                        print(f"⚠️ CASCADE DETECTED: {fixed_module} broke {dependent}!")
                        self.cascade_detected = True
                        results.append((dependent, False))
                    else:
                        results.append((dependent, True))

        return results

    def update_baseline(self, module: str, test_result: Dict):
        """Update baseline test results"""
        self.module_test_results[module] = test_result

# ============================================================================
# PROMPT LIBRARY - GRANULAR & SPECIFIC
# ============================================================================


class PromptType(Enum):
    """Types of prompts for different fix categories"""
    IMPORTS_ONLY = "imports_only"
    TYPES_ONLY = "types_only"
    ASYNC_ONLY = "async_only"
    RESOURCES_ONLY = "resources_only"
    ERRORS_ONLY = "errors_only"
    TEST_FIXES = "test_fixes"
    SLOW_TESTS = "slow_tests"
    TEST_VERIFICATION = "test_verification"  # New comprehensive test check
    TEST_COVERAGE = "test_coverage"  # New coverage enforcement
    INTEGRATION = "integration"
    # Trading-specific prompts
    SERVICE_LAYER = "service_layer"
    DATABASE_MODELS = "database_models"
    FINANCIAL_PRECISION = "financial_precision"
    DEPENDENCY_INJECTION = "dependency_injection"
    WEBSOCKET_ASYNC = "websocket_async"
    FACTORY_PATTERNS = "factory_patterns"


# More granular, specific prompts with anti-hallucination and no-report instruction
MODULE_PROMPTS = {
    PromptType.IMPORTS_ONLY: """Use Task tool: system-design-architect
Fix ONLY these specific import issues in src/{module}/:

ISSUES TO FIX:
1. NameError for undefined names - add the missing import
2. ImportError for modules not found - fix the import path
3. Circular imports - reorganize imports to break cycles

RULES:
- ONLY add imports for names that are undefined
- Use existing imports from other files in the same module as examples
- Import order: stdlib → third-party → local (from src.*)
- DO NOT create new functions or classes
- DO NOT change any logic inside functions
- DO NOT import anything not in requirements.txt or src/

VALIDATION:
After changes, this must pass: python -m py_compile src/{module}/*.py

{anti_hallucination}
""",

    PromptType.TYPES_ONLY: """Use Task tool: code-guardian-enforcer
Fix ONLY type annotation issues in src/{module}/:

ISSUES TO FIX:
1. Missing type hints on function parameters
2. Missing return type annotations
3. Incorrect type annotations causing mypy errors

RULES:
- Add type hints ONLY where missing
- Use types from typing module: Optional, List, Dict, Any, Union, Tuple
- Use domain types from src.core.types: OrderType, PositionType, etc.
- DO NOT change function logic
- DO NOT create new type definitions
- Common patterns:
  - Prices/amounts: Decimal
  - IDs: int or str  
  - Timestamps: datetime
  - Async functions: async def func() -> Dict[str, Any]

VALIDATION:
After changes, this must pass: mypy src/{module}/ --ignore-missing-imports

{anti_hallucination}
""",

    PromptType.ASYNC_ONLY: """Use Task tool: financial-api-architect
Fix ONLY async/await issues in src/{module}/:

ISSUES TO FIX:
1. "coroutine was never awaited" - add missing await
2. Blocking I/O in async function - make it async or run in executor
3. Missing async with for context managers

RULES:
- ONLY add await where coroutines are called without await
- ONLY add async to functions that use await
- DO NOT change business logic
- DO NOT create new async functions
- Pattern: If you see `some_async_func()` without await, add await

VALIDATION:
No "coroutine was never awaited" warnings when running tests

{anti_hallucination}
""",

    PromptType.RESOURCES_ONLY: """Use Task tool: infrastructure-wizard
Fix ONLY resource leak issues in src/{module}/:

ISSUES TO FIX:
1. Unclosed database connections
2. Unclosed file handles  
3. Unclosed websocket connections
4. Missing finally blocks

RULES:
- Add try/finally or context managers ONLY where resources aren't closed
- Use existing patterns from the codebase
- DO NOT change business logic
- DO NOT create new functions
- Pattern: 
  ```python
  conn = None
  try:
      conn = get_connection()
      # use conn
  finally:
      if conn:
          conn.close()
  ```

{anti_hallucination}
""",

    PromptType.ERRORS_ONLY: """Use Task tool: quality-control-enforcer
Fix ONLY error handling issues in src/{module}/:

ISSUES TO FIX:
1. Bare except clauses - replace with specific exceptions
2. Missing error handling for operations that can fail
3. Swallowed exceptions that should be re-raised

RULES:
- Replace `except:` with `except Exception:` or specific exception
- Use exceptions from src.core.exceptions where they exist
- DO NOT change success case logic
- DO NOT create new exception classes
- Pattern:
  ```python
  try:
      risky_operation()
  except SpecificError as e:
      logger.error(f"Operation failed: {{e}}")
      raise
  ```

{anti_hallucination}
""",

    PromptType.TEST_FIXES: """Use Task tool: integration-test-architect
Fix ONLY these test failures in {test_dir}:

TEST FAILURES:
{failures}

RULES:
- Fix ONLY the failing assertions or mocks
- DO NOT change test logic or coverage
- DO NOT skip or remove tests
- Update mocks to match actual interfaces
- Fix expected values to match actual behavior
- Common fixes:
  - Update mock.return_value to match actual return
  - Fix mock side_effect for exceptions
  - Add missing await for async tests
  - Update expected values in assertions

{anti_hallucination}
""",

    PromptType.SLOW_TESTS: """Use Task tool: performance-optimization-specialist
Optimize ONLY these slow tests in {test_dir}:

SLOW TESTS (>{timeout}s):
{slow_tests}

COMPREHENSIVE OPTIMIZATION STRATEGIES:
1. **Mock Heavy Operations**: Mock database queries, network calls, file I/O with lightweight responses
2. **Use Fixtures**: Convert repeated setUp/tearDown to pytest fixtures with scope="module" or "session"
3. **In-Memory Databases**: Replace real databases with sqlite:///:memory: for tests
4. **Reduce Data Size**: Use minimal datasets (5-10 items instead of 100+)
5. **Mock Time Operations**: Mock time.sleep(), datetime.now(), and time-based waits
6. **Async Optimization**: Use pytest.mark.asyncio properly, avoid unnecessary awaits
7. **Disable Logging**: Set logging to CRITICAL during tests to reduce I/O
8. **Batch Assertions**: Group related assertions to reduce test setup overhead
9. **Skip Integration Parts**: Mock external service calls instead of real connections
10. **Use Test Doubles**: Replace heavy objects with lightweight test doubles

CRITICAL RULES:
- DO NOT change what the test validates
- DO NOT reduce test coverage
- DO NOT skip important edge cases
- Maintain test reliability and determinism
- DO NOT write any reports or lengthy explanations - just apply the fixes

{anti_hallucination}
""",

    PromptType.TEST_VERIFICATION: """Use Task tool: financial-qa-engineer
Comprehensive test verification for {module} module - ensure tests are ACTUALLY testing logic correctly:

VERIFICATION CHECKLIST:
1. **Test Logic Correctness**:
   - Are assertions actually checking the RIGHT values?
   - Do mocks return realistic data (not just None or empty)?
   - Are edge cases properly tested (empty data, nulls, extreme values)?
   - Are financial calculations verified with exact expected values?
   - Do tests cover BOTH success AND failure scenarios?

2. **Data Validity**:
   - Are test fixtures using realistic data shapes and values?
   - For trading tests: prices > 0, quantities > 0, proper decimal precision
   - For timestamps: realistic dates, proper timezone handling
   - For IDs: consistent types (int vs string) across tests
   - Mock data matches actual API response structures

3. **Critical Path Coverage**:
   - Main business logic paths have tests
   - Error handling paths are tested
   - Resource cleanup (finally blocks) are tested
   - Async operations properly awaited in tests
   - Database transactions are tested for both commit and rollback

4. **Financial Accuracy** (for trading modules):
   - Price calculations use Decimal, not float
   - Commission/fee calculations are verified
   - Position sizing respects limits
   - Risk calculations match expected formulas
   - P&L calculations are precise to 8 decimal places

5. **Test Independence**:
   - Tests don't depend on execution order
   - Each test has proper setup/teardown
   - No shared mutable state between tests
   - Database tests use transactions or test databases

FIX THESE COMMON ISSUES:
- Tests that always pass (assertions like assert True or assert result is not None)
- Mocks that return None when code expects real objects
- Missing assertions on critical return values
- Tests that don't actually call the function being tested
- Incomplete error scenario testing

VALIDATION:
After fixes, run: python -m pytest {test_dir} -v --tb=short
All tests must pass AND actually validate the module's behavior

{anti_hallucination}
""",

    PromptType.TEST_COVERAGE: """Use Task tool: financial-qa-engineer
Ensure {module} module has MINIMUM 70% test coverage (90% for critical modules):

COVERAGE REQUIREMENTS:
- MINIMUM: 70% overall coverage
- CRITICAL MODULES ({critical_modules}): 90% coverage required
- Financial calculations: 100% coverage required

CHECK AND ADD MISSING TESTS FOR:
1. **Uncovered Functions**: Add tests for any function with 0% coverage
2. **Uncovered Branches**: Add tests for if/else branches not covered
3. **Exception Paths**: Add tests that trigger exception handling
4. **Edge Cases**: Empty inputs, None values, boundary conditions
5. **Integration Points**: Where module interacts with others

GENERATE MISSING TESTS:
```python
# For each uncovered function, create tests like:
def test_function_name_success_case():
    # Arrange
    # Act  
    # Assert

def test_function_name_edge_case():
    # Test with edge inputs

def test_function_name_error_case():
    # Test error handling
```

CRITICAL TEST SCENARIOS FOR FINANCIAL MODULES:
- Decimal precision preservation
- Overflow/underflow handling
- Negative value handling
- Zero value edge cases
- Maximum position size limits
- Minimum order size validation
- Rate limiting behavior
- Connection failure recovery

RUN COVERAGE CHECK:
```bash
python -m pytest {test_dir} --cov=src/{module} --cov-report=term-missing --cov-fail-under=70
```

For critical modules, use --cov-fail-under=90

ADD TESTS UNTIL COVERAGE MEETS REQUIREMENTS!

{anti_hallucination}
""",

    PromptType.SERVICE_LAYER: """Use Task tool: system-design-architect
Fix ONLY service layer violations in src/{module}/:

ISSUES TO FIX:
1. Controllers calling repositories directly (should call services)
2. Business logic in controllers (should be in services)
3. Services with tight coupling to infrastructure
4. Missing service interfaces/protocols
5. Circular dependencies between services

RULES:
- Move business logic from controllers to services
- Controllers should ONLY call services, never repositories
- Services should use dependency injection for infrastructure
- Create service interfaces where missing
- Break circular dependencies with events or interfaces
- Use existing service patterns from other modules

{anti_hallucination}
""",

    PromptType.DATABASE_MODELS: """Use Task tool: postgres-timescale-architect
Fix ONLY database model issues in src/{module}/:

ISSUES TO FIX:
1. Missing foreign key relationships
2. Incorrect relationship configurations (back_populates, cascade)
3. Missing database constraints (CheckConstraint, UniqueConstraint)
4. Incorrect column types for financial data (should use DECIMAL)
5. Missing indexes for performance
6. Circular import issues in model relationships

RULES:
- Use DECIMAL(20, 8) for all financial amounts (prices, quantities)
- Add proper foreign keys with CASCADE options
- Configure back_populates correctly on both sides
- Add check constraints for business rules (quantity > 0, etc.)
- Add indexes for frequently queried columns
- Import models properly to avoid circular imports

{anti_hallucination}
""",

    PromptType.FINANCIAL_PRECISION: """Use Task tool: risk-management-expert
Fix ONLY financial precision issues in src/{module}/:

ISSUES TO FIX:
1. Using float instead of Decimal for financial calculations
2. Missing precision in price/quantity calculations
3. Rounding errors in profit/loss calculations
4. Incorrect decimal places for different asset types
5. Currency conversion precision issues

RULES:
- Use Decimal for ALL financial calculations (never float)
- Use DECIMAL(20, 8) precision for crypto (8 decimal places)
- Use DECIMAL(20, 4) precision for forex (4 decimal places)
- Use DECIMAL(20, 2) precision for stocks (2 decimal places)
- Always specify decimal context for rounding
- Validate precision in calculations and comparisons

{anti_hallucination}
""",

    PromptType.DEPENDENCY_INJECTION: """Use Task tool: system-design-architect
Fix ONLY dependency injection issues in src/{module}/:

ISSUES TO FIX:
1. Services creating dependencies directly (should inject)
2. Missing service registration in containers
3. Circular dependencies in DI container
4. Incorrect service lifetimes (singleton vs transient)
5. Missing factory patterns for complex dependencies

RULES:
- Use constructor injection for dependencies
- Register services in DI container with correct lifetime
- Use factory patterns for complex service creation
- Break circular dependencies with interfaces
- Follow existing DI patterns from other modules
- Use ServiceContainer.register_singleton for stateful services

{anti_hallucination}
""",

    PromptType.WEBSOCKET_ASYNC: """Use Task tool: realtime-dashboard-architect
Fix ONLY WebSocket and async issues in src/{module}/:

ISSUES TO FIX:
1. Missing await on async WebSocket operations
2. Blocking operations in async WebSocket handlers
3. Memory leaks from unclosed WebSocket connections
4. Race conditions in concurrent WebSocket handling
5. Improper error handling in WebSocket streams

RULES:
- Add await to all async WebSocket operations
- Use async context managers for WebSocket connections
- Add proper error handling and connection cleanup
- Use asyncio.gather() for concurrent operations
- Implement proper backpressure handling
- Add connection timeout and heartbeat mechanisms

{anti_hallucination}
""",

    PromptType.FACTORY_PATTERNS: """Use Task tool: system-design-architect
Fix ONLY factory pattern issues in src/{module}/:

ISSUES TO FIX:
1. Factory methods not using dependency injection
2. Hard-coded service creation in factories
3. Missing factory registration in service containers
4. Factories not following interface patterns
5. Complex factory logic that should be simplified

RULES:
- Use dependency injection in factory methods
- Register factories in service container
- Return interfaces, not concrete classes
- Keep factory logic simple and focused
- Use existing factory patterns from other modules
- Follow service locator pattern for complex creation

{anti_hallucination}
"""
}

# Integration prompts
INTEGRATION_PROMPTS = {
    "verify_usage": """Use Task tool: integration-architect
Verify {module} correctly uses {dependency} and fix any issues:

CHECK & FIX:
1. Is {module} calling {dependency}'s functions correctly?
2. Are the parameters being passed correct types?
3. Is {module} handling {dependency}'s exceptions?
4. Is there duplicate code that {dependency} already provides?

RULES:
- ONLY fix incorrect usage of {dependency}
- DO NOT create new integration points
- DO NOT change {dependency} itself
- Use {dependency}'s existing functions instead of duplicating

{anti_hallucination}
""",

    "prevent_breaking": """Use Task tool: system-design-architect  
Cascade changes from {module} to {dependent}:

CHECK:
1. Did we change any function signatures that {dependent} uses?
2. Did we change any return types?
3. Did we remove any functions/classes {dependent} imports?

FIX by cascading changes:
- If function signature changed, UPDATE all calls in {dependent}
- If return type changed, UPDATE handling in {dependent}
- If function/class removed, REMOVE or UPDATE imports in {dependent}
- DO NOT keep backward compatibility - we're in dev stage
- DO NOT create duplicate/enhanced versions
- Make the breaking changes cascade properly

{anti_hallucination}
""",

    "check_cross_module": """Use Task tool: integration-architect
Check cross-module integration between {module} and {dependency}:

CHECK:
1. Are database models consistent between modules?
2. Are event names matching between publisher and subscriber?
3. Are API endpoints called with correct parameters?
4. Are shared types properly imported from core.types?
5. Are error codes consistent across modules?

FIX:
- Align database model definitions
- Match event names exactly
- Update API calls with correct parameters
- Import shared types from src.core.types
- Use consistent error codes from src.core.exceptions

{anti_hallucination}
""",

    "fix_service_layer": """Use Task tool: system-design-architect
Fix service layer violations in {module}:

CHECK:
1. Is business logic in controllers instead of services?
2. Are repositories called directly from controllers?
3. Are services tightly coupled to infrastructure?
4. Missing service abstractions?
5. Cross-service dependencies creating cycles?

FIX:
- Move business logic from controllers to services
- Controllers should only call services, not repositories
- Use dependency injection for infrastructure
- Create service interfaces/protocols
- Break cycles through event-driven patterns

{anti_hallucination}
""",

    "align_data_flow": """Use Task tool: data-pipeline-maestro
Align data flow patterns between {module} and {dependency}:

CHECK:
1. Data transformation consistency
2. Message queue patterns (pub/sub vs req/reply)
3. Batch vs stream processing alignment
4. Data validation at boundaries
5. Consistent error propagation

FIX:
- Use consistent data transformation patterns
- Align on messaging patterns
- Match processing paradigms
- Add validation at module boundaries
- Propagate errors consistently

{anti_hallucination}
"""
}

# ============================================================================
# PROGRESS TRACKER
# ============================================================================


@dataclass
class ModuleProgress:
    """Tracks progress for a single module"""
    module: str
    status: str = "pending"  # pending, in_progress, completed, failed
    iteration: int = 0
    prompts_completed: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    cascade_checks: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ProgressTracker:
    """Manages progress persistence and recovery"""

    def __init__(self, progress_file: Path = Path("progress.json")):
        self.progress_file = progress_file
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.modules: Dict[str, ModuleProgress] = {}
        self.completed_modules: Set[str] = set()
        self.failed_modules: Set[str] = set()
        self._lock = threading.Lock()
        self._load_progress()

    def _load_progress(self):
        """Load progress from file if exists"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    for module_data in data.get("modules", []):
                        module = ModuleProgress(**module_data)
                        self.modules[module.module] = module
                        if module.status == "completed":
                            self.completed_modules.add(module.module)
                        elif module.status == "failed":
                            self.failed_modules.add(module.module)
                    print(f"📚 Loaded progress: {len(self.completed_modules)} completed")
            except Exception as e:
                print(f"⚠️ Could not load progress: {e}")

    def _save_progress_internal(self):
        """Internal save without lock - must be called with lock already held"""
        try:
            data = {
                "modules": [
                    {
                        "module": m.module,
                        "status": m.status,
                        "iteration": m.iteration,
                        "prompts_completed": m.prompts_completed,
                        "test_results": m.test_results,
                        "errors": m.errors,
                        "cascade_checks": m.cascade_checks,
                        "timestamp": m.timestamp
                    }
                    for m in self.modules.values()
                ],
                "last_updated": datetime.now().isoformat(),
                "dry_run": DRY_RUN_MODE  # Save dry_run flag
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            # Progress saves silently - details in log file only
        except Exception as e:
            print(f"⚠️ Could not save progress: {e}")

    def save_progress(self):
        """Save current progress to file"""
        with self._lock:
            self._save_progress_internal()

    def _get_module_progress_internal(self, module: str) -> ModuleProgress:
        """Internal get/create without lock - must be called with lock already held"""
        if module not in self.modules:
            self.modules[module] = ModuleProgress(module=module)
        return self.modules[module]

    def get_module_progress(self, module: str) -> ModuleProgress:
        """Get or create module progress"""
        with self._lock:
            return self._get_module_progress_internal(module)

    def update_module(self, module: str, **kwargs):
        """Update module progress"""
        with self._lock:
            # Use internal method that doesn't acquire lock again
            progress = self._get_module_progress_internal(module)
            for key, value in kwargs.items():
                if hasattr(progress, key):
                    setattr(progress, key, value)

            if progress.status == "completed":
                self.completed_modules.add(module)
                self.failed_modules.discard(module)
            elif progress.status == "failed":
                self.failed_modules.add(module)
                self.completed_modules.discard(module)

            # Log progress update (detailed logging only)
            if detailed_logger:
                detailed_logger.log_progress_update(module, **kwargs)

            # Call internal save that doesn't acquire lock again
            self._save_progress_internal()

    def is_module_complete(self, module: str) -> bool:
        """Check if module is complete"""
        return module in self.completed_modules

# ============================================================================
# RATE LIMITER
# ============================================================================


class RateLimiter:
    """Manages API rate limiting"""

    def __init__(self):
        self.calls_per_minute = deque()
        self.calls_per_hour = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self) -> float:
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()

            # Clean old entries
            minute_ago = now - 60
            hour_ago = now - 3600

            self.calls_per_minute = deque(
                t for t in self.calls_per_minute if t > minute_ago
            )
            self.calls_per_hour = deque(
                t for t in self.calls_per_hour if t > hour_ago
            )

            # Check limits
            wait_time = 0
            if len(self.calls_per_minute) >= API_RATE_LIMIT_PER_MINUTE:
                wait_time = 60 - (now - self.calls_per_minute[0])
            elif len(self.calls_per_hour) >= API_RATE_LIMIT_PER_HOUR:
                wait_time = 3600 - (now - self.calls_per_hour[0])

            if wait_time > 0:
                print(f"⏳ Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

            # Record call
            self.calls_per_minute.append(now)
            self.calls_per_hour.append(now)

            return wait_time

# ============================================================================
# CLAUDE API EXECUTOR
# ============================================================================


class ClaudeExecutor:
    """Executes Claude API calls with error handling and retry logic"""

    def __init__(self, rate_limiter: RateLimiter, hallucination_detector: HallucinationDetector):
        self.rate_limiter = rate_limiter
        self.hallucination_detector = hallucination_detector
        self.max_retries = 3
        self.retry_delay = 5
        self.total_prompts = 0
        self.current_prompt = 0

    def execute_prompt(self, prompt: str, description: str = "") -> Tuple[bool, str]:
        """Execute a Claude prompt with anti-hallucination and retry logic"""

        self.current_prompt += 1

        # Minimal Claude call indicator - use newline for parallel safety
        print(
            f"🤖 API Call {self.current_prompt}/{self.total_prompts if self.total_prompts > 0 else '?'}")

        # Check for dry-run mode
        if DRY_RUN_MODE:
            print(f"🎭 DRY-RUN MODE: Skipping actual Claude API call")
            print(f"📄 Would send prompt of {len(prompt)} characters")
            return True, "[DRY-RUN] Mock response - no actual changes made"

        # Add instruction to not write reports
        no_reports_instruction = "\n\nIMPORTANT: DO NOT write lengthy reports, summaries, or explanations. Only apply the fixes directly."

        # Add anti-hallucination constraints to prompt
        if "{anti_hallucination}" in prompt:
            prompt = prompt.replace(
                "{anti_hallucination}",
                self.hallucination_detector.get_constraints() + no_reports_instruction
            )
        else:
            prompt += no_reports_instruction

        retries = 0
        while retries < self.max_retries:
            # Wait for rate limit
            self.rate_limiter.wait_if_needed()

            # Prepare command
            cmd = [
                "claude",
                "--dangerously-skip-permissions",
                "--model", "claude-sonnet-4-20250514",
                "-p", prompt
            ]

            try:
                if retries > 0:
                    print(f"🔄 Retry {retries}/{self.max_retries} for {description}...")
                else:
                    print(f"🚀 Sending to Claude API...")
                    # Log Claude request start (only on first attempt)
                    if detailed_logger:
                        detailed_logger.log_claude_request(
                            self.current_prompt, self.total_prompts, description,
                            len(prompt)
                        )

                start_time = time.time()
                print(f"⏳ Executing command...")

                # No timeout for Claude API - it can legitimately take a long time
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=None  # No timeout - Claude can run as long as needed
                )

                elapsed = time.time() - start_time

                # Log successful Claude response (details go to log file only)
                if detailed_logger:
                    detailed_logger.log_claude_response(
                        elapsed, len(result.stdout), True
                    )

                # Success - return result (no console spam)
                return True, result.stdout

            except subprocess.TimeoutExpired as e:
                elapsed = time.time() - start_time
                error_msg = f"Timeout after {elapsed:.1f}s"
                print(f"⚠️ Claude API timeout: {error_msg}")
                if detailed_logger:
                    detailed_logger.log_claude_response(
                        elapsed, 0, False, error_msg
                    )
                return False, error_msg
            except subprocess.CalledProcessError as e:
                elapsed = time.time() - start_time
                # Capture full error details
                error_msg = f"Exit code: {e.returncode}\n"
                if e.stderr:
                    error_msg += f"Stderr: {e.stderr[:1000]}\n"
                if e.stdout:
                    error_msg += f"Stdout: {e.stdout[:500]}"
                if not e.stderr and not e.stdout:
                    error_msg = f"Command failed with exit code {e.returncode}: {str(e)}"

                # Log failed Claude response with full error
                if detailed_logger:
                    detailed_logger.log_claude_response(
                        elapsed, 0, False, error_msg[:500]
                    )

                # Print error to console for visibility
                print(f"❌ Claude API error: {error_msg[:200]}")

                # Check for 500 error or rate limit
                if "500" in error_msg or "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    retries += 1
                    if retries < self.max_retries:
                        print(
                            f"⚠️ API error (likely 500 or rate limit), retrying in {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                        continue

                return False, f"Error: {error_msg[:200]}"
            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = f"{type(e).__name__}: {str(e)}"

                # Log exception with full details
                if detailed_logger:
                    detailed_logger.log_claude_response(
                        elapsed, 0, False, error_msg
                    )

                # Print to console
                print(f"⚠️ Unexpected error: {error_msg}")

                retries += 1
                if retries < self.max_retries:
                    print(f"⚠️ Exception occurred: {e}, retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    continue
                return False, str(e)

        # Log max retries exceeded
        if detailed_logger:
            detailed_logger.log_claude_response(
                0, 0, False, f"Max retries ({self.max_retries}) exceeded"
            )
        return False, f"Max retries ({self.max_retries}) exceeded"

    def request_feedback(self, module: str, issue: str, agent: str = "code-guardian-enforcer") -> Tuple[bool, str]:
        """Request feedback from specialized agent"""

        feedback_prompt = f"""Use Task tool: {agent}

Please review the following issue in {module} module and provide targeted feedback:

ISSUE:
{issue}

Provide ONLY:
1. Is this a real issue that needs fixing? (YES/NO)
2. If YES, what specific fix is needed? (1-2 sentences)
3. Any critical warnings about potential side effects?

DO NOT write reports or lengthy explanations.
"""

        return self.execute_prompt(feedback_prompt, f"Getting feedback from {agent}")

# ============================================================================
# TEST RUNNER
# ============================================================================


class TestRunner:
    """Runs tests and analyzes results"""

    def __init__(self):
        self.slow_test_threshold = 5

    def run_tests(self, module: str, timeout: int = TEST_TIMEOUT_PER_MODULE) -> Dict[str, Any]:
        """Run tests for a module"""

        print(f"\n🧪 RUNNING TESTS for {module}")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")

        test_dir = TEST_DIRECTORIES.get(module, f"tests/unit/test_{module}")
        print(f"📁 Test directory: {test_dir}")

        # Check for dry-run mode
        if DRY_RUN_MODE:
            print(f"🎭 DRY-RUN MODE: Simulating test execution")
            print(f"📄 Would run: pytest {test_dir} -v --tb=short")
            # Return simulated test results
            return {
                "success": True,
                "passed": True,
                "failures": [],
                "slow_tests": [],
                "warnings": False,
                "output": "[DRY-RUN] Simulated test output",
                "returncode": 0,
                "dry_run": True
            }

        if not Path(test_dir).exists():
            print(f"⚠️  No test directory found")
            return {
                "success": True,
                "skipped": True,
                "message": "No tests found"
            }

        cmd = [
            "python", "-m", "pytest",
            test_dir,
            "-v", "--tb=short",
            f"--timeout={TEST_TIMEOUT_PER_TEST}",
            "--timeout-method=thread",
            "-W", "ignore::DeprecationWarning"
        ]

        print(f"🔧 Command: pytest {test_dir} -v --tb=short")
        print(f"⏱️  Timeout: {timeout}s")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            elapsed = time.time() - start_time
            print(f"✅ Tests completed in {elapsed:.1f}s")

            output = result.stdout + result.stderr
            passed = "passed" in output
            failures = self._extract_failures(output)
            slow_tests = self._extract_slow_tests(output)
            warnings = "warnings summary" in output.lower()

            # Log test results
            print(
                f"📊 Result: {'PASS' if result.returncode == 0 else 'FAIL'} (return code: {result.returncode})")
            if failures:
                print(f"❌ {len(failures)} test failures found")
            if slow_tests:
                print(f"🐌 {len(slow_tests)} slow tests found")

            test_result = {
                "success": result.returncode == 0,
                "passed": passed,
                "failures": failures,
                "slow_tests": slow_tests,
                "warnings": warnings,
                "output": output[:5000],
                "returncode": result.returncode
            }

            # Log detailed test results
            if detailed_logger:
                detailed_logger.log_test_run(module, test_dir, timeout, test_result)

            return test_result

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "timeout": True,
                "message": f"Tests exceeded {timeout}s timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_failures(self, output: str) -> List[str]:
        """Extract test failure information"""
        failures = []
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'FAILED' in line or 'ERROR' in line:
                failures.append(line.strip())
                for j in range(1, min(4, len(lines) - i)):
                    if lines[i + j].strip():
                        failures.append(f"  {lines[i + j].strip()}")
        return failures[:10]  # Limit to 10

    def _extract_slow_tests(self, output: str) -> List[str]:
        """Extract slow test information"""
        slow_tests = []
        lines = output.split('\n')
        for line in lines:
            if 's]' in line and 'test_' in line:
                try:
                    parts = line.split('[')
                    if len(parts) > 1:
                        time_part = parts[-1].split('s]')[0]
                        test_time = float(time_part)
                        if test_time > self.slow_test_threshold:
                            slow_tests.append(f"{line.strip()}")
                except:
                    pass
        return slow_tests[:5]  # Limit to 5

# ============================================================================
# SMART PARALLEL EXECUTOR
# ============================================================================


class SmartParallelExecutor:
    """Manages parallel execution preventing correlated modules from running together"""

    def __init__(self, max_workers: int = MAX_PARALLEL_MODULES):
        self.max_workers = max_workers
        self.currently_processing = set()
        self.lock = threading.Lock()

    def can_process_module(self, module: str) -> bool:
        """Check if module can be processed now without conflicts"""
        with self.lock:
            # Check if any dependency is being processed
            deps = MODULE_DEPENDENCIES.get(module, [])
            for dep in deps:
                if dep in self.currently_processing:
                    return False

            # Check if any dependent is being processed
            dependents = DEPENDENT_MODULES.get(module, [])
            for dependent in dependents:
                if dependent in self.currently_processing:
                    return False

            return True

    def start_processing(self, module: str):
        """Mark module as being processed"""
        with self.lock:
            self.currently_processing.add(module)

    def finish_processing(self, module: str):
        """Mark module as done processing"""
        with self.lock:
            self.currently_processing.discard(module)

    def get_safe_groups(self) -> List[List[str]]:
        """Get groups of modules that can be safely processed in parallel"""
        groups = []
        processed = set(SKIP_MODULES)
        remaining = [m for m in MODULE_HIERARCHY if m not in SKIP_MODULES]

        while remaining:
            group = []
            for module in remaining[:]:
                deps = MODULE_DEPENDENCIES.get(module, [])
                dependents = DEPENDENT_MODULES.get(module, [])

                # Can add to group if:
                # 1. All dependencies are already processed
                # 2. No module in current group depends on it
                # 3. It doesn't depend on any module in current group
                if all(d in processed for d in deps):
                    conflict = False
                    for g_module in group:
                        g_deps = MODULE_DEPENDENCIES.get(g_module, [])
                        if module in g_deps or g_module in deps:
                            conflict = True
                            break
                        if g_module in dependents or module in DEPENDENT_MODULES.get(g_module, []):
                            conflict = True
                            break

                    if not conflict:
                        group.append(module)
                        remaining.remove(module)

            if group:
                groups.append(group)
                processed.update(group)
            else:
                # Process remaining one by one
                if remaining:
                    groups.append([remaining[0]])
                    processed.add(remaining[0])
                    remaining.pop(0)

        return groups

# ============================================================================
# MODULE PROCESSOR WITH CASCADE PROTECTION
# ============================================================================


class ModuleProcessor:
    """Processes a single module with cascade protection"""

    def __init__(
        self,
        claude: ClaudeExecutor,
        test_runner: TestRunner,
        progress_tracker: ProgressTracker,
        cascade_protector: CascadeProtector
    ):
        self.claude = claude
        self.test_runner = test_runner
        self.progress = progress_tracker
        self.cascade_protector = cascade_protector
        self.current_step = 0
        self.total_steps = 0
        self._calculate_total_prompts()

    def _calculate_total_prompts(self):
        """Calculate total number of prompts for all modules"""
        # Base prompts per module
        prompts_per_module = len([p for p in PromptType if p in MODULE_PROMPTS])
        # Estimate iterations and additional prompts
        estimated_iterations = 3  # Average
        self.total_prompts_estimate = len(MODULE_HIERARCHY) * \
            prompts_per_module * estimated_iterations

    def _calculate_module_steps(self, module: str) -> int:
        """Calculate total steps for a module"""
        steps = 0
        steps += 1  # Update status to in_progress
        steps += 1  # Baseline test

        # Calculate actual number of fix prompts
        fix_prompts_count = len([p for p in PromptType if p in MODULE_PROMPTS and p !=
                                PromptType.TEST_FIXES and p != PromptType.SLOW_TESTS])

        if MAX_ITERATIONS_PER_MODULE > 0:
            steps += fix_prompts_count * MAX_ITERATIONS_PER_MODULE  # Fix prompts
            steps += MAX_ITERATIONS_PER_MODULE  # Test runs per iteration
        else:
            # For infinite iterations, estimate reasonable default for progress bar
            estimated_iterations = 3
            steps += fix_prompts_count * estimated_iterations  # Fix prompts
            steps += estimated_iterations  # Test runs per iteration

        steps += len(MODULE_DEPENDENCIES.get(module, []))  # Integration checks
        steps += len(DEPENDENT_MODULES.get(module, []))  # Cascade checks
        return steps

    def _log_step(self, action: str, module: str = "unknown", phase: str = "PROCESSING"):
        """Log current step with clean progress indicator"""
        self.current_step += 1
        progress_pct = (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0

        # Clean, concise output - use newline instead of carriage return for parallel safety
        print(f"🔧 {module.upper()} | {phase} | {action} | {progress_pct:.0f}%")

        # Log to detailed logger only
        if detailed_logger:
            detailed_logger.log_step(module, self.current_step, self.total_steps, action)

    def process_module(self, module: str) -> bool:
        """Process a module with granular fixes and cascade checks"""

        # Skip if already complete
        if self.progress.is_module_complete(module):
            print(f"\n✅ {module.upper()} | SKIPPED | Already completed")
            return True

        # Initialize step tracking for this module
        self.current_step = 0
        self.total_steps = self._calculate_module_steps(module)

        # Clean module start indicator
        print(f"\n📦 {module.upper()} | STARTING | {self.total_steps} steps")

        # Log module start
        if detailed_logger:
            detailed_logger.log_module_start(module, self.total_steps)

        # Update progress
        self._log_step("Initializing", module, "SETUP")
        self.progress.update_module(module, status="in_progress")

        # Get baseline test results
        self._log_step("Baseline tests", module, "TESTING")
        baseline = self.test_runner.run_tests(module)
        self.cascade_protector.update_baseline(module, baseline)

        try:
            module_progress = self.progress.get_module_progress(module)

            # Process with granular prompts and verification after each
            # If MAX_ITERATIONS_PER_MODULE = 0, iterate indefinitely until tests pass
            iteration = module_progress.iteration
            consecutive_api_failures = 0
            max_consecutive_failures = 5  # Safety: exit after 5 consecutive API failures

            while True:
                # Check iteration limit (if not infinite)
                if MAX_ITERATIONS_PER_MODULE > 0 and iteration >= MAX_ITERATIONS_PER_MODULE:
                    break

                # Safety check: prevent infinite loops due to persistent API failures
                if MAX_ITERATIONS_PER_MODULE == 0 and consecutive_api_failures >= max_consecutive_failures:
                    print(f"\n❌ {module.upper()} | FAILED | Too many consecutive API failures")
                    self.progress.update_module(module, status="failed")
                    return False
                # Display iteration info
                if MAX_ITERATIONS_PER_MODULE > 0:
                    self._log_step(
                        f"Iteration {iteration + 1}/{MAX_ITERATIONS_PER_MODULE}", module, "FIXING")
                else:
                    self._log_step(f"Iteration {iteration + 1} (infinite)", module, "FIXING")
                self.progress.update_module(module, iteration=iteration + 1)

                # Granular fix sequence - run all prompts then test
                fix_sequence = [
                    (PromptType.IMPORTS_ONLY, "Fixing imports"),
                    (PromptType.TYPES_ONLY, "Fixing types"),
                    (PromptType.ASYNC_ONLY, "Fixing async/await"),
                    (PromptType.RESOURCES_ONLY, "Fixing resources"),
                    (PromptType.ERRORS_ONLY, "Fixing error handling"),
                    # Trading-specific fixes
                    (PromptType.SERVICE_LAYER, "Fixing service layer"),
                    (PromptType.DATABASE_MODELS, "Fixing database models"),
                    (PromptType.FINANCIAL_PRECISION, "Fixing financial precision"),
                    (PromptType.DEPENDENCY_INJECTION, "Fixing dependency injection"),
                    (PromptType.WEBSOCKET_ASYNC, "Fixing WebSocket/async"),
                    (PromptType.FACTORY_PATTERNS, "Fixing factory patterns"),
                    # Test quality verification - run after code fixes
                    (PromptType.TEST_VERIFICATION, "Verifying test logic correctness"),
                    (PromptType.TEST_COVERAGE, "Ensuring 70%+ test coverage"),
                ]

                # Execute all fix prompts for this iteration
                for prompt_type, description in fix_sequence:
                    if prompt_type.value in module_progress.prompts_completed:
                        continue

                    self._log_step(description, module, "FIXING")

                    # Prepare prompt with module-specific info
                    if prompt_type not in MODULE_PROMPTS:
                        continue

                    # Format prompt with appropriate parameters
                    format_args = {
                        "module": module,
                        "anti_hallucination": "{anti_hallucination}",
                        "test_dir": TEST_DIRECTORIES.get(module, f"tests/unit/test_{module}")
                    }
                    
                    # Add specific parameters for certain prompt types
                    if prompt_type == PromptType.TEST_COVERAGE:
                        format_args["critical_modules"] = ", ".join(FINANCIAL_CRITICAL_MODULES)
                    
                    prompt = MODULE_PROMPTS[prompt_type].format(**format_args)

                    success, output = self.claude.execute_prompt(prompt, description)

                    if success:
                        consecutive_api_failures = 0  # Reset failure counter on success
                        module_progress.prompts_completed.append(prompt_type.value)
                        self.progress.update_module(
                            module,
                            prompts_completed=module_progress.prompts_completed
                        )
                    else:
                        consecutive_api_failures += 1

                # Full test run
                self._log_step("Running tests", module, "TESTING")
                test_results = self.test_runner.run_tests(module)
                self.progress.update_module(module, test_results=test_results)

                if test_results.get("skipped"):
                    print(f"\n✅ {module.upper()} | COMPLETED | No tests found")
                    self.progress.update_module(module, status="completed")
                    return True

                elif test_results["success"]:
                    print(f"\n✅ {module.upper()} | COMPLETED | All tests passing")

                    # Handle slow tests and cascade checks silently
                    if test_results.get("slow_tests"):
                        self._log_step("Optimizing slow tests", module, "OPTIMIZING")
                        slow_tests_str = "\n".join(test_results["slow_tests"])
                        prompt = MODULE_PROMPTS[PromptType.SLOW_TESTS].format(
                            test_dir=TEST_DIRECTORIES.get(module),
                            module=module,
                            slow_tests=slow_tests_str,
                            timeout=TEST_TIMEOUT_PER_TEST,
                            anti_hallucination="{anti_hallucination}"
                        )
                        self.claude.execute_prompt(prompt, "Optimizing slow tests")

                    # Final cascade check
                    if CASCADE_CHECK_ENABLED:
                        self._log_step("Cascade check", module, "VALIDATING")
                        cascade_results = self.cascade_protector.check_dependents(module)
                        for dependent, ok in cascade_results:
                            if not ok:
                                if detailed_logger:
                                    detailed_logger.log_cascade_check(module, dependent, False)

                                # Request feedback on cascade failure
                                self._log_step(
                                    f"Analyzing cascade: {dependent}", module, "FEEDBACK")
                                feedback_success, feedback = self.claude.request_feedback(
                                    module,
                                    f"Cascade failure detected: {module} changes broke {dependent}. Should we proceed with fixes or rollback?",
                                    "system-design-architect"
                                )

                                if feedback_success:
                                    if "ROLLBACK" in feedback.upper():
                                        print(
                                            f"\n🤖 {module.upper()} | FEEDBACK | Rollback recommended for cascade failure")
                                        # Could implement rollback logic here if needed
                                    else:
                                        print(
                                            f"\n🤖 {module.upper()} | FEEDBACK | Proceeding with cascade fixes for {dependent}")

                    self.progress.update_module(module, status="completed")
                    return True

                # Comprehensive test failure fixing
                elif test_results.get("failures"):
                    failure_count = len(test_results["failures"])

                    # Request feedback before attempting to fix failures
                    self._log_step("Getting feedback on failures", module, "ANALYZING")
                    # Limit to first 3 for feedback
                    first_few_failures = test_results["failures"][:3]
                    failure_summary = "\n".join(first_few_failures)

                    feedback_success, feedback = self.claude.request_feedback(
                        module,
                        f"Test failures detected:\n{failure_summary}\n\nShould these failures be fixed or are they false positives?",
                        "integration-test-architect"
                    )

                    # Skip fixing if feedback indicates issues aren't real
                    if feedback_success and "NO" in feedback.upper():
                        self._log_step("Skipping fixes (feedback: not real issues)",
                                       module, "SKIPPING")
                        print(
                            f"\n🤖 {module.upper()} | FEEDBACK | Skipping test fixes - identified as false positives")
                        iteration += 1
                        continue

                    # Fix ALL failures at once
                    self._log_step(f"Fixing {failure_count} failures", module, "FIXING")

                    prompt = MODULE_PROMPTS[PromptType.TEST_FIXES].format(
                        test_dir=TEST_DIRECTORIES.get(module),
                        failures="\n".join(test_results["failures"]),
                        anti_hallucination="{anti_hallucination}"
                    )

                    success, output = self.claude.execute_prompt(prompt, "Fixing test failures")

                    if success:
                        consecutive_api_failures = 0  # Reset failure counter on success
                        # Re-test to check if issues are resolved
                        self._log_step("Verifying fixes", module, "TESTING")
                        test_results = self.test_runner.run_tests(module)
                        self.progress.update_module(module, test_results=test_results)

                        if test_results["success"]:
                            print(f"\n✅ {module.upper()} | COMPLETED | All failures fixed")

                            # Handle optimizations silently
                            if test_results.get("slow_tests"):
                                self._log_step("Optimizing slow tests", module, "OPTIMIZING")
                                slow_tests_str = "\n".join(test_results["slow_tests"])
                                prompt = MODULE_PROMPTS[PromptType.SLOW_TESTS].format(
                                    test_dir=TEST_DIRECTORIES.get(module),
                                    module=module,
                                    slow_tests=slow_tests_str,
                                    timeout=TEST_TIMEOUT_PER_TEST,
                                    anti_hallucination="{anti_hallucination}"
                                )
                                self.claude.execute_prompt(prompt, "Optimizing slow tests")

                            self.progress.update_module(module, status="completed")
                            return True
                    else:
                        consecutive_api_failures += 1

                # Increment iteration counter and continue loop
                iteration += 1

            # Max iterations reached (only applies when MAX_ITERATIONS_PER_MODULE > 0)
            if MAX_ITERATIONS_PER_MODULE > 0:
                # Request feedback before marking as failed
                module_progress = self.progress.get_module_progress(module)
                last_failures = module_progress.test_results.get("failures", [])[:3]
                failure_summary = "\n".join(
                    last_failures) if last_failures else "Unknown test failures"

                feedback_success, feedback = self.claude.request_feedback(
                    module,
                    f"Module {module} reached max iterations ({MAX_ITERATIONS_PER_MODULE}) with persistent issues:\n{failure_summary}\n\nShould we continue or mark as failed?",
                    "quality-control-enforcer"
                )

                if feedback_success and "CONTINUE" in feedback.upper():
                    print(f"\n🤖 {module.upper()} | FEEDBACK | Extending iterations based on feedback")
                    # Could extend MAX_ITERATIONS_PER_MODULE here or switch to infinite mode
                    # For now, just log the recommendation
                    if detailed_logger:
                        detailed_logger.log_error(
                            module, f"Feedback recommended continuing beyond max iterations: {feedback[:200]}", "")

                print(
                    f"\n❌ {module.upper()} | FAILED | Max iterations ({MAX_ITERATIONS_PER_MODULE}) reached")
            else:
                # This should never happen with infinite iterations, but just in case
                print(f"\n❌ {module.upper()} | FAILED | Unexpected exit from infinite loop")
            self.progress.update_module(module, status="failed")
            return False

        except Exception as e:
            print(f"\n❌ {module.upper()} | FAILED | {str(e)}")
            self.progress.update_module(
                module,
                status="failed",
                errors=[str(e)]
            )
            if detailed_logger:
                detailed_logger.log_error(module, str(e), traceback.format_exc())
            return False

# ============================================================================
# INTEGRATION CHECKER
# ============================================================================


class IntegrationChecker:
    """Checks and fixes integration between modules"""

    def __init__(self, claude: ClaudeExecutor, progress_tracker: ProgressTracker):
        self.claude = claude
        self.progress = progress_tracker

    def check_integration(self, module: str, dependency: str) -> bool:
        """Check integration between module and dependency"""

        if not self.progress.is_module_complete(module):
            return False

        if not self.progress.is_module_complete(dependency):
            return False

        print(f"🔗 {module.upper()} → {dependency.upper()} | INTEGRATING")

        # Request feedback before integration checks
        feedback_success, feedback = self.claude.request_feedback(
            module,
            f"About to run integration checks between {module} and {dependency}. Are there known integration issues or conflicts?",
            "integration-architect"
        )

        # Skip integration if feedback indicates major conflicts
        if feedback_success and "SKIP" in feedback.upper():
            print(
                f"\n🤖 {module.upper()} → {dependency.upper()} | FEEDBACK | Skipping integration (conflicts detected)")
            return False
        elif feedback_success and "CAUTION" in feedback.upper():
            print(f"\n🤖 {module.upper()} → {dependency.upper()} | FEEDBACK | Proceeding with caution")

        # 1. Verify correct usage
        prompt = INTEGRATION_PROMPTS["verify_usage"].format(
            module=module,
            dependency=dependency,
            anti_hallucination="{anti_hallucination}"
        )
        success, output = self.claude.execute_prompt(
            prompt,
            f"Verifying {module} uses {dependency} correctly"
        )

        # 2. Check cross-module integration
        prompt = INTEGRATION_PROMPTS["check_cross_module"].format(
            module=module,
            dependency=dependency,
            anti_hallucination="{anti_hallucination}"
        )
        self.claude.execute_prompt(
            prompt,
            f"Checking cross-module integration between {module} and {dependency}"
        )

        # 3. Fix service layer violations
        prompt = INTEGRATION_PROMPTS["fix_service_layer"].format(
            module=module,
            anti_hallucination="{anti_hallucination}"
        )
        self.claude.execute_prompt(
            prompt,
            f"Fixing service layer violations in {module}"
        )

        # 4. Align data flow patterns
        prompt = INTEGRATION_PROMPTS["align_data_flow"].format(
            module=module,
            dependency=dependency,
            anti_hallucination="{anti_hallucination}"
        )
        self.claude.execute_prompt(
            prompt,
            f"Aligning data flow between {module} and {dependency}"
        )

        # 5. Prevent breaking changes in dependent modules
        dependents = DEPENDENT_MODULES.get(module, [])
        for dependent in dependents:
            if self.progress.is_module_complete(dependent):
                prompt = INTEGRATION_PROMPTS["prevent_breaking"].format(
                    module=module,
                    dependent=dependent,
                    anti_hallucination="{anti_hallucination}"
                )
                self.claude.execute_prompt(
                    prompt,
                    f"Ensuring {module} doesn't break {dependent}"
                )

        print(f"✅ Integration verified: {module} → {dependency}")
        return True

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================


class AutoFixer:
    """Main orchestrator with cascade protection"""

    def __init__(self, skip_modules: List[str] = None):
        global SKIP_MODULES
        if skip_modules:
            SKIP_MODULES = skip_modules

        print("🔍 Scanning codebase...")
        self.hallucination_detector = HallucinationDetector()
        print("✅ Hallucination detector ready")

        self.rate_limiter = RateLimiter()
        self.claude = ClaudeExecutor(self.rate_limiter, self.hallucination_detector)
        self.test_runner = TestRunner()

        print("📁 Loading progress...")
        self.progress = ProgressTracker()

        self.cascade_protector = CascadeProtector(self.test_runner)
        self.module_processor = ModuleProcessor(
            self.claude,
            self.test_runner,
            self.progress,
            self.cascade_protector
        )
        self.integration_checker = IntegrationChecker(
            self.claude,
            self.progress
        )
        self.parallel_executor = SmartParallelExecutor()

        # Calculate total prompts
        modules_to_process = [
            m for m in MODULE_HIERARCHY if m not in SKIP_MODULES and not self.progress.is_module_complete(m)]
        prompts_per_module = 5  # Base prompts
        estimated_iterations = 2  # Average iterations
        self.claude.total_prompts = len(modules_to_process) * \
            prompts_per_module * estimated_iterations
        print(f"📊 Estimated total prompts: ~{self.claude.total_prompts}")

    def run_sequential(self) -> bool:
        """Run fixes sequentially (safest)"""
        all_success = True

        for module in MODULE_HIERARCHY:
            if module in SKIP_MODULES:
                continue

            success = self.module_processor.process_module(module)

            # Check integrations immediately after module completes successfully
            if success:
                deps = MODULE_DEPENDENCIES.get(module, [])
                for dep in deps:
                    if dep not in SKIP_MODULES and self.progress.is_module_complete(dep):
                        self.integration_checker.check_integration(module, dep)

                # Also check if this module is a dependency for any completed modules
                for other_module in MODULE_HIERARCHY:
                    if other_module in SKIP_MODULES or other_module == module:
                        continue
                    if self.progress.is_module_complete(other_module):
                        other_deps = MODULE_DEPENDENCIES.get(other_module, [])
                        if module in other_deps:
                            self.integration_checker.check_integration(other_module, module)
            else:
                all_success = False
                if self.cascade_protector.cascade_detected:
                    print("⛔ Stopping due to cascade failure")
                    break

        return all_success

    def run_smart_parallel(self) -> bool:
        """Run in smart parallel mode preventing conflicts"""
        all_success = True
        # groups of modules that can be safely processed in parallel
        groups = self.parallel_executor.get_safe_groups()

        print(f"\n📊 Smart execution plan: {len(groups)} groups")
        for i, group in enumerate(groups, 1):
            print(f"  Group {i}: {', '.join(group)} (safe to run in parallel)")

        for i, group in enumerate(groups, 1):
            print(f"\n{'='*60}")
            print(f"🚀 Processing group {i}/{len(groups)}: {', '.join(group)}")
            print(f"{'='*60}")

            if len(group) == 1:
                # Single module, run directly
                module = group[0]
                self.parallel_executor.start_processing(module)
                success = self.module_processor.process_module(module)
                self.parallel_executor.finish_processing(module)

                if not success:
                    all_success = False
                    if self.cascade_protector.cascade_detected:
                        print("⛔ Cascade detected, stopping parallel execution")
                        return False
            else:
                # Multiple modules, run in parallel with safety checks
                with ThreadPoolExecutor(max_workers=min(len(group), self.parallel_executor.max_workers)) as executor:
                    futures = {}

                    for module in group:
                        # Double-check safety before submitting
                        if self.parallel_executor.can_process_module(module):
                            self.parallel_executor.start_processing(module)
                            future = executor.submit(
                                self.module_processor.process_module,
                                module
                            )
                            futures[future] = module
                        else:
                            print(f"⚠️ Skipping {module} - conflict detected")

                    for future in as_completed(futures):
                        module = futures[future]
                        try:
                            success = future.result()
                            if not success:
                                all_success = False
                        except Exception as e:
                            print(f"❌ {module}: Exception - {e}")
                            all_success = False
                        finally:
                            self.parallel_executor.finish_processing(module)

                        if self.cascade_protector.cascade_detected:
                            print("⛔ Cascade detected, cancelling remaining tasks")
                            # Cancel remaining futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            return False

            # Check integrations after each group
            for module in group:
                if self.progress.is_module_complete(module):
                    deps = MODULE_DEPENDENCIES.get(module, [])
                    for dep in deps:
                        if self.progress.is_module_complete(dep):
                            self.integration_checker.check_integration(module, dep)

                    # Check integration with other modules in the same group
                    for other in group:
                        if other != module and self.progress.is_module_complete(other):
                            # Check if they have dependencies on each other
                            if other in deps:
                                self.integration_checker.check_integration(module, other)
                            other_deps = MODULE_DEPENDENCIES.get(other, [])
                            if module in other_deps:
                                self.integration_checker.check_integration(other, module)

        return all_success

    def print_summary(self):
        """Print final summary"""
        completed = self.progress.completed_modules
        failed = self.progress.failed_modules

        print(f"\n{'='*60}")
        print(f"📊 FINAL SUMMARY")
        print(f"{'='*60}")

        print(f"✅ Completed: {len(completed)} modules")
        if completed:
            for module in sorted(completed):
                print(f"   • {module}")

        print(f"\n❌ Failed: {len(failed)} modules")
        if failed:
            for module in sorted(failed):
                progress = self.progress.get_module_progress(module)
                errors = progress.errors[:2] if progress.errors else ["Unknown"]
                print(f"   • {module}: {', '.join(errors)}")

        if self.cascade_protector.cascade_detected:
            print(f"\n⚠️ CASCADE FAILURES DETECTED - Review changes carefully!")

        total = len(MODULE_HIERARCHY)
        processed = len(completed) + len(failed)
        success_rate = len(completed) / processed * 100 if processed > 0 else 0

        print(f"\n📈 Progress: {processed}/{total} modules")
        print(f"🎯 Success rate: {success_rate:.1f}%")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    global detailed_logger

    print(f"\n{'='*80}")
    print(f"🚀 AUTO_FINAL.py - Financial Application Module Fixer")
    print(f"📅 Started at: {datetime.now().isoformat()}")
    print(f"{'='*80}")

    parser = argparse.ArgumentParser(
        description="AUTO_FINAL - Production-Ready Module Fixer"
    )
    parser.add_argument("--skip", nargs="+", help="Modules to skip", default=[])
    parser.add_argument("--parallel", action="store_true", help="Run in smart parallel mode")
    parser.add_argument("--reset", action="store_true", help="Reset progress")
    parser.add_argument("--module", help="Process single module")
    parser.add_argument("--no-cascade-check", action="store_true", help="Disable cascade checking")
    parser.add_argument("--no-hallucination-check", action="store_true",
                        help="Disable hallucination checking")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode: no code changes or Claude API calls")

    args = parser.parse_args()

    # Configure global settings
    global CASCADE_CHECK_ENABLED, HALLUCINATION_CHECK_ENABLED, DRY_RUN_MODE
    CASCADE_CHECK_ENABLED = not args.no_cascade_check
    HALLUCINATION_CHECK_ENABLED = not args.no_hallucination_check
    DRY_RUN_MODE = args.dry_run

    if DRY_RUN_MODE:
        print(f"\n🎭 DRY-RUN MODE ENABLED")
        print(f"  • No code files will be modified")
        print(f"  • No Claude API calls will be made")
        print(f"  • Tests will be simulated")
        print(f"  • Progress will still be tracked")

    # Reset progress if requested
    if args.reset:
        progress_file = Path("progress.json")
        if progress_file.exists():
            progress_file.unlink()
            print("🔄 Progress reset")
        
        # Also handle log file - rename old one if exists
        log_file = Path("scripts/AUTO.logs")
        if log_file.exists():
            # Create timestamp for backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"scripts/AUTO.logs.{timestamp}.bak"
            log_file.rename(backup_name)
            print(f"📁 Log file backed up to: {backup_name}")

    # Initialize detailed logger (after reset handling)
    detailed_logger = DetailedLogger()
    print(f"📝 Detailed logging enabled: scripts/AUTO.logs")

    # Initialize fixer first so we can access it in signal handler
    fixer = AutoFixer(skip_modules=args.skip)

    # Setup signal handler with actual progress saving
    def signal_handler(sig, frame):
        print("\n\n⚠️ Interrupted! Saving progress...")
        try:
            fixer.progress.save_progress()
            print("✅ Progress saved to progress.json")
        except Exception as e:
            print(f"❌ Could not save progress: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"\n🎮 Execution Mode:")

    try:
        if args.module:
            print(f"  ➡️ Single module: {args.module}")
            success = fixer.module_processor.process_module(args.module)
        elif args.parallel:
            print(f"  ➡️ Parallel execution")
            success = fixer.run_smart_parallel()
        else:
            print(f"  ➡️ Sequential execution")
            success = fixer.run_sequential()

        fixer.print_summary()

        # Log final summary
        if detailed_logger:
            completed = fixer.progress.completed_modules
            failed = fixer.progress.failed_modules
            processed = len(completed) + len(failed)
            success_rate = len(completed) / processed * 100 if processed > 0 else 0
            detailed_logger.log_final_summary(completed, failed, success_rate)

        print(f"\n{'='*80}")
        print(f"📅 Finished at: {datetime.now().isoformat()}")
        print(f"🎯 Overall result: {'SUCCESS ✅' if success else 'FAILURE ❌'}")
        print(f"{'='*80}")

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()

        # Log fatal error
        if detailed_logger:
            detailed_logger.log_error("MAIN", str(e), traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()
