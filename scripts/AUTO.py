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
        stats = result.get("stats", {})
        success = stats.get("failed", 0) == 0 and stats.get("total", 0) > 0
        failures_count = len(result.get("failures", []))

        self.logger.info(f"TEST_RUN | {module} | dir={test_dir} | timeout={timeout}s")
        self.logger.info(
            f"TEST_RESULT | {module} | passed={stats.get('passed', 0)} | failed={stats.get('failed', 0)} | skipped={stats.get('skipped', 0)} | warnings={stats.get('warnings', 0)}")

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

    def log_session_created(self, session_key: str, session_id: str):
        """Log session creation"""
        self.logger.info(f"SESSION_CREATED | key={session_key} | id={session_id}")

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
MAX_ITERATIONS_PER_MODULE = 20  # Increased to 20 for more thorough fixing
MAX_PARALLEL_MODULES = 2  # Reduced for safety
API_RATE_LIMIT_PER_MINUTE = 5
API_RATE_LIMIT_PER_HOUR = 100
TEST_TIMEOUT_PER_MODULE = 300
TEST_TIMEOUT_PER_TEST = 10
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

FOR TEST FILES SPECIFICALLY:
11. ALWAYS check the actual source file before mocking a function
12. Count the EXACT parameters - do not add extra ones
13. If a test calls SomeClass(param1, param2, param3) but the actual class only takes param1, param2 - DECiDE LOGICALLY IF THE TEST IS CORRECT OR IMPLEMENTATION IS CORRECT
14. Mock functions MUST match the actual function signature exactly
15. Do NOT mock functions/methods that don't exist in the source
16. When in doubt, use Read tool to verify the actual implementation
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
    TEST_WARNINGS = "test_warnings"
    TEST_SKIPPED = "test_skipped"
    SLOW_TESTS = "slow_tests"
    COLLECTION_ERRORS = "collection_errors"  # Fix import/patch errors during test collection
    TEST_VERIFICATION = "test_verification"  # New comprehensive test check
    TEST_COVERAGE = "test_coverage"  # New coverage enforcement
    INTEGRATION = "integration"
    # Trading-specific prompts
    SERVICE_LAYER = "service_layer"
    DATABASE_MODELS = "database_models"
    FINANCIAL_PRECISION = "financial_precision"
    DEPENDENCY_INJECTION = "dependency_injection"
    WEBSOCKET_ASYNC = "websocket_async"
    REMOVE_ENHANCED = "remove_enhanced"  # Remove enhanced versions and simplify code
    FACTORY_PATTERNS = "factory_patterns"
    CODE_DUPLICATION = "code_duplication"
    CIRCULAR_DEPENDENCIES = "circular_dependencies"
    MODULE_INTEGRATION = "module_integration"
    FINAL_CLEANUP = "final_cleanup"


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
4. Properly using error_handling module (except core, and utils modules)

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
MENDATORY: You need to complete this in one go, do not stop in the middle, do not break the code into smaller steps, do not ask for confirmation, just do it.

{anti_hallucination}
""",

    PromptType.TEST_FIXES: """Use Task tool: integration-test-architect
Fix ONLY these test failures in {test_dir}:

TEST FAILURES:
{failures}

ANALYSIS FROM PREVIOUS REVIEW:
{feedback}

BASED ON THE ANALYSIS ABOVE:
- If the analysis indicates these are real failures, fix them
- If the analysis provides specific insights, use them to guide your fixes
- If the analysis suggests a root cause, address it

CRITICAL TEST FIXING RULES:
1. VERIFY BEFORE MOCKING:
   - Read the ACTUAL source file to check function signatures
   - Count the EXACT number of parameters the function takes
   - Check if the function is async or sync
   - Verify the function actually exists in the module

2. MOCK ACCURATELY:
   - Mock.return_value must match actual function return type
   - Mock.call_args must match actual function parameters
   - Do NOT add extra parameters that don't exist
   - Do NOT mock functions that don't exist

3. FIX ONLY FAILURES:
   - Fix ONLY the failing assertions or mocks
   - DO NOT change test logic or coverage
   - DO NOT skip or remove tests
   - Update expected values to match actual behavior

4. COMMON FIXES:
   - Update mock.return_value to match actual return
   - Fix mock side_effect for exceptions
   - Add missing await for async tests
   - Remove extra parameters from mock calls
   - Fix import paths if functions were moved

5. BEFORE WRITING ANY MOCK:
   - First: Check if the function exists
   - Second: Count its parameters
   - Third: Check its return type
   - Fourth: Write the mock correctly

{anti_hallucination}
""",

    PromptType.TEST_WARNINGS: """Use Task tool: financial-qa-engineer
Fix these warnings in the {module} module:

{warnings}

RULES:
1. Fix the root cause of warnings, not just suppress them
2. Common warnings and their fixes:
   - DeprecationWarning: Update to use modern APIs
   - RuntimeWarning (coroutine never awaited): Add proper await statements
   - ImportWarning: Fix import organization and cycles
   - UserWarning: Address the underlying issue causing the warning
3. DO NOT use warning filters or suppressions unless absolutely necessary
4. Ensure all async functions are properly awaited
5. Update deprecated code to use current best practices
6. Follow the project's coding standards strictly

Provide the fixes directly without asking questions.
{anti_hallucination}
""",

    PromptType.TEST_SKIPPED: """Use Task tool: integration-test-architect
Fix these skipped tests in {module} module - ALL unit tests should run:

{skipped_tests}

SKIPPED TEST ANALYSIS:
1. Unit tests should NEVER be skipped - they must either pass or fail
2. Common reasons for skipped tests that need fixing:
   - Missing test implementations (add the test logic)
   - Conditional skips based on environment (remove conditions for unit tests)
   - Missing dependencies (mock them instead)
   - Platform-specific tests (make them platform-independent)
   - Feature flags (test both enabled and disabled states)

FIXES TO APPLY:
1. If test is not implemented:
   - Add proper test implementation
   - Test the actual functionality
   - Add appropriate assertions

2. If test has @pytest.skip or pytest.skip():
   - Remove the skip decorator/call
   - Fix the underlying issue causing the skip
   - Add mocks for any external dependencies

3. If test has @pytest.skipif:
   - Remove conditional skipping for unit tests
   - Mock environment-specific features
   - Make tests run regardless of platform/environment

4. If test imports are failing:
   - Fix import errors
   - Add missing dependencies to requirements
   - Mock unavailable modules

RULES:
- ALL unit tests must run and produce pass/fail results
- Integration tests can be skipped, but not unit tests
- Use mocks instead of skipping due to dependencies
- Implement placeholder tests rather than skipping

{anti_hallucination}
""",

    PromptType.SLOW_TESTS: """Use Task tool: performance-optimization-specialist
Optimize ONLY these slow tests in {test_dir}:

SLOW TESTS (>{timeout}s):
{slow_tests}

IMPORTANT: Review your previous attempts in this session before making changes.
If you've already tried fixing these tests, check what approaches you used and their results.
DO NOT repeat fixes that caused import errors or other test failures.
If this issue has appeared multiple times, you may be undoing previous fixes - try a DIFFERENT approach.

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

CRITICAL: VERIFY FUNCTIONS ACTUALLY EXIST:
0. **Function Existence Check** (DO THIS FIRST):
   - For EVERY mocked function: verify it EXISTS in the source code
   - For EVERY function call: verify parameter count matches source
   - If a test calls obj.some_method(a, b, c) - CHECK the source has 3 params
   - If source has obj.some_method(a, b) - FIX test to remove extra param c
   - Do NOT mock non-existent functions - remove or fix the test

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

    PromptType.COLLECTION_ERRORS: """Fix test collection/import errors in {test_dir}:

COLLECTION ERRORS DETECTED:
{collection_errors}

IMPORTANT: Review your previous attempts in this session before making changes.
If you've already tried fixing these collection errors, check what caused them.
DO NOT undo fixes that resolved timeout issues - find a solution that works for both.
If alternating between collection errors and timeouts, the issue likely needs a different approach entirely.

COMMON FIXES:
1. **Wrong patch paths**: 
   - WRONG: patch('src.module.file.time.time') when 'file' is a module not a package
   - RIGHT: patch('time.time') or patch('src.module.file', 'time')
   
2. **Missing dependencies**:
   - Check if required packages are installed (pip install needed-package)
   - Mock external dependencies that aren't available
   
3. **Incorrect mock targets**:
   - Patch at the point of use, not definition
   - Use spec/autospec for better type safety
   
4. **Module-level patches causing import failures**:
   - Move patches inside test methods instead of module level
   - Use setUp/tearDown for common patches

5. **Circular imports**:
   - Delay imports or reorganize code structure
   - Use local imports inside functions

FIX RULES:
- Fix ALL import and collection errors
- Ensure tests can be collected without errors
- Keep existing test logic intact
- Test with: pytest {test_dir} --collect-only

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
""",

    PromptType.CODE_DUPLICATION: """Use Task tool: code-guardian-enforcer
Identify and eliminate code duplication in src/{module}/:

CODE DUPLICATION ANALYSIS:
1. Search for duplicate functions/methods across files
2. Identify repeated logic patterns that should be extracted
3. Find copy-pasted code blocks that need refactoring
4. Detect similar data structures that should be unified
5. Check for redundant utility functions

REFACTORING RULES:
1. Extract common logic into shared utility functions in src/utils/
2. Create base classes for repeated patterns
3. Use dependency injection to share services
4. Move domain-specific utilities to appropriate service modules
5. Ensure DRY principle without over-abstracting

INTEGRATION REQUIREMENTS:
- When extracting to utils, ensure proper imports in all affected files
- Update tests to use refactored code
- Maintain backward compatibility
- Document extracted utilities properly
- Ensure no circular dependencies are created

DO NOT:
- Create unnecessary abstractions
- Move business logic out of service layers
- Break existing module boundaries
- Create utils for single-use cases

{anti_hallucination}
""",

    PromptType.CIRCULAR_DEPENDENCIES: """Use Task tool: system-design-architect
Detect and fix circular dependencies in src/{module}/:

CIRCULAR DEPENDENCY DETECTION:
1. Analyze all imports in the module
2. Trace import chains to detect cycles
3. Identify type-only imports that cause cycles
4. Find service layer circular dependencies
5. Detect hidden cycles through third modules

FIXING STRATEGIES:
1. Use TYPE_CHECKING for type-only imports:
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from other_module import SomeType
   ```

2. Move shared interfaces to src/core/base/interfaces.py

3. Use dependency injection instead of direct imports:
   ```python
   # Instead of: from other_service import OtherService
   def __init__(self, other_service: 'OtherService'):
       self.other_service = other_service
   ```

4. Extract common types to src/core/types/

5. Use string annotations for forward references:
   ```python
   def process(self, data: 'FutureType') -> None:
   ```

VERIFICATION:
- Ensure all imports still work after fixes
- Run tests to verify functionality
- Check that type hints are preserved
- Validate service initialization order

{anti_hallucination}
""",

    PromptType.MODULE_INTEGRATION: """Use Task tool: module-integration-validator
Verify {module} properly integrates with and uses other modules:

INTEGRATION VERIFICATION:
1. Check that {module} correctly uses its dependencies:
   - Verify service injection patterns
   - Ensure proper API usage
   - Validate data contracts
   - Check error handling from dependencies

2. Verify other modules correctly consume {module}:
   - Find all modules that import {module}
   - Verify they use the correct APIs
   - Check they handle {module}'s exceptions
   - Ensure they respect {module}'s contracts

3. Integration patterns to verify:
   - Service layer properly uses repositories
   - Controllers correctly use services
   - Proper event publishing/subscription
   - Correct use of shared utilities
   - Appropriate use of base classes

4. Common integration issues to fix:
   - Direct database access bypassing repositories
   - Skipping service layer business logic
   - Incorrect error propagation
   - Missing dependency injection
   - Improper transaction boundaries

5. Module boundary verification:
   - Ensure {module} doesn't access internals of other modules
   - Verify only public APIs are used
   - Check that module responsibilities are respected
   - Validate data flow follows architecture

FIXES TO APPLY:
- Add missing service injections
- Fix improper direct access patterns
- Implement proper error handling
- Add integration tests for module boundaries
- Update incorrect API usage

DO NOT:
- Break existing working integrations
- Add unnecessary coupling
- Bypass architectural layers
- Create circular dependencies

{anti_hallucination}
""",

    PromptType.FINAL_CLEANUP: """Use Task tool: quality-control-enforcer
Perform FINAL production-ready cleanup for {module} module:

PRODUCTION READINESS CHECKLIST:

1. **Code Quality**:
   - Remove all debug print statements and console.log calls
   - Remove commented-out code blocks
   - Remove TODO/FIXME comments and if it's future work, convert to proper issues with TODO: TBD:
   - Ensure all temporary test code is removed
   - Clean up any experimental or unused code

2. **Documentation**:
   - Ensure all public APIs have proper docstrings
   - Verify type hints are complete and accurate
   - Update module-level docstrings with current functionality
   - Document any complex algorithms or business logic
   - Ensure examples in docstrings are correct

3. **Error Handling**:
   - Verify all exceptions have proper error messages
   - Ensure no bare except clauses
   - Check that errors are logged appropriately
   - Verify graceful degradation for external dependencies

4. **Performance**:
   - Remove any unnecessary loops or iterations
   - Optimize database queries (use select_related/prefetch_related)
   - Ensure proper indexing is suggested for database fields
   - Remove redundant API calls
   - Implement proper caching where beneficial

5. **Security**:
   - Ensure no hardcoded credentials or API keys
   - Verify input validation and sanitization
   - Check for SQL injection vulnerabilities
   - Ensure proper authentication/authorization checks
   - Remove any sensitive data from logs

6. **Configuration**:
   - Move magic numbers to configuration constants
   - Ensure all environment-specific values use config
   - Verify feature flags are properly implemented
   - Check that default values are production-appropriate

7. **Testing**:
   - Ensure test coverage is adequate (70%+ minimum)
   - Verify edge cases are tested
   - Check that mocks are properly cleaned up
   - Ensure tests are deterministic (no random failures)
   - Verify integration tests cover critical paths

8. **Logging**:
   - Ensure appropriate log levels (DEBUG/INFO/WARNING/ERROR)
   - Verify no sensitive data in logs
   - Check that logs are actionable and meaningful
   - Ensure structured logging format

9. **Resource Management**:
   - Verify all connections are properly closed
   - Check for memory leaks
   - Ensure file handles are closed
   - Verify database connections use connection pooling
   - Check that async resources are properly awaited

10. **Code Organization**:
    - Ensure consistent naming conventions
    - Verify proper separation of concerns
    - Check that business logic is in services, not controllers
    - Ensure DRY principle is followed
    - Verify SOLID principles are applied

11. **Dependencies**:
    - Remove unused imports
    - Ensure requirements.txt is up to date
    - Verify no conflicting dependencies
    - Check for security vulnerabilities in dependencies

12. **Financial System Specific**:
    - Verify all monetary calculations use Decimal
    - Ensure proper transaction isolation
    - Check idempotency for critical operations
    - Verify audit trail for financial operations
    - Ensure proper rate limiting for external APIs

APPLY FIXES:
- Fix all issues found during the checklist review
- Ensure changes maintain backward compatibility
- Do not break existing functionality
- Keep changes minimal and focused

DO NOT:
- Add new features
- Refactor working code without clear benefit
- Change public API signatures
- Remove necessary defensive programming

{anti_hallucination}
""",

    PromptType.REMOVE_ENHANCED: """Use Task tool: code-guardian-enforcer
Remove ALL enhanced versions and simplify code in {module} module:

CRITICAL INSTRUCTIONS FOR SIMPLIFICATION:



1. **REFACTOR ORIGINAL CODE**:
   - Find ALL files with '_enhanced', '_v2', '_refactored', '_optimized' suffixes
   - Take the SIMPLEST working parts from enhanced versions (if any)
   - Integrate them into the ORIGINAL file without over-complicating
   - Remove unnecessary abstractions and layers
   - Eliminate redundant wrapper classes
   - Remove excessive validation that duplicates what's done elsewhere

2. **REMOVE ENHANCED VERSIONS**:
   - Find ALL files with '_enhanced', '_v2', '_refactored', '_optimized' suffixes
   - Delete these enhanced version files completely
   - Keep ONLY the original simple implementation

3. **UPDATE ALL REFERENCES**:
   - Search for ALL imports of enhanced versions across the ENTIRE codebase
   - Update them to use the original simple version
   - Fix any method name changes or parameter differences
   - Ensure all tests still pass after updates

4. **SIMPLIFICATION RULES**:
   - One class per responsibility - no "Manager of Managers"
   - Direct function calls over complex event systems
   - Clear variable names over clever abstractions

5. **PRESERVE FUNCTIONALITY**:
   - Keep ALL features that are actually used
   - Maintain existing API contracts
   - Preserve error handling and logging using src.core.exceptions and src.core.logging
   - Keep necessary validation and security checks
   - Maintain test coverage

6. **REMOVE COMPLEXITY**:
   - Delete unused configuration options
   - Remove feature flags for features that are always on
   - Eliminate dead code paths

7. **SEARCH AND REPLACE**:
   After removing enhanced versions, search ENTIRE codebase for:
   - from src.{module}.something_enhanced import
   - from src.{module}.something_v2 import
   - from src.{module}.refactored_something import
   - {module}_enhanced
   - {module}_v2
   
   Replace ALL with the simple original version

8. **TEST VALIDATION**:
   After simplification, ensure:
   - All unit tests pass
   - No import errors
   - No undefined references
   - Coverage remains above 70%

EXAMPLE TRANSFORMATION:
Before: UserManagerEnhancedV2 → UserServiceAdapter → UserRepository → Database
After: UserService → Database

DO NOT:
- Create new enhanced versions
- Add new abstraction layers
- Introduce new design patterns
- Make the code more "elegant" or "sophisticated"
- Add features not already present

FOCUS: Make the code SIMPLER, CLEARER, and EASIER TO UNDERSTAND while keeping it WORKING.
MENDATORY: You need to complete this in one go, do not stop in the middle, do not break the code into smaller steps, do not ask for confirmation, just do it.

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
    hanging_tests: Optional[Dict[str, Any]] = None  # NEW: Track slow/hanging tests
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ProgressTracker:
    """Manages progress persistence and recovery"""

    def __init__(self, progress_file: Path = Path("progress.json"), dry_run: bool = False):
        # Use separate file for dry runs to avoid contaminating real progress
        if dry_run:
            self.progress_file = Path("progress.dry.json")
        else:
            self.progress_file = progress_file
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.modules: Dict[str, ModuleProgress] = {}
        self.completed_modules: Set[str] = set()
        self.failed_modules: Set[str] = set()
        self.completed_integrations: Set[str] = set()  # Track completed integration checks
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
                    # Load completed integrations
                    self.completed_integrations = set(data.get("completed_integrations", []))
                    print(
                        f"📚 Loaded progress: {len(self.completed_modules)} completed, {len(self.completed_integrations)} integrations")
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
                # Save integration checks
                "completed_integrations": list(self.completed_integrations),
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

    def is_integration_complete(self, module1: str, module2: str) -> bool:
        """Check if integration between two modules is complete"""
        # Normalize module names to lowercase for consistent keys
        key = f"{module1.lower()}->{module2.lower()}"
        return key in self.completed_integrations

    def mark_integration_complete(self, module1: str, module2: str):
        """Mark integration between two modules as complete"""
        with self._lock:
            # Normalize module names to lowercase for consistent keys
            key = f"{module1.lower()}->{module2.lower()}"
            self.completed_integrations.add(key)
            self._save_progress_internal()

# ============================================================================
# RATE LIMITER
# ============================================================================


class RateLimiter:
    """Manages API rate limiting"""

    def __init__(self):
        self.calls_per_minute = deque()
        self.calls_per_hour = deque()
        self.lock = threading.Lock()
        # Allow up to 3 concurrent API calls for true parallel execution
        self.api_semaphore = threading.Semaphore(3)

    def wait_if_needed(self) -> float:
        """Wait if rate limit would be exceeded"""
        # Acquire semaphore to limit concurrent API calls to 3
        self.api_semaphore.acquire()
        try:
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
        finally:
            # Always release semaphore to allow next API call
            self.api_semaphore.release()

# ============================================================================
# CLAUDE API EXECUTOR
# ============================================================================


class ClaudeExecutor:
    """Executes Claude API calls with error handling and retry logic"""

    def __init__(self, rate_limiter: RateLimiter, hallucination_detector: HallucinationDetector, dry_run: bool = False):
        self.rate_limiter = rate_limiter
        self.hallucination_detector = hallucination_detector
        self.dry_run = dry_run
        self.max_retries = 3
        self.retry_delay = 5
        self.total_prompts = 0
        self.current_prompt = 0
        self.session_per_milestone = {}  # Store session IDs for each module-milestone pair
        self.session_lock = threading.Lock()  # Protect session dictionary from race conditions
        # FIXED: Remove single current_session_key - we'll pass it as parameter instead
        self._load_existing_sessions()  # Load any existing sessions from progress file

    def _load_existing_sessions(self):
        """Load existing session IDs from progress file"""
        try:
            progress_file = Path("progress.dry.json" if self.dry_run else "progress.json")
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)

                # Extract all session IDs from progress
                for module_data in progress.get("modules", []):
                    module_name = module_data["module"]
                    if "sessions" in module_data:
                        for milestone_key, session_data in module_data["sessions"].items():
                            # Extract milestone number from key like "milestone_0"
                            milestone_num = milestone_key.split('_')[-1]
                            session_key = f"{module_name}_{milestone_num}"
                            session_id = session_data.get("session_id")

                            if session_id:
                                self.session_per_milestone[session_key] = session_id
                                print(f"📂 Loaded session {session_id[:8]}... for {session_key}")

                if self.session_per_milestone:
                    print(f"✅ Loaded {len(self.session_per_milestone)} existing sessions")

        except Exception as e:
            print(f"⚠️ Could not load existing sessions: {e}")

    def get_session_key(self, module: str, milestone: int) -> str:
        """Get the session key for a module-milestone pair"""
        return f"{module}_{milestone}"

    def has_session_for_context(self, module: str, milestone: int) -> bool:
        """Check if we have a session for the given context"""
        session_key = self.get_session_key(module, milestone)
        with self.session_lock:
            return session_key in self.session_per_milestone

    def clear_sessions(self):
        """Clear all stored sessions (useful when --reset is used)"""
        self.session_per_milestone.clear()
        print("🔄 Session cache cleared")

    def _save_session_to_progress(self, session_key: str, session_id: str):
        """Save session ID to progress file for tracking"""
        try:
            # Load current progress
            progress_file = Path("progress.dry.json" if self.dry_run else "progress.json")
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
            else:
                progress = {"modules": []}

            # Find the module in progress
            # Use rsplit to handle module names with underscores (e.g., "error_handling_0")
            if '_' in session_key:
                module_name, milestone_str = session_key.rsplit('_', 1)
                milestone = int(milestone_str)
            else:
                module_name = session_key
                milestone = 0

            for module_data in progress.get("modules", []):
                if module_data["module"] == module_name:
                    # Add sessions field if it doesn't exist
                    if "sessions" not in module_data:
                        module_data["sessions"] = {}

                    # Store session ID with milestone key
                    module_data["sessions"][f"milestone_{milestone}"] = {
                        "session_id": session_id,
                        "created_at": datetime.now().isoformat()
                    }
                    break

            # Save updated progress
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            progress_filename = "progress.dry.json" if self.dry_run else "progress.json"
            print(f"⚠️ Could not save session to {progress_filename}: {e}")

    def execute_prompt(self, prompt: str, description: str = "", module: str = None, milestone: int = None) -> Tuple[bool, str]:
        """Execute a Claude prompt with anti-hallucination and retry logic

        Args:
            prompt: The prompt to send to Claude
            description: Description of what this prompt does
            module: The module being processed (for session management)
            milestone: The milestone number (for session management)
        """

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
        # else:
        #     prompt += no_reports_instruction

        retries = 0
        while retries < self.max_retries:
            # Wait for rate limit
            self.rate_limiter.wait_if_needed()

            # Prepare command
            cmd = [
                "claude",
                "--dangerously-skip-permissions",
                "--model", "claude-sonnet-4-20250514"
            ]

            # Check if we have a session to resume (sessions are created separately in _initialize_session)
            if module and milestone is not None:
                session_key = self.get_session_key(module, milestone)
                with self.session_lock:
                    if session_key in self.session_per_milestone:
                        # Resume existing session
                        session_id = self.session_per_milestone[session_key]
                        if session_id:  # Only use if we have a valid session ID
                            cmd.extend(["--resume", session_id])
                            print(f"📌 Resuming session {session_id[:8]}... for {session_key}")

            # Always add the prompt normally (no JSON output needed here)
            cmd.extend(["-p", prompt])

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

                # Since we initialize sessions separately now, we just handle the plain text response
                response_text = result.stdout

                # Success - return result (no console spam)
                return True, response_text

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

                if "5-hour limit reached" in error_msg and "resets" in error_msg:
                    import re
                    from datetime import datetime, time as dt_time

                    # Extract reset time (e.g., "8am")
                    reset_match = re.search(r'resets (\d+)(am|pm)', error_msg)
                    if reset_match:
                        reset_hour = int(reset_match.group(1))
                        is_pm = reset_match.group(2) == 'pm'

                        # Convert to 24-hour format
                        if is_pm and reset_hour != 12:
                            reset_hour += 12
                        elif not is_pm and reset_hour == 12:
                            reset_hour = 0

                        # Calculate wait time until reset
                        now = datetime.now()
                        reset_time = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)

                        # If reset time is in the past today, it's tomorrow
                        if reset_time <= now:
                            reset_time = reset_time.replace(day=reset_time.day + 1)

                        wait_seconds = (reset_time - now).total_seconds()

                        print(
                            f"⏰ 5-hour limit reached. Waiting until {reset_hour:02d}:00 ({wait_seconds/3600:.1f} hours)")
                        time.sleep(wait_seconds)
                        continue  # Don't increment retries, just continue

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

    def request_feedback(self, module: str, issue: str, agent: str = "code-guardian-enforcer", milestone: int = 0) -> Tuple[bool, str]:
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

        return self.execute_prompt(feedback_prompt, f"Getting feedback from {agent}", module=module, milestone=milestone)

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
            # Return simulated test results in the new JSON report format
            return {
                "stats": {
                    "passed": 10,  # Simulate some passing tests
                    "failed": 2,   # Simulate some failures for testing
                    "skipped": 1,
                    "errors": 0,
                    "total": 13,
                    "warnings": 3  # Simulate some warnings
                },
                "failures": [
                    {"test": "test_example_failure", "error": "AssertionError: Expected True but got False"},
                    {"test": "test_another_failure", "error": "ValueError: Invalid input"}
                ],
                "warnings": [
                    {"category": "DeprecationWarning", "message": "Use of deprecated API",
                        "filename": "test_core.py", "lineno": 42},
                    {"category": "RuntimeWarning", "message": "coroutine was never awaited",
                        "filename": "test_async.py", "lineno": 100},
                    {"category": "UserWarning", "message": "Resource not closed properly",
                        "filename": "test_resources.py", "lineno": 55}
                ],
                "skipped": [{"test": "test_skipped_example", "reason": "Not implemented yet"}],
                "dry_run": True
            }

        if not Path(test_dir).exists():
            print(f"⚠️  No test directory found")
            return {
                "success": False,  # This is a failure, not success!
                "skipped": True,
                "message": "No test directory found",
                "stats": {
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 1,  # Count as error
                    "total": 0,
                    "warnings": 0
                },
                "failures": [],
                "warnings": [],
                "skipped": []
            }

        # Use pytest-json-report for structured output with microsecond precision to avoid collisions
        json_report_file = f"/tmp/test_report_{module}_{int(time.time() * 1000000)}.json"
        cmd = [
            "python", "-m", "pytest",
            test_dir,
            "--json-report",
            f"--json-report-file={json_report_file}",
            "--json-report-omit", "collectors,keywords,metadata",  # Keep report small
            "-q",  # Quiet mode
            "--tb=short"  # Short traceback
        ]

        # Add timeout if pytest-timeout is available
        # We'll skip it for now to avoid errors

        print(f"🔧 Command: pytest {test_dir} --json-report -q --tb=short")
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

            # Check if pytest had collection errors (exit code 2 or 4)
            if result.returncode in [2, 4]:
                print(f"❌ Test collection failed - likely import/syntax errors")
                # Return a special result indicating collection failure
                return {
                    "success": False,
                    "collection_error": True,
                    "stdout": result.stdout[:2000],  # First 2000 chars of output
                    "stderr": result.stderr[:2000] if result.stderr else "",
                    "message": "Test collection failed - check for import errors"
                }

            # Parse JSON report
            test_result = {}
            if Path(json_report_file).exists():
                with open(json_report_file, 'r') as f:
                    json_data = json.load(f)

                # Extract summary - clean structure without duplication
                summary = json_data.get('summary', {})
                test_result = {
                    "stats": {
                        "passed": summary.get('passed', 0),
                        "failed": summary.get('failed', 0),
                        "skipped": summary.get('skipped', 0),
                        "errors": summary.get('error', 0),
                        "total": summary.get('total', 0),
                        "warnings": 0  # Will be set after counting warnings
                    },
                    "failures": [],
                    "warnings": [],
                    "skipped": []
                }

                # Extract detailed test results
                tests = json_data.get('tests', [])
                for test in tests:
                    nodeid = test.get('nodeid', '')
                    outcome = test.get('outcome', '')

                    if outcome == 'failed':
                        error_msg = ""
                        if test.get('call', {}).get('longrepr'):
                            error_msg = str(test['call']['longrepr'])[:500]
                        test_result['failures'].append({
                            "test": nodeid,
                            "error": error_msg
                        })
                    elif outcome == 'skipped':
                        test_result['skipped'].append({
                            "test": nodeid,
                            "reason": test.get('setup', {}).get('longrepr', '')
                        })

                # Extract warnings from JSON report
                warnings_data = json_data.get('warnings', [])
                for warning in warnings_data:
                    test_result['warnings'].append({
                        "category": warning.get('category', 'Warning'),
                        "message": warning.get('message', ''),
                        "filename": warning.get('filename', ''),
                        "lineno": warning.get('lineno', 0)
                    })

                # Also count warnings in stats
                test_result['stats']['warnings'] = len(warnings_data)

                # Display summary
                summary_str = f"❌ {test_result['stats']['failed']} failed" if test_result['stats']['failed'] else ""
                summary_str += f", ✅ {test_result['stats']['passed']} passed" if test_result['stats']['passed'] else ""
                summary_str += f", ⏭️ {test_result['stats']['skipped']} skipped" if test_result['stats']['skipped'] else ""
                summary_str += f", ⚠️ {test_result['stats'].get('warnings', 0)} warnings" if test_result['stats'].get(
                    'warnings', 0) else ""
                print(f"📊 {summary_str.strip(', ')}")

                # Clean up JSON report file
                try:
                    os.remove(json_report_file)
                except:
                    pass
            else:
                print(f"❌ No JSON report generated")
                test_result = {
                    "stats": {
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "errors": 0,
                        "total": 0,
                        "warnings": 0
                    },
                    "failures": [],
                    "warnings": [],
                    "skipped": []
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

    def _extract_failures(self, output: str) -> tuple[List[str], int]:
        """Extract test failure information - returns (limited_list, total_count)"""
        failures_set = set()  # Use set to avoid duplicates
        lines = output.split('\n')

        for line in lines:
            # Only capture lines that actually show a FAILED or ERROR test
            if ('FAILED' in line or 'ERROR' in line) and '::' in line and 'test_' in line:
                # Extract the test path (everything between start and FAILED/ERROR)
                # This handles both formats:
                # - tests/unit/test_core/test.py::TestClass::test_method FAILED [50%]
                # - FAILED tests/unit/test_core/test.py::TestClass::test_method
                if 'FAILED' in line:
                    test_path = line.split('FAILED')[0].strip()
                elif 'ERROR' in line:
                    test_path = line.split('ERROR')[0].strip()
                else:
                    continue

                # Remove percentage if present
                if '[' in test_path and '%]' in line:
                    test_path = test_path.split('[')[0].strip()

                # Clean up test path - remove leading FAILED/ERROR if present
                test_path = test_path.replace('FAILED ', '').replace('ERROR ', '').strip()

                # Only add if it looks like a valid test path
                if test_path and '::' in test_path and 'test_' in test_path:
                    failures_set.add(test_path)

        failures = list(failures_set)  # Convert back to list
        total_count = len(failures)
        limited_list = failures[:20]  # Return up to 20 failures for display
        return limited_list, total_count

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

    def identify_slow_tests(self, module: str, timeout: int = 30) -> Dict[str, Any]:
        """Run tests with durations to identify slow/hanging tests"""
        test_dir = TEST_DIRECTORIES.get(module)
        if not test_dir:
            return {"slowest": [], "error": f"No test directory for module {module}"}

        cmd = f"python -m pytest {test_dir} --durations=10 --tb=no -q --maxfail=1"
        print(f"🔍 Identifying slow tests: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=timeout
            )
            # Parse durations from output
            return self._parse_test_durations(result.stdout)
        except subprocess.TimeoutExpired:
            return {
                "hanging_tests": ["UNKNOWN - entire suite hung during duration analysis"],
                "slowest": [],
                "timeout": True
            }
        except Exception as e:
            return {
                "slowest": [],
                "error": f"Failed to identify slow tests: {str(e)}"
            }

    def _parse_test_durations(self, output: str) -> Dict[str, Any]:
        """Parse pytest --durations output to extract slow tests"""
        lines = output.split('\n')
        slowest = []
        parsing_durations = False

        for line in lines:
            # Start parsing after the durations header
            if "slowest" in line and "durations" in line:
                parsing_durations = True
                continue

            # Stop when we hit the summary line
            if parsing_durations and ("passed" in line or "failed" in line):
                break

            # Parse duration lines: "0.05s teardown tests/unit/..."
            if parsing_durations and line.strip():
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].endswith('s'):
                    duration = parts[0]  # "0.05s"
                    phase = parts[1]     # "teardown", "call", "setup"
                    # "tests/unit/test_monitoring/test_alerting.py::TestAlertRule::test_alert_rule_creation"
                    full_test_path = parts[2]

                    # Extract just the test method name
                    test_name = full_test_path.split("::")[-1]  # "test_alert_rule_creation"

                    slowest.append({
                        "test": test_name,
                        "duration": duration,
                        "phase": phase,
                        "full_path": full_test_path
                    })

        return {"slowest": slowest}

    def _format_slow_tests_for_prompt(self, slow_test_info: Dict[str, Any]) -> str:
        """Format slow test info for Claude prompt"""
        if slow_test_info.get("hanging_tests"):
            return f"HANGING TESTS: {', '.join(slow_test_info['hanging_tests'])}"

        if slow_test_info.get("timeout"):
            return "TIMEOUT: Unable to identify specific slow tests - entire test suite hung"

        if slow_test_info.get("error"):
            return f"ERROR: {slow_test_info['error']}"

        slowest = slow_test_info.get("slowest", [])
        if not slowest:
            return "ALL TESTS - Unable to identify specific slow tests"

        # Filter for significant slow tests (>3s)
        significant_slow = [
            test for test in slowest
            if float(test["duration"].replace('s', '')) > 3.0
        ]

        if not significant_slow:
            return "NO SIGNIFICANT SLOW TESTS - All tests complete in <3s"

        formatted = "SLOWEST TESTS (>3s):\n"
        for test in significant_slow:
            formatted += f"- {test['test']}: {test['duration']} ({test['phase']})\n"
            formatted += f"  Path: {test['full_path']}\n"

        return formatted

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
        self.current_module = None  # Track current module being processed
        self.current_milestone = 0  # Track current milestone
        self._calculate_total_prompts()

        # Calculate total modules to process for global progress
        self.total_modules = len([m for m in MODULE_HIERARCHY if m not in SKIP_MODULES])
        self.modules_to_process = [m for m in MODULE_HIERARCHY
                                   if m not in SKIP_MODULES
                                   and not self.progress.is_module_complete(m)]

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

    def _log_step(self, action: str, module: str = "unknown", phase: str = "PROCESSING", milestone_progress: float = None):
        """Log current step with global progress indicator"""
        self.current_step += 1

        # Calculate global progress based on all modules
        # Each module has 3 milestones (0=fixes, 1=tests, 2=integration)
        completed_modules = len([m for m in MODULE_HIERARCHY
                                if m not in SKIP_MODULES
                                and self.progress.is_module_complete(m)])

        # milestone_progress is now passed as 0, 1, or 2 directly
        milestone = milestone_progress if milestone_progress is not None else 0

        # Apply the formula: ((completed_modules * 3) + milestone) / (total_modules * 3) * 100
        if self.total_modules > 0:
            progress_pct = ((completed_modules * 3) + milestone) / \
                (self.total_modules * 3) * 100
        else:
            progress_pct = 0

        # Clean, concise output - use newline instead of carriage return for parallel safety
        print(f"🔧 {module.upper()} | {phase} | {action} | {progress_pct:.1f}%")

        # Log to detailed logger only
        if detailed_logger:
            detailed_logger.log_step(module, self.current_step, self.total_steps, action)

    def _initialize_session(self, module: str, milestone: int):
        """Initialize a new Claude session for this milestone"""
        print(f"📝 Initializing session for {module} milestone {milestone}")

        # Simple command to get session ID - run directly without going through execute_prompt
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--model", "claude-sonnet-4-20250514",
            "-p", "Are you available? Please respond with yes.",
            "--output-format", "json"
        ]

        print(f"🔍 Running direct session init command")
        if DRY_RUN_MODE:
            print(f"🔍 Skipping session init in dry-run mode")
            return True

        try:
            # Run the command directly without all the complexity
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30  # 30 second timeout for init
            )

            if result.returncode == 0 and result.stdout:
                # Parse JSON response
                try:
                    response_json = json.loads(result.stdout)
                    session_id = response_json.get('session_id')

                    if session_id:
                        # Save the session ID with lock protection
                        session_key = f"{module}_{milestone}"
                        with self.claude.session_lock:
                            self.claude.session_per_milestone[session_key] = session_id
                        print(
                            f"✅ Session {session_id[:8]}... initialized for {module} milestone {milestone}")

                        # Log session ID
                        if detailed_logger:
                            detailed_logger.log_session_created(session_key, session_id)

                        # Save to progress.json
                        self.claude._save_session_to_progress(session_key, session_id)

                        return True
                    else:
                        print(f"⚠️ No session_id in response")

                except json.JSONDecodeError as e:
                    print(f"⚠️ Failed to parse JSON: {e}")
                    print(f"Response: {result.stdout[:200]}")
            else:
                print(f"⚠️ Session init failed: return code {result.returncode}")
                if result.stderr:
                    print(f"⚠️ Error: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print(f"⚠️ Session initialization timed out")
        except Exception as e:
            print(f"⚠️ Error initializing session: {e}")

        print(f"⚠️ Proceeding without session management")
        return False

    def process_module(self, module: str) -> bool:
        """Process a module with granular fixes and cascade checks"""

        # Skip if already complete
        if self.progress.is_module_complete(module):
            print(f"\n✅ {module.upper()} | SKIPPED | Already completed")
            return True

        # Create thread-local TestRunner to avoid race conditions in parallel execution
        test_runner = TestRunner()

        # Set current module for this processor instance
        self.current_module = module

        # Initialize step tracking for this module
        self.current_step = 0
        self.total_steps = self._calculate_module_steps(module)

        # Track milestones (3 major phases)
        # Using 0-2 scale for milestones to align with global progress calculation
        MILESTONE_FIXES = 0         # Milestone 0: Applying fixes
        MILESTONE_TESTS = 1         # Milestone 1: Running tests
        MILESTONE_INTEGRATION = 2   # Milestone 2: Integration checks

        self.current_milestone = MILESTONE_FIXES

        # Clean module start indicator
        print(f"\n📦 {module.upper()} | STARTING | 3 milestones")

        # Initialize session for this milestone if needed
        if not self.claude.has_session_for_context(module, self.current_milestone):
            self._initialize_session(module, self.current_milestone)

        # Log module start
        if detailed_logger:
            detailed_logger.log_module_start(module, self.total_steps)

        # Update progress
        self._log_step("Initializing", module, "SETUP", self.current_milestone)
        self.progress.update_module(module, status="in_progress")

        # Get baseline test results
        # COMMENTED OUT: Baseline tests to speed up the process
        # self._log_step("Baseline tests", module, "TESTING")
        # baseline = self.test_runner.run_tests(module)
        # self.cascade_protector.update_baseline(module, baseline)

        try:
            module_progress = self.progress.get_module_progress(module)

            # Process with granular prompts and verification after each
            # If MAX_ITERATIONS_PER_MODULE = 0, iterate indefinitely until tests pass
            iteration = module_progress.iteration
            consecutive_api_failures = 0
            max_consecutive_failures = 10  # Safety: exit after 5 consecutive API failures

            while True:
                # Check iteration limit (if not infinite)
                if MAX_ITERATIONS_PER_MODULE > 0 and iteration >= MAX_ITERATIONS_PER_MODULE:
                    break

                # Safety check: prevent infinite loops due to persistent API failures
                if MAX_ITERATIONS_PER_MODULE == 0 and consecutive_api_failures >= max_consecutive_failures:
                    print(f"\n❌ {module.upper()} | FAILED | Too many consecutive API failures")
                    self.progress.update_module(module, status="failed")
                    return False
                # Display iteration info with milestone progress
                if MAX_ITERATIONS_PER_MODULE > 0:
                    self._log_step(
                        f"Iteration {iteration + 1}/{MAX_ITERATIONS_PER_MODULE}", module, "FIXING", self.current_milestone)
                else:
                    self._log_step(f"Iteration {iteration + 1}",
                                   module, "FIXING", self.current_milestone)
                self.progress.update_module(module, iteration=iteration + 1)

                # DO NOT clear completed prompts - we need to track what's already been done
                # Clearing prompts_completed was causing infinite loops by re-running already completed fixes
                # Each prompt should only be executed once per module

                # Granular fix sequence - run all prompts then test
                fix_sequence = [
                    # Basic fixes first - get tests running
                    (PromptType.IMPORTS_ONLY, "Fixing imports"),
                    (PromptType.TYPES_ONLY, "Fixing types"),
                    (PromptType.ASYNC_ONLY, "Fixing async/await"),
                    (PromptType.RESOURCES_ONLY, "Fixing resources"),
                    (PromptType.ERRORS_ONLY, "Fixing error handling"),
                    # Test quality verification early - catch issues sooner
                    (PromptType.TEST_VERIFICATION, "Verifying test logic correctness"),
                    (PromptType.TEST_COVERAGE, "Ensuring 70%+ test coverage"),
                    # Trading-specific fixes
                    (PromptType.SERVICE_LAYER, "Fixing service layer"),
                    (PromptType.DATABASE_MODELS, "Fixing database models"),
                    (PromptType.FINANCIAL_PRECISION, "Fixing financial precision"),
                    (PromptType.DEPENDENCY_INJECTION, "Fixing dependency injection"),
                    (PromptType.WEBSOCKET_ASYNC, "Fixing WebSocket/async"),
                    (PromptType.FACTORY_PATTERNS, "Fixing factory patterns"),
                    # Simplification after tests are more stable
                    (PromptType.REMOVE_ENHANCED, "Removing enhanced versions and simplifying"),
                    # Code quality checks
                    (PromptType.CODE_DUPLICATION, "Eliminating code duplication"),
                    (PromptType.CIRCULAR_DEPENDENCIES, "Fixing circular dependencies"),
                    (PromptType.MODULE_INTEGRATION, "Verifying module integration"),
                    # Final cleanup only at the very end
                    (PromptType.FINAL_CLEANUP, "Final production cleanup"),
                ]

                # Check if all prompts are already completed
                all_prompts_completed = all(
                    prompt_type.value in module_progress.prompts_completed
                    for prompt_type, _ in fix_sequence
                )

                # Execute all fix prompts for this iteration (skip if all complete)
                if not all_prompts_completed:
                    for prompt_type, description in fix_sequence:
                        if prompt_type.value in module_progress.prompts_completed:
                            continue

                        # Show both the description and the prompt type for clarity
                        prompt_display = f"{description} [{prompt_type.value}]"
                        self._log_step(prompt_display, module, "FIXING", self.current_milestone)

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

                        success, output = self.claude.execute_prompt(
                            prompt, description, module=module, milestone=self.current_milestone)

                        if success:
                            consecutive_api_failures = 0  # Reset failure counter on success
                            module_progress.prompts_completed.append(prompt_type.value)
                            self.progress.update_module(
                                module,
                                prompts_completed=module_progress.prompts_completed
                            )

                            # Special handling for FINAL_CLEANUP - run verification tests
                            if prompt_type == PromptType.FINAL_CLEANUP:
                                self._log_step("Final verification after cleanup",
                                               module, "TESTING", self.current_milestone)
                                cleanup_test_results = test_runner.run_tests(module)
                                self.progress.update_module(
                                    module, test_results=cleanup_test_results)

                                cleanup_stats = cleanup_test_results.get('stats', {})

                                # If cleanup introduced issues, mark cleanup as incomplete and continue iteration
                                if (cleanup_stats.get('failed', 0) > 0 or
                                        len(cleanup_test_results.get('warnings', [])) > 0):
                                    print(
                                        f"\n⚠️  {module.upper()} | Cleanup introduced issues, removing from completed")
                                    # Remove cleanup from completed prompts so it can be re-run
                                    module_progress.prompts_completed.remove(
                                        PromptType.FINAL_CLEANUP.value)
                                    self.progress.update_module(
                                        module,
                                        prompts_completed=module_progress.prompts_completed
                                    )
                                else:
                                    print(
                                        f"\n✅ {module.upper()} | Production cleanup verified successfully")
                        else:
                            consecutive_api_failures += 1

                # Full test run - Move to TESTING milestone (only if not already tested in this iteration)
                self.current_milestone = MILESTONE_TESTS

                # Check if we need to run tests (avoid duplicate runs)
                needs_test_run = True

                # Get current test results to check if they're fresh from this iteration
                current_test_results = self.progress.get_module_progress(module).test_results

                # If we have recent test results and they were updated this iteration, use them
                if current_test_results and current_test_results.get('_test_iteration') == iteration:
                    self._log_step("Using recent test results", module,
                                   "TESTING", self.current_milestone)
                    test_results = current_test_results
                    needs_test_run = False

                if needs_test_run:
                    self._log_step("Running tests", module, "TESTING", self.current_milestone)
                    test_results = test_runner.run_tests(module)
                    # Mark these results with current iteration to avoid re-running
                    test_results['_test_iteration'] = iteration
                    self.progress.update_module(module, test_results=test_results)

                # Check for collection errors first (before trying to get stats)
                if test_results.get('collection_error'):
                    print(f"\n❌ {module.upper()} | Test collection failed - fixing import errors")

                    # Use COLLECTION_ERRORS prompt to fix import issues
                    collection_errors = test_results.get(
                        'stdout', '') + test_results.get('stderr', '')
                    collection_errors_prompt = MODULE_PROMPTS[PromptType.COLLECTION_ERRORS].format(
                        test_dir=TEST_DIRECTORIES.get(module),
                        collection_errors=collection_errors[:2000],
                        anti_hallucination="{anti_hallucination}"
                    )

                    success, output = self.claude.execute_prompt(
                        collection_errors_prompt,
                        "Fixing test collection errors",
                        module=module,
                        milestone=self.current_milestone
                    )

                    if success:
                        # Mark that we've tried to fix collection errors
                        if PromptType.COLLECTION_ERRORS.value not in module_progress.prompts_completed:
                            module_progress.prompts_completed.append(
                                PromptType.COLLECTION_ERRORS.value)
                            self.progress.update_module(
                                module,
                                prompts_completed=module_progress.prompts_completed
                            )

                        # Re-run tests after fixing
                        test_results = test_runner.run_tests(module)
                        test_results['_test_iteration'] = iteration
                        self.progress.update_module(module, test_results=test_results)

                    # Continue iteration regardless
                    iteration += 1
                    continue

                # Log test results summary
                stats = test_results.get('stats', {})
                print(f"\n📊 Test Results:")
                print(f"   - ✅ Passed: {stats.get('passed', 0)}")
                print(f"   - ❌ Failed: {stats.get('failed', 0)}")
                print(f"   - ⏭️ Skipped: {stats.get('skipped', 0)}")
                print(f"   - ⚠️ Warnings: {len(test_results.get('warnings', []))}")

                # Determine test status - HANDLE ALL EDGE CASES
                # First, check for timeout (tests hanging)
                if test_results.get('timeout'):
                    print(f"\n⚠️  {module.upper()} | Tests timed out after {TEST_TIMEOUT_PER_MODULE}s")

                    # Check if it's actually a collection error by running --collect-only
                    test_dir = TEST_DIRECTORIES.get(module)
                    collect_cmd = f"python -m pytest {test_dir} --collect-only -q 2>&1"
                    print(f"🔍 Checking if timeout is due to collection errors...")

                    try:
                        collect_result = subprocess.run(
                            collect_cmd, shell=True, capture_output=True,
                            text=True, timeout=10
                        )

                        # Check for collection errors in output (check both stdout and stderr)
                        output = collect_result.stdout + collect_result.stderr
                        if "ERROR collecting" in output or \
                           "ModuleNotFoundError" in output or \
                           "ImportError" in output or \
                           "AttributeError" in output:

                            print(f"\n❌ {module.upper()} | Collection errors detected - fixing imports")

                            # Use COLLECTION_ERRORS prompt instead of SLOW_TESTS
                            collection_errors_prompt = MODULE_PROMPTS[PromptType.COLLECTION_ERRORS].format(
                                test_dir=test_dir,
                                # First 2000 chars of errors from both stdout and stderr
                                collection_errors=output[:2000],
                                anti_hallucination="{anti_hallucination}"
                            )

                            success, claude_response = self.claude.execute_prompt(
                                collection_errors_prompt,
                                "Fixing test collection errors",
                                module=module,
                                milestone=self.current_milestone
                            )

                            if success:
                                module_progress.prompts_completed.append(
                                    PromptType.COLLECTION_ERRORS.value)
                                self.progress.update_module(
                                    module,
                                    prompts_completed=module_progress.prompts_completed
                                )
                        else:
                            # Real timeout - tests are actually slow/hanging
                            print(
                                f"\n⚠️  {module.upper()} | Tests are genuinely slow/hanging - identifying specific slow tests")

                            # NEW: Identify specific slow tests
                            slow_test_info = test_runner.identify_slow_tests(module, timeout=30)

                            # Record hanging tests in progress
                            self.progress.update_module(module, hanging_tests=slow_test_info)

                            # Create detailed slow tests prompt
                            slow_tests_details = test_runner._format_slow_tests_for_prompt(
                                slow_test_info)

                            slow_tests_prompt = MODULE_PROMPTS[PromptType.SLOW_TESTS].format(
                                test_dir=test_dir,
                                module=module,
                                slow_tests=slow_tests_details,  # SPECIFIC instead of "ALL TESTS"
                                timeout=TEST_TIMEOUT_PER_TEST,
                                anti_hallucination="{anti_hallucination}"
                            )

                            success, output = self.claude.execute_prompt(
                                slow_tests_prompt,
                                "Fixing slow/hanging tests",
                                module=module,
                                milestone=self.current_milestone
                            )

                    except subprocess.TimeoutExpired:
                        # Even collect-only timed out - likely infinite loop in imports
                        print(f"\n❌ {module.upper()} | Even collection timed out - major import issues")
                        collection_errors_prompt = MODULE_PROMPTS[PromptType.COLLECTION_ERRORS].format(
                            test_dir=test_dir,
                            collection_errors="Collection phase timed out - likely infinite loop in imports or module-level code",
                            anti_hallucination="{anti_hallucination}"
                        )

                        success, output = self.claude.execute_prompt(
                            collection_errors_prompt,
                            "Fixing import timeout issues",
                            module=module,
                            milestone=self.current_milestone
                        )

                    # Re-run tests after fixing
                    if success:
                        test_results = test_runner.run_tests(module)
                        test_results['_test_iteration'] = iteration
                        self.progress.update_module(module, test_results=test_results)
                        stats = test_results.get('stats', {})

                    # Continue iteration regardless
                    iteration += 1
                    continue

                # Check if stats exist (missing stats = import error)
                elif 'stats' not in test_results and not test_results.get('success', True):
                    print(f"\n❌ {module.upper()} | Test collection failed - likely import errors")
                    # Continue iteration to fix import issues
                    iteration += 1
                    continue

                # Now we have stats, check various conditions
                has_errors = stats.get('errors', 0) > 0
                has_failures = stats.get('failed', 0) > 0
                has_warnings = len(test_results.get('warnings', [])) > 0
                has_skipped = len(test_results.get('skipped', [])) > 0
                total_tests = stats.get('total', 0)

                # Check for test ERRORS (not just failures)
                if has_errors:
                    print(f"\n❌ {module.upper()} | {stats.get('errors', 0)} test ERRORS detected")
                    # Treat errors like failures - they need fixing
                    test_results['failures'] = test_results.get('failures', [])
                    # Add errors to failures list for processing
                    for i in range(stats.get('errors', 0)):
                        test_results['failures'].append({
                            'test': f'error_{i}',
                            'error': 'Test error (not failure) - check test collection'
                        })

                # Check if no tests exist (very rare - usually means directory issue)
                if total_tests == 0:
                    print(
                        f"\n⚠️  {module.upper()} | No tests found - likely collection error not caught")

                    # This is likely a collection error that wasn't caught
                    # Try running with --collect-only to get the actual error
                    test_dir = TEST_DIRECTORIES.get(module)
                    collect_cmd = f"python -m pytest {test_dir} --collect-only -q 2>&1"

                    try:
                        collect_result = subprocess.run(
                            collect_cmd, shell=True, capture_output=True,
                            text=True, timeout=10
                        )

                        if "ERROR collecting" in collect_result.stdout or \
                           "ERROR collecting" in collect_result.stderr:
                            print(f"\n❌ {module.upper()} | Collection errors detected")
                            # Will be handled by COLLECTION_ERRORS prompt on next iteration
                            iteration += 1
                            continue
                    except:
                        pass

                    # If we've tried many times and still have 0 tests, fail the module
                    if iteration >= 10:
                        print(f"\n❌ {module.upper()} | No tests found after {iteration} attempts")
                        self.progress.update_module(module, status="failed",
                                                    errors=[f"No tests found after {iteration} attempts - check test directory"])
                        return False

                    iteration += 1
                    continue

                # Now determine if tests actually passed (no failures AND no errors)
                tests_passed = not has_failures and not has_errors and total_tests > 0

                # Check completion conditions
                if tests_passed and not has_warnings and not has_skipped:
                    # Check if ALL prompts including FINAL_CLEANUP are completed
                    all_prompts_completed = all(
                        prompt_type.value in module_progress.prompts_completed
                        for prompt_type, _ in fix_sequence
                    )

                    if all_prompts_completed:
                        # Move to INTEGRATION milestone when tests pass and ALL prompts complete
                        self.current_milestone = MILESTONE_INTEGRATION
                        print(
                            f"\n✅ {module.upper()} | TESTS PASSING & ALL PROMPTS COMPLETE | Moving to integration")
                    else:
                        print(
                            f"\n⚠️  {module.upper()} | Tests pass but missing prompts, continuing iteration")
                        iteration += 1
                        continue

                    # Handle slow tests and cascade checks silently
                    if test_results.get("slow_tests"):
                        self._log_step("Optimizing slow tests", module,
                                       "OPTIMIZING", self.current_milestone)
                        slow_tests_str = "\n".join(test_results["slow_tests"])
                        prompt = MODULE_PROMPTS[PromptType.SLOW_TESTS].format(
                            test_dir=TEST_DIRECTORIES.get(module),
                            module=module,
                            slow_tests=slow_tests_str,
                            timeout=TEST_TIMEOUT_PER_TEST,
                            anti_hallucination="{anti_hallucination}"
                        )
                        self.claude.execute_prompt(
                            prompt, "Optimizing slow tests", module=module, milestone=self.current_milestone)

                    # Final cascade check
                    if CASCADE_CHECK_ENABLED:
                        self._log_step("Cascade check", module,
                                       "VALIDATING", self.current_milestone)
                        cascade_results = self.cascade_protector.check_dependents(module)
                        for dependent, ok in cascade_results:
                            if not ok:
                                if detailed_logger:
                                    detailed_logger.log_cascade_check(module, dependent, False)

                                # Request feedback on cascade failure
                                self._log_step(
                                    f"Analyzing cascade: {dependent}", module, "FEEDBACK", self.current_milestone)
                                feedback_success, feedback = self.claude.request_feedback(
                                    module,
                                    f"Cascade failure detected: {module} changes broke {dependent}. Should we proceed with fixes or rollback?",
                                    "system-design-architect",
                                    milestone=self.current_milestone
                                )

                                if feedback_success:
                                    if "ROLLBACK" in feedback.upper():
                                        print(
                                            f"\n🤖 {module.upper()} | FEEDBACK | Rollback recommended for cascade failure")
                                        # Could implement rollback logic here if needed
                                    else:
                                        print(
                                            f"\n🤖 {module.upper()} | FEEDBACK | Proceeding with cascade fixes for {dependent}")

                    # Final cleanup is now handled in the main prompt sequence above
                    # All prompts including cleanup have been completed at this point

                    # Commit and push the module to git before marking complete
                    print(f"\n📦 {module.upper()} | Committing to git...")

                    git_commit_prompt = f"""Commit and push the {module} module to git:

1. Stage files for {module} module:
   git add src/{module}/ tests/unit/test_{module}/

2. Create commit with message:
   git commit -m "fix({module}): resolve all test failures and warnings

   - Fixed all test failures  
   - Resolved all warnings
   - Tests passing: {stats.get('passed', 0)}/{stats.get('total', 0)}
   - Module ready for production"

3. Push to remote:
   git push origin main

Execute these commands in order. Only commit files for THIS module.
"""

                    commit_success, commit_output = self.claude.execute_prompt(
                        git_commit_prompt,
                        f"Committing {module} to git",
                        module=module,
                        milestone=self.current_milestone
                    )

                    if commit_success:
                        print(f"✅ {module.upper()} | Committed and pushed to git")
                    else:
                        print(f"⚠️  {module.upper()} | Git commit may have failed - check manually")

                    # Mark as complete
                    self.progress.update_module(module, status="completed")
                    # Calculate final progress after this module completes
                    completed_modules = len([m for m in MODULE_HIERARCHY
                                            if m not in SKIP_MODULES
                                            and self.progress.is_module_complete(m)])
                    final_progress = (completed_modules * 3) / (self.total_modules *
                                                                3) * 100 if self.total_modules > 0 else 100
                    print(f"\n✅ {module.upper()} | COMPLETED | PRODUCTION READY | {final_progress:.1f}%")
                    return True

                # Handle skipped tests (higher priority than warnings)
                elif has_skipped:
                    skipped_count = len(test_results.get('skipped', []))
                    self._log_step(f"Fixing {skipped_count} skipped tests", module, "FIXING")

                    # SIMPLIFIED: Let Claude investigate skipped tests directly
                    prompt = f"""
Run tests for the {module} module and fix skipped tests.

Test directory: tests/unit/test_{module}/
Current status: {skipped_count} skipped tests

Instructions:
1. Run: python -m pytest tests/unit/test_{module}/ -v --tb=short
2. Identify why tests are being skipped (look for skip markers, missing dependencies)
3. Fix the root causes:
   - Missing dependencies: install or mock them
   - Conditional skips: ensure conditions are met
   - Platform-specific: add appropriate handling
4. Run tests again to verify skipped tests now run

IMPORTANT: Make skipped tests actually run and pass.
{{anti_hallucination}}
"""

                    success, output = self.claude.execute_prompt(
                        prompt, "Fixing skipped tests", module=module, milestone=self.current_milestone)

                    if success:
                        # Re-test to check if skipped tests are fixed
                        self._log_step("Verifying skipped test fixes", module, "TESTING")
                        test_results = test_runner.run_tests(module)
                        # Mark results with current iteration to avoid duplicate main test run
                        test_results['_test_iteration'] = iteration
                        self.progress.update_module(module, test_results=test_results)

                        stats = test_results.get('stats', {})
                        new_skipped_count = len(test_results.get('skipped', []))

                        if new_skipped_count < skipped_count:
                            print(
                                f"\n✅ Reduced skipped tests from {skipped_count} to {new_skipped_count}")

                        # If all skipped tests are fixed and tests pass, check warnings next
                        if new_skipped_count == 0 and stats.get('failed', 0) == 0:
                            if len(test_results.get('warnings', [])) == 0:
                                print(f"\n✅ {module.upper()} | All skipped tests fixed, no warnings")
                                # Don't mark as complete yet - need to run cleanup
                                # Continue to next iteration
                                pass
                            else:
                                print(
                                    f"\n⚠️  {module.upper()} | Skipped tests fixed, {len(test_results.get('warnings', []))} warnings remain")

                    # Continue to next iteration
                    iteration += 1
                    continue

                # Handle warnings even if tests pass
                elif tests_passed and has_warnings:
                    warning_count = len(test_results.get('warnings', []))
                    self._log_step(f"Fixing {warning_count} warnings", module, "FIXING")

                    # SIMPLIFIED: Let Claude run tests and see warnings directly
                    prompt = f"""
Run tests for the {module} module and fix the warnings.

Test directory: tests/unit/test_{module}/
Current status: {warning_count} warnings

Instructions:
1. Run: python -m pytest tests/unit/test_{module}/ -v --tb=short
2. Analyze ALL warnings in the output
3. Identify patterns (e.g., if you see 600+ identical warnings, they have ONE root cause)
4. Fix the root causes:
   - For asyncio warnings: check for nest_asyncio conflicts, pytest-asyncio config
   - For deprecation warnings: update to current APIs
   - For import warnings: fix module dependencies
5. Run tests again to verify warnings are reduced/eliminated

IMPORTANT: Fix root causes, not symptoms. Pattern recognition is key.
{{anti_hallucination}}
"""

                    success, output = self.claude.execute_prompt(
                        prompt, "Fixing warnings", module=module, milestone=self.current_milestone)

                    if success:
                        # Re-test to check if warnings are resolved
                        self._log_step("Verifying warning fixes", module, "TESTING")
                        test_results = test_runner.run_tests(module)
                        # Mark results with current iteration to avoid duplicate main test run
                        test_results['_test_iteration'] = iteration
                        self.progress.update_module(module, test_results=test_results)

                        stats = test_results.get('stats', {})
                        new_warning_count = len(test_results.get('warnings', []))

                        if new_warning_count < warning_count:
                            print(
                                f"\n✅ Reduced warnings from {warning_count} to {new_warning_count}")

                        # If all warnings are fixed and tests still pass, continue to cleanup
                        if new_warning_count == 0 and stats.get('failed', 0) == 0:
                            print(f"\n✅ {module.upper()} | All warnings fixed, proceeding to cleanup")
                            # Continue to next iteration which will trigger cleanup
                            # No explicit action needed here

                    # Continue to next iteration to try again or handle failures
                    iteration += 1
                    continue

                # Comprehensive test failure fixing
                elif test_results.get("failures"):
                    failure_count = len(test_results.get("failures", []))
                    error_count = test_results.get('stats', {}).get('errors', 0)

                    self._log_step(
                        f"Fixing {failure_count} failures and {error_count} errors", module, "FIXING")

                    # SIMPLIFIED: Let Claude run tests and see everything
                    prompt = f"""
Run tests for the {module} module and fix all failures and errors.

Test directory: tests/unit/test_{module}/
Current status: {failure_count} failures, {error_count} errors

Instructions:
1. Run: python -m pytest tests/unit/test_{module}/ -xvs --tb=short
2. Analyze the COMPLETE output to understand all issues
3. Fix the root causes of failures:
   - Import errors: check module dependencies
   - Async errors: fix await/async issues, check event loop
   - Mock errors: ensure proper mock configuration
   - Timeout errors: optimize slow code or add timeout handling
4. Run tests again to verify all failures are fixed
5. If issues persist, try alternative approaches

IMPORTANT: 
- See the actual error messages by running tests yourself
- Fix root causes, not symptoms
- Ensure tests actually pass after fixes
{{anti_hallucination}}
"""

                    # Store previous failure count before fixing
                    previous_failed = failure_count

                    success, output = self.claude.execute_prompt(
                        prompt, "Fixing test failures", module=module, milestone=self.current_milestone)

                    if success:
                        consecutive_api_failures = 0  # Reset failure counter on success

                        # CRITICAL: Verify files were actually modified
                        git_diff = subprocess.run(
                            ["git", "diff", "--name-only"], capture_output=True, text=True)
                        if not git_diff.stdout.strip():
                            print(
                                f"⚠️  {module.upper()} | No files were modified by Claude - response may be analysis only")
                            # Track attempts with no actual fixes
                            if not hasattr(module_progress, 'no_fix_attempts'):
                                module_progress.no_fix_attempts = 0
                            module_progress.no_fix_attempts += 1

                            # Exit early if Claude isn't actually fixing anything
                            if module_progress.no_fix_attempts >= 3:
                                print(
                                    f"❌ {module.upper()} | Claude not producing actual code fixes after 3 attempts")
                                self.progress.update_module(module, status="failed",
                                                            errors=["Claude responses contain no actual code changes"])
                                return False
                        else:
                            modified_files = git_diff.stdout.strip().split('\n')
                            print(
                                f"✅ {module.upper()} | Files modified: {len(modified_files)} files changed")
                            module_progress.no_fix_attempts = 0  # Reset counter

                        # Re-test to check if issues are resolved
                        self._log_step("Verifying fixes", module, "TESTING")
                        test_results = test_runner.run_tests(module)
                        # Mark results with current iteration to avoid duplicate main test run
                        test_results['_test_iteration'] = iteration
                        self.progress.update_module(module, test_results=test_results)

                        stats = test_results.get('stats', {})
                        new_failed = stats.get('failed', 0)

                        # Check if we're making progress
                        if previous_failed > 0 and new_failed >= previous_failed:
                            if not hasattr(module_progress, 'no_improvement_count'):
                                module_progress.no_improvement_count = 0
                            module_progress.no_improvement_count += 1
                            print(
                                f"⚠️  {module.upper()} | No improvement: {new_failed} failures (was {previous_failed})")

                            # Exit if no improvement after multiple attempts
                            if module_progress.no_improvement_count >= 3:
                                print(f"❌ {module.upper()} | No improvement after 3 fix attempts")
                                self.progress.update_module(module, status="failed",
                                                            errors=[f"Stuck at {new_failed} failures after multiple attempts"])
                                return False
                        elif previous_failed > new_failed:
                            module_progress.no_improvement_count = 0  # Reset if we see improvement
                            print(
                                f"✅ {module.upper()} | Improvement: {previous_failed} → {new_failed} failures")

                        stats = test_results.get('stats', {})
                        # Check if tests pass AND no warnings remain
                        if (stats.get('failed', 0) == 0 and
                            len(test_results.get('warnings', [])) == 0 and
                                stats.get('total', 0) > 0):
                            print(f"\n✅ {module.upper()} | All failures and warnings fixed")
                            # Don't mark as complete yet - need to run cleanup first
                            # Continue to next iteration which will trigger the cleanup phase
                            pass
                        # If tests pass but warnings exist, continue to handle them
                        elif stats.get('failed', 0) == 0 and len(test_results.get('warnings', [])) > 0:
                            print(
                                f"\n⚠️  {module.upper()} | Tests passing but {len(test_results.get('warnings', []))} warnings remain")
                            # Continue to next iteration to handle warnings
                            pass
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
                    "quality-control-enforcer",
                    milestone=self.current_milestone
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

    def check_integration(self, module: str, dependency: str, milestone: int = 2) -> bool:
        """Check integration between module and dependency

        Args:
            module: The module to check
            dependency: The dependency module
            milestone: The milestone number (default 2 for integration phase)
        """

        if not self.progress.is_module_complete(module):
            return False

        if not self.progress.is_module_complete(dependency):
            return False

        # Check if integration already completed
        if self.progress.is_integration_complete(module, dependency):
            print(f"✅ {module.upper()} → {dependency.upper()} | Already integrated")
            return True

        print(f"🔗 {module.upper()} → {dependency.upper()} | INTEGRATING")

        # Request feedback before integration checks
        feedback_success, feedback = self.claude.request_feedback(
            module,
            f"About to run integration checks between {module} and {dependency}. Are there known integration issues or conflicts?",
            "integration-architect",
            milestone=milestone
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
            f"Verifying {module} uses {dependency} correctly",
            module=module,
            milestone=milestone
        )

        # 2. Check cross-module integration
        prompt = INTEGRATION_PROMPTS["check_cross_module"].format(
            module=module,
            dependency=dependency,
            anti_hallucination="{anti_hallucination}"
        )
        self.claude.execute_prompt(
            prompt,
            f"Checking cross-module integration between {module} and {dependency}",
            module=module,
            milestone=milestone
        )

        # 3. Fix service layer violations
        prompt = INTEGRATION_PROMPTS["fix_service_layer"].format(
            module=module,
            anti_hallucination="{anti_hallucination}"
        )
        self.claude.execute_prompt(
            prompt,
            f"Fixing service layer violations in {module}",
            module=module,
            milestone=milestone
        )

        # 4. Align data flow patterns
        prompt = INTEGRATION_PROMPTS["align_data_flow"].format(
            module=module,
            dependency=dependency,
            anti_hallucination="{anti_hallucination}"
        )
        self.claude.execute_prompt(
            prompt,
            f"Aligning data flow between {module} and {dependency}",
            module=module,
            milestone=milestone
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
                    f"Ensuring {module} doesn't break {dependent}",
                    module=module,
                    milestone=milestone
                )

        print(f"✅ Integration verified: {module} → {dependency}")

        # Mark this integration as complete
        self.progress.mark_integration_complete(module, dependency)

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
        self.claude = ClaudeExecutor(
            self.rate_limiter, self.hallucination_detector, dry_run=DRY_RUN_MODE)
        self.test_runner = TestRunner()

        print("📁 Loading progress...")
        self.progress = ProgressTracker(dry_run=DRY_RUN_MODE)

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
                # Only check A→B when processing module A (and B is already complete)
                deps = MODULE_DEPENDENCIES.get(module, [])
                for dep in deps:
                    if dep not in SKIP_MODULES and self.progress.is_module_complete(dep):
                        self.integration_checker.check_integration(module, dep, milestone=2)
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
                            self.integration_checker.check_integration(module, dep, milestone=2)

                    # Check integration with other modules in the same group
                    for other in group:
                        if other != module and self.progress.is_module_complete(other):
                            # Check if they have dependencies on each other
                            if other in deps:
                                self.integration_checker.check_integration(
                                    module, other, milestone=2)
                            other_deps = MODULE_DEPENDENCIES.get(other, [])
                            if module in other_deps:
                                self.integration_checker.check_integration(
                                    other, module, milestone=2)

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
