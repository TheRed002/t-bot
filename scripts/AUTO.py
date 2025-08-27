#!/usr/bin/env python3
"""
Module Integration Checker - Systematic module and dependency validation.
First checks each module's internal implementation, then verifies all its integrations.
"""

import subprocess
import sys
import time
import argparse

# Module dependency hierarchy (order matters - from lowest to highest level)
MODULE_HIERARCHY = [
    "core",           # Base module - no dependencies
    "utils",          # Depends on: core
    "error_handling",  # Depends on: core, utils
    "database",       # Depends on: core, utils, error_handling
    "state",          # Depends on: core, utils, error_handling, database
    "monitoring",     # Depends on: core, utils, error_handling, database
    "exchanges",      # Depends on: core, utils, error_handling, database, monitoring
    "risk_management",  # Depends on: core, utils, error_handling, database, monitoring
    "execution",      # Depends on: core, utils, error_handling, database, exchanges, risk_management
    "data",           # Depends on: core, utils, error_handling, database, monitoring
    "ml",             # Depends on: core, utils, error_handling, database, data
    "strategies",     # Depends on: core, utils, error_handling, database, risk_management, execution
    "backtesting",    # Depends on: core, utils, error_handling, database, strategies, execution
    "capital_management",  # Depends on: core, utils, error_handling, database, risk_management
    "bot_management",  # Depends on: core, utils, error_handling, database, strategies, execution, capital_management
    "web_interface",  # Depends on: core, utils, error_handling, database, bot_management
]

# Modules to skip (already verified or to be checked later)
# Add module names here to skip them during checking
SKIP_MODULES = [
    # "core", "utils"  # Example: Uncomment to skip core and utils modules
    # Add modules you want to skip here
    "core",
    "utils",
    "error_handling",
    "database",
    "state",
    "monitoring",
    "exchanges",
    "risk_management",
    "execution"
]

# You can also skip specific integrations
# Format: ("module", "dependency")
SKIP_INTEGRATIONS = [
    # ("database", "core"), ("database", "utils")  # Example
    # Add specific integrations you want to skip
    ("utils", "core"),
    ("error_handling", "core"), ("error_handling", "utils"),
    ("database", "core"), ("database", "utils"), ("database", "error_handling"),
    ("state", "core"), ("state", "utils"), ("state",
                                            "error_handling"), ("state", "database"), ("state", "monitoring"),
    ("monitoring", "core"), ("monitoring", "utils"), ("monitoring",
                                                      "error_handling"), ("monitoring", "database"),
    ("exchanges", "core"), ("exchanges", "utils"), ("exchanges", "error_handling"), ("exchanges",
                                                                                     "database"), ("exchanges", "monitoring"), ("exchanges", "state"),
    ("risk_management", "core"), ("risk_management", "utils"), ("risk_management", "error_handling"), (
        "risk_management", "database"), ("risk_management", "monitoring"), ("risk_management", "state"),
    ("execution", "core"), ("execution", "utils"), ("execution", "error_handling"), ("execution", "database"), ("execution",
                                                                                                                "exchanges"), ("execution", "risk_management"), ("execution", "monitoring"), ("execution", "state"),
]

# Define which modules each module depends on
MODULE_DEPENDENCIES = {
    "core": [],
    "utils": ["core"],
    "error_handling": ["core", "utils"],
    "database": ["core", "utils", "error_handling"],
    "state": ["core", "utils", "error_handling", "database", "monitoring"],
    "monitoring": ["core", "utils", "error_handling", "database"],
    "exchanges": ["core", "utils", "error_handling", "database", "monitoring", "state"],
    "data": ["core", "utils", "error_handling", "database", "monitoring"],
    "risk_management": ["core", "utils", "error_handling", "database", "monitoring", "state"],
    "execution": ["core", "utils", "error_handling", "database", "exchanges", "risk_management", "monitoring", "state"],
    "ml": ["core", "utils", "error_handling", "database", "data"],
    "strategies": ["core", "utils", "error_handling", "database", "risk_management", "execution", "data", "monitoring"],
    "backtesting": ["core", "utils", "error_handling", "database", "strategies", "execution", "data"],
    "capital_management": ["core", "utils", "error_handling", "database", "risk_management", "state"],
    "bot_management": ["core", "utils", "error_handling", "database", "strategies", "execution", "capital_management", "monitoring", "state", "risk_management"],
    "web_interface": ["core", "utils", "error_handling", "database", "bot_management", "monitoring", "state"],
}

# Prompt template for checking module implementation V2
MODULE_CHECK_TEMPLATE_V2 = """
CRITICAL: Fix ALL backend issues in {module} module to ensure 100% test pass rate with ZERO errors/warnings.

**PHASE 1: COMPREHENSIVE ISSUE SCANNING**
Use specialized agents to detect ALL issues:

1. **SYNTAX & IMPORTS** (Task: performance-optimization-specialist)
   - Missing/incorrect imports, circular dependencies
   - Undefined variables, typos in function names
   - Import order violations (stdlib → third-party → local)
   - Unused imports that cause warnings

2. **TYPE SAFETY** (Task: financial-qa-engineer)
   - Missing/incorrect type hints causing mypy errors
   - Type mismatches in function signatures
   - Generic types not properly parameterized
   - Optional/Union types not handled correctly
   - Return type mismatches

3. **ASYNC/AWAIT PATTERNS** (Task: financial-api-architect)
   - Missing await keywords causing coroutine warnings
   - Blocking I/O in async functions
   - Improper async context manager usage
   - Race conditions in concurrent operations
   - Deadlocks from circular awaits

4. **RESOURCE MANAGEMENT** (Task: infrastructure-wizard)
   - Database connections not closed properly
   - File handles left open
   - Memory leaks from unclosed streams
   - WebSocket connections not cleaned up
   - Thread/process pools not terminated

5. **BUSINESS LOGIC** (Task: algo-trading-specialist)
   - Incorrect financial calculations (fees, PnL, position sizing)
   - Wrong decimal precision for prices/quantities
   - Missing boundary checks (min order size, max leverage)
   - Incorrect order state transitions
   - Missing risk validation checks

6. **ERROR HANDLING** (Task: quality-control-enforcer)
   - Unhandled exceptions causing test failures
   - Missing try/except blocks
   - Catching too broad exceptions (bare except)
   - Not re-raising critical errors
   - Error messages not properly formatted

7. **TEST COMPATIBILITY** (Task: integration-test-architect)
   - Mock objects not matching real interfaces
   - Test fixtures with incorrect data types
   - Missing test database cleanup
   - Hardcoded test values that break
   - Async tests not properly awaited

8. **LOGGING & WARNINGS** (Task: financial-docs-writer)
   - Logger not properly initialized
   - Sensitive data in log messages
   - Incorrect log levels causing test noise
   - Deprecation warnings from libraries

**PHASE 2: SYSTEMATIC FIXING**

1. **Priority Order**:
   - CRITICAL: Causes test failures/crashes
   - HIGH: Causes warnings or flaky tests
   - MEDIUM: Code quality issues that affect maintainability

2. **Fix Strategy**:
   - Use Task tool with appropriate specialized agent for each issue type
   - Fix imports and types first (foundation)
   - Then async/await patterns
   - Then resource management
   - Then business logic
   - Finally error handling and logging

3. **Validation After Each Fix**:
   ```bash
   # Run module-specific tests
   python -m pytest tests/unit/test_{module}/ -xvs --tb=short
   python -m pytest tests/integration/*{module}* -xvs --tb=short
   
   # Check for warnings
   python -m pytest tests/unit/test_{module}/ -W error
   
   # Type checking
   mypy src/{module}/ --ignore-missing-imports --strict
   
   # Linting
   ruff check src/{module}/ --fix
   ruff format src/{module}/
   ```

**PHASE 3: COMPREHENSIVE TESTING**

After all fixes, run complete test suite regarding the module:
```bash
# All unit tests for module
python -m pytest tests/unit/test_{module}/ -v --tb=short --no-warnings

# All integration tests involving module
python -m pytest tests/integration/ -k {module} -v --tb=short

# Coverage check
python -m pytest tests/unit/test_{module}/ --cov=src/{module} --cov-report=term-missing
```

**SUCCESS CRITERIA**:
- ✅ ALL tests pass (100% success rate)
- ✅ ZERO warnings in test output
- ✅ ZERO mypy errors
- ✅ ZERO ruff violations
- ✅ Coverage > 80% for critical paths

**TRACKING**:
Create .claude_experiments/{module}/{module}_audit.md with:
```markdown
# {module} Module Audit

## Issues Found
| ID | Status | Severity | File:Line | Issue | Fix Applied | Test Result |
|----|--------|----------|-----------|-------|-------------|-------------|
| {module}-001 | FIXED | CRITICAL | base.py:45 | Missing await | Added await keyword | PASS |

## Test Results
- Before: X failures, Y warnings
- After: 0 failures, 0 warnings
- Coverage: XX%

## Validation Commands Run
- [ ] pytest unit tests
- [ ] pytest integration tests  
- [ ] mypy type checking
- [ ] ruff linting
```

MANDATORY COMPLETION CRITERIA:
You MUST achieve ALL of these before marking complete:

1. Run full test suite and paste the output showing:
    ========= X passed, 0 failed, 0 errors, 0 warnings =========

2. Run mypy and paste output showing:
    Success: no issues found in N source files

3. Create a checklist and check EVERY item:
- [ ] All tests pass
- [ ] Zero warnings
- [ ] Zero type errors
- [ ] All validation errors handled
- [ ] All connections properly managed
- [ ] All Decimal precision maintained

4. If ANY criterion is not met, you MUST:
- Fix the issue immediately
- Re-run validation
- Update the checklist

DO NOT report "remaining issues" - FIX THEM ALL.

IMPORTANT: Do NOT move to next module until current module has 100% test pass rate with zero warnings!
"""

# Keep the old template for backward compatibility
MODULE_CHECK_TEMPLATE = MODULE_CHECK_TEMPLATE_V2

# Prompt template for checking integration V2
INTEGRATION_CHECK_V2 = """
CRITICAL: Ensure PERFECT integration between {module} → {dependency} with ZERO test failures/warnings.

**PHASE 1: DEEP INTEGRATION ANALYSIS**
Use specialized agents to verify ALL integration points:

1. **IMPORT & DEPENDENCY ANALYSIS** (Task: system-design-architect)
   - Verify all imports from {dependency} exist and are public APIs
   - Check for circular dependencies causing import errors
   - Ensure correct import paths (no relative imports across modules)
   - Validate module initialization order dependencies
   - Check for missing __init__.py exports

2. **INTERFACE COMPLIANCE** (Task: integration-architect)
   - Verify {module} uses correct {dependency} interfaces
   - Check method signatures match expected contracts
   - Validate return types match {dependency} specifications
   - Ensure proper use of abstract base classes/protocols
   - Check for interface version compatibility

3. **ASYNC INTEGRATION** (Task: pipeline-execution-orchestrator)
   - Verify async/await consistency across module boundaries
   - Check for missing awaits when calling {dependency} async methods
   - Validate proper async context propagation
   - Ensure no blocking calls in async chains
   - Check for proper async resource cleanup

4. **DATA FLOW VALIDATION** (Task: data-pipeline-maestro)
   - Verify data types passed to {dependency} are correct
   - Check for proper data validation before passing to {dependency}
   - Validate data transformations at boundaries
   - Ensure no data loss or corruption
   - Check for proper null/None handling

5. **ERROR PROPAGATION** (Task: quality-control-enforcer)
   - Verify all {dependency} exceptions are properly caught
   - Check error handling doesn't swallow critical errors
   - Validate error context is preserved across boundaries
   - Ensure proper logging of integration failures
   - Check for appropriate retry mechanisms

6. **RESOURCE MANAGEMENT** (Task: infrastructure-wizard)
   - Verify {dependency} resources are properly acquired/released
   - Check for connection pool usage where appropriate
   - Validate transaction boundaries are respected
   - Ensure no resource leaks across module boundaries
   - Check for proper cleanup in error scenarios

7. **DEPENDENCY INJECTION** (Task: system-design-architect)
   - Verify proper use of dependency injection patterns
   - Check for hardcoded dependencies that should be injected
   - Validate mock-ability for testing
   - Ensure proper factory/builder patterns where needed
   - Check for service locator anti-patterns

**PHASE 2: INTEGRATION TESTING**

1. **Test Coverage Analysis**:
   ```bash
   # Find all test files that test this integration
   grep -r "from src.{dependency}" tests/unit/test_{module}/
   grep -r "from src.{dependency}" tests/integration/
   
   # Run integration-specific tests
   python -m pytest tests/integration/ -k "{module} and {dependency}" -xvs
   ```

2. **Mock Verification**:
   - Ensure mocks match actual {dependency} interfaces
   - Verify mock return values match real implementations
   - Check for over-mocking that hides integration issues
   - Validate test doubles are kept in sync

3. **Contract Testing**:
   - Create contract tests for {module} → {dependency} interface
   - Verify both sides of integration honor contracts
   - Test edge cases and error conditions
   - Validate performance characteristics

**PHASE 3: SYSTEMATIC FIXES**

1. **Fix Priority**:
   - CRITICAL: Breaks {module} → {dependency} communication
   - HIGH: Causes test failures or warnings
   - MEDIUM: Violates best practices but works

2. **Fix Process**:
   ```python
   # 1. Fix import issues first
   # 2. Fix interface compliance
   # 3. Fix async/await patterns
   # 4. Fix data flow issues
   # 5. Fix error handling
   # 6. Fix resource management
   ```

3. **Validation After Each Fix**:
   ```bash
   # Test specific integration
   python -m pytest tests/ -k "{module} and {dependency}" -xvs --tb=short
   
   # Check for import errors
   python -c "from src.{module} import *; from src.{dependency} import *"
   
   # Run type checking
   mypy src/{module}/ --follow-imports=normal
   ```

**PHASE 4: COMPREHENSIVE VALIDATION**

Run complete integration test suite:
```bash
# All tests involving both modules
python -m pytest tests/ -k "{module} and {dependency}" -v --no-warnings

# Check for memory leaks
python -m pytest tests/integration/ -k "{module}" --memprof

# Verify no performance regression
python -m pytest tests/performance/ -k "{module}" --benchmark-only
```

**SUCCESS CRITERIA**:
- ✅ All integration tests pass
- ✅ No import errors or warnings
- ✅ No resource leaks detected
- ✅ Proper error propagation verified
- ✅ Contract tests all pass

**TRACKING**:
Create .claude_experiments/{module}/{module}_to_{dependency}_integration.md:
```markdown
# Integration Audit: {module} → {dependency}

## Integration Points
| File | Line | Integration Type | Status |
|------|------|-----------------|--------|
| service.py | 45 | API Call | ✅ Fixed |

## Issues Fixed
| ID | Severity | Issue | Fix | Test Result |
|----|----------|-------|-----|-------------|
| INT-001 | CRITICAL | Missing await | Added await | PASS |

## Test Coverage
- Integration tests: X/Y passing
- Contract tests: A/B passing
- Performance: No regression
```

MANDATORY COMPLETION CRITERIA:
You MUST achieve ALL of these before marking complete:

1. Run full test suite and paste the output showing:
    ========= X passed, 0 failed, 0 errors, 0 warnings =========

2. Run mypy and paste output showing:
    Success: no issues found in N source files

3. Create a checklist and check EVERY item:
- [ ] All tests pass
- [ ] Zero warnings
- [ ] Zero type errors
- [ ] All validation errors handled
- [ ] All connections properly managed
- [ ] All Decimal precision maintained

4. If ANY criterion is not met, you MUST:
- Fix the issue immediately
- Re-run validation
- Update the checklist

DO NOT report "remaining issues" - FIX THEM ALL.

IMPORTANT: Every integration must work flawlessly with zero test failures!
"""

# Prompt template for checking integration (keep old one for compatibility)
INTEGRATION_TEMPLATE = INTEGRATION_CHECK_V2


# Final validation template for entire codebase
FINAL_VALIDATION_TEMPLATE = """
FINAL VALIDATION: Ensure ENTIRE codebase is production-ready with 100% test pass rate.

**COMPREHENSIVE TEST SUITE**

1. **Run ALL Unit Tests**:
   ```bash
   # Run all unit tests with strict error checking
   python -m pytest tests/unit/ -v --tb=short -W error --strict-markers
   
   # Check coverage
   python -m pytest tests/unit/ --cov=src --cov-report=term-missing --cov-fail-under=80
   ```

2. **Run ALL Integration Tests**:
   ```bash
   # Run all integration tests
   python -m pytest tests/integration/ -v --tb=short -W error
   
   # Run with different test orders to catch state dependencies
   python -m pytest tests/integration/ --random-order
   ```

3. **Static Analysis**:
   ```bash
   # Type checking on entire codebase
   mypy src/ --ignore-missing-imports --strict --no-error-summary
   
   # Linting
   ruff check src/ --select=ALL --ignore=D,ANN101,ANN102
   ruff format src/ --check
   
   # Security scanning
   bandit -r src/ -ll
   
   # Complexity analysis
   radon cc src/ -s -nb
   ```

4. **Performance Testing**:
   ```bash
   # Run performance benchmarks
   python -m pytest tests/performance/ --benchmark-only
   
   # Memory profiling
   python -m pytest tests/integration/ --memprof
   ```

5. **Mock Mode Testing**:
   ```bash
   # Test in mock mode
   MOCK_MODE=true python -m pytest tests/ -k mock
   ```

**CRITICAL VALIDATION POINTS**

1. **Module Coherence**:
   - All modules properly initialized
   - No circular dependencies
   - Clean module boundaries
   - Proper abstraction levels

2. **Async Consistency**:
   - All async functions properly awaited
   - No blocking I/O in async contexts
   - Proper async context managers
   - Clean shutdown of async resources

3. **Resource Management**:
   - All database connections closed
   - All file handles released
   - All WebSocket connections cleaned up
   - No memory leaks detected

4. **Error Handling**:
   - All exceptions properly caught
   - Critical errors re-raised
   - Proper error logging
   - Graceful degradation

5. **Trading Logic Integrity**:
   - Position calculations accurate
   - Risk limits enforced
   - Order state transitions valid
   - Fee calculations correct
   - Decimal precision maintained

**SUCCESS METRICS**:
- ✅ 100% of unit tests pass
- ✅ 100% of integration tests pass  
- ✅ 0 mypy errors
- ✅ 0 ruff violations
- ✅ 0 security issues (bandit)
- ✅ Code coverage > 80%
- ✅ No performance regressions
- ✅ No memory leaks
- ✅ All async properly handled
- ✅ Clean module dependencies

**FINAL REPORT**:
Generate .claude_experiments/{module}/FINAL_VALIDATION_REPORT.md with:
- Total tests run and passed
- Coverage percentage
- Static analysis results
- Performance metrics
- Any remaining warnings (should be 0)
- Production readiness assessment

**MANDATORY COMPLETION CRITERIA**:
You MUST achieve ALL of these before marking complete:

1. Run full test suite and paste the output showing:
    ========= X passed, 0 failed, 0 errors, 0 warnings =========

2. Run mypy and paste output showing:
    Success: no issues found in N source files

3. Create a checklist and check EVERY item:
- [ ] All tests pass
- [ ] Zero warnings
- [ ] Zero type errors
- [ ] All validation errors handled
- [ ] All connections properly managed
- [ ] All Decimal precision maintained

4. If ANY criterion is not met, you MUST:
- Fix the issue immediately
- Re-run validation
- Update the checklist

DO NOT report "remaining issues" - FIX THEM ALL.

ONLY mark as complete when ALL metrics show green!
"""


def execute_claude_prompt(prompt):
    """Execute a prompt using Claude Code CLI."""
    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "--model", "claude-opus-4-20250514",
        "-p",
        prompt
    ]

    try:
        subprocess.run(cmd, capture_output=False, text=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Error executing prompt")
        return False
    except FileNotFoundError:
        print("✗ Claude Code CLI not found")
        sys.exit(1)


def check_module(module):
    """Check module's internal implementation."""
    prompt = MODULE_CHECK_TEMPLATE.format(module=module)

    print(f"\n{'='*60}")
    print(f"Checking Module: {module} (internal implementation)")
    print('='*60)

    success = execute_claude_prompt(prompt)

    if success:
        print(f"✓ Module check completed: {module}")

    # Delay to avoid rate limiting
    time.sleep(3)

    return success


def check_integration(module, dependency):
    """Check if a module properly uses its dependency."""
    prompt = INTEGRATION_TEMPLATE.format(module=module, dependency=dependency)

    print(f"\n{'='*60}")
    print(f"Checking Integration: {module} → {dependency}")
    print('='*60)

    success = execute_claude_prompt(prompt)

    if success:
        print(f"✓ Integration check completed: {module} → {dependency}")

    # Delay to avoid rate limiting
    time.sleep(3)

    return success


def run_final_validation():
    """Run final validation on entire codebase."""
    print("\n" + "="*70)
    print(" FINAL VALIDATION - ENTIRE CODEBASE")
    print("="*70)

    prompt = FINAL_VALIDATION_TEMPLATE

    print("\nRunning comprehensive validation on entire codebase...")
    print("This will check:")
    print("  - All unit tests pass")
    print("  - All integration tests pass")
    print("  - Static analysis (mypy, ruff, bandit)")
    print("  - Performance benchmarks")
    print("  - Memory profiling")
    print("  - Mock mode testing")

    success = execute_claude_prompt(prompt)

    if success:
        print("\n✓ Final validation completed!")
        print("Check .claude_experiments/FINAL_VALIDATION_REPORT.md for results")
    else:
        print("\n✗ Final validation failed!")

    return success


def main():
    """Run systematic module and integration checks."""
    print("="*70)
    print(" MODULE & INTEGRATION CHECKER V2")
    print("="*70)

    # Filter out skipped modules
    modules_to_check = [m for m in MODULE_HIERARCHY if m not in SKIP_MODULES]

    # Calculate actual checks (excluding skipped ones)
    total_module_checks = len(modules_to_check)
    total_integration_checks = sum(
        len([d for d in deps if (module, d) not in SKIP_INTEGRATIONS])
        for module, deps in MODULE_DEPENDENCIES.items()
        if module not in SKIP_MODULES
    )
    total_checks = total_module_checks + total_integration_checks

    # Show skip information if any
    if SKIP_MODULES:
        print(f"\n⚠️  Skipping modules: {', '.join(SKIP_MODULES)}")
    if SKIP_INTEGRATIONS:
        print(f"⚠️  Skipping integrations: {', '.join([f'{m}→{d}' for m, d in SKIP_INTEGRATIONS])}")

    current_check = 0
    failed_module_checks = []
    failed_integration_checks = []

    print(f"\nTotal checks to perform:")
    print(f"  - Module checks: {total_module_checks}")
    print(f"  - Integration checks: {total_integration_checks}")
    print(f"  - Total: {total_checks}")
    if SKIP_MODULES or SKIP_INTEGRATIONS:
        print(f"  - Skipped: {len(SKIP_MODULES)} modules, {len(SKIP_INTEGRATIONS)} integrations")
    print("\nStarting systematic validation...\n")

    # Process each module completely before moving to the next
    for module in MODULE_HIERARCHY:
        # Skip if module is in skip list
        if module in SKIP_MODULES:
            print(f"\n{'#'*70}")
            print(f"# MODULE: {module} [SKIPPED]")
            print(f"{'#'*70}")
            continue

        print(f"\n{'#'*70}")
        print(f"# MODULE: {module}")
        print(f"{'#'*70}")

        # Step 1: Check the module itself
        current_check += 1
        print(f"\n[{current_check}/{total_checks}] Step 1: Checking {module} module implementation")

        success = check_module(module)
        if not success:
            failed_module_checks.append(module)

        # Step 2: Check all integrations for this module
        dependencies = MODULE_DEPENDENCIES[module]

        if dependencies:
            # Filter out skipped integrations
            deps_to_check = [d for d in dependencies if (module, d) not in SKIP_INTEGRATIONS]
            skipped_deps = [d for d in dependencies if (module, d) in SKIP_INTEGRATIONS]

            if skipped_deps:
                print(
                    f"\n⚠️  Skipping integrations: {', '.join([f'{module}→{d}' for d in skipped_deps])}")

            if deps_to_check:
                print(f"\nStep 2: Checking {len(deps_to_check)} integration(s) for {module}")

                for dependency in deps_to_check:
                    current_check += 1
                    print(f"\n[{current_check}/{total_checks}]", end="")

                    success = check_integration(module, dependency)

                    if not success:
                        failed_integration_checks.append((module, dependency))
            else:
                print(f"\nStep 2: All integrations skipped for {module}")
        else:
            print(f"\nStep 2: No dependencies to check for {module}")

        print(f"\n{'='*70}")
        print(f"Completed all checks for module: {module}")
        print(f"{'='*70}")

    # Summary
    print("\n" + "#"*70)
    print("# VALIDATION SUMMARY")
    print("#"*70)

    if failed_module_checks or failed_integration_checks:
        if failed_module_checks:
            print(f"\n✗ {len(failed_module_checks)} module check(s) failed:")
            for module in failed_module_checks:
                print(f"  - {module}")

        if failed_integration_checks:
            print(f"\n✗ {len(failed_integration_checks)} integration check(s) failed:")
            for module, dep in failed_integration_checks:
                print(f"  - {module} → {dep}")
    else:
        print("\n✓ All module and integration checks completed successfully!")

    print(f"\nTotal checks performed: {current_check}/{total_checks}")
    print(f"  - Module checks: {total_module_checks}")
    print(f"  - Integration checks: {total_integration_checks}")
    print("#"*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module Integration Checker V2")
    parser.add_argument("--test", action="store_true",
                        help="Test mode - only check first 4 modules")
    parser.add_argument("--skip", nargs="+", help="Modules to skip (e.g., --skip core utils)")
    parser.add_argument("--skip-integration", nargs="+",
                        help="Integrations to skip (e.g., --skip-integration database:core database:utils)")
    parser.add_argument("--only", nargs="+",
                        help="Only check specific modules (e.g., --only database state)")
    parser.add_argument(
        "--start-from", help="Start from a specific module (e.g., --start-from database)")
    parser.add_argument("--final-validation", action="store_true",
                        help="Run final validation on entire codebase after module checks")
    parser.add_argument("--final-only", action="store_true",
                        help="Skip module checks and only run final validation")

    args = parser.parse_args()

    if args.test:
        print("TEST MODE - Only checking first 4 modules")
        MODULE_HIERARCHY = MODULE_HIERARCHY[:4]

    if args.skip:
        SKIP_MODULES.extend(args.skip)
        print(f"Command-line skip: {', '.join(args.skip)}")

    if args.skip_integration:
        for integration in args.skip_integration:
            if ":" in integration:
                module, dep = integration.split(":", 1)
                SKIP_INTEGRATIONS.append((module, dep))
        print(f"Command-line skip integrations: {', '.join(args.skip_integration)}")

    if args.only:
        # Only check specified modules
        MODULE_HIERARCHY = [m for m in MODULE_HIERARCHY if m in args.only]
        print(f"Only checking: {', '.join(MODULE_HIERARCHY)}")

    if args.start_from:
        # Start from a specific module
        if args.start_from in MODULE_HIERARCHY:
            start_index = MODULE_HIERARCHY.index(args.start_from)
            MODULE_HIERARCHY = MODULE_HIERARCHY[start_index:]
            print(f"Starting from module: {args.start_from}")
        else:
            print(f"Warning: Module '{args.start_from}' not found. Starting from beginning.")

    # Handle final validation options
    if args.final_only:
        # Only run final validation
        run_final_validation()
    else:
        # Run module checks
        main()

        # Run final validation if requested
        if args.final_validation:
            print("\n" + "="*70)
            print("Starting final validation after module checks...")
            print("="*70)
            run_final_validation()
