#!/usr/bin/env python3
"""
Micro-Fix Architecture for Trading Bot - Fast, focused, verifiable fixes.
Replaces monolithic prompts with targeted micro-prompts for rapid iteration.
Terminal-only reporting - no file writes to save time.
"""

import subprocess
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Module dependency hierarchy (order matters - from lowest to highest level)
MODULE_HIERARCHY = [
    "core", "utils", "error_handling", "database", "state", "monitoring",
    "exchanges", "risk_management", "execution", "data", "ml", "analytics",
    "optimization", "strategies", "backtesting", "capital_management", 
    "bot_management", "web_interface"
]

# Modules to skip
SKIP_MODULES = [
    "core", "utils",
    # "core", "utils", "error_handling", "database", "state", "monitoring",
    # "exchanges", "risk_management", "execution"  # Add modules to skip here
]

# Module dependencies
MODULE_DEPENDENCIES = {
    "core": [],
    "utils": ["core"],
    "error_handling": ["core", "utils"],
    "database": ["core", "utils", "error_handling"],
    "state": ["core", "utils", "error_handling", "database", "monitoring"],
    "monitoring": ["core", "utils", "error_handling", "database"],
    "exchanges": ["core", "utils", "error_handling", "database", "monitoring", "state"],
    "data": ["core", "utils", "error_handling", "database", "monitoring", "state"],
    "risk_management": ["core", "utils", "error_handling", "database", "monitoring", "state", "data"],
    "execution": ["core", "utils", "error_handling", "database", "exchanges", "risk_management", "monitoring", "state"],
    "ml": ["core", "utils", "error_handling", "database", "data", "monitoring"],
    "analytics": ["core", "utils", "error_handling", "database", "monitoring", "data", "risk_management"],
    "optimization": ["core", "utils", "error_handling", "database", "data", "ml"],
    "strategies": ["core", "utils", "error_handling", "database", "risk_management", "execution", "data", "monitoring", "ml", "optimization"],
    "backtesting": ["core", "utils", "error_handling", "database", "strategies", "execution", "data", "risk_management"],
    "capital_management": ["core", "utils", "error_handling", "database", "risk_management", "state", "exchanges"],
    "bot_management": ["core", "utils", "error_handling", "database", "strategies", "execution", "capital_management", "monitoring", "state", "risk_management", "data"],
    "web_interface": ["core", "utils", "error_handling", "database", "bot_management", "monitoring", "state", "analytics", "ml"],
}

# Micro-prompts for module fixes - focused, single-purpose tasks
MICRO_PROMPTS = {
    "imports_syntax": """Use Task: system-design-architect agent to fix ONLY import and syntax issues in {module} module:
1. Find missing/incorrect imports causing undefined name errors
2. Fix circular import issues
3. Correct import order (stdlib → third-party → local)
4. Remove unused imports
5. Fix basic syntax errors (typos, missing colons, etc.)

FIX ONLY what breaks imports. Do NOT change business logic.""",

    "type_hints": """Use Task: code-guardian-enforcer agent to fix ONLY type annotation issues in {module} module:
1. Add missing type hints causing mypy errors
2. Fix incorrect type annotations
3. Add proper Generic types and TypeVars
4. Fix Optional/Union type usage
5. Correct return type annotations

FIX ONLY type annotations. Do NOT change function logic.""",

    "async_await": """Use Task: financial-api-architect agent to fix ONLY async/await patterns in {module} module:
1. Add missing `await` keywords for coroutines
2. Fix blocking I/O calls in async functions
3. Correct async context manager usage (`async with`)
4. Fix async generator patterns
5. Add proper async error handling

FIX ONLY async patterns. Do NOT change business logic.""",

    "resource_cleanup": """Use Task: infrastructure-wizard agent to fix ONLY resource management issues in {module} module:
1. Add missing `finally` blocks or context managers
2. Ensure database connections are closed
3. Fix file handle leaks
4. Add WebSocket connection cleanup
5. Fix memory leaks from unclosed resources

FIX ONLY resource management. Do NOT change core logic.""",

    "business_logic": """Fix ONLY business logic errors in {module} module using Task: algo-trading-specialist:
1. Correct decimal precision for financial calculations
2. Fix position sizing logic
3. Validate order state transitions
4. Fix fee calculation errors
5. Add boundary checks (min/max values)

FIX ONLY calculation errors. Preserve existing algorithms.""",

    "error_handling": """Use Task: quality-control-enforcer agent to fix ONLY error handling issues in {module} module:
1. Add try/except blocks for expected failures
2. Replace bare `except:` with specific exceptions
3. Ensure critical errors are re-raised
4. Add proper error logging with context
5. Fix exception chaining (`raise X from Y`)

FIX ONLY error handling. Do NOT change success paths.""",

    "test_compatibility": """Fix ONLY test-related issues in {module} module using Task: integration-test-architect:
1. Fix mock objects that don't match real interfaces
2. Correct test fixture data types
3. Add missing test database cleanup
4. Fix hardcoded test values
5. Ensure async tests use proper await patterns

FIX ONLY test code issues. Do NOT change production code.""",

    "logging_warnings": """Use Task: monitoring-core-integration-analyst agent to fix ONLY logging and warning issues in {module} module:
1. Initialize loggers properly
2. Remove sensitive data from log messages
3. Set appropriate log levels
4. Fix deprecation warnings
5. Remove debug prints and temporary logging

FIX ONLY logging issues. Do NOT change functionality.""",

    "slow_tests": """Use Task: performance-optimization-specialist agent to optimize SLOW RUNNING tests in {module} module that exceed {timeout} seconds:

SLOW TESTS DETECTED:
{slow_tests}

OPTIMIZATION REQUIREMENTS:
1. Mock expensive I/O operations (database, network, file system)
2. Use pytest fixtures with scope='module' or 'session' for shared setup
3. Replace time.sleep() and asyncio.sleep() with mock time
4. Reduce dataset sizes and iteration counts
5. Cache expensive computations in fixtures
6. Use in-memory databases instead of real ones
7. Mock external API calls and websocket connections
8. Implement test data builders for faster object creation
9. Use pytest-xdist for parallel execution where appropriate
10. Optimize async test patterns (proper await, event loop management)

TARGET: Each test should complete in < 5 seconds, module total < {module_timeout} seconds

CRITICAL: Maintain test coverage and reliability while improving speed.
FIX test implementation, NOT test assertions or business logic validation."""
}

# Integration micro-prompts - focused on specific integration aspects
INTEGRATION_PROMPTS = {
    "import_dependencies": """Use Task: integration-architect agent to verify ONLY import dependencies between {module} → {dependency}:
1. Check all imports from {dependency} actually exist
2. Verify import paths are correct
3. Ensure no circular dependencies
4. Check __init__.py exports are used correctly
5. Validate module initialization order

FIX ONLY import issues between these modules.""",

    "interface_compliance": """Check ONLY interface compliance {module} → {dependency} using Task: integration-architect:
1. Verify method signatures match {dependency} contracts
2. Check return types are compatible
3. Ensure proper abstract base class usage
4. Validate protocol implementations
5. Check interface version compatibility

FIX ONLY interface mismatches.""",

    "data_flow": """Validate ONLY data flow between {module} → {dependency} using Task: data-pipeline-maestro:
1. Check data types passed to {dependency} are correct
2. Verify data validation before passing
3. Ensure no data corruption at boundaries
4. Check proper null/None handling
5. Validate data transformations

FIX ONLY data passing issues.""",

    "async_integration": """Use Task: pipeline-execution-orchestrator agent to fix ONLY async integration {module} → {dependency}:
1. Verify async/await consistency across boundaries
2. Check missing awaits for {dependency} async calls
3. Ensure proper async context propagation
4. Fix blocking calls in async chains
5. Add async resource cleanup

FIX ONLY async integration patterns.""",

    "error_propagation": """Use Task: quality-control-enforcer agent to fix ONLY error propagation {module} → {dependency}:
1. Ensure {dependency} exceptions are caught properly
2. Verify error context is preserved
3. Check critical errors are re-raised
4. Add proper error logging for integration failures
5. Implement appropriate retry mechanisms

FIX ONLY error handling between modules."""
}

# Test orchestration prompt - handles test failures systematically
TEST_FIX_PROMPT = """Use Task: tactical-task-coordinator agent to fix ALL test failures for {module} module:

CURRENT TEST RESULTS:
{test_output}

SYSTEMATIC FIX APPROACH:
1. Use Task: financial-qa-engineer agent to identify root cause of each test failure
2. Use Task: integration-test-architect agent to group related failures
3. Use Task: algo-trading-specialist agent for business logic fixes
4. Use Task: strategic-project-coordinator agent for architectural decisions
5. Use Task: performance-optimization-specialist agent for performance issues

SUCCESS CRITERIA: ALL tests pass with 0 failures, 0 errors, 0 warnings.

IMPORTANT: Leverage all available specialized agents to fix the actual code issues causing test failures, not the tests themselves."""


class MicroFixer:
    def __init__(self, parallel=True, max_workers=4, test_timeout=300):
        self.parallel = parallel
        self.max_workers = max_workers
        self.test_timeout = test_timeout
        self.stats = {'fixes': {}, 'tests': {}, 'integrations': {}, 'slow_tests': {}}
        self.total_prompts = 0
        self.current_prompt = 0

    def calculate_total_prompts(self, modules_to_process):
        """Calculate total number of prompts that will be executed."""
        total = 0
        
        for module in modules_to_process:
            # Micro-fixes for each module
            total += len(MICRO_PROMPTS)
            
            # Test fix prompt (likely needed for most modules)
            total += 1
            
            # Slow test fix prompt (if needed)
            total += 1
            
            # Integration prompts for dependencies
            dependencies = MODULE_DEPENDENCIES.get(module, [])
            for dep in dependencies:
                if dep not in SKIP_MODULES:
                    total += len(INTEGRATION_PROMPTS)
        
        return total
    
    def execute_prompt(self, prompt, prompt_description="", max_retries=3):
        """Execute a focused micro-prompt without timeout, with retry logic for 500 errors."""
        self.current_prompt += 1
        
        # Show progress
        if self.total_prompts > 0:
            progress = f"[{self.current_prompt}/{self.total_prompts}]"
            if prompt_description:
                print(f"  {progress} {prompt_description}")
            else:
                print(f"  {progress} Executing prompt...")
        
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--model", "claude-sonnet-4-20250514",
            "-p", prompt
        ]

        retry_count = 0
        while retry_count <= max_retries:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                    # No timeout - let it run as long as needed
                )
                return True, result.stdout
            except subprocess.CalledProcessError as e:
                error_output = e.stderr.lower()
                # Check for 500 error or server error indicators
                if any(err in error_output for err in ['500', 'server error', 'internal server error', 'service unavailable']):
                    retry_count += 1
                    if retry_count <= max_retries:
                        print(f"    ⚠️  Server error detected (attempt {retry_count}/{max_retries}), retrying in 5 seconds...")
                        time.sleep(5)  # Wait 5 seconds before retrying
                        continue
                    else:
                        print(f"    ❌ Server error persisted after {max_retries} retries")
                        return False, f"Server error after {max_retries} retries: {e.stderr}"
                else:
                    # Non-500 error, don't retry
                    return False, f"Error: {e.stderr}"
            except FileNotFoundError:
                print("✗ Claude Code CLI not found")
                sys.exit(1)
        
        return False, "Maximum retries exceeded"

    def run_module_tests(self, module, timeout=300, check_timing=False):
        """Run tests for a specific module and return results with timing info."""
        import os
        import json
        import re
        
        # Map module names to their test directories or files
        test_dir_mapping = {
            "exchanges": "test_exchange",  # Different naming
            "ml": "test_ml",
            "optimization": "test_optimization",
            "analytics": None,  # No tests yet
            "data": "test_data",  # Has test files directly in directory
        }
        
        test_dir = test_dir_mapping.get(module, f"test_{module}")
        
        # Skip if no test directory
        if test_dir is None:
            return True, "No test directory for this module", []
        
        test_path = f"tests/unit/{test_dir}/"
        
        # Check if test path exists
        if not os.path.exists(f"/mnt/e/Work/P-41 Trading/code/t-bot/{test_path}"):
            # Try without trailing slash for direct test files
            test_path = f"tests/unit/{test_dir}"
            if not os.path.exists(f"/mnt/e/Work/P-41 Trading/code/t-bot/{test_path}"):
                return True, f"Test path not found: {test_path}", []
        
        cmd = [
            "python", "-m", "pytest",
            test_path,
            "-v", "--tb=short", "--no-header",
            f"--timeout={timeout}",
            "--timeout-method=thread"
        ]
        
        # Add durations reporting if checking timing
        if check_timing:
            cmd.extend(["--durations=10", "--durations-min=1.0"])

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/mnt/e/Work/P-41 Trading/code/t-bot",
                timeout=timeout + 30  # Give extra time for cleanup
            )
            duration = time.time() - start_time
            
            # Extract slow tests from output
            slow_tests = []
            if check_timing:
                slow_tests = self.extract_slow_tests(result.stdout + result.stderr)
            
            # Check if module exceeded timeout
            if duration > timeout:
                print(f"  ⚠️  Module {module} took {duration:.2f}s (exceeds {timeout}s timeout)")
                slow_tests.append(("Module Total", duration))
            
            return result.returncode == 0, result.stdout + result.stderr, slow_tests
        except subprocess.TimeoutExpired:
            return False, f"Module {module} timeout exceeded ({timeout}s)", [("Module Total", timeout)]
        except Exception as e:
            return False, str(e), []
    
    def extract_slow_tests(self, output):
        """Extract slow test information from pytest output."""
        slow_tests = []
        
        # Look for slowest durations section in pytest output
        duration_section = False
        for line in output.split('\n'):
            if 'slowest' in line.lower() and 'duration' in line.lower():
                duration_section = True
                continue
            
            if duration_section:
                # Parse lines like: "1.23s call     test_module.py::test_function"
                match = re.match(r'^\s*(\d+\.\d+)s\s+\w+\s+(.+?)(?:\[|$)', line)
                if match:
                    duration = float(match.group(1))
                    test_name = match.group(2).split('::')[-1] if '::' in match.group(2) else match.group(2)
                    slow_tests.append((test_name, duration))
                elif line.strip() == '' or '=' in line:
                    duration_section = False
        
        return slow_tests

    def quick_validate(self, module, check_type):
        """Quick validation specific to check type."""
        validations = {
            "imports_syntax": f"python -c 'import src.{module}'",
            "type_hints": f"mypy src/{module}/ --ignore-missing-imports --no-error-summary",
            "async_await": f"python -m pytest tests/unit/test_{module}/ -W ignore::DeprecationWarning -q",
            "resource_cleanup": f"python -m pytest tests/unit/test_{module}/ -W error::ResourceWarning -q",
        }

        cmd = validations.get(check_type)
        if not cmd:
            return True, "No specific validation"

        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                cwd="/mnt/e/Work/P-41 Trading/code/t-bot"
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)

    def log_fix_result(self, module, fix_type, success, validation_result=None):
        """Log fix result for terminal reporting."""
        if module not in self.stats['fixes']:
            self.stats['fixes'][module] = {}
        self.stats['fixes'][module][fix_type] = {
            'success': success,
            'validation': validation_result
        }

    def print_module_summary(self, module, final_result):
        """Print concise module completion summary."""
        print(f"\n📋 MODULE {module.upper()} SUMMARY:")

        if module in self.stats['fixes']:
            fixes = self.stats['fixes'][module]
            passed = sum(1 for f in fixes.values() if f['success'])
            total = len(fixes)
            print(f"  🔧 Micro-fixes: {passed}/{total} successful")

            # Show failed fixes only
            failed = [name for name, result in fixes.items() if not result['success']]
            if failed:
                print(f"  ❌ Failed: {', '.join(failed)}")

        test_status = "✅ PASS" if final_result else "❌ FAIL"
        print(f"  🧪 Final tests: {test_status}")
        print("-" * 50)

    def fix_module_micro(self, module):
        """Apply all micro-fixes to a module."""
        print(f"\n{'='*60}")
        print(f"🔧 MICRO-FIXING MODULE: {module}")
        print('='*60)

        # Apply each micro-fix
        for fix_type, prompt_template in MICRO_PROMPTS.items():
            print(f"\n[{fix_type.upper()}] Fixing {module}...")

            prompt = prompt_template.format(module=module)
            success, output = self.execute_prompt(prompt)

            if success:
                # Quick validation
                valid, val_output = self.quick_validate(module, fix_type)
                status = "✓" if valid else "⚠"
                print(f"{status} {fix_type}: {'PASS' if valid else 'PARTIAL'}")
                self.log_fix_result(module, fix_type, success, valid)
            else:
                print(f"✗ {fix_type}: FAILED - {output[:100]}")
                self.log_fix_result(module, fix_type, False)

        # Final module test
        print(f"\n[TESTING] Running tests for {module}...")
        test_passed, test_output = self.run_module_tests(module)

        if not test_passed:
            print(f"⚠ Tests failed, orchestrating fixes...")
            final_result = self.fix_test_failures(module, test_output)
        else:
            print(f"✓ All tests passed for {module}")
            final_result = True

        self.stats['tests'][module] = final_result
        self.print_module_summary(module, final_result)
        return final_result

    def fix_slow_tests(self, module, slow_tests):
        """Fix slow-running tests in a module."""
        if not slow_tests:
            return True
        
        # Format slow tests for prompt
        slow_tests_str = "\n".join([f"  - {test}: {duration:.2f}s" for test, duration in slow_tests[:10]])
        
        prompt = MICRO_PROMPTS["slow_tests"].format(
            module=module,
            timeout=self.test_timeout,
            slow_tests=slow_tests_str,
            module_timeout=self.test_timeout
        )
        
        print(f"⚡ Optimizing {len(slow_tests)} slow tests...")
        success, output = self.execute_prompt(prompt, f"Optimizing slow tests in {module}")
        
        if success:
            # Re-run tests with timing check
            test_passed, new_output, new_slow_tests = self.run_module_tests(module, self.test_timeout, check_timing=True)
            
            if new_slow_tests and len(new_slow_tests) < len(slow_tests):
                print(f"  ✓ Reduced slow tests from {len(slow_tests)} to {len(new_slow_tests)}")
                self.stats['slow_tests'][module] = {'before': len(slow_tests), 'after': len(new_slow_tests)}
                return True
            elif not new_slow_tests:
                print(f"  ✓ All slow tests optimized!")
                self.stats['slow_tests'][module] = {'before': len(slow_tests), 'after': 0}
                return True
            else:
                print(f"  ⚠ Still have {len(new_slow_tests)} slow tests")
                self.stats['slow_tests'][module] = {'before': len(slow_tests), 'after': len(new_slow_tests)}
                return False
        else:
            print("  ✗ Failed to optimize slow tests")
            return False
    
    def fix_test_failures(self, module, test_output):
        """Use orchestrator to fix test failures."""
        prompt = TEST_FIX_PROMPT.format(module=module, test_output=test_output[:2000])

        print("🤖 Orchestrator fixing test failures...")
        success, output = self.execute_prompt(prompt, f"Fixing test failures in {module}")

        if success:
            # Re-run tests with timing check
            test_passed, new_output, slow_tests = self.run_module_tests(module, self.test_timeout, check_timing=True)
            if test_passed:
                print("✓ Orchestrator fixed all test failures")
                
                # Check for slow tests even after passing
                if slow_tests and any(duration > 5.0 for _, duration in slow_tests):
                    print(f"  ⚠ Tests pass but {len(slow_tests)} are slow")
                    self.fix_slow_tests(module, slow_tests)
                
                return True
            else:
                print("⚠ Some test failures remain, may need manual review")
                print(new_output[:500])
                return False
        else:
            print("✗ Orchestrator failed to fix issues")
            return False

    def fix_integration_micro(self, module, dependency):
        """Apply micro-fixes to module integration."""
        print(f"\n[INTEGRATION] {module} → {dependency}")

        success_count = 0
        for fix_type, prompt_template in INTEGRATION_PROMPTS.items():
            prompt = prompt_template.format(module=module, dependency=dependency)
            success, output = self.execute_prompt(prompt)

            status = "✓" if success else "✗"
            print(f"  {status} {fix_type}")
            if success:
                success_count += 1

        integration_key = f"{module}→{dependency}"
        self.stats['integrations'][integration_key] = success_count == len(INTEGRATION_PROMPTS)
        return success_count == len(INTEGRATION_PROMPTS)

    def process_module_parallel(self, modules):
        """Process multiple modules in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_module = {
                executor.submit(self.fix_module_micro, module): module
                for module in modules
            }

            results = {}
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                try:
                    results[module] = future.result()
                except Exception as e:
                    print(f"✗ {module} failed: {e}")
                    results[module] = False

            return results

    def print_final_summary(self, results):
        """Print comprehensive final summary to terminal."""
        print(f"\n\n🎯 EXECUTION SUMMARY")
        print("=" * 60)

        # Module results
        passed_modules = [m for m, r in results.items() if r]
        failed_modules = [m for m, r in results.items() if not r]

        print(f"\n📦 MODULES ({len(passed_modules)}/{len(results)} passed):")
        for module in passed_modules:
            print(f"  ✅ {module}")

        if failed_modules:
            print(f"\n❌ FAILED MODULES ({len(failed_modules)}):")
            for module in failed_modules:
                print(f"  ❌ {module}")

        # Integration results
        if self.stats['integrations']:
            int_passed = sum(1 for r in self.stats['integrations'].values() if r)
            int_total = len(self.stats['integrations'])
            print(f"\n🔗 INTEGRATIONS ({int_passed}/{int_total} passed):")

            for integration, success in self.stats['integrations'].items():
                status = "✅" if success else "❌"
                print(f"  {status} {integration}")

        # Overall stats
        total_fixes = sum(len(fixes) for fixes in self.stats['fixes'].values())
        successful_fixes = sum(
            sum(1 for f in fixes.values() if f['success'])
            for fixes in self.stats['fixes'].values()
        )

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  • Total micro-fixes applied: {successful_fixes}/{total_fixes}")
        print(f"  • Modules with passing tests: {len(passed_modules)}/{len(results)}")
        if self.stats['integrations']:
            print(f"  • Integration checks passed: {int_passed}/{int_total}")

        success_rate = (len(passed_modules) / len(results)) * 100 if results else 0
        print(f"  • Overall success rate: {success_rate:.1f}%")
        print("=" * 60)

    def run_systematic_fixes(self):
        """Run systematic micro-fixes across all modules."""
        modules_to_process = [m for m in MODULE_HIERARCHY if m not in SKIP_MODULES]

        print(f"🚀 MICRO-FIX PIPELINE: {len(modules_to_process)} modules")
        print(f"📊 Parallel processing: {self.parallel} (workers: {self.max_workers})")

        if self.parallel and len(modules_to_process) > 1:
            # Process independent modules in parallel
            results = self.process_module_parallel(modules_to_process[:3])  # Start small
        else:
            # Sequential processing
            results = {}
            for module in modules_to_process:
                results[module] = self.fix_module_micro(module)

                # Process integrations for this module
                dependencies = MODULE_DEPENDENCIES.get(module, [])
                for dep in dependencies:
                    if dep not in SKIP_MODULES:
                        self.fix_integration_micro(module, dep)

        return results


def main():
    parser = argparse.ArgumentParser(description="Micro-Fix Architecture V2")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--module", help="Process specific module only")

    args = parser.parse_args()

    fixer = MicroFixer(parallel=args.parallel, max_workers=args.workers)

    if args.module:
        # Process single module
        print(f"Processing single module: {args.module}")
        result = fixer.fix_module_micro(args.module)
        print(f"Result: {'✓ SUCCESS' if result else '✗ FAILED'}")
    else:
        # Process all modules
        results = fixer.run_systematic_fixes()

        # Summary
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        print(f"\n{'='*60}")
        print(f"📊 FINAL RESULTS: {passed}/{total} modules fixed")
        print('='*60)

        for module, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {module}")

        fixer.print_final_summary(results)


if __name__ == "__main__":
    main()
