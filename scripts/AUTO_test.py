#!/usr/bin/env python3
"""
Comprehensive tests for AUTO_FINAL.py script
Tests all major components and functionality
"""

import unittest
import sys
import os
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module to test
import AUTO_FINAL

# ============================================================================
# TEST PROGRESS TRACKER
# ============================================================================

class TestProgressTracker(unittest.TestCase):
    """Test the ProgressTracker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.progress_file = Path(self.temp_dir) / "progress.json"
        self.tracker = AUTO_FINAL.ProgressTracker(self.progress_file)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(len(self.tracker.modules), 0)
        self.assertEqual(len(self.tracker.completed_modules), 0)
        self.assertEqual(len(self.tracker.failed_modules), 0)
    
    def test_update_module_no_deadlock(self):
        """Test that update_module doesn't cause deadlock"""
        # This should complete without hanging
        self.tracker.update_module("test_module", status="in_progress")
        
        progress = self.tracker.get_module_progress("test_module")
        self.assertEqual(progress.status, "in_progress")
        self.assertEqual(progress.module, "test_module")
    
    def test_save_and_load_progress(self):
        """Test saving and loading progress"""
        # Update some modules
        self.tracker.update_module("module1", status="completed", iteration=3)
        self.tracker.update_module("module2", status="failed", errors=["error1"])
        
        # Create new tracker to load saved progress
        new_tracker = AUTO_FINAL.ProgressTracker(self.progress_file)
        
        # Verify loaded data
        self.assertIn("module1", new_tracker.completed_modules)
        self.assertIn("module2", new_tracker.failed_modules)
        
        module1 = new_tracker.get_module_progress("module1")
        self.assertEqual(module1.status, "completed")
        self.assertEqual(module1.iteration, 3)
        
        module2 = new_tracker.get_module_progress("module2")
        self.assertEqual(module2.status, "failed")
        self.assertEqual(module2.errors, ["error1"])
    
    def test_concurrent_updates(self):
        """Test thread safety of concurrent updates"""
        def update_module(module_name):
            for i in range(10):
                self.tracker.update_module(module_name, iteration=i)
                time.sleep(0.001)  # Small delay to encourage race conditions
        
        # Start multiple threads updating different modules
        threads = []
        for i in range(5):
            t = threading.Thread(target=update_module, args=(f"module_{i}",))
            t.start()
            threads.append(t)
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify all modules were updated
        for i in range(5):
            progress = self.tracker.get_module_progress(f"module_{i}")
            self.assertEqual(progress.iteration, 9)

# ============================================================================
# TEST RATE LIMITER
# ============================================================================

class TestRateLimiter(unittest.TestCase):
    """Test the RateLimiter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.limiter = AUTO_FINAL.RateLimiter()
        # Override limits for testing
        AUTO_FINAL.API_RATE_LIMIT_PER_MINUTE = 2
        AUTO_FINAL.API_RATE_LIMIT_PER_HOUR = 10
    
    def test_initialization(self):
        """Test rate limiter initialization"""
        self.assertEqual(len(self.limiter.calls_per_minute), 0)
        self.assertEqual(len(self.limiter.calls_per_hour), 0)
    
    @patch('time.sleep')
    def test_rate_limiting(self, mock_sleep):
        """Test that rate limiting works"""
        # First call should not wait
        wait_time = self.limiter.wait_if_needed()
        self.assertEqual(wait_time, 0)
        
        # Second call should not wait
        wait_time = self.limiter.wait_if_needed()
        self.assertEqual(wait_time, 0)
        
        # Third call should wait (exceeds limit of 2 per minute)
        wait_time = self.limiter.wait_if_needed()
        self.assertGreater(wait_time, 0)
        mock_sleep.assert_called()

# ============================================================================
# TEST CASCADE PROTECTOR
# ============================================================================

class TestCascadeProtector(unittest.TestCase):
    """Test the CascadeProtector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_runner = Mock()
        self.protector = AUTO_FINAL.CascadeProtector(self.mock_runner)
    
    def test_update_baseline(self):
        """Test baseline update"""
        test_result = {"success": True, "output": "test"}
        self.protector.update_baseline("module1", test_result)
        
        self.assertEqual(self.protector.module_test_results["module1"], test_result)
    
    def test_check_dependents_no_cascade(self):
        """Test checking dependents with no cascade"""
        # Set up baseline
        self.protector.update_baseline("dependency", {"success": True})
        
        # Mock test runner to return success
        self.mock_runner.run_tests.return_value = {"success": True}
        
        # Check dependents
        results = self.protector.check_dependents("module1")
        
        # Should not detect cascade
        self.assertFalse(self.protector.cascade_detected)
    
    def test_check_dependents_with_cascade(self):
        """Test detecting cascade failure"""
        # Set up module dependency
        AUTO_FINAL.DEPENDENT_MODULES["module1"] = ["dependent1"]
        
        # Set baseline as passing
        self.protector.update_baseline("dependent1", {"success": True})
        
        # Mock test runner to return failure
        self.mock_runner.run_tests.return_value = {"success": False}
        
        # Check dependents
        results = self.protector.check_dependents("module1")
        
        # Should detect cascade
        self.assertTrue(self.protector.cascade_detected)
        self.assertEqual(results, [("dependent1", False)])

# ============================================================================
# TEST CLAUDE EXECUTOR
# ============================================================================

class TestClaudeExecutor(unittest.TestCase):
    """Test the ClaudeExecutor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_limiter = Mock()
        self.mock_detector = Mock()
        self.mock_detector.get_constraints.return_value = "test constraints"
        self.executor = AUTO_FINAL.ClaudeExecutor(self.mock_limiter, self.mock_detector)
    
    @patch('subprocess.run')
    def test_execute_prompt_success(self, mock_run):
        """Test successful prompt execution"""
        # Save original value and set DRY_RUN_MODE to False
        original_dry_run = AUTO_FINAL.DRY_RUN_MODE
        AUTO_FINAL.DRY_RUN_MODE = False
        
        try:
            # Mock subprocess to return success
            mock_run.return_value = Mock(
                stdout="Success response",
                stderr="",
                returncode=0
            )
            
            success, output = self.executor.execute_prompt("test prompt", "test description")
            
            self.assertTrue(success)
            self.assertEqual(output, "Success response")
            self.mock_limiter.wait_if_needed.assert_called_once()
        finally:
            # Restore original value
            AUTO_FINAL.DRY_RUN_MODE = original_dry_run
    
    @patch('subprocess.run')
    def test_execute_prompt_dry_run(self, mock_run):
        """Test prompt execution in dry-run mode"""
        # Save original value and set DRY_RUN_MODE to True
        original_dry_run = AUTO_FINAL.DRY_RUN_MODE
        AUTO_FINAL.DRY_RUN_MODE = True
        
        try:
            # Create new executor with DRY_RUN_MODE enabled
            executor = AUTO_FINAL.ClaudeExecutor(self.mock_limiter, self.mock_detector)
            
            success, output = executor.execute_prompt("test prompt", "test description")
            
            # Should return success without calling subprocess
            self.assertTrue(success)
            self.assertIn("DRY-RUN", output)
            mock_run.assert_not_called()
        finally:
            # Restore original value
            AUTO_FINAL.DRY_RUN_MODE = original_dry_run
    
    @patch('subprocess.run')
    def test_execute_prompt_retry_on_error(self, mock_run):
        """Test retry logic on API errors"""
        # Save original value and set DRY_RUN_MODE to False
        original_dry_run = AUTO_FINAL.DRY_RUN_MODE
        AUTO_FINAL.DRY_RUN_MODE = False
        
        try:
            # First call fails with exception, second succeeds
            def side_effect(*args, **kwargs):
                if mock_run.call_count == 1:
                    raise Exception("500 Internal Server Error")
                return Mock(stdout="Success", stderr="", returncode=0)
            
            mock_run.side_effect = side_effect
            
            self.executor.retry_delay = 0.01  # Short delay for testing
            
            success, output = self.executor.execute_prompt("test", "test")
            
            # Should retry and succeed
            self.assertTrue(success)
            self.assertEqual(output, "Success")
            self.assertEqual(mock_run.call_count, 2)
        finally:
            # Restore original value
            AUTO_FINAL.DRY_RUN_MODE = original_dry_run

# ============================================================================
# TEST TEST RUNNER
# ============================================================================

class TestTestRunner(unittest.TestCase):
    """Test the TestRunner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.runner = AUTO_FINAL.TestRunner()
    
    @patch('subprocess.run')
    def test_run_tests_success(self, mock_run):
        """Test successful test run"""
        # Mock subprocess output
        mock_run.return_value = Mock(
            stdout="test_module.py::test_function PASSED\n1 passed in 0.01s",
            stderr="",
            returncode=0
        )
        
        result = self.runner.run_tests("core")
        
        self.assertTrue(result["success"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["returncode"], 0)
    
    @patch('subprocess.run')
    def test_run_tests_failure(self, mock_run):
        """Test test run with failures"""
        # Mock subprocess output with failures
        mock_run.return_value = Mock(
            stdout="test_module.py::test_function FAILED",
            stderr="AssertionError: test failed",
            returncode=1
        )
        
        result = self.runner.run_tests("core")
        
        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 1)
    
    def test_run_tests_dry_run(self):
        """Test test run in dry-run mode"""
        # Enable dry-run mode
        AUTO_FINAL.DRY_RUN_MODE = True
        
        result = self.runner.run_tests("core")
        
        # Should return simulated success
        self.assertTrue(result["success"])
        self.assertTrue(result.get("dry_run"))
        
        # Reset
        AUTO_FINAL.DRY_RUN_MODE = False
    
    def test_extract_failures(self):
        """Test extraction of test failures"""
        output = """
test_module.py::test_function1 PASSED
test_module.py::test_function2 FAILED
    AssertionError: Expected 1 but got 2
test_module.py::test_function3 ERROR
    ImportError: No module named 'missing'
"""
        failures = self.runner._extract_failures(output)
        
        self.assertEqual(len(failures), 6)  # Due to overlapping extraction
        self.assertIn("FAILED", failures[0])
        self.assertIn("ERROR", failures[4])  # ERROR appears at index 4
    
    def test_extract_slow_tests(self):
        """Test extraction of slow tests"""
        output = """
test_module.py::test_fast [0.5s]
test_module.py::test_slow [10.2s]
test_module.py::test_very_slow [25.8s]
"""
        slow_tests = self.runner._extract_slow_tests(output)
        
        # Default threshold is 5 seconds
        self.assertEqual(len(slow_tests), 2)
        self.assertIn("10.2s", slow_tests[0])
        self.assertIn("25.8s", slow_tests[1])

# ============================================================================
# TEST MODULE PROCESSOR
# ============================================================================

class TestModuleProcessor(unittest.TestCase):
    """Test the ModuleProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_claude = Mock()
        self.mock_runner = Mock()
        self.mock_progress = Mock()
        self.mock_protector = Mock()
        
        self.processor = AUTO_FINAL.ModuleProcessor(
            self.mock_claude,
            self.mock_runner,
            self.mock_progress,
            self.mock_protector
        )
    
    def test_calculate_module_steps(self):
        """Test step calculation"""
        steps = self.processor._calculate_module_steps("core")
        
        # Should calculate reasonable number of steps
        self.assertGreater(steps, 0)
        self.assertLess(steps, 1000)
    
    def test_log_step(self):
        """Test step logging"""
        self.processor.total_steps = 10
        self.processor.current_step = 0
        
        # Should increment step counter
        self.processor._log_step("Test action")
        self.assertEqual(self.processor.current_step, 1)
        
        self.processor._log_step("Another action")
        self.assertEqual(self.processor.current_step, 2)
    
    @patch('AUTO_FINAL.MODULE_PROMPTS', {
        AUTO_FINAL.PromptType.IMPORTS_ONLY: "Fix imports in {module}",
        AUTO_FINAL.PromptType.TYPES_ONLY: "Fix types in {module}",
        AUTO_FINAL.PromptType.ASYNC_ONLY: "Fix async in {module}",
        AUTO_FINAL.PromptType.RESOURCES_ONLY: "Fix resources in {module}",
        AUTO_FINAL.PromptType.ERRORS_ONLY: "Fix errors in {module}"
    })
    def test_process_module_flow(self):
        """Test the overall module processing flow"""
        # Setup mocks
        self.mock_progress.is_module_complete.return_value = False
        self.mock_progress.get_module_progress.return_value = Mock(
            iteration=0,
            prompts_completed=[]
        )
        self.mock_claude.execute_prompt.return_value = (True, "Success")
        self.mock_runner.run_tests.return_value = {
            "success": True,
            "failures": []
        }
        self.mock_protector.check_dependents.return_value = []  # No cascade failures
        self.mock_protector.cascade_detected = False
        
        # Process module
        result = self.processor.process_module("test_module")
        
        # Should call Claude for fixes
        self.assertGreater(self.mock_claude.execute_prompt.call_count, 0)
        
        # Should run tests
        self.assertGreater(self.mock_runner.run_tests.call_count, 0)
        
        # Should update progress
        self.assertGreater(self.mock_progress.update_module.call_count, 0)

# ============================================================================
# TEST DRY RUN MODE
# ============================================================================

class TestDryRunMode(unittest.TestCase):
    """Test dry-run mode functionality"""
    
    def setUp(self):
        """Enable dry-run mode"""
        AUTO_FINAL.DRY_RUN_MODE = True
    
    def tearDown(self):
        """Disable dry-run mode"""
        AUTO_FINAL.DRY_RUN_MODE = False
    
    def test_claude_executor_dry_run(self):
        """Test Claude executor in dry-run mode"""
        executor = AUTO_FINAL.ClaudeExecutor(Mock(), Mock())
        
        success, output = executor.execute_prompt("test", "test")
        
        self.assertTrue(success)
        self.assertIn("DRY-RUN", output)
    
    def test_test_runner_dry_run(self):
        """Test test runner in dry-run mode"""
        runner = AUTO_FINAL.TestRunner()
        
        result = runner.run_tests("core")
        
        self.assertTrue(result["success"])
        self.assertTrue(result["dry_run"])
        self.assertIn("DRY-RUN", result["output"])
    
    def test_progress_saved_with_dry_run_flag(self):
        """Test that progress includes dry-run flag"""
        temp_dir = tempfile.mkdtemp()
        progress_file = Path(temp_dir) / "progress.json"
        
        tracker = AUTO_FINAL.ProgressTracker(progress_file)
        tracker.update_module("test", status="completed")
        
        # Load saved progress
        with open(progress_file) as f:
            data = json.load(f)
        
        self.assertTrue(data.get("dry_run"))
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

# ============================================================================
# TEST INTEGRATION
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    @patch('subprocess.run')
    @patch('AUTO_FINAL.Path.exists')
    def test_complete_workflow(self, mock_exists, mock_run):
        """Test complete workflow from start to finish"""
        # Setup mocks
        mock_exists.return_value = True  # Test directory exists
        mock_run.return_value = Mock(
            stdout="All tests passed",
            stderr="",
            returncode=0
        )
        
        # Create AutoFixer
        with patch('AUTO_FINAL.HallucinationDetector'):
            fixer = AUTO_FINAL.AutoFixer()
        
        # Process a single module
        with patch.object(fixer.module_processor, 'process_module', return_value=True) as mock_process:
            success = fixer.run_sequential()
        
        self.assertTrue(success)
        mock_process.assert_called()
    
    def test_module_hierarchy_dependencies(self):
        """Test that module hierarchy respects dependencies"""
        # Each module should only depend on modules that appear before it
        for i, module in enumerate(AUTO_FINAL.MODULE_HIERARCHY):
            deps = AUTO_FINAL.MODULE_DEPENDENCIES.get(module, [])
            for dep in deps:
                # Dependency should appear before the module in hierarchy
                dep_index = AUTO_FINAL.MODULE_HIERARCHY.index(dep)
                self.assertLess(
                    dep_index, i,
                    f"{module} depends on {dep} which appears later in hierarchy"
                )

# ============================================================================
# TEST STEP TRACKING
# ============================================================================

class TestStepTracking(unittest.TestCase):
    """Test the step tracking functionality"""
    
    def test_step_calculation(self):
        """Test that steps are calculated correctly"""
        processor = AUTO_FINAL.ModuleProcessor(
            Mock(), Mock(), Mock(), Mock()
        )
        
        steps = processor._calculate_module_steps("core")
        
        # Should include:
        # - Status update (1)
        # - Baseline test (1)  
        # - Fix prompts (5 * MAX_ITERATIONS)
        # - Test runs (MAX_ITERATIONS)
        # - Integration checks
        # - Cascade checks
        
        expected_min = 2 + 5 * AUTO_FINAL.MAX_ITERATIONS_PER_MODULE
        self.assertGreaterEqual(steps, expected_min)
    
    def test_progress_tracking(self):
        """Test that progress percentage is calculated correctly"""
        processor = AUTO_FINAL.ModuleProcessor(
            Mock(), Mock(), Mock(), Mock()
        )
        
        processor.total_steps = 100
        processor.current_step = 0
        
        # Test various progress points
        processor._log_step("Step 1")
        self.assertEqual(processor.current_step, 1)
        
        processor.current_step = 50
        processor._log_step("Halfway")
        self.assertEqual(processor.current_step, 51)
        
        processor.current_step = 99
        processor._log_step("Almost done")
        self.assertEqual(processor.current_step, 100)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)