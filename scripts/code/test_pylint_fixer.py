#!/usr/bin/env python3
"""
Comprehensive tests for the pylint fixer components
"""

import unittest
import tempfile
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pylint_error_parser import PylintError, PylintErrorParser
from prompt_builder import PromptBuilder
from claude_executor import ClaudeExecutor
from pylint_fixer_orchestrator import PylintFixerOrchestrator


class TestPylintErrorParser(unittest.TestCase):
    """Test the PylintErrorParser class"""
    
    def setUp(self):
        self.parser = PylintErrorParser()
        self.sample_output = """Pylint Error Report for core
==================================================

************* Module src.core.config
src/core/config.py:389:15: E1136: Value 'self.supported_exchanges' is unsubscriptable (unsubscriptable-object)
src/core/config.py:407:15: E1101: Instance of 'FieldInfo' has no 'get' member (no-member)
************* Module src.core.dependency_injection
src/core/dependency_injection.py:86:67: E1101: Instance of 'DependencyContainer' has no 'has_service' member (no-member)
src/core/dependency_injection.py:99:35: E0602: Undefined variable 'container' (undefined-variable)

------------------------------------------------------------------
Your code has been rated at 9.77/10 (previous run: 9.77/10, +0.00)
"""
    
    def test_parse_content(self):
        """Test parsing pylint output content"""
        errors = self.parser.parse_content(self.sample_output)
        
        self.assertEqual(len(errors), 4)
        
        # Check first error
        first_error = errors[0]
        self.assertEqual(first_error.file_path, "src/core/config.py")
        self.assertEqual(first_error.line_number, 389)
        self.assertEqual(first_error.column, 15)
        self.assertEqual(first_error.error_code, "E1136")
        self.assertIn("unsubscriptable", first_error.error_message)
        
        # Check last error
        last_error = errors[3]
        self.assertEqual(last_error.error_code, "E0602")
        self.assertIn("Undefined variable", last_error.error_message)
    
    def test_group_by_file(self):
        """Test grouping errors by file"""
        self.parser.parse_content(self.sample_output)
        grouped = self.parser.group_by_file()
        
        self.assertEqual(len(grouped), 2)
        self.assertIn("src/core/config.py", grouped)
        self.assertIn("src/core/dependency_injection.py", grouped)
        self.assertEqual(len(grouped["src/core/config.py"]), 2)
        self.assertEqual(len(grouped["src/core/dependency_injection.py"]), 2)
    
    def test_group_by_error_type(self):
        """Test grouping errors by error type"""
        self.parser.parse_content(self.sample_output)
        grouped = self.parser.group_by_error_type()
        
        self.assertIn("E1101", grouped)
        self.assertIn("E1136", grouped)
        self.assertIn("E0602", grouped)
        self.assertEqual(len(grouped["E1101"]), 2)
    
    def test_batch_errors_by_file(self):
        """Test batching errors by file"""
        self.parser.parse_content(self.sample_output)
        batches = self.parser.batch_errors(batch_size=2, strategy='file')
        
        # Should have 2 batches (2 errors per file)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertLessEqual(len(batch), 2)
    
    def test_get_priority_errors(self):
        """Test getting priority errors"""
        self.parser.parse_content(self.sample_output)
        priority = self.parser.get_priority_errors()
        
        # E0602 should come first (undefined variable)
        self.assertEqual(priority[0].error_code, "E0602")
    
    def test_get_summary(self):
        """Test getting error summary"""
        self.parser.parse_content(self.sample_output)
        summary = self.parser.get_summary()
        
        self.assertEqual(summary['total_errors'], 4)
        self.assertEqual(summary['files_affected'], 2)
        self.assertIn('E1101', summary['error_types'])
        self.assertEqual(summary['error_types']['E1101'], 2)


class TestPromptBuilder(unittest.TestCase):
    """Test the PromptBuilder class"""
    
    def setUp(self):
        self.builder = PromptBuilder()
        self.sample_errors = [
            PylintError(
                file_path="src/core/config.py",
                line_number=389,
                column=15,
                error_code="E1136",
                error_message="Value 'self.supported_exchanges' is unsubscriptable",
                module="src.core.config"
            ),
            PylintError(
                file_path="src/core/config.py",
                line_number=407,
                column=15,
                error_code="E1101",
                error_message="Instance of 'FieldInfo' has no 'get' member",
                module="src.core.config"
            )
        ]
    
    def test_get_module_context_without_reference(self):
        """Test getting module context without REFERENCE.md"""
        context = self.builder.get_module_context("test_module")
        
        self.assertIn("cryptocurrency trading bot", context)
        self.assertIn("Use Decimal for all financial calculations", context)
        self.assertIn("service/repository/controller pattern", context)
    
    @patch('prompt_builder.Path.exists')
    @patch('builtins.open')
    def test_get_module_context_with_reference(self, mock_open, mock_exists):
        """Test getting module context with REFERENCE.md"""
        mock_exists.return_value = True
        mock_reference = """## BUSINESS CONTEXT
The core module provides base functionality.

### Critical Notes
- Always use Decimal for money
- Never use float

### Dependencies
- utils: Validation and formatting
- database: Persistence layer"""
        
        mock_open.return_value.__enter__.return_value.read.return_value = mock_reference
        
        context = self.builder.get_module_context("core")
        
        self.assertIn("BUSINESS CONTEXT", context)
        self.assertIn("Critical Notes", context)
        self.assertIn("Dependencies", context)
    
    def test_build_error_fix_prompt(self):
        """Test building error fix prompt"""
        prompt = self.builder.build_error_fix_prompt(
            errors=self.sample_errors,
            module_name="core",
            batch_number=1,
            total_batches=2
        )
        
        # Check prompt structure
        self.assertIn("Fix the following pylint errors", prompt)
        self.assertIn("(Batch 1/2)", prompt)
        self.assertIn("MODULE CONTEXT", prompt)
        self.assertIn("PROJECT STANDARDS", prompt)
        self.assertIn("ERRORS TO FIX", prompt)
        self.assertIn("SPECIFIC FIXES REQUIRED", prompt)
        
        # Check error details
        self.assertIn("src/core/config.py", prompt)
        self.assertIn("Line 389", prompt)
        self.assertIn("Line 407", prompt)
        self.assertIn("E1136", prompt)
        self.assertIn("E1101", prompt)
        
        # Check specific guidance
        self.assertIn("E1101 (no-member)", prompt)
        self.assertIn("E1136 (unsubscriptable)", prompt)
    
    def test_build_verification_prompt(self):
        """Test building verification prompt"""
        prompt = self.builder.build_verification_prompt("core")
        
        self.assertIn("Verify that all pylint errors have been fixed", prompt)
        self.assertIn("core module", prompt)
        self.assertIn("Use Decimal for financial calculations", prompt)


class TestClaudeExecutor(unittest.TestCase):
    """Test the ClaudeExecutor class"""
    
    def setUp(self):
        self.executor = ClaudeExecutor(model="test-model")
    
    @patch('subprocess.run')
    def test_execute_prompt_success(self, mock_run):
        """Test successful prompt execution"""
        mock_run.return_value = Mock(
            stdout="Fixed the errors",
            stderr="",
            returncode=0
        )
        
        success, output = self.executor.execute_prompt("Fix errors", "Test description")
        
        self.assertTrue(success)
        self.assertEqual(output, "Fixed the errors")
        
        # Check command construction
        call_args = mock_run.call_args[0][0]
        self.assertIn("claude", call_args[0])
        self.assertIn("--dangerously-skip-permissions", call_args)
        self.assertIn("--model", call_args)
        self.assertIn("test-model", call_args)
        self.assertIn("-p", call_args)
    
    @patch('subprocess.run')
    def test_execute_prompt_failure(self, mock_run):
        """Test failed prompt execution"""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "claude", stderr="Error message"
        )
        
        success, output = self.executor.execute_prompt("Fix errors")
        
        self.assertFalse(success)
    
    @patch('subprocess.run')
    @patch('builtins.open')
    @patch('prompt_builder.Path.exists')
    def test_fix_pylint_errors(self, mock_exists, mock_open, mock_run):
        """Test fixing pylint errors"""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = """Pylint Error Report for test
==================================================

src/test/file.py:10:5: E0602: Undefined variable 'x' (undefined-variable)"""
        
        mock_run.return_value = Mock(
            stdout="Fixed",
            stderr="",
            returncode=0
        )
        
        success, output = self.executor.fix_pylint_errors("test")
        
        self.assertTrue(success)
        self.assertEqual(output, "Fixed")


class TestPylintFixerOrchestrator(unittest.TestCase):
    """Test the PylintFixerOrchestrator class"""
    
    def setUp(self):
        self.orchestrator = PylintFixerOrchestrator(
            max_iterations=3,
            batch_size=5,
            batch_strategy='file'
        )
    
    @patch('subprocess.run')
    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.is_dir')
    @patch('prompt_builder.Path.exists')
    def test_run_pylint_success(self, mock_exists, mock_is_dir, mock_glob, mock_run):
        """Test successful pylint run"""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        # Mock finding Python files in the directory
        mock_glob.return_value = [Path("src/test/config.py")]
        
        mock_run.return_value = Mock(
            stdout="""************* Module src.test.config
src/test/config.py:10:5: E0602: Undefined variable 'x' (undefined-variable)""",
            stderr="",
            returncode=1
        )
        
        success, errors = self.orchestrator.run_pylint("test")
        
        self.assertTrue(success)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_code, "E0602")
    
    @patch('subprocess.run')
    @patch('prompt_builder.Path.exists')
    def test_run_pylint_module_not_exists(self, mock_exists, mock_run):
        """Test pylint run with non-existent module"""
        mock_exists.return_value = False
        
        success, errors = self.orchestrator.run_pylint("nonexistent")
        
        self.assertFalse(success)
        self.assertEqual(len(errors), 0)
    
    @patch('builtins.open', create=True)
    @patch('pathlib.Path.mkdir')
    @patch.object(PylintFixerOrchestrator, 'run_pylint')
    @patch.object(ClaudeExecutor, 'execute_prompt')
    def test_fix_module_success(self, mock_execute, mock_run_pylint, mock_mkdir, mock_open):
        """Test successful module fixing"""
        # Mock file operations
        mock_open.return_value.__enter__ = Mock(return_value=Mock())
        mock_open.return_value.__exit__ = Mock(return_value=None)
        
        # First iteration: has errors
        mock_run_pylint.side_effect = [
            (True, [Mock(error_code="E0602", file_path="test.py", line_number=10, column=5,
                        error_message="Undefined", module="test")]),
            # After fix: no errors
            (True, []),
            # Final check: no errors
            (True, [])
        ]
        
        mock_execute.return_value = (True, "Fixed")
        
        success = self.orchestrator.fix_module("test")
        
        self.assertTrue(success)
        self.assertEqual(len(self.orchestrator.iteration_history), 1)
    
    @patch('builtins.open', create=True)
    @patch('pathlib.Path.mkdir')
    @patch.object(PylintFixerOrchestrator, 'run_pylint')
    def test_fix_module_max_iterations(self, mock_run_pylint, mock_mkdir, mock_open):
        """Test reaching max iterations"""
        # Mock file operations
        mock_open.return_value.__enter__ = Mock(return_value=Mock())
        mock_open.return_value.__exit__ = Mock(return_value=None)
        
        # Create mock errors that reduce each iteration but never fully disappear
        error1 = Mock(error_code="E0602", file_path="test.py", line_number=10, 
                     column=5, error_message="Undefined1", module="test")
        error2 = Mock(error_code="E0602", file_path="test.py", line_number=20,
                     column=5, error_message="Undefined2", module="test")
        error3 = Mock(error_code="E0602", file_path="test.py", line_number=30,
                     column=5, error_message="Undefined3", module="test")
        
        # Add __str__ methods for proper display
        error1.__str__ = Mock(return_value="test.py:10:5: E0602: Undefined1")
        error2.__str__ = Mock(return_value="test.py:20:5: E0602: Undefined2")
        error3.__str__ = Mock(return_value="test.py:30:5: E0602: Undefined3")
        
        # Simulate gradual reduction in errors but never reaching zero
        # Each iteration: initial check, then verify check after fix
        mock_run_pylint.side_effect = [
            # Iteration 1
            (True, [error1, error2, error3]),  # Initial check: 3 errors
            (True, [error1, error2]),           # After fix: 2 errors (progress!)
            # Iteration 2
            (True, [error1, error2]),           # Initial check: 2 errors
            (True, [error1]),                   # After fix: 1 error (progress!)
            # Iteration 3
            (True, [error1]),                   # Initial check: 1 error
            (True, [error1]),                   # After fix: still 1 error (no progress but last iteration)
            # Final check
            (True, [error1])                    # Final errors check
        ]
        
        # Mock Claude to prevent actual API calls
        with patch.object(ClaudeExecutor, 'execute_prompt', return_value=(True, "Fixed")):
            success = self.orchestrator.fix_module("test")
        
        self.assertFalse(success)  # Should fail because errors remain
        # Should have tried max_iterations times
        self.assertEqual(len(self.orchestrator.iteration_history), 3)  # max_iterations=3
    
    def test_get_statistics(self):
        """Test getting statistics"""
        self.orchestrator.iteration_history = [
            {'iteration': 1, 'total_errors': 10, 'error_types': {'E0602': 5, 'E1101': 5}},
            {'iteration': 2, 'total_errors': 5, 'error_types': {'E1101': 5}},
            {'iteration': 3, 'total_errors': 0, 'error_types': {}}
        ]
        
        stats = self.orchestrator.get_statistics()
        
        self.assertEqual(stats['iterations_used'], 3)
        self.assertEqual(stats['initial_errors'], 10)
        self.assertEqual(stats['final_errors'], 0)
        self.assertEqual(stats['errors_fixed'], 10)
        self.assertEqual(stats['success_rate'], 100.0)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPylintErrorParser))
    suite.addTests(loader.loadTestsFromTestCase(TestPromptBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestClaudeExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestPylintFixerOrchestrator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(run_tests())