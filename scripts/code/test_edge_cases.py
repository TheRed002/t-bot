#!/usr/bin/env python3
"""
Comprehensive edge case testing for pylint-fixer
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pylint_fixer_orchestrator_v2 import PylintFixerOrchestratorV2
from pylint_error_parser import PylintErrorParser
from prompt_builder import PromptBuilder
from claude_executor_v2 import ClaudeExecutorV2


def test_empty_module():
    """Test with a module that has no Python files"""
    print("\n=== TEST: Empty Module ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty module
        module_path = Path(tmpdir) / "src" / "empty_module"
        module_path.mkdir(parents=True)
        
        orch = PylintFixerOrchestratorV2(project_root=Path(tmpdir))
        success, errors = orch.run_pylint("empty_module")
        
        assert success == True, "Should succeed with empty module"
        assert len(errors) == 0, "Should have no errors"
        print("✅ Empty module handled correctly")


def test_module_with_subdirs():
    """Test with a module that has subdirectories"""
    print("\n=== TEST: Module with Subdirectories ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create module with subdirs
        module_path = Path(tmpdir) / "src" / "nested_module"
        module_path.mkdir(parents=True)
        
        # Create files at different levels
        (module_path / "root.py").write_text("x = 1")
        (module_path / "sub1").mkdir()
        (module_path / "sub1" / "file1.py").write_text("y = undefined_var")
        (module_path / "sub1" / "sub2").mkdir()
        (module_path / "sub1" / "sub2" / "file2.py").write_text("z = another_undefined")
        
        orch = PylintFixerOrchestratorV2(project_root=Path(tmpdir))
        success, errors = orch.run_pylint("nested_module", recursive=True)
        
        assert success == True, "Should succeed with nested module"
        assert len(errors) >= 2, f"Should find undefined variables, got {len(errors)}"
        print(f"✅ Found {len(errors)} errors in nested structure")


def test_non_python_files():
    """Test with a module containing non-Python files"""
    print("\n=== TEST: Non-Python Files ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "src" / "mixed_module"
        module_path.mkdir(parents=True)
        
        # Create mixed files
        (module_path / "code.py").write_text("valid = 1")
        (module_path / "README.md").write_text("# Documentation")
        (module_path / "data.json").write_text('{"key": "value"}')
        (module_path / "config.yaml").write_text("setting: true")
        
        orch = PylintFixerOrchestratorV2(project_root=Path(tmpdir))
        success, errors = orch.run_pylint("mixed_module")
        
        assert success == True, "Should handle mixed file types"
        print("✅ Non-Python files ignored correctly")


def test_large_error_count():
    """Test with a file containing many errors"""
    print("\n=== TEST: Large Error Count ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "src" / "error_heavy"
        module_path.mkdir(parents=True)
        
        # Create file with many errors
        code = ""
        for i in range(50):
            code += f"var_{i} = undefined_{i}\n"  # 50 undefined variables
            code += f"obj_{i}.nonexistent_{i}\n"  # 50 no-member errors
        
        (module_path / "many_errors.py").write_text(code)
        
        parser = PylintErrorParser()
        orch = PylintFixerOrchestratorV2(project_root=Path(tmpdir), batch_size=10)
        success, errors = orch.run_pylint("error_heavy")
        
        assert success == True, "Should handle many errors"
        print(f"✅ Handled {len(errors)} errors successfully")


def test_prompt_size_limits():
    """Test prompt size limitations"""
    print("\n=== TEST: Prompt Size Limits ===")
    
    # Test with huge prompt
    executor = ClaudeExecutorV2(max_prompt_size=1000)
    huge_prompt = "x" * 2000
    
    success, output = executor.execute_prompt(huge_prompt, "Test huge prompt")
    
    assert success == False, "Should reject oversized prompt"
    assert "exceeds maximum size" in output.lower() or "too large" in output.lower(), f"Should mention size issue, got: {output}"
    print("✅ Prompt size limits enforced")


def test_timeout_handling():
    """Test timeout handling (mock)"""
    print("\n=== TEST: Timeout Handling ===")
    
    with patch('subprocess.run') as mock_run:
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired('cmd', 1)
        
        executor = ClaudeExecutorV2(timeout=1)
        success, output = executor.execute_prompt("test", "Test timeout")
        
        assert success == False, "Should handle timeout"
        assert "timed out" in output.lower(), "Should mention timeout"
        print("✅ Timeout handled correctly")


def test_file_permissions():
    """Test handling of permission errors"""
    print("\n=== TEST: File Permissions ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "src" / "restricted"
        module_path.mkdir(parents=True)
        
        test_file = module_path / "test.py"
        test_file.write_text("x = 1")
        
        # Make file read-only
        os.chmod(test_file, 0o444)
        
        try:
            orch = PylintFixerOrchestratorV2(project_root=Path(tmpdir))
            success, errors = orch.run_pylint("restricted")
            
            # Should still be able to read and analyze
            assert success == True, "Should read read-only files"
            print("✅ Read-only files handled")
        finally:
            # Restore permissions for cleanup
            os.chmod(test_file, 0o644)


def test_no_progress_detection():
    """Test detection of no progress across iterations"""
    print("\n=== TEST: No Progress Detection ===")
    
    orch = PylintFixerOrchestratorV2(max_iterations=5)
    
    # Mock run_pylint to always return same errors
    persistent_error = Mock()
    persistent_error.error_code = "E0602"
    persistent_error.file_path = "test.py"
    persistent_error.__str__ = Mock(return_value="test.py:1:1: E0602: Undefined")
    
    with patch.object(orch, 'run_pylint', return_value=(True, [persistent_error])):
        with patch.object(orch.claude, 'execute_prompt', return_value=(True, "Fixed")):
            success = orch.fix_module("test_module", dry_run=False)
            
            assert success == False, "Should fail when no progress"
            print("✅ No-progress detection works")


def test_path_traversal_safety():
    """Test that path traversal attempts are handled safely"""
    print("\n=== TEST: Path Traversal Safety ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = PylintFixerOrchestratorV2(project_root=Path(tmpdir))
        
        # Try various path traversal attempts
        dangerous_paths = [
            "../../../etc/passwd",
            "../../sensitive",
            "/etc/passwd",
            "C:\\Windows\\System32",
        ]
        
        for path in dangerous_paths:
            success, errors = orch.run_pylint(path)
            assert success == False, f"Should reject dangerous path: {path}"
        
        print("✅ Path traversal attempts blocked")


def test_memory_cleanup():
    """Test that resources are properly cleaned up"""
    print("\n=== TEST: Memory Cleanup ===")
    
    import gc
    import tracemalloc
    
    tracemalloc.start()
    
    # Create and destroy multiple orchestrators
    for i in range(5):
        orch = PylintFixerOrchestratorV2()
        parser = PylintErrorParser()
        builder = PromptBuilder()
        executor = ClaudeExecutorV2()
        
        # Use them briefly
        parser.parse_content("test")
        builder.get_module_context("test")
        
        # Delete references
        del orch, parser, builder, executor
    
    # Force garbage collection
    gc.collect()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Memory should be reasonable (less than 10MB for these simple operations)
    assert current < 10_000_000, f"Memory usage too high: {current / 1_000_000:.1f}MB"
    print(f"✅ Memory usage reasonable: {current / 1_000_000:.2f}MB")


def test_unicode_handling():
    """Test handling of Unicode in error messages"""
    print("\n=== TEST: Unicode Handling ===")
    
    parser = PylintErrorParser()
    
    # Test with Unicode in error messages
    unicode_output = """************* Module test
test.py:1:1: E0602: Undefined variable '变量' (undefined-variable)
test.py:2:1: E0602: Undefined variable 'café' (undefined-variable)
test.py:3:1: E0602: Undefined variable '🔥' (undefined-variable)"""
    
    errors = parser.parse_content(unicode_output)
    
    assert len(errors) == 3, "Should parse Unicode errors"
    print("✅ Unicode handled correctly")


def test_concurrent_safety():
    """Test that multiple instances don't interfere"""
    print("\n=== TEST: Concurrent Safety ===")
    
    import threading
    
    results = []
    
    def run_orch(name):
        try:
            orch = PylintFixerOrchestratorV2()
            # Each instance should be independent
            orch.iteration_history.append({"name": name})
            results.append(len(orch.iteration_history))
        except Exception as e:
            results.append(f"Error: {e}")
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=run_orch, args=(f"thread_{i}",))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Each should have exactly 1 item in history
    assert all(r == 1 for r in results), f"Concurrent instances interfered: {results}"
    print("✅ Concurrent instances are independent")


def run_all_tests():
    """Run all edge case tests"""
    print("\n" + "="*60)
    print("COMPREHENSIVE EDGE CASE TESTING")
    print("="*60)
    
    tests = [
        test_empty_module,
        test_module_with_subdirs,
        test_non_python_files,
        test_large_error_count,
        test_prompt_size_limits,
        test_timeout_handling,
        test_file_permissions,
        test_no_progress_detection,
        test_path_traversal_safety,
        test_memory_cleanup,
        test_unicode_handling,
        test_concurrent_safety,
    ]
    
    failed = []
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"❌ {test.__name__}: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"❌ {test.__name__}: Unexpected error: {e}")
            failed.append(test.__name__)
    
    print("\n" + "="*60)
    if failed:
        print(f"❌ {len(failed)} tests failed: {', '.join(failed)}")
        return 1
    else:
        print("✅ ALL EDGE CASE TESTS PASSED!")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())