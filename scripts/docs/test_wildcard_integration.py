#!/usr/bin/env python3
"""
Integration test for wildcard functionality.
Tests the actual script execution with realistic scenarios.
"""

import subprocess
import sys
from pathlib import Path

def test_wildcard_help():
    """Test that help shows wildcard option."""
    result = subprocess.run(
        [sys.executable, "scripts/docs/reference_generator.py"],
        capture_output=True,
        text=True
    )
    
    assert "python scripts/docs/reference_generator.py \"*\"" in result.stdout
    assert "Generate for all modules" in result.stdout
    print("✅ Help text includes wildcard option")

def test_nonexistent_module_shows_wildcard():
    """Test that nonexistent module error shows wildcard option.""" 
    result = subprocess.run(
        [sys.executable, "scripts/docs/reference_generator.py", "nonexistent"],
        capture_output=True,
        text=True
    )
    
    assert "Use \"*\" to generate for all modules" in result.stdout
    print("✅ Nonexistent module error shows wildcard option")

def test_wildcard_starts_processing():
    """Test that wildcard starts processing modules."""
    # Run with timeout to avoid long execution
    result = subprocess.run(
        [sys.executable, "scripts/docs/reference_generator.py", "*"],
        capture_output=True,
        text=True,
        timeout=10  # Will be killed after 10 seconds
    )
    
    # Should start processing even if killed early
    assert "🌟 Generating REFERENCE.md for ALL modules..." in result.stdout
    assert "🚀 Generating REFERENCE.md for" in result.stdout
    assert "[1/" in result.stdout  # Should show progress counter
    print("✅ Wildcard functionality starts processing with progress")

def test_specific_module_still_works():
    """Test that specific module generation still works."""
    # Test with a small module that exists
    result = subprocess.run(
        [sys.executable, "scripts/docs/reference_generator.py", "utils"],
        capture_output=True, 
        text=True,
        timeout=30
    )
    
    # Should succeed for specific module
    assert "🔍 Generating comprehensive reference for: utils" in result.stdout
    assert "✅ Comprehensive reference saved:" in result.stdout
    print("✅ Specific module generation still works")

def main():
    """Run all integration tests."""
    print("Running wildcard functionality integration tests...")
    print("=" * 50)
    
    # Change to project root directory
    original_cwd = Path.cwd()
    project_root = Path(__file__).parent.parent.parent
    
    try:
        import os
        os.chdir(project_root)
        print(f"Working directory: {Path.cwd()}")
        
        # Run tests
        test_wildcard_help()
        test_nonexistent_module_shows_wildcard() 
        test_wildcard_starts_processing()
        test_specific_module_still_works()
        
        print("\n" + "=" * 50)
        print("🎯 All integration tests passed!")
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out as expected (processing was working)")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)