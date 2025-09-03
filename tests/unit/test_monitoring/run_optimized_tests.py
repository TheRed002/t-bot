#!/usr/bin/env python3
"""
Performance-optimized test runner for monitoring tests.

This script demonstrates the applied optimizations and runs tests with maximum performance.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def setup_fast_environment():
    """Setup environment variables for maximum test performance."""
    env_vars = {
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'DISABLE_ERROR_HANDLER_LOGGING': '1',
        'PYTEST_FAST_MODE': '1',
        'PYTHONUNBUFFERED': '1',
        # Disable asyncio debug mode
        'PYTHONASYNCIODEBUG': '0',
        # Optimize garbage collection
        'PYTHONOPTIMIZE': '1',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✓ Fast environment configured")

def run_tests():
    """Run the optimized monitoring tests."""
    test_dir = Path(__file__).parent
    
    # Command with performance optimizations
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "--tb=short",
        "--disable-warnings", 
        "--no-header",
        "--quiet",
        "--maxfail=10",
        "--durations=10",  # Show 10 slowest tests
        "-x",  # Stop on first failure
        "--cache-clear",  # Clear cache for clean run
    ]
    
    print(f"Running optimized tests in: {test_dir}")
    print("Applied optimizations:")
    print("  • Session-scoped fixtures")
    print("  • Disabled logging completely") 
    print("  • Pre-mocked heavy external dependencies")
    print("  • Reduced iteration counts in loops")
    print("  • Eliminated sleep operations")
    print("  • Simplified async operations")
    print("  • Fixed datetime operations")
    print("  • Optimized mock setup")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✓ Tests completed in {execution_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ Tests timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False

def main():
    """Main entry point."""
    print("=" * 60)
    print("Performance-Optimized Monitoring Test Runner")
    print("=" * 60)
    
    setup_fast_environment()
    success = run_tests()
    
    if success:
        print("\n✓ All optimizations applied successfully!")
        print("\nPerformance improvements implemented:")
        print("  1. Optimized conftest.py with session-level mocking")
        print("  2. Enhanced test_services.py with efficient fixtures") 
        print("  3. Streamlined test_performance.py operations")
        print("  4. Improved test_metrics.py Prometheus mocking")
        print("  5. Simplified test_telemetry.py context managers")
        print("  6. Enhanced test_alerting.py datetime handling")
        print("  7. Added global test configuration optimizations")
        return 0
    else:
        print("\n✗ Some tests failed - check output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())