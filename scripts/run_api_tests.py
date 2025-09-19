#!/usr/bin/env python3
"""
API Test Runner Script for T-Bot Trading System.

This script provides a convenient way to run various test suites
for the Web Interface APIs with different configurations.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


class TestRunner:
    """Test runner for Web Interface APIs."""

    def __init__(self):
        """Initialize test runner."""
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        
        # Ensure Python path includes project root
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

    def run_command(self, command: list[str], verbose: bool = False) -> int:
        """Run a shell command and return exit code."""
        if verbose:
            print(f"Running: {' '.join(command)}")
        
        result = subprocess.run(command, cwd=self.project_root)
        return result.returncode

    def run_unit_tests(self, module: str = None, verbose: bool = False, coverage: bool = False):
        """Run unit tests for Web Interface APIs."""
        print("\n" + "="*60)
        print("Running Unit Tests")
        print("="*60)
        
        test_path = str(self.test_dir / "unit" / "test_web_interface")
        
        if module:
            test_path = f"{test_path}/test_api_{module}.py"
            print(f"Testing module: {module}")
        
        command = ["python", "-m", "pytest", test_path]
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=src/web_interface",
                "--cov-report=term-missing",
                "--cov-report=html"
            ])
        
        return self.run_command(command, verbose)

    def run_integration_tests(self, verbose: bool = False):
        """Run integration tests."""
        print("\n" + "="*60)
        print("Running Integration Tests")
        print("="*60)
        
        test_files = [
            "test_web_interface_comprehensive.py",
            "test_web_interface_api_comprehensive.py",
            "test_web_interface_integration.py"
        ]
        
        exit_code = 0
        for test_file in test_files:
            test_path = self.test_dir / "integration" / test_file
            if test_path.exists():
                command = ["python", "-m", "pytest", str(test_path)]
                
                if verbose:
                    command.append("-v")
                
                code = self.run_command(command, verbose)
                if code != 0:
                    exit_code = code
        
        return exit_code

    def run_performance_tests(self, verbose: bool = False):
        """Run performance tests."""
        print("\n" + "="*60)
        print("Running Performance Tests")
        print("="*60)
        
        test_path = str(self.test_dir / "integration" / "test_web_interface_comprehensive.py::TestPerformanceMetrics")
        
        command = ["python", "-m", "pytest", test_path]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, verbose)

    def run_specific_endpoint(self, endpoint: str, verbose: bool = False):
        """Run tests for a specific endpoint."""
        print("\n" + "="*60)
        print(f"Running Tests for Endpoint: {endpoint}")
        print("="*60)
        
        # Map endpoint names to test classes
        endpoint_map = {
            "portfolio": "TestPortfolioMetricsEndpoints",
            "risk": "TestRiskMetricsEndpoints",
            "alerts": "TestAlertEndpoints",
            "capital": "TestCapitalAllocationEndpoints",
            "funds": "TestFundFlowEndpoints",
            "currency": "TestCurrencyManagementEndpoints",
            "data": "TestDataPipelineEndpoints",
            "exchange": "TestExchangeConnectionEndpoints",
            "ml": "TestModelLifecycleEndpoints",
            "websocket": "TestWebSocketHandlers"
        }
        
        test_class = endpoint_map.get(endpoint.lower())
        if not test_class:
            print(f"Unknown endpoint: {endpoint}")
            print(f"Available endpoints: {', '.join(endpoint_map.keys())}")
            return 1
        
        # Find the test file containing this class
        test_files = list(self.test_dir.glob(f"**/test_*.py"))
        
        for test_file in test_files:
            with open(test_file, 'r') as f:
                if test_class in f.read():
                    command = ["python", "-m", "pytest", f"{test_file}::{test_class}"]
                    
                    if verbose:
                        command.append("-v")
                    
                    return self.run_command(command, verbose)
        
        print(f"Test class {test_class} not found")
        return 1

    def run_all_tests(self, verbose: bool = False, coverage: bool = False):
        """Run all API tests."""
        print("\n" + "="*60)
        print("Running All API Tests")
        print("="*60)
        
        command = [
            "python", "-m", "pytest",
            str(self.test_dir / "unit" / "test_web_interface"),
            str(self.test_dir / "integration" / "test_web_interface*"),
        ]
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=src/web_interface",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--cov-report=json"
            ])
        
        return self.run_command(command, verbose)

    def run_failed_tests(self, verbose: bool = False):
        """Run only previously failed tests."""
        print("\n" + "="*60)
        print("Running Failed Tests")
        print("="*60)
        
        command = ["python", "-m", "pytest", "--lf"]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, verbose)

    def generate_coverage_report(self):
        """Generate HTML coverage report."""
        print("\n" + "="*60)
        print("Generating Coverage Report")
        print("="*60)
        
        command = [
            "python", "-m", "pytest",
            str(self.test_dir / "unit" / "test_web_interface"),
            "--cov=src/web_interface",
            "--cov-report=html",
            "--cov-report=term"
        ]
        
        exit_code = self.run_command(command)
        
        if exit_code == 0:
            print(f"\nCoverage report generated: {self.project_root}/htmlcov/index.html")
        
        return exit_code

    def run_with_markers(self, markers: str, verbose: bool = False):
        """Run tests with specific markers."""
        print("\n" + "="*60)
        print(f"Running Tests with Markers: {markers}")
        print("="*60)
        
        command = ["python", "-m", "pytest", "-m", markers]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, verbose)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run API tests for T-Bot Trading System")
    
    parser.add_argument(
        "command",
        choices=[
            "all", "unit", "integration", "performance",
            "endpoint", "failed", "coverage", "quick"
        ],
        help="Test command to run"
    )
    
    parser.add_argument(
        "--module",
        choices=["analytics", "capital", "data", "exchanges", "ml", "websocket"],
        help="Specific module to test (for unit tests)"
    )
    
    parser.add_argument(
        "--endpoint",
        help="Specific endpoint to test"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "-m", "--markers",
        help="Run tests with specific markers"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    exit_code = 0
    
    try:
        if args.command == "all":
            exit_code = runner.run_all_tests(args.verbose, args.coverage)
        
        elif args.command == "unit":
            exit_code = runner.run_unit_tests(args.module, args.verbose, args.coverage)
        
        elif args.command == "integration":
            exit_code = runner.run_integration_tests(args.verbose)
        
        elif args.command == "performance":
            exit_code = runner.run_performance_tests(args.verbose)
        
        elif args.command == "endpoint":
            if not args.endpoint:
                print("Error: --endpoint required for endpoint command")
                exit_code = 1
            else:
                exit_code = runner.run_specific_endpoint(args.endpoint, args.verbose)
        
        elif args.command == "failed":
            exit_code = runner.run_failed_tests(args.verbose)
        
        elif args.command == "coverage":
            exit_code = runner.generate_coverage_report()
        
        elif args.command == "quick":
            # Run a quick smoke test
            print("Running quick smoke tests...")
            exit_code = runner.run_with_markers("not slow", args.verbose)
        
        if args.markers:
            exit_code = runner.run_with_markers(args.markers, args.verbose)
        
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        exit_code = 130
    
    except Exception as e:
        print(f"\nError running tests: {e}")
        exit_code = 1
    
    # Print summary
    print("\n" + "="*60)
    if exit_code == 0:
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
    print("="*60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()