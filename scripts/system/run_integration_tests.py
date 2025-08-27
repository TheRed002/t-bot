#!/usr/bin/env python3
"""
Integration Test Runner Script

This script orchestrates the execution of comprehensive integration tests for the T-Bot trading system.
It provides detailed reporting, performance metrics, and test categorization.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Manages execution and reporting of integration tests."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests" / "integration"
        self.results = {}
        self.performance_metrics = {}
        
        # Test categories and their corresponding files
        self.test_categories = {
            "end_to_end": "test_end_to_end_trading_workflow.py",
            "multi_exchange": "test_multi_exchange_integration.py",
            "state_management": "test_state_management_integration.py", 
            "risk_management": "test_risk_management_integration.py",
            "realtime_data": "test_realtime_data_flow_integration.py",
            "security_auth": "test_security_authentication_integration.py",
            "performance_load": "test_performance_load_testing.py"
        }
        
    async def run_tests(
        self, 
        categories: Optional[List[str]] = None,
        verbose: bool = False,
        generate_report: bool = True,
        run_performance_tests: bool = True
    ) -> Dict:
        """Run integration tests with specified options."""
        logger.info("Starting integration test execution")
        
        # Validate categories
        if categories:
            invalid_categories = set(categories) - set(self.test_categories.keys())
            if invalid_categories:
                raise ValueError(f"Invalid test categories: {invalid_categories}")
        else:
            categories = list(self.test_categories.keys())
            
        # Skip performance tests if requested
        if not run_performance_tests:
            categories = [cat for cat in categories if cat != "performance_load"]
            
        logger.info(f"Running test categories: {categories}")
        
        # Execute tests for each category
        total_start_time = time.time()
        
        for category in categories:
            logger.info(f"Running {category} tests...")
            
            category_start_time = time.time()
            result = await self._run_test_category(category, verbose)
            category_duration = time.time() - category_start_time
            
            self.results[category] = {
                **result,
                "duration": category_duration
            }
            
            # Log category results
            status = "PASSED" if result["success"] else "FAILED"
            logger.info(f"{category} tests {status} in {category_duration:.2f}s")
            
            if not result["success"]:
                logger.error(f"Failures in {category}: {result.get('failures', [])}")
                
        total_duration = time.time() - total_start_time
        
        # Compile overall results
        overall_results = {
            "total_duration": total_duration,
            "categories_run": len(categories),
            "categories_passed": sum(1 for r in self.results.values() if r["success"]),
            "categories_failed": sum(1 for r in self.results.values() if not r["success"]),
            "category_results": self.results,
            "performance_metrics": self.performance_metrics
        }
        
        # Generate report if requested
        if generate_report:
            await self._generate_test_report(overall_results)
            
        logger.info(f"Integration tests completed in {total_duration:.2f}s")
        logger.info(f"Categories: {overall_results['categories_passed']}/{overall_results['categories_run']} passed")
        
        return overall_results
        
    async def _run_test_category(self, category: str, verbose: bool) -> Dict:
        """Run tests for a specific category."""
        test_file = self.test_categories[category]
        test_path = self.test_dir / test_file
        
        if not test_path.exists():
            logger.error(f"Test file not found: {test_path}")
            return {
                "success": False,
                "error": f"Test file not found: {test_file}",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0
            }
            
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest", 
            str(test_path),
            "--tb=short",  # Short traceback format
            "-x",  # Stop on first failure for faster feedback
            "--asyncio-mode=auto"  # Async support
        ]
        
        if verbose:
            cmd.append("-v")
            
        # Add performance-specific options for performance tests
        if category == "performance_load":
            cmd.extend([
                "--timeout=300",  # 5 minute timeout for performance tests
                "-s"  # Don't capture output for performance tests
            ])
        else:
            cmd.append("--timeout=60")  # 1 minute timeout for other tests
            
        # Add coverage for non-performance tests
        if category != "performance_load":
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing"
            ])
            
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        # Execute pytest
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute overall timeout
            )
            
            # Parse pytest output
            return self._parse_pytest_result(result, category)
            
        except subprocess.TimeoutExpired:
            logger.error(f"Test category {category} timed out")
            return {
                "success": False,
                "error": "Test execution timed out",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0
            }
        except Exception as e:
            logger.error(f"Error running {category} tests: {e}")
            return {
                "success": False,
                "error": str(e),
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0
            }
            
    def _parse_pytest_result(self, result: subprocess.CompletedProcess, category: str) -> Dict:
        """Parse pytest execution result."""
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
        
        # Initialize result structure
        parsed_result = {
            "success": return_code == 0,
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "warnings": [],
            "failures": []
        }
        
        # Parse test statistics from output
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for test result summary
            if "passed" in line or "failed" in line:
                # Parse patterns like "3 passed, 1 failed in 2.34s"
                if " passed" in line or " failed" in line:
                    try:
                        # Extract numbers
                        words = line.split()
                        for i, word in enumerate(words):
                            if word.isdigit():
                                if i + 1 < len(words):
                                    if words[i + 1] == "passed":
                                        parsed_result["tests_passed"] = int(word)
                                    elif words[i + 1] == "failed":
                                        parsed_result["tests_failed"] = int(word)
                    except (IndexError, ValueError):
                        pass
                        
            # Look for warnings
            if "warning" in line.lower():
                parsed_result["warnings"].append(line)
                
            # Look for failures
            if "FAILED" in line and "::" in line:
                parsed_result["failures"].append(line)
                
        # Calculate total tests run
        parsed_result["tests_run"] = parsed_result["tests_passed"] + parsed_result["tests_failed"]
        
        # Extract performance metrics for performance tests
        if category == "performance_load":
            self.performance_metrics[category] = self._extract_performance_metrics(stdout)
            
        return parsed_result
        
    def _extract_performance_metrics(self, output: str) -> Dict:
        """Extract performance metrics from test output."""
        metrics = {}
        
        # Look for common performance indicators in output
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for timing information
            if "duration:" in line.lower() or "time:" in line.lower():
                try:
                    # Extract duration values
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace('.', '').replace('s', '').isdigit():
                            if 's' in part or i + 1 < len(parts) and 's' in parts[i + 1]:
                                duration = float(part.replace('s', ''))
                                metrics[f"extracted_duration_{len(metrics)}"] = duration
                except (ValueError, IndexError):
                    pass
                    
            # Look for throughput information
            if "rps" in line.lower() or "requests per second" in line.lower():
                try:
                    parts = line.split()
                    for part in parts:
                        if part.replace('.', '').isdigit():
                            throughput = float(part)
                            metrics[f"throughput_rps"] = throughput
                            break
                except (ValueError, IndexError):
                    pass
                    
            # Look for success rates
            if "success rate" in line.lower() or "%" in line:
                try:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            rate = float(part.replace('%', ''))
                            metrics[f"success_rate_percent"] = rate
                            break
                except (ValueError, IndexError):
                    pass
                    
        return metrics
        
    async def _generate_test_report(self, results: Dict):
        """Generate comprehensive test report."""
        report_file = self.project_root / "integration_test_report.json"
        html_report_file = self.project_root / "integration_test_report.html"
        
        # Save JSON report
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Test report saved to: {report_file}")
        
        # Generate HTML report
        html_content = self._generate_html_report(results)
        with open(html_report_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML test report saved to: {html_report_file}")
        
    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML test report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>T-Bot Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ background-color: #d4edda; }}
        .failure {{ background-color: #f8d7da; }}
        .metrics {{ font-family: monospace; background: #f8f9fa; padding: 10px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>T-Bot Integration Test Report</h1>
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Duration: {results['total_duration']:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Categories Run</td>
                <td>{results['categories_run']}</td>
            </tr>
            <tr>
                <td>Categories Passed</td>
                <td>{results['categories_passed']}</td>
            </tr>
            <tr>
                <td>Categories Failed</td>
                <td>{results['categories_failed']}</td>
            </tr>
            <tr>
                <td>Success Rate</td>
                <td>{results['categories_passed']/results['categories_run']*100:.1f}%</td>
            </tr>
        </table>
    </div>
    
    <div class="categories">
        <h2>Category Results</h2>
"""
        
        # Add category results
        for category, result in results['category_results'].items():
            status_class = "success" if result['success'] else "failure"
            status_text = "PASSED" if result['success'] else "FAILED"
            
            html += f"""
        <div class="category {status_class}">
            <h3>{category.replace('_', ' ').title()} - {status_text}</h3>
            <p><strong>Duration:</strong> {result.get('duration', 0):.2f} seconds</p>
            <p><strong>Tests Run:</strong> {result.get('tests_run', 0)}</p>
            <p><strong>Tests Passed:</strong> {result.get('tests_passed', 0)}</p>
            <p><strong>Tests Failed:</strong> {result.get('tests_failed', 0)}</p>
            
            {f'<div class="metrics"><strong>Error:</strong> {result.get("error", "")}</div>' if result.get("error") else ""}
            
            {f'<div class="metrics"><strong>Failures:</strong><br>{"<br>".join(result.get("failures", []))}</div>' if result.get("failures") else ""}
        </div>
"""
        
        # Add performance metrics if available
        if results.get('performance_metrics'):
            html += """
    </div>
    
    <div class="performance">
        <h2>Performance Metrics</h2>
"""
            for category, metrics in results['performance_metrics'].items():
                if metrics:
                    html += f"""
        <div class="category">
            <h3>{category.replace('_', ' ').title()}</h3>
            <div class="metrics">
"""
                    for metric, value in metrics.items():
                        html += f"<strong>{metric}:</strong> {value}<br>"
                    html += """
            </div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        return html


async def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run T-Bot integration tests")
    parser.add_argument(
        "--categories", 
        nargs="*", 
        choices=["end_to_end", "multi_exchange", "state_management", "risk_management", 
                "realtime_data", "security_auth", "performance_load"],
        help="Test categories to run (default: all)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    parser.add_argument("--no-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IntegrationTestRunner(args.project_root)
    
    try:
        # Run tests
        results = await runner.run_tests(
            categories=args.categories,
            verbose=args.verbose,
            generate_report=not args.no_report,
            run_performance_tests=not args.no_performance
        )
        
        # Exit with appropriate code
        if results['categories_failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())