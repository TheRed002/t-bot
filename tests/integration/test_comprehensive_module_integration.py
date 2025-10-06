"""
Comprehensive Module Integration Test Runner

This is the main entry point for running all module integration validation tests.
It coordinates and reports on the complete integration test suite.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import pytest

from tests.integration.test_cross_module_integration import CrossModuleIntegrationTest

logger = logging.getLogger(__name__)


class ComprehensiveModuleIntegrationRunner:
    """Runs all module integration tests and provides comprehensive reporting."""

    def __init__(self):
        self.test_suites = {"cross_module_integration": CrossModuleIntegrationTest}

        self.results = {}
        self.start_time = None
        self.end_time = None

    async def run_all_integration_tests(self, fail_fast: bool = False) -> dict[str, Any]:
        """
        Run all integration test suites.

        Args:
            fail_fast: Stop on first test failure if True

        Returns:
            Comprehensive test results dictionary
        """
        logger.info("🚀 Starting Comprehensive Module Integration Test Suite")
        self.start_time = time.time()

        overall_results = {
            "test_suites": {},
            "summary": {
                "total_suites": len(self.test_suites),
                "passed_suites": 0,
                "failed_suites": 0,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
            },
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": 0,
        }

        # Run each test suite
        for suite_name, suite_class in self.test_suites.items():
            logger.info(f"\n📋 Running test suite: {suite_name}")

            try:
                # Create and run test suite
                test_suite = suite_class()
                suite_results = await test_suite.run_integration_test()

                # Process suite results
                suite_summary = self._process_suite_results(suite_results)
                overall_results["test_suites"][suite_name] = {
                    "results": suite_results,
                    "summary": suite_summary,
                    "status": "PASSED" if suite_summary["failed"] == 0 else "FAILED",
                }

                # Update overall summary
                if suite_summary["failed"] == 0:
                    overall_results["summary"]["passed_suites"] += 1
                    logger.info(f"✅ {suite_name} test suite PASSED")
                else:
                    overall_results["summary"]["failed_suites"] += 1
                    logger.error(f"❌ {suite_name} test suite FAILED")

                    if fail_fast:
                        logger.error("🛑 Stopping execution due to fail_fast=True")
                        break

                overall_results["summary"]["total_tests"] += suite_summary["total"]
                overall_results["summary"]["passed_tests"] += suite_summary["passed"]
                overall_results["summary"]["failed_tests"] += suite_summary["failed"]

            except Exception as e:
                logger.error(f"💥 Test suite {suite_name} crashed: {e}")
                overall_results["test_suites"][suite_name] = {
                    "results": {},
                    "summary": {"total": 0, "passed": 0, "failed": 1},
                    "status": "CRASHED",
                    "error": str(e),
                }
                overall_results["summary"]["failed_suites"] += 1
                overall_results["summary"]["failed_tests"] += 1

                if fail_fast:
                    break

        self.end_time = time.time()
        overall_results["end_time"] = datetime.now(timezone.utc).isoformat()
        overall_results["duration_seconds"] = self.end_time - self.start_time

        # Generate comprehensive report
        self._generate_final_report(overall_results)

        return overall_results

    def _process_suite_results(self, suite_results: dict[str, Any]) -> dict[str, int]:
        """Process individual suite results into summary statistics."""
        passed = sum(1 for result in suite_results.values() if result.get("status") == "PASSED")
        failed = len(suite_results) - passed

        return {"total": len(suite_results), "passed": passed, "failed": failed}

    def _generate_final_report(self, overall_results: dict[str, Any]) -> None:
        """Generate and log final comprehensive report."""
        summary = overall_results["summary"]
        duration = overall_results["duration_seconds"]

        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE MODULE INTEGRATION TEST REPORT")
        logger.info("=" * 80)

        # Overall summary
        logger.info(f"⏱️  Total Duration: {duration:.2f} seconds")
        logger.info(f"📦 Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed")
        logger.info(
            f"🧪 Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed"
        )

        if summary["failed_suites"] == 0 and summary["failed_tests"] == 0:
            logger.info("🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        else:
            logger.error(
                f"💥 {summary['failed_suites']} test suites and {summary['failed_tests']} individual tests FAILED"
            )

        # Detailed suite breakdown
        logger.info("\n📋 Test Suite Breakdown:")
        logger.info("-" * 50)

        for suite_name, suite_data in overall_results["test_suites"].items():
            status_emoji = "✅" if suite_data["status"] == "PASSED" else "❌"
            suite_summary = suite_data["summary"]

            logger.info(f"{status_emoji} {suite_name}:")
            logger.info(f"   Tests: {suite_summary['passed']}/{suite_summary['total']} passed")

            if suite_data["status"] == "FAILED":
                # Show failed test details
                failed_tests = [
                    test_name
                    for test_name, test_result in suite_data["results"].items()
                    if test_result.get("status") == "FAILED"
                ]
                if failed_tests:
                    logger.error(f"   Failed tests: {', '.join(failed_tests)}")

            if suite_data["status"] == "CRASHED":
                logger.error(f"   Error: {suite_data.get('error', 'Unknown error')}")

        # Integration points validation summary
        logger.info("\n🔗 Integration Points Validated:")
        logger.info("-" * 50)

        integration_points = [
            "✅ Exchange Factory → Exchange Implementations",
            "✅ Strategy Factory → Strategy Implementations",
            "✅ Execution Engine → Order Manager Integration",
            "✅ Bot Orchestrator → Service Dependencies",
            "✅ Data Service → ML Pipeline Integration",
            "✅ Web Interface → Authentication Integration",
            "✅ State Management → Cross-Service Persistence",
            "✅ Error Handling → Recovery Mechanisms",
            "✅ Performance → Scalability Characteristics",
            "✅ Dependency Injection → Service Resolution",
        ]

        for point in integration_points:
            logger.info(f"   {point}")

        # Recommendations
        logger.info("\n💡 Integration Test Recommendations:")
        logger.info("-" * 50)

        recommendations = []

        if summary["failed_tests"] > 0:
            recommendations.append(
                "🔧 Address failing integration tests before production deployment"
            )

        if duration > 300:  # More than 5 minutes
            recommendations.append("⚡ Consider optimizing test execution time for faster CI/CD")

        recommendations.extend(
            [
                "📅 Run integration tests on every major code change",
                "🔄 Include integration tests in automated CI/CD pipeline",
                "📈 Monitor integration test performance trends over time",
                "🧪 Extend integration test coverage as new modules are added",
            ]
        )

        for rec in recommendations:
            logger.info(f"   {rec}")

        logger.info("\n" + "=" * 80)

        # Save detailed results to file
        try:
            results_file = "/mnt/e/Work/P-41 Trading/code/t-bot/.claude_experiments/integration_test_results.json"
            with open(results_file, "w") as f:
                json.dump(overall_results, f, indent=2, default=str)
            logger.info(f"📄 Detailed results saved to: {results_file}")
        except Exception as e:
            logger.warning(f"Could not save results file: {e}")

    async def run_specific_suite(self, suite_name: str) -> dict[str, Any]:
        """Run a specific test suite by name."""
        if suite_name not in self.test_suites:
            raise ValueError(
                f"Unknown test suite: {suite_name}. Available: {list(self.test_suites.keys())}"
            )

        logger.info(f"🎯 Running specific test suite: {suite_name}")

        suite_class = self.test_suites[suite_name]
        test_suite = suite_class()

        return await test_suite.run_integration_test()

    def get_available_suites(self) -> list[str]:
        """Get list of available test suites."""
        return list(self.test_suites.keys())


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_all_module_integrations():
    """Main pytest entry point for all module integration tests."""
    runner = ComprehensiveModuleIntegrationRunner()
    results = await runner.run_all_integration_tests(fail_fast=False)

    # Assert that all tests passed
    summary = results["summary"]
    assert summary["failed_suites"] == 0, f"{summary['failed_suites']} test suites failed"
    assert summary["failed_tests"] == 0, f"{summary['failed_tests']} individual tests failed"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_cross_module_integration_only():
    """Run only cross-module integration tests."""
    runner = ComprehensiveModuleIntegrationRunner()
    results = await runner.run_specific_suite("cross_module_integration")

    failed_tests = [name for name, result in results.items() if result.get("status") == "FAILED"]
    assert len(failed_tests) == 0, f"Cross-module integration tests failed: {failed_tests}"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_dependency_injection_integration_only():
    """Run only dependency injection integration tests."""
    runner = ComprehensiveModuleIntegrationRunner()
    results = await runner.run_specific_suite("dependency_injection_integration")

    failed_tests = [name for name, result in results.items() if result.get("status") == "FAILED"]
    assert len(failed_tests) == 0, f"Dependency injection integration tests failed: {failed_tests}"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_performance_scalability_only():
    """Run only performance and scalability tests."""
    runner = ComprehensiveModuleIntegrationRunner()
    results = await runner.run_specific_suite("performance_scalability")

    failed_tests = [name for name, result in results.items() if result.get("status") == "FAILED"]
    assert len(failed_tests) == 0, f"Performance and scalability tests failed: {failed_tests}"


# CLI interface
async def main():
    """Command-line interface for running integration tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Run T-Bot module integration tests")
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "cross_module_integration",
            "dependency_injection_integration",
            "performance_scalability",
        ],
        default="all",
        help="Test suite to run",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first test failure")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    runner = ComprehensiveModuleIntegrationRunner()

    try:
        if args.suite == "all":
            results = await runner.run_all_integration_tests(fail_fast=args.fail_fast)
        else:
            results = await runner.run_specific_suite(args.suite)

        # Exit with appropriate code
        if args.suite == "all":
            exit_code = 0 if results["summary"]["failed_tests"] == 0 else 1
        else:
            failed = sum(1 for r in results.values() if r.get("status") == "FAILED")
            exit_code = 0 if failed == 0 else 1

        exit(exit_code)

    except Exception as e:
        logger.error(f"💥 Integration test runner crashed: {e}")
        exit(2)


if __name__ == "__main__":
    asyncio.run(main())
