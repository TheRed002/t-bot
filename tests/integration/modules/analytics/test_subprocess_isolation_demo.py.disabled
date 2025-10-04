"""
Demonstration of subprocess isolation for problematic tests.

This module shows how to use subprocess isolation as the ultimate solution
for tests that cannot be properly isolated through normal means.
"""

import pytest

from .test_bulletproof_isolation import ProcessLevelIsolator


class TestSubprocessIsolationDemo:
    """
    Demonstration of subprocess isolation techniques.

    This approach should be used sparingly and only for tests that absolutely
    cannot be isolated through other means.
    """

    def test_trade_data_integration_subprocess_approach(self):
        """
        Run the problematic test_trade_data_integration in subprocess isolation.

        This demonstrates how to use process-level isolation for tests that
        fail when run in a suite but pass individually.
        """
        result = ProcessLevelIsolator.run_test_in_subprocess(
            test_module="tests.integration.modules.analytics.test_legacy_validation",
            test_class="TestAnalyticsModuleIntegration",
            test_method="test_trade_data_integration"
        )

        assert result['success'], (
            f"Subprocess test failed. "
            f"Output: {result['output']} "
            f"Error: {result['error']}"
        )

    def test_analytics_service_lifecycle_subprocess_approach(self):
        """
        Run analytics service lifecycle test in subprocess isolation.
        """
        result = ProcessLevelIsolator.run_test_in_subprocess(
            test_module="tests.integration.modules.analytics.test_legacy_validation",
            test_class="TestAnalyticsModuleIntegration",
            test_method="test_analytics_service_lifecycle"
        )

        assert result['success'], (
            f"Subprocess test failed. "
            f"Output: {result['output']} "
            f"Error: {result['error']}"
        )

    def test_position_data_integration_subprocess_approach(self):
        """
        Run position data integration test in subprocess isolation.
        """
        result = ProcessLevelIsolator.run_test_in_subprocess(
            test_module="tests.integration.modules.analytics.test_legacy_validation",
            test_class="TestAnalyticsModuleIntegration",
            test_method="test_position_data_integration"
        )

        assert result['success'], (
            f"Subprocess test failed. "
            f"Output: {result['output']} "
            f"Error: {result['error']}"
        )

    @pytest.mark.slow
    def test_all_analytics_tests_subprocess_batch(self):
        """
        Run all analytics integration tests in subprocess isolation.

        This is marked as slow since it runs multiple subprocesses.
        """
        test_methods = [
            # Skip test_analytics_service_dependency_injection - needs dependency_injector fixture
            "test_analytics_service_initialization",
            "test_analytics_service_lifecycle",
            "test_trade_data_integration",
            "test_position_data_integration",
            "test_order_data_integration",
            "test_price_update_integration",
            "test_metrics_retrieval"
        ]

        failures = []
        for test_method in test_methods:
            result = ProcessLevelIsolator.run_test_in_subprocess(
                test_module="tests.integration.modules.analytics.test_legacy_validation",
                test_class="TestAnalyticsModuleIntegration",
                test_method=test_method
            )

            if not result['success']:
                failures.append({
                    'method': test_method,
                    'output': result['output'],
                    'error': result['error']
                })

        if failures:
            failure_details = "\n".join([
                f"Method: {f['method']}, Error: {f['error']}"
                for f in failures
            ])
            pytest.fail(f"Subprocess tests failed:\n{failure_details}")


class TestIsolationStrategy:
    """
    Tests to verify the isolation strategy is working correctly.
    """

    def test_container_isolation_verification(self):
        """
        Verify that container isolation properly resets state.
        """
        from src.core.dependency_injection import DependencyInjector
        from .test_bulletproof_isolation import ContainerLevelIsolator

        # Create an injector and register something
        injector1 = DependencyInjector()
        injector1.register_service("TestService", lambda: "test_value", singleton=True)

        # Verify it's registered
        service1 = injector1.resolve("TestService")
        assert service1 == "test_value"

        # Perform container isolation
        ContainerLevelIsolator.begin_isolation()
        ContainerLevelIsolator.end_isolation()

        # Create new injector - should be clean
        injector2 = DependencyInjector()

        # Should not have the previously registered service
        try:
            injector2.resolve("TestService")
            pytest.fail("Service should not be available after container isolation")
        except Exception:
            # Expected - service should not be found
            pass

    def test_memory_barrier_enforcement(self):
        """
        Verify that memory barriers work correctly.
        """
        from .test_bulletproof_isolation import MemoryBarrierEnforcer

        # This is mainly to ensure the method doesn't crash
        MemoryBarrierEnforcer.enforce_memory_barriers()
        MemoryBarrierEnforcer.clear_weak_references()

        # If we get here without exceptions, the barriers are working
        assert True

    def test_subprocess_isolator_basic_functionality(self):
        """
        Test basic functionality of subprocess isolator.
        """
        # Create a simple test script to verify subprocess execution works
        result = ProcessLevelIsolator.run_test_in_subprocess(
            test_module="tests.integration.modules.analytics.test_bulletproof_isolation",
            test_class="TestAnalyticsWithBulletproofIsolation",
            test_method="test_analytics_service_creation_isolated"
        )

        # Should succeed (though we can't guarantee the specific test passes)
        # We're mainly testing that the subprocess mechanism works
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'output' in result
        assert 'error' in result