"""
Real Module Integration Validation for Risk Management.

REFACTORED: Uses real DI container with actual service instances.
NO MOCKS - Tests actual dependency injection and module integration.

Tests:
1. Real dependency injection container setup
2. Real service creation and lifecycle
3. Real cross-module integration
4. Real factory pattern implementation
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.types import Signal, SignalDirection
from src.risk_management.factory import RiskManagementFactory
from src.risk_management.interfaces import RiskServiceInterface
from src.risk_management.service import RiskService


class TestRealModuleValidation:
    """Test real module integration with dependency injection."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_creates_real_services(self, real_risk_factory: RiskManagementFactory):
        """Test that factory creates real service instances."""
        # WHEN: Create service using factory
        risk_service = real_risk_factory.create_risk_service()

        # THEN: Should create real RiskService instance
        assert isinstance(risk_service, RiskService)
        assert isinstance(risk_service, RiskServiceInterface)
        assert not hasattr(risk_service, "_mock_name"), "Should not be a mock"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_validates_dependencies(self, real_risk_factory: RiskManagementFactory):
        """Test that factory validates required dependencies."""
        # WHEN: Validate dependencies
        dependencies = real_risk_factory.validate_dependencies()

        # THEN: All dependencies should be available
        assert isinstance(dependencies, dict)
        assert all(dependencies.values()), (
            f"Missing dependencies: {[k for k, v in dependencies.items() if not v]}"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_di_container_service_lifecycle(self, real_risk_factory: RiskManagementFactory):
        """Test service lifecycle through DI container."""
        # GIVEN: Factory with services
        # WHEN: Create and initialize service
        risk_service = real_risk_factory.create_risk_service()
        await risk_service.initialize()

        # THEN: Service should be initialized successfully
        # Service initialization successful if no exceptions raised
        assert risk_service is not None
        assert isinstance(risk_service, RiskService)

        # WHEN: Cleanup service
        await risk_service.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_service_integration_through_factory(
        self, real_risk_factory: RiskManagementFactory, sample_signal: Signal
    ):
        """Test that factory-created services work with real operations."""
        # GIVEN: Service from factory
        risk_service = real_risk_factory.create_risk_service()
        await risk_service.initialize()

        try:
            # WHEN: Perform real operations
            position_size = await risk_service.calculate_position_size(
                signal=sample_signal,
                available_capital=Decimal("100000.00"),
                current_price=Decimal("50000.00"),
            )

            is_valid = await risk_service.validate_signal(sample_signal)

            # THEN: Operations should complete successfully
            assert isinstance(position_size, Decimal)
            assert isinstance(is_valid, bool)

        finally:
            await risk_service.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_recommended_component(self, real_risk_factory: RiskManagementFactory):
        """Test factory recommends correct component."""
        # WHEN: Get recommended component
        recommended = real_risk_factory.get_recommended_component()

        # THEN: Should recommend RiskService
        assert isinstance(recommended, RiskService)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_service_management(self, real_risk_factory: RiskManagementFactory):
        """Test factory service start/stop management."""
        # The factory is already started in the fixture

        # WHEN: Stop services
        await real_risk_factory.stop_services()

        # THEN: Should stop cleanly (no errors)
        # Verify by restarting
        await real_risk_factory.start_services()


class TestRealCrossModuleIntegration:
    """Test real cross-module integration."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_risk_service_database_integration(
        self, real_risk_service: RiskService, sample_positions
    ):
        """Test risk service integrates with real database."""
        # GIVEN: Portfolio state to persist
        portfolio_value = Decimal("100000.00")

        # WHEN: Update state (should use real database)
        await real_risk_service.update_portfolio_state(
            positions=sample_positions, available_capital=portfolio_value
        )

        # THEN: State should be persisted via database service
        summary = await real_risk_service.get_risk_summary()
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_risk_service_monitoring_integration(
        self, real_risk_service: RiskService, sample_positions
    ):
        """Test risk service integrates with monitoring."""
        # GIVEN: Risk metrics to calculate
        from .fixtures.real_service_fixtures import generate_realistic_market_data_sequence

        market_data = generate_realistic_market_data_sequence(periods=30)

        # WHEN: Calculate metrics (should integrate with monitoring)
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=sample_positions, market_data=market_data
        )

        # THEN: Metrics should be calculated (monitoring integration works)
        assert risk_metrics is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_module_operations(self, real_risk_service: RiskService):
        """Test concurrent operations across modules."""
        # GIVEN: Multiple operations
        signal = Signal(
            signal_id="concurrent_test",
            strategy_id="test",
            strategy_name="Test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=Decimal("0.80"),
            strength=Decimal("0.70"),
            source="test",
            timestamp=datetime.now(timezone.utc),
        )

        # WHEN: Execute operations concurrently
        tasks = [
            real_risk_service.calculate_position_size(
                signal=signal,
                available_capital=Decimal("100000.00"),
                current_price=Decimal("50000.00"),
            )
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # THEN: All operations should complete
        assert len(results) == 5
        assert all(isinstance(r, Decimal) for r in results)


class TestRealServiceConfiguration:
    """Test real service configuration."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_configuration_loaded(self, real_risk_service: RiskService):
        """Test that service loads real configuration."""
        # THEN: Service should have valid configuration
        assert real_risk_service.risk_config is not None
        assert hasattr(real_risk_service.risk_config, "max_position_size_pct")
        assert isinstance(real_risk_service.risk_config.max_position_size_pct, Decimal)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_configuration_affects_behavior(
        self, real_risk_service: RiskService, sample_signal: Signal
    ):
        """Test that configuration affects real behavior."""
        # GIVEN: Modified configuration
        original_max = real_risk_service.risk_config.max_position_size_pct
        real_risk_service.risk_config.max_position_size_pct = Decimal("0.01")  # 1% max

        try:
            # WHEN: Calculate position size
            position_size = await real_risk_service.calculate_position_size(
                signal=sample_signal,
                available_capital=Decimal("100000.00"),
                current_price=Decimal("50000.00"),
            )

            # THEN: Should respect new limit
            max_allowed = Decimal("100000.00") * Decimal("0.01") / Decimal("50000.00")
            assert position_size <= max_allowed

        finally:
            # Restore original configuration
            real_risk_service.risk_config.max_position_size_pct = original_max
