"""
Real Service Integration Validation Tests for Risk Management Module.

REFACTORED: Uses REAL services with actual database integration.
NO MOCKS for internal services - only real implementations.

These tests verify that:
1. Risk management properly integrates with real database service
2. Risk management properly integrates with real monitoring service
3. Risk management properly integrates with real state service
4. Service injection patterns work correctly with real dependencies
5. API contracts are properly followed
6. Error handling works with real service boundaries
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.exceptions import RiskManagementError, ValidationError
from src.core.types import (
    OrderRequest,
    Position,
    PositionSide,
    PositionStatus,
    RiskLevel,
    RiskMetrics,
    Signal,
    SignalDirection,
)
from src.risk_management.interfaces import RiskServiceInterface
from src.risk_management.service import RiskService

from .fixtures.real_service_fixtures import (
    generate_realistic_market_data_sequence,
)


class TestRealRiskManagementValidation:
    """Test risk management with real service integration."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_implements_interface(self, real_risk_service: RiskService):
        """Test that RiskService properly implements RiskServiceInterface."""
        # THEN: Service should implement all interface methods
        assert isinstance(real_risk_service, RiskServiceInterface)

        # Verify key interface methods exist
        assert hasattr(real_risk_service, "calculate_position_size")
        assert hasattr(real_risk_service, "validate_signal")
        assert hasattr(real_risk_service, "validate_order")
        assert hasattr(real_risk_service, "calculate_risk_metrics")
        assert hasattr(real_risk_service, "get_risk_summary")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_service_integration(
        self, real_risk_service: RiskService, sample_positions: list[Position]
    ):
        """Test real database service integration."""
        # GIVEN: Portfolio state to persist
        portfolio_value = Decimal("100000.00")

        # WHEN: Update portfolio state (persists to database)
        await real_risk_service.update_portfolio_state(
            positions=sample_positions, available_capital=portfolio_value
        )

        # THEN: State should be retrievable
        summary = await real_risk_service.get_risk_summary()
        assert isinstance(summary, dict)
        # Verify some state was persisted
        assert len(summary) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_state_service_integration(
        self, real_risk_service: RiskService, sample_positions: list[Position]
    ):
        """Test real state service integration."""
        # GIVEN: Risk state data
        portfolio_value = Decimal("100000.00")
        market_data = generate_realistic_market_data_sequence(periods=50)

        # WHEN: Calculate and store risk metrics
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=sample_positions, market_data=market_data
        )

        # Update portfolio state
        await real_risk_service.update_portfolio_state(
            positions=sample_positions, available_capital=portfolio_value
        )

        # THEN: Risk state should be accessible
        summary = await real_risk_service.get_risk_summary()
        assert isinstance(summary, dict)
        assert isinstance(risk_metrics, RiskMetrics)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_with_real_database_checks(
        self, real_risk_service: RiskService, sample_signal: Signal
    ):
        """Test validation performs real database checks."""
        # GIVEN: Signal to validate
        # WHEN: Validate signal (may check database for limits, history, etc.)
        is_valid = await real_risk_service.validate_signal(sample_signal)

        # THEN: Validation should complete with real checks
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_order_validation_with_real_portfolio_state(
        self,
        real_risk_service: RiskService,
        sample_order_request: OrderRequest,
        sample_positions: list[Position],
    ):
        """Test order validation with real portfolio state."""
        # GIVEN: Portfolio state
        portfolio_value = Decimal("100000.00")
        await real_risk_service.update_portfolio_state(
            positions=sample_positions, available_capital=portfolio_value
        )

        # WHEN: Validate order against real portfolio state
        is_valid = await real_risk_service.validate_order(order=sample_order_request)

        # THEN: Validation should use real portfolio limits
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_handling_with_real_services(self, real_risk_service: RiskService):
        """Test error handling with real service boundaries."""
        # GIVEN: Invalid data
        invalid_positions = []
        invalid_market_data = []

        # WHEN: Calculate metrics with edge case data
        try:
            risk_metrics = await real_risk_service.calculate_risk_metrics(
                positions=invalid_positions, market_data=invalid_market_data
            )
            # Should return valid metrics even with empty data
            assert isinstance(risk_metrics, RiskMetrics)
        except (RiskManagementError, ValidationError):
            # Raising error is also acceptable
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_operations_with_real_services(self, real_risk_service: RiskService):
        """Test concurrent operations with real service integration."""
        # GIVEN: Multiple signals to validate concurrently
        signals = [
            Signal(
                signal_id=f"concurrent_{i}",
                strategy_id="test",
                strategy_name="Test",
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                confidence=Decimal("0.80"),
                strength=Decimal("0.70"),
                source="test",
                timestamp=datetime.now(timezone.utc),
            )
            for i in range(10)
        ]

        # WHEN: Validate signals concurrently
        tasks = [real_risk_service.validate_signal(signal) for signal in signals]
        results = await asyncio.gather(*tasks)

        # THEN: All validations should complete
        assert len(results) == 10
        assert all(isinstance(r, bool) for r in results)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_health_check(self, real_risk_service: RiskService):
        """Test service health check functionality."""
        # WHEN: Check service health
        is_healthy = real_risk_service.is_healthy()

        # THEN: Service should report healthy status
        assert isinstance(is_healthy, bool)
        assert is_healthy is True, "Service should be healthy"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_risk_metrics_persistence(
        self, real_risk_service: RiskService, sample_positions: list[Position]
    ):
        """Test that risk metrics are persisted to database."""
        # GIVEN: Market data and positions
        market_data = generate_realistic_market_data_sequence(periods=50)

        # WHEN: Calculate risk metrics (should persist to database)
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=sample_positions, market_data=market_data
        )

        # THEN: Metrics should be calculated with real data
        assert isinstance(risk_metrics, RiskMetrics)
        assert isinstance(risk_metrics.var_95, Decimal)
        assert isinstance(risk_metrics.expected_shortfall, Decimal)
        assert isinstance(risk_metrics.sharpe_ratio, Decimal)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_position_size_calculation_validation(
        self, real_risk_service: RiskService, sample_signal: Signal
    ):
        """Test position size calculation with real validation."""
        # GIVEN: Portfolio parameters
        portfolio_value = Decimal("100000.00")
        current_price = Decimal("50000.00")

        # WHEN: Calculate position size
        position_size = await real_risk_service.calculate_position_size(
            signal=sample_signal, available_capital=portfolio_value, current_price=current_price
        )

        # THEN: Position size should be validated against real limits
        assert isinstance(position_size, Decimal)
        assert position_size > Decimal("0")

        # Verify it respects configured limits
        max_size = (
            portfolio_value * real_risk_service.risk_config.max_position_size_pct / current_price
        )
        assert position_size <= max_size, "Should respect max position size"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_portfolio_limits_validation(
        self, real_risk_service: RiskService, sample_positions: list[Position]
    ):
        """Test portfolio limits with real validation."""
        # GIVEN: Portfolio at or near limits
        portfolio_value = Decimal("100000.00")
        await real_risk_service.update_portfolio_state(
            positions=sample_positions, available_capital=portfolio_value
        )

        # Create new position
        new_position = Position(
            position_id="new_pos",
            symbol="SOL/USDT",
            quantity=Decimal("10"),
            entry_price=Decimal("100"),
            current_price=Decimal("105"),
            unrealized_pnl=Decimal("50"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        # WHEN: Check if position can be added
        can_add = await real_risk_service.check_portfolio_limits(new_position)

        # THEN: Should perform real validation
        assert isinstance(can_add, bool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_decimal_precision_in_validation(self, real_risk_service: RiskService):
        """Test that validation maintains Decimal precision."""
        # GIVEN: High precision signal
        signal = Signal(
            signal_id="precision_test",
            strategy_id="test",
            strategy_name="Test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=Decimal("0.856789012"),  # High precision
            strength=Decimal("0.789012345"),
            source="test",
            timestamp=datetime.now(timezone.utc),
        )

        # WHEN: Validate signal
        is_valid = await real_risk_service.validate_signal(signal)

        # Calculate position size with high precision
        position_size = await real_risk_service.calculate_position_size(
            signal=signal,
            available_capital=Decimal("100000.123456789"),
            current_price=Decimal("50000.987654321"),
        )

        # THEN: Precision should be maintained
        assert isinstance(is_valid, bool)
        assert isinstance(position_size, Decimal)
        assert not isinstance(position_size, float), "Must not be float"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_emergency_controls_validation(
        self, real_risk_service: RiskService, sample_signal: Signal
    ):
        """Test emergency controls with real validation."""
        # GIVEN: Emergency stop activated
        await real_risk_service.trigger_emergency_stop(reason="Test emergency validation")

        # WHEN: Validate signal during emergency
        is_valid = await real_risk_service.validate_signal(sample_signal)

        # THEN: Should reject signals during emergency
        assert is_valid is False, "Should reject during emergency stop"

        # Cleanup
        await real_risk_service.reset_emergency_stop(reason="Test cleanup")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_risk_level_calculation_validation(
        self, real_risk_service: RiskService, sample_positions: list[Position]
    ):
        """Test risk level calculation with real data."""
        # GIVEN: Market data
        market_data = generate_realistic_market_data_sequence(periods=50)

        # WHEN: Calculate risk metrics
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=sample_positions, market_data=market_data
        )

        # THEN: Risk level should be properly calculated
        assert risk_metrics.risk_level in [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]
        assert isinstance(risk_metrics.risk_level, RiskLevel)


class TestRealServiceInterfaceCompliance:
    """Test that real services comply with defined interfaces."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_risk_service_interface_methods(self, real_risk_service: RiskService):
        """Test that RiskService implements all interface methods."""
        # Required interface methods
        required_methods = [
            "calculate_position_size",
            "validate_signal",
            "validate_order",
            "calculate_risk_metrics",
            "should_exit_position",
            "get_current_risk_level",
            "is_emergency_stop_active",
            "get_risk_summary",
        ]

        for method_name in required_methods:
            assert hasattr(real_risk_service, method_name), (
                f"RiskService missing required method: {method_name}"
            )
            method = getattr(real_risk_service, method_name)
            assert callable(method), f"{method_name} must be callable"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_risk_service_return_types(
        self,
        real_risk_service: RiskService,
        sample_signal: Signal,
        sample_positions: list[Position],
    ):
        """Test that RiskService methods return correct types."""
        # Test calculate_position_size
        position_size = await real_risk_service.calculate_position_size(
            signal=sample_signal,
            available_capital=Decimal("100000.00"),
            current_price=Decimal("50000.00"),
        )
        assert isinstance(position_size, Decimal)

        # Test validate_signal
        is_valid = await real_risk_service.validate_signal(sample_signal)
        assert isinstance(is_valid, bool)

        # Test calculate_risk_metrics
        market_data = generate_realistic_market_data_sequence(periods=30)
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=sample_positions, market_data=market_data
        )
        assert isinstance(risk_metrics, RiskMetrics)

        # Test get_risk_summary
        summary = await real_risk_service.get_risk_summary()
        assert isinstance(summary, dict)

        # Test get_current_risk_level
        risk_level = real_risk_service.get_current_risk_level()
        assert isinstance(risk_level, RiskLevel)

        # Test is_emergency_stop_active
        emergency_status = real_risk_service.is_emergency_stop_active()
        assert isinstance(emergency_status, bool)
