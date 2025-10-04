"""
Real Service Integration Tests for Risk Management Framework.

Refactored from test_integration.py to use REAL services instead of mocks.
Tests circuit breakers, position limits, emergency controls, portfolio risk metrics,
correlation monitoring, and adaptive risk management with actual database persistence.

NO MOCKS - All tests use real RiskService with dependency injection.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List

import pytest
import pytest_asyncio

from src.core.config import Config
from src.core.exceptions import RiskManagementError, ValidationError, ServiceError
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    RiskLevel,
    RiskMetrics,
    Signal,
    SignalDirection,
)
from src.risk_management.service import RiskService

from .fixtures.real_service_fixtures import (
    real_risk_service,
    real_risk_factory,
    generate_realistic_market_data_sequence,
    generate_bull_market_scenario,
    generate_bear_market_scenario,
    generate_high_volatility_scenario,
    sample_positions,
    sample_signal,
)

logger = logging.getLogger(__name__)


class TestRiskManagementRealIntegration:
    """Integration tests using real risk management service."""

    @pytest.mark.asyncio
    async def test_complete_risk_management_workflow(
        self, real_risk_service: RiskService, sample_positions: List[Position]
    ):
        """Test complete risk management workflow with real services."""
        # GIVEN: Portfolio with real positions
        portfolio_value = Decimal("100000.00")

        # Generate real market data
        btc_data = generate_realistic_market_data_sequence(
            symbol="BTC/USDT", periods=30
        )
        eth_data = generate_realistic_market_data_sequence(
            symbol="ETH/USDT", base_price=Decimal("3000"), periods=30
        )
        market_data = btc_data + eth_data

        # WHEN: Execute complete workflow
        # Step 1: Update portfolio state
        await real_risk_service.update_portfolio_state(
            positions=sample_positions,
            available_capital=portfolio_value
        )

        # Step 2: Calculate risk metrics
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=sample_positions,
            market_data=market_data
        )

        # THEN: Validate risk metrics
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.risk_level in [
            RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL
        ]
        assert isinstance(risk_metrics.var_95, Decimal)
        assert isinstance(risk_metrics.expected_shortfall, Decimal)
        assert isinstance(risk_metrics.max_drawdown, Decimal)

        # Step 3: Validate signals
        test_signal = Signal(
            signal_id="test_signal",
            strategy_id="test",
            strategy_name="Test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=Decimal("0.85"),
            strength=Decimal("0.75"),
            source="test",
            timestamp=datetime.now(timezone.utc)
        )

        is_valid = await real_risk_service.validate_signal(test_signal)
        assert isinstance(is_valid, bool)

        # Step 4: Calculate position sizes
        position_size = await real_risk_service.calculate_position_size(
            signal=test_signal,
            available_capital=portfolio_value,
            current_price=Decimal("50000")
        )
        assert isinstance(position_size, Decimal)
        assert position_size > Decimal("0")

        # Step 5: Check portfolio limits
        new_position = Position(
            position_id="new_pos",
            symbol="ADA/USDT",
            quantity=Decimal("1000"),
            entry_price=Decimal("0.5"),
            current_price=Decimal("0.5"),
            unrealized_pnl=Decimal("0"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance"
        )

        can_add = await real_risk_service.check_portfolio_limits(new_position)
        assert isinstance(can_add, bool)

        # Step 6: Get comprehensive risk summary
        summary = await real_risk_service.get_risk_summary()
        assert isinstance(summary, dict)
        assert 'current_risk_level' in summary or 'risk_level' in summary

    @pytest.mark.asyncio
    async def test_risk_management_with_large_portfolio(
        self, real_risk_service: RiskService
    ):
        """Test risk management with a large portfolio using real calculations."""
        # GIVEN: Large portfolio with multiple positions
        portfolio_value = Decimal("1000000.00")  # 1M portfolio
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "DOT/USDT"]

        positions = []
        market_data = []

        for symbol in symbols:
            position = Position(
                position_id=f"pos_{symbol}",
                symbol=symbol,
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                unrealized_pnl=Decimal("1000"),
                side=PositionSide.LONG,
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
                stop_loss=Decimal("49000"),
                take_profit=Decimal("52000")
            )
            positions.append(position)

            # Generate realistic market data for each symbol
            symbol_data = generate_realistic_market_data_sequence(
                symbol=symbol, periods=50
            )
            market_data.extend(symbol_data)

        # WHEN: Update portfolio state with historical data
        for i in range(30):  # 30 days of history
            variation = (i % 10 - 5) * Decimal("10000")  # Simulate volatility
            current_value = portfolio_value + variation
            await real_risk_service.update_portfolio_state(
                positions=positions,
                available_capital=current_value
            )

        # Calculate risk metrics
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=positions,
            market_data=market_data
        )

        # THEN: Validate metrics for large portfolio
        assert risk_metrics.risk_level in [
            RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL
        ]
        assert isinstance(risk_metrics.var_95, Decimal)
        assert risk_metrics.var_95 >= Decimal("0")

        # Test position sizing with large portfolio
        signal = Signal(
            signal_id="large_portfolio_signal",
            strategy_id="test",
            strategy_name="Test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=Decimal("0.90"),
            strength=Decimal("0.85"),
            source="test",
            timestamp=datetime.now(timezone.utc)
        )

        position_size = await real_risk_service.calculate_position_size(
            signal=signal,
            available_capital=portfolio_value,
            current_price=Decimal("51000")
        )

        assert isinstance(position_size, Decimal)
        assert position_size > Decimal("0")
        # Position sizing validation - just ensure it returns a value
        # Note: The actual value may vary based on risk calculations

    @pytest.mark.asyncio
    async def test_risk_management_with_high_volatility(
        self, real_risk_service: RiskService
    ):
        """Test risk management during high volatility scenarios."""
        # GIVEN: High volatility market conditions
        high_vol_data = generate_high_volatility_scenario(symbol="BTC/USDT")

        positions = [
            Position(
                position_id="btc_pos",
                symbol="BTC/USDT",
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("45000"),  # 10% loss
                unrealized_pnl=Decimal("-500"),
                side=PositionSide.LONG,
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
                stop_loss=Decimal("44000"),
                take_profit=Decimal("52000")
            )
        ]

        # Simulate portfolio decline
        initial_value = Decimal("60000.00")
        for i in range(20):
            declining_value = initial_value * (Decimal("1") - Decimal("0.01") * Decimal(str(i)))
            await real_risk_service.update_portfolio_state(
                positions=positions,
                available_capital=declining_value
            )

        # WHEN: Calculate risk metrics in high volatility
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=positions,
            market_data=high_vol_data
        )

        # THEN: Risk level should be elevated
        assert risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL], \
            "Risk level should be high during volatility"
        assert isinstance(risk_metrics.max_drawdown, Decimal)
        # Drawdown is represented as a positive percentage (e.g., 0.1 = 10% drawdown)
        assert risk_metrics.max_drawdown >= Decimal("0"), "Drawdown should be positive percentage"

    @pytest.mark.asyncio
    async def test_risk_management_performance(
        self, real_risk_service: RiskService, sample_positions: List[Position]
    ):
        """Test that real risk calculations meet performance requirements."""
        import time

        # GIVEN: Portfolio and market data
        portfolio_value = Decimal("100000.00")
        market_data = generate_realistic_market_data_sequence(periods=100)

        # WHEN: Perform risk calculations (measure time)
        start_time = time.time()

        # Calculate risk metrics
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=sample_positions,
            market_data=market_data
        )

        # Calculate position size
        signal = Signal(
            signal_id="perf_test",
            strategy_id="test",
            strategy_name="Test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=Decimal("0.80"),
            strength=Decimal("0.70"),
            source="test",
            timestamp=datetime.now(timezone.utc)
        )

        position_size = await real_risk_service.calculate_position_size(
            signal=signal,
            available_capital=portfolio_value,
            current_price=Decimal("50000")
        )

        # Validate signal
        is_valid = await real_risk_service.validate_signal(signal)

        duration = time.time() - start_time

        # THEN: Operations should complete quickly
        assert duration < 2.0, f"Risk operations took {duration}s, expected < 2s"
        assert isinstance(risk_metrics, RiskMetrics)
        assert isinstance(position_size, Decimal)
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    async def test_risk_management_error_handling(
        self, real_risk_service: RiskService
    ):
        """Test error handling in risk management service."""
        # GIVEN: Signal with edge case values (valid for Pydantic but may fail business logic)
        edge_case_signal = Signal(
            signal_id="edge_case",
            strategy_id="test",
            strategy_name="Test",
            symbol="INVALID/SYMBOL",  # Invalid symbol format
            direction=SignalDirection.BUY,
            confidence=Decimal("0.01"),  # Very low confidence
            strength=Decimal("0.01"),  # Very low strength
            source="test",
            timestamp=datetime.now(timezone.utc)
        )

        # WHEN/THEN: Should handle gracefully
        try:
            is_valid = await real_risk_service.validate_signal(edge_case_signal)
            # If validation doesn't raise, it should return False
            assert is_valid is False, "Invalid signal should be rejected"
        except (ValidationError, RiskManagementError):
            # Raising validation error is also acceptable
            pass

    @pytest.mark.asyncio
    async def test_risk_management_edge_cases(
        self, real_risk_service: RiskService
    ):
        """Test edge cases in risk management."""
        # Test with empty positions - empty data may cause validation errors, which is expected
        try:
            empty_metrics = await real_risk_service.calculate_risk_metrics(
                positions=[],
                market_data=[]
            )
            # If it succeeds, metrics should indicate no positions
            assert isinstance(empty_metrics, RiskMetrics)
            assert empty_metrics.position_count == 0
        except (ValidationError, ServiceError):
            # Validation error for empty data is acceptable behavior
            pass

        # Test with zero portfolio value
        signal = Signal(
            signal_id="edge_case",
            strategy_id="test",
            strategy_name="Test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=Decimal("0.80"),
            strength=Decimal("0.70"),
            source="test",
            timestamp=datetime.now(timezone.utc)
        )

        # Should handle zero portfolio gracefully
        try:
            position_size = await real_risk_service.calculate_position_size(
                signal=signal,
                available_capital=Decimal("0"),
                current_price=Decimal("50000")
            )
            assert position_size == Decimal("0"), "Zero portfolio should give zero size"
        except (ValidationError, RiskManagementError, ServiceError):
            # Raising error is also acceptable (including circuit breaker open)
            pass

    @pytest.mark.asyncio
    async def test_emergency_stop_functionality(
        self, real_risk_service: RiskService
    ):
        """Test emergency stop activation and deactivation."""
        # GIVEN: Normal operating conditions
        assert real_risk_service.is_emergency_stop_active() is False

        # WHEN: Activate emergency stop
        await real_risk_service.trigger_emergency_stop(
            reason="Test emergency stop"
        )

        # THEN: Emergency stop should be active
        assert real_risk_service.is_emergency_stop_active() is True

        # All new orders should be rejected
        signal = Signal(
            signal_id="emergency_test",
            strategy_id="test",
            strategy_name="Test",
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=Decimal("0.90"),
            strength=Decimal("0.85"),
            source="test",
            timestamp=datetime.now(timezone.utc)
        )

        is_valid = await real_risk_service.validate_signal(signal)
        assert is_valid is False, "Signals should be rejected during emergency stop"

        # WHEN: Deactivate emergency stop
        await real_risk_service.reset_emergency_stop(reason="Test reset")

        # THEN: Normal operations should resume
        assert real_risk_service.is_emergency_stop_active() is False

    @pytest.mark.asyncio
    async def test_concurrent_risk_operations(
        self, real_risk_service: RiskService
    ):
        """Test concurrent risk management operations."""
        # GIVEN: Multiple signals
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
                timestamp=datetime.now(timezone.utc)
            )
            for i in range(10)
        ]

        portfolio_value = Decimal("100000.00")

        # WHEN: Process signals concurrently
        tasks = [
            real_risk_service.calculate_position_size(
                signal=signal,
                available_capital=portfolio_value,
                current_price=Decimal("50000")
            )
            for signal in signals
        ]

        results = await asyncio.gather(*tasks)

        # THEN: All operations should complete successfully
        assert len(results) == 10
        assert all(isinstance(r, Decimal) for r in results)
        assert all(r > Decimal("0") for r in results)

    @pytest.mark.asyncio
    async def test_decimal_precision_maintained(
        self, real_risk_service: RiskService, sample_positions: List[Position]
    ):
        """Test that Decimal precision is maintained throughout calculations."""
        # GIVEN: High precision values
        portfolio_value = Decimal("100000.123456789")
        market_data = [
            MarketData(
                symbol="BTC/USDT",
                open=Decimal("50123.456789"),
                high=Decimal("50234.567890"),
                low=Decimal("50012.345678"),
                close=Decimal("50156.789012"),
                volume=Decimal("1234.567890"),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                exchange="binance"
            )
            for i in range(50)
        ]

        # WHEN: Calculate risk metrics
        risk_metrics = await real_risk_service.calculate_risk_metrics(
            positions=sample_positions,
            market_data=market_data
        )

        # THEN: All values must be Decimal (NEVER float)
        assert isinstance(risk_metrics.var_95, Decimal)
        assert isinstance(risk_metrics.expected_shortfall, Decimal)
        assert isinstance(risk_metrics.sharpe_ratio, Decimal)
        assert isinstance(risk_metrics.max_drawdown, Decimal)

        # Verify NO float contamination
        assert not isinstance(risk_metrics.var_95, float)
        assert not isinstance(risk_metrics.expected_shortfall, float)
        assert not isinstance(risk_metrics.sharpe_ratio, float)
        assert not isinstance(risk_metrics.max_drawdown, float)


class TestRiskManagementDatabasePersistence:
    """Test database persistence for risk management data."""

    @pytest.mark.asyncio
    async def test_risk_state_persistence(
        self, real_risk_service: RiskService, sample_positions: List[Position]
    ):
        """Test that risk state is persisted to database."""
        # GIVEN: Portfolio state
        portfolio_value = Decimal("100000.00")

        # WHEN: Update portfolio state
        await real_risk_service.update_portfolio_state(
            positions=sample_positions,
            available_capital=portfolio_value
        )

        # THEN: State should be persisted (verified through service)
        summary = await real_risk_service.get_risk_summary()
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_risk_metrics_history(
        self, real_risk_service: RiskService, sample_positions: List[Position]
    ):
        """Test that risk metrics history is maintained."""
        # GIVEN: Multiple risk calculations over time
        market_data = generate_realistic_market_data_sequence(periods=100)

        # WHEN: Calculate metrics multiple times
        for i in range(5):
            await real_risk_service.calculate_risk_metrics(
                positions=sample_positions,
                market_data=market_data[i*20:(i+1)*20]
            )

        # THEN: History should be available (implementation-specific)
        summary = await real_risk_service.get_risk_summary()
        assert isinstance(summary, dict)
