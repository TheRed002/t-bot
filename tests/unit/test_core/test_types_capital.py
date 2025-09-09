"""
Unit tests for core types capital module.

Tests for capital management specific type definitions.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from src.core.types.capital import (
    CapitalFundFlow,
    CapitalCurrencyExposure,
    CapitalExchangeAllocation,
    ExtendedCapitalProtection,
    ExtendedWithdrawalRule,
)


class TestCapitalFundFlow:
    """Test CapitalFundFlow model."""

    def test_capital_fund_flow_creation_minimal(self):
        """Test CapitalFundFlow creation with minimal required fields."""
        fund_flow = CapitalFundFlow(
            amount=Decimal("1000.00"),
            reason="deposit",
            timestamp=datetime.now(timezone.utc),
        )

        assert fund_flow.amount == Decimal("1000.00")
        assert fund_flow.reason == "deposit"
        assert fund_flow.currency == "USDT"  # default value
        assert fund_flow.from_strategy is None
        assert fund_flow.to_strategy is None
        assert fund_flow.metadata == {}

    def test_capital_fund_flow_creation_complete(self):
        """Test CapitalFundFlow creation with all fields."""
        now = datetime.now(timezone.utc)
        fund_flow = CapitalFundFlow(
            from_strategy="strategy_1",
            to_strategy="strategy_2",
            from_exchange="binance",
            to_exchange="coinbase",
            amount=Decimal("5000.00"),
            currency="BTC",
            converted_amount=Decimal("0.1"),
            exchange_rate=Decimal("50000.00"),
            reason="reallocation",
            timestamp=now,
            metadata={"note": "monthly rebalance"},
            fees=Decimal("10.00"),
            fee_amount=Decimal("0.0002"),
        )

        assert fund_flow.from_strategy == "strategy_1"
        assert fund_flow.to_strategy == "strategy_2"
        assert fund_flow.from_exchange == "binance"
        assert fund_flow.to_exchange == "coinbase"
        assert fund_flow.amount == Decimal("5000.00")
        assert fund_flow.currency == "BTC"
        assert fund_flow.converted_amount == Decimal("0.1")
        assert fund_flow.exchange_rate == Decimal("50000.00")
        assert fund_flow.reason == "reallocation"
        assert fund_flow.timestamp == now
        assert fund_flow.metadata["note"] == "monthly rebalance"
        assert fund_flow.fees == Decimal("10.00")
        assert fund_flow.fee_amount == Decimal("0.0002")

    def test_capital_fund_flow_different_reasons(self):
        """Test CapitalFundFlow with different reasons."""
        reasons = ["deposit", "withdrawal", "reallocation", "auto_compound", "currency_conversion"]
        
        for reason in reasons:
            fund_flow = CapitalFundFlow(
                amount=Decimal("1000.00"),
                reason=reason,
                timestamp=datetime.now(timezone.utc),
            )
            assert fund_flow.reason == reason

    def test_capital_fund_flow_precision(self):
        """Test CapitalFundFlow maintains decimal precision."""
        fund_flow = CapitalFundFlow(
            amount=Decimal("1234.56789012"),
            exchange_rate=Decimal("0.00000001"),
            timestamp=datetime.now(timezone.utc),
            reason="test",
        )
        
        assert str(fund_flow.amount) == "1234.56789012"
        assert str(fund_flow.exchange_rate) in ["0.00000001", "1E-8"]


class TestCapitalCurrencyExposure:
    """Test CapitalCurrencyExposure model."""

    def test_currency_exposure_creation_minimal(self):
        """Test CapitalCurrencyExposure creation with minimal fields."""
        exposure = CapitalCurrencyExposure(
            currency="BTC",
            total_exposure=Decimal("2.5"),
            base_currency_equivalent=Decimal("125000.00"),
            exposure_percentage=Decimal("0.25"),
            timestamp=datetime.now(timezone.utc),
        )

        assert exposure.currency == "BTC"
        assert exposure.total_exposure == Decimal("2.5")
        assert exposure.base_currency_equivalent == Decimal("125000.00")
        assert exposure.exposure_percentage == Decimal("0.25")
        assert exposure.hedging_required is False  # default
        assert exposure.hedge_amount == Decimal("0")  # default

    def test_currency_exposure_with_hedging(self):
        """Test CapitalCurrencyExposure with hedging enabled."""
        exposure = CapitalCurrencyExposure(
            currency="ETH",
            total_exposure=Decimal("10.0"),
            base_currency_equivalent=Decimal("30000.00"),
            exposure_percentage=Decimal("0.15"),
            hedging_required=True,
            hedge_amount=Decimal("2.5"),
            timestamp=datetime.now(timezone.utc),
        )

        assert exposure.currency == "ETH"
        assert exposure.hedging_required is True
        assert exposure.hedge_amount == Decimal("2.5")

    def test_currency_exposure_edge_cases(self):
        """Test CapitalCurrencyExposure edge cases."""
        # Zero exposure
        zero_exposure = CapitalCurrencyExposure(
            currency="USDT",
            total_exposure=Decimal("0"),
            base_currency_equivalent=Decimal("0"),
            exposure_percentage=Decimal("0"),
            timestamp=datetime.now(timezone.utc),
        )
        assert zero_exposure.total_exposure == Decimal("0")

        # High precision exposure
        precise_exposure = CapitalCurrencyExposure(
            currency="BTC",
            total_exposure=Decimal("0.00000001"),
            base_currency_equivalent=Decimal("0.0005"),
            exposure_percentage=Decimal("0.0000001"),
            timestamp=datetime.now(timezone.utc),
        )
        assert precise_exposure.total_exposure == Decimal("0.00000001")


class TestCapitalExchangeAllocation:
    """Test CapitalExchangeAllocation model."""

    def test_exchange_allocation_creation_minimal(self):
        """Test CapitalExchangeAllocation creation with minimal fields."""
        allocation = CapitalExchangeAllocation(
            exchange="binance",
            allocated_amount=Decimal("50000.00"),
            available_amount=Decimal("50000.00"),
            last_rebalance=datetime.now(timezone.utc),
        )

        assert allocation.exchange == "binance"
        assert allocation.allocated_amount == Decimal("50000.00")
        assert allocation.available_amount == Decimal("50000.00")
        assert allocation.utilized_amount == Decimal("0")  # default
        assert allocation.utilization_rate == Decimal("0.0")  # default
        assert allocation.liquidity_score == Decimal("0.5")  # default
        assert allocation.fee_efficiency == Decimal("0.5")  # default
        assert allocation.reliability_score == Decimal("0.5")  # default

    def test_exchange_allocation_creation_complete(self):
        """Test CapitalExchangeAllocation creation with all fields."""
        now = datetime.now(timezone.utc)
        allocation = CapitalExchangeAllocation(
            exchange="coinbase",
            allocated_amount=Decimal("100000.00"),
            utilized_amount=Decimal("75000.00"),
            available_amount=Decimal("25000.00"),
            utilization_rate=Decimal("0.75"),
            liquidity_score=Decimal("0.9"),
            fee_efficiency=Decimal("0.8"),
            reliability_score=Decimal("0.95"),
            last_rebalance=now,
        )

        assert allocation.exchange == "coinbase"
        assert allocation.allocated_amount == Decimal("100000.00")
        assert allocation.utilized_amount == Decimal("75000.00")
        assert allocation.available_amount == Decimal("25000.00")
        assert allocation.utilization_rate == Decimal("0.75")
        assert allocation.liquidity_score == Decimal("0.9")
        assert allocation.fee_efficiency == Decimal("0.8")
        assert allocation.reliability_score == Decimal("0.95")
        assert allocation.last_rebalance == now

    def test_exchange_allocation_different_exchanges(self):
        """Test CapitalExchangeAllocation with different exchanges."""
        exchanges = ["binance", "coinbase", "okx", "kraken"]
        
        for exchange in exchanges:
            allocation = CapitalExchangeAllocation(
                exchange=exchange,
                allocated_amount=Decimal("10000.00"),
                available_amount=Decimal("10000.00"),
                last_rebalance=datetime.now(timezone.utc),
            )
            assert allocation.exchange == exchange


class TestExtendedCapitalProtection:
    """Test ExtendedCapitalProtection model."""

    def test_extended_capital_protection_creation_minimal(self):
        """Test ExtendedCapitalProtection creation with minimal fields."""
        protection = ExtendedCapitalProtection(
            protection_id="prot_001",
            min_capital_threshold=Decimal("10000.00"),
            stop_trading_threshold=Decimal("5000.00"),
            reduce_size_threshold=Decimal("7500.00"),
            size_reduction_factor=Decimal("0.5"),
            max_daily_loss=Decimal("500.00"),
            max_weekly_loss=Decimal("2000.00"),
            max_monthly_loss=Decimal("5000.00"),
            emergency_threshold=Decimal("4000.00"),
        )

        assert protection.protection_id == "prot_001"
        assert protection.enabled is True  # default
        assert protection.min_capital_threshold == Decimal("10000.00")
        assert protection.stop_trading_threshold == Decimal("5000.00")
        assert protection.reduce_size_threshold == Decimal("7500.00")
        assert protection.size_reduction_factor == Decimal("0.5")
        assert protection.max_daily_loss == Decimal("500.00")
        assert protection.emergency_threshold == Decimal("4000.00")

        # Test default additional fields
        assert protection.emergency_reserve_pct == Decimal("0.1")
        assert protection.max_daily_loss_pct == Decimal("0.05")
        assert protection.max_weekly_loss_pct == Decimal("0.10")
        assert protection.max_monthly_loss_pct == Decimal("0.20")
        assert protection.profit_lock_pct == Decimal("0.5")
        assert protection.auto_compound_enabled is True

    def test_extended_capital_protection_creation_complete(self):
        """Test ExtendedCapitalProtection creation with all fields."""
        protection = ExtendedCapitalProtection(
            protection_id="prot_002",
            enabled=False,
            min_capital_threshold=Decimal("20000.00"),
            stop_trading_threshold=Decimal("10000.00"),
            reduce_size_threshold=Decimal("15000.00"),
            size_reduction_factor=Decimal("0.3"),
            max_daily_loss=Decimal("1000.00"),
            max_weekly_loss=Decimal("4000.00"),
            max_monthly_loss=Decimal("10000.00"),
            emergency_threshold=Decimal("8000.00"),
            emergency_reserve_pct=Decimal("0.15"),
            max_daily_loss_pct=Decimal("0.03"),
            max_weekly_loss_pct=Decimal("0.08"),
            max_monthly_loss_pct=Decimal("0.15"),
            profit_lock_pct=Decimal("0.7"),
            auto_compound_enabled=False,
        )

        assert protection.protection_id == "prot_002"
        assert protection.enabled is False
        assert protection.emergency_reserve_pct == Decimal("0.15")
        assert protection.max_daily_loss_pct == Decimal("0.03")
        assert protection.profit_lock_pct == Decimal("0.7")
        assert protection.auto_compound_enabled is False

    def test_extended_capital_protection_edge_cases(self):
        """Test ExtendedCapitalProtection edge cases."""
        # Very low thresholds
        low_protection = ExtendedCapitalProtection(
            protection_id="prot_low",
            min_capital_threshold=Decimal("1.00"),
            stop_trading_threshold=Decimal("0.50"),
            reduce_size_threshold=Decimal("0.75"),
            size_reduction_factor=Decimal("0.1"),
            max_daily_loss=Decimal("0.10"),
            max_weekly_loss=Decimal("0.50"),
            max_monthly_loss=Decimal("1.00"),
            emergency_threshold=Decimal("0.25"),
        )
        assert low_protection.min_capital_threshold == Decimal("1.00")


class TestExtendedWithdrawalRule:
    """Test ExtendedWithdrawalRule model."""

    def test_extended_withdrawal_rule_creation_minimal(self):
        """Test ExtendedWithdrawalRule creation with minimal fields."""
        rule = ExtendedWithdrawalRule(
            name="basic_rule",
        )

        assert rule.name == "basic_rule"
        assert rule.description == ""  # default
        assert rule.enabled is True  # default
        assert rule.threshold is None  # default
        assert rule.min_amount is None  # default
        assert rule.max_percentage is None  # default
        assert rule.cooldown_hours is None  # default

    def test_extended_withdrawal_rule_creation_complete(self):
        """Test ExtendedWithdrawalRule creation with all fields."""
        rule = ExtendedWithdrawalRule(
            name="profit_withdrawal",
            description="Withdraw profits when threshold reached",
            enabled=True,
            threshold=Decimal("5000.00"),
            min_amount=Decimal("1000.00"),
            max_percentage=Decimal("0.20"),
            cooldown_hours=24,
        )

        assert rule.name == "profit_withdrawal"
        assert rule.description == "Withdraw profits when threshold reached"
        assert rule.enabled is True
        assert rule.threshold == Decimal("5000.00")
        assert rule.min_amount == Decimal("1000.00")
        assert rule.max_percentage == Decimal("0.20")
        assert rule.cooldown_hours == 24

    def test_extended_withdrawal_rule_disabled(self):
        """Test ExtendedWithdrawalRule when disabled."""
        rule = ExtendedWithdrawalRule(
            name="disabled_rule",
            enabled=False,
            threshold=Decimal("1000.00"),
        )

        assert rule.name == "disabled_rule"
        assert rule.enabled is False
        assert rule.threshold == Decimal("1000.00")

    def test_extended_withdrawal_rule_edge_cases(self):
        """Test ExtendedWithdrawalRule edge cases."""
        # Zero threshold
        zero_rule = ExtendedWithdrawalRule(
            name="zero_threshold",
            threshold=Decimal("0"),
        )
        assert zero_rule.threshold == Decimal("0")

        # Very long cooldown
        long_cooldown_rule = ExtendedWithdrawalRule(
            name="long_cooldown",
            cooldown_hours=8760,  # 1 year
        )
        assert long_cooldown_rule.cooldown_hours == 8760

    def test_extended_withdrawal_rule_validation_scenarios(self):
        """Test ExtendedWithdrawalRule various validation scenarios."""
        # High percentage
        high_pct_rule = ExtendedWithdrawalRule(
            name="high_percentage",
            max_percentage=Decimal("0.95"),
        )
        assert high_pct_rule.max_percentage == Decimal("0.95")

        # Very small minimum amount
        small_min_rule = ExtendedWithdrawalRule(
            name="small_min",
            min_amount=Decimal("0.01"),
        )
        assert small_min_rule.min_amount == Decimal("0.01")


class TestCapitalTypesIntegration:
    """Test integration between different capital types."""

    def test_fund_flow_with_exchange_allocation(self):
        """Test fund flow integration with exchange allocation."""
        # Create exchange allocation
        allocation = CapitalExchangeAllocation(
            exchange="binance",
            allocated_amount=Decimal("100000.00"),
            utilized_amount=Decimal("50000.00"),
            available_amount=Decimal("50000.00"),
            last_rebalance=datetime.now(timezone.utc),
        )

        # Create fund flow that could affect this allocation
        fund_flow = CapitalFundFlow(
            to_exchange="binance",
            amount=Decimal("10000.00"),
            reason="deposit",
            timestamp=datetime.now(timezone.utc),
        )

        # Verify they can work together
        assert fund_flow.to_exchange == allocation.exchange
        assert fund_flow.amount <= allocation.available_amount

    def test_protection_with_withdrawal_rule(self):
        """Test capital protection integration with withdrawal rules."""
        protection = ExtendedCapitalProtection(
            protection_id="prot_001",
            min_capital_threshold=Decimal("10000.00"),
            stop_trading_threshold=Decimal("5000.00"),
            reduce_size_threshold=Decimal("7500.00"),
            size_reduction_factor=Decimal("0.5"),
            max_daily_loss=Decimal("500.00"),
            max_weekly_loss=Decimal("2000.00"),
            max_monthly_loss=Decimal("5000.00"),
            emergency_threshold=Decimal("4000.00"),
        )

        rule = ExtendedWithdrawalRule(
            name="emergency_withdrawal",
            threshold=protection.emergency_threshold,
            max_percentage=Decimal("0.5"),
        )

        # Verify they are compatible
        assert rule.threshold == protection.emergency_threshold

    def test_currency_exposure_calculations(self):
        """Test currency exposure calculations work correctly."""
        btc_exposure = CapitalCurrencyExposure(
            currency="BTC",
            total_exposure=Decimal("2.0"),
            base_currency_equivalent=Decimal("100000.00"),
            exposure_percentage=Decimal("0.5"),
            timestamp=datetime.now(timezone.utc),
        )

        eth_exposure = CapitalCurrencyExposure(
            currency="ETH",
            total_exposure=Decimal("20.0"),
            base_currency_equivalent=Decimal("60000.00"),
            exposure_percentage=Decimal("0.3"),
            timestamp=datetime.now(timezone.utc),
        )

        # Total exposure should be reasonable
        total_base_equivalent = btc_exposure.base_currency_equivalent + eth_exposure.base_currency_equivalent
        total_percentage = btc_exposure.exposure_percentage + eth_exposure.exposure_percentage

        assert total_base_equivalent == Decimal("160000.00")
        assert total_percentage == Decimal("0.8")

    def test_decimal_precision_across_types(self):
        """Test that all capital types maintain decimal precision consistently."""
        high_precision = Decimal("123.456789012345")
        
        # Test CapitalFundFlow
        fund_flow = CapitalFundFlow(
            amount=high_precision,
            reason="test",
            timestamp=datetime.now(timezone.utc),
        )
        assert fund_flow.amount == high_precision

        # Test CapitalCurrencyExposure
        exposure = CapitalCurrencyExposure(
            currency="TEST",
            total_exposure=high_precision,
            base_currency_equivalent=high_precision * 1000,
            exposure_percentage=high_precision / 1000,
            timestamp=datetime.now(timezone.utc),
        )
        assert exposure.total_exposure == high_precision

        # Test CapitalExchangeAllocation
        allocation = CapitalExchangeAllocation(
            exchange="test",
            allocated_amount=high_precision,
            available_amount=high_precision,
            last_rebalance=datetime.now(timezone.utc),
        )
        assert allocation.allocated_amount == high_precision