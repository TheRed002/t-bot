"""
Unit tests for FundFlowManager class.

This module tests the deposit/withdrawal management including:
- Deposit processing and validation
- Withdrawal processing with rules
- Strategy reallocation
- Auto-compounding
- Performance tracking
- Capital protection rules
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.capital_management.fund_flow_manager import FundFlowManager
from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.types import FundFlow


class TestFundFlowManager:
    """Test cases for FundFlowManager class."""

    @pytest.fixture
    def config(self):
        """Create test configuration with capital management settings."""
        config = Config()
        config.capital_management.total_capital = 100000.0
        config.capital_management.min_deposit_amount = 1000.0
        config.capital_management.min_withdrawal_amount = 100.0
        config.capital_management.max_withdrawal_pct = 0.2
        config.capital_management.max_daily_reallocation_pct = 0.1
        config.capital_management.fund_flow_cooldown_minutes = 30
        config.capital_management.auto_compound_enabled = True
        config.capital_management.auto_compound_frequency = "daily"
        config.capital_management.profit_threshold = 0.05
        config.capital_management.max_daily_loss_pct = 0.05
        config.capital_management.max_weekly_loss_pct = 0.15
        config.capital_management.max_monthly_loss_pct = 0.25
        config.capital_management.profit_lock_pct = 0.1
        config.capital_management.withdrawal_rules = {
            "emergency": {
                "name": "emergency",
                "description": "Emergency withdrawal rule",
                "enabled": True,
                "max_percentage": 0.5,
                "cooldown_hours": 0,
            },
            "regular": {
                "name": "regular",
                "description": "Regular withdrawal rule",
                "enabled": True,
                "max_percentage": 0.2,
                "cooldown_hours": 24,
            },
        }
        return config

    @pytest.fixture
    def fund_flow_manager(self, config):
        """Create fund flow manager instance."""
        return FundFlowManager(config)

    @pytest.fixture
    def sample_fund_flows(self):
        """Create sample fund flows."""
        return [
            FundFlow(
                from_strategy="strategy_1",
                to_strategy="strategy_2",
                from_exchange="binance",
                to_exchange="okx",
                amount=Decimal("10000"),
                currency="USDT",
                converted_amount=None,
                exchange_rate=None,
                reason="strategy_reallocation",
                timestamp=datetime.now() - timedelta(hours=2),
            ),
            FundFlow(
                from_strategy=None,
                to_strategy="strategy_1",
                from_exchange=None,
                to_exchange="binance",
                amount=Decimal("5000"),
                currency="USDT",
                converted_amount=None,
                exchange_rate=None,
                reason="deposit",
                timestamp=datetime.now() - timedelta(hours=1),
            ),
            FundFlow(
                from_strategy="strategy_1",
                to_strategy=None,
                from_exchange="binance",
                to_exchange=None,
                amount=Decimal("2000"),
                currency="USDT",
                converted_amount=None,
                exchange_rate=None,
                reason="withdrawal",
                timestamp=datetime.now() - timedelta(minutes=30),
            ),
        ]

    def test_initialization(self, fund_flow_manager, config):
        """Test fund flow manager initialization."""
        assert fund_flow_manager.config == config
        assert fund_flow_manager.capital_config == config.capital_management
        assert fund_flow_manager.fund_flows == []
        assert fund_flow_manager.strategy_performance == {}
        # Check if last_compound_date is within the last 60 seconds
        assert (datetime.now() - fund_flow_manager.last_compound_date).total_seconds() < 60
        assert fund_flow_manager.total_profit == Decimal("0")
        assert fund_flow_manager.locked_profit == Decimal("0")
        assert fund_flow_manager.total_capital == Decimal("0")  # Initially 0 until set

    @pytest.mark.asyncio
    async def test_update_total_capital(self, fund_flow_manager):
        """Test updating total capital."""
        total_capital = Decimal("50000")

        await fund_flow_manager.update_total_capital(total_capital)

        assert fund_flow_manager.total_capital == total_capital

    @pytest.mark.asyncio
    async def test_get_total_capital(self, fund_flow_manager):
        """Test getting total capital."""
        total_capital = Decimal("75000")
        fund_flow_manager.total_capital = total_capital

        result = await fund_flow_manager.get_total_capital()

        assert result == total_capital

    @pytest.mark.asyncio
    async def test_process_deposit_basic(self, fund_flow_manager):
        """Test basic deposit processing."""
        amount = Decimal("5000")
        currency = "USDT"
        exchange = "binance"

        result = await fund_flow_manager.process_deposit(amount, currency, exchange)

        assert isinstance(result, FundFlow)
        assert len(fund_flow_manager.fund_flows) == 1

        flow = fund_flow_manager.fund_flows[0]
        assert flow.to_exchange == exchange
        assert flow.amount == amount
        assert flow.currency == currency
        assert flow.reason == "deposit"

    @pytest.mark.asyncio
    async def test_process_deposit_below_minimum(self, fund_flow_manager):
        """Test deposit processing below minimum amount."""
        amount = Decimal("500")  # Below minimum
        currency = "USDT"
        exchange = "binance"

        with pytest.raises(ValidationError):
            await fund_flow_manager.process_deposit(amount, currency, exchange)

    @pytest.mark.asyncio
    async def test_process_deposit_negative_amount(self, fund_flow_manager):
        """Test deposit processing with negative amount."""
        amount = Decimal("-1000")
        currency = "USDT"
        exchange = "binance"

        with pytest.raises(ValidationError):
            await fund_flow_manager.process_deposit(amount, currency, exchange)

    @pytest.mark.asyncio
    async def test_process_withdrawal_basic(self, fund_flow_manager):
        """Test basic withdrawal processing."""
        # Set total capital first
        await fund_flow_manager.update_total_capital(Decimal("100000"))

        amount = Decimal("2000")
        currency = "USDT"
        exchange = "binance"
        reason = "withdrawal"

        result = await fund_flow_manager.process_withdrawal(amount, currency, exchange, reason)

        assert isinstance(result, FundFlow)
        assert len(fund_flow_manager.fund_flows) == 1

        flow = fund_flow_manager.fund_flows[0]
        assert flow.from_exchange == exchange
        assert flow.amount == amount
        assert flow.currency == currency
        assert flow.reason == reason

    @pytest.mark.asyncio
    async def test_process_withdrawal_below_minimum(self, fund_flow_manager):
        """Test withdrawal processing below minimum amount."""
        amount = Decimal("50")  # Below minimum
        currency = "USDT"
        exchange = "binance"
        reason = "withdrawal"

        with pytest.raises(ValidationError):
            await fund_flow_manager.process_withdrawal(amount, currency, exchange, reason)

    @pytest.mark.asyncio
    async def test_process_withdrawal_without_total_capital(self, fund_flow_manager):
        """Test withdrawal processing when total capital is not set."""
        amount = Decimal("2000")
        currency = "USDT"
        exchange = "binance"
        reason = "withdrawal"

        # Should work even without total capital set
        result = await fund_flow_manager.process_withdrawal(amount, currency, exchange, reason)

        assert isinstance(result, FundFlow)

    @pytest.mark.asyncio
    async def test_process_withdrawal_exceeds_max_percentage(self, fund_flow_manager):
        """Test withdrawal processing exceeding max percentage."""
        # Set total capital
        await fund_flow_manager.update_total_capital(Decimal("100000"))

        amount = Decimal("25000")  # 25% of 100000, exceeds 20% max
        currency = "USDT"
        exchange = "binance"
        reason = "withdrawal"

        # Should fail due to exceeding max percentage
        with pytest.raises(ValidationError):
            await fund_flow_manager.process_withdrawal(amount, currency, exchange, reason)

    @pytest.mark.asyncio
    async def test_process_strategy_reallocation(self, fund_flow_manager):
        """Test strategy reallocation processing."""
        from_strategy = "strategy_1"
        to_strategy = "strategy_2"
        amount = Decimal("5000")
        reason = "reallocation"

        result = await fund_flow_manager.process_strategy_reallocation(
            from_strategy, to_strategy, amount, reason
        )

        assert isinstance(result, FundFlow)
        assert len(fund_flow_manager.fund_flows) == 1

        flow = fund_flow_manager.fund_flows[0]
        assert flow.from_strategy == from_strategy
        assert flow.to_strategy == to_strategy
        assert flow.amount == amount
        assert flow.reason == reason

    @pytest.mark.asyncio
    async def test_process_strategy_reallocation_with_total_capital(self, fund_flow_manager):
        """Test strategy reallocation with total capital set."""
        # Set total capital
        await fund_flow_manager.update_total_capital(Decimal("100000"))

        from_strategy = "strategy_1"
        to_strategy = "strategy_2"
        amount = Decimal("5000")
        reason = "reallocation"

        result = await fund_flow_manager.process_strategy_reallocation(
            from_strategy, to_strategy, amount, reason
        )

        assert isinstance(result, FundFlow)

    @pytest.mark.asyncio
    async def test_process_auto_compound(self, fund_flow_manager):
        """Test auto-compounding processing."""

        result = await fund_flow_manager.process_auto_compound()

        # May return None if auto-compounding is not triggered
        if result is not None:
            assert isinstance(result, FundFlow)
            assert result.reason == "auto_compound"

    @pytest.mark.asyncio
    async def test_process_auto_compound_disabled(self, fund_flow_manager, config):
        """Test auto-compounding when disabled."""
        config.capital_management.auto_compound_enabled = False

        result = await fund_flow_manager.process_auto_compound()

        assert result is None
        assert len(fund_flow_manager.fund_flows) == 0

    @pytest.mark.asyncio
    async def test_process_auto_compound_below_threshold(self, fund_flow_manager):
        """Test auto-compounding below profit threshold."""

        result = await fund_flow_manager.process_auto_compound()

        assert result is None
        assert len(fund_flow_manager.fund_flows) == 0

    @pytest.mark.asyncio
    async def test_update_performance(self, fund_flow_manager):
        """Test updating performance metrics."""
        strategy_name = "test_strategy"
        performance_metrics = {"total_pnl": 1000.0, "initial_capital": 10000.0}

        result = await fund_flow_manager.update_performance(strategy_name, performance_metrics)

        assert result is None
        assert strategy_name in fund_flow_manager.strategy_performance
        assert fund_flow_manager.strategy_performance[strategy_name]["total_pnl"] == 1000.0

    @pytest.mark.asyncio
    async def test_update_performance_negative_pnl(self, fund_flow_manager):
        """Test updating performance with negative PnL."""
        strategy_name = "test_strategy"
        performance_metrics = {"total_pnl": -500.0, "initial_capital": 10000.0}

        result = await fund_flow_manager.update_performance(strategy_name, performance_metrics)

        assert result is None
        assert fund_flow_manager.total_profit == Decimal("-500")

    @pytest.mark.asyncio
    async def test_get_flow_history(self, fund_flow_manager):
        """Test getting flow history."""
        # Setup fund flows
        fund_flow_manager.fund_flows = [
            FundFlow(
                from_strategy="strategy_1",
                to_strategy="strategy_2",
                from_exchange="binance",
                to_exchange="okx",
                amount=Decimal("10000"),
                currency="USDT",
                converted_amount=None,
                exchange_rate=None,
                reason="strategy_reallocation",
                timestamp=datetime.now() - timedelta(hours=2),
            ),
            FundFlow(
                from_strategy=None,
                to_strategy="strategy_1",
                from_exchange=None,
                to_exchange="binance",
                amount=Decimal("5000"),
                currency="USDT",
                converted_amount=None,
                exchange_rate=None,
                reason="deposit",
                timestamp=datetime.now() - timedelta(hours=1),
            ),
        ]

        history = await fund_flow_manager.get_flow_history()

        assert len(history) == 2
        assert history[0].reason == "strategy_reallocation"
        assert history[1].reason == "deposit"

    @pytest.mark.asyncio
    async def test_get_flow_history_empty(self, fund_flow_manager):
        """Test getting flow history when empty."""
        history = await fund_flow_manager.get_flow_history()

        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_flow_summary(self, fund_flow_manager):
        """Test getting flow summary."""
        # Setup fund flows
        fund_flow_manager.fund_flows = [
            FundFlow(
                from_strategy=None,
                to_strategy="strategy_1",
                from_exchange=None,
                to_exchange="binance",
                amount=Decimal("10000"),
                currency="USDT",
                converted_amount=None,
                exchange_rate=None,
                reason="deposit",
                timestamp=datetime.now() - timedelta(hours=2),
            ),
            FundFlow(
                from_strategy="strategy_1",
                to_strategy=None,
                from_exchange="binance",
                to_exchange=None,
                amount=Decimal("2000"),
                currency="USDT",
                converted_amount=None,
                exchange_rate=None,
                reason="withdrawal",
                timestamp=datetime.now() - timedelta(hours=1),
            ),
        ]

        summary = await fund_flow_manager.get_flow_summary()

        assert isinstance(summary, dict)
        assert "total_deposits" in summary
        assert "total_withdrawals" in summary
        assert "total_reallocations" in summary
        assert "net_flow" in summary
        assert summary["total_deposits"] == 10000.0
        assert summary["total_withdrawals"] == 2000.0
        assert summary["net_flow"] == 8000.0

    @pytest.mark.asyncio
    async def test_get_capital_protection_status(self, fund_flow_manager):
        """Test getting capital protection status."""
        status = await fund_flow_manager.get_capital_protection_status()

        assert isinstance(status, dict)
        assert "emergency_reserve_pct" in status
        assert "max_daily_loss_pct" in status
        assert "max_weekly_loss_pct" in status
        assert "max_monthly_loss_pct" in status
        assert "profit_lock_pct" in status
        assert "total_profit" in status
        assert "locked_profit" in status
        assert "auto_compound_enabled" in status
        assert "next_compound_date" in status
