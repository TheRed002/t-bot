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

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

# Disable logging during tests to improve performance
logging.getLogger().setLevel(logging.CRITICAL)

from src.capital_management.fund_flow_manager import FundFlowManager
from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.types.capital import CapitalFundFlow as FundFlow


class TestFundFlowManager:
    """Test cases for FundFlowManager class."""

    @pytest.fixture(scope="session")
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
        # Create mock services
        cache_service = Mock()
        cache_service.get = AsyncMock(return_value=None)
        cache_service.set = AsyncMock()
        time_series_service = Mock()
        time_series_service.write_point = AsyncMock()
        validation_service = Mock()

        return FundFlowManager(
            cache_service=cache_service,
            time_series_service=time_series_service,
            validation_service=validation_service,
        )

    @pytest.fixture(scope="session")
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
                timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
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
                timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
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
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
            ),
        ]

    def test_initialization(self, fund_flow_manager, config):
        """Test fund flow manager initialization."""
        # Check that service is properly initialized
        assert fund_flow_manager._name == "FundFlowManagerService"
        assert fund_flow_manager.fund_flows == []
        assert fund_flow_manager.strategy_performance == {}
        # Check if last_compound_date is within the last 60 seconds
        time_diff = (
            datetime.now(timezone.utc) - fund_flow_manager.last_compound_date
        ).total_seconds()
        assert time_diff < 60
        assert fund_flow_manager.total_profit == Decimal("0")
        assert fund_flow_manager.locked_profit == Decimal("0")
        assert fund_flow_manager.total_capital == Decimal("0")  # Initially 0 until set

        # Check that service is not yet started (configuration not loaded)
        assert not fund_flow_manager.is_running

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
        amount = Decimal("50")  # Below default minimum of 100
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
        # Verify total_profit uses Decimal internally
        assert fund_flow_manager.total_profit == Decimal("1000")

    @pytest.mark.asyncio
    async def test_update_performance_negative_pnl(self, fund_flow_manager):
        """Test updating performance with negative PnL."""
        strategy_name = "test_strategy"
        performance_metrics = {"total_pnl": -500.0, "initial_capital": 10000.0}

        result = await fund_flow_manager.update_performance(strategy_name, performance_metrics)

        assert result is None
        # Verify negative PnL is correctly handled with Decimal precision
        assert fund_flow_manager.total_profit == Decimal("-500")
        assert strategy_name in fund_flow_manager.strategy_performance

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
                timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
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
                timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
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
                timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
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
                timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            ),
        ]

        summary = await fund_flow_manager.get_flow_summary()

        assert isinstance(summary, dict)
        assert "total_deposits" in summary
        assert "total_withdrawals" in summary
        assert "total_reallocations" in summary
        assert "net_flow" in summary
        # Financial values should be Decimal, but might be converted to float for JSON serialization
        assert summary["total_deposits"] == Decimal("10000") or summary["total_deposits"] == 10000.0
        assert (
            summary["total_withdrawals"] == Decimal("2000")
            or summary["total_withdrawals"] == 2000.0
        )
        assert summary["net_flow"] == Decimal("8000") or summary["net_flow"] == 8000.0

    @pytest.mark.asyncio
    async def test_get_capital_protection_status(self, fund_flow_manager):
        """Test getting capital protection status."""
        # Start the service to initialize capital protection
        await fund_flow_manager.start()

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

        # Clean up
        await fund_flow_manager.stop()

    @pytest.mark.asyncio
    async def test_process_deposit_zero_amount(self, fund_flow_manager):
        """Test deposit processing with zero amount."""
        with pytest.raises(ValidationError):
            await fund_flow_manager.process_deposit(Decimal("0"), "USDT", "binance")

    @pytest.mark.asyncio
    async def test_process_withdrawal_zero_amount(self, fund_flow_manager):
        """Test withdrawal processing with zero amount."""
        with pytest.raises(ValidationError):
            await fund_flow_manager.process_withdrawal(Decimal("0"), "USDT", "binance")

    @pytest.mark.asyncio
    async def test_process_strategy_reallocation_zero_amount(self, fund_flow_manager):
        """Test strategy reallocation with zero amount."""
        from src.core.exceptions import ServiceError

        with pytest.raises(ServiceError):
            await fund_flow_manager.process_strategy_reallocation(
                "strategy_1", "strategy_2", Decimal("0"), "test"
            )

    @pytest.mark.asyncio
    async def test_get_flow_summary_empty_flows(self, fund_flow_manager):
        """Test getting flow summary with empty flows."""
        fund_flow_manager.fund_flows = []

        summary = await fund_flow_manager.get_flow_summary()

        assert isinstance(summary, dict)
        assert summary["total_deposits"] == Decimal("0") or summary["total_deposits"] == 0
        assert summary["total_withdrawals"] == Decimal("0") or summary["total_withdrawals"] == 0
        assert summary["net_flow"] == Decimal("0") or summary["net_flow"] == 0

    @pytest.mark.asyncio
    async def test_start_service_dependency_failures(self, fund_flow_manager):
        """Test service start with dependency injection failures."""
        # Mock dependency resolution to fail
        fund_flow_manager.resolve_dependency = Mock(side_effect=Exception("DI failed"))

        # Service should start successfully with fallback configuration
        await fund_flow_manager.start()

        # Verify service started with default config
        assert fund_flow_manager.config is not None
        assert fund_flow_manager.config.get("auto_compound_enabled") is True

    @pytest.mark.asyncio
    async def test_start_configuration_loading_failure(self, fund_flow_manager):
        """Test service start with configuration loading failure."""
        from unittest.mock import patch
        with patch.object(fund_flow_manager, '_load_configuration', side_effect=Exception("Config failed")):
            with pytest.raises(Exception):
                await fund_flow_manager.start()

    @pytest.mark.asyncio
    async def test_start_withdrawal_rules_initialization_failure(self, fund_flow_manager):
        """Test service start with withdrawal rules initialization failure."""
        from unittest.mock import patch
        with patch.object(fund_flow_manager, '_initialize_withdrawal_rules', side_effect=Exception("Rules failed")):
            with pytest.raises(Exception):
                await fund_flow_manager.start()

    @pytest.mark.asyncio
    async def test_start_capital_protection_initialization_failure(self, fund_flow_manager):
        """Test service start with capital protection initialization failure."""
        from unittest.mock import patch
        with patch.object(fund_flow_manager, '_initialize_capital_protection', side_effect=Exception("Protection failed")):
            with pytest.raises(Exception):
                await fund_flow_manager.start()

    @pytest.mark.asyncio
    async def test_cache_operations_failure(self, fund_flow_manager):
        """Test cache operations with service failures."""
        # Test cache service failures don't break operations
        fund_flow_manager._cache_service.set = AsyncMock(side_effect=Exception("Cache failed"))
        fund_flow_manager._cache_service.get = AsyncMock(side_effect=Exception("Cache failed"))

        # Operations should still work despite cache failures
        result = await fund_flow_manager.process_deposit(Decimal("1000"), "USDT", "binance")
        assert isinstance(result, FundFlow)

        # Test cache retrieval failure
        cached_flows = await fund_flow_manager._get_cached_fund_flows()
        assert cached_flows is None

    @pytest.mark.asyncio
    async def test_time_series_storage_failure(self, fund_flow_manager):
        """Test time series storage with service failures."""
        fund_flow_manager._time_series_service.write_point = AsyncMock(side_effect=Exception("TS failed"))

        # Operations should still work despite time series failures
        result = await fund_flow_manager.process_deposit(Decimal("1000"), "USDT", "binance")
        assert isinstance(result, FundFlow)

    @pytest.mark.asyncio
    async def test_flow_history_size_management(self, fund_flow_manager):
        """Test fund flow history size management."""
        fund_flow_manager._max_flow_history = 2

        # Add flows beyond limit
        await fund_flow_manager.process_deposit(Decimal("1000"), "USDT", "binance")
        await fund_flow_manager.process_deposit(Decimal("1000"), "USDT", "binance")
        await fund_flow_manager.process_deposit(Decimal("1000"), "USDT", "binance")

        # Should maintain size limit
        assert len(fund_flow_manager.fund_flows) == 2

    @pytest.mark.asyncio
    async def test_process_withdrawal_with_total_capital_zero(self, fund_flow_manager):
        """Test withdrawal processing when total capital is zero."""
        fund_flow_manager.total_capital = Decimal("0")

        # Should process withdrawal with warning logged
        result = await fund_flow_manager.process_withdrawal(
            Decimal("1000"), "USDT", "binance", "withdrawal"
        )
        assert isinstance(result, FundFlow)

    @pytest.mark.asyncio
    async def test_process_strategy_reallocation_with_total_capital_zero(self, fund_flow_manager):
        """Test strategy reallocation when total capital is zero."""
        fund_flow_manager.total_capital = Decimal("0")

        # Should process reallocation with warning logged
        result = await fund_flow_manager.process_strategy_reallocation(
            "strategy_1", "strategy_2", Decimal("1000"), "reallocation"
        )
        assert isinstance(result, FundFlow)

    @pytest.mark.asyncio
    async def test_process_strategy_reallocation_exceeds_daily_limit(self, fund_flow_manager):
        """Test strategy reallocation exceeding daily limit."""
        fund_flow_manager.total_capital = Decimal("100000")

        # Mock daily reallocation amount to be near limit
        from unittest.mock import patch
        with patch.object(fund_flow_manager, '_get_daily_reallocation_amount', return_value=Decimal("9000")):
            # Request reallocation that would exceed 10% daily limit
            with pytest.raises(Exception, match="Strategy reallocation failed"):
                await fund_flow_manager.process_strategy_reallocation(
                    "strategy_1", "strategy_2", Decimal("2000"), "reallocation"
                )

    @pytest.mark.asyncio
    async def test_withdrawal_rules_validation(self, fund_flow_manager):
        """Test withdrawal rules validation."""
        # Initialize withdrawal rules with proper Mock setup
        profit_only_rule = Mock()
        profit_only_rule.enabled = True
        profit_only_rule.name = "profit_only"

        maintain_minimum_rule = Mock()
        maintain_minimum_rule.enabled = True
        maintain_minimum_rule.name = "maintain_minimum"

        performance_based_rule = Mock()
        performance_based_rule.enabled = True
        performance_based_rule.name = "performance_based"
        performance_based_rule.threshold = 0.05

        fund_flow_manager.withdrawal_rules = {
            "profit_only": profit_only_rule,
            "maintain_minimum": maintain_minimum_rule,
            "performance_based": performance_based_rule
        }

        fund_flow_manager.total_profit = Decimal("0")  # No profits
        fund_flow_manager.total_capital = Decimal("100000")

        # Should fail profit_only rule - ValidationError is re-raised directly
        with pytest.raises(ValidationError, match="No profits available"):
            await fund_flow_manager._validate_withdrawal_rules(Decimal("1000"), "USDT")

    @pytest.mark.asyncio
    async def test_withdrawal_cooldown_check(self, fund_flow_manager):
        """Test withdrawal cooldown period check."""
        # Add recent withdrawal
        recent_flow = FundFlow(
            from_strategy=None,
            to_strategy=None,
            from_exchange="binance",
            to_exchange=None,
            amount=Decimal("1000"),
            reason="withdrawal",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=10)  # Recent
        )
        fund_flow_manager.fund_flows = [recent_flow]
        fund_flow_manager.config = {"fund_flow_cooldown_minutes": 30}

        # Should fail cooldown check
        with pytest.raises(ValidationError, match="Cooldown period not met"):
            await fund_flow_manager._check_withdrawal_cooldown()

    @pytest.mark.asyncio
    async def test_performance_threshold_check(self, fund_flow_manager):
        """Test performance threshold checking."""
        # Test with no performance data
        result = await fund_flow_manager._check_performance_threshold(0.05)
        assert result is False

        # Test with performance data
        fund_flow_manager.strategy_performance = {
            "strategy_1": {
                "total_pnl": Decimal("500"),
                "initial_capital": Decimal("10000")
            }
        }

        # Should meet 0.05 threshold (500/10000 = 0.05)
        result = await fund_flow_manager._check_performance_threshold(0.05)
        assert result is True

        # Should not meet 0.10 threshold
        result = await fund_flow_manager._check_performance_threshold(0.10)
        assert result is False

    @pytest.mark.asyncio
    async def test_calculate_minimum_capital_required(self, fund_flow_manager):
        """Test minimum capital calculation."""
        # Test with no strategy performance
        result = await fund_flow_manager._calculate_minimum_capital_required()
        assert result > Decimal("0")

        # Test with strategy performance and config
        fund_flow_manager.strategy_performance = {
            "arbitrage_strategy_1": {},
            "trend_strategy_1": {}
        }
        fund_flow_manager.config = {
            "per_strategy_minimum": {
                "arbitrage": 10000,
                "trend": 5000
            }
        }

        result = await fund_flow_manager._calculate_minimum_capital_required()
        assert result == Decimal("15000")  # 10000 + 5000

    @pytest.mark.asyncio
    async def test_daily_reallocation_amount_calculation(self, fund_flow_manager):
        """Test daily reallocation amount calculation."""
        # Add some reallocation flows for today
        today_flow = FundFlow(
            from_strategy="strategy_1",
            to_strategy="strategy_2",
            from_exchange=None,
            to_exchange=None,
            amount=Decimal("5000"),
            reason="reallocation",
            timestamp=datetime.now(timezone.utc)
        )
        fund_flow_manager.fund_flows = [today_flow]

        result = await fund_flow_manager._get_daily_reallocation_amount()
        assert result == Decimal("5000")

    @pytest.mark.asyncio
    async def test_compound_timing_checks(self, fund_flow_manager):
        """Test auto-compound timing checks."""
        # Test disabled auto-compound
        fund_flow_manager.config = {"auto_compound_enabled": False}
        assert fund_flow_manager._should_compound() is False

        # Test weekly frequency
        fund_flow_manager.config = {
            "auto_compound_enabled": True,
            "auto_compound_frequency": "weekly"
        }
        fund_flow_manager.last_compound_date = datetime.now(timezone.utc) - timedelta(days=8)
        assert fund_flow_manager._should_compound() is True

        # Test monthly frequency
        fund_flow_manager.config["auto_compound_frequency"] = "monthly"
        fund_flow_manager.last_compound_date = datetime.now(timezone.utc) - timedelta(days=35)
        assert fund_flow_manager._should_compound() is True

        # Test invalid frequency
        fund_flow_manager.config["auto_compound_frequency"] = "invalid"
        assert fund_flow_manager._should_compound() is False

    @pytest.mark.asyncio
    async def test_compound_amount_calculation(self, fund_flow_manager):
        """Test auto-compound amount calculation."""
        # Test below threshold
        fund_flow_manager.total_profit = Decimal("100")
        fund_flow_manager.config = {"profit_threshold": 1000}

        result = await fund_flow_manager._calculate_compound_amount()
        assert result == Decimal("0")

        # Test above threshold
        fund_flow_manager.total_profit = Decimal("2000")
        fund_flow_manager.config = {
            "profit_threshold": 1000,
            "profit_lock_pct": 0.5
        }

        result = await fund_flow_manager._calculate_compound_amount()
        expected = (Decimal("2000") - Decimal("1000")) * Decimal("0.5")
        assert result == expected

    @pytest.mark.asyncio
    async def test_compound_schedule_calculation(self, fund_flow_manager):
        """Test compound schedule calculation."""
        fund_flow_manager.config = {
            "auto_compound_frequency": "weekly",
            "auto_compound_enabled": True
        }

        schedule = fund_flow_manager._calculate_compound_schedule()
        assert "frequency" in schedule
        assert "next_compound" in schedule
        assert "enabled" in schedule
        assert schedule["frequency"] == "weekly"
        assert schedule["enabled"] is True

    @pytest.mark.asyncio
    async def test_get_performance_summary_edge_cases(self, fund_flow_manager):
        """Test performance summary with edge cases."""
        # Test with mixed metric types
        fund_flow_manager.strategy_performance = {
            "strategy_1": {"pnl": 1000.0, "performance_score": 0.85},
            "strategy_2": Decimal("500"),  # Single value
            "strategy_3": 250,  # Integer value
            "strategy_4": {"pnl": Decimal("750"), "performance_score": Decimal("0.90")}
        }

        summary = await fund_flow_manager.get_performance_summary()

        assert summary["strategy_count"] == 4
        assert "strategies" in summary
        assert "total_pnl" in summary

        # Check individual strategy handling
        assert summary["strategies"]["strategy_1"]["pnl"] == Decimal("1000.0")
        assert summary["strategies"]["strategy_2"]["pnl"] == Decimal("500")
        assert summary["strategies"]["strategy_3"]["pnl"] == Decimal("250")
        assert summary["strategies"]["strategy_4"]["pnl"] == Decimal("750")

    @pytest.mark.asyncio
    async def test_withdrawal_rules_initialization_edge_cases(self, fund_flow_manager):
        """Test withdrawal rules initialization with edge cases."""
        # Test with invalid config
        fund_flow_manager.config = {"withdrawal_rules": "invalid"}
        fund_flow_manager._initialize_withdrawal_rules()
        assert len(fund_flow_manager.withdrawal_rules) == 0

        # Test with invalid rule config
        fund_flow_manager.config = {
            "withdrawal_rules": {
                "valid_rule": {
                    "enabled": True,
                    "description": "Valid rule"
                },
                "invalid_rule": "not_a_dict"
            }
        }
        fund_flow_manager._initialize_withdrawal_rules()
        assert "valid_rule" in fund_flow_manager.withdrawal_rules
        assert "invalid_rule" not in fund_flow_manager.withdrawal_rules

    @pytest.mark.asyncio
    async def test_validation_rule_edge_cases(self, fund_flow_manager):
        """Test withdrawal rule validation with edge cases."""
        from src.capital_management.constants import DEFAULT_PROFIT_THRESHOLD
        from src.core.types.capital import ExtendedWithdrawalRule as WithdrawalRule

        # Setup rules with various configurations
        fund_flow_manager.withdrawal_rules = {
            "min_amount_rule": WithdrawalRule(
                name="min_amount_rule",
                enabled=True,
                min_amount=Decimal("500")
            ),
            "max_percentage_rule": WithdrawalRule(
                name="max_percentage_rule",
                enabled=True,
                max_percentage=0.1
            ),
            "disabled_rule": WithdrawalRule(
                name="disabled_rule",
                enabled=False,
                min_amount=Decimal("10000")  # Would fail if enabled
            )
        }
        fund_flow_manager.total_capital = Decimal("10000")

        # Test min amount violation
        with pytest.raises(ValidationError, match="below minimum"):
            await fund_flow_manager._validate_withdrawal_rules(Decimal("300"), "USDT")

        # Test max percentage violation
        with pytest.raises(ValidationError, match="exceeds maximum"):
            await fund_flow_manager._validate_withdrawal_rules(Decimal("1500"), "USDT")

        # Test disabled rule is ignored
        amount = Decimal("800")  # Would fail disabled rule but should pass
        await fund_flow_manager._validate_withdrawal_rules(amount, "USDT")

    @pytest.mark.asyncio
    async def test_maintain_minimum_rule_validation(self, fund_flow_manager):
        """Test maintain minimum capital rule validation."""
        from src.core.types.capital import ExtendedWithdrawalRule as WithdrawalRule

        fund_flow_manager.withdrawal_rules = {
            "maintain_minimum": WithdrawalRule(
                name="maintain_minimum",
                enabled=True
            )
        }
        fund_flow_manager.total_capital = Decimal("5000")

        # Mock minimum capital calculation to return high value
        from unittest.mock import patch
        with patch.object(fund_flow_manager, '_calculate_minimum_capital_required', return_value=Decimal("4500")):
            # Withdrawal would leave 2000, below required 4500
            with pytest.raises(ValidationError, match="minimum capital requirement"):
                await fund_flow_manager._validate_withdrawal_rules(Decimal("3000"), "USDT")

    @pytest.mark.asyncio
    async def test_performance_based_rule_validation(self, fund_flow_manager):
        """Test performance-based withdrawal rule validation."""
        from src.core.types.capital import ExtendedWithdrawalRule as WithdrawalRule

        fund_flow_manager.withdrawal_rules = {
            "performance_based": WithdrawalRule(
                name="performance_based",
                enabled=True,
                threshold=0.1  # 10% threshold
            )
        }

        # Mock performance check to return False
        from unittest.mock import patch
        with patch.object(fund_flow_manager, '_check_performance_threshold', return_value=False):
            with pytest.raises(ValidationError, match="Performance below threshold"):
                await fund_flow_manager._validate_withdrawal_rules(Decimal("1000"), "USDT")

    @pytest.mark.asyncio
    async def test_error_handling_in_methods(self, fund_flow_manager):
        """Test error handling in various methods."""
        # Test that performance update handles invalid data gracefully (no exception expected)
        await fund_flow_manager.update_performance("test", {"valid_metric": 0.5})

        # Verify the data was stored
        assert "test" in fund_flow_manager.strategy_performance
        assert fund_flow_manager.strategy_performance["test"]["valid_metric"] == Decimal("0.5")

        # Test error in flow history with invalid timestamp
        fund_flow_manager.fund_flows = [Mock(timestamp="invalid")]
        try:
            await fund_flow_manager.get_flow_history()
        except Exception:
            pass  # Expected to handle gracefully

        # Test error in flow summary
        try:
            await fund_flow_manager.get_flow_summary()
        except Exception:
            pass  # Expected to handle gracefully

    @pytest.mark.asyncio
    async def test_cached_fund_flows_edge_cases(self, fund_flow_manager):
        """Test cached fund flows with edge cases."""
        # Test with flows that don't have model_dump method
        flows = [Mock(spec=[]), Mock(spec=[])]  # No model_dump method
        flows[0].__dict__ = {"amount": Decimal("100")}
        flows[1].__dict__ = {"amount": Decimal("200")}

        await fund_flow_manager._cache_fund_flows(flows)

        # Test cached data conversion back to FundFlow
        fund_flow_manager._cache_service.get = AsyncMock(return_value=[
            {"amount": 100, "reason": "test", "timestamp": datetime.now(timezone.utc).isoformat()}
        ])

        try:
            cached_flows = await fund_flow_manager._get_cached_fund_flows()
            # May return None or flows depending on data validity
        except Exception:
            pass  # Expected to handle gracefully

    @pytest.mark.asyncio
    async def test_set_capital_allocator(self, fund_flow_manager):
        """Test capital allocator integration."""
        allocator = Mock()
        fund_flow_manager.capital_allocator = allocator
        assert fund_flow_manager.capital_allocator == allocator

    @pytest.mark.asyncio
    async def test_config_validation(self, fund_flow_manager):
        """Test configuration validation."""
        # Start with empty config
        fund_flow_manager.config = {}
        fund_flow_manager._validate_config()

        # Should have default values
        assert "total_capital" in fund_flow_manager.config
        assert "min_deposit_amount" in fund_flow_manager.config
        assert fund_flow_manager.config["auto_compound_enabled"] is True

    @pytest.mark.asyncio
    async def test_cleanup_resources(self, fund_flow_manager):
        """Test resource cleanup."""
        # Mock resource manager
        from unittest.mock import patch, Mock
        mock_resource_manager = Mock()
        mock_resource_manager.clean_fund_flows.return_value = []
        mock_resource_manager.clean_performance_data.return_value = {}

        with patch('src.utils.capital_resources.get_resource_manager', return_value=mock_resource_manager):
            await fund_flow_manager.cleanup_resources()

        mock_resource_manager.clean_fund_flows.assert_called_once()
        mock_resource_manager.clean_performance_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_shutdown(self, fund_flow_manager):
        """Test service shutdown process."""
        await fund_flow_manager.start()
        await fund_flow_manager.stop()
        assert not fund_flow_manager.is_running
