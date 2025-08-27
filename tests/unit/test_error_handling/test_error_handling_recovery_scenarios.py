"""
Unit tests for recovery scenarios component.

These tests verify specific recovery procedures for common failure scenarios
including partial order fills, network disconnections, exchange maintenance,
data feed interruptions, order rejections, and API rate limits.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Config
from src.error_handling.recovery_scenarios import (
    APIRateLimitRecovery,
    DataFeedInterruptionRecovery,
    ExchangeMaintenanceRecovery,
    NetworkDisconnectionRecovery,
    OrderRejectionRecovery,
    PartialFillRecovery,
    RecoveryScenario,
)


class TestRecoveryScenario:
    """Test base recovery scenario functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def recovery_scenario(self, config):
        """Provide base recovery scenario instance."""
        return RecoveryScenario(config)

    def test_recovery_scenario_initialization(self, config):
        """Test recovery scenario initialization."""
        scenario = RecoveryScenario(config)
        assert scenario.config == config
        assert isinstance(scenario.recovery_config, dict)

    @pytest.mark.asyncio
    async def test_recovery_scenario_execute_recovery_not_implemented(self, recovery_scenario):
        """Test that base recovery scenario raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await recovery_scenario.execute_recovery({})


class TestPartialFillRecovery:
    """Test partial fill recovery functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def partial_fill_recovery(self, config):
        """Provide partial fill recovery instance."""
        return PartialFillRecovery(config)

    def test_partial_fill_recovery_initialization(self, config):
        """Test partial fill recovery initialization."""
        recovery = PartialFillRecovery(config)
        assert recovery.config == config
        assert recovery.min_fill_percentage == recovery.recovery_config.get("partial_fill_min_percentage", 0.1)
        assert recovery.cancel_remainder is True
        assert recovery.log_details is True

    @pytest.mark.asyncio
    async def test_partial_fill_recovery_successful_fill(self, partial_fill_recovery):
        """Test successful partial fill recovery."""
        # Create a mock order object with quantity attribute
        order = MagicMock()
        order.quantity = Decimal("1.0")
        order.get.return_value = "order_123"

        context = {"order": order, "filled_quantity": Decimal("0.8")}

        with (
            patch.object(partial_fill_recovery, "_update_position") as mock_update,
            patch.object(partial_fill_recovery, "_adjust_stop_loss") as mock_adjust,
        ):
            result = await partial_fill_recovery.execute_recovery(context)

            assert result is True
            # Should call update_position and adjust_stop_loss for successful fill
            mock_update.assert_called_once_with(order, Decimal("0.8"))
            mock_adjust.assert_called_once_with(order, Decimal("0.8"))

    @pytest.mark.asyncio
    async def test_partial_fill_recovery_insufficient_fill(self, partial_fill_recovery):
        """Test insufficient partial fill recovery."""
        # Create a mock order object with quantity attribute
        order = MagicMock()
        order.quantity = Decimal("1.0")
        order.get.return_value = "order_123"

        context = {
            "order": order,
            "filled_quantity": Decimal("0.3"),  # Below minimum threshold
        }

        with (
            patch.object(partial_fill_recovery, "_cancel_order") as mock_cancel,
            patch.object(partial_fill_recovery, "_log_partial_fill") as mock_log,
            patch.object(partial_fill_recovery, "_reevaluate_signal") as mock_reevaluate,
        ):
            result = await partial_fill_recovery.execute_recovery(context)

            assert result is True
            mock_cancel.assert_called_once_with("order_123")
            mock_log.assert_called_once()
            mock_reevaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_fill_recovery_invalid_context(self, partial_fill_recovery):
        """Test partial fill recovery with invalid context."""
        # Missing order
        context = {"filled_quantity": Decimal("0.5")}
        result = await partial_fill_recovery.execute_recovery(context)
        assert result is False

        # Missing filled_quantity
        context = {"order": {"id": "order_123", "quantity": Decimal("1.0")}}
        result = await partial_fill_recovery.execute_recovery(context)
        assert result is False

        # Empty context
        result = await partial_fill_recovery.execute_recovery({})
        assert result is False


class TestNetworkDisconnectionRecovery:
    """Test network disconnection recovery functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def network_recovery(self, config):
        """Provide network disconnection recovery instance."""
        return NetworkDisconnectionRecovery(config)

    def test_network_recovery_initialization(self, network_recovery):
        """Test network disconnection recovery initialization."""
        # Check that the recovery scenario has the expected attributes
        assert hasattr(network_recovery, "max_offline_duration")
        assert hasattr(network_recovery, "sync_on_reconnect")
        assert hasattr(network_recovery, "conservative_mode")
        assert network_recovery.sync_on_reconnect is True
        assert network_recovery.conservative_mode is True

    @pytest.mark.asyncio
    async def test_network_recovery_successful_reconnection(self, network_recovery):
        """Test successful network reconnection."""
        context = {"component": "exchange", "disconnection_duration": 30}

        with (
            patch.object(network_recovery, "_try_reconnect", return_value=True) as mock_reconnect,
            patch.object(network_recovery, "_reconcile_positions") as mock_reconcile_pos,
            patch.object(network_recovery, "_reconcile_orders") as mock_reconcile_orders,
            patch.object(network_recovery, "_verify_balances") as mock_verify_balances,
            patch.object(network_recovery, "_switch_to_online_mode") as mock_switch_online,
        ):
            result = await network_recovery.execute_recovery(context)

            assert result is True  # Should return True for successful reconnection
            mock_reconnect.assert_called_once_with("exchange")
            mock_reconcile_pos.assert_called_once_with("exchange")
            mock_reconcile_orders.assert_called_once_with("exchange")
            mock_verify_balances.assert_called_once_with("exchange")
            mock_switch_online.assert_called_once_with("exchange")

    @pytest.mark.asyncio
    async def test_network_recovery_failed_reconnection(self, network_recovery):
        """Test failed network reconnection."""
        context = {"component": "exchange", "disconnection_duration": 300}

        with (
            patch.object(network_recovery, "_try_reconnect", return_value=False) as mock_reconnect,
            patch.object(network_recovery, "_enter_safe_mode") as mock_safe_mode,
            patch("asyncio.sleep") as mock_sleep,
        ):
            result = await network_recovery.execute_recovery(context)

            assert result is False  # Should return False for failed reconnection
            mock_reconnect.assert_called()  # Should be called multiple times due to retry loop
            mock_safe_mode.assert_called_once_with("exchange")


class TestExchangeMaintenanceRecovery:
    """Test exchange maintenance recovery functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def maintenance_recovery(self, config):
        """Provide exchange maintenance recovery instance."""
        return ExchangeMaintenanceRecovery(config)

    def test_maintenance_recovery_initialization(self, config):
        """Test exchange maintenance recovery initialization."""
        recovery = ExchangeMaintenanceRecovery(config)
        assert recovery.config == config
        assert recovery.detect_maintenance is True
        assert recovery.redistribute_capital is True
        assert recovery.pause_new_orders is True

    @pytest.mark.asyncio
    async def test_maintenance_recovery_execution(self, maintenance_recovery):
        """Test exchange maintenance recovery execution."""
        import time
        context = {"exchange": "binance", "maintenance_start": time.time()}

        with (
            patch.object(maintenance_recovery, "_detect_maintenance_schedule") as mock_detect,
            patch.object(maintenance_recovery, "_redistribute_capital") as mock_redistribute,
            patch.object(maintenance_recovery, "_pause_new_orders") as mock_pause,
        ):
            result = await maintenance_recovery.execute_recovery(context)

            assert result is True
            mock_detect.assert_called_once_with("binance")
            mock_redistribute.assert_called_once_with("binance")
            mock_pause.assert_called_once_with("binance")


class TestDataFeedInterruptionRecovery:
    """Test data feed interruption recovery functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def data_feed_recovery(self, config):
        """Provide data feed interruption recovery instance."""
        return DataFeedInterruptionRecovery(config)

    def test_data_feed_recovery_initialization(self, data_feed_recovery):
        """Test data feed interruption recovery initialization."""
        # Check that the recovery scenario has the expected attributes
        assert hasattr(data_feed_recovery, "max_staleness")
        assert hasattr(data_feed_recovery, "fallback_sources")
        assert hasattr(data_feed_recovery, "conservative_trading")
        assert data_feed_recovery.fallback_sources == ["backup_feed", "static_data"]
        assert data_feed_recovery.conservative_trading is True


class TestOrderRejectionRecovery:
    """Test order rejection recovery functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def order_rejection_recovery(self, config):
        """Provide order rejection recovery instance."""
        return OrderRejectionRecovery(config)

    def test_order_rejection_recovery_initialization(self, order_rejection_recovery):
        """Test order rejection recovery initialization."""
        assert (
            order_rejection_recovery.max_retry_attempts
            == order_rejection_recovery.recovery_config.get("order_rejection_max_retries", 3)
        )
        assert order_rejection_recovery.analyze_rejection_reason is True
        assert order_rejection_recovery.adjust_parameters is True

    @pytest.mark.asyncio
    async def test_order_rejection_recovery_execution(self, order_rejection_recovery):
        """Test order rejection recovery execution."""
        order = {"id": "order_123", "symbol": "BTCUSDT", "quantity": Decimal("1.0")}
        rejection_reason = "insufficient_balance"

        context = {"order": order, "rejection_reason": rejection_reason}

        with (
            patch.object(order_rejection_recovery, "_analyze_rejection_reason") as mock_analyze,
            patch.object(order_rejection_recovery, "_adjust_order_parameters") as mock_adjust,
        ):
            result = await order_rejection_recovery.execute_recovery(context)

            assert result is True
            mock_analyze.assert_called_once_with(order, rejection_reason)
            mock_adjust.assert_called_once_with(order, rejection_reason)


class TestAPIRateLimitRecovery:
    """Test API rate limit recovery functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def api_rate_limit_recovery(self, config):
        """Provide API rate limit recovery instance."""
        return APIRateLimitRecovery(config)

    def test_api_rate_limit_recovery_initialization(self, api_rate_limit_recovery):
        """Test API rate limit recovery initialization."""
        assert api_rate_limit_recovery.respect_retry_after is True
        assert api_rate_limit_recovery.max_retry_attempts == 3
        assert api_rate_limit_recovery.base_delay == 5

    @pytest.mark.asyncio
    async def test_api_rate_limit_recovery_execution(self, api_rate_limit_recovery):
        """Test API rate limit recovery execution."""
        context = {"api_endpoint": "/api/v3/order", "retry_after": 60}

        with patch("asyncio.sleep") as mock_sleep:
            result = await api_rate_limit_recovery.execute_recovery(context)

            assert result is True
            # Should sleep for retry_after duration plus exponential backoff
            assert mock_sleep.call_count >= 1

    @pytest.mark.asyncio
    async def test_api_rate_limit_recovery_without_retry_after(self, api_rate_limit_recovery):
        """Test API rate limit recovery without retry_after."""
        context = {"api_endpoint": "/api/v3/order"}

        with patch("asyncio.sleep") as mock_sleep:
            result = await api_rate_limit_recovery.execute_recovery(context)

            assert result is True
            # Should sleep for base_delay duration plus exponential backoff
            assert mock_sleep.call_count >= 1


class TestRecoveryScenariosIntegration:
    """Test recovery scenarios integration scenarios."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.mark.asyncio
    async def test_all_recovery_scenarios_initialization(self, config):
        """Test all recovery scenarios can be initialized."""
        scenarios = [
            PartialFillRecovery(config),
            NetworkDisconnectionRecovery(config),
            ExchangeMaintenanceRecovery(config),
            DataFeedInterruptionRecovery(config),
            OrderRejectionRecovery(config),
            APIRateLimitRecovery(config),
        ]

        assert len(scenarios) == 6
        for scenario in scenarios:
            assert scenario.config == config
            assert hasattr(scenario, "execute_recovery")

    @pytest.mark.asyncio
    async def test_recovery_scenarios_error_handling(self, config):
        """Test recovery scenarios error handling."""
        # Test with invalid context
        partial_fill_recovery = PartialFillRecovery(config)

        # Test with missing order
        context = {"filled_quantity": Decimal("0.5")}
        result = await partial_fill_recovery.execute_recovery(context)

        # Should return False for invalid context
        assert result is False

    def test_recovery_scenarios_config_integration(self, config):
        """Test recovery scenarios configuration integration."""
        # Test that all recovery scenarios use the same config
        partial_fill_recovery = PartialFillRecovery(config)
        network_recovery = NetworkDisconnectionRecovery(config)
        maintenance_recovery = ExchangeMaintenanceRecovery(config)

        assert partial_fill_recovery.config == config
        assert network_recovery.config == config
        assert maintenance_recovery.config == config