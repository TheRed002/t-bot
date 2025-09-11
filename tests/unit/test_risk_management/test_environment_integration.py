"""Tests for risk management environment integration."""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
import pytest

from src.core.exceptions import RiskManagementError, ConfigurationError
from src.core.integration.environment_aware_service import EnvironmentContext, EnvironmentAwareServiceMixin
from src.core.config.environment import ExchangeEnvironment
from src.core.types import OrderRequest, Position, OrderSide, OrderType
from src.risk_management.environment_integration import (
    EnvironmentAwareRiskConfiguration,
    EnvironmentAwareRiskManager
)


class TestEnvironmentAwareRiskConfiguration:
    """Test environment-specific risk configurations."""

    def test_get_sandbox_risk_config(self):
        """Test getting sandbox risk configuration."""
        config = EnvironmentAwareRiskConfiguration.get_sandbox_risk_config()
        
        assert isinstance(config, dict)
        assert config["max_position_size_pct"] == Decimal("0.20")
        assert config["risk_per_trade"] == Decimal("0.05")
        assert config["max_total_exposure_pct"] == Decimal("1.00")
        assert config["max_daily_loss_pct"] == Decimal("0.10")
        assert config["position_timeout_minutes"] == 60
        assert config["enable_stop_loss"] is True
        assert config["enable_take_profit"] is True
        assert config["slippage_tolerance_pct"] == Decimal("0.05")
        assert config["max_order_value_usd"] == Decimal("10000")
        assert config["leverage_multiplier"] == Decimal("2.0")
        assert config["correlation_limit"] == Decimal("0.8")
        assert config["volatility_adjustment"] is True
        assert config["kelly_adjustment_factor"] == Decimal("0.8")
        assert config["circuit_breaker_threshold"] == Decimal("0.15")

    def test_get_live_risk_config(self):
        """Test getting live/production risk configuration."""
        config = EnvironmentAwareRiskConfiguration.get_live_risk_config()
        
        assert isinstance(config, dict)
        assert config["max_position_size_pct"] == Decimal("0.05")
        assert config["risk_per_trade"] == Decimal("0.02")
        assert config["max_total_exposure_pct"] == Decimal("0.50")
        assert config["max_daily_loss_pct"] == Decimal("0.05")
        assert config["position_timeout_minutes"] == 30
        assert config["enable_stop_loss"] is True
        assert config["enable_take_profit"] is True
        assert config["slippage_tolerance_pct"] == Decimal("0.02")
        assert config["max_order_value_usd"] == Decimal("50000")
        assert config["leverage_multiplier"] == Decimal("1.0")
        assert config["correlation_limit"] == Decimal("0.5")
        assert config["volatility_adjustment"] is True
        assert config["kelly_adjustment_factor"] == Decimal("0.25")
        assert config["circuit_breaker_threshold"] == Decimal("0.08")

    def test_configuration_differences(self):
        """Test that sandbox and live configs have appropriate differences."""
        sandbox_config = EnvironmentAwareRiskConfiguration.get_sandbox_risk_config()
        live_config = EnvironmentAwareRiskConfiguration.get_live_risk_config()
        
        # Sandbox should be more permissive
        assert sandbox_config["max_position_size_pct"] > live_config["max_position_size_pct"]
        assert sandbox_config["risk_per_trade"] > live_config["risk_per_trade"]
        assert sandbox_config["max_total_exposure_pct"] > live_config["max_total_exposure_pct"]
        assert sandbox_config["max_daily_loss_pct"] > live_config["max_daily_loss_pct"]
        assert sandbox_config["slippage_tolerance_pct"] > live_config["slippage_tolerance_pct"]
        assert sandbox_config["leverage_multiplier"] > live_config["leverage_multiplier"]
        assert sandbox_config["correlation_limit"] > live_config["correlation_limit"]
        assert sandbox_config["kelly_adjustment_factor"] > live_config["kelly_adjustment_factor"]
        assert sandbox_config["circuit_breaker_threshold"] > live_config["circuit_breaker_threshold"]


class TestEnvironmentAwareRiskManager:
    """Test environment-aware risk manager functionality."""

    @pytest.fixture
    def manager(self):
        """Create an environment-aware risk manager."""
        return EnvironmentAwareRiskManager()

    @pytest.fixture
    def sandbox_context(self):
        """Create a sandbox environment context."""
        return EnvironmentContext(
            environment=ExchangeEnvironment.SANDBOX,
            exchange_name="binance_sandbox",
            is_production=False,
            risk_level="medium"
        )

    @pytest.fixture
    def production_context(self):
        """Create a production environment context."""
        return EnvironmentContext(
            environment=ExchangeEnvironment.LIVE,
            exchange_name="binance_live",
            is_production=True,
            risk_level="conservative"
        )

    @pytest.fixture
    def sample_order_request(self):
        """Create a sample order request."""
        return OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            order_type=OrderType.LIMIT
        )

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        from datetime import datetime, timezone
        from src.core.types import PositionSide, PositionStatus
        
        return [
            Position(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("49000.00"),
                current_price=Decimal("50000.00"),
                unrealized_pnl=Decimal("100.00"),
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="test_exchange"
            )
        ]

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert hasattr(manager, "_environment_risk_configs")
        assert hasattr(manager, "_environment_risk_states")
        assert isinstance(manager._environment_risk_configs, dict)
        assert isinstance(manager._environment_risk_states, dict)

    async def test_update_service_environment_sandbox(self, manager, sandbox_context):
        """Test updating service environment for sandbox."""
        with patch.object(EnvironmentAwareServiceMixin, "_update_service_environment") as mock_super:
            mock_super.return_value = None
            
            await manager._update_service_environment(sandbox_context)
            
            assert "binance_sandbox" in manager._environment_risk_configs
            assert "binance_sandbox" in manager._environment_risk_states
            
            config = manager._environment_risk_configs["binance_sandbox"]
            assert config["max_position_size_pct"] == Decimal("0.20")  # Sandbox config
            
            state = manager._environment_risk_states["binance_sandbox"]
            assert state["daily_pnl"] == Decimal("0")
            assert state["total_exposure"] == Decimal("0")
            assert state["active_positions"] == 0
            assert state["circuit_breaker_triggered"] is False

    async def test_update_service_environment_production(self, manager, production_context):
        """Test updating service environment for production."""
        with patch.object(EnvironmentAwareServiceMixin, "_update_service_environment") as mock_super:
            mock_super.return_value = None
            
            await manager._update_service_environment(production_context)
            
            assert "binance_live" in manager._environment_risk_configs
            assert "binance_live" in manager._environment_risk_states
            
            config = manager._environment_risk_configs["binance_live"]
            assert config["max_position_size_pct"] == Decimal("0.05")  # Production config

    def test_get_environment_risk_config_existing(self, manager):
        """Test getting existing environment risk config."""
        # Set up existing config
        test_config = {"test": "value"}
        manager._environment_risk_configs["test_exchange"] = test_config
        
        result = manager.get_environment_risk_config("test_exchange")
        assert result == test_config

    def test_get_environment_risk_config_new_production(self, manager):
        """Test getting new environment risk config for production."""
        with patch.object(manager, "get_environment_context") as mock_context:
            mock_context.return_value = Mock(is_production=True)
            
            result = manager.get_environment_risk_config("new_exchange")
            
            assert "new_exchange" in manager._environment_risk_configs
            assert result["max_position_size_pct"] == Decimal("0.05")  # Production config

    def test_get_environment_risk_config_new_sandbox(self, manager):
        """Test getting new environment risk config for sandbox."""
        with patch.object(manager, "get_environment_context") as mock_context:
            mock_context.return_value = Mock(is_production=False)
            
            result = manager.get_environment_risk_config("new_exchange")
            
            assert "new_exchange" in manager._environment_risk_configs
            assert result["max_position_size_pct"] == Decimal("0.20")  # Sandbox config

    def test_calculate_environment_aware_position_size_basic(self, manager, sample_order_request):
        """Test basic position size calculation."""
        # Setup
        exchange = "test_exchange"
        account_balance = Decimal("10000.00")
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config:
            
            mock_context.return_value = Mock(is_production=False)
            mock_config.return_value = {
                "risk_per_trade": Decimal("0.02"),
                "max_position_size_pct": Decimal("0.10"),
                "max_order_value_usd": Decimal("5000")
            }
            
            result = manager.calculate_environment_aware_position_size(
                sample_order_request, exchange, account_balance
            )
            
            # Calculate expected: account_balance * risk_per_trade * 1.5 (sandbox multiplier)
            # = 10000 * 0.02 * 1.5 = 300, but capped by max_order_value_usd
            # max_order_qty = 5000 / 50000 = 0.1
            max_order_qty = Decimal("5000") / sample_order_request.price  # 0.1
            expected = max_order_qty
            
            assert result == expected

    def test_calculate_environment_aware_position_size_production(self, manager, sample_order_request):
        """Test position size calculation for production environment."""
        exchange = "test_exchange"
        account_balance = Decimal("10000.00")
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config:
            
            mock_context.return_value = Mock(is_production=True)
            mock_config.return_value = {
                "risk_per_trade": Decimal("0.02"),
                "max_position_size_pct": Decimal("0.05"),
            }
            
            result = manager.calculate_environment_aware_position_size(
                sample_order_request, exchange, account_balance
            )
            
            # Production: base_position_size = 10000 * 0.02 = 200
            # max_position_size = 10000 * 0.05 = 500
            # Result should be min(200, 500) = 200
            assert result == Decimal("200.00")

    def test_calculate_environment_aware_position_size_with_correlation(self, manager, sample_order_request):
        """Test position size calculation with correlation adjustment."""
        exchange = "test_exchange"
        account_balance = Decimal("10000.00")
        
        # Add the _check_portfolio_correlation method to the manager instance for this test
        def mock_check_portfolio_correlation(symbol, exchange):
            return Decimal("0.8")
        manager._check_portfolio_correlation = mock_check_portfolio_correlation
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config:
            
            mock_context.return_value = Mock(is_production=True)
            mock_config.return_value = {
                "risk_per_trade": Decimal("0.02"),
                "max_position_size_pct": Decimal("0.10"),
            }
            
            result = manager.calculate_environment_aware_position_size(
                sample_order_request, exchange, account_balance
            )
            
            # Base: 10000 * 0.02 = 200
            # After correlation adjustment: 200 * 0.8 = 160
            assert result == Decimal("160.00")

    def test_calculate_environment_aware_position_size_with_volatility(self, manager, sample_order_request):
        """Test position size calculation with volatility adjustment."""
        exchange = "test_exchange"
        account_balance = Decimal("10000.00")
        market_data = {"volatility": "0.03"}
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config, \
             patch.object(manager, "_calculate_volatility_adjustment", return_value=Decimal("0.9")):
            
            mock_context.return_value = Mock(is_production=False)
            mock_config.return_value = {
                "risk_per_trade": Decimal("0.02"),
                "max_position_size_pct": Decimal("0.10"),
                "volatility_adjustment": True
            }
            
            result = manager.calculate_environment_aware_position_size(
                sample_order_request, exchange, account_balance, market_data
            )
            
            # Sandbox: (10000 * 0.02 * 1.5) * 0.9 = 300 * 0.9 = 270
            assert result == Decimal("270.00")

    def test_calculate_environment_aware_position_size_max_order_value_limit(self, manager, sample_order_request):
        """Test position size calculation with max order value limit."""
        exchange = "test_exchange"
        account_balance = Decimal("100000.00")  # Large balance
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config:
            
            mock_context.return_value = Mock(is_production=False)
            mock_config.return_value = {
                "risk_per_trade": Decimal("0.10"),  # High risk
                "max_position_size_pct": Decimal("0.20"),
                "max_order_value_usd": Decimal("5000")  # $5000 max order
            }
            
            result = manager.calculate_environment_aware_position_size(
                sample_order_request, exchange, account_balance
            )
            
            # Max order qty = 5000 / 50000 = 0.1
            # This should cap the position size
            max_order_qty = Decimal("5000") / sample_order_request.price
            assert result == max_order_qty

    def test_calculate_environment_aware_position_size_missing_config(self, manager, sample_order_request):
        """Test position size calculation with missing configuration."""
        exchange = "test_exchange"
        account_balance = Decimal("10000.00")
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config:
            
            mock_context.return_value = Mock(is_production=False)
            mock_config.return_value = {"incomplete": "config"}  # Missing required keys
            
            with pytest.raises(RiskManagementError):
                manager.calculate_environment_aware_position_size(
                    sample_order_request, exchange, account_balance
                )

    def test_calculate_environment_aware_position_size_exception_handling(self, manager, sample_order_request):
        """Test position size calculation exception handling."""
        exchange = "test_exchange"
        account_balance = Decimal("10000.00")
        
        with patch.object(manager, "get_environment_context", side_effect=Exception("Test error")):
            
            with pytest.raises(RiskManagementError):
                manager.calculate_environment_aware_position_size(
                    sample_order_request, exchange, account_balance
                )

    async def test_validate_environment_order_circuit_breaker(self, manager, sample_order_request):
        """Test order validation with circuit breaker triggered."""
        exchange = "test_exchange"
        
        # Set up circuit breaker triggered state
        manager._environment_risk_states[exchange] = {"circuit_breaker_triggered": True}
        
        with patch.object(manager, "get_environment_context") as mock_context:
            mock_context.return_value = Mock(environment=ExchangeEnvironment.SANDBOX)
            
            with pytest.raises(RiskManagementError, match="Circuit breaker triggered"):
                await manager.validate_environment_order(sample_order_request, exchange)

    async def test_validate_environment_order_production(self, manager, sample_order_request):
        """Test order validation for production environment."""
        exchange = "test_exchange"
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config, \
             patch.object(manager, "_validate_production_order", return_value=True), \
             patch.object(manager, "_validate_common_risk_rules", return_value=True):
            
            mock_context.return_value = Mock(is_production=True, environment=ExchangeEnvironment.LIVE)
            mock_config.return_value = {}
            manager._environment_risk_states[exchange] = {}
            
            result = await manager.validate_environment_order(sample_order_request, exchange)
            assert result is True

    async def test_validate_environment_order_sandbox(self, manager, sample_order_request):
        """Test order validation for sandbox environment."""
        exchange = "test_exchange"
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config, \
             patch.object(manager, "_validate_sandbox_order", return_value=True), \
             patch.object(manager, "_validate_common_risk_rules", return_value=True):
            
            mock_context.return_value = Mock(is_production=False, environment=ExchangeEnvironment.SANDBOX)
            mock_config.return_value = {}
            manager._environment_risk_states[exchange] = {}
            
            result = await manager.validate_environment_order(sample_order_request, exchange)
            assert result is True

    async def test_validate_production_order_success(self, manager, sample_order_request):
        """Test successful production order validation."""
        exchange = "test_exchange"
        risk_config = {
            "leverage_multiplier": Decimal("1.0"),
            "slippage_tolerance_pct": Decimal("0.02")
        }
        
        with patch.object(manager, "_detect_suspicious_order_pattern", return_value=False):
            result = await manager._validate_production_order(sample_order_request, exchange, risk_config)
            assert result is True

    async def test_validate_production_order_excessive_leverage(self, manager):
        """Test production order validation with excessive leverage."""
        exchange = "test_exchange"
        risk_config = {"leverage_multiplier": Decimal("1.0")}
        
        # Create order with leverage
        order_with_leverage = Mock()
        order_with_leverage.leverage = Decimal("2.0")  # Exceeds limit
        
        result = await manager._validate_production_order(order_with_leverage, exchange, risk_config)
        assert result is False

    async def test_validate_production_order_excessive_slippage(self, manager):
        """Test production order validation with excessive slippage."""
        exchange = "test_exchange"
        risk_config = {"slippage_tolerance_pct": Decimal("0.02")}
        
        # Create order with expected slippage
        order_with_slippage = Mock()
        order_with_slippage.expected_slippage = Decimal("0.05")  # Exceeds tolerance
        order_with_slippage.leverage = None  # No leverage for this test
        
        result = await manager._validate_production_order(order_with_slippage, exchange, risk_config)
        assert result is False

    async def test_validate_production_order_suspicious_pattern(self, manager, sample_order_request):
        """Test production order validation with suspicious pattern."""
        exchange = "test_exchange"
        risk_config = {}
        
        with patch.object(manager, "_detect_suspicious_order_pattern", return_value=True):
            result = await manager._validate_production_order(sample_order_request, exchange, risk_config)
            assert result is False

    async def test_validate_sandbox_order_success(self, manager, sample_order_request):
        """Test successful sandbox order validation."""
        exchange = "test_exchange"
        risk_config = {"max_order_value_usd": Decimal("10000")}
        
        result = await manager._validate_sandbox_order(sample_order_request, exchange, risk_config)
        assert result is True

    async def test_validate_sandbox_order_exceeds_limit(self, manager, sample_order_request):
        """Test sandbox order validation exceeding order value limit."""
        exchange = "test_exchange"
        risk_config = {"max_order_value_usd": Decimal("1000")}  # Lower than order value
        
        result = await manager._validate_sandbox_order(sample_order_request, exchange, risk_config)
        # Sandbox should still return True (just warn)
        assert result is True

    async def test_validate_common_risk_rules_success(self, manager, sample_order_request):
        """Test successful common risk rules validation."""
        exchange = "test_exchange"
        risk_config = {
            "max_total_positions": 10,
            "max_daily_loss_pct": Decimal("0.10")
        }
        current_positions = []
        
        manager._environment_risk_states[exchange] = {"daily_pnl": Decimal("0")}
        
        result = await manager._validate_common_risk_rules(
            sample_order_request, exchange, risk_config, current_positions
        )
        assert result is True

    async def test_validate_common_risk_rules_too_many_positions(self, manager, sample_order_request, sample_positions):
        """Test common risk rules validation with too many positions."""
        exchange = "test_exchange"
        risk_config = {"max_total_positions": 1}  # Low limit
        current_positions = sample_positions * 2  # Exceeds limit
        
        result = await manager._validate_common_risk_rules(
            sample_order_request, exchange, risk_config, current_positions
        )
        assert result is False

    async def test_validate_common_risk_rules_daily_loss_exceeded(self, manager, sample_order_request):
        """Test common risk rules validation with daily loss limit exceeded."""
        exchange = "test_exchange"
        risk_config = {"max_daily_loss_pct": Decimal("0.05")}
        
        manager._environment_risk_states[exchange] = {"daily_pnl": Decimal("-0.10")}  # Exceeds limit
        
        result = await manager._validate_common_risk_rules(
            sample_order_request, exchange, risk_config, []
        )
        assert result is False

    def test_calculate_volatility_adjustment_high_volatility(self, manager):
        """Test volatility adjustment calculation with high volatility."""
        market_data = {"volatility": "0.05"}
        risk_config = {"volatility_target": Decimal("0.02")}
        
        result = manager._calculate_volatility_adjustment(market_data, risk_config)
        
        # High volatility should reduce position size
        # target / estimated = 0.02 / 0.05 = 0.4
        assert result == Decimal("0.4")

    def test_calculate_volatility_adjustment_low_volatility(self, manager):
        """Test volatility adjustment calculation with low volatility."""
        market_data = {"volatility": "0.01"}
        risk_config = {"volatility_target": Decimal("0.02")}
        
        result = manager._calculate_volatility_adjustment(market_data, risk_config)
        
        # Low volatility should allow larger position, but capped at 1.2
        # target / estimated = 0.02 / 0.025 = 0.8, but we use placeholder logic
        assert result == Decimal("1.2")

    def test_calculate_volatility_adjustment_exception(self, manager):
        """Test volatility adjustment calculation with exception."""
        # Force an exception by mocking the division operation
        market_data = {"volatility": "0.01"}
        risk_config = {"volatility_target": Decimal("0.02")}
        
        # Mock min() to raise an exception during calculation
        with patch('src.risk_management.environment_integration.min') as mock_min:
            mock_min.side_effect = Exception("Test exception")
            result = manager._calculate_volatility_adjustment(market_data, risk_config)
            
        assert result == Decimal("1.0")  # Default on error

    async def test_detect_suspicious_order_pattern(self, manager, sample_order_request):
        """Test suspicious order pattern detection."""
        exchange = "test_exchange"
        
        result = await manager._detect_suspicious_order_pattern(sample_order_request, exchange)
        assert result is False  # Placeholder always returns False

    async def test_update_environment_risk_state_new_exchange(self, manager):
        """Test updating risk state for new exchange."""
        exchange = "new_exchange"
        
        with patch.object(manager, "get_environment_risk_config") as mock_config, \
             patch.object(manager, "_notify_circuit_breaker_triggered") as mock_notify:
            
            mock_config.return_value = {"circuit_breaker_threshold": Decimal("0.10")}
            
            await manager.update_environment_risk_state(
                exchange, 
                pnl_change=Decimal("-0.05"),  # Small loss that won't trigger circuit breaker
                exposure_change=Decimal("100.00"),
                position_change=1
            )
            
            assert exchange in manager._environment_risk_states
            state = manager._environment_risk_states[exchange]
            assert state["daily_pnl"] == Decimal("-0.05")
            assert state["total_exposure"] == Decimal("100.00")
            assert state["active_positions"] == 1
            assert state["circuit_breaker_triggered"] is False  # Not triggered yet

    async def test_update_environment_risk_state_circuit_breaker(self, manager):
        """Test updating risk state triggering circuit breaker."""
        exchange = "test_exchange"
        
        with patch.object(manager, "get_environment_risk_config") as mock_config, \
             patch.object(manager, "_notify_circuit_breaker_triggered") as mock_notify:
            
            mock_config.return_value = {"circuit_breaker_threshold": Decimal("0.10")}
            
            await manager.update_environment_risk_state(
                exchange,
                pnl_change=Decimal("-0.15"),  # Triggers circuit breaker
                exposure_change=Decimal("0.00"),
                position_change=0
            )
            
            state = manager._environment_risk_states[exchange]
            assert state["circuit_breaker_triggered"] is True
            mock_notify.assert_called_once_with(exchange, Decimal("-0.15"))

    async def test_notify_circuit_breaker_triggered(self, manager):
        """Test circuit breaker notification."""
        exchange = "test_exchange"
        loss_amount = Decimal("-100.00")
        
        # Should not raise exception
        await manager._notify_circuit_breaker_triggered(exchange, loss_amount)

    async def test_reset_environment_risk_state(self, manager):
        """Test resetting environment risk state."""
        exchange = "test_exchange"
        
        # Set up existing state
        manager._environment_risk_states[exchange] = {
            "daily_pnl": Decimal("-50.00"),
            "total_exposure": Decimal("100.00"),
            "circuit_breaker_triggered": True,
            "last_reset": None
        }
        
        await manager.reset_environment_risk_state(exchange)
        
        state = manager._environment_risk_states[exchange]
        assert state["daily_pnl"] == Decimal("0")
        assert state["total_exposure"] == Decimal("0")
        assert state["circuit_breaker_triggered"] is False
        assert state["last_reset"] is not None

    async def test_reset_environment_risk_state_nonexistent(self, manager):
        """Test resetting risk state for nonexistent exchange."""
        exchange = "nonexistent_exchange"
        
        # Should not raise exception
        await manager.reset_environment_risk_state(exchange)

    def test_get_environment_risk_metrics(self, manager):
        """Test getting environment risk metrics."""
        exchange = "test_exchange"
        
        # Set up test data
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config:
            
            mock_context.return_value = Mock(
                environment=ExchangeEnvironment.SANDBOX,
                is_production=False,
                risk_level="medium"
            )
            mock_config.return_value = {
                "max_position_size_pct": Decimal("0.20"),
                "risk_per_trade": Decimal("0.05"),
                "max_daily_loss_pct": Decimal("0.10"),
                "circuit_breaker_threshold": Decimal("0.15")
            }
            manager._environment_risk_states[exchange] = {
                "daily_pnl": Decimal("-25.00"),
                "total_exposure": Decimal("500.00"),
                "active_positions": 3,
                "circuit_breaker_triggered": False
            }
            
            result = manager.get_environment_risk_metrics(exchange)
            
            assert result["exchange"] == exchange
            assert result["environment"] == "sandbox"
            assert result["is_production"] is False
            assert result["risk_level"] == "medium"
            assert result["daily_pnl"] == "-25.00000000"
            assert result["total_exposure"] == "500.00000000"
            assert result["active_positions"] == 3
            assert result["circuit_breaker_active"] is False
            assert result["max_position_size_pct"] == "0.20000000"
            assert result["risk_per_trade"] == "0.05000000"
            assert result["max_daily_loss_pct"] == "0.10000000"
            assert result["circuit_breaker_threshold"] == "0.15000000"
            assert "last_updated" in result

    def test_get_environment_risk_metrics_empty_state(self, manager):
        """Test getting environment risk metrics with empty state."""
        exchange = "test_exchange"
        
        with patch.object(manager, "get_environment_context") as mock_context, \
             patch.object(manager, "get_environment_risk_config") as mock_config:
            
            mock_context.return_value = Mock(
                environment=ExchangeEnvironment.LIVE,
                is_production=True,
                risk_level="conservative"
            )
            mock_config.return_value = {
                "max_position_size_pct": Decimal("0.05"),
                "risk_per_trade": Decimal("0.02"),
                "max_daily_loss_pct": Decimal("0.05"),
                "circuit_breaker_threshold": Decimal("0.08")
            }
            
            result = manager.get_environment_risk_metrics(exchange)
            
            # Should handle empty state gracefully with defaults
            assert result["daily_pnl"] == "0"
            assert result["total_exposure"] == "0"
            assert result["active_positions"] == 0
            assert result["circuit_breaker_active"] is False