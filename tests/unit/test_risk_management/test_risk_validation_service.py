"""Tests for risk validation service."""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
import pytest

from src.core.exceptions import ValidationError, RiskManagementError
from src.core.types import OrderRequest, Position, RiskLevel, Signal, OrderSide, OrderType, PositionSide, PositionStatus, StateType, SignalDirection
from src.risk_management.services.risk_validation_service import RiskValidationService
from src.utils.decimal_utils import ZERO


class TestRiskValidationService:
    """Test risk validation service functionality."""

    @pytest.fixture
    def mock_database_service(self):
        """Create mock database service."""
        return AsyncMock()

    @pytest.fixture
    def mock_state_service(self):
        """Create mock state service."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.risk = Mock()
        config.risk.max_position_size = "10000"
        config.risk.max_portfolio_risk = "0.1"
        config.risk.max_correlation = "0.7"
        config.risk.max_leverage = "3.0"
        config.risk.max_drawdown = "0.2"
        config.risk.max_daily_loss = "0.05"
        config.risk.max_positions = 10
        config.risk.min_liquidity_ratio = "0.2"
        config.risk.max_total_positions = 10
        config.risk.max_positions_per_symbol = 3
        config.risk.max_position_size_pct = "0.25"
        config.risk.max_portfolio_exposure_pct = "0.80"
        config.__dict__ = {
            "risk": config.risk
        }
        return config

    @pytest.fixture
    def service(self, mock_database_service, mock_state_service, mock_config):
        """Create risk validation service."""
        return RiskValidationService(
            portfolio_repository=mock_database_service,
            state_service=mock_state_service,
            config=mock_config,
            correlation_id="test-correlation-id"
        )

    @pytest.fixture
    def service_without_config(self, mock_database_service, mock_state_service):
        """Create risk validation service without config."""
        return RiskValidationService(
            portfolio_repository=mock_database_service,
            state_service=mock_state_service,
            config=None,
            correlation_id="test-correlation-id"
        )

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal."""
        from datetime import datetime, timezone
        return Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(timezone.utc),
            source="test_source"
        )

    @pytest.fixture
    def weak_signal(self):
        """Create a weak trading signal."""
        from datetime import datetime, timezone
        return Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            strength=Decimal("0.2"),  # Below threshold
            timestamp=datetime.now(timezone.utc),
            source="test_source"
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
        return [
            Position(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.1"),
                entry_price=Decimal("49000.00"),
                current_price=Decimal("50000.00"),
                unrealized_pnl=Decimal("100.00"),
                realized_pnl=Decimal("0.00"),
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="binance"
            ),
            Position(
                symbol="ETH/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000.00"),
                current_price=Decimal("3100.00"),
                unrealized_pnl=Decimal("100.00"),
                realized_pnl=Decimal("0.00"),
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="binance"
            )
        ]

    def test_service_initialization_with_config(self, service, mock_config):
        """Test service initialization with configuration."""
        assert service.portfolio_repository is not None
        assert service.state_service is not None
        assert service.config == mock_config
        assert service.validator is not None
        assert service.name == "RiskValidationService"

    def test_service_initialization_without_config(self, service_without_config):
        """Test service initialization without configuration."""
        assert service_without_config.portfolio_repository is not None
        assert service_without_config.state_service is not None
        assert service_without_config.config is None
        assert service_without_config.validator is not None

    async def test_validate_signal_success(self, service, sample_signal):
        """Test successful signal validation."""
        with patch.object(service, '_get_current_risk_level', return_value=RiskLevel.LOW), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_check_symbol_position_limits', return_value=True), \
             patch.object(service.validator, 'validate_signal', return_value=(True, None)), \
             patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            
            result = await service.validate_signal(sample_signal)
            assert result is True

    async def test_validate_signal_weak_strength(self, service, weak_signal):
        """Test signal validation with weak strength."""
        with patch.object(service, '_get_current_risk_level', return_value=RiskLevel.LOW), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_check_symbol_position_limits', return_value=True), \
             patch.object(service.validator, 'validate_signal', return_value=(True, None)), \
             patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            
            result = await service.validate_signal(weak_signal)
            assert result is False

    async def test_validate_signal_critical_risk_level(self, service, sample_signal):
        """Test signal validation with critical risk level."""
        with patch.object(service, '_get_current_risk_level', return_value=RiskLevel.CRITICAL), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service.validator, 'validate_signal', return_value=(True, None)), \
             patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            
            result = await service.validate_signal(sample_signal)
            assert result is False

    async def test_validate_signal_emergency_stop_active(self, service, sample_signal):
        """Test signal validation with emergency stop active."""
        with patch.object(service, '_get_current_risk_level', return_value=RiskLevel.LOW), \
             patch.object(service, '_is_emergency_stop_active', return_value=True), \
             patch.object(service.validator, 'validate_signal', return_value=(True, None)), \
             patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            
            result = await service.validate_signal(sample_signal)
            assert result is False

    async def test_validate_signal_symbol_position_limits_exceeded(self, service, sample_signal):
        """Test signal validation with symbol position limits exceeded."""
        with patch.object(service, '_get_current_risk_level', return_value=RiskLevel.LOW), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_check_symbol_position_limits', return_value=False), \
             patch.object(service.validator, 'validate_signal', return_value=(True, None)), \
             patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            
            result = await service.validate_signal(sample_signal)
            assert result is False

    async def test_validate_signal_validator_failure(self, service, sample_signal):
        """Test signal validation with centralized validator failure."""
        with patch.object(service, '_get_current_risk_level', return_value=RiskLevel.LOW), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_check_symbol_position_limits', return_value=True), \
             patch.object(service.validator, 'validate_signal', return_value=(False, "Validation failed")), \
             patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            
            result = await service.validate_signal(sample_signal)
            assert result is False

    async def test_validate_signal_boundary_validation_error(self, service, sample_signal):
        """Test signal validation with boundary validation error."""
        with patch.object(service, '_get_current_risk_level', return_value=RiskLevel.LOW), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity',
                  side_effect=Exception("Boundary validation failed")):
            
            result = await service.validate_signal(sample_signal)
            assert result is False

    async def test_validate_signal_structure_validation_error(self, service):
        """Test signal validation with invalid signal structure."""
        invalid_signal = Mock()
        invalid_signal.symbol = None  # Invalid structure
        
        with patch.object(service, '_get_current_risk_level', return_value=RiskLevel.LOW), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_validate_signal_structure', return_value=False), \
             patch.object(service.validator, 'validate_signal', return_value=(True, None)), \
             patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            
            result = await service.validate_signal(invalid_signal)
            assert result is False

    async def test_validate_signal_unexpected_exception(self, service, sample_signal):
        """Test signal validation with unexpected exception."""
        with patch.object(service, '_get_current_risk_level', side_effect=Exception("Unexpected error")):
            
            result = await service.validate_signal(sample_signal)
            assert result is False

    async def test_validate_order_success(self, service, sample_order_request):
        """Test successful order validation."""
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_positions', return_value=[]), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_validate_order_size_limits', return_value=True), \
             patch.object(service, '_validate_portfolio_exposure', return_value=True), \
             patch.object(service.validator, 'validate_order', return_value=(True, None)):
            
            result = await service.validate_order(sample_order_request)
            assert result is True

    async def test_validate_order_emergency_stop_active(self, service, sample_order_request):
        """Test order validation with emergency stop active."""
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_positions', return_value=[]), \
             patch.object(service, '_is_emergency_stop_active', return_value=True), \
             patch.object(service.validator, 'validate_order', return_value=(True, None)):
            
            result = await service.validate_order(sample_order_request)
            assert result is False

    async def test_validate_order_size_limits_exceeded(self, service, sample_order_request):
        """Test order validation with size limits exceeded."""
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_positions', return_value=[]), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_validate_order_size_limits', return_value=False), \
             patch.object(service.validator, 'validate_order', return_value=(True, None)):
            
            result = await service.validate_order(sample_order_request)
            assert result is False

    async def test_validate_order_portfolio_exposure_exceeded(self, service, sample_order_request):
        """Test order validation with portfolio exposure exceeded."""
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_positions', return_value=[]), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_validate_order_size_limits', return_value=True), \
             patch.object(service, '_validate_portfolio_exposure', return_value=False), \
             patch.object(service.validator, 'validate_order', return_value=(True, None)):
            
            result = await service.validate_order(sample_order_request)
            assert result is False

    async def test_validate_order_validator_failure(self, service, sample_order_request):
        """Test order validation with centralized validator failure."""
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_positions', return_value=[]), \
             patch.object(service, '_is_emergency_stop_active', return_value=False), \
             patch.object(service, '_validate_order_size_limits', return_value=True), \
             patch.object(service, '_validate_portfolio_exposure', return_value=True), \
             patch.object(service.validator, 'validate_order', return_value=(False, "Order invalid")):
            
            result = await service.validate_order(sample_order_request)
            assert result is False

    async def test_validate_order_exception(self, service, sample_order_request):
        """Test order validation with exception."""
        with patch.object(service, '_get_portfolio_value', side_effect=Exception("Error getting portfolio")):
            
            result = await service.validate_order(sample_order_request)
            assert result is False

    async def test_validate_portfolio_limits_success(self, service, sample_positions):
        """Test successful portfolio limits validation."""
        new_position = sample_positions[0]
        
        with patch.object(service, '_get_current_positions', return_value=[]), \
             patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_positions_for_symbol', return_value=[]), \
             patch.object(service, '_validate_position_exposure', return_value=True), \
             patch.object(service.validator, 'validate_position', return_value=(True, None)), \
             patch('src.utils.risk_validation.check_position_limits', return_value=(True, None)):
            
            result = await service.validate_portfolio_limits(new_position)
            assert result is True

    async def test_validate_portfolio_limits_total_limit_exceeded(self, service, sample_positions):
        """Test portfolio limits validation with total limit exceeded."""
        new_position = sample_positions[0]
        # Create list with max positions
        current_positions = sample_positions * 10
        
        with patch.object(service, '_get_current_positions', return_value=current_positions), \
             patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_max_total_positions', return_value=10), \
             patch('src.utils.risk_validation.check_position_limits', return_value=(True, None)):
            
            result = await service.validate_portfolio_limits(new_position)
            assert result is False

    async def test_validate_portfolio_limits_symbol_limit_exceeded(self, service, sample_positions):
        """Test portfolio limits validation with symbol limit exceeded."""
        new_position = sample_positions[0]
        symbol_positions = [sample_positions[0]] * 5  # More than max per symbol
        
        with patch.object(service, '_get_current_positions', return_value=[]), \
             patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_positions_for_symbol', return_value=symbol_positions), \
             patch.object(service, '_get_max_positions_per_symbol', return_value=3), \
             patch.object(service.validator, 'validate_position', return_value=(True, None)), \
             patch('src.utils.risk_validation.check_position_limits', return_value=(True, None)):
            
            result = await service.validate_portfolio_limits(new_position)
            assert result is False

    async def test_validate_portfolio_limits_position_exposure_exceeded(self, service, sample_positions):
        """Test portfolio limits validation with position exposure exceeded."""
        new_position = sample_positions[0]
        
        with patch.object(service, '_get_current_positions', return_value=[]), \
             patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_positions_for_symbol', return_value=[]), \
             patch.object(service, '_validate_position_exposure', return_value=False), \
             patch.object(service.validator, 'validate_position', return_value=(True, None)), \
             patch('src.utils.risk_validation.check_position_limits', return_value=(True, None)):
            
            result = await service.validate_portfolio_limits(new_position)
            assert result is False

    async def test_validate_portfolio_limits_exception(self, service, sample_positions):
        """Test portfolio limits validation with exception."""
        new_position = sample_positions[0]
        
        with patch.object(service, '_get_current_positions', side_effect=Exception("Error getting positions")):
            
            result = await service.validate_portfolio_limits(new_position)
            assert result is False

    def test_validate_signal_structure_success(self, service, sample_signal):
        """Test successful signal structure validation."""
        result = service._validate_signal_structure(sample_signal)
        assert result is True

    def test_validate_signal_structure_none_signal(self, service):
        """Test signal structure validation with None signal."""
        result = service._validate_signal_structure(None)
        assert result is False

    def test_validate_signal_structure_missing_symbol(self, service):
        """Test signal structure validation with missing symbol."""
        signal = Mock()
        signal.symbol = None
        signal.direction = "BUY"
        signal.strength = 0.8
        
        result = service._validate_signal_structure(signal)
        assert result is False

    def test_validate_signal_structure_missing_direction(self, service):
        """Test signal structure validation with missing direction."""
        signal = Mock()
        signal.symbol = "BTC/USDT"
        signal.direction = None
        signal.strength = 0.8
        
        result = service._validate_signal_structure(signal)
        assert result is False

    def test_validate_signal_structure_invalid_strength(self, service):
        """Test signal structure validation with invalid strength."""
        signal = Mock()
        signal.symbol = "BTC/USDT"
        signal.direction = "BUY"
        signal.strength = "invalid"  # Not numeric
        
        result = service._validate_signal_structure(signal)
        assert result is False

    def test_validate_signal_structure_strength_out_of_range(self, service):
        """Test signal structure validation with strength out of range."""
        signal = Mock()
        signal.symbol = "BTC/USDT"
        signal.direction = "BUY"
        signal.strength = 1.5  # > 1
        
        result = service._validate_signal_structure(signal)
        assert result is False

    def test_validate_order_structure_success(self, service, sample_order_request):
        """Test successful order structure validation."""
        result = service._validate_order_structure(sample_order_request)
        assert result is True

    def test_validate_order_structure_none_order(self, service):
        """Test order structure validation with None order."""
        result = service._validate_order_structure(None)
        assert result is False

    def test_validate_order_structure_missing_symbol(self, service):
        """Test order structure validation with missing symbol."""
        order = Mock()
        order.symbol = None
        order.side = OrderSide.BUY
        order.quantity = Decimal("0.1")
        
        result = service._validate_order_structure(order)
        assert result is False

    def test_validate_order_structure_missing_side(self, service):
        """Test order structure validation with missing side."""
        order = Mock()
        order.symbol = "BTC/USDT"
        order.side = None
        order.quantity = Decimal("0.1")
        
        result = service._validate_order_structure(order)
        assert result is False

    def test_validate_order_structure_invalid_quantity(self, service):
        """Test order structure validation with invalid quantity."""
        order = Mock()
        order.symbol = "BTC/USDT"
        order.side = OrderSide.BUY
        order.quantity = ZERO
        
        result = service._validate_order_structure(order)
        assert result is False

    async def test_validate_order_size_limits_success(self, service, sample_order_request):
        """Test successful order size limits validation."""
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_price', return_value=Decimal("50000")), \
             patch.object(service, '_get_max_position_size_pct', return_value=Decimal("0.60")):  # 60% limit allows $5000 order
            
            result = await service._validate_order_size_limits(sample_order_request)
            assert result is True

    async def test_validate_order_size_limits_zero_portfolio(self, service, sample_order_request):
        """Test order size limits validation with zero portfolio value."""
        with patch.object(service, '_get_portfolio_value', return_value=ZERO):
            
            result = await service._validate_order_size_limits(sample_order_request)
            assert result is True  # Should allow if no portfolio to limit against

    async def test_validate_order_size_limits_exceeded(self, service):
        """Test order size limits validation with size exceeded."""
        large_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),  # Large quantity
            price=Decimal("50000.00"),
            order_type=OrderType.LIMIT
        )
        
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("1000")), \
             patch.object(service, '_get_current_price', return_value=Decimal("50000")):
            
            result = await service._validate_order_size_limits(large_order)
            assert result is False

    async def test_validate_order_size_limits_no_price(self, service, sample_order_request):
        """Test order size limits validation with no price available."""
        sample_order_request.price = None
        
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_price', return_value=None):
            
            result = await service._validate_order_size_limits(sample_order_request)
            assert result is True  # Should allow if price cannot be determined

    async def test_validate_order_size_limits_exception(self, service, sample_order_request):
        """Test order size limits validation with exception."""
        with patch.object(service, '_get_portfolio_value', side_effect=Exception("Error")):
            
            result = await service._validate_order_size_limits(sample_order_request)
            assert result is True  # Should allow on error

    async def test_validate_portfolio_exposure_success(self, service, sample_order_request):
        """Test successful portfolio exposure validation."""
        with patch.object(service, '_get_current_exposure', return_value=Decimal("1000")), \
             patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_price', return_value=Decimal("50000")):
            
            result = await service._validate_portfolio_exposure(sample_order_request)
            assert result is True

    async def test_validate_portfolio_exposure_exceeded(self, service):
        """Test portfolio exposure validation with exposure exceeded."""
        large_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            price=Decimal("50000.00"),
            order_type=OrderType.LIMIT
        )
        
        with patch.object(service, '_get_current_exposure', return_value=Decimal("7000")), \
             patch.object(service, '_get_portfolio_value', return_value=Decimal("10000")), \
             patch.object(service, '_get_current_price', return_value=Decimal("50000")):
            
            result = await service._validate_portfolio_exposure(large_order)
            assert result is False

    async def test_validate_portfolio_exposure_zero_portfolio(self, service, sample_order_request):
        """Test portfolio exposure validation with zero portfolio value."""
        with patch.object(service, '_get_current_exposure', return_value=Decimal("1000")), \
             patch.object(service, '_get_portfolio_value', return_value=ZERO):
            
            result = await service._validate_portfolio_exposure(sample_order_request)
            assert result is True

    async def test_validate_portfolio_exposure_exception(self, service, sample_order_request):
        """Test portfolio exposure validation with exception."""
        with patch.object(service, '_get_current_exposure', side_effect=Exception("Error")):
            
            result = await service._validate_portfolio_exposure(sample_order_request)
            assert result is True

    async def test_validate_position_exposure_success(self, service, sample_positions):
        """Test successful position exposure validation."""
        position = sample_positions[0]
        
        # Use larger portfolio value so position is within 25% limit
        # Position value: 0.1 * 50000 = 5000
        # Portfolio value: 25000, so position is 5000/25000 = 20% < 25% limit
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("25000")):
            
            result = await service._validate_position_exposure(position)
            assert result is True

    async def test_validate_position_exposure_exceeded(self, service):
        """Test position exposure validation with exposure exceeded."""
        from datetime import datetime, timezone
        large_position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("10.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0.00"),
            realized_pnl=Decimal("0.00"),
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance"
        )
        
        with patch.object(service, '_get_portfolio_value', return_value=Decimal("1000")):  # Small portfolio
            
            result = await service._validate_position_exposure(large_position)
            assert result is False

    async def test_validate_position_exposure_zero_portfolio(self, service, sample_positions):
        """Test position exposure validation with zero portfolio value."""
        position = sample_positions[0]
        
        with patch.object(service, '_get_portfolio_value', return_value=ZERO):
            
            result = await service._validate_position_exposure(position)
            assert result is True

    async def test_validate_position_exposure_exception(self, service, sample_positions):
        """Test position exposure validation with exception."""
        position = sample_positions[0]
        
        with patch.object(service, '_get_portfolio_value', side_effect=Exception("Error")):
            
            result = await service._validate_position_exposure(position)
            assert result is True

    async def test_check_symbol_position_limits_success(self, service):
        """Test successful symbol position limits check."""
        with patch.object(service, '_get_positions_for_symbol', return_value=[]), \
             patch.object(service, '_get_max_positions_per_symbol', return_value=3):
            
            result = await service._check_symbol_position_limits("BTC/USDT")
            assert result is True

    async def test_check_symbol_position_limits_exceeded(self, service, sample_positions):
        """Test symbol position limits check with limit exceeded."""
        symbol_positions = sample_positions * 5  # More than max
        
        with patch.object(service, '_get_positions_for_symbol', return_value=symbol_positions), \
             patch.object(service, '_get_max_positions_per_symbol', return_value=3):
            
            result = await service._check_symbol_position_limits("BTC/USDT")
            assert result is False

    async def test_check_symbol_position_limits_exception(self, service):
        """Test symbol position limits check with exception."""
        with patch.object(service, '_get_positions_for_symbol', side_effect=Exception("Error")):
            
            result = await service._check_symbol_position_limits("BTC/USDT")
            assert result is True

    async def test_get_current_risk_level_success(self, service, mock_state_service):
        """Test successful current risk level retrieval."""
        mock_state_service.get_state.return_value = {"risk_level": "LOW"}
        
        result = await service._get_current_risk_level()
        assert result == RiskLevel.LOW
        mock_state_service.get_state.assert_called_once_with(StateType.RISK_STATE, "current_level")

    async def test_get_current_risk_level_no_state(self, service, mock_state_service):
        """Test current risk level retrieval with no state."""
        mock_state_service.get_state.return_value = None
        
        result = await service._get_current_risk_level()
        assert result == RiskLevel.LOW

    async def test_get_current_risk_level_exception(self, service, mock_state_service):
        """Test current risk level retrieval with exception."""
        mock_state_service.get_state.side_effect = Exception("State error")
        
        result = await service._get_current_risk_level()
        assert result == RiskLevel.LOW

    async def test_is_emergency_stop_active_true(self, service, mock_state_service):
        """Test emergency stop check when active."""
        mock_state_service.get_state.return_value = {"active": True}
        
        result = await service._is_emergency_stop_active()
        assert result is True
        mock_state_service.get_state.assert_called_once_with(StateType.RISK_STATE, "emergency_stop")

    async def test_is_emergency_stop_active_false(self, service, mock_state_service):
        """Test emergency stop check when not active."""
        mock_state_service.get_state.return_value = {"active": False}
        
        result = await service._is_emergency_stop_active()
        assert result is False

    async def test_is_emergency_stop_active_no_state(self, service, mock_state_service):
        """Test emergency stop check with no state."""
        mock_state_service.get_state.return_value = None
        
        result = await service._is_emergency_stop_active()
        assert result is False

    async def test_is_emergency_stop_active_exception(self, service, mock_state_service):
        """Test emergency stop check with exception."""
        mock_state_service.get_state.side_effect = Exception("State error")
        
        result = await service._is_emergency_stop_active()
        assert result is False

    async def test_get_current_positions_success(self, service, mock_state_service, sample_positions):
        """Test successful current positions retrieval."""
        mock_state_service.get_state.return_value = {"open_positions": sample_positions}
        
        result = await service._get_current_positions()
        assert len(result) == len(sample_positions)
        mock_state_service.get_state.assert_called_once_with(StateType.PORTFOLIO_STATE, "positions")

    async def test_get_current_positions_no_state(self, service, mock_state_service):
        """Test current positions retrieval with no state."""
        mock_state_service.get_state.return_value = None
        
        result = await service._get_current_positions()
        assert result == []

    async def test_get_current_positions_exception(self, service, mock_state_service):
        """Test current positions retrieval with exception."""
        mock_state_service.get_state.side_effect = Exception("State error")
        
        result = await service._get_current_positions()
        assert result == []

    async def test_get_positions_for_symbol_success(self, service, sample_positions):
        """Test successful positions for symbol retrieval."""
        with patch.object(service, '_get_current_positions', return_value=sample_positions):
            
            result = await service._get_positions_for_symbol("BTC/USDT")
            btc_positions = [pos for pos in sample_positions if pos.symbol == "BTC/USDT"]
            assert len(result) == len(btc_positions)

    async def test_get_positions_for_symbol_exception(self, service):
        """Test positions for symbol retrieval with exception."""
        with patch.object(service, '_get_current_positions', side_effect=Exception("Error")):
            
            result = await service._get_positions_for_symbol("BTC/USDT")
            assert result == []

    async def test_get_portfolio_value(self, service):
        """Test portfolio value retrieval (not implemented)."""
        result = await service._get_portfolio_value()
        assert result == ZERO

    async def test_get_current_exposure(self, service):
        """Test current exposure retrieval (not implemented)."""
        result = await service._get_current_exposure()
        assert result == ZERO

    async def test_get_current_price(self, service):
        """Test current price retrieval (not implemented)."""
        result = await service._get_current_price("BTC/USDT")
        assert result is None

    def test_get_max_total_positions_with_config(self, service):
        """Test getting max total positions from config."""
        result = service._get_max_total_positions()
        assert result == 10

    def test_get_max_total_positions_without_config(self, service_without_config):
        """Test getting max total positions without config."""
        result = service_without_config._get_max_total_positions()
        assert result == 10

    def test_get_max_positions_per_symbol_with_config(self, service):
        """Test getting max positions per symbol from config."""
        result = service._get_max_positions_per_symbol()
        assert result == 3

    def test_get_max_positions_per_symbol_without_config(self, service_without_config):
        """Test getting max positions per symbol without config."""
        result = service_without_config._get_max_positions_per_symbol()
        assert result == 3

    def test_get_max_position_size_pct_with_config(self, service):
        """Test getting max position size percentage from config."""
        result = service._get_max_position_size_pct()
        assert result == Decimal("0.25")

    def test_get_max_position_size_pct_without_config(self, service_without_config):
        """Test getting max position size percentage without config."""
        result = service_without_config._get_max_position_size_pct()
        assert result == Decimal("0.25")

    def test_get_max_portfolio_exposure_pct_with_config(self, service):
        """Test getting max portfolio exposure percentage from config."""
        result = service._get_max_portfolio_exposure_pct()
        assert result == Decimal("0.80")

    def test_get_max_portfolio_exposure_pct_without_config(self, service_without_config):
        """Test getting max portfolio exposure percentage without config."""
        result = service_without_config._get_max_portfolio_exposure_pct()
        assert result == Decimal("0.80")