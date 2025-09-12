"""Tests for risk management controller."""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
import pytest

from src.core.exceptions import RiskManagementError, ValidationError
from src.core.types import MarketData, OrderRequest, Position, RiskMetrics, Signal, OrderSide, OrderType, RiskLevel, SignalDirection, PositionSide, PositionStatus
from src.risk_management.controller import RiskManagementController
from src.utils.messaging_patterns import MessagingCoordinator


class TestRiskManagementController:
    """Test risk management controller functionality."""

    @pytest.fixture
    def mock_position_sizing_service(self):
        """Create mock position sizing service."""
        return AsyncMock()

    @pytest.fixture
    def mock_risk_validation_service(self):
        """Create mock risk validation service."""
        return AsyncMock()

    @pytest.fixture
    def mock_risk_metrics_service(self):
        """Create mock risk metrics service."""
        return AsyncMock()

    @pytest.fixture
    def mock_risk_monitoring_service(self):
        """Create mock risk monitoring service."""
        return AsyncMock()

    @pytest.fixture
    def mock_portfolio_limits_service(self):
        """Create mock portfolio limits service."""
        return AsyncMock()

    @pytest.fixture
    def mock_messaging_coordinator(self):
        """Create mock messaging coordinator."""
        return Mock(spec=MessagingCoordinator)

    @pytest.fixture
    def controller(
        self, 
        mock_position_sizing_service,
        mock_risk_validation_service,
        mock_risk_metrics_service,
        mock_risk_monitoring_service,
        mock_portfolio_limits_service,
        mock_messaging_coordinator
    ):
        """Create risk management controller."""
        return RiskManagementController(
            position_sizing_service=mock_position_sizing_service,
            risk_validation_service=mock_risk_validation_service,
            risk_metrics_service=mock_risk_metrics_service,
            risk_monitoring_service=mock_risk_monitoring_service,
            portfolio_limits_service=mock_portfolio_limits_service,
            messaging_coordinator=mock_messaging_coordinator,
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
                exchange="binance"
            ),
            Position(
                symbol="ETH/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000.00"),
                current_price=Decimal("3100.00"),
                unrealized_pnl=Decimal("100.00"),
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="binance"
            )
        ]

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        timestamp = datetime.now(timezone.utc)
        return [
            MarketData(
                symbol="BTC/USDT",
                open=Decimal("49500.00"),
                high=Decimal("50200.00"),
                low=Decimal("49000.00"),
                close=Decimal("50000.00"),
                volume=Decimal("1000.00"),
                timestamp=timestamp,
                exchange="binance",
                bid_price=Decimal("49990.00"),
                ask_price=Decimal("50010.00")
            ),
            MarketData(
                symbol="ETH/USDT",
                open=Decimal("3050.00"),
                high=Decimal("3120.00"),
                low=Decimal("3020.00"),
                close=Decimal("3100.00"),
                volume=Decimal("5000.00"),
                timestamp=timestamp,
                exchange="binance",
                bid_price=Decimal("3095.00"),
                ask_price=Decimal("3105.00")
            )
        ]

    @pytest.fixture
    def sample_risk_metrics(self):
        """Create sample risk metrics."""
        return RiskMetrics(
            portfolio_value=Decimal("10000.00"),
            total_exposure=Decimal("5000.00"),
            var_1d=Decimal("-200.00"),
            risk_level=RiskLevel.MEDIUM,
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.05")
        )

    def test_controller_initialization(self, controller, mock_messaging_coordinator):
        """Test controller initialization."""
        assert controller is not None
        assert hasattr(controller, "_position_sizing_service")
        assert hasattr(controller, "_risk_validation_service")
        assert hasattr(controller, "_risk_metrics_service")
        assert hasattr(controller, "_risk_monitoring_service")
        assert hasattr(controller, "_messaging_coordinator")
        assert hasattr(controller, "_request_count")
        assert controller._request_count == 0

    def test_controller_initialization_without_messaging_coordinator(
        self, 
        mock_position_sizing_service,
        mock_risk_validation_service,
        mock_risk_metrics_service,
        mock_risk_monitoring_service,
        mock_portfolio_limits_service
    ):
        """Test controller initialization without messaging coordinator."""
        controller = RiskManagementController(
            position_sizing_service=mock_position_sizing_service,
            risk_validation_service=mock_risk_validation_service,
            risk_metrics_service=mock_risk_metrics_service,
            risk_monitoring_service=mock_risk_monitoring_service,
            portfolio_limits_service=mock_portfolio_limits_service
        )
        
        assert controller._messaging_coordinator is not None
        assert isinstance(controller._messaging_coordinator, MessagingCoordinator)

    async def test_calculate_position_size_success(self, controller, mock_position_sizing_service, sample_signal):
        """Test successful position size calculation."""
        available_capital = Decimal("10000.00")
        current_price = Decimal("50000.00")
        expected_size = Decimal("0.2")
        
        mock_position_sizing_service.calculate_size.return_value = expected_size
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            result = await controller.calculate_position_size(
                signal=sample_signal,
                available_capital=available_capital,
                current_price=current_price,
                method="kelly"
            )
        
        assert result == expected_size
        mock_position_sizing_service.calculate_size.assert_called_once_with(
            signal=sample_signal,
            available_capital=available_capital,
            current_price=current_price,
            method="kelly"
        )

    async def test_calculate_position_size_without_method(self, controller, mock_position_sizing_service, sample_signal):
        """Test position size calculation without method specified."""
        available_capital = Decimal("10000.00")
        current_price = Decimal("50000.00")
        expected_size = Decimal("0.2")
        
        mock_position_sizing_service.calculate_size.return_value = expected_size
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            result = await controller.calculate_position_size(
                signal=sample_signal,
                available_capital=available_capital,
                current_price=current_price
            )
        
        assert result == expected_size
        mock_position_sizing_service.calculate_size.assert_called_once_with(
            signal=sample_signal,
            available_capital=available_capital,
            current_price=current_price,
            method=None
        )

    async def test_calculate_position_size_service_error(self, controller, mock_position_sizing_service, sample_signal):
        """Test position size calculation with service error."""
        available_capital = Decimal("10000.00")
        current_price = Decimal("50000.00")
        
        mock_position_sizing_service.calculate_size.side_effect = RiskManagementError("Service error")
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'), \
             patch.object(controller, 'propagate_service_error') as mock_propagate:
            
            with pytest.raises(RiskManagementError):
                await controller.calculate_position_size(
                    signal=sample_signal,
                    available_capital=available_capital,
                    current_price=current_price
                )
            
            mock_propagate.assert_called_once()

    async def test_calculate_position_size_validation_error(self, controller, sample_signal):
        """Test position size calculation with validation error."""
        available_capital = Decimal("10000.00")
        current_price = Decimal("50000.00")
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity', 
                  side_effect=ValidationError("Invalid input")), \
             patch.object(controller, 'propagate_service_error') as mock_propagate:
            
            with pytest.raises(ValidationError):
                await controller.calculate_position_size(
                    signal=sample_signal,
                    available_capital=available_capital,
                    current_price=current_price
                )
            
            mock_propagate.assert_called_once()

    async def test_validate_signal_success(self, controller, mock_risk_validation_service, sample_signal):
        """Test successful signal validation."""
        mock_risk_validation_service.validate_signal.return_value = True
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            result = await controller.validate_signal(sample_signal)
        
        assert result is True
        mock_risk_validation_service.validate_signal.assert_called_once_with(sample_signal)

    async def test_validate_signal_failure(self, controller, mock_risk_validation_service, sample_signal):
        """Test signal validation failure."""
        mock_risk_validation_service.validate_signal.return_value = False
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            result = await controller.validate_signal(sample_signal)
        
        assert result is False
        mock_risk_validation_service.validate_signal.assert_called_once_with(sample_signal)

    async def test_validate_signal_exception(self, controller, mock_risk_validation_service, sample_signal):
        """Test signal validation with exception."""
        mock_risk_validation_service.validate_signal.side_effect = Exception("Validation error")
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'), \
             patch.object(controller, 'propagate_validation_error') as mock_propagate:
            
            result = await controller.validate_signal(sample_signal)
            
            assert result is False
            mock_propagate.assert_called_once()

    async def test_validate_order_success(self, controller, mock_risk_validation_service, sample_order_request):
        """Test successful order validation."""
        mock_risk_validation_service.validate_order.return_value = True
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            result = await controller.validate_order(sample_order_request)
        
        assert result is True
        mock_risk_validation_service.validate_order.assert_called_once_with(sample_order_request)

    async def test_validate_order_failure(self, controller, mock_risk_validation_service, sample_order_request):
        """Test order validation failure."""
        mock_risk_validation_service.validate_order.return_value = False
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            result = await controller.validate_order(sample_order_request)
        
        assert result is False
        mock_risk_validation_service.validate_order.assert_called_once_with(sample_order_request)

    async def test_validate_order_exception(self, controller, mock_risk_validation_service, sample_order_request):
        """Test order validation with exception."""
        mock_risk_validation_service.validate_order.side_effect = Exception("Validation error")
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'), \
             patch.object(controller, 'propagate_validation_error') as mock_propagate:
            
            result = await controller.validate_order(sample_order_request)
            
            assert result is False
            mock_propagate.assert_called_once()

    async def test_calculate_risk_metrics_success(
        self, 
        controller, 
        mock_risk_metrics_service, 
        sample_positions, 
        sample_market_data, 
        sample_risk_metrics
    ):
        """Test successful risk metrics calculation."""
        mock_risk_metrics_service.calculate_metrics.return_value = sample_risk_metrics
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'), \
             patch('src.risk_management.data_transformer.RiskDataTransformer.transform_position_to_event_data',
                  return_value={"transformed": "data"}):
            
            result = await controller.calculate_risk_metrics(sample_positions, sample_market_data)
        
        assert result == sample_risk_metrics
        mock_risk_metrics_service.calculate_metrics.assert_called_once_with(
            positions=sample_positions,
            market_data=sample_market_data
        )

    async def test_calculate_risk_metrics_empty_lists(self, controller, mock_risk_metrics_service, sample_risk_metrics):
        """Test risk metrics calculation with empty lists."""
        mock_risk_metrics_service.calculate_metrics.return_value = sample_risk_metrics
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'):
            result = await controller.calculate_risk_metrics([], [])
        
        assert result == sample_risk_metrics
        mock_risk_metrics_service.calculate_metrics.assert_called_once_with(
            positions=[],
            market_data=[]
        )

    async def test_calculate_risk_metrics_service_error(
        self, 
        controller, 
        mock_risk_metrics_service, 
        sample_positions, 
        sample_market_data
    ):
        """Test risk metrics calculation with service error."""
        mock_risk_metrics_service.calculate_metrics.side_effect = RiskManagementError("Calculation failed")
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'), \
             patch('src.risk_management.data_transformer.RiskDataTransformer.transform_position_to_event_data',
                  return_value={"transformed": "data"}), \
             patch.object(controller, 'propagate_service_error') as mock_propagate:
            
            with pytest.raises(RiskManagementError):
                await controller.calculate_risk_metrics(sample_positions, sample_market_data)
            
            mock_propagate.assert_called_once()

    async def test_calculate_risk_metrics_transformer_import_error(
        self, 
        controller, 
        mock_risk_metrics_service, 
        sample_positions, 
        sample_market_data
    ):
        """Test risk metrics calculation with transformer import error."""
        mock_risk_metrics_service.calculate_metrics.side_effect = ImportError("Cannot import transformer")
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'), \
             patch.object(controller, 'propagate_service_error') as mock_propagate:
            
            with pytest.raises(ImportError):
                await controller.calculate_risk_metrics(sample_positions, sample_market_data)
            
            mock_propagate.assert_called_once()

    async def test_validate_portfolio_limits_success(self, controller, mock_portfolio_limits_service, sample_positions):
        """Test successful portfolio limits validation."""
        new_position = sample_positions[0]
        mock_portfolio_limits_service.check_portfolio_limits.return_value = True
        
        result = await controller.validate_portfolio_limits(new_position)
        
        assert result is True
        mock_portfolio_limits_service.check_portfolio_limits.assert_called_once_with(new_position)

    async def test_validate_portfolio_limits_failure(self, controller, mock_portfolio_limits_service, sample_positions):
        """Test portfolio limits validation failure."""
        new_position = sample_positions[0]
        mock_portfolio_limits_service.check_portfolio_limits.return_value = False
        
        result = await controller.validate_portfolio_limits(new_position)
        
        assert result is False
        mock_portfolio_limits_service.check_portfolio_limits.assert_called_once_with(new_position)

    async def test_validate_portfolio_limits_exception(self, controller, mock_portfolio_limits_service, sample_positions):
        """Test portfolio limits validation with exception."""
        new_position = sample_positions[0]
        mock_portfolio_limits_service.check_portfolio_limits.side_effect = Exception("Validation error")
        
        result = await controller.validate_portfolio_limits(new_position)
        
        assert result is False

    async def test_start_monitoring_success(self, controller, mock_risk_monitoring_service):
        """Test successful monitoring start."""
        interval = 120
        
        await controller.start_monitoring(interval)
        
        mock_risk_monitoring_service.start_monitoring.assert_called_once_with(interval)

    async def test_start_monitoring_default_interval(self, controller, mock_risk_monitoring_service):
        """Test monitoring start with default interval."""
        await controller.start_monitoring()
        
        mock_risk_monitoring_service.start_monitoring.assert_called_once_with(60)

    async def test_start_monitoring_error(self, controller, mock_risk_monitoring_service):
        """Test monitoring start with error."""
        mock_risk_monitoring_service.start_monitoring.side_effect = Exception("Failed to start")
        
        with pytest.raises(Exception, match="Failed to start"):
            await controller.start_monitoring()

    async def test_stop_monitoring_success(self, controller, mock_risk_monitoring_service):
        """Test successful monitoring stop."""
        await controller.stop_monitoring()
        
        mock_risk_monitoring_service.stop_monitoring.assert_called_once()

    async def test_stop_monitoring_error(self, controller, mock_risk_monitoring_service):
        """Test monitoring stop with error."""
        mock_risk_monitoring_service.stop_monitoring.side_effect = Exception("Failed to stop")
        
        with pytest.raises(Exception, match="Failed to stop"):
            await controller.stop_monitoring()

    async def test_get_risk_summary_success(self, controller, mock_risk_monitoring_service):
        """Test successful risk summary retrieval."""
        expected_summary = {
            "portfolio_value": 10000.0,
            "total_exposure": 5000.0,
            "risk_level": "MEDIUM"
        }
        mock_risk_monitoring_service.get_risk_summary.return_value = expected_summary
        
        result = await controller.get_risk_summary()
        
        assert result == expected_summary
        mock_risk_monitoring_service.get_risk_summary.assert_called_once()

    async def test_get_risk_summary_error(self, controller, mock_risk_monitoring_service):
        """Test risk summary retrieval with error."""
        mock_risk_monitoring_service.get_risk_summary.side_effect = Exception("Service unavailable")
        
        result = await controller.get_risk_summary()
        
        assert "error" in result
        assert result["error"] == "Service unavailable"
        assert "timestamp" in result

    async def test_get_risk_summary_empty_response(self, controller, mock_risk_monitoring_service):
        """Test risk summary with empty response."""
        mock_risk_monitoring_service.get_risk_summary.return_value = {}
        
        result = await controller.get_risk_summary()
        
        assert result == {}
        mock_risk_monitoring_service.get_risk_summary.assert_called_once()

    def test_controller_attributes_after_initialization(
        self, 
        mock_position_sizing_service,
        mock_risk_validation_service,
        mock_risk_metrics_service,
        mock_risk_monitoring_service,
        mock_portfolio_limits_service
    ):
        """Test controller attributes are properly set after initialization."""
        controller = RiskManagementController(
            position_sizing_service=mock_position_sizing_service,
            risk_validation_service=mock_risk_validation_service,
            risk_metrics_service=mock_risk_metrics_service,
            risk_monitoring_service=mock_risk_monitoring_service,
            portfolio_limits_service=mock_portfolio_limits_service,
            correlation_id="test-id"
        )
        
        assert controller._position_sizing_service == mock_position_sizing_service
        assert controller._risk_validation_service == mock_risk_validation_service
        assert controller._risk_metrics_service == mock_risk_metrics_service
        assert controller._risk_monitoring_service == mock_risk_monitoring_service
        assert controller._portfolio_limits_service == mock_portfolio_limits_service
        assert controller._request_count == 0

    async def test_multiple_operations_sequence(
        self, 
        controller, 
        mock_position_sizing_service,
        mock_risk_validation_service,
        mock_risk_metrics_service,
        mock_risk_monitoring_service,
        sample_signal,
        sample_order_request,
        sample_positions,
        sample_market_data,
        sample_risk_metrics
    ):
        """Test sequence of multiple operations."""
        # Setup mocks
        mock_position_sizing_service.calculate_size.return_value = Decimal("0.2")
        mock_risk_validation_service.validate_signal.return_value = True
        mock_risk_validation_service.validate_order.return_value = True
        mock_risk_metrics_service.calculate_metrics.return_value = sample_risk_metrics
        mock_risk_monitoring_service.get_risk_summary.return_value = {"status": "healthy"}
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'), \
             patch('src.risk_management.data_transformer.RiskDataTransformer.transform_position_to_event_data',
                  return_value={"transformed": "data"}):
            
            # Execute sequence of operations
            position_size = await controller.calculate_position_size(
                sample_signal, Decimal("10000.00"), Decimal("50000.00")
            )
            signal_valid = await controller.validate_signal(sample_signal)
            order_valid = await controller.validate_order(sample_order_request)
            metrics = await controller.calculate_risk_metrics(sample_positions, sample_market_data)
            summary = await controller.get_risk_summary()
            
            # Verify results
            assert position_size == Decimal("0.2")
            assert signal_valid is True
            assert order_valid is True
            assert metrics == sample_risk_metrics
            assert summary == {"status": "healthy"}
            
            # Verify all services were called
            mock_position_sizing_service.calculate_size.assert_called_once()
            mock_risk_validation_service.validate_signal.assert_called_once()
            mock_risk_validation_service.validate_order.assert_called_once()
            mock_risk_metrics_service.calculate_metrics.assert_called_once()
            mock_risk_monitoring_service.get_risk_summary.assert_called_once()

    async def test_boundary_validation_calls(
        self, 
        controller, 
        mock_position_sizing_service,
        sample_signal
    ):
        """Test that boundary validation is properly called."""
        mock_position_sizing_service.calculate_size.return_value = Decimal("0.2")
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity') as mock_validator:
            await controller.calculate_position_size(
                sample_signal, Decimal("10000.00"), Decimal("50000.00"), "kelly"
            )
            
            mock_validator.assert_called_once()
            call_args = mock_validator.call_args[0]
            assert "signal" in call_args[0]
            assert "available_capital" in call_args[0]
            assert "current_price" in call_args[0]
            assert call_args[1] == "calculate_position_size"

    def test_inheritance_and_mixins(self, controller):
        """Test that controller properly inherits from base classes."""
        from src.core.base.component import BaseComponent
        from src.utils.messaging_patterns import ErrorPropagationMixin
        
        assert isinstance(controller, BaseComponent)
        assert isinstance(controller, ErrorPropagationMixin)
        assert hasattr(controller, 'propagate_service_error')
        assert hasattr(controller, 'propagate_validation_error')
        assert hasattr(controller, '_logger')

    async def test_logging_calls(self, controller, mock_position_sizing_service, sample_signal):
        """Test that logging is properly called."""
        mock_position_sizing_service.calculate_size.return_value = Decimal("0.2")
        
        with patch('src.utils.messaging_patterns.BoundaryValidator.validate_database_entity'), \
             patch.object(controller, '_logger') as mock_logger:
            
            await controller.calculate_position_size(
                sample_signal, Decimal("10000.00"), Decimal("50000.00")
            )
            
            # Verify logging calls
            assert mock_logger.info.call_count >= 2  # Start and end logging
            
            # Check log messages
            call_args_list = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Calculating position size" in msg for msg in call_args_list)
            assert any("Position size calculated" in msg for msg in call_args_list)