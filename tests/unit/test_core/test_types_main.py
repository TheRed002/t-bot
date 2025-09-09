"""
Unit tests for core types main module.

Tests to ensure all types can be imported from the main types.py file and verify the __all__ export list.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone


class TestTypesMainImports:
    """Test that all types can be imported from main types module."""

    def test_base_types_import(self):
        """Test base types can be imported."""
        from src.core.types import (
            AlertSeverity,
            BaseValidatedModel,
            ConfigDict,
            ConnectionType,
            ExchangeType,
            MarketType,
            RequestType,
            TradingMode,
            ValidationLevel,
            ValidationResult,
        )

        # Test enums have correct values
        assert hasattr(AlertSeverity, 'HIGH')
        assert hasattr(ConnectionType, 'TICKER')
        assert hasattr(ExchangeType, 'BINANCE')
        assert hasattr(ValidationLevel, 'CRITICAL')
        assert hasattr(ValidationResult, 'PASS')
        assert hasattr(TradingMode, 'LIVE')

    def test_bot_types_import(self):
        """Test bot types can be imported."""
        from src.core.types import (
            BotConfiguration,
            BotPriority,
            BotState,
            BotStatus,
            BotType,
        )

        # Test enums exist
        assert hasattr(BotPriority, 'HIGH')
        assert hasattr(BotStatus, 'RUNNING')
        assert hasattr(BotType, 'TRADING')
        # BotState is a model, not enum - test it has required fields
        assert hasattr(BotState, 'model_fields')

    def test_data_types_import(self):
        """Test data types can be imported."""
        from src.core.types import (
            DriftType,
            ErrorPattern,
            FeatureSet,
            IngestionMode,
            MLMarketData,
            PipelineStatus,
            PredictionResult,
            ProcessingStep,
            QualityLevel,
            StorageMode,
        )

        # Test data models can be instantiated
        assert FeatureSet
        assert MLMarketData
        assert PredictionResult

    def test_execution_types_import(self):
        """Test execution types can be imported."""
        from src.core.types import (
            ExecutionAlgorithm,
            ExecutionInstruction,
            ExecutionResult,
            ExecutionStatus,
        )

        # Test execution models exist
        assert ExecutionAlgorithm
        assert ExecutionInstruction
        assert ExecutionResult
        assert ExecutionStatus

    def test_market_types_import(self):
        """Test market types can be imported."""
        from src.core.types import (
            ExchangeGeneralInfo,
            ExchangeInfo,
            ExchangeStatus,
            MarketData,
            OrderBook,
            OrderBookLevel,
            Ticker,
            Trade,
        )

        # Test market models exist
        assert ExchangeGeneralInfo
        assert ExchangeInfo
        assert ExchangeStatus
        assert MarketData
        assert OrderBook
        assert OrderBookLevel
        assert Ticker
        assert Trade

    def test_risk_types_import(self):
        """Test risk types can be imported."""
        from src.core.types import (
            AllocationStrategy,
            CapitalAllocation,
            CapitalMetrics,
            CapitalProtection,
            CircuitBreakerEvent,
            CircuitBreakerStatus,
            CircuitBreakerType,
            CurrencyExposure,
            EmergencyAction,
            ExchangeAllocation,
            FundFlow,
            PortfolioState,
            PositionLimits,
            PositionSizeMethod,
            RiskAlert,
            RiskLevel,
            RiskLimits,
            RiskMetrics,
            WithdrawalRule,
        )

        # Test risk models exist
        assert AllocationStrategy
        assert CapitalAllocation
        assert CapitalMetrics
        assert CapitalProtection
        assert CircuitBreakerEvent
        assert RiskLevel

    def test_strategy_types_import(self):
        """Test strategy types can be imported."""
        from src.core.types import (
            MarketRegime,
            RegimeChangeEvent,
            Signal,
            SignalDirection,
            StrategyConfig,
            StrategyMetrics,
            StrategyStatus,
            StrategyType,
        )

        # Test strategy models exist
        assert MarketRegime
        assert RegimeChangeEvent
        assert Signal
        assert SignalDirection
        assert StrategyConfig
        assert StrategyMetrics

    def test_trading_types_import(self):
        """Test trading types can be imported."""
        from src.core.types import (
            Balance,
            Order,
            OrderRequest,
            OrderResponse,
            OrderSide,
            OrderStatus,
            OrderType,
            Position,
            PositionSide,
            PositionStatus,
            TimeInForce,
        )

        # Test trading models exist
        assert Balance
        assert Order
        assert OrderRequest
        assert OrderResponse
        assert OrderSide
        assert OrderStatus
        assert OrderType

    def test_all_exports_list(self):
        """Test that __all__ list contains all expected exports."""
        from src.core.types import __all__

        # Verify __all__ is a list and contains expected number of items
        assert isinstance(__all__, list)
        assert len(__all__) > 50  # Should have many exports

        # Test some key exports are present
        expected_exports = [
            "AlertSeverity",
            "AllocationStrategy",
            "Balance",
            "BaseValidatedModel",
            "BotConfiguration",
            "CapitalAllocation",
            "CapitalMetrics",
            "CircuitBreakerEvent",
            "ExchangeInfo",
            "ExecutionResult",
            "MarketData",
            "Order",
            "RiskLevel",
            "Signal",
            "TradingMode",
        ]

        for export in expected_exports:
            assert export in __all__, f"{export} not found in __all__"

    def test_type_instantiation_examples(self):
        """Test that key types can be instantiated with proper data."""
        from src.core.types import (
            ExchangeInfo,
            RiskMetrics,
            CapitalAllocation,
            OrderBook,
        )
        from src.core.types.market import OrderBookLevel

        # Test ExchangeInfo creation
        exchange_info = ExchangeInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            min_price=Decimal("0.01"),
            max_price=Decimal("1000000.00"),
            tick_size=Decimal("0.01"),
            min_quantity=Decimal("0.00001"),
            max_quantity=Decimal("1000.00"),
            step_size=Decimal("0.00001"),
            exchange="binance",
        )
        assert exchange_info.symbol == "BTCUSDT"

        # Test RiskMetrics creation
        risk_metrics = RiskMetrics(
            portfolio_value=Decimal("100000"),
            total_exposure=Decimal("80000"),
            var_1d=Decimal("0.05"),
            risk_level="medium",
            timestamp=datetime.now(timezone.utc),
        )
        assert risk_metrics.portfolio_value == Decimal("100000")

        # Test CapitalAllocation creation
        capital_allocation = CapitalAllocation(
            allocation_id="alloc_001",
            allocated_amount=Decimal("10000.00"),
            utilized_amount=Decimal("5000.00"),
            available_amount=Decimal("5000.00"),
            allocation_percentage=Decimal("0.1"),
            target_allocation_pct=Decimal("0.15"),
            min_allocation=Decimal("1000.00"),
            max_allocation=Decimal("20000.00"),
            last_rebalance=datetime.now(timezone.utc),
        )
        assert capital_allocation.allocation_id == "alloc_001"

        # Test OrderBook creation  
        order_book = OrderBook(
            symbol="BTC/USDT",
            bids=[
                OrderBookLevel(price=Decimal("50000"), quantity=Decimal("1.5")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("50001"), quantity=Decimal("1.2")),
            ],
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )
        assert order_book.symbol == "BTC/USDT"

    def test_enum_values_consistency(self):
        """Test that enum values are consistent and properly defined."""
        from src.core.types import (
            ExchangeType,
            OrderSide,
            OrderStatus,
            TradingMode,
            RiskLevel,
        )

        # Test ExchangeType
        assert ExchangeType.BINANCE.value == "binance"
        assert ExchangeType.COINBASE.value == "coinbase"
        assert ExchangeType.OKX.value == "okx"

        # Test OrderSide
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

        # Test OrderStatus
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"

        # Test TradingMode
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.BACKTEST.value == "backtest"

        # Test RiskLevel
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling for type definitions."""
        from src.core.types import ExchangeInfo

        # Test validation error on missing required field
        with pytest.raises(Exception):
            ExchangeInfo(
                symbol="BTCUSDT",
                # Missing required fields should cause validation error
            )

    def test_decimal_precision_handling(self):
        """Test that Decimal types maintain precision as required for financial calculations."""
        from src.core.types import CapitalMetrics

        # Test high precision decimals
        high_precision_capital = Decimal("123456.12345678")
        
        metrics = CapitalMetrics(
            total_capital=high_precision_capital,
            allocated_amount=Decimal("100000.00000000"),
            available_amount=Decimal("23456.12345678"),
            total_pnl=Decimal("5000.12345678"),
            realized_pnl=Decimal("3000.12345678"),
            unrealized_pnl=Decimal("2000.00000000"),
            daily_return=Decimal("0.05123456"),
            weekly_return=Decimal("0.12345678"),
            monthly_return=Decimal("0.25000000"),
            yearly_return=Decimal("0.45123456"),
            total_return=Decimal("0.50000000"),
            sharpe_ratio=Decimal("1.50000000"),
            sortino_ratio=Decimal("1.80000000"),
            calmar_ratio=Decimal("2.00000000"),
            current_drawdown=Decimal("0.03000000"),
            max_drawdown=Decimal("0.15000000"),
            var_95=Decimal("1000.12345678"),
            expected_shortfall=Decimal("1500.12345678"),
            strategies_active=3,
            positions_open=10,
            leverage_used=Decimal("1.50000000"),
            timestamp=datetime.now(timezone.utc),
        )

        # Verify precision is maintained
        assert metrics.total_capital == high_precision_capital
        assert str(metrics.var_95) == "1000.12345678"