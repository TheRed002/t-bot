"""
Extended unit tests for core type definitions.

These tests cover additional types and enums not covered in the basic types test.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.core.types import (
    AllocationStrategy,
    CapitalAllocation,
    CapitalMetrics,
    CapitalProtection,
    CircuitBreakerEvent,
    CircuitBreakerStatus,
    CircuitBreakerType,
    ConnectionType,
    CurrencyExposure,
    DriftType,
    ErrorPattern,
    ExchangeAllocation,
    ExchangeInfo,
    ExchangeStatus,
    ExchangeType,
    FundFlow,
    IngestionMode,
    MarketRegime,
    NewsSentiment,
    OrderBook,
    OrderSide,
    OrderStatus,
    PipelineStatus,
    PositionLimits,
    PositionSizeMethod,
    ProcessingStep,
    QualityLevel,
    RegimeChangeEvent,
    RequestType,
    RiskLevel,
    RiskMetrics,
    SocialSentiment,
    StorageMode,
    StrategyConfig,
    StrategyMetrics,
    StrategyStatus,
    StrategyType,
    TimeInForce,
    TradingMode,
    ValidationLevel,
    ValidationResult,
    WithdrawalRule,
)


class TestExchangeType:
    """Test ExchangeType enum values."""

    def test_exchange_type_enum(self):
        """Test ExchangeType enum values."""
        assert ExchangeType.BINANCE.value == "binance"
        assert ExchangeType.OKX.value == "okx"
        assert ExchangeType.COINBASE.value == "coinbase"


class TestRequestType:
    """Test RequestType enum values."""

    def test_request_type_enum(self):
        """Test RequestType enum values."""
        assert RequestType.MARKET_DATA.value == "market_data"
        assert RequestType.ORDER_PLACEMENT.value == "order_placement"
        assert RequestType.ORDER_CANCELLATION.value == "order_cancellation"
        assert RequestType.BALANCE_QUERY.value == "balance_query"


class TestConnectionType:
    """Test ConnectionType enum values."""

    def test_connection_type_enum(self):
        """Test ConnectionType enum values."""
        assert ConnectionType.TICKER.value == "ticker"
        assert ConnectionType.ORDERBOOK.value == "orderbook"
        assert ConnectionType.TRADES.value == "trades"
        assert ConnectionType.USER_DATA.value == "user_data"
        assert ConnectionType.MARKET_DATA.value == "market_data"


class TestValidationLevel:
    """Test ValidationLevel enum values."""

    def test_validation_level_enum(self):
        """Test ValidationLevel enum values."""
        assert ValidationLevel.CRITICAL.value == "critical"
        assert ValidationLevel.HIGH.value == "high"
        assert ValidationLevel.MEDIUM.value == "medium"
        assert ValidationLevel.LOW.value == "low"


class TestValidationResult:
    """Test ValidationResult enum values."""

    def test_validation_result_enum(self):
        """Test ValidationResult enum values."""
        assert ValidationResult.PASS.value == "pass"
        assert ValidationResult.FAIL.value == "fail"
        assert ValidationResult.WARNING.value == "warning"
        assert ValidationResult.SKIP.value == "skip"


class TestQualityLevel:
    """Test QualityLevel enum values."""

    def test_quality_level_enum(self):
        """Test QualityLevel enum values."""
        assert QualityLevel.EXCELLENT.value == "excellent"
        assert QualityLevel.GOOD.value == "good"
        assert QualityLevel.ACCEPTABLE.value == "acceptable"
        assert QualityLevel.POOR.value == "poor"
        assert QualityLevel.UNUSABLE.value == "unusable"


class TestIngestionMode:
    """Test IngestionMode enum values."""

    def test_ingestion_mode_enum(self):
        """Test IngestionMode enum values."""
        assert IngestionMode.BATCH.value == "batch"
        assert IngestionMode.STREAMING.value == "streaming"
        assert IngestionMode.HYBRID.value == "hybrid"
        assert IngestionMode.MANUAL.value == "manual"


class TestProcessingStep:
    """Test ProcessingStep enum values."""

    def test_processing_step_enum(self):
        """Test ProcessingStep enum values."""
        assert ProcessingStep.INGESTION.value == "ingestion"
        assert ProcessingStep.VALIDATION.value == "validation"
        assert ProcessingStep.CLEANING.value == "cleaning"
        assert ProcessingStep.TRANSFORMATION.value == "transformation"
        assert ProcessingStep.ENRICHMENT.value == "enrichment"
        assert ProcessingStep.AGGREGATION.value == "aggregation"
        assert ProcessingStep.STORAGE.value == "storage"
        assert ProcessingStep.DISTRIBUTION.value == "distribution"


class TestStorageMode:
    """Test StorageMode enum values."""

    def test_storage_mode_enum(self):
        """Test StorageMode enum values."""
        assert StorageMode.HOT.value == "hot"
        assert StorageMode.WARM.value == "warm"
        assert StorageMode.COLD.value == "cold"
        assert StorageMode.ARCHIVE.value == "archive"


class TestPipelineStatus:
    """Test PipelineStatus enum values."""

    def test_pipeline_status_enum(self):
        """Test PipelineStatus enum values."""
        assert PipelineStatus.IDLE.value == "idle"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.PAUSED.value == "paused"
        assert PipelineStatus.FAILED.value == "failed"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.RETRYING.value == "retrying"


class TestMarketRegime:
    """Test MarketRegime enum values."""

    def test_market_regime_enum(self):
        """Test MarketRegime enum values."""
        assert MarketRegime.TRENDING_UP.value == "trending_up"
        assert MarketRegime.TRENDING_DOWN.value == "trending_down"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"
        assert MarketRegime.LOW_VOLATILITY.value == "low_volatility"
        assert MarketRegime.UNKNOWN.value == "unknown"


class TestDriftType:
    """Test DriftType enum values."""

    def test_drift_type_enum(self):
        """Test DriftType enum values."""
        assert DriftType.CONCEPT.value == "concept"
        assert DriftType.FEATURE.value == "feature"
        assert DriftType.PREDICTION.value == "prediction"
        assert DriftType.LABEL.value == "label"
        assert DriftType.SCHEMA.value == "schema"


class TestSocialSentiment:
    """Test SocialSentiment enum values."""

    def test_social_sentiment_enum(self):
        """Test SocialSentiment enum values."""
        assert SocialSentiment.EXTREMELY_POSITIVE.value == "extremely_positive"
        assert SocialSentiment.POSITIVE.value == "positive"
        assert SocialSentiment.NEUTRAL.value == "neutral"
        assert SocialSentiment.NEGATIVE.value == "negative"
        assert SocialSentiment.EXTREMELY_NEGATIVE.value == "extremely_negative"


class TestNewsSentiment:
    """Test NewsSentiment enum values."""

    def test_news_sentiment_enum(self):
        """Test NewsSentiment enum values."""
        assert NewsSentiment.VERY_BULLISH.value == "very_bullish"
        assert NewsSentiment.BULLISH.value == "bullish"
        assert NewsSentiment.NEUTRAL.value == "neutral"
        assert NewsSentiment.BEARISH.value == "bearish"
        assert NewsSentiment.VERY_BEARISH.value == "very_bearish"


class TestTradingMode:
    """Test TradingMode enum values."""

    def test_trading_mode_enum(self):
        """Test TradingMode enum values."""
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.BACKTEST.value == "backtest"


class TestAllocationStrategy:
    """Test AllocationStrategy enum values."""

    def test_allocation_strategy_enum(self):
        """Test AllocationStrategy enum values."""
        assert AllocationStrategy.EQUAL_WEIGHT.value == "equal_weight"
        assert AllocationStrategy.RISK_PARITY.value == "risk_parity"
        assert AllocationStrategy.MOMENTUM_BASED.value == "momentum_based"
        assert AllocationStrategy.VOLATILITY_INVERSE.value == "volatility_inverse"
        assert AllocationStrategy.CUSTOM.value == "custom"


class TestCircuitBreakerStatus:
    """Test CircuitBreakerStatus enum values."""

    def test_circuit_breaker_status_enum(self):
        """Test CircuitBreakerStatus enum values."""
        assert CircuitBreakerStatus.ACTIVE.value == "active"
        assert CircuitBreakerStatus.TRIGGERED.value == "triggered"
        assert CircuitBreakerStatus.COOLDOWN.value == "cooldown"
        assert CircuitBreakerStatus.DISABLED.value == "disabled"


class TestCircuitBreakerType:
    """Test CircuitBreakerType enum values."""

    def test_circuit_breaker_type_enum(self):
        """Test CircuitBreakerType enum values."""
        # Test some of the enum values (adjust based on actual implementation)
        assert hasattr(CircuitBreakerType, "DAILY_LOSS_LIMIT")
        assert hasattr(CircuitBreakerType, "DRAWDOWN_LIMIT")
        assert hasattr(CircuitBreakerType, "VOLATILITY_SPIKE")


class TestOrderStatus:
    """Test OrderStatus enum values."""

    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"


class TestOrderSide:
    """Test OrderSide enum values."""

    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestTimeInForce:
    """Test TimeInForce enum values."""

    def test_time_in_force_enum(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"
        assert TimeInForce.GTX.value == "GTX"
        assert TimeInForce.DAY.value == "DAY"


class TestRiskLevel:
    """Test RiskLevel enum values."""

    def test_risk_level_enum(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestPositionSizeMethod:
    """Test PositionSizeMethod enum values."""

    def test_position_size_method_enum(self):
        """Test PositionSizeMethod enum values."""
        assert PositionSizeMethod.FIXED.value == "fixed"
        assert PositionSizeMethod.FIXED_PERCENTAGE.value == "fixed_percentage"
        assert PositionSizeMethod.KELLY_CRITERION.value == "kelly_criterion"
        assert PositionSizeMethod.VOLATILITY_ADJUSTED.value == "volatility_adjusted"
        assert PositionSizeMethod.RISK_PARITY.value == "risk_parity"
        assert PositionSizeMethod.CONFIDENCE_WEIGHTED.value == "confidence_weighted"


class TestExchangeStatus:
    """Test ExchangeStatus enum values."""

    def test_exchange_status_enum(self):
        """Test ExchangeStatus enum values."""
        assert ExchangeStatus.ONLINE.value == "online"
        assert ExchangeStatus.MAINTENANCE.value == "maintenance"
        assert ExchangeStatus.DEGRADED.value == "degraded"
        assert ExchangeStatus.OFFLINE.value == "offline"


class TestExchangeInfo:
    """Test ExchangeInfo model creation."""

    def test_exchange_info_creation(self):
        """Test ExchangeInfo model creation."""
        info = ExchangeInfo(
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

        assert info.symbol == "BTCUSDT"
        assert info.base_asset == "BTC"
        assert info.quote_asset == "USDT"
        assert info.status == "TRADING"
        assert info.min_price == Decimal("0.01")
        assert info.exchange == "binance"


class TestOrderBook:
    """Test OrderBook model creation."""

    def test_order_book_creation(self):
        """Test OrderBook model creation."""
        from src.core.types.market import OrderBookLevel

        order_book = OrderBook(
            symbol="BTC/USDT",
            bids=[
                OrderBookLevel(price=Decimal("50000"), quantity=Decimal("1.5")),
                OrderBookLevel(price=Decimal("49999"), quantity=Decimal("2.0")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("50001"), quantity=Decimal("1.2")),
                OrderBookLevel(price=Decimal("50002"), quantity=Decimal("1.8")),
            ],
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        assert order_book.symbol == "BTC/USDT"
        assert order_book.exchange == "binance"
        assert len(order_book.bids) == 2
        assert len(order_book.asks) == 2
        assert order_book.bids[0].price == Decimal("50000")
        assert order_book.asks[0].price == Decimal("50001")


class TestRiskMetrics:
    """Test RiskMetrics model creation."""

    def test_risk_metrics_creation(self):
        """Test RiskMetrics model creation."""
        metrics = RiskMetrics(
            portfolio_value=Decimal("100000"),
            total_exposure=Decimal("80000"),
            var_1d=Decimal("0.05"),
            risk_level=RiskLevel.MEDIUM,
            timestamp=datetime.now(timezone.utc),
        )

        assert metrics.portfolio_value == Decimal("100000")
        assert metrics.total_exposure == Decimal("80000")
        assert metrics.var_1d == Decimal("0.05")
        assert metrics.risk_level == RiskLevel.MEDIUM


class TestPositionLimits:
    """Test PositionLimits model creation."""

    def test_position_limits_creation(self):
        """Test PositionLimits model creation."""
        limits = PositionLimits(
            max_position_size=Decimal("10000"),
            max_positions=5,
            max_leverage=Decimal("3.0"),
            min_position_size=Decimal("100"),
        )

        assert limits.max_position_size == Decimal("10000")
        assert limits.max_positions == 5
        assert limits.max_leverage == Decimal("3.0")
        assert limits.min_position_size == Decimal("100")


class TestRegimeChangeEvent:
    """Test RegimeChangeEvent model creation."""

    def test_regime_change_event_creation(self):
        """Test RegimeChangeEvent model creation."""
        event = RegimeChangeEvent(
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            previous_regime=MarketRegime.TRENDING_UP,
            new_regime=MarketRegime.TRENDING_DOWN,
            confidence=0.85,
            indicators={"rsi": 30, "macd": -0.5},
        )

        assert event.previous_regime == MarketRegime.TRENDING_UP
        assert event.new_regime == MarketRegime.TRENDING_DOWN
        assert event.confidence == 0.85
        assert event.indicators["rsi"] == 30


class TestCircuitBreakerEvent:
    """Test CircuitBreakerEvent model creation."""

    def test_circuit_breaker_event_creation(self):
        """Test CircuitBreakerEvent model creation."""
        event = CircuitBreakerEvent(
            breaker_id="cb_001",
            breaker_type=CircuitBreakerType.DAILY_LOSS_LIMIT,
            status=CircuitBreakerStatus.TRIGGERED,
            triggered_at=datetime.now(timezone.utc),
            trigger_value=5.2,
            threshold_value=5.0,
            cooldown_period=3600,
            reason="Daily loss exceeded 5%",
            metadata={"loss_percentage": 0.052},
        )

        assert event.breaker_type == CircuitBreakerType.DAILY_LOSS_LIMIT
        assert event.status == CircuitBreakerStatus.TRIGGERED
        assert event.reason == "Daily loss exceeded 5%"
        assert event.metadata["loss_percentage"] == 0.052


class TestCapitalAllocation:
    """Test CapitalAllocation model creation."""

    def test_capital_allocation_creation(self):
        """Test CapitalAllocation model creation."""
        allocation = CapitalAllocation(
            allocation_id="alloc_001",
            strategy_id="test_strategy",
            allocated_amount=Decimal("10000.00"),
            utilized_amount=Decimal("5000.00"),
            available_amount=Decimal("5000.00"),
            allocation_percentage=Decimal("0.1"),
            target_allocation_pct=Decimal("0.15"),
            min_allocation=Decimal("1000.00"),
            max_allocation=Decimal("20000.00"),
            last_rebalance=datetime.now(timezone.utc),
        )

        assert allocation.allocation_id == "alloc_001"
        assert allocation.strategy_id == "test_strategy"
        assert allocation.allocated_amount == Decimal("10000.00")
        assert allocation.utilized_amount == Decimal("5000.00")
        assert allocation.available_amount == Decimal("5000.00")
        assert allocation.allocation_percentage == Decimal("0.1")


class TestFundFlow:
    """Test FundFlow model creation."""

    def test_fund_flow_creation(self):
        """Test FundFlow model creation."""
        fund_flow = FundFlow(
            flow_id="flow_001",
            flow_type="deposit",
            amount=Decimal("1000.00"),
            currency="USDT",
            exchange="binance",
            status="completed",
            requested_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            reference="ref_123",
        )

        assert fund_flow.flow_id == "flow_001"
        assert fund_flow.flow_type == "deposit"
        assert fund_flow.amount == Decimal("1000.00")
        assert fund_flow.currency == "USDT"
        assert fund_flow.exchange == "binance"
        assert fund_flow.status == "completed"
        assert fund_flow.reference == "ref_123"


class TestCapitalMetrics:
    """Test CapitalMetrics model creation."""

    def test_capital_metrics_creation(self):
        """Test CapitalMetrics model creation."""
        metrics = CapitalMetrics(
            total_capital=Decimal("100000.00"),
            allocated_amount=Decimal("80000.00"),
            available_amount=Decimal("20000.00"),
            total_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("3000.00"),
            unrealized_pnl=Decimal("2000.00"),
            daily_return=Decimal("0.05"),
            weekly_return=Decimal("0.12"),
            monthly_return=Decimal("0.25"),
            yearly_return=Decimal("0.45"),
            total_return=Decimal("0.50"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("1.8"),
            calmar_ratio=Decimal("2.0"),
            current_drawdown=Decimal("0.03"),
            max_drawdown=Decimal("0.15"),
            var_95=Decimal("1000.00"),
            expected_shortfall=Decimal("1500.00"),
            strategies_active=3,
            positions_open=10,
            leverage_used=Decimal("1.5"),
            timestamp=datetime.now(timezone.utc),
        )

        assert metrics.total_capital == Decimal("100000.00")
        assert metrics.allocated_amount == Decimal("80000.00")
        assert metrics.available_amount == Decimal("20000.00")
        assert metrics.total_pnl == Decimal("5000.00")
        assert metrics.sharpe_ratio == Decimal("1.5")


class TestCurrencyExposure:
    """Test CurrencyExposure model creation."""

    def test_currency_exposure_creation(self):
        """Test CurrencyExposure model creation."""
        exposure = CurrencyExposure(
            currency="BTC",
            exposure_amount=Decimal("2.5"),
            exposure_pct=0.25,
            hedge_amount=Decimal("0.5"),
            net_exposure=Decimal("2.0"),
            exchange_rate=Decimal("50000.00"),
            base_currency="USD",
            updated_at=datetime.now(timezone.utc),
        )

        assert exposure.currency == "BTC"
        assert exposure.exposure_amount == Decimal("2.5")
        assert exposure.exposure_pct == 0.25
        assert exposure.hedge_amount == Decimal("0.5")
        assert exposure.net_exposure == Decimal("2.0")
        assert exposure.exchange_rate == Decimal("50000.00")


class TestExchangeAllocation:
    """Test ExchangeAllocation model creation."""

    def test_exchange_allocation_creation(self):
        """Test ExchangeAllocation model creation."""
        allocation = ExchangeAllocation(
            exchange="binance",
            allocated_amount=Decimal("50000.00"),
            utilized_amount=Decimal("20000.00"),
            available_amount=Decimal("30000.00"),
            allocation_percentage=Decimal("0.5"),
            num_positions=10,
            total_pnl=Decimal("2000.00"),
            last_activity=datetime.now(timezone.utc),
        )

        assert allocation.exchange == "binance"
        assert allocation.allocated_amount == Decimal("50000.00")
        assert allocation.utilized_amount == Decimal("20000.00")
        assert allocation.available_amount == Decimal("30000.00")
        assert allocation.allocation_percentage == Decimal("0.5")
        assert allocation.num_positions == 10


class TestWithdrawalRule:
    """Test WithdrawalRule model creation."""

    def test_withdrawal_rule_creation(self):
        """Test WithdrawalRule model creation."""
        rule = WithdrawalRule(
            rule_id="rule_001",
            name="profit_only",
            enabled=True,
            trigger_type="profit_threshold",
            trigger_value={"threshold": 0.05},
            withdrawal_pct=0.2,
            min_withdrawal=Decimal("1000.00"),
            max_withdrawal=Decimal("10000.00"),
            destination="cold_wallet",
        )

        assert rule.rule_id == "rule_001"
        assert rule.name == "profit_only"
        assert rule.enabled is True
        assert rule.trigger_type == "profit_threshold"
        assert rule.withdrawal_pct == Decimal("0.2")
        assert rule.min_withdrawal == Decimal("1000.00")
        assert rule.max_withdrawal == Decimal("10000.00")


class TestCapitalProtection:
    """Test CapitalProtection model creation."""

    def test_capital_protection_creation(self):
        """Test CapitalProtection model creation."""
        protection = CapitalProtection(
            protection_id="prot_001",
            enabled=True,
            min_capital_threshold=Decimal("10000.00"),
            stop_trading_threshold=Decimal("5000.00"),
            reduce_size_threshold=Decimal("7500.00"),
            size_reduction_factor=0.5,
            max_daily_loss=Decimal("500.00"),
            max_weekly_loss=Decimal("2000.00"),
            max_monthly_loss=Decimal("5000.00"),
            emergency_liquidation=True,
            emergency_threshold=Decimal("4000.00"),
        )

        assert protection.protection_id == "prot_001"
        assert protection.enabled is True
        assert protection.min_capital_threshold == Decimal("10000.00")
        assert protection.stop_trading_threshold == Decimal("5000.00")
        assert protection.reduce_size_threshold == Decimal("7500.00")
        assert protection.size_reduction_factor == 0.5


class TestStrategyType:
    """Test StrategyType enum values."""

    def test_strategy_type_enum(self):
        """Test StrategyType enum values."""
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.ARBITRAGE.value == "arbitrage"
        assert StrategyType.MARKET_MAKING.value == "market_making"
        assert StrategyType.TREND_FOLLOWING.value == "trend_following"
        assert StrategyType.PAIRS_TRADING.value == "pairs_trading"
        assert StrategyType.STATISTICAL_ARBITRAGE.value == "statistical_arbitrage"
        assert StrategyType.CUSTOM.value == "custom"


class TestStrategyStatus:
    """Test StrategyStatus enum values."""

    def test_strategy_status_enum(self):
        """Test StrategyStatus enum values."""
        assert StrategyStatus.INACTIVE.value == "inactive"
        assert StrategyStatus.STARTING.value == "starting"
        assert StrategyStatus.ACTIVE.value == "active"
        assert StrategyStatus.PAUSED.value == "paused"
        assert StrategyStatus.STOPPING.value == "stopping"
        assert StrategyStatus.STOPPED.value == "stopped"
        assert StrategyStatus.ERROR.value == "error"


class TestStrategyConfig:
    """Test StrategyConfig model creation."""

    def test_strategy_config_creation(self):
        """Test StrategyConfig model creation."""
        config = StrategyConfig(
            strategy_id="strat_001",
            strategy_type=StrategyType.MEAN_REVERSION,
            name="test_strategy",
            symbol="BTC/USDT",
            timeframe="1h",
            enabled=True,
            parameters={"param1": "value1"},
            risk_parameters={"stop_loss": 0.02},
            metadata={},
        )

        assert config.strategy_id == "strat_001"
        assert config.name == "test_strategy"
        assert config.strategy_type == StrategyType.MEAN_REVERSION
        assert config.enabled is True
        assert config.symbol == "BTC/USDT"
        assert config.timeframe == "1h"
        assert config.parameters["param1"] == "value1"


class TestStrategyMetrics:
    """Test StrategyMetrics model creation."""

    def test_strategy_metrics_creation(self):
        """Test StrategyMetrics model creation."""
        metrics = StrategyMetrics(
            strategy_id="strat_001",
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            total_pnl=Decimal("5000.00"),
            realized_pnl=Decimal("4000.00"),
            unrealized_pnl=Decimal("1000.00"),
            win_rate=0.6,
            avg_win=Decimal("100.00"),
            avg_loss=Decimal("50.00"),
            profit_factor=2.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=0.15,
            calmar_ratio=1.8,
            updated_at=datetime.now(timezone.utc),
        )

        assert metrics.strategy_id == "strat_001"
        assert metrics.total_trades == 100
        assert metrics.winning_trades == 60
        assert metrics.losing_trades == 40
        assert metrics.total_pnl == Decimal("5000.00")
        assert metrics.win_rate == 0.6
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == 0.15


class TestErrorPattern:
    """Test ErrorPattern constants."""

    def test_error_pattern_constants(self):
        """Test ErrorPattern constant values."""
        assert ErrorPattern.MISSING_REQUIRED_FIELD == "missing_required_field"
        assert ErrorPattern.INVALID_DATA_TYPE == "invalid_data_type"
        assert ErrorPattern.OUT_OF_RANGE == "out_of_range"
        assert ErrorPattern.DUPLICATE_RECORD == "duplicate_record"
        assert ErrorPattern.SCHEMA_MISMATCH == "schema_mismatch"
        assert ErrorPattern.TIMESTAMP_ERROR == "timestamp_error"
        assert ErrorPattern.ENCODING_ERROR == "encoding_error"
        assert ErrorPattern.PARSING_ERROR == "parsing_error"
