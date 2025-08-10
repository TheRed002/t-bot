"""
Extended unit tests for core type definitions.

These tests cover additional types and enums not covered in the basic types test.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

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
    Ticker,
    Trade,
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


class TestConnectionType:
    """Test ConnectionType enum values."""

    def test_connection_type_enum(self):
        """Test ConnectionType enum values."""
        assert ConnectionType.TICKER.value == "ticker"
        assert ConnectionType.ORDERBOOK.value == "orderbook"
        assert ConnectionType.TRADES.value == "trades"


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


class TestQualityLevel:
    """Test QualityLevel enum values."""

    def test_quality_level_enum(self):
        """Test QualityLevel enum values."""
        assert QualityLevel.EXCELLENT.value == "excellent"
        assert QualityLevel.GOOD.value == "good"
        assert QualityLevel.FAIR.value == "fair"
        assert QualityLevel.POOR.value == "poor"
        assert QualityLevel.CRITICAL.value == "critical"


class TestDriftType:
    """Test DriftType enum values."""

    def test_drift_type_enum(self):
        """Test DriftType enum values."""
        assert DriftType.CONCEPT_DRIFT.value == "concept_drift"
        assert DriftType.COVARIATE_DRIFT.value == "covariate_drift"
        assert DriftType.LABEL_DRIFT.value == "label_drift"
        assert DriftType.DISTRIBUTION_DRIFT.value == "distribution_drift"


class TestIngestionMode:
    """Test IngestionMode enum values."""

    def test_ingestion_mode_enum(self):
        """Test IngestionMode enum values."""
        assert IngestionMode.REAL_TIME.value == "real_time"
        assert IngestionMode.BATCH.value == "batch"
        assert IngestionMode.HYBRID.value == "hybrid"


class TestPipelineStatus:
    """Test PipelineStatus enum values."""

    def test_pipeline_status_enum(self):
        """Test PipelineStatus enum values."""
        assert PipelineStatus.STOPPED.value == "stopped"
        assert PipelineStatus.STARTING.value == "starting"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.PAUSED.value == "paused"
        assert PipelineStatus.ERROR.value == "error"
        assert PipelineStatus.STOPPING.value == "stopping"


class TestProcessingStep:
    """Test ProcessingStep enum values."""

    def test_processing_step_enum(self):
        """Test ProcessingStep enum values."""
        assert ProcessingStep.NORMALIZE.value == "normalize"
        assert ProcessingStep.ENRICH.value == "enrich"
        assert ProcessingStep.AGGREGATE.value == "aggregate"
        assert ProcessingStep.TRANSFORM.value == "transform"
        assert ProcessingStep.VALIDATE.value == "validate"
        assert ProcessingStep.FILTER.value == "filter"


class TestStorageMode:
    """Test StorageMode enum values."""

    def test_storage_mode_enum(self):
        """Test StorageMode enum values."""
        assert StorageMode.REAL_TIME.value == "real_time"
        assert StorageMode.BATCH.value == "batch"
        assert StorageMode.BUFFER.value == "buffer"


class TestNewsSentiment:
    """Test NewsSentiment enum values."""

    def test_news_sentiment_enum(self):
        """Test NewsSentiment enum values."""
        assert NewsSentiment.VERY_POSITIVE.value == "very_positive"
        assert NewsSentiment.POSITIVE.value == "positive"
        assert NewsSentiment.NEUTRAL.value == "neutral"
        assert NewsSentiment.NEGATIVE.value == "negative"
        assert NewsSentiment.VERY_NEGATIVE.value == "very_negative"


class TestSocialSentiment:
    """Test SocialSentiment enum values."""

    def test_social_sentiment_enum(self):
        """Test SocialSentiment enum values."""
        assert SocialSentiment.VERY_BULLISH.value == "very_bullish"
        assert SocialSentiment.BULLISH.value == "bullish"
        assert SocialSentiment.NEUTRAL.value == "neutral"
        assert SocialSentiment.BEARISH.value == "bearish"
        assert SocialSentiment.VERY_BEARISH.value == "very_bearish"


class TestExchangeInfo:
    """Test ExchangeInfo model creation."""

    def test_exchange_info_creation(self):
        """Test ExchangeInfo model creation."""
        exchange_info = ExchangeInfo(
            name="test_exchange",
            supported_symbols=["BTC/USDT", "ETH/USDT"],
            rate_limits={"requests_per_minute": 1000},
            features=["spot_trading", "futures_trading"],
            api_version="v1",
        )

        assert exchange_info.name == "test_exchange"
        assert exchange_info.supported_symbols == ["BTC/USDT", "ETH/USDT"]
        assert exchange_info.rate_limits["requests_per_minute"] == 1000
        assert exchange_info.features == ["spot_trading", "futures_trading"]
        assert exchange_info.api_version == "v1"


class TestTicker:
    """Test Ticker model creation."""

    def test_ticker_creation(self):
        """Test Ticker model creation."""
        ticker = Ticker(
            symbol="BTC/USDT",
            bid=Decimal("50000.00"),
            ask=Decimal("50001.00"),
            last_price=Decimal("50000.50"),
            volume_24h=Decimal("1000.5"),
            price_change_24h=Decimal("500.00"),
            timestamp=datetime.now(timezone.utc),
        )

        assert ticker.symbol == "BTC/USDT"
        assert ticker.bid == Decimal("50000.00")
        assert ticker.ask == Decimal("50001.00")
        assert ticker.last_price == Decimal("50000.50")
        assert ticker.volume_24h == Decimal("1000.5")
        assert ticker.price_change_24h == Decimal("500.00")


class TestOrderBook:
    """Test OrderBook model creation."""

    def test_order_book_creation(self):
        """Test OrderBook model creation."""
        order_book = OrderBook(
            symbol="BTC/USDT",
            bids=[[Decimal("49999.00"), Decimal("1.0")], [Decimal("49998.00"), Decimal("2.0")]],
            asks=[[Decimal("50001.00"), Decimal("1.0")], [Decimal("50002.00"), Decimal("2.0")]],
            timestamp=datetime.now(timezone.utc),
        )

        assert order_book.symbol == "BTC/USDT"
        assert len(order_book.bids) == 2
        assert len(order_book.asks) == 2
        assert order_book.bids[0][0] == Decimal("49999.00")
        assert order_book.asks[0][0] == Decimal("50001.00")


class TestExchangeStatus:
    """Test ExchangeStatus enum values."""

    def test_exchange_status_enum(self):
        """Test ExchangeStatus enum values."""
        assert ExchangeStatus.ONLINE.value == "online"
        assert ExchangeStatus.OFFLINE.value == "offline"
        assert ExchangeStatus.MAINTENANCE.value == "maintenance"


class TestOrderStatus:
    """Test OrderStatus enum values."""

    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"
        assert OrderStatus.UNKNOWN.value == "unknown"


class TestTrade:
    """Test Trade model creation."""

    def test_trade_creation(self):
        """Test Trade model creation."""
        trade = Trade(
            id="trade_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
            fee=Decimal("25.00"),
            fee_currency="USDT",
        )

        assert trade.id == "trade_123"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == Decimal("1.0")
        assert trade.price == Decimal("50000.00")
        assert trade.fee == Decimal("25.00")
        assert trade.fee_currency == "USDT"


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
        assert PositionSizeMethod.FIXED_PCT.value == "fixed_percentage"
        assert PositionSizeMethod.KELLY_CRITERION.value == "kelly_criterion"
        assert PositionSizeMethod.VOLATILITY_ADJUSTED.value == "volatility_adjusted"
        assert PositionSizeMethod.CONFIDENCE_WEIGHTED.value == "confidence_weighted"


class TestRiskMetrics:
    """Test RiskMetrics model creation."""

    def test_risk_metrics_creation(self):
        """Test RiskMetrics model creation."""
        risk_metrics = RiskMetrics(
            var_1d=Decimal("1000.00"),
            var_5d=Decimal("2500.00"),
            expected_shortfall=Decimal("1500.00"),
            max_drawdown=Decimal("5000.00"),
            sharpe_ratio=Decimal("1.5"),
            current_drawdown=Decimal("1000.00"),
            risk_level=RiskLevel.MEDIUM,
            timestamp=datetime.now(timezone.utc),
        )

        assert risk_metrics.var_1d == Decimal("1000.00")
        assert risk_metrics.var_5d == Decimal("2500.00")
        assert risk_metrics.expected_shortfall == Decimal("1500.00")
        assert risk_metrics.max_drawdown == Decimal("5000.00")
        assert risk_metrics.sharpe_ratio == Decimal("1.5")
        assert risk_metrics.current_drawdown == Decimal("1000.00")
        assert risk_metrics.risk_level == RiskLevel.MEDIUM


class TestPositionLimits:
    """Test PositionLimits model creation."""

    def test_position_limits_creation(self):
        """Test PositionLimits model creation."""
        position_limits = PositionLimits(
            max_position_size=Decimal("10000.00"),
            max_positions_per_symbol=2,
            max_total_positions=20,
            max_portfolio_exposure=Decimal("0.95"),
            max_sector_exposure=Decimal("0.25"),
            max_correlation_exposure=Decimal("0.5"),
            max_leverage=Decimal("2.0"),
        )

        assert position_limits.max_position_size == Decimal("10000.00")
        assert position_limits.max_positions_per_symbol == 2
        assert position_limits.max_total_positions == 20
        assert position_limits.max_portfolio_exposure == Decimal("0.95")
        assert position_limits.max_sector_exposure == Decimal("0.25")
        assert position_limits.max_correlation_exposure == Decimal("0.5")
        assert position_limits.max_leverage == Decimal("2.0")


class TestCircuitBreakerStatus:
    """Test CircuitBreakerStatus enum values."""

    def test_circuit_breaker_status_enum(self):
        """Test CircuitBreakerStatus enum values."""
        assert CircuitBreakerStatus.CLOSED.value == "closed"
        assert CircuitBreakerStatus.OPEN.value == "open"
        assert CircuitBreakerStatus.HALF_OPEN.value == "half_open"


class TestCircuitBreakerType:
    """Test CircuitBreakerType enum values."""

    def test_circuit_breaker_type_enum(self):
        """Test CircuitBreakerType enum values."""
        assert CircuitBreakerType.DAILY_LOSS_LIMIT.value == "daily_loss_limit"
        assert CircuitBreakerType.DRAWDOWN_LIMIT.value == "drawdown_limit"
        assert CircuitBreakerType.VOLATILITY_SPIKE.value == "volatility_spike"
        assert CircuitBreakerType.MODEL_CONFIDENCE.value == "model_confidence"
        assert CircuitBreakerType.SYSTEM_ERROR_RATE.value == "system_error_rate"
        assert CircuitBreakerType.MANUAL_TRIGGER.value == "manual_trigger"


class TestCircuitBreakerEvent:
    """Test CircuitBreakerEvent model creation."""

    def test_circuit_breaker_event_creation(self):
        """Test CircuitBreakerEvent model creation."""
        event = CircuitBreakerEvent(
            trigger_type=CircuitBreakerType.DAILY_LOSS_LIMIT,
            threshold=Decimal("0.05"),
            actual_value=Decimal("0.06"),
            timestamp=datetime.now(timezone.utc),
            description="Daily loss limit exceeded",
            metadata={"strategy": "test_strategy"},
        )

        assert event.trigger_type == CircuitBreakerType.DAILY_LOSS_LIMIT
        assert event.threshold == Decimal("0.05")
        assert event.actual_value == Decimal("0.06")
        assert event.description == "Daily loss limit exceeded"
        assert event.metadata["strategy"] == "test_strategy"


class TestMarketRegime:
    """Test MarketRegime enum values."""

    def test_market_regime_enum(self):
        """Test MarketRegime enum values."""
        assert MarketRegime.LOW_VOLATILITY.value == "low_volatility"
        assert MarketRegime.MEDIUM_VOLATILITY.value == "medium_volatility"
        assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"
        assert MarketRegime.TRENDING_UP.value == "trending_up"
        assert MarketRegime.TRENDING_DOWN.value == "trending_down"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.HIGH_CORRELATION.value == "high_correlation"
        assert MarketRegime.LOW_CORRELATION.value == "low_correlation"
        assert MarketRegime.CRISIS.value == "crisis"


class TestRegimeChangeEvent:
    """Test RegimeChangeEvent model creation."""

    def test_regime_change_event_creation(self):
        """Test RegimeChangeEvent model creation."""
        event = RegimeChangeEvent(
            from_regime=MarketRegime.LOW_VOLATILITY,
            to_regime=MarketRegime.HIGH_VOLATILITY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            trigger_metrics={"volatility": 0.05},
            description="Volatility spike detected",
        )

        assert event.from_regime == MarketRegime.LOW_VOLATILITY
        assert event.to_regime == MarketRegime.HIGH_VOLATILITY
        assert event.confidence == 0.8
        assert event.trigger_metrics["volatility"] == 0.05
        assert event.description == "Volatility spike detected"


class TestAllocationStrategy:
    """Test AllocationStrategy enum values."""

    def test_allocation_strategy_enum(self):
        """Test AllocationStrategy enum values."""
        assert AllocationStrategy.EQUAL_WEIGHT.value == "equal_weight"
        assert AllocationStrategy.PERFORMANCE_WEIGHTED.value == "performance_weighted"
        assert AllocationStrategy.VOLATILITY_WEIGHTED.value == "volatility_weighted"
        assert AllocationStrategy.RISK_PARITY.value == "risk_parity"
        assert AllocationStrategy.DYNAMIC.value == "dynamic"


class TestCapitalAllocation:
    """Test CapitalAllocation model creation."""

    def test_capital_allocation_creation(self):
        """Test CapitalAllocation model creation."""
        allocation = CapitalAllocation(
            strategy_id="test_strategy",
            exchange="binance",
            allocated_amount=Decimal("10000.00"),
            utilized_amount=Decimal("5000.00"),
            available_amount=Decimal("5000.00"),
            allocation_percentage=0.1,
            last_rebalance=datetime.now(timezone.utc),
        )

        assert allocation.strategy_id == "test_strategy"
        assert allocation.exchange == "binance"
        assert allocation.allocated_amount == Decimal("10000.00")
        assert allocation.utilized_amount == Decimal("5000.00")
        assert allocation.available_amount == Decimal("5000.00")
        assert allocation.allocation_percentage == 0.1


class TestFundFlow:
    """Test FundFlow model creation."""

    def test_fund_flow_creation(self):
        """Test FundFlow model creation."""
        fund_flow = FundFlow(
            from_strategy="strategy_a",
            to_strategy="strategy_b",
            from_exchange="binance",
            to_exchange="okx",
            amount=Decimal("1000.00"),
            currency="USDT",
            reason="rebalancing",
            timestamp=datetime.now(timezone.utc),
            converted_amount=Decimal("1000.00"),
            exchange_rate=Decimal("1.0"),
        )

        assert fund_flow.from_strategy == "strategy_a"
        assert fund_flow.to_strategy == "strategy_b"
        assert fund_flow.from_exchange == "binance"
        assert fund_flow.to_exchange == "okx"
        assert fund_flow.amount == Decimal("1000.00")
        assert fund_flow.currency == "USDT"
        assert fund_flow.reason == "rebalancing"
        assert fund_flow.converted_amount == Decimal("1000.00")
        assert fund_flow.exchange_rate == Decimal("1.0")


class TestCapitalMetrics:
    """Test CapitalMetrics model creation."""

    def test_capital_metrics_creation(self):
        """Test CapitalMetrics model creation."""
        metrics = CapitalMetrics(
            total_capital=Decimal("100000.00"),
            allocated_capital=Decimal("80000.00"),
            available_capital=Decimal("20000.00"),
            utilization_rate=0.8,
            allocation_efficiency=1.2,
            rebalance_frequency_hours=24,
            emergency_reserve=Decimal("10000.00"),
            last_updated=datetime.now(timezone.utc),
            allocation_count=5,
        )

        assert metrics.total_capital == Decimal("100000.00")
        assert metrics.allocated_capital == Decimal("80000.00")
        assert metrics.available_capital == Decimal("20000.00")
        assert metrics.utilization_rate == 0.8
        assert metrics.allocation_efficiency == 1.2
        assert metrics.rebalance_frequency_hours == 24
        assert metrics.emergency_reserve == Decimal("10000.00")
        assert metrics.allocation_count == 5


class TestCurrencyExposure:
    """Test CurrencyExposure model creation."""

    def test_currency_exposure_creation(self):
        """Test CurrencyExposure model creation."""
        exposure = CurrencyExposure(
            currency="BTC",
            total_exposure=Decimal("2.5"),
            base_currency_equivalent=Decimal("125000.00"),
            exposure_percentage=0.25,
            hedging_required=True,
            hedge_amount=Decimal("0.5"),
            timestamp=datetime.now(timezone.utc),
        )

        assert exposure.currency == "BTC"
        assert exposure.total_exposure == Decimal("2.5")
        assert exposure.base_currency_equivalent == Decimal("125000.00")
        assert exposure.exposure_percentage == 0.25
        assert exposure.hedging_required is True
        assert exposure.hedge_amount == Decimal("0.5")


class TestExchangeAllocation:
    """Test ExchangeAllocation model creation."""

    def test_exchange_allocation_creation(self):
        """Test ExchangeAllocation model creation."""
        allocation = ExchangeAllocation(
            exchange="binance",
            allocated_amount=Decimal("50000.00"),
            available_amount=Decimal("30000.00"),
            utilization_rate=0.6,
            liquidity_score=0.9,
            fee_efficiency=0.8,
            reliability_score=0.95,
            last_rebalance=datetime.now(timezone.utc),
        )

        assert allocation.exchange == "binance"
        assert allocation.allocated_amount == Decimal("50000.00")
        assert allocation.available_amount == Decimal("30000.00")
        assert allocation.utilization_rate == 0.6
        assert allocation.liquidity_score == 0.9
        assert allocation.fee_efficiency == 0.8
        assert allocation.reliability_score == 0.95


class TestWithdrawalRule:
    """Test WithdrawalRule model creation."""

    def test_withdrawal_rule_creation(self):
        """Test WithdrawalRule model creation."""
        rule = WithdrawalRule(
            name="profit_only",
            description="Only withdraw realized profits",
            enabled=True,
            threshold=0.05,
            min_amount=Decimal("1000.00"),
            max_percentage=0.2,
            cooldown_hours=24,
        )

        assert rule.name == "profit_only"
        assert rule.description == "Only withdraw realized profits"
        assert rule.enabled is True
        assert rule.threshold == 0.05
        assert rule.min_amount == Decimal("1000.00")
        assert rule.max_percentage == 0.2
        assert rule.cooldown_hours == 24


class TestCapitalProtection:
    """Test CapitalProtection model creation."""

    def test_capital_protection_creation(self):
        """Test CapitalProtection model creation."""
        protection = CapitalProtection(
            emergency_reserve_pct=0.1,
            max_daily_loss_pct=0.05,
            max_weekly_loss_pct=0.10,
            max_monthly_loss_pct=0.15,
            profit_lock_pct=0.5,
            auto_compound_enabled=True,
            auto_compound_frequency="weekly",
            profit_threshold=Decimal("100.00"),
        )

        assert protection.emergency_reserve_pct == 0.1
        assert protection.max_daily_loss_pct == 0.05
        assert protection.max_weekly_loss_pct == 0.10
        assert protection.max_monthly_loss_pct == 0.15
        assert protection.profit_lock_pct == 0.5
        assert protection.auto_compound_enabled is True
        assert protection.auto_compound_frequency == "weekly"
        assert protection.profit_threshold == Decimal("100.00")


class TestStrategyType:
    """Test StrategyType enum values."""

    def test_strategy_type_enum(self):
        """Test StrategyType enum values."""
        assert StrategyType.STATIC.value == "static"
        assert StrategyType.DYNAMIC.value == "dynamic"
        assert StrategyType.ARBITRAGE.value == "arbitrage"
        assert StrategyType.MARKET_MAKING.value == "market_making"
        assert StrategyType.EVOLUTIONARY.value == "evolutionary"
        assert StrategyType.HYBRID.value == "hybrid"
        assert StrategyType.AI_ML.value == "ai_ml"


class TestStrategyStatus:
    """Test StrategyStatus enum values."""

    def test_strategy_status_enum(self):
        """Test StrategyStatus enum values."""
        assert StrategyStatus.STOPPED.value == "stopped"
        assert StrategyStatus.STARTING.value == "starting"
        assert StrategyStatus.RUNNING.value == "running"
        assert StrategyStatus.PAUSED.value == "paused"
        assert StrategyStatus.ERROR.value == "error"


class TestStrategyConfig:
    """Test StrategyConfig model creation."""

    def test_strategy_config_creation(self):
        """Test StrategyConfig model creation."""
        config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.STATIC,
            enabled=True,
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframe="1h",
            min_confidence=0.7,
            max_positions=3,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            parameters={"param1": "value1"},
        )

        assert config.name == "test_strategy"
        assert config.strategy_type == StrategyType.STATIC
        assert config.enabled is True
        assert config.symbols == ["BTC/USDT", "ETH/USDT"]
        assert config.timeframe == "1h"
        assert config.min_confidence == 0.7
        assert config.max_positions == 3
        assert config.position_size_pct == 0.02
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.04
        assert config.parameters["param1"] == "value1"


class TestStrategyMetrics:
    """Test StrategyMetrics model creation."""

    def test_strategy_metrics_creation(self):
        """Test StrategyMetrics model creation."""
        metrics = StrategyMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            total_pnl=Decimal("5000.00"),
            win_rate=0.6,
            sharpe_ratio=1.2,
            max_drawdown=Decimal("2000.00"),
            last_updated=datetime.now(timezone.utc),
        )

        assert metrics.total_trades == 100
        assert metrics.winning_trades == 60
        assert metrics.losing_trades == 40
        assert metrics.total_pnl == Decimal("5000.00")
        assert metrics.win_rate == 0.6
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == Decimal("2000.00")


class TestErrorPattern:
    """Test ErrorPattern model creation."""

    def test_error_pattern_creation(self):
        """Test ErrorPattern model creation."""
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            component="test_component",
            error_type="test_error",
            frequency=5.0,
            severity="high",
            first_detected=datetime.now(timezone.utc),
            last_detected=datetime.now(timezone.utc),
            occurrence_count=10,
            confidence=0.8,
            description="Test error pattern",
            suggested_action="Restart component",
            is_active=True,
        )

        assert pattern.pattern_id == "test_pattern"
        assert pattern.pattern_type == "frequency"
        assert pattern.component == "test_component"
        assert pattern.error_type == "test_error"
        assert pattern.frequency == 5.0
        assert pattern.severity == "high"
        assert pattern.occurrence_count == 10
        assert pattern.confidence == 0.8
        assert pattern.description == "Test error pattern"
        assert pattern.suggested_action == "Restart component"
        assert pattern.is_active is True

    def test_error_pattern_to_dict(self):
        """Test ErrorPattern to_dict method."""
        now = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            component="test_component",
            error_type="test_error",
            frequency=5.0,
            severity="high",
            first_detected=now,
            last_detected=now,
            occurrence_count=10,
            confidence=0.8,
            description="Test error pattern",
            suggested_action="Restart component",
            is_active=True,
        )

        pattern_dict = pattern.to_dict()

        assert pattern_dict["pattern_id"] == "test_pattern"
        assert pattern_dict["pattern_type"] == "frequency"
        assert pattern_dict["component"] == "test_component"
        assert pattern_dict["error_type"] == "test_error"
        assert pattern_dict["frequency"] == 5.0
        assert pattern_dict["severity"] == "high"
        assert pattern_dict["occurrence_count"] == 10
        assert pattern_dict["confidence"] == 0.8
        assert pattern_dict["description"] == "Test error pattern"
        assert pattern_dict["suggested_action"] == "Restart component"
        assert pattern_dict["is_active"] is True
        assert "first_detected" in pattern_dict
        assert "last_detected" in pattern_dict


class TestValidationErrorScenarios:
    """Test validation error scenarios for field validators."""

    def test_capital_allocation_invalid_percentage(self):
        """Test CapitalAllocation validation with invalid allocation percentage."""
        with pytest.raises(ValidationError):
            CapitalAllocation(
                strategy_id="test_strategy",
                exchange="binance",
                allocated_amount=Decimal("1000"),
                available_amount=Decimal("1000"),
                allocation_percentage=1.5,  # Invalid: > 1.0
                last_rebalance=datetime.now(timezone.utc),
            )

    def test_capital_allocation_negative_percentage(self):
        """Test CapitalAllocation validation with negative allocation percentage."""
        with pytest.raises(ValidationError):
            CapitalAllocation(
                strategy_id="test_strategy",
                exchange="binance",
                allocated_amount=Decimal("1000"),
                available_amount=Decimal("1000"),
                allocation_percentage=-0.1,  # Invalid: < 0.0
                last_rebalance=datetime.now(timezone.utc),
            )

    def test_fund_flow_invalid_amount(self):
        """Test FundFlow validation with invalid amount."""
        with pytest.raises(ValueError, match="Fund flow amount must be positive"):
            FundFlow(
                amount=Decimal("-100"),  # Invalid: negative amount
                currency="USDT",
                reason="Test flow",
                timestamp=datetime.now(timezone.utc),
            )

    def test_fund_flow_zero_amount(self):
        """Test FundFlow validation with zero amount."""
        with pytest.raises(ValueError, match="Fund flow amount must be positive"):
            FundFlow(
                amount=Decimal("0"),  # Invalid: zero amount
                currency="USDT",
                reason="Test flow",
                timestamp=datetime.now(timezone.utc),
            )

    def test_capital_metrics_invalid_utilization_rate(self):
        """Test CapitalMetrics validation with invalid utilization rate."""
        with pytest.raises(ValidationError):
            CapitalMetrics(
                total_capital=Decimal("10000"),
                allocated_capital=Decimal("8000"),
                available_capital=Decimal("2000"),
                utilization_rate=1.2,  # Invalid: > 1.0
                allocation_efficiency=1.0,
                rebalance_frequency_hours=24,
                emergency_reserve=Decimal("1000"),
                last_updated=datetime.now(timezone.utc),
                allocation_count=5,
            )

    def test_capital_metrics_invalid_allocation_efficiency(self):
        """Test CapitalMetrics validation with invalid allocation efficiency."""
        with pytest.raises(ValidationError):
            CapitalMetrics(
                total_capital=Decimal("10000"),
                allocated_capital=Decimal("8000"),
                available_capital=Decimal("2000"),
                utilization_rate=0.8,
                allocation_efficiency=3.5,  # Invalid: > 3.0
                rebalance_frequency_hours=24,
                emergency_reserve=Decimal("1000"),
                last_updated=datetime.now(timezone.utc),
                allocation_count=5,
            )

    def test_currency_exposure_invalid_percentage(self):
        """Test CurrencyExposure validation with invalid exposure percentage."""
        with pytest.raises(ValueError, match="Exposure percentage must be between 0 and 1"):
            CurrencyExposure(
                currency="BTC",
                total_exposure=Decimal("1000"),
                base_currency_equivalent=Decimal("1000"),
                exposure_percentage=1.1,  # Invalid: > 1.0
                hedging_required=False,
                timestamp=datetime.now(timezone.utc),
            )

    def test_exchange_allocation_invalid_score_fields(self):
        """Test ExchangeAllocation validation with invalid score fields."""
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            ExchangeAllocation(
                exchange="binance",
                allocated_amount=Decimal("1000"),
                available_amount=Decimal("1000"),
                utilization_rate=0.8,
                liquidity_score=1.2,  # Invalid: > 1.0
                fee_efficiency=0.7,
                reliability_score=0.9,
                last_rebalance=datetime.now(timezone.utc),
            )

    def test_withdrawal_rule_invalid_percentage_fields(self):
        """Test WithdrawalRule validation with invalid percentage fields."""
        with pytest.raises(ValueError, match="Percentage must be between 0 and 1"):
            WithdrawalRule(
                name="Test Rule",
                description="Test description",
                threshold=1.5,  # Invalid: > 1.0
                min_amount=Decimal("100"),
                max_percentage=0.8,
                cooldown_hours=24,
            )

    def test_capital_protection_invalid_percentage_fields(self):
        """Test CapitalProtection validation with invalid percentage fields."""
        with pytest.raises(ValueError, match="Percentage must be between 0 and 1"):
            CapitalProtection(
                emergency_reserve_pct=1.2,  # Invalid: > 1.0
                max_daily_loss_pct=0.05,
                max_weekly_loss_pct=0.10,
                max_monthly_loss_pct=0.15,
                profit_lock_pct=0.5,
            )

    def test_strategy_config_invalid_percentage_fields(self):
        """Test StrategyConfig validation with invalid percentage fields."""
        with pytest.raises(ValidationError):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=["BTCUSDT"],
                min_confidence=1.2,  # Invalid: > 1.0
                position_size_pct=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )

    def test_strategy_config_invalid_positive_integer(self):
        """Test StrategyConfig validation with invalid positive integer."""
        with pytest.raises(ValidationError):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=["BTCUSDT"],
                max_positions=0,  # Invalid: not positive
                position_size_pct=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )

    def test_strategy_metrics_invalid_win_rate(self):
        """Test StrategyMetrics validation with invalid win rate."""
        with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
            StrategyMetrics(
                total_trades=100,
                winning_trades=60,
                losing_trades=40,
                total_pnl=Decimal("1000"),
                win_rate=1.1,  # Invalid: > 1.0
                last_updated=datetime.now(timezone.utc),
            )

    def test_strategy_config_invalid_symbols_empty_list(self):
        """Test StrategyConfig validation with empty symbols list."""
        with pytest.raises(ValidationError):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=[],  # Invalid: empty list
                position_size_pct=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )

    def test_strategy_config_invalid_symbols_none_list(self):
        """Test StrategyConfig validation with None symbols list."""
        with pytest.raises(ValidationError):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=None,  # Invalid: None
                position_size_pct=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )

    def test_strategy_config_invalid_symbol_format(self):
        """Test StrategyConfig validation with invalid symbol format."""
        with pytest.raises(ValueError, match="Invalid symbol format: AB"):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=["AB"],  # Invalid: too short (< 3 chars)
                position_size_pct=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )

    def test_strategy_config_invalid_symbol_type(self):
        """Test StrategyConfig validation with non-string symbol."""
        with pytest.raises(ValidationError):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=[123],  # Invalid: not a string
                position_size_pct=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )

    def test_strategy_config_invalid_percentage_fields_negative(self):
        """Test StrategyConfig validation with negative percentage fields."""
        with pytest.raises(ValidationError):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=["BTCUSDT"],
                min_confidence=-0.1,  # Invalid: negative
                position_size_pct=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )

    def test_strategy_config_invalid_percentage_fields_above_one(self):
        """Test StrategyConfig validation with percentage fields above 1."""
        with pytest.raises(ValidationError):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=["BTCUSDT"],
                min_confidence=0.6,
                position_size_pct=1.1,  # Invalid: > 1.0
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )

    def test_strategy_config_invalid_positive_integer_negative(self):
        """Test StrategyConfig validation with negative max_positions."""
        with pytest.raises(ValidationError):
            StrategyConfig(
                name="Test Strategy",
                strategy_type=StrategyType.STATIC,
                symbols=["BTCUSDT"],
                max_positions=-1,  # Invalid: negative
                position_size_pct=0.02,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
            )
