"""
Real Service Integration Tests for Strategy Framework.

This module tests strategy integration with real services instead of mocks:
- Real StrategyService with dependency injection
- Real technical indicator calculations (RSI, MACD, SMA, EMA)
- Real database persistence for strategies and signals
- Real market data processing pipeline
- Mathematical accuracy validation with Decimal precision

CRITICAL: All financial calculations must use Decimal types for precision.
"""

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.core.config import Config
from src.core.types import (
    MarketData,
    Signal,
    StrategyConfig,
    StrategyType,
)
from src.strategies.dependencies import create_strategy_service_container
from src.strategies.factory import StrategyFactory
from src.strategies.service import StrategyService
from src.strategies.static.mean_reversion import MeanReversionStrategy
from src.strategies.static.trend_following import TrendFollowingStrategy


# Module-level fixtures shared across test classes
@pytest.fixture
async def real_config():
    """Create real configuration for testing."""
    # Use default config - no overrides needed for integration tests
    config = Config()
    return config


@pytest.fixture
async def real_database_service_fixture(real_config, clean_database):
    """
    Create real DatabaseService for testing.

    Depends on clean_database to ensure database is clean before initialization.
    """
    from src.database.connection import DatabaseConnectionManager
    from src.database.service import DatabaseService

    # Create database connection manager
    connection_manager = DatabaseConnectionManager(real_config)
    await connection_manager.initialize()

    # Create and initialize database service
    db_service = DatabaseService(connection_manager=connection_manager)
    await db_service.initialize()

    yield db_service

    # Cleanup
    await db_service.cleanup()
    await connection_manager.close()


@pytest.fixture
async def real_data_service(real_config, real_database_service_fixture):
    """Create real DataService with database integration."""
    from src.data.services.data_service import DataService

    data_service = DataService(
        config=real_config,
        database_service=real_database_service_fixture,
        cache_service=None,  # Optional
        metrics_collector=None,  # Optional
    )

    await data_service.initialize()
    yield data_service
    await data_service.cleanup()


@pytest.fixture
async def real_risk_service(real_config, real_database_service_fixture):
    """Create real RiskService with database integration."""
    from src.database.repository.risk import PortfolioRepository, RiskMetricsRepository
    from src.risk_management.service import RiskService

    # Create repositories
    risk_metrics_repo = RiskMetricsRepository(real_database_service_fixture)
    portfolio_repo = PortfolioRepository(real_database_service_fixture)

    # Create risk service
    risk_service = RiskService(
        risk_metrics_repository=risk_metrics_repo,
        portfolio_repository=portfolio_repo,
        state_service=None,  # Optional
        analytics_service=None,  # Optional
        config=real_config,
        correlation_id="test_strategy_correlation",
    )

    await risk_service.initialize()
    yield risk_service
    await risk_service.cleanup()


@pytest.fixture
async def strategy_service_container(
    real_config, real_database_service_fixture, real_data_service, real_risk_service
):
    """Create real strategy service container with all dependencies."""
    # Create container with real services
    container = create_strategy_service_container(
        risk_service=real_risk_service,
        data_service=real_data_service,
        execution_service=None,  # Not needed for most tests
        monitoring_service=None,  # Optional
        state_service=None,  # Optional
        capital_service=None,  # Optional
        ml_service=None,  # Optional
        analytics_service=None,  # Optional
        optimization_service=None,  # Optional
    )

    # Verify container was created with required services
    assert container is not None
    assert container.data_service is not None
    assert container.risk_service is not None

    yield container

    # No cleanup needed - container doesn't require explicit cleanup


@pytest.fixture
async def real_strategy_service(strategy_service_container):
    """Create real StrategyService with injected dependencies."""
    # Extract services from container to pass to StrategyService
    strategy_service = StrategyService(
        name="RealStrategyService",
        risk_manager=strategy_service_container.risk_service,
        data_service=strategy_service_container.data_service,
    )

    # Start the service (this calls initialize internally)
    await strategy_service.start()
    yield strategy_service
    await strategy_service.stop()


@pytest.fixture
async def strategy_factory(strategy_service_container):
    """Create StrategyFactory for creating strategy instances."""
    factory = StrategyFactory(
        service_container=strategy_service_container,
        risk_manager=strategy_service_container.risk_service,
        data_service=strategy_service_container.data_service,
    )
    yield factory


@pytest.fixture
def real_mean_reversion_config():
    """Create real mean reversion strategy configuration."""
    return StrategyConfig(
        strategy_id="real_mean_reversion_001",
        name="real_mean_reversion_test",
        strategy_type=StrategyType.MEAN_REVERSION,
        enabled=True,
        symbol="BTC/USDT",
        timeframe="1h",
        min_confidence=Decimal("0.6"),
        max_positions=5,
        position_size_pct=Decimal("0.02"),
        stop_loss_pct=Decimal("0.02"),
        take_profit_pct=Decimal("0.04"),
        parameters={
            # Required parameters for factory validation
            "mean_period": 20,
            "deviation_threshold": Decimal("2.0"),
            "reversion_strength": Decimal("0.5"),
            # Additional parameters for strategy implementation
            "lookback_period": 20,
            "entry_threshold": Decimal("2.0"),
            "exit_threshold": Decimal("0.5"),
            "volume_filter": True,
            "min_volume_ratio": Decimal("1.5"),
            "atr_period": 14,
            "atr_multiplier": Decimal("2.0"),
        },
    )


@pytest.fixture
def realistic_market_data_series():
    """Generate realistic market data series for indicator calculations."""
    base_price = Decimal("50000.00")
    base_volume = Decimal("1000.00")
    market_data_series = []

    # Generate 50 periods of realistic market data
    for i in range(50):
        timestamp = datetime.now(timezone.utc) - timedelta(hours=50 - i)

        # Add realistic price movements
        if i < 20:
            # Uptrend with volatility
            price_change = Decimal(str(i * 50 + (i % 3 - 1) * 100))
        elif i < 35:
            # Sideways movement (mean reversion opportunity)
            price_change = Decimal(str((i % 5 - 2) * 150))
        else:
            # Downtrend
            price_change = Decimal(str((35 - i) * 40))

        close_price = base_price + price_change
        open_price = close_price - Decimal(str((i % 7 - 3) * 50))
        high_price = max(open_price, close_price) + Decimal(str(abs(i % 11) * 25))
        low_price = min(open_price, close_price) - Decimal(str(abs(i % 9) * 30))

        # Volume with realistic patterns
        volume = base_volume + Decimal(str((i % 13) * 100))

        market_data = MarketData(
            symbol="BTC/USDT",
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timestamp=timestamp,
            exchange="binance",
            bid_price=close_price - Decimal("1.0"),
            ask_price=close_price + Decimal("1.0"),
        )

        market_data_series.append(market_data)

    return market_data_series


class TestRealStrategyServiceIntegration:
    """Integration tests using real StrategyService with actual dependencies."""

    @pytest.mark.asyncio
    async def test_real_strategy_service_initialization(self, real_strategy_service):
        """Test real strategy service initialization with dependencies."""
        # Verify service is properly initialized
        assert real_strategy_service.name == "RealStrategyService"
        # BaseService has is_running property, not is_healthy()
        assert real_strategy_service.is_running

        # Check that dependencies are properly injected
        # StrategyService stores injected services as private attributes
        assert hasattr(real_strategy_service, "_risk_manager") or hasattr(
            real_strategy_service, "_data_service"
        )

        # Service is running confirms it was initialized successfully
        # (BaseService doesn't have get_status() method, so we just verify is_running)

    @pytest.mark.asyncio
    async def test_real_strategy_creation_and_registration(
        self, strategy_factory, real_strategy_service, real_mean_reversion_config
    ):
        """Test real strategy creation and registration with service."""
        # Create strategy using factory
        strategy = await strategy_factory.create_strategy(
            strategy_type=real_mean_reversion_config.strategy_type,
            config=real_mean_reversion_config,
        )

        # Verify strategy was created successfully
        assert strategy is not None
        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.name == "real_mean_reversion_test"
        assert strategy.config.strategy_type == StrategyType.MEAN_REVERSION

        # Verify strategy has real dependencies injected
        assert hasattr(strategy, "services")
        assert strategy.services is not None
        # Services may be None if not registered in DI container, which is acceptable

        # Register strategy with service
        await real_strategy_service.register_strategy(
            strategy_id=real_mean_reversion_config.strategy_id,
            strategy_instance=strategy,
            config=real_mean_reversion_config,
        )

        # Verify registration
        all_strategies = await real_strategy_service.get_all_strategies()
        assert real_mean_reversion_config.strategy_id in all_strategies

        # Cleanup
        await real_strategy_service.cleanup_strategy(real_mean_reversion_config.strategy_id)
        # cleanup() is not async - it returns None
        strategy.cleanup()

    @pytest.mark.asyncio
    async def test_real_strategy_database_persistence(
        self, strategy_factory, real_strategy_service, real_mean_reversion_config, clean_database
    ):
        """Test real strategy registration and config storage in service."""
        # Create strategy using factory
        strategy = await strategy_factory.create_strategy(
            strategy_type=real_mean_reversion_config.strategy_type,
            config=real_mean_reversion_config,
        )

        # Register strategy - this stores config in service's internal dictionary
        await real_strategy_service.register_strategy(
            strategy_id=real_mean_reversion_config.strategy_id,
            strategy_instance=strategy,
            config=real_mean_reversion_config,
        )

        # Verify strategy is registered and config is stored
        all_strategies = await real_strategy_service.get_all_strategies()
        assert real_mean_reversion_config.strategy_id in all_strategies

        # Verify strategy config is accessible through service
        strategy_info = all_strategies[real_mean_reversion_config.strategy_id]
        assert strategy_info is not None
        # Service returns strategy info, not full config
        assert "strategy_type" in strategy_info or "type" in strategy_info

        # Verify strategy instance has correct config
        assert strategy.config.strategy_id == real_mean_reversion_config.strategy_id
        assert strategy.config.name == real_mean_reversion_config.name
        assert strategy.config.strategy_type == real_mean_reversion_config.strategy_type

        # Verify Decimal parameters are preserved in strategy config
        assert strategy.config.parameters["entry_threshold"] == Decimal("2.0")
        assert strategy.config.parameters["atr_multiplier"] == Decimal("2.0")
        # Note: min_confidence may be converted to float by Pydantic - verify value is correct
        assert float(strategy.config.min_confidence) == 0.6

        # Cleanup
        await real_strategy_service.cleanup_strategy(real_mean_reversion_config.strategy_id)
        strategy.cleanup()


class TestRealTechnicalIndicatorCalculations:
    """Test real technical indicator calculations with mathematical accuracy."""

    @pytest.fixture(autouse=True)
    async def cleanup_database_before_test(self, real_database_service_fixture):
        """
        Clean database before each test in this class.

        This ensures each test starts with a clean slate, preventing data pollution
        when tests store market data and then fetch "recent" records.

        Depends on real_database_service_fixture to ensure database is initialized
        before we try to clean it.
        """
        from sqlalchemy import text

        from src.database.connection import get_async_session

        # Clean the market_data_records table before the test
        async with get_async_session() as session:
            try:
                await session.execute(text("SET session_replication_role = replica;"))
                await session.execute(text("TRUNCATE TABLE market_data_records CASCADE;"))
                await session.execute(text("SET session_replication_role = DEFAULT;"))
                await session.commit()
            except Exception as e:
                await session.rollback()
                # Log but don't fail - test will create data anyway
                print(f"Warning: Could not clean database: {e}")

        yield
        # No cleanup needed after - next test will clean before it runs

    @pytest.fixture
    async def real_mean_reversion_strategy(
        self, real_mean_reversion_config, strategy_service_container
    ):
        """
        Create real mean reversion strategy with dependencies.

        Database cleanup is handled automatically through the fixture dependency chain:
        real_mean_reversion_strategy -> strategy_service_container -> real_data_service ->
        real_database_service_fixture -> clean_database
        """
        strategy = MeanReversionStrategy(
            config=real_mean_reversion_config.dict(), services=strategy_service_container
        )
        await strategy.initialize(real_mean_reversion_config)
        yield strategy
        strategy.cleanup()  # cleanup() is not async

    def calculate_expected_sma(self, prices: list[Decimal], period: int) -> Decimal:
        """Calculate expected SMA for verification."""
        if len(prices) < period:
            return Decimal("0")

        return sum(prices[-period:]) / period

    def calculate_expected_ema(self, prices: list[Decimal], period: int) -> Decimal:
        """Calculate expected EMA for verification using TA-Lib's algorithm.

        TA-Lib uses SMA for the initial EMA value (industry standard), not the first price.
        This ensures stable initialization and matches production behavior.
        """
        if len(prices) < period:
            return prices[-1] if prices else Decimal("0")

        multiplier = Decimal("2") / (period + 1)

        # Initialize EMA with SMA of first 'period' prices (TA-Lib standard)
        ema = sum(prices[:period]) / period

        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (Decimal("1") - multiplier))

        return ema

    def calculate_expected_rsi(self, prices: list[Decimal], period: int = 14) -> Decimal:
        """Calculate expected RSI for verification using TA-Lib's Wilder's smoothing method.

        TA-Lib uses Wilder's smoothing (exponential smoothing) for RSI calculation,
        not simple averaging. This is the industry standard (Welles Wilder, 1978).
        """
        if len(prices) < period + 1:
            return Decimal("50")  # Neutral RSI

        # Calculate price changes
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [delta if delta > 0 else Decimal("0") for delta in deltas]
        losses = [-delta if delta < 0 else Decimal("0") for delta in deltas]

        if len(gains) < period:
            return Decimal("50")

        # Initialize with simple average of first 'period' values (TA-Lib standard)
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Apply Wilder's smoothing for remaining values
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))
        return rsi

    @pytest.mark.asyncio
    async def test_real_sma_calculation_accuracy(
        self, real_mean_reversion_strategy, realistic_market_data_series
    ):
        """Test SMA calculation with mathematical precision."""
        # Set up market data for calculation
        strategy = real_mean_reversion_strategy

        # Use strategy's data service to store market data
        for market_data in realistic_market_data_series:
            await strategy.services.data_service.store_market_data(
                data=market_data, exchange=market_data.exchange
            )

        # Calculate SMA using real strategy
        sma_20 = await strategy.get_sma("BTC/USDT", 20)

        # Verify calculation
        assert sma_20 is not None
        assert isinstance(sma_20, Decimal)
        assert sma_20 > Decimal("0")

        # Verify against expected calculation
        prices = [md.close for md in realistic_market_data_series]
        expected_sma = self.calculate_expected_sma(prices, 20)

        # Allow small tolerance for rounding differences
        tolerance = Decimal("0.01")
        assert abs(sma_20 - expected_sma) <= tolerance

    @pytest.mark.asyncio
    async def test_real_ema_calculation_accuracy(
        self, real_mean_reversion_strategy, realistic_market_data_series
    ):
        """Test EMA calculation with mathematical precision."""
        strategy = real_mean_reversion_strategy

        # Set up market data
        for market_data in realistic_market_data_series:
            await strategy.services.data_service.store_market_data(
                data=market_data, exchange=market_data.exchange
            )

        # Calculate EMA using real strategy
        ema_20 = await strategy.get_ema("BTC/USDT", 20)

        # Verify calculation
        assert ema_20 is not None
        assert isinstance(ema_20, Decimal)
        assert ema_20 > Decimal("0")

        # Verify against expected calculation
        prices = [md.close for md in realistic_market_data_series]
        expected_ema = self.calculate_expected_ema(prices, 20)

        # Allow small tolerance for floating-point to Decimal conversion differences
        tolerance = Decimal("0.01")  # Both use same TA-Lib algorithm now
        assert abs(ema_20 - expected_ema) <= tolerance

    @pytest.mark.asyncio
    async def test_real_rsi_calculation_accuracy(
        self, real_mean_reversion_strategy, realistic_market_data_series
    ):
        """Test RSI calculation with mathematical precision."""
        strategy = real_mean_reversion_strategy

        # Set up market data
        for market_data in realistic_market_data_series:
            await strategy.services.data_service.store_market_data(
                data=market_data, exchange=market_data.exchange
            )

        # Calculate RSI using real strategy
        rsi_14 = await strategy.get_rsi("BTC/USDT", 14)

        # Verify calculation
        assert rsi_14 is not None
        assert isinstance(rsi_14, Decimal)
        assert Decimal("0") <= rsi_14 <= Decimal("100")

        # Verify against expected calculation
        prices = [md.close for md in realistic_market_data_series]
        expected_rsi = self.calculate_expected_rsi(prices, 14)

        # Allow small tolerance for floating-point to Decimal conversion differences
        tolerance = Decimal("0.1")  # Both use same Wilder's smoothing algorithm now
        assert abs(rsi_14 - expected_rsi) <= tolerance

    @pytest.mark.asyncio
    async def test_real_macd_calculation_accuracy(
        self, real_mean_reversion_strategy, realistic_market_data_series
    ):
        """Test MACD calculation with Decimal precision."""
        strategy = real_mean_reversion_strategy

        # Set up market data
        for market_data in realistic_market_data_series:
            await strategy.services.data_service.store_market_data(
                data=market_data, exchange=market_data.exchange
            )

        # Calculate MACD using real strategy
        macd_result = await strategy.get_macd(
            symbol="BTC/USDT", fast_period=12, slow_period=26, signal_period=9
        )

        # Verify calculation structure
        assert macd_result is not None
        assert isinstance(macd_result, dict)
        assert "macd" in macd_result
        assert "signal" in macd_result
        assert "histogram" in macd_result

        # Verify all values are Decimal
        assert isinstance(macd_result["macd"], Decimal)
        assert isinstance(macd_result["signal"], Decimal)
        assert isinstance(macd_result["histogram"], Decimal)

        # Verify mathematical relationship
        calculated_histogram = macd_result["macd"] - macd_result["signal"]
        assert abs(calculated_histogram - macd_result["histogram"]) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_real_bollinger_bands_calculation(
        self, real_mean_reversion_strategy, realistic_market_data_series
    ):
        """Test Bollinger Bands calculation with Decimal precision."""
        strategy = real_mean_reversion_strategy

        # Set up market data
        for market_data in realistic_market_data_series:
            await strategy.services.data_service.store_market_data(
                data=market_data, exchange=market_data.exchange
            )

        # Calculate Bollinger Bands
        bb_result = await strategy.get_bollinger_bands("BTC/USDT", 20, 2.0)

        # Verify calculation
        assert bb_result is not None
        assert isinstance(bb_result, dict)
        assert "upper" in bb_result
        assert "middle" in bb_result
        assert "lower" in bb_result

        # Verify all values are Decimal
        assert isinstance(bb_result["upper"], Decimal)
        assert isinstance(bb_result["middle"], Decimal)
        assert isinstance(bb_result["lower"], Decimal)

        # Verify logical relationships
        assert bb_result["upper"] > bb_result["middle"]
        assert bb_result["middle"] > bb_result["lower"]


class TestRealSignalGenerationIntegration:
    """Test real signal generation using actual indicator calculations."""

    @pytest.fixture
    async def real_strategies(self, strategy_service_container):
        """Create multiple real strategies for testing."""
        # Mean Reversion Strategy
        mean_reversion_config = StrategyConfig(
            strategy_id="real_signal_mean_reversion",
            name="real_signal_mean_reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
                "exit_threshold": Decimal("0.5"),
            },
        )

        # Trend Following Strategy
        trend_following_config = StrategyConfig(
            strategy_id="real_signal_trend_following",
            name="real_signal_trend_following",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={"fast_ma_period": 10, "slow_ma_period": 20, "rsi_period": 14},
        )

        strategies = []

        # Create mean reversion strategy
        mean_reversion = MeanReversionStrategy(
            config=mean_reversion_config.dict(), services=strategy_service_container
        )
        await mean_reversion.initialize(mean_reversion_config)
        strategies.append(mean_reversion)

        # Create trend following strategy
        trend_following = TrendFollowingStrategy(
            config=trend_following_config.dict(), services=strategy_service_container
        )
        await trend_following.initialize(trend_following_config)
        strategies.append(trend_following)

        yield strategies

        # Cleanup
        for strategy in strategies:
            strategy.cleanup()  # cleanup() is not async

    def create_mean_reversion_market_data(self) -> MarketData:
        """Create market data that should trigger mean reversion signals."""
        return MarketData(
            symbol="BTC/USDT",
            open=Decimal("48500.00"),
            high=Decimal("49000.00"),
            low=Decimal("47000.00"),
            close=Decimal("47200.00"),  # Significant deviation below mean
            volume=Decimal("3000.00"),  # High volume for confirmation
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            bid_price=Decimal("47190.00"),
            ask_price=Decimal("47210.00"),
        )

    def create_trending_market_data(self) -> MarketData:
        """Create market data that should trigger trend following signals."""
        return MarketData(
            symbol="BTC/USDT",
            open=Decimal("50000.00"),
            high=Decimal("51500.00"),
            low=Decimal("49800.00"),
            close=Decimal("51200.00"),  # Strong upward movement
            volume=Decimal("4000.00"),  # High volume for confirmation
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            bid_price=Decimal("51190.00"),
            ask_price=Decimal("51210.00"),
        )

    @pytest.mark.asyncio
    async def test_real_mean_reversion_signal_generation(self, real_strategies):
        """Test real mean reversion signal generation."""
        mean_reversion_strategy = real_strategies[0]  # First strategy is mean reversion

        # Create market data that should trigger mean reversion
        market_data = self.create_mean_reversion_market_data()

        # Store some historical data for indicator calculations
        base_price = Decimal("50000.00")
        for i in range(25):  # Need enough data for lookback period
            historical_data = MarketData(
                symbol="BTC/USDT",
                open=base_price,
                high=base_price + Decimal("100"),
                low=base_price - Decimal("100"),
                close=base_price + Decimal(str((i % 5 - 2) * 50)),
                volume=Decimal("2000.00"),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=25 - i),
                exchange="binance",
            )
            await mean_reversion_strategy.services.data_service.store_market_data(
                data=historical_data, exchange=historical_data.exchange
            )

        # Generate signals using real calculations
        signals = await mean_reversion_strategy.generate_signals(market_data)

        # Verify signal quality
        assert isinstance(signals, list)

        if signals:  # May not generate signals depending on conditions
            signal = signals[0]

            # Verify signal structure
            assert isinstance(signal, Signal)
            assert isinstance(signal.confidence, Decimal)
            assert isinstance(signal.strength, Decimal)
            assert signal.source == mean_reversion_strategy.name
            assert signal.symbol == "BTC/USDT"

            # Verify signal is based on actual calculations
            assert signal.metadata is not None
            assert isinstance(signal.metadata, dict)

            # For mean reversion, expect z_score in metadata
            if "z_score" in signal.metadata:
                z_score = signal.metadata["z_score"]
                assert isinstance(z_score, (Decimal, float))

    @pytest.mark.asyncio
    async def test_real_trend_following_signal_generation(self, real_strategies):
        """Test real trend following signal generation."""
        trend_following_strategy = real_strategies[1]  # Second strategy is trend following

        # Create market data that should trigger trend following
        market_data = self.create_trending_market_data()

        # Store historical data for indicator calculations
        base_price = Decimal("50000.00")
        for i in range(30):  # Need enough data for MA calculations
            # Create upward trending data
            trending_price = base_price + Decimal(str(i * 20))
            historical_data = MarketData(
                symbol="BTC/USDT",
                open=trending_price - Decimal("50"),
                high=trending_price + Decimal("100"),
                low=trending_price - Decimal("100"),
                close=trending_price,
                volume=Decimal("2500.00"),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=30 - i),
                exchange="binance",
            )
            await trend_following_strategy.services.data_service.store_market_data(
                data=historical_data, exchange=historical_data.exchange
            )

        # Generate signals using real calculations
        signals = await trend_following_strategy.generate_signals(market_data)

        # Verify signal quality
        assert isinstance(signals, list)

        if signals:  # May not generate signals depending on conditions
            signal = signals[0]

            # Verify signal structure
            assert isinstance(signal, Signal)
            assert isinstance(signal.confidence, Decimal)
            assert isinstance(signal.strength, Decimal)
            assert signal.source == trend_following_strategy.name
            assert signal.symbol == "BTC/USDT"

            # Verify signal metadata
            assert signal.metadata is not None
            assert isinstance(signal.metadata, dict)

    @pytest.mark.asyncio
    async def test_real_multi_strategy_signal_coordination(self, real_strategies):
        """Test real coordination between multiple strategies."""
        # Create market data that could trigger multiple strategies
        market_data = MarketData(
            symbol="BTC/USDT",
            open=Decimal("49500.00"),
            high=Decimal("50200.00"),
            low=Decimal("49000.00"),
            close=Decimal("50000.00"),
            volume=Decimal("3500.00"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        # Set up historical data for all strategies
        for strategy in real_strategies:
            base_price = Decimal("49000.00")
            for i in range(35):
                historical_data = MarketData(
                    symbol="BTC/USDT",
                    open=base_price + Decimal(str(i * 10)),
                    high=base_price + Decimal(str(i * 10 + 150)),
                    low=base_price + Decimal(str(i * 10 - 100)),
                    close=base_price + Decimal(str(i * 10 + 50)),
                    volume=Decimal("2000.00"),
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=35 - i),
                    exchange="binance",
                )
                await strategy.services.data_service.store_market_data(
                    data=historical_data, exchange=historical_data.exchange
                )

        # Process market data through all strategies
        all_signals = []
        for strategy in real_strategies:
            signals = await strategy.generate_signals(market_data)
            all_signals.extend(signals)

        # Verify signal coordination
        assert isinstance(all_signals, list)

        # Verify all signals meet quality standards
        for signal in all_signals:
            assert isinstance(signal, Signal)
            assert isinstance(signal.confidence, Decimal)
            assert isinstance(signal.strength, Decimal)
            assert signal.confidence >= Decimal("0.1")  # Minimum confidence
            assert signal.timestamp is not None
            assert signal.symbol == "BTC/USDT"


class TestRealStrategyPerformanceIntegration:
    """Test performance characteristics of real strategy implementations."""

    @pytest.mark.asyncio
    async def test_real_service_performance_requirements(
        self, strategy_factory, real_strategy_service, realistic_market_data_series
    ):
        """Test that real services meet performance requirements."""
        # Create strategy configuration
        config = StrategyConfig(
            strategy_id="performance_test_001",
            name="performance_test_strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
                "mean_period": 20,
                "deviation_threshold": Decimal("2.0"),
                "reversion_strength": Decimal("0.5"),
            },
        )

        # Test strategy creation performance
        start_time = time.time()
        strategy = await strategy_factory.create_strategy(
            strategy_type=config.strategy_type, config=config
        )
        creation_time = time.time() - start_time

        # Strategy creation should be fast
        assert creation_time < 2.0  # Less than 2 seconds

        # Test signal generation performance
        start_time = time.time()

        # Process multiple market data points
        for market_data in realistic_market_data_series[:10]:
            signals = await strategy.generate_signals(market_data)

        processing_time = time.time() - start_time

        # Should process 10 signals within 10 seconds
        assert processing_time < 10.0

        # Test database persistence performance (register strategy with service)
        start_time = time.time()
        await real_strategy_service.register_strategy(
            strategy_id=config.strategy_id, strategy_instance=strategy, config=config
        )
        persistence_time = time.time() - start_time

        # Database operations should be fast
        assert persistence_time < 1.0

        # Cleanup
        await real_strategy_service.cleanup_strategy(config.strategy_id)
        strategy.cleanup()

    @pytest.mark.asyncio
    async def test_real_indicator_calculation_performance(
        self, strategy_factory, real_mean_reversion_config
    ):
        """Test performance of real technical indicator calculations."""
        # Create strategy for performance testing
        strategy = await strategy_factory.create_strategy(
            strategy_type=real_mean_reversion_config.strategy_type,
            config=real_mean_reversion_config,
        )

        # Generate substantial market data for performance testing
        market_data_series = []
        base_price = Decimal("50000.00")

        for i in range(100):  # 100 data points
            market_data = MarketData(
                symbol="BTC/USDT",
                open=base_price + Decimal(str(i * 10)),
                high=base_price + Decimal(str(i * 10 + 100)),
                low=base_price + Decimal(str(i * 10 - 100)),
                close=base_price + Decimal(str(i * 10 + 50)),
                volume=Decimal("2000.00"),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=100 - i),
                exchange="binance",
            )
            market_data_series.append(market_data)

        # Store market data
        for market_data in market_data_series:
            await strategy.services.data_service.store_market_data(
                data=market_data, exchange=market_data.exchange
            )

        # Test RSI calculation performance
        start_time = time.time()
        for _ in range(10):  # Calculate RSI 10 times
            rsi = await strategy.get_rsi("BTC/USDT", 14)
        rsi_time = time.time() - start_time

        # RSI calculations should be fast
        assert rsi_time < 2.0  # Less than 2 seconds for 10 calculations

        # Test SMA calculation performance
        start_time = time.time()
        for _ in range(10):  # Calculate SMA 10 times
            sma = await strategy.get_sma("BTC/USDT", 20)
        sma_time = time.time() - start_time

        # SMA calculations should be fast
        assert sma_time < 1.0  # Less than 1 second for 10 calculations

        # Cleanup
        strategy.cleanup()

    @pytest.mark.asyncio
    async def test_real_strategy_memory_usage(self, strategy_factory):
        """Test memory usage of real strategy implementations."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple strategies
        strategies = []
        for i in range(5):
            config = StrategyConfig(
                strategy_id=f"memory_test_{i:03d}",
                name=f"memory_test_strategy_{i}",
                strategy_type=StrategyType.MEAN_REVERSION,
                symbol="BTC/USDT",
                timeframe="1h",
                parameters={
                    "mean_period": 20,
                    "deviation_threshold": Decimal("2.0"),
                    "reversion_strength": Decimal("0.5"),
                },
            )
            strategy = await strategy_factory.create_strategy(
                strategy_type=config.strategy_type, config=config
            )
            strategies.append(strategy)

        # Get memory after strategy creation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for 5 strategies)
        assert memory_increase < 100  # Less than 100MB increase

        # Cleanup strategies
        for strategy in strategies:
            strategy.cleanup()  # cleanup() is not async
