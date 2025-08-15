"""
Tests for Correlation Monitoring System.

This module provides comprehensive tests for correlation-based circuit breakers
and portfolio risk management functionality.

CRITICAL: Tests must verify Decimal precision, thread-safety, and various correlation scenarios.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import Config
from src.core.types import MarketData, Position, OrderSide
from src.risk_management.correlation_monitor import (
    CorrelationMonitor,
    CorrelationThresholds,
    CorrelationMetrics,
)


@pytest.fixture
def sample_config():
    """Create sample configuration for tests."""
    config = Config()
    return config


@pytest.fixture
def correlation_thresholds():
    """Create sample correlation thresholds."""
    return CorrelationThresholds(
        warning_threshold=Decimal("0.6"),
        critical_threshold=Decimal("0.8"),
        max_positions_high_corr=3,
        max_positions_critical_corr=1,
        lookback_periods=20,
        min_periods=5
    )


@pytest.fixture
def correlation_monitor(sample_config, correlation_thresholds):
    """Create correlation monitor for testing."""
    return CorrelationMonitor(sample_config, correlation_thresholds)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    base_time = datetime.now()
    return [
        MarketData(
            symbol="BTC/USD",
            price=Decimal("50000.00"),
            volume=Decimal("1000.0"),
            timestamp=base_time - timedelta(minutes=5)
        ),
        MarketData(
            symbol="BTC/USD", 
            price=Decimal("50100.00"),
            volume=Decimal("1200.0"),
            timestamp=base_time - timedelta(minutes=4)
        ),
        MarketData(
            symbol="BTC/USD",
            price=Decimal("49950.00"), 
            volume=Decimal("800.0"),
            timestamp=base_time - timedelta(minutes=3)
        ),
        MarketData(
            symbol="ETH/USD",
            price=Decimal("3000.00"),
            volume=Decimal("2000.0"),
            timestamp=base_time - timedelta(minutes=5)
        ),
        MarketData(
            symbol="ETH/USD",
            price=Decimal("3030.00"),
            volume=Decimal("2200.0"),
            timestamp=base_time - timedelta(minutes=4)
        ),
        MarketData(
            symbol="ETH/USD",
            price=Decimal("2985.00"),
            volume=Decimal("1800.0"),
            timestamp=base_time - timedelta(minutes=3)
        ),
    ]


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    timestamp = datetime.now()
    return [
        Position(
            symbol="BTC/USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("50100.00"),
            unrealized_pnl=Decimal("100.00"),
            side=OrderSide.BUY,
            timestamp=timestamp
        ),
        Position(
            symbol="ETH/USD", 
            quantity=Decimal("10.0"),
            entry_price=Decimal("3000.00"),
            current_price=Decimal("3030.00"),
            unrealized_pnl=Decimal("300.00"),
            side=OrderSide.BUY,
            timestamp=timestamp
        ),
    ]


class TestCorrelationThresholds:
    """Test correlation threshold configuration."""
    
    def test_default_thresholds(self):
        """Test default correlation thresholds."""
        thresholds = CorrelationThresholds()
        
        assert thresholds.warning_threshold == Decimal("0.6")
        assert thresholds.critical_threshold == Decimal("0.8")
        assert thresholds.max_positions_high_corr == 3
        assert thresholds.max_positions_critical_corr == 1
        assert thresholds.lookback_periods == 50
        assert thresholds.min_periods == 10
    
    def test_custom_thresholds(self):
        """Test custom correlation thresholds."""
        thresholds = CorrelationThresholds(
            warning_threshold=Decimal("0.7"),
            critical_threshold=Decimal("0.9"),
            max_positions_high_corr=2,
            max_positions_critical_corr=0,
            lookback_periods=30,
            min_periods=8
        )
        
        assert thresholds.warning_threshold == Decimal("0.7")
        assert thresholds.critical_threshold == Decimal("0.9")
        assert thresholds.max_positions_high_corr == 2
        assert thresholds.max_positions_critical_corr == 0


class TestCorrelationMonitor:
    """Test correlation monitor functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, correlation_monitor):
        """Test correlation monitor initialization."""
        assert correlation_monitor is not None
        assert correlation_monitor.thresholds.warning_threshold == Decimal("0.6")
        assert len(correlation_monitor.price_history) == 0
        assert len(correlation_monitor.return_history) == 0
    
    @pytest.mark.asyncio
    async def test_update_price_data(self, correlation_monitor, sample_market_data):
        """Test updating price data."""
        market_data = sample_market_data[0]
        
        await correlation_monitor.update_price_data(market_data)
        
        assert len(correlation_monitor.price_history["BTC/USD"]) == 1
        assert len(correlation_monitor.return_history["BTC/USD"]) == 0  # Need 2 prices for return
        
        # Add second price data point
        market_data2 = sample_market_data[1]
        await correlation_monitor.update_price_data(market_data2)
        
        assert len(correlation_monitor.price_history["BTC/USD"]) == 2
        assert len(correlation_monitor.return_history["BTC/USD"]) == 1
        
        # Check return calculation
        returns = list(correlation_monitor.return_history["BTC/USD"])
        expected_return = (Decimal("50100.00") - Decimal("50000.00")) / Decimal("50000.00")
        assert returns[0] == expected_return
    
    @pytest.mark.asyncio
    async def test_pairwise_correlation_insufficient_data(self, correlation_monitor):
        """Test pairwise correlation with insufficient data."""
        correlation = await correlation_monitor.calculate_pairwise_correlation("BTC/USD", "ETH/USD")
        assert correlation is None
    
    @pytest.mark.asyncio
    async def test_pairwise_correlation_same_symbol(self, correlation_monitor):
        """Test pairwise correlation for same symbol."""
        correlation = await correlation_monitor.calculate_pairwise_correlation("BTC/USD", "BTC/USD")
        assert correlation == Decimal("1.0")
    
    @pytest.mark.asyncio
    async def test_pairwise_correlation_calculation(self, correlation_monitor):
        """Test pairwise correlation calculation with sufficient data."""
        # Add correlated price data (both moving up and down together)
        base_time = datetime.now()
        
        # BTC prices: 50000 -> 50100 -> 49950 -> 50200 -> 49800 -> 50300
        btc_prices = [50000, 50100, 49950, 50200, 49800, 50300]
        # ETH prices: 3000 -> 3030 -> 2985 -> 3060 -> 2970 -> 3090 (correlated moves)
        eth_prices = [3000, 3030, 2985, 3060, 2970, 3090]
        
        for i, (btc_price, eth_price) in enumerate(zip(btc_prices, eth_prices)):
            timestamp = base_time - timedelta(minutes=10-i)
            
            btc_data = MarketData(
                symbol="BTC/USD",
                price=Decimal(str(btc_price)),
                volume=Decimal("1000.0"),
                timestamp=timestamp
            )
            eth_data = MarketData(
                symbol="ETH/USD", 
                price=Decimal(str(eth_price)),
                volume=Decimal("2000.0"),
                timestamp=timestamp
            )
            
            await correlation_monitor.update_price_data(btc_data)
            await correlation_monitor.update_price_data(eth_data)
        
        correlation = await correlation_monitor.calculate_pairwise_correlation("BTC/USD", "ETH/USD")
        
        assert correlation is not None
        # Should be positive correlation due to correlated price movements
        assert correlation > Decimal("0.0")
        assert Decimal("-1.0") <= correlation <= Decimal("1.0")
    
    @pytest.mark.asyncio
    async def test_portfolio_correlation_empty_positions(self, correlation_monitor):
        """Test portfolio correlation with empty positions."""
        metrics = await correlation_monitor.calculate_portfolio_correlation([])
        
        assert metrics.average_correlation == Decimal("0.0")
        assert metrics.max_pairwise_correlation == Decimal("0.0")
        assert metrics.correlation_spike is False
        assert metrics.correlated_pairs_count == 0
    
    @pytest.mark.asyncio
    async def test_portfolio_correlation_single_position(self, correlation_monitor, sample_positions):
        """Test portfolio correlation with single position."""
        single_position = [sample_positions[0]]
        metrics = await correlation_monitor.calculate_portfolio_correlation(single_position)
        
        assert metrics.average_correlation == Decimal("0.0")
        assert metrics.max_pairwise_correlation == Decimal("0.0")
        assert metrics.correlation_spike is False
        assert metrics.correlated_pairs_count == 0
    
    @pytest.mark.asyncio
    async def test_portfolio_correlation_multiple_positions(self, correlation_monitor, sample_positions):
        """Test portfolio correlation with multiple positions."""
        # First populate with price data to enable correlation calculation
        base_time = datetime.now()
        prices_btc = [50000, 50100, 49950, 50200, 49800, 50300]
        prices_eth = [3000, 3030, 2985, 3060, 2970, 3090]
        
        for i, (btc_price, eth_price) in enumerate(zip(prices_btc, prices_eth)):
            timestamp = base_time - timedelta(minutes=10-i)
            
            await correlation_monitor.update_price_data(MarketData(
                symbol="BTC/USD", price=Decimal(str(btc_price)),
                volume=Decimal("1000.0"), timestamp=timestamp
            ))
            await correlation_monitor.update_price_data(MarketData(
                symbol="ETH/USD", price=Decimal(str(eth_price)),
                volume=Decimal("2000.0"), timestamp=timestamp
            ))
        
        metrics = await correlation_monitor.calculate_portfolio_correlation(sample_positions)
        
        assert isinstance(metrics.average_correlation, Decimal)
        assert isinstance(metrics.max_pairwise_correlation, Decimal)
        assert isinstance(metrics.correlation_spike, bool)
        assert isinstance(metrics.correlated_pairs_count, int)
        assert isinstance(metrics.portfolio_concentration_risk, Decimal)
        assert len(metrics.correlation_matrix) >= 0
    
    @pytest.mark.asyncio
    async def test_position_limits_normal_correlation(self, correlation_monitor):
        """Test position limits with normal correlation."""
        metrics = CorrelationMetrics(
            average_correlation=Decimal("0.3"),
            max_pairwise_correlation=Decimal("0.4"),
            correlation_spike=False,
            correlated_pairs_count=0,
            portfolio_concentration_risk=Decimal("0.2"),
            timestamp=datetime.now(),
            correlation_matrix={}
        )
        
        limits = await correlation_monitor.get_position_limits_for_correlation(metrics)
        
        assert limits["max_positions"] is None
        assert limits["correlation_based_reduction"] == Decimal("1.0")
        assert limits["warning_level"] == "normal"
    
    @pytest.mark.asyncio
    async def test_position_limits_warning_correlation(self, correlation_monitor):
        """Test position limits with warning level correlation."""
        metrics = CorrelationMetrics(
            average_correlation=Decimal("0.5"),
            max_pairwise_correlation=Decimal("0.7"),  # Above warning threshold (0.6)
            correlation_spike=True,
            correlated_pairs_count=2,
            portfolio_concentration_risk=Decimal("0.4"),
            timestamp=datetime.now(),
            correlation_matrix={}
        )
        
        limits = await correlation_monitor.get_position_limits_for_correlation(metrics)
        
        assert limits["max_positions"] == 3
        assert limits["correlation_based_reduction"] == Decimal("0.6")
        assert limits["warning_level"] == "warning"
    
    @pytest.mark.asyncio
    async def test_position_limits_critical_correlation(self, correlation_monitor):
        """Test position limits with critical level correlation."""
        metrics = CorrelationMetrics(
            average_correlation=Decimal("0.7"),
            max_pairwise_correlation=Decimal("0.9"),  # Above critical threshold (0.8)
            correlation_spike=True,
            correlated_pairs_count=5,
            portfolio_concentration_risk=Decimal("0.8"),
            timestamp=datetime.now(),
            correlation_matrix={}
        )
        
        limits = await correlation_monitor.get_position_limits_for_correlation(metrics)
        
        assert limits["max_positions"] == 1
        assert limits["correlation_based_reduction"] == Decimal("0.3")
        assert limits["warning_level"] == "critical"
    
    @pytest.mark.asyncio
    async def test_correlation_caching(self, correlation_monitor):
        """Test correlation calculation caching."""
        # Add price data for correlation calculation (need enough data points)
        base_time = datetime.now()
        prices = [
            (50000, 3000), (50100, 3030), (49950, 2985), (50200, 3060),
            (49800, 2940), (50300, 3090), (49700, 2910), (50400, 3120),
            (49600, 2880), (50500, 3150), (49500, 2850)
        ]
        
        for i, (btc_price, eth_price) in enumerate(prices):
            timestamp = base_time - timedelta(minutes=len(prices)-i)
            
            await correlation_monitor.update_price_data(MarketData(
                symbol="BTC/USD", price=Decimal(str(btc_price)),
                volume=Decimal("1000.0"), timestamp=timestamp
            ))
            await correlation_monitor.update_price_data(MarketData(
                symbol="ETH/USD", price=Decimal(str(eth_price)),
                volume=Decimal("2000.0"), timestamp=timestamp
            ))
        
        # First call should calculate and cache
        correlation1 = await correlation_monitor.calculate_pairwise_correlation("BTC/USD", "ETH/USD")
        
        # Second call should use cache
        correlation2 = await correlation_monitor.calculate_pairwise_correlation("BTC/USD", "ETH/USD")
        
        assert correlation1 == correlation2
        assert len(correlation_monitor._correlation_cache) > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, correlation_monitor, sample_market_data):
        """Test cleanup of old price data."""
        # Add old data
        old_data = sample_market_data[0]
        await correlation_monitor.update_price_data(old_data)
        
        assert len(correlation_monitor.price_history["BTC/USD"]) == 1
        
        # Cleanup data older than current time (should remove all data)
        cutoff_time = datetime.now()
        await correlation_monitor.cleanup_old_data(cutoff_time)
        
        assert len(correlation_monitor.price_history) == 0
        assert len(correlation_monitor.return_history) == 0
        assert len(correlation_monitor._correlation_cache) == 0
    
    @pytest.mark.asyncio
    async def test_thread_safety(self, correlation_monitor):
        """Test thread safety of correlation monitor."""
        base_time = datetime.now()
        
        async def update_prices(symbol_suffix):
            for i in range(10):
                market_data = MarketData(
                    symbol=f"TEST{symbol_suffix}/USD",
                    price=Decimal(str(1000 + i)),
                    volume=Decimal("100.0"),
                    timestamp=base_time - timedelta(minutes=10-i)
                )
                await correlation_monitor.update_price_data(market_data)
        
        # Run concurrent updates
        await asyncio.gather(
            update_prices("1"),
            update_prices("2"), 
            update_prices("3"),
        )
        
        # Verify all data was stored correctly
        assert len(correlation_monitor.price_history) == 3
        for symbol in ["TEST1/USD", "TEST2/USD", "TEST3/USD"]:
            assert len(correlation_monitor.price_history[symbol]) == 10
            assert len(correlation_monitor.return_history[symbol]) == 9
    
    def test_get_status(self, correlation_monitor):
        """Test status reporting."""
        status = correlation_monitor.get_status()
        
        assert "monitored_symbols" in status
        assert "cache_size" in status
        assert "thresholds" in status
        assert "data_points" in status
        
        assert status["monitored_symbols"] == 0
        assert status["cache_size"] == 0
        assert status["thresholds"]["warning"] == "0.6"
        assert status["thresholds"]["critical"] == "0.8"


class TestCorrelationDecimalPrecision:
    """Test Decimal precision in correlation calculations."""
    
    @pytest.mark.asyncio
    async def test_correlation_decimal_precision(self, correlation_monitor):
        """Test that correlation calculations maintain Decimal precision."""
        # Add precise price data
        base_time = datetime.now()
        
        # Create precise price movements (need at least 10 data points for default min_periods)
        btc_prices = [
            Decimal("50000.123456789"),
            Decimal("50001.987654321"), 
            Decimal("49999.111111111"),
            Decimal("50002.222222222"),
            Decimal("49998.333333333"),
            Decimal("50003.444444444"),
            Decimal("49997.555555555"),
            Decimal("50004.666666666"),
            Decimal("49996.777777777"),
            Decimal("50005.888888888"),
            Decimal("49995.999999999")
        ]
        
        eth_prices = [
            Decimal("3000.987654321"),
            Decimal("3001.123456789"),
            Decimal("2999.555555555"),
            Decimal("3002.444444444"),
            Decimal("2998.777777777"),
            Decimal("3003.111111111"),
            Decimal("2997.222222222"),
            Decimal("3004.333333333"),
            Decimal("2996.444444444"),
            Decimal("3005.555555555"),
            Decimal("2995.666666666")
        ]
        
        for i, (btc_price, eth_price) in enumerate(zip(btc_prices, eth_prices)):
            timestamp = base_time - timedelta(minutes=10-i)
            
            await correlation_monitor.update_price_data(MarketData(
                symbol="BTC/USD", price=btc_price,
                volume=Decimal("1000.0"), timestamp=timestamp
            ))
            await correlation_monitor.update_price_data(MarketData(
                symbol="ETH/USD", price=eth_price,
                volume=Decimal("2000.0"), timestamp=timestamp
            ))
        
        correlation = await correlation_monitor.calculate_pairwise_correlation("BTC/USD", "ETH/USD")
        
        assert correlation is not None
        assert isinstance(correlation, Decimal)
        assert Decimal("-1.0") <= correlation <= Decimal("1.0")
    
    @pytest.mark.asyncio
    async def test_portfolio_metrics_decimal_precision(self, correlation_monitor):
        """Test that portfolio correlation metrics maintain Decimal precision."""
        # Add price data first
        base_time = datetime.now()
        for i in range(10):
            timestamp = base_time - timedelta(minutes=10-i)
            
            await correlation_monitor.update_price_data(MarketData(
                symbol="BTC/USD", price=Decimal(f"50{i:03d}.123"),
                volume=Decimal("1000.0"), timestamp=timestamp
            ))
            await correlation_monitor.update_price_data(MarketData(
                symbol="ETH/USD", price=Decimal(f"30{i:02d}.456"),
                volume=Decimal("2000.0"), timestamp=timestamp
            ))
        
        positions = [
            Position(
                symbol="BTC/USD", 
                quantity=Decimal("1.123456789"),
                entry_price=Decimal("50000.123456789"),
                current_price=Decimal("50005.123456789"),
                unrealized_pnl=Decimal("5.0"), 
                side=OrderSide.BUY,
                timestamp=datetime.now()
            ),
            Position(
                symbol="ETH/USD", 
                quantity=Decimal("10.987654321"),
                entry_price=Decimal("3000.987654321"),
                current_price=Decimal("3005.987654321"),
                unrealized_pnl=Decimal("55.0"), 
                side=OrderSide.BUY,
                timestamp=datetime.now()
            ),
        ]
        
        metrics = await correlation_monitor.calculate_portfolio_correlation(positions)
        
        assert isinstance(metrics.average_correlation, Decimal)
        assert isinstance(metrics.max_pairwise_correlation, Decimal)
        assert isinstance(metrics.portfolio_concentration_risk, Decimal)
        
        # Verify all correlation matrix values are Decimal
        for correlation in metrics.correlation_matrix.values():
            assert isinstance(correlation, Decimal)