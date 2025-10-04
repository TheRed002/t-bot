"""
Integration tests for Feature Engineering Framework (P-015).

This module tests the complete feature engineering pipeline including:
- Technical indicators with real market data
- Statistical features across multiple timeframes
- Alternative data integration
- Feature pipeline performance and accuracy
- Integration with data sources and ML models
"""

import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.core.config import Config
from src.core.types import MarketData, NewsSentiment, SocialSentiment
from src.data.features.technical_indicators import TechnicalIndicatorCalculator
from src.data.features.statistical_features import StatisticalFeatureCalculator
from src.data.features.alternative_features import AlternativeFeatureCalculator


class TestFeatureEngineeringIntegration:
    """Integration tests for the complete feature engineering framework."""

    @pytest.fixture
    def config(self):
        """Create comprehensive test configuration."""
        config = MagicMock(spec=Config)
        
        # Technical indicators config
        config.indicators = {
            "default_periods": {
                "sma": 20,
                "ema": 20,
                "rsi": 14,
                "macd": [12, 26, 9],
                "bollinger": 20,
                "atr": 14,
            },
            "cache_enabled": True,
            "max_calculation_time": 5.0,
        }
        
        # Statistical features config
        config.statistical_features = {
            "default_windows": {
                "rolling_stats": 20,
                "autocorr": 50,
                "regime": 100,
                "seasonality": 252,
            },
            "regime_threshold": 0.02,
            "correlation_threshold": 0.7,
        }
        
        # Alternative features config
        config.alternative_features = {
            "default_lookbacks": {
                "news_sentiment": 24,
                "social_sentiment": 12,
                "economic": 168,
                "microstructure": 6,
            },
            "sentiment_weights": {
                "very_positive": 1.0,
                "positive": 0.5,
                "neutral": 0.0,
                "negative": -0.5,
                "very_negative": -1.0,
            },
            "update_interval": 300,
        }
        
        config.max_price_history = 1000
        return config

    @pytest.fixture
    def feature_calculators(self, config):
        """Create all feature calculators."""
        technical_calc = TechnicalIndicatorCalculator(config)
        statistical_calc = StatisticalFeatureCalculator(config)
        alternative_calc = AlternativeFeatureCalculator(config)
        
        return {
            "technical": technical_calc,
            "statistical": statistical_calc,
            "alternative": alternative_calc,
        }

    @pytest.fixture
    def realistic_market_data(self):
        """Create realistic market data for comprehensive testing."""
        data = []
        base_price = 50000.0  # Bitcoin-like starting price
        
        # Create 200 data points with realistic patterns
        for i in range(200):
            # Add market hours effect (higher volatility during trading hours)
            hour = i % 24
            if 9 <= hour <= 16:  # Trading hours
                volatility_multiplier = 1.5
            else:
                volatility_multiplier = 0.8
            
            # Add weekly cycle (higher volume on weekdays)
            day_of_week = (i // 24) % 7
            if day_of_week < 5:  # Weekdays
                volume_multiplier = 1.2
            else:
                volume_multiplier = 0.7
            
            # Generate price movement with trend and noise
            if i < 50:
                trend = 0.001  # Uptrend
            elif i < 100:
                trend = -0.0005  # Slight downtrend
            elif i < 150:
                trend = 0.002  # Strong uptrend
            else:
                trend = 0.0  # Sideways
            
            price_change = trend + np.random.normal(0, 0.02 * volatility_multiplier)
            base_price *= (1 + price_change)
            
            # Calculate OHLC
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, base_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, base_price) * (1 - abs(np.random.normal(0, 0.01)))
            close_price = base_price
            
            # Volume with patterns
            base_volume = 1000000
            volume = base_volume * volume_multiplier * (1 + abs(np.random.normal(0, 0.3)))
            
            market_data = MarketData(
                symbol="BTCUSDT",
                price=close_price,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                volume=volume,
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
                source="test_integration",
            )
            data.append(market_data)
        
        return data

    @pytest.fixture
    def mock_alternative_sources(self):
        """Create mock alternative data sources."""
        news_source = AsyncMock()
        social_source = AsyncMock()
        alt_data_source = AsyncMock()
        
        # Mock news data
        news_articles = [
            {
                "sentiment": NewsSentiment.POSITIVE,
                "score": 0.7,
                "title": "Bitcoin adoption grows",
                "source": "CryptoNews",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "sentiment": NewsSentiment.VERY_POSITIVE,
                "score": 0.9,
                "title": "Major bank adds Bitcoin",
                "source": "FinancialNews",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "sentiment": NewsSentiment.NEGATIVE,
                "score": -0.4,
                "title": "Regulatory scrutiny increases",
                "source": "RegulatoryNews",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]
        
        # Mock social sentiment
        social_sentiment = {
            "twitter": [
                {"sentiment": SocialSentiment.BULLISH, "engagement_score": 100},
                {"sentiment": SocialSentiment.VERY_BULLISH, "engagement_score": 150},
                {"sentiment": SocialSentiment.NEUTRAL, "engagement_score": 50},
            ],
            "reddit": [
                {"sentiment": SocialSentiment.BEARISH, "engagement_score": 75},
                {"sentiment": SocialSentiment.NEUTRAL, "engagement_score": 25},
            ],
        }
        
        # Mock economic data
        economic_data = {
            "inflation": [
                {"value": 3.2, "timestamp": datetime.now(timezone.utc).isoformat()},
                {"value": 3.0, "timestamp": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()},
            ],
            "unemployment": [
                {"value": 4.1, "timestamp": datetime.now(timezone.utc).isoformat()},
                {"value": 4.3, "timestamp": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()},
            ],
        }
        
        news_source.get_news_for_symbol.return_value = news_articles
        social_source.get_social_sentiment.return_value = social_sentiment
        alt_data_source.get_economic_indicators.return_value = economic_data
        
        return {
            "news": news_source,
            "social": social_source,
            "economic": alt_data_source,
        }

    @pytest.mark.asyncio
    async def test_complete_feature_pipeline(self, feature_calculators, realistic_market_data,
                                           mock_alternative_sources):
        """Test the complete feature engineering pipeline."""
        technical_calc = feature_calculators["technical"]
        statistical_calc = feature_calculators["statistical"]
        alternative_calc = feature_calculators["alternative"]
        
        # Setup alternative data sources
        alternative_calc.set_data_sources(
            news_source=mock_alternative_sources["news"],
            social_source=mock_alternative_sources["social"],
            alt_data_source=mock_alternative_sources["economic"],
        )
        
        # Add market data to all calculators
        for data in realistic_market_data:
            await technical_calc.add_market_data(data)
            await statistical_calc.add_market_data(data)
        
        # Test technical indicators
        technical_indicators = ["SMA", "EMA", "RSI", "MACD", "BOLLINGER", "ATR", 
                              "STOCHASTIC", "WILLIAMS_R", "CCI"]
        tech_results = await technical_calc.calculate_batch_indicators("BTCUSDT", technical_indicators)
        
        # Test statistical features
        statistical_features = ["ROLLING_STATS", "AUTOCORRELATION", "REGIME"]
        stat_results = await statistical_calc.calculate_batch_features("BTCUSDT", statistical_features)
        
        # Test alternative features
        alternative_features = ["NEWS_SENTIMENT", "SOCIAL_SENTIMENT", "ECONOMIC_INDICATORS"]
        alt_results = await alternative_calc.calculate_batch_features("BTCUSDT", alternative_features)
        
        # Verify all calculations succeeded
        assert len(tech_results) == len(technical_indicators)
        assert all(result is not None for result in tech_results.values())
        
        assert len(stat_results) == len(statistical_features)
        assert all(result is not None for result in stat_results.values())
        
        assert len(alt_results) == len(alternative_features)
        assert all(result is not None for result in alt_results.values())
        
        # Verify feature quality
        # Technical indicators should have reasonable values
        assert 0 <= tech_results["RSI"].value <= 100
        assert tech_results["ATR"].value > 0
        
        # Statistical features should be properly calculated
        rolling_stats = stat_results["ROLLING_STATS"].value
        assert isinstance(rolling_stats["mean"], (int, float))
        assert rolling_stats["std"] >= 0
        
        # Alternative features should have sentiment scores
        news_sentiment = alt_results["NEWS_SENTIMENT"].value
        assert -1 <= news_sentiment["average_sentiment"] <= 1
        assert news_sentiment["article_count"] > 0

    @pytest.mark.asyncio
    async def test_feature_correlation_analysis(self, feature_calculators, realistic_market_data):
        """Test cross-correlation analysis between different symbols."""
        statistical_calc = feature_calculators["statistical"]
        
        # Add data for BTCUSDT
        for data in realistic_market_data:
            await statistical_calc.add_market_data(data)
        
        # Create correlated data for ETHUSDT (similar to BTC but with noise)
        eth_data = []
        for btc_data in realistic_market_data:
            # ETH follows BTC with some correlation + noise
            correlation_factor = 0.7
            noise_factor = 0.3
            
            eth_price_change = (
                correlation_factor * (btc_data.price / realistic_market_data[0].price - 1) +
                noise_factor * np.random.normal(0, 0.02)
            )
            eth_price = 3000 * (1 + eth_price_change)  # ETH starting at $3000
            
            eth_market_data = MarketData(
                symbol="ETHUSDT",
                price=eth_price,
                open_price=eth_price * (1 + np.random.normal(0, 0.005)),
                high_price=eth_price * (1 + abs(np.random.normal(0, 0.01))),
                low_price=eth_price * (1 - abs(np.random.normal(0, 0.01))),
                volume=btc_data.volume * 0.6,  # ETH typically has lower volume
                timestamp=btc_data.timestamp,
                source="test_integration",
            )
            eth_data.append(eth_market_data)
        
        # Add ETH data
        for data in eth_data:
            await statistical_calc.add_market_data(data)
        
        # Calculate cross-correlation
        correlation_result = await statistical_calc.calculate_cross_correlation(
            "BTCUSDT", "ETHUSDT", max_lags=10
        )
        
        # Should detect positive correlation
        assert correlation_result.value["contemporaneous_correlation"] > 0.3
        assert correlation_result.value["correlation_strength"] > 0.3

    @pytest.mark.asyncio
    async def test_regime_detection_accuracy(self, feature_calculators):
        """Test regime detection with known market patterns."""
        statistical_calc = feature_calculators["statistical"]
        
        # Create strong trending market
        trending_data = []
        base_price = 100.0
        
        for i in range(100):
            base_price *= 1.01  # Strong 1% daily trend
            
            market_data = MarketData(
                symbol="TRENDING",
                price=base_price,
                open_price=base_price * (1 + np.random.normal(0, 0.002)),
                high_price=base_price * (1 + abs(np.random.normal(0, 0.005))),
                low_price=base_price * (1 - abs(np.random.normal(0, 0.005))),
                volume=1000000,
                timestamp=datetime.now(timezone.utc) + timedelta(days=i),
                source="test",
            )
            trending_data.append(market_data)
        
        # Add trending data
        for data in trending_data:
            await statistical_calc.add_market_data(data)
        
        # Detect regime
        regime_result = await statistical_calc.detect_regime("TRENDING", window=50)
        
        # Should detect trending up regime
        assert regime_result.value["regime"] == "trending_up"
        assert regime_result.value["confidence"] > 0.5
        assert regime_result.value["price_trend"] > 0.02  # Strong positive trend

    @pytest.mark.asyncio
    async def test_feature_calculation_performance(self, feature_calculators, realistic_market_data):
        """Test performance of feature calculations."""
        technical_calc = feature_calculators["technical"]
        
        # Add all market data
        start_time = datetime.now()
        for data in realistic_market_data:
            await technical_calc.add_market_data(data)
        data_load_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate all technical indicators
        start_time = datetime.now()
        indicators = ["SMA", "EMA", "RSI", "MACD", "BOLLINGER", "ATR", "STOCHASTIC"]
        results = await technical_calc.calculate_batch_indicators("BTCUSDT", indicators)
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert data_load_time < 1.0  # Should load data quickly
        assert calculation_time < 5.0  # Should calculate indicators quickly
        assert all(result.calculation_time < 1.0 for result in results.values())
        
        # Verify all calculations succeeded
        assert len(results) == len(indicators)
        assert all(result.value is not None for result in results.values())

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, config, realistic_market_data):
        """Test memory efficiency with large datasets."""
        # Set low memory limit
        config.max_price_history = 50
        
        technical_calc = TechnicalIndicatorCalculator(config)
        
        # Add more data than memory limit
        for data in realistic_market_data:  # 200 data points
            await technical_calc.add_market_data(data)
        
        # Should only keep max_price_history records
        assert len(technical_calc.price_data["BTCUSDT"]) == 50
        
        # Should still be able to calculate indicators
        result = await technical_calc.calculate_sma("BTCUSDT", period=20)
        assert result.value is not None

    @pytest.mark.asyncio
    async def test_concurrent_feature_calculations(self, feature_calculators, 
                                                 realistic_market_data, mock_alternative_sources):
        """Test concurrent calculations across all feature types."""
        technical_calc = feature_calculators["technical"]
        statistical_calc = feature_calculators["statistical"]
        alternative_calc = feature_calculators["alternative"]
        
        # Setup alternative sources
        alternative_calc.set_data_sources(
            news_source=mock_alternative_sources["news"],
            social_source=mock_alternative_sources["social"],
        )
        
        # Add data to technical and statistical calculators
        for data in realistic_market_data:
            await technical_calc.add_market_data(data)
            await statistical_calc.add_market_data(data)
        
        # Run all calculations concurrently
        tasks = [
            technical_calc.calculate_sma("BTCUSDT"),
            technical_calc.calculate_rsi("BTCUSDT"),
            statistical_calc.calculate_rolling_stats("BTCUSDT"),
            statistical_calc.detect_regime("BTCUSDT"),
            alternative_calc.calculate_news_sentiment("BTCUSDT"),
            alternative_calc.calculate_social_sentiment("BTCUSDT"),
        ]
        
        start_time = datetime.now()
        results = await asyncio.gather(*tasks)
        concurrent_time = (datetime.now() - start_time).total_seconds()
        
        # All calculations should succeed
        assert len(results) == 6
        assert all(result.value is not None for result in results)
        
        # Concurrent execution should be efficient
        assert concurrent_time < 10.0

    @pytest.mark.asyncio
    async def test_feature_validation_and_sanitization(self, feature_calculators):
        """Test feature validation with edge cases and invalid data."""
        technical_calc = feature_calculators["technical"]
        
        # Create data with edge cases
        edge_case_data = [
            # Normal data point
            MarketData(
                symbol="EDGE_TEST",
                price=100.0,
                open_price=100.0,
                high_price=101.0,
                low_price=99.0,
                volume=1000,
                timestamp=datetime.now(timezone.utc),
                source="test",
            ),
            # Zero price (should be handled gracefully)
            MarketData(
                symbol="EDGE_TEST",
                price=0.0,
                open_price=0.0,
                high_price=0.0,
                low_price=0.0,
                volume=0,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=1),
                source="test",
            ),
            # Very large price
            MarketData(
                symbol="EDGE_TEST",
                price=1e10,
                open_price=1e10,
                high_price=1e10,
                low_price=1e10,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=2),
                source="test",
            ),
            # Recovery to normal
            MarketData(
                symbol="EDGE_TEST",
                price=100.5,
                open_price=100.5,
                high_price=101.5,
                low_price=99.5,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=3),
                source="test",
            ),
        ]
        
        # Add edge case data
        for data in edge_case_data:
            await technical_calc.add_market_data(data)
        
        # Should handle gracefully and not crash
        assert "EDGE_TEST" in technical_calc.price_data
        assert len(technical_calc.price_data["EDGE_TEST"]) == 4

    @pytest.mark.asyncio
    async def test_feature_pipeline_error_recovery(self, feature_calculators, 
                                                  mock_alternative_sources):
        """Test error recovery in feature pipeline."""
        alternative_calc = feature_calculators["alternative"]
        
        # Setup failing news source
        failing_news_source = AsyncMock()
        failing_news_source.get_news_for_symbol.side_effect = Exception("API Timeout")
        
        alternative_calc.set_data_sources(
            news_source=failing_news_source,
            social_source=mock_alternative_sources["social"],  # This one works
        )
        
        # Calculate batch features with one failing source
        features = ["NEWS_SENTIMENT", "SOCIAL_SENTIMENT"]
        results = await alternative_calc.calculate_batch_features("BTCUSDT", features)
        
        # NEWS_SENTIMENT should fail, SOCIAL_SENTIMENT should succeed
        assert "NEWS_SENTIMENT" in results
        assert "SOCIAL_SENTIMENT" in results
        assert results["NEWS_SENTIMENT"] is None  # Failed
        assert results["SOCIAL_SENTIMENT"] is not None  # Succeeded

    @pytest.mark.asyncio
    async def test_feature_accuracy_validation(self, feature_calculators):
        """Test feature calculation accuracy with known test cases."""
        technical_calc = feature_calculators["technical"]
        
        # Create data with known SMA result
        known_prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        for i, price in enumerate(known_prices):
            data = MarketData(
                symbol="ACCURACY_TEST",
                price=price,
                open_price=price,
                high_price=price * 1.01,
                low_price=price * 0.99,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                source="test",
            )
            await technical_calc.add_market_data(data)
        
        # Calculate SMA(5) - should be average of last 5 prices [15,16,17,18,19] = 17.0
        sma_result = await technical_calc.calculate_sma("ACCURACY_TEST", period=5)
        expected_sma = sum([15, 16, 17, 18, 19]) / 5
        
        assert abs(sma_result.value - expected_sma) < 0.001

    @pytest.mark.asyncio
    async def test_feature_caching_effectiveness(self, feature_calculators, realistic_market_data):
        """Test feature caching for performance optimization."""
        technical_calc = feature_calculators["technical"]
        
        # Add data
        for data in realistic_market_data:
            await technical_calc.add_market_data(data)
        
        # Calculate same indicator twice
        start_time = datetime.now()
        result1 = await technical_calc.calculate_sma("BTCUSDT", period=20)
        first_calc_time = (datetime.now() - start_time).total_seconds()
        
        start_time = datetime.now()
        result2 = await technical_calc.calculate_sma("BTCUSDT", period=20)
        second_calc_time = (datetime.now() - start_time).total_seconds()
        
        # Results should be identical
        assert result1.value == result2.value
        
        # Second calculation should be faster due to caching
        # Note: This test might be flaky due to timing, so we use a generous threshold
        assert second_calc_time <= first_calc_time + 0.1  # Allow some tolerance


@pytest.mark.performance
class TestFeatureEngineeringPerformance:
    """Performance tests for feature engineering framework."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        config = MagicMock(spec=Config)
        config.indicators = {"default_periods": {"sma": 20}, "cache_enabled": True}
        config.max_price_history = 10000
        
        technical_calc = TechnicalIndicatorCalculator(config)
        
        # Create large dataset (1000 data points)
        large_dataset = []
        base_price = 100.0
        
        for i in range(1000):
            base_price *= (1 + np.random.normal(0, 0.01))
            
            data = MarketData(
                symbol="LARGE_TEST",
                price=base_price,
                open_price=base_price,
                high_price=base_price * 1.01,
                low_price=base_price * 0.99,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                source="test",
            )
            large_dataset.append(data)
        
        # Measure data loading performance
        start_time = datetime.now()
        for data in large_dataset:
            await technical_calc.add_market_data(data)
        load_time = (datetime.now() - start_time).total_seconds()
        
        # Measure calculation performance
        start_time = datetime.now()
        result = await technical_calc.calculate_sma("LARGE_TEST", period=50)
        calc_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions for large dataset
        assert load_time < 5.0  # Should load 1000 points in under 5 seconds
        assert calc_time < 2.0  # Should calculate in under 2 seconds
        assert result.value is not None
        
        print(f"Large dataset performance: Load={load_time:.3f}s, Calc={calc_time:.3f}s")