"""
Unit tests for AlternativeFeatureCalculator.

This module tests alternative data feature calculations including:
- News sentiment analysis, social media sentiment
- Economic indicators, market microstructure features
- Data source integration and error handling
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.core.config import Config
from src.core.exceptions import DataError
from src.core.types import NewsSentiment, SocialSentiment
from src.data.features.alternative_features import (
    AlternativeFeatureCalculator,
    AlternativeResult,
    AlternativeFeatureType,
    SentimentStrength,
    AlternativeConfig,
)


class TestAlternativeFeatureCalculator:
    """Test suite for AlternativeFeatureCalculator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock(spec=Config)
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
        return config

    @pytest.fixture
    def calculator(self, config):
        """Create test alternative feature calculator."""
        return AlternativeFeatureCalculator(config)

    @pytest.fixture
    def mock_news_source(self):
        """Create mock news data source."""
        news_source = AsyncMock()
        news_source.get_news_for_symbol = AsyncMock()
        return news_source

    @pytest.fixture
    def mock_social_source(self):
        """Create mock social media data source."""
        social_source = AsyncMock()
        social_source.get_social_sentiment = AsyncMock()
        return social_source

    @pytest.fixture
    def mock_alt_data_source(self):
        """Create mock alternative data source."""
        alt_data_source = AsyncMock()
        alt_data_source.get_economic_indicators = AsyncMock()
        return alt_data_source

    @pytest.fixture
    def sample_news_articles(self):
        """Create sample news articles."""
        return [
            {
                "sentiment": NewsSentiment.POSITIVE,
                "score": 0.7,
                "title": "Bitcoin shows strong momentum",
                "source": "CryptoNews",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "sentiment": NewsSentiment.VERY_POSITIVE,
                "score": 0.9,
                "title": "Major institutional adoption of Bitcoin",
                "source": "FinancialTimes",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "sentiment": NewsSentiment.NEGATIVE,
                "score": -0.6,
                "title": "Regulatory concerns for cryptocurrency",
                "source": "RegulatoryNews",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "sentiment": NewsSentiment.NEUTRAL,
                "score": 0.0,
                "title": "Bitcoin price analysis",
                "source": "TechnicalAnalysis",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]

    @pytest.fixture
    def sample_social_sentiment(self):
        """Create sample social media sentiment."""
        return {
            "twitter": [
                {
                    "sentiment": SocialSentiment.VERY_BULLISH,
                    "engagement_score": 100,
                    "text": "Bitcoin to the moon!",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "sentiment": SocialSentiment.BULLISH,
                    "engagement_score": 50,
                    "text": "Bitcoin looking good",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ],
            "reddit": [
                {
                    "sentiment": SocialSentiment.BEARISH,
                    "engagement_score": 75,
                    "text": "Bitcoin might drop",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "sentiment": SocialSentiment.NEUTRAL,
                    "engagement_score": 25,
                    "text": "Bitcoin price discussion",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ],
        }

    @pytest.fixture
    def sample_economic_data(self):
        """Create sample economic indicator data."""
        return {
            "inflation": [
                {
                    "value": 3.2,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "value": 3.1,
                    "timestamp": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                },
            ],
            "unemployment": [
                {
                    "value": 4.1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "value": 4.3,
                    "timestamp": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                },
            ],
            "gdp_growth": [
                {
                    "value": 2.8,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "value": 2.5,
                    "timestamp": (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_calculator_initialization(self, config):
        """Test calculator initialization."""
        calculator = AlternativeFeatureCalculator(config)
        
        assert calculator.config == config
        assert calculator.default_lookbacks["news_sentiment"] == 24
        assert calculator.sentiment_weights["positive"] == 0.5
        assert calculator.news_source is None
        assert calculator.social_source is None
        assert calculator.alt_data_source is None

    @pytest.mark.asyncio
    async def test_set_data_sources(self, calculator, mock_news_source, 
                                   mock_social_source, mock_alt_data_source):
        """Test setting data sources."""
        calculator.set_data_sources(
            news_source=mock_news_source,
            social_source=mock_social_source,
            alt_data_source=mock_alt_data_source
        )
        
        assert calculator.news_source == mock_news_source
        assert calculator.social_source == mock_social_source
        assert calculator.alt_data_source == mock_alt_data_source

    @pytest.mark.asyncio
    async def test_initialize_data_sources(self, calculator, mock_news_source):
        """Test data source initialization."""
        calculator.set_data_sources(news_source=mock_news_source)
        
        await calculator.initialize()
        
        mock_news_source.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_news_sentiment_calculation(self, calculator, mock_news_source, 
                                            sample_news_articles):
        """Test news sentiment calculation."""
        # Setup mock
        mock_news_source.get_news_for_symbol.return_value = sample_news_articles
        calculator.set_data_sources(news_source=mock_news_source)
        
        # Calculate news sentiment
        result = await calculator.calculate_news_sentiment("BTCUSDT", lookback_hours=24)
        
        assert isinstance(result, AlternativeResult)
        assert result.feature_name == "NEWS_SENTIMENT"
        assert result.symbol == "BTCUSDT"
        assert isinstance(result.value, dict)
        
        # Check sentiment metrics
        required_fields = ["average_sentiment", "weighted_sentiment", "sentiment_std",
                          "sentiment_trend", "sentiment_strength", "article_count"]
        assert all(field in result.value for field in required_fields)
        
        # Should have processed 4 articles
        assert result.value["article_count"] == 4
        
        # Sentiment should be mixed (positive and negative articles)
        assert -1 <= result.value["average_sentiment"] <= 1

    @pytest.mark.asyncio
    async def test_news_sentiment_no_source(self, calculator):
        """Test news sentiment without data source."""
        result = await calculator.calculate_news_sentiment("BTCUSDT")
        
        assert result.value is None
        assert result.metadata["reason"] == "news_source_not_available"

    @pytest.mark.asyncio
    async def test_news_sentiment_no_articles(self, calculator, mock_news_source):
        """Test news sentiment with no articles."""
        mock_news_source.get_news_for_symbol.return_value = []
        calculator.set_data_sources(news_source=mock_news_source)
        
        result = await calculator.calculate_news_sentiment("BTCUSDT")
        
        assert result.value is None
        assert result.metadata["reason"] == "no_news_data"

    @pytest.mark.asyncio
    async def test_social_sentiment_calculation(self, calculator, mock_social_source,
                                              sample_social_sentiment):
        """Test social media sentiment calculation."""
        # Setup mock
        mock_social_source.get_social_sentiment.return_value = sample_social_sentiment
        calculator.set_data_sources(social_source=mock_social_source)
        
        # Calculate social sentiment
        result = await calculator.calculate_social_sentiment("BTCUSDT", lookback_hours=12)
        
        assert isinstance(result, AlternativeResult)
        assert result.feature_name == "SOCIAL_SENTIMENT"
        assert isinstance(result.value, dict)
        
        # Check social sentiment metrics
        required_fields = ["average_sentiment", "volume_weighted_sentiment", 
                          "sentiment_momentum", "total_posts", "platform_sentiments"]
        assert all(field in result.value for field in required_fields)
        
        # Should have processed posts from both platforms
        assert "twitter" in result.value["platform_sentiments"]
        assert "reddit" in result.value["platform_sentiments"]
        
        # Total posts should be 4 (2 from each platform)
        assert result.value["total_posts"] == 4

    @pytest.mark.asyncio
    async def test_social_sentiment_no_source(self, calculator):
        """Test social sentiment without data source."""
        result = await calculator.calculate_social_sentiment("BTCUSDT")
        
        assert result.value is None
        assert result.metadata["reason"] == "social_source_not_available"

    @pytest.mark.asyncio
    async def test_economic_indicators_calculation(self, calculator, mock_alt_data_source,
                                                 sample_economic_data):
        """Test economic indicators calculation."""
        # Setup mock
        mock_alt_data_source.get_economic_indicators.return_value = sample_economic_data
        calculator.set_data_sources(alt_data_source=mock_alt_data_source)
        
        # Calculate economic indicators
        result = await calculator.calculate_economic_indicators("BTCUSDT", lookback_hours=168)
        
        assert isinstance(result, AlternativeResult)
        assert result.feature_name == "ECONOMIC_INDICATORS"
        assert isinstance(result.value, dict)
        
        # Check economic indicator metrics
        required_fields = ["economic_score", "indicator_values", "indicator_changes"]
        assert all(field in result.value for field in required_fields)
        
        # Should have calculated changes for each indicator
        assert "inflation_change" in result.value["indicator_changes"]
        assert "unemployment_change" in result.value["indicator_changes"]
        
        # Economic score should be a number
        assert isinstance(result.value["economic_score"], (int, float))

    @pytest.mark.asyncio
    async def test_economic_indicators_no_source(self, calculator):
        """Test economic indicators without data source."""
        result = await calculator.calculate_economic_indicators("BTCUSDT")
        
        assert result.value is None
        assert result.metadata["reason"] == "alt_data_source_not_available"

    @pytest.mark.asyncio
    async def test_market_microstructure_calculation(self, calculator):
        """Test market microstructure calculation (placeholder implementation)."""
        result = await calculator.calculate_market_microstructure("BTCUSDT")
        
        assert isinstance(result, AlternativeResult)
        assert result.feature_name == "MARKET_MICROSTRUCTURE"
        assert isinstance(result.value, dict)
        
        # Should have placeholder fields
        microstructure_fields = ["bid_ask_spread_avg", "order_flow_imbalance",
                               "trade_size_distribution", "market_impact",
                               "liquidity_score", "price_discovery_efficiency"]
        assert all(field in result.value for field in microstructure_fields)
        
        # All values should be None in placeholder implementation
        assert all(result.value[field] is None for field in microstructure_fields)

    @pytest.mark.asyncio
    async def test_batch_features_calculation(self, calculator, mock_news_source,
                                            mock_social_source, sample_news_articles,
                                            sample_social_sentiment):
        """Test batch alternative features calculation."""
        # Setup mocks
        mock_news_source.get_news_for_symbol.return_value = sample_news_articles
        mock_social_source.get_social_sentiment.return_value = sample_social_sentiment
        calculator.set_data_sources(
            news_source=mock_news_source,
            social_source=mock_social_source
        )
        
        # Calculate multiple features
        features = ["NEWS_SENTIMENT", "SOCIAL_SENTIMENT", "MARKET_MICROSTRUCTURE"]
        results = await calculator.calculate_batch_features("BTCUSDT", features)
        
        assert isinstance(results, dict)
        assert len(results) == len(features)
        
        # Check that all features were calculated
        for feature in features:
            assert feature in results
            assert results[feature] is not None
            assert isinstance(results[feature], AlternativeResult)

    @pytest.mark.asyncio
    async def test_sentiment_trend_calculation(self, calculator):
        """Test sentiment trend calculation helper method."""
        # Test with trending sentiment
        sentiment_scores = [0.1, 0.2, 0.3, 0.4, 0.5]  # Increasing trend
        trend = calculator._calculate_sentiment_trend(sentiment_scores)
        
        assert trend > 0  # Should be positive trend
        
        # Test with decreasing trend
        sentiment_scores = [0.5, 0.4, 0.3, 0.2, 0.1]  # Decreasing trend
        trend = calculator._calculate_sentiment_trend(sentiment_scores)
        
        assert trend < 0  # Should be negative trend
        
        # Test with insufficient data
        sentiment_scores = [0.1, 0.2]  # Only 2 points
        trend = calculator._calculate_sentiment_trend(sentiment_scores)
        
        assert trend == 0.0  # Should return 0 for insufficient data

    @pytest.mark.asyncio
    async def test_sentiment_strength_determination(self, calculator):
        """Test sentiment strength determination helper method."""
        # Test very strong sentiment
        strength = calculator._determine_sentiment_strength(0.8, 0.1)  # High sentiment, low std
        assert strength == SentimentStrength.VERY_STRONG
        
        # Test strong sentiment
        strength = calculator._determine_sentiment_strength(0.6, 0.2)  # Good sentiment, low std
        assert strength == SentimentStrength.STRONG
        
        # Test moderate sentiment
        strength = calculator._determine_sentiment_strength(0.4, 0.3)  # Moderate sentiment
        assert strength == SentimentStrength.MODERATE
        
        # Test weak sentiment
        strength = calculator._determine_sentiment_strength(0.2, 0.2)  # Weak sentiment
        assert strength == SentimentStrength.WEAK
        
        # Test neutral sentiment
        strength = calculator._determine_sentiment_strength(0.05, 0.1)  # Very low sentiment
        assert strength == SentimentStrength.NEUTRAL

    @pytest.mark.asyncio
    async def test_calculation_summary(self, calculator, mock_news_source,
                                     sample_news_articles):
        """Test calculation statistics summary."""
        # Setup and perform calculations
        mock_news_source.get_news_for_symbol.return_value = sample_news_articles
        calculator.set_data_sources(news_source=mock_news_source)
        
        await calculator.calculate_news_sentiment("BTCUSDT")
        await calculator.calculate_market_microstructure("BTCUSDT")
        
        # Get summary
        summary = await calculator.get_calculation_summary()
        
        assert isinstance(summary, dict)
        assert "statistics" in summary
        assert "success_rate" in summary
        assert "data_sources_available" in summary
        assert summary["statistics"]["successful_calculations"] >= 2

    @pytest.mark.asyncio
    async def test_error_handling_news_sentiment(self, calculator, mock_news_source):
        """Test error handling in news sentiment calculation."""
        # Setup mock to raise exception
        mock_news_source.get_news_for_symbol.side_effect = Exception("API Error")
        calculator.set_data_sources(news_source=mock_news_source)
        
        with pytest.raises(DataError, match="News sentiment calculation failed"):
            await calculator.calculate_news_sentiment("BTCUSDT")

    @pytest.mark.asyncio
    async def test_error_handling_social_sentiment(self, calculator, mock_social_source):
        """Test error handling in social sentiment calculation."""
        # Setup mock to raise exception
        mock_social_source.get_social_sentiment.side_effect = Exception("API Error")
        calculator.set_data_sources(social_source=mock_social_source)
        
        with pytest.raises(DataError, match="Social sentiment calculation failed"):
            await calculator.calculate_social_sentiment("BTCUSDT")

    @pytest.mark.asyncio
    async def test_error_handling_economic_indicators(self, calculator, mock_alt_data_source):
        """Test error handling in economic indicators calculation."""
        # Setup mock to raise exception
        mock_alt_data_source.get_economic_indicators.side_effect = Exception("API Error")
        calculator.set_data_sources(alt_data_source=mock_alt_data_source)
        
        with pytest.raises(DataError, match="Economic indicators calculation failed"):
            await calculator.calculate_economic_indicators("BTCUSDT")

    @pytest.mark.asyncio
    async def test_sentiment_mapping_social_to_news(self, calculator, mock_social_source):
        """Test social sentiment to news sentiment mapping."""
        # Create social sentiment with different types
        social_data = {
            "twitter": [
                {"sentiment": SocialSentiment.VERY_BULLISH, "engagement_score": 100},
                {"sentiment": SocialSentiment.BULLISH, "engagement_score": 50},
                {"sentiment": SocialSentiment.NEUTRAL, "engagement_score": 25},
                {"sentiment": SocialSentiment.BEARISH, "engagement_score": 75},
                {"sentiment": SocialSentiment.VERY_BEARISH, "engagement_score": 100},
            ]
        }
        
        mock_social_source.get_social_sentiment.return_value = social_data
        calculator.set_data_sources(social_source=mock_social_source)
        
        result = await calculator.calculate_social_sentiment("BTCUSDT")
        
        # Should have processed all 5 posts
        assert result.value["total_posts"] == 5
        
        # Average sentiment should reflect the mix of bullish and bearish
        assert -1 <= result.value["average_sentiment"] <= 1

    @pytest.mark.asyncio
    async def test_concurrent_calculations(self, calculator, mock_news_source,
                                         mock_social_source, sample_news_articles,
                                         sample_social_sentiment):
        """Test concurrent alternative feature calculations."""
        # Setup mocks
        mock_news_source.get_news_for_symbol.return_value = sample_news_articles
        mock_social_source.get_social_sentiment.return_value = sample_social_sentiment
        calculator.set_data_sources(
            news_source=mock_news_source,
            social_source=mock_social_source
        )
        
        # Run multiple calculations concurrently
        tasks = [
            calculator.calculate_news_sentiment("BTCUSDT"),
            calculator.calculate_social_sentiment("BTCUSDT"),
            calculator.calculate_market_microstructure("BTCUSDT"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All calculations should succeed
        assert len(results) == 3
        assert all(isinstance(result, AlternativeResult) for result in results)
        assert results[0].feature_name == "NEWS_SENTIMENT"
        assert results[1].feature_name == "SOCIAL_SENTIMENT"
        assert results[2].feature_name == "MARKET_MICROSTRUCTURE"


class TestAlternativeConfig:
    """Test suite for AlternativeConfig dataclass."""
    
    def test_alternative_config_creation(self):
        """Test AlternativeConfig creation."""
        config = AlternativeConfig(
            feature_name="NEWS_SENTIMENT",
            feature_type=AlternativeFeatureType.NEWS_SENTIMENT,
            lookback_hours=24,
            enabled=True,
            parameters={"min_articles": 5}
        )
        
        assert config.feature_name == "NEWS_SENTIMENT"
        assert config.feature_type == AlternativeFeatureType.NEWS_SENTIMENT
        assert config.lookback_hours == 24
        assert config.enabled is True
        assert config.parameters["min_articles"] == 5


class TestSentimentStrength:
    """Test suite for SentimentStrength enum."""
    
    def test_sentiment_strength_values(self):
        """Test SentimentStrength enum values."""
        assert SentimentStrength.VERY_STRONG.value == "very_strong"
        assert SentimentStrength.STRONG.value == "strong"
        assert SentimentStrength.MODERATE.value == "moderate"
        assert SentimentStrength.WEAK.value == "weak"
        assert SentimentStrength.NEUTRAL.value == "neutral"


@pytest.mark.integration
class TestAlternativeFeatureIntegration:
    """Integration tests for alternative feature calculations."""
    
    @pytest.mark.asyncio
    async def test_news_sentiment_integration(self):
        """Test news sentiment calculation with realistic data."""
        config = MagicMock(spec=Config)
        config.alternative_features = {
            "default_lookbacks": {"news_sentiment": 24},
            "sentiment_weights": {
                "very_positive": 1.0,
                "positive": 0.5,
                "neutral": 0.0,
                "negative": -0.5,
                "very_negative": -1.0,
            },
        }
        
        calculator = AlternativeFeatureCalculator(config)
        
        # Create realistic news articles
        news_articles = [
            {"sentiment": NewsSentiment.POSITIVE, "score": 0.6},
            {"sentiment": NewsSentiment.POSITIVE, "score": 0.7},
            {"sentiment": NewsSentiment.NEUTRAL, "score": 0.0},
            {"sentiment": NewsSentiment.NEGATIVE, "score": -0.4},
        ]
        
        mock_news_source = AsyncMock()
        mock_news_source.get_news_for_symbol.return_value = news_articles
        calculator.set_data_sources(news_source=mock_news_source)
        
        result = await calculator.calculate_news_sentiment("BTCUSDT")
        
        # Expected average: (0.5 + 0.5 + 0.0 + (-0.5)) / 4 = 0.125
        expected_avg = (0.5 + 0.5 + 0.0 + (-0.5)) / 4
        
        assert abs(result.value["average_sentiment"] - expected_avg) < 0.001
        assert result.value["article_count"] == 4