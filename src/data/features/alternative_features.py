"""
Alternative Features Calculator

This module provides alternative data feature extraction:
- News sentiment scores
- Social media sentiment and trend analysis
- Economic indicator derivatives
- Market microstructure features
- Cross-asset correlation features

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
- P-000A: Data sources integration
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

from src.core.config import Config

# Import from P-001 core components
from src.core.exceptions import DataError
from src.core.logging import get_logger

# Import sentiment enums from core types (moved there to avoid circular deps)
from src.core.types import NewsSentiment, SocialSentiment

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import cache_result, time_execution

logger = get_logger(__name__)


class AlternativeFeatureType(Enum):
    """Alternative feature type enumeration"""

    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_SENTIMENT = "social_sentiment"
    ECONOMIC_INDICATORS = "economic_indicators"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    CROSS_ASSET_CORRELATION = "cross_asset_correlation"


class SentimentStrength(Enum):
    """Sentiment strength enumeration"""

    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


@dataclass
class AlternativeConfig:
    """Alternative feature calculation configuration"""

    feature_name: str
    feature_type: AlternativeFeatureType
    lookback_hours: int
    enabled: bool = True
    parameters: dict[str, Any] = None


@dataclass
class AlternativeResult:
    """Alternative feature calculation result"""

    feature_name: str
    symbol: str
    timestamp: datetime
    value: float | dict[str, Any] | None
    metadata: dict[str, Any]
    calculation_time: float


class AlternativeFeatureCalculator:
    """
    Comprehensive alternative data feature calculator.

    This class integrates news sentiment, social media sentiment,
    economic indicators, and market microstructure features.
    """

    def __init__(self, config: Config):
        """Initialize alternative feature calculator."""
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Alternative data configuration
        alt_config = getattr(config, "alternative_features", {})
        if hasattr(alt_config, "get"):
            self.default_lookbacks = alt_config.get(
                "default_lookbacks",
                {
                    "news_sentiment": 24,  # 24 hours
                    "social_sentiment": 12,  # 12 hours
                    "economic": 168,  # 1 week
                    "microstructure": 6,  # 6 hours
                },
            )
            self.sentiment_weights = alt_config.get(
                "sentiment_weights",
                {
                    "very_positive": 1.0,
                    "positive": 0.5,
                    "neutral": 0.0,
                    "negative": -0.5,
                    "very_negative": -1.0,
                },
            )
            self.update_interval = alt_config.get(
                "update_interval", 300)  # 5 minutes
        else:
            self.default_lookbacks = {
                "news_sentiment": 24,
                "social_sentiment": 12,
                "economic": 168,
                "microstructure": 6,
            }
            self.sentiment_weights = {
                "very_positive": 1.0,
                "positive": 0.5,
                "neutral": 0.0,
                "negative": -0.5,
                "very_negative": -1.0,
            }
            self.update_interval = 300

        # Data sources will be injected by the data pipeline
        self.news_source = None
        self.social_source = None
        self.alt_data_source = None

        # Data storage
        self.news_data: dict[str, list[dict[str, Any]]] = {}
        self.social_data: dict[str, list[dict[str, Any]]] = {}
        self.economic_data: dict[str, list[dict[str, Any]]] = {}
        self.microstructure_data: dict[str, list[dict[str, Any]]] = {}

        # Feature cache
        self.feature_cache: dict[str, dict[str, Any]] = {}

        # Statistics
        self.calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "news_articles_processed": 0,
            "social_posts_processed": 0,
            "avg_calculation_time": 0.0,
        }

        logger.info("AlternativeFeatureCalculator initialized")

    def set_data_sources(self, news_source=None, social_source=None, alt_data_source=None) -> None:
        """
        Inject data sources to avoid circular dependencies.

        Args:
            news_source: NewsDataSource instance
            social_source: SocialMediaDataSource instance
            alt_data_source: AlternativeDataSource instance
        """
        self.news_source = news_source
        self.social_source = social_source
        self.alt_data_source = alt_data_source
        logger.info("Alternative data sources injected")

    async def initialize(self) -> None:
        """Initialize data sources if they are available."""
        try:
            if self.news_source:
                await self.news_source.initialize()
            if self.social_source:
                await self.social_source.initialize()
            if self.alt_data_source:
                await self.alt_data_source.initialize()
            logger.info("Alternative data sources initialized")
        except Exception as e:
            logger.error(
                f"Failed to initialize alternative data sources: {e!s}")
            raise DataError(
                f"Alternative data source initialization failed: {e!s}")

    @time_execution
    @cache_result(ttl_seconds=300)
    async def calculate_news_sentiment(
        self, symbol: str, lookback_hours: int = None
    ) -> AlternativeResult:
        """
        Calculate news sentiment features for a symbol.

        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back for news

        Returns:
            AlternativeResult: Calculation result
        """
        start_time = datetime.now()

        try:
            lookback_hours = lookback_hours or self.default_lookbacks["news_sentiment"]

            # Get news data
            if not self.news_source:
                return AlternativeResult(
                    feature_name="NEWS_SENTIMENT",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={
                        "lookback_hours": lookback_hours,
                        "reason": "news_source_not_available",
                    },
                    calculation_time=(
                        datetime.now() - start_time).total_seconds(),
                )

            news_articles = await self.news_source.get_news_for_symbol(
                symbol=symbol, hours_back=lookback_hours, max_articles=100
            )

            if not news_articles:
                return AlternativeResult(
                    feature_name="NEWS_SENTIMENT",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"lookback_hours": lookback_hours,
                              "reason": "no_news_data"},
                    calculation_time=(
                        datetime.now() - start_time).total_seconds(),
                )

            # Process sentiment scores
            sentiment_scores = []
            sentiment_counts = Counter()
            article_count = 0

            for article in news_articles:
                if "sentiment" in article and "score" in article:
                    sentiment_enum = article["sentiment"]
                    if isinstance(sentiment_enum, NewsSentiment):
                        sentiment_str = sentiment_enum.value
                    else:
                        sentiment_str = str(sentiment_enum).lower()

                    # Map sentiment to numerical score
                    if sentiment_str in self.sentiment_weights:
                        sentiment_scores.append(
                            self.sentiment_weights[sentiment_str])
                        sentiment_counts[sentiment_str] += 1
                        article_count += 1

            if not sentiment_scores:
                return AlternativeResult(
                    feature_name="NEWS_SENTIMENT",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"lookback_hours": lookback_hours,
                              "reason": "no_sentiment_scores"},
                    calculation_time=(
                        datetime.now() - start_time).total_seconds(),
                )

            # Calculate sentiment features
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            sentiment_trend = self._calculate_sentiment_trend(sentiment_scores)

            # Calculate time-weighted sentiment (recent news weighted more)
            time_weights = np.exp(-np.arange(len(sentiment_scores)) * 0.1)
            weighted_sentiment = np.average(
                sentiment_scores, weights=time_weights)

            # Determine sentiment strength
            sentiment_strength = self._determine_sentiment_strength(
                avg_sentiment, sentiment_std)

            sentiment_values = {
                "average_sentiment": avg_sentiment,
                "weighted_sentiment": weighted_sentiment,
                "sentiment_std": sentiment_std,
                "sentiment_trend": sentiment_trend,
                "sentiment_strength": sentiment_strength.value,
                "article_count": article_count,
                "sentiment_distribution": dict(sentiment_counts),
                "positive_ratio": sentiment_counts.get("positive", 0) / max(article_count, 1),
                "negative_ratio": sentiment_counts.get("negative", 0) / max(article_count, 1),
            }

            self.calculation_stats["successful_calculations"] += 1
            self.calculation_stats["news_articles_processed"] += article_count

            return AlternativeResult(
                feature_name="NEWS_SENTIMENT",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=sentiment_values,
                metadata={
                    "lookback_hours": lookback_hours,
                    "sources_count": len(set(a.get("source", "") for a in news_articles)),
                },
                calculation_time=(datetime.now() - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            logger.error(
                f"News sentiment calculation failed for {symbol}: {e!s}")
            raise DataError(f"News sentiment calculation failed: {e!s}")

    @time_execution
    @cache_result(ttl_seconds=300)
    async def calculate_social_sentiment(
        self, symbol: str, lookback_hours: int = None
    ) -> AlternativeResult:
        """
        Calculate social media sentiment features for a symbol.

        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back for social data

        Returns:
            AlternativeResult: Calculation result
        """
        start_time = datetime.now()

        try:
            lookback_hours = lookback_hours or self.default_lookbacks["social_sentiment"]

            # Get social media sentiment data
            if not self.social_source:
                return AlternativeResult(
                    feature_name="SOCIAL_SENTIMENT",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={
                        "lookback_hours": lookback_hours,
                        "reason": "social_source_not_available",
                    },
                    calculation_time=(
                        datetime.now() - start_time).total_seconds(),
                )

            social_sentiment = await self.social_source.get_social_sentiment(
                symbol=symbol,
                hours_back=lookback_hours,
                platforms=["twitter", "reddit"],
                max_posts=500,
            )

            if not social_sentiment:
                return AlternativeResult(
                    feature_name="SOCIAL_SENTIMENT",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"lookback_hours": lookback_hours,
                              "reason": "no_social_data"},
                    calculation_time=(
                        datetime.now() - start_time).total_seconds(),
                )

            # Process social sentiment scores
            sentiment_scores = []
            platform_sentiments = {}
            total_posts = 0
            total_engagement = 0

            for platform, posts in social_sentiment.items():
                platform_scores = []
                platform_engagement = 0

                for post in posts:
                    if "sentiment" in post:
                        sentiment_enum = post["sentiment"]
                        if isinstance(sentiment_enum, SocialSentiment):
                            # Map social sentiment to news sentiment scale
                            social_to_news_map = {
                                "very_bullish": "very_positive",
                                "bullish": "positive",
                                "neutral": "neutral",
                                "bearish": "negative",
                                "very_bearish": "very_negative",
                            }
                            sentiment_str = social_to_news_map.get(
                                sentiment_enum.value, "neutral")
                        else:
                            sentiment_str = str(sentiment_enum).lower()

                        if sentiment_str in self.sentiment_weights:
                            score = self.sentiment_weights[sentiment_str]
                            sentiment_scores.append(score)
                            platform_scores.append(score)

                            # Weight by engagement (likes, shares, etc.)
                            engagement = post.get("engagement_score", 1)
                            total_engagement += engagement
                            platform_engagement += engagement
                            total_posts += 1

                if platform_scores:
                    platform_sentiments[platform] = {
                        "avg_sentiment": np.mean(platform_scores),
                        "post_count": len(platform_scores),
                        "engagement": platform_engagement,
                    }

            if not sentiment_scores:
                return AlternativeResult(
                    feature_name="SOCIAL_SENTIMENT",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"lookback_hours": lookback_hours,
                              "reason": "no_sentiment_scores"},
                    calculation_time=(
                        datetime.now() - start_time).total_seconds(),
                )

            # Calculate overall social sentiment features
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)

            # Calculate volume-weighted sentiment
            if total_engagement > 0:
                engagement_weights = [
                    post.get("engagement_score", 1) / total_engagement
                    for post in [p for posts in social_sentiment.values() for p in posts]
                ]
                volume_weighted_sentiment = np.average(
                    sentiment_scores, weights=engagement_weights[: len(
                        sentiment_scores)]
                )
            else:
                volume_weighted_sentiment = avg_sentiment

            # Calculate sentiment momentum (recent vs older posts)
            if len(sentiment_scores) >= 10:
                recent_sentiment = np.mean(
                    sentiment_scores[-len(sentiment_scores) // 3:])
                older_sentiment = np.mean(
                    sentiment_scores[: len(sentiment_scores) // 3])
                sentiment_momentum = recent_sentiment - older_sentiment
            else:
                sentiment_momentum = 0.0

            social_values = {
                "average_sentiment": avg_sentiment,
                "volume_weighted_sentiment": volume_weighted_sentiment,
                "sentiment_std": sentiment_std,
                "sentiment_momentum": sentiment_momentum,
                "total_posts": total_posts,
                "total_engagement": total_engagement,
                "platform_sentiments": platform_sentiments,
                "engagement_per_post": total_engagement / max(total_posts, 1),
            }

            self.calculation_stats["successful_calculations"] += 1
            self.calculation_stats["social_posts_processed"] += total_posts

            return AlternativeResult(
                feature_name="SOCIAL_SENTIMENT",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=social_values,
                metadata={
                    "lookback_hours": lookback_hours,
                    "platforms": list(social_sentiment.keys()),
                },
                calculation_time=(datetime.now() - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            logger.error(
                f"Social sentiment calculation failed for {symbol}: {e!s}")
            raise DataError(f"Social sentiment calculation failed: {e!s}")

    @time_execution
    @cache_result(ttl_seconds=1800)
    async def calculate_economic_indicators(
        self, symbol: str, lookback_hours: int = None
    ) -> AlternativeResult:
        """
        Calculate economic indicator features.

        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back for economic data

        Returns:
            AlternativeResult: Calculation result
        """
        start_time = datetime.now()

        try:
            lookback_hours = lookback_hours or self.default_lookbacks["economic"]

            # Get economic indicators
            if not self.alt_data_source:
                return AlternativeResult(
                    feature_name="ECONOMIC_INDICATORS",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={
                        "lookback_hours": lookback_hours,
                        "reason": "alt_data_source_not_available",
                    },
                    calculation_time=(
                        datetime.now() - start_time).total_seconds(),
                )

            economic_data = await self.alt_data_source.get_economic_indicators(
                indicators=["inflation", "unemployment",
                            "gdp_growth", "interest_rates"],
                lookback_days=lookback_hours // 24,
            )

            if not economic_data:
                return AlternativeResult(
                    feature_name="ECONOMIC_INDICATORS",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    value=None,
                    metadata={"lookback_hours": lookback_hours,
                              "reason": "no_economic_data"},
                    calculation_time=(
                        datetime.now() - start_time).total_seconds(),
                )

            # Process economic indicators
            indicator_values = {}
            indicator_changes = {}

            for indicator_name, data_points in economic_data.items():
                if len(data_points) >= 2:
                    latest_value = data_points[-1]["value"]
                    previous_value = data_points[-2]["value"]
                    change = (
                        (latest_value - previous_value) / previous_value
                        if previous_value != 0
                        else 0
                    )

                    indicator_values[indicator_name] = latest_value
                    indicator_changes[f"{indicator_name}_change"] = change

            # Calculate composite economic score
            # This is a simplified example - in practice, you'd use economic
            # models
            economic_score = 0.0
            weight_sum = 0.0

            weights = {
                "inflation": -0.3,
                "unemployment": -0.3,
                "gdp_growth": 0.4,
                "interest_rates": -0.2,
            }
            for indicator, weight in weights.items():
                if f"{indicator}_change" in indicator_changes:
                    economic_score += weight * \
                        indicator_changes[f"{indicator}_change"]
                    weight_sum += abs(weight)

            if weight_sum > 0:
                economic_score /= weight_sum

            economic_values = {
                "economic_score": economic_score,
                "indicator_values": indicator_values,
                "indicator_changes": indicator_changes,
                "data_freshness_hours": min(
                    [
                        (
                            datetime.now(timezone.utc)
                            - datetime.fromisoformat(
                                data_points[-1]["timestamp"].replace(
                                    "Z", "+00:00")
                            )
                        ).total_seconds()
                        / 3600
                        for data_points in economic_data.values()
                        if data_points
                    ]
                ),
            }

            self.calculation_stats["successful_calculations"] += 1

            return AlternativeResult(
                feature_name="ECONOMIC_INDICATORS",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=economic_values,
                metadata={"lookback_hours": lookback_hours,
                          "indicators_count": len(economic_data)},
                calculation_time=(datetime.now() - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            logger.error(
                f"Economic indicators calculation failed for {symbol}: {e!s}")
            raise DataError(f"Economic indicators calculation failed: {e!s}")

    def _calculate_sentiment_trend(self, sentiment_scores: list[float]) -> float:
        """Calculate sentiment trend using linear regression."""
        if len(sentiment_scores) < 3:
            return 0.0

        x = np.arange(len(sentiment_scores))
        y = np.array(sentiment_scores)

        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def _determine_sentiment_strength(
        self, avg_sentiment: float, sentiment_std: float
    ) -> SentimentStrength:
        """Determine sentiment strength based on average and consistency."""
        abs_sentiment = abs(avg_sentiment)

        # High consistency (low std) and strong sentiment
        if sentiment_std < 0.2 and abs_sentiment > 0.7:
            return SentimentStrength.VERY_STRONG
        elif sentiment_std < 0.3 and abs_sentiment > 0.5:
            return SentimentStrength.STRONG
        elif abs_sentiment > 0.3:
            return SentimentStrength.MODERATE
        elif abs_sentiment > 0.1:
            return SentimentStrength.WEAK
        else:
            return SentimentStrength.NEUTRAL

    @time_execution
    async def calculate_market_microstructure(
        self, symbol: str, lookback_hours: int = None
    ) -> AlternativeResult:
        """
        Calculate market microstructure features.

        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back for microstructure analysis

        Returns:
            AlternativeResult: Calculation result
        """
        start_time = datetime.now()

        try:
            lookback_hours = lookback_hours or self.default_lookbacks["microstructure"]

            # This would typically analyze order book data, trade sizes, bid-ask spreads
            # For now, we'll create a placeholder implementation

            microstructure_values = {
                "bid_ask_spread_avg": None,
                "order_flow_imbalance": None,
                "trade_size_distribution": None,
                "market_impact": None,
                "liquidity_score": None,
                "price_discovery_efficiency": None,
            }

            return AlternativeResult(
                feature_name="MARKET_MICROSTRUCTURE",
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                value=microstructure_values,
                metadata={"lookback_hours": lookback_hours,
                          "status": "placeholder_implementation"},
                calculation_time=(datetime.now() - start_time).total_seconds(),
            )

        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            logger.error(
                f"Market microstructure calculation failed for {symbol}: {e!s}")
            raise DataError(f"Market microstructure calculation failed: {e!s}")

    @time_execution
    async def calculate_batch_features(
        self, symbol: str, features: list[str]
    ) -> dict[str, AlternativeResult]:
        """
        Calculate multiple alternative features in batch.

        Args:
            symbol: Trading symbol
            features: List of feature names to calculate

        Returns:
            Dict[str, AlternativeResult]: Results by feature name
        """
        try:
            results = {}

            for feature in features:
                try:
                    if feature.upper() == "NEWS_SENTIMENT":
                        results["NEWS_SENTIMENT"] = await self.calculate_news_sentiment(symbol)
                    elif feature.upper() == "SOCIAL_SENTIMENT":
                        results["SOCIAL_SENTIMENT"] = await self.calculate_social_sentiment(symbol)
                    elif feature.upper() == "ECONOMIC_INDICATORS":
                        results["ECONOMIC_INDICATORS"] = await self.calculate_economic_indicators(
                            symbol
                        )
                    elif feature.upper() == "MARKET_MICROSTRUCTURE":
                        results["MARKET_MICROSTRUCTURE"] = (
                            await self.calculate_market_microstructure(symbol)
                        )
                    else:
                        logger.warning(
                            f"Unknown alternative feature: {feature}")

                except Exception as e:
                    logger.error(
                        f"Failed to calculate {feature} for {symbol}: {e!s}")
                    results[feature] = None

            successful_count = len([r for r in results.values() if r is not None])
            logger.info(
                f"Calculated {successful_count} alternative features for {symbol}"
            )
            return results

        except Exception as e:
            logger.error(
                f"Batch alternative feature calculation failed for {symbol}: {e!s}")
            raise DataError(f"Batch calculation failed: {e!s}")

    @time_execution
    async def get_calculation_summary(self) -> dict[str, Any]:
        """Get calculation statistics and summary."""
        try:
            total = self.calculation_stats["total_calculations"]
            success_rate = (
                (self.calculation_stats["successful_calculations"] / total * 100)
                if total > 0
                else 0
            )

            return {
                "statistics": self.calculation_stats.copy(),
                "success_rate": f"{success_rate:.2f}%",
                "data_sources_available": {
                    "news": hasattr(self.news_source, "_api_key"),
                    "social": hasattr(self.social_source, "_api_key"),
                    "economic": hasattr(self.alt_data_source, "_api_key"),
                },
                "update_interval_minutes": self.update_interval / 60,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to generate calculation summary: {e!s}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
