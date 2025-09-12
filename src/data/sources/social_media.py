"""
Social Media Data Source Integration

This module provides social media sentiment data ingestion:
- Twitter/X sentiment monitoring
- Reddit discussions analysis
- Social sentiment aggregation
- Real-time social media monitoring

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from aiohttp import ClientSession

from src.core import BaseComponent, get_logger
from src.core.config import Config
from src.core.exceptions import DataSourceError
from src.core.logging import get_logger

# Import from P-001 core components
from src.core.types import SocialSentiment

# Import from P-002A error handling
from src.error_handling import ErrorHandler, with_retry

# Import from P-007A utilities
from src.utils.decorators import time_execution

# SocialSentiment enum moved to src.core.types to avoid circular dependencies


logger = get_logger(__name__)


@dataclass
class SocialPost:
    """Social media post data structure"""

    id: str
    platform: str
    content: str
    author: str
    created_at: datetime
    sentiment: SocialSentiment
    sentiment_score: float
    engagement_score: float
    symbols: list[str]
    metrics: dict[str, int]  # likes, retweets, comments, etc.
    metadata: dict[str, Any]


@dataclass
class SocialMetrics:
    """Aggregated social metrics for a symbol"""

    symbol: str
    total_posts: int
    sentiment_distribution: dict[str, int]
    average_sentiment: float
    engagement_volume: int
    trending_score: float
    time_period: str
    last_updated: datetime


class SocialMediaDataSource(BaseComponent):
    """
    Social media data source for sentiment analysis and trend detection.

    This class monitors social media platforms for trading-related discussions
    and provides sentiment analysis for market decision support.
    """

    def __init__(self, config: Config):
        """
        Initialize social media data source.

        Args:
            config: Application configuration
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Platform configurations
        social_media_config = getattr(config, "social_media", {})
        self.twitter_config = social_media_config.get("twitter", {})
        self.reddit_config = social_media_config.get("reddit", {})

        # HTTP session
        self.session: ClientSession | None = None

        # Data storage
        self.posts_cache: dict[str, list[SocialPost]] = {}
        self.metrics_cache: dict[str, SocialMetrics] = {}

        # Monitoring settings
        self.active = False
        self.update_interval = social_media_config.get("update_interval", 300)  # 5 minutes
        self.max_posts_per_symbol = social_media_config.get("max_posts", 200)

        # Statistics
        self.stats: dict[str, Any] = {
            "total_posts_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "last_update_time": None,
            "platform_stats": {
                "twitter": {"posts": 0, "requests": 0},
                "reddit": {"posts": 0, "requests": 0},
            },
        }

        self.logger.info("SocialMediaDataSource initialized")

    @with_retry(max_attempts=3, base_delay=2.0, exponential=True)
    async def initialize(self) -> None:
        """Initialize social media data source connections."""
        session = None
        try:
            # Create HTTP session with appropriate headers
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "TradingBot/1.0", "Accept": "application/json"},
            )

            # Test connections to available platforms
            self.session = session
            await self._test_connections()

            self.logger.info("SocialMediaDataSource initialized successfully")

        except Exception as e:
            # Cleanup session on initialization failure
            if session:
                try:
                    await session.close()
                except Exception as cleanup_e:
                    logger.error(f"Failed to close session during cleanup: {cleanup_e}")
                    # Continue cleanup process
                # Don't set self.session if initialization failed
                if self.session == session:
                    self.session = None

            self.logger.error(f"Failed to initialize SocialMediaDataSource: {e!s}")
            raise DataSourceError(f"Social media data source initialization failed: {e!s}")

    @with_retry(max_attempts=2, base_delay=1.0, exponential=True)
    async def _test_connections(self) -> None:
        """Test connections to social media platforms."""
        try:
            # For demo purposes, we'll simulate connection tests
            # In production, this would test actual API connections

            platforms_available = []

            if self.twitter_config.get("enabled", False):
                # Twitter API test would go here
                platforms_available.append("twitter")
                self.logger.info("Twitter API connection available")

            if self.reddit_config.get("enabled", False):
                # Reddit API test would go here
                platforms_available.append("reddit")
                self.logger.info("Reddit API connection available")

            if not platforms_available:
                self.logger.warning("No social media platforms configured")
            else:
                self.logger.info(f"Social media platforms available: {platforms_available}")

        except Exception as e:
            self.logger.error(f"Social media connection test failed: {e!s}")
            raise

    @time_execution
    @with_retry(max_attempts=3, base_delay=2.0, exponential=True)
    async def get_social_sentiment(
        self, symbol: str, hours_back: int = 24, platforms: list[str] | None = None
    ) -> SocialMetrics:
        """
        Get social media sentiment for a trading symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            hours_back: How many hours back to analyze
            platforms: List of platforms to check (default: all available)

        Returns:
            SocialMetrics: Aggregated social sentiment metrics
        """
        try:
            if not self.session:
                raise DataSourceError("SocialMediaDataSource not initialized")

            platforms = platforms or ["twitter", "reddit"]
            all_posts = []

            # Collect posts from all platforms
            for platform in platforms:
                if platform == "twitter" and self.twitter_config.get("enabled", False):
                    posts = await self._get_twitter_posts(symbol, hours_back)
                    all_posts.extend(posts)
                elif platform == "reddit" and self.reddit_config.get("enabled", False):
                    posts = await self._get_reddit_posts(symbol, hours_back)
                    all_posts.extend(posts)

            # Calculate aggregated metrics
            metrics = self._calculate_social_metrics(symbol, all_posts, hours_back)

            # Cache results
            cache_key = f"{symbol}_{hours_back}h"
            self.posts_cache[cache_key] = all_posts
            self.metrics_cache[symbol] = metrics

            self.stats["total_posts_processed"] += len(all_posts)
            self.stats["successful_requests"] += 1
            self.stats["last_update_time"] = datetime.now(timezone.utc)

            self.logger.info(f"Analyzed {len(all_posts)} social posts for {symbol}")
            return metrics

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Failed to get social sentiment for {symbol}: {e!s}")
            raise DataSourceError(f"Social sentiment analysis failed: {e!s}")

    async def _get_twitter_posts(self, symbol: str, hours_back: int) -> list[SocialPost]:
        """Get Twitter posts for a symbol (simulated implementation)."""
        try:
            # In production, this would use Twitter API v2
            # For now, we'll simulate data

            posts = []
            base_time = datetime.now(timezone.utc)

            # Simulate some Twitter posts
            for i in range(10):  # Simulate 10 posts
                post_time = base_time - timedelta(hours=i * 2.4)

                # Simulate post content
                content = f"  # {symbol} looking {'bullish' if i % 2 == 0 else 'bearish'} today!"

                # Simulate sentiment analysis
                sentiment, sentiment_score = self._analyze_social_sentiment(content)

                post = SocialPost(
                    id=f"twitter_{symbol}_{i}",
                    platform="twitter",
                    content=content,
                    author=f"trader_{i}",
                    created_at=post_time,
                    sentiment=sentiment,
                    sentiment_score=sentiment_score,
                    engagement_score=float(i * 10),  # Simulate engagement
                    symbols=[symbol],
                    metrics={"likes": i * 5, "retweets": i * 2, "replies": i * 1},
                    metadata={"platform_specific": "twitter_data"},
                )
                posts.append(post)

            self.stats["platform_stats"]["twitter"]["posts"] += len(posts)
            self.stats["platform_stats"]["twitter"]["requests"] += 1

            return posts

        except Exception as e:
            self.logger.error(f"Failed to get Twitter posts for {symbol}: {e!s}")
            return []

    async def _get_reddit_posts(self, symbol: str, hours_back: int) -> list[SocialPost]:
        """Get Reddit posts for a symbol (simulated implementation)."""
        try:
            # In production, this would use Reddit API (PRAW)
            # For now, we'll simulate data

            posts = []
            base_time = datetime.now(timezone.utc)

            # Simulate some Reddit posts
            for i in range(8):  # Simulate 8 posts
                post_time = base_time - timedelta(hours=i * 3)

                # Simulate post content
                content = f"Discussion about {symbol} price action and future prospects..."

                # Simulate sentiment analysis
                sentiment, sentiment_score = self._analyze_social_sentiment(content)

                post = SocialPost(
                    id=f"reddit_{symbol}_{i}",
                    platform="reddit",
                    content=content,
                    author=f"redditor_{i}",
                    created_at=post_time,
                    sentiment=sentiment,
                    sentiment_score=sentiment_score,
                    engagement_score=float(i * 15),  # Simulate engagement
                    symbols=[symbol],
                    metrics={"upvotes": i * 8, "downvotes": i * 1, "comments": i * 3},
                    metadata={"subreddit": "CryptoCurrency"},
                )
                posts.append(post)

            self.stats["platform_stats"]["reddit"]["posts"] += len(posts)
            self.stats["platform_stats"]["reddit"]["requests"] += 1

            return posts

        except Exception as e:
            self.logger.error(f"Failed to get Reddit posts for {symbol}: {e!s}")
            return []

    def _analyze_social_sentiment(self, content: str) -> tuple[SocialSentiment, float]:
        """Analyze sentiment of social media content."""
        if not content:
            return SocialSentiment.NEUTRAL, 0.0

        content = content.lower()

        # Social media specific sentiment words
        bullish_words = [
            "bullish",
            "moon",
            "pump",
            "rocket",
            "🚀",
            "buy",
            "hodl",
            "bullrun",
            "gain",
            "profit",
            "up",
            "rise",
            "surge",
            "breakout",
        ]
        bearish_words = [
            "bearish",
            "dump",
            "crash",
            "sell",
            "drop",
            "fall",
            "dip",
            "rekt",
            "loss",
            "down",
            "decline",
            "correction",
            "bear market",
        ]

        bullish_count = sum(1 for word in bullish_words if word in content)
        bearish_count = sum(1 for word in bearish_words if word in content)

        # Weight by content length
        content_length = max(len(content.split()), 1)
        sentiment_score = (bullish_count - bearish_count) / content_length

        # Determine sentiment category
        if sentiment_score > 0.1:
            sentiment = (
                SocialSentiment.VERY_BULLISH if sentiment_score > 0.2 else SocialSentiment.BULLISH
            )
        elif sentiment_score < -0.1:
            sentiment = (
                SocialSentiment.VERY_BEARISH if sentiment_score < -0.2 else SocialSentiment.BEARISH
            )
        else:
            sentiment = SocialSentiment.NEUTRAL

        return sentiment, sentiment_score

    def _calculate_social_metrics(
        self, symbol: str, posts: list[SocialPost], hours_back: int
    ) -> SocialMetrics:
        """Calculate aggregated social metrics."""
        if not posts:
            return SocialMetrics(
                symbol=symbol,
                total_posts=0,
                sentiment_distribution={},
                average_sentiment=0.0,
                engagement_volume=0,
                trending_score=0.0,
                time_period=f"{hours_back}h",
                last_updated=datetime.now(timezone.utc),
            )

        # Calculate sentiment distribution
        sentiment_counts = {}
        for sentiment_type in SocialSentiment:
            sentiment_counts[sentiment_type.value] = sum(
                1 for post in posts if post.sentiment == sentiment_type
            )

        # Calculate average sentiment
        total_sentiment = sum(post.sentiment_score for post in posts)
        average_sentiment = total_sentiment / len(posts)

        # Calculate engagement volume
        engagement_volume = sum(post.engagement_score for post in posts)

        # Calculate trending score (based on recent posts and engagement)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
        recent_posts = [post for post in posts if post.created_at > recent_cutoff]

        if recent_posts:
            recent_engagement = sum(post.engagement_score for post in recent_posts)
            trending_score = (len(recent_posts) / len(posts)) * (
                recent_engagement / engagement_volume
            )
        else:
            trending_score = 0.0

        return SocialMetrics(
            symbol=symbol,
            total_posts=len(posts),
            sentiment_distribution=sentiment_counts,
            average_sentiment=average_sentiment,
            engagement_volume=int(engagement_volume),
            trending_score=min(trending_score, 1.0),
            time_period=f"{hours_back}h",
            last_updated=datetime.now(timezone.utc),
        )

    @time_execution
    async def get_trending_symbols(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get trending cryptocurrency symbols based on social media activity.

        Args:
            limit: Maximum number of symbols to return

        Returns:
            List of trending symbols with metrics
        """
        try:
            # Common crypto symbols to check
            symbols = ["BTC", "ETH", "ADA", "DOT", "LINK", "MATIC", "SOL", "AVAX"]
            trending_data = []

            for symbol in symbols:
                try:
                    metrics = await self.get_social_sentiment(symbol, hours_back=6)

                    trending_data.append(
                        {
                            "symbol": symbol,
                            "trending_score": metrics.trending_score,
                            "total_posts": metrics.total_posts,
                            "average_sentiment": metrics.average_sentiment,
                            "engagement_volume": metrics.engagement_volume,
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to get metrics for {symbol}: {e!s}")
                    continue

            # Sort by trending score
            trending_data.sort(key=lambda x: x["trending_score"], reverse=True)

            return trending_data[:limit]

        except Exception as e:
            self.logger.error(f"Failed to get trending symbols: {e!s}")
            raise DataSourceError(f"Trending symbols analysis failed: {e!s}")

    @time_execution
    async def monitor_symbol_mentions(
        self, symbols: list[str], callback: Callable[[str, list[SocialPost]], None]
    ) -> None:
        """
        Monitor real-time mentions of trading symbols.

        Args:
            symbols: List of symbols to monitor
            callback: Callback function for new mentions
        """
        try:
            self.active = True

            while self.active:
                for symbol in symbols:
                    try:
                        # Get recent posts (last hour)
                        recent_posts = []

                        # Get Twitter posts
                        if self.twitter_config.get("enabled", False):
                            twitter_posts = await self._get_twitter_posts(symbol, 1)
                            recent_posts.extend(twitter_posts)

                        # Get Reddit posts
                        if self.reddit_config.get("enabled", False):
                            reddit_posts = await self._get_reddit_posts(symbol, 1)
                            recent_posts.extend(reddit_posts)

                        # Filter for very recent posts (last 15 minutes)
                        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=15)
                        new_posts = [post for post in recent_posts if post.created_at > cutoff_time]

                        if new_posts:
                            callback(symbol, new_posts)

                    except Exception as e:
                        self.logger.error(f"Error monitoring {symbol}: {e!s}")
                        continue

                # Wait before next check
                await asyncio.sleep(self.update_interval)

        except Exception as e:
            self.logger.error(f"Error in social media monitoring: {e!s}")
        finally:
            self.active = False

    async def get_platform_statistics(self) -> dict[str, Any]:
        """Get social media platform statistics."""
        return {
            "total_posts_processed": self.stats["total_posts_processed"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "last_update_time": self.stats["last_update_time"],
            "platform_stats": self.stats["platform_stats"].copy(),
            "cached_symbols": list(self.metrics_cache.keys()),
            "cache_sizes": {
                "posts": sum(len(posts) for posts in self.posts_cache.values()),
                "metrics": len(self.metrics_cache),
            },
        }

    async def cleanup(self) -> None:
        """Cleanup social media data source resources."""
        session = None
        try:
            self.active = False

            if self.session:
                session = self.session
                self.session = None
                await session.close()

            self.posts_cache.clear()
            self.metrics_cache.clear()

            self.logger.info("SocialMediaDataSource cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during SocialMediaDataSource cleanup: {e!s}")
        finally:
            if session and not session.closed:
                try:
                    await session.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close session in finally block: {e!s}")
