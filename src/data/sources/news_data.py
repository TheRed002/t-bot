"""
News Data Source Integration

This module provides news sentiment data ingestion:
- News articles from NewsAPI
- Sentiment analysis for market impact
- Real-time news monitoring
- Historical news data

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import DataSourceError

# Import from P-001 core components
from src.core.types import NewsSentiment

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import retry, time_execution


@dataclass
class NewsArticle:
    """News article data structure"""

    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    sentiment: NewsSentiment
    sentiment_score: float
    relevance_score: float
    symbols: list[str]
    metadata: dict[str, Any]


class NewsDataSource(BaseComponent):
    """
    News data source for sentiment analysis and market impact assessment.

    This class manages news data ingestion and sentiment analysis for
    trading decision support.
    """

    def __init__(self, config: Config):
        """
        Initialize news data source.

        Args:
            config: Application configuration
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.error_handler = ErrorHandler(config)

        # API configuration
        self.api_key = config.news_api.get("api_key")
        self.base_url = config.news_api.get("base_url", "https://newsapi.org/v2")
        self.session: aiohttp.ClientSession | None = None

        # Data storage
        self.news_cache: dict[str, list[NewsArticle]] = {}
        self.sentiment_cache: dict[str, dict[str, float]] = {}

        # Monitoring
        self.active = False
        self.update_interval = config.news_api.get("update_interval", 300)  # 5 minutes
        self.max_articles_per_symbol = config.news_api.get("max_articles", 100)

        # Statistics
        self.stats = {
            "total_articles_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "last_update_time": None,
        }

        self.logger.info("NewsDataSource initialized")

    @retry(max_attempts=3, base_delay=2.0)
    async def initialize(self) -> None:
        """Initialize news data source connections."""
        try:
            if not self.api_key:
                raise DataSourceError("NewsAPI key not configured")

            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "TradingBot/1.0"},
            )

            # Test API connection
            await self._test_connection()

            self.logger.info("NewsDataSource initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize NewsDataSource: {e!s}")
            raise DataSourceError(f"News data source initialization failed: {e!s}")

    @retry(max_attempts=3, base_delay=2.0)
    async def _test_connection(self) -> None:
        """Test connection to NewsAPI."""
        try:
            url = f"{self.base_url}/everything"
            params = {"q": "bitcoin", "pageSize": 1, "sortBy": "publishedAt"}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    self.logger.info("NewsAPI connection test successful")
                else:
                    raise DataSourceError(f"NewsAPI test failed with status {response.status}")

        except Exception as e:
            self.logger.error(f"NewsAPI connection test failed: {e!s}")
            raise

    @time_execution
    @retry(max_attempts=3, base_delay=1.0)
    async def get_news_for_symbol(
        self, symbol: str, hours_back: int = 24, max_articles: int = 50
    ) -> list[NewsArticle]:
        """
        Get news articles for a specific trading symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            hours_back: How many hours back to search
            max_articles: Maximum number of articles to return

        Returns:
            List[NewsArticle]: List of relevant news articles
        """
        try:
            if not self.session:
                raise DataSourceError("NewsDataSource not initialized")

            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)

            # Build search query
            queries = self._build_search_queries(symbol)
            all_articles = []

            for query in queries:
                articles = await self._fetch_articles(
                    query=query,
                    from_date=start_time,
                    to_date=end_time,
                    page_size=min(max_articles, 100),
                )
                all_articles.extend(articles)

            # Remove duplicates and sort by relevance
            unique_articles = self._deduplicate_articles(all_articles)
            relevant_articles = self._filter_and_score_articles(unique_articles, symbol)

            # Sort by relevance and limit results
            relevant_articles.sort(key=lambda x: x.relevance_score, reverse=True)
            result = relevant_articles[:max_articles]

            # Cache results
            cache_key = f"{symbol}_{hours_back}h"
            self.news_cache[cache_key] = result

            self.stats["total_articles_processed"] += len(result)
            self.stats["successful_requests"] += 1
            self.stats["last_update_time"] = datetime.now(timezone.utc)

            self.logger.info(f"Retrieved {len(result)} news articles for {symbol}")
            return result

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Failed to get news for {symbol}: {e!s}")
            raise DataSourceError(f"News retrieval failed: {e!s}")

    @retry(max_attempts=3, base_delay=1.0)
    async def _fetch_articles(
        self, query: str, from_date: datetime, to_date: datetime, page_size: int = 50
    ) -> list[NewsArticle]:
        """Fetch articles from NewsAPI."""
        try:
            url = f"{self.base_url}/everything"
            params = {
                "q": query,
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "language": "en",
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = []

                    for article_data in data.get("articles", []):
                        article = self._parse_article(article_data, query)
                        if article:
                            articles.append(article)

                    return articles
                else:
                    self.logger.warning(f"NewsAPI request failed with status {response.status}")
                    return []

        except Exception as e:
            self.logger.error(f"Failed to fetch articles: {e!s}")
            return []

    def _parse_article(self, article_data: dict[str, Any], query: str) -> NewsArticle | None:
        """Parse article data from NewsAPI response."""
        try:
            # Basic validation
            if not article_data.get("title") or not article_data.get("publishedAt"):
                return None

            # Parse published date
            published_at = datetime.fromisoformat(
                article_data["publishedAt"].replace("Z", "+00:00")
            )

            # Calculate sentiment (simplified - in production use proper NLP)
            sentiment, sentiment_score = self._analyze_sentiment(
                article_data.get("title", "") + " " + article_data.get("description", "")
            )

            # Extract relevant symbols
            symbols = self._extract_symbols(article_data, query)

            # Calculate relevance score
            relevance_score = self._calculate_relevance(article_data, query)

            return NewsArticle(
                title=article_data.get("title", ""),
                description=article_data.get("description", ""),
                url=article_data.get("url", ""),
                source=article_data.get("source", {}).get("name", "Unknown"),
                published_at=published_at,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
                symbols=symbols,
                metadata={
                    "author": article_data.get("author"),
                    "url_to_image": article_data.get("urlToImage"),
                    # First 200 chars
                    "content": article_data.get("content", "")[:200],
                },
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse article: {e!s}")
            return None

    def _build_search_queries(self, symbol: str) -> list[str]:
        """Build search queries for a trading symbol."""
        # Map common trading symbols to search terms
        symbol_mapping = {
            "BTC": ["bitcoin", "BTC"],
            "ETH": ["ethereum", "ETH"],
            "ADA": ["cardano", "ADA"],
            "DOT": ["polkadot", "DOT"],
            "LINK": ["chainlink", "LINK"],
        }

        queries = []
        if symbol in symbol_mapping:
            for term in symbol_mapping[symbol]:
                queries.append(f'"{term}" AND (crypto OR cryptocurrency OR trading)')
        else:
            queries.append(f'"{symbol}" AND (crypto OR cryptocurrency OR trading)')

        return queries

    def _analyze_sentiment(self, text: str) -> tuple[NewsSentiment, float]:
        """Analyze sentiment of news text (simplified implementation)."""
        if not text:
            return NewsSentiment.NEUTRAL, 0.0

        text = text.lower()

        # Simple keyword-based sentiment analysis
        positive_words = ["bullish", "surge", "rally", "gain", "up", "high", "positive", "rise"]
        negative_words = ["bearish", "crash", "drop", "fall", "down", "low", "negative", "decline"]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        sentiment_score = (positive_count - negative_count) / max(len(text.split()), 1)

        if sentiment_score > 0.05:
            sentiment = (
                NewsSentiment.POSITIVE if sentiment_score < 0.15 else NewsSentiment.VERY_POSITIVE
            )
        elif sentiment_score < -0.05:
            sentiment = (
                NewsSentiment.NEGATIVE if sentiment_score > -0.15 else NewsSentiment.VERY_NEGATIVE
            )
        else:
            sentiment = NewsSentiment.NEUTRAL

        return sentiment, sentiment_score

    def _extract_symbols(self, article_data: dict[str, Any], query: str) -> list[str]:
        """Extract relevant trading symbols from article."""
        symbols = []
        text = (article_data.get("title", "") + " " + article_data.get("description", "")).upper()

        # Common crypto symbols
        crypto_symbols = ["BTC", "ETH", "ADA", "DOT", "LINK", "MATIC", "SOL", "AVAX"]

        for symbol in crypto_symbols:
            if symbol in text:
                symbols.append(symbol)

        return list(set(symbols))  # Remove duplicates

    def _calculate_relevance(self, article_data: dict[str, Any], query: str) -> float:
        """Calculate relevance score for an article."""
        relevance_score = 0.0
        text = (article_data.get("title", "") + " " + article_data.get("description", "")).lower()

        # Check for query terms
        query_terms = query.lower().split()
        for term in query_terms:
            if term in text:
                relevance_score += 0.3

        # Check for trading-related terms
        trading_terms = ["trading", "price", "market", "exchange", "crypto", "cryptocurrency"]
        for term in trading_terms:
            if term in text:
                relevance_score += 0.1

        # Bonus for recent articles
        published_at = datetime.fromisoformat(article_data["publishedAt"].replace("Z", "+00:00"))
        hours_old = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600

        if hours_old < 6:
            relevance_score += 0.2
        elif hours_old < 24:
            relevance_score += 0.1

        return min(relevance_score, 1.0)

    def _deduplicate_articles(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()

        for article in articles:
            # Simple deduplication based on title
            title_key = article.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        return unique_articles

    def _filter_and_score_articles(
        self, articles: list[NewsArticle], symbol: str
    ) -> list[NewsArticle]:
        """Filter and score articles for relevance."""
        filtered_articles = []

        for article in articles:
            # Filter out articles with very low relevance
            if article.relevance_score < 0.2:
                continue

            # Boost relevance if symbol is mentioned
            if symbol.upper() in article.symbols:
                article.relevance_score += 0.3

            filtered_articles.append(article)

        return filtered_articles

    @time_execution
    async def get_market_sentiment(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """
        Get overall market sentiment for multiple symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dict with sentiment scores for each symbol
        """
        try:
            sentiment_data = {}

            for symbol in symbols:
                articles = await self.get_news_for_symbol(symbol, hours_back=24)

                if articles:
                    # Calculate aggregate sentiment
                    total_score = sum(article.sentiment_score for article in articles)
                    avg_score = total_score / len(articles)

                    # Count sentiment types
                    sentiment_counts = {}
                    for sentiment_type in NewsSentiment:
                        sentiment_counts[sentiment_type.value] = sum(
                            1 for article in articles if article.sentiment == sentiment_type
                        )

                    sentiment_data[symbol] = {
                        "average_sentiment_score": avg_score,
                        "total_articles": len(articles),
                        "sentiment_distribution": sentiment_counts,
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    sentiment_data[symbol] = {
                        "average_sentiment_score": 0.0,
                        "total_articles": 0,
                        "sentiment_distribution": {},
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                    }

            # Cache sentiment data
            self.sentiment_cache.update(sentiment_data)

            return sentiment_data

        except Exception as e:
            self.logger.error(f"Failed to get market sentiment: {e!s}")
            raise DataSourceError(f"Market sentiment calculation failed: {e!s}")

    async def cleanup(self) -> None:
        """Cleanup news data source resources."""
        try:
            self.active = False

            if self.session:
                await self.session.close()

            self.news_cache.clear()
            self.sentiment_cache.clear()

            self.logger.info("NewsDataSource cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during NewsDataSource cleanup: {e!s}")
