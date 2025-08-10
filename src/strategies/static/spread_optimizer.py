"""
Spread Optimizer for Market Making Strategy.

This module implements dynamic spread optimization for the market making strategy,
including volatility-based adjustments, order book imbalance detection,
competitor spread analysis, and market impact assessment.

CRITICAL: This integrates with the market making strategy and provides
sophisticated spread optimization capabilities.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio

# MANDATORY: Import from P-001
from src.core.types import MarketData, OrderBook, Ticker
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_price
from src.utils.formatters import format_percentage

logger = get_logger(__name__)


class SpreadOptimizer:
    """
    Spread Optimizer for Market Making Strategy.

    This class implements dynamic spread optimization including:
    - Volatility-based spread adjustment (2x multiplier default)
    - Order book imbalance detection
    - Competitor spread analysis
    - Market impact assessment
    - Optimal bid-ask spread calculation
    - Adaptive spread widening during high volatility
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Spread Optimizer.

        Args:
            config: Configuration dictionary with spread optimization parameters
        """
        self.config = config

        # Volatility parameters
        self.volatility_multiplier = config.get("volatility_multiplier", 2.0)
        self.volatility_window = config.get("volatility_window", 20)
        self.min_volatility = config.get("min_volatility", 0.001)  # 0.1%
        self.max_volatility = config.get("max_volatility", 0.05)   # 5%

        # Order book parameters
        self.imbalance_threshold = config.get(
            "imbalance_threshold", 0.1)  # 10%
        self.depth_levels = config.get("depth_levels", 5)
        self.min_spread = Decimal(
            str(config.get("min_spread", 0.0001)))  # 0.01%
        self.max_spread = Decimal(str(config.get("max_spread", 0.01)))    # 1%

        # Competition parameters
        self.competitor_monitoring = config.get("competitor_monitoring", True)
        self.competitor_weight = config.get("competitor_weight", 0.3)
        self.max_competitor_spread = Decimal(
            str(config.get("max_competitor_spread", 0.005)))  # 0.5%

        # Market impact parameters
        self.impact_threshold = config.get("impact_threshold", 0.001)  # 0.1%
        self.impact_multiplier = config.get("impact_multiplier", 1.5)

        # State tracking
        self.price_history: List[float] = []
        self.spread_history: List[float] = []
        self.volatility_history: List[float] = []
        self.competitor_spreads: List[float] = []

        # Performance tracking
        self.optimization_count = 0
        self.volatility_adjustments = 0
        self.imbalance_adjustments = 0
        self.competitor_adjustments = 0

        logger.info(
            "Spread Optimizer initialized",
            volatility_multiplier=self.volatility_multiplier,
            imbalance_threshold=self.imbalance_threshold,
            competitor_monitoring=self.competitor_monitoring
        )

    @time_execution
    async def optimize_spread(self,
                              base_spread: Decimal,
                              market_data: MarketData,
                              order_book: Optional[OrderBook] = None,
                              competitor_spreads: Optional[List[float]] = None) -> Decimal:
        """
        Optimize spread based on market conditions.

        Args:
            base_spread: Base spread to optimize
            market_data: Current market data
            order_book: Current order book (optional)
            competitor_spreads: List of competitor spreads (optional)

        Returns:
            Optimized spread
        """
        try:
            # Update price and spread history
            self._update_history(market_data)

            # Calculate volatility adjustment
            volatility_adjustment = await self._calculate_volatility_adjustment(base_spread)

            # Calculate order book imbalance adjustment
            imbalance_adjustment = await self._calculate_imbalance_adjustment(
                base_spread, order_book
            )

            # Calculate competitor adjustment
            competitor_adjustment = await self._calculate_competitor_adjustment(
                base_spread, competitor_spreads
            )

            # Calculate market impact adjustment
            impact_adjustment = await self._calculate_impact_adjustment(base_spread)

            # Combine all adjustments
            total_adjustment = (
                volatility_adjustment +
                imbalance_adjustment +
                competitor_adjustment +
                impact_adjustment
            )

            # Apply adjustment to base spread
            optimized_spread = base_spread * (1 + total_adjustment)

            # Ensure spread is within bounds
            optimized_spread = max(optimized_spread, self.min_spread)
            optimized_spread = min(optimized_spread, self.max_spread)

            self.optimization_count += 1

            logger.debug(
                "Spread optimization completed",
                base_spread=float(base_spread),
                volatility_adjustment=float(volatility_adjustment),
                imbalance_adjustment=float(imbalance_adjustment),
                competitor_adjustment=float(competitor_adjustment),
                impact_adjustment=float(impact_adjustment),
                total_adjustment=float(total_adjustment),
                optimized_spread=float(optimized_spread)
            )

            return optimized_spread

        except Exception as e:
            logger.error("Spread optimization failed", error=str(e))
            return base_spread

    @time_execution
    async def _calculate_volatility_adjustment(
            self, base_spread: Decimal) -> Decimal:
        """
        Calculate spread adjustment based on volatility.

        Args:
            base_spread: Base spread

        Returns:
            Volatility-based adjustment
        """
        try:
            if len(self.price_history) < self.volatility_window:
                return Decimal("0")

            # Calculate current volatility
            prices = np.array(self.price_history[-self.volatility_window:])
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Clamp volatility to bounds
            volatility = max(volatility, self.min_volatility)
            volatility = min(volatility, self.max_volatility)

            # Calculate adjustment
            adjustment = volatility * self.volatility_multiplier

            # Normalize to spread percentage
            adjustment = min(adjustment, 0.01)  # Max 1% adjustment

            self.volatility_adjustments += 1

            logger.debug(
                "Volatility adjustment calculated",
                volatility=volatility,
                adjustment=float(adjustment),
                multiplier=self.volatility_multiplier
            )

            return Decimal(str(adjustment))

        except Exception as e:
            logger.error(
                "Volatility adjustment calculation failed",
                error=str(e))
            return Decimal("0")

    @time_execution
    async def _calculate_imbalance_adjustment(
            self,
            base_spread: Decimal,
            order_book: Optional[OrderBook]) -> Decimal:
        """
        Calculate spread adjustment based on order book imbalance.

        Args:
            base_spread: Base spread
            order_book: Current order book

        Returns:
            Imbalance-based adjustment
        """
        try:
            if not order_book or not order_book.bids or not order_book.asks:
                return Decimal("0")

            # Calculate bid and ask volumes at specified depth
            bid_volume = sum(float(bid[1])
                             for bid in order_book.bids[:self.depth_levels])
            ask_volume = sum(float(ask[1])
                             for ask in order_book.asks[:self.depth_levels])

            if bid_volume == 0 or ask_volume == 0:
                return Decimal("0")

            # Calculate imbalance ratio
            total_volume = bid_volume + ask_volume
            imbalance_ratio = abs(bid_volume - ask_volume) / total_volume

            # Apply adjustment if imbalance exceeds threshold
            if imbalance_ratio > self.imbalance_threshold:
                # Wider spread when imbalance is high
                adjustment = imbalance_ratio * 0.5  # 50% of imbalance ratio
                adjustment = min(adjustment, 0.005)  # Max 0.5% adjustment

                self.imbalance_adjustments += 1

                logger.debug(
                    "Imbalance adjustment calculated",
                    imbalance_ratio=imbalance_ratio,
                    adjustment=float(adjustment),
                    bid_volume=bid_volume,
                    ask_volume=ask_volume
                )

                return Decimal(str(adjustment))

            return Decimal("0")

        except Exception as e:
            logger.error(
                "Imbalance adjustment calculation failed",
                error=str(e))
            return Decimal("0")

    @time_execution
    async def _calculate_competitor_adjustment(
            self, base_spread: Decimal, competitor_spreads: Optional[List[float]]) -> Decimal:
        """
        Calculate spread adjustment based on competitor spreads.

        Args:
            base_spread: Base spread
            competitor_spreads: List of competitor spreads

        Returns:
            Competitor-based adjustment
        """
        try:
            if not self.competitor_monitoring or not competitor_spreads:
                return Decimal("0")

            # Calculate average competitor spread
            valid_spreads = [
                s for s in competitor_spreads if 0 < s < float(
                    self.max_competitor_spread)]

            if not valid_spreads:
                return Decimal("0")

            avg_competitor_spread = np.mean(valid_spreads)
            current_spread = float(base_spread)

            # Calculate adjustment to match competitor spread
            spread_diff = avg_competitor_spread - current_spread

            # Apply weighted adjustment
            adjustment = spread_diff * self.competitor_weight

            # Clamp adjustment
            adjustment = max(adjustment, -0.002)  # Max 0.2% reduction
            adjustment = min(adjustment, 0.002)   # Max 0.2% increase

            self.competitor_adjustments += 1

            logger.debug(
                "Competitor adjustment calculated",
                avg_competitor_spread=avg_competitor_spread,
                current_spread=current_spread,
                adjustment=adjustment,
                competitor_count=len(valid_spreads)
            )

            return Decimal(str(adjustment))

        except Exception as e:
            logger.error(
                "Competitor adjustment calculation failed",
                error=str(e))
            return Decimal("0")

    @time_execution
    async def _calculate_impact_adjustment(
            self, base_spread: Decimal) -> Decimal:
        """
        Calculate spread adjustment based on market impact.

        Args:
            base_spread: Base spread

        Returns:
            Impact-based adjustment
        """
        try:
            # Calculate recent spread volatility as proxy for market impact
            if len(self.spread_history) < 10:
                return Decimal("0")

            recent_spreads = self.spread_history[-10:]
            spread_volatility = np.std(recent_spreads)

            # Apply adjustment if spread volatility is high
            if spread_volatility > self.impact_threshold:
                adjustment = spread_volatility * self.impact_multiplier
                adjustment = min(adjustment, 0.003)  # Max 0.3% adjustment

                logger.debug(
                    "Impact adjustment calculated",
                    spread_volatility=spread_volatility,
                    adjustment=float(adjustment),
                    threshold=self.impact_threshold
                )

                return Decimal(str(adjustment))

            return Decimal("0")

        except Exception as e:
            logger.error("Impact adjustment calculation failed", error=str(e))
            return Decimal("0")

    def _update_history(self, market_data: MarketData) -> None:
        """
        Update price and spread history.

        Args:
            market_data: Current market data
        """
        try:
            # Update price history
            self.price_history.append(float(market_data.price))

            # Keep only recent history
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]

            # Update spread history
            if market_data.bid and market_data.ask:
                spread = (market_data.ask - market_data.bid) / market_data.bid
                self.spread_history.append(float(spread))

                if len(self.spread_history) > 100:
                    self.spread_history = self.spread_history[-100:]

            # Update volatility history
            if len(self.price_history) >= self.volatility_window:
                prices = np.array(self.price_history[-self.volatility_window:])
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)
                self.volatility_history.append(volatility)

                if len(self.volatility_history) > 50:
                    self.volatility_history = self.volatility_history[-50:]

        except Exception as e:
            logger.error("History update failed", error=str(e))

    @time_execution
    async def calculate_optimal_spread(self,
                                       market_data: MarketData,
                                       order_book: Optional[OrderBook] = None,
                                       competitor_spreads: Optional[List[float]] = None) -> Tuple[Decimal,
                                                                                                  Decimal]:
        """
        Calculate optimal bid and ask spreads.

        Args:
            market_data: Current market data
            order_book: Current order book
            competitor_spreads: List of competitor spreads

        Returns:
            Tuple of (bid_spread, ask_spread)
        """
        try:
            # Calculate base spread from market data
            if market_data.bid and market_data.ask:
                base_spread = (
                    market_data.ask - market_data.bid) / market_data.bid
            else:
                base_spread = Decimal("0.001")  # Default 0.1%

            # Optimize spread
            optimized_spread = await self.optimize_spread(
                base_spread, market_data, order_book, competitor_spreads
            )

            # Calculate bid and ask spreads
            bid_spread = optimized_spread / Decimal("2")
            ask_spread = optimized_spread / Decimal("2")

            logger.debug(
                "Optimal spreads calculated",
                base_spread=float(base_spread),
                optimized_spread=float(optimized_spread),
                bid_spread=float(bid_spread),
                ask_spread=float(ask_spread)
            )

            return bid_spread, ask_spread

        except Exception as e:
            logger.error("Optimal spread calculation failed", error=str(e))
            # Return default spreads
            default_spread = Decimal("0.001")  # 0.1%
            return default_spread / Decimal("2"), default_spread / Decimal("2")

    @time_execution
    async def should_widen_spread(self, market_data: MarketData) -> bool:
        """
        Determine if spread should be widened due to market conditions.

        Args:
            market_data: Current market data

        Returns:
            True if spread should be widened, False otherwise
        """
        try:
            # Check for high volatility
            if len(self.volatility_history) > 0:
                recent_volatility = np.mean(self.volatility_history[-5:])
                if recent_volatility > self.max_volatility * 0.8:
                    logger.info("Spread widening due to high volatility",
                                volatility=recent_volatility)
                    return True

            # Check for wide recent spreads
            if len(self.spread_history) > 0:
                recent_spreads = self.spread_history[-5:]
                avg_recent_spread = np.mean(recent_spreads)
                if avg_recent_spread > float(self.max_spread) * 0.8:
                    logger.info("Spread widening due to wide recent spreads",
                                avg_spread=avg_recent_spread)
                    return True

            # Check for rapid price movements
            if len(self.price_history) > 10:
                recent_prices = self.price_history[-10:]
                price_changes = np.diff(recent_prices) / recent_prices[:-1]
                max_change = np.max(np.abs(price_changes))

                if max_change > 0.01:  # 1% price change
                    logger.info("Spread widening due to rapid price movements",
                                max_change=max_change)
                    return True

            return False

        except Exception as e:
            logger.error("Spread widening check failed", error=str(e))
            return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary.

        Returns:
            Dictionary with optimization information
        """
        try:
            return {
                "optimization_count": self.optimization_count,
                "volatility_adjustments": self.volatility_adjustments,
                "imbalance_adjustments": self.imbalance_adjustments,
                "competitor_adjustments": self.competitor_adjustments,
                "volatility_multiplier": self.volatility_multiplier,
                "imbalance_threshold": self.imbalance_threshold,
                "competitor_monitoring": self.competitor_monitoring,
                "min_spread": float(self.min_spread),
                "max_spread": float(self.max_spread),
                "price_history_length": len(self.price_history),
                "spread_history_length": len(self.spread_history),
                "volatility_history_length": len(self.volatility_history)
            }

        except Exception as e:
            logger.error(
                "Optimization summary generation failed",
                error=str(e))
            return {}
