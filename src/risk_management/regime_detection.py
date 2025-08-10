"""
Market Regime Detection Module

This module implements market regime classification and detection for dynamic risk management.
It detects volatility regimes, trend regimes, and correlation regimes with change detection.

CRITICAL: This module integrates with existing risk management framework from P-008/P-009.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import RiskManagementError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import MarketData, MarketRegime, RegimeChangeEvent

# MANDATORY: Import from P-008/P-009
# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class MarketRegimeDetector:
    """
    Market regime detection and classification system.

    This class implements sophisticated market regime detection algorithms
    for dynamic risk management adaptation.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize market regime detector with configuration."""
        self.config = config
        self.volatility_window = config.get("volatility_window", 20)
        self.trend_window = config.get("trend_window", 50)
        self.correlation_window = config.get("correlation_window", 30)
        self.regime_change_threshold = config.get("regime_change_threshold", 0.7)

        # Volatility thresholds (annualized)
        self.volatility_thresholds = {
            MarketRegime.LOW_VOLATILITY: 0.15,  # < 15% annualized
            MarketRegime.MEDIUM_VOLATILITY: 0.30,  # 15-30% annualized
            MarketRegime.HIGH_VOLATILITY: 0.50,  # > 30% annualized
        }

        # Trend strength thresholds
        self.trend_thresholds = {"weak_trend": 0.3, "strong_trend": 0.7}

        # Correlation thresholds
        self.correlation_thresholds = {
            MarketRegime.LOW_CORRELATION: 0.3,
            MarketRegime.HIGH_CORRELATION: 0.7,
        }

        # State tracking
        self.current_regime = MarketRegime.MEDIUM_VOLATILITY
        self.regime_history: list[RegimeChangeEvent] = []
        self.price_data: dict[str, list[float]] = {}
        self.last_update = datetime.now()

        logger.info(
            "Market regime detector initialized",
            volatility_window=self.volatility_window,
            trend_window=self.trend_window,
        )

    @time_execution
    async def detect_volatility_regime(self, symbol: str, price_data: list[float]) -> MarketRegime:
        """
        Detect volatility regime based on price data.

        Args:
            symbol: Trading symbol
            price_data: List of price values

        Returns:
            MarketRegime: Detected volatility regime
        """
        try:
            if len(price_data) < self.volatility_window:
                logger.warning(
                    "Insufficient data for volatility regime detection",
                    symbol=symbol,
                    data_points=len(price_data),
                )
                return MarketRegime.MEDIUM_VOLATILITY

            # Calculate returns
            returns = np.diff(np.log(price_data))

            # Calculate rolling volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Classify volatility regime
            if volatility < self.volatility_thresholds[MarketRegime.LOW_VOLATILITY]:
                regime = MarketRegime.LOW_VOLATILITY
            elif volatility < self.volatility_thresholds[MarketRegime.MEDIUM_VOLATILITY]:
                regime = MarketRegime.MEDIUM_VOLATILITY
            else:
                regime = MarketRegime.HIGH_VOLATILITY

            logger.debug(
                "Volatility regime detected",
                symbol=symbol,
                volatility=volatility,
                regime=regime.value,
            )

            return regime

        except Exception as e:
            logger.error("Error detecting volatility regime", symbol=symbol, error=str(e))
            raise RiskManagementError(f"Volatility regime detection failed: {e!s}")

    @time_execution
    async def detect_trend_regime(self, symbol: str, price_data: list[float]) -> MarketRegime:
        """
        Detect trend regime based on price data.

        Args:
            symbol: Trading symbol
            price_data: List of price values

        Returns:
            MarketRegime: Detected trend regime
        """
        try:
            if len(price_data) < self.trend_window:
                logger.warning(
                    "Insufficient data for trend regime detection",
                    symbol=symbol,
                    data_points=len(price_data),
                )
                return MarketRegime.RANGING

            # Calculate linear regression slope
            x = np.arange(len(price_data))
            slope, intercept = np.polyfit(x, price_data, 1)

            # Calculate R-squared (trend strength)
            y_pred = slope * x + intercept
            r_squared = 1 - np.sum((price_data - y_pred) ** 2) / np.sum(
                (price_data - np.mean(price_data)) ** 2
            )

            # Classify trend regime
            if r_squared < self.trend_thresholds["weak_trend"]:
                regime = MarketRegime.RANGING
            elif slope > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN

            logger.debug(
                "Trend regime detected",
                symbol=symbol,
                slope=slope,
                r_squared=r_squared,
                regime=regime.value,
            )

            return regime

        except Exception as e:
            logger.error("Error detecting trend regime", symbol=symbol, error=str(e))
            raise RiskManagementError(f"Trend regime detection failed: {e!s}")

    @time_execution
    async def detect_correlation_regime(
        self, symbols: list[str], price_data_dict: dict[str, list[float]]
    ) -> MarketRegime:
        """
        Detect correlation regime across multiple symbols.

        Args:
            symbols: List of trading symbols
            price_data_dict: Dictionary of price data by symbol

        Returns:
            MarketRegime: Detected correlation regime
        """
        try:
            if len(symbols) < 2:
                logger.warning("Insufficient symbols for correlation regime detection")
                return MarketRegime.LOW_CORRELATION

            # Calculate returns for all symbols
            returns_dict = {}
            min_length = float("inf")

            for symbol in symbols:
                if (
                    symbol in price_data_dict
                    and len(price_data_dict[symbol]) >= self.correlation_window
                ):
                    returns = np.diff(np.log(price_data_dict[symbol]))
                    returns_dict[symbol] = returns
                    min_length = min(min_length, len(returns))
                else:
                    logger.warning(f"Insufficient data for symbol {symbol}")
                    return MarketRegime.LOW_CORRELATION

            if min_length < 10:  # Minimum data requirement
                logger.warning("Insufficient data for correlation analysis")
                return MarketRegime.LOW_CORRELATION

            # Align returns to same length
            aligned_returns = {}
            for symbol, returns in returns_dict.items():
                aligned_returns[symbol] = returns[-min_length:]

            # Calculate correlation matrix
            returns_df = pd.DataFrame(aligned_returns)
            correlation_matrix = returns_df.corr()

            # Calculate average correlation (excluding diagonal)
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            avg_correlation = correlation_matrix.values[mask].mean()

            # Classify correlation regime
            if avg_correlation > self.correlation_thresholds[MarketRegime.HIGH_CORRELATION]:
                regime = MarketRegime.HIGH_CORRELATION
            else:
                regime = MarketRegime.LOW_CORRELATION

            logger.debug(
                "Correlation regime detected", avg_correlation=avg_correlation, regime=regime.value
            )

            return regime

        except Exception as e:
            logger.error("Error detecting correlation regime", error=str(e))
            raise RiskManagementError(f"Correlation regime detection failed: {e!s}")

    @time_execution
    async def detect_comprehensive_regime(self, market_data: list[MarketData]) -> MarketRegime:
        """
        Detect comprehensive market regime combining all factors.

        Args:
            market_data: List of market data points

        Returns:
            MarketRegime: Comprehensive regime classification
        """
        try:
            if not market_data:
                logger.warning("No market data provided for regime detection")
                return MarketRegime.MEDIUM_VOLATILITY

            # Group data by symbol
            symbol_data: dict[str, list[float]] = {}
            for data in market_data:
                if data.symbol not in symbol_data:
                    symbol_data[data.symbol] = []
                symbol_data[data.symbol].append(float(data.price))

            # Detect individual regimes
            volatility_regimes = []
            trend_regimes = []

            for symbol, prices in symbol_data.items():
                if len(prices) >= max(self.volatility_window, self.trend_window):
                    vol_regime = await self.detect_volatility_regime(symbol, prices)
                    trend_regime = await self.detect_trend_regime(symbol, prices)
                    volatility_regimes.append(vol_regime)
                    trend_regimes.append(trend_regime)

            # Detect correlation regime
            correlation_regime = await self.detect_correlation_regime(
                list(symbol_data.keys()), symbol_data
            )

            # Combine regimes for comprehensive classification
            comprehensive_regime = self._combine_regimes(
                volatility_regimes, trend_regimes, correlation_regime
            )

            # Check for regime change
            await self._check_regime_change(comprehensive_regime)

            return comprehensive_regime

        except Exception as e:
            logger.error("Error in comprehensive regime detection", error=str(e))
            raise RiskManagementError(f"Comprehensive regime detection failed: {e!s}")

    def _combine_regimes(
        self,
        volatility_regimes: list[MarketRegime],
        trend_regimes: list[MarketRegime],
        correlation_regime: MarketRegime,
    ) -> MarketRegime:
        """
        Combine individual regimes into comprehensive classification.

        Args:
            volatility_regimes: List of volatility regimes
            trend_regimes: List of trend regimes
            correlation_regime: Correlation regime

        Returns:
            MarketRegime: Combined regime classification
        """
        # Count regime frequencies
        vol_counts = {}
        trend_counts = {}

        for regime in volatility_regimes:
            vol_counts[regime] = vol_counts.get(regime, 0) + 1

        for regime in trend_regimes:
            trend_counts[regime] = trend_counts.get(regime, 0) + 1

        # Determine dominant volatility regime
        dominant_vol = (
            max(vol_counts.items(), key=lambda x: x[1])[0]
            if vol_counts
            else MarketRegime.MEDIUM_VOLATILITY
        )

        # Determine dominant trend regime
        dominant_trend = (
            max(trend_counts.items(), key=lambda x: x[1])[0]
            if trend_counts
            else MarketRegime.RANGING
        )

        # Combine for comprehensive classification
        if dominant_vol == MarketRegime.HIGH_VOLATILITY:
            if correlation_regime == MarketRegime.HIGH_CORRELATION:
                return MarketRegime.CRISIS
            else:
                return MarketRegime.HIGH_VOLATILITY
        elif dominant_vol == MarketRegime.LOW_VOLATILITY:
            return MarketRegime.LOW_VOLATILITY
        else:
            return dominant_trend  # Use trend regime for medium volatility

    async def _check_regime_change(self, new_regime: MarketRegime) -> None:
        """
        Check for regime change and trigger alerts.

        Args:
            new_regime: Newly detected regime
        """
        if new_regime != self.current_regime:
            # Calculate change confidence based on historical patterns
            confidence = self._calculate_change_confidence(new_regime)

            if confidence > self.regime_change_threshold:
                event = RegimeChangeEvent(
                    from_regime=self.current_regime,
                    to_regime=new_regime,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    trigger_metrics={
                        "volatility_window": self.volatility_window,
                        "trend_window": self.trend_window,
                        "correlation_window": self.correlation_window,
                    },
                    description=f"Regime change from {self.current_regime.value} to {
                        new_regime.value
                    }",
                )

                self.regime_history.append(event)
                self.current_regime = new_regime

                logger.warning(
                    "Market regime change detected",
                    from_regime=self.current_regime.value,
                    to_regime=new_regime.value,
                    confidence=confidence,
                )

                # TODO: Remove in production - Debug logging
                logger.debug("Regime change event created", event_data=event.model_dump())

    def _calculate_change_confidence(self, new_regime: MarketRegime) -> float:
        """
        Calculate confidence in regime change.

        Args:
            new_regime: Newly detected regime

        Returns:
            float: Confidence level (0-1)
        """
        # Simple confidence calculation based on regime stability
        if len(self.regime_history) < 1:
            return 0.8  # High confidence for first change

        # Check if this is a reversal or continuation
        recent_changes = (
            self.regime_history[-3:] if len(self.regime_history) >= 3 else self.regime_history
        )

        # Higher confidence if this is a new regime type
        if new_regime not in [event.to_regime for event in recent_changes]:
            return 0.9

        # Lower confidence if frequent changes
        if len(recent_changes) >= 2:
            return 0.6

        return 0.8

    def get_regime_history(self, limit: int = 10) -> list[RegimeChangeEvent]:
        """
        Get recent regime change history.

        Args:
            limit: Maximum number of events to return

        Returns:
            List[RegimeChangeEvent]: Recent regime change events
        """
        return self.regime_history[-limit:] if self.regime_history else []

    def get_current_regime(self) -> MarketRegime:
        """
        Get current market regime.

        Returns:
            MarketRegime: Current regime
        """
        return self.current_regime

    def get_regime_statistics(self) -> dict[str, Any]:
        """
        Get regime detection statistics.

        Returns:
            Dict[str, Any]: Statistics about regime detection
        """
        if not self.regime_history:
            return {
                "total_changes": 0,
                "current_regime": self.current_regime.value,
                "regime_duration_hours": 0,
                "last_change": None,
            }

        total_changes = len(self.regime_history)
        regime_duration = (
            datetime.now() - self.regime_history[-1].timestamp
        ).total_seconds() / 3600

        stats = {
            "total_changes": total_changes,
            "current_regime": self.current_regime.value,
            "regime_duration_hours": regime_duration,
            "last_change": self.regime_history[-1].timestamp.isoformat(),
        }

        return stats
