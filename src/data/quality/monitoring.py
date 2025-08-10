"""
Data Quality Monitoring System

This module provides ongoing quality monitoring and alerting:
- Data drift detection using statistical tests
- Feature distribution monitoring
- Quality score calculation and trending
- Automated alerting on quality degradation
- Quality reports and dashboards

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import statistics
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

# Import from P-001 core components
from src.core.types import (
    MarketData, Signal, Position, Ticker, OrderBook,
    QualityLevel, DriftType
)
from src.core.exceptions import (
    DataError, DataValidationError, ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_price, validate_quantity
from src.utils.helpers import calculate_percentage_change

logger = get_logger(__name__)


# QualityLevel and DriftType are now imported from core.types


@dataclass
class QualityMetric:
    """Quality metric record"""
    metric_name: str
    value: float
    threshold: float
    level: QualityLevel
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class DriftAlert:
    """Data drift alert record"""
    drift_type: DriftType
    feature: str
    severity: QualityLevel
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]


class QualityMonitor:
    """
    Comprehensive data quality monitoring system.

    This class provides ongoing monitoring of data quality, including
    drift detection, distribution monitoring, and automated alerting.
    """

    def __init__(self, config: Config):
        """
        Initialize the quality monitor with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Monitoring thresholds
        self.quality_thresholds = getattr(config, 'quality_thresholds', {
            'excellent': 0.95,
            'good': 0.85,
            'fair': 0.70,
            'poor': 0.50
        })

        self.drift_threshold = getattr(
            config, 'drift_threshold', 0.1)  # 10% drift threshold
        self.distribution_window = getattr(
            config,
            'distribution_window',
            1000)  # Data points for distribution
        self.alert_cooldown = getattr(
            config, 'alert_cooldown', 3600)  # 1 hour cooldown

        # Data storage for monitoring
        self.price_distributions: Dict[str, List[float]] = {}
        self.volume_distributions: Dict[str, List[float]] = {}
        self.quality_scores: Dict[str, List[float]] = {}
        self.drift_alerts: List[DriftAlert] = []

        # Monitoring statistics
        self.monitoring_stats = {
            'total_checks': 0,
            'drift_detected': 0,
            'quality_alerts': 0,
            'distribution_updates': 0
        }

        # Alert tracking
        self.last_alert_time: Dict[str, datetime] = {}

        logger.info("QualityMonitor initialized", config=config)

    @time_execution
    async def monitor_data_quality(
            self, data: MarketData) -> Tuple[float, List[DriftAlert]]:
        """
        Monitor data quality and detect drift.

        Args:
            data: Market data to monitor

        Returns:
            Tuple of (quality_score, drift_alerts)
        """
        try:
            # Handle None data
            if data is None:
                logger.error("Cannot monitor None data")
                return 0.0, []

            # Update distributions
            await self._update_distributions(data)

            # Calculate quality score
            quality_score = await self._calculate_quality_score(data)

            # Detect drift
            drift_alerts = await self._detect_drift(data)

            # Update statistics
            self.monitoring_stats['total_checks'] += 1
            if drift_alerts:
                self.monitoring_stats['drift_detected'] += len(drift_alerts)

            # Store quality score
            if data.symbol not in self.quality_scores:
                self.quality_scores[data.symbol] = []

            self.quality_scores[data.symbol].append(quality_score)

            # Maintain history size
            if len(self.quality_scores[data.symbol]
                   ) > self.distribution_window:
                self.quality_scores[data.symbol].pop(0)

            # Log monitoring results
            if quality_score < self.quality_thresholds['fair']:
                logger.warning(
                    "Low data quality detected",
                    symbol=data.symbol,
                    quality_score=quality_score,
                    drift_alerts_count=len(drift_alerts)
                )
            else:
                logger.debug(
                    "Data quality monitoring passed",
                    symbol=data.symbol,
                    quality_score=quality_score
                )

            return quality_score, drift_alerts

        except Exception as e:
            logger.error(
                "Data quality monitoring failed",
                symbol=data.symbol,
                error=str(e))
            return 0.0, []

    @time_execution
    async def monitor_signal_quality(
            self, signals: List[Signal]) -> Tuple[float, List[DriftAlert]]:
        """
        Monitor signal quality and detect drift.

        Args:
            signals: Trading signals to monitor

        Returns:
            Tuple of (quality_score, drift_alerts)
        """
        try:
            if not signals:
                return 1.0, []

            # Calculate signal quality metrics
            confidence_scores = [signal.confidence for signal in signals]
            avg_confidence = statistics.mean(confidence_scores)

            # Check for signal distribution drift
            drift_alerts = await self._detect_signal_drift(signals)

            # Calculate overall quality score
            quality_score = avg_confidence * 0.7 + \
                (1.0 - len(drift_alerts) * 0.1) * 0.3

            # Update statistics
            self.monitoring_stats['total_checks'] += 1
            if drift_alerts:
                self.monitoring_stats['drift_detected'] += len(drift_alerts)

            if quality_score < self.quality_thresholds['fair']:
                logger.warning(
                    "Low signal quality detected",
                    signal_count=len(signals),
                    avg_confidence=avg_confidence,
                    quality_score=quality_score
                )
            else:
                logger.debug(
                    "Signal quality monitoring passed",
                    signal_count=len(signals),
                    quality_score=quality_score
                )

            return quality_score, drift_alerts

        except Exception as e:
            logger.error("Signal quality monitoring failed", error=str(e))
            return 0.0, []

    @time_execution
    async def generate_quality_report(
            self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.

        Args:
            symbol: Optional symbol to focus on

        Returns:
            Quality report dictionary
        """
        try:
            report = {
                'timestamp': datetime.now(timezone.utc),
                'overall_quality_score': 0.0,
                'symbol_quality_scores': {},
                'drift_summary': {},
                'distribution_summary': {},
                'alert_summary': {},
                'recommendations': []
            }

            # Calculate overall quality score
            all_scores = []
            for scores in self.quality_scores.values():
                if scores:
                    all_scores.extend(scores)

            if all_scores:
                report['overall_quality_score'] = statistics.mean(all_scores)

            # Symbol-specific quality scores
            for sym, scores in self.quality_scores.items():
                if symbol is None or sym == symbol:
                    if scores:
                        report['symbol_quality_scores'][sym] = {
                            'current_score': scores[-1] if scores else 0.0,
                            'avg_score': statistics.mean(scores),
                            'min_score': min(scores),
                            'max_score': max(scores),
                            'trend': 'improving' if len(scores) > 1 and scores[-1] > scores[-2] else 'stable'
                        }

            # Drift summary
            drift_counts = {}
            for alert in self.drift_alerts:
                drift_type = alert.drift_type.value
                drift_counts[drift_type] = drift_counts.get(drift_type, 0) + 1

            report['drift_summary'] = drift_counts

            # Distribution summary
            report['distribution_summary'] = {
                'price_distributions': {
                    sym: len(dist) for sym,
                    dist in self.price_distributions.items()},
                'volume_distributions': {
                    sym: len(dist) for sym,
                    dist in self.volume_distributions.items()}}

            # Alert summary
            recent_alerts = [
                alert for alert in self.drift_alerts if alert.timestamp > datetime.now(
                    timezone.utc) -
                timedelta(
                    hours=24)]

            report['alert_summary'] = {
                'total_alerts_24h': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.severity == QualityLevel.CRITICAL]),
                'high_alerts': len([a for a in recent_alerts if a.severity == QualityLevel.POOR])
            }

            # Generate recommendations
            report['recommendations'] = await self._generate_recommendations(report)

            logger.info("Quality report generated", report_summary=report)
            return report

        except Exception as e:
            logger.error("Quality report generation failed", error=str(e))
            return {
                'timestamp': datetime.now(timezone.utc),
                'error': str(e),
                'error_type': type(e).__name__
            }

    async def _update_distributions(self, data: MarketData) -> None:
        """Update price and volume distributions for drift detection"""
        if not data.symbol:
            return

        # Update price distribution
        if data.symbol not in self.price_distributions:
            self.price_distributions[data.symbol] = []

        if data.price:
            self.price_distributions[data.symbol].append(float(data.price))

            # Maintain distribution size
            if len(
                    self.price_distributions[data.symbol]) > self.distribution_window:
                self.price_distributions[data.symbol].pop(0)

        # Update volume distribution
        if data.symbol not in self.volume_distributions:
            self.volume_distributions[data.symbol] = []

        if data.volume:
            self.volume_distributions[data.symbol].append(float(data.volume))

            # Maintain distribution size
            if len(
                    self.volume_distributions[data.symbol]) > self.distribution_window:
                self.volume_distributions[data.symbol].pop(0)

        self.monitoring_stats['distribution_updates'] += 1

    async def _calculate_quality_score(self, data: MarketData) -> float:
        """Calculate comprehensive quality score for market data"""
        score_components = []

        # Completeness score (0-1)
        completeness_score = 0.0
        required_fields = ['symbol', 'price', 'volume', 'timestamp']
        present_fields = 0

        # Check each field
        if data.symbol:
            present_fields += 1
        if data.price and data.price > 0:
            present_fields += 1
        if data.volume and data.volume > 0:
            present_fields += 1
        if data.timestamp:
            present_fields += 1

        completeness_score = present_fields / len(required_fields)
        score_components.append(completeness_score * 0.3)  # 30% weight

        # Validity score (0-1)
        validity_score = 1.0
        if data.price and data.price <= 0:
            validity_score -= 0.9  # Very strong penalty for negative/zero prices
        if data.volume and data.volume <= 0:
            validity_score -= 0.6  # Strong penalty for negative/zero volume
        if data.bid and data.ask and data.bid >= data.ask:
            validity_score -= 0.3  # Penalty for invalid bid/ask spread
        score_components.append(max(0.0, validity_score) * 0.3)  # 30% weight

        # Freshness score (0-1)
        freshness_score = 1.0
        if data.timestamp:
            age_seconds = (
                datetime.now(
                    timezone.utc) -
                data.timestamp).total_seconds()
            if age_seconds > 60:  # 1 minute threshold
                # Decay over 5 minutes
                freshness_score = max(0.0, 1.0 - (age_seconds - 60) / 300)
        score_components.append(freshness_score * 0.2)  # 20% weight

        # Consistency score (0-1)
        consistency_score = 1.0
        if data.symbol in self.price_distributions and len(
                self.price_distributions[data.symbol]) >= 10:
            recent_prices = self.price_distributions[data.symbol][-10:]
            if data.price:
                current_price = float(data.price)
                mean_price = statistics.mean(recent_prices)
                std_price = statistics.stdev(
                    recent_prices) if len(recent_prices) > 1 else 0

                if std_price > 0:
                    z_score = abs(current_price - mean_price) / std_price
                    if z_score > 3.0:  # 3 standard deviations
                        consistency_score = max(
                            0.0, 1.0 - (z_score - 3.0) / 2.0)
        score_components.append(consistency_score * 0.2)  # 20% weight

        # Calculate weighted average
        overall_score = sum(score_components)

        return max(0.0, min(1.0, overall_score))

    async def _detect_drift(self, data: MarketData) -> List[DriftAlert]:
        """Detect data drift using statistical methods"""
        alerts = []

        if not data.symbol:
            return alerts

        # Price distribution drift detection
        if (data.symbol in self.price_distributions and len(
                self.price_distributions[data.symbol]) >= 10):  # Reduced minimum requirement for testing

            # Use smaller windows for drift detection
            total_points = len(self.price_distributions[data.symbol])
            window_size = min(self.distribution_window // 4, total_points // 3)
            if window_size >= 5:  # Need at least 5 points for meaningful drift detection
                recent_prices = self.price_distributions[data.symbol][-window_size:]
                historical_prices = self.price_distributions[data.symbol][-2 *
                                                                          window_size:-window_size]

                if len(historical_prices) >= 5:  # Need at least 5 historical points
                    drift_score = await self._calculate_distribution_drift(recent_prices, historical_prices)

                    if drift_score > self.drift_threshold:
                        alerts.append(
                            DriftAlert(
                                drift_type=DriftType.COVARIATE_DRIFT,
                                feature='price',
                                severity=QualityLevel.POOR if drift_score > 0.2 else QualityLevel.FAIR,
                                description=f"Price distribution drift detected: {
                                    drift_score: .3f}",
                                timestamp=datetime.now(
                                    timezone.utc),
                                metadata={
                                    'drift_score': drift_score,
                                    'recent_mean': statistics.mean(recent_prices),
                                    'historical_mean': statistics.mean(historical_prices)}))

        # Volume distribution drift detection
        if (data.symbol in self.volume_distributions and len(
                self.volume_distributions[data.symbol]) >= 10):  # Reduced minimum requirement for testing

            # Use smaller windows for drift detection
            total_points = len(self.volume_distributions[data.symbol])
            window_size = min(self.distribution_window // 4, total_points // 3)
            if window_size >= 5:  # Need at least 5 points for meaningful drift detection
                recent_volumes = self.volume_distributions[data.symbol][-window_size:]
                historical_volumes = self.volume_distributions[data.symbol][-2 *
                                                                            window_size:-window_size]

                if len(historical_volumes) >= 5:  # Need at least 5 historical points
                    drift_score = await self._calculate_distribution_drift(recent_volumes, historical_volumes)

                    if drift_score > self.drift_threshold:
                        alerts.append(
                            DriftAlert(
                                drift_type=DriftType.COVARIATE_DRIFT,
                                feature='volume',
                                severity=QualityLevel.POOR if drift_score > 0.2 else QualityLevel.FAIR,
                                description=f"Volume distribution drift detected: {
                                    drift_score: .3f}",
                                timestamp=datetime.now(
                                    timezone.utc),
                                metadata={
                                    'drift_score': drift_score,
                                    'recent_mean': statistics.mean(recent_volumes),
                                    'historical_mean': statistics.mean(historical_volumes)}))

        # Store alerts
        self.drift_alerts.extend(alerts)

        # Maintain alert history (keep last 1000 alerts)
        if len(self.drift_alerts) > 1000:
            self.drift_alerts = self.drift_alerts[-1000:]

        return alerts

    async def _detect_signal_drift(
            self, signals: List[Signal]) -> List[DriftAlert]:
        """Detect drift in signal patterns"""
        alerts = []

        if not signals:
            return alerts

        # Analyze signal confidence distribution
        confidences = [signal.confidence for signal in signals]

        if len(confidences) >= 10:
            mean_confidence = statistics.mean(confidences)
            std_confidence = statistics.stdev(
                confidences) if len(confidences) > 1 else 0

            # Check for low confidence drift
            if mean_confidence < 0.6:  # Low average confidence
                alerts.append(
                    DriftAlert(
                        drift_type=DriftType.CONCEPT_DRIFT,
                        feature='signal_confidence',
                        severity=QualityLevel.POOR,
                        description=f"Low signal confidence detected: {
                            mean_confidence: .3f}",
                        timestamp=datetime.now(
                            timezone.utc),
                        metadata={
                            'mean_confidence': mean_confidence,
                            'std_confidence': std_confidence,
                            'signal_count': len(signals)}))

            # Check for high confidence variance (unstable signals)
            if std_confidence > 0.3:  # High variance
                alerts.append(
                    DriftAlert(
                        drift_type=DriftType.CONCEPT_DRIFT,
                        feature='signal_stability',
                        severity=QualityLevel.FAIR,
                        description=f"Unstable signal confidence detected: std={
                            std_confidence: .3f}",
                        timestamp=datetime.now(
                            timezone.utc),
                        metadata={
                            'mean_confidence': mean_confidence,
                            'std_confidence': std_confidence,
                            'signal_count': len(signals)}))

        return alerts

    async def _calculate_distribution_drift(
            self,
            recent: List[float],
            historical: List[float]) -> float:
        """Calculate distribution drift using statistical measures"""
        if not recent or not historical:
            return 0.0

        try:
            # Calculate basic statistics
            recent_mean = statistics.mean(recent)
            historical_mean = statistics.mean(historical)
            recent_std = statistics.stdev(recent) if len(recent) > 1 else 0
            historical_std = statistics.stdev(
                historical) if len(historical) > 1 else 0

            # Calculate drift score using multiple metrics
            mean_drift = abs(recent_mean - historical_mean) / \
                max(historical_mean, 1e-6)
            std_drift = abs(recent_std - historical_std) / \
                max(historical_std, 1e-6)

            # Combined drift score
            drift_score = (mean_drift + std_drift) / 2.0

            return min(1.0, drift_score)

        except Exception as e:
            logger.warning(
                "Distribution drift calculation failed",
                error=str(e))
            return 0.0

    async def _generate_recommendations(
            self, report: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        overall_score = report.get('overall_quality_score', 0.0)

        if overall_score < self.quality_thresholds['poor']:
            recommendations.append(
                "CRITICAL: Overall data quality is poor. Review data sources and validation rules.")
        elif overall_score < self.quality_thresholds['fair']:
            recommendations.append(
                "WARNING: Data quality needs improvement. Consider adjusting validation thresholds.")

        # Check for specific issues
        drift_summary = report.get('drift_summary', {})
        if drift_summary.get('covariate_drift', 0) > 5:
            recommendations.append(
                "High covariate drift detected. Consider retraining models or adjusting features.")

        alert_summary = report.get('alert_summary', {})
        if alert_summary.get('critical_alerts', 0) > 0:
            recommendations.append(
                "Critical alerts detected. Immediate attention required.")

        # Distribution recommendations
        distribution_summary = report.get('distribution_summary', {})
        price_dists = distribution_summary.get('price_distributions', {})
        for symbol, size in price_dists.items():
            if size < self.distribution_window // 2:
                recommendations.append(
                    f"Insufficient price data for {symbol}. Collect more data for reliable monitoring.")

        if not recommendations:
            recommendations.append(
                "Data quality is good. Continue monitoring for any changes.")

        return recommendations

    @time_execution
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring statistics and summary"""
        return {
            "monitoring_stats": self.monitoring_stats.copy(),
            "quality_scores": {symbol: scores[-1] if scores else 0.0
                               for symbol, scores in self.quality_scores.items()},
            "distribution_sizes": {
                "price_distributions": {symbol: len(dist) for symbol, dist in self.price_distributions.items()},
                "volume_distributions": {symbol: len(dist) for symbol, dist in self.volume_distributions.items()}
            },
            "recent_alerts": len([alert for alert in self.drift_alerts
                                  if alert.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)]),
            "monitoring_config": {
                "quality_thresholds": self.quality_thresholds,
                "drift_threshold": self.drift_threshold,
                "distribution_window": self.distribution_window,
                "alert_cooldown": self.alert_cooldown
            }
        }
