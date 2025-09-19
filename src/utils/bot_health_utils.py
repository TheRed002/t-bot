"""
Shared health check utilities for bot management module.

Extracted from duplicated code in BotMonitor, BotHealthService, and other monitoring services.
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from decimal import Decimal

from src.core.logging import get_logger
from src.core.types import BotMetrics, BotStatus

logger = get_logger(__name__)


class HealthCheckUtils:
    """Utilities for bot health checking and scoring."""

    @staticmethod
    def calculate_health_score(bot_status: BotStatus, metrics: BotMetrics | None = None) -> float:
        """
        Calculate standardized health score for a bot.

        Args:
            bot_status: Current bot status
            metrics: Optional bot metrics for detailed scoring

        Returns:
            Health score between 0.0 and 1.0
        """
        try:
            base_score = HealthCheckUtils._calculate_status_score(bot_status)

            if metrics is None:
                return base_score

            # Factor in metrics-based scores
            resource_score = HealthCheckUtils._calculate_resource_score(metrics)
            error_score = HealthCheckUtils._calculate_error_rate_score(metrics)
            performance_score = HealthCheckUtils._calculate_performance_score(metrics)

            # Weighted average
            weights = {'status': 0.4, 'resource': 0.2, 'error': 0.2, 'performance': 0.2}
            weighted_score = (
                base_score * weights['status'] +
                resource_score * weights['resource'] +
                error_score * weights['error'] +
                performance_score * weights['performance']
            )

            return max(0.0, min(1.0, weighted_score))

        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.5  # Default neutral score

    @staticmethod
    def _calculate_status_score(bot_status: BotStatus) -> float:
        """Calculate score based on bot status."""
        status_scores = {
            BotStatus.RUNNING: 1.0,
            BotStatus.STARTING: 0.7,
            BotStatus.PAUSED: 0.6,
            BotStatus.STOPPING: 0.4,
            BotStatus.STOPPED: 0.3,
            BotStatus.ERROR: 0.1,
            BotStatus.DEAD: 0.0
        }
        return status_scores.get(bot_status, 0.5)

    @staticmethod
    def _calculate_resource_score(metrics: BotMetrics) -> float:
        """Calculate score based on resource usage."""
        try:
            cpu_score = max(0, 1.0 - (metrics.cpu_usage / 100.0))
            memory_score = max(0, 1.0 - (metrics.memory_usage / 100.0))
            return (cpu_score + memory_score) / 2.0
        except (AttributeError, TypeError, ZeroDivisionError):
            return 0.8  # Default good score if metrics unavailable

    @staticmethod
    def _calculate_error_rate_score(metrics: BotMetrics) -> float:
        """Calculate score based on error rate."""
        try:
            if metrics.total_executions == 0:
                return 1.0  # Perfect score for no executions

            error_rate = metrics.error_count / metrics.total_executions
            return max(0, 1.0 - (error_rate * 2))  # Penalize errors heavily
        except (AttributeError, TypeError, ZeroDivisionError):
            return 0.8  # Default good score

    @staticmethod
    def _calculate_performance_score(metrics: BotMetrics) -> float:
        """Calculate score based on performance metrics."""
        try:
            # Use win rate as primary performance indicator
            if hasattr(metrics, 'win_rate') and metrics.win_rate is not None:
                return float(metrics.win_rate) / 100.0

            # Fallback to trade success rate
            if metrics.total_executions > 0:
                success_rate = (metrics.total_executions - metrics.error_count) / metrics.total_executions
                return success_rate

            return 0.8  # Default neutral score
        except (AttributeError, TypeError, ZeroDivisionError):
            return 0.8


class HealthAnalyzer:
    """Health analysis utilities extracted from multiple services."""

    @staticmethod
    def analyze_performance_health(metrics_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze performance health from metrics history."""
        if not metrics_history:
            return {
                'score': 0.5,
                'trend': 'stable',
                'issues': ['No metrics available'],
                'recommendations': ['Start collecting metrics']
            }

        try:
            recent_metrics = metrics_history[-10:]  # Last 10 data points
            performance_scores = []
            pnl_values = []

            for metric in recent_metrics:
                if 'total_pnl' in metric:
                    pnl_values.append(float(metric['total_pnl']))

                # Calculate performance score for this metric
                win_rate = metric.get('win_rate', 50)
                trades_count = metric.get('total_executions', 0)

                if trades_count > 0:
                    score = (win_rate / 100.0) * min(1.0, trades_count / 10)
                    performance_scores.append(score)

            if not performance_scores:
                avg_score = 0.5
            else:
                avg_score = sum(performance_scores) / len(performance_scores)

            # Analyze trend
            trend = HealthAnalyzer._calculate_trend(performance_scores) if len(performance_scores) > 1 else 'stable'

            # Identify issues
            issues = []
            recommendations = []

            if avg_score < 0.3:
                issues.append('Poor performance')
                recommendations.append('Review trading strategy')

            if pnl_values and len(pnl_values) > 1:
                if pnl_values[-1] < pnl_values[0]:
                    issues.append('Declining PnL')
                    recommendations.append('Analyze recent trades for patterns')

            return {
                'score': avg_score,
                'trend': trend,
                'issues': issues,
                'recommendations': recommendations,
                'avg_pnl': sum(pnl_values) / len(pnl_values) if pnl_values else 0,
                'pnl_trend': HealthAnalyzer._calculate_trend(pnl_values) if len(pnl_values) > 1 else 'stable'
            }

        except Exception as e:
            logger.error(f"Error analyzing performance health: {e}")
            return {
                'score': 0.5,
                'trend': 'unknown',
                'issues': [f'Analysis error: {str(e)}'],
                'recommendations': ['Check metrics data quality']
            }

    @staticmethod
    def analyze_resource_health(metrics_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze resource health from metrics history."""
        if not metrics_history:
            return {
                'score': 0.8,
                'issues': [],
                'recommendations': []
            }

        try:
            recent_metrics = metrics_history[-5:]  # Last 5 data points

            cpu_values = []
            memory_values = []

            for metric in recent_metrics:
                cpu_values.append(metric.get('cpu_usage', 0))
                memory_values.append(metric.get('memory_usage', 0))

            avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
            avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0

            issues = []
            recommendations = []

            if avg_cpu > 80:
                issues.append('High CPU usage')
                recommendations.append('Optimize trading algorithms')

            if avg_memory > 80:
                issues.append('High memory usage')
                recommendations.append('Check for memory leaks')

            # Calculate score
            cpu_score = max(0, 1.0 - (avg_cpu / 100.0))
            memory_score = max(0, 1.0 - (avg_memory / 100.0))
            score = (cpu_score + memory_score) / 2.0

            return {
                'score': score,
                'avg_cpu': avg_cpu,
                'avg_memory': avg_memory,
                'issues': issues,
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Error analyzing resource health: {e}")
            return {
                'score': 0.5,
                'issues': [f'Analysis error: {str(e)}'],
                'recommendations': ['Check resource monitoring']
            }

    @staticmethod
    def _calculate_trend(values: list[float]) -> str:
        """Calculate trend from a series of values."""
        if len(values) < 2:
            return 'stable'

        try:
            # Simple linear trend calculation
            n = len(values)
            x_sum = sum(range(n))
            y_sum = sum(values)
            xy_sum = sum(i * values[i] for i in range(n))
            x2_sum = sum(i ** 2 for i in range(n))

            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum ** 2)

            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'
            else:
                return 'stable'

        except (ZeroDivisionError, TypeError):
            return 'stable'


class AlertGenerator:
    """Generate standardized alerts from health analysis."""

    @staticmethod
    def generate_health_alerts(bot_id: str, health_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate alerts based on health analysis results."""
        alerts = []

        try:
            overall_score = health_results.get('overall_score', 0.5)

            # Critical health alert
            if overall_score < 0.2:
                alerts.append({
                    'bot_id': bot_id,
                    'type': 'critical_health',
                    'severity': 'critical',
                    'message': f'Bot {bot_id} health is critical (score: {overall_score:.2f})',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'data': health_results
                })

            # Poor performance alert
            elif overall_score < 0.4:
                alerts.append({
                    'bot_id': bot_id,
                    'type': 'poor_health',
                    'severity': 'high',
                    'message': f'Bot {bot_id} health is poor (score: {overall_score:.2f})',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'data': health_results
                })

            # Check for specific component issues
            for component, data in health_results.get('components', {}).items():
                if isinstance(data, dict) and data.get('score', 1.0) < 0.3:
                    alerts.append({
                        'bot_id': bot_id,
                        'type': f'{component}_issue',
                        'severity': 'medium',
                        'message': f'Bot {bot_id} {component} health is poor',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'data': data
                    })

            return alerts

        except Exception as e:
            logger.error(f"Error generating health alerts for bot {bot_id}: {e}")
            return []

    @staticmethod
    def generate_resource_alerts(bot_id: str, resource_analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate alerts for resource usage issues."""
        alerts = []

        try:
            cpu_usage = resource_analysis.get('avg_cpu', 0)
            memory_usage = resource_analysis.get('avg_memory', 0)

            if cpu_usage > 90:
                alerts.append({
                    'bot_id': bot_id,
                    'type': 'high_cpu',
                    'severity': 'high',
                    'message': f'Bot {bot_id} CPU usage is very high: {cpu_usage:.1f}%',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            if memory_usage > 90:
                alerts.append({
                    'bot_id': bot_id,
                    'type': 'high_memory',
                    'severity': 'high',
                    'message': f'Bot {bot_id} memory usage is very high: {memory_usage:.1f}%',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            return alerts

        except Exception as e:
            logger.error(f"Error generating resource alerts for bot {bot_id}: {e}")
            return []