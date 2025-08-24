"""
Bot Health Service for comprehensive health monitoring and analysis.

This service provides advanced health monitoring capabilities for bot instances,
including predictive health analysis, trend detection, and automated remediation.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError


class BotHealthService(BaseService):
    """
    Advanced bot health monitoring and analysis service.

    This service provides:
    - Comprehensive health scoring and analysis
    - Predictive health alerts based on trends
    - Health history tracking and analysis
    - Automated remediation recommendations
    - Cross-bot health comparison and benchmarking
    """

    def __init__(self):
        """
        Initialize bot health service.

        Note: Dependencies are resolved during startup via dependency injection.
        """
        super().__init__(name="BotHealthService")

        # Declare required service dependencies for DI resolution
        self.add_dependency("ConfigService")
        self.add_dependency("DatabaseService")

        # Service instances (resolved during startup via DI)
        self._config_service = None
        self._database_service = None

        # Health tracking
        self._bot_health_history: dict[str, list[dict[str, Any]]] = {}
        self._health_baselines: dict[str, dict[str, float]] = {}
        self._health_trends: dict[str, dict[str, Any]] = {}

        # Health monitoring task
        self._health_monitoring_task: asyncio.Task | None = None
        self._health_analysis_interval = 60  # seconds

        # Health scoring weights
        self._health_weights = {
            "performance": 0.3,
            "stability": 0.25,
            "resource_usage": 0.2,
            "error_rate": 0.15,
            "connectivity": 0.1,
        }

        # Health thresholds
        self._health_thresholds = {
            "critical": 0.3,
            "warning": 0.6,
            "healthy": 0.8,
        }

        self._logger.info("BotHealthService initialized")

    async def _do_start(self) -> None:
        """Start the health service and resolve dependencies."""
        # Resolve service dependencies through DI container
        self._config_service = self.resolve_dependency("ConfigService")
        self._database_service = self.resolve_dependency("DatabaseService")

        # Verify critical dependencies are available
        if not self._config_service or not self._database_service:
            raise ServiceError("Failed to resolve required service dependencies")

        # Load configuration
        await self._load_configuration()

        # Start health monitoring task
        self._health_monitoring_task = asyncio.create_task(self._health_analysis_loop())

        self._logger.info("BotHealthService started successfully")

    async def _do_stop(self) -> None:
        """Stop the health service."""
        # Cancel health monitoring task
        if self._health_monitoring_task and not self._health_monitoring_task.done():
            self._health_monitoring_task.cancel()
            try:
                await self._health_monitoring_task
            except asyncio.CancelledError:
                pass

        # Clear tracking
        self._bot_health_history.clear()
        self._health_baselines.clear()
        self._health_trends.clear()

        self._logger.info("BotHealthService stopped successfully")

    async def _load_configuration(self) -> None:
        """Load health service configuration."""
        try:
            config = self._config_service.get_config().get("bot_health", {})

            self._health_analysis_interval = config.get("health_analysis_interval", 60)

            # Update health weights from config
            weights = config.get("health_weights", {})
            self._health_weights.update(weights)

            # Update thresholds from config
            thresholds = config.get("health_thresholds", {})
            self._health_thresholds.update(thresholds)

            self._logger.debug("Health service configuration loaded")

        except Exception as e:
            self._logger.warning(f"Failed to load configuration, using defaults: {e}")

    # Core Health Analysis

    async def analyze_bot_health(self, bot_id: str) -> dict[str, Any]:
        """
        Perform comprehensive health analysis for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            dict: Comprehensive health analysis results
        """
        return await self.execute_with_monitoring(
            "analyze_bot_health", self._analyze_bot_health_impl, bot_id
        )

    async def _analyze_bot_health_impl(self, bot_id: str) -> dict[str, Any]:
        """Implementation of comprehensive health analysis."""
        current_time = datetime.now(timezone.utc)

        # Get recent health data
        health_checks = await self._database_service.get_bot_health_checks(bot_id, limit=10)
        metrics = await self._database_service.get_bot_metrics(bot_id, limit=10)

        if not health_checks and not metrics:
            return {
                "bot_id": bot_id,
                "status": "insufficient_data",
                "health_score": 0.0,
                "message": "Insufficient health data for analysis",
            }

        # Calculate health components
        health_components = await asyncio.gather(
            self._analyze_performance_health(bot_id, metrics),
            self._analyze_stability_health(bot_id, health_checks),
            self._analyze_resource_health(bot_id, metrics),
            self._analyze_error_health(bot_id, metrics),
            self._analyze_connectivity_health(bot_id, health_checks),
            return_exceptions=True,
        )

        component_names = [
            "performance",
            "stability",
            "resource_usage",
            "error_rate",
            "connectivity",
        ]

        # Calculate weighted health score
        total_score = 0.0
        total_weight = 0.0
        component_scores = {}

        for i, component_name in enumerate(component_names):
            component_result = health_components[i]

            if isinstance(component_result, Exception):
                self._logger.warning(f"Health component analysis failed: {component_result}")
                component_scores[component_name] = {"score": 0.0, "error": str(component_result)}
            else:
                weight = self._health_weights.get(component_name, 0.2)
                score = component_result.get("score", 0.0)

                total_score += score * weight
                total_weight += weight
                component_scores[component_name] = component_result

        overall_health_score = total_score / total_weight if total_weight > 0 else 0.0

        # Determine health status
        if overall_health_score >= self._health_thresholds["healthy"]:
            health_status = "healthy"
        elif overall_health_score >= self._health_thresholds["warning"]:
            health_status = "warning"
        elif overall_health_score >= self._health_thresholds["critical"]:
            health_status = "critical"
        else:
            health_status = "unhealthy"

        # Generate recommendations
        recommendations = await self._generate_health_recommendations(bot_id, component_scores)

        # Detect trends
        trends = await self._analyze_health_trends(bot_id, overall_health_score)

        # Create health analysis result
        health_analysis = {
            "bot_id": bot_id,
            "timestamp": current_time.isoformat(),
            "overall_health_score": overall_health_score,
            "health_status": health_status,
            "component_scores": component_scores,
            "recommendations": recommendations,
            "trends": trends,
            "analysis_id": str(uuid4()),
        }

        # Store health analysis
        await self._store_health_analysis(health_analysis)

        # Update health history
        await self._update_health_history(bot_id, health_analysis)

        return health_analysis

    async def _analyze_performance_health(
        self, bot_id: str, metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze trading performance health."""
        if not metrics:
            return {"score": 0.5, "status": "no_data", "details": {}}

        recent_metrics = metrics[:5]  # Last 5 metrics

        # Calculate performance indicators
        total_pnl = sum(m.get("total_pnl", 0) for m in recent_metrics)
        avg_win_rate = sum(m.get("win_rate", 0) for m in recent_metrics) / len(recent_metrics)
        avg_trades_per_day = sum(m.get("total_trades", 0) for m in recent_metrics) / len(
            recent_metrics
        )

        # Score components
        pnl_score = min(1.0, max(0.0, (total_pnl + 1000) / 2000))  # -1000 to +1000 range
        win_rate_score = avg_win_rate  # Already 0-1 range
        activity_score = min(1.0, avg_trades_per_day / 10)  # Normalize to 10 trades/day

        # Weighted performance score
        performance_score = (pnl_score * 0.5) + (win_rate_score * 0.3) + (activity_score * 0.2)

        return {
            "score": performance_score,
            "details": {
                "total_pnl": total_pnl,
                "avg_win_rate": avg_win_rate,
                "avg_trades_per_day": avg_trades_per_day,
                "pnl_score": pnl_score,
                "win_rate_score": win_rate_score,
                "activity_score": activity_score,
            },
            "issues": self._identify_performance_issues(
                total_pnl, avg_win_rate, avg_trades_per_day
            ),
        }

    async def _analyze_stability_health(
        self, bot_id: str, health_checks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze bot stability health."""
        if not health_checks:
            return {"score": 0.5, "status": "no_data", "details": {}}

        recent_checks = health_checks[:10]  # Last 10 health checks

        # Calculate stability indicators
        successful_checks = sum(
            1 for check in recent_checks if check.get("overall_health") == "healthy"
        )
        uptime_percentage = successful_checks / len(recent_checks)

        # Check for consecutive failures
        consecutive_failures = 0
        for check in recent_checks:
            if check.get("overall_health") in ["critical", "warning"]:
                consecutive_failures += 1
            else:
                break

        # Score stability
        uptime_score = uptime_percentage
        stability_penalty = min(0.5, consecutive_failures * 0.1)  # Penalty for consecutive failures
        stability_score = max(0.0, uptime_score - stability_penalty)

        return {
            "score": stability_score,
            "details": {
                "uptime_percentage": uptime_percentage,
                "successful_checks": successful_checks,
                "total_checks": len(recent_checks),
                "consecutive_failures": consecutive_failures,
            },
            "issues": self._identify_stability_issues(uptime_percentage, consecutive_failures),
        }

    async def _analyze_resource_health(
        self, bot_id: str, metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze resource usage health."""
        if not metrics:
            return {"score": 0.5, "status": "no_data", "details": {}}

        recent_metrics = metrics[:5]  # Last 5 metrics

        # Calculate resource usage indicators
        avg_cpu_usage = sum(m.get("cpu_usage", 0) for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.get("memory_usage", 0) for m in recent_metrics) / len(
            recent_metrics
        )

        # Score resource usage (lower usage = better score)
        cpu_score = max(0.0, 1.0 - (avg_cpu_usage / 100))  # Normalize CPU percentage
        memory_score = max(0.0, 1.0 - (avg_memory_usage / 1000))  # Normalize memory MB

        # Weighted resource score
        resource_score = (cpu_score * 0.6) + (memory_score * 0.4)

        return {
            "score": resource_score,
            "details": {
                "avg_cpu_usage": avg_cpu_usage,
                "avg_memory_usage": avg_memory_usage,
                "cpu_score": cpu_score,
                "memory_score": memory_score,
            },
            "issues": self._identify_resource_issues(avg_cpu_usage, avg_memory_usage),
        }

    async def _analyze_error_health(
        self, bot_id: str, metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze error rate health."""
        if not metrics:
            return {"score": 0.5, "status": "no_data", "details": {}}

        recent_metrics = metrics[:5]  # Last 5 metrics

        # Calculate error indicators
        total_trades = sum(m.get("total_trades", 0) for m in recent_metrics)
        total_errors = sum(m.get("error_count", 0) for m in recent_metrics)

        error_rate = total_errors / total_trades if total_trades > 0 else 0

        # Score error rate (lower error rate = better score)
        error_score = max(0.0, 1.0 - (error_rate * 2))  # Scale error rate

        return {
            "score": error_score,
            "details": {
                "total_trades": total_trades,
                "total_errors": total_errors,
                "error_rate": error_rate,
            },
            "issues": self._identify_error_issues(error_rate, total_errors),
        }

    async def _analyze_connectivity_health(
        self, bot_id: str, health_checks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze connectivity health."""
        if not health_checks:
            return {"score": 0.5, "status": "no_data", "details": {}}

        recent_checks = health_checks[:5]  # Last 5 health checks

        # Check connectivity issues in health checks
        connectivity_issues = 0
        heartbeat_issues = 0

        for check in recent_checks:
            checks = check.get("checks", {})

            if checks.get("heartbeat", {}).get("score", 1.0) < 0.5:
                heartbeat_issues += 1

            # Check for connectivity-related issues
            issues = check.get("issues", [])
            for issue in issues:
                if any(
                    keyword in issue.lower() for keyword in ["connection", "network", "timeout"]
                ):
                    connectivity_issues += 1
                    break

        # Score connectivity
        heartbeat_score = max(0.0, 1.0 - (heartbeat_issues / len(recent_checks)))
        connectivity_score = max(0.0, 1.0 - (connectivity_issues / len(recent_checks)))

        overall_connectivity_score = (heartbeat_score * 0.7) + (connectivity_score * 0.3)

        return {
            "score": overall_connectivity_score,
            "details": {
                "heartbeat_issues": heartbeat_issues,
                "connectivity_issues": connectivity_issues,
                "total_checks": len(recent_checks),
                "heartbeat_score": heartbeat_score,
                "connectivity_score": connectivity_score,
            },
            "issues": self._identify_connectivity_issues(heartbeat_issues, connectivity_issues),
        }

    # Health Recommendations

    async def _generate_health_recommendations(
        self, bot_id: str, component_scores: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate health improvement recommendations."""
        recommendations = []

        for component_name, component_data in component_scores.items():
            if isinstance(component_data, dict):
                score = component_data.get("score", 1.0)
                issues = component_data.get("issues", [])

                if score < 0.6:  # Poor health component
                    recommendation = await self._get_component_recommendation(
                        component_name, component_data
                    )
                    if recommendation:
                        recommendations.append(recommendation)

                # Add specific issue recommendations
                for issue in issues:
                    issue_recommendation = await self._get_issue_recommendation(issue)
                    if issue_recommendation:
                        recommendations.append(issue_recommendation)

        return recommendations

    async def _get_component_recommendation(
        self, component_name: str, component_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Get recommendation for a specific health component."""
        recommendations_map = {
            "performance": {
                "title": "Improve Trading Performance",
                "actions": [
                    "Review and optimize trading strategy parameters",
                    "Analyze market conditions and adjust strategy",
                    "Consider reducing position size if experiencing losses",
                ],
                "priority": "high",
            },
            "stability": {
                "title": "Improve Bot Stability",
                "actions": [
                    "Check for memory leaks or resource issues",
                    "Review error logs for recurring issues",
                    "Consider restarting the bot if instability persists",
                ],
                "priority": "critical",
            },
            "resource_usage": {
                "title": "Optimize Resource Usage",
                "actions": [
                    "Monitor CPU and memory usage patterns",
                    "Optimize algorithm efficiency",
                    "Consider scaling resources if needed",
                ],
                "priority": "medium",
            },
            "error_rate": {
                "title": "Reduce Error Rate",
                "actions": [
                    "Investigate and fix recurring errors",
                    "Improve error handling and recovery",
                    "Review API usage and rate limits",
                ],
                "priority": "high",
            },
            "connectivity": {
                "title": "Improve Connectivity",
                "actions": [
                    "Check network connectivity and stability",
                    "Review API endpoint health",
                    "Consider implementing connection retry logic",
                ],
                "priority": "high",
            },
        }

        return recommendations_map.get(component_name)

    async def _get_issue_recommendation(self, issue: str) -> dict[str, Any] | None:
        """Get recommendation for a specific issue."""
        # Issue-specific recommendations
        issue_lower = issue.lower()

        if "cpu" in issue_lower and "high" in issue_lower:
            return {
                "title": "High CPU Usage Detected",
                "actions": [
                    "Optimize algorithm efficiency",
                    "Reduce trading frequency if possible",
                    "Check for infinite loops or inefficient operations",
                ],
                "priority": "medium",
            }
        elif "memory" in issue_lower and ("high" in issue_lower or "leak" in issue_lower):
            return {
                "title": "Memory Issues Detected",
                "actions": [
                    "Check for memory leaks",
                    "Optimize data structures and cleanup",
                    "Consider restarting the bot",
                ],
                "priority": "high",
            }
        elif "error" in issue_lower and "rate" in issue_lower:
            return {
                "title": "High Error Rate",
                "actions": [
                    "Review recent error logs",
                    "Check API rate limits and usage",
                    "Improve error handling logic",
                ],
                "priority": "high",
            }

        return None

    # Health Trends Analysis

    async def _analyze_health_trends(
        self, bot_id: str, current_health_score: float
    ) -> dict[str, Any]:
        """Analyze health trends for predictive insights."""
        if bot_id not in self._bot_health_history:
            return {"trend": "insufficient_data", "prediction": "unknown"}

        history = self._bot_health_history[bot_id]
        if len(history) < 3:
            return {"trend": "insufficient_data", "prediction": "unknown"}

        # Get recent health scores
        recent_scores = [h.get("overall_health_score", 0) for h in history[-10:]]
        recent_scores.append(current_health_score)

        # Calculate trend
        if len(recent_scores) >= 3:
            trend_slope = self._calculate_trend_slope(recent_scores)

            if trend_slope > 0.05:  # Improving
                trend = "improving"
                prediction = "health_likely_to_improve"
            elif trend_slope < -0.05:  # Declining
                trend = "declining"
                prediction = "health_likely_to_decline"
            else:  # Stable
                trend = "stable"
                prediction = "health_stable"
        else:
            trend = "unknown"
            prediction = "insufficient_data"

        # Calculate trend confidence
        variance = self._calculate_variance(recent_scores)
        confidence = max(0.0, 1.0 - variance)  # Lower variance = higher confidence

        return {
            "trend": trend,
            "prediction": prediction,
            "trend_slope": trend_slope if "trend_slope" in locals() else 0.0,
            "confidence": confidence,
            "recent_scores": recent_scores[-5:],  # Last 5 scores
        }

    # Health History and Baselines

    async def _store_health_analysis(self, health_analysis: dict[str, Any]) -> None:
        """Store health analysis in database."""
        try:
            await self._database_service.store_bot_health_analysis(health_analysis)
        except Exception as e:
            self._logger.warning(f"Failed to store health analysis: {e}")

    async def _update_health_history(self, bot_id: str, health_analysis: dict[str, Any]) -> None:
        """Update bot health history."""
        if bot_id not in self._bot_health_history:
            self._bot_health_history[bot_id] = []

        self._bot_health_history[bot_id].append(health_analysis)

        # Keep only recent history (limit to prevent memory bloat)
        if len(self._bot_health_history[bot_id]) > 100:
            self._bot_health_history[bot_id] = self._bot_health_history[bot_id][-100:]

    async def get_bot_health_history(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]:
        """Get bot health history."""
        return await self.execute_with_monitoring(
            "get_bot_health_history", self._get_bot_health_history_impl, bot_id, hours
        )

    async def _get_bot_health_history_impl(self, bot_id: str, hours: int) -> list[dict[str, Any]]:
        """Implementation of get bot health history."""
        # Get from database first
        db_history = await self._database_service.get_bot_health_analyses(bot_id, hours)

        # Supplement with in-memory history if needed
        if bot_id in self._bot_health_history:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            memory_history = [
                h
                for h in self._bot_health_history[bot_id]
                if datetime.fromisoformat(h.get("timestamp", "")) > cutoff_time
            ]

            # Combine and deduplicate
            combined_history = db_history + memory_history

            # Remove duplicates based on analysis_id
            seen_ids = set()
            unique_history = []
            for h in combined_history:
                analysis_id = h.get("analysis_id")
                if analysis_id and analysis_id not in seen_ids:
                    seen_ids.add(analysis_id)
                    unique_history.append(h)

            return sorted(unique_history, key=lambda x: x.get("timestamp", ""), reverse=True)

        return db_history

    # Cross-Bot Health Comparison

    async def compare_bot_health(self) -> dict[str, Any]:
        """Compare health across all monitored bots."""
        return await self.execute_with_monitoring(
            "compare_bot_health", self._compare_bot_health_impl
        )

    async def _compare_bot_health_impl(self) -> dict[str, Any]:
        """Implementation of cross-bot health comparison."""
        # Get recent health analyses for all bots
        all_bot_analyses = await self._database_service.get_recent_health_analyses(hours=1)

        if not all_bot_analyses:
            return {
                "comparison_status": "no_data",
                "bot_rankings": [],
                "health_distribution": {},
                "recommendations": [],
            }

        # Group by bot_id and get latest analysis for each
        latest_analyses = {}
        for analysis in all_bot_analyses:
            bot_id = analysis["bot_id"]
            timestamp = analysis.get("timestamp", "")

            if bot_id not in latest_analyses or timestamp > latest_analyses[bot_id].get(
                "timestamp", ""
            ):
                latest_analyses[bot_id] = analysis

        # Rank bots by health score
        bot_rankings = sorted(
            latest_analyses.values(), key=lambda x: x.get("overall_health_score", 0), reverse=True
        )

        # Calculate health distribution
        health_distribution = {"healthy": 0, "warning": 0, "critical": 0, "unhealthy": 0}
        for analysis in latest_analyses.values():
            status = analysis.get("health_status", "unknown")
            if status in health_distribution:
                health_distribution[status] += 1

        # Generate system-wide recommendations
        system_recommendations = await self._generate_system_recommendations(latest_analyses)

        return {
            "comparison_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_bots_analyzed": len(latest_analyses),
            "bot_rankings": [
                {
                    "bot_id": analysis["bot_id"],
                    "health_score": analysis.get("overall_health_score", 0),
                    "health_status": analysis.get("health_status", "unknown"),
                    "last_analysis": analysis.get("timestamp", ""),
                }
                for analysis in bot_rankings
            ],
            "health_distribution": health_distribution,
            "system_recommendations": system_recommendations,
            "best_performer": bot_rankings[0]["bot_id"] if bot_rankings else None,
            "worst_performer": bot_rankings[-1]["bot_id"] if bot_rankings else None,
        }

    async def _generate_system_recommendations(
        self, latest_analyses: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate system-wide health recommendations."""
        recommendations = []

        # Analyze overall system health
        total_bots = len(latest_analyses)
        unhealthy_count = sum(
            1
            for analysis in latest_analyses.values()
            if analysis.get("overall_health_score", 0) < 0.6
        )

        if unhealthy_count > total_bots * 0.3:  # More than 30% unhealthy
            recommendations.append(
                {
                    "type": "system_alert",
                    "priority": "critical",
                    "title": "High Number of Unhealthy Bots",
                    "description": f"{unhealthy_count} of {total_bots} bots are unhealthy",
                    "actions": [
                        "Investigate common issues affecting multiple bots",
                        "Check system-wide resources and connectivity",
                        "Consider suspending unhealthy bots temporarily",
                    ],
                }
            )

        # Check for common issues
        common_issues = {}
        for analysis in latest_analyses.values():
            for component_name, component_data in analysis.get("component_scores", {}).items():
                if isinstance(component_data, dict) and component_data.get("score", 1.0) < 0.6:
                    common_issues[component_name] = common_issues.get(component_name, 0) + 1

        for issue_type, count in common_issues.items():
            if count > total_bots * 0.2:  # More than 20% affected
                recommendations.append(
                    {
                        "type": "common_issue",
                        "priority": "high",
                        "title": f"Common {issue_type.replace('_', ' ').title()} Issues",
                        "description": f"{count} bots affected by {issue_type} problems",
                        "actions": [
                            f"Investigate system-wide {issue_type} issues",
                            f"Review {issue_type} configuration and limits",
                        ],
                    }
                )

        return recommendations

    # Utility Methods

    def _identify_performance_issues(
        self, total_pnl: float, avg_win_rate: float, avg_trades_per_day: float
    ) -> list[str]:
        """Identify performance-related issues."""
        issues = []

        if total_pnl < -500:
            issues.append(f"Significant losses: ${total_pnl:.2f}")

        if avg_win_rate < 0.3:
            issues.append(f"Low win rate: {avg_win_rate:.1%}")

        if avg_trades_per_day < 1:
            issues.append(f"Low trading activity: {avg_trades_per_day:.1f} trades/day")

        return issues

    def _identify_stability_issues(
        self, uptime_percentage: float, consecutive_failures: int
    ) -> list[str]:
        """Identify stability-related issues."""
        issues = []

        if uptime_percentage < 0.8:
            issues.append(f"Low uptime: {uptime_percentage:.1%}")

        if consecutive_failures > 3:
            issues.append(f"Multiple consecutive failures: {consecutive_failures}")

        return issues

    def _identify_resource_issues(self, avg_cpu_usage: float, avg_memory_usage: float) -> list[str]:
        """Identify resource-related issues."""
        issues = []

        if avg_cpu_usage > 80:
            issues.append(f"High CPU usage: {avg_cpu_usage:.1f}%")

        if avg_memory_usage > 800:
            issues.append(f"High memory usage: {avg_memory_usage:.1f} MB")

        return issues

    def _identify_error_issues(self, error_rate: float, total_errors: int) -> list[str]:
        """Identify error-related issues."""
        issues = []

        if error_rate > 0.1:
            issues.append(f"High error rate: {error_rate:.1%}")

        if total_errors > 50:
            issues.append(f"High error count: {total_errors}")

        return issues

    def _identify_connectivity_issues(
        self, heartbeat_issues: int, connectivity_issues: int
    ) -> list[str]:
        """Identify connectivity-related issues."""
        issues = []

        if heartbeat_issues > 2:
            issues.append(f"Heartbeat issues: {heartbeat_issues}")

        if connectivity_issues > 1:
            issues.append(f"Connectivity issues: {connectivity_issues}")

        return issues

    def _calculate_trend_slope(self, values: list[float]) -> float:
        """Calculate trend slope using simple linear regression."""
        n = len(values)
        if n < 2:
            return 0.0

        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values, strict=False))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        return numerator / denominator if denominator != 0 else 0.0

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    # Health Analysis Loop

    async def _health_analysis_loop(self) -> None:
        """Health analysis loop for continuous monitoring."""
        try:
            while self.is_running:
                try:
                    # Get all active bots from database
                    active_bots = await self._database_service.get_active_bots()

                    # Analyze health for each bot
                    for bot_record in active_bots:
                        bot_id = bot_record.get("bot_id")
                        if bot_id:
                            try:
                                await self._analyze_bot_health_impl(bot_id)
                            except Exception as e:
                                self._logger.warning(
                                    f"Health analysis failed for bot: {e}", bot_id=bot_id
                                )

                    await asyncio.sleep(self._health_analysis_interval)

                except Exception as e:
                    self._logger.error(f"Health analysis loop error: {e}")
                    await asyncio.sleep(30)

        except asyncio.CancelledError:
            self._logger.info("Health analysis loop cancelled")

    # Service-specific health check
    async def _service_health_check(self) -> Any:
        """Service-specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check if database service is healthy
            if not hasattr(self, "_database_service") or not self._database_service:
                return HealthStatus.UNHEALTHY

            db_health = await self._database_service.health_check()
            if db_health.get("status") != "healthy":
                return HealthStatus.DEGRADED

            # Check health analysis task
            if self._health_monitoring_task and self._health_monitoring_task.done():
                exception = self._health_monitoring_task.exception()
                if exception:
                    return HealthStatus.DEGRADED

            # Check if we have health history data
            if not self._bot_health_history:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(f"Service health check failed: {e}")
            return HealthStatus.UNHEALTHY
